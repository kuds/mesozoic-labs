"""Stage 1 episode metrics: the walk, turn and start-and-stop measures of ``STAGE1_PLAN.md`` section 4 on a simulated
episode, for the tuning objective and gate G5 (``tune``), and the noise floor that sets R2's rest speed.

``Recorder`` is handed every physics step and feedback update of an episode by ``control.run`` (it only reads the
plant): per physics step the two feet's vertical floor loads (the plant's contact sensors, net force in the world
frame from that foot's floor contacts) and every drive's squared motor-space force (``actuator_force``, as
``control.run`` sums it for the episode's RMS); per feedback update (100 Hz) the true state: time and reference
time, the CoM and its velocity, the feet, each foot's sole centre (the mean of its pads' bottom corners, ``estimator.
sole_corners``) and load, the root (the trunk's origin), the lowest point of each foot (every foot geom: pads, toes,
claws, hallux, plate, as ``carpet.swing_profiles`` takes it), the trunk's roll and pitch about its home attitude and
its yaw, the loaded feet's slip speeds (``control.foot_slip_speed``), the IMU point's horizontal acceleration (the
accelerometer without noise),
the joint angles, the true DCM error against the reference (relative to the anchor foot in its frame, as the
feedback's) and, when asked, the pose every ``legs_every`` updates for the leg-to-leg check (``legs.LegGap``).  Every
time is that of the state the derived quantities describe: the start of the physics step just taken.

Measures (``window``, ``start_reading``, ``acceleration_peak``, ``joint_margins``, ``leg_gap``), with section 4's
readings:

* contact: a foot carrying more than ``control.CONTACT_LOAD_FRAC`` (2 %) of the weight, per physics step, runs
  shorter than ``DEBOUNCE_S`` (20 ms) merged into their neighbours (a stance foot's load drops to zero for one or two
  physics steps now and then, contact chatter); flight: a physics step with neither foot in contact (the steps with
  neither foot over the threshold before the debounce are reported as ``unloaded_steps``);
* duty: each foot's share of the window's steps in contact; cadence: 2 over the mean interval between a foot's
  successive touchdowns in the window; stride: the mean distance between a foot's successive touchdown positions,
  over H; GRF: the largest total vertical floor load over the weight;
* drive RMS: each drive's motor-space RMS force over the window's physics steps over its sustained torque (the window
  form of ``control.run``'s ``worst_sustained_ratio``), the worst drive;
* roll and pitch about home: RMS and peak over the window's updates (``control._roll_pitch`` of R R_home^T);
* toe clearance: per swing of the reference (its swing foot and swing progress at the update's reference time), the
  lowest point of that foot over the swing's middle half (``clearance.MID_SWING``, as the screens measured it on the
  gait's own clock), within the window; the worst swing;
* slip ("foot site vs contact point", BEHAVIOR_SPEC section 1): per stance (a contact run) within the window, how far
  the foot's sole centre moves horizontally while the foot carries more than ``control.SLIP_LOAD_FRAC`` of the weight,
  from where it was when that load came on (the largest distance); the median and the worst stance.  The sole centre
  lies on the pads' bottom, so a foot rocking on a pad edge barely moves it sideways, and a foot spinning on its pads
  moves it as the pads slide.  Reported beside it: the loaded contacts' slip speed (``control.foot_slip_speed``, the
  fastest loaded contact) integrated over the stance, a path length that also gathers the contacts' jitter;
* speed: the CoM's displacement along the episode's starting heading over the window, over its duration; yaw rate:
  pi over the time the trunk takes to turn through 180 deg from the window's start (the window's mean rate when it
  does not get there);
* start: the forward CoM speed averaged over the last commanded stride first reaching ``START_SPEED_FRAC`` (90 %) of
  the command, in strides from the start command (the start of the start's speed ramp, as decision 18 times the
  stop); the stop is decision 18's reading (``control._stop_result``);
* acceleration (decision 10): the IMU point's acceleration along the trunk heading, low-passed at ``ACCEL_LP_HZ``
  (1 Hz; a second-order Butterworth run forward and backward), its largest magnitude in a window;
* joint range: each joint's smallest distance to its plant range over the episode;
* leg interference: ``legs.LegGap`` on the recorded poses (50 Hz), interference-free and the smallest gap.

``standing_noise_floor`` measures what R2's sensor noise does to a standing body: the true CoM's horizontal speed
while R2 stands still from the settled stance (``transitions.REST_R2_RULE``).
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .. import carpet, clearance, statics
from . import control, estimator

DEBOUNCE_S = 0.02
START_SPEED_FRAC = 0.9
ACCEL_LP_HZ = 1.0
ACCEL_LP_ORDER = 2
#: Columns of a recorder's per-update rows.
COLS = ("t", "t_ref", "com_x", "com_y", "v_x", "v_y", "foot0_x", "foot0_y", "foot1_x", "foot1_y", "low0", "low1",
        "roll", "pitch", "yaw", "slip0", "slip1", "acc_x", "acc_y", "e_x", "e_y", "ref_swing", "ref_u", "sole0_x",
        "sole0_y", "sole1_x", "sole1_y", "load0", "load1", "root_x", "root_y")
C = {name: i for i, name in enumerate(COLS)}


def _lowest_point_tables(m, feet) -> tuple:
    """Per foot body: its geoms with extent (``carpet.foot_geoms``), split into boxes and capsules (every foot geom of
    these bodies is one or the other), for the vectorised lowest point."""
    fg = carpet.foot_geoms(m)
    boxes, caps = [], []
    for side, f in zip(statics.SIDES, feet, strict=True):
        gs = [g for g in fg[side] if m.geom_bodyid[g] == f]
        other = [g for g in gs if m.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE)]
        if other:
            raise ValueError(f"{side} foot geoms {other} are neither boxes nor capsules")
        boxes.append(np.array([g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX], int))
        caps.append(np.array([g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE], int))
    return boxes, caps


def foot_lowest(m, d, boxes, caps) -> np.ndarray:
    """Height of each foot's lowest point (world z): boxes c_z - sum_j |R_2j| h_j, capsules c_z - |R_22| h - r (exact,
    as ``carpet.lowest_point``)."""
    out = np.empty(2)
    for i in (0, 1):
        b, c = boxes[i], caps[i]
        zb = d.geom_xpos[b, 2] - np.einsum("ij,ij->i", np.abs(d.geom_xmat[b, 6:9]), m.geom_size[b, :3])
        zc = d.geom_xpos[c, 2] - np.abs(d.geom_xmat[c, 8]) * m.geom_size[c, 1] - m.geom_size[c, 0]
        out[i] = min(float(zb.min()) if len(b) else math.inf, float(zc.min()) if len(c) else math.inf)
    return out


class Recorder:
    """An episode's traces for the measures (module docstring); hand it to ``control.run(..., recorder=)``."""

    def __init__(self, legs_every: int = 0):
        self.legs_every = int(legs_every)

    # ------------------------------------------------------------------ control.run's hooks
    def begin(self, tk: control.Tracker, n: int) -> None:
        p = tk.p
        m = p.m
        self.tk, self.n = tk, n
        self.dt = float(m.opt.timestep)
        self.weight = float(m.body_subtreemass[p.trunk]) * control.G
        self.loads = np.zeros((n, 2))
        self.force_sq = np.zeros((n, m.nu))
        self.sustained = np.array([dr.sustained for dr in p.drives], float)
        self.actuators = [dr.actuator for dr in p.drives]
        rows = n // control.FEEDBACK_EVERY + 1
        self.U = np.full((rows, len(COLS)), np.nan)
        self.Q = np.zeros((rows, len(p.joints)))
        self.poses = [] if self.legs_every else None
        self._lidx = np.array([p._sens[f"{s}_foot_load"][0] + 3 for s in statics.SIDES])
        self._acc = p._sens["imu_accel"][0]
        self._boxes, self._caps = _lowest_point_tables(m, p.feet)
        self._sole = [c.mean(0) for c in estimator.sole_corners(m, p.feet)]  # each foot's sole centre, foot frame
        self._v = p._v  # the plant's views (plant.MjViews): the same arrays as p.d's attributes
        self.rows = 0
        self.steps = 0

    def step(self, k: int) -> None:
        v = self._v
        self.loads[k] = v.sensordata[self._lidx]
        np.square(v.actuator_force, out=self.force_sq[k])

    def update(self, k: int, loads, slip_v, Rr, yaw: float) -> None:
        tk = self.tk
        p = tk.p
        m, d, vw = p.m, p.d, self._v
        xpos, xmat = vw.xpos, vw.xmat
        mujoco.mj_subtreeVel(m, d)
        com, v = vw.subtree_com[p.trunk], vw.subtree_linvel[p.trunk]
        t_ref = tk.tau_ref - self.dt * tk.rate * (1.0 + tk.delta)  # the reference time of the step's start state
        f0, f1 = xpos[p.feet[0]], xpos[p.feet[1]]
        low = foot_lowest(m, d, self._boxes, self._caps) - p.z_floor
        roll, pitch = control._roll_pitch(Rr)
        R = xmat[p.trunk].reshape(3, 3)
        acc = R @ vw.sensordata[self._acc : self._acc + 3]
        _, ref = tk.track.info(t_ref)
        a = int(ref["anchor"])
        if a < 2:
            pa, ya = xpos[p.feet[a]][:2], control._yaw(xmat[p.feet[a]])
        else:
            fy = [control._yaw(xmat[f]) for f in p.feet]
            pa = 0.5 * (f0[:2] + f1[:2])
            ya = math.atan2(math.sin(fy[0]) + math.sin(fy[1]), math.cos(fy[0]) + math.cos(fy[1]))
        xi = com[:2] + v[:2] / tk.track.omega
        e = control._rot2(ya).T @ (xi - pa) - tk.dcm_reference(ref)
        s0 = f0 + xmat[p.feet[0]].reshape(3, 3) @ self._sole[0]
        s1 = f1 + xmat[p.feet[1]].reshape(3, 3) @ self._sole[1]
        root = xpos[p.trunk]
        i = self.rows
        self.U[i] = (d.time - self.dt, t_ref, com[0], com[1], v[0], v[1], f0[0], f0[1], f1[0], f1[1], low[0], low[1],
                     roll, pitch, yaw, slip_v[0], slip_v[1], acc[0], acc[1], e[0], e[1], ref["swing"], ref["u"], s0[0],
                     s0[1], s1[0], s1[1], loads[0], loads[1], root[0], root[1])
        self.Q[i] = vw.qpos[p.qadr]
        if self.poses is not None and i % self.legs_every == 0:
            self.poses.append(vw.qpos.copy())
        self.rows = i + 1

    def end(self, steps: int, out: dict) -> None:
        self.steps = int(steps)
        self.loads = self.loads[:steps]
        self.force_sq = self.force_sq[:steps]
        self.U = self.U[: self.rows]
        self.Q = self.Q[: self.rows]
        self.out = out

    # ------------------------------------------------------------------ derived series
    def col(self, name: str) -> np.ndarray:
        return self.U[:, C[name]]

    @property
    def t(self) -> np.ndarray:
        return self.col("t")

    def sim_time(self, t_ref) -> float | None:
        """The simulated time at which the reference clock reached ``t_ref`` (None: not reached)."""
        tr = self.col("t_ref")
        if not len(tr) or t_ref > tr[-1] + 1e-12 or t_ref < tr[0] - 1e-12:
            return None
        return float(np.interp(t_ref, tr, self.t))

    def contact(self) -> np.ndarray:
        """Per physics step and foot: in contact (debounced, module docstring)."""
        raw = self.loads > control.CONTACT_LOAD_FRAC * self.weight
        k = max(1, int(round(DEBOUNCE_S / self.dt)))
        return np.column_stack([debounce(raw[:, i], k) for i in (0, 1)])

    def yaw_unwrapped(self) -> np.ndarray:
        return np.unwrap(self.col("yaw"))


def runs(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs of equal values: starts, ends (exclusive) and values."""
    flag = np.asarray(flag, bool)
    if not len(flag):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    change = np.flatnonzero(flag[1:] != flag[:-1]) + 1
    s = np.r_[0, change]
    e = np.r_[change, len(flag)]
    return s, e, flag[s]


def debounce(flag: np.ndarray, min_len: int) -> np.ndarray:
    """``flag`` with every inner run shorter than ``min_len`` merged into its neighbours (shortest first)."""
    f = np.asarray(flag, bool).copy()
    while True:
        s, e, v = runs(f)
        inner = np.flatnonzero((e - s < min_len) & (s > 0) & (e < len(f)))
        if not len(inner):
            return f
        i = inner[np.argmin((e - s)[inner])]
        f[s[i] : e[i]] = not v[i]


# ------------------------------------------------------------------------------------------------
# Measures
# ------------------------------------------------------------------------------------------------
def window(rec: Recorder, ta: float, tb: float, heading: float, H: float, yaw_target_sign: float = 0.0) -> dict:
    """Section 4's walk measures over the simulated times [ta, tb] (module docstring); ``heading`` the walk's
    forward direction (rad), ``yaw_target_sign`` the turn's direction (+1 left, -1 right, 0 no turn)."""
    t, dt, U = rec.t, rec.dt, rec.U
    sel = (t >= ta - 1e-9) & (t <= tb + 1e-9)
    k0, k1 = max(0, int(math.ceil(ta / dt - 1e-9))), min(rec.steps, int(math.floor(tb / dt + 1e-9)) + 1)
    out = dict(t_start_s=ta, t_end_s=tb, seconds=tb - ta, updates=int(sel.sum()), steps=max(0, k1 - k0))
    if out["updates"] < 2 or out["steps"] < 2:
        return dict(out, complete=False)
    cont = rec.contact()
    raw = rec.loads > control.CONTACT_LOAD_FRAC * rec.weight
    cw = cont[k0:k1]
    out["duty"] = [float(np.mean(cw[:, i])) for i in (0, 1)]
    flight = ~cw[:, 0] & ~cw[:, 1]
    s, e, v = runs(flight)
    out["flight_s"] = float(flight.sum() * dt)
    out["longest_flight_s"] = float(max([(b - a) * dt for a, b, x in zip(s, e, v, strict=True) if x], default=0.0))
    out["unloaded_steps"] = int(np.sum(~raw[k0:k1, 0] & ~raw[k0:k1, 1]))  # before the debounce (contact chatter)
    out["grf_peak_bw"] = float(np.max(rec.loads[k0:k1].sum(1)) / rec.weight)
    ratio = np.sqrt(np.mean(rec.force_sq[k0:k1], axis=0)) / rec.sustained  # each drive's RMS over the window's steps
    j = int(np.argmax(ratio))
    out["drive_sustained_worst"] = [rec.actuators[j], float(ratio[j])]
    rp = U[sel][:, [C["roll"], C["pitch"]]]
    out["roll_rms_deg"], out["pitch_rms_deg"] = (float(np.degrees(np.sqrt(np.mean(rp[:, j] ** 2)))) for j in (0, 1))
    out["roll_peak_deg"], out["pitch_peak_deg"] = (float(np.degrees(np.max(np.abs(rp[:, j])))) for j in (0, 1))
    # contact events
    tds, stances = {0: [], 1: []}, []
    for i in (0, 1):
        s, e, v = runs(cont[:, i])
        for a, b, x in zip(s, e, v, strict=True):
            if x and a > 0:
                tds[i].append(a)
                if b < len(cont):
                    stances.append((i, a, b))
    fpos = {0: U[:, [C["foot0_x"], C["foot0_y"]]], 1: U[:, [C["foot1_x"], C["foot1_y"]]]}

    def at_step(k):  # the first update at or after step k
        return min(int(np.searchsorted(t, k * dt - 1e-9)), len(t) - 1)

    intervals, strides = [], []
    for i in (0, 1):
        ks = [k for k in tds[i] if k0 <= k < k1]
        intervals += [(b - a) * dt for a, b in zip(ks[:-1], ks[1:], strict=True)]
        strides += [float(np.linalg.norm(fpos[i][at_step(b)] - fpos[i][at_step(a)])) for a, b in zip(ks[:-1], ks[1:],
                                                                                                     strict=True)]
    out["touchdowns"] = [int(sum(k0 <= k < k1 for k in tds[i])) for i in (0, 1)]
    out["cadence_steps_s"] = 2.0 / float(np.mean(intervals)) if intervals else None
    out["stride_H"] = float(np.mean(strides)) / H if strides else None
    # toe clearance: each reference swing's middle half in the window (the screens' reading, on the gait's clock)
    lows, mins = U[:, [C["low0"], C["low1"]]], []
    sw, u = U[:, C["ref_swing"]], U[:, C["ref_u"]]
    mid = sel & (u >= clearance.MID_SWING[0]) & (u <= clearance.MID_SWING[1])
    for i in (0, 1):
        s, e, v = runs(mid & (sw == i))
        mins += [float(np.min(lows[a:b, i])) for a, b, x in zip(s, e, v, strict=True) if x]
    out["swings"] = len(mins)
    out["toe_clearance_m"] = float(min(mins)) if mins else None
    # slip: stances within the window
    slips, paths = [], []
    heavy = control.SLIP_LOAD_FRAC * rec.weight
    for i, a, b in stances:
        if a * dt < ta - 1e-9 or b * dt > tb + 1e-9:
            continue
        m = (t >= a * dt - 1e-9) & (t < b * dt - 1e-9)
        paths.append(float(np.sum(U[m, C[f"slip{i}"]]) * control.FEEDBACK_EVERY * dt))
        sole = U[m][:, [C[f"sole{i}_x"], C[f"sole{i}_y"]]][U[m, C[f"load{i}"]] > heavy]
        slips.append(float(np.max(np.linalg.norm(sole - sole[0], axis=1))) if len(sole) else 0.0)
    out["stances"] = len(slips)
    out["slip_median_m"] = float(np.median(slips)) if slips else None
    out["slip_worst_m"] = float(max(slips)) if slips else None
    out["slip_contact_path_median_m"] = float(np.median(paths)) if paths else None
    out["slip_contact_path_worst_m"] = float(max(paths)) if paths else None
    # speed and heading
    cx, cy = np.interp([ta, tb], t, rec.col("com_x")), np.interp([ta, tb], t, rec.col("com_y"))
    dxy = np.array([cx[1] - cx[0], cy[1] - cy[0]])
    fwd = np.array([math.cos(heading), math.sin(heading)])
    out["speed_m_s"] = float(dxy @ fwd / (tb - ta))
    out["lateral_m"] = float(dxy @ np.array([-fwd[1], fwd[0]]))
    yaw = rec.yaw_unwrapped()
    ys = np.interp([ta, tb], t, yaw)
    out["heading_change_rad"] = float(ys[1] - ys[0])
    if yaw_target_sign:
        tt = t[sel]
        turned = yaw_target_sign * (yaw[sel] - ys[0])
        k = np.flatnonzero(turned >= math.pi)
        if len(k):
            j = int(k[0])
            t_pi = float(np.interp(math.pi, turned[j - 1 : j + 1], tt[j - 1 : j + 1])) if j > 0 else float(tt[0])
            out["yaw_rate_rad_s"], out["turned_180"] = math.pi / max(t_pi - ta, 1e-9), True
        else:
            out["yaw_rate_rad_s"], out["turned_180"] = float(turned[-1]) / (tb - ta), False
    e = U[sel][:, [C["e_x"], C["e_y"]]]
    out["dcm_error_rms_m"] = float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
    out["complete"] = True
    return out


def start_reading(rec: Recorder, t_cmd: float, speed_cmd: float, stride_s: float, heading: float) -> dict:
    """Strides from the start command until the forward CoM speed averaged over the last commanded stride first
    reaches ``START_SPEED_FRAC`` of the command (module docstring)."""
    t = rec.t
    out = dict(command_s=t_cmd, command_speed_m_s=speed_cmd, stride_s=stride_s, speed_frac=START_SPEED_FRAC)
    if not len(t) or t[-1] < t_cmd:
        return dict(out, reached=False, strides=None)
    fwd = np.array([math.cos(heading), math.sin(heading)])
    x = rec.col("com_x") * fwd[0] + rec.col("com_y") * fwd[1]
    sel = t >= t_cmd
    tt = t[sel]
    back = np.interp(tt - stride_s, t, x)
    v = (x[sel] - back) / stride_s
    k = np.flatnonzero(v >= START_SPEED_FRAC * speed_cmd)
    if not len(k):
        return dict(out, reached=False, strides=None)
    return dict(out, reached=True, strides=float((tt[k[0]] - t_cmd) / stride_s))


def lowpass(x: np.ndarray, fs: float, hz: float = ACCEL_LP_HZ, order: int = ACCEL_LP_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass (forward and backward); short series are returned as their mean."""
    from scipy.signal import butter, filtfilt

    b, a = butter(order, hz, fs=fs)
    if len(x) <= 3 * max(len(a), len(b)):
        return np.full(len(x), float(np.mean(x)) if len(x) else 0.0)
    return filtfilt(b, a, x)


def forward_acceleration(rec: Recorder) -> np.ndarray:
    """The IMU point's acceleration along the trunk heading, low-passed (decision 10)."""
    yaw = rec.col("yaw")
    a = rec.col("acc_x") * np.cos(yaw) + rec.col("acc_y") * np.sin(yaw)
    return lowpass(a, 1.0 / (control.FEEDBACK_EVERY * rec.dt))


def acceleration_peak(rec: Recorder, ta: float, tb: float, a_lp: np.ndarray | None = None) -> float | None:
    a_lp = forward_acceleration(rec) if a_lp is None else a_lp
    sel = (rec.t >= ta) & (rec.t <= tb)
    return float(np.max(np.abs(a_lp[sel]))) if sel.any() else None


def joint_margins(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Each joint's smallest distance to its range over the rows of ``q`` (negative: beyond it)."""
    q = np.atleast_2d(q)
    return np.minimum(q.min(0) - lo, hi - q.max(0))


def leg_gap(rec: Recorder, lg) -> dict:
    """``legs.LegGap`` over the recorded poses."""
    if not rec.poses:
        return dict(poses=0, interference_free=None, min_gap_all_mm=None)
    r = lg.trace(rec.poses)
    return {k: r[k] for k in ("poses", "interference_free", "interference_poses", "tight_poses", "min_gap_all_mm",
                              "min_gap_all_pair", "min_gap_moving_mm", "min_gap_moving_pair")}


# ------------------------------------------------------------------------------------------------
# R2's rest speed: the standing noise floor
# ------------------------------------------------------------------------------------------------
NOISE_FLOOR_SEEDS = tuple(range(20))  # sensor-noise seeds (G4's, apart from the tuning, validation and panel seeds)
NOISE_FLOOR_SECONDS = 12.0
NOISE_FLOOR_DISCARD_S = 2.0


class _Speed:
    """A recorder of the true CoM's horizontal speed at the feedback updates."""

    def begin(self, tk, n):
        self.tk, self.rows = tk, []

    def step(self, k):
        pass

    def update(self, k, loads, slip_v, Rr, yaw):
        p = self.tk.p
        mujoco.mj_subtreeVel(p.m, p.d)
        v = p.d.subtree_linvel[p.trunk]
        self.rows.append((float(p.d.time), float(math.hypot(v[0], v[1]))))

    def end(self, steps, out):
        pass


def standing_noise_floor(body: str, run_dir: Path, seeds=NOISE_FLOOR_SEEDS, seconds: float = NOISE_FLOOR_SECONDS,
                         discard_s: float = NOISE_FLOOR_DISCARD_S) -> dict:
    """R1 (one run) and R2 (one run per sensor-noise seed) standing still at the tuning start's gains from the settled
    stance, holding the start transition's first frame for ``seconds``: the true CoM's horizontal speed at the
    feedback updates after the first ``discard_s`` (mm/s: median, 90th, 99th and 99.9th percentiles, largest)."""
    from . import plant
    from .refs import Reference

    run_dir = Path(run_dir)
    p = plant.build(body, "hard")
    om = control.omega0(p, Reference.load(run_dir / "refs" / "nominal.npz").q_home)
    tr = control.build_track(p, run_dir, [("hold", seconds + 1.0), ("transition", "start")], om)
    out = dict(body=body, seconds=seconds, discard_s=discard_s, gains="start", floor="hard")
    for mode, ss in (("R1", (0,)), ("R2", tuple(seeds))):
        tk = control.Tracker(p, tr, mode)
        allv, per = [], []
        for s in ss:
            rec = _Speed()
            r = control.run(tk, seconds, start="stance", seed=s, recorder=rec)
            a = np.array(rec.rows)
            v = a[a[:, 0] >= discard_s, 1]
            allv.append(v)
            per.append(dict(seed=int(s), fell=bool(r["fell"]), max_mm_s=1e3 * float(v.max()),
                            p99_mm_s=1e3 * float(np.percentile(v, 99))))
        v = np.concatenate(allv)
        out[mode] = dict(seeds=list(ss), samples=int(len(v)), median_mm_s=1e3 * float(np.median(v)),
                         p90_mm_s=1e3 * float(np.percentile(v, 90)), p99_mm_s=1e3 * float(np.percentile(v, 99)),
                         p999_mm_s=1e3 * float(np.percentile(v, 99.9)), max_mm_s=1e3 * float(v.max()), per_seed=per,
                         noise=tk.noise.describe() if mode == "R2" else None)
    return out
