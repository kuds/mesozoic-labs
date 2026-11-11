"""Stage 2 Track A: the look ladder's machine metrics, path plots and phase charts (``STAGE2_PLAN.md`` section 4 A.4
and A.5; ``stage2/research/research_look.md`` 4.1 items 3 and 5).

Every reading is taken on a rung's arrays (each file checked against the sha256 ``look_refs.json`` records): the
kinematic reference with perfect tracking, not a simulation.  Each is defined on the frames (positions, angles and
the floor contact detected from the feet's lowest points), not on the spec, so that the same code can read the
closed-loop traces later (research 4.2 item 2).  Nothing is adopted (P24).

Metrics per rung (``metrics``; research 4.1 item 5)
----------------------------------------------------
* **Hip bob**: the hip midpoint's height (the hip-roll axes' midpoint, ``hip_mid``), peak to peak over the window and
  per step (mean), mm and H; the trough's place in stance (``look.measure``); the crest hold, the share of the window
  with the hips within 5 % of the bob's range of its crest (a sinusoidal bob's: 0.144), as a long hold runs the hips
  level and straight (research 4.1 item 3's straight segments).
* **Trunk roll and pitch**: the root's attitude relative to home as roll, pitch and yaw (intrinsic z-y-x), RMS about
  home and peak to peak, deg.
* **Sway**: the model CoM's lateral peak to peak, the largest per stride and over the window, H (``screen.sway``;
  BEHAVIOR_SPEC's 0.12 H).
* **Head**: its pitch in the world (``head_pitch_world``), RMS about its mean, deg (research 4.2's band: 1.5 deg); the
  head's and the trunk's angular speeds in the world (RMS of |omega| from the bodies' orientations, deg/s) and the
  stabilisation ratio, head over trunk (None where the trunk does not rotate: V0-V3 keep it level and rigid, so there
  the head's own rate is the reading); the same with the pitch components alone (None while the trunk never pitches,
  which F8 keeps so on every rung).
* **Tail**: the tail yaw's fitted amplitude and phase against the stride clock (tail = A sin(2 pi phase + psi), phase 0
  at the left foot's lift-off), its offset from today's pi/2 in cycles (research 4.2's band: 0.1 cycle), and the share
  of swings in which the tip moves toward the stance side (F9's sign; ``look.measure``).
* **Toe-off**: the foot's heel-up pitch at each lift-off (mean, range) and its peak, rad and deg (``look.measure``).
* **Swing time, duty, stride and cadence** from the detected floor contact, beside the bird curves (``bird_curves``):
  the tinamou's duty band 0.55-0.70 and relative stride 2.14 Fr^0.26 H, a swing time near 1.6 sqrt(H/g) at every speed
  (research F3), and at the animal pace the dynamically similar T. rex's cadence (0.66 steps/s at a 3.1 m hip scaled
  by sqrt(3.1 / H); research Bottom line 5).
* **Step-time variation**: the intervals between successive detected touchdowns (left and right alternate): CV, the
  largest deviation from their mean, and left-to-right over right-to-left; a CV of 0 is research 4.1's mechanical
  tell "identical steps".
* **Smoothness (SPARC**, Balasubramanian et al. 2015): the feet's, the published spectral arc length (``sparc``: the
  speed profile's zero-padded magnitude spectrum, normalised, up to the last frequency under 10 Hz where it is at
  least 0.05) of the toe tip's speed over each detected swing, a discrete movement from rest to rest; mean and lowest
  per foot.  The head's (``sparc_periodic``): the head moves on all through the walk, so the arc length is taken on
  its periodic window, the snout's velocity relative to the walk's mean forward motion (what the following camera
  shows) tiled to a whole number of windows at least 20 s long, the vector's magnitude spectrum with no padding (the
  window is exactly periodic, so every bin is a harmonic of it), normalised by its peak, from 0 Hz to the same
  cutoff.  A sinusoid reads -1.95, a sinusoid with a 30 % fifth harmonic -3.27, a square wave -4.44; less negative is
  smoother.  F14's slow drifts add spectral lines below the stride frequency, which this reading counts as
  complexity, so a V6 head reads less smooth than its V5 partly for its variation.
* **Stance-pad yaw moment** and the Stage 0 verdict, copied from ``screen.json`` (``stage0_screen``).
* **Mechanical tells** (research 4.1 item 5): a constant hip height (bob under 1 mm), flat feet at lift-off (toe-off
  under 0.01 rad), identical steps (step-time CV under 1e-4), a metronome nod (the neck's two-per-stride component
  over 0.01 rad), and synchronous extremes: the pairs of parts (``PARTS``) whose maxima fall within 0.02 of a cycle of
  each other on at least half of the cycles (``extremes``).

Plots (``path_plot``, ``phase_chart``; research 4.1 item 3), PNG, in the media directory
--------------------------------------------------------------------------------------
* Paths of the snout, the left hip (the hip-pitch axis), the left toe tip (the middle claw's tip) and the tail tip in
  the frame moving with the walk's mean forward speed (what the following camera sees), each relative to its own
  mean, side (forward and up) and top (forward and left), mm; V0 in grey and the compared rung in blue; each panel
  scaled to its own range (the shapes, not the sizes, compare: arcs, straight segments and corners survive scaling).
* A phase chart per rung: for each part (``PARTS``) its normalised signal over the stride (every stride of the window
  drawn) and its maxima (filled) and minima (open) at their stride phase (0: the left foot's lift-off); the feet's
  swings shaded.  Staggered extremes read as overlap, all at once as a machine (research 4.1 item 3).
* Per body: V0 against V6 at each speed (paths and phase charts), each of V1-V5 against V0 at 0.40 m/s (paths), and
  the ladder V0-V6 at 0.40 m/s (one phase chart).

Run from hardware/trex_form_study:
    python -m designlab.stage2.motion run --media <dir>      # metrics.json (stage2/look/) and the plots (<dir>/<body>/)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from .. import gait
from ..stage1 import plant
from . import look, screen

OUT = look.OUT
G = gait.G
METRICS_JSON = "metrics.json"
#: SPARC's parameters (Balasubramanian et al. 2015's defaults): cutoff (Hz), amplitude threshold, zero-padding levels;
#: the periodic reading's shortest tiled span (s).
SPARC_FC, SPARC_AMP, SPARC_PAD, SPARC_SPAN_S = 10.0, 0.05, 4, 20.0
#: Below these the trunk "does not rotate" (deg/s RMS) and a signal is "flat" (its range, in its unit).
STILL_DEG_S = 0.05
FLAT = 1e-6
#: Research 4.2's look-fidelity bands (for the closed loop; reported beside the reference's readings).
BANDS = dict(bob_rel=0.30, head_world_pitch_rms_deg=1.5, tail_phase_cycles=0.1, toe_off_rel=0.70)
#: The bird and T. rex curves (``bird_curves``; research section 2 and Bottom line 5).
TINAMOU_DUTY = (0.55, 0.70, 0.64)
TINAMOU_STRIDE = (2.14, 0.26)
SWING_TIME_K = 1.6
TREX = dict(hip_m=3.1, steps_per_s=0.66, froude=0.054, stride_rel=1.25)
#: Parts of the phase chart: key, label, the extremes marked, cycles per stride (for the reader; the extremes are
#: found on the signal itself).
PARTS = (("hip_height", "hip height (bob)", ("max", "min")),
         ("trunk_roll", "trunk roll (left side up +)", ("max", "min")),
         ("head_pitch", "head pitch in the world (up +)", ("max", "min")),
         ("snout_height", "snout height", ("max", "min")),
         ("tail_yaw", "tail yaw (tip to the right +)", ("max", "min")),
         ("sway", "CoM lateral (left +)", ("max", "min")),
         ("left_foot_height", "left foot height", ("max",)),
         ("left_foot_pitch", "left foot pitch (heel up +)", ("max", "min")))
#: The bob's crest hold: the share of the window with the hips within ``CREST_BAND`` of the bob's range of its crest
#: (a sinusoidal bob: arccos(1 - 2 CREST_BAND) / pi = 0.144); a long hold reads as a level, straight run of the hips.
CREST_BAND = 0.05
CREST_SINUSOID = math.acos(1 - 2 * CREST_BAND) / math.pi
#: A peak's prominence relative to its signal's range for it to count as an extreme.
PROMINENCE = 0.25
SYNC_CYCLES = 0.02
PATH_MARKERS = (("snout", "snout"), ("hip_l", "left hip"), ("toe_l", "left toe tip"), ("tail_tip", "tail tip"))
#: The narrowest axis of a path panel (mm): a coordinate that hardly moves is not magnified into noise.
MIN_SPAN_MM = 10.0
#: Plot styling (the dataviz reference palette: slot 1 blue for the rung, de-emphasis grey for V0, light surface).
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, GREY = "#2a78d6", "#898781"


# ------------------------------------------------------------------------------------------------
# Readings
# ------------------------------------------------------------------------------------------------
def sparc(speed, fs: float, padlevel: int = SPARC_PAD, fc: float = SPARC_FC, amp_th: float = SPARC_AMP) -> float:
    """The spectral arc length of a speed profile (Balasubramanian et al. 2015's published algorithm): the profile's
    magnitude spectrum, zero-padded to 2^(ceil(log2 n) + padlevel) points and normalised by its maximum, kept from its
    first to its last point at or above ``amp_th`` below ``fc``; minus the arc length of that curve with frequency
    normalised by the band."""
    speed = np.asarray(speed, float)
    nfft = int(2 ** (math.ceil(math.log2(len(speed))) + padlevel))
    f = np.arange(nfft) * fs / nfft
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()
    keep = f <= fc
    f, mag = f[keep], mag[keep]
    idx = np.flatnonzero(mag >= amp_th)
    f, mag = f[idx[0] : idx[-1] + 1], mag[idx[0] : idx[-1] + 1]
    if len(f) < 2:
        return 0.0
    return -float(np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(mag) ** 2)))


def sparc_periodic(vel, dt: float, fc: float = SPARC_FC, amp_th: float = SPARC_AMP, span_s: float = SPARC_SPAN_S
                   ) -> dict:
    """The spectral arc length of a periodic vector signal (module docstring): ``vel`` (n, d) over one exact period,
    its mean removed, tiled to a whole number of periods at least ``span_s`` (and at least 2), the magnitude of its
    vector spectrum (no padding) normalised by its peak, from 0 Hz to the last bin at or below ``fc`` at or above
    ``amp_th``; minus that curve's arc length with frequency normalised by the band."""
    v = np.asarray(vel, float)
    v = v.reshape(len(v), -1)
    n = len(v)
    reps = max(2, int(math.ceil(span_s / (n * dt) - 1e-9)))
    x = np.tile(v - v.mean(axis=0), (reps, 1))
    spec = np.sqrt((np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(axis=1))
    f = np.fft.rfftfreq(len(x), dt)
    if spec.max() <= 0:
        return dict(sparc=0.0, cutoff_hz=0.0, repeats=reps)
    mag = spec / spec.max()
    keep = f <= fc
    f, mag = f[keep], mag[keep]
    last = int(np.flatnonzero(mag >= amp_th)[-1])
    f, mag = f[: last + 1], mag[: last + 1]
    if len(f) < 2:
        return dict(sparc=0.0, cutoff_hz=float(f[-1]), repeats=reps)
    arc = float(np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(mag) ** 2)))
    return dict(sparc=-arc, cutoff_hz=float(f[-1]), repeats=reps)


def periodic_velocity(x, dt: float, advance: float = 0.0) -> np.ndarray:
    """Central differences over a periodic window whose next copy is moved by ``advance`` along x (axis 0 of the last
    dimension); x (n, ..., 3) or (n,) with ``advance`` 0."""
    x = np.asarray(x, float)
    prev, nxt = np.roll(x, 1, axis=0), np.roll(x, -1, axis=0)
    if advance:
        prev[0, ..., 0] -= advance
        nxt[-1, ..., 0] += advance
    return (nxt - prev) / (2 * dt)


def _qmul(a, b):
    return np.array(gait._quat_mul(a, b))


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def angular_velocity(quats, dt: float) -> np.ndarray:
    """World-frame angular velocity (rad/s) of a periodic sequence of attitudes (w, x, y, z), by central differences:
    the rotation from frame k-1 to k+1, q_{k+1} conj(q_{k-1}), as an axis-angle over 2 dt."""
    qs = np.asarray(quats, float)
    n = len(qs)
    out = np.zeros((n, 3))
    for k in range(n):
        d = _qmul(qs[(k + 1) % n], _qconj(qs[(k - 1) % n]))
        if d[0] < 0:
            d = -d
        s = np.linalg.norm(d[1:])
        ang = 2 * math.atan2(s, d[0])
        out[k] = (d[1:] / s * ang if s > 1e-15 else np.zeros(3)) / (2 * dt)
    return out


def relative_euler(quats, q_ref) -> np.ndarray:
    """Roll, pitch and yaw (rad; intrinsic z-y-x) of each attitude relative to ``q_ref`` (conj(q_ref) q)."""
    qc = _qconj(np.asarray(q_ref, float))
    out = []
    for q in np.asarray(quats, float):
        w, x, y, z = _qmul(qc, q)
        out.append((math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                    math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
                    math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    return np.array(out)


_MODELS: dict = {}


def model(body: str) -> mujoco.MjModel:
    """The body's compiled MJCF (``plant.model_files``, checked by ``plant.verify``), loaded once."""
    if body not in _MODELS:
        xml, _ = plant.model_files(body)
        plant.verify(body, xml)
        _MODELS[body] = mujoco.MjModel.from_xml_path(str(xml))
    return _MODELS[body]


def body_quats(m: mujoco.MjModel, q, names=("trunk", "head")) -> dict:
    """Each named body's world attitude on every pose (forward kinematics)."""
    d = mujoco.MjData(m)
    ids = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in names}
    out = {nm: np.zeros((len(q), 4)) for nm in names}
    for k, qk in enumerate(np.asarray(q, float)):
        d.qpos[:] = qk
        mujoco.mj_kinematics(m, d)
        for nm, b in ids.items():
            out[nm][k] = d.xquat[b]
    return out


def step_times(arrs: dict) -> dict:
    """The intervals between successive detected touchdowns (``look.contact_events``; the window wraps): CV, the
    largest deviation from their mean (fractions), left-to-right over right-to-left, and whether they alternate."""
    ce = look.contact_events(arrs)
    W = len(arrs["t"]) * float(arrs["dt"])
    ev = sorted((b, j) for j in range(2) for _, b in ce["swings"][j])
    if len(ev) < 2:
        return dict(n=len(ev), cv=None, max_dev=None, left_to_right_over_right_to_left=None, alternate=None)
    tds = np.array([x for x, _ in ev])
    gaps = np.diff(np.concatenate([tds, [tds[0] + W]]))
    feet = [j for _, j in ev]
    lr = [g for g, j in zip(gaps, feet, strict=True) if j == 0]  # from a left touchdown to the next (right) one
    rl = [g for g, j in zip(gaps, feet, strict=True) if j == 1]
    return dict(n=len(gaps), mean_s=float(gaps.mean()), cv=float(gaps.std() / gaps.mean()),
                max_dev=float(np.max(np.abs(gaps / gaps.mean() - 1.0))),
                left_to_right_over_right_to_left=float(np.mean(lr) / np.mean(rl)) if lr and rl else None,
                alternate=bool(all(a != b for a, b in zip(feet, feet[1:] + feet[:1], strict=True))))


def foot_sparc(arrs: dict) -> dict:
    """``sparc`` of each toe tip's world speed over each detected swing (from the last frame on the floor before it to
    the first after it): per foot the mean and lowest, and over both."""
    dt, n = float(arrs["dt"]), len(arrs["t"])
    adv = float(np.asarray(arrs["stride_motion"])[3])
    names = list(np.asarray(arrs["point_names"]))
    P = np.asarray(arrs["points"], float)
    ce = look.contact_events(arrs)
    on = ce["on_floor"]
    out = {}
    allv = []
    for j, side in enumerate(look.SIDES):
        sp = np.linalg.norm(periodic_velocity(P[:, names.index(f"toe_{side[0]}")], dt, adv), axis=1)
        vals = []
        k = int(np.argmax(on[:, j]))
        for _ in range(n):  # each run of frames off the floor, with the floor frame on each side
            k1 = (k + 1) % n
            if on[k, j] and not on[k1, j]:
                e = k1
                while not on[e % n, j]:
                    e += 1
                idx = np.arange(k, e + 1) % n
                vals.append(sparc(sp[idx], 1.0 / dt))
            k = k1
        out[side] = dict(mean=float(np.mean(vals)), lowest=float(np.min(vals)), swings=len(vals))
        allv += vals
    out["mean"], out["lowest"] = float(np.mean(allv)), float(np.min(allv))
    return out


def signals(arrs: dict, euler=None) -> dict:
    """The phase chart's signals (``PARTS``) on every frame of the window."""
    names = list(np.asarray(arrs["point_names"]))
    if euler is None:
        euler = relative_euler(np.asarray(arrs["q"])[:, 3:7], np.asarray(arrs["q_home"])[3:7])
    return dict(hip_height=np.asarray(arrs["hip_mid"])[:, 2], trunk_roll=euler[:, 0],
                head_pitch=np.asarray(arrs["head_pitch_world"]),
                snout_height=np.asarray(arrs["points"])[:, names.index("snout"), 2], tail_yaw=np.asarray(arrs["tail"]),
                sway=np.asarray(arrs["com"])[:, 1], left_foot_height=np.asarray(arrs["pad_floor_z"])[:, 0],
                left_foot_pitch=np.asarray(arrs["foot_pitch_solved"])[:, 0])


def extremes(arrs: dict, sig: dict | None = None) -> dict:
    """Each part's (``PARTS``) maxima and minima over the periodic window as stride phases (the fractional gait phase:
    0 at the left foot's lift-off): peaks with a prominence of at least ``PROMINENCE`` of the signal's range; a signal
    whose range is under ``FLAT`` is flat (no extremes).  Also the synchronous pairs (module docstring)."""
    from scipy.signal import find_peaks

    sig = signals(arrs) if sig is None else sig
    ph = np.mod(np.asarray(arrs["phase"], float), 1.0)
    n = len(ph)
    out = {}
    for key, _, kinds in PARTS:
        y = np.asarray(sig[key], float)
        rng = float(np.ptp(y))
        row = dict(range=rng, flat=rng < FLAT)
        for kind in kinds:
            if row["flat"]:
                row[kind], row[f"{kind}_frames"] = [], []
                continue
            yy = np.tile(y if kind == "max" else -y, 3)
            pk, _ = find_peaks(yy, prominence=PROMINENCE * rng)
            pk = sorted(int(k) for k in pk[(pk >= n) & (pk < 2 * n)] - n)
            row[kind], row[f"{kind}_frames"] = [float(ph[k]) for k in pk], pk
        out[key] = row
    sync = []
    keys = [k for k, _, _ in PARTS]
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            A, B = out[a].get("max", []), out[b].get("max", [])
            if not A or not B:
                continue
            near = [min(abs(((x - y) + 0.5) % 1.0 - 0.5) for y in B) <= SYNC_CYCLES for x in A]
            if np.mean(near) >= 0.5:
                sync.append([a, b])
    return dict(parts=out, synchronous_pairs=sync, prominence=PROMINENCE, sync_cycles=SYNC_CYCLES)


def bird_curves(H: float, froude: float, speed: str) -> dict:
    """The comparison curves (module docstring) at hip height H and Froude number ``froude``."""
    out = dict(tinamou_duty_band=list(TINAMOU_DUTY[:2]), tinamou_duty_mean=TINAMOU_DUTY[2],
               tinamou_stride_H=float(TINAMOU_STRIDE[0] * froude ** TINAMOU_STRIDE[1]),
               bird_swing_time_s=float(SWING_TIME_K * math.sqrt(H / G)))
    if speed == "fr05":
        out["trex_cadence_steps_s"] = float(TREX["steps_per_s"] * math.sqrt(TREX["hip_m"] / H))
        out["trex_stride_H"] = TREX["stride_rel"]
    return out


def metrics(body: str, speed: str, variant: str, entry: dict, arrs: dict) -> dict:
    """One rung's readings (module docstring)."""
    info = entry["info"]
    H = float(info["stride"]) / float(entry["spec"]["stride_rel"])
    dt = float(arrs["dt"])
    n = len(arrs["t"])
    K = int(info.get("strides", 1))
    adv = float(np.asarray(arrs["stride_motion"])[3])
    meas = look.measure(arrs, dict(info, look=entry["spec"]), H, crouch=entry["spec"]["crouch"])
    # bob
    z = np.asarray(arrs["hip_mid"])[:, 2]
    step = np.floor(2 * np.asarray(arrs["phase"], float) + 1e-9).astype(int) % (2 * K)
    per_step = [float(np.ptp(z[step == i])) for i in range(2 * K) if np.count_nonzero(step == i) > 1]
    rz = float(np.ptp(z))
    bob = dict(pp_mm=1e3 * rz, pp_H=rz / H, per_step_pp_mm=1e3 * float(np.mean(per_step)),
               trough_frac_stance=meas["bob_trough_frac_stance"], share_after_touchdown=meas["bob_share_after_touchdown"],
               crest_hold_share=float(np.mean(z >= z.max() - CREST_BAND * rz)) if rz >= FLAT else None,
               crest_hold_share_sinusoid=CREST_SINUSOID)
    # trunk attitude and the head's
    eul = relative_euler(np.asarray(arrs["q"])[:, 3:7], np.asarray(arrs["q_home"])[3:7])
    deg = np.degrees(eul)
    trunk = dict(roll_rms_deg=float(np.sqrt(np.mean(deg[:, 0] ** 2))), pitch_rms_deg=float(np.sqrt(np.mean(deg[:, 1] ** 2))),
                 yaw_rms_deg=float(np.sqrt(np.mean(deg[:, 2] ** 2))), roll_pp_deg=float(np.ptp(deg[:, 0])),
                 pitch_pp_deg=float(np.ptp(deg[:, 1])), roll_toward_stance=meas.get("trunk_roll_toward_stance"))
    bq = body_quats(model(body), arrs["q"])
    w_trunk = np.degrees(angular_velocity(bq["trunk"], dt))
    w_head = np.degrees(angular_velocity(bq["head"], dt))
    rate_t = float(np.sqrt(np.mean(np.sum(w_trunk**2, axis=1))))
    rate_h = float(np.sqrt(np.mean(np.sum(w_head**2, axis=1))))
    hp = np.asarray(arrs["head_pitch_world"], float)
    pr_h = float(np.sqrt(np.mean(np.degrees(periodic_velocity(hp, dt)) ** 2)))
    pr_t = float(np.sqrt(np.mean(np.degrees(periodic_velocity(eul[:, 1], dt)) ** 2)))
    head = dict(world_pitch_rms_deg=float(np.degrees(np.sqrt(np.mean((hp - hp.mean()) ** 2)))),
                world_pitch_pp_deg=float(np.degrees(np.ptp(hp))), angular_rate_rms_deg_s=rate_h,
                trunk_angular_rate_rms_deg_s=rate_t,
                stabilisation_ratio=(rate_h / rate_t) if rate_t >= STILL_DEG_S else None,
                pitch_rate_rms_deg_s=pr_h, trunk_pitch_rate_rms_deg_s=pr_t,
                pitch_stabilisation_ratio=(pr_h / pr_t) if pr_t >= STILL_DEG_S else None,
                nod_amp_rad=meas["nod_amp_rad"], settle_lag_s=meas.get("settle_lag_s"),
                band_world_pitch_rms_deg=BANDS["head_world_pitch_rms_deg"])
    # sway (screen.sway: Stage 0's, the CoM's per stride and over the window)
    sw = screen.sway(arrs, info, H)
    sway = dict(com_pp_per_stride_max_H=sw["com_pp_per_stride_max_H"], com_pp_H=sw["com_pp_H"],
                stage0_pp_H=sw["stage0_pp_H"], hip_mid_pp_H=sw["hip_mid_pp_H"], spec_H=sw["spec_H"])
    # tail
    psi = float(meas["tail_phase_rad"])
    tail = dict(amp_rad=meas["tail_amp_rad"], phase_rad=psi,
                phase_from_today_cycles=float(((psi - 0.5 * math.pi) / (2 * math.pi) + 0.5) % 1.0 - 0.5),
                tip_toward_stance_in_swing=meas["tail_tip_toward_stance_in_swing"],
                band_cycles=BANDS["tail_phase_cycles"])
    # toe-off, swing, duty, cadence, stride
    toe = dict(at_lift_off_rad=meas["toe_off_at_lift_off_rad"], at_lift_off_range_rad=meas["toe_off_at_lift_off_range_rad"],
               at_lift_off_deg=math.degrees(meas["toe_off_at_lift_off_rad"]), peak_rad=meas["toe_off_peak_rad"],
               reach_rad=-meas["foot_pitch_min_rad"], ground_roll_s=meas["toe_off_ground_roll_s"])
    gait_ = dict(duty=meas["duty"], duty_planned=meas["duty_planned"], swing_time_s=meas["swing_time_s"],
                 swing_time_spread_s=meas["swing_time_spread_s"], cadence_steps_s=meas["cadence_steps_s"],
                 stride_H=meas["stride_H"], lift_peak_frac_swing=meas["lift_peak_frac_swing"],
                 speed_m_s=float(info["speed"]), froude=float(info["speed"]) ** 2 / (G * H),
                 period_s=float(info["period"]))
    birds = bird_curves(H, gait_["froude"], speed)
    # step times, smoothness, extremes
    st = step_times(arrs)
    names = list(np.asarray(arrs["point_names"]))
    v_sn = periodic_velocity(np.asarray(arrs["points"])[:, names.index("snout")], dt, adv)
    v_rel = v_sn - np.array([adv / (n * dt), 0.0, 0.0])
    smooth = dict(foot=foot_sparc(arrs), head=sparc_periodic(v_rel, dt))
    ex = extremes(arrs, signals(arrs, eul))
    tells = dict(constant_hip_height=bob["pp_mm"] < 1.0, flat_feet_at_lift_off=toe["at_lift_off_rad"] < 0.01,
                 identical_steps=(st["cv"] is not None and st["cv"] < 1e-4), metronome_nod=head["nod_amp_rad"] > 0.01,
                 synchronous_extremes=ex["synchronous_pairs"])
    return dict(body=body, speed=speed, variant=variant, features=entry["features"], rung_sha256=entry["sha256"],
                H_m=H, window=dict(frames=n, dt_s=dt, strides=K, seconds=n * dt), bob=bob, trunk=trunk, sway=sway,
                head=head, tail=tail, toe_off=toe, gait=gait_, bird_curves=birds, step_times=st, smoothness=smooth,
                mechanical_tells=tells, extremes=ex)


# ------------------------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def marker_paths(arrs: dict) -> dict:
    """The path markers (``PATH_MARKERS``) in the frame moving with the mean forward speed, each relative to its mean
    (m); a one-stride window closed by its first point a window later."""
    dt, n = float(arrs["dt"]), len(arrs["t"])
    adv = float(np.asarray(arrs["stride_motion"])[3])
    t = np.asarray(arrs["t"], float)
    v = adv / (n * dt)
    names = list(np.asarray(arrs["point_names"]))
    P = np.asarray(arrs["points"], float)
    out = {}
    for key, _ in PATH_MARKERS:
        p = P[:, names.index(key)].copy()
        p[:, 0] -= v * t
        p = np.vstack([p, p[:1]])  # the next window's first point: x + adv - v W = x
        out[key] = p - p[:-1].mean(axis=0)
    return out


def path_plot(title: str, rungs: list, path: Path) -> None:
    """Paths (module docstring) of ``rungs`` [(label, arrays, colour)], V0 first, to ``path`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(PATH_MARKERS), 2, figsize=(11, 11.5), dpi=100, facecolor=SURFACE)
    paths = [(lab, marker_paths(a), c) for lab, a, c in rungs]
    for i, (key, name) in enumerate(PATH_MARKERS):
        for j, (ia, ib, lab_a, lab_b) in enumerate(((0, 2, "forward (mm)", "up (mm)"),
                                                    (0, 1, "forward (mm)", "left (mm)"))):
            ax = axes[i, j]
            _style(ax)
            for lab, p, c in paths:
                many = len(p[key]) > 400  # a window of many strides: thinner, so the strides stay apart
                ax.plot(1e3 * p[key][:, ia], 1e3 * p[key][:, ib], color=c, lw=1.0 if many else 1.6,
                        alpha=1.0 if c == BLUE else 0.9, solid_joinstyle="round", solid_capstyle="round", label=lab)
            for get, put in ((ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)):
                lo, hi = get()
                if hi - lo < MIN_SPAN_MM:  # no axis narrower than MIN_SPAN_MM (a still coordinate is not magnified)
                    mid = 0.5 * (lo + hi)
                    put(mid - 0.5 * MIN_SPAN_MM, mid + 0.5 * MIN_SPAN_MM)
            ax.set_xlabel(lab_a, fontsize=8, color=INK2)
            ax.set_ylabel(lab_b, fontsize=8, color=INK2)
            ax.set_title(f"{name}, {'side' if j == 0 else 'top'}", fontsize=10, color=INK, loc="left")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, x=0.02, y=0.992, ha="left", fontsize=11, color=INK)
    fig.text(0.02, 0.967, "kinematic reference, not a simulation; each path in the frame moving with the walk's mean "
             f"forward speed, relative to its own mean; each panel scaled to its own range (at least {MIN_SPAN_MM:g} mm)",
             fontsize=8, color=INK2)
    fig.legend(h, lab, loc="upper left", ncol=len(lab), frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.015, 0.962))
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def phase_chart(title: str, rungs: list, path: Path) -> None:
    """Phase charts (module docstring) of ``rungs`` [(label, arrays, metrics)], one panel each, to ``path`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(rungs)
    fig, axes = plt.subplots(1, k, figsize=(max(7.5, 2.15 * k + 2.2), 6.6), dpi=100, facecolor=SURFACE,
                             sharey=True, squeeze=False)
    rows = len(PARTS)
    for c, (lab, arrs, met) in enumerate(rungs):
        ax = axes[0, c]
        _style(ax)
        ax.grid(False)
        ph = np.mod(np.asarray(arrs["phase"], float), 1.0)
        sig = signals(arrs)
        duty = met["gait"]["duty_planned"]
        for j, (x0, d, name) in enumerate(((0.0, duty[0], "L swing"), (0.5, duty[1], "R swing"))):  # shaded swings
            sw, yb = 1.0 - d, rows + 0.12 + 0.42 * j
            spans = [(x0, min(x0 + sw, 1.0) - x0)] + ([(0.0, x0 + sw - 1.0)] if x0 + sw > 1.0 else [])
            ax.broken_barh(spans, (yb, 0.32), color=GRID, lw=0)
            ax.text(x0 + 0.01, yb + 0.16, name, fontsize=6.5, color=INK2, va="center")
        for r, (key, name, _) in enumerate(PARTS):
            y0 = rows - 1 - r
            ax.axhline(y0, color=GRID, lw=0.6)
            e = met["extremes"]["parts"][key]
            if e["flat"]:
                ax.text(0.5, y0 + 0.4, "flat", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            y = np.asarray(sig[key], float)
            yn = (y - y.min()) / max(np.ptp(y), 1e-12)
            cut = np.flatnonzero(np.diff(ph) < -0.5) + 1  # a new stride starts
            for seg in np.split(np.arange(len(ph)), cut):
                ax.plot(ph[seg], y0 + 0.08 + 0.74 * yn[seg], color=INK2, lw=0.8, alpha=0.6 if len(cut) else 0.9)
            for kind in ("max", "min"):
                ks = e.get(f"{kind}_frames", [])
                if not ks:
                    continue
                ax.scatter(ph[ks], y0 + 0.08 + 0.74 * yn[ks], s=26, zorder=3, linewidth=1.2, edgecolor=BLUE,
                           facecolor=BLUE if kind == "max" else SURFACE)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, rows + 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("stride phase (0: left lift-off)", fontsize=8, color=INK2)
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
        if c == 0:
            ax.set_yticks([rows - 1 - r + 0.45 for r in range(rows)])
            ax.set_yticklabels([name for _, name, _ in PARTS], fontsize=8, color=INK)
            ax.tick_params(axis="y", length=0)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="", markerfacecolor=BLUE, markeredgecolor=BLUE, label="maximum"),
               Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=BLUE, label="minimum"),
               Line2D([], [], color=INK2, lw=0.8, label="signal, each stride, scaled to its range")]
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.948, "kinematic reference, not a simulation; staggered extremes read as overlap, extremes all at "
             "once as a machine (research 4.1 item 3)", fontsize=7.5, color=INK2)
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=8, labelcolor=INK2,
               bbox_to_anchor=(0.005, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------------------------
def compact(m: dict) -> dict:
    """A rung's metrics as ``metrics.json`` keeps them: the extremes' stride phases to 4 decimals, without the frame
    indices the phase chart places its markers by."""
    parts = {}
    for key, row in m["extremes"]["parts"].items():
        parts[key] = {k: ([round(float(x), 4) for x in v] if k in ("max", "min") else v) for k, v in row.items()
                      if not k.endswith("_frames")}
    return dict(m, extremes=dict(m["extremes"], parts=parts))


def stage0_screen(rung: dict) -> dict:
    """From the rung's ``screen.json`` block (the Stage 0 screen, read before any render): research 4.1 item 5's
    stance-pad yaw moment (round two's straight-run reading: the single-support stance pad's peak and RMS free yaw
    moment and the friction need) and the screen's verdict."""
    ss = rung["friction"]["single_support"]
    return dict(stance_pad_yaw=dict(peak_nm=ss["mz_abs_max_nm"], rms_nm=ss["mz_rms_nm"], mu_req_max=ss["mu_req_max"],
                                     source="screen.json rungs.<body>.<speed>.<variant>.friction.single_support"),
                stage0=dict(within_limits=rung["within_limits"], over=rung["over"], verdict=rung["verdict"]))


def plot_set(body: str, speeds=look.SPEEDS) -> list[tuple[str, str, tuple]]:
    """The plots per body (module docstring): (kind, speed, variants)."""
    out = []
    for s in speeds:
        out.append(("paths", s, ("V0", "V6")))
        out.append(("phase", s, ("V0", "V6")))
    for v in ("V1", "V2", "V3", "V4", "V5"):
        out.append(("paths", "v040", ("V0", v)))
    out.append(("phase", "v040", ("V0", "V1", "V2", "V3", "V4", "V5", "V6")))
    return out


def _label(variant: str, entry: dict) -> str:
    from .clips import RUNG_TEXT, feature_text

    return f"{variant}: {RUNG_TEXT.get(variant, feature_text(entry['features']))}"


def run(out_dir: Path = OUT, media: Path | None = None, bodies=look.BODIES) -> dict:
    """Every rung's metrics to ``<out>/metrics.json`` and, with ``media``, the plots (``plot_set``) to
    ``<media>/<body>/``."""
    t0 = time.perf_counter()
    prov = plant.provenance()
    doc = json.loads((Path(out_dir) / "look_refs.json").read_text())
    sp_ = Path(out_dir) / screen.SCREEN_JSON
    sdoc = json.loads(sp_.read_text()) if sp_.exists() else None
    res, full, arrays = {}, {}, {}
    for b in bodies:
        for s, entries in doc["bodies"][b].items():
            for v, e in entries.items():
                arrs = screen.load_arrays(e, out_dir)
                m = metrics(b, s, v, e, arrs)
                if sdoc is not None:
                    m.update(stage0_screen(sdoc["rungs"][b][s][v]))
                full[(b, s, v)] = m
                res.setdefault(b, {}).setdefault(s, {})[v] = look._plain(compact(m))
                arrays[(b, s, v)] = arrs
    plots = []
    if media is not None:
        for b in bodies:
            d = Path(media) / b
            d.mkdir(parents=True, exist_ok=True)
            for kind, s, vs in plot_set(b):
                sp = screen.SPEED_LABEL.get(s, s)
                if kind == "paths":
                    name = f"{b}_{s}_paths_{vs[0]}_{vs[1]}.png"
                    rungs = [(_label(v, doc["bodies"][b][s][v]), arrays[(b, s, v)], GREY if v == "V0" else BLUE)
                             for v in vs]
                    path_plot(f"{screen.BODY_LABEL.get(b, b)}, {sp}: paths of the snout, hip, toe and tail tip", rungs,
                              d / name)
                else:
                    name = f"{b}_{s}_phase_{'-'.join(vs)}.png" if len(vs) > 2 else f"{b}_{s}_phase_{vs[0]}_{vs[1]}.png"
                    rungs = [(_label(v, doc["bodies"][b][s][v]), arrays[(b, s, v)], full[(b, s, v)]) for v in vs]
                    phase_chart(f"{screen.BODY_LABEL.get(b, b)}, {sp}: when each part peaks in the stride", rungs,
                                d / name)
                plots.append(f"{b}/{name}")
    out = dict(what="Stage 2 look ladder: machine metrics per rung (research_look 4.1 item 5), on the kinematic "
                    "references (perfect tracking, not a simulation)",
               method=(__doc__ or "").split("Plots (")[0].split("\n\n", 1)[1].strip(),
               inputs=dict(look_refs="look_refs.json (each rung's arrays checked against its sha256)",
                           screen=screen.SCREEN_JSON),
               v040=dict(rule=look.VALUES["v040"]["rule"], per_body={b: screen.v040_between(
                   doc["body_info"][b]["H_m"]) for b in bodies}),
               bands_closed_loop=BANDS, sparc=dict(fc_hz=SPARC_FC, amp_th=SPARC_AMP, padlevel=SPARC_PAD,
                                                   periodic_span_s=SPARC_SPAN_S),
               parts=[dict(key=k, label=lab, extremes=list(kinds)) for k, lab, kinds in PARTS],
               bodies=res, plots=plots, provenance=prov, versions=look._versions(), seconds=time.perf_counter() - t0)
    (Path(out_dir) / METRICS_JSON).write_text(json.dumps(look._plain(out), indent=1))
    return out


# ------------------------------------------------------------------------------------------------
# The summary page (RENDERS.md), from metrics.json and clips.json alone
# ------------------------------------------------------------------------------------------------
RENDERS_MD = "RENDERS.md"
#: The summary's rows: label, path in a rung's metrics, decimals (``_cell``).
SUMMARY_ROWS = (
    ("hip bob, peak to peak (mm)", ("bob", "pp_mm"), 1),
    ("bob crest hold (share; a sinusoid's 0.144)", ("bob", "crest_hold_share"), 3),
    ("trunk roll RMS (deg)", ("trunk", "roll_rms_deg"), 2),
    ("trunk pitch RMS (deg)", ("trunk", "pitch_rms_deg"), 2),
    ("sway, CoM per stride (H; spec 0.12)", ("sway", "com_pp_per_stride_max_H"), 3),
    ("head pitch in the world, RMS (deg; band 1.5)", ("head", "world_pitch_rms_deg"), 2),
    ("head angular rate, RMS (deg/s)", ("head", "angular_rate_rms_deg_s"), 1),
    ("stabilisation ratio, head / trunk", ("head", "stabilisation_ratio"), 2),
    ("tail phase from today's (cycles)", ("tail", "phase_from_today_cycles"), 3),
    ("tail tip toward the stance side in swing (share)", ("tail", "tip_toward_stance_in_swing"), 2),
    ("toe-off at lift-off (rad)", ("toe_off", "at_lift_off_rad"), 3),
    ("swing time (s)", ("gait", "swing_time_s"), 3),
    ("duty (left)", ("gait", "duty", 0), 3),
    ("stride (H)", ("gait", "stride_H"), 3),
    ("cadence (steps/s)", ("gait", "cadence_steps_s"), 2),
    ("step-time CV (%)", ("step_times", "cv"), 2),
    ("SPARC, toe tip per swing (mean)", ("smoothness", "foot", "mean"), 2),
    ("SPARC, head (periodic)", ("smoothness", "head", "sparc"), 2),
    ("stance-pad peak yaw moment (N m)", ("stance_pad_yaw", "peak_nm"), 2),
    ("Stage 0", ("stage0", "within_limits"), None),
)


def _get(d, path):
    for k in path:
        if d is None:
            return None
        d = d[k] if isinstance(k, int) else d.get(k)
    return d


def _cell(v, nd, key=None) -> str:
    if v is None:
        return "-"
    if nd is None:
        return "within" if v else "**over**"
    if key == ("step_times", "cv"):
        v = 100.0 * v
    return f"{v:.{nd}f}"


def markdown(mdoc: dict, cdoc: dict | None) -> str:
    """``RENDERS.md``: the clips (``clips.json``) and every ladder rung's headline readings (``metrics.json``)."""
    out = ["# Stage 2 look: kinematic clips, plots and machine metrics", "",
           "Kinematic references, not simulations: each rung's reference walk played back with perfect tracking and no "
           "physics (`designlab/stage2/clips.py`), and read on its frames (`designlab/stage2/motion.py`); this page is "
           "rendered from `metrics.json` and `clips.json` alone (`python -m designlab.stage2.motion md`). The media "
           "(MP4, PNG) are outside git; `clips.json` names each file with its size and sha256. Nothing is adopted (P24).",
           ""]
    if cdoc is not None:
        clips_ = cdoc["clips"]
        cam = cdoc["cameras"]
        out += ["## Clips", "",
                f"{len(clips_)} clips, {cdoc['seconds']:g} s each at {cdoc['fps']} fps, {cdoc['size'][0]}x"
                f"{cdoc['size'][1]} H.264, each under {cdoc['max_bytes'] / 1e6:g} MB, every frame labelled "
                f"\"{cdoc['label']}\". Cameras: "
                + "; ".join(f"{k}: eye {v['height_m']:g} m high, {v['distance_m']:g} m from the look-at point along "
                            f"the floor, bearing {v['bearing_deg']:g} deg from the walking direction, field of view "
                            f"{v['fovy_deg']:g} deg" for k, v in cam.items())
                + "; both follow the body's mean forward motion. Point-light: "
                f"{', '.join(cdoc['points'])} on black. Render and encode time {cdoc['render_cpu_s'] / 60:.1f} CPU-min "
                f"({cdoc['wall_s'] / 60:.1f} min wall).", "",
                "| Body | Speed | Rung | View | Mode | File | MB |", "|---|---|---|---|---|---|---|"]
        for e in clips_:
            out.append(f"| {e['body']} | {e['speed']} | {e['variant']} | {e['view']} | {e['mode']} | "
                       f"`{e['file']}` | {e['bytes'] / 1e6:.2f} |")
        out.append("")
    out += ["## Machine metrics (research 4.1 item 5)", "",
            "Each rung's readings on its frames (definitions: `metrics.json` `method`). 0.40 m/s: "
            + mdoc["v040"]["rule"] + ".", ""]
    for b, speeds in mdoc["bodies"].items():
        for s in look.SPEEDS:
            rungs = speeds.get(s, {})
            cols = [v for v in look.VARIANTS + ("V5F13",) if v in rungs]
            out += [f"### {screen.BODY_LABEL.get(b, b)}, {screen.SPEED_LABEL.get(s, s)}", "",
                    "| Reading | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for lab, path, nd in SUMMARY_ROWS:
                out.append(f"| {lab} | " + " | ".join(_cell(_get(rungs[v], path), nd, path) for v in cols) + " |")
            out.append("")
    return "\n".join(out) + "\n"


def write_markdown(out_dir: Path = OUT) -> Path:
    mdoc = json.loads((Path(out_dir) / METRICS_JSON).read_text())
    cp = Path(out_dir) / "clips.json"
    path = Path(out_dir) / RENDERS_MD
    path.write_text(markdown(mdoc, json.loads(cp.read_text()) if cp.exists() else None))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.motion", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="metrics.json for every rung and, with --media, the plots")
    p.add_argument("--media", default=None)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--body", action="append", choices=look.BODIES)
    m = sub.add_parser("md", help="RENDERS.md from metrics.json and clips.json")
    m.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(Path(a.out), None if a.media is None else Path(a.media), tuple(a.body or look.BODIES))
    else:
        write_markdown(Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
