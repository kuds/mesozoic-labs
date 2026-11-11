"""Stage 2 Track A: the Stage 0 screen of the look ladder, before anyone looks at it (``STAGE2_PLAN.md`` section 4 A.2
and A.3; ``stage2/research/research_look.md`` 4.1 item 2 and Bottom line 3 and 4).

Simulation only: prescribed references with perfect tracking, priced on the round-two models; nothing is adopted
(P24).  The ladder is not regenerated: every rung of ``stage2/look/look_refs.json`` (V0-V6 and the extras on
``60-N-T-10`` and ``75-M-H-10`` at the nominal walk, 0.40 m/s and Froude 0.05) is read back from its git-ignored
arrays, each file checked against the sha256 the ladder recorded.  Only the tail-phase check generates walks (V0 and V4
with other tails, by ``look.generate`` as the ladder does).  Nothing here changes another module.

1. The tail-phase check (``tail_phase``; research Bottom line 4, plan 4 A.2), first
-----------------------------------------------------------------------------------
Round two read a straight walk's friction from ``turning.screen_model``'s straight run: ``turning.inverse_dynamics``
(mu 0.8, the exact foot-axis split, the double-support friction LPs) summarised by ``turning.summarize``, whose
single-support block is ``turning._foot_stats`` per foot (``runs.straight`` and ``runs.straight_brisk`` in
``<case>/review/screening/<id>_turning.json``; DECISION_LOG F21's friction need is its ``mu_req_max``).  The same
reading (``straight_run``) runs here on each walk as ``look.turning_traj`` hands it to ``turning.inverse_dynamics`` (the
path frame at yaw rate 0, the window's exact unwrap).  At each body and speed:

* V0 (today's tail: 0.12 rad at pi/2), V0 with no tail, V0 with F9's phase at today's 0.12 rad (the phase alone), V0
  with F9's tail (0.125 rad at F9's phase), and a control: V0 with today's phase written pi/2 + 2 pi, which takes the
  look generator's path (as every non-default phase does) instead of today's code;
* V4 (the look walk before F9, with today's tail), V4 with no tail, V4 with F9's tail alone, and the ladder's V5 (F9
  with F10's head), V6, and at 75 cm's animal pace V5F13 (F13's stride with its own F9 phase).

F9's phase is the ladder's (``tail_phase.json``: the minimiser of the stance pad's peak single-support yaw moment on V4,
``look.tail_check``).  Each reading: the stance pad's free yaw moment M_z about its CoP in single support (peak, RMS and
where in the stride the peak comes), the friction need mu_req (F21's: the ellipsoidal limit surface with the pads'
uniform-pressure radius; also mu_req_cop and the utilisation at mu 0.8), and the double-support friction requirement
(``turning.ds_min_mu``) over the double-support frames where no foot rolls (F5's rolling foot stands on its toe edge,
not on the pad hull that LP assumes; those frames are counted, not read).  The record check: V0 at the nominal walk
against round two's recorded ``runs.straight`` and today's brisk walk (``sizing.BRISK``) against
``runs.straight_brisk``, field by field (``_record_compare``; on 75-M-H-10 ``gait.generate`` and the record's
``turning.generate`` classify one touchdown frame differently, turning.py's Events rule, which moves that frame into
single support and a minimum such as ``r_cop_min`` with it, not the friction need).  Gate G2 also runs here (``g2``),
on the code that runs the screen.

2. The screen of every rung (``screen_rung``), before any render
-----------------------------------------------------------------
* ``drives``: ``look.price`` re-run on the arrays (and checked against the price ``look_refs.json`` records):
  ``look.look_inverse_dynamics`` (``gait.inverse_dynamics`` with F5's rolling contact, identical to it on every frame
  where no foot rolls; accuracy r2) -> ``sizing._motor_space`` -> ``envelope._ratios`` against the record's drives at
  its bus: each group's RMS, peak and speed, cont = RMS / sustained (A1: 0.57 x rated), peak = 1.25 x peak / planning
  peak, speed = 1.15 x speed / speed at the bus (the knee's speed at the animal pace, hip pitch, the ankles' two
  motors, neck, tail, hip roll, thigh twist).
* ``passive`` (research Bottom line 3): the neck and tail with the MJCF's joint damping b and friction loss f, which
  the plant keeps and Stage 0 leaves out: tau + b omega + f sign(omega), RMS and peak in N m with and without, the
  passive terms alone, the ratios (``look._passive_ratios``' own, checked equal) and the servo's mean power (negative:
  the body's motion drives the joint and the servo brakes it, which the damping then helps).
* ``summary``: ``gait.summarize`` (round two's sizing summary) on the same inverse dynamics: the CoP margins over every
  frame's flat feet and over single support, the split's largest friction ratio.
* ``contact``: ``look.price``'s block (single-support CoP margin against the envelope rule's -10 mm, the flat feet's
  double-support CoP margin (split-dependent, a diagnostic), the rolling CoP on its contact, the split's failures, the
  whole contact force's friction ratio) and the multibody ZMP's margin inside the support (``look.measure``, recorded).
* ``friction``: ``straight_run`` on the rung (section 1's reading), against the plan's grips mu 0.9 and 0.54.
* ``sway`` (``sway``): Stage 0's, as round two and Stage 1's G7 read it (2 x the generator's ``sway_amp`` / H, the
  pendulum CoM path's lateral peak to peak); the model CoM's lateral peak to peak per stride (Stage 1's closed-loop
  reading, ``panels.sway``, applied to the reference) and the hip midpoint's; against BEHAVIOR_SPEC's 0.12 H (reported,
  not gated: Stage 1 decision 9); trunk roll RMS against 4 deg.
* ``accel`` (``accelerations``): the CoM's acceleration over the periodic window (second differences, the window's
  advance unwrapped): the horizontal peak (decision 10's reading of the reference, 2.1-2.2 m/s^2 on today's walks),
  its forward and lateral parts, the forward part low-passed at 1 Hz as decision 10's gate reads it
  (``stage1.metrics.lowpass`` on the window repeated), the vertical range (F1/F2's bob); the feet's peak vertical
  acceleration (research F3: the swing foot).  Against the spec's 1.5 m/s^2, a start/stop limit (not gated here).
* ``clearance``: ``look._clearance_gates`` (``clearance.screen``, gates r2, on the round-two MJCF): the swing gate
  (0.04 H over the middle half of each swing), the joint-range margin (0.10 rad), interference, the tail; and the
  foot's lowest point against the floor (``look.measure``).
* ``plain_gait_id`` (F5 rungs only, informational): ``gait.inverse_dynamics`` as round two would run it, whose split
  keeps a rolling foot's CoP in its flat pad box (under a lifted heel); its drive ratios beside the screen's.
* ``over``: ``look.price``'s list (a drive ratio over 1, IK, the CoP rule, the split, a clearance gate); ``near``: drive
  ratios within ``turning.NEAR_FLAG_BAND`` (0.02, the method's own sensitivity) below 1; ``over_with_passive``: the
  neck or tail over a planning limit once the plant's damping and friction are added (outside Stage 0's rule; the
  plan's reading 3, "shown but for A1", and Track D's levers 2 and 4 are for it), and ``near_with_passive`` within
  0.02 below one.  ``rung``: the plan's rule
  (``look.LIMIT_RULE``): a rung over a drive limit keeps its features and loses that speed rung unless the owner
  decides otherwise.  ``cause``: for a drive over its limit, where its peak comes and how it follows the step timing.
* ``vs_V0``: each headline number against V0 at the same body and speed.

Outputs: ``stage2/look/screen.json`` (every number under the source named in ``sources`` and in each block's
``source``) and ``stage2/look/SCREEN.md`` (tables, ``markdown``).

Run from hardware/trex_form_study:
    python -m designlab.stage2.screen run --procs 3 [--resume]      # screen.json and SCREEN.md
    python -m designlab.stage2.screen md                            # SCREEN.md again from screen.json
"""

from __future__ import annotations

import os

if __name__ == "__main__":  # one core per process, as the Stage 1 runner and look.prepare
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .. import envelope, gait, sizing, statics, turning  # noqa: E402
from . import look  # noqa: E402

G = gait.G
SIDES = statics.SIDES
OUT = look.OUT
SCREEN_JSON = "screen.json"
SCREEN_MD = "SCREEN.md"
#: The plan's two grips (STAGE2_PLAN section 5: every bar item at mu 0.9 and mu 0.54).
GRIPS = (0.9, 0.54)
#: BEHAVIOR_SPEC: lateral sway (peak to peak, H; reported against it, not gated: Stage 1 decision 9), trunk roll RMS
#: (deg) and the start/stop horizontal acceleration (m/s^2; decision 10).
SWAY_SPEC_H = 0.12
ROLL_SPEC_RMS_DEG = 4.0
ACCEL_SPEC = 1.5
#: Ratios within this below a limit are listed as near it (``turning.NEAR_FLAG_BAND``: halving the frame interval moves
#: these ratios by up to 0.010).
NEAR_BAND = turning.NEAR_FLAG_BAND
KINDS = ("cont_ratio", "peak_ratio", "speed_ratio")
SPEED_LABEL = {"nominal": "the nominal walk (Froude 0.012)", "v040": "0.40 m/s (the agility bar's speed)",
               "fr05": "the animal pace (Froude 0.05)"}
#: The rungs in ladder order, extras after the rung they extend.
ORDER = ("V0", "V1B50", "V1", "V1B100", "V2", "V3", "V3T", "V4", "V4L", "V5", "V5F13", "V6", "V6S")
#: The tail-phase check's walks (``tail_phase``), in the order of its tables.
TAIL_WALKS = ("V0", "V0_no_tail", "V0_F9_phase", "V0_F9", "V0_look_path", "V4", "V4_no_tail", "V4_F9", "V5", "V5F13",
              "V6")

SOURCES = {
    "inputs": "look_refs.json (the ladder: specs, recorded prices and measures, each rung's arrays' sha256) and "
              "tail_phase.json (F9's phases), as committed; every rung's arrays (stage2/look/refs/, git-ignored) "
              "checked against the recorded sha256 before use (screen.load_arrays)",
    "limits": "the record's drives (look.source -> stage1.refs.source: the body rebuilt from its record; design.act) at "
              "the record's bus: catalog.Actuator.sustained (0.57 x rated, A1), .peak (planning peak), "
              ".speed_at(bus) (servos: rated speed)",
    "tail_phase": "screen.tail_phase: look.generate (V0 and V4 with other tails) or the rung's arrays, each read by "
                  "screen.straight_run; the record check against round two's turning screen (runs.straight, "
                  "runs.straight_brisk)",
    "straight_run": "turning.inverse_dynamics(M, look.turning_traj(traj, M, q_home), mu 0.8, split 'exact', "
                    "ds_friction=True): per foot turning._foot_stats over its single support (turning.summarize's "
                    "single_support block), the stance pad's M_z and mu_req over every single-support frame, "
                    "turning.ds_min_mu's min_mu over the double-support frames where no foot rolls",
    "drives": "look.price (re-run) -> look.look_inverse_dynamics (accuracy r2) -> sizing._motor_space -> "
              "envelope._ratios: rms, peak (N m), speed (rad/s), cont_ratio = rms / sustained, peak_ratio = 1.25 peak / "
              "planning peak, speed_ratio = 1.15 speed / speed at the bus, worst",
    "passive": "screen.passive_loads on look.look_inverse_dynamics' tau and omega: tau + b omega + f sign(omega) with "
               "the MJCF's joint damping b and friction loss f (the plant's: plant.model_files' MJCF, checked); its "
               "ratios equal look._passive_ratios'",
    "summary": "gait.summarize on look.look_inverse_dynamics (round two's sizing summary): min_cop_margin, "
               "min_cop_margin_single, max_friction_ratio",
    "contact": "look.price's contact block (look.look_inverse_dynamics; statics.cop, statics.support_margin) and "
               "look.measure's zmp_margin_min_mm (the multibody ZMP inside each frame's support hull, as "
               "look_refs.json records it)",
    "friction": "screen.straight_run on the rung's trajectory (look.trajectory of its arrays), its rolling frames "
                "excluded from the double-support reading",
    "sway": "screen.sway: stage0_pp_H = 2 x info.sway_amp / H (gait.generate's pendulum CoM path; round two's and "
            "Stage 1 G7's reading); com_pp_* from the arrays' model CoM (lateral peak to peak per stride, "
            "stage1.panels.sway's closed-loop reading); hip_mid_pp_H from the arrays' hip midpoint; trunk roll RMS "
            "from look.measure (recorded)",
    "accel": "screen.accelerations: second differences of the arrays' model CoM over the periodic window (the "
             "window's x advance unwrapped, as look.measure); forward_lp1hz by stage1.metrics.lowpass (1 Hz, 2nd-order "
             "Butterworth, forward and backward; decision 10) on the window repeated; feet from the arrays' foot origins",
    "clearance": "look.price's clearance block (look._clearance_gates: clearance.screen, gates r2, on the round-two "
                 "MJCF) and look.measure's foot_below_floor_max_mm (recorded)",
    "plain_gait_id": "gait.inverse_dynamics (accuracy r2; round two's split, a rolling foot's CoP kept in its flat "
                     "pad box) -> sizing._motor_space -> envelope._ratios: each group's worst ratio, the governing "
                     "ratio, those over 1 and the largest difference of any ratio from the screen's; F5 rungs only, "
                     "informational",
    "over": "look.price's over list: a drive ratio over 1, IK, the envelope's single-support CoP rule (-10 mm), the "
            "rolling split, a clearance gate",
    "near": "drive ratios within turning.NEAR_FLAG_BAND below 1",
    "over_with_passive": "screen.passive_loads' ratios over 1 (outside Stage 0's rule; for the owner)",
    "near_with_passive": "screen.passive_loads' ratios within turning.NEAR_FLAG_BAND below 1",
    "rung": "look.LIMIT_RULE (STAGE2_PLAN 4 A.3) applied to over",
    "cause": "screen.peak_speed_cause: the joint's speed peak (look.look_inverse_dynamics' omega) per detected swing "
             "(look.contact_events), from 0.15 T before its lift-off to its touchdown, against the swing's duration",
    "look": "look.measure, as look_refs.json records it (the features' measured amounts, context for the costs)",
    "vs_V0": "screen.changes: the rung's value against V0's at the same body and speed",
    "g2": "screen.g2: look.g2_hashes (stage1.refs.generate at r2, refs.write's npz) against the sha256 "
          "stage1/runs/2026-11-08/<body>/refs.json records, for nominal, brisk and fr05, on the code that ran the screen",
}


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(e: dict, out_dir: Path = OUT) -> dict:
    """A rung's arrays, checked against the sha256 ``look_refs.json`` records for them."""
    p = Path(out_dir) / e["file"]
    sha = _sha(p)
    if sha != e["sha256"]:
        raise ValueError(f"{p} is not the file look_refs.json records (sha256 {sha[:12]}..., recorded "
                         f"{e['sha256'][:12]}...)")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


_DOCS: dict = {}


def _docs(out_dir: Path = OUT) -> tuple[dict, dict]:
    key = str(Path(out_dir).resolve())
    if key not in _DOCS:
        _DOCS[key] = (json.loads((Path(out_dir) / "look_refs.json").read_text()),
                      json.loads((Path(out_dir) / "tail_phase.json").read_text()))
    return _DOCS[key]


def limits(src: dict) -> dict:
    """The record's drives at its bus: each group's sustained (A1), planning peak and usable speed."""
    bus = src["record"]["selection"]["bus"]
    out = {}
    for g in sizing.GROUPS:
        a = src["design"].act(g)
        out[g] = dict(actuator=a.id, sustained_nm=float(a.sustained), peak_nm=float(a.peak),
                      speed_rad_s=float(a.speed_at(bus if a.box is None else None)),
                      bus_v=bus if a.box is None else None)
    return out


def v040_between(H: float) -> dict:
    """0.40 m/s's place between the two walks ``look.speed_spec`` mixes (the brisk walk, ``sizing.BRISK``, and the
    animal pace, ``envelope.FAST``): their speeds, periods and duties and the speed's weight w."""
    a, b = sizing.BRISK.describe(H), envelope.FAST.describe(H)
    return dict(weight_w=float((look.AGILITY_SPEED - a["speed"]) / (b["speed"] - a["speed"])),
                brisk=dict(speed_m_s=float(a["speed"]), period_s=float(a["period"]), duty=float(a["duty"]),
                           stride_rel=float(a["stride_rel"])),
                fr05=dict(speed_m_s=float(b["speed"]), period_s=float(b["period"]), duty=float(b["duty"]),
                          stride_rel=float(b["stride_rel"])))


def plant_joint_friction(body: str) -> dict:
    """The neck-pitch and tail-yaw joints' damping and friction loss in the plant's MJCF (``plant.model_files``)."""
    from ..stage1 import plant

    xml, _ = plant.model_files(body)
    m = mujoco.MjModel.from_xml_path(str(xml))
    out = {}
    for jn in ("neck_pitch", "tail_yaw"):
        dof = m.jnt_dofadr[statics.jid(m, jn)]
        out[jn] = dict(damping=float(m.dof_damping[dof]), frictionloss=float(m.dof_frictionloss[dof]))
    return out


# ------------------------------------------------------------------------------------------------
# Readings
# ------------------------------------------------------------------------------------------------
def _finite_max(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.max()) if len(x) else None


def straight_run(src: dict, traj: dict, rolling=None, phase=None) -> dict:
    """Round two's straight-run friction reading on one walk (module docstring, section 1): per foot
    ``turning._foot_stats`` over its single support; over every single-support frame the stance pad's M_z (peak, RMS,
    the stride fraction of its peak) and the friction need (mu_req, mu_req_cop, the utilisation at mu 0.8, the stride
    fraction of the largest mu_req); the double-support requirement over the frames where no foot rolls; the grips."""
    M, q0 = src["M"], src["q_home"]
    idr = turning.inverse_dynamics(M, look.turning_traj(traj, M, q0), turning.MU, split="exact", ds_friction=True)
    phases = traj["phases"]
    n = len(phases)
    weight = float(M.mass() * G)
    rol = np.zeros((n, 2), bool) if rolling is None else np.asarray(rolling, bool)
    ph = np.arange(n) / n if phase is None else np.mod(np.asarray(phase, float), 1.0)
    feet = {}
    for side in SIDES:
        single = np.array([k for k, p in enumerate(phases) if p == (side,)], int)
        feet[side] = turning._foot_stats(idr["feet"][side], single, phases)
    ss = np.array([k for k, p in enumerate(phases) if len(p) == 1], int)
    if rol[ss].any():
        raise ValueError("a foot rolls in single support")
    stance = [phases[k][0] for k in ss]

    def per_frame(key):
        return np.array([idr["feet"][s][key][k] for s, k in zip(stance, ss, strict=True)], float)

    mz, mu, mu_cop, util, fn = (per_frame(k) for k in ("mz", "mu_req", "mu_req_cop", "util", "fn"))
    k_mz = int(ss[int(np.nanargmax(np.abs(mz)))])
    k_mu = int(ss[int(np.nanargmax(mu))])
    ds = [k for k, p in enumerate(phases) if len(p) == 2]
    flat = np.array([k for k in ds if not rol[k].any()], int)
    D = idr["ds"]
    mu_max = float(np.nanmax(mu))
    return dict(
        single_support=dict(frames=len(ss), mz_abs_max_nm=float(np.nanmax(np.abs(mz))),
                            mz_rms_nm=float(np.sqrt(np.nanmean(mz**2))), mz_peak_stride_frac=float(ph[k_mz]),
                            mz_peak_stance=phases[k_mz][0], mu_req_max=mu_max, mu_req_max_stride_frac=float(ph[k_mu]),
                            mu_req_max_stance=phases[k_mu][0], mu_req_cop_max=_finite_max(mu_cop),
                            util_max=_finite_max(util), fn_min_frac_weight=float(np.nanmin(fn) / weight),
                            mu_req_peak_frame_is_mz_peak_frame=bool(k_mz == k_mu)),
        feet=look._plain(feet),
        double_support=dict(frames=len(ds), read=len(flat), rolling_frames_not_read=len(ds) - len(flat),
                            min_mu_max=_finite_max(D["min_mu"][flat]) if len(flat) else None,
                            min_mu_lower_max=_finite_max(D["min_mu_lower"][flat]) if len(flat) else None,
                            infeasible_frames=int(np.sum(~D["feasible"][flat])) if len(flat) else 0),
        grips={f"{g:g}": bool(mu_max <= g) for g in GRIPS}, mu_assumed=turning.MU)


def passive_loads(M: statics.Model, design, tau, omega) -> dict:
    """The neck and tail loads with the MJCF's joint damping b and friction loss f added to the inverse-dynamics torque
    (tau + b omega + f sign(omega); research Bottom line 3), in N m and as ratios to the planning limits."""
    m = M.m
    out = {}
    for grp, jn in (("neck", "neck_pitch"), ("tail", "tail_yaw")):
        i = M.names.index(jn)
        dof = m.jnt_dofadr[statics.jid(m, jn)]
        b, f = float(m.dof_damping[dof]), float(m.dof_frictionloss[dof])
        t_id, w = np.asarray(tau)[:, i], np.asarray(omega)[:, i]
        t_p = b * w + f * np.sign(w)
        t_tot = t_id + m.dof_damping[dof] * w + m.dof_frictionloss[dof] * np.sign(w)  # look._passive_ratios' order
        a = design.act(grp)

        def rms(x):
            return float(np.sqrt(np.mean(np.square(x))))

        out[grp] = dict(joint=jn, actuator=a.id, damping=b, frictionloss=f, sustained_nm=float(a.sustained),
                        planning_peak_nm=float(a.peak), rms_id_nm=rms(t_id), rms_passive_alone_nm=rms(t_p),
                        rms_total_nm=rms(t_tot), peak_id_nm=float(np.max(np.abs(t_id))),
                        peak_total_nm=float(np.max(np.abs(t_tot))), cont_ratio_id=rms(t_id) / a.sustained,
                        cont_ratio_total=rms(t_tot) / a.sustained,
                        peak_ratio_id=sizing.PEAK_MARGIN * float(np.max(np.abs(t_id))) / a.peak,
                        peak_ratio_total=sizing.PEAK_MARGIN * float(np.max(np.abs(t_tot))) / a.peak,
                        rms_total_over_id=rms(t_tot) / max(rms(t_id), 1e-12),
                        speed_rms_rad_s=rms(w), moving_frames_frac=float(np.mean(np.abs(w) > 0)),
                        # who drives the joint: the servo's mean power without and with the passive terms (W; negative:
                        # the body's motion drives the joint and the servo brakes it, which the damping then helps)
                        power_mean_id_w=float(np.mean(t_id * w)), power_mean_total_w=float(np.mean(t_tot * w)))
    return out


def sway(arrs: dict, info: dict, H: float, measured: dict | None = None) -> dict:
    """Lateral sway (peak to peak, H): Stage 0's (2 x the generator's ``sway_amp`` / H), the model CoM's over the window
    and per stride (mean and largest), the hip midpoint's; trunk roll RMS (deg, ``look.measure``)."""
    K = int(info.get("strides", 1))
    y = np.asarray(arrs["com"])[:, 1]
    idx = np.floor(np.asarray(arrs["phase"], float) + 1e-9).astype(int) % K
    per = [float(np.ptp(y[idx == i]) / H) for i in range(K) if np.count_nonzero(idx == i) > 1]
    out = dict(stage0_pp_H=float(2.0 * info["sway_amp"] / H), com_pp_H=float(np.ptp(y) / H),
               com_pp_per_stride_mean_H=float(np.mean(per)), com_pp_per_stride_max_H=float(np.max(per)),
               hip_mid_pp_H=float(np.ptp(np.asarray(arrs["hip_mid"])[:, 1]) / H), spec_H=SWAY_SPEC_H,
               strides=K)
    if measured is not None:
        out.update(trunk_roll_rms_deg=float(measured["trunk_roll_rms_deg"]), trunk_roll_spec_rms_deg=ROLL_SPEC_RMS_DEG)
    return out


def _periodic_second_difference(x: np.ndarray, dt: float, advance: np.ndarray | None = None) -> np.ndarray:
    prev, nxt = np.roll(x, 1, axis=0), np.roll(x, -1, axis=0)
    if advance is not None:
        prev[0] -= advance
        nxt[-1] += advance
    return (nxt - 2 * x + prev) / dt**2


def accelerations(arrs: dict) -> dict:
    """The model CoM's accelerations over the periodic window (m/s^2; the window's x advance unwrapped): horizontal
    peak, forward and lateral peaks, the forward part low-passed at 1 Hz (decision 10's filter, on the window repeated
    to cover at least 20 s each side), vertical range; the feet's (origins') peak vertical acceleration."""
    from ..stage1 import metrics

    dt = float(arrs["dt"])
    com = np.asarray(arrs["com"], float)
    n = len(com)
    adv = np.array([float(np.asarray(arrs["stride_motion"])[3]), 0.0, 0.0])
    acc = _periodic_second_difference(com, dt, adv)
    reps = 2 * int(math.ceil(20.0 / (n * dt))) + 1
    lp = metrics.lowpass(np.tile(acc[:, 0], reps), 1.0 / dt)[(reps // 2) * n : (reps // 2 + 1) * n]
    feet_z = np.asarray(arrs["feet"], float)[:, :, 2]
    fa = _periodic_second_difference(feet_z, dt)
    return dict(com_horizontal_peak=float(np.max(np.linalg.norm(acc[:, :2], axis=1))),
                com_forward_peak=float(np.max(np.abs(acc[:, 0]))), com_lateral_peak=float(np.max(np.abs(acc[:, 1]))),
                com_forward_lp1hz_peak=float(np.max(np.abs(lp))),
                com_vertical_min=float(acc[:, 2].min()), com_vertical_max=float(acc[:, 2].max()),
                foot_vertical_peak=float(np.max(np.abs(fa))), foot_vertical_up_peak=float(fa.max()),
                spec_start_stop=ACCEL_SPEC)


def _drive_rows(drives: dict) -> list[tuple[str, str, float]]:
    return [(g, k, float(v[k])) for g, v in drives.items() for k in KINDS]


def governing(drives: dict) -> list:
    g, k, r = max(_drive_rows(drives), key=lambda x: x[2])
    return [g, k.replace("_ratio", ""), r]


def peak_speed_cause(src: dict, arrs: dict, info: dict, omega, joint: str) -> dict:
    """Where a joint's speed peak comes: the frame (side, stride fraction, swing, rolling or flat stance, the swing
    fraction) and, per detected swing (``look.contact_events``), the joint's peak speed from 0.15 T before the swing's
    lift-off (the late-stance roll) to its touchdown against the swing's duration (their correlation)."""
    names = src["M"].names
    t, dt = np.asarray(arrs["t"]), float(arrs["dt"])
    n = len(t)
    W, T = n * dt, float(info["period"])
    lead = 0.15 * T
    ce = look.contact_events(arrs)
    cols = [names.index(f"{s}_{joint}") for s in SIDES]
    w = np.abs(np.asarray(omega)[:, cols])
    k, j = np.unravel_index(int(np.argmax(w)), w.shape)
    sw, rol = np.asarray(arrs["swing"], bool), np.asarray(arrs["rolling"], bool)
    state = "swing" if sw[k, j] else ("rolling stance" if rol[k, j] else "flat stance")
    rows = []
    for jj, s in enumerate(SIDES):
        for a, b in ce["swings"][jj]:
            dur = (b - a) % W
            sel = ((t - (a - lead)) % W) <= dur + lead
            rows.append(dict(side=s, lift_off_s=float(a), swing_s=float(dur), peak_rad_s=float(w[sel, jj].max())))
    d = np.array([r["swing_s"] for r in rows])
    p = np.array([r["peak_rad_s"] for r in rows])
    corr = float(np.corrcoef(d, p)[0, 1]) if len(rows) > 2 and np.ptp(d) > 0 and np.ptp(p) > 0 else None
    fastest = int(np.argmax(p))
    return dict(joint=joint, peak_rad_s=float(w[k, j]), side=SIDES[j], t_s=float(t[k]),
                stride_frac=float(np.mod(np.asarray(arrs["phase"])[k], 1.0)), state=state,
                swing_frac=(float(np.asarray(arrs["swing_u"])[k, j]) if sw[k, j] else None),
                swings=len(rows), swing_s_range=[float(d.min()), float(d.max())],
                peak_rad_s_range=[float(p.min()), float(p.max())], fastest_swing_s=float(d[fastest]),
                swing_vs_peak_corr=corr)


# ------------------------------------------------------------------------------------------------
# The screen of one rung
# ------------------------------------------------------------------------------------------------
def _max_abs_diff(a, b, path: str = "") -> tuple[float, list]:
    """The largest absolute difference over two JSON-like trees' numeric leaves, and the paths whose other leaves or
    keys differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        worst, bad = 0.0, []
        if set(a) != set(b):
            bad.append(f"{path}: keys")
        for k in set(a) & set(b):
            w, bb = _max_abs_diff(a[k], b[k], f"{path}.{k}")
            worst, bad = max(worst, w), bad + bb
        return worst, bad
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 0.0, [f"{path}: length"]
        worst, bad = 0.0, []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            w, bb = _max_abs_diff(x, y, f"{path}[{i}]")
            worst, bad = max(worst, w), bad + bb
        return worst, bad
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        return 0.0, ([] if a == b else [path])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return 0.0, []
        return float(abs(a - b)), []
    return 0.0, ([] if a == b else [path])


LOOK_KEYS = ("bob_pp_H", "duty", "swing_time_s", "cadence_steps_s", "lift_peak_frac_swing", "toe_off_at_lift_off_rad",
             "trunk_roll_amp_rad", "trunk_roll_rms_deg", "tail_amp_rad", "tail_phase_rad", "nod_amp_rad",
             "head_world_pitch_rms_deg", "stride_H", "step_time_dev_max", "zmp_margin_min_mm",
             "foot_below_floor_max_mm")


def screen_rung(src: dict, speed: str, variant: str, e: dict, out_dir: Path = OUT) -> dict:
    """One rung's Stage 0 screen (module docstring, section 2), from its arrays."""
    t0 = time.perf_counter()
    M, design = src["M"], src["design"]
    arrs = load_arrays(e, out_dir)
    info = e["info"]
    tr = look.trajectory(arrs, info)
    pr = look._plain(look.price(src, arrs, info))
    diff, bad = _max_abs_diff(pr, e["price"])
    with look.accuracy():
        idr = look.look_inverse_dynamics(M, tr, arrs["rolling"], arrs["toe_edge"])
    bus = src["record"]["selection"]["bus"]
    drives = envelope._ratios(design, bus, sizing._motor_space(M.names, idr["tau"], idr["omega"], src["layout"]))
    d2, _ = _max_abs_diff(look._plain(drives), pr["drives"])
    if d2 != 0.0:
        raise AssertionError(f"the screen's drive ratios differ from look.price's by {d2}")
    passive = passive_loads(M, design, idr["tau"], idr["omega"])
    ref = look._passive_ratios(M, design, idr["tau"], idr["omega"])
    for grp in ("neck", "tail"):
        for mine, theirs in (("cont_ratio_total", "cont_ratio"), ("peak_ratio_total", "peak_ratio")):
            if not math.isclose(passive[grp][mine], ref[grp][theirs], rel_tol=1e-12, abs_tol=0.0):
                raise AssertionError(f"passive_loads differs from look._passive_ratios on the {grp} ({mine})")
    s = gait.summarize(M, tr, idr)
    summary = dict(min_cop_margin_mm=1e3 * s["min_cop_margin"], min_cop_margin_single_mm=1e3 * s["min_cop_margin_single"],
                   max_friction_ratio=s["max_friction_ratio"], ik_ok=s["ik_ok"], ik_fail_frames=s["ik_fail_frames"])
    rolling = np.asarray(arrs["rolling"], bool)
    fr = straight_run(src, tr, rolling, arrs["phase"])
    m = e["measured"]
    contact = dict(pr["contact"], zmp_margin_min_mm=m["zmp_margin_min_mm"])
    clear = dict(pr["clearance"], foot_below_floor_max_mm=m["foot_below_floor_max_mm"])
    clear.pop("joint_range_margins", None)
    plain = None
    if rolling.any():
        with look.accuracy():
            gi = gait.inverse_dynamics(M, tr)
        pd = envelope._ratios(design, bus, sizing._motor_space(M.names, gi["tau"], gi["omega"], src["layout"]))
        plain = dict(worst={g: float(v["worst"]) for g, v in pd.items()}, governing=governing(pd),
                     over=[f"{g} {k.replace('_ratio', '')}" for g, k, r in _drive_rows(pd) if r > 1.0],
                     max_ratio_diff=max(abs(float(pd[g][k]) - float(pr["drives"][g][k])) for g in pd for k in KINDS))
    over = pr["over"]
    near = [dict(limit=f"{g} {k.replace('_ratio', '')}", value=r) for g, k, r in _drive_rows(pr["drives"])
            if 1.0 - NEAR_BAND < r <= 1.0]
    over_p = [dict(limit=f"{grp} {kind.replace('_ratio_total', '')} with the plant's damping and friction",
                   value=passive[grp][kind]) for grp in ("neck", "tail")
              for kind in ("cont_ratio_total", "peak_ratio_total") if passive[grp][kind] > 1.0]
    near_p = [dict(limit=f"{grp} {kind.replace('_ratio_total', '')} with the plant's damping and friction",
                   value=passive[grp][kind]) for grp in ("neck", "tail")
              for kind in ("cont_ratio_total", "peak_ratio_total") if 1.0 - NEAR_BAND < passive[grp][kind] <= 1.0]
    drive_over = [o for o in over if o["limit"].split()[0] in sizing.GROUPS]
    if over:
        rung = (f"over ({'; '.join(o['limit'] for o in over)}): keeps its features "
                f"({', '.join(e['features']) or 'none'}) and loses {SPEED_LABEL[speed]} on {src['body']} unless the "
                "owner decides otherwise (plan 4 A.3)")
    else:
        rung = "within the Stage 0 limits: kept at this speed"
    cause = {}
    for o in drive_over:
        g, kind = o["limit"].split()[:2]
        if kind == "speed" and g in sizing.JOINT_OF:
            cause[o["limit"]] = peak_speed_cause(src, arrs, info, idr["omega"], sizing.JOINT_OF[g])
    return look._plain(dict(
        body=src["body"], speed=speed, variant=variant, features=e["features"], flags=e["flags"],
        arrays=dict(file=e["file"], sha256=e["sha256"], frames=e["frames"], window_s=info.get("window_s"),
                    strides=info.get("strides", 1)),
        gait=dict(speed_m_s=info["speed"], period_s=info["period"], duty=info["duty"], stride_rel=info["stride_rel"],
                  swing_time_s=info.get("swing_time")),
        drives=pr["drives"], governing=governing(pr["drives"]), passive=passive, summary=summary, contact=contact,
        friction=fr, sway=sway(arrs, info, src["H"], m), accel=accelerations(arrs), clearance=clear,
        plain_gait_id=plain, over=over, near=near, over_with_passive=over_p, near_with_passive=near_p,
        verdict=pr["verdict"], rung=rung,
        within_limits=pr["within_limits"], cause=cause or None, look={k: m.get(k) for k in LOOK_KEYS},
        reprice=dict(max_abs_diff=diff, other_differences=bad, equal_to_recorded=bool(diff == 0.0 and not bad)),
        seconds=time.perf_counter() - t0))


# ------------------------------------------------------------------------------------------------
# The tail-phase check
# ------------------------------------------------------------------------------------------------
def _reading_from_arrays(src: dict, e: dict, out_dir: Path) -> dict:
    arrs = load_arrays(e, out_dir)
    tr = look.trajectory(arrs, e["info"])
    r = straight_run(src, tr, arrs["rolling"], arrs["phase"])
    return dict(r, tail=dict(amp_rad=e["spec"]["tail_amp"], phase_rad=e["spec"]["tail_phase"]), source="rung arrays",
                spec_differs_from=None)


def _reading_generated(src: dict, spec: gait.GaitSpec, base: str, change: str) -> tuple[dict, dict]:
    g = look.generate(src, spec)
    r = straight_run(src, g, g["look"]["rolling"], g["look"]["phase"])
    return dict(r, tail=dict(amp_rad=spec.tail_amp, phase_rad=spec.tail_phase), source="look.generate",
                spec_differs_from={base: change}, look_path=gait.look_active(spec)), g


#: The friction fields of a turning-screen straight run (``turning._foot_stats`` per foot) that set the friction need.
FRICTION_FIELDS = ("mz_abs_max", "mu_req_max", "mu_req_cop_max", "mu_req_centroid_max", "util_max", "util_cop_max",
                   "ft_over_fn_max")


def _record_compare(mine: dict, rec: dict) -> dict:
    """``straight_run``'s per-foot single-support stats and double-support requirement against a recorded turning-screen
    run (``runs.<key>`` of ``<id>_turning.json``), field by field: the largest relative difference over the friction
    fields (``FRICTION_FIELDS`` and the double-support min mu) and over every numeric field, each field that differs by
    more than 1e-9, and each foot's single-support frame count.  ``gait.generate`` and ``turning.generate`` (the record's)
    can classify a touchdown frame differently (``turning`` docstring, Events), which moves a frame between double and
    single support: the counts then differ by one and the fields that frame enters (a minimum such as ``r_cop_min``)
    with them."""
    rel_f, rel_all, n, differing = 0.0, 0.0, 0, []
    for side in SIDES:
        a, b = mine["feet"][side], rec["feet"][side]["single_support"]
        for k, v in b.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            n += 1
            rel = abs(a[k] - v) / max(abs(v), 1e-12)
            rel_all = max(rel_all, rel)
            if k in FRICTION_FIELDS:
                rel_f = max(rel_f, rel)
            if rel > 1e-9:
                differing.append(dict(side=side, field=k, mine=a[k], record=v, rel=rel))
    rd, md = rec["double_support"]["min_mu_max"], mine["double_support"]["min_mu_max"]
    rel_ds = abs(md - rd) / max(abs(rd), 1e-12)
    frames = {s: [mine["feet"][s]["frames"], rec["feet"][s]["single_support"]["frames"]] for s in SIDES}
    return dict(fields=n + 1, friction_max_rel_diff=max(rel_f, rel_ds), all_max_rel_diff=max(rel_all, rel_ds),
                differing=differing, single_support_frames=frames,
                frame_rule_note=(None if all(a == b for a, b in frames.values()) else
                                 "gait.generate and turning.generate classify a touchdown frame differently (turning "
                                 "docstring, Events): one frame more or less in single support"),
                mu_req_max=[max(mine["feet"][s]["mu_req_max"] for s in SIDES),
                            max(rec["feet"][s]["single_support"]["mu_req_max"] for s in SIDES)],
                mz_abs_max_nm=[max(mine["feet"][s]["mz_abs_max"] for s in SIDES),
                               max(rec["feet"][s]["single_support"]["mz_abs_max"] for s in SIDES)],
                ds_min_mu_max=[md, rd])


def _change(a: dict, b: dict) -> dict:
    """Walk b against walk a: the peak and RMS yaw moment, the friction need and the double-support requirement."""
    sa, sb = a["single_support"], b["single_support"]
    out = {}
    for k in ("mz_abs_max_nm", "mz_rms_nm", "mu_req_max", "mu_req_cop_max"):
        out[k] = dict(abs=sb[k] - sa[k], rel=sb[k] / sa[k] - 1.0)
    da, db = a["double_support"]["min_mu_max"], b["double_support"]["min_mu_max"]
    out["ds_min_mu_max"] = None if da is None or db is None else dict(abs=db - da, rel=db / da - 1.0)
    return out


def tail_phase(src: dict, speed: str, out_dir: Path = OUT) -> dict:
    """The tail-phase check at one body and speed (module docstring, section 1)."""
    t0 = time.perf_counter()
    body, H = src["body"], src["H"]
    ladder_doc, tail_doc = _docs(out_dir)
    rungs = ladder_doc["bodies"][body][speed]
    trec = tail_doc["bodies"][body][speed]
    phi9 = float(trec["tail_phase_F9_rad"])
    amp9 = float(look.feature_values(body, H, speed)["F9F10"]["tail_amp"])
    lad = look.ladder(body, H, speed, phi9, tail_phase_f13=trec.get("tail_phase_F9_F13_rad"))
    for v in ("V0", "V4", "V5"):
        if look._plain(lad[v].__dict__) != rungs[v]["spec"]:
            raise ValueError(f"{body} {speed} {v}: the ladder's spec is not the one look_refs.json records")
    v0, v4 = lad["V0"], lad["V4"]
    walks = {}
    walks["V0"] = _reading_from_arrays(src, rungs["V0"], out_dir)
    walks["V0_no_tail"], _ = _reading_generated(src, replace(v0, tail_amp=0.0), "V0", "tail_amp 0")
    walks["V0_F9_phase"], _ = _reading_generated(src, replace(v0, tail_phase=phi9), "V0", f"tail_phase {phi9:.4f} (F9's)")
    walks["V0_F9"], _ = _reading_generated(src, replace(v0, tail_amp=amp9, tail_phase=phi9), "V0",
                                           f"tail_amp {amp9}, tail_phase {phi9:.4f} (F9's tail)")
    walks["V0_look_path"], gl = _reading_generated(src, replace(v0, tail_phase=0.5 * math.pi + 2 * math.pi), "V0",
                                                   "tail_phase pi/2 + 2 pi (today's, by the look path)")
    walks["V4"] = _reading_from_arrays(src, rungs["V4"], out_dir)
    walks["V4_no_tail"], _ = _reading_generated(src, replace(v4, tail_amp=0.0), "V4", "tail_amp 0")
    walks["V4_F9"], _ = _reading_generated(src, replace(v4, tail_amp=amp9, tail_phase=phi9), "V4",
                                           f"tail_amp {amp9}, tail_phase {phi9:.4f} (F9's tail alone)")
    for v in ("V5", "V5F13", "V6"):
        if v in rungs:
            walks[v] = _reading_from_arrays(src, rungs[v], out_dir)
    # the control: the look path at today's phase against today's code path (the same walk but for the path)
    arrs0 = load_arrays(rungs["V0"], out_dir)
    hinge = [src["M"].m.jnt_qposadr[statics.jid(src["M"].m, nm)] for nm in src["M"].names]
    control = dict(
        joint_max_diff_rad=float(np.max(np.abs(gl["q"][:, hinge] - np.asarray(arrs0["q"])[:, hinge]))),
        root_xyz_max_diff_mm=float(1e3 * np.max(np.abs(gl["q"][:, :3] - np.asarray(arrs0["q"])[:, :3]))),
        mz_abs_max_diff_nm=abs(walks["V0_look_path"]["single_support"]["mz_abs_max_nm"]
                               - walks["V0"]["single_support"]["mz_abs_max_nm"]),
        mu_req_max_diff=abs(walks["V0_look_path"]["single_support"]["mu_req_max"]
                            - walks["V0"]["single_support"]["mu_req_max"]))
    out = dict(body=body, speed=speed, f9_phase_rad=phi9, f9_amp_rad=amp9, today_phase_rad=0.5 * math.pi,
               today_amp_rad=float(v0.tail_amp), f9_phase_cycles_from_today=float(
                   ((phi9 - 0.5 * math.pi) / (2 * math.pi) + 0.5) % 1.0 - 0.5),
               f9_f13_phase_rad=trec.get("tail_phase_F9_F13_rad"), walks=walks, look_path_control=control,
               changes=dict(V0_phase_today_to_F9=_change(walks["V0"], walks["V0_F9_phase"]),
                            V0_today_to_F9_tail=_change(walks["V0"], walks["V0_F9"]),
                            V0_no_tail_to_today=_change(walks["V0_no_tail"], walks["V0"]),
                            V0_no_tail_to_F9_phase=_change(walks["V0_no_tail"], walks["V0_F9_phase"]),
                            V4_today_to_F9_tail=_change(walks["V4"], walks["V4_F9"]),
                            V4_no_tail_to_today=_change(walks["V4_no_tail"], walks["V4"]),
                            V4_no_tail_to_F9_tail=_change(walks["V4_no_tail"], walks["V4_F9"]),
                            V4_to_V5=_change(walks["V4"], walks["V5"])))
    sv0 = walks["V0"]["single_support"]
    out["today_adds_to_leg_yaw_moment"] = bool(sv0["mz_abs_max_nm"] > walks["V0_no_tail"]["single_support"]["mz_abs_max_nm"])
    out["f9_phase_lowers_peak_yaw_moment"] = bool(walks["V0_F9_phase"]["single_support"]["mz_abs_max_nm"]
                                                  < sv0["mz_abs_max_nm"])
    out["f9_phase_lowers_friction_need"] = bool(walks["V0_F9_phase"]["single_support"]["mu_req_max"] < sv0["mu_req_max"])
    if speed == "nominal":  # round two's recorded straight runs: today's nominal and brisk walks
        from ..stage1 import candidate_id, case_dir

        path = case_dir(body) / "review" / "screening" / f"{candidate_id(body)}_turning.json"
        doc = json.loads(path.read_text())
        brisk, _ = _reading_generated(src, sizing.BRISK, "sizing.BRISK", "today's brisk walk")
        out["record_check"] = dict(record=path.relative_to(look.STUDY).as_posix(),
                                   straight=_record_compare(walks["V0"], doc["runs"]["straight"]),
                                   straight_brisk=_record_compare(brisk, doc["runs"]["straight_brisk"]),
                                   record_spec=doc.get("spec"))
    out["seconds"] = time.perf_counter() - t0
    return look._plain(out)


# ------------------------------------------------------------------------------------------------
# Changes against V0, the over-limit list and the document
# ------------------------------------------------------------------------------------------------
#: The headline numbers compared with V0 (label -> path in a rung's entry).
HEADLINES = {
    "governing ratio": ("governing", 2),
    "knee speed": ("drives", "knee", "speed_ratio"),
    "knee cont": ("drives", "knee", "cont_ratio"),
    "hip pitch worst": ("drives", "hip_pitch", "worst"),
    "hip pitch speed": ("drives", "hip_pitch", "speed_ratio"),
    "hip roll worst": ("drives", "hip_roll", "worst"),
    "thigh twist worst": ("drives", "hip_yaw", "worst"),
    "ankle A worst": ("drives", "ankle_a", "worst"),
    "ankle B worst": ("drives", "ankle_b", "worst"),
    "neck cont": ("drives", "neck", "cont_ratio"),
    "neck cont with passive": ("passive", "neck", "cont_ratio_total"),
    "tail cont": ("drives", "tail", "cont_ratio"),
    "tail peak": ("drives", "tail", "peak_ratio"),
    "tail cont with passive": ("passive", "tail", "cont_ratio_total"),
    "tail peak with passive": ("passive", "tail", "peak_ratio_total"),
    "neck peak with passive": ("passive", "neck", "peak_ratio_total"),
    "tail power, Stage 0 (W)": ("passive", "tail", "power_mean_id_w"),
    "tail RMS with passive (N m)": ("passive", "tail", "rms_total_nm"),
    "single-support CoP margin (mm)": ("contact", "single_support_cop_margin_min_mm"),
    "ZMP margin (mm)": ("contact", "zmp_margin_min_mm"),
    "friction need mu_req": ("friction", "single_support", "mu_req_max"),
    "stance-pad peak |Mz| (N m)": ("friction", "single_support", "mz_abs_max_nm"),
    "double-support min mu": ("friction", "double_support", "min_mu_max"),
    "sway Stage 0 (H)": ("sway", "stage0_pp_H"),
    "sway CoM per stride (H)": ("sway", "com_pp_per_stride_max_H"),
    "CoM horizontal accel (m/s^2)": ("accel", "com_horizontal_peak"),
    "CoM vertical accel max (m/s^2)": ("accel", "com_vertical_max"),
    "CoM vertical accel min (m/s^2)": ("accel", "com_vertical_min"),
    "foot vertical accel (m/s^2)": ("accel", "foot_vertical_peak"),
    "mid-swing clearance (H)": ("clearance", "mid_swing_min_frac_H"),
    "tightest joint margin (rad)": ("clearance", "tightest_joint", 1),
}


def _get(d, path):
    for p in path:
        if d is None:
            return None
        d = d[p]
    return d


def changes(entry: dict, v0: dict) -> dict:
    out = {}
    for label, path in HEADLINES.items():
        a, b = _get(v0, path), _get(entry, path)
        out[label] = [a, b, None if a is None or b is None else b - a]
    return out


def over_limit(rungs: dict) -> list:
    """Every rung over a Stage 0 limit, with the plan's rule's consequence and the cause found."""
    out = []
    for body, d in rungs.items():
        for speed, ents in d.items():
            for v in ORDER:
                r = ents.get(v)
                if r is None or r["within_limits"]:
                    continue
                out.append(dict(body=body, speed=speed, variant=v, features=r["features"], over=r["over"],
                                rung=r["rung"], cause=r["cause"]))
    return out


def assemble(tails: dict, rungs: dict, extra: dict) -> dict:
    for d in rungs.values():
        for ents in d.values():
            for r in ents.values():
                r["vs_V0"] = changes(r, ents["V0"])
    near, passive_over, passive_near = [], [], []
    for body, d in rungs.items():
        for speed, ents in d.items():
            for v in ORDER:
                r = ents.get(v)
                if r is None:
                    continue
                if r["near"]:
                    near.append(dict(body=body, speed=speed, variant=v, near=r["near"]))
                if r["over_with_passive"]:
                    passive_over.append(dict(body=body, speed=speed, variant=v, over=r["over_with_passive"]))
                if r["near_with_passive"]:
                    passive_near.append(dict(body=body, speed=speed, variant=v, near=r["near_with_passive"]))
    return dict(extra, sources=SOURCES, tail_phase=tails, rungs=rungs, over_limit=over_limit(rungs), near_limit=near,
                over_with_passive=passive_over, near_with_passive=passive_near)


# ------------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------------
#: Gate G2's references: Stage 1's walks of each body, regenerated as ``stage1.refs.prepare`` builds them.
G2_NAMES = ("nominal", "brisk", "fr05")


def g2(body: str) -> dict:
    """Gate G2 on the code that runs the screen: Stage 1's nominal, brisk and fr05 references of ``body`` regenerated
    (``look.g2_hashes``: ``stage1.refs.generate`` at r2 and ``refs.write``'s npz; one BLAS thread when run from the
    command line) against the sha256 ``stage1/runs/2026-11-08/<body>/refs.json`` records."""
    t0 = time.perf_counter()
    rec = json.loads((look.STUDY / "stage1" / "runs" / "2026-11-08" / body / "refs.json").read_text())
    with look.accuracy():  # refs.generate sets round two's accuracy for the process; restored after
        got = look.g2_hashes(body, G2_NAMES)
    refs = {nm: dict(sha256=got[nm]["sha256"], recorded=rec["references"][nm]["sha256"],
                     bit_identical=got[nm]["sha256"] == rec["references"][nm]["sha256"],
                     record_leaves_match=got[nm]["matches_records"]) for nm in G2_NAMES}
    return dict(record=f"stage1/runs/2026-11-08/{body}/refs.json", accuracy=rec["accuracy"], references=refs,
                passes=all(r["bit_identical"] and r["record_leaves_match"] for r in refs.values()),
                blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"), seconds=time.perf_counter() - t0)


def _job(args):
    kind, body, speed, variant, out_dir = args
    out_dir = Path(out_dir)
    if kind == "g2":
        return kind, body, speed, variant, g2(body)
    src = look._source_cached(body)
    if kind == "tail":
        return kind, body, speed, variant, tail_phase(src, speed, out_dir)
    ladder_doc, _ = _docs(out_dir)
    return kind, body, speed, variant, screen_rung(src, speed, variant, ladder_doc["bodies"][body][speed][variant],
                                                   out_dir)


def _partial(out_dir: Path, kind: str, body: str, speed: str, variant: str | None) -> Path:
    return Path(out_dir) / "partial" / "screen" / f"{kind}__{body}__{speed}__{variant or '-'}.json"


def run(out_dir: Path = OUT, procs: int = 1, resume: bool = False, bodies=look.BODIES, speeds=look.SPEEDS) -> dict:
    """The tail-phase check and every rung's screen (in parallel over ``procs`` processes), then ``screen.json`` and
    ``SCREEN.md``; each job's result is kept in ``<out>/partial/screen/`` until the end (``resume`` keeps them)."""
    import multiprocessing as mp

    from ..stage1 import plant

    out_dir = Path(out_dir)
    prov = plant.provenance()
    t0 = time.perf_counter()
    refs_sha, tail_sha = _sha(out_dir / "look_refs.json"), _sha(out_dir / "tail_phase.json")
    ladder_doc, _ = _docs(out_dir)
    jobs = [("g2", b, "-", None, str(out_dir)) for b in bodies]
    jobs += [("tail", b, s, None, str(out_dir)) for b in bodies for s in speeds]
    jobs += [("rung", b, s, v, str(out_dir)) for b in bodies for s in speeds for v in ORDER
             if v in ladder_doc["bodies"][b][s]]
    # the long ones first: G2, the tail checks, then the many-stride windows
    jobs.sort(key=lambda j: ({"g2": 0, "tail": 1}.get(j[0], 2),
                             -(ladder_doc["bodies"][j[1]][j[2]][j[3]]["frames"] if j[3] else 0)))
    results = {}
    todo = []
    for j in jobs:
        p = _partial(out_dir, *j[:4])
        if resume and p.exists():
            r = json.loads(p.read_text())
            if r.get("inputs") == [refs_sha, tail_sha]:
                results[tuple(j[:4])] = r["result"]
                continue
        todo.append(j)
    for b in bodies:  # build each body once before forking
        look._source_cached(b)

    def keep(res):
        kind, body, speed, variant, r = res
        results[(kind, body, speed, variant)] = r
        p = _partial(out_dir, kind, body, speed, variant)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(inputs=[refs_sha, tail_sha], result=r)))
        tmp.replace(p)
        print(f"  {kind} {body} {speed} {variant or ''}: {r['seconds']:.0f} s", flush=True)

    if procs > 1 and todo:
        with mp.get_context("fork").Pool(procs) as pool:
            for res in pool.imap_unordered(_job, todo, chunksize=1):
                keep(res)
    else:
        for j in todo:
            keep(_job(j))
    g2s = {b: results[("g2", b, "-", None)] for b in bodies}
    tails = {b: {s: results[("tail", b, s, None)] for s in speeds} for b in bodies}
    rungs = {b: {s: {v: results[("rung", b, s, v)] for v in ORDER if ("rung", b, s, v) in results} for s in speeds}
             for b in bodies}
    extra = dict(
        what="Stage 2 Track A: the Stage 0 screen of the look ladder, before any render (STAGE2_PLAN 4 A.2-A.3; "
             "research_look 4.1 item 2)",
        note="prescribed references with perfect tracking on the round-two models; nothing adopted (P24)",
        method=(__doc__ or "").split("\n\n")[0].replace("\n", " "),
        inputs=dict(look_refs_json_sha256=refs_sha, tail_phase_json_sha256=tail_sha,
                    arrays_checked=sum(len(e) for d in rungs.values() for e in d.values())),
        speeds={s: SPEED_LABEL[s] for s in speeds},
        v040_rule=look.VALUES["v040"]["rule"],
        v040={b: {k: rungs[b]["v040"]["V0"]["gait"][k] for k in ("speed_m_s", "period_s", "duty", "stride_rel")}
              | dict(lift_frac=ladder_doc["bodies"][b]["v040"]["V0"]["spec"]["lift_frac"],
                     duty_with_F3=rungs[b]["v040"]["V2"]["gait"]["duty"]) | v040_between(look._source_cached(b)["H"])
              for b in bodies},
        limits={b: limits(look._source_cached(b)) for b in bodies},
        plant_joint_friction={b: plant_joint_friction(b) for b in bodies},
        margins=dict(peak=sizing.PEAK_MARGIN, speed=sizing.SPEED_MARGIN, cont=sizing.CONT_MARGIN, near_band=NEAR_BAND),
        grips=list(GRIPS), specs=dict(sway_H=SWAY_SPEC_H, trunk_roll_rms_deg=ROLL_SPEC_RMS_DEG,
                                       start_stop_accel_ms2=ACCEL_SPEC),
        limit_rule=look.LIMIT_RULE, g2=g2s, provenance=prov, versions=look._versions(), procs=procs)
    doc = assemble(tails, rungs, extra)
    doc["seconds"] = time.perf_counter() - t0
    path = out_dir / SCREEN_JSON
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(look._plain(doc), indent=1))
    tmp.replace(path)
    (out_dir / SCREEN_MD).write_text(markdown(doc))
    for j in jobs:
        _partial(out_dir, *j[:4]).unlink(missing_ok=True)
    for p in (out_dir / "partial" / "screen", out_dir / "partial"):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    return doc


# ------------------------------------------------------------------------------------------------
# SCREEN.md
# ------------------------------------------------------------------------------------------------
GROUP_LABEL = {"hip_roll": "hip roll", "hip_pitch": "hip pitch", "hip_yaw": "thigh twist", "knee": "knee",
               "ankle_a": "ankle A", "ankle_b": "ankle B", "neck": "neck", "tail": "tail"}
KIND_LETTER = {"cont": "c", "peak": "p", "speed": "s"}
BODY_LABEL = {"60-N-T-10": "60-N-T-10 (60 cm Tapered)", "75-M-H-10": "75-M-H-10 (75 cm Hybrid)"}
SPEED_SHORT = {"nominal": "nominal (Fr 0.012)", "v040": "0.40 m/s", "fr05": "animal pace (Fr 0.05)"}


def _f(x, nd: int = 3, signed: bool = False) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return "nan"
        return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return str(x)


def _pct(x, nd: int = 1) -> str:
    return "-" if x is None else f"{100 * x:+.{nd}f} %"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    def cells(r):
        return " | ".join(str(c).replace("|", "\\|") for c in r)  # a pipe in a cell (|Mz|) would split it

    out = ["| " + cells(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + cells(r) + " |" for r in rows]
    return out


def _worst_kind(d: dict) -> str:
    k = max(KINDS, key=lambda x: d[x])
    return f"{d[k]:.3f} {KIND_LETTER[k.replace('_ratio', '')]}"


def _gov(r: dict) -> str:
    g, k, v = r["governing"]
    return f"{GROUP_LABEL[g]} {k} {v:.3f}"


def _range(vals, nd=3, pct=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    lo, hi = min(vals), max(vals)
    if pct:
        return f"{100 * lo:+.1f} to {100 * hi:+.1f} %" if abs(hi - lo) > 5e-4 else f"{100 * lo:+.1f} %"
    return f"{lo:.{nd}f}-{hi:.{nd}f}" if abs(hi - lo) >= 0.5 * 10**-nd else f"{lo:.{nd}f}"


def _status(r: dict) -> str:
    if not r["within_limits"]:
        return "**over**: " + "; ".join(f"{o['limit']} {o['value']:.3f}" if isinstance(o["value"], float)
                                        else o["limit"] for o in r["over"])
    if r["near"]:
        return "near: " + "; ".join(f"{o['limit']} {o['value']:.3f}" for o in r["near"])
    return "within"


def _tri(vals, nd: int = 3, pct: bool = False) -> str:
    """Values at the three speeds (nominal / 0.40 m/s / animal pace) as 'a / b / c'."""
    if pct:
        return " / ".join("-" if v is None else f"{100 * v:+.1f}" for v in vals) + " %"
    return " / ".join(_f(v, nd) for v in vals)


TRI = "(nominal / 0.40 m/s / animal pace)"


def _brief(doc: dict) -> list[str]:
    tp, rungs = doc["tail_phase"], doc["rungs"]
    L = []
    # the tail phase
    per = []
    for b, d in tp.items():
        def ch(key, q, d=d):
            return [d[s]["changes"][key][q]["rel"] for s in d]

        per.append(f"{b}: today's tail over no tail, peak {_tri(ch('V0_no_tail_to_today', 'mz_abs_max_nm'), pct=True)}, "
                   f"mu_req {_tri(ch('V0_no_tail_to_today', 'mu_req_max'), pct=True)}; F9's phase instead of today's on "
                   f"the same walk, peak {_tri(ch('V0_phase_today_to_F9', 'mz_abs_max_nm'), pct=True)}, mu_req "
                   f"{_tri(ch('V0_phase_today_to_F9', 'mu_req_max'), pct=True)}; on the look walk (V4 to V4 with F9's "
                   f"tail), peak {_tri(ch('V4_today_to_F9_tail', 'mz_abs_max_nm'), pct=True)}, mu_req "
                   f"{_tri(ch('V4_today_to_F9_tail', 'mu_req_max'), pct=True)}")
    rc = [x for d in tp.values() for s in d.values() if s.get("record_check") for x in
          (s["record_check"]["straight"]["friction_max_rel_diff"],
           s["record_check"]["straight_brisk"]["friction_max_rel_diff"])]
    frame_rule = [b for b, d in tp.items() for s in d.values() if s.get("record_check") for k in ("straight", "straight_brisk")
                  if s["record_check"][k]["frame_rule_note"]]
    adds_all = all(s["today_adds_to_leg_yaw_moment"] for d in tp.values() for s in d.values())
    lowers_all = all(s["f9_phase_lowers_peak_yaw_moment"] for d in tp.values() for s in d.values())
    lowers_mu = [f"{b} {SPEED_SHORT[s]}" for b, d in tp.items() for s, x in d.items()
                 if not x["f9_phase_lowers_friction_need"]]
    L.append(f"- **Tail phase (research Bottom line 4): {'confirmed' if adds_all and lowers_all else 'see section 2'}.** "
             "Round two's straight-run reading of the stance pad's peak single-support yaw moment |Mz| and friction need "
             f"mu_req, changes {TRI}: " + "; ".join(per) + ". "
             + ("F9's phase lowers the friction need on every V0 walk. " if not lowers_mu else
                f"F9's phase does not lower the friction need on V0 at {', '.join(lowers_mu)}. ")
             + f"The reading reproduces the friction need of round two's recorded straight runs (nominal and brisk, "
               f"both bodies) to {max(rc):.1e} relative"
             + (f" (on {', '.join(sorted(set(frame_rule)))} one touchdown frame is classified differently by the two "
                "generators, which moves a minimum such as r_cop_min, not the friction need: section 2)."
                if frame_rule else "."))
    # over the limits
    ov = doc["over_limit"]
    if ov:
        items = []
        for o in ov:
            c = next(iter((o["cause"] or {}).values()), None)
            why = ""
            if c:
                why = (f"; the knee's peak comes in {c['state']} at {c['swing_frac'] or 0:.2f} of it, on the shortest "
                       f"swing ({c['fastest_swing_s']:.3f} s; swings {c['swing_s_range'][0]:.3f}-"
                       f"{c['swing_s_range'][1]:.3f} s; correlation of swing time and peak speed "
                       f"{_f(c['swing_vs_peak_corr'], 2)})") if c["joint"] == "knee" else ""
            items.append(f"{o['body']} {SPEED_SHORT[o['speed']]} {o['variant']} ("
                         + "; ".join(f"{x['limit']} {x['value']:.3f}" if isinstance(x["value"], float) else x["limit"]
                                     for x in o["over"]) + why + ")")
        L.append("- **Over a Stage 0 limit:** " + "; ".join(items) + ". Under the plan's rule (4 A.3) each keeps its "
                 "features and loses that speed rung unless the owner decides otherwise (recorded per rung, "
                 "section 5).")
    else:
        L.append("- **Over a Stage 0 limit:** none.")
    nr = doc["near_limit"]
    L.append("- **Within 0.02 of a limit:** " + ("; ".join(
        f"{n['body']} {SPEED_SHORT[n['speed']]} {n['variant']} ("
        + ", ".join(f"{x['limit']} {x['value']:.3f}" for x in n["near"]) + ")" for n in nr) if nr else "none") + ".")
    po = doc["over_with_passive"]
    groups = {}
    for p in po:
        for x in p["over"]:
            key = (p["body"], p["speed"], x["limit"].split(" with")[0])
            groups.setdefault(key, []).append((p["variant"], x["value"]))
    L.append("- **Over a planning limit once the plant's joint damping and friction are added** (research Bottom line "
             "3; outside Stage 0's rule, so no rung is lost; section 3b): " + ("; ".join(
                 f"{b} {SPEED_SHORT[s]} {lim} on {', '.join(v for v, _ in vs)} (up to {max(x for _, x in vs):.3f})"
                 for (b, s, lim), vs in groups.items()) if groups else "none") + ".")
    # the neck and tail with the plant's damping and friction
    pas = []
    for b, d in rungs.items():
        def vals(v, grp, key, d=d):
            return [e[v]["passive"][grp][key] if v in e else None for e in d.values()]

        neck14 = [max(e[v]["passive"]["neck"]["peak_ratio_total"] for v in ("V1", "V2", "V3", "V4")) for e in d.values()]
        pas.append(f"{b}: tail cont {_tri(vals('V0', 'tail', 'cont_ratio_total'))} on V0 (Stage 0 alone "
                   f"{_tri(vals('V0', 'tail', 'cont_ratio_id'))}; servo power {_tri(vals('V0', 'tail', 'power_mean_id_w'))} "
                   f"W) and {_tri(vals('V5', 'tail', 'cont_ratio_total'))} with F9's phase on V5 (power "
                   f"{_tri(vals('V5', 'tail', 'power_mean_id_w'))} W); neck peak {_tri(vals('V0', 'neck', 'peak_ratio_total'))} "
                   f"on V0, up to {_tri(neck14)} on V1-V4 (F1/F2's bob with today's nod) and "
                   f"{_tri(vals('V5', 'neck', 'peak_ratio_total'))} on V5 (F10's stabilised head)")
    L.append(f"- **Neck and tail with the plant's joint damping and friction** {TRI} (research Bottom line 3; Stage 0 "
             "leaves them out, so they cost no rung): " + "; ".join(pas) + ". With today's phase the servo drives the "
             "tail (positive power) and the damping adds to its load; with F9's the body's sway drives it and the servo "
             "brakes, which the damping helps.")
    # each feature's step on the friction need and the CoM's vertical acceleration
    steps = (("F1/F2", "V0", "V1"), ("F3", "V1", "V2"), ("F4/F5", "V2", "V3"), ("F6", "V3", "V4"),
             ("F9/F10", "V4", "V5"), ("F14", "V5", "V6"))
    fs = []
    for b, d in rungs.items():
        parts_ = []
        for lab, x, y in steps:
            rel = [e[y]["friction"]["single_support"]["mu_req_max"] / e[x]["friction"]["single_support"]["mu_req_max"] - 1
                   for e in d.values()]
            parts_.append(f"{lab} {_tri(rel, pct=True)}")
        fs.append(f"{b}: " + ", ".join(parts_))
    L.append(f"- **Each feature's step on the friction need** (mu_req, relative change {TRI}): " + "; ".join(fs)
             + ". F3 changes nothing at the animal pace (its swing time is that walk's own) and at the nominal walk "
               "shortens the swing to 0.32-0.36 s, which accelerates the swing leg harder at lift-off: the stance pad's "
               "peak yaw moment rises with it.")
    az = []
    for b, d in rungs.items():
        def up(v, d=d):
            return [e[v]["accel"]["com_vertical_max"] for e in d.values()]

        az.append(f"{b}: V0 {_tri(up('V0'), 2)}, V1 {_tri(up('V1'), 2)} (F2's 0.1 s drop), V3 {_tri(up('V3'), 2)} "
                  f"(F5's short roll), V6 {_tri(up('V6'), 2)}")
    L.append(f"- **The CoM's upward acceleration peak** (m/s^2 {TRI}; the stance legs carry it): " + "; ".join(az) + ".")
    # grip
    g54 = [(b, s, v) for b, d in rungs.items() for s, e in d.items() for v in ORDER if v in e
           and not e[v]["friction"]["grips"]["0.54"]]
    g90 = [(b, s, v) for b, d in rungs.items() for s, e in d.items() for v in ORDER if v in e
           and not e[v]["friction"]["grips"]["0.9"]]

    def where(lst):
        by = {}
        for b, s, v in lst:
            by.setdefault((b, s), []).append(v)
        return "; ".join(f"{b} {SPEED_SHORT[s]}: {', '.join(vs)}" for (b, s), vs in by.items()) or "none"

    L.append(f"- **Grip (F21's friction need on the straight walk, against the plan's mu 0.54 and 0.9):** over 0.9: "
             f"{where(g90)}. Over 0.54: {where(g54)}.")
    # V0 to V6
    parts = []
    for b, d in rungs.items():
        def pair(label, nd=3, d=d):
            v0 = [e["V6"]["vs_V0"][label][0] for e in d.values()]
            v6 = [e["V6"]["vs_V0"][label][1] for e in d.values()]
            return f"{_tri(v0, nd)} to {_tri(v6, nd)}"

        parts.append(f"{b}: governing {pair('governing ratio')}; knee speed {pair('knee speed')}; hip pitch "
                     f"{pair('hip pitch worst')}; thigh twist {pair('thigh twist worst')}; tail cont with passive "
                     f"{pair('tail cont with passive')}; neck peak with passive {pair('neck peak with passive')}; mu_req "
                     f"{pair('friction need mu_req')}; sway {pair('sway Stage 0 (H)')} H; CoM upward accel "
                     f"{pair('CoM vertical accel max (m/s^2)', 2)} m/s^2")
    L.append(f"- **V0 to V6** {TRI} (section 6 for every number): " + "; ".join(parts) + ".")
    return L


def markdown(doc: dict) -> str:
    """SCREEN.md from screen.json's document: every number comes from it."""
    tp, rungs, lim = doc["tail_phase"], doc["rungs"], doc["limits"]
    prov = doc["provenance"]["git"]
    L = ["# Stage 2 look ladder: the Stage 0 screen (before any render)", ""]
    L += [f"Numbers from `python -m designlab.stage2.screen run` (`designlab/stage2/screen.py`) at commit "
          f"`{(prov.get('commit') or '?')[:12]}`" + (" (clean tree)" if prov.get("dirty") is False else
                                                     f" (dirty: {', '.join(prov.get('dirty_paths') or [])})")
          + ", in `stage2/look/screen.json`, each block under the source function its `sources` names; this page is "
          "rendered from that file alone (`screen.markdown`; `python -m designlab.stage2.screen md`). "
          "Simulation only: prescribed references with perfect tracking on the round-two models (STAGE2_PLAN 4 A.2-A.3; "
          "research_look 4.1 item 2). Nothing is adopted (P24).", ""]
    L += [f"Inputs: `look_refs.json` (sha256 `{doc['inputs']['look_refs_json_sha256'][:12]}...`) and `tail_phase.json` "
          f"(`{doc['inputs']['tail_phase_json_sha256'][:12]}...`) as committed; the {doc['inputs']['arrays_checked']} "
          "rungs' arrays (git-ignored, `stage2/look/refs/`) each checked against its recorded sha256. Ratios are against "
          f"the record's drives with the sizing margins: cont = RMS / sustained (A1), peak = {doc['margins']['peak']} x "
          f"peak / planning peak, speed = {doc['margins']['speed']} x speed / speed at the bus; over 1 is over the limit. "
          "Kind letters: c cont, p peak, s speed.", ""]
    L += ["## In brief", ""] + _brief(doc) + [""]

    # 1. method and limits
    L += ["## 1. Limits, speeds and method", ""]
    bodies = list(lim)
    head = ["Drive group"] + [f"{b}: part, sustained / planning peak (N m), speed (rad/s)" for b in bodies]
    rows = []
    for g in sizing.GROUPS:
        rows.append([GROUP_LABEL[g]] + [f"{lim[b][g]['actuator']}: {lim[b][g]['sustained_nm']:.3f} / "
                                        f"{lim[b][g]['peak_nm']:.3f}, {lim[b][g]['speed_rad_s']:.2f}"
                                        + (f" at {lim[b][g]['bus_v']:g} V" if lim[b][g]["bus_v"] else "")
                                        for b in bodies])
    L += _table(head, rows) + [""]
    pj = doc["plant_joint_friction"]
    L += ["The plant's MJCF gives the neck pitch and tail yaw " + "; ".join(
        f"{b}: damping {pj[b]['neck_pitch']['damping']:g} / {pj[b]['tail_yaw']['damping']:g} N m s/rad, friction loss "
        f"{pj[b]['neck_pitch']['frictionloss']:g} / {pj[b]['tail_yaw']['frictionloss']:g} N m" for b in bodies)
        + " (neck / tail); Stage 0's inverse dynamics leave them out, the `passive` columns add them.", ""]
    L += ["**0.40 m/s** (the agility bar's speed) is defined from the existing rules: " + doc["v040_rule"]
          + " (`look.speed_spec`, documented in `look.py`).", ""]
    rows = [[b, _f(v["speed_m_s"], 3), _f(v["period_s"], 3), _f(v["stride_rel"], 3), _f(v["duty"], 3),
             _f(v["lift_frac"], 3), _f(v["duty_with_F3"], 3), _f(v.get("weight_w"), 3),
             (f"{v['brisk']['speed_m_s']:.3f} m/s, T {v['brisk']['period_s']:.3f} s, duty {v['brisk']['duty']:.2f} / "
              f"{v['fr05']['speed_m_s']:.3f} m/s, T {v['fr05']['period_s']:.3f} s, duty {v['fr05']['duty']:.2f}")
             if "brisk" in v else "-"] for b, v in doc["v040"].items()]
    L += _table(["Body", "Speed (m/s)", "Period (s)", "Stride (H)", "Duty", "Lift (H)", "Duty with F3 (V2+)",
                 "Weight w", "Brisk / animal pace (the two walks mixed)"], rows)
    L += ["", "Method, per rung (`screen.screen_rung`; full list in screen.json `sources`): `look.price` re-run on the "
          "arrays (`look.look_inverse_dynamics`, i.e. `gait.inverse_dynamics` with F5's rolling contact, then "
          "`sizing._motor_space` and `envelope._ratios`; equal to the price `look_refs.json` records), "
          "`gait.summarize`, the plant's joint damping and friction on the neck and tail (`screen.passive_loads`), "
          "round two's straight-run friction reading (`screen.straight_run`: `turning.inverse_dynamics` and "
          "`turning._foot_stats`), sway, CoM and foot accelerations, and the clearance screen's walking gates "
          "(`clearance.screen`, gates r2).", ""]

    # 2. the tail phase
    L += ["## 2. The tail-phase check (research Bottom line 4; plan 4 A.2)", "",
          "Round two's straight-run reading on each walk: the stance pad's free yaw moment about its CoP over single "
          "support (peak and RMS, N m), the friction need mu_req (F21's: limit surface with the pads' uniform-pressure "
          "radius) and mu_req_cop (the CoP-based radius), and the double-support requirement over the frames where no "
          "foot rolls (frames read / all). Tail: amplitude (rad) at phase (rad); today's is 0.12 at pi/2 = 1.571; F9's "
          "phase is the ladder's (the V4 minimiser of the peak, `tail_phase.json`).", ""]
    for b, d in tp.items():
        L += [f"### {BODY_LABEL.get(b, b)}", ""]
        rows = []
        for s, x in d.items():
            for w in TAIL_WALKS:
                if w not in x["walks"]:
                    continue
                r = x["walks"][w]
                ss, ds = r["single_support"], r["double_support"]
                rows.append([SPEED_SHORT[s], w, f"{r['tail']['amp_rad']:.3f} at {r['tail']['phase_rad']:+.3f}",
                             _f(ss["mz_abs_max_nm"]), _f(ss["mz_rms_nm"]), _f(ss["mu_req_max"]),
                             _f(ss["mu_req_cop_max"]), f"{_f(ds['min_mu_max'])} ({ds['read']}/{ds['frames']})"])
        L += _table(["Speed", "Walk", "Tail", "Peak |Mz|", "RMS |Mz|", "mu_req", "mu_req_cop", "DS min mu (read)"], rows)
        L += [""]
        rows = []
        for s, x in d.items():
            c = x["changes"]
            rows.append([SPEED_SHORT[s], f"{x['f9_phase_rad']:+.3f} ({x['f9_phase_cycles_from_today']:+.3f})",
                         f"{_pct(c['V0_no_tail_to_today']['mz_abs_max_nm']['rel'])} / "
                         f"{_pct(c['V0_no_tail_to_today']['mu_req_max']['rel'])}",
                         f"{_pct(c['V0_phase_today_to_F9']['mz_abs_max_nm']['rel'])} / "
                         f"{_pct(c['V0_phase_today_to_F9']['mu_req_max']['rel'])}",
                         f"{_pct(c['V0_today_to_F9_tail']['mz_abs_max_nm']['rel'])} / "
                         f"{_pct(c['V0_today_to_F9_tail']['mu_req_max']['rel'])}",
                         f"{_pct(c['V4_today_to_F9_tail']['mz_abs_max_nm']['rel'])} / "
                         f"{_pct(c['V4_today_to_F9_tail']['mu_req_max']['rel'])}",
                         f"{_pct(c['V4_no_tail_to_F9_tail']['mz_abs_max_nm']['rel'])} / "
                         f"{_pct(c['V4_no_tail_to_F9_tail']['mu_req_max']['rel'])}",
                         f"{_pct(c['V4_to_V5']['mz_abs_max_nm']['rel'])} / {_pct(c['V4_to_V5']['mu_req_max']['rel'])}"])
        L += _table(["Speed", "F9 phase (cycles from today)", "V0: no tail to today (peak / mu)",
                     "V0: today to F9's phase", "V0: today to F9's tail", "V4: today to F9's tail",
                     "V4: no tail to F9's tail", "V4 to V5 (F9 + F10)"], rows)
        x0 = d.get("nominal", {})
        if x0.get("record_check"):
            rc = x0["record_check"]
            def _diffs(c):
                return "; ".join(f"{x['side']} {x['field']} {x['mine']:.6g} against {x['record']:.6g}" for x in c["differing"])

            L += ["", f"Record check (`{rc['record']}`): V0 at the nominal walk against `runs.straight` "
                  f"({rc['straight']['fields']} fields; friction fields to {rc['straight']['friction_max_rel_diff']:.1e}, "
                  f"every field to {rc['straight']['all_max_rel_diff']:.1e}; mu_req {rc['straight']['mu_req_max'][0]:.4f} "
                  f"against {rc['straight']['mu_req_max'][1]:.4f}"
                  + (f"; differing: {_diffs(rc['straight'])} ({rc['straight']['frame_rule_note']})"
                     if rc["straight"]["differing"] else "") + ") and "
                  f"today's brisk walk against `runs.straight_brisk` (friction fields to "
                  f"{rc['straight_brisk']['friction_max_rel_diff']:.1e}; mu_req "
                  f"{rc['straight_brisk']['mu_req_max'][0]:.4f}"
                  + (f"; differing: {_diffs(rc['straight_brisk'])} ({rc['straight_brisk']['frame_rule_note']})"
                     if rc["straight_brisk"]["differing"] else "") + "). Control (today's phase through the look path): "
                  + "; ".join(f"{SPEED_SHORT[s]} joints {x['look_path_control']['joint_max_diff_rad']:.1e} rad, "
                              f"|Mz| {x['look_path_control']['mz_abs_max_diff_nm']:.1e} N m" for s, x in d.items()) + "."]
        L += [""]

    # 3. drives
    L += ["## 3. Drives: ratios against the planning limits (1.15 speed / 1.25 peak margins)", "",
          "Governing: the largest ratio. Each group: its worst ratio and kind. Neck and tail: Stage 0's (inverse "
          "dynamics) / with the plant's joint damping and friction. Status: the plan's rule (4 A.3); near = within "
          f"{doc['margins']['near_band']} of a limit.", ""]
    for b, d in rungs.items():
        for s, e in d.items():
            L += [f"### {b}, {SPEED_SHORT[s]}", ""]
            rows = []
            for v in ORDER:
                if v not in e:
                    continue
                r = e[v]
                dr, ps = r["drives"], r["passive"]
                rows.append([v, ", ".join(r["features"]) or "today", _gov(r), _f(dr["knee"]["speed_ratio"]),
                             _f(dr["knee"]["cont_ratio"]), _worst_kind(dr["hip_pitch"]), _worst_kind(dr["ankle_a"]),
                             _worst_kind(dr["ankle_b"]), _worst_kind(dr["hip_roll"]), _worst_kind(dr["hip_yaw"]),
                             f"{_f(ps['neck']['cont_ratio_id'])} / {_f(ps['neck']['cont_ratio_total'])}",
                             f"{_f(ps['tail']['cont_ratio_id'])} / {_f(ps['tail']['cont_ratio_total'])}",
                             f"{_f(ps['tail']['peak_ratio_id'])} / {_f(ps['tail']['peak_ratio_total'])}", _status(r)])
            L += _table(["Rung", "Features", "Governing", "Knee s", "Knee c", "Hip pitch", "Ankle A", "Ankle B",
                         "Hip roll", "Thigh twist", "Neck c", "Tail c", "Tail p", "Status"], rows) + [""]

    # 3b. neck and tail with the plant's damping and friction (research Bottom line 3)
    L += ["## 3b. Neck and tail with the plant's joint damping and friction (research Bottom line 3)", "",
          "Stage 0's inverse-dynamics torque (ID) against the servo's torque with the plant's damping and friction loss "
          "(total = ID + b omega + f sign omega). RMS and peak in N m; c and p: ratios to the sustained (A1) and planning "
          "peak (x 1.25). Power: the servo's mean power on the ID torque (W; negative: the body's motion drives the "
          "joint and the servo brakes it, so the damping helps).", ""]
    for b, d in rungs.items():
        for s, e in d.items():
            L += [f"### {b}, {SPEED_SHORT[s]}", ""]
            rows = []
            for v in ORDER:
                if v not in e:
                    continue
                n_, t_ = e[v]["passive"]["neck"], e[v]["passive"]["tail"]
                rows.append([v, f"{n_['rms_id_nm']:.3f} / {n_['rms_total_nm']:.3f}",
                             f"{n_['cont_ratio_id']:.3f} / {n_['cont_ratio_total']:.3f}",
                             f"{n_['peak_ratio_id']:.3f} / {n_['peak_ratio_total']:.3f}",
                             f"{t_['rms_id_nm']:.3f} / {t_['rms_passive_alone_nm']:.3f} / {t_['rms_total_nm']:.3f}",
                             f"{t_['cont_ratio_id']:.3f} / {t_['cont_ratio_total']:.3f}",
                             f"{t_['peak_ratio_id']:.3f} / {t_['peak_ratio_total']:.3f}",
                             f"{t_['power_mean_id_w']:+.3f}"])
            L += _table(["Rung", "Neck RMS ID / total", "Neck c", "Neck p", "Tail RMS ID / passive alone / total",
                         "Tail c", "Tail p", "Tail power (W)"], rows) + [""]

    # 4. balance, grip, sway, accelerations, clearance
    L += ["## 4. Balance, grip, sway, accelerations and the walking clearance gates", "",
          "SS CoP: the single-support CoP's margin inside the stance pad (mm; envelope rule over -10). ZMP: the "
          "multibody ZMP's margin inside the support (mm). mu_req and peak |Mz| (N m): the straight-run reading "
          "(section 2); DS: the double-support requirement. Sway (H, peak to peak; spec 0.12): Stage 0's (pendulum "
          "CoM) and the model CoM's largest per stride. Accelerations (m/s^2): the CoM's horizontal peak, forward "
          "part at 1 Hz, vertical range; the feet's vertical peak. Swing: the lowest pad point over the middle half of "
          "each swing (H; gate 0.04). Joint: the tightest margin inside the usable ranges (rad; gate 0.10).", ""]
    for b, d in rungs.items():
        for s, e in d.items():
            L += [f"### {b}, {SPEED_SHORT[s]}", ""]
            rows = []
            for v in ORDER:
                if v not in e:
                    continue
                r = e[v]
                c, f, sw, ac, cl = r["contact"], r["friction"], r["sway"], r["accel"], r["clearance"]
                rows.append([v, _f(c["single_support_cop_margin_min_mm"], 1), _f(c["zmp_margin_min_mm"], 1),
                             _f(f["single_support"]["mu_req_max"]), _f(f["single_support"]["mz_abs_max_nm"], 2),
                             _f(f["double_support"]["min_mu_max"]), _f(sw["stage0_pp_H"]),
                             _f(sw["com_pp_per_stride_max_H"]), _f(ac["com_horizontal_peak"], 2),
                             _f(ac["com_forward_lp1hz_peak"], 2),
                             f"{ac['com_vertical_min']:+.2f} / {ac['com_vertical_max']:+.2f}",
                             _f(ac["foot_vertical_peak"], 1), _f(cl["mid_swing_min_frac_H"], 4),
                             f"{_f(cl['tightest_joint'][1], 3)} ({cl['tightest_joint'][0]})",
                             "pass" if cl["walking_gates"] else "**fail**"])
            L += _table(["Rung", "SS CoP", "ZMP", "mu_req", "Peak |Mz|", "DS mu", "Sway S0", "Sway CoM",
                         "a horiz", "a fwd 1 Hz", "a vert", "a foot", "Swing (H)", "Joint", "Gates"], rows) + [""]

    # 5. over, near and the plan's rule
    L += ["## 5. Over-limit and near-limit rungs: the plan's rule", "", f"Rule: {doc['limit_rule']}.", ""]
    if doc["over_limit"]:
        for o in doc["over_limit"]:
            L += [f"- **{o['body']}, {SPEED_SHORT[o['speed']]}, {o['variant']}** (features {', '.join(o['features'])}): "
                  f"{o['rung']}."]
            for lim_name, c in (o["cause"] or {}).items():
                L += [f"  - Cause ({lim_name}): the peak, {c['peak_rad_s']:.3f} rad/s, comes on the {c['side']} "
                      f"{c['joint']} in {c['state']}" + (f" at {c['swing_frac']:.2f} of the swing" if c["swing_frac"]
                                                         is not None else "")
                      + f"; over the window's {c['swings']} swings (durations {c['swing_s_range'][0]:.3f}-"
                      f"{c['swing_s_range'][1]:.3f} s) the {c['joint']}'s peak speed runs {c['peak_rad_s_range'][0]:.3f}-"
                      f"{c['peak_rad_s_range'][1]:.3f} rad/s, correlation with the swing's duration "
                      f"{_f(c['swing_vs_peak_corr'], 3)}: the fastest swing is the shortest ({c['fastest_swing_s']:.3f} s)."]
    else:
        L += ["- None."]
    L += [""]
    for n in doc["near_limit"]:
        L += [f"- Near ({n['body']}, {SPEED_SHORT[n['speed']]}, {n['variant']}): "
              + ", ".join(f"{x['limit']} {x['value']:.3f}" for x in n["near"]) + "."]
    for key, word in (("over_with_passive", "Over"), ("near_with_passive", "Near")):
        groups = {}
        for p in doc.get(key, []):
            for x in p["over" if key == "over_with_passive" else "near"]:
                groups.setdefault((p["body"], p["speed"], x["limit"].split(" with")[0]), []).append(
                    f"{p['variant']} {x['value']:.3f}")
        for (b, s, lim), vs in groups.items():
            L += [f"- {word} with the plant's damping and friction ({b}, {SPEED_SHORT[s]}, {lim}): {', '.join(vs)}."]
    if doc["over_with_passive"] or doc.get("near_with_passive"):
        L += ["", "The passive-inclusive loads are outside Stage 0's rule (round two's screens left the joint damping and "
              "friction out), so they cost no rung; the plan's reading 3, 'shown but for A1', and Track D's levers 2 (an "
              "STS3250 tail) and 4 (damping and friction x0.25) are for them. They are recorded for the owner."]
    L += [""]

    # 6. V0 to V6
    L += ["## 6. Changes from V0 to V6 (and to every rung: screen.json `vs_V0`)", "",
          "V6 carries every feature of the ladder (F1-F6, F9, F10, F14; F13 at 75 cm's animal pace). Each cell: V0, V6 "
          "(change).", ""]
    for b, d in rungs.items():
        L += [f"### {BODY_LABEL.get(b, b)}", ""]
        rows = []
        for label in HEADLINES:
            row = [label]
            for s, e in d.items():
                a, v, dlt = e["V6"]["vs_V0"][label]
                nd = 1 if "(mm)" in label or "foot" in label else (2 if "accel" in label or "|Mz|" in label else 3)
                row.append(f"{_f(a, nd)}, {_f(v, nd)} ({_f(dlt, nd, signed=True)})")
            rows.append(row)
        L += _table(["Number"] + [SPEED_SHORT[s] for s in d], rows) + [""]

    # 7. checks
    rep = [r["reprice"]["equal_to_recorded"] for d in rungs.values() for e in d.values() for r in e.values()]
    plain = [(b, s, v, r) for b, d in rungs.items() for s, e in d.items() for v, r in e.items() if r["plain_gait_id"]]
    diff_verdict = [(b, s, v) for b, s, v, r in plain if set(r["plain_gait_id"]["over"])
                    != {o["limit"] for o in r["over"] if o["limit"].split()[0] in sizing.GROUPS}]
    gov_plain = [r["plain_gait_id"]["governing"][2] - r["governing"][2] for _, _, _, r in plain]
    any_plain = max(((r["plain_gait_id"]["max_ratio_diff"], f"{b} {SPEED_SHORT[s]} {v}") for b, s, v, r in plain),
                    default=(None, None))
    worst_plain = max((abs(r["plain_gait_id"]["worst"][g] - r["drives"][g]["worst"]) for _, _, _, r in plain
                       for g in r["drives"]), default=None)
    L += ["## 7. Checks", ""]
    L += [f"- Re-price: on {sum(rep)} of {len(rep)} rungs `look.price` on the arrays equals the price `look_refs.json` "
          "records, number for number."]
    if doc.get("g2"):
        L += ["- G2 (gait.py with every look field at its default reproduces today's references; `screen.g2` on the code "
              "that ran the screen): " + "; ".join(
                  f"{b}: " + ", ".join(f"{nm} {'bit-identical' if x['bit_identical'] else 'DIFFERS'}"
                                       for nm, x in g["references"].items())
                  + (", record leaves match" if all(x["record_leaves_match"] for x in g["references"].values()) else
                     ", record leaves DIFFER") for b, g in doc["g2"].items())
              + " (against `stage1/runs/2026-11-08/<body>/refs.json`; one BLAS thread). The ten references of each body, "
                "turns included, are `tests/test_stage2_look.py`'s G2 test."]
    if plain:
        L += [f"- Round two's split on the F5 rungs (`gait.inverse_dynamics`, a rolling foot's CoP kept in its pad box, "
              f"{len(plain)} rungs): its governing ratio differs from the screen's by {min(gov_plain):+.3f} to "
              f"{max(gov_plain):+.3f} and no group's worst ratio by more than {worst_plain:.3f}; a single ratio below its "
              f"group's worst moves by up to {any_plain[0]:.3f} ({any_plain[1]})"
              + (f"; the verdict differs on {', '.join(' '.join(x) for x in diff_verdict)}" if diff_verdict else
                 "; no verdict changes") + ". The rolling contact matters frame by frame, not for the governing limits."]
    L += ["- No rolling foot in single support on any rung (the friction reading's single-support wrench is unique); the "
          "double-support friction LP is not read on rolling frames (counted in section 2's tables)."]
    L += [f"- Provenance: {json.dumps(doc['provenance']['git'])}; versions {json.dumps(doc['versions'])}; "
          f"{doc['procs']} processes, {doc['seconds'] / 60:.0f} min."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.screen", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="the tail-phase check and every rung's screen; screen.json and SCREEN.md")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    q = sub.add_parser("md", help="SCREEN.md again from screen.json")
    q.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(Path(a.out), a.procs, a.resume)
    else:
        doc = json.loads((Path(a.out) / SCREEN_JSON).read_text())
        (Path(a.out) / SCREEN_MD).write_text(markdown(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
