"""Stage 1 tracker: controllers R0, R1 and R2 (``STAGE1_PLAN.md`` section 2; section 3 step 5, gate G4).

* **R0**: the reference with feedforward and servo PD only.  Every 2 ms the servos get the reference joint angles
  and rates and the feedforward torque: the inverse-dynamics torque, the MJCF's passive joint terms (damping x qd_ref
  + frictionloss x tanh(qd_ref / 0.05)) and the plant's extra ankle armature x qdd_ref, exactly as G2's replay
  (``gates.replay``).
* **R1**: the tracker on the true (privileged) state: CoM, its velocity, the feet, the trunk's orientation and rate.
* **R2**: the tracker on the estimate (``estimator``: encoders, IMU with noise, stance-leg kinematics at the loaded
  pad's contact point, decision 19).

The controller's own model of the robot (``Tracker.model``) is the plant's unless one is given: the PPO envs randomise
the plant's trunk mass and give the tracker the nominal model (its mass, R2's estimator and its sensors' weight), a
deliberate model error (``env.MASS_RULE``).

Servo target (plan section 2), every physics step (500 Hz), in joint space and set in motor coordinates by
``plant.Plant.command``::

    ctrl = q_ref + dq + (kv qd_ref + tau_ff + dtau) / kp

tau_ff is R0's feedforward; dq and dtau come from the feedback.  The feedback is computed at 100 Hz (every
``FEEDBACK_EVERY`` steps) and reaches the servo targets through a first-order hold (``FEEDBACK_HOLD``: each update's
dq and dtau are approached linearly over the next 10 ms).  A zero-order hold steps the stiff ankle servos every 10 ms
and rocks the stance feet on their pads: on the nominal walk (60-N-T-10, R1) the loaded feet's contact points then
move at 11 / 8 / 19 mm/s RMS (x / y / z) against 5 / 5 / 5 mm/s with the ramp and 5 / 4 / 5 mm/s for R0, which
spoils the stance-leg odometry R2 relies on.

A speed command (``Tracker.rate``, ``COMMAND_RATE_RULE``): the tracker plays its track ``rate`` times faster than
real time (the tuning commands plan section 4's 5 % above the targets, rate 1.05); the reference rates, accelerations
and DCM follow the faster clock, the inverse-dynamics torque does not.  Rate 1 (G2, G4) is the tracker as built, bit
for bit.

Ankle torque cap (``STAGE1_PLAN.md`` decision 15; ``ANKLE_CAP``, ``cap_ankle_torque``): every physics step the
tracker limits each ankle's total commanded torque (feedforward, the servo's PD predicted from the encoders, true
angles and rates for R1 and the last reading for R2, and the feedback) to at most one motor's peak on each axis (the
screened model's per-axis box, inside the two-motor linkage's |tau_p| + |tau_r| <= 2 peak) and, on a loaded foot
(R1: carrying more than ``CONTACT_LOAD_FRAC`` of the weight; R2: in contact by its estimator's detection, whose load
readings are as noisy as that threshold), to what the stance pad can carry with its centre of pressure
``COP_INSET`` (5 mm) inside the pad (the CoP the torques imply under the foot's load, quasi-statically: ``ankle_cop``).
The torque kept is the feasible one nearest to the command.  More torque only rolls the foot; the cap limits the
controller, not the plant.  Every R1 and R2 episode records how far the realised ankle torques broke the cap on the
true state (``CAP_RESIDUAL_RULE``, ``Tracker.cap_residual``): R2 caps on its sensor readings.

The tunable gains are one named vector (``GAIN_SPEC``: name, tuning start, earlier default, CMA-ES bounds, note).  The
tuning start (decision 16, ``START_GAINS``, the tracker's default and G4's point, decision 17) is the design-study
prototype's terms (``track_proto.py``): ankle and step feedback and trunk PD at its values (k_dcm 2.0, cop 0.020 /
0.012 m, k_step 1.0, step 0.05 / 0.03 m, decay 1.0, trunk 40 / 2.0), the heading and phase-rate PI gains zero.  The
build's untuned defaults (``EARLIER_DEFAULTS``: heading 0.5 / 0.1, phase rate 2.0 / 1.0, chosen before any run) stay
available for the record (``GAIN_SETS``).

The feedback acts on the divergent component of motion xi = c + c' / omega0 (omega0 the references' sqrt(g / z_c)):
its error e against the reference, both relative to the anchor foot (the stance foot; in double support the foot that
landed last, before the first step the first stance foot; in a stand the midpoint of the feet) and in its frame:

* ankle: the ZMP is shifted by dp = clip(k_dcm e, +-(cop_x, cop_y)), the shifted ZMP kept ``COP_INSET`` (5 mm) inside
  the support (the stance pad polygon, in double support the hull of both), realised as ankle torque on each stance
  foot in proportion to the reference's normal forces (plan section 2: "DCM feedback drives the ankle CoP, 5 mm
  inside the pad"), and capped as above;
* step: the swing foot's touchdown moves by clip(k_step e) (+-step_x forward, step_out outward) through the swing
  leg's Jacobian, frozen after 80 % of the swing; the offset blends in over the swing (minimum jerk) and decays over
  the following stance (``decay``: e^-decay of it remains at lift-off).  Inward the bound is the leg-to-leg clearance
  (decision 20, ``inward_room``): the swing pad, at its reference touchdown pose relative to the stance foot moved by
  the offset, stays at least ``PAD_CLEARANCE`` (5 mm) from the stance pad.  (The turning screen's turn gap, the
  pelvis plate's clearance to the same-side thigh, ``turn_gap_info``, bounds nothing: decision 20.);
* trunk: roll and pitch PD (k_trunk, d_trunk) about the reference attitude through the stance hips;
* heading: a PI on the trunk heading error (averaged over the last stride and sampled at each lift-off; k_yaw is the
  fraction corrected per step) through the swing leg's hip yaw, which then decays in stance like the step offset;
* phase rate: a PI on the forward DCM error (k_phase, ki_phase) sets the reference's clock rate, within
  +-``PHASE_RATE_LIMIT`` (5 %); reference rates and accelerations are scaled with it;
* capture step: standing, if the DCM leaves the support polygon by more than ``capture_margin``, the foot on the
  side it leaves toward (the less loaded one for a push along the feet) steps so that its pad centre lands on the
  DCM's predicted position at touchdown (diverging from the support edge it crossed; for a push along the feet the foot
  keeps its place sideways), kept out of the other foot's way (``CAPTURE_*``); the body then stands on the new
  stance.

Falls are ``plant.Plant.fall_cause`` (trunk tilt over 0.6 rad, root below 0.7 x home height, a ``cl_*`` proxy below the
floor).  The fall's onset is the last time the tilt from home rose through ``FALL_ONSET_RAD`` (0.2 rad) before it: by
the 0.6 rad of the fall itself every drive is fighting it.  Every feedback update records which of plan section 4's
hardware limits were active (``HW_LIMITS``): a leg drive at its torque limit, a leg drive at a limit the envelope had
narrowed, the ankle linkage (both motors of an ankle at their limits), slip (a foot carrying more than
``SLIP_LOAD_FRAC`` of the weight sliding faster than ``SLIP_SPEED`` at one of its pad contacts that carries load: the
foot's velocity at the contact point, ``foot_slip_speed``, so that a foot rocking or turning on its pad is not counted),
the CoP at a pad edge with the ankle saturated (the cap holding a loaded foot's CoP at the pad's 5 mm inset, or the
feedback's ZMP clipped into the pad while an ankle motor is at its limit), the leg-to-leg clearance (the inward step
bound clipping the step, decision 20) and joint range.  Recorded beside them, not hardware limits
(``REPORTED_LIMITS``): neck and tail drives at their limits, the cap's per-axis box (the controller's own: the plant's
linkage reaches twice it on one axis), the step clips and the phase-rate limit.  The hardware limits active at any
update of the ``LIMIT_WINDOW_S`` (0.5 s, plan section 4) before a fall's onset are its candidates.

Attribution is by condition (decision 20, ``gates.attribute``): a condition's falls are body-limited by a hardware
limit only if lifting that limit from the start (``run``'s ``relax``, ``relaxed``: the leg drives' torque limits
lifted, the envelope off, the ankle motors' limits lifted, friction ``CF_FRICTION``, the inward bound lifted with the
feet's pads passing through each other, the hinges' ranges widened by ``CF_RANGE_RAD``) improves survival over the
condition's 20 starts significantly in a paired test; otherwise they are controller-limited.  A 1 mm/s null (the
same starts, nothing lifted) is reported beside the label as a chaos diagnostic.  The CoP at a pad edge cannot be
lifted in the plant (the pads' size).

The episode-level counterfactual of the build (``run``'s ``confirm``; ``EPISODE_CF_RULE``) stays as a diagnostic: the
episode is restored to its state ``CF_BRANCH_S`` (1 s) before the onset (``Tracker.save`` and ``Tracker.load``:
MuJoCo's integration state, every controller and estimator variable and the sensor-noise generator, so that the
continuation without any change repeats the fall bit for bit) and continued with each candidate relaxed until
``CF_HORIZON_S`` after the original fall, beside a null continuation with the root velocity changed by
``CF_NULL_M_S`` (1 mm/s).  It tests a fall's last second; decision 20 judges a condition over its starts instead.

Speed (decision 21: the faster tracker, a drop-in replacement whose every physics step's qpos, qvel, ctrl and
sensordata are bit for bit the earlier tracker's, ``designlab.stage1.replay``).  Each product with a matrix, each
solve and each reduction stays the numpy (BLAS, LAPACK) call it was, with the same operands: on this build BLAS fuses
multiplies and adds, so a Python formula would differ in the last bit.  What changed is the work around them: MuJoCo
arrays read through views kept per plant and estimator (``plant.MjViews``), the cap's and the ZMP clip's geometry
reused while the kinematics they come from have not changed, per-step buffers, and elementwise and two-element
arithmetic in Python floats (the same IEEE operations; ``_clip`` keeps ``np.clip``'s ties).  Decisions that only need
the side of a boundary (the ZMP inside its polygon, the capture trigger, the cap's box and pad, whether the inward bound
binds) are taken from a cheap bound when it clears the boundary by far more than any rounding (``CERTIFY_TOL``,
``INWARD_CERTIFY_M``), and from the exact computation otherwise.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from .. import statics
from . import case_dir, plant
from . import estimator as est_mod
from .refs import Reference
from .transitions import N_FRAMES, TRANSITIONS, Transition, stop_command_phase

_Views = est_mod.MjViews

try:  # the LAPACK gufunc behind np.linalg.solve (``_solve2``): numpy's private module, so used only where it exists
    from numpy.linalg import _umath_linalg as _np_umath_linalg

    _SOLVE1 = _np_umath_linalg.solve1
except (ImportError, AttributeError):  # pragma: no cover - a numpy without the private gufunc
    _SOLVE1 = None
#: The solve ``_solve2`` runs: numpy's gufunc directly, or ``np.linalg.solve`` where numpy has no such private module
#: (the same LAPACK call and results on the pinned numpy 2.5.4, only with numpy's argument checks; recorded with runs).
SOLVE2_PATH = "numpy.linalg._umath_linalg.solve1" if _SOLVE1 is not None else "numpy.linalg.solve"

FEEDBACK_EVERY = 5
#: How the 100 Hz feedback reaches the 500 Hz servo targets: "ramp" (first-order hold: each update's offsets are
#: reached linearly over the next ``FEEDBACK_EVERY`` steps) or "hold" (zero-order hold: steps every 10 ms).
FEEDBACK_HOLD = "ramp"
FRICTION_SMOOTH = 0.05
COP_INSET = 0.005
#: ``STAGE1_PLAN.md`` decision 20: the inward swing-foot bound, the leg-to-leg clearance: the swing pad stays at least
#: this far from the stance pad.
PAD_CLEARANCE = 0.005
#: Decision 15: the tracker's cap on each ankle's total torque command, "box" (each axis at most one motor's peak) and
#: "pad" (a loaded foot's CoP ``COP_INSET`` inside its pad); the decided cap first, the others for diagnostics.
ANKLE_CAP = ("box", "pad")
ANKLE_CAPS = (ANKLE_CAP, ("box",), ("pad",), ())
#: Decision 20: what bounds the swing foot's inward step; None (no bound) is the leg-to-leg clearance lifted.
INWARD_BOUNDS = ("pad_clearance", None)
PHASE_RATE_LIMIT = 0.05
SWING_FREEZE_U = 0.8
LIMIT_WINDOW_S = 0.5
SLIP_SPEED = 0.05  # m/s: a loaded stance foot sliding faster than this slips
SLIP_LOAD_FRAC = 0.2
#: A foot carrying more than this fraction of the weight is in contact: its ankle torque is CoP-limited.
CONTACT_LOAD_FRAC = 0.02
CAPTURE_SWING_S = 0.25  # s, the capture step's swing
CAPTURE_LIFT_H = 0.08  # x H
CAPTURE_MIN_WIDTH = 0.6  # x the home stance width, kept between the feet
G = 9.81
LEGS = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
TERMS = ("ankle", "step", "trunk", "heading", "phase", "capture")
MODES = ("R0", "R1", "R2")

#: The tracker's named parameter vector: (name, tuning start, earlier default, CMA-ES lower bound, upper bound, note).
#: The tuning start is ``STAGE1_PLAN.md`` decision 16: the design-study prototype's terms (ankle and step feedback
#: and trunk PD at its values; heading and phase-rate gains at zero), from which CMA-ES may move every gain.  The
#: earlier defaults are the build's untuned defaults (G4's first run), kept for the record.
GAIN_SPEC = (
    ("k_dcm", 2.0, 2.0, 0.0, 6.0, "ankle: ZMP shift per unit DCM error (prototype)"),
    ("cop_x", 0.020, 0.020, 0.0, 0.05, "ankle: ZMP shift limit forward/back, m (prototype)"),
    ("cop_y", 0.012, 0.012, 0.0, 0.03, "ankle: ZMP shift limit sideways, m (prototype)"),
    ("k_step", 1.0, 1.0, 0.0, 3.0, "step: touchdown offset per unit DCM error (prototype)"),
    ("step_x", 0.05, 0.05, 0.0, 0.10, "step: offset limit forward/back, m (prototype)"),
    ("step_out", 0.03, 0.03, 0.0, 0.06, "step: offset limit outward, m (prototype; inward: the swing pad 5 mm from the "
                                        "stance pad, decision 20)"),
    ("decay", 1.0, 1.0, 0.0, 3.0, "step and heading offsets' decay over a stance (prototype)"),
    ("k_trunk", 40.0, 40.0, 0.0, 150.0, "trunk roll/pitch P, N m/rad (prototype)"),
    ("d_trunk", 2.0, 2.0, 0.0, 10.0, "trunk roll/pitch D, N m s/rad (prototype)"),
    ("k_yaw", 0.0, 0.5, 0.0, 1.0, "heading P: fraction of the stride-mean error corrected per step (start: zero, "
                                  "decision 16; earlier default: the prototype's value)"),
    ("ki_yaw", 0.0, 0.1, 0.0, 0.5, "heading I per step (start: zero; earlier default: a fifth of the P gain, chosen "
                                   "before any run)"),
    ("k_phase", 0.0, 2.0, 0.0, 10.0, "phase rate P, 1/m (start: zero; earlier default: 2 % per cm of forward DCM "
                                     "error, chosen before any run)"),
    ("ki_phase", 0.0, 1.0, 0.0, 10.0, "phase rate I, 1/(m s) (start: zero; earlier default chosen before any run)"),
    ("capture_margin", 0.0, 0.0, -0.02, 0.05, "capture step: DCM distance outside the support that triggers it, m "
                                              "(standing only)"),
)
GAIN_NAMES = tuple(g[0] for g in GAIN_SPEC)
START_GAINS = np.array([g[1] for g in GAIN_SPEC])
EARLIER_DEFAULTS = np.array([g[2] for g in GAIN_SPEC])
#: The named gain sets: the tuning start (decision 16, the tracker's default) and the build's earlier defaults.
GAIN_SETS = {"start": START_GAINS, "earlier_defaults": EARLIER_DEFAULTS}
DEFAULT_GAINS = START_GAINS
GAIN_BOUNDS = np.array([(g[3], g[4]) for g in GAIN_SPEC])
GAINS_SOURCE = {"start": "STAGE1_PLAN.md decision 16 (the tuning start)",
                "earlier_defaults": "the build's untuned defaults (G4's first run), kept for the record"}


def gains_dict(x=None) -> dict:
    """The gain vector as a dict: None (the tuning start), a set name of ``GAIN_SETS``, or the values in
    ``GAIN_NAMES`` order."""
    if x is None or isinstance(x, str):
        if x is not None and x not in GAIN_SETS:
            raise ValueError(f"gain set {x!r}: choose from {', '.join(GAIN_SETS)}")
        x = GAIN_SETS[x or "start"]
    x = np.asarray(x, float)
    if x.shape != START_GAINS.shape:
        raise ValueError(f"{len(x)} gains; the tracker has {len(START_GAINS)} ({', '.join(GAIN_NAMES)})")
    return dict(zip(GAIN_NAMES, (float(v) for v in x), strict=True))


TURN_CASES = ("walk_0.35", "in_place_0.50")  # the target turns (plan section 4)


def _turning_screen(body: str) -> dict:
    return json.loads((case_dir(body) / "review" / "screening" / f"{body}_turning.json").read_text())


def turn_gap(body: str) -> float:
    """The target turns' smallest gap between screened parts (m), from the turning screen (``shortlist``'s rule)."""
    doc = _turning_screen(body)
    gaps = [c["min_gap_mm"] for name, c in doc["clearance"]["cases"].items()
            if name in TURN_CASES and c.get("min_gap_mm") is not None]
    return 1e-3 * float(min(gaps))


def turn_gap_info(body: str) -> dict:
    """What the turn gap is: per target turn, the turning screen's smallest gap, the pair of parts it is between
    (geoms, bodies, class, where) and the screen's lateral gap between the legs.  The pair is the trunk's pelvis plate
    and the same-side thigh, not two legs, so the turn gap is no hardware limit and bounds no step (decision 20)."""
    cases = _turning_screen(body)["clearance"]["cases"]
    out = {}
    for name in TURN_CASES:
        c = cases[name]
        deep = c.get("deepest") or {}
        bodies = deep.get("bodies") or []
        out[name] = dict(min_gap_mm=c.get("min_gap_mm"), pair=deep.get("geoms"), bodies=bodies,
                         pair_class=deep.get("class"), at=deep.get("at"), lateral_gap_mm=c.get("lateral_gap_mm"),
                         leg_to_leg=bool(len(bodies) == 2 and all(b.startswith(("left_", "right_")) for b in bodies)
                                         and bodies[0].split("_")[0] != bodies[1].split("_")[0]))
    return out


def _inward(side: int, yaw: float) -> np.ndarray:
    """The inward direction (toward the stance foot) of the swinging foot ``side`` in the frame of yaw ``yaw``: the
    left foot's outward is +y (``Tracker.feedback``)."""
    return (-1.0 if side == 0 else 1.0) * np.array([-math.sin(yaw), math.cos(yaw)])


def _swing_runs(swing_mask: np.ndarray, periodic: bool) -> list[np.ndarray]:
    """The runs of consecutive frames where ``swing_mask`` holds (a periodic run that wraps across the stride
    boundary is one run, in time order)."""
    fr = np.flatnonzero(swing_mask)
    if not len(fr):
        return []
    runs = np.split(fr, np.flatnonzero(np.diff(fr) > 1) + 1)
    n = len(swing_mask)
    if periodic and len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
        runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
    return runs


def inward_step_clearance(body: str, run_dir: Path, deltas=(0.0, 0.01, 0.02, 0.03, 0.05), swing_u=(0.6, 0.8, 1.0),
                          references=None) -> dict:
    """What the inward bound (decision 20) allows, with the clearance screen's geometry and distance oracle
    (``legs.LegGap``), on the nominal walk and on the turns the turning screen found its smallest gaps on: per swing,
    the bound at the reference touchdown (``inward_room``: how far inward the swing pad may land before it is within
    ``PAD_CLEARANCE`` of the stance pad); and on the poses at ``swing_u`` of each swing (near touchdown, where the
    tracker's step offset is blended in), with the swing foot moved inward (toward the stance foot, in its frame) by
    each of ``deltas`` and by the side's smallest bound (m) through the swing leg's Jacobian as ``Tracker.feedback``
    moves it, the smallest distance of the turning screen's pair (the turn gap's), of the leg-to-leg pairs that only
    come close in motion (``LegGap.moving``), and of every leg-to-leg pair (distances beyond ``LegGap.cap`` read as
    the cap).  Only the pads collide in the plant, so these show whether other leg parts meet within the bound.
    ``references`` names other references of ``run_dir`` to use instead."""
    from .legs import LegGap

    info = turn_gap_info(body)
    deltas = tuple(float(x) for x in deltas)
    lg = LegGap(body)
    S = lg.S
    m = S.m
    d = mujoco.MjData(m)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    feet = [statics.bid(m, f"{s}_foot") for s in statics.SIDES]
    pads = pad_polygons(m)
    jids = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{x}") for x in LEGS] for s in statics.SIDES]
    dofs = [np.array([m.jnt_dofadr[j] for j in js]) for js in jids]
    qadr = [np.array([m.jnt_qposadr[j] for j in js]) for js in jids]
    mov = np.flatnonzero(lg.moving)
    refs_ = {"nominal": None}
    for name, c in info.items():
        if c["at"] and c["at"].get("gait"):
            refs_[c["at"]["gait"]] = name
    if references is not None:
        refs_ = {name: refs_.get(name) for name in references}
    out = dict(rule="per swing, the inward bound at the reference touchdown (the swing pad at least "
                    f"{1e3 * PAD_CLEARANCE:g} mm from the stance pad, decision 20); near touchdown (the given fractions "
                    "of each swing) the swing foot moved inward by each delta and by the side's smallest bound through "
                    "the swing leg's Jacobian; smallest distance, by the clearance screen's oracle, of the turning "
                    "screen's pair (the turn gap), of the leg-to-leg pairs that only come close in motion, and of "
                    "every leg-to-leg pair (capped)",
               pad_clearance_m=PAD_CLEARANCE, deltas_m=list(deltas), swing_u=list(swing_u), cap_mm=1e3 * lg.cap,
               references={})

    def pose(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    for refname, case in refs_.items():
        ref = Reference.load(Path(run_dir) / "refs" / f"{refname}.npz")
        n = ref.n
        cases = [case] if case else list(info)
        pairs = {cs: [S.gid(g) for g in info[cs]["pair"]] for cs in cases if info[cs]["pair"]}
        rows = {}
        for side in (0, 1):
            runs = _swing_runs(~ref.support[:, side], periodic=True)
            if not runs:
                continue
            bounds = []
            for r in runs:  # the bound at each swing's touchdown, in the stance foot's frame
                pose(ref.q[(int(r[-1]) + 1) % n])
                ys, yw = _yaw(d.xmat[feet[1 - side]]), _yaw(d.xmat[feet[side]])
                rel = _rot2(ys).T @ (d.xpos[feet[side]][:2] - d.xpos[feet[1 - side]][:2])
                swing_pad = pads[side] @ _rot2(_wrap(yw - ys)).T + rel
                bounds.append(inward_room(pads[1 - side], swing_pad, _inward(side, 0.0)))
            bound = float(min(bounds))
            pick = [int(r[int(round(u * (len(r) - 1)))]) for r in runs for u in swing_u]
            per = []
            for delta in deltas + ((bound,) if math.isfinite(bound) else ()):
                best = dict(moving_leg_pairs_mm=math.inf, all_leg_pairs_mm=math.inf)
                who = dict(moving_leg_pairs=None, all_leg_pairs=None)
                pair_mm = {cs: math.inf for cs in pairs}
                for k in pick:
                    q = ref.q[k].copy()
                    pose(q)
                    Rst = d.xmat[feet[1 - side]].reshape(3, 3)
                    mujoco.mj_jacBody(m, d, jacp, jacr, feet[side])
                    J = np.vstack([jacp, jacr])[:, dofs[side]]
                    dfw = delta * _inward(side, math.atan2(Rst[1, 0], Rst[0, 0]))
                    q[qadr[side]] += np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ np.r_[dfw, 0.0, 0.0, 0.0, 0.0])
                    dist = lg.at(q)
                    for cs, (gi, gj) in pairs.items():
                        pair_mm[cs] = min(pair_mm[cs], 1e3 * min(S.dist(gi, gj, lg.cap), lg.cap))
                    ia = int(np.argmin(dist))
                    if 1e3 * dist[ia] < best["all_leg_pairs_mm"]:
                        best["all_leg_pairs_mm"], who["all_leg_pairs"] = 1e3 * float(dist[ia]), lg.pair(ia)
                    if len(mov):
                        im = int(mov[np.argmin(dist[mov])])
                        if 1e3 * dist[im] < best["moving_leg_pairs_mm"]:
                            best["moving_leg_pairs_mm"], who["moving_leg_pairs"] = 1e3 * float(dist[im]), lg.pair(im)
                per.append(dict(inward_m=float(delta), at_bound=bool(len(per) == len(deltas)),
                                turn_gap_pair_mm={cs: float(v) for cs, v in pair_mm.items()},
                                **{k_: float(v) for k_, v in best.items()}, moving_pair=who["moving_leg_pairs"],
                                all_pair=who["all_leg_pairs"]))
            rows[f"{statics.SIDES[side]}_swing"] = dict(bound_m=bound, bounds_m=[float(b) for b in bounds], rows=per)
        out["references"][refname] = dict(turn_gap_case=cases, turn_gap_pair={cs: info[cs]["pair"] for cs in pairs},
                                          poses_per_swing=len(swing_u), swings=rows)
    return out


def _min_jerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u**3 * (10 - 15 * u + 6 * u * u)


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _yaw(R) -> float:
    R = np.asarray(R).reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def _roll_pitch(R) -> tuple[float, float]:
    """Roll and pitch of R = Rz(yaw) Ry(pitch) Rx(roll) (relative to the heading frame)."""
    R = np.asarray(R).reshape(3, 3)
    return math.atan2(R[2, 1], R[2, 2]), math.asin(-max(-1.0, min(1.0, R[2, 0])))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------------
# Support polygons
# ------------------------------------------------------------------------------------------------
def pad_polygons(m) -> list[np.ndarray]:
    """Per foot, the pads' bottom-face polygon in the foot body's frame (xy, convex hull, counter-clockwise).  The
    pads are the foot's box geoms named ``*_pad_*`` (on every floor: a pile model turns their contype off)."""
    out = []
    for side in statics.SIDES:
        b = statics.bid(m, f"{side}_foot")
        pts = []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or "_pad_" not in name:
                continue
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            R = R.reshape(3, 3)
            h = m.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append((m.geom_pos[g] + R @ np.array([sx * h[0], sy * h[1], -h[2]]))[:2])
        pts = np.array(pts)
        out.append(pts[ConvexHull(pts).vertices])
    return out


def _halfplanes_np(poly: np.ndarray, inset: float = 0.0):
    """``halfplanes`` in numpy, as first written: the fallback for a polygon with a zero-length (or non-finite) edge,
    where the normal is 0 / 0."""
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    n = np.column_stack([e[:, 1], -e[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    return n, np.einsum("ij,ij->i", n, a) - inset


def halfplanes(poly: np.ndarray, inset: float = 0.0):
    """Outward normals N and offsets d of a counter-clockwise convex polygon (N x <= d inside), moved in by
    ``inset``.

    Computed in Python floats with exactly the operations of the numpy version (``_halfplanes_np``, bit for bit): the
    edge e = b - a, the normal (e_y, -e_x) divided by sqrt(e_y^2 + e_x^2) (numpy's row norm: one sum of two squares),
    the offset n . a as numpy's einsum forms it (two rounded products and one rounded sum, an exact zero as +0.0), less
    the inset.  Only elementwise IEEE operations are involved, so the result does not depend on the BLAS build."""
    pts = np.asarray(poly, float)
    k = len(pts)
    if k < 2 or pts.ndim != 2:
        return _halfplanes_np(pts, inset)
    P = pts.tolist()
    N, d = [], []
    for i in range(k):
        ax, ay = P[i]
        bx, by = P[i + 1] if i + 1 < k else P[0]
        ex, ey = bx - ax, by - ay
        mex = -ex
        nrm = math.sqrt(ey * ey + mex * mex)
        if not 0.0 < nrm < math.inf:
            return _halfplanes_np(pts, inset)
        nx, ny = ey / nrm, mex / nrm
        N.append((nx, ny))
        d.append((nx * ax + ny * ay + 0.0) - inset)
    return np.array(N), np.array(d)


def _solve2(A: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``np.linalg.solve(A, rhs)`` for a float64 (n, n) matrix and a float64 (n,) vector: the same LAPACK gufunc numpy
    calls (``_umath_linalg.solve1``, signature dd->d), without numpy's argument checks.  A result that is not finite
    (a singular matrix, or an overflow) is recomputed by ``np.linalg.solve`` itself, which raises or returns as it
    did.  Where numpy has no such private gufunc (``SOLVE2_PATH``), ``np.linalg.solve`` throughout."""
    if _SOLVE1 is None:  # pragma: no cover - a numpy without the private gufunc
        return np.linalg.solve(A, rhs)
    r = _SOLVE1(A, rhs, signature="dd->d")
    for v in r.tolist():
        if not math.isfinite(v):
            return np.linalg.solve(A, rhs)
    return r


def shrink(poly: np.ndarray, inset: float) -> np.ndarray:
    """The polygon moved in by ``inset`` (vertices of the offset edges' intersections)."""
    N, d = halfplanes(poly, inset)
    out = []
    k = len(poly)
    for i in range(k):
        j = (i - 1) % k
        A = np.vstack([N[j], N[i]])
        out.append(np.linalg.solve(A, np.array([d[j], d[i]])))
    return np.array(out)


def project_into(poly: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, bool]:
    """The point of a convex polygon closest to x (x itself if inside); and whether x was outside.

    The dot products stay numpy's (BLAS: on this build a fused multiply-add, so a Python formula would differ in the
    last bit); the elementwise steps (edges, the clamped edge parameter, the closest point) are the same IEEE
    operations in Python floats."""
    N, d = halfplanes(poly)
    s = (N @ x).tolist()
    for si, di in zip(s, d.tolist(), strict=True):
        if not si - di <= 0.0:
            break
    else:
        return x, False
    P = np.asarray(poly, float)
    k = len(P)
    E = np.concatenate((P[1:], P[:1])) - P  # e = b - a per edge
    XA = x - P  # x - a per edge
    Pl, El = P.tolist(), E.tolist()
    x0, x1 = float(x[0]), float(x[1])
    best, bd = x, math.inf
    for i in range(k):
        t = min(1.0, max(0.0, float(XA[i] @ E[i]) / float(E[i] @ E[i])))
        (ax, ay), (ex, ey) = Pl[i], El[i]
        cx, cy = ax + t * ex, ay + t * ey
        v = np.array((x0 - cx, x1 - cy))
        dist = math.sqrt(v.dot(v))  # np.linalg.norm of a contiguous 2-vector, bit for bit
        if dist < bd:
            best, bd = (cx, cy), dist
    return (best if best is x else np.array(best)), True


def _ankle_rows(kd, jids) -> tuple[np.ndarray, np.ndarray]:
    """The linear map between a foot's CoP c (world xy) and its ankle torques: tau / Fz = -(A c + b), from the moment
    of a vertical floor load at c about each ankle axis (quasi-static foot: the ankle torques balance it)."""
    A, b = np.zeros((2, 2)), np.zeros(2)
    for r, jid in enumerate(jids):
        ax, an = kd.xaxis[jid], kd.xanchor[jid]
        A[r] = (-ax[1], ax[0])
        b[r] = -an[1] * ax[0] + an[0] * ax[1]
    return A, b


def ankle_cop(kd, jids, tau, Fz: float) -> np.ndarray:
    """The CoP (world xy) that ankle torques ``tau`` (pitch, roll; on the foot) imply under a vertical load ``Fz``."""
    A, b = _ankle_rows(kd, jids)
    return np.linalg.solve(A, -np.asarray(tau, float) / Fz - b)


def ankle_torque(kd, jids, cop, Fz: float) -> np.ndarray:
    """The ankle torques (pitch, roll) that put the CoP of a vertical load ``Fz`` at ``cop`` (``ankle_cop``'s
    inverse)."""
    A, b = _ankle_rows(kd, jids)
    return -Fz * (A @ np.asarray(cop, float) + b)


def signed_distance(poly: np.ndarray, x: np.ndarray) -> float:
    """Distance of x outside the polygon (negative inside: minus the distance to the nearest edge line)."""
    N, d = halfplanes(poly)
    s = N @ x - d
    if np.all(s <= 0):
        return float(np.max(s))
    return float(np.linalg.norm(x - project_into(poly, x)[0]))


#: The margin (m, or N m for torques) by which a cheap estimate must clear a boundary before a decision is taken from
#: it instead of from the exact computation (``_certified_inside``): the estimates differ from the exact values by
#: rounding only (a few 1e-15 at these magnitudes), so this cannot flip a decision; within it the exact code decides.
CERTIFY_TOL = 1e-9


def _certified_inside(N: list, d: list, x0: float, x1: float, tol: float) -> bool | None:
    """Whether the point (x0, x1) is inside the half-planes N x <= d (lists of floats) with every margin beyond
    ``tol`` (True), outside with some margin beyond ``tol`` (False), or too close to decide this way (None)."""
    worst = -math.inf
    for (n0, n1), di in zip(N, d, strict=True):
        s = n0 * x0 + n1 * x1 - di
        if s != s:
            return None
        if s > tol:
            return False
        if s > worst:
            worst = s
    return True if worst < -tol else None


def clip_polygon(poly: np.ndarray, N: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The convex polygon ``poly`` (counter-clockwise) cut to the half-planes N x <= d (Sutherland-Hodgman); an empty
    result has shape (0, 2)."""
    out = np.asarray(poly, float)
    for n, c in zip(N, d, strict=True):
        if not len(out):
            break
        s = out @ n - c
        nxt = []
        k = len(out)
        for i in range(k):
            j = (i + 1) % k
            if s[i] <= 0:
                nxt.append(out[i])
            if (s[i] <= 0) != (s[j] <= 0):
                nxt.append(out[i] + (out[j] - out[i]) * (s[i] / (s[i] - s[j])))
        out = np.array(nxt) if nxt else np.zeros((0, 2))
    return out


_BOX_N = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def _clip_box(poly: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """``clip_polygon(poly, _BOX_N, [p0, p0, p1, p1])`` (the torque box |x| <= p0, |y| <= p1, p0 and p1 positive) in
    Python floats, bit for bit: against a unit axis normal the side value n . x - c is +-x_i - c exactly (products with
    1 and 0, sums with an exact zero), and the intersections are the same elementwise operations."""
    out = [tuple(r) for r in np.asarray(poly, float).tolist()]
    for axis, sign, c in ((0, 1.0, p0), (0, -1.0, p0), (1, 1.0, p1), (1, -1.0, p1)):
        if not out:
            break
        s = [(pt[axis] if sign > 0 else -pt[axis]) - c for pt in out]
        nxt = []
        k = len(out)
        for i in range(k):
            j = (i + 1) % k
            si, sj = s[i], s[j]
            if si <= 0:
                nxt.append(out[i])
            if (si <= 0) != (sj <= 0):
                r = si / (si - sj)
                (ax, ay), (bx, by) = out[i], out[j]
                nxt.append((ax + (bx - ax) * r, ay + (by - ay) * r))
        out = nxt
    return np.array(out) if out else np.zeros((0, 2))


def minkowski_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The convex polygon {x - y : x in a, y in b} of two convex polygons (counter-clockwise)."""
    pts = (np.asarray(a, float)[:, None, :] - np.asarray(b, float)[None, :, :]).reshape(-1, 2)
    return pts[ConvexHull(pts).vertices]


def inward_room(stance: np.ndarray, swing: np.ndarray, inward, clearance: float = PAD_CLEARANCE) -> float:
    """Decision 20's inward bound: how far the convex polygon ``swing`` (the swing pad at its touchdown) may move
    along the unit vector ``inward`` before it comes within ``clearance`` of the convex polygon ``stance`` (the stance
    pad): 0 if it already is, inf if it never comes that close.  The clearance is taken around the polygons' edge
    lines (sharp corners), so the bound keeps at least ``clearance`` everywhere and a little more near corners.

    The swing pad moved by t along ``inward`` is within the clearance of the stance pad when -t ``inward`` lies in
    their Minkowski difference grown by the clearance; the ray enters that convex polygon at the bound."""
    N, d = halfplanes(minkowski_difference(swing, stance), -clearance)
    nu = N @ -np.asarray(inward, float)
    lo, hi = -math.inf, math.inf
    for a, b in zip(nu.tolist(), d.tolist(), strict=True):
        if a > 1e-15:
            hi = min(hi, b / a)
        elif a < -1e-15:
            lo = max(lo, b / a)
        elif b < 0.0:
            return math.inf  # moving along an edge that stays outside: never within the clearance
    if lo > hi or hi < 0.0:
        return math.inf
    return max(lo, 0.0)


#: ``Tracker._inward_clear``: the bound is skipped only when the pads clear it by this much (m, plus the same share of
#: the step), far above the rounding of ``inward_room`` (about 1e-15 m).
INWARD_CERTIFY_M = 1e-5


def _inward_corner(pads, clearance: float) -> float | None:
    """How far a corner of ``inward_room``'s grown polygon (the Minkowski difference of the two pads with its edge
    lines moved out by ``clearance``, sharp corners) can lie from the difference itself: clearance / cos(theta / 2),
    theta the pads' largest exterior angle plus 1e-6 rad (the difference's exterior angles are pieces of its pads';
    rotating a pad keeps them).  None, so that ``Tracker._inward_clear`` certifies nothing, unless both pads are
    strictly convex counter-clockwise polygons with edges of at least 1e-6 m and exterior angles under 150 degrees."""
    theta = 0.0
    for P in pads:
        P = np.asarray(P, float)
        if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 2:
            return None
        e = np.roll(P, -1, axis=0) - P
        if not np.all(np.hypot(e[:, 0], e[:, 1]) >= 1e-6):
            return None
        e_in = np.roll(e, 1, axis=0)
        ang = np.arctan2(e_in[:, 0] * e[:, 1] - e_in[:, 1] * e[:, 0], np.einsum("ij,ij->i", e_in, e))
        if not (np.all(ang > 0.0) and abs(float(ang.sum()) - 2.0 * math.pi) < 1e-9):
            return None
        theta = max(theta, float(ang.max()))
    if not theta < math.radians(150.0):
        return None
    return clearance / math.cos(0.5 * theta + 1e-6)


def _det_negative(A: np.ndarray) -> bool:
    """``np.linalg.det(A) < 0.0`` for a 2 x 2 matrix: from a00 a11 - a01 a10 when that is clear of zero by far more
    than the rounding of either computation (1e-9 of its terms' size), else from ``np.linalg.det`` itself."""
    (a00, a01), (a10, a11) = A.tolist()
    p, q = a00 * a11, a01 * a10
    det = p - q
    if abs(det) > 1e-9 * (abs(p) + abs(q)):
        return det < 0.0
    return bool(np.linalg.det(A) < 0.0)


def _cap_pre(pad, A, b) -> tuple:
    """The parts of the cap that depend on the pad and the ankle rows only: the pad's half-planes, the pad's torques
    per unit load ``pad @ A.T + b`` (``-Fz`` times it is the pad's torque polygon) and whether det(A) < 0."""
    A = np.asarray(A, float)
    return halfplanes(pad), np.asarray(pad, float) @ A.T + np.asarray(b, float), _det_negative(A)


def cap_ankle_torque(tau, peak=None, Fz: float | None = None, A=None, b=None, pad=None) -> tuple[np.ndarray, bool, bool]:
    """Decision 15: the ankle torques ``tau`` (pitch, roll, on the foot) limited to the box |tau_i| <= ``peak``_i (one
    motor's peak per axis; None: no box) and, for a loaded foot (``Fz`` > 0 and ``pad``: the pad polygon
    ``COP_INSET`` inside, world xy, counter-clockwise), to the torques whose CoP under ``Fz`` lies in ``pad`` (``A``,
    ``b``: ``_ankle_rows``, tau = -Fz (A c + b)).  Returns the feasible torques nearest to ``tau`` (Euclidean, N m),
    and whether ``tau`` broke the box and the pad limit."""
    return _cap_ankle_torque(tau, peak, Fz, A, b, pad)


def _cap_ankle_torque(tau, peak=None, Fz: float | None = None, A=None, b=None, pad=None, c=None, pre=None):
    """``cap_ankle_torque``; ``c`` is the CoP ``solve(A, -tau / Fz - b)`` and ``pre`` (``_cap_pre``) the pad's
    half-planes, ``pad @ A.T + b`` and the sign of det(A) when the caller already has them (the same computations, so
    the same bits)."""
    tau = np.asarray(tau, float)
    box = peak is not None and bool(np.any(np.abs(tau) > peak))
    loaded = Fz is not None and Fz > 0.0 and pad is not None
    pad_out = False
    if loaded:
        A = np.asarray(A, float)
        if c is None:
            c = _solve2(A, -tau / Fz - np.asarray(b, float))
        N, d = halfplanes(pad) if pre is None else pre[0]
        pad_out = bool(np.any(N @ c - d > 0.0))
    if not (box or pad_out):
        return tau, False, False
    clipped = np.clip(tau, -peak, peak) if peak is not None else tau
    if not loaded:
        return clipped, box, False
    if pre is None:
        pre = _cap_pre(pad, A, b)
    V = -Fz * pre[1]  # the pad's torques (a convex polygon)
    if pre[2]:
        V = V[::-1]  # keep it counter-clockwise
    Nt, dt = halfplanes(V)
    if np.all(Nt @ clipped - dt <= 1e-12):  # the box's nearest point already in the pad: the nearest feasible one
        return clipped, box, pad_out
    if peak is not None:
        p = np.asarray(peak, float) * np.ones(2)
        V = _clip_box(V, float(p[0]), float(p[1])) if p[0] > 0.0 and p[1] > 0.0 else clip_polygon(
            V, _BOX_N, np.r_[p[0], p[0], p[1], p[1]])
        if len(V) < 3:  # no torque meets both (not met on these feet: no torque, the CoP under the ankle, does)
            return clipped, box, pad_out
    return project_into(V, tau)[0], box, pad_out


# ------------------------------------------------------------------------------------------------
# Track: the reference sequence the tracker follows
# ------------------------------------------------------------------------------------------------
@dataclass
class Segment:
    """One piece of a track: a periodic reference repeated ``strides`` times, a transition (finite frames), or a
    held frame.  ``S`` stacks the joint arrays per frame ([q, qd, tau, qdd], each in the plant's joint order),
    ``info`` the feedback's reference quantities per frame (``frame_info``)."""

    kind: str
    name: str
    t0: float
    duration: float
    S: np.ndarray
    info: dict
    step: float = 0.0  # periodic: frame interval
    n: int = 0  # periodic: frames per stride
    t: np.ndarray | None = None  # finite: frame times (with the next segment's first frame appended)
    period: float = 1.0
    duty: float = 0.6
    yaw_per_stride: float = 0.0


def _swing_of(support: np.ndarray) -> np.ndarray:
    """Per frame, the swinging foot (0 left, 1 right, -1 none)."""
    swing = np.full(len(support), -1)
    for j in (0, 1):
        swing[~support[:, j]] = j
    return swing


def _swing_progress(swing: np.ndarray, periodic: bool) -> np.ndarray:
    """Swing progress per frame from the run of swing frames it belongs to (0 at lift-off; a periodic run may wrap
    across the stride boundary)."""
    n = len(swing)
    u = np.zeros(n)
    idx = np.arange(n)
    for j in (0, 1):
        on = swing == j
        if not on.any():
            continue
        if periodic and on.all():
            continue
        start = np.flatnonzero(on & ~np.roll(on, 1)) if periodic else np.flatnonzero(on & ~np.r_[False, on[:-1]])
        for k0 in start:
            k1 = k0
            while on[k1 % n] and (periodic or k1 < n) and k1 - k0 < n:
                k1 += 1
                if not periodic and k1 >= n:
                    break
            ks = idx[np.arange(k0, k1) % n]
            u[ks] = np.arange(k1 - k0) / (k1 - k0)
    return u


def _anchor(swing: np.ndarray, periodic: bool, half: int) -> np.ndarray:
    """Anchor per frame (module docstring): the stance foot; in double support the foot that landed last (within
    half a stride), else the next stance foot (within half a stride), else 2 (a stand: the feet's midpoint)."""
    n = len(swing)
    anchor = np.full(n, 2)
    for k in range(n):
        if swing[k] >= 0:
            anchor[k] = 1 - swing[k]
            continue
        back = fwd = None
        for dk in range(1, (n if periodic else k + 1)):
            kk = k - dk
            if not periodic and kk < 0:
                break
            if swing[kk % n] >= 0:
                back = (dk, int(swing[kk % n]))
                break
        for dk in range(1, (n if periodic else n - k)):
            kk = k + dk
            if not periodic and kk >= n:
                break
            if swing[kk % n] >= 0:
                fwd = (dk, int(swing[kk % n]))
                break
        if back is not None and back[0] <= half:
            anchor[k] = back[1]
        elif fwd is not None and fwd[0] <= half:
            anchor[k] = 1 - fwd[1]
    return anchor


def _landings(swing: np.ndarray, feet: np.ndarray, fyaw: np.ndarray, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per frame of a swing: the swing foot's touchdown pose relative to the stance foot, in the stance foot's frame
    (xy and yaw; NaN outside the swings).  The touchdown is the first frame after the swing's run (wrapping for a
    periodic reference; a run that reaches the end of a finite segment lands at its last frame)."""
    n = len(swing)
    rel, yaw = np.full((n, 2), np.nan), np.full(n, np.nan)
    for j in (0, 1):
        for r in _swing_runs(swing == j, periodic):
            k = (int(r[-1]) + 1) % n if periodic else min(int(r[-1]) + 1, n - 1)
            s = 1 - j
            rel[r] = _rot2(fyaw[k, s]).T @ (feet[k, j] - feet[k, s])
            yaw[r] = _wrap(fyaw[k, j] - fyaw[k, s])
    return rel, yaw


def frame_info(p: plant.Plant, q: np.ndarray, qd: np.ndarray, support: np.ndarray, zmp_mb: np.ndarray,
               normal: np.ndarray, omega: float, periodic: bool, swing_u: np.ndarray | None = None,
               period: np.ndarray | float = 1.0, frames_per_stride: int = 160) -> dict:
    """The feedback's reference quantities per frame (module docstring): anchor (0 left, 1 right, 2 the feet's
    midpoint), swinging foot (-1 none) and its progress, the DCM and the ZMP relative to the anchor in its frame,
    the support polygon there (``COP_INSET`` inside, and as is), the normal-force split, the trunk's roll, pitch and
    (unwrapped) yaw, the local stride period, and in a swing the swing foot's touchdown pose relative to the stance
    foot (``_landings``: the inward bound of decision 20)."""
    m = p.m
    kd = mujoco.MjData(m)
    n = len(q)
    xi = np.zeros((n, 2))
    cv = np.zeros((n, 2))
    feet = np.zeros((n, 2, 2))
    fyaw = np.zeros((n, 2))
    rp = np.zeros((n, 2))
    yaw = np.zeros(n)
    for k in range(n):
        kd.qpos[:], kd.qvel[:] = q[k], qd[k]
        mujoco.mj_kinematics(m, kd)
        mujoco.mj_comPos(m, kd)
        mujoco.mj_comVel(m, kd)
        mujoco.mj_subtreeVel(m, kd)
        xi[k] = kd.subtree_com[p.trunk][:2] + kd.subtree_linvel[p.trunk][:2] / omega
        cv[k] = kd.subtree_linvel[p.trunk][:2]
        for j, f in enumerate(p.feet):
            feet[k, j] = kd.xpos[f][:2]
            fyaw[k, j] = _yaw(kd.xmat[f])
        rp[k] = _roll_pitch(kd.xmat[p.trunk])
        yaw[k] = _yaw(kd.xmat[p.trunk])
    yaw = np.unwrap(yaw)
    fyaw = np.unwrap(fyaw, axis=0)
    swing = _swing_of(support)
    if swing_u is not None:
        u = np.array([float(swing_u[k, swing[k]]) if swing[k] >= 0 else 0.0 for k in range(n)])
        u = np.nan_to_num(u)
    else:
        u = _swing_progress(swing, periodic)
    anchor = _anchor(swing, periodic, frames_per_stride // 2)
    polys = pad_polygons(m)
    inset = [shrink(P, COP_INSET) for P in polys]
    vmax = 2 * max(len(P) for P in polys)
    poly_in = np.full((n, vmax, 2), np.nan)
    poly_raw = np.full((n, vmax, 2), np.nan)
    poly_n = np.zeros(n, int)
    poly_rn = np.zeros(n, int)
    dcm_rel = np.zeros((n, 2))
    comv_rel = np.zeros((n, 2))
    zmp_rel = np.zeros((n, 2))
    for k in range(n):
        if anchor[k] < 2:
            pa, ya = feet[k, anchor[k]], fyaw[k, anchor[k]]
        else:
            pa, ya = feet[k].mean(0), float(np.mean(fyaw[k]))
        Rt = _rot2(ya).T
        dcm_rel[k] = Rt @ (xi[k] - pa)
        comv_rel[k] = Rt @ cv[k]
        zmp_rel[k] = Rt @ (zmp_mb[k] - pa) if np.all(np.isfinite(zmp_mb[k])) else dcm_rel[k]
        for target, store, count in ((inset, poly_in, poly_n), (polys, poly_raw, poly_rn)):
            pts = np.vstack([(feet[k, j] + target[j] @ _rot2(fyaw[k, j]).T - pa) @ Rt.T for j in (0, 1)
                             if support[k, j]])
            hull = pts[ConvexHull(pts).vertices]
            store[k, : len(hull)] = hull
            count[k] = len(hull)
    w = np.where(support, np.maximum(normal, 0.0), 0.0)
    tot = w.sum(1, keepdims=True)
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), np.where(support, 0.5, 0.0))
    per = np.broadcast_to(np.asarray(period, float), (n,)).copy()
    land_rel, land_yaw = _landings(swing, feet, fyaw, periodic)
    # each frame's support polygons' half-planes (``halfplanes``, as lists of floats): the feedback's quick inside tests
    poly_hp = [tuple(x.tolist() for x in halfplanes(poly_in[k, : poly_n[k]])) for k in range(n)]
    poly_raw_hp = [tuple(x.tolist() for x in halfplanes(poly_raw[k, : poly_rn[k]])) for k in range(n)]
    return dict(anchor=anchor, swing=swing, u=u, dcm_rel=dcm_rel, comv_rel=comv_rel, zmp_rel=zmp_rel, poly=poly_in,
                poly_n=poly_n, poly_raw=poly_raw, poly_raw_n=poly_rn, weights=w, roll_pitch=rp, yaw=yaw, period=per,
                stand=(swing < 0) & (anchor == 2), land_rel=land_rel, land_yaw=land_yaw, poly_hp=poly_hp,
                poly_raw_hp=poly_raw_hp)


def _clip(x: float, lo: float, hi: float) -> float:
    """``float(np.clip(x, lo, hi))`` of float scalars.  Where every implementation agrees (lo < hi and x strictly
    inside, below or above) the answer is taken directly; a tie with a bound (numpy's scalar and array paths keep
    different signs of zero there), a NaN or lo >= hi goes to ``np.clip`` itself."""
    if lo < hi:
        if lo < x < hi:
            return x
        if x < lo:
            return lo
        if x > hi:
            return hi
    return float(np.clip(x, lo, hi))


def _joint_stack(p: plant.Plant, q, qd, tau, qdd) -> np.ndarray:
    return np.hstack([q[:, p.qadr], qd[:, p.dofs], tau, qdd[:, p.dofs]])


def omega0(p: plant.Plant, q_home, crouch_frac: float = 0.03) -> float:
    """The references' LIPM frequency sqrt(g / z_c), z_c the home CoM height less the crouch (``gait.generate``)."""
    kd = mujoco.MjData(p.m)
    kd.qpos[:] = q_home
    mujoco.mj_kinematics(p.m, kd)
    mujoco.mj_comPos(p.m, kd)
    return math.sqrt(G / (float(kd.subtree_com[p.trunk][2]) - crouch_frac * float(p.meta["H_m"])))


class Track:
    """A sequence of segments on one time axis (``Segment``): ``joint(t)`` interpolates the joint arrays (as
    ``refs.Reference.interp`` for a periodic segment, so R0 on a periodic track is G2's replay bit for bit),
    ``info(t)`` the feedback's reference quantities.  ``q0`` and ``qd0`` are the first frame's full state."""

    def __init__(self, p: plant.Plant, omega: float):
        self.p, self.omega = p, omega
        self.segments: list[Segment] = []
        #: Commands on the reference clock: a stop transition's stop command (decision 18, ``add_transition``).
        self.events: list[dict] = []
        self.t_end = 0.0
        self.q0 = self.qd0 = None
        self._i = 0

    # ------------------------------------------------------------------ building
    def add_periodic(self, ref: Reference, strides: float, name: str | None = None) -> "Track":
        p = self.p
        if list(ref.joints) != p.joints:
            raise ValueError("reference and plant joints differ")
        S = _joint_stack(p, ref.q, ref.qd, ref.tau, ref.qdd)
        info = frame_info(p, ref.q, ref.qd, ref.support, ref.zmp_mb, ref.normal, self.omega, periodic=True,
                          period=ref.period, frames_per_stride=ref.n)
        seg = Segment("periodic", name or ref.name, self.t_end, strides * ref.period, S, info, step=ref.step, n=ref.n,
                      period=ref.period, duty=float(np.mean(ref.support[:, 0])),
                      yaw_per_stride=float(ref.stride_motion[0]))
        self._append(seg, ref.q[0], ref.qd[0])
        return self

    def add_transition(self, tr: Transition, next_frame: np.ndarray | None = None, next_dt: float | None = None) -> "Track":
        """A transition; ``next_frame`` (a ``_joint_stack`` row) is the following segment's first frame, ``next_dt``
        after the transition's last, for interpolation across the join (default: the last frame held)."""
        p = self.p
        S = _joint_stack(p, tr.q, tr.qd, tr.tau, tr.qdd)
        per = np.diff(tr.t) * N_FRAMES
        per = np.r_[per, per[-1]]
        info = frame_info(p, tr.q, tr.qd, tr.support, tr.zmp_mb, tr.normal, self.omega, periodic=False,
                          swing_u=tr.swing_u, period=per)
        dt_last = float(tr.t[-1] - tr.t[-2]) if next_dt is None else float(next_dt)
        S = np.vstack([S, S[-1:] if next_frame is None else np.asarray(next_frame)[None]])
        t = np.concatenate([tr.t, [tr.t[-1] + dt_last]])
        seg = Segment("finite", tr.name, self.t_end, float(t[-1]), S, info, t=t, period=float(per[-1]),
                      duty=float(np.mean(tr.support[:, 0])))
        if TRANSITIONS.get(tr.name, (None, None))[1] == "stand" and "phase" in tr.arrays:
            # a stop: its command is where its path speed starts to ramp down (the first gait's duty and speed)
            n0 = min(N_FRAMES, len(tr.t) - 1)
            duty = float(np.mean(tr.support[:n0, 0]))
            ph = stop_command_phase(duty)
            speed = float(np.linalg.norm(tr.path[1] - tr.path[0]) / (tr.t[1] - tr.t[0]))
            self.events.append(dict(kind="stop", name=tr.name, t_ref=self.t_end + float(np.interp(ph, tr.phase, tr.t)),
                                    command_phase=ph, speed_m_s=speed, period_s=float(tr.t[n0] - tr.t[0])))
        self._append(seg, tr.q[0], tr.qd[0])
        return self

    def add_hold(self, q, tau, zmp, seconds: float, name: str = "stand", period: float = 1.0) -> "Track":
        """Hold one pose (``q`` full qpos, feedforward ``tau``) at rest for ``seconds``: both feet down."""
        p = self.p
        q = np.asarray(q, float)[None]
        qd = np.zeros((1, p.m.nv))
        S = _joint_stack(p, q, qd, np.asarray(tau, float)[None], qd)
        info = frame_info(p, q, qd, np.ones((1, 2), bool), np.asarray(zmp, float)[None], np.ones((1, 2)), self.omega,
                          periodic=False, period=period)
        seg = Segment("hold", name, self.t_end, float(seconds), S, info, period=period)
        self._append(seg, q[0], qd[0])
        return self

    def _append(self, seg: Segment, q0, qd0) -> None:
        if self.segments:  # the reference yaw continues across the join
            seg.info["yaw"] = seg.info["yaw"] - seg.info["yaw"][0] + self._yaw_at_end(self.segments[-1])
        else:
            self.q0, self.qd0 = np.asarray(q0, float).copy(), np.asarray(qd0, float).copy()
        self.segments.append(seg)
        self.t_end = seg.t0 + seg.duration

    @staticmethod
    def _yaw_at_end(seg: Segment) -> float:
        y = seg.info["yaw"]
        if seg.kind == "periodic":
            strides = seg.duration / seg.period
            whole = math.floor(strides + 1e-9)
            k = int(round((strides - whole) * seg.n)) % seg.n
            return float(y[k] + whole * seg.yaw_per_stride)
        if seg.kind == "finite":
            return float(y[-1] + (y[-1] - y[-2]))
        return float(y[-1])

    # ------------------------------------------------------------------ lookup
    def segment(self, t: float) -> Segment:
        i = self._i
        segs = self.segments
        if t < segs[i].t0:
            i = 0
        while i + 1 < len(segs) and t >= segs[i + 1].t0:
            i += 1
        self._i = i
        return segs[i]

    def joint(self, t: float) -> np.ndarray:
        seg = self.segment(t)
        tl = t - seg.t0
        if seg.kind == "periodic":
            s = (tl / seg.step) % seg.n
            k0 = int(s) % seg.n
            a = s - int(s)
            return (1 - a) * seg.S[k0] + a * seg.S[(k0 + 1) % seg.n]
        if seg.kind == "hold":
            return seg.S[0]
        tt = seg.t
        k0 = min(max(int(tt.searchsorted(tl, side="right")) - 1, 0), len(tt) - 2)  # (np.searchsorted's own call)
        a = min(max((tl - tt[k0]) / (tt[k0 + 1] - tt[k0]), 0.0), 1.0)
        return (1 - a) * seg.S[k0] + a * seg.S[k0 + 1]

    def info(self, t: float) -> tuple[Segment, dict]:
        """The reference quantities at t: the frame at or before t, with the DCM, ZMP, weights, attitude and yaw
        interpolated toward the next frame when it has the same anchor."""
        seg = self.segment(t)
        tl = t - seg.t0
        inf = seg.info
        if seg.kind == "periodic":
            s = tl / seg.step
            strides = math.floor(s / seg.n)
            k0 = int(s) % seg.n
            k1 = (k0 + 1) % seg.n
            a = s - math.floor(s)
            off0 = strides * seg.yaw_per_stride
            off1 = off0 + (seg.yaw_per_stride if k1 == 0 else 0.0)
        elif seg.kind == "hold":
            k0 = k1 = 0
            a = off0 = off1 = 0.0
        else:
            nf = len(inf["anchor"])
            k0 = min(max(int(seg.t.searchsorted(tl, side="right")) - 1, 0), nf - 1)
            k1 = min(k0 + 1, nf - 1)
            a = 0.0 if k1 == k0 else min(max((tl - seg.t[k0]) / (seg.t[k1] - seg.t[k0]), 0.0), 1.0)
            off0 = off1 = 0.0
        frames = seg.__dict__.get("_frames")
        if frames is None or frames[0] is not inf:  # each frame's entries, picked once per segment (and its info)
            frames = seg._frames = (inf, [{key: v[k] for key, v in inf.items()} for k in range(len(inf["anchor"]))])
        out = dict(frames[1][k0])
        if a > 0 and inf["anchor"][k1] == inf["anchor"][k0]:
            for key in ("dcm_rel", "comv_rel", "zmp_rel", "weights", "roll_pitch"):
                out[key] = (1 - a) * inf[key][k0] + a * inf[key][k1]
            if inf["swing"][k1] == inf["swing"][k0]:
                out["u"] = (1 - a) * inf["u"][k0] + a * inf["u"][k1]
        out["yaw"] = (1 - a) * (inf["yaw"][k0] + off0) + a * (inf["yaw"][k1] + off1)
        return seg, out


def build_track(p: plant.Plant, run_dir: Path, plan: list, omega: float) -> Track:
    """A track from saved references and transitions (``run_dir``/refs, /transitions): ``plan`` lists ("ref", name,
    strides), ("transition", name) and ("hold", seconds) items in order.  A hold holds the next transition's first
    frame if it comes first, else the previous segment's last frame, at rest."""
    run_dir = Path(run_dir)
    objs = []
    for item in plan:
        if item[0] == "ref":
            objs.append(Reference.load(run_dir / "refs" / f"{item[1]}.npz"))
        elif item[0] == "transition":
            objs.append(Transition.load(run_dir / "transitions" / f"{item[1]}.npz"))
        elif item[0] == "hold":
            objs.append(None)
        else:
            raise ValueError(f"track item {item!r}")
    tr = Track(p, omega)
    for i, (item, obj) in enumerate(zip(plan, objs, strict=True)):
        if item[0] == "ref":
            tr.add_periodic(obj, item[2])
        elif item[0] == "transition":
            nxt = objs[i + 1] if i + 1 < len(objs) else None
            if isinstance(nxt, Reference):
                tr.add_transition(obj, _joint_stack(p, nxt.q[:1], nxt.qd[:1], nxt.tau[:1], nxt.qdd[:1])[0], nxt.step)
            elif isinstance(nxt, Transition):  # e.g. a start into a turn entry: on to the next one's first frame
                tr.add_transition(obj, _joint_stack(p, nxt.q[:1], nxt.qd[:1], nxt.tau[:1], nxt.qdd[:1])[0],
                                  float(nxt.t[1] - nxt.t[0]))
            else:
                tr.add_transition(obj)
        else:
            src = objs[i + 1] if i == 0 else objs[i - 1]
            k = 0 if i == 0 else -1
            period = float(np.median(np.diff(src.t))) * N_FRAMES if isinstance(src, Transition) else src.period
            tr.add_hold(src.q[k], src.tau[k], src.zmp_mb[k], item[1], period=period)
    return tr


# ------------------------------------------------------------------------------------------------
# State views
# ------------------------------------------------------------------------------------------------
class TrueState:
    """R1's privileged state, from the plant's data after the last step (positions as of that step's start, like
    the sensors; no extra ``mj_forward``)."""

    def __init__(self, p: plant.Plant):
        self.p = p

    def view(self) -> dict:
        p, m, d, v = self.p, self.p.m, self.p.d, self.p._v  # (the plant's views follow p.d)
        mujoco.mj_subtreeVel(m, d)
        xpos, xmat, t = v.xpos, v.xmat, p.trunk
        f0, f1 = p.feet
        return dict(d=d, com=v.subtree_com[t].copy(), com_vel=v.subtree_linvel[t].copy(),
                    feet=np.array([xpos[f0][:2], xpos[f1][:2]]),
                    feet_yaw=np.array([math.atan2(xmat[f0][3], xmat[f0][0]), math.atan2(xmat[f1][3], xmat[f1][0])]),
                    R=xmat[t].reshape(3, 3).copy(), omega=v.cvel[t][:3].copy())


class EstimatedState:
    """R2's state: the estimator's view."""

    def __init__(self, est: est_mod.Estimator):
        self.est = est

    def view(self) -> dict:
        e = self.est
        return dict(d=e.d, com=e.com.copy(), com_vel=e.com_vel.copy(), feet=e.foot[:, :2].copy(),
                    feet_yaw=e.foot_yaw.copy(), R=e.R.copy(), omega=e.omega_world.copy())


# ------------------------------------------------------------------------------------------------
# Tracker
# ------------------------------------------------------------------------------------------------
class Tracker:
    """R0, R1 or R2 on a plant following a track (module docstring)."""

    def __init__(self, p: plant.Plant, track: Track, mode: str = "R1", gains=None, terms=TERMS,
                 noise: est_mod.SensorNoise | None = None, seed: int = 0, est_gains: est_mod.EstimatorGains | None = None,
                 inward_bound: str | None = INWARD_BOUNDS[0], ankle_cap=ANKLE_CAP, rate: float = 1.0,
                 model: mujoco.MjModel | None = None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: choose from {', '.join(MODES)}")
        if not (rate > 0.0 and math.isfinite(rate)) or (mode == "R0" and rate != 1.0):
            raise ValueError(f"rate {rate!r}: a positive clock rate, and 1 for R0 (G2's replay clock)")
        self.p, self.track, self.mode = p, track, mode
        #: The controller's own model of the robot (None: the plant's model): the mass the tracker works with, and R2's
        #: estimator's kinematic and inertial model and its sensors' weight.  A plant whose masses are randomised
        #: (``env.py``) keeps the nominal model here: a deliberate model error, as a real robot's controller has.
        self.model = model
        #: The commanded clock rate (``COMMAND_RATE_RULE``): the track runs this much faster than real time, the
        #: phase-rate term's +-5 % about it; 1 is the references' own timing (G4).
        self.rate = float(rate)
        self.g = gains_dict(gains)
        self.gain_set = gains if isinstance(gains, str) else "start" if gains is None else "given"
        self.terms = frozenset(() if mode == "R0" else terms)
        unknown = self.terms - set(TERMS)
        if unknown:
            raise ValueError(f"unknown feedback terms {sorted(unknown)}")
        m = p.m
        self.dt = float(m.opt.timestep)
        self.nj = len(p.joints)
        self.damp = m.dof_damping[p.dofs].copy()
        self.fric = m.dof_frictionloss[p.dofs].copy()
        self.extra = p.armature_extra if np.any(p.armature_extra) else None
        self.mass = float((m if model is None else model).body_subtreemass[p.trunk])
        self.H = float(p.meta["H_m"])
        if inward_bound not in INWARD_BOUNDS:
            raise ValueError(f"inward bound {inward_bound!r}: choose from {INWARD_BOUNDS}")
        #: Decision 20: the swing foot's inward step keeps its pad ``PAD_CLEARANCE`` from the stance pad (None: lifted).
        self.inward_bound = inward_bound
        self.idx = {jn: i for i, jn in enumerate(p.joints)}
        self.jid = {jn: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in p.joints}
        self.leg = [np.array([self.idx[f"{s}_{x}"] for x in LEGS]) for s in statics.SIDES]
        self.leg_dofs = [p.dofs[ix] for ix in self.leg]
        self.leg_qadr = [p.qadr[ix] for ix in self.leg]
        self.lo = m.jnt_range[[self.jid[jn] for jn in p.joints], 0].copy()
        self.hi = m.jnt_range[[self.jid[jn] for jn in p.joints], 1].copy()
        self.noise = noise if noise is not None else est_mod.SensorNoise()
        self.seed = int(seed)
        self.est_gains = est_gains
        self.pads = pad_polygons(m)
        self.pads_in = [shrink(P, COP_INSET) for P in self.pads]
        self.pad_centre = [P.mean(0) for P in self.pads]
        self.pads_in_hp = [halfplanes(P) for P in self.pads_in]  # foot frame: the cap's quick inside test
        self._inward_key = self._inward_k = self._inward_ext = None  # ``_inward_clear``'s pad geometry, per pads
        self._jacp, self._jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._ik = mujoco.MjData(m)
        self.hold = FEEDBACK_HOLD
        #: Diagnostics (not the plan's tracker): ``cop_clip`` False drops the pad clip; ``ds_weights`` "equal" splits
        #: double support 50/50 as the design-study prototype did.
        self.cop_clip = True
        self.ds_weights = "normal"
        #: Decision 15: the cap on each ankle's total torque command (``ANKLE_CAP``; the others of ``ANKLE_CAPS`` are
        #: diagnostics, () none).
        if tuple(ankle_cap) not in ANKLE_CAPS:
            raise ValueError(f"ankle cap {ankle_cap!r}: choose from {ANKLE_CAPS}")
        self.ankle_cap = tuple(ankle_cap)
        Gm, kp, kv = p.G, p.kp, p.kv
        self.ankle_ix, self.ankle_K, self.ankle_D, self.ankle_jid = [], [], [], []
        for side in statics.SIDES:
            ix = np.array([self.idx[f"{side}_ankle_pitch"], self.idx[f"{side}_ankle_roll"]])
            K = Gm.T @ np.diag(kp) @ Gm
            D = Gm.T @ np.diag(kv) @ Gm
            self.ankle_ix.append(ix)
            self.ankle_K.append(K[np.ix_(ix, ix)])
            self.ankle_D.append(D[np.ix_(ix, ix)])
            self.ankle_jid.append([self.jid[f"{side}_ankle_pitch"], self.jid[f"{side}_ankle_roll"]])
        self.ankle_peak = np.array([[p.drives[i].peak for i in ix] for ix in self.ankle_ix])
        #: The ankle axes' rows of G^T (joint torques from the motors' forces): left pitch, roll, right pitch, roll.
        self.ankle_Gt = np.ascontiguousarray(Gm.T[np.concatenate(self.ankle_ix)])
        #: The leg drives (every drive but the neck's and the tail's) and the ankle motors, by actuator.
        self.leg_act = np.array([dr.group not in ("neck", "tail") for dr in p.drives])
        self.ankle_act = np.array([dr.group in ("ankle_a", "ankle_b") for dr in p.drives])
        self.hinge_jid = np.array([self.jid[jn] for jn in p.joints])
        self.foot_of = foot_geoms(p)
        # --- constants of the fast paths (same values as the expressions they replace)
        #: Each ankle's (pitch, roll) joint indices are adjacent: index of the pitch.
        self._ank0 = [int(ix[0]) for ix in self.ankle_ix]
        if any(list(ix) != [i0, i0 + 1] for ix, i0 in zip(self.ankle_ix, self._ank0, strict=True)):
            raise ValueError("the fast ankle cap expects each ankle's pitch and roll joints adjacent")
        self._ankle_Kl = [K.tolist() for K in self.ankle_K]
        self._ankle_Dl = [D.tolist() for D in self.ankle_D]
        self._ankle_peakl = [pk.tolist() for pk in self.ankle_peak]
        self._pads_in_hpl = [(N.tolist(), dd.tolist()) for N, dd in self.pads_in_hp]
        self._floor_load = CONTACT_LOAD_FRAC * self.mass * G  # as _ankle_cap writes it
        self._side_jids = [dict(pitch_roll=[self.jid[f"{s}_ankle_pitch"], self.jid[f"{s}_ankle_roll"]],
                                hips=[self.jid[f"{s}_hip_pitch"], self.jid[f"{s}_hip_roll"]],
                                hip_yaw=self.jid[f"{s}_hip_yaw"]) for s in statics.SIDES]
        self._side_idx = [dict(pitch_roll=[self.idx[f"{s}_ankle_pitch"], self.idx[f"{s}_ankle_roll"]],
                               hips=[self.idx[f"{s}_hip_pitch"], self.idx[f"{s}_hip_roll"]],
                               hip_yaw=self.idx[f"{s}_hip_yaw"]) for s in statics.SIDES]
        self._eye6 = 1e-6 * np.eye(6)
        self._zeros4 = np.zeros(4)
        self._ff_bufs = (np.empty(self.nj), np.empty(self.nj))
        #: A residual policy's inputs (``env.py``, plan section 2's R3): joint offsets added to every servo target
        #: (rad, joint order; None: none) and a factor on the clock rate (1: none).  With None and 1 the tracker is
        #: exactly R2 (the offsets are not added, the rate not multiplied).
        self.residual: np.ndarray | None = None
        self.rate_factor = 1.0
        #: Everything set so far is configuration; what ``reset`` adds is the episode's state (``save``).
        self._config = frozenset(self.__dict__) | {"_config", "sensors", "est", "state"}

    # ------------------------------------------------------------------ episode
    def reset(self, start: str = "frame", velocity_offset=None, joint_noise: float = 0.0, seed: int | None = None,
              root_velocity_noise: float = 0.0, noise_seed: int | None = None) -> None:
        """``start="frame"``: the track's first frame, already moving (as G2's replay and the design-study
        prototype start); ``"stance"``: the plant's settled stance at rest (with ``joint_noise`` and
        ``root_velocity_noise`` drawn from ``seed``, ``plant.Plant.reset``).  ``velocity_offset`` (m/s, world) is
        added to the root velocity of a frame start.  R2's sensor noise is drawn from ``noise_seed`` (default:
        ``seed``)."""
        p = self.p
        seed = self.seed if seed is None else int(seed)
        noise_seed = seed if noise_seed is None else int(noise_seed)
        if start == "frame":
            qd0 = self.track.qd0.copy()
            if velocity_offset is not None:
                qd0[:3] += np.asarray(velocity_offset, float)
            p.set_state(self.track.q0, qd0)
        elif start == "stance":
            p.reset(joint_noise=joint_noise, root_velocity_noise=root_velocity_noise, seed=seed)
        else:
            raise ValueError(f"start {start!r}: 'frame' or 'stance'")
        self.start = start
        self.track._i = 0
        self.tau_ref = 0.0
        self.delta = 0.0
        self.k = 0
        nj = self.nj
        self.dq, self.dtau = np.zeros(nj), np.zeros(nj)
        self.step_tgt = [np.zeros(6), np.zeros(6)]
        self.step_land = [np.zeros(6), np.zeros(6)]
        self.yaw_tgt = [0.0, 0.0]
        self.yaw_land = [0.0, 0.0]
        self.prev_sw = -1
        self.I_yaw = self.I_x = 0.0
        self.yaw_buf = []
        self.yaw_prev = None
        self.yaw_unwrapped = 0.0
        self.yaw0_ref = None
        self.capture = None
        self.capture_log = []
        self.flags = {}
        self.e = np.zeros(2)
        self.dq_prev, self.dtau_prev = np.zeros(nj), np.zeros(nj)
        self._ddq = self._ddtau = None  # the ramp's increments over the current feedback interval
        self._geo = self._geo_kv = None  # R2: the stance feet's cap geometry on the estimate (``_cap_geometry``)
        self._geo_np = [None, None]  # R2: its exact path's arrays (``_exact_arrays``)
        self.cap_bound = [False, False]  # the cap's box and pad bound since the last feedback update
        self.step_off = [np.zeros(2), np.zeros(2)]  # the swing foot's touchdown offset (anchor frame, m)
        self.sensors = self.est = None
        if self.mode == "R2":
            self.sensors = est_mod.NoisySensors(p, self.noise, noise_seed,
                                                weight=None if self.model is None else self.mass * G)
            self.est = est_mod.Estimator(p, self.est_gains, model=self.model)
            if start == "frame":  # the start state is the commanded one (the estimate's world frame)
                self.est.reset(p.d.qpos.copy(), p.d.qvel.copy(), self.sensors.read())
            else:  # at rest: from the sensors; the start's heading and position define the estimate's world frame
                self.est.reset_at_rest(self.sensors.read(), yaw=_yaw(p.d.xmat[p.trunk]), base_xy=p.d.qpos[:2].copy())
            self.state = EstimatedState(self.est)
        else:
            self.state = TrueState(p)

    # ------------------------------------------------------------------ episode state (counterfactuals)
    def save(self, prev_state: np.ndarray) -> dict:
        """The episode's state for ``load``.  ``prev_state`` is MuJoCo's integration state (``MJ_STATE``: physics,
        inputs and the solver's warm start) at the start of the physics step just taken (``run`` keeps it), so that
        ``load`` can replay that step and every derived quantity (positions, contacts, sensors, which MuJoCo leaves at
        the step's start) is as it was; with it the servo command, every controller variable (all that ``reset``
        sets), the estimator's state and the sensor-noise generator's."""
        d = self.p.d
        snap = dict(prev=np.array(prev_state, copy=True), ctrl=d.ctrl.copy(), time=float(d.time),
                    tracker={k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in self._config},
                    track_i=self.track._i)
        if self.est is not None:
            e = self.est
            snap["est"] = {k: copy.deepcopy(v) for k, v in e.__dict__.items() if k not in EST_STATIC}
            snap["est_qpos"], snap["est_qvel"] = e.d.qpos.copy(), e.d.qvel.copy()
            snap["rng"] = copy.deepcopy(self.sensors.rng.bit_generator.state)
        return snap

    def load(self, snap: dict) -> None:
        """Back to a state ``save`` took: MuJoCo's state before the step that led to it, that step replayed with its
        command (the envelope set from the drive speeds as ``Plant.step`` sets it), then the controller's, the
        estimator's (its kinematic view recomputed from the pose it held) and the noise generator's state."""
        p = self.p
        m, d = p.m, p.d
        mujoco.mj_setState(m, d, snap["prev"], MJ_STATE)
        d.ctrl[:] = snap["ctrl"]
        p.envelope.restore()
        p.step()
        if float(d.time) != snap["time"]:
            raise RuntimeError(f"replayed step reached t = {d.time!r}, the snapshot was taken at {snap['time']!r}")
        for k, v in snap["tracker"].items():
            setattr(self, k, copy.deepcopy(v))
        self.track._i = snap["track_i"]
        if self.est is not None:
            e = self.est
            for k, v in snap["est"].items():
                setattr(e, k, copy.deepcopy(v))
            e.d.qpos[:], e.d.qvel[:] = snap["est_qpos"], snap["est_qvel"]
            mujoco.mj_kinematics(e.m, e.d)  # (the estimator's own model: the plant's unless the tracker has a model)
            mujoco.mj_comPos(e.m, e.d)
            mujoco.mj_comVel(e.m, e.d)
            mujoco.mj_subtreeVel(e.m, e.d)
            self.sensors.rng.bit_generator.state = copy.deepcopy(snap["rng"])

    # ------------------------------------------------------------------ one physics step
    def step(self) -> None:
        p = self.p
        t_ref = p.d.time if self.mode == "R0" else self.tau_ref
        x = self.track.joint(t_ref)
        nj = self.nj
        q, qd, tau = x[:nj], x[nj : 2 * nj], x[2 * nj : 3 * nj]
        s = self.rate * (1.0 + self.delta)  # (rate 1: exactly 1 + delta, as before the commanded rate)
        if self.rate_factor != 1.0:  # a residual policy's clock rate (env.py); 1: the tracker as built
            s = s * self.rate_factor
        scaled = s != 1.0
        qd_c = s * qd if scaled else qd
        b1, b2 = self._ff_bufs
        ff = np.add(tau, np.multiply(self.damp, qd_c, out=b1), out=b1)  # tau + damp qd_c + fric tanh(qd_c / 0.05)
        ff += np.multiply(self.fric, np.tanh(np.divide(qd_c, FRICTION_SMOOTH, out=b2), out=b2), out=b2)
        if self.extra is not None:
            ff += self.extra @ (s * s * x[3 * nj :] if scaled else x[3 * nj :])
        if self.mode == "R0":
            p.command(q, qd, ff)
        else:
            j = self.k % FEEDBACK_EVERY
            if j == 0:
                self.dq_prev, self.dtau_prev = self.dq, self.dtau
                self.feedback(t_ref, q)
                self._ddq, self._ddtau = self.dq - self.dq_prev, self.dtau - self.dtau_prev
            if self.hold == "ramp":
                a = (j + 1) / FEEDBACK_EVERY
                dq = self.dq_prev + a * self._ddq
                dtau = self.dtau_prev + a * self._ddtau
            else:
                dq, dtau = self.dq, self.dtau
            q_cmd = q + dq
            if self.residual is not None:  # a residual policy's joint offsets (env.py); None: the tracker as built
                q_cmd = q_cmd + self.residual
            tau_cmd = ff + dtau
            if self.ankle_cap:
                capped = self._ankle_cap(q_cmd, qd_c, tau_cmd, dtau)
                if capped is not dtau:
                    tau_cmd = ff + capped
            p.command(q_cmd, qd_c, tau_cmd)
        p.step()
        self.k += 1
        self.tau_ref += self.dt * s
        if self.est is not None:
            self.est.step(self.sensors.read())

    # ------------------------------------------------------------------ ankle torque cap (500 Hz)
    def _foot_geometry(self, v: "_Views", i: int) -> tuple:
        """Foot ``i``'s cap geometry on the kinematics ``v`` (views of an MjData) as Python floats, the same values as
        ``_ankle_rows`` and ``_rot2(_yaw(...))``: (a00, a01, a10, a11, b0, b1, cos, sin, x, y), the foot origin (x,
        y)."""
        jp, jr = self.ankle_jid[i]
        f = self.p.feet[i]
        xa, xn = v.xaxis, v.xanchor
        p0, p1, _ = xa[jp].tolist()
        n0, n1, _ = xn[jp].tolist()
        r0, r1, _ = xa[jr].tolist()
        m0, m1, _ = xn[jr].tolist()
        R = v.xmat[f]
        yaw = math.atan2(R[3], R[0])
        x, y, _ = v.xpos[f].tolist()
        return (-p1, p0, -r1, r0, -n1 * p0 + n0 * p1, -m1 * r0 + m0 * r1, math.cos(yaw), math.sin(yaw), x, y)

    def _cap_geometry(self) -> list:
        """R2: both feet's cap geometry on the estimate, recomputed only when the estimator's kinematics changed (its
        kinematics are refreshed at its 100 Hz update, so four of five physics steps reuse them)."""
        e = self.est
        if self._geo_kv != e._kv or self._geo is None:
            v = e._views
            self._geo = [self._foot_geometry(v, 0), self._foot_geometry(v, 1)]
            self._geo_np = [None, None]  # (_exact_arrays, made when needed)
            self._geo_kv = e._kv
        return self._geo

    def _exact_arrays(self, i: int, g: tuple, cache: list | None) -> tuple:
        """The exact path's arrays for foot ``i`` from its geometry ``g`` (``_foot_geometry``): A, b (``_ankle_rows``),
        Rz, the foot's xy, its inset pad in world xy and the cap's pad parts (``_cap_pre``); kept in ``cache`` (R2:
        until the estimate's next update) when given."""
        if cache is not None and cache[i] is not None:
            return cache[i]
        a00, a01, a10, a11, b0, b1, cz, sz, fx, fy = g
        A, b = np.array([[a00, a01], [a10, a11]]), np.array([b0, b1])
        Rz = np.array([[cz, -sz], [sz, cz]])
        xy = np.array([fx, fy])
        poly = xy + self.pads_in[i] @ Rz.T
        out = (A, b, Rz, xy, poly, _cap_pre(poly, A, b))
        if cache is not None:
            cache[i] = out
        return out

    def _ankle_cap(self, q_cmd, qd_cmd, tau_cmd, dtau):
        """``dtau`` with each ankle's total torque command capped (decision 15, ``cap_ankle_torque``): the servo's PD
        torque is predicted from the encoders (true angles and rates for R1, the last reading for R2) and added to the
        feedforward and the feedback; each axis is held to one motor's peak, and a foot carrying more than
        ``CONTACT_LOAD_FRAC`` of the weight (the foot loads: true for R1, read for R2) to the torques that put its
        CoP (quasi-static foot, ``ankle_cop``) ``COP_INSET`` inside its pad; the nearest such torque replaces the
        total.  R2 takes its loaded feet from the estimator's contact detection (``EstimatorGains.contact_on`` /
        ``contact_off``, its odometry's), since its load readings carry noise of the same size as the 2 % threshold,
        and their loads from the readings (at least that threshold).

        Most steps no foot is near either limit.  A foot is passed over without the exact computation only when a
        float estimate of its torque and CoP clears the box and the inset pad by ``CERTIFY_TOL`` scaled to the sizes
        involved (``_certified_inside``): the exact test would then also pass it over.  Every other foot takes the
        exact path, the operations as before."""
        p = self.p
        floor_load = self._floor_load
        est = self.est
        if est is not None:  # the estimate's encoders, loads, contacts and kinematics (est.d)
            q, qd = est.q, est.qd
            loads = [x if (x >= floor_load or x != x) else floor_load for x in est.load.tolist()]  # np.maximum
            in_contact = est.contact
            geo = self._cap_geometry()
            views = est._views
        else:  # the true state (p.d)
            views = p._v
            q, qd = views.qpos[p.qadr], views.qvel[p.dofs]
            loads_a = p.foot_loads()
            loads = loads_a.tolist()
            in_contact = loads_a >= floor_load
            geo = None
        box_on, pad_on = "box" in self.ankle_cap, "pad" in self.ankle_cap
        qcl, ql, qdcl, qdl, tcl = q_cmd.tolist(), q.tolist(), qd_cmd.tolist(), qd.tolist(), tau_cmd.tolist()
        out = dtau
        for i in (0, 1):
            loaded = pad_on and bool(in_contact[i])
            # --- the float estimate (the exact torque is K (q_cmd - q) + D (qd_cmd - qd) + tau_cmd by BLAS)
            i0 = self._ank0[i]
            e0, e1 = qcl[i0] - ql[i0], qcl[i0 + 1] - ql[i0 + 1]
            v0, v1 = qdcl[i0] - qdl[i0], qdcl[i0 + 1] - qdl[i0 + 1]
            (k00, k01), (k10, k11) = self._ankle_Kl[i]
            (d00, d01), (d10, d11) = self._ankle_Dl[i]
            ka0, ka1 = k00 * e0 + k01 * e1, k10 * e0 + k11 * e1
            da0, da1 = d00 * v0 + d01 * v1, d10 * v0 + d11 * v1
            c0, c1 = tcl[i0], tcl[i0 + 1]
            t0, t1 = ka0 + da0 + c0, ka1 + da1 + c1
            size = 1.0 + abs(ka0) + abs(ka1) + abs(da0) + abs(da1) + abs(c0) + abs(c1)
            sure = size < math.inf  # (NaN fails this too)
            if sure and box_on:
                pk0, pk1 = self._ankle_peakl[i]
                sure = max(abs(t0) - pk0, abs(t1) - pk1) < -CERTIFY_TOL * size
            if sure and loaded:
                a00, a01, a10, a11, b0, b1, cz, sz, fx, fy = geo[i] if geo is not None else self._foot_geometry(views, i)
                Fz = loads[i]
                det = a00 * a11 - a01 * a10
                if Fz > 0.0 and abs(det) > 1e-3:
                    r0, r1 = -t0 / Fz - b0, -t1 / Fz - b1
                    x0, x1 = (r0 * a11 - r1 * a01) / det - fx, (a00 * r1 - a10 * r0) / det - fy
                    cf0, cf1 = cz * x0 + sz * x1, -sz * x0 + cz * x1  # Rz^T (c - foot)
                    tol = CERTIFY_TOL * (1.0 + abs(r0) + abs(r1) + abs(fx) + abs(fy) + size / Fz)
                    N, dd = self._pads_in_hpl[i]
                    sure = _certified_inside(N, dd, cf0, cf1, tol) is True
                else:
                    sure = False
            if sure:
                continue
            # --- the exact computation, as written before
            ix = self.ankle_ix[i]
            tau = self.ankle_K[i] @ (q_cmd[ix] - q[ix]) + self.ankle_D[i] @ (qd_cmd[ix] - qd[ix]) + tau_cmd[ix]
            peak = self.ankle_peak[i] if box_on else None
            Fz = float(loads[i])
            A = b = c = poly = pre = None
            inside = True
            if loaded:  # quick test in the foot's (yaw) frame, where the pad polygon is fixed, before the exact cap
                A, b, Rz, xy, poly, pre = self._exact_arrays(
                    i, geo[i] if geo is not None else self._foot_geometry(views, i),
                    self._geo_np if geo is not None else None)
                c = _solve2(A, -tau / Fz - b)
                cf = Rz.T @ (c - xy)
                N, dd = self.pads_in_hp[i]
                inside = bool(np.all(N @ cf <= dd))
            if inside and (peak is None or bool(np.all(np.abs(tau) <= peak))):
                continue
            new, box_b, pad_b = _cap_ankle_torque(tau, peak, Fz if loaded else None, A, b, poly, c, pre)
            self.cap_bound[0] = self.cap_bound[0] or box_b
            self.cap_bound[1] = self.cap_bound[1] or pad_b
            if np.any(new != tau):
                if out is dtau:
                    out = dtau.copy()
                out[ix] += new - tau
        return out

    def dcm_reference(self, ref: dict) -> np.ndarray:
        """The reference DCM relative to the anchor (``frame_info``): the track played at the commanded rate moves its
        CoM ``rate`` times as fast, so its DCM is c + rate c' / omega0 (``COMMAND_RATE_RULE``; rate 1: the
        reference's own)."""
        if self.rate == 1.0:
            return ref["dcm_rel"]
        return ref["dcm_rel"] + (self.rate - 1.0) * ref["comv_rel"] / self.track.omega

    def cap_residual(self, loads) -> tuple[float, float, float, int, int]:
        """How far the realised ankle torques break decision 15's cap on the true state (``CAP_RESIDUAL_RULE``): the
        largest excess of an axis over one motor's peak (N m, 0 within); over the feet carrying more than
        ``CONTACT_LOAD_FRAC`` of the weight (``loads`` the true foot loads) and over those carrying more than
        ``CAP_RESIDUAL_HEAVY_FRAC``, the largest distance (N m) from a foot's torques to the nearest torques that put
        its quasi-static CoP inside its inset pad (``cap_ankle_torque`` without the box); and how many feet were so
        loaded.  The motors' forces and the kinematics are those of the step just taken (its start state)."""
        v = self.p._v
        tau = self.ankle_Gt @ v.actuator_force
        weight = self.mass * G
        tl = tau.tolist()
        box = pad = pad_heavy = 0.0
        loaded = heavy = 0
        for i in (0, 1):
            t0, t1 = tl[2 * i], tl[2 * i + 1]
            pk0, pk1 = self._ankle_peakl[i]
            x0, x1 = abs(t0) - pk0, abs(t1) - pk1
            box = max(box, float(x0 if (x0 >= x1 or x0 != x0) else x1))  # (np.max of the two)
            Fz = float(loads[i])
            if Fz <= CONTACT_LOAD_FRAC * weight:
                continue
            loaded += 1
            is_heavy = Fz > CAP_RESIDUAL_HEAVY_FRAC * weight
            heavy += is_heavy
            a00, a01, a10, a11, b0, b1, cz, sz, fx, fy = self._foot_geometry(v, i)
            det = a00 * a11 - a01 * a10
            if abs(det) > 1e-3 and Fz > 0.0:  # the CoP clearly inside the inset pad: no excess (_certified_inside)
                r0, r1 = -t0 / Fz - b0, -t1 / Fz - b1
                u0, u1 = (r0 * a11 - r1 * a01) / det - fx, (a00 * r1 - a10 * r0) / det - fy
                N, dd = self._pads_in_hpl[i]
                tol = CERTIFY_TOL * (1.0 + abs(r0) + abs(r1) + abs(fx) + abs(fy))
                if _certified_inside(N, dd, cz * u0 + sz * u1, -sz * u0 + cz * u1, tol) is True:
                    continue
            t = tau[2 * i : 2 * i + 2]
            A, b = np.array([[a00, a01], [a10, a11]]), np.array([b0, b1])
            Rz = np.array([[cz, -sz], [sz, cz]])
            xy = np.array([fx, fy])
            c = _solve2(A, -t / Fz - b)
            cf = Rz.T @ (c - xy)  # the CoP in the foot frame
            N, dd = self.pads_in_hp[i]
            if np.any(N @ cf > dd):
                poly = xy + self.pads_in[i] @ Rz.T
                x = float(np.linalg.norm(_cap_ankle_torque(t, None, Fz, A, b, poly, c)[0] - t))
                pad = max(pad, x)
                if is_heavy:
                    pad_heavy = max(pad_heavy, x)
        return box, pad, pad_heavy, loaded, heavy

    # ------------------------------------------------------------------ feedback (100 Hz)
    def _anchor_frame(self, st: dict, a: int):
        if a < 2:
            return st["feet"][a], float(st["feet_yaw"][a])
        fy = st["feet_yaw"]
        return st["feet"].mean(0), math.atan2(float(np.mean(np.sin(fy))), float(np.mean(np.cos(fy))))

    def _inward_room(self, ref: dict, sw: int, dfx: float) -> float:
        """Decision 20: how far inward the swing foot ``sw`` may step, its touchdown moved forward by ``dfx``, before
        its pad comes within ``PAD_CLEARANCE`` of the stance pad: the reference touchdown pose relative to the stance
        foot (``frame_info``'s ``land_rel``, ``land_yaw``), both pads in the stance foot's frame (the anchor's)."""
        rel, yaw = ref["land_rel"], float(ref["land_yaw"])
        if not (np.all(np.isfinite(rel)) and math.isfinite(yaw)):
            return math.inf
        swing = self.pads[sw] @ _rot2(yaw).T + np.asarray(rel) + np.array([dfx, 0.0])
        return inward_room(self.pads[1 - sw], swing, _inward(sw, 0.0))

    def _inward_clear(self, ref: dict, sw: int, dfx: float, lat: float, so: float) -> bool:
        """Whether decision 20's bound certainly leaves the inward step ``lat`` (<= 0, at most 10 m) alone, so that
        ``_inward_room`` (a convex hull) need not run: with ``so`` > 0, ``_clip(lat, -room, so)`` is then lat and the
        clearance flag False whatever the room's exact value.

        The room is where the ray {t w : t >= 0} (w the step's direction reversed, ``inward_room``) enters the grown
        polygon G: the Minkowski difference D of the swing pad (at the same touchdown pose ``_inward_room`` uses) and
        the stance pad, its edge lines moved out by the clearance.  G lies within ``_inward_corner`` (k) of D, so the
        room exceeds -lat when the segment {t w : 0 <= t <= -lat} stays farther than k from D, and is infinite when the
        whole ray does.  Separations along the inward axis and the forward axis bound that distance from below:
        D's least coordinate along an axis is the swing pad's least less the stance pad's greatest (and rounding
        those differences keeps their order, so it is the same float), and the segment's points have forward
        coordinate 0.  ``INWARD_CERTIFY_M`` covers the rounding of ``inward_room`` (whose finite result r satisfies
        every edge's inequality at r w to about 1e-15 m) and of these sums."""
        need = -lat
        if not (0.0 <= need <= 10.0 and so > 0.0 and len(self.pads) == 2):
            return False
        P0, P1 = (np.asarray(P, float) for P in self.pads)
        clearance = inward_room.__defaults__[-1]  # the clearance ``_inward_room`` gets: inward_room's default
        key = (P0.tobytes(), P1.tobytes(), P0.shape, P1.shape, clearance)
        if key != self._inward_key:
            ext = [(P.min(0).tolist(), P.max(0).tolist()) for P in (P0, P1)]
            self._inward_key, self._inward_k, self._inward_ext = key, _inward_corner((P0, P1), clearance), ext
        k = self._inward_k
        if k is None:
            return False
        rel, yaw = ref["land_rel"], float(ref["land_yaw"])
        swing = self.pads[sw] @ _rot2(yaw).T + np.asarray(rel) + np.array([dfx, 0.0])
        (x0, y0), (x1, y1) = swing.min(0).tolist(), swing.max(0).tolist()
        (sx0, sy0), (sx1, sy1) = self._inward_ext[1 - sw]
        side = y0 - sy1 if sw == 0 else sy0 - y1  # D's least coordinate along w: the left foot steps toward -y
        ahead, behind = x0 - sx1, sx0 - x1
        m = INWARD_CERTIFY_M
        return side - k > need + m * (1.0 + need) or ahead - k > m or behind - k > m

    def _leg_jacobian(self, kd, side: int) -> np.ndarray:
        mujoco.mj_jacBody(self.p.m, kd, self._jacp, self._jacr, self.p.feet[side])
        return np.vstack([self._jacp, self._jacr])[:, self.leg_dofs[side]]

    def feedback(self, t_ref: float, q_ref: np.ndarray) -> None:
        """The 100 Hz update (module docstring).  Every product with a matrix and every solve is numpy's as before;
        the scalar logic runs in Python floats with numpy's semantics (``_clip``); the ZMP clip and the capture trigger
        decide from a float test against each frame's precomputed half-planes when it clears the polygon by
        ``CERTIFY_TOL`` (``_certified_inside``), else by the exact code; and the inward bound (decision 20) is computed
        only for an inward step the pads' separation does not already clear (``_inward_clear``), the only case where
        its value enters the step."""
        g = self.g
        dt_fb = FEEDBACK_EVERY * self.dt
        if self.est is not None:
            self.est.update(dt_fb)
            xaxis = self.est._views.xaxis
        else:
            xaxis = self.p._v.xaxis
        seg, ref = self.track.info(t_ref)
        st = self.state.view()
        kd, R = st["d"], st["R"]
        roll, pitch = _roll_pitch(R)
        yaw = _yaw(R)
        if self.yaw_prev is None:
            self.yaw_prev, self.yaw0_ref = yaw, float(ref["yaw"])
        self.yaw_unwrapped += _wrap(yaw - self.yaw_prev)
        self.yaw_prev = yaw
        xi = st["com"][:2] + st["com_vel"][:2] / self.track.omega
        a = int(ref["anchor"])
        pa, ya = self._anchor_frame(st, a)
        Ra = _rot2(ya)
        e = Ra.T @ (xi - pa) - self.dcm_reference(ref)
        self.e = e
        e0, e1 = e.tolist()
        dtau = np.zeros(self.nj)
        sw = int(ref["swing"])
        flags = dict(cop_clip=False, step_clip=False, leg_to_leg_clearance=False, phase_rate_saturated=False,
                     capture=False)
        if "capture" in self.terms:
            self._capture(ref, st, xi, q_ref)
        cap = self.capture
        if cap is not None:
            sw = cap["side"]
            flags["capture"] = True
        stance = [1 - sw] if sw >= 0 else [0, 1]
        # --- ankle: ZMP shift within the support, COP_INSET inside the pads
        if "ankle" in self.terms:
            kdcm, cx, cy = g["k_dcm"], g["cop_x"], g["cop_y"]
            x0, x1 = kdcm * e0, kdcm * e1
            if -cx < x0 < cx and -cy < x1 < cy:  # strictly inside both bounds: np.clip returns them unchanged
                dp = np.array([x0, x1])
            else:  # (a tie or a clip: np.clip itself, as before)
                dp = np.clip(kdcm * e, [-cx, -cy], [cx, cy])
            if cap is not None:  # single support on the capture step's stance foot
                o = stance[0]
                poly = (st["feet"][o] + self.pads_in[o] @ _rot2(float(st["feet_yaw"][o])).T - pa) @ Ra
                w = np.zeros(2)
                w[o] = 1.0
                z_rel = Ra.T @ (st["feet"][o] + _rot2(float(st["feet_yaw"][o])) @ self.pad_centre[o] - pa)
            else:
                poly = ref["poly"][: int(ref["poly_n"])]
                w = ref["weights"] if self.ds_weights == "normal" or sw >= 0 else np.array([0.5, 0.5])
                z_rel = ref["zmp_rel"]
            x = z_rel + dp
            if not self.cop_clip:
                cmd, clipped = x, False
            else:
                inside = None
                if cap is None:
                    x0, x1 = x.tolist()
                    N, d_ = ref["poly_hp"]
                    inside = _certified_inside(N, d_, x0, x1, CERTIFY_TOL * (1.0 + abs(x0) + abs(x1)))
                cmd, clipped = (x, False) if inside is True else project_into(poly, x)
            flags["cop_clip"] = clipped
            dpw0, dpw1 = (Ra @ (cmd - z_rel)).tolist()
            Fz = self.mass * G
            for i, wi in enumerate(w.tolist()):
                if wi > 0:
                    M = wi * Fz * np.array([-dpw1, dpw0, 0.0])
                    (jp, jr), (ip, ir) = self._side_jids[i]["pitch_roll"], self._side_idx[i]["pitch_roll"]
                    dtau[ip] += float(xaxis[jp] @ M)
                    dtau[ir] += float(xaxis[jr] @ M)
        # --- trunk attitude through the stance hips
        if "trunk" in self.terms:
            cy, sy = math.cos(yaw), math.sin(yaw)
            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            w0, w1, _ = (Rz.T @ st["omega"]).tolist()
            rp0, rp1 = ref["roll_pitch"].tolist()
            er, ep = roll - rp0, pitch - rp1
            kt, dt_ = g["k_trunk"], g["d_trunk"]
            Mw = Rz @ np.array([-kt * er - dt_ * w0, -kt * ep - dt_ * w1, 0.0])
            ns = len(stance)
            for s_ in stance:
                (jp, jr), (ip, ir) = self._side_jids[s_]["hips"], self._side_idx[s_]["hips"]
                dtau[ip] -= float(xaxis[jp] @ Mw) / ns
                dtau[ir] -= float(xaxis[jr] @ Mw) / ns
        # --- heading error, averaged over the last stride
        e_yaw = _wrap(self.yaw_unwrapped - (float(ref["yaw"]) - self.yaw0_ref))
        self.yaw_buf.append(e_yaw)
        period = float(ref["period"])
        nbuf = max(1, int(round(period / dt_fb)))
        if len(self.yaw_buf) > nbuf:
            del self.yaw_buf[: len(self.yaw_buf) - nbuf]
        # --- events: a touchdown keeps the swing leg's offsets, a lift-off sets the heading target
        if cap is None and sw != self.prev_sw:
            if self.prev_sw >= 0:
                self.step_land[self.prev_sw] = self.step_tgt[self.prev_sw].copy()
                self.yaw_land[self.prev_sw] = self.yaw_tgt[self.prev_sw]
            if sw >= 0:
                self.step_tgt[sw] = self.step_land[sw].copy()
                self.yaw_tgt[sw] = self.yaw_land[sw]
                if "heading" in self.terms:
                    em = float(np.mean(self.yaw_buf))
                    self.I_yaw = float(_clip(self.I_yaw + em, -0.5, 0.5))
                    ax = float(xaxis[self._side_jids[sw]["hip_yaw"]][2])
                    corr = -(g["k_yaw"] * em + g["ki_yaw"] * self.I_yaw)
                    self.yaw_tgt[sw] = corr / ax if abs(ax) > 0.2 else 0.0
            self.prev_sw = sw
        # --- step: the swing foot's touchdown offset through the swing leg's Jacobian
        u = float(ref["u"])
        if "step" in self.terms and cap is None and sw >= 0 and u < SWING_FREEZE_U:
            ks, sx, so = g["k_step"], g["step_x"], g["step_out"]
            df0, df1 = ks * e0, ks * e1
            dfx = float(_clip(df0, -sx, sx))
            out_sign = 1.0 if sw == 0 else -1.0  # the left foot's outward is +y in the anchor's frame
            lat = out_sign * df1
            if self.inward_bound == "pad_clearance" and not lat > 0.0 and not self._inward_clear(ref, sw, dfx, lat, so):
                room = self._inward_room(ref, sw, dfx)
            else:  # no bound, an outward step (-room <= 0 < lat) or a certified clear one: its value does not enter
                room = math.inf
            dfy = out_sign * float(_clip(lat, -room, so))
            flags["leg_to_leg_clearance"] = lat < -room
            flags["step_clip"] = abs(df0) > sx or lat > so
            off = np.array([dfx, dfy])
            self.step_off[sw] = off
            dfw = Ra @ off
            J = self._leg_jacobian(kd, sw)
            self.step_tgt[sw] = _solve2(J.T @ J + self._eye6, J.T @ np.concatenate((dfw, self._zeros4)))
        # --- the swing leg's offsets blend in over the swing, the stance legs' decay
        dq = np.zeros(self.nj)
        decay = min(1.0, max(0.0, 1.0 - g["decay"] * dt_fb / max(1e-6, period * seg.duty)))
        for s_ in (0, 1):
            iy = self._side_idx[s_]["hip_yaw"]
            if cap is not None and s_ == cap["side"]:
                dq[self.leg[s_]] = cap["dq"]
            elif s_ == sw:
                b = _min_jerk(u)
                dq[self.leg[s_]] = (1 - b) * self.step_land[s_] + b * self.step_tgt[s_]
                dq[iy] += (1 - b) * self.yaw_land[s_] + b * self.yaw_tgt[s_]
            else:
                self.step_land[s_] = self.step_land[s_] * decay
                self.yaw_land[s_] *= decay
                dq[self.leg[s_]] = self.step_land[s_]
                dq[iy] += self.yaw_land[s_]
        # --- phase rate
        if "phase" in self.terms and not bool(ref["stand"]) and cap is None:
            lim = PHASE_RATE_LIMIT
            self.I_x += e0 * dt_fb
            ki = g["ki_phase"]
            if ki > 0:
                self.I_x = float(_clip(self.I_x, -lim / ki, lim / ki))
            raw = g["k_phase"] * e0 + ki * self.I_x
            self.delta = float(_clip(raw, -lim, lim))
            flags["phase_rate_saturated"] = abs(raw) > lim
        else:
            self.delta = 0.0
        self.dq, self.dtau, self.flags = dq, dtau, flags

    # ------------------------------------------------------------------ capture step
    def _capture(self, ref: dict, st: dict, xi: np.ndarray, q_ref: np.ndarray) -> None:
        """Standing: trigger, swing and land a capture step (module docstring)."""
        p, kd = self.p, st["d"]
        dt_fb = FEEDBACK_EVERY * self.dt
        c = self.capture
        if c is None:
            if not bool(ref["stand"]):
                return
            pa, ya = self._anchor_frame(st, int(ref["anchor"]))
            x_rel = _rot2(ya).T @ (xi - pa)
            # a DCM inside the support by more than CERTIFY_TOL, its exact signed distance (the largest side value)
            # below the margin by as much, triggers nothing: decided without the exact computation
            x0, x1 = x_rel.tolist()
            tol = CERTIFY_TOL * (1.0 + abs(x0) + abs(x1))
            worst = -math.inf
            for (n0, n1), di in zip(*ref["poly_raw_hp"], strict=True):
                s = n0 * x0 + n1 * x1 - di
                worst = s if (s > worst or s != s) else worst
            if worst < -tol and worst + tol <= self.g["capture_margin"]:
                return
            out = signed_distance(ref["poly_raw"][: int(ref["poly_raw_n"])], x_rel)
            if out <= self.g["capture_margin"]:
                return
            width = float(np.linalg.norm(st["feet"][0] - st["feet"][1]))
            lat = float(x_rel[1])
            sideways = abs(lat) > 0.25 * width
            side = (0 if lat > 0 else 1) if sideways else int(np.argmin(np.asarray(ref["weights"])))
            other = 1 - side
            # where the DCM will be at touchdown (LIPM, diverging from the support's edge it left, along the direction
            # it left in: the stance foot's lateral CoP shift is not modelled), and the stepping foot's pad centre
            # lands there; sideways the stepping foot keeps at least CAPTURE_MIN_WIDTH of the stance width
            n_r = int(ref["poly_raw_n"])
            edge, _ = project_into(ref["poly_raw"][:n_r], x_rel)
            xi_td_rel = edge + (x_rel - edge) * math.exp(self.track.omega * CAPTURE_SWING_S)
            xi_td = pa + _rot2(ya) @ xi_td_rel
            yaw_f = float(st["feet_yaw"][side])
            to = xi_td - _rot2(yaw_f) @ self.pad_centre[side]
            nrm = _rot2(ya) @ np.array([0.0, 1.0 if side == 0 else -1.0])  # outward for the stepping foot
            if not sideways:  # a push along the feet: the foot steps along them, keeping its place sideways
                to = to + float((st["feet"][side] - to) @ nrm) * nrm
            gap = float((to - st["feet"][other]) @ nrm)
            need = CAPTURE_MIN_WIDTH * width
            if gap < need:
                to = to + (need - gap) * nrm
            self.capture = c = dict(side=side, t=0.0, landed=0.0, frm=st["feet"][side].copy(), to=to, yaw=yaw_f,
                                    z0=float(kd.xpos[p.feet[side]][2]), dq=self.step_land[side].copy())
            self.capture_log.append(dict(t_s=float(p.d.time), side=statics.SIDES[side], dcm_outside_m=out,
                                         target=[float(v) for v in to]))
        c["t"] += dt_fb
        u = min(1.0, c["t"] / CAPTURE_SWING_S)
        b = _min_jerk(u)
        target = np.r_[(1 - b) * c["frm"] + b * c["to"], c["z0"] + CAPTURE_LIFT_H * self.H * math.sin(math.pi * u) ** 2]
        # IK of the swing leg at the current trunk pose, from the commanded angles: the offsets to the reference
        side = c["side"]
        ik = self._ik
        ik.qpos[:] = kd.qpos
        ik.qpos[self.leg_qadr[side]] = q_ref[self.leg[side]] + c["dq"]
        R_t = np.array([[math.cos(c["yaw"]), -math.sin(c["yaw"]), 0.0], [math.sin(c["yaw"]), math.cos(c["yaw"]), 0.0],
                        [0.0, 0.0, 1.0]])
        for _ in range(3):
            mujoco.mj_kinematics(p.m, ik)
            mujoco.mj_comPos(p.m, ik)
            f = p.feet[side]
            Rf = ik.xmat[f].reshape(3, 3)
            E = R_t @ Rf.T
            err = np.r_[target - ik.xpos[f], 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])]
            mujoco.mj_jacBody(p.m, ik, self._jacp, self._jacr, f)
            J = np.vstack([self._jacp, self._jacr])[:, self.leg_dofs[side]]
            ik.qpos[self.leg_qadr[side]] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), err)
        c["dq"] = ik.qpos[self.leg_qadr[side]] - q_ref[self.leg[side]]
        if u >= 1.0:
            c["landed"] += dt_fb
            if c["landed"] >= 0.2:  # standing on the new stance: its offsets become the leg's held offsets
                self.step_land[side] = c["dq"].copy()
                self.yaw_land[side] = 0.0
                self.capture = None


# ------------------------------------------------------------------------------------------------
# Episodes
# ------------------------------------------------------------------------------------------------
#: MuJoCo's integration state: what ``mj_step`` depends on (physics, inputs, the solver's warm start).
MJ_STATE = mujoco.mjtState.mjSTATE_INTEGRATION
#: The estimator's attributes that are configuration, not state (``Tracker.save``).
EST_STATIC = frozenset({"p", "g", "m", "d", "dt", "weight", "trunk", "feet", "r_imu", "qadr", "dofs", "_res",
                        "foot_z0", "sole", "_views", "_qs", "_vs", "_Rb"})
#: Plan section 4's hardware limits as decisions 15 and 20 leave them, measured at every feedback update (module
#: docstring).
HW_LIMITS = ("leg_drive_at_torque_limit", "leg_drive_at_envelope_limit", "ankle_linkage", "slip",
             "cop_at_pad_edge_with_ankle_saturated", "leg_to_leg_clearance", "joint_range")
#: Recorded beside them, not hardware limits (module docstring): the cap's per-axis box is the controller's own.
REPORTED_LIMITS = ("neck_tail_at_torque_limit", "ankle_cap_box", "step_clip", "phase_rate_saturated")
#: The relaxations that lift hardware limits (``relaxed``), and the one that lifts each limit (None: not relaxable).
RELAXATIONS = ("drive_torque", "envelope", "ankle_linkage", "slip", "leg_to_leg_clearance", "joint_range")
RELAXES = {"leg_drive_at_torque_limit": "drive_torque", "leg_drive_at_envelope_limit": "envelope",
           "ankle_linkage": "ankle_linkage", "slip": "slip", "cop_at_pad_edge_with_ankle_saturated": None,
           "leg_to_leg_clearance": "leg_to_leg_clearance", "joint_range": "joint_range"}
_NO_SLIP = np.zeros(2)  # (read only) no loaded foot: no slip speed
#: The realised ankle torques against decision 15's cap (``Tracker.cap_residual``), at every feedback update.
CAP_RESIDUAL_TOL_NM = 1e-6
CAP_RESIDUAL_HEAVY_FRAC = 0.2  # R2's contact detection switches on at this share of the weight (EstimatorGains)
CAP_RESIDUAL_RULE = ("at every feedback update, the realised ankle torques (the ankle motors' forces in pitch and roll, "
                     "G^T f) against decision 15's cap on the true state: an axis beyond one motor's peak, and a foot "
                     f"carrying more than {CONTACT_LOAD_FRAC:g} of the weight (its true load; also reported for "
                     f"{CAP_RESIDUAL_HEAVY_FRAC:g}, where R2's contact detection switches on) whose torques put its "
                     f"quasi-static CoP outside its pad's {1e3 * COP_INSET:g} mm inset, measured as the distance from the "
                     "torques to the nearest torques that keep the CoP inside (N m; control.cap_ankle_torque); an "
                     f"excess over {CAP_RESIDUAL_TOL_NM:g} N m counts.  The check uses the loads and kinematics of the "
                     "step the torques acted in; the cap had those of the step before (what the sensors give), so a "
                     "load that changes within a step leaves a residual even for R1, which caps on the true joint "
                     "state; R2 caps on its last sensor reading and pad-limits only the feet its estimator detects in "
                     "contact")
#: How a speed command reaches the tracker (``Tracker.rate``).
COMMAND_RATE_RULE = ("a command of r times the reference's speed plays the track r times faster than real time (the "
                     "reference clock's rate r, the phase-rate term's +-5 % about it): cadence, speed and yaw rate times "
                     "r, stride and footprints unchanged, the reference rates times r and accelerations times r^2 in the "
                     "servo targets and the armature feedforward, the reference DCM c + r c' / omega0 (the CoM moving r "
                     "times as fast), the inverse-dynamics torque and the reference ZMP as the reference's")
FALL_ONSET_RAD = 0.2  # the fall's onset: the last time the trunk tilt rose through this before the fall
HIST_S = 3.0  # how far back before a fall its onset is looked for
LIMIT_NEAR_S = 0.1  # the activity in this last part of the window before the onset is reported as well
CF_EVERY = 25  # physics steps between the snapshots a counterfactual branches from (50 ms)
CF_BRANCH_S = 1.0  # a counterfactual branches this long before the fall's onset
CF_KEEP_S = HIST_S + CF_BRANCH_S + 0.5  # how long snapshots are kept
CF_HORIZON_S = 1.0  # a counterfactual runs until this long after the original fall
CF_NULL_M_S = 1e-3  # the null counterfactual: the root velocity changed by this (world x) at the branch
CF_FRICTION = 5.0  # the "slip" relaxation: sliding friction of every colliding geom
CF_RANGE_RAD = 1.0  # the "joint_range" relaxation: every hinge's range widened by this on each side
#: The candidates of a fall (decision 20 attributes conditions, not falls; ``gates.attribute``).
CANDIDATE_RULE = (f"the hardware limits (HW_LIMITS) active at an update of the {LIMIT_WINDOW_S:g} s before the fall's "
                  f"onset (the last time the trunk tilt rose through {FALL_ONSET_RAD:g} rad)")
#: The build's episode-level counterfactual (``run``'s ``confirm``), a diagnostic since decision 20.
EPISODE_CF_RULE = (f"a candidate binds if the episode, restored to {CF_BRANCH_S:g} s before the onset and continued "
                   f"with that limit relaxed, has not fallen and is upright (tilt below the onset's) {CF_HORIZON_S:g} s "
                   "after the original fall: 'body-limited: <limits>' for the candidates whose relaxation alone does "
                   "that, '... together' if only the candidates relaxed together do, else 'controller-limited'; "
                   "'undetermined' if the unchanged continuation does not repeat the fall or if a "
                   f"{1e3 * CF_NULL_M_S:g} mm/s change of the root velocity at the branch avoids it too; 'unconfirmed' "
                   "if only a limit the plant cannot relax was a candidate (a diagnostic: decision 20 attributes "
                   "conditions over their starts)")


def set_friction(p: plant.Plant, mu: float) -> None:
    """Every colliding geom's sliding friction set to ``mu`` (MuJoCo takes the larger of a pair's, so the pads' floor
    contacts get ``mu``), as the design-study prototype set it."""
    cg = (p.m.geom_contype | p.m.geom_conaffinity) > 0
    p.m.geom_friction[cg, 0] = mu


def foot_geoms(p: plant.Plant) -> np.ndarray:
    """Per geom of ``p.m``: the foot it belongs to (0 left, 1 right), else -1."""
    out = np.full(p.m.ngeom, -1)
    for i, f in enumerate(p.feet):
        out[p.m.geom_bodyid == f] = i
    return out


def _normal_forces(m, d, idx: np.ndarray, contact=None) -> np.ndarray:
    """The normal force of each contact ``idx`` (``contact``: ``d.contact``, when the caller has it)."""
    if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):  # the cone's first row is the normal force
        adr = np.asarray((d.contact if contact is None else contact).efc_address)[idx]
        return np.where(adr >= 0, d.efc_force[np.maximum(adr, 0)], 0.0)
    f6 = np.zeros(6)
    out = np.zeros(len(idx))
    for j, i in enumerate(idx):
        mujoco.mj_contactForce(m, d, int(i), f6)
        out[j] = f6[0]
    return out


def foot_slip_speed(p: plant.Plant, feet=(0, 1), foot_of: np.ndarray | None = None) -> np.ndarray:
    """Per foot (left, right): the largest horizontal speed of the foot, relative to what it touches, at its contacts
    that carry load (positive normal force); 0 for a foot without one or not in ``feet``.  As of the last step's start,
    like the sensors.  A foot that rocks on its pad's edge or rolls over it moves its body (its origin on the ankle
    axis is 20-25 mm above the floor) but not its loaded contacts, so it does not slip here."""
    m, d = p.m, p.d
    out = np.zeros(2)
    n = d.ncon
    if not n:
        return out
    foot_of = foot_geoms(p) if foot_of is None else foot_of
    con = d.contact
    g = np.asarray(con.geom)[:n]
    f1, f2 = foot_of[g[:, 0]], foot_of[g[:, 1]]
    idx = np.flatnonzero((f1 >= 0) != (f2 >= 0))  # a foot against something else (not foot on foot)
    if not len(idx):
        return out
    is1 = f1[idx] >= 0
    foot = np.where(is1, f1[idx], f2[idx])
    other = m.geom_bodyid[np.where(is1, g[idx, 1], g[idx, 0])]
    loaded = _normal_forces(m, d, idx, con) > 0.0
    pos = np.asarray(con.pos)[:n][idx]
    xpos = p._v.xpos if hasattr(p, "_v") else d.xpos
    v6 = np.zeros(6)
    for f in feet:
        sel = np.flatnonzero((foot == f) & loaded)
        if not len(sel):
            continue
        b = p.feet[f]
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, v6, 0)  # at the body frame's origin, xpos
        v = v6[3:] + est_mod.cross3(v6[:3], pos[sel] - xpos[b])
        for j, ob in enumerate(other[sel]):
            if ob != 0:  # a moving body (a pile probe): less its own velocity at the contact
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, int(ob), v6, 0)
                v[j] -= v6[3:] + est_mod.cross3(v6[:3], pos[sel[j]] - xpos[ob])
        out[f] = float(np.hypot(v[:, 0], v[:, 1]).max())  # (np.max's own reduction)
    return out


@contextmanager
def relaxed(tk: Tracker, names):
    """The plant with the hardware limits ``names`` (``RELAXATIONS``) lifted: "drive_torque" the leg drives' torque
    limits (peak and envelope), "envelope" the torque-speed envelope (every drive at its peak at any speed),
    "ankle_linkage" the ankle motors' torque limits, "slip" friction ``CF_FRICTION`` on every colliding geom,
    "leg_to_leg_clearance" the inward step bound (``Tracker.inward_bound`` None) with the two feet's pads no longer
    colliding with each other (they still meet the floor), "joint_range" every hinge's range widened by
    ``CF_RANGE_RAD`` on each side.  The tracker's own limits (the ankle cap, the step clips) stay.  Restored on exit."""
    names = set(names)
    if names - set(RELAXATIONS):
        raise ValueError(f"unknown relaxations {sorted(names - set(RELAXATIONS))}: choose from {', '.join(RELAXATIONS)}")
    p = tk.p
    m, env = p.m, p.envelope
    saved = (m.actuator_forcelimited.copy(), m.geom_friction.copy(), m.jnt_range.copy(), env.enabled, tk.inward_bound,
             m.geom_contype.copy())
    try:
        if "envelope" in names:
            env.enabled = False
        if "drive_torque" in names:
            m.actuator_forcelimited[tk.leg_act] = 0
        if "ankle_linkage" in names:
            m.actuator_forcelimited[tk.ankle_act] = 0
        if "slip" in names:
            m.geom_friction[(m.geom_contype | m.geom_conaffinity) > 0, 0] = CF_FRICTION
        if "leg_to_leg_clearance" in names:
            tk.inward_bound = None
            pads = (tk.foot_of >= 0) & (m.geom_contype == 1) & (m.geom_conaffinity == 1)
            m.geom_contype[pads] = 2  # pad on floor: floor contype 1 & pad conaffinity 1; pad on pad: none
        if "joint_range" in names:
            m.jnt_range[tk.hinge_jid, 0] -= CF_RANGE_RAD
            m.jnt_range[tk.hinge_jid, 1] += CF_RANGE_RAD
        yield
    finally:
        m.actuator_forcelimited[:] = saved[0]
        m.geom_friction[:] = saved[1]
        m.jnt_range[:] = saved[2]
        env.enabled = saved[3]  # (and the force ranges back to the peaks)
        tk.inward_bound = saved[4]
        m.geom_contype[:] = saved[5]


def _apply_push(p: plant.Plant, push) -> None:
    if push is not None:
        p.d.xfrc_applied[p.trunk, :2] = push[2:4] if push[0] <= p.d.time < push[0] + push[1] else 0.0


def _continue(tk: Tracker, snap: dict, until_s: float, push, null: bool = False) -> dict:
    """``snap`` restored (with ``null`` the root velocity changed by ``CF_NULL_M_S``) and run on to ``until_s`` or a
    fall; avoided: no fall and the tilt from home below ``FALL_ONSET_RAD`` at the end."""
    p = tk.p
    d = p.d
    tk.load(snap)
    if null:
        d.qvel[0] += CF_NULL_M_S
    cause = fell = None
    for _ in range(max(0, int(round((until_s - float(d.time)) / tk.dt)))):
        _apply_push(p, push)
        tk.step()
        cause = p.fall_cause()
        if cause is not None:
            fell = float(d.time)
            break
    tilt = p.tilt_from_home()
    return dict(fell=fell is not None, fall_time_s=fell, fall_cause=cause, tilt_end_deg=math.degrees(tilt),
                avoided=bool(fell is None and tilt < FALL_ONSET_RAD), q_end=d.qpos.copy())


def _confirm(tk: Tracker, ring, final: dict, onset: float, fell_at: float, cause: str, q_fall: np.ndarray, push,
             candidates: list[str]) -> dict:
    """The episode-level counterfactual test of a fall's candidate limits (a diagnostic, ``EPISODE_CF_RULE``);
    ``tk`` is back at the fall after it."""
    t0 = time.perf_counter()
    t_branch = onset - CF_BRANCH_S
    snaps = list(ring)
    snap = next((s for s in reversed(snaps) if s["time"] <= t_branch + 1e-9), snaps[0] if snaps else None)
    tests = [r for r in RELAXATIONS if any(RELAXES[c] == r for c in candidates)]
    untested = [c for c in candidates if RELAXES[c] is None]
    until = fell_at + CF_HORIZON_S
    out = dict(candidates=list(candidates), relaxations=tests, not_relaxable=untested,
               branch_s=None if snap is None else snap["time"], branch_wanted_s=t_branch, horizon_s=until)

    def strip(r):
        return {k: v for k, v in r.items() if k != "q_end"}

    if snap is None:
        out.update(reproduces=False, label="undetermined: no snapshot to branch from")
        return out
    base = _continue(tk, snap, until, push)
    out["reproduces"] = bool(base["fell"] and base["fall_time_s"] == fell_at and base["fall_cause"] == cause
                             and np.array_equal(base["q_end"], q_fall))
    if not out["reproduces"]:
        tk.load(final)
        out.update(unchanged=strip(base), label="undetermined: the restored episode did not repeat the fall",
                   wall_s=time.perf_counter() - t0)
        return out
    out["null"] = strip(_continue(tk, snap, until, push, null=True))
    alone = {}
    for r in tests:
        with relaxed(tk, (r,)):
            alone[r] = strip(_continue(tk, snap, until, push))
    together = None
    if len(tests) > 1:
        with relaxed(tk, tests):
            together = strip(_continue(tk, snap, until, push))
    with relaxed(tk, RELAXATIONS):
        every = strip(_continue(tk, snap, until, push))
    tk.load(final)
    binding = [r for r in tests if alone[r]["avoided"]]
    if out["null"]["avoided"]:
        label = "undetermined: chaotic (the null change avoids the fall)"
    elif binding:
        label = "body-limited: " + ", ".join(binding)
    elif together is not None and together["avoided"]:
        label = "body-limited: " + " + ".join(tests) + " together"
    elif untested:
        label = "unconfirmed: " + ", ".join(untested) + " (not relaxable in the plant)"
    else:
        label = "controller-limited"
    out.update(alone=alone, together=together, every_relaxation=every, binding=binding, label=label,
               wall_s=time.perf_counter() - t0)
    return out


def _activity(hist: list, onset: float, drives) -> dict:
    """The limits active in the window before the onset (fractions of its feedback updates)."""
    win = [h for h in hist if onset - LIMIT_WINDOW_S <= h[0] <= onset]
    near = [h for h in win if h[0] >= onset - LIMIT_NEAR_S]

    def frac(rows, col, j):
        return float(np.mean([h[col][j] for h in rows])) if rows else 0.0

    def last(j):
        t = [h[0] for h in win if h[2][j]]
        return float(onset - max(t)) if t else None

    return dict(window_s=LIMIT_WINDOW_S, updates=len(win), near_onset_s=LIMIT_NEAR_S,
                hardware={k: frac(win, 2, j) for j, k in enumerate(HW_LIMITS)},
                hardware_near_onset={k: frac(near, 2, j) for j, k in enumerate(HW_LIMITS)},
                hardware_last_active_before_onset_s={k: last(j) for j, k in enumerate(HW_LIMITS)},
                reported={k: frac(win, 3, j) for j, k in enumerate(REPORTED_LIMITS)},
                leg_drives=sorted({drives[i].actuator for h in win for i in h[4]}),
                neck_tail_drives=sorted({drives[i].actuator for h in win for i in h[5]}))


def _estimator_stats(E: np.ndarray) -> dict:
    """Estimator errors over rows (t, CoM relative to the feet m, CoM velocity m/s, tilt rad, yaw rad, CoM height m)."""
    return dict(updates=len(E), com_rel_feet_err_rms_mm=1e3 * float(np.sqrt(np.mean(E[:, 1] ** 2))),
                com_rel_feet_err_max_mm=1e3 * float(E[:, 1].max()),
                com_vel_err_rms_m_s=float(np.sqrt(np.mean(E[:, 2] ** 2))),
                tilt_err_rms_deg=float(np.degrees(np.sqrt(np.mean(E[:, 3] ** 2)))),
                tilt_err_max_deg=float(np.degrees(E[:, 3].max())), yaw_err_end_deg=float(np.degrees(E[-1, 4])),
                com_height_err_rms_mm=1e3 * float(np.sqrt(np.mean(E[:, 5] ** 2))))


def _stop_sample(tk: Tracker, stops: list, yaw: float) -> None:
    """At a feedback update: a stop's command is reached once the reference clock passes it (its forward direction
    the trunk's heading then); after it, the true CoM's speed along that direction and its horizontal speed."""
    p = tk.p
    t_ref = tk.tau_ref if tk.mode != "R0" else float(p.d.time)
    for s in stops:
        if s["t_sim"] is None and t_ref >= s["t_ref"] - 1e-12:
            s["t_sim"], s["fwd"] = float(p.d.time), np.array([math.cos(yaw), math.sin(yaw)])
    live = [s for s in stops if s["t_sim"] is not None]
    if live:
        mujoco.mj_subtreeVel(p.m, p.d)
        v = p.d.subtree_linvel[p.trunk][:2]
        s = live[-1]  # samples belong to the latest stop reached
        s["samples"].append((float(p.d.time), float(v @ s["fwd"]), float(np.hypot(v[0], v[1]))))


def rest_speed(mode: str) -> float:
    """The CoM speed below which an episode's body counts as at rest after a stop (decision 18's reported time to
    rest): ``transitions.REST_SPEED_M_S`` (1 mm/s) for R0 and R1, ``transitions.REST_SPEED_R2_M_S`` for R2, whose
    sensor noise keeps a standing body moving faster than 1 mm/s (``transitions.REST_R2_RULE``)."""
    from .transitions import REST_SPEED_M_S, REST_SPEED_R2_M_S

    return REST_SPEED_R2_M_S if mode == "R2" else REST_SPEED_M_S


def _stop_result(s: dict, rate: float = 1.0, rest: float | None = None) -> dict:
    """Decision 18's reading of one stop of an episode (None values: not reached, or not stopped by the end).  On a
    track run at the commanded clock rate ``rate`` the command is the reference's speed times the rate and a stride
    its period over the rate; ``rest`` is the rest speed (default ``transitions.REST_SPEED_M_S``)."""
    from .transitions import REST_R2_RULE, REST_SPEED_M_S, REST_SPEED_R2_M_S, stop_reading

    rest = REST_SPEED_M_S if rest is None else float(rest)
    speed, period = s["speed_m_s"] * rate, s["period_s"] / rate  # (rate 1: exactly the reference's)
    head = dict(name=s["name"], command_ref_s=s["t_ref"], command_s=s["t_sim"], command_speed_m_s=speed,
                rest_speed_m_s=rest, rest_rule=REST_R2_RULE if rest == REST_SPEED_R2_M_S else None)
    if s["t_sim"] is None or not s["samples"]:
        return dict(head, reached=False)
    t, v, vh = (np.array(x) for x in zip(*s["samples"], strict=True))
    out = stop_reading(t, v, speed, s["t_sim"], period, rest=vh < rest)
    return dict(head, reached=True, samples=len(t), **out)


def run(tk: Tracker, seconds: float, start: str = "frame", velocity_offset=None, push=None, seed: int | None = None,
        period: float | None = None, confirm: bool = False, relax=(), recorder=None, joint_noise: float = 0.0,
        root_velocity_noise: float = 0.0, noise_seed: int | None = None) -> dict:
    """One episode: reset, then ``seconds`` of simulated time or until a fall.  ``push`` = (t0, duration, fx, fy) N
    on the trunk.  Returns the outcome; a fall's cause (``plant.Plant.fall_cause``), its onset (the last time the
    tilt rose through ``FALL_ONSET_RAD``), the limits active in the ``LIMIT_WINDOW_S`` before the onset and the
    fall's candidates (``CANDIDATE_RULE``; decision 20 attributes conditions, ``gates.attribute``); tracking and drive
    statistics, how often the ankle cap and the inward bound bound, and how far the realised ankle torques broke the
    cap (``CAP_RESIDUAL_RULE``: R2 caps from its sensor readings); each stop's reading (decision 18,
    ``transitions.stop_reading``) when the track holds a stop; for R2 the estimator's errors against the true state,
    over the episode and, for a fall, up to the window before its onset.  ``period`` is the stride count's unit
    (default: the first periodic segment's period; strides are of the commanded gait, the period over the tracker's
    ``rate``).  ``relax`` lifts hardware limits (``relaxed``) for the whole episode.  ``confirm`` runs the build's
    episode-level counterfactual on a fall (a diagnostic, ``EPISODE_CF_RULE``) and labels the fall by it.
    ``recorder`` (``metrics.Recorder``) is handed every physics step and feedback update and changes nothing.  A
    ``"stance"`` start takes ``joint_noise`` (rad) and ``root_velocity_noise`` (m/s), drawn from the episode's seed
    (``plant.Plant.reset``); R2's sensor noise is drawn from ``noise_seed`` (default: the seed)."""
    tk.reset(start, velocity_offset=velocity_offset, joint_noise=joint_noise, seed=seed,
             root_velocity_noise=root_velocity_noise, noise_seed=noise_seed)
    relax = tuple(relax or ())
    if relax:
        confirm = False
    with relaxed(tk, relax) if relax else nullcontext():
        out, _, fall = _episode(tk, seconds, push, period, confirm, recorder)
    if fall is not None and confirm:
        cf = _confirm(tk, *fall)
        out.update(fall_label=cf["label"], counterfactual=cf)
    out.update(start=start, relaxed=list(relax))
    return out


def _episode(tk: Tracker, seconds: float, push, period: float | None, confirm: bool, recorder=None):
    """``run``'s episode loop (the tracker reset): the outcome, the fall's onset, and with ``confirm`` what
    ``_confirm`` needs of a fall.  ``recorder`` gets ``begin``, every physics step's ``step``, every feedback
    update's ``update`` and ``end`` (it only reads)."""
    p = tk.p
    m, d, env = p.m, p.d, p.envelope
    per_ref = period or next((sg.period for sg in tk.track.segments if sg.kind == "periodic"), 1.0)
    per = per_ref / tk.rate  # a stride of the commanded gait (rate 1: the reference's period)
    n = int(round(seconds / tk.dt))
    capped = bool(tk.ankle_cap) and tk.mode != "R0"
    cap_res = np.zeros(3)  # the realised ankle torques' largest box, pad and heavy-foot pad excess (N m)
    cap_res_n = np.zeros(5, int)  # updates with a box excess; loaded feet and those over the pad; the same, heavy
    if recorder is not None:
        recorder.begin(tk, n)
    nu = m.nu
    peak = env.peak
    tau_sq, w, wmax, sat_steps = np.zeros(nu), np.zeros(nu), np.zeros(nu), np.zeros(nu)
    pairs = [(tk.idx[f"{s}_ankle_pitch"], tk.idx[f"{s}_ankle_roll"]) for s in statics.SIDES]  # their motors a, b
    hist = deque(maxlen=int(round(HIST_S / (FEEDBACK_EVERY * tk.dt))))  # per feedback update: time, tilt, limits
    ring = deque(maxlen=max(1, int(math.ceil(CF_KEEP_S / (CF_EVERY * tk.dt)))))  # snapshots (counterfactuals)
    bufs = [np.empty(mujoco.mj_stateSize(m, MJ_STATE)) for _ in range(2)]
    leg, neck_tail = tk.leg_act, ~tk.leg_act
    weight = tk.mass * G
    x0 = d.qpos[:2].copy()
    R_home = p._R_home
    yaw_prev, yaw_turned = _yaw(d.xmat[p.trunk]), 0.0
    tilts, rolls, pitches, deltas, est_err, e_dcm = [], [], [], [], [], []
    fell_at = cause = q_fall = final = None
    clamp = np.zeros(nu, bool)
    sat_any = np.zeros(nu, bool)
    bound = np.zeros(3)  # feedback intervals with the cap's box, its pad, the inward bound binding
    stops = [dict(ev, t_sim=None, samples=[]) for ev in tk.track.events if ev["kind"] == "stop"]
    tk.cap_bound = [False, False]
    # the same per-step arithmetic as before, into preallocated buffers and through views kept for the episode
    v = p._v
    qvel, f, xmat, xpos = v.qvel, v.actuator_force, v.xmat, v.xpos
    frange = m.actuator_forcerange
    sq, aw = np.empty(nu), np.empty(nu)
    hi_t, lo_t = np.empty(nu), np.empty(nu)
    b1, b2, sat = np.empty(nu, bool), np.empty(nu, bool), np.empty(nu, bool)
    slip_load = SLIP_LOAD_FRAC * weight
    jlo, jhi = tk.lo + 1e-3, tk.hi - 1e-3
    trunk, feet = p.trunk, p.feet
    qadr = p.qadr
    t0 = time.perf_counter()
    k = 0
    for k in range(n):
        if push is not None:
            _apply_push(p, push)
        if confirm:
            bufs.reverse()  # bufs[0]: the state at this step's start, bufs[1]: at the last one's
            mujoco.mj_getState(m, d, bufs[0], MJ_STATE)
            if k % CF_EVERY == 1:
                ring.append(tk.save(bufs[1]))
        np.dot(env.moment, qvel, out=w)
        tk.step()
        np.multiply(f, f, out=sq)
        np.add(tau_sq, sq, out=tau_sq)  # tau_sq += f * f
        np.abs(w, out=aw)
        np.maximum(wmax, aw, out=wmax)
        np.subtract(frange[:, 1], 1e-9, out=hi_t)
        np.add(frange[:, 0], 1e-9, out=lo_t)
        np.greater_equal(f, hi_t, out=b1)
        np.less_equal(f, lo_t, out=b2)
        np.logical_or(b1, b2, out=sat)  # sat = (f >= fr[:, 1] - 1e-9) | (f <= fr[:, 0] + 1e-9)
        np.add(sat_steps, sat, out=sat_steps)
        np.logical_or(sat_any, sat, out=sat_any)
        if env.active:
            clamp |= sat & ((frange[:, 1] < peak - 1e-12) | (frange[:, 0] > -peak + 1e-12))
        if recorder is not None:
            recorder.step(k)
        if k % FEEDBACK_EVERY == 0:
            qj = v.qpos[qadr]
            loads = p.foot_loads()
            ll = loads.tolist()
            loaded = [i for i in (0, 1) if ll[i] > slip_load]
            slip_v = foot_slip_speed(p, loaded, tk.foot_of) if loaded else _NO_SLIP
            slip = bool(slip_v[0] > SLIP_SPEED or slip_v[1] > SLIP_SPEED)
            if capped:
                box_x, pad_x, heavy_x, n_loaded, n_heavy = tk.cap_residual(loads)
                np.maximum(cap_res, (box_x, pad_x, heavy_x), out=cap_res)
                cap_res_n += (box_x > CAP_RESIDUAL_TOL_NM, n_loaded, pad_x > CAP_RESIDUAL_TOL_NM, n_heavy,
                              heavy_x > CAP_RESIDUAL_TOL_NM)
            fl = tk.flags
            sa = sat_any.tolist()
            ankle_sat = any(sa[i] or sa[j_] for i, j_ in pairs)
            cap_box, cap_pad = tk.cap_bound
            tk.cap_bound = [False, False]
            inward = bool(fl.get("leg_to_leg_clearance"))
            bound += (cap_box, cap_pad, inward)
            cop_edge = cap_pad or (bool(fl.get("cop_clip")) and ankle_sat)
            R = xmat[trunk].reshape(3, 3)
            Rr = R @ R_home.T
            tilt = float(np.arccos(_clip(float(Rr[2, 2]), -1.0, 1.0)))  # _clip: np.clip of a float, bit for bit
            sat_leg = sat_any & leg
            sat_nt = sat_any & neck_tail
            hw = (bool(sat_leg.any()), bool((clamp & leg).any()), any(sa[i] and sa[j_] for i, j_ in pairs), slip,
                  cop_edge, inward, bool(((qj <= jlo) | (qj >= jhi)).any()))
            rep = (bool(sat_nt.any()), cap_box, bool(fl.get("step_clip")), bool(fl.get("phase_rate_saturated")))
            hist.append((float(d.time), tilt, hw, rep, tuple(np.flatnonzero(sat_leg)), tuple(np.flatnonzero(sat_nt))))
            clamp[:] = False
            sat_any[:] = False
            tilts.append(tilt)
            r_, p_ = _roll_pitch(Rr)
            rolls.append(r_)
            pitches.append(p_)
            deltas.append(tk.delta)
            e_ = tk.e
            e_dcm.append(math.sqrt(e_.dot(e_)))  # np.linalg.norm of a contiguous 2-vector, bit for bit
            yaw = _yaw(R)
            yaw_turned += _wrap(yaw - yaw_prev)
            yaw_prev = yaw
            if recorder is not None:
                recorder.update(k, loads, slip_v, Rr, yaw)
            if stops:
                _stop_sample(tk, stops, yaw)
            if tk.est is not None:
                mujoco.mj_subtreeVel(m, d)
                es = tk.est
                sc, sl = v.subtree_com[trunk], v.subtree_linvel[trunk]
                ft = np.array([xpos[feet[0]][:2], xpos[feet[1]][:2]])
                z0, z1 = float(xpos[feet[0]][2]), float(xpos[feet[1]][2])
                r1 = (es.com[:2] - es.foot[:, :2].mean(0)) - (sc[:2] - ft.mean(0))
                r2 = es.com_vel[:2] - sl[:2]
                est_err.append((float(d.time), math.sqrt(r1.dot(r1)), math.sqrt(r2.dot(r2)),
                                float(np.arccos(_clip(float(es.R[:, 2] @ R[:, 2]), -1.0, 1.0))),
                                abs(_wrap(_yaw(es.R) - yaw)), float(es.com[2] - es.foot[:, 2].mean())
                                - float(sc[2] - (z0 + z1) / 2)))
        cause = p.fall_cause()
        if cause is not None:
            fell_at = float(d.time)
            if confirm:
                q_fall, final = d.qpos.copy(), tk.save(bufs[0])
            break
    wall = time.perf_counter() - t0
    steps = k + 1
    dur = float(d.time)
    heading0 = _yaw(R_home)
    dxy = d.qpos[:2] - x0
    fwd = np.array([math.cos(heading0), math.sin(heading0)])
    rms = np.sqrt(tau_sq / steps)
    sustained = np.array([dr.sustained for dr in p.drives])
    ratio_w = wmax / env.rated
    out = dict(mode=tk.mode, terms=sorted(tk.terms), start=tk.start, rate=tk.rate, seconds=dur, strides=dur / per,
               reference_strides=(tk.tau_ref if tk.mode != "R0" else dur) / per_ref, fell=fell_at is not None,
               fall_time_s=fell_at, fall_cause=cause, fall_onset_s=None, fall_label=None, limits_before_onset=None,
               counterfactual=None, mean_forward_speed_m_s=float(dxy @ fwd / max(dur, 1e-9)),
               lateral_drift_m=float(dxy @ np.array([-fwd[1], fwd[0]])), yaw_turned_deg=float(np.degrees(yaw_turned)),
               mean_yaw_rate_rad_s=yaw_turned / max(dur, 1e-9),
               tilt_from_home_max_deg=float(np.degrees(max(tilts))) if tilts else 0.0,
               roll_rms_deg=float(np.degrees(np.sqrt(np.mean(np.square(rolls))))) if rolls else 0.0,
               pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(np.square(pitches))))) if pitches else 0.0,
               roll_peak_deg=float(np.degrees(np.max(np.abs(rolls)))) if rolls else 0.0,
               pitch_peak_deg=float(np.degrees(np.max(np.abs(pitches)))) if pitches else 0.0,
               dcm_error_rms_mm=1e3 * float(np.sqrt(np.mean(np.square(e_dcm)))) if e_dcm else 0.0,
               phase_rate_range=[float(min(deltas)), float(max(deltas))] if deltas else [0.0, 0.0],
               saturated_frac={p.drives[i].actuator: float(sat_steps[i] / steps) for i in np.flatnonzero(sat_steps)},
               worst_sustained_ratio=[p.drives[int(np.argmax(rms / sustained))].actuator, float(np.max(rms / sustained))],
               worst_speed_ratio=[p.drives[int(np.argmax(ratio_w))].actuator, float(np.max(ratio_w))],
               captures=list(tk.capture_log), segment_reached=tk.track.segment(tk.tau_ref if tk.mode != "R0" else dur).name,
               reference_time_s=float(tk.tau_ref if tk.mode != "R0" else dur), fall_candidates=None,
               ankle_cap=list(tk.ankle_cap), inward_bound=tk.inward_bound,
               bound_frac=dict(zip(("ankle_cap_box", "ankle_cap_pad", "leg_to_leg_clearance"),
                                   (float(v) for v in bound / max(1, len(tilts))), strict=True)),
               ankle_cap_residual=None if not capped else dict(
                   rule=CAP_RESIDUAL_RULE, updates=len(tilts), box_updates=int(cap_res_n[0]),
                   box_frac=float(cap_res_n[0] / max(1, len(tilts))), box_max_excess_Nm=float(cap_res[0]),
                   loaded_foot_updates=int(cap_res_n[1]), pad_updates=int(cap_res_n[2]),
                   pad_frac=float(cap_res_n[2] / max(1, len(tilts))), pad_max_excess_Nm=float(cap_res[1]),
                   heavy_foot_updates=int(cap_res_n[3]), heavy_pad_updates=int(cap_res_n[4]),
                   heavy_pad_frac=float(cap_res_n[4] / max(1, len(tilts))), heavy_pad_max_excess_Nm=float(cap_res[2])),
               stops=[_stop_result(s, tk.rate, rest_speed(tk.mode)) for s in stops] if stops else None, wall_s=wall,
               steps_per_s=steps / wall)
    onset = fall = None
    if fell_at is not None:
        hist = list(hist)
        i_on = len(hist) - 1
        while i_on > 0 and hist[i_on - 1][1] >= FALL_ONSET_RAD:
            i_on -= 1
        onset = hist[i_on][0]
        active = _activity(hist, onset, p.drives)
        candidates = [key for key in HW_LIMITS if active["hardware"][key] > 0.0]
        if confirm:
            fall = (ring, final, onset, fell_at, cause, q_fall, push, candidates)
        out.update(fall_onset_s=onset, fall_candidates=candidates, limits_before_onset=active)
    if est_err:
        E = np.array(est_err)
        out["estimator"] = dict(_estimator_stats(E), noise=tk.noise.describe(), gains=asdict(tk.est.g))
        if onset is not None:
            cut = onset - LIMIT_WINDOW_S
            pre = E[E[:, 0] <= cut]
            out["estimator"]["before_fall"] = dict(
                until_s=cut, rule=f"updates up to {LIMIT_WINDOW_S:g} s before the fall's onset",
                **(_estimator_stats(pre) if len(pre) else dict(updates=0)))
            upto = E[E[:, 0] <= onset]
            out["estimator"]["until_onset"] = dict(until_s=onset, rule="updates up to the fall's onset",
                                                   **(_estimator_stats(upto) if len(upto) else dict(updates=0)))
    if recorder is not None:
        recorder.end(steps, out)
    return out, onset, fall
