"""Emergency stop of the adopted 60 cm Triceratops model (D25, D41): a damped sit, a hard sit and a passive collapse,
from standing and from mid-stride of the frozen model-based trot (``walk/walk60.py``). SIMULATION, plus DESK estimates
of the electrical side. Reported, not judged: no pre-registration covers it. Report: ``sim/REPORT_SIT60.md``.

Why: the leg drives have no brakes, so every stop must first command a damped sit (main's
``docs/hardware/LESSONS_LEARNED.md:99-100``), and the bench plan's B7 needs the regen energy of a hard sit of twelve
drives as its input (``bench/BENCH_PLAN_FORELEG.md:65``, row B7).

The plant is the training env (``rl/trike_env.py``) built as the trot's plant (``walk60.plant_config``) on the adopted
model ``sim/generated/trike60_C_closed_capsule_fore128.xml`` (12.751 kg), domain randomisation off, sensor noise on,
servo delay 10 ms (D9), 1 ms physics. The robot stands (the trot controller at 0 m/s) or trots at 0.25 m/s; at the stop
input (t = 0 below) this script takes over the env's MuJoCo model and data and steps them itself:

- **sit_1.0 / sit_0.3 (sit, then cut).** Each ankle moves on a minimum-jerk path from where it is at t = 0 (chassis
  frame) to the same fore-aft and sideways place at the sit depth, in 1.0 s (the damped sit) or 0.3 s (the hard sit),
  joint targets from the trot's own inverse kinematics on the model, through the same 10 ms delay queue, with the
  drives' modelled gains and torque limits (kp 598.5 N m/rad, kv 11.51-24.48 N m s/rad, 0.70 x maker peak; A2). The
  sit depth is the deepest the forelegs reach with the ankle under the hip and the elbow at most 32 deg, 4 deg short
  of its +36 deg fold stop; the body stays level. Power is cut (every actuator to zero torque) at 1.5 s and 0.5 s:
  the Pilz relay's delayed contacts are set between 0.5 and 1.5 s
  (``hardware/triceratops_kickoff/PARTS_PREMIUM.md:300-301``). B8 wants the sit pose reached at least 0.2 s before
  power drops (``BENCH_PLAN_FORELEG.md:66``, a rule written for one leg on a catch rig). With the 10 ms delay a sit's
  last target reaches the drives 0.01 s after its nominal length: 0.49 s before the 1.5 s cut for sit_1.0, 0.29 s
  before the 0.5 s cut for sit_0.2, and 0.19 s before it for sit_0.3, 0.01 s short of B8's 0.2 s (with a 0.5 s relay
  setting, 0.29 s is the longest sit that meets it). ``t_sit_reached_s`` (the chassis within 5 mm of its height at
  the cut) comes earlier, 0.24 s before the cut for sit_0.3 from standing.
- **collapse.** The 12 leg drives go to zero torque at t = 0 (they lose their commands; what the real drives do then
  is not confirmed: they have a time-out setting, planned as torque-off,
  ``hardware/trex_form_study/stage2/jetson75/LATENCY_AND_BUS.md:365``). The head and tail servos keep their targets.
- Sensitivity cases (headline starts only): ``collapse_friction`` adds each drive's back-drive torque from the
  catalog as joint friction (X4 0.2 N m, X6 1.6 N m; ``hardware/trex_form_study/designlab/data/actuators.json:125``,
  ``:208``); ``collapse_shorted`` adds the damping of a drive whose phases are shorted, Kt^2 / (3 R) at the output
  (X4 5.4, X6 7.2 N m s/rad; inductance neglected, so an upper bound). Whether the drives do either on a lost command
  is not known. DESK constants, simulated.

Variants (``skid_spec``): ``adopted`` is the model file as it is. Its belly cannot reach the floor in a sit (the legs
fold only to +36 deg; the parts list has "the skid and bumpers take the sit", ``PARTS_PREMIUM.md:304``, but the model
has no skid), so the others are TEST COPIES written at run time, each the adopted XML with one added geom, a massless
box under the belly (``belly_skid_TEST``, the chassis's contact class, its top at the belly shell's lowest point). The
adopted model file is never written.

- ``skid``: 0.30 x 0.10 m, centred under the standing centre of mass; its bottom reaches 10 mm deeper below the
  chassis than the soles do at the sit pose (``SKID_EARLY_M``), so the body lands on it 10 mm before the sit ends and
  the folded legs lift their feet. The 10 mm is a choice of this study, not a design figure, and it sets the landing
  speed and force: ``skid_e5mm`` and ``skid_e20mm`` are ``skid`` at 5 and 20 mm.
- ``skid_wide``: 0.18 m wide (nearly the belly shell's +-0.093 m), shortened at the front to end at x = 0.110 m
  (chassis frame) so that it clears the fore elbow drives, whose X6 stators reach back to x = 0.126 m standing and
  0.131 m in the sit pose; its rear end is ``skid``'s. A full-length 0.18 m skid (the first run, 27 Nov 2026) touched
  those stators in every stop from the trot, up to 1.3 kN.
- ``skid_soft``: ``skid`` with ``solref`` 0.03 s and ``priority`` 1, so its floor contact runs at 0.03 s instead of
  the model's 0.008 s, a stand-in for a foam pad. Without the priority MuJoCo averages the two geoms' solref by
  solmix (both 1) to 0.019 s, which is what the first run used. The skid's friction, solimp and condim equal the
  floor's, so only the stiffness changes (checked in the tests).

``impratio10`` repeats the four main stops of ``adopted`` and ``skid`` with the stiffer friction setting the build
checks every result under, at every start the model's contact runs them at: standing and all 14 trot triggers, both
seeds. (The first runs, 27 Nov 2026, repeated only the two headline trot triggers, and their impratio-10 rows were
then compared with rows over all 14 phases.)

Measures (from t = 0; definitions in ``measures``): time to rest, the chassis's lowest and final height, peak drive
torque and speed per drive group against the model's limits (0.70 x maker peak) and the speed allowances the gait
report uses (``rl/analysis/gait_pipe1.py:40``: X4 83 rpm, X6 108 rpm at the joint), peak floor contact force per part
group (soles, belly or skid, leg links, shell) and the body's landing speed, peak self-contact force and the geom
pairs that carry it, peak torque on the joints' hard stops, the mechanical energy the drives can at most take net
while powered (kinetic energy at the stop input + potential energy released before the cut: a ceiling on their net
take, braking less their own positive work, for any sit that reaches the same height before the cut; not on their
gross braking), and the electrical side: per sample, each drive's bus power = mechanical
power at the drive (x 0.9 through the flexion linkages while braking, / 0.9 while motoring) + copper loss
3 R (tau / Kt)^2 at 25 C, with Kt and R from the catalog the thermal model and the demo course read
(``rl/analytics/gait_report.drive_table``; ``rl/analytics/gait_core.electrical_power``; X4 1.8 N m/A rms and 0.2 ohm,
X6 2.1 N m/A rms and 0.205 ohm, ``actuators.json:104-105``, ``:186-187``). Energy pushed back onto the bus: per drive
alone (gross), and on the shared bus (one drive's braking feeds another's motoring; the net surplus is what the clamp
must take), also less a 25-40 W hotel load. Left out, so the regen is an UPPER bound: gearbox and bearing losses in
the drives (the joints' own model damping is not counted as drive power), inverter switching and conduction losses,
iron losses, the drives' own standby draw, wiring and capacitor losses, and the rise of winding resistance with
temperature (copper at 25 C; a warmer winding, as in D48's 35 C room, loses more and returns less).

Settling (``settling``): from 1 s after the reference time (the power cut for a sit, the stop input for a collapse)
to the end of the window, how far the chassis still sinks, how far the most-travelled loaded sole slides along the
floor (contact-point velocity, tangential), and the soles' friction use |Ft| / Fn in the samples where they slide
(against the contact's friction coefficient, 0.85 for the soles). MuJoCo's contact is soft: a sole can creep under a
tangential force below the friction limit, and the stiffer setting (impratio 10) mostly removes that creep, so these
measures tell the creep apart from a real slide. Chassis heights at fixed times after the reference
(``z_after_ref_m``) are kept for every run, in both contact settings, and so are the leg joints resting on their range
limits at the end of the window, with the limit torque (``end_joints_on_limits_nm``): what holds the unpowered legs.

Desk envelope: the B7 pulse in the summary is an ENVELOPE, the largest bus surplus of one run with that run's span,
beside the largest peak powers over all runs (``max_peak_*``, often from other runs); the worst-energy run's own peaks
are given as ``its_peak_*``.

Usage (from anywhere; CPU):
  python hardware/triceratops_build/sim/sit60.py run --out DIR [--workers 2] [--quick]
  python hardware/triceratops_build/sim/sit60.py summarise --raw FILE.json.gz --out RESULTS_DIR
``run`` writes one gzipped JSON of every run's measures (and the headline runs' traces) to DIR (keep it out of the
repository: it is large); ``summarise`` writes ``sit60_summary.json`` and ``sit60_tables.md`` to RESULTS_DIR.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
RL = BUILD / "rl"
REPO = BUILD.parents[1]
for _p in (BUILD / "walk", RL, RL / "analysis", RL / "analytics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MODEL = HERE / "generated" / "trike60_C_closed_capsule_fore128.xml"
SEEDS = (1, 2)
DELAY_STEPS = 10  # D9
TROT_COMMAND = 0.25  # m/s, the 60 cm walking speed (D2); 0.40 m/s is not a walking target
STAND_TRIGGER_S = 2.0  # standing start: the trot controller at 0 m/s for 2 s
TROT_TRIGGERS_S = tuple(round(5.94 + 0.05 * j, 3) for j in range(14))  # one 0.70 s stride, every 0.05 s
HEADLINE_TROT = (5.94, 6.29)  # nearest mid-swing of each pair (phase 0.771: left fore + right hind; 0.271: the other)
SETTLE_AFTER_S = 1.0  # settling measures start this long after the reference time (cut, or the stop input)
SETTLE_LOADED_N = 2.0  # a sole carries load when its floor contacts' normal force sums to at least this
SETTLE_SLIDE_MPS = 0.002  # a loaded sole slides when its contact point moves faster than this along the floor
Z_AFTER_REF_S = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)  # chassis heights kept at these times after the reference
WINDOW_S = 8.0  # simulated after the stop input
SIT_FLEX_MAX_DEG = 32.0  # elbow and knee fold limit for the sit pose: 4 deg short of the +36 deg stop
SIT = {"sit_1.0": (1.0, 1.5), "sit_0.3": (0.3, 0.5), "sit_0.2": (0.2, 0.5)}  # (sit duration, power cut), s after
# the stop input; sit_0.2 is a bounding case for B7 (faster than the hard sit; its peak deceleration, 8.2 m/s^2 over
# the 57 mm drop, is still under g, so the feet stay loaded)
SKID_EARLY_M = 0.010  # the skid reaches this much deeper below the chassis than the soles at the sit pose
SKID_EARLY_SWEEP = {"skid_e5mm": 0.005, "skid_e20mm": 0.020}  # the same skid at other early-contact depths
SKID_HALF_X, SKID_HALF_Y = 0.15, 0.05
SKID_WIDE_HALF_Y = 0.09  # skid_wide: nearly the belly shell's width (torso_contact: +-0.0928 m); the soles' inner
# edges are at about +-0.116 m
SKID_WIDE_FRONT_X = 0.110  # skid_wide's front end (chassis frame): the fore X6 flexion stators' rear faces are at
# x = 0.126 m standing and 0.131 m in the sit pose (the upper link points back from the shoulder, so no pitch angle
# brings them further back); 16 mm clear
SKID_TOP_Z = -0.075  # the belly shell's lowest point (torso_contact ellipsoid: 0.004 - 0.0792 m), chassis frame
SKID_SOFT_SOLREF = (0.03, 1.0)
SKID_SOFT_PRIORITY = 1  # over the floor's 0, so the skid's solref is used as it is (not solmix-averaged to 0.019 s)
HOTEL_W = (25.0, 30.0, 40.0)  # the hotel load range the battery estimates use (rl/analytics/gait_report.py)
REST = {"lin_mps": 0.005, "ang_rps": 0.02, "joint_rps": 0.05, "hold_s": 0.5}
# Catalog figures (hardware/trex_form_study/designlab/data/actuators.json:125 and :208, notes; maker data read 7 Nov
# 2026)
BACKDRIVE_NM = {"X4-36-L": 0.2, "X6-60": 1.6}
BACK_EMF_V_PER_KRPM = {"X4-36-L": 4.3, "X6-60": 16.0}  # motor side, maker figures
GEAR = {"X4-36-L": 36.0, "X6-60": 19.612}
BUS_V = {"pack_full": 49.2, "tether": 50.0, "clamp": 52.5, "drive_max": 55.0, "low": 42.0}
GROUPS = ("roll", "pitch", "front elbow", "hind knee")
PART_GROUPS = ("soles", "belly", "skid", "legs", "shell")


# ---------------------------------------------------------------- models
def skid_depth_m(sit_height_m: float) -> float:
    """Depth of the skid's bottom below the chassis origin: SKID_EARLY_M deeper than the soles at the sit pose."""
    return sit_height_m + SKID_EARLY_M


def skid_geom_xml(depth_m: float, com_x_m: float, solref: tuple[float, float] | None = None,
                  half_y: float = SKID_HALF_Y, front_x: float | None = None, priority: int | None = None) -> str:
    """The skid geom: its bottom ``depth_m`` below the chassis origin, its top at SKID_TOP_Z; in x from
    ``com_x_m - SKID_HALF_X`` to ``com_x_m + SKID_HALF_X``, or to ``front_x`` if that is given (chassis frame)."""
    bot = -float(depth_m)
    hz = 0.5 * (SKID_TOP_Z - bot)
    zc = 0.5 * (SKID_TOP_Z + bot)
    x0, x1 = com_x_m - SKID_HALF_X, (com_x_m + SKID_HALF_X if front_x is None else float(front_x))
    if x1 <= x0:
        raise ValueError("the skid's front end must be ahead of its rear end")
    extra = f' solref="{solref[0]:g} {solref[1]:g}"' if solref else ""
    extra += f' priority="{int(priority)}"' if priority is not None else ""
    return (f'      <geom name="belly_skid_TEST" type="box" size="{0.5 * (x1 - x0):.5f} {half_y:g} {hz:.5f}" '
            f'pos="{0.5 * (x0 + x1):.5f} 0 {zc:.5f}" class="chassis_collision" group="1" mass="0" '
            f'rgba=".85 .35 .2 1"{extra} />\n')


def test_copy_xml(depth_m: float, com_x_m: float, solref=None, half_y: float = SKID_HALF_Y,
                  front_x: float | None = None, priority: int | None = None) -> str:
    """The adopted model's XML with the skid added after ``torso_contact`` (the only change)."""
    text = MODEL.read_text()
    anchor = '<geom name="torso_contact"'
    if text.count(anchor) != 1:
        raise RuntimeError("torso_contact must appear exactly once in the adopted model")
    i = text.index(anchor)
    j = text.index("\n", i) + 1
    return text[:j] + skid_geom_xml(depth_m, com_x_m, solref, half_y, front_x, priority) + text[j:]


def write_test_copy(depth_m: float, com_x_m: float, solref=None, folder: str | None = None,
                    half_y: float = SKID_HALF_Y, front_x: float | None = None, priority: int | None = None,
                    tag: str | None = None) -> Path:
    folder = folder or tempfile.mkdtemp(prefix="sit60_")
    if tag is None:
        tag = (("skid_soft" if solref else "skid") + ("" if half_y == SKID_HALF_Y else f"_halfy{half_y:g}")
               + ("" if front_x is None else f"_front{front_x:g}") + ("" if priority is None else f"_prio{priority}"))
    p = Path(folder) / f"trike60_C_closed_capsule_fore128_{tag}_TESTCOPY.xml"
    p.write_text(test_copy_xml(depth_m, com_x_m, solref, half_y, front_x, priority))
    return p


def skid_spec(variant: str, sit_height_m: float) -> dict:
    """The skid of a test-copy variant: the keyword arguments of ``write_test_copy`` (without folder and tag)."""
    early = SKID_EARLY_SWEEP.get(variant, SKID_EARLY_M)
    spec = {"depth_m": sit_height_m + early, "solref": None, "half_y": SKID_HALF_Y, "front_x": None,
            "priority": None}
    if variant == "skid_wide":
        spec.update(half_y=SKID_WIDE_HALF_Y, front_x=SKID_WIDE_FRONT_X)
    elif variant == "skid_soft":
        spec.update(solref=SKID_SOFT_SOLREF, priority=SKID_SOFT_PRIORITY)
    elif variant not in ("skid", *SKID_EARLY_SWEEP):
        raise ValueError(variant)
    return spec


TEST_VARIANTS = ("skid", "skid_wide", "skid_soft", *SKID_EARLY_SWEEP)


# ---------------------------------------------------------------- the sit pose
def sit_geometry(model_path: Path = MODEL) -> dict:
    """The sit depth (ankle below the chassis origin, level body), the chassis height it gives (soles' lowest point),
    and the standing centre of mass's fore-aft position (chassis frame)."""
    import walk60

    kin = walk60.Kinematics(str(model_path))
    m = kin.m
    lim = math.radians(SIT_FLEX_MAX_DEG)
    for c in kin.hi:
        kin.hi[c] = kin.hi[c].copy()
        kin.hi[c][2] = min(kin.hi[c][2], lim)
    # deepest reachable (least depth) for each leg, ankle at its home fore-aft and sideways place: bisection on z
    reach = {}
    for c in kin.p_home:
        lo, hi = 0.15, 0.30  # depth below the chassis origin, m: lo unreachable, hi reachable
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            tgt = np.array([kin.p_home[c][0], kin.p_home[c][1], -mid])
            q, err = kin.ik(c, tgt, kin.q_home[c], iters=60, tol=1e-7)
            if err < 1e-5:
                hi = mid
            else:
                lo = mid
        reach[c] = hi
    depth = max(reach.values())  # a level body: the leg that folds least deep sets the depth for all four
    q_sit = {}
    for c in kin.p_home:
        tgt = np.array([kin.p_home[c][0], kin.p_home[c][1], -depth])
        q_sit[c], err = kin.ik(c, tgt, kin.q_home[c], iters=60, tol=1e-7)
        assert err < 1e-4, (c, err)
    # chassis height over the floor with each sole flat (the passive ankle turns the foot level: its spring is soft,
    # 2.03 N m/rad, and the tilt at the sit pose is inside its +-22 deg): ankle depth + the sole's offset below the
    # ankle + its radius
    sole = {c: m.geom(f"{c}_sole").id for c in kin.p_home}
    below = {c: float(-m.geom_pos[g][2] + m.geom_size[g][0]) for c, g in sole.items()}
    sit_height = max(depth + below[c] for c in kin.p_home)
    stand_height = max(-kin.p_home[c][2] + below[c] for c in kin.p_home)
    kin.set_q(q_sit)
    d = kin.d
    foot_tilt = {c: float(np.degrees(math.atan2(d.xmat[kin.foot[c]].reshape(3, 3)[2, 0],
                                                d.xmat[kin.foot[c]].reshape(3, 3)[2, 2]))) for c in kin.p_home}
    # standing centre of mass (chassis frame), home pose
    kin.set_q(kin.q_home)
    mujoco.mj_comPos(m, d)
    com_x = float(d.subtree_com[m.body("chassis").id][0])
    return {"sit_depth_m": float(depth), "reach_depth_m": {c: float(v) for c, v in reach.items()},
            "q_sit_deg": {c: np.degrees(v).round(3).tolist() for c, v in q_sit.items()},
            "foot_tilt_at_sit_deg": foot_tilt, "sole_below_ankle_m": below,
            "sit_height_m": float(sit_height), "stand_height_m": float(stand_height), "com_x_m": com_x,
            "flex_max_deg": SIT_FLEX_MAX_DEG}


# ---------------------------------------------------------------- the plant and the starts
def make_env(model_path: Path, impratio: float | None):
    import trike_env as te
    import walk60

    env = te.TrikeEnv(walk60.plant_config(model_path=str(model_path), domain_randomization=False,
                                          observation_noise=True))
    if impratio is not None:
        env.model.opt.impratio = float(impratio)
    return env


def starts(model_path: Path, impratio: float | None, seed: int, kind: str, triggers) -> list[dict]:
    """Run the plant to each trigger time and snapshot it: model and data copies, the delay queue, the phase."""
    import walk60

    env = make_env(model_path, impratio)
    P = walk60.WalkParams()
    ctrl = walk60.TrotController(env, P)
    command = 0.0 if kind == "stand" else TROT_COMMAND
    obs, _ = env.reset(seed=seed, options={"command_speed": command, "delay_steps": DELAY_STEPS})
    out, steps = [], sorted(int(round(t / env.dt)) for t in triggers)
    skid = _geom_id(env.model, "belly_skid_TEST")
    skid_touch = skid_self = False  # sampled once per control step (10 ms), not every physics step
    k = 0
    for target in steps:
        while k < target:
            obs, _, term, trunc, info = env.step(ctrl.act(obs))
            k += 1
            if term or trunc:
                raise RuntimeError(f"start {kind} seed {seed} ended at {k}: {info.get('termination')}")
            if skid >= 0:
                skid_touch |= _touches_floor(env.model, env.data, skid)
                skid_self |= _touches_other_than_floor(env.data, skid)
        t_walk = k * env.dt - P.stand_s
        out.append({"kind": kind, "seed": seed, "t_trigger_s": round(k * env.dt, 3),
                    "phase": None if kind == "stand" else round((t_walk / P.period_s) % 1.0, 4),
                    "model": copy.copy(env.model), "data": copy.copy(env.data), "fifo": deque(env._fifo,
                                                                                               maxlen=env._fifo.maxlen),
                    "leg_ctrl_index": env._leg_ctrl_index, "leg_act": env._leg_act.copy(),
                    "leg_qadr": env._leg_qadr.copy(), "leg_dadr": env._leg_dadr.copy(), "chassis": env._chassis,
                    "floor": env._floor, "skid_touched_before": bool(skid_touch),
                    "skid_self_touched_before": bool(skid_self),
                    "speed_mps": float(info["forward_speed"]) if kind != "stand" else 0.0})
    return out


def _geom_id(m, name: str) -> int:
    return int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name))


def _touches_floor(m, d, g: int) -> bool:
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 == g and c.geom2 == 0) or (c.geom2 == g and c.geom1 == 0):
            return True
    return False


def _touches_other_than_floor(d, g: int) -> bool:
    """Whether geom ``g`` is in contact with any geom other than the floor (geom 0): the skid with a leg."""
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 == g and c.geom2 != 0) or (c.geom2 == g and c.geom1 != 0):
            return True
    return False


# ---------------------------------------------------------------- min-jerk sit trajectory
def min_jerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def sit_targets(model_path: Path, q0: np.ndarray, depth_m: float, duration_s: float, dt: float) -> tuple:
    """Joint targets (samples x 12, the env's leg actuator order) for the sit from leg angles ``q0`` (12): each ankle
    from its place at q0 to the same fore-aft and sideways place at ``depth_m`` below the chassis, minimum jerk,
    one IK per 1 ms sample (warm-started). Also the largest IK residual (m)."""
    import walk60
    from trike_env import CORNERS, LEG_ACTUATORS

    kin = walk60.Kinematics(str(model_path))
    lim = math.radians(SIT_FLEX_MAX_DEG)
    for c in kin.hi:
        kin.hi[c] = kin.hi[c].copy()
        kin.hi[c][2] = min(kin.hi[c][2], lim)
    idx = {c: np.array([LEG_ACTUATORS.index(f"{c}_{a}") for a in ("roll", "pitch", "flexion")]) for c in CORNERS}
    qd = {c: np.asarray(q0)[idx[c]].copy() for c in CORNERS}
    p0, _ = kin.foot_and_jacobian(qd)
    p1 = {c: np.array([p0[c][0], p0[c][1], -depth_m]) for c in CORNERS}
    n = max(1, int(round(duration_s / dt)))
    out = np.zeros((n, 12))
    guess = {c: np.clip(qd[c], kin.lo[c], kin.hi[c]) for c in CORNERS}
    err_max = 0.0
    for k in range(n):
        s = min_jerk((k + 1) / n)
        for c in CORNERS:
            q, err = kin.ik(c, p0[c] + (p1[c] - p0[c]) * s, guess[c], iters=8)
            guess[c] = q
            out[k, idx[c]] = q
            if k == n - 1:
                err_max = max(err_max, err)
    return out, err_max, {c: p1[c].tolist() for c in CORNERS}


# ---------------------------------------------------------------- one stop
def part_group_table(m) -> np.ndarray:
    """Per geom: index into PART_GROUPS (or -1 for the floor and geoms that never touch)."""
    gid = np.full(m.ngeom, -1, dtype=np.int64)
    skid = _geom_id(m, "belly_skid_TEST")
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) or g == 0:
            continue
        name = m.geom(g).name
        ct = int(m.geom_contype[g])
        if g == skid:
            gid[g] = PART_GROUPS.index("skid")
        elif name.endswith("_sole"):
            gid[g] = PART_GROUPS.index("soles")
        elif ct == 2:
            gid[g] = PART_GROUPS.index("belly")
        elif ct in (16, 32):
            gid[g] = PART_GROUPS.index("legs")
        elif ct == 4:
            gid[g] = PART_GROUPS.index("shell")
    return gid


def run_stop(start: dict, behaviour: str, depth_m: float, model_path: Path, window_s: float = WINDOW_S,
             keep_trace: bool = False) -> dict:
    """Simulate one stop from a snapshot. Returns the logged arrays' measures (and the trace if asked)."""
    import gait_report as gr

    m = copy.copy(start["model"])
    d = copy.copy(start["data"])
    fifo = deque(start["fifo"], maxlen=start["fifo"].maxlen)
    idx, leg_act, qadr, dadr = start["leg_ctrl_index"], start["leg_act"], start["leg_qadr"], start["leg_dadr"]
    ch, dt = start["chassis"], float(m.opt.timestep)
    names = [m.actuator(int(i)).name for i in leg_act]
    groups = [gr.gp.group_of(n) for n in names]
    drives = gr.drive_table()
    drive_of = [drives[g]["drive"] for g in groups]
    eta = np.array([drives[g]["transmission_efficiency"] for g in groups])
    kt = np.array([drives[g]["kt_nm_per_a_rms"] for g in groups])
    res = np.array([drives[g]["phase_resistance_ohm"] for g in groups])
    allow = np.array([drives[g]["speed_allowance_rpm"] for g in groups])
    limit = m.actuator_forcerange[leg_act, 1].copy()
    leg_jnt = m.actuator_trnid[leg_act, 0].copy()
    # kinetic energy at the stop input (the whole robot, head and tail included), from a recomputed copy
    d_ke = copy.copy(d)
    mujoco.mj_forward(m, d_ke)
    Mv = np.zeros(m.nv)
    mujoco.mj_mulM(m, d_ke, Mv, d_ke.qvel.copy())
    ke0 = float(0.5 * d_ke.qvel @ Mv)
    ankle_jnt = np.array([m.joint(f"{c}_ankle").id for c in ("left_fore", "right_fore", "left_hind", "right_hind")])
    gid = part_group_table(m)

    def cut(which):
        m.actuator_gainprm[which, 0] = 0.0
        m.actuator_biasprm[which, 1] = 0.0
        m.actuator_biasprm[which, 2] = 0.0

    sit = behaviour in SIT
    targets, ik_err, p_sit = None, 0.0, None
    t_cut = None
    if sit:
        dur, t_cut = SIT[behaviour]
        # from the last target the walking controller sent (not the measured angles: a stance drive holds its load
        # with a tracking error, and starting from the measured angle would drop that load at once)
        q0 = np.asarray(fifo[-1], dtype=np.float64).copy()
        targets, ik_err, p_sit = sit_targets(model_path, q0, depth_m, dur, dt)
    else:  # a collapse: the leg drives lose torque now
        cut(leg_act)
        if behaviour == "collapse_friction":
            m.dof_frictionloss[dadr] += np.array([BACKDRIVE_NM[dv] for dv in drive_of])
        elif behaviour == "collapse_shorted":
            m.dof_damping[dadr] += np.array([drives[g]["kt_nm_per_a_rms"] ** 2 / (3.0 * drives[g]
                                                                                      ["phase_resistance_ohm"])
                                             for g in groups])
        elif behaviour != "collapse":
            raise ValueError(behaviour)

    n = int(round(window_s / dt))
    L = {"t": np.zeros(n), "z": np.zeros(n), "com_z": np.zeros(n), "vlin": np.zeros(n), "vz": np.zeros(n),
         "wang": np.zeros(n), "roll": np.zeros(n), "pitch": np.zeros(n), "q": np.zeros((n, 12)),
         "qd": np.zeros((n, 12)), "tau": np.zeros((n, 12)), "om": np.zeros((n, 12)),
         "floor_f": np.zeros((n, len(PART_GROUPS))), "self_f": np.zeros(n), "stop_tau": np.zeros((n, 12)),
         "ankle_stop_tau": np.zeros(n), "sole_fn": np.zeros((n, 4)), "sole_ft": np.zeros((n, 4)),
         "sole_slip": np.zeros((n, 4))}
    sole_of = {int(m.geom(f"{c}_sole").id): i for i, c in enumerate(("left_fore", "right_fore", "left_hind",
                                                                     "right_hind"))}
    sole_mu = 0.0
    v6 = np.zeros(6)
    GEOM = mujoco.mjtObj.mjOBJ_GEOM
    geom_peak: dict[str, float] = {}
    self_pair_peak: dict[str, float] = {}
    first_body_contact = None
    f6 = np.zeros(6)
    leg_pos = {int(j): k for k, j in enumerate(leg_jnt)}
    ankle_set = set(int(j) for j in ankle_jnt)
    LIMIT_JOINT = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
    hold = None
    cut_done = False
    for k in range(n):
        t = (k + 1) * dt
        if sit:
            if t_cut is not None and not cut_done and t >= t_cut - 1e-9:
                cut(np.arange(m.nu))  # power off: every actuator
                cut_done = True
            tgt = targets[min(k, len(targets) - 1)]
            fifo.append(tgt)
            d.ctrl[idx] = fifo[0]
        mujoco.mj_step(m, d)
        L["t"][k] = t
        L["z"][k] = d.xpos[ch, 2]
        L["com_z"][k] = d.subtree_com[ch, 2]
        L["vlin"][k] = float(np.linalg.norm(d.qvel[0:3]))
        L["vz"][k] = d.qvel[2]
        L["wang"][k] = float(np.linalg.norm(d.qvel[3:6]))
        R = d.xmat[ch].reshape(3, 3)
        L["roll"][k] = math.atan2(R[2, 1], R[2, 2])
        L["pitch"][k] = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        L["q"][k] = d.qpos[qadr]
        L["qd"][k] = d.qvel[dadr]
        L["tau"][k] = d.actuator_force[leg_act]
        L["om"][k] = d.actuator_velocity[leg_act]
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(float(f6[0]))
            if g1 == 0 or g2 == 0:
                g = g2 if g1 == 0 else g1
                pg = gid[g]
                if pg >= 0:
                    L["floor_f"][k, pg] += fn
                    nm = m.geom(g).name
                    if fn > geom_peak.get(nm, 0.0):
                        geom_peak[nm] = fn
                    if first_body_contact is None and PART_GROUPS[pg] != "soles" and fn > 1.0:
                        first_body_contact = {"t_s": t, "geom": nm, "vz_mps": float(d.qvel[2])}
                    s = sole_of.get(g)
                    if s is not None:  # a sole on the floor: its friction use and its contact point's slip
                        L["sole_fn"][k, s] += fn
                        L["sole_ft"][k, s] += math.hypot(float(f6[1]), float(f6[2]))
                        sole_mu = max(sole_mu, float(c.friction[0]))
                        mujoco.mj_objectVelocity(m, d, GEOM, g, v6, 0)  # world frame, about the geom's centre
                        r = c.pos - d.geom_xpos[g]
                        w = v6[:3]
                        vp = v6[3:] + np.array([w[1] * r[2] - w[2] * r[1], w[2] * r[0] - w[0] * r[2],
                                                w[0] * r[1] - w[1] * r[0]])
                        nrm = c.frame[:3]
                        vt = vp - float(vp @ nrm) * nrm  # the floor is still: this is the slip
                        L["sole_slip"][k, s] = max(L["sole_slip"][k, s], float(np.linalg.norm(vt)))
            else:
                L["self_f"][k] += fn
                key = "|".join(sorted((m.geom(g1).name, m.geom(g2).name)))
                if fn > self_pair_peak.get(key, 0.0):
                    self_pair_peak[key] = fn
        if d.nefc:
            et = d.efc_type[: d.nefc]
            rows = np.flatnonzero(et == LIMIT_JOINT)
            for r in rows:
                j = int(d.efc_id[r])
                if j in leg_pos:
                    L["stop_tau"][k, leg_pos[j]] += abs(float(d.efc_force[r]))
                elif j in ankle_set:
                    L["ankle_stop_tau"][k] = max(L["ankle_stop_tau"][k], abs(float(d.efc_force[r])))
    out = measures(L, eta, kt, res, limit, groups, drive_of, dt, float(m.body_mass.sum()), float(-m.opt.gravity[2]),
                   t_cut, allow=allow, ke0=ke0)
    out["settling"] = settling(L, dt, t_cut or 0.0, sole_mu)
    # what holds the legs at the end of the window: the joints resting on their range limits, and the limit torque
    end_lim: dict[str, float] = {}
    for r in range(d.nefc):
        if int(d.efc_type[r]) == LIMIT_JOINT:
            nm = m.joint(int(d.efc_id[r])).name
            end_lim[nm] = end_lim.get(nm, 0.0) + abs(float(d.efc_force[r]))
    out["end_joints_on_limits_nm"] = dict(sorted(end_lim.items()))
    out.update({"behaviour": behaviour, "kind": start["kind"], "seed": start["seed"], "t_trigger_s":
                start["t_trigger_s"], "phase": start["phase"], "speed_at_trigger_mps": start["speed_mps"],
                "skid_touched_before_trigger": start["skid_touched_before"],
                "skid_self_touched_before_trigger": start.get("skid_self_touched_before", False),
                "geom_peak_n": geom_peak,
                "self_pair_peak_n": dict(sorted(self_pair_peak.items(), key=lambda kv: -kv[1])[:6]),
                "first_body_contact": first_body_contact, "ik_residual_m": ik_err, "sit_ankle_targets": p_sit,
                "t_cut_s": t_cut, "z0_m": float(start["data"].xpos[ch, 2])})
    if keep_trace:
        step = 5  # every 5 ms
        el = _electric(L["tau"], L["om"], eta, kt, res)
        out["trace"] = {"t": L["t"][::step].round(4).tolist(), "z": L["z"][::step].round(5).tolist(),
                        "floor_f": L["floor_f"][::step].round(2).tolist(),
                        "bus_w": (-el["net"])[::step].round(3).tolist(),
                        "tau_abs_max_by_group": _by_group(np.abs(L["tau"]), groups)[::step].round(3).tolist(),
                        "rpm_abs_max_by_group": (_by_group(np.abs(L["om"]), groups) * 60 / (2 * math.pi))[
                            ::step].round(2).tolist()}
    return out


def _by_group(x: np.ndarray, groups) -> np.ndarray:
    return np.stack([x[:, [i for i, gg in enumerate(groups) if gg == g]].max(1) for g in GROUPS], axis=1)


def _electric(tau, om, eta, kt, res) -> dict:
    import gait_core as gc

    el = gc.electrical_power(tau, om, eta, kt, res)
    p = el["drive_mech"] + el["copper"]  # samples x drives, W drawn from the bus (+) or pushed back (-)
    return {"per_drive": p, "net": p.sum(1), "copper": el["copper"], "mech": el["drive_mech"]}


def events_j(net: np.ndarray, dt: float) -> float:
    """The largest energy (J) of one contiguous braking event (net draw below zero); as rl/demo_course.py."""
    neg = np.r_[False, net < 0, False]
    edges = np.flatnonzero(np.diff(neg.astype(int)))
    return max((float(-net[a:b].sum() * dt) for a, b in zip(edges[::2], edges[1::2])), default=0.0)


def regen(tau, om, eta, kt, res, dt: float, hotel_w=HOTEL_W) -> dict:
    """The electrical side of a stop (W, J, s): see the module docstring."""
    e = _electric(np.asarray(tau, float), np.asarray(om, float), np.asarray(eta, float), np.asarray(kt, float),
                  np.asarray(res, float))
    p, net = e["per_drive"], e["net"]
    surplus = np.maximum(0.0, -net)
    on = np.flatnonzero(surplus > 0)
    out = {
        "gross_regen_j": float(np.maximum(0.0, -p).sum() * dt),
        "gross_regen_j_per_drive": (np.maximum(0.0, -p).sum(0) * dt).round(4).tolist(),
        "bus_surplus_j": float(surplus.sum() * dt),
        "bus_surplus_peak_w": float(max(0.0, surplus.max())) if len(surplus) else 0.0,
        "bus_surplus_time_s": float(len(on) * dt),
        "bus_surplus_span_s": [float(on[0] * dt), float((on[-1] + 1) * dt)] if len(on) else None,
        "bus_largest_event_j": events_j(net, dt),
        "bus_drawn_j": float(np.maximum(0.0, net).sum() * dt),
        "bus_peak_draw_w": float(max(0.0, net.max())) if len(net) else 0.0,
        "copper_j": float(e["copper"].sum() * dt),
        "mech_braking_j": float(np.maximum(0.0, -(np.asarray(tau) * np.asarray(om))).sum() * dt),
        "beyond_hotel_j": {f"{h:g}": float(np.maximum(0.0, surplus - h).sum() * dt) for h in hotel_w},
        "beyond_hotel_peak_w": {f"{h:g}": float(max(0.0, surplus.max() - h)) if len(surplus) else 0.0
                                for h in hotel_w},
    }
    # the largest surplus over any 10 ms and 100 ms (a pulse's power, smoothed by bus capacitance)
    for w in (10, 100):
        if len(surplus) >= w:
            c = np.convolve(surplus, np.ones(w) / w, "valid")
            out[f"bus_surplus_peak_{w}ms_w"] = float(c.max())
    return out


def rest_time(ok: np.ndarray, t: np.ndarray, dt: float, after_s: float = 0.0) -> float | None:
    """Time to final rest: the start of the last stretch in which ``ok`` (per sample) holds to the end of the window,
    if that stretch lasts at least REST['hold_s'], and no earlier than ``after_s`` (a sit's power cut: a sit is not
    at rest while it is still powered and its legs may still drop); None if the motion has not ended by then."""
    h = int(round(REST["hold_s"] / dt))
    if len(ok) < h or not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    start = 0 if len(bad) == 0 else int(bad[-1]) + 1
    if len(ok) - start < h:
        return None
    return max(float(t[start] - dt), float(after_s))


def settling(L: dict, dt: float, t_ref: float, mu: float, after_s: float = SETTLE_AFTER_S) -> dict:
    """What still happens from ``after_s`` after ``t_ref`` (the cut, or the stop input) to the end of the window: the
    chassis's further sink (m), the largest distance one loaded sole slides along the floor (mm), and the soles'
    friction use |Ft| / Fn in the samples where a loaded sole slides (median, 95th percentile, max; ``mu`` is the
    contact's friction coefficient). Also the chassis height at the times in Z_AFTER_REF_S after ``t_ref``. None
    where the window ends first."""
    t, z = L["t"], L["z"]
    z_at = {}
    for s in Z_AFTER_REF_S:
        k = int(round((t_ref + s) / dt)) - 1
        z_at[f"{s:g}"] = float(z[k]) if 0 <= k < len(z) else None
    out = {"t_ref_s": float(t_ref), "after_s": float(after_s), "mu": float(mu), "z_after_ref_m": z_at,
           "sink_m": None, "sole_slide_mm": None, "sliding_samples": None, "sliding_ratio": None}
    k1 = int(round((t_ref + after_s) / dt)) - 1
    if not 0 <= k1 < len(z) - 1:
        return out
    fn, ft, slip = L["sole_fn"][k1 + 1:], L["sole_ft"][k1 + 1:], L["sole_slip"][k1 + 1:]
    loaded = fn >= SETTLE_LOADED_N
    sliding = loaded & (slip > SETTLE_SLIDE_MPS)
    out["sink_m"] = float(z[k1] - z[-1])
    out["sole_slide_mm"] = float((np.where(loaded, slip, 0.0).sum(0) * dt * 1000.0).max())
    out["sliding_samples"] = int(sliding.sum())
    if sliding.any():
        r = ft[sliding] / fn[sliding]
        out["sliding_ratio"] = {"p50": float(np.median(r)), "p95": float(np.percentile(r, 95)),
                                "max": float(r.max())}
    return out


def measures(L, eta, kt, res, limit, groups, drive_of, dt, mass, g, t_cut: float | None = None,
             allow=None, ke0: float | None = None) -> dict:
    """The measures of one stop from its logged arrays (see the module docstring). ``allow``: each drive's speed
    allowance (rpm at the joint); ``ke0``: the robot's kinetic energy at the stop input (J)."""
    t = L["t"]
    z_end = float(L["z"][-1])
    body_ok = (L["vlin"] < REST["lin_mps"]) & (L["wang"] < REST["ang_rps"])
    legs_ok = np.abs(L["qd"]).max(1) < REST["joint_rps"]
    low = np.flatnonzero(L["z"] <= z_end + 0.005)
    out = {"time_to_rest_s": rest_time(body_ok, t, dt, after_s=t_cut or 0.0),
           "time_to_legs_still_s": rest_time(body_ok & legs_ok, t, dt, after_s=t_cut or 0.0),
           "z_min_m": float(L["z"].min()), "z_end_m": z_end,
           "t_within_5mm_of_end_s": float(t[low[0]]) if len(low) else None,
           "roll_end_deg": float(np.degrees(L["roll"][-1])), "pitch_end_deg": float(np.degrees(L["pitch"][-1])),
           "roll_abs_max_deg": float(np.degrees(np.abs(L["roll"]).max())),
           "tipped_over": bool(max(np.abs(L["roll"]).max(), np.abs(L["pitch"]).max()) > math.radians(45.0)),
           "pitch_abs_max_deg": float(np.degrees(np.abs(L["pitch"]).max())),
           "potential_energy_released_j": float(mass * g * (L["com_z"][0] - L["com_z"][-1])),
           "body_weight_n": mass * g}
    if t_cut is not None:  # a sit: when the chassis first came within 5 mm of its height at the power cut
        kc = min(len(t) - 1, max(0, int(round(t_cut / dt)) - 1))
        z_cut = float(L["z"][kc])
        hit = np.flatnonzero(L["z"][: kc + 1] <= z_cut + 0.005)
        out["z_at_cut_m"] = z_cut
        out["t_sit_reached_s"] = float(t[hit[0]]) if len(hit) else None
        out["sit_margin_before_cut_s"] = (t_cut - out["t_sit_reached_s"]) if len(hit) else None
        out["drop_after_cut_m"] = z_cut - z_end
        # what the drives can at most take net while powered: the kinetic energy at the stop input plus the potential
        # energy released before the cut (energy balance; the joints' damping and contacts only take from it). Net:
        # work a drive puts in and brakes out again is not bounded by it
        out["pe_released_powered_j"] = float(mass * g * (L["com_z"][0] - L["com_z"][kc]))
    else:
        out["pe_released_powered_j"] = None  # a collapse: no powered phase
    out["ke_at_stop_input_j"] = ke0
    out["powered_ceiling_j"] = (None if out["pe_released_powered_j"] is None or ke0 is None
                                else max(0.0, out["pe_released_powered_j"]) + ke0)
    tau, om = np.abs(L["tau"]), np.abs(L["om"])
    rpm = om * 60.0 / (2.0 * math.pi)
    per = {}
    for gname in GROUPS:
        cols = [i for i, gg in enumerate(groups) if gg == gname]
        lim = float(limit[cols[0]])
        per[gname] = {"drive": drive_of[cols[0]], "limit_nm": lim,
                      "tau_peak_nm": float(tau[:, cols].max()), "tau_peak_over_limit": float(tau[:, cols].max() / lim),
                      "tau_peak_t_s": float(t[int(tau[:, cols].max(1).argmax())]),
                      "at_limit_share": float((tau[:, cols] >= 0.999 * lim).mean()),
                      "rpm_peak": float(rpm[:, cols].max()), "rpm_peak_t_s": float(t[int(rpm[:, cols].max(1).argmax())]),
                      "speed_allowance_rpm": None if allow is None else float(allow[cols[0]]),
                      "rpm_peak_over_allow": None if allow is None else float(rpm[:, cols].max() / allow[cols[0]]),
                      "tau_peak_after_20ms_nm": float(tau[t > 0.020][:, cols].max()),
                      "tau_peak_after_20ms_over_limit": float(tau[t > 0.020][:, cols].max() / lim),
                      "stop_tau_peak_nm": float(L["stop_tau"][:, cols].max())}
    out["drives"] = per
    out["ankle_stop_tau_peak_nm"] = float(L["ankle_stop_tau"].max())
    w = mass * g
    out["floor_peak_n"] = {pg: float(L["floor_f"][:, i].max()) for i, pg in enumerate(PART_GROUPS)}
    out["floor_peak_bw"] = {pg: v / w for pg, v in out["floor_peak_n"].items()}
    out["self_contact_peak_n"] = float(L["self_f"].max())
    fin = L["floor_f"][-int(round(0.1 / dt)):].mean(0)  # last 0.1 s: what carries the robot at the end
    out["end_support_share"] = {pg: float(fin[i] / w) for i, pg in enumerate(PART_GROUPS)}
    out["electric"] = regen(L["tau"], L["om"], eta, kt, res, dt)
    return out


# ---------------------------------------------------------------- the case list
def case_list(quick: bool = False) -> list[tuple]:
    """Six-field tuples (variant, contact, kind, seed, triggers, behaviours): ``contact`` is "model" or "impratio10";
    ``kind`` is "stand", "trot" or "trot_headline" (trot starts at the headline triggers only, for the sensitivity
    behaviours). Under impratio 10 the main stops of ``adopted`` and ``skid`` run at the same starts as under the
    model's contact (standing and every trot trigger, both seeds), so the two contact settings are compared over the
    same phases. Whether a run is a headline run (its trace kept, in both contact settings) is decided per start in
    ``_task``: every standing start and the trot starts at HEADLINE_TROT."""
    main_b = ("sit_1.0", "sit_0.3", "sit_0.2", "collapse")
    sens_b = ("collapse_friction", "collapse_shorted")
    seeds = SEEDS[:1] if quick else SEEDS
    trot = HEADLINE_TROT if quick else TROT_TRIGGERS_S
    cases = []
    for variant in ("adopted", "skid"):
        for seed in seeds:
            cases.append((variant, "model", "stand", seed, (STAND_TRIGGER_S,), main_b + sens_b))
            cases.append((variant, "model", "trot", seed, trot, main_b))
            if not quick:
                cases.append((variant, "model", "trot_headline", seed, HEADLINE_TROT, sens_b))
                cases.append((variant, "impratio10", "stand", seed, (STAND_TRIGGER_S,), main_b))
                cases.append((variant, "impratio10", "trot", seed, trot, main_b))
    if not quick:
        for seed in seeds:
            for variant in ("skid_wide", *SKID_EARLY_SWEEP):
                cases.append((variant, "model", "stand", seed, (STAND_TRIGGER_S,), main_b))
                cases.append((variant, "model", "trot", seed, trot, main_b))
            cases.append(("skid_soft", "model", "stand", seed, (STAND_TRIGGER_S,), main_b))
            cases.append(("skid_soft", "model", "trot", seed, HEADLINE_TROT, main_b))
    return cases


_GEO: dict = {}
_PATHS: dict = {}


def _init(geo: dict, paths: dict) -> None:
    _GEO.update(geo)
    _PATHS.update({k: Path(v) for k, v in paths.items()})


def _task(case) -> list[dict]:
    variant, contact, kind, seed, triggers, behaviours = case
    path = _PATHS[variant]
    imp = 10.0 if contact == "impratio10" else None
    snaps = starts(path, imp, seed, "stand" if kind == "stand" else "trot", triggers)
    label = "stand" if kind == "stand" else "trot"
    out = []
    for s in snaps:
        headline = kind == "stand" or s["t_trigger_s"] in HEADLINE_TROT
        for b in behaviours:
            r = run_stop(s, b, _GEO["sit_depth_m"], path, keep_trace=headline)
            r.update({"variant": variant, "contact": contact, "start": label, "headline": headline})
            out.append(r)
    return out


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _script_state() -> dict:
    """This script's sha256 and whether git tracks it unchanged at the recorded commit (the commit alone does not
    identify an untracked or edited script)."""
    me = Path(__file__).resolve()
    out = {"script": str(me.relative_to(REPO)), "script_sha256": hashlib.sha256(me.read_bytes()).hexdigest()}
    try:
        st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", str(me)], capture_output=True,
                            text=True, check=True).stdout.strip()
        out["script_git_status"] = ("untracked" if st.startswith("??") else "modified" if st else
                                    "tracked, unchanged")
    except Exception:  # noqa: BLE001
        out["script_git_status"] = "unknown"
    return out


def write_variant_copies(geo: dict, folder: str) -> dict:
    """Every test-copy variant's model file in ``folder``, plus the adopted model: {variant: path}."""
    paths = {"adopted": str(MODEL)}
    for v in TEST_VARIANTS:
        paths[v] = str(write_test_copy(com_x_m=geo["com_x_m"], folder=folder, tag=v,
                                       **skid_spec(v, geo["sit_height_m"])))
    return paths


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geo = sit_geometry()
    depth = skid_depth_m(geo["sit_height_m"])
    tmp = tempfile.mkdtemp(prefix="sit60_models_")
    paths = write_variant_copies(geo, tmp)
    cases = case_list(args.quick)
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(geo, paths)) as pool:
            parts = pool.map(_task, cases, chunksize=1)
    else:
        _init(geo, paths)
        parts = [_task(c) for c in cases]
    runs = [r for p in parts for r in p]
    meta = {"simulation_only": True, "model": str(MODEL.relative_to(REPO)),
            "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(), "git_commit": _git_commit(),
            **_script_state(), "workers": int(args.workers),
            "mujoco": mujoco.__version__, "geometry": geo, "skid_depth_below_chassis_m": depth,
            "skid_geom_xml": {v: skid_geom_xml(com_x_m=geo["com_x_m"], **skid_spec(v, geo["sit_height_m"])).strip()
                              for v in TEST_VARIANTS},
            "test_copy_sha256": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in paths.items()},
            "settings": {"seeds": SEEDS, "delay_steps": DELAY_STEPS, "trot_command_mps": TROT_COMMAND,
                         "stand_trigger_s": STAND_TRIGGER_S, "trot_triggers_s": TROT_TRIGGERS_S,
                         "headline_trot_s": HEADLINE_TROT, "window_s": WINDOW_S, "sit": SIT, "rest": REST,
                         "hotel_w": HOTEL_W, "skid_early_m": SKID_EARLY_M, "skid_early_sweep_m": SKID_EARLY_SWEEP,
                         "skid_soft_solref": SKID_SOFT_SOLREF, "skid_soft_priority": SKID_SOFT_PRIORITY,
                         "skid_wide_half_y_m": SKID_WIDE_HALF_Y, "skid_wide_front_x_m": SKID_WIDE_FRONT_X,
                         "backdrive_nm": BACKDRIVE_NM,
                         "settle": {"after_s": SETTLE_AFTER_S, "loaded_n": SETTLE_LOADED_N,
                                    "slide_mps": SETTLE_SLIDE_MPS, "z_after_ref_s": Z_AFTER_REF_S},
                         "quick": bool(args.quick)},
            "wall_s": round(time.time() - t0, 1), "n_runs": len(runs)}
    (out / "sit60_raw.json.gz").write_bytes(gzip.compress(json.dumps({"meta": meta, "runs": runs}).encode()))
    print(f"{len(runs)} stop runs in {meta['wall_s']} s -> {out / 'sit60_raw.json.gz'}")
    return 0


# ---------------------------------------------------------------- summary (desk comparisons included)
def back_emf_threshold_rpm(drive: str, bus_v: float) -> float:
    """Output speed (rpm) at which a drive's line back-EMF peak reaches ``bus_v``: above it a drive whose bridge is
    off pushes current into a bus held at ``bus_v`` through its diodes; below it, it pushes none into a bus held above
    its back-EMF (by the pack or the tether, 42-50 V). Into a DC link that has decayed (the contactor open and the
    capacitors drained) a drive with its bridge off rectifies at any speed, up to its back-EMF: at the simulated
    collapse speeds (X4 77 rpm, X6 74 rpm at the output) about 12 V for the X4 and 13-23 V for the X6 (from Kt and
    from the maker's figure). DESK, from the maker's back-EMF figure (motor side) and the gear ratio."""
    return bus_v / BACK_EMF_V_PER_KRPM[drive] * 1000.0 / GEAR[drive]


def bus_energy_j_per_mf(v0: float, v1: float) -> float:
    """Energy (J) one millifarad of bus capacitance takes rising from v0 to v1 (V)."""
    return 0.5 * 1e-3 * (v1 * v1 - v0 * v0)


def _stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"n": 0, "n_none": len(vals)}
    a = np.asarray(v, dtype=float)
    return {"n": len(v), "n_none": len(vals) - len(v), "min": float(a.min()), "median": float(np.median(a)),
            "max": float(a.max())}


def _get(r, path):
    x = r
    for k in path:
        if x is None:
            return None
        x = x.get(k) if isinstance(x, dict) else None
    return x


SUMMARY_FIELDS = {
    "time_to_rest_s": ("time_to_rest_s",), "time_to_legs_still_s": ("time_to_legs_still_s",),
    "z_end_m": ("z_end_m",), "z_min_m": ("z_min_m",), "drop_after_cut_m": ("drop_after_cut_m",),
    "sit_margin_before_cut_s": ("sit_margin_before_cut_s",), "roll_abs_max_deg": ("roll_abs_max_deg",),
    "pitch_abs_max_deg": ("pitch_abs_max_deg",), "potential_energy_released_j": ("potential_energy_released_j",),
    "self_contact_peak_n": ("self_contact_peak_n",), "self_contact_peak_bw": ("_self_bw",),
    "ankle_stop_tau_peak_nm": ("ankle_stop_tau_peak_nm",),
    "landing_speed_mps": ("_landing",),
    "pe_released_powered_j": ("pe_released_powered_j",), "ke_at_stop_input_j": ("ke_at_stop_input_j",),
    "powered_ceiling_j": ("powered_ceiling_j",),
    **{f"end_support_{pg}": ("end_support_share", pg) for pg in PART_GROUPS},
    **{f"floor_peak_bw_{pg}": ("floor_peak_bw", pg) for pg in PART_GROUPS},
    **{f"{g}_{k}": ("drives", g, k) for g in GROUPS for k in ("tau_peak_nm", "tau_peak_over_limit", "at_limit_share",
                                                             "rpm_peak", "rpm_peak_over_allow", "stop_tau_peak_nm",
                                                             "tau_peak_after_20ms_nm",
                                                             "tau_peak_after_20ms_over_limit")},
    **{f"e_{k}": ("electric", k) for k in ("gross_regen_j", "bus_surplus_j", "bus_surplus_peak_w",
                                            "bus_surplus_peak_10ms_w", "bus_surplus_peak_100ms_w",
                                            "bus_largest_event_j", "bus_drawn_j", "bus_peak_draw_w", "copper_j",
                                            "mech_braking_j", "bus_surplus_time_s")},
    "e_span_s": ("electric", "_span_s"), "e_leg_max_j": ("electric", "_leg_max_j"),
    "e_drive_net_take_j": ("electric", "_net_take_j"),
    "settle_sink_m": ("settling", "sink_m"), "settle_sole_slide_mm": ("settling", "sole_slide_mm"),
    "settle_sliding_samples": ("settling", "sliding_samples"),
    **{f"settle_sliding_ratio_{q}": ("settling", "sliding_ratio", q) for q in ("p50", "p95", "max")},
    **{f"settle_z_after_{x:g}s_m": ("settling", "z_after_ref_m", f"{x:g}") for x in Z_AFTER_REF_S},
    **{f"e_beyond_hotel_{h:g}w_j": ("electric", "beyond_hotel_j", f"{h:g}") for h in HOTEL_W},
}


def _derive(r: dict) -> dict:
    r = dict(r)
    fb = r.get("first_body_contact")
    r["_landing"] = abs(fb["vz_mps"]) if fb else None
    r["_self_bw"] = r["self_contact_peak_n"] / r["body_weight_n"]
    e = dict(r["electric"])
    sp = e.get("bus_surplus_span_s")
    e["_span_s"] = (sp[1] - sp[0]) if sp else 0.0
    # the drives' net take of mechanical energy over the stop (at the drive, after the linkage): what they braked
    # less what they put in themselves; exactly bus surplus + copper - drawn, since the bus power is mech + copper
    e["_net_take_j"] = e["bus_surplus_j"] + e["copper_j"] - e["bus_drawn_j"]
    per = e["gross_regen_j_per_drive"]  # the env's leg order: 4 corners x (roll, pitch, flexion)
    e["_leg_max_j"] = max(sum(per[3 * i: 3 * i + 3]) for i in range(4))
    r["electric"] = e
    return r


END_LIMIT_KINDS = ("fore flexion", "hind flexion", "fore pitch", "hind pitch", "roll", "ankle")


def _limit_kind(joint: str) -> str | None:
    """A leg joint's kind for the end-of-window limit count (None for a joint that is not a leg joint)."""
    for c in ("left_fore", "right_fore", "left_hind", "right_hind"):
        if joint.startswith(c + "_"):
            a = joint[len(c) + 1:]
            end = "fore" if "fore" in c else "hind"
            if a in ("flexion", "pitch"):
                return f"{end} {a}"
            return a if a in ("roll", "ankle") else None
    return None


def summarise(raw: dict) -> dict:
    runs = [_derive(r) for r in raw["runs"]]
    keys = sorted({(r["variant"], r["contact"], r["start"], r["behaviour"]) for r in runs})
    groups = {}
    for key in keys:
        rs = [r for r in runs if (r["variant"], r["contact"], r["start"], r["behaviour"]) == key]
        g = {"n_runs": len(rs), "seeds": sorted({r["seed"] for r in rs}),
             "triggers_s": sorted({r["t_trigger_s"] for r in rs}),
             "stats": {f: _stats([_get(r, p) for r in rs]) for f, p in SUMMARY_FIELDS.items()}}
        # the run with the largest bus surplus (the clamp's case) and with the largest landing force
        worst = max(rs, key=lambda r: r["electric"]["bus_surplus_j"])
        g["worst_bus_surplus_run"] = {"seed": worst["seed"], "t_trigger_s": worst["t_trigger_s"],
                                      "phase": worst["phase"], "bus_surplus_j": worst["electric"]["bus_surplus_j"],
                                      "span_s": worst["electric"]["bus_surplus_span_s"]}
        geoms: dict[str, float] = {}
        for r in rs:
            for nm, v in r["geom_peak_n"].items():
                geoms[nm] = max(geoms.get(nm, 0.0), v)
        g["geom_peak_n_top"] = dict(sorted(geoms.items(), key=lambda kv: -kv[1])[:6])
        pairs: dict[str, float] = {}
        for r in rs:
            for nm, v in r.get("self_pair_peak_n", {}).items():
                pairs[nm] = max(pairs.get(nm, 0.0), v)
        g["self_pair_peak_n_top"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:4])
        g["n_runs_with_self_contact"] = sum(1 for r in rs if r["self_contact_peak_n"] > 0.0)
        g["first_body_contact_geoms"] = sorted({r["first_body_contact"]["geom"] for r in rs
                                                if r["first_body_contact"]})
        g["skid_touched_before_trigger"] = any(r["skid_touched_before_trigger"] for r in rs)
        g["skid_self_touched_before_trigger"] = any(r.get("skid_self_touched_before_trigger", False) for r in rs)
        g["settle_mu"] = max((r["settling"]["mu"] for r in rs if r.get("settling")), default=None)
        # the joints resting on their range limits at the end of the window: runs with any joint of a kind there
        kinds = {kd: 0 for kd in END_LIMIT_KINDS}
        top = 0.0
        for r in rs:
            el = r.get("end_joints_on_limits_nm") or {}
            for kd in END_LIMIT_KINDS:
                if any(_limit_kind(j) == kd for j in el):
                    kinds[kd] += 1
            top = max([top] + [v for j, v in el.items() if _limit_kind(j) is not None])
        g["end_limit_runs"] = kinds
        g["end_limit_max_nm"] = top
        g["tipped_over_runs"] = [[r["seed"], r["t_trigger_s"], r["phase"]] for r in rs if r["tipped_over"]]
        g["tipped_over_detail"] = [{"seed": r["seed"], "t_trigger_s": r["t_trigger_s"], "phase": r["phase"],
                                    "roll_abs_max_deg": r["roll_abs_max_deg"], "z_end_m": r["z_end_m"],
                                    "first_body_contact": r["first_body_contact"],
                                    "top_floor_geoms_n": dict(sorted(r["geom_peak_n"].items(),
                                                                     key=lambda kv: -kv[1])[:3])}
                                   for r in rs if r["tipped_over"]]
        groups["|".join(key)] = g
    # ---- desk comparisons
    main = [r for r in runs if r["behaviour"] in SIT]
    by_b = {}
    for b in SIT:
        rb = [r for r in main if r["behaviour"] == b]
        if not rb:
            continue
        w = max(rb, key=lambda r: r["electric"]["bus_surplus_j"])
        p = max(rb, key=lambda r: r["electric"]["bus_surplus_peak_w"])
        by_b[b] = {"n_runs": len(rb), "max_bus_surplus_j": w["electric"]["bus_surplus_j"],
                   "max_bus_surplus_case": [w["variant"], w["contact"], w["start"], w["seed"], w["t_trigger_s"]],
                   "its_span_s": w["electric"]["bus_surplus_span_s"],
                   "its_largest_event_j": w["electric"]["bus_largest_event_j"],
                   "max_gross_regen_j": max(r["electric"]["gross_regen_j"] for r in rb),
                   "max_peak_w": p["electric"]["bus_surplus_peak_w"],
                   "max_peak_10ms_w": max(r["electric"].get("bus_surplus_peak_10ms_w", 0.0) for r in rb),
                   "max_peak_100ms_w": max(r["electric"].get("bus_surplus_peak_100ms_w", 0.0) for r in rb),
                   "max_peak_case": [p["variant"], p["contact"], p["start"], p["seed"], p["t_trigger_s"]],
                   # the worst-energy run's own peaks: the max_peak_* above are over all runs (an envelope)
                   "its_peak_w": w["electric"]["bus_surplus_peak_w"],
                   "its_peak_10ms_w": w["electric"].get("bus_surplus_peak_10ms_w", 0.0),
                   "its_peak_100ms_w": w["electric"].get("bus_surplus_peak_100ms_w", 0.0),
                   "max_beyond_hotel_j": {f"{h:g}": max(r["electric"]["beyond_hotel_j"][f"{h:g}"] for r in rb)
                                          for h in HOTEL_W},
                   "max_leg_gross_j": max(r["electric"]["_leg_max_j"] for r in rb),
                   "max_pe_released_j": max(r["potential_energy_released_j"] for r in rb),
                   "max_pe_released_powered_j": max(r["pe_released_powered_j"] for r in rb),
                   "max_ke_at_stop_input_j": max(r["ke_at_stop_input_j"] for r in rb),
                   "max_powered_ceiling_j": max(r["powered_ceiling_j"] for r in rb),
                   "max_braking_at_joints_j": max(r["electric"]["mech_braking_j"] for r in rb),
                   "max_drive_net_take_j": max(r["electric"]["_net_take_j"] for r in rb),
                   "n_runs_net_take_over_ceiling": sum(1 for r in rb if r["electric"]["_net_take_j"]
                                                       > r["powered_ceiling_j"] + 1e-9),
                   "n_runs_bus_surplus_over_ceiling": sum(1 for r in rb if r["electric"]["bus_surplus_j"]
                                                          > r["powered_ceiling_j"] + 1e-9),
                   "n_runs_braking_over_ceiling": sum(1 for r in rb if r["electric"]["mech_braking_j"]
                                                      > r["powered_ceiling_j"] + 1e-9),
                   # a sit routine that lowered the body all the way to where the cut leaves it while still
                   # powered: the whole stop's potential energy plus the kinetic energy at the stop input
                   "max_full_drop_ceiling_j": max(r["potential_energy_released_j"] + r["ke_at_stop_input_j"]
                                                  for r in rb)}
    collapses = [r for r in runs if r["behaviour"].startswith("collapse")]
    peak_rpm_collapse = {g: max(r["drives"][g]["rpm_peak"] for r in collapses) for g in GROUPS} if collapses else {}
    resistors_w = 2 * 50.0  # PARTS_PREMIUM.md:393, continuous ratings
    desk = {
        "by_sit": by_b,
        "power_board": {"clamp_v": BUS_V["clamp"], "resistors_continuous_w": resistors_w,
                        "resistance_for_peak_ohm": {b: (BUS_V["clamp"] ** 2 / v["max_peak_w"]) if v["max_peak_w"] > 0
                                                    else None for b, v in by_b.items()},
                        "clamp_current_at_peak_a": {b: v["max_peak_w"] / BUS_V["clamp"] for b, v in by_b.items()},
                        "ten_pulses_j": {b: 10.0 * v["max_bus_surplus_j"] for b, v in by_b.items()},
                        "resistor_seconds_at_rating_for_one_pulse_s": {b: v["max_bus_surplus_j"] / resistors_w
                                                                       for b, v in by_b.items()},
                        "resistor_seconds_at_rating_for_the_powered_ceiling_s": {
                            b: v["max_powered_ceiling_j"] / resistors_w for b, v in by_b.items()},
                        "resistor_seconds_at_rating_for_the_full_drop_ceiling_s": {
                            b: v["max_full_drop_ceiling_j"] / resistors_w for b, v in by_b.items()}},
        "bus_capacitance": {"j_per_mf_50_to_52.5": bus_energy_j_per_mf(BUS_V["tether"], BUS_V["clamp"]),
                            "j_per_mf_49.2_to_52.5": bus_energy_j_per_mf(BUS_V["pack_full"], BUS_V["clamp"]),
                            "j_per_mf_50_to_55": bus_energy_j_per_mf(BUS_V["tether"], BUS_V["drive_max"]),
                            "mf_to_hold_worst_hard_sit_under_52.5_from_50": (
                                by_b["sit_0.3"]["max_bus_surplus_j"] / bus_energy_j_per_mf(BUS_V["tether"],
                                                                                         BUS_V["clamp"])
                                if "sit_0.3" in by_b else None),
                            "mf_to_hold_worst_hard_sit_under_55_from_50": (
                                by_b["sit_0.3"]["max_bus_surplus_j"] / bus_energy_j_per_mf(BUS_V["tether"],
                                                                                         BUS_V["drive_max"])
                                if "sit_0.3" in by_b else None)},
        "back_emf_threshold_rpm": {dv: {f"{v:g}V": back_emf_threshold_rpm(dv, v) for v in (BUS_V["low"],
                                                                                         BUS_V["pack_full"],
                                                                                         BUS_V["tether"])}
                                   for dv in BACK_EMF_V_PER_KRPM},
        "back_emf_threshold_rpm_from_kt": {dv: BUS_V["low"] / (math.sqrt(6.0) / 3.0 * (kt / GEAR[dv]) *
                                                                  1000.0 * 2.0 * math.pi / 60.0) * 1000.0 / GEAR[dv]
                                           for dv, kt in (("X4-36-L", 1.8), ("X6-60", 2.1))},
        "collapse_peak_rpm_by_group": peak_rpm_collapse,
        "pack_charge_acceptance": "not given in the records; the ideal diodes block regen from the pack "
                                  "(hardware/triceratops_kickoff/PARTS_PREMIUM.md:311)",
    }
    return {"meta": raw["meta"], "groups": groups, "desk": desk}


def _f(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float) and not math.isfinite(x):
        return "-"
    return f"{x:.{nd}f}"


def _cell(st: dict, key: str, nd=2, how=("median", "max")) -> str:
    s = st.get(key, {})
    if not s or s.get("n", 0) == 0:
        return "not at rest" if key.startswith("time_to") and s.get("n_none") else "-"
    txt = " / ".join(_f(s[h], nd) for h in how) if s["n"] > 1 else _f(s["max"], nd)
    if s.get("n_none"):
        if key.startswith("time_to"):
            txt += f" ({s['n_none']} of {s['n'] + s['n_none']} not at rest)"
        else:  # e.g. the landing speed: the runs in which no body part touched the floor have none
            txt += f" (in {s['n']} of {s['n'] + s['n_none']} runs)"
    return txt


def tables(summary: dict) -> str:
    G = summary["groups"]
    lines = ["# Sit and collapse tables (generated by sim/sit60.py summarise; SIMULATION, desk lines marked)", "",
             "Cells: median / max over the runs of a row (seeds, and trot trigger phases), min / median / max for the "
             "chassis's end height and the drop after the cut; one value when one run. BW: body weights "
             "(12.751 kg x 9.81 m/s^2 = 125.1 N). Report: sim/REPORT_SIT60.md.", ""]
    order_b = ("sit_1.0", "sit_0.3", "sit_0.2", "collapse", "collapse_friction", "collapse_shorted")
    order_v = ("adopted", "skid", "skid_wide", "skid_soft", "skid_e5mm", "skid_e20mm")
    notes = {"adopted": "the model file as it is (no skid)",
             "skid": "TEST COPY: 0.30 x 0.10 m skid, 10 mm early contact",
             "skid_wide": "TEST COPY: 0.18 m wide skid, front end at x = 0.110 m (clears the fore elbow drives)",
             "skid_soft": "TEST COPY: skid with solref 0.03 s and priority 1 (contact at 0.03 s)",
             "skid_e5mm": "TEST COPY: skid with 5 mm early contact",
             "skid_e20mm": "TEST COPY: skid with 20 mm early contact"}

    def rows(variant, contact):
        ks = [k for k in G if k.split("|")[0] == variant and k.split("|")[1] == contact]
        return sorted(ks, key=lambda k: (k.split("|")[2] != "stand", order_b.index(k.split("|")[3])))

    for variant in order_v:
        for contact in ("model", "impratio10"):
            ks = rows(variant, contact)
            if not ks:
                continue
            lines += [f"## {variant}, contact {contact}", "", f"{notes[variant]}.", "",
                      "| start | stop | runs | tipped over (roll or pitch > 45 deg) | time to rest s (body) | legs still s |"
                      " chassis end m | drop after cut m |"
                      " sit reached before cut s | end support: soles / skid / legs | first body-part touch, vertical speed m/s |"
                      " peak floor force, BW: soles / belly / skid / legs / shell | peak self-contact BW (runs with any) |"
                      " PE released J (whole stop) | PE released while powered J | KE at stop input J |",
                      "|" + "---|" * 16]
            for k in ks:
                v, c, s, b = k.split("|")
                st = G[k]["stats"]
                sup = " / ".join(_f(st[f"end_support_{pg}"].get("median"), 2) for pg in ("soles", "skid", "legs"))
                fl = " / ".join(_f(st[f"floor_peak_bw_{pg}"].get("max"), 2) for pg in PART_GROUPS)
                sc = f"{_f(st['self_contact_peak_bw'].get('max'), 2)} ({G[k]['n_runs_with_self_contact']})"
                lines.append(f"| {s} | {b} | {G[k]['n_runs']} | {len(G[k]['tipped_over_runs'])} | "
                             f"{_cell(st, 'time_to_rest_s')} | "
                             f"{_cell(st, 'time_to_legs_still_s')} | {_cell(st, 'z_end_m', 3, ('min', 'median', 'max'))} | "
                             f"{_cell(st, 'drop_after_cut_m', 3, ('min', 'median', 'max'))} | {_cell(st, 'sit_margin_before_cut_s')} | {sup} | "
                             f"{_cell(st, 'landing_speed_mps')} | {fl} | {sc} | "
                             f"{_cell(st, 'potential_energy_released_j', 1)} | {_cell(st, 'pe_released_powered_j', 1)} | "
                             f"{_cell(st, 'ke_at_stop_input_j', 2)} |")
            tipped = [(k, d) for k in ks for d in G[k]["tipped_over_detail"]]
            if tipped:
                lines += ["", "Tipped over (roll or pitch over 45 deg): " + "; ".join(
                    f"{k.split('|')[2]} {k.split('|')[3]} seed {d['seed']} at {d['t_trigger_s']:g} s (phase "
                    f"{_f(d['phase'], 3)}): roll up to {_f(d['roll_abs_max_deg'], 1)} deg, chassis end "
                    f"{_f(d['z_end_m'], 3)} m, first body touch {d['first_body_contact']['geom'] if d['first_body_contact'] else '-'}"
                    f", largest floor forces " + ", ".join(f"{n} {_f(x, 0)} N" for n, x in d["top_floor_geoms_n"].items())
                    for k, d in tipped) + "."]
            selfc = [(k, G[k]["self_pair_peak_n_top"]) for k in ks if G[k]["self_pair_peak_n_top"]]
            if selfc:
                lines += ["", "Self-contact (peak N, the geom pairs, max over the row's runs): " + "; ".join(
                    f"{k.split('|')[2]} {k.split('|')[3]}: " + ", ".join(f"{n} {_f(x, 0)}" for n, x in p.items())
                    for k, p in selfc) + "."]
            lines += ["", "Drives (joint side): peak torque N m (share of the model's limit) over the stop, then the same "
                      "from 20 ms on (without the walking controller's own torque at the stop input and the 10 ms delay), "
                      "peak speed rpm (share of the speed allowance: X4 83 rpm, X6 108 rpm), peak torque on the joint's "
                      "hard stops N m; max over the row's runs.", "",
                      "| start | stop | roll | pitch | front elbow | hind knee | share of samples at a limit (max group) |",
                      "|---|---|---|---|---|---|---|"]
            for k in ks:
                v, c, s, b = k.split("|")
                st = G[k]["stats"]
                cells = []
                for g in GROUPS:
                    cells.append(f"{_f(st[f'{g}_tau_peak_nm'].get('max'), 1)} ({_f(st[f'{g}_tau_peak_over_limit'].get('max'), 2)}); "
                                 f"{_f(st[f'{g}_tau_peak_after_20ms_nm'].get('max'), 1)} ({_f(st[f'{g}_tau_peak_after_20ms_over_limit'].get('max'), 2)}),"
                                 f" {_f(st[f'{g}_rpm_peak'].get('max'), 0)} rpm ({_f(st[f'{g}_rpm_peak_over_allow'].get('max'), 2)}),"
                                 f" stop {_f(st[f'{g}_stop_tau_peak_nm'].get('max'), 1)}")
                lim = max(st[f"{g}_at_limit_share"].get("max", 0.0) for g in GROUPS)
                lines.append(f"| {s} | {b} | " + " | ".join(cells) + f" | {_f(100 * lim, 2)} % |")
            lines += ["", "Electrical (DESK model on the simulated torques and speeds; regen an upper bound): J and W.", "",
                      "| start | stop | gross regen J | bus surplus J | peak W (1 ms / 10 ms / 100 ms) | surplus time s |"
                      " largest event J | beyond 30 W hotel J | drawn from bus J | copper J | braking at joints J |"
                      " one leg's gross regen J |", "|" + "---|" * 12]
            for k in ks:
                v, c, s, b = k.split("|")
                st = G[k]["stats"]
                pk = " / ".join(_f(st[f"e_{x}"].get("max"), 1) for x in ("bus_surplus_peak_w", "bus_surplus_peak_10ms_w",
                                                                           "bus_surplus_peak_100ms_w"))
                lines.append(f"| {s} | {b} | {_cell(st, 'e_gross_regen_j')} | {_cell(st, 'e_bus_surplus_j')} | {pk} | "
                             f"{_cell(st, 'e_bus_surplus_time_s', 3)} | {_cell(st, 'e_bus_largest_event_j')} | "
                             f"{_cell(st, 'e_beyond_hotel_30w_j')} | {_cell(st, 'e_bus_drawn_j', 1)} | "
                             f"{_cell(st, 'e_copper_j', 1)} | {_cell(st, 'e_mech_braking_j', 1)} | "
                             f"{_cell(st, 'e_leg_max_j')} |")
            lines.append("")
    # ---- the skid's early-contact depth (a choice of this study)
    sweep = [("skid_e5mm", 5), ("skid", 10), ("skid_e20mm", 20)]
    if any(f"{v}|model|stand|sit_0.3" in G for v, _ in sweep):
        lines += ["## The skid's early-contact depth (TEST COPIES; contact model)", "",
                  "How far below the soles' sit-pose depth the skid reaches is this study's choice; the landing speed and "
                  "force follow from it. Same skid otherwise (0.30 x 0.10 m).", "",
                  "| early contact mm | start | stop | runs | first touch, vertical speed m/s (median / max) |"
                  " skid peak force BW (median / max) | legs peak force BW (max) | tipped over | bus surplus J (max) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s in ("stand", "trot"):
            for b in ("sit_1.0", "sit_0.3", "sit_0.2", "collapse"):
                for v, mm in sweep:
                    k = f"{v}|model|{s}|{b}"
                    if k not in G:
                        continue
                    st = G[k]["stats"]
                    lines.append(f"| {mm} | {s} | {b} | {G[k]['n_runs']} | {_cell(st, 'landing_speed_mps')} | "
                                 f"{_cell(st, 'floor_peak_bw_skid')} | {_f(st['floor_peak_bw_legs'].get('max'), 2)} | "
                                 f"{len(G[k]['tipped_over_runs'])} | {_f(st['e_bus_surplus_j'].get('max'), 2)} |")
        lines.append("")
    st_rows = [k for v in ("adopted", "skid") for c in ("model", "impratio10")
               for k in sorted((k for k in G if k.split("|")[0] == v and k.split("|")[1] == c),
                               key=lambda k: (k.split("|")[2] != "stand", k.split("|")[2],
                                              order_b.index(k.split("|")[3])))]
    if any(G[k]["stats"].get("settle_sink_m", {}).get("n") for k in st_rows):
        lines += ["## Settling from 1 s after the reference (the cut for a sit, the stop input for a collapse)", "",
                  "Chassis height at fixed times after the reference (median over the row's runs), the further sink "
                  "from 1 s after it to the end of the 8 s window, the largest distance one loaded sole slid along the "
                  "floor in that time, the sole-samples (1 ms, per sole) in which a loaded sole slid faster than "
                  f"{1000 * SETTLE_SLIDE_MPS:g} mm/s, and the soles' friction use |Ft| / Fn in those samples (median / "
                  "95th percentile / max; the soles' friction coefficient, mu, in its own column). A slide at a "
                  "friction use under the coefficient is the soft contact's creep, not a slip. Last column: in how "
                  "many of the row's runs a leg joint of each kind rests on its range limit (its hard stop) at the end "
                  "of the window, and the largest such limit torque: what holds the unpowered legs.", "",
                  "| variant | contact | start | stop | runs | chassis m at +0.1 / +1 / +2 / +4 s / end | sink after +1 s m "
                  "(median / max) | sole slide after +1 s mm (median / max) | sliding sole-samples (median / max) |"
                  " friction use while sliding: median of p50 / max of p95 / max | mu |"
                  " runs ending with joints on their range limits: " + " / ".join(END_LIMIT_KINDS) + " (largest limit"
                  " torque N m) |", "|" + "---|" * 12]
        for k in st_rows:
            v, c, s0, b = k.split("|")
            st = G[k]["stats"]
            if not st.get("settle_sink_m", {}).get("n"):
                continue
            zz = " / ".join(_f(st[f"settle_z_after_{x}s_m"].get("median"), 3) for x in ("0.1", "1", "2", "4"))
            zz += f" / {_f(st['z_end_m'].get('median'), 3)}"
            fr = (f"{_f(st['settle_sliding_ratio_p50'].get('median'), 2)} / "
                  f"{_f(st['settle_sliding_ratio_p95'].get('max'), 2)} / {_f(st['settle_sliding_ratio_max'].get('max'), 2)}")
            mu = G[k].get("settle_mu")
            lines.append(f"| {v} | {c} | {s0} | {b} | {G[k]['n_runs']} | {zz} | {_cell(st, 'settle_sink_m', 3)} | "
                         f"{_cell(st, 'settle_sole_slide_mm', 1)} | {_cell(st, 'settle_sliding_samples', 0)} | {fr} | "
                         f"{_f(mu, 2)} | " + " / ".join(str(G[k].get("end_limit_runs", {}).get(kd, "-"))
                                                       for kd in END_LIMIT_KINDS)
                         + f" ({_f(G[k].get('end_limit_max_nm'), 1)}) |")
        lines.append("")
    D = summary["desk"]
    lines += ["## Desk comparisons", "",
              "Over every variant and contact setting (the worst case is named). The B7 pulse is an ENVELOPE: the largest "
              "bus surplus of one run with that run's own span, beside the largest peak powers over all runs (often "
              "other runs); the worst-energy run's own peaks are in their own column.", "",
              "| sit | runs | max bus surplus J (case) | its span s | its own peak W (1 / 10 / 100 ms) |"
              " largest peak W over all runs (1 / 10 / 100 ms) |"
              " max beyond 25 / 30 / 40 W hotel J | max one-leg gross J | ten pulses J | resistance for the largest peak at "
              "52.5 V, ohm | clamp current at the largest peak A |", "|" + "---|" * 11]
    pb = D["power_board"]
    for b, v in D["by_sit"].items():
        sp = v["its_span_s"]
        lines.append(f"| {b} | {v['n_runs']} | {_f(v['max_bus_surplus_j'])} ({', '.join(str(x) for x in v['max_bus_surplus_case'])}) | "
                     f"{'-' if not sp else _f(sp[1] - sp[0], 3)} | {_f(v.get('its_peak_w'), 1)} / "
                     f"{_f(v.get('its_peak_10ms_w'), 1)} / {_f(v.get('its_peak_100ms_w'), 1)} | "
                     f"{_f(v['max_peak_w'], 1)} / {_f(v['max_peak_10ms_w'], 1)} / "
                     f"{_f(v['max_peak_100ms_w'], 1)} | "
                     + " / ".join(_f(v['max_beyond_hotel_j'][f'{h:g}']) for h in HOTEL_W)
                     + f" | {_f(v['max_leg_gross_j'])} | {_f(pb['ten_pulses_j'][b], 1)} | "
                       f"{_f(pb['resistance_for_peak_ohm'][b], 0)} | {_f(pb['clamp_current_at_peak_a'][b], 2)} |")
    lines += ["", "Ceilings on the mechanical energy the drives can take net while powered (DESK, energy balance on the "
              "simulated motion, before the linkage and copper losses; max over the sit's runs): the powered-phase "
              "ceiling is the kinetic energy at the stop input plus the potential energy released before the cut; it holds "
              "for any sit that reaches the same height before the cut. It bounds the drives' net take (what they brake "
              "less what they put in), not their gross braking: a drive's own positive work can be braked out again, so "
              "the gross braking at the joints can exceed it, and so could the bus surplus of a routine that pushes and "
              "brakes in turn. The full-drop ceiling adds the drop after the cut, for a routine that lowered the body all "
              "the way while powered. Seconds: of the two 50 W resistors' 100 W continuous rating.", "",
              "| sit | KE at stop input J | PE released while powered J | powered ceiling J (s at 100 W) |"
              " drives' net take J (runs over the ceiling) | gross braking at the joints J (runs over the ceiling) |"
              " modelled bus surplus J (runs over the ceiling) | full-drop ceiling J (s at 100 W) |",
              "|---|---|---|---|---|---|---|---|"]
    for b, v in D["by_sit"].items():
        lines.append(f"| {b} | {_f(v['max_ke_at_stop_input_j'])} | {_f(v['max_pe_released_powered_j'])} | "
                     f"{_f(v['max_powered_ceiling_j'])} ({_f(pb['resistor_seconds_at_rating_for_the_powered_ceiling_s'][b], 3)}) | "
                     f"{_f(v['max_drive_net_take_j'])} ({v['n_runs_net_take_over_ceiling']} of {v['n_runs']}) | "
                     f"{_f(v['max_braking_at_joints_j'])} ({v.get('n_runs_braking_over_ceiling', '-')} of {v['n_runs']}) | "
                     f"{_f(v['max_bus_surplus_j'])} "
                     f"({v['n_runs_bus_surplus_over_ceiling']} of {v['n_runs']}) | "
                     f"{_f(v['max_full_drop_ceiling_j'])} ({_f(pb['resistor_seconds_at_rating_for_the_full_drop_ceiling_s'][b], 3)}) |")
    bc = D["bus_capacitance"]
    lines += ["", f"Bus capacitance (DESK): {bc['j_per_mf_50_to_52.5']:.3f} J per mF from 50.0 to 52.5 V, "
              f"{bc['j_per_mf_49.2_to_52.5']:.3f} J per mF from 49.2 V, {bc['j_per_mf_50_to_55']:.3f} J per mF from 50.0 to "
              f"55 V; capacitance that alone holds the worst hard sit under 52.5 V: "
              f"{_f(bc['mf_to_hold_worst_hard_sit_under_52.5_from_50'], 1)} mF, under 55 V: "
              f"{_f(bc['mf_to_hold_worst_hard_sit_under_55_from_50'], 1)} mF.", "",
              "Back-EMF threshold (DESK; output rpm at which a drive's line back-EMF reaches a bus held at that voltage; "
              "below it a drive with its bridge off pushes no current into a held bus, but into a decayed link it "
              "rectifies at any speed): "
              + "; ".join(f"{dv} " + ", ".join(f"{_f(x, 0)} rpm at {k}" for k, x in t.items())
                          for dv, t in D["back_emf_threshold_rpm"].items())
              + ". From Kt instead, at 42 V: " + ", ".join(f"{dv} {_f(x, 0)} rpm" for dv, x in
                                                         D["back_emf_threshold_rpm_from_kt"].items())
              + ". Peak drive speeds over every collapse: " + ", ".join(f"{g} {_f(x, 0)} rpm" for g, x in
                                                                   D["collapse_peak_rpm_by_group"].items()) + ".", "",
              f"Pack charge acceptance: {D['pack_charge_acceptance']}.", ""]
    return "\n".join(lines)


def _rounded(x, sig: int = 4):
    """Floats to ``sig`` significant figures (the summary is committed; keep it small)."""
    if isinstance(x, float):
        return float(f"{x:.{sig}g}") if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _rounded(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v, sig) for v in x]
    return x


STATS_NOTE = ("groups.*.stats: per measure n (runs with a value; left out when it equals the row's n_runs), n_none "
              "(runs without one; left out when 0), min, median, max; floats to 4 significant figures")


def compact(summary: dict) -> dict:
    """The summary as committed: rounded, with each stat's ``n`` and ``n_none`` left out where they say nothing
    (STATS_NOTE), to keep the file small."""
    s = _rounded(summary)
    for g in s["groups"].values():
        for st in g["stats"].values():
            if st.get("n_none") == 0:
                st.pop("n_none")
            if st.get("n") == g["n_runs"]:
                st.pop("n")
    s["meta"] = {**s["meta"], "stats_note": STATS_NOTE}
    return s


def cmd_summarise(args) -> int:
    raw = json.loads(gzip.decompress(Path(args.raw).read_bytes()))
    s = summarise(raw)
    s["meta"] = {**s["meta"], "summarised_by": _script_state()}  # the run's own script state stays as recorded
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sit60_summary.json").write_text(json.dumps(compact(s), separators=(",", ":")) + "\n")
    (out / "sit60_tables.md").write_text(tables(s))
    print(f"wrote {out / 'sit60_summary.json'} and {out / 'sit60_tables.md'} ({len(s['groups'])} rows)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--quick", action="store_true", help="one seed, the headline starts only")
    s = sub.add_parser("summarise")
    s.add_argument("--raw", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_summarise(args)


if __name__ == "__main__":
    raise SystemExit(main())
