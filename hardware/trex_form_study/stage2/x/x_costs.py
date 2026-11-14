"""Stage 2 S2-2: what the expressive head (``-X``) costs, per body, against today's head (``STAGE2_PLAN.md``
section 3; DECISION_LOG P29; research note ``stage2/research/research_look.md`` [look] section 5).  Simulation only;
nothing is adopted (P24), and P2's deferral of a jaw for the build stands.  It writes ``stage2/x/x_costs.json``;
``stage2/x/X_COSTS.md`` reports it.  Each block of the JSON names its source.

Inputs, each checked before use:

* the four bodies rebuilt from their records (``refs.source``: ``pipeline.build_candidate``, the XML's sha256
  against the record's ``xml_sha256`` and the registry's ``SHA256_PREFIX``): ``60-N-T-10`` and ``75-M-H-10``
  (today's head) and ``60-N-T-10-X`` and ``75-M-H-10-X``, at full-precision home poses;
* Stage 1's nominal and animal-pace reference arrays of each body (``stage1/runs/2026-11-08/<body>/refs/`` and
  ``stage2/x/<body>/refs/``, git-ignored), each checked by the sha256 its ``refs.json`` records and regenerated here
  (``refs.generate``), so the screen below is the regenerated walk and the arrays are shown to equal it;
* the round-two records: the cases' candidate records (each -X record's ``head_x`` block) and envelope screens;
* the Stage 1 plant (``plant.build``) of each -X body on the hard floor.

Parts (``COSTS_RULES``):

1. ``statics``: mass, the robot's centre of mass, and the single-support holds as round one's screens compute them
   (``sizing.requirements``: ``statics.single_support_hold`` with the swing foot lifted 0.05 H, trunk roll searched;
   ``DEFINITIONS.md``), with each case's sizing settings (``refine._init``), on each body at its own home and on the
   -X body posed at today's home, all against the records they reproduce;
2. ``walks``: the Stage 0 walking screen of today's nominal (``sizing.NOMINAL``, Froude 0.012) and animal-pace
   (``envelope.FAST``, Froude 0.05) references on both heads, the -X bodies' new joints held still at 0 (the
   generator moves only the legs, the neck pitch and the tail): drive ratios as the round-two envelope screen prices a
   walk (``envelope._ratios``: RMS over the sustained planning torque A1, 1.25 x peak over the planning peak, 1.15 x
   speed over the speed at the bus), the passive terms of the head and tail drives (informational, as Stage 0 leaves
   them out), contact and the clearance screen's walking gates, then the ratios that move between the heads; beside
   them the round-two envelope screens' records of both heads;
3. ``moves``: the character moves on the STS3215s with the MJCF's default joint damping and friction loss
   (``designlab/mjcf.py``: 0.2 N m s/rad and 0.15 N m on every hinge, unsourced): a 0.6 rad gaze shift in 0.5 and
   0.8 s (neck yaw, -0.3 to +0.3 rad), a 0.6 rad jaw opening in 0.4 s (and its closing), and the open jaw held, each a
   minimum-jerk profile from rest to rest.  Read two ways: rigid-body inverse dynamics with the trunk held at the
   standing home (``mj_fullM`` and ``qfrc_bias``, the armature included, as ``statics.contact_torques``) plus
   ``damping x rate + friction loss x sign(rate)``; and the same moves in the Stage 1 plant, standing on the hard
   floor, commanded with the rigid-body feedforward only (as Stage 1's references carry no damping or friction) and
   read from the servo's force.  Against the STS3215's planning limits (``catalog``: sustained 0.57 x rated, peak 0.70 x
   peak, rated speed), raw and with the sizing's margins (1.25 x peak, 1.15 x speed); and the fastest 0.6 rad move
   within each.

Run from hardware/trex_form_study with PYTHONPATH=. (at most ``--procs`` processes, one per body pair):

    python stage2/x/x_costs.py [--procs 2] [--out stage2/x/x_costs.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
STUDY = RUN.parents[1]  # hardware/trex_form_study
OUT = RUN / "x_costs.json"
#: The body pairs: (today's body, its -X body, today's refine case, the -X case).
PAIRS = (("60-N-T-10", "60-N-T-10-X", "narrow_carpet_60", "head_x_60"),
         ("75-M-H-10", "75-M-H-10-X", "feet_75", "head_x_75"))
#: Where each body's Stage 1 reference arrays are (``refs.json`` beside ``refs/``).
REFS_DIRS = {"60-N-T-10": "stage1/runs/2026-11-08/60-N-T-10", "75-M-H-10": "stage1/runs/2026-11-08/75-M-H-10",
             "60-N-T-10-X": "stage2/x/60-N-T-10-X", "75-M-H-10-X": "stage2/x/75-M-H-10-X"}
#: The two references screened (``refs.walk_specs``): the nominal walk and the animal pace (Froude 0.05).
WALKS = ("nominal", "fr05")
#: A drive ratio "moves" when the heads differ by at least this much of the limit.
MOVE_THRESHOLD = 0.005
#: The head and tail drives whose passive terms the walk screen reports (group, joint).
PASSIVE = (("neck", "neck_pitch"), ("tail", "tail_yaw"), ("neck_yaw", "neck_yaw"), ("jaw", "jaw"))
#: The clearance screen's walking gates the walk screen reads (as Stage 2's look ladder reads them).
CLEARANCE_GATES = ("swing_clearance", "joint_range_margin", "interference_free_home", "interference_free_nominal_gait",
                   "component_overlap_free", "tail_clearance")
#: The character moves (minimum-jerk, rest to rest): joint, start angle, signed amplitude, duration.
MOVES = {
    "gaze_0.6rad_0.5s": dict(joint="neck_yaw", start=-0.3, amplitude=0.6, seconds=0.5),
    "gaze_0.6rad_0.8s": dict(joint="neck_yaw", start=-0.3, amplitude=0.6, seconds=0.8),
    "jaw_open_0.6rad_0.4s": dict(joint="jaw", start=0.0, amplitude=0.6, seconds=0.4),
    "jaw_close_0.6rad_0.4s": dict(joint="jaw", start=0.6, amplitude=-0.6, seconds=0.4),
}
#: Static holds at the standing home: joint at an angle (the head's other joints at home).
HOLDS = {"jaw_open_0.6rad": dict(joint="jaw", q=0.6), "gaze_held_0.6rad": dict(joint="neck_yaw", q=0.6)}
#: Samples per move for the inverse dynamics (the profile is analytic, so this sets only the peak's resolution).
SAMPLES = 2001
#: The sizing's margins (``sizing.PEAK_MARGIN``, ``sizing.SPEED_MARGIN``), restated in the rule text.
MIN_JERK_PEAK_SPEED = 1.875  # x amplitude / duration: a minimum-jerk profile's peak rate, at mid-move
#: The busiest gaze rhythm the research gives: a retarget every 2 s ([look] 5.1, "every 2-6 s").
GAZE_PERIOD_S = 2.0
#: In the plant: the head is first brought to the move's start (minimum-jerk over ``LEAD_S``) and settles
#: ``SETTLE_S``; after the move it holds ``HOLD_S`` (the open jaw: ``JAW_HOLD_S``, then the closing move).
LEAD_S, SETTLE_S, HOLD_S, JAW_HOLD_S = 0.5, 1.0, 1.0, 1.5
#: The plant's reading window: the move plus this much after it (the servo's lag).
WINDOW_TAIL_S = 0.1
#: The durations (s) of the 0.6 rad moves run again in the plant, to read the fastest within the limits there.
PLANT_SWEEP = {"gaze": (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80), "jaw": (0.40, 0.45, 0.50, 0.55, 0.60)}

COSTS_RULES = dict(
    statics=("sizing.requirements with each case's sizing settings (refine._init) on the body rebuilt from its record "
             "(refs.source): the hold is the quasi-static single-support hold (statics.single_support_hold: one foot "
             "lifted 0.05 H, the centre of mass over the stance pad, the trunk's roll searched over +-6 deg; "
             "DEFINITIONS.md 'Single-support hold'), the largest of both sides' |torque| on the drive's coordinate; "
             "double_support the static torque at home on both feet; rms the nominal walk's; the ratios over the drive's "
             "sustained planning torque A1 (catalog.SUSTAINED_FRACTION 0.57 x rated); cont = max(rms, hold) / A1, the "
             "sizing's continuous requirement.  'at_todays_pose': the -X body posed at today's home (the new joints at "
             "0), as refine.head_x_at_pick_pose; mass and centre of mass from the compiled model (statics.Model)"),
    walks=("each reference regenerated (refs.generate: gait.generate and gait.inverse_dynamics at round two's "
           "accuracy, the record's model and home) and checked equal to Stage 1's saved arrays; its drive ratios by "
           "envelope._ratios against the record's drives at the record's bus (rms / A1, 1.25 x peak / planning peak, "
           "1.15 x speed / speed at the bus); passive: tau + damping x rate + friction loss x sign(rate) on the head and "
           "tail drives (informational: Stage 0 leaves the MJCF's damping and friction out); contact from the inverse "
           "dynamics (single-support centre-of-pressure margin; the split's largest per-foot friction ratio, with the "
           "lighter loaded foot's share of the weight at that frame) and the split-free friction need of the whole robot "
           "(|horizontal| / vertical of m (a_com - g)); clearance.screen (gates r2) with the "
           "reference standing in for the nominal walk, on the round-two MJCF; a ratio moves when the heads differ by at "
           f"least {MOVE_THRESHOLD} of the limit"),
    moves=("minimum-jerk profiles (q = A (10 s^3 - 15 s^4 + 6 s^5), s = t / T; peak rate 1.875 A / T) at the -X body's "
           "standing home (full precision, refs.source), the other joints at home.  inverse_dynamics: the trunk held "
           "still, (M qdd + qfrc_bias) on the joint's dof (mj_fullM, armature included) plus the MJCF's joint damping x "
           "rate and friction loss x sign(rate) (the compiled model's dof_damping, dof_frictionloss); its parts at the "
           "peak; the neck-pitch torque over the move; what the feet must take to hold the trunk still (the free "
           "joint's generalized force: the horizontal force and the yaw moment about the vertical through the support "
           "centre, the move's inertia only, the passive terms being internal).  plant: plant.build on the hard floor "
           "(settled), the head brought to the start and settled, then the move commanded every 2 ms step "
           "(Plant.command: target, rate and the rigid-body feedforward, without damping or friction, as Stage 1's "
           "references), no delay; read: the servo's actuator force (the plant's envelope applied), the tracking error, "
           "the trunk's yaw.  Ratios: peak over the planning peak, raw and x 1.25 (sizing.PEAK_MARGIN); speed over the "
           "rated speed, raw and x 1.15 (sizing.SPEED_MARGIN); RMS over the move and over the busiest gaze rhythm "
           f"(one shift every {GAZE_PERIOD_S:g} s, the hold between free of gravity on a vertical yaw axis) over A1, "
           "which is a heat limit over minutes, not seconds.  fastest: the shortest duration whose peak torque and peak "
           "rate are within the limit (bisection on the inverse dynamics; both fall as the move slows); the plant's "
           "sweep runs each 0.6 rad move again at the durations of PLANT_SWEEP and reads the fastest within there"),
)


# ------------------------------------------------------------------------------------------------
# Setup and checks
# ------------------------------------------------------------------------------------------------
def _setup():
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    import designlab  # noqa: F401  (selects the GL backend before mujoco is imported)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path) -> str:
    """``path`` relative to hardware/trex_form_study, read lexically: the round-two models of today's bodies are
    links into the main checkout, which resolving would follow."""
    return Path(os.path.relpath(Path(path).absolute(), STUDY)).as_posix()


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _rel_diff(a: float, b: float) -> float:
    return 0.0 if a == b else abs(a - b) / max(abs(a), abs(b))


# ------------------------------------------------------------------------------------------------
# 1. Statics: mass, centre of mass, single-support holds
# ------------------------------------------------------------------------------------------------
def pose_at(M_to, M_from, q_from) -> np.ndarray:
    """``M_to``'s qpos with every joint ``M_from`` has at ``q_from`` and the others at their reference (0): the -X
    body posed at today's home, as ``refine.head_x_at_pick_pose`` poses it."""
    import mujoco

    m0, mx = M_from.m, M_to.m
    q = mx.qpos0.copy()
    for j in range(m0.njnt):
        name = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
        k = mujoco.mj_name2id(mx, mujoco.mjtObj.mjOBJ_JOINT, name)
        n = 7 if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        a0, ax = int(m0.jnt_qposadr[j]), int(mx.jnt_qposadr[k])
        q[ax:ax + n] = q_from[a0:a0 + n]
    return q


def _group(req: dict, design, g: str) -> dict:
    r = req["groups"][g]
    a = design.act(g)
    return dict(actuator=a.id, sustained_Nm=a.sustained, peak_Nm=a.peak, hold_Nm=r["hold"],
                double_support_Nm=r["double_support"], nominal_rms_Nm=r["rms"], hold_over_A1=r["hold"] / a.sustained,
                cont_over_A1=r["cont"] / a.sustained, peak_req_over_peak=r["peak_req"] / a.peak)


def _static_torque(M, q, joint: str) -> float:
    """The static torque on ``joint`` at pose ``q`` standing on both feet (``statics.contact_torques``)."""
    from designlab import statics

    tau, _ = statics.contact_torques(M, q, support=("left", "right"))
    return float(tau[M.names.index(joint)])


def statics_costs(t: dict, x: dict, case_t: str, case_x: str) -> dict:
    """Part 1 (``COSTS_RULES["statics"]``) for one pair: ``t`` and ``x`` are ``refs.source`` results."""
    from designlab import refine, sizing, statics

    refine._init(case_t)
    req_t = sizing.requirements(t["M"], t["q_home"], t["H"], t["layout"])
    refine._init(case_x)
    req_x = sizing.requirements(x["M"], x["q_home"], x["H"], x["layout"])
    q_tp = pose_at(x["M"], t["M"], t["q_home"])
    req_xp = sizing.requirements(x["M"], q_tp, x["H"], x["layout"])

    def com(M, q):
        M.forward(q)
        return M.com()

    c_t, c_x, c_xp = com(t["M"], t["q_home"]), com(x["M"], x["q_home"]), com(x["M"], q_tp)
    m = x["M"].m
    q_turn = x["q_home"].copy()
    q_turn[m.jnt_qposadr[statics.jid(m, "neck_yaw")]] = 0.6
    c_turn = com(x["M"], q_turn)
    q_jaw = x["q_home"].copy()
    q_jaw[m.jnt_qposadr[statics.jid(m, "jaw")]] = 0.6
    groups = sizing.groups(t["M"].names)
    loads = {}
    for g in groups:
        a, b, c = _group(req_t, t["design"], g), _group(req_x, x["design"], g), _group(req_xp, x["design"], g)
        loads[g] = dict(today=a, x=b, x_at_todays_pose=c,
                        hold_change_Nm=b["hold_Nm"] - a["hold_Nm"],
                        hold_over_A1_change=b["hold_over_A1"] - a["hold_over_A1"],
                        cont_over_A1_change=b["cont_over_A1"] - a["cont_over_A1"])
    for g in sizing.HEAD_X_GROUPS:
        loads[g] = dict(x=_group(req_x, x["design"], g), x_at_todays_pose=_group(req_xp, x["design"], g))
    neck = dict(
        joint="neck_pitch",
        a1_Nm=x["design"].act("neck").sustained,
        static_home_Nm=dict(today=_static_torque(t["M"], t["q_home"], "neck_pitch"),
                            x=_static_torque(x["M"], x["q_home"], "neck_pitch"),
                            x_jaw_open_0_6=_static_torque(x["M"], q_jaw, "neck_pitch"),
                            x_gaze_held_0_6=_static_torque(x["M"], q_turn, "neck_pitch")),
        research_5_2=dict(text="the jaw servo in the head raises the 60 cm static hold from 0.75 to 0.84 of A1 (0.92 "
                               "with the yaw servo in the head); at 75 cm 0.66, 0.71 and 0.75",
                          source="stage2/research/research_look.md section 5.2"))
    neck["static_home_over_A1"] = {k: abs(v) / neck["a1_Nm"] for k, v in neck["static_home_Nm"].items()}
    # what this reproduces: the records' head_x block (and today's own record)
    rec_t, rec_x = t["record"], x["record"]
    hx = rec_x["head_x"]
    checks = []

    def check(label, ours, theirs):
        checks.append(dict(what=label, measured=ours, record=theirs, rel_diff=_rel_diff(ours, theirs)))

    check("mass_kg today (record metrics.mass)", t["M"].mass(), rec_t["metrics"]["mass"])
    check("mass_kg -X (record metrics.mass)", x["M"].mass(), rec_x["metrics"]["mass"])
    check("com_ahead_of_hip today (record head_x.com_ahead_of_hip_m.base)", float(c_t[0]), hx["com_ahead_of_hip_m"]["base"])
    check("com_ahead_of_hip -X (head_x.com_ahead_of_hip_m.head_x)", float(c_x[0]), hx["com_ahead_of_hip_m"]["head_x"])
    check("com_ahead_of_hip -X at today's home (head_x.com_ahead_of_hip_m.head_x_at_pick_pose)", float(c_xp[0]),
          hx["com_ahead_of_hip_m"]["head_x_at_pick_pose"])
    for g in groups:
        ab = hx["against_base"][g]
        check(f"{g} hold today (head_x.against_base.{g}.base_hold_Nm)", loads[g]["today"]["hold_Nm"], ab["base_hold_Nm"])
        check(f"{g} hold today (today's record margins.{g}.hold)", loads[g]["today"]["hold_Nm"], rec_t["margins"][g]["hold"])
        check(f"{g} hold -X (head_x.against_base.{g}.hold_Nm)", loads[g]["x"]["hold_Nm"], ab["hold_Nm"])
        check(f"{g} cont ratio -X (head_x.against_base.{g}.ratios.cont)", loads[g]["x"]["cont_over_A1"], ab["ratios"]["cont"])
        check(f"{g} cont ratio today (head_x.against_base.{g}.base_ratios.cont)", loads[g]["today"]["cont_over_A1"],
              ab["base_ratios"]["cont"])
        ap = hx["against_base_at_pick_pose"][g]
        check(f"{g} hold -X at today's home (head_x.against_base_at_pick_pose.{g}.hold_Nm)",
              loads[g]["x_at_todays_pose"]["hold_Nm"], ap["hold_Nm"])
    worst = max(checks, key=lambda c: c["rel_diff"])

    def power(rec):
        p = rec["power"]
        return dict(total_w=p["total_w"], drives_w=p["drives_w"], mean_w={g: v["mean_w"] for g, v in p["drives"].items()})

    cid = t["record"]["id"]
    records_costs = dict(
        sources=[f"round2/refine/{case_t}/candidates/{cid}.json", f"round2/refine/{case_x}/candidates/{cid}.json"],
        rule="copied from the case records: cost.estimate (USD: low, nominal, high), power.walking_power on the nominal "
             "walk (W, each group's drives) and power.runtime on the record's pack; [today, -X]; the yaw stage's own "
             "bearing is in neither the mass nor the cost (HEAD_X.md section 7)",
        cost_usd={k: [rec_t["cost"][k], rec_x["cost"][k]] for k in ("low", "nominal", "high")},
        walking_power=dict(today=power(rec_t), x=power(rec_x)),
        runtime_min=[rec_t["runtime"]["minutes"], rec_x["runtime"]["minutes"]],
        pack=[rec_t["runtime"]["pack"], rec_x["runtime"]["pack"]])
    return dict(
        rule=COSTS_RULES["statics"], records_costs=records_costs,
        mass_kg=dict(today=t["M"].mass(), x=x["M"].mass(), added=x["M"].mass() - t["M"].mass(),
                     added_parts=[dict(name=p["name"], body=p["body"], mass_kg=p["mass_kg"], source=p["source"])
                                  for p in hx["parts"]],
                     head_neck=dict(today=rec_t["metrics"]["head_neck_mass"], x=rec_x["metrics"]["head_neck_mass"])),
        com_m=dict(today_at_home=c_t.tolist(), x_at_home=c_x.tolist(), x_at_todays_pose=c_xp.tolist(),
                   x_gaze_held_0_6=c_turn.tolist(),
                   shift_own_homes_mm=(1e3 * (c_x - c_t)).tolist(), shift_same_pose_mm=(1e3 * (c_xp - c_t)).tolist(),
                   gaze_held_0_6_shift_mm=(1e3 * (c_turn - c_x)).tolist(),
                   note="world frame, x forward, y left, z up, the hips at x = 0 (metrics.measure's com_ahead_of_hip is "
                        "x).  shift_same_pose_mm: the added parts alone (the -X body at today's home); "
                        "shift_own_homes_mm: each body at its own re-solved stance, whose feet move under the centre of "
                        "mass (HEAD_X.md section 6)",
                   research_5_2="the centre of mass moves 5.7 mm (60 cm) or 5.1 mm (75 cm) forward (est.)"),
        loads=loads, neck_pitch=neck,
        records_reproduced=dict(checks=checks, n=len(checks), max_rel_diff=worst["rel_diff"], worst=worst["what"],
                                all_match=all(c["rel_diff"] <= 1e-9 for c in checks)),
        sizing_walks=list(sizing.EXTRA_WALKS))


# ------------------------------------------------------------------------------------------------
# 2. The Stage 0 walking screen of the nominal and animal-pace references
# ------------------------------------------------------------------------------------------------
def _passive(M, design, tau, omega) -> dict:
    from designlab import sizing, statics

    m = M.m
    out = {}
    for g, jn in PASSIVE:
        if jn not in M.names:
            continue
        i = M.names.index(jn)
        dof = m.jnt_dofadr[statics.jid(m, jn)]
        b, f = float(m.dof_damping[dof]), float(m.dof_frictionloss[dof])
        tp = tau[:, i] + b * omega[:, i] + f * np.sign(omega[:, i])
        a = design.act(g)
        out[g] = dict(damping_Nms_per_rad=b, frictionloss_Nm=f, rms_Nm=float(np.sqrt(np.mean(tp**2))),
                      peak_Nm=float(np.max(np.abs(tp))), cont_ratio=float(np.sqrt(np.mean(tp**2)) / a.sustained),
                      peak_ratio=float(sizing.PEAK_MARGIN * np.max(np.abs(tp)) / a.peak))
    return out


def _clearance(src: dict, traj: dict) -> dict:
    from designlab import clearance
    from designlab.stage1 import plant

    xml, _ = plant.model_files(src["body"])
    plant.verify(src["body"], xml)
    cl = clearance.screen(str(xml), src["q_home"], {"nominal": dict(q=traj["q"], phases=traj["phases"],
                                                                    ik_ok=traj["ik_ok"])}, H=src["H"], gates="r2")
    s = cl["summary"]
    sw = s["swing_clearance"]["nominal"]
    gates = {k: s["pass"].get(k) for k in CLEARANCE_GATES}
    return dict(gates=gates, walking_gates=all(v is True for v in gates.values()),
                mid_swing_min_frac_H=sw["mid_swing_min_frac_H"], mid_swing_min_mm=sw["mid_swing_min_mm"],
                screen="designlab.clearance.screen (gates r2), the reference standing in for the nominal walk",
                model=_rel(xml))


def walk_screen(src: dict, name: str) -> dict:
    """One reference of one body (``COSTS_RULES["walks"]``)."""
    from designlab import envelope, sizing, statics
    from designlab.stage1 import refs

    d = STUDY / REFS_DIRS[src["body"]]
    entry = json.loads((d / "refs.json").read_text())["references"][name]
    path = d / entry["file"]
    sha_ok = sha256(path) == entry["sha256"]
    g = refs.generate(src, name)
    with np.load(path, allow_pickle=False) as z:
        saved = {k: z[k] for k in ("q", "tau", "omega", "cop_margin", "joints")}
    M = src["M"]
    tau, omega = g["idr"]["tau"], g["idr"]["omega"]
    same = dict(q=bool(np.array_equal(g["traj"]["q"], saved["q"])), tau=bool(np.array_equal(tau, saved["tau"])),
                omega=bool(np.array_equal(omega, saved["omega"])),
                cop_margin=bool(np.array_equal(g["idr"]["cop_margin"], saved["cop_margin"], equal_nan=True)),
                joints=list(saved["joints"]) == list(M.names))
    bus = src["record"]["selection"]["bus"]
    ms = sizing._motor_space(M.names, tau, omega, src["layout"])
    drives = envelope._ratios(src["design"], bus, ms)
    worst = max(drives.items(), key=lambda kv: kv[1]["worst"])
    info = g["traj"]["info"]
    s = g["summary"]
    still = {}
    for jn in ("neck_yaw", "jaw"):
        if jn in M.names:
            i = M.names.index(jn)
            qa = M.m.jnt_qposadr[statics.jid(M.m, jn)]
            still[jn] = dict(q_min=float(np.min(g["traj"]["q"][:, qa])), q_max=float(np.max(g["traj"]["q"][:, qa])),
                             rate_max=float(np.max(np.abs(omega[:, i]))), torque_rms_Nm=float(np.sqrt(np.mean(tau[:, i] ** 2))),
                             torque_peak_Nm=float(np.max(np.abs(tau[:, i]))))
    return dict(
        body=src["body"], reference=name, arrays=dict(file=_rel(path), sha256=entry["sha256"], sha256_matches=sha_ok,
                                                      regenerated_equal=same, all_equal=all(same.values()) and sha_ok),
        walk=dict(speed_m_s=float(info["speed"]), stride_m=float(info["stride"]), cadence_steps_s=float(info["cadence_hz"]),
                  period_s=float(info["period"]), froude=float(g["how"]["spec"]["froude"]),
                  duty=float(g["how"]["spec"]["duty"])),
        bus_V=bus, drives=drives, worst_group=worst[0], worst_ratio=float(worst[1]["worst"]),
        passive=_passive(M, src["design"], tau, omega),
        contact=_contact(M, g, s), new_joints_still=still, clearance=_clearance(src, g["traj"]))


def _contact(M, g: dict, s: dict) -> dict:
    """Contact readings of one walk: the single-support centre-of-pressure margin and the largest per-foot friction
    ratio of the inverse dynamics' split (with the lighter loaded foot's share of the weight at that frame: in double
    support the split can load a nearly unloaded foot sideways), and the split-free friction need of the whole robot,
    |horizontal| / vertical of the total contact force m (a_com - g), the centre of mass's acceleration by periodic
    second differences (the stride unwrapped in x)."""
    tr, idr = g["traj"], g["idr"]
    q, dt, n = tr["q"], float(tr["dt"]), len(tr["q"])
    com = np.zeros((n, 3))
    for k in range(n):
        M.forward(q[k])
        com[k] = M.com()
    step = np.array([float(tr["info"]["stride"]), 0.0, 0.0])
    ext = np.vstack([com[-1] - step, com, com[0] + step])
    acc = (ext[2:] - 2.0 * ext[1:-1] + ext[:-2]) / dt**2
    need = np.hypot(acc[:, 0], acc[:, 1]) / (9.81 + acc[:, 2])
    fr, normal = np.asarray(idr["friction"]), np.asarray(idr["normal"])
    k = int(np.argmax(fr))
    loaded = [normal[k, j] for j in range(2) if normal[k, j] > 0.0]
    return dict(min_cop_margin_single_mm=1e3 * float(s["min_cop_margin_single"]),
                max_friction_ratio=float(s["max_friction_ratio"]),
                max_friction_ratio_frame=dict(frame=k, lighter_foot_load_frac=float(min(loaded) / normal[k].sum())),
                total_friction_need_max=float(np.max(need)), ik_ok=bool(s["ik_ok"]))


def walk_moves(a: dict, b: dict) -> dict:
    """The drive ratios that move between today's head (``a``) and the -X head (``b``) on one reference."""
    moved, still = [], []
    for g, ra in a["drives"].items():
        rb = b["drives"].get(g)
        for k in ("cont_ratio", "peak_ratio", "speed_ratio"):
            row = dict(group=g, ratio=k, today=ra[k], x=rb[k], change=rb[k] - ra[k])
            (moved if abs(row["change"]) >= MOVE_THRESHOLD else still).append(row)
    moved.sort(key=lambda r: -abs(r["change"]))
    new = [dict(group=g, ratio=k, x=b["drives"][g][k]) for g in b["drives"] if g not in a["drives"]
           for k in ("cont_ratio", "peak_ratio", "speed_ratio")]
    pas = {g: dict(today=a["passive"][g], x=b["passive"][g]) for g in a["passive"] if g in b["passive"]}
    return dict(threshold=MOVE_THRESHOLD, moved=moved, n_unmoved=len(still),
                largest_unmoved=max((abs(r["change"]) for r in still), default=0.0), new_drives=new,
                worst=dict(today=[a["worst_group"], a["worst_ratio"]], x=[b["worst_group"], b["worst_ratio"]]),
                passive=pas,
                contact=dict(today=a["contact"], x=b["contact"]),
                clearance=dict(today=a["clearance"]["walking_gates"], x=b["clearance"]["walking_gates"],
                               mid_swing_min_frac_H=[a["clearance"]["mid_swing_min_frac_H"],
                                                     b["clearance"]["mid_swing_min_frac_H"]]))


def envelope_records(case_t: str, case_x: str, cid: str) -> dict:
    """The round-two envelope screens' records of both heads (``envelope.screen``, round two's "Stage 0")."""
    pa = STUDY / "round2/refine" / case_t / "review/screening" / f"{cid}_envelope.json"
    pb = STUDY / "round2/refine" / case_x / "review/screening" / f"{cid}_envelope.json"
    a, b = json.loads(pa.read_text()), json.loads(pb.read_text())
    keys = ("fast_walk_worst", "fast_walk_speed_ms", "stop_max_one_foot_decel_ms2", "push_max_dv_ms",
            "push_max_impulse_ns", "push_max_step_over_H", "session_worst")
    fw = {}
    for g, rb in b["fast_walk"]["groups"].items():
        ra = a["fast_walk"]["groups"].get(g)
        fw[g] = {k: [None if ra is None else ra[k], rb[k]] for k in ("cont_ratio", "peak_ratio", "speed_ratio")}
    heat = {g: [a["session_heat"]["groups"].get(g, {}).get("ratio"), v["ratio"]]
            for g, v in b["session_heat"]["groups"].items()}
    return dict(sources=[_rel(pa), _rel(pb)], summary={k: [a["summary"][k], b["summary"][k]] for k in keys},
                fast_walk=fw, session_heat=heat,
                note="[today, -X]; fast_walk: the animal pace by turning.generate at zero yaw rate (the CoM tracked to "
                     "1e-6 m), not refs' gait.generate; session_heat: RMS over envelope.SESSION's mix over A1")


# ------------------------------------------------------------------------------------------------
# 3. The character moves on the STS3215s
# ------------------------------------------------------------------------------------------------
def min_jerk(t, T: float, A: float):
    s = np.clip(np.asarray(t, float) / T, 0.0, 1.0)
    inside = (np.asarray(t) >= 0) & (np.asarray(t) <= T)
    q = A * (10 * s**3 - 15 * s**4 + 6 * s**5)
    qd = np.where(inside, A / T * (30 * s**2 - 60 * s**3 + 30 * s**4), 0.0)
    qdd = np.where(inside, A / T**2 * (60 * s - 180 * s**2 + 120 * s**3), 0.0)
    return q, qd, qdd


class HeadDynamics:
    """Rigid-body inverse dynamics of one head joint at the standing home, the trunk held (``COSTS_RULES``)."""

    def __init__(self, M, q_home):
        import mujoco
        from designlab import statics

        self.M, self.m, self.q_home = M, M.m, np.asarray(q_home, float)
        self.d = mujoco.MjData(self.m)
        self.Mf = np.zeros((self.m.nv, self.m.nv))
        self.root = M.root  # the body of the free joint, whose rotational dofs are in its own frame
        self.dof_pitch = int(self.m.jnt_dofadr[statics.jid(self.m, "neck_pitch")])
        M.forward(self.q_home)  # the support centre: midway between the two pads' centroids at home
        self.support = np.mean([statics._polygon_centroid(M.pad_polygon(s)) for s in statics.SIDES], axis=0)

    def joint(self, name: str):
        from designlab import statics

        j = statics.jid(self.m, name)
        return int(self.m.jnt_qposadr[j]), int(self.m.jnt_dofadr[j])

    def at(self, joint: str, q: float, qd: float, qdd: float) -> dict:
        import mujoco

        qa, dof = self.joint(joint)
        m, d = self.m, self.d
        d.qpos[:] = self.q_home
        d.qpos[qa] = q
        d.qvel[:] = 0.0
        d.qvel[dof] = qd
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, self.Mf)
        acc = np.zeros(m.nv)
        acc[dof] = qdd
        rhs = self.Mf @ acc + d.qfrc_bias
        b, f = float(m.dof_damping[dof]), float(m.dof_frictionloss[dof])
        damp, fric = b * qd, f * float(np.sign(qd))
        # what holding the trunk still takes from the feet: the free joint's force (world, at the root's origin) and
        # moment (root frame); about the vertical through the support centre, gravity adds no yaw and no horizontal
        # force, and the joint's damping and friction act between head and trunk, so these are the move's inertia
        F = rhs[0:3]
        Mo = d.xmat[self.root].reshape(3, 3) @ rhs[3:6]
        o = d.xpos[self.root][:2] - self.support
        return dict(rigid=float(rhs[dof]), damping=damp, friction=fric, total=float(rhs[dof]) + damp + fric,
                    pitch=float(rhs[self.dof_pitch]), feet_yaw=float(Mo[2] + o[0] * F[1] - o[1] * F[0]),
                    feet_horizontal=float(math.hypot(F[0], F[1])))

    def profile(self, mv: dict, n: int = SAMPLES) -> dict:
        T, A, q0 = mv["seconds"], mv["amplitude"], mv["start"]
        t = np.linspace(0.0, T, n)
        q, qd, qdd = min_jerk(t, T, A)
        rows = [self.at(mv["joint"], q0 + q[k], qd[k], qdd[k]) for k in range(n)]
        out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        out.update(t=t, q=q0 + q, qd=qd, qdd=qdd)
        return out

    def peak_total(self, mv: dict, n: int = 801) -> float:
        return float(np.max(np.abs(self.profile(mv, n)["total"])))


def move_inverse_dynamics(hd: HeadDynamics, design, mv: dict) -> dict:
    from designlab import sizing

    g = "neck_yaw" if mv["joint"] == "neck_yaw" else "jaw"
    a = design.act(g)
    spd = a.speed_at(None)
    p = hd.profile(mv)
    k = int(np.argmax(np.abs(p["total"])))
    rms = float(np.sqrt(np.mean(p["total"] ** 2)))
    peak = float(abs(p["total"][k]))
    speed = float(np.max(np.abs(p["qd"])))
    out = dict(
        joint=mv["joint"], drive=dict(group=g, actuator=a.id, sustained_Nm=a.sustained, peak_Nm=a.peak, speed_rad_s=spd),
        move=dict(mv, peak_rate_rad_s=speed, peak_acc_rad_s2=float(np.max(np.abs(p["qdd"])))),
        peak_Nm=peak, peak_at_s=float(p["t"][k]),
        peak_parts_Nm=dict(rigid=float(p["rigid"][k]), damping=float(p["damping"][k]), friction=float(p["friction"][k])),
        rigid_peak_Nm=float(np.max(np.abs(p["rigid"]))), damping_peak_Nm=float(np.max(np.abs(p["damping"]))),
        rms_Nm=rms,
        ratios=dict(peak_raw=peak / a.peak, peak_with_margin=sizing.PEAK_MARGIN * peak / a.peak,
                    speed_raw=speed / spd, speed_with_margin=sizing.SPEED_MARGIN * speed / spd,
                    rms_over_A1=rms / a.sustained),
        neck_pitch_Nm=dict(min=float(np.min(p["pitch"])), max=float(np.max(p["pitch"]))),
        feet_reaction=dict(yaw_moment_peak_Nm=float(np.max(np.abs(p["feet_yaw"]))),
                           horizontal_force_peak_N=float(np.max(p["feet_horizontal"])),
                           about="the vertical through the support centre (midway between the pads' centroids at "
                                 "home); the move's inertia only (gravity adds none, the passive terms are internal)"))
    if mv["joint"] == "neck_yaw":
        out["gaze_rhythm"] = dict(period_s=GAZE_PERIOD_S, rms_Nm=rms * math.sqrt(mv["seconds"] / GAZE_PERIOD_S),
                                  rms_over_A1=rms * math.sqrt(mv["seconds"] / GAZE_PERIOD_S) / a.sustained,
                                  note="one shift every period, the hold between at no torque (the yaw axis is "
                                       "vertical at home: gravity_at_0_6 below)")
    return out


def fastest(hd: HeadDynamics, mv: dict, torque_limit: float, speed_limit: float) -> dict:
    """The shortest duration of ``mv``'s profile whose peak torque (with the passive terms) is within ``torque_limit``
    and whose peak rate is within ``speed_limit`` (``COSTS_RULES["moves"]``)."""
    A = abs(mv["amplitude"])
    t_speed = MIN_JERK_PEAK_SPEED * A / speed_limit

    def ok(T):
        return hd.peak_total(dict(mv, seconds=T)) <= torque_limit

    lo, hi = 0.02, 5.0
    if not ok(hi):
        return dict(seconds=None, by_speed_s=t_speed, by_torque_s=None, note="no duration up to 5 s is within the torque")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return dict(seconds=max(hi, t_speed), by_speed_s=t_speed, by_torque_s=hi,
                bound="speed" if t_speed >= hi else "torque")


def plant_moves(body: str, hd: HeadDynamics) -> dict:
    """The moves in the Stage 1 plant, standing on the hard floor (``COSTS_RULES["moves"]``)."""
    from designlab import statics
    from designlab.stage1 import plant

    p = plant.build(body, "hard")
    m = p.m
    dt = float(m.opt.timestep)
    home = p.q_home[p.qadr].copy()  # the servos' home targets, joint space
    act = {dr.coordinate: i for i, dr in enumerate(p.drives)}
    drives = {dr.coordinate: dr for dr in p.drives}
    trunk = statics.bid(m, "trunk")
    nj = len(p.joints)

    def yaw_of(d) -> float:
        R = d.xmat[trunk].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def run(segments) -> tuple[np.ndarray, dict]:
        """Command ``segments`` one after another from the settled stance: each (joint, start, amplitude, seconds,
        name), a minimum-jerk move (a hold when the amplitude is 0) with the rigid-body feedforward; a joint keeps
        its last target in the later segments.  Returns the rows (global time, servo force, neck-pitch force, the
        angle before the step, its target, its rate, the trunk's yaw from the start, the servo's force limit) and each
        named segment's (start time, joint, seconds)."""
        p.reset()
        d = p.d
        targets, rows, starts = {}, [], {}
        yaw0, t_glob = yaw_of(d), 0.0
        for joint, q0, A, T, name in segments:
            j, ai, pi = p.joints.index(joint), act[joint], act["neck_pitch"]
            if name:
                starts[name] = (t_glob, joint, T)
            for k in range(int(round(T / dt))):
                q, qd, qdd = (float(v) for v in min_jerk(k * dt, T, A))
                cmd = home.copy()
                for jn, v in targets.items():
                    cmd[p.joints.index(jn)] = v
                cmd[j] = q0 + q
                rate, tau = np.zeros(nj), np.zeros(nj)
                rate[j] = qd
                tau[j] = hd.at(joint, q0 + q, qd, qdd)["rigid"]
                q_now = float(d.qpos[p.qadr[j]])
                p.command(cmd, rate, tau)
                p.step(1)  # the force read after the step is the one computed from this step's start
                rows.append((t_glob, float(d.actuator_force[ai]), float(d.actuator_force[pi]), q_now, q0 + q,
                             float(d.qvel[p.dofs[j]]), yaw_of(d) - yaw0, float(m.actuator_forcerange[ai, 1])))
                t_glob += dt
            targets[joint] = q0 + A
        return np.array(rows), starts

    def move_summary(rows, start, drive) -> dict:
        t0, _, T = start
        t, f, fp, q, qs, qd, yaw, fr = rows.T
        w = (t >= t0 - 1e-12) & (t <= t0 + T + WINDOW_TAIL_S)
        mv = (t >= t0 - 1e-12) & (t < t0 + T - 1e-12)
        k = int(np.argmax(np.abs(f[w])))
        return dict(peak_force_Nm=float(np.abs(f[w])[k]), peak_at_s=float(t[w][k] - t0),
                    rms_force_Nm=float(np.sqrt(np.mean(f[mv] ** 2))),
                    tracking_error_max_rad=float(np.max(np.abs(q[w] - qs[w]))),
                    peak_rate_rad_s=float(np.max(np.abs(qd[w]))),
                    at_envelope_steps=int(np.sum(np.abs(f[w]) >= fr[w] - 1e-9)),
                    peak_over_planning_peak=float(np.abs(f[w])[k]) / drive.peak,
                    neck_pitch_force_Nm=[float(np.min(fp[w])), float(np.max(fp[w]))],
                    trunk_yaw_excursion_deg=float(np.degrees(np.max(np.abs(yaw[w])))),
                    window_s=[0.0, T + WINDOW_TAIL_S])

    def hold_summary(rows, start) -> dict:
        t0, _, T = start
        t, f, fp, q, qs, qd, yaw, fr = rows.T
        last = (t >= t0 + T - 0.5 - 1e-12) & (t < t0 + T - 1e-12)
        return dict(force_mean_last_0_5s_Nm=float(np.mean(f[last])), q_mean_last_0_5s=float(np.mean(q[last])),
                    q_target=float(np.mean(qs[last])),
                    neck_pitch_force_mean_last_0_5s_Nm=float(np.mean(fp[last])))

    out = dict(floor="hard", settle=(p.meta.get("settle") or {}).get("status"), settled=p.settled, timestep_s=dt,
               feedforward="rigid-body inverse dynamics of the moving joint (no damping or friction), as Stage 1's "
                           "references", delay_s=0.0)
    for name in ("gaze_0.6rad_0.5s", "gaze_0.6rad_0.8s"):
        mv = MOVES[name]
        rows, starts = run([("neck_yaw", 0.0, mv["start"], LEAD_S, ""), ("neck_yaw", mv["start"], 0.0, SETTLE_S, ""),
                            ("neck_yaw", mv["start"], mv["amplitude"], mv["seconds"], name),
                            ("neck_yaw", mv["start"] + mv["amplitude"], 0.0, HOLD_S, "")])
        out[name] = move_summary(rows, starts[name], drives["neck_yaw"])
    op, cl = MOVES["jaw_open_0.6rad_0.4s"], MOVES["jaw_close_0.6rad_0.4s"]
    rows, starts = run([("jaw", 0.0, 0.0, SETTLE_S, ""),
                        ("jaw", op["start"], op["amplitude"], op["seconds"], "jaw_open_0.6rad_0.4s"),
                        ("jaw", op["start"] + op["amplitude"], 0.0, JAW_HOLD_S, "jaw_open_hold"),
                        ("jaw", cl["start"], cl["amplitude"], cl["seconds"], "jaw_close_0.6rad_0.4s"),
                        ("jaw", cl["start"] + cl["amplitude"], 0.0, HOLD_S, "")])
    out["jaw_open_0.6rad_0.4s"] = move_summary(rows, starts["jaw_open_0.6rad_0.4s"], drives["jaw"])
    out["jaw_close_0.6rad_0.4s"] = move_summary(rows, starts["jaw_close_0.6rad_0.4s"], drives["jaw"])
    out["jaw_open_hold"] = dict(hold_summary(rows, starts["jaw_open_hold"]), seconds=JAW_HOLD_S,
                                note="the jaw's range ends at 0.6 rad (its MJCF and plant range), so the open jaw rests "
                                     "on that stop, which can carry the load, and the joint's friction loss can carry up "
                                     "to 0.15 N m of it at rest")
    # the same 0.6 rad moves at other durations (PLANT_SWEEP), each run alone from the settled stance
    from designlab import sizing

    def row(s, T, drive):
        speed = MIN_JERK_PEAK_SPEED * 0.6 / T / drive.rated_speed
        pk = s["peak_force_Nm"] / drive.peak
        free = s["at_envelope_steps"] == 0
        return dict(seconds=T, peak_force_Nm=s["peak_force_Nm"], peak_raw=pk, peak_with_margin=sizing.PEAK_MARGIN * pk,
                    speed_raw=speed, speed_with_margin=sizing.SPEED_MARGIN * speed,
                    at_envelope_steps=s["at_envelope_steps"], tracking_error_max_rad=s["tracking_error_max_rad"],
                    within_raw=bool(free and pk <= 1.0 and speed <= 1.0),
                    within_margins=bool(free and sizing.PEAK_MARGIN * pk <= 1.0 and sizing.SPEED_MARGIN * speed <= 1.0))

    sweep = dict(rule="each move alone from the settled stance as above; speed is the profile's peak rate (1.875 A / T) "
                      "over the drive's rated speed; within: no step at the envelope's force limit (a clamped servo "
                      "is at its limit, not within it) and peak and speed within the limits, raw or with the sizing's "
                      "margins (1.25 x peak, 1.15 x speed)", gaze={}, jaw_open={}, jaw_close={})
    for T in PLANT_SWEEP["gaze"]:
        rows, starts = run([("neck_yaw", 0.0, -0.3, LEAD_S, ""), ("neck_yaw", -0.3, 0.0, SETTLE_S, ""),
                            ("neck_yaw", -0.3, 0.6, T, "m"), ("neck_yaw", 0.3, 0.0, HOLD_S, "")])
        sweep["gaze"][f"{T:.2f}"] = row(move_summary(rows, starts["m"], drives["neck_yaw"]), T, drives["neck_yaw"])
    for T in PLANT_SWEEP["jaw"]:
        rows, starts = run([("jaw", 0.0, 0.0, SETTLE_S, ""), ("jaw", 0.0, 0.6, T, "o"), ("jaw", 0.6, 0.0, 0.5, ""),
                            ("jaw", 0.6, -0.6, T, "c"), ("jaw", 0.0, 0.0, HOLD_S, "")])
        sweep["jaw_open"][f"{T:.2f}"] = row(move_summary(rows, starts["o"], drives["jaw"]), T, drives["jaw"])
        sweep["jaw_close"][f"{T:.2f}"] = row(move_summary(rows, starts["c"], drives["jaw"]), T, drives["jaw"])
    sweep["fastest_swept"] = {mv: {k: min((r["seconds"] for r in sweep[mv].values() if r[f"within_{k}"]), default=None)
                                   for k in ("raw", "margins")} for mv in ("gaze", "jaw_open", "jaw_close")}
    out["sweep"] = sweep
    return out


def moves_costs(x: dict) -> dict:
    """Part 3 for one -X body."""
    from designlab import sizing

    hd = HeadDynamics(x["M"], x["q_home"])
    design = x["design"]
    out = dict(rule=COSTS_RULES["moves"], pose="the -X body's standing home (refs.source, full precision)")
    m = hd.m
    out["passive_terms"] = {jn: dict(damping_Nms_per_rad=float(m.dof_damping[hd.joint(jn)[1]]),
                                     frictionloss_Nm=float(m.dof_frictionloss[hd.joint(jn)[1]]),
                                     armature_kg_m2=float(m.dof_armature[hd.joint(jn)[1]]),
                                     source="designlab/mjcf.py: <default><joint damping=0.2 frictionloss=0.15>; armature "
                                            "the catalog's (designlab/data/actuators.json)")
                            for jn in ("neck_yaw", "jaw")}
    out["inverse_dynamics"] = {name: move_inverse_dynamics(hd, design, mv) for name, mv in MOVES.items()}
    # the same gaze shift from the centre to one side: the loads of a vertical yaw axis do not depend on the start
    side = dict(MOVES["gaze_0.6rad_0.5s"], start=0.0)
    out["gaze_from_centre_0.5s_peak_Nm"] = hd.peak_total(side, SAMPLES)
    holds = {}
    for name, h in HOLDS.items():
        g = "neck_yaw" if h["joint"] == "neck_yaw" else "jaw"
        a = design.act(g)
        r = hd.at(h["joint"], h["q"], 0.0, 0.0)
        r0 = hd.at(h["joint"], 0.0, 0.0, 0.0)
        holds[name] = dict(joint=h["joint"], q=h["q"], gravity_Nm=r["rigid"], gravity_at_0_Nm=r0["rigid"],
                           over_A1=abs(r["rigid"]) / a.sustained, neck_pitch_Nm=r["pitch"],
                           neck_pitch_at_0_Nm=r0["pitch"])
    out["holds"] = holds
    sts = design.act("neck_yaw")
    lim = dict(raw=(sts.peak, sts.speed_at(None)),
               with_margins=(sts.peak / sizing.PEAK_MARGIN, sts.speed_at(None) / sizing.SPEED_MARGIN))
    fast = {}
    for name in ("gaze_0.6rad_0.5s", "jaw_open_0.6rad_0.4s", "jaw_close_0.6rad_0.4s"):
        fast[name.rsplit("_", 1)[0]] = {k: fastest(hd, MOVES[name], *v) for k, v in lim.items()}
    out["fastest_0.6rad"] = dict(limits_Nm_rad_s={k: list(v) for k, v in lim.items()}, moves=fast)
    out["plant"] = plant_moves(x["body"], hd)
    return out


# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------
def pair_costs(pair) -> dict:
    """Everything for one pair (one process)."""
    _setup()
    from designlab import gait
    from designlab.stage1 import refs

    today, xb, case_t, case_x = pair
    t0 = time.perf_counter()
    gait.set_accuracy(refs.ACCURACY)
    t, x = refs.source(today), refs.source(xb)
    st = statics_costs(t, x, case_t, case_x)
    gait.set_accuracy(refs.ACCURACY)
    walks = {}
    for name in WALKS:
        a, b = walk_screen(t, name), walk_screen(x, name)
        walks[name] = dict(today=a, x=b, moves=walk_moves(a, b))
    walks["round2_envelope"] = envelope_records(case_t, case_x, t["record"]["id"])
    mv = moves_costs(x)
    return dict(today=today, x=xb, cases=[case_t, case_x], models=dict(today=t["model"], x=x["model"]),
                H_m=x["H"], statics=st, walks=walks, moves=mv, seconds=time.perf_counter() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=2, help="processes (one body pair each; default 2)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    _setup()
    from designlab.stage1 import plant

    t0 = time.perf_counter()
    procs = max(1, min(a.procs, len(PAIRS)))
    if procs > 1:
        with get_context("spawn").Pool(procs) as pool:
            results = pool.map(pair_costs, PAIRS)
    else:
        results = [pair_costs(p) for p in PAIRS]
    doc = dict(
        title="Stage 2 S2-2: what the expressive head (-X) costs, per body, against today's head",
        plan="STAGE2_PLAN.md section 3 (S2-2); DECISION_LOG P29 (nothing adopted, P24; P2's jaw deferral stands)",
        research="stage2/research/research_look.md section 5",
        script="stage2/x/x_costs.py", script_sha256=sha256(Path(__file__)), rules=COSTS_RULES,
        bodies={r["x"]: r for r in results}, procs=procs, seconds=time.perf_counter() - t0,
        provenance=plant.provenance(), omp_num_threads=os.environ.get("OMP_NUM_THREADS"))
    Path(a.out).write_text(json.dumps(_plain(doc), indent=1) + "\n")
    for r in results:
        st, mv = r["statics"], r["moves"]
        print(f"{r['x']}: +{st['mass_kg']['added']:.3f} kg; records reproduced {st['records_reproduced']['all_match']} "
              f"(max rel diff {st['records_reproduced']['max_rel_diff']:.2e}); hip-pitch hold "
              f"{st['loads']['hip_pitch']['today']['hold_Nm']:.3f} -> {st['loads']['hip_pitch']['x']['hold_Nm']:.3f} N m; "
              f"gaze 0.5 s peak {mv['inverse_dynamics']['gaze_0.6rad_0.5s']['peak_Nm']:.3f} N m "
              f"(plant {mv['plant']['gaze_0.6rad_0.5s']['peak_force_Nm']:.3f}); jaw 0.4 s peak "
              f"{mv['inverse_dynamics']['jaw_open_0.6rad_0.4s']['peak_Nm']:.3f} N m "
              f"(plant {mv['plant']['jaw_open_0.6rad_0.4s']['peak_force_Nm']:.3f}); {r['seconds']:.0f} s", flush=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
