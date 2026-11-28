"""Form-factor study, 50-80 cm to the frill crown: a parametric design-space model scored on the pre-registered desk
criteria C1-C7 (fixed by the lead on 27 Nov 2026, before any configuration was scored). 28 Nov 2026.

WHAT KIND OF RESULT THIS IS. Every figure is one of: DESIGN (a choice recorded in the build), SIM (a stored or new
MuJoCo result), DESK (an estimate, a catalog or datasheet figure, or a calculation here). Nothing is measured: nothing
in the build is built, bought, weighed or tested (``README.md:4-5``, ``design/PARTS_60CM.md:3``). A configuration
"passes the desk criteria" when C1-C5 all pass for the worst of the four 128 mm policies' profiles; even then it is a
desk result on simulated torques, not a measurement and not a decision. Nothing here is adopted: every option other
than the chosen design (D0 drives, 60 cm) changes the design and is marked as needing the owner's decision.

A CONFIGURATION is a size (frill crown 0.50-0.80 m in 0.05 m steps), a drive set (the drive at each of the 12 leg
joints; head and tail as the build, with the budget's servo step above 67.5 cm), a leg layout and a cooling case:

- layouts: ``L1`` the knee drive on the upper link, offset from its strut (the build's arrangement; at the hind leg the
  128 mm front leg's offset is applied, which the as-drawn model lacks), driving the knee through a 1:1 pushrod
  parallelogram; ``L2`` the knee drive at the hip, coaxial with the pitch axis and outboard of it, driving the knee
  through a 1:1 parallelogram the length of the upper link (a belt would occupy the same lane);
- cooling: passive (open air), the build's model. No fact sheet gives a sourced thermal resistance for forced air
  (the RobStride ratings assume an aluminium heat sink, which is not forced air), so no forced-air case is scored.

THE TORQUE SCALING RULE (stated, justified and checked below, ``torque_factor``): at a Froude-matched walk (same
v / sqrt(g L)), a geometrically similar robot's joint torques scale as m g L, its joint speeds as sqrt(g / L) and its
mechanical power as m g v. With S the crown height, s = S / 0.60 m and m the configuration's all-in mass:
    k = (m / 12.751 kg) x s          (every leg joint's torque, the demo loop's and the peaks)
the budget's own rule (``budget/size_budget.py:55``, k = (m / 15 kg) x s75), normalised to the simulated 60 cm model.
The demo loop walks at the Froude-matched speed of the 60 cm loop's 0.25 m/s, capped at the owner's 0.25 m/s
slow-walk ceiling: v = min(0.25, 0.25 sqrt(s)) m/s (0.228 m/s at 50 cm). Above 60 cm the cap makes the robot walk at a
lower Froude number than the 60 cm loop; its 60 cm-equivalent command, 0.25 / sqrt(s) m/s, is interpolated linearly
between the stored 0.15 and 0.25 m/s segments. Checked against the 75 cm design's simulated standing torques (F4, F10)
and recorded for the Simulate phase at 0.50, 0.70 and 0.80 m (``simulate_predictions``).

REVISED 28 Nov 2026 after an independent review (findings PH-*, CTRL-*, MECH-*, F*): C4 is scored as written
(``score_packaging``: no configuration passes, so nothing passes C1-C5); the first version's narrower C4 is kept only
as a supporting reading labelled "changed after a result" (``passes_C1_C3_C5_and_C4_except_9_11``); the known-clash
list follows the packaging record's current table; the legs are swept against each other left to right and front to
hind with the model's foot body and ankle; each rod is sized for its drive (``size_linkage``) and drawn at that size;
the knee axle, the thigh's twisting moment and the bay's fit are checked beside C4; heavy drive sets carry heavier
structure (``mass_model(struct_by_load=True)``); C7 runs to the sole and is also given at the operating rating; each
configuration records whether the torque rule was checked in simulation for its drive set and layout
(``rule_check_for``, from ``sim/results/form_factor_2026-11-27/scaled_trike_compare.json``); and sensitivities give
the X4 on its own 10 and 30 rpm maker rows, the even left-right split, walking at 0.25 m/s at every size, the X6 peak
at 45 N m and a pessimistic battery bound. Revised again after a third review (R3-1): each knee axle is drawn at the
larger of the model's 12.8 mm and its own solid diameter for SF 2.0 at the static rod case, as the rods already were.

Usage (from hardware/triceratops_build):
  python design/form_factor_study.py run [--out design/form_factor_out.json] [--table PATH.md]
  python design/form_factor_study.py peaks --ckpt-root DIR   (SIM probe, writes sim/results/form_factor_2026-11-27/)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
for _p in (BUILD / "rl", BUILD / "rl" / "analysis", BUILD / "rl" / "analytics", BUILD / "walk", BUILD / "budget"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MODEL60 = BUILD / "sim" / "generated" / "trike60_C_closed_capsule_fore128.xml"
LEDGER60 = BUILD / "sim" / "generated" / "mass_ledger_trike60_C_closed_capsule_fore128.json"
LEDGER75 = BUILD / "sim" / "generated" / "mass_ledger_T15_closed.json"
DEMO_DIR = BUILD / "rl" / "analysis" / "results" / "d22" / "demo"
HEAT35_JSON = BUILD / "rl" / "analysis" / "results" / "heat35" / "thermal_ambient_2026-11-27.json"
GAIT_REPORT_DRIVES = BUILD / "rl" / "analytics" / "results" / "fore128-s0" / "gait_report.json"
RESULTS = BUILD / "sim" / "results" / "form_factor_2026-11-27"
PEAKS_JSON = RESULTS / "force_limit_probe60.json"
OUT_JSON = HERE / "form_factor_out.json"

SIZES_M = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
S60 = 0.60
M60_SIM_KG = 12.7506  # the simulated model's mass (ledger total_mass_kg; DECISION_LOG D41; demo courses ran on it)
G = 9.81
SEEDS = ("s0", "s1", "s2", "s3")
ROOM_C = 35.0          # D48 (DECISION_LOG.md (D48)): heat rules in degrees assume a 35 C room
LIMIT_C, MARGIN_C = 90.0, 80.0   # C1 as pre-registered (bench B3's 90 C rule; "with margin" 80 C)
C2_LIMIT = C3_LIMIT = 0.80
C4_GAP_M = 0.005
C5_MIN = 30.0
BUDGET_USD = 20_000.0
SCORED_TC_S = 480.0    # the build's headline time constant (rl/analysis/thermal_ambient.py SHOWN_TC); 4 and 15 min shown
WALK_S, STAND_S = 90.0, 30.0   # the demo loop (rl/demo_course.py main(): --walk-s 90 --stand-s 30)
PACK = {"pack_wh": 146.5, "usable_share": 0.80, "v_low": 42.0, "hotel_w": 30.0}  # rl/demo_course.py:47-48
ETA = 0.90             # transmission efficiency of the 1:1 flexion linkages (sim/data/drives.json:40)
BMS_KG, BMS_USD = 0.302, 540.0  # D47 battery cut-off: 0.25 kg EST x 1.2088 growth; $440 list (WS 18 Nov 2026) + $100 EST
#                                 current sensor (design/PARTS_60CM.md:25-26, :164-177; working tree)

# ---------------------------------------------------------------------------------------------------------------- drives
# Joint-side continuous ratings are the drive's thermal-equilibrium torque x ETA at the flexion joints (rl/trike_env.py
# :165-168). Kt and R are output-side, A rms, per phase (Y-equivalent); in the build's thermal model they set the
# copper loss (energy) but not the temperature in units of the rating, since each drive is fitted to its own
# equilibrium torque (rl/analysis/thermal_ambient.py:226-227; checked in the tests).
DRIVES = {
    "X4-36-L": dict(maker="MyActuator", model="RMD-X4-P36-36-E-L (V4.1)", peak=30.0, rated=10.5, eq=7.5,
                    eq_rated_speed=7.5, kt=1.8, r=0.2, kg=0.36, dia=0.055, length=0.061, usd=269.0, rpm_allow=83.0,
                    like_for_like=True,
                    src="sim/data/drives.json:4-10 (maker figures via the repo catalog, 7 Nov 2026); eq 7.5 N m = the "
                        "83 rpm 90-min equilibrium; Kt/R rl/analytics/results/fore128-s0/gait_report.json 'drives'; "
                        "price myactuator.us 18 Nov 2026 (DECISION_LOG.md (F9) F9); 83 rpm cap DECISION_LOG.md (E1) E1, "
                        "confirmed from the maker's V4.1 sheet read 28 Nov 2026 (study actuator fact sheet)"),
    "X6-60": dict(maker="MyActuator", model="RMD-X6-P20-60-E (V4)", peak=60.0, rated=20.0, eq=12.0, eq_rated_speed=9.0,
                  kt=2.1, r=0.205, kg=0.82, dia=0.080, length=0.0675, usd=428.0, rpm_allow=0.70 * 176 * 42 / 48,
                  like_for_like=True,
                  src="sim/data/drives.json:12-20; eq 12 N m = the X6-50-L's 0 rpm table (no V4 table), 9 N m at its "
                      "153 rpm rated speed (MyActuator V4.1 X6-50-L sheet read 28 Nov 2026); Kt/R gait_report.json; "
                      "price myactuator.us 18 Nov 2026 (F9)"),
    "X8-150": dict(maker="MyActuator", model="RMD-X8-P20-150-E-03 (V4.1)", peak=120.0, rated=43.0, eq=34.0,
                   eq_rated_speed=20.0, kt=2.97, r=0.090, kg=1.33, dia=0.096, length=0.071, usd=725.0,
                   rpm_allow=0.70 * 158 * 42 / 48, like_for_like=True,
                   src="sim/data/drives.json:21-29 (eq 34 / 38 / 36 / 20 N m at 0 / 10 / 30 / 127 rpm; 34 used, as "
                       "the build's file); Kt 2.97 N m/A rms and 0.180 ohm line (0.090 per phase): MyActuator V4.1 "
                       "spec book read 28 Nov 2026 (study actuator fact sheet, 'Electrical constants'); price "
                       "myactuator.us 18 Nov 2026 (F9)"),
    "RS03": dict(maker="RobStride", model="RobStride 03", peak=60.0, rated=21.0, eq=13.0, eq_rated_speed=13.0,
                 kt=2.36, r=0.293, kg=0.90, dia=0.106, length=0.056, usd=225.0, rpm_allow=0.70 * 171,
                 like_for_like=False,
                 src="RobStride maker README (github.com/RobStride/Product_Information README_main.md:20, read 28 Nov "
                     "2026): '21 N m @ 100 rpm | 60 N m', 900 g +-20 g; third-party extracts of the manual (scratch "
                     "facts_actuators.md:151, :231; RS03_huphy.txt:151, :171): the aluminium heat-sink condition, 13 N "
                     "m held still (used here as eq: NOT an open-air equilibrium, so not comparable with MyActuator's "
                     "tables), Kt 2.36 N m/A rms, 880 g +-20 g; resistance NOT FOUND: 0.293 ohm is a DESK assumption giving the X6-50-L's copper "
                     "loss at equilibrium (26.7 W); 106 x 56 mm, 0.90 kg; $225 Seeed (datasheet page updated 21 Oct "
                     "2026, hardware/triceratops_kickoff/PARTS_PREMIUM.md:217); no US stock confirmed, so 41.5 % "
                     "duty in every scenario (DECISION_LOG.md E4)"),
}
for _d in DRIVES.values():
    _d["km"] = _d["kt"] / math.sqrt(3 * _d["r"])  # N m per sqrt(W), output side
# Added 28 Nov 2026 after review. ``static``: the largest torque sim/data/drives.json gives (stall_nm if above peak_nm,
# else peak_nm), the packaging record's static rod case (design/pushrod60.py drive_torques): X4 36 (stall), X6 60
# (peak; no stall figure), X8 150 (stall), RS03 60 (peak; not in drives.json). ``row_rise_k``: the temperature rise of
# the maker row each ``eq`` comes from (scratch facts_actuators.md:107-110: X4 7.5 N m 26->96 C at 83 rpm, X6-50-L
# 12 N m 26->98 C at 0 rpm, X8 34 N m 28->99 C at 0 rpm); the build's calibration assumes 75 K (25 -> 100 C,
# rl/telemetry/thermal.py:10-13). RS03: no row (None).
for _k, _st, _rise in (("X4-36-L", 36.0, 70.0), ("X6-60", 60.0, 72.0), ("X8-150", 150.0, 71.0), ("RS03", 60.0, None)):
    DRIVES[_k]["static"] = _st
    DRIVES[_k]["row_rise_k"] = _rise
# The X4-36-L's other maker rows (90-min equilibrium, start -> end C; facts_actuators.md:107, the maker's V4.1 sheet
# read 28 Nov 2026), for the symmetric rating sensitivity (finding PH-2): the build fits every drive to its eq from
# 25 C to 100 C; these refit the X4 to the row itself.
X4_ROWS = {"0 rpm": (8.5, 27.0, 98.0), "10 rpm": (8.0, 26.0, 94.0), "30 rpm": (8.5, 27.0, 98.0),
           "83 rpm": (7.5, 26.0, 96.0)}
X6_PEAK_LOW_NM = 45.0   # the X6-P20-50-E-L's peak (facts_actuators.md:54; the store page shows 50 N m): finding PH-8

AXES = ("roll", "pitch", "fore_knee", "hind_knee")
DRIVE_SETS = {
    "D0": dict(name="current: 10 X4 + 2 X6 (fore knees)", roll="X4-36-L", pitch="X4-36-L", fore_knee="X6-60",
               hind_knee="X4-36-L", owner_decision=False),
    "D1": dict(name="X6 at both knees", roll="X4-36-L", pitch="X4-36-L", fore_knee="X6-60", hind_knee="X6-60",
               owner_decision=True),
    "D2": dict(name="X6 at knees and hip pitch", roll="X4-36-L", pitch="X6-60", fore_knee="X6-60", hind_knee="X6-60",
               owner_decision=True),
    "D3": dict(name="X8 at knees and hip pitch", roll="X4-36-L", pitch="X8-150", fore_knee="X8-150",
               hind_knee="X8-150", owner_decision=True),
    "D4": dict(name="X8 at hip pitch, X6 knees", roll="X4-36-L", pitch="X8-150", fore_knee="X6-60",
               hind_knee="X6-60", owner_decision=True),
    "D5": dict(name="X6 everywhere", roll="X6-60", pitch="X6-60", fore_knee="X6-60", hind_knee="X6-60",
               owner_decision=True),
    "D6": dict(name="X6 roll, X8 at hip pitch and knees", roll="X6-60", pitch="X8-150", fore_knee="X8-150",
               hind_knee="X8-150", owner_decision=True),
    "D7": dict(name="RobStride RS03 at hip pitch and knees, X4 roll (reference only)", roll="X4-36-L", pitch="RS03",
               fore_knee="RS03", hind_knee="RS03", owner_decision=True),
    "D8": dict(name="X8 everywhere", roll="X8-150", pitch="X8-150", fore_knee="X8-150", hind_knee="X8-150",
               owner_decision=True),
}
LAYOUTS = {"L1": "knee drive on the upper link, offset from the strut (as the build; offset applied at the hind leg)",
           "L2": "knee drive at the hip, coaxial with the pitch axis, 1:1 parallelogram the length of the upper link"}
GROUP_OF_AXIS = {"roll": "roll", "pitch": "pitch", "fore_knee": "front elbow", "hind_knee": "hind knee"}


def axis_of(joint: str) -> str:
    end = "fore" if "_fore_" in joint else "hind"
    if joint.endswith("_roll"):
        return "roll"
    if joint.endswith("_pitch"):
        return "pitch"
    return f"{end}_knee"


def drives_of(dset: dict) -> dict:
    return {a: DRIVES[dset[a]] for a in AXES}


def eta_of(axis: str) -> float:
    return ETA if axis.endswith("knee") else 1.0


# ---------------------------------------------------------------------------------------------------------------- mass
GROW = re.compile(r"x (\d\.\d+) reserve growth")
SERVO_KG = {"XM540-W270-R": 0.165, "XM430-W350-R": 0.082}   # sim/data/drives.json:32-34 (catalog)


def part_class(row: dict) -> tuple[str, float | None]:
    """(class, size exponent) of a 60 cm ledger part, from its basis text (the model's own rules,
    sim/build_model60.py:43-67, as the study's internal fact sheet reads them)."""
    b, g, cat = row["basis"], row["geom"], row["category"]
    if cat == "drives":
        return "drive", None
    if cat == "battery":
        return "battery", 0.0
    if cat == "reserve":
        return "reserve block", 0.0
    if cat == "leg transmissions":
        return "linkage", None                      # sized by the knee drive's peak torque (see mass_model)
    if "offset_motor_mount" in g or "mount_standoff" in g:
        return "knee drive mount", None             # carries the knee drive: scaled by its mass
    if "drop_bracket" in g:
        return "roll drive bracket", None           # carries the roll drive: scaled by its mass
    if "tof_mount" in g:
        return "fixed: sensor mount", 0.0
    if "xc430" in g or "xl430" in g:
        return "servo", None                        # by the budget's size class
    if "xl330" in g:
        return "fixed: eyelid servo", 0.0
    if "neck_yaw_bearing" in g:
        return "scales: small head part", 2.0
    if "frill_lights" in g:
        return "fixed: lights", 0.0
    if cat == "shells" or "x 0.8^2" in b and "x 0.8^2.5" not in b:
        return "scales: skin and harness", 2.0
    if "x 0.8^2.5" in b or "length ratio" in b:
        return "scales: structure", 2.5
    if "x 0.8^3" in b:
        return "scales: head and tail", 3.0
    return "fixed: bought electronics, power, safety", 0.0


def servo_set(S: float) -> dict:
    """The neck, beak and tail servos by the budget's size class (budget/size_budget.py:124-135: XM above s75 0.9,
    i.e. a crown above 67.5 cm). Values: replacement mass per ledger geom, or None to keep the ledger's."""
    if S / 0.75 <= 0.9:
        return {}
    return {"xc430_w240_neck_pitch_case": SERVO_KG["XM540-W270-R"], "xc430_w240_neck_yaw_case": SERVO_KG["XM430-W350-R"],
            "xc430_w240_beak_case": SERVO_KG["XM430-W350-R"], "xl430_w250_tail_yaw_case": SERVO_KG["XM430-W350-R"],
            "xl430_w250_tail_pitch_case": SERVO_KG["XM430-W350-R"]}


def load_ledger60() -> dict:
    return json.loads(LEDGER60.read_text())


STRUCT_LOAD_EXP = 2.0 / 3.0   # structure mass ~ (load carried)^(2/3): solid or similar sections, sigma = M / Z, M ~ m g L,
#                              Z ~ d^3, mass ~ d^2 L at a fixed length (DESK; thin walls of fixed diameter give 1.0)


def mass_model(S: float, dset: dict, layout: str | None = "L1", bms: bool = True, ledger: dict | None = None,
               struct_by_load: bool = False, load_exp: float = STRUCT_LOAD_EXP) -> dict:
    """``struct_by_load`` (added 28 Nov 2026 after review, MECH-8): the structure classes also carry the mass they
    hold up: x (this configuration's total / the same size's D0 total)^load_exp, iterated to a fixed point, so a heavy
    drive set's struts are as stressed as D0's at that size (D0 is unchanged). The scored configurations use it; the
    scaled simulation models do not (the model's own rule, s^2.5 only)."""
    if struct_by_load:
        ref = mass_model(S, DRIVE_SETS["D0"], layout, bms, ledger)["total_kg"]
        f = 1.0
        for _ in range(50):
            mm = _mass_model(S, dset, layout, bms, ledger, struct_factor=f)
            f_new = (mm["total_kg"] / ref) ** load_exp
            if abs(f_new - f) < 1e-10:
                break
            f = f_new
        mm = _mass_model(S, dset, layout, bms, ledger, struct_factor=f)
        mm["struct_load_factor"] = f
        return mm
    return _mass_model(S, dset, layout, bms, ledger)


def _mass_model(S: float, dset: dict, layout: str | None = "L1", bms: bool = True, ledger: dict | None = None,
                struct_factor: float = 1.0) -> dict:
    """All-in mass at crown height S. ``layout`` None: the model as built (no layout additions); with ``bms`` False and
    D0 at 0.60 m that is the ledger itself. Rules (each part's 60 cm closed mass x):
    - drives: the drive set's catalog masses; battery, electronics, power and safety, sensors, reserve block: fixed;
    - structure (struts, soles, rails, yokes, tray, stays, hubs, crossmembers, cradle, axles): s^2.5 (the model's rule,
      sim/build_model60.py:57; geometric sections carrying a mass that does not scale with s^3);
    - shells, limb covers, harness: s^2 (area); head and tail parts: s^3 (volume); neck bearing s^2;
    - linkage (cranks, rods): x (knee drive peak / the 60 cm knee drive's peak), size-independent: crank radius and
      rod length both go as s, so the rod's section goes as peak / s and its mass as peak (DESK);
    - PR #697 mounts: x (mass carried / 0.82 kg X6 or 0.36 kg X4); servos by the budget's class (``servo_set``);
    - layout additions: L1 adds the 128 mm front leg's offset mount at each hind knee (0.045 kg x hind knee drive mass /
      0.82 kg); L2 lengthens each rod to the upper link's length and each knee axle by the stack depth;
    - D47 battery cut-off (+0.302 kg) when ``bms``."""
    led = ledger or load_ledger60()
    s = S / S60
    dr = drives_of(dset)
    d60 = drives_of(DRIVE_SETS["D0"])
    servos = servo_set(S)
    by: dict[str, float] = {}
    for row in led["parts"]:
        g, kg = row["geom"], row["mass_kg"]
        cls, p = part_class(row)
        end = "fore" if "_fore_" in g else ("hind" if "_hind_" in g else None)
        if cls == "drive":
            ax = "roll" if "roll" in g else ("pitch" if "pitch" in g else f"{end}_knee")
            kg = dr[ax]["kg"]
            cls = f"drive: {ax}"
        elif cls == "linkage":
            ax = f"{end}_knee"
            kg = kg * dr[ax]["peak"] / d60[ax]["peak"]
            if layout == "L2" and "pushrod" in g:
                kg *= L2_ROD_OVER_L1[end]
        elif cls == "knee drive mount":
            kg = kg * dr["fore_knee"]["kg"] / d60["fore_knee"]["kg"]
        elif cls == "roll drive bracket":
            kg = kg * dr["roll"]["kg"] / d60["roll"]["kg"]
        elif cls == "servo":
            kg = servos.get(g, kg)
        elif p:
            kg = kg * s ** p
            if cls == "scales: structure":
                kg *= struct_factor
            if layout == "L2" and "elbow_axle" in g:
                kg *= 2.0  # the knee axle reaches the outboard crank lane (DESK)
        by[cls] = by.get(cls, 0.0) + kg
    if layout == "L1":
        by["layout: hind knee offset mounts"] = 2 * 0.045 * dr["hind_knee"]["kg"] / 0.82
    if bms:
        by["battery cut-off (D47, EST)"] = BMS_KG
    total = sum(by.values())
    return {"total_kg": total, "by_class_kg": by, "s": s, "struct_load_factor": struct_factor}


L2_ROD_OVER_L1 = {"fore": 0.128 / 0.0815, "hind": 0.188 / 0.098}  # rod pin-to-pin, L2 (upper link) over the build's


def mass_split_60(ledger: dict | None = None) -> dict:
    """The 60 cm ledger by kind: fixed, scaled (with its exponent), torque- or drive-sized (checked in the tests)."""
    led = ledger or load_ledger60()
    out: dict[str, float] = {}
    for row in led["parts"]:
        cls, p = part_class(row)
        key = cls if p is None else (f"{cls} (s^{p:g})" if p else cls)
        out[key] = out.get(key, 0.0) + row["mass_kg"]
    return out


# ---------------------------------------------------------------------------------------------------------------- torque
def torque_factor(m_kg: float, S: float, m_ref: float = M60_SIM_KG, s_ref: float = S60) -> float:
    """k = (m / m_ref) x (S / S_ref): joint torque at a Froude-matched gait for a geometrically similar robot (m g L)."""
    return (m_kg / m_ref) * (S / s_ref)


def walk_speed(S: float, cap: float = 0.25) -> tuple[float, float]:
    """(walking speed at size S, the 60 cm-equivalent command): Froude-matched to the 60 cm loop's 0.25 m/s, capped.
    A negative ``cap`` means a fixed speed of -cap at every size (the sensitivity 'walk at 0.25 m/s everywhere')."""
    s = S / S60
    v = -cap if cap < 0 else min(cap, 0.25 * math.sqrt(s))
    return v, v / math.sqrt(s)


def load_demos() -> dict:
    out = {}
    for sd in SEEDS:
        f = DEMO_DIR / f"fore128-{sd}" / "demo_course.json"
        d = json.loads(f.read_text())
        d["_path"] = str(f.relative_to(BUILD))
        d["_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        out[sd] = d
    return out


EXTRAPOLATE = {"on": False}   # sensitivity only: extend the 0.15-0.25 m/s line above 0.25 (finding PH-6)


def seg_at(demo: dict, key: str, v60: float) -> np.ndarray:
    """A per-joint stored quantity of the walking segment at the 60 cm-equivalent command ``v60`` (linear between
    the stored 0.15 and 0.25 m/s commands; exactly the 0.25 segment at 0.25; above 0.25 only when EXTRAPOLATE is on,
    linearly, a DESK extrapolation)."""
    a = np.asarray(demo["segments"]["walk_0.15"][key], dtype=float)
    b = np.asarray(demo["segments"]["walk_0.25"][key], dtype=float)
    if v60 >= 0.25 - 1e-12 and not (EXTRAPOLATE["on"] and v60 > 0.25 + 1e-12):
        return b
    w = (v60 - 0.15) / 0.10
    return a + w * (b - a)


def seg_scalar(demo: dict, key: str, v60: float) -> float:
    a = float(demo["segments"]["walk_0.15"][key])
    b = float(demo["segments"]["walk_0.25"][key])
    if v60 >= 0.25 - 1e-12 and not (EXTRAPOLATE["on"] and v60 > 0.25 + 1e-12):
        return b
    return a + (v60 - 0.15) / 0.10 * (b - a)


def config_profile(demo: dict, S: float, k: float, dset: dict, cap: float = 0.25) -> dict:
    """The configuration's per-joint mean-squared drive torque (N^2 m^2) on its walking and standing segments, and the
    load ratio (RMS over the new drive's equilibrium torque) on each and over the loop."""
    joints = demo["segments"]["stand"]["joints"]
    v, v60 = walk_speed(S, cap)
    walk = seg_at(demo, "tau_drive_sq_mean", v60) * k * k
    stand = np.asarray(demo["segments"]["stand"]["tau_drive_sq_mean"], dtype=float) * k * k
    dr = drives_of(dset)
    eq = np.array([dr[axis_of(j)]["eq"] for j in joints])
    loop = (WALK_S * walk + STAND_S * stand) / (WALK_S + STAND_S)
    return {"joints": joints, "v_mps": v, "v60_equiv_mps": v60, "walk_sq": walk, "stand_sq": stand,
            "eq_nm": eq, "loop_ratio": np.sqrt(loop) / eq, "walk_ratio": np.sqrt(walk) / eq,
            "stand_ratio": np.sqrt(stand) / eq}


# ---------------------------------------------------------------------------------------------------------------- energy
def energy(demo: dict, S: float, k: float, m_kg: float, dset: dict, cap: float = 0.25) -> dict:
    """C5: copper loss scaled as (k x Km_60 / Km_new)^2 per drive; mechanical power (stored bus draw less copper) as
    (m / m60) x sqrt(s), i.e. m g v at the Froude-matched gait; 30 W hotel load; 146.5 Wh x 80 % (rl/demo_course.py)."""
    s = S / S60
    _, v60 = walk_speed(S, cap)
    joints = demo["segments"]["stand"]["joints"]
    dr, d0 = drives_of(dset), drives_of(DRIVE_SETS["D0"])
    kmr = np.array([(d0[axis_of(j)]["km"] / dr[axis_of(j)]["km"]) ** 2 for j in joints])
    out = {}
    for name in ("walk", "stand"):
        if name == "walk":
            cu60 = seg_at(demo, "copper_w", v60)
            draw60 = seg_scalar(demo, "mean_draw_w", v60)
            peak60 = seg_scalar(demo, "peak_draw_w", v60)
        else:
            g = demo["segments"]["stand"]
            cu60, draw60, peak60 = np.asarray(g["copper_w"], dtype=float), float(g["mean_draw_w"]), g["peak_draw_w"]
        mech60 = draw60 - float(cu60.sum())
        cu = float((cu60 * kmr).sum()) * k * k
        mech = mech60 * (m_kg / M60_SIM_KG) * math.sqrt(s)
        out[name] = {"copper_w": cu, "mech_w": mech, "draw_w": cu + mech,
                     "peak_draw_w_est": peak60 * k * k * float(np.mean(kmr))}
    mean = (WALK_S * out["walk"]["draw_w"] + STAND_S * out["stand"]["draw_w"]) / (WALK_S + STAND_S)
    total = mean + PACK["hotel_w"]
    usable = PACK["pack_wh"] * PACK["usable_share"]
    peak_a = (max(out["walk"]["peak_draw_w_est"], out["stand"]["peak_draw_w_est"]) + PACK["hotel_w"]) / PACK["v_low"]
    # finding PH-7: a pessimistic bound (DESK), the scored figure being an upper bound: copper at the windings'
    # running heat (resistance x 1.29, 100 C against the model's 25 C: 1 + 0.00393 x 75), no braking recovery and
    # gearbox loss as 13 W more mechanical draw at 60 cm (an allowance: a mechanical cost of transport of about 0.43
    # at 0.24 m/s, scaled as m g v), and the hotel load at 40 W (the top of rl/demo_course.py's 25-40 W range)
    mech_extra = 13.0 * (m_kg / M60_SIM_KG) * math.sqrt(s) * (WALK_S / (WALK_S + STAND_S))
    cu_mean = (WALK_S * out["walk"]["copper_w"] + STAND_S * out["stand"]["copper_w"]) / (WALK_S + STAND_S)
    pess = mean + 0.29 * cu_mean + mech_extra + 40.0
    return {"segments": out, "mean_drive_w": mean, "mean_total_w": total, "runtime_min": usable / total * 60.0,
            "runtime_min_hotel_25_40": [usable / (mean + 25) * 60.0, usable / (mean + 40) * 60.0],
            "runtime_min_pessimistic": usable / pess * 60.0, "peak_pack_a_est": peak_a}


# ---------------------------------------------------------------------------------------------------------------- heat
_TA = None


def _ta():
    global _TA
    if _TA is None:
        import thermal_ambient  # rl/analysis/thermal_ambient.py, imported read-only
        _TA = thermal_ambient
    return _TA


_FITS: dict = {}


@contextlib.contextmanager
def _persistent_fits():
    """As rl/analysis/thermal_ambient._cached_fits, but the fits are kept for the whole study: each drive's R_th and
    C are fitted once per (drive figures, time constant); the joint's name does not enter a fit. (The tests check that
    the stored 25 C and 35 C figures are reproduced exactly through this path.)"""
    from dataclasses import replace

    from telemetry.thermal import DriveThermal, ThermalNode

    orig = DriveThermal.calibrate

    def calibrate(self, time_constant_s, room_c=None):
        key = (replace(self, name=""), float(time_constant_s))
        if key not in _FITS:
            _FITS[key] = orig(self, time_constant_s)
        n = _FITS[key]
        return ThermalNode(self, n.r_th_k_per_w, n.c_j_per_k, n.step_s, room_c=room_c)

    DriveThermal.calibrate = calibrate
    try:
        yield
    finally:
        DriveThermal.calibrate = orig


def thermal_inputs(demo: dict, walk_sq, stand_sq, dset: dict, minutes: float, loop=None):
    """The build's DriveThermal per joint (the new drive's Kt, R, equilibrium torque) and its loss history over the
    demo: the loop repeated for ``minutes`` (rl/demo_course.py build_sequence)."""
    import demo_course
    from telemetry.thermal import DriveThermal

    joints = demo["segments"]["stand"]["joints"]
    dr = drives_of(dset)
    loop = loop or [("walk", WALK_S), ("stand", STAND_S)]
    seq = demo_course.build_sequence(loop, minutes)
    therm, per = {}, {}
    for j, jn in enumerate(joints):
        d = dr[axis_of(jn)]
        therm[jn] = DriveThermal(jn, d["kt"], d["r"], float(d["eq"]), ambient_c=d.get("fit_start_c", 25.0),
                                 ref_temp_c=d.get("fit_end_c", 100.0))
        sq = {"walk": float(walk_sq[j]), "stand": float(stand_sq[j])}
        per[jn] = [(sec, sq[name]) for name, sec in seq]
    return therm, per


def heat(demo, walk_sq, stand_sq, dset, minutes, room_c=ROOM_C, method="A") -> dict:
    """C1 by the build's model (rl/telemetry/thermal.py through rl/analysis/thermal_ambient.temperatures): every
    drive's peak winding temperature over the demo, open air, method A (fitted at the 25 C rating room, run in the
    room), for the 4, 8 and 15 min time constants."""
    from telemetry.thermal import TIME_CONSTANTS_S, tc_key

    ta = _ta()
    therm, per = thermal_inputs(demo, walk_sq, stand_sq, dset, minutes)
    with _persistent_fits():
        t = ta.temperatures(therm, per, room_c, method, TIME_CONSTANTS_S)
    out = {}
    for tc in TIME_CONSTANTS_S:
        k = tc_key(tc)
        jn = max(t, key=lambda j: t[j][k]["peak_c"])
        cont = [v[k]["continuous_c"] for v in t.values()]
        out[k] = {"joint": jn, "peak_c": t[jn][k]["peak_c"],
                  "settled_cycle_max_c": None if any(c is None for c in cont) else max(cont)}
    return {"by_tc": out, "per_joint_peak_c_8min": {j: t[j][tc_key(SCORED_TC_S)]["peak_c"] for j in t}}


def mirror_of(joint: str) -> str:
    return joint.replace("left_", "RIGHT_").replace("right_", "left_").replace("RIGHT_", "right_")


def even_split(prof: dict) -> dict:
    """DESK arithmetic, not a pass (findings CTRL-1, PH-4): each left / right pair of drives given the pair's mean
    squared torque (the same total copper loss, split evenly), on the walking and standing segments."""
    js = prof["joints"]
    idx = {j: i for i, j in enumerate(js)}
    walk, stand = np.array(prof["walk_sq"], dtype=float), np.array(prof["stand_sq"], dtype=float)
    for arr in (walk, stand):
        src = arr.copy()
        for j, i in idx.items():
            arr[i] = 0.5 * (src[i] + src[idx[mirror_of(j)]])
    loop = (WALK_S * walk + STAND_S * stand) / (WALK_S + STAND_S)
    return {"walk_sq": walk, "stand_sq": stand, "loop_ratio": np.sqrt(loop) / prof["eq_nm"]}


def deciding_and_mirror(prof: dict) -> dict:
    """The policy's deciding drive (largest loop RMS / continuous) beside its mirror image on the same policy."""
    js = prof["joints"]
    r = np.asarray(prof["loop_ratio"])
    i = int(np.argmax(r))
    j = js[i]
    im = js.index(mirror_of(j))
    st = np.sqrt(np.asarray(prof["stand_sq"]))
    return {"joint": j, "ratio": float(r[i]), "mirror": js[im], "mirror_ratio": float(r[im]),
            "stand_rms_nm": float(st[i]), "mirror_stand_rms_nm": float(st[im])}


# ---------------------------------------------------------------------------------------------------------------- peaks
def peak_need_factor(probe: dict | None) -> dict:
    """From the SIM force-limit probe: per walker, the lowest force-limit factor tested at and above which no episode
    fell (walking and 0.5 m/s pushes both ways and both axes); the worst over the four policies."""
    if probe is None:
        return {"available": False}
    per = {}
    for w, rec in probe["walkers"].items():
        ok = []
        for f in sorted(probe["force_factors"], reverse=True):
            cases = [c for c in rec["cases"].values() if c["factor"] == f]
            if any(c["falls"] for c in cases):
                break
            ok.append(f)
        per[w] = min(ok) if ok else None
    pol = [v for w, v in per.items() if w.startswith("policy")]
    worst = None if any(v is None for v in pol) else max(pol)
    return {"available": True, "per_walker": per, "worst_policy_factor": worst,
            "factors_tested": probe["force_factors"], "push_dv_mps": probe["push"]["dv_mps"]}


def peaks(k: float, dset: dict, need: dict) -> dict:
    """C3. The simulated peaks at 60 cm sit at the model's clamp, 0.70 x the maker peak at every leg joint (drive side:
    joint clamp / ETA at the flexion joints), in the trot, the pushes and the learned policies alike; so the clamp IS
    the stored simulated peak, scaled here by k against the new drive's maker peak. Supporting (not the criterion):
    the same with the lowest force-limit factor at which the 60 cm walkers still walked and took 0.5 m/s pushes."""
    dr, d0 = drives_of(dset), drives_of(DRIVE_SETS["D0"])
    clamp = {a: k * 0.70 * d0[a]["peak"] / dr[a]["peak"] for a in AXES}
    out = {"ratio_by_axis": clamp, "worst": max(clamp.values()),
           "worst_axis": max(clamp, key=clamp.get)}
    # finding PH-8: for D0 this is 0.70 k at every axis, so C3 is k <= 1.143 and separates nothing that k does not;
    # and the X6's peak is uncertain (60 N m X6-60, 45-50 N m X6-50-L): every X6 axis at 45 N m instead
    x6lo = {a: (k * 0.70 * d0[a]["peak"] / X6_PEAK_LOW_NM if dset[a] == "X6-60" else v) for a, v in clamp.items()}
    out["x6_peak_45nm_worst"] = max(x6lo.values())
    out["x6_peak_45nm_axis"] = max(x6lo, key=x6lo.get)
    f = need.get("worst_policy_factor") if need.get("available") else None
    if f:
        sup = {a: v * f for a, v in clamp.items()}
        out["supporting_need_ratio_by_axis"] = sup
        out["supporting_need_worst"] = max(sup.values())
        out["supporting_need_factor"] = f
    return out


# ---------------------------------------------------------------------------------------------------------------- C7
# Published quadrupeds with sim-to-real learned walking (study existence fact sheet, Table 1, read 28 Nov 2026; WS =
# web-search summary of the named page, MF = model or code file opened). (mass kg, thigh m, shank m, hip peak, knee
# peak, hip continuous, knee continuous). C7 as pre-registered: tau / (m g l), l = hip to foot (thigh + shank) for the
# hip pitch, knee to foot (shank) for the knee.
PUBLISHED = {
    "Mini Cheetah": (9.0, 0.209, 0.195, 17.0, 26.0, 6.9, 10.7, "mass WS (Margolis et al. RSS 2022); levers and peaks MF "
                     "(Cheetah-Software); 6.9 N m continuous per drive WS secondary source; knee x 9.33/6 desk"),
    "ANYmal B": (30.0, 0.25, 0.315, 40.0, 40.0, 15.0, 15.0, "mass WS; levers MF (legged_gym URDF knee joint at xyz "
                 "'0.0 0.109 -0.25': 0.25 m in the leg's plane, the 0.109 m sideways offset gives no lever about the "
                 "hip-pitch axis; 0.273 was used before the 28 Nov 2026 review); 40 / 15 N m peak / 'nominal' WS "
                 "(ANYdrive flyer)"),
    "Unitree A1": (12.0, 0.20, 0.20, 33.5, 33.5, None, None, "WS mass; MF unitree_ros const.xacro"),
    "Unitree Go1": (12.0, 0.213, 0.213, 23.7, 35.55, None, None, "WS; MF"),
    "Barkour vB": (14.5, 0.19, 0.19, 18.0, 18.0, None, None, "MF Menagerie model (mass of the model)"),
    "Unitree Go2": (15.0, 0.213, 0.213, 23.7, 45.43, None, None, "WS; MF"),
    "Unitree Aliengo": (23.24, 0.25, 0.25, 35.28, 44.4, None, None, "MF URDF"),
    "Spot": (32.7, 0.321, 0.337, 45.0, 113.2, None, None, "WS mass; MF Isaac Lab (knee limit by angle, maximum)"),
    "ANYmal C": (50.0, 0.302, 0.328, 80.0, 80.0, None, None, "WS mass; MF Isaac Lab effort limit"),
    "Unitree B1": (50.0, 0.35, 0.35, 93.3, 140.0, None, None, "WS; MF"),
    "Unitree B2": (60.0, 0.32, 0.35, 200.0, 320.0, None, None, "WS; MF"),
    "Solo-12": (2.5, 0.16, 0.16, 2.7, 2.7, None, None, "WS (current-limited maximum)"),
}


def published_c7() -> dict:
    out = {}
    for n, (m, th, sh, hp, kp, hc, kc, src) in PUBLISHED.items():
        out[n] = {"hip_peak": hp / (m * G * (th + sh)), "knee_peak": kp / (m * G * sh),
                  "hip_cont": None if hc is None else hc / (m * G * (th + sh)),
                  "knee_cont": None if kc is None else kc / (m * G * sh), "source": src}
    return out


LEG = {  # 60 cm model lengths (sim/generated/trike60_C_closed_capsule_fore128.xml:186, :195, :250, :259)
    "fore": dict(l_u=0.128, l_l=0.184, u=(-0.0946259392, -0.0861970511), w=(0.0946259392, -0.157802949),
                 hip=(0.26, -0.04), flex_axis=(0.0, -1.0, 0.0)),
    "hind": dict(l_u=0.188, l_l=0.184, u=(0.120121, -0.14462), w=(-0.120121, -0.13938), hip=(-0.26, 0.0),
                 flex_axis=(0.0, 1.0, 0.0)),
}


SOLE_BELOW_ANKLE_M = 0.012   # the sole capsule's axis below the ankle (fore128.xml:265, 'fromto ... -0.01200')


def knee_lever_m(end: str, s: float) -> float:
    """Knee to the sole's centre at the neutral pose (the published calf lengths run to the foot's centre): the lower
    strut plus the foot body's 12 mm (x s) below the ankle, straight line (finding CTRL-8; 0.193 m hind, 0.194 m fore
    at 60 cm, against the strut's 0.184 m alone before the review)."""
    w = np.asarray(LEG[end]["w"])
    return float(np.linalg.norm(w + np.array([0.0, -SOLE_BELOW_ANKLE_M]))) * s


def operating_eq(d: dict) -> float | None:
    """A drive's continuous rating in a 35 C room with a 90 C limit: eq x sqrt(55 / the row's own rise), copper loss
    only (scratch facts_actuators.md:103, 'DESK 35 C room / 90 C'); None without a row."""
    return None if d.get("row_rise_k") is None else d["eq"] * math.sqrt((LIMIT_C - ROOM_C) / d["row_rise_k"])


def c7(S: float, m_kg: float, dset: dict) -> dict:
    """C7 as pre-registered, with the levers to the sole's centre (hip pitch: upper link + ``knee_lever_m``; knee:
    ``knee_lever_m``), at the drive's rating (``eq``) and at its operating rating (``operating_eq``)."""
    s = S / S60
    dr = drives_of(dset)
    out = {}
    for end in ("fore", "hind"):
        lu, ll = LEG[end]["l_u"] * s, knee_lever_m(end, s)
        kn = dr[f"{end}_knee"]
        out[f"{end}_hip_pitch_cont"] = dr["pitch"]["eq"] / (m_kg * G * (lu + ll))
        out[f"{end}_knee_cont"] = kn["eq"] * ETA / (m_kg * G * ll)
        op_p, op_k = operating_eq(dr["pitch"]), operating_eq(kn)
        out[f"{end}_hip_pitch_cont_operating"] = None if op_p is None else op_p / (m_kg * G * (lu + ll))
        out[f"{end}_knee_cont_operating"] = None if op_k is None else op_k * ETA / (m_kg * G * ll)
        out[f"{end}_hip_pitch_peak_clamp"] = 0.70 * dr["pitch"]["peak"] / (m_kg * G * (lu + ll))
        out[f"{end}_knee_peak_clamp"] = 0.70 * kn["peak"] * ETA / (m_kg * G * ll)
    pub = published_c7()
    hip_c = {n: v["hip_cont"] for n, v in pub.items() if v["hip_cont"] is not None}
    knee_c = {n: v["knee_cont"] for n, v in pub.items() if v["knee_cont"] is not None}
    worst_hip = min(out["fore_hip_pitch_cont"], out["hind_hip_pitch_cont"])
    worst_knee = min(out["fore_knee_cont"], out["hind_knee_cont"])
    ops = [out[k] for k in ("fore_hip_pitch_cont_operating", "hind_hip_pitch_cont_operating")]
    opk = [out[k] for k in ("fore_knee_cont_operating", "hind_knee_cont_operating")]
    worst_hip_op = None if None in ops else min(ops)
    worst_knee_op = None if None in opk else min(opk)

    def where(x, ref):
        # two reference points with different rating definitions (ANYmal B 'nominal', the Mini Cheetah's secondary
        # continuous figure), not a range (finding CTRL-8)
        if x is None:
            return None
        lo, hi = min(ref.values()), max(ref.values())
        return "below both reference points" if x < lo else ("above both" if x > hi else "between the two")

    out["position"] = {"hip_cont_worst": worst_hip, "hip_published": hip_c, "hip_where": where(worst_hip, hip_c),
                       "knee_cont_worst": worst_knee, "knee_published": knee_c,
                       "knee_where": where(worst_knee, knee_c),
                       "hip_cont_worst_operating": worst_hip_op, "hip_where_operating": where(worst_hip_op, hip_c),
                       "knee_cont_worst_operating": worst_knee_op, "knee_where_operating": where(worst_knee_op, knee_c),
                       "hind_knee_peak_clamp": out["hind_knee_peak_clamp"],
                       "peak_hip_published_range": [min(v["hip_peak"] for v in pub.values()),
                                                    max(v["hip_peak"] for v in pub.values())],
                       "peak_knee_published_range": [min(v["knee_peak"] for v in pub.values()),
                                                     max(v["knee_peak"] for v in pub.values())]}
    return out


# ---------------------------------------------------------------------------------------------------------------- cost
def cost(S: float, dset: dict, layout: str, frame_fixed: float = 0.0) -> dict:
    """C6 by budget/size_budget.py's own lines and landed rule (bom, expression, overheads; imported read-only): its
    non-drive lines at s75 = S / 0.75 (frame cost by area unless ``frame_fixed``), full expression (servo step above
    67.5 cm), the fail-safe radio stop ($250 EST, the chosen build, size_budget.py:237), the drive set at list price
    with one spare of each drive type (as size_budget.py:94-96), landed = (hardware + 5 % freight + duty x drives +
    brokerage + 7 % tax) x 1.07 (size_budget.py:160-164) for the three duty scenarios (size_budget.py:170-171).
    RobStride lines pay 41.5 % in every scenario (no US stock confirmed). The D47 battery cut-off is added after."""
    import size_budget as sb

    s75 = S / 0.75
    lines = sb.bom(("x", s75, "12S", [(12, "X4-36-L", "roll", 1.0)]), frame_fixed)
    keep = [ln for ln in lines if not ln[3]]  # the duty-bearing lines are the drives and the spare drives
    n_by = {}
    for a in AXES:
        n = 4 if a in ("roll", "pitch") else 2
        n_by[dset[a]] = n_by.get(dset[a], 0) + n
    drive_lines = [(d, n * DRIVES[d]["usd"] + DRIVES[d]["usd"], DRIVES[d]["maker"]) for d, n in n_by.items()]
    other = sum(ln[2] for ln in keep) + sb.expression("full", s75)[1] + 250.0
    drives_usd = sum(x for _, x, _ in drive_lines)
    hw = other + drives_usd
    landed = {}
    for label, rate, brok in sb.DUTY_SCENARIOS:
        duty = sum(x * (0.415 if mk != "MyActuator" else rate) for _, x, mk in drive_lines)
        b = brok if (rate > 0 or any(mk != "MyActuator" for _, _, mk in drive_lines)) else 0.0
        landed[label] = (hw + 0.05 * hw + duty + b + 0.07 * hw) * 1.07
    bms = BMS_USD * (1 + 0.05 + 0.07) * 1.07
    keys = list(landed)
    return {"hardware_usd": hw, "drives_with_spares_usd": drives_usd, "drive_counts": n_by, "landed_usd": landed,
            "landed_0pct_usd": landed[keys[0]], "landed_41pct_usd": landed[keys[2]],
            "landed_0pct_with_bms_usd": landed[keys[0]] + bms, "landed_41pct_with_bms_usd": landed[keys[2]] + bms,
            "under_20k_at_0pct": landed[keys[0]] < BUDGET_USD,
            "under_20k_at_0pct_with_bms": landed[keys[0]] + bms < BUDGET_USD}


# ---------------------------------------------------------------------------------------------------------------- C4
@dataclass
class Part:
    name: str
    body: str
    kind: str          # capsule | cylinder | box
    a: tuple           # fromto start (capsule/cylinder) or centre (box)
    b: tuple           # fromto end, or half sizes (box)
    r: float = 0.0


JOINED = {frozenset(p) for p in (
    ("yoke", "roll_drive"), ("yoke", "pitch_drive"), ("upper_strut", "pitch_drive"), ("upper_strut", "yoke"),
    ("upper_strut", "lower_strut"), ("axle", "upper_strut"), ("axle", "lower_strut"), ("axle", "output_crank"),
    ("drive_crank", "knee_drive"), ("drive_crank", "rod"), ("output_crank", "rod"), ("ankle_hub", "lower_strut"),
    ("ankle_hub", "sole"), ("lower_strut", "sole"), ("roll_drive", "bracket"), ("bracket", "rail"), ("bracket", "cm"),
    ("rail", "cm"))}
JOINED_L2 = {frozenset(("upper_strut", "knee_drive"))}   # the coaxial knee drive bolts to the thigh's root plate


def _base(name: str) -> str:
    for pre in ("fore_", "hind_"):
        if name.startswith(pre):
            name = name[len(pre):]
    if name.startswith("cm"):
        return "cm"
    if name.startswith("rail"):
        return "rail"
    if name.startswith("bracket"):
        return "bracket"
    return name


def _end(name: str):
    return "fore" if name.startswith("fore_") or name.endswith("_fore") else (
        "hind" if name.startswith("hind_") or name.endswith("_hind") else None)


def crank_radius(s: float) -> float:
    """The pushrod cranks: the 60 cm model's 28 mm (design/PACKAGING_PUSHROD_60CM.md:86), scaled up with the robot but
    never below 28 mm, so that the rod force at the drive's limit (1 / crank radius) does not rise in a smaller robot
    and the rod clears the knee axle by the 60 cm margin."""
    return 0.028 * max(s, 1.0)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _rot_y(phi: float, v2):
    """Rotate an (x, z) vector by phi about +y (x' = x cos + z sin, z' = -x sin + z cos)."""
    c, s_ = math.cos(phi), math.sin(phi)
    return np.array([v2[0] * c + v2[1] * s_, -v2[0] * s_ + v2[1] * c])


def _seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(x, dtype=float) for x in (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def knee_drive_candidates(end: str, s: float, rk: float, rp: float, r_us: float, r_ls: float,
                          disc_r: float) -> list[dict]:
    """L1: candidate positions of the knee drive's axis on the upper link (x-z plane of the link, origin at the pitch
    axis): along the strut at fraction t = 0.25-0.85, offset sideways (either side) by rk + r_us + 9 mm (the 128 mm
    front leg's 8.7 mm, rounded up). Each carries its least analytic clearance in the link's plane (to the pitch drive,
    to the lower strut over the knee range, between the two crank discs). Returned nearest the hip first among those
    clearing by 5 mm or more, then the rest by clearance; ``make_layout`` takes the first that the full geometry
    (chassis, rod, crossmembers over roll x pitch x flexion) also clears."""
    L = LEG[end]
    u, w = _unit(L["u"]), _unit(L["w"])
    K = u * L["l_u"] * s
    d_off = rk + r_us + 0.009
    sign_flex = -1.0 if L["flex_axis"][1] < 0 else 1.0
    cands = []
    for side in (+1.0, -1.0):
        n = side * np.array([u[1], -u[0]])
        for t in np.linspace(0.25, 0.85, 13):
            P = t * K + n * d_off
            c_pitch = float(np.linalg.norm(P)) - rp - rk
            c_ls = min(_seg_dist(P, K, K + _rot_y(sign_flex * math.radians(th), w) * L["l_l"] * s)
                       for th in np.arange(-47.0, 36.01, 1.0)) - rk - r_ls
            c_disc = float(np.linalg.norm(K - P)) - 2 * disc_r
            cands.append({"t": float(t), "side": side, "P": P, "K": K, "d_off": d_off,
                          "min_clear_m": min(c_pitch, c_ls, c_disc), "clear_pitch_m": c_pitch,
                          "clear_lower_strut_m": c_ls, "clear_discs_m": c_disc,
                          "rod_len_m": float(np.linalg.norm(K - P)), "dist_from_hip_m": float(np.linalg.norm(P))})
    good = sorted([c for c in cands if c["min_clear_m"] >= C4_GAP_M], key=lambda c: c["dist_from_hip_m"])
    rest = sorted([c for c in cands if c["min_clear_m"] < C4_GAP_M], key=lambda c: -c["min_clear_m"])
    return good + rest


# --------------------------------------------------------------------------------- linkage sizing (finding MECH-3)
# The packaging record's rules (design/PACKAGING_PUSHROD_60CM.md:103-131, design/pushrod60.py): rod force = the
# drive's static torque / (crank radius x sin 43 deg); safety factor 2.0 on static tension (proof x tensile stress
# area), compression (Johnson's parabola or Euler, pinned, over the pin-to-pin length) and the rod end's static rating.
# Rods: rolled class 8.8 studs, proof 580 MPa, E 200 GPa (pushrod60.SOURCES '8.8'); stress areas by the record's
# formula 0.7854((d2 + d3)/2)^2 on ISO pitch diameters (M8 36.6 mm2 as the record; M10 58.0, M12 84.3: DESK).
# Rod ends (steel on steel, DIN ISO 12240-4): M8 C0 12.9 kN (the record's rule figure; its lowest found 7.9 kN);
# M10 17.6 kN (MISUMI, uk.misumi-ec.com) and M12 20.8 kN (INA, uk.rs-online.com 2600271): the lowest of a web-search
# summary read 28 Nov 2026 (pages not opened; others list 23.5-25.5 kN).
ROD_THREADS = {"M8": dict(nominal_mm=8.0, as_mm2=36.6, c0_kn=12.9, c0_low_kn=7.9),
               "M10": dict(nominal_mm=10.0, as_mm2=58.0, c0_kn=17.6, c0_low_kn=17.6),
               "M12": dict(nominal_mm=12.0, as_mm2=84.3, c0_kn=20.8, c0_low_kn=20.8)}
ROD_PROOF_MPA, ROD_E_MPA, ROD_SF = 580.0, 200e3, 2.0
TRANSMISSION_DEG = 43.0
AXLE_D_M = 0.0128    # the model's knee and elbow axles (r 6.4 mm, fore128.xml:194, :259)


def _johnson_euler_n(dia_mm: float, L_mm: float, e=ROD_E_MPA, sy=ROD_PROOF_MPA) -> float:
    """Pinned column (as design/pushrod60.py johnson_euler): Johnson's parabola below the transition, else Euler."""
    A, I, k = math.pi * dia_mm ** 2 / 4, math.pi * dia_mm ** 4 / 64, dia_mm / 4
    sl, cc = L_mm / k, math.sqrt(2 * math.pi ** 2 * e / sy)
    return A * sy * (1 - sy * sl ** 2 / (4 * math.pi ** 2 * e)) if sl < cc else math.pi ** 2 * e * I / L_mm ** 2


def size_linkage(drive: dict, s: float, rod_len_m: float) -> dict:
    """The smallest class 8.8 rod (M8, M10, M12) that passes the record's three static checks at SF 2.0 for this knee
    drive, crank and rod length; its drawn radius is half its nominal size plus 0.5 mm (M8: the model's 4.5 mm)."""
    rc = crank_radius(s)
    f = drive["static"] / (rc * math.sin(math.radians(TRANSMISSION_DEG)))
    rows = {}
    chosen = None
    for th, t in ROD_THREADS.items():
        dia = math.sqrt(4 * t["as_mm2"] / math.pi)
        tens = ROD_PROOF_MPA * t["as_mm2"]
        comp = min(tens, _johnson_euler_n(dia, 1000 * rod_len_m))
        r = {"sf_tension": tens / f, "sf_compression": comp / f, "sf_rod_end": 1000 * t["c0_kn"] / f,
             "sf_rod_end_lowest_found": 1000 * t["c0_low_kn"] / f}
        r["passes"] = min(r["sf_tension"], r["sf_compression"], r["sf_rod_end"]) >= ROD_SF
        rows[th] = r
        if chosen is None and r["passes"]:
            chosen = th
    return {"static_torque_nm": drive["static"], "crank_m": rc, "rod_len_m": rod_len_m, "rod_force_n": f,
            "by_thread": rows, "thread": chosen, "sized": chosen is not None,
            "rod_radius_m": (ROD_THREADS[chosen or "M12"]["nominal_mm"] / 2 + 0.5) / 1000.0,
            "m8_passes": rows["M8"]["passes"]}


def axle_and_thigh(force_n: float, peak_force_n: float, rod_y_m: float, r_ls_m: float) -> dict:
    """Finding MECH-4 (DESK): the knee axle as a solid 12.8 mm cantilever carrying the rod force at the crank plane,
    arm = crank plane to the lower strut's surface; bending against class 8.8 proof (580 MPa), and the diameter for
    SF 2.0; and the thigh's twisting moment from the rod's offset off the strut (F x rod plane)."""
    arm = max(rod_y_m - r_ls_m, 0.0)
    z = math.pi * AXLE_D_M ** 3 / 32
    sig_s, sig_p = force_n * arm / z / 1e6, peak_force_n * arm / z / 1e6
    d_req = (32 * force_n * arm * ROD_SF / (math.pi * ROD_PROOF_MPA * 1e6)) ** (1 / 3)
    return {"arm_m": arm, "bending_static_mpa": sig_s, "bending_at_limit_mpa": sig_p,
            "sf_static_on_proof": ROD_PROOF_MPA / sig_s if sig_s > 0 else math.inf,
            "axle_d_for_sf2_m": d_req, "thigh_moment_static_nm": force_n * rod_y_m,
            "thigh_moment_at_limit_nm": peak_force_n * rod_y_m}


def layout_parts(S: float, dset: dict, layout: str, as_drawn: bool = False, place_by_end: dict | None = None,
                 tail_extra_m: float = 0.0, size_rods: bool = True) -> dict:
    """The desk layout: every solid part as a capsule, cylinder or box, by body. Positions and structural sections
    scale with s = S / 0.60 from the 60 cm model; drives keep their size; rod (4.5 mm) and axles (6.4 mm) keep the
    model's size only ``as_drawn`` or without ``size_rods``.
    Study chassis = the packaging record's layout C pattern (design/PACKAGING_PUSHROD_60CM.md:45-63): rails inset to
    y 75.2 mm x s and ended at crossmembers placed inboard of the roll drives (and of the pitch drive's radius) by
    6 mm, each roll drive on a bracket to its inboard face, linkages outboard at both legs, tail pitch servo 16 mm aft.
    ``as_drawn``: the 60 cm model as built (crossmembers at x +-260 mm through the roll drives, rails at 83.2 mm, the
    hind knee drive on its strut's axis 90 mm along it), for the validation against the packaging report.
    Revised 28 Nov 2026 after review: each foot is its own body on an ankle joint (-22 to +22 deg), the sole's axis
    12 mm x s below the ankle as in the model (MECH-7; before, 32.8 mm x s on the shank); with ``size_rods`` each rod
    is sized for its knee drive (``size_linkage``) and drawn at that radius, its lane (and the cranks and axle) moved
    outboard to keep the 5.5 mm rod-to-drive-face gap (MECH-3). Revised again 28 Nov 2026 after the third review
    (R3-1): with ``size_rods`` each knee axle is also drawn at the larger of the model's 12.8 mm and its own solid
    diameter for SF 2.0 at the static rod case (``axle_and_thigh``), so that C4 is not scored on an axle that would
    yield. No outboard bearing or fork is drawn: none is designed."""
    s = S / S60
    dr = drives_of(dset)
    rr, Lr = dr["roll"]["dia"] / 2, dr["roll"]["length"]
    rp, Lp = dr["pitch"]["dia"] / 2, dr["pitch"]["length"]
    r_us, r_ls, r_yoke = 0.0112 * s, 0.0096 * s, 0.0112 * s
    r_rail, r_cm = 0.0096 * s, 0.0136 * s
    r_rod, r_axle = 0.0045, 0.0064
    crank_r = crank_radius(s)
    disc_r = crank_r + 0.007
    y_r = max(0.076 * s, rr + 0.0025)
    parts: list[Part] = []
    info = {"s": s, "y_roll_m": y_r, "ends": {}}
    x_cm = {}
    for end in ("fore", "hind"):
        xh, zh = LEG[end]["hip"][0] * s, LEG[end]["hip"][1] * s
        sgn = 1.0 if end == "fore" else -1.0
        x_cm[end] = sgn * 0.26 * s if as_drawn else xh - sgn * (max(Lr / 2, rp) + 0.006 + r_cm)
        parts.append(Part(f"{end}_roll_drive", "chassis", "cylinder", (xh - Lr / 2, y_r, zh), (xh + Lr / 2, y_r, zh),
                          rr))
        if not as_drawn:
            x_in = x_cm[end] + sgn * r_cm
            x_face = xh - sgn * Lr / 2
            parts.append(Part(f"bracket_{end}", "chassis", "box", ((x_in + x_face) / 2, y_r, zh),
                              (abs(x_face - x_in) / 2, 0.8 * rr, 0.006)))
    y_rail = 0.0832 * s if as_drawn else 0.0752 * s
    z_rail = -0.012 * s
    if as_drawn:
        parts.append(Part("rail", "chassis", "capsule", (-0.26 * s, y_rail, z_rail), (0.26 * s, y_rail, z_rail), r_rail))
        for end in ("fore", "hind"):
            parts.append(Part(f"cm_{end}", "chassis", "capsule", (x_cm[end], -0.136 * s, 0.0), (x_cm[end], 0.136 * s, 0.0),
                              r_cm))
    else:
        parts.append(Part("rail", "chassis", "capsule", (x_cm["hind"], y_rail, z_rail), (x_cm["fore"], y_rail, z_rail),
                          r_rail))
        for end in ("fore", "hind"):
            # between the rails, ending on their centrelines (its rounded ends do not stand out past the rails)
            parts.append(Part(f"cm_{end}", "chassis", "capsule", (x_cm[end], -(y_rail - r_cm), 0.0),
                              (x_cm[end], y_rail - r_cm, 0.0), r_cm))
        xb0, xb1 = x_cm["hind"] + r_cm + 0.003, x_cm["fore"] - r_cm - 0.003
        yb = y_rail - r_rail - 0.002
        parts.append(Part("bay", "chassis", "box", ((xb0 + xb1) / 2, 0.0, (-0.064 * s + 0.03 * s) / 2),
                          ((xb1 - xb0) / 2, yb, (0.03 * s + 0.064 * s) / 2)))
        info["bay_box_m"] = [xb1 - xb0, 2 * yb, 0.094 * s]
    # tail (sim/generated/...fore128.xml:170-176): yaw servo on the chassis, pitch servo on the tail body
    parts.append(Part("tail_yaw_servo", "chassis", "box", (-0.248 * s, 0.0, 0.038 * s), (0.0143, 0.017, 0.0233)))
    parts.append(Part("tail_pitch_servo", "tail", "box",
                      (-0.024 * s - (0.0 if as_drawn else 0.016 + tail_extra_m), 0.0, 0.0), (0.0143, 0.017, 0.0233)))
    info["tail_pitch_servo_aft_m"] = 0.0 if as_drawn else 0.016 + tail_extra_m
    info["tail_pos"] = (-0.26 * s, 0.0, 0.0)
    for end in ("fore", "hind"):
        L = LEG[end]
        kn = dr[f"{end}_knee"]
        rk, Lk = kn["dia"] / 2, kn["length"]
        u, w = _unit(L["u"]), _unit(L["w"])
        K = u * L["l_u"] * s
        e_p = max(0.064 * s, rr + 0.006 + Lp / 2) if not as_drawn else 0.064
        if layout == "L2" and not as_drawn:
            y_link = e_p + Lp / 2 + r_us + 0.002
            face = r_us + 0.002 + Lk
            axis_pt = np.zeros(2)
            place = {"t": 0.0, "P": axis_pt, "K": K, "rod_len_m": L["l_u"] * s, "dist_from_hip_m": 0.0}
            knee_from, knee_to = (0.0, r_us + 0.002, 0.0), (0.0, r_us + 0.002 + Lk, 0.0)
        else:
            y_link = e_p
            if as_drawn and end == "hind":
                P = u * 0.090 * s
                place = {"t": 0.090 / L["l_u"], "P": P, "K": K, "rod_len_m": float(np.linalg.norm(K - P)),
                         "dist_from_hip_m": 0.090 * s, "as_drawn": True}
            elif as_drawn:
                P = np.array([-0.0941178467, -0.0047025603]) * s
                place = {"t": None, "P": P, "K": K, "rod_len_m": float(np.linalg.norm(K - P)),
                         "dist_from_hip_m": float(np.linalg.norm(P)), "as_drawn": True}
            elif place_by_end and end in place_by_end:
                place = dict(place_by_end[end])
            else:
                place = knee_drive_candidates(end, s, rk, rp, r_us, r_ls, disc_r)[0]
            P = place["P"]
            face = max(Lk / 2, (e_p + Lp / 2) - y_link)
            knee_from, knee_to = (P[0], -Lk / 2, P[1]), (P[0], Lk / 2, P[1])
            axis_pt = P
        rod_len = float(np.linalg.norm(np.asarray(K) - np.asarray(axis_pt)))
        sizing = size_linkage(kn, s, rod_len)
        r_rod_e = sizing["rod_radius_m"] if (size_rods and not as_drawn) else r_rod
        rod_y = face + ROD_FACE_GAP_M + r_rod_e
        pk_force = 0.70 * kn["peak"] / (crank_radius(s) * math.sin(math.radians(TRANSMISSION_DEG)))
        axle = axle_and_thigh(sizing["rod_force_n"], pk_force, rod_y, r_ls)
        r_axle_e = max(AXLE_D_M, axle["axle_d_for_sf2_m"]) / 2 if (size_rods and not as_drawn) else r_axle
        axle["drawn_d_m"] = 2 * r_axle_e
        axle["bending_static_drawn_mpa"] = axle["bending_static_mpa"] * (AXLE_D_M / (2 * r_axle_e)) ** 3
        yoke = f"{end}_yoke"
        parts += [Part(f"{end}_yoke", yoke, "capsule", (0, 0, 0), (0, e_p, 0), r_yoke),
                  Part(f"{end}_pitch_drive", yoke, "cylinder", (0, e_p - Lp / 2, 0), (0, e_p + Lp / 2, 0), rp)]
        th = f"{end}_thigh"
        parts += [Part(f"{end}_upper_strut", th, "capsule", (0, 0, 0), (K[0], 0, K[1]), r_us),
                  Part(f"{end}_knee_drive", th, "cylinder", knee_from, knee_to, rk),
                  Part(f"{end}_drive_crank", th, "cylinder", (axis_pt[0], rod_y - 0.005, axis_pt[1]),
                       (axis_pt[0], rod_y + 0.005, axis_pt[1]), disc_r),
                  Part(f"{end}_axle", th, "cylinder", (K[0], -0.004, K[1]), (K[0], rod_y + 0.007, K[1]), r_axle_e)]
        sh = f"{end}_shank"
        F = w * L["l_l"] * s
        ft = f"{end}_foot"
        parts += [Part(f"{end}_lower_strut", sh, "capsule", (0, 0, 0), (F[0], 0, F[1]), r_ls),
                  Part(f"{end}_output_crank", sh, "cylinder", (0, rod_y - 0.005, 0), (0, rod_y + 0.005, 0), disc_r),
                  Part(f"{end}_ankle_hub", ft, "capsule", (0, 0, 0), (0, 0, -0.0208 * s), 0.0096 * s),
                  Part(f"{end}_sole", ft, "capsule", (-0.02 * s, 0, -SOLE_BELOW_ANKLE_M * s),
                       (0.02 * s, 0, -SOLE_BELOW_ANKLE_M * s), 0.024 * s)]
        KP = np.asarray(K) - np.asarray(axis_pt)
        nrm = np.array([KP[1], -KP[0]]) / max(np.linalg.norm(KP), 1e-9)
        crank0 = nrm * crank_r
        parts.append(Part(f"{end}_rod", f"{end}_rod", "capsule", (0, 0, -np.linalg.norm(KP) / 2),
                          (0, 0, np.linalg.norm(KP) / 2), r_rod_e))
        info["ends"][end] = {"hip": (L["hip"][0] * s, y_r, L["hip"][1] * s), "e_p": e_p, "y_link": y_link,
                             "K": tuple(K), "F": tuple(F), "axis_pt": tuple(axis_pt), "crank0": tuple(crank0),
                             "rod_y": rod_y, "rod_radius_m": r_rod_e, "linkage": sizing,
                             "axle": axle,
                             "face_m": face, "knee_drive_place": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                                  for k, v in place.items()},
                             "half_width_m": y_r + y_link + rod_y + 0.005,
                             "knee_axle_overhang_m": rod_y + 0.007}
    info["roll_drive_pair_gap_m"] = 2 * (y_r - rr)
    return {"parts": parts, "info": info, "layout": layout, "as_drawn": as_drawn,
            "joined": JOINED | (JOINED_L2 if layout == "L2" and not as_drawn else set())}


def _fmt(v):
    return " ".join(f"{float(x):.6f}" for x in v)


LEG_BODIES = ("yoke", "thigh", "shank", "foot", "rod")


def _mirror_parts(parts: list) -> list:
    """The right-hand legs: each left leg part (and its roll drive) mirrored in y, names and bodies prefixed 'R'."""
    out = []
    for p in parts:
        leg = any(p.body == f"{e}_{b}" for e in ("fore", "hind") for b in LEG_BODIES)
        if not (leg or p.name.endswith("_roll_drive")):
            continue
        a = (p.a[0], -p.a[1], p.a[2])
        b = p.b if p.kind == "box" else (p.b[0], -p.b[1], p.b[2])
        out.append(Part("R" + p.name, p.body if p.body == "chassis" else "R" + p.body, p.kind, a, b, p.r))
    return out


def build_mjcf(lay: dict, mirror: bool = False) -> str:
    """The desk layout as MJCF (geometry only, no contacts). ``mirror``: the right-hand legs too (prefix 'R'), each
    the left leg's mirror image (roll about -x, as the model's right legs), for the left-against-right checks."""
    parts, info = list(lay["parts"]), lay["info"]
    if mirror:
        parts = parts + _mirror_parts(parts)

    def geoms(body):
        out = []
        for p in parts:
            if p.body != body:
                continue
            if p.kind == "box":
                out.append(f'<geom name="{p.name}" type="box" pos="{_fmt(p.a)}" size="{_fmt(p.b)}"/>')
            else:
                out.append(f'<geom name="{p.name}" type="{p.kind}" fromto="{_fmt(p.a)} {_fmt(p.b)}" '
                           f'size="{p.r:.6f}"/>')
        return "\n".join(out)

    legs = []
    sides = [("", 1.0)] + ([("R", -1.0)] if mirror else [])
    for pre, sg in sides:
        for end in ("fore", "hind"):
            e = info["ends"][end]
            ax = LEG[end]["flex_axis"]
            n = f"{pre}{end}"
            hip = (e["hip"][0], sg * e["hip"][1], e["hip"][2])
            legs.append(f"""
<body name="{n}_yoke" pos="{_fmt(hip)}"><joint name="{n}_roll" axis="{sg:g} 0 0" range="-18 24"/>
{geoms(f'{n}_yoke')}
<body name="{n}_thigh" pos="0 {sg * e['y_link']:.6f} 0"><joint name="{n}_pitch" axis="0 1 0" range="-38 38"/>
{geoms(f'{n}_thigh')}
<body name="{n}_shank" pos="{e['K'][0]:.6f} 0 {e['K'][1]:.6f}"><joint name="{n}_flexion" axis="{_fmt(ax)}"
 range="-47 36"/>
{geoms(f'{n}_shank')}
<body name="{n}_foot" pos="{e['F'][0]:.6f} 0 {e['F'][1]:.6f}"><joint name="{n}_ankle" axis="0 1 0" range="-22 22"/>
{geoms(f'{n}_foot')}
</body></body></body></body>""")
    rods = "\n".join(f'<body name="{pre}{end}_rod" mocap="true">{geoms(f"{pre}{end}_rod")}</body>'
                     for pre, _ in sides for end in ("fore", "hind"))
    return f"""<mujoco><compiler angle="degree" autolimits="true"/><option gravity="0 0 0"/>
<default><geom contype="0" conaffinity="0"/></default>
<worldbody>
<body name="chassis">
{geoms('chassis')}
{''.join(legs)}
<body name="tail" pos="{_fmt(info['tail_pos'])}"><joint name="tail_yaw" axis="0 0 1" range="-24 24"/>
<joint name="tail_pitch" axis="0 1 0" range="-15 15"/>
{geoms('tail')}
</body>
</body>
{rods}
</worldbody></mujoco>"""


class Geometry:
    """A compiled desk layout with its pose setter and distance function (mujoco 3.10.0 mj_geomDistance: negative =
    overlap depth). An exact 0.0, which the solver returns on some failed solves (sim/check_packaging60.py:219-241),
    is re-measured at +-1e-3 rad on the swept joints, as the build's packaging check does."""

    def __init__(self, lay: dict, mirror: bool = False):
        import mujoco

        self.mj = mujoco
        self.lay = lay
        self.mirror = mirror
        self.m = mujoco.MjModel.from_xml_string(build_mjcf(lay, mirror))
        self.d = mujoco.MjData(self.m)
        self.gid = {self.m.geom(i).name: i for i in range(self.m.ngeom)}
        self.body_of = {n: self.m.body(int(self.m.geom_bodyid[i])).name for n, i in self.gid.items()}
        self.jadr = {self.m.joint(j).name: int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt)}
        self.fromto = np.zeros(6)

    def set_pose(self, q: dict):
        m, d, mj = self.m, self.d, self.mj
        d.qpos[:] = 0.0
        for jn, v in q.items():
            d.qpos[self.jadr[jn]] = math.radians(v)
        mj.mj_kinematics(m, d)
        sides = [("", 1.0)] + ([("R", -1.0)] if self.mirror else [])
        for pre, sg, end in [(p, g, e) for p, g in sides for e in ("fore", "hind")]:
            e = self.lay["info"]["ends"][end]
            n = f"{pre}{end}"
            bt, bs = m.body(f"{n}_thigh").id, m.body(f"{n}_shank").id
            Rt, Rs = d.xmat[bt].reshape(3, 3), d.xmat[bs].reshape(3, 3)
            c0 = np.array([e["crank0"][0], 0.0, e["crank0"][1]])
            crank_w = Rs @ c0
            top = d.xpos[bt] + Rt @ np.array([e["axis_pt"][0], sg * e["rod_y"], e["axis_pt"][1]]) + crank_w
            bot = d.xpos[bt] + Rt @ np.array([e["K"][0], sg * e["rod_y"], e["K"][1]]) + crank_w
            v = top - bot
            qt = np.zeros(4)
            mj.mju_quatZ2Vec(qt, v / np.linalg.norm(v))
            mb = m.body(f"{n}_rod").mocapid[0]
            d.mocap_pos[mb] = 0.5 * (top + bot)
            d.mocap_quat[mb] = qt
        mj.mj_kinematics(m, d)

    def dist(self, a: str, b: str, q: dict | None = None) -> float:
        x = float(self.mj.mj_geomDistance(self.m, self.d, self.gid[a], self.gid[b], 0.05, self.fromto))
        if x == 0.0 and q:
            tries = []
            for sgn in (-1e-3, 1e-3):
                self.set_pose({j: v + math.degrees(sgn) for j, v in q.items()})
                tries.append(float(self.mj.mj_geomDistance(self.m, self.d, self.gid[a], self.gid[b], 0.05,
                                                           self.fromto)))
            self.set_pose(q)
            good = [t for t in tries if t != 0.0]
            x = min(good) if good else 0.0
        return x


def _pairs_for_end(geo: Geometry, end: str) -> list[tuple[str, str]]:
    lay = geo.lay
    names = list(geo.gid)
    leg_bodies = {f"{end}_{b}" for b in LEG_BODIES}
    mine = [n for n in names if geo.body_of[n] in leg_bodies] + [f"{end}_roll_drive"]
    chassis = [n for n in names if geo.body_of[n] == "chassis" and _end(n) in (None, end) and not n.startswith("tail")
               and not n.startswith("R")]
    out = set()
    for a in mine:
        for b in mine + chassis:
            if a == b:
                continue
            if frozenset((_base(a), _base(b))) in lay["joined"]:
                continue
            if geo.body_of[a] == "chassis" and geo.body_of[b] == "chassis" and "roll_drive" not in a + b:
                continue
            out.add(tuple(sorted((a, b))))
    return sorted(out)


GRID = {"roll": np.linspace(-18, 24, 6), "pitch": np.linspace(-38, 38, 7), "flexion": np.linspace(-47, 36, 7)}
GRID_COARSE = {"roll": np.linspace(-18, 24, 4), "pitch": np.linspace(-38, 38, 5), "flexion": np.linspace(-47, 36, 5)}
PRUNE_M = 0.03   # a pair whose bounding spheres are this far apart is not measured at that pose (its true distance is
#                  larger; the sphere gap, a lower bound, is kept instead)


def _min_over_poses(geo: Geometry, pairs, poses) -> dict:
    gi = np.array([[geo.gid[a], geo.gid[b]] for a, b in pairs], dtype=int).reshape(-1, 2)
    rb = geo.m.geom_rbound
    best = {p: (1.0, None) for p in pairs}
    for q in poses:
        geo.set_pose(q)
        xp = geo.d.geom_xpos
        gap = np.linalg.norm(xp[gi[:, 0]] - xp[gi[:, 1]], axis=1) - rb[gi[:, 0]] - rb[gi[:, 1]]
        for i, pr in enumerate(pairs):
            x = float(gap[i]) if gap[i] > PRUNE_M else geo.dist(*pr, q=q)
            if x < best[pr][0]:
                best[pr] = (x, tuple(round(float(v), 2) for v in q.values()))
    return best


def _leg_poses(end: str, grid) -> list[dict]:
    return [{f"{end}_roll": float(r), f"{end}_pitch": float(p_), f"{end}_flexion": float(f)}
            for r in grid["roll"] for p_ in grid["pitch"] for f in grid["flexion"]]


TAIL_PAIRS = (("tail_pitch_servo", "cm_hind"), ("tail_pitch_servo", "tail_yaw_servo"), ("tail_yaw_servo", "cm_hind"),
              ("tail_pitch_servo", "hind_roll_drive"))
TAIL_POSES = [{"tail_yaw": float(y), "tail_pitch": float(p_)} for y in np.linspace(-24, 24, 5)
              for p_ in np.linspace(-15, 15, 5)]
MOVERS = ("knee_drive", "drive_crank", "rod", "output_crank", "axle")
ROD_FACE_GAP_M = 0.0055   # by construction: rod centre 10 mm off the drive's outer face, rod radius 4.5 mm


def make_layout(S: float, dset: dict, layout: str) -> dict:
    """The study's desk layout for a configuration: the tail pitch servo moved aft (16 mm, the packaging record's
    fix, plus 10 mm steps up to 40 mm more) until the tail pairs clear by 6 mm; in L1, each knee drive at the first
    candidate (``knee_drive_candidates``) whose knee drive, cranks, rod and axle clear everything by the rod's own
    constructed 5.5 mm on a coarse roll x pitch x flexion grid (else the candidate with the largest such clearance)."""
    extra = 0.0
    for extra in (0.0, 0.01, 0.02, 0.03, 0.04):
        geo = Geometry(layout_parts(S, dset, layout, tail_extra_m=extra))
        tb = _min_over_poses(geo, [p for p in TAIL_PAIRS], TAIL_POSES)
        if min(v[0] for v in tb.values()) >= 0.006:
            break
    if layout != "L1":
        return layout_parts(S, dset, layout, tail_extra_m=extra)
    s = S / S60
    dr = drives_of(dset)
    chosen, tried = {}, {}
    for end in ("fore", "hind"):
        kn = dr[f"{end}_knee"]
        cands = knee_drive_candidates(end, s, kn["dia"] / 2, dr["pitch"]["dia"] / 2, 0.0112 * s, 0.0096 * s,
                                      crank_radius(s) + 0.007)
        best, n = None, 0
        for c in cands[:14]:
            n += 1
            lay = layout_parts(S, dset, layout, place_by_end={**chosen, end: c}, tail_extra_m=extra)
            geo = Geometry(lay)
            prs = [pr for pr in _pairs_for_end(geo, end) if any(_base(x) in MOVERS for x in pr)]
            mn = min(v[0] for v in _min_over_poses(geo, prs, _leg_poses(end, GRID_COARSE)).values())
            if best is None or mn > best[0]:
                best = (mn, c)
            if mn >= ROD_FACE_GAP_M - 1e-6:   # nothing nearer than the rod's own constructed gap to its drive
                break
        chosen[end] = best[1]
        tried[end] = {"candidates_tried": n, "coarse_min_m": best[0]}
    lay = layout_parts(S, dset, layout, place_by_end=chosen, tail_extra_m=extra)
    lay["info"]["placement_search"] = tried
    return lay


CROSS_SEL = ("lower_strut", "ankle_hub", "sole", "output_crank", "knee_drive")
FOOT_GRID = {"roll": np.linspace(-18, 24, 4), "pitch": np.linspace(-38, 38, 5), "flexion": np.linspace(-47, 36, 5),
             "ankle": np.array([-22.0, 22.0])}
FH_GRID = {"roll": np.array([-18.0, 3.0, 24.0]), "pitch": np.linspace(-38, 38, 5), "flexion": np.linspace(-47, 36, 5),
           "ankle": np.array([-22.0, 0.0, 22.0])}
LR_GRID = {"roll": np.linspace(-18, 24, 8), "pitch": np.linspace(-38, 38, 9), "flexion": np.linspace(-47, 36, 9),
           "ankle": np.linspace(-22, 22, 5)}
RANGES = {"roll": (-18.0, 24.0), "pitch": (-38.0, 38.0), "flexion": (-47.0, 36.0), "ankle": (-22.0, 22.0)}


def _refine(geo: Geometry, pairs, q0: dict, iters: int = 400) -> tuple[float, dict]:
    """A local search (Nelder-Mead, each joint clipped to its range) from a grid pose, on the least distance over
    ``pairs``: the record's 'local search between grid points' (PACKAGING_PUSHROD_60CM.md:13-16)."""
    from scipy.optimize import minimize

    names = list(q0)
    lo = np.array([RANGES[n.split("_")[-1]][0] for n in names])
    hi = np.array([RANGES[n.split("_")[-1]][1] for n in names])

    def f(x):
        q = dict(zip(names, np.clip(x, lo, hi)))
        geo.set_pose(q)
        return min(geo.dist(a, b) for a, b in pairs)

    x0 = np.array([q0[n] for n in names], dtype=float)
    r = minimize(f, x0, method="Nelder-Mead", options={"maxfev": iters, "xatol": 0.2, "fatol": 1e-5,
                                                       "initial_simplex": np.vstack([x0] + [x0 + 4.0 * np.eye(len(x0))[i]
                                                                                            for i in range(len(x0))])})
    best = min(float(r.fun), f(x0))
    xq = np.clip(r.x if r.fun <= f(x0) else x0, lo, hi)
    return best, {n: round(float(v), 2) for n, v in zip(names, xq)}


def _cross_sweep(geo: Geometry, pairs, poses, refine_from: int = 3) -> dict:
    """Least distance per pair over ``poses``; then a local search from the ``refine_from`` worst poses over all the
    pairs. Returns {'pairs': {pair: {min_m, pose_deg}}, 'least_m', 'least_pair', 'least_pose_deg'}."""
    gi = np.array([[geo.gid[a], geo.gid[b]] for a, b in pairs], dtype=int)
    rb = geo.m.geom_rbound
    best = {p: (1.0, None) for p in pairs}
    worst_poses = []
    for q in poses:
        geo.set_pose(q)
        xp = geo.d.geom_xpos
        gap = np.linalg.norm(xp[gi[:, 0]] - xp[gi[:, 1]], axis=1) - rb[gi[:, 0]] - rb[gi[:, 1]]
        lst = 1.0
        for i in np.flatnonzero(gap <= PRUNE_M):
            pr = pairs[i]
            x = geo.dist(*pr)
            lst = min(lst, x)
            if x < best[pr][0]:
                best[pr] = (x, {k: round(float(v), 2) for k, v in q.items()})
        worst_poses.append((lst, q))
    worst_poses.sort(key=lambda t: t[0])
    kmin = min(best, key=lambda p: best[p][0])
    least, lpose = best[kmin]
    ref = None
    for lst, q in worst_poses[:refine_from]:
        if lst >= PRUNE_M:
            break
        x, qq = _refine(geo, pairs, q)
        if x < least:
            least, lpose, ref = x, qq, x
    lpair = min(best, key=lambda p: best[p][0]) if best else None
    if ref is not None:  # which pair is least at the refined pose
        geo.set_pose(lpose)
        lpair = min(pairs, key=lambda p: geo.dist(*p))
    return {"pairs": {f"{a} | {b}": {"min_m": x, "pose_deg": p} for (a, b), (x, p) in best.items()},
            "least_m": least, "least_pair": None if lpair is None else f"{lpair[0]} | {lpair[1]}",
            "least_pose_deg": lpose}


def sweep_layout(lay: dict, grid=GRID) -> dict:
    """Every pair of each leg's parts (and its roll drive) against each other and the chassis, over roll x pitch x
    flexion (6 x 7 x 7 poses, ankle 0), the foot's parts again with the ankle at its range ends (4 x 5 x 5 x 2), the
    tail over its yaw x pitch range (5 x 5), and, revised 28 Nov 2026 after review (F2, MECH-5, MECH-7), the legs
    against each other with the ankles in the sweep: the front and hind legs of a side (struts, ankle hubs, soles,
    output cranks, knee drives; both legs at one roll, -18 / +3 / +24 deg, x pitch 5 x flexion 5 x ankle 3 each:
    16,875 poses), and the two hind legs and the two front legs against each other (mirror-image poses on roll 8 x
    pitch 9 x flexion 9 x ankle 5), each followed by a local search over every joint of both legs from the worst
    poses. Least signed distance per pair, m."""
    geo = Geometry(lay, mirror=True)
    res = {"pairs": {}, "tail": {}, "leg_to_leg": {}, "left_right": {}}
    for end in ("fore", "hind"):
        prs = _pairs_for_end(geo, end)
        for pr, (x, pose) in _min_over_poses(geo, prs, _leg_poses(end, grid)).items():
            res["pairs"][f"{pr[0]} | {pr[1]}"] = {"min_m": x, "pose_deg": pose, "end": end}
        fprs = [pr for pr in prs if any(_base(x) in ("sole", "ankle_hub") for x in pr)]
        fposes = [{f"{end}_roll": float(r), f"{end}_pitch": float(p_), f"{end}_flexion": float(f),
                   f"{end}_ankle": float(a)} for r in FOOT_GRID["roll"] for p_ in FOOT_GRID["pitch"]
                  for f in FOOT_GRID["flexion"] for a in FOOT_GRID["ankle"]]
        for pr, (x, pose) in _min_over_poses(geo, fprs, fposes).items():
            k = f"{pr[0]} | {pr[1]}"
            if x < res["pairs"][k]["min_m"]:
                res["pairs"][k] = {"min_m": x, "pose_deg": pose, "end": end, "ankle_swept": True}
    res["tail"] = {f"{a} | {b}": {"min_m": x, "pose_deg": pose}
                   for (a, b), (x, pose) in _min_over_poses(geo, list(TAIL_PAIRS), TAIL_POSES).items()}
    # the front and hind legs of one side (record #9)
    lpairs = [(f"fore_{a}", f"hind_{b}") for a in CROSS_SEL for b in CROSS_SEL]
    one = [(p_, f, a) for p_ in FH_GRID["pitch"] for f in FH_GRID["flexion"] for a in FH_GRID["ankle"]]
    poses = [{"fore_roll": float(r), "fore_pitch": float(a[0]), "fore_flexion": float(a[1]), "fore_ankle": float(a[2]),
              "hind_roll": float(r), "hind_pitch": float(b[0]), "hind_flexion": float(b[1]), "hind_ankle": float(b[2])}
             for r in FH_GRID["roll"] for a in one for b in one]
    res["leg_to_leg"] = _cross_sweep(geo, lpairs, poses)
    # left against right (record #10 at the hind; the front pair is checked the same way)
    for end in ("hind", "fore"):
        prs = [(f"{end}_{a}", f"R{end}_{b}") for a in CROSS_SEL for b in CROSS_SEL]
        poses = [{f"{end}_roll": float(r), f"{end}_pitch": float(p_), f"{end}_flexion": float(f), f"{end}_ankle": float(a),
                  f"R{end}_roll": float(r), f"R{end}_pitch": float(p_), f"R{end}_flexion": float(f),
                  f"R{end}_ankle": float(a)}
                 for r in LR_GRID["roll"] for p_ in LR_GRID["pitch"] for f in LR_GRID["flexion"] for a in LR_GRID["ankle"]]
        res["left_right"][end] = _cross_sweep(geo, prs, poses)
    return res


KNOWN_CLASHES = {  # the 60 cm design's known clashes, numbered as the packaging record's current table
    # (design/PACKAGING_PUSHROD_60CM.md:33-44, revised 28 Nov 2026; this study's list was brought in line with it
    # after review, F2 / MECH-5)
    1: ("hind knee drive inside its own upper strut", [("hind_knee_drive", "hind_upper_strut")]),
    2: ("pelvic crossmember through the hind roll / pitch drives, yoke and strut",
        [("cm_hind", "hind_roll_drive"), ("cm_hind", "hind_pitch_drive"), ("cm_hind", "hind_yoke"),
         ("cm_hind", "hind_upper_strut")]),
    3: ("shoulder crossmember into the front roll / pitch drives, yoke and strut",
        [("cm_fore", "fore_roll_drive"), ("cm_fore", "fore_pitch_drive"), ("cm_fore", "fore_yoke"),
         ("cm_fore", "fore_upper_strut")]),
    4: ("side rails into the roll drives", [("fore_roll_drive", "rail"), ("hind_roll_drive", "rail")]),
    5: ("side rails into the moving hind yokes", [("hind_yoke", "rail")]),
    6: ("hind cranks and pushrod against the rails",
        [("hind_drive_crank", "rail"), ("hind_output_crank", "rail"), ("hind_rod", "rail")]),
    7: ("tail pitch servo into the pelvic crossmember", [("tail_pitch_servo", "cm_hind")]),
    8: ("tail pitch servo into the tail yaw servo", [("tail_pitch_servo", "tail_yaw_servo")]),
    9: ("front and hind legs of a side meet under the body: feet, ankle hubs and lower struts",
        [(f"fore_{a}", f"hind_{b}") for a in ("lower_strut", "ankle_hub", "sole")
         for b in ("lower_strut", "ankle_hub", "sole")]),
    10: ("the two hind feet meet at the inward roll stop",
         [(f"hind_{a}", f"Rhind_{b}") for a in ("lower_strut", "ankle_hub", "sole")
          for b in ("lower_strut", "ankle_hub", "sole")]),
    11: ("neck cradle end into the neck yaw servo", []),
    12: ("elbow and knee axles stop short of their output cranks (0.75 mm gap)", []),
}


def _lookup(sw: dict, a: str, b: str):
    for sect in ("pairs", "tail"):
        for key in (f"{a} | {b}", f"{b} | {a}"):
            if key in sw[sect]:
                return sw[sect][key]["min_m"]
    for sect in [sw["leg_to_leg"]] + list(sw["left_right"].values()):
        for key in (f"{a} | {b}", f"{b} | {a}"):
            if key in sect["pairs"]:
                return sect["pairs"][key]["min_m"]
    return None


def score_packaging(lay: dict, sw: dict) -> dict:
    """C4, revised 28 Nov 2026 after review (MECH-1, F1, F2, MECH-5).

    ``pass`` is C4 AS WRITTEN: every leg pair (drives, linkages and structure; own leg, chassis, bay box; the foot
    with its ankle swept), the tail pairs, the legs against each other (front against hind, left against right) at
    least 5 mm over the full ranges; the left and right roll drives 5 mm apart; and every known clash of the record
    removed. No layout here removes #9 (the front and hind legs of a side meeting under the body), and #11 (the neck)
    is not modelled, so no configuration passes C4 as written.

    ``pass_except_9_and_11`` is a SUPPORTING reading, CHANGED AFTER A RESULT: the reading the first version of this
    study scored as C4 (chosen once it was seen that no layout removes #9), now with #10 and the legs' left-against-
    right sweep inside it. It leaves out #9, whose fix (a range stop or a coupled limit, record row 9) narrows ranges
    the gaits use and is not swept or checked for its torque effect here, and #11, a neck clevis. #12 is removed by
    construction (each knee axle runs through its output crank's hub to its outer face)."""
    leg = {k: v for k, v in sw["pairs"].items()}
    worst_key = min(leg, key=lambda k: leg[k]["min_m"])
    other = {k: v for k, v in leg.items() if not ("knee_drive" in k and "_rod" in k)}
    other_key = min(other, key=lambda k: other[k]["min_m"])
    tail_worst = min(sw["tail"].values(), key=lambda v: v["min_m"])["min_m"]
    fails = sorted([(k, v["min_m"], v["pose_deg"]) for k, v in leg.items() if v["min_m"] < C4_GAP_M],
                   key=lambda x: x[1])
    fails += sorted([(k, v["min_m"], v["pose_deg"]) for k, v in sw["tail"].items() if v["min_m"] < C4_GAP_M],
                    key=lambda x: x[1])
    known = {}
    for n, (label, prs) in KNOWN_CLASHES.items():
        vals = [x for x in (_lookup(sw, a, b) for a, b in prs) if x is not None]
        if n == 12:
            known[n] = {"label": label, "min_m": None, "removed": True, "how": "by construction (axle through the hub)"}
        elif n == 1 and lay["layout"] == "L2":
            known[n] = {"label": label, "min_m": None, "removed": True,
                        "how": "by the layout (the knee drive sits at the hip, bolted to the thigh's root plate)"}
        elif n == 11:
            known[n] = {"label": label, "min_m": None, "removed": None, "how": "not modelled here (record: clevis)"}
        else:
            sweep = {9: sw["leg_to_leg"], 10: sw["left_right"]["hind"]}.get(n)
            if sweep is not None and sweep["least_pair"] in {f"{a} | {b}" for a, b in prs}:
                vals.append(sweep["least_m"])   # the local search's figure, when its least pair is this clash's
            mn = min(vals) if vals else None
            known[n] = {"label": label, "min_m": mn, "removed": None if mn is None else mn >= C4_GAP_M}
    roll_gap = lay["info"]["roll_drive_pair_gap_m"]
    lr = {e: sw["left_right"][e]["least_m"] for e in ("fore", "hind")}
    fh = sw["leg_to_leg"]["least_m"]
    least_own = min(leg[worst_key]["min_m"], tail_worst, roll_gap)
    least_lr = min(lr.values())
    least_all = min(least_own, least_lr, fh)
    sized = all(e["linkage"]["sized"] for e in lay["info"]["ends"].values())
    passed = least_all >= C4_GAP_M and all(known[n]["removed"] for n in KNOWN_CLASHES) and sized
    sup_scope = [n for n in KNOWN_CLASHES if n not in (9, 11)]
    passed_sup = least_own >= C4_GAP_M and least_lr >= C4_GAP_M and sized and all(known[n]["removed"] for n in sup_scope)
    return {"pass": passed, "pass_except_9_and_11": passed_sup,
            "least_clearance_m": least_own, "margin_m": least_own - C4_GAP_M, "worst_pair": worst_key,
            "next_tightest_pair": other_key, "next_tightest_m": other[other_key]["min_m"],
            "next_tightest_pose_deg": other[other_key]["pose_deg"],
            "worst_pair_m": leg[worst_key]["min_m"], "worst_pair_pose_deg": leg[worst_key]["pose_deg"],
            "tail_least_m": tail_worst, "roll_drive_pair_gap_m": roll_gap,
            "pairs_under_5mm": [{"pair": k, "min_m": x, "pose_deg": p} for k, x, p in fails[:8]],
            "n_pairs_under_5mm": len(fails), "known_clashes": known,
            "leg_to_leg_least_m": fh, "leg_to_leg_pair": sw["leg_to_leg"]["least_pair"],
            "leg_to_leg_pose_deg": sw["leg_to_leg"]["least_pose_deg"],
            "left_right_least_m": lr,
            "left_right_pair": {e: sw["left_right"][e]["least_pair"] for e in lr},
            "left_right_pose_deg": {e: sw["left_right"][e]["least_pose_deg"] for e in lr},
            "linkage": {e: v["linkage"] for e, v in lay["info"]["ends"].items()},
            "linkage_sized": sized,
            "axle": {e: v["axle"] for e, v in lay["info"]["ends"].items()},
            "half_width_m": max(e["half_width_m"] for e in lay["info"]["ends"].values()),
            "knee_axle_overhang_m": max(e["knee_axle_overhang_m"] for e in lay["info"]["ends"].values()),
            "knee_drive_place": {e: v["knee_drive_place"] for e, v in lay["info"]["ends"].items()},
            "placement_search": lay["info"].get("placement_search"),
            "tail_pitch_servo_aft_m": lay["info"].get("tail_pitch_servo_aft_m"),
            "bay_box_m": lay["info"].get("bay_box_m")}


def validate_as_drawn() -> dict:
    """The desk-layout code on the 60 cm model as drawn reproduces the packaging report's overlaps (sim/results/
    packaging60_2026-11-27/packaging60_report.txt:41-42, :130-135): hind knee drive in its strut 38.70 mm, pelvic
    crossmember through the hind roll drive 41.10 mm, rails into the front / hind roll drives 8.19 / 23.10 mm."""
    lay = layout_parts(0.60, DRIVE_SETS["D0"], "L1", as_drawn=True)
    geo = Geometry(lay)
    geo.set_pose({})
    got = {"hind knee drive in its strut": geo.dist("hind_knee_drive", "hind_upper_strut"),
           "pelvic crossmember through the hind roll drive": geo.dist("cm_hind", "hind_roll_drive"),
           "rail into the front roll drive": geo.dist("fore_roll_drive", "rail"),
           "rail into the hind roll drive": geo.dist("hind_roll_drive", "rail")}
    ref = {"hind knee drive in its strut": -0.03870, "pelvic crossmember through the hind roll drive": -0.04110,
           "rail into the front roll drive": -0.00819, "rail into the hind roll drive": -0.02310}
    return {"measured_m": got, "report_m": ref, "max_abs_err_mm": max(abs(got[k] - ref[k]) for k in ref) * 1000}


# ---------------------------------------------------------------------------------------------------------------- side
BAY_ITEMS = ("jetson_orin_nano_super_kit", "nvme_and_vibration_tray", "gtech_6s_3300_1", "gtech_6s_3300_2",
             "pilz_relay_contactor_watchdog_radio_rx", "power_board_diode_or_precharge_clamp",
             "dc_dc_rails_and_pack_monitor", "teensy41_can_carriers_x2", "io_and_sensor_carriers",
             "audio_and_servo_bus_adapter", "drive_bay_blowers_and_ducts", "reserve_small_parts_est", "trunk_imu")


BAY_STACKS = {"Jetson on its NVMe tray": ("jetson_orin_nano_super_kit", "nvme_and_vibration_tray")}
BAY_ALLOWANCE_M = 0.005   # DESK: 5 mm added to each of an item's three dimensions for wiring, connectors and mounts


def bay_item_boxes(allowance: float = BAY_ALLOWANCE_M) -> dict:
    """The bay items' boxes from the model (fore128.xml geoms; bought parts, so not scaled with size), the Jetson
    stacked on its tray as one item, each dimension + ``allowance``."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(MODEL60))
    dims = {n: 2 * np.asarray(m.geom_size[m.geom(n).id], dtype=float) for n in BAY_ITEMS}
    out = {}
    stacked = {x for v in BAY_STACKS.values() for x in v}
    for name, members in BAY_STACKS.items():
        d = [dims[x] for x in members]
        out[name] = np.array([max(x[0] for x in d), max(x[1] for x in d), sum(x[2] for x in d)])
    for n, d in dims.items():
        if n not in stacked:
            out[n] = d
    return {n: tuple(float(x) + allowance for x in d) for n, d in out.items()}


def bay_fit(box_m, allowance: float = BAY_ALLOWANCE_M) -> dict:
    """Finding MECH-6 (DESK): can the bay items be packed into the configuration's bay box? A guillotine packer
    (largest item first; each item in any of its six orientations at the lowest, then rearmost, corner of a free box,
    the box then split in three). A greedy packer can miss a packing that exists, so a failure reads 'not shown to
    fit', not 'cannot fit'."""
    import itertools

    items = bay_item_boxes(allowance)
    free = [((0.0, 0.0, 0.0), tuple(float(x) for x in box_m))]
    placed, unplaced = {}, []
    for name, dims in sorted(items.items(), key=lambda kv: -np.prod(kv[1])):
        best = None
        for fi, (o, sz) in enumerate(free):
            for dd in set(itertools.permutations(dims)):
                if all(dd[i] <= sz[i] + 1e-9 for i in range(3)):
                    key = (o[2], o[0], o[1], -np.prod(sz))
                    if best is None or key < best[0]:
                        best = (key, fi, dd)
        if best is None:
            unplaced.append(name)
            continue
        _, fi, dd = best
        o, sz = free.pop(fi)
        placed[name] = {"at_m": o, "size_m": dd}
        free += [((o[0] + dd[0], o[1], o[2]), (sz[0] - dd[0], sz[1], sz[2])),
                 ((o[0], o[1] + dd[1], o[2]), (dd[0], sz[1] - dd[1], sz[2])),
                 ((o[0], o[1], o[2] + dd[2]), (dd[0], dd[1], sz[2] - dd[2]))]
        free = [f for f in free if min(f[1]) > 1e-6]
    vol = sum(np.prod(v) for v in items.values())
    return {"fits": not unplaced, "unplaced": unplaced, "allowance_m": allowance, "box_m": list(box_m),
            "items_volume_over_box": float(vol / np.prod(box_m)), "placed": placed}


def bay_items_volume_m3() -> float:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(MODEL60))
    return float(sum(8 * np.prod(m.geom_size[m.geom(n).id]) for n in BAY_ITEMS))


def side_checks(S: float, dset: dict, m_kg: float, k: float, peak_rpm60: dict, lay_info: dict | None,
                bay_vol: float, bay_vol_60: float | None) -> dict:
    """Reported beside the criteria, not scored: joint speed against each drive's allowance (peak walking rpm at 60 cm
    from the SIM probe x 1 / sqrt(s): at a Froude-matched walk and at the capped one alike), the rod force at the knee
    drive's maker peak (28 mm x s cranks, 43 deg transmission angle; the record sizes the 60 cm front rod for 3,142 N),
    the neck servo's gravity load (0.3083 N m at 60 cm, x s^4) against its continuous figure, and the bay's fill."""
    s = S / S60
    dr = drives_of(dset)
    speed = {a: peak_rpm60[a] / math.sqrt(s) / dr[a]["rpm_allow"] for a in AXES}
    rod = {end: dr[f"{end}_knee"]["peak"] / (crank_radius(s) * math.sin(math.radians(43.0))) for end in ("fore", "hind")}
    neck = 0.3083 * s ** 4
    servo_lim = 3.11 if S / 0.75 > 0.9 else 1.9 / 3
    out = {"speed_ratio_by_axis": speed, "speed_worst": max(speed.values()),
           "rod_force_at_knee_peak_n": rod, "rod_force_over_60cm_sizing": max(rod.values()) / 3142.0,
           "neck_gravity_nm": neck, "neck_servo_continuous_nm": servo_lim, "neck_ratio": neck / servo_lim}
    if lay_info and lay_info.get("bay_box_m"):
        bv = float(np.prod(lay_info["bay_box_m"]))
        out["bay_fill"] = bay_vol / bv
        if bay_vol_60:
            out["bay_fill_over_60cm"] = (bay_vol / bv) / bay_vol_60
        bf = bay_fit(lay_info["bay_box_m"])
        bf0 = bay_fit(lay_info["bay_box_m"], allowance=0.0)
        out["bay_fit"] = {"fits_with_5mm_allowance": bf["fits"], "unplaced_with_5mm_allowance": bf["unplaced"],
                          "fits_with_no_allowance": bf0["fits"], "unplaced_with_no_allowance": bf0["unplaced"],
                          "box_m": bf["box_m"]}
    return out


def stand_torques_60() -> dict:
    """SIM (this script): the adopted model settled 3 s on the floor at the neutral stance (ctrl 0), each leg
    actuator's torque, N m, joint side: the Simulate phase's simplest check of the rule."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(MODEL60))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    d.ctrl[:] = 0.0
    for _ in range(3000):
        mujoco.mj_step(m, d)
    out = {}
    for i in range(m.nu):
        n = m.actuator(i).name
        if any(n.endswith(x) for x in ("_roll", "_pitch", "_flexion")):
            out[n] = float(abs(d.actuator_force[i]))
    grp = {}
    for n, v in out.items():
        g = GROUP_OF_AXIS[axis_of(n)] if not n.endswith("_pitch") else ("fore pitch" if "_fore_" in n else "hind pitch")
        grp[g] = max(grp.get(g, 0.0), v)
    return {"per_actuator_nm": out, "by_group_nm": grp, "mass_kg": float(m.body_mass.sum())}


# ---------------------------------------------------------------------------------------------------------------- study
def group_rms(demo: dict, key="walk_0.25") -> dict:
    seg = demo["segments"][key]
    out = {}
    for jn, v in zip(seg["joints"], seg["tau_drive_rms"]):
        g = GROUP_OF_AXIS[axis_of(jn)] if not jn.endswith("_pitch") else ("fore pitch" if "_fore_" in jn else
                                                                          "hind pitch")
        out[g] = max(out.get(g, 0.0), float(v))
    return out


def probe_summary(probe: dict | None) -> dict:
    """From the SIM probe: per walker at x1.0 walking, the group RMS (joint side) and peak joint rpm."""
    if probe is None:
        return {}
    out = {}
    for w, rec in probe["walkers"].items():
        c = rec["cases"]["x1 walk"]
        rms = np.max([e["rms_nm"] for e in c["episodes"]], axis=0)
        rpm = np.max([e["peak_rpm"] for e in c["episodes"]], axis=0)
        grp, grpm = {}, {}
        for jn, v, r in zip(rec["joints"], rms, rpm):
            g = GROUP_OF_AXIS[axis_of(jn)] if not jn.endswith("_pitch") else ("fore pitch" if "_fore_" in jn else
                                                                              "hind pitch")
            grp[g] = max(grp.get(g, 0.0), float(v))
            grpm[axis_of(jn)] = max(grpm.get(axis_of(jn), 0.0), float(r))
        out[w] = {"rms_joint_nm": grp, "peak_rpm_by_axis": grpm,
                  "speed_mps": float(np.mean([e["speed_mps"] for e in c["episodes"]]))}
    return out


def evaluate(S: float, dkey: str, layout: str, demos: dict, need: dict, peak_rpm60: dict, geo_cache: dict,
             bay_vol: float, bay_vol_60: float | None, ledger: dict, tc_s: float = SCORED_TC_S,
             cap: float = 0.25, mass_exp: float = 1.0, struct_by_load: bool = True, even_heat: bool = False,
             sim_checks: dict | None = None) -> dict:
    from telemetry.thermal import tc_key

    dset = DRIVE_SETS[dkey]
    mm = mass_model(S, dset, layout, bms=True, ledger=ledger, struct_by_load=struct_by_load)
    m = mm["total_kg"]
    k = (m / M60_SIM_KG) ** mass_exp * (S / S60)
    per = {}
    for sd, demo in demos.items():
        prof = config_profile(demo, S, k, dset, cap)
        en = energy(demo, S, k, m, dset, cap)
        ht = heat(demo, prof["walk_sq"], prof["stand_sq"], dset, en["runtime_min"])
        ev = even_split(prof)
        per[sd] = {"prof": prof, "energy": en, "heat": ht, "even": ev, "decide": deciding_and_mirror(prof)}
        if even_heat:
            per[sd]["even_heat"] = heat(demo, ev["walk_sq"], ev["stand_sq"], dset, en["runtime_min"])
    k8 = tc_key(tc_s)
    worst_heat_sd = max(per, key=lambda sd: per[sd]["heat"]["by_tc"][k8]["peak_c"])
    t_peak = per[worst_heat_sd]["heat"]["by_tc"][k8]["peak_c"]
    t_all = [per[sd]["heat"]["by_tc"][kk]["peak_c"] for sd in per for kk in per[sd]["heat"]["by_tc"]]
    c2_by = {sd: float(per[sd]["prof"]["loop_ratio"].max()) for sd in per}
    worst_c2_sd = max(c2_by, key=c2_by.get)
    jw = per[worst_c2_sd]["prof"]["joints"][int(np.argmax(per[worst_c2_sd]["prof"]["loop_ratio"]))]
    pk = peaks(k, dset, need)
    run_by = {sd: per[sd]["energy"]["runtime_min"] for sd in per}
    worst_run_sd = min(run_by, key=run_by.get)
    gkey = (S, dkey, layout)
    if gkey not in geo_cache:
        lay = make_layout(S, dset, layout)
        geo_cache[gkey] = (lay, score_packaging(lay, sweep_layout(lay)))
    lay, c4 = geo_cache[gkey]
    cst = cost(S, dset, layout)
    c7r = c7(S, m, dset)
    side = side_checks(S, dset, m, k, peak_rpm60, lay["info"], bay_vol, bay_vol_60)
    like = all(DRIVES[dset[a]]["like_for_like"] for a in AXES)
    crit = {
        "C1": {"value_c": t_peak, "limit_c": LIMIT_C, "margin_c": LIMIT_C - t_peak, "pass": t_peak <= LIMIT_C,
               "with_margin_80c": t_peak <= MARGIN_C, "worst_policy": worst_heat_sd,
               "hottest_joint": per[worst_heat_sd]["heat"]["by_tc"][k8]["joint"],
               "range_4_15min_c": [min(t_all), max(t_all)],
               "pass_at_all_time_constants": max(t_all) <= LIMIT_C,
               "by_policy_c": {sd: per[sd]["heat"]["by_tc"][k8]["peak_c"] for sd in per},
               "settled_no_break_c": per[worst_heat_sd]["heat"]["by_tc"][k8]["settled_cycle_max_c"],
               "comparable": like},
        "C2": {"value": c2_by[worst_c2_sd], "limit": C2_LIMIT, "margin": C2_LIMIT - c2_by[worst_c2_sd],
               "pass": c2_by[worst_c2_sd] <= C2_LIMIT, "worst_policy": worst_c2_sd, "joint": jw,
               "by_policy": c2_by, "comparable": like,
               "deciding_drive_and_mirror_by_policy": {sd: per[sd]["decide"] for sd in per},
               "even_split_by_policy_desk": {sd: float(per[sd]["even"]["loop_ratio"].max()) for sd in per},
               "even_split_worst_desk": max(float(per[sd]["even"]["loop_ratio"].max()) for sd in per)},
        "C3": {"value": pk["worst"], "limit": C3_LIMIT, "margin": C3_LIMIT - pk["worst"], "pass": pk["worst"] <= C3_LIMIT,
               "axis": pk["worst_axis"], "ratio_by_axis": pk["ratio_by_axis"],
               "supporting_need_worst": pk.get("supporting_need_worst"),
               "x6_peak_45nm_worst": pk["x6_peak_45nm_worst"], "x6_peak_45nm_axis": pk["x6_peak_45nm_axis"],
               "basis": "the model's clamp (0.70 x maker peak), which every stored and probed simulated peak sits at, "
                        "x k"},
        "C4": {"value_m": c4["least_clearance_m"], "limit_m": C4_GAP_M, "margin_m": c4["margin_m"], "pass": c4["pass"],
               "pass_as_written": c4["pass"],
               "pass_except_9_and_11_changed_after_a_result": c4["pass_except_9_and_11"],
               **{kk: c4[kk] for kk in ("worst_pair", "worst_pair_pose_deg", "pairs_under_5mm", "n_pairs_under_5mm",
                                        "next_tightest_pair", "next_tightest_m", "next_tightest_pose_deg",
                                        "tail_pitch_servo_aft_m", "bay_box_m",
                                        "known_clashes", "leg_to_leg_least_m", "leg_to_leg_pair",
                                        "leg_to_leg_pose_deg", "left_right_least_m", "left_right_pair",
                                        "left_right_pose_deg", "linkage_sized", "tail_least_m",
                                        "roll_drive_pair_gap_m")},
               "linkage_thread": {e: v["thread"] for e, v in c4["linkage"].items()},
               "rod_force_static_n": {e: v["rod_force_n"] for e, v in c4["linkage"].items()},
               "m8_rod_sf": {e: {kk: v["by_thread"]["M8"][kk] for kk in ("sf_tension", "sf_compression", "sf_rod_end",
                                                                         "sf_rod_end_lowest_found")}
                             for e, v in c4["linkage"].items()},
               "axle": c4["axle"]},
        "C5": {"value_min": run_by[worst_run_sd], "limit_min": C5_MIN, "margin_min": run_by[worst_run_sd] - C5_MIN,
               "pass": run_by[worst_run_sd] >= C5_MIN, "worst_policy": worst_run_sd, "by_policy_min": run_by,
               "mean_total_w": per[worst_run_sd]["energy"]["mean_total_w"],
               "pessimistic_min_desk": min(per[sd]["energy"]["runtime_min_pessimistic"] for sd in per),
               "peak_pack_a_est": max(per[sd]["energy"]["peak_pack_a_est"] for sd in per)},
        "C6": cst,
        "C7": c7r,
    }
    # relative margins, for ranking and the binding criterion: C1 as the share of the allowed rise (90 - 35 C) left;
    # C4 on the supporting reading (except #9 and #11), on the least clearance other than the rod-to-drive-face gap,
    # which is 5.5 mm by construction in every layout
    c4_sup = c4["pass_except_9_and_11"]
    c4_free = min(c4["next_tightest_m"], c4["tail_least_m"], c4["roll_drive_pair_gap_m"],
                  min(c4["left_right_least_m"].values()))
    rel = {"C1": (LIMIT_C - t_peak) / (LIMIT_C - ROOM_C), "C2": crit["C2"]["margin"] / C2_LIMIT,
           "C3": crit["C3"]["margin"] / C3_LIMIT, "C4": (c4_free - C4_GAP_M) / C4_GAP_M if c4_sup else
           (min(c4["margin_m"], c4_free - C4_GAP_M)) / C4_GAP_M, "C5": crit["C5"]["margin_min"] / C5_MIN}
    others = all(crit[c]["pass"] for c in ("C1", "C2", "C3", "C5"))
    passes = others and c4["pass"]                       # C1-C5 as pre-registered (C4 as written)
    passes_sup = others and c4_sup                       # SUPPORTING: C4 except #9 and #11 (changed after a result)
    sup = pk.get("supporting_need_worst")
    passes_sup_c3 = (sup is not None and sup <= C3_LIMIT and c4_sup and
                     all(crit[c]["pass"] for c in ("C1", "C2", "C5")))
    binding = min(rel, key=rel.get)
    return {"size_m": S, "drive_set": dkey, "drive_set_name": dset["name"], "layout": layout, "cooling": "passive",
            "owner_decision": dset["owner_decision"] or S != S60, "changes_drawn_model": True,
            "thermal_like_for_like": like, "mass_kg": m, "mass_by_class_kg": mm["by_class_kg"], "k": k,
            "struct_load_factor": mm.get("struct_load_factor", 1.0),
            "walk_mps": per[SEEDS[0]]["prof"]["v_mps"], "criteria": crit, "relative_margins": rel,
            "binding": binding, "passes_C1_C5": passes, "passes_C1_C5_like_for_like": passes and like,
            "passes_C1_C3_C5_and_C4_except_9_11": passes_sup,
            "passes_C1_C3_C5_and_C4_except_9_11_like_for_like": passes_sup and like,
            "passes_with_supporting_C3_not_the_criterion": passes_sup_c3 and like,
            "rule_check_in_simulation": rule_check_for(S, dkey, layout, sim_checks),
            "even_split_heat_c": ({sd: per[sd]["even_heat"]["by_tc"][k8]["peak_c"] for sd in per} if even_heat
                                  else None),
            "side_checks": side, "half_width_m": c4["half_width_m"], "knee_axle_overhang_m": c4["knee_axle_overhang_m"],
            "knee_drive_place": c4["knee_drive_place"]}


SIM_COMPARE = RESULTS / "scaled_trike_compare.json"


def load_sim_checks() -> dict | None:
    """S2 and the slow-gait count per scaled-trike case (sim/build_scaled_trike.py compare), or None."""
    if not SIM_COMPARE.exists():
        return None
    d = json.loads(SIM_COMPARE.read_text())
    return {n: {"S2": c["trot"]["S2_holds"], "S1": c["stand"]["S1_pass"],
                "slow_seeds_of_10": c["trot"]["slow_mode_10_20s_all_seeds"], "kp_rule": c["kp_rule"],
                "layout": c.get("layout"), "drives": c["drives"], "S": c["S"]} for n, c in d["cases"].items()}


def rule_check_for(S: float, dkey: str, layout: str, sim: dict | None) -> dict:
    """Whether the torque rule was checked in simulation (S2, the scripted trot) for this configuration's drive set
    at this size in the as-built geometry, and in this layout's geometry where that case was run (findings PH-1,
    CTRL-7, MECH-2). 'checked' needs S2 to hold on the drive-set case and, where run, on the layout case; a layout
    not run is 'layout not simulated'."""
    if sim is None:
        return {"status": "no simulation results"}
    base = f"H{S:.2f}" if dkey == "D0" else f"H{S:.2f}_{dkey}"
    lay = f"H{S:.2f}_{dkey}_{layout}"
    b, lc = sim.get(base), sim.get(lay)
    if b is None:
        st = "drive set not simulated at this size"
    elif not b["S2"]:
        st = "S2 fails for this drive set (as built)"
    elif lc is not None and not lc["S2"]:
        st = f"S2 fails in layout {layout}"
    elif lc is None:
        st = f"S2 holds as built; layout {layout} not simulated"
    else:
        st = "checked: S2 holds as built and in the layout"
    return {"status": st, "drive_set_case": base if b else None, "drive_set_S2": None if b is None else b["S2"],
            "drive_set_slow_of_10": None if b is None else b["slow_seeds_of_10"], "layout_case": lay if lc else None,
            "layout_S2": None if lc is None else lc["S2"],
            "checked": bool(b and b["S2"] and (lc is None or lc["S2"]))}


def simulate_predictions(demos, stand60, probe_sum, ledger) -> dict:
    """The rule's predictions for the Simulate phase (current drive set, model-as-built mass rules without the layout
    additions or the battery cut-off, as a scaled copy of the simulated model would be built): mass, k, walking speed,
    static standing torque by group (k x the 60 cm SIM stand, this script), the scripted trot's and the four policies'
    walking RMS by group (k x 60 cm), peak joint speed (x 1 / sqrt(s))."""
    out = {}
    for S in (0.50, 0.70, 0.80):
        mm = mass_model(S, DRIVE_SETS["D0"], layout=None, bms=False, ledger=ledger)
        k = torque_factor(mm["total_kg"], S)
        v, v60 = walk_speed(S)
        pol = {sd: {g: k * x for g, x in group_rms(d, "walk_0.25").items()} for sd, d in demos.items()}
        row = {"mass_kg": mm["total_kg"], "k": k, "walk_mps_froude_capped": v, "walk_mps_froude_uncapped":
               0.25 * math.sqrt(S / S60), "v60_equiv_mps": v60,
               "stand_by_group_nm": {g: k * x for g, x in stand60["by_group_nm"].items()},
               "policies_walk_rms_drive_side_nm_at_froude_uncapped": pol}
        if probe_sum:
            tr = probe_sum.get("scripted trot")
            if tr:
                row["scripted_trot_walk_rms_joint_nm_at_froude_uncapped"] = {g: k * x for g, x in
                                                                             tr["rms_joint_nm"].items()}
                row["scripted_trot_peak_rpm_by_axis"] = {a: x / math.sqrt(S / S60) for a, x in
                                                         tr["peak_rpm_by_axis"].items()}
        out[f"{S:.2f}"] = row
    return out


def rule_check_75() -> dict:
    """The rule against the 75 cm design's simulated standing torques: T15 at 15.00 kg holds 5.7 N m (front elbow) and
    4.7 N m (hind knee) (DECISION_LOG.md (F4), F4); the 60 cm 188 mm model at 12.69 kg holds 4.0 and 3.2 N m (:54, F10).
    The 188 mm 60 cm model is PR #669's geometry x 0.8, so the pair is geometrically similar."""
    k = torque_factor(15.00, 0.75, m_ref=12.69, s_ref=0.60)
    pred = {"front elbow": 4.0 * k, "hind knee": 3.2 * k}
    got = {"front elbow": 5.7, "hind knee": 4.7}
    return {"k": k, "predicted_nm": pred, "simulated_nm": got,
            "error": {g: pred[g] / got[g] - 1 for g in pred},
            "note": "SIM standing figures; the rule over-predicts by the error shown (conservative)."}


def mass_check_75(ledger) -> dict:
    """The mass rules at 75 cm with the T15's drives (8 X4 + 4 X6) against the T15 ledger (14.999 kg closed): an
    orientation check of the exponents. The parts lists differ (T15 has foot load cells and an OAK-D Pro W, no limb
    covers, one tail servo, a 0.39 kg reserve), so the difference is not an error of the rule alone."""
    t15 = dict(DRIVE_SETS["D0"], fore_knee="X6-60", hind_knee="X6-60")
    mm = mass_model(0.75, t15, layout=None, bms=False, ledger=ledger)
    ref = json.loads(LEDGER75.read_text())["total_mass_kg"] if LEDGER75.exists() else None
    return {"predicted_kg": mm["total_kg"], "t15_ledger_kg": ref,
            "difference": None if ref is None else mm["total_kg"] / ref - 1}


def slim(res: dict) -> dict:
    """The committed JSON (under 500 KB): per configuration, the criteria's values and margins without the repeated
    reference tables (published C7 figures and the known-clash labels are given once, at the top)."""
    out = dict(res)
    out["known_clash_labels"] = {n: v[0] for n, v in KNOWN_CLASHES.items()}
    out["config_notes"] = {"C3": "value = the model's clamp (0.70 x maker peak), which every stored and probed simulated "
                                 "peak sits at, x k, over the new drive's maker peak; supporting_need_worst = the same x "
                                 "the lowest force-limit factor at which the 60 cm walkers still walked and took 0.5 m/s "
                                 "pushes (not the criterion)",
                           "mass": "per class in the full run (form_factor_study.mass_model); summary here",
                           "C7": "continuous only here (and the hind knee's peak at the clamp); peaks in published_c7 "
                                 "and in the full run",
                           "C4": "pass = C4 as written; pass_except_9_and_11_changed_after_a_result = the supporting "
                                 "reading; known_clashes n: [least distance m, removed]; axle end: [bending at the "
                                 "static rod case on the model's 12.8 mm axle MPa, solid diameter for SF 2.0 on 580 MPa "
                                 "m, thigh twisting moment N m]; the layout draws each axle at the larger of 12.8 mm and "
                                 "that diameter (revised after the third review, R3-1); left_right_least_m 1.0 = no pair "
                                 "within 30 mm (not measured closer)",
                           "limits": "limits and margins are the pre-registered lines (C1 90 C, C2 and C3 0.80, C4 5 mm, "
                                     "C5 30 min) and are not repeated per configuration"}
    out["known_clash_notes"] = {"1 (L2)": "removed by the layout: the knee drive sits at the hip, bolted to the thigh's "
                                          "root plate", "11": "not modelled here (the record's fix: a clevis)",
                                "12": "removed by construction: each knee axle runs through its output crank's hub"}
    cs = []
    for c in res["configs"]:
        c = json.loads(json.dumps(_clean(c)))
        pos = c["criteria"]["C7"]["position"]
        for k in ("hip_published", "knee_published", "peak_hip_published_range", "peak_knee_published_range"):
            pos.pop(k, None)
        for kc in c["criteria"]["C4"]["known_clashes"].values():
            kc.pop("label", None)
            kc.pop("how", None)
        mb = c.pop("mass_by_class_kg")
        c["mass_summary_kg"] = {"drives": round(sum(v for k, v in mb.items() if k.startswith("drive:")), 3),
                                "scaled_with_size": round(sum(v for k, v in mb.items() if k.startswith("scales")), 3),
                                "fixed_and_drive_sized": round(sum(v for k, v in mb.items()
                                                                   if not k.startswith(("drive:", "scales"))), 3)}
        c.pop("drive_set_name", None)
        c["criteria"]["C3"].pop("basis", None)
        for k in [k for k in c["criteria"]["C7"] if k.endswith("_peak_clamp")]:
            c["criteria"]["C7"].pop(k)
        for e, v in c["knee_drive_place"].items():
            c["knee_drive_place"][e] = {k: v.get(k) for k in ("t", "side", "d_off", "min_clear_m", "dist_from_hip_m",
                                                               "rod_len_m")}
        c["criteria"]["C6"].pop("landed_usd", None)
        c4 = c["criteria"]["C4"]
        for k in ("worst_pair_pose_deg", "next_tightest_pose_deg"):
            c4.pop(k, None)
        rc = c["rule_check_in_simulation"]
        c["rule_check_in_simulation"] = {k: rc.get(k) for k in ("status", "checked")}
        # under 500 KB: the per-pose and per-policy detail is in the full run (``run_study``), not committed. Axle:
        # [bending at the static case MPa, diameter for SF 2.0 m, thigh twisting moment N m]; known clashes:
        # [least distance m, removed]; limits and margins follow from the pre-registered lines (section 1.1)
        c4["axle"] = {e: [v["bending_static_mpa"], v["axle_d_for_sf2_m"], v["thigh_moment_static_nm"]]
                      for e, v in c4["axle"].items()}
        c4["m8_rod_sf_least"] = {e: min(v.values()) for e, v in c4.pop("m8_rod_sf").items()}
        c4["known_clashes"] = {n: [v.get("min_m"), v.get("removed")] for n, v in c4["known_clashes"].items()}
        for k in ("leg_to_leg_pose_deg", "left_right_pair", "left_right_pose_deg", "rod_force_static_n",
                  "pairs_under_5mm", "bay_box_m"):
            c4.pop(k, None)
        for x in ("C1", "C2", "C3", "C4", "C5"):
            for f in ("limit", "limit_c", "limit_m", "limit_min", "margin", "margin_c", "margin_m", "margin_min",
                      "comparable"):
                c["criteria"][x].pop(f, None)
        for f in ("drive_counts", "landed_41pct_with_bms_usd"):
            c["criteria"]["C6"].pop(f, None)
        for f in ("cooling", "changes_drawn_model"):
            c.pop(f, None)
        c2 = c["criteria"]["C2"]
        c2["deciding_drive_and_mirror_by_policy"] = {
            sd: [v["joint"], v["ratio"], v["mirror_ratio"]] for sd, v in c2["deciding_drive_and_mirror_by_policy"].items()}
        c2.pop("even_split_by_policy_desk", None)
        c7 = c["criteria"]["C7"]
        c["criteria"]["C7"] = {"position": {k: v for k, v in c7["position"].items()
                                            if k in ("hip_cont_worst", "knee_cont_worst", "hip_where", "knee_where",
                                                     "hip_cont_worst_operating", "knee_cont_worst_operating",
                                                     "hind_knee_peak_clamp")}}
        c.pop("knee_drive_place", None)
        sc = c["side_checks"]
        sc.pop("rod_force_at_knee_peak_n", None)
        sc["speed_ratio_by_axis"] = {"pitch": sc["speed_ratio_by_axis"]["pitch"]}
        bf = sc.get("bay_fit")
        if bf:
            sc["bay_fit"] = {"fits_with_5mm_allowance": bf["fits_with_5mm_allowance"],
                             "fits_with_no_allowance": bf["fits_with_no_allowance"],
                             "n_unplaced_with_5mm_allowance": len(bf["unplaced_with_5mm_allowance"])}
        c1 = c["criteria"]["C1"]
        c1.pop("settled_no_break_c", None)
        cs.append(_clean(c, 4))
    out["configs"] = cs
    return out


def _clean(x, nd: int = 5):
    if isinstance(x, dict):
        return {str(k): _clean(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v, nd) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _clean(x.tolist(), nd)
    return x


@contextlib.contextmanager
def _eq_at_rated_speed():
    """Sensitivity: every drive's continuous rating taken at its rated speed (X4 7.5, X6 9, X8 20 N m; the maker
    tables' lowest figures) instead of the build's values (X4 7.5 at 83 rpm, X6 12 and X8 34 at 0 rpm)."""
    old = {k: d["eq"] for k, d in DRIVES.items()}
    try:
        for d in DRIVES.values():
            d["eq"] = d["eq_rated_speed"]
        yield
    finally:
        for k, v in old.items():
            DRIVES[k]["eq"] = v


@contextlib.contextmanager
def _x4_row(row: str):
    """Sensitivity (finding PH-2): the X4-36-L's rating and thermal fit from one of its own maker rows (``X4_ROWS``:
    torque, start and end C over 90 min) instead of the build's 7.5 N m from 25 to 100 C. C2 then uses the row's
    torque; C1 the drive refitted to the row's own rise."""
    d = DRIVES["X4-36-L"]
    old = {k: d.get(k) for k in ("eq", "fit_start_c", "fit_end_c")}
    try:
        d["eq"], d["fit_start_c"], d["fit_end_c"] = X4_ROWS[row]
        yield
    finally:
        for k, v in old.items():
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v


def score_as_drawn(demos: dict) -> dict:
    """The 60 cm design as drawn (12.751 kg, D0, k = 1, each demo's own length; the stored figures): C1 and C2 per
    policy, each policy's deciding drive with its mirror, the even split (DESK arithmetic, not a pass), and the X4
    maker-row sensitivity (C1, C2 worst over the four policies)."""
    from telemetry.thermal import tc_key

    k8 = tc_key(SCORED_TC_S)
    D0 = DRIVE_SETS["D0"]
    out = {"C1_by_policy_c": {}, "C2_by_policy": {}, "deciding_and_mirror": {}, "even_split_C2": {},
           "even_split_C1_c": {}, "x4_rows": {}}
    for sd, d in demos.items():
        p = config_profile(d, 0.60, 1.0, D0)
        h = heat(d, p["walk_sq"], p["stand_sq"], D0, d["minutes"])
        ev = even_split(p)
        he = heat(d, ev["walk_sq"], ev["stand_sq"], D0, d["minutes"])
        out["C1_by_policy_c"][sd] = h["by_tc"][k8]["peak_c"]
        out["C2_by_policy"][sd] = float(p["loop_ratio"].max())
        out["deciding_and_mirror"][sd] = deciding_and_mirror(p)
        out["even_split_C2"][sd] = float(ev["loop_ratio"].max())
        out["even_split_C1_c"][sd] = he["by_tc"][k8]["peak_c"]
    for row in ("10 rpm", "30 rpm"):
        with _x4_row(row):
            c1, c2 = [], []
            for sd, d in demos.items():
                p = config_profile(d, 0.60, 1.0, D0)
                c1.append(heat(d, p["walk_sq"], p["stand_sq"], D0, d["minutes"])["by_tc"][k8]["peak_c"])
                c2.append(float(p["loop_ratio"].max()))
            out["x4_rows"][row] = {"C1_c_worst": max(c1), "C2_worst": max(c2), "X4_rating_nm": X4_ROWS[row][0]}
    out["C1_worst_c"] = max(out["C1_by_policy_c"].values())
    out["C2_worst"] = max(out["C2_by_policy"].values())
    out["even_split_C2_worst"] = max(out["even_split_C2"].values())
    out["even_split_C1_worst_c"] = max(out["even_split_C1_c"].values())
    return out


def run_study(sizes=SIZES_M, dsets=tuple(DRIVE_SETS), layouts=tuple(LAYOUTS), quick: bool = False) -> dict:
    t0 = time.time()
    ledger = load_ledger60()
    demos = load_demos()
    probe = json.loads(PEAKS_JSON.read_text()) if PEAKS_JSON.exists() else None
    need = peak_need_factor(probe)
    psum = probe_summary(probe)
    peak_rpm60 = {a: max(v["peak_rpm_by_axis"][a] for w, v in psum.items() if w.startswith("policy"))
                  for a in AXES} if psum else {"roll": 70.0, "pitch": 97.0, "fore_knee": 97.0, "hind_knee": 70.0}
    bay_vol = bay_items_volume_m3()
    lay60 = layout_parts(0.60, DRIVE_SETS["D0"], "L1")
    bay_vol_60 = bay_vol / float(np.prod(lay60["info"]["bay_box_m"]))
    geo_cache: dict = {}
    sim = load_sim_checks()
    configs = []
    for S in sizes:
        for dk in dsets:
            for lo in layouts:
                configs.append(evaluate(S, dk, lo, demos, need, peak_rpm60, geo_cache, bay_vol, bay_vol_60, ledger,
                                        even_heat=(dk == "D0"), sim_checks=sim))
        print(f"size {S:.2f} m done ({time.time() - t0:.0f} s)", flush=True)
    SUP = "passes_C1_C3_C5_and_C4_except_9_11"

    def brief_sens(r):
        return {"C1_c": r["criteria"]["C1"]["value_c"], "C2": r["criteria"]["C2"]["value"],
                "C3": r["criteria"]["C3"]["value"], "passes_C1_C5": r["passes_C1_C5"], SUP: r[SUP]}

    sens = {}
    if not quick:
        for S in sizes:
            for label, kw in (("uncapped Froude speed", {"cap": 99.0}),
                              ("walk at 0.25 m/s at every size (DESK: the 0.15-0.25 m/s line extended)", {"cap": -0.25}),
                              ("torque ~ mass^0.91 x s (SIM mass slope, largest seed)", {"mass_exp": 0.91}),
                              ("scored at the 4 min time constant", {"tc_s": 240.0})):
                EXTRAPOLATE["on"] = kw.get("cap") == -0.25
                try:
                    r = evaluate(S, "D0", "L1", demos, need, peak_rpm60, geo_cache, bay_vol, bay_vol_60, ledger,
                                 sim_checks=sim, **kw)
                finally:
                    EXTRAPOLATE["on"] = False
                sens.setdefault(f"{S:.2f}", {})[label] = brief_sens(r)
            for row in ("10 rpm", "30 rpm"):
                with _x4_row(row):
                    r = evaluate(S, "D0", "L1", demos, need, peak_rpm60, geo_cache, bay_vol, bay_vol_60, ledger,
                                 sim_checks=sim)
                sens[f"{S:.2f}"][f"X4 rated on its {row} maker row ({X4_ROWS[row][0]:g} N m, refitted)"] = brief_sens(r)
    eq_rated, x4_rows, struct_sens = {}, {}, {}
    sup_like = [c for c in configs if c[SUP + "_like_for_like"]]
    with _eq_at_rated_speed():
        for c in sup_like:
            r = evaluate(c["size_m"], c["drive_set"], c["layout"], demos, need, peak_rpm60, geo_cache, bay_vol,
                         bay_vol_60, ledger, sim_checks=sim)
            eq_rated[f"{c['size_m']:.2f} {c['drive_set']} {c['layout']}"] = brief_sens(r)
    for row in ("10 rpm", "30 rpm"):
        with _x4_row(row):
            for c in sup_like:
                r = evaluate(c["size_m"], c["drive_set"], c["layout"], demos, need, peak_rpm60, geo_cache, bay_vol,
                             bay_vol_60, ledger, sim_checks=sim)
                x4_rows.setdefault(f"{c['size_m']:.2f} {c['drive_set']} {c['layout']}", {})[row] = brief_sens(r)
    for c in sup_like:
        if c["drive_set"] == "D0":
            continue
        for label, kw in (("structure by the model's rule only (s^2.5, as before the review)", {"struct_by_load": False}),):
            r = evaluate(c["size_m"], c["drive_set"], c["layout"], demos, need, peak_rpm60, geo_cache, bay_vol,
                         bay_vol_60, ledger, sim_checks=sim, **kw)
            struct_sens[f"{c['size_m']:.2f} {c['drive_set']} {c['layout']}"] = {
                **brief_sens(r), "mass_kg": r["mass_kg"], "k": r["k"]}
    passing = list(sup_like)
    passing.sort(key=lambda c: (-min(c["relative_margins"][x] for x in ("C1", "C2", "C3")),
                                c["criteria"]["C6"]["landed_0pct_usd"]))
    best_by_size, binding_by_size = {}, {}

    def brief(c):
        return {"config": f"{c['drive_set']} {c['layout']}", "drive_set_name": c["drive_set_name"],
                "passes_C1_C5": c["passes_C1_C5"], SUP: c[SUP], "binding": c["binding"],
                "relative_margins": c["relative_margins"], "rule_check": c["rule_check_in_simulation"]["status"],
                "mass_kg": c["mass_kg"], "landed_0pct_usd": c["criteria"]["C6"]["landed_0pct_usd"],
                "landed_41pct_usd": c["criteria"]["C6"]["landed_41pct_usd"],
                "landed_0pct_with_bms_usd": c["criteria"]["C6"]["landed_0pct_with_bms_usd"]}

    for S in sizes:
        here = [c for c in configs if abs(c["size_m"] - S) < 1e-9]
        ok = [c for c in here if c[SUP + "_like_for_like"]]
        phys = lambda c: min(c["relative_margins"][x] for x in ("C1", "C2", "C3"))  # noqa: E731
        if ok:
            most = max(ok, key=lambda c: (phys(c), -c["criteria"]["C6"]["landed_0pct_usd"]))
            cheap = min(ok, key=lambda c: c["criteria"]["C6"]["landed_0pct_usd"])
        else:
            most = cheap = max(here, key=lambda c: (min(c["relative_margins"].values()),
                                                    -c["criteria"]["C6"]["landed_0pct_usd"]))
        sup = [c for c in here if c["passes_with_supporting_C3_not_the_criterion"]]
        best_by_size[f"{S:.2f}"] = {"any_pass_as_written": any(c["passes_C1_C5"] for c in here),
                                    "any_pass": bool(ok), "pass_reading": SUP, "most_margin": brief(most),
                                    "cheapest_pass": brief(cheap), "n_pass": len(ok),
                                    "cheapest_with_supporting_C3_not_the_criterion":
                                        brief(min(sup, key=lambda c: c["criteria"]["C6"]["landed_0pct_usd"]))
                                        if sup else None}
        cur = [c for c in here if c["drive_set"] == "D0" and c["layout"] == "L1"][0]
        binding_by_size[f"{S:.2f}"] = {"current_drives_L1": {"binding": cur["binding"],
                                                             "fails": [x for x in ("C1", "C2", "C3", "C4", "C5")
                                                                       if not cur["criteria"][x]["pass"]],
                                                             "fails_with_C4_except_9_11":
                                                                 [x for x in ("C1", "C2", "C3", "C5")
                                                                  if not cur["criteria"][x]["pass"]] +
                                                                 ([] if cur["criteria"]["C4"][
                                                                     "pass_except_9_and_11_changed_after_a_result"]
                                                                  else ["C4"]),
                                                             "relative_margins": cur["relative_margins"]},
                                       "most_margin": {"binding": most["binding"],
                                                       "config": f"{most['drive_set']} {most['layout']}"},
                                       "cheapest_pass": {"binding": cheap["binding"],
                                                         "config": f"{cheap['drive_set']} {cheap['layout']}"}}
    res = {
        "about": {"date": "2026-11-28", "kind": "DESK model on SIMULATED torques; nothing measured; not a decision",
                  "criteria_fixed": "27 Nov 2026 by the lead, before scoring (the study brief)",
                  "scored_time_constant_s": SCORED_TC_S, "room_c": ROOM_C, "method": "A (thermal_ambient.py)",
                  "torque_rule": "k = (m / 12.751 kg) x (S / 0.60 m); walk at min(0.25, 0.25 sqrt(S / 0.60)) m/s; no "
                                 "layout term; checked in simulation (S2) only for D0 as built and in L1",
                  "inputs": {sd: {"path": d["_path"], "sha256": d["_sha256"]} for sd, d in demos.items()},
                  "probe": str(PEAKS_JSON.relative_to(BUILD)) if probe else None,
                  "drives": {k: {kk: v for kk, v in d.items() if kk != "km"} | {"km": d["km"]}
                             for k, d in DRIVES.items()},
                  "drive_sets": DRIVE_SETS, "layouts": LAYOUTS,
                  "cooling": "passive only: no fact sheet gives a sourced forced-air thermal resistance"},
        "mass_split_60cm_kg": mass_split_60(ledger),
        "rule_check_75cm": rule_check_75(), "mass_check_75cm": mass_check_75(ledger),
        "packaging_validation_as_drawn": validate_as_drawn(),
        "stand60_sim": stand_torques_60(), "probe_summary": psum, "peak_need": need,
        "published_c7": published_c7(),
        "simulate_predictions": None, "sensitivity_current_drives_L1": sens,
        "configs": configs, "ranked_passing": [f"{c['size_m']:.2f} {c['drive_set']} {c['layout']}" for c in passing],
        "ranked_passing_reading": SUP + " (supporting; no configuration passes C1-C5 with C4 as written)",
        "passing_as_written": [f"{c['size_m']:.2f} {c['drive_set']} {c['layout']}" for c in configs if c["passes_C1_C5"]],
        "passing_not_comparable": [f"{c['size_m']:.2f} {c['drive_set']} {c['layout']}" for c in configs
                                   if c[SUP] and not c["thermal_like_for_like"]],
        "sensitivity_eq_at_rated_speed_for_passing": eq_rated,
        "sensitivity_x4_maker_rows_for_passing": x4_rows,
        "sensitivity_structure_rule_for_passing": struct_sens,
        "as_drawn_60cm": score_as_drawn(demos),
        "pass_rule_note": "passes_C1_C5 is the pre-registered rule with C4 as written (no configuration passes: no layout "
                          "removes known clash #9, and #11 is not modelled); passes_C1_C3_C5_and_C4_except_9_11 is a "
                          "SUPPORTING reading, changed after a result; ranked_passing lists it",
        "best_by_size": best_by_size, "binding_by_size": binding_by_size,
        "runtime_s": time.time() - t0,
    }
    res["simulate_predictions"] = simulate_predictions(demos, res["stand60_sim"], psum, ledger)
    return res


# ---------------------------------------------------------------------------------------------------------------- table
def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def markdown(res: dict) -> str:
    L = ["# Form-factor study, 50-80 cm: configuration table", "",
         "Generated by `design/form_factor_study.py` (28 Nov 2026). DESK model on SIMULATED 60 cm torques, scaled by "
         "the stated rule; nothing is measured; nothing here is a decision. Worst of the four 128 mm policies; 35 C "
         "room, open air, 8 min time constant (4-15 min range in the JSON). C1 hottest winding (pass <= 90 C), C2 "
         "loop RMS / continuous (<= 0.80), C3 scaled clamp peak / maker peak (<= 0.80), C4 least clearance (>= 5 mm), "
         "C5 runtime (>= 30 min), C6 landed at 0 % / 41.5 % duty (+BMS adds ~$0.68k), C7 continuous tau / (m g l) "
         "hip pitch / knee (worst leg). RS03 rows (D7) rest on heat-sink ratings: not comparable, never counted as a "
         "pass.", "",
         "| Size | Drives | Layout | kg | k | C1 C | C2 | C3 | C3 need* | C4 mm | C5 min | C6 $k | C7 hip / knee | "
         "Passes C1-C5 (C4 as written) | C4 except #9, #11** | Rule checked in simulation | Binding |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["configs"]:
        cr = c["criteria"]
        ok = "yes" if c["passes_C1_C5"] else "no"
        sup = "yes" if c["passes_C1_C3_C5_and_C4_except_9_11"] else "no"
        if c["passes_C1_C3_C5_and_C4_except_9_11"] and not c["thermal_like_for_like"]:
            sup = "not comparable"
        L.append(f"| {c['size_m']:.2f} | {c['drive_set']} | {c['layout']} | {c['mass_kg']:.2f} | {c['k']:.2f} | "
                 f"{cr['C1']['value_c']:.1f} | {cr['C2']['value']:.2f} | {cr['C3']['value']:.2f} | "
                 f"{_f(cr['C3'].get('supporting_need_worst'))} | "
                 f"{cr['C4']['value_m'] * 1000:.1f} | {cr['C5']['value_min']:.0f} | "
                 f"{cr['C6']['landed_0pct_usd'] / 1000:.1f} / {cr['C6']['landed_41pct_usd'] / 1000:.1f} | "
                 f"{cr['C7']['position']['hip_cont_worst']:.3f} / {cr['C7']['position']['knee_cont_worst']:.3f} | {ok} | "
                 f"{sup} | {c['rule_check_in_simulation']['status']} | {c['binding']} |")
    L += ["", "*C3 need: SUPPORTING, not the criterion: the scaled clamp x 0.5, the lowest force-limit factor at which "
          "all four 60 cm policies and the trot still walked and took 0.5 m/s pushes (sim/results/form_factor_2026-11-27/"
          "force_limit_probe60.json); walking and 0.5 m/s pushes only, so it says nothing about a moderate shove.",
          "**SUPPORTING reading, changed after a result: C1-C3, C5 and C4 without known clashes #9 (the front and hind "
          "legs meeting under the body, which no layout here removes) and #11 (the neck, not modelled).", "",
          "## Best at each size (on the supporting reading)", "",
          "| Size | Passes? | Most margin (C1-C3) | Its binding | landed 0 % / 41.5 % | Cheapest pass | landed 0 % / "
          "41.5 % (+BMS 0 %) | Cheapest with the supporting C3 reading |", "|---|---|---|---|---|---|---|---|"]
    for S, b in res["best_by_size"].items():
        m, c, sp = b["most_margin"], b["cheapest_pass"], b["cheapest_with_supporting_C3_not_the_criterion"]
        L.append(f"| {S} | {'yes (' + str(b['n_pass']) + ')' if b['any_pass'] else 'no'} | {m['config']} | "
                 f"{m['binding']} | {m['landed_0pct_usd'] / 1000:.1f} / {m['landed_41pct_usd'] / 1000:.1f} | "
                 f"{c['config'] if b['any_pass'] else '-'} | {c['landed_0pct_usd'] / 1000:.1f} / "
                 f"{c['landed_41pct_usd'] / 1000:.1f} ({c['landed_0pct_with_bms_usd'] / 1000:.1f}) | "
                 f"{(sp['config'] + ' $' + format(sp['landed_0pct_usd'] / 1000, '.1f') + 'k') if sp else '-'} |")
    L += ["", "Drive sets: " + "; ".join(f"{k} {v['name']}" for k, v in res["about"]["drive_sets"].items()) + ".",
          "Layouts: " + "; ".join(f"{k} {v}" for k, v in res["about"]["layouts"].items()) + "."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--out", default=str(OUT_JSON))
    r.add_argument("--table", default=None)
    p = sub.add_parser("peaks")
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out", default=str(PEAKS_JSON))
    a = ap.parse_args(argv)
    if a.cmd == "peaks":
        run_peaks(Path(a.ckpt_root), Path(a.out))
        return 0
    if a.cmd in (None, "run"):
        res = run_study()
        out = Path(getattr(a, "out", OUT_JSON) or OUT_JSON)
        out.write_text(json.dumps(_clean(slim(res)), separators=(",", ":")) + "\n")
        if getattr(a, "table", None):
            Path(a.table).write_text(markdown(_clean(res)))
        print("wrote", out, f"({out.stat().st_size / 1024:.0f} KB)")
    return 0


# ---------------------------------------------------------------------------------------------------------------- probe
FORCE_FACTORS = (1.0, 0.7, 0.5)
PUSH_DV = 0.5           # m/s, free-body velocity change of a 0.1 s push at 8 s (the sweep's form, rl/trike_env.py:232-238)
PUSH_T0_S, PUSH_DUR_S = 8.0, 0.1
PROBE_SEEDS = (810001, 810002)
CONDITIONS = (("walk", None), ("side+", ("sideways", +PUSH_DV)), ("side-", ("sideways", -PUSH_DV)),
              ("fore+", ("fore_aft", +PUSH_DV)), ("back", ("fore_aft", -PUSH_DV)))


def _probe_episode(env, act, seed: int, command: float, push, factor: float, seconds: float = 20.0) -> dict:
    """One episode with every leg actuator's force limit at ``factor`` x the model's own (set after reset, which
    restores the nominal limits, rl/trike_env.py:780), an optional push, every physics step's torque logged."""
    idx = np.asarray(env._leg_act, dtype=np.int64)
    env.cfg = env.cfg.replace(episode_seconds=seconds)
    env.max_episode_steps = int(round(seconds / env.dt))
    obs, _ = env.reset(seed=seed, options={"command_speed": command, "delay_steps": 10})
    m, d = env.model, env.data
    m.actuator_forcerange[idx] *= factor
    lim = m.actuator_forcerange[idx, 1].copy()
    chassis, mass = env._chassis, float(m.body_mass.sum())
    taus, oms, st = [], [], {"f": None, "end": None}

    def hook(e):
        taus.append(d.actuator_force[idx].copy())
        oms.append(d.actuator_velocity[idx].copy())
        if push is not None:
            axis, dv = push
            if st["f"] is None and d.time >= PUSH_T0_S:
                R = d.xmat[chassis].reshape(3, 3)
                v = R[:, 0 if axis == "fore_aft" else 1].copy()
                v[2] = 0.0
                v /= max(float(np.linalg.norm(v)), 1e-9)
                st["f"], st["end"] = v * mass * dv / PUSH_DUR_S, d.time + PUSH_DUR_S
            if st["f"] is not None:
                d.xfrc_applied[chassis, :3] = st["f"] if d.time < st["end"] else 0.0

    env.substep_hook = hook
    fell, speeds = False, []
    try:
        while True:
            obs, _, term, trunc, info = env.step(act(obs))
            if info["time_s"] > 2.0:
                speeds.append(info["forward_speed"])
            if term or trunc:
                fell = bool(term)
                break
    finally:
        env.substep_hook = None
        d.xfrc_applied[:] = 0.0
        m.actuator_forcerange[idx] /= factor
    a = np.abs(np.asarray(taus))
    w = a[2000:] if len(a) > 2000 else a  # from 2 s, as the demo course measures
    om = np.abs(np.asarray(oms))[2000:] if len(oms) > 2000 else np.abs(np.asarray(oms))
    r4 = lambda x: [round(float(v), 4) for v in x]  # noqa: E731
    return {"seed": seed, "fell": fell, "length_s": round(len(taus) * 1e-3, 3),
            "speed_mps": round(float(np.mean(speeds)), 4) if speeds else 0.0,
            "limit_nm": r4(lim), "max_nm": r4(w.max(0)), "p999_nm": r4(np.quantile(w, 0.999, axis=0)),
            "rms_nm": r4(np.sqrt((w ** 2).mean(0))), "share_at_limit": r4((w >= 0.999 * lim).mean(0)),
            "peak_rpm": r4(om.max(0) * 60 / (2 * math.pi))}


def run_peaks(ckpt_root: Path, out: Path = PEAKS_JSON) -> dict:
    """SIMULATION (60 cm adopted model only): the four 128 mm policies (as rl/demo_course.py runs them: nominal model,
    noise on, 10 ms delay) and the frozen model-based trot, at force limits x1.0, x0.7 and x0.5 of the model's, walking
    at 0.25 m/s with and without a 0.5 m/s push (sideways both ways, fore-aft both ways), two seeds each."""
    import torch

    torch.set_num_threads(1)
    import demo_course
    import trike_env as te
    import walk60

    walkers = {}
    for s in SEEDS:
        ck = ckpt_root / f"fore128-{s}" / "checkpoints"
        make, label = demo_course.policy_factory(ck)
        zips = sorted(ck.glob("*.zip"))
        walkers[f"policy {s}"] = (make, label, {z.name: hashlib.sha256(z.read_bytes()).hexdigest() for z in zips})

    def trot_make(command):
        env = te.TrikeEnv(walk60.plant_config(model_path=str(MODEL60), domain_randomization=False,
                                              observation_noise=True))
        return env, walk60.TrotController(env, walk60.WalkParams()).act

    walkers["scripted trot"] = (trot_make, "walk/walk60.py TrotController (frozen) on the 128 mm capsule model", {})
    res = {"simulation_only": True, "model": str(MODEL60.relative_to(BUILD)), "date": "2026-11-28",
           "what": "force-limit probe: falls and torque statistics at x1.0 / x0.7 / x0.5 of the model's force limits",
           "push": {"dv_mps": PUSH_DV, "start_s": PUSH_T0_S, "duration_s": PUSH_DUR_S,
                    "form": "force on the chassis centre of mass along its horizontal left (sideways) or forward "
                            "(fore-aft) axis at onset, impulse = robot mass x dv (as rl/trike_env.py _apply_push)"},
           "force_factors": list(FORCE_FACTORS), "seeds": list(PROBE_SEEDS), "walkers": {}}
    t0 = time.time()
    for name, (make, label, sha) in walkers.items():
        env, act = make(0.25)
        joints = [env.model.actuator(int(i)).name for i in env._leg_act]
        rec = {"label": label, "checkpoint_sha256": sha, "joints": joints, "cases": {}}
        for f in FORCE_FACTORS:
            for cname, push in CONDITIONS:
                eps = [_probe_episode(env, act, sd, 0.25, push, f) for sd in PROBE_SEEDS]
                rec["cases"][f"x{f:g} {cname}"] = {"factor": f, "condition": cname, "episodes": eps,
                                                   "falls": sum(e["fell"] for e in eps)}
                print(f"{name:14s} x{f:g} {cname:6s} falls {sum(e['fell'] for e in eps)}/{len(eps)} "
                      f"({time.time() - t0:.0f} s)", flush=True)
        res["walkers"][name] = rec
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=0, default=float) + "\n")
    return res


if __name__ == "__main__":
    sys.exit(main())
