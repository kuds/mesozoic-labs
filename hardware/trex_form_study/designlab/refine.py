"""Refinement experiments on the round-two leaders (overnight session after the round-two review package).

Each experiment changes one lever on a leading body, re-sizes it with the round-two rules (generator revision 2,
Pi 5 compute, round-two walk accuracy, the packaging search) and screens it exactly like the round-two candidates
(``round2.screen_one``), so its row compares like for like with ``round2/`` and is scored with the round-two
scorecard.  No controller is involved: these are design levers checked with the same prescribed-motion screens.

Levers (``CASES``):

* ``headroom_60``: the 60 cm leaders sized with 20 % headroom on every planning limit and the Froude 0.05 walk as
  a sizing case (``sizing.HEADROOM`` 0.80, ``sizing.EXTRA_WALKS``), as the size study does at 75 and 80 cm.
* ``toes_long``, ``pads_large``, ``foot_both``: digit length x1.15 (override ``toe_scale``), contact-pad footprint
  x1.25 in each direction (``pad_scale``), or both.  A longer and broader foot moves the pads' front edge forward
  and spreads them (one-foot stopping, push recovery, the yaw moment a pad can carry in a turn), at the cost of
  distal mass and of toe-to-toe room in turns.
* ``narrow_3mm``: the narrow stance searched until the target turns keep at least 3 mm of closest approach
  (``round2.TURN_SEARCH_MIN_GAP_MM``), instead of merely no contact: stance width traded for build tolerance.
* ``no_calf``: the shank left uncovered (override ``calf_fairing`` 0; owner question on the calf fairings).  On the
  round-two bodies knee flexion stops where the upper-thigh shell meets the calf fairing (0.81 rad at 60 cm,
  0.76-0.81 rad at 75 cm), and a higher carpet swing needs more knee flexion.

* ``head_x_60``, ``head_x_75`` (Stage 2, ``STAGE2_PLAN.md`` S2-2, owner decision P29): the two picks with the
  expressive head, a neck yaw at the neck base and a hinged jaw, each driven by an STS3215 (``HEAD_X_PART``;
  ``build.NECK_YAW_RANGE``).  Not re-sized: each is its pick's sized record with the two joints and their drives
  added (``head_x_one``), so with the two joints locked it is today's body plus the added parts, and its record holds
  the pick's drives against the expressive head's loads.  Nothing is adopted (P24), and P2's deferral of a jaw for
  the build stands.

* ``jetson_75`` (Stage 2, ``JETSON_PLEM_REVIEW.md`` section 5 task 6, owner decision P30): the 75 cm pick with the
  Jetson Orin Nano Super dev kit as its compute part and a 6S-to-12 V regulator for the dev kit's carrier
  (``JETSON_RULE``; ``build.COMPUTE_REGULATOR_STANDOFF_M``).  Not re-sized, as the expressive head's cases are: the
  pick's sized record with the compute changed and the regulator placed (``jetson_one``); the Teensy control stack
  stays (the owner's baseline: the walking loop on the microcontroller).  Nothing is adopted (P24); P14's Raspberry
  Pi 5 set stays the adopted compute, and no power mode is chosen (``jetson.power_modes`` holds both).

``carpet_lift`` (separate pass, no re-sizing): the carpet swing target needs a higher swing than the reference
walks' 0.10-0.11 H.  For each listed body and each lift, the nominal and brisk walks are regenerated with that
lift and checked against the body's selected drives (sizing rules: continuous = max(nominal RMS, hold),
peak = 1.25 x max(walk peaks, hold), speed = 1.15 x walk peak speed), against the round-two clearance gates on
those walks, and against the carpet screen's toe catch (pile-top clearance and the 0.06 H / 15 mm target).

Run from hardware/trex_form_study:
    python -m designlab.refine --out round2/refine --workers 3 [--cases ...] [--carpet-lift]
Outputs: <out>/<case>/candidates, <out>/<case>/review/..., <out>/carpet_lift.json, <out>/REFINE.md, refine.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

LEADERS_60 = ("60-M-T-10", "60-M-H-10", "60-N-T-10", "60-N-H-10")
#: material and power levers (owner question overnight).  ``cf`` uses the provisional 2 mm carbon floor (the
#: optimistic sensitivity); ``cfqi`` the sourced 3 mm quasi-isotropic plate (hardware.STRUCT_MATERIALS, MATERIALS.md)
MAT_LEVERS = (("cf", (("structure_material", "cf_plate"),), "carbon-fibre frame plates (2 mm floor, optimistic)"),
              ("cfqi", (("structure_material", "cf_qi"),), "carbon-fibre frame plates (3 mm quasi-isotropic, sourced)"),
              ("thin", (("cover_wall_scale", 0.75),), "covering walls x0.75"),
              ("tether", (("power_mode", "tether"),), "tether-only power"),
              ("light", (("structure_material", "cf_plate"), ("cover_wall_scale", 0.75)),
               "carbon plates (2 mm) + covering walls x0.75"),
              ("lightqi", (("structure_material", "cf_qi"), ("cover_wall_scale", 0.75)),
               "carbon plates (3 mm, sourced) + covering walls x0.75"),
              # the research's recommended coverings: MJF PA11 (ductile) with graded 1.4 / 1.2 / 1.0 mm walls, about
              # x0.75 of the 1.6-1.8 mm walls on average
              ("pa11", (("cover_material", "PA11_MJF"), ("cover_wall_scale", 0.75)), "PA11 coverings, walls x0.75"),
              ("cfqi4", (("structure_material", "cf_qi_eqrig"),),
               "carbon-fibre frame plates at equal rigidity (3.99 mm quasi-isotropic)"),
              ("thinal", (("structure_material", "al6061_t254"),), "6061 frame plates at 2.54 mm (brackets 3 mm)"),
              ("best", (("structure_material", "cf_qi"), ("cover_material", "PA11_MJF"), ("cover_wall_scale", 0.75)),
               "carbon plates (3 mm, sourced) + PA11 coverings, walls x0.75"))
CASES = {
    "headroom_60": dict(ids=LEADERS_60, overrides=(), headroom=0.80, fr05=True, min_gap=0.0,
                        label="20 % headroom + Froude 0.05 sizing"),
    "toes_long": dict(ids=("60-M-T-10", "60-M-H-10", "60-N-H-10"), overrides=(("toe_scale", 1.15),), headroom=1.0,
                      fr05=False, min_gap=0.0, label="toes x1.15"),
    "pads_large": dict(ids=("60-M-T-10", "60-M-H-10", "60-N-H-10"), overrides=(("pad_scale", 1.25),), headroom=1.0,
                       fr05=False, min_gap=0.0, label="pads x1.25"),
    "foot_both": dict(ids=("60-M-T-10", "60-M-H-10", "60-N-H-10", "75-M-T-10"),
                      overrides=(("toe_scale", 1.15), ("pad_scale", 1.25)), headroom=1.0, fr05=False, min_gap=0.0,
                      label="toes x1.15 + pads x1.25"),
    "narrow_3mm": dict(ids=("60-N-T-10", "60-N-H-10"), overrides=(), headroom=1.0, fr05=False, min_gap=3.0,
                       label="narrow stance, target turns >= 3 mm"),
    "no_calf": dict(ids=("60-M-T-10", "60-M-H-10", "60-N-H-10", "75-M-T-10"), overrides=(("calf_fairing", 0),),
                    headroom=1.0, fr05=False, min_gap=0.0, label="no calf fairing (exposed ankle drives)"),
    # carpet-ready 60 cm: the ankle drives lowered to 0.60 of the shank (knee range 0.81 -> about 1.0 rad with the
    # calf fairing kept, found overnight) and sized for the carpet swing (nominal walk, swing lifted to 0.19 H) and
    # the Froude 0.05 walk
    "carpet_60": dict(ids=("60-M-T-10", "60-M-H-10"), overrides=(("ankle_frac", 0.60),), headroom=1.0, fr05=True,
                      walks=("fr05", "lift19"), min_gap=0.0,
                      label="60 cm: ankle drives at 0.60, sized for the 0.19 H carpet swing and Froude 0.05"),
    # material and power experiments on the two refined favorites (owner question overnight): carbon-fibre frame
    # plates (motor brackets stay aluminium), covering walls 25 % thinner, tether-only power, and the lightest
    # combination; each on top of carpet_60 (60 cm) or targeted_75 (75 cm)
    **{f"mat60_{k}": dict(ids=("60-M-T-10",), overrides=(("ankle_frac", 0.60),) + ov, headroom=1.0, fr05=True,
                          walks=("fr05", "lift19"), min_gap=0.0, label=f"60 cm carpet-ready + {lab}")
       for k, ov, lab in MAT_LEVERS},
    **{f"mat75_{k}": dict(ids=("75-M-T-10",), overrides=ov, headroom=1.0, fr05=True, min_gap=0.0,
                          headroom_by_group={"neck": 0.85, "hip_roll": 0.85}, label=f"75 cm targeted + {lab}")
       for k, ov, lab in MAT_LEVERS},
    # targeted upgrade at 75 cm (cross-check recommendation): the what-if's Froude 0.05 sizing case plus 15 %
    # headroom on the two drives every judge found thin, the neck servo and the hip roll
    "targeted_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(), headroom=1.0, fr05=True, min_gap=0.0,
                        headroom_by_group={"neck": 0.85, "hip_roll": 0.85},
                        label="75 cm: Froude 0.05 sizing + 15 % headroom on neck and hip roll"),
    # carpet-ready 75 cm: targeted_75 plus the carpet swing as a sizing walk.  The lifted-walk pass on targeted_75
    # puts the 0.06 H target (30.1-30.4 mm above the pile) at a lift of about 0.176 H, so the nominal walk is
    # lifted to 0.18 H; the knee clears its range there (0.135 rad at 0.19 H on Tapered), so no ankle change.
    # Result: the sizing selects exactly the targeted_75 drives (same SHA-256), so carpet_75 is not screened
    # separately; the lifted walks are checked with ``--carpet-lift --lift-cases targeted_75 --lifts 0.175 0.18``
    "carpet_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(), headroom=1.0, fr05=True, screen=False,
                      walks=("fr05", "lift18"), min_gap=0.0, headroom_by_group={"neck": 0.85, "hip_roll": 0.85},
                      label="75 cm: targeted drives, sized for the 0.18 H carpet swing and Froude 0.05"),
    # knee headroom: on the carpet-ready bodies the governing limit is the knee's speed (0.95 at both sizes), a
    # kinematic requirement that a lighter body does not change (same-drive pass); 15 % headroom on the knee
    # lets the sizing pick a faster knee drive instead of the lightest one that passes
    # the narrow stance with 3 mm turn clearance tops the version-2 scorecard but keeps the round-two ankle station;
    # combined here with the carpet-ready ankle drives (0.60 of the shank) and the carpet swing sizing
    "narrow_carpet_60": dict(ids=("60-N-T-10", "60-N-H-10"), overrides=(("ankle_frac", 0.60),), headroom=1.0, fr05=True,
                             walks=("fr05", "lift19"), min_gap=3.0,
                             label="60 cm narrow stance (turns >= 3 mm) + carpet-ready ankle drives"),
    # the foot lever on the refined bodies: longer toes (x1.15) and larger pads (x1.25) cut the friction need with the
    # stance foot's yaw moment by about 8 % and raise the one-foot stop from 1.24 to 1.45 m/s^2 on the round-two
    # bodies (foot_both), so they are combined here with the carpet-ready 60 cm and targeted 75 cm bodies
    "feet_60": dict(ids=("60-M-T-10", "60-M-H-10"),
                    overrides=(("ankle_frac", 0.60), ("toe_scale", 1.15), ("pad_scale", 1.25)), headroom=1.0, fr05=True,
                    walks=("fr05", "lift19"), min_gap=0.0,
                    label="60 cm carpet-ready + toes x1.15 and pads x1.25"),
    "feet_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(("toe_scale", 1.15), ("pad_scale", 1.25)), headroom=1.0,
                    fr05=True, min_gap=0.0, headroom_by_group={"neck": 0.85, "hip_roll": 0.85},
                    label="75 cm targeted + toes x1.15 and pads x1.25"),
    "knee_60": dict(ids=("60-M-T-10", "60-M-H-10"), overrides=(("ankle_frac", 0.60),), headroom=1.0, fr05=True,
                    walks=("fr05", "lift19"), min_gap=0.0, headroom_by_group={"knee": 0.85},
                    label="60 cm carpet-ready + 15 % knee headroom"),
    "knee_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(), headroom=1.0, fr05=True, walks=("fr05", "lift18"), min_gap=0.0,
                    headroom_by_group={"neck": 0.85, "hip_roll": 0.85, "knee": 0.85},
                    label="75 cm carpet-ready + 15 % knee headroom"),
    # Stage 1b arm K (STAGE1B_PLAN.md section 2): the narrow carpet-ready picks (narrow_carpet_60) with knee_60's 15 %
    # knee headroom, which on the base-width bodies selects a RobStride 06 knee (ROUND2_REVIEW.md section 7.3)
    "narrow_knee_60": dict(ids=("60-N-T-10", "60-N-H-10"), overrides=(("ankle_frac", 0.60),), headroom=1.0, fr05=True,
                           walks=("fr05", "lift19"), min_gap=3.0, headroom_by_group={"knee": 0.85},
                           label="60 cm narrow carpet-ready + 15 % knee headroom"),
}
#: Stage 2's expressive head (``STAGE2_PLAN.md`` S2-2; module docstring): the part that drives each of its two joints
#: (the plan's STS3215: 0.055 kg, $22, planning sustained 0.491 N m, peak 0.862 N m, 2.83 rad/s), the overrides that
#: add them (``build.Design``), and the pick each case builds on.  A case's sizing settings (headroom, sizing walks,
#: turn gap) are its base case's, so its requirements are evaluated on the base's walks; none of them selects anything.
HEAD_X_PART = "feetech_sts3215_c018"
HEAD_X_OVERRIDES = (("neck_yaw_joint", 1), ("jaw_joint", 1))
HEAD_X_BASE = {"head_x_60": ("narrow_carpet_60", "60-N-T-10"), "head_x_75": ("feet_75", "75-M-H-10")}
HEAD_X_RULE = ("the base case's sized record of the pick (its overrides with the packaging search's, B, drives, pack "
               "and the sizing's mass guess) with the expressive head's overrides and an STS3215 for each of its two "
               "joints added to the selection; built once, not re-sized: no drive, pack or packaging step is "
               "re-selected, so with the two joints locked the body is the pick plus the added parts (the integration "
               "reserve leaves them out, mjcf.body_inertials); its requirements use the base case's sizing walks and "
               "its margins are against the pick's drives")
#: The head-X body's home pose against its pick's (``head_x.home_pose``, ``against_base_at_pick_pose``).
HEAD_X_HOME_RULE = (
    "the build solves the stance for the body as built (pipeline.build_candidate: statics.solve_stance shifts the feet "
    "fore and aft so that the centre of mass is centred over the support, the trunk pose fixed), so the head-X home is "
    "not the pick's: the legs' joints differ (joint_change_rad) and against_base and com_ahead_of_hip_m's change compare "
    "two stances, the added parts with the re-solved stance.  against_base_at_pick_pose and com_ahead_of_hip_m's "
    "change_at_pick_pose compare the head-X body posed at the pick's home (the new joints at 0) with the pick: the "
    "added parts alone, as sizing.requirements reads them from that pose.  The plant's usable leg ranges, read by the "
    "clearance screen from sweeps from home, move a little with the home")
#: Neither new joint's range is screened at its seam (``head_x.joint_ranges``).
HEAD_X_RANGES_RULE = (
    "neither new joint's range is screened at its own seam: the clearance screen pairs no body with its parent "
    "(clearance.SEAM_RULE), so its sweeps cannot see the neck against the chest (neck yaw) or the jaw against the "
    "skull and the neck (jaw), and joint_range_usable, which the plant takes, stays at the MJCF limits.  The seam "
    "sweep (clearance.seam_sweep) is recorded here for the two and, for comparison, for today's neck pitch and tail "
    "yaw, which share the blind spot")
HEAD_X_RANGES_DECISION = (
    "annotated, not cut: neither sweep finds a new contact; what grows is the overlap of coarse proxies that already "
    "overlap at the seam at home, of the order today's neck pitch shows over its +-0.35 rad (accepted in rounds one and "
    "two), and the neck proxy against the neck-yaw servo and against today's pitch servo, which this model keeps on the "
    "trunk (on a yaw stage it would turn with the neck), is packaging the yaw stage's design must settle.  The jaw "
    "cover is today's, so its part behind the hinge swings up into the skull and the neck as the jaw opens "
    "(cover_behind_hinge): a build would trim the cover at the hinge; the model keeps it, so that the locked body stays "
    "today's plus the added parts, and renders of an open jaw show it")
for _case, (_base, _cid) in HEAD_X_BASE.items():
    CASES[_case] = dict(CASES[_base], ids=(_cid,), overrides=tuple(CASES[_base]["overrides"]) + HEAD_X_OVERRIDES,
                        base_case=_base, head_x=True,
                        label=f"{CASES[_base]['label']} + expressive head (neck yaw, jaw; not re-sized)")
del _case, _base, _cid
#: Stage 2's Jetson body (``JETSON_PLEM_REVIEW.md`` section 5 task 6; module docstring): the compute part, the regulator
#: placed for its carrier with the catalog parts it was chosen from, the override that places it (``build.Design``) and
#: the pick the case builds on.  As for the expressive head, the case's sizing settings are its base case's.
JETSON_COMPUTE = "nvidia_orin_nano_super_devkit"
JETSON_REGULATOR = "pololu_d42v110f12"
JETSON_REGULATOR_OPTIONS = ("pololu_d36v50f12", "pololu_d42v110f12")
JETSON_OVERRIDES = (("compute_regulator", 1),)
JETSON_BASE = {"jetson_75": ("feet_75", "75-M-H-10")}
#: The most the dev kit's carrier can pass to USB devices, W: its four USB 3.2 Type-A ports are two stacked pairs with
#: a 3 A VBUS limit each, at 5 V (``designlab/data/research/parts_2026-11-07.json``, ``jetson_orin_nano_super_devkit``:
#: "VBUS 3 A per stacked pair"; the compute survey: "3 A VBUS limit per dual stack").  The head camera is fed there.
JETSON_USB_VBUS_W = 2 * 3.0 * 5.0
JETSON_RULE = (
    "the base case's sized record of the pick (its overrides with the packaging search's, B, drives, pack, structure "
    "and the sizing's mass guess) with the selection's compute part changed to the Jetson Orin Nano Super dev kit, "
    "which the build places in the chest tray as it places any compute part, and the compute_regulator option on (the "
    "selection's 6S-to-12 V regulator as a box on the power box: build.COMPUTE_REGULATOR_STANDOFF_M); built once, not "
    "re-sized: no drive, pack or packaging step is re-selected, and the Teensy control stack stays (owner decision "
    "P30: the walking loop stays on the microcontroller).  Its requirements use the base case's sizing walks and its "
    "margins are against the pick's drives.  The record's power and runtime book the dev kit at power.COMPUTE_BUDGET's "
    "row, as every record books its compute part (round one's 31 W allowance, A13); jetson.power_modes books it at "
    "each of its two power modes (power.COMPUTE_MODE_BUDGET), since no mode is chosen")
JETSON_REGULATOR_RULE = (
    "the lightest of the catalog's 12 V regulators (JETSON_REGULATOR_OPTIONS; the carrier's DC jack takes 9-20 V and a "
    "6S pack is at 25.2 V full, DECISION_LOG E7) whose rated output covers the dev kit's worst case at the jack in "
    "both power modes, since no mode is chosen: the mode's rail budget (power.COMPUTE_MODE_BUDGET: the module's cap "
    "plus 3 W of carrier, storage and cooling) plus the most the carrier can pass to its USB ports "
    "(JETSON_USB_VBUS_W), the carrier's own conversion loss left out.  The rated output is the catalog's power_max_w "
    "(12 V times the vendor's continuous current at its stated input voltage; the vendor gives a range of currents "
    "over input voltage and no figure at 6S, so this is the catalog's one point, not a rating at 20-25 V)")
JETSON_REGULATOR_CONSEQUENCE_RULE = (
    "what the rule's worst case costs in mass: on the modes' rail budgets alone (no USB load) the lighter option "
    "covers both modes (load_fraction_at_rail_budget), and the body would then be lighter by the two catalog masses' "
    "difference (lighter_option.body_mass_change_kg; with_reserve_kg adds the model's 6 % integration reserve).  "
    "Whether the worst case can occur at the jack is not sourced: the catalog does not give the rating of the kit's "
    "own 19 V supply.  If the power box already holds the computer's regulator (may_be_counted_twice), the body is "
    "lighter than modelled by the placed row with its reserve (placed_row_with_reserve_kg)")
JETSON_DOUBLE_COUNT_NOTE = (
    "the power box's 0.30 kg is PR #662's power bay allowance, and that robot's bay held three regulators, a "
    "D36V50F12 for its computer among them (r0_posture_study_2026-11-07/study/hardware/component_basis.md); the bay's "
    "own breakdown is not in this repository, so the regulator placed here may duplicate one the box already holds: "
    "the body is then too heavy by at most the regulator's installed mass with its reserve")
JETSON_HOME_RULE = (
    "the build solves the stance for the body as built (pipeline.build_candidate: statics.solve_stance shifts the feet "
    "fore and aft so that the centre of mass is centred over the support, the trunk pose fixed), so the Jetson body's "
    "home is not the pick's: the legs' joints differ (joint_change_rad) and against_base and com_ahead_of_hip_m's "
    "change compare two stances, the changed parts with the re-solved stance.  against_base_at_pick_pose and "
    "com_ahead_of_hip_m's change_at_pick_pose compare the Jetson body posed at the pick's home with the pick: the "
    "changed parts alone, as sizing.requirements reads them from that pose")
#: What the pick's record holds of the Raspberry Pi set, and what leaves with it (``jetson.pi_set``).
JETSON_PI_SET_DECISION = (
    "leaves: the compute row (catalog part rpi5_8gb_ai_hat_plus_13tops, one part: the Pi 5, its cooler and the AI "
    "HAT+; its 0.100 kg leaves out the SSD) and the set's two power rows (pi5_ai_hat_plus, usb_ssd), which the dev "
    "kit's rows replace.  Stays: the compute tray's 0.12 kg allowance (NVMe, fan duct, mounts and leads: the build "
    "gives it to every compute part and resizes its envelope with the board), the 0.30 kg power box, the control "
    "stack and the IMU (the Teensy stays), and the 4 W MCU row.  The record holds no 5 V supply for the Pi and no SSD "
    "as items of their own (the catalog part's note names a 5 V, 5 A buck and a USB SSD; neither has a ledger row), so "
    "there is none to remove")
#: The Jetson-direct option's parts (``JETSON_PLEM_REVIEW.md`` section 6 decision 2), which are not the baseline and
#: are not placed: the USB-CAN adapter (one per CAN bus; the study's control stack runs two) and a supervisor.
JETSON_DIRECT_ADAPTER = "pcan_usb_fd"
JETSON_NOT_PLACED_RULE = (
    "not the baseline (owner decision P30: the microcontroller runs the walking loop and the CAN buses), so not "
    "placed and in no mass, centre of mass or screen of this body.  would_add_kg is the catalog mass times the "
    "count, and with_reserve_kg that plus the model's 6 % integration reserve: estimates of what the Jetson-direct "
    "option would add, with no position.  The catalog holds no supervisor part, so no mass is given for one")
for _case, (_base, _cid) in JETSON_BASE.items():
    CASES[_case] = dict(CASES[_base], ids=(_cid,), overrides=tuple(CASES[_base]["overrides"]) + JETSON_OVERRIDES,
                        base_case=_base, jetson=True,
                        label=f"{CASES[_base]['label']} + Jetson Orin Nano Super dev kit and its 12 V regulator "
                              "(not re-sized; power and runtime here are at round one's 31 W compute allowance, "
                              "A13; the two power modes' figures are in the record's jetson.power_modes)")
del _case, _base, _cid
CARPET_LIFT_IDS = ("60-N-H-10", "60-N-T-10", "60-M-T-10", "60-M-H-10", "75-M-T-10-bighead")
LIFTS = (0.13, 0.16, 0.19, 0.22)


def extra_walks(names) -> dict:
    """Extra sizing walks by name: ``fr05`` (the behavior envelope's Froude 0.05 walk) and ``lift<NN>`` (the
    nominal walk with its swing lifted to 0.NN H, the carpet swing case)."""
    from . import envelope, sizing

    out = {}
    for n in names:
        if n == "fr05":
            out[n] = envelope.FAST
        elif n.startswith("lift"):
            out[n] = replace(sizing.NOMINAL, lift_frac=int(n[4:]) / 100.0)
        else:
            raise ValueError(n)
    return out


def _init(case: str | None) -> None:
    from . import gait, round2, sizing

    gait.set_accuracy(round2.GAIT_ACCURACY)
    c = CASES.get(case) if case else None
    sizing.HEADROOM = c["headroom"] if c else 1.0
    sizing.HEADROOM_BY_GROUP = dict(c.get("headroom_by_group") or {}) if c else {}
    names = (c.get("walks") or (("fr05",) if c["fr05"] else ())) if c else ()
    sizing.EXTRA_WALKS = extra_walks(names)
    round2.TURN_SEARCH_MIN_GAP_MM = c["min_gap"] if c else 0.0


def _size(a):
    from . import round2

    case, cid, out = a
    _init(case)
    if CASES[case].get("head_x"):  # Stage 2's expressive head: the pick plus the head's drives, not re-sized
        return dict(case=case, **head_x_one(case, cid, out))
    if CASES[case].get("jetson"):  # Stage 2's Jetson body: the pick with the dev kit and its regulator, not re-sized
        return dict(case=case, **jetson_one(case, cid, out))
    r = round2.size_one(cid, out, str(Path(out) / "models"), base_overrides=CASES[case]["overrides"])
    p = Path(out, "candidates", f"{cid}.json")
    d = json.loads(p.read_text())
    c = CASES[case]
    d.update(refine_case=case, headroom=c["headroom"], headroom_by_group=dict(c.get("headroom_by_group") or {}),
             fr05_sizing=c["fr05"], turn_min_gap_mm=c["min_gap"])  # records sized before 10 Nov lack the groups
    p.write_text(json.dumps(d, indent=1))
    return dict(case=case, **r)


def _screen(a):
    from . import round2

    case, cid, out = a
    _init(case)
    return dict(case=case, **round2._screen_star((cid, Path(out))))


# ------------------------------------------------------------------------------------------------
# Stage 2's expressive head (head_x_60, head_x_75): the pick plus the head's drives, not re-sized
# ------------------------------------------------------------------------------------------------
def head_x_base(case: str, out) -> tuple[Path, dict]:
    """The base case's sized record of a head-X case's pick (``HEAD_X_BASE``): ``<out>/../<base>/candidates/<id>.json``
    (``out`` the case's own directory), its path and contents."""
    base, cid = HEAD_X_BASE[case]
    p = Path(out).parent / base / "candidates" / f"{cid}.json"
    return p, json.loads(p.read_text())


def head_x_candidate(case: str, base_rec: dict) -> tuple:
    """(candidate, selection, B) of a head-X body (``HEAD_X_RULE``): the base record's with the expressive head's
    overrides added to its overrides and an ``HEAD_X_PART`` for each of the head's groups added to its selection."""
    from . import round2, sizing

    _, cid = HEAD_X_BASE[case]
    ov = tuple(tuple(o) for o in base_rec.get("overrides", ())) + HEAD_X_OVERRIDES
    sel = dict(base_rec["selection"], **{g: HEAD_X_PART for g in sizing.HEAD_X_GROUPS})
    return round2.candidate(cid, ov), sel, float(base_rec["B"])


def _margins(design, req: dict, sel: dict) -> dict:
    """Each group's requirements against its drive (``pipeline.size_candidate``'s margins), for drives ``sel``."""
    margins = {}
    for g, r in req["groups"].items():
        a = design.act(g)
        spd = a.speed_at(sel["bus"] if a.box is None else None)
        margins[g] = dict(actuator=a.id, sustained=a.sustained, peak=a.peak, speed=spd,
                          cont_req=r["cont"], peak_req=r["peak_req"], speed_req=r["speed_req"],
                          cont_ratio=r["cont"] / a.sustained, peak_ratio=r["peak_req"] / a.peak,
                          speed_ratio=r["speed_req"] / max(spd, 1e-6), rms=r["rms"], hold=r["hold"],
                          double_support=r["double_support"])
    return margins


def _evaluate(design, res: dict, req: dict, sel: dict, history: list) -> dict:
    """What ``pipeline.size_candidate`` records for its final build (requirements, holds, walks, margins against
    the drives, walking power, runtime, cost; the same fields in the same order), for a build whose drives and pack
    are ``sel``."""
    from . import cost, power, sizing

    nom = req["runs"]["nominal"]["idr"]
    ms = sizing._motor_space(res["model"].names, nom["tau"], nom["omega"], design.cand.leg_layout)
    pw = power.walking_power(design, req, ms)
    rt = power.runtime(power.catalog.part(sel["battery"]), pw["total_w"])
    margins = _margins(design, req, sel)
    return dict(requirements={g: r for g, r in req["groups"].items()}, holds=req["holds"], gaits=req["gaits"],
                walk_feasible=req["walk_feasible"], margins=margins, power=pw, runtime=rt, sizing_history=history,
                cost=cost.estimate(design, res["rows"], sel))


def head_x_build(case: str, base_rec: dict, models_dir) -> dict:
    """Build a head-X body once (``HEAD_X_RULE``) into ``models_dir`` and evaluate it as the sizing evaluates its
    final build: writes ``<models_dir>/<id>.xml`` and the model ledger ``<id>.json`` with its evaluation, as
    ``pipeline.size_candidate`` writes them.  The case's sizing settings must be set first (``_init``)."""
    from . import mjcf, pipeline, sizing

    cand, sel, B = head_x_candidate(case, base_rec)
    models_dir = Path(models_dir)
    res = pipeline.build_candidate(cand, models_dir, selection=sel, B=B, write=False)
    design = res["design"]
    req = sizing.requirements(res["model"], res["q_home"], design.sk.H, cand.leg_layout)
    history = [dict(iteration="head_x", mass=res["summary"]["mass"],
                    selection=dict(joints=sel["joints"], neck=sel["neck"], tail=sel["tail"], bus=sel["bus"],
                                   battery=sel["battery"], **{g: sel[g] for g in sizing.HEAD_X_GROUPS}),
                    notes=[f"not re-sized: {HEAD_X_RULE}"])]
    ev = _evaluate(design, res, req, sel, history)
    mjcf.write_ledger(design, res["rows"], models_dir / f"{cand.id}.json",
                      extra=dict(summary=res["summary"], evaluation=ev))
    return dict(res, requirements=req, evaluation=ev, selection=sel, candidate=cand)


def _against(x: dict, base: dict) -> dict:
    """A head-X record's loads and margins beside its base record's, group by group."""
    out = {}
    for g, m in x["margins"].items():
        b = base["margins"].get(g)
        row = dict(actuator=m["actuator"], hold_Nm=m["hold"], rms_Nm=m["rms"],
                   ratios=dict(cont=m["cont_ratio"], peak=m["peak_ratio"], speed=m["speed_ratio"]))
        if b is not None:
            row.update(base_hold_Nm=b["hold"], hold_change_Nm=m["hold"] - b["hold"], base_rms_Nm=b["rms"],
                       base_ratios=dict(cont=b["cont_ratio"], peak=b["peak_ratio"], speed=b["speed_ratio"]),
                       hold_over_sustained=m["hold"] / m["sustained"], base_hold_over_sustained=b["hold"] / b["sustained"])
        out[g] = row
    return out


def _stance_effect(own: dict, at_pick: dict, floor_nm: float = 0.01) -> dict:
    """How far the loads move between the two stances (``against_base`` as ``own``, ``against_base_at_pick_pose`` as
    ``at_pick``): the largest difference of any group's ratio to its drive's limits (continuous, peak, speed; in
    fractions of the limit), and the largest relative difference of a hold or an RMS torque of at least ``floor_nm``,
    each with where."""
    ratio, torque = (0.0, None, None), (0.0, None, None)
    for g, a in own.items():
        b = at_pick.get(g)
        if b is None:
            continue
        for k, v in a["ratios"].items():
            d = abs(v - b["ratios"][k])
            if d > ratio[0]:
                ratio = (d, g, k)
        for k in ("hold_Nm", "rms_Nm"):
            x, y = a[k], b[k]
            if max(abs(x), abs(y)) >= floor_nm:
                rel = abs(x - y) / max(abs(x), abs(y))
                if rel > torque[0]:
                    torque = (rel, g, k)
    return dict(ratio_max_difference=ratio[0], ratio_where=[ratio[1], ratio[2]],
                torque_max_relative_difference=torque[0], torque_where=[torque[1], torque[2]],
                torque_floor_Nm=floor_nm,
                rule="the largest change between the two stances (against_base, against_base_at_pick_pose) of a "
                     "group's ratio to its drive's limits, and of a hold or RMS torque (relative, those of at least "
                     "the floor): the sizing's holds and walks re-centre the centre of mass over the support from "
                     "whichever home they start, so the re-solved stance moves the loads only this much, while the "
                     "centre of mass at home moves with it (com_ahead_of_hip_m)")


def head_x_seams(xml) -> dict:
    """``head_x.joint_ranges``: ``HEAD_X_RANGES_RULE`` with the seam sweep of the model at ``xml`` (each swept joint's
    range, whether a new contact appears, the deepest growth of a home overlap and, per direction, the first new
    contact and the three deepest growths), the jaw cover's part behind the hinge, and ``HEAD_X_RANGES_DECISION``."""
    from . import clearance, sizing

    sw = clearance.seam_sweep(xml)
    seams = {}
    for jn, r in sw["joints"].items():
        seams[jn] = dict(moving=r["moving"], parent=r["parent"], range=r["range"], new_contact=r["new_contact"],
                         deepens_mm_max=r["deepens_mm_max"],
                         directions={end: dict(limit=d["limit"], first_new_contact=d["first_new_contact"],
                                               deepening=d["deepening"][:3])
                                     for end, d in r["directions"].items() if d.get("swept")})
    jaw = sw["joints"].get("jaw", {})
    return dict(screened=False, rule=HEAD_X_RANGES_RULE, seam_rule=clearance.SEAM_RULE,
                deepen_threshold_mm=sw["deepen_threshold_mm"],
                joints=[sizing.HEAD_X_JOINT[g] for g in sizing.HEAD_X_GROUPS if sizing.HEAD_X_JOINT[g] in sw["joints"]],
                seams=seams, cover_behind_hinge=jaw.get("cover_behind_hinge"), decision=HEAD_X_RANGES_DECISION)


def head_x_at_pick_pose(base_rec: dict, r: dict, sel: dict, rule: str = HEAD_X_HOME_RULE, pick_reader=None) -> dict:
    """``HEAD_X_HOME_RULE``: the pick rebuilt from its record (its model's sha256 checked) for its home pose; the
    head-X model's home against it (the trunk's pose and each leg joint's change); and the head-X body posed at the
    pick's home (the new joints at 0): its centre of mass ahead of the hip and ``sizing.requirements`` from that pose,
    against the pick's drives (``_margins``).  ``r`` is ``head_x_build``'s result.

    The Jetson body uses it too (``jetson_one``, ``r`` from ``jetson_build``), with its own ``rule`` and a
    ``pick_reader(pick build, pick xml)`` whose result is returned as ``pick`` (the rebuilt pick's files are gone
    afterwards); neither changes what the head-X records hold."""
    import hashlib
    import tempfile

    import mujoco
    import numpy as np

    from . import pipeline, round2, sizing

    design, M, qx = r["design"], r["model"], np.asarray(r["q_home"], float)
    with tempfile.TemporaryDirectory(prefix="head_x_pick_") as tmp:
        pk = pipeline.build_candidate(round2.candidate(base_rec["id"], base_rec.get("overrides", ())), Path(tmp),
                                      selection=dict(base_rec["selection"]), B=float(base_rec["B"]), write=False)
        sha = hashlib.sha256((Path(tmp) / f"{base_rec['id']}.xml").read_bytes()).hexdigest()
        picked = pick_reader(pk, Path(tmp) / f"{base_rec['id']}.xml") if pick_reader is not None and sha == base_rec[
            "xml_sha256"] else None
    if sha != base_rec["xml_sha256"]:
        raise ValueError(f"the pick rebuilt with xml sha256 {sha[:12]}..., its record {base_rec['xml_sha256'][:12]}...")
    m0, q0 = pk["model"].m, np.asarray(pk["q_home"], float)
    mx = M.m
    q = mx.qpos0.copy()  # the new joints at 0 (their reference), every other joint at the pick's home
    root, joints = 0.0, {}
    for j in range(m0.njnt):
        name = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
        k = mujoco.mj_name2id(mx, mujoco.mjtObj.mjOBJ_JOINT, name)
        n = 7 if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        a0, ax = int(m0.jnt_qposadr[j]), int(mx.jnt_qposadr[k])
        q[ax:ax + n] = q0[a0:a0 + n]
        if n == 7:
            root = float(np.max(np.abs(qx[ax:ax + 7] - q0[a0:a0 + 7])))
        elif qx[ax] != q0[a0]:
            joints[name] = float(qx[ax] - q0[a0])
    M.forward(q)
    com_vec = [float(v) for v in M.com()]
    com_pick = com_vec[0]
    M.forward(qx)
    worst = max(joints.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
    out = dict(home_pose=dict(rule=rule, pick_xml_sha256=sha, trunk_pose_change_max=root,
                              joint_change_rad=joints, largest_joint_change=[worst[0], worst[1]]),
               com_ahead_of_hip_at_pick_pose_m=com_pick)
    if pick_reader is not None:  # the Jetson body's readings (the head-X records take the keys above only)
        out.update(com_at_pick_pose_m=com_vec, pick=picked)
    try:
        req = sizing.requirements(M, q, design.sk.H, design.cand.leg_layout)
        out["margins_at_pick_pose"] = _margins(design, req, sel)
    except Exception as e:  # recorded, not fatal: the record's own evaluation stands
        out["margins_at_pick_pose_error"] = repr(e)
    M.forward(qx)
    return out


def head_x_one(case: str, cid: str, out: str) -> dict:
    """The sized record of one head-X body: ``head_x_build`` into ``<out>/models``, then ``<out>/candidates/<id>.json``
    with ``round2.size_one``'s fields, the case's settings (as ``_size`` adds them) and ``head_x``: the rule, the base
    record it builds on (path and sha256, its model's sha256), the parts it adds, its loads and margins beside the
    base's, and what one selection step on its loads would pick (recorded only; nothing is re-selected)."""
    import hashlib
    import traceback

    import designlab  # noqa: F401

    from . import gait, metrics, mjcf, power, round2, sizing
    from .stage_a import _mass_by_category, _mass_by_volume_class

    gait.set_accuracy(round2.GAIT_ACCURACY)
    t0 = time.time()
    c = CASES[case]
    try:
        bp, base = head_x_base(case, out)
        r = head_x_build(case, base, Path(out) / "models")
        d, sel, cand, ev, req = r["design"], r["selection"], r["candidate"], r["evaluation"], r["requirements"]
        meas = metrics.measure(r["model"], d, r["q_home"])
        added = [row for row in r["rows"] if mjcf._head_x(row)]
        pick = sizing.select(req, mass=r["summary"]["mass"], layout=cand.leg_layout)
        current = dict(sel["joints"], **{g: sel[g] for g in ("neck", "tail") + sizing.HEAD_X_GROUPS})
        nom = req["runs"]["nominal"]["idr"]
        pw = power.walking_power(d, req, sizing._motor_space(r["model"].names, nom["tau"], nom["omega"],
                                                             cand.leg_layout))
        bat = power.select_battery(pick["bus"], pw["total_w"], pw["peak_drives_w"] + pw["aux"]["input_w"])
        over = sorted(g for g, m in ev["margins"].items() if max(m["cont_ratio"], m["peak_ratio"], m["speed_ratio"]) > 1.0)
        at_pick = head_x_at_pick_pose(base, r, sel)
        seams = head_x_seams(Path(out) / "models" / f"{cid}.xml")
        head_x = dict(
            rule=HEAD_X_RULE, case=case, base_case=c["base_case"], base_record=bp.as_posix(),
            base_record_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(), base_xml_sha256=base["xml_sha256"],
            part=HEAD_X_PART, overrides_added=[list(o) for o in HEAD_X_OVERRIDES],
            groups=list(sizing.HEAD_X_GROUPS),
            parts=[dict(name=row["name"], body=row["body"], group=row["meta"][mjcf.HEAD_X_META],
                        category=row["category"], mass_kg=row["mass_kg"], com_body=row["com_body"],
                        source=row["source"]) for row in added],
            added_mass_kg=sum(row["mass_kg"] for row in added),
            mass_kg=dict(head_x=meas["mass"], base=base["metrics"]["mass"]),
            com_ahead_of_hip_m=dict(head_x=meas["com_ahead_of_hip"], base=base["metrics"]["com_ahead_of_hip"],
                                    change=meas["com_ahead_of_hip"] - base["metrics"]["com_ahead_of_hip"],
                                    head_x_at_pick_pose=at_pick["com_ahead_of_hip_at_pick_pose_m"],
                                    change_at_pick_pose=at_pick["com_ahead_of_hip_at_pick_pose_m"]
                                    - base["metrics"]["com_ahead_of_hip"],
                                    note="change: each body at its own home (two stances); change_at_pick_pose: the "
                                         "head-X body at the pick's home (home_pose)"),
            home_pose=at_pick["home_pose"],
            head_neck_mass_kg=dict(head_x=meas["head_neck_mass"], base=base["metrics"]["head_neck_mass"]),
            groups_over_planning_limit=over,
            meets_case_headroom={g: bool(sizing.meets(d.act(g), req["groups"][g],
                                                      sel["bus"] if d.act(g).box is None else None, g))
                                 for g in req["groups"]},
            against_base=_against(dict(margins=ev["margins"]), base),
            against_base_at_pick_pose=(_against(dict(margins=at_pick["margins_at_pick_pose"]), base)
                                       if "margins_at_pick_pose" in at_pick
                                       else dict(error=at_pick["margins_at_pick_pose_error"])),
            loads_between_stances=(_stance_effect(_against(dict(margins=ev["margins"]), base),
                                                  _against(dict(margins=at_pick["margins_at_pick_pose"]), base))
                                   if "margins_at_pick_pose" in at_pick else None),
            joint_ranges=seams,
            selection_if_resized=dict(
                rule="one step of sizing.select on these loads with the case's headroom, and power.select_battery "
                     "on its walking power: recorded only (the sizing loop, which would iterate on the re-selected "
                     "parts' masses, is not run, and nothing here is re-selected)",
                choice=pick["choice"], bus=pick["bus"], battery=bat["pack"], notes=pick["notes"],
                differs={g: [current.get(g), p] for g, p in pick["choice"].items() if current.get(g) != p},
                battery_differs=bat["pack"] != sel["battery"], bus_differs=pick["bus"] != sel["bus"]))
        doc = dict(id=cid, rev=round2.REV, candidate=dict(height=cand.height, width=cand.width, family=cand.family,
                                                          posture=cand.posture, proportion=cand.proportion,
                                                          leg_layout=cand.leg_layout, battery_pos=cand.battery_pos),
                   B=d.B, H=d.sk.H, xml_sha256=r["summary"]["xml_sha256"], selection=dict(sel), metrics=meas,
                   margins=ev["margins"], holds=ev["holds"], power=ev["power"], runtime=ev["runtime"], cost=ev["cost"],
                   gaits=ev["gaits"], sizing_history=ev["sizing_history"], walk_feasible=ev["walk_feasible"],
                   geometry_checks=d.geometry_checks, notes=d.notes,
                   overrides=[list(o) for o in cand.overrides if o[0] != "rev"], packaging_search=[],
                   mass_by_category=_mass_by_category(r["rows"]), mass_by_volume_class=_mass_by_volume_class(r["rows"]),
                   gait_accuracy=gait.ACCURACY, seconds=time.time() - t0, ok=True,
                   refine_case=case, headroom=c["headroom"], headroom_by_group=dict(c.get("headroom_by_group") or {}),
                   fr05_sizing=c["fr05"], turn_min_gap_mm=c["min_gap"], head_x=head_x)
    except Exception as e:  # keep failed cases, as round2.size_one does
        doc = dict(id=cid, rev=round2.REV, ok=False, error=repr(e), traceback=traceback.format_exc(),
                   seconds=time.time() - t0, refine_case=case)
    Path(out, "candidates").mkdir(parents=True, exist_ok=True)
    Path(out, "candidates", f"{cid}.json").write_text(json.dumps(doc, indent=1, default=round2._json))
    return dict(id=cid, ok=doc["ok"], seconds=doc["seconds"])


# ------------------------------------------------------------------------------------------------
# Stage 2's Jetson body (jetson_75): the pick with the dev kit and its regulator, not re-sized
# ------------------------------------------------------------------------------------------------
#: Trunk hardware pairs whose home-pose gap ``jetson_chest`` reads (ledger item names; a pair with an item the model
#: lacks is left out): the compute board and its tray against the power box, the hip-roll motors and what else shares
#: the chest, the power box against the hip-roll motors, and the regulator against everything near it.
JETSON_CHEST_PAIRS = tuple(
    [(a, b) for a in ("compute", "compute_tray") for b in ("power_distribution", "left_hip_roll_motor",
                                                           "right_hip_roll_motor", "spine_beam", "neck_motor", "tof_l",
                                                           "tof_r")]
    + [("power_distribution", b) for b in ("left_hip_roll_motor", "right_hip_roll_motor", "tof_l", "tof_r",
                                           "pelvis_plate_l", "pelvis_plate_r")]
    + [("compute_regulator", b) for b in ("power_distribution", "compute", "compute_tray", "left_hip_roll_motor",
                                          "right_hip_roll_motor", "spine_beam", "pelvis_plate_l", "pelvis_plate_r",
                                          "tof_l", "tof_r", "battery_cassette", "control_stack")])
JETSON_CHEST_ITEMS = ("compute", "compute_tray", "power_distribution", "compute_regulator")
JETSON_CHEST_RULE = (
    "at the home pose, with the clearance screen's geoms and distance oracle (clearance._Screen): gaps_mm, the signed "
    "distance of each listed pair of trunk hardware boxes (negative: they intersect; a box inside its own envelope, "
    "as the compute board in its tray, reads negative and is counted under nested_envelopes); nearest_other_body, "
    "the closest screened geom of another rigid body (a leg, the head, the tail) to each item; "
    "same_body_component_overlaps, the screen's own reading behind its component_overlap_free gate; and "
    "trunk_profile_margin_mm, each item's corners against the as-built trunk profile with the build's 8 mm + 2 mm "
    "allowance (Design.profile_margin; positive: inside).  The trunk's own covers are not paired with its hardware: "
    "the screen's cover pieces are convex hulls that contain it.  The walking gates read the moving pairs over the "
    "walks (the case's clearance screen)")


def jetson_base(case: str, out) -> tuple[Path, dict]:
    """The base case's sized record of the Jetson case's pick (``JETSON_BASE``): its path and contents."""
    base, cid = JETSON_BASE[case]
    p = Path(out).parent / base / "candidates" / f"{cid}.json"
    return p, json.loads(p.read_text())


def jetson_candidate(case: str, base_rec: dict) -> tuple:
    """(candidate, selection, B) of the Jetson body (``JETSON_RULE``): the base record's with the regulator's override
    added to its overrides, and in its selection the compute part changed and the regulator named."""
    from . import round2

    _, cid = JETSON_BASE[case]
    ov = tuple(tuple(o) for o in base_rec.get("overrides", ())) + JETSON_OVERRIDES
    sel = dict(base_rec["selection"], compute=JETSON_COMPUTE, compute_regulator=JETSON_REGULATOR)
    return round2.candidate(cid, ov), sel, float(base_rec["B"])


def jetson_build(case: str, base_rec: dict, models_dir) -> dict:
    """Build the Jetson body once (``JETSON_RULE``) into ``models_dir`` and evaluate it as the sizing evaluates its
    final build: writes ``<models_dir>/<id>.xml`` and the model ledger ``<id>.json`` with its evaluation, as
    ``pipeline.size_candidate`` writes them.  The case's sizing settings must be set first (``_init``)."""
    from . import mjcf, pipeline, sizing

    cand, sel, B = jetson_candidate(case, base_rec)
    models_dir = Path(models_dir)
    res = pipeline.build_candidate(cand, models_dir, selection=sel, B=B, write=False)
    design = res["design"]
    req = sizing.requirements(res["model"], res["q_home"], design.sk.H, cand.leg_layout)
    history = [dict(iteration="jetson", mass=res["summary"]["mass"],
                    selection=dict(joints=sel["joints"], neck=sel["neck"], tail=sel["tail"], bus=sel["bus"],
                                   battery=sel["battery"], compute=sel["compute"],
                                   compute_regulator=sel["compute_regulator"]),
                    notes=[f"not re-sized: {JETSON_RULE}"])]
    ev = _evaluate(design, res, req, sel, history)
    mjcf.write_ledger(design, res["rows"], models_dir / f"{cand.id}.json",
                      extra=dict(summary=res["summary"], evaluation=ev))
    return dict(res, requirements=req, evaluation=ev, selection=sel, candidate=cand)


def jetson_regulators() -> dict:
    """``JETSON_REGULATOR_RULE`` on the catalog: each mode's worst case at the dev kit's jack, each candidate
    regulator's catalog figures and which modes its rated output covers, the rule's choice, and what that choice
    costs in mass against the lighter option (``JETSON_REGULATOR_CONSEQUENCE_RULE``)."""
    from . import build, catalog, mjcf, power

    modes = {m: sum(v.values()) for m, v in power.COMPUTE_MODE_BUDGET[JETSON_COMPUTE].items()}
    worst = {m: w + JETSON_USB_VBUS_W for m, w in modes.items()}
    kit = catalog.part(JETSON_COMPUTE)
    options = {}
    for pid in JETSON_REGULATOR_OPTIONS:
        r = catalog.part(pid)
        options[pid] = dict(model=r.model, mass_kg=r.mass, dims_m=list(r.dims), price_usd=r.price,
                            max_input_v=r.voltage, output_v=12.0, rated_output_a=r.max_current,
                            rated_output_w=r.power_max, basis=r.basis,
                            covers={m: bool(w <= r.power_max) for m, w in worst.items()},
                            load_fraction_at_rail_budget={m: w / r.power_max for m, w in modes.items()},
                            load_fraction_at_worst_case={m: w / r.power_max for m, w in worst.items()})
    both = sorted((o["mass_kg"], pid) for pid, o in options.items() if all(o["covers"].values()))
    placed = catalog.part(JETSON_REGULATOR)
    reserve = 1.0 / (1.0 - mjcf.INTEGRATION_RESERVE_FRACTION)
    light_mass, light = min((o["mass_kg"], pid) for pid, o in options.items())
    consequence = dict(
        rule=JETSON_REGULATOR_CONSEQUENCE_RULE,
        lighter_option=dict(part=light, covers_both_modes_at_rail_budget=bool(
            all(v <= 1.0 for v in options[light]["load_fraction_at_rail_budget"].values())),
            body_mass_change_kg=light_mass - placed.mass, with_reserve_kg=(light_mass - placed.mass) * reserve),
        placed_row_kg=placed.mass + build.COMPUTE_REGULATOR_LEADS_KG,
        placed_row_with_reserve_kg=(placed.mass + build.COMPUTE_REGULATOR_LEADS_KG) * reserve)
    return dict(rule=JETSON_REGULATOR_RULE, pack_full_v=25.2, carrier_input_v="9-20 V at the DC jack (compute survey: "
                "designlab/data/research/compute_2026-11-07.json, jetson_orin_nano_super_devkit); the catalog part's "
                f"voltage_v is its own supply's {kit.voltage:g} V",
                rail_budget_w=modes, usb_vbus_limit_w=JETSON_USB_VBUS_W, worst_case_at_jack_w=worst, options=options,
                choice=both[0][1] if both else None, placed=JETSON_REGULATOR, consequence=consequence,
                may_be_counted_twice=JETSON_DOUBLE_COUNT_NOTE,
                efficiency="the regulator's loss is in power.ETA_DCDC (0.88, A6), which every auxiliary rail takes; "
                           "the vendor gives over 90 % for this part (catalog research record), so 0.88 is the lower "
                           "of the two")


def jetson_power(design, res: dict, req: dict, sel: dict, base_rec: dict) -> dict:
    """``jetson.power_modes``: the Jetson body's walking power and runtime with the dev kit booked at each of its
    power modes (``power.COMPUTE_MODE_BUDGET``), by the method of every record's ``power`` and ``runtime``
    (``power.walking_power`` on the body's own nominal walk, ``power.runtime`` on its pack); and, for comparison only,
    the 15 W mode at the compute survey's typical draw."""
    from . import power, sizing

    nom = req["runs"]["nominal"]["idr"]
    ms = sizing._motor_space(res["model"].names, nom["tau"], nom["omega"], design.cand.leg_layout)
    pack = power.catalog.part(sel["battery"])
    base_min = base_rec["runtime"]["minutes"]
    modes = {}
    for mode, loads in power.COMPUTE_MODE_BUDGET[sel["compute"]].items():
        pw = power.walking_power(design, req, ms, compute_mode=mode)
        rt = power.runtime(pack, pw["total_w"])
        modes[mode] = dict(compute_rail_w=sum(loads.values()), compute_loads=dict(loads), drives_w=pw["drives_w"],
                           aux=pw["aux"], total_w=pw["total_w"], runtime=rt,
                           runtime_change_against_base_min=rt["minutes"] - base_min,
                           meets_target=bool(rt["minutes"] >= power.TARGET_MIN))
    m15 = modes["15w"]
    typ_rail = 12.0 + power.COMPUTE_MODE_BUDGET[sel["compute"]]["15w"]["carrier_storage_cooling"]
    typ_total = m15["total_w"] - (m15["compute_rail_w"] - typ_rail) / power.ETA_DCDC
    return dict(
        rule="power.walking_power(design, requirements, the nominal walk's motor series, compute_mode) and "
             "power.runtime(pack, total): the record's own power and runtime method, the dev kit's row replaced by the "
             "mode's (power.COMPUTE_MODE_BUDGET); drives, MCU, sensors, the 0.88 converter and the 75 % usable energy "
             "as in every record",
        no_mode_chosen="DECISION_LOG P30", target_min=power.TARGET_MIN, pack=pack.id, modes=modes,
        record_default=dict(rule="the record's own power and runtime: power.COMPUTE_BUDGET's row for the dev kit, "
                                 "round one's 31 W allowance (A13), above either mode's budget",
                            compute_rail_w=sum(power.COMPUTE_BUDGET[sel["compute"]].values())),
        base=dict(compute=base_rec["selection"]["compute"],
                  compute_rail_w=sum(power.COMPUTE_BUDGET[base_rec["selection"]["compute"]].values()),
                  drives_w=base_rec["power"]["drives_w"], total_w=base_rec["power"]["total_w"],
                  runtime_min=base_min),
        survey_typical_15w=dict(
            rule="for comparison, not a mode budget and not a measurement in the 15 W mode: the compute survey's "
                 "typical figure for this architecture, 12 W at the module plus the same 3 W of carrier, storage and "
                 "cooling (designlab/data/research/compute_2026-11-07.json, A_orin_nano_super_devkit: 'Typical 15.0 "
                 "W').  The 12 W was measured with locked MAXN SUPER clocks under a heavier detector load (arXiv "
                 "2609.13551), and the survey has no measurement in the 15 W mode ('No light-load measurement exists "
                 "in the 15 W mode').  The base's row is a typical draw too (A14).  First order: the 15 W mode's "
                 "result with the rail's difference taken off through the converter",
            compute_rail_w=typ_rail, total_w=typ_total, runtime_min=60.0 * pack.capacity_wh * power.USABLE / typ_total,
            runtime_change_against_base_min=60.0 * pack.capacity_wh * power.USABLE / typ_total - base_min))


def jetson_chest(xml, q_home, design) -> dict:
    """``JETSON_CHEST_RULE``: the home-pose readings at the chest and the abdomen of the model at ``xml`` (the Jetson
    body's, or its pick's for the comparison)."""
    import numpy as np

    from . import clearance

    S = clearance._Screen(xml)
    S.set_q(np.asarray(q_home, float))
    items = {it.name: it for it in design.bodies["trunk"].items}

    def gid(name):
        g = S.gid(f"hw_{name}")
        return g if g >= 0 else None

    gaps = {}
    for a, b in JETSON_CHEST_PAIRS:
        ga, gb = gid(a), gid(b)
        if ga is not None and gb is not None:
            gaps[f"{a}|{b}"] = 1e3 * float(S.dist(ga, gb, 1.0))
    nearest = {}
    for name in JETSON_CHEST_ITEMS:
        g = gid(name)
        if g is None:
            continue
        best = (np.inf, None)
        for x in map(int, S.screened):
            if S.geom_grp[x] != S.geom_grp[g]:
                d = float(S.dist(g, x, 1.0))
                if d < best[0]:
                    best = (d, x)
        nearest[name] = dict(geom=S.gname[best[1]], body=S.body_of(best[1]), gap_mm=1e3 * best[0])
    same = clearance._same_body_overlaps(S)
    present = [n for n in JETSON_CHEST_ITEMS if n in items]
    return dict(rule=JETSON_CHEST_RULE, gaps_mm=gaps, nearest_other_body=nearest,
                same_body_component_overlaps=dict(overlaps=same["overlaps"], nested_envelopes=same["nested_envelopes"],
                                                  structure_contacts=same["structure_contacts"]),
                trunk_profile_margin_mm={n: {k: 1e3 * v for k, v in design.profile_margin(items[n]).items()}
                                         for n in present},
                centre_trunk_mm={n: [1e3 * float(v) for v in items[n].pos] for n in present},
                size_mm={n: [1e3 * float(v) for v in items[n].size] for n in present})


def _ledger_delta(rows0: list, rows1: list) -> list:
    """The ledger rows that differ between two builds (mass, centre, size or body), added and removed rows included:
    each with both masses, the change and both centres in its body's frame."""
    a, b = {r["name"]: r for r in rows0}, {r["name"]: r for r in rows1}
    out = []
    for name in list(b) + [n for n in a if n not in b]:
        r0, r1 = a.get(name), b.get(name)
        if r0 is not None and r1 is not None and all(
                json.dumps(r0[k], default=str) == json.dumps(r1[k], default=str)
                for k in ("mass_kg", "com_body", "size", "body")):
            continue
        ref = r1 if r1 is not None else r0
        m0, m1 = (0.0 if r is None else float(r["mass_kg"]) for r in (r0, r1))
        out.append(dict(name=name, body=ref["body"], category=ref["category"],
                        status="added" if r0 is None else "removed" if r1 is None else "changed",
                        mass_kg=[m0, m1], mass_change_kg=m1 - m0,
                        com_body=[None if r is None else [float(v) for v in r["com_body"]] for r in (r0, r1)],
                        size=[None if r is None else [float(v) for v in r["size"]] for r in (r0, r1)],
                        source=[None if r is None else r["source"] for r in (r0, r1)]))
    return out


def _pi_set(base_rec: dict, rows: list) -> dict:
    """``jetson.pi_set``: what the pick's record and ledger hold of the Raspberry Pi set (its compute rows, its power
    rows, and any row naming a supply, a converter, an SSD or a HAT), with ``JETSON_PI_SET_DECISION``."""
    tokens = ("ssd", "nvme", "5 v", "buck", "regulator", "dc-dc", "dcdc", "converter", "hat", "rpi", "pi 5")
    row = lambda r: dict(name=r["name"], body=r["body"], category=r["category"], mass_kg=float(r["mass_kg"]),  # noqa: E731
                         source=r["source"])
    named = [row(r) for r in rows if r["category"] != "compute"
             and any(t in f"{r['name']} {r['source']}".lower() for t in tokens)]
    from . import power

    loads = base_rec["power"]["aux"]["loads"]
    own = power.COMPUTE_BUDGET[base_rec["selection"]["compute"]]
    return dict(compute_part=base_rec["selection"]["compute"],
                compute_rows=[row(r) for r in rows if r["category"] == "compute"],
                compute_power_rows_w={k: loads[k] for k in own},
                other_power_rows_w={k: v for k, v in loads.items() if k not in own},
                other_rows_naming_a_supply_converter_or_storage=named, searched_for=list(tokens),
                decision=JETSON_PI_SET_DECISION)


def jetson_one(case: str, cid: str, out: str) -> dict:
    """The sized record of the Jetson body: ``jetson_build`` into ``<out>/models``, then ``<out>/candidates/<id>.json``
    with ``round2.size_one``'s fields, the case's settings (as ``_size`` adds them) and ``jetson``: the rule, the base
    record it builds on (path and sha256, its model's sha256), the parts it changes and adds against the pick's ledger,
    the regulator's choice, what leaves with the Pi set, the home-pose readings at the chest beside the pick's, its
    loads and margins beside the base's at both homes, its power and runtime in each power mode, the Jetson-direct
    parts it does not place, and what one selection step on its loads would pick (recorded only)."""
    import hashlib
    import traceback

    import designlab  # noqa: F401

    from . import catalog, gait, metrics, mjcf, power, round2, sizing
    from .stage_a import _mass_by_category, _mass_by_volume_class

    gait.set_accuracy(round2.GAIT_ACCURACY)
    t0 = time.time()
    c = CASES[case]
    try:
        import numpy as np

        bp, base = jetson_base(case, out)
        r = jetson_build(case, base, Path(out) / "models")
        d, sel, cand, ev, req = r["design"], r["selection"], r["candidate"], r["evaluation"], r["requirements"]
        meas = metrics.measure(r["model"], d, r["q_home"])

        def read_pick(pk, xml):
            M0, q0 = pk["model"], np.asarray(pk["q_home"], float)
            M0.forward(q0)
            return dict(rows=pk["rows"], com=[float(v) for v in M0.com()], design=pk["design"],
                        chest=jetson_chest(xml, q0, pk["design"]))

        at_pick = head_x_at_pick_pose(base, r, sel, rule=JETSON_HOME_RULE, pick_reader=read_pick)
        pick = at_pick["pick"]
        delta = _ledger_delta(pick["rows"], r["rows"])
        # the parts' sum: the pick's mass and centre with the changed rows' mass and first moment exchanged (every
        # changed row is on the trunk, whose pose the two homes share: home_pose.trunk_pose_change_max)
        m0 = sum(float(x["mass_kg"]) for x in pick["rows"])
        m1 = sum(float(x["mass_kg"]) for x in r["rows"])
        moment = np.zeros(3)
        for row in delta:
            if row["body"] != "trunk":
                raise ValueError(f"{row['name']}: a changed ledger row off the trunk ({row['body']})")
            for sign, mass, com in ((-1.0, row["mass_kg"][0], row["com_body"][0]), (1.0, row["mass_kg"][1],
                                                                                   row["com_body"][1])):
                if com is not None:
                    moment += sign * mass * d.sk.to_world(np.asarray(com, float))
        com_parts = (m0 * np.asarray(pick["com"]) + moment) / m1
        chest = jetson_chest(Path(out) / "models" / f"{cid}.xml", r["q_home"], d)
        pick_one = sizing.select(req, mass=r["summary"]["mass"], layout=cand.leg_layout)
        current = dict(sel["joints"], neck=sel["neck"], tail=sel["tail"])
        pmodes = jetson_power(d, r, req, sel, base)
        pw = ev["power"]
        peak_aux = lambda p: p["peak_drives_w"] + p["aux"]["input_w"]  # noqa: E731
        bat = power.select_battery(pick_one["bus"], pw["total_w"], peak_aux(pw))
        bat_modes = {m: power.select_battery(pick_one["bus"], v["total_w"], pw["peak_drives_w"] + v["aux"]["input_w"])
                     for m, v in pmodes["modes"].items()}
        over = sorted(g for g, m in ev["margins"].items() if max(m["cont_ratio"], m["peak_ratio"], m["speed_ratio"]) > 1.0)
        kit, pi, reg = (catalog.part(x) for x in (JETSON_COMPUTE, base["selection"]["compute"], JETSON_REGULATOR))
        adapter = catalog.part(JETSON_DIRECT_ADAPTER)
        reserve = mjcf.INTEGRATION_RESERVE_FRACTION / (1 - mjcf.INTEGRATION_RESERVE_FRACTION)
        part = lambda x: dict(part=x.id, model=x.model, mass_kg=x.mass, dims_m=list(x.dims), price_usd=x.price,  # noqa: E731
                              power_typ_w=x.power_typ, power_max_w=x.power_max, basis=x.basis)
        jetson = dict(
            rule=JETSON_RULE, case=case, base_case=c["base_case"], base_record=bp.as_posix(),
            base_record_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(), base_xml_sha256=base["xml_sha256"],
            decision="DECISION_LOG P30 (14 Nov 2026): the Jetson is studied as a refine case; P14's Raspberry Pi 5 set "
                     "stays adopted; the walking loop stays on the microcontroller; no power mode is chosen",
            compute=dict(part(kit), replaces=part(pi), placed="the chest tray, as build.py places a compute part"),
            regulator=dict(jetson_regulators(), geometry=d.geometry_checks["compute_regulator"]),
            overrides_added=[list(o) for o in JETSON_OVERRIDES],
            selection_changed=dict(compute=[base["selection"]["compute"], sel["compute"]],
                                   compute_regulator=[None, sel["compute_regulator"]]),
            control_layer=dict(kept=["control_stack (Teensy 4.1, CAN x2, IMU), 0.18 kg", "imu", "the 4 W MCU row"],
                               rule="the owner's baseline (P30): Jetson + microcontroller"),
            pi_set=_pi_set(base, pick["rows"]),
            not_placed=dict(rule=JETSON_NOT_PLACED_RULE, kind="estimate",
                            usb_can_adapter=dict(part(adapter), count_per_can_bus=1,
                                                 would_add_kg={"1": adapter.mass, "2": 2 * adapter.mass},
                                                 with_reserve_kg={"1": adapter.mass * (1 + reserve),
                                                                  "2": 2 * adapter.mass * (1 + reserve)}),
                            supervisor=dict(part=None, would_add_kg=None, note="no catalog part")),
            parts=delta, parts_mass_change_kg=sum(row["mass_change_kg"] for row in delta),
            ledger_mass_kg=dict(jetson=m1, base=m0, change=m1 - m0),
            mass_kg=dict(jetson=meas["mass"], base=base["metrics"]["mass"],
                         change=meas["mass"] - base["metrics"]["mass"]),
            com_at_pick_pose_m=dict(jetson=at_pick["com_at_pick_pose_m"], base=pick["com"],
                                    change=[a - b for a, b in zip(at_pick["com_at_pick_pose_m"], pick["com"],
                                                                  strict=True)],
                                    from_parts=[float(v) for v in com_parts],
                                    from_parts_change=[float(a - b) for a, b in zip(com_parts, pick["com"], strict=True)],
                                    note="both bodies at the pick's home pose; from_parts: the pick's mass and centre "
                                         "with the changed ledger rows' masses and first moments exchanged"),
            com_ahead_of_hip_m=dict(jetson=meas["com_ahead_of_hip"], base=base["metrics"]["com_ahead_of_hip"],
                                    change=meas["com_ahead_of_hip"] - base["metrics"]["com_ahead_of_hip"],
                                    jetson_at_pick_pose=at_pick["com_ahead_of_hip_at_pick_pose_m"],
                                    change_at_pick_pose=at_pick["com_ahead_of_hip_at_pick_pose_m"]
                                    - base["metrics"]["com_ahead_of_hip"],
                                    note="change: each body at its own home (two stances); change_at_pick_pose: the "
                                         "Jetson body at the pick's home (home_pose)"),
            com_height_m=dict(jetson=meas["com_height"], base=base["metrics"]["com_height"],
                              change=meas["com_height"] - base["metrics"]["com_height"]),
            home_pose=at_pick["home_pose"],
            exterior_m={k: dict(jetson=meas[k], base=base["metrics"][k], change=meas[k] - base["metrics"][k])
                        for k in ("crown", "max_width", "torso_width", "hardware_width", "thigh_width", "length")},
            trunk_widened_for_hardware=dict(jetson=d.geometry_checks.get("trunk_widened_for_hardware"),
                                            base=base["geometry_checks"].get("trunk_widened_for_hardware")),
            chest=dict(jetson=chest, base=pick["chest"]),
            groups_over_planning_limit=over,
            meets_case_headroom={g: bool(sizing.meets(d.act(g), req["groups"][g],
                                                      sel["bus"] if d.act(g).box is None else None, g))
                                 for g in req["groups"]},
            against_base=_against(dict(margins=ev["margins"]), base),
            against_base_at_pick_pose=(_against(dict(margins=at_pick["margins_at_pick_pose"]), base)
                                       if "margins_at_pick_pose" in at_pick
                                       else dict(error=at_pick["margins_at_pick_pose_error"])),
            loads_between_stances=(_stance_effect(_against(dict(margins=ev["margins"]), base),
                                                  _against(dict(margins=at_pick["margins_at_pick_pose"]), base))
                                   if "margins_at_pick_pose" in at_pick else None),
            power_modes=pmodes,
            heat=dict(modelled=False,
                      note="no thermal model of the chest, the electronics or the air path exists in this study; the "
                           "drives' heating is read only through their sustained-torque ratios (the envelope "
                           "screen's session_heat).  The dev kit's 15-25 W module (18-28 W at the rail) against the "
                           "Pi set's 8-9.5 W, in a closed chest, is an open hardware question"),
            selection_if_resized=dict(
                rule="one step of sizing.select on these loads with the case's headroom, and power.select_battery "
                     "on its walking power (the record's own, and each power mode's): recorded only (the sizing loop, "
                     "which would iterate on the re-selected parts' masses, is not run, and nothing here is "
                     "re-selected)",
                choice=pick_one["choice"], bus=pick_one["bus"], battery=bat["pack"], notes=pick_one["notes"],
                differs={g: [current.get(g), p] for g, p in pick_one["choice"].items() if current.get(g) != p},
                battery_differs=bat["pack"] != sel["battery"], bus_differs=pick_one["bus"] != sel["bus"],
                battery_by_mode={m: dict(pack=b["pack"], differs=b["pack"] != sel["battery"],
                                         meets_target=b["meets_target"]) for m, b in bat_modes.items()}))
        doc = dict(id=cid, rev=round2.REV, candidate=dict(height=cand.height, width=cand.width, family=cand.family,
                                                          posture=cand.posture, proportion=cand.proportion,
                                                          leg_layout=cand.leg_layout, battery_pos=cand.battery_pos),
                   B=d.B, H=d.sk.H, xml_sha256=r["summary"]["xml_sha256"], selection=dict(sel), metrics=meas,
                   margins=ev["margins"], holds=ev["holds"], power=ev["power"], runtime=ev["runtime"], cost=ev["cost"],
                   gaits=ev["gaits"], sizing_history=ev["sizing_history"], walk_feasible=ev["walk_feasible"],
                   geometry_checks=d.geometry_checks, notes=d.notes,
                   overrides=[list(o) for o in cand.overrides if o[0] != "rev"], packaging_search=[],
                   mass_by_category=_mass_by_category(r["rows"]), mass_by_volume_class=_mass_by_volume_class(r["rows"]),
                   gait_accuracy=gait.ACCURACY, seconds=time.time() - t0, ok=True,
                   refine_case=case, headroom=c["headroom"], headroom_by_group=dict(c.get("headroom_by_group") or {}),
                   fr05_sizing=c["fr05"], turn_min_gap_mm=c["min_gap"], jetson=jetson)
    except Exception as e:  # keep failed cases, as round2.size_one does
        doc = dict(id=cid, rev=round2.REV, ok=False, error=repr(e), traceback=traceback.format_exc(),
                   seconds=time.time() - t0, refine_case=case)
    Path(out, "candidates").mkdir(parents=True, exist_ok=True)
    Path(out, "candidates", f"{cid}.json").write_text(json.dumps(doc, indent=1, default=round2._json))
    return dict(id=cid, ok=doc["ok"], seconds=doc["seconds"])


# ------------------------------------------------------------------------------------------------
# Carpet swing height
# ------------------------------------------------------------------------------------------------
def carpet_lift_one(a) -> dict:
    """One body, every lift in ``LIFTS``: drives, clearance and toe catch of the lifted walks.

    ``a = (id, case dir, scratch dir[, lifts[, nominal_only]])``; with ``nominal_only`` only the nominal walk is
    lifted and the brisk walk stays at its reference lift (the carpet gait walks at the nominal speed)."""
    import numpy as np

    from . import carpet, clearance, round2, sizing
    from .pipeline import build_candidate

    cid, r2dir, scratch = a[:3]
    lifts = a[3] if len(a) > 3 else LIFTS
    nominal_only = bool(a[4]) if len(a) > 4 else False
    _init(None)
    t0 = time.time()
    rec = json.loads(Path(r2dir, "candidates", f"{cid}.json").read_text())
    res = build_candidate(round2.candidate(cid, rec.get("overrides", ())), Path(scratch), selection=dict(rec["selection"]),
                          B=rec["B"], write=False)
    M, q0, design = res["model"], res["q_home"], res["design"]
    H, layout, bus = design.sk.H, design.cand.leg_layout, rec["selection"]["bus"]
    xml = str(Path(scratch) / f"{cid}.xml")
    piles = carpet.pile_classes()
    unders = carpet.foot_underside(M, q0, "whole_foot")
    nominal0, brisk0 = sizing.NOMINAL, sizing.BRISK
    out = dict(id=cid, H_m=H, carpet_target_mm=1e3 * carpet.carpet_target(H), reference_lift=[nominal0.lift_frac, brisk0.lift_frac],
               lifted="nominal walk only" if nominal_only else "nominal and brisk walks", lifts={})
    try:
        for lift in lifts:
            sizing.NOMINAL = replace(nominal0, lift_frac=lift)
            if not nominal_only:
                sizing.BRISK = replace(brisk0, lift_frac=lift)
            req = sizing.requirements(M, q0, H, layout)
            ratios = {}
            for g, r in req["groups"].items():
                act = design.act(g)
                spd = act.speed_at(bus if act.box is None else None)
                ratios[g] = max(r["cont"] / act.sustained, r["peak_req"] / act.peak, r["speed_req"] / max(spd, 1e-6))
            worst = max(ratios.items(), key=lambda t: t[1])
            trajs = {k: v["traj"] for k, v in req["runs"].items()}
            cl = clearance.screen(xml, q0, trajs, H=H, gates="r2")["summary"]
            mg = cl.get("joint_range_margins") or {}
            tight = min(((j, v) for j, _, v in clearance.gated_margins(mg)), key=lambda t: t[1], default=(None, None))
            toe = {}
            for name, tr in trajs.items():
                idr = req["runs"][name]["idr"]
                F = float(np.max(idr["normal"]))
                sink = {p: carpet.pile_sinkage(unders["left"], pc["stiffness_pa_per_m"], F) for p, pc in piles.items()}
                tc = carpet.toe_catch(M, tr, H, piles, sink)
                toe[name] = dict(above_pile_top_mm={p: v["mid_swing_min_above_pile_top_mm"] for p, v in tc["piles"].items()},
                                 meets_carpet_target={p: v["meets_carpet_target"] for p, v in tc["piles"].items()},
                                 hard_mid_swing_mm=tc["hard_floor"]["mid_swing_min_mm"])
            out["lifts"][f"{lift:g}"] = dict(
                drives_within_limits=all(v <= 1.0 + 1e-9 for v in ratios.values()), worst_group=worst[0],
                worst_ratio=worst[1], ratios=ratios, ik_ok=all(req["gaits"][k]["ik_ok"] for k in req["gaits"]),
                clearance_pass=cl["pass"], walking_gates=all(cl["pass"].get(k) is True for k in round2.WALK_GATES),
                tightest_joint=tight, toe=toe,
                meets_target_all_piles=all(all(t["meets_carpet_target"].values()) for t in toe.values()),
                min_above_pile_top_mm=min(v for t in toe.values() for v in t["above_pile_top_mm"].values()),
                walk_power_hint_w=None)
    finally:
        sizing.NOMINAL, sizing.BRISK = nominal0, brisk0
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------------------------------------
# Same drives: what a lighter body buys when the drives are kept
# ------------------------------------------------------------------------------------------------
#: reference case whose drives and pack a material/power case keeps in the same-drive pass
FIXED_REF = {"mat60": "carpet_60", "mat75": "targeted_75"}


def fixed_one(a) -> dict:
    """One material or power case rebuilt with its reference case's drives and pack (no re-selection).

    Re-sizing a lighter body selects lighter drives, so its margins stay near the limits and the saving shows up
    as drive mass and cost.  Keeping the reference drives shows the other use of the saving: margin.  Same
    sizing walks as the case (``_init``), same ratio definitions as the sizing record (continuous / sustained,
    peak / peak, speed / speed at the bus).  Clearance is not re-run: the shapes are the reference body's, apart
    from the covering wall thickness."""
    from . import cost, power, round2, sizing
    from .pipeline import build_candidate

    case, cid, out, scratch = a
    _init(case)
    t0 = time.time()
    ref_case = FIXED_REF[case.split("_")[0]]
    ref = json.loads(Path(out, ref_case, "candidates", f"{cid}.json").read_text())
    rec_p = Path(out, case, "candidates", f"{cid}.json")
    if rec_p.exists():  # re-sized: the case's own overrides, packaging-search changes included
        ov = tuple(tuple(o) for o in json.loads(rec_p.read_text()).get("overrides", ()))
    else:  # not re-sized: the reference body's overrides with the case's levers on top
        merged = dict(tuple(o) for o in ref.get("overrides", ()))
        merged.update(dict(CASES[case]["overrides"]))
        ov = tuple(merged.items())
    sel = dict(ref["selection"])
    cand = round2.candidate(cid, ov)
    res = build_candidate(cand, Path(scratch), selection=sel, write=False)
    M, q0, design = res["model"], res["q_home"], res["design"]
    req = sizing.requirements(M, q0, design.sk.H, cand.leg_layout)
    ratios = {}
    for g, r in req["groups"].items():
        act = design.act(g)
        spd = act.speed_at(sel["bus"] if act.box is None else None)
        ratios[g] = dict(actuator=act.id, cont=r["cont"] / act.sustained, peak=r["peak_req"] / act.peak,
                         speed=r["speed_req"] / max(spd, 1e-6), hold=r["hold"])
    worst = max(((g, k, v[k]) for g, v in ratios.items() for k in ("cont", "peak", "speed")), key=lambda t: t[2])
    leg_worst = max(((g, k, v[k]) for g, v in ratios.items() if g not in ("neck", "tail")
                     for k in ("cont", "peak", "speed")), key=lambda t: t[2])
    nom = req["runs"]["nominal"]["idr"]
    ms = sizing._motor_space(M.names, nom["tau"], nom["omega"], cand.leg_layout)
    pw = power.walking_power(design, req, ms)
    tether = getattr(design, "power_mode", "battery") == "tether"
    rt = None if tether else power.runtime(power.catalog.part(sel["battery"]), pw["total_w"])
    ref_m = ref["margins"]
    ref_worst = max(((g, k, m[f"{k}_ratio"]) for g, m in ref_m.items() for k in ("cont", "peak", "speed")),
                    key=lambda t: t[2])
    ref_leg = max(((g, k, m[f"{k}_ratio"]) for g, m in ref_m.items() if g not in ("neck", "tail")
                   for k in ("cont", "peak", "speed")), key=lambda t: t[2])
    return dict(case=case, id=cid, reference_case=ref_case, label=CASES[case]["label"], mass_kg=M.mass(),
                reference_mass_kg=ref["metrics"]["mass"], com_ahead_of_hip_m=_com_ahead(M, q0, design),
                reference_com_ahead_of_hip_m=ref["metrics"].get("com_ahead_of_hip"),
                worst=worst, leg_worst=leg_worst, reference_worst=ref_worst, reference_leg_worst=ref_leg,
                ratios=ratios, holds={g: r["hold"] for g, r in req["groups"].items()},
                reference_holds={g: m["hold"] for g, m in ref_m.items()},
                walk_w=pw["total_w"], reference_walk_w=ref["power"]["total_w"],
                runtime_min=None if rt is None else rt["minutes"], reference_runtime_min=ref["runtime"]["minutes"],
                tether=tether, walk_feasible=req["walk_feasible"],
                cost=cost.estimate(design, res["rows"], sel), reference_cost=ref["cost"],
                ik_ok=all(req["gaits"][k]["ik_ok"] for k in req["gaits"]), seconds=time.time() - t0)


def _com_ahead(M, q0, design) -> float:
    from . import metrics

    return float(metrics.measure(M, design, q0)["com_ahead_of_hip"])


def fixed_markdown(rows: list) -> str:
    lines = ["", "## Same drives: the material and power cases with the reference body's drives and pack", "",
             "Each case rebuilt with the drives and pack of its reference (60 cm: `carpet_60`; 75 cm: `targeted_75`) "
             "instead of being re-sized.  Ratios use the same sizing walks; clearance is not re-run.", "",
             "| Body | Case | Mass (reference) | CoM ahead of hip | Governing leg drive (reference) | Governing drive, any "
             "(reference) | Hip-pitch hold | Walking power | Runtime | Cost nominal (reference) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lw, rlw, w, rw = r["leg_worst"], r["reference_leg_worst"], r["worst"], r["reference_worst"]
        hp, rhp = r["holds"].get("hip_pitch"), r["reference_holds"].get("hip_pitch")
        rt = "tethered" if r["tether"] else f"{r['runtime_min']:.1f} min ({r['reference_runtime_min']:.1f})"
        lines.append(
            f"| `{r['id']}` | {r['label']} | {r['mass_kg']:.2f} kg ({r['reference_mass_kg']:.2f}) | "
            f"{1e3 * r['com_ahead_of_hip_m']:.0f} mm ({1e3 * r['reference_com_ahead_of_hip_m']:.0f}) | "
            f"{lw[0]} {lw[1]} {lw[2]:.2f} ({rlw[0]} {rlw[1]} {rlw[2]:.2f}) | {w[0]} {w[1]} {w[2]:.2f} "
            f"({rw[0]} {rw[1]} {rw[2]:.2f}) | {hp:.2f} N·m ({rhp:.2f}) | {r['walk_w']:.1f} W ({r['reference_walk_w']:.1f}) | "
            f"{rt} | ${r['cost']['nominal'] / 1e3:.1f}k (${r['reference_cost']['nominal'] / 1e3:.1f}k) |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------
# Plates: round two against the refined bodies
# ------------------------------------------------------------------------------------------------
PLATE_PAIRS = (("60-M-T-10", "carpet_60"), ("60-M-H-10", "carpet_60"), ("75-M-T-10", "targeted_75"))
#: The recommended verification set (scorecard version 2 picks, ROUND2_REVIEW.md section 6).
PICK_PAIRS = (("60-N-T-10", "narrow_carpet_60"), ("75-M-H-10", "feet_75"), ("60-N-H-10", "narrow_carpet_60"))


def plates_refined(out: Path, r2: Path = Path("round2"), pairs=PLATE_PAIRS, stem: str = "refined",
                   cut_title: str = "Installed hardware: 60 cm ankle drives lowered to 0.60 of the shank; 75 cm drive upgrade",
                   side_title: str = "Round two against the refined bodies") -> list:
    """Side exterior and side cutaway grids: each row one body, round two on the left, the refined body on the
    right, at one physical scale (``plates.matrix``)."""
    from . import plates

    grid = [[str(r2 / "models" / f"{cid}.xml"), str(out / case / "models" / f"{cid}.xml")] for cid, case in pairs]
    labels = [f"{cid} ({case})" for cid, case in pairs]
    P = out / "plates"
    P.mkdir(parents=True, exist_ok=True)
    made = [plates.matrix(grid, labels, ["Round two", "Refined overnight"], str(P / f"{stem}_side.png"), view="side",
                          mode="exterior", title=side_title),
            plates.matrix(grid, labels, ["Round two", "Refined overnight"], str(P / f"{stem}_cutaway.png"),
                          view="side", mode="cutaway", title=cut_title)]
    return made


# ------------------------------------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------------------------------------
def compare(out: Path, r2: Path = Path("round2")) -> dict:
    from .shortlist import detail
    from .sizestudy import row

    def load(base: Path, cid: str):
        p = base / "candidates" / f"{cid}.json"
        if not p.exists():
            return None, {}
        idx = base / "review" / "review_index.json"
        rv = json.loads(idx.read_text()).get(cid, {}) if idx.exists() else {}
        return json.loads(p.read_text()), dict(rv, _detail=detail(base, cid))

    rows = []
    ids = sorted({cid for c in CASES.values() for cid in c["ids"]})
    for cid in ids:
        d, rv = load(r2, cid)
        if d:
            rows.append(dict(case="round2", **row("round two", d, rv, None, rv.get("_detail"))))
    for case, c in CASES.items():
        if c.get("screen") is False:  # sized only (identical to another case's body)
            continue
        for cid in c["ids"]:
            d, rv = load(out / case, cid)
            ref, _ = load(r2, cid)
            if d:
                rows.append(dict(case=case, **row(c["label"], d, rv, ref, rv.get("_detail"))))
    lift = json.loads((out / "carpet_lift.json").read_text()) if (out / "carpet_lift.json").exists() else None
    fixed = json.loads((out / "fixed_drives.json").read_text()) if (out / "fixed_drives.json").exists() else None
    return dict(cases={k: v["label"] for k, v in CASES.items()}, rows=rows, carpet_lift=lift, fixed_drives=fixed)


def markdown(cmp: dict) -> str:
    from .sizestudy import markdown as md_rows

    text = md_rows(dict(rows=cmp["rows"])).replace("# Size study (round two)", "# Refinement experiments (round two)")
    text = text.replace("Generated by `python -m designlab.sizestudy`.", "Generated by `python -m designlab.refine`.")
    lift = cmp.get("carpet_lift")
    if lift:
        lines = ["", "## Carpet swing height (lifted reference walks, the selected drives)", "",
                 "| Body | Carpet target | Lift (H) | Drives within limits (governing) | Walking clearance gates | "
                 "Tightest joint margin | Lowest clearance above a pile top | Meets the target on every pile |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in lift:
            for L, v in r["lifts"].items():
                tj = v["tightest_joint"]
                lines.append(f"| `{r['id']}` | {r['carpet_target_mm']:.1f} mm | {L} | "
                             f"{'yes' if v['drives_within_limits'] else '**no**'} ({v['worst_group']} {v['worst_ratio']:.2f}) | "
                             f"{'pass' if v['walking_gates'] else '**fail**'} | "
                             f"{'–' if tj[0] is None else f'{tj[0]} {tj[1]:.2f} rad'} | {v['min_above_pile_top_mm']:.1f} mm | "
                             f"{'yes' if v['meets_target_all_piles'] else 'no'} |")
        text += "\n".join(lines) + "\n"
    if cmp.get("fixed_drives"):
        text += fixed_markdown(cmp["fixed_drives"])
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="round2/refine")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--carpet-lift", action="store_true")
    ap.add_argument("--lift-cases", nargs="*", default=None, help="run the carpet swing pass on these cases' bodies")
    ap.add_argument("--lifts", nargs="*", type=float, default=None, help="swing lifts (fraction of H) for that pass")
    ap.add_argument("--nominal-only", action="store_true", help="carpet swing pass: lift the nominal walk only")
    ap.add_argument("--fixed", action="store_true", help="same-drive pass over the material and power cases")
    ap.add_argument("--fixed-cases", nargs="*", default=None, help="limit the same-drive pass to these cases (merged)")
    ap.add_argument("--plates", action="store_true", help="draw round two against the refined bodies")
    ap.add_argument("--plates-picks", action="store_true", help="draw round two against the recommended bodies")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip bodies already sized or screened")
    a = ap.parse_args()
    import designlab  # noqa: F401

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    if not a.report_only:
        cases = a.cases if a.cases is not None else list(CASES)
        tasks = [(c, cid, str(out / c)) for c in cases for cid in CASES[c]["ids"]]
        with ctx.Pool(a.workers) as pool:
            if tasks:
                # --resume: keep sized records and finished screens from an interrupted run
                size_tasks = [t for t in tasks if not (a.resume and (Path(t[2]) / "candidates" / f"{t[1]}.json").exists())]
                for r in pool.imap_unordered(_size, size_tasks):
                    print("sized", r, flush=True)
                screen_tasks = [t for t in tasks if not (a.resume and (Path(t[2]) / "review" / "screening"
                                                                       / f"{t[1]}_envelope.json").exists())]
                results = list(pool.imap_unordered(_screen, screen_tasks))
                if a.resume:  # keep the earlier screens' index entries
                    for c in cases:
                        idx = out / c / "review" / "review_index.json"
                        if idx.exists():
                            done = json.loads(idx.read_text())
                            results += [dict(case=c, **v) for k, v in done.items()
                                        if not any(r["case"] == c and r["id"] == k for r in results)]
                for c in cases:
                    idx = out / c / "review" / "review_index.json"
                    index = json.loads(idx.read_text()) if idx.exists() else {}
                    index.update({r["id"]: {k: v for k, v in r.items() if k != "case"} for r in results if r["case"] == c})
                    idx.parent.mkdir(parents=True, exist_ok=True)
                    idx.write_text(json.dumps(index, indent=1, default=str))
                for r in results:
                    print("screened", r["case"], r["id"], "sha ok" if r.get("matches_sized") else "SHA DIFFERS",
                          {k: v for k, v in r.items() if k.endswith("_error") or k == "error"}, flush=True)
            if a.carpet_lift:
                scratch = out / "carpet_lift_models"
                todo = [(cid, "round2", str(scratch)) for cid in CARPET_LIFT_IDS] if not a.lift_cases else [
                    (cid, str(out / c), str(scratch / c)) for c in a.lift_cases for cid in CASES[c]["ids"]]
                lifts = tuple(a.lifts) if a.lifts else LIFTS
                lift = list(pool.imap(carpet_lift_one, [t + (lifts, a.nominal_only) for t in todo]))
                for r, t in zip(lift, todo, strict=True):
                    r["source"] = t[1]
                name = "carpet_lift.json" if not a.lift_cases else f"carpet_lift_{'_'.join(a.lift_cases)}.json"
                if a.lifts:
                    name = name.replace(".json", "_" + "_".join(f"{v:.3f}" for v in lifts) + ".json")
                if a.nominal_only:
                    name = name.replace(".json", "_nominal_only.json")
                (out / name).write_text(json.dumps(lift, indent=1, default=str))
                print("carpet lift done", flush=True)
            if a.fixed:
                names = a.fixed_cases or [c for c in CASES if c.split("_")[0] in FIXED_REF]
                todo = [(c, cid, str(out), str(out / "fixed_models" / c)) for c in names
                        for cid in CASES[c]["ids"]
                        if (out / FIXED_REF[c.split("_")[0]] / "candidates" / f"{cid}.json").exists()]
                fixed = list(pool.imap(fixed_one, todo))
                fp = out / "fixed_drives.json"
                if a.fixed_cases and fp.exists():  # merge: keep the other cases' rows, in CASES order
                    new_keys = {(r["case"], r["id"]) for r in fixed}
                    keep = [r for r in json.loads(fp.read_text()) if (r["case"], r["id"]) not in new_keys]
                    order = {c: k for k, c in enumerate(CASES)}
                    fixed = sorted(keep + fixed, key=lambda r: (order.get(r["case"], 999), r["id"]))
                fp.write_text(json.dumps(fixed, indent=1, default=str))
                print("same-drive pass done", flush=True)
    if a.plates:
        for m in plates_refined(out):
            print("plate", m, flush=True)
    if a.plates_picks:
        for m in plates_refined(out, pairs=PICK_PAIRS, stem="picks", side_title="Round two against the recommended bodies",
                                cut_title="Installed hardware of the recommended bodies"):
            print("plate", m, flush=True)
    cmp = compare(out)
    (out / "refine.json").write_text(json.dumps(cmp, indent=1, default=str))
    (out / "REFINE.md").write_text(markdown(cmp))
    print(markdown(cmp))


if __name__ == "__main__":
    main()
