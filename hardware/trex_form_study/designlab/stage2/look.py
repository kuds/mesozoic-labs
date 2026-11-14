"""Stage 2 Track A: the look ladder (``STAGE2_PLAN.md`` section 4 A; ``stage2/research/research_look.md`` 4.1).

Each body's reference walk is rebuilt from its record exactly as Stage 1's references are (``stage1.refs.source``),
then generated at round two's accuracy (``gait.set_accuracy("r2")``) with ``gait.GaitSpec``'s look fields added one
feature group at a time, so the owner sees what each feature buys (look gate L1).  Simulation only: a prescribed
reference with perfect tracking, priced by the Stage 0 screen before any render; nothing is adopted (P24).

Bodies and speeds (``BODIES``, ``SPEEDS``)
-----------------------------------------
* ``60-N-T-10`` (60 cm Tapered) and ``75-M-H-10`` (75 cm Hybrid), current heads (neck pitch, tail yaw).
* ``nominal``: ``sizing.NOMINAL`` (Froude 0.012).  ``fr05``: ``envelope.FAST`` (Froude 0.05, the animal pace).
* ``v040``: 0.40 m/s, the agility bar's speed (``speed_spec``).  Its stride, duty and lift are the gait that Stage 1's
  ``brisk_to_fr05`` speed change passes through at 0.40 m/s: ``transitions.py`` mixes the period, duty and lift of the
  two walks with the same weight as their speeds, so w = (0.40 - v_brisk) / (v_fr05 - v_brisk) and T, duty and lift
  are mixed with w (60 cm: w 0.651, T 0.751 s, stride 0.742 H, duty 0.567; 75 cm: w 0.338, T 0.861 s, stride
  0.686 H, duty 0.573; lift 0.11 H, both walks' lift).  It is a reference the study already passes through, between
  two screened walks.

The ladder (``VARIANTS``; research 4.1, one feature group per rung)
------------------------------------------------------------------
V0 today's walk; V1 + F1/F2 (bob, weight acceptance); V2 + F3 (swing time); V3 + F4/F5 (lift shape, toe-off);
V4 + F6 (trunk roll); V5 + F9/F10 (tail phase, stabilised head); V6 + F14 (variation), and at 75 cm F13 (a longer
stride at the animal pace).  The look rungs run 12 multibody-ZMP correction passes (``LOOK_ZMP_ITERS``; today's 6):
the bob and the toe-off slow the correction's convergence, and 12 passes bring it back to about today's error; V0 run
at 12 passes measures what the extra passes alone change (``zmp_passes_check``, in ``look_refs.json``'s ``checks``).
Extras (``EXTRAS``): V1B50 and V1B100, V1 with the bob at 50 % and 100 % of the enforced shortening (research F1's
screen); V3T, V3 with F5 at the research's middle (0.40 rad by lift-off at the nominal walk: its knee-speed cost);
V4L, V4 with the roll's other direction (the swing side drops, as human pelvic list; the direction is the owner's
call, research F6); V5F13 (75 cm, animal pace only), V5 with F13's stride and its own tail phase (a tail check on V4
with that stride), from which V6 adds F14 alone there; and V6S, V6 with F7 (the single-support ZMP moved inward; sway
reduction, optional).  F8 (no trunk pitch) is today's; F11 and F12 are deferred, and so are research 4.1's walking
turn and start-stop and plan 4 A.3's 90-degree turn from 0.40 m/s and two-step stop (``DEFERRED``).

Values (``feature_values``; research 3.1's starting values, the middle of each range unless ``VALUES`` says why not)
-------------------------------------------------------------------------------------------------------------------
* F1 ``bob``: 75 % (the middle of 50-100 %) of the stance-leg shortening the flat walk enforces,
  (duty stride / 2)^2 / (2 (1 - crouch) H), with V0's duty and stride at that speed (0.0140 H nominal and 0.0259 H at
  the animal pace, the research's 0.014 and 0.026 H), so the bob stays the same from V1 to V6 at a speed.  F3's longer
  stance and F13's stride enforce more shortening, so the same bob is a smaller share of the rung's own: ``measure``'s
  ``bob_share_of_enforced`` (57 % on the nominal V2+, 55 % on 75 cm animal-pace V6).
* F2 ``bob_skew`` 0.125 of stance (10-15 %) and ``bob_drop`` 0.10 s (research F2's "settling over 0.1-0.15 s"; the
  reviewer's "about 0.1 s"): the hips hold the crest through late single support, drop over 0.1 s to the trough and
  rise slowly to the next mid-stance.  Where the stance is short the drop starts before the touchdown (half of the dip
  comes after it at the animal pace, nearly all at the nominal walk): a 0.1 s drop of the animal-pace bob entirely
  after the touchdown would accelerate the hips down at more than g.
* F3 ``swing_time``: the animal-pace walk's swing (1 - 0.56) T_fr05, 0.320 s at 60 cm and 0.356 s at 75 cm (the
  research's 0.32 / 0.36 s).  At the nominal walk it gives duty 0.69, outside the spec's 0.55-0.65: those rungs need
  owner decision 2 (the band widened to 0.72 below 0.30 m/s), else F3 starts at the brisk walk (``flags``).
* F4 ``lift_peak`` 0.375 (35-40 %) with ``gait.lift_profile``'s descent, which keeps 0.46 of the lift at 75 % of the
  swing (the old warp kept 0.28 and failed the clearance screen's 0.04 H).
* F5 ``toe_off`` 0.30 rad (the research's 0.3-0.5, its low end), ``toe_off_start`` 0.825 (80-85 %), ``toe_reach``
  0.075 rad (0.05-0.10); the roll also waits until the ZMP reference is 5 mm inside the rolling support
  (``toe_off_margin``, not in the research: a rigid foot on its toe carries load only there), peaks at the lift-off
  and only falls in the air (``gait.toe_off_bump``: no heel flick).  The rigid foot rolls on its real convex sole
  (``gait.roll_offset``): about its toe pad's front edge, then over a short transition arc (274 / 410 mm radius over
  23 / 28 mm, the real foot at most 0.23 mm above the floor meanwhile) on to the middle claw's tip, which reaches the
  floor at 0.082 rad (60 cm) / 0.068 rad (75 cm); the plant has no contact there, a hardware question for the owner
  (``CLAW_QUESTION``).  Rolling straight from the edge on to the claw at a finite rate would be an impact (the foot's
  velocity jumps); the arc keeps it C1 and no part of the foot goes below the floor.
* F5's caps, ``toe_off_rate`` 3.4 rad/s and ``toe_off_accel`` 220 rad/s^2 (not in the research): the 60 cm nominal
  walk's roll (0.30 rad over its 93 ms on the floor, min-jerk: 3.2 rad/s and 199 rad/s^2), with room for F14's
  step-time jitter (a roll up to 4 % shorter), so both nominal walks reach 0.30 rad by lift-off on every step.  With
  F3's short double support the roll has 23-38 ms on the floor at 0.40 m/s and the animal pace; at the nominal roll's
  rate and acceleration it reaches only 0.02-0.06 rad there (``flags``: a full toe-off at those speeds needs the late
  single-support ZMP moved toward the toe line, a new feature for the owner).  Without the acceleration cap those
  short rolls whip the shank (a 0.12 rad roll in 38 ms takes the CoM's vertical acceleration to about 10 m/s^2).
  Why 0.30, not the middle: the knee's late-stance speed follows the roll's rate, and 0.40 rad by lift-off takes the
  knee to 0.98-0.99 of its speed limit at the nominal walk; V3T (0.40 rad at 4.5 rad/s and 290 rad/s^2) shows it.
* F6 ``roll_amp`` 0.0262 rad (1.5 deg, 1-2 deg), banking toward the stance leg (V4L: the swing side drops).
* F9 ``tail_amp`` 0.125 rad (0.10-0.15) at the phase that minimises the stance pad's peak yaw moment on V4
  (``tail_check``; near the flipped phase -pi/2 except at 75 cm nominal, -2.23 rad).  That phase also lowers the
  single-support friction need below today's phase's at every body and speed, but not always below the tail-less
  walk's: at 75 cm nominal and 0.40 m/s the friction need's worst frame is not the yaw moment's peak frame
  (``tail_phase.json``: ``mu_req_max`` beside each peak); F10 ``head_mode`` "stabilise", ``head_gain`` 0.9 (0.8-1.0), ``head_settle`` 0.015 rad (0.01-0.02)
  at ``settle_lag`` 0.075 s (0.05-0.1).
* F13 ``stride_rel`` 0.875 H (0.85-0.90) at the animal pace, 75 cm only.  F7 ``zmp_inset_y`` 10 mm (8-12).
* F14: ``phase_jitter`` 1.5 % (1-2 %), ``asym`` 2.5 % (2-3 %), ``var_band`` 0.05-0.2 Hz, ``gaze_var`` 0.015 rad (the
  research gives no size for walking gaze drift: the idle micro-motion's 0.01-0.02 rad), ``tail_var`` 0.2 (the tail
  amplitude drifts over F9's 0.10-0.15 rad), ``noise_seed`` 2026, and a window of whole strides at least ``CLIP_S``
  (the renders' 10 s clip) long, periodic as a whole.

The Stage 0 price of every rung (``price``; plan 4 A.3, research 4.1 step 2), before any render
----------------------------------------------------------------------------------------------
* Drives: ``look_inverse_dynamics`` (``gait.inverse_dynamics`` with F5's rolling contact: a heel-up foot's centre of
  pressure on its toe edge or claw tip, not in its flat pad box; identical on every other frame) and
  ``sizing._motor_space``, against the record's drives at its bus with the sizing margins (``envelope._ratios``:
  walk RMS over the sustained torque, 1.25 x peak over the planning peak, 1.15 x speed over the speed at the bus), for
  every drive group; the neck and tail also with the MJCF's joint damping and friction loss (informational: Stage 0
  leaves them out).
* Contact: centre-of-pressure margins (single support; the flat foot in double support; a rolling foot's CoP on its
  contact), the friction need and the split's convergence.
* Clearance: the round-two clearance screen (``clearance.screen``, gates "r2", the rung standing in for the
  reference walk) on the round-two MJCF: the swing gate (the lowest pad point over the middle half of each swing at
  least 0.04 H), the joint-range margin (0.10 rad inside the usable ranges), interference along the walk.
* The plan's rule: a rung over a limit keeps its features and loses that speed rung unless the owner decides
  otherwise (``verdict``).

Outputs (``prepare``): ``<out>/refs/<body>/<speed>/<variant>.npz`` (``arrays``: frames, joint trajectories, timing and
the per-frame quantities the screen and the renders read; git-ignored like Stage 1's arrays), ``<out>/look_refs.json``
(specs, timing, convergence, margins, measured metrics, the Stage 0 price, flags for the owner, each file's sha256;
``body_info``: foot geometry with the claw contact, markers and ``toe_clearance``; ``checks``; ``deferred``) and
``<out>/tail_phase.json`` (``tail_check``: research Bottom line 4's sign check on V0, against the turning screen's
recorded straight run at the nominal walk, F9's phase on V4, and on V4 with F13's stride at 75 cm's animal pace).
Both are updated as each body and speed finishes; ``<out>/partial/`` holds a job's progress until then (``--resume``
keeps it after a restart).

Sources note: this branch edits ``designlab/gait.py`` (and ``turning.generate`` now refuses a look spec), two of the
designlab modules ``stage1.panels.panel_sources`` hashes, so its sources key differs from the parent's (733f2b80... at
e59ab7d) and ``stage1/report.py`` reads Stage 1b's records as keyed to "other panel sources".  Gate G2 (the tests)
shows today's references regenerate bit for bit across the edit, so that flag is not a behaviour change.

Run from hardware/trex_form_study:
    python -m designlab.stage2.look prepare --body 60-N-T-10 --body 75-M-H-10 --out stage2/look --procs 3 [--resume]
    python -m designlab.stage2.look g2 --body 60-N-T-10 [--name walk_0.35_left]  # Stage 1's arrays' sha256
"""

from __future__ import annotations

import os

if __name__ == "__main__":  # one core per process, as the Stage 1 runner (the arrays are then bit-reproducible)
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .. import carpet, clearance, envelope, gait, sizing, statics, turning  # noqa: E402

G = gait.G
SIDES = statics.SIDES
STUDY = Path(__file__).resolve().parents[2]
OUT = STUDY / "stage2" / "look"
ACCURACY = "r2"
BODIES = ("60-N-T-10", "75-M-H-10")
SPEEDS = ("nominal", "v040", "fr05")
#: The agility bar's walking speed (m/s; STAGE2_PLAN section 1).
AGILITY_SPEED = 0.40
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V6")
EXTRAS = ("V1B50", "V1B100", "V3T", "V4L", "V5F13", "V6S")
#: F14's window: whole strides lasting at least the renders' clip (research 4.1: 10 s clips at 30 fps).
CLIP_S = 10.0
#: The F1 bob's share of the stance-leg shortening the flat walk enforces (research F1: screen 50 % and 100 %; the
#: ladder takes the middle, V1B50 and V1B100 the ends).
BOB_SHARE = 0.75
BOB_SHARES_EXTRA = {"V1B50": 0.50, "V1B100": 1.00}
#: F2: the dip into the trough (s).
BOB_DROP_S = 0.10
#: F5: the toe-off by lift-off (rad), the roll's mean rate (rad/s) and peak pitch acceleration (rad/s^2) on the floor:
#: the ladder's, and V3T's (the research's middle at the nominal walk; module docstring).
TOE_OFF, TOE_OFF_RATE, TOE_OFF_ACCEL = 0.30, 3.4, 220.0
TOE_OFF_FULL, TOE_OFF_RATE_FULL, TOE_OFF_ACCEL_FULL = 0.40, 4.5, 290.0
#: F13's stride at the animal pace (75 cm only).
F13_STRIDE = 0.875
#: The multibody-ZMP correction passes of the look rungs (V1 on; today's walk, V0, keeps sizing's 6).  The bob's vertical
#: acceleration and the toe-off's foot pitch slow the correction's convergence: on 60-N-T-10's animal-pace V3 the error
#: after 6 passes is 3.2 mm (V0's 1.95 mm), after 12 about 1.9 mm, falling geometrically by about 0.93 a pass.
LOOK_ZMP_ITERS = 12
#: The hardware question F5's rigid-foot roll raises (finding: the claws reach ahead of the pads).
CLAW_QUESTION = ("A rigid foot pitched heel-up about its toe pad's front edge touches the floor with the middle claw at "
                 "0.08 rad (60 cm) / 0.07 rad (75 cm) and then rolls on the claw tip, as digit III does in birds; the "
                 "reference rolls it so, over a short transition arc (no part below the floor).  The plant has no contact on the claws (contype 0), "
                 "so in simulation the claw passes through the floor and a controller cannot load it; on the hardware "
                 "the claw tip would carry the trailing foot's load through the roll.  For the owner: a load-bearing "
                 "claw tip (and claw contact in the plant), a shorter claw (the roll then stays on the pad edge), or a "
                 "toe-off capped near 0.07-0.08 rad.")
#: Behaviours research 4.1 and plan 4 A.3 list beside the ladder that this phase does not build (finding: they were
#: missing and unrecorded), with why.
DEFERRED = {
    "turns": "research 4.1's 90-degree walking turn for V0 and V6 and plan 4 A.3's 90-degree turn from 0.40 m/s, with "
             "the thigh-twist range and grip need",
    "starts_and_stops": "research 4.1's start-stop for V0 and V6 and plan 4 A.3's two-step stop",
    "expressive_head": "plan 4 A.3's look-at and roar on -X (no -X model yet)",
    "why": "the turn and transition generators (turning.generate; stage1.transitions.gait_of and its plans) read "
           "spec.duty and today's head nod and tail law, so a look spec would silently give today's walk; turning."
           "generate now refuses a look spec (gait.look_active) and transitions.gait_of takes only refs.walk_specs' "
           "walks.  Building these rungs needs look-aware turn and transition paths, beside the old ones (G2 keeps "
           "those unchanged)",
}
#: Research 3.1's ranges and the values taken (the middle where a range is given, unless the rule says why not).
VALUES = {
    "F1 bob": dict(range="50-100 % of the enforced stance-leg shortening (0.007-0.014 H nominal, 0.013-0.026 H at the "
                         "animal pace)", value=BOB_SHARE,
                   rule="bob = 0.75 (duty stride / 2)^2 / (2 (1 - crouch) H) with V0's duty and stride at that speed, "
                        "held from V1 to V6; V1B50 and V1B100 take 50 % and 100 %",
                   drift="F3's longer stance (nominal) and F13's stride (75 cm, animal pace) enforce more shortening, so "
                         "the same bob is a smaller share of the rung's own: about 57 % on the nominal V2-V6 and 55 % on "
                         "the 75 cm animal-pace V6 (measured per rung: bob_share_of_enforced)"),
    "F1 bob_phase": dict(range="hips at today's height at mid single support", value=0.0),
    "F2 bob_skew": dict(range="trough at 10-15 % of stance", value=0.125),
    "F2 bob_drop": dict(range="settling over 0.1-0.15 s; a quick dip then a slow rise", value=BOB_DROP_S,
                        rule="hold the crest through late single support, drop (min-jerk) over 0.1 s to the trough, "
                             "rise (min-jerk) to the next mid-stance; at the faster walks the drop starts before the "
                             "touchdown (a 0.1 s drop after it would need more than g downward)"),
    "F3 swing_time": dict(range="the animal-pace walk's swing time, 0.32 s at 60 cm and 0.36 s at 75 cm", value=None,
                          rule="(1 - envelope.FAST.duty) x the Froude 0.05 walk's period on the body",
                          owner_decision_2="at the nominal walk the duty is 0.69, outside the spec's 0.55-0.65 band: "
                                           "the nominal V2+ rungs need the band widened to 0.72 below 0.30 m/s (owner "
                                           "decision 2), else F3 starts at the brisk walk"),
    "F4 lift_peak": dict(range="35-40 % of the swing", value=0.375,
                         rule="gait.lift_profile: sin^2(pi g(s)), g = s + A s^2 (1 - s)^4, keeping 0.46 of the lift at "
                              "75 % of the swing (the clearance screen's 0.04 H with the 0.10 H lift)"),
    "F5 toe_off": dict(range="0.3-0.5 rad by lift-off", value=TOE_OFF,
                       rule="the range's low end: the knee's late-stance speed follows the roll's rate, and 0.40 rad by "
                            "lift-off takes the knee to 0.98-0.99 of its speed limit at the nominal walk (V3T: 0.40 "
                            "rad at 4.5 rad/s and 290 rad/s^2)"),
    "F5 toe_off_rate": dict(range="not in the research", value=TOE_OFF_RATE,
                            rule="the 60 cm nominal walk's roll (0.30 rad over its 93 ms on the floor, 3.2 rad/s), "
                                 "with room for F14's jitter (rolls up to 4 % shorter) so both nominal walks reach "
                                 "0.30 on every step; where the floor roll is shorter (0.40 m/s, the animal "
                                 "pace) the toe-off is what the roll reaches by lift-off within this rate and "
                                 "toe_off_accel, and the pitch only falls after it (no heel flick in the air)"),
    "F5 toe_off_accel": dict(range="not in the research", value=TOE_OFF_ACCEL,
                             rule="the same roll's min-jerk peak pitch acceleration (5.77 x 0.30 / 0.093^2 = 199, with "
                                  "the same room for jitter): the 23-38 ms rolls at 0.40 m/s and the animal pace reach "
                                  "0.02-0.06 rad by lift-off; a 0.12 rad roll there whips the shank (the CoM's "
                                  "vertical acceleration near 10 m/s^2)"),
    "F5 toe_off_start": dict(range="80-85 % of stance", value=0.825),
    "F5 toe_reach": dict(range="0.05-0.10 rad", value=0.075),
    "F5 toe_off_margin": dict(range="not in the research", value=0.005,
                              rule="the roll waits until the ZMP reference is 5 mm inside the rolling support (the "
                                   "trailing toe pad's front edge and the leading foot's pads); every rolling frame is "
                                   "checked against its actual contact (the edge or the claw tip)"),
    "F5 pivot": dict(range="toes roll up; digit III pivots on its claw",
                     value="toe pad edge, a transition arc, then the middle claw tip (gait.roll_offset)",
                     rule=CLAW_QUESTION),
    "F6 roll_amp": dict(range="1-2 deg", value=math.radians(1.5),
                        rule="toward the stance leg at its mid-stance (V4); V4L the other way"),
    "F7 zmp_inset_y": dict(range="screen 8-12 mm", value=0.010, rule="optional extra V6S"),
    "F9 tail_amp": dict(range="0.10-0.15 rad", value=0.125),
    "F9 tail_phase": dict(range="the phase minimising the stance foot's yaw moment", value=None,
                          rule="tail_check on V4 per body and speed: the peak |M_z| of the single-support stance pad"),
    "F10 head_gain": dict(range="0.8-1.0", value=0.9),
    "F10 head_settle": dict(range="0.01-0.02 rad", value=0.015),
    "F10 settle_lag": dict(range="0.05-0.1 s", value=0.075),
    "F13 stride_rel": dict(range="0.85-0.90 H at the animal pace, 75 cm only", value=F13_STRIDE,
                           rule="V5F13 adds it to V5 with its own tail phase (tail_check on V4 with this stride); V6 "
                                "adds F14 to V5F13 there"),
    "F14 phase_jitter": dict(range="step time +-1-2 %", value=0.015),
    "F14 asym": dict(range="left-right asymmetry <= 2-3 %", value=0.025),
    "F14 var_band": dict(range="0.05-0.2 Hz", value=[0.05, 0.2]),
    "F14 gaze_var": dict(range="not sized in the research", value=0.015,
                         rule="the idle micro-motion's 0.01-0.02 rad (research 5.1), middle"),
    "F14 tail_var": dict(range="not sized in the research", value=0.2,
                         rule="the tail amplitude drifts over F9's 0.10-0.15 rad (0.125 +- 20 %)"),
    "F14 noise_seed": dict(range="any", value=2026),
    "zmp_iters": dict(range="sizing's 6 passes", value=LOOK_ZMP_ITERS,
                      rule="look rungs (V1 on) run 12 multibody-ZMP correction passes: the bob and the toe-off slow the "
                           "convergence (60-N-T-10 animal-pace V3: 3.2 mm after 6, 1.9 mm after 12; V0 1.95 mm after 6)",
                      effect_alone="V1 changes the passes as well as adding F1/F2; V0 regenerated at 12 passes "
                                   "measures what the passes alone change, at the sub-millimetre level of the CoM path "
                                   "(look_refs.json checks.<body>.<speed>.zmp_passes)"),
    "v040": dict(range="0.40 m/s", value=AGILITY_SPEED,
                 rule="the gait the brisk_to_fr05 speed change passes through at 0.40 m/s: period, duty and lift mixed "
                      "with the speed's weight (transitions.py)"),
    "deferred": DEFERRED,
}
#: The plan's limit rule (STAGE2_PLAN section 4 A.3), as each rung's verdict states it.
LIMIT_RULE = ("over a drive limit (or a Stage 0 gate), a variant keeps its feature and loses that speed rung "
              "(character first) unless the owner decides otherwise")
#: The single-support CoP margin below which a reference's drive loads are not meaningful (envelope.fast_walk's rule).
COP_RULE_M = -0.010
#: The tail-phase check: the phases run (rad) and the grid the linear fit is minimised on.
TAIL_PHASES = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
TAIL_GRID = 720
#: Point-light markers (research 4.1 step 3) the renders draw: name -> (body, how the point is found).
POINTS = ("snout", "eye", "neck_base", "hip_l", "hip_r", "knee_l", "knee_r", "mtp_l", "mtp_r", "toe_l", "toe_r",
          "tail_mid", "tail_tip")


@contextmanager
def accuracy(level: str = ACCURACY):
    """``gait.set_accuracy(level)`` for the block, restoring the process's level after (it is module state)."""
    prev = gait.ACCURACY
    gait.set_accuracy(level)
    try:
        yield
    finally:
        gait.set_accuracy(prev)


# ------------------------------------------------------------------------------------------------
# Specs
# ------------------------------------------------------------------------------------------------
def speed_spec(H: float, v: float = AGILITY_SPEED, lo: gait.GaitSpec = sizing.BRISK,
               hi: gait.GaitSpec = envelope.FAST) -> gait.GaitSpec:
    """The gait ``transitions.py``'s ``lo``-to-``hi`` speed change passes through at speed v (module docstring)."""
    a, b = lo.describe(H), hi.describe(H)
    w = (v - a["speed"]) / (b["speed"] - a["speed"])
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"{v} m/s is outside {a['speed']:.3f}-{b['speed']:.3f} m/s")
    T = a["period"] + w * (b["period"] - a["period"])
    return replace(lo, froude=v * v / (G * H), stride_rel=v * T / H, duty=lo.duty + w * (hi.duty - lo.duty),
                   lift_frac=lo.lift_frac + w * (hi.lift_frac - lo.lift_frac))


def base_spec(H: float, speed: str) -> gait.GaitSpec:
    """V0 at a speed (``SPEEDS``)."""
    if speed == "nominal":
        return sizing.NOMINAL
    if speed == "fr05":
        return envelope.FAST
    if speed == "v040":
        return speed_spec(H)
    raise KeyError(speed)


def enforced_shortening(spec: gait.GaitSpec) -> float:
    """The stance leg's shortening (fraction of H) that a constant hip height enforces: (duty stride / 2)^2 /
    (2 (1 - crouch)), the hip passing over the foot at the walk's height (research section 1)."""
    return (spec.duty * spec.stride_rel / 2) ** 2 / (2 * (1 - spec.crouch))


def swing_time(H: float) -> float:
    """F3: the animal-pace walk's swing time on a body of hip height H (s)."""
    return float((1.0 - envelope.FAST.duty) * envelope.FAST.describe(H)["period"])


def f13_applies(body: str, speed: str) -> bool:
    """F13 (a longer stride at the animal pace) is screened at 75 cm only (research F13: at 60 cm the knee rules it out)."""
    return speed == "fr05" and body.startswith("75")


def feature_values(body: str, H: float, speed: str, tail_phase: float | None = None) -> dict:
    """The look fields each feature group sets at this body and speed (``VALUES``)."""
    v0 = base_spec(H, speed)
    out = {
        "F1F2": dict(bob=BOB_SHARE * enforced_shortening(v0), bob_skew=0.125, bob_drop=BOB_DROP_S),
        "F3": dict(swing_time=swing_time(H)),
        "F4F5": dict(lift_peak=0.375, toe_off=TOE_OFF, toe_off_rate=TOE_OFF_RATE, toe_off_accel=TOE_OFF_ACCEL,
                     toe_off_start=0.825, toe_reach=0.075, toe_off_margin=0.005),
        "F6": dict(roll_amp=math.radians(1.5)),
        "F9F10": dict(tail_amp=0.125, head_mode="stabilise", head_gain=0.9, head_settle=0.015, settle_lag=0.075),
        "F14": dict(phase_jitter=0.015, asym=0.025, gaze_var=0.015, tail_var=0.2, var_band=(0.05, 0.2),
                    noise_seed=2026),
        "F13": dict(stride_rel=F13_STRIDE) if f13_applies(body, speed) else {},
        "F7": dict(zmp_inset_y=0.010),
        "F5full": dict(toe_off=TOE_OFF_FULL, toe_off_rate=TOE_OFF_RATE_FULL, toe_off_accel=TOE_OFF_ACCEL_FULL),
    }
    if tail_phase is not None:
        out["F9F10"]["tail_phase"] = float(tail_phase)
    return out


def window_strides(spec: gait.GaitSpec, H: float, clip_s: float = CLIP_S) -> int:
    """F14: the fewest whole strides lasting at least ``clip_s``."""
    return int(math.ceil(clip_s / spec.describe(H)["period"] - 1e-9))


def ladder(body: str, H: float, speed: str, tail_phase: float | None = None, extras: bool = True,
           tail_phase_f13: float | None = None) -> dict:
    """{variant: GaitSpec} at one body and speed.  V5, V6 and V6S need F9's ``tail_phase`` (``tail_check`` on V4);
    without it they are left out.  Where F13 applies (``f13_applies``), V5F13 is V5 with F13's stride and
    ``tail_phase_f13`` (the tail check on V4 with that stride) and V6 adds F14 to it; without ``tail_phase_f13`` V5F13,
    V6 and V6S are left out there."""
    f = feature_values(body, H, speed, tail_phase)
    s = {"V0": base_spec(H, speed)}
    s["V1"] = replace(s["V0"], **f["F1F2"], zmp_iters=LOOK_ZMP_ITERS)
    if extras:
        for name, share in BOB_SHARES_EXTRA.items():
            s[name] = replace(s["V1"], bob=share * enforced_shortening(s["V0"]))
    s["V2"] = replace(s["V1"], **f["F3"])
    s["V3"] = replace(s["V2"], **f["F4F5"])
    if extras:
        s["V3T"] = replace(s["V3"], **f["F5full"])
    s["V4"] = replace(s["V3"], **f["F6"])
    if extras:
        s["V4L"] = replace(s["V3"], roll_amp=-f["F6"]["roll_amp"])
    if tail_phase is not None:
        s["V5"] = replace(s["V4"], **f["F9F10"])
        base = s["V5"]
        if f["F13"]:
            if tail_phase_f13 is None:
                return s
            s["V5F13"] = base = replace(s["V5"], **f["F13"], tail_phase=float(tail_phase_f13))
        v6 = replace(base, **f["F14"])
        s["V6"] = replace(v6, strides=window_strides(v6, H))
        if extras:
            s["V6S"] = replace(s["V6"], **f["F7"])
    if not extras:
        s.pop("V5F13", None)
    return s


def features_of(variant: str, body: str | None = None, speed: str | None = None) -> list[str]:
    """The research's feature numbers a ladder rung carries (F13 on V6 where it applies)."""
    rungs = {"V0": [], "V1": ["F1", "F2"], "V2": ["F3"], "V3": ["F4", "F5"], "V4": ["F6"], "V5": ["F9", "F10"],
             "V6": ["F14"]}
    extra = {"V1B50": ("V1", ["F1 at 50 %"]), "V1B100": ("V1", ["F1 at 100 %"]),
             "V3T": ("V3", ["F5 at 0.40 rad (the research's middle)"]), "V4L": ("V3", ["F6 (swing side drops)"]),
             "V5F13": ("V5", ["F13"]), "V6S": ("V6", ["F7"])}
    if variant in extra:
        parent, add = extra[variant]
        return features_of(parent, body, speed) + add
    if variant not in VARIANTS:
        return []
    out = []
    for v in VARIANTS[: VARIANTS.index(variant) + 1]:
        out += rungs[v]
        if v == "V5" and variant == "V6" and body is not None and f13_applies(body, speed):
            out.append("F13")
    return out


# ------------------------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------------------------
def source(body: str) -> dict:
    """The body rebuilt from its record (``stage1.refs.source``: model checked against the record's xml_sha256,
    full-precision home pose) plus its foot geometry and point-light markers."""
    from ..stage1 import refs

    src = refs.source(body)
    src["M"].forward(src["q_home"])
    src["geometry"] = gait.foot_geometry(src["M"])
    src["markers"] = markers(src["M"])
    src["M"].forward(src["q_home"])
    return src


def toe_clearance(src: dict, side: str = "left", pitch: float = 0.40, iters: int = 30) -> dict:
    """F5 on the rigid foot at home: the foot pitched heel-up about its toe pivot (``gait.foot_geometry``; the other foot
    and the trunk at home, IK as the generator's) and the lowest point of each of its geoms that is not a floor pad
    (the toes, claws, hallux and plate: no contact in the plant, ``contype`` 0).  Returns the largest pitch at which
    every such geom stays at or above the floor (bisection) and, at ``pitch`` (F5's ``toe_off``), how far below the
    floor each one reaches (mm; negative: below).  In the plant these geoms pass through the floor; on the hardware
    they would touch it first and take the pivot from the pads."""
    M, q0, geo = src["M"], src["q_home"], src["geometry"]
    m, d = M.m, M.d
    M.forward(q0)
    base = {s: M.foot_pose(s)[0].copy() for s in SIDES}
    others = [g for g in carpet.foot_geoms(m)[side] if m.geom_contype[g] != 1]

    def lows(th: float) -> dict:
        R = gait._rot_y(th)
        tg = {s: (base[s].copy(), np.eye(3)) for s in SIDES}
        tg[side] = (base[side] + (np.eye(3) - R) @ geo[side]["toe_pivot"], R)
        q, ok = statics.leg_ik(M, q0.copy(), tg, iters=200)
        if not ok:
            raise RuntimeError(f"toe_clearance: IK failed at a pitch of {th} rad")
        M.forward(q)
        out = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): float(carpet.lowest_point(m, d, g)[2]) for g in others}
        M.forward(q0)
        return out

    lo, hi = 0.0, float(pitch)
    if min(lows(hi).values()) >= 0.0:
        lo = hi
    else:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if min(lows(mid).values()) >= 0.0 else (lo, mid)
    at = lows(pitch)
    worst = min(at, key=at.get)
    return dict(side=side, pivot=geo[side]["toe_pad"] + " front bottom edge", clear_pitch_rad=float(lo),
                at_pitch_rad=float(pitch), lowest_mm_at_pitch={k: 1e3 * v for k, v in sorted(at.items(), key=lambda kv: kv[1])},
                deepest=worst, deepest_mm=1e3 * at[worst])


def generate(src: dict, spec: gait.GaitSpec) -> dict:
    """One variant's trajectory at round two's accuracy, with the look per-frame quantities completed for today's
    walk too (its frames carry no ``look`` block)."""
    M, q0, H = src["M"], src["q_home"], src["H"]
    t0 = time.perf_counter()
    with accuracy(ACCURACY):
        tr = gait.generate(M, q0, H, spec)
        qd, qdd = gait.derivatives(M, tr["q"], tr["dt"], stride=tr["info"]["stride"] * tr["info"].get("strides", 1))
    if "look" not in tr:
        tr = _complete_v0(src, spec, tr)
    tr["qd"], tr["qdd"] = qd, qdd
    tr["seconds"] = time.perf_counter() - t0
    return tr


def _complete_v0(src: dict, spec: gait.GaitSpec, tr: dict) -> dict:
    """A default-field walk (``gait.generate``'s own path) with the look block its frames imply: phase t / T, flat
    feet, no bob or roll; its margins against the pads (``gait.support_margins``)."""
    M, q0 = src["M"], src["q_home"]
    info = tr["info"]
    T, duty, n = info["period"], info["duty"], len(tr["t"])
    phi = tr["t"] / T
    sw = 1.0 - duty
    swing = np.zeros((n, 2), bool)
    swing_u = np.full((n, 2), np.nan)
    for j, off in enumerate((0.0, 0.5)):
        u = np.mod(np.mod(phi, 1.0) - off, 1.0)
        swing[:, j] = u < sw
        swing_u[swing[:, j], j] = u[swing[:, j]] / sw
    neck_adr = M.m.jnt_qposadr[statics.jid(M.m, "neck_pitch")]
    tail_adr = M.m.jnt_qposadr[statics.jid(M.m, "tail_yaw")]
    margin, margin_ref, hulls = gait.support_margins(M, tr["q"], tr["phases"], tr["zmp_mb"], tr["zmp_ref"])
    feet = np.zeros((n, 2, 3))
    for k in range(n):
        M.forward(tr["q"][k])
        feet[k] = [M.foot_pose(s)[0] for s in SIDES]
    touchdowns = sorted(float((i + off + sw) * T) for off in (0.0, 0.5) for i in range(-1, 2))
    info = info | dict(strides=1, window_s=float(T), swing_time=float(sw * T), look=dict(gait.LOOK_DEFAULTS),
                       bob_m=0.0, bob_trough_step=0.5, step_deviations=[0.0, 0.0], gaze_freqs_hz=[], tail_freqs_hz=[],
                       rolling_entry=None, zmp_margin_min_m=float(np.min(margin)),
                       zmp_margin_ref_min_m=float(np.min(margin_ref)))
    look = dict(phase=phi, swing=swing, swing_u=swing_u, foot_pitch=np.zeros((n, 2)), rolling=np.zeros((n, 2), bool),
                rolling_contact=np.full((n, 2, 2, 3), np.nan), on_claw=np.zeros((n, 2), bool),
                foot_base=feet, foot_target=feet, bob=np.zeros(n), trunk_roll=np.zeros(n),
                trunk_quat=np.tile(q0[3:7], (n, 1)), neck=tr["q"][:, neck_adr].copy(), tail=tr["q"][:, tail_adr].copy(),
                head_settle=np.zeros(n), gaze_var=np.zeros(n), tail_var=np.zeros(n), zmp_margin=margin,
                zmp_ref_margin=margin_ref, support_hull=hulls, touchdowns=touchdowns, rolls={s: [] for s in SIDES},
                geometry=src["geometry"], warp=gait._Warp(T, 1))
    return dict(tr, info=info, look=look)


# ------------------------------------------------------------------------------------------------
# Per-frame quantities for the screen and the renders
# ------------------------------------------------------------------------------------------------
def markers(M: statics.Model) -> list[tuple[str, int, np.ndarray]]:
    """The point-light markers (``POINTS``) as (name, body id, point in the body's frame), found on the model at its
    current pose (call at home): the skull cover's foremost vertex (snout), an eye, the neck-pitch axis (neck base),
    the hip-pitch and knee axes, the foot bodies' origins (the ankle, the animal's MTP), the middle claw's tip, the
    tail cover's rearmost vertex (tail tip: over every ``cover_tail*`` mesh on the tail body, one cover on the Tapered
    tail, a root and segments on the Hybrid's) and the point midway from the tail-yaw axis to it (mid-tail)."""
    m, d = M.m, M.d

    def local(b, p):
        return d.xmat[b].reshape(3, 3).T @ (np.asarray(p) - d.xpos[b])

    def geom(name):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            raise KeyError(f"no geom {name} on the model")
        return g

    def vertices(g):
        mid = m.geom_dataid[g]
        a, k = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return d.geom_xpos[g] + m.mesh_vert[a : a + k] @ d.geom_xmat[g].reshape(3, 3).T

    head, tail = statics.bid(m, "head"), statics.bid(m, "tail")
    out = []
    skull = vertices(geom("cover_skull"))
    out.append(("snout", head, local(head, skull[np.argmax(skull[:, 0])])))
    out.append(("eye", head, local(head, d.geom_xpos[geom("hw_eye_0")])))
    jn = statics.jid(m, "neck_pitch")
    out.append(("neck_base", int(m.jnt_bodyid[jn]), m.jnt_pos[jn].copy()))
    for jname, tag in (("hip_pitch", "hip"), ("knee", "knee")):
        for side in SIDES:
            j = statics.jid(m, f"{side}_{jname}")
            out.append((f"{tag}_{side[0]}", int(m.jnt_bodyid[j]), m.jnt_pos[j].copy()))
    for side in SIDES:
        out.append((f"mtp_{side[0]}", M.foot[side], np.zeros(3)))
    for side in SIDES:
        g = geom(f"hw_{side}_claw_0")
        c, a = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)[:, 2]
        ends = [c + m.geom_size[g][1] * a, c - m.geom_size[g][1] * a]
        out.append((f"toe_{side[0]}", M.foot[side], local(M.foot[side], max(ends, key=lambda e: e[0]))))
    covers = [g for g in range(m.ngeom) if m.geom_bodyid[g] == tail and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
              and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("cover_tail")]
    if not covers:
        raise KeyError("no cover_tail* mesh on the tail body")
    tip = np.vstack([vertices(g) for g in covers])
    tip = tip[np.argmin(tip[:, 0])]
    out.append(("tail_mid", tail, local(tail, 0.5 * (tip + d.xanchor[statics.jid(m, "tail_yaw")]))))
    out.append(("tail_tip", tail, local(tail, tip)))
    assert [o[0] for o in out] == list(POINTS)
    return out


def _foot_pitch(R: np.ndarray) -> float:
    """Heel-up pitch of a foot about its lateral axis (rad): atan2(-R[2, 0], R[0, 0])."""
    return float(math.atan2(-R[2, 0], R[0, 0]))


def arrays(src: dict, g: dict) -> dict:
    """The arrays of one variant for its npz file.  Beside Stage 1's reference keys (t, dt, q, qd, qdd, support,
    com_ref, zmp_ref, zmp_mb, feet, com, q_home, joints, stride_motion): the gait phase (strides, warped by F14), swing
    flags and swing fraction, the target and solved foot pitch, which feet roll and which roll past the toe edge
    (``on_claw``: on the transition arc or the claw tip), each foot's front contact on the solved pose (``toe_edge``:
    the toe pad's front edge's two end points, or the arc's or claw tip's floor point twice past it; a rolling foot's
    whole support; ``toe_pivot`` their midpoint),
    foot orientations, the support hull (NaN-padded) and the ZMP's margin inside it, the hip midpoint (the hip-roll
    axes', fixed on the trunk), bob, trunk roll and orientation, neck and tail angles (and F10's settle and F14's
    variations), the head's pitch in the world, the lowest pad and toe/claw points of each foot, the point-light
    markers, the events, and the IK's result (``ik_ok``, ``ik_fail_frames``)."""
    M, tr, lk = src["M"], g, g["look"]
    m, d = M.m, M.d
    n = len(tr["q"])
    feet = np.zeros((n, 2, 3))
    foot_quat = np.zeros((n, 2, 4))
    pitch_solved = np.zeros((n, 2))
    edge = np.zeros((n, 2, 2, 3))
    com = np.zeros((n, 3))
    hip_mid = np.zeros((n, 3))
    head_pitch = np.zeros(n)
    pad_z = np.zeros((n, 2))
    toe_z = np.zeros((n, 2))
    pts = np.zeros((n, len(POINTS), 3))
    geoms = carpet.foot_geoms(m)
    hips = [statics.jid(m, f"{s}_hip_roll") for s in SIDES]  # on the trunk: the hip-pitch axes move with hip roll
    head = statics.bid(m, "head")
    geo = src["geometry"]
    on_claw = np.asarray(lk["on_claw"], bool)
    for k in range(n):
        M.forward(tr["q"][k])
        for j, s in enumerate(SIDES):
            p, R = M.foot_pose(s)
            feet[k, j] = p
            mujoco.mju_mat2Quat(foot_quat[k, j], R.ravel())
            pitch_solved[k, j] = _foot_pitch(R)
            # the front contact on the solved pose: the toe edge, or past it the transition arc's or claw tip's point
            P, _ = gait._contact_in_foot(geo[s], max(float(lk["foot_pitch"][k, j]), 0.0))
            edge[k, j] = p + P @ R.T
            low = {gg: carpet.lowest_point(m, d, gg)[2] for gg in geoms[s]}
            pad_z[k, j] = min(z for gg, z in low.items() if m.geom_contype[gg] == 1)
            toe_z[k, j] = min((z for gg, z in low.items() if carpet._is_toe(m, gg)), default=np.nan)
        com[k] = M.com()
        hip_mid[k] = 0.5 * (d.xanchor[hips[0]] + d.xanchor[hips[1]])
        head_pitch[k] = gait._nose_up(d.xquat[head])
        for i, (_, b, loc) in enumerate(src["markers"]):
            pts[k, i] = d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc
    hmax = max(len(h) for h in lk["support_hull"])
    hull = np.full((n, hmax, 2), np.nan)
    for k, h in enumerate(lk["support_hull"]):
        hull[k, : len(h)] = h
    info = tr["info"]
    K = int(info.get("strides", 1))
    ev = events(lk["phase"], tr["t"], info["duty"], K)
    return dict(t=tr["t"], dt=np.array(float(tr["dt"])), q=tr["q"], qd=tr["qd"], qdd=tr["qdd"],
                support=np.array([[s in ph for s in SIDES] for ph in tr["phases"]]), com_ref=tr["com_ref"],
                zmp_ref=tr["zmp_ref"], zmp_mb=tr["zmp_mb"], feet=feet, com=com, q_home=src["q_home"],
                joints=np.array(M.names), stride_motion=np.array([0.0, 0.0, 0.0, K * float(info["stride"]), 0.0]),
                phase=lk["phase"], swing=lk["swing"], swing_u=lk["swing_u"], foot_pitch=lk["foot_pitch"],
                foot_pitch_solved=pitch_solved, rolling=lk["rolling"], on_claw=on_claw, toe_pivot=edge.mean(axis=2),
                toe_edge=edge, foot_quat=foot_quat, ik_ok=np.array(bool(tr["ik_ok"])),
                ik_fail_frames=np.asarray(tr["ik_fail_frames"], dtype=np.int64).reshape(-1),
                foot_target=lk["foot_target"], support_hull=hull, zmp_margin=lk["zmp_margin"],
                zmp_ref_margin=lk["zmp_ref_margin"], hip_mid=hip_mid, bob=lk["bob"], trunk_roll=lk["trunk_roll"],
                trunk_quat=lk["trunk_quat"], neck=lk["neck"], tail=lk["tail"], head_settle=lk["head_settle"],
                gaze_var=lk["gaze_var"], tail_var=lk["tail_var"], head_pitch_world=head_pitch, pad_floor_z=pad_z,
                toe_floor_z=toe_z, points=pts, point_names=np.array(POINTS),
                lift_off_t=ev["lift_off"], touchdown_t=ev["touchdown"])


def events(phase, t, duty: float, K: int) -> dict:
    """Lift-off and touchdown times (s) of each foot over the window (left then right; the left foot lifts at integer
    phases, the right at half-integers; the left lands at phase 1 - duty), from the frames' phase by interpolation."""
    sw = 1.0 - duty
    out = {}
    for key, shift in (("lift_off", 0.0), ("touchdown", sw)):
        rows = []
        for off in (0.0, 0.5):
            f = np.arange(K) + off + shift
            rows.append(np.interp(f, phase, t, left=np.nan, right=np.nan))
        out[key] = np.array(rows)
    return out


def trajectory(arrs: dict, info: dict) -> dict:
    """A ``gait.generate``-style trajectory dict from a variant's arrays and its info (for ``look_inverse_dynamics``,
    ``gait.inverse_dynamics``, ``clearance.screen`` and ``turning_traj``): t, q, phases (support sets), dt, the ZMP and
    CoM references, info, and the IK's result as the arrays record it (arrays from before they did are refused)."""
    if "ik_ok" not in arrs or "ik_fail_frames" not in arrs:
        raise KeyError("these arrays do not record the IK's result (ik_ok, ik_fail_frames): regenerate them")
    sup = np.asarray(arrs["support"], bool)
    phases = [tuple(s for j, s in enumerate(SIDES) if sup[k, j]) for k in range(len(sup))]
    return dict(t=np.asarray(arrs["t"]), q=np.asarray(arrs["q"]), phases=phases, dt=float(arrs["dt"]),
                zmp_ref=np.asarray(arrs["zmp_ref"]), com_ref=np.asarray(arrs["com_ref"]), zmp_mb=np.asarray(arrs["zmp_mb"]),
                info=dict(info), ik_ok=bool(np.asarray(arrs["ik_ok"])),
                ik_fail_frames=[int(k) for k in np.asarray(arrs["ik_fail_frames"]).reshape(-1)])


def turning_traj(traj: dict, M: statics.Model, q_home) -> dict:
    """The trajectory as ``turning.inverse_dynamics`` reads a straight run (yaw rate 0): the path frame from the
    hip-pitch midpoint at home, the stride motion of the window and the exact unwrap."""
    info = traj["info"]
    M.forward(q_home)
    origin = 0.5 * sum(M.d.xanchor[statics.jid(M.m, f"{s}_hip_pitch")][:2] for s in SIDES)
    adv = float(info["stride"]) * int(info.get("strides", 1))
    path = origin[None, :] + np.column_stack([info["speed"] * traj["t"], np.zeros(len(traj["t"]))])
    return dict(traj, transform=(0.0, origin.copy(), origin + np.array([adv, 0.0])), path=path,
                heading=np.zeros(len(traj["t"])),
                info=dict(info, yaw_rate=0.0, unwrap="exact", origin=origin.tolist()))


# ------------------------------------------------------------------------------------------------
# The Stage 0 price (module docstring): F5-aware inverse dynamics, drive ratios, contact, clearance, verdict
# ------------------------------------------------------------------------------------------------
def _support_rows(o: int, nw: int, p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """A rolling foot's centre-of-pressure constraints on its floor contact, segment a-b (world xy; a point if a = b),
    for the wrench at offset o of w (force then moment at the foot origin p, world axes; ``statics.cop``'s reading):
    with X = (c_x - a_x) f_z = -(m_y + h f_x) + (p_x - a_x) f_z and Y = (c_y - a_y) f_z = (m_x - h f_y) + (p_y - a_y) f_z
    (h = p_z), equalities E w = 0 (on the line, or at the point) and inequalities G w >= 0 (between the ends)."""
    h = p[2]
    X, Y = np.zeros(nw), np.zeros(nw)
    X[o + 0], X[o + 2], X[o + 4] = -h, p[0] - a[0], -1.0
    Y[o + 1], Y[o + 2], Y[o + 3] = -h, p[1] - a[1], 1.0
    d = np.asarray(b, float) - np.asarray(a, float)
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return np.vstack([X, Y]), np.zeros((0, nw))
    tx, ty = d / L
    along = tx * X + ty * Y
    fz = np.zeros(nw)
    fz[o + 2] = L
    return (-ty * X + tx * Y)[None, :], np.vstack([along, fz - along])


def _rolling_contact_torques(M: statics.Model, q, qd, qdd, support, contacts: dict, mu: float = 0.8):
    """``statics.contact_torques`` in double support with F5's rolling contact: the minimum-joint-effort split of the
    (unique) total contact wrench between the two feet (``statics._double_support_qp``'s objective, normal force,
    friction pyramid and the flat foot's CoP in its pad box), with each rolling foot's CoP held on its floor contact
    (``contacts[side]``: the toe edge's or claw tip's two end points, world; ``_support_rows``).  Returns (tau,
    wrenches, solved)."""
    from scipy.optimize import minimize

    m, d = M.m, M.d
    d.qpos[:] = q
    d.qvel[:] = qd
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, Mfull)
    rhs = Mfull @ qdd + d.qfrc_bias
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    Js = []
    for side in support:
        mujoco.mj_jacBody(m, d, jacp, jacr, M.foot[side])
        Js.append(np.vstack([jacp, jacr]).copy())
    J = np.vstack(Js)
    act = M.act_dofs
    A, b = J[:, np.arange(6)].T, rhs[np.arange(6)]
    Ja, ra = J[:, act].T, rhs[act]
    nw = J.shape[0]
    Hq = Ja.T @ Ja + 1e-6 * np.eye(nw)
    g = -Ja.T @ ra
    E_rows, G_rows = [A], []
    for k, side in enumerate(support):
        o = 6 * k
        p, _ = M.foot_pose(side)
        e = np.zeros((5, nw))
        e[0, o + 2] = 1  # fz >= 0
        e[1, o + 2], e[1, o + 0] = mu, -1
        e[2, o + 2], e[2, o + 0] = mu, 1
        e[3, o + 2], e[3, o + 1] = mu, -1
        e[4, o + 2], e[4, o + 1] = mu, 1
        G_rows.append(e)
        if side in contacts:
            Eq, Gi = _support_rows(o, nw, p, contacts[side][0][:2], contacts[side][1][:2])
            E_rows.append(Eq)
            G_rows.append(Gi)
        else:  # the flat foot: its CoP in its pad box (statics._double_support_qp's rows)
            poly = M.pad_polygon(side) - p[:2]
            (xmin, ymin), (xmax, ymax) = poly.min(0), poly.max(0)
            hgt = p[2]
            c = np.zeros((4, nw))
            c[0, o + 4], c[0, o + 0], c[0, o + 2] = -1, -hgt, -xmin
            c[1, o + 4], c[1, o + 0], c[1, o + 2] = 1, hgt, xmax
            c[2, o + 3], c[2, o + 1], c[2, o + 2] = 1, -hgt, -ymin
            c[3, o + 3], c[3, o + 1], c[3, o + 2] = -1, hgt, ymax
            G_rows.append(c)
    Em = np.vstack(E_rows)
    be = np.concatenate([b, np.zeros(len(Em) - 6)])
    Gm = np.vstack(G_rows)
    K = np.block([[Hq, Em.T], [Em, np.zeros((len(Em), len(Em)))]])
    w0 = np.linalg.lstsq(K, np.concatenate([-g, be]), rcond=None)[0][:nw]
    if np.all(Gm @ w0 >= -1e-9) and np.allclose(Em @ w0, be, atol=1e-9):
        w, ok = w0, True
    else:
        res = minimize(lambda x: 0.5 * x @ Hq @ x + g @ x, w0, jac=lambda x: Hq @ x + g, method="SLSQP",
                       constraints=[dict(type="eq", fun=lambda x: Em @ x - be, jac=lambda x: Em),
                                    dict(type="ineq", fun=lambda x: Gm @ x, jac=lambda x: Gm)],
                       options=dict(maxiter=500, ftol=1e-12))
        w = res.x
        ok = bool(res.success and np.all(Gm @ w >= -1e-6) and np.allclose(Em @ w, be, atol=1e-6))
    tau = rhs[act] - J[:, act].T @ w
    return tau, {side: w[6 * k : 6 * k + 6] for k, side in enumerate(support)}, ok


def look_inverse_dynamics(M: statics.Model, traj: dict, rolling, contact, mu: float = 0.8) -> dict:
    """``gait.inverse_dynamics`` with F5's rolling contact (``gait.inverse_dynamics`` and ``statics`` are unchanged).
    Their double-support split keeps every stance foot's centre of pressure in its flat pad box, so a heel-up foot's
    falls under a heel in the air; here a rolling foot's (``rolling[k, j]``) is held on its floor contact
    (``contact[k, j]``: the arrays' ``toe_edge``, the toe edge or the claw tip; ``_rolling_contact_torques``).  Every
    other frame is ``gait.inverse_dynamics``'s exactly (``statics.contact_torques``, the same derivatives).  Beside its
    keys: per frame and foot, the rolling CoP's distance off its contact (m; the toe edge's line or the claw tip's
    point) and its margin along the edge (m, positive between the ends; NaN on the claw), the whole contact force's
    friction need (horizontal over vertical, independent of the split) and the frames whose split did not solve."""
    qs, dt, phases = traj["q"], traj["dt"], traj["phases"]
    info = traj.get("info", {})
    stride = info.get("stride") if not info.get("yaw_rate") else None
    if stride is not None and info.get("strides", 1) != 1:
        stride = stride * info["strides"]
    qd, qdd = gait.derivatives(M, qs, dt, stride=stride)
    rolling, contact = np.asarray(rolling, bool), np.asarray(contact, float)
    n, na = len(qs), len(M.act_dofs)
    tau = np.zeros((n, na))
    cop_margin = np.full((n, 2), np.nan)
    off_line = np.full((n, 2), np.nan)
    along = np.full((n, 2), np.nan)
    friction = np.zeros(n)
    total_friction = np.zeros(n)
    normal = np.zeros((n, 2))
    failed = []
    for k in range(n):
        sup = phases[k] if phases[k] else ("left", "right")
        rol = {s: contact[k, j] for j, s in enumerate(SIDES) if s in sup and rolling[k, j]}
        if rol and len(sup) == 2:
            tk, w, ok = _rolling_contact_torques(M, qs[k], qd[k], qdd[k], sup, rol, mu)
            if not ok:
                failed.append(k)
        else:
            tk, w = statics.contact_torques(M, qs[k], qd[k], qdd[k], support=sup)
        tau[k] = tk
        ftot = sum(w[s_][:3] for s_ in w)
        total_friction[k] = np.linalg.norm(ftot[:2]) / max(ftot[2], 1e-6)
        for j, side in enumerate(SIDES):
            if side not in w:
                continue
            f = w[side][:3]
            normal[k, j] = f[2]
            friction[k] = max(friction[k], np.linalg.norm(f[:2]) / max(f[2], 1e-6))
            c = statics.cop(M, side, w[side])
            M.forward(qs[k])
            if side in rol:
                a, b = rol[side][0][:2], rol[side][1][:2]
                dv = b - a
                L = float(np.linalg.norm(dv))
                if L < 1e-9:  # the claw tip: a point (no along-edge margin)
                    off_line[k, j] = float(np.linalg.norm(c - a))
                else:
                    tv = dv / L
                    s_ = float(np.dot(c - a, tv))
                    off_line[k, j] = abs(float(tv[0] * (c - a)[1] - tv[1] * (c - a)[0]))
                    along[k, j] = min(s_, L - s_)
            else:
                cop_margin[k, j] = statics.support_margin(c, M.pad_polygon(side))
    omega = qd[:, M.act_dofs]
    return dict(tau=tau, omega=omega, cop_margin=cop_margin, friction=friction, normal=normal, qd=qd, qdd=qdd,
                total_friction=total_friction, rolling_cop_off_line=off_line, rolling_cop_along=along,
                split_failed_frames=failed)


def _passive_ratios(M: statics.Model, design, tau, omega, groups=("neck", "tail")) -> dict:
    """The neck and tail ratios with the MJCF's joint damping and friction loss added to the inverse-dynamics torque
    (tau + b omega + f sign(omega)): informational, as Stage 0 leaves them out (research Bottom line 3)."""
    m = M.m
    out = {}
    for grp, jn in (("neck", "neck_pitch"), ("tail", "tail_yaw")):
        if grp not in groups:
            continue
        i = M.names.index(jn)
        dof = m.jnt_dofadr[statics.jid(m, jn)]
        tp = tau[:, i] + m.dof_damping[dof] * omega[:, i] + m.dof_frictionloss[dof] * np.sign(omega[:, i])
        a = design.act(grp)
        out[grp] = dict(damping=float(m.dof_damping[dof]), frictionloss=float(m.dof_frictionloss[dof]),
                        cont_ratio=float(np.sqrt(np.mean(tp**2)) / a.sustained),
                        peak_ratio=float(sizing.PEAK_MARGIN * np.max(np.abs(tp)) / a.peak))
    return out


def _clearance_gates(src: dict, traj: dict) -> dict:
    """The round-two clearance screen on one rung (``clearance.screen``, gates "r2", the rung standing in for the
    nominal walk) on the round-two MJCF: its swing gate, joint-range margins and interference along the walk."""
    from ..stage1 import plant

    xml, _ = plant.model_files(src["body"])
    plant.verify(src["body"], xml)
    cl = clearance.screen(str(xml), src["q_home"], {"nominal": dict(q=traj["q"], phases=traj["phases"],
                                                                    ik_ok=traj["ik_ok"])}, H=src["H"], gates="r2")
    s = cl["summary"]
    sw = s["swing_clearance"]["nominal"]
    mg = s.get("joint_range_margins") or {}
    tight = min(((j, v) for j, r in mg.items() for v in (r.get("margin_min"), r.get("margin_max")) if v is not None),
                key=lambda x: x[1], default=(None, None))
    gates = {k: s["pass"].get(k) for k in ("swing_clearance", "joint_range_margin", "interference_free_home",
                                           "interference_free_nominal_gait", "component_overlap_free", "tail_clearance")}
    return dict(gates=gates, walking_gates=all(v is True for v in gates.values()),
                mid_swing_min_frac_H=sw["mid_swing_min_frac_H"], mid_swing_min_mm=sw["mid_swing_min_mm"],
                mid_swing_threshold_frac_H=clearance.SWING_MIN_FRAC, lateral_gap_mm=sw["lateral_gap_mm"],
                tightest_joint=list(tight), joint_range_margin_rule_rad=clearance.RANGE_MARGIN_RAD,
                joint_range_margins=mg, nominal_gait_interference_pairs=s["counts"]["gait_nominal"]["interference_pairs"],
                tail_min_frac_H=s["tail_min_height"]["frac_H"], screen="designlab.clearance.screen (gates r2)")


def price(src: dict, arrs: dict, info: dict, with_clearance: bool = True) -> dict:
    """One rung's Stage 0 price (module docstring): drive ratios against the record's drives (all groups; the neck and
    tail also with the passive terms), contact (CoP margins, friction, the split), the clearance screen's gates and
    the plan's verdict (``LIMIT_RULE``)."""
    M = src["M"]
    tr = trajectory(arrs, info)
    with accuracy(ACCURACY):  # the exact unwrap, as the arrays' qd
        idr = look_inverse_dynamics(M, tr, arrs["rolling"], arrs["toe_edge"])
    bus = src["record"]["selection"]["bus"]
    ms = sizing._motor_space(M.names, idr["tau"], idr["omega"], src["layout"])
    drives = envelope._ratios(src["design"], bus, ms)
    worst = max(drives.items(), key=lambda kv: kv[1]["worst"])
    single = [k for k, ph in enumerate(tr["phases"]) if len(ph) == 1]
    cm = idr["cop_margin"]
    weight = M.mass() * G
    rol = np.asarray(arrs["rolling"], bool)
    ds = [k for k, ph in enumerate(tr["phases"]) if len(ph) == 2]
    contact = dict(
        single_support_cop_margin_min_mm=float(1e3 * np.nanmin(cm[single])) if single else None,
        double_support_flat_cop_margin_min_mm=(float(1e3 * np.nanmin(cm[ds])) if ds and np.isfinite(cm[ds]).any()
                                               else None),
        rolling_frames=int(np.count_nonzero(rol.any(axis=1))),
        rolling_cop_off_contact_max_mm=(float(1e3 * np.nanmax(idr["rolling_cop_off_line"]))
                                        if np.isfinite(idr["rolling_cop_off_line"]).any() else None),
        rolling_cop_along_min_mm=(float(1e3 * np.nanmin(idr["rolling_cop_along"]))
                                  if np.isfinite(idr["rolling_cop_along"]).any() else None),
        rolling_normal_max_N=float(np.max(idr["normal"][rol])) if rol.any() else None,
        friction_ratio_max=float(np.max(idr["friction"])), split_failed_frames=len(idr["split_failed_frames"]),
        total_normal_frac_weight=[float(np.min(idr["normal"].sum(axis=1)) / weight),
                                  float(np.max(idr["normal"].sum(axis=1)) / weight)],
        total_friction_need_max=float(np.max(idr["total_friction"])), cop_rule_mm=1e3 * COP_RULE_M)
    out = dict(dynamics="look_inverse_dynamics (gait.inverse_dynamics with F5's rolling contact; accuracy r2)",
               drives_at=dict(bus_v=bus, margins=dict(peak=sizing.PEAK_MARGIN, speed=sizing.SPEED_MARGIN)),
               drives=drives, worst_group=worst[0], worst_ratio=float(worst[1]["worst"]),
               passive=_passive_ratios(M, src["design"], idr["tau"], idr["omega"]), contact=contact)
    over = [dict(limit=f"{g} {k.replace('_ratio', '')}", value=float(v[k]))
            for g, v in drives.items() for k in ("cont_ratio", "peak_ratio", "speed_ratio") if v[k] > 1.0]
    if not tr["ik_ok"]:
        over.append(dict(limit="IK", value=len(tr["ik_fail_frames"])))
    if single and contact["single_support_cop_margin_min_mm"] < 1e3 * COP_RULE_M:
        over.append(dict(limit="single-support CoP off the pad (envelope rule, -10 mm)",
                         value=contact["single_support_cop_margin_min_mm"]))
    if contact["split_failed_frames"]:
        over.append(dict(limit="rolling-contact split did not solve", value=contact["split_failed_frames"]))
    if with_clearance:
        out["clearance"] = cl = _clearance_gates(src, tr)
        for k, v in cl["gates"].items():
            if v is not True:
                val = cl["mid_swing_min_frac_H"] if k == "swing_clearance" else (
                    cl["tightest_joint"][1] if k == "joint_range_margin" else None)
                over.append(dict(limit=f"clearance {k}", value=val))
    out["over"] = over
    out["within_limits"] = not over
    out["verdict"] = ("within the Stage 0 limits" if not over else
                      "over " + "; ".join(f"{o['limit']} {o['value']:.3f}" if isinstance(o["value"], float) else
                                          f"{o['limit']}" for o in over) + f": under the plan's rule, {LIMIT_RULE}")
    return out


# ------------------------------------------------------------------------------------------------
# Measured look metrics (research 4.1 step 5), from a variant's arrays
# ------------------------------------------------------------------------------------------------
def _sub_frame(y, k: int, dt: float) -> float:
    """Time offset (s) of an extreme at frame k refined by the parabola through k-1, k, k+1 (periodic)."""
    n = len(y)
    a, b, c = y[(k - 1) % n], y[k], y[(k + 1) % n]
    den = a - 2 * b + c
    return 0.0 if den == 0 else float(0.5 * (a - c) / den * dt)


def _fit_phase(phase, y, harmonic: float = 1.0) -> tuple[float, float]:
    """Least-squares y = A sin(2 pi h phase + psi) + c: (A, psi)."""
    X = np.column_stack([np.sin(2 * np.pi * harmonic * phase), np.cos(2 * np.pi * harmonic * phase), np.ones(len(y))])
    a, b, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(math.hypot(a, b)), float(math.atan2(b, a))


#: A foot is on the floor where its lowest point (pads, toes and claws) is within this of it (m): above the 0.23 mm the
#: real foot can hover over F5's transition arc while it rolls (``gait.foot_geometry``), far below a swing's lift.
CONTACT_TOL = 3e-4


def _leave_time(t0: float, t1: float, z0: float, z1: float) -> float:
    """When a foot left (or met) the floor, from two frames of its lowest point's height just off it: a swing leaves
    and meets the floor with zero vertical speed, so z grows as the square of the time from the event and sqrt(z) is
    linear in time; extrapolated to sqrt(z) = 0 (t0, z0 the frame nearer the event)."""
    r0, r1 = math.sqrt(max(z0, 0.0)), math.sqrt(max(z1, 0.0))
    if r1 <= r0:
        return t0
    return t0 - r0 * (t1 - t0) / (r1 - r0)


def contact_events(arrs: dict) -> dict:
    """Each foot's floor contact detected on the frames (not the gait's phase flags): off the floor where its lowest
    point over every foot geom (``pad_floor_z``, ``toe_floor_z``) is more than ``CONTACT_TOL`` above it; per foot, its
    swings as (lift-off, touchdown) times (s, within the window), each extrapolated from the first and last two frames
    off the floor (``_leave_time``), and the duty (one less the swings' share of the window)."""
    t, dt = np.asarray(arrs["t"]), float(arrs["dt"])
    n, W = len(t), len(t) * float(arrs["dt"])
    low = np.minimum(np.asarray(arrs["pad_floor_z"]), np.nan_to_num(np.asarray(arrs["toe_floor_z"]), nan=np.inf))
    on = low <= CONTACT_TOL
    out = dict(on_floor=on, duty=[], swings=[])
    for j in range(2):
        sw = []
        if not on[:, j].all() and on[:, j].any():
            k = int(np.argmax(on[:, j]))  # a frame on the floor: walk the window from there
            for _ in range(n):
                k1 = (k + 1) % n
                if on[k, j] and not on[k1, j]:  # the first frame off the floor is k1
                    a = k1
                    m = a
                    while not on[(m + 1) % n, j]:
                        m = (m + 1) % n
                    b = m  # the last frame off the floor
                    t_lo = _leave_time(a * dt, (a + 1) * dt, low[a, j], low[(a + 1) % n, j])
                    t_td = _leave_time(b * dt, (b - 1) * dt, low[b, j], low[(b - 1) % n, j])
                    sw.append((t_lo % W, t_td % W))
                k = k1
        out["swings"].append(sorted(sw))
        out["duty"].append(float(1.0 - sum((b_ - a_) % W for a_, b_ in sw) / W))
    return out


def measure(arrs: dict, info: dict, H: float, crouch: float = gait.GaitSpec.crouch) -> dict:
    """Each feature measured on the frames (not read from the spec): bob amplitude, trough time, the dip's fall and rise
    and how much of it comes after the touchdown (F2), the bob's share of the shortening the rung's own walk enforces,
    duty, swing time and cadence from the detected floor contact (``contact_events``), lift peak, toe-off pitch at
    lift-off and its peak, the claw roll and the foot's lowest point against the floor, trunk roll amplitude and phase,
    sway, the centre of mass's peak accelerations, the tail's phase and whether its tip moves toward the stance side in
    each swing, the head's nod and settle, the stride, the step times (F14), the ZMP's margin (from the arrays' support
    hull) and the extremes' phases (research 4.1's phase chart)."""
    t, dt = np.asarray(arrs["t"]), float(arrs["dt"])
    n = len(t)
    T, duty = float(info["period"]), float(info["duty"])
    K = int(info.get("strides", 1))
    phase = np.asarray(arrs["phase"])
    sup = np.asarray(arrs["support"], bool)
    lo, td = np.asarray(arrs["lift_off_t"]), np.asarray(arrs["touchdown_t"])
    W = n * dt
    out = {}
    # F1/F2 bob: the hip midpoint's height; the trough after each touchdown as a fraction of stance
    z = np.asarray(arrs["hip_mid"])[:, 2]
    out["bob_pp_H"] = float((z.max() - z.min()) / H)
    tr_frac, falls, rises, after = [], [], [], []
    crest = float(z.max())
    for tt in td.ravel():
        if not np.isfinite(tt):
            continue
        a0, a1 = tt - 0.5 * (1 - duty) * T, tt + 0.5 * duty * T
        ks = np.flatnonzero(((t - a0) % W) <= (a1 - a0))
        if len(ks) == 0 or out["bob_pp_H"] < 1e-9:
            continue
        k = int(ks[np.argmin(z[ks])])
        tk = t[k] + _sub_frame(z, k, dt)
        tr_frac.append(((tk - tt + 0.5 * W) % W - 0.5 * W) / (duty * T))
        depth = crest - z[k]
        # F2: the dip's fall (from 2 % of its depth to the trough) and rise (to within 2 % of the crest), and the share
        # of the depth still to come at the touchdown
        b = k
        while crest - z[(b - 1) % n] > 0.02 * depth and (k - b) < n:
            b -= 1
        e = k
        while crest - z[(e + 1) % n] > 0.02 * depth and (e - k) < n:
            e += 1
        falls.append((k - b + 1) * dt)
        rises.append((e - k + 1) * dt)
        after.append(float((np.interp(tt % W, t, z, period=W) - z[k]) / depth))
    out["bob_trough_frac_stance"] = float(np.mean(tr_frac)) if tr_frac else None
    # research F2's "settling over 0.1-0.15 s": the trough's time after the touchdown
    out["bob_trough_after_touchdown_s"] = float(np.mean(tr_frac) * duty * T) if tr_frac else None
    out["bob_fall_s"] = float(np.mean(falls)) if falls else None
    out["bob_rise_s"] = float(np.mean(rises)) if rises else None
    out["bob_share_after_touchdown"] = float(np.mean(after)) if after else None
    enforced = (duty * float(info["stride_rel"]) / 2) ** 2 / (2 * (1 - crouch))
    out["bob_share_of_enforced"] = float(out["bob_pp_H"] / enforced)
    # F3 duty and swing time: from the detected floor contact; the gait's planned flags beside them
    ce = contact_events(arrs)
    out["duty_planned"] = [float(sup[:, j].mean()) for j in range(2)]
    out["duty"] = ce["duty"]
    sw_t = [(b - a) % W for j in range(2) for a, b in ce["swings"][j]]
    out["swings_detected"] = [len(s) for s in ce["swings"]]
    out["swing_time_s"] = float(np.mean(sw_t)) if sw_t else None
    out["swing_time_spread_s"] = float(np.ptp(sw_t)) if sw_t else None
    ev = sorted((b, j) for j in range(2) for _, b in ce["swings"][j])  # detected touchdowns (time, foot)
    tds = np.array([x for x, _ in ev])
    out["cadence_steps_s"] = float(len(tds) / W) if len(tds) else None  # touchdowns per second over the window
    out["step_interval_spread_s"] = float(np.ptp(np.diff(np.concatenate([tds, [tds[0] + W]])))) if len(tds) > 1 else None
    out["touchdowns_alternate"] = bool(len(ev) > 1 and all(a[1] != b[1] for a, b in zip(ev, ev[1:] + ev[:1], strict=True)))
    # F4 lift peak (the lowest pad point's height over each swing), F5 toe-off
    pad_z, pitch = np.asarray(arrs["pad_floor_z"]), np.asarray(arrs["foot_pitch_solved"])
    rol = np.asarray(arrs["rolling"], bool)
    peaks, at_lo, pmax, pk_after, ground = [], [], [], [], []
    for j in range(2):
        for a, b in zip(lo[j], td[j], strict=True):
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            dur = (b - a) % W
            ks = np.flatnonzero(((t - a) % W) < dur)
            k = int(ks[np.argmax(pad_z[ks, j])])
            peaks.append((((t[k] + _sub_frame(pad_z[:, j], k, dt)) - a) % W) / dur)
            at_lo.append(float(np.interp(a, t, pitch[:, j], period=W)))
            near = np.flatnonzero(((t - a + 0.5 * dur) % W) < dur)  # half a swing either side of the lift-off
            pmax.append(float(pitch[near, j].max()))
            if pmax[-1] > 1e-9:  # the roll's peak relative to the lift-off, and how long the foot rolled on the floor
                kp = int(near[np.argmax(pitch[near, j])])
                pk_after.append(((t[kp] + _sub_frame(pitch[:, j], kp, dt)) - a + 0.5 * W) % W - 0.5 * W)
                before = np.flatnonzero(((a - t) % W) <= duty * T)  # the stance that ends at this lift-off
                ground.append(float(np.count_nonzero(rol[before, j]) * dt))
    out["lift_peak_frac_swing"] = float(np.mean(peaks))
    out["toe_off_at_lift_off_rad"] = float(np.mean(at_lo))
    out["toe_off_at_lift_off_range_rad"] = [float(np.min(at_lo)), float(np.max(at_lo))]
    out["toe_off_peak_rad"] = float(np.max(pmax))
    # the roll's peak relative to the lift-off (0: the foot leaves the floor at its peak pitch) and its time on the floor
    out["toe_off_peak_after_lift_off_s"] = float(np.mean(pk_after)) if pk_after else None
    out["toe_off_ground_roll_s"] = float(np.mean(ground)) if ground else None
    out["foot_pitch_min_rad"] = float(pitch.min())
    stance_flat = sup & ~rol
    out["stance_pad_height_max_mm"] = float(1e3 * np.max(np.abs(pad_z[stance_flat]))) if stance_flat.any() else None
    piv = np.asarray(arrs["toe_pivot"])
    out["rolling_pivot_height_max_mm"] = float(1e3 * np.max(np.abs(piv[rol][:, 2]))) if rol.any() else None
    claw = np.asarray(arrs["on_claw"], bool)
    out["claw_roll_frames"] = int(np.count_nonzero(claw & rol))  # on the floor past the toe edge (arc or claw tip)
    # no part of the foot below the floor: the pads, and the toes and claws (no contact in the plant), which a roll
    # about the toe pad's edge alone would take below it (``toe_clearance``); the reference rolls on the claw instead
    toe_z = np.asarray(arrs["toe_floor_z"])
    out["toe_geoms_below_floor_max_mm"] = float(max(0.0, -1e3 * np.nanmin(toe_z)))
    out["toe_geoms_below_floor_frames"] = int(np.count_nonzero(np.nanmin(toe_z, axis=1) < -1e-4))
    out["foot_below_floor_max_mm"] = float(max(0.0, -1e3 * min(np.nanmin(toe_z), float(pad_z.min()))))
    # F6 trunk roll (about the trunk's long axis, relative to home) and sway
    qh = np.asarray(arrs["q_home"])[3:7]
    qc = np.array([qh[0], -qh[1], -qh[2], -qh[3]])
    roll = []
    for qt in np.asarray(arrs["q"])[:, 3:7]:
        r = gait._quat_mul(qc, qt)
        roll.append(2 * math.atan2(r[1], r[0]))
    roll = np.array(roll)
    out["trunk_roll_amp_rad"] = float(0.5 * (roll.max() - roll.min()))
    out["trunk_roll_rms_deg"] = float(math.degrees(np.sqrt(np.mean(roll**2))))
    if out["trunk_roll_amp_rad"] > 1e-9:
        amp, psi = _fit_phase(phase, roll)
        # roll = -A cos(2 pi (phase - (1 - duty/2))) = A sin(2 pi phase - 2 pi (1 - duty/2) - pi/2)
        out["trunk_roll_toward_stance"] = bool(np.cos(psi - (-2 * np.pi * (1 - duty / 2) - np.pi / 2)) > 0)
    com = np.asarray(arrs["com"])
    out["sway_amp_H"] = float(0.5 * (com[:, 1].max() - com[:, 1].min()) / H)
    # the centre of mass's peak accelerations over the periodic window (research 4.1 step 2), the x advance unwrapped
    adv = float(np.asarray(arrs["stride_motion"])[3])
    prev, nxt = np.roll(com, 1, axis=0), np.roll(com, -1, axis=0)
    prev[0, 0] -= adv
    nxt[-1, 0] += adv
    acc = (nxt - 2 * com + prev) / dt**2
    out["com_accel_horizontal_peak_ms2"] = float(np.max(np.linalg.norm(acc[:, :2], axis=1)))
    out["com_accel_vertical_peak_ms2"] = [float(acc[:, 2].min()), float(acc[:, 2].max())]
    # F9 tail: phase and the tip's motion in each swing; F10 head: nod, settle, world pitch
    tail = np.asarray(arrs["tail"])
    amp, psi = _fit_phase(phase, tail)
    out["tail_amp_rad"], out["tail_phase_rad"] = amp, psi
    tip_y = np.asarray(arrs["points"])[:, list(POINTS).index("tail_tip"), 1]
    vy = (np.roll(tip_y, -1) - np.roll(tip_y, 1)) / (2 * dt)
    toward = []
    for j, s in enumerate(SIDES):
        stance_sign = -1.0 if s == "left" else 1.0  # the left swings over the right (-y) stance foot
        for a, b in zip(lo[j], td[j], strict=True):
            if np.isfinite(a) and np.isfinite(b):
                tm = a + 0.5 * ((b - a) % W)
                toward.append(bool(np.sign(np.interp(tm % W, t, vy, period=W)) == stance_sign))
    out["tail_tip_toward_stance_in_swing"] = float(np.mean(toward))
    neck = np.asarray(arrs["neck"])
    out["nod_amp_rad"] = _fit_phase(phase, neck, 2.0)[0]
    settle = np.asarray(arrs["head_settle"])
    if settle.max() > 0:
        lags = []
        for tt in td.ravel():
            if np.isfinite(tt):
                ks = np.flatnonzero(((t - tt) % W) < 0.25 * T)
                k = int(ks[np.argmin(neck[ks])])
                lags.append(((t[k] + _sub_frame(neck, k, dt)) - tt) % W)
        out["settle_lag_s"] = float(np.mean(lags))
        out["settle_depth_rad"] = float(np.max(settle * float(info["look"]["head_settle"])))
    hp = np.asarray(arrs["head_pitch_world"])
    out["head_world_pitch_rms_deg"] = float(math.degrees(np.sqrt(np.mean((hp - hp.mean()) ** 2))))
    # F13 stride: each foot's footprint spacing (the foot origin at each touchdown)
    feet = np.asarray(arrs["feet"])
    strides = []
    for j in range(2):
        xs = [float(np.interp(tt, t, feet[:, j, 0])) for tt in td[j] if np.isfinite(tt) and t[0] <= tt <= t[-1]]
        strides += list(np.diff(xs))
    out["stride_H"] = float(np.mean(strides) / H) if strides else float(info["stride"] / H)
    # F14 step times: the merged lift-offs, left and right alternating
    ev = np.sort(lo[np.isfinite(lo)])
    steps = np.diff(np.concatenate([ev, [ev[0] + W]]))
    dev = steps / (0.5 * T) - 1.0
    first_left = bool(np.isclose(ev[0], np.nanmin(lo[0])))
    left = dev[0::2] if first_left else dev[1::2]
    right = dev[1::2] if first_left else dev[0::2]
    out["step_time_dev_max"] = float(np.max(np.abs(dev)))
    out["step_time_dev_std"] = float(np.std(dev))
    out["step_time_left_over_right"] = float((1 + left.mean()) / (1 + right.mean()))
    # the multibody ZMP's margin inside each frame's support, recomputed from the saved hull (NaN-padded)
    zmb, hull = np.asarray(arrs["zmp_mb"]), np.asarray(arrs["support_hull"])
    marg = []
    for k in range(n):
        h = hull[k][np.isfinite(hull[k][:, 0])]
        marg.append(statics.support_margin(zmb[k], h) if len(h) >= 3 else -float(np.min(np.linalg.norm(h - zmb[k], axis=1))))
    out["zmp_margin_min_mm"] = float(1e3 * np.min(marg))
    out["window_strides"] = K
    # phase chart: when each part peaks (stride fractions of its own maximum), research 4.1 step 3
    chart = {"hip_height_max": z, "trunk_roll_max": roll, "neck_max": neck, "tail_max": tail,
             "left_foot_height_max": pad_z[:, 0]}
    out["extremes_phase"] = {k: (float(np.mod(phase[int(np.argmax(v))], 1.0)) if np.ptp(v) > 1e-9 else None)
                             for k, v in chart.items()}
    out["mechanical_tells"] = dict(constant_hip_height=out["bob_pp_H"] < 1e-3, flat_feet_at_lift_off=
                                   out["toe_off_at_lift_off_rad"] < 0.01, identical_steps=out["step_time_dev_std"] < 1e-4,
                                   metronome_nod=out["nod_amp_rad"] > 0.01)
    return out


# ------------------------------------------------------------------------------------------------
# The tail-phase check (research Bottom line 4; F9's phase)
# ------------------------------------------------------------------------------------------------
def stance_yaw_moment(src: dict, traj: dict) -> dict:
    """``turning.inverse_dynamics`` on the walk as a straight run (``turning_traj``; exact split, mu 0.8): the
    single-support stance pad's free yaw moment M_z about its CoP per frame (NaN in double support), its peak and RMS,
    and the single-support friction need mu_req."""
    M, q0 = src["M"], src["q_home"]
    idr = turning.inverse_dynamics(M, turning_traj(traj, M, q0), turning.MU, split="exact", ds_friction=False)
    n = len(traj["q"])
    mz, mu = np.full(n, np.nan), np.full(n, np.nan)
    for k, ph in enumerate(traj["phases"]):
        if len(ph) == 1:
            mz[k] = idr["feet"][ph[0]]["mz"][k]
            mu[k] = idr["feet"][ph[0]]["mu_req"][k]
    return dict(mz=mz, peak_nm=float(np.nanmax(np.abs(mz))), rms_nm=float(np.sqrt(np.nanmean(mz**2))),
                mu_req_max=float(np.nanmax(mu)))


def tail_check(src: dict, spec: gait.GaitSpec, amp: float, phases=TAIL_PHASES, grid: int = TAIL_GRID) -> dict:
    """The stance pad's single-support yaw moment against the tail's phase on one walk.  The walk runs with no tail
    and with ``amp`` at each of ``phases``; per frame M_z = A + B cos(p) + C sin(p) is fitted (the tail's effect is
    linear at these amplitudes; the fit's residual and the no-tail run against A are reported), its peak |M_z| is
    minimised over ``grid`` phases, and the walk is run again at the minimiser to confirm it.  Today's phase is pi/2
    (``gait.GaitSpec.tail_phase``); research Bottom line 4 reads it as adding to the legs' yaw momentum."""
    runs = {}
    none = stance_yaw_moment(src, generate(src, replace(spec, tail_amp=0.0)))
    Z = []
    for p in phases:
        r = stance_yaw_moment(src, generate(src, replace(spec, tail_amp=amp, tail_phase=float(p))))
        runs[f"{p:.6f}"] = dict(phase_rad=float(p), peak_nm=r["peak_nm"], rms_nm=r["rms_nm"], mu_req_max=r["mu_req_max"])
        Z.append(r["mz"])
    Z = np.array(Z)
    ss = np.isfinite(Z).all(axis=0)
    X = np.column_stack([np.ones(len(phases)), np.cos(phases), np.sin(phases)])
    coef, *_ = np.linalg.lstsq(X, Z[:, ss], rcond=None)
    resid = float(np.max(np.abs(X @ coef - Z[:, ss])))
    grid_p = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    pred = np.array([np.max(np.abs(coef[0] + coef[1] * np.cos(p) + coef[2] * np.sin(p))) for p in grid_p])
    best = float(grid_p[int(np.argmin(pred))])
    conf = stance_yaw_moment(src, generate(src, replace(spec, tail_amp=amp, tail_phase=best)))

    def predicted(p):
        return float(np.max(np.abs(coef[0] + coef[1] * np.cos(p) + coef[2] * np.sin(p))))

    today, flipped = 0.5 * math.pi, -0.5 * math.pi
    pred = dict(today=predicted(today), flipped=predicted(flipped), best=predicted(best))
    return dict(amp_rad=amp, no_tail=dict(peak_nm=none["peak_nm"], rms_nm=none["rms_nm"], mu_req_max=none["mu_req_max"]),
                runs=runs, fit_max_residual_nm=resid, no_tail_vs_fit_mean_nm=float(np.max(np.abs(coef[0] - none["mz"][ss]))),
                predicted_peak_nm=pred,
                peak_vs_no_tail_rel={k: v / none["peak_nm"] - 1.0 for k, v in pred.items()},
                best_phase_rad=best, best_phase_cycles_from_today=float(((best - today) / (2 * np.pi) + 0.5) % 1.0 - 0.5),
                confirmed=dict(peak_nm=conf["peak_nm"], rms_nm=conf["rms_nm"], mu_req_max=conf["mu_req_max"]),
                today_adds_to_leg_yaw_momentum=bool(predicted(today) > none["peak_nm"]),
                flipped_lowers_peak=bool(predicted(flipped) < predicted(today)))


def zmp_passes_check(src: dict, v0: gait.GaitSpec, passes: int = LOOK_ZMP_ITERS) -> dict:
    """What V1's extra multibody-ZMP correction passes change by themselves: today's walk (V0, ``v0``) generated with
    sizing's passes and with ``passes``, compared frame by frame (the CoM reference, the root's position, the joints)
    with each one's final ZMP error (finding: V1 changes the passes as well as adding F1/F2)."""
    a = generate(src, v0)
    b = generate(src, replace(v0, zmp_iters=passes))
    hinge = [src["M"].m.jnt_qposadr[statics.jid(src["M"].m, nm)] for nm in src["M"].names]
    return dict(passes=[int(v0.zmp_iters), int(passes)],
                com_ref_max_diff_mm=float(1e3 * np.max(np.abs(a["com_ref"] - b["com_ref"]))),
                root_xyz_max_diff_mm=float(1e3 * np.max(np.abs(a["q"][:, :3] - b["q"][:, :3]))),
                joint_max_diff_rad=float(np.max(np.abs(a["q"][:, hinge] - b["q"][:, hinge]))),
                zmp_error_final_mm=[1e3 * a["info"]["zmp_error_history_m"][-1], 1e3 * b["info"]["zmp_error_history_m"][-1]])


def turning_record(body: str) -> dict:
    """The turning screen's recorded straight run of the nominal walk (``turning.screen_model``'s ``runs.straight``:
    ``turning.generate`` at yaw rate 0, exact split, mu 0.8; round two's record of the body): the single-support stance
    pads' largest |M_z| and friction need.  The tail check's run at today's phase on V0 at the nominal walk is the same
    walk through ``gait.generate`` (``tail_check``; research Bottom line 4's "turning screen's straight runs")."""
    from ..stage1 import candidate_id, case_dir

    path = case_dir(body) / "review" / "screening" / f"{candidate_id(body)}_turning.json"
    doc = json.loads(path.read_text())
    ss = {s: doc["runs"]["straight"]["feet"][s]["single_support"] for s in SIDES}
    return dict(record=path.relative_to(STUDY).as_posix(), spec=doc["spec"],
                peak_nm=max(ss[s]["mz_abs_max"] for s in SIDES), mu_req_max=max(ss[s]["mu_req_max"] for s in SIDES))


# ------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------
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


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: Flagged for the owner where a rung needs a decision or its feature falls short of the research's (``flags``).
DUTY_BAND = (0.55, 0.65)
DUTY_BAND_SPEED = 0.30


#: The research's toe-off range by lift-off (F5: 0.3-0.5 rad); below its low end a rung's F5 is flagged as reduced.
TOE_OFF_RANGE = (0.3, 0.5)


def flags(spec: gait.GaitSpec, info: dict, m: dict, pr: dict | None, rolls: dict | None = None) -> list[str]:
    """What the owner should know about a rung on the L1 page: owner decision 2 (a duty outside the spec's band below
    0.30 m/s), F5 reduced (the window's smallest roll amplitude, the pitch at lift-off by construction, below the
    research's 0.3 rad; the measured mean at the lift-offs beside it) and its claw roll, and the Stage 0 verdict where
    it is over a limit."""
    out = []
    duty, v = float(info["duty"]), float(info["speed"])
    if spec.swing_time is not None and v < DUTY_BAND_SPEED and not DUTY_BAND[0] <= duty <= DUTY_BAND[1]:
        out.append(f"owner decision 2: F3 puts the duty at {duty:.3f}, outside the spec's {DUTY_BAND[0]}-{DUTY_BAND[1]} "
                   f"band at {v:.2f} m/s; this rung needs the band widened to 0.72 below 0.30 m/s, else F3 starts at "
                   "the brisk walk")
    if spec.toe_off:
        amps = [r["amplitude"] for side in (rolls or {}).values() for r in side]
        at = min(amps) if amps else m["toe_off_at_lift_off_rad"]
        if at < TOE_OFF_RANGE[0] - 1e-9:
            out.append(f"F5 reduced: {at:.3f} rad by lift-off (the research's 0.3-0.5), as the roll has "
                       f"{1e3 * m['toe_off_ground_roll_s']:.0f} ms on the floor within {spec.toe_off_rate} rad/s and "
                       f"{spec.toe_off_accel} rad/s^2; a full toe-off at this speed needs the late single-support ZMP "
                       "moved toward the toe line, a new feature for the owner to approve")
        if m.get("claw_roll_frames"):
            out.append("F5 rolls past the toe pad's edge on to the middle claw's tip (over a transition arc, the real "
                       "foot within 0.23 mm above the floor), where the plant has no contact: CLAW_QUESTION")
    if pr is not None and not pr["within_limits"]:
        out.append(pr["verdict"])
    return out


def write(src: dict, speed: str, variant: str, spec: gait.GaitSpec, g: dict, out_dir: Path) -> dict:
    """Save one variant's arrays (``refs/<body>/<speed>/<variant>.npz``) and return its metadata entry, with its
    measures, its Stage 0 price (``price``) and its flags for the owner."""
    path = Path(out_dir) / "refs" / src["body"] / speed / f"{variant}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = arrays(src, g)
    np.savez(path, **arrs)
    info = g["info"]
    keep = {k: v for k, v in info.items() if k not in ("look",)}
    rolls = {s: [{k: v for k, v in r.items()} for r in g["look"]["rolls"][s] if 0 <= r["t_lift_off"] < info["window_s"]]
             for s in SIDES}
    t0 = time.perf_counter()
    m = _plain(measure(arrs, info, src["H"], crouch=spec.crouch))
    pr = _plain(price(src, arrs, info))
    return dict(body=src["body"], speed=speed, variant=variant, features=features_of(variant, src["body"], speed),
                file=str(path.relative_to(out_dir)), sha256=_sha(path), frames=len(g["q"]), spec=_plain(asdict(spec)),
                info=_plain(keep), ik_ok=bool(g["ik_ok"]), ik_fail_frames=len(g["ik_fail_frames"]),
                zmp_error_final_mm=1e3 * info["zmp_error_history_m"][-1],
                zmp_margin_min_mm=1e3 * info["zmp_margin_min_m"], rolls=_plain(rolls), measured=m, price=pr,
                flags=flags(spec, info, m, pr, rolls), seconds=g["seconds"], price_seconds=time.perf_counter() - t0)


def _partial_path(out_dir: Path, body: str, speed: str) -> Path:
    return Path(out_dir) / "partial" / f"{body}__{speed}.json"


def _save_partial(path: Path, res: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_plain(res), indent=1))
    tmp.replace(path)


def _entry_current(e: dict, spec: gait.GaitSpec, out_dir: Path) -> bool:
    """A saved entry is this spec's and its file is the one it records."""
    f = Path(out_dir) / e["file"]
    return e["spec"] == _plain(asdict(spec)) and f.exists() and _sha(f) == e["sha256"]


def body_info(src: dict) -> dict:
    """Per body: H, the foot geometry in the foot frame (pads, toe and heel pivot edges, and the claw that becomes the
    front contact past its touch pitch: ``gait.foot_geometry``), the point-light markers, F5's toe clearance about the
    pad edge alone (``toe_clearance``: why the reference rolls on to the claw) and the hardware question it raises."""
    geo = src["geometry"]
    feet = {s: dict(toe_pad=geo[s]["toe_pad"], toe_edge=geo[s]["toe_edge"], heel_pad=geo[s]["heel_pad"],
                    heel_edge=geo[s]["heel_edge"], pads={k: v for k, v in geo[s]["pads"].items()},
                    claw=None if geo[s]["claw"] is None else {k: v for k, v in geo[s]["claw"].items()}) for s in SIDES}
    return _plain(dict(H_m=src["H"], foot_frame="the foot body's frame at home (world-aligned)", feet=feet,
                       markers={nm: dict(body=mujoco.mj_id2name(src["M"].m, mujoco.mjtObj.mjOBJ_BODY, b), point=p)
                                for nm, b, p in src["markers"]},
                       toe_clearance=toe_clearance(src, pitch=TOE_OFF_FULL), claw_question=CLAW_QUESTION))


def _job(args) -> dict:
    """One body and speed: the tail checks (V0: today's tail; V4: F9's phase), then every rung.  Progress is saved to
    ``<out>/partial/<body>__<speed>.json`` after the tail checks and after each rung, and a run with ``resume`` keeps
    what that file holds (a rung whose spec and file still match), so a restart loses at most the rung in hand."""
    body, speed, out_dir, variants, resume = args
    out_dir = Path(out_dir)
    src = _source_cached(body)
    H = src["H"]
    t0 = time.perf_counter()
    part = _partial_path(out_dir, body, speed)
    res = dict(body=body, speed=speed, entries={}, tail={}, tail_phase=None, tail_phase_f13=None, checks={})
    if resume and part.exists():
        res.update(json.loads(part.read_text()))
    lad0 = ladder(body, H, speed)
    if not res["tail"]:
        res["tail"]["V0"] = tail_check(src, lad0["V0"], amp=lad0["V0"].tail_amp)
        if speed == "nominal":  # the same walk as the turning screen's recorded straight run
            rec = turning_record(body)
            ours = res["tail"]["V0"]["runs"][f"{0.5 * math.pi:.6f}"]
            res["tail"]["V0"]["turning_screen"] = dict(rec, today_peak_rel=ours["peak_nm"] / rec["peak_nm"] - 1.0,
                                                       today_mu_req_rel=ours["mu_req_max"] / rec["mu_req_max"] - 1.0)
        amp = feature_values(body, H, speed)["F9F10"]["tail_amp"]
        res["tail"]["V4"] = tail_check(src, lad0["V4"], amp=amp)
        res["tail_phase"] = res["tail"]["V4"]["best_phase_rad"]
        if f13_applies(body, speed):  # F13's own tail phase: the check on V4 with F13's stride
            res["tail"]["V4F13"] = tail_check(src, replace(lad0["V4"], stride_rel=F13_STRIDE), amp=amp)
            res["tail_phase_f13"] = res["tail"]["V4F13"]["best_phase_rad"]
        _save_partial(part, res)
        print(f"  {body} {speed}: tail checks done, F9 phase {res['tail_phase']:.4f} rad"
              + (f" (with F13 {res['tail_phase_f13']:.4f})" if res["tail_phase_f13"] is not None else ""), flush=True)
    if "zmp_passes" not in res["checks"]:
        res["checks"]["zmp_passes"] = zmp_passes_check(src, lad0["V0"])
        _save_partial(part, res)
    lad = ladder(body, H, speed, res["tail_phase"], tail_phase_f13=res["tail_phase_f13"])
    for v in variants:
        if v not in lad:
            continue
        if v in res["entries"] and _entry_current(res["entries"][v], lad[v], out_dir):
            continue
        g = generate(src, lad[v])
        res["entries"][v] = e = write(src, speed, v, lad[v], g, out_dir)
        _save_partial(part, res)
        print(f"  {body} {speed} {v}: {e['seconds']:.0f} s, ik {e['ik_ok']}, zmp err {e['zmp_error_final_mm']:.2f} mm, "
              f"margin {e['zmp_margin_min_mm']:.1f} mm", flush=True)
    res["seconds"] = time.perf_counter() - t0
    return res


_SOURCES: dict = {}


def _source_cached(body: str) -> dict:
    if body not in _SOURCES:
        _SOURCES[body] = source(body)
    return _SOURCES[body]


def _merge(out_dir: Path, results: list, infos: dict, t0: float, prov: dict) -> dict:
    """Merge finished jobs into ``look_refs.json`` and ``tail_phase.json`` (rung by rung; other bodies, speeds and rungs
    are kept), with the run's provenance ``prov`` (taken before it wrote anything)."""
    refs_path, tail_path = out_dir / "look_refs.json", out_dir / "tail_phase.json"
    doc = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    tdoc = json.loads(tail_path.read_text()) if tail_path.exists() else {}
    doc.setdefault("bodies", {})
    doc.setdefault("body_info", {})
    tdoc.setdefault("bodies", {})
    doc.setdefault("checks", {})
    for r in results:
        doc["bodies"].setdefault(r["body"], {}).setdefault(r["speed"], {}).update(r["entries"])
        doc["checks"].setdefault(r["body"], {})[r["speed"]] = r.get("checks", {})
        tdoc["bodies"].setdefault(r["body"], {})[r["speed"]] = dict(
            tail_phase_F9_rad=r["tail_phase"], **({} if r.get("tail_phase_f13") is None else
                                                  dict(tail_phase_F9_F13_rad=r["tail_phase_f13"])), **r["tail"])
    doc["body_info"].update(infos)
    common = dict(accuracy=ACCURACY, values=VALUES, variants=list(VARIANTS), extras=list(EXTRAS),
                  speeds={s: ("0.40 m/s" if s == "v040" else s) for s in SPEEDS}, provenance=prov,
                  versions=_versions())
    doc.update(common, generator="gait.generate (look fields; designlab/stage2/look.py)",
               note="kinematic references with perfect tracking; priced by the Stage 0 screen before rendering; "
                    "nothing adopted (P24)", limit_rule=LIMIT_RULE, deferred=DEFERRED,
               price_rule=(price.__doc__ or "").split("\n\n")[0].replace("\n", " "),
               seconds=time.perf_counter() - t0)
    tdoc.update(common, method=(tail_check.__doc__ or "").split("\n\n")[0].replace("\n", " "),
                convention="tail = tail_amp sin(2 pi phase + tail_phase), phase 0 at the left foot's lift-off; a "
                           "positive tail_yaw swings the tip to the robot's right (axis about +z); today's tail_phase "
                           "is pi/2")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, d in ((refs_path, doc), (tail_path, tdoc)):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_plain(d), indent=1))
        tmp.replace(path)
    return doc


def _info_job(body: str) -> tuple[str, dict]:
    return body, body_info(_source_cached(body))


def prepare(bodies=BODIES, out_dir: Path = OUT, speeds=SPEEDS, variants=VARIANTS + EXTRAS, procs: int = 1,
            resume: bool = False) -> dict:
    """Generate, measure and save the ladder; ``<out>/look_refs.json`` and ``<out>/tail_phase.json`` are updated in
    place as each body and speed finishes (``_merge``), and its partial file is then removed.  With ``resume``, each
    job keeps the rungs its partial file holds (``_job``)."""
    import multiprocessing as mp

    from ..stage1 import plant

    out_dir = Path(out_dir)
    prov = plant.provenance()  # before this run writes anything
    jobs = [(b, s, str(out_dir), tuple(variants), resume) for b in bodies for s in speeds]
    t0 = time.perf_counter()
    infos = dict(_info_job(b) for b in bodies)
    doc = _merge(out_dir, [], infos, t0, prov)
    if procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            for r in pool.imap_unordered(_job, jobs, chunksize=1):
                doc = _merge(out_dir, [r], {}, t0, prov)
                _partial_path(out_dir, r["body"], r["speed"]).unlink(missing_ok=True)
                print(f"  {r['body']} {r['speed']}: merged ({r['seconds']:.0f} s)", flush=True)
    else:
        for j in jobs:
            r = _job(j)
            doc = _merge(out_dir, [r], {}, t0, prov)
            _partial_path(out_dir, r["body"], r["speed"]).unlink(missing_ok=True)
    part = out_dir / "partial"
    if part.is_dir() and not any(part.iterdir()):
        part.rmdir()
    return doc


def remeasure(out_dir: Path = OUT) -> dict:
    """``measure`` again on every saved variant (each file checked against its recorded sha256 first); rewrites the
    measured blocks of ``look_refs.json``."""
    path = Path(out_dir) / "look_refs.json"
    doc = json.loads(path.read_text())
    for d in doc["bodies"].values():
        for entries in d.values():
            for e in entries.values():
                f = Path(out_dir) / e["file"]
                if _sha(f) != e["sha256"]:
                    raise ValueError(f"{f} is not the file look_refs.json records")
                with np.load(f, allow_pickle=False) as z:
                    arrs = {k: z[k] for k in z.files}
                H = e["info"]["stride"] / e["spec"]["stride_rel"]
                e["measured"] = _plain(measure(arrs, dict(e["info"], look=e["spec"]), H, crouch=e["spec"]["crouch"]))
    path.write_text(json.dumps(_plain(doc), indent=1))
    return doc


def reflag(out_dir: Path = OUT) -> dict:
    """``flags`` again on every saved entry, from its recorded spec, info, measures, price and rolls (nothing is
    regenerated: the arrays, measures and prices stay as recorded); rewrites the flags of ``look_refs.json`` and
    records the pass's provenance (``reflag``)."""
    from ..stage1 import plant

    path = Path(out_dir) / "look_refs.json"
    doc = json.loads(path.read_text())
    for d in doc["bodies"].values():
        for entries in d.values():
            for e in entries.values():
                spec = gait.GaitSpec(**e["spec"])
                e["flags"] = flags(spec, e["info"], e["measured"], e["price"], e["rolls"])
    doc["reflag"] = dict(provenance=plant.provenance(), rule=(flags.__doc__ or "").split("\n\n")[0].replace("\n", " "))
    path.write_text(json.dumps(_plain(doc), indent=1))
    return doc


def _versions() -> dict:
    import platform

    import scipy

    return dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                scipy=scipy.__version__)


def g2_hashes(body: str, names=("nominal", "brisk", "fr05")) -> dict:
    """Stage 1's references of ``body`` regenerated as ``stage1.refs.prepare`` builds them (``refs.generate`` at r2,
    ``refs.write``'s npz), each file's sha256 (gate G2 compares them with ``stage1/runs/2026-11-08/<body>/refs.json``;
    run with one BLAS thread, as the Stage 1 runner)."""
    from ..stage1 import refs

    src = refs.source(body)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            g = refs.generate(src, name)
            e = refs.write(src, g, Path(tmp))
            out[name] = dict(sha256=e["sha256"], matches_records=e["matches_records"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.look", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="generate, measure and save the look ladder")
    p.add_argument("--body", action="append", choices=BODIES)
    p.add_argument("--speed", action="append", choices=SPEEDS)
    p.add_argument("--variant", action="append", choices=VARIANTS + EXTRAS)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="keep the rungs each job's partial file holds")
    q = sub.add_parser("g2", help="Stage 1's nominal, brisk and fr05 arrays' sha256 (JSON on stdout)")
    q.add_argument("--body", required=True)
    q.add_argument("--name", action="append", help="reference names (default: nominal, brisk, fr05)")
    r = sub.add_parser("remeasure", help="recompute look_refs.json's measured blocks from the saved arrays")
    r.add_argument("--out", default=str(OUT))
    f = sub.add_parser("reflag", help="recompute look_refs.json's flags from its recorded entries")
    f.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(tuple(a.body or BODIES), Path(a.out), tuple(a.speed or SPEEDS), tuple(a.variant or VARIANTS + EXTRAS),
                a.procs, a.resume)
    elif a.cmd == "remeasure":
        remeasure(Path(a.out))
    elif a.cmd == "reflag":
        reflag(Path(a.out))
    else:
        print(json.dumps(g2_hashes(a.body, tuple(a.name or ("nominal", "brisk", "fr05")))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
