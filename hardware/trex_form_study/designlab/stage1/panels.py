"""Stage 1 behaviour panels and gate G7 (``STAGE1_PLAN.md`` section 3 step 8, section 4, section 7): the controller
rows R0, R1 and R2 on every condition of section 4, 20 episodes each, judged by section 4's success criteria, with the
pass rule, decision 20's fall attribution, the verdicts and a summary grid per body.

Run from hardware/trex_form_study (the fit's venv)::

    python -m designlab.stage1 eval --body 60-N-T-10 [--rows R0,R1,R2/42,R2/43] [--behaviors stand,walk,...]
                                    [--conditions walk/nominal,...] [--workers 4] [--attribute] [--dry-run]
                                    [--episodes 20] --out stage1/runs/2026-11-08

Conditions (``conditions``, ``CONDITIONS_RULE``): stand (20 s; hard floor, robot_floor_carpet, low pile), walk
(nominal, brisk and Fr 0.05 at mu 0.9; nominal at mu 0.54; brisk and Fr 0.05 at mu 0.54 reported), turn (the 0.35 rad/s
walking turn and the 0.5 rad/s in-place turn, each direction; the 0.5 rad/s walking turn reported), start and stop
(stand, nominal, stand, twice), push (standing and walking, four directions) and carpet (the carpet walk on
robot_floor_carpet and low pile; medium and high pile reported).  The plans and their lengths are ``Planner``'s.

Rows (``ROWS_RULE``): R0 and R1 run the hard-floor conditions only, R2/42 and R2/43 every condition.  The row
interface (``Row``) is what the later R3 and R4 rows implement: ``Row.episode`` runs one episode of a condition from
the condition's built plant, track, seeds and push (``Context.built``, ``episode_seeds``) and returns
``control.run``'s outcome and a filled ``metrics.Recorder``; ``Row.run`` judges it with ``judge`` (every row the same
success judgement).  R3 is a ``TrackerRow`` in mode R2 with a ``Policy`` that sets the tracker's residual
(``Policy``; a zero residual is R2 bit for bit, ``ZeroResidual``); R4 needs its own ``episode``.

Episodes (``EPISODE_RULE``): seeds 3042-3061, each from the settled stance with tune's start noise and R2's sensor
noise drawn from ``tune.derived_seed``, commanded 5 % above (``tune.COMMAND_RATE``).  Judgement: ``judge`` with
``STAND_RULE``, ``WALK_RULE``, ``TURN_RULE``, ``START_STOP_RULE``, ``PUSH_RULE`` and ``RECOVERY_RULE``,
``CARPET_RULE``, ``LIMITS_RULE``; reported metrics ``LOOK_RULE``.  Each condition and row gives one record
``<out>/<body>/eval/<row>/<behavior>/<condition>.json`` (``record_path``): per-episode metrics and failed terms, the
counts with their one-sided Clopper-Pearson lower bounds (``LCB_RULE``), the pass (``PASS_RULE``), the failure label
(``FAILURE_LABEL_RULE``: decision 20's attribution for falls, ``PANEL_ATTRIBUTION_RULE``; the term classes for
success failures, ``SUCCESS_LABEL_RULE``), the rules, provenance and wall time.  ``summarize`` writes
``<out>/<body>/eval/summary.json``: the grid with the verdicts (``verdict``, ``VERDICT_RULE``) and G7 (``g7``,
``G7_RULE``).

Running (``evaluate``): ``tune.Pool`` with ``--workers`` processes, one job per episode, deterministic by episode
seed; a condition whose record exists with the same key (``RESUME_RULE``) is skipped; ``dry_run`` lists every
episode and estimates the core-hours from a measured cost per simulated second (``COST_RULE``).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import math
import os
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .. import carpet
from . import BODIES, STUDY, control, env, estimator, gates, metrics, plant, tune
from .refs import Reference
from .transitions import STOP_RULE, STOP_SPEED_FRAC, Transition, _settled_from, stop_command_phase

SCHEMA = 1
SEEDS = tuple(range(3042, 3062))
RUNS = len(SEEDS)
MIN_SUCCEED = 17
LCB_ALPHA = 0.05
RATE = tune.COMMAND_RATE
ROWS = ("R0", "R1", "R2/42", "R2/43")
#: The rows whose records the verdict reads (each CMA-ES seed's R2; later the R3 seeds, ``verdict``).
VERDICT_ROWS = ("R2/42", "R2/43")
#: The controllers (a row's name before its seed) whose falls ``--attribute`` attributes (decision 20).
ATTRIBUTED = ("R2",)
BEHAVIORS = ("stand", "walk", "turn", "start_stop", "push", "carpet")
STAND_S = 20.0
STAND_TILT_DEG = carpet.UPRIGHT_TILT_DEG
MU_LOW = 0.54
MU_CARPET = env.TASK_FRICTION["carpet_walk"]
WALK_RANGE_RAD = tune.WALK_RANGE_RAD
#: Pushes (``PUSH_RULE``).
PUSH_S = 0.1
PUSH_FRAC = 0.5
PUSH_DIRECTIONS = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
PUSH_STAND_AT_S = 2.0  # standing: the pulse starts this long after the episode's start ...
PUSH_STAND_SPREAD_S = 1.0  # ... plus up to this, drawn per seed
PUSH_WALK_AT_STRIDES = 1.0  # walking: this many commanded strides after the start transition's steady point ...
PUSH_WALK_SPREAD_STRIDES = 1.0  # ... plus up to one more stride, drawn per seed (the gait phase it lands in)
RECOVERY_S = 3.0
PUSH_HOLD_S = 2.0
STAND_STOPPED_FRAC = STOP_SPEED_FRAC
WALK_BACK_FRAC = metrics.START_SPEED_FRAC
#: Start and stop (``START_STOP_RULE``).
START_STOP_CYCLES = 2
START_STOP_STRIDES = 1
START_STOP_STAND_S = 2.0
#: Brisk and Fr 0.05 walks end in their steady gait; their window is the last 10 s (``WALK_RULE``).
CLOCK_SLACK = control.PHASE_RATE_LIMIT
WINDOW_MARGIN_STRIDES = 0.5
#: Episode slots: the start, sensor-noise and push seeds of an episode are tune.derived_seed(seed, slot, purpose).
SLOTS = {"walk": 0, "carpet": 0, "walking_turn": 1, "in_place_turn": 2}
PURPOSES = {"start": 0, "noise": 1, "push": 2}
#: Measured with tune.measure (the fit's and G5's measures); the others are judged here.
TUNE_JUDGED = ("walk", "walking_turn", "in_place_turn", "carpet")
LIMIT_TERMS = ("drive_sustained", "drive_speed", "joint_range")
#: The success terms that are hardware limits (``SUCCESS_LABEL_RULE``); every other term is the controller's.
HARDWARE_TERMS = ("drive_sustained", "drive_speed", "joint_range", "slip_median", "slip_worst",
                  "no_leg_interference")
COST_SAMPLE_S = 6.0
LOOK_JOINTS = {"head": "neck_pitch", "tail": "tail_yaw"}
#: BEHAVIOR_SPEC section 1's head and tail motion (rad, amplitude), reported against.
LOOK_BANDS = {"head": (0.05, 0.15), "tail": (0.10, 0.20)}
SWAY_SPEC_H = 0.12

# ------------------------------------------------------------------------------------------------
# Rules (each copied into every record)
# ------------------------------------------------------------------------------------------------
CONDITIONS_RULE = (
    "per body, plan section 4: stand 20 s on hard floor, robot_floor_carpet and low pile; walk nominal, brisk and Fr "
    "0.05 at mu 0.9 (the hard floor as compiled) and nominal at mu 0.54 (control.set_friction), brisk and Fr 0.05 at "
    "mu 0.54 reported; turn: the 0.35 rad/s walking turn and the 0.5 rad/s in-place turn, left and right (each a "
    "condition), the 0.5 rad/s walking turn reported; start and stop: stand, nominal, stand, twice; push: standing "
    "and walking, forward, backward, left and right (each a condition); carpet: the carpet walk "
    f"(env.TASKS['carpet_walk'], the carpet swing at mu {MU_CARPET:g}) on robot_floor_carpet and low pile, medium and "
    "high pile reported.  Gated conditions decide the verdicts; reported ones are run and judged the same way but "
    "decide nothing.  R0 and R1 run the conditions on the hard floor (any friction); R2 every condition.  Piles are "
    "plan section 2's evaluation piles (carpet.floor_variants: 4 mm probes; the 8 mm training pile is PPO's)")
ROWS_RULE = (
    "R0: the reference with feedforward and servo PD only, played on the commanded clock: control.Tracker with no "
    "feedback term and no ankle cap (terms=(), ankle_cap=(); its R1 mode, whose state view then sets nothing), which "
    "commands exactly R0's targets (control.Tracker's R0 mode refuses another clock than the references' own; at rate "
    "1 the two are the same bit for bit).  R0 has no balance, step or capture: in a push it holds the reference "
    "through the servos' PD alone, in a turn it plays the turn reference open loop; both are judged by the same "
    "rules.  R1: the tracker on the true state with the body-level G5 selection (env.tuned_gains, env.GAINS_RULE).  "
    "R2/42 and R2/43: the tracker on the estimate with each CMA-ES seed's own G5-selected candidate "
    "(tune.json seeds[s].selected_gains).  Every row: control.TERMS (R1, R2), the decided ankle cap and inward bound, "
    f"the capture margin as selected (0.0, tune.FIXED_RULE), clock rate {RATE:g}")
EPISODE_RULE = (
    f"seeds {SEEDS[0]}-{SEEDS[-1]} for every condition and row (pushes paired by seed); episode seed s of a condition "
    "starts from the plant's settled stance with tune's start noise (joint angles +-"
    f"{tune.START_JOINT_NOISE_RAD:g} rad, root velocity +-{tune.START_ROOT_VELOCITY_NOISE_M_S:g} m/s) drawn from "
    "tune.derived_seed(s, slot, 0) and R2's sensor noise (estimator.SensorNoise defaults) from tune.derived_seed(s, "
    "slot, 1), slot the episode kind's place in tune's mix (walk and carpet walk 0, walking turn 1, in-place turn 2, "
    "every other condition 0), so a panel episode on a G5 validation seed is G5's episode; a push's timing is drawn "
    "from tune.derived_seed(s, slot, 2); the nominal plant (no randomisation, no latency), the floor and friction of "
    f"the condition, commanded {100 * tune.COMMAND_MARGIN:g} % above (tune.COMMAND_RATE, control.COMMAND_RATE_RULE); "
    "a fall is plant.Plant.fall_cause (an episode during which MuJoCo warned counts as a fall at its start, "
    "tune.FAILURE_RULE); an episode that raises, or whose worker is lost twice, is a fall at its start with the "
    "reason (FAILED_EPISODE_RULE)")
FAILED_EPISODE_RULE = ("an episode that raises, or whose worker process dies or does not answer within "
                       f"{tune.EVAL_TIMEOUT_S:g} s on each of {tune.ATTEMPTS} attempts (tune.Pool), is recorded as a "
                       "fall at its start with every success term failed and the traceback or reason")
LIMITS_RULE = (
    "every success also needs every drive's motor-space RMS at most its sustained torque and every drive's peak "
    "speed at most speed_at(24 V) (control.run's worst ratios), and every joint keeping its required usable range "
    f"(walks, stands, pushes, starts and stops {WALK_RANGE_RAD:g} rad; turns decision 14's margin, "
    "tune.SUCCESS_RULE): for walks, turns and the carpet walk read as tune.SUCCESS_RULE reads them (the drive RMS over "
    "the judged window, peak speed and range over the episode); for stands, pushes and starts and stops over the "
    "whole episode")
STAND_RULE = (f"a stand succeeds if it survives {STAND_S:g} s and the trunk's tilt from home stays below "
              f"{STAND_TILT_DEG:g} deg at every feedback update (control.run's tilt_from_home_max_deg), with "
              "LIMITS_RULE; the reference holds the start transition's first frame (the home stance)")
WALK_RULE = (
    "the nominal walk at mu 0.9 is tune's walk episode exactly (start, 7 nominal strides, stop; tune.build_spec), and "
    "at mu 0.54 the same; brisk and Fr 0.05 walk from the start through the speed changes (start, 1 nominal stride, "
    "nominal_to_brisk, [1 brisk stride, brisk_to_fr05,] K strides of the target gait) and end in it, K the fewest "
    "strides that keep the last 10 s in the target gait's steady part even with the clock "
    f"{100 * CLOCK_SLACK:g} % slow (plus {WINDOW_MARGIN_STRIDES:g} stride); judged by tune.measure and "
    "tune.SUCCESS['walk'] over the 10 s before the stop command (nominal) or the episode's last 10 s (brisk, Fr "
    "0.05), the speed against the target gait's speed (commanded 5 % above); cadence and stride are judged at "
    "nominal only (the Walk row's 'at nominal'), the brisk and Fr 0.05 walks drop those two terms")
TURN_RULE = (
    "the 0.35 rad/s walking turn and the 0.5 rad/s in-place turn are tune's episodes exactly (tune.episode_plan, "
    "tune.build_spec), each direction its own condition (the slot of tune's mix, so G5's validation episodes are "
    "these conditions' episodes); the 0.5 rad/s walking turn runs the 0.35 turn's plan with the 0.5 references "
    "(start, entry, 5 strides, exit, stop); judged by tune.measure and tune.SUCCESS (yaw rate over the 180 deg of "
    "steady turning at least the target, the walk limits over it, decision 14's joint margins, legs.LegGap at 50 Hz, "
    "in-place drift)")
START_STOP_RULE = (
    f"stand, nominal, stand, twice: (start, {START_STOP_STRIDES} nominal stride, stop, {START_STOP_STAND_S:g} s "
    f"standing) x {START_STOP_CYCLES}.  Each start: the forward CoM speed along the trunk's heading at the start "
    f"command, averaged over the last commanded stride, reaches {100 * metrics.START_SPEED_FRAC:g} % of the "
    f"commanded speed within {tune.START_WITHIN_STRIDES:g} strides of the start command (metrics.start_reading; the "
    "command is where the start's speed ramp begins); each stop: decision 18 (transitions.STOP_RULE; control.run's "
    f"stop readings): stopped within {tune.STOP_WITHIN_STRIDES:g} strides; acceleration (decision 10): the IMU "
    f"point's forward acceleration low-passed at {metrics.ACCEL_LP_HZ:g} Hz at most {tune.ACCEL_M_S2:g} m/s^2 over "
    "[start command - 0.25 stride, + 2 strides] and [stop command, + 2 strides] (tune.measure's windows); with "
    "LIMITS_RULE")
PUSH_RULE = (
    f"a {PUSH_S:g} s pulse on the trunk (control.run's push, applied at the body's centre of mass) of force m x "
    f"{PUSH_FRAC:g} x the Stage 0 forward push bound / {PUSH_S:g} s, m the robot's mass, so the impulse changes the "
    "whole robot's CoM velocity by half the bound (the round-two envelope screen's one-step capturability bound, "
    "<case>/review/screening/<body>_envelope.json push_recovery.max_push_dv_ms, as plan section 5's ledger and "
    "env.PUSH_BOUND_M_S give it: 0.487 / 0.478 / 0.589 m/s), forward, backward, left or right of the home heading; "
    f"standing, the pulse starts {PUSH_STAND_AT_S:g} s plus u x {PUSH_STAND_SPREAD_S:g} s after the start; walking, "
    f"(the start transition's steady point + ({PUSH_WALK_AT_STRIDES:g} + u x {PUSH_WALK_SPREAD_STRIDES:g}) nominal "
    "strides) / the clock rate, so it lands at any phase of a stride; u uniform in [0, 1) from the episode's push "
    f"seed; the episode runs to the latest such start + {PUSH_S:g} s + {RECOVERY_S:g} s + {PUSH_HOLD_S:g} s.  "
    "Success: no fall, back to the command within 3 s (RECOVERY_RULE), with LIMITS_RULE")
RECOVERY_RULE = (
    "back to the command: there is a feedback update within "
    f"{RECOVERY_S:g} s of the pulse's end from which the body is on its command at every update to the episode's end "
    f"(at least {PUSH_HOLD_S:g} s more); on command standing: the true CoM's horizontal speed below "
    f"{100 * STAND_STOPPED_FRAC:g} % of the nominal walk's commanded speed (decision 18's 'stopped'); on command "
    "walking: the CoM's speed along the episode's starting heading, averaged over the last commanded stride (the "
    f"start's reading, metrics.start_reading), at least {100 * WALK_BACK_FRAC:g} % of the commanded speed; the time "
    "from the pulse's end is recorded")
CARPET_RULE = (
    "the carpet walk (start, 7 strides of the carpet swing reference, stop) judged as the nominal walk "
    "(tune.SUCCESS['walk'], cadence and stride included: it is a nominal-speed walk) plus the toe target: over the "
    "middle half of every reference swing in the window (clearance.MID_SWING, the screens' reading) the swing foot's "
    "lowest point, toes and claws included (metrics.window's toe clearance; on a pile its probe spheres left out, "
    "env.foot_tables), at least carpet.carpet_target(H) above the pile top, and no toe catch: that point above the "
    "pile top in every such swing (carpet.toe_catch's clears_pile_top on the simulated swing).  The pile top is the "
    "floor plane: on a pile model the uncompressed pile top (carpet.pile_model; the stance sinks below it), on "
    "robot_floor_carpet the floor")
LOOK_RULE = (
    "reported, not gated, on every walk and carpet walk over its judged window: sway = the CoM's lateral peak to peak "
    "within each whole commanded stride from the window's start, lateral to the CoM's displacement over those whole "
    f"strides, / H (mean and largest; against BEHAVIOR_SPEC's {SWAY_SPEC_H:g} H, decision 9); head and tail = the "
    "neck_pitch and tail_yaw joints' half peak to peak and their RMS and largest error against the reference at the "
    f"same reference time (against BEHAVIOR_SPEC's +-{LOOK_BANDS['head'][0]:g}-{LOOK_BANDS['head'][1]:g} rad and "
    f"+-{LOOK_BANDS['tail'][0]:g}-{LOOK_BANDS['tail'][1]:g} rad)")
PASS_RULE = (f"a condition passes if all {RUNS} episodes survive and at least {MIN_SUCCEED} succeed "
             f"(STAGE1_PLAN.md section 4, decision 6); a record of fewer episodes or other seeds is not judged")
LCB_RULE = (f"one-sided {100 * (1 - LCB_ALPHA):g} % Clopper-Pearson lower bound on the success probability: the "
            f"{LCB_ALPHA:g} quantile of Beta(k, n - k + 1), 0 for k = 0 (20/20 -> 0.861, 17/20 -> 0.656; the "
            "repository's recovery_gate.binomial_lcb by bisection gives the same)")
SUCCESS_LABEL_RULE = (
    "a condition that survives but succeeds in too few episodes is labelled from its failed episodes' failed terms: "
    f"the hardware terms ({', '.join(HARDWARE_TERMS)}: drive RMS, drive speed, joint range, slip at the pad, the "
    "legs' clearance) against the controller's (speed, yaw rate, cadence, stride, duty, flight, GRF, roll, pitch and "
    "tilt, toe clearance and the carpet's toe terms, drift, start and stop, push recovery); body-limited by the "
    "hardware terms that failed in more than half of the failed episodes, else controller-limited; every failed term "
    "is listed with its class and count.  The CoP at a pad edge with the ankle saturated and step reach are hardware "
    "limits with no success term of their own: they enter through falls (control.HW_LIMITS, decision 20)")
PANEL_ATTRIBUTION_RULE = (
    "with --attribute, a condition of an R2 row with falls is attributed by decision 20 (gates.ATTRIBUTION_RULE, "
    "gates.attribute): every relaxable hardware limit that is a candidate of at least one of its falls "
    "(control.CANDIDATE_RULE) is lifted in turn (control.run's relax) over the same 20 seeds, and the condition is "
    "body-limited by the limits whose lifting improves survival significantly (gates.sign_test, one-sided, p < "
    f"{gates.ATTRIBUTION_ALPHA:g}), else controller-limited; that costs (lifted limits x 20) episodes; the null "
    "(1 mm/s) and the together runs of G4 are diagnostics and are not run; R0 and R1 are not attributed")
FAILURE_LABEL_RULE = (
    "a failing record's label: with falls, decision 20's attribution label (PANEL_ATTRIBUTION_RULE) or 'not "
    "attributed' until it runs (R2) or because it is not run (R0, R1); without falls, SUCCESS_LABEL_RULE's label")
VERDICT_RULE = (
    "per condition, from the records of the rows that carry the verdict (R2/42 and R2/43; later the R3 seeds): shown "
    "if every one passes, provisional if one does, else body-limited if every failing one is labelled body-limited "
    "(the limits or terms listed), not attributed if any failing one has unattributed falls, else controller-limited; "
    "incomplete while a record is missing or not judged, not run if the floor has no settled stance")
NOT_RUN_RULE = ("a condition on a floor where the body's zero-command stance does not settle (plant.Plant.settled False, "
                "plant.json floors.<floor>.stand_status: the 60 cm bodies on medium and high pile) has no settled "
                "stance to start from: it is recorded as not run, with the settle's status, and judged not at all")
RESUME_RULE = (
    "a condition's record is reused (not run again) if it holds every episode of its seeds and its key is this run's: "
    "the schema, the body, the row's description (controller, gains and their selection), the condition and its "
    "resolved plan and length, the seeds, the sha256 of every reference and transition file it loads (tune.INPUTS_RULE) "
    "and the panels' sources (PANEL_SOURCES_RULE); --attribute adds the attribution to a reused record that lacks it")
PANEL_SOURCES_RULE = ("sha256 over the designlab/stage1 modules the panels import, followed as "
                      "tune.TUNER_SOURCES_RULE follows the tuner's (from panels.py and the package's __init__.py, "
                      "transitively; env.py and gates.py among them), the module list recorded beside it")
COST_RULE = (f"the dry run's cost per simulated second: per floor, R2/42 walking the nominal walk's first "
             f"{COST_SAMPLE_S:g} s from the settled stance in this process, wall seconds over simulated seconds (the "
             "machine's load average beside it); core-hours = sum over episodes of their planned seconds x their "
             "floor's cost / 3600 (R0 and R1 at R2's cost, an upper bound); the attribution's bound lifts every "
             "relaxable limit on every R2 condition")
G7_RULE = (
    "STAGE1_PLAN.md section 3 step 8: the panels' measures run on the nominal reference itself reproduce Stage 0's.  "
    "The reference is played kinematically on the plant's model (its frames interpolated in time at the 2 ms physics "
    "step, the root carried one stride per period; no physics) into a metrics.Recorder: the CoM, feet, soles and "
    "lowest points from the frames' kinematics at 100 Hz, each foot's load the weight shared over the reference's "
    "support flags at every physics step; over 10 s from the second stride, metrics.window's cadence and LOOK_RULE's "
    "sway against Stage 0's: the reference's own (refs.json: cadence_steps_s, 2 / period; sway = 2 x gait.generate's "
    "sway_amp / H, the peak-to-peak lateral CoM of the stride, as round two reported it) within half a unit of the "
    "plan's last digit (0.0005 H, 0.005 steps/s), and equal to the plan's figures when rounded to their digits "
    "(sway 0.168 / 0.169 / 0.164 H, cadence 1.96 / 1.97 / 1.76 steps/s); duty, stride, toe clearance and the head and "
    "tail amplitudes are reported beside them")
#: The plan's G7 figures (section 3 step 8): sway (H, 3 decimals) and cadence (steps/s, 2 decimals).
G7_PLAN = {"60-N-T-10": (0.168, 1.96), "60-N-H-10": (0.169, 1.97), "75-M-H-10": (0.164, 1.76)}
G7_TOL = dict(sway_H=0.0005, cadence_steps_s=0.005)
G7_SECONDS = 10.0
RULES = dict(conditions=CONDITIONS_RULE, rows=ROWS_RULE, episode=EPISODE_RULE, failed_episode=FAILED_EPISODE_RULE,
             limits=LIMITS_RULE, stand=STAND_RULE, walk=WALK_RULE, turn=TURN_RULE, start_stop=START_STOP_RULE,
             push=PUSH_RULE, recovery=RECOVERY_RULE, carpet=CARPET_RULE, look=LOOK_RULE, pass_=PASS_RULE,
             lcb=LCB_RULE, success_label=SUCCESS_LABEL_RULE, attribution=PANEL_ATTRIBUTION_RULE,
             decision_20=gates.ATTRIBUTION_RULE, candidates=control.CANDIDATE_RULE, failure_label=FAILURE_LABEL_RULE,
             verdict=VERDICT_RULE, not_run=NOT_RUN_RULE, resume=RESUME_RULE, sources=PANEL_SOURCES_RULE,
             success=tune.SUCCESS_RULE, command_rate=control.COMMAND_RATE_RULE, stop=STOP_RULE, failures=tune.FAILURE_RULE)


#: The rules every record carries; ``rules_of`` adds its behaviour's.
COMMON_RULES = ("conditions", "rows", "episode", "failed_episode", "limits", "pass_", "lcb", "success_label",
                "attribution", "decision_20", "candidates", "failure_label", "verdict", "not_run", "resume", "sources",
                "command_rate", "failures")
JUDGE_RULES = {"stand": ("stand",), "walk": ("walk", "success", "look"), "walking_turn": ("turn", "success"),
               "in_place_turn": ("turn", "success"), "carpet": ("carpet", "walk", "success", "look"),
               "start_stop": ("start_stop", "stop"), "push_stand": ("push", "recovery"),
               "push_walk": ("push", "recovery")}


def rules_of(cond: "Condition") -> dict:
    """The rules a condition's record is judged by (``RULES``' common ones and its behaviour's)."""
    return {k: RULES[k] for k in COMMON_RULES + JUDGE_RULES[cond.judge]}


# ------------------------------------------------------------------------------------------------
# Conditions
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """One panel condition (``CONDITIONS_RULE``): its behaviour and name, whether it is gated, how its episodes are
    judged (``judge``: stand, walk, walking_turn, in_place_turn, start_stop, push_stand, push_walk, carpet), its floor
    and friction (None: the floor's own), the gait whose speed a walk targets, a turn's case and a turn's or push's
    direction, and the lengths that shape it (a stand's seconds; a push's timing and recovery)."""

    behavior: str
    name: str
    gated: bool
    judge: str
    floor: str = "hard"
    friction: float | None = None
    gait: str = "nominal"
    case: str | None = None
    direction: str | None = None
    seconds: float | None = None  # a stand's length; for any other condition, a cut (smoke checks)
    push_at: float | None = None
    push_spread: float | None = None
    recovery_s: float = RECOVERY_S
    hold_s: float = PUSH_HOLD_S

    @property
    def key(self) -> str:
        return f"{self.behavior}/{self.name}"

    @property
    def kind(self) -> str:
        """tune's episode kind (the measure and the seed slot)."""
        return {"carpet": "walk"}.get(self.judge, self.judge)

    def describe(self) -> dict:
        return dict(dataclasses.asdict(self), key=self.key)


def conditions() -> list[Condition]:
    """Every panel condition of a body, in run order (``CONDITIONS_RULE``)."""
    out = [Condition("stand", f, True, "stand", floor=f, seconds=STAND_S) for f in ("hard", "robot_floor_carpet", "low")]
    for gait, mu, gated in (("nominal", None, True), ("brisk", None, True), ("fr05", None, True),
                            ("nominal", MU_LOW, True), ("brisk", MU_LOW, False), ("fr05", MU_LOW, False)):
        out.append(Condition("walk", gait if mu is None else f"{gait}_mu{mu:.2f}", gated, "walk", friction=mu,
                             gait=gait))
    for case, judge, gated in (("walk_0.35", "walking_turn", True), ("in_place_0.50", "in_place_turn", True),
                               ("walk_0.50", "walking_turn", False)):
        for d in ("left", "right"):
            prefix = "walking" if judge == "walking_turn" else "in_place"
            out.append(Condition("turn", f"{prefix}_{case.rsplit('_', 1)[1]}_{d}", gated, judge, case=case,
                                 direction=d))
    out.append(Condition("start_stop", "twice", True, "start_stop"))
    for mode in ("standing", "walking"):
        for d in PUSH_DIRECTIONS:
            out.append(Condition("push", f"{mode}_{d}", True, "push_stand" if mode == "standing" else "push_walk",
                                 direction=d, push_at=PUSH_STAND_AT_S if mode == "standing" else PUSH_WALK_AT_STRIDES,
                                 push_spread=PUSH_STAND_SPREAD_S if mode == "standing" else PUSH_WALK_SPREAD_STRIDES))
    for f, gated in (("robot_floor_carpet", True), ("low", True), ("medium", False), ("high", False)):
        out.append(Condition("carpet", f, gated, "carpet", floor=f, friction=MU_CARPET))
    return out


def select(conds: list[Condition], behaviors=None, names=None) -> list[Condition]:
    """The conditions of ``behaviors`` (None: all) and, if given, only those whose key ("walk/nominal") or name is in
    ``names``."""
    if behaviors is not None:
        unknown = set(behaviors) - set(BEHAVIORS)
        if unknown:
            raise ValueError(f"behaviours {sorted(unknown)}: choose from {', '.join(BEHAVIORS)}")
        conds = [c for c in conds if c.behavior in behaviors]
    if names is not None:
        keys = {c.key for c in conds} | {c.name for c in conds}
        unknown = set(names) - keys
        if unknown:
            raise ValueError(f"conditions {sorted(unknown)} are not among {sorted(c.key for c in conds)}")
        conds = [c for c in conds if c.key in names or c.name in names]
    return conds


def episode_seeds(cond: Condition, seed: int) -> dict:
    """The episode's slot and its start, sensor-noise and push seeds (``EPISODE_RULE``)."""
    slot = SLOTS.get(cond.kind, 0)
    return dict(slot=slot, **{k: tune.derived_seed(seed, slot, v) for k, v in PURPOSES.items()})


def push_u(cond: Condition, seed: int) -> float:
    """The push's uniform draw in [0, 1) of an episode (``PUSH_RULE``)."""
    return float(np.random.default_rng(episode_seeds(cond, seed)["push"]).random())


# ------------------------------------------------------------------------------------------------
# Plans: the tracks of the conditions and their lengths, from the saved references and transitions
# ------------------------------------------------------------------------------------------------
def template(cond: Condition) -> list:
    """A condition's track plan with the lengths ``Planner.resolve`` fills in (None)."""
    j = cond.judge
    if j in ("stand", "push_stand"):
        return [("hold", None), ("transition", "start")]
    if j == "walk":
        if cond.gait == "nominal":
            return tune.episode_plan("walk", None)
        head = [("transition", "start"), ("ref", "nominal", 1), ("transition", "nominal_to_brisk")]
        if cond.gait == "brisk":
            return head + [("ref", "brisk", None)]
        if cond.gait == "fr05":
            return head + [("ref", "brisk", 1), ("transition", "brisk_to_fr05"), ("ref", "fr05", None)]
        raise ValueError(f"walk gait {cond.gait!r}")
    if j == "walking_turn":
        c = f"{cond.case}_{cond.direction}"
        return [("transition", "start"), ("transition", f"enter_{c}"), ("ref", c, tune.TURN_STRIDES),
                ("transition", f"exit_{c}"), ("transition", "stop")]
    if j == "in_place_turn":
        return tune.episode_plan("in_place_turn", cond.direction)
    if j == "start_stop":
        cycle = [("transition", "start"), ("ref", "nominal", START_STOP_STRIDES), ("transition", "stop"),
                 ("hold", START_STOP_STAND_S)]
        return cycle * START_STOP_CYCLES
    if j == "push_walk":
        return [("transition", "start"), ("ref", "nominal", None)]
    if j == "carpet":
        return [tuple(x) for x in env.TASKS["carpet_walk"][0]]
    raise ValueError(f"judge {j!r}")


def files_of(cond: Condition) -> list[str]:
    """The reference and transition files a condition loads (with refs/nominal.npz, omega0's)."""
    out = {"refs/nominal.npz"}
    for item in template(cond):
        if item[0] == "ref":
            out.add(f"refs/{item[1]}.npz")
        elif item[0] == "transition":
            out.add(f"transitions/{item[1]}.npz")
    return sorted(out)


def panel_files(conds=None) -> list[str]:
    """Every reference and transition file the panels load."""
    return sorted({f for c in (conditions() if conds is None else conds) for f in files_of(c)})


class Planner:
    """The resolved plans and planned lengths of the conditions on one body's saved references and transitions, without
    building a plant: the segment times follow ``control.build_track``'s rules (``segment_times``)."""

    def __init__(self, run_dir: Path, refs_doc: dict, tr_doc: dict):
        self.run_dir, self.refs_doc, self.tr_doc = Path(run_dir), refs_doc, tr_doc
        self._objs = {}

    def load(self, item):
        if item[0] == "hold":
            return None
        key = (item[0], item[1])
        if key not in self._objs:
            cls, sub = (Reference, "refs") if item[0] == "ref" else (Transition, "transitions")
            self._objs[key] = cls.load(self.run_dir / sub / f"{item[1]}.npz")
        return self._objs[key]

    def segment_times(self, plan: list) -> list[tuple[float, float]]:
        """(t0, duration) of each item on the reference clock, as ``control.build_track`` lays them out."""
        objs = [self.load(item) for item in plan]
        out, t = [], 0.0
        for i, (item, obj) in enumerate(zip(plan, objs, strict=True)):
            if item[0] == "ref":
                dur = item[2] * obj.period
            elif item[0] == "transition":
                nxt = objs[i + 1] if i + 1 < len(objs) else None
                if isinstance(nxt, Reference):
                    dt_last = nxt.step
                elif isinstance(nxt, Transition):
                    dt_last = float(nxt.t[1] - nxt.t[0])
                else:
                    dt_last = float(obj.t[-1] - obj.t[-2])
                dur = float(obj.t[-1] + dt_last)
            else:
                dur = float(item[1])
            out.append((t, dur))
            t = t + dur
        return out

    def phase_time(self, plan: list, i: int, phase: float) -> float:
        """The reference time of a transition item's phase."""
        tr = self.load(plan[i])
        return float(self.segment_times(plan)[i][0] + np.interp(phase, tr.phase, tr.t))

    def period(self, name: str) -> float:
        return self.load(("ref", name, 1)).period

    def resolve(self, cond: Condition) -> dict:
        """The condition's plan, its planned seconds (simulated) and what its judge needs: the start commands (reference
        time) of a start and stop, a push's timing, the steady point of a walk that ends in its gait.  A condition
        other than a stand that gives ``seconds`` is cut there (a smoke check: its episodes end early, unjudged)."""
        out = self._resolve(cond)
        if cond.seconds is not None and cond.judge != "stand" and cond.seconds < out["seconds"]:
            out = dict(out, seconds=float(cond.seconds), cut_from_s=out["seconds"])
        return out

    def _resolve(self, cond: Condition) -> dict:
        plan = template(cond)
        j = cond.judge
        out = dict(steady_ref_s=None, starts_ref_s=None, push=None)
        if j in ("stand", "push_stand"):
            if j == "stand":
                seconds = float(cond.seconds)
            else:
                seconds = cond.push_at + cond.push_spread + PUSH_S + cond.recovery_s + cond.hold_s
                out["push"] = dict(kind="standing", at_s=float(cond.push_at), spread_s=float(cond.push_spread))
            plan = [("hold", seconds + 1.0), plan[1]]
            return dict(out, plan=plan, seconds=seconds)
        if j == "walk" and cond.gait != "nominal":
            k = len(plan) - 1
            P = self.period(cond.gait)
            probe = plan[:k] + [("ref", cond.gait, 1)]
            t0 = self.segment_times(probe)[k][0]
            steady = self.phase_time(probe, k - 1, 1.0)
            need = (steady + WINDOW_MARGIN_STRIDES * P) / (1.0 - CLOCK_SLACK) + tune.WALK_WINDOW_S * RATE
            plan = plan[:k] + [("ref", cond.gait, max(1, math.ceil((need - t0) / P)))]
            out["steady_ref_s"] = steady
        if j == "push_walk":
            P = self.period("nominal")
            probe = [plan[0], ("ref", "nominal", 1)]
            steady = self.phase_time(probe, 0, 1.0)
            last = (steady + (cond.push_at + cond.push_spread) * P) / RATE
            seconds = last + PUSH_S + cond.recovery_s + cond.hold_s
            t0 = self.segment_times(probe)[1][0]
            plan = [plan[0], ("ref", "nominal", max(1, math.ceil(((1.0 + CLOCK_SLACK) * RATE * seconds - t0) / P) + 1))]
            out["push"] = dict(kind="walking", steady_ref_s=steady, at_strides=float(cond.push_at),
                               spread_strides=float(cond.push_spread), period_s=P)
            return dict(out, plan=plan, seconds=seconds, steady_ref_s=steady)
        times = self.segment_times(plan)
        if j == "start_stop":
            duty = float(self.tr_doc["start"]["gaits"]["b"]["duty"])
            out["starts_ref_s"] = [self.phase_time(plan, i, stop_command_phase(duty)) for i, item in enumerate(plan)
                                   if item == ("transition", "start")]
        t_end = times[-1][0] + times[-1][1]
        return dict(out, plan=plan, seconds=t_end / RATE)


def plan_is_tunes(cond: Condition, plan: list) -> bool:
    """Whether a condition's plan is tune's mix episode of its kind (then ``tune.build_spec`` builds it as G5 did)."""
    if cond.kind not in tune.EPISODES:
        return False
    return [tuple(x) for x in plan] == [tuple(x) for x in tune.episode_plan(cond.kind, cond.direction)]


# ------------------------------------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------------------------------------
R2_GAINS_RULE = ("R2/s takes tune.json's seeds[s]: the candidate G5 selected for CMA-ES seed s (selected, "
                 "selected_gains), refused if that seed has no selection")


def r2_gains(run_dir: Path, seed: int) -> tuple[np.ndarray, dict]:
    """CMA-ES seed ``seed``'s own G5 selection (``R2_GAINS_RULE``)."""
    path = Path(run_dir) / "tune.json"
    doc = json.loads(path.read_text())
    v = (doc.get("seeds") or {}).get(str(int(seed))) or {}
    if "selected" not in v:
        raise ValueError(f"{path}: CMA-ES seed {seed} has no G5 selection ({v.get('reason')})")
    gains = np.array([float(v["selected_gains"][n]) for n in control.GAIN_NAMES])
    return gains, dict(rule=R2_GAINS_RULE, file=str(path), sha256=plant.sha256(path), cma_seed=int(seed),
                       candidate=v["selected"], g5_pass=v.get("pass_"), g5_label=v.get("label"))


class Row:
    """A controller row of the panels.  What a row must provide (the later R3 and R4 rows included):

    * ``name`` ("R0", "R2/42", "R3/43", ...) and ``hard_only`` (True: it runs the hard-floor conditions only);
    * ``describe()``: a JSON description (controller, gains or policy checkpoint with their selection and sha256):
      it names the row in every record and enters the resume key, so it must change whenever the controller does;
    * ``spec()``: a picklable dict from which ``row_from_spec`` rebuilds the row in a worker process (its "class" is
      the row's dotted class path, so a row defined in another module needs no change here);
    * ``episode(ctx, cond, seed, relax)`` -> (built, out, rec): one episode of ``cond`` on ``ctx.built(cond)``'s
      plant, track and length, from the settled stance with ``episode_seeds(cond, seed)``'s start and sensor-noise
      seeds and tune's start noise, the condition's push (``Built.push``, applied before every physics step), the
      hardware limits ``relax`` lifted (``control.relaxed``), until the condition's seconds or a fall; ``out`` is
      ``control.run``'s outcome (with ``tune.warned_fall``) and ``rec`` a ``metrics.Recorder`` made by
      ``Built.recorder()`` that received ``control.run``'s hooks.  The simplest way is ``control.run`` itself on a
      ``control.Tracker`` (``TrackerRow``; R3 adds a ``Policy``); R4 can subclass ``control.Tracker`` so that
      ``control.run`` still detects falls, onsets, candidates, stops and drive statistics;
    * ``run(ctx, cond, seed, relax)`` -> the episode record: ``judge`` on the episode (the same for every row)."""

    name: str = ""
    hard_only: bool = False

    def runs(self, cond: Condition) -> bool:
        return not self.hard_only or cond.floor == "hard"

    def describe(self) -> dict:
        raise NotImplementedError

    def spec(self) -> dict:
        raise NotImplementedError

    def episode(self, ctx: "Context", cond: Condition, seed: int, relax=()):
        raise NotImplementedError

    def run(self, ctx: "Context", cond: Condition, seed: int, relax=()) -> dict:
        t0 = time.perf_counter()
        b, out, rec = self.episode(ctx, cond, int(seed), tuple(relax))
        e = judge(b, out, rec, int(seed), ctx.lg if b.legs else None)
        e.update(relaxed=list(relax), wall_s=time.perf_counter() - t0)
        return e


class Policy:
    """A learned controller on a tracker row (plan section 2's R3; ``TrackerRow(policy=...)``): ``begin(tk, built,
    seed)`` after the episode's reset and before its first physics step, ``act(tk, k)`` after physics step k for k =
    ``every`` - 1, 2 ``every`` - 1, ...: it sets ``tk.residual`` (joint offsets, rad, the plant's joint order, or None)
    and ``tk.rate_factor`` for the steps that follow, from what it observes (it may read ``tk``, ``tk.est`` and the
    plant).  ``describe()`` names it for the records."""

    every = env.ENV_SUBSTEPS

    def begin(self, tk: control.Tracker, built: "Built", seed: int) -> None:
        pass

    def act(self, tk: control.Tracker, k: int) -> None:
        pass

    def describe(self) -> dict:
        return dict(policy=type(self).__name__)


class ZeroResidual(Policy):
    """The zero residual: the tracker's residual None and rate factor 1, so the row is its tracker row bit for bit
    (env.ZERO_RESIDUAL_RULE's check, through the panels' runner)."""

    def begin(self, tk, built, seed):
        tk.residual, tk.rate_factor = None, 1.0

    def act(self, tk, k):
        tk.residual, tk.rate_factor = None, 1.0


class _Acting:
    """``control.run``'s recorder hooks forwarded to a ``metrics.Recorder``, with a ``Policy`` acting at the hooks."""

    def __init__(self, rec: metrics.Recorder, policy: Policy, built: "Built", seed: int):
        self.rec, self.policy, self.built, self.seed = rec, policy, built, seed

    def begin(self, tk, n):
        self.tk = tk
        self.rec.begin(tk, n)
        self.policy.begin(tk, self.built, self.seed)

    def step(self, k):
        self.rec.step(k)
        if (k + 1) % self.policy.every == 0:
            self.policy.act(self.tk, k)

    def update(self, *args):
        self.rec.update(*args)

    def end(self, steps, out):
        self.rec.end(steps, out)


class TrackerRow(Row):
    """A row run by ``control.Tracker`` (R0, R1, R2; with a ``Policy``, R3): ``ROWS_RULE``."""

    def __init__(self, name: str, mode: str, gains=None, gains_info: dict | None = None, terms=control.TERMS,
                 ankle_cap=control.ANKLE_CAP, hard_only: bool = False, policy: Policy | None = None,
                 rate: float = RATE, controller: str = ""):
        self.name, self.mode = name, mode
        self.gains = None if gains is None else np.asarray(gains, float)
        self.gains_info = gains_info
        self.terms, self.ankle_cap = tuple(terms), tuple(ankle_cap)
        self.hard_only, self.policy, self.rate, self.controller = bool(hard_only), policy, float(rate), controller

    def describe(self) -> dict:
        return dict(row=self.name, controller=self.controller, tracker_mode=self.mode, terms=list(self.terms),
                    ankle_cap=list(self.ankle_cap), inward_bound=control.INWARD_BOUNDS[0], rate=self.rate,
                    gains=None if self.gains is None else control.gains_dict(self.gains), gains_info=self.gains_info,
                    hard_only=self.hard_only, policy=None if self.policy is None else self.policy.describe())

    def spec(self) -> dict:
        if self.policy is not None and type(self.policy) is not ZeroResidual:
            raise ValueError("a TrackerRow with a learned policy needs its own spec (the policy's checkpoint)")
        return dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name, mode=self.mode,
                    gains=None if self.gains is None else self.gains.tolist(), gains_info=self.gains_info,
                    terms=list(self.terms), ankle_cap=list(self.ankle_cap), hard_only=self.hard_only, rate=self.rate,
                    controller=self.controller, zero_residual=self.policy is not None)

    @classmethod
    def from_spec(cls, s: dict) -> "TrackerRow":
        return cls(s["name"], s["mode"], s["gains"], s["gains_info"], s["terms"], s["ankle_cap"], s["hard_only"],
                   ZeroResidual() if s.get("zero_residual") else None, s["rate"], s.get("controller", ""))

    def episode(self, ctx: "Context", cond: Condition, seed: int, relax=()):
        b = ctx.built(cond)
        p = b.p
        tk = control.Tracker(p, b.track, self.mode, gains=self.gains, terms=self.terms, ankle_cap=self.ankle_cap,
                             rate=self.rate)
        rec = b.recorder()
        hook = rec if self.policy is None else _Acting(rec, self.policy, b, seed)
        s = episode_seeds(cond, seed)
        out = control.run(tk, b.seconds, start="stance", seed=s["start"], noise_seed=s["noise"], recorder=hook,
                          joint_noise=tune.START_JOINT_NOISE_RAD, root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S,
                          push=b.push(seed), relax=tuple(relax))
        return b, tune.warned_fall(out, tune.sim_warnings(p.d)), rec


def row_from_spec(s: dict) -> Row:
    """A row rebuilt in a worker from its ``Row.spec``."""
    mod, _, cls = s["cls"].rpartition(".")
    return getattr(importlib.import_module(mod), cls).from_spec(s)


def tracker_rows(body: str, run_dir: Path, names=ROWS) -> dict[str, TrackerRow]:
    """The rows R0, R1 and R2/s of ``ROWS_RULE`` on ``run_dir``'s tune.json."""
    out = {}
    for name in names:
        if name == "R0":
            out[name] = TrackerRow("R0", "R1", None, dict(source="R0 has no feedback: no gains"), terms=(),
                                   ankle_cap=(), hard_only=True,
                                   controller="R0: feedforward + PD on the commanded clock (ROWS_RULE)")
        elif name == "R1":
            g, info = env.tuned_gains(Path(run_dir), body)
            out[name] = TrackerRow("R1", "R1", g, info, hard_only=True,
                                   controller="R1: the tracker on the true state, the body-level G5 selection")
        elif name.startswith("R2/"):
            g, info = r2_gains(run_dir, int(name.split("/", 1)[1]))
            out[name] = TrackerRow(name, "R2", g, info,
                                   controller=f"R2: the tracker on the estimate, CMA-ES seed {info['cma_seed']}'s G5 "
                                              f"selection {info['candidate']}")
        else:
            raise ValueError(f"row {name!r}: choose from {', '.join(ROWS)} (R3 and R4 rows are added later)")
    return out


# ------------------------------------------------------------------------------------------------
# Built conditions (per worker): plant, track, length, push
# ------------------------------------------------------------------------------------------------
@dataclass
class Built:
    """A condition built on a worker's plant: the track (or tune's spec), the episode's seconds, the joint ranges and
    required margins, the judge's reference quantities and the push."""

    cond: Condition
    p: plant.Plant
    plan: list
    track: control.Track
    spec: tune.EpisodeSpec | None
    seconds: float
    resolved: dict
    lo: np.ndarray
    hi: np.ndarray
    required: np.ndarray
    H: float
    mass: float
    speed_cmd: float
    stride_s: float
    push_bound_m_s: float
    home_yaw: float

    @property
    def legs(self) -> bool:
        return self.cond.judge in ("walking_turn", "in_place_turn")

    def recorder(self) -> metrics.Recorder:
        return metrics.Recorder(legs_every=tune.LEGS_EVERY if self.legs else 0,
                                foot_tables=env.foot_tables if self.cond.floor in carpet.PILES else None)

    def push_onset(self, seed: int) -> float | None:
        """The pulse's start (simulated s) of an episode (``PUSH_RULE``)."""
        pu = self.resolved.get("push")
        if pu is None:
            return None
        u = push_u(self.cond, seed)
        if pu["kind"] == "standing":
            return pu["at_s"] + u * pu["spread_s"]
        return (pu["steady_ref_s"] + (pu["at_strides"] + u * pu["spread_strides"]) * pu["period_s"]) / RATE

    def push_force(self) -> np.ndarray | None:
        """The pulse's force (N, world xy): m x frac x bound / duration along the direction in the home heading."""
        if self.resolved.get("push") is None:
            return None
        F = self.mass * PUSH_FRAC * self.push_bound_m_s / PUSH_S
        dx, dy = PUSH_DIRECTIONS[self.cond.direction]
        c, s = math.cos(self.home_yaw), math.sin(self.home_yaw)
        return F * np.array([c * dx - s * dy, s * dx + c * dy])

    def push(self, seed: int):
        """``control.run``'s push (t0, duration, fx, fy) of an episode, or None."""
        t = self.push_onset(seed)
        if t is None:
            return None
        f = self.push_force()
        return (float(t), PUSH_S, float(f[0]), float(f[1]))


class Context:
    """One worker's plants (per floor and friction), built conditions and leg-gap oracle for a body."""

    def __init__(self, body: str, run_dir: Path, docs: dict):
        self.body, self.run_dir = body, Path(run_dir)
        self.refs_doc = docs["refs.json"]["references"]
        self.tr_doc = docs["transitions.json"]["transitions"]
        self.planner = Planner(self.run_dir, self.refs_doc, self.tr_doc)
        self._plants, self._built, self._lg = {}, {}, None

    @property
    def lg(self):
        if self._lg is None:
            from .legs import LegGap

            self._lg = LegGap(self.body)
        return self._lg

    def plant(self, floor: str, friction: float | None) -> plant.Plant:
        key = (floor, friction)
        if key not in self._plants:
            p = plant.build(self.body, floor)
            if p.settled is False:
                raise plant.PlantError(f"{self.body} does not stand on {floor} (NOT_RUN_RULE)")
            if friction is not None:
                control.set_friction(p, friction)
            self._plants[key] = p
        return self._plants[key]

    def built(self, cond: Condition) -> Built:
        if cond in self._built:
            return self._built[cond]
        p = self.plant(cond.floor, cond.friction)
        res = self.planner.resolve(cond)
        plan = res["plan"]
        omega = control.omega0(p, Reference.load(self.run_dir / "refs" / "nominal.npz").q_home)
        spec = None
        m = p.m
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in p.joints]
        lo, hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        required = np.full(len(p.joints), WALK_RANGE_RAD)
        if cond.judge in TUNE_JUDGED:
            if plan_is_tunes(cond, plan):  # tune's episode exactly (G5's build)
                spec = tune.build_spec(p, self.run_dir, cond.kind, cond.direction, omega, self.refs_doc, self.tr_doc,
                                       RATE)
            else:
                spec = tune.build_spec(p, self.run_dir, cond.kind, cond.direction, omega, self.refs_doc, self.tr_doc,
                                       RATE, plan=plan, case=cond.case, gait=cond.gait)
            track, required = spec.track, spec.required_margins
            seconds = spec.seconds if "cut_from_s" not in res else res["seconds"]
        else:
            track = control.build_track(p, self.run_dir, plan, omega)
            seconds = res["seconds"]
        nominal = self.refs_doc["nominal"]
        b = Built(cond=cond, p=p, plan=plan, track=track, spec=spec, seconds=float(seconds), resolved=res, lo=lo,
                  hi=hi, required=required, H=float(p.meta["H_m"]), mass=float(m.body_subtreemass[p.trunk]),
                  speed_cmd=float(nominal["speed_m_s"]) * RATE, stride_s=float(nominal["period_s"]) / RATE,
                  push_bound_m_s=float(env.PUSH_BOUND_M_S[self.body]), home_yaw=control._yaw(p._R_home))
        self._built[cond] = b
        return b


# ------------------------------------------------------------------------------------------------
# Judgement
# ------------------------------------------------------------------------------------------------
START_STOP_TERMS = tuple(f"{k}{i}_{t}" for i in range(1, START_STOP_CYCLES + 1)
                         for k, ts in (("start", ("speed_up", "acceleration")), ("stop", ("stopped", "acceleration")))
                         for t in ts)


def terms_of(cond: Condition) -> tuple:
    """The success terms a condition's episodes are judged on (each success needs all of them)."""
    j = cond.judge
    if j == "stand":
        return ("survived", "tilt") + LIMIT_TERMS
    if j == "walk":
        t = tune.SUCCESS["walk"]
        return t if cond.gait == "nominal" else tuple(x for x in t if x not in ("cadence", "stride"))
    if j in ("walking_turn", "in_place_turn"):
        return tune.SUCCESS[j]
    if j == "carpet":
        return tune.SUCCESS["walk"] + ("toe_target", "no_toe_catch")
    if j == "start_stop":
        return ("survived",) + START_STOP_TERMS + LIMIT_TERMS
    if j in ("push_stand", "push_walk"):
        return ("survived", "recovered") + LIMIT_TERMS
    raise ValueError(f"judge {j!r}")


def term_class(term: str) -> str | None:
    """"hardware" or "controller" (``SUCCESS_LABEL_RULE``); None for survival."""
    if term == "survived":
        return None
    return "hardware" if term in HARDWARE_TERMS else "controller"


def _compact_limits(a: dict | None) -> dict | None:
    if not a:
        return None
    return {k: a.get(k) for k in ("window_s", "updates", "hardware", "hardware_near_onset",
                                  "hardware_last_active_before_onset_s", "reported", "leg_drives", "neck_tail_drives")}


def _base(b: Built, out: dict, rec: metrics.Recorder, seed: int) -> dict:
    """The fields every episode record has: the fall, its onset and candidates, the drives, the joints."""
    margins = metrics.joint_margins(rec.Q, b.lo, b.hi) if rec.rows else np.full(len(b.lo), -1.0)
    short = b.required - margins
    w = int(np.argmax(short))
    e = dict(seed=int(seed), fell=bool(out["fell"]), fall_time_s=out["fall_time_s"], fall_cause=out["fall_cause"],
             fall_onset_s=out.get("fall_onset_s"), fall_candidates=out.get("fall_candidates"),
             limits_before_onset=_compact_limits(out.get("limits_before_onset")), survival_s=float(out["seconds"]),
             planned_s=b.seconds, strides=float(out["strides"]), worst_sustained_ratio=out["worst_sustained_ratio"],
             worst_speed_ratio=out["worst_speed_ratio"], tilt_from_home_max_deg=out.get("tilt_from_home_max_deg"),
             joint_margin_min_rad=float(margins.min()), joint_shortfall_rad=float(short[w]),
             joint_worst=b.p.joints[w], estimator=tune._est_summary(out), ankle_cap_residual=tune._cap_summary(out),
             captures=len(out.get("captures") or ()), phase_rate_range=out.get("phase_rate_range"))
    if out.get("mujoco_warnings"):
        e["mujoco_warnings"] = out["mujoco_warnings"]
    return e


def _limits_ok(e: dict) -> dict:
    """``LIMITS_RULE`` over the episode (stands, pushes, starts and stops)."""
    return dict(drive_sustained=float(e["worst_sustained_ratio"][1]) <= 1.0,
                drive_speed=float(e["worst_speed_ratio"][1]) <= 1.0, joint_range=e["joint_shortfall_rad"] <= 0.0)


WINDOW_KEYS = ("complete", "t_start_s", "t_end_s", "speed_m_s", "lateral_m", "heading_change_rad", "yaw_rate_rad_s",
               "turned_180", "duty", "flight_s", "cadence_steps_s", "stride_H", "grf_peak_bw", "roll_rms_deg",
               "pitch_rms_deg", "roll_peak_deg", "pitch_peak_deg", "toe_clearance_m", "slip_median_m", "slip_worst_m",
               "slip_contact_path_median_m", "slip_contact_path_worst_m", "drive_sustained_worst", "swings",
               "stances", "unloaded_steps")
MEASURE_KEYS = ("survived_frac", "dcm_error_rms_m", "drift_m", "legs", "window_sustained_ratio", "start",
                "stop_reading", "stop")


def judge(b: Built, out: dict, rec: metrics.Recorder, seed: int, lg=None) -> dict:
    """The episode record of one episode (the judgement of ``b.cond``'s rule): ``_base``'s fields, the success terms
    (``terms_of``), whether it succeeded, the failed terms and the behaviour's measures."""
    cond = b.cond
    e = _base(b, out, rec, seed)
    fell = e["fell"]
    j = cond.judge
    ok, measures = {}, {}
    if j in TUNE_JUDGED:
        window = None
        if j == "walk" and cond.gait != "nominal" and not fell and rec.rows:
            tb = float(rec.t[-1])
            window = (tb - tune.WALK_WINDOW_S, tb)
        res = tune.measure(b.spec, rec, out, lg, window=window)
        w = res.get("window") or {}
        measures = {k: res.get(k) for k in MEASURE_KEYS}
        measures["window"] = {k: w.get(k) for k in WINDOW_KEYS}
        ok = dict(res["success"])
        if j == "carpet":
            ct = carpet_terms(w, b.H)
            ok.update(ct.pop("ok"))
            measures["carpet"] = dict(ct, catching_swings=_catching_swings(rec, w) if w.get("complete") else None)
        if j in ("walk", "carpet") and w.get("complete"):
            measures["look"] = look(rec, w["t_start_s"], w["t_end_s"], b.H, b.spec.stride_s, b.track, b.p.joints)
    elif j == "stand":
        ok = dict(survived=not fell, tilt=(not fell) and e["tilt_from_home_max_deg"] < STAND_TILT_DEG)
        ok.update(_limits_ok(e))
    elif j == "start_stop":
        measures["start_stop"] = start_stop(b, out, rec)
        ok = dict(survived=not fell, **measures["start_stop"]["ok"])
        ok.update(_limits_ok(e))
    elif j in ("push_stand", "push_walk"):
        t_on = b.push_onset(seed)
        f = b.push_force()
        rcv = recovery(rec, t_on + PUSH_S, standing=j == "push_stand", speed_cmd=b.speed_cmd, stride_s=b.stride_s,
                       limit_s=cond.recovery_s, until_s=b.seconds)
        measures["push"] = dict(onset_s=t_on, end_s=t_on + PUSH_S, direction=cond.direction, force_N=f.tolist(),
                                dv_m_s=float(np.linalg.norm(f) * PUSH_S / b.mass), impulse_N_s=float(np.linalg.norm(f)
                                                                                                    * PUSH_S),
                                bound_m_s=b.push_bound_m_s, fell_before_pulse=bool(fell and out["fall_time_s"] < t_on))
        measures["recovery"] = rcv
        ok = dict(survived=not fell, recovered=(not fell) and rcv["recovered"])
        ok.update(_limits_ok(e))
    else:
        raise ValueError(f"judge {j!r}")
    terms = terms_of(cond)
    success = {t: bool(ok.get(t, False)) and not (fell and t != "survived") for t in terms}
    if fell:
        success["survived"] = False
    e.update(success=success, succeeded=bool(all(success.values())),
             failed_terms=[t for t, v in success.items() if not v], measures=measures)
    return e


def carpet_terms(w: dict, H: float) -> dict:
    """``CARPET_RULE``'s toe terms from a walk window (``metrics.window``): the toe target and no toe catch."""
    toe = w.get("toe_clearance_m") if w.get("complete") else None
    target = carpet.carpet_target(H)
    return dict(target_m=target, toe_clearance_m=toe, swings=w.get("swings"),
                ok=dict(toe_target=toe is not None and toe >= target, no_toe_catch=toe is not None and toe > 0.0))


def _catching_swings(rec: metrics.Recorder, w: dict) -> int:
    """Swings of the window whose foot's lowest point dips to the pile top (``CARPET_RULE``; metrics.window's
    swings)."""
    from .. import clearance

    t, U = rec.t, rec.U
    sel = (t >= w["t_start_s"] - 1e-9) & (t <= w["t_end_s"] + 1e-9)
    lows = U[:, [metrics.C["low0"], metrics.C["low1"]]]
    sw, u = U[:, metrics.C["ref_swing"]], U[:, metrics.C["ref_u"]]
    mid = sel & (u >= clearance.MID_SWING[0]) & (u <= clearance.MID_SWING[1])
    n = 0
    for i in (0, 1):
        s, e_, v = metrics.runs(mid & (sw == i))
        n += sum(1 for a, b, x in zip(s, e_, v, strict=True) if x and float(np.min(lows[a:b, i])) <= 0.0)
    return int(n)


def start_stop(b: Built, out: dict, rec: metrics.Recorder) -> dict:
    """``START_STOP_RULE``'s readings: each start (metrics.start_reading along the heading at its command), each stop
    (control.run's decision-18 reading), their acceleration peaks, and the terms."""
    S, v_cmd = b.stride_s, b.speed_cmd
    a_lp = metrics.forward_acceleration(rec) if rec.rows > 3 else None
    yaw = rec.yaw_unwrapped() if rec.rows else np.zeros(0)
    starts = []
    for i, t_ref in enumerate(b.resolved["starts_ref_s"], start=1):
        ts = rec.sim_time(t_ref)
        if ts is None:
            st = dict(command_ref_s=t_ref, command_s=None, reached=False, strides=None, acceleration_peak_m_s2=None)
        else:
            heading = float(np.interp(ts, rec.t, yaw))
            st = dict(metrics.start_reading(rec, ts, v_cmd, S, heading), command_ref_s=t_ref, heading_rad=heading)
            st["acceleration_peak_m_s2"] = (metrics.acceleration_peak(rec, ts - 0.25 * S,
                                                                      ts + tune.START_WITHIN_STRIDES * S, a_lp)
                                            if a_lp is not None else None)
        starts.append(st)
    stops = []
    readings = [s for s in (out.get("stops") or []) if s["name"] == "stop"]
    for i in range(1, START_STOP_CYCLES + 1):
        s = readings[i - 1] if i <= len(readings) else dict(reached=False)
        t_stop = s.get("command_s") if s.get("reached") else None
        peak = (metrics.acceleration_peak(rec, t_stop, t_stop + 2.0 * S, a_lp)
                if t_stop is not None and a_lp is not None else None)
        stops.append(dict({k: s.get(k) for k in ("name", "command_ref_s", "command_s", "reached",
                                                  "forward_stopped_strides", "within", "at_rest_strides",
                                                  "at_rest_s", "rest_speed_m_s")}, acceleration_peak_m_s2=peak))
    return dict(starts=starts, stops=stops, ok=start_stop_terms(starts, stops))


def start_stop_terms(starts: list[dict], stops: list[dict]) -> dict:
    """``START_STOP_RULE``'s terms from the starts' and stops' readings (start i: reached within 2 strides, its
    acceleration peak at most 1.5 m/s^2; stop i: decision 18's within, its acceleration peak)."""
    ok = {}
    for i in range(1, START_STOP_CYCLES + 1):
        st = starts[i - 1] if i <= len(starts) else {}
        sp = stops[i - 1] if i <= len(stops) else {}
        a, b = st.get("acceleration_peak_m_s2"), sp.get("acceleration_peak_m_s2")
        ok[f"start{i}_speed_up"] = bool(st.get("reached") and st.get("strides") is not None
                                        and st["strides"] <= tune.START_WITHIN_STRIDES)
        ok[f"start{i}_acceleration"] = a is not None and a <= tune.ACCEL_M_S2
        ok[f"stop{i}_stopped"] = bool(sp.get("within"))
        ok[f"stop{i}_acceleration"] = b is not None and b <= tune.ACCEL_M_S2
    return ok


def stride_speed(t: np.ndarray, com: np.ndarray, heading: float, stride_s: float) -> np.ndarray:
    """The CoM's speed along ``heading`` averaged over the last ``stride_s`` at every sample (metrics.start_reading's
    reading; NaN before one stride has passed)."""
    x = com[:, 0] * math.cos(heading) + com[:, 1] * math.sin(heading)
    back = np.interp(t - stride_s, t, x)
    v = (x - back) / stride_s
    v[t - stride_s < t[0] - 1e-12] = np.nan
    return v


def recovery(rec: metrics.Recorder, t_end: float, standing: bool, speed_cmd: float, stride_s: float,
             limit_s: float = RECOVERY_S, until_s: float | None = None) -> dict:
    """``RECOVERY_RULE`` on a recorded episode: when after the pulse's end ``t_end`` the body is back on its command
    for good, and whether that was within ``limit_s``; ``until_s`` is the episode's planned end, which the trace must
    reach (to the last feedback update, within two): a trace a fall cut short has not recovered."""
    if not rec.rows:
        return dict(recovered=False, after_s=None, standing=standing)
    t = rec.t
    com = rec.U[:, [metrics.C["com_x"], metrics.C["com_y"]]]
    if standing:
        v = np.hypot(rec.col("v_x"), rec.col("v_y"))
        thr = STAND_STOPPED_FRAC * speed_cmd
        flags = v < thr
        reading = dict(speed="the true CoM's horizontal speed", threshold_m_s=thr)
    else:
        heading = float(rec.col("yaw")[0])
        v = stride_speed(t, com, heading, stride_s)
        thr = WALK_BACK_FRAC * speed_cmd
        flags = v >= thr  # (NaN: not on command)
        reading = dict(speed="the CoM's speed along the starting heading over the last stride", threshold_m_s=thr,
                       heading_rad=heading)
    t_on = _settled_from(t, flags, t_end)
    after = None if t_on is None else float(t_on - t_end)
    full = until_s is None or float(t[-1]) >= until_s - 2 * control.FEEDBACK_EVERY * rec.dt
    k0 = int(np.searchsorted(t, t_end - 1e-9))
    tail = v[k0:]
    return dict(reading, standing=standing, after_s=after, recovered=bool(full and after is not None
                                                                          and after <= limit_s),
                limit_s=limit_s, trace_until_s=float(t[-1]), planned_until_s=until_s, on_command_frac_after_pulse=float(np.mean(flags[k0:]))
                if len(tail) else None, speed_min_after_pulse_m_s=float(np.nanmin(tail)) if np.isfinite(tail).any()
                else None, speed_max_after_pulse_m_s=float(np.nanmax(tail)) if np.isfinite(tail).any() else None,
                com_displacement_m=float(np.max(np.linalg.norm(com[k0:] - com[min(k0, len(t) - 1)], axis=1)))
                if len(tail) else None)


def sway(rec: metrics.Recorder, ta: float, tb: float, H: float, stride_s: float) -> dict:
    """``LOOK_RULE``'s sway over [ta, tb]: per whole commanded stride from ta, the CoM's lateral peak to peak (lateral
    to the CoM's displacement over the whole strides) / H."""
    t = rec.t
    k = int(math.floor((tb - ta) / stride_s + 1e-9))
    if k < 1 or not rec.rows:
        return dict(strides=0, mean_H=None, max_H=None, per_stride_H=[])
    c = rec.U[:, [metrics.C["com_x"], metrics.C["com_y"]]]
    ends = [ta + i * stride_s for i in range(k + 1)]
    ce = np.column_stack([np.interp(ends, t, c[:, 0]), np.interp(ends, t, c[:, 1])])
    d = ce[-1] - ce[0]
    if np.linalg.norm(d) < 1e-6:  # no displacement (in place): lateral to the heading at the window's start
        y0 = float(np.interp(ta, t, rec.yaw_unwrapped()))
        d = np.array([math.cos(y0), math.sin(y0)])
    u = d / np.linalg.norm(d)
    n = np.array([-u[1], u[0]])
    p2p = []
    for i in range(k):
        sel = (t > ends[i]) & (t < ends[i + 1])
        lat = np.concatenate([(c[sel] - ce[0]) @ n, (ce[i : i + 2] - ce[0]) @ n])
        p2p.append(float(lat.max() - lat.min()))
    return dict(strides=k, mean_H=float(np.mean(p2p)) / H, max_H=float(np.max(p2p)) / H,
                per_stride_H=[x / H for x in p2p], spec_H=SWAY_SPEC_H)


def look(rec: metrics.Recorder, ta: float, tb: float, H: float, stride_s: float, track: control.Track,
         joints: list[str]) -> dict:
    """``LOOK_RULE``: sway, head and tail over [ta, tb]."""
    out = dict(sway=sway(rec, ta, tb, H, stride_s))
    t = rec.t
    sel = (t >= ta - 1e-9) & (t <= tb + 1e-9)
    tref = rec.col("t_ref")[sel]
    nj = len(joints)
    q_ref = np.array([track.joint(float(x))[:nj] for x in tref]) if len(tref) else np.zeros((0, nj))
    for part, jn in LOOK_JOINTS.items():
        j = joints.index(jn)
        q, qr = rec.Q[sel, j], q_ref[:, j]
        if not len(q):
            out[part] = None
            continue
        amp, amp_ref = float(q.max() - q.min()) / 2, float(qr.max() - qr.min()) / 2
        lo, hi = LOOK_BANDS[part]
        out[part] = dict(joint=jn, amplitude_rad=amp, reference_amplitude_rad=amp_ref,
                         error_rms_rad=float(np.sqrt(np.mean((q - qr) ** 2))), error_max_rad=float(np.max(np.abs(q - qr))),
                         band_rad=[lo, hi], in_band=bool(lo <= amp <= hi))
    return out


def failed_episode(cond: Condition, seed: int, reason: str, relax=()) -> dict:
    """``FAILED_EPISODE_RULE``: a fall at the start, every term failed, with the reason."""
    terms = terms_of(cond)
    return dict(seed=int(seed), fell=True, fall_time_s=0.0, fall_cause="evaluation failed", fall_onset_s=None,
                fall_candidates=[], limits_before_onset=None, survival_s=0.0, planned_s=None, strides=0.0,
                succeeded=False, success={t: False for t in terms}, failed_terms=list(terms), measures={},
                relaxed=list(relax), error=str(reason), wall_s=0.0)


# ------------------------------------------------------------------------------------------------
# Counts, labels, verdicts
# ------------------------------------------------------------------------------------------------
def lcb(k: int, n: int, alpha: float = LCB_ALPHA) -> float | None:
    """``LCB_RULE``."""
    if n <= 0:
        return None
    if k <= 0:
        return 0.0
    from scipy.stats import beta

    return float(beta.ppf(alpha, k, n - k + 1))


def counts(episodes: list[dict], seeds=SEEDS) -> dict:
    n = len(episodes)
    surv = int(sum(not e["fell"] for e in episodes))
    succ = int(sum(e["succeeded"] for e in episodes))
    judged = n == RUNS and [e["seed"] for e in episodes] == list(seeds) == list(SEEDS)
    return dict(episodes=n, survived=surv, succeeded=succ, falls=n - surv, survival_lcb=lcb(surv, n),
                success_lcb=lcb(succ, n), judged=judged,
                pass_=bool(surv == RUNS and succ >= MIN_SUCCEED) if judged else None)


def success_label(episodes: list[dict]) -> dict:
    """``SUCCESS_LABEL_RULE`` over the episodes that survived but failed."""
    failed = [e for e in episodes if not e["fell"] and not e["succeeded"]]
    c = Counter(t for e in failed for t in e["failed_terms"])
    hw = [t for t in HARDWARE_TERMS if c.get(t, 0) > len(failed) / 2]
    return dict(failed_episodes=len(failed), terms={t: dict(count=n, cls=term_class(t)) for t, n in c.most_common()},
                hardware_terms=hw, label=("body-limited: " + ", ".join(hw)) if hw else "controller-limited")


def failure(row: str, episodes: list[dict], attribution: dict | None, pass_: bool | None) -> dict | None:
    """``FAILURE_LABEL_RULE``: what a record's failure is and its label (None if it passes or is not judged)."""
    if pass_ is None or pass_:
        return None
    falls = [e for e in episodes if e["fell"]]
    cands = Counter(c for e in falls for c in (e.get("fall_candidates") or ()))
    out = dict(falls=len(falls), fall_causes=dict(Counter(str(e["fall_cause"]) for e in falls)),
               fall_candidates=dict(cands), success=success_label(episodes))
    if falls:
        if attribution is not None:
            label = attribution["label"]
        elif row.split("/")[0] in ATTRIBUTED:
            label = "not attributed (falls; run eval --attribute)"
        else:
            label = f"not attributed (falls; attribution is applied to {', '.join(ATTRIBUTED)} rows)"
        return dict(out, kind="falls", label=label)
    return dict(out, kind="success", label=out["success"]["label"])


def verdict(records: list) -> dict:
    """``VERDICT_RULE`` on the records of the verdict's rows (R2/42 and R2/43; later the R3 seeds)."""
    if any(r is not None and r.get("status") == "not run" for r in records):
        return dict(verdict="not run", rule=VERDICT_RULE)
    if any(r is None or not r.get("counts", {}).get("judged") for r in records):
        return dict(verdict="incomplete", rule=VERDICT_RULE, missing=[i for i, r in enumerate(records) if r is None])
    passes = [bool(r["counts"]["pass_"]) for r in records]
    labels = [None if p else (r.get("failure") or {}).get("label") for r, p in zip(records, passes, strict=True)]
    if all(passes):
        v = "shown"
    elif any(passes):
        v = "provisional"
    else:
        if any(lab is None or lab.startswith("not attributed") for lab in labels):
            v = "not attributed"
        elif all(lab.startswith("body-limited") for lab in labels):
            v = "body-limited"
        else:
            v = "controller-limited"
    return dict(verdict=v, rule=VERDICT_RULE, rows=[r["row"] for r in records], passes=passes, labels=labels)


# ------------------------------------------------------------------------------------------------
# Records and keys
# ------------------------------------------------------------------------------------------------
def record_path(out: Path, body: str, row: str, cond: Condition) -> Path:
    """``<out>/<body>/eval/<row>/<behavior>/<condition>.json`` (plan section 7; a row "R2/42" is two directories)."""
    return Path(out) / body / "eval" / row / cond.behavior / f"{cond.name}.json"


def _jsonable(x):
    def default(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"{type(o)} is not JSON serialisable")

    return json.loads(json.dumps(x, default=default))


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(doc), indent=1))
    os.replace(tmp, path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def panel_sources(files: dict | None = None) -> dict:
    """``PANEL_SOURCES_RULE``."""
    files = tune.package_files() if files is None else files
    mods = tune.tuner_modules(files, root="panels.py")
    return dict(sha256=plant.sources_sha256({m: files[m] for m in mods}), rule=PANEL_SOURCES_RULE, modules=mods)


def _portable(x):
    """``x`` with every absolute path under the study directory made relative to it (a key does not depend on where
    the checkout is)."""
    if isinstance(x, dict):
        return {k: _portable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_portable(v) for v in x]
    if isinstance(x, str) and x.startswith(str(STUDY) + os.sep):
        return Path(x).relative_to(STUDY).as_posix()
    return x


def record_key(body: str, row: Row, cond: Condition, resolved: dict, seeds, inputs: dict, sources: str) -> dict:
    """``RESUME_RULE``'s key of a record."""
    files = files_of(cond)
    return _jsonable(dict(schema=SCHEMA, body=body, row=_portable(row.describe()), condition=cond.describe(),
                          plan=[list(x) for x in resolved["plan"]], seconds=resolved["seconds"], seeds=list(seeds),
                          inputs=dict(files={f: inputs["files"][f] for f in files}, records=inputs["records"]),
                          sources=sources))


def condition_record(body: str, row: Row, cond: Condition, episodes: list[dict], key: dict, seeds, prov: dict,
                     inputs: dict, wall_s: float, attribution: dict | None = None, started_at: str | None = None) -> dict:
    """A condition's record (module docstring)."""
    c = counts(episodes, seeds)
    return dict(schema=SCHEMA, body=body, row=row.name, behavior=cond.behavior, condition=cond.name, key_name=cond.key,
                gated=cond.gated, status="done", controller=row.describe(), definition=cond.describe(),
                plan=key["plan"], seconds=key["seconds"], seeds=list(seeds), counts=c,
                failure=failure(row.name, episodes, attribution, c["pass_"]), attribution=attribution,
                terms=list(terms_of(cond)), term_classes={t: term_class(t) for t in terms_of(cond)},
                episodes=episodes, rules=rules_of(cond), key=key, provenance=prov,
                inputs=dict(dir=inputs.get("dir"), rule=tune.INPUTS_RULE, files=key["inputs"]["files"],
                            records=key["inputs"]["records"]), wall_s=wall_s,
                episode_wall_s=float(sum(e.get("wall_s") or 0.0 for e in episodes)),
                simulated_s=float(sum(e.get("survival_s") or 0.0 for e in episodes)), started_at=started_at,
                ended_at=tune._now())


def not_run_record(body: str, row: Row, cond: Condition, key: dict, settle: dict | None, prov: dict) -> dict:
    """``NOT_RUN_RULE``'s record."""
    return dict(schema=SCHEMA, body=body, row=row.name, behavior=cond.behavior, condition=cond.name, key_name=cond.key,
                gated=cond.gated, status="not run", reason=NOT_RUN_RULE, settle=settle, controller=row.describe(),
                definition=cond.describe(), counts=dict(episodes=0, judged=False, pass_=None), failure=None,
                episodes=[], rules=rules_of(cond), key=key, provenance=prov, ended_at=tune._now())


def _attribution_needed(rec: dict, attribute: bool) -> bool:
    """With --attribute, an R2 record with falls and no attribution yet (a smoke panel of fewer seeds is attributed
    over its own seeds, for the record only: it is not judged)."""
    return bool(attribute and rec.get("status") == "done" and rec["row"].split("/")[0] in ATTRIBUTED
                and rec.get("attribution") is None and rec["counts"]["falls"] > 0)


def attribution_plan(rec: dict) -> tuple[list, list]:
    """The candidates of a record's falls (``control.HW_LIMITS`` order) and the relaxations that lift them."""
    cands = [c for c in control.HW_LIMITS if any(c in (e.get("fall_candidates") or ()) for e in rec["episodes"]
                                                  if e["fell"])]
    lift = [r for r in control.RELAXATIONS if any(control.RELAXES[c] == r for c in cands)]
    return cands, lift


def attribution_record(rec: dict, cands: list, lifted: dict, wall_s: float) -> dict:
    """``PANEL_ATTRIBUTION_RULE``: gates.attribute on the record's episodes and the lifted runs (no null)."""
    base = [dict(fell=e["fell"], strides=e["strides"]) for e in rec["episodes"]]
    rows = {r: [dict(fell=x["fell"], strides=x["strides"]) for x in v] for r, v in lifted.items()}
    a = gates.attribute(base, rows, None, cands)
    return dict(a, panel_rule=PANEL_ATTRIBUTION_RULE, episodes=int(sum(len(v) for v in lifted.values())),
                runs={r: [dict(seed=x["seed"], fell=x["fell"], strides=x["strides"], fall_time_s=x["fall_time_s"],
                               fall_cause=x["fall_cause"], succeeded=x["succeeded"]) for x in v]
                      for r, v in lifted.items()}, wall_s=wall_s)


# ------------------------------------------------------------------------------------------------
# Workers (tune.Pool with the panels' evaluator)
# ------------------------------------------------------------------------------------------------
class PanelWorker:
    """A worker's evaluator for ``tune.Pool``: ``evaluate((row, condition, relax), seed)`` runs one episode."""

    def __init__(self, body: str, run_dir: str, mode: str = "panels", rate: float = RATE,
                 expected_inputs: dict | None = None, rows=()):
        self.body, self.run_dir = body, Path(run_dir)
        self.inputs, docs = tune.read_inputs(self.run_dir, panel_files())
        if expected_inputs is not None:
            diff = tune.input_differences(expected_inputs, self.inputs)
            if diff:
                raise tune.InputError(f"{self.run_dir}: the inputs differ from the ones expected: " + "; ".join(diff))
        self.ctx = Context(body, self.run_dir, docs)
        self.rows = {s["name"]: row_from_spec(s) for s in rows}

    def describe(self) -> dict:
        return dict(body=self.body, rows=sorted(self.rows), panel_sources=panel_sources()["sha256"])

    def evaluate(self, job, seed) -> dict:
        row, cond, relax = job
        try:
            return self.rows[row].run(self.ctx, cond, int(seed), tuple(relax))
        except Exception:
            return failed_episode(cond, int(seed), traceback.format_exc(), relax)


def _worker(body, run_dir, mode, rate, expected_inputs=None, rows=()):
    return PanelWorker(body, run_dir, mode, rate, expected_inputs, rows)


def _lost(job, reason: str) -> dict:
    (row, cond, relax), seed = job
    return failed_episode(cond, seed, f"worker lost: {reason}", relax)


# ------------------------------------------------------------------------------------------------
# Running a body
# ------------------------------------------------------------------------------------------------
def stands(run_dir: Path, floor: str) -> tuple[bool, dict | None]:
    """Whether the body's zero-command stance settles on ``floor`` (plant.json's record; hard floor always)."""
    if floor == "hard":
        return True, None
    doc = _read(Path(run_dir) / "plant.json") or {}
    f = (doc.get("floors") or {}).get(floor)
    if f is None:
        return True, None
    return bool(f.get("stands", True)), dict(stand_status=f.get("stand_status"), settle=f.get("settle"))


def evaluate(body: str, out: Path, rows=ROWS, behaviors=None, names=None, workers: int = 1, attribute: bool = False,
             episodes: int = RUNS, inputs: Path | None = None, timeout_s: float = tune.EVAL_TIMEOUT_S,
             log=print) -> dict:
    """The panels of ``body`` (module docstring) into ``<out>/<body>/eval/``; the references are read from
    ``<inputs>/<body>/`` (default: out).  Returns the session's summary."""
    if body not in BODIES:
        raise KeyError(body)
    t_start = time.perf_counter()
    run_dir = Path(inputs or out) / body
    seeds = SEEDS[: int(episodes)]
    conds = select(conditions(), behaviors, names)
    inp, docs = tune.read_inputs(run_dir, panel_files())
    planner = Planner(run_dir, docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
    row_objs = tracker_rows(body, run_dir, rows)
    prov, src = plant.provenance(), panel_sources()
    pool = None
    session = dict(started_at=tune._now(), rows=list(rows), behaviors=None if behaviors is None else list(behaviors),
                   conditions=None if names is None else list(names), seeds=list(seeds), workers=int(workers),
                   attribute=bool(attribute), episodes_run=0, attribution_episodes_run=0, skipped=[], not_run=[],
                   incidents=[])

    def get_pool():
        nonlocal pool
        if pool is None:
            pool = tune.Pool(body, run_dir, workers, mode="panels", rate=RATE, timeout_s=timeout_s,
                             factory=functools.partial(_worker, rows=[r.spec() for r in row_objs.values()]))
        return pool

    try:
        for cond in conds:
            res = planner.resolve(cond)
            todo, ok_floor = [], stands(run_dir, cond.floor)
            for row in row_objs.values():
                if not row.runs(cond):
                    continue
                path = record_path(out, body, row.name, cond)
                key = record_key(body, row, cond, res, seeds, inp, src["sha256"])
                old = _read(path)
                if not ok_floor[0]:
                    if not (old and old.get("key") == key and old.get("status") == "not run"):
                        _write_json(path, not_run_record(body, row, cond, key, ok_floor[1], prov))
                    session["not_run"].append(f"{row.name} {cond.key}")
                    continue
                if old and old.get("key") == key and old.get("status") == "done" and len(old["episodes"]) == len(seeds):
                    session["skipped"].append(f"{row.name} {cond.key}")
                    todo.append((row, path, key, old))
                    continue
                todo.append((row, path, key, None))
            jobs = [((row.name, cond, ()), s) for row, _, _, old in todo if old is None for s in seeds]
            if jobs:
                t0 = time.perf_counter()
                results = get_pool().map(jobs, failed=_lost)
                wall = time.perf_counter() - t0
                session["episodes_run"] += len(jobs)
                i = 0
                n_new = sum(1 for _, _, _, old in todo if old is None)
                for k, (row, path, key, old) in enumerate(todo):
                    if old is not None:
                        continue
                    eps = results[i : i + len(seeds)]
                    i += len(seeds)
                    rec = condition_record(body, row, cond, eps, key, seeds, prov, inp, wall / max(1, n_new),
                                           started_at=session["started_at"])
                    _write_json(path, rec)
                    todo[k] = (row, path, key, rec)
                    c = rec["counts"]
                    log(f"{body} {row.name} {cond.key}: {c['survived']}/{c['episodes']} survive, {c['succeeded']}"
                        f"/{c['episodes']} succeed" + ("" if c["pass_"] is None else
                                                       f" ({'pass' if c['pass_'] else 'FAIL'})")
                        + ("" if cond.gated else " [reported]"))
            # decision 20's attribution of the R2 rows' falls (PANEL_ATTRIBUTION_RULE), every row's lifts in one batch
            plans = [(row, path, rec, *attribution_plan(rec)) for row, path, key, rec in todo
                     if rec is not None and _attribution_needed(rec, attribute)]
            ajobs = [((row.name, cond, (r,)), s) for row, _, _, _, lift in plans for r in lift for s in seeds]
            t0 = time.perf_counter()
            res_a = get_pool().map(ajobs, failed=_lost) if ajobs else []
            wall_a, i = time.perf_counter() - t0, 0
            for row, path, rec, cands, lift in plans:
                lifted = {}
                for r in lift:
                    lifted[r] = res_a[i : i + len(seeds)]
                    i += len(seeds)
                n = len(lift) * len(seeds)
                rec["attribution"] = attribution_record(rec, cands, lifted, wall_a * n / max(1, len(ajobs)))
                rec["failure"] = failure(row.name, rec["episodes"], rec["attribution"], rec["counts"]["pass_"])
                _write_json(path, rec)
                session["attribution_episodes_run"] += n
                log(f"{body} {row.name} {cond.key}: attribution {rec['attribution']['label']} ({n} episodes; "
                    f"candidates {', '.join(cands) or 'none'})")
        if pool is not None:
            session["incidents"] = list(pool.incidents)
    finally:
        if pool is not None:
            pool.close()
    session.update(ended_at=tune._now(), wall_s=time.perf_counter() - t_start, provenance=prov, panel_sources=src)
    return summarize(body, out, session=session, inputs=inputs)


# ------------------------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------------------------
def _cell(rec: dict | None) -> dict | None:
    if rec is None:
        return None
    c = rec.get("counts") or {}
    f = rec.get("failure") or {}
    return dict(status=rec.get("status"), episodes=c.get("episodes"), survived=c.get("survived"),
                succeeded=c.get("succeeded"), survival_lcb=c.get("survival_lcb"), success_lcb=c.get("success_lcb"),
                judged=c.get("judged"), pass_=c.get("pass_"), label=f.get("label"),
                attribution_episodes=(rec.get("attribution") or {}).get("episodes"), wall_s=rec.get("wall_s"))


def summarize(body: str, out: Path, rows=None, session: dict | None = None, inputs: Path | None = None) -> dict:
    """``<out>/<body>/eval/summary.json``: per behaviour and condition the rows' cells and the verdict, the episode
    counts, G7, and the sessions (each ``evaluate`` call appends its own)."""
    edir = Path(out) / body / "eval"
    path = edir / "summary.json"
    old = _read(path) or {}
    run_dir = Path(inputs or out) / body
    rows = list(rows or ROWS)
    found = {"/".join(q.relative_to(edir).parts[:-2]) for q in edir.glob("**/*.json") if len(q.relative_to(edir).parts)
             >= 3}
    rows += sorted(r for r in found if r and r not in rows)
    grid, n_eps, n_attr = {}, Counter(), Counter()
    for cond in conditions():
        recs = {r: _read(record_path(out, body, r, cond)) for r in rows}
        cells = {r: _cell(v) for r, v in recs.items() if v is not None}
        for r, v in recs.items():
            if v is not None:
                n_eps[r] += len(v.get("episodes") or ())
                n_attr[r] += int((v.get("attribution") or {}).get("episodes") or 0)
        grid.setdefault(cond.behavior, {})[cond.name] = dict(
            gated=cond.gated, floor=cond.floor, friction=cond.friction, rows=cells,
            verdict=verdict([recs.get(r) for r in VERDICT_ROWS]))
    gated = [v["verdict"]["verdict"] for b in grid.values() for v in b.values() if v["gated"]]
    try:
        g = g7(body, run_dir)
        _write_json(Path(out) / body / "eval" / "g7.json", g)
    except FileNotFoundError as exc:
        g = dict(error=str(exc))
    doc = dict(schema=SCHEMA, body=body, rules=RULES, verdict_rows=list(VERDICT_ROWS), grid=grid,
               gated_verdicts=dict(Counter(gated)), episodes=dict(n_eps), attribution_episodes=dict(n_attr), G7=g,
               sessions=list(old.get("sessions") or []) + ([session] if session else []), written_at=tune._now())
    _write_json(path, doc)
    return doc


def summary_text(doc: dict) -> str:
    """The summary grid as text (one line per condition: its verdict and each row's counts and label)."""
    v = ", ".join(f"{k} {n}" for k, n in sorted(doc["gated_verdicts"].items()))
    lines = [f"{doc['body']} panels, gated conditions: {v or 'none'}; episodes "
             + ", ".join(f"{r} {n}" for r, n in doc["episodes"].items())
             + (f"; attribution episodes {sum(doc['attribution_episodes'].values())}"
                if any(doc["attribution_episodes"].values()) else "")]
    for beh, conds in doc["grid"].items():
        for name, c in conds.items():
            cells = []
            for r, x in c["rows"].items():
                if not x:
                    continue
                if x.get("status") != "done":
                    cells.append(f"{r} {x.get('status')}")
                    continue
                verdict_ = "" if x["pass_"] is None else (" pass" if x["pass_"] else " FAIL")
                cells.append(f"{r} {x['survived']}/{x['episodes']} survive, {x['succeeded']} succeed{verdict_}"
                             + (f" ({x['label']})" if x.get("label") else ""))
            lines.append(f"  {beh}/{name}{'' if c['gated'] else ' [reported]'}: {c['verdict']['verdict']}"
                         + (f"; {'; '.join(cells)}" if cells else ""))
    g = doc.get("G7") or {}
    if "measured" in g:
        lines.append(f"  G7 {'pass' if g['pass_'] else 'FAIL'}: sway {g['measured']['sway_H']:.4f} H (Stage 0 "
                     f"{g['stage0']['sway_H']:.4f}, plan {g['plan']['sway_H']}), cadence "
                     f"{g['measured']['cadence_steps_s']:.4f} steps/s (Stage 0 {g['stage0']['cadence_steps_s']:.4f}, "
                     f"plan {g['plan']['cadence_steps_s']})")
    elif g:
        lines.append(f"  G7 not run: {g.get('error')}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------------------------------------
def measure_costs(body: str, run_dir: Path, floors, seconds: float = COST_SAMPLE_S, log=print) -> dict:
    """``COST_RULE``: wall seconds per simulated second of R2/42 on each floor (this process, now)."""
    inp, docs = tune.read_inputs(run_dir, panel_files())
    ctx = Context(body, run_dir, docs)
    row = tracker_rows(body, run_dir, ("R2/42",))["R2/42"]
    out = {}
    for floor in floors:
        t0 = time.perf_counter()
        cond = Condition("walk", f"cost_{floor}", False, "walk", floor=floor)
        p = ctx.plant(floor, None)
        build = time.perf_counter() - t0
        b = ctx.built(cond)
        tk = control.Tracker(p, b.track, "R2", gains=row.gains, rate=RATE)
        t1 = time.perf_counter()
        r = control.run(tk, seconds, start="stance", seed=1, noise_seed=2, joint_noise=tune.START_JOINT_NOISE_RAD,
                        root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S)
        wall = time.perf_counter() - t1
        out[floor] = dict(wall_per_sim_s=wall / max(r["seconds"], 1e-9), simulated_s=r["seconds"], wall_s=wall,
                          plant_build_s=build, load_average=list(os.getloadavg()))
        log(f"{body} {floor}: {out[floor]['wall_per_sim_s']:.3f} wall s per simulated s (plant built in {build:.1f} s, "
            f"load {os.getloadavg()[0]:.1f})")
    return dict(rule=COST_RULE, floors=out, cpu_count=os.cpu_count())


def dry_run(body: str, out: Path, rows=ROWS, behaviors=None, names=None, episodes: int = RUNS,
            inputs: Path | None = None, measure: bool = True, costs: dict | None = None, log=print) -> dict:
    """Every episode the panels would run (``evaluate``'s plan), which records are up to date, and the core-hours from
    the measured cost per simulated second (``COST_RULE``), with the attribution's bound."""
    run_dir = Path(inputs or out) / body
    seeds = SEEDS[: int(episodes)]
    conds = select(conditions(), behaviors, names)
    inp, docs = tune.read_inputs(run_dir, panel_files())
    planner = Planner(run_dir, docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
    row_objs = tracker_rows(body, run_dir, rows)
    src = panel_sources()["sha256"]
    lines, total = [], Counter()
    for cond in conds:
        res = planner.resolve(cond)
        st = stands(run_dir, cond.floor)[0]
        for row in row_objs.values():
            if not row.runs(cond):
                continue
            key = record_key(body, row, cond, res, seeds, inp, src)
            old = _read(record_path(out, body, row.name, cond))
            if not st:
                status = "not run (no settled stance)"
            elif old and old.get("key") == key and old.get("status") == "done":
                status = "up to date"
            else:
                status = "to run"
            lines.append(dict(row=row.name, condition=cond.key, gated=cond.gated, floor=cond.floor,
                              friction=cond.friction, seeds=list(seeds), seconds=res["seconds"], status=status,
                              plan=[list(x) for x in res["plan"]]))
    floors = sorted({x["floor"] for x in lines if not x["status"].startswith("not run")})
    if costs is None and measure:
        costs = measure_costs(body, run_dir, floors, log=log)
    cost = {f: ((costs or {}).get("floors") or {}).get(f, {}).get("wall_per_sim_s") for f in floors}
    for x in lines:
        n = 0 if x["status"].startswith("not run") else len(seeds)
        x["episodes"] = n
        x["simulated_s"] = n * x["seconds"]
        c = cost.get(x["floor"])
        x["core_h"] = None if c is None else x["simulated_s"] * c / 3600
        x["core_h_to_run"] = (x["core_h"] if x["status"] == "to run" else 0.0) if c is not None else None
        bound = len(control.RELAXATIONS) * x["simulated_s"] if x["row"].startswith("R2") else 0.0
        x["attribution_bound_core_h"] = None if c is None else bound * c / 3600
        total["episodes"] += n
        total["simulated_s"] += x["simulated_s"]
        for k in ("core_h", "core_h_to_run", "attribution_bound_core_h"):
            total[k] += x[k] or 0.0
        log(f"{x['row']:6s} {x['condition']:32s} {'gated' if x['gated'] else 'reported':8s} {x['floor']:18s} "
            f"{n:2d} x {x['seconds']:6.2f} s (seeds {seeds[0]}-{seeds[-1]})  "
            + (f"{x['core_h']:.3f} core-h" if x["core_h"] is not None else "cost not measured") + f"  [{x['status']}]")
    per_row = {r: int(sum(x["episodes"] for x in lines if x["row"] == r)) for r in row_objs}
    doc = dict(body=body, rule=COST_RULE, costs=costs, episodes=lines, per_row_episodes=per_row, totals=dict(total))
    log(f"{body}: {total['episodes']} episodes ({', '.join(f'{r} {n}' for r, n in per_row.items())}), "
        f"{total['simulated_s'] / 3600:.2f} simulated h, {total['core_h']:.2f} core-h "
        f"({total['core_h_to_run']:.2f} to run); attribution at most {total['attribution_bound_core_h']:.2f} core-h "
        "more (every relaxable limit lifted on every R2 condition)")
    return doc


# ------------------------------------------------------------------------------------------------
# G7: the measures on the references themselves
# ------------------------------------------------------------------------------------------------
def reference_recorder(p: plant.Plant, ref: Reference, seconds: float) -> tuple[metrics.Recorder, control.Track]:
    """A ``metrics.Recorder`` filled by playing the straight periodic reference ``ref`` kinematically on ``p``'s model
    for ``seconds`` (``G7_RULE``), and the track of the reference (its swing clock, its joints)."""
    sm = np.asarray(ref.stride_motion, float)
    if sm[0] != 0.0 or sm[1] != 0.0 or sm[2] != 0.0 or sm[4] != 0.0:
        raise ValueError(f"{ref.name}: G7 plays a straight walk (stride motion {sm.tolist()})")
    stride, n_fr, step = float(sm[3]), ref.n, ref.step
    m = p.m
    d = mujoco.MjData(m)
    dt = float(m.opt.timestep)
    n = int(round(seconds / dt))
    track = control.Track(p, control.omega0(p, ref.q_home)).add_periodic(ref, math.ceil(seconds / ref.period) + 2)
    rec = metrics.Recorder()
    rec.tk, rec.n, rec.dt = None, n, dt
    rec.weight = float(m.body_subtreemass[p.trunk]) * control.G
    rec.force_sq = np.zeros((n, m.nu))
    rec.sustained = np.array([dr.sustained for dr in p.drives], float)
    rec.actuators = [dr.actuator for dr in p.drives]
    sup = np.asarray(ref.support, bool)
    k_fr = (np.arange(n) * dt / step).astype(int) % n_fr
    nsup = np.maximum(1, sup[k_fr].sum(1, keepdims=True))
    rec.loads = sup[k_fr] * rec.weight / nsup
    boxes, caps = metrics._lowest_point_tables(m, p.feet)
    sole = [c.mean(0) for c in estimator.sole_corners(m, p.feet)]
    q_all, qd_all = np.asarray(ref.q, float), np.asarray(ref.qd, float)

    def frame(k):
        q = q_all[k % n_fr].copy()
        q[0] += (k // n_fr) * stride
        return q

    rows = []
    Q = []
    for k in range(0, n, control.FEEDBACK_EVERY):
        t = k * dt
        s = t / step
        k0 = int(math.floor(s))
        a = s - k0
        q = (1 - a) * frame(k0) + a * frame(k0 + 1)
        q[3:7] /= np.linalg.norm(q[3:7])
        qd = (1 - a) * qd_all[k0 % n_fr] + a * qd_all[(k0 + 1) % n_fr]
        d.qpos[:], d.qvel[:] = q, qd
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_subtreeVel(m, d)
        com, v = d.subtree_com[p.trunk], d.subtree_linvel[p.trunk]
        f0, f1 = d.xpos[p.feet[0]], d.xpos[p.feet[1]]
        low = metrics.foot_lowest(m, d, boxes, caps) - p.z_floor
        R = d.xmat[p.trunk].reshape(3, 3)
        roll, pitch = control._roll_pitch(R @ p._R_home.T)
        _, info = track.info(t)
        s0 = f0 + d.xmat[p.feet[0]].reshape(3, 3) @ sole[0]
        s1 = f1 + d.xmat[p.feet[1]].reshape(3, 3) @ sole[1]
        ld = rec.loads[k]
        rows.append((t, t, com[0], com[1], v[0], v[1], f0[0], f0[1], f1[0], f1[1], low[0], low[1], roll, pitch,
                     control._yaw(R), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, info["swing"], info["u"], s0[0], s0[1], s1[0], s1[1],
                     ld[0], ld[1], q[0], q[1]))
        Q.append(q[p.qadr])
    rec.U, rec.Q = np.array(rows, float), np.array(Q, float)
    rec.rows, rec.steps, rec.poses = len(rows), n, None
    return rec, track


def g7(body: str, run_dir: Path, seconds: float = G7_SECONDS) -> dict:
    """``G7_RULE`` on one body's nominal reference."""
    run_dir = Path(run_dir)
    doc = json.loads((run_dir / "refs.json").read_text())["references"]["nominal"]
    ref = Reference.load(run_dir / "refs" / "nominal.npz")
    p = plant.build(body, "hard", settle=False)
    H = float(p.meta["H_m"])
    T = ref.period
    rec, track = reference_recorder(p, ref, seconds + 3 * T)
    ta, tb = T, T + seconds
    w = metrics.window(rec, ta, tb, float(rec.col("yaw")[0]), H)
    lk = look(rec, ta, tb, H, T, track, p.joints)
    measured = dict(sway_H=lk["sway"]["mean_H"], sway_max_H=lk["sway"]["max_H"], cadence_steps_s=w["cadence_steps_s"],
                    duty=w["duty"], stride_H=w["stride_H"], toe_clearance_H=w["toe_clearance_m"] / H,
                    speed_m_s=w["speed_m_s"], head_amplitude_rad=lk["head"]["amplitude_rad"],
                    tail_amplitude_rad=lk["tail"]["amplitude_rad"])
    stage0 = dict(sway_H=2.0 * float(doc["info"]["sway_amp"]) / H, cadence_steps_s=float(doc["cadence_steps_s"]),
                  speed_m_s=float(doc["speed_m_s"]),
                  source="refs.json references.nominal: info.sway_amp (gait.generate: half the CoM path's lateral "
                         "peak to peak) and cadence_steps_s, the records round two's reports read (decision 9)")
    plan = dict(zip(("sway_H", "cadence_steps_s"), G7_PLAN[body], strict=True))
    diff = {k: measured[k] - stage0[k] for k in G7_TOL}
    within = {k: bool(abs(diff[k]) <= G7_TOL[k]) for k in G7_TOL}
    rounds = dict(sway_H=round(measured["sway_H"], 3) == plan["sway_H"],
                  cadence_steps_s=round(measured["cadence_steps_s"], 2) == plan["cadence_steps_s"])
    return dict(gate="G7", rule=G7_RULE, body=body, reference="nominal", seconds=seconds, window_s=[ta, tb], H_m=H,
                measured=measured, stage0=stage0, plan=plan, tolerance=G7_TOL, difference=diff, within=within,
                plan_rounding=rounds, pass_=bool(all(within.values()) and all(rounds.values())),
                reference_sha256=plant.sha256(run_dir / "refs" / "nominal.npz"))


__all__ = ["BEHAVIORS", "Built", "Condition", "Context", "PanelWorker", "Planner", "Policy", "ROWS", "RULES", "Row",
           "SEEDS", "TrackerRow", "ZeroResidual", "conditions", "counts", "dry_run", "evaluate", "g7", "judge",
           "lcb", "recovery", "select", "summarize", "sway", "terms_of", "tracker_rows", "verdict"]
