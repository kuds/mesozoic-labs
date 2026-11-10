"""Stage 1 behaviour panels and gate G7 (``STAGE1_PLAN.md`` section 3 step 8, section 4, section 7): the controller
rows R0, R1 and R2 on every condition of section 4, 20 episodes each, judged by section 4's success criteria, with the
pass rule, decision 20's fall attribution, the verdicts and a summary grid per body.

Run from hardware/trex_form_study (the fit's venv)::

    python -m designlab.stage1 eval --body 60-N-T-10 [--rows R0,R1,R2/42,R2/43] [--behaviors stand,walk,...]
                                    [--conditions walk/nominal,...] [--workers 4] [--attribute] [--dry-run]
                                    [--episodes 20] [--row-module MODULE] [--no-traces] --out stage1/runs/2026-11-08

Plan section 7 writes each step as ``--body <id> --seed <k>``; the panels run several controllers and seeds at once,
so ``eval`` names them as rows instead (``--rows R2/42`` is CMA-ES seed 42's R2).

Conditions (``conditions``, ``CONDITIONS_RULE``): stand (20 s; hard floor, robot_floor_carpet, low pile), walk
(nominal, brisk and Fr 0.05 at mu 0.9; nominal at mu 0.54; brisk and Fr 0.05 at mu 0.54 reported), turn (the 0.35 rad/s
walking turn and the 0.5 rad/s in-place turn, each direction; the 0.5 rad/s walking turn reported), start and stop
(stand, nominal, stand, twice), push (standing and walking, four directions) and carpet (the carpet walk on
robot_floor_carpet and low pile; medium and high pile reported; for R0-R2 every carpet walk is reported, decision 23,
``GATING_RULE``).  The plans and their lengths are ``Planner``'s.

Rows (``ROWS_RULE``): R0 and R1 run the hard-floor conditions only, R2/42 and R2/43 every condition.  The row
interface (``Row``) is what the later R3 and R4 rows implement: ``Row.episode`` runs one episode of a condition from
the condition's built plant, track, seeds and push (``Context.built``, ``episode_seeds``) and returns
``control.run``'s outcome and a filled ``metrics.Recorder``; ``Row.run`` judges it with ``judge`` (every row the same
success judgement).  R3 is a ``TrackerRow`` in mode R2 with a ``Policy`` that sets the tracker's residual
(``Policy``; a zero residual is R2 bit for bit, ``ZeroResidual``); R4 needs its own ``episode``.  Rows are added
from their own module (``register_rows``, ``eval --row-module``), without a change here: ``VERDICT_ROWS`` and
``ATTRIBUTED`` already name R3, and the records' keys do not depend on modules outside the panels' (``RESUME_RULE``).

Episodes (``EPISODE_RULE``): seeds 3042-3061, each from the settled stance with tune's start noise and R2's sensor
noise drawn from ``tune.derived_seed``, commanded 5 % above (``tune.COMMAND_RATE``).  Judgement: ``judge`` with
``STAND_RULE``, ``WALK_RULE``, ``TURN_RULE``, ``START_STOP_RULE``, ``PUSH_RULE`` and ``RECOVERY_RULE`` (decision 30),
``CARPET_RULE``, ``LIMITS_RULE``; reported metrics ``LOOK_RULE``.  Each condition and row gives one record
``<out>/<body>/eval/<row>/<behavior>/<condition>.json`` (``record_path``): per-episode metrics and failed terms, the
counts with their one-sided Clopper-Pearson lower bounds (``LCB_RULE``), the pass (``PASS_RULE``), the failure label
(``FAILURE_LABEL_RULE``: decision 20's attribution for falls, ``PANEL_ATTRIBUTION_RULE``; the term classes for
success failures, ``SUCCESS_LABEL_RULE``), the rules, provenance and wall time; each episode's 50 Hz trace goes to
``<out>/<body>/eval/traces/`` (``TRACE_RULE``, git-ignored).  ``summarize`` writes ``<out>/<body>/eval/summary.json``:
the grid with each controller's verdicts (``verdict``, ``VERDICT_RULE``) and G7 (``g7``, ``G7_RULE``).  Plan section
7's ``<body>/summary.json`` is the body's summary over every step (the report step); the panels' part of it is this
file.

Running (``evaluate``): ``tune.Pool`` with ``--workers`` processes, one job per episode, deterministic by episode
seed; a condition whose record exists with the same key (``RESUME_RULE``) is skipped, and an episode that failed to
run (``FAILED_EPISODE_RULE``) is run again; ``dry_run`` lists every episode and estimates the core-hours from a
measured cost per simulated second (``COST_RULE``).
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib
import inspect
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
from . import BODIES, STUDY, case_dir, control, env, estimator, gates, metrics, plant, tune
from .refs import Reference
from .transitions import STOP_RULE, STOP_SPEED_FRAC, Transition, _settled_from, stop_command_phase, stop_reading

SCHEMA = 2
SEEDS = tuple(range(3042, 3062))
RUNS = len(SEEDS)
MIN_SUCCEED = 17
LCB_ALPHA = 0.05
RATE = tune.COMMAND_RATE
ROWS = ("R0", "R1", "R2/42", "R2/43")
#: The controllers whose records carry verdicts, each from its two seeds' rows (``VERDICT_RULE``).
VERDICT_ROWS = {"R2": ("R2/42", "R2/43"), "R3": ("R3/42", "R3/43")}
#: The controller plan section 2 judges on every behaviour (R2 only if G6c fails, plan section 3 step 7).
JUDGED_CONTROLLER = "R3"
#: The controllers (a row's name before its seed) whose falls ``--attribute`` attributes (decision 20).
ATTRIBUTED = ("R2", "R3")
#: The tracker's own rows: the carpet walk is reported for them (decision 23, ``GATING_RULE``).
TRACKER_CONTROLLERS = ("R0", "R1", "R2")
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
#: Back to the command (``RECOVERY_RULE``, decision 30): standing, the CoM's speed below this share of the nominal
#: command; walking, the CoM's mean velocity over the stride ahead within these shares of the command, along the
#: starting heading (low, high) and across it.
STAND_STOPPED_FRAC = STOP_SPEED_FRAC
WALK_BACK_FRAC = metrics.START_SPEED_FRAC
WALK_OVER_FRAC = 1.1
WALK_ACROSS_FRAC = 0.1
#: Start and stop (``START_STOP_RULE``).
START_STOP_CYCLES = 2
START_STOP_STRIDES = 1
START_STOP_STAND_S = 2.0
#: Brisk and Fr 0.05 walks end in their steady gait; their window is the last 10 s (``WALK_RULE``).
CLOCK_SLACK = control.PHASE_RATE_LIMIT
WINDOW_MARGIN_STRIDES = 0.5
#: The fastest reference clock a row can run, against real time: the commanded rate, the tracker's phase-rate term
#: and a policy row's rate residual (env.RESIDUAL_RATE, R3 and R4); the stands' holds and the walking pushes' strides
#: are sized for it (``Planner``), so no row leaves them before its episode ends.
CLOCK_MAX = RATE * (1.0 + control.PHASE_RATE_LIMIT) * (1.0 + env.RESIDUAL_RATE)
#: Episode slots: the start, sensor-noise and push seeds of an episode are tune.derived_seed(seed, slot, purpose).
SLOTS = {"walk": 0, "carpet": 0, "walking_turn": 1, "in_place_turn": 2}
PURPOSES = {"start": 0, "noise": 1, "push": 2}
#: Measured with tune.measure (the fit's and G5's measures); the others are judged here.
TUNE_JUDGED = ("walk", "walking_turn", "in_place_turn", "carpet")
LIMIT_TERMS = ("drive_sustained", "drive_speed", "joint_range")
#: The success terms that are hardware limits (``SUCCESS_LABEL_RULE``); every other term is the controller's.
HARDWARE_TERMS = ("drive_sustained", "drive_speed", "joint_range", "slip_median", "slip_worst",
                  "no_leg_interference")
#: Walks judged at nominal speed only (plan section 4's Walk row, ``WALK_RULE``).
NOMINAL_ONLY_TERMS = ("cadence", "stride", "grf")
#: The fewest falls in which decision 20's one-sided sign test can reach p < alpha (every fall improved).
ATTRIBUTION_MIN_FALLS = next(n for n in range(1, 64) if 0.5**n < gates.ATTRIBUTION_ALPHA)
COST_SKIP_S = 6.0
COST_SAMPLE_S = 6.0
#: Traces (``TRACE_RULE``): every this many feedback updates (100 Hz) a row.
TRACE_EVERY = 2
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
    "decide nothing; whether a condition is gated depends on the row as well (GATING_RULE: decision 23 reports the "
    "carpet walk for R0-R2).  R0 and R1 run the conditions on the hard floor (any friction); R2 every condition.  "
    "Piles are plan section 2's evaluation piles (carpet.floor_variants: 4 mm probes; the 8 mm training pile is "
    "PPO's)")
GATING_RULE = (
    "a condition is gated for a row as plan section 4 gates it, except that STAGE1_PLAN.md decision 23 reports the "
    "carpet walk, on every floor, for R0, R1 and R2 (the tracker was never fitted on carpet: it is reported with its "
    "fall attribution) and leaves its verdict to the policy rows R3 and R4, for which robot_floor_carpet and low pile "
    "are gated; each record carries its row's gating (gated) beside the plan's (plan_gated)")
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
    "tune.FAILURE_RULE); an episode that raises, or whose worker is lost twice, failed to run and is not a result "
    "(FAILED_EPISODE_RULE)")
FAILED_EPISODE_RULE = (
    "an episode that raises, or whose worker process dies or does not answer within "
    f"{tune.EVAL_TIMEOUT_S:g} s on each of {tune.ATTEMPTS} attempts (tune.Pool), failed to run: it is recorded with "
    "the traceback or reason (error) as a fall at its start with every success term failed, but it is not a result: "
    "the counts give it apart (errors), a record that holds one is not judged (pass_ None, its verdict incomplete), "
    "the next eval of the same key runs those seeds again and merges them in (RESUME_RULE), and attribution leaves "
    "it out (PANEL_ATTRIBUTION_RULE)")
LIMITS_RULE = (
    "every success also needs every drive's motor-space RMS at most its sustained torque and every drive's peak "
    "speed at most speed_at(24 V) (control.run's worst ratios), and every joint keeping its required usable range "
    f"(walks, stands, pushes, starts and stops {WALK_RANGE_RAD:g} rad; turns decision 14's margin, "
    "tune.SUCCESS_RULE): for walks, turns and the carpet walk read as tune.SUCCESS_RULE reads them (the drive RMS over "
    "the judged window, peak speed and range over the episode); for stands, pushes and starts and stops over the "
    "whole episode")
STAND_RULE = (f"a stand succeeds if it survives {STAND_S:g} s and the trunk's tilt from home stays below "
              f"{STAND_TILT_DEG:g} deg at every feedback update (control.run's tilt_from_home_max_deg), with "
              "LIMITS_RULE; the reference holds the start transition's first frame (the home stance) for the stand's "
              f"seconds x {CLOCK_MAX:.4g} + 1 s, the fastest clock a row can run (CLOCK_MAX: the commanded rate, the "
              "phase-rate term and a policy's rate residual), so no row leaves it before the stand ends; a standing "
              "push holds the same way")
WALK_RULE = (
    "the nominal walk at mu 0.9 is tune's walk episode exactly (start, 7 nominal strides, stop; tune.build_spec), and "
    "at mu 0.54 the same; brisk and Fr 0.05 walk from the start through the speed changes (start, 1 nominal stride, "
    "nominal_to_brisk, [1 brisk stride, brisk_to_fr05,] K strides of the target gait) and end in it, K the fewest "
    "strides that keep the last 10 s in the target gait's steady part even with the clock "
    f"{100 * CLOCK_SLACK:g} % slow (plus {WINDOW_MARGIN_STRIDES:g} stride); judged by tune.measure and "
    "tune.SUCCESS['walk'] over the 10 s before the stop command (nominal) or the episode's last 10 s (brisk, Fr "
    "0.05), the speed against the target gait's speed (commanded 5 % above).  Cadence, stride and the peak GRF are "
    "judged at nominal only: plan section 4's Walk row groups them as 'at nominal, cadence 1.6-2.2 steps/s, stride "
    "0.5-0.65 H, GRF <= 1.6 BW', and BEHAVIOR_SPEC section 1's Impact row reads 'peak vertical ground reaction <= "
    "1.6 x body weight at nominal speed'; the brisk and Fr 0.05 walks drop those three terms (the GRF stays in their "
    "window's measures, reported), and their start and stop readings are not recorded (their start goes to the "
    "nominal speed, not the target gait's, and they end without a stop)")
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
    "command is where the start's speed ramp begins); each stop: decision 18 (transitions.STOP_RULE, "
    "transitions.stop_reading): the true CoM's speed along the trunk's heading at the stop command below "
    f"{100 * STOP_SPEED_FRAC:g} % of the commanded speed from some time to the end of the stop's own part of the "
    f"episode, within {tune.STOP_WITHIN_STRIDES:g} strides of the command.  A stop's part runs from its command (the "
    "feedback update at which the reference clock passes it, as control.run reads it) to the beginning of the next "
    "start transition (the end of the stand after it), the last stop's to the episode's end; it is read from the "
    "recorder's feedback updates (the CoM velocity control.run's stop reading takes at the same updates, the time to "
    "rest with control.rest_speed's threshold).  control.run's own stop readings give every update after a stop to "
    "the latest stop reached, so the first stop's would run through the second start and its walking and never "
    "read stopped; they are recorded beside (control_reading).  Acceleration (decision 10): the IMU point's forward "
    f"acceleration low-passed at {metrics.ACCEL_LP_HZ:g} Hz at most {tune.ACCEL_M_S2:g} m/s^2 over [start command - "
    "0.25 stride, + 2 strides] and [stop command, + 2 strides] (tune.measure's windows); with LIMITS_RULE")
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
    "STAGE1_PLAN.md decision 30: back to the command within "
    f"{RECOVERY_S:g} s means a feedback update at most {RECOVERY_S:g} s after the pulse's end (at or after it) from "
    f"which the body is on its command to the episode's end (at least {PUSH_HOLD_S:g} s more).  Standing, on command "
    f"at an update: the true CoM's horizontal speed below {100 * STAND_STOPPED_FRAC:g} % of the nominal walk's "
    "commanded speed (decision 18's 'stopped'), at every update to the end.  Walking, on command from an update: the "
    "CoM's mean velocity over the commanded stride that starts there (its displacement over that stride / the "
    f"stride's time) is {100 * WALK_BACK_FRAC:g}-{100 * WALK_OVER_FRAC:g} % of the commanded speed along the "
    f"episode's starting heading (the straight walk's) and at most {100 * WALK_ACROSS_FRAC:g} % of it across, for "
    "every stride that starts at an update from then until one stride before the episode's end.  The window starts "
    "at the pulse's end or later, so the reading after the pulse is not the walking before it.  An update at the "
    "pulse's end itself is 'never left the command'.  A trace a fall cut short has not recovered; the time from the "
    "pulse's end is recorded")
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
    "the success failures (episodes that survive but fail a term) are labelled from their failed terms: the hardware "
    f"terms ({', '.join(HARDWARE_TERMS)}: drive RMS, drive speed, joint range, slip at the pad, and the legs' "
    "clearance, legs.LegGap's interference, the leg-to-leg clearance that STAGE1_PLAN.md decision 20 puts in the "
    "body-limited list in place of the turn gap: on the narrow bodies the legs are 32-58 mm apart at the reference "
    "stance, so their spacing is the body's) against the controller's (speed, yaw rate, cadence, stride, duty, "
    "flight, GRF, roll, pitch and tilt, toe clearance and the carpet's toe terms, drift, start and stop, push "
    "recovery); body-limited by the hardware terms that failed in more than half of the failed episodes, else "
    "controller-limited; every failed term is listed with its class and count.  A fall's failed term is 'survived' "
    "alone: a fall fails every term (tune.SUCCESS_RULE), so its other terms say nothing about what bound, and falls "
    "do not count here.  The CoP at a pad edge with the ankle saturated and step reach are hardware limits with no "
    "success term of their own: they enter through falls (control.HW_LIMITS, decision 20)")
PANEL_ATTRIBUTION_RULE = (
    "with --attribute, a condition of an R2 or R3 row with falls is attributed by decision 20 (gates.ATTRIBUTION_RULE, "
    "gates.attribute): every relaxable hardware limit that is a candidate of at least one of its falls "
    "(control.CANDIDATE_RULE) is lifted in turn (control.run's relax) over the same 20 seeds, and the condition is "
    "body-limited by the limits whose lifting improves survival significantly (gates.sign_test, one-sided, p < "
    f"{gates.ATTRIBUTION_ALPHA:g}), else controller-limited; that costs (lifted limits x 20) episodes.  The sign test "
    f"can reach p < {gates.ATTRIBUTION_ALPHA:g} only with at least {ATTRIBUTION_MIN_FALLS} falls (every one of them "
    f"improved: 1/2^{ATTRIBUTION_MIN_FALLS} = {0.5 ** ATTRIBUTION_MIN_FALLS:.4g}; with "
    f"{ATTRIBUTION_MIN_FALLS - 1}, {0.5 ** (ATTRIBUTION_MIN_FALLS - 1):.4g}), so a condition with fewer falls is "
    "controller-limited by decision 20 and no limit is lifted for it (its label says so).  A candidate the plant "
    "cannot relax (control.RELAXES None: cop_at_pad_edge_with_ankle_saturated, the pad's size) cannot be tested; a "
    "controller-limited label names such candidates as untestable.  Episodes that failed to run "
    "(FAILED_EPISODE_RULE) give no candidates and are left out of the sign test's pairs; an attribution whose lifted "
    "runs had such failures is run again by the next --attribute.  The null (1 mm/s) and the together runs of G4 "
    "are diagnostics and are not run; R0 and R1 are not attributed")
FAILURE_LABEL_RULE = (
    "a failing record's failure holds both labels: the falls' (decision 20's attribution label, "
    "PANEL_ATTRIBUTION_RULE, or 'not attributed' until it runs, R2 and R3, or because it is not run, R0 and R1) and "
    "the success failures' (SUCCESS_LABEL_RULE).  Its label is the success failures' when nothing falls, the falls' "
    f"when the success failures alone would not fail it (at most {RUNS - MIN_SUCCEED} of them), and otherwise that "
    "of the kind with more failed episodes, the falls' on a tie (plan section 4: a hardware limit binds before most "
    "failures)")
VERDICT_RULE = (
    "per controller (VERDICT_ROWS: R2 from R2/42 and R2/43, R3 from R3/42 and R3/43) and condition, from its two "
    "seeds' records: shown if both pass, provisional if one does, else body-limited if every failing one is labelled "
    "body-limited (the limits or terms listed), not attributed if any failing one has unattributed falls, else "
    "controller-limited; incomplete while a record is missing, not judged or holding an episode that failed to run "
    "(FAILED_EPISODE_RULE), not run if the floor has no settled stance.  A controller's gated verdicts are those of "
    "the conditions gated for its rows (GATING_RULE: decision 23 reports the carpet walk for R2).  The judged "
    f"controller is {JUDGED_CONTROLLER} (plan section 2; R2 only if G6c fails, plan section 3 step 7): until its "
    "records exist the summary shows R2's verdicts, marked interim")
NOT_RUN_RULE = ("a condition on a floor where the body's zero-command stance does not settle (plant.Plant.settled False, "
                "plant.json floors.<floor>.stand_status: the 60 cm bodies on medium and high pile) has no settled "
                "stance to start from: it is recorded as not run, with the settle's status, and judged not at all")
RESUME_RULE = (
    "a condition's record is reused (not run again) if it holds every episode of its seeds, none of which failed to "
    "run (those seeds are run again and merged in, FAILED_EPISODE_RULE), and its key is this run's: the schema, the "
    "body, the row's description (controller, gains and their selection) and the sources it names of its own "
    "(Row.sources: a policy row's modules), the condition and its resolved plan and length, the seeds, the push "
    "bound, the sha256 of every reference and transition file it loads (tune.INPUTS_RULE) and of the round-two files "
    "the plant reads (the body's record, MJCF, ledger and clearance screen), what the panels take from env.py (the "
    "push bound, the rate residual's bound, the env step, the carpet walk's plan and friction, by value, and "
    "env.foot_tables by its source's sha256), the library versions (python, numpy, mujoco, scipy) and the panels' "
    "sources (PANEL_SOURCES_RULE); --attribute adds the attribution to a reused record that lacks it.  Every worker "
    "of a session checks that it reproduces the session's sources and this context (PanelWorker), so a worker "
    "started after an edit refuses to run")
PANEL_SOURCES_RULE = (
    "sha256 over the modules an episode and its judgement run: the designlab/stage1 modules panels.py imports, "
    "followed as tune.TUNER_SOURCES_RULE follows the tuner's (from panels.py and the package's __init__.py, "
    "transitively) except env.py and gates.py, and the designlab modules outside designlab/stage1 that these import "
    "when they are imported (module level), followed the same way through designlab's own module-level imports "
    "(statics, carpet, clearance, sizing, gait, ...; a missing model's rebuild through round2 and pipeline is a "
    "function-level import, and plant.verify checks the model against its record's xml_sha256 anyway), each set "
    "hashed as plant.SOURCES_RULE hashes files, the module lists recorded beside.  env.py (the PPO environments) and "
    "gates.py (the gates) are left out: the panels take from env.py the carpet walk's plan and friction, the push "
    "bound, the pile floors' foot tables, the rate residual's bound and the env step, which the key holds by value "
    "or by source (RESUME_RULE), and from gates.py only the attribution, whose record holds gates.py's sha256.  So "
    "a row added in its own module, or a change to env.py's training code, leaves the R0-R2 records' keys as they "
    "are")
COST_RULE = (
    "the dry run's cost per simulated second: per floor, R2/42 walking the nominal walk from the settled stance in "
    "this process with the panels' recorder, timed over its simulated seconds "
    f"{COST_SKIP_S:g}-{COST_SKIP_S + COST_SAMPLE_S:g} (steady walking; from the start when it falls before "
    f"{COST_SKIP_S + 1:g} s): CPU seconds (process time) and wall seconds (the load average beside) over simulated "
    "seconds; core-hours = the sum over episodes of their planned seconds x their floor's CPU cost / 3600 (R0 and R1 "
    "at R2's cost, an upper bound), wall hours with N workers about core-hours x (wall / CPU) / N; the attribution's "
    "bound lifts every relaxable limit on every condition of an attributed row (R2, R3)")
TRACE_RULE = (
    "each panel episode's trace (not the attribution's lifted runs) is written to "
    "<out>/<body>/eval/traces/<row>/<behavior>/<condition>/<seed>.npz (git-ignored, stage1/.gitignore): the "
    f"recorder's rows every {TRACE_EVERY} feedback updates (50 Hz; metrics.COLS: time and reference time, CoM and its "
    "velocity, feet, lowest points, roll, pitch and yaw, slip speeds, IMU acceleration, DCM error, the reference's "
    "swing, sole centres, loads, root) and the joint angles (plant order), the times in float64 and the rest in "
    "float32; the episode record holds its path and sha256")
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
RULES = dict(conditions=CONDITIONS_RULE, gating=GATING_RULE, rows=ROWS_RULE, episode=EPISODE_RULE,
             failed_episode=FAILED_EPISODE_RULE, limits=LIMITS_RULE, stand=STAND_RULE, walk=WALK_RULE, turn=TURN_RULE,
             start_stop=START_STOP_RULE, push=PUSH_RULE, recovery=RECOVERY_RULE, carpet=CARPET_RULE, look=LOOK_RULE,
             pass_=PASS_RULE, lcb=LCB_RULE, success_label=SUCCESS_LABEL_RULE, attribution=PANEL_ATTRIBUTION_RULE,
             decision_20=gates.ATTRIBUTION_RULE, candidates=control.CANDIDATE_RULE, failure_label=FAILURE_LABEL_RULE,
             verdict=VERDICT_RULE, not_run=NOT_RUN_RULE, resume=RESUME_RULE, sources=PANEL_SOURCES_RULE,
             trace=TRACE_RULE, success=tune.SUCCESS_RULE, command_rate=control.COMMAND_RATE_RULE, stop=STOP_RULE,
             failures=tune.FAILURE_RULE)


#: The rules every record carries; ``rules_of`` adds its behaviour's.
COMMON_RULES = ("conditions", "gating", "rows", "episode", "failed_episode", "limits", "pass_", "lcb",
                "success_label", "attribution", "decision_20", "candidates", "failure_label", "verdict", "not_run",
                "resume", "sources", "trace", "command_rate", "failures")
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


def controller_of(row: str) -> str:
    """A row's controller: its name before the seed ("R2/42" -> "R2")."""
    return row.split("/", 1)[0]


def gated_for(row: str, cond: Condition) -> bool:
    """Whether ``cond`` is gated for the row ``row`` (``GATING_RULE``: decision 23 reports the carpet walk for the
    tracker's rows)."""
    return bool(cond.gated and not (cond.behavior == "carpet" and controller_of(row) in TRACKER_CONTROLLERS))


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
            plan = [("hold", seconds * CLOCK_MAX + 1.0), plan[1]]  # held at the fastest clock (STAND_RULE)
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
            plan = [plan[0], ("ref", "nominal", max(1, math.ceil((CLOCK_MAX * seconds - t0) / P) + 1))]
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
    * ``sources()``: the sha256 of the modules outside the panels' sources (``PANEL_SOURCES_RULE``) that the row's
      episodes run, by path (a policy row: its policy's module and env.py; ``TrackerRow`` names them for a
      ``Policy``); they enter the resume key with ``describe()``;
    * ``run(ctx, cond, seed, relax, trace_root)`` -> the episode record: ``judge`` on the episode (the same for every
      row), and the episode's trace under ``trace_root`` (``TRACE_RULE``) when it is given.

    A row's module makes its rows known with ``register_rows`` when it is imported (``eval --row-module``); whether
    a condition is gated for the row is ``GATING_RULE``'s (``gated``)."""

    name: str = ""
    hard_only: bool = False

    def runs(self, cond: Condition) -> bool:
        return not self.hard_only or cond.floor == "hard"

    def gated(self, cond: Condition) -> bool:
        return gated_for(self.name, cond)

    def describe(self) -> dict:
        raise NotImplementedError

    def sources(self) -> dict:
        return {}

    def spec(self) -> dict:
        raise NotImplementedError

    def episode(self, ctx: "Context", cond: Condition, seed: int, relax=()):
        raise NotImplementedError

    def run(self, ctx: "Context", cond: Condition, seed: int, relax=(), trace_root: Path | None = None) -> dict:
        t0 = time.perf_counter()
        b, out, rec = self.episode(ctx, cond, int(seed), tuple(relax))
        e = judge(b, out, rec, int(seed), ctx.lg if b.legs else None)
        if trace_root is not None:
            e["trace"] = write_trace(trace_root, trace_path(self.name, cond, int(seed)), rec)
        e.update(relaxed=list(relax), wall_s=time.perf_counter() - t0)
        return e


def module_sha256(obj) -> dict:
    """{path: sha256} of the source file that defines ``obj`` (a path under the study directory relative to it)."""
    path = Path(inspect.getsourcefile(obj)).resolve()
    return {_portable(str(path)): plant.sha256(path)}


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

    def sources(self) -> dict:
        """A policy's module and env.py (whose scaling and filter a policy applies, ``Policy``); none without one."""
        if self.policy is None:
            return {}
        return {**module_sha256(env), **module_sha256(type(self.policy))}

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


#: Rows added from other modules (``register_rows``): a controller ("R3") -> factory(body, run_dir, name) -> Row.
ROW_FACTORIES: dict = {}


def register_rows(controller: str, factory) -> None:
    """Make the rows ``<controller>`` and ``<controller>/<seed>`` known to ``tracker_rows`` (and ``eval --rows``):
    ``factory(body, run_dir, name)`` builds one.  A row's module calls this when it is imported, so the R3 and R4
    rows need no change here (``eval --row-module`` imports the module)."""
    if controller in ("R0", "R1", "R2"):
        raise ValueError(f"{controller} is the panels' own row")
    ROW_FACTORIES[controller] = factory


def tracker_rows(body: str, run_dir: Path, names=ROWS) -> dict[str, Row]:
    """The rows R0, R1 and R2/s of ``ROWS_RULE`` on ``run_dir``'s tune.json, and any row of a registered controller
    (``register_rows``)."""
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
        elif controller_of(name) in ROW_FACTORIES:
            out[name] = ROW_FACTORIES[controller_of(name)](body, Path(run_dir), name)
        else:
            raise ValueError(f"row {name!r}: choose from {', '.join(ROWS)} or a registered controller's rows "
                             f"({', '.join(sorted(ROW_FACTORIES)) or 'none registered'}; register_rows)")
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
        return t if cond.gait == "nominal" else tuple(x for x in t if x not in NOMINAL_ONLY_TERMS)
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
        if j == "walk" and cond.gait != "nominal":  # its start goes to the nominal speed, and it has no stop (WALK_RULE)
            measures.update(start=None, stop_reading=None, stop=None)
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
    # a fall fails every term (tune.SUCCESS_RULE); its failed term is the fall alone (SUCCESS_LABEL_RULE)
    e.update(success=success, succeeded=bool(all(success.values())),
             failed_terms=["survived"] if fell else [t for t, v in success.items() if not v], measures=measures)
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
    (decision 18 over its own part of the episode, ``stop_part_reading``), their acceleration peaks, and the terms."""
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
    # each stop's part ends where the next start transition begins (the stand after the stop ends there)
    next_start = [seg.t0 for seg in b.track.segments if seg.kind == "finite" and seg.name == "start"][1:]
    for i in range(1, START_STOP_CYCLES + 1):
        s = readings[i - 1] if i <= len(readings) else dict(name="stop", reached=False)
        r = stop_part_reading(rec, s, next_start[i - 1] if i <= len(next_start) else None)
        t_stop = r.get("command_s") if r.get("reached") else None
        peak = (metrics.acceleration_peak(rec, t_stop, t_stop + 2.0 * S, a_lp)
                if t_stop is not None and a_lp is not None else None)
        stops.append(dict(r, acceleration_peak_m_s2=peak))
    return dict(starts=starts, stops=stops, ok=start_stop_terms(starts, stops))


STOP_READING_KEYS = ("forward_stopped_strides", "within", "at_rest_strides", "at_rest_s")


def stop_part_reading(rec: metrics.Recorder, s: dict, end_ref: float | None) -> dict:
    """Decision 18 on one stop of a start-and-stop episode over its own part (``START_STOP_RULE``): ``s`` is
    control.run's reading of the stop (its command's update, the commanded speed and stride, the rest speed), the part
    runs from that update to the last update before the reference clock reaches ``end_ref`` (the next start
    transition's beginning; None: to the trace's end).  The recorder's updates hold the CoM velocity control.run's
    stop reading takes at the same updates; their times are taken as control.run takes them (the end of the update's
    physics step), so on the episode's last stop this is control.run's reading.  control.run's own reading is kept
    beside (``control_reading``)."""
    head = {k: s.get(k) for k in ("name", "command_ref_s", "command_s", "command_speed_m_s", "rest_speed_m_s")}
    own = {k: s.get(k) for k in STOP_READING_KEYS + ("samples",)}
    if not s.get("reached") or not rec.rows:
        return dict(head, reached=False, control_reading=own, **dict.fromkeys(STOP_READING_KEYS))
    t = rec.t + rec.dt  # control.run's sample times: d.time after the update's physics step
    k0 = int(np.argmin(np.abs(t - float(s["command_s"]))))
    t_to = None if end_ref is None else rec.sim_time(end_ref)
    k1 = len(t) if t_to is None else max(k0 + 1, int(np.searchsorted(rec.t, t_to - 1e-12)))
    yaw = float(rec.col("yaw")[k0])  # the trunk's heading at the command (control.run's forward direction)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    v = rec.U[k0:k1][:, [metrics.C["v_x"], metrics.C["v_y"]]]
    r = stop_reading(t[k0:k1], v @ fwd, float(s["command_speed_m_s"]), float(s["command_s"]), float(s["stride_s"]),
                     rest=np.hypot(v[:, 0], v[:, 1]) < float(s["rest_speed_m_s"]))
    return dict(head, reached=True, part_from_s=float(t[k0]), part_to_s=float(t[k1 - 1]), samples=k1 - k0,
                part_ends="the next start transition" if t_to is not None else "the episode's end",
                command_update_offset_s=float(t[k0] - float(s["command_s"])), stride_s=float(s["stride_s"]),
                control_reading=own, **{k: r[k] for k in STOP_READING_KEYS})


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


def stride_velocity(t: np.ndarray, com: np.ndarray, heading: float, stride_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The CoM's mean velocity over the stride of ``stride_s`` that starts at each sample (its displacement over the
    stride / ``stride_s``), along ``heading`` and across it (to the left); NaN where that stride runs past the trace
    (``RECOVERY_RULE``)."""
    ahead = np.column_stack([np.interp(t + stride_s, t, com[:, 0]), np.interp(t + stride_s, t, com[:, 1])])
    v = (ahead - com) / stride_s
    u = np.array([math.cos(heading), math.sin(heading)])
    along, across = v @ u, v @ np.array([-u[1], u[0]])
    past = t + stride_s > t[-1] + 1e-9
    along[past] = np.nan
    across[past] = np.nan
    return along, across


def recovery(rec: metrics.Recorder, t_end: float, standing: bool, speed_cmd: float, stride_s: float,
             limit_s: float = RECOVERY_S, until_s: float | None = None) -> dict:
    """``RECOVERY_RULE`` (decision 30) on a recorded episode: from when after the pulse's end ``t_end`` the body is
    on its command to the end, and whether that was within ``limit_s``; ``until_s`` is the episode's planned end,
    which the trace must reach (to the last feedback update, within two): a trace a fall cut short has not
    recovered."""
    if not rec.rows:
        return dict(recovered=False, after_s=None, never_left_command=False, standing=standing)
    t = rec.t
    com = rec.U[:, [metrics.C["com_x"], metrics.C["com_y"]]]
    k0 = int(np.searchsorted(t, t_end - 1e-9))
    if standing:
        v = np.hypot(rec.col("v_x"), rec.col("v_y"))
        thr = STAND_STOPPED_FRAC * speed_cmd
        tt, flags = t, v < thr
        after = v[k0:]
        reading = dict(speed="the true CoM's horizontal speed at each feedback update", threshold_m_s=thr,
                       speed_max_after_pulse_m_s=float(np.max(after)) if len(after) else None)
    else:
        heading = float(rec.col("yaw")[0])
        along, across = stride_velocity(t, com, heading, stride_s)
        lo, hi, side = WALK_BACK_FRAC * speed_cmd, WALK_OVER_FRAC * speed_cmd, WALK_ACROSS_FRAC * speed_cmd
        ok = np.isfinite(along)  # the strides that end within the trace
        tt = t[ok]
        flags = (along[ok] >= lo) & (along[ok] <= hi) & (np.abs(across[ok]) <= side)
        a_, c_ = along[ok][tt >= t_end - 1e-9], across[ok][tt >= t_end - 1e-9]
        reading = dict(speed="the CoM's mean velocity over the commanded stride from each feedback update, along the "
                             "starting heading and across it", heading_rad=heading, stride_s=stride_s,
                       along_m_s=[lo, hi], across_max_m_s=side,
                       along_min_after_pulse_m_s=float(np.min(a_)) if len(a_) else None,
                       along_max_after_pulse_m_s=float(np.max(a_)) if len(a_) else None,
                       across_max_after_pulse_m_s=float(np.max(np.abs(c_))) if len(c_) else None)
    t_on = _settled_from(tt, flags, t_end) if len(tt) else None
    after_s = None if t_on is None else float(t_on - t_end)
    first = tt[tt >= t_end - 1e-12]
    never_left = bool(t_on is not None and len(first) and t_on == float(first[0]))
    full = until_s is None or float(t[-1]) >= until_s - 2 * control.FEEDBACK_EVERY * rec.dt
    sel = tt >= t_end - 1e-12
    return dict(reading, standing=standing, after_s=after_s, never_left_command=never_left,
                recovered=bool(full and after_s is not None and after_s <= limit_s), limit_s=limit_s,
                trace_until_s=float(t[-1]), planned_until_s=until_s,
                on_command_frac_after_pulse=float(np.mean(flags[sel])) if sel.any() else None,
                com_displacement_m=float(np.max(np.linalg.norm(com[k0:] - com[min(k0, len(t) - 1)], axis=1)))
                if k0 < len(t) else None)


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
    """The episodes' counts and bounds, and the pass (``PASS_RULE``); episodes that failed to run
    (``FAILED_EPISODE_RULE``) are counted apart (errors, not falls) and leave the record unjudged."""
    n = len(episodes)
    errors = int(sum(1 for e in episodes if e.get("error")))
    surv = int(sum(not e["fell"] for e in episodes))
    succ = int(sum(e["succeeded"] for e in episodes))
    judged = n == RUNS and [e["seed"] for e in episodes] == list(seeds) == list(SEEDS) and errors == 0
    return dict(episodes=n, survived=surv, succeeded=succ, falls=n - surv - errors, errors=errors,
                survival_lcb=lcb(surv, n), success_lcb=lcb(succ, n), judged=judged,
                pass_=bool(surv == RUNS and succ >= MIN_SUCCEED) if judged else None)


def success_label(episodes: list[dict]) -> dict:
    """``SUCCESS_LABEL_RULE`` over the episodes that survived but failed."""
    failed = [e for e in episodes if not e["fell"] and not e["succeeded"]]
    c = Counter(t for e in failed for t in e["failed_terms"])
    hw = [t for t in HARDWARE_TERMS if c.get(t, 0) > len(failed) / 2]
    return dict(failed_episodes=len(failed), terms={t: dict(count=n, cls=term_class(t)) for t, n in c.most_common()},
                hardware_terms=hw, label=("body-limited: " + ", ".join(hw)) if hw else "controller-limited")


def failure(row: str, episodes: list[dict], attribution: dict | None, pass_: bool | None) -> dict | None:
    """``FAILURE_LABEL_RULE``: what a record's failure is, both labels and the record's (None if it passes or is
    not judged)."""
    if pass_ is None or pass_:
        return None
    falls = [e for e in episodes if e["fell"] and not e.get("error")]
    cands = Counter(c for e in falls for c in (e.get("fall_candidates") or ()))
    sl = success_label(episodes)
    out = dict(falls=len(falls), fall_causes=dict(Counter(str(e["fall_cause"]) for e in falls)),
               fall_candidates=dict(cands), success=sl, success_failures=sl["failed_episodes"])
    falls_label = None
    if falls:
        if attribution is not None:
            falls_label = attribution["label"]
        elif controller_of(row) in ATTRIBUTED:
            falls_label = "not attributed (falls; run eval --attribute)"
        else:
            falls_label = f"not attributed (falls; attribution is applied to {', '.join(ATTRIBUTED)} rows)"
    success_label_ = sl["label"] if sl["failed_episodes"] else None
    if not falls:
        kind = "success"
    elif sl["failed_episodes"] <= RUNS - MIN_SUCCEED:  # the success failures alone would not fail it
        kind = "falls"
    else:
        kind = "falls" if len(falls) >= sl["failed_episodes"] else "success"
    return dict(out, falls_label=falls_label, success_label=success_label_, kind=kind,
                label=falls_label if kind == "falls" else success_label_,
                untestable=list((attribution or {}).get("not_relaxable") or []))


def verdict(records: list) -> dict:
    """``VERDICT_RULE`` on one controller's records (its two seeds' rows, ``VERDICT_ROWS``)."""
    if any(r is not None and r.get("status") == "not run" for r in records):
        return dict(verdict="not run", rule=VERDICT_RULE)
    if any(r is None or not r.get("counts", {}).get("judged") for r in records):
        return dict(verdict="incomplete", rule=VERDICT_RULE, missing=[i for i, r in enumerate(records) if r is None],
                    errors=[int((r.get("counts") or {}).get("errors") or 0) for r in records if r is not None])
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
    untestable = sorted({c for r, p in zip(records, passes, strict=True) if not p
                         for c in (r.get("failure") or {}).get("untestable") or ()})
    return dict(verdict=v, rule=VERDICT_RULE, rows=[r["row"] for r in records], passes=passes, labels=labels,
                untestable=untestable)


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


#: The designlab/stage1 modules the key leaves out (``PANEL_SOURCES_RULE``: held by value or source instead).
KEY_EXCLUDED = ("env.py", "gates.py")
#: The libraries whose versions the key holds (``RESUME_RULE``).
KEY_VERSIONS = ("python", "numpy", "mujoco", "scipy")
DESIGNLAB = plant.SOURCES.parent


def designlab_files() -> dict[str, bytes]:
    """designlab's own ``*.py`` files (outside designlab/stage1) as they are on disk: name -> bytes."""
    return {q.name: q.read_bytes() for q in DESIGNLAB.glob("*.py")}


def _module_imports(source: bytes, level: int) -> set[str]:
    """The designlab modules (file names) a module imports when it is imported: its top-level import statements
    (and those in top-level if and try blocks), relative at ``level`` (2 from designlab/stage1, 1 from designlab) or
    absolute (designlab.x)."""
    out, todo = set(), list(ast.parse(source).body)
    while todo:
        node = todo.pop(0)
        if isinstance(node, (ast.If, ast.Try)):
            todo[:0] = (list(node.body) + list(node.orelse) + [x for h in getattr(node, "handlers", ()) for x in h.body]
                        + list(getattr(node, "finalbody", ())))
        elif isinstance(node, ast.ImportFrom):
            if node.level == level:
                mods = [node.module.split(".")[0]] if node.module else [a.name for a in node.names]
            elif node.level == 0 and node.module and node.module.split(".")[0] == "designlab":
                parts = node.module.split(".")
                mods = [parts[1]] if len(parts) > 1 else [a.name for a in node.names]
            else:
                continue
            out.update(f"{m}.py" for m in mods)
            out.add("__init__.py")
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "designlab":
                    out.add("__init__.py")
                    if len(parts) > 1:
                        out.add(f"{parts[1]}.py")
    return out


def designlab_modules(stage1_files: dict[str, bytes], files: dict[str, bytes] | None = None) -> list[str]:
    """The designlab modules outside designlab/stage1 that the modules ``stage1_files`` (name -> bytes) import when
    they are imported, followed through designlab's own module-level imports (``PANEL_SOURCES_RULE``)."""
    files = designlab_files() if files is None else files
    todo = sorted(set().union(*(_module_imports(b, 2) for b in stage1_files.values())) & set(files))
    seen = set()
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo.extend(sorted((_module_imports(files[name], 1) & set(files)) - seen))
    return sorted(seen)


def key_modules(files: dict[str, bytes] | None = None) -> list[str]:
    """The designlab/stage1 modules in the panels' sources: panels.py's imports followed as ``tune.tuner_modules``
    follows the tuner's, without descending into ``KEY_EXCLUDED``."""
    files = tune.package_files() if files is None else files
    seen, todo = set(), ["__init__.py", "panels.py"]
    while todo:
        name = todo.pop()
        if name not in seen and name not in KEY_EXCLUDED:
            seen.add(name)
            todo.extend(sorted(tune._imports(name, files) - seen))
    return sorted(seen)


def panel_sources(files: dict | None = None, dl_files: dict | None = None) -> dict:
    """``PANEL_SOURCES_RULE``: the hash of the panels' sources (designlab/stage1 and designlab modules), with the
    module lists and each part's hash."""
    files = tune.package_files() if files is None else files
    dl_files = designlab_files() if dl_files is None else dl_files
    mods = key_modules(files)
    dl = designlab_modules({m: files[m] for m in mods}, dl_files)
    stage1_sha = plant.sources_sha256({m: files[m] for m in mods})
    dl_sha = plant.sources_sha256({m: dl_files[m] for m in dl})
    sha = hashlib.sha256(f"designlab/stage1\0{stage1_sha}\0designlab\0{dl_sha}\0".encode()).hexdigest()
    return dict(sha256=sha, rule=PANEL_SOURCES_RULE, modules=mods, stage1_sha256=stage1_sha, designlab_modules=dl,
                designlab_sha256=dl_sha, excluded=list(KEY_EXCLUDED))


def library_versions() -> dict:
    """``KEY_VERSIONS``' versions (as plant.provenance reads them)."""
    import platform
    from importlib import metadata

    out = {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__}
    for pkg in KEY_VERSIONS[3:]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def env_values(body: str) -> dict:
    """What the panels take from env.py (``PANEL_SOURCES_RULE``), by value, and env.foot_tables by its source's
    sha256."""
    return dict(push_bound_m_s=float(env.PUSH_BOUND_M_S[body]), residual_rate=float(env.RESIDUAL_RATE),
                env_substeps=int(env.ENV_SUBSTEPS), carpet_walk=[list(x) for x in env.TASKS["carpet_walk"][0]],
                carpet_friction=float(env.TASK_FRICTION["carpet_walk"]),
                foot_tables_sha256=hashlib.sha256(inspect.getsource(env.foot_tables).encode()).hexdigest())


def key_context(body: str) -> dict:
    """What every record key of ``body`` holds besides its row, condition and inputs (``RESUME_RULE``): the sha256 of
    the round-two files the plant reads, the library versions and what the panels take from env.py
    (``env_values``)."""
    xml, ledger = plant.model_files(body)
    cd = case_dir(body)
    files = dict(record=cd / "candidates" / f"{body}.json", mjcf=xml, ledger=ledger,
                 clearance=cd / "review" / "screening" / f"{body}_clearance.json")
    return dict(plant_inputs={k: plant.sha256(v) for k, v in files.items()}, versions=library_versions(),
                env=env_values(body))


def fingerprint(sources: str, context: dict) -> str:
    """The sha256 of a session's panel sources and key context, which every worker must reproduce (``PanelWorker``)."""
    return hashlib.sha256(json.dumps(dict(sources=sources, context=_jsonable(context)), sort_keys=True).encode()
                          ).hexdigest()


def _portable(x):
    """``x`` with every absolute path under the study directory made relative to it (a key does not depend on where
    the checkout is)."""
    if isinstance(x, dict):
        return {k if not isinstance(k, str) else _portable(k): _portable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_portable(v) for v in x]
    if isinstance(x, str) and x.startswith(str(STUDY) + os.sep):
        return Path(x).relative_to(STUDY).as_posix()
    return x


def record_key(body: str, row: Row, cond: Condition, resolved: dict, seeds, inputs: dict, sources: str,
               context: dict | None = None) -> dict:
    """``RESUME_RULE``'s key of a record (``context``: ``key_context(body)``)."""
    files = files_of(cond)
    context = key_context(body) if context is None else context
    push = cond.judge in ("push_stand", "push_walk")
    return _jsonable(dict(schema=SCHEMA, body=body, row=_portable(row.describe()), row_sources=_portable(row.sources()),
                          condition=cond.describe(), plan=[list(x) for x in resolved["plan"]],
                          seconds=resolved["seconds"], seeds=list(seeds),
                          push_bound_m_s=float(env.PUSH_BOUND_M_S[body]) if push else None,  # (and in context)
                          inputs=dict(files={f: inputs["files"][f] for f in files}, records=inputs["records"]),
                          context=context, sources=sources))


def condition_record(body: str, row: Row, cond: Condition, episodes: list[dict], key: dict, seeds, prov: dict,
                     inputs: dict, wall_s: float, attribution: dict | None = None, started_at: str | None = None) -> dict:
    """A condition's record (module docstring)."""
    c = counts(episodes, seeds)
    return dict(schema=SCHEMA, body=body, row=row.name, behavior=cond.behavior, condition=cond.name, key_name=cond.key,
                gated=row.gated(cond), plan_gated=cond.gated, status="done", controller=row.describe(),
                definition=cond.describe(), plan=key["plan"], seconds=key["seconds"], seeds=list(seeds), counts=c,
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
                gated=row.gated(cond), plan_gated=cond.gated, status="not run", reason=NOT_RUN_RULE, settle=settle,
                controller=row.describe(), definition=cond.describe(),
                counts=dict(episodes=0, errors=0, judged=False, pass_=None), failure=None, episodes=[],
                rules=rules_of(cond), key=key, provenance=prov, ended_at=tune._now())


def _attribution_needed(rec: dict, attribute: bool) -> bool:
    """With --attribute, a record of an attributed controller with falls and no attribution yet, or one whose lifted
    runs failed to run (a smoke panel of fewer seeds is attributed over its own seeds, for the record only: it is not
    judged); not while the record holds episodes that failed to run (they are run again first)."""
    if not (attribute and rec.get("status") == "done" and controller_of(rec["row"]) in ATTRIBUTED):
        return False
    c = rec["counts"]
    if c.get("errors") or c["falls"] <= 0:
        return False
    a = rec.get("attribution")
    return a is None or bool(a.get("lifted_errors"))


def attribution_plan(rec: dict) -> tuple[list, list]:
    """The candidates of a record's falls (``control.HW_LIMITS`` order) and the relaxations that lift them; none are
    lifted for fewer than ``ATTRIBUTION_MIN_FALLS`` falls (``PANEL_ATTRIBUTION_RULE``)."""
    falls = [e for e in rec["episodes"] if e["fell"] and not e.get("error")]
    cands = [c for c in control.HW_LIMITS if any(c in (e.get("fall_candidates") or ()) for e in falls)]
    lift = [r for r in control.RELAXATIONS if any(control.RELAXES[c] == r for c in cands)]
    return cands, (lift if len(falls) >= ATTRIBUTION_MIN_FALLS else [])


def attribution_record(rec: dict, cands: list, lifted: dict, wall_s: float) -> dict:
    """``PANEL_ATTRIBUTION_RULE``: gates.attribute on the record's episodes and the lifted runs (no null), the pairs
    with an episode that failed to run left out; the label says why a controller-limited condition could not be
    found body-limited (too few falls, untestable candidates) and is 'not attributed' while lifted runs failed."""
    eps = rec["episodes"]
    keep = [i for i, e in enumerate(eps) if not e.get("error") and not any(v[i].get("error") for v in lifted.values())]
    base = [dict(fell=eps[i]["fell"], strides=eps[i]["strides"]) for i in keep]
    rows = {r: [dict(fell=v[i]["fell"], strides=v[i]["strides"]) for i in keep] for r, v in lifted.items()}
    a = gates.attribute(base, rows, None, cands)
    falls = int(sum(1 for e in eps if e["fell"] and not e.get("error")))
    lifted_errors = int(sum(1 for v in lifted.values() for x in v if x.get("error")))
    label = a["label"]
    if lifted_errors:
        label = (f"not attributed ({lifted_errors} lifted episodes failed to run, FAILED_EPISODE_RULE; run eval "
                 "--attribute again)")
    elif not a["binding"]:
        notes = []
        if falls < ATTRIBUTION_MIN_FALLS:
            notes.append(f"{falls} falls: the sign test needs {ATTRIBUTION_MIN_FALLS} for p < "
                         f"{gates.ATTRIBUTION_ALPHA:g}, so no limit was lifted")
        if a["not_relaxable"]:
            notes.append("untestable candidates: " + ", ".join(a["not_relaxable"]))
        if notes:
            label = f"controller-limited (decision 20; {'; '.join(notes)})"
    return dict(a, label=label, decision_20_label=a["label"], panel_rule=PANEL_ATTRIBUTION_RULE, falls=falls,
                min_falls=ATTRIBUTION_MIN_FALLS, pairs=len(keep), pairs_left_out=len(eps) - len(keep),
                lifted_errors=lifted_errors, episodes=int(sum(len(v) for v in lifted.values())),
                gates_sha256=plant.sha256(Path(gates.__file__)),
                runs={r: [dict(seed=x["seed"], fell=x["fell"], strides=x["strides"], fall_time_s=x["fall_time_s"],
                               fall_cause=x["fall_cause"], succeeded=x["succeeded"], error=bool(x.get("error")))
                          for x in v] for r, v in lifted.items()}, wall_s=wall_s)


# ------------------------------------------------------------------------------------------------
# Workers (tune.Pool with the panels' evaluator)
# ------------------------------------------------------------------------------------------------
def trace_path(row: str, cond: Condition, seed: int) -> Path:
    """``TRACE_RULE``'s file of an episode, relative to the traces' directory (``<out>/<body>/eval/traces``)."""
    return Path(row) / cond.behavior / cond.name / f"{int(seed)}.npz"


def write_trace(root: Path, rel: Path, rec: metrics.Recorder) -> dict:
    """``TRACE_RULE``: the recorder's rows at 50 Hz and the joint angles into ``root / rel`` (written whole, then
    renamed); its path (relative to ``root``), sha256 and rows."""
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    U = rec.U[: rec.rows : TRACE_EVERY] if rec.rows else np.zeros((0, len(metrics.COLS)))
    Q = rec.Q[: rec.rows : TRACE_EVERY] if rec.rows else np.zeros((0, 0))
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, t=U[:, metrics.C["t"]].astype(np.float64), t_ref=U[:, metrics.C["t_ref"]].astype(np.float64),
                        rows=U.astype(np.float32), cols=np.array(metrics.COLS), q=Q.astype(np.float32))
    os.replace(tmp, path)
    return dict(path=Path(rel).as_posix(), sha256=plant.sha256(path), rows=int(len(U)),
                every_s=TRACE_EVERY * control.FEEDBACK_EVERY * float(getattr(rec, "dt", 0.0) or 0.0))


class PanelWorker:
    """A worker's evaluator for ``tune.Pool``: ``evaluate((row, condition, relax), seed)`` runs one episode.  It refuses
    to start on other panel sources or key context than the session's (``sources``: the session's ``fingerprint``,
    ``PANEL_SOURCES_RULE``), so a replacement worker never runs other code than the records' key names; ``traces`` is
    the traces' directory (``TRACE_RULE``; None: none written)."""

    def __init__(self, body: str, run_dir: str, mode: str = "panels", rate: float = RATE,
                 expected_inputs: dict | None = None, rows=(), sources: str | None = None, traces: str | None = None):
        if sources is not None:
            mine = fingerprint(panel_sources()["sha256"], key_context(body))
            if mine != sources:
                raise tune.PoolError(f"the panels' sources or key context changed during the session ({mine[:12]}..., "
                                     f"the session's {sources[:12]}...): restore them and run eval again")
        self.body, self.run_dir = body, Path(run_dir)
        self.inputs, docs = tune.read_inputs(self.run_dir, panel_files())
        if expected_inputs is not None:
            diff = tune.input_differences(expected_inputs, self.inputs)
            if diff:
                raise tune.InputError(f"{self.run_dir}: the inputs differ from the ones expected: " + "; ".join(diff))
        self.ctx = Context(body, self.run_dir, docs)
        self.rows = {s["name"]: row_from_spec(s) for s in rows}
        self.traces = None if traces is None else Path(traces)

    def describe(self) -> dict:
        return dict(body=self.body, rows=sorted(self.rows), panel_sources=panel_sources()["sha256"])

    def evaluate(self, job, seed) -> dict:
        row, cond, relax = job
        root = self.traces if not relax else None  # the attribution's lifted runs leave no trace
        try:
            return self.rows[row].run(self.ctx, cond, int(seed), tuple(relax), trace_root=root)
        except Exception:
            return failed_episode(cond, int(seed), traceback.format_exc(), relax)


def _worker(body, run_dir, mode, rate, expected_inputs=None, rows=(), sources=None, traces=None):
    return PanelWorker(body, run_dir, mode, rate, expected_inputs, rows, sources, traces)


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
             traces: bool = True, log=print) -> dict:
    """The panels of ``body`` (module docstring) into ``<out>/<body>/eval/``; the references are read from
    ``<inputs>/<body>/`` (default: out); ``traces`` writes each episode's trace (``TRACE_RULE``).  Returns the
    session's summary."""
    if body not in BODIES:
        raise KeyError(body)
    t_start = time.perf_counter()
    run_dir = Path(inputs or out) / body
    seeds = SEEDS[: int(episodes)]
    conds = select(conditions(), behaviors, names)
    inp, docs = tune.read_inputs(run_dir, panel_files())
    planner = Planner(run_dir, docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
    row_objs = tracker_rows(body, run_dir, rows)
    prov, src, kctx = plant.provenance(), panel_sources(), key_context(body)
    trace_root = Path(out) / body / "eval" / "traces" if traces else None
    pool = None
    session = dict(started_at=tune._now(), rows=list(rows), behaviors=None if behaviors is None else list(behaviors),
                   conditions=None if names is None else list(names), seeds=list(seeds), workers=int(workers),
                   attribute=bool(attribute), traces=bool(traces), episodes_run=0, attribution_episodes_run=0,
                   skipped=[], rerun_failed=[], not_run=[], incidents=[])

    def get_pool():
        nonlocal pool
        if pool is None:
            pool = tune.Pool(body, run_dir, workers, mode="panels", rate=RATE, timeout_s=timeout_s,
                             factory=functools.partial(_worker, rows=[r.spec() for r in row_objs.values()],
                                                       sources=fingerprint(src["sha256"], kctx),
                                                       traces=None if trace_root is None else str(trace_root)))
        return pool

    try:
        for cond in conds:
            res = planner.resolve(cond)
            todo, ok_floor = [], stands(run_dir, cond.floor)  # (row, path, key, record or None, seeds to run)
            for row in row_objs.values():
                if not row.runs(cond):
                    continue
                path = record_path(out, body, row.name, cond)
                key = record_key(body, row, cond, res, seeds, inp, src["sha256"], kctx)
                old = _read(path)
                if not ok_floor[0]:
                    if not (old and old.get("key") == key and old.get("status") == "not run"):
                        _write_json(path, not_run_record(body, row, cond, key, ok_floor[1], prov))
                    session["not_run"].append(f"{row.name} {cond.key}")
                    continue
                if (old and old.get("key") == key and old.get("status") == "done"
                        and [e["seed"] for e in old["episodes"]] == list(seeds)):
                    redo = [e["seed"] for e in old["episodes"] if e.get("error")]  # FAILED_EPISODE_RULE: run again
                    if redo:
                        session["rerun_failed"].append(f"{row.name} {cond.key}: seeds {redo}")
                    else:
                        session["skipped"].append(f"{row.name} {cond.key}")
                    todo.append((row, path, key, old, redo))
                    continue
                todo.append((row, path, key, None, list(seeds)))
            jobs = [((row.name, cond, ()), s) for row, _, _, _, run in todo for s in run]
            if jobs:
                t0 = time.perf_counter()
                results = get_pool().map(jobs, failed=_lost)
                wall = time.perf_counter() - t0
                session["episodes_run"] += len(jobs)
                i = 0
                for k, (row, path, key, old, run) in enumerate(todo):
                    if not run:
                        continue
                    new = dict(zip(run, results[i : i + len(run)], strict=True))
                    i += len(run)
                    eps = [new[s] for s in seeds] if old is None else [new.get(e["seed"], e) for e in old["episodes"]]
                    share = wall * len(run) / len(jobs) + (float(old.get("wall_s") or 0.0) if old else 0.0)
                    rec = condition_record(body, row, cond, eps, key, seeds, prov, inp, share,
                                           started_at=session["started_at"])
                    _write_json(path, rec)
                    todo[k] = (row, path, key, rec, run)
                    c = rec["counts"]
                    log(f"{body} {row.name} {cond.key}: {c['survived']}/{c['episodes']} survive, {c['succeeded']}"
                        f"/{c['episodes']} succeed" + (f", {c['errors']} failed to run" if c["errors"] else "")
                        + ("" if c["pass_"] is None else f" ({'pass' if c['pass_'] else 'FAIL'})")
                        + ("" if row.gated(cond) else " [reported]"))
            # decision 20's attribution of the attributed rows' falls (PANEL_ATTRIBUTION_RULE), in one batch
            plans = [(row, path, rec, *attribution_plan(rec)) for row, path, key, rec, _ in todo
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
    session.update(ended_at=tune._now(), wall_s=time.perf_counter() - t_start, provenance=prov, panel_sources=src,
                   key_context=kctx)
    return summarize(body, out, session=session, inputs=inputs)


# ------------------------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------------------------
def _cell(rec: dict | None) -> dict | None:
    if rec is None:
        return None
    c = rec.get("counts") or {}
    f = rec.get("failure") or {}
    a = rec.get("attribution") or {}
    return dict(status=rec.get("status"), gated=rec.get("gated"), episodes=c.get("episodes"),
                survived=c.get("survived"), succeeded=c.get("succeeded"), errors=c.get("errors"),
                survival_lcb=c.get("survival_lcb"), success_lcb=c.get("success_lcb"), judged=c.get("judged"),
                pass_=c.get("pass_"), label=f.get("label"), kind=f.get("kind"), falls_label=f.get("falls_label"),
                success_label=f.get("success_label"), untestable=a.get("not_relaxable") or None,
                attribution_episodes=a.get("episodes"), wall_s=rec.get("wall_s"))


def verdict_controller(found) -> tuple[str, str]:
    """The controller whose verdicts the summary shows (``VERDICT_RULE``) and why: the judged one once its records
    exist, else R2's, interim."""
    if any(r in found for r in VERDICT_ROWS[JUDGED_CONTROLLER]):
        return JUDGED_CONTROLLER, f"{JUDGED_CONTROLLER}: the controller judged on every behaviour (plan section 2)"
    return "R2", (f"R2 (interim: {JUDGED_CONTROLLER} is the controller judged on every behaviour, plan section 2; R2 "
                  "is judged only if G6c fails, plan section 3 step 7)")


def summarize(body: str, out: Path, rows=None, session: dict | None = None, inputs: Path | None = None) -> dict:
    """``<out>/<body>/eval/summary.json``: per behaviour and condition the rows' cells and each controller's verdict
    (the one shown: ``verdict_controller``), the episode counts, G7, and the sessions (each ``evaluate`` call appends
    its own)."""
    edir = Path(out) / body / "eval"
    path = edir / "summary.json"
    old = _read(path) or {}
    run_dir = Path(inputs or out) / body
    rows = list(rows or ROWS)
    found = {"/".join(q.relative_to(edir).parts[:-2]) for q in edir.glob("**/*.json")
             if len(q.relative_to(edir).parts) >= 3 and q.relative_to(edir).parts[0] != "traces"}
    rows += sorted(r for r in found if r and r not in rows)
    shown, note = verdict_controller(found)
    ctrls = [c for c, rs in VERDICT_ROWS.items() if c == shown or any(r in found for r in rs)]
    grid, n_eps, n_attr = {}, Counter(), Counter()
    for cond in conditions():
        recs = {r: _read(record_path(out, body, r, cond)) for r in rows}
        cells = {r: _cell(v) for r, v in recs.items() if v is not None}
        for r, v in recs.items():
            if v is not None:
                n_eps[r] += len(v.get("episodes") or ())
                n_attr[r] += int((v.get("attribution") or {}).get("episodes") or 0)
        verdicts = {c: dict(verdict([_read(record_path(out, body, r, cond)) for r in VERDICT_ROWS[c]]),
                            gated=gated_for(VERDICT_ROWS[c][0], cond)) for c in ctrls}
        grid.setdefault(cond.behavior, {})[cond.name] = dict(
            gated=verdicts[shown]["gated"], plan_gated=cond.gated, floor=cond.floor, friction=cond.friction,
            rows=cells, verdict=verdicts[shown], verdicts=verdicts)
    by_ctrl = {c: dict(Counter(v["verdicts"][c]["verdict"] for b in grid.values() for v in b.values()
                               if v["verdicts"][c]["gated"])) for c in ctrls}
    try:
        g = g7(body, run_dir)
        _write_json(Path(out) / body / "eval" / "g7.json", g)
    except FileNotFoundError as exc:
        g = dict(error=str(exc))
    doc = dict(schema=SCHEMA, body=body, rules=RULES, verdict_rows={c: list(v) for c, v in VERDICT_ROWS.items()},
               verdict_controller=shown, verdict_controller_note=note, grid=grid, gated_verdicts=by_ctrl.get(shown, {}),
               gated_verdicts_by_controller=by_ctrl, episodes=dict(n_eps), attribution_episodes=dict(n_attr), G7=g,
               sessions=list(old.get("sessions") or []) + ([session] if session else []), written_at=tune._now())
    _write_json(path, doc)
    return doc


def summary_text(doc: dict) -> str:
    """The summary grid as text (one line per condition: its verdict and each row's counts and label)."""
    v = ", ".join(f"{k} {n}" for k, n in sorted(doc["gated_verdicts"].items()))
    lines = [f"{doc['body']} panels, verdicts of {doc.get('verdict_controller_note', doc.get('verdict_controller'))}; "
             f"gated conditions: {v or 'none'}; episodes " + ", ".join(f"{r} {n}" for r, n in doc["episodes"].items())
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
                label = x.get("label")
                other = x.get("success_label") if x.get("kind") == "falls" else x.get("falls_label")
                cells.append(f"{r} {x['survived']}/{x['episodes']} survive, {x['succeeded']} succeed{verdict_}"
                             + (f", {x['errors']} failed to run" if x.get("errors") else "")
                             + (f" ({label}" + (f"; also {other}" if other else "") + ")" if label else "")
                             + ("" if x.get("gated", True) else " [reported]"))
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
class _Timed:
    """``control.run``'s recorder hooks forwarded to a ``metrics.Recorder``, with the process and wall clocks read
    at physics step ``k_mark`` (``COST_RULE``)."""

    def __init__(self, rec: metrics.Recorder, k_mark: int):
        self.rec, self.k_mark, self.mark = rec, int(k_mark), None

    def begin(self, tk, n):
        self.rec.begin(tk, n)

    def step(self, k):
        self.rec.step(k)
        if k == self.k_mark:
            self.mark = (time.process_time(), time.perf_counter())

    def update(self, *args):
        self.rec.update(*args)

    def end(self, steps, out):
        self.rec.end(steps, out)


def measure_costs(body: str, run_dir: Path, floors, skip_s: float = COST_SKIP_S, seconds: float = COST_SAMPLE_S,
                  log=print) -> dict:
    """``COST_RULE``: CPU and wall seconds per simulated second of R2/42 walking on each floor (this process, now),
    over the steady part of the walk."""
    inp, docs = tune.read_inputs(run_dir, panel_files())
    ctx = Context(body, run_dir, docs)
    row = tracker_rows(body, run_dir, ("R2/42",))["R2/42"]
    out = {}
    for floor in floors:
        t0 = time.perf_counter()
        cond = Condition("walk", f"cost_{floor}", False, "walk", floor=floor, seconds=skip_s + seconds)
        p = ctx.plant(floor, None)
        build = time.perf_counter() - t0
        b = ctx.built(cond)
        tk = control.Tracker(p, b.track, "R2", gains=row.gains, rate=RATE)
        timed = _Timed(b.recorder(), int(round(skip_s / float(p.m.opt.timestep))))
        c0, w0 = time.process_time(), time.perf_counter()
        r = control.run(tk, skip_s + seconds, start="stance", seed=1, noise_seed=2, recorder=timed,
                        joint_noise=tune.START_JOINT_NOISE_RAD, root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S)
        c1, w1 = time.process_time(), time.perf_counter()
        steady = timed.mark is not None and r["seconds"] >= skip_s + 1.0
        if steady:
            cpu, wall, sim = c1 - timed.mark[0], w1 - timed.mark[1], r["seconds"] - skip_s
        else:
            cpu, wall, sim = c1 - c0, w1 - w0, r["seconds"]
        sim = max(sim, 1e-9)
        out[floor] = dict(cpu_per_sim_s=cpu / sim, wall_per_sim_s=wall / sim, simulated_s=sim, cpu_s=cpu, wall_s=wall,
                          part=f"simulated {skip_s:g}-{r['seconds']:.2f} s (steady walking)" if steady else
                          f"simulated 0-{r['seconds']:.2f} s (from the start: the walk fell at {r['fall_time_s']})",
                          fell=bool(r["fell"]), plant_build_s=build, load_average=list(os.getloadavg()))
        log(f"{body} {floor}: {out[floor]['cpu_per_sim_s']:.3f} CPU s and {out[floor]['wall_per_sim_s']:.3f} wall s per "
            f"simulated s over {out[floor]['part']} (plant built in {build:.1f} s, load {os.getloadavg()[0]:.1f})")
    return dict(rule=COST_RULE, floors=out, cpu_count=os.cpu_count())


def dry_run(body: str, out: Path, rows=ROWS, behaviors=None, names=None, episodes: int = RUNS,
            inputs: Path | None = None, measure: bool = True, costs: dict | None = None, workers: int = 1,
            log=print) -> dict:
    """Every episode the panels would run (``evaluate``'s plan), which records are up to date, and the core-hours
    (CPU) and wall hours with ``workers`` processes from the cost per simulated second measured now (``COST_RULE``),
    with the attribution's bound."""
    run_dir = Path(inputs or out) / body
    seeds = SEEDS[: int(episodes)]
    conds = select(conditions(), behaviors, names)
    inp, docs = tune.read_inputs(run_dir, panel_files())
    planner = Planner(run_dir, docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
    row_objs = tracker_rows(body, run_dir, rows)
    src, kctx = panel_sources()["sha256"], key_context(body)
    lines, total = [], Counter()
    for cond in conds:
        res = planner.resolve(cond)
        st = stands(run_dir, cond.floor)[0]
        for row in row_objs.values():
            if not row.runs(cond):
                continue
            key = record_key(body, row, cond, res, seeds, inp, src, kctx)
            old = _read(record_path(out, body, row.name, cond))
            n_run = len(seeds)
            if not st:
                status, n_run = "not run (no settled stance)", 0
            elif (old and old.get("key") == key and old.get("status") == "done"
                  and [e["seed"] for e in old["episodes"]] == list(seeds)):
                failed = sum(1 for e in old["episodes"] if e.get("error"))
                status, n_run = ("up to date", 0) if not failed else (f"to run ({failed} failed episodes)", failed)
            else:
                status = "to run"
            lines.append(dict(row=row.name, condition=cond.key, gated=row.gated(cond), floor=cond.floor,
                              friction=cond.friction, seeds=list(seeds), seconds=res["seconds"], status=status,
                              episodes_to_run=n_run, plan=[list(x) for x in res["plan"]]))
    floors = sorted({x["floor"] for x in lines if not x["status"].startswith("not run")})
    if costs is None and measure:
        costs = measure_costs(body, run_dir, floors, log=log)
    fl = (costs or {}).get("floors") or {}
    cpu = {f: (fl.get(f) or {}).get("cpu_per_sim_s") for f in floors}
    wall = {f: (fl.get(f) or {}).get("wall_per_sim_s") for f in floors}
    for x in lines:
        n = 0 if x["status"].startswith("not run") else len(seeds)
        x["episodes"] = n
        x["simulated_s"] = n * x["seconds"]
        c, w = cpu.get(x["floor"]), wall.get(x["floor"])
        x["core_h"] = None if c is None else x["simulated_s"] * c / 3600
        x["core_h_to_run"] = None if c is None else x["episodes_to_run"] * x["seconds"] * c / 3600
        x["wall_h_to_run_1_worker"] = None if w is None else x["episodes_to_run"] * x["seconds"] * w / 3600
        bound = len(control.RELAXATIONS) * x["simulated_s"] if controller_of(x["row"]) in ATTRIBUTED else 0.0
        x["attribution_bound_core_h"] = None if c is None else bound * c / 3600
        total["episodes"] += n
        total["episodes_to_run"] += x["episodes_to_run"]
        total["simulated_s"] += x["simulated_s"]
        for k in ("core_h", "core_h_to_run", "wall_h_to_run_1_worker", "attribution_bound_core_h"):
            total[k] += x[k] or 0.0
        log(f"{x['row']:6s} {x['condition']:32s} {'gated' if x['gated'] else 'reported':8s} {x['floor']:18s} "
            f"{n:2d} x {x['seconds']:6.2f} s (seeds {seeds[0]}-{seeds[-1]})  "
            + (f"{x['core_h']:.3f} core-h" if x["core_h"] is not None else "cost not measured") + f"  [{x['status']}]")
    per_row = {r: int(sum(x["episodes"] for x in lines if x["row"] == r)) for r in row_objs}
    w_n = max(1, int(workers))
    doc = dict(body=body, rule=COST_RULE, costs=costs, episodes=lines, per_row_episodes=per_row, workers=w_n,
               totals=dict(total, wall_h_to_run=total["wall_h_to_run_1_worker"] / w_n))
    log(f"{body}: {total['episodes']} episodes ({', '.join(f'{r} {n}' for r, n in per_row.items())}), "
        f"{total['simulated_s'] / 3600:.2f} simulated h, {total['core_h']:.2f} core-h (CPU) "
        f"({total['core_h_to_run']:.2f} to run, about {total['wall_h_to_run_1_worker'] / w_n:.2f} h of wall clock with "
        f"{w_n} worker{'s' if w_n > 1 else ''} at today's load); attribution at most "
        f"{total['attribution_bound_core_h']:.2f} core-h more (every relaxable limit lifted on every attributed "
        "condition)")
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
           "SEEDS", "TrackerRow", "ZeroResidual", "conditions", "counts", "dry_run", "evaluate", "g7", "gated_for",
           "judge", "lcb", "panel_sources", "recovery", "register_rows", "select", "summarize", "sway", "terms_of",
           "tracker_rows", "verdict"]
