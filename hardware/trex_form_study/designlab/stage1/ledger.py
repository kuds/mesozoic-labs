"""Stage 1 margin ledger (``STAGE1_PLAN.md`` section 5; section 3 step 8's sweeps; section 7's ``sweeps/``): each
Stage 0 margin bisected to failure on a controller row, on episode seeds 3042-3044, with the behaviour panels' runner,
conditions and judgement (``panels``).

Run from hardware/trex_form_study of the main checkout (see the end of this docstring), in the PPO venv for R3::

    python -m designlab.stage1 sweep --body 60-N-T-10 [--body ...] [--rows R2,R3] [--margins grip,push,...]
                                     [--workers 4] [--dry-run] [--no-measure] [--row-module MODULE]
                                     [--inputs DIR] [--allow-rebuilt-models] --out stage1/runs/2026-11-08

Rows (``ROWS_RULE``): R2 is the panels' row of the CMA-ES seed the body-level G5 selection comes from (``--rows R2``),
R3 is the policy rows' R3/42 (``--rows R3``; ``policy_rows.r3_row``, registered through ``panels.register_rows``:
``register_policy_rows`` registers the policy rows' R3 and R4 when a row of theirs is asked for and no module has
registered it; run it in the PPO venv), left out for a body whose R3 is judged with R2 (``train.R2_JUDGED_RULE``); any
row ``panels.tracker_rows`` knows can be named.  Margins (``margins``, plan section 5's table; ``MARGIN_RULES``):
grip, push, stop, drive speed, drive torque, the model doubts (servo kp, latency, trunk mass, pile damping) and pack
voltage.  Each margin is one or more searches (``Search``: one parameter on one panel condition, from its nominal value
toward the far end of a stated interval) and points (``Point``: one value).  How each parameter enters the plant:
``VARIANT_RULE`` and the margin's own rule.  How the ledger reads plan section 5 where it is not explicit:
STAGE1_PLAN.md decision 33.

A search (``SEARCH_RULE``, ``Bisect``) evaluates the nominal value, then the far end, then halves the bracket 7 times;
a value works when all three episodes survive and none fails a success term its seed passed at the nominal value
(``WORKS_RULE``), and every search is also run on survival alone, sharing the evaluations.  The value that bounds a
threshold is labelled by what failed there (``LABEL_RULE``), its falls by lifting each candidate hardware limit on the
same 3 seeds (``LIFT_RULE``).  Each (body, row, margin) gives one record ``<out>/<body>/sweeps/<row>/<margin>.json``
(``RECORD_RULE``), written after every round of evaluations; a sweep resumes from the evaluations its records hold
(``RESUME_RULE``).  ``dry_run`` lists every search with the episodes it can still need and their cost
(``COST_RULE``).

Sweep from the checkout whose run directory the records join (the main checkout): every evaluation key holds the
panels' record key, which holds the sha256 of the round-two model files the plant reads (``panels.key_context``); a
checkout without the git-ignored ``round2/refine/<case>/models`` rebuilds them into ``stage1/models`` with another
model-ledger JSON, so its keys differ from the main checkout's and everything it records would be run again there
(``sweep`` refuses such a checkout unless ``--allow-rebuilt-models``; copy ``round2/refine/*/models`` in instead).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import math
import os
import time
import traceback
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .. import carpet, sizing
from . import BODIES, STUDY, case_dir, control, drive, env, metrics, panels, plant, tune
from .transitions import TRANSITIONS, Transition

SCHEMA = 2
#: Plan section 5: the ledger's episode seeds, the panels' first three (``SEEDS_RULE``).
SEEDS = panels.SEEDS[:3]
HALVINGS = 7
CRITERIA = ("success", "survival")
#: The rows plan section 5 bisects on: "R2" is the body-level G5 selection's row, "R3" R3/42 (``ROWS_RULE``).
DEFAULT_ROWS = ("R2", "R3")
R3_ROW = "R3/42"
#: The intervals (each search's far end; the nominal value is the panels' condition as it runs).
MU_NOMINAL, MU_FAR = 0.9, 0.1
PUSH_FAR_BOUNDS = 2.0  # x the Stage 0 forward push bound
SCALE_FAR = 0.25
KP_FAR = 0.25
LATENCY_MAX_STEPS = 10  # physics steps of 2 ms: 20 ms
MASS_POINT, MASS_FAR = 1.1, 1.3
#: Each body's trunk body and whole-robot mass (kg; the screened MJCF's body_mass and body_subtreemass of 'trunk',
#: the root: checked against the model in the tests), for MASS_RULE's two readings of plan section 5's '+10 %'.
MASSES = {"60-N-T-10": (4.6998, 10.8117), "60-N-H-10": (4.8028, 10.9066), "75-M-H-10": (6.1609, 14.8549)}
#: The trunk factor that adds 10 % of the robot's mass (1 + 0.1 robot / trunk, 3 decimals; MASS_RULE).
ROBOT_MASS_POINT = {b: round(1.0 + 0.1 * robot / trunk, 3) for b, (trunk, robot) in MASSES.items()}
DAMPING_LOW, DAMPING_HIGH = 0.5, 2.0
VOLTAGES = (25.2, 22.2, 21.0)
#: The plan's decision that reads plan section 5 for the ledger (stop, works, labels, seeds, mass, pile damping).
DECISION = 33
#: The parameter a variant changes: name and unit (for the records and the report).
PARAMETERS = {"friction": ("sliding friction mu", ""), "push_dv": ("push velocity change", "m/s"),
              "speed_scale": ("envelope speed scale", ""), "torque_scale": ("drive torque scale", ""),
              "kp": ("servo kp factor", ""), "latency_steps": ("actuation latency", "physics steps of 2 ms"),
              "mass": ("trunk mass factor", ""), "pile_damping": ("pile damping factor", ""),
              "bus_v": ("bus voltage", "V")}
KINDS = tuple(PARAMETERS)
#: Plan section 5's Stage 0 column, per body (STAGE0_RULE): the round-two screens' values as the plan's table gives
#: them, its grip cell read per body as walk / turn / brisk (ROUND2_REVIEW.md 7.4 and 7.6, 'Friction need: walk / turn /
#: brisk'), and beside the drives' the values the ledger's own limits compare (no speed margin; the walk's RMS alone).
STAGE0 = {
    "60-N-T-10": dict(grip_walk_nominal=0.48, grip_turn=0.54, grip_walk_brisk=0.75, push_forward=0.487,
                      stop_one_foot=1.25, drive_speed_knee=0.93, drive_speed_knee_no_margin=0.81,
                      drive_torque_sustained=0.76, drive_torque_walk_rms=0.76),
    "60-N-H-10": dict(grip_walk_nominal=0.47, grip_turn=0.53, grip_walk_brisk=0.73, push_forward=0.478,
                      stop_one_foot=1.25, drive_speed_knee=0.95, drive_speed_knee_no_margin=0.83,
                      drive_torque_sustained=0.76, drive_torque_walk_rms=0.76),
    "75-M-H-10": dict(grip_walk_nominal=0.36, grip_turn=0.41, grip_walk_brisk=0.56, push_forward=0.589,
                      stop_one_foot=1.41, drive_speed_knee=0.93, drive_speed_knee_no_margin=0.81,
                      drive_torque_sustained=0.77, drive_torque_walk_rms=0.66),
}
STAGE0_VOLTAGE = ("Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement "
                  "including the screens' 15 % speed margin (8.08-8.12 rad/s; the peak swing speed itself is about "
                  "7.0 rad/s)")
#: The turning screen's runs behind the grip values (``screens``).
TURNING_RUNS = dict(walk_nominal="straight", turn_left="walk_0.35_left", turn_right="walk_0.35_right",
                    walk_brisk="straight_brisk")
STAGE0_RULE = (
    f"plan section 5's Stage 0 column per body, from the round-two records (STAGE1_PLAN.md decision {DECISION}): grip, "
    "the friction the turning screen requires in single support (<case>/review/screening/<body>_turning.json, the "
    "worse foot's runs.<run>.feet.*.single_support.mu_req_max: 'straight' for the nominal walk, 'walk_0.35_left' and "
    "'_right' for the 0.35 rad/s walking turn, the worse direction for 'turn', 'straight_brisk' for the brisk walk; "
    "the plan's grip cell is laid out per body, walk / turn / brisk, as ROUND2_REVIEW.md 7.4 and 7.6 give it); push, "
    "the envelope screen's forward bound; stop, the one-foot deceleration; drive speed, the knee's sizing speed ratio "
    f"at Fr 0.05 (<case>/candidates/<body>.json margins.knee.speed_ratio = {sizing.SPEED_MARGIN:g} x the peak speed / "
    "the rated speed at 24 V: it includes the screens' 15 % speed margin, sizing.SPEED_MARGIN; the value without it, "
    "the peak speed over the rated speed, is what the ledger's limit compares, peak speed <= corner speed x s, so "
    "equal performance gives s near it: 0.81 / 0.83 / 0.81); drive torque, the worst drive's sustained-torque ratio "
    "margins.<drive>.cont_ratio = the larger of the nominal walk's RMS and the one-foot hold over the sustained torque "
    "(sizing.requirements: cont = max(rms, hold)), with the nominal walk's worst RMS ratio beside (what the ledger's "
    "RMS limit on the walk compares: 0.76 / 0.76 / 0.66); each record holds the screens' unrounded values and the "
    "files' sha256 (screens)")

# ------------------------------------------------------------------------------------------------
# Rules (each copied into every record)
# ------------------------------------------------------------------------------------------------
ROWS_RULE = (
    "plan section 5 bisects on R2 and R3: R2 is the panels' row of the CMA-ES seed the body-level G5 selection comes "
    "from (env.tuned_gains, env.GAINS_RULE: 60-N-T-10 R2/42, 60-N-H-10 R2/43, 75-M-H-10 R2/43; the row name 'R2' "
    "names it), the tracker on the estimate with that seed's own G5 selection (panels.R2_GAINS_RULE); R3 is R3/42, the "
    "policy rows' row (policy_rows.r3_row, policy_rows.POLICY_ROWS_RULE: R2 of the body-level G5 selection plus PPO "
    "seed 42's residual policy acting as training's env acts) at its judged checkpoint (train.JUDGED_RULE, "
    "policy_rows.CHECKPOINT_SELECTION_RULE), registered through panels.register_rows: the ledger registers the policy "
    "rows' R3 and R4 (policy_rows.ROW_FACTORIES) when a row of theirs is asked for and no module has registered that "
    "controller (a --row-module may register another, which is kept); a policy row is built only in the PPO venv "
    "(its run's libraries).  The row name 'R3' leaves R3 out for a body judged with R2 (train.R2_JUDGED_RULE: G6c "
    "failed on seed 42, or FIX_RULE stopped one of its R3 runs), recorded in the session; R3/42 named explicitly is "
    "built as asked.  Any row panels.tracker_rows knows can be named, and a row that runs the hard floor only (R0, R1) "
    "has no pile search.  One seed per controller (the body-level selection's CMA-ES seed for R2, PPO seed 42 for R3), "
    f"not plan section 6's 'R2 and R3 on 2 seeds each': the budget of decision 21 (STAGE1_PLAN.md decision "
    f"{DECISION})")
SEEDS_RULE = (
    f"episode seeds {SEEDS[0]}-{SEEDS[-1]} (panels.SEEDS' first three, plan section 5); each episode is the panels' "
    "episode of its condition on that seed (panels.EPISODE_RULE: the start noise, the sensor noise and a push's timing "
    "from the seed), on the panels' plant and track with the search's variant applied (VARIANT_RULE), so the nominal "
    "value of a condition the variant leaves unchanged is the panels' own episode bit for bit; no traces are written")
WORKS_RULE = (
    "a value works if none of the 3 episodes falls and none fails a success term that its seed passed at the search's "
    "nominal value (a point: its condition's nominal evaluation, which it then needs as well), by the panels' "
    "judgement of the condition (panels.judge: its success terms with panels.LIMITS_RULE; a push succeeds if the body "
    "is back on its command within 3 s, decision 30, panels.RECOVERY_RULE).  The margin is how far the parameter moves "
    "before the row loses something it had at nominal: a term a seed already fails at nominal does not end the search "
    "(60-N-T-10's R2/42 nominal walk passes the panels 20/20 survived and 18/20 succeeded, and seed 3042 fails the "
    "peak GRF alone; read outright, every margin on that walk would end 'fails at nominal'), and at the nominal value "
    "itself works is survives.  Every evaluation also records the outright reading (how many of the 3 succeed, and "
    "whether all do).  A value survives if none of the 3 falls (plant.Plant.fall_cause, an episode with a MuJoCo "
    "warning counting as a fall, tune.FAILURE_RULE).  An episode that failed to run (panels.FAILED_EPISODE_RULE) "
    f"leaves the value unjudged: the search waits, and the next sweep runs that value again (decision {DECISION})")
SEARCH_RULE = (
    "each search moves one parameter on one panel condition from its nominal value (the condition as the panels run "
    "it) toward the far end of its interval: the nominal value first, and if it does not work the search ends 'fails "
    "at nominal'; then the far end, and if it works the search ends 'works across the interval'; otherwise the bracket "
    f"between the last value that works and the nearest that does not is halved {HALVINGS} times (to 1/"
    f"{2 ** HALVINGS} of the interval; an integer parameter is halved until the bracket is one step wide).  Every "
    "search runs twice, on success ('works', WORKS_RULE) and on survival alone ('survives'), from the same "
    "evaluations: a value evaluated for one is not run again for the other.  The bisection assumes one crossing in "
    "the interval; every value it evaluated is recorded with its outcome, and its result is the bracket it found: the "
    "last value that works (the threshold) and the nearest that does not")
VARIANT_RULE = (
    "the swept parameter is applied to the panels' plant of the condition after it is built and settled (the panels' "
    "settled stance is every episode's start, as env.py applies its randomised factors), for one episode, and restored "
    "exactly after it (the same model arrays bit for bit): friction (control.set_friction), servo kp (the actuators' "
    "gain and bias), trunk mass (env.set_trunk_mass), pile damping (the probes' solref), latency (a delayed servo "
    "command), the drives (the envelope and the drive ratings), each by its margin's rule; a push's velocity change "
    "replaces the panels' half bound in the pulse's force (PushedBuilt); a lift (LIFT_RULE) runs inside the variant "
    "(control.run enters control.relaxed after its reset and restores it before the variant is undone)")
LABEL_RULE = (
    "a threshold names what failed at its fails end (the nearest value that does not work, or does not survive): its "
    "success failures (episodes that survive but fail a term their seed passed at nominal, WORKS_RULE) by those new "
    "failed terms, classed as panels.SUCCESS_LABEL_RULE classes them (body-limited by the hardware terms that failed "
    "in more than half of those episodes, else controller-limited), and its falls by LIFT_RULE; the label is the "
    "success failures' when nothing falls, else that of the kind with more failed episodes, the falls' on a tie, as "
    "panels.FAILURE_LABEL_RULE breaks it (plan section 4: a hardware limit binds before most failures); a survival "
    "search takes the falls' label.  A search that fails at nominal also cites the panels' own reading of the row and "
    "condition (its 20-seed record: pass, counts, failure label with decision 20's attribution), when the record "
    "exists, and is flagged if that record passes")
LIFT_RULE = (
    "decision 20 attributes falls over 20 starts and its sign test needs at least "
    f"{panels.ATTRIBUTION_MIN_FALLS} falls (panels.ATTRIBUTION_MIN_FALLS), which 3 seeds cannot give; so at a fails "
    "end (and a point) whose episodes fall, the value is run again on the same 3 seeds once for each relaxable "
    "hardware limit that is a candidate of at least one of its falls (control.CANDIDATE_RULE: the limits active in the "
    "0.5 s before the onset; each lifted from the start by control.relaxed, control.RELAXES, as the panels' "
    "attribution lifts it), and its falls are body-limited by the lifts under which all 3 seeds survive, else "
    "controller-limited (lifting none of the candidates saves them); a candidate the plant cannot relax "
    "(cop_at_pad_edge_with_ankle_saturated) is named untestable, and falls with no candidate are controller-limited "
    f"with nothing to lift (decision 20).  Each lift is an evaluation of its own, keyed with its relaxation "
    f"(STAGE1_PLAN.md decision {DECISION})")
RECORD_RULE = (
    "one record per (body, row, margin): <out>/<body>/sweeps/<row>/<margin>.json (a row 'R2/43' is two directories) "
    "with each search's two results (status, the threshold and the nearest failing value, every value evaluated, the "
    "label at the fails end with its lifts, the panels' reading at 'fails at nominal'), the points' outcomes, every "
    "evaluation the margin used (its condition, variant, lift, seeds and each episode's outcome: the fall, its cause "
    "and candidates, the failed terms, the readings the margin turns on, wall and CPU seconds), Stage 0's values (plan "
    "section 5, STAGE0_RULE), the rules, the key, the panel sources and key context it was made with, the provenance "
    "and the cost; written after every round of evaluations, so an interrupted sweep keeps what it ran; "
    "<out>/<body>/sweeps/summary.json lists the sessions and every record's state")
RESUME_RULE = (
    "a margin whose record holds this run's key and is done is skipped; otherwise every evaluation that any record of "
    "the row holds under this run's evaluation key is reused: the sha256 of the panels' record key of its condition on "
    "the ledger's seeds (panels.record_key: the row and its sources, the condition and its plan, the inputs, the "
    "plant's round-two files, the library versions, the panels' sources, panels.RESUME_RULE) with its variant, its "
    "lift (LIFT_RULE) and the ledger's sources (SOURCES_RULE); evaluations that failed to run are run again.  A search "
    "replays its evaluations in order, so a resumed sweep ends where an uninterrupted one would.  The margin's key "
    "adds its definition (searches, intervals, points, seeds, halvings)")
SOURCES_RULE = ("the ledger's sources: the sha256 of designlab/stage1/ledger.py (its margins, variants and searches); "
                "the panels' sources are in each evaluation's panel key (panels.PANEL_SOURCES_RULE); every worker "
                "checks both against the session's, so a worker started after an edit refuses to run")
COST_RULE = (
    "the dry run counts, per search, the values it can still need after replaying the evaluations recorded under this "
    f"run's keys: at most 2 + {HALVINGS} on success (an integer search: 2 + its halvings) and as many more on survival "
    "(none when the two agree), each x 3 seeds at the condition's planned seconds (panels.Planner); a point once, with "
    "its condition's nominal evaluation (a value two searches share, such as the nominal walk's nominal value, is "
    "counted in each and run once); the lifts (LIFT_RULE) as an upper bound beside: every relaxation at each "
    "criterion's fails end; the cost per simulated second is panels.COST_RULE's for a tracker row (R2/42 walking on "
    "each floor, measured now) and a policy row's own for R3 and R4 (policy_rows.POLICY_COST_RULE: the row, its "
    "network included, on the nominal walk on each floor its searches still need, measured now); core-hours = "
    "simulated seconds x CPU seconds per simulated second / 3600")
GRIP_RULE = (
    "the lowest sliding friction that works: every colliding geom's sliding friction set to mu (control.set_friction, "
    f"as the panels set mu 0.54) on the nominal walk (walk/nominal), the 0.35 rad/s walking turn left and right "
    "(turn/walking_0.35_left and _right; the report gives the worse of the two, the higher mu) and the brisk walk "
    f"(walk/brisk), from mu {MU_NOMINAL:g} (the hard floor as compiled, every colliding geom at 0.9: the panels' "
    f"condition) down to {MU_FAR:g}; Stage 0: the friction the round-two turning screen requires in single support, "
    "per body walk / turn / brisk (STAGE0_RULE)")
PUSH_RULE = (
    "the largest push recovered: the panels' push conditions (standing and walking; forward, backward, left and right; "
    "the pulse's timing and the episode's length as panels.PUSH_RULE) with the pulse's velocity change dv in place of "
    "half the Stage 0 bound: force m dv / 0.1 s (m the robot's mass) along the direction in the home heading, from dv "
    f"0 (no push) up to {PUSH_FAR_BOUNDS:g} x the Stage 0 forward bound (env.PUSH_BOUND_M_S); works: back on the "
    "command within 3 s (decision 30, panels.RECOVERY_RULE) with the limits; Stage 0: the forward bound (round two's "
    "one-step capturability bound; it has no other direction)")
STOP_RULE = (
    f"not run, a deviation from plan section 5 (STAGE1_PLAN.md decision {DECISION}, for the owner): the deceleration a "
    "stop holds needs stop references that decelerate at other rates; the run holds one stop transition from the walk "
    "to the stand (transitions.TRANSITIONS 'stop', its speed ramp fixed by transitions.PATH_HALF_WIDTH; the in-place "
    "turns' stops end a turn), and another stop would be a new reference, not this plan's; sweeping the stop "
    "transition's own clock rate instead would change its stepping as well as its deceleration and needs a reference "
    "clock whose rate changes by segment, which control.Track does not have.  Recorded beside "
    "it: the deceleration the existing stop asks for (its reference CoM's forward acceleration on the commanded clock, "
    "low-passed at 1 Hz as decision 10 reads acceleration, metrics.lowpass) and, when the row's start-and-stop panel "
    "record exists, what its episodes read (each stop's acceleration peak and whether it stopped within 1.5 strides, "
    "panels.START_STOP_RULE)")
DRIVE_SPEED_RULE = (
    "the smallest scale s on every drive's speed limits at which the Fr 0.05 walk (walk/fr05) works: each drive's "
    "corner and no-load speeds x s in the torque-speed envelope (drive.Envelope's speed_scale, as plant.build("
    "speed_scale=s) builds it), and with them the success limit on peak speed (panels.LIMITS_RULE reads the envelope's "
    f"corner speed: control.run's worst_speed_ratio), from 1 down to {SCALE_FAR:g}; peak torques unchanged; Stage 0: "
    "the knee's speed need at Fr 0.05 over its rated speed (plan section 5), including the screens' 15 % speed margin "
    "(0.93 / 0.95 / 0.93); without it 0.81 / 0.83 / 0.81, which is what this limit compares (STAGE0_RULE)")
DRIVE_TORQUE_RULE = (
    "the smallest scale s on every drive's torque at which the nominal walk (walk/nominal) works: each drive's peak "
    "torque x s in the envelope (drive.Envelope's torque_scale, as plant.build(torque_scale=s)) and in the drive "
    "ratings the controller and the judgement read (plant.Plant.drives: the tracker's ankle cap, decision 15, at the "
    "scaled peak; the success limit on each drive's RMS torque at its sustained torque x s, panels.LIMITS_RULE), the "
    "servo gains unchanged (kp = the nominal sustained torque / 0.01 rad: the kp margin sweeps them), from 1 down to "
    f"{SCALE_FAR:g}; Stage 0: the worst drive's sustained-torque ratio, the larger of the nominal walk's RMS and the "
    "one-foot hold (0.76 / 0.76 / 0.77; on 75-M-H-10 the hip yaw's one-foot hold), with the nominal walk's own worst "
    "RMS ratio beside (0.76 / 0.76 / 0.66), which is what this limit compares (STAGE0_RULE)")
KP_RULE = (
    "the lowest servo stiffness factor f at which the nominal walk works: every drive's servo kp (the actuator's gain "
    "and its proportional bias) x f in the plant, kv unchanged, the controller keeping the nominal gains (env.KP_RULE: "
    "its servo targets carry the feedforward as tau / kp_nominal, so the plant delivers f x the feedforward torque), "
    f"from 1 down to {KP_FAR:g} (plan section 5's kp x 0.25-1)")
LATENCY_RULE = (
    "the largest actuation latency at which the nominal walk works: every servo target reaches the servos k physics "
    "steps (2 ms each) after the controller sets it (env.py's latency model, Stage1Env._latency_on: a queue of the "
    "last k targets, filled with the reset's targets at the episode's start), k whole steps from 0 to "
    f"{LATENCY_MAX_STEPS} (0-{2 * LATENCY_MAX_STEPS} ms, plan section 5); an integer search, halved until the "
    "bracket is one step (2 ms) wide")
MASS_RULE = (
    "the trunk's mass and inertia x f in the plant (env.set_trunk_mass, env.MASS_RULE: MuJoCo's derived constants "
    "recomputed), the controller keeping the nominal model (every control.Tracker the row builds during the episode "
    "gets the plant's nominal model when its caller gives none, as env.ResidualEnv builds R3's: the tracker's mass, "
    "R2's estimator and its sensors' weight, and a policy row's observed weight and R4's estimator, which are its "
    "tracker's, policy_rows.POLICY_RULE and R4_RULE, as env.py keeps them; each episode records the trunk mass its "
    "controller worked with and the plant's, and with them the whole robot's mass change), the judgement on the true "
    "body (metrics.Recorder reads the plant's mass: the GRF over the true weight), on the nominal walk.  Plan section "
    f"5's '+10 %' is read both ways, as two points: the trunk x {MASS_POINT:g} (env.MASS_RULE's factor, the trunk "
    "alone: +4.15-4.40 % of the robot's mass) and the trunk factor that adds 10 % of the robot's mass (MASSES, "
    "ROBOT_MASS_POINT: x 1.230 / 1.227 / 1.241); and the largest f that works from 1 up to "
    f"{MASS_FAR:g} (+12-13 % of the robot: beyond the plan's +10 %, to find the margin; STAGE1_PLAN.md decision "
    f"{DECISION})")
PILE_DAMPING_RULE = (
    "every pile probe's contact damping x f with its stiffness unchanged: each probe's solref (time constant tc, "
    "damping ratio z) becomes (tc / f, z f), so MuJoCo's reference-acceleration damping B = 2 / (dmax tc) is x f and "
    "its stiffness K = 1 / (dmax^2 tc^2 z^2) is unchanged (tc / f must stay above MuJoCo's 2-step floor, checked); "
    f"from 1 down to {DAMPING_LOW:g} and from 1 up to {DAMPING_HIGH:g} (plan section 5; carpet damping is not "
    "sourced, A16), on low pile (plan section 4's evaluation pile): the 20 s stand (stand/low), which R2 passes on "
    "every body, and the carpet walk at mu 0.54 (carpet/low), which decision 23 found R2 falling on in every start "
    "tried, so R2's carpet searches end 'fails at nominal' at the cost of one evaluation while its stand searches "
    "measure the damping margin; both for every row that runs the pile, R3's carpet walk among them (STAGE1_PLAN.md "
    f"decision {DECISION})")
VOLTAGE_RULE = (
    "the pack's bus voltage V in place of the record's 24 V for every leg drive, as plant.build(bus=V) builds it "
    "(drive.drives(bus=V)): each leg drive's corner speed is Actuator.speed_at(V) = its rated speed x min(1, V / its "
    "nominal voltage) and its no-load speed is derated alike, with no credit above the nominal voltage; peak and "
    "sustained torques unchanged; the neck and tail box servos are on their own supply and unchanged; the envelope's "
    "corner speed is also the success limit on peak speed (panels.LIMITS_RULE, through control.run's "
    "worst_speed_ratio); the nominal walk and the Fr 0.05 walk at 25.2 V (full), 22.2 V (nominal) and 21 V (near "
    "empty) on the 3 seeds; a voltage at which no drive's speeds change (above every leg drive's nominal voltage: "
    "25.2 V on the 60 cm bodies' 24 V drives) is the nominal plant and takes its evaluation")
MARGIN_RULES = dict(grip=GRIP_RULE, push=PUSH_RULE, stop=STOP_RULE, drive_speed=DRIVE_SPEED_RULE,
                    drive_torque=DRIVE_TORQUE_RULE, kp=KP_RULE, latency=LATENCY_RULE, mass=MASS_RULE,
                    pile_damping=PILE_DAMPING_RULE, voltage=VOLTAGE_RULE)
RULES = dict(rows=ROWS_RULE, seeds=SEEDS_RULE, works=WORKS_RULE, search=SEARCH_RULE, variant=VARIANT_RULE,
             label=LABEL_RULE, lift=LIFT_RULE, record=RECORD_RULE, resume=RESUME_RULE, sources=SOURCES_RULE,
             cost=COST_RULE, stage0=STAGE0_RULE, success=tune.SUCCESS_RULE, limits=panels.LIMITS_RULE,
             recovery=panels.RECOVERY_RULE, candidates=control.CANDIDATE_RULE,
             failed_episode=panels.FAILED_EPISODE_RULE)
MODEL_DOUBTS = ("kp", "latency", "mass", "pile_damping")


# ------------------------------------------------------------------------------------------------
# Margins: searches and points
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """A change to one episode's plant or push (``VARIANT_RULE``): the parameter's kind (``KINDS``) and value; no kind
    is the panels' plant as it is."""

    kind: str | None = None
    value: float | int | None = None

    def __post_init__(self):
        if self.kind is not None and self.kind not in KINDS:
            raise ValueError(f"variant {self.kind!r}: choose from {', '.join(KINDS)}")

    @property
    def key(self) -> str:
        return "nominal" if self.kind is None else f"{self.kind}={self.value!r}"

    def describe(self) -> dict:
        return dict(kind=self.kind, value=self.value)


NOMINAL = Variant()


@dataclass(frozen=True)
class Search:
    """One bisection (``SEARCH_RULE``): parameter ``kind`` on the panel condition ``condition`` from ``nominal`` toward
    ``far`` (an ``integer`` parameter takes whole values)."""

    name: str
    condition: str
    kind: str
    nominal: float
    far: float
    integer: bool = False

    @property
    def seek(self) -> str:
        return "lowest" if self.far < self.nominal else "largest"

    def variant(self, value) -> Variant:
        """The variant of a value: the nominal value of a plant parameter is the panels' plant as it is (a push's
        nominal, dv 0, is still a pulse: a zero one)."""
        if self.kind != "push_dv" and value == self.nominal:
            return NOMINAL
        return Variant(self.kind, value)

    def midpoint(self, a, b):
        if not self.integer:
            return (a + b) / 2
        a, b = int(a), int(b)
        return a + (b - a) // 2 if b > a else a - (a - b) // 2

    def max_values(self, halvings: int = HALVINGS) -> int:
        """The most values one criterion's bisection evaluates (``SEARCH_RULE``)."""
        if not self.integer:
            return 2 + int(halvings)
        span = abs(int(self.far) - int(self.nominal))
        return 2 + (math.ceil(math.log2(span)) if span > 1 else 0)

    def describe(self) -> dict:
        name, unit = PARAMETERS[self.kind]
        return dict(name=self.name, condition=self.condition, kind=self.kind, parameter=name, unit=unit,
                    nominal=self.nominal, far=self.far, seek=self.seek, integer=self.integer)


@dataclass(frozen=True)
class Point:
    """One value of a parameter on a panel condition, evaluated once (works or not, survives or not)."""

    name: str
    condition: str
    kind: str
    value: float

    def describe(self) -> dict:
        name, unit = PARAMETERS[self.kind]
        return dict(name=self.name, condition=self.condition, kind=self.kind, parameter=name, unit=unit,
                    value=self.value)


@dataclass(frozen=True)
class Margin:
    """One row of plan section 5's ledger: its searches and points, the searches whose worse result is reported
    together (``worse``: name -> searches), or why it is not run."""

    name: str
    title: str
    swept: str
    searches: tuple = ()
    points: tuple = ()
    worse: tuple = ()
    not_run: str | None = None

    @property
    def rule(self) -> str:
        return MARGIN_RULES[self.name]

    def conditions(self) -> list[str]:
        return sorted({s.condition for s in self.searches} | {p.condition for p in self.points})

    def describe(self) -> dict:
        return dict(name=self.name, title=self.title, swept=self.swept, searches=[s.describe() for s in self.searches],
                    points=[p.describe() for p in self.points], worse=[[n, list(s)] for n, s in self.worse],
                    not_run=self.not_run)


def margins(body: str) -> list[Margin]:
    """Plan section 5's ledger for ``body`` (``MARGIN_RULES``), in the plan's order."""
    if body not in BODIES:
        raise KeyError(body)
    walk, fr05 = "walk/nominal", "walk/fr05"
    push_far = PUSH_FAR_BOUNDS * float(env.PUSH_BOUND_M_S[body])
    S = Search
    return [
        Margin("grip", "Grip", "the lowest sliding friction mu that works", searches=(
            S("walk_nominal", walk, "friction", MU_NOMINAL, MU_FAR),
            S("turn_left", "turn/walking_0.35_left", "friction", MU_NOMINAL, MU_FAR),
            S("turn_right", "turn/walking_0.35_right", "friction", MU_NOMINAL, MU_FAR),
            S("walk_brisk", "walk/brisk", "friction", MU_NOMINAL, MU_FAR)),
               worse=(("turn", ("turn_left", "turn_right")),)),
        Margin("push", "Push", "the largest velocity change recovered, per direction, standing and walking",
               searches=tuple(S(f"{mode}_{d}", f"push/{mode}_{d}", "push_dv", 0.0, push_far)
                              for mode in ("standing", "walking") for d in panels.PUSH_DIRECTIONS)),
        Margin("stop", "Stop", "the deceleration held", not_run=(
            f"not run (plan deviation, decision {DECISION}): the deceleration a stop holds needs stop references that "
            "decelerate at other rates, and the run holds one stop transition (STOP_RULE)")),
        Margin("drive_speed", "Drive speed", "the smallest envelope speed scale at which the Fr 0.05 walk works",
               searches=(S("walk_fr05", fr05, "speed_scale", 1.0, SCALE_FAR),)),
        Margin("drive_torque", "Drive torque", "the smallest torque scale at which the nominal walk works",
               searches=(S("walk_nominal", walk, "torque_scale", 1.0, SCALE_FAR),)),
        Margin("kp", "Model doubt: servo kp", "the lowest servo kp factor at which the nominal walk works",
               searches=(S("walk_nominal", walk, "kp", 1.0, KP_FAR),)),
        Margin("latency", "Model doubt: latency", "the largest latency at which the nominal walk works",
               searches=(S("walk_nominal", walk, "latency_steps", 0, LATENCY_MAX_STEPS, integer=True),)),
        Margin("mass", "Model doubt: trunk mass", "+10 % (the trunk x 1.1, and the trunk factor that adds 10 % of the "
               "robot's mass), and the largest trunk mass factor at which the nominal walk works",
               searches=(S("walk_nominal", walk, "mass", 1.0, MASS_FAR),),
               points=(Point("walk_nominal_trunk_x1.1", walk, "mass", MASS_POINT),
                       Point("walk_nominal_robot_x1.1", walk, "mass", ROBOT_MASS_POINT[body]))),
        Margin("pile_damping", "Model doubt: pile damping", "the lowest and largest pile damping factor at which the "
               "stand and the carpet walk on low pile work",
               searches=(S("stand_low_down", "stand/low", "pile_damping", 1.0, DAMPING_LOW),
                         S("stand_low_up", "stand/low", "pile_damping", 1.0, DAMPING_HIGH),
                         S("carpet_low_down", "carpet/low", "pile_damping", 1.0, DAMPING_LOW),
                         S("carpet_low_up", "carpet/low", "pile_damping", 1.0, DAMPING_HIGH))),
        Margin("voltage", "Pack voltage", "the nominal and Fr 0.05 walks at 25.2, 22.2 and 21 V",
               points=tuple(Point(f"{n}_{v:g}V", key, "bus_v", v) for n, key in (("walk_nominal", walk),
                                                                                  ("walk_fr05", fr05))
                            for v in VOLTAGES)),
    ]


MARGIN_NAMES = ("grip", "push", "stop", "drive_speed", "drive_torque", "kp", "latency", "mass", "pile_damping",
                "voltage")


def select_margins(ms: list[Margin], names=None) -> list[Margin]:
    """The margins named (``model_doubts``: kp, latency, mass and pile damping), in the plan's order."""
    if names is None:
        return list(ms)
    want = set()
    for n in names:
        want.update(MODEL_DOUBTS if n == "model_doubts" else (n,))
    unknown = want - {m.name for m in ms}
    if unknown:
        raise ValueError(f"margins {sorted(unknown)}: choose from {', '.join(MARGIN_NAMES)} or model_doubts")
    return [m for m in ms if m.name in want]


@functools.lru_cache(maxsize=None)
def screens(body: str) -> dict:
    """``STAGE0_RULE``: the round-two screens' unrounded Stage 0 values of a body (the turning screen's friction need,
    the candidate record's knee speed ratio and drive torque ratios), with the files' sha256."""
    cd = case_dir(body)
    tpath, cpath = cd / "review" / "screening" / f"{body}_turning.json", cd / "candidates" / f"{body}.json"
    turning, cand = json.loads(tpath.read_text()), json.loads(cpath.read_text())
    grip = {k: max(float(turning["runs"][run]["feet"][f]["single_support"]["mu_req_max"]) for f in ("left", "right"))
            for k, run in TURNING_RUNS.items()}
    grip["turn"] = max(grip["turn_left"], grip["turn_right"])
    m = cand["margins"]
    worst = max(m, key=lambda g: (m[g]["cont_ratio"], g))
    worst_rms = max(m, key=lambda g: (m[g]["rms"] / m[g]["sustained"], g))
    knee = float(m["knee"]["speed_ratio"])
    return dict(grip=grip, grip_runs=dict(TURNING_RUNS),
                drive_speed=dict(knee_speed_ratio=knee, knee_peak_over_rated=knee / sizing.SPEED_MARGIN,
                                 speed_margin=sizing.SPEED_MARGIN),
                drive_torque=dict(drive=worst, cont_ratio=float(m[worst]["cont_ratio"]),
                                  by=("one-foot hold" if m[worst]["hold"] >= m[worst]["rms"] else "nominal walk RMS"),
                                  walk_rms_drive=worst_rms,
                                  walk_rms_ratio=float(m[worst_rms]["rms"] / m[worst_rms]["sustained"])),
                files={str(p.relative_to(STUDY)): plant.sha256(p) for p in (tpath, cpath)}, rule=STAGE0_RULE)


def stage0(body: str, margin: str) -> dict:
    """Plan section 5's Stage 0 value(s) of a margin, with what they are (``STAGE0_RULE``) and the screens' unrounded
    values (``screens``)."""
    v = STAGE0[body]
    if margin == "grip":
        sc = screens(body)
        return dict(values=dict(walk_nominal=v["grip_walk_nominal"], turn=v["grip_turn"], turn_left=v["grip_turn"],
                                turn_right=v["grip_turn"], walk_brisk=v["grip_walk_brisk"]),
                    screens=dict(sc["grip"], runs=sc["grip_runs"], files=sc["files"]),
                    text="the lowest friction the round-two turning screen requires in single support, per body: "
                         "nominal walk / 0.35 rad/s walking turn (the worse direction) / brisk walk (STAGE0_RULE)")
    if margin == "push":
        return dict(values={f"{m}_forward": v["push_forward"] for m in ("standing", "walking")},
                    forward_bound_m_s=v["push_forward"],
                    text="the forward push bound (m/s; round two's one-step capturability bound, forward only)")
    if margin == "stop":
        return dict(values=dict(one_foot_m_s2=v["stop_one_foot"]), text="the deceleration held on one foot (m/s^2)")
    if margin == "drive_speed":
        sc = screens(body)
        return dict(values=dict(walk_fr05=v["drive_speed_knee"]),
                    without_margin=dict(walk_fr05=v["drive_speed_knee_no_margin"]),
                    screens=dict(sc["drive_speed"], files=sc["files"]),
                    text="the knee's speed need at Fr 0.05 over its rated speed, including the screens' 15 % speed "
                         f"margin; without it {v['drive_speed_knee_no_margin']:g} (0.81 / 0.83 / 0.81), which is what "
                         "the ledger's limit compares (STAGE0_RULE)")
    if margin == "drive_torque":
        sc = screens(body)["drive_torque"]
        return dict(values=dict(walk_nominal=v["drive_torque_sustained"]),
                    walk_rms=dict(walk_nominal=v["drive_torque_walk_rms"]),
                    screens=dict(sc, files=screens(body)["files"]),
                    text=f"the worst drive's sustained-torque ratio, the larger of the nominal walk's RMS and the "
                         f"one-foot hold ({sc['drive']}'s {sc['by']}); the nominal walk's RMS alone "
                         f"{v['drive_torque_walk_rms']:g} ({sc['walk_rms_drive']}), which is what the ledger's limit "
                         "compares (STAGE0_RULE)")
    if margin == "voltage":
        return dict(values={}, text=STAGE0_VOLTAGE)
    return dict(values={}, text="no Stage 0 value (a model doubt)")


# ------------------------------------------------------------------------------------------------
# The bisection
# ------------------------------------------------------------------------------------------------
class Bisect:
    """One search on one criterion (``SEARCH_RULE``): ``need()`` is the value it must know next (None once it is done)
    and ``feed(value, ok)`` gives that value's outcome (``ok``: works, or survives)."""

    def __init__(self, search: Search, criterion: str, halvings: int = HALVINGS):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion {criterion!r}: choose from {', '.join(CRITERIA)}")
        self.search, self.criterion, self.halvings = search, criterion, int(halvings)
        self.works = self.fails = None
        self.status: str | None = None
        self.trail: list[tuple] = []  # (value, ok) in the order evaluated
        self.halved = 0

    @property
    def done(self) -> bool:
        return self.status is not None

    def need(self):
        if self.status is not None:
            return None
        s = self.search
        if not self.trail:
            return s.nominal
        if len(self.trail) == 1:
            return s.far
        return s.midpoint(self.works, self.fails)

    def feed(self, value, ok: bool) -> None:
        want = self.need()
        if want is None or value != want:
            raise ValueError(f"{self.search.name} ({self.criterion}) needs {want!r}, not {value!r}")
        ok = bool(ok)
        self.trail.append((value, ok))
        if len(self.trail) == 1:
            if ok:
                self.works = value
            else:
                self.fails, self.status = value, "fails at nominal"
            return
        if len(self.trail) == 2:
            if ok:
                self.works, self.status = value, "works across the interval"
            else:
                self.fails = value
                self._close()
            return
        self.halved += 1
        if ok:
            self.works = value
        else:
            self.fails = value
        self._close()

    def _close(self) -> None:
        if self.search.integer:
            if abs(int(self.fails) - int(self.works)) <= 1:
                self.status = "bisected"
        elif self.halved >= self.halvings:
            self.status = "bisected"

    def result(self) -> dict:
        out = dict(criterion=self.criterion, status=self.status or "incomplete", threshold=self.works,
                   works=self.works, fails=self.fails, halvings=self.halved,
                   values=[dict(value=v, ok=ok) for v, ok in self.trail])
        if self.status == "bisected":
            out["resolution"] = abs(self.fails - self.works)
        return out


def standing(res: dict, search: Search) -> float | None:
    """How far a result reached toward the far end, as a fraction of the interval: -1 fails at nominal, 1 works across
    it, None incomplete (``worse``)."""
    st = res.get("status")
    if st == "fails at nominal":
        return -1.0
    if st == "works across the interval":
        return 1.0
    if st == "bisected":
        return (res["works"] - search.nominal) / (search.far - search.nominal)
    return None


def worse(results: dict, searches: list[Search]) -> dict:
    """Of searches' results on one criterion, the one with the least margin (the report's worse turn direction)."""
    rank = {s.name: standing(results[s.name], s) for s in searches}
    if any(v is None for v in rank.values()):
        return dict(status="incomplete", of=[s.name for s in searches])
    name = min(rank, key=lambda n: (rank[n], n))
    return dict(results[name], search=name, of=[s.name for s in searches])


# ------------------------------------------------------------------------------------------------
# Variants (worker side)
# ------------------------------------------------------------------------------------------------
@dataclass
class PushedBuilt(panels.Built):
    """A push condition built by the panels with the pulse's velocity change ``dv_m_s`` (``PUSH_RULE``): force
    m dv / 0.1 s along the direction in the home heading."""

    dv_m_s: float = 0.0

    def push_force(self) -> np.ndarray | None:
        if self.resolved.get("push") is None:
            return None
        F = self.mass * self.dv_m_s / panels.PUSH_S
        dx, dy = panels.PUSH_DIRECTIONS[self.cond.direction]
        c, s = math.cos(self.home_yaw), math.sin(self.home_yaw)
        return F * np.array([c * dx - s * dy, s * dx + c * dy])


@contextmanager
def controller_model(model: mujoco.MjModel):
    """Every ``control.Tracker`` built inside (a subclass's too: its ``__init__`` is wrapped, not the class replaced)
    gets ``model`` as its own model when its caller gives none (``MASS_RULE``); yields the list of the trackers built,
    each (whether its caller gave a model, the trunk mass its controller works with)."""
    base_init = control.Tracker.__dict__["__init__"]
    sig = inspect.signature(base_init)
    made: list[tuple] = []

    @functools.wraps(base_init)
    def init(*args, **kw):
        ba = sig.bind(*args, **kw)
        given = ba.arguments.get("model") is not None
        if not given:
            ba.arguments["model"] = model
        base_init(*ba.args, **ba.kwargs)
        made.append((given, float(ba.arguments["self"].mass)))

    control.Tracker.__init__ = init
    try:
        yield made
    finally:
        control.Tracker.__init__ = base_init


def _probes(m: mujoco.MjModel) -> np.ndarray:
    """The pile probes of a plant's model (``carpet.add_probes``: spheres of contype ``carpet.PROBE_BIT``, conaffinity
    0), as env.foot_tables tells them apart."""
    return np.flatnonzero((m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) & (m.geom_contype == carpet.PROBE_BIT)
                          & (m.geom_conaffinity == 0))


def _latency(p: plant.Plant, steps: int):
    """``LATENCY_RULE`` on ``p`` until the returned undo is called: every command reaches the servos ``steps`` physics
    steps late (env.Stage1Env._latency_on's queue), the queue refilled with the reset's command at every reset."""
    command, reset = type(p).command.__get__(p), type(p).reset.__get__(p)
    queue = deque(maxlen=steps + 1)

    def fill():
        queue.clear()
        queue.extend(p.d.ctrl.copy() for _ in range(steps))

    def delayed_reset(*args, **kw):
        reset(*args, **kw)
        fill()

    def delayed_command(*args, **kw):
        command(*args, **kw)
        ctrl = p._v.ctrl
        queue.append(ctrl.copy())
        ctrl[:] = queue[0]

    fill()
    p.reset, p.command = delayed_reset, delayed_command

    def undo():
        p.__dict__.pop("reset", None)
        p.__dict__.pop("command", None)

    return undo


class LedgerContext(panels.Context):
    """A worker's ``panels.Context`` with the ledger's variants: ``variant(cond, v)`` applies ``v`` to the condition's
    plant (built and settled by the panels, before any change) for one episode and restores it after
    (``VARIANT_RULE``); while a push variant is applied, ``built`` gives the condition with its pulse
    (``PushedBuilt``)."""

    def __init__(self, body: str, run_dir: Path, docs: dict):
        super().__init__(body, run_dir, docs)
        self.current = NOMINAL
        self._nominal_models: dict[int, mujoco.MjModel] = {}
        self._selection = None

    @classmethod
    def of(cls, ctx: panels.Context) -> "LedgerContext":
        """A ledger context on the same body and inputs as a panels context."""
        return cls(ctx.body, ctx.run_dir, {"refs.json": dict(references=ctx.refs_doc),
                                           "transitions.json": dict(transitions=ctx.tr_doc)})

    def built(self, cond: panels.Condition) -> panels.Built:
        b = super().built(cond)
        v = self.current
        if v.kind != "push_dv":
            return b
        if b.resolved.get("push") is None:
            raise ValueError(f"{cond.key} has no push to change")
        fields = {f.name: getattr(b, f.name) for f in dataclasses.fields(panels.Built)}
        return PushedBuilt(**fields, dv_m_s=float(v.value))

    def nominal_model(self, p: plant.Plant) -> mujoco.MjModel:
        """A copy of ``p``'s model as built (taken while no variant is applied); refused if mj_setConst would change
        the model as compiled (the mass variant could not restore it, env.MASS_RULE)."""
        if id(p) not in self._nominal_models:
            changes = env.setconst_changes(p.m)
            if changes:
                raise plant.PlantError(f"mj_setConst changes the {p.body} plant on {p.floor} as built "
                                       f"({', '.join(changes)}): the mass variant could not restore it")
            self._nominal_models[id(p)] = copy.deepcopy(p.m)
        return self._nominal_models[id(p)]

    def _drives(self, p: plant.Plant, v: Variant):
        """Swap ``p``'s envelope and drive ratings for the drive variants (``DRIVE_SPEED_RULE``, ``DRIVE_TORQUE_RULE``,
        ``VOLTAGE_RULE``); returns the undo."""
        old_env, old_drives = p.envelope, p.drives
        speed = float(v.value) if v.kind == "speed_scale" else 1.0
        torque = float(v.value) if v.kind == "torque_scale" else 1.0
        drives = list(old_drives)
        if v.kind == "bus_v":
            if self._selection is None:
                self._selection = plant.record(p.body)["selection"]
            drives = drive.drives(self._selection, p.joints, bus=float(v.value), layout=p.meta["leg_layout"])
        new_env = drive.Envelope(drives, old_env.moment, enabled=old_env.enabled, speed_scale=speed,
                                 torque_scale=torque, braking=old_env.braking)
        if v.kind == "torque_scale":
            drives = [dataclasses.replace(d, peak=d.peak * torque, sustained=d.sustained * torque) for d in drives]
        p.envelope, p.drives = new_env, drives
        new_env.bind(p.m, p.d)

        def undo():
            p.envelope, p.drives = old_env, old_drives
            old_env.bind(p.m, p.d)

        return undo

    def _apply(self, p: plant.Plant, v: Variant):
        """Apply ``v`` to ``p``; returns the undo that restores it exactly."""
        m = p.m
        if v.kind is None or v.kind == "push_dv":
            return lambda: None
        if v.kind == "friction":
            saved = m.geom_friction.copy()
            control.set_friction(p, float(v.value))

            def undo():
                m.geom_friction[:] = saved

            return undo
        if v.kind == "kp":
            g0, b0 = m.actuator_gainprm[:, 0].copy(), m.actuator_biasprm[:, 1].copy()
            m.actuator_gainprm[:, 0] = g0 * float(v.value)
            m.actuator_biasprm[:, 1] = b0 * float(v.value)

            def undo():
                m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = g0, b0

            return undo
        if v.kind == "latency_steps":
            k = int(v.value)
            if k != v.value or k < 0:
                raise ValueError(f"latency {v.value!r}: whole physics steps, at least 0")
            return _latency(p, k) if k else (lambda: None)
        if v.kind == "mass":
            nominal = self.nominal_model(p)
            scratch = mujoco.MjData(m)
            env.set_trunk_mass(m, p.trunk, float(v.value), nominal, scratch)

            def undo():
                env.set_trunk_mass(m, p.trunk, 1.0, nominal, scratch)
                left = [k for k in env.model_differences(m, nominal) if k != "actuator_forcerange"]
                if left:
                    raise RuntimeError(f"the trunk mass did not restore the plant exactly: {', '.join(left)}")

            return undo
        if v.kind == "pile_damping":
            probes = _probes(m)
            if not len(probes):
                raise ValueError(f"{p.floor}: no pile probes (the pile damping applies to a pile floor)")
            saved = m.geom_solref[probes].copy()
            if (saved[:, 0] <= 0.0).any():
                raise ValueError("a pile probe has a direct (negative) solref: PILE_DAMPING_RULE scales the "
                                 "time-constant form only")
            f = float(v.value)
            tc = saved[:, 0] / f
            refsafe = not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))
            if refsafe and (tc < 2.0 * m.opt.timestep).any():
                raise ValueError(f"pile damping x {f:g}: a probe's time constant {tc.min():.4g} s falls below MuJoCo's "
                                 f"2-step floor ({2.0 * m.opt.timestep:g} s): its stiffness would change")
            m.geom_solref[probes, 0] = tc
            m.geom_solref[probes, 1] = saved[:, 1] * f

            def undo():
                m.geom_solref[probes] = saved

            return undo
        if v.kind in ("speed_scale", "torque_scale", "bus_v"):
            return self._drives(p, v)
        raise ValueError(f"variant {v.kind!r}")

    @contextmanager
    def variant(self, cond: panels.Condition, v: Variant):
        """``v`` applied to ``cond``'s plant for the block (``VARIANT_RULE``); yields a dict the block's episode
        record takes (the mass variant: whether the row's tracker took the nominal model)."""
        b = panels.Context.built(self, cond)  # built (and cached) on the plant as it is, before any change
        p = b.p
        nominal = self.nominal_model(p) if v.kind == "mass" else None
        info: dict = {}
        undo = self._apply(p, v)
        self.current = v
        try:
            if nominal is not None:
                with controller_model(nominal) as made:
                    yield info
                m0, m1 = float(nominal.body_subtreemass[p.trunk]), float(p.m.body_subtreemass[p.trunk])
                info["controller_model"] = dict(
                    nominal=bool(made) and all(mass == m0 for _, mass in made), trackers=len(made),
                    given_by_row=sum(1 for g, _ in made if g), trunk_mass_kg=[mass for _, mass in made],
                    nominal_trunk_mass_kg=m0, plant_trunk_mass_kg=m1,
                    trunk_body_kg=float(nominal.body_mass[p.trunk]), plant_trunk_body_kg=float(p.m.body_mass[p.trunk]),
                    robot_mass_change=m1 / m0 - 1.0)  # (the trunk is the root: its subtree is the whole robot)
            else:
                yield info
        finally:
            self.current = NOMINAL
            try:
                undo()
            except Exception:
                self.evict(p)  # never run another episode on a plant that may not be restored
                raise

    def evict(self, p: plant.Plant) -> None:
        """Forget ``p`` and every condition built on it: the next episode builds them afresh."""
        self._plants = {k: v for k, v in self._plants.items() if v is not p}
        self._built = {k: v for k, v in self._built.items() if v.p is not p}
        self._nominal_models.pop(id(p), None)


EPISODE_KEYS = ("seed", "fell", "fall_time_s", "fall_cause", "fall_onset_s", "fall_candidates", "survival_s",
                "planned_s", "succeeded", "failed_terms", "worst_sustained_ratio", "worst_speed_ratio",
                "joint_shortfall_rad", "joint_worst", "captures")
WINDOW_READINGS = ("complete", "speed_m_s", "yaw_rate_rad_s", "duty", "flight_s", "slip_median_m", "slip_worst_m",
                   "grf_peak_bw", "drive_sustained_worst")


def episode_summary(e: dict, cpu_s: float | None = None, extra: dict | None = None) -> dict:
    """The part of a panels episode record the ledger keeps (``RECORD_RULE``)."""
    out = {k: e.get(k) for k in EPISODE_KEYS}
    ms = e.get("measures") or {}
    w = ms.get("window")
    if w:
        out["window"] = {k: w.get(k) for k in WINDOW_READINGS}
    if ms.get("push"):
        out["push"] = {k: ms["push"].get(k) for k in ("dv_m_s", "onset_s", "direction")}
    if ms.get("recovery"):
        out["recovery"] = {k: ms["recovery"].get(k) for k in ("recovered", "after_s", "never_left_command")}
    if ms.get("carpet"):
        out["carpet"] = {k: ms["carpet"].get(k) for k in ("toe_clearance_m", "target_m", "catching_swings")}
    if e.get("mujoco_warnings"):
        out["mujoco_warnings"] = e["mujoco_warnings"]
    if e.get("error"):
        out["error"] = str(e["error"])[-2000:]
    out.update(wall_s=e.get("wall_s"), cpu_s=cpu_s, **(extra or {}))
    return out


class LedgerWorker(panels.PanelWorker):
    """A worker for ``tune.Pool``: ``evaluate((row, condition, kind, value[, relax]), seed)`` runs one episode of a row
    on a condition with a variant (``LedgerContext``) and the hardware limits ``relax`` lifted (``LIFT_RULE``) and
    returns its summary.  It refuses to start on other panel sources, key context or ledger sources than the session's
    (``SOURCES_RULE``)."""

    def __init__(self, body: str, run_dir: str, mode: str = "ledger", rate: float = panels.RATE,
                 expected_inputs: dict | None = None, rows=(), sources: str | None = None, ledger: str | None = None):
        if ledger is not None and ledger_sources()["sha256"] != ledger:
            raise tune.PoolError("the ledger's sources changed during the session: restore them and sweep again")
        super().__init__(body, run_dir, mode, rate, expected_inputs, rows, sources, traces=None)
        self.ctx = LedgerContext.of(self.ctx)

    def evaluate(self, job, seed) -> dict:
        row, cond, kind, value, *rest = job
        relax = tuple(rest[0]) if rest else ()
        c0 = time.process_time()
        info: dict = {}
        try:
            with self.ctx.variant(cond, Variant(kind, value)) as info:
                e = self.rows[row].run(self.ctx, cond, int(seed), relax, trace_root=None)
        except Exception:
            e = panels.failed_episode(cond, int(seed), traceback.format_exc(), relax)
        return episode_summary(e, time.process_time() - c0, dict(info, relaxed=list(relax)) if relax else info)


def _worker(body, run_dir, mode, rate, expected_inputs=None, rows=(), sources=None, ledger=None):
    return LedgerWorker(body, run_dir, mode, rate, expected_inputs, rows, sources, ledger)


def _lost(job, reason: str) -> dict:
    (row, cond, kind, value, *rest), seed = job
    relax = tuple(rest[0]) if rest else ()
    return episode_summary(panels.failed_episode(cond, int(seed), f"worker lost: {reason}", relax), None)


# ------------------------------------------------------------------------------------------------
# Evaluations, labels, keys
# ------------------------------------------------------------------------------------------------
def ledger_sources() -> dict:
    """``SOURCES_RULE``."""
    path = Path(__file__).resolve()
    return dict(sha256=plant.sha256(path), file=path.name, rule=SOURCES_RULE)


def _sha(x) -> str:
    return hashlib.sha256(json.dumps(panels._jsonable(x), sort_keys=True).encode()).hexdigest()


def evaluation_entry(cond_key: str, var: Variant, episodes: list[dict], valid: str, relax=()) -> dict:
    """One evaluation (a condition with a variant, and any lift, on the ledger's seeds): its episodes and outright
    outcomes (``succeeds_all``: every episode succeeds; ``survives``: none falls).  Whether it works is judged against
    its condition's nominal evaluation (``works``, ``WORKS_RULE``)."""
    errors = int(sum(1 for e in episodes if e.get("error")))
    judged = errors == 0 and len(episodes) == len(SEEDS)
    return dict(condition=cond_key, variant=var.describe(), relax=list(relax), seeds=[int(e["seed"]) for e in episodes],
                episodes=episodes, succeeded=int(sum(bool(e["succeeded"]) for e in episodes)),
                survived=int(sum(not e["fell"] for e in episodes)), errors=errors,
                succeeds_all=bool(all(e["succeeded"] for e in episodes)) if judged else None,
                survives=bool(not any(e["fell"] for e in episodes)) if judged else None, valid=valid,
                wall_s=float(sum(e.get("wall_s") or 0.0 for e in episodes)),
                cpu_s=float(sum(e.get("cpu_s") or 0.0 for e in episodes)),
                simulated_s=float(sum(e.get("survival_s") or 0.0 for e in episodes)), at=tune._now())


def new_terms(entry: dict, ref: dict | None) -> dict[int, list[str]]:
    """Per seed, the success terms an evaluation's surviving episode failed that its seed passed in ``ref`` (the
    condition's nominal evaluation; ``WORKS_RULE``); every failed term without a reference."""
    base = {int(e["seed"]): set(e.get("failed_terms") or ()) for e in (ref or {}).get("episodes") or ()}
    return {int(e["seed"]): [t for t in e.get("failed_terms") or () if t not in base.get(int(e["seed"]), set())]
            for e in entry["episodes"] if not e.get("error") and not e["fell"]}


def works(entry: dict | None, ref: dict | None) -> bool | None:
    """``WORKS_RULE``: none of the episodes falls and none fails a term its seed passed in ``ref`` (the condition's
    nominal evaluation on the same seeds; ``entry`` itself at the nominal value); None while either is unjudged."""
    if entry is None or ref is None or entry.get("errors") or ref.get("errors") or entry.get("survives") is None:
        return None
    if [int(s) for s in entry["seeds"]] != [int(s) for s in ref["seeds"]]:
        raise ValueError(f"{entry['condition']}: its seeds {entry['seeds']} are not its reference's {ref['seeds']}")
    return bool(entry["survives"] and not any(new_terms(entry, ref).values()))


def lifts_of(entry: dict | None) -> tuple[list[str], list[str]]:
    """``LIFT_RULE``: the relaxations that lift the candidates of an evaluation's falls (``control.RELAXATIONS``
    order), and its candidates the plant cannot relax."""
    if not entry or entry.get("errors"):
        return [], []
    cands = {c for e in entry["episodes"] if e["fell"] for c in (e.get("fall_candidates") or ())}
    lifts = [r for r in control.RELAXATIONS if any(control.RELAXES.get(c) == r for c in cands)]
    return lifts, sorted(c for c in cands if control.RELAXES.get(c) is None)


def falls_label(entry: dict, lifted: dict | None) -> tuple[str | None, dict]:
    """``LIFT_RULE``'s label of an evaluation's falls from its lifts (relaxation -> the lifted evaluation, or None
    while it is not known), and what each lift gave."""
    falls = [e for e in entry["episodes"] if e["fell"] and not e.get("error")]
    if not falls:
        return None, {}
    lifts, untestable = lifts_of(entry)
    got = {r: (lifted or {}).get(r) for r in lifts}
    tests = {r: None if (x is None or x.get("errors")) else dict(survives=bool(x["survives"]), survived=x["survived"])
             for r, x in got.items()}
    note = f"; untestable: {', '.join(untestable)}" if untestable else ""
    if not lifts:
        return ("controller-limited (falls with no relaxable hardware limit active before the onset: nothing to "
                f"lift, decision 20{note})"), tests
    if any(t is None for t in tests.values()):
        return f"falls: lifts of {', '.join(lifts)} pending (LIFT_RULE{note})", tests
    body = [r for r in lifts if tests[r]["survives"]]
    if body:
        return f"body-limited: {', '.join(body)} (lifted, all 3 seeds survive{note})", tests
    return f"controller-limited (lifting {', '.join(lifts)} saves none of the falls{note})", tests


def binding(entry: dict | None, criterion: str, ref: dict | None = None, lifted: dict | None = None) -> dict | None:
    """``LABEL_RULE`` at a search's fails end (an evaluation that did not work, or did not survive): ``ref`` is the
    condition's nominal evaluation (``WORKS_RULE``) and ``lifted`` its lifts (``LIFT_RULE``)."""
    if entry is None:
        return None
    eps = [e for e in entry["episodes"] if not e.get("error")]
    falls = [e for e in eps if e["fell"]]
    terms = new_terms(dict(entry, episodes=eps), ref)
    failed = [dict(e, failed_terms=terms[int(e["seed"])]) for e in eps if not e["fell"] and terms[int(e["seed"])]]
    cands = Counter(c for e in falls for c in (e.get("fall_candidates") or ()))
    f_label, tests = falls_label(dict(entry, episodes=eps), lifted)
    sl = panels.success_label(failed)
    if criterion == "survival" or not failed:
        label = f_label
    elif not falls:
        label = sl["label"]
    else:  # panels.FAILURE_LABEL_RULE's tie-break: the falls'
        label = f_label if len(falls) >= len(failed) else sl["label"]
    lifts, untestable = lifts_of(dict(entry, episodes=eps))
    return dict(falls=len(falls), fall_causes=dict(Counter(str(e["fall_cause"]) for e in falls)),
                fall_candidates=dict(cands), lifts=tests, untestable=untestable, success_failures=len(failed),
                failed_terms=sl["terms"], hardware_terms=sl["hardware_terms"],
                success_label=sl["label"] if failed else None, falls_label=f_label, label=label,
                relative_to_nominal=ref is not None)


def panel_reading(out: Path, body: str, row: str, cond_key: str) -> dict | None:
    """``LABEL_RULE``'s citation at 'fails at nominal': the panels' 20-seed record of the row and condition (pass,
    counts, failure label with decision 20's attribution), or None if it does not exist."""
    cond = next((c for c in panels.conditions() if c.key == cond_key), None)
    rec = panels._read(panels.record_path(out, body, row, cond)) if cond is not None else None
    if not rec or rec.get("status") != "done":
        return None
    c, f, a = rec.get("counts") or {}, rec.get("failure") or {}, rec.get("attribution") or {}
    return dict(record=str(panels.record_path(Path(""), body, row, cond)), pass_=c.get("pass_"),
                episodes=c.get("episodes"), survived=c.get("survived"), succeeded=c.get("succeeded"),
                label=f.get("label"), falls_label=f.get("falls_label"), success_label=f.get("success_label"),
                attribution=a.get("label"), decision_20_label=a.get("decision_20_label"))


def stop_reference(run_dir: Path, rate: float = panels.RATE) -> dict:
    """``STOP_RULE``'s reading of the existing stop: its reference CoM's forward deceleration on the commanded clock,
    low-passed at 1 Hz (decision 10's reading), and unfiltered."""
    path = Path(run_dir) / "transitions" / "stop.npz"
    tr = Transition.load(path)
    t = np.asarray(tr.t, float) / float(rate)
    h0 = float(np.asarray(tr.heading)[0])
    s = np.asarray(tr.com, float)[:, :2] @ np.array([math.cos(h0), math.sin(h0)])
    v = np.gradient(s, t)
    a = np.gradient(v, t)
    a_lp = metrics.lowpass(a, 1.0 / float(np.median(np.diff(t))))
    return dict(transition="stop", file=str(path.relative_to(Path(run_dir))), sha256=plant.sha256(path),
                from_to=list(TRANSITIONS["stop"]), rate=float(rate),
                peak_deceleration_m_s2=float(max(0.0, -float(a_lp.min()))),
                peak_deceleration_unfiltered_m_s2=float(max(0.0, -float(a.min()))),
                speed_before_m_s=float(v[: max(1, len(v) // 5)].mean()),
                rule="the reference CoM's position along the walk's starting heading, differentiated twice on the "
                     f"commanded clock (rate {rate:g}), low-passed at {metrics.ACCEL_LP_HZ:g} Hz (metrics.lowpass): "
                     "its most negative value over the transition")


def stop_panel_reading(out: Path, body: str, row: str) -> dict | None:
    """What the row's start-and-stop panel record read of its stops (``STOP_RULE``), if it exists."""
    cond = next(c for c in panels.conditions() if c.key == "start_stop/twice")
    rec = panels._read(panels.record_path(out, body, row, cond))
    if not rec or rec.get("status") != "done":
        return None
    peaks, within, n = [], 0, 0
    for e in rec.get("episodes") or ():
        for st in ((e.get("measures") or {}).get("start_stop") or {}).get("stops") or ():
            n += 1
            within += bool(st.get("within"))
            if st.get("acceleration_peak_m_s2") is not None:
                peaks.append(float(st["acceleration_peak_m_s2"]))
    return dict(record=str(panels.record_path(Path(""), body, row, cond)), episodes=len(rec.get("episodes") or ()),
                stops=n, stopped_within=within, acceleration_peak_max_m_s2=max(peaks) if peaks else None,
                acceleration_peak_median_m_s2=float(np.median(peaks)) if peaks else None,
                counts=rec.get("counts"))


# ------------------------------------------------------------------------------------------------
# One margin of one row
# ------------------------------------------------------------------------------------------------
class MarginRun:
    """One margin on one row: its searches' bisections on both criteria, its points and the lifts at their fails ends
    (``LIFT_RULE``), advanced over the evaluations known (``advance``)."""

    def __init__(self, row: panels.Row, margin: Margin, conds: dict, point_variants: dict, halvings: int = HALVINGS):
        self.row, self.margin, self.halvings = row, margin, int(halvings)
        self.applicable = {s.name: row.runs(conds[s.condition]) for s in margin.searches}
        self.applicable.update({p.name: row.runs(conds[p.condition]) for p in margin.points})
        self.bisects = {(s.name, c): Bisect(s, c, halvings) for s in margin.searches for c in CRITERIA
                        if self.applicable[s.name]}
        self.point_variants = point_variants
        #: A point's (evaluation, its condition's nominal evaluation) once both are known.
        self.points: dict[str, tuple | None] = {p.name: None for p in margin.points if self.applicable[p.name]}
        self.waiting: set = set()  # (search or point, criterion): an evaluation failed to run this session
        self.lifts_pending: set = set()  # (search or point, criterion, relaxation): a lift not known yet
        self.used: dict[str, tuple] = {}  # evaluation id -> (condition, variant, relax)

    @property
    def done(self) -> bool:
        return (self.margin.not_run is not None
                or (all(b.done for b in self.bisects.values()) and all(v is not None for v in self.points.values())
                    and not self.lifts_pending))

    def _get(self, lookup, cond: str, var: Variant, relax=()):
        self.used[eval_id(cond, var, relax)] = (cond, var, tuple(relax))
        return lookup(cond, var, tuple(relax))

    def _lift(self, lookup, cond: str, var: Variant, need: list, who: tuple) -> None:
        """``LIFT_RULE``: the lifts a fails end (or a point) with falls needs."""
        for r in lifts_of(lookup(cond, var, ()))[0]:
            x = self._get(lookup, cond, var, (r,))
            if x is None:
                need.append((cond, var, (r,)))
            if x is None or x["errors"]:
                self.lifts_pending.add(who + (r,))

    def advance(self, lookup) -> list[tuple]:
        """Feed every bisection and point from ``lookup(condition, variant, relax)`` (an evaluation entry or None) as
        far as the known evaluations go; the (condition, variant, relax) evaluations still needed."""
        need = []
        self.lifts_pending = set()
        for (name, crit), b in self.bisects.items():
            s = b.search
            while not b.done:
                v = b.need()
                var = s.variant(v)
                e = self._get(lookup, s.condition, var)
                if e is None:
                    need.append((s.condition, var, ()))
                    break
                if e["errors"]:
                    self.waiting.add((name, crit))
                    break
                self.waiting.discard((name, crit))
                if crit == "survival":
                    ok = e["survives"]
                else:  # WORKS_RULE: against the search's nominal value, the first one evaluated
                    ok = works(e, e if not b.trail else lookup(s.condition, s.variant(s.nominal), ()))
                b.feed(v, ok)
            if b.done and b.fails is not None:
                self._lift(lookup, s.condition, s.variant(b.fails), need, (name, crit))
        for p in self.margin.points:
            if p.name not in self.points:
                continue
            var = self.point_variants[p.name]
            if self.points[p.name] is None:
                e = self._get(lookup, p.condition, var)
                ref = e if var == NOMINAL else self._get(lookup, p.condition, NOMINAL)
                missing = [x for x, got in dict.fromkeys(((var, e is None), (NOMINAL, ref is None))) if got]
                if missing:
                    need.extend((p.condition, x, ()) for x in missing)
                    continue
                if e["errors"] or ref["errors"]:
                    self.waiting.add((p.name, None))
                    continue
                self.waiting.discard((p.name, None))
                self.points[p.name] = (e, ref)
            e, ref = self.points[p.name]
            if not works(e, ref):
                self._lift(lookup, p.condition, var, need, (p.name, None))
        return need


def eval_id(cond_key: str, var: Variant, relax=()) -> str:
    return f"{cond_key}|{var.key}" + (f"|lift={'+'.join(relax)}" if relax else "")


def _lifted(lookup, cond: str, var: Variant) -> dict:
    """The lifts known of an evaluation's falls (``LIFT_RULE``): relaxation -> its evaluation or None."""
    return {r: lookup(cond, var, (r,)) for r in lifts_of(lookup(cond, var, ()))[0]}


def _search_results(mr: MarginRun, lookup, panel=None) -> dict:
    """Each search's two results with the label at the fails end; at 'fails at nominal', ``panel(condition)`` (the
    panels' reading of the row and condition, ``panel_reading``) is cited (``LABEL_RULE``)."""
    out = {}
    for s in mr.margin.searches:
        if not mr.applicable[s.name]:
            out[s.name] = dict(s.describe(), status="not run", reason=f"{mr.row.name} runs the hard floor only and "
                                                                        f"{s.condition} is not on it")
            continue
        r = dict(s.describe())
        ref = lookup(s.condition, s.variant(s.nominal), ())
        for crit in CRITERIA:
            b = mr.bisects[(s.name, crit)]
            res = b.result()
            if (s.name, crit) in mr.waiting:
                res["waiting"] = "an episode failed to run (WORKS_RULE): the next sweep runs it again"
            res["binding"] = None
            if b.fails is not None:
                var = s.variant(b.fails)
                fails = lookup(s.condition, var, ())
                res["binding"] = binding(fails, crit, ref, _lifted(lookup, s.condition, var)) if fails else None
            if res["status"] == "fails at nominal" and panel is not None:
                pr = panel(s.condition)
                res["panels"] = pr
                if pr and pr.get("pass_"):
                    res["flag"] = (f"the panels pass {s.condition} ({pr['survived']}/{pr['episodes']} survive, "
                                   f"{pr['succeeded']} succeed), the ledger's 3 seeds fail at nominal")
            r[crit] = res
        out[s.name] = r
    return out


def _point_results(mr: MarginRun, lookup) -> dict:
    out = {}
    for p in mr.margin.points:
        if p.name not in mr.points:
            out[p.name] = dict(p.describe(), status="not run", reason=f"{mr.row.name} does not run {p.condition}")
            continue
        var = mr.point_variants[p.name]
        base = dict(p.describe(), variant=var.describe(),
                    same_as_nominal=var == NOMINAL and p.kind != "push_dv")
        if mr.points[p.name] is None:
            out[p.name] = dict(base, status="waiting" if (p.name, None) in mr.waiting else "incomplete")
            continue
        e, ref = mr.points[p.name]
        w = works(e, ref)
        cm = next((x.get("controller_model") for x in e["episodes"] if x.get("controller_model")), None)
        out[p.name] = dict(base, status="done", works=w, works_outright=e["succeeds_all"], survives=e["survives"],
                           succeeded=e["succeeded"], survived=e["survived"],
                           robot_mass_change=(cm or {}).get("robot_mass_change"),
                           binding=None if w else binding(e, "success" if e["survives"] else "survival", ref,
                                                          _lifted(lookup, p.condition, var)))
    return out


def margin_summary(margin: Margin, searches: dict, points: dict) -> dict:
    """The margin's results as the report shows them: each search's threshold on both criteria (and the worse of
    the searches ``Margin.worse`` groups), each point's outcome."""
    rows = {}
    for s in margin.searches:
        r = searches[s.name]
        rows[s.name] = {c: _brief(r.get(c)) for c in CRITERIA} if r.get("status") != "not run" else dict(
            status="not run", reason=r.get("reason"))
    for name, group in margin.worse:
        specs = [s for s in margin.searches if s.name in group]
        if all(searches[s.name].get("status") != "not run" for s in specs):
            rows[name] = {c: _brief(worse({s.name: searches[s.name][c] for s in specs}, specs)) for c in CRITERIA}
            rows[name]["of"] = list(group)
    pts = {n: {k: p.get(k) for k in ("status", "value", "condition", "works", "works_outright", "survives",
                                     "succeeded", "survived", "same_as_nominal", "robot_mass_change")}
           | {"label": (p.get("binding") or {}).get("label")} for n, p in points.items()}
    return dict(searches=rows, points=pts)


def _brief(res: dict | None) -> dict | None:
    if res is None:
        return None
    return dict(status=res.get("status"), threshold=res.get("threshold"), fails=res.get("fails"),
                resolution=res.get("resolution"), search=res.get("search"),
                label=(res.get("binding") or {}).get("label"), evaluated=len(res.get("values") or ()),
                panels=res.get("panels"), flag=res.get("flag"))


# ------------------------------------------------------------------------------------------------
# Rows, records, keys
# ------------------------------------------------------------------------------------------------
#: The controllers the policy rows' module defines (``policy_rows.ROW_FACTORIES``, ``ROWS_RULE``).
POLICY_CONTROLLERS = ("R3", "R4")


def register_policy_rows(names) -> list[str]:
    """``ROWS_RULE``: the policy rows' R3 and R4 (``policy_rows.ROW_FACTORIES``) registered through
    ``panels.register_rows`` for the rows of ``names`` whose controller no module has registered; a controller another
    module registered (``--row-module``) keeps its rows.  Returns the controllers it registered."""
    missing = sorted(({panels.controller_of(n) for n in names} & set(POLICY_CONTROLLERS)) - set(panels.ROW_FACTORIES))
    if missing:
        before = dict(panels.ROW_FACTORIES)
        from . import policy_rows  # (its first import registers R3 and R4)

        panels.ROW_FACTORIES.update(before)
        for c in missing:
            panels.register_rows(c, policy_rows.ROW_FACTORIES[c])
    return missing


def runs_root(run_dir: Path) -> Path:
    """The training run directory a policy row reads (``policy_rows.runs_dir``: ``STAGE1_POLICY_RUNS``, default the
    parent of ``<run>/<body>``), without importing policy_rows (its import registers its rows, ``ROWS_RULE``)."""
    v = os.environ.get("STAGE1_POLICY_RUNS")
    return Path(v) if v else Path(run_dir).parent


def resolve_rows(body: str, run_dir: Path, names, left_out: dict | None = None) -> list[str]:
    """``ROWS_RULE``: 'R2' -> the body-level G5 selection's R2 row, 'R3' -> R3/42 unless the body's R3 is judged with
    R2 (``train.judged_with_r2``: then left out, the reason in ``left_out``); other names as given."""
    from . import train

    out = []
    for n in names:
        if n == "R2":
            _, info = env.tuned_gains(Path(run_dir), body)
            if info.get("cma_seed") is None or info.get("status") != "passed":
                raise ValueError(f"{body}: no body-level G5 selection that passes ({info.get('status')}); name an R2 "
                                 "row (R2/42 or R2/43)")
            n = f"R2/{int(info['cma_seed'])}"
        elif n == "R3":
            j = train.judged_with_r2(runs_root(run_dir), body)
            if j["with_r2"]:
                if left_out is not None:
                    left_out[R3_ROW] = f"{body}'s R3 is judged with R2: {j['reason']} (train.R2_JUDGED_RULE)"
                continue
            n = R3_ROW
        if n not in out:
            out.append(n)
    return out


def record_path(out: Path, body: str, row: str, margin: str) -> Path:
    """``RECORD_RULE``: ``<out>/<body>/sweeps/<row>/<margin>.json``."""
    return Path(out) / body / "sweeps" / row / f"{margin}.json"


@functools.lru_cache(maxsize=None)
def bus_changes(body: str, volts: float) -> bool:
    """Whether a bus voltage changes any drive's speeds against the record's bus (``VOLTAGE_RULE``)."""
    xml, _ = plant.model_files(body)
    joints = plant._hinges(mujoco.MjModel.from_xml_path(str(xml)))
    sel = plant.record(body)["selection"]
    layout = plant.leg_layout(body)
    a, b = drive.drives(sel, joints, layout=layout), drive.drives(sel, joints, bus=float(volts), layout=layout)
    return any((x.rated_speed, x.no_load_speed) != (y.rated_speed, y.no_load_speed) for x, y in zip(a, b, strict=True))


class Session:
    """The planning a sweep and its dry run share for one body: inputs, rows, conditions, keys and the evaluations the
    records hold (``RESUME_RULE``)."""

    def __init__(self, body: str, out: Path, rows=DEFAULT_ROWS, names=None, inputs: Path | None = None,
                 cut_s: float | None = None, halvings: int = HALVINGS):
        if body not in BODIES:
            raise KeyError(body)
        self.body, self.out = body, Path(out)
        self.run_dir = Path(inputs or out) / body
        self.cut_s, self.halvings = cut_s, int(halvings)
        self.inputs, docs = tune.read_inputs(self.run_dir, panels.panel_files())
        self.planner = panels.Planner(self.run_dir, docs["refs.json"]["references"],
                                      docs["transitions.json"]["transitions"])
        self.left_out: dict[str, str] = {}
        self.row_names = resolve_rows(body, self.run_dir, rows, self.left_out)
        self.registered = register_policy_rows(self.row_names)
        self.rows = panels.tracker_rows(body, self.run_dir, self.row_names)
        self.margins = select_margins(margins(body), names)
        by_key = {c.key: c for c in panels.conditions()}
        keys = sorted({k for m in self.margins for k in m.conditions()})
        self.conds = {k: by_key[k] if cut_s is None else dataclasses.replace(by_key[k], seconds=float(cut_s))
                      for k in keys}
        self.resolved = {k: self.planner.resolve(c) for k, c in self.conds.items()}
        self.sources, self.context = panels.panel_sources(), panels.key_context(body)
        self.ledger = ledger_sources()
        self.panel_keys = {(r, k): _sha(panels.record_key(body, row, c, self.resolved[k], SEEDS, self.inputs,
                                                           self.sources["sha256"], self.context))
                           for r, row in self.rows.items() for k, c in self.conds.items()}
        self.point_variants = {}
        for m in self.margins:
            for p in m.points:
                var = Variant(p.kind, p.value)
                if p.kind == "bus_v" and not bus_changes(body, p.value):
                    var = NOMINAL
                self.point_variants[(m.name, p.name)] = var
        self.cache: dict[tuple, dict] = {}
        self.records: dict[tuple, dict | None] = {}
        for r in self.rows:
            for m in self.margins:
                self.records[(r, m.name)] = panels._read(record_path(self.out, body, r, m.name))
            for path in sorted((self.out / body / "sweeps" / r).glob("*.json")):
                rec = panels._read(path)
                for entry in ((rec or {}).get("evaluations") or {}).values():
                    k = entry.get("condition")
                    if k not in self.conds or entry.get("errors"):
                        continue
                    var, relax = Variant(**entry["variant"]), tuple(entry.get("relax") or ())
                    if entry.get("valid") == self.valid(r, k, var, relax):
                        self.cache[(r, k, var.key, relax)] = entry
        self.runs = {(r, m.name): MarginRun(row, m, self.conds, {p.name: self.point_variants[(m.name, p.name)]
                                                                  for p in m.points}, self.halvings)
                     for r, row in self.rows.items() for m in self.margins}

    def valid(self, row: str, cond_key: str, var: Variant, relax=()) -> str:
        """``RESUME_RULE``'s evaluation key (a lift's relaxation added, ``LIFT_RULE``)."""
        key = dict(panel=self.panel_keys[(row, cond_key)], variant=var.describe(), seeds=list(SEEDS),
                   ledger=self.ledger["sha256"])
        if relax:
            key["relax"] = list(relax)
        return _sha(key)

    def key(self, row: str, m: Margin) -> dict:
        """A margin record's key (``RESUME_RULE``)."""
        return panels._jsonable(dict(schema=SCHEMA, body=self.body, row=row, margin=m.describe(), seeds=list(SEEDS),
                                     halvings=self.halvings, cut_s=self.cut_s, ledger=self.ledger["sha256"],
                                     panel_keys={k: self.panel_keys[(row, k)] for k in m.conditions()},
                                     points={p.name: self.point_variants[(m.name, p.name)].describe()
                                             for p in m.points}))

    def up_to_date(self, row: str, m: Margin) -> bool:
        rec = self.records.get((row, m.name))
        return bool(rec and rec.get("key") == self.key(row, m) and rec.get("status") in ("done", "not run"))

    def lookup(self, row: str):
        return lambda cond_key, var, relax=(): self.cache.get((row, cond_key, var.key, tuple(relax)))

    def advance(self) -> dict:
        """Every margin run fed from the evaluations known; the evaluations still needed: (row, condition, variant
        key, relax) -> (row, condition, variant, relax)."""
        need = {}
        for (r, _), mr in self.runs.items():
            for k, var, relax in mr.advance(self.lookup(r)):
                need[(r, k, var.key, tuple(relax))] = (r, k, var, tuple(relax))
        return need

    def record(self, row: str, m: Margin, prov: dict, session: dict) -> dict:
        """The margin's record (``RECORD_RULE``)."""
        mr = self.runs[(row, m.name)]
        lookup = self.lookup(row)
        searches = _search_results(mr, lookup, lambda k: panel_reading(self.out, self.body, row, k))
        points = _point_results(mr, lookup)
        evals = {i: self.cache[(row, k, var.key, relax)] for i, (k, var, relax) in sorted(mr.used.items())
                 if (row, k, var.key, relax) in self.cache}
        if m.not_run:
            status = "not run"
        elif mr.done:
            status = "done"
        else:
            status = "partial"
        row_obj = self.rows[row]
        doc = dict(schema=SCHEMA, body=self.body, row=row, margin=m.name, title=m.title, swept=m.swept, status=status,
                   controller=row_obj.describe(), seeds=list(SEEDS), halvings=self.halvings, cut_s=self.cut_s,
                   definition=m.describe(), rule=m.rule, searches=searches, points=points,
                   summary=margin_summary(m, searches, points), stage0=stage0(self.body, m.name),
                   evaluations=evals, rules=RULES, key=self.key(row, m), panel_sources=self.sources["sha256"],
                   key_context=panels.fingerprint(self.sources["sha256"], self.context), provenance=prov,
                   cost=dict(evaluations=len(evals), episodes=int(sum(len(e["episodes"]) for e in evals.values())),
                             simulated_s=float(sum(e["simulated_s"] for e in evals.values())),
                             wall_s=float(sum(e["wall_s"] for e in evals.values())),
                             cpu_s=float(sum(e["cpu_s"] for e in evals.values())),
                             note="each evaluation's own cost; one shared with another margin of the row is in both "
                                  "records (the report counts it once, by its key)"),
                   session=dict(started_at=session["started_at"]), written_at=tune._now())
        if m.name == "mass":
            trunk, robot = MASSES[self.body]
            doc["masses"] = dict(trunk_body_kg=trunk, robot_kg=robot, trunk_share=trunk / robot,
                                 robot_point=ROBOT_MASS_POINT[self.body], rule=MASS_RULE)
        if m.not_run:
            doc["not_run"] = m.not_run
            if m.name == "stop":
                try:
                    doc["reference_stop"] = stop_reference(self.run_dir)
                except (OSError, KeyError, ValueError) as exc:
                    doc["reference_stop"] = dict(error=str(exc))
                doc["panel_reading"] = stop_panel_reading(self.out, self.body, row)
        return doc


# ------------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------------
def sweep(body: str, out: Path, rows=DEFAULT_ROWS, names=None, workers: int = 1, inputs: Path | None = None,
          cut_s: float | None = None, halvings: int = HALVINGS, timeout_s: float = tune.EVAL_TIMEOUT_S,
          log=print) -> dict:
    """The margin ledger of ``body`` (module docstring) into ``<out>/<body>/sweeps/``; references from
    ``<inputs>/<body>/`` (default: out).  ``cut_s`` cuts every condition to that many simulated seconds (a smoke check:
    its keys differ, nothing it records counts for the full conditions); ``halvings`` below 7 likewise.  Returns the
    session's summary."""
    t_start = time.perf_counter()
    ses = Session(body, out, rows, names, inputs, cut_s, halvings)
    prov = plant.provenance()
    session = dict(started_at=tune._now(), rows=list(ses.row_names), rows_left_out=dict(ses.left_out),
                   margins=[m.name for m in ses.margins], workers=int(workers), cut_s=cut_s, halvings=ses.halvings,
                   seeds=list(SEEDS), rounds=0, episodes_run=0, evaluations_reused=len(ses.cache), skipped=[],
                   incidents=[])
    for r, why in ses.left_out.items():
        log(f"{body}: {r} left out: {why}")
    todo = []
    for r in ses.rows:
        for m in ses.margins:
            if ses.up_to_date(r, m):
                session["skipped"].append(f"{r} {m.name}")
                ses.runs.pop((r, m.name))
            else:
                todo.append((r, m))
    pool = None
    failed: set = set()

    def write_all():
        for r, m in todo:
            panels._write_json(record_path(ses.out, body, r, m.name), ses.record(r, m, prov, session))

    try:
        while True:
            need = {k: v for k, v in ses.advance().items() if k not in failed}
            if not need:
                break
            if pool is None:
                pool = tune.Pool(body, ses.run_dir, workers, mode="ledger", rate=panels.RATE, timeout_s=timeout_s,
                                 factory=functools.partial(_worker, rows=[ses.rows[r].spec() for r in ses.rows],
                                                           sources=panels.fingerprint(ses.sources["sha256"],
                                                                                      ses.context),
                                                           ledger=ses.ledger["sha256"]))
            jobs = [((r, ses.conds[k], var.kind, var.value, relax), s) for (r, k, var, relax) in need.values()
                    for s in SEEDS]
            results = pool.map(jobs, failed=_lost)
            session["rounds"] += 1
            session["episodes_run"] += len(jobs)
            n = len(SEEDS)
            for i, (key, (r, k, var, relax)) in enumerate(need.items()):
                entry = evaluation_entry(k, var, results[n * i: n * (i + 1)], ses.valid(r, k, var, relax), relax)
                ses.cache[key] = entry
                if entry["errors"]:
                    failed.add(key)
            write_all()
            log(f"{body} round {session['rounds']}: {len(need)} evaluations ({len(jobs)} episodes): "
                + "; ".join(f"{r} {eval_id(k, var, relax)}" for r, k, var, relax in list(need.values())[:6])
                + (" ..." if len(need) > 6 else ""))
        ses.advance()
        write_all()
        if pool is not None:
            session["incidents"] = list(pool.incidents)
    finally:
        if pool is not None:
            pool.close()
    session.update(ended_at=tune._now(), wall_s=time.perf_counter() - t_start, provenance=prov,
                   ledger_sources=ses.ledger, panel_sources=ses.sources["sha256"])
    return summarize(body, out, ses, session, log)


def summarize(body: str, out: Path, ses: Session | None = None, session: dict | None = None, log=print) -> dict:
    """``<out>/<body>/sweeps/summary.json``: every margin record's state and summary, and the sessions (each sweep
    appends its own)."""
    sdir = Path(out) / body / "sweeps"
    path = sdir / "summary.json"
    old = panels._read(path) or {}
    recs = {}
    for f in sorted(sdir.glob("*/**/*.json")):
        rec = panels._read(f)
        if rec and rec.get("margin") and rec.get("row"):
            kinds = {x["name"]: x["kind"] for x in (rec.get("definition") or {}).get("searches") or ()}
            recs.setdefault(rec["row"], {})[rec["margin"]] = dict(status=rec.get("status"), summary=rec.get("summary"),
                                                                  kinds=kinds, cost=rec.get("cost"),
                                                                  cut_s=rec.get("cut_s"))
    doc = dict(schema=SCHEMA, body=body, rules=RULES, rows=recs,
               sessions=list(old.get("sessions") or []) + ([session] if session else []), written_at=tune._now())
    panels._write_json(path, doc)
    for r, ms in recs.items():
        for m, x in ms.items():
            log(f"{body} {r} {m}: {x['status']}" + "".join(f"\n  {line}" for line in summary_lines(x.get("summary"),
                                                                                                   x.get("kinds"))))
    return doc


def _value(x, kind: str | None) -> str:
    if x is None:
        return "?"
    if kind == "latency_steps":
        return f"{int(x)} steps ({2 * int(x)} ms)"
    return f"{x:.4g}"


def summary_lines(summary: dict | None, kinds: dict | None = None) -> list[str]:
    """A margin summary as text lines (the sweep's log)."""
    if not summary:
        return []
    lines = []
    for name, r in (summary.get("searches") or {}).items():
        if r.get("status") == "not run":
            lines.append(f"{name}: not run ({r.get('reason')})")
            continue
        parts = []
        for crit in CRITERIA:
            b = r.get(crit) or {}
            st = b.get("status")
            kind = (kinds or {}).get(name)
            if st == "bisected":
                parts.append(f"{crit} {_value(b['threshold'], kind)} (fails {_value(b['fails'], kind)}"
                             + (f"; {b['label']}" if b.get("label") else "") + ")")
            elif st == "fails at nominal":
                parts.append(f"{crit} fails at nominal" + (f" ({b['label']})" if b.get("label") else ""))
            else:
                parts.append(f"{crit} {st}")
        lines.append(f"{name}: " + "; ".join(parts))
    for name, p in (summary.get("points") or {}).items():
        lines.append(f"{name}: " + ("not run" if p.get("status") == "not run" else
                                    f"works {p.get('works')} (outright {p.get('works_outright')}), survives "
                                    f"{p.get('survives')}" + (f" ({p['label']})" if p.get("label") else "")))
    return lines


# ------------------------------------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------------------------------------
def policy_costs(ses: Session, lines: list[dict], log=print) -> dict:
    """``COST_RULE``: each policy row's own cost per simulated second on the floors its searches still need
    (``policy_rows.measure_costs``: the row, its network included, on the nominal walk), {row: {floor: cost}}."""
    need: dict[str, set] = {}
    for x in lines:
        if x.get("floor") and x["values_max"] and panels.controller_of(x["row"]) in POLICY_CONTROLLERS:
            need.setdefault(x["row"], set()).add(x["floor"])
    if not need:
        return {}
    from . import policy_rows

    out = {}
    for row, floors in sorted(need.items()):
        m = policy_rows.measure_costs(ses.body, ses.run_dir, (row,), tuple(sorted(floors)), r2=False, log=log)
        out[row] = {f: dict(cpu_per_sim_s=v["cpu_per_sim_s"], wall_per_sim_s=v["wall_per_sim_s"],
                            checkpoint=(m.get("checkpoints") or {}).get(row)) for f, v in m["rows"][row].items()}
    return out


def dry_run(body: str, out: Path, rows=DEFAULT_ROWS, names=None, inputs: Path | None = None,
            cut_s: float | None = None, halvings: int = HALVINGS, measure: bool = True, costs: dict | None = None,
            workers: int = 1, log=print) -> dict:
    """Every search and point the sweep would run, the evaluations it can still need after replaying the recorded
    ones, and their cost (``COST_RULE``; the lifts, ``LIFT_RULE``, as an upper bound beside); runs nothing (except the
    cost measurement)."""
    ses = Session(body, out, rows, names, inputs, cut_s, halvings)
    ses.advance()
    lines = []
    n_relax = len(control.RELAXATIONS)
    for r, why in ses.left_out.items():
        log(f"{body}: {r} left out: {why}")
    for (r, mname), mr in ses.runs.items():
        m = mr.margin
        up = ses.up_to_date(r, m)
        if m.not_run:
            lines.append(dict(row=r, margin=m.name, search=None, status="not run", reason=m.not_run, values_min=0,
                              values_max=0, lifts_max=0, condition=None, floor=None, seconds=0.0))
            continue
        covered = set()  # conditions whose nominal evaluation a search or a point of the row's margins makes
        for mm in ses.margins:
            covered |= {s.condition for s in mm.searches if s.variant(s.nominal) == NOMINAL}
            covered |= {p.condition for p in mm.points if ses.point_variants[(mm.name, p.name)] == NOMINAL}
        for s in m.searches:
            if not mr.applicable[s.name]:
                lines.append(dict(row=r, margin=m.name, search=s.name, status="not applicable", values_min=0,
                                  values_max=0, lifts_max=0, condition=s.condition, floor=ses.conds[s.condition].floor,
                                  seconds=0.0))
                continue
            left = {c: 0 if up else s.max_values(halvings) - len(mr.bisects[(s.name, c)].trail) for c in CRITERIA}
            lifts = 0
            for c in CRITERIA:
                if mr.bisects[(s.name, c)].done:
                    left[c] = 0
                elif not up:
                    lifts += n_relax  # an upper bound: every relaxation at this criterion's fails end
            if not up:
                lifts += sum(1 for w in mr.lifts_pending if w[0] == s.name)
            lines.append(dict(row=r, margin=m.name, search=s.name, condition=s.condition,
                              floor=ses.conds[s.condition].floor, interval=[s.nominal, s.far], kind=s.kind,
                              status="up to date" if up else "to run",
                              known_values={c: len(mr.bisects[(s.name, c)].trail) for c in CRITERIA},
                              values_min=max(left.values()), values_max=sum(left.values()), lifts_max=lifts,
                              seconds=ses.resolved[s.condition]["seconds"]))
        refs = set()
        for p in m.points:
            if p.name not in mr.points:
                continue
            todo = 0 if up or mr.points[p.name] is not None else 1
            lifts = 0 if up else (n_relax if mr.points[p.name] is None
                                  else sum(1 for w in mr.lifts_pending if w[0] == p.name))
            lines.append(dict(row=r, margin=m.name, search=p.name, condition=p.condition,
                              floor=ses.conds[p.condition].floor, point=p.value, kind=p.kind,
                              variant=mr.point_variants[p.name].describe(),
                              status="up to date" if not todo else "to run",
                              values_min=todo, values_max=todo, lifts_max=lifts,
                              seconds=ses.resolved[p.condition]["seconds"]))
            if todo and p.condition not in covered and ses.lookup(r)(p.condition, NOMINAL) is None:
                refs.add(p.condition)
        for k in sorted(refs):  # WORKS_RULE: a point's reference no search or point of the session makes
            lines.append(dict(row=r, margin=m.name, search=f"{k} nominal", condition=k, floor=ses.conds[k].floor,
                              kind=None, variant=NOMINAL.describe(), status="to run (a point's reference)",
                              values_min=1, values_max=1, lifts_max=0, seconds=ses.resolved[k]["seconds"]))
    floors = sorted({x["floor"] for x in lines if x.get("floor") and (x["values_max"] or x["lifts_max"])})
    if costs is None and measure and floors:
        costs = dict(panels.measure_costs(body, ses.run_dir, floors, log=log), rows=policy_costs(ses, lines, log))
    fl = (costs or {}).get("floors") or {}
    per_row = (costs or {}).get("rows") or {}
    total = Counter()
    for x in lines:
        n_min, n_max = len(SEEDS) * x["values_min"], len(SEEDS) * x["values_max"]
        n_lift = len(SEEDS) * x["lifts_max"]
        c = ((per_row.get(x["row"]) or fl).get(x.get("floor")) or {}).get("cpu_per_sim_s")
        x.update(episodes_min=n_min, episodes_max=n_max, simulated_s_max=n_max * x["seconds"],
                 core_h_max=None if c is None else n_max * x["seconds"] * c / 3600,
                 core_h_min=None if c is None else n_min * x["seconds"] * c / 3600,
                 lift_episodes_max=n_lift, lift_core_h_max=None if c is None else n_lift * x["seconds"] * c / 3600)
        total["episodes_min"] += n_min
        total["episodes_max"] += n_max
        total["lift_episodes_max"] += n_lift
        total["simulated_s_max"] += x["simulated_s_max"]
        for k in ("core_h_min", "core_h_max", "lift_core_h_max"):
            total[k] += x[k] or 0.0
        if x["search"] is not None:
            log(f"{x['row']:7s} {x['margin']:13s} {x['search']:22s} {x['condition'] or '':28s} "
                f"{x['episodes_min']:3d}-{x['episodes_max']:3d} episodes (lifts up to {n_lift}) "
                f"x {x['seconds']:6.2f} s  "
                + (f"{x['core_h_max']:.3f} core-h at most" if x["core_h_max"] is not None else "cost not measured")
                + f"  [{x['status']}]")
        else:
            log(f"{x['row']:7s} {x['margin']:13s} {x['status']}: {x['reason']}")
    w_n = max(1, int(workers))
    log(f"{body}: {total['episodes_min']}-{total['episodes_max']} episodes and up to {total['lift_episodes_max']} "
        f"lifted ({len(ses.cache)} evaluations recorded under this run's keys), at most "
        f"{total['simulated_s_max'] / 3600:.2f} simulated h, "
        + (f"{total['core_h_min']:.2f}-{total['core_h_max']:.2f} core-h (CPU) and up to {total['lift_core_h_max']:.2f} "
           f"for the lifts, about {total['core_h_max'] / w_n:.2f} h of wall clock at most with {w_n} "
           f"worker{'s' if w_n > 1 else ''} before the lifts"
           if fl else "cost not measured"))
    return dict(body=body, rule=COST_RULE, rows=list(ses.row_names), rows_left_out=dict(ses.left_out),
                margins=[m.name for m in ses.margins], lines=lines, costs=costs, totals=dict(total), workers=w_n,
                evaluations_recorded=len(ses.cache))


__all__ = ["Bisect", "DEFAULT_ROWS", "HALVINGS", "LedgerContext", "LedgerWorker", "MARGIN_NAMES", "MASSES", "Margin",
           "NOMINAL", "Point", "PushedBuilt", "ROBOT_MASS_POINT", "RULES", "SEEDS", "STAGE0", "Search", "Session",
           "Variant", "binding", "dry_run", "margins", "record_path", "register_policy_rows", "resolve_rows", "screens",
           "select_margins", "stage0", "stop_reference", "summarize", "sweep", "works", "worse"]
