"""Stage 1 margin ledger (``STAGE1_PLAN.md`` section 5; section 3 step 8's sweeps; section 7's ``sweeps/``): each
Stage 0 margin bisected to failure on a controller row, on episode seeds 3042-3044, with the behaviour panels' runner,
conditions and judgement (``panels``).

Run from hardware/trex_form_study::

    python -m designlab.stage1 sweep --body 60-N-T-10 [--body ...] [--rows R2,R3/42] [--margins grip,push,...]
                                     [--workers 4] [--dry-run] [--no-measure] [--row-module MODULE]
                                     [--inputs DIR] --out stage1/runs/2026-11-08

Rows (``ROWS_RULE``): R2 is the panels' row of the CMA-ES seed the body-level G5 selection comes from (``--rows R2``),
R3 is R3/42 from the policy rows' module (``--row-module``, ``panels.register_rows``); any row ``panels.tracker_rows``
knows can be named.  Margins (``margins``, plan section 5's table; ``MARGIN_RULES``): grip, push, stop, drive speed,
drive torque, the model doubts (servo kp, latency, trunk mass, pile damping) and pack voltage.  Each margin is one or
more searches (``Search``: one parameter on one panel condition, from its nominal value toward the far end of a stated
interval) and points (``Point``: one value).  How each parameter enters the plant: ``VARIANT_RULE`` and the margin's
own rule.

A search (``SEARCH_RULE``, ``Bisect``) evaluates the nominal value, then the far end, then halves the bracket 7 times;
a value works when all three episodes succeed by the panels' judgement (``WORKS_RULE``), and every search is also run
on survival alone, sharing the evaluations.  The value that bounds a threshold is labelled by what failed there
(``LABEL_RULE``).  Each (body, row, margin) gives one record ``<out>/<body>/sweeps/<row>/<margin>.json``
(``RECORD_RULE``), written after every round of evaluations; a sweep resumes from the evaluations its records hold
(``RESUME_RULE``).  ``dry_run`` lists every search with the episodes it can still need and their cost
(``COST_RULE``).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import math
import time
import traceback
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .. import carpet
from . import BODIES, control, drive, env, metrics, panels, plant, tune
from .transitions import TRANSITIONS, Transition

SCHEMA = 1
#: Plan section 5: the ledger's episode seeds, the panels' first three (``SEEDS_RULE``).
SEEDS = panels.SEEDS[:3]
HALVINGS = 7
CRITERIA = ("success", "survival")
#: The rows plan section 5 bisects on: "R2" is the body-level G5 selection's row, "R3" R3/42 (``ROWS_RULE``).
DEFAULT_ROWS = ("R2", "R3/42")
R3_ROW = "R3/42"
#: The intervals (each search's far end; the nominal value is the panels' condition as it runs).
MU_NOMINAL, MU_FAR = 0.9, 0.1
PUSH_FAR_BOUNDS = 2.0  # x the Stage 0 forward push bound
SCALE_FAR = 0.25
KP_FAR = 0.25
LATENCY_MAX_STEPS = 10  # physics steps of 2 ms: 20 ms
MASS_POINT, MASS_FAR = 1.1, 1.3
DAMPING_LOW, DAMPING_HIGH = 0.5, 2.0
VOLTAGES = (25.2, 22.2, 21.0)
#: The parameter a variant changes: name and unit (for the records and the report).
PARAMETERS = {"friction": ("sliding friction mu", ""), "push_dv": ("push velocity change", "m/s"),
              "speed_scale": ("envelope speed scale", ""), "torque_scale": ("drive torque scale", ""),
              "kp": ("servo kp factor", ""), "latency_steps": ("actuation latency", "physics steps of 2 ms"),
              "mass": ("trunk mass factor", ""), "pile_damping": ("pile damping factor", ""),
              "bus_v": ("bus voltage", "V")}
KINDS = tuple(PARAMETERS)
#: Plan section 5's Stage 0 column, per body (the round-two screens' values as the plan's table gives them).
STAGE0 = {
    "60-N-T-10": dict(grip_walk_nominal=0.48, grip_turn=0.47, grip_walk_brisk=0.36, push_forward=0.487,
                      stop_one_foot=1.25, drive_speed_knee=0.93, drive_torque_sustained=0.76),
    "60-N-H-10": dict(grip_walk_nominal=0.54, grip_turn=0.53, grip_walk_brisk=0.41, push_forward=0.478,
                      stop_one_foot=1.25, drive_speed_knee=0.95, drive_torque_sustained=0.76),
    "75-M-H-10": dict(grip_walk_nominal=0.75, grip_turn=0.73, grip_walk_brisk=0.56, push_forward=0.589,
                      stop_one_foot=1.41, drive_speed_knee=0.93, drive_torque_sustained=0.77),
}
STAGE0_VOLTAGE = ("Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement "
                  "including the screens' 15 % speed margin (8.08-8.12 rad/s; the peak swing speed itself is about "
                  "7.0 rad/s)")

# ------------------------------------------------------------------------------------------------
# Rules (each copied into every record)
# ------------------------------------------------------------------------------------------------
ROWS_RULE = (
    "plan section 5 bisects on R2 and R3: R2 is the panels' row of the CMA-ES seed the body-level G5 selection comes "
    "from (env.tuned_gains, env.GAINS_RULE: 60-N-T-10 R2/42, 60-N-H-10 R2/43, 75-M-H-10 R2/43; the row name 'R2' "
    "names it), the tracker on the estimate with that seed's own G5 selection (panels.R2_GAINS_RULE); R3 is R3/42, the "
    "residual policy at its judged checkpoint (train.JUDGED_RULE), a row of the policy rows' module (panels."
    "register_rows, --row-module); any row panels.tracker_rows knows can be named, and a row that runs the hard floor "
    "only (R0, R1) has no pile search")
SEEDS_RULE = (
    f"episode seeds {SEEDS[0]}-{SEEDS[-1]} (panels.SEEDS' first three, plan section 5); each episode is the panels' "
    "episode of its condition on that seed (panels.EPISODE_RULE: the start noise, the sensor noise and a push's timing "
    "from the seed), on the panels' plant and track with the search's variant applied (VARIANT_RULE), so the nominal "
    "value of a condition the variant leaves unchanged is the panels' own episode bit for bit; no traces are written")
WORKS_RULE = (
    "a value works if all 3 episodes succeed by the panels' judgement of the condition (panels.judge: its success "
    "terms with panels.LIMITS_RULE; a push succeeds if the body is back on its command within 3 s, decision 30, "
    "panels.RECOVERY_RULE), and survives if none of the 3 falls (plant.Plant.fall_cause, an episode with a MuJoCo "
    "warning counting as a fall, tune.FAILURE_RULE); an episode that failed to run (panels.FAILED_EPISODE_RULE) leaves "
    "the value unjudged: the search waits, and the next sweep runs that value again")
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
    "replaces the panels' half bound in the pulse's force (PushedBuilt)")
LABEL_RULE = (
    "a threshold names what failed at its fails end (the nearest value that does not work): the success failures' "
    "failed terms classed as panels.SUCCESS_LABEL_RULE classes them (body-limited by the hardware terms that failed in "
    "more than half of the episodes that survived but failed, else controller-limited) and the falls' causes and "
    "candidate limits (control.CANDIDATE_RULE: the hardware limits active in the 0.5 s before the onset; falls with "
    "none are controller-limited by decision 20, nothing could be lifted); the label is the success failures' when "
    "they are at least as many as the falls, else the falls' (a survival search: the falls').  Decision 20 attributes "
    "falls over 20 starts and its sign test needs at least "
    f"{panels.ATTRIBUTION_MIN_FALLS} falls (panels.ATTRIBUTION_MIN_FALLS), so the ledger's 3 seeds do not attribute: a "
    "fall's candidates are named, not tested")
RECORD_RULE = (
    "one record per (body, row, margin): <out>/<body>/sweeps/<row>/<margin>.json (a row 'R2/43' is two directories) "
    "with each search's two results (status, the threshold and the nearest failing value, every value evaluated, the "
    "label at the fails end), the points' outcomes, every evaluation the margin used (its condition, variant, seeds "
    "and each episode's outcome: the fall, its cause and candidates, the failed terms, the readings the margin turns "
    "on, wall and CPU seconds), Stage 0's value (plan section 5), the rules, the key, the provenance and the cost; "
    "written after every round of evaluations, so an interrupted sweep keeps what it ran; "
    "<out>/<body>/sweeps/summary.json lists the sessions and every record's state")
RESUME_RULE = (
    "a margin whose record holds this run's key and is done is skipped; otherwise every evaluation that any record of "
    "the row holds under this run's evaluation key is reused: the sha256 of the panels' record key of its condition on "
    "the ledger's seeds (panels.record_key: the row and its sources, the condition and its plan, the inputs, the "
    "plant's round-two files, the library versions, the panels' sources, panels.RESUME_RULE) with its variant and the "
    "ledger's sources (SOURCES_RULE); evaluations that failed to run are run again.  A search replays its evaluations "
    "in order, so a resumed sweep ends where an uninterrupted one would.  The margin's key adds its definition "
    "(searches, intervals, points, seeds, halvings)")
SOURCES_RULE = ("the ledger's sources: the sha256 of designlab/stage1/ledger.py (its margins, variants and searches); "
                "the panels' sources are in each evaluation's panel key (panels.PANEL_SOURCES_RULE); every worker "
                "checks both against the session's, so a worker started after an edit refuses to run")
COST_RULE = (
    "the dry run counts, per search, the values it can still need after replaying the evaluations recorded under this "
    f"run's keys: at most 2 + {HALVINGS} on success (an integer search: 2 + its halvings) and as many more on survival "
    "(none when the two agree), each x 3 seeds at the condition's planned seconds (panels.Planner); a point once (a "
    "value two searches share, such as the nominal walk's nominal value, is counted in each and run once); the "
    "cost per simulated second is panels.COST_RULE's (R2/42 walking on each floor, measured now; a policy row's "
    "network adds its own); core-hours = simulated seconds x CPU seconds per simulated second / 3600")
GRIP_RULE = (
    "the lowest sliding friction that works: every colliding geom's sliding friction set to mu (control.set_friction, "
    f"as the panels set mu 0.54) on the nominal walk (walk/nominal), the 0.35 rad/s walking turn left and right "
    "(turn/walking_0.35_left and _right; the report gives the worse of the two, the higher mu) and the brisk walk "
    f"(walk/brisk), from mu {MU_NOMINAL:g} (the hard floor as compiled, every colliding geom at 0.9: the panels' "
    f"condition) down to {MU_FAR:g}; Stage 0: the friction the round-two screens require (plan section 5)")
PUSH_RULE = (
    "the largest push recovered: the panels' push conditions (standing and walking; forward, backward, left and right; "
    "the pulse's timing and the episode's length as panels.PUSH_RULE) with the pulse's velocity change dv in place of "
    "half the Stage 0 bound: force m dv / 0.1 s (m the robot's mass) along the direction in the home heading, from dv "
    f"0 (no push) up to {PUSH_FAR_BOUNDS:g} x the Stage 0 forward bound (env.PUSH_BOUND_M_S); works: back on the "
    "command within 3 s (decision 30, panels.RECOVERY_RULE) with the limits; Stage 0: the forward bound (round two's "
    "one-step capturability bound; it has no other direction)")
STOP_RULE = (
    "the deceleration a stop holds needs stop references that decelerate at other rates; the run holds one stop "
    "transition from the walk to the stand (transitions.TRANSITIONS 'stop', its speed ramp fixed by "
    "transitions.PATH_HALF_WIDTH; the in-place turns' stops end a turn), so this margin is not run: another stop would "
    "be a new reference, not this plan's.  Recorded beside it: the deceleration the existing stop asks for (its "
    "reference CoM's forward acceleration on the commanded clock, low-passed at 1 Hz as decision 10 reads "
    "acceleration, metrics.lowpass) and, when the row's start-and-stop panel record exists, what its episodes read "
    "(each stop's acceleration peak and whether it stopped within 1.5 strides, panels.START_STOP_RULE)")
DRIVE_SPEED_RULE = (
    "the smallest scale s on every drive's speed limits at which the Fr 0.05 walk (walk/fr05) works: each drive's "
    "corner and no-load speeds x s in the torque-speed envelope (drive.Envelope's speed_scale, as plant.build("
    "speed_scale=s) builds it), and with them the success limit on peak speed (panels.LIMITS_RULE reads the envelope's "
    f"corner speed: control.run's worst_speed_ratio), from 1 down to {SCALE_FAR:g}; peak torques unchanged; Stage 0: "
    "the knee's speed need at Fr 0.05 over its rated speed (plan section 5)")
DRIVE_TORQUE_RULE = (
    "the smallest scale s on every drive's torque at which the nominal walk (walk/nominal) works: each drive's peak "
    "torque x s in the envelope (drive.Envelope's torque_scale, as plant.build(torque_scale=s)) and in the drive "
    "ratings the controller and the judgement read (plant.Plant.drives: the tracker's ankle cap, decision 15, at the "
    "scaled peak; the success limit on each drive's RMS torque at its sustained torque x s, panels.LIMITS_RULE), the "
    "servo gains unchanged (kp = the nominal sustained torque / 0.01 rad: the kp margin sweeps them), from 1 down to "
    f"{SCALE_FAR:g}; Stage 0: the worst drive's sustained-torque ratio on the nominal walk (plan section 5)")
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
    "R2's estimator and its sensors' weight; each episode records the trunk mass its controller worked with and the "
    "plant's), the judgement on the true body "
    f"(metrics.Recorder reads the plant's mass: the GRF over the true weight), on the nominal walk: f {MASS_POINT:g} "
    f"(plan section 5's +10 %: works or not) and the largest f that works from 1 up to {MASS_FAR:g}")
PILE_DAMPING_RULE = (
    "every pile probe's contact damping x f with its stiffness unchanged: each probe's solref (time constant tc, "
    "damping ratio z) becomes (tc / f, z f), so MuJoCo's reference-acceleration damping B = 2 / (dmax tc) is x f and "
    "its stiffness K = 1 / (dmax^2 tc^2 z^2) is unchanged (tc / f must stay above MuJoCo's 2-step floor, checked); "
    "on the carpet walk on low pile (carpet/low: plan section 4's evaluation pile, mu 0.54), from 1 down to "
    f"{DAMPING_LOW:g} and from 1 up to {DAMPING_HIGH:g} (plan section 5; carpet damping is not sourced, A16)")
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
             label=LABEL_RULE, record=RECORD_RULE, resume=RESUME_RULE, sources=SOURCES_RULE, cost=COST_RULE,
             success=tune.SUCCESS_RULE, limits=panels.LIMITS_RULE, recovery=panels.RECOVERY_RULE,
             candidates=control.CANDIDATE_RULE, failed_episode=panels.FAILED_EPISODE_RULE)
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
            "the deceleration a stop holds needs stop references that decelerate at other rates, and the run holds one "
            "stop transition (STOP_RULE)")),
        Margin("drive_speed", "Drive speed", "the smallest envelope speed scale at which the Fr 0.05 walk works",
               searches=(S("walk_fr05", fr05, "speed_scale", 1.0, SCALE_FAR),)),
        Margin("drive_torque", "Drive torque", "the smallest torque scale at which the nominal walk works",
               searches=(S("walk_nominal", walk, "torque_scale", 1.0, SCALE_FAR),)),
        Margin("kp", "Model doubt: servo kp", "the lowest servo kp factor at which the nominal walk works",
               searches=(S("walk_nominal", walk, "kp", 1.0, KP_FAR),)),
        Margin("latency", "Model doubt: latency", "the largest latency at which the nominal walk works",
               searches=(S("walk_nominal", walk, "latency_steps", 0, LATENCY_MAX_STEPS, integer=True),)),
        Margin("mass", "Model doubt: trunk mass", "+10 %, and the largest trunk mass factor at which the nominal walk "
               "works", searches=(S("walk_nominal", walk, "mass", 1.0, MASS_FAR),),
               points=(Point("walk_nominal_x1.1", walk, "mass", MASS_POINT),)),
        Margin("pile_damping", "Model doubt: pile damping", "the lowest and largest pile damping factor at which the "
               "carpet walk on low pile works",
               searches=(S("carpet_low_down", "carpet/low", "pile_damping", 1.0, DAMPING_LOW),
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


def stage0(body: str, margin: str) -> dict:
    """Plan section 5's Stage 0 value(s) of a margin, with what they are."""
    v = STAGE0[body]
    if margin == "grip":
        return dict(values=dict(walk_nominal=v["grip_walk_nominal"], turn=v["grip_turn"], turn_left=v["grip_turn"],
                                turn_right=v["grip_turn"], walk_brisk=v["grip_walk_brisk"]),
                    text="the lowest friction the round-two screens require: nominal walk, 0.35 rad/s turn, brisk walk")
    if margin == "push":
        return dict(values={f"{m}_forward": v["push_forward"] for m in ("standing", "walking")},
                    forward_bound_m_s=v["push_forward"],
                    text="the forward push bound (m/s; round two's one-step capturability bound, forward only)")
    if margin == "stop":
        return dict(values=dict(one_foot_m_s2=v["stop_one_foot"]), text="the deceleration held on one foot (m/s^2)")
    if margin == "drive_speed":
        return dict(values=dict(walk_fr05=v["drive_speed_knee"]),
                    text="the knee's speed need at Fr 0.05 over its rated speed (the envelope speed scale)")
    if margin == "drive_torque":
        return dict(values=dict(walk_nominal=v["drive_torque_sustained"]),
                    text="the worst drive's sustained-torque ratio (the torque scale)")
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
                m0 = float(nominal.body_subtreemass[p.trunk])
                info["controller_model"] = dict(
                    nominal=bool(made) and all(mass == m0 for _, mass in made), trackers=len(made),
                    given_by_row=sum(1 for g, _ in made if g), trunk_mass_kg=[mass for _, mass in made],
                    nominal_trunk_mass_kg=m0, plant_trunk_mass_kg=float(p.m.body_subtreemass[p.trunk]))
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
    """A worker for ``tune.Pool``: ``evaluate((row, condition, kind, value), seed)`` runs one episode of a row on a
    condition with a variant (``LedgerContext``) and returns its summary.  It refuses to start on other panel sources,
    key context or ledger sources than the session's (``SOURCES_RULE``)."""

    def __init__(self, body: str, run_dir: str, mode: str = "ledger", rate: float = panels.RATE,
                 expected_inputs: dict | None = None, rows=(), sources: str | None = None, ledger: str | None = None):
        if ledger is not None and ledger_sources()["sha256"] != ledger:
            raise tune.PoolError("the ledger's sources changed during the session: restore them and sweep again")
        super().__init__(body, run_dir, mode, rate, expected_inputs, rows, sources, traces=None)
        self.ctx = LedgerContext.of(self.ctx)

    def evaluate(self, job, seed) -> dict:
        row, cond, kind, value = job
        c0 = time.process_time()
        info: dict = {}
        try:
            with self.ctx.variant(cond, Variant(kind, value)) as info:
                e = self.rows[row].run(self.ctx, cond, int(seed), (), trace_root=None)
        except Exception:
            e = panels.failed_episode(cond, int(seed), traceback.format_exc())
        return episode_summary(e, time.process_time() - c0, info)


def _worker(body, run_dir, mode, rate, expected_inputs=None, rows=(), sources=None, ledger=None):
    return LedgerWorker(body, run_dir, mode, rate, expected_inputs, rows, sources, ledger)


def _lost(job, reason: str) -> dict:
    (row, cond, kind, value), seed = job
    return episode_summary(panels.failed_episode(cond, int(seed), f"worker lost: {reason}"), None)


# ------------------------------------------------------------------------------------------------
# Evaluations, labels, keys
# ------------------------------------------------------------------------------------------------
def ledger_sources() -> dict:
    """``SOURCES_RULE``."""
    path = Path(__file__).resolve()
    return dict(sha256=plant.sha256(path), file=path.name, rule=SOURCES_RULE)


def _sha(x) -> str:
    return hashlib.sha256(json.dumps(panels._jsonable(x), sort_keys=True).encode()).hexdigest()


def evaluation_entry(cond_key: str, var: Variant, episodes: list[dict], valid: str) -> dict:
    """One evaluation (a condition with a variant on the ledger's seeds): its episodes and outcomes (``WORKS_RULE``)."""
    errors = int(sum(1 for e in episodes if e.get("error")))
    judged = errors == 0 and len(episodes) == len(SEEDS)
    return dict(condition=cond_key, variant=var.describe(), seeds=[int(e["seed"]) for e in episodes],
                episodes=episodes, succeeded=int(sum(bool(e["succeeded"]) for e in episodes)),
                survived=int(sum(not e["fell"] for e in episodes)), errors=errors,
                works=bool(all(e["succeeded"] for e in episodes)) if judged else None,
                survives=bool(not any(e["fell"] for e in episodes)) if judged else None, valid=valid,
                wall_s=float(sum(e.get("wall_s") or 0.0 for e in episodes)),
                cpu_s=float(sum(e.get("cpu_s") or 0.0 for e in episodes)),
                simulated_s=float(sum(e.get("survival_s") or 0.0 for e in episodes)), at=tune._now())


def binding(entry: dict | None, criterion: str) -> dict | None:
    """``LABEL_RULE`` at a search's fails end (an evaluation that did not work)."""
    if entry is None:
        return None
    eps = [e for e in entry["episodes"] if not e.get("error")]
    falls = [e for e in eps if e["fell"]]
    failed = [e for e in eps if not e["fell"] and not e["succeeded"]]
    cands = Counter(c for e in falls for c in (e.get("fall_candidates") or ()))
    if falls:
        falls_label = ("controller-limited (falls with no hardware limit active before the onset)" if not cands else
                       "falls: candidates " + ", ".join(f"{c} x{n}" for c, n in cands.most_common())
                       + " (not attributed: 3 seeds)")
    else:
        falls_label = None
    sl = panels.success_label(eps)
    if criterion == "survival" or not failed:
        label = falls_label
    else:
        label = sl["label"] if len(failed) >= len(falls) else falls_label
    return dict(falls=len(falls), fall_causes=dict(Counter(str(e["fall_cause"]) for e in falls)),
                fall_candidates=dict(cands), success_failures=len(failed), failed_terms=sl["terms"],
                hardware_terms=sl["hardware_terms"], success_label=sl["label"] if failed else None,
                falls_label=falls_label, label=label)


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
    """One margin on one row: its searches' bisections on both criteria and its points, advanced over the
    evaluations known (``advance``)."""

    def __init__(self, row: panels.Row, margin: Margin, conds: dict, point_variants: dict, halvings: int = HALVINGS):
        self.row, self.margin, self.halvings = row, margin, int(halvings)
        self.applicable = {s.name: row.runs(conds[s.condition]) for s in margin.searches}
        self.applicable.update({p.name: row.runs(conds[p.condition]) for p in margin.points})
        self.bisects = {(s.name, c): Bisect(s, c, halvings) for s in margin.searches for c in CRITERIA
                        if self.applicable[s.name]}
        self.point_variants = point_variants
        self.points: dict[str, dict | None] = {p.name: None for p in margin.points if self.applicable[p.name]}
        self.waiting: set = set()  # (search or point, criterion): an evaluation failed to run this session
        self.used: dict[str, tuple] = {}  # evaluation id -> (condition, variant)

    @property
    def done(self) -> bool:
        return (self.margin.not_run is not None
                or (all(b.done for b in self.bisects.values()) and all(v is not None for v in self.points.values())))

    def advance(self, lookup) -> list[tuple]:
        """Feed every bisection and point from ``lookup(condition, variant)`` (an evaluation entry or None) as far as
        the known evaluations go; the (condition, variant) pairs still needed."""
        need = []
        for (name, crit), b in self.bisects.items():
            while not b.done:
                v = b.need()
                var = b.search.variant(v)
                e = lookup(b.search.condition, var)
                self.used[eval_id(b.search.condition, var)] = (b.search.condition, var)
                if e is None:
                    need.append((b.search.condition, var))
                    break
                if e["errors"]:
                    self.waiting.add((name, crit))
                    break
                self.waiting.discard((name, crit))
                b.feed(v, e["works"] if crit == "success" else e["survives"])
        for p in self.margin.points:
            if p.name not in self.points or self.points[p.name] is not None:
                continue
            var = self.point_variants[p.name]
            e = lookup(p.condition, var)
            self.used[eval_id(p.condition, var)] = (p.condition, var)
            if e is None:
                need.append((p.condition, var))
            elif e["errors"]:
                self.waiting.add((p.name, None))
            else:
                self.waiting.discard((p.name, None))
                self.points[p.name] = e
        return need


def eval_id(cond_key: str, var: Variant) -> str:
    return f"{cond_key}|{var.key}"


def _search_results(mr: MarginRun, lookup) -> dict:
    out = {}
    for s in mr.margin.searches:
        if not mr.applicable[s.name]:
            out[s.name] = dict(s.describe(), status="not run", reason=f"{mr.row.name} runs the hard floor only and "
                                                                        f"{s.condition} is not on it")
            continue
        r = dict(s.describe())
        for crit in CRITERIA:
            b = mr.bisects[(s.name, crit)]
            res = b.result()
            if (s.name, crit) in mr.waiting:
                res["waiting"] = "an episode failed to run (WORKS_RULE): the next sweep runs it again"
            fails = None if b.fails is None else lookup(s.condition, s.variant(b.fails))
            res["binding"] = binding(fails, crit) if fails is not None else None
            r[crit] = res
        out[s.name] = r
    return out


def _point_results(mr: MarginRun) -> dict:
    out = {}
    for p in mr.margin.points:
        if p.name not in mr.points:
            out[p.name] = dict(p.describe(), status="not run", reason=f"{mr.row.name} does not run {p.condition}")
            continue
        e = mr.points[p.name]
        var = mr.point_variants[p.name]
        base = dict(p.describe(), variant=var.describe(),
                    same_as_nominal=var == NOMINAL and p.kind != "push_dv")
        if e is None:
            out[p.name] = dict(base, status="waiting" if (p.name, None) in mr.waiting else "incomplete")
            continue
        out[p.name] = dict(base, status="done", works=e["works"], survives=e["survives"], succeeded=e["succeeded"],
                           survived=e["survived"], binding=None if e["works"] else binding(
                               e, "success" if e["survives"] else "survival"))
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
    pts = {n: {k: p.get(k) for k in ("status", "value", "condition", "works", "survives", "succeeded", "survived",
                                     "same_as_nominal")} | {"label": (p.get("binding") or {}).get("label")}
           for n, p in points.items()}
    return dict(searches=rows, points=pts)


def _brief(res: dict | None) -> dict | None:
    if res is None:
        return None
    return dict(status=res.get("status"), threshold=res.get("threshold"), fails=res.get("fails"),
                resolution=res.get("resolution"), search=res.get("search"),
                label=(res.get("binding") or {}).get("label"), evaluated=len(res.get("values") or ()))


# ------------------------------------------------------------------------------------------------
# Rows, records, keys
# ------------------------------------------------------------------------------------------------
def resolve_rows(body: str, run_dir: Path, names) -> list[str]:
    """``ROWS_RULE``: 'R2' -> the body-level G5 selection's R2 row, 'R3' -> R3/42; other names as given."""
    out = []
    for n in names:
        if n == "R2":
            _, info = env.tuned_gains(Path(run_dir), body)
            if info.get("cma_seed") is None or info.get("status") != "passed":
                raise ValueError(f"{body}: no body-level G5 selection that passes ({info.get('status')}); name an R2 "
                                 "row (R2/42 or R2/43)")
            n = f"R2/{int(info['cma_seed'])}"
        elif n == "R3":
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
        self.row_names = resolve_rows(body, self.run_dir, rows)
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
                    var = Variant(**entry["variant"])
                    if entry.get("valid") == self.valid(r, k, var):
                        self.cache[(r, k, var.key)] = entry
        self.runs = {(r, m.name): MarginRun(row, m, self.conds, {p.name: self.point_variants[(m.name, p.name)]
                                                                  for p in m.points}, self.halvings)
                     for r, row in self.rows.items() for m in self.margins}

    def valid(self, row: str, cond_key: str, var: Variant) -> str:
        """``RESUME_RULE``'s evaluation key."""
        return _sha(dict(panel=self.panel_keys[(row, cond_key)], variant=var.describe(), seeds=list(SEEDS),
                         ledger=self.ledger["sha256"]))

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
        return lambda cond_key, var: self.cache.get((row, cond_key, var.key))

    def advance(self) -> dict:
        """Every margin run fed from the evaluations known; the evaluations still needed: (row, condition, variant
        key) -> (row, condition, variant)."""
        need = {}
        for (r, _), mr in self.runs.items():
            for k, var in mr.advance(self.lookup(r)):
                need[(r, k, var.key)] = (r, k, var)
        return need

    def record(self, row: str, m: Margin, prov: dict, session: dict) -> dict:
        """The margin's record (``RECORD_RULE``)."""
        mr = self.runs[(row, m.name)]
        lookup = self.lookup(row)
        searches, points = _search_results(mr, lookup), _point_results(mr)
        evals = {i: self.cache[(row, k, var.key)] for i, (k, var) in sorted(mr.used.items())
                 if (row, k, var.key) in self.cache}
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
                   evaluations=evals, rules=RULES, key=self.key(row, m), provenance=prov,
                   cost=dict(evaluations=len(evals), episodes=int(sum(len(e["episodes"]) for e in evals.values())),
                             simulated_s=float(sum(e["simulated_s"] for e in evals.values())),
                             wall_s=float(sum(e["wall_s"] for e in evals.values())),
                             cpu_s=float(sum(e["cpu_s"] for e in evals.values()))),
                   session=dict(started_at=session["started_at"]), written_at=tune._now())
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
    session = dict(started_at=tune._now(), rows=list(ses.row_names), margins=[m.name for m in ses.margins],
                   workers=int(workers), cut_s=cut_s, halvings=ses.halvings, seeds=list(SEEDS), rounds=0,
                   episodes_run=0, evaluations_reused=len(ses.cache), skipped=[], incidents=[])
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
            jobs = [((r, ses.conds[k], var.kind, var.value), s) for (r, k, var) in need.values() for s in SEEDS]
            results = pool.map(jobs, failed=_lost)
            session["rounds"] += 1
            session["episodes_run"] += len(jobs)
            n = len(SEEDS)
            for i, (key, (r, k, var)) in enumerate(need.items()):
                entry = evaluation_entry(k, var, results[n * i: n * (i + 1)], ses.valid(r, k, var))
                ses.cache[key] = entry
                if entry["errors"]:
                    failed.add(key)
            write_all()
            log(f"{body} round {session['rounds']}: {len(need)} evaluations ({len(jobs)} episodes): "
                + "; ".join(f"{r} {k} {var.key}" for r, k, var in list(need.values())[:6])
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
                                    f"works {p.get('works')}, survives {p.get('survives')}"
                                    + (f" ({p['label']})" if p.get("label") else "")))
    return lines


# ------------------------------------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------------------------------------
def dry_run(body: str, out: Path, rows=DEFAULT_ROWS, names=None, inputs: Path | None = None,
            cut_s: float | None = None, halvings: int = HALVINGS, measure: bool = True, costs: dict | None = None,
            workers: int = 1, log=print) -> dict:
    """Every search and point the sweep would run, the evaluations it can still need after replaying the recorded
    ones, and their cost (``COST_RULE``); runs nothing (except the cost measurement)."""
    ses = Session(body, out, rows, names, inputs, cut_s, halvings)
    ses.advance()
    lines = []
    for (r, mname), mr in ses.runs.items():
        m = mr.margin
        up = ses.up_to_date(r, m)
        if m.not_run:
            lines.append(dict(row=r, margin=m.name, search=None, status="not run", reason=m.not_run, values_min=0,
                              values_max=0, condition=None, floor=None, seconds=0.0))
            continue
        for s in m.searches:
            if not mr.applicable[s.name]:
                lines.append(dict(row=r, margin=m.name, search=s.name, status="not applicable", values_min=0,
                                  values_max=0, condition=s.condition, floor=ses.conds[s.condition].floor, seconds=0.0))
                continue
            left = {c: 0 if up else s.max_values(halvings) - len(mr.bisects[(s.name, c)].trail) for c in CRITERIA}
            for c in CRITERIA:
                if mr.bisects[(s.name, c)].done:
                    left[c] = 0
            lines.append(dict(row=r, margin=m.name, search=s.name, condition=s.condition,
                              floor=ses.conds[s.condition].floor, interval=[s.nominal, s.far], kind=s.kind,
                              status="up to date" if up else "to run",
                              known_values={c: len(mr.bisects[(s.name, c)].trail) for c in CRITERIA},
                              values_min=max(left.values()), values_max=sum(left.values()),
                              seconds=ses.resolved[s.condition]["seconds"]))
        for p in m.points:
            if p.name not in mr.points:
                continue
            todo = 0 if up or mr.points[p.name] is not None else 1
            lines.append(dict(row=r, margin=m.name, search=p.name, condition=p.condition,
                              floor=ses.conds[p.condition].floor, point=p.value, kind=p.kind,
                              variant=mr.point_variants[p.name].describe(),
                              status="up to date" if not todo else "to run",
                              values_min=todo, values_max=todo, seconds=ses.resolved[p.condition]["seconds"]))
    floors = sorted({x["floor"] for x in lines if x.get("floor") and x["values_max"]})
    if costs is None and measure and floors:
        costs = panels.measure_costs(body, ses.run_dir, floors, log=log)
    fl = (costs or {}).get("floors") or {}
    total = Counter()
    for x in lines:
        n_min, n_max = len(SEEDS) * x["values_min"], len(SEEDS) * x["values_max"]
        c = (fl.get(x.get("floor")) or {}).get("cpu_per_sim_s")
        x.update(episodes_min=n_min, episodes_max=n_max, simulated_s_max=n_max * x["seconds"],
                 core_h_max=None if c is None else n_max * x["seconds"] * c / 3600,
                 core_h_min=None if c is None else n_min * x["seconds"] * c / 3600)
        total["episodes_min"] += n_min
        total["episodes_max"] += n_max
        total["simulated_s_max"] += x["simulated_s_max"]
        for k in ("core_h_min", "core_h_max"):
            total[k] += x[k] or 0.0
        if x["search"] is not None:
            log(f"{x['row']:7s} {x['margin']:13s} {x['search']:22s} {x['condition'] or '':28s} "
                f"{x['episodes_min']:3d}-{x['episodes_max']:3d} episodes x {x['seconds']:6.2f} s  "
                + (f"{x['core_h_max']:.3f} core-h at most" if x["core_h_max"] is not None else "cost not measured")
                + f"  [{x['status']}]")
        else:
            log(f"{x['row']:7s} {x['margin']:13s} {x['status']}: {x['reason']}")
    w_n = max(1, int(workers))
    log(f"{body}: {total['episodes_min']}-{total['episodes_max']} episodes ({len(ses.cache)} evaluations recorded "
        f"under this run's keys), at most {total['simulated_s_max'] / 3600:.2f} simulated h, "
        + (f"{total['core_h_min']:.2f}-{total['core_h_max']:.2f} core-h (CPU), about "
           f"{total['core_h_max'] / w_n:.2f} h of wall clock at most with {w_n} worker{'s' if w_n > 1 else ''}"
           if fl else "cost not measured"))
    return dict(body=body, rule=COST_RULE, rows=list(ses.row_names), margins=[m.name for m in ses.margins],
                lines=lines, costs=costs, totals=dict(total), workers=w_n, evaluations_recorded=len(ses.cache))


__all__ = ["Bisect", "DEFAULT_ROWS", "HALVINGS", "LedgerContext", "LedgerWorker", "MARGIN_NAMES", "Margin", "NOMINAL",
           "Point", "PushedBuilt", "RULES", "SEEDS", "STAGE0", "Search", "Session", "Variant", "binding", "dry_run",
           "margins", "record_path", "resolve_rows", "select_margins", "stage0", "stop_reference", "summarize",
           "sweep", "worse"]
