"""Stage 1b, the speed round (``STAGE1B_PLAN.md`` sections 3 and 4): the speed fit of the R2 tracker (R2s), its
selection ("speed G5") and the validation rows of the behaviour panels, on any registered body (``BODIES``).

Run from hardware/trex_form_study (the fit's venv; ``stage1/runs/2026-11-10-speed/run_speed_round.sh`` holds the
round's exact commands per arm: arm S reads its references from Stage 1's run, ``--inputs stage1/runs/2026-11-08``)::

    python -m designlab.stage1 speedfit  --body 60-N-T-10 [--seed 42] [--evals 360] [--workers 1]
                                         [--start-from PICK] [--resume] [--stage1 DIR] [--inputs DIR]
                                         --out stage1/runs/2026-11-10-speed
    python -m designlab.stage1 speedg5   --body 60-N-T-10 [--seeds 42] [--workers 1] [--at-generation G]
                                         [--partial] [--episodes 20] [--inputs DIR] --out stage1/runs/2026-11-10-speed
    python -m designlab.stage1 eval      --body 60-N-T-10 --row-module designlab.stage1.speedfit
                                         --rows R2s/42,R2s/42@10ms,R2@10ms --behaviors walk,turn --attribute
                                         [--workers 4] [--inputs DIR] --out stage1/runs/2026-11-10-speed
    python -m designlab.stage1 speedcost --body 60-N-T-10 [--inputs DIR] [--out DIR]
    python -m designlab.stage1 speedreport --out stage1/runs/2026-11-10-speed [--body B] [--dest DIR]  (speedreport)

The objective (``OBJECTIVE``, ``stage1-tune-speed-v1``) is Stage 1's fit (``tune``, ``stage1-tune-v2``, decision 22)
with the episode mix widened (``MIX_RULE``): every evaluation runs, from the settled stance on hard floor at mu 0.9,

* the three episodes of Stage 1's mix exactly (``tune.episode_plan``, ``tune.build_spec``, ``tune.measure``): the
  nominal walk, the 0.35 rad/s walking turn and the 0.5 rad/s in-place turn, the turns' directions by the episode
  seed's parity (``tune.directions``);
* the brisk walk and the Froude 0.05 walk, built and judged as the panels build and judge them (decision 31,
  ``SPEED_WALK_RULE``): the panels' plans (``panels.Planner``: start, a nominal stride, the speed changes, the target
  gait to the end), ``tune.build_spec(plan=, gait=)``, ``tune.measure`` over the episode's last 10 s, cadence, stride
  and peak GRF dropped (judged at nominal only) and no start or stop terms (their start goes to the nominal speed and
  they end in their gait).

Each episode draws a servo delay of 0-5 physics steps (0-10 ms) from its episode seed (``LATENCY_RULE``), applied
through env.py's latency model as the margin ledger's latency margin applies it (``servo_delay``: every servo target
reaches the servos k physics steps of 2 ms after the controller sets it).  The seeds (``SEED_RULE``), the penalty
structure (``SCORE_RULE``: survival S, performance P, J = S + P, ``tune.score`` over the five episodes), the terms and
their scales (``tune.TERMS_RULE``), the failures (``FAILURE_RULE``) and the held capture margin (``tune.FIXED_RULE``)
are Stage 1's.

CMA-ES (``tune.CMA``, ``CMA_SETTINGS``): popsize 12, 360 evaluations (30 generations), seed 42, from the body's
Stage 1 G5 selection (``START_RULE``: ``env.tuned_gains``; a faster-knee body starts from its pick's, the pick being
its candidate id) with Stage 1's step size (``SIGMA0``, ``SIGMA0_RULE``); generation g on training episode
``tune.training_order(seed)[g]``, every member on it; checkpointed after every generation
(``<out>/<body>/speedfit/<seed>/cma.json`` and ``evals.jsonl``), ``--resume`` continues exactly (``RESUME_RULE``),
``--workers`` evaluates a generation in that many processes (``tune.Pool``).  The start is recorded once per body
(``<out>/<body>/speedfit/start.json``).

Selection, "speed G5" (``SELECTION_RULE``, ``speed_g5``): the start and the CMA-ES mean after each third of the
generations judged (10, 20 and 30 of a full run; ``JUDGED_RULE`` for a run stopped by the budget), each on the 20
validation episodes 2000-2019 of the speed mix, ranked by ``tune.selection_key`` (fewest falls, then the lower mean
S, then the lower mean P); per kind the survive and success counts (and by delay) are recorded in
``<out>/<body>/speedg5.json``; every validation evaluation is kept as it finishes, so a rerun continues
(``G5_RESUME_RULE``, ``<out>/<body>/speedg5_evals.jsonl``).

Rows (``ROWS_RULE``; ``panels.register_rows``, ``eval --row-module designlab.stage1.speedfit``): ``R2s/<s>`` is fit
seed s's speed G5 selection, ``R2s/<s>@<ms>ms`` the same with ms of servo delay, ``R2@10ms`` the fit's start (Stage
1's body-level G5 selection) at 10 ms and ``R2@0ms`` without delay; each runs only ``ROW_CONDITIONS`` (the nominal,
brisk and Fr 0.05 walks and both directions of both target turns, on hard floor at mu 0.9) on episodes 3042-3061,
with records under ``<out>/<body>/eval/<row>/<behavior>/<condition>.json``; their falls are attributed by decision 20
as R2's are (``eval --attribute``; importing this module adds their controllers to ``panels.ATTRIBUTED``).  The
panels' ``summary.json`` reads Stage 1's verdict rule (R2/42 and R2/43), which does not apply to these rows: plan
section 4's reading of them is ``speedreport``'s.

This module is not among the tuner's or the panels' sources (``tune.TUNER_SOURCES_RULE``,
``panels.PANEL_SOURCES_RULE``), so Stage 1's fit, G5 and panel records are unchanged by it; it enters only its own
records' keys (``SOURCES_RULE``, ``Row.sources``).  Importing it changes two things in the panels for the speed rows'
bodies, both outside the panels' sources: the speed controllers are attributed (above), and ``panels.g7`` reports a
body without G7 figures (a faster-knee body) as not run instead of raising (``G7_RULE``).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import time
import traceback
from collections import Counter, deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import BODIES, STUDY, control, env, metrics, panels, plant, tune
from .refs import Reference

OBJECTIVE_NAME = "stage1-tune-speed-v1"
#: The round's run directory (STAGE1B_PLAN.md section 5) and Stage 1's, where the starts' G5 records are.
RUN = STUDY / "stage1" / "runs" / "2026-11-10-speed"
STAGE1_RUN = tune.RUN
SCHEMA = 1
MODE = tune.MODE
RATE = tune.COMMAND_RATE
BUDGET = 360
SEEDS = (42,)
POPSIZE = tune.POPSIZE
SIGMA0 = tune.SIGMA0
TRAIN_EPISODES = tune.TRAIN_EPISODES
VALIDATION_EPISODES = tune.VALIDATION_EPISODES
VALIDATION_RUNS = tune.G5_RUNS
#: The speed mix: Stage 1's three episodes, then the two speed walks; an episode's index here is its slot j.
SPEED_WALKS = ("brisk", "fr05")
EPISODES = tune.EPISODES + SPEED_WALKS
CONDITIONS = dict(tune.CONDITIONS, brisk="brisk walk", fr05="Fr 0.05 walk")
#: The panel conditions the speed walks are (``SPEED_WALK_RULE``).
PANEL_WALKS = {"brisk": "walk/brisk", "fr05": "walk/fr05"}
#: Servo delay (``LATENCY_RULE``): physics steps (2 ms), inclusive; the purpose index of its seed.
LATENCY_STEPS = (0, 5)
LATENCY_PURPOSE = 3
#: The objective's terms of the speed walks: the walk's without the 'at nominal' ones (``SPEED_WALK_RULE``).
TRACKING = dict(tune.TRACKING, **{k: tune.TRACKING["walk"] for k in SPEED_WALKS})
LIMITS = dict(tune.LIMITS, **{k: tuple(t for t in tune.LIMITS["walk"] if t not in panels.NOMINAL_ONLY_TERMS)
                              for k in SPEED_WALKS})
SUCCESS = dict(tune.SUCCESS, **{k: tuple(t for t in tune.SUCCESS["walk"] if t not in panels.NOMINAL_ONLY_TERMS)
                                for k in SPEED_WALKS})
#: The validation rows' conditions (``ROWS_RULE``; STAGE1B_PLAN.md section 3's validation panels).
ROW_CONDITIONS = ("walk/nominal", "walk/brisk", "walk/fr05", "turn/walking_0.35_left", "turn/walking_0.35_right",
                  "turn/in_place_0.50_left", "turn/in_place_0.50_right")
ROW_DELAY_MS = 10
#: The delays of the start's rows R2@<ms>ms (``ROWS_RULE``): 10 ms, the baseline for delay; 0 ms, a faster-knee
#: body's start without a refit.
R2_ROW_DELAYS_MS = (0, ROW_DELAY_MS)
LATENCY_MAX_ROW_MS = 20
#: The validation rows' controllers (``panels.controller_of``), attributed by decision 20 as R2's rows are
#: (``ROWS_RULE``).
SPEED_CONTROLLERS = ("R2s",) + tuple(f"R2@{ms}ms" for ms in R2_ROW_DELAYS_MS)
#: Speed G5's log of validation evaluations, beside speedg5.json (``G5_RESUME_RULE``).
G5_LOG = "speedg5_evals.jsonl"
#: The environment variable naming the speed run directory a row reads its gains from (``ROWS_RULE``).
SPEED_RUN_ENV = "STAGE1B_SPEED_RUN"
#: Plan section 4's pass rule, read on the validation episodes (reported, not a gate here).
PASS_SURVIVE, PASS_SUCCEED = 20, 17

# ------------------------------------------------------------------------------------------------
# Rules (each copied into the records)
# ------------------------------------------------------------------------------------------------
MIX_RULE = (
    "every evaluation is five episodes from the settled stance on hard floor (mu 0.9, the plant as compiled), in this "
    "order (j = 0-4, each episode's slot): the nominal walk, the 0.35 rad/s walking turn and the 0.5 rad/s in-place "
    "turn, exactly as Stage 1's mix runs them (tune.episode_plan, tune.build_spec, tune.measure; the turns' "
    "directions by the episode seed's parity, tune.directions: even, the walking turn left and the in-place turn "
    "right), then the brisk walk and the Fr 0.05 walk (SPEED_WALK_RULE); every speed commanded "
    f"{100 * tune.COMMAND_MARGIN:g} % above its target (tune.COMMAND_RATE); the tracker R2 on the estimate with "
    "Stage 1's start noise and sensor noise (tune.START_JOINT_NOISE_RAD, tune.START_ROOT_VELOCITY_NOISE_M_S, "
    "estimator.SensorNoise defaults) and each episode's servo delay (LATENCY_RULE)")
SPEED_WALK_RULE = (
    "the brisk and Fr 0.05 walks are the panels' walk/brisk and walk/fr05 (STAGE1_PLAN.md decision 31, "
    "panels.WALK_RULE): "
    "the plan panels.Planner resolves (start, 1 nominal stride, nominal_to_brisk, [1 brisk stride, brisk_to_fr05,] K "
    "strides of the target gait, K the fewest that keep the last 10 s in its steady part even with the clock 5 % "
    "slow, plus half a stride), built by tune.build_spec(plan=, gait=) as panels.Context.built builds it, ending in "
    "the gait without a stop; judged by tune.measure over the episode's last 10 s (window=, as panels.judge reads "
    "it), when the episode ran its planned length (to within a physics step: a walk that fell, or one cut short in a "
    "smoke check, cut_s, has no such window and scores every term 1, as a fallen or cut nominal walk does); success "
    "terms tune.SUCCESS['walk'] without cadence, stride and peak GRF (panels.NOMINAL_ONLY_TERMS: 'at "
    "nominal'), as panels.terms_of; the objective's terms are tune.measure's with the same three limit terms "
    "dropped (tracking: DCM error, speed against the target gait's command, heading change; limits: the window "
    "limits but GRF, the 10 s mean speed under the target, drives and range) and no start or stop terms (the start "
    "goes to the nominal speed, and there is no stop)")
LATENCY_RULE = (
    f"each episode's servo delay is k physics steps, k uniform on {LATENCY_STEPS[0]}-{LATENCY_STEPS[1]} "
    f"(0-{2 * LATENCY_STEPS[1]} ms) drawn as numpy default_rng(tune.derived_seed(episode seed, j, "
    f"{LATENCY_PURPOSE})).integers({LATENCY_STEPS[0]}, {LATENCY_STEPS[1] + 1}), j the episode's slot (MIX_RULE), "
    "so the five episodes of an evaluation draw independently and every member of a generation gets the same delays "
    "(common random numbers); applied by servo_delay, env.py's latency model as the margin ledger's latency margin "
    "applies it (ledger.LATENCY_RULE, ledger._latency, env.Stage1Env._latency_on): every servo target reaches the "
    "servos k physics steps (2 ms each) after the controller sets it, through a queue of the last k + 1 targets "
    "refilled with the reset's targets (the home stance) at the episode's reset; the controller, its estimator and "
    "the sensors are unchanged (the delay is the plant's); k = 0 changes nothing (the episode is Stage 1's)")
SEED_RULE = (
    f"training episodes {TRAIN_EPISODES[0]}-{TRAIN_EPISODES[1] - 1} in tune.training_order(seed) (generation g on "
    "episode 1000 + permutation(1000)[g] from default_rng([seed, 1]), every member on it), validation episodes "
    f"{VALIDATION_EPISODES[0]}-{VALIDATION_EPISODES[1] - 1}, as Stage 1's fit; an episode of slot j of episode seed s "
    "draws its start perturbation from tune.derived_seed(s, j, 0), its sensor noise from tune.derived_seed(s, j, 1) "
    f"and its servo delay from tune.derived_seed(s, j, {LATENCY_PURPOSE}); so at zero delay the speed mix's first "
    "three episodes are Stage 1's episodes of the same seed bit for bit")
SCORE_RULE = (
    "Stage 1's penalty structure (tune.score) over the five episodes: J = S + P; S = sum over the fallen episodes of "
    "(1 + the share of the episode not survived); P = 0.5 T + 0.5 L, T the mean of every episode's tracking terms, L "
    "the mean over the behaviour groups (walk, walking_turn, in_place_turn, brisk, fr05, and the start and stop groups "
    "of the nominal walk and the walking turn) of (max + mean) / 2 of the group's limit terms; a fallen episode "
    "scores 1 on every term; the CMA-ES fitness adds the box repair (tune.repair_penalty); capture_margin held at 0.0 "
    "(tune.FIXED_RULE)")
FAILURE_RULE = (
    "tune.FAILURE_RULE on the speed mix: an evaluation that raises scores as the worst case (every episode a fall at "
    f"its start: S = 2 per episode, every term 1, J = {2 * len(EPISODES) + 1}), with its traceback; an episode during "
    "which MuJoCo warned counts as a fall at its start; a worker that dies or hangs is replaced and its evaluation run "
    f"again, up to {tune.ATTEMPTS} attempts, then scored as the worst case with the reason")
START_RULE = (
    "the fit starts from a Stage 1 G5 selection: env.tuned_gains (env.GAINS_RULE) on <Stage 1 run>/<start "
    "body>/tune.json, the start body being the body itself when it has a Stage 1 fit (<Stage 1 run>/<body>/tune.json) "
    "and otherwise (a faster-knee body) the pick it is built from, its round-two candidate id (the registry's "
    "candidate_id); --start-from may name that start body and no other; refused unless that selection passed G5 and "
    "holds the fixed gains (tune.FIXED_GAINS); the start is recorded once per body (speedfit/start.json: the record's "
    "path and sha256, the candidate and its gains) and every seed and session must start from it")
SIGMA0_RULE = (
    f"sigma0 {SIGMA0:g} of each searched gain's bound range (the [0, 1] scaling): Stage 1's (tune.SIGMA0), as "
    "STAGE1B_PLAN.md section 3 has it (Stage 1's fit with the wider mix, otherwise unchanged); the same on every "
    "body, so the arms differ only by their bodies; the start stays a selection candidate (SELECTION_RULE), so a "
    "fit that finds nothing better keeps it")
SOURCES_RULE = (
    "sha256 over the designlab/stage1 modules the speed fit imports, followed as tune.TUNER_SOURCES_RULE follows the "
    "tuner's (from speedfit.py and the package's __init__.py, transitively, module- and function-level imports, read "
    "with ast) and hashed as plant.SOURCES_RULE hashes files, the module list recorded beside; every session and "
    "worker must run the same (a worker that computes another refuses to start), a resume and the selection refuse "
    "a fit made on other sources; the designlab modules outside designlab/stage1 are named by each session's commit "
    "and dirty paths (plant.provenance)")
INPUTS_RULE = (
    "tune.INPUTS_RULE on the speed mix's files (mix_files: Stage 1's mix files and the speed walks' references and "
    "transitions: refs/brisk.npz, refs/fr05.npz, transitions/nominal_to_brisk.npz and transitions/brisk_to_fr05.npz), "
    "checked against the sha256 that refs.json and transitions.json record; a resume and the selection refuse other "
    "inputs than the fit's first session's")
RESUME_RULE = (
    "the state is checkpointed after every generation (cma.json, written atomically after the generation's "
    "evaluations are appended to evals.jsonl and synced); --resume continues from it exactly (tune.CMA.from_state) and "
    "drops evaluations logged after the checkpoint; refused if the settings (body, seed, mode, rate, objective, CMA "
    "settings, delay rule, start, cut), the speed fit's sources (SOURCES_RULE), the library versions or the inputs "
    "(INPUTS_RULE) differ from the run's first session; an existing run is never restarted over")
SELECTION_RULE = (
    "speed G5: per fit seed the candidates are the start (START_RULE) and the CMA-ES mean (projected into the bounds) "
    "after each third of the generations judged (tune.candidate_generations: 10, 20 and 30 of a full run; "
    "JUDGED_RULE), each evaluated on the validation episodes 2000-2019 of the speed mix (their delays drawn as "
    "LATENCY_RULE draws them); ranked by tune.SELECTION_RULE's key (tune.selection_key): fewer falls over the "
    "validation episodes first, then the lower mean S, then the lower mean P, the lower mean J and then the earlier "
    "candidate breaking ties; the first is selected; per kind its survive and success counts are recorded, overall "
    "and by delay, with plan section 4's pass rule (20/20 survive, at least 17/20 succeed) read on them as a "
    "reported value, not a gate: those counts pool delays of 0-10 ms, so they are not the 0 ms or the 10 ms answer "
    "(the round's verdicts come from the validation rows, STAGE1B_PLAN.md section 4, speedreport); a fit made on "
    "other sources, inputs or start, or on cut episodes (a smoke check), is not judged; the validation evaluations "
    "are kept as they finish (G5_RESUME_RULE)")
JUDGED_RULE = (
    "a fit is judged at its last generation once it is done (the budget's 30), or at --at-generation G "
    "(STAGE1B_PLAN.md section 5: if the fits run long, every arm is judged at the last generation all of them "
    "reached), its candidates the means after each third of G; --partial ranks an unfinished run's candidates at its "
    "last generation as a smoke check: recorded with partial true and judged false, so no row takes its selection "
    "(r2s_gains), whatever the number of validation episodes")
G5_RESUME_RULE = (
    f"speed G5 appends every validation evaluation to <out>/<body>/{G5_LOG} as it finishes (one JSON line, flushed "
    "and synced; with several workers, after every two evaluations per worker), keyed by the candidate's gains "
    "(sha256 of their values as JSON), the episode, the body, mode and rate, the speed fit's sources (SOURCES_RULE), "
    "the inputs' sha256 (INPUTS_RULE) and the library versions; a rerun reuses every line whose key matches and runs "
    "only the evaluations missing, so a restart loses at most those in flight; an evaluation that failed to run or "
    "raised (FAILURE_RULE) is not kept and runs again; a line cut short when the process died is dropped; reused and "
    "new results are the same JSON values, so speedg5.json does not depend on how many sessions made it")
ATTRIBUTION_RULE = (
    "the validation rows' falls are attributed by decision 20 as Stage 1's R2 rows' are (panels.PANEL_ATTRIBUTION_RULE: "
    "eval --attribute lifts each relaxable candidate of a record's falls over the condition's 20 starts when it has "
    f"at least {panels.ATTRIBUTION_MIN_FALLS} falls): importing this module adds the speed controllers "
    f"({', '.join(SPEED_CONTROLLERS)}) to panels.ATTRIBUTED, read at call time, without a change to panels.py; the "
    "lifted runs keep the row's servo delay (DelayedRow.episode wraps the panels' episode, relax included)")
G7_RULE = (
    "G7 (STAGE1_PLAN.md section 3 step 8) is Stage 1's gate on its three bodies (panels.G7_PLAN): importing this "
    "module makes panels.g7 raise FileNotFoundError for a body without G7 figures (a faster-knee body), which "
    "panels.summarize records as 'G7 not run' with the reason, instead of a KeyError after the rows have run; the "
    "three bodies' G7 is unchanged")
ROWS_RULE = (
    "rows of the behaviour panels (panels.Row, registered with panels.register_rows): R2s/<s> is the tracker R2 with "
    "fit seed s's speed G5 selection (<speed run>/<body>/speedg5.json seeds[s].selected_gains; refused unless judged, "
    "JUDGED_RULE), no delay; R2s/<s>@<ms>ms the same with ms of servo delay (whole 2 ms steps, at most "
    f"{LATENCY_MAX_ROW_MS} ms); R2@{ROW_DELAY_MS}ms the speed fit's start (speedfit/start.json: Stage 1's "
    f"body-level G5 selection, of the pick for a faster-knee body) at {ROW_DELAY_MS} ms, the baseline for delay, "
    "and R2@0ms the same without delay (on a faster-knee body its pick's gains without a refit, which tells a trade "
    "from the body's effect; on a pick it repeats Stage 1's R2 row of the selection's seed); the delay is "
    "servo_delay's (LATENCY_RULE) for the whole episode; each row runs only "
    f"{', '.join(ROW_CONDITIONS)} (hard floor, mu 0.9) on the panels' episodes 3042-3061 with the panels' seeds, "
    "judgement and records (<out>/<body>/eval/<row>/<behavior>/<condition>.json), and its falls are attributed "
    f"(ATTRIBUTION_RULE; run eval --attribute); the speed run, where the gains are read, is {SPEED_RUN_ENV} if set, "
    "else the round's run directory (RUN, stage1/runs/2026-11-10-speed), whatever --inputs the panels read; the rows' "
    "keys name this module's and the gains' files by sha256 (Row.sources, Row.describe); the panels' "
    "eval/summary.json and its verdict column are Stage 1's verdict rule (VERDICT_ROWS: R2/42 and R2/43) and do not "
    "apply to these rows: plan section 4's reading of them is speedreport's (speedreport.READING_RULE)")


# ------------------------------------------------------------------------------------------------
# The mix
# ------------------------------------------------------------------------------------------------
def directions(episode_seed: int) -> dict:
    """The episodes' directions of an episode seed: the turns' by its parity (``tune.directions``), none for the
    walks."""
    return dict(tune.directions(episode_seed), **{k: None for k in SPEED_WALKS})


def latency_steps(episode_seed: int, j: int) -> int:
    """The servo delay (physics steps) of slot ``j`` of episode seed ``episode_seed`` (``LATENCY_RULE``)."""
    rng = np.random.default_rng(tune.derived_seed(episode_seed, j, LATENCY_PURPOSE))
    return int(rng.integers(LATENCY_STEPS[0], LATENCY_STEPS[1] + 1))


def latencies(episode_seed: int) -> dict:
    """Each episode kind's servo delay (physics steps) on ``episode_seed``."""
    return {kind: latency_steps(episode_seed, j) for j, kind in enumerate(EPISODES)}


def panel_condition(kind: str) -> panels.Condition:
    """The panel condition a speed walk is (``SPEED_WALK_RULE``)."""
    return next(c for c in panels.conditions() if c.key == PANEL_WALKS[kind])


def mix_files() -> list[str]:
    """The reference and transition files the speed evaluator loads (``INPUTS_RULE``)."""
    out = set(tune.mix_files())
    for kind in SPEED_WALKS:
        out.update(panels.files_of(panel_condition(kind)))
    return sorted(out)


TERM_ORDER = tuple(f"{k}.tracking.{n}" for k in EPISODES for n in TRACKING[k]) + tuple(
    f"{k}.limits.{n}" for k in EPISODES for n in LIMITS[k]) + tuple(
    f"{k}.{g}.{n}" for k in tune.WITH_START_STOP for g, names in (("start", tune.START_TERMS),
                                                                   ("stop", tune.STOP_TERMS)) for n in names)


def term_vector(eps: dict) -> list:
    out = []
    for name in TERM_ORDER:
        kind, group, n = name.split(".")
        key = dict(tracking="tracking", limits="limits", start="start_terms", stop="stop_terms")[group]
        out.append(round(float(eps[kind][key][n]), 6))
    return out


# ------------------------------------------------------------------------------------------------
# Servo delay
# ------------------------------------------------------------------------------------------------
@contextmanager
def servo_delay(p: plant.Plant, steps: int):
    """``LATENCY_RULE``'s delay on ``p`` for the block: every servo target reaches the servos ``steps`` physics steps
    after ``p.command`` sets it (the margin ledger's ``ledger._latency`` and env.py's ``Stage1Env._latency_on``: a queue
    of the last ``steps`` + 1 targets, refilled with the targets ``p.reset`` leaves at every reset); yields the queue
    (None for 0 steps, which changes nothing) and restores ``p`` when the block ends, whatever happens in it."""
    k = int(steps)
    if k != steps or k < 0:
        raise ValueError(f"servo delay {steps!r}: whole physics steps, at least 0")
    if k == 0:
        yield None
        return
    if "command" in p.__dict__ or "reset" in p.__dict__:
        raise RuntimeError(f"{p.body}: the plant's command or reset is already replaced (another delay or variant)")
    command, reset = type(p).command.__get__(p), type(p).reset.__get__(p)
    queue = deque(maxlen=k + 1)

    def fill():
        queue.clear()
        queue.extend(p.d.ctrl.copy() for _ in range(k))

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
    try:
        yield queue
    finally:
        p.__dict__.pop("reset", None)
        p.__dict__.pop("command", None)


# ------------------------------------------------------------------------------------------------
# Episodes and evaluations
# ------------------------------------------------------------------------------------------------
def measure_speed_walk(kind: str, spec: tune.EpisodeSpec, rec: metrics.Recorder, out: dict, dt: float) -> dict:
    """A speed walk's measures, objective terms and success (``SPEED_WALK_RULE``): ``tune.measure`` over the
    episode's last 10 s, as ``panels.judge`` reads a walk that ends in its gait, when it ran its planned length to
    within a physics step ``dt``; a walk cut short (``Evaluator.cut_s``) or fallen has no such window and is unjudged
    (every term 1)."""
    window = None
    if not out["fell"] and rec.rows and float(out["seconds"]) >= spec.seconds - float(dt):
        tb = float(rec.t[-1])
        window = (tb - tune.WALK_WINDOW_S, tb)
    res = tune.measure(spec, rec, out, None, window=window)
    res.update(kind=kind, gait=panel_condition(kind).gait, judged_window_s=None if window is None else list(window),
               start=None, stop_reading=None)
    res["tracking"] = {n: res["tracking"][n] for n in TRACKING[kind]}
    res["limits"] = {n: res["limits"][n] for n in LIMITS[kind]}
    res["start_terms"], res["stop_terms"] = {}, {}
    res["success"] = {n: res["success"][n] for n in SUCCESS[kind]}
    res["succeeded"] = bool(all(res["success"].values()))
    return res


def failed_episode(kind: str, direction: str | None, cause: str, planned_seconds: float | None = None) -> dict:
    """An episode scored as the worst case (``FAILURE_RULE``)."""
    e = tune.failed_episode("walk" if kind in SPEED_WALKS else kind, direction, cause, planned_seconds)
    if kind in SPEED_WALKS:
        e.update(kind=kind, gait=panel_condition(kind).gait, tracking={n: 1.0 for n in TRACKING[kind]},
                 limits={n: 1.0 for n in LIMITS[kind]}, start_terms={}, stop_terms={},
                 success={n: False for n in SUCCESS[kind]})
    e["latency_steps"] = None
    return e


def failed_evaluation(episode_seed: int, reason: str, planned: dict | None = None, wall_s: float = 0.0) -> dict:
    """An evaluation scored as the worst case (``FAILURE_RULE``)."""
    dirs = directions(episode_seed)
    eps = {k: failed_episode(k, dirs[k], "evaluation failed", (planned or {}).get(k)) for k in EPISODES}
    return dict(tune.score(eps), episode_seed=int(episode_seed), directions=dirs, latency_steps=latencies(episode_seed),
                episodes=eps, terms=term_vector(eps), wall_s=float(wall_s), cpu_s=0.0, simulated_s=0.0,
                error=str(reason))


class Evaluator:
    """One body's plant and the speed mix's episode tracks, built once; ``evaluate`` runs the mix (``MIX_RULE``) for a
    gain vector and an episode seed, every episode on a fresh tracker from the settled stance, so an evaluation does
    not depend on the ones before it.  The inputs are checked before they are loaded (``INPUTS_RULE``) and against
    ``expected_inputs`` when given.  ``cut_s`` cuts every episode to that many simulated seconds (tests and smoke
    checks only: an episode cut short of its window has no complete window and scores its terms 1, a speed walk
    whenever it is cut, ``SPEED_WALK_RULE``)."""

    def __init__(self, body: str, run_dir: Path | None = None, mode: str = MODE, rate: float = RATE,
                 legs: bool = True, expected_inputs: dict | None = None, cut_s: float | None = None):
        from .legs import LegGap

        if float(rate) != RATE:
            raise ValueError(f"rate {rate!r}: the speed walks' plans are the panels', at {RATE:g}")
        self.body, self.mode, self.rate = body, mode, float(rate)
        self.run_dir = Path(run_dir) if run_dir is not None else RUN / body
        self.cut_s = None if cut_s is None else float(cut_s)
        self.inputs, docs = tune.read_inputs(self.run_dir, mix_files())
        if expected_inputs is not None:
            diff = tune.input_differences(expected_inputs, self.inputs)
            if diff:
                raise tune.InputError(f"{self.run_dir}: the inputs differ from the ones expected: " + "; ".join(diff))
        self.p = plant.build(body, "hard")
        refs_doc = docs["refs.json"]["references"]
        tr_doc = docs["transitions.json"]["transitions"]
        self.omega = control.omega0(self.p, Reference.load(self.run_dir / "refs" / "nominal.npz").q_home)
        self.specs, self.resolved = {}, {}
        for kind in tune.EPISODES:
            for d in ((None,) if kind == "walk" else ("left", "right")):
                self.specs[(kind, d)] = tune.build_spec(self.p, self.run_dir, kind, d, self.omega, refs_doc, tr_doc,
                                                        self.rate)
        planner = panels.Planner(self.run_dir, refs_doc, tr_doc)
        for kind in SPEED_WALKS:  # as panels.Context.built builds walk/brisk and walk/fr05
            cond = panel_condition(kind)
            res = planner.resolve(cond)
            self.resolved[kind] = res
            self.specs[(kind, None)] = tune.build_spec(self.p, self.run_dir, cond.kind, cond.direction, self.omega,
                                                       refs_doc, tr_doc, self.rate, plan=res["plan"], case=cond.case,
                                                       gait=cond.gait)
        loaded = {"refs/nominal.npz"} | {f"{'refs' if seg.kind == 'periodic' else 'transitions'}/{seg.name}.npz"
                                         for s in self.specs.values() for seg in s.track.segments if seg.kind != "hold"}
        unverified = loaded - set(self.inputs["files"])
        if unverified:
            raise tune.InputError(f"the tracks hold files that were not checked: {sorted(unverified)}")
        self.loaded = loaded
        self.lg = LegGap(body) if legs else None

    def seconds(self, kind: str, direction: str | None) -> float:
        s = self.specs[(kind, direction)].seconds
        return s if self.cut_s is None else min(s, self.cut_s)

    def describe(self) -> dict:
        """The mix as built on this body (for the records)."""
        out = {}
        for (kind, d), s in self.specs.items():
            out[kind + (f"_{d}" if d else "")] = dict(
                plan=[list(x) for x in s.plan], seconds=s.seconds, run_seconds=self.seconds(kind, d),
                strides=s.seconds / s.stride_s, period_s=s.period_s, commanded_stride_s=s.stride_s,
                speed_command_m_s=s.speed_cmd, speed_target_m_s=s.speed_target, yaw_rate_command_rad_s=s.yaw_cmd,
                yaw_rate_target_rad_s=s.yaw_target, start_command_ref_s=s.start_ref, stop_command_ref_s=s.stop_ref,
                window_ref_s=s.window_ref, reference_turned_rad=s.info["turned_ref_rad"],
                steady_ref_s=(self.resolved.get(kind) or {}).get("steady_ref_s"),
                required_joint_margins_rad=dict(zip(self.p.joints, (float(x) for x in s.required_margins),
                                                    strict=True)))
        dirs = directions(0)
        out["simulated_s_per_evaluation"] = sum(self.seconds(k, dirs[k]) for k in EPISODES)
        out["cut_s"] = self.cut_s
        return out

    def planned(self, episode_seed: int) -> dict:
        dirs = directions(episode_seed)
        return {k: self.seconds(k, dirs[k]) for k in EPISODES if (k, dirs[k]) in getattr(self, "specs", {})}

    def episode(self, kind: str, direction: str | None, gains, episode_seed: int, j: int,
                latency: int | None = None) -> dict:
        """One episode of the mix: slot ``j``'s seeds (``SEED_RULE``) and delay (``LATENCY_RULE``; ``latency`` forces
        another, for checks)."""
        spec = self.specs[(kind, direction)]
        k = latency_steps(episode_seed, j) if latency is None else int(latency)
        c0 = time.process_time()
        tk = control.Tracker(self.p, spec.track, self.mode, gains=np.asarray(gains, float), rate=self.rate)
        rec = metrics.Recorder(legs_every=tune.LEGS_EVERY if (self.lg is not None and spec.kind != "walk") else 0)
        with servo_delay(self.p, k):
            out = control.run(tk, self.seconds(kind, direction), start="stance",
                              seed=tune.derived_seed(episode_seed, j, 0),
                              noise_seed=tune.derived_seed(episode_seed, j, 1),
                              recorder=rec, joint_noise=tune.START_JOINT_NOISE_RAD,
                              root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S)
        out = tune.warned_fall(out, tune.sim_warnings(self.p.d))
        if kind in SPEED_WALKS:
            res = measure_speed_walk(kind, spec, rec, out, self.p.m.opt.timestep)
        else:
            res = tune.measure(spec, rec, out, self.lg)
        res.update(latency_steps=k, cpu_s=time.process_time() - c0)
        return res

    def evaluate(self, gains, episode_seed: int) -> dict:
        """The speed mix for ``gains`` on ``episode_seed``; an evaluation that raises scores as the worst case
        (``FAILURE_RULE``)."""
        t0, c0 = time.perf_counter(), time.process_time()
        dirs = directions(episode_seed)
        try:
            eps = {kind: self.episode(kind, dirs[kind], gains, episode_seed, j) for j, kind in enumerate(EPISODES)}
            return dict(tune.score(eps), episode_seed=int(episode_seed), directions=dirs,
                        latency_steps={k: e["latency_steps"] for k, e in eps.items()}, episodes=eps,
                        terms=term_vector(eps), wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0,
                        simulated_s=float(sum(e["seconds"] for e in eps.values())))
        except Exception:
            return failed_evaluation(episode_seed, traceback.format_exc(), self.planned(episode_seed),
                                     time.perf_counter() - t0)


# ------------------------------------------------------------------------------------------------
# Sources, workers
# ------------------------------------------------------------------------------------------------
SPEED_ROOT = "speedfit.py"


def speed_sources(files: dict[str, bytes] | None = None) -> dict:
    """``SOURCES_RULE``: the hash of the speed fit's modules with the module list."""
    files = tune.package_files() if files is None else files
    mods = tune.tuner_modules(files, root=SPEED_ROOT)
    return dict(sha256=plant.sources_sha256({m: files[m] for m in mods}), rule=SOURCES_RULE, modules=mods)


def _evaluator(body: str, run_dir: str, mode: str, rate: float, expected_inputs: dict | None = None,
               sources: str | None = None, cut_s: float | None = None) -> Evaluator:
    """``tune.Pool``'s factory: an evaluator, refused on other speed-fit sources than the session's
    (``SOURCES_RULE``)."""
    if sources is not None:
        mine = speed_sources()["sha256"]
        if mine != sources:
            raise tune.PoolError(f"the speed fit's sources changed during the session ({mine[:12]}..., the session's "
                                 f"{sources[:12]}...): restore them and resume")
    return Evaluator(body, Path(run_dir), mode, rate, expected_inputs=expected_inputs, cut_s=cut_s)


def _planned_from_mix(mix: dict | None, seed: int) -> dict:
    mix = mix or {}
    return {k: (mix.get(k + (f"_{d}" if d else "")) or {}).get("run_seconds") for k, d in directions(seed).items()}


def _pool(body: str, run_dir: Path, workers: int, sources: str, cut_s: float | None = None):
    return Pool(body, run_dir, workers, mode=MODE, rate=RATE,
                factory=functools.partial(_evaluator, sources=sources, cut_s=cut_s))


def _map(pool, jobs: list) -> list:
    """The pool's results of (gains, episode seed) jobs; a job whose worker was lost on every attempt scores as the
    worst case (``FAILURE_RULE``)."""
    mix = pool.describe()
    return pool.map([(np.asarray(g, float), int(s)) for g, s in jobs],
                    failed=lambda job, reason: failed_evaluation(job[1], reason, _planned_from_mix(mix, job[1])))


#: The evaluation pool (``tune.Pool``; tests replace it).
Pool = tune.Pool


# ------------------------------------------------------------------------------------------------
# The start
# ------------------------------------------------------------------------------------------------
def pick_of(body: str) -> str:
    """The round-two candidate a registered body is built from (``START_RULE``): the registry's ``candidate_id`` (a
    faster-knee body's is its pick's), or the body itself in a registry without it."""
    if body not in BODIES:
        raise KeyError(f"{body} is not a registered body ({', '.join(BODIES)})")
    try:
        from . import candidate_id
    except ImportError:
        return body
    return candidate_id(body)


def start_body(body: str, start_from: str | None = None, stage1: Path = STAGE1_RUN) -> str:
    """``START_RULE``'s start body of ``body``: itself if it has a Stage 1 fit, else its pick (``pick_of``);
    ``start_from`` may name that body and no other."""
    if body not in BODIES:
        raise SystemExit(f"{body}: not a registered body ({', '.join(BODIES)})")
    own = (Path(stage1) / body / "tune.json").exists()
    want = body if own else pick_of(body)
    if start_from is not None and start_from != want:
        why = (f"{body} has its own Stage 1 fit ({Path(stage1) / body / 'tune.json'}) and starts from it" if own else
               f"{body} has no Stage 1 fit and starts from the pick it is built from, {want} (candidate_id)")
        raise SystemExit(f"--start-from {start_from}: {why} (START_RULE)")
    if want not in BODIES:
        raise SystemExit(f"{body}: its pick {want} is not a registered body ({', '.join(BODIES)})")
    return want


def start_record(body: str, start_from: str | None = None, stage1: Path = STAGE1_RUN) -> dict:
    """``START_RULE``'s start of ``body``: the Stage 1 G5 selection of its start body (``start_body``)."""
    src = start_body(body, start_from, stage1)
    path = Path(stage1) / src / "tune.json"
    if not path.exists():
        raise SystemExit(f"{path}: no Stage 1 G5 record for {src}" + (
            "" if src != body else f" (and {body} is no faster-knee body: its candidate id is its own name)"))
    gains, info = env.tuned_gains(path.parent, src, path)
    if info.get("status") != "passed":
        raise SystemExit(f"{path}: {src}'s G5 selection is {info.get('status')} ({info.get('flag')}): START_RULE "
                         "starts only from a selection that passed G5")
    fixed = {n: float(gains[control.GAIN_NAMES.index(n)]) for n in tune.FIXED_GAINS}
    if fixed != tune.FIXED_GAINS:
        raise SystemExit(f"{path}: the selection holds {fixed}, not the fixed gains {tune.FIXED_GAINS} (FIXED_RULE)")
    return dict(rule=START_RULE, body=body, start_from=src, file=panels._portable(str(path.resolve())),
                sha256=info["sha256"], candidate=info["candidate"], cma_seed=info["cma_seed"], status=info["status"],
                source=f"{src}'s Stage 1 G5 selection ({info['source']})", gains=control.gains_dict(gains),
                gains_rule=env.GAINS_RULE)


def start_key(rec: dict) -> dict:
    """What a fit's start is (the settings a resume and the selection compare)."""
    return {k: rec.get(k) for k in ("body", "start_from", "sha256", "candidate", "cma_seed", "gains")}


def start_path(out: Path, body: str) -> Path:
    return Path(out) / body / "speedfit" / "start.json"


def gains_vector(named: dict) -> np.ndarray:
    return np.array([float(named[n]) for n in control.GAIN_NAMES])


# ------------------------------------------------------------------------------------------------
# The fit
# ------------------------------------------------------------------------------------------------
CMA_SETTINGS = dict(tune.CMA_SETTINGS, sigma0=SIGMA0, sigma0_rule=SIGMA0_RULE, budget=BUDGET,
                    generations=BUDGET // POPSIZE,
                    x0="the start (START_RULE) of the searched gains in the [0, 1] scaling of their "
                       "control.GAIN_BOUNDS")
OBJECTIVE = dict(
    name=OBJECTIVE_NAME, defined="10 Nov 2026, before any speed-fit run, from STAGE1B_PLAN.md section 3",
    base=tune.OBJECTIVE["name"],
    changes=["the mix widened to five episodes: Stage 1's three and the brisk and Fr 0.05 walks (MIX_RULE, "
             "SPEED_WALK_RULE)", "each episode's servo delay of 0-5 physics steps (LATENCY_RULE)",
             "everything else is stage1-tune-v2's: seeds, terms, scales, score, failures, the held capture margin"],
    controller=MODE, gains=list(tune.SEARCHED), fixed_gains=dict(tune.FIXED_GAINS), fixed_rule=tune.FIXED_RULE,
    mix=dict(episodes=list(EPISODES), slots={k: j for j, k in enumerate(EPISODES)}, rule=MIX_RULE,
             plans={**{k: [list(x) for x in tune.episode_plan(k, "left" if k != "walk" else None)]
                       for k in tune.EPISODES},
                    **{k: [list(x) for x in panels.template(panel_condition(k))] for k in SPEED_WALKS}},
             strides=dict(walk=tune.WALK_STRIDES, walking_turn=tune.TURN_STRIDES, in_place_turn=tune.IN_PLACE_STRIDES,
                          brisk="panels.Planner (K strides of brisk)", fr05="panels.Planner (K strides of fr05)"),
             directions="episode seed even: walking turn left, in-place turn right; odd: the reverse",
             floor="hard (mu 0.9)", command_margin=tune.COMMAND_MARGIN, command_rule=control.COMMAND_RATE_RULE,
             walk_window_s=tune.WALK_WINDOW_S),
    speed_walks=SPEED_WALK_RULE, latency=LATENCY_RULE, seeds=SEED_RULE, score=SCORE_RULE, terms=tune.TERMS_RULE,
    success=tune.SUCCESS_RULE, success_terms={k: list(v) for k, v in SUCCESS.items()},
    tracking_terms={k: list(v) for k, v in TRACKING.items()}, limit_terms={k: list(v) for k, v in LIMITS.items()},
    term_order=list(TERM_ORDER), scales=tune.SCALES, failures=FAILURE_RULE, panel_walk_rule=panels.WALK_RULE)
FIXED_RECORD = tune.FIXED_RECORD


def run_dir_of(out: Path, body: str, seed: int) -> Path:
    return Path(out) / body / "speedfit" / str(int(seed))


def _settings(body: str, seed: int, start: dict, cut_s: float | None) -> dict:
    return dict(body=body, seed=int(seed), mode=MODE, rate=RATE, objective=OBJECTIVE_NAME, cma=CMA_SETTINGS,
                latency=dict(steps=list(LATENCY_STEPS), purpose=LATENCY_PURPOSE), start=start_key(start),
                cut_s=None if cut_s is None else float(cut_s))


def _episode_line(e: dict) -> dict:
    out = dict(dir=e["direction"], lat=e.get("latency_steps"), fell=e["fell"], t_fall=e["fall_time_s"],
               cause=e["fall_cause"], surv=round(e["survived_frac"], 6), ok=e["succeeded"])
    if e.get("mujoco_warnings"):
        out["warn"] = e["mujoco_warnings"]
    return out


def _eval_line(gen: int, i: int, x: np.ndarray, fit: float, r: dict) -> dict:
    line = dict(gen=gen, i=i, episode=r["episode_seed"], x=[float(v) for v in x], J=r["J"], fitness=fit, S=r["S"],
                P=r["P"], T=r["T"], L=r["L"], groups=r["groups"], falls=r["falls"],
                episodes={k: _episode_line(e) for k, e in r["episodes"].items()}, terms=r["terms"],
                sim_s=round(r["simulated_s"], 3), wall_s=round(r["wall_s"], 3), cpu_s=round(r.get("cpu_s") or 0.0, 3))
    if r.get("error"):
        line["error"] = r["error"]
    return line


def _write_start(out: Path, body: str, start: dict, log) -> None:
    """``START_RULE``: record the body's start once; refuse a fit from another."""
    sp = start_path(out, body)
    old = panels._read(sp)
    if old is None:
        tune._write_json(sp, dict(start, recorded_at=tune._now()))
        log(f"{body}: start recorded ({start['source']}) in {sp}")
    elif start_key(old) != start_key(start):
        raise SystemExit(f"{sp}: the body's speed fit starts from {old.get('start_from')} {old.get('candidate')} "
                         f"(sha256 {str(old.get('sha256'))[:12]}...), not {start['start_from']} {start['candidate']} "
                         f"({start['sha256'][:12]}...): every seed and session starts from the recorded start")


def fit(body: str, seed: int = SEEDS[0], evals: int = BUDGET, resume: bool = False, out: Path = RUN, workers: int = 1,
        inputs: Path | None = None, start_from: str | None = None, stage1: Path = STAGE1_RUN,
        cut_s: float | None = None, log=print) -> dict:
    """The speed fit of one body and seed (module docstring) until ``evals`` evaluations (whole generations), into
    ``<out>/<body>/speedfit/<seed>/``; the references and transitions are read from ``<inputs>/<body>/`` (default: out),
    the start from ``<stage1>/<start body>/tune.json`` (``START_RULE``)."""
    if body not in BODIES:
        raise KeyError(body)
    src = Path(inputs or out) / body
    d = run_dir_of(out, body, seed)
    path, elog = d / "cma.json", d / "evals.jsonl"
    prov = plant.provenance()
    ssrc = speed_sources()
    start = start_record(body, start_from, stage1)
    settings = _settings(body, seed, start, cut_s)
    if path.exists() and not resume:
        raise SystemExit(f"{path} exists: pass --resume to continue it (or remove it to start over)")
    doc = None
    if resume:
        if not path.exists():
            raise SystemExit(f"{path}: nothing to resume")
        doc = json.loads(path.read_text())
        if doc["settings"] != json.loads(json.dumps(settings)):
            raise SystemExit(f"{path}: its settings differ from this code's and start; refusing to resume")
        first = doc["sessions"][0]
        was = (first.get("speed_sources") or {}).get("sha256")
        if was != ssrc["sha256"]:
            raise SystemExit(f"{path}: the speed fit's sources changed since the run started ({str(was)[:12]} -> "
                             f"{ssrc['sha256'][:12]}; modules {', '.join(ssrc['modules'])}); a resumed run would not "
                             "continue the same computation")
        if first["provenance"]["versions"] != prov["versions"]:
            raise SystemExit(f"{path}: library versions changed since the run started; refusing to resume")
    _write_start(out, body, start, log)
    pool = _pool(body, src, workers, ssrc["sha256"], cut_s)
    try:
        session = dict(started_at=tune._now(), from_generation=0, workers=pool.workers, provenance=prov,
                       speed_sources=ssrc, tuner_sources=tune.tuner_sources(), inputs=pool.inputs, incidents=[])
        if resume:
            diff = tune.input_differences(doc["sessions"][0].get("inputs"), pool.inputs)
            if diff:
                raise SystemExit(f"{path}: its inputs changed since the run started ({'; '.join(diff)}); refusing to "
                                 "resume")
            cma = tune.CMA.from_state(doc["state"], SIGMA0, seed)
            n_done = int(doc["evals"])
            lines = elog.read_text().splitlines() if elog.exists() else []
            if len(lines) < n_done:
                raise SystemExit(f"{elog}: {len(lines)} evaluations logged, the checkpoint has {n_done}")
            if len(lines) > n_done:  # evaluations after the last checkpoint: the resumed run repeats them exactly
                elog.write_text("".join(line + "\n" for line in lines[:n_done]))
                log(f"{body} seed {seed}: dropped {len(lines) - n_done} evaluations logged after the checkpoint")
            session["from_generation"] = cma.gen
            doc["sessions"].append(session)
        else:
            cma = tune.CMA(tune.to_unit(gains_vector(start["gains"])), SIGMA0, POPSIZE, seed)
            doc = dict(schema=SCHEMA, body=body, seed=int(seed), settings=settings, objective=OBJECTIVE, start=start,
                       fixed_gains=FIXED_RECORD, rules=RULES, sessions=[session], history=[], evals=0,
                       status="running", state=cma.state())
            d.mkdir(parents=True, exist_ok=True)
            elog.write_text("")
            tune._write_json(path, doc)
        doc["budget"] = int(evals)
        doc["inputs"] = str(src)
        doc.setdefault("mix", pool.describe())
        order = tune.training_order(seed)
        seen = 0
        while doc["evals"] + cma.lam <= evals:
            g = cma.gen
            ep = int(order[g % len(order)])
            X = cma.ask()
            t0 = time.perf_counter()
            res = _map(pool, [(tune.from_unit(x), ep) for x in X])
            fitness = [r["J"] + tune.repair_penalty(x) for x, r in zip(X, res, strict=True)]
            with elog.open("a") as fh:
                for i, (x, f, r) in enumerate(zip(X, fitness, res, strict=True)):
                    fh.write(json.dumps(_eval_line(g, i, x, f, r)) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            cma.tell(X, fitness)
            J = np.array([r["J"] for r in res])
            best = int(np.argmin(fitness))
            new = pool.incidents[seen:]
            seen = len(pool.incidents)
            doc["sessions"][-1]["incidents"].extend(new)
            falls = {k: int(sum(r["episodes"][k]["fell"] for r in res)) for k in EPISODES}
            doc["history"].append(dict(gen=g, episode=ep, latency_steps=latencies(ep), evals=doc["evals"] + cma.lam,
                                       sigma=cma.sigma, mean=cma.mean.tolist(),
                                       mean_gains=tune.from_unit(cma.mean).tolist(),
                                       J_min=float(J.min()), J_median=float(np.median(J)), J_max=float(J.max()),
                                       falls=int(sum(r["falls"] for r in res)), falls_by_kind=falls, best=best,
                                       best_gains=tune.from_unit(X[best]).tolist(), wall_s=time.perf_counter() - t0,
                                       cpu_s=float(sum(r.get("cpu_s") or 0.0 for r in res)),
                                       sim_s=float(sum(r["simulated_s"] for r in res)),
                                       errors=int(sum(bool(r.get("error")) for r in res)),
                                       warned_episodes=int(sum(tune._warned(r) for r in res)), incidents=len(new)))
            doc["evals"] += cma.lam
            doc["state"] = cma.state()
            tune._write_json(path, doc)
            h = doc["history"][-1]
            extra = "".join(f", {h[k]} {k.replace('_', ' ')}" for k in ("errors", "warned_episodes", "incidents")
                            if h[k])
            log(f"{body} seed {seed} gen {g}: episode {ep}, J min/median {h['J_min']:.4f}/{h['J_median']:.4f}, falls "
                f"{h['falls']} ({', '.join(f'{k} {v}' for k, v in falls.items() if v) or 'none'}){extra}, sigma "
                f"{cma.sigma:.4f}, {h['wall_s']:.1f} s ({h['wall_s'] / cma.lam:.2f} s/eval)")
            for x in new:
                log(f"{body} seed {seed} gen {g}: job {x['job']} (episode {x['episode']}): {x['what']}; {x['then']}")
    finally:
        pool.close()
    doc["status"] = "done" if doc["evals"] >= BUDGET else "partial"
    s = doc["sessions"][-1]
    s["ended_at"] = tune._now()
    s["speed_sources_unchanged_during_session"] = speed_sources()["sha256"] == ssrc["sha256"]
    s["package_sources_unchanged_during_session"] = plant.sources_sha256() == prov["stage1_sources"]["sha256"]
    tune._write_json(path, doc)
    return doc


# ------------------------------------------------------------------------------------------------
# Selection: speed G5
# ------------------------------------------------------------------------------------------------
def _per_kind(results: list[dict]) -> dict:
    """Per kind: runs, survived, succeeded, plan section 4's pass rule read on them, and the same by delay."""
    out = {}
    for k in EPISODES:
        eps = [r["episodes"][k] for r in results]
        by = {}
        for e in eps:
            b = by.setdefault(str(e.get("latency_steps")), dict(runs=0, survived=0, succeeded=0))
            b["runs"] += 1
            b["survived"] += int(not e["fell"])
            b["succeeded"] += int(bool(e["succeeded"]))
        surv, succ = int(sum(not e["fell"] for e in eps)), int(sum(bool(e["succeeded"]) for e in eps))
        out[k] = dict(condition=CONDITIONS[k], runs=len(eps), survived=surv, succeeded=succ,
                      pass_rule_met=bool(len(eps) == PASS_SURVIVE and surv == PASS_SURVIVE and succ >= PASS_SUCCEED)
                      if len(eps) == PASS_SURVIVE else None,
                      failed_terms=dict(Counter(t for e in eps if not e["fell"] for t, ok in e["success"].items()
                                                if not ok)),
                      by_latency_steps=dict(sorted(by.items(), key=lambda kv: (kv[0] == "None", kv[0]))))
    return out


def validation_summary(results: list[dict]) -> dict:
    J = [r["J"] for r in results]
    return dict(falls=int(sum(r["falls"] for r in results)), S_mean=float(np.mean([r["S"] for r in results])),
                P_mean=float(np.mean([r["P"] for r in results])), J_mean=float(np.mean(J)), J=J,
                errors=int(sum(bool(r.get("error")) for r in results)),
                warned_episodes=int(sum(tune._warned(r) for r in results)), conditions=_per_kind(results))


def validation_record(r: dict) -> dict:
    out = tune._validation_record(r)
    out.update(latency_steps=r.get("latency_steps"), cpu_s=r.get("cpu_s"))
    for k, e in r["episodes"].items():
        out["episodes"][k].update(latency_steps=e.get("latency_steps"), judged_window_s=e.get("judged_window_s"))
    return out


def _sha256_json(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def gains_sha256(gains) -> str:
    """A gain vector's sha256: of its values as a JSON list (``G5_RESUME_RULE``)."""
    return _sha256_json([float(v) for v in np.asarray(gains, float)])


def g5_context(body: str, speed_sources_sha256: str, inputs: dict | None, versions: dict) -> dict:
    """What a kept validation evaluation was computed on besides its gains and episode (``G5_RESUME_RULE``)."""
    inputs = inputs or {}
    return dict(body=body, mode=MODE, rate=RATE, objective=OBJECTIVE_NAME, speed_sources=speed_sources_sha256,
                inputs=_sha256_json(dict(files=inputs.get("files"), records=inputs.get("records"))), versions=versions)


class G5Log:
    """``G5_RESUME_RULE``'s log of validation evaluations: ``get`` a kept one (None if there is none for this
    context), ``add`` a new one (appended, flushed and synced; not kept if it failed to run or raised)."""

    def __init__(self, path: Path, context: dict):
        self.path, self.context = Path(path), dict(context)
        self.context_sha256 = _sha256_json(self.context)
        self.kept: dict[str, dict] = {}
        self.dropped = self.added = self.reused = 0
        good = []
        for line in self.path.read_text().splitlines() if self.path.exists() else ():
            try:
                x = json.loads(line)
                key, value = x["key"], x["result"]
            except (ValueError, KeyError, TypeError):  # a line cut short when the process died
                self.dropped += 1
                continue
            good.append(line)
            self.kept[key] = value
        if self.dropped:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text("".join(x + "\n" for x in good))
            os.replace(tmp, self.path)

    def key(self, gains, episode: int) -> str:
        return _sha256_json(dict(context=self.context_sha256, gains=gains_sha256(gains), episode=int(episode)))

    def get(self, gains, episode: int) -> dict | None:
        v = self.kept.get(self.key(gains, episode))
        self.reused += v is not None
        return v

    def add(self, gains, episode: int, value: dict) -> dict:
        value = json.loads(json.dumps(value))
        if value.get("error"):
            return value
        key = self.key(gains, episode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(dict(key=key, context=self.context_sha256, gains_sha256=gains_sha256(gains),
                                     episode=int(episode), result=value)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.kept[key] = value
        self.added += 1
        return value

    def describe(self) -> dict:
        return dict(rule=G5_RESUME_RULE, file=panels._portable(str(self.path.resolve())), context=self.context,
                    context_sha256=self.context_sha256, reused=self.reused, added=self.added,
                    dropped_lines=self.dropped)


def speed_g5(body: str, out: Path = RUN, seeds=SEEDS, workers: int = 1, episodes: int = VALIDATION_RUNS,
             require_done: bool = True, at_generation: int | None = None, inputs: Path | None = None,
             log=print) -> dict:
    """Speed G5 (``SELECTION_RULE``, ``JUDGED_RULE``) into ``<out>/<body>/speedg5.json``, the references read from
    ``<inputs>/<body>/`` (default: out); the validation evaluations are kept as they finish and reused by a rerun
    (``G5_RESUME_RULE``)."""
    if body not in BODIES:
        raise KeyError(body)
    prov, ssrc = plant.provenance(), speed_sources()
    t0 = time.perf_counter()
    val = list(range(VALIDATION_EPISODES[0], VALIDATION_EPISODES[0] + int(episodes)))
    start = panels._read(start_path(out, body))
    if start is None:
        raise SystemExit(f"{start_path(out, body)}: no recorded start (run speedfit first)")
    cands = {"start": dict(gains=gains_vector(start["gains"]).tolist(), source=f"the speed fit's start: "
                                                                                f"{start['source']}")}
    runs, verdicts = {}, {}
    pool = _pool(body, Path(inputs or out) / body, workers, ssrc["sha256"])
    try:
        for seed in seeds:
            path = run_dir_of(out, body, seed) / "cma.json"
            if not path.exists():
                runs[seed] = dict(status="missing", sources_match=None, inputs_match=None, start_matches=None)
                continue
            doc = json.loads(path.read_text())
            fit_src = [(s.get("speed_sources") or {}).get("sha256") for s in doc["sessions"]]
            n_gen = len(doc["history"])
            runs[seed] = dict(status=doc["status"], evals=doc["evals"], generations=n_gen, budget=doc.get("budget"),
                              sigma=doc["state"]["sigma"], sessions=len(doc["sessions"]),
                              speed_sources_sha256=fit_src[0],
                              sources_match=bool(fit_src) and all(x == ssrc["sha256"] for x in fit_src),
                              inputs_match=all(not tune.input_differences(s.get("inputs"), pool.inputs)
                                               for s in doc["sessions"]),
                              start_matches=start_key(doc["start"]) == start_key(start),
                              cut_s=doc["settings"].get("cut_s"))
            r = runs[seed]
            if not (r["sources_match"] and r["inputs_match"] and r["start_matches"]):
                continue
            if r["cut_s"] is not None:
                r["reason"] = f"a fit on episodes cut to {r['cut_s']:g} s (a smoke check) is not judged"
                continue
            if at_generation is not None:
                if n_gen < int(at_generation):
                    r["reason"] = f"generation {n_gen} of {int(at_generation)} reached"
                    continue
                judged_at, partial = int(at_generation), False
            elif require_done and doc["status"] != "done":
                r["reason"] = f"not done ({doc['status']}, generation {n_gen})"
                continue
            else:
                judged_at, partial = n_gen, doc["status"] != "done"  # (--partial on an unfinished fit: JUDGED_RULE)
            gens = tune.candidate_generations(judged_at)
            r.update(judged_at_generation=judged_at, partial=partial, candidates=[f"seed{seed}_gen{g}" for g in gens])
            for g in gens:
                h = doc["history"][g - 1]
                cands[f"seed{seed}_gen{g}"] = dict(gains=h["mean_gains"], source=(
                    f"the CMA-ES mean after generation {g} of speed-fit seed {seed} ({h['evals']} evals)"))
        names = list(cands)
        jobs = [(np.asarray(cands[n]["gains"], float), ep) for n in names for ep in val]
        glog = G5Log(Path(out) / body / G5_LOG, g5_context(body, ssrc["sha256"], pool.inputs, prov["versions"]))
        vals = [glog.get(g, ep) for g, ep in jobs]
        todo = [i for i, v in enumerate(vals) if v is None]
        log(f"{body}: {len(names)} candidates x {len(val)} validation episodes = {len(jobs)} evaluations, "
            f"{len(jobs) - len(todo)} kept from an earlier session ({G5_LOG})")
        step = 1 if pool.workers == 1 else 2 * pool.workers
        for a in range(0, len(todo), step):
            idx = todo[a : a + step]
            for i, r in zip(idx, _map(pool, [jobs[i] for i in idx]), strict=True):
                vals[i] = glog.add(*jobs[i], validation_record(r))
            done = min(a + step, len(todo))
            if done == len(todo) or done // 10 > a // 10:
                log(f"{body}: {done}/{len(todo)} validation evaluations run")
        by = {n: vals[i * len(val) : (i + 1) * len(val)] for i, n in enumerate(names)}
        for n in names:
            cands[n]["gains_named"] = control.gains_dict(cands[n]["gains"])
            cands[n]["validation"] = validation_summary(by[n])
            cands[n]["runs"] = by[n]
        for seed in seeds:
            r = runs[seed]
            base = dict(sources_match=r["sources_match"], inputs_match=r["inputs_match"],
                        start_matches=r.get("start_matches"), refused=False)
            if r["status"] == "missing":
                verdicts[seed] = dict(base, judged=False, reason="no run")
            elif not (r["sources_match"] and r["inputs_match"] and r["start_matches"]):
                what = [w for w, ok in (("sources", r["sources_match"]), ("inputs", r["inputs_match"]),
                                        ("start", r["start_matches"])) if not ok]
                verdicts[seed] = dict(base, judged=False, refused=True, reason=(
                    f"the fit ran on other {', '.join(what)} than this selection: refusing to judge"))
            elif not r.get("candidates"):
                verdicts[seed] = dict(base, judged=False, reason=r.get("reason") or f"no finished run ({r['status']})")
            else:
                pool_names = ["start"] + r["candidates"]
                ranked = sorted(pool_names, key=lambda n: tune.selection_key(cands[n]["validation"],
                                                                             pool_names.index(n)))
                sel = ranked[0]
                full, partial = len(val) == VALIDATION_RUNS, bool(r["partial"])
                reason = None
                if partial:
                    reason = (f"smoke check (--partial): the fit is {r['status']} at generation "
                              f"{r['judged_at_generation']}; an unfinished fit is judged with --at-generation G, the "
                              "last generation every arm reached (JUDGED_RULE)")
                elif not full:
                    reason = f"smoke check: {len(val)} validation episodes, not {VALIDATION_RUNS}"
                verdicts[seed] = dict(
                    base, judged=full and not partial, partial=partial, reason=reason,
                    judged_at_generation=r["judged_at_generation"], selected=sel,
                    selected_gains=cands[sel]["gains_named"],
                    selection=[dict(candidate=n, rank=ranked.index(n) + 1,
                                    **{k: cands[n]["validation"][k] for k in ("falls", "S_mean", "P_mean", "J_mean")})
                               for n in pool_names],
                    conditions=cands[sel]["validation"]["conditions"],
                    start_conditions=cands["start"]["validation"]["conditions"])
        incidents, pool_inputs, kept = list(pool.incidents), pool.inputs, glog.describe()
    finally:
        pool.close()
    doc = dict(schema=SCHEMA, body=body, gate="speed G5", rule=SELECTION_RULE, judged_rule=JUDGED_RULE,
               selection_rule=tune.SELECTION_RULE, objective=OBJECTIVE, rules=RULES,
               settings=dict(cma=CMA_SETTINGS, mode=MODE, rate=RATE), validation_episodes=val, start=start,
               runs={str(s): v for s, v in runs.items()}, candidates=cands,
               seeds={str(s): v for s, v in verdicts.items()},
               seeds_selected=[s for s, v in verdicts.items() if "selected" in v],
               seeds_refused=[s for s, v in verdicts.items() if v.get("refused")], speed_sources=ssrc,
               inputs=pool_inputs, evaluations_kept=kept,
               incidents=incidents, failure_rule=FAILURE_RULE, seconds=time.perf_counter() - t0, provenance=prov)
    doc["speed_sources"]["unchanged_during_run"] = speed_sources()["sha256"] == ssrc["sha256"]
    tune._write_json(Path(out) / body / "speedg5.json", doc)
    return doc


def speed_g5_summary(doc: dict) -> str:
    lines = [f"speed G5 {doc['body']}: start {doc['start']['start_from']} {doc['start']['candidate']}"]
    for s, v in doc["seeds"].items():
        if "selected" not in v:
            lines.append(f"   seed {s}: {v.get('reason')}")
            continue
        sel = next(x for x in v["selection"] if x["candidate"] == v["selected"])
        conds = "; ".join(f"{c['condition']} {c['succeeded']}/{c['runs']} ({c['survived']} survive)"
                          for c in v["conditions"].values())
        lines.append(f"   seed {s}: selected {v['selected']} at generation {v['judged_at_generation']} (falls "
                     f"{sel['falls']}, mean S {sel['S_mean']:.4f}, mean P {sel['P_mean']:.4f}); {conds}"
                     + ("" if v["judged"] else f" (not judged: {v.get('reason')})"))
        for x in v["selection"]:
            lines.append(f"      {x['rank']}. {x['candidate']}: falls {x['falls']}, mean S {x['S_mean']:.4f}, mean P "
                         f"{x['P_mean']:.4f}, mean J {x['J_mean']:.4f}")
    k = doc.get("evaluations_kept") or {}
    if k.get("reused"):
        lines.append(f"   {k['reused']} validation evaluations reused from {G5_LOG}, {k.get('added', 0)} run now")
    if doc.get("incidents"):
        lines.append(f"   {len(doc['incidents'])} worker incidents (speedg5.json incidents)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# The validation rows (panels.Row)
# ------------------------------------------------------------------------------------------------
class DelayedRow(panels.TrackerRow):
    """The tracker R2 with given gains and a servo delay for the whole episode (``ROWS_RULE``), on
    ``ROW_CONDITIONS`` only; with no delay it is ``panels.TrackerRow`` bit for bit."""

    def __init__(self, name: str, gains, gains_info: dict | None, latency_steps: int = 0, controller: str = ""):
        super().__init__(name, MODE, gains, gains_info, controller=controller)
        k = int(latency_steps)
        if k != latency_steps or k < 0:
            raise ValueError(f"{name}: servo delay {latency_steps!r}: whole physics steps, at least 0")
        self.latency_steps = k

    def runs(self, cond: panels.Condition) -> bool:
        return cond.key in ROW_CONDITIONS

    def describe(self) -> dict:
        return dict(super().describe(), latency_steps=self.latency_steps, latency_ms=2 * self.latency_steps,
                    latency_rule=LATENCY_RULE, rows_rule=ROWS_RULE, conditions=list(ROW_CONDITIONS))

    def sources(self) -> dict:
        """This module (the row and the delay model it runs) by sha256 (``panels.PANEL_SOURCES_RULE``)."""
        return panels.module_sha256(type(self))

    def spec(self) -> dict:
        return dict(super().spec(), latency_steps=self.latency_steps)

    @classmethod
    def from_spec(cls, s: dict) -> "DelayedRow":
        return cls(s["name"], s["gains"], s["gains_info"], s["latency_steps"], s.get("controller", ""))

    def episode(self, ctx: panels.Context, cond: panels.Condition, seed: int, relax=()):
        b = ctx.built(cond)
        with servo_delay(b.p, self.latency_steps):
            return super().episode(ctx, cond, seed, relax)


def ms_steps(ms: int, name: str) -> int:
    """Whole 2 ms physics steps of a row's delay."""
    dt_ms = 2
    if ms % dt_ms or not 0 <= ms <= LATENCY_MAX_ROW_MS:
        raise ValueError(f"row {name!r}: a delay of whole {dt_ms} ms physics steps, 0-{LATENCY_MAX_ROW_MS} ms")
    return ms // dt_ms


def speed_run_of(run_dir: Path | None = None) -> Path:
    """The speed run directory a row reads its gains from (``ROWS_RULE``): ``STAGE1B_SPEED_RUN`` if set, else the
    round's (``RUN``), whatever run directory the panels read their references from (``run_dir``, unused)."""
    v = os.environ.get(SPEED_RUN_ENV)
    return Path(v) if v else RUN


def _file_info(path: Path) -> dict:
    return dict(file=panels._portable(str(Path(path).resolve())), sha256=plant.sha256(path))


def r2s_gains(speed_run: Path, body: str, seed: int) -> tuple[np.ndarray, dict]:
    """Fit seed ``seed``'s speed G5 selection (``ROWS_RULE``)."""
    path = Path(speed_run) / body / "speedg5.json"
    if not path.exists():
        raise ValueError(f"{path}: no speed G5 record (run speedg5 first, or set {SPEED_RUN_ENV})")
    doc = json.loads(path.read_text())
    v = (doc.get("seeds") or {}).get(str(int(seed))) or {}
    if "selected" not in v:
        raise ValueError(f"{path}: speed-fit seed {seed} has no selection ({v.get('reason')})")
    if not v.get("judged"):
        raise ValueError(f"{path}: speed-fit seed {seed}'s selection is a smoke check, not judged "
                         f"({v.get('reason') or 'fewer than 20 validation episodes'}); refusing it as a row "
                         "(JUDGED_RULE)")
    gains = gains_vector(v["selected_gains"])
    return gains, dict(rule=ROWS_RULE, **_file_info(path), fit_seed=int(seed), candidate=v["selected"],
                       judged_at_generation=v.get("judged_at_generation"),
                       start=dict(start_from=doc["start"]["start_from"], candidate=doc["start"]["candidate"],
                                  sha256=doc["start"]["sha256"]))


def start_gains(speed_run: Path, body: str) -> tuple[np.ndarray, dict]:
    """The speed fit's recorded start (``ROWS_RULE``'s R2@10ms), checked against the Stage 1 record it names."""
    path = start_path(speed_run, body)
    rec = panels._read(path)
    if rec is None:
        raise ValueError(f"{path}: no recorded start (speedfit records it when a fit starts; or set {SPEED_RUN_ENV})")
    g5 = STUDY / rec["file"] if not Path(rec["file"]).is_absolute() else Path(rec["file"])
    if g5.exists() and plant.sha256(g5) != rec["sha256"]:
        raise ValueError(f"{g5}: its sha256 is not the one {path} records: the Stage 1 G5 record changed")
    gains = gains_vector(rec["gains"])
    return gains, dict(rule=ROWS_RULE, **_file_info(path), start_from=rec["start_from"], candidate=rec["candidate"],
                       cma_seed=rec["cma_seed"], stage1_file=rec["file"], stage1_sha256=rec["sha256"])


_ROW_NAME = re.compile(r"^R2s/(\d+)(?:@(\d+)ms)?$")


def r2s_row(body: str, run_dir: Path, name: str) -> DelayedRow:
    """R2s/<s> and R2s/<s>@<ms>ms (``ROWS_RULE``)."""
    m = _ROW_NAME.match(name)
    if not m:
        raise ValueError(f"row {name!r}: R2s/<seed> or R2s/<seed>@<ms>ms")
    seed, ms = int(m.group(1)), int(m.group(2) or 0)
    gains, info = r2s_gains(speed_run_of(run_dir), body, seed)
    return DelayedRow(name, gains, info, ms_steps(ms, name), controller=(
        f"R2s: the tracker on the estimate with speed-fit seed {seed}'s speed G5 selection {info['candidate']}, "
        f"{ms} ms of servo delay"))


def r2_delayed_row(body: str, run_dir: Path, name: str) -> DelayedRow:
    """R2@10ms and R2@0ms (``ROWS_RULE``): the speed fit's start at that delay."""
    m = re.match(r"^R2@(\d+)ms$", name)
    if not m or int(m.group(1)) not in R2_ROW_DELAYS_MS:
        raise ValueError(f"row {name!r}: " + " or ".join(f"R2@{ms}ms" for ms in R2_ROW_DELAYS_MS))
    ms = int(m.group(1))
    gains, info = start_gains(speed_run_of(run_dir), body)
    return DelayedRow(name, gains, info, ms_steps(ms, name), controller=(
        f"R2: the tracker on the estimate with Stage 1's body-level G5 selection ({info['start_from']} "
        f"{info['candidate']}), {ms} ms of servo delay"))


def _g7_guarded(g7):
    """``G7_RULE``: ``panels.g7`` for Stage 1's bodies; FileNotFoundError for a body without G7 figures."""

    @functools.wraps(g7)
    def guarded(body: str, run_dir: Path, *args, **kw) -> dict:
        if body not in panels.G7_PLAN:
            raise FileNotFoundError(f"{body}: the plan gives no G7 figures (panels.G7_PLAN: {', '.join(panels.G7_PLAN)}"
                                    "); G7 is Stage 1's gate on its three bodies (speedfit.G7_RULE)")
        return g7(body, run_dir, *args, **kw)

    guarded.speed_round = True
    return guarded


def _hook_panels() -> None:
    """Register the rows (``ROWS_RULE``), attribute their controllers (``ATTRIBUTION_RULE``) and guard G7
    (``G7_RULE``); idempotent."""
    panels.register_rows("R2s", r2s_row)
    for ms in R2_ROW_DELAYS_MS:
        panels.register_rows(f"R2@{ms}ms", r2_delayed_row)
    panels.ATTRIBUTED = tuple(dict.fromkeys(tuple(panels.ATTRIBUTED) + SPEED_CONTROLLERS))
    if not getattr(panels.g7, "speed_round", False):
        panels.g7 = _g7_guarded(panels.g7)


_hook_panels()


# ------------------------------------------------------------------------------------------------
# Costs
# ------------------------------------------------------------------------------------------------
#: The measured slowdown of four fits run at once on the 4 cores against one alone, at the same CPU cost per simulated
#: second (the round's review sample, another builder's prepare also running): wall = CPU x this (``COST_RULE``).
PARALLEL_SLOWDOWN = (1.6, 1.7)
#: Decision 20's lifted runs per failing condition of a row (``ATTRIBUTION_RULE``): typically 2-3 relaxable
#: candidates of 20 starts each (``control.RELAXATIONS`` bounds it at 6).
ATTRIBUTION_EPISODES = (40, 60)
COST_RULE = (
    "one full evaluation of the speed mix at the start's gains on training episode training_order(42)[0] (1865) in "
    "this process: its CPU seconds (process time) and wall seconds, per episode and in all, with the load average "
    "beside; CPU per simulated second = the evaluation's CPU / its simulated seconds (falls end episodes early); "
    "projections at that rate over the planned seconds (every episode run to its end, an upper bound once the fit "
    "stops falling), as CPU hours and as wall hours under two conditions: one idle core per job (wall = CPU / "
    "workers) and four such jobs at once on the 4 cores (wall = CPU x PARALLEL_SLOWDOWN, 1.6-1.7, measured with "
    "another builder's prepare on the machine; it assumes nothing else heavy runs): the fit's 360 evaluations, speed "
    "G5 (4 candidates x 20 evaluations), the three validation rows (R2s/42, R2s/42@10ms, R2@10ms: 7 conditions x 20 "
    "episodes, the panels' planned seconds), R2@0ms (a faster-knee body's start without a refit, ROWS_RULE), "
    "decision 20's attribution per row (ATTRIBUTION_RULE: ATTRIBUTION_EPISODES, 40-60 lifted episodes for each of the "
    "two speed walks, as if both failed with at least 5 falls: a planning figure, not a bound) and per body, run one "
    "after the other: arm S the fit, speed G5, the three rows and their attribution, arm K those and R2@0ms with its "
    "attribution")


def measure_cost(body: str, run_dir: Path, gains, seed: int | None = None, log=print) -> dict:
    """``COST_RULE`` with ``gains`` (the start's, ``start_record``) on ``run_dir``'s inputs."""
    t0 = time.perf_counter()
    ev = Evaluator(body, run_dir)
    build_s = time.perf_counter() - t0
    seed = int(tune.training_order(SEEDS[0])[0]) if seed is None else int(seed)
    load0 = list(os.getloadavg())
    r = ev.evaluate(gains, seed)
    per = {k: dict(seconds=e["seconds"], planned_s=ev.seconds(k, e["direction"]), fell=e["fell"],
                   wall_s=e["wall_s"], cpu_s=e["cpu_s"], latency_steps=e["latency_steps"])
           for k, e in r["episodes"].items()}
    cpu_per_sim = r["cpu_s"] / max(r["simulated_s"], 1e-9)
    wall_per_sim = r["wall_s"] / max(r["simulated_s"], 1e-9)
    planned_eval = ev.describe()["simulated_s_per_evaluation"]
    _, docs = tune.read_inputs(Path(run_dir), panels.panel_files(panels.select(panels.conditions(),
                                                                               names=ROW_CONDITIONS)))
    planner = panels.Planner(Path(run_dir), docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
    row_s = {c.key: planner.resolve(c)["seconds"] for c in panels.select(panels.conditions(), names=ROW_CONDITIONS)}
    row_sim = panels.RUNS * sum(row_s.values())  # one row
    walks_sim = row_s["walk/brisk"] + row_s["walk/fr05"]
    lo, hi = PARALLEL_SLOWDOWN

    def job(sim_s, **extra):
        cpu_h = sim_s * cpu_per_sim / 3600
        return dict(extra, simulated_s=sim_s, cpu_h=cpu_h, wall_h_idle_core=cpu_h,
                    wall_h_four_at_once=[cpu_h * lo, cpu_h * hi])

    fit_sim = BUDGET * planned_eval
    g5_sim = 4 * VALIDATION_RUNS * planned_eval
    att = [n * walks_sim for n in ATTRIBUTION_EPISODES]  # per row: both speed walks, n lifted episodes each
    proj = dict(
        fit_360=job(fit_sim, evaluations=BUDGET),
        fit_360_at_this_evaluation=dict(cpu_h=BUDGET * r["cpu_s"] / 3600),
        speed_g5=job(g5_sim, evaluations=4 * VALIDATION_RUNS),
        rows=job(3 * row_sim, rows=3, conditions=row_s, episodes=3 * panels.RUNS * len(row_s)),
        r2_at_0ms=job(row_sim, rows=1, episodes=panels.RUNS * len(row_s)),
        attribution_per_row=dict(job(att[0], episodes=2 * ATTRIBUTION_EPISODES[0]),
                                 high=job(att[1], episodes=2 * ATTRIBUTION_EPISODES[1])))
    a_lo, a_hi = proj["attribution_per_row"]["cpu_h"], proj["attribution_per_row"]["high"]["cpu_h"]
    base = sum(proj[k]["cpu_h"] for k in ("fit_360", "speed_g5", "rows"))
    proj["per_body"] = dict(arm_S=dict(rows=3, cpu_h=[base + 3 * a_lo, base + 3 * a_hi]),
                            arm_K=dict(rows=4, cpu_h=[base + proj["r2_at_0ms"]["cpu_h"] + 4 * a_lo,
                                                      base + proj["r2_at_0ms"]["cpu_h"] + 4 * a_hi]))
    for v in proj["per_body"].values():
        v["wall_h_four_at_once"] = [v["cpu_h"][0] * lo, v["cpu_h"][1] * hi]
    out = dict(rule=COST_RULE, body=body, episode_seed=seed, build_s=build_s, evaluation=dict(
        J=r["J"], falls=r["falls"], simulated_s=r["simulated_s"], cpu_s=r["cpu_s"], wall_s=r["wall_s"], episodes=per,
        error=r.get("error")), cpu_per_sim_s=cpu_per_sim, wall_per_sim_s=wall_per_sim,
        planned_s_per_evaluation=planned_eval, load_average=[load0, list(os.getloadavg())],
        parallel_slowdown=list(PARALLEL_SLOWDOWN), attribution_episodes=list(ATTRIBUTION_EPISODES), projections=proj,
        cpu_count=os.cpu_count(), speed_sources=speed_sources(), provenance=plant.provenance())
    eps = ", ".join(f"{k} {'fell at ' + format(e['seconds'], '.1f') + ' s' if e['fell'] else 'up'}, "
                    f"{e['latency_steps']} steps" for k, e in per.items())
    log(f"{body}: one evaluation {r['simulated_s']:.1f} simulated s (planned {planned_eval:.1f}), {r['cpu_s']:.1f} CPU "
        f"s, {r['wall_s']:.1f} wall s ({cpu_per_sim:.3f} CPU s per simulated s; load {load0[0]:.1f}); falls "
        f"{r['falls']} ({eps})")
    f, g5, rows = proj["fit_360"], proj["speed_g5"], proj["rows"]
    k = proj["per_body"]["arm_K"]
    log(f"{body}: fit 360 evaluations {f['cpu_h']:.2f} core-h ({f['wall_h_four_at_once'][0]:.2f}-"
        f"{f['wall_h_four_at_once'][1]:.2f} h with four at once); speed G5 {g5['cpu_h']:.2f} core-h; the three rows "
        f"{rows['episodes']} episodes, {rows['cpu_h']:.2f} core-h; R2@0ms {proj['r2_at_0ms']['cpu_h']:.2f} core-h; "
        f"attribution {a_lo:.2f}-{a_hi:.2f} core-h per row; per K body "
        f"{k['cpu_h'][0]:.2f}-{k['cpu_h'][1]:.2f} core-h, {k['wall_h_four_at_once'][0]:.1f}-"
        f"{k['wall_h_four_at_once'][1]:.1f} h with four bodies at once (every episode to its end)")
    return out


RULES = dict(mix=MIX_RULE, speed_walks=SPEED_WALK_RULE, latency=LATENCY_RULE, seeds=SEED_RULE, score=SCORE_RULE,
             failures=FAILURE_RULE, start=START_RULE, sigma0=SIGMA0_RULE, sources=SOURCES_RULE, inputs=INPUTS_RULE,
             resume=RESUME_RULE, selection=SELECTION_RULE, judged=JUDGED_RULE, g5_resume=G5_RESUME_RULE,
             rows=ROWS_RULE, attribution=ATTRIBUTION_RULE, g7=G7_RULE, cost=COST_RULE, fixed=tune.FIXED_RULE,
             terms=tune.TERMS_RULE, success=tune.SUCCESS_RULE)

__all__ = ["BUDGET", "EPISODES", "Evaluator", "DelayedRow", "G5Log", "OBJECTIVE", "ROW_CONDITIONS", "RULES", "SIGMA0",
           "SPEED_CONTROLLERS", "fit", "latencies", "latency_steps", "measure_cost", "pick_of", "servo_delay", "speed_g5",
           "start_body", "start_record"]
