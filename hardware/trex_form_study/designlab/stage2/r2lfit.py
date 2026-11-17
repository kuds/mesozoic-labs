"""R2L: Stage 1's tracker refitted on the bar's tasks, its selection and its rows (round C design D21, sections 2.2
and 3.8; Track B rules Q1-Q6 as settled; ``STAGE2_PLAN.md`` S2-18).  Simulation only: every number here is a
simulated episode's.  ``designlab/stage1/speedfit.py`` is imported and never edited; no round-A module is edited.

* **Rotation** (``rotation``, rule Q2): per generation g, block ``B = g // 6``, the episode seed
  ``tune.training_order(seed)[g]`` and five slots j = 0-4 (``walk/v0.4``, ``turn/q90_<side>``, ``stop/two_step``,
  ``nudge/<direction>``, ``walk/nominal``) at the servo delay ``k = (g + j) mod 6``; the side by ``B mod 2``, the
  direction by ``B mod 4``.  The table of 30 generations is written to ``rotation.json`` before generation 0.
* **Evaluator** (``Evaluator``, ``_evaluator``: ``tune.Pool``'s worker and factory): the five episodes of an
  evaluation, each built by ``bar.BarContext`` on the fit's frozen set and run as the bar's R2 row runs it
  (``panels.TrackerRow.episode``, ``run_slot``) inside ``speedfit.servo_delay``, judged by ``bar.judge``, then
  scored by the objective ``stage2-r2l-v1`` (``OBJECTIVE``, rule Q3: ``tune.score`` over the item terms).
* **Fit** (``fit``): ``tune.CMA``, population 12, sigma0 0.1, from ``speedfit.start_record(body)``'s gains, seed
  42; checkpointed per generation (``cma.json``, ``evals.jsonl``); the ledger ``ledger.json`` sums the evaluations'
  CPU seconds; blocks of six generations by rule Q5's block rule; at most 30 generations.
* **Selection** (``select``, rule Q4): the decisive cell first and whole (every candidate on validation seeds
  2000-2019 at k = 5), then the reported cells at k = 0, 2, 4, 1, 3 (the selected candidate, then the start) while
  the fit's share holds each cell's estimate (rule Q5 as settled on 16 Nov 2026); ``r2lg5.json``.
* **Publish** (``publish``): the fit's own finished records (``PUBLISH_RULE``) copied to
  ``stage2/tb/r2l/<set>/<body>/`` (``nb.collect``, each verified by sha256) and the entry
  ``{set, body, seed, r2lg5: {path, sha256}}`` added to ``index.json``.
* **Rows** (``r2l_row``, rule Q6): importing this module registers ``R2L-<set>`` for every set of the committed
  index (``panels.register_rows``); ``R2L-<set>/<seed>`` is ``lookbar.LookRow`` around ``R2LRow`` (the panels' R2
  tracker row with the indexed gains, their file's sha256 checked), which refuses a condition of another set.

Outputs default to ``stage2/tb/work/r2l/<set>/<body>/`` (ignored by ``stage2/tb/.gitignore``): ``start.json``,
``r2lfit/<seed>/{rotation.json, cma.json, evals.jsonl, ledger.json}``, ``r2lg5.json``, ``r2lg5_evals.jsonl``.

Run from hardware/trex_form_study (one BLAS thread)::

    python -m designlab.stage2.r2lfit {rotation|fit|select|publish} --body B --set S [--seed 42]
        [--out stage2/tb/work/r2l] [--workers 1] [--resume] [--budget-core-h 2.4] [--max-generations 30]
        [--popsize 12] [--reported-core-h 0.4] [--index stage2/tb/r2l/index.json]

``rotation`` prints the table (and whether it equals a written ``rotation.json``) and writes nothing.  Exit codes:
0; 1 a refusal.  A fit on a (set, body) that rule Q1 does not name, or with another seed, population, budget or
generation cap than the rules', is a development fit (``ROLE_RULE``): no record, and ``select`` and ``publish`` refuse
it; a selection with another ``--reported-core-h`` than 0.4 is a development selection, which ``publish`` refuses.  A
fit or a selection on a look set, or on a version of one, waits for G9-ref: as the gates package reads it
(``gates2.require_g9_ref``) and as ``lookbar``'s look-set rule lets the row ``R2L-<set>/<seed>`` run there
(``lookbar.look_set_rule``, the lead's ruling 16 of 17 Nov 2026) (``GATE_RULE``, ``gate``).

This file is in the sources of every R2L row (``R2LRow.sources``) and of every R2L record (``sources``).
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("disable")

import os  # noqa: E402
import sys  # noqa: E402

if __name__ == "__main__":  # before any numpy import (round A's module skeleton)
    for _v in tbenv.THREAD_VARS:
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from designlab.stage1 import control, panels, plant, speedfit, tune  # noqa: E402
from designlab.stage2 import bar, readings, s2io, setio  # noqa: E402

SCHEMA = "s2-r2l-1"
INDEX_SCHEMA = "s2-r2l-index-1"
OBJECTIVE_NAME = "stage2-r2l-v1"
CONTROLLER = "R2L"
MODE = tune.MODE  # R2: the tracker on the estimate
RATE = panels.RATE  # the bar's rows' commanded clock (1.05)
SEED = 42
#: Rule Q3's search (Stage 1's CMA-ES settings; a test holds them equal to ``tune``'s).
POPSIZE = 12
SIGMA0 = 0.1
GENERATIONS = 30
BLOCK = 6
DELAYS = 6  # k = 0..5 physics steps of 2 ms
#: The five slots of an evaluation (rule Q2), in order j = 0-4.
KINDS = ("walk", "turn", "stop", "nudge", "nominal")
SIDES = ("left", "right")
NUDGE_DIRECTIONS = ("forward", "left", "backward", "right")
#: Rule Q4's cells.
VALIDATION_SEEDS = tuple(range(2000, 2020))
DECISIVE_K = 5
REPORTED_K = (0, 2, 4, 1, 3)
REPORTED_SEEDS = tuple(range(2000, 2005))
#: Rule Q5's share of a fit (core-hours of summed evaluation CPU seconds) and its parts.
SHARE_CORE_H = 4.0
FIT_CORE_H = 2.4
DECISIVE_CORE_H = 1.2
REPORTED_CORE_H = 0.4
REPORTED_MARGIN = 1.25
#: Rule Q1's fits (set, body): the two heads on V6noF5 and the candidate on F25F9cl.
FITS = (("V6noF5", "75-M-H-10"), ("V6noF5", "60-N-T-10"), ("F25F9cl", "75-M-H-10"))
#: The look sets (``lookbar.LOOK_SETS``; a test holds the two equal): a fit or a selection there, or on a version of
#: one (``look_set_of``), waits for G9-ref (rules Q1, X1; ``GATE_RULE``).
LOOK_SETS = ("V6", "V6noF5")
#: Where things live (tests point these elsewhere).
WORK = s2io.STUDY / "stage2" / "tb" / "work" / "r2l"
INDEX = s2io.STUDY / "stage2" / "tb" / "r2l" / "index.json"
GATES = s2io.STUDY / "stage2" / "tb" / "gates"
SET_ROOT = setio.ROOT
EVAL_LOG = "r2lg5_evals.jsonl"
#: A fit's finished states (``cma.json`` ``status``).
FINISHED = ("done", "ended_by_budget")
#: The item terms' scales (rule Q3).
LINE_SCALE_M = readings.LINE_MAX_M  # 0.25 m
HEADING_SCALE_DEG = 20.0
TOUCHDOWNS_FREE = readings.STOP_TOUCHDOWNS_MAX  # 2
TOUCHDOWNS_SCALE = 2.0
STOP_SCALE_STRIDES = tune.STOP_WITHIN_STRIDES  # 1.5
RECOVERY_SCALE_S = panels.RECOVERY_S  # 3.0 s


class R2LError(s2io.S2Error):
    """R2L refuses (a resume on other settings or another table, a set that has not passed G9-ref, a development
    fit asked to be selected or published, a row on another set, a gains file that changed)."""


# ------------------------------------------------------------------------------------------------
# Rules (each copied into the records)
# ------------------------------------------------------------------------------------------------
ROTATION_RULE = (
    "STAGE2_TRACKB_PLAN rule Q2 (S2-18): every evaluation of generation g, block B = g // 6, runs five bar episodes "
    "at mu 0.9 (the bar's gated conditions without the _mu0.54 grip) in slots j = 0-4: walk/v0.4, turn/q90_<side>, "
    "stop/two_step, nudge/<direction> (0.25 x the push bound), walk/nominal; the episode seed is "
    "tune.training_order(seed)[g]; slot j runs at a servo delay of k = (g + j) mod 6 physics steps (2 k ms); the side "
    "is ('left', 'right')[B mod 2] and the direction ('forward', 'left', 'backward', 'right')[B mod 4], so within a "
    "block every slot meets all six delays in one direction, the two turn sides meet every delay in any two "
    "consecutive blocks and the four nudge directions in any four; slot j's start perturbation and sensor noise are "
    "drawn from tune.derived_seed(episode seed, j, 0) and tune.derived_seed(episode seed, j, 1) (Stage 1b's rule: "
    "five different starts per evaluation, unlike the bar's one seed slot, on purpose); a nudge's onset is the bar's "
    "draw for the episode seed (panels.push_u, tune.derived_seed(episode seed, 0, 2)); every member of a generation "
    "runs the same five episodes; the table of 30 generations is written to rotation.json, with written_at, before "
    "generation 0, and a resume refuses a table that differs; a fit that ends before 30 generations names in "
    "cma.json (rotation_met) the (kind, direction) pairs and the delays it did not meet")
EPISODE_RULE = (
    "an episode is run as the bar's R2 row runs it (panels.TrackerRow.episode: control.Tracker in mode R2 with the "
    "candidate's 14 gains, control.TERMS and control.ANKLE_CAP, at the bar's clock rate 1.05; control.run from the "
    "settled stance with tune's start noise and the condition's push) inside speedfit.servo_delay(b.p, k), on the "
    "plant and track bar.BarContext builds for the condition on the fit's frozen set; it differs from the bar's row "
    "only in its start and noise seeds (ROTATION_RULE); it is judged by bar.judge; an episode during which MuJoCo "
    "warned is a fall at its start (tune.warned_fall)")
TERMS_RULE = (
    "STAGE2_TRACKB_PLAN rule Q3 as r2lfit reads it (the builder's reading of the item terms, handed to the lead with "
    "its alternatives on 17 Nov 2026, before any fit).  Every term lies in [0, 1] and is "
    "graded by tune's functions (hinge, hinge_below, error: a value that was not measured scores 1); an episode that "
    "fell scores 1 on every term.  Tracking: tune.measure's over the item's judged window: on walk/v0.4 and "
    "walk/nominal its three walk terms dcm (the DCM error over the episode / 0.02 m), speed (|10 s mean speed - "
    "command| / 0.1 command) and heading (|heading change| / 0.1 rad) over the bar's window (the last 10 s); on the "
    "turn, the stop and the nudge its dcm alone (its speed and heading are a straight walk's and would score the "
    "commanded turn or stop as an error).  Limits: on the two walks tune.measure's walk limits over the window less "
    "the three 'at nominal' ones (cadence, stride, grf; panels.NOMINAL_ONLY_TERMS), with toe_clearance on the bar's "
    "own-swing reading (hinge_below(value, 0.04 H, 0.04 H); 1 when the bar's own-swing term fails on an interrupted "
    "swing or too few swings), plus along = hinge_below(along, target, target) (the shortfall of the distance along "
    "the commanded heading over the bar's target, 10 s x the target speed: 4.0 m on walk/v0.4) and line = "
    "min(1, line deviation / 0.25 m); on the turn the same window limits over the bar's turn window [command, "
    "command + at_s + 3 s] without the straight walk's speed, the joint range as the bar reads it there (its "
    "shortfall against decision 14's margins, hinge(x, 0, 0.10 rad)), the own-swing toe clearance, and heading_at = "
    "min(1, |heading error at at_s| / 20 deg) and heading_hold = min(1, largest |heading error| in the 3 s after / "
    "20 deg); on the stop and the nudge, which the bar judges over the episode, the episode limits as the bar reads "
    "them (drive_sustained: every drive's RMS over the episode against its rating, hinge(r, 1, 1); drive_speed: "
    "hinge(peak speed ratio, 1, 1); joint_range: hinge(shortfall, 0, 0.10 rad)), plus on the stop touchdowns = "
    "min(1, max(0, touchdowns after the command - 2) / 2) and stop_time = min(1, strides to stopped / 1.5) (1 when it "
    "never stopped), on the nudge recovery = min(1, recovery time / 3 s) (1 when it never recovered)")
SCORE_RULE = (
    "tune.score over the five episodes (falls first, then tracking and limit terms): J = S + P; S = sum over the "
    "fallen episodes of (1 + the share of the episode's planned seconds not survived); P = 0.5 T + 0.5 L, T the mean "
    "of every episode's tracking terms, L the mean over the five items of (largest + mean) / 2 of the item's limit "
    "terms (no start or stop groups); the CMA-ES fitness adds the box repair (tune.repair_penalty); capture_margin "
    "held at 0.0 (tune.FIXED_RULE)")
FAILURE_RULE = (
    "an evaluation that raises (an episode the bar cannot read among them: bar.Unreadable) scores as the worst case: "
    "every episode a fall at its start, every term 1, J = 11, with its traceback; a worker that dies or hangs is "
    f"replaced and its evaluation run again, up to {tune.ATTEMPTS} attempts, then scored as the worst case with the "
    "reason (tune.Pool)")
SEARCH_RULE = (
    "STAGE2_TRACKB_PLAN rule Q3: tune.CMA, population 12, sigma0 0.1 in the [0, 1] scaling of control.GAIN_BOUNDS, "
    "from the body's Stage 1 G5 selection (speedfit.start_record, speedfit.gains_vector), seed 42; the 13 searched "
    "gains, capture_margin held at 0.0; generation g on the rotation's episodes; at most 30 generations")
BUDGET_RULE = (
    "STAGE2_TRACKB_PLAN rule Q5 as settled on 15 and 16 Nov 2026: a fit's share is 4.0 core-hours of summed "
    "evaluation CPU seconds (each evaluation's time.process_time in the process that ran it; an evaluation counted "
    "once, where it was first kept), counted in ledger.json; block 0 (six generations) always runs; after each whole "
    "block the next starts only if the fit's summed seconds plus the seconds of its costliest block so far are at "
    "most 2.4 core-hours (budget_core_h), or 30 generations are done; a started block finishes and any excess over "
    "2.4 is recorded and comes off the 0.4 of the reported cells; the decisive cell then finishes whatever it costs "
    "(1.2 reserved; its overrun is recorded and is the only one that may pass the share without limit); the reported "
    "cells run in rule Q4's order, and a reported cell starts only when the share still holds its estimate (the "
    "decisive cell's mean seconds per episode x 25 x 1.25), what the share still holds being 4.0 core-hours less "
    "every second the fit has spent (its blocks, its decisive cell and its earlier reported cells); otherwise it "
    "reads 'not run: budget'; a started cell finishes, and if it passes the share all the same the excess is "
    "recorded (overrun_reported) and no further cell starts.  The share is budget_core_h + 1.2 + reported_core_h "
    "(4.0 with the rules' 2.4 and 0.4)")
SELECTION_RULE = (
    "STAGE2_TRACKB_PLAN rule Q4: candidates the start and the CMA-ES mean (projected into the bounds) after each "
    "third of the generations judged (tune.candidate_generations of the generations the fit ran); the decisive cell "
    "is run first and always whole: every candidate runs the five kinds on validation seeds 2000-2019 at k = 5 (turn "
    "to the left for even seeds, to the right for odd; nudge direction forward, left, backward, right for seed mod 4 "
    "= 0-3; slot j's start and noise seeds as ROTATION_RULE's), ranked by tune.selection_key on those 100 episodes "
    "per candidate alone (fewer falls, then the lower mean S, mean P and mean J, then the earlier candidate); then, "
    "reported and not used, as far as BUDGET_RULE reaches: at each of k = 0, 2, 4, 1, 3 in that order, the selected "
    "candidate and then the start, on validation seeds 2000-2004 with the same directions (25 episodes per cell); a "
    "cell not reached reads 'not run: budget'; every validation evaluation is kept in r2lg5_evals.jsonl as it "
    "finishes (keyed by the candidate's gains, the evaluation's episodes, the sources, the inputs and the versions) "
    "and reused by a rerun")
RESUME_RULE = (
    "speedfit.RESUME_RULE on R2L: the state is checkpointed after every generation (cma.json, written atomically "
    "after the generation's evaluations are appended to evals.jsonl and synced); resume continues from it exactly "
    "(tune.CMA.from_state) and drops evaluations logged after the checkpoint; refused if the settings, the rotation "
    "table (rotation.json against rotation(seed)), the sources (sources()), the library versions or the inputs (the "
    "set's files and freeze) differ from the run's first session; an existing run is never restarted over; a resume "
    "of a finished fit (done, ended_by_budget) changes nothing and returns its record")
ROLE_RULE = (
    "a fit is a record only on the (set, body) pairs rule Q1 names (V6noF5 on 75-M-H-10 and 60-N-T-10; F25F9cl on "
    "75-M-H-10, the candidate) with the rules' seed 42 (rules Q1, Q2), population 12, budget 2.4 core-hours and cap "
    "of 30 generations; any other fit is a development fit (a smoke check): its records say so, it is no record, and "
    "select and publish refuse it; a selection whose reported cells' part is not 0.4 core-hours (so a share other "
    "than rule Q5's 4.0) is a development selection: r2lg5.json says so (role), publish refuses it and no row reads "
    "it")
GATE_RULE = (
    "STAGE2_TRACKB_PLAN rules Q1, X1 and Z2, with the lead's rulings 16, 31 and 35 of 17 Nov 2026: a fit, and a "
    "selection, on a look set (V6, V6noF5) or on a version of one (lookbar.look_set_of: the part of the set's name "
    "before its first dot) starts only after G9-ref passed for that set and body, read in two ways before anything "
    "is written or run: (1) as the gates package reads it, the one way a G9-ref record is accepted: "
    "gates2.require_g9_ref(body, set, stage2/tb/gates/<set>/<body>/g9_ref.json) must return its citation (there: a "
    "G9-ref record of its schema, its own sha256 intact, of this set and body, pass_ true, made on the set's latest "
    "freeze); its refusal refuses, and so does no citation (the gates package not reading the set as a look set, a "
    "version of one included) or no gates package (designlab/stage2/gates2.py); (2) as lookbar's look-set rule lets "
    "the row R2L-<set>/<seed> run there (lookbar.look_set_rule(set, body, [R2L-<set>/<seed>], the set root, that "
    "record) and its check, as lookbar.episodes asks it), whose refusal refuses; the two must cite the same file "
    "with the same sha256; each fit session (gate) and each selection (g9_ref) records the citation (path, sha256, "
    "the record's time and freeze), the file's modification time, what checked it and the look-set rule's row and "
    "citation; on any other set nothing is read and the record holds None")
PUBLISH_RULE = (
    "publish copies only the fit's own finished records: start.json, r2lg5.json, r2lg5_evals.jsonl and "
    "r2lfit/<seed>/ (nothing else under <set>/<body>/, such as another seed's development fit), staged under "
    "<set>/<body>/.publish/<seed>/ and collected from there by nb.collect (each copy verified by sha256, "
    "COLLECTED.json), under the set and body's lock; a published file is never replaced")
ROWS_RULE = (
    "STAGE2_TRACKB_PLAN rule Q6: R2L-<set>/<seed> is the panels' tracker row in mode R2 (control.TERMS, "
    "control.ANKLE_CAP, the bar's clock) with the gains of the fit of seed <seed> on set <set> and the row's body, "
    "found through the committed index stage2/tb/r2l/index.json; the gains are r2lg5.json's selected_gains, read only "
    "when the file's sha256 equals the index entry's (and again in every worker); the row refuses a condition of "
    "another set (cond.set); it is wrapped in lookbar.LookRow, and bar.rows wraps it for @<ms>ms; its description "
    "names the gains file's path and sha256 (gains_info, gains_file)")

RULES = dict(rotation=ROTATION_RULE, episode=EPISODE_RULE, terms=TERMS_RULE, score=SCORE_RULE, failures=FAILURE_RULE,
             search=SEARCH_RULE, budget=BUDGET_RULE, selection=SELECTION_RULE, resume=RESUME_RULE, role=ROLE_RULE,
             gate=GATE_RULE, publish=PUBLISH_RULE, rows=ROWS_RULE)


def _now() -> str:
    """UTC with microseconds: written_at and started_at must order within one second."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _anchored(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else s2io.STUDY / p


# ------------------------------------------------------------------------------------------------
# The rotation (rule Q2)
# ------------------------------------------------------------------------------------------------
def condition_key(kind: str, side: str | None = None, direction: str | None = None) -> str:
    """The bar condition a slot runs (gated, mu 0.9; ``walk/nominal`` is reported)."""
    if kind == "walk":
        return f"walk/{bar.BAR_WALK}"
    if kind == "turn":
        if side not in SIDES:
            raise ValueError(f"turn side {side!r}: {', '.join(SIDES)}")
        return f"turn/q90_{side}"
    if kind == "stop":
        return "stop/two_step"
    if kind == "nudge":
        if direction not in NUDGE_DIRECTIONS:
            raise ValueError(f"nudge direction {direction!r}: {', '.join(NUDGE_DIRECTIONS)}")
        return f"nudge/{direction}"
    if kind == "nominal":
        return f"walk/{setio.NOMINAL}"
    raise ValueError(f"kind {kind!r}: choose from {', '.join(KINDS)}")


#: Every condition a fit or its selection can run.
CONDITIONS = tuple(dict.fromkeys(
    [condition_key("walk")] + [condition_key("turn", s) for s in SIDES] + [condition_key("stop")]
    + [condition_key("nudge", direction=d) for d in NUDGE_DIRECTIONS] + [condition_key("nominal")]))
#: The (kind, direction) labels the rotation's coverage is read on.
LABELS = ("walk",) + tuple(f"turn/{s}" for s in SIDES) + ("stop",) + tuple(f"nudge/{d}" for d in NUDGE_DIRECTIONS) + (
    "nominal",)


def slot_direction(kind: str, side: str, direction: str) -> str | None:
    return side if kind == "turn" else direction if kind == "nudge" else None


def slots(side: str, direction: str, ks) -> list[dict]:
    """The five slots of an evaluation: ``{j, kind, condition, k, direction}``."""
    ks = [int(k) for k in ks]
    if len(ks) != len(KINDS) or any(not 0 <= k < DELAYS for k in ks):
        raise ValueError(f"delays {ks}: five, each 0-{DELAYS - 1}")
    return [dict(j=j, kind=kind, condition=condition_key(kind, side, direction), k=k,
                 direction=slot_direction(kind, side, direction)) for j, (kind, k) in enumerate(zip(KINDS, ks))]


def label(slot: dict) -> str:
    return slot["kind"] + (f"/{slot['direction']}" if slot.get("direction") else "")


def coverage(table: list, generations: int | None = None) -> dict:
    """Per (kind, direction) the delays met in the table's first ``generations`` rows (all: None) and those not."""
    rows = table if generations is None else table[: int(generations)]
    met = {x: set() for x in LABELS}
    for row in rows:
        for s in row["slots"]:
            met[label(s)].add(int(s["k"]))
    return {x: dict(met=sorted(m), not_met=sorted(set(range(DELAYS)) - m)) for x, m in met.items()}


def rotation(seed: int = SEED, generations: int = GENERATIONS) -> dict:
    """Rule Q2's table (``ROTATION_RULE``): per generation the block, the episode seed and the five slots (kind,
    condition, delay, direction, start and noise seeds), with the coverage per (kind, direction)."""
    order = tune.training_order(seed)
    n = int(generations)
    if not 1 <= n <= len(order):
        raise ValueError(f"generations {generations!r}: 1-{len(order)}")
    table = []
    for g in range(n):
        b = g // BLOCK
        side, direction = SIDES[b % 2], NUDGE_DIRECTIONS[b % 4]
        ep = int(order[g])
        ss = slots(side, direction, [(g + j) % DELAYS for j in range(len(KINDS))])
        for s in ss:
            s.update(start_seed=tune.derived_seed(ep, s["j"], 0), noise_seed=tune.derived_seed(ep, s["j"], 1))
        table.append(dict(generation=g, block=b, episode=ep, side=side, direction=direction, slots=ss))
    return dict(schema="s2-r2l-rotation-1", seed=int(seed), generations=n, block=BLOCK, delays_ms=[2 * k for k in
                range(DELAYS)], rule=ROTATION_RULE, table=table, coverage=coverage(table))


def spec_of(row: dict) -> dict:
    """An evaluation's episodes: the episode seed and, per slot, j, kind, condition and k (what the evaluator and
    the log keys read)."""
    return dict(seed=int(row["episode"]), slots=[{k: s[k] for k in ("j", "kind", "condition", "k")}
                                                 for s in row["slots"]])


def validation_spec(seed: int, k: int) -> dict:
    """A validation evaluation (rule Q4): the five kinds at delay ``k``, the turn to the left for an even seed, the
    nudge direction by ``seed mod 4``."""
    s = int(seed)
    return spec_of(dict(episode=s, slots=slots(SIDES[s % 2], NUDGE_DIRECTIONS[s % 4], [k] * len(KINDS))))


def rotation_text(doc: dict) -> str:
    lines = [f"R2L rotation, seed {doc['seed']}, {doc['generations']} generations in blocks of {doc['block']} "
             "(slot@delay in ms; rule Q2)"]
    for row in doc["table"]:
        cells = "  ".join(f"{s['condition']}@{2 * s['k']}ms" for s in row["slots"])
        lines.append(f"g{row['generation']:>2} B{row['block']} ep {row['episode']}  {cells}")
    lines.append("delays met per (kind, direction): " + "; ".join(
        f"{x} {','.join(str(2 * k) for k in c['met'])} ms" for x, c in doc["coverage"].items()))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# The objective (rule Q3)
# ------------------------------------------------------------------------------------------------
_WALK_WINDOW = tuple(t for t in tune.LIMITS["walk"] if t not in panels.NOMINAL_ONLY_TERMS and t != "toe_clearance")
_TURN_WINDOW = tuple(t for t in _WALK_WINDOW if t not in ("speed", "joint_range"))
TOE = "toe_clearance_own_swing"
TRACKING = {"walk": tune.TRACKING["walk"], "turn": ("dcm",), "stop": ("dcm",), "nudge": ("dcm",),
            "nominal": tune.TRACKING["walk"]}
LIMITS = {"walk": _WALK_WINDOW + (TOE, "along", "line"),
          "turn": _TURN_WINDOW + ("joint_range", TOE, "heading_at", "heading_hold"),
          "stop": tune.EPISODE_LIMITS + ("touchdowns", "stop_time"),
          "nudge": tune.EPISODE_LIMITS + ("recovery",),
          "nominal": _WALK_WINDOW + (TOE, "along", "line")}
TERM_ORDER = tuple(f"{k}.tracking.{n}" for k in KINDS for n in TRACKING[k]) + tuple(
    f"{k}.limits.{n}" for k in KINDS for n in LIMITS[k])
OBJECTIVE = dict(
    name=OBJECTIVE_NAME, defined="17 Nov 2026, before any R2L fit, from STAGE2_TRACKB_PLAN rule Q3 and round C "
                                 "design section 3.8 (r2lfit's reading of the item terms: see TERMS_RULE)",
    base=tune.OBJECTIVE["name"], controller=MODE, gains=list(tune.SEARCHED), fixed_gains=dict(tune.FIXED_GAINS),
    fixed_rule=tune.FIXED_RULE, kinds=list(KINDS), rotation=ROTATION_RULE, episodes=EPISODE_RULE, terms=TERMS_RULE,
    score=SCORE_RULE, failures=FAILURE_RULE, tracking_terms={k: list(v) for k, v in TRACKING.items()},
    limit_terms={k: list(v) for k, v in LIMITS.items()}, term_order=list(TERM_ORDER), scales=dict(
        tune.SCALES, line_m=LINE_SCALE_M, heading_deg=HEADING_SCALE_DEG, touchdowns_free=TOUCHDOWNS_FREE,
        touchdowns=TOUCHDOWNS_SCALE, stop_strides=STOP_SCALE_STRIDES, recovery_s=RECOVERY_SCALE_S),
    stage1_terms=tune.TERMS_RULE)
WORST_J = 2.0 * len(KINDS) + 1.0


def _bar_of(e: dict) -> dict:
    return ((e.get("measures") or {}).get("bar")) or {}


def toe_term(toe: dict | None, H: float) -> float:
    """The own-swing toe clearance as a limit term (``TERMS_RULE``)."""
    if not toe:
        return 1.0
    v = toe.get("toe_clearance_own_swing_m")
    if v is None or toe.get("interrupted_swings") or not toe.get("count_ok", True):
        return 1.0
    lim = readings.TOE_CLEARANCE_FRAC_H * float(H)
    return tune.hinge_below(v, lim, lim)


def _worst_drive(drives: dict | None) -> float | None:
    w = (drives or {}).get("worst")
    return None if not w else float(w[1])


def episode_terms(kind: str, e: dict, m: dict | None, H: float) -> tuple[dict, dict, dict]:
    """(tracking, limits, readings) of one judged episode (``TERMS_RULE``): ``e`` the bar's episode record
    (``bar.judge``), ``m`` ``tune.measure``'s result over the item's judged window (None: not measured), ``H`` the hip
    height.  A fallen episode scores 1 on every term."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r}")
    ones = ({n: 1.0 for n in TRACKING[kind]}, {n: 1.0 for n in LIMITS[kind]})
    bm = _bar_of(e)
    m = m or {}
    rd = dict(dcm_m=m.get("dcm_error_rms_m"))
    if e.get("fell"):
        return ones[0], ones[1], rd
    mt, ml = m.get("tracking") or {}, m.get("limits") or {}
    if kind in ("walk", "nominal"):
        tracking = {n: float(mt.get(n, 1.0)) for n in TRACKING[kind]}
    else:
        tracking = dict(dcm=tune.error(m.get("dcm_error_rms_m"), tune.SCALES["dcm_m"]))
    lim: dict = {}
    if kind in ("walk", "nominal"):
        for n in _WALK_WINDOW:
            lim[n] = float(ml.get(n, 1.0))
        along, target = bm.get("along_commanded_heading_m"), bm.get("along_target_m")
        line = bm.get("lateral_from_line_max_m")
        lim[TOE] = toe_term(bm.get("toe"), H)
        lim["along"] = tune.hinge_below(along, target, target) if target else 1.0
        lim["line"] = tune.error(line, LINE_SCALE_M)
        rd.update(along_m=along, along_target_m=target, line_m=line,
                  toe_own_swing_m=(bm.get("toe") or {}).get("toe_clearance_own_swing_m"))
    elif kind == "turn":
        for n in _TURN_WINDOW:
            lim[n] = float(ml.get(n, 1.0))
        jr = bm.get("joint_range") or {}
        lim["joint_range"] = tune.hinge(jr.get("joint_shortfall_rad"), 0.0, tune.SCALES["joint_range_rad"])
        lim[TOE] = toe_term(bm.get("toe"), H)
        at, hold = bm.get("heading_error_at_deg"), bm.get("heading_error_hold_max_deg")
        lim["heading_at"] = tune.error(at, HEADING_SCALE_DEG)
        lim["heading_hold"] = tune.error(hold, HEADING_SCALE_DEG)
        rd.update(heading_at_deg=at, heading_hold_deg=hold, joint_shortfall_rad=jr.get("joint_shortfall_rad"),
                  toe_own_swing_m=(bm.get("toe") or {}).get("toe_clearance_own_swing_m"))
    else:
        dr = _worst_drive((bm.get("drives") or {}).get("episode"))
        lim["drive_sustained"] = tune.hinge(dr, 1.0, 1.0)
        spd = e.get("worst_speed_ratio")
        lim["drive_speed"] = tune.hinge(None if not spd else float(spd[1]), 1.0, 1.0)
        lim["joint_range"] = tune.hinge(e.get("joint_shortfall_rad"), 0.0, tune.SCALES["joint_range_rad"])
        rd.update(drive_sustained_ratio=dr, joint_shortfall_rad=e.get("joint_shortfall_rad"))
        if kind == "stop":
            n = bm.get("touchdowns_after_command")
            strides = bm.get("forward_stopped_strides")
            lim["touchdowns"] = (1.0 if n is None else
                                 float(min(1.0, max(0.0, float(n) - TOUCHDOWNS_FREE) / TOUCHDOWNS_SCALE)))
            lim["stop_time"] = tune.error(strides, STOP_SCALE_STRIDES)
            rd.update(touchdowns=n, stop_strides=strides, stopped_by_end=bm.get("stopped_by_end"))
        else:
            rcv = (e.get("measures") or {}).get("recovery") or {}
            after = rcv.get("after_s") if rcv.get("recovered") else None
            lim["recovery"] = tune.error(after, RECOVERY_SCALE_S)
            rd.update(recovery_s=rcv.get("after_s"), recovered=rcv.get("recovered"))
    return tracking, {n: lim[n] for n in LIMITS[kind]}, rd


def blank_episode(slot: dict) -> dict:
    """An episode of ``slot`` that survived with every term 0 (tests build their stub evaluations on it)."""
    kind = slot["kind"]
    return dict(kind=kind, condition=slot["condition"], k=int(slot["k"]), fell=False, fall_time_s=None,
                fall_cause=None, survived_frac=1.0, seconds=0.0, planned_seconds=None, succeeded=True, failed_terms=[],
                tracking={n: 0.0 for n in TRACKING[kind]}, limits={n: 0.0 for n in LIMITS[kind]}, start_terms={},
                stop_terms={}, readings={})


def failed_episode(slot: dict, cause: str) -> dict:
    """An episode scored as the worst case: a fall at its start, every term 1 (``FAILURE_RULE``)."""
    e = blank_episode(slot)
    e.update(fell=True, fall_time_s=0.0, fall_cause=cause, survived_frac=0.0, succeeded=False,
             failed_terms=["survived"], tracking={n: 1.0 for n in e["tracking"]},
             limits={n: 1.0 for n in e["limits"]})
    return e


def term_vector(eps: dict) -> list:
    out = []
    for name in TERM_ORDER:
        kind, group, n = name.split(".")
        out.append(round(float(eps[kind][group][n]), 6))
    return out


def scored(spec: dict, eps: dict, wall_s: float = 0.0, cpu_s: float = 0.0, error: str | None = None) -> dict:
    """An evaluation's result from its five episodes (``SCORE_RULE``)."""
    if list(eps) != list(KINDS):
        raise ValueError(f"episodes {list(eps)}: one per kind, in the order {', '.join(KINDS)}")
    out = dict(tune.score(eps), episode_seed=int(spec["seed"]), spec=spec, episodes=eps, terms=term_vector(eps),
               wall_s=float(wall_s), cpu_s=float(cpu_s), simulated_s=float(sum(e["seconds"] for e in eps.values())))
    if error is not None:
        out["error"] = str(error)
    return out


def failed_evaluation(spec: dict, reason: str, wall_s: float = 0.0, cpu_s: float = 0.0) -> dict:
    """An evaluation scored as the worst case (``FAILURE_RULE``)."""
    eps = {s["kind"]: failed_episode(s, "evaluation failed") for s in spec["slots"]}
    return scored(spec, eps, wall_s, cpu_s, error=reason)


# ------------------------------------------------------------------------------------------------
# Episodes and the evaluator
# ------------------------------------------------------------------------------------------------
def run_slot(ctx: bar.BarContext, cond, gains, seed: int, j: int, k: int):
    """One episode of slot ``j`` (``EPISODE_RULE``): ``panels.TrackerRow.episode`` statement for statement (mode R2,
    the candidate's gains, control's terms and ankle cap, the bar's clock, the condition's push for ``seed``), with
    slot ``j``'s start and noise seeds, inside ``speedfit.servo_delay(b.p, k)``.  At ``j = 0`` the seeds are the
    bar's (``panels.episode_seeds`` gives every bar item slot 0), so the episode is the bar's R2 row's at ``k``
    (a slow test checks it).  Returns ``(built, outcome, recorder)``."""
    b = ctx.built(cond)
    p = b.p
    tk = control.Tracker(p, b.track, MODE, gains=np.asarray(gains, float), terms=control.TERMS,
                         ankle_cap=control.ANKLE_CAP, rate=RATE)
    rec = b.recorder()
    with speedfit.servo_delay(p, int(k)):
        out = control.run(tk, b.seconds, start="stance", seed=tune.derived_seed(seed, j, 0),
                          noise_seed=tune.derived_seed(seed, j, 1), recorder=rec,
                          joint_noise=tune.START_JOINT_NOISE_RAD,
                          root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S,
                          push=b.push(int(seed)), relax=())
    return b, tune.warned_fall(out, tune.sim_warnings(p.d)), rec


def judged_window(kind: str, b, e: dict, rec) -> tuple | None:
    """The item's judged window (simulated seconds) for ``tune.measure``: a walk's last 10 s, a turn's [command,
    command + at_s + 3 s]; None for the stop and the nudge (judged over the episode) or where there is none."""
    if e.get("fell") or not rec.rows:
        return None
    if kind in ("walk", "nominal"):
        tb = float(rec.t[-1])
        return (tb - tune.WALK_WINDOW_S, tb)
    if kind == "turn" and e.get("window_reached") and e.get("command_s") is not None:
        t_c = float(e["command_s"])
        return (t_c, t_c + float(b.resolved["at_s"]) + readings.HEADING_HOLD_S)
    return None


class Evaluator:
    """One body's plant and the fit's nine bar conditions on its frozen set, built once (``bar.BarContext``);
    ``evaluate(gains, spec)`` runs an evaluation's five episodes (``EPISODE_RULE``) and scores them
    (``SCORE_RULE``).  The set must be frozen and every file its conditions read must be in the freeze
    (``setio.read_inputs``); ``expected_inputs`` (a replacement worker's) must be equal."""

    def __init__(self, body: str, set_name: str, run_dir, expected_inputs: dict | None = None):
        s2io.require_one_thread()
        bar.die_with_parent()
        self.body, self.set_name, self.run_dir = body, str(set_name), Path(run_dir)
        docs_all = setio.load(self.run_dir)
        set_doc = docs_all["set.json"]
        if set_doc.get("set") != self.set_name:
            raise R2LError(f"{self.run_dir}: set.json names the set {set_doc.get('set')!r}, not {self.set_name!r}")
        bar.check_manifest(set_doc)
        planner = bar.BarPlanner(self.run_dir, docs_all["refs.json"]["references"],
                                 docs_all["transitions.json"]["transitions"], set_doc)
        known = {c.key: c for c in bar.conditions(docs_all, bar.HEADLINE_BLOCK, every=True)}
        self.conds, self.resolved = {}, {}
        for key in CONDITIONS:
            c = known.get(key)
            why = None if c is None else planner.missing(c)
            if c is None or why is not None:
                raise R2LError(f"{self.set_name}/{body}: the fit's condition {key} cannot be run "
                               f"({'not defined' if c is None else why.get('reason')})")
            self.conds[key], self.resolved[key] = c, planner.resolve(c)
        files = sorted({f for r in self.resolved.values() for f in r["files"]})
        self.inputs, docs = setio.read_inputs(self.run_dir, files, require_frozen=True)
        self.inputs["dir"] = s2io.portable(self.inputs["dir"])
        if expected_inputs is not None:
            diff = input_differences(expected_inputs, self.inputs)
            if diff:
                raise tune.InputError(f"{self.run_dir}: the inputs differ from the ones expected: " + "; ".join(diff))
        self.crouch = bar.set_crouch(set_doc, docs["refs.json"]["references"], bar.walks_of(files))
        self.ctx = bar.BarContext(body, self.run_dir, docs, set_doc, self.crouch)

    def describe(self) -> dict:
        return dict(set=self.set_name, body=self.body, crouch_frac=self.crouch, freeze=self.inputs.get("freeze"),
                    conditions={k: dict(plan=[list(x) for x in r["plan"]], seconds=float(r["seconds"]),
                                        files=list(r["files"])) for k, r in self.resolved.items()})

    def episode(self, gains, slot: dict, seed: int) -> dict:
        """One episode and its objective record (``TERMS_RULE``)."""
        kind = slot["kind"]
        c0 = time.process_time()
        b, out, rec = run_slot(self.ctx, self.conds[slot["condition"]], gains, int(seed), int(slot["j"]),
                               int(slot["k"]))
        e = bar.judge(b, out, rec, int(seed), self.ctx.lg if b.legs else None)
        window = judged_window(kind, b, e, rec)
        m = tune.measure(b.spec, rec, out, None, window=window)
        tracking, limits, rd = episode_terms(kind, e, m, b.H)
        fell = bool(e["fell"])
        planned = float(e["planned_s"])
        res = dict(kind=kind, condition=slot["condition"], k=int(slot["k"]), fell=fell, fall_time_s=e["fall_time_s"],
                   fall_cause=e["fall_cause"], survived_frac=1.0 if not fell else
                   min(1.0, max(0.0, float(e["fall_time_s"] or 0.0) / planned)), seconds=float(e["survival_s"]),
                   planned_seconds=planned, succeeded=bool(e["succeeded"]), failed_terms=list(e["failed_terms"]),
                   window_reached=bool(e.get("window_reached")), judged_window_s=None if window is None else
                   [float(window[0]), float(window[1])], tracking=tracking, limits=limits, start_terms={},
                   stop_terms={}, readings=rd)
        if e.get("mujoco_warnings"):
            res["mujoco_warnings"] = e["mujoco_warnings"]
        res["cpu_s"] = time.process_time() - c0
        return res

    def evaluate(self, gains, spec: dict) -> dict:
        """The five episodes of ``spec`` for ``gains``; an evaluation that raises scores as the worst case."""
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            eps = {s["kind"]: self.episode(gains, s, spec["seed"]) for s in spec["slots"]}
            return scored(spec, eps, time.perf_counter() - t0, time.process_time() - c0)
        except Exception:
            return failed_evaluation(spec, traceback.format_exc(), time.perf_counter() - t0,
                                     time.process_time() - c0)


# ------------------------------------------------------------------------------------------------
# Sources, workers
# ------------------------------------------------------------------------------------------------
def sources() -> dict:
    """What a fit and its selection run, by sha256: this file, the bar's sources (``bar.bar_sources``: bar.py,
    readings.py, setio.py, speedfit.py), the panels' sources and the tuner's; ``sha256`` over the four."""
    parts = dict(r2lfit=s2io.sha256_file(Path(__file__).resolve()), bar=bar.bar_sources()["sha256"],
                 panels=panels.panel_sources()["sha256"], tuner=tune.tuner_sources()["sha256"])
    return dict(parts, sha256=s2io.sha256_json(parts))


def _evaluator(body: str, run_dir: str, mode: str, rate: float, expected_inputs: dict | None = None, *,
               set_name: str, sources_sha256: str | None = None) -> Evaluator:
    """``tune.Pool``'s factory (top level, bound with ``functools.partial``): an evaluator, refused on other sources
    than the session's (``RESUME_RULE``)."""
    if mode != MODE or float(rate) != RATE:
        raise ValueError(f"mode {mode!r} at rate {rate!r}: R2L runs {MODE} at {RATE:g}")
    if sources_sha256 is not None:
        mine = sources()["sha256"]
        if mine != sources_sha256:
            raise tune.PoolError(f"R2L's sources changed during the session ({mine[:12]}..., the session's "
                                 f"{sources_sha256[:12]}...): restore them and resume")
    return Evaluator(body, set_name, run_dir, expected_inputs)


#: The evaluation pool (``tune.Pool``; tests may replace it).
Pool = tune.Pool


def _pool(body: str, set_name: str, workers: int, sources_sha256: str):
    run_dir = setio.set_dir(set_name, body, SET_ROOT)
    return Pool(body, run_dir, workers, mode=MODE, rate=RATE,
                factory=functools.partial(_evaluator, set_name=set_name, sources_sha256=sources_sha256))


def input_differences(a: dict | None, b: dict | None) -> list[str]:
    """``tune.input_differences`` (each file's sha256 and entry hash) and the set's freeze."""
    out = tune.input_differences(a, b)
    if a and b and a.get("freeze") != b.get("freeze"):
        out.append(f"the set's freeze: {a.get('freeze')} -> {b.get('freeze')}")
    return out


def _map(pool, jobs: list) -> list:
    """The pool's results of ``(gains, spec)`` jobs, in job order; a job whose worker was lost on every attempt
    scores as the worst case (``FAILURE_RULE``)."""
    return pool.map([(np.asarray(g, float), s) for g, s in jobs],
                    failed=lambda job, reason: failed_evaluation(job[1], reason))


# ------------------------------------------------------------------------------------------------
# Paths, roles, the gate, the start
# ------------------------------------------------------------------------------------------------
def set_body_dir(out, set_name: str, body: str) -> Path:
    for part in (set_name, body):
        if not part or Path(part).name != part or part in (".", ".."):
            raise ValueError(f"{part!r} is not a set or body name")
    return Path(out) / set_name / body


def fit_dir(out, set_name: str, body: str, seed: int) -> Path:
    return set_body_dir(out, set_name, body) / "r2lfit" / str(int(seed))


def _out(out) -> Path:
    return WORK if out is None else _anchored(out)


def role_of(set_name: str, body: str, popsize: int = POPSIZE, budget_core_h: float = FIT_CORE_H,
            max_generations: int = GENERATIONS, seed: int = SEED) -> dict:
    """``ROLE_RULE``: ``{role: "fit" | "development", reasons}``."""
    why = []
    if (set_name, body) not in FITS:
        why.append(f"rule Q1 names no fit of {body} on {set_name}")
    if int(seed) != SEED:
        why.append(f"seed {seed}, not {SEED} (rules Q1, Q2)")
    if int(popsize) != POPSIZE:
        why.append(f"population {popsize}, not {POPSIZE}")
    if float(budget_core_h) != FIT_CORE_H:
        why.append(f"budget {budget_core_h:g} core-hours, not {FIT_CORE_H:g}")
    if int(max_generations) != GENERATIONS:
        why.append(f"at most {max_generations} generations, not {GENERATIONS}")
    return dict(role="development" if why else "fit", reasons=why, rule=ROLE_RULE)


GATES2 = "designlab.stage2.gates2"


def _gates2():
    """The gates package's module (``GATE_RULE``), or None where it is not here; any other import error is raised."""
    import importlib

    try:
        return importlib.import_module(GATES2)
    except ModuleNotFoundError as exc:
        if exc.name == GATES2:
            return None
        raise


def look_set_of(set_name) -> str | None:
    """The look set ``set_name`` stands for, as ``lookbar`` reads it (``lookbar.look_set_of``: one of ``LOOK_SETS``,
    the name itself or the part before its first dot, a version of one; the lead's ruling 31 of 17 Nov 2026, point
    3); None for any other set."""
    from designlab.stage2 import lookbar

    return lookbar.look_set_of(set_name)


def row_name(set_name: str, seed: int = SEED) -> str:
    """The R2L row of the fit of ``seed`` on ``set_name`` (``ROWS_RULE``): ``R2L-<set>/<seed>``."""
    return f"{CONTROLLER}-{set_name}/{int(seed)}"


def look_rule(set_name: str, body: str, seed: int = SEED, g9_ref=None) -> dict:
    """``GATE_RULE``'s second reading: ``lookbar``'s look-set rule (``lookbar.look_set_rule``; the lead's ruling 16 of
    17 Nov 2026) for the row ``R2L-<set>/<seed>`` on ``set_name``, ``body`` and the set root ``SET_ROOT``, with the
    G9-ref record ``g9_ref``, asked as ``lookbar.episodes`` asks it (``LookSetRule.check``, before anything is read or
    run).  Returns ``{checked_by, row, look_set, cited: {path, sha256}}``; ``R2LError`` where the rule refuses the row
    or lets it in without a G9-ref citation."""
    from designlab.stage2 import lookbar

    row = row_name(set_name, seed)
    try:
        with lookbar.look_set_rule(set_name, body, [row], SET_ROOT, g9_ref=g9_ref) as rule:
            cited = rule.check()
    except (lookbar.LookBarError, bar.BarError) as exc:
        raise R2LError(f"{set_name}/{body}: lookbar's look-set rule does not let {row} run ({exc}): nothing starts "
                       "(rule X1; the lead's ruling 16 of 17 Nov 2026; GATE_RULE)") from None
    if not isinstance(cited, dict) or not cited.get("path") or not cited.get("sha256"):
        raise R2LError(f"{set_name}/{body}: lookbar's look-set rule cited no G9-ref record for {row} ({cited!r}): "
                       "nothing starts (GATE_RULE)")
    return dict(checked_by="designlab.stage2.lookbar.look_set_rule", row=row, look_set=rule.look_set,
                cited=dict(path=cited["path"], sha256=cited["sha256"]))


def gate(set_name: str, body: str, seed: int = SEED) -> dict | None:
    """``GATE_RULE``: on a look set or a version of one (``look_set_of``), the citation of the passed G9-ref record as
    ``gates2.require_g9_ref`` gives it, with the file's modification time and ``look_set_rule``, ``lookbar``'s rule
    letting ``R2L-<set>/<seed>`` run (``look_rule``) on the same file; None on any other set (nothing is read).
    ``R2LError`` where the gates package refuses it, gives no citation or is not here, where lookbar's rule refuses,
    or where the two read different files."""
    look = look_set_of(set_name)
    if look is None:
        return None
    path = Path(GATES) / set_name / body / "g9_ref.json"
    g2 = _gates2()
    if g2 is None:
        raise R2LError(f"{path}: the gates package ({GATES2}) is not here, so G9-ref cannot be read: R2L's set "
                       f"{set_name} must pass G9-ref for {body} before its first episode (rules Q1, X1, Z2; "
                       "GATE_RULE): nothing starts")
    try:
        cite = g2.require_g9_ref(body, set_name, path=path, root=SET_ROOT)
    except g2.GateError as exc:
        raise R2LError(f"{path}: G9-ref has not passed as the gates package reads it ({exc}): nothing starts "
                       "(rules Q1, X1, Z2)") from None
    if not isinstance(cite, dict) or not cite.get("path") or not cite.get("sha256"):
        what = "a look set" if look == set_name else f"a version of the look set {look}"
        raise R2LError(f"{path}: {GATES2}.require_g9_ref gave no citation (path, sha256) of a G9-ref record "
                       f"({cite!r}; None: the gates package does not read {set_name}, {what}, as a look set): "
                       "nothing starts (GATE_RULE)")
    ctx = look_rule(set_name, body, seed, path)
    if (ctx["cited"]["path"], ctx["cited"]["sha256"]) != (cite["path"], cite["sha256"]):
        raise R2LError(f"{path}: the gates package and lookbar's look-set rule cite different G9-ref records "
                       f"({cite['path']} {str(cite['sha256'])[:12]}..., {ctx['cited']['path']} "
                       f"{str(ctx['cited']['sha256'])[:12]}...): nothing starts (GATE_RULE)")
    st = path.stat()
    return dict(cite, modified_at=datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"), checked_by=f"{GATES2}.require_g9_ref", look_set_rule=ctx, rule=GATE_RULE)


def start_key(rec: dict) -> dict:
    return speedfit.start_key(rec)


def _write_start(sd: Path, start: dict, log) -> None:
    """The body's start, once per set and body (``speedfit.START_RULE``); a fit from another is refused."""
    sp = sd / "start.json"
    old = panels._read(sp)
    if old is None:
        s2io.write_json(sp, dict(start, recorded_at=_now()))
        log(f"{sd.parent.name}/{sd.name}: start recorded ({start['source']})")
    elif start_key(old) != start_key(start):
        raise R2LError(f"{sp}: the fit starts from {old.get('start_from')} {old.get('candidate')}, not "
                       f"{start['start_from']} {start['candidate']}: every session starts from the recorded start")


def _write_rotation(path: Path, table: dict) -> dict:
    """``rotation.json`` before generation 0: written with ``written_at``, or an equal one kept as it is; one
    with another table is refused."""
    old = panels._read(path)
    if old is not None:
        if s2io.sha256_json(old.get("table")) != s2io.sha256_json(table["table"]):
            raise R2LError(f"{path}: another rotation table than rotation({table['seed']}) (ROTATION_RULE): refused")
        return old
    doc = dict(table, written_at=_now(), table_sha256=s2io.sha256_json(table["table"]))
    s2io.write_json(path, doc)
    return doc


def _provenance(out: Path) -> dict:
    prov = plant.provenance()
    prov["git"] = s2io.git_state(own_outputs=[str(Path(out).resolve())])
    prov["blas_threads"] = s2io.blas_threads()
    return prov


# ------------------------------------------------------------------------------------------------
# The fit
# ------------------------------------------------------------------------------------------------
def cma_settings(popsize: int, max_generations: int) -> dict:
    return dict(tune.CMA_SETTINGS, popsize=int(popsize), sigma0=SIGMA0, generations=int(max_generations),
                rule=SEARCH_RULE,
                x0="the start (speedfit.start_record) of the searched gains in the [0, 1] scaling of their "
                   "control.GAIN_BOUNDS")


def _settings(body, set_name, seed, start, budget_core_h, max_generations, popsize, table) -> dict:
    return dict(body=body, set=set_name, seed=int(seed), mode=MODE, rate=RATE, objective=OBJECTIVE_NAME,
                cma=cma_settings(popsize, max_generations), budget_core_h=float(budget_core_h),
                max_generations=int(max_generations), block=BLOCK, rotation_sha256=s2io.sha256_json(table["table"]),
                start=start_key(start), role=role_of(set_name, body, popsize, budget_core_h, max_generations, seed))


def _episode_line(e: dict) -> dict:
    out = dict(cond=e["condition"], k=e["k"], fell=e["fell"], t_fall=e["fall_time_s"], cause=e["fall_cause"],
               surv=round(float(e["survived_frac"]), 6), ok=e["succeeded"], failed=e.get("failed_terms"))
    if e.get("mujoco_warnings"):
        out["warn"] = e["mujoco_warnings"]
    return out


def _eval_line(gen: int, i: int, x, fit_value: float, r: dict) -> dict:
    line = dict(gen=gen, i=i, episode=r["episode_seed"], x=[float(v) for v in x], J=r["J"], fitness=fit_value,
                S=r["S"], P=r["P"], T=r["T"], L=r["L"], groups=r["groups"], falls=r["falls"],
                episodes={k: _episode_line(e) for k, e in r["episodes"].items()}, terms=r["terms"],
                sim_s=round(float(r["simulated_s"]), 3), wall_s=round(float(r["wall_s"]), 3),
                cpu_s=float(r.get("cpu_s") or 0.0))
    if r.get("error"):
        line["error"] = r["error"]
    return line


def block_costs(history: list) -> list[dict]:
    """The fit's blocks of six generations with their summed evaluation CPU seconds (the last one possibly
    unfinished)."""
    out = {}
    for h in history:
        b = int(h["gen"]) // BLOCK
        x = out.setdefault(b, dict(block=b, generations=[], cpu_s=0.0))
        x["generations"].append(int(h["gen"]))
        x["cpu_s"] += float(h["cpu_s"])
    for x in out.values():
        x["complete"] = len(x["generations"]) == BLOCK
        x["generations"] = [min(x["generations"]), max(x["generations"])]
    return [out[b] for b in sorted(out)]


def next_block(history: list, budget_core_h: float) -> dict:
    """``BUDGET_RULE``'s decision at a block boundary: whether the next block starts."""
    blocks = block_costs(history)
    spent = float(sum(b["cpu_s"] for b in blocks))
    costliest = float(max((b["cpu_s"] for b in blocks), default=0.0))
    limit = float(budget_core_h) * 3600.0
    return dict(at_generation=len(history), spent_s=spent, costliest_block_s=costliest, limit_s=limit,
                start=bool(spent + costliest <= limit))


def share_core_h(budget_core_h: float, reported_core_h: float) -> float:
    """A fit's share (``BUDGET_RULE``): its budget, the decisive cell's 1.2 and the reported cells' part (4.0 with
    the rules' 2.4 and 0.4), rounded to 1e-9 core-hours."""
    return round(float(budget_core_h) + DECISIVE_CORE_H + float(reported_core_h), 9)


def ledger_doc(doc: dict, sel: dict | None = None) -> dict:
    """``ledger.json``: the seconds of each block, of the decisive cell and of each reported cell, in the order run
    (``BUDGET_RULE``)."""
    st = doc["settings"]
    blocks = block_costs(doc["history"])
    fit_s = float(sum(b["cpu_s"] for b in blocks))
    order = [dict(item=f"block {b['block']}", generations=b["generations"], cpu_s=b["cpu_s"], complete=b["complete"])
             for b in blocks]
    out = dict(schema="s2-r2l-ledger-1", set=st["set"], body=st["body"], seed=st["seed"], rule=BUDGET_RULE,
               unit="CPU seconds of evaluations (time.process_time of the process that ran each)",
               share_core_h=share_core_h(st["budget_core_h"], REPORTED_CORE_H), fit_core_h=st["budget_core_h"],
               decisive_core_h=DECISIVE_CORE_H, reported_core_h=REPORTED_CORE_H, blocks=blocks,
               block_decisions=doc.get("block_decisions", []), fit_cpu_s=fit_s,
               fit_excess_s=max(0.0, fit_s - 3600.0 * st["budget_core_h"]), decisive=None, reported=[],
               overrun=None, overrun_reported=None, order=order, spent_s=fit_s, updated_at=_now())
    if sel is not None:
        out.update(share_core_h=sel["share_core_h"], reported_core_h=sel["reported_core_h"],
                   decisive=dict(cpu_s=sel["decisive"]["cpu_s"], episodes=sel["decisive"]["episodes"],
                                 mean_s_per_episode=sel["decisive"]["mean_s_per_episode"]),
                   reported=[{k: c.get(k) for k in ("k", "ms", "candidate", "status", "estimate_s", "share_left_s",
                                                    "started", "cpu_s")} for c in sel["by_delay"]],
                   overrun=sel["overrun"], overrun_reported=sel["overrun_reported"], spent_s=sel["spent_s"])
        order.append(dict(item="decisive cell", k=DECISIVE_K, cpu_s=sel["decisive"]["cpu_s"]))
        order.extend(dict(item=f"reported cell k={c['k']} {c['candidate']}", cpu_s=c.get("cpu_s") or 0.0,
                          started=c["started"]) for c in sel["by_delay"])
    return out


def fit(body: str, set_name: str, seed: int = SEED, out=None, workers: int = 1, resume: bool = False,
        budget_core_h: float = FIT_CORE_H, max_generations: int = GENERATIONS, log=print, *,
        popsize: int = POPSIZE) -> dict:
    """R2L's fit of ``body`` on ``set_name`` (module docstring; ``SEARCH_RULE``, ``ROTATION_RULE``,
    ``BUDGET_RULE``, ``RESUME_RULE``) into ``<out>/<set>/<body>/r2lfit/<seed>/``.  ``popsize`` other than 12 (a smoke
    check) makes a development fit (``ROLE_RULE``).  Returns ``cma.json``'s document."""
    out = _out(out)
    if not 1 <= int(max_generations) <= GENERATIONS:
        raise ValueError(f"max_generations {max_generations!r}: 1-{GENERATIONS} (rule Q3: at most {GENERATIONS})")
    if int(popsize) < 2:
        raise ValueError(f"popsize {popsize!r}: at least 2")
    sd, d = set_body_dir(out, set_name, body), fit_dir(out, set_name, body, seed)
    rot_path, path, elog, lpath = d / "rotation.json", d / "cma.json", d / "evals.jsonl", d / "ledger.json"
    gate_rec = gate(set_name, body, seed)
    start = speedfit.start_record(body)
    table = rotation(seed, GENERATIONS)
    settings = _settings(body, set_name, seed, start, budget_core_h, max_generations, popsize, table)
    src = sources()
    prov = _provenance(out)
    with s2io.Lock(sd / ".lock", log=log):
        s2io.sweep_tmp(sd)
        doc = None
        if path.exists() and not resume:
            raise R2LError(f"{path} exists: pass --resume to continue it (an existing fit is never restarted over)")
        if resume:
            if not path.exists():
                raise R2LError(f"{path}: nothing to resume")
            doc = s2io.read_json(path)
            if doc["settings"] != json.loads(json.dumps(s2io.plain(settings))):
                raise R2LError(f"{path}: its settings differ from this code's, start and arguments; refusing to resume")
            rot = panels._read(rot_path)
            if rot is None or s2io.sha256_json(rot.get("table")) != s2io.sha256_json(table["table"]):
                raise R2LError(f"{rot_path}: missing, or another table than rotation({seed}) (ROTATION_RULE): "
                               "refusing to resume")
            if doc.get("status") in FINISHED:
                log(f"{set_name}/{body} seed {seed}: the fit is {doc['status']} at generation {len(doc['history'])}: "
                    "nothing to resume, nothing written (RESUME_RULE)")
                return doc
            first = doc["sessions"][0]
            if (first.get("sources") or {}).get("sha256") != src["sha256"]:
                raise R2LError(f"{path}: R2L's sources changed since the fit started; a resumed fit would not continue "
                               "the same computation")
            if first["provenance"]["versions"] != prov["versions"]:
                raise R2LError(f"{path}: library versions changed since the fit started; refusing to resume")
        else:
            d.mkdir(parents=True, exist_ok=True)
            _write_rotation(rot_path, table)
            _write_start(sd, start, log)
        pool = _pool(body, set_name, workers, src["sha256"])
        try:
            session = dict(started_at=_now(), from_generation=0, workers=pool.workers, provenance=prov, sources=src,
                           inputs=pool.inputs, gate=gate_rec, incidents=[])
            if resume:
                diff = input_differences(doc["sessions"][0].get("inputs"), pool.inputs)
                if diff:
                    raise R2LError(f"{path}: its inputs changed since the fit started ({'; '.join(diff)}); refusing "
                                   "to resume")
                cma = tune.CMA.from_state(doc["state"], SIGMA0, seed)
                n_done = int(doc["evals"])
                lines = elog.read_text().splitlines() if elog.exists() else []
                if len(lines) < n_done:
                    raise R2LError(f"{elog}: {len(lines)} evaluations logged, the checkpoint has {n_done}")
                if len(lines) > n_done:  # evaluations after the last checkpoint: the resumed run repeats them
                    tmp = elog.with_name(elog.name + ".tmp")
                    tmp.write_text("".join(line + "\n" for line in lines[:n_done]))
                    os.replace(tmp, elog)
                    log(f"{set_name}/{body} seed {seed}: dropped {len(lines) - n_done} evaluations logged after the "
                        "checkpoint")
                session["from_generation"] = cma.gen
                doc["sessions"].append(session)
                doc["status"] = "running"
            else:
                cma = tune.CMA(tune.to_unit(speedfit.gains_vector(start["gains"])), SIGMA0, int(popsize), seed)
                doc = dict(schema=SCHEMA, kind="fit", body=body, set=set_name, seed=int(seed), settings=settings,
                           objective=OBJECTIVE, start=start, fixed_gains=tune.FIXED_RECORD, rules=RULES,
                           rotation=dict(path="rotation.json", table_sha256=settings["rotation_sha256"]),
                           sessions=[session], history=[], block_decisions=[], evals=0, status="running",
                           ended=None, state=cma.state())
                elog.write_text("")
                s2io.write_json(path, doc)
            doc["mix"] = pool.describe()
            s2io.write_json(lpath, ledger_doc(doc))
            seen = 0
            while True:
                g = cma.gen
                if g >= int(max_generations):
                    doc["ended"] = dict(reason=f"{g} generations (the cap)", at_generation=g)
                    break
                if g and g % BLOCK == 0:
                    dec = next_block(doc["history"], budget_core_h)
                    kept = [x for x in doc["block_decisions"] if x["at_generation"] == g]
                    if not kept:
                        doc["block_decisions"].append(dec)
                    elif kept[0]["start"] != dec["start"]:
                        raise R2LError(f"{path}: the block decision at generation {g} was {kept[0]['start']}, now "
                                       f"{dec['start']}: the ledger changed")
                    if not dec["start"]:
                        doc["ended"] = dict(reason="budget (rule Q5's block rule)", at_generation=g, decision=dec)
                        break
                row = table["table"][g]
                spec = spec_of(row)
                X = cma.ask()
                t0 = time.perf_counter()
                res = _map(pool, [(tune.from_unit(x), spec) for x in X])
                fitness = [float(r["J"]) + tune.repair_penalty(x) for x, r in zip(X, res, strict=True)]
                with elog.open("a") as fh:
                    for i, (x, f, r) in enumerate(zip(X, fitness, res, strict=True)):
                        fh.write(json.dumps(s2io.plain(_eval_line(g, i, x, f, r))) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                cma.tell(X, fitness)
                J = np.array([r["J"] for r in res])
                best = int(np.argmin(fitness))
                new = pool.incidents[seen:]
                seen = len(pool.incidents)
                doc["sessions"][-1]["incidents"].extend(new)
                falls = {k: int(sum(r["episodes"][k]["fell"] for r in res)) for k in KINDS}
                doc["history"].append(dict(
                    gen=g, block=g // BLOCK, episode=spec["seed"],
                    slots=[dict(condition=s["condition"], k=s["k"], direction=s["direction"]) for s in row["slots"]],
                    evals=doc["evals"] + cma.lam, sigma=cma.sigma, mean=cma.mean.tolist(),
                    mean_gains=tune.from_unit(cma.mean).tolist(), J_min=float(J.min()), J_median=float(np.median(J)),
                    J_max=float(J.max()), falls=int(sum(r["falls"] for r in res)), falls_by_kind=falls, best=best,
                    best_gains=tune.from_unit(X[best]).tolist(), wall_s=time.perf_counter() - t0,
                    cpu_s=float(sum(float(r.get("cpu_s") or 0.0) for r in res)),
                    sim_s=float(sum(float(r["simulated_s"]) for r in res)),
                    errors=int(sum(bool(r.get("error")) for r in res)), incidents=len(new)))
                doc["evals"] += cma.lam
                doc["state"] = cma.state()
                s2io.write_json(path, doc)
                s2io.write_json(lpath, ledger_doc(doc))
                h = doc["history"][-1]
                log(f"{set_name}/{body} seed {seed} gen {g} (block {g // BLOCK}): episode {spec['seed']}, J min/median "
                    f"{h['J_min']:.4f}/{h['J_median']:.4f}, falls {h['falls']} "
                    f"({', '.join(f'{k} {v}' for k, v in falls.items() if v) or 'none'}), sigma {cma.sigma:.4f}, "
                    f"{h['cpu_s']:.1f} CPU s ({h['cpu_s'] / cma.lam:.2f} per evaluation), {h['wall_s']:.1f} s wall")
        finally:
            pool.close()
        n = len(doc["history"])
        doc["status"] = "done" if n >= GENERATIONS or n >= int(max_generations) else "ended_by_budget"
        cov = coverage(table["table"], n)
        doc["rotation_met"] = dict(generations=n, coverage=cov,
                                   not_met={x: c["not_met"] for x, c in cov.items() if c["not_met"]})
        s = doc["sessions"][-1]
        s["ended_at"] = _now()
        s["sources_unchanged_during_session"] = sources()["sha256"] == src["sha256"]
        s2io.write_json(path, doc)
        s2io.write_json(lpath, ledger_doc(doc))
    return doc


# ------------------------------------------------------------------------------------------------
# The selection (rules Q4, Q5)
# ------------------------------------------------------------------------------------------------
class EvalLog:
    """``SELECTION_RULE``'s log of validation evaluations: ``get`` a kept one (None if there is none for this
    context), ``add`` a new one (appended, flushed and synced; one that failed to run is not kept).  A line cut short
    when the process died is dropped."""

    def __init__(self, path: Path, context: dict):
        self.path, self.context = Path(path), s2io.plain(context)
        self.context_sha256 = s2io.sha256_json(self.context)
        self.kept: dict[str, dict] = {}
        self.dropped = self.added = self.reused = 0
        good = []
        for line in self.path.read_text().splitlines() if self.path.exists() else ():
            try:
                x = json.loads(line)
                key, value = x["key"], x["result"]
            except (ValueError, KeyError, TypeError):
                self.dropped += 1
                continue
            good.append(line)
            self.kept[key] = value
        if self.dropped:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text("".join(x + "\n" for x in good))
            os.replace(tmp, self.path)

    def key(self, gains, spec: dict) -> str:
        return s2io.sha256_json(dict(context=self.context_sha256, gains=speedfit.gains_sha256(gains), spec=spec))

    def get(self, gains, spec: dict) -> dict | None:
        v = self.kept.get(self.key(gains, spec))
        self.reused += v is not None
        return v

    def add(self, gains, spec: dict, value: dict) -> dict:
        value = json.loads(json.dumps(s2io.plain(value)))
        if value.get("error"):
            return value
        key = self.key(gains, spec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(dict(key=key, context=self.context_sha256, gains_sha256=speedfit.gains_sha256(gains),
                                     spec=spec, result=value)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.kept[key] = value
        self.added += 1
        return value

    def describe(self) -> dict:
        return dict(file=s2io.portable(str(self.path.resolve())), context=self.context,
                    context_sha256=self.context_sha256, reused=self.reused, added=self.added,
                    dropped_lines=self.dropped)


def validation_record(r: dict) -> dict:
    """What the log keeps of a validation evaluation."""
    eps = {k: {x: e.get(x) for x in ("condition", "k", "fell", "fall_time_s", "fall_cause", "survived_frac",
                                      "succeeded", "failed_terms", "tracking", "limits", "readings", "mujoco_warnings",
                                      "seconds", "planned_seconds")} for k, e in r["episodes"].items()}
    out = dict(episode=r["episode_seed"], J=r["J"], S=r["S"], P=r["P"], T=r["T"], L=r["L"], groups=r["groups"],
               falls=r["falls"], terms=r["terms"], episodes=eps, wall_s=r["wall_s"], cpu_s=float(r.get("cpu_s") or 0.0),
               simulated_s=r["simulated_s"])
    if r.get("error"):
        out["error"] = r["error"]
    return out


def validation_summary(results: list[dict]) -> dict:
    """Falls, mean S, P and J (``tune.selection_key``'s), and per kind the runs, survived and succeeded (the bar's
    success, reading 2)."""
    kinds = {}
    for k in KINDS:
        eps = [r["episodes"][k] for r in results]
        kinds[k] = dict(runs=len(eps), survived=int(sum(not e["fell"] for e in eps)),
                        succeeded=int(sum(bool(e["succeeded"]) for e in eps)),
                        conditions=sorted({e["condition"] for e in eps}))
    return dict(evaluations=len(results), episodes=len(results) * len(KINDS),
                falls=int(sum(int(r["falls"]) for r in results)), S_mean=float(np.mean([r["S"] for r in results])),
                P_mean=float(np.mean([r["P"] for r in results])), J_mean=float(np.mean([r["J"] for r in results])),
                errors=int(sum(bool(r.get("error")) for r in results)), kinds=kinds)


def _run_logged(pool, glog: EvalLog, jobs: list, log, what: str) -> list[dict]:
    """``jobs`` ((gains, spec)) through the log: kept ones reused, the others run (in chunks) and kept as they
    finish."""
    vals = [glog.get(g, s) for g, s in jobs]
    todo = [i for i, v in enumerate(vals) if v is None]
    if todo:
        log(f"{what}: {len(jobs)} evaluations, {len(jobs) - len(todo)} kept from an earlier session")
    step = 1 if pool.workers == 1 else 2 * pool.workers
    for a in range(0, len(todo), step):
        idx = todo[a : a + step]
        for i, r in zip(idx, _map(pool, [jobs[i] for i in idx]), strict=True):
            vals[i] = glog.add(*jobs[i], validation_record(r))
    return vals


def _charge(glog: EvalLog, jobs: list, vals: list, counted: set) -> float:
    """The CPU seconds of the evaluations not counted yet (``BUDGET_RULE``: an evaluation is counted once)."""
    total = 0.0
    for (g, s), v in zip(jobs, vals, strict=True):
        key = glog.key(g, s)
        if key in counted:
            continue
        counted.add(key)
        total += float(v.get("cpu_s") or 0.0)
    return total


def selection_role(settings: dict, reported_core_h: float) -> dict:
    """``ROLE_RULE`` for a selection: ``{role: "fit" | "development", reasons, rule}`` from the fit's role and the
    reported cells' part (a share other than rule Q5's 4.0 makes a development selection)."""
    why = list((settings.get("role") or {}).get("reasons") or [])
    if (settings.get("role") or {}).get("role") != "fit" and not why:
        why.append("the fit is not one")
    if float(reported_core_h) != REPORTED_CORE_H:
        why.append(f"the reported cells' part {float(reported_core_h):g} core-hours, not {REPORTED_CORE_H:g} (a share "
                   f"of {share_core_h(settings['budget_core_h'], reported_core_h):g}, not {SHARE_CORE_H:g}: rule Q5)")
    return dict(role="development" if why else "fit", reasons=why, rule=ROLE_RULE)


def select(body: str, set_name: str, seed: int = SEED, out=None, workers: int = 1,
           reported_core_h: float = REPORTED_CORE_H, *, log=print) -> dict:
    """Rule Q4's selection with rule Q5's budget (``SELECTION_RULE``, ``BUDGET_RULE``) into
    ``<out>/<set>/<body>/r2lg5.json``; the validation evaluations are kept in ``r2lg5_evals.jsonl`` and reused by a
    rerun; the ledger gets the decisive and the reported cells.  On a look set, or a version of one, it starts only
    once G9-ref has passed (``GATE_RULE``, ``gate``; r2lg5.json's ``g9_ref``)."""
    out = _out(out)
    sd, d = set_body_dir(out, set_name, body), fit_dir(out, set_name, body, seed)
    path, lpath = d / "cma.json", d / "ledger.json"
    gate_rec = gate(set_name, body, seed)
    src = sources()
    with s2io.Lock(sd / ".lock", log=log):
        s2io.sweep_tmp(sd)
        doc = panels._read(path)
        if doc is None:
            raise R2LError(f"{path}: no fit to select from (run fit first)")
        st = doc["settings"]
        if st["role"]["role"] != "fit":
            raise R2LError(f"{path}: a development fit ({'; '.join(st['role']['reasons'])}) is not judged (ROLE_RULE)")
        if doc["status"] not in FINISHED:
            raise R2LError(f"{path}: the fit is {doc['status']} at generation {len(doc['history'])}: resume it first")
        fit_src = [(s.get("sources") or {}).get("sha256") for s in doc["sessions"]]
        if any(x != src["sha256"] for x in fit_src):
            raise R2LError(f"{path}: the fit ran on other sources than this selection's: refusing to judge it")
        role = selection_role(st, reported_core_h)
        n_gen = len(doc["history"])
        gens = tune.candidate_generations(n_gen)
        start = doc["start"]
        cands = {"start": dict(gains=speedfit.gains_vector(start["gains"]).tolist(), generation=0,
                               source=f"the fit's start: {start['source']}")}
        for g in gens:
            h = doc["history"][g - 1]
            cands[f"gen{g}"] = dict(gains=list(h["mean_gains"]), generation=g,
                                    source=f"the CMA-ES mean after generation {g} ({h['evals']} evaluations)")
        names = list(cands)
        share_s = 3600.0 * share_core_h(st["budget_core_h"], reported_core_h)
        fit_s = float(sum(b["cpu_s"] for b in block_costs(doc["history"])))
        pool = _pool(body, set_name, workers, src["sha256"])
        try:
            for s in doc["sessions"]:
                diff = input_differences(s.get("inputs"), pool.inputs)
                if diff:
                    raise R2LError(f"{path}: the fit ran on other inputs ({'; '.join(diff)}): refusing to judge it")
            prov = _provenance(out)
            ctx = dict(body=body, set=set_name, mode=MODE, rate=RATE, objective=OBJECTIVE_NAME, sources=src,
                       inputs=s2io.sha256_json(dict(files=pool.inputs.get("files"), freeze=pool.inputs.get("freeze"))),
                       versions=prov["versions"])
            glog = EvalLog(sd / EVAL_LOG, ctx)
            counted: set = set()
            # -- the decisive cell: first and whole
            jobs = [(np.asarray(cands[n]["gains"], float), validation_spec(s, DECISIVE_K)) for n in names
                    for s in VALIDATION_SEEDS]
            vals = _run_logged(pool, glog, jobs, log, f"{set_name}/{body}: decisive cell (k = {DECISIVE_K})")
            dec_s = _charge(glog, jobs, vals, counted)
            m = len(VALIDATION_SEEDS)
            by = {n: vals[i * m : (i + 1) * m] for i, n in enumerate(names)}
            for n in names:
                cands[n]["gains_named"] = control.gains_dict(cands[n]["gains"])
                cands[n]["decisive"] = validation_summary(by[n])
            ranked = sorted(names, key=lambda n: tune.selection_key(cands[n]["decisive"], names.index(n)))
            sel = ranked[0]
            spent = fit_s + dec_s
            n_ep = len(names) * m * len(KINDS)
            mean_ep = dec_s / n_ep if n_ep else 0.0
            estimate = mean_ep * len(REPORTED_SEEDS) * len(KINDS) * REPORTED_MARGIN
            reserve = 3600.0 * DECISIVE_CORE_H
            overrun = dict(decisive_cpu_s=dec_s, reserve_s=reserve, over_reserve_s=max(0.0, dec_s - reserve),
                           share_s=share_s, spent_after_decisive_s=spent,
                           over_share_s=max(0.0, spent - share_s))
            # -- the reported cells, while the share holds each one's estimate
            cells, stopped, overrun_reported = [], False, None
            for k in REPORTED_K:
                for cname in (sel, "start"):
                    left = share_s - spent
                    cell = dict(k=k, ms=2 * k, candidate=cname, seeds=list(REPORTED_SEEDS), estimate_s=estimate,
                                share_left_s=left)
                    if stopped or left < estimate:
                        cell.update(status="not run: budget", started=False, cpu_s=None,
                                    reason="an earlier reported cell passed the share" if stopped else
                                    "the share no longer holds the cell's estimate")
                        cells.append(cell)
                        continue
                    cj = [(np.asarray(cands[cname]["gains"], float), validation_spec(s, k)) for s in REPORTED_SEEDS]
                    cv = _run_logged(pool, glog, cj, log, f"{set_name}/{body}: reported cell k = {k} ({cname})")
                    c_s = _charge(glog, cj, cv, counted)
                    spent += c_s
                    cell.update(status="run", started=True, cpu_s=c_s, summary=validation_summary(cv),
                                runs=cv)
                    if spent > share_s:
                        overrun_reported = dict(after_cell=f"k={k} {cname}", over_share_s=spent - share_s)
                        cell["overrun_reported_s"] = spent - share_s
                        stopped = True
                    cells.append(cell)
            incidents, inputs, kept = list(pool.incidents), pool.inputs, glog.describe()
        finally:
            pool.close()
        rot = panels._read(d / "rotation.json") or {}
        res = dict(
            schema=SCHEMA, kind="selection", gate="R2L G5 (rule Q4)", body=body, set=set_name, seed=int(seed),
            role=role, g9_ref=gate_rec, rules=RULES, objective=OBJECTIVE, judged=True, judged_at_generation=n_gen,
            fit_status=doc["status"],
            fit_ended=doc.get("ended"), rotation_met=doc.get("rotation_met"),
            fit_record=dict(path=s2io.portable(str(path.resolve())), sha256=s2io.sha256_file(path)),
            rotation_record=dict(path=s2io.portable(str((d / "rotation.json").resolve())),
                                 table_sha256=rot.get("table_sha256")),
            start=start, candidates={n: {k: v for k, v in c.items()} for n, c in cands.items()},
            selection=[dict(candidate=n, rank=ranked.index(n) + 1,
                            **{k: cands[n]["decisive"][k] for k in ("falls", "S_mean", "P_mean", "J_mean")})
                       for n in names],
            selected=sel, selected_gains=cands[sel]["gains_named"], gains=cands[sel]["gains_named"],
            decisive=dict(k=DECISIVE_K, ms=2 * DECISIVE_K, seeds=list(VALIDATION_SEEDS), candidates=names,
                          cpu_s=dec_s, episodes=n_ep, mean_s_per_episode=mean_ep,
                          runs={n: by[n] for n in names}),
            by_delay=cells, share_core_h=share_s / 3600.0, reported_core_h=float(reported_core_h),
            fit_cpu_s=fit_s, spent_s=spent, overrun=overrun, overrun_reported=overrun_reported,
            evaluations_kept=kept, sources=src, inputs=inputs, incidents=incidents, provenance=prov,
            written_at=_now())
        s2io.write_json(sd / "r2lg5.json", res)
        s2io.write_json(lpath, ledger_doc(doc, res))
    return res


def selection_text(doc: dict) -> str:
    role = doc.get("role") or {}
    lines = [f"R2L selection {doc['set']}/{doc['body']} seed {doc['seed']} ({role.get('role')}"
             + (f": {'; '.join(role['reasons'])}" if role.get("reasons") else "") + "): judged at generation "
             f"{doc['judged_at_generation']} ({doc['fit_status']}); selected {doc['selected']}"]
    for x in doc["selection"]:
        lines.append(f"   {x['rank']}. {x['candidate']}: falls {x['falls']}, mean S {x['S_mean']:.4f}, mean P "
                     f"{x['P_mean']:.4f}, mean J {x['J_mean']:.4f}")
    dec = doc["decisive"]
    lines.append(f"   decisive cell: {dec['cpu_s']:.0f} CPU s, {dec['mean_s_per_episode']:.2f} s per episode"
                 + (f"; overrun {doc['overrun']['over_reserve_s']:.0f} s over its reserve"
                    if doc["overrun"]["over_reserve_s"] else ""))
    for c in doc["by_delay"]:
        what = (f"{c['summary']['falls']} falls" if c["started"] else c["status"])
        lines.append(f"   k={c['k']} ({c['ms']} ms) {c['candidate']}: {what} (estimate {c['estimate_s']:.0f} s, "
                     f"share left {c['share_left_s']:.0f} s)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# Publishing and the index
# ------------------------------------------------------------------------------------------------
def read_index(index=None) -> dict:
    """The index (``{"schema": "s2-r2l-index-1", "fits": [...]}``); an empty one where there is no file."""
    p = _anchored(INDEX if index is None else index)
    if not p.exists():
        return dict(schema=INDEX_SCHEMA, fits=[])
    doc = s2io.read_json(p)
    if not isinstance(doc, dict) or doc.get("schema") != INDEX_SCHEMA or not isinstance(doc.get("fits"), list):
        raise R2LError(f"{p}: not an R2L index (schema {INDEX_SCHEMA}, a list of fits)")
    return doc


def index_entry(set_name: str, body: str, seed: int, index=None) -> dict:
    """The index's entry of a fit; ``R2LError`` where there is none (or two)."""
    hits = [e for e in read_index(index)["fits"]
            if e.get("set") == set_name and e.get("body") == body and int(e.get("seed", -1)) == int(seed)]
    if len(hits) != 1:
        raise R2LError(f"{_anchored(INDEX if index is None else index)}: {len(hits)} entries for the fit of {body} on "
                       f"{set_name}, seed {seed} (one is needed)")
    return hits[0]


#: What ``publish`` copies from ``<out>/<set>/<body>/`` besides the fit's own ``r2lfit/<seed>/`` (``PUBLISH_RULE``).
PUBLISHED_FILES = ("start.json", "r2lg5.json", EVAL_LOG)
#: The files a published fit must hold (``r2lfit/<seed>/`` ones relative to that directory).
REQUIRED_FILES = ("start.json", "r2lg5.json")
REQUIRED_FIT_FILES = ("rotation.json", "cma.json", "evals.jsonl", "ledger.json")


def publish_files(src, seed: int) -> list[str]:
    """``PUBLISH_RULE``: the records ``publish`` copies from ``src`` (``<out>/<set>/<body>/``), of what
    ``nb.collect_files`` finds there: ``start.json``, ``r2lg5.json``, ``r2lg5_evals.jsonl`` and the fit's own
    ``r2lfit/<seed>/``; nothing else (another seed's fit, a staging copy)."""
    from designlab.stage2 import nb

    own = f"r2lfit/{int(seed)}/"
    return [r for r in nb.collect_files(src) if r in PUBLISHED_FILES or r.startswith(own)]


def publish(body: str, set_name: str, seed: int = SEED, out=None, index=INDEX) -> dict:
    """Copy the finished records of a fit and its selection (``publish_files``: ``PUBLISH_RULE``) to ``<index's
    directory>/<set>/<body>/`` (staged, then ``nb.collect``: each copy verified by sha256, ``COLLECTED.json``; a
    published file is never replaced) and add the entry ``{set, body, seed, r2lg5: {path, sha256}}`` to the index,
    holding the set and body's lock.  Returns the entry."""
    import shutil

    from designlab.stage2 import nb

    out = _out(out)
    idx_path = _anchored(index)
    src = set_body_dir(out, set_name, body)
    fd = fit_dir(out, set_name, body, seed)
    if not src.is_dir():
        raise R2LError(f"{src}: no such directory (run fit and select first)")
    with s2io.Lock(src / ".lock"):
        sel = panels._read(src / "r2lg5.json")
        doc = panels._read(fd / "cma.json")
        if sel is None or doc is None:
            raise R2LError(f"{src}: no r2lg5.json or no cma.json (run fit and select first)")
        if (sel.get("set"), sel.get("body"), int(sel.get("seed", -1))) != (set_name, body, int(seed)):
            raise R2LError(f"{src / 'r2lg5.json'}: the selection of {sel.get('body')} on {sel.get('set')} seed "
                           f"{sel.get('seed')}, not {body} on {set_name} seed {seed}")
        if doc["settings"]["role"]["role"] != "fit" or not sel.get("judged"):
            raise R2LError(f"{src}: a development fit is not published (ROLE_RULE)")
        role = sel.get("role") or {}
        if role.get("role") != "fit":
            raise R2LError(f"{src / 'r2lg5.json'}: a development selection is not published (ROLE_RULE: "
                           f"{'; '.join(role.get('reasons') or ['it names no role'])})")
        if s2io.sha256_file(fd / "cma.json") != sel["fit_record"]["sha256"]:
            raise R2LError(f"{src}: cma.json changed after the selection was made: select again")
        rels = publish_files(src, seed)
        missing = [r for r in REQUIRED_FILES + tuple(f"r2lfit/{int(seed)}/{x}" for x in REQUIRED_FIT_FILES)
                   if r not in rels]
        if missing:
            raise R2LError(f"{src}: the fit's records are not all there ({', '.join(missing)} missing): nothing was "
                           "published")
        idx = read_index(idx_path)
        dst = idx_path.parent / set_name / body
        sha = s2io.sha256_file(src / "r2lg5.json")
        entry = dict(set=set_name, body=body, seed=int(seed),
                     r2lg5=dict(path=s2io.portable(str((dst / "r2lg5.json").resolve())), sha256=sha))
        same = [e for e in idx["fits"] if (e.get("set"), e.get("body"), int(e.get("seed", -1))) == (set_name, body,
                                                                                                    int(seed))]
        if same and s2io.plain(same[0]) != s2io.plain(entry):
            raise R2LError(f"{idx_path}: the fit of {body} on {set_name} seed {seed} is indexed already with another "
                           "gains file: an indexed fit is never replaced")
        if same and all((dst / r).is_file() and s2io.sha256_file(dst / r) == s2io.sha256_file(src / r) for r in rels):
            return entry  # published already, every file the same: nothing is written again
        stage = src / ".publish" / str(int(seed))
        try:
            if stage.exists():
                shutil.rmtree(stage)
            for r in rels:
                (stage / r).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / r, stage / r)
                if s2io.sha256_file(stage / r) != s2io.sha256_file(src / r):
                    raise R2LError(f"{stage / r}: the staged copy's sha256 is not the source's")
            try:
                nb.collect(stage, dst, "r2l")
            except nb.NbError as exc:
                raise R2LError(f"publish {set_name}/{body}: {exc}") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            try:
                stage.parent.rmdir()
            except OSError:
                pass
        if s2io.sha256_file(dst / "r2lg5.json") != sha:
            raise R2LError(f"{dst / 'r2lg5.json'}: the copy's sha256 is not the source's")
        if not same:
            idx["fits"].append(entry)
            s2io.write_json(idx_path, idx)
    return entry


# ------------------------------------------------------------------------------------------------
# The rows (rule Q6)
# ------------------------------------------------------------------------------------------------
_ROW_NAME = re.compile(r"^R2L-(?P<set>[A-Za-z0-9_.\-]+)/(?P<seed>\d+)$")


def parse_row_name(name: str) -> tuple[str, int]:
    """(set, seed) of ``R2L-<set>/<seed>``; ``ValueError`` for another name."""
    m = _ROW_NAME.match(name if isinstance(name, str) else "")
    if not m:
        raise ValueError(f"row {name!r}: R2L-<set>/<seed> (R2L-V6noF5/42)")
    return m.group("set"), int(m.group("seed"))


def gains_of(path, sha256: str, set_name: str, body: str, seed: int) -> tuple[np.ndarray, dict]:
    """The selected gains of an ``r2lg5.json`` whose sha256 is ``sha256`` (``ROWS_RULE``), with their info."""
    p = _anchored(path)
    if not p.is_file():
        raise R2LError(f"{p}: the indexed gains file is missing")
    got = s2io.sha256_file(p)
    if got != sha256:
        raise R2LError(f"{p}: sha256 {got[:12]}..., the index holds {str(sha256)[:12]}...: the gains file changed")
    doc = s2io.read_json(p)
    if (doc.get("set"), doc.get("body"), int(doc.get("seed", -1))) != (set_name, body, int(seed)):
        raise R2LError(f"{p}: the selection of {doc.get('body')} on {doc.get('set')} seed {doc.get('seed')}, not "
                       f"{body} on {set_name} seed {seed}")
    if not doc.get("judged"):
        raise R2LError(f"{p}: not a judged selection")
    if (doc.get("role") or {}).get("role") != "fit":
        raise R2LError(f"{p}: a development selection is no row's gains (ROLE_RULE)")
    gains = speedfit.gains_vector(doc["selected_gains"])
    info = dict(rule=ROWS_RULE, file=s2io.portable(str(p.resolve())), sha256=got, set=set_name, body=body,
                seed=int(seed), candidate=doc.get("selected"), judged_at_generation=doc.get("judged_at_generation"),
                source=f"R2L's selection of {body} on {set_name}, fit seed {seed}")
    return gains, info


class R2LRow(panels.TrackerRow):
    """``R2L-<set>/<seed>`` (``ROWS_RULE``): ``panels.TrackerRow(name, "R2", gains, gains_info)`` with the indexed
    gains; its episode refuses a condition whose set is not the fit's."""

    def __init__(self, name: str, fit_set: str, gains, gains_info: dict):
        super().__init__(name, MODE, gains, gains_info,
                         controller=f"R2L: the tracker on the estimate refitted on the bar's tasks of set {fit_set} "
                                    f"(r2lfit; rule Q6), {gains_info.get('candidate')} of {gains_info.get('file')}")
        self.fit_set = str(fit_set)

    def episode(self, ctx, cond, seed: int, relax=()):
        set_name = getattr(cond, "set", None) or bar.STOCK_SET
        if set_name != self.fit_set:
            raise R2LError(f"{self.name}: the condition {cond.key} is of set {set_name}, the row's fit of "
                           f"{self.fit_set} (rule Q6: a row refuses a condition of another set)")
        return super().episode(ctx, cond, seed, relax)

    def describe(self) -> dict:
        return dict(super().describe(), fit_set=self.fit_set, rows_rule=ROWS_RULE,
                    gains_file=dict(path=self.gains_info["file"], sha256=self.gains_info["sha256"]))

    def sources(self) -> dict:
        """This module by sha256 (``lookbar.LookRow`` adds lookbar.py and lookread.py; ``bar.Delayed`` the bar's)."""
        return panels.module_sha256(R2LRow)

    def spec(self) -> dict:
        return dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name, fit_set=self.fit_set,
                    gains=self.gains.tolist(), gains_info=self.gains_info)

    @classmethod
    def from_spec(cls, s: dict) -> "R2LRow":
        info = s["gains_info"]
        _, seed = parse_row_name(s["name"])
        gains, _ = gains_of(info["file"], info["sha256"], s["fit_set"], info["body"], seed)
        if not np.array_equal(gains, np.asarray(s["gains"], float)):
            raise R2LError(f"{info['file']}: its gains are not the ones the row {s['name']!r} was built with")
        return cls(s["name"], s["fit_set"], gains, info)


def r2l_row(body: str, run_dir, name: str) -> panels.Row:
    """``panels.register_rows``' factory: ``R2L-<set>/<seed>`` of ``body`` (``ROWS_RULE``) inside
    ``lookbar.LookRow``; ``run_dir`` (the panels' Stage 1 run directory) is not read."""
    from designlab.stage2 import lookbar

    set_name, seed = parse_row_name(name)
    e = index_entry(set_name, body, seed)
    gains, info = gains_of(e["r2lg5"]["path"], e["r2lg5"]["sha256"], set_name, body, seed)
    return lookbar.LookRow(R2LRow(name, set_name, gains, info))


def register(index=None) -> list[str]:
    """``panels.register_rows(f"R2L-{set}", r2l_row)`` for every set of the index; the controllers registered."""
    out = []
    for e in read_index(index)["fits"]:
        ctrl = f"{CONTROLLER}-{e['set']}"
        if ctrl not in out:
            panels.register_rows(ctrl, r2l_row)
            out.append(ctrl)
    return out


REGISTERED = register()


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.r2lfit", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("rotation", "fit", "select", "publish"))
    ap.add_argument("--body", required=True)
    ap.add_argument("--set", dest="set_name", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(WORK))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--budget-core-h", type=float, default=FIT_CORE_H)
    ap.add_argument("--max-generations", type=int, default=GENERATIONS)
    ap.add_argument("--popsize", type=int, default=POPSIZE, help="other than 12: a development fit (a smoke check)")
    ap.add_argument("--reported-core-h", type=float, default=REPORTED_CORE_H)
    ap.add_argument("--index", default=str(INDEX))
    a = ap.parse_args(argv)
    try:
        if a.command == "rotation":
            doc = rotation(a.seed)
            print(rotation_text(doc))
            rec = panels._read(fit_dir(_out(a.out), a.set_name, a.body, a.seed) / "rotation.json")
            if rec is not None:
                same = s2io.sha256_json(rec.get("table")) == s2io.sha256_json(doc["table"])
                print(f"rotation.json ({rec.get('written_at')}): {'equals' if same else 'DIFFERS FROM'} this table")
            return 0
        s2io.require_one_thread()
        if a.command == "fit":
            doc = fit(a.body, a.set_name, a.seed, a.out, a.workers, a.resume, a.budget_core_h, a.max_generations,
                      popsize=a.popsize)
            n = len(doc["history"])
            cpu = sum(h["cpu_s"] for h in doc["history"])
            evals = sum(int(doc["settings"]["cma"]["popsize"]) for _ in doc["history"])
            print(f"{a.set_name}/{a.body} seed {a.seed}: {doc['status']} at generation {n} "
                  f"({doc['settings']['role']['role']}); {evals} evaluations, {cpu:.1f} CPU s"
                  + (f" ({cpu / evals:.2f} per evaluation, {cpu / evals / len(KINDS):.2f} per episode)"
                     if evals else ""))
        elif a.command == "select":
            print(selection_text(select(a.body, a.set_name, a.seed, a.out, a.workers, a.reported_core_h)))
        else:
            e = publish(a.body, a.set_name, a.seed, a.out, a.index)
            print(f"published {e['set']}/{e['body']} seed {e['seed']}: {e['r2lg5']['path']} ({e['r2lg5']['sha256']})")
        return 0
    except (R2LError, s2io.S2Error, tune.InputError, tune.PoolError, ValueError) as exc:
        print(f"r2lfit: {exc}", file=sys.stderr)
        return 1


__all__ = ["BUDGET_RULE", "CONDITIONS", "Evaluator", "EvalLog", "FITS", "INDEX", "KINDS", "LIMITS", "LOOK_SETS",
           "OBJECTIVE", "PUBLISH_RULE", "R2LError", "R2LRow", "ROTATION_RULE", "RULES", "SELECTION_RULE", "TRACKING",
           "blank_episode", "coverage", "episode_terms", "failed_evaluation", "fit", "gate", "index_entry", "look_rule",
           "look_set_of", "main", "publish", "publish_files", "r2l_row", "read_index", "register", "role_of", "rotation",
           "row_name", "run_slot", "scored", "select", "selection_role", "sources", "spec_of", "validation_spec"]

if __name__ == "__main__":
    from designlab.stage2 import r2lfit as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
