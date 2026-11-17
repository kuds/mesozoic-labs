"""Track B's monitor: the bar's own episodes on a saved checkpoint, the gain rule, G8b with its one fix, the stop
rule and S2-10's record (round C design D15, section 3.6; Track B rules M1-M6, M3b, K5; the lead's rulings of 16 Nov
2026 on a check's attempts).

**Episodes** (rule M1).  18 episodes: the bar's conditions ``CONDITIONS`` of the run's set at mu 0.9, three each,
every episode on its own seed (``SEEDS``: 6001-6018), at 10 ms (``K`` = 5 physics steps), under the checkpoint's row
``R5/<seed>@10ms`` (or ``R6/...``), and once per run under ``R0L@10ms`` (``baseline``).  They are run by
``lookbar.episodes`` (work package CW) in this process: ``_episodes`` is the one place this module touches
``lookbar`` and ``s2rows``, through imports made inside it.  No episode of a policy row is ever run twice.  A tiny
run (``tiny``) runs each condition's first seed only; on the Stage 1 view (set ``S1``: development, tiny runs only)
the conditions are ``S1_CONDITIONS``.

Each episode has one outcome: *read*; *fell*; *unread* (a policy row's episode that did not fall and that the bar
could not read: not succeeding in any reading; reason ``clock_behind`` or ``unreadable``); *error* (an
infrastructure fault: not succeeding, zero credited survival, not up).  A condition without a reference is
``not_run``: not succeeding and zero survival for both rows.

**Records.**  ``run`` writes ``monitor/monitor_<n>.json`` and ``baseline`` ``monitor/r0l.json``: per episode the
companion episode as ``lookbar.episodes`` returned it with its ``survival_credit``, then N (reading 1, with the
look), S (the mean credited survival), the counts in readings 2 and 3 and "but A1 with the look",
``failures_by_term``, ``mean_clock_factor``, ``distinct_start_seeds``, the checkpoint's three file hashes, and the
sources hash the trainer gave beside this process's own (``run`` refuses to run if they differ).  The episodes'
traces go under ``monitor/traces/<n>/`` (the baseline's under ``monitor/traces/r0l/``), at the bar's own relative
paths (``<row>/<behaviour>/<condition>/<seed>.npz``): ``clip2`` draws a clip from one of them.

**Refusals** (rule M1 with the lead's ruling of 16 Nov 2026, 19:01 UTC: a monitor that refuses to run is no
attempt; the sixth list of 16 Nov, 3, and the rulings of 17 Nov, 10 and 11).  ``run`` and ``baseline`` raise
``MonitorRefusal`` before any episode when the sources are not the trainer's (kind ``sources``), when the run's
``run.json`` is missing, cannot be read or is no JSON object (``run``), when the checkpoint is not intact
(``checkpoint``: no record, a record that cannot be read or has the wrong shape, a file that is not the one its
record names), when the checkpoint is another run's (``other_run``) and when the row refuses the checkpoint while it
is built, before any episode (``ROW_REFUSALS``: ``s2rows.RowError``, kind ``row``; ``s2load.CheckpointError``, kind
``checkpoint``, among them a fingerprint not reproduced at the row's load).  The CLI then leaves with its own exit
code, ``EXIT_REFUSAL`` (8), and writes the reason beside where the record would be (``refusal_path``:
``monitor_<n>.refused.json``, the baseline's ``r0l.refused.json``); a later call of the same check takes that note
away before it runs.  A row that refuses only after the check's episodes (the further delays of rule M6) is no
refusal: the call ends with exit code 1.

**Rules on records** (pure functions: the same state from the same records, live or at a resume):
``survival_credit``, ``gain`` (M3), ``g8b`` and ``fix_weights`` (M4), ``decide`` (everything read after a check:
the gain of every check, G8b at its two named checks with the one fix, M5's stop), ``s2_10`` (S2-10's one record,
with M3b's route), ``no_record`` (a check counted without a record after two closed attempts: its episodes are
errors), ``is_record`` (what "a record" of a check is).

**Every set goes through ``lookbar``** (the lead's ruling 17 of 17 Nov 2026: the Stage 1 view's stand-in is
deleted).  ``lookbar`` and ``s2rows`` are merged, so ``_episodes`` builds the row and runs the episodes through them
on every set, the Stage 1 view's included; a checkout that cannot import ``lookbar`` runs no monitor (the import
error is raised).  The record names its episodes' maker (``episodes_by``: ``EPISODES_BY``).

**R0L's baseline without a job** (ruling 17).  ``baseline`` takes a job's directory only to name the run in its
record and to refuse a ``run.json`` that cannot be read; without one (``--out`` then names the directory) it makes the
baseline that rule M3 commits as ``R0L_BASELINE.json`` before S2-4 starts (the gates' chain, step V8), and the
trainer compares S2-4's own baseline with that file (``train2 run --expect-baseline``).

    python -m designlab.stage2.monitor2 run --job DIR --steps N --body B --set S [--k 5] [--tiny] [--also-k 0,..]
                                            [--sources SHA256] [--out DIR]
    python -m designlab.stage2.monitor2 baseline [--job DIR] --body B --set S [--k 5] [--tiny] [--also-k 0,..]
                                            [--sources SHA256] [--out DIR]     (--out is required without --job)
    python -m designlab.stage2.monitor2 s2-10 --job DIR [--committed-baseline PATH] [--out PATH]

(also reachable as ``python -m designlab.stage2.train2 monitor ...``).  Exit code 0: the record was written (for
``baseline``: and holds no error); 1: no usable record, with the reason on stderr; 2: the arguments are wrong (no
``--out`` for a baseline without a job, among others); 8: the monitor refused to run (**Refusals**): no attempt was
made.
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("disable")

import os  # noqa: E402
import sys  # noqa: E402

os.environ.setdefault("GIT_OPTIONAL_LOCKS", "0")  # (as in train2: a killed ``git status`` leaves no index.lock)

if __name__ == "__main__":  # before any numpy import (round A's module skeleton)
    for _v in tbenv.THREAD_VARS:
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from designlab.stage2 import s2io  # noqa: E402

SCHEMA = "s2-monitor-1"
BASELINE_SCHEMA = "s2-monitor-r0l-1"
S2_10_SCHEMA = "s2-10-1"
REFUSAL_SCHEMA = "s2-monitor-refusal-1"
#: The CLI's exit code when the monitor refuses to run (``RULES["refusal"]``): no other ending of this process, of
#: the trainer or of a pool worker has it (0, 1, 2, 3, 4, 5, 130).
EXIT_REFUSAL = 8
#: What a refusal can be (``MonitorRefusal.kind``).
REFUSALS = ("sources", "checkpoint", "other_run", "run", "row")
#: How a row refuses a checkpoint while it is built (``s2rows.configure``, ``lookbar.rows``), before any episode:
#: (module, class name, refusal kind).  The class is looked up in ``sys.modules`` and never imported for it: an
#: exception of that class can only have been raised once its module was loaded (``s2rows`` is merged; the row
#: imports it while it is built).
#: An exception of another class, of the same name or not, is no refusal (``_row_refusal``).
ROW_REFUSALS = (("designlab.stage2.s2rows", "RowError", "row"),
                ("designlab.stage2.s2load", "CheckpointError", "checkpoint"))
#: What makes a record's episodes (``episodes_by``): every set goes through ``lookbar`` (the lead's ruling 17 of
#: 17 Nov 2026: the Stage 1 view's stand-in is deleted).
EPISODES_BY = "lookbar.episodes"

#: Rule M1: the bar's conditions of the monitor, in the order that gives each its seeds.
CONDITIONS = ("walk/nominal", "walk/v0.4", "turn/q90_left", "stop/two_step", "nudge/forward", "nudge/left")
#: Rule M1: condition i, repetition r: seed 6001 + 3 i + r (6001-6018; one episode each).
SEEDS = {c: (6001 + 3 * i, 6002 + 3 * i, 6003 + 3 * i) for i, c in enumerate(CONDITIONS)}
#: Stock conditions for tiny runs on the Stage 1 view (development only).
S1_CONDITIONS = ("walk/nominal", "walk/brisk")
S1_SET = "S1"
#: The monitor's servo delay in physics steps (10 ms) and the milliseconds of one step (``bar.STEP_MS``).
K = 5
STEP_MS = 2
#: The commanded clock (``tune.COMMAND_RATE``): a row that does not slow its clock covers 1.05 s of reference time
#: per simulated second.
RATE = 1.05
#: Rule M3's credit (the lead's ruling 17 of 17 Nov 2026): a reference clock that covered at least 1.05 x the
#: survival time, to within this share of it, covered it, and the credit is the survival time itself.  The share
#: is float rounding only (the reference time is a sum of one increment per physics step: ~1e-13 of it over an
#: episode); a clock that falls behind by one 2 ms physics step in 20 s is 1e-4 behind.
CREDIT_RTOL = 1e-9
OUTCOMES = ("read", "fell", "unread", "error")
NOT_RUN = "not_run"
#: Rule M4: "up" is an episode that did not fall and is no error.
UP = ("read", "unread")
#: Rule M4's table: the training tasks whose weight is doubled when a monitor condition has fewer than 2 of 3 up.
FIX_TASKS = {"walk/nominal": ("walk_nominal",), "walk/v0.4": ("walk_v0.4",), "nudge/forward": ("walk_v0.4",),
             "nudge/left": ("walk_v0.4",), "turn/q90_left": ("turn_q90_left", "turn_q90_right"),
             "stop/two_step": ("stop", "start_stop")}
#: The same on the Stage 1 view's task table (``r5lib.S1_TASKS``).
S1_FIX_TASKS = {"walk/nominal": ("walk",), "walk/brisk": ("brisk",)}
G8B_CONDITION = "walk/nominal"
#: Rule M3: a gain is at least this many more episodes with the look, or this much more credited survival (s).
GAIN_N, GAIN_S = 2, 1.0
#: Rule M4: G8b is read at the checks of these multiples; rule M5: the stop rule counts from this one.
G8B_AT, G8B_AGAIN_AT = 5_000_000, 10_000_000
STOP_FROM, STOP_CHECKS = 20_000_000, 3
#: Rule M1: closed attempts at a check that left no record, after which the check is counted without one; the
#: lead's ruling of 16 Nov: attempts that were never closed, after which the run halts.
ATTEMPTS = 2
#: Rule M3b: the survival route is closed when the mean planned length less R0L's S is under this (s).
ROUTE_CLOSED_S = 1.0

RULES = dict(
    episodes=(
        "rule M1: 18 episodes: the bar's conditions walk/nominal, walk/v0.4, turn/q90_left, stop/two_step, "
        "nudge/forward, nudge/left of the run's set at mu 0.9, three each, condition i (0-5), repetition r (0-2) on "
        "seed 6001 + 3 i + r, at 10 ms, as the bar builds and judges them; once per run the same under R0L@10ms.  A "
        "tiny run takes each condition's first seed; on the Stage 1 view (set S1, development) the conditions are "
        "the stock walk/nominal and walk/brisk"),
    outcomes=(
        "rule M1: each episode ends read, fell, unread (a policy row's episode that did not fall and that the bar "
        "could not read: not succeeding in any reading; reason clock_behind when its mean clock factor is below 1, "
        "else unreadable) or error (an infrastructure fault).  A policy row's episode is never run twice: an error "
        "there counts as not succeeding, as zero credited survival and as not up.  A condition without a reference "
        "is not_run: not succeeding and zero survival for both rows.  R0L's episodes must each end read or fell: a "
        "baseline that holds an error is kept beside as a failed attempt and no check is read against it"),
    survival=(
        "rule M3: an episode's credited survival is its survival time (the fall time, or the episode's length) "
        "times min(1, reference time covered / (1.05 x survival time)): the survival time itself for a row that "
        "does not slow its clock.  A clock that covered at least 1.05 x the survival time, to within 1e-9 of it "
        "(float rounding), is credited the survival time itself, so that R0L's S is exactly its plain mean "
        "survival (the lead's ruling 17 of 17 Nov 2026).  S is the mean over the episodes; N the number succeeding "
        "with the look (reading 1).  Printed beside: the counts in readings 2 and 3, the count succeeding but for "
        "A1 with the look bands kept, and the plain mean survival, which decide nothing.  The credit needs the "
        "episode's clock times "
        "(clock.t_ref_reached, clock.t_ref_planned) and its planned length: a policy row's episode that ended read, "
        "fell or unread and came back without them is not credited its survival time (a slowed clock would buy "
        "it); its table is no record of the check.  R0L, which has no clock to slow, is then credited its survival "
        "time"),
    gain=(
        "rule M3: a check shows a gain when N >= N* + 2 or S >= S* + 1.0 s; N* is the largest N, and S* the largest "
        "S, over R0L's episodes and every earlier check of the run, each taken separately (for the first check: "
        "R0L's)"),
    m3b=(
        "rule M3b (S2-4's reading alone): when the mean planned length of the episodes less R0L's S is under 1.0 s, "
        "a gain is also at least 2 more episodes than R0L succeeding in reading 3 with the look bands kept"),
    g8b=(
        "rule M4: at the check of the multiple 5,000,000: walk/nominal stays up (read or unread: it did not fall and "
        "is no error) in at least 2 of its 3 episodes.  Otherwise, once: each training task that the table gives to "
        "at least one monitor condition with fewer than 2 of 3 up has its weight doubled, once (walk/nominal: "
        "walk_nominal; walk/v0.4, nudge/forward, nudge/left: walk_v0.4; turn/q90_left: turn_q90_left and "
        "turn_q90_right; stop/two_step: stop and start_stop); the reward's weights do not change.  At the check of "
        "the multiple 10,000,000 the same reading; a second failure stops the run.  Checks are named by their "
        "multiple.  A tiny run reads at its first check and at the next; with fewer than 3 episodes of a condition "
        "\"2 of 3\" is ceil(2 n / 3) of n"),
    stop=(
        "rule M5: counting from the check of the multiple 20,000,000, the run stops at the third consecutive check "
        "without a gain"),
    attempts=(
        "rule M1 with the lead's rulings of 16 Nov 2026: an attempt at a check is logged in run.json monitor[] "
        "before the monitor starts, with the checkpoint's file hashes, and closed with the monitor's exit code when "
        "it ends.  A record is a monitor_<n>.json that names the checkpoint's hashes and holds an outcome for every "
        "one of its episodes.  After two closed attempts that left no record the check counts as without a gain and "
        "as a failed G8b reading, each of its episodes an error.  An attempt that was never closed (the session "
        "died while the check ran) is an interruption: the resume runs the check again and it does not count; after "
        "two of them at one check the run halts and goes to the lead.  A condition without a reference is held in a "
        "record as \"not run\" (the lead's fifth list of 16 Nov 2026, 1).  A monitor that refuses to run is no "
        "attempt and is not one of the two (refusal)"),
    refusal=(
        "rule M1 with the lead's ruling of 16 Nov 2026, 19:01 UTC (the fifth list, 6) and the sixth list of 16 Nov "
        "2026 (3): a monitor that refuses to run is no attempt.  It refuses, before any episode, when its sources "
        "are not the ones the trainer hashed (the checkout moved under the run), when the run's run.json is missing, "
        "cannot be read or is no JSON object, when the checkpoint is not intact (no record, a record that cannot be "
        "read or has the wrong shape: its files, and a layout and a fingerprint object, the rulings of 17 Nov 2026, "
        "12; or a file that is not the one its record names), when the checkpoint is "
        "another run's (its run_uuid or settings hash is not the run's), and when the row refuses the checkpoint "
        "while it is built (s2rows.RowError; s2load.CheckpointError: among them a fingerprint not reproduced at the "
        "row's load).  The monitor then leaves with exit code 8 and writes its reason to monitor_<n>.refused.json "
        "(the baseline: r0l.refused.json).  The trainer closes the attempt with that exit code, marks it refused "
        "with the reason, does not count it toward the two closed attempts after which a check counts as without a "
        "gain, and halts with exit code 5: the run goes to the lead, and a resume runs the check again (a resume "
        "with --retry-check when the refused attempt was the one that parameter let run; a halt for moved sources, "
        "the monitor's own or the row's rule K9, names --allow-source-change, which resolves it)"),
    sources=(
        "the trainer hashes its sources once per session and gives the hash to this process, which refuses to run "
        "if its own differs (train2.sources)"),
)


class MonitorError(RuntimeError):
    """The monitor cannot run or cannot write a usable record: nothing is recorded as a check's result."""


class MonitorRefusal(MonitorError):
    """The monitor refuses to run (``RULES["refusal"]``): no episode was run and no attempt at the check was made.
    ``kind``: one of ``REFUSALS``."""

    def __init__(self, kind: str, message: str):
        if kind not in REFUSALS:
            raise ValueError(f"refusal kind {kind!r} is none of {', '.join(REFUSALS)}")
        super().__init__(message)
        self.kind = kind


# ------------------------------------------------------------------------------------------------
# Conditions, seeds, rows
# ------------------------------------------------------------------------------------------------
def conditions_for(set_name: str) -> tuple:
    """The monitor's conditions on a set: ``CONDITIONS``; on the Stage 1 view ``S1_CONDITIONS``."""
    return S1_CONDITIONS if set_name == S1_SET else CONDITIONS


def seeds_for(set_name: str) -> dict:
    """condition -> its three seeds on a set (rule M1's formula on the set's own condition order)."""
    return {c: (6001 + 3 * i, 6002 + 3 * i, 6003 + 3 * i) for i, c in enumerate(conditions_for(set_name))}


def pairs_for(set_name: str, tiny: bool = False) -> list:
    """``[(condition key, seed)]`` of a check: every seed of every condition, or each condition's first (tiny)."""
    seeds = seeds_for(set_name)
    return [(c, s) for c in conditions_for(set_name) for s in (seeds[c][:1] if tiny else seeds[c])]


def fix_tasks_for(set_name: str) -> dict:
    return {c: list(t) for c, t in (S1_FIX_TASKS if set_name == S1_SET else FIX_TASKS).items()}


def row_name(mode: str | None, seed: int | None, k: int = K) -> str:
    """``R5/42@10ms`` (a checkpoint's row), ``R0L@10ms`` (``mode`` None): the bar's names, the delay in ms."""
    ms = "%gms" % (STEP_MS * int(k))
    return f"R0L@{ms}" if mode is None else "%s/%d@%s" % (mode, int(seed), ms)


def rules(monitor_every: int, tiny: bool = False, set_name: str = "V6") -> dict:
    """The numbers ``decide`` reads, with the rule texts: ``monitor_every``; G8b's two named checks (a tiny run:
    its first check and the next) and its table; the stop rule's start (None in a tiny run: it has two checks)."""
    every = int(monitor_every)
    if every <= 0:
        raise ValueError(f"monitor_every {monitor_every!r} must be positive")
    return dict(monitor_every=every, tiny=bool(tiny), set=set_name,
                gain=dict(n_more=GAIN_N, s_more_s=GAIN_S),
                g8b=dict(condition=G8B_CONDITION, at=every if tiny else G8B_AT,
                         again_at=2 * every if tiny else G8B_AGAIN_AT, fix_tasks=fix_tasks_for(set_name)),
                stop=dict(start=None if tiny else STOP_FROM, checks=STOP_CHECKS), attempts=ATTEMPTS,
                texts={k: RULES[k] for k in ("gain", "g8b", "stop", "attempts", "refusal")})


# ------------------------------------------------------------------------------------------------
# One episode, and a table of episodes
# ------------------------------------------------------------------------------------------------
def _num(x, default: float = 0.0) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def survival_credit(e: dict) -> float:
    """Rule M3's S of one episode: ``survival_s x min(1, reference time covered / (1.05 x survival_s))``.  The
    reference time covered is ``clock.t_ref_reached`` less the track's start, ``clock.t_ref_planned - 1.05 x
    planned_s``.  A clock that covered at least 1.05 x ``survival_s`` (to within ``CREDIT_RTOL`` of it: float
    rounding) is credited the survival time itself, not a product that rounds below it: R0L's S is then exactly
    its mean survival (the lead's ruling 17 of 17 Nov 2026).  An episode whose record holds no clock times is
    credited its survival time: that is R0L's case alone, for ``run`` takes no policy row's table that holds such
    an episode (``has_clock``; ``_checked``).  An *error* and a condition that was not run are credited 0."""
    if e.get("outcome") in ("error", NOT_RUN):
        return 0.0
    surv = _num(e.get("survival_s"))
    if surv <= 0.0:
        return 0.0
    clock = e.get("clock") or {}
    reached, planned = clock.get("t_ref_reached"), clock.get("t_ref_planned")
    if reached is None or planned is None or e.get("planned_s") is None:
        return surv
    covered = _num(reached) - (_num(planned) - RATE * _num(e["planned_s"]))
    full = RATE * surv
    if covered >= full * (1.0 - CREDIT_RTOL):
        return surv
    return surv * max(0.0, covered) / full


def has_clock(e: dict) -> bool:
    """Whether an episode holds what its credited survival is computed from: both clock times and its planned
    length."""
    clock = e.get("clock") or {}
    return (clock.get("t_ref_reached") is not None and clock.get("t_ref_planned") is not None
            and e.get("planned_s") is not None)


def _checked(episodes, pairs, policy: bool = False) -> list:
    """The episodes as returned, one per pair and in the pairs' order, each with a known outcome; ``MonitorError``
    otherwise (a table that is not the check's is no record).  ``policy``: the row is a checkpoint's; an episode
    of it that ended read, fell or unread must hold its clock times (rule M3's credit exists to stop a slowed
    clock from buying survival, so it does not fall back to the plain survival time there)."""
    episodes = list(episodes)
    if len(episodes) != len(pairs):
        raise MonitorError(f"{len(episodes)} episodes came back for {len(pairs)} (condition, seed) pairs")
    out = []
    for e, (cond, seed) in zip(episodes, pairs, strict=True):
        if not isinstance(e, dict) or e.get("condition") != cond or int(e.get("seed", -1)) != int(seed):
            raise MonitorError(f"episode {e.get('condition') if isinstance(e, dict) else e!r} seed "
                               f"{e.get('seed') if isinstance(e, dict) else None} came back for {cond} seed {seed}")
        if e.get("outcome") not in OUTCOMES + (NOT_RUN,):
            raise MonitorError(f"{cond} seed {seed}: outcome {e.get('outcome')!r} is none of "
                               f"{', '.join(OUTCOMES + (NOT_RUN,))}")
        if policy and e.get("outcome") in ("read", "fell", "unread") and not has_clock(e):
            raise MonitorError(f"{cond} seed {seed}: a policy row's episode ({e.get('outcome')}) came back without "
                               "its clock times (clock.t_ref_reached, clock.t_ref_planned, planned_s): its "
                               "credited survival cannot be computed (rule M3), so the table is no record")
        e = dict(e)
        e["survival_credit"] = survival_credit(e)
        out.append(e)
    return out


def _ok(e: dict, key: str) -> bool:
    """Whether an episode succeeds in a reading: only a *read* episode can (rule M1)."""
    return e.get("outcome") == "read" and bool(e.get(key))


def summarize(episodes: list) -> dict:
    """What a record holds of its episodes (each with ``survival_credit``): N, S, the reported counts and means,
    the outcomes, per condition how many stayed up, the failures per term, the mean clock factor, the number of
    distinct start seeds and L (the mean planned length; a condition not run counts 0)."""
    n = len(episodes)
    by_cond: dict = {}
    terms: dict = {}
    outcomes = {o: 0 for o in OUTCOMES + (NOT_RUN,)}
    reasons: dict = {}
    factors = []
    for e in episodes:
        outcomes[e["outcome"]] += 1
        c = by_cond.setdefault(e["condition"], dict(up=0, of=0))
        c["of"] += 1
        c["up"] += e["outcome"] in UP
        for t in e.get("failed_terms") or ():
            terms[str(t)] = terms.get(str(t), 0) + 1
        if e["outcome"] == "unread":
            r = str(e.get("reason") or "unreadable")
            reasons[r] = reasons.get(r, 0) + 1
        f = (e.get("clock") or {}).get("mean_factor")
        if f is not None and e["outcome"] not in ("error", NOT_RUN):
            factors.append(float(f))
    survived = [0.0 if e["outcome"] in ("error", NOT_RUN) else _num(e.get("survival_s")) for e in episodes]
    planned = [0.0 if e["outcome"] == NOT_RUN else _num(e.get("planned_s")) for e in episodes]
    return dict(
        n_episodes=n, N=sum(_ok(e, "succeeded_look") for e in episodes),
        S=sum(survival_credit(e) for e in episodes) / n if n else 0.0,
        N_reading2=sum(_ok(e, "succeeded") for e in episodes),
        N_reading3=sum(_ok(e, "succeeded_but_A1") for e in episodes),
        N_but_A1_look=sum(_ok(e, "succeeded_but_A1_look") for e in episodes),
        mean_survival_s=sum(survived) / n if n else 0.0, L_planned_s=sum(planned) / n if n else 0.0,
        outcomes=outcomes, unread_reasons=reasons, up_by_condition=by_cond,
        failures_by_term={t: terms[t] for t in sorted(terms)},
        mean_clock_factor=sum(factors) / len(factors) if factors else None,
        distinct_start_seeds=len({int(e.get("start_seed", e["seed"])) for e in episodes}))


SUMMARY_KEYS = ("N", "S", "N_reading2", "N_reading3", "N_but_A1_look", "mean_survival_s", "mean_clock_factor",
                "failures_by_term", "outcomes")


# ------------------------------------------------------------------------------------------------
# The rules on records (pure)
# ------------------------------------------------------------------------------------------------
def gain(now: dict, earlier: list, baseline: dict) -> dict:
    """Rule M3 on records: ``now`` shows a gain when its N is at least N* + 2 or its S at least S* + 1.0 s, N* and
    S* the largest N and the largest S over R0L's episodes (``baseline``) and the earlier checks, each taken
    separately.  Every N and S is recomputed from the records' episodes."""
    mine = summarize(now["episodes"])
    others = [summarize(baseline["episodes"])] + [summarize(m["episodes"]) for m in earlier]
    n_star, s_star = max(o["N"] for o in others), max(o["S"] for o in others)
    by_n, by_s = mine["N"] >= n_star + GAIN_N, mine["S"] >= s_star + GAIN_S
    return dict(N=mine["N"], S=mine["S"], N_star=n_star, S_star=s_star, by_N=bool(by_n), by_S=bool(by_s),
                gain=bool(by_n or by_s), against=dict(baseline=True, earlier_checks=len(earlier)))


def need_up(of: int) -> int:
    """"2 of 3" for ``of`` episodes of a condition: ceil(2 of / 3) (2 of 3; a tiny run's 1 of 1)."""
    return int(math.ceil(2 * int(of) / 3))


def _up(mon: dict) -> dict:
    return summarize(mon["episodes"])["up_by_condition"]


def g8b(mon: dict) -> dict:
    """Rule M4's reading on one check's record: ``walk/nominal`` up (read or unread) in at least 2 of its 3."""
    c = _up(mon).get(G8B_CONDITION, dict(up=0, of=0))
    need = need_up(c["of"])
    return dict(condition=G8B_CONDITION, up=c["up"], of=c["of"], need=need,
                **{"pass": bool(c["of"] > 0 and c["up"] >= need)})


def conditions_down(mon: dict) -> list:
    """The record's conditions with fewer than 2 of 3 up, in the record's order."""
    return [c for c, v in _up(mon).items() if v["up"] < need_up(v["of"])]


def fix_weights(mon: dict, weights: dict) -> dict:
    """Rule M4's fix: ``weights`` (task -> weight) with the weight of every task doubled, once, that the record's
    table (``fix_tasks``) gives to at least one of its conditions with fewer than 2 of 3 up.  A task the table
    names and ``weights`` does not hold (its reference is absent from the set) is left out."""
    table = mon.get("fix_tasks") or FIX_TASKS
    doubled = []
    for c in conditions_down(mon):
        for t in table.get(c, ()):
            if t in weights and t not in doubled:
                doubled.append(t)
    return {t: (2.0 * float(w) if t in doubled else float(w)) for t, w in weights.items()}


def _summary(mon: dict) -> dict:
    s = summarize(mon["episodes"])
    return {k: s[k] for k in SUMMARY_KEYS}


def decide(history: list, baseline: dict, state: dict, rules: dict) -> dict:
    """Everything that is read after a check, from records alone.

    ``history``: the run's checks in order, each a monitor record (or ``no_record``'s stand-in for a check counted
    without one).  ``baseline``: R0L's record.  ``state``: ``{"task_weights": {task: weight}}`` as the run started,
    before any fix.  ``rules``: ``rules(monitor_every, tiny, set)``.

    Returns the run state after the last check: ``checks`` (per check: ``at_steps``, its ``multiple``, the
    summary, the ``gain``, a G8b reading where one was made, ``no_gain_count``), ``gates`` (``g8b``,
    ``g8b_after_fix``), ``fix`` (``{rule, at_steps, multiple, conditions_down, tasks, weights_before, weights}`` or
    None), ``task_weights`` (in force after the last check), ``no_gain_count``, ``checks_done`` and ``stopped``
    (``{at_steps, multiple, status, reason}`` or None).  The same call on the same records gives the same state,
    whether it is made live after a check or at a resume."""
    every = int(rules["monitor_every"])
    weights = {str(t): float(w) for t, w in state["task_weights"].items()}
    first = again = fix = stopped = None
    no_gain, checks = 0, []
    for i, mon in enumerate(history):
        n = int(mon["at_steps"])
        if stopped is not None:
            raise ValueError(f"check {n}: the run stopped at {stopped['at_steps']} steps and has no later check")
        if i and n <= int(history[i - 1]["at_steps"]):
            raise ValueError(f"the checks are not in order ({history[i - 1]['at_steps']}, then {n})")
        multiple = (n // every) * every
        g = gain(mon, history[:i], baseline)
        entry = dict(at_steps=n, multiple=multiple, no_record=bool(mon.get("no_record")), **_summary(mon), gain=g)
        if first is None and multiple >= rules["g8b"]["at"]:
            first = dict(g8b(mon), at_steps=n, multiple=multiple)
            entry["g8b"] = first
            if not first["pass"]:
                new = fix_weights(dict(mon, fix_tasks=mon.get("fix_tasks") or rules["g8b"]["fix_tasks"]), weights)
                fix = dict(rule=RULES["g8b"], at_steps=n, multiple=multiple, conditions_down=conditions_down(mon),
                           tasks=[t for t in weights if new[t] != weights[t]], weights_before=dict(weights),
                           weights=dict(new))
                weights = new
        elif first is not None and not first["pass"] and again is None and multiple >= rules["g8b"]["again_at"]:
            again = dict(g8b(mon), at_steps=n, multiple=multiple)
            entry["g8b_after_fix"] = again
            if not again["pass"]:
                stopped = dict(at_steps=n, multiple=multiple, status="stopped_g8b",
                               reason=f"G8b failed again at the check of the multiple {multiple} (rule M4: "
                                      f"{again['condition']} up in {again['up']} of {again['of']})")
        start = rules["stop"]["start"]
        if start is not None and multiple >= start:
            no_gain = 0 if g["gain"] else no_gain + 1
            if stopped is None and no_gain >= rules["stop"]["checks"]:
                stopped = dict(at_steps=n, multiple=multiple, status="stopped_no_gain",
                               reason=f"{no_gain} consecutive checks without a gain from the check of the multiple "
                                      f"{start} (rule M5)")
        entry["no_gain_count"] = no_gain
        checks.append(entry)
    return dict(checks=checks, gates=dict(g8b=first, g8b_after_fix=again), fix=fix, task_weights=weights,
                no_gain_count=no_gain, checks_done=[c["at_steps"] for c in checks], stopped=stopped)


def no_record(at_steps: int, set_name: str, tiny: bool = False, attempts: int = ATTEMPTS, k: int = K) -> dict:
    """The stand-in for a check that left no record after ``attempts`` closed attempts (rule M1; the lead's ruling
    of 16 Nov 2026): each of its episodes is an *error*, so the check is without a gain and its G8b reading
    fails."""
    eps = [dict(condition=c, seed=s, outcome="error", reason=None, fell=None, survival_s=0.0, planned_s=None,
                succeeded=False, succeeded_but_A1=False, succeeded_look=False, succeeded_but_A1_look=False,
                failed_terms=[], look=None, clock=None, error="no record after %d closed attempts" % attempts,
                survival_credit=0.0) for c, s in pairs_for(set_name, tiny)]
    return dict(schema=SCHEMA, kind="check", no_record=True, closed_attempts=int(attempts), at_steps=int(at_steps),
                set=set_name, tiny=bool(tiny), k=int(k), conditions=list(conditions_for(set_name)),
                fix_tasks=fix_tasks_for(set_name), episodes=eps, **summarize(eps), rule=RULES["attempts"])


def checkpoint_hashes(record: dict) -> dict:
    """``{model, vecnormalize, fingerprint}`` -> sha256, from a checkpoint's record."""
    return {kind: record["files"][kind]["sha256"] for kind in ("model", "vecnormalize", "fingerprint")}


def is_record(doc, checkpoint_sha256: dict, pairs: list) -> bool:
    """Whether ``doc`` is a record of the check on this checkpoint (the lead's ruling of 16 Nov 2026): a
    ``monitor_<n>.json`` that names the checkpoint's three file hashes and holds an outcome for every one of its
    episodes."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("no_record"):
        return False
    if doc.get("checkpoint_sha256") != dict(checkpoint_sha256):
        return False
    eps = doc.get("episodes")
    if not isinstance(eps, list) or len(eps) != len(pairs):
        return False
    return all(isinstance(e, dict) and e.get("condition") == c and e.get("seed") == s
               and e.get("outcome") in OUTCOMES + (NOT_RUN,) for e, (c, s) in zip(eps, pairs, strict=True))


def baseline_problem(doc) -> str | None:
    """None when R0L's record can be read against (rule M1: every episode read or fell; a condition without a
    reference is not run), else why not."""
    if not isinstance(doc, dict) or doc.get("schema") != BASELINE_SCHEMA or not isinstance(doc.get("episodes"), list):
        return "no baseline record"
    bad = [f"{e.get('condition')} seed {e.get('seed')}: {e.get('outcome')}" for e in doc["episodes"]
           if e.get("outcome") not in ("read", "fell", NOT_RUN)]
    return ("R0L's baseline holds episodes that are neither read nor fell (" + "; ".join(bad) + ")") if bad else None


#: The keys an episode holds its state digest under (``lookbar``'s ``state_digest``; ``digest``).
DIGEST_KEYS = ("state_digest", "digest")


def digest_compared(x: dict, y: dict) -> bool:
    """Whether two episodes' state digests are compared (the lead's ruling 33 of 17 Nov 2026): under one of
    ``DIGEST_KEYS`` both hold a non-empty string (a digest of None, which ``lookbar`` writes when taking it
    raised, is none)."""
    return any(isinstance(x.get(k), str) and x[k] != "" and isinstance(y.get(k), str) and y[k] != ""
               for k in DIGEST_KEYS)


def baseline_equal(r0l: dict, committed: dict) -> dict:
    """Rule M3: S2-4's own ``monitor/r0l.json`` against the committed ``R0L_BASELINE.json``, episode for episode:
    the condition, the seed, the outcome, the survival and, where both hold one, the state digest.
    ``digests_compared`` (the lead's ruling 33 of 17 Nov 2026): the digests were compared in every episode that
    ran, that is in every pair of which at least one episode ran (its outcome is not ``NOT_RUN``) both hold one
    (``digest_compared``); two episodes not run (``lookbar`` gives a condition without a reference no digest) are
    compared on their outcome and survival alone.  False when no episode ran."""
    a, b = list(r0l.get("episodes") or []), list(committed.get("episodes") or [])
    differ, ran = [], []
    if len(a) != len(b):
        differ.append(f"{len(a)} episodes against {len(b)}")
    for x, y in zip(a, b, strict=False):
        keys = ["condition", "seed", "outcome", "survival_s"]
        keys += [k for k in DIGEST_KEYS if k in x and k in y]
        bad = [k for k in keys if x.get(k) != y.get(k)]
        if bad:
            differ.append(f"{x.get('condition')} seed {x.get('seed')}: {', '.join(bad)}")
        if x.get("outcome") != NOT_RUN or y.get("outcome") != NOT_RUN:
            ran.append(digest_compared(x, y))
    return dict(equal=not differ, episodes=len(a), differ=differ, digests_compared=bool(ran) and all(ran))


def s2_10(final: dict, earlier: list, baseline: dict) -> dict:
    """S2-10's one record (rules M3, M3b): S2-4's final checkpoint read against R0L alone, with both rows'
    episodes, N, S, the plain mean survival, the reported counts, the failures per term and the mean clock factor;
    the earlier checks reported (they decide nothing); M3b's route when the survival route is closed."""
    pol, r0l = summarize(final["episodes"]), summarize(baseline["episodes"])
    g = gain(final, [], baseline)
    L = r0l["L_planned_s"]
    closed = bool(L - r0l["S"] < ROUTE_CLOSED_S)
    by_a1 = bool(pol["N_but_A1_look"] >= r0l["N_but_A1_look"] + GAIN_N)
    keys = ("N", "S", "mean_survival_s", "N_reading2", "N_reading3", "N_but_A1_look", "failures_by_term",
            "mean_clock_factor", "outcomes", "unread_reasons", "up_by_condition", "distinct_start_seeds")
    doc = dict(
        schema=S2_10_SCHEMA, at_steps=int(final["at_steps"]), row=final.get("row"), baseline_row=baseline.get("row"),
        job=final.get("job"), run_uuid=final.get("run_uuid"), checkpoint_sha256=final.get("checkpoint_sha256"),
        policy={k: pol[k] for k in keys}, r0l={k: r0l[k] for k in keys},
        gain=g, L_planned_s=L, survival_route_closed=closed,
        m3b=dict(applies=closed, N_but_A1_look=pol["N_but_A1_look"], N_but_A1_look_r0l=r0l["N_but_A1_look"],
                 gain=bool(closed and by_a1)),
        go=bool(g["gain"] or (closed and by_a1)),
        earlier=[dict(at_steps=int(m["at_steps"]), **_summary(m), gain_against_r0l=gain(m, [], baseline),
                      note="reported; decides nothing") for m in earlier],
        episodes=dict(policy=final["episodes"], r0l=baseline["episodes"]), baseline_equal=None,
        rule=dict(gain=RULES["gain"], survival=RULES["survival"], m3b=RULES["m3b"], outcomes=RULES["outcomes"],
                  episodes=RULES["episodes"]),
        note="a gain lets Colab start; no gain goes to the owner first, with both rows' failures per term")
    return doc


# ------------------------------------------------------------------------------------------------
# The episodes (the one indirection to lookbar and s2rows)
# ------------------------------------------------------------------------------------------------
def _row_refusal(exc: BaseException) -> str | None:
    """The refusal kind of an exception raised while a row is built (``ROW_REFUSALS``), or None: an instance of
    ``s2rows.RowError`` (``row``) or of ``s2load.CheckpointError`` (``checkpoint``), each class looked up in
    ``sys.modules``."""
    for module, name, kind in ROW_REFUSALS:
        cls = getattr(sys.modules.get(module), name, None)
        if isinstance(cls, type) and isinstance(exc, cls):
            return kind
    return None


def _row_refused(row: str, kind: str, exc: BaseException) -> MonitorRefusal:
    return MonitorRefusal(kind, f"the row {row} refused the checkpoint while it was built, before any episode "
                                f"({type(exc).__name__}: {exc})")


def _episodes(body: str, set_name: str, row: str, pairs: list, policies: dict | None = None, traces=None) -> list:
    """One companion episode per ``(condition key, seed)`` of ``pairs`` under the row ``row`` (``R0L@10ms``, or a
    checkpoint's ``R5/42@10ms`` with ``policies = {"R5/42": {"job_dir", "steps"}}``): ``lookbar.episodes`` on
    ``lookbar.rows(body, [row])`` after ``s2rows.configure(policies)``.  ``traces``: where the episodes' traces
    go.  Every set, the Stage 1 view's included, goes through ``lookbar`` (module docstring): there is no stand-in,
    and a checkout that cannot import ``lookbar`` raises its import error.

    The row is built before any episode: a ``ROW_REFUSALS`` exception raised there (by ``s2rows.configure`` or
    ``lookbar.rows``) is the monitor's ``MonitorRefusal``.  Whatever ``lookbar.episodes`` raises is not (an
    episode's own fault is an *error* in its table)."""
    import importlib

    lookbar = importlib.import_module("designlab.stage2.lookbar")
    try:
        if policies:
            s2rows = importlib.import_module("designlab.stage2.s2rows")
            s2rows.configure({name: dict(job_dir=str(p["job_dir"]), steps=int(p["steps"]))
                              for name, p in policies.items()})
        rows = lookbar.rows(body, [row])
    except Exception as exc:
        kind = _row_refusal(exc)
        if kind is None:
            raise
        raise _row_refused(row, kind, exc) from exc
    return lookbar.episodes(body, set_name, rows, list(pairs), traces=traces)


# ------------------------------------------------------------------------------------------------
# The records
# ------------------------------------------------------------------------------------------------
def _job_doc(job_dir, refuse: bool = False) -> dict:
    """The run's ``run.json``.  One that is missing, cannot be read or is no JSON object raises ``MonitorError``;
    with ``refuse`` (``run`` and ``baseline``, before any episode: the sixth list of 16 Nov 2026, 3) it is a
    ``MonitorRefusal`` of kind ``run``."""
    path = Path(job_dir) / "run.json"

    def fail(text: str) -> MonitorError:
        return MonitorRefusal("run", text) if refuse else MonitorError(text)

    if not path.is_file():
        raise fail(f"{path}: no run.json (the monitor reads the run's mode, seed and run id from it)")
    try:
        doc = s2io.read_json(path)
    except (OSError, ValueError) as exc:
        raise fail(f"{path}: the run's run.json cannot be read ({type(exc).__name__})") from exc
    if not isinstance(doc, dict):
        raise fail(f"{path}: the run's run.json is no JSON object")
    return doc


def _sources(given: str | None) -> dict:
    """``{given, own}``: the trainer's sources hash and this process's; ``MonitorRefusal`` if they differ."""
    from designlab.stage2 import train2

    own = train2.sources()["sha256"]
    if given is not None and str(given) != own:
        raise MonitorRefusal("sources", f"the sources differ: the trainer gave {str(given)[:16]}..., this process "
                                        f"has {own[:16]}... (the checkout moved under the run): no episode is run")
    return dict(given=None if given is None else str(given), own=own)


def _checkpoint(job_dir, steps: int) -> dict:
    """The checkpoint's record, after its three files' sizes and sha256 were checked against it (nothing is
    unpickled here: the row loads the checkpoint through ``s2load.load_checkpoint``).  ``MonitorRefusal`` on a
    checkpoint that is not intact: no record, a record that cannot be read, a record of the wrong shape (``files``
    must map each of ``s2load.FILE_KINDS`` to ``{bytes: int, sha256: str}``, and ``layout`` and ``fingerprint``
    must be objects: the row's load reads both, the lead's ruling 12 of 17 Nov 2026), a file that is not the one
    its record names."""
    from designlab.stage2 import s2load

    files = s2load.checkpoint_files(job_dir, steps)
    if not files["record"].is_file():
        raise MonitorRefusal("checkpoint", f"{files['record']}: no checkpoint record (checkpoint {steps} is not "
                                           "intact)")
    try:
        rec = s2io.read_json(files["record"])
    except (OSError, ValueError) as exc:
        raise MonitorRefusal("checkpoint", f"{files['record']}: the checkpoint's record cannot be read "
                                           f"({type(exc).__name__}; checkpoint {steps} is not intact)") from exc
    if not isinstance(rec, dict):
        raise MonitorRefusal("checkpoint", f"{files['record']}: the checkpoint's record is no JSON object "
                                           f"(checkpoint {steps} is not intact)")
    named = rec.get("files")
    shaped = isinstance(named, dict) and all(
        isinstance(named.get(kind), dict) and type(named[kind].get("bytes")) is int
        and isinstance(named[kind].get("sha256"), str) for kind in s2load.FILE_KINDS)
    if not shaped:
        raise MonitorRefusal("checkpoint", f"{files['record']}: the checkpoint's record has the wrong shape (files: "
                                           f"{', '.join(s2load.FILE_KINDS)} -> {{bytes, sha256}}; checkpoint {steps} "
                                           "is not intact)")
    lacks = [key for key in ("layout", "fingerprint") if not isinstance(rec.get(key), dict)]
    if lacks:
        raise MonitorRefusal("checkpoint", f"{files['record']}: the checkpoint's record has the wrong shape (no "
                                           f"{' and no '.join(lacks)} object; checkpoint {steps} is not intact)")
    for kind in s2load.FILE_KINDS:
        want = named[kind]
        p = files[kind]
        if not p.is_file() or p.stat().st_size != want.get("bytes") or s2io.sha256_file(p) != want.get("sha256"):
            raise MonitorRefusal("checkpoint", f"{p}: not the file its record names (checkpoint {steps} is not "
                                               "intact)")
    return rec


def _also(body, set_name, mode, seed, pairs, policies, also_k, k) -> dict:
    """The same episodes at further delays (rule M6: reported; they decide nothing).  They run after the check's
    own episodes: a row that refuses here makes no refusal of the call (``MonitorError``)."""
    out = {}
    for kk in also_k:
        kk = int(kk)
        if kk == int(k):
            continue
        row = row_name(mode, seed, kk)
        try:
            got = _episodes(body, set_name, row, pairs, policies, None)
        except MonitorRefusal as exc:
            raise MonitorError(f"{exc} (at a further delay, after the check's own episodes: no refusal)") from exc
        eps = _checked(got, pairs, policy=mode is not None)
        out["%d" % kk] = dict(row=row, k=kk, episodes=eps, **summarize(eps), note="reported (rule M6)")
    return out


def installed_versions() -> dict:
    """The installed versions of the libraries a checkpoint records (``s2load.VERSION_KEYS``), read from the
    distributions' metadata: nothing is imported for it (the baseline's process holds no torch)."""
    import platform
    from importlib import metadata

    out = {"python": platform.python_version()}
    for key, dist in (("numpy", "numpy"), ("mujoco", "mujoco"), ("scipy", "scipy"), ("torch", "torch"),
                      ("stable_baselines3", "stable-baselines3"), ("gymnasium", "gymnasium"),
                      ("cloudpickle", "cloudpickle")):
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    return out


def refusal_path(out, steps=None) -> Path:
    """Where the CLI writes a refusal's reason: ``<out>/monitor_<steps>.refused.json`` for a check,
    ``<out>/r0l.refused.json`` for the baseline (``steps`` None)."""
    return Path(out) / ("r0l.refused.json" if steps is None else "monitor_%d.refused.json" % int(steps))


def _note_refusal(path: Path, cmd: str, steps, exc: MonitorRefusal) -> None:
    """The refusal's note (``RULES["refusal"]``): what the trainer names in its halt.  A note that cannot be written
    changes nothing: the exit code alone says that the monitor refused."""
    try:
        s2io.write_json(path, dict(schema=REFUSAL_SCHEMA, cmd=cmd, at_steps=None if steps is None else int(steps),
                                   kind=exc.kind, reason=str(exc), rule=RULES["refusal"], at=s2io.now_utc()))
    except OSError as err:
        print(f"monitor2 {cmd}: the refusal's note {path} could not be written ({err})", file=sys.stderr, flush=True)


def _write(path: Path, doc: dict) -> dict:
    doc = s2io.plain(doc)
    doc["sha256"] = s2io.sha256_json(doc)  # of the record without this key
    s2io.write_json(path, doc)
    return doc


def run(job_dir, steps, body, set_name, out, k=K, tiny=False, also_k=(), sources_sha256=None) -> dict:
    """The check of checkpoint ``steps`` of a job: its episodes under the checkpoint's row, the record
    ``<out>/monitor_<steps>.json`` (also returned).  ``sources_sha256``: the trainer's sources hash; with it this
    process refuses to run when its own differs.  ``MonitorRefusal`` (no episode, no record: no attempt) when the
    sources differ, the run's ``run.json`` is missing or cannot be read, the checkpoint is not intact, the
    checkpoint is another run's or the row refuses it while it is built; ``MonitorError`` (and no record) when the
    episodes that come back are not the check's."""
    t0 = time.perf_counter()
    n = int(steps)
    job_dir, out = Path(job_dir), Path(out)
    job = _job_doc(job_dir, refuse=True)
    src = _sources(sources_sha256)
    ck = _checkpoint(job_dir, n)
    for key in ("run_uuid", "settings_sha256"):
        if ck.get(key) != job.get(key):
            raise MonitorRefusal("other_run", f"checkpoint {n}: its {key} is not the run's: another run's "
                                              "checkpoint is not judged")
    mode, seed = job["mode"], int(job["seed"])
    pairs = pairs_for(set_name, tiny)
    row = row_name(mode, seed, k)
    policies = {row.split("@", 1)[0]: dict(job_dir=str(job_dir), steps=n)}
    traces = out / "traces" / ("%d" % n)
    eps = _checked(_episodes(body, set_name, row, pairs, policies, traces), pairs, policy=True)
    doc = dict(
        schema=SCHEMA, kind="check", job=job.get("job"), run_uuid=job.get("run_uuid"), run_id=job.get("run_id"),
        settings_sha256=job.get("settings_sha256"), mode=mode, body=body, set=set_name, ppo_seed=seed, row=row,
        at_steps=n, k=int(k), tiny=bool(tiny), conditions=list(conditions_for(set_name)),
        pairs=[[c, s] for c, s in pairs], fix_tasks=fix_tasks_for(set_name),
        checkpoint=dict(steps=n, files=ck["files"]), checkpoint_sha256=checkpoint_hashes(ck), episodes=eps,
        **summarize(eps), also=_also(body, set_name, mode, seed, pairs, policies, also_k, k),
        traces="traces/%d" % n if traces.is_dir() else None, episodes_by=EPISODES_BY,
        sources=src, flags=dict(job.get("flags") or {}),
        rules={key: RULES[key] for key in ("episodes", "outcomes", "survival", "sources")},
        provenance=s2io.git_state(own_outputs=[str(job_dir.resolve())]), versions=installed_versions(),
        wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    return _write(out / ("monitor_%d.json" % n), doc)


def baseline(body, set_name, out, k=K, tiny=False, also_k=(), sources_sha256=None, job_dir=None) -> dict:
    """R0L on the same episodes, once per run: the record ``<out>/r0l.json`` (also returned).  A baseline that
    holds an episode which is neither read nor fell (rule M1) is written beside instead, as
    ``r0l.failed.<i>.json``, and ``MonitorError`` is raised: the fault is repaired and all its episodes are run
    again before the run's first rollout."""
    t0 = time.perf_counter()
    out = Path(out)
    src = _sources(sources_sha256)
    job = _job_doc(job_dir, refuse=True) if job_dir is not None else {}
    pairs = pairs_for(set_name, tiny)
    row = row_name(None, None, k)
    traces = out / "traces" / "r0l"
    eps = _checked(_episodes(body, set_name, row, pairs, None, traces), pairs)
    doc = dict(
        schema=BASELINE_SCHEMA, kind="baseline", job=job.get("job"), run_uuid=job.get("run_uuid"),
        run_id=job.get("run_id"), body=body, set=set_name, row=row, k=int(k), tiny=bool(tiny),
        conditions=list(conditions_for(set_name)), pairs=[[c, s] for c, s in pairs], episodes=eps, **summarize(eps),
        also=_also(body, set_name, None, None, pairs, None, also_k, k),
        traces="traces/r0l" if traces.is_dir() else None, episodes_by=EPISODES_BY,
        sources=src, rules={key: RULES[key] for key in ("episodes", "outcomes", "survival", "sources")},
        provenance=s2io.git_state(own_outputs=[str(Path(job_dir).resolve())] if job_dir is not None else [str(out)]),
        versions=installed_versions(), wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    why = baseline_problem(doc)
    if why is not None:
        i = 0
        while (out / ("r0l.failed.%d.json" % i)).exists():
            i += 1
        _write(out / ("r0l.failed.%d.json" % i), dict(doc, valid=False, problem=why))
        raise MonitorError(f"{why}; the attempt is kept as r0l.failed.{i}.json (rule M1: repair the fault, then "
                           "all of R0L's episodes are run again before the run's first rollout)")
    return _write(out / "r0l.json", dict(doc, valid=True))


def write_s2_10(job_dir, out=None, committed_baseline=None) -> dict:
    """``s2_10`` on a job's records: its last check is the final checkpoint's, the others are reported.  With
    ``committed_baseline`` (the committed ``R0L_BASELINE.json``) the record holds ``baseline_equal`` (rule M3)."""
    job_dir = Path(job_dir)
    job = _job_doc(job_dir)
    mons = []
    for entry in job.get("monitor") or []:
        if not entry.get("record"):
            raise MonitorError(f"the check at {entry.get('at_steps')} steps has no record: S2-10 reads records")
        mons.append(s2io.read_json(job_dir / entry["record"]))
    if not mons:
        raise MonitorError(f"{job_dir}: the run has no check")
    r0l = s2io.read_json(job_dir / "monitor" / "r0l.json")
    doc = s2_10(mons[-1], mons[:-1], r0l)
    doc.update(final_is_last_checkpoint=bool(int(mons[-1]["at_steps"]) == int(job.get("num_timesteps", -1))),
               status=job.get("status"), budget=job.get("budget"))
    if committed_baseline is not None:
        doc["baseline_equal"] = dict(baseline_equal(r0l, s2io.read_json(committed_baseline)),
                                     committed=s2io.portable(str(Path(committed_baseline).resolve())),
                                     committed_sha256=s2io.sha256_file(committed_baseline))
    doc.update(provenance=s2io.git_state(own_outputs=[str(job_dir.resolve())]), versions=installed_versions(),
               at=s2io.now_utc())
    return _write(Path(out) if out is not None else job_dir / "S2_10.json", doc)


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _ks(text) -> tuple:
    return tuple(int(x) for x in str(text).split(",") if x.strip() != "") if text else ()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.monitor2", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "baseline"):
        a = sub.add_parser(name, help="the checkpoint's episodes" if name == "run" else "R0L on the same episodes")
        a.add_argument("--job", required=name == "run", default=None,
                       help="the job's directory (<root>/jobs/<job>)" + ("" if name == "run" else
                                                                        "; a baseline without a job needs --out"))
        if name == "run":
            a.add_argument("--steps", type=int, required=True)
        a.add_argument("--body", required=True)
        a.add_argument("--set", required=True)
        a.add_argument("--k", type=int, default=K, help="the servo delay in physics steps (5: 10 ms)")
        a.add_argument("--tiny", action="store_true")
        a.add_argument("--also-k", default="", help="further delays, reported (rule M6), e.g. 0")
        a.add_argument("--sources", default=None, help="the trainer's sources hash")
        a.add_argument("--out", default=None, help="default: <job>/monitor" + ("" if name == "run" else
                                                                                  " (required without --job)"))
    a = sub.add_parser("s2-10", help="S2-10's one record from a job's records")
    a.add_argument("--job", required=True)
    a.add_argument("--committed-baseline", default=None)
    a.add_argument("--out", default=None, help="default: <job>/S2_10.json")
    args = ap.parse_args(argv)
    if args.cmd == "baseline" and args.job is None and args.out is None:  # (the lead's ruling 17 of 17 Nov 2026)
        ap.error("baseline: without --job, --out names the directory of R0L's record")
    s2io.require_one_thread()
    note = steps = None
    try:
        if args.cmd == "s2-10":
            doc = write_s2_10(args.job, args.out, args.committed_baseline)
            print(f"S2-10 at {doc['at_steps']} steps: N {doc['policy']['N']} (R0L {doc['r0l']['N']}), S "
                  f"{doc['policy']['S']:.3f} s (R0L {doc['r0l']['S']:.3f} s): "
                  f"{'a gain' if doc['go'] else 'no gain'}", flush=True)
            return 0
        out = Path(args.out) if args.out else Path(args.job) / "monitor"
        steps = args.steps if args.cmd == "run" else None
        note = refusal_path(out, steps)
        note.unlink(missing_ok=True)  # (a note is its last call's: an earlier refusal of this check is in run.json)
        if args.cmd == "run":
            doc = run(args.job, args.steps, args.body, args.set, out, k=args.k, tiny=args.tiny,
                      also_k=_ks(args.also_k), sources_sha256=args.sources)
            print(f"monitor {doc['row']} at {doc['at_steps']} steps: N {doc['N']}, S {doc['S']:.3f} s "
                  f"({len(doc['episodes'])} episodes, {doc['wall_s']:.0f} s)", flush=True)
        else:
            doc = baseline(args.body, args.set, out, k=args.k, tiny=args.tiny, also_k=_ks(args.also_k),
                           sources_sha256=args.sources, job_dir=args.job)
            print(f"baseline {doc['row']}: N {doc['N']}, S {doc['S']:.3f} s ({len(doc['episodes'])} episodes, "
                  f"{doc['wall_s']:.0f} s)", flush=True)
        return 0
    except MonitorRefusal as exc:  # (no attempt was made: the trainer halts and does not count it, rule M1)
        print(f"monitor2 {args.cmd}: refused to run: {exc}", file=sys.stderr, flush=True)
        if note is not None:
            _note_refusal(note, args.cmd, steps, exc)
        return EXIT_REFUSAL
    except MonitorError as exc:
        print(f"monitor2 {args.cmd}: {exc}", file=sys.stderr, flush=True)
        return 1


__all__ = ["ATTEMPTS", "BASELINE_SCHEMA", "CONDITIONS", "CREDIT_RTOL", "DIGEST_KEYS", "EPISODES_BY", "EXIT_REFUSAL",
           "FIX_TASKS",
           "G8B_AGAIN_AT", "G8B_AT",
           "G8B_CONDITION", "K", "MonitorError", "MonitorRefusal", "OUTCOMES", "REFUSALS", "REFUSAL_SCHEMA",
           "ROW_REFUSALS", "RULES",
           "S1_CONDITIONS", "S1_FIX_TASKS", "S2_10_SCHEMA", "SCHEMA",
           "SEEDS", "STOP_CHECKS", "STOP_FROM", "baseline", "baseline_equal", "baseline_problem",
           "checkpoint_hashes", "conditions_down", "conditions_for", "decide", "digest_compared", "fix_tasks_for",
           "fix_weights", "g8b",
           "gain", "has_clock", "installed_versions", "is_record", "main", "need_up", "no_record", "pairs_for",
           "refusal_path", "row_name", "rules", "run", "s2_10",
           "seeds_for", "summarize", "survival_credit", "write_s2_10"]

if __name__ == "__main__":
    from designlab.stage2 import monitor2 as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
