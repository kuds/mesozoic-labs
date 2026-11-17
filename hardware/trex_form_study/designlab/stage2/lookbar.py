"""Track B's rows for the bar runner: the look reading beside any row (``LookRow``), the row ``R0L``, in-process
episodes for the monitor, and the companion records that hold reading 1 (round C design D15, D17, D18; sections
2.2, 3.5; Track B rules U2-U6, M1; the lead's rulings of 16 Nov 2026).  ``bar.py`` is not edited: everything here
goes through what it offers to any row (``bar.Delayed``, ``bar.evaluate(rows=<built rows>)``, ``bar.run_episode``,
``bar.counts``).

* ``LookRow`` wraps a row.  For a bar walk or turn item it lends the built condition a recorder with 50 Hz poses
  for the one episode (an instance attribute over ``BarBuilt.recorder``, taken back in a ``finally``; a recorder
  someone else lent for that episode, a gate's digest keeper, is kept and only made to hold poses), runs the inner
  row unchanged and returns its ``(built, outcome, recorder)`` as it is, so the bar judges and traces what it would
  have without the wrapper.  From the recorder it then reads: the look reading (``lookread``: the window, the four
  numbers, the reference's own by kinematic playback over ``lookread.window_ref``, the terms), the clock block,
  for a stop item the term ``on_command_at_stop``, the short air runs of the window, and the final state's
  digest.  That is the episode's *block*: kept in ``last`` and, with a sidecar root, written to
  ``<root>/<row>/<condition key>/<seed>.json`` before ``episode`` returns (so before the bar writes its partial).
  The reading never touches the plant, and nothing of it is raised: a fault is recorded in the block.
* ``R0L`` is the panels' R0 row inside ``LookRow`` (registered with ``panels.register_rows`` at import).
* ``rows`` builds the bar's rows by name, as ``bar.rows`` does, every inner row a ``LookRow``.
* ``episodes`` runs ``(condition key, seed)`` pairs in this process (the monitor's entry) and returns one
  *companion episode* each.
* ``collect`` joins each finished bar record of a ``LookRow`` row with its sidecars into a companion record
  ``<out>/<set>/<body>/<block>/look/<row>/<behaviour>/<condition>.json`` (schema ``s2-look-1``); ``run`` is
  ``bar.evaluate`` with these rows, then ``collect``.  A record whose sidecars are missing or are not its episodes'
  is refused one: a companion of another bar record left at its path is moved aside
  (``<companion>.superseded/<12 hex>.json``, the bar's own convention), and the summary (``look/summary.json``)
  lists the record under ``refused_records`` with what became of its companion.

**Outcome of an episode** (rules U6, M1; ``companion_episode``).  *read*: the bar judged it and it did not fall.
*fell*.  *unread*: the bar could not read it (``bar.Unreadable``), it did not fall, and the row is a policy row (its
inner row has ``provenance()``): not succeeding in any reading, with the reason ``clock_behind`` when its mean
clock factor is below 1 and ``unreadable`` otherwise.  *error*: anything else that kept the bar from judging it
(an infrastructure fault, or an unreadable episode of a row without a policy, for which ``STAGE2_75_PLAN.md`` C8
stands), and, on a look set, a walk or turn item whose look reading cannot be given (rule U5: a reference that
cannot be played, or whose own window holds too few steps, is a fault of the definition, never "no look reading").
An episode during which MuJoCo warned is a fall at its start (``tune.warned_fall``; the lead's ruling 9 of 17 Nov
2026): *fell*, with ``warned`` true, ``fall_time_s`` 0 and ``survival_s`` 0 in its companion episode, so no
survival is credited for it (the time simulated is ``simulated_s``).

**Readings** (rule U4; the lead's rulings of 16 Nov 2026).  Reading 1 counts on the look sets ``LOOK_SETS``.  There
a walk or turn episode succeeds with the look when the bar's ``succeeded`` holds and all four terms do; a nudge or
stop episode's reading 1 is its reading 2, labelled; a stop episode that fails ``on_command_at_stop`` succeeds in no
reading of the companion.  On any other set the look numbers and the term are recorded and decide nothing:
``succeeded_look`` and ``counts_look`` are null and the companion's counts are the bar's own, episode for episode
(also for an episode the bar could not read: the bar's count stands there).

**A block** (schema ``s2-look-ep-1``; the sidecar file): ``row`` (the bar row's name), ``look_row``, ``condition``,
``seed``, ``set``, ``judge``, ``steps``, ``fell``, ``survival_s``, ``planned_s``, ``fall_cause``, ``fall_time_s``,
``mujoco_warnings``, ``clock {mean_factor, min_factor, t_ref_reached, t_ref_planned, ...}``, then the look reading:
``read`` (whether there is one), ``reason`` (why an item is not read: a nudge, a stop or a stock item, a fall, a
window not reached), ``fault`` (what went wrong in the reading itself, as ``<what>: <Type>: <message>`` with no
absolute path of this machine), ``window``, ``window_ref``, ``own``, ``reference``, ``terms``, ``not_read``,
``too_few_steps``, ``look_ok``, ``compare`` (the ratios and differences), ``air_runs``; ``on_command_at_stop`` (a stop
item); ``policy_check`` (a policy row's ``provenance()``); ``state_digest``; ``row_sha256``.  **A companion episode**:
``condition``, ``seed``, ``outcome``, ``reason``, ``fell``, ``warned``, ``fall_time_s``, ``survival_s`` (0 for an
episode during which MuJoCo warned: ``RULES["outcome"]``), ``simulated_s``, ``planned_s``, the four readings'
``succeeded...``, ``failed_terms``, ``look`` (the block's reading, None where there is none), ``clock``,
``on_command_at_stop``, ``error``, ``state_digest``, ``policy_check`` and the bar's own verdicts under ``bar``.

No episode here is a record by itself: ``run`` writes the bar's records and their companions; ``episodes`` writes
nothing but the traces it is asked for.

**What may run on a look set** (``RULES["look_sets"]``; the lead's rulings 16 and 31 of 17 Nov 2026; rule X1).  The
bar's runner keeps its own rule of what may run where (``bar.RULES["checks"]``): on a reference set that is none of
the 75 cm round's three it lets only rows of the controller ``R0`` run, so ``R0L``, ``R5``, ``R6``, ``R5G`` and
``R2L`` rows are refused there, and ``bar.py`` is not edited.  ``look_set_rule(set, body, rows, root)`` is a context
inside which a bar evaluation of that set, body and set root, the set one of ``LOOK_SETS`` or a version of one, lets
the rows of ``LOOK_SET_ROWS`` (``R0L``, ``R5/<seed>``, ``R6/<seed>``, ``R5G/<seed>``, ``R2L-<set>/<seed>``, each with
or without a delay) run, and only once the set's G9-ref has passed for the body on its latest freeze
(``require_g9_ref``, which is ``gates2.require_g9_ref``); every other row is refused on a look set, inside the
context and in ``episodes``.  It stands in front of ``bar._check_rule`` for a look set's evaluations only: every
other set, body or root, and every call outside it, is the bar's own check unchanged, and the bar's check is put
back when the context ends, whatever happens, where the context's own is the one in place (else ``LookBarError``).
``run`` evaluates inside it; ``episodes`` (which does not go through the runner) asks it before it reads a look set
and runs inside it.  The refit tool (``r2lfit``) is to use the same context.  A version of a look set is a look set
here, in ``episodes`` and in ``collect``.  The rows it lets in carry the G9-ref record's citation (path and sha256)
into every episode's block, and so into the companion record (``g9_ref``).

**Join rule J1's leave point** (``RULES["leave_point"]``; the lead's ruling 21 of 17 Nov 2026).  On a look set a
walk's file is a window of K strides and its turns and two-step stop were built to leave it ``a_phase`` strides in
(``look.window_a``).  The bar counts its lead before them in whole windows of the file, and ``bar.py`` is not
edited, so while the rule lets an evaluation of a look set run the context also stands in front of the bar
planner's ``BarPlanner._resolve``: such a turn or stop is planned with the walk ending at the leave point
(``leave_resolve``: ``m + a_windows`` windows, the bar's own arithmetic otherwise) after this module's phase check
(``leave_of``, the env's ``r5lib.leave_point``); every other planner and item is the bar's own, and the bar's piece is
put back when the context ends.  A spawned worker of the evaluation, which rebuilds the rows from their specs, holds
the same placement for its life (``hold_for_life``).  Outside the context the bar's own placement stands.

``look_set_placement(set, body, root)`` is that placement alone (the lead's ruling 37 (b) of 17 Nov 2026): no row
rule and no G9-ref requirement, because G9-ref itself runs before G9-ref has passed.  ``look_set_rule`` places
through it.  On a look set every other reader that plays a look set's turns or stop runs inside it: G9-ref part (a)
(``gates2._refcheck_record``, its ``bar.refcheck``), G8a (``gates2.g8a``), ``s2rows.equivalence_check`` and R2L's
fit and selection (``r2lfit``, in process and in every pool worker).  On entry each context lets the process's cached
bar context of the set forget what it built (``_release_cached``: a turn built outside it is not replayed inside it);
at its end the look sets' rule gives every row it stamped back the citation and placement it held before.

    python -m designlab.stage2.lookbar {run|collect} --set S --body B --rows R0L@10ms,R5/42@10ms --out DIR
        [--conditions walk/v0.4,...] [--procs 2] [--block seeds3042] [--episodes 20] [--task 3|4] [--unit NAME]
        [--cost FILE] [--no-traces] [--no-traces-for R0] [--supersede --supersede-reason TEXT] [--dev-episodes FILE]
        [--dev-reason TEXT] [--set-root DIR] [--stage1-run DIR] [--policies FILE] [--row-module M]

Exit codes: 0; 1 a refusal (the bar's, or a sidecar that is missing or is not its episode's); 3 the budget cut the
unit; 4 a record of the session still holds an episode that failed to run.

This file and ``lookread.py`` are in the sources of every Track B row (``LookRow.sources``).
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
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from designlab.stage1 import control, env, metrics, panels, tune  # noqa: E402
from designlab.stage2 import bar, lookread, readings, s2io, setio  # noqa: E402

#: The sets on which reading 1 counts (rule U4).
LOOK_SETS = ("V6", "V6noF5")
SIDECAR_SCHEMA = "s2-look-ep-1"
COMPANION_SCHEMA = "s2-look-1"
SUMMARY_SCHEMA = "s2-look-summary-1"
RATE = panels.RATE
#: The bar's judges whose items have a look reading (rule U2), and the stop item's.
READ_JUDGES = ("bar_walk", "bar_turn")
STOP_JUDGE = "bar_stop"
OUTCOMES = ("read", "fell", "unread", "error")
NOT_RUN = "not_run"
#: Rule U4's term of a stop item: the CoM's stride mean before the command at least this share of the commanded
#: speed (``STAGE2_75_PLAN.md`` C6's lower band: ``panels.WALK_BACK_FRAC``).
ON_COMMAND_FRAC = panels.WALK_BACK_FRAC
ON_COMMAND_TERM = "on_command_at_stop"
#: An episode of a bar row starts with the track's clock at 0 (``control.Tracker.reset``).
T_REF_START = 0.0
#: A mean clock factor below ``1 - CLOCK_TOL`` is "below 1" (the factor of a row that keeps the commanded clock is
#: 1 to rounding).
CLOCK_TOL = 1e-9
#: How ``tune.warned_fall`` begins the fall cause of an episode during which MuJoCo warned (``is_warned``).
WARNED_CAUSE = "MuJoCo warning "
#: What a block holds of a read episode's look reading: the companion episode's ``look``.
LOOK_KEYS = ("window", "window_ref", "own", "reference", "terms", "not_read", "too_few_steps", "look_ok", "compare",
             "air_runs")
#: What a block keeps of ``lookread.compare``'s value beside the terms: the ratios and differences, the robot's own
#: tail amplitude with its side of its limit, the counts of steps and lift-offs, the reference's values with their
#: sides of the threshold, and the bands.
COMPARE_KEYS = ("bob_ratio", "head_world_pitch_rms_deg", "tail_phase_diff_cycles", "tail_amp_rad", "tail_amp_side",
                "tail_amp_min_rad", "tail_amp_ratio", "toe_off_ratio", "steps", "lift_offs", "reference", "bands")
#: The keys ``bar.Delayed.describe`` adds to its inner row's description (a test holds the two together).
DELAYED_KEYS = ("delayed_row", "latency_steps", "latency_ms", "latency_rule", "servo_delay_rule")
#: The modules whose import registers a controller's rows (``panels.register_rows``): looked up by the controller's
#: name or its beginning, and imported by ``rows`` when the controller is not registered yet.
ROW_MODULES = (("R5G", "designlab.stage2.s2rows"), ("R5", "designlab.stage2.s2rows"),
               ("R6", "designlab.stage2.s2rows"), ("R2L", "designlab.stage2.r2lfit"))
READING_1 = dict(
    walk_turn="reading 1: the bar's succeeded and the four look terms",
    other="reading 1 is reading 2 on this item (it is defined for walks and turns)",
    not_counted="reading 1 does not count on this set: the look numbers are recorded and decide nothing")

RULES = dict(
    wrapper=(
        "the row is its inner row with the look reading read beside it.  For a bar walk or turn item the built "
        "condition lends the episode a recorder with 50 Hz poses (metrics.Recorder(legs_every=tune.LEGS_EVERY), an "
        "instance attribute for the one episode, taken back whatever happens); the inner row runs unchanged and its "
        "(built condition, outcome, recorder) is returned as it is.  The reading is computed from the recorder "
        "afterwards, reads the plant's final state and changes nothing; a fault in it is recorded in the episode's "
        "block and never raised"),
    reference_use=(  # (not "reference": that key is lookread.RULES' own text of rule U5; ``rule_texts``)
        "STAGE2_TRACKB_PLAN rule U5: the reference's values are lookread.playback of the built track over "
        "lookread.window_ref of the built condition (the reference times rule U2's window covers at the nominal "
        "commanded clock), the same for every episode of the condition.  A walk or turn item whose track cannot be "
        "played, or whose playback holds too few steps or lacks a value, is a fault of the definition, never an "
        "item without a look reading: on a look set such an episode is an error"),
    readings=(
        "STAGE2_TRACKB_PLAN rule U4 with the lead's rulings of 16 Nov 2026.  Reading 1 counts on the sets V6 and "
        "V6noF5: a walk or turn episode succeeds with the look when the bar's succeeded holds and all four look "
        "terms do; for a nudge or stop item reading 1 is reading 2, labelled; printed beside, reading 3 with the "
        "four terms kept.  On any other set the look numbers are recorded, reading 1 is null and the companion's "
        "counts are the bar's own"),
    on_command_at_stop=(
        "STAGE2_TRACKB_PLAN rule U4: a stop item of a Track B row has the term on_command_at_stop: the CoM's mean "
        "forward speed over the stride before t_c at least 0.9 x the commanded speed.  "
        + lookread.MODULE_READING +
        "t_c is the simulated time at which the reference clock reached the command (the bar's own); the stride is "
        "the walk's commanded stride on the simulated clock; the mean speed is the CoM's displacement from t_c less "
        "one stride to t_c along the commanded heading (the home heading), over the stride, both positions "
        "interpolated between update rows (STAGE2_75_PLAN.md C6 reads stride means so); the commanded speed is "
        "the walk's target speed x the commanded clock rate, as the bar's stop reading takes it; compared without "
        "slack.  A trace that does not reach t_c, or begins less than one stride before it, fails the term.  On a "
        "look set an episode that fails it succeeds in no reading of the companion; on any other set it is "
        "recorded and decides nothing"),
    clock=(
        "the clock factor is the slope of the recorder's reference time against its simulated time at the update "
        "rows over the commanded clock rate (mean: over the whole trace; min: over one update interval); "
        "t_ref_reached is the reference clock at the episode's end (control.run's reference_time_s; the last "
        "update row's where an outcome holds none); t_ref_planned is the commanded clock rate x the planned seconds "
        "from the track's start, which is 0 for a bar row.  A mean factor below 1 - 1e-9 is below 1"),
    air_runs=(
        "contact is read literally (rule U1: the 2 % load rule with the 20 ms debounce), so any air run longer than "
        "the debounce is a lift-off and a touchdown of the look reading.  Recorded beside it, deciding nothing: how "
        "many complete air runs lie wholly inside the window and how many of them are shorter than the bar's "
        "air_gap_max_s of the condition (half the track's shortest planned swing on the simulated clock)"),
    outcome=(
        "rules U6 and M1: an episode ends read; fell; unread (the bar could not read it, bar.Unreadable, it did not "
        "fall and the row is a policy row: not succeeding in any reading, with the reason clock_behind when its "
        "mean clock factor is below 1, else unreadable, and the reference time reached); or error (an "
        "infrastructure fault; an unreadable episode of a row without a policy, for which STAGE2_75_PLAN.md C8 "
        "stands; on a look set, a walk or turn item whose look reading cannot be given).  An episode during which "
        "MuJoCo warned is a fall at its start, as tune.warned_fall has it (the lead's ruling 9 of 17 Nov 2026): its "
        "companion episode holds warned true, fall_time_s 0 and survival_s 0, so no survival is credited for it; the "
        "time simulated is kept as simulated_s, and the sidecar and the bar record keep theirs"),
    counts=(
        "rule U6: the readings' counts by the bar's own arithmetic (bar.counts) on the companion's episodes: read "
        "and unread episodes survived, an unread episode succeeded in none; a fell episode is a fall; an error "
        "episode is counted as the bar counts an episode that failed to run, and leaves the record unjudged.  On a "
        "set that is not a look set the counts are bar.counts on the bar record's own episodes and equal the bar "
        "record's.  A turn frozen with a1_over reads \"not read: a1_over\" in readings 1 and 2, as the bar's "
        "record does in reading 2"),
    sidecar=(
        "one file per episode, written by the row before it returns: steps, fell and survival_s must equal the bar "
        "record's episode and row_sha256 the hash of the record's controller less the delay wrapper's keys, or the "
        "record gets no companion (run that condition again so that record and sidecars are of one run).  A "
        "companion found at the refused record's path is moved aside to <companion>.superseded/<first 12 hex of the "
        "file's sha256>.json unless it names the bar record as it is (bar_record.sha256): that one was made of this "
        "bar record while its sidecars were there, and stays.  The summary lists every refused record with what "
        "became of its companion"),
    look_sets=(
        "the lead's rulings 16 and 31 of 17 Nov 2026, with STAGE2_TRACKB_PLAN rule X1: the bar's own set check "
        "(bar._check_rule, bar.RULES['checks']) lets only R0 rows run on a set that is none of the 75 cm round's "
        "three, and bar.py is not edited (its hash is in every bar record's key).  lookbar.look_set_rule is a "
        "context, entered by lookbar run and by lookbar.episodes, inside which a bar evaluation (bar.evaluate) of "
        "the context's set, body and set root, where the set is a look set (V6, V6noF5) or a version of one (the "
        "part of its name before the first dot; a version is a look set in the context, in episodes and in "
        "collect), lets its rows run when every one of them is among the rows the context was entered with and "
        "every one of those is R0L, R5/<seed>, R6/<seed>, R5G/<seed> or R2L-<set>/<seed> (the fit of that very "
        "set), each with or without a delay @<ms>ms, and only once that set's G9-ref has passed: the record "
        "stage2/tb/gates/<set>/<body>/g9_ref.json, of schema s2-gate-g9-ref-1 with its own sha256 intact, of that "
        "set and body, with pass_ true, made on the set's latest freeze (its set_freeze an object whose "
        "manifest_sha256 is that freeze's), read when the evaluation is checked by gates2.require_g9_ref itself; "
        "without it the evaluation is refused and nothing runs.  Every row that runs then carries that record's "
        "citation (path and sha256) into each episode's block and its companion record (g9_ref), and the set's "
        "turns and stop leave the walk at join rule J1's leave point (leave_point).  On a look set every row "
        "outside those families (R0@<ms>ms included) is refused inside the context and in lookbar.episodes, "
        "whoever calls the check.  Every other evaluation inside the context (another set, body or set root, or "
        "these rows beside one the context was not entered with) goes through the bar's own check unchanged, and "
        "so does every evaluation outside it (python -m designlab.stage2.bar run on a look set stays the bar's "
        "own: rule X1 forbids it and no step of the chain calls it).  lookbar.episodes, which does not go through "
        "the bar's runner, checks a look set's rows in the same way before it reads the set and runs inside the "
        "context; it checks nothing on any other set.  The context enforces X1's first clause only (what runs "
        "between G9-ref and S2-4's start stays with the chain).  The bar's check is put back when the context ends, "
        "also after an exception, where the context's own check is the one in place; otherwise LookBarError and "
        "nothing is put back"),
    leave_point=(
        "join rule J1 (STAGE2_TRACKB_PLAN.md section L, 'Windows.' and J1; the lead's ruling 21 of 17 Nov 2026): "
        "on a look set the bar counts its lead before a turn or the two-step stop (bar.TURN_LEAD_STRIDES, "
        "bar.STOP_LEAD_STRIDES: 2 strides of v0.4 before a left turn, 2.5 before a right turn and the stop) in "
        "whole windows of the walk's file, which on a look set are K strides each, so the move would start at a "
        "frame it was not built from.  Inside lookbar.look_set_rule, where the rule lets the rows run, a bar "
        "evaluation of that set places such a move whose transitions.json entry carries look.window_a and an "
        "a_phase where it was built: the walk plays m + a_windows windows of its file (a_windows = a_time_s / "
        "window_s, the move's own leave time; m whole windows, m x K + a_phase = the lead in strides) and the move "
        "starts at its first frame; every other item and set is the bar's own plan.  This module's phase check "
        "counts the lead's strides inside the window (a left command at a whole stride, a right one at a half) by "
        "the env's reading (r5lib.leave_point, r5lib.LEAVE_RULE), because bar.check_phase reads a lead modulo one "
        "stride.  The bar's planner piece BarPlanner._resolve is stood in front of while the context is open, as "
        "_check_rule is, and put back at its end (where it is the context's own); a spawned worker of the "
        "evaluation, which rebuilds its rows from their specs, holds the same placement for its life.  The "
        "record's plan, seconds and command (command.leave) are those of the placed move.  Outside the context "
        "the bar's own placement in whole windows stands.  lookbar.look_set_placement is the same placement "
        "without the row rule and without the G9-ref requirement (the lead's ruling 37 (b) of 17 Nov 2026), and "
        "look_set_rule places through it; on a look set G9-ref part (a) (its bar.refcheck), G8a, "
        "s2rows.equivalence_check and R2L's fit and selection (in process and in every pool worker) run inside it, "
        "and off the look sets it does nothing.  On entry either context lets the process's cached bar context of "
        "the set forget its built conditions, so that nothing built outside is played inside; at its end the look "
        "sets' rule gives every row it stamped back its earlier citation and placement.  The phase check also "
        "reads the walk's file: a_time_s must lie in the frame interval over which the file's own phase array "
        "passes a_phase (r5lib.leave_point; ruling 37 (c))"),
)


class LookBarError(s2io.S2Error):
    """A companion record cannot be made (a sidecar is missing or is not its episode's), or ``episodes`` was given
    rows it cannot read."""


def rule_texts() -> dict:
    """The look reading's rule texts (``lookread.RULES``) and this module's (``RULES``) as one table: what a row's
    description holds under ``look_reading.rule`` and a companion record under ``rules``.  The two tables must
    share no key, or one text of the two would be left out of every description and record: ``LookBarError``."""
    shared = sorted(set(lookread.RULES) & set(RULES))
    if shared:
        raise LookBarError(f"lookread.RULES and lookbar.RULES both hold the key(s) {', '.join(shared)}: one table of "
                           "both would drop a rule text")
    return dict(lookread.RULES, **RULES)


#: An absolute path in an exception's message (not the "/" of "m/s" or of a condition key "walk/v0.4").
_ABS_PATH = re.compile(r"(?<![\w.~-])/[^\s'\"`,;:()\[\]{}<>]+")


def _no_paths(text: str, roots=()) -> str:
    """``text`` with no absolute path of this machine (the lead's ruling 8 of 17 Nov 2026): a path under one of
    ``roots`` (an output root: its summary's own paths are relative to it) or under the study directory made
    relative to it, any other one cut to ``.../<its last three parts>``, and never to its first part (the lead's
    ruling 12 of 17 Nov 2026: ``/home/user/a`` is ``.../user/a``, a one-part ``/content`` is ``...``)."""
    bases = [str(Path(r)) + os.sep for r in roots if r is not None and Path(r).is_absolute()]
    bases.append(str(s2io.STUDY) + os.sep)

    def cut(m) -> str:
        p = m.group(0)
        for base in bases:
            if p.startswith(base) and len(p) > len(base):
                return p[len(base):]
        parts = [x for x in p.split("/") if x]
        keep = parts[1:][-3:]
        return ".../" + "/".join(keep) if keep else "..."

    return _ABS_PATH.sub(cut, str(text))


def _fault(what: str, exc: BaseException) -> str:
    """A fault as a block records it: ``<what>: <Type>: <message>``, no traceback and no absolute path of this
    machine (``_no_paths``), so a sidecar and a companion record read the same wherever they were made."""
    return _no_paths(f"{what}: {type(exc).__name__}: {exc}")


# ------------------------------------------------------------------------------------------------
# What a block holds beside the look reading
# ------------------------------------------------------------------------------------------------
def clock_block(out: dict, rec, planned_s: float) -> dict:
    """``RULES["clock"]`` of one episode: ``mean_factor``, ``min_factor`` (None with fewer than two update rows),
    ``t_ref_reached``, ``t_ref_planned``, and beside them the track's start, the last update row's reference time
    and the commanded rate."""
    rows = int(getattr(rec, "rows", 0) or 0)
    mean = low = last = None
    if rows >= 1:
        t, tr = np.asarray(rec.t, float), np.asarray(rec.col("t_ref"), float)
        last = float(tr[-1])
        if rows >= 2 and t[-1] > t[0]:
            mean = float((tr[-1] - tr[0]) / (t[-1] - t[0]) / RATE)
            low = float(np.min(np.diff(tr) / np.diff(t)) / RATE)
    reached = out.get("reference_time_s")
    return dict(mean_factor=mean, min_factor=low, t_ref_reached=last if reached is None else float(reached),
                t_ref_planned=T_REF_START + RATE * float(planned_s), t_ref_start=T_REF_START, t_ref_last_update=last,
                rate=RATE, updates=rows)


def on_command_at_stop(b, rec) -> dict:
    """``RULES["on_command_at_stop"]`` of a stop item's episode: ``term``, ``speed_m_s`` (None where it cannot be
    read, with ``reason``), ``needed_m_s`` and what they were read with."""
    cmd, stride, heading = float(b.speed_cmd), float(b.stride_s), float(b.home_yaw)
    out = dict(term=False, speed_m_s=None, needed_m_s=ON_COMMAND_FRAC * cmd, command_speed_m_s=cmd,
               frac=ON_COMMAND_FRAC, stride_s=stride, heading_rad=heading, command_s=None, reason=None)
    if not getattr(rec, "rows", 0) or getattr(b, "command", None) is None:
        return dict(out, reason="no trace, or the item has no command")
    t_c = rec.sim_time(float(b.command["ref_s"]))
    if t_c is None:
        return dict(out, reason="the reference clock never reached the command")
    t = np.asarray(rec.t, float)
    ta = float(t_c) - stride
    if ta < t[0] - readings.EPS_S:
        return dict(out, command_s=float(t_c), reason="the trace begins less than one stride before the command")
    com = rec.U[:, [metrics.C["com_x"], metrics.C["com_y"]]]
    h = np.array([np.cos(heading), np.sin(heading)])
    c0 = np.array([np.interp(ta, t, com[:, 0]), np.interp(ta, t, com[:, 1])])
    c1 = np.array([np.interp(t_c, t, com[:, 0]), np.interp(t_c, t, com[:, 1])])
    speed = float((c1 - c0) @ h / stride)
    return dict(out, command_s=float(t_c), speed_m_s=speed, term=bool(speed >= out["needed_m_s"]))


def air_runs(ser: dict, ta: float, tb: float, air_gap_max_s: float) -> dict:
    """``RULES["air_runs"]``: the complete air runs of both feet that lie wholly inside ``[ta, tb]`` of a look
    series (lift-off and touchdown inside it), and those shorter than ``air_gap_max_s``."""
    dt, eps = float(ser["dt"]), readings.EPS_S
    t0 = float(ser["t50"][0]) if len(ser["t50"]) else 0.0
    inside, short = 0, []
    for i in (0, 1):
        for a, b in readings.air_runs(ser["contact"][:, i], dt):
            lo, td = t0 + a * dt, t0 + b * dt
            if lo >= ta - eps and td <= tb + eps:
                inside += 1
                if td - lo < float(air_gap_max_s) - eps:
                    short.append([i, float(lo), float(td - lo)])
    return dict(in_window=inside, shorter_than_air_gap_max=len(short), air_gap_max_s=float(air_gap_max_s),
                short=sorted(short, key=lambda x: x[1]), short_cols=["foot", "t_lo_s", "duration_s"])


def state_digest(p) -> str:
    """The sha256 of the plant's ``qpos``, ``qvel``, ``ctrl`` and ``sensordata`` as they are (read, not changed)."""
    return hashlib.sha256(env._state_digest(p.d)).hexdigest()


def look_of(block: dict | None) -> dict | None:
    """The look reading of a block as a companion episode holds it, or None where it was not read."""
    if not block or not block.get("read"):
        return None
    return {k: block.get(k) for k in LOOK_KEYS}


def _recorder_of(rec):
    """The ``metrics.Recorder`` under a chain of hook wrappers (each holds the next as ``rec``, as
    ``panels._Acting`` and ``g8a._Digests`` do)."""
    seen = 0
    while not isinstance(rec, metrics.Recorder) and "rec" in getattr(rec, "__dict__", {}) and seen < 8:
        rec, seen = rec.__dict__["rec"], seen + 1
    return rec if isinstance(rec, metrics.Recorder) else None


def _lender(outer):
    """What the built condition hands out for the episode: a recorder with 50 Hz poses; where a recorder was lent
    from outside for the same episode (``outer``), that one, made to hold the poses."""
    if outer is None:
        return lambda: metrics.Recorder(legs_every=tune.LEGS_EVERY)

    def make():
        rec = outer()
        inner = _recorder_of(rec)
        if inner is not None and not inner.legs_every:
            inner.legs_every = tune.LEGS_EVERY
        return rec

    return make


# ------------------------------------------------------------------------------------------------
# The row wrapper
# ------------------------------------------------------------------------------------------------
class LookRow(panels.Row):
    """Any row with the look reading read beside it (module docstring; ``RULES["wrapper"]``).  ``runs``, ``gated``
    and ``hard_only`` are the inner row's, and so is the name unless one is given.  ``sidecar``: the root under which
    every episode's block is written (None: none is); ``sidecar_row``: the row name it is filed under (``rows``
    gives the bar row's, ``R0L@10ms``).  ``last``: the last episode's block.  On a look set the look sets' rule gives
    the row ``g9_ref`` (the citation, path and sha256, of the G9-ref record under which its episodes run: every
    block holds it) and ``placement`` (the set whose turns and stop leave the walk at join rule J1's leave point);
    both travel in the row's spec to a worker (``LookSetRule.stamp``, ``hold_for_life``)."""

    def __init__(self, inner: panels.Row, sidecar: str | None = None, name: str | None = None):
        if isinstance(inner, LookRow):
            raise ValueError(f"row {inner.name!r} is a LookRow already")
        self.inner = inner
        self.name = name or inner.name
        self.hard_only = bool(inner.hard_only)
        self.sidecar = None if sidecar is None else str(sidecar)
        self.sidecar_row = self.name
        self.last: dict | None = None
        self.g9_ref: dict | None = None
        self.placement: dict | None = None
        self._refs: dict = {}  # condition -> (its built condition, the reference's entry): per process

    def runs(self, cond) -> bool:
        return self.inner.runs(cond)

    def gated(self, cond) -> bool:
        return self.inner.gated(cond)

    def describe(self) -> dict:
        return dict(self.inner.describe(),
                    look_reading=dict(bands=dict(lookread.BANDS), rule=rule_texts(), look_sets=list(LOOK_SETS)))

    def sources(self) -> dict:
        """The inner row's sources with this module and ``lookread.py`` by sha256."""
        return {**self.inner.sources(), **panels.module_sha256(LookRow), **lookread.sources()}

    def spec(self) -> dict:
        out = dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name, sidecar=self.sidecar,
                   sidecar_row=self.sidecar_row, inner=self.inner.spec())
        for key in ("g9_ref", "placement"):  # (only a look set's rows carry them)
            if getattr(self, key) is not None:
                out[key] = dict(getattr(self, key))
        return out

    @classmethod
    def from_spec(cls, s: dict) -> "LookRow":
        """The row a spec describes; a spec that carries a placement makes this process hold it for its life
        (``hold_for_life``: a spawned bar worker plays the look set's turns and stop as the runner planned them)
        unless a context of this process holds it already."""
        row = cls(panels.row_from_spec(s["inner"]), s.get("sidecar"), s["name"])
        row.sidecar_row = s.get("sidecar_row") or row.name
        row.g9_ref = dict(s["g9_ref"]) if isinstance(s.get("g9_ref"), dict) else None
        row.placement = dict(s["placement"]) if isinstance(s.get("placement"), dict) else None
        if row.placement is not None:
            hold_for_life(row.placement)
        return row

    def row_sha256(self) -> str:
        """The hash of this row's description: what ``collect`` finds again in a bar record's ``controller``."""
        return s2io.sha256_json(s2io.portable(self.describe()))

    def provenance(self) -> dict | None:
        """The inner row's ``provenance()`` (a policy row's version sets and fingerprint check), or None."""
        fn = getattr(self.inner, "provenance", None)
        return fn() if callable(fn) else None

    # ------------------------------------------------------------------ the episode
    def episode(self, ctx, cond, seed: int, relax=()):
        b = ctx.built(cond)
        lend = bar.is_bar(cond) and cond.judge in READ_JUDGES
        outer = vars(b).get("recorder")
        if lend:
            b.recorder = _lender(outer)
        self.last = None
        try:
            built, out, rec = self.inner.episode(ctx, cond, seed, relax)
        finally:
            if lend:
                if outer is None:
                    vars(b).pop("recorder", None)  # the class's method is back
                else:
                    b.recorder = outer
        block = self._block(built, out, rec, cond, int(seed), tuple(relax))
        self.last = block
        if self.sidecar is not None and not relax:  # (an attribution's lifted run is not the record's episode)
            s2io.write_json(sidecar_path(self.sidecar, self.sidecar_row, cond.key, seed), block)
        return built, out, rec

    def _reference(self, b, cond) -> dict:
        """The reference's reading of a walk or turn item (``RULES["reference_use"]``), once per built condition:
        ``{window_ref, reading, fault}``."""
        kept = self._refs.get(cond)
        if kept is not None and kept[0] is b:
            return kept[1]
        entry = dict(window_ref=None, reading=None, fault=None)
        try:
            wr = lookread.window_ref(b)
            if wr is not None:
                entry["window_ref"] = [float(wr[0]), float(wr[1])]
                entry["reading"] = lookread.playback(b.p, b.track, b.track.run_dir, *wr)
        except Exception as exc:  # noqa: BLE001 - a LookError, or whatever a faulty definition raises: recorded
            entry["fault"] = f"{type(exc).__name__}: {exc}"  # (its paths are cut where the reading records it)
        self._refs[cond] = (b, entry)
        return entry

    def _block(self, b, out: dict, rec, cond, seed: int, relax: tuple) -> dict:
        """The episode's block (the sidecar's document; module docstring).  Never raises for the reading's sake.
        What is read is the ``metrics.Recorder`` itself, also where the row returned it inside a hook wrapper (a
        gate's digest keeper has attributes of its own under a recorder's names)."""
        rec = _recorder_of(rec) or rec
        is_bar = bar.is_bar(cond)
        judge = str(cond.judge)
        fell = bool(out["fell"])
        fall_time = out.get("fall_time_s")
        block = dict(
            schema=SIDECAR_SCHEMA, row=self.sidecar_row, look_row=self.name, condition=cond.key, seed=int(seed),
            set=cond.set if is_bar else bar.STOCK_SET, judge=judge, relaxed=list(relax),
            steps=int(getattr(rec, "steps", 0) or 0), fell=fell, survival_s=float(out["seconds"]),
            planned_s=float(b.seconds), fall_cause=out.get("fall_cause"),
            fall_time_s=None if fall_time is None else float(fall_time),
            mujoco_warnings=out.get("mujoco_warnings") or None, clock=None, read=False, reason=None,
            fault=None, window=None, window_ref=None, own=None, reference=None, terms=None, not_read=None,
            too_few_steps=None, look_ok=None, compare=None, air_runs=None, on_command_at_stop=None,
            policy_check=None, state_digest=None, row_sha256=self.row_sha256(), at=s2io.now_utc())
        if self.g9_ref is not None:  # (a look set's rows: the G9-ref record its episode ran under, RULES["look_sets"])
            block["g9_ref"] = dict(self.g9_ref)
        faults = []
        for key, fn in (("clock", lambda: clock_block(out, rec, b.seconds)),
                        ("state_digest", lambda: state_digest(b.p)), ("policy_check", self.provenance)):
            try:
                block[key] = fn()
            except Exception as exc:  # noqa: BLE001 - recorded, never raised (RULES["wrapper"])
                faults.append(_fault(key, exc))
        if is_bar and judge == STOP_JUDGE:
            try:
                block["on_command_at_stop"] = on_command_at_stop(b, rec)
            except Exception as exc:  # noqa: BLE001
                faults.append(_fault(ON_COMMAND_TERM, exc))
        if not is_bar:
            block["reason"] = "a stock condition has no look reading"
        elif judge not in READ_JUDGES:
            block["reason"] = "nudge and stop items are not read (rule U2)"
        else:
            try:
                self._read(block, b, out, rec, cond)
            except Exception as exc:  # noqa: BLE001 - in the form of the other faults: no traceback and no path of
                faults.append(_fault("look reading", exc))  # this machine in a sidecar or a record (``_fault``)
                block.update(read=False, look_ok=None, terms=None)
        if faults:
            block["fault"] = "; ".join(faults)
        return block

    def _read(self, block: dict, b, out: dict, rec, cond) -> None:
        """The look reading of a walk or turn item into ``block`` (rules U1-U3, U5)."""
        ref = self._reference(b, cond)
        block.update(window_ref=ref["window_ref"], reference=ref["reading"])
        if ref["fault"] is not None:
            raise lookread.LookError(f"the reference of {cond.key}: {ref['fault']}")
        if bool(out["fell"]) and cond.judge == "bar_walk":
            block["reason"] = "fell: a walk item that fell has no window"
            return
        win = lookread.window(b, rec)
        if win is None:
            block["reason"] = "fell before its window" if out["fell"] else "the window was not reached"
            return
        if ref["reading"] is None:
            raise lookread.LookError(f"{cond.key} has a window and no reference times (lookread.window_ref)")
        poses = getattr(rec, "poses", None)
        if not poses or int(getattr(rec, "legs_every", 0) or 0) != tune.LEGS_EVERY:
            raise lookread.LookError("the episode's recorder holds no 50 Hz poses (the inner row did not take the "
                                     "built condition's recorder)")
        every = tune.LEGS_EVERY * control.FEEDBACK_EVERY
        ser = lookread.series(b.p, np.asarray(poses, float), rec.loads, rec.dt, every)
        own = lookread.reading(ser, *win)
        block.update(window=[float(win[0]), float(win[1])], own=own,
                     air_runs=air_runs(ser, win[0], win[1], b.air_gap_max_s))
        c = lookread.compare(own, ref["reading"])
        block.update(read=True, terms=c["terms"], not_read=c["not_read"], too_few_steps=c["too_few_steps"],
                     look_ok=bool(c["ok"]), compare={k: c[k] for k in COMPARE_KEYS})


def sidecar_path(root, row: str, condition: str, seed: int) -> Path:
    """``<root>/<row>/<condition key>/<seed>.json`` (a row ``R5/42@10ms`` and a key ``walk/v0.4`` are two
    directories each)."""
    return Path(root) / row / condition / f"{int(seed)}.json"


def look_row_of(row) -> LookRow | None:
    """The ``LookRow`` inside a row (``bar.Delayed`` holds it as ``inner``), or None."""
    seen = 0
    while row is not None and not isinstance(row, LookRow) and seen < 8:
        row, seen = getattr(row, "inner", None), seen + 1
    return row if isinstance(row, LookRow) else None


# ------------------------------------------------------------------------------------------------
# R0L and the rows by name
# ------------------------------------------------------------------------------------------------
def _r0l(body: str, run_dir, name: str) -> panels.Row:
    if name != "R0L":
        raise ValueError(f"row {name!r}: the open-loop row with the look reading is R0L (it has no seed)")
    return LookRow(panels.tracker_rows(body, run_dir, ("R0",))["R0"], name="R0L")


panels.register_rows("R0L", _r0l)


def _register(inner: str) -> None:
    """Import the module that registers ``inner``'s controller, if it is one of ``ROW_MODULES`` and not there yet."""
    ctrl = panels.controller_of(inner)
    if ctrl in ("R0", "R1", "R2") or ctrl in panels.ROW_FACTORIES:
        return
    for start, module in ROW_MODULES:
        if ctrl == start or (start == "R2L" and ctrl.startswith(start)):
            try:
                importlib.import_module(module)
            except ModuleNotFoundError as exc:
                if exc.name != module:
                    raise
            return


def rows(body: str, names, sidecar=None, stage1_run=bar.STAGE1_RUN) -> dict[str, panels.Row]:
    """The bar's rows by name (``bar.rows``' names: ``R0L@10ms``, ``R5/42@10ms``, ``R0@10ms``, ...), each a
    ``bar.Delayed`` around a ``LookRow``: a registered row that is one already (``R0L``, the policy rows) as it is,
    any other row wrapped.  ``sidecar``: the root every row writes its episodes' blocks under, filed by the bar
    row's name."""
    run_dir = setio._anchored(stage1_run) / body
    out = {}
    for name in names:
        inner, k = bar.parse_row(name)
        _register(inner)
        row = panels.tracker_rows(body, run_dir, (inner,))[inner]
        look = row if isinstance(row, LookRow) else LookRow(row)
        look.sidecar = None if sidecar is None else str(sidecar)
        look.sidecar_row = name
        out[name] = bar.Delayed(look, k, name)
    return out


# ------------------------------------------------------------------------------------------------
# The look sets' own rule of what may run there (``RULES["look_sets"]``; the lead's rulings 16 and 31 of 17 Nov 2026)
# ------------------------------------------------------------------------------------------------
#: The rows ``look_set_rule`` lets run on a look set, by family, each with or without a delay ``@<ms>ms``
#: (``look_set_row``): ``R2L``'s ``<set>`` is the set evaluated.  No other row runs on a look set inside the context
#: or in ``episodes`` (the lead's ruling 31 of 17 Nov 2026, point 1).
LOOK_SET_ROWS = ("R0L", "R5/<seed>", "R6/<seed>", "R5G/<seed>", "R2L-<set>/<seed>")
_LOOK_SET_ROW = re.compile(r"^(?:R0L|R5/\d+|R6/\d+|R5G/\d+|R2L-(?P<set>[^/@\s]+)/\d+)$")
#: The code of ``bar.evaluate``: the one caller of the bar's set check that ``LookSetRule`` answers itself.
_EVALUATE_CODE = bar.evaluate.__code__


class LookSetRefused(LookBarError, bar.BarError):
    """Rule X1 on a look set (``RULES["look_sets"]``): the set's G9-ref has not passed for the body on its latest
    freeze, or a row outside ``LOOK_SET_ROWS`` was asked to run there, so nothing runs.  A refusal of the bar's kind
    (``bar.BarError``) and of this module's."""


def look_set_of(set_name) -> str | None:
    """The look set a set's name stands for: one of ``LOOK_SETS``, the name itself or the part before its first
    dot (``V6noF5.2`` is a version of ``V6noF5``, as ``bar.registered_name`` reads a version of the 75 cm round's
    sets; ``gates2.look_set_of`` reads it so too); None for any other set.  A version of a look set is a look set in
    the context, in ``episodes`` and in ``collect`` (the lead's ruling 31 of 17 Nov 2026, point 3)."""
    base = str(set_name).split(".", 1)[0]
    return base if base in LOOK_SETS else None


def look_set_row(name, set_name: str) -> bool:
    """Whether ``name`` is a row ``look_set_rule`` lets run on the look set ``set_name`` (``LOOK_SET_ROWS``):
    ``R0L``, ``R5/<seed>``, ``R6/<seed>``, ``R5G/<seed>`` or ``R2L-<set_name>/<seed>`` (the fit of that very set),
    with or without a delay ``@<ms>ms`` that ``bar.ms_steps`` accepts."""
    if not isinstance(name, str):
        return False
    inner, m = name, bar._ROW_NAME.match(name)
    if m:
        try:
            bar.ms_steps(int(m.group("ms")), name)
        except ValueError:
            return False
        inner = m.group("inner")
    r = _LOOK_SET_ROW.match(inner)
    return r is not None and r.group("set") in (None, str(set_name))


def require_g9_ref(set_name: str, body: str, root=setio.ROOT, path=None) -> dict:
    """Rule X1 on a look set or a version of one: ``gates2.require_g9_ref``, the gates' own check, called (the lead's
    ruling 31 of 17 Nov 2026, point 3: it covers versions; this module keeps no copy of it).  ``LookSetRefused``, with
    the gates' words, unless ``path`` (default ``gates2.gate_path(set, body, "g9_ref")``) is a G9-ref record of this
    set and body that passed and was made on the set's latest freeze under ``root``.  Returns its citation
    (``{path, sha256, pass_, at, set_freeze}``); ``LookSetRefused`` for a set that is no look set (nothing to cite)."""
    from designlab.stage2 import gates2

    try:
        cited = gates2.require_g9_ref(body, set_name, path, root)
    except gates2.Refused as exc:
        raise LookSetRefused(str(exc)) from None
    if not isinstance(cited, dict):
        raise LookSetRefused(f"{set_name}/{body}: the gates' G9-ref check names no record ({cited!r}): {set_name} is "
                             "no look set and no version of one")
    return cited


def _same_root(a, b) -> bool:
    return Path(setio._anchored(a)).resolve() == Path(setio._anchored(b)).resolve()


def _outside(set_name, row_names) -> list[str]:
    """The rows of ``row_names`` that are not of ``LOOK_SET_ROWS`` for the look set ``set_name``."""
    return [str(r) for r in row_names if not look_set_row(r, set_name)]


def _refuse_rows(set_name, body, rows: list[str]) -> LookSetRefused:
    who = f"{set_name}/{body}" if body is not None else str(set_name)
    return LookSetRefused(
        f"{who}: the row(s) {', '.join(rows)} never run on a look set or a version of one; only "
        f"{', '.join(LOOK_SET_ROWS)} do, each with or without a delay, once its G9-ref has passed (the lead's ruling "
        "31 of 17 Nov 2026; rule X1)")


class LookSetRule:
    """What ``look_set_rule`` yields: the set, body, set root and rows it was opened for; ``look_set`` (the look
    set the set's name stands for, None for any other set: then nothing is stood in front of); ``applies`` (the set
    is a look set and every row is one of ``LOOK_SET_ROWS`` for it); ``own`` (the bar's check in place when the
    context was entered); ``cited`` (the G9-ref record that let the last evaluation in, None when none did);
    ``placement`` (``placement_key`` of the set: join rule J1's leave point is placed for it while the rule applies,
    ``RULES["leave_point"]``)."""

    def __init__(self, set_name: str, body: str, row_names, root=setio.ROOT, g9_ref=None):
        self.set_name, self.body, self.root = str(set_name), str(body), root
        self.rows = tuple(str(r) for r in row_names)
        self.g9_ref = None if g9_ref is None else str(g9_ref)
        self.look_set = look_set_of(self.set_name)
        self.applies = bool(self.look_set and self.rows and not _outside(self.set_name, self.rows))
        self.own = bar._check_rule
        self.cited: dict | None = None
        self.placement = placement_key(self.set_name, self.body, setio.set_dir(self.set_name, self.body, root)) \
            if self.applies else None
        self._stamped: dict = {}  # id of a LookRow -> (the row, its g9_ref and placement before the first stamp)

    def covers(self, set_name, row_names, body, root) -> bool:
        """Whether an evaluation is the one the rule answers: of this set, body and set root, every row one the
        context was opened for (all of ``LOOK_SET_ROWS``: ``applies``)."""
        names = list(row_names)
        return bool(self.applies and str(set_name) == self.set_name and str(body) == self.body
                    and _same_root(root, self.root) and names and all(r in self.rows for r in names))

    def require(self) -> dict:
        """The G9-ref check (``require_g9_ref``) for this set and body, read now; ``cited`` its citation."""
        self.cited = None
        self.cited = require_g9_ref(self.set_name, self.body, self.root, self.g9_ref)
        return self.cited

    def stamp(self, rows) -> None:
        """Give each row of an evaluation the rule lets in (``rows``: its row objects) what its episodes carry: the
        G9-ref citation (``LookRow.g9_ref``, path and sha256: each episode's block holds it, and so its companion)
        and the set's placement (``LookRow.placement``: a worker that rebuilds the row from its spec places join
        rule J1's leave point too).  ``LookSetRefused`` for a row that holds no ``LookRow``: on a look set every row
        that runs is one (the families' factories make them so), or a worker could not be told either."""
        cite = None if self.cited is None else dict(path=self.cited["path"], sha256=self.cited["sha256"])
        looks = {name: look_row_of(row) for name, row in dict(rows).items()}
        bare = sorted(str(name) for name, look in looks.items() if look is None)
        if bare:
            raise LookSetRefused(f"{self.set_name}/{self.body}: the row(s) {', '.join(bare)} hold no lookbar.LookRow: "
                                 "on a look set every row that runs is one (RULES['look_sets'])")
        for look in looks.values():
            self._stamped.setdefault(id(look), (look, look.g9_ref, look.placement))
            look.g9_ref = cite
            look.placement = None if self.placement is None else placement_of(self.placement)

    def unstamp(self) -> None:
        """Every row ``stamp`` gave a citation and a placement gets back the ``g9_ref`` and ``placement`` it held
        before this rule first stamped it (``look_set_rule`` calls it at its end, whatever happens: F2 of the check
        of LT, ruled 37 of 17 Nov 2026), so that a row used again after the context carries neither."""
        while self._stamped:
            _, (look, g9_ref, placement) = self._stamped.popitem()
            look.g9_ref, look.placement = g9_ref, placement

    def bar_check(self, set_name, row_names, episodes, task, dev_episodes, passed_of, control,
                  block: str = bar.HEADLINE_BLOCK) -> None:
        """``bar._check_rule`` while the context is open (its signature): a call by ``bar.evaluate`` (its own frame:
        the body, the set root and the row objects are read there) that ``covers`` is let in once ``require`` holds,
        its rows stamped (``stamp``), and refused (``LookSetRefused``) when it does not; an evaluation of a look set
        with a row outside ``LOOK_SET_ROWS`` is refused; any other call is the bar's own check, with the same
        arguments."""
        caller = sys._getframe(1)
        try:
            loc = caller.f_locals
            at = ((loc.get("body"), loc.get("root"), loc.get("row_objs")) if caller.f_code is _EVALUATE_CODE else None)
        finally:
            del caller
        return self._decide(at, (set_name, row_names, episodes, task, dev_episodes, passed_of, control, block))

    def _decide(self, at, args: tuple) -> None:
        """``bar_check``'s answer for the evaluation ``at`` (its ``(body, root, row objects)``, None: no call of
        ``bar.evaluate``) and the check's arguments ``args``.  On a look set a row outside ``LOOK_SET_ROWS`` is refused
        whoever calls (the lead's ruling 31 of 17 Nov 2026, point 1); a call it does not cover goes to the check it
        found in place, which is another context's when contexts are nested (that one is asked with the same
        evaluation)."""
        if at is not None and self.covers(args[0], args[1], at[0], at[1]):
            self.require()
            self.stamp(at[2] or {})
            return None
        if look_set_of(args[0]) is not None:
            outside = _outside(args[0], args[1])
            if outside:
                raise _refuse_rows(args[0], None if at is None else at[0], outside)
        outer = getattr(self.own, "__self__", None)
        if isinstance(outer, LookSetRule) and getattr(self.own, "__func__", None) is LookSetRule.bar_check:
            return outer._decide(at, args)
        return self.own(*args)

    def check(self) -> dict | None:
        """What the rule decides for its own set, body, root and rows, as ``episodes`` asks it (that function does
        not go through the bar's runner): off the look sets nothing is checked (None); on a look set the G9-ref
        check when ``applies`` (its citation), else ``LookSetRefused`` (a row outside ``LOOK_SET_ROWS``)."""
        if self.look_set is None:
            return None
        if self.applies:
            return self.require()
        raise _refuse_rows(self.set_name, self.body, _outside(self.set_name, self.rows) or ["(none given)"])


# ------------------------------------------------------------------------------------------------
# Join rule J1's leave point on a look set (``RULES["leave_point"]``; the lead's ruling 21 of 17 Nov 2026)
# ------------------------------------------------------------------------------------------------
#: The bar planner's own ``_resolve`` (``bar.BarPlanner``), as this module found it.
_BAR_RESOLVE = bar.BarPlanner.__dict__["_resolve"]
#: The sets for which a bar evaluation places join rule J1's leave point: ``placement_key`` -> the number of open
#: contexts that hold it, and the keys a worker holds for its whole life (``LookRow.from_spec``).
_PLACED: dict = {}
_FOR_LIFE: set = set()
#: The keys for which a turn or a stop was placed while a context held them (``look_set_rule`` then lets the
#: process's cached bar context forget what it built).
_USED: set = set()


def placement_key(set_name: str, body: str, set_dir) -> tuple[str, str, str]:
    """``(set, body, the set directory resolved)``: what a bar planner is matched by (``set.json``'s ``set`` and
    ``body``, and its ``set_dir``)."""
    return str(set_name), str(body), str(Path(setio._anchored(set_dir)).resolve())


def placement_of(key) -> dict:
    """A placement key as a row's spec and a worker carry it: ``{set, body, set_dir}`` (``hold_for_life``)."""
    return dict(zip(("set", "body", "set_dir"), key))


def _planner_key(planner) -> tuple[str, str, str]:
    doc = planner.set_doc or {}
    return placement_key(doc.get("set"), doc.get("body"), planner.set_dir)


def placed() -> set:
    """The keys for which bar planners place join rule J1's leave point now (``_PLACED`` and ``_FOR_LIFE``)."""
    return set(_PLACED) | _FOR_LIFE


def leave_of(planner, cond) -> dict | None:
    """This module's phase check of a look set's causal move (``RULES["leave_point"]``; ``bar.check_phase`` reads a
    lead modulo one stride and ``bar.py`` is not edited): the strides of the bar's lead before the move
    (``bar.TURN_LEAD_STRIDES``, ``bar.STOP_LEAD_STRIDES``) counted inside the window the move leaves, a left
    command at a whole stride and a right one at a half, by the env's own reading (``r5lib.leave_point``, given the
    walk's file: ``a_time_s`` must be where the file's own ``phase`` array passes ``a_phase``).  None for
    a condition that is no turn or stop, or whose move carries no leave point (the bar's own placement then holds);
    else the leave point; ``tune.InputError`` where it does not fit."""
    from designlab.stage2 import r5lib

    if not (bar.is_bar(cond) and cond.judge in ("bar_turn", "bar_stop") and cond.move):
        return None
    name = setio.parse_name(cond.move)
    lead = (bar.TURN_LEAD_STRIDES[name["direction"]] if cond.judge == "bar_turn"
            else bar.STOP_LEAD_STRIDES[name["steps"]])
    entry = planner.refs_doc.get(cond.walk)
    rel = entry.get("file") if isinstance(entry, dict) and entry.get("file") else f"refs/{cond.walk}.npz"
    try:
        return r5lib.leave_point(cond.move, planner.tr_doc.get(cond.move), cond.walk, entry, lead,
                                 walk_file=Path(setio._anchored(planner.set_dir)) / rel)
    except tune.InputError as exc:
        raise tune.InputError(f"{cond.key}: {exc}") from None


def leave_resolve(planner, cond) -> dict | None:
    """``bar.BarPlanner._resolve``'s value of a look set's turn or stop item with the walk before the move ended at
    join rule J1's leave point (``RULES["leave_point"]``): the bar's own arithmetic of that item (its plan, seconds,
    command and window), with the walk's item ``("ref", walk, m + a_windows)`` in place of the bar's whole windows
    and the command's ``leave`` naming the leave point.  None where ``leave_of`` gives none (the bar's own value
    then holds)."""
    leave = leave_of(planner, cond)
    if leave is None:
        return None
    W, name = cond.walk, setio.parse_name(cond.move)
    lead = planner.lead_in(W)
    P, T = planner.window_s(W), planner.period_s(W)
    out = dict(command_ref_s=None, steady_ref_s=None, window=None, push=None, at_s=None, command=None,
               rest_ref_s=None, target_speed_m_s=planner.speed(W), period_s=T)
    k = len(lead)
    t_walk = planner.segment_times(lead + [("ref", W, 1)])[k][0]
    move = planner.tr_doc[cond.move].get("move") or {}
    note = dict(rule="join rule J1's leave point (lookbar.RULES['leave_point'])",
                **{key: leave[key] for key in ("windows", "whole_windows", "strides", "lead_strides", "a_phase",
                                               "a_time_s", "a_windows")})
    head = lead + [("ref", W, leave["windows"]), ("transition", cond.move)]
    if cond.judge == "bar_turn":
        times = planner.segment_times(head + [("ref", W, 1)])
        t_cmd, t_after = times[len(head) - 1][0], times[len(head)][0]
        at_s = name["angle_rad"] / abs(name["rate_rad_s"])
        seconds = t_cmd / (bar.SLOW * bar.RATE) + at_s + readings.HEADING_HOLD_S + bar.END_MARGIN_S
        n = max(1, math.ceil((seconds * bar.FAST * bar.RATE - t_after) / P))
        return dict(out, plan=head + [("ref", W, n)], seconds=seconds, command_ref_s=t_cmd, at_s=at_s,
                    steady_ref_s=t_walk, sign=name["sign"], angle_rad=name["angle_rad"],
                    command=dict(name=cond.move, ref_s=t_cmd, phase=move.get("command_phase"),
                                 convention="causal: the move's first frame", leave=note),
                    window=dict(kind="from the command", seconds=at_s + readings.HEADING_HOLD_S))
    t_seg, dur = planner.segment_times(head)[-1]
    t_rest = t_seg + dur
    seconds = t_rest / (bar.SLOW * bar.RATE) + bar.STOP_HOLD_S + bar.END_MARGIN_S
    hold = seconds * bar.FAST * bar.RATE - t_rest
    command = dict(name=cond.move, ref_s=t_seg, phase=move.get("command_phase"),
                   convention="causal: the move's first frame", leave=note)
    return dict(out, plan=head + [("hold", hold)], seconds=seconds, command_ref_s=t_seg, steady_ref_s=t_walk,
                rest_ref_s=t_rest, command=command,
                window=dict(kind="the whole episode", hold_after_rest_s=bar.STOP_HOLD_S))


def _placed_resolve(self, cond) -> dict:
    """``bar.BarPlanner._resolve`` while a set is ``placed()``: for a planner of such a set, a turn or stop item whose
    move carries a leave point is ``leave_resolve``'s; every other planner and item is the bar's own, unchanged."""
    if bar.is_bar(cond) and cond.judge in ("bar_turn", "bar_stop"):
        key = _planner_key(self)
        if key in placed():
            res = leave_resolve(self, cond)
            if res is not None:
                _USED.add(key)
                return res
    return _BAR_RESOLVE(self, cond)


def _install() -> None:
    now = bar.BarPlanner.__dict__.get("_resolve")
    if now is _placed_resolve:
        return
    if now is not _BAR_RESOLVE:
        raise LookBarError("bar.BarPlanner._resolve is neither the bar's own nor this module's: another hand "
                           "replaced it, so the look sets' placement is not stood in front of it")
    bar.BarPlanner._resolve = _placed_resolve


def _uninstall() -> None:
    if placed():
        return
    now = bar.BarPlanner.__dict__.get("_resolve")
    if now is _BAR_RESOLVE:
        return
    if now is not _placed_resolve:
        raise LookBarError("bar.BarPlanner._resolve is not this module's: it is not put back (another hand replaced "
                           "it while a look set's context was open)")
    bar.BarPlanner._resolve = _BAR_RESOLVE


def _hold(key) -> None:
    _install()
    _PLACED[key] = _PLACED.get(key, 0) + 1


def _release_cached(key) -> bool:
    """The process's cached bar context (``bar.context_for``, which ``open_set`` and a bar worker in this process
    use) forgets its plants, tracks and built conditions (``BarContext.release``) when it is the set's of ``key`` (its
    run directory is the key's set directory); whether it did.  A context calls it on entry (F1 of the check of LT,
    ruled 37 of 17 Nov 2026: a turn or stop built there before, outside the placement, is not played inside it) and
    ``_let_go`` at its end (one built inside is not played outside)."""
    ctx = bar._CONTEXT.get("ctx")
    if ctx is not None and str(Path(ctx.run_dir).resolve()) == key[2]:
        ctx.release()
        return True
    return False


def _let_go(key) -> None:
    """The context's end for its key: the count of open contexts goes down, the bar's ``_resolve`` is put back when
    no set is placed any more, and where a turn or stop was placed for the key the process's cached bar context
    (``bar.context_for``) of that set forgets its plants and built conditions (``_release_cached``), so that a
    later evaluation outside the context builds the bar's own placement."""
    n = _PLACED.get(key, 0) - 1
    if n > 0:
        _PLACED[key] = n
    else:
        _PLACED.pop(key, None)
    try:
        _uninstall()
    finally:
        if key not in placed() and key in _USED:
            _USED.discard(key)
            _release_cached(key)


def hold_for_life(placement) -> bool:
    """A worker's hold of a set's placement (``LookRow.from_spec``: a spawned bar worker rebuilds its rows from their
    specs and lives for one evaluation): the key held for the process's life unless an open context of this process
    holds it already (an evaluation in this process: the context's own hold stands, and is let go with it).  Whether
    it was held now."""
    if not isinstance(placement, dict):
        return False
    key = placement_key(placement.get("set"), placement.get("body"), placement.get("set_dir"))
    if key in _PLACED or key in _FOR_LIFE:
        return False
    _FOR_LIFE.add(key)
    _install()
    return True


@contextlib.contextmanager
def look_set_placement(set_name: str, body: str, root=setio.ROOT):
    """Join rule J1's leave point alone (``RULES["leave_point"]``; the lead's ruling 37 (b) of 17 Nov 2026): a
    context inside which every bar planner of ``set_name`` (a look set or a version of one), ``body`` and the set root
    ``root`` plans the set's turns and two-step stop with the walk ending at their leave point (``leave_resolve``),
    without the look sets' row rule and without the G9-ref requirement, because G9-ref itself runs before G9-ref has
    passed.  It yields the placement (``{set, body, set_dir}``: what a spawned worker is given to hold for its life,
    ``hold_for_life``).  On a set that is no look set it yields None and does nothing at all.

    It is narrow and restoring as ``look_set_rule`` is (which places through it): it stands in front of
    ``bar.BarPlanner._resolve`` only, acts for that set's planners and their turn and stop items with a leave point
    only, and puts the bar's piece back when the last such context ends, also after an exception, where its own is
    the one in place (else ``LookBarError``).  On entry the process's cached bar context of the set forgets what it
    built (``_release_cached``; F1 of the check of LT), so that a condition built before, outside the placement, is
    built again inside it; at its end a cached condition placed inside is forgotten in the same way (``_let_go``).
    These run inside it on a look set: G9-ref part (a) (``gates2._refcheck_record``, ``bar.refcheck``) and part (b)
    (``gates2.part_b``, its walk and turn items), G8a (``gates2.g8a``), ``s2rows.equivalence_check``, R2L's fit and
    selection (``r2lfit``), the owner's views (``barviews.SetView.resolve``, in a child process where this process
    renders) and the placed refcheck command (``gates2.refcheck``, ``gates2 refcheck``; the lead's ruling 39 (b)).
    ``bar.py`` is not edited and its hash does not move."""
    if look_set_of(set_name) is None:
        yield None
        return
    key = placement_key(str(set_name), str(body), setio.set_dir(str(set_name), str(body), root))
    _hold(key)
    try:
        _release_cached(key)
        yield placement_of(key)
    finally:
        _let_go(key)


@contextlib.contextmanager
def look_set_rule(set_name: str, body: str, row_names, root=setio.ROOT, g9_ref=None):
    """The look sets' own rule of what may run there (``RULES["look_sets"]``; the lead's rulings 16 and 31 of 17 Nov
    2026): a context, yielding its ``LookSetRule``, inside which a bar evaluation (``bar.evaluate``) of ``set_name`` (a
    look set or a version of one), ``body`` and the set root ``root`` whose rows are all of ``row_names`` lets those
    rows run if every one of ``row_names`` is of ``LOOK_SET_ROWS`` (``R0L``, ``R5/<seed>``, ``R6/<seed>``,
    ``R5G/<seed>``, ``R2L-<set>/<seed>``, each with or without a delay), and only once the set's G9-ref has passed for
    the body on its latest freeze (rule X1, ``require_g9_ref``, read at the evaluation's check; ``g9_ref``: the
    record's path, default ``stage2/tb/gates/<set>/<body>/g9_ref.json``); its rows then carry the record's citation
    and its turns and stop leave the walk at join rule J1's leave point (``RULES["leave_point"]``, through
    ``look_set_placement``, held while the context is open where the rule applies).  When the context ends every row
    it stamped gets back the citation and placement it held before (``LookSetRule.unstamp``).  An evaluation of
    a look set with any row outside ``LOOK_SET_ROWS`` is refused; every other evaluation inside it, of another set,
    body, root or rows, goes through the bar's own check unchanged; outside it the bar's check is the bar's.  On a set
    that is no look set nothing is replaced at all.  When the context ends the bar's ``_check_rule`` (and the
    planner's ``_resolve``) is put back, also after an exception, but only where this context's own is the one in
    place: otherwise ``LookBarError`` and nothing is put back.  ``bar.py`` is not edited and its hash does not
    move."""
    rule = LookSetRule(set_name, body, row_names, root, g9_ref)
    if rule.look_set is None:
        yield rule
        return
    bar._check_rule = rule.bar_check
    try:
        if rule.placement is None:
            yield rule
        else:
            with look_set_placement(rule.set_name, rule.body, rule.root):
                yield rule
    finally:
        try:
            rule.unstamp()
        finally:
            if bar._check_rule != rule.bar_check:
                raise LookBarError("bar._check_rule is not this context's own check at its end (another context "
                                   "ended out of order, or another hand replaced it): it is not put back")
            bar._check_rule = rule.own


# ------------------------------------------------------------------------------------------------
# One companion episode
# ------------------------------------------------------------------------------------------------
_UNREADABLE = re.compile(r"^(?:[\w.]*\.)?Unreadable: ")


def is_unreadable(error) -> bool:
    """Whether an episode's error text is the bar's ``Unreadable``: its last line names that exception (a
    traceback's last line, or the line ``episodes`` writes)."""
    lines = str(error).strip().splitlines() if error else []
    return bool(lines) and _UNREADABLE.match(lines[-1]) is not None


def is_warned(*docs) -> bool:
    """Whether an episode is one during which MuJoCo warned, by any of ``docs`` (its block, the bar's episode
    record; None is skipped): it holds ``mujoco_warnings``, or its fall cause is the one ``tune.warned_fall``
    writes (``WARNED_CAUSE``)."""
    return any(bool(d.get("mujoco_warnings")) or str(d.get("fall_cause") or "").startswith(WARNED_CAUSE)
               for d in docs if d)


def companion_episode(condition: str, judge: str, seed: int, e: dict | None, block: dict | None, look_set: bool,
                      error: str | None = None) -> dict:
    """One episode as the companion holds it (module docstring: the outcome and the readings).

    ``e``: the bar's episode record (``bar.run_episode``'s, or a bar record's: an episode that failed to run holds
    ``error``), None when ``bar.run_episode`` raised; ``error``: what it raised then (``is_unreadable`` reads
    whether that was ``bar.Unreadable``); ``block``: the row's block of the episode (the sidecar), None when there
    is none.  ``succeeded`` (reading 2), ``succeeded_but_A1`` (reading 3; None for a stock condition), and on a
    look set ``succeeded_look`` (reading 1) and ``succeeded_but_A1_look``; ``failed_terms`` of reading 2 as the
    companion counts it (on a look set with ``on_command_at_stop`` and the look terms as ``look_<term>``)."""
    err = error if error is not None else (e or {}).get("error")
    block = block or None
    is_bar = str(judge).startswith("bar_")
    read_item = judge in READ_JUDGES
    policy = bool(block and block.get("policy_check") is not None)
    clock = (block or {}).get("clock")
    mean = (clock or {}).get("mean_factor")
    reason = unreadable = None
    if err:
        if is_unreadable(err) and block is not None and not block.get("fell") and policy:
            outcome = "unread"
            reason = "clock_behind" if (mean is not None and mean < 1.0 - CLOCK_TOL) else "unreadable"
            unreadable, err = str(err).strip().splitlines()[-1], None
        else:
            outcome = "error"
    elif e is None or block is None:
        outcome, err = "error", "the episode left no bar episode and look block to read"
    elif e["fell"]:
        outcome = "fell"
    elif look_set and read_item and not block.get("read"):
        outcome = "error"
        err = ("the look reading cannot be given (rule U5: a fault of the definition, or of the reading): "
               + str(block.get("fault") or block.get("reason")))
    else:
        outcome = "read"
    read = outcome == "read"
    ok2 = bool(read and e.get("succeeded"))
    ok3 = bool(read and e.get("succeeded_but_A1")) if is_bar else None
    failed = list(e.get("failed_terms") or []) if read else ["survived"] if outcome == "fell" else []
    failed3 = (list(e.get("failed_terms_but_A1") or []) if read else ["survived"] if outcome == "fell" else []) \
        if is_bar else None
    oc = (block or {}).get(ON_COMMAND_TERM)
    if look_set and judge == STOP_JUDGE and read and not (oc and oc.get("term")):
        ok2 = ok3 = False
        failed.append(ON_COMMAND_TERM)
        failed3.append(ON_COMMAND_TERM)
    look = look_of(block)
    look_failed = [t for t in lookread.TERMS if not look["terms"][t]] if look else []
    if not look_set:
        ok1 = ok13 = None
        reading_1 = READING_1["not_counted"]
    elif read_item:
        good = bool(look and look["look_ok"])
        ok1, ok13 = bool(ok2 and good), bool(ok3 and good)
        reading_1 = READING_1["walk_turn"]
        if read:
            failed += [f"look_{t}" for t in look_failed]
            failed3 += [f"look_{t}" for t in look_failed]
    else:
        ok1, ok13 = ok2, ok3
        reading_1 = READING_1["other"]
    src = block if block is not None else (e or {})
    fell = None if block is None and (e is None or (e or {}).get("error")) else bool(src.get("fell"))
    simulated = float(src.get("survival_s") or 0.0)
    # (RULES["outcome"], the lead's ruling 9 of 17 Nov 2026: an episode during which MuJoCo warned fell at its start)
    warned = is_warned(block, e)
    fall_time = 0.0 if warned else None if not fell else src.get("fall_time_s", (e or {}).get("fall_time_s"))
    return dict(
        condition=condition, seed=int(seed), outcome=outcome, reason=reason, unreadable=unreadable,
        fell=fell, warned=warned, fall_time_s=None if fall_time is None else float(fall_time),
        survival_s=0.0 if warned else simulated, simulated_s=simulated, planned_s=src.get("planned_s"),
        steps=(block or {}).get("steps"),
        succeeded=ok2, succeeded_but_A1=ok3, succeeded_look=ok1, succeeded_but_A1_look=ok13, reading_1=reading_1,
        failed_terms=failed, failed_terms_but_A1=failed3, look=look, look_ok=None if look is None else look["look_ok"],
        look_failed_terms=look_failed, look_reason=(block or {}).get("reason"), look_fault=(block or {}).get("fault"),
        on_command_at_stop=oc, clock=clock, error=None if err is None else str(err),
        state_digest=(block or {}).get("state_digest"), policy_check=(block or {}).get("policy_check"),
        fall_cause=src.get("fall_cause"),
        bar=None if (e is None or e.get("error")) else dict(
            succeeded=bool(e.get("succeeded")), succeeded_but_A1=e.get("succeeded_but_A1"),
            failed_terms=list(e.get("failed_terms") or []), failed_terms_but_A1=e.get("failed_terms_but_A1"),
            command_s=e.get("command_s"), window_reached=e.get("window_reached"), trace=e.get("trace"),
            wall_s=e.get("wall_s")))


def _counted(ep: dict, key: str) -> dict:
    """A companion episode as ``bar.counts`` reads it in the reading ``key`` (``RULES["counts"]``)."""
    o = ep["outcome"]
    return dict(seed=ep["seed"], fell=o in ("fell", "error"),
                error=(ep.get("error") or "error") if o == "error" else None, succeeded=bool(ep.get(key)))


def counts_of(episodes: list, seeds, key: str, reading: str) -> dict:
    """``bar.counts`` on the companion's episodes in one reading (``key``: the episodes' field), named ``reading``."""
    return dict(bar.counts([_counted(ep, key) for ep in episodes], seeds, "success"), reading=reading)


# ------------------------------------------------------------------------------------------------
# Episodes in this process (the monitor's entry)
# ------------------------------------------------------------------------------------------------
def open_set(body: str, set_name: str, keys, root=setio.ROOT, stage1_run=None) -> dict:
    """A set read as ``bar.BarWorker`` reads it, for the condition keys ``keys``: ``{ctx, conds {key: condition},
    missing {key: {reference, reason, numbers}}, inputs, run_dir, set_docs}``.  The pseudo-set ``S1``: the Stage 1
    run directory of the body (``stage1_run``; default ``r5lib.s1_root(body)``) with the stock conditions.  Any
    other set: frozen, its manifest's hash and its crouch checked, every file the conditions play read under its
    latest freeze.  ``ValueError`` for a key the bar does not define."""
    keys = list(dict.fromkeys(keys))
    if set_name == bar.STOCK_SET:
        from designlab.stage2 import r5lib

        run_root = setio._anchored(r5lib.s1_root(body) if stage1_run is None else stage1_run)
        run_dir = run_root / body
        inputs, docs = setio.stage1_view(run_root, body)
        set_docs, set_doc = None, None
        known = {c.key: c for c in bar.stock_conditions()}
        _known(keys, known, set_name)
        missing = {}
        for k in keys:
            stands, settle = panels.stands(run_dir, known[k].floor)
            if not stands:
                missing[k] = dict(reference=f"floor {known[k].floor}", reason=panels.NOT_RUN_RULE, numbers=settle)
    else:
        run_dir = setio.set_dir(set_name, body, root)
        set_docs = setio.load(run_dir)
        set_doc = set_docs["set.json"]
        if set_doc.get("set") != set_name:
            raise bar.BarError(f"{run_dir}: set.json names the set {set_doc.get('set')!r}")
        bar.check_manifest(set_doc)
        known = {c.key: c for c in bar.conditions(set_docs, every=True)}
        _known(keys, known, set_name)
        planner = bar.BarPlanner(run_dir, set_docs["refs.json"]["references"],
                                 set_docs["transitions.json"]["transitions"], set_doc)
        missing = {k: why for k in keys if (why := planner.missing(known[k])) is not None}
        files = sorted({f for k in keys if k not in missing for f in planner.resolve(known[k])["files"]})
        inputs, docs = setio.read_inputs(run_dir, files, require_frozen=True)
        bar.set_crouch(set_doc, docs["refs.json"]["references"], bar.walks_of(files))
    ident = s2io.sha256_json(dict(files=inputs["files"], freeze=inputs.get("freeze")))
    ctx = bar.context_for(body, run_dir, docs, set_doc, None, ident)
    return dict(ctx=ctx, conds={k: known[k] for k in keys}, missing=missing, inputs=inputs, run_dir=run_dir,
                set_docs=set_docs)


def _known(keys, known: dict, set_name: str) -> None:
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise ValueError(f"conditions {unknown} are not among the conditions of {set_name} ({', '.join(known)})")


def episodes(body: str, set_name: str, rows: dict, pairs: list, traces=None, root=setio.ROOT,
             stage1_run=None) -> list[dict]:
    """One companion episode per row of ``rows`` (``rows()``'s: each holds a ``LookRow``) and ``(condition key,
    seed)`` of ``pairs``, in that order, run in this process: ``bar.run_episode`` on the set read as the bar's
    workers read it (``open_set``), ``bar.Unreadable`` and any other exception caught apart, then
    ``companion_episode``.  A condition whose reference the set does not hold is ``not_run``.  ``traces``: where
    the episodes' traces go (the bar's relative paths), None: none is written.  Each episode also holds its
    ``row``, ``set``, ``start_seed`` and ``wall_s``.

    On a look set (or a version of one) the rows are checked first, before the set is read
    (``RULES["look_sets"]``, ``LookSetRule.check``): rows of ``LOOK_SET_ROWS`` only once the set's G9-ref has
    passed for the body on its latest freeze under ``root``, and no other row (``LookSetRefused`` otherwise; the
    lead's ruling 31 of 17 Nov 2026).  Everything then runs inside the look sets' context: each row's blocks carry
    the G9-ref record's citation and the set's turns and stop leave the walk at join rule J1's leave point
    (``RULES["leave_point"]``).  On any other set nothing is checked here and nothing is placed."""
    s2io.require_one_thread()
    rows = dict(rows)
    looks = {name: look_row_of(row) for name, row in rows.items()}
    plain = [name for name, look in looks.items() if look is None]
    if plain or not rows:
        raise LookBarError(f"episodes takes rows that hold a LookRow (lookbar.rows): {plain or 'none given'}")
    pairs = [(str(k), int(s)) for k, s in pairs]
    with look_set_rule(set_name, body, list(rows), root) as rule:  # (rule X1: before anything is read or run)
        if rule.check() is not None:
            rule.stamp(rows)
        return _episodes(body, set_name, rows, looks, pairs, traces, root, stage1_run)


def _episodes(body: str, set_name: str, rows: dict, looks: dict, pairs: list, traces, root, stage1_run) -> list[dict]:
    """``episodes``' run, once its rule let it in."""
    opened = open_set(body, set_name, [k for k, _ in pairs], root, stage1_run)
    ctx, conds, missing = opened["ctx"], opened["conds"], opened["missing"]
    look_set = look_set_of(set_name) is not None
    out = []
    for name, row in rows.items():
        look = looks[name]
        for key, seed in pairs:
            cond = conds[key]
            if key in missing:
                out.append(dict(
                    row=name, set=set_name, condition=key, seed=seed, outcome=NOT_RUN,
                    reason=str(missing[key].get("reason")), reference=missing[key].get("reference"), fell=None,
                    warned=False, fall_time_s=None, survival_s=0.0, simulated_s=0.0, planned_s=None, succeeded=False,
                    succeeded_but_A1=False, succeeded_look=False, succeeded_but_A1_look=False, failed_terms=[],
                    look=None, clock=None, error=None))
                continue
            t0 = time.perf_counter()
            look.last = None
            e = err = None
            try:
                e = bar.run_episode(row, ctx, cond, seed, trace_root=traces)
            except bar.Unreadable as exc:
                err = f"{bar.Unreadable.__module__}.Unreadable: {exc}"
            except Exception:  # noqa: BLE001 - an infrastructure fault: the episode is an error, the others go on
                err = traceback.format_exc()
            ep = companion_episode(key, cond.judge, seed, e, look.last, look_set, err)
            ep.update(row=name, set=set_name, start_seed=panels.episode_seeds(cond, seed)["start"],
                      wall_s=time.perf_counter() - t0)
            out.append(ep)
    return out


# ------------------------------------------------------------------------------------------------
# The companion records
# ------------------------------------------------------------------------------------------------
def _read_json(path) -> dict | None:
    try:
        doc = s2io.read_json(path)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def is_look_record(rec: dict | None) -> bool:
    """Whether a bar record is of a ``LookRow`` row (its controller's description holds ``look_reading``)."""
    return bool(rec and rec.get("schema") == bar.SCHEMA and "key_name" in rec
                and isinstance(rec.get("controller"), dict) and "look_reading" in rec["controller"])


def record_row_sha256(rec: dict) -> str:
    """``LookRow.row_sha256`` found again in a bar record: the hash of its ``controller`` less the delay wrapper's
    keys."""
    return s2io.sha256_json({k: v for k, v in rec["controller"].items() if k not in DELAYED_KEYS})


def _policy_block(blocks: list) -> dict | None:
    """The companion's ``policy`` block from its episodes' ``policy_check`` (None for a row without a policy)."""
    checks = [b["policy_check"] for b in blocks if b and b.get("policy_check") is not None]
    if not checks:
        return None
    first = checks[0]
    diffs = [c.get("fingerprint_max_abs_diff") for c in checks if c.get("fingerprint_max_abs_diff") is not None]
    return dict(versions_trained=first.get("versions_trained"), versions_judged=first.get("versions_judged"),
                cross_version=first.get("cross_version"),
                fingerprint_max_abs_diff=max(diffs) if diffs else None, fingerprint_tol=first.get("fingerprint_tol"),
                checkpoint=first.get("checkpoint"), source_change=first.get("source_change"),
                source_hashes=first.get("source_hashes"), run=first.get("run"),
                set_freeze_now=first.get("set_freeze_now"), later_freeze=first.get("later_freeze"),
                episodes_checked=len(checks),
                same_checkpoint=all(c.get("checkpoint") == first.get("checkpoint") for c in checks))


def g9_ref_of(blocks) -> dict | list | None:
    """The G9-ref citation (path and sha256) under which a record's episodes ran, from their blocks (the lead's
    ruling 31 of 17 Nov 2026, point 5): the one citation they hold, a list of them in the order they first appear
    where they hold more than one, None where none holds one (a set that is no look set)."""
    cites = []
    for b in blocks:
        c = (b or {}).get("g9_ref")
        if isinstance(c, dict) and c not in cites:
            cites.append(dict(c))
    return None if not cites else cites[0] if len(cites) == 1 else cites


def companion(path, rec: dict, out, side_root, look_set: bool, g9_ref=None) -> dict:
    """The companion record of one bar record (``rec``, read from ``path``; schema ``s2-look-1``), with the G9-ref
    citation its episodes ran under (``g9_ref_of``); on a look set the companion of a record "not run", which has no
    episode, cites ``g9_ref`` (the G9-ref record of the evaluation that wrote it, as ``run`` gives it to ``collect``:
    path and sha256) as the others cite theirs (the lead's ruling 42 of 17 Nov 2026, after ruling 37 (c)).
    ``LookBarError`` where a judged episode's sidecar is missing or is not that episode's (``RULES["sidecar"]``)."""
    path, out = Path(path), Path(out)
    row, key, judge = rec["row"], rec["key_name"], (rec.get("definition") or {}).get("judge")
    seeds = [int(s) for s in rec.get("seeds") or []]
    is_bar = not (rec.get("definition") or {}).get("stock")
    want = record_row_sha256(rec)
    eps, blocks, hashes, bad = [], [], {}, []
    for e in rec.get("episodes") or []:
        seed = int(e["seed"])
        sp = sidecar_path(side_root, row, key, seed)
        block = _read_json(sp)
        failed = bool(e.get("error"))
        if block is None:
            if not failed or is_unreadable(e.get("error")):
                bad.append(f"seed {seed}: no sidecar {sp.name} under {sp.parent}")
                continue
        else:
            hashes[str(seed)] = s2io.sha256_file(sp)
            diff = [] if block.get("row_sha256") == want else ["row_sha256 (another row's description)"]
            if (block.get("schema"), block.get("condition"), block.get("seed")) != (SIDECAR_SCHEMA, key, seed):
                diff.append("schema, condition or seed")
            if not failed:
                diff += [k for k in ("fell", "survival_s") if block.get(k) != e.get(k)]
                steps = (e.get("trace") or {}).get("steps")
                if steps is not None and block.get("steps") != steps:
                    diff.append("steps")
            if diff:
                bad.append(f"seed {seed}: the sidecar is not this episode's ({', '.join(diff)} differ)")
                continue
        blocks.append(block)
        eps.append(companion_episode(key, judge, seed, e, block, look_set))
    if bad:
        raise LookBarError(f"{path}: " + "; ".join(bad) + f" ({RULES['sidecar']})")
    done = rec.get("status") == "done"
    a1_over = bool(rec.get("a1_over"))
    c1 = c2 = c3 = c13 = equal = None
    if done and look_set:
        c2 = counts_of(eps, seeds, "succeeded", "success")
        c3 = counts_of(eps, seeds, "succeeded_but_A1", "but_A1") if is_bar else None
        c1 = counts_of(eps, seeds, "succeeded_look", "look")
        c13 = counts_of(eps, seeds, "succeeded_but_A1_look", "but_A1_look") if is_bar else None
        if a1_over:  # as the bar's record: readings 1 and 2 give a turn frozen with a1_over no verdict
            c2 = dict(c2, pass_=None, not_read="a1_over", reason="not read: a1_over", pass_if_read=c2["pass_"])
            c1 = dict(c1, pass_=None, not_read="a1_over", reason="not read: a1_over", pass_if_read=c1["pass_"])
    elif done:
        c2 = bar.counts(rec["episodes"], seeds, "success")
        c3 = bar.counts(rec["episodes"], seeds, "but_A1") if is_bar else None
        keys = ("episodes", "survived", "succeeded", "falls", "errors")
        equal = bool(all(c2[k] == (rec.get("counts") or {}).get(k) for k in keys)
                     and (c3 is None or all(c3[k] == (rec.get("counts_but_A1") or {}).get(k) for k in keys)))
        if not equal:
            raise LookBarError(f"{path}: its counts are not bar.counts of its own episodes (the record was changed)")
        c2, c3 = rec["counts"], rec.get("counts_but_A1")
    outcomes = {o: sum(1 for ep in eps if ep["outcome"] == o) for o in OUTCOMES}
    reasons = Counter(ep["reason"] for ep in eps if ep["outcome"] == "unread")
    terms = Counter(t for ep in eps for t in ep["failed_terms"])
    look_terms = Counter(t for ep in eps for t in ep["look_failed_terms"])
    short = [ep["look"]["air_runs"]["shorter_than_air_gap_max"] for ep in eps if ep["look"] and ep["look"]["air_runs"]]
    cited = g9_ref_of(blocks)
    if cited is None and look_set and g9_ref is not None and rec.get("status") == "not run" and not eps:
        cited = dict(path=g9_ref["path"], sha256=g9_ref["sha256"])
    doc = dict(
        schema=COMPANION_SCHEMA,
        bar_record=dict(path=Path(os.path.relpath(path, out)).as_posix(), sha256=s2io.sha256_file(path),
                        key_sha256=rec.get("key_sha256"), status=rec.get("status")),
        set=rec["set"], body=rec["body"], block=rec["block"], row=row, condition=key, behavior=rec["behavior"],
        name=rec["condition"], judge=judge, gated=rec.get("gated"), look_set=bool(look_set),
        look_sets=list(LOOK_SETS), reading_1=(READING_1["not_counted"] if not look_set else READING_1["walk_turn"]
                                              if judge in READ_JUDGES else READING_1["other"]),
        bands=dict(lookread.BANDS), rules=rule_texts(), seeds=seeds, a1_over=a1_over,
        episodes=eps, counts_look=c1, counts=c2, counts_but_A1=c3, counts_but_A1_look=c13,
        counts_look_reason=None if c1 is not None else READING_1["not_counted"] if done else rec.get("reason"),
        counts_equal_bar=equal, outcomes=outcomes, unread_reasons=dict(sorted(reasons.items())),
        failures_by_term=dict(sorted(terms.items())), look_failures_by_term=dict(sorted(look_terms.items())),
        look_faults=[dict(seed=ep["seed"], fault=ep["look_fault"]) for ep in eps if ep["look_fault"]],
        short_air_runs=dict(episodes_read=len(short), episodes_with_any=sum(1 for n in short if n), total=sum(short)),
        policy=_policy_block(blocks), g9_ref=cited,
        sidecars=dict(root=Path(os.path.relpath(side_root, out)).as_posix(), sha256=hashes),
        sources={**panels.module_sha256(LookRow), **lookread.sources(), "bar": bar.bar_sources()["sha256"]})
    doc["inputs_sha256"] = s2io.sha256_json(dict(bar=doc["bar_record"]["sha256"], sidecars=hashes,
                                                 sources=doc["sources"], look_set=bool(look_set)))
    return json.loads(json.dumps(s2io.plain(doc)))  # (what the file will hold)


def companion_path(out, set_name: str, body: str, block: str, row: str, behavior: str, name: str,
                   control: str | None = None) -> Path:
    """``<out>/<set>/<body>/<block>/look/<row>/<behaviour>/<condition>.json``: beside the bar's
    ``eval/<row>/<behaviour>/<condition>.json``."""
    return bar.block_dir(out, set_name, body, block, control) / "look" / row / behavior / f"{name}.json"


def stale_path(path, sha256: str) -> Path:
    """Where a companion that is not its bar record's any more is moved to (``RULES["sidecar"]``):
    ``<companion>.superseded/<first 12 hex of the file's sha256>.json``, beside it as the bar keeps a record it moved
    aside (``bar.RULES["no_overwrite"]``), so that a reader which skips ``.superseded`` directories, as the bar's
    and this module's do, never takes it for a current companion."""
    path = Path(path)
    return path.with_name(path.name + ".superseded") / f"{str(sha256)[:12]}.json"


def _refused(q: Path, rec: dict, path: Path, out: Path, reason: str) -> dict:
    """The summary's cell of a bar record that gets no companion (``RULES["sidecar"]``), after dealing with the
    file at its companion's path: ``companion`` is ``none`` (there is none), ``kept`` (it names the bar record as
    it is: ``record`` is its path) or ``moved_aside`` (it is another bar record's, or cannot be read:
    ``moved_to``)."""
    bar_sha = s2io.sha256_file(q)
    state, kept, moved = "none", None, None
    if path.is_file():
        old = _read_json(path)
        if old is not None and (old.get("bar_record") or {}).get("sha256") == bar_sha:
            state, kept = "kept", Path(os.path.relpath(path, out)).as_posix()
        else:
            dest = stale_path(path, s2io.sha256_file(path))
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, dest)
            state, moved = "moved_aside", Path(os.path.relpath(dest, out)).as_posix()
    return dict(row=rec["row"], condition=rec["key_name"], status=rec.get("status"),
                bar_record=Path(os.path.relpath(q, out)).as_posix(), bar_record_sha256=bar_sha, reason=reason,
                companion=state, record=kept, moved_to=moved)


def collect(out, set_name: str, body: str, block: str = bar.HEADLINE_BLOCK, control: str | None = None,
            g9_ref=None) -> dict:
    """The companion records of every bar record of a ``LookRow`` row under
    ``<out>/<set>/<body>/<block>/eval/`` (``companion``), and their summary ``.../look/summary.json`` (returned).
    ``g9_ref``: the citation of the G9-ref record that let the evaluation run (``run`` gives it), which the companion
    of a record "not run" on a look set cites (the lead's ruling 42 of 17 Nov 2026).  A companion whose bar record,
    sidecars and sources are unchanged is left as it is, unless it cites no G9-ref record where one is now given.
    A record with an episode whose sidecar is missing or is not that episode's gets no companion: a companion of
    another bar record found at its path is moved aside (``stale_path``; one that names the bar record as it is
    stays), the summary lists the record under ``refused_records``, every other record's companion is written, then
    ``LookBarError`` lists the refusals."""
    out = Path(out)
    bdir = bar.block_dir(out, set_name, body, block, control)
    edir, side_root, ldir = bdir / "eval", bdir / "look_partial", bdir / "look"
    look_set = look_set_of(set_name) is not None  # (a version of a look set is one: the lead's ruling 31, point 3)
    cells, refused, refused_cells = [], [], []
    reading_1 = READING_1["walk_turn"] if look_set else READING_1["not_counted"]
    if not edir.is_dir():  # (no bar record yet: nothing is read and nothing written, not even the lock)
        return dict(schema=SUMMARY_SCHEMA, set=set_name, body=body, block=block, control=control, look_set=look_set,
                    look_sets=list(LOOK_SETS), reading_1=reading_1, records=[], refused=[], refused_records=[])
    with s2io.Lock(out / ".lock"):
        for q in sorted(edir.glob("**/*.json")):
            parts = q.relative_to(edir).parts
            if parts[0] == "traces" or q == edir / "summary.json" or any(x.endswith(".superseded") for x in parts):
                continue
            rec = _read_json(q)
            if not is_look_record(rec):
                continue
            path = companion_path(out, set_name, body, block, rec["row"], rec["behavior"], rec["condition"], control)
            try:
                doc = companion(q, rec, out, side_root, look_set, g9_ref)
            except LookBarError as exc:  # (the lead's ruling 12 of 17 Nov 2026: a refusal's text in the summary
                why = _no_paths(str(exc), (out, out.resolve()))  # names no absolute path: relative to ``out``)
                refused.append(why)
                refused_cells.append(_refused(q, rec, path, out, why))
                continue
            old = _read_json(path)
            if old is not None and old.get("inputs_sha256") == doc["inputs_sha256"] and (
                    old.get("g9_ref") is not None or doc["g9_ref"] is None):
                doc = old
            else:
                doc.update(provenance=s2io.git_state(own_outputs=[str(out.resolve())]),
                           versions=panels.library_versions(), blas_threads=s2io.blas_threads(), at=s2io.now_utc())
                doc["sha256"] = s2io.sha256_json(doc)  # of the record without this key
                s2io.write_json(path, doc)
            cells.append(dict(
                row=doc["row"], condition=doc["condition"], status=doc["bar_record"]["status"],
                record=Path(os.path.relpath(path, out)).as_posix(), bar_record=doc["bar_record"]["path"],
                counts_look=doc["counts_look"], counts=doc["counts"], counts_but_A1=doc["counts_but_A1"],
                counts_but_A1_look=doc["counts_but_A1_look"], outcomes=doc["outcomes"],
                unread_reasons=doc["unread_reasons"], failures_by_term=doc["failures_by_term"],
                look_failures_by_term=doc["look_failures_by_term"], look_faults=len(doc["look_faults"]),
                short_air_runs=doc["short_air_runs"], a1_over=doc["a1_over"],
                cross_version=(doc["policy"] or {}).get("cross_version"),
                fingerprint_max_abs_diff=(doc["policy"] or {}).get("fingerprint_max_abs_diff"),
                g9_ref=doc.get("g9_ref")))
        summary = dict(schema=SUMMARY_SCHEMA, set=set_name, body=body, block=block, control=control,
                       look_set=look_set, look_sets=list(LOOK_SETS), reading_1=reading_1, records=cells,
                       refused=refused, refused_records=refused_cells,
                       rules={k: RULES[k] for k in ("readings", "outcome", "counts", "sidecar")},
                       provenance=s2io.git_state(own_outputs=[str(out.resolve())]), written_at=s2io.now_utc())
        summary = json.loads(json.dumps(s2io.plain(summary)))
        summary["sha256"] = s2io.sha256_json(summary)  # of the summary without this key
        if cells or refused:
            s2io.write_json(ldir / "summary.json", summary)
    if refused:
        raise LookBarError("\n".join(refused))
    return summary


def run(set_name: str, body: str, out, row_names, conditions=None, block: str = bar.HEADLINE_BLOCK, procs: int = 2,
        **kw) -> dict:
    """``bar.evaluate`` of ``row_names`` (``rows()``: every row a ``LookRow`` that writes its sidecars under
    ``<out>/<set>/<body>/<block>/look_partial``) on the set, without attribution unless ``attribute`` is given,
    then ``collect`` (given the G9-ref citation, which the companions of conditions not run cite).  ``kw`` goes to
    ``bar.evaluate`` (``episodes``, ``task``, ``rates``, ``root``, ``stage1_run``, ...).  Returns the look summary
    with the bar's own summary under ``bar`` and, under ``g9_ref``, the G9-ref record that let the rows run on a look
    set (None where the bar's own check did).  The evaluation runs inside ``look_set_rule`` (``RULES["look_sets"]``):
    on a look set the rows of ``LOOK_SET_ROWS`` run only once its G9-ref has passed; any other set or row is the bar's
    own rule of what may run where (``bar.RULES["checks"]``), unchanged."""
    out = Path(out)
    control_ = kw.get("control")
    side = bar.block_dir(out, set_name, body, block, control_) / "look_partial"
    names = list(row_names)
    built = rows(body, names, sidecar=str(side), stage1_run=kw.get("stage1_run", bar.STAGE1_RUN))
    kw.setdefault("attribute", "none")
    with look_set_rule(set_name, body, names, kw.get("root", setio.ROOT)) as rule:
        doc = bar.evaluate(set_name, body, out, built, conditions, block, procs, **kw)
    return dict(collect(out, set_name, body, block, control_, g9_ref=rule.cited), bar=doc, g9_ref=rule.cited)


def summary_text(doc: dict) -> str:
    """The look summary as lines: per row and condition the readings' counts and the outcomes, then the records
    that were refused a companion."""
    lines = [f"{doc['set']} {doc['body']} {doc['block']}: {doc['reading_1']}"]
    for c in doc["records"]:
        def n(x):
            return "-" if x is None else f"{x['succeeded']}/{x['episodes']}"

        o = c["outcomes"]
        lines.append(f"  {c['row']} {c['condition']}: with the look {n(c['counts_look'])}, reading 2 {n(c['counts'])}, "
                     f"reading 3 {n(c['counts_but_A1'])}; read {o['read']}, fell {o['fell']}, unread {o['unread']}, "
                     f"error {o['error']}" + (f"; {c['look_faults']} look faults" if c["look_faults"] else ""))
    for c in doc.get("refused_records") or []:
        if c["companion"] == "kept":
            what = f"the companion made earlier of this bar record is kept ({c['record']})"
        elif c["companion"] == "moved_aside":
            what = f"the companion of another bar record was moved aside to {c['moved_to']}"
        else:
            what = "it has no companion"
        lines.append(f"  {c['row']} {c['condition']}: refused; {what}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _csv(text: str | None) -> list[str] | None:
    return None if text is None else [x for x in (s.strip() for s in text.split(",")) if x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.lookbar", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("run", "collect"))
    ap.add_argument("--set", dest="set_name", required=True, help="the reference set (V6, V6noF5, ...) or S1")
    ap.add_argument("--body", required=True)
    ap.add_argument("--rows", default=None, help="comma-separated rows: R0L@10ms,R5/42@10ms")
    ap.add_argument("--conditions", default=None, help="comma-separated keys, names or the bar's groups")
    ap.add_argument("--block", default=bar.HEADLINE_BLOCK, choices=tuple(bar.BLOCKS))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=bar.RUNS)
    ap.add_argument("--task", type=int, default=None, choices=tuple(bar.BUDGET_CORE_H))
    ap.add_argument("--unit", default=None)
    ap.add_argument("--cost", default=None, help="the committed COST.md with the rates (default <out>/COST.md)")
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--no-traces-for", default=None)
    ap.add_argument("--supersede", action="store_true")
    ap.add_argument("--supersede-reason", default=None)
    ap.add_argument("--dev-episodes", default=None)
    ap.add_argument("--dev-reason", default=None)
    ap.add_argument("--set-root", default=str(setio.ROOT))
    ap.add_argument("--stage1-run", default=str(bar.STAGE1_RUN))
    ap.add_argument("--policies", default=None, help="a JSON file: policy row name -> {job_dir, steps}")
    ap.add_argument("--row-module", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        s2io.require_one_thread()
        if a.command == "collect":
            print(summary_text(collect(a.out, a.set_name, a.body, a.block)))
            return 0
        if a.rows is None:
            ap.error("run needs --rows")
        if a.policies is not None:
            os.environ["STAGE2_POLICIES"] = str(Path(a.policies).resolve())
        for mod in a.row_module:
            importlib.import_module(mod)
        rates = None
        if a.task is not None:
            rates = bar.read_cost(a.cost or Path(a.out) / "COST.md")["rates"]
        doc = run(a.set_name, a.body, a.out, _csv(a.rows), _csv(a.conditions), a.block, a.procs,
                  traces=not a.no_traces, episodes=a.episodes, task=a.task, supersede=a.supersede,
                  stage1_run=a.stage1_run, root=a.set_root, unit=a.unit, rates=rates,
                  supersede_reason=a.supersede_reason, dev_episodes=a.dev_episodes,
                  no_traces_for=_csv(a.no_traces_for) or (), dev_reason=a.dev_reason)
        print(bar.summary_text(doc["bar"]))
        print(summary_text(doc))
        s = doc["bar"]["sessions"][-1]
        if s.get("incomplete"):
            print("lookbar: episodes failed to run in the bar's records (a policy row's unread episode is one of "
                  "them there; its companion holds it as unread): " + "; ".join(s["incomplete"]), file=sys.stderr)
            return 4
        return 0
    except bar.BudgetCut as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (bar.BarError, s2io.S2Error, tune.InputError, tune.PoolError, ValueError) as exc:
        print(f"lookbar: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        if type(exc).__name__ != "CheckpointError":  # (s2load's refusal of a checkpoint: not judged)
            raise
        print(f"lookbar: {exc}", file=sys.stderr)
        return 1


__all__ = ["COMPANION_SCHEMA", "DELAYED_KEYS", "LOOK_SETS", "LOOK_SET_ROWS", "LookBarError", "LookRow",
           "LookSetRefused", "LookSetRule", "RULES", "SIDECAR_SCHEMA", "WARNED_CAUSE", "air_runs", "clock_block",
           "collect", "companion", "companion_episode", "companion_path", "counts_of", "episodes", "g9_ref_of",
           "hold_for_life", "is_look_record", "is_unreadable", "is_warned", "leave_of", "leave_resolve", "look_of",
           "look_row_of", "look_set_of", "look_set_placement", "look_set_row", "look_set_rule", "main",
           "on_command_at_stop", "open_set", "placed", "placement_key", "placement_of", "record_row_sha256", "require_g9_ref", "rows", "rule_texts", "run",
           "sidecar_path", "stale_path", "state_digest", "summary_text"]

if __name__ == "__main__":
    from designlab.stage2 import lookbar as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
