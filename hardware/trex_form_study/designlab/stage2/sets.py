"""Stage 2 reference sets: their definitions, and building, verifying and freezing a set directory.

Simulation only: prescribed references on the round-two model; nothing is adopted.

A *reference set* is a directory ``stage2/sets/<set>/<body>/`` in Stage 1's run-directory layout (``setio``), so that
Stage 1's planner, track and tuner load it unchanged: periodic walks in ``refs/<name>.npz`` (Stage 1's reference
format) with a twin in the look ladder's format in ``look/<name>.npz``, moves (a start, speed changes, stops, finite
turns) in ``transitions/<name>.npz``, and the records ``refs.json``, ``transitions.json`` and ``set.json``.  The arrays
are git-ignored and the records are committed.  This module works for any set of ``SETS`` on any body of
``designlab.stage1.BODIES``: nothing here names a body, and a set is its entry in ``SETS`` and nothing else.

1. Definitions (``STAGE2_75_PLAN.md`` A1 to A3)
-----------------------------------------------
``SETS``: ``V0`` (control: today's walk at every speed, ``v0_spec``), ``F25F9`` (control: the pull request's tuned
walk, ``tuned_f9_spec``, copied verbatim) and ``F25F9cl`` (candidate: that walk with duty 0.59 and lift 0.095 H).
``items(set, body)`` lists what a set holds, walks first: the nominal walk and one walk per speed of
``setio.SPEEDS``; ``start``; ``nominal_to_v<v>`` for every speed; the two-touchdown stop ``stop2_v0.4`` and Stage 1's
one-step stop ``stop1_v0.4`` (a labelled comparison); ``turn90_<rate>_<left|right>`` for every rate of
``setio.TURN_RATES``.  Names are made and parsed by ``setio`` only.  An item's ``definition`` is its identity as
asked: the specs, every builder argument, the body's model sha256 and the rules' constants.  A set whose nominal walk
is not a rung but whose 0.40 m/s walk is starts from standing straight into that walk (``start_v0.4``, made by the
``start`` item; ``set.json`` ``lead_in``; ``RULES["S2S-lead-in"]``, the plan's A8 as amended).

2. Building (``build``; the plan's A4 to A12)
---------------------------------------------
One item is built by ``build_item`` in one process, inside ``s2io.r2()`` (accuracy r2 on entry and on exit), on a
body rebuilt for it (``look.source``; a move's arrays never depend on what the process built before):

* **A walk** (``walk_arrays``): ``look.generate``, ``gait.inverse_dynamics``, ``gait.summarize``, Stage 1's arrays
  (``refs.arrays``) and the look ladder's (``look.arrays``) from the one generation, with the look ladder's measures,
  price and flags (``look.measure``, ``look.price``, ``look.flags``).  Gates that stop the build: the two formats
  hold one motion (``q``, ``qd``, ``qdd`` equal); where Stage 1's run holds a walk of the same spec the Stage
  1-format file has its sha256; where the look ladder holds a rung of the same spec the look-format file has its.
* **A move** (``moves``): generated, summarised (the plan's A7 gates: ``pass_``) and screened at Stage 0 with the
  set's own screen (``setscreen.screen_move``: the plan's A12 gates, ``stage0.blocks_freeze``), so that the
  pre-registered consequences are applied where the move is built: a start, change or stop that fails is retried by
  ``RULES["S2S-fallback"]`` (A8) and no other way; a turn follows the stride rule (A11; ``moves.pick_stride_factor``
  once per rate, both directions at the chosen factor, ``moves.step_down`` on the built pair).
* **What has no reference** is an entry too: ``available: false`` with ``reason`` and ``numbers`` and no file (a walk
  over a Stage 0 gate is dropped from the ladder; a move outside its gates; a turn rate without a stride factor or
  outside the heading rule), listed in ``set.json`` ``not_available``.  No constant is changed for it.

``build`` holds the set's lock, sweeps what a killed run left, merges finished items found in ``partial/``, then
runs the missing items, walks before moves, each in a fresh spawned process (``--procs`` at a time).  A worker writes
its arrays into a scratch directory under ``partial/``, moves them into place, writes ``partial/<item>.json`` and
returns; the parent merges that into ``refs.json``, ``transitions.json`` and ``set.json`` (each written atomically)
and deletes the partial in the same step.  An item is kept when its entry holds the definition ``items`` gives now,
was built from the walks the set holds now, and its files have the recorded sha256; an entry with no reference is
kept while the Stage 2 modules that decided it are today's (or the set is frozen).  A rebuild that reproduces an
entry's arrays leaves the entry, and its provenance, as they were.  A worker writes into the set only while the
set's lock names the builder that started it, and a first merge that was killed is made again.  Wall times go to ``logs/sets.log`` in the set
directory and into no record.  **On a frozen set** an entry of the latest manifest is never written again: a missing
array is rebuilt and put in place only when its sha256 is the manifest's, and a difference fails the build with
both hashes; entries the manifest does not hold may still be added.

3. ``verify``, ``freeze``, ``status``
-------------------------------------
``verify`` reads the set as it is (no array is rebuilt) and returns ``{ok, failures, notes}``: the files against
their entries and the latest manifest, names, definitions, the moves' gates, the stops' and turns' own readings, the
walks' two formats, the hash gates, the labels (a walk's ``label``, every entry's ``margin_rule``), the screen
(``screen.json``: a current block for every available entry, none blocking the freeze) and provenance; its notes
also give the bar's kinematic check of the references where the set directory holds its record (``refcheck.json``,
made by ``python -m designlab.stage2.bar refcheck --out <the sets' root>``; a report, not a gate).  ``freeze`` (the
lead's act) appends a manifest of every available entry to ``set.json`` ``freezes``; it needs ``verify`` to pass, a
passing ``stage2/sets/legacy_check.json`` of today's ``moves.py`` and every built entry made by that ``moves.py``,
and one source state on a clean tree unless a flag says otherwise; it never drops or changes an entry an earlier
freeze holds, a later freeze that adds references says when it was made, and it cites the kinematic check's record
where there is one (``RULES["S2S-freeze"]``).  The one entry a freeze drops is one the lead marked not available on
the G9-ref record that named it (``mark``; ``MARK_RULE``, the lead's ruling 32 (A) of 17 Nov 2026): the freeze
records it under ``dropped`` with the mark and that record; earlier freezes are not rewritten.  ``verify`` names
every stop of the set: those ``set.json`` ``blocked`` lists and any the records show that an older ``blocked``
leaves out (``blocks_in_records``; ruling 36).

4. The look sets (``LOOK_SETS``; ``STAGE2_TRACKB_PLAN.md`` section L)
--------------------------------------------------------------------
``V6`` (the look the owner picked: every look feature in the walks) and ``V6noF5`` (V6 without F5's toe-off, for the
tracker fit) stand beside ``SETS`` in a registry of their own, and every function here takes either (``_set``).  They
are built, verified and frozen as the other sets are, with these differences (``LOOK_RULES``; the look's laws
themselves are ``designlab.stage2.moves_look``'s, loaded by name where a look set is built):

* a walk is the look ladder's chain on today's walk at that speed (``v6_spec``), F9's tail phase from the ladder's
  record at its two speeds and from ``stage2/sets/V6/tail_phase.json`` at the others (``make_tail_phases``; the
  ``tail-phase`` command); its file holds a window of K strides (``look_walk_arrays``), and on a pick the look-format
  files of ``nominal`` and ``v0.4`` must be the ladder's V6 rungs (``ladder_twins``);
* a move is bound to the look's laws and to the windows of the walks it joins (``_Job.move``:
  ``moves_look.bind_move``) and generated by ``moves_look.generate`` (a causal move's preview grid from the frames
  up to its command: ruling 28, ``moves_look.RULES["S2L-causal-grid"]``), a turn's stride factor is picked on the
  bound turns (``moves_look.pick_stride_factor``), and its entry holds the look's keys beside the others (the
  move's instant in ``move.command_phase`` for a start and a change too; ``look``; ``join_events``);
* a causal move's warm-up reading of ``join_command`` is held to 5.0 mm, the owner's decision P42
  (``moves_look.LOOK_WARMUP_ZMP_LIMIT_MM``, ``moves_look.RULES["S2L-P42-warmup"]``, read by
  ``moves_look.entry_extra``); every other limit and gate is the round's, whose three sets keep 4.0 mm.  The entry's
  ``join_command.limits`` holds the limit used, and ``transitions.json`` ``causal_limits`` says so
  (``look_causal_limits``);
* ``set.json`` names the join code's independent review by path and sha256 (``JOIN_REVIEW``; ``LOOK_RULES
  ["S2L-join-review"]``; STAGE2_TRACKB_PLAN.md L12 with rulings 28 to 30), the cut in force (none) and why
  (``look_review``), and every move whose join gate fails with its numbers (``joins_failed``);
* the turn in place (``inplace_0.5_<dir>`` with its start and its stop; ``_produce_inplace``) is an optional group
  per direction: built when it is named (``--only inplace_0.5_left``), not missed by ``verify`` when it is not
  built.  The set's screen (``setscreen``) reads every entry of ``refs.json`` as a walk with a look-format file and
  cannot read a periodic turn, so a set that holds the turn in place cannot be screened as a whole until that
  module learns it; its three files are screened where they are built (``_screen_file``);
* a head body (``<pick>-X``) walks its pick's look on its own model.

This module is in no record key.

Run from hardware/trex_form_study:
    python -m designlab.stage2.sets build  --set V0 --body 75-M-H-10 [--only NAME ...] [--procs 2] [--root DIR]
    python -m designlab.stage2.sets verify --set V0 --body 75-M-H-10 [--root DIR]
    python -m designlab.stage2.sets status --set V0 --body 75-M-H-10 [--root DIR]
    python -m designlab.stage2.sets freeze --set V0 --body 75-M-H-10 [--allow-mixed] [--allow-dirty]
        [--change-after-result REASON | --after-check-episodes REASON | --before-any-episode] [--refcheck FILE]
        [--root DIR]
    python -m designlab.stage2.sets mark   --set V6 --body 75-M-H-10 --name NAME --reason TEXT --g9-ref FILE
        [--root DIR]     (the lead's: an entry a G9-ref record named, not available; the next freeze drops it)
    python -m designlab.stage2.sets tail-phase --body 75-M-H-10     (the look sets' F9 tail phases on a pick)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # before any numpy import: one core per process (bit-reproducible arrays)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402

from .. import envelope, gait, sizing, stage1, statics  # noqa: E402
from ..stage1 import plant, refs, transitions  # noqa: E402
from . import look, moves, s2io, setio, setscreen  # noqa: E402

STUDY = s2io.STUDY
SCHEMA = 1
STAGE1_RUN = STUDY / "stage1" / "runs" / "2026-11-08"
LOOK_REFS = STUDY / "stage2" / "look" / "look_refs.json"
LOOK_SCREEN = STUDY / "stage2" / "look" / "screen.json"
LEGACY_CHECK = setio.ROOT / "legacy_check.json"
#: The pull request's script, whose ``_tuned_f9_spec`` is copied here (``tuned_f9_spec``); read by ``verify`` only.
PR_SCRIPT = STUDY / "stage2" / "jetson_plem" / "jetson_plem_study.py"
#: The Stage 2 modules that make an entry (``s2io.built_by``).
MODULES = ("sets.py", "moves.py", "setscreen.py", "screen.py", "look.py", "setio.py", "s2io.py")
PARTIAL = "partial"
SCRATCH_PREFIX = "build-"
LOG_FILE = "logs/sets.log"
#: The bar's kinematic check of a set's references, where it is kept beside the set's records (``refcheck_state``):
#: ``python -m designlab.stage2.bar refcheck --set S --body B --out <the sets' root>`` writes it there.
REFCHECK = "refcheck.json"
WALK_GENERATOR = "look.generate + gait.inverse_dynamics + refs.arrays"
#: The item kinds, and the builders of ``moves`` they call.
KINDS = ("walk", "start", "change", "stop_two_step", "stop_one_step", "turn")
#: The look sets (``LOOK_SETS``; section 1): the Stage 2 module that holds their laws, loaded by name where a look
#: set is built (``_ml``); their walks' generator; the kinds of the turn in place's three references (the periodic
#: gait, its start from the stand and its stop to it: one group per direction).
LOOK_MODULE = "moves_look.py"
LOOK_WALK_GENERATOR = "look.generate + look.look_inverse_dynamics + Stage 1's keys with window_strides (moves_look.walk_arrays)"
INPLACE, INPLACE_START, INPLACE_STOP = "inplace", "inplace_start", "inplace_stop"
LOOK_KINDS = (INPLACE, INPLACE_START, INPLACE_STOP)
#: The turn in place (STAGE2_TRACKB_PLAN.md L13): its rate (rad/s), the hold strides past Stage 1's start window, and
#: which of them is the reference (the second).
INPLACE_RATE = 0.5
INPLACE_HOLD = 4
INPLACE_CUT = 2
NO_PERIODIC = "the turn in place is not periodic within its gate"
#: F9's tail phases of the look sets' walks: the look ladder's own at its two speeds (``KNOTS``), this record's at
#: the other rungs (``tail_phases``; ``RULES["S2L-tail-phase"]``).
LOOK_TAIL = STUDY / "stage2" / "look" / "tail_phase.json"
SET_TAIL = setio.ROOT / "V6" / "tail_phase.json"
KNOTS = {setio.NOMINAL: "nominal", setio.walk_name(setio.BAR_SPEED): "v040"}
#: The gates of a turn's entry that are the heading rule (the plan's A10): a turn that fails these and no other has
#: "no reference within the heading rule".
HEADING_GATES = ("heading_total", "heading_at_bar", "heading_at_slowest_clock")
NO_HEADING = "no reference within the heading rule"
NO_TWIST = "no reference within the thigh twist and the leg drives"
NO_GATES = "no transition within the gates"
#: A test's kill point (``STAGE2_SETS_FAIL_AT=<point>[:<item>]``): the process exits at once there, as if killed.
FAIL_AT = "STAGE2_SETS_FAIL_AT"
FAIL_POINTS = ("npz", "partial", "refs.json", "transitions.json", "set.json", "deleted")

#: The independent review of the join code that STAGE2_TRACKB_PLAN.md L12 requires before a cut (ruling 19), by
#: path (from the study directory) and sha256, as the lead's rulings 28 to 30 of 17 Nov 2026 name it: every look set
#: built after it names it in ``set.json`` ``rules`` (``LOOK_RULES["S2L-join-review"]``) and ``look.join_review``.
JOIN_REVIEW = dict(path="stage2/design/checkpoint/reviews/roundC__review_JR_17nov.md",
                   sha256="b3ff457673d537f65d8821c1910c27b6bebc7eb27acd43f0d0e0e96d9424212a")
#: The review's probe of cut 3's leave point on the causal grid, per pick, as a record (ruling 29; the review's
#: scratch readings of V6's turns, prescribed motion): a body it lists takes no cut 3, because the failing reading is
#: over its limit there too.  Nothing in this module names a body: the record does.
CUT3_PROBE_FILE = setio.ROOT / "V6" / "cut3_probe.json"

RULES = {
    "S2S-sets": (
        "three reference sets per body, each a directory stage2/sets/<set>/<body>/ in Stage 1's run-directory layout: "
        "V0 (control: today's walk), F25F9 (control: the pull request's _tuned_f9_spec, verbatim) and F25F9cl "
        "(candidate: that spec with duty 0.59 and lift 0.095 H at every speed) (STAGE2_75_PLAN.md A1)"),
    "S2S-speeds": (
        "walks nominal (sizing.NOMINAL's Froude number) and v0.3, v0.35, v0.4, v0.45, v0.5 (A2).  V0: sizing.NOMINAL "
        "at the nominal speed; look.speed_spec between sizing.NOMINAL and sizing.BRISK below BRISK's speed and "
        "between sizing.BRISK and envelope.FAST from there to FAST's speed; above FAST's speed FAST's stride, duty "
        "and lift at the higher Froude number, labelled new.  F25F9cl: every rung other than 0.40, 0.45 and 0.50 m/s "
        "labelled new"),
    "S2S-margin": "duty and lift set after results on seeds 3042-3061 (JPS 2.3)",  # (every entry's margin_rule: A3)
    "S2S-crouch": (
        "every walk of a set and body has one crouch (a share of the hip height), read from its spec and recorded as "
        "set.json crouch_frac and in every walk entry; a move joins two walks of one crouch (one LIPM frequency)"),
    "S2S-process": (
        "every array is made inside s2io.r2() (gait accuracy r2, checked on entry and on exit) with one BLAS thread, "
        "each item in a fresh process on a body rebuilt for it (look.source); a walk's inverse dynamics, summary and "
        "both array formats come from the one generation in the one block"),
    "S2S-formats": (
        "a walk is written twice from one generation: refs/<name>.npz (Stage 1's reference format, which the bar "
        "plays and a freeze lists) and look/<name>.npz (the look ladder's format, which the Stage 0 screen and the "
        "clips read); the build stops unless q, qd and qdd of the two are equal"),
    "S2S-twins": (
        "where Stage 1's run (stage1/runs/2026-11-08/<body>) holds a walk of the same spec, or a transition between "
        "walks of the same specs made with Stage 1's own schedule, the set's file must have its sha256; where the "
        "look ladder (stage2/look/look_refs.json) holds a rung of the same spec, the look-format file must have its "
        "sha256; a difference stops the build with both hashes (STAGE2_75_PLAN.md A4).  V0 owes them: on a body that "
        "has a Stage 1 run record its nominal and start must be Stage 1's own files, and on a body the look ladder "
        "holds its look-format nominal and v0.4 must be the ladder's rungs; a twin that is not found there stops the "
        "build and fails sets verify (SetDef.stage1_twins, ladder_twins)"),
    "S2S-stage0": (
        "the Stage 0 gates that block a freeze (STAGE2_75_PLAN.md A12) are read where a reference is built, with the "
        "set screen's own functions: a walk by setscreen.judge_walk on its look.price, a move by "
        "setscreen.screen_move on its file; the set's screen.json is made afterwards by setscreen and must agree "
        "(sets verify)"),
    "S2S-fallback": (
        "a start, change or stop that fails a gate of its entry (A7) or a blocking Stage 0 gate (A12) is retried in "
        "this order and no other (STAGE2_75_PLAN.md A8): (1) a change with a ramp of 2 strides instead of 1 (same "
        "name; the entry records ramp_strides 2 and this rule); (2) for a ladder rung other than 0.40 m/s, a change "
        "from the neighbouring rung toward 0.40 m/s (v0.4_to_v0.45, v0.45_to_v0.5, v0.4_to_v0.35, v0.35_to_v0.3), "
        "with one stride of ramp, the rung's plan then walking through 0.40 m/s; (3) the item is recorded 'no "
        "transition within the gates' with its numbers.  A start and the stops have no fallback and nominal_to_v0.4 "
        "has step (1) only.  A failed start or nominal_to_v0.4 stops the set and goes to the lead (a set that starts "
        "into its 0.40 m/s walk, S2S-lead-in: a failed start_v0.4)"),
    "S2S-lead-in": (
        "a set whose nominal walk is a rung starts into it (start; set.json lead_in 'start').  A set whose nominal "
        "walk is not a rung (A12) but whose 0.40 m/s walk is starts from standing straight into its 0.40 m/s walk: "
        "start_v0.4, built and gated like start, with no fallback; set.json says lead_in 'start_v0.4', and so does "
        "every record of the set (an exception, labelled where its rows are printed).  The items that need the "
        "nominal walk are then not available, with that reason, and a ladder rung is reached by the fallback rule's "
        "step (2), a change from the neighbouring rung toward 0.40 m/s.  A start_v0.4 that fails a gate stops the "
        "set: it is not run.  A set with neither walk has no lead-in (lead_in null) and its conditions are not run "
        "(STAGE2_75_PLAN.md A8 as amended: the owner's decision of 15 Nov, DECISION_LOG.md P34, after the first "
        "builds of the three sets; changed after a result)"),
    "S2S-turns": (
        "a turn rate is built once for both directions (STAGE2_75_PLAN.md A10, A11): moves.pick_stride_factor on the "
        "set's 0.40 m/s walk, both directions with all passes at the chosen factor, then moves.step_down on the "
        "built pair (ik_ok in both, the worse thigh-twist use, the tighter margin of another joint, a leg drive over "
        "a Stage 0 limit in either, a neck or tail limit in either).  No factor, or a leg drive over its limit: "
        f"'{NO_TWIST}', and the reason also names the entry gates the last built pair missed.  A built turn that "
        "fails another gate (A7's, A10's heading readings, another blocking gate "
        f"of A12) has no further retry and no reference; where only the heading readings fail: '{NO_HEADING}'.  A "
        "turn over a neck or tail limit alone is kept with a1_over true"),
    "S2S-not-available": (
        "an item with no reference is an entry with available false, a reason and its numbers, and no file; it is "
        "listed in set.json not_available; a walk that fails a blocking Stage 0 gate is dropped from the ladder in "
        "this way and no move is built from or to it (STAGE2_75_PLAN.md A8, A11, A12).  No ramp, grid, gate or "
        "constant is changed for it.  Its verdict is the code's that built it (built_by): while the set is not "
        "frozen, a build by other Stage 2 modules makes it again; an array an earlier build left under its name is "
        "removed"),
    "S2S-freeze": (
        "a freeze lists every available entry: name -> kind, the file's sha256 and the sha256 of the entry's hashed "
        "subset (setio.entry_sha256), a walk's look-format sha256 beside them.  A later freeze may add entries and "
        "never drops or changes one.  It needs sets verify to pass, a passing stage2/sets/legacy_check.json made by "
        "today's moves.py, every entry a build decided built by that moves.py (A4; no flag waives it), entries built "
        "from one source state and a clean tree (else the manifest says mixed_sources or dirty; a tree git cannot "
        "read counts as dirty).  A later freeze that adds references says when it was made: change_after_result "
        "with its reason (after a registered episode of the round: 'a change after a result'), after_check_episodes "
        "true with reason (after check episodes only: no change after a result; rule R4), or before_any_episode.  "
        "Where the set directory holds refcheck.json (the bar's kinematic check of each reference: no physics, no "
        "verdict) it must be of the files being frozen, and the freeze cites its sha256, pass_ and failing "
        "conditions; sets verify prints them as notes (a report: no rule here makes it block a freeze).  "
        "On a frozen set a build never writes a frozen entry again and puts an array in "
        "place only when its sha256 is the manifest's (STAGE2_75_PLAN.md A13)"),
    "S2S-omega": "control.omega0(p, q_home, crouch_frac=<the set's crouch_frac>): the track's LIPM frequency is the set's own",
}
#: The look sets' rules (``LOOK_SETS``; STAGE2_TRACKB_PLAN.md section L).  They join ``RULES`` in a look set's records
#: and in no other set's.
LOOK_RULES = {
    "S2L-sets": (
        "the look sets: V6 (the look the owner picked at gate L1: every look feature in the walks) and V6noF5 (V6 "
        "with F5, the toe-off and the toes-up reach, off in the walks as well: the tracker fit's set), each a "
        "directory stage2/sets/<set>/<body>/ built, verified and frozen as every other set (STAGE2_TRACKB_PLAN.md "
        "L14).  A head body (<pick>-X) walks its pick's look on its own model"),
    "S2L-chain": (
        "a V6 walk at a speed is look.ladder's chain on today's walk at that speed (the V0 set's, S2S-speeds): V1 = "
        "F1/F2 (bob = look.BOB_SHARE x that walk's enforced shortening, bob_skew, bob_drop, 12 correction passes), "
        "V2 = F3 (the animal pace's swing time), V3 = F4/F5, V4 = F6, V5 = F9/F10 with F9's tail phase, V6 = F14 "
        "with the fewest whole strides lasting 10 s; F13 (the longer stride of the 75 cm animal pace) is in no walk "
        "of a set.  At the nominal speed and at 0.40 m/s on the two picks that is the look ladder's V6 rung: its "
        "look-format file must hash to the ladder's (S2S-twins)"),
    "S2L-tail-phase": (
        "F9's tail phase is the look ladder's own rule at every speed: look.tail_check on the walk's V4 (the phase "
        "that minimises the single-support stance pad's yaw moment).  At the nominal speed and at 0.40 m/s it is the "
        "ladder's recorded value (stage2/look/tail_phase.json), so that the walk is the ladder's rung; at the other "
        "rungs it is read from stage2/sets/V6/tail_phase.json, made by python -m designlab.stage2.sets tail-phase "
        "on the pick.  V6noF5 and a head body take the pick's V6 phases"),
    "S2L-noF5": "V6noF5 is V6 with F5's fields at their defaults in the walks: toe_off 0, toe_reach 0, no rate or "
                "acceleration cap, the default start and margin",
    "S2L-format": (
        "a look walk's file holds a window of K whole strides that is periodic as a whole: Stage 1's reference keys "
        "with 160 K frames and the window's stride motion, the feedforward from look.look_inverse_dynamics (F5's "
        "rolling split), then window_strides (K) and the look keys; its entry holds period_s (the stride's), "
        "window_s and strides (K).  A plan item counts windows of the file"),
    "S2L-moves": (
        "every move of a look set is built by designlab.stage2.moves with the look's laws and solver "
        "(designlab.stage2.moves_look; its rules are in transitions.json) bound to the windows of the walks it "
        "joins, and is gated, screened and given its fallbacks as every other set's (S2S-fallback, S2S-turns).  A "
        "causal move leaves its walk's window at the bar's lead (join rule J1 as designed: 2 strides after the "
        "window's first frame for a left turn, 2.5 for a right turn and for the two-touchdown stop; the entry's "
        "look.a_phase, look.a_time_s and look.a_windows, the windows of the walk's file a track plays before it).  "
        "No cut of STAGE2_TRACKB_PLAN.md L12 is taken by the builder: a move outside a join gate is recorded with "
        "its numbers, which are what L12's trigger reads"),
    "S2L-inplace": (
        f"the turn in place at {INPLACE_RATE:g} rad/s, left and right (inplace_{INPLACE_RATE:g}_<dir>, with "
        f"start_inplace_{INPLACE_RATE:g}_<dir> and stop_inplace_{INPLACE_RATE:g}_<dir>; STAGE2_TRACKB_PLAN.md L13), "
        "is the set's nominal look gait stepping in place (path speed 0) with flat feet as its own law (F5's fields "
        "at their defaults) and without F14 (one stride).  It is cut from one long plan of the same builder and "
        f"laws: the start from the stand, then {INPLACE_HOLD} hold strides more than Stage 1's start writes.  The "
        "reference is the second of those hold strides; it is gated for periodicity against the hold stride before "
        "it and the one after it (model CoM 1.0 mm, multibody ZMP 2.0 mm, joints 1e-3 rad, each mapped by the "
        "plan's own stride motion); no other plan length and no other stride is tried.  Its start is the same plan "
        "up to that stride; its stop is Stage 1's one-step stop from the reference.  Each of the three files is "
        "screened at Stage 0 as a move (setscreen.screen_move).  A gait that fails is recorded not available with "
        "its numbers, and its start and stop with it.  The three references of a direction are one optional group: "
        "a build makes them when they are named (--only), because the set's screen (setscreen) reads every entry of "
        "refs.json as a walk with a look-format file and cannot yet read a periodic turn"),
    "S2L-join-review": (
        "the join code of the look sets was reviewed independently before any cut (STAGE2_TRACKB_PLAN.md L12; "
        f"ruling 19 of stage2/design/ROUND_C_RULINGS_17NOV.md): the review's record is {JOIN_REVIEW['path']}, "
        f"sha256 {JOIN_REVIEW['sha256']}, ruled on 17 Nov 2026 (rulings 28 to 30).  Its gate 3 is built: a causal "
        "move's preview grid comes from the frames up to the command (ruling 28; moves_look.RULES S2L-causal-grid, "
        "in transitions.json).  L12's fallback, cut 3, is not taken for a body when the review's probe, on the causal "
        "grid, gives the failing reading at cut 3's leave point (the window's boundary) as over its limit too (L12's "
        "sentence settled 17 Nov 2026, labelled changed after a result: ruling 29); the move that fails at J1's leave "
        "point is then not available with its numbers (A11 for a turn; H3 for the stop; a start or nominal_to_v0.4 "
        "that fails stops the set, A8), and no limit is changed.  The probe's readings per body are recorded in "
        "stage2/sets/V6/cut3_probe.json (the review's numbers, copied); a body it lists takes no cut 3, for its V6 and "
        "V6noF5 sets on both heads (a cut is a body's, L12).  set.json look holds the review (join_review), the cut in "
        "force (cut, null for none), why (cut_reason) and the probe's readings for the body (cut3_probe), and every "
        "move whose join_command or join_b gate fails, with those numbers (joins_failed)"),
}
#: The text of ``mark`` (``sets mark``) and of a freeze that drops a marked entry (the lead's ruling 32 (A) of 17 Nov
#: 2026, after ruling 14).  It is carried by the marked entry and by the freeze that drops it, and is in no set's
#: ``rules``: the round's three frozen sets' records are not rewritten.
MARK_RULE = (
    "an entry of a frozen set that a G9-ref record names as a failing item's reference still available in the set "
    "(gates2: parts.a.still_available; the lead's ruling 14 of 17 Nov 2026) is marked not available by python -m "
    "designlab.stage2.sets mark, with the lead's reason and that record (path and sha256); the record must be a "
    "G9-ref record of this set and body (its own sha256 intact) made on the set's latest freeze.  The entry keeps "
    "its definition and provenance, gains available false, the reason, its numbers (the record's reading of it) "
    "and the mark (marked: when, why, the record, the freeze it was made on, the file it named), and names no file; "
    "set.json lists it under not_available.  The next freeze drops it and records the dropped entry with the mark "
    "and the record (freezes[].dropped); it is the only entry a freeze drops: an entry that an earlier freeze holds "
    "and the set no longer holds as a reference without such a mark is still refused.  Earlier freezes are not "
    "rewritten (ruling 32 (A) of 17 Nov 2026)")


class SetError(s2io.S2Error):
    """A set cannot be built, verified or frozen as asked."""


class GateError(SetError):
    """A built-in gate of the builder failed (two formats of one walk differ, a file that must hash to Stage 1's or
    to the look ladder's does not): the build stops."""


# ------------------------------------------------------------------------------------------------
# 1. Definitions
# ------------------------------------------------------------------------------------------------
def nominal_speed(H: float) -> float:
    """The nominal walk's speed on a body of hip height ``H`` (``sizing.NOMINAL``'s Froude number), m/s."""
    return float(sizing.NOMINAL.describe(H)["speed"])


def v0_spec(body: str, H: float, v: float | None = None) -> gait.GaitSpec:
    """Today's walk at speed ``v`` (None: the nominal walk) by ``RULES["S2S-speeds"]``."""
    if v is None:
        return sizing.NOMINAL
    brisk, fast = sizing.BRISK.describe(H)["speed"], envelope.FAST.describe(H)["speed"]
    if v < brisk:
        return look.speed_spec(H, v, sizing.NOMINAL, sizing.BRISK)
    if v <= fast:
        return look.speed_spec(H, v, sizing.BRISK, envelope.FAST)
    return replace(envelope.FAST, froude=v * v / (gait.G * H))


def tuned_f9_spec(body: str, H: float, v: float) -> gait.GaitSpec:
    """The pull request's ``_tuned_f9_spec`` (``stage2/jetson_plem/jetson_plem_study.py``), verbatim; ``verify`` and a
    test compare the two field by field at every rung."""
    fr = v * v / (gait.G * H)
    crouch = 0.07 if body.startswith("60") else 0.05
    duty = 0.58 if v < 0.36 else 0.55
    base_sr = 0.713 if body.startswith("60") else 0.863
    sr_cap = 0.90 if body.startswith("60") else 0.94
    sr = round(min(sr_cap, max(0.62, base_sr * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
    return gait.GaitSpec(
        froude=fr,
        duty=duty,
        lift_frac=0.081,
        head_amp=0.05,
        tail_amp=-0.125,
        sway_gain=1.0,
        stride_rel=sr,
        crouch=crouch,
        zmp_iters=6,
        zmp_gain=0.5,
        n=160,
    )


#: The candidate's margin (``RULES["S2S-margin"]``) and the rungs the pull request ran it at (m/s).
CL_DUTY, CL_LIFT = 0.59, 0.095
CL_RUNGS = (0.40, 0.45, 0.50)


def _f9(body: str, H: float, v: float | None) -> gait.GaitSpec:
    return tuned_f9_spec(body, H, nominal_speed(H) if v is None else v)


def _f9cl(body: str, H: float, v: float | None) -> gait.GaitSpec:
    return replace(_f9(body, H, v), duty=CL_DUTY, lift_frac=CL_LIFT)


def _v0_label(body: str, H: float, v: float | None) -> str | None:
    return "new" if v is not None and v > envelope.FAST.describe(H)["speed"] else None


def _cl_label(body: str, H: float, v: float | None) -> str | None:
    return None if v in CL_RUNGS else "new"


def _ml():
    """The look's side of the builder (``designlab.stage2.moves_look``), loaded by name as ``moves.generate`` loads
    it: a set of today's laws never reaches it."""
    return importlib.import_module("designlab.stage2.moves_look")


def base_body(body: str) -> str:
    """The pick a body walks the look of: itself, or ``<pick>`` for the head body ``<pick>-X``."""
    return body[:-2] if body.endswith("-X") else body


@functools.lru_cache(maxsize=8)
def _json_at(path: str, mtime_ns: int):
    return s2io.read_json(Path(path))


def _doc(path: Path):
    """A JSON record as it is on disk now (read once per change of the file), or None."""
    path = Path(path)
    return _json_at(str(path), path.stat().st_mtime_ns) if path.is_file() else None


def tail_phase(body: str, v: float | None) -> float:
    """F9's tail phase (rad) of a look set's walk at speed ``v`` (None: nominal) on ``body``
    (``LOOK_RULES["S2L-tail-phase"]``); ``SetError`` where no record holds it."""
    pick, name = base_body(body), setio.walk_name(v)
    if name in KNOTS:
        rec = (((_doc(LOOK_TAIL) or {}).get("bodies") or {}).get(pick) or {}).get(KNOTS[name]) or {}
        val, where = rec.get("tail_phase_F9_rad"), LOOK_TAIL
    else:
        rec = (((_doc(SET_TAIL) or {}).get("bodies") or {}).get(pick) or {}).get(name) or {}
        val, where = rec.get("tail_phase_rad"), SET_TAIL
    if val is None:
        raise SetError(f"no F9 tail phase of {pick} at {name} in {s2io.portable(str(where))}: the look sets are "
                       "defined on the bodies the look ladder holds, their head bodies included (python -m "
                       "designlab.stage2.sets tail-phase --body <pick> makes the phases of the other rungs)")
    return float(val)


def look_chain(body: str, H: float, v0: gait.GaitSpec, phase: float | None, upto: str = "V6") -> gait.GaitSpec:
    """``look.ladder``'s chain on the walk ``v0`` (``LOOK_RULES["S2L-chain"]``) up to the rung ``upto`` ("V4" needs
    no tail phase); ``body`` is the pick."""
    f = look.feature_values(body, H, "nominal", phase)
    s = replace(v0, **dict(f["F1F2"], bob=look.BOB_SHARE * look.enforced_shortening(v0)), zmp_iters=look.LOOK_ZMP_ITERS)
    s = replace(s, **f["F3"])
    s = replace(s, **f["F4F5"])
    s = replace(s, **f["F6"])
    if upto == "V4":
        return s
    if phase is None:
        raise SetError("V5 and V6 need F9's tail phase")
    s = replace(s, **f["F9F10"])
    if upto == "V5":
        return s
    s = replace(s, **f["F14"])
    return replace(s, strides=look.window_strides(s, H))


def strip_f5(spec: gait.GaitSpec) -> gait.GaitSpec:
    """``spec`` with F5's fields at their defaults (``LOOK_RULES["S2L-noF5"]``)."""
    d = gait.LOOK_DEFAULTS
    return replace(spec, **{k: d[k] for k in ("toe_off", "toe_reach", "toe_off_rate", "toe_off_accel", "toe_off_start",
                                              "toe_off_margin")})


def strip_f14(spec: gait.GaitSpec) -> gait.GaitSpec:
    """``spec`` as one stride without F14's variation (step times, gaze, tail)."""
    d = gait.LOOK_DEFAULTS
    return replace(spec, strides=1, **{k: d[k] for k in ("phase_jitter", "asym", "gaze_var", "tail_var")})


def v6_spec(body: str, H: float, v: float | None = None) -> gait.GaitSpec:
    """The V6 walk at speed ``v`` (None: nominal) on ``body`` (``LOOK_RULES["S2L-chain"]``)."""
    return look_chain(base_body(body), H, v0_spec(body, H, v), tail_phase(body, v))


def v6nof5_spec(body: str, H: float, v: float | None = None) -> gait.GaitSpec:
    return strip_f5(v6_spec(body, H, v))


def inplace_spec(set_name: str, body: str, H: float) -> gait.GaitSpec:
    """The gait of a look set's turn in place: its nominal walk with flat feet and no F14
    (``LOOK_RULES["S2L-inplace"]``)."""
    return strip_f14(strip_f5(_set(set_name).spec_of(body, H, None)))


def inplace_names(direction: str) -> dict:
    """The names of the turn in place's three references for one direction (design D5's, as ``r5lib`` reads them;
    ``setio`` makes none of them)."""
    if direction not in setio.DIRECTIONS:
        raise ValueError(f"turn direction {direction!r}: choose from {', '.join(setio.DIRECTIONS)}")
    tag = f"inplace_{INPLACE_RATE:g}_{direction}"
    return dict(ref=tag, start=f"start_{tag}", stop=f"stop_{tag}")


def inplace_of(name: str) -> dict | None:
    """``{role, direction, group}`` of a name of the turn in place (``inplace_names``), else None."""
    for d in setio.DIRECTIONS:
        for role, n in inplace_names(d).items():
            if n == name:
                return dict(role=role, direction=d, group=inplace_names(d)["ref"])
    return None


@dataclass(frozen=True)
class SetDef:
    """One reference set: its walk at a speed on a body, that walk's label, and what the set is."""

    name: str
    role: str
    spec_of: Callable[[str, float, float | None], gait.GaitSpec]  # (body, H, speed or None: nominal) -> spec
    label_of: Callable[[str, float, float | None], str | None]
    note: str
    margin_rule: str | None = None
    pr_spec: bool = False  # its walks come from tuned_f9_spec: verify compares that with the pull request's function
    #: The entries that must be Stage 1's own files, and the walks whose look-format files must be the look ladder's
    #: rungs, on every body that has a Stage 1 run record or ladder rungs (``RULES["S2S-twins"]``; the plan's A4).
    stage1_twins: tuple = ()
    ladder_twins: tuple = ()
    #: A look set: its walks carry look features (a window of K strides, ``moves_look.walk_arrays``), its moves are
    #: built with the look's laws and solver, and it has the turn in place (``LOOK_RULES``).
    look: bool = False
    features: tuple = ()  # the research's feature numbers its walks carry (its look entries' ``features``)


SETS: dict[str, SetDef] = {
    "V0": SetDef("V0", "control", v0_spec, _v0_label, "today's walk at every speed",
                 stage1_twins=(setio.NOMINAL, setio.START),
                 ladder_twins=(setio.NOMINAL, setio.walk_name(setio.BAR_SPEED))),
    "F25F9": SetDef("F25F9", "control", _f9, lambda body, H, v: None,
                    "the pull request's tuned walk (_tuned_f9_spec), verbatim", pr_spec=True),
    "F25F9cl": SetDef("F25F9cl", "candidate", _f9cl, _cl_label,
                      f"the pull request's tuned walk with duty {CL_DUTY:g} and lift {CL_LIFT:g} H at every speed",
                      margin_rule=RULES["S2S-margin"], pr_spec=True),
}


#: The look sets (STAGE2_TRACKB_PLAN.md L14), beside ``SETS``: every function of this module takes either.
LOOK_SETS: dict[str, SetDef] = {
    "V6": SetDef("V6", "look", v6_spec, _v0_label,
                 "the look the owner picked at gate L1: every look feature in the walks; inside a move the feet are "
                 "flat while the gait is changing", look=True, features=tuple(look.features_of("V6")),
                 ladder_twins=(setio.NOMINAL, setio.walk_name(setio.BAR_SPEED))),
    "V6noF5": SetDef("V6noF5", "look, for the tracker fit R2L", v6nof5_spec, _v0_label,
                     "V6 with F5 (toe-off, toes-up reach) off in the walks as well", look=True,
                     features=tuple(f for f in look.features_of("V6") if f != "F5")),
}


@dataclass(frozen=True)
class Item:
    """One thing a set holds.  ``definition`` is its identity as asked (JSON-plain); items that are built together
    (the two directions of a turn rate; the three references of a turn in place) share ``definition["group"]``.  An
    ``optional`` item is built only when it is named (``build``'s ``only``), and a set without it verifies."""

    name: str
    kind: str
    depends: tuple
    definition: dict
    optional: bool = False

    @property
    def group(self) -> str:
        return self.definition.get("group", self.name)


def _set(set_name: str) -> SetDef:
    if set_name in SETS:
        return SETS[set_name]
    if set_name in LOOK_SETS:
        return LOOK_SETS[set_name]
    raise SetError(f"{set_name!r} is not a reference set ({', '.join([*SETS, *LOOK_SETS])})")


def modules_of(st: SetDef) -> tuple:
    """The Stage 2 modules that make an entry of the set (``s2io.built_by``): ``MODULES``, and the look's for a look
    set."""
    return MODULES + ((LOOK_MODULE,) if st.look else ())


def body_facts(body: str) -> tuple[float, str]:
    """(hip height H, the model's sha256) of a Stage 1 body from its committed record: what ``look.source`` rebuilds
    and checks (``build_item`` asserts both)."""
    if body not in stage1.BODIES:
        raise SetError(f"{body!r} is not a registered body ({', '.join(stage1.BODIES)})")
    rec = plant.record(body)
    return float(rec["H"]), str(rec["xml_sha256"])


def walk_names() -> list[str]:
    return [setio.NOMINAL] + [setio.walk_name(v) for v in setio.SPEEDS]


def walk_specs(set_name: str, body: str, H: float | None = None) -> dict:
    """name -> spec of every walk of a set on a body; ``SetError`` unless they have one crouch
    (``RULES["S2S-crouch"]``)."""
    st = _set(set_name)
    H = body_facts(body)[0] if H is None else H
    out = {setio.walk_name(v): st.spec_of(body, H, v) for v in (None, *setio.SPEEDS)}
    crouches = sorted({float(s.crouch) for s in out.values()})
    if len(crouches) != 1:
        raise SetError(f"{set_name} {body}: its walks have {len(crouches)} crouches ({crouches}); a set and body has one")
    return out


def neighbour(v: float) -> float | None:
    """The neighbouring rung of ``v`` toward the bar's speed (None for the bar's speed itself)."""
    i = setio.SPEEDS.index(v)
    if v == setio.BAR_SPEED:
        return None
    return setio.SPEEDS[i + (1 if v < setio.BAR_SPEED else -1)]


def stride_constants() -> dict:
    """The stride rule's constants, as a turn item's definition holds them."""
    return dict(grid=[float(f) for f in moves.STRIDE_GRID], limit_kinematic=float(moves.TWIST_KINEMATIC),
                limit_final=float(moves.TWIST_FINAL), limit_other_joints_rad=float(moves.OTHER_JOINT_MARGIN_RAD),
                steps_down_max=int(moves.STEPS_DOWN_MAX))


TURN_RULES = ("S2A-causal", "S2A-rule-F", "S2A-fade", "S2A-ramp", "S2A-appendage", "S2A-stride", "S2A-gates")


def items(set_name: str, body: str) -> list[Item]:
    """Everything the set holds on the body, in build order: the walks, then the moves (module docstring, section
    1).  No model is built: the body's height and model sha256 come from its record."""
    H, model = body_facts(body)
    st = _set(set_name)
    specs = walk_specs(set_name, body, H)
    out = [Item(name, "walk", (), refs._plain(dict(kind="walk", spec=asdict(spec), body_model_sha256=model,
                                                    generator=LOOK_WALK_GENERATOR if st.look else WALK_GENERATOR)))
           for name, spec in specs.items()]
    gx = {name: moves.GaitX.of(name, spec, H) for name, spec in specs.items()}
    nom, bar = setio.NOMINAL, setio.walk_name(setio.BAR_SPEED)

    def plain(mv: moves.Move) -> dict:  # a move's identity: with the look's laws for a look set
        return _ml().definition_of(mv) if st.look else mv.definition()

    def of(kind: str, mv: moves.Move, depends: tuple, **more) -> Item:
        return Item(mv.name, kind, depends, refs._plain(dict(kind=kind, **plain(mv), body_model_sha256=model, **more)))

    out.append(of("start", moves.start(setio.START, gx[nom]), (nom, bar), fallback=None, lead_in=dict(
        rule="S2S-lead-in", name=setio.start_name(bar), start=plain(moves.start(setio.start_name(bar), gx[bar])))))
    for v in setio.SPEEDS:
        b, nb = setio.walk_name(v), neighbour(v)
        steps = [dict(step=1, ramp_strides=2.0)]
        if nb is not None:
            steps.append(dict(step=2, change=setio.change_name(setio.walk_name(nb), b), ramp_strides=1.0))
        out.append(of("change", moves.change(setio.change_name(nom, b), gx[nom], gx[b]),
                      (nom, b) + (() if nb is None else (setio.walk_name(nb),)),
                      fallback=dict(rule="S2S-fallback", steps=steps)))
    out.append(of("stop_two_step", moves.stop_two_step(setio.stop_name(2, bar), gx[bar]), (bar,), fallback=None))
    out.append(of("stop_one_step", moves.stop_one_step(setio.stop_name(1, bar), gx[bar]), (bar,), fallback=None,
                  label="comparison: Stage 1's one-step stop on the same walk (not causal)"))
    for label, rate in setio.TURN_RATES.items():
        for direction in setio.DIRECTIONS:
            sign = 1.0 if direction == "left" else -1.0
            d = dict(kind="turn", builder="turn", group=f"turn{setio.TURN_ANGLE_DEG}_{label}",
                     args=dict(angle=setio.TURN_ANGLE_RAD, rate=sign * rate, ramp_s=setio.RAMP_S),
                     specs=dict(a=gx[bar].plain(), b=gx[bar].plain()), causal=True, laws=moves.TODAY.name,
                     rules=list(TURN_RULES), stride_rule=stride_constants(), body_model_sha256=model)
            if st.look:
                ml = _ml()
                d.update(laws="look", rules=list(TURN_RULES) + list(ml.causal_rule_ids()),
                         leave_phase=ml.LEAVE_PHASE[f"turn_{direction}"])
            out.append(Item(setio.turn_name(label, direction), "turn", (bar,), refs._plain(d)))
    if st.look:  # the turn in place: one optional group of three references per direction (S2L-inplace)
        ml = _ml()
        spec = inplace_spec(set_name, body, H)
        for direction in setio.DIRECTIONS:
            nm = inplace_names(direction)
            sign = 1.0 if direction == "left" else -1.0
            g = moves.GaitX.of(nm["ref"], spec, H, yaw_rate=sign * INPLACE_RATE, in_place=True)
            common = dict(group=nm["ref"], gait=g.plain(), rate_rad_s=sign * INPLACE_RATE, hold_strides=INPLACE_HOLD,
                          cut_hold_stride=INPLACE_CUT, laws="look", rules=list(ml.INPLACE_RULE_IDS),
                          periodic=dict(com_mm=ml.PERIODIC_COM_MM, zmp_mm=ml.PERIODIC_ZMP_MM,
                                        joint_rad=ml.PERIODIC_JOINT_RAD), body_model_sha256=model)
            for kind, key in ((INPLACE, "ref"), (INPLACE_START, "start"), (INPLACE_STOP, "stop")):
                out.append(Item(nm[key], kind, (nom,), refs._plain(dict(kind=kind, **common)), optional=True))
    return out


def groups(its: list[Item]) -> dict:
    """group -> its items, in build order."""
    out: dict = {}
    for it in its:
        out.setdefault(it.group, []).append(it)
    return out


def lead_in_of(references: dict) -> str | None:
    """The start a set's plans begin with (``RULES["S2S-lead-in"]``), from its walk entries: ``start`` where the
    nominal walk is a rung; ``start_v0.4`` where it is not and the 0.40 m/s walk is; None where neither is, or while
    either is not built."""
    nom, bar = references.get(setio.NOMINAL), references.get(setio.walk_name(setio.BAR_SPEED))
    if nom is None or bar is None:
        return None
    if nom.get("available") is not False:
        return setio.START
    return setio.start_name(setio.walk_name(setio.BAR_SPEED)) if bar.get("available") is not False else None


# ------------------------------------------------------------------------------------------------
# 2. Twins: the files of Stage 1's run and of the look ladder that a set's file must equal
# ------------------------------------------------------------------------------------------------
def _read(path: Path):
    return s2io.read_json(path) if Path(path).is_file() else None


def stage1_walk_twin(body: str, spec: gait.GaitSpec) -> dict | None:
    """``{name, sha256}`` of Stage 1's reference of the same spec on the body (``refs.walk_specs``; the run's
    ``refs.json``), or None."""
    doc = _read(STAGE1_RUN / body / "refs.json")
    if doc is None:
        return None
    for name, s in refs.walk_specs(body).items():
        e = doc["references"].get(name)
        if s == spec and e is not None:
            return dict(name=name, sha256=e["sha256"], record=s2io.portable(str(STAGE1_RUN / body / "refs.json")))
    return None


def stage1_move_twin(body: str, mv: moves.Move, twin_a: dict | None, twin_b: dict | None) -> dict | None:
    """``{name, sha256}`` of Stage 1's transition that ``mv`` reproduces: a move of Stage 1's own schedule (a start,
    a change with Stage 1's ramp, the one-step stop) between walks that are Stage 1's own files (``twin_a``,
    ``twin_b``: their twins, None for the stand); None otherwise."""
    if mv.causal or mv.builder not in ("start", "change", "stop_one_step"):
        return None
    if mv.builder == "change" and mv.args.get("ramp_strides") != transitions.WALK_RAMP_STRIDES:
        return None
    ends = []
    for g, twin in ((mv.a, twin_a), (mv.b, twin_b)):
        if g.spec is None:
            ends.append("stand")
        elif twin is None:
            return None
        else:
            ends.append(twin["name"])
    doc = _read(STAGE1_RUN / body / "transitions.json")
    if doc is None:
        return None
    for name, pair in transitions.TRANSITIONS.items():
        e = doc["transitions"].get(name)
        if tuple(pair) == tuple(ends) and e is not None:
            return dict(name=name, sha256=e["sha256"], record=s2io.portable(str(STAGE1_RUN / body / "transitions.json")))
    return None


def stage1_has(body: str, record: str) -> bool:
    """Whether Stage 1's run holds ``record`` (``refs.json``, ``transitions.json``) of the body."""
    return (STAGE1_RUN / body / record).is_file()


def ladder_has(body: str) -> bool:
    """Whether the look ladder holds rungs of the body."""
    return bool(((_read(LOOK_REFS) or {}).get("bodies") or {}).get(body))


def twin_owed(st: "SetDef", body: str, name: str, walk: bool) -> dict:
    """What the set owes for the entry ``name`` (``RULES["S2S-twins"]``): ``stage1`` (it must be Stage 1's own file)
    and ``ladder`` (a walk's look-format file must be the look ladder's rung), each true only on a body that has
    Stage 1's record or ladder rungs."""
    return dict(stage1=name in st.stage1_twins and stage1_has(body, "refs.json" if walk else "transitions.json"),
                ladder=walk and name in st.ladder_twins and ladder_has(body))


def ladder_twin(body: str, spec: gait.GaitSpec) -> dict | None:
    """``{speed, variant, sha256}`` of the look ladder's rung of the same spec on the body
    (``stage2/look/look_refs.json``), or None."""
    doc = _read(LOOK_REFS)
    want = look._plain(asdict(spec))
    for speed, rungs in ((doc or {}).get("bodies", {}).get(body) or {}).items():
        for variant, e in rungs.items():
            if e.get("spec") == want:
                return dict(speed=speed, variant=variant, sha256=e["sha256"], record=s2io.portable(str(LOOK_REFS)))
    return None


# ------------------------------------------------------------------------------------------------
# 3. One item
# ------------------------------------------------------------------------------------------------
def walk_arrays(src: dict, spec: gait.GaitSpec) -> tuple[dict, dict, dict]:
    """One periodic walk of ``src``'s body (``look.source``) from one generation, inside ``s2io.r2()``: (Stage
    1-format arrays, look-format arrays, parts).  ``parts``: ``info`` (the generator's, less its ``look`` block),
    ``summary`` (``gait.summarize``), ``frames``, ``ik_ok``, ``ik_fail_frames``, ``rolls``, ``measured``, ``price``
    and ``flags`` as ``look.write`` computes them, and ``seconds`` (for the log).  ``GateError`` unless ``q``, ``qd``
    and ``qdd`` of the two formats are equal (the largest difference is reported)."""
    if gait.look_active(spec):
        return look_walk_arrays(src, spec)
    t0 = time.perf_counter()
    with s2io.r2():
        M = src["M"]
        g = look.generate(src, spec)
        idr = gait.inverse_dynamics(M, g)
        summary = gait.summarize(M, g, idr)
        s1 = refs.arrays(src, dict(name=None, traj=g, idr=idr, summary=summary))
        la = look.arrays(src, g)
        info = g["info"]
        rolls = {s: [dict(r) for r in g["look"]["rolls"][s] if 0 <= r["t_lift_off"] < info["window_s"]]
                 for s in statics.SIDES}
        measured = look._plain(look.measure(la, info, src["H"], crouch=spec.crouch))
        price = look._plain(look.price(src, la, info))
        flags = look.flags(spec, info, measured, price, rolls)
    worst = {}
    for key in ("q", "qd", "qdd"):
        a, b = np.asarray(s1[key]), np.asarray(la[key])
        if a.shape != b.shape:
            raise GateError(f"the two formats of one walk hold {key} of shapes {a.shape} and {b.shape}")
        if not np.array_equal(a, b):
            worst[key] = float(np.max(np.abs(a - b)))
    if worst:
        raise GateError("the two formats of one walk differ (" + ", ".join(f"{k}: largest difference {v:.3e}"
                                                                           for k, v in worst.items())
                        + "): the Stage 1-format file's inverse dynamics did not run at the look-format file's "
                          "accuracy; nothing is written")
    parts = dict(info=look._plain({k: v for k, v in info.items() if k != "look"}), summary=summary,
                 frames=len(g["q"]), ik_ok=bool(g["ik_ok"]), ik_fail_frames=len(g["ik_fail_frames"]),
                 rolls=look._plain(rolls), measured=measured, price=price, flags=flags,
                 seconds=time.perf_counter() - t0)
    return s1, la, parts


def look_walk_arrays(src: dict, spec: gait.GaitSpec) -> tuple[dict, dict, dict]:
    """``walk_arrays`` for a walk with look features (``LOOK_RULES["S2L-format"]``): a window of ``spec.strides``
    whole strides from one generation, inside ``s2io.r2()``.  The look-format arrays are ``look.arrays``' (the look
    ladder's own file); the Stage 1-format arrays are Stage 1's keys with the feedforward of
    ``look.look_inverse_dynamics`` (F5's rolling split), then ``window_strides`` and the look keys
    (``moves_look.walk_arrays``).  ``parts`` has ``walk_arrays``' keys and, beside them, ``strides``,
    ``rolling_frames`` and ``split_failed_frames``; its ``summary`` is ``gait.summarize`` on that feedforward."""
    t0 = time.perf_counter()
    with s2io.r2():
        s1, la, got = _ml().walk_arrays(src, spec)
        info = dict(got["info"], look=got["look_fields"])
        rolls = got["rolls"]
        measured = look._plain(look.measure(la, info, src["H"], crouch=spec.crouch))
        price = look._plain(look.price(src, la, info))
        flags = look.flags(spec, info, measured, price, rolls)
    for key in ("q", "qd", "qdd"):
        if not np.array_equal(np.asarray(s1[key]), np.asarray(la[key])):
            raise GateError(f"the two formats of one look walk differ in {key}; nothing is written")
    parts = dict(info=look._plain(got["info"]), summary=got["summary"], frames=got["frames"],
                 ik_ok=bool(np.asarray(la["ik_ok"])), ik_fail_frames=int(len(np.asarray(la["ik_fail_frames"]).reshape(-1))),
                 rolls=look._plain(rolls), measured=measured, price=price, flags=flags, strides=int(got["strides"]),
                 rolling_frames=int(got["rolling_frames"]), split_failed_frames=int(got["split_failed_frames"]),
                 seconds=time.perf_counter() - t0)
    return s1, la, parts


def walk_stage0(body: str, price: dict) -> dict:
    """A walk's Stage 0 verdict at build time (``RULES["S2S-stage0"]``): ``setscreen.judge_walk`` on the walk's own
    ``look.price`` (its drive ratios, ``over`` list, contact and joint-range margins)."""
    margins = (price.get("clearance") or {}).get("joint_range_margins")
    rng = setscreen.walk_range(margins, plant.usable_ranges(body))
    v = setscreen.judge_walk(dict(drives=price["drives"], over=price["over"], near=[], verdict=price["verdict"],
                                  contact=price["contact"], range=rng))
    worst = max(price["drives"].items(), key=lambda kv: kv[1]["worst"])
    kind = max(setscreen.KINDS, key=lambda k: worst[1][k])
    return s2io.plain(dict(source="setscreen.judge_walk on the walk's look.price (at build time)",
                           governing=[worst[0], kind.replace("_ratio", ""), float(worst[1][kind])],
                           within_limits=v["within_limits"], over=v["over"], blocks_freeze=v["blocks_freeze"],
                           a1_over=v["a1_over"], tightest_joint=rng["tightest"], text=v["text"]))


def _fail_at(point: str, name: str) -> None:
    """A test's kill point: with ``STAGE2_SETS_FAIL_AT=<point>`` or ``<point>:<item>`` the process exits here at
    once, with no cleanup, as a killed one does."""
    want = os.environ.get(FAIL_AT)
    if want and want in (point, f"{point}:{name}"):
        os._exit(86)


def _unavailable(name: str, kind: str, reason: str, numbers: dict, **more) -> dict:
    return dict(name=name, kind=kind, available=False, reason=reason, numbers=numbers, **more)


def scratch_dir(d: Path, group: str) -> Path:
    """Where this process writes a group's arrays before they are moved into place: ``partial/build-<group>-<pid>``,
    its own, so that a worker whose builder was killed and a builder started after it never share a directory."""
    return Path(d) / PARTIAL / f"{SCRATCH_PREFIX}{group}-{os.getpid()}"


def _held_by(d: Path, owner: int | None) -> None:
    """``SetError`` unless the set's lock still names the builder ``owner`` (a pid) that started this item: a worker
    whose builder is gone writes nothing into the set.  None: no builder to answer to (an item built on its own)."""
    if owner is None:
        return
    try:
        holder = json.loads((Path(d) / ".lock").read_text())
    except (OSError, ValueError):
        holder = None
    pid = holder.get("pid") if isinstance(holder, dict) else None
    if pid != owner:
        raise SetError(f"{d}: the set's lock names " + ("no holder" if pid is None else f"pid {pid}")
                       + f", not the builder that started this item (pid {owner}): nothing of it is written")


class _Job:
    """One group of items being built in this process: the set's records as they are, a scratch directory, and what
    the group makes (``out``)."""

    def __init__(self, set_name: str, body: str, group: str, root):
        self.set_name, self.body, self.group = set_name, body, group
        self.st = _set(set_name)
        self.dir = setio.set_dir(set_name, body, root)
        self.items = {i.name: i for i in items(set_name, body)}
        self.mine = [i for i in self.items.values() if i.group == group]
        if not self.mine:
            raise SetError(f"{set_name} {body}: no item or group {group!r}")
        self.H, self.model = body_facts(body)
        self.specs = walk_specs(set_name, body, self.H)
        self.scratch = scratch_dir(self.dir, group)
        refs_doc, tr_doc = _read(self.dir / "refs.json"), _read(self.dir / "transitions.json")
        self.refs = (refs_doc or {}).get("references", {})
        self.trans = (tr_doc or {}).get("transitions", {})
        self.out = dict(item=group, set=set_name, body=body, refs={}, transitions={}, look_entries={},
                        stride_rules={}, files={}, header=None, blocks_set=None, log=[])
        self._src = None
        self.by: dict | None = None  # the provenance of what this process built last (``built_by``)

    # --- the body
    def source(self, fresh: bool = True) -> dict:
        """The body, rebuilt (``look.source``) unless ``fresh`` is false and one is at hand; checked against the
        record the definitions were made from."""
        if fresh or self._src is None:
            src = look.source(self.body)
            if float(src["H"]) != self.H or src["model"]["file_sha256"] != self.model:
                raise SetError(f"{self.body}: the rebuilt body (H {src['H']!r}, model {src['model']['file_sha256'][:12]}...) "
                               f"is not its record's (H {self.H!r}, {self.model[:12]}...)")
            self._src = src
            if self.out["header"] is None:
                self.out["header"] = dict(model=src["model"], H_m=float(src["H"]),
                                          q_home=[float(v) for v in src["q_home"]], joints=list(src["M"].names))
        return self._src

    def built_by(self) -> dict:
        """The provenance of what is being built now (call it inside the item's ``r2()`` block); kept as ``by`` for
        an entry with no reference that this build decided."""
        self.by = s2io.built_by(list(modules_of(self.st)), own_outputs=[str(self.dir)])
        return self.by

    # --- the walks a move reads
    def walk(self, name: str) -> tuple[moves.GaitX, refs.Reference, dict, dict | None] | None:
        """(gait, reference, entry, Stage 1 twin) of a walk the set holds, its file checked against its entry and
        its entry against the definition made now; None where the set records it as not available.  In a look set
        the reference is the walk's window of K strides (``moves_look.Window``), which a move is bound to."""
        e = self.refs.get(name)
        if e is None:
            raise SetError(f"{self.set_name} {self.body}: the walk {name!r} is not built yet")
        if e.get("available") is False:
            return None
        spec = self.specs[name]
        if e.get("definition") != self.items[name].definition:
            raise SetError(f"{self.set_name} {self.body}: the walk {name!r} in refs.json is not the one defined now; "
                           "build it again first")
        path = self.dir / e["file"]
        if not path.is_file() or s2io.sha256_file(path) != e["sha256"]:
            raise SetError(f"{path} is missing or is not the file its entry records")
        if self.st.look:
            return moves.GaitX.of(name, spec, self.H), _ml().Window.load(path, spec, self.H, name), e, None
        return (moves.GaitX.of(name, spec, self.H), refs.Reference.load(path), e,
                stage1_walk_twin(self.body, spec))

    def inputs(self, names) -> dict:
        """walk -> the sha256 of its Stage 1-format file as the set records it (None: not available)."""
        return {n: (None if self.refs[n].get("available") is False else self.refs[n]["sha256"]) for n in names}

    def missing_walk(self, item: Item, names) -> dict | None:
        """The entry of a move one of whose walks the set does not hold as a rung, or None."""
        gone = [n for n in names if self.refs.get(n, {}).get("available") is False]
        if not gone:
            return None
        why = "; ".join(f"{n}: {self.refs[n].get('reason')}" for n in gone)
        return _unavailable(item.name, item.kind,
                            f"the walk {', '.join(gone)} is not a rung of the set ({why})",
                            dict(walks={n: self.refs[n].get("numbers") for n in gone}), item=item.definition,
                            group=item.group, inputs=self.inputs(names), set=self.set_name, pass_=None)

    # --- arrays and entries
    def put(self, sub: str, name: str, arrays: dict) -> str:
        rel = f"{sub}/{name}.npz"
        sha = s2io.write_npz(self.scratch / rel, arrays)
        self.out["files"][rel] = sha
        return sha

    def drop(self, sub: str, name: str) -> None:
        rel = f"{sub}/{name}.npz"
        self.out["files"].pop(rel, None)
        (self.scratch / rel).unlink(missing_ok=True)

    def note(self, what: str, seconds: float, text: str) -> None:
        self.out["log"].append(dict(what=what, seconds=float(seconds), text=text))

    def move(self, item: Item, mv: moves.Move, ref_a, ref_b, e_a: dict | None, e_b: dict | None,
             twin_a: dict | None, twin_b: dict | None, depends) -> tuple[dict, dict]:
        """Generate one move on a body rebuilt for it, write its file into the scratch directory and screen it at
        Stage 0: (entry, flags).  ``flags``: ``ok`` (the entry's gates pass and nothing blocks the freeze),
        ``over_leg_drive`` and ``a1_over`` (``setscreen.drive_flags``), ``tightest_other``.  In a look set ``mv`` is
        first bound to the look's laws and to the windows ``ref_a`` and ``ref_b`` (``moves_look.bind_move``; a
        reference of one stride, the turn in place's, is passed on as it is), generated by ``moves_look.generate``
        (a causal move on the grid of its frames up to the command: ruling 28), and its entry gains the look's keys
        (``moves_look.entry_extra``: the instant in ``move.command_phase``, ``look``, ``join_events``)."""
        t0 = time.perf_counter()
        src = self.source()
        bound = None
        if self.st.look:
            ml = _ml()
            wa, wb = (r if isinstance(r, ml.Window) else None for r in (ref_a, ref_b))
            bound = ml.bind_move(src, mv, wa, wb)
            mv = bound.move
            ref_a, ref_b = (b if w is not None else r for b, w, r in ((bound.ref_a, wa, ref_a), (bound.ref_b, wb, ref_b)))
        with s2io.r2():
            # a look set's move: moves_look.generate (a causal move on the causal grid, moves_look.RULES
            # S2L-causal-grid: ruling 28 of 17 Nov 2026); every other set's: moves.generate as it stands
            g = (_ml().generate if bound is not None else moves.generate)(src, mv, ref_a)
            entry = moves.summary(g, ref_a, ref_b, src["q_home"])
            if bound is not None:
                _ml().entry_extra(bound, g, entry)
            arrs = moves.arrays(g)
            sha = self.put("transitions", mv.name, arrs)
            other = moves.tightest_other(src, g["q"])
            prov = self.built_by()
        entry.update(file=f"transitions/{mv.name}.npz", sha256=sha)
        twin = stage1_move_twin(self.body, mv, twin_a, twin_b)
        if twin is None and twin_owed(self.st, self.body, mv.name, walk=False)["stage1"]:
            raise GateError(f"{self.set_name} {self.body} {mv.name}: the set must hold Stage 1's own transition under "
                            f"this name and Stage 1's run has none between walks of these specs ({RULES['S2S-twins']})")
        if twin is not None and twin["sha256"] != sha:
            raise GateError(f"{self.set_name} {self.body} {mv.name}: its file hashes to {sha}, Stage 1's "
                            f"{twin['name']} ({twin['record']}) to {twin['sha256']}: the move is Stage 1's own "
                            "transition and must reproduce it")
        block = setscreen.screen_move(src, self.scratch, mv.name, entry, None)
        with s2io.r2():
            df = setscreen.drive_flags(src, arrs["tau"], arrs["omega"], arrs["t"])
        v = block["verdict"]
        if bool(v["a1_over"]) != bool(df["a1_over"]):
            raise SetError(f"{mv.name}: the screen's a1_over ({v['a1_over']}) is not setscreen.drive_flags' ({df['a1_over']})")
        entry.update(
            available=True, set=self.set_name, group=item.group, item=item.definition,
            definition=refs._plain(dict(mv.definition(), body_model_sha256=self.model,
                                        ref_a_sha256=None if e_a is None else e_a["sha256"],
                                        ref_b_sha256=None if e_b is None else e_b["sha256"])),
            inputs=self.inputs(depends), a1_over=bool(v["a1_over"]), fallback=None, tightest_other=other,
            stage1_sha256=None if twin is None else twin["sha256"], equals_stage1=None if twin is None else True,
            stage1_twin=None if twin is None else twin["name"],
            stage0=dict(source="setscreen.screen_move on the built file (at build time)", governing=block["governing"],
                        within_limits=v["within_limits"], over=v["over"], blocks_freeze=v["blocks_freeze"],
                        a1_over=v["a1_over"], over_leg_drive=bool(df["over_leg_drive"]), text=v["text"],
                        tightest_joint=block["range"].get("tightest"),
                        accel_forward_lp1hz_peak_m_s2=block["accel"].get("com_forward_lp1hz_peak")),
            built_by=prov)
        entry = refs._plain(entry)
        ok = bool(entry["pass_"] and not v["blocks_freeze"])
        failed = sorted(k for k, x in entry["gates"].items() if not x)
        self.note(mv.name, time.perf_counter() - t0,
                  f"pass_ {entry['pass_']}" + (f" (failing: {', '.join(failed)})" if failed else "")
                  + f"; Stage 0: {v['text']}; governing {block['governing'][0]} {block['governing'][1]} "
                    f"{block['governing'][2]:.4f}; thigh twist {entry['hip_yaw_range_use']:.4f}"
                  + ("" if other is None else f"; tightest other joint {other['joint']} {other['end']} "
                                              f"{other['margin_rad']:.4f} rad"))
        return entry, dict(ok=ok, over_leg_drive=bool(df["over_leg_drive"]), a1_over=bool(df["a1_over"]),
                           tightest_other=other, failed=failed, blocks=v["blocks_freeze"])


def _numbers(entry: dict, flags: dict, **more) -> dict:
    """What a failed build leaves in the record: its gates, joins, histories and what blocked at Stage 0."""
    keep = ("gates", "pass_", "ik_ok", "ik_fail_frames", "join_a", "join_b", "join_command", "zmp_error_history_m",
            "zmp_consistency_history_m", "post_window_zmp_error_m", "zmp_consistency_all_m", "cop_margin_single_min",
            "min_cop_margin_single_m", "hip_yaw_range_use", "tightest_other", "reference", "turn", "frames",
            "duration_s")
    if "look" in entry:  # a look set's move: its instant, its joins and its rolling frames stay with its numbers
        keep += ("move", "look", "join_events", "rolling_frames", "rolling_split_failed_frames")
    out = {k: entry[k] for k in keep if k in entry}
    if isinstance(out.get("turn"), dict):
        out["turn"] = {k: v for k, v in out["turn"].items() if k != "stride_rule"}
    out.update(failed_gates=flags["failed"], stage0=entry.get("stage0"), built_sha256=entry.get("sha256"), **more)
    return out


def _to_unavailable(job: _Job, item: Item, entry: dict, flags: dict, reason: str, **more) -> dict:
    """A built move that has no reference: its file is not kept; its numbers are."""
    job.drop("transitions", entry["name"])
    return refs._plain(_unavailable(
        entry["name"], item.kind, reason, _numbers(entry, flags), item=entry["item"], group=item.group,
        inputs=entry["inputs"], set=job.set_name, pass_=entry["pass_"], gates=entry["gates"],
        definition=entry["definition"], built_by=entry["built_by"], **more))


def _produce_walk(job: _Job, item: Item) -> None:
    name, body, st = item.name, job.body, job.st
    spec = job.specs[name]
    v = setio.parse_name(name)["speed_m_s"]
    src = job.source()
    s1, la, parts = walk_arrays(src, spec)
    with s2io.r2():
        sha = job.put("refs", name, s1)
        sha_look = job.put("look", name, la)
        prov = job.built_by()
    twin, ladder = stage1_walk_twin(body, spec), ladder_twin(body, spec)
    owed = twin_owed(st, body, name, walk=True)
    if (twin is None and owed["stage1"]) or (ladder is None and owed["ladder"]):
        raise GateError(f"{job.set_name} {body} {name}: the set must hold "
                        + ("Stage 1's own walk" if twin is None and owed["stage1"] else "the look ladder's rung")
                        + f" under this name and none of the same spec is recorded ({RULES['S2S-twins']})")
    if twin is not None and twin["sha256"] != sha:
        raise GateError(f"{job.set_name} {body} {name}: its Stage 1-format file hashes to {sha}, Stage 1's "
                        f"{twin['name']} ({twin['record']}) to {twin['sha256']}")
    if ladder is not None and ladder["sha256"] != sha_look:
        raise GateError(f"{job.set_name} {body} {name}: its look-format file hashes to {sha_look}, the look ladder's "
                        f"{ladder['speed']} {ladder['variant']} ({ladder['record']}) to {ladder['sha256']}")
    info, summ, price = parts["info"], parts["summary"], parts["price"]
    le = dict(body=body, speed=name, variant=job.set_name, features=list(st.features), file=f"look/{name}.npz",
              sha256=sha_look,
              frames=parts["frames"], spec=look._plain(asdict(spec)), info=info, ik_ok=parts["ik_ok"],
              ik_fail_frames=parts["ik_fail_frames"], zmp_error_final_mm=1e3 * info["zmp_error_history_m"][-1],
              zmp_margin_min_mm=1e3 * info["zmp_margin_min_m"], rolls=parts["rolls"], measured=parts["measured"],
              price=price, flags=parts["flags"])
    stage0 = walk_stage0(body, price)
    entry = dict(
        name=name, file=f"refs/{name}.npz", sha256=sha, frames=parts["frames"], dt_s=float(s1["dt"]),
        period_s=float(info["period"]), speed_m_s=float(info["speed"]), stride_m=float(info["stride"]),
        lift_m=float(info["lift"]), cadence_steps_s=float(info["cadence_hz"]), yaw_rate_rad_s=0.0,
        stride_motion=[float(x) for x in s1["stride_motion"]],
        min_cop_margin_single_m=float(summ["min_cop_margin_single"]),
        max_friction_ratio=float(summ["max_friction_ratio"]), ik_ok=bool(summ["ik_ok"]),
        generator="look.generate (gait.generate at accuracy r2)",
        dynamics="look.look_inverse_dynamics (gait.inverse_dynamics with F5's rolling contact)" if st.look
        else "gait.inverse_dynamics",
        spec=look._plain(asdict(spec)), info=info,
        set=job.set_name, kind="walk", label=st.label_of(body, job.H, v), margin_rule=st.margin_rule,
        crouch_frac=float(spec.crouch), strides=int(info.get("strides", 1)), window_s=float(info["window_s"]),
        look_file=f"look/{name}.npz", look_sha256=sha_look,
        stage1_sha256=None if twin is None else twin["sha256"], equals_stage1=None if twin is None else True,
        stage1_twin=None if twin is None else twin["name"],
        ladder_sha256=None if ladder is None else ladder["sha256"], equals_ladder=None if ladder is None else True,
        ladder_twin=None if ladder is None else [ladder["speed"], ladder["variant"]],
        stage0=stage0, available=True, group=item.group, definition=item.definition, built_by=prov)
    if st.look:  # a window of K whole strides (S2L-format)
        entry.update(duty=float(info["duty"]), tail_phase_rad=float(spec.tail_phase),
                     rolling_frames=parts["rolling_frames"], split_failed_frames=parts["split_failed_frames"])
        if entry["strides"] != int(spec.strides) or abs(entry["window_s"] - entry["strides"] * entry["period_s"]) > 1e-9:
            raise SetError(f"{name}: the window is not {spec.strides} strides of the spec's walk (strides "
                           f"{entry['strides']}, window {entry['window_s']!r} s, period {entry['period_s']!r} s)")
    elif entry["strides"] != 1 or entry["window_s"] != entry["period_s"]:
        raise SetError(f"{name}: a walk of today's laws is one stride (strides {entry['strides']}, window "
                       f"{entry['window_s']!r} s, period {entry['period_s']!r} s)")
    text = (f"ik {entry['ik_ok']}; governing {stage0['governing'][0]} {stage0['governing'][1]} "
            f"{stage0['governing'][2]:.4f}; Stage 0: {stage0['text']}"
            + ("" if twin is None else f"; equals Stage 1's {twin['name']}")
            + ("" if ladder is None else f"; equals the look ladder's {ladder['speed']} {ladder['variant']}"))
    job.note(name, parts["seconds"], text)
    if stage0["blocks_freeze"]:
        job.drop("refs", name)
        job.drop("look", name)
        entry = _unavailable(
            name, "walk", "dropped from the ladder: " + stage0["text"],
            dict(stage0=stage0, speed_m_s=entry["speed_m_s"], period_s=entry["period_s"], ik_ok=entry["ik_ok"],
                 min_cop_margin_single_m=entry["min_cop_margin_single_m"], built_sha256=sha,
                 built_look_sha256=sha_look, price=price),
            set=job.set_name, label=entry["label"], margin_rule=entry["margin_rule"], crouch_frac=entry["crouch_frac"],
            spec=entry["spec"], dropped=True, group=item.group, definition=item.definition, built_by=prov)
    else:
        job.out["look_entries"][name] = look._plain(le)
    job.out["refs"][name] = refs._plain(entry)


def _step_two(job: _Job, item: Item, p: dict, tried: list, nominal_gone: str | None = None) -> bool:
    """Step (2) of ``RULES["S2S-fallback"]`` for the ladder rung ``p["b"]``: a change from the neighbouring rung
    toward 0.40 m/s.  True when it stands: the change's entry and the item's (not available, reached by the change)
    are written.  ``nominal_gone``: why the nominal walk is no rung, where that is why the direct change was not
    built (``RULES["S2S-lead-in"]``)."""
    name = item.name
    v = setio.parse_name(p["b"])["speed_m_s"]
    nb = None if v not in setio.SPEEDS else neighbour(v)
    if nb is None:
        return False
    nbn = setio.walk_name(nb)
    other = setio.change_name(nbn, p["b"])
    got = job.walk(nbn)
    if got is None:
        tried.append(dict(change=other, not_built=f"the walk {nbn} is not a rung of the set"))
        return False
    gn, rn, en, tn = got
    gb, rb, eb, tb = job.walk(p["b"])
    e2, fl2 = job.move(item, moves.change(other, gn, gb), rn, rb, en, eb, tn, tb, (nbn, p["b"]))
    e2["ramp_strides"] = 1.0
    if not fl2["ok"]:
        tried.append(_numbers(e2, fl2, ramp_strides=1.0, change=other))
        job.drop("transitions", other)
        return False
    e2["fallback"] = dict(rule="S2S-fallback", text=RULES["S2S-fallback"], step=2, of=name, tried=tried)
    job.out["transitions"][other] = e2
    why = ("no direct transition within the gates" if nominal_gone is None else nominal_gone)
    job.out["transitions"][name] = refs._plain(_unavailable(
        name, item.kind, f"{why}; the rung is reached from {nbn} ({other}; fallback step 2)", dict(tried=tried),
        item=item.definition, group=item.group, inputs=job.inputs((p["a"], p["b"], nbn)), set=job.set_name,
        pass_=False if nominal_gone is None else None, fallback=dict(rule="S2S-fallback", step=2, by=other),
        built_by=job.by))
    return True


def _produce_lead_in(job: _Job, item: Item) -> None:
    """The ``start`` item of a set whose nominal walk is no rung (``RULES["S2S-lead-in"]``): the start from standing
    straight into the 0.40 m/s walk, built and gated like ``start``, under the name ``setio.start_name`` gives it; the
    item's own entry says why it has no reference and which start the set has."""
    nom, bar = item.depends
    alt = setio.start_name(bar)
    gone = job.missing_walk(item, (nom, bar))
    got = job.walk(bar)
    if got is None:  # neither walk is a rung: the set has no lead-in
        job.out["transitions"][item.name] = gone
        job.note(item.name, 0.0, "not available: " + gone["reason"])
        return
    gb, rb, eb, tb = got
    entry, fl = job.move(item, moves.start(alt, gb), None, rb, None, eb, None, tb, (nom, bar))
    rule = dict(rule="S2S-lead-in", text=RULES["S2S-lead-in"], of=item.name)
    if fl["ok"]:
        entry["lead_in"] = rule
        job.out["transitions"][alt] = entry
        gone = dict(gone, reason=f"{gone['reason']}; the set starts from standing straight into {bar} ({alt}; "
                                 "STAGE2_75_PLAN.md A8 as amended)", lead_in=dict(rule="S2S-lead-in", by=alt))
    else:
        job.out["transitions"][alt] = _to_unavailable(job, item, entry, fl, NO_GATES, lead_in=rule,
                                                      fallback=dict(rule="S2S-fallback", step=3, tried=1))
        gone = dict(gone, reason=f"{gone['reason']}; {alt}: {NO_GATES}", lead_in=dict(rule="S2S-lead-in", by=None))
        job.out["blocks_set"] = f"{alt}: {NO_GATES} ({RULES['S2S-lead-in']})"
    job.out["transitions"][item.name] = refs._plain(gone)
    job.note(item.name, 0.0, "not available: " + gone["reason"])


def _produce_straight(job: _Job, item: Item) -> None:
    """A start, a speed change or a stop, with the fallbacks of ``RULES["S2S-fallback"]`` and the start of a set
    whose nominal walk is no rung (``RULES["S2S-lead-in"]``)."""
    name, kind = item.name, item.kind
    nom, bar = setio.NOMINAL, setio.walk_name(setio.BAR_SPEED)
    if kind == "start" and job.refs.get(nom, {}).get("available") is False:
        _produce_lead_in(job, item)
        return
    gone = job.missing_walk(item, item.depends[: 2 if kind == "change" else 1])
    if gone is not None:
        p = setio.parse_name(name) if kind == "change" else {}
        only_nominal = (kind == "change" and p["a"] == nom and p["b"] != bar
                        and job.refs.get(p["b"], {}).get("available") is not False)
        # a ladder rung of a set without its nominal walk: step (2), from the neighbouring rung (S2S-lead-in)
        tried = [dict(change=name, not_built=gone["reason"])]
        if only_nominal and _step_two(job, item, p, tried, nominal_gone=gone["reason"]):
            return
        if only_nominal:
            gone = refs._plain(dict(gone, reason=f"{gone['reason']}; no change from the neighbouring rung within the "
                                                 "gates", numbers=dict(gone["numbers"], tried=tried),
                                    inputs=job.inputs(item.depends),
                                    fallback=dict(rule="S2S-fallback", step=3, tried=len(tried)),
                                    **({} if job.by is None else dict(built_by=job.by))))  # (a change was built)
        job.out["transitions"][name] = gone
        job.note(name, 0.0, "not available: " + gone["reason"])
        return
    blocks = kind == "start" or name == setio.change_name(nom, bar)
    if kind == "change":
        p = setio.parse_name(name)
        (ga, ra, ea, ta), (gb, rb, eb, tb) = job.walk(p["a"]), job.walk(p["b"])
        tried = []
        for step, ramp in ((0, 1.0), (1, 2.0)):
            mv = moves.change(name, ga, gb, ramp_strides=ramp)
            entry, fl = job.move(item, mv, ra, rb, ea, eb, ta, tb, (p["a"], p["b"]))
            entry["ramp_strides"] = ramp
            if fl["ok"]:
                if step:
                    entry["fallback"] = dict(rule="S2S-fallback", text=RULES["S2S-fallback"], step=1, ramp_strides=ramp,
                                             tried=tried)
                job.out["transitions"][name] = entry
                return
            tried.append(_numbers(entry, fl, ramp_strides=ramp, change=name))
            job.drop("transitions", name)
        if _step_two(job, item, p, tried):
            return
        job.out["transitions"][name] = refs._plain(_unavailable(
            name, kind, NO_GATES, dict(tried=tried), item=item.definition, group=item.group,
            inputs=job.inputs(item.depends), set=job.set_name, pass_=False,
            fallback=dict(rule="S2S-fallback", step=3, tried=len(tried)), built_by=job.by))
        if blocks:
            job.out["blocks_set"] = f"{name}: {NO_GATES} ({RULES['S2S-fallback']})"
        return
    if kind == "start":
        gb, rb, eb, tb = job.walk(nom)
        entry, fl = job.move(item, moves.start(name, gb), None, rb, None, eb, None, tb, (nom,))
    else:
        ga, ra, ea, ta = job.walk(bar)
        make = moves.stop_two_step if kind == "stop_two_step" else moves.stop_one_step
        entry, fl = job.move(item, make(name, ga), ra, None, ea, None, ta, None, (bar,))
        if item.definition.get("label"):
            entry["label"] = item.definition["label"]
    if fl["ok"]:
        job.out["transitions"][name] = entry
        return
    job.out["transitions"][name] = _to_unavailable(job, item, entry, fl, NO_GATES,
                                                   fallback=dict(rule="S2S-fallback", step=3, tried=1))
    if blocks:
        job.out["blocks_set"] = f"{name}: {NO_GATES} ({RULES['S2S-fallback']})"


def _produce_turns(job: _Job) -> None:
    """One turn rate, both directions (``RULES["S2S-turns"]``)."""
    its = {setio.parse_name(i.name)["direction"]: i for i in job.mine}
    first = job.mine[0]
    p = setio.parse_name(first.name)
    label, rate, angle = p["rate"], p["rate_rad_s"], p["angle_rad"]
    bar = first.depends[0]
    gone = [job.missing_walk(i, (bar,)) for i in job.mine]
    if gone[0] is not None:
        for e in gone:
            job.out["transitions"][e["name"]] = e
        job.out["stride_rules"][label] = dict(outcome="none", chosen=None, reason=gone[0]["reason"], references={
            e["name"]: False for e in gone}, no_reference=gone[0]["reason"])
        return
    ga, ra, ea, ta = job.walk(bar)
    t0 = time.perf_counter()
    c = first.definition["stride_rule"]
    with s2io.r2():
        pick = _ml().pick_stride_factor if job.st.look else moves.pick_stride_factor
        rec = pick(job.source(), ga, ra, angle, rate, grid=tuple(c["grid"]),
                   limit=c["limit_kinematic"], other_margin=c["limit_other_joints_rad"])
        picked_by = job.built_by()
    rec = refs._plain(dict(rec, walk=bar, walk_sha256=ea["sha256"]))
    job.note(f"stride rule {label}", time.perf_counter() - t0, "tried " + "; ".join(
        f"f {r['f']}: twist {r['left']} / {r['right']}, other "
        + " / ".join("-" if (r.get("tightest_other") or {}).get(d) is None else
                     f"{r['tightest_other'][d]['joint']} {r['tightest_other'][d]['margin_rad']:.4f}"
                     for d in setio.DIRECTIONS) + f", ik {r['ik_ok']}, pass {r['pass_']}" for r in rec["tried"])
             + f"; chosen {rec['chosen']}")
    built: dict = {}
    while rec["chosen"] is not None and rec["outcome"] in ("picked", "rebuild"):
        f = rec["chosen"]
        built = {}
        for direction, item in its.items():
            sign = 1.0 if direction == "left" else -1.0
            mv = moves.turn(item.name, ga, angle, sign * rate, f, ramp_s=item.definition["args"]["ramp_s"])
            built[direction] = job.move(item, mv, ra, ra, ea, ea, ta, ta, (bar,))
        others = {d: fl["tightest_other"] for d, (_, fl) in built.items()}
        rec = refs._plain(moves.step_down(rec, dict(
            ik_ok=bool(all(e["ik_ok"] for e, _ in built.values())),
            hip_yaw_range_use=float(max(e["hip_yaw_range_use"] for e, _ in built.values())),
            tightest_other_rad=min(o["margin_rad"] for o in others.values()) if all(others.values()) else None,
            tightest_other=others,
            over_leg_drive=bool(any(fl["over_leg_drive"] for _, fl in built.values())),
            a1_over=bool(any(fl["a1_over"] for _, fl in built.values())),
            pass_={d: bool(e["pass_"]) for d, (e, _) in built.items()},
            stage0_blocks={d: fl["blocks"] for d, (_, fl) in built.items()})))
        job.note(f"stride rule {label}", 0.0, f"built at f {f:g}: {rec['outcome']}"
                 + ("" if rec["reason"] is None else f" ({rec['reason']})"))
    references, why, whys = {}, None, {}
    for direction, item in its.items():
        name = item.name
        if rec["outcome"] != "frozen":
            why = f"{NO_TWIST}: {rec['reason']}"
            if job.st.look:  # (ruling 26 of 17 Nov 2026: each direction names the gates its own build failed)
                also = sorted(built[direction][1]["failed"]) if direction in built else []
            else:  # (the same words for both directions: what the last built pair missed beside the stride rule)
                also = sorted({g for _, fl in built.values() for g in fl["failed"]})
            if also:
                why += f"; also outside {', '.join(also)}"
            whys[name] = why
            if direction in built:
                entry, fl = built[direction]
                job.out["transitions"][name] = _to_unavailable(job, item, entry, fl, why)
            else:
                job.out["transitions"][name] = refs._plain(_unavailable(
                    name, item.kind, why, dict(stride_rule="set.json stride_rules[" + label + "]"), item=item.definition,
                    group=item.group, inputs=job.inputs((bar,)), set=job.set_name, pass_=None, built_by=picked_by))
            references[name] = False
            continue
        entry, fl = built[direction]
        if fl["ok"]:
            references[name] = True
            job.out["transitions"][name] = entry
            continue
        only_heading = bool(fl["failed"]) and set(fl["failed"]) <= set(HEADING_GATES) and not fl["blocks"]
        why = NO_HEADING if only_heading else (
            "no reference within the gates (" + "; ".join(
                ([f"entry gates: {', '.join(fl['failed'])}"] if fl["failed"] else [])
                + ([f"Stage 0: {entry['stage0']['text']}"] if fl["blocks"] else [])) + ")")
        job.out["transitions"][name] = _to_unavailable(job, item, entry, fl, why)
        references[name] = False
        whys[name] = why
    if job.st.look and len(set(whys.values())) > 1:  # (each direction's own reason, where they differ)
        why = "; ".join(f"{n}: {w}" for n, w in whys.items())
    rec = dict(rec, references=references, no_reference=None if all(references.values()) else why)
    job.out["stride_rules"][label] = rec
    for name in references:
        e = job.out["transitions"][name]
        if e.get("available") is not False:
            e["turn"]["stride_rule"] = rec
        else:
            e["numbers"]["stride_rule"] = {k: v for k, v in rec.items() if k != "text"}


def _screen_file(job: _Job, sub: str, name: str, entry: dict, kind: str) -> tuple[dict, dict]:
    """One file of the turn in place screened at Stage 0 as a move (``setscreen.screen_move`` on the file in the
    scratch directory, with the gates of ``kind``: "start", "stop" or "turn"): (the entry's ``stage0`` block,
    flags).  The file has a move's keys; ``entry`` gives ``split``, ``ik_ok`` and the two recorded contact numbers."""
    src = job.source(fresh=False)
    probe = dict(entry, name=name, file=f"{sub}/{name}.npz", move=dict(kind=kind, builder=kind, causal=False))
    block = setscreen.screen_move(src, job.scratch, name, probe, None)
    with np.load(job.scratch / probe["file"], allow_pickle=False) as z:
        with s2io.r2():
            df = setscreen.drive_flags(src, z["tau"], z["omega"], z["t"])
    v = block["verdict"]
    stage0 = dict(source=f"setscreen.screen_move on the built file, with the gates of a {kind} (at build time)",
                  governing=block["governing"], within_limits=v["within_limits"], over=v["over"],
                  blocks_freeze=v["blocks_freeze"], a1_over=v["a1_over"], over_leg_drive=bool(df["over_leg_drive"]),
                  text=v["text"], tightest_joint=block["range"].get("tightest"),
                  hip_yaw_range_use=block["range"].get("hip_yaw_range_use"),
                  accel_forward_lp1hz_peak_m_s2=block["accel"].get("com_forward_lp1hz_peak"))
    return s2io.plain(stage0), dict(blocks=v["blocks_freeze"], a1_over=bool(v["a1_over"]))


def _produce_inplace(job: _Job) -> None:
    """The turn in place of one direction: its periodic gait, its start and its stop
    (``LOOK_RULES["S2L-inplace"]``; STAGE2_TRACKB_PLAN.md L13)."""
    ml = _ml()
    its = {i.kind: i for i in job.mine}
    ref_it, start_it, stop_it = its[INPLACE], its[INPLACE_START], its[INPLACE_STOP]
    nom = ref_it.depends[0]

    def put(item: Item, entry: dict) -> None:
        entry = dict(entry)
        entry.setdefault("margin_rule", job.st.margin_rule)
        (job.out["refs"] if item.kind == INPLACE else job.out["transitions"])[item.name] = refs._plain(entry)

    gone = job.missing_walk(ref_it, (nom,))
    if gone is not None:
        for it in job.mine:
            put(it, dict(gone, name=it.name, kind=it.kind, item=it.definition, definition=it.definition))
        job.note(ref_it.name, 0.0, "not available: " + gone["reason"])
        return
    _, _, e_nom, _ = job.walk(nom)
    d = ref_it.definition
    spec = inplace_spec(job.set_name, job.body, job.H)
    b = moves.GaitX.of(ref_it.name, spec, job.H, yaw_rate=float(d["rate_rad_s"]), in_place=True)
    if b.plain() != d["gait"]:
        raise SetError(f"{ref_it.name}: the gait built now is not the item's")
    src = job.source()
    t0 = time.perf_counter()
    bd = ml.hold_move(src, start_it.name, b, int(d["hold_strides"]))
    first = float(bd.move.schedule.window[0])
    cut = float(transitions.WINDOWS["start"][1]) + int(d["cut_hold_stride"]) - 1.0  # the reference's first phase
    n = moves.N_FRAMES
    with s2io.r2():
        g = ml.generate(src, bd.move, None)  # (not causal: moves.generate's own)
        per = ml.periodicity(g, cut)
        full = moves.summary(g, None, None, src["q_home"])
        ml.entry_extra(bd, g, full)
        use = moves.range_use(src, g["q"])
        arr = ml.periodic_cut(src, g, cut)
        full_arr = moves.arrays(g)
        prov = job.built_by()
    k1 = int(round((cut - first) * n))
    start_arr = {k: (v[:k1] if getattr(v, "ndim", 0) >= 1 and len(v) == full["frames"] else v) for k, v in full_arr.items()}
    inputs = job.inputs((nom,))
    common = dict(available=True, set=job.set_name, group=ref_it.group, inputs=inputs, built_by=prov,
                  margin_rule=job.st.margin_rule, crouch_frac=float(spec.crouch))
    numbers = dict(periodicity=per, ik_ok=bool(g["ik_ok"]), hip_yaw_range_use=moves.hip_yaw_use(use),
                   zmp_error_history_m=g["zmp_error_history_m"],
                   long_plan=dict(output_phase=full["output_phase"], frames=full["frames"], gates=full["gates"]))
    job.note(ref_it.name, time.perf_counter() - t0,
             f"periodicity CoM {per['com_mm']:.4f} mm, ZMP {per['zmp_mm']:.4f} mm, joints {per['joint_rad']:.3e} rad: "
             f"{'ok' if per['ok'] else 'FAILS'}; ik {g['ik_ok']}; thigh twist {numbers['hip_yaw_range_use']:.4f}")

    def none(reason: str, more: dict | None = None) -> None:
        for it in job.mine:
            put(it, _unavailable(it.name, it.kind, reason, dict(numbers, **(more or {})), item=it.definition,
                                 definition=it.definition, group=it.group, inputs=inputs, set=job.set_name,
                                 pass_=False, built_by=prov, margin_rule=job.st.margin_rule))

    if not (per["ok"] and g["ik_ok"]):
        none(NO_PERIODIC if not per["ok"] else "the turn in place's long plan has frames whose IK did not solve")
        return
    sha = job.put("refs", ref_it.name, arr)
    sha_s = job.put("transitions", start_it.name, start_arr)
    single = np.sum(arr["swing"], axis=1) == 1
    T = float(arr["dt"]) * n
    split = full["split"]
    e_ref = dict(
        common, name=ref_it.name, kind="turn_in_place", file=f"refs/{ref_it.name}.npz", sha256=sha, frames=n,
        dt_s=float(arr["dt"]), period_s=T, speed_m_s=0.0, stride_m=0.0, lift_m=float(spec.lift_frac * job.H),
        cadence_steps_s=2.0 / T, yaw_rate_rad_s=float(d["rate_rad_s"]), in_place=True,
        stride_motion=[float(x) for x in arr["stride_motion"]],
        min_cop_margin_single_m=float(np.nanmin(arr["cop_margin"][single])),
        max_friction_ratio=float(np.max(arr["friction"])), ik_ok=bool(g["ik_ok"]), ik_fail_frames=0, split=split,
        generator="one stride cut from a long plan of designlab.stage2.moves (moves_look.periodic_cut)",
        dynamics="moves_look.idyn (the exact split of a turn gait)", spec=look._plain(asdict(spec)),
        label="turn in place", strides=1, window_s=T, look_file=None, look_sha256=None, periodicity=per,
        hip_yaw_range_use=numbers["hip_yaw_range_use"], range_use=use,
        cut=dict(from_plan=start_it.name, stride_phase=[cut, cut + 1.0], hold_strides=int(d["hold_strides"]),
                 hold_stride=int(d["cut_hold_stride"])),
        zmp_error_history_m=g["zmp_error_history_m"], rules=list(ml.INPLACE_RULE_IDS), definition=ref_it.definition,
        gates=dict(ik_ok=bool(g["ik_ok"]), periodicity=bool(per["ok"]),
                   zmp_error=bool(g["zmp_error_history_m"][-1] <= moves.ZMP_HISTORY_LIMIT_M)))
    e_ref["pass_"] = bool(all(e_ref["gates"].values()))
    e_ref["stage0"], fl_ref = _screen_file(job, "refs", ref_it.name, e_ref, "turn")
    e_start = dict(
        full, **common, name=start_it.name, file=f"transitions/{start_it.name}.npz", sha256=sha_s, frames=k1,
        duration_s=float(start_arr["t"][-1] - start_arr["t"][0]), output_phase=[full["output_phase"][0], cut],
        to=ref_it.name, item=start_it.definition,
        definition=refs._plain(dict(bd.move.definition(), body_model_sha256=job.model, ref_a_sha256=None,
                                    ref_b_sha256=sha)),
        fallback=None, tightest_other=moves.tightest_other(src, start_arr["q"]),
        join_b=dict(note="the periodic reference is the next stride of the same plan: the join is exact; the "
                         "reference's own periodicity is gated", periodicity=per, com_ok=True, zmp_ok=True,
                    com_max_mm=0.0, zmp_max_mm=0.0),
        g3=None, label="the long plan up to the hold stride cut as the periodic reference")
    e_start["gates"] = dict(ik_ok=bool(g["ik_ok"]), join_a=bool(full["gates"]["join_a"]), periodicity=bool(per["ok"]),
                            zmp_error=bool(g["zmp_error_history_m"][-1] <= moves.ZMP_HISTORY_LIMIT_M))
    e_start["pass_"] = bool(all(e_start["gates"].values()))
    e_start["stage0"], fl_start = _screen_file(job, "transitions", start_it.name, e_start, "start")
    e_start["a1_over"] = fl_start["a1_over"]
    bad = [f"{nm}: " + ("its gates " + ", ".join(k for k, x in e["gates"].items() if not x) if not e["pass_"] else
                        "Stage 0: " + e["stage0"]["text"])
           for nm, e, fl in ((ref_it.name, e_ref, fl_ref), (start_it.name, e_start, fl_start))
           if not e["pass_"] or fl["blocks"]]
    if bad:
        job.drop("refs", ref_it.name)
        job.drop("transitions", start_it.name)
        none("the turn in place has no reference within the gates (" + "; ".join(bad) + ")",
             dict(reference=dict(gates=e_ref["gates"], stage0=e_ref["stage0"]),
                  start=dict(gates=e_start["gates"], stage0=e_start["stage0"])))
        return
    # the stop: Stage 1's one-step stop from the reference
    ref = refs.Reference.load(job.scratch / "refs" / f"{ref_it.name}.npz")
    e_stop, fl = job.move(stop_it, moves.stop_one_step(stop_it.name, b), ref, None, dict(sha256=sha), None, None,
                          None, (nom,))
    e_stop["label"] = "Stage 1's one-step stop from the turn in place (not causal)"
    put(ref_it, e_ref)
    put(start_it, e_start)
    if fl["ok"]:
        put(stop_it, e_stop)
    else:
        put(stop_it, _to_unavailable(job, stop_it, e_stop, fl, NO_GATES))


def produce(set_name: str, body: str, name: str, root=setio.ROOT) -> dict:
    """Everything of one item, or of the group it is built with, made in this process: the arrays written into the
    group's scratch directory (``partial/build-<group>/``), and the result ``{item, set, body, refs, transitions,
    look_entries, stride_rules, files {relative path: sha256}, header, blocks_set, log}``.  Nothing of the set is
    changed."""
    its = {i.name: i for i in items(set_name, body)}
    group = its[name].group if name in its else name
    job = _Job(set_name, body, group, root)
    shutil.rmtree(job.scratch, ignore_errors=True)
    job.scratch.mkdir(parents=True)  # (this process's own: scratch_dir)
    kind = job.mine[0].kind
    if kind == "walk":
        _produce_walk(job, job.mine[0])
    elif kind == "turn":
        _produce_turns(job)
    elif kind in LOOK_KINDS:
        _produce_inplace(job)
    else:
        _produce_straight(job, job.mine[0])
    for e in job.out["transitions"].values():  # every entry of a set carries its margin rule (the plan's A3)
        e["margin_rule"] = job.st.margin_rule
    return job.out  # (header None: nothing was generated, as for a move whose walk is not a rung; no body was built)


def partial_path(d: Path, group: str) -> Path:
    return d / PARTIAL / f"{group}.json"


def _frozen_rows(d: Path) -> dict:
    """name -> row of the set's latest manifest ({} for a set that is not frozen, or not built)."""
    doc = _read(d / "set.json")
    fz = None if doc is None else setio.frozen(doc)
    return {} if fz is None else fz["entries"]


def _expected(rows: dict, entries: dict, rel: str) -> str | None:
    """The sha256 a frozen set holds for the file ``rel`` (``refs/x.npz``, ``transitions/x.npz``, ``look/x.npz``),
    or None where the latest manifest does not list its name."""
    sub, stem = rel.split("/")[0], Path(rel).stem
    row = rows.get(stem)
    if row is None:
        return None
    if sub == "look":
        return row.get("look_sha256") or (entries.get(stem) or {}).get("look_sha256")
    return row["sha256"]


def build_item(set_name: str, body: str, name: str, root=setio.ROOT, owner: int | None = None) -> dict:
    """One item (or its group) in this process: ``produce``, then its arrays moved from the scratch directory into
    place and ``partial/<group>.json`` written.  Returns that partial's content.  On a frozen set a file of an entry
    the latest manifest holds is put in place only when its sha256 is the manifest's; the others are listed under
    ``mismatch`` and left out.  ``owner``: the pid of the builder that started the item and holds the set's lock; if
    the lock names another holder when the arrays are ready (the builder was killed and another took the set over),
    nothing is put in place and ``SetError`` is raised."""
    s2io.require_one_thread()
    out = produce(set_name, body, name, root)
    d = setio.set_dir(set_name, body, root)
    group = out["item"]
    scratch = scratch_dir(d, group)
    try:
        _held_by(d, owner)
    except SetError:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    rows = _frozen_rows(d)
    entries = ((_read(d / "refs.json") or {}).get("references", {}) if rows else {})
    out["mismatch"] = []
    for name_, e in {**out["refs"], **out["transitions"]}.items():
        if name_ in rows and e.get("available") is False:
            out["mismatch"].append(dict(name=name_, frozen=rows[name_]["sha256"], built=None,
                                        why=f"built now it has no reference: {e.get('reason')}"))
    for rel, sha in sorted(out["files"].items()):
        want = _expected(rows, entries, rel)
        if want is not None and want != sha:
            out["mismatch"].append(dict(name=Path(rel).stem, file=rel, frozen=want, built=sha,
                                        why="the rebuilt file is not the frozen one"))
            continue
        dest = d / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch / rel, dest)
    shutil.rmtree(scratch, ignore_errors=True)
    _fail_at("npz", group)
    s2io.write_json(partial_path(d, group), out)
    _fail_at("partial", group)
    return out


def _worker(args) -> str:
    """A pool worker: one group in a fresh process, for the builder ``owner`` (a pid); the group's name."""
    set_name, body, group, root, owner = args
    s2io.require_one_thread()
    build_item(set_name, body, group, root, owner=owner)
    return group


# ------------------------------------------------------------------------------------------------
# 4. The set's records, and merging a finished item into them
# ------------------------------------------------------------------------------------------------
def _new_docs(set_name: str, body: str, header: dict) -> dict:
    st = _set(set_name)
    H, _ = body_facts(body)
    specs = walk_specs(set_name, body, H)
    crouch = float(next(iter(specs.values())).crouch)
    speeds = {setio.walk_name(v): (nominal_speed(H) if v is None else float(v)) for v in (None, *setio.SPEEDS)}
    docs = _plain_docs(set_name, body, header, st, crouch, speeds)
    if st.look:  # the look's rules beside the builder's; what is in force of STAGE2_TRACKB_PLAN.md L12
        ml = _ml()
        docs["refs.json"]["rules"].update({k: LOOK_RULES[k] for k in ("S2L-sets", "S2L-chain", "S2L-tail-phase",
                                                                     "S2L-noF5", "S2L-format", "S2L-inplace")})
        docs["refs.json"]["feedforward"] = ("inverse-dynamics joint torque, contact-consistent (a walk: "
                                            "look.look_inverse_dynamics, F5's rolling contact); MJCF joint damping "
                                            "and friction not included")
        docs["transitions.json"]["rules"].update(ml.RULES)
        docs["transitions.json"]["rules"].update({k: LOOK_RULES[k] for k in ("S2L-moves", "S2L-inplace")})
        docs["transitions.json"]["causal_limits"] = look_causal_limits(docs["transitions.json"]["causal_limits"])
        docs["transitions.json"]["method"] += ("; with the look's laws and solver (designlab.stage2.moves_look), each "
                                               "move bound to the windows of the walks it joins")
        docs["set.json"]["rules"] = dict(RULES, **LOOK_RULES)
        docs["set.json"]["spec_rule"] = RULES["S2S-speeds"] + "; " + LOOK_RULES["S2L-chain"]
        docs["set.json"]["look"] = dict(
            pick=base_body(body), join_rule="J1 as designed (moves_look.RULES S2L-J1 to S2L-J3)",
            leave_phase=dict(ml.LEAVE_PHASE), **look_review(body), joins_failed={})
    return docs


def look_review(body: str) -> dict:
    """``set.json`` ``look``'s fields of STAGE2_TRACKB_PLAN.md L12 for a look set on ``body`` (``LOOK_RULES
    ["S2L-join-review"]``): the join code's review (``JOIN_REVIEW``: path and sha256), the cut in force (None: none;
    the builder takes no cut) and why: on a pick whose cut-3 leave point the review probed (``cut3_probe``) cut 3 is
    not taken (L12 with ruling 29), with the probe's readings; elsewhere no cut is in force and a cut is the lead's."""
    pick = base_body(body)
    probe = cut3_probe(pick)
    if probe is not None:
        why = (f"cut 3 is not taken for {pick} (L12 with ruling 29 of 17 Nov 2026): the join-code review's probe on "
               f"the causal grid reads {probe['left_turns_mm']:g} mm (left turns) and {probe['right_turns_mm']:g} mm "
               f"(right turns) at cut 3's leave point, over the {probe['limit_mm']!r} mm limit; a move that fails a "
               "join gate at J1's leave point is not available with its numbers (A11 for a turn, H3 for the stop; a "
               "start or nominal_to_v0.4 stops the set, A8), and no limit is changed")
    else:
        why = (f"no cut is in force: the join-code review probed no cut-3 leave point on {pick}; whether a cut is "
               "taken for this body is the lead's (L12)")
    return dict(join_review=dict(JOIN_REVIEW), cut=None, cut_reason=why, cut3_probe=None if probe is None else dict(probe),
                note="join rule J1 as designed; the builder takes no cut of STAGE2_TRACKB_PLAN.md L12 (cut_reason); "
                     "joins_failed lists every move whose join_command or join_b gate fails, with those numbers")


def look_causal_limits(limits: dict) -> dict:
    """``transitions.json`` ``causal_limits`` of a look set: the round's (``_plain_docs``) with the warm-up reading
    of ``join_command`` at the look sets' limit and the rule that sets it (``moves_look.LOOK_WARMUP_ZMP_LIMIT_MM``,
    ``moves_look.RULES["S2L-P42-warmup"]``: the owner's decision P42).  Every other limit is left as it is."""
    ml = _ml()
    return dict(limits, warmup_zmp_mm=ml.LOOK_WARMUP_ZMP_LIMIT_MM, warmup_zmp_rule=ml.WARMUP_RULE_ID)


def cut3_probe(pick: str) -> dict | None:
    """The join-code review's probe of cut 3's leave point on ``pick`` (``CUT3_PROBE_FILE``), or None where it read
    none there."""
    rec = ((_doc(CUT3_PROBE_FILE) or {}).get("bodies") or {}).get(pick)
    return None if rec is None else dict(rec)


#: The join gates of STAGE2_TRACKB_PLAN.md L12's trigger.
JOIN_GATES = ("join_command", "join_b")


def joins_failed(docs: dict) -> dict:
    """move -> what L12's Check reads of a move whose ``join_command`` or ``join_b`` gate fails (available or not):
    ``{available, failing, join_command, join_b}``, the two blocks' gated numbers (``set.json`` ``look
    .joins_failed``)."""
    out = {}
    for name, e in docs["transitions.json"]["transitions"].items():
        gates = e.get("gates") or {}
        bad = [g for g in JOIN_GATES if gates.get(g) is False]
        if not bad:
            continue
        src = e if e.get("available") is not False else (e.get("numbers") or {})
        rec = dict(available=e.get("available") is not False, failing=bad)
        jc, jb = src.get("join_command"), src.get("join_b")
        if isinstance(jc, dict):
            rec["join_command"] = dict(
                {k: jc.get(k) for k in ("reference", "warmup_zmp_max_mm", "warmup_com_max_mm", "warmup_joint_max_rad",
                                        "ok")},
                at_command={k: v for k, v in (jc.get("at_command") or {}).items() if k != "zmp_note"},
                limits={k: v for k, v in (jc.get("limits") or {}).items() if k != "zmp_frames"})
        if isinstance(jb, dict):
            rec["join_b"] = {k: jb.get(k) for k in ("reference", "com_max_mm", "zmp_max_mm", "com_ok", "zmp_ok")}
        out[name] = rec
    return refs._plain(out)


def _plain_docs(set_name: str, body: str, header: dict, st: SetDef, crouch: float, speeds: dict) -> dict:
    return {
        "refs.json": dict(
            body=body, set=set_name, schema=SCHEMA, accuracy=s2io.ACCURACY, model=header["model"], H_m=header["H_m"],
            q_home=header["q_home"], joints=header["joints"],
            feedforward="inverse-dynamics joint torque, contact-consistent; MJCF joint damping and friction not "
                        "included",
            rules={k: RULES[k] for k in ("S2S-sets", "S2S-speeds", "S2S-margin", "S2S-crouch", "S2S-process",
                                         "S2S-formats", "S2S-twins", "S2S-stage0", "S2S-not-available")},
            references={}),
        "transitions.json": dict(
            body=body, set=set_name, schema=SCHEMA, transitions={},
            method="designlab.stage2.moves: Stage 1's transition (ZMP preview control on the designlab model) with "
                   "its schedule passed as data; a commanded move (a stop, a turn) is causal",
            preview=dict(qe=transitions.PREVIEW_QE, r=transitions.PREVIEW_R, preview_s=transitions.PREVIEW_S),
            shift_strides=transitions.SHIFT_STRIDES, settle_strides=transitions.SETTLE_STRIDES,
            path_half_width=transitions.PATH_HALF_WIDTH, walk_ramp_strides=transitions.WALK_RAMP_STRIDES,
            windows=transitions.WINDOWS, g3_com_mm=transitions.G3_COM_MM, g3_zmp_mm=transitions.G3_ZMP_MM,
            causal_limits=dict(warmup_com_mm=transitions.G3_COM_MM, warmup_zmp_mm=moves.WARMUP_ZMP_LIMIT_MM,
                               joint_at_command_rad=moves.JOINT_AT_COMMAND_RAD,
                               consistency_first_stride_mm=1e3 * moves.CONSISTENCY_LIMIT_M,
                               zmp_after_first_stride_mm=1e3 * moves.ZMP_HISTORY_LIMIT_M,
                               cop_margin_single_mm=1e3 * moves.CAUSAL_COP_MARGIN_M,
                               joint_range_rad=moves.OTHER_JOINT_MARGIN_RAD),
            rules=dict(moves.RULES, **{k: RULES[k] for k in ("S2S-process", "S2S-twins", "S2S-stage0", "S2S-fallback",
                                                              "S2S-lead-in", "S2S-turns", "S2S-not-available")})),
        "set.json": dict(
            set=set_name, body=body, schema=SCHEMA, role=st.role, note=st.note, spec_rule=RULES["S2S-speeds"],
            margin_rule=st.margin_rule, speeds=speeds, crouch_frac=crouch, omega_rule=RULES["S2S-omega"], rules=RULES,
            lead_in=None, look_entries={}, stride_rules={}, not_available={}, blocked=None, freezes=[]),
    }


def load_docs(d: Path) -> dict | None:
    """The set's three records as they are, or None while the set has none (a set never built)."""
    got = {n: _read(d / n) for n in ("refs.json", "transitions.json", "set.json")}
    have = [n for n, v in got.items() if v is not None]
    if not have:
        return None
    if len(have) != 3:
        raise SetError(f"{d}: it holds {', '.join(have)} and not the other record(s) of a set")
    return got


def _files_ok(d: Path, entry: dict, look_too: bool) -> bool:
    for key_f, key_s in (("file", "sha256"),) + ((("look_file", "look_sha256"),) if look_too else ()):
        p = d / entry[key_f]
        if not p.is_file() or s2io.sha256_file(p) != entry[key_s]:
            return False
    return True


def _entry_of(docs: dict, item_kind: str, name: str) -> dict | None:
    if item_kind in ("walk", INPLACE):  # (the periodic gaits: refs.json)
        return docs["refs.json"]["references"].get(name)
    return docs["transitions.json"]["transitions"].get(name)


def _same_inputs(docs: dict, entry: dict) -> bool:
    """Whether the walks an entry was built from are the ones the set holds now (their sha256, or their being no
    rung)."""
    refs_ = docs["refs.json"]["references"]
    for walk, sha in (entry.get("inputs") or {}).items():
        e = refs_.get(walk)
        now = None if e is None or e.get("available") is False else e.get("sha256")
        if e is None or now != sha:
            return False
    return True


def current(d: Path, docs: dict | None, item: Item, rows: dict, frozen: bool | None = None) -> bool:
    """Whether an item needs no build: its entry holds the definition made now, was built from the walks the set
    holds now, and its files have the recorded sha256.  An entry with no reference has no file: it is current while
    the Stage 2 modules that decided it (``built_by.stage2_modules``) are today's, or the set is frozen (``frozen``;
    default: whether ``rows`` holds anything).  An entry of the latest manifest (``rows``) is current when its files
    are in place: it is never built into the records again."""
    if docs is None:
        return False
    frozen = bool(rows) if frozen is None else frozen
    e = _entry_of(docs, item.kind, item.name)
    if e is None:
        return False
    if marked(e) is not None:  # (ruling 32 (A): the lead's mark stands; nothing is built for it)
        return True
    if item.name in rows:
        return _files_ok(d, e, item.kind == "walk")
    if e.get("item" if item.kind not in ("walk", INPLACE) else "definition") != item.definition \
            or not _same_inputs(docs, e):
        return False
    if e.get("available") is False:
        # no array pins it: the verdict is the code's that built it, so other code builds it again (an entry that
        # follows from its walks alone has no built_by); a frozen set's records are not rebuilt for a change of code
        by = e.get("built_by")
        return bool(frozen) or by is None or by.get("stage2_modules") == modules_for(docs["set.json"].get("set"))
    return _files_ok(d, e, item.kind == "walk")


def extras_current(d: Path, docs: dict | None, group: str, names, rows: dict) -> bool:
    """Whether the references a group made beside its items (a fallback's change from the neighbouring rung, the
    start of a set into its 0.40 m/s walk) need no build: each one's file is in place with its sha256 and, unless
    the latest manifest (``rows``) holds it, it was built from the walks the set holds now."""
    if docs is None:
        return True
    for name, e in docs["transitions.json"]["transitions"].items():
        if e.get("group") != group or name in names or e.get("available") is False:
            continue
        if not _files_ok(d, e, False) or (name not in rows and not _same_inputs(docs, e)):
            return False
    return True


def _write_if_changed(path: Path, doc: dict, old: dict | None, point: str, group: str) -> bool:
    if old is not None and s2io.canonical(doc) == s2io.canonical(old):  # (canonical: a stand's NaN equals itself)
        return False
    s2io.write_json(path, doc)
    _fail_at(point, group)
    return True


#: The groups whose failure stops a set (the start and the change into the bar's walk), in the order a build runs
#: them (``build``); ``set.json`` ``blocked`` lists the failing ones in this order.
STOP_GROUPS = (setio.START, setio.change_name(setio.NOMINAL, setio.walk_name(setio.BAR_SPEED)))


def blocked_list(value) -> list[dict]:
    """``set.json`` ``blocked`` as a list of ``{by, reason}``, one per failing group that stops the set (ruling 23
    of 17 Nov 2026: a list, so that a set whose start and change into the bar's walk both fail names both).  It
    reads the list this code writes, an older record's dict (one group) and null (the set is not stopped)."""
    if not value:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    return [dict(b) for b in value]


def _stop_order(group: str) -> tuple:
    return (STOP_GROUPS.index(group) if group in STOP_GROUPS else len(STOP_GROUPS), group)


def blocks_in_records(docs: dict) -> dict:
    """group -> the reason it stops the set, read from the records alone, as ``produce`` decides it (its
    ``blocks_set``): the start that has no transition within its gates (``RULES["S2S-fallback"]``); the start into
    the 0.40 m/s walk of a set whose nominal walk is no rung, when that has none (``RULES["S2S-lead-in"]``); the
    change into the bar's walk that has none.  ``merge`` keeps these for the groups it is not merging, so that a
    record whose ``blocked`` named one group only (the dict of older code, which the last blocking merge
    overwrote) loses no stop when another group's merge passes."""
    tr = docs["transitions.json"]["transitions"]
    start, change = STOP_GROUPS
    alt = setio.start_name(setio.walk_name(setio.BAR_SPEED))
    out = {}
    e = tr.get(start)
    if e is not None and e.get("available") is False:
        lead = e.get("lead_in") or {}
        a = tr.get(alt) or {}
        if e.get("reason") == NO_GATES:
            out[start] = f"{start}: {NO_GATES} ({RULES['S2S-fallback']})"
        elif lead.get("rule") == "S2S-lead-in" and lead.get("by") is None and a.get("available") is False \
                and a.get("reason") == NO_GATES:
            out[start] = f"{alt}: {NO_GATES} ({RULES['S2S-lead-in']})"
    e = tr.get(change)
    if e is not None and e.get("available") is False and e.get("reason") == NO_GATES:
        out[change] = f"{change}: {NO_GATES} ({RULES['S2S-fallback']})"
    return out


def _blocked_after(set_doc: dict, docs: dict, group: str, blocks: str | None) -> list[dict] | None:
    """``set.json`` ``blocked`` once ``group`` is merged: every other failing group it named, or the records show
    (``blocks_in_records``), stays; ``group`` is named by its own result alone (``blocks``: its partial's
    ``blocks_set``).  None when no group stops the set (as a set that was never stopped records it)."""
    named = {b.get("by"): b.get("reason") for b in blocked_list(set_doc.get("blocked")) if b.get("by") != group}
    for g, why in blocks_in_records(docs).items():
        if g != group:
            named.setdefault(g, why)
    if blocks:
        named[group] = blocks
    return [dict(by=g, reason=named[g]) for g in sorted(named, key=_stop_order)] or None


def merge(d: Path, part: dict, log=print) -> dict:
    """Merge one finished item (a ``partial/<group>.json``'s content) into the set's records and delete the partial.
    An entry the latest manifest holds is never written; an entry whose arrays were reproduced (same definition,
    same inputs, same sha256) is left as it was, provenance included.  The array of a name that holds no reference
    after the merge (an entry replaced by one with no reference, a fallback's change no longer made) is removed.
    Returns ``{written [...], kept [...], frozen [...], removed [...]}``."""
    set_name, body, group = part["set"], part["body"], part["item"]
    docs = load_docs(d)
    old = None if docs is None else {k: json.loads(json.dumps(v)) for k, v in docs.items()}
    if docs is None:
        if part.get("header") is None:
            raise SetError(f"{d}: the set has no record yet and {group} built no body to make one from")
        docs = _new_docs(set_name, body, part["header"])
    rows = setio.frozen(docs["set.json"])
    rows = {} if rows is None else rows["entries"]
    res = dict(written=[], kept=[], frozen=[], removed=[])
    set_doc = docs["set.json"]
    orphans: list = []  # the files of names that hold no reference after this merge (never a frozen name's)
    for key, target in (("refs", docs["refs.json"]["references"]), ("transitions", docs["transitions.json"]["transitions"])):
        made = part[key]
        if key == "transitions":  # an entry this group made before and does not make now (a fallback's change)
            for name in [n for n, e in target.items() if e.get("group") == group and n not in made and n not in rows]:
                del target[name]
                set_doc["not_available"].pop(name, None)
                res["written"].append(name)
                orphans.append(f"transitions/{name}.npz")
        for name, e in made.items():
            if name in rows:
                res["frozen"].append(name)
                continue
            was = target.get(name)
            same = (was is not None and was.get("available") is not False and e.get("available") is not False
                    and was.get("sha256") == e.get("sha256") and was.get("item") == e.get("item")
                    and was.get("definition") == e.get("definition") and was.get("inputs") == e.get("inputs")
                    and was.get("look_sha256") == e.get("look_sha256"))
            if same:
                res["kept"].append(name)
            else:
                target[name] = e
                res["written"].append(name)
            now = target[name]  # what set.json says of it follows the entry (also when this merge is a repeated one)
            if now.get("available") is False:
                set_doc["not_available"][name] = dict(reason=now["reason"], numbers=now.get("numbers"))
                set_doc["look_entries"].pop(name, None)
                orphans += [f"{key}/{name}.npz"] + ([f"look/{name}.npz"] if key == "refs" else [])
            else:
                set_doc["not_available"].pop(name, None)
                if (key == "refs" and now.get("look_sha256") is not None  # (the turn in place has no look-format file)
                        and (set_doc["look_entries"].get(name) or {}).get("sha256") != now["look_sha256"]):
                    set_doc["look_entries"][name] = part["look_entries"][name]
    for label, rec in part.get("stride_rules", {}).items():
        names = [setio.turn_name(label, x) for x in setio.DIRECTIONS]
        if not any(n in rows for n in names):
            set_doc["stride_rules"][label] = rec
    if res["written"] and part.get("header") is not None:
        for k, v in part["header"].items():
            docs["refs.json"][k] = v
    set_doc["blocked"] = _blocked_after(set_doc, docs, group, part.get("blocks_set"))
    if _set(set_name).look:  # the text of every rule of the look's laws that an entry may name (a set built before a
        for key, text in _ml().RULES.items():  # rule was added gains it; a text already recorded stays as it is)
            docs["transitions.json"]["rules"].setdefault(key, text)
        if not setio.frozen(set_doc):  # L12's record (ruling 30): the join code's review, the cut, the failing joins
            # (and the causal limits as P42 holds them on a look set: a set built before P42 says so from now on)
            docs["transitions.json"]["causal_limits"] = look_causal_limits(docs["transitions.json"]["causal_limits"])
            for key, text in LOOK_RULES.items():
                set_doc["rules"].setdefault(key, text)
            look_doc = set_doc.setdefault("look", {})
            if look_doc.get("join_review") is None:
                look_doc.update(look_review(body))
            look_doc["joins_failed"] = joins_failed(docs)
    if not setio.frozen(set_doc):  # (a frozen set's lead-in is its freeze's: its walks are never written again)
        set_doc["lead_in"] = lead_in_of(docs["refs.json"]["references"])
    d.mkdir(parents=True, exist_ok=True)
    for name in ("refs.json", "transitions.json", "set.json"):
        _write_if_changed(d / name, docs[name], None if old is None else old[name], name, group)
    for rel in orphans:  # (after the records: an entry never names a file this merge removed)
        if (d / rel).is_file():
            (d / rel).unlink()
            res["removed"].append(rel)
            log(f"{set_name} {body}: removed {rel}: no entry of the set names it any more")
    partial_path(d, group).unlink(missing_ok=True)
    _fail_at("deleted", group)
    for line in part.get("log", []):
        _log_time(d, set_name, body, line["what"], line["seconds"], line["text"])
        log(f"{set_name} {body} {line['what']}: {line['text']}"
            + (f"; {line['seconds']:.0f} s" if line["seconds"] else ""))
    return res


def _log_time(d: Path, set_name: str, body: str, what: str, seconds: float, text: str) -> None:
    path = d / LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{s2io.now_utc()} {set_name} {body} {what} seconds={seconds:.1f} {text}\n")


def _undo_killed_first_merge(d: Path, log=print) -> None:
    """A build killed inside the first merge of a new set leaves ``refs.json``, or ``refs.json`` and
    ``transitions.json``, without ``set.json`` (the three are written in that order) and the merged item's partial.
    Where every entry of the records left is one a partial still in ``partial/`` made, the records are removed: the
    partials are merged again from nothing, to the same records.  Any other incomplete set of records is left for
    ``load_docs`` to refuse."""
    got = {n: _read(d / n) for n in ("refs.json", "transitions.json", "set.json")}
    if got["set.json"] is not None or (got["refs.json"] is None and got["transitions.json"] is None):
        return
    made: set = set()
    pdir = d / PARTIAL
    for p in sorted(pdir.glob("*.json")) if pdir.is_dir() else []:
        try:
            part = s2io.read_json(p)
            if p == partial_path(d, part["item"]):
                made |= set(part.get("refs", {})) | set(part.get("transitions", {}))
        except (ValueError, KeyError, TypeError):
            continue
    left = (set((got["refs.json"] or {}).get("references", {}))
            | set((got["transitions.json"] or {}).get("transitions", {})))
    if not left or not left <= made:
        return
    for n in ("refs.json", "transitions.json"):
        if got[n] is not None:
            (d / n).unlink()
            log(f"removed {n}: the first merge of the set was killed before set.json was written; its item is "
                "merged again")


def _frozen_entries_held(d: Path) -> None:
    """``SetError`` if the latest freeze lists a name that the set's records hold no reference for: a frozen entry
    is never built into the records again, so nothing can be built for it; the committed record must come back."""
    docs = load_docs(d)
    fz = None if docs is None else setio.frozen(docs["set.json"])
    for name, row in ({} if fz is None else fz["entries"]).items():
        rec = "refs.json" if row.get("kind") in setio.WALK_KINDS else "transitions.json"
        e = docs[rec]["references" if rec == "refs.json" else "transitions"].get(name)
        if e is None or (e.get("available") is False and marked(e) is None):
            raise SetError(f"{d}: the latest freeze lists {name} and {rec} holds "
                           + ("no such entry" if e is None else "it as not available")
                           + ": restore the committed record (nothing is built)")


def _partial_ok(d: Path, part: dict, set_name: str, body: str, names: set) -> bool:
    """Whether a partial found at the start of a build is a finished item of this set as it is defined now: every
    file it made is in place with its sha256, and every entry holds the definition made now."""
    if part.get("set") != set_name or part.get("body") != body:
        return False
    its = {i.name: i for i in items(set_name, body)}
    made = {**part.get("refs", {}), **part.get("transitions", {})}
    if not made or part.get("item") not in names:
        return False
    for name, e in made.items():
        it = its.get(name)
        if it is not None and e.get("item" if it.kind not in ("walk", INPLACE) else "definition") != it.definition:
            return False
        if it is None and e.get("group") not in names:
            return False
    wrong = {m.get("file") for m in part.get("mismatch", [])}
    for rel, sha in part.get("files", {}).items():
        if rel in wrong:
            continue
        p = d / rel
        if not p.is_file() or s2io.sha256_file(p) != sha:
            return False
    return True


def _require_models(body: str) -> None:
    """The body's round-two model files must be here: the Stage 0 clearance gates read them through
    ``plant.model_files``, which would otherwise rebuild them into ``stage1/models/<body>/``."""
    xml = stage1.case_dir(body) / "models" / f"{stage1.candidate_id(body)}.xml"
    if not (xml.is_file() and xml.with_suffix(".json").is_file()):
        raise SetError(f"{body}: its round-two model files are not here ({s2io.portable(str(xml))} and its ledger "
                       "JSON); restore them first (python tools/restore_inputs.py restore, or copy-from DIR)")


def build(set_name: str, body: str, root=setio.ROOT, only=None, procs: int = 2, log=print) -> dict:
    """Build what the set lacks (module docstring, section 2), resumably, under the set's lock.  ``only``: item or
    group names (their walks must be there).  ``procs`` spawned workers at a time, each item in a fresh process (0:
    the items run in this process, one after another, for tests).  Returns ``{dir, ran, kept, recovered, restored,
    skipped, mismatch, blocked, not_available, lead_in, seconds}`` (``blocked``: the set's record's list of the
    failing groups that stop it, ``{by, reason}`` each, this run's among them; None if nothing stops it);
    ``SetError`` after the run if a frozen entry could not be reproduced; ``GateError`` if a built-in gate stopped
    it."""
    s2io.require_one_thread()
    its = items(set_name, body)
    by_group = groups(its)
    known = set(by_group) | {i.name for i in its}
    for n in only or ():
        if n not in known:
            raise SetError(f"{set_name} {body}: no item or group {n!r} ({', '.join(by_group)})")
    wanted = None if not only else {next(i.group for i in its if n in (i.name, i.group)) for n in only}
    _require_models(body)
    d = setio.set_dir(set_name, body, root)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = dict(dir=str(d), ran=[], kept=[], recovered=[], restored=[], skipped=[], mismatch=[], blocked=None,
               not_available={}, lead_in=None, seconds=None)
    stopped: list = []  # the groups that stopped this run ({by, reason}); the set's record names them with the others
    with s2io.Lock(d / ".lock", log=log):
        for f in s2io.sweep_tmp(d):
            log(f"removed a stale temporary file: {f}")
        pdir = d / PARTIAL
        for p in sorted(pdir.glob(f"{SCRATCH_PREFIX}*")) if pdir.is_dir() else []:
            shutil.rmtree(p, ignore_errors=True)
            log(f"removed a stale scratch directory: {PARTIAL}/{p.name}")
        _undo_killed_first_merge(d, log=log)
        _frozen_entries_held(d)

        def take(part: dict, how: str) -> None:
            group = part["item"]
            m = merge(d, part, log=log)
            res["mismatch"] += part.get("mismatch", [])
            res[how].append(group)
            if m["frozen"] and not part.get("mismatch"):
                res["restored"] += m["frozen"]
            if part.get("blocks_set"):
                stopped.append(dict(by=group, reason=part["blocks_set"]))

        # finished items a killed run left in partial/
        for p in sorted(pdir.glob("*.json")) if pdir.is_dir() else []:
            if p.name.startswith("screen__"):  # the set screen's own partials
                continue
            try:
                part = s2io.read_json(p)
                ok = p == partial_path(d, part["item"]) and _partial_ok(d, part, set_name, body, set(by_group))
            except (ValueError, KeyError, TypeError):
                ok = False
            if ok:
                take(part, "recovered")
            else:
                p.unlink(missing_ok=True)
                log(f"removed a partial item that is not of the set as it is now: {p.name}")

        def run(phase_groups: list) -> None:
            if not phase_groups or stopped:
                return
            jobs = [(set_name, body, g, str(root), os.getpid()) for g in phase_groups]
            log(f"{set_name} {body}: building {len(jobs)} item(s) with {min(max(1, procs), len(jobs))} process(es): "
                + ", ".join(phase_groups))
            if procs < 1:
                for j in jobs:
                    if stopped:
                        res["skipped"].append(j[2])
                        continue
                    take(build_item(*j[:3], root=j[3], owner=j[4]), "ran")
                return
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=min(procs, len(jobs)), mp_context=mp.get_context("spawn"),
                                     max_tasks_per_child=1) as pool:
                futures = {pool.submit(_worker, j): j[2] for j in jobs}
                try:
                    for fut in as_completed(futures):
                        if fut.cancelled():
                            continue
                        group = fut.result()
                        take(s2io.read_json(partial_path(d, group)), "ran")
                        if stopped:
                            for other in futures:
                                if other.cancel():
                                    res["skipped"].append(futures[other])
                except BaseException:
                    for other in futures:
                        other.cancel()
                    raise

        for phase in (True, False):  # the walks, then the moves
            docs = load_docs(d)
            rows = {} if docs is None else (setio.frozen(docs["set.json"]) or {}).get("entries", {})
            todo = []
            for group, members in by_group.items():
                if (members[0].kind == "walk") != phase or (wanted is not None and group not in wanted):
                    continue
                if members[0].optional and (wanted is None or group not in wanted):  # built only when named
                    continue
                if (all(current(d, docs, i, rows) for i in members)
                        and extras_current(d, docs, group, {i.name for i in members}, rows)):
                    if group not in res["recovered"]:
                        res["kept"].append(group)
                    continue
                todo.append(group)
            if not phase:  # the start and the change into the bar's walk first: a failure of either stops the set
                first = list(STOP_GROUPS)
                todo.sort(key=lambda g: (first.index(g) if g in first else len(first)))
            run(todo)
        docs = load_docs(d)
        if docs is not None:
            res["not_available"] = {n: v.get("reason") for n, v in docs["set.json"].get("not_available", {}).items()}
            res["lead_in"] = docs["set.json"].get("lead_in")
        # every failing group that stops the set: this run's and the record's (ruling 23); None: not stopped
        res["blocked"] = (blocked_list(docs["set.json"].get("blocked")) if docs is not None else stopped) or None
    res["seconds"] = time.perf_counter() - t0
    if res["mismatch"]:
        raise SetError(f"{d}: a frozen entry was not reproduced; nothing of it was written: " + "; ".join(
            f"{m['name']}: {m['why']} (frozen sha256 {m['frozen']}, built {m['built']})" for m in res["mismatch"]))
    return res


# ------------------------------------------------------------------------------------------------
# 5. verify
# ------------------------------------------------------------------------------------------------
def pr_function():
    """The pull request's ``_tuned_f9_spec``, imported from its script by path (None where the script is not
    here)."""
    if not PR_SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location("jetson_plem_study_for_sets", PR_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    saved = {v: os.environ.get(v) for v in (*s2io.THREAD_VARS, "MUJOCO_GL")}
    try:
        spec.loader.exec_module(mod)
    finally:  # the script sets defaults for these at import
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod._tuned_f9_spec


def pr_differences(body: str, H: float) -> list[str] | None:
    """Where ``tuned_f9_spec`` differs from the pull request's function at a rung of the body (field by field), or
    None where the script is not here."""
    fn = pr_function()
    if fn is None:
        return None
    out = []
    for v in (nominal_speed(H), *setio.SPEEDS):
        a, b = asdict(tuned_f9_spec(body, H, v)), asdict(fn(body, H, v))
        out += [f"{body} at {v:g} m/s, {k}: {a.get(k)!r} here, {b.get(k)!r} in the pull request's function"
                for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return out


def source_hash(built_by: dict) -> str:
    """One hash of what built an entry: its Stage 2 modules' and the panels' source hashes."""
    return s2io.sha256_json(dict(stage2_modules=built_by.get("stage2_modules"),
                                 panel_sources_sha256=built_by.get("panel_sources_sha256")))


def modules_now() -> dict:
    return {s2io.portable(str(s2io.module_path(m))): s2io.sha256_file(s2io.module_path(m)) for m in MODULES}


def modules_for(set_name) -> dict:
    """``modules_now`` for an entry of the set ``set_name``: with the look's module for a look set (``modules_of``)."""
    now = modules_now()
    if set_name in LOOK_SETS:
        now[s2io.portable(str(s2io.module_path(LOOK_MODULE)))] = s2io.sha256_file(s2io.module_path(LOOK_MODULE))
    return now


def _check_name(name: str, kind: str, e: dict, item: Item | None, bad: list) -> None:
    """A name against its entry: it parses back (``setio.parse_name``) to the values its definition holds."""
    if kind != "walk" and (name == "stop" or name.startswith("stop_in_place_")):
        bad.append(f"{name}: a transition may not carry a name by which Stage 1 registers its own stop command")
        return
    tip = inplace_of(name)
    if tip is not None:  # the turn in place's names are design D5's, not setio's
        if item is None or item.kind not in LOOK_KINDS or (tip["role"] == "ref") != (kind == "walk"):
            bad.append(f"{name}: a name of the turn in place on an entry that is not its {tip['role']}")
        return
    try:
        p = setio.parse_name(name)
    except ValueError as exc:
        bad.append(f"{name}: not a name setio makes ({exc})")
        return
    if kind == "walk":
        if p["kind"] != "walk":
            bad.append(f"{name}: a walk's entry under a {p['kind']}'s name")
            return
        v, got = p["speed_m_s"], e.get("speed_m_s")
        if v is not None and got is not None and abs(float(got) - v) > 1e-9 * v:
            bad.append(f"{name}: its entry's speed is {got!r} m/s, its name holds {v:g}")
        return
    d = e.get("definition") or (item.definition if item is not None else {})
    builder = d.get("builder")
    want = {"start": "start", "change": "change", "turn": "turn"}.get(p["kind"])
    if p["kind"] == "stop":
        want = {2: "stop_two_step", 1: "stop_one_step"}[p["steps"]]
    if p["kind"] == "walk" or builder != want:
        bad.append(f"{name}: its name reads as a {p['kind']}, its definition's builder is {builder!r}")
        return
    a, b = (d.get("specs") or {}).get("a") or {}, (d.get("specs") or {}).get("b") or {}
    if p["kind"] == "start" and e.get("available") is not False and b.get("name") != p.get("walk", setio.NOMINAL):
        bad.append(f"{name}: its definition starts into {b.get('name')!r}")
    if p["kind"] == "change" and (a.get("name"), b.get("name")) != (p["a"], p["b"]):
        bad.append(f"{name}: its definition joins {a.get('name')!r} to {b.get('name')!r}")
    if p["kind"] == "stop" and a.get("name") != p["walk"]:
        bad.append(f"{name}: its definition stops from {a.get('name')!r}")
    if p["kind"] == "turn":
        args = d.get("args") or {}
        if args.get("rate") != p["sign"] * p["rate_rad_s"] or args.get("angle") != p["angle_rad"]:
            bad.append(f"{name}: its definition turns {args.get('angle')!r} rad at {args.get('rate')!r} rad/s, its "
                       f"name {p['angle_rad']!r} rad at {p['sign'] * p['rate_rad_s']!r} rad/s")


def verify(set_name: str, body: str, root=setio.ROOT) -> dict:
    """The set as it is against its rules (module docstring, section 3; the round-A design's 4.3): ``{ok, failures,
    notes, set, body, frozen, available, not_available}``.  No array is rebuilt.  A reference the set records as not
    available (a rung over a limit, a turn rate without a reference, a move outside its gates) is a result, listed
    in ``notes`` with its reason; a difference between the set and its records or rules is a failure."""
    d = setio.set_dir(set_name, body, root)
    bad: list = []
    notes: list = []
    out = dict(ok=False, failures=bad, notes=notes, set=set_name, body=body, frozen=None, available=[],
               not_available={})
    try:
        docs = setio.load(d)
    except Exception as exc:  # tune.InputError: a record is missing
        bad.append(f"the set cannot be read: {exc}")
        return out
    refs_doc, tr_doc, set_doc, screen = (docs[k] for k in ("refs.json", "transitions.json", "set.json", "screen.json"))
    its = {i.name: i for i in items(set_name, body)}
    group_names = {i.group for i in its.values()}
    H, model = body_facts(body)
    specs = walk_specs(set_name, body, H)
    st = _set(set_name)
    fz = setio.frozen(set_doc)
    rows = {} if fz is None else fz["entries"]
    out["frozen"] = None if fz is None else dict(index=fz["index"], manifest_sha256=fz["manifest_sha256"],
                                                 entries=len(rows))
    for doc, name in ((refs_doc, "refs.json"), (tr_doc, "transitions.json"), (set_doc, "set.json")):
        if doc.get("set") != set_name or doc.get("body") != body:
            bad.append(f"{name} is of {doc.get('set')!r} {doc.get('body')!r}")
    entries = {n: ("walk", e) for n, e in refs_doc["references"].items()}
    for n, e in tr_doc["transitions"].items():
        if n in entries:
            bad.append(f"{n}: names a walk and a move")
        entries[n] = ("move", e)
    blocks = (screen or {}).get("blocks", {})
    if screen is None:
        bad.append("screen.json is missing: the set is not screened (python -m designlab.stage2.setscreen run)")
    now_modules = modules_for(set_name)
    sources, dirty_entries = {}, []
    for name, it in its.items():
        if name not in entries and it.optional:
            notes.append(f"{name}: not built (an optional item: python -m designlab.stage2.sets build --only {it.group})")
        elif name not in entries:
            bad.append(f"{name}: not built (python -m designlab.stage2.sets build)")
    for name, (kind, e) in entries.items():
        item = its.get(name)
        if item is None and e.get("group") not in group_names:
            notes.append(f"{name}: an entry of no item defined now")
        elif item is None and e["group"] in its and (e.get("item") != its[e["group"]].definition
                                                    or not _same_inputs(docs, e)):
            # (a group of one item bears its item's name: a fallback's change, the start into the 0.40 m/s walk)
            bad.append(f"{name}: made by the item {e['group']}, whose definition is not the one made now, or built "
                       "from another file of its walk than the set holds now (python -m designlab.stage2.sets build)")
        _check_name(name, kind, e, item, bad)
        by = e.get("built_by")
        if by is not None:
            if by.get("blas_threads") != 1 or by.get("accuracy") != s2io.ACCURACY:
                bad.append(f"{name}: built with blas_threads {by.get('blas_threads')!r} at accuracy "
                           f"{by.get('accuracy')!r} (1 and {s2io.ACCURACY!r} are required)")
            if by.get("stage2_modules") != now_modules:  # (item 6: reported, not a failure; nothing is rebuilt here)
                changed = sorted({Path(k).name for k in set(now_modules) | set(by.get("stage2_modules") or {})
                                  if now_modules.get(k) != (by.get("stage2_modules") or {}).get(k)})
                notes.append(f"{name}: built by other code (not rebuilt: "
                             + ("its verdict is that code's" if e.get("available") is False else
                                "its arrays are its entry's and were not compared with what today's code builds")
                             + f"; {', '.join(changed)} moved since)")
        if e.get("margin_rule") != st.margin_rule:
            bad.append(f"{name}: its entry's margin_rule is {e.get('margin_rule')!r}, the set's is {st.margin_rule!r} "
                       "(STAGE2_75_PLAN.md A3; python -m designlab.stage2.sets build in an empty directory)")
        if kind == "walk" and name in specs:  # (a dropped rung's entry carries its label too)
            want = st.label_of(body, H, setio.parse_name(name)["speed_m_s"])
            if e.get("label") != want:
                bad.append(f"{name}: its entry's label is {e.get('label')!r}, the set's at this speed is {want!r} "
                           "(STAGE2_75_PLAN.md A2)")
        tip = e.get("kind") == "turn_in_place" or (item is not None and item.kind == INPLACE)  # (in refs.json)
        if item is not None and (e.get("item" if kind != "walk" else "definition") != item.definition
                                 or not _same_inputs(docs, e)):
            bad.append(f"{name}: its entry is not of the definition made now, or was built from another file of its "
                       "walk than the set holds now (python -m designlab.stage2.sets build)")
        if e.get("available") is False:  # ---- an item with no reference
            out["not_available"][name] = e.get("reason")
            if not e.get("reason") or "numbers" not in e:
                bad.append(f"{name}: recorded as not available without a reason and its numbers")
            if e.get("file"):
                bad.append(f"{name}: recorded as not available and names a file")
            if (set_doc.get("not_available") or {}).get(name, {}).get("reason") != e.get("reason"):
                bad.append(f"{name}: set.json not_available does not hold its reason")
            mk = marked(e)
            if mk is not None:  # (ruling 32 (A): marked on a G9-ref record; the next freeze drops it)
                _verify_mark(name, mk, rows, set_doc, bad, notes)
            elif name in rows:
                bad.append(f"{name}: the latest freeze lists it and its entry is not available")
            notes.append(f"{name}: not available: {e.get('reason')}")
            continue
        out["available"].append(name)
        if by is None:
            bad.append(f"{name}: no built_by")
        else:
            sources.setdefault(source_hash(by), []).append(name)
            if by.get("dirty"):
                dirty_entries.append(name)
        # ---- item 1: files against entries and the latest manifest
        files = [("file", "sha256")] + ([("look_file", "look_sha256")] if kind == "walk" and not tip else [])
        here = True
        for key_f, key_s in files:
            p = d / e[key_f]
            if not p.is_file():
                bad.append(f"{name}: {e[key_f]} is missing (python -m designlab.stage2.sets build puts it back)")
                here = False
            elif s2io.sha256_file(p) != e[key_s]:
                bad.append(f"{name}: {e[key_f]} has sha256 {s2io.sha256_file(p)}, its entry records {e[key_s]}")
                here = False
        if name in rows:
            row = rows[name]
            esha = setio.entry_sha256("walk" if kind == "walk" else "move", e)
            if row.get("sha256") != e["sha256"]:
                bad.append(f"{name}: its entry records sha256 {e['sha256']}, the freeze {row.get('sha256')}")
            if row.get("entry_sha256") != esha:
                bad.append(f"{name}: its entry's hashed subset is {esha}, the freeze holds {row.get('entry_sha256')}")
            if (row.get("kind") in setio.WALK_KINDS) != (kind == "walk"):
                bad.append(f"{name}: the freeze lists it as a {row.get('kind')}")
            if kind == "walk" and row.get("look_sha256") not in (None, e["look_sha256"]):
                bad.append(f"{name}: its look-format file is {e['look_sha256']}, the freeze holds {row['look_sha256']}")
        elif fz is not None:
            notes.append(f"{name}: not in the latest freeze (index {fz['index']})")
        if tip:
            _verify_inplace(name, e, set_doc, bad)
        elif kind == "walk":
            _verify_walk(d, name, e, specs.get(name), set_doc, body, here, bad)
        else:
            _verify_move(d, name, e, body, bad, notes)
        # ---- the entries that must be Stage 1's own files and the look ladder's rungs (A4)
        owed = twin_owed(st, body, name, walk=kind == "walk")
        if owed["stage1"] and e.get("equals_stage1") is not True:
            bad.append(f"{name}: the set must hold Stage 1's own file under this name and its entry's equals_stage1 is "
                       f"{e.get('equals_stage1')!r} (STAGE2_75_PLAN.md A4)")
        if owed["ladder"] and e.get("equals_ladder") is not True:
            bad.append(f"{name}: its look-format file must be the look ladder's rung and its entry's equals_ladder is "
                       f"{e.get('equals_ladder')!r} (STAGE2_75_PLAN.md A4)")
        # ---- item 5: the screen
        if screen is not None:
            b = blocks.get(name)
            if b is None:
                bad.append(f"{name}: no block in screen.json (python -m designlab.stage2.setscreen run)")
                continue
            want = e["look_sha256"] if kind == "walk" and not tip else e["sha256"]
            if (b.get("arrays") or {}).get("sha256") != want:
                bad.append(f"{name}: its screen block is of another file ({(b.get('arrays') or {}).get('sha256')})")
            if kind == "walk" and not tip and (b.get("reference") or {}).get("sha256") != e["sha256"]:
                bad.append(f"{name}: its screen block names another Stage 1-format file")
            v = b.get("verdict") or {}
            if v.get("blocks_freeze"):
                bad.append(f"{name}: the screen blocks its freeze and the set holds it as a reference: "
                           + "; ".join(str(o.get("limit")) + ("" if o.get("value") is None else f" {o['value']}")
                                       for o in v["blocks_freeze"]))
            over_a1 = any(float(((b.get("drives") or {}).get(g) or {}).get(k, 0.0)) > 1.0
                          for g in setscreen.A1_GROUPS for k in setscreen.KINDS)
            if kind != "walk":
                is_turn = (e.get("move") or {}).get("builder") == "turn"
                if bool(v.get("a1_over")) != bool(e.get("a1_over")):
                    bad.append(f"{name}: its entry's a1_over is {e.get('a1_over')!r}, the screen's {v.get('a1_over')!r}")
                if over_a1 and not (is_turn and e.get("a1_over") is True
                                    and "S2A-appendage" in ((e.get("move") or {}).get("rules") or [])):
                    bad.append(f"{name}: a neck or tail ratio is above 1 and it is not a turn with a1_over true and "
                               "the S2A-appendage rule")
                if is_turn:
                    use = (b.get("range") or {}).get("hip_yaw_range_use")
                    if use is None or use > moves.TWIST_FINAL:
                        bad.append(f"{name}: the screen reads a thigh-twist use of {use!r} (at most {moves.TWIST_FINAL:g})")
            elif over_a1:
                bad.append(f"{name}: a neck or tail ratio is above 1 on a walk the set holds as a reference")
            if kind == "walk" and e.get("ladder_twin"):
                speed, variant = e["ladder_twin"]
                rung = (((_read(LOOK_SCREEN) or {}).get("rungs", {}).get(body) or {}).get(speed) or {}).get(variant)
                if rung is None:
                    notes.append(f"{name}: the look ladder's screen holds no block of {speed} {variant}")
                else:
                    diff = setscreen.ladder_differences(b, rung)
                    if diff:
                        bad.append(f"{name}: its screen block differs from the look ladder's {speed} {variant}: "
                                   + "; ".join(diff[:4]) + (f"; ... {len(diff) - 4} more" if len(diff) > 4 else ""))
    # ---- screen blocks of what the set does not hold
    for name in blocks:
        if name in out["available"]:
            continue
        if name in entries and marked(entries[name][1]) is not None:
            notes.append(f"{name}: screen.json holds a block of it from before its mark (python -m "
                         "designlab.stage2.setscreen run drops it)")
        else:
            bad.append(f"{name}: screen.json holds a block of a reference the set does not hold")
    # ---- the twins a body cannot be held to; arrays that no entry lists
    if st.stage1_twins and not stage1_has(body, "refs.json"):
        notes.append(f"Stage 1's run holds no record of {body}: {', '.join(st.stage1_twins)} were not compared with "
                     "Stage 1's files")
    if st.ladder_twins and not ladder_has(body):
        notes.append(f"the look ladder holds no rung of {body}: {', '.join(st.ladder_twins)} were not compared with it")
    listed = {e[k] for _, e in entries.values() if e.get("available") is not False
              for k in ("file", "look_file") if e.get(k)}
    for sub_ in ("refs", "look", "transitions"):
        for f in sorted((d / sub_).glob("*.npz")) if (d / sub_).is_dir() else []:
            if f"{sub_}/{f.name}" not in listed:
                notes.append(f"{sub_}/{f.name}: an array that no entry of the set lists (nothing reads it; remove it)")
    # ---- item 4: one crouch; the pull request's function
    crouches = {float(s.crouch) for s in specs.values()}
    if set_doc.get("crouch_frac") is None or {float(set_doc["crouch_frac"])} != crouches:
        bad.append(f"set.json records crouch_frac {set_doc.get('crouch_frac')!r}, the set's specs {sorted(crouches)}")
    if st.pr_spec:
        diff = pr_differences(body, H)
        if diff is None:
            notes.append(f"the pull request's script is not here ({s2io.portable(str(PR_SCRIPT))}): tuned_f9_spec was "
                         "not compared with it")
        else:
            bad += [f"tuned_f9_spec is not the pull request's function: {x}" for x in diff]
    # ---- item 3: the stride rule's record of every turn rate
    for label in setio.TURN_RATES:
        names = [setio.turn_name(label, x) for x in setio.DIRECTIONS]
        if not any(n in entries for n in names):
            continue
        rec = (set_doc.get("stride_rules") or {}).get(label)
        need = ("outcome", "chosen", "references")
        if rec is None or any(k not in rec for k in need):
            bad.append(f"turn rate {label}: set.json stride_rules holds no complete record")
            continue
        if "tried" in rec:
            for k in ("grid", "limit_kinematic", "limit_final", "limit_other_joints_rad", "steps_down", "builds", "text"):
                if k not in rec:
                    bad.append(f"turn rate {label}: the stride rule's record lacks {k}")
            for r in rec["tried"]:
                if "error" in r:
                    continue
                for x in setio.DIRECTIONS:
                    if r.get(x) is None or not (r.get("tightest_other") or {}).get(x):
                        bad.append(f"turn rate {label}: the stride rule's row at f {r.get('f')!r} lacks the {x} "
                                   "turn's twist use or tightest other joint")
        held = {n: entries[n][1].get("available") is not False for n in names if n in entries}
        if rec["references"] != held:
            bad.append(f"turn rate {label}: the stride rule's record says {rec['references']}, the entries {held}")
        if not all(held.values()):
            notes.append(f"turn rate {label}: {rec.get('no_reference') or rec.get('reason')}")
        else:
            notes.append(f"turn rate {label}: stride factor {rec['chosen']:g}"
                         + (f", {rec['steps_down']} step(s) down" if rec.get("steps_down") else "")
                         + (", a1_over" if rec.get("a1_over") else ""))
    # ---- the lead-in (RULES["S2S-lead-in"])
    lead, alt = lead_in_of(refs_doc["references"]), setio.start_name(setio.walk_name(setio.BAR_SPEED))
    if set_doc.get("lead_in") != lead and all(n in entries for n in walk_names()):
        bad.append(f"set.json records lead_in {set_doc.get('lead_in')!r}, its walks give {lead!r}")
    lead = set_doc.get("lead_in")
    if lead == alt:
        notes.append(f"lead-in: {alt} (the nominal walk is not a rung: the set starts from standing straight into "
                     "its 0.40 m/s walk; an exception, STAGE2_75_PLAN.md A8 as amended)")
        if alt not in entries:
            bad.append(f"{alt}: not built (python -m designlab.stage2.sets build)")
    elif alt in entries:
        bad.append(f"{alt}: the set holds it and its lead-in is {lead!r}")
    if lead is None and all(n in entries for n in walk_names()):
        notes.append("the set has no lead-in (neither its nominal nor its 0.40 m/s walk is a rung): its conditions "
                     "are not run")
    named = set()
    for b in blocked_list(set_doc.get("blocked")):  # (a list; an older record's dict or null reads too)
        named.add(b.get("by"))
        bad.append(f"the set is stopped: {b.get('reason')} (it goes to the lead)")
    for group, why in blocks_in_records(docs).items():  # (ruling 36 of 17 Nov 2026: a stop the records show and an
        if group not in named:  # older record's blocked leaves out is named too)
            bad.append(f"the set is stopped: {why} (it goes to the lead; the records show it and set.json blocked "
                       "does not name it)")
    if st.look:  # STAGE2_TRACKB_PLAN.md L12 (ruling 30): the set names the join code's review by path and sha256
        look_doc = set_doc.get("look") or {}
        if "S2L-join-review" not in (set_doc.get("rules") or {}) or look_doc.get("join_review") != JOIN_REVIEW:
            bad.append("set.json does not name the join code's review (rules S2L-join-review and look.join_review: "
                       f"{JOIN_REVIEW['path']}, sha256 {JOIN_REVIEW['sha256']}; STAGE2_TRACKB_PLAN.md L12)")
        elif not (STUDY / JOIN_REVIEW["path"]).is_file() or \
                s2io.sha256_file(STUDY / JOIN_REVIEW["path"]) != JOIN_REVIEW["sha256"]:
            bad.append(f"{JOIN_REVIEW['path']}: the join code's review the set names is missing or has another sha256")
        elif s2io.canonical(look_doc.get("joins_failed")) != s2io.canonical(joins_failed(docs)):
            bad.append("set.json look.joins_failed is not what the moves' join gates give (python -m "
                       "designlab.stage2.sets build merges it again)")
    for name, rec in (set_doc.get("not_available") or {}).items():
        if name not in entries or entries[name][1].get("available") is not False:
            bad.append(f"{name}: set.json not_available lists it and its entry does not say so")
    for name in set_doc.get("look_entries") or {}:
        if name not in out["available"]:
            bad.append(f"{name}: set.json holds a look entry of a walk the set does not hold")
    # ---- the bar's kinematic check of the references (a report; no rule makes it a failure here)
    rc = refcheck_state(d, docs)
    if rc is None:
        notes.append(f"no {REFCHECK} beside the set's records: its references were not played through the bar's own "
                     "readings (python -m designlab.stage2.bar refcheck --out <the sets' root>)")
    else:
        notes += refcheck_lines(rc)
    cited = (set_doc["freezes"][-1].get("refcheck") if fz is not None else None)
    if cited and (rc is None or rc["sha256"] != cited.get("sha256")):
        for cond, x in (cited.get("failing") or {}).items():
            notes.append(f"the latest freeze cites a refcheck ({cited.get('sha256', '')[:12]}...) in which {cond} FAILS "
                         f"{', '.join(x.get('fails') or [])}")
    # ---- item 6: provenance
    mixed = len(sources) > 1
    if fz is not None:
        last = set_doc["freezes"][-1]
        frozen_sources = {source_hash(entries[n][1]["built_by"]) for n in rows if n in entries
                          and entries[n][1].get("built_by")}
        frozen_dirty = [n for n in rows if n in dirty_entries]
        if len(frozen_sources) > 1 and not last.get("mixed_sources"):
            bad.append("the frozen entries were built from differing sources and the freeze does not say mixed_sources")
        if frozen_dirty and not last.get("dirty"):
            bad.append(f"frozen entries were built on a dirty tree and the freeze does not say dirty ({', '.join(frozen_dirty)})")
        if s2io.sha256_json(last.get("manifest")) != last.get("manifest_sha256"):
            bad.append("the latest freeze's manifest_sha256 is not its manifest's")
        for n in rows:
            if n not in entries:
                bad.append(f"{n}: the latest freeze lists it and the set has no such entry")
    else:
        if mixed:
            notes.append(f"the entries were built from {len(sources)} differing source states: a freeze refuses them "
                         "without --allow-mixed")
        if dirty_entries:
            notes.append(f"{len(dirty_entries)} entries were built on a dirty tree: a freeze refuses them without "
                         "--allow-dirty")
    out["ok"] = not bad
    return out


def _verify_walk(d: Path, name: str, e: dict, spec, set_doc: dict, body: str, here: bool, bad: list) -> None:
    """Item 4 of the round-A design's 4.3 for one walk: the two formats hold one motion; the entry's spec is the
    set's; its crouch is the set's; the twins' hashes."""
    if spec is not None and e.get("spec") != look._plain(asdict(spec)):
        bad.append(f"{name}: the spec in its entry is not the set's at this speed")
    if e.get("crouch_frac") is None or float(e["crouch_frac"]) != float(set_doc.get("crouch_frac", math.nan)):
        bad.append(f"{name}: its entry's crouch_frac is {e.get('crouch_frac')!r}, set.json's {set_doc.get('crouch_frac')!r}")
    le = (set_doc.get("look_entries") or {}).get(name)
    if le is None or le.get("sha256") != e.get("look_sha256") or le.get("file") != e.get("look_file"):
        bad.append(f"{name}: set.json holds no look entry of its look-format file")
    if here:
        with np.load(d / e["file"], allow_pickle=False) as a, np.load(d / e["look_file"], allow_pickle=False) as b:
            for key in ("q", "qd", "qdd"):
                if not np.array_equal(a[key], b[key]):
                    bad.append(f"{name}: {key} of the Stage 1-format file is not the look-format file's")
    if spec is not None:
        twin, ladder = stage1_walk_twin(body, spec), ladder_twin(body, spec)
        if (None if twin is None else twin["sha256"]) != e.get("stage1_sha256"):
            bad.append(f"{name}: Stage 1's file of the same spec is {None if twin is None else twin['sha256']}, its "
                       f"entry records {e.get('stage1_sha256')}")
        elif twin is not None and (twin["sha256"] != e["sha256"] or e.get("equals_stage1") is not True):
            bad.append(f"{name}: it does not hash to Stage 1's {twin['name']} ({e['sha256']} against {twin['sha256']})")
        if (None if ladder is None else ladder["sha256"]) != e.get("ladder_sha256"):
            bad.append(f"{name}: the look ladder's rung of the same spec is "
                       f"{None if ladder is None else ladder['sha256']}, its entry records {e.get('ladder_sha256')}")
        elif ladder is not None and (ladder["sha256"] != e["look_sha256"] or e.get("equals_ladder") is not True):
            bad.append(f"{name}: its look-format file does not hash to the look ladder's {ladder['speed']} "
                       f"{ladder['variant']} ({e['look_sha256']} against {ladder['sha256']})")


def _verify_inplace(name: str, e: dict, set_doc: dict, bad: list) -> None:
    """The turn in place's periodic gait against ``LOOK_RULES["S2L-inplace"]`` (STAGE2_TRACKB_PLAN.md L13's check)."""
    if e.get("pass_") is not True or (e.get("periodicity") or {}).get("ok") is not True:
        bad.append(f"{name}: held as a reference with pass_ {e.get('pass_')!r} and periodicity "
                   f"{(e.get('periodicity') or {}).get('ok')!r}")
    if (e.get("stage0") or {}).get("blocks_freeze"):
        bad.append(f"{name}: held as a reference although a Stage 0 gate blocked it where it was built")
    if abs(float(e.get("yaw_rate_rad_s") or 0.0)) != INPLACE_RATE or e.get("speed_m_s") != 0.0 or e.get("strides") != 1:
        bad.append(f"{name}: yaw rate {e.get('yaw_rate_rad_s')!r} rad/s, speed {e.get('speed_m_s')!r} m/s, strides "
                   f"{e.get('strides')!r} (a turn in place at {INPLACE_RATE:g} rad/s is one stride at speed 0)")
    if e.get("crouch_frac") is None or float(e["crouch_frac"]) != float(set_doc.get("crouch_frac", math.nan)):
        bad.append(f"{name}: its entry's crouch_frac is {e.get('crouch_frac')!r}, set.json's {set_doc.get('crouch_frac')!r}")


def _verify_move(d: Path, name: str, e: dict, body: str, bad: list, notes: list) -> None:
    """Items 2 and 3 of the round-A design's 4.3 for one move the set holds as a reference."""
    if e.get("pass_") is not True:
        failed = sorted(k for k, x in (e.get("gates") or {}).items() if not x)
        bad.append(f"{name}: held as a reference with pass_ {e.get('pass_')!r} (failing: {', '.join(failed) or '-'})")
    if (e.get("stage0") or {}).get("blocks_freeze"):
        bad.append(f"{name}: held as a reference although a Stage 0 gate blocked it where it was built")
    if e.get("a1_over") not in (True, False):
        bad.append(f"{name}: its entry's a1_over is {e.get('a1_over')!r}")
    move, ref = e.get("move") or {}, e.get("reference") or {}
    if e.get("stage1_sha256") is not None and (e["stage1_sha256"] != e["sha256"] or e.get("equals_stage1") is not True):
        bad.append(f"{name}: it does not hash to Stage 1's {e.get('stage1_twin')} ({e['sha256']} against "
                   f"{e['stage1_sha256']})")
    if e.get("fallback"):
        notes.append(f"{name}: built by fallback step {e['fallback'].get('step')} ({RULES['S2S-fallback'][:60]}...)")
    if move.get("builder") == "stop_two_step":
        fwd = ref.get("forward_stopped_strides")
        if ref.get("touchdowns_after_command") != moves.STOP_TOUCHDOWNS:
            bad.append(f"{name}: {ref.get('touchdowns_after_command')!r} touchdowns after the command, not "
                       f"{moves.STOP_TOUCHDOWNS}")
        if fwd is None or fwd > transitions.STOP_WITHIN_STRIDES:
            bad.append(f"{name}: stopped {fwd!r} strides after the command (at most {transitions.STOP_WITHIN_STRIDES:g})")
        lp = ref.get("accel_forward_lp1hz_peak_m_s2")
        if lp is None or lp > moves.STOP_ACCEL_LP_LIMIT_M_S2:
            bad.append(f"{name}: 1 Hz forward acceleration {lp!r} m/s^2 (at most {moves.STOP_ACCEL_LP_LIMIT_M_S2:g})")
    if move.get("builder") == "turn":
        t = e.get("turn") or {}
        total = t.get("heading_total_rad")
        if total is None or abs(abs(total) - setio.TURN_ANGLE_RAD) > moves.TURN_CLOSURE_RAD:
            bad.append(f"{name}: total heading {total!r} rad (within {moves.TURN_CLOSURE_RAD:g} of "
                       f"{setio.TURN_ANGLE_RAD!r})")
        for key in ("heading_at_bar_deg", "heading_at_slowest_clock_deg"):
            if ref.get(key) is None or ref[key] < moves.TURN_HEADING_MIN_DEG:
                bad.append(f"{name}: {key} {ref.get(key)!r} (at least {moves.TURN_HEADING_MIN_DEG:g})")
        if e.get("hip_yaw_range_use") is None or e["hip_yaw_range_use"] > moves.TWIST_FINAL:
            bad.append(f"{name}: thigh-twist use {e.get('hip_yaw_range_use')!r} (at most {moves.TWIST_FINAL:g})")
        other = e.get("tightest_other")
        if not other or other["margin_rad"] < moves.OTHER_JOINT_MARGIN_RAD:
            bad.append(f"{name}: tightest other joint {other!r} (at least {moves.OTHER_JOINT_MARGIN_RAD:g} rad)")
        if "S2A-appendage" not in (move.get("rules") or []):
            bad.append(f"{name}: S2A-appendage is not among its rules")
        if not isinstance(t.get("stride_rule"), dict) or t["stride_rule"].get("outcome") != "frozen":
            bad.append(f"{name}: its entry holds no stride rule record with the outcome 'frozen'")
        if t.get("ramp_s") != setio.RAMP_S:
            bad.append(f"{name}: its ramps last {t.get('ramp_s')!r} s, not {setio.RAMP_S:g}")
        if e.get("a1_over"):
            notes.append(f"{name}: a1_over (a neck or tail limit over at Stage 0): read under the third reading only")


# ------------------------------------------------------------------------------------------------
# 6. freeze, status
# ------------------------------------------------------------------------------------------------
def legacy_state() -> dict:
    """``stage2/sets/legacy_check.json`` against today's ``moves.py``: ``{ok, why, sha256}``."""
    doc = _read(LEGACY_CHECK)
    if doc is None:
        return dict(ok=False, why=f"{s2io.portable(str(LEGACY_CHECK))} is missing", sha256=None, moves_sha256=None)
    why = []
    if doc.get("moves_sha256") != s2io.sha256_file(s2io.module_path("moves.py")):
        why.append("it was made by another moves.py")
    for b in look.BODIES:
        rec = (doc.get("bodies") or {}).get(b)
        if rec is None or not rec.get("complete") or not all(x.get("equal") for x in rec.get("names", {}).values()):
            why.append(f"{b} is not complete with every transition equal")
    return dict(ok=not why, why="; ".join(why) or None, sha256=s2io.sha256_file(LEGACY_CHECK),
                moves_sha256=doc.get("moves_sha256"))


def refcheck_state(d: Path, docs: dict, path=None) -> dict | None:
    """The bar's kinematic check of the set's references as a record holds it (``path``; default the set
    directory's ``refcheck.json``): each reference played with perfect tracking through the bar's own readings, no
    physics, no verdict.  None where there is no record; else ``{file, sha256, pass_, failing {condition: {fails,
    toe_clearance_own_swing_H}}, not_run [...], files [...], current, why}``.  ``current``: the record is of this set and body
    and every file it read has the sha256 its entry records now (``why``: what differs).  This is a report: no rule
    makes it block a freeze, and what it says of a reference is the lead's to rule on."""
    path = Path(d) / REFCHECK if path is None else Path(path)
    if not path.is_file():
        return None
    doc = s2io.read_json(path)
    now = {f"refs/{n}.npz": e.get("sha256") for n, e in docs["refs.json"]["references"].items()}
    now.update({f"transitions/{n}.npz": e.get("sha256") for n, e in docs["transitions.json"]["transitions"].items()})
    files = (doc.get("inputs") or {}).get("files") or {}
    why = []
    if (doc.get("set"), doc.get("body")) != (docs["set.json"].get("set"), docs["set.json"].get("body")):
        why.append(f"it is of {doc.get('set')!r} {doc.get('body')!r}")
    other = sorted(f for f, v in files.items() if now.get(f) is None or now[f] != (v or {}).get("sha256"))
    if other:
        why.append(f"{len(other)} of the files it read are not the set's now ({', '.join(other[:4])}"
                   + (", ..." if len(other) > 4 else "") + ")")
    if not files:
        why.append("it read no file")
    items = doc.get("items") or {}
    failing = {k: dict(fails=sorted(t for t, ok in (x.get("ok") or {}).items() if not ok),
                       toe_clearance_own_swing_H=(x.get("toe") or {}).get("toe_clearance_own_swing_H"))
               for k, x in items.items() if x.get("status") == "done" and not x.get("pass_")}
    return s2io.plain(dict(file=s2io.portable(str(path.resolve())), sha256=s2io.sha256_file(path),
                           pass_=bool(doc.get("pass_")), failing=failing,
                           not_run=sorted(k for k, x in items.items() if x.get("status") != "done"),
                           files=sorted(files), current=not why, why="; ".join(why) or None))


def refcheck_lines(state: dict) -> list[str]:
    """What a kinematic check says, one line per failing condition (``sets verify``'s notes)."""
    head = "refcheck (each reference played kinematically through the bar's own readings; no physics)"
    if not state["current"]:
        return [f"{head}: {state['file']} is not current: {state['why']} (python -m designlab.stage2.bar refcheck)"]
    if not state["failing"]:
        return [f"{head}: every reference that was played passes its own bar item"]
    out = []
    for cond, x in state["failing"].items():
        toe = x.get("toe_clearance_own_swing_H")
        out.append(f"{head}: {cond} FAILS {', '.join(x['fails'])}"
                   + ("" if toe is None else f" (own-swing toe clearance {toe:.4g} H)")
                   + ": the reference itself does not pass this term of its bar item")
    return out


def manifest_of(docs: dict) -> dict:
    """name -> ``{kind, sha256, entry_sha256}`` (a walk: ``look_sha256`` beside them) of every available entry."""
    out = {}
    for name, e in docs["refs.json"]["references"].items():
        if e.get("available") is not False:
            out[name] = dict(kind="walk", sha256=e["sha256"], entry_sha256=setio.entry_sha256("walk", e),
                             look_sha256=e.get("look_sha256"))
    for name, e in docs["transitions.json"]["transitions"].items():
        if e.get("available") is not False:
            kind = (e.get("item") or {}).get("kind") or (e.get("move") or {}).get("builder") or "move"
            out[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256("move", e))
    return out


def freeze(set_name: str, body: str, root=setio.ROOT, allow_mixed: bool = False, allow_dirty: bool = False,
           change_after_result: str | None = None, log=print, after_check_episodes: str | None = None,
           before_any_episode: bool = False, refcheck=None) -> dict:
    """Append a freeze to ``set.json`` (``RULES["S2S-freeze"]``) and return it; ``SetError`` when the set cannot be
    frozen (the reasons, all of them).  A freeze that would add nothing appends nothing and returns the latest.
    A later freeze that adds references says when it was made, by exactly one of ``change_after_result`` (a reason:
    after a registered episode of the round; the label "change after a result", the plan's A13),
    ``after_check_episodes`` (a reason: after check episodes only, which is no change after a result; rule R4) and
    ``before_any_episode``.  ``refcheck``: the record of the bar's kinematic check to cite (default: the set
    directory's ``refcheck.json`` where there is one); it must be of the files being frozen."""
    d = setio.set_dir(set_name, body, root)
    with s2io.Lock(d / ".lock", log=log):
        v = verify(set_name, body, root)
        why = [f"sets verify: {x}" for x in v["failures"]]
        docs = setio.load(d)
        screen = docs["screen.json"]
        if screen is not None and screen.get("refused"):
            why.append(f"the screen refused {', '.join(screen['refused'])}: screen.json is not complete")
        leg = legacy_state()
        if not leg["ok"]:
            why.append(f"the legacy check is not current: {leg['why']} (python -m designlab.stage2.moves legacy --body "
                       "<pick> --record stage2/sets/legacy_check.json, both picks)")
        manifest = manifest_of(docs)
        set_doc = docs["set.json"]
        freezes = list(set_doc.get("freezes") or [])
        prev = freezes[-1]["manifest"] if freezes else {}
        everything = {**docs["refs.json"]["references"], **docs["transitions.json"]["transitions"]}
        dropped = {}
        for name, row in prev.items():
            now = manifest.get(name)
            mk = marked(everything.get(name)) if now is None else None
            if mk is not None and (mk.get("freeze") or {}).get("manifest_sha256") == freezes[-1]["manifest_sha256"]:
                # (ruling 32 (A): an entry the lead marked not available on a G9-ref record of this freeze is dropped)
                dropped[name] = dict(kind=row.get("kind"), sha256=row.get("sha256"), entry_sha256=row.get("entry_sha256"),
                                     reason=mk.get("reason"), marked_at=mk.get("at"), g9_ref=mk.get("g9_ref"),
                                     rule=MARK_RULE)
            elif mk is not None:
                why.append(f"{name}: marked not available on a G9-ref record of another freeze than the latest "
                           f"({str((mk.get('freeze') or {}).get('manifest_sha256'))[:12]}...): mark it again on a "
                           "G9-ref record of the latest freeze")
            elif now is None:
                why.append(f"{name}: an earlier freeze holds it and the set no longer does (a frozen entry is never "
                           "dropped unless the set marks it not available on the G9-ref record that named it: python "
                           "-m designlab.stage2.sets mark)")
            elif (now["sha256"], now["entry_sha256"]) != (row["sha256"], row["entry_sha256"]):
                why.append(f"{name}: an earlier freeze holds sha256 {row['sha256']} (entry {row['entry_sha256'][:12]}...), "
                           f"the set now {now['sha256']} (entry {now['entry_sha256'][:12]}...): a changed reference "
                           "takes a new name or a new set version")
        entries = {**docs["refs.json"]["references"], **docs["transitions.json"]["transitions"]}
        # every entry a build decided, with or without a reference, is of the moves.py the legacy check passed (A4)
        key = s2io.portable(str(s2io.module_path("moves.py")))
        other = sorted(n for n, e in entries.items() if e.get("built_by")
                       and (e["built_by"].get("stage2_modules") or {}).get(key) != leg["moves_sha256"])
        if other:
            why.append(f"{len(other)} entries were built by another moves.py than the legacy check's "
                       f"({', '.join(other[:6])}{', ...' if len(other) > 6 else ''}; STAGE2_75_PLAN.md A4: "
                       "moves_sha256 is the sha256 of the moves.py that built the sets): build the set again in an "
                       "empty directory")
        mixed = len({source_hash(entries[n]["built_by"]) for n in manifest if entries[n].get("built_by")}) > 1
        git = s2io.git_state(own_outputs=[str(d)])
        dirty_entries = sorted(n for n in manifest if (entries[n].get("built_by") or {}).get("dirty"))
        unknown = git["commit"] is None  # git could not tell: the tree is not known to be clean
        dirty = bool(git["dirty"] or dirty_entries or unknown)
        if mixed and not allow_mixed:
            why.append("the entries were built from differing source states (--allow-mixed records it and freezes)")
        if unknown and not allow_dirty:
            why.append("git cannot tell the state of the tree (--allow-dirty records it and freezes)")
        elif dirty and not allow_dirty:
            why.append("the tree is dirty" + (f" ({'; '.join((git['dirty_paths'] or [])[:5])})" if git["dirty"] else "")
                       + (f"; built on a dirty tree: {', '.join(dirty_entries[:6])}" if dirty_entries else "")
                       + " (--allow-dirty records it and freezes)")
        # when the freeze was made: a later freeze that adds references says so (A13; rule R4)
        added = sorted(set(manifest) - set(prev))
        said = [k for k, v in (("--change-after-result", change_after_result),
                               ("--after-check-episodes", after_check_episodes),
                               ("--before-any-episode", before_any_episode)) if v]
        if len(said) > 1:
            why.append(f"{' and '.join(said)} are given: a freeze was made at one time")
        if freezes and added and not said:
            why.append(f"this freeze would add {', '.join(added[:6])}{', ...' if len(added) > 6 else ''} to freeze "
                       f"{freezes[-1]['index']}: say when it is made, with --change-after-result REASON (after a "
                       "registered episode of the round: the label 'change after a result', STAGE2_75_PLAN.md A13), "
                       "--after-check-episodes REASON (after check episodes only: rule R4) or --before-any-episode")
        # the bar's kinematic check of the references, where there is a record of it: it must be of these files
        cited = None
        if refcheck is not None and not Path(refcheck).is_file():
            why.append(f"{refcheck}: no such refcheck record")
        elif refcheck is not None or (d / REFCHECK).is_file():
            cited = refcheck_state(d, docs, refcheck)
            beyond = sorted(set(cited["files"]) - {f"{'refs' if r['kind'] == 'walk' else 'transitions'}/{n}.npz"
                                                   for n, r in manifest.items()})
            if not cited["current"] or beyond:
                why.append(f"the refcheck record {cited['file']} is not of the files being frozen ("
                           + (cited["why"] or f"it read {', '.join(beyond[:4])}")
                           + "): run python -m designlab.stage2.bar refcheck again")
        if why:
            raise SetError(f"{d} is not frozen: " + "; ".join(why))
        if freezes and manifest == prev:
            log(f"{set_name} {body}: nothing to add to freeze {freezes[-1]['index']}")
            return freezes[-1]
        rec = dict(index=len(freezes), at=s2io.now_utc(), commit=git["commit"], manifest=manifest,
                   manifest_sha256=s2io.sha256_json(manifest), legacy_check_sha256=leg["sha256"],
                   mixed_sources=bool(mixed), dirty=bool(dirty), added=added)
        if change_after_result:
            rec["change_after_result"] = str(change_after_result)
        if after_check_episodes:
            rec.update(after_check_episodes=True, reason=str(after_check_episodes))
        if before_any_episode:
            rec["before_any_episode"] = True
        if dropped:  # (ruling 32 (A): each dropped entry with its mark and the G9-ref record that named it)
            rec["dropped"] = dropped
            for name, x in dropped.items():
                log(f"{set_name} {body}: {name} dropped: {x['reason']}")
        if cited is not None:
            rec["refcheck"] = {k: cited[k] for k in ("file", "sha256", "pass_", "failing", "not_run")}
            for line in refcheck_lines(cited):
                log(f"{set_name} {body}: {line}")
        set_doc["freezes"] = freezes + [rec]
        s2io.write_json(d / "set.json", set_doc)
        log(f"{set_name} {body}: freeze {rec['index']}: {len(manifest)} entries, manifest {rec['manifest_sha256'][:12]}...")
        return rec


def status(set_name: str, body: str, root=setio.ROOT) -> dict:
    """What the set holds, read now: ``{dir, built, frozen, items {name: state}, missing, not_available, blocked,
    lead_in}``;
    an item's state is ``built``, ``built (frozen)``, ``array missing``, ``stale``, ``missing`` or ``not available:
    <reason>``."""
    d = setio.set_dir(set_name, body, root)
    its = items(set_name, body)
    docs = load_docs(d) if d.is_dir() else None
    out = dict(dir=str(d), built=docs is not None, frozen=None, items={}, missing=[], not_available={}, blocked=None,
               lead_in=None)
    if docs is None:
        out["missing"] = [i.name for i in its if not i.optional]
        out["items"] = {i.name: "not built (optional: built when it is named)" if i.optional else "missing" for i in its}
        return out
    fz = setio.frozen(docs["set.json"])
    rows = {} if fz is None else fz["entries"]
    out["frozen"] = None if fz is None else dict(index=fz["index"], manifest_sha256=fz["manifest_sha256"],
                                                 entries=len(rows))
    out["blocked"] = blocked_list(docs["set.json"].get("blocked")) or None
    for i in its:
        e = _entry_of(docs, i.kind, i.name)
        if e is None and i.optional:
            state = "not built (optional: built when it is named)"
        elif e is None:
            state = "missing"
            out["missing"].append(i.name)
        elif e.get("available") is False:
            state = f"not available: {e.get('reason')}"
            out["not_available"][i.name] = e.get("reason")
        elif not _files_ok(d, e, i.kind == "walk"):
            state = "array missing"
        elif i.name in rows:
            state = "built (frozen)"
        else:
            state = "built" if current(d, docs, i, rows) else "stale"
        out["items"][i.name] = state
    for name, e in docs["transitions.json"]["transitions"].items():
        if name not in out["items"]:
            out["items"][name] = (f"built (made by the item {e.get('group')})" if e.get("available") is not False
                                  else f"not available: {e.get('reason')}")
    out["lead_in"] = docs["set.json"].get("lead_in")
    return out


# ------------------------------------------------------------------------------------------------
# Marking a frozen entry not available on the G9-ref record that named it (ruling 32 (A); MARK_RULE)
# ------------------------------------------------------------------------------------------------
#: ``gates2.SCHEMAS["g9_ref"]`` (``gates2`` is not imported here: it selects MuJoCo's GL backend on import).
G9_REF_SCHEMA = "s2-gate-g9-ref-1"


def marked(e: dict | None) -> dict | None:
    """The mark of an entry that ``mark`` set not available (its ``marked``), or None."""
    if isinstance(e, dict) and e.get("available") is False and isinstance(e.get("marked"), dict):
        return e["marked"]
    return None


def _verify_mark(name: str, mk: dict, rows: dict, set_doc: dict, bad: list, notes: list) -> None:
    """``verify`` of a marked entry: its mark names a G9-ref record (path and sha256) and the freeze it was made on;
    while the latest freeze still lists it, that is the freeze of the mark (the next freeze drops it)."""
    g9 = mk.get("g9_ref") or {}
    if not (g9.get("path") and g9.get("sha256") and mk.get("reason") and isinstance(mk.get("freeze"), dict)):
        bad.append(f"{name}: marked not available without its reason, its G9-ref record (path and sha256) or the "
                   "freeze it was made on")
        return
    if name in rows:
        last = (set_doc.get("freezes") or [{}])[-1]
        if mk["freeze"].get("manifest_sha256") != last.get("manifest_sha256"):
            bad.append(f"{name}: the latest freeze lists it and it was marked not available on another freeze")
        else:
            notes.append(f"{name}: marked not available on the G9-ref record {g9['path']} (sha256 "
                         f"{g9['sha256'][:12]}...); freeze {last.get('index')} lists it and the next freeze drops it")


def _g9_ref_naming(path: Path, set_name: str, body: str, ref: str, latest: dict) -> tuple[dict, list]:
    """(the record's citation ``{path, sha256}``, its reading of ``ref``): ``SetError`` unless ``path`` is a G9-ref
    record of this set and body, its own sha256 intact, made on the set's latest freeze (``latest``), naming no
    failing blocking item, that names ``ref`` among the failing items' references still available in the set."""
    path = Path(path)
    try:
        doc = s2io.read_json(path) if path.is_file() else None
    except ValueError:
        doc = None
    if not isinstance(doc, dict):
        raise SetError(f"{path}: no G9-ref record can be read there")
    rest = {k: v for k, v in doc.items() if k != "sha256"}
    if doc.get("schema") != G9_REF_SCHEMA or "sha256" not in doc or s2io.sha256_json(rest) != doc["sha256"]:
        raise SetError(f"{path}: not a G9-ref record (schema {doc.get('schema')!r}, or its sha256 is not its own)")
    if (doc.get("set"), doc.get("body")) != (set_name, body):
        raise SetError(f"{path}: a G9-ref record of {doc.get('set')}/{doc.get('body')}, not of {set_name}/{body}")
    if (doc.get("set_freeze") or {}).get("manifest_sha256") != latest["manifest_sha256"]:
        raise SetError(f"{path}: a G9-ref record made on another freeze of the set "
                       f"({str((doc.get('set_freeze') or {}).get('manifest_sha256'))[:12]}..., the latest is "
                       f"{latest['manifest_sha256'][:12]}...)")
    a = (doc.get("parts") or {}).get("a") or {}
    if a.get("blocked"):
        raise SetError(f"{path}: the record names failing blocking items ({', '.join(a['blocked'])}): the set goes to "
                       "the lead and nothing is marked")
    if ref not in (a.get("still_available") or []):
        raise SetError(f"{path}: the record does not name {ref} among the failing items' references still available "
                       f"in the set ({', '.join(a.get('still_available') or []) or 'none'})")
    reading = [dict(item=x.get("item"), failing=x.get("failing")) for x in a.get("failing_references") or []
               if x.get("reference") == ref]
    return dict(path=s2io.portable(str(path.resolve())), sha256=s2io.sha256_file(path)), reading


def mark(set_name: str, body: str, name: str, reason: str, g9_ref, root=setio.ROOT, log=print) -> dict:
    """Mark the entry ``name`` of a frozen set not available, with the lead's ``reason`` and the G9-ref record
    ``g9_ref`` that named it (``MARK_RULE``; the lead's ruling 32 (A) of 17 Nov 2026, after ruling 14).  The entry
    keeps its definition and provenance and gains ``available`` false, the reason, its numbers and ``marked``; it
    names no file.  ``set.json`` lists it under ``not_available`` (and a turn's stride rule record no longer holds it
    as a reference).  The next ``freeze`` drops it and records the drop.  Returns the mark; ``SetError`` when the
    entry cannot be marked so (nothing is written)."""
    if not str(reason or "").strip():
        raise SetError("a mark needs the lead's reason")
    d = setio.set_dir(set_name, body, root)
    with s2io.Lock(d / ".lock", log=log):
        docs = load_docs(d)
        if docs is None:
            raise SetError(f"{d}: the set has no records")
        set_doc = docs["set.json"]
        if setio.frozen(set_doc) is None:
            raise SetError(f"{set_name} {body}: the set is not frozen (G9-ref reads a frozen set; an entry of a set that "
                           "is not frozen is built again, not marked)")
        latest = set_doc["freezes"][-1]
        refs_, trs = docs["refs.json"]["references"], docs["transitions.json"]["transitions"]
        sub, target = ("refs", refs_) if name in refs_ else ("transitions", trs) if name in trs else (None, None)
        if target is None:
            raise SetError(f"{set_name} {body}: the set has no entry {name!r}")
        e = target[name]
        if e.get("available") is False:
            raise SetError(f"{set_name} {body}: {name} is already not available ({e.get('reason')})")
        ref = e["file"]
        cite, reading = _g9_ref_naming(Path(g9_ref), set_name, body, ref, latest)
        users = sorted(n for n, x in {**refs_, **trs}.items() if n != name and x.get("available") is not False
                       and name in (x.get("inputs") or {}))
        if users:
            raise SetError(f"{set_name} {body}: {', '.join(users)} were built from {name} and the set holds them as "
                           "references: they are marked first (each on a G9-ref record that names it), or it goes "
                           "to the lead")
        kind = "walk" if sub == "refs" else "move"
        was = {k: e[k] for k in ("file", "sha256", "look_file", "look_sha256") if k in e}
        was["entry_sha256"] = setio.entry_sha256(kind, e)
        mk = dict(rule=MARK_RULE, at=s2io.now_utc(), reason=str(reason), g9_ref=cite,
                  freeze=dict(index=latest["index"], manifest_sha256=latest["manifest_sha256"]), was=was)
        why = (f"{reason} (marked not available by the lead on the G9-ref record {cite['path']}, sha256 "
               f"{cite['sha256'][:12]}..., of freeze {latest['index']})")
        new = {k: v for k, v in e.items() if k not in ("file", "sha256", "look_file", "look_sha256")}
        new.update(available=False, reason=why, numbers=dict(g9_ref=reading), marked=mk)
        target[name] = refs._plain(new)
        set_doc["not_available"][name] = dict(reason=why, numbers=new["numbers"])
        set_doc["look_entries"].pop(name, None)
        for label, rec in (set_doc.get("stride_rules") or {}).items():
            if name in (rec.get("references") or {}):
                rec["references"][name] = False
                rec["no_reference"] = rec.get("no_reference") or why
        for rec_name in ("refs.json", "transitions.json", "set.json"):
            s2io.write_json(d / rec_name, docs[rec_name])
        log(f"{set_name} {body}: {name} marked not available ({reason}); the next freeze drops it")
        return mk


# ------------------------------------------------------------------------------------------------
# The look sets' tail phases
# ------------------------------------------------------------------------------------------------
def make_tail_phases(body: str, log=print, path=None) -> dict:
    """F9's tail phase of every rung the look ladder does not hold, on a pick (``LOOK_RULES["S2L-tail-phase"]``):
    ``look.tail_check`` on the rung's V4 walk (six generations of one stride each), written into
    ``stage2/sets/V6/tail_phase.json`` under the pick with the check's numbers; the other bodies' records are kept.
    Prescribed motion on the round-two model; no episode.  Returns the pick's record."""
    s2io.require_one_thread()
    if base_body(body) != body:
        raise SetError(f"{body}: a head body takes its pick's tail phases (make them on {base_body(body)})")
    path = SET_TAIL if path is None else Path(path)
    H, model = body_facts(body)
    src = look.source(body)
    amp = look.feature_values(body, H, "nominal")["F9F10"]["tail_amp"]
    out = {}
    for v in setio.SPEEDS:
        name = setio.walk_name(v)
        if name in KNOTS:
            continue
        v4 = look_chain(body, H, v0_spec(body, H, v), None, upto="V4")
        t0 = time.perf_counter()
        with s2io.r2():
            tc = look.tail_check(src, v4, amp=amp)
            prov = s2io.built_by(["sets.py", "look.py", "s2io.py"], own_outputs=[str(path)])
        out[name] = s2io.plain(dict(
            tail_phase_rad=float(tc["best_phase_rad"]), speed_m_s=float(v), body_model_sha256=model,
            source="look.tail_check on this rung's V4 walk (designlab.stage2.sets make_tail_phases)",
            v4_spec=look._plain(asdict(v4)),
            check={k: tc[k] for k in ("amp_rad", "fit_max_residual_nm", "predicted_peak_nm", "confirmed", "no_tail",
                                      "best_phase_cycles_from_today")}, built_by=prov))
        log(f"{body} {name}: tail phase {tc['best_phase_rad']:.6f} rad (confirmed peak "
            f"{tc['confirmed']['peak_nm']:.4f} N m, no tail {tc['no_tail']['peak_nm']:.4f}); "
            f"{time.perf_counter() - t0:.0f} s")
    doc = _read(path) or dict(what="F9's tail phases of the look sets' walks at the rungs the look ladder does not "
                                   "hold (stage2/look/tail_phase.json holds nominal and v0.4)",
                              rule=LOOK_RULES["S2L-tail-phase"], bodies={})
    doc["bodies"][body] = out
    s2io.write_json(path, doc)
    return out


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    s2io.require_one_thread()
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.sets", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tail-phase", help="F9's tail phases of the look sets' rungs on a pick (LOOK_RULES)")
    p.add_argument("--body", required=True, choices=list(look.BODIES))
    for cmd in ("build", "verify", "freeze", "status", "mark"):
        p = sub.add_parser(cmd)
        p.add_argument("--set", required=True, choices=[*SETS, *LOOK_SETS])
        p.add_argument("--body", required=True, choices=list(stage1.BODIES))
        p.add_argument("--root", default=str(setio.ROOT), help="the directory that holds the sets (default: stage2/sets)")
        if cmd == "build":
            p.add_argument("--only", action="extend", nargs="+", metavar="NAME", help="items or groups to build")
            p.add_argument("--procs", type=int, default=2)
        if cmd == "mark":
            p.add_argument("--name", required=True, help="the entry the G9-ref record names")
            p.add_argument("--reason", required=True, help="the lead's reason")
            p.add_argument("--g9-ref", required=True, metavar="FILE", help="the G9-ref record that named it")
        if cmd == "freeze":
            p.add_argument("--allow-mixed", action="store_true")
            p.add_argument("--allow-dirty", action="store_true")
            when = p.add_mutually_exclusive_group()
            when.add_argument("--change-after-result", metavar="REASON",
                              help="a later freeze made after a registered episode of the round (labelled so)")
            when.add_argument("--after-check-episodes", metavar="REASON",
                              help="a freeze made after check episodes only (no change after a result)")
            when.add_argument("--before-any-episode", action="store_true",
                              help="a later freeze made before any episode on the set")
            p.add_argument("--refcheck", metavar="FILE", help="the record of bar refcheck to cite (default: the set "
                                                              f"directory's {REFCHECK}, where there is one)")
    a = ap.parse_args(argv)

    def say(line: str) -> None:
        print(line, flush=True)

    try:
        if a.cmd == "tail-phase":
            make_tail_phases(a.body, log=say)
            return 0
        if a.cmd == "build":
            r = build(a.set, a.body, a.root, a.only, a.procs, log=say)
            say(f"{a.set} {a.body}: built {len(r['ran'])}, kept {len(r['kept'])}, recovered {len(r['recovered'])}"
                + (f", arrays of frozen entries put back: {', '.join(r['restored'])}" if r["restored"] else "")
                + f"; {r['seconds']:.0f} s")
            for name, why in r["not_available"].items():
                say(f"  not available: {name}: {why}")
            say(f"  lead-in: {r['lead_in']}")
            if r["blocked"]:
                say(f"  the set is stopped: {'; '.join(b['reason'] for b in r['blocked'])}"
                    + (f"; not built: {', '.join(r['skipped'])}" if r["skipped"] else ""))
                return 1
            return 0
        if a.cmd == "verify":
            r = verify(a.set, a.body, a.root)
            for line in r["notes"]:
                say(f"  note: {line}")
            for line in r["failures"]:
                say(f"  FAILS: {line}")
            say(f"{a.set} {a.body}: verify {'ok' if r['ok'] else 'FAILED'}: {len(r['available'])} references, "
                f"{len(r['not_available'])} not available, {len(r['failures'])} failures"
                + ("" if r["frozen"] is None else f"; freeze {r['frozen']['index']}"))
            return 0 if r["ok"] else 1
        if a.cmd == "mark":
            mk = mark(a.set, a.body, a.name, a.reason, a.g9_ref, a.root, log=say)
            say(json.dumps(mk, indent=1))
            return 0
        if a.cmd == "freeze":
            r = freeze(a.set, a.body, a.root, a.allow_mixed, a.allow_dirty, a.change_after_result, log=say,
                       after_check_episodes=a.after_check_episodes, before_any_episode=a.before_any_episode,
                       refcheck=a.refcheck)
            say(json.dumps({k: v for k, v in r.items() if k != "manifest"}, indent=1))
            return 0
        r = status(a.set, a.body, a.root)
        for name, state in r["items"].items():
            say(f"  {name}: {state}")
        say(f"{a.set} {a.body}: " + ("not built" if not r["built"] else
                                     ("not frozen" if r["frozen"] is None else f"freeze {r['frozen']['index']}"))
            + (f"; missing {len(r['missing'])}" if r["missing"] else "") + f"; lead-in {r['lead_in']}"
            + (f"; stopped: {'; '.join(b['reason'] for b in r['blocked'])}" if r["blocked"] else ""))
        return 0
    except SetError as exc:
        say(f"{type(exc).__name__}: {exc}")
        return 2 if isinstance(exc, GateError) else 1
    except s2io.S2Error as exc:
        say(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    from designlab.stage2 import sets as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
