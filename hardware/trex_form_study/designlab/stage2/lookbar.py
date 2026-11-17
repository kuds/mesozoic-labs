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
  (``<companion>.stale/<12 hex>.json``), and the summary (``look/summary.json``) lists the record under
  ``refused_records`` with what became of its companion.

**Outcome of an episode** (rules U6, M1; ``companion_episode``).  *read*: the bar judged it and it did not fall.
*fell*.  *unread*: the bar could not read it (``bar.Unreadable``), it did not fall, and the row is a policy row (its
inner row has ``provenance()``): not succeeding in any reading, with the reason ``clock_behind`` when its mean
clock factor is below 1 and ``unreadable`` otherwise.  *error*: anything else that kept the bar from judging it
(an infrastructure fault, or an unreadable episode of a row without a policy, for which ``STAGE2_75_PLAN.md`` C8
stands), and, on a look set, a walk or turn item whose look reading cannot be given (rule U5: a reference that
cannot be played, or whose own window holds too few steps, is a fault of the definition, never "no look reading").

**Readings** (rule U4; the lead's rulings of 16 Nov 2026).  Reading 1 counts on the look sets ``LOOK_SETS``.  There
a walk or turn episode succeeds with the look when the bar's ``succeeded`` holds and all four terms do; a nudge or
stop episode's reading 1 is its reading 2, labelled; a stop episode that fails ``on_command_at_stop`` succeeds in no
reading of the companion.  On any other set the look numbers and the term are recorded and decide nothing:
``succeeded_look`` and ``counts_look`` are null and the companion's counts are the bar's own, episode for episode
(also for an episode the bar could not read: the bar's count stands there).

**A block** (schema ``s2-look-ep-1``; the sidecar file): ``row`` (the bar row's name), ``look_row``, ``condition``,
``seed``, ``set``, ``judge``, ``steps``, ``fell``, ``survival_s``, ``planned_s``, ``clock {mean_factor, min_factor,
t_ref_reached, t_ref_planned, ...}``, then the look reading: ``read`` (whether there is one), ``reason`` (why an
item is not read: a nudge, a stop or a stock item, a fall, a window not reached), ``fault`` (what went wrong in the
reading itself), ``window``, ``window_ref``, ``own``, ``reference``, ``terms``, ``not_read``, ``too_few_steps``,
``look_ok``, ``compare`` (the ratios and differences), ``air_runs``; ``on_command_at_stop`` (a stop item);
``policy_check`` (a policy row's ``provenance()``); ``state_digest``; ``row_sha256``.  **A companion episode**:
``condition``, ``seed``, ``outcome``, ``reason``, ``fell``, ``survival_s``, ``planned_s``, the four readings'
``succeeded...``, ``failed_terms``, ``look`` (the block's reading, None where there is none), ``clock``,
``on_command_at_stop``, ``error``, ``state_digest``, ``policy_check`` and the bar's own verdicts under ``bar``.

No episode here is a record by itself: ``run`` writes the bar's records and their companions; ``episodes`` writes
nothing but the traces it is asked for.  Which sets an episode may run on is the caller's to keep (rule X1), and
the bar's runner keeps its own rule of what may run where (``bar.RULES["checks"]``: as built, on a reference set
that is none of the 75 cm round's three it lets only rows of the controller ``R0`` run, so ``run`` is refused there
for ``R0L``, ``R5``, ``R6`` and ``R2L`` rows; ``episodes`` does not go through the runner).

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
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
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
        "stands; on a look set, a walk or turn item whose look reading cannot be given)"),
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
        "companion found at the refused record's path is moved aside to <companion>.stale/<first 12 hex of the "
        "file's sha256>.json unless it names the bar record as it is (bar_record.sha256): that one was made of this "
        "bar record while its sidecars were there, and stays.  The summary lists every refused record with what "
        "became of its companion"),
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
    gives the bar row's, ``R0L@10ms``).  ``last``: the last episode's block."""

    def __init__(self, inner: panels.Row, sidecar: str | None = None, name: str | None = None):
        if isinstance(inner, LookRow):
            raise ValueError(f"row {inner.name!r} is a LookRow already")
        self.inner = inner
        self.name = name or inner.name
        self.hard_only = bool(inner.hard_only)
        self.sidecar = None if sidecar is None else str(sidecar)
        self.sidecar_row = self.name
        self.last: dict | None = None
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
        return dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name, sidecar=self.sidecar,
                    sidecar_row=self.sidecar_row, inner=self.inner.spec())

    @classmethod
    def from_spec(cls, s: dict) -> "LookRow":
        row = cls(panels.row_from_spec(s["inner"]), s.get("sidecar"), s["name"])
        row.sidecar_row = s.get("sidecar_row") or row.name
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
            entry["fault"] = f"{type(exc).__name__}: {exc}"
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
        block = dict(
            schema=SIDECAR_SCHEMA, row=self.sidecar_row, look_row=self.name, condition=cond.key, seed=int(seed),
            set=cond.set if is_bar else bar.STOCK_SET, judge=judge, relaxed=list(relax),
            steps=int(getattr(rec, "steps", 0) or 0), fell=fell, survival_s=float(out["seconds"]),
            planned_s=float(b.seconds), fall_cause=out.get("fall_cause"), clock=None, read=False, reason=None,
            fault=None, window=None, window_ref=None, own=None, reference=None, terms=None, not_read=None,
            too_few_steps=None, look_ok=None, compare=None, air_runs=None, on_command_at_stop=None,
            policy_check=None, state_digest=None, row_sha256=self.row_sha256(), at=s2io.now_utc())
        faults = []
        for key, fn in (("clock", lambda: clock_block(out, rec, b.seconds)),
                        ("state_digest", lambda: state_digest(b.p)), ("policy_check", self.provenance)):
            try:
                block[key] = fn()
            except Exception as exc:  # noqa: BLE001 - recorded, never raised (RULES["wrapper"])
                faults.append(f"{key}: {type(exc).__name__}: {exc}")
        if is_bar and judge == STOP_JUDGE:
            try:
                block["on_command_at_stop"] = on_command_at_stop(b, rec)
            except Exception as exc:  # noqa: BLE001
                faults.append(f"{ON_COMMAND_TERM}: {type(exc).__name__}: {exc}")
        if not is_bar:
            block["reason"] = "a stock condition has no look reading"
        elif judge not in READ_JUDGES:
            block["reason"] = "nudge and stop items are not read (rule U2)"
        else:
            try:
                self._read(block, b, out, rec, cond)
            except Exception as exc:  # noqa: BLE001 - in the form of the other faults: no traceback, so no path of
                faults.append(f"look reading: {type(exc).__name__}: {exc}")  # this machine in a sidecar or a record
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
# One companion episode
# ------------------------------------------------------------------------------------------------
_UNREADABLE = re.compile(r"^(?:[\w.]*\.)?Unreadable: ")


def is_unreadable(error) -> bool:
    """Whether an episode's error text is the bar's ``Unreadable``: its last line names that exception (a
    traceback's last line, or the line ``episodes`` writes)."""
    lines = str(error).strip().splitlines() if error else []
    return bool(lines) and _UNREADABLE.match(lines[-1]) is not None


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
    return dict(
        condition=condition, seed=int(seed), outcome=outcome, reason=reason, unreadable=unreadable,
        fell=None if block is None and (e is None or (e or {}).get("error")) else bool(src.get("fell")),
        survival_s=float(src.get("survival_s") or 0.0), planned_s=src.get("planned_s"),
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
    ``row``, ``set``, ``start_seed`` and ``wall_s``."""
    s2io.require_one_thread()
    rows = dict(rows)
    looks = {name: look_row_of(row) for name, row in rows.items()}
    plain = [name for name, look in looks.items() if look is None]
    if plain or not rows:
        raise LookBarError(f"episodes takes rows that hold a LookRow (lookbar.rows): {plain or 'none given'}")
    pairs = [(str(k), int(s)) for k, s in pairs]
    opened = open_set(body, set_name, [k for k, _ in pairs], root, stage1_run)
    ctx, conds, missing = opened["ctx"], opened["conds"], opened["missing"]
    look_set = set_name in LOOK_SETS
    out = []
    for name, row in rows.items():
        look = looks[name]
        for key, seed in pairs:
            cond = conds[key]
            if key in missing:
                out.append(dict(
                    row=name, set=set_name, condition=key, seed=seed, outcome=NOT_RUN,
                    reason=str(missing[key].get("reason")), reference=missing[key].get("reference"), fell=None,
                    survival_s=0.0, planned_s=None, succeeded=False, succeeded_but_A1=False, succeeded_look=False,
                    succeeded_but_A1_look=False, failed_terms=[], look=None, clock=None, error=None))
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


def companion(path, rec: dict, out, side_root, look_set: bool) -> dict:
    """The companion record of one bar record (``rec``, read from ``path``; schema ``s2-look-1``).
    ``LookBarError`` where a judged episode's sidecar is missing or is not that episode's
    (``RULES["sidecar"]``)."""
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
        policy=_policy_block(blocks), sidecars=dict(root=Path(os.path.relpath(side_root, out)).as_posix(),
                                                    sha256=hashes),
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
    ``<companion>.stale/<first 12 hex of the file's sha256>.json``."""
    path = Path(path)
    return path.with_name(path.name + ".stale") / f"{str(sha256)[:12]}.json"


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


def collect(out, set_name: str, body: str, block: str = bar.HEADLINE_BLOCK, control: str | None = None) -> dict:
    """The companion records of every bar record of a ``LookRow`` row under
    ``<out>/<set>/<body>/<block>/eval/`` (``companion``), and their summary ``.../look/summary.json`` (returned).
    A companion whose bar record, sidecars and sources are unchanged is left as it is.  A record with an episode
    whose sidecar is missing or is not that episode's gets no companion: a companion of another bar record found
    at its path is moved aside (``stale_path``; one that names the bar record as it is stays), the summary lists
    the record under ``refused_records``, every other record's companion is written, then ``LookBarError`` lists
    the refusals."""
    out = Path(out)
    bdir = bar.block_dir(out, set_name, body, block, control)
    edir, side_root, ldir = bdir / "eval", bdir / "look_partial", bdir / "look"
    look_set = set_name in LOOK_SETS
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
                doc = companion(q, rec, out, side_root, look_set)
            except LookBarError as exc:
                refused.append(str(exc))
                refused_cells.append(_refused(q, rec, path, out, str(exc)))
                continue
            old = _read_json(path)
            if old is not None and old.get("inputs_sha256") == doc["inputs_sha256"]:
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
                fingerprint_max_abs_diff=(doc["policy"] or {}).get("fingerprint_max_abs_diff")))
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
    then ``collect``.  ``kw`` goes to ``bar.evaluate`` (``episodes``, ``task``, ``rates``, ``root``,
    ``stage1_run``, ...).  Returns the look summary with the bar's own summary under ``bar``.  The bar's own rules
    of what may run where apply unchanged (``bar.RULES["checks"]``)."""
    out = Path(out)
    control_ = kw.get("control")
    side = bar.block_dir(out, set_name, body, block, control_) / "look_partial"
    built = rows(body, list(row_names), sidecar=str(side), stage1_run=kw.get("stage1_run", bar.STAGE1_RUN))
    kw.setdefault("attribute", "none")
    doc = bar.evaluate(set_name, body, out, built, conditions, block, procs, **kw)
    return dict(collect(out, set_name, body, block, control_), bar=doc)


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


__all__ = ["COMPANION_SCHEMA", "DELAYED_KEYS", "LOOK_SETS", "LookBarError", "LookRow", "RULES", "SIDECAR_SCHEMA",
           "air_runs", "clock_block", "collect", "companion", "companion_episode", "companion_path", "counts_of",
           "episodes", "is_look_record", "is_unreadable", "look_of", "look_row_of", "main", "on_command_at_stop",
           "open_set", "record_row_sha256", "rows", "rule_texts", "run", "sidecar_path", "stale_path", "state_digest",
           "summary_text"]

if __name__ == "__main__":
    from designlab.stage2 import lookbar as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
