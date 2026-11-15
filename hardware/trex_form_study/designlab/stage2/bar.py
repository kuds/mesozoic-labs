"""Stage 2's bar: its conditions, plans, rows, judge, runner and records, on the panels' classes.

Stage 1 judged controllers with panels (``designlab/stage1/panels.py``).  The bar is a new set of conditions on a
*reference set* (``stage2/sets/<set>/<body>/`` in Stage 1's run-directory layout, read through ``setio``): a walk
judged on distance and line, a 90 degree turn judged on heading at a fixed time after its command, nudges, and a stop
judged on touchdowns after its command; a servo delay per row, two floor grips, two seed blocks and three readings of
success.  This module is an own runner on the panels' classes (``Condition``, ``Context``, ``Built``, ``Row``,
``tune.Pool``); it changes no byte of Stage 1.

* **Conditions** (``BarCondition``, ``conditions``): 16 gated per set and body (``walk/v0.4``, ``turn/q90_left``,
  ``turn/q90_right``, ``nudge/forward|backward|left|right``, ``stop/two_step``, each also with ``_mu0.54``) and the
  reported ones the set has references for.  The pseudo-set ``S1`` (a Stage 1 run directory through
  ``setio.stage1_view``) takes Stage 1's own ``panels.Condition``: built by ``panels.Context.built``, judged by
  ``panels.judge``, recorded with ``definition.stock = true``.
* **Plans** (``BarPlanner``): every item starts with the walk item's lead-in and takes the walk's seed slot; lengths
  hold at the slowest clock and tracks at the fastest (``RULES["plans"]``).
* **Rows** (``Delayed``, ``rows``): any ``panels.Row`` inside ``speedfit.servo_delay``: ``R2/42@10ms``, ``R0@10ms``.
* **Judge** (``judge``, ``run_episode``): ``readings``' functions on the live recorder; two success vectors per
  episode (reading 2: every limit; reading 3: the sustained-drive term on the leg drives only).
* **Records** (``record_key``, ``condition_record``): one JSON per set, body, block, row and condition with rules,
  provenance and hashes.  A finished record with another key is never overwritten (``--supersede`` keeps it beside
  the new one).  Episodes are written one by one by their workers and merged when a condition's seeds are complete,
  so a killed run resumes per episode.
* **Runner** (``evaluate``): ``tune.Pool`` with ``BarWorker``; attribution by decision 20; replication on the second
  seed block of what passed on the first; the budget ledger of a task (``Budget``).
* ``dry_run`` (cost from measured rates; writes nothing), ``refcheck`` (each reference played kinematically through
  the bar's readings), ``reread`` (stock episodes of Stage 1 re-run, their traces checked against the committed
  records' sha256, with the own-swing toe clearance read beside Stage 1's).

Run from hardware/trex_form_study (one BLAS thread)::

    python -m designlab.stage2.bar {run|dry-run|summary|refcheck|reread} --set S --body B --rows R2/42@10ms,R0@10ms
        [--conditions walk/v0.4,...] [--block seeds3042|seeds5042] [--passed-of seeds3042] [--procs 2]
        [--attribute judged|all|none] [--no-traces] [--no-traces-for R0] [--episodes 20] [--task 3|4] [--unit NAME]
        [--cost FILE] [--control omega_s1] [--supersede --supersede-reason TEXT] [--dev-episodes FILE]
        [--dev-reason TEXT] [--set-root DIR] [--stage1-run DIR] [--records DIR] [--row-module M] --out DIR

This file is hashed into every bar record's key (``bar_sources``), with ``readings.py``, ``setio.py`` and Stage 1's
``speedfit.py``.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # before any numpy import: one core per process
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..stage1 import control, env, estimator, metrics, panels, plant, speedfit, tune  # noqa: E402
from ..stage1.refs import Reference  # noqa: E402
from ..stage1.transitions import N_FRAMES, Transition, stop_command_phase  # noqa: E402
from . import readings, s2io, setio  # noqa: E402

SCHEMA = "bar-1"
RATE = panels.RATE
#: The seed blocks: the headline block (the study's seeds) and the replication block.
BLOCKS = {"seeds3042": tuple(range(3042, 3062)), "seeds5042": tuple(range(5042, 5062))}
HEADLINE_BLOCK, REPLICATION_BLOCK = "seeds3042", "seeds5042"
RUNS = panels.RUNS
MIN_SUCCEED = panels.MIN_SUCCEED
#: The pseudo-set of a Stage 1 run directory, and the run directory the rows read their gains from.
STOCK_SET = "S1"
STAGE1_RUN = s2io.STUDY / "stage1" / "runs" / "2026-11-08"
#: The three sets of the 75 cm round: on these a run is a task's unit (``--task``) or a listed build check
#: (``RULES["checks"]``).
PREREGISTERED_SETS = ("V0", "F25F9", "F25F9cl")
CHECK_SET = "V0"
MU_LOW = panels.MU_LOW
GRIPS = ((None, ""), (MU_LOW, f"_mu{MU_LOW:g}"))
BEHAVIORS = ("walk", "turn", "nudge", "stop")
BAR_JUDGES = ("bar_walk", "bar_turn", "bar_nudge", "bar_stop")
BAR_WALK = setio.walk_name(setio.BAR_SPEED)
#: Nudges: the pulse's length, its share of the Stage 0 push bound (gated; reported), its onset in strides of the
#: walk after the walk's segment begins (plus a uniform draw of one stride), the recovery limit and the hold.
PUSH_S = panels.PUSH_S
NUDGE_FRAC = 0.25
NUDGE_FRACS_REPORTED = (0.375, 0.5)
NUDGE_AT_STRIDES = 2.0
NUDGE_SPREAD_STRIDES = 1.0
RECOVERY_S = panels.RECOVERY_S
HOLD_S = panels.PUSH_HOLD_S
#: Strides of the walk before a turn's command (a right turn half a stride more: its command is a right lift-off)
#: and before a stop (the two-touchdown stop's command is a right lift-off; Stage 1's stop file begins on a stride).
TURN_LEAD_STRIDES = {"left": 2.0, "right": 2.5}
STOP_LEAD_STRIDES = {2: 2.5, 1: 2.0}
#: The stop: the trace holds this long after the planned rest; every commanded item's episode ends this much later.
STOP_HOLD_S = 2.0
END_MARGIN_S = 0.2
#: The slowest and the fastest clock of a tracker row against the commanded one (``control.PHASE_RATE_LIMIT``).
SLOW = 1.0 - control.PHASE_RATE_LIMIT
FAST = 1.0 + control.PHASE_RATE_LIMIT
#: A rung whose target speed is below this takes the wider duty band (owner decision 9).
SLOW_WALK_M_S = 0.30
DUTY_SLOW = (0.55, 0.72)
#: Rows: the judged delay, the largest delay a row name may carry, the physics step in ms.
JUDGED_MS = 10
DELAY_MAX_MS = 20
STEP_MS = 2
#: Controllers whose rows give no verdict and are not replicated.
NO_VERDICT_CONTROLLERS = ("R0",)
#: A task's budget in core-hours (the sum of its episodes' wall seconds / 3600).
BUDGET_CORE_H = {3: 16.0, 4: 12.0}
#: Reported controls: name -> (the crouch the track's LIPM frequency is computed with, the record's label).
CONTROLS = {"omega_s1": (0.03, "omega at Stage 1's 0.03 H crouch")}
#: Dry-run rates: Stage 1's own committed condition they are measured on, the cut of each episode and its seeds.
COST_CONDITION = "walk/brisk"
COST_SECONDS = 8.0
COST_SEEDS = (3042, 3043)
COST_SCHEMA = "bar-cost-1"
EPS_S = readings.EPS_S

# ------------------------------------------------------------------------------------------------
# Rules (each record carries the ones it was made by)
# ------------------------------------------------------------------------------------------------
RULES = dict(
    clocks=(
        "three clocks: the reference clock (track time); simulated time; the bar's clock, simulated seconds since "
        "the command instant t_c = the simulated time at which the reference clock reaches the commanded move's "
        "first frame (a causal move's file begins at its command).  The track is played at the commanded rate "
        f"{RATE:g}, times the tracker's phase-rate factor (within +-{100 * control.PHASE_RATE_LIMIT:g} %).  Every "
        "judged time (a turn's (pi/2)/|rate| s, the 3 s hold, the walk's 10 s) is in simulated seconds.  A nudge's "
        "command is its onset.  stop/one_step plays Stage 1's non-causal one-step stop and is read from Stage 1's "
        "command (decision 18's convention: where the reference's path speed begins to ramp down), labelled so"),
    readings=(
        "every episode is read three ways, in the plan's order and never pooled: (1) shown with the look: not "
        "defined for a set without look bands (the block is null with that reason); (2) every limit (success); (3) "
        "every limit but for A1 (success_but_A1): the same terms with the sustained-drive term read on the leg "
        "drives only (plant.leg_drives: every drive but the neck's and the tail's).  The drives' RMS is taken over "
        "the judged window for the walk and the turn and over the whole episode for nudges and the stop; both "
        "spans' per-drive ratios are recorded.  A turn frozen with a1_over runs and records both readings, but "
        "reading 2 gives it no verdict (not read: a1_over, which is not a pass)"),
    toe=(
        "toe clearance over each foot's own swing (readings.toe_clearance_own_swing): contact is Stage 1's (a foot's "
        "load above 2 % of weight, 20 ms debounce, per physics step); an air run of a foot is a swing if any 100 Hz "
        "update inside it falls in the reference's swing of that foot, or if it lasts at least half the shortest "
        f"planned swing of the track on the simulated clock (its reference seconds / {RATE:g}), else an air gap "
        "(counted, not read); each swing wholly inside the window is read as the foot's lowest point over the middle "
        "half of its own duration (interpolated at 25 % and 75 %, and every update between); the term passes when "
        f"the smallest reading is at least {readings.TOE_CLEARANCE_FRAC_H:g} H, no reference swing in the window is "
        "interrupted and each foot has at least its number of reference swings less one.  Stage 1's reading (on the "
        "reference's clock) is recorded beside it"),
    walk=(
        "the walk item: window = the last 10 s of a complete episode; along: the CoM covers at least 10 s x the "
        "rung's target speed along the commanded heading (the home heading); line: the CoM stays within "
        f"{readings.LINE_MAX_M:g} m of the line along the commanded heading through its position at the window's "
        "first update (the deviation from the line through the episode's start is reported); Stage 1's limits over "
        f"the window: duty {tune.DUTY[0]:g}-{tune.DUTY[1]:g} ({DUTY_SLOW[0]:g}-{DUTY_SLOW[1]:g} for a rung whose "
        f"target speed is below {SLOW_WALK_M_S:g} m/s), no flight, roll and pitch {tune.ROLL_PITCH_RMS_DEG:g} deg "
        f"RMS and {tune.ROLL_PITCH_PEAK_DEG:g} deg peak, the own-swing toe clearance, slip "
        f"{1e3 * tune.SLIP_MEDIAN_M:g} mm median and {1e3 * tune.SLIP_WORST_M:g} mm worst, the drives' RMS over the "
        "window at most their sustained rating, and over the episode the drives' peak speed and "
        f"{tune.WALK_RANGE_RAD:g} rad of every joint's range.  Peak ground reaction, cadence and stride are "
        "reported, and judged as Stage 1 judges them in walk/nominal only.  Stage 1's own reading of the same "
        "episode (tune.measure's terms restricted to the panels' list for that walk) is recorded under "
        "measures.stage1: it gives no verdict"),
    turn=(
        "the turn item: heading: the trunk's heading is within "
        f"{readings.HEADING_TOL_DEG:g} deg of the commanded heading at the command + 90 deg at t_c + at_s, at_s = "
        f"(pi/2) / |rate|, and at every update for {readings.HEADING_HOLD_S:g} s after; over [t_c, t_c + at_s + "
        f"{readings.HEADING_HOLD_S:g} s]: the walk's limits (without along and line), the joint range by decision 14 "
        f"(min({tune.WALK_RANGE_RAD:g}, the track's own smallest margin of that joint - "
        f"{tune.TURN_RANGE_SLACK_RAD:g}), at least {tune.AT_LIMIT_RAD:g} rad) and no leg interference (legs.LegGap "
        "at 50 Hz).  Peak ground reaction is reported"),
    nudge=(
        f"a nudge: a {PUSH_S:g} s pulse at the trunk's centre of mass of m x frac x the Stage 0 push bound / "
        f"{PUSH_S:g} s in the commanded heading's frame, frac {NUDGE_FRAC:g} (reported: "
        f"{', '.join(f'{x:g}' for x in NUDGE_FRACS_REPORTED)}); onset (t_steady + ({NUDGE_AT_STRIDES:g} + u) T) / "
        f"{RATE:g}, t_steady the track time at which the walk's segment begins, T the walk's stride, u uniform in "
        "[0, 1) from the episode's push seed; no fall, back on the command within "
        f"{RECOVERY_S:g} s of the pulse's end and held to the planned end (panels.RECOVERY_RULE, decision 30), the "
        "drive and range limits over the episode"),
    stop=(
        "the stop item: t_stop = the first update at or after the command from which the magnitude of the CoM's "
        f"forward speed stays under {100 * readings.STOP_SPEED_FRAC:g} % of the commanded speed to the end of the "
        "trace; touchdowns_after_command: at most "
        f"{readings.STOP_TOUCHDOWNS_MAX} touchdowns (a foot's contact beginning) with t_c < t <= t_stop; "
        "stopped_by_end; stop_acceleration_1hz: the IMU point's forward acceleration low-passed at 1 Hz at most "
        f"{readings.ACCEL_MAX_M_S2:g} m/s^2 over two commanded strides from the command (a zero-phase 1 Hz filter "
        "turns a speed change dv into at most 2.22 dv m/s^2: the term sees rocking, not abruptness; the unfiltered "
        "deceleration peak is recorded); the drive and range limits over the episode.  Decision 18's reading "
        "(control.run's) is recorded beside"),
    plans=(
        "every item starts with the walk item's lead-in (start, one stride of nominal, the change to the walk, or "
        "the path through the neighbouring rung toward 0.40 m/s where the direct change is not available) and takes "
        "the walk's seed slot (slot 0), so the items of a seed are not independent: a fall in the lead-in shows in "
        "all of them (this differs from Stage 1, which gives turns their own slots).  Walk: K strides of the walk, "
        f"the fewest that hold the last 10 s in its steady part with the clock {100 * (1 - SLOW):g} % slow (Stage 1's "
        f"rule).  Turn: {TURN_LEAD_STRIDES['left']:g} strides ({TURN_LEAD_STRIDES['right']:g} for a right turn), the "
        "turn, then the walk; seconds = t_cmd / (slowest clock) + at_s + "
        f"{readings.HEADING_HOLD_S:g} + {END_MARGIN_S:g}.  Nudge: seconds = the latest onset at the slowest clock + "
        f"{PUSH_S:g} + {RECOVERY_S:g} + {HOLD_S:g}.  Stop: {STOP_LEAD_STRIDES[2]:g} strides "
        f"({STOP_LEAD_STRIDES[1]:g} before the one-step stop), the stop, a hold; seconds = t_rest / (slowest clock) "
        f"+ {STOP_HOLD_S:g} + {END_MARGIN_S:g}.  Every track is long enough at the fastest clock ({FAST:g} x "
        f"{RATE:g}).  walk/nominal is start, then nominal to the end"),
    rows=(
        f"R2/<s>@<ms>ms: Stage 1's fitted tracker of CMA-ES seed s with the gains of the Stage 1 run's tune.json "
        f"unchanged; R0@<ms>ms: the panels' open-loop row; <any registered row>@<ms>ms likewise.  <ms> is a servo "
        f"delay of k = ms / {STEP_MS} whole physics steps for the whole episode (speedfit.servo_delay: every servo "
        "target reaches the servos k steps after the controller sets it); with k = 0 the wrapper changes nothing"),
    seeds=(
        f"blocks: {HEADLINE_BLOCK} ({BLOCKS[HEADLINE_BLOCK][0]}-{BLOCKS[HEADLINE_BLOCK][-1]}, the headline) and "
        f"{REPLICATION_BLOCK} ({BLOCKS[REPLICATION_BLOCK][0]}-{BLOCKS[REPLICATION_BLOCK][-1]}, replication); an "
        "episode's start, sensor-noise and push seeds are tune.derived_seed(seed, 0, purpose) (panels.EPISODE_RULE "
        "with slot 0 for every item)"),
    pass_=(
        f"a record passes a reading if all {RUNS} episodes of its block survive and at least {MIN_SUCCEED} succeed "
        "in that reading; a record of fewer episodes, of other seeds or with an episode that failed to run is not "
        "judged.  The judged delay is 10 ms; at another delay a record 'passes at <ms> ms' and gives no verdict.  "
        "R0 rows, Stage 1's reading, the replication block alone and a control give no verdict"),
    replication=(
        f"every {HEADLINE_BLOCK} record of a row with a verdict that passes in reading 2 or in reading 3 is run "
        f"again on {REPLICATION_BLOCK} with the same row; the new record names the record and the reading that "
        "triggered it (replication_of); the blocks are never pooled; a pass of an R0 row is not replicated"),
    omega=(
        "the track's LIPM frequency is the set's own: control.omega0(p, q_home, crouch_frac = the set's crouch); "
        "the controllers' gains are not refitted; omega and crouch_frac are in every record and in its key"),
    error=(
        "an episode that raised, or that did not fall and whose trace ends before its judged window, failed to "
        "run: it is recorded with its reason, the same seed is run again on the next run of the same key, and a "
        "record that still holds such an episode is not judged; it is never a failed term"),
    key=(
        "a record's key holds what an episode executes: the row's description and sources, the condition, the "
        "plan, seconds, seeds and command, omega and crouch_frac, panels.key_context(body), per input file its "
        "sha256 and the sha256 of the entry numbers the planner reads (setio.entry_sha256), and the source hashes "
        "of the panels' modules and of bar.py, readings.py, setio.py and speedfit.py"),
    no_overwrite=(
        "a finished record whose key differs is never replaced silently: the run stops with the differing key "
        "fields; --supersede moves the old record to <record>.superseded/<first 12 hex of its key's sha256>.json "
        "and the new record states its reason (supersedes.reason): it is a re-run after a result"),
    trace=(
        "each episode's trace (not the attribution's lifted runs): Stage 1's 50 Hz arrays (panels.TRACE_RULE) plus "
        "the 500 Hz contact flags (bit-packed), both feet's 100 Hz low points with their times and the reference's "
        "swing foot at those updates, the per-swing table and the command instant; the episode record holds its "
        "path and sha256"),
    attribution=(
        "decision 20 (panels.PANEL_ATTRIBUTION_RULE, gates.attribute) on R2 rows' records with "
        f"{panels.ATTRIBUTION_MIN_FALLS} or more falls: each candidate limit lifted over the same seeds.  A turn, "
        f"nudge or stop record with fewer than {panels.ATTRIBUTION_MIN_FALLS} falls after its command is not "
        "attributed itself: it cites the walk record of the same set, row, grip and block.  A record whose "
        "attribution the budget cut reads 'not attributed (budget, pre-registered order)'"),
    budget=(
        "a task's budget is the sum of its episodes' wall seconds (attribution's lifted episodes and re-runs "
        "included) in budget.json.  It is spent unit by unit (one run command; one record's attribution): a unit "
        "starts only if the task's spent core-hours plus the unit's estimate (its planned simulated seconds x the "
        "committed rates) fit; a started unit finishes; the first unit that does not fit is cut with every unit "
        "after it"),
    checks=(
        "on the three sets of the 75 cm round a run is either a task's unit (--task) or a build check: at most one "
        f"episode (seed {BLOCKS[HEADLINE_BLOCK][0]}) per condition and R0 row on set {CHECK_SET}, each listed with its "
        "reason in the file --dev-episodes names; a check is smoke and no record"),
    stock=(
        "a Stage 1 condition on the pseudo-set S1 (a Stage 1 run directory) is built by panels.Context.built and "
        "judged by panels.judge unchanged, with Stage 1's terms, trace and omega (0.03 H crouch); its record has "
        "definition.stock = true and one reading"),
    convention=(
        "the recorder's convention: an update row's time is the start of the physics step just taken and a step's "
        "loads are read after it, so physics step a and a row with t = a dt are one instant "
        "(readings.STEP_OFFSET); convention_check reads it on a real episode"),
)
COMMON_RULE_KEYS = ("clocks", "readings", "plans", "rows", "seeds", "pass_", "replication", "omega", "error", "key",
                    "no_overwrite", "trace", "attribution", "budget", "checks", "convention")
JUDGE_RULE_KEYS = {"bar_walk": ("walk", "toe"), "bar_turn": ("turn", "walk", "toe"), "bar_nudge": ("nudge",),
                   "bar_stop": ("stop",)}
#: Stage 1's rules a bar record relies on (copied into it under ``rules.stage1``).
STAGE1_RULE_KEYS = ("rows", "episode", "failed_episode", "limits", "lcb", "success_label", "attribution",
                    "decision_20", "candidates", "failure_label", "trace", "command_rate", "failures", "recovery",
                    "success", "stop", "sources")
LOOK_NOT_DEFINED = "not defined: the set has no look bands"
BUDGET_CUT_LABEL = "not attributed (budget, pre-registered order)"
BUDGET_CUT_REPORT = "not attributed: cut by the pre-registered order"


class BarError(s2io.S2Error):
    """The bar refuses to run (a record of another key, a set that is not frozen, a check outside its rule)."""


class KeyMismatch(BarError):
    """A status-done record holds another key (``RULES["no_overwrite"]``)."""


class Unreadable(RuntimeError):
    """An episode that did not fall and cannot be read (``RULES["error"]``): the worker records it as failed to run."""


def rules_of(cond, set_docs: dict | None = None) -> dict:
    """The rules a condition's record is made by: the bar's common ones and its judge's, Stage 1's rules they rely
    on, and for a bar condition the set's own (``set.json`` ``rules`` and the walk entry's ``margin_rule``)."""
    if not isinstance(cond, BarCondition):
        return dict({k: RULES[k] for k in ("stock", "rows", "seeds", "pass_", "error", "key", "no_overwrite")},
                    stage1=panels.rules_of(cond))
    out = {k: RULES[k] for k in COMMON_RULE_KEYS + JUDGE_RULE_KEYS[cond.judge]}
    out["stage1"] = {k: panels.RULES[k] for k in STAGE1_RULE_KEYS}
    out["delay"] = speedfit.LATENCY_RULE
    if set_docs is not None:
        walk = (set_docs["refs.json"]["references"].get(cond.walk) or {})
        out["set"] = dict(rules=set_docs["set.json"].get("rules"), margin_rule=walk.get("margin_rule"),
                          label=walk.get("label"), omega_rule=set_docs["set.json"].get("omega_rule"))
    return out


# ------------------------------------------------------------------------------------------------
# Conditions
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BarCondition(panels.Condition):
    """One condition of the bar on a set: ``panels.Condition``'s fields (``judge`` one of ``BAR_JUDGES``; ``gait``
    is the walk's name; ``case`` a turn's rate label; ``push_at`` and ``push_spread`` a nudge's onset in strides)
    plus the set, the walk every item leaves from, the commanded move's name, a nudge's share of the push bound
    and the seed block."""

    set: str = ""
    walk: str = BAR_WALK
    move: str | None = None
    frac: float = NUDGE_FRAC
    block: str = HEADLINE_BLOCK

    def describe(self) -> dict:
        return dict(dataclasses.asdict(self), key=self.key, stock=False)


def is_bar(cond) -> bool:
    return isinstance(cond, BarCondition)


def describe_condition(cond) -> dict:
    """``cond.describe()``; a stock condition's gets ``stock: true``."""
    return cond.describe() if is_bar(cond) else dict(cond.describe(), stock=True)


def conditions(set_docs: dict, block: str = HEADLINE_BLOCK, every: bool = False) -> list[BarCondition]:
    """The bar's conditions on a set (``setio.load``'s documents): the 16 gated ones, always, and the reported ones
    the set has entries for (an entry recorded ``available: false`` counts: its condition gets a "not run"
    record).  With ``every`` the reported ones are all there, whatever the set holds (a condition named in a run
    whose reference the set never listed gets a "not run" record too)."""
    if block not in BLOCKS:
        raise ValueError(f"block {block!r}: choose from {', '.join(BLOCKS)}")
    name = set_docs["set.json"]["set"]
    refs = set_docs["refs.json"]["references"]
    trs = set_docs["transitions.json"]["transitions"]
    out: list[BarCondition] = []

    def add(behavior, cname, gated, judge, **kw):
        kw.setdefault("gait", kw.get("walk", BAR_WALK))
        out.append(BarCondition(behavior, cname, gated, judge, set=name, block=block, **kw))

    def nudge(d, tag, gated, mu, frac):
        add("nudge", f"{d}{tag}", gated, "bar_nudge", friction=mu, direction=d, frac=frac, push_at=NUDGE_AT_STRIDES,
            push_spread=NUDGE_SPREAD_STRIDES, recovery_s=RECOVERY_S, hold_s=HOLD_S)

    for mu, tag in GRIPS:
        add("walk", BAR_WALK + tag, True, "bar_walk", friction=mu)
    for d in setio.DIRECTIONS:
        for mu, tag in GRIPS:
            add("turn", f"q90_{d}{tag}", True, "bar_turn", friction=mu, case=setio.QUICK, direction=d,
                move=setio.turn_name(setio.QUICK, d))
    for d in panels.PUSH_DIRECTIONS:
        for mu, tag in GRIPS:
            nudge(d, tag, True, mu, NUDGE_FRAC)
    for mu, tag in GRIPS:
        add("stop", f"two_step{tag}", True, "bar_stop", friction=mu, move=setio.stop_name(2, BAR_WALK))
    # reported: the speed ladder, the turn-rate ladder, the larger nudges, Stage 1's one-step stop
    for w in [setio.NOMINAL] + [setio.walk_name(v) for v in setio.SPEEDS if v != setio.BAR_SPEED]:
        if every or w in refs:
            for mu, tag in GRIPS:
                add("walk", w + tag, False, "bar_walk", friction=mu, walk=w)
    for rate in setio.TURN_RATES:
        for d in setio.DIRECTIONS:
            if rate != setio.QUICK and (every or setio.turn_name(rate, d) in trs):
                add("turn", f"t90_{rate}_{d}", False, "bar_turn", case=rate, direction=d, move=setio.turn_name(rate, d))
    for frac in NUDGE_FRACS_REPORTED:
        for d in panels.PUSH_DIRECTIONS:
            nudge(d, f"_x{frac:g}", False, None, frac)
    if every or setio.stop_name(1, BAR_WALK) in trs:
        add("stop", "one_step", False, "bar_stop", move=setio.stop_name(1, BAR_WALK))
    return out


#: Groups of conditions a run may name (``choose``): each takes the conditions the set has entries for.
GROUPS = {
    "@gated": lambda c: c.gated,
    "@walk": lambda c: c.gated and c.judge == "bar_walk",  # the walk item on both grips (task 3)
    "@rest": lambda c: c.gated and c.judge != "bar_walk",  # the 14 turn, nudge and stop conditions (task 4)
    "@speed-ladder": lambda c: not c.gated and c.judge == "bar_walk" and c.friction is None,
    "@ladders": lambda c: not c.gated and c.judge != "bar_walk",  # turn rates, larger nudges, the one-step stop
}


def choose(set_docs: dict, block: str, names=None) -> list[BarCondition]:
    """The conditions a run takes: without names the 16 gated ones; a name (key or bare name) of any condition the
    bar defines, whether or not the set lists its reference; a group of ``GROUPS`` (the conditions of that group
    the set has entries for).  In the order of ``conditions``; ``ValueError`` for an unknown name."""
    have = conditions(set_docs, block)
    if names is None:
        return [c for c in have if c.gated]
    known = conditions(set_docs, block, every=True)
    picked, unknown = set(), []
    for n in names:
        if n in GROUPS:
            picked |= {c.key for c in have if GROUPS[n](c)}
        elif any(n in (c.key, c.name) for c in known):
            picked |= {c.key for c in known if n in (c.key, c.name)}
        else:
            unknown.append(n)
    if unknown:
        raise ValueError(f"conditions {unknown} are not among {sorted(c.key for c in known)} or the groups "
                         f"{', '.join(GROUPS)}")
    return [c for c in known if c.key in picked]


def stock_conditions() -> list[panels.Condition]:
    """Stage 1's conditions the pseudo-set ``S1`` takes: every one of ``panels.conditions()``."""
    return panels.conditions()


def stock_default() -> list[panels.Condition]:
    """Stage 1's gated hard-floor conditions (its "no loss" conditions: stand, four walks, four turns, start and
    stop, and the standing and the walking pushes, a record per direction)."""
    return [c for c in panels.conditions() if c.gated and c.floor == "hard"]


def select(conds: list, names=None, default=None) -> list:
    """The conditions whose key ("walk/v0.4") or name is in ``names``, in ``conds``' order; without names
    ``default`` (None: all)."""
    if names is None:
        return list(conds if default is None else default)
    names = list(names)
    known = {c.key for c in conds} | {c.name for c in conds}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"conditions {unknown} are not among {sorted(c.key for c in conds)}")
    return [c for c in conds if c.key in names or c.name in names]


# ------------------------------------------------------------------------------------------------
# Plans
# ------------------------------------------------------------------------------------------------
def stage1_stop_event(tr: Transition, t0: float) -> dict:
    """Stage 1's stop command of a non-causal stop transition whose segment begins at ``t0`` on a track
    (``control.Track.add_transition``'s lines for the names it registers): where the path speed starts to ramp
    down."""
    n0 = min(N_FRAMES, len(tr.t) - 1)
    duty = float(np.mean(tr.support[:n0, 0]))
    ph = stop_command_phase(duty)
    speed = float(np.linalg.norm(tr.path[1] - tr.path[0]) / (tr.t[1] - tr.t[0]))
    return dict(kind="stop", name=tr.name, t_ref=float(t0) + float(np.interp(ph, tr.phase, tr.t)), command_phase=ph,
                speed_m_s=speed, period_s=float(tr.t[n0] - tr.t[0]))


class BarPlanner:
    """The bar's plans and planned lengths on one set, without a plant (``RULES["plans"]``): ``resolve(cond)`` gives
    ``dict(plan, seconds, command_ref_s, steady_ref_s, window, push, files, at_s, command, rest_ref_s)``.  Segment
    times follow ``control.build_track``; a periodic item lasts its strides times the entry's ``window_s`` (its
    ``period_s`` where the entry has none)."""

    def __init__(self, set_dir, refs_doc: dict, tr_doc: dict, set_doc: dict | None = None):
        self.set_dir, self.refs_doc, self.tr_doc = Path(set_dir), refs_doc, tr_doc
        self.set_doc = set_doc or {}
        self._files = panels.Planner(self.set_dir, refs_doc, tr_doc)  # loads the arrays a plan's times need

    # ---------------------------------------------------------------- what the set has
    def why_not(self, sub: str, name: str) -> dict | None:
        """None if the set has a usable entry ``name`` (``sub``: "refs" or "transitions"), else ``{reference,
        reason, numbers}``."""
        e = (self.refs_doc if sub == "refs" else self.tr_doc).get(name)
        if e is not None and e.get("available") is not False and e.get("file"):
            return None
        noted = (self.set_doc.get("not_available") or {}).get(name) or {}
        if e is None:
            return dict(reference=f"{sub}/{name}", reason=str(noted.get("reason") or f"the set has no entry {name!r}"),
                        numbers=noted.get("numbers"))
        return dict(reference=f"{sub}/{name}",
                    reason=str(e.get("reason") or noted.get("reason") or "recorded as not available"),
                    numbers=e.get("numbers", noted.get("numbers")))

    def usable(self, sub: str, name: str) -> bool:
        return self.why_not(sub, name) is None

    def lead_in(self, walk: str) -> list | None:
        """The plan items before the steady walk ``walk``: ``start`` for the nominal walk; else start, one stride
        of nominal and the direct change; else the neighbouring rung's lead-in, one stride of that rung and the
        change from it (the path through 0.40 m/s); None if the set has neither."""
        if walk == setio.NOMINAL:
            return [("transition", setio.START)]
        direct = setio.change_name(setio.NOMINAL, walk)
        if self.usable("transitions", direct):
            return [("transition", setio.START), ("ref", setio.NOMINAL, 1), ("transition", direct)]
        v = setio.parse_name(walk)["speed_m_s"]
        if walk == BAR_WALK or v not in setio.SPEEDS:
            return None
        i = setio.SPEEDS.index(v)
        nb = setio.walk_name(setio.SPEEDS[i + (1 if v < setio.BAR_SPEED else -1)])
        step = setio.change_name(nb, walk)
        head = self.lead_in(nb)
        if head is None or not self.usable("transitions", step) or not self.usable("refs", nb):
            return None
        return head + [("ref", nb, 1), ("transition", step)]

    def missing(self, cond: BarCondition) -> dict | None:
        """Why ``cond`` cannot be run on this set (``{reference, reason, numbers}``), or None."""
        need = [("refs", setio.NOMINAL), ("transitions", setio.START), ("refs", cond.walk)]
        if cond.move:
            need.append(("transitions", cond.move))
        for sub, name in need:
            why = self.why_not(sub, name)
            if why is not None:
                return why
        if self.lead_in(cond.walk) is None:
            direct = setio.change_name(setio.NOMINAL, cond.walk)
            why = self.why_not("transitions", direct) or {}
            return dict(reference=f"transitions/{direct}",
                        reason=f"no transition into {cond.walk} ({why.get('reason') or 'none in the set'})",
                        numbers=why.get("numbers"))
        return None

    # ---------------------------------------------------------------- times
    def period_s(self, name: str) -> float:
        """One stride of the walk ``name`` (reference seconds)."""
        return float(self.refs_doc[name]["period_s"])

    def window_s(self, name: str) -> float:
        """What one unit of a ``("ref", name, n)`` item lasts: the entry's window."""
        e = self.refs_doc[name]
        return float(e["window_s"]) if e.get("window_s") is not None else float(e["period_s"])

    def speed(self, name: str) -> float:
        """The walk's target speed: the speed its name holds, or the nominal walk's own."""
        v = setio.parse_name(name).get("speed_m_s")
        return float(self.refs_doc[name]["speed_m_s"]) if v is None else float(v)

    def segment_times(self, plan: list) -> list[tuple[float, float]]:
        """(t0, duration) of each item on the reference clock, as ``control.build_track`` lays them out."""
        objs = [None if item[0] != "transition" else self._files.load(item) for item in plan]
        out, t = [], 0.0
        for i, (item, obj) in enumerate(zip(plan, objs, strict=True)):
            if item[0] == "ref":
                dur = item[2] * self.window_s(item[1])
            elif item[0] == "transition":
                nxt = plan[i + 1] if i + 1 < len(plan) else None
                if nxt is not None and nxt[0] == "ref":
                    dt_last = float(self.refs_doc[nxt[1]]["dt_s"])
                elif nxt is not None and nxt[0] == "transition":
                    dt_last = float(objs[i + 1].t[1] - objs[i + 1].t[0])
                else:
                    dt_last = float(obj.t[-1] - obj.t[-2])
                dur = float(obj.t[-1] + dt_last)
            else:
                dur = float(item[1])
            out.append((t, dur))
            t = t + dur
        return out

    def phase_time(self, plan: list, i: int, phase: float) -> float:
        tr = self._files.load(plan[i])
        return float(self.segment_times(plan)[i][0] + np.interp(phase, tr.phase, tr.t))

    @staticmethod
    def files_of(plan: list) -> list[str]:
        """The files a plan loads, with ``refs/nominal.npz`` (the home pose of ``control.omega0``)."""
        out = {f"refs/{setio.NOMINAL}.npz"}
        for item in plan:
            if item[0] == "ref":
                out.add(f"refs/{item[1]}.npz")
            elif item[0] == "transition":
                out.add(f"transitions/{item[1]}.npz")
        return sorted(out)

    # ---------------------------------------------------------------- the plan table
    def resolve(self, cond: BarCondition) -> dict:
        """The condition's plan and planned seconds (simulated) and what its judge needs.  A condition that gives
        ``seconds`` is cut there (a smoke check: its episodes cannot be judged)."""
        why = self.missing(cond)
        if why is not None:
            raise tune.InputError(f"{cond.key}: {why['reference']}: {why['reason']}")
        out = self._resolve(cond)
        out["files"] = self.files_of(out["plan"])
        if cond.seconds is not None and cond.seconds < out["seconds"]:
            out = dict(out, seconds=float(cond.seconds), cut_from_s=out["seconds"])
        return out

    def _resolve(self, cond: BarCondition) -> dict:
        j, W = cond.judge, cond.walk
        lead = self.lead_in(W)
        P, T = self.window_s(W), self.period_s(W)
        out = dict(command_ref_s=None, steady_ref_s=None, window=None, push=None, at_s=None, command=None,
                   rest_ref_s=None, target_speed_m_s=self.speed(W), period_s=T)
        k = len(lead)
        probe = lead + [("ref", W, 1)]
        t_walk = self.segment_times(probe)[k][0]  # the track time at which the walk's segment begins
        if j == "bar_walk":
            steady = self.phase_time(probe, k - 1, 1.0)
            need = (steady + panels.WINDOW_MARGIN_STRIDES * P) / SLOW + tune.WALK_WINDOW_S * RATE
            K = max(1, math.ceil((need - t_walk) / P))
            return dict(out, plan=lead + [("ref", W, K)], seconds=(t_walk + K * P) / RATE, steady_ref_s=steady,
                        window=dict(kind="the last seconds of a complete episode", seconds=tune.WALK_WINDOW_S))
        if j == "bar_nudge":
            latest = (t_walk + (cond.push_at + cond.push_spread) * T) / (SLOW * RATE)
            seconds = latest + PUSH_S + cond.recovery_s + cond.hold_s
            K = max(1, math.ceil((seconds * FAST * RATE - t_walk) / P))
            return dict(out, plan=lead + [("ref", W, K)], seconds=seconds, steady_ref_s=t_walk,
                        push=dict(kind="walking", steady_ref_s=t_walk, at_strides=float(cond.push_at),
                                  spread_strides=float(cond.push_spread), period_s=T, frac=float(cond.frac)),
                        window=dict(kind="the whole episode"))
        if j == "bar_turn":
            name = setio.parse_name(cond.move)
            head = lead + [("ref", W, TURN_LEAD_STRIDES[name["direction"]]), ("transition", cond.move)]
            times = self.segment_times(head + [("ref", W, 1)])
            t_cmd, t_after = times[len(head) - 1][0], times[len(head)][0]
            at_s = name["angle_rad"] / abs(name["rate_rad_s"])
            seconds = t_cmd / (SLOW * RATE) + at_s + readings.HEADING_HOLD_S + END_MARGIN_S
            K = max(1, math.ceil((seconds * FAST * RATE - t_after) / P))
            phase = ((self.tr_doc[cond.move].get("move") or {}).get("command_phase"))
            return dict(out, plan=head + [("ref", W, K)], seconds=seconds, command_ref_s=t_cmd, at_s=at_s,
                        steady_ref_s=t_walk, sign=name["sign"], angle_rad=name["angle_rad"],
                        command=dict(name=cond.move, ref_s=t_cmd, phase=phase, convention="causal: the move's first "
                                                                                          "frame"),
                        window=dict(kind="from the command", seconds=at_s + readings.HEADING_HOLD_S))
        if j == "bar_stop":
            name = setio.parse_name(cond.move)
            head = lead + [("ref", W, STOP_LEAD_STRIDES[name["steps"]]), ("transition", cond.move)]
            t_seg, dur = self.segment_times(head)[-1]
            t_rest = t_seg + dur
            move = self.tr_doc[cond.move].get("move") or {}
            if move.get("causal"):
                command = dict(name=cond.move, ref_s=t_seg, phase=move.get("command_phase"),
                               convention="causal: the move's first frame")
            else:
                ev = stage1_stop_event(self._files.load(head[-1]), t_seg)
                command = dict(name=cond.move, ref_s=ev["t_ref"], phase=ev["command_phase"],
                               convention="Stage 1's (decision 18): where the reference's path speed begins to ramp "
                                          "down; the reference anticipates this command")
            seconds = t_rest / (SLOW * RATE) + STOP_HOLD_S + END_MARGIN_S
            hold = seconds * FAST * RATE - t_rest
            return dict(out, plan=head + [("hold", hold)], seconds=seconds, command_ref_s=command["ref_s"],
                        steady_ref_s=t_walk, rest_ref_s=t_rest, command=command,
                        window=dict(kind="the whole episode", hold_after_rest_s=STOP_HOLD_S))
        raise ValueError(f"judge {j!r}")


# ------------------------------------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------------------------------------
_ROW_NAME = re.compile(r"^(?P<inner>.+)@(?P<ms>\d+)ms$")


def ms_steps(ms: int, name: str = "") -> int:
    """Whole physics steps of a row's delay: even milliseconds, 0 to ``DELAY_MAX_MS``."""
    if isinstance(ms, bool) or int(ms) != ms or ms % STEP_MS or not 0 <= ms <= DELAY_MAX_MS:
        raise ValueError(f"row {name!r}: a delay of whole {STEP_MS} ms physics steps, 0-{DELAY_MAX_MS} ms, not {ms!r}")
    return int(ms) // STEP_MS


def parse_row(name: str) -> tuple[str, int]:
    """(inner row's name, delay steps) of ``<row>@<ms>ms``; ``ValueError`` for another name or delay."""
    m = _ROW_NAME.match(name if isinstance(name, str) else "")
    if not m:
        raise ValueError(f"row {name!r}: <row>@<ms>ms (R2/42@10ms, R0@10ms)")
    return m.group("inner"), ms_steps(int(m.group("ms")), name)


def delay_ms(row_name: str) -> int | None:
    """The delay a row's name carries (None: it carries none)."""
    m = _ROW_NAME.match(row_name)
    return int(m.group("ms")) if m else None


def controller_of(row_name: str) -> str:
    """A row's controller: its name before the seed and the delay ("R2/42@10ms" -> "R2", "R0@10ms" -> "R0")."""
    m = _ROW_NAME.match(row_name)
    return panels.controller_of(m.group("inner") if m else row_name)


class Delayed(panels.Row):
    """Any ``panels.Row`` with a servo delay of ``k`` physics steps for the whole episode (``RULES["rows"]``): the
    inner row's ``episode`` inside ``speedfit.servo_delay`` on the condition's plant.  With ``k = 0`` it is the
    inner row bit for bit.  It runs every condition it is given; a condition is gated for it as the condition
    says."""

    def __init__(self, inner: panels.Row, k: int, name: str | None = None):
        if isinstance(k, bool) or int(k) != k or not 0 <= k <= DELAY_MAX_MS // STEP_MS:
            raise ValueError(f"servo delay {k!r}: whole physics steps, 0-{DELAY_MAX_MS // STEP_MS}")
        self.inner, self.k = inner, int(k)
        self.name = name or f"{inner.name}@{STEP_MS * self.k:d}ms"
        self.hard_only = False

    def runs(self, cond) -> bool:
        return True

    def gated(self, cond) -> bool:
        return bool(cond.gated)

    def describe(self) -> dict:
        return dict(self.inner.describe(), delayed_row=self.name, latency_steps=self.k, latency_ms=STEP_MS * self.k,
                    latency_rule=RULES["rows"], servo_delay_rule=speedfit.LATENCY_RULE)

    def sources(self) -> dict:
        """The inner row's sources with this module and ``speedfit.py`` (the delay model) by sha256."""
        return {**self.inner.sources(), **panels.module_sha256(speedfit), **panels.module_sha256(Delayed)}

    def spec(self) -> dict:
        return dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name, k=self.k,
                    inner=self.inner.spec())

    @classmethod
    def from_spec(cls, s: dict) -> "Delayed":
        return cls(panels.row_from_spec(s["inner"]), s["k"], s["name"])

    def episode(self, ctx, cond, seed: int, relax=()):
        b = ctx.built(cond)
        with speedfit.servo_delay(b.p, self.k):
            return self.inner.episode(ctx, cond, seed, relax)


def rows(body: str, names, stage1_run=STAGE1_RUN) -> dict[str, panels.Row]:
    """The bar's rows by name: ``R2/<s>@<ms>ms`` (gains by ``panels.r2_gains(<stage1 run>/<body>, s)``),
    ``R0@<ms>ms`` (the panels' R0) and ``<any registered row>@<ms>ms``, each a ``Delayed`` around the panels' row;
    even ms, 0 to ``DELAY_MAX_MS``."""
    run_dir = setio._anchored(stage1_run) / body
    out = {}
    for name in names:
        inner, k = parse_row(name)
        out[name] = Delayed(panels.tracker_rows(body, run_dir, (inner,))[inner], k, name)
    return out


# ------------------------------------------------------------------------------------------------
# Built conditions (per worker)
# ------------------------------------------------------------------------------------------------
@dataclass
class BarBuilt(panels.Built):
    """A bar condition built on a worker's plant: ``panels.Built`` (``speed_cmd`` and ``stride_s`` are the walk's on
    the commanded clock) plus the track's LIPM frequency and crouch, the command, the air-gap threshold, the leg
    drives' mask and the walk's target speed."""

    omega: float = 0.0
    crouch_frac: float = 0.03
    command: dict | None = None
    air_gap_max_s: float = 0.0
    leg_mask: np.ndarray | None = None
    target_speed: float = 0.0
    events: list = dataclasses.field(default_factory=list)

    @property
    def legs(self) -> bool:
        return self.cond.judge == "bar_turn"

    def recorder(self) -> metrics.Recorder:
        return metrics.Recorder(legs_every=tune.LEGS_EVERY if self.legs else 0)

    def push_force(self) -> np.ndarray | None:
        """The pulse's force (N, world xy): m x frac x bound / duration along the direction in the home heading."""
        if self.resolved.get("push") is None:
            return None
        F = self.mass * float(self.cond.frac) * self.push_bound_m_s / PUSH_S
        dx, dy = panels.PUSH_DIRECTIONS[self.cond.direction]
        c, s = math.cos(self.home_yaw), math.sin(self.home_yaw)
        return F * np.array([c * dx - s * dy, s * dx + c * dy])


class BarContext(panels.Context):
    """One worker's plants and built conditions on a set.  ``built`` gives a ``BarBuilt`` for a ``BarCondition``
    and ``panels.Context.built``'s value for a stock condition (the pseudo-set ``S1``).  ``crouch_frac`` is the
    crouch of the track's LIPM frequency (``RULES["omega"]``; a control may give Stage 1's)."""

    def __init__(self, body: str, run_dir, docs: dict, set_doc: dict | None = None, crouch_frac: float | None = None):
        super().__init__(body, run_dir, docs)
        self.set_doc = set_doc or {}
        self.crouch_frac = float(self.set_doc.get("crouch_frac", 0.03) if crouch_frac is None else crouch_frac)
        self.bar_planner = BarPlanner(self.run_dir, self.refs_doc, self.tr_doc, self.set_doc)
        self._q_home = None

    @property
    def q_home(self) -> np.ndarray:
        if self._q_home is None:
            self._q_home = Reference.load(self.run_dir / "refs" / f"{setio.NOMINAL}.npz").q_home
        return self._q_home

    def release(self) -> None:
        """Forget every plant, track and built condition (a worker calls it when its set changes)."""
        self._plants, self._built, self._lg, self._q_home = {}, {}, None, None
        self.planner = panels.Planner(self.run_dir, self.refs_doc, self.tr_doc)
        self.bar_planner = BarPlanner(self.run_dir, self.refs_doc, self.tr_doc, self.set_doc)

    def built(self, cond):
        if not is_bar(cond):
            return panels.Context.built(self, cond)
        if cond in self._built:
            return self._built[cond]
        p = self.plant(cond.floor, cond.friction)
        res = self.bar_planner.resolve(cond)
        plan = res["plan"]
        omega = control.omega0(p, self.q_home, crouch_frac=self.crouch_frac)
        spec = tune.build_spec(p, self.run_dir, "walk", None, omega, self.refs_doc, self.tr_doc, RATE, plan=plan,
                               gait=cond.walk)
        track = spec.track
        events = setio.add_events(track, plan, self.tr_doc)
        command = res.get("command")
        if command is not None:
            i = next(k for k, item in enumerate(plan) if item == ("transition", cond.move))
            seg = track.segments[i]
            if not events:  # a non-causal stop: Stage 1's command, registered as Stage 1 registers its own stops
                ev = stage1_stop_event(Transition.load(self.run_dir / "transitions" / f"{cond.move}.npz"), seg.t0)
                track.events.append(ev)
                events = [ev]
            t_ref = float(events[0]["t_ref"])
            if abs(t_ref - command["ref_s"]) > 1e-9:
                raise tune.InputError(f"{cond.key}: the track puts the command at {t_ref!r} s, the plan at "
                                      f"{command['ref_s']!r} s")
            command = dict(command, ref_s=t_ref)
        m = p.m
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in p.joints]
        lo, hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        required = np.full(len(p.joints), tune.WALK_RANGE_RAD)
        if cond.judge == "bar_turn":  # decision 14, from the whole track's own margins
            required = np.maximum(np.minimum(tune.WALK_RANGE_RAD, spec.ref_margins - tune.TURN_RANGE_SLACK_RAD),
                                  tune.AT_LIMIT_RAD)
        track.run_dir = self.run_dir  # (for reference_recorder)
        b = BarBuilt(cond=cond, p=p, plan=plan, track=track, spec=spec, seconds=float(res["seconds"]), resolved=res,
                     lo=lo, hi=hi, required=required, H=float(p.meta["H_m"]), mass=float(m.body_subtreemass[p.trunk]),
                     speed_cmd=float(res["target_speed_m_s"]) * RATE, stride_s=float(res["period_s"]) / RATE,
                     push_bound_m_s=float(env.PUSH_BOUND_M_S[self.body]), home_yaw=control._yaw(p._R_home),
                     omega=float(omega), crouch_frac=self.crouch_frac, command=command,
                     air_gap_max_s=readings.air_gap_max(setio.min_planned_swing_s(track), RATE),
                     leg_mask=plant.leg_drives(p.drives), target_speed=float(res["target_speed_m_s"]), events=events)
        self._built[cond] = b
        return b


#: A process's context (one set at a time): ``context_for`` releases the cached plants and tracks when the set, the
#: body, the inputs or the crouch change.
_CONTEXT: dict = {}


def context_for(body: str, run_dir, docs: dict, set_doc: dict | None, crouch_frac: float | None, ident: str) -> BarContext:
    key = (body, str(run_dir), ident, crouch_frac)
    if _CONTEXT.get("key") != key:
        old = _CONTEXT.get("ctx")
        if old is not None:
            old.release()
        _CONTEXT.clear()
        _CONTEXT.update(key=key, ctx=BarContext(body, run_dir, docs, set_doc, crouch_frac))
    return _CONTEXT["ctx"]


def set_crouch(set_doc: dict) -> float:
    """The crouch a set records in ``set.json`` (``crouch_frac``, a share of the hip height): the crouch of every
    spec of the set, with which the track's LIPM frequency is computed.  A set that records none is refused."""
    if set_doc.get("crouch_frac") is None:
        raise BarError(f"set.json of {set_doc.get('set')!r} records no crouch_frac ({RULES['omega']})")
    return float(set_doc["crouch_frac"])


def omega_of(body: str, run_dir, crouch_frac: float) -> float:
    """The track's LIPM frequency of a set and body (``RULES["omega"]``), as ``BarContext.built`` computes it."""
    p = plant.build(body, "hard", settle=False)
    q_home = Reference.load(Path(run_dir) / "refs" / f"{setio.NOMINAL}.npz").q_home
    return float(control.omega0(p, q_home, crouch_frac=crouch_frac))


# ------------------------------------------------------------------------------------------------
# Judgement
# ------------------------------------------------------------------------------------------------
WALK_LIMIT_TERMS = ("duty", "no_flight", "roll_rms", "pitch_rms", "roll_peak", "pitch_peak", "toe_clearance_own_swing",
                    "slip_median", "slip_worst", "drive_sustained", "drive_speed", "joint_range")
WALK_TERMS = ("survived", "along", "line") + WALK_LIMIT_TERMS
NOMINAL_WALK_TERMS = WALK_TERMS + panels.NOMINAL_ONLY_TERMS
TURN_TERMS = ("survived", "heading") + WALK_LIMIT_TERMS + ("no_leg_interference",)
NUDGE_TERMS = ("survived", "recovered") + panels.LIMIT_TERMS
STOP_TERMS = ("survived", "stopped_by_end", "touchdowns_after_command", "stop_acceleration_1hz") + panels.LIMIT_TERMS
A1_TERM, LEGS_TERM = "drive_sustained", "drive_sustained_legs"
HARDWARE_TERMS = panels.HARDWARE_TERMS + (LEGS_TERM,)
DECISION_18_KEYS = ("name", "command_ref_s", "command_s", "command_speed_m_s", "rest_speed_m_s", "reached",
                    "forward_stopped_strides", "within", "at_rest_strides", "at_rest_s")


def terms_of(cond) -> tuple:
    """The success terms of a condition's episodes in reading 2 (each success needs all of them)."""
    if not is_bar(cond):
        return tuple(panels.terms_of(cond))
    j = cond.judge
    if j == "bar_walk":
        return NOMINAL_WALK_TERMS if cond.walk == setio.NOMINAL else WALK_TERMS
    return {"bar_turn": TURN_TERMS, "bar_nudge": NUDGE_TERMS, "bar_stop": STOP_TERMS}[j]


def terms_but_a1(cond) -> tuple | None:
    """The terms of reading 3: reading 2's with the sustained-drive term read on the leg drives only (None for a
    stock condition: Stage 1's judge has one reading)."""
    if not is_bar(cond):
        return None
    return tuple(LEGS_TERM if t == A1_TERM else t for t in terms_of(cond))


def term_class(term: str) -> str | None:
    """"hardware" or "controller" (``panels.SUCCESS_LABEL_RULE``); None for survival."""
    if term == "survived":
        return None
    return "hardware" if term in HARDWARE_TERMS else "controller"


def _window_steps(rec: metrics.Recorder, ta: float, tb: float) -> tuple[int, int]:
    """The physics steps of [ta, tb], as ``metrics.window`` takes them."""
    dt = rec.dt
    return max(0, int(math.ceil(ta / dt - 1e-9))), min(rec.steps, int(math.floor(tb / dt + 1e-9)) + 1)


def _lows(rec: metrics.Recorder) -> np.ndarray:
    return rec.U[:, [metrics.C["low0"], metrics.C["low1"]]]


def _com(rec: metrics.Recorder) -> np.ndarray:
    return rec.U[:, [metrics.C["com_x"], metrics.C["com_y"]]]


def _toe(b: BarBuilt, rec: metrics.Recorder, ta: float, tb: float) -> tuple[dict, list]:
    """Reading (a) over [ta, tb]: (the reading with its term and without the per-swing table, the table)."""
    r = readings.toe_clearance_own_swing(rec.t, _lows(rec), rec.contact(), rec.dt, rec.col("ref_swing"), ta, tb,
                                         b.air_gap_max_s)
    table = r.pop("swings")
    v = r["toe_clearance_own_swing_m"]
    r.update(ok=readings.toe_clearance_own_swing_ok(r, b.H), limit_m=readings.TOE_CLEARANCE_FRAC_H * b.H,
             toe_clearance_own_swing_H=None if v is None else v / b.H, window_s=[float(ta), float(tb)])
    return r, table


def _window_limits(w: dict, cond: BarCondition, target_speed: float) -> dict:
    """Stage 1's window limits on a complete ``metrics.window`` (the walk's and the turn's)."""
    lo, hi = DUTY_SLOW if target_speed < SLOW_WALK_M_S else tune.DUTY
    return dict(duty=bool(all(lo <= x <= hi for x in w["duty"])), no_flight=w["flight_s"] == 0.0,
                roll_rms=w["roll_rms_deg"] <= tune.ROLL_PITCH_RMS_DEG, pitch_rms=w["pitch_rms_deg"] <= tune.ROLL_PITCH_RMS_DEG,
                roll_peak=w["roll_peak_deg"] <= tune.ROLL_PITCH_PEAK_DEG,
                pitch_peak=w["pitch_peak_deg"] <= tune.ROLL_PITCH_PEAK_DEG,
                slip_median=w["slip_median_m"] is not None and w["slip_median_m"] <= tune.SLIP_MEDIAN_M,
                slip_worst=w["slip_worst_m"] is not None and w["slip_worst_m"] <= tune.SLIP_WORST_M,
                grf=w["grf_peak_bw"] <= tune.GRF_BW,
                cadence=w["cadence_steps_s"] is not None and tune.CADENCE[0] <= w["cadence_steps_s"] <= tune.CADENCE[1],
                stride=w["stride_H"] is not None and tune.STRIDE_H[0] <= w["stride_H"] <= tune.STRIDE_H[1])


def _planned_s(b: BarBuilt) -> float:
    """The episode's planned length before any cut."""
    return float(b.resolved.get("cut_from_s", b.seconds))


def _ran_to_end(b: BarBuilt, rec: metrics.Recorder) -> bool:
    """Whether the trace reaches the planned end (to the last feedback update, within two, as ``panels.recovery``
    reads it)."""
    return bool(rec.rows and float(rec.t[-1]) >= _planned_s(b) - 2 * control.FEEDBACK_EVERY * rec.dt)


def stage1_walk_reading(b: BarBuilt, rec: metrics.Recorder, out: dict, window) -> tuple[dict, dict]:
    """Stage 1's reading of a walk episode: ``tune.measure``'s success restricted to the panels' list for that walk
    (``tune.SUCCESS["walk"]``; less ``panels.NOMINAL_ONLY_TERMS`` for a walk other than the nominal one), with
    ``succeeded`` recomputed from that list; and the window's measures (``panels.WINDOW_KEYS``)."""
    res = tune.measure(b.spec, rec, out, None, window=window)
    names = tune.SUCCESS["walk"]
    if b.cond.walk != setio.NOMINAL:
        names = tuple(t for t in names if t not in panels.NOMINAL_ONLY_TERMS)
    fell = bool(out["fell"])
    success = {t: bool(res["success"].get(t, False)) and not (fell and t != "survived") for t in names}
    if fell:
        success["survived"] = False
    w = res.get("window") or {}
    return (dict(rule="tune.measure restricted to the panels' terms of this walk; gives no verdict", terms=list(names),
                 success=success, succeeded=bool(all(success.values())),
                 failed_terms=["survived"] if fell else [t for t, v in success.items() if not v],
                 speed_target_m_s=b.spec.speed_target, window_sustained_ratio=res.get("window_sustained_ratio"),
                 dcm_error_rms_m=res.get("dcm_error_rms_m")),
            {k: w.get(k) for k in panels.WINDOW_KEYS})


def _judge(b: BarBuilt, out: dict, rec: metrics.Recorder, seed: int, lg=None) -> tuple[dict, dict]:
    """The episode record of a bar condition (``judge``) and what its trace takes beside the recorder: the
    per-swing table and the command instant."""
    cond = b.cond
    e = panels._base(b, out, rec, seed)
    fell = e["fell"]
    j = cond.judge
    H, psi = b.H, b.home_yaw
    dt = rec.dt
    limits = panels._limits_ok(e)  # the drives' peak speed and the joint range over the episode (and the RMS)
    ok: dict = dict(survived=not fell, drive_speed=limits["drive_speed"], joint_range=limits["joint_range"])
    bar: dict = {}
    measures: dict = dict(bar=bar, window=None, stage1=None, legs=None, push=None, recovery=None)
    swings: list = []
    drives_episode = (readings.drive_ratios(rec.force_sq, rec.sustained, rec.actuators, b.leg_mask, 0, rec.steps)
                      if rec.steps else None)
    drives_window = None
    command_ref = None if b.command is None else float(b.command["ref_s"])
    t_c = None if command_ref is None else rec.sim_time(command_ref)
    window_reached = False
    if j == "bar_walk":
        ended = _ran_to_end(b, rec)
        if not fell and not ended:
            raise Unreadable(f"{cond.key} seed {seed}: no fall, and the trace ends at "
                             f"{float(rec.t[-1]) if rec.rows else 0.0:.3f} s, before the planned {_planned_s(b):.3f} s")
        window = None
        if not fell:
            tb = float(rec.t[-1])
            window = (tb - tune.WALK_WINDOW_S, tb)
        measures["stage1"], measures["window"] = stage1_walk_reading(b, rec, out, window)
        if window is not None:
            ta, tb = window
            w = metrics.window(rec, ta, tb, psi, H)
            line = readings.line_reading(rec.t, _com(rec), psi, ta, tb, c_episode_start=_com(rec)[0])
            steady = rec.sim_time(float(b.resolved["steady_ref_s"]))
            if not (w.get("complete") and line["window_reached"] and line["complete"]
                    and steady is not None and steady <= ta + EPS_S):
                raise Unreadable(f"{cond.key} seed {seed}: the last {tune.WALK_WINDOW_S:g} s ([{ta:.3f}, {tb:.3f}] s) "
                                 f"are not all in the walk's steady part (it begins at {steady} s)")
            window_reached = True
            toe, swings = _toe(b, rec, ta, tb)
            drives_window = readings.drive_ratios(rec.force_sq, rec.sustained, rec.actuators, b.leg_mask,
                                                  *_window_steps(rec, ta, tb))
            target = tune.WALK_WINDOW_S * b.target_speed
            ok.update(_window_limits(w, cond, b.target_speed))
            ok.update(along=line["along_commanded_heading_m"] >= target, line=line["lateral_from_line_max_m"] <= readings.LINE_MAX_M,
                      toe_clearance_own_swing=toe["ok"])
            bar.update(line, along_target_m=target, line_max_m=readings.LINE_MAX_M, toe=toe,
                       heading_drift_deg=math.degrees(w["heading_change_rad"]), grf_peak_bw=w["grf_peak_bw"],
                       cadence_steps_s=w["cadence_steps_s"], stride_H=w["stride_H"], duty=w["duty"],
                       stage1_toe_clearance_m=w["toe_clearance_m"])
            measures["look"] = panels.look(rec, ta, tb, H, b.spec.stride_s, b.track, b.p.joints)
    elif j == "bar_turn":
        at_s, sign = float(b.resolved["at_s"]), float(b.resolved["sign"])
        if t_c is None and not fell:
            raise Unreadable(f"{cond.key} seed {seed}: no fall, and the reference clock never reached the command")
        if t_c is not None:
            tr = readings.turn_reading(rec.t, rec.yaw_unwrapped(), t_c, psi, sign, float(b.resolved["angle_rad"]), at_s)
            bar.update(tr)
            if not fell and not tr["window_reached"]:
                raise Unreadable(f"{cond.key} seed {seed}: no fall, and the trace ends at {float(rec.t[-1]):.3f} s, "
                                 f"before the command + {at_s + readings.HEADING_HOLD_S:g} s "
                                 f"({t_c + at_s + readings.HEADING_HOLD_S:.3f} s)")
            if tr["window_reached"]:
                ta, tb = t_c, t_c + at_s + readings.HEADING_HOLD_S
                w = metrics.window(rec, ta, tb, psi, H)
                if not w.get("complete"):
                    raise Unreadable(f"{cond.key} seed {seed}: the window [{ta:.3f}, {tb:.3f}] s cannot be read")
                window_reached = True
                toe, swings = _toe(b, rec, ta, tb)
                drives_window = readings.drive_ratios(rec.force_sq, rec.sustained, rec.actuators, b.leg_mask,
                                                      *_window_steps(rec, ta, tb))
                measures["window"] = {k: w.get(k) for k in panels.WINDOW_KEYS}
                ok.update(_window_limits(w, cond, b.target_speed))
                ok.update(heading=tr["heading"], toe_clearance_own_swing=toe["ok"])
                bar.update(toe=toe, grf_peak_bw=w["grf_peak_bw"], duty=w["duty"],
                           stage1_toe_clearance_m=w["toe_clearance_m"])
        measures["legs"] = metrics.leg_gap(rec, lg) if lg is not None else None
        ok["no_leg_interference"] = bool(measures["legs"] and measures["legs"]["interference_free"])
    elif j == "bar_nudge":
        t_on = b.push_onset(seed)
        f = b.push_force()
        t_c = float(t_on)  # a nudge's command is its onset
        ended = _ran_to_end(b, rec)
        if not fell and not ended:
            raise Unreadable(f"{cond.key} seed {seed}: no fall, and the trace ends at "
                             f"{float(rec.t[-1]) if rec.rows else 0.0:.3f} s, before the planned {_planned_s(b):.3f} s")
        window_reached = ended
        rcv = panels.recovery(rec, t_on + PUSH_S, standing=False, speed_cmd=b.speed_cmd, stride_s=b.stride_s,
                              limit_s=cond.recovery_s, until_s=_planned_s(b))
        measures["push"] = dict(onset_s=t_on, end_s=t_on + PUSH_S, direction=cond.direction, frac=float(cond.frac),
                                force_N=f.tolist(), dv_m_s=float(np.linalg.norm(f) * PUSH_S / b.mass),
                                impulse_N_s=float(np.linalg.norm(f) * PUSH_S), bound_m_s=b.push_bound_m_s,
                                fell_before_pulse=bool(fell and out["fall_time_s"] < t_on))
        measures["recovery"] = rcv
        ok["recovered"] = (not fell) and bool(rcv["recovered"])
    elif j == "bar_stop":
        if t_c is None and not fell:
            raise Unreadable(f"{cond.key} seed {seed}: no fall, and the reference clock never reached the command")
        t_rest = rec.sim_time(float(b.resolved["rest_ref_s"]))
        window_reached = bool(t_rest is not None and float(rec.t[-1]) >= t_rest + STOP_HOLD_S - EPS_S)
        if not fell and not window_reached:
            raise Unreadable(f"{cond.key} seed {seed}: no fall, and the trace does not hold {STOP_HOLD_S:g} s after "
                             f"the planned rest ({t_rest} s; it ends at {float(rec.t[-1]):.3f} s)")
        if t_c is not None:
            sr = readings.stop_reading(rec.t, _com(rec), rec.U[:, [metrics.C["v_x"], metrics.C["v_y"]]], rec.contact(),
                                       dt, t_c, psi, b.speed_cmd, b.stride_s)
            acc = (metrics.acceleration_peak(rec, t_c, t_c + 2.0 * b.stride_s) if rec.rows > 3 else None)
            n = sr["touchdowns_after_command"]
            ok.update(stopped_by_end=bool(sr["stopped_by_end"]),
                      touchdowns_after_command=n is not None and n <= readings.STOP_TOUCHDOWNS_MAX,
                      stop_acceleration_1hz=acc is not None and acc <= readings.ACCEL_MAX_M_S2)
            d18 = next((s for s in (out.get("stops") or []) if s.get("name") == cond.move), None)
            bar.update(sr, stop_acceleration_1hz_m_s2=acc, rest_s=t_rest, command_convention=b.command["convention"],
                       decision_18=None if d18 is None else {k: d18.get(k) for k in DECISION_18_KEYS})
    else:
        raise ValueError(f"judge {j!r}")
    if j in ("bar_nudge", "bar_stop") and rec.rows > 1:  # reported: the own-swing reading over the whole trace
        try:
            toe, swings = _toe(b, rec, float(rec.t[0]), float(rec.t[-1]))
            bar["toe_reported"] = toe
        except ValueError as exc:
            bar["toe_reported"] = dict(error=str(exc))
    span = drives_window if j in ("bar_walk", "bar_turn") else drives_episode
    ok[A1_TERM] = bool(span is not None and span["drive_sustained"])
    ok[LEGS_TERM] = bool(span is not None and span["drive_sustained_legs"])
    bar["drives"] = dict(judged_span="window" if j in ("bar_walk", "bar_turn") else "episode", window=drives_window,
                         episode=drives_episode)
    success = {t: bool(ok.get(t, False)) and not (fell and t != "survived") for t in terms_of(cond)}
    but_a1 = {t: bool(ok.get(t, False)) and not (fell and t != "survived") for t in terms_but_a1(cond)}
    if fell:
        success["survived"] = but_a1["survived"] = False
    before = None if (t_c is None and j == "bar_walk") else bool(fell and (t_c is None or out["fall_time_s"] < t_c))
    measures["convention"] = convention_check(rec) if rec.rows > 2 else None
    e.update(success=success, succeeded=bool(all(success.values())), success_but_A1=but_a1,
             succeeded_but_A1=bool(all(but_a1.values())),
             failed_terms=["survived"] if fell else [t for t, v in success.items() if not v],
             failed_terms_but_A1=["survived"] if fell else [t for t, v in but_a1.items() if not v],
             command_s=t_c, command_ref_s=command_ref, fell_before_command=before,
             segment_reached=out.get("segment_reached"), window_reached=bool(window_reached), omega=b.omega,
             crouch_frac=b.crouch_frac, measures=measures)
    return e, dict(swings=swings, command_s=t_c)


def judge(b, out: dict, rec: metrics.Recorder, seed: int, lg=None) -> dict:
    """The episode record of one episode: ``panels.judge``'s for a stock condition; for a bar condition
    ``panels._base``'s fields, the two success vectors (``success``, ``success_but_A1``), the failed terms, the
    command, and the item's measures.  An episode that did not fall and cannot be read raises ``Unreadable``."""
    if not is_bar(b.cond):
        return panels.judge(b, out, rec, seed, lg)
    return _judge(b, out, rec, seed, lg)[0]


def swing_table(swings: list) -> np.ndarray:
    """The per-swing table as an array [n, 5]: foot, t_lo, t_td, min (NaN: not read), class (its index in
    ``readings.SWING_CLASSES``)."""
    cls = {c: i for i, c in enumerate(readings.SWING_CLASSES)}
    return np.array([[r[0], r[1], r[2], math.nan if r[3] is None else r[3], cls[r[4]]] for r in swings],
                    float).reshape(-1, 5)


def trace_arrays(rec: metrics.Recorder, swings: list, command_s: float | None) -> dict:
    """``RULES["trace"]``'s arrays: ``panels.write_trace``'s five, unchanged, then the contact bits (per foot,
    ``np.packbits``; ``steps`` their count), ``dt``, ``weight``, the updates' times, both feet's low points and the
    reference's swing foot at every update (what reading (a) takes), the per-swing table and the command instant
    (NaN: none)."""
    every = panels.TRACE_EVERY
    U = rec.U[: rec.rows : every] if rec.rows else np.zeros((0, len(metrics.COLS)))
    Q = rec.Q[: rec.rows : every] if rec.rows else np.zeros((0, 0))
    contact = rec.contact() if rec.steps else np.zeros((0, 2), bool)
    return dict(t=U[:, metrics.C["t"]].astype(np.float64), t_ref=U[:, metrics.C["t_ref"]].astype(np.float64),
                rows=U.astype(np.float32), cols=np.array(metrics.COLS), q=Q.astype(np.float32),
                contact=np.packbits(contact.T, axis=1), steps=np.array(len(contact)), dt=np.array(float(rec.dt)),
                weight=np.array(float(rec.weight)), t100=np.asarray(rec.t, np.float64).copy(),
                low100=_lows(rec).astype(np.float32) if rec.rows else np.zeros((0, 2), np.float32),
                ref_swing100=(np.rint(rec.col("ref_swing")).astype(np.int8) if rec.rows else np.zeros(0, np.int8)),
                swings=swing_table(swings), swing_classes=np.array(readings.SWING_CLASSES),
                command_s=np.array(math.nan if command_s is None else float(command_s)))


def write_trace(root, rel, rec: metrics.Recorder, swings: list, command_s: float | None) -> dict:
    """A bar episode's trace into ``root / rel`` (compressed, written whole, then moved into place); its path
    (relative to ``root``), sha256 and rows."""
    arrays = trace_arrays(rec, swings, command_s)
    sha = s2io.write_npz(Path(root) / rel, arrays, compressed=True)
    return dict(path=Path(rel).as_posix(), sha256=sha, rows=int(len(arrays["t"])),
                every_s=panels.TRACE_EVERY * control.FEEDBACK_EVERY * float(rec.dt), updates=int(rec.rows),
                steps=int(rec.steps))


def read_trace_contact(path) -> np.ndarray:
    """The contact flags [steps, 2] of a bar trace."""
    with np.load(path, allow_pickle=False) as z:
        n = int(z["steps"])
        return np.unpackbits(z["contact"], axis=1)[:, :n].T.astype(bool)


def run_episode(row: panels.Row, ctx: panels.Context, cond, seed: int, relax=(), trace_root=None) -> dict:
    """One episode of ``row`` on ``cond`` and its record: ``row.episode``, then ``judge`` (``panels.judge`` and
    ``panels.write_trace`` unchanged for a stock condition), the trace under ``trace_root`` when given, the lifted
    limits and the wall time."""
    t0 = time.perf_counter()
    b, out, rec = row.episode(ctx, cond, int(seed), tuple(relax))
    lg = ctx.lg if b.legs else None
    rel = panels.trace_path(row.name, cond, int(seed))
    if is_bar(cond):
        e, extra = _judge(b, out, rec, int(seed), lg)
        if trace_root is not None:
            e["trace"] = write_trace(trace_root, rel, rec, extra["swings"], extra["command_s"])
    else:
        e = panels.judge(b, out, rec, int(seed), lg)
        if trace_root is not None:
            e["trace"] = panels.write_trace(trace_root, rel, rec)
    e.update(relaxed=list(relax), wall_s=time.perf_counter() - t0)
    return e


def failed_episode(cond, seed: int, reason: str, relax=(), wall_s: float = 0.0) -> dict:
    """``RULES["error"]``: an episode that failed to run, recorded as a fall at the start with every term failed and
    the reason; it is not a result."""
    if not is_bar(cond):
        return dict(panels.failed_episode(cond, seed, reason, relax), wall_s=float(wall_s))
    terms, t3 = terms_of(cond), terms_but_a1(cond)
    return dict(seed=int(seed), fell=True, fall_time_s=0.0, fall_cause="evaluation failed", fall_onset_s=None,
                fall_candidates=[], limits_before_onset=None, survival_s=0.0, planned_s=None, strides=0.0,
                succeeded=False, success={t: False for t in terms}, failed_terms=list(terms), succeeded_but_A1=False,
                success_but_A1={t: False for t in t3}, failed_terms_but_A1=list(t3), command_s=None,
                command_ref_s=None, fell_before_command=None, segment_reached=None, window_reached=False,
                measures={}, relaxed=list(relax), error=str(reason), wall_s=float(wall_s))


def convention_check(rec: metrics.Recorder) -> dict:
    """The recorder's one-step timing convention read on an episode (``RULES["convention"]``): at every touchdown
    step b that an update row shares (``t = b dt``), the foot's low point at that row against one physics step's
    travel (the low point's largest change per step over the two updates before).  By the convention the row shows
    the foot after its touchdown step, so its low point is at the floor or pressed a little into it; were the rows
    a step early, the foot would still be above the floor.  ``holds``: no such low point is more than one step's
    travel above the floor.  ``within_one_step`` counts the rows whose low point is within one step's travel of the
    floor on either side (the soft contact lets a foot sink further before its load passes 2 % of the weight, so
    this count can be below the number of rows while the convention holds); the low point one update earlier is
    reported beside."""
    t, dt = rec.t, rec.dt
    low, contact = _lows(rec), rec.contact()
    rows_ = []
    for x, i in readings.touchdowns(contact, dt):
        k = int(np.searchsorted(t, x - EPS_S))
        if k >= len(t) or abs(t[k] - x) > EPS_S or k < 2:
            continue
        travel = float(np.max(np.abs(np.diff(low[k - 2 : k + 1, i]))) / control.FEEDBACK_EVERY)
        rows_.append((float(low[k, i]), travel, float(low[k - 1, i])))
    if not rows_:
        return dict(touchdowns_on_rows=0, holds=None, step_offset=readings.STEP_OFFSET)
    a = np.array(rows_)

    def mm(x):
        return dict(min=1e3 * float(np.min(x)), median=1e3 * float(np.median(x)), max=1e3 * float(np.max(x)))

    return dict(touchdowns_on_rows=len(a), low_at_touchdown_row_mm=mm(a[:, 0]), step_travel_mm=mm(a[:, 1]),
                low_one_update_before_mm=mm(a[:, 2]), above_floor=int(np.sum(a[:, 0] > 0.0)),
                within_one_step=int(np.sum(np.abs(a[:, 0]) <= a[:, 1] + 1e-12)),
                holds=bool(np.all(a[:, 0] <= a[:, 1] + 1e-12)), step_offset=readings.STEP_OFFSET)


# ------------------------------------------------------------------------------------------------
# Counts and labels
# ------------------------------------------------------------------------------------------------
READINGS = {"success": ("succeeded", "failed_terms"), "but_A1": ("succeeded_but_A1", "failed_terms_but_A1")}
ATTRIBUTED_CONTROLLERS = ("R2",)


def counts(episodes: list[dict], seeds, which: str = "success") -> dict:
    """The episodes' counts and bounds in one reading (``which``: "success", reading 2; "but_A1", reading 3) and
    the pass (``RULES["pass_"]``): judged only when the record holds the 20 seeds of one block in order and no
    episode failed to run."""
    skey = READINGS[which][0]
    n = len(episodes)
    errors = int(sum(1 for e in episodes if e.get("error")))
    surv = int(sum(not e["fell"] for e in episodes))
    succ = int(sum(bool(e.get(skey)) for e in episodes))
    seeds = [int(s) for s in seeds]
    judged = bool(n == RUNS and [e["seed"] for e in episodes] == seeds and errors == 0
                  and any(seeds == list(b) for b in BLOCKS.values()))
    return dict(reading=which, episodes=n, survived=surv, succeeded=succ, falls=n - surv - errors, errors=errors,
                survival_lcb=panels.lcb(surv, n), success_lcb=panels.lcb(succ, n), judged=judged,
                pass_=bool(surv == RUNS and succ >= MIN_SUCCEED) if judged else None)


def success_label(episodes: list[dict], which: str = "success") -> dict:
    """``panels.SUCCESS_LABEL_RULE`` over the episodes that survived but failed, in one reading."""
    skey, fkey = READINGS[which]
    failed = [e for e in episodes if not e["fell"] and not e.get(skey)]
    c = Counter(t for e in failed for t in (e.get(fkey) or ()))
    hw = [t for t in HARDWARE_TERMS if c.get(t, 0) > len(failed) / 2]
    return dict(failed_episodes=len(failed), terms={t: dict(count=n, cls=term_class(t)) for t, n in c.most_common()},
                hardware_terms=hw, label=("body-limited: " + ", ".join(hw)) if hw else "controller-limited")


def failure(row: str, episodes: list[dict], attribution: dict | None, pass_: bool | None,
            which: str = "success") -> dict | None:
    """``panels.FAILURE_LABEL_RULE`` in one reading: both labels and the record's (None if it passes or is not
    judged)."""
    if pass_ is None or pass_:
        return None
    falls = [e for e in episodes if e["fell"] and not e.get("error")]
    cands = Counter(c for e in falls for c in (e.get("fall_candidates") or ()))
    sl = success_label(episodes, which)
    out = dict(reading=which, falls=len(falls), fall_causes=dict(Counter(str(e["fall_cause"]) for e in falls)),
               fall_candidates=dict(cands), success=sl, success_failures=sl["failed_episodes"])
    falls_label = None
    if falls:
        if attribution is not None:
            falls_label = attribution["label"]
        elif panels.controller_of(row) in ATTRIBUTED_CONTROLLERS:
            falls_label = "not attributed (falls; run with --attribute)"
        else:
            falls_label = (f"not attributed (falls; attribution is applied to {', '.join(ATTRIBUTED_CONTROLLERS)} "
                           "rows)")
    success_label_ = sl["label"] if sl["failed_episodes"] else None
    if not falls:
        kind = "success"
    elif sl["failed_episodes"] <= RUNS - MIN_SUCCEED:
        kind = "falls"
    else:
        kind = "falls" if len(falls) >= sl["failed_episodes"] else "success"
    return dict(out, falls_label=falls_label, success_label=success_label_, kind=kind,
                label=falls_label if kind == "falls" else success_label_,
                untestable=list((attribution or {}).get("not_relaxable") or []))


def fall_split(episodes: list[dict]) -> dict:
    """Falls before and after the command (a walk has none: its falls are ``no_command``); episodes that failed to
    run are left out."""
    falls = [e for e in episodes if e["fell"] and not e.get("error")]
    return dict(before_command=sum(1 for e in falls if e.get("fell_before_command") is True),
                after_command=sum(1 for e in falls if e.get("fell_before_command") is False),
                no_command=sum(1 for e in falls if e.get("fell_before_command") is None))


# ------------------------------------------------------------------------------------------------
# Sources, keys, paths
# ------------------------------------------------------------------------------------------------
BAR_SOURCE_FILES = (Path(__file__).resolve(), Path(readings.__file__).resolve(), Path(setio.__file__).resolve(),
                    Path(speedfit.__file__).resolve())


def bar_sources() -> dict:
    """The sha256 of the modules a bar episode runs beside the panels' sources: ``bar.py``, ``readings.py``,
    ``setio.py`` and ``speedfit.py`` (each by its path relative to the study directory), and one hash over them."""
    files = {s2io.portable(str(p)): s2io.sha256_file(p) for p in BAR_SOURCE_FILES}
    return dict(sha256=s2io.sha256_json(files), files=files, rule=RULES["key"])


def _plain_json(x):
    """``x`` after a JSON round trip (what a record read back from its file holds)."""
    return json.loads(json.dumps(s2io.plain(x)))


def record_key(body: str, set_name: str, block: str, row: panels.Row, cond, resolved: dict | None, seeds,
               inputs: dict, context: dict, sources: dict, omega: float | None, crouch_frac: float | None) -> dict:
    """``RULES["key"]``'s key of a record.  ``resolved`` is the planner's value (None for a condition without a
    reference: its "not run" record's key holds no plan); ``inputs`` is ``setio.read_inputs``' first value;
    ``sources`` is ``{"panels": sha256, "bar": sha256}``."""
    files = [] if resolved is None else list(resolved["files"])
    return _plain_json(dict(
        schema=SCHEMA, body=body, set=set_name, block=block, row=s2io.portable(row.describe()),
        row_sources=s2io.portable(row.sources()), condition=describe_condition(cond),
        plan=None if resolved is None else [list(x) for x in resolved["plan"]],
        seconds=None if resolved is None else resolved["seconds"], seeds=[int(s) for s in seeds],
        command=None if resolved is None else resolved.get("command"), omega=omega, crouch_frac=crouch_frac,
        inputs={f: inputs["files"][f] for f in files}, context=context, sources=dict(sources)))


def key_differences(a, b, path: str = "key") -> list[str]:
    """The fields in which two keys differ, as ``path: old -> new`` lines (long values shortened)."""
    def short(x):
        s = json.dumps(x, sort_keys=True)
        return s if len(s) <= 80 else s[:77] + "..."

    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: (absent) -> {short(b[k])}")
            elif k not in b:
                out.append(f"{path}.{k}: {short(a[k])} -> (absent)")
            else:
                out += key_differences(a[k], b[k], f"{path}.{k}")
        return out
    return [] if a == b else [f"{path}: {short(a)} -> {short(b)}"]


def run_root(out, control: str | None = None) -> Path:
    """The directory records go under: ``out``, or ``out/controls/<control>`` for a reported control."""
    if control is not None and control not in CONTROLS:
        raise ValueError(f"control {control!r}: choose from {', '.join(CONTROLS)}")
    return Path(out) if control is None else Path(out) / "controls" / control


def block_dir(out, set_name: str, body: str, block: str, control: str | None = None) -> Path:
    return run_root(out, control) / set_name / body / block


def record_path(out, set_name: str, body: str, block: str, row: str, cond, control: str | None = None) -> Path:
    """``<out>/<set>/<body>/<block>/eval/<row>/<behavior>/<condition>.json`` (a row "R2/42@10ms" is two
    directories)."""
    return block_dir(out, set_name, body, block, control) / "eval" / row / cond.behavior / f"{cond.name}.json"


def partial_root(out, set_name: str, body: str, block: str, row: str, cond, control: str | None = None) -> Path:
    return block_dir(out, set_name, body, block, control) / "partial" / row / cond.behavior / cond.name


def _read(path) -> dict | None:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def read_partials(pdir: Path, key_sha256: str, seeds) -> dict:
    """{seed: partial document} of the finished episodes under ``pdir`` that belong to the key ``key_sha256``: a
    file that cannot be parsed (torn), that names another key or seed, or whose episode failed to run is left out
    (the seed is run again)."""
    out = {}
    for s in seeds:
        doc = _read(pdir / f"{int(s)}.json")
        if not doc or doc.get("key_sha256") != key_sha256:
            continue
        e = doc.get("episode")
        if isinstance(e, dict) and e.get("seed") == int(s) and not e.get("error"):
            out[int(s)] = doc
    return out


def drop_partials(cond_dir: Path) -> None:
    """Remove a condition's partial files (every key's) and the directories they leave empty, up to ``partial/``."""
    shutil.rmtree(cond_dir, ignore_errors=True)
    d = cond_dir.parent
    while "partial" in d.parts:
        try:
            d.rmdir()
        except OSError:
            return
        if d.name == "partial":
            return
        d = d.parent


# ------------------------------------------------------------------------------------------------
# The budget ledger
# ------------------------------------------------------------------------------------------------
class BudgetCut(BarError):
    """A unit does not fit its task's budget, or an earlier unit of the task was cut (``RULES["budget"]``)."""


class Budget:
    """``<out>/budget.json``: per task the units in the order started (each with its estimate, the core-hours spent
    before it, its start time and the core-hours it used) and the cut units.  Every write is atomic.  Wall seconds
    are added by item (a record's episodes of one session): adding an item again replaces its value if larger, so a
    resumed run never counts an episode twice."""

    SCHEMA = "bar-budget-1"

    def __init__(self, path, task: int, cap_core_h: float | None = None):
        self.path, self.task = Path(path), int(task)
        if cap_core_h is None and self.task not in BUDGET_CORE_H:
            raise ValueError(f"task {task!r}: choose from {', '.join(str(t) for t in BUDGET_CORE_H)}")
        self.cap = float(BUDGET_CORE_H[self.task] if cap_core_h is None else cap_core_h)

    def load(self) -> dict:
        doc = _read(self.path) or dict(schema=self.SCHEMA, rule=RULES["budget"], tasks={})
        t = doc["tasks"].setdefault(str(self.task), dict(cap_core_h=self.cap, units=[], cut=[]))
        t["cap_core_h"] = self.cap
        return doc

    def _task(self, doc: dict) -> dict:
        return doc["tasks"][str(self.task)]

    @staticmethod
    def _used(unit: dict) -> float:
        return float(sum(unit["items"].values())) / 3600.0

    def spent(self, doc: dict | None = None) -> float:
        """The task's spent core-hours."""
        return float(sum(self._used(u) for u in self._task(doc or self.load())["units"]))

    def unit(self, name: str, doc: dict | None = None) -> dict | None:
        return next((u for u in self._task(doc or self.load())["units"] if u["unit"] == name), None)

    def start(self, name: str, estimate_core_h: float, detail: dict | None = None) -> bool:
        """Start (or resume) the unit ``name``.  False, with the unit appended to the cut list, when an earlier
        unit of the task was cut or the spent core-hours plus the estimate exceed the cap; a unit that was started
        before always continues."""
        doc = self.load()
        t = self._task(doc)
        if any(u["unit"] == name for u in t["units"]):
            return True
        spent = self.spent(doc)
        why = None
        if t["cut"]:
            why = f"an earlier unit of task {self.task} was cut ({t['cut'][0]['unit']}): none is skipped"
        elif spent + estimate_core_h > self.cap:
            why = (f"spent {spent:.3f} + estimate {estimate_core_h:.3f} core-h exceeds task {self.task}'s "
                   f"{self.cap:g} core-h")
        if why is not None:
            if not any(c["unit"] == name for c in t["cut"]):
                t["cut"].append(dict(unit=name, estimate_core_h=float(estimate_core_h), spent_before_core_h=spent,
                                     at=s2io.now_utc(), reason=why, label="cut by the pre-registered order",
                                     detail=detail))
                s2io.write_json(self.path, doc)
            return False
        t["units"].append(dict(unit=name, estimate_core_h=float(estimate_core_h), spent_before_core_h=spent,
                               started_at=s2io.now_utc(), used_core_h=0.0, items={}, detail=detail))
        s2io.write_json(self.path, doc)
        return True

    def cut_reason(self, name: str) -> str | None:
        return next((c["reason"] for c in self._task(self.load())["cut"] if c["unit"] == name), None)

    def add(self, name: str, item: str, wall_s: float) -> None:
        """Count ``wall_s`` seconds of episodes under ``item`` of the unit ``name`` (the larger of the item's
        value and ``wall_s``)."""
        doc = self.load()
        u = self.unit(name, doc)
        if u is None:  # an item of a unit this ledger never started (its name was changed): keep it, labelled
            u = dict(unit=name, estimate_core_h=None, spent_before_core_h=self.spent(doc), started_at=s2io.now_utc(),
                     used_core_h=0.0, items={}, detail=dict(note="started without an estimate"))
            self._task(doc)["units"].append(u)
        u["items"][item] = max(float(u["items"].get(item, 0.0)), float(wall_s))
        u["used_core_h"] = self._used(u)
        u["updated_at"] = s2io.now_utc()
        s2io.write_json(self.path, doc)


def read_cost(path) -> dict:
    """The rates of a committed ``COST.md`` (``dry_run``'s output): its fenced JSON block, ``{"schema":
    "bar-cost-1", "rates": {"R2": {"wall_per_sim_s": ..}, "R0": {..}}, ..}``."""
    text = Path(path).read_text()
    for m in re.finditer(r"```json\n(.*?)\n```", text, re.DOTALL):
        try:
            doc = json.loads(m.group(1))
        except ValueError:
            continue
        if isinstance(doc, dict) and doc.get("schema") == COST_SCHEMA:
            return doc
    raise BarError(f"{path}: no {COST_SCHEMA} block (write it with: bar dry-run ... > COST.md)")


def rate_of(rates: dict, row_name: str) -> float:
    """Wall seconds per simulated second of a row: its controller's measured rate, else the largest measured."""
    r = rates.get(controller_of(row_name))
    if r is None:
        return max(float(v["wall_per_sim_s"]) for v in rates.values())
    return float(r["wall_per_sim_s"])


# ------------------------------------------------------------------------------------------------
# Records
# ------------------------------------------------------------------------------------------------
def condition_record(body: str, set_name: str, block: str, row: panels.Row, cond, episodes: list[dict], key: dict,
                     seeds, prov: dict, inputs: dict, sessions: list, set_docs: dict | None = None,
                     attribution: dict | None = None, extra: dict | None = None) -> dict:
    """A condition's record: Stage 1's keys (``panels.condition_record``) with the bar's (module docstring).
    ``counts`` is reading 2, ``counts_but_A1`` reading 3 (None for a stock condition), ``counts_look`` reading 1
    (None, with its reason)."""
    bar = is_bar(cond)
    c2 = counts(episodes, seeds, "success")
    c3 = counts(episodes, seeds, "but_A1") if bar else None
    a1_over = False
    if bar and cond.move and set_docs is not None:
        a1_over = bool((set_docs["transitions.json"]["transitions"].get(cond.move) or {}).get("a1_over"))
    f2 = failure(row.name, episodes, attribution, c2["pass_"], "success")
    f3 = failure(row.name, episodes, attribution, c3["pass_"], "but_A1") if bar else None
    if a1_over:  # reading 2 gives a turn frozen with a1_over no verdict
        c2 = dict(c2, pass_=None, not_read="a1_over", reason="not read: a1_over", pass_if_read=c2["pass_"])
    for e in episodes:
        if bar and not e.get("error") and e.get("omega") is not None and e["omega"] != key["omega"]:
            raise BarError(f"{row.name} {cond.key} seed {e['seed']}: the episode ran at omega {e['omega']!r}, the "
                           f"key holds {key['omega']!r}")
    t3 = terms_but_a1(cond)
    rec = dict(schema=SCHEMA, body=body, set=set_name, block=block, row=row.name, behavior=cond.behavior,
               condition=cond.name, key_name=cond.key, gated=row.gated(cond), plan_gated=cond.gated, status="done",
               controller=s2io.portable(row.describe()), definition=describe_condition(cond), plan=key["plan"],
               seconds=key["seconds"], seeds=[int(s) for s in seeds], command=key["command"], omega=key["omega"],
               crouch_frac=key["crouch_frac"], counts=c2, counts_but_A1=c3, counts_look=None,
               counts_look_reason=LOOK_NOT_DEFINED if bar else "a stock condition has Stage 1's one reading",
               a1_over=a1_over, failure=f2, failure_but_A1=f3, attribution=attribution, falls=fall_split(episodes),
               terms=list(terms_of(cond)), terms_but_A1=None if t3 is None else list(t3),
               term_classes={t: term_class(t) for t in tuple(terms_of(cond)) + tuple(t3 or ())},
               episodes=episodes, rules=rules_of(cond, set_docs), key=key, key_sha256=s2io.sha256_json(key),
               provenance=prov, inputs=dict(dir=s2io.portable(inputs.get("dir")), files=key["inputs"]),
               set_freeze=inputs.get("freeze"), sessions=list(sessions),
               episode_wall_s=float(sum(e.get("wall_s") or 0.0 for e in episodes)),
               simulated_s=float(sum(e.get("survival_s") or 0.0 for e in episodes)), ended_at=s2io.now_utc())
    rec.update(extra or {})
    return _plain_json(rec)


def not_run_record(body: str, set_name: str, block: str, row: panels.Row, cond, key: dict, why: dict, prov: dict,
                   inputs: dict | None = None, set_docs: dict | None = None, extra: dict | None = None) -> dict:
    """The record of a condition without a reference (or, for a stock condition, without a settled stance): status
    "not run" with the reason and the numbers of the missing reference.  It is not a pass."""
    rec = dict(schema=SCHEMA, body=body, set=set_name, block=block, row=row.name, behavior=cond.behavior,
               condition=cond.name, key_name=cond.key, gated=row.gated(cond), plan_gated=cond.gated, status="not run",
               reason=why.get("reason"), reference=why.get("reference"), numbers=why.get("numbers"),
               controller=s2io.portable(row.describe()), definition=describe_condition(cond),
               counts=dict(reading="success", episodes=0, errors=0, judged=False, pass_=None),
               counts_but_A1=dict(reading="but_A1", episodes=0, errors=0, judged=False, pass_=None) if is_bar(cond)
               else None, counts_look=None, counts_look_reason=LOOK_NOT_DEFINED, failure=None, failure_but_A1=None,
               attribution=None, episodes=[], rules=rules_of(cond, set_docs), key=key,
               key_sha256=s2io.sha256_json(key), provenance=prov, set_freeze=(inputs or {}).get("freeze"),
               sessions=[], ended_at=s2io.now_utc())
    rec.update(extra or {})
    return _plain_json(rec)


def supersede_record(path: Path, old: dict, reason: str) -> dict:
    """Move a finished record aside (``RULES["no_overwrite"]``) and return the new record's ``supersedes`` field."""
    kh = old.get("key_sha256") or s2io.sha256_json(old.get("key"))
    dest = path.with_name(path.name + ".superseded") / f"{kh[:12]}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise BarError(f"{dest} exists already: the record of this key was superseded before")
    os.replace(path, dest)
    return dict(reason=str(reason), label="re-run after a result", record=dest.name, key_sha256=kh,
                registered=dict(counts=old.get("counts"), counts_but_A1=old.get("counts_but_A1"),
                                ended_at=old.get("ended_at")), at=s2io.now_utc())


# ------------------------------------------------------------------------------------------------
# Workers (tune.Pool with the bar's evaluator)
# ------------------------------------------------------------------------------------------------
class BarWorker:
    """A worker's evaluator for ``tune.Pool``: ``evaluate((row, condition, relax, partial dir, key sha256), seed)``
    runs one episode and writes it as a partial file.  It refuses to start with another thread count than one, on
    other bar sources (``bar``: ``bar_sources()["sha256"]``) or on other panel sources or key context than the
    session's (``sources``: ``panels.fingerprint``); it reads the set itself (frozen and matching, or the ``S1``
    view) and rebuilds the rows from their specs."""

    def __init__(self, body: str, run_dir: str, mode: str = "bar", rate: float = RATE,
                 expected_inputs: dict | None = None, *, set_name: str, files=(), rows=(), sources: str | None = None,
                 bar: str | None = None, traces: str | None = None, crouch_frac: float | None = None,
                 session: dict | None = None, no_traces_for=()):
        s2io.require_one_thread()
        if bar is not None and bar_sources()["sha256"] != bar:
            raise tune.PoolError("the bar's sources (bar.py, readings.py, setio.py, speedfit.py) changed during the "
                                 "session: restore them and run again")
        if sources is not None:
            mine = panels.fingerprint(panels.panel_sources()["sha256"], panels.key_context(body))
            if mine != sources:
                raise tune.PoolError(f"the panels' sources or key context changed during the session ({mine[:12]}"
                                     f"..., the session's {sources[:12]}...): restore them and run again")
        self.body, self.set_name, self.run_dir = body, set_name, Path(run_dir)
        if set_name == STOCK_SET:
            self.inputs, docs = setio.stage1_view(self.run_dir.parent, body)
            set_doc = None
        else:
            self.inputs, docs = setio.read_inputs(self.run_dir, files, require_frozen=True)
            set_doc = setio.load(self.run_dir)["set.json"]
            if crouch_frac is None:
                set_crouch(set_doc)
        if expected_inputs is not None:
            diff = tune.input_differences(expected_inputs, self.inputs)
            if diff:
                raise tune.InputError(f"{self.run_dir}: the inputs differ from the ones expected: " + "; ".join(diff))
        ident = s2io.sha256_json(dict(files=self.inputs["files"], freeze=self.inputs.get("freeze")))
        self.ctx = context_for(body, self.run_dir, docs, set_doc, crouch_frac, ident)
        self.rows = {s["name"]: panels.row_from_spec(s) for s in rows}
        self.traces = None if traces is None else Path(traces)
        self.no_traces_for = tuple(no_traces_for)
        self.session = session

    def describe(self) -> dict:
        return dict(body=self.body, set=self.set_name, rows=sorted(self.rows), bar_sources=bar_sources()["sha256"])

    def evaluate(self, job, seed) -> dict:
        row, cond, relax, pdir, key_sha256 = job
        relax = tuple(relax)
        root = self.traces if not relax else None  # the attribution's lifted runs leave no trace
        if controller_of(row) in self.no_traces_for:
            root = None
        t0 = time.perf_counter()
        try:
            e = run_episode(self.rows[row], self.ctx, cond, int(seed), relax, trace_root=root)
        except Exception:
            e = failed_episode(cond, int(seed), traceback.format_exc(), relax, time.perf_counter() - t0)
        if pdir is not None:
            s2io.write_json(Path(pdir) / f"{int(seed)}.json",
                            dict(schema="bar-partial-1", key_sha256=key_sha256, seed=int(seed), relax=list(relax),
                                 session=self.session, episode=e, written_at=s2io.now_utc()))
        return e


def _worker(body, run_dir, mode, rate, expected_inputs=None, **kw):
    """``tune.Pool``'s factory (a top-level function of this importable module, bound with ``functools.partial``)."""
    return BarWorker(body, run_dir, mode, rate, expected_inputs, **kw)


def _lost(job, reason: str) -> dict:
    (row, cond, relax, pdir, key_sha256), seed = job
    return failed_episode(cond, seed, f"worker lost: {reason}", relax)


# ------------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------------
def _check_rule(set_name, row_names, episodes, task, dev_episodes, passed_of, control,
                block: str = HEADLINE_BLOCK) -> None:
    """``RULES["checks"]``: on a set of the 75 cm round a run without ``--task`` is a listed build check."""
    if set_name not in PREREGISTERED_SETS or task is not None:
        return
    bad = []
    if set_name != CHECK_SET:
        bad.append(f"the set is {set_name}, not {CHECK_SET}")
    if any(controller_of(r) != "R0" for r in row_names):
        bad.append("a row is not an R0 row")
    if int(episodes) != 1 or block != HEADLINE_BLOCK:
        bad.append(f"{episodes} episodes per condition on {block}, not 1 (seed {BLOCKS[HEADLINE_BLOCK][0]})")
    if dev_episodes is None:
        bad.append("--dev-episodes names no file to list the episodes in")
    if passed_of or control:
        bad.append("a replication or a control is no check")
    if bad:
        raise BarError(f"a run on {set_name} without --task must be a build check ({RULES['checks']}): "
                       + "; ".join(bad))


def _list_dev_episodes(path, entries: list) -> None:
    if not entries:
        return
    doc = _read(path) or dict(rule=RULES["checks"], note="check episodes of the build: smoke, no record", episodes=[])
    doc["episodes"] += entries
    s2io.write_json(path, doc)


def _attribution_needed(rec: dict, mode: str) -> bool:
    """Whether a record takes decision 20's attribution now (``RULES["attribution"]``): a finished record of an R2
    row with falls, none of its episodes failed to run, not attributed yet (or attributed with lifted runs that
    failed to run); ``mode`` "judged" takes the rows at the judged delay only."""
    if mode == "none" or rec.get("status") != "done" or panels.controller_of(rec["row"]) not in ATTRIBUTED_CONTROLLERS:
        return False
    if mode == "judged" and delay_ms(rec["row"]) != JUDGED_MS:
        return False
    c = rec["counts"]
    if c.get("errors") or c["falls"] <= 0:
        return False
    a = rec.get("attribution")
    return a is None or bool(a.get("lifted_errors"))


def walk_record_of(cond: BarCondition) -> str:
    """The key of the walk condition a turn, nudge or stop record cites: the same walk and grip."""
    tag = next(t for mu, t in GRIPS if mu == cond.friction)
    return f"walk/{cond.walk}{tag}"


def evaluate(set_name: str, body: str, out, rows, conditions=None, block: str = HEADLINE_BLOCK, procs: int = 2,
             attribute: str = "judged", traces: bool = True, episodes: int = RUNS, task: int | None = None,
             supersede: bool = False, log=print, *, stage1_run=STAGE1_RUN, root=setio.ROOT, passed_of: str | None = None,
             control: str | None = None, unit: str | None = None, rates: dict | None = None,
             supersede_reason: str | None = None, dev_episodes=None, timeout_s: float = tune.EVAL_TIMEOUT_S,
             budget_cap_core_h: float | None = None, no_traces_for=(), dev_reason: str | None = None) -> dict:
    """Run the bar's conditions ``conditions`` (keys or names; default: the 16 gated ones, or Stage 1's gated
    hard-floor conditions on the pseudo-set ``S1``) of ``set_name`` on ``body`` under ``rows`` (names, or a dict of
    built rows) into ``out``: one record per row and condition under
    ``<out>/<set>/<body>/<block>/eval/``.  Resumes per episode; never overwrites a finished record of another key
    (``supersede`` with ``supersede_reason`` keeps it beside the new one); ``attribute`` runs decision 20's
    attribution ("judged": rows at 10 ms; "all"; "none"); ``passed_of`` selects the records of that block that
    passed (replication); ``task`` counts every episode in ``<out>/budget.json`` and needs ``rates`` (``read_cost``'s
    ``rates``); ``control`` runs a reported control into ``<out>/controls/<control>/``; ``no_traces_for`` names
    controllers ("R0") whose rows write no trace.  Returns the block's summary with the session."""
    s2io.require_one_thread()
    if block not in BLOCKS:
        raise ValueError(f"block {block!r}: choose from {', '.join(BLOCKS)}")
    if attribute not in ("judged", "all", "none"):
        raise ValueError(f"attribute {attribute!r}: judged, all or none")
    if passed_of is not None and (passed_of not in BLOCKS or passed_of == block):
        raise ValueError(f"--passed-of {passed_of!r}: another block than {block}")
    if supersede and not supersede_reason:
        raise BarError("--supersede needs --supersede-reason: the new record states why it was run again")
    out = Path(out)
    t_start = time.perf_counter()
    seeds = [int(s) for s in BLOCKS[block][: int(episodes)]]
    stock = set_name == STOCK_SET
    row_objs = dict(rows) if isinstance(rows, dict) else globals()["rows"](body, list(rows), stage1_run)
    _check_rule(set_name, list(row_objs), episodes, task, dev_episodes, passed_of, control, block)
    crouch_override = None if control is None else CONTROLS[control][0]
    extra = {} if control is None else dict(control=CONTROLS[control][1], control_name=control)
    with s2io.Lock(out / ".lock"):
        s2io.sweep_tmp(out)
        if stock:
            if control is not None:
                raise BarError("a control runs on a reference set, not on the pseudo-set S1")
            run_dir = setio._anchored(stage1_run) / body
            inputs, docs = setio.stage1_view(stage1_run, body)
            set_docs, planner = None, panels.Planner(run_dir, docs["refs.json"]["references"],
                                                     docs["transitions.json"]["transitions"])
            all_conds = stock_conditions()
            conds = select(all_conds, conditions, stock_default())
            crouch, resolved, missing, files = 0.03, {}, {}, []
            for c in conds:
                stands, settle = panels.stands(run_dir, c.floor)
                if not stands:
                    missing[c] = dict(reference=f"floor {c.floor}", reason=panels.NOT_RUN_RULE, numbers=settle)
                else:
                    resolved[c] = dict(planner.resolve(c), files=panels.files_of(c), command=None)
        else:
            run_dir = setio.set_dir(set_name, body, root)
            set_docs = setio.load(run_dir)
            if set_docs["set.json"].get("set") != set_name:
                raise BarError(f"{run_dir}: set.json names the set {set_docs['set.json'].get('set')!r}")
            planner = BarPlanner(run_dir, set_docs["refs.json"]["references"],
                                 set_docs["transitions.json"]["transitions"], set_docs["set.json"])
            all_conds = globals()["conditions"](set_docs, block, every=True)
            conds = choose(set_docs, block, conditions)
            crouch = set_crouch(set_docs["set.json"]) if crouch_override is None else crouch_override
            resolved, missing = {}, {}
            for c in conds:
                why = planner.missing(c)
                if why is None:
                    resolved[c] = planner.resolve(c)
                else:
                    missing[c] = why
            files = sorted({f for r in resolved.values() for f in r["files"]})
            inputs, _ = setio.read_inputs(run_dir, files, require_frozen=True)
        omega = omega_of(body, run_dir, crouch) if resolved else None
        src, bsrc, kctx = panels.panel_sources(), bar_sources(), panels.key_context(body)
        key_sources = dict(panels=src["sha256"], bar=bsrc["sha256"])
        prov = dict(plant.provenance(), panel_sources=src, bar_sources=bsrc)
        git = s2io.git_state()
        started_at = s2io.now_utc()
        sess = dict(id=f"{started_at}/{os.getpid()}", started_at=started_at, commit=git["commit"], dirty=git["dirty"],
                    versions=panels.library_versions(), blas_threads=s2io.blas_threads(), procs=int(procs), task=task,
                    unit=None, block=block, attribute=attribute)
        session = dict(sess, set=set_name, body=body, rows=list(row_objs), conditions=[c.key for c in conds],
                       seeds=seeds, traces=bool(traces), passed_of=passed_of, control=control, episodes_run=0,
                       attribution_episodes_run=0, skipped=[], reused_partial=[], rerun_failed=[], not_run=[],
                       not_selected=[], superseded=[], attribution_cut=[], incidents=[])
        trace_root = block_dir(out, set_name, body, block, control) / "eval" / "traces" if traces else None

        def rel(path) -> str:
            return Path(os.path.relpath(path, out)).as_posix()

        # -- the plan of the session: every row and condition, before any episode
        todo, mismatched = [], []
        for cond in conds:
            for row in row_objs.values():
                if not row.runs(cond):
                    continue
                path = record_path(out, set_name, body, block, row.name, cond, control)
                rep = None
                if passed_of is not None:
                    first = _read(record_path(out, set_name, body, passed_of, row.name, cond, control))
                    passed = [w for w, k in (("success", "counts"), ("but_A1", "counts_but_A1"))
                              if first and first.get("status") == "done" and (first.get(k) or {}).get("pass_") is True]
                    if controller_of(row.name) in NO_VERDICT_CONTROLLERS or not passed:
                        session["not_selected"].append(f"{row.name} {cond.key}")
                        continue
                    rep = dict(block=passed_of, record=rel(record_path(out, set_name, body, passed_of, row.name, cond,
                                                                        control)),
                               key_sha256=first.get("key_sha256"), readings=passed, counts=first.get("counts"),
                               counts_but_A1=first.get("counts_but_A1"))
                res = resolved.get(cond)
                key = record_key(body, set_name, block, row, cond, res, seeds, inputs, kctx, key_sources,
                                 omega if res is not None else None, crouch if res is not None else None)
                kh = s2io.sha256_json(key)
                old = _read(path)
                item = dict(row=row, cond=cond, path=path, key=key, kh=kh, old=None, run=[], reused={}, rep=rep,
                            pdir=partial_root(out, set_name, body, block, row.name, cond, control) / kh[:16],
                            supersedes=None, rec=None, why=missing.get(cond))
                if old is not None and old.get("status") == "done" and old.get("key") != key:
                    if not supersede:
                        mismatched.append((path, key_differences(old.get("key"), key)))
                        continue
                    item["supersedes"] = (old, str(supersede_reason))
                elif old is not None and old.get("status") == "done":
                    item["old"] = old
                elif old is not None and old.get("key") == key:
                    item["old"] = old  # a "not run" record of this key
                todo.append(item)
        if mismatched:
            lines = [f"{p}: a finished record holds another key:\n    " + "\n    ".join(d[:12] + (["..."] if len(d) > 12
                                                                                               else []))
                     for p, d in mismatched]
            raise KeyMismatch("\n".join(lines) + "\nA finished record is never overwritten: use a new --out, or "
                              "--supersede with --supersede-reason (the old record is kept beside the new one).")
        for it in todo:
            if it["why"] is not None or it["supersedes"] is not None:
                continue
            old = it["old"]
            if old is not None and old.get("status") == "done":
                redo = [e["seed"] for e in old["episodes"] if e.get("error")]
                it["run"] = redo
                (session["rerun_failed"] if redo else session["skipped"]).append(
                    f"{it['row'].name} {it['cond'].key}" + (f": seeds {redo}" if redo else ""))
            else:
                it["reused"] = read_partials(it["pdir"], it["kh"], seeds)
                it["run"] = [s for s in seeds if s not in it["reused"]]
                if it["reused"]:
                    session["reused_partial"].append(f"{it['row'].name} {it['cond'].key}: seeds {sorted(it['reused'])}")
        for it in todo:
            if it["supersedes"] is not None and it["why"] is None:
                it["run"] = list(seeds)

        # -- the budget: the unit starts only if it fits
        budget = None
        unit_name = unit or (f"{set_name}{'' if control is None else ' control ' + control} {body} {block}"
                             f"{'' if passed_of is None else ' of ' + passed_of} rows {','.join(row_objs)} conditions "
                             f"{','.join(c.key for c in conds)}")
        sess["unit"] = session["unit"] = unit_name
        if task is not None:
            if rates is None:
                raise BarError("--task needs the committed rates (COST.md, read_cost): the unit's estimate is its "
                               "planned simulated seconds times them")
            budget = Budget(out / "budget.json", task, budget_cap_core_h)
            sim = {}
            for it in todo:
                if it["why"] is None:
                    sim[it["row"].name] = sim.get(it["row"].name, 0.0) + len(seeds) * float(it["key"]["seconds"])
            estimate = sum(s * rate_of(rates, r) for r, s in sim.items()) / 3600.0
            if not budget.start(unit_name, estimate, dict(simulated_s=sim, records=len(todo))):
                raise BudgetCut(f"cut by the pre-registered order: {unit_name}: {budget.cut_reason(unit_name)}")

        pool = None

        def get_pool():
            nonlocal pool
            if pool is None:
                pool = tune.Pool(body, run_dir, procs, mode="bar", rate=RATE, timeout_s=timeout_s,
                                 factory=functools.partial(
                                     _worker, set_name=set_name, files=files,
                                     rows=[r.spec() for r in row_objs.values()],
                                     sources=panels.fingerprint(src["sha256"], kctx), bar=bsrc["sha256"],
                                     traces=None if trace_root is None else str(trace_root),
                                     crouch_frac=crouch_override, session=sess,
                                     no_traces_for=tuple(no_traces_for)))
            return pool

        def count(unit_, item, wall):
            if budget is not None:
                budget.add(unit_, item, wall)

        def prior_sessions(it) -> list:
            """The sessions of reused partial episodes (a killed run's), counted in the ledger by item."""
            by = {}
            for s, doc in sorted(it["reused"].items()):
                ps = doc.get("session") or {}
                g = by.setdefault(ps.get("id"), dict(ps, seeds_run=[], wall_s=0.0, ended_at=None,
                                                     note="episodes kept from a run that did not finish"))
                g["seeds_run"].append(s)
                g["wall_s"] += float(doc["episode"].get("wall_s") or 0.0)
            for sid, g in by.items():
                if g.get("task") == task and task is not None:
                    count(g.get("unit") or unit_name, f"{rel(it['path'])}#{sid}", g["wall_s"])
            return list(by.values())

        try:
            for cond in conds:
                batch = [it for it in todo if it["cond"] is cond]
                for it in batch:
                    row, path, key = it["row"], it["path"], it["key"]
                    if it["why"] is not None:
                        old = it["old"]
                        if it["supersedes"] is not None:
                            it["supersedes"] = supersede_record(path, *it["supersedes"])
                        if not (old and old.get("status") == "not run" and old.get("key") == key):
                            s2io.write_json(path, not_run_record(
                                body, set_name, block, row, cond, key, it["why"], prov, inputs, set_docs,
                                dict(extra, **({} if it["supersedes"] is None else dict(supersedes=it["supersedes"])),
                                     **({} if it["rep"] is None else dict(replication_of=it["rep"])))))
                        session["not_run"].append(f"{row.name} {cond.key}: {it['why']['reason']}")
                        log(f"{set_name} {body} {block} {row.name} {cond.key}: not run ({it['why']['reason']})")
                    elif it["supersedes"] is not None:
                        it["supersedes"] = supersede_record(path, *it["supersedes"])
                        session["superseded"].append(f"{row.name} {cond.key}: {it['supersedes']['record']}")
                jobs = [((it["row"].name, cond, (), str(it["pdir"]), it["kh"]), s) for it in batch for s in it["run"]]
                results = get_pool().map(jobs, failed=_lost) if jobs else []
                session["episodes_run"] += len(jobs)
                i = 0
                listed = []
                for it in batch:
                    if it["why"] is not None:
                        continue
                    row, path, key, old = it["row"], it["path"], it["key"], it["old"]
                    new = dict(zip(it["run"], results[i : i + len(it["run"])], strict=True))
                    i += len(it["run"])
                    if old is not None and old.get("status") == "done" and not it["run"]:
                        it["rec"] = old
                        continue
                    sessions = list((old or {}).get("sessions") or []) if old and old.get("status") == "done" else []
                    sessions += prior_sessions(it)
                    wall = float(sum(e.get("wall_s") or 0.0 for e in new.values()))
                    if new:
                        sessions.append(dict(sess, ended_at=s2io.now_utc(), seeds_run=sorted(new), wall_s=wall))
                        count(unit_name, f"{rel(path)}#{sess['id']}", wall)
                    have = {e["seed"]: e for e in (old or {}).get("episodes") or []}
                    have.update({s: doc["episode"] for s, doc in it["reused"].items()})
                    have.update(new)
                    more = dict(extra)
                    if it["supersedes"] is not None:
                        more["supersedes"] = it["supersedes"]
                    if it["rep"] is not None:
                        more["replication_of"] = it["rep"]
                    if old is not None and old.get("attribution") is not None and not new:
                        more["attribution"] = old["attribution"]
                    rec = condition_record(body, set_name, block, row, cond, [have[s] for s in seeds], key, seeds,
                                           prov, inputs, sessions, set_docs, extra=more)
                    s2io.write_json(path, rec)
                    drop_partials(it["pdir"].parent)
                    it["rec"] = rec
                    listed += [dict(set=set_name, body=body, block=block, row=row.name, condition=cond.key, seed=s,
                                    at=s2io.now_utc(), commit=git["commit"], bar_sha256=bsrc["sha256"], out=str(out),
                                    error=bool(new[s].get("error")), fell=bool(new[s].get("fell")),
                                    reason=dev_reason or "the bar's build check: smoke, no record")
                               for s in sorted(new)]
                    c2, c3 = rec["counts"], rec["counts_but_A1"]
                    log(f"{set_name} {body} {block} {row.name} {cond.key}: {c2['survived']}/{c2['episodes']} survive, "
                        f"{c2['succeeded']} succeed" + ("" if c3 is None else f", {c3['succeeded']} but for A1")
                        + (f", {c2['errors']} failed to run" if c2["errors"] else "")
                        + ("" if c2["pass_"] is None else f" ({'pass' if c2['pass_'] else 'FAIL'})")
                        + ("" if row.gated(cond) else " [reported]"))
                if dev_episodes is not None:
                    _list_dev_episodes(dev_episodes, listed)
                # -- decision 20's attribution, record by record (each a unit of the budget)
                for it in batch:
                    rec = it["rec"]
                    if rec is None or not _attribution_needed(rec, attribute):
                        continue
                    row, path = it["row"], it["path"]
                    falls = rec["falls"]
                    if is_bar(cond) and cond.judge != "bar_walk" and falls["after_command"] < panels.ATTRIBUTION_MIN_FALLS:
                        cite = walk_record_of(cond)
                        wc = next((c for c in all_conds if c.key == cite), None)
                        rec["attribution"] = dict(
                            label=f"lead-in falls: see {cite}", cites=cite,
                            record=None if wc is None else rel(record_path(out, set_name, body, block, row.name, wc,
                                                                            control)),
                            falls=rec["counts"]["falls"], falls_after_command=falls["after_command"],
                            falls_before_command=falls["before_command"], min_falls=panels.ATTRIBUTION_MIN_FALLS,
                            rule=RULES["attribution"], episodes=0)
                    else:
                        cands, lift = panels.attribution_plan(rec)
                        a_unit = f"attribution {rel(path)}"
                        if lift and budget is not None:
                            est = len(lift) * len(seeds) * float(rec["seconds"]) * rate_of(rates, row.name) / 3600.0
                            if not budget.start(a_unit, est, dict(lifted=lift, record=rel(path))):
                                rec["attribution"] = dict(label=BUDGET_CUT_LABEL, cut=True, candidates=cands,
                                                          reported_as=BUDGET_CUT_REPORT, not_lifted=lift,
                                                          reason=budget.cut_reason(a_unit),
                                                          rule=RULES["attribution"], episodes=0)
                                rec["failure"] = failure(row.name, rec["episodes"], rec["attribution"],
                                                         (rec["counts"].get("pass_if_read", rec["counts"]["pass_"])))
                                s2io.write_json(path, rec)
                                session["attribution_cut"].append(f"{row.name} {cond.key}")
                                log(f"{set_name} {body} {block} {row.name} {cond.key}: {BUDGET_CUT_LABEL}")
                                continue
                        lifted, t0 = {}, time.perf_counter()
                        for r in lift:
                            ldir = it["pdir"] / f"lift-{r}"
                            kept = read_partials(ldir, it["kh"], seeds)
                            run = [s for s in seeds if s not in kept]
                            ajobs = [((row.name, cond, (r,), str(ldir), it["kh"]), s) for s in run]
                            got = dict(zip(run, get_pool().map(ajobs, failed=_lost) if ajobs else [], strict=True))
                            wall_new = float(sum(e.get("wall_s") or 0.0 for e in got.values()))
                            by = {}
                            for s, doc in kept.items():
                                sid = (doc.get("session") or {}).get("id")
                                by[sid] = by.get(sid, 0.0) + float(doc["episode"].get("wall_s") or 0.0)
                            for sid, wall in by.items():
                                count(a_unit, f"{rel(path)}#lift-{r}#{sid}", wall)
                            if got:
                                count(a_unit, f"{rel(path)}#lift-{r}#{sess['id']}", wall_new)
                            session["attribution_episodes_run"] += len(got)
                            lifted[r] = [got[s] if s in got else kept[s]["episode"] for s in seeds]
                        rec["attribution"] = dict(panels.attribution_record(rec, cands, lifted,
                                                                            time.perf_counter() - t0),
                                                  rule_bar=RULES["attribution"])
                    pass2 = rec["counts"].get("pass_if_read", rec["counts"]["pass_"])
                    rec["failure"] = failure(row.name, rec["episodes"], rec["attribution"], pass2, "success")
                    if rec.get("counts_but_A1") is not None:
                        rec["failure_but_A1"] = failure(row.name, rec["episodes"], rec["attribution"],
                                                        rec["counts_but_A1"]["pass_"], "but_A1")
                    s2io.write_json(path, _plain_json(rec))
                    drop_partials(it["pdir"].parent)
                    log(f"{set_name} {body} {block} {row.name} {cond.key}: attribution {rec['attribution']['label']}")
            if pool is not None:
                session["incidents"] = list(pool.incidents)
        finally:
            if pool is not None:
                pool.close()
        session.update(ended_at=s2io.now_utc(), wall_s=time.perf_counter() - t_start)
        doc = summarize(out, set_name, body, block, control, session=session)
    return doc


# ------------------------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------------------------
def _cell(rec: dict) -> dict:
    def part(c):
        if c is None:
            return None
        return {k: c.get(k) for k in ("episodes", "survived", "succeeded", "falls", "errors", "survival_lcb",
                                      "success_lcb", "judged", "pass_", "not_read")}

    f2, f3 = rec.get("failure") or {}, rec.get("failure_but_A1") or {}
    return dict(status=rec.get("status"), gated=rec.get("gated"), reason=rec.get("reason"),
                counts=part(rec.get("counts")), counts_but_A1=part(rec.get("counts_but_A1")), counts_look=None,
                a1_over=rec.get("a1_over"), falls=rec.get("falls"), label=f2.get("label"), label_but_A1=f3.get("label"),
                attribution=(rec.get("attribution") or {}).get("label"), replication_of=rec.get("replication_of"),
                supersedes=rec.get("supersedes"), control=rec.get("control"), key_sha256=rec.get("key_sha256"))


def verdict(cells: list, ms: int | None, which: str = "success") -> dict:
    """A controller's cell of one condition, delay and reading from its two seeds' records (``RULES["pass_"]``): at
    the judged delay ``shown`` if both pass and ``provisional`` if one does (reading 3: ``... but for A1``), else
    Stage 1's failing words (``panels.VERDICT_RULE``); at another delay ``passes at <ms> ms`` and never either
    word; ``not read: a1_over`` in reading 2 of a turn frozen with ``a1_over``."""
    ckey, lkey = ("counts", "label") if which == "success" else ("counts_but_A1", "label_but_A1")
    tail = "" if which == "success" else " but for A1"
    if any(c is not None and c.get("status") == "not run" for c in cells):
        return dict(verdict="not run", reading=which)
    if any(c is None for c in cells):
        return dict(verdict="incomplete", reading=which)
    cs = [c.get(ckey) or {} for c in cells]
    if any(c.get("not_read") for c in cs):
        return dict(verdict=f"not read: {cs[0].get('not_read') or cs[-1].get('not_read')}", reading=which)
    if any(not c.get("judged") for c in cs):
        return dict(verdict="incomplete", reading=which, errors=[int(c.get("errors") or 0) for c in cs])
    passes = [bool(c["pass_"]) for c in cs]
    if ms != JUDGED_MS:
        word = (f"passes at {ms} ms" if all(passes) else f"one of two passes at {ms} ms" if any(passes)
                else f"does not pass at {ms} ms")
        return dict(verdict=word + tail, reading=which, passes=passes)
    if all(passes):
        v = "shown" + tail
    elif any(passes):
        v = "provisional" + tail
    else:
        labels = [c.get(lkey) for c in cells]
        if any(x is None or x.startswith("not attributed") for x in labels):
            v = "not attributed"
        elif all(x.startswith("body-limited") for x in labels):
            v = "body-limited"
        else:
            v = "controller-limited"
    return dict(verdict=v, reading=which, passes=passes)


def summarize(out, set_name: str, body: str, block: str, control: str | None = None,
              session: dict | None = None) -> dict:
    """``<out>/<set>/<body>/<block>/eval/summary.json``: per condition the rows' cells in both readings and, per
    controller with a verdict (``panels.VERDICT_ROWS``) and delay, its cell in each reading; the sessions (each
    ``evaluate`` call appends its own).  The two blocks have a summary each and are never pooled."""
    edir = block_dir(out, set_name, body, block, control) / "eval"
    path = edir / "summary.json"
    old = _read(path) or {}
    grid: dict = {}
    for q in sorted(edir.glob("**/*.json")) if edir.is_dir() else []:
        parts = q.relative_to(edir).parts
        if parts[0] == "traces" or q == path or any(x.endswith(".superseded") for x in parts):
            continue
        rec = _read(q)
        if rec and rec.get("schema") == SCHEMA and "key_name" in rec:
            grid.setdefault(rec["key_name"], {})[rec["row"]] = _cell(rec)
    verdicts: dict = {}
    no_verdict = ("the replication block gives no verdict of its own" if block != HEADLINE_BLOCK else
                  "a control gives no verdict" if control is not None else None)
    for ckey, cells in ({} if no_verdict else grid).items():
        for ctrl, pair in panels.VERDICT_ROWS.items():
            for ms in sorted({delay_ms(r) for r in cells if controller_of(r) == ctrl and delay_ms(r) is not None}):
                two = [cells.get(f"{r}@{ms}ms") for r in pair]
                verdicts.setdefault(ckey, {})[f"{ctrl}@{ms}ms"] = dict(
                    rows=[f"{r}@{ms}ms" for r in pair], judged_delay=ms == JUDGED_MS, block=block,
                    reading_1=dict(verdict=None, reason=LOOK_NOT_DEFINED), reading_2=verdict(two, ms, "success"),
                    reading_3=verdict(two, ms, "but_A1"))
    doc = dict(schema=SCHEMA, set=set_name, body=body, block=block, control=control,
               rules={k: RULES[k] for k in ("readings", "pass_", "replication", "seeds")}, grid=grid,
               verdicts=verdicts, verdicts_note=no_verdict,
               sessions=list(old.get("sessions") or []) + ([session] if session else []),
               written_at=s2io.now_utc())
    if session is not None or grid:
        s2io.write_json(path, doc)
    return doc


def summary_text(doc: dict) -> str:
    """The block's grid as text: one line per condition and row with both readings' counts."""
    lines = [f"{doc['set']} {doc['body']} {doc['block']}" + (f" (control {doc['control']})" if doc.get("control") else "")]
    for ckey, cells in doc["grid"].items():
        for r, x in cells.items():
            if x["status"] != "done":
                lines.append(f"  {ckey:28s} {r:14s} {x['status']}: {x.get('reason')}")
                continue
            c2, c3 = x["counts"], x["counts_but_A1"]

            def word(c):
                if c is None:
                    return ""
                if c.get("not_read"):
                    return f" (not read: {c['not_read']})"
                return "" if c["pass_"] is None else (" pass" if c["pass_"] else " FAIL")

            lines.append(f"  {ckey:28s} {r:14s} {c2['survived']}/{c2['episodes']} survive; every limit: "
                         f"{c2['succeeded']} succeed{word(c2)}"
                         + ("" if c3 is None else f"; but for A1: {c3['succeeded']} succeed{word(c3)}")
                         + (f"; {c2['errors']} failed to run" if c2.get("errors") else "")
                         + ("" if x.get("gated", True) else " [reported]"))
        for name, v in (doc["verdicts"].get(ckey) or {}).items():
            lines.append(f"  {ckey:28s} {name}: {v['reading_2']['verdict']}; {v['reading_3']['verdict']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------------------------------------
def measure_rates(body: str, row_names, stage1_run=STAGE1_RUN, seconds: float = COST_SECONDS, seeds=COST_SEEDS) -> dict:
    """Wall seconds per simulated second of each row type (controller) among ``row_names``: two short episodes of
    its first row on Stage 1's own committed condition ``COST_CONDITION`` (the ``S1`` view, cut to ``seconds``),
    run and judged as the runner does, the trace written to a temporary directory.  Never on a reference of a
    Stage 2 set."""
    run_dir = setio._anchored(stage1_run) / body
    _, docs = setio.stage1_view(stage1_run, body)
    ctx = BarContext(body, run_dir, docs)
    cond = dataclasses.replace(next(c for c in panels.conditions() if c.key == COST_CONDITION), seconds=float(seconds))
    first = {}
    for name in row_names:
        first.setdefault(controller_of(name), name)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for ctrl, name in first.items():
            row = rows(body, [name], stage1_run)[name]
            eps = [run_episode(row, ctx, cond, s, trace_root=Path(tmp)) for s in seeds]
            wall, sim = sum(e["wall_s"] for e in eps), sum(e["survival_s"] for e in eps)
            out[ctrl] = dict(wall_per_sim_s=wall / max(sim, 1e-9), row=name, wall_s=wall, simulated_s=sim,
                             fell=[bool(e["fell"]) for e in eps])
    return dict(schema=COST_SCHEMA, rates=out,
                measured_on=dict(set=STOCK_SET, stage1_run=s2io.portable(str(setio._anchored(stage1_run))), body=body,
                                 condition=COST_CONDITION, cut_s=float(seconds), seeds=[int(s) for s in seeds],
                                 note="Stage 1's own committed condition, never a reference of a Stage 2 set"),
                load_average=list(os.getloadavg()), cpu_count=os.cpu_count(), at=s2io.now_utc())


def dry_run(set_name: str, body: str, out, rows, conditions=None, block: str = HEADLINE_BLOCK, episodes: int = RUNS,
            *, stage1_run=STAGE1_RUN, root=setio.ROOT, rates: dict | None = None, measure: bool = True,
            control: str | None = None, log=print) -> dict:
    """Every episode a run would make, its planned seconds and its cost from rates measured now (``measure_rates``)
    or given (``rates``: a ``read_cost`` document).  Writes nothing; ``text`` is the document ``COST.md`` holds (it
    is logged: ``bar dry-run ... > COST.md``)."""
    s2io.require_one_thread()
    seeds = BLOCKS[block][: int(episodes)]
    row_names = list(rows)
    if set_name == STOCK_SET:
        run_dir = setio._anchored(stage1_run) / body
        _, docs = setio.stage1_view(stage1_run, body)
        planner = panels.Planner(run_dir, docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
        conds = select(stock_conditions(), conditions, stock_default())
        plan = {c: (planner.resolve(c), None) for c in conds}
    else:
        run_dir = setio.set_dir(set_name, body, root)
        set_docs = setio.load(run_dir)
        planner = BarPlanner(run_dir, set_docs["refs.json"]["references"], set_docs["transitions.json"]["transitions"],
                             set_docs["set.json"])
        conds = choose(set_docs, block, conditions)
        plan = {}
        for c in conds:
            why = planner.missing(c)
            plan[c] = (None, why) if why is not None else (planner.resolve(c), None)
    cost = rates if rates is not None else (measure_rates(body, row_names, stage1_run) if measure else None)
    lines, totals = [], Counter()
    for c in conds:
        res, why = plan[c]
        for r in row_names:
            parse_row(r)
            old = _read(record_path(out, set_name, body, block, r, c, control))
            if why is not None:
                status, n = f"not run: {why['reason']}", 0
            elif old and old.get("status") == "done" and old.get("seeds") == list(seeds):
                status, n = "recorded", len(seeds)
            else:
                status, n = "to run", len(seeds)
            sec = None if res is None else float(res["seconds"])
            rate = None if cost is None or sec is None else rate_of(cost["rates"], r)
            core_h = None if rate is None else n * sec * rate / 3600.0
            lines.append(dict(row=r, condition=c.key, gated=bool(c.gated), seconds=sec, episodes=n, status=status,
                              simulated_s=None if sec is None else n * sec, core_h=core_h))
            totals["episodes"] += n
            totals["simulated_s"] += 0.0 if sec is None else n * sec
            totals["core_h"] += core_h or 0.0
            if status == "to run":
                totals["core_h_to_run"] += core_h or 0.0
    text = ["# Cost of the bar's episodes", "",
            f"Set `{set_name}`, body `{body}`, block `{block}`, {len(seeds)} episodes per record; made "
            f"{s2io.now_utc()} by `designlab.stage2.bar dry-run` (it writes nothing; a smoke estimate, not a result).",
            ""]
    if cost is not None:
        mo = cost.get("measured_on") or {}
        text += ["Rates (wall seconds per simulated second, one process), measured on Stage 1's own committed "
                 f"condition `{mo.get('condition')}` of `{mo.get('body')}` (`{mo.get('stage1_run')}`), cut to "
                 f"{mo.get('cut_s')} s, seeds {mo.get('seeds')}; never on a reference of a Stage 2 set:", "",
                 "| Row type | Row measured | Wall s per simulated s |", "|---|---|---|"]
        text += [f"| {k} | `{v['row']}` | {v['wall_per_sim_s']:.4f} |" for k, v in cost["rates"].items()]
        text += ["", "```json", json.dumps(s2io.plain(cost), indent=1), "```", ""]
    text += ["| Row | Condition | Episodes | Planned s each | Core-hours | Status |", "|---|---|---|---|---|---|"]
    for x in lines:
        text.append(f"| `{x['row']}` | `{x['condition']}`{'' if x['gated'] else ' (reported)'} | {x['episodes']} | "
                    + ("-" if x["seconds"] is None else f"{x['seconds']:.2f}") + " | "
                    + ("-" if x["core_h"] is None else f"{x['core_h']:.4f}") + f" | {x['status']} |")
    text += ["", f"Total: {totals['episodes']} episodes, {totals['simulated_s'] / 3600.0:.3f} simulated hours, "
                 f"{totals['core_h']:.3f} core-hours ({totals['core_h_to_run']:.3f} to run).", ""]
    doc = dict(set=set_name, body=body, block=block, seeds=list(seeds), cost=cost, episodes=lines, totals=dict(totals),
               text="\n".join(text))
    log(doc["text"])
    return doc


# ------------------------------------------------------------------------------------------------
# References played kinematically (G9: each reference passes its own bar)
# ------------------------------------------------------------------------------------------------
def _xy_yaw(q) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in q[3:7])
    return float(q[0]), float(q[1]), math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _move(T, q) -> np.ndarray:
    """The full pose ``q`` moved in the plane by ``T = (yaw, tx, ty)``: rotated about the vertical, then shifted."""
    yaw, tx, ty = T
    out = np.array(q, float)
    c, s = math.cos(yaw), math.sin(yaw)
    out[0], out[1] = c * q[0] - s * q[1] + tx, s * q[0] + c * q[1] + ty
    hw, hz = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    w, x, y, z = q[3:7]
    out[3:7] = (hw * w - hz * z, hw * x - hz * y, hw * y + hz * x, hw * z + hz * w)
    return out


def _onto(pose, first) -> tuple[float, float, float]:
    """The planar move that takes a pose ``first = (x, y, yaw)`` onto ``pose``."""
    yaw = pose[2] - first[2]
    c, s = math.cos(yaw), math.sin(yaw)
    return yaw, pose[0] - (c * first[0] - s * first[1]), pose[1] - (s * first[0] + c * first[1])


class _Played:
    """A track's segments with their files' full poses placed in one world frame: each segment begins where the
    one before it ends (a walk's stride carried along its own heading; a move's last frame carried on by one frame
    interval, as the track joins it to the next segment's first frame)."""

    def __init__(self, track: control.Track, run_dir):
        self.track, run_dir = track, Path(run_dir)
        self.objs = []
        for seg in track.segments:
            if seg.kind == "periodic":
                ref = Reference.load(run_dir / "refs" / f"{seg.name}.npz")
                sm = np.asarray(ref.stride_motion, float)
                if sm[0] != 0.0 or sm[1] != 0.0 or sm[2] != 0.0:
                    raise ValueError(f"{seg.name}: a periodic segment must be a straight walk (stride motion "
                                     f"{sm.tolist()})")
                self.objs.append(ref)
            elif seg.kind == "finite":
                self.objs.append(Transition.load(run_dir / "transitions" / f"{seg.name}.npz"))
            else:
                self.objs.append(None)
        self.T: list = [None] * len(self.objs)
        self.held: list = [None] * len(self.objs)
        pose = None
        for i, (seg, obj) in enumerate(zip(track.segments, self.objs, strict=True)):
            if obj is None:  # a hold: the pose before it (or, first, the next segment's first frame)
                if i == 0:
                    self.held[i] = np.array(self.objs[1].q[0], float)
                    pose = _xy_yaw(self.held[i])
                else:
                    self.held[i] = self._file_end_frame(i - 1)
                self.T[i] = (0.0, 0.0, 0.0)
                continue
            first = _xy_yaw(obj.q[0])
            self.T[i] = (0.0, 0.0, 0.0) if pose is None else _onto(pose, first)
            pose = _xy_yaw(_move(self.T[i], self._end_pose(i)))

    def _periodic(self, i: int, k: int) -> np.ndarray:
        ref = self.objs[i]
        q = np.array(ref.q[k % ref.n], float)
        sm = ref.stride_motion
        q[0] += (k // ref.n) * float(sm[3])
        q[1] += (k // ref.n) * float(sm[4])
        return q

    def _end_pose(self, i: int) -> np.ndarray:
        """The pose at the end of segment i, in its file's frame."""
        seg, obj = self.track.segments[i], self.objs[i]
        if seg.kind == "periodic":
            return self._file_q(i, seg.duration)
        q = np.array(obj.q[-1], float)
        ratio = (seg.duration - float(obj.t[-1])) / float(obj.t[-1] - obj.t[-2])
        x0, y0, a0 = _xy_yaw(obj.q[-2])
        x1, y1, a1 = _xy_yaw(obj.q[-1])
        return _move(_onto((x1 + ratio * (x1 - x0), y1 + ratio * (y1 - y0), a1 + ratio * control._wrap(a1 - a0)),
                           (x1, y1, a1)), q)

    def _file_end_frame(self, i: int) -> np.ndarray:
        """The world pose a hold after segment i holds: that segment's last frame."""
        return _move(self.T[i], self.objs[i].q[-1]) if self.objs[i] is not None else self.held[i]

    def _file_q(self, i: int, tl: float) -> np.ndarray:
        seg, obj = self.track.segments[i], self.objs[i]
        if seg.kind == "periodic":
            s = tl / seg.step
            k0 = int(math.floor(s + 1e-12))
            a = min(max(s - k0, 0.0), 1.0)
            return (1.0 - a) * self._periodic(i, k0) + a * self._periodic(i, k0 + 1)
        t = obj.t
        k0 = min(max(int(np.searchsorted(t, tl, side="right")) - 1, 0), len(t) - 1)
        if k0 >= len(t) - 1:
            return np.array(obj.q[-1], float)
        a = min(max((tl - t[k0]) / (t[k0 + 1] - t[k0]), 0.0), 1.0)
        return (1.0 - a) * np.asarray(obj.q[k0], float) + a * np.asarray(obj.q[k0 + 1], float)

    def index(self, t_ref: float) -> int:
        segs = self.track.segments
        i = 0
        while i + 1 < len(segs) and t_ref >= segs[i + 1].t0:
            i += 1
        return i

    def q(self, t_ref: float) -> np.ndarray:
        """The world pose at track time ``t_ref`` (the last pose past the track's end)."""
        i = self.index(t_ref)
        seg, obj = self.track.segments[i], self.objs[i]
        tl = min(max(t_ref - seg.t0, 0.0), seg.duration)
        if obj is None:
            q = np.array(self.held[i], float)
        elif seg.kind == "finite" and tl > float(obj.t[-1]):  # the last interval: on to the next segment's first frame
            a = min((tl - float(obj.t[-1])) / (seg.duration - float(obj.t[-1])), 1.0)
            last = _move(self.T[i], obj.q[-1])
            if i + 1 < len(self.objs):
                nxt = (_move(self.T[i + 1], self.objs[i + 1].q[0]) if self.objs[i + 1] is not None
                       else self.held[i + 1])
            else:
                nxt = _move(self.T[i], self._end_pose(i))
            q = (1.0 - a) * last + a * nxt
        else:
            q = _move(self.T[i], self._file_q(i, tl))
        q[3:7] /= np.linalg.norm(q[3:7])
        return q

    def loads(self, t_ref: float, weight: float) -> np.ndarray:
        """Each foot's planned vertical load (N) at ``t_ref``: the file's ``normal`` at the frame at or before it
        where its ``support`` flag is set (never under the contact threshold there), else 0; a hold shares the
        weight."""
        i = self.index(t_ref)
        seg, obj = self.track.segments[i], self.objs[i]
        if obj is None:
            return np.array([0.5 * weight, 0.5 * weight])
        tl = min(max(t_ref - seg.t0, 0.0), seg.duration)
        if seg.kind == "periodic":
            k = int(math.floor(tl / seg.step + 1e-12)) % obj.n
        else:
            k = min(max(int(np.searchsorted(obj.t, tl, side="right")) - 1, 0), len(obj.t) - 1)
        sup = np.asarray(obj.support[k], bool)
        floor = 1.001 * control.CONTACT_LOAD_FRAC * weight
        return np.where(sup, np.maximum(np.asarray(obj.normal[k], float), floor), 0.0)


def reference_recorder(p: plant.Plant, track: control.Track, seconds: float, run_dir=None,
                       rate: float = 1.0) -> metrics.Recorder:
    """A ``metrics.Recorder`` filled by playing any track kinematically on ``p``'s model for ``seconds`` of
    simulated time with the reference clock at ``rate`` (no physics): the frames of the track's own files
    interpolated in time and placed in one world frame, the CoM, feet, soles and lowest points from their
    kinematics at 100 Hz, each foot's load from the reference's ``normal`` under its ``support`` flags at every
    physics step, the CoM's velocity from its positions.  ``run_dir`` holds the files (default: the directory the
    track was built from, ``track.run_dir``).  ``panels.reference_recorder`` is the same for one straight walk."""
    run_dir = getattr(track, "run_dir", None) if run_dir is None else run_dir
    if run_dir is None:
        raise ValueError("reference_recorder needs the directory of the track's files (run_dir)")
    play = _Played(track, run_dir)
    m = p.m
    d = mujoco.MjData(m)
    dt = float(m.opt.timestep)
    n = int(round(seconds / dt))
    rec = metrics.Recorder()
    rec.tk, rec.n, rec.dt = None, n, dt
    rec.weight = float(m.body_subtreemass[p.trunk]) * control.G
    rec.force_sq = np.zeros((n, m.nu))
    rec.sustained = np.array([dr.sustained for dr in p.drives], float)
    rec.actuators = [dr.actuator for dr in p.drives]
    t_last = track.t_end - 1e-9
    rec.loads = np.array([play.loads(min(rate * k * dt, t_last), rec.weight) for k in range(n)]).reshape(n, 2)
    boxes, caps = metrics._lowest_point_tables(m, p.feet)
    sole = [c.mean(0) for c in estimator.sole_corners(m, p.feet)]
    rows_, Q = [], []
    for k in range(0, n, control.FEEDBACK_EVERY):
        t = k * dt
        t_ref = min(rate * t, t_last)
        q = play.q(t_ref)
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        com = d.subtree_com[p.trunk]
        f0, f1 = d.xpos[p.feet[0]], d.xpos[p.feet[1]]
        low = metrics.foot_lowest(m, d, boxes, caps) - p.z_floor
        R = d.xmat[p.trunk].reshape(3, 3)
        roll, pitch = control._roll_pitch(R @ p._R_home.T)
        _, info = track.info(t_ref)
        s0 = f0 + d.xmat[p.feet[0]].reshape(3, 3) @ sole[0]
        s1 = f1 + d.xmat[p.feet[1]].reshape(3, 3) @ sole[1]
        ld = rec.loads[k]
        rows_.append((t, rate * t, com[0], com[1], 0.0, 0.0, f0[0], f0[1], f1[0], f1[1], low[0], low[1], roll, pitch,
                      control._yaw(R), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, info["swing"], info["u"], s0[0], s0[1], s1[0],
                      s1[1], ld[0], ld[1], q[0], q[1]))
        Q.append(q[p.qadr])
    rec.U, rec.Q = np.array(rows_, float).reshape(-1, len(metrics.COLS)), np.array(Q, float)
    if len(rec.U) > 2:
        rec.U[:, metrics.C["v_x"]] = np.gradient(rec.U[:, metrics.C["com_x"]], rec.U[:, metrics.C["t"]])
        rec.U[:, metrics.C["v_y"]] = np.gradient(rec.U[:, metrics.C["com_y"]], rec.U[:, metrics.C["t"]])
    rec.rows, rec.steps, rec.poses = len(rows_), n, None
    return rec


def reference_item(p: plant.Plant, run_dir, cond: BarCondition, planner: BarPlanner, omega: float) -> dict:
    """One condition's track played kinematically at the commanded clock and read with the bar's readings: what
    the reference itself does of its own item (G9)."""
    res = planner.resolve(cond)
    tr_doc = planner.tr_doc
    track = setio.build_track(p, run_dir, res["plan"], omega, tr_doc)
    rec = reference_recorder(p, track, res["seconds"], run_dir, rate=RATE)
    H = float(p.meta["H_m"])
    psi = float(rec.col("yaw")[0])
    gap = readings.air_gap_max(setio.min_planned_swing_s(track), RATE)
    out = dict(condition=cond.key, plan=[list(x) for x in res["plan"]], seconds=res["seconds"], clock_rate=RATE,
               files=res["files"])

    def toe(ta, tb):
        r = readings.toe_clearance_own_swing(rec.t, _lows(rec), rec.contact(), rec.dt, rec.col("ref_swing"), ta, tb, gap)
        r.pop("swings")
        v = r["toe_clearance_own_swing_m"]
        return dict(r, ok=readings.toe_clearance_own_swing_ok(r, H), limit_m=readings.TOE_CLEARANCE_FRAC_H * H,
                    toe_clearance_own_swing_H=None if v is None else v / H)

    j = cond.judge
    if j == "bar_walk":
        tb = float(rec.t[-1])
        ta = tb - tune.WALK_WINDOW_S
        line = readings.line_reading(rec.t, _com(rec), psi, ta, tb, c_episode_start=_com(rec)[0])
        target = tune.WALK_WINDOW_S * float(res["target_speed_m_s"])
        t = toe(ta, tb)
        ok = dict(along=line["along_commanded_heading_m"] >= target, line=line["lateral_from_line_max_m"] <= readings.LINE_MAX_M,
                  toe_clearance_own_swing=t["ok"])
        out.update(along_commanded_heading_m=line["along_commanded_heading_m"], along_target_m=target,
                   lateral_from_line_max_m=line["lateral_from_line_max_m"], toe=t)
    elif j == "bar_turn":
        t_c = rec.sim_time(res["command_ref_s"])
        tr = readings.turn_reading(rec.t, rec.yaw_unwrapped(), t_c, psi, res["sign"], res["angle_rad"], res["at_s"])
        t = toe(t_c, t_c + res["at_s"] + readings.HEADING_HOLD_S)
        with np.load(Path(run_dir) / "transitions" / f"{cond.move}.npz", allow_pickle=False) as z:
            own = readings.reference_readings(dict(t=z["t"], com=z["com"], heading=z["heading"], support=z["support"]),
                                              "turn", dict(at_s=res["at_s"]))
        ok = dict(heading=tr["heading"], toe_clearance_own_swing=t["ok"], heading_at_80deg=own["heading_ok"])
        out.update(command_s=t_c, turn=tr, toe=t, file=own,
                   heading_reached_at_bar_time_deg=tr["angle_deg"] + res["sign"] * (tr["heading_error_at_deg"] or 0.0)
                   if tr["heading_error_at_deg"] is not None else None)
    elif j == "bar_stop":
        t_c = rec.sim_time(res["command_ref_s"])
        T, v = float(res["period_s"]), float(res["target_speed_m_s"])
        sr = readings.stop_reading(rec.t, _com(rec), rec.U[:, [metrics.C["v_x"], metrics.C["v_y"]]], rec.contact(), rec.dt,
                                   t_c, psi, v * RATE, T / RATE)
        n = sr["touchdowns_after_command"]
        ok = dict(stopped_by_end=bool(sr["stopped_by_end"]),
                  touchdowns_after_command=n is not None and n <= readings.STOP_TOUCHDOWNS_MAX)
        out.update(command_s=t_c, command_convention=res["command"]["convention"], stop=sr)
    else:
        raise ValueError(f"{cond.key}: a {j} item has no reference of its own")
    return dict(out, ok={k: bool(v) for k, v in ok.items()}, pass_=bool(all(ok.values())))


def refcheck(set_name: str, body: str, root=setio.ROOT, out=None, log=print) -> dict:
    """Every reference of a set through ``reference_recorder`` and the bar's readings (G9: each reference passes
    its own bar): the walk, turn and stop conditions the set has references for, at the commanded clock, mu 0.9.
    Nothing is simulated.  With ``out`` the record goes to ``<out>/<set>/<body>/refcheck.json``."""
    s2io.require_one_thread()
    run_dir = setio.set_dir(set_name, body, root)
    set_docs = setio.load(run_dir)
    planner = BarPlanner(run_dir, set_docs["refs.json"]["references"], set_docs["transitions.json"]["transitions"],
                         set_docs["set.json"])
    conds = [c for c in conditions(set_docs) if c.friction is None and c.judge != "bar_nudge"]
    usable = [c for c in conds if planner.missing(c) is None]
    files = sorted({f for c in usable for f in planner.resolve(c)["files"]})
    inputs, _ = setio.read_inputs(run_dir, files, require_frozen=False)
    crouch = set_crouch(set_docs["set.json"])
    p = plant.build(body, "hard", settle=False)
    omega = float(control.omega0(p, Reference.load(run_dir / "refs" / f"{setio.NOMINAL}.npz").q_home,
                                 crouch_frac=crouch)) if usable else None
    items = {}
    for c in conds:
        why = planner.missing(c)
        if why is not None:
            items[c.key] = dict(condition=c.key, status="not run", reason=why["reason"], reference=why["reference"],
                                pass_=None)
            log(f"{set_name} {body} {c.key}: not run ({why['reason']})")
            continue
        r = reference_item(p, run_dir, c, planner, omega)
        items[c.key] = dict(r, status="done", gated=bool(c.gated))
        log(f"{set_name} {body} {c.key}: {'pass' if r['pass_'] else 'FAIL'} "
            + ", ".join(f"{k} {'ok' if v else 'FAILS'}" for k, v in r["ok"].items()) + _refcheck_numbers(r))
    done = [x for x in items.values() if x["status"] == "done"]
    doc = dict(schema="bar-refcheck-1", set=set_name, body=body, rule="each reference, played kinematically at the "
               f"commanded clock ({RATE:g}), read with the bar's own readings; no physics, no verdict",
               omega=omega, crouch_frac=crouch, inputs=dict(dir=s2io.portable(inputs["dir"]), files=inputs["files"]),
               set_freeze=inputs.get("freeze"), items=items, pass_=bool(done and all(x["pass_"] for x in done)),
               bar_sources=bar_sources(), provenance=dict(git=s2io.git_state(), versions=panels.library_versions(),
                                                           blas_threads=s2io.blas_threads()), at=s2io.now_utc())
    if out is not None:
        s2io.write_json(Path(out) / set_name / body / "refcheck.json", doc)
    return doc


def _refcheck_numbers(r: dict) -> str:
    if "stop" in r:
        s = r["stop"]["forward_stopped_s"]
        return (f" ({r['stop']['touchdowns_after_command']} touchdowns after the command, stopped "
                + ("never" if s is None else f"{s:.3f} s after it") + ")")
    t = r.get("toe") or {}
    toe = "" if t.get("toe_clearance_own_swing_H") is None else f", clearance {t['toe_clearance_own_swing_H']:.4f} H"
    if "turn" in r:
        return f" ({r['heading_reached_at_bar_time_deg']:.2f} deg at {r['turn']['at_s']:g} s{toe})"
    return f" ({r['along_commanded_heading_m']:.3f} m in 10 s, target {r['along_target_m']:g} m{toe})"


# ------------------------------------------------------------------------------------------------
# Stock episodes re-run with the own-swing reading (G9's toe-clearance evidence)
# ------------------------------------------------------------------------------------------------
def reread(stage1_run, body: str, rows, conditions, out, records=None, row_modules=(), episodes: int = RUNS,
           log=print) -> dict:
    """Stage 1's stock rows (``rows``: names ``panels.tracker_rows`` knows after importing ``row_modules``) on
    stock conditions read from ``stage1_run``, one process: each episode is judged by ``panels.judge`` and its
    stock trace written by ``panels.write_trace``; the trace's sha256 is compared with the committed episode
    record's under ``records`` (default ``stage1_run``) and only then is the own-swing toe clearance read on the
    live recorder, over the window Stage 1 judged.  The record ``<out>/<body>/reread.json`` lists, per episode,
    both clearance readings, their difference, interrupted swings and air gaps.  Nothing under ``records`` is
    written."""
    s2io.require_one_thread()
    for mod in row_modules:
        importlib.import_module(mod)
    run_dir = setio._anchored(stage1_run) / body
    rec_root = setio._anchored(records if records is not None else stage1_run)
    out = Path(out)
    inputs, docs = setio.stage1_view(stage1_run, body)
    ctx = BarContext(body, run_dir, docs)
    conds = select(stock_conditions(), conditions, stock_default())
    row_objs = panels.tracker_rows(body, run_dir, list(rows))
    seeds = BLOCKS[HEADLINE_BLOCK][: int(episodes)]
    path = out / body / "reread.json"
    eps, t_start = [], time.perf_counter()
    doc = {}
    with s2io.Lock(out / ".lock"):
        for row in row_objs.values():
            for cond in conds:
                committed = _read(panels.record_path(rec_root, body, row.name, cond)) or {}
                by_seed = {e["seed"]: e for e in committed.get("episodes") or []}
                for seed in seeds:
                    t0 = time.perf_counter()
                    b, o, rec = row.episode(ctx, cond, int(seed), ())
                    e = panels.judge(b, o, rec, int(seed), ctx.lg if b.legs else None)
                    tr = panels.write_trace(out / body / "traces", panels.trace_path(row.name, cond, int(seed)), rec)
                    want = ((by_seed.get(seed) or {}).get("trace") or {}).get("sha256")
                    line = dict(row=row.name, condition=cond.key, seed=int(seed), trace_sha256=tr["sha256"],
                                committed_trace_sha256=want, trace_equal=bool(want is not None and want == tr["sha256"]),
                                fell=bool(e["fell"]), succeeded=bool(e["succeeded"]),
                                committed_succeeded=(by_seed.get(seed) or {}).get("succeeded"))
                    w = (e.get("measures") or {}).get("window") or {}
                    if line["trace_equal"] and w.get("complete"):
                        gap = readings.air_gap_max(setio.min_planned_swing_s(b.track), RATE)
                        r = readings.toe_clearance_own_swing(rec.t, _lows(rec), rec.contact(), rec.dt,
                                                             rec.col("ref_swing"), w["t_start_s"], w["t_end_s"], gap)
                        r.pop("swings")
                        own, ref = r["toe_clearance_own_swing_m"], w.get("toe_clearance_m")
                        line.update(window_s=[w["t_start_s"], w["t_end_s"]], H_m=b.H,
                                    toe_clearance_own_swing_m=own, toe_clearance_reference_clock_m=ref,
                                    difference_m=None if own is None or ref is None else own - ref,
                                    own_swing_ok=readings.toe_clearance_own_swing_ok(r, b.H),
                                    reference_clock_ok=bool(ref is not None and ref >= tune.TOE_H * b.H),
                                    own_swings=r["own_swings"], ref_swings=r["ref_swings"],
                                    interrupted_swings=r["interrupted_swings"], air_gaps=r["air_gaps"],
                                    air_gap_longest_s=r["air_gap_longest_s"], air_gap_max_s=r["air_gap_max_s"],
                                    convention=convention_check(rec))
                    else:
                        line["not_read"] = ("the stock trace differs from the committed record's" if not
                                            line["trace_equal"] else "Stage 1 judged no window (the episode fell)"
                                            if e["fell"] else "Stage 1's window is not complete")
                    line["wall_s"] = time.perf_counter() - t0
                    eps.append(line)
                    log(f"{body} {row.name} {cond.key} {seed}: trace "
                        + ("equal" if line["trace_equal"] else "DIFFERS" if want else "has no committed record")
                        + (f"; own swing {1e3 * line['toe_clearance_own_swing_m']:.2f} mm, reference clock "
                           f"{1e3 * line['toe_clearance_reference_clock_m']:.2f} mm"
                           if line.get("toe_clearance_own_swing_m") is not None
                           and line.get("toe_clearance_reference_clock_m") is not None else ""))
                    doc = dict(schema="bar-reread-1", body=body, rule=(
                        "stock rows on stock conditions, re-run; a stock trace's sha256 must equal the committed "
                        "episode record's before the own-swing toe clearance is read on the live recorder, over "
                        "the window Stage 1 judged; only the trace identity is asserted"),
                        stage1_run=s2io.portable(str(setio._anchored(stage1_run))),
                        records=s2io.portable(str(rec_root)), rows=list(row_objs), conditions=[c.key for c in conds],
                        seeds=[int(s) for s in seeds], episodes=eps, traces_equal=sum(x["trace_equal"] for x in eps),
                        traces=len(eps), pass_=bool(all(x["trace_equal"] for x in eps)),
                        complete=len(eps) == len(row_objs) * len(conds) * len(seeds), bar_sources=bar_sources(),
                        panel_sources=panels.panel_sources()["sha256"], inputs=dict(files=inputs["files"]),
                        provenance=dict(git=s2io.git_state(), versions=panels.library_versions(),
                                        blas_threads=s2io.blas_threads()),
                        wall_s=time.perf_counter() - t_start, at=s2io.now_utc())
                s2io.write_json(path, doc)
    return doc


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _csv(text: str | None) -> list[str] | None:
    return None if text is None else [x for x in (s.strip() for s in text.split(",")) if x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.bar", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("run", "dry-run", "summary", "refcheck", "reread"))
    ap.add_argument("--set", dest="set_name", default=None, help="the reference set (V0, F25F9, F25F9cl, ...) or S1")
    ap.add_argument("--body", required=True)
    ap.add_argument("--rows", default=None, help="comma-separated rows: R2/42@10ms,R0@10ms")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated keys, names or groups (" + ", ".join(GROUPS) + "; default: the gated ones)")
    ap.add_argument("--block", default=HEADLINE_BLOCK, choices=tuple(BLOCKS))
    ap.add_argument("--passed-of", default=None, choices=tuple(BLOCKS))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--attribute", default="judged", choices=("judged", "all", "none"))
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--no-traces-for", default=None, help="controllers whose rows write no trace (R0)")
    ap.add_argument("--episodes", type=int, default=RUNS)
    ap.add_argument("--task", type=int, default=None, choices=tuple(BUDGET_CORE_H))
    ap.add_argument("--unit", default=None, help="the unit's name in budget.json")
    ap.add_argument("--cost", default=None, help="the committed COST.md with the rates (default <out>/COST.md)")
    ap.add_argument("--control", default=None, choices=tuple(CONTROLS))
    ap.add_argument("--supersede", action="store_true")
    ap.add_argument("--supersede-reason", default=None)
    ap.add_argument("--dev-episodes", default=None, help="the file a build check's episodes are listed in")
    ap.add_argument("--dev-reason", default=None, help="why the check episodes are run (listed with them)")
    ap.add_argument("--set-root", default=str(setio.ROOT))
    ap.add_argument("--stage1-run", default=str(STAGE1_RUN))
    ap.add_argument("--records", default=None)
    ap.add_argument("--row-module", action="append", default=[])
    ap.add_argument("--no-measure", action="store_true", help="dry-run: list the episodes without measuring rates")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        s2io.require_one_thread()
        for mod in a.row_module if a.command != "reread" else ():
            importlib.import_module(mod)
        need = dict(run=("set_name", "rows"), **{"dry-run": ("set_name", "rows")}, summary=("set_name",),
                    refcheck=("set_name",), reread=("rows",))[a.command]
        for n in need:
            if getattr(a, n) is None:
                ap.error(f"{a.command} needs --{n.replace('_name', '').replace('_', '-')}")
        if a.command == "run":
            rates = None
            if a.task is not None:
                rates = read_cost(a.cost or Path(a.out) / "COST.md")["rates"]
            doc = evaluate(a.set_name, a.body, a.out, _csv(a.rows), _csv(a.conditions), a.block, a.procs, a.attribute,
                           not a.no_traces, a.episodes, a.task, a.supersede, stage1_run=a.stage1_run, root=a.set_root,
                           passed_of=a.passed_of, control=a.control, unit=a.unit, rates=rates,
                           supersede_reason=a.supersede_reason, dev_episodes=a.dev_episodes,
                           no_traces_for=_csv(a.no_traces_for) or (), dev_reason=a.dev_reason)
            print(summary_text(doc))
            s = doc["sessions"][-1]
            print(f"session: {s['episodes_run']} episodes run, {s['attribution_episodes_run']} lifted, "
                  f"{len(s['skipped'])} records up to date, {len(s['not_run'])} not run, {s['wall_s']:.1f} s")
            return 0
        if a.command == "dry-run":
            rates = read_cost(a.cost) if a.cost else None
            dry_run(a.set_name, a.body, a.out, _csv(a.rows), _csv(a.conditions), a.block, a.episodes,
                    stage1_run=a.stage1_run, root=a.set_root, rates=rates, measure=not a.no_measure, control=a.control)
            return 0
        if a.command == "summary":
            print(summary_text(summarize(a.out, a.set_name, a.body, a.block, a.control)))
            return 0
        if a.command == "refcheck":
            doc = refcheck(a.set_name, a.body, a.set_root, a.out)
            print(f"refcheck {a.set_name} {a.body}: {'pass' if doc['pass_'] else 'FAIL'}")
            return 0 if doc["pass_"] else 1
        doc = reread(a.stage1_run, a.body, _csv(a.rows), _csv(a.conditions), a.out, a.records, a.row_module, a.episodes)
        print(f"reread {a.body}: {doc['traces_equal']} of {doc['traces']} stock traces equal the committed records' "
              "sha256")
        return 0 if doc["pass_"] else 1
    except BudgetCut as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (BarError, s2io.S2Error, tune.InputError, tune.PoolError, ValueError) as exc:
        print(f"bar: {exc}", file=sys.stderr)
        return 1


__all__ = ["BarBuilt", "BarCondition", "BarContext", "BarPlanner", "BarWorker", "Budget", "Delayed", "bar_sources",
           "condition_record", "conditions", "counts", "dry_run", "evaluate", "failed_episode", "failure", "judge",
           "not_run_record", "record_key", "refcheck", "reference_recorder", "reread", "rows", "run_episode",
           "summarize"]

if __name__ == "__main__":
    from designlab.stage2 import bar as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
