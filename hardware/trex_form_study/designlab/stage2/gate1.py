"""Gate 1's runner of the served panels: condition 2 of ``BUY_GATE_DRAFT.md`` (approved by the owner,
``DECISION_LOG.md`` P41) on the servo model of ``stage2/design/servo_model/SERVO_MODEL_DESIGN.md`` (section 4; 5.1;
package S3).

**The route** the lead ruled on 16 Nov 2026 (design section 8, choice 18, route (c)): a runner of its own outside
``bar.py``, so that the bar's hash and every registered record stay as they are.  It imports the bar's rows
(``bar.rows``: ``bar.Delayed`` around a registered row), conditions, seed blocks, planner, context, episode
(``bar.run_episode``: the row's episode, then the bar's judge), worker (``bar.BarWorker``, subclassed here), counts and
record key, and edits none of them; it carries its own rules text (``RULES``), schema, key and directory.

**Simulation only.**  The servo values are a model, none is measured on this build's hardware, and no record of this
runner is a gate 1 record: gate 1's servo rules (design package S5) are not registered, so every record says
``development: true`` and none is written under ``stage2/runs/`` (``RULES["development"]``).  Package S4 measures
first; nothing here has been run as a panel.

What it holds
-------------
* ``plan``: the records a run makes: each served row (``servo.NAME_RULE``: ``SV0`` the served R0 stand-in,
  ``SV2/<seed>`` the served R2 stand-in), setting (``servo.NAMED``, ``D`` the drawn panel: 20 drawn sets by seed),
  delay and condition on one of the bar's seed blocks.  The second block (5042-5061) takes the drawn panel alone, and
  only a record of a row that gives a verdict whose headline reading passed (P41's repeat; the lead's fifth list: a
  served stand-in row's D record is reported and never sent there).
* ``dry_run``: the plan with each record's planned seconds and a cost estimate from committed rates; it runs no
  episode and writes nothing (``RULES["cost"]``).
* ``run``: the episodes (``bar.run_episode`` in ``GateWorker``, ``bar.BarWorker``'s form, one process or a pool), one
  record per row and condition (``record_path``) with the servo model's readings of every run and of the record
  (``servo.record_readings``: reported only), resumable per episode; a finished record of another key is never
  overwritten.  A run takes an episode cap (``max_episodes``) and refuses a plan over it (``RULES["cap"]``); it
  refuses SV2 rows on a set that is not registered (``RULES["placement"]``); a run on V0, F25F9 or F25F9cl is a
  development run, listed in ``DEV_RUNS`` (``RULES["development"]``); a cut condition (``--cut-s``, tests only) is
  never judged (``RULES["cut"]``).
* ``reading``: condition 2's pass rule as approved (``RULES["pass_"]``): at a corner only staying up is judged, the
  other limits recorded; the drawn panel by the bar's whole rule; its repeat on the second block.  ``verdict``: a
  served stand-in row gives none (``servo.NO_VERDICT_CONTROLLERS``) and no served row is attributed.
* ``summarize``: the records of a set and body, each with its reading, its verdict and, for the drawn panel, its
  repeat; whether condition 2 holds needs served policy rows (``servo.SV5_HOOK``), which do not exist yet.

Command line: ``python -m designlab.stage2.gate1 {dry-run,run,summary} --set V0 --body 75-M-H-10 --rows SV0,SV2/42
--settings @named,D --delays 4,10`` (``--root`` the sets' root; ``run --out`` a directory of its own and
``--max-episodes`` its cap).
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("disable")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from ..stage1 import panels, plant, tune  # noqa: E402
from . import bar, s2io, servo, setio  # noqa: E402

SCHEMA = "gate1-servo-1"
PARTIAL_SCHEMA = "gate1-partial-1"
#: Gate 1's walk: the 0.30 m/s rung (the bar's own walk condition of that rung, its plan, length and readings) on
#: the grippy floor (mu 0.9), and the same on the slippery floor (mu 0.54), reported (P41; design section 4).
WALK = setio.walk_name(0.30)
CONDITION = f"walk/{WALK}"
SLIPPERY = f"walk/{WALK}_mu{bar.MU_LOW:g}"
#: Gate 1's two delays, ms (P40, P41).
DELAYS_MS = (4, 10)
HEADLINE, SECOND = bar.HEADLINE_BLOCK, bar.REPLICATION_BLOCK
DRAWN = "D"
#: Condition 2's six extremes (judged by staying up alone) and the drawn panel (judged by the bar's whole rule).
CORNERS = tuple(s for s in servo.JUDGED if s != DRAWN)
#: Setting groups a run may name, beside a setting's own name.
SETTING_GROUPS = {"@judged": servo.JUDGED, "@reported": servo.REPORTED,
                  "@named": tuple(s for s in servo.NAMED if s != DRAWN), "@all": servo.NAMED}
#: The served rows' inner controllers, whose committed rates the cost estimate reads (``RULES["cost"]``).
INNER_CONTROLLER = {servo.SERVED_R0: "R0", servo.SERVED_R2: "R2"}
#: The bar's committed rates (``bar dry-run``'s ``COST.md`` of the 75 cm round) and the layer's estimated share on top
#: of them (design section 6: 28 us against about 260 us per physics step, about 10 %; [E], not measured: S4).
COST_MD = s2io.STUDY / "stage2" / "runs" / "2026-11-15-75cm" / "COST.md"
LAYER_COST = 1.10
#: Gate 1's servo rules (design package S5) are not registered: every record of this runner is a development record.
RULES_REGISTERED = False
#: This runner's own listing of its development runs on the registered sets (``RULES["development"]``; the lead's fifth
#: list of 17 Nov 2026), beside rule X1's list ``stage2/tb/dev_episodes.json``, which is ``gates2``'s and which this
#: runner does not write.  A run names another file with ``dev_runs`` (``--dev-runs``).
DEV_RUNS = s2io.STUDY / "stage2" / "tb" / "gate1_dev_runs.json"
DEV_RUNS_SCHEMA = "gate1-dev-runs-1"
#: The settings whose runs that leave the supply window are not counted as stayed up (design section 4: ``V-``; ``W1``
#: holds ``V-``; the lead's fifth list).  Both are reported only, so no verdict moves.
SUPPLY_SETTINGS = ("V-", "W1")
#: Why a served stand-in row's D record has no repeat (``RULES["repeat"]``; the lead's fifth list, the bar's form).
NO_REPEAT = ("a served stand-in row gives no verdict: its D record is reported and never sent to the second block "
             "(the bar's form)")

RULES = dict(
    route=(
        "the route the lead ruled on 16 Nov 2026 (SERVO_MODEL_DESIGN.md section 8, choice 18, route (c)): a runner "
        "outside bar.py, so the bar's hash and every registered record stay as they are.  It imports the bar's rows "
        "(bar.rows), conditions (bar.choose), seed blocks (bar.BLOCKS), planner (bar.BarPlanner), episode "
        "(bar.run_episode: the row's episode and the bar's judge), worker (bar.BarWorker), counts (bar.counts) and "
        "record key (bar.record_key), and edits none of them"),
    rows=(
        servo.NAME_RULE + ".  A run names served rows by their base (SV0, SV2/<seed>), settings and delays; the row of "
        "a record is <base>/<setting>@<ms>ms, built by bar.rows (bar.Delayed around servo.Served): "
        + servo.SERVED_RULE),
    condition=(
        "gate 1's condition 2 (P41; BUY_GATE_DRAFT.md): the 0.30 m/s walk (the bar's walk condition walk/v0.3: its "
        "plan, length, window and readings) on the grippy floor (mu 0.9) at 4 and at 10 ms; the same on the slippery "
        "floor (walk/v0.3_mu0.54) is reported with no verdict; any other condition, delay or a cut condition (a "
        "smoke check) gives a record that is read by no rule here"),
    seeds=(
        "the bar's blocks, 20 seeds per record in order: 3042-3061 (the headline block) and 5042-5061 (the second "
        "block); no seed is dropped or exchanged; an episode that failed to run is recorded as the bar records it "
        "(bar.failed_episode), is never a count, and is run again on the next run of the same key (C8)"),
    pass_=(
        "condition 2 as the owner approved it (P41): at a corner (F-, F+, P0, P+, M-, M+) a record passes when all "
        "20 runs stay up; the number that also meet every limit is recorded and not judged there; the drawn panel D "
        "passes by the bar's whole rule (H1: 20 of 20 stay up and at least 17 meet every limit; reading 2, "
        "bar.counts); every other setting (N, today, V-, J+, G5, T+, X6L, W1, W2, D/<s>) is read and judged by no "
        "rule here.  A reading is judged only for gate 1's walk on the grippy floor at 4 or 10 ms, uncut, when the "
        "record holds the 20 seeds of a block in order and no episode failed to run (bar.counts' judged)"),
    repeat=(
        "P41's repeat: a D record whose headline reading passes is run once more on the second block (5042-5061: by "
        "SeedSequence([s, 3]) 20 other drawn sets), and the repeat must pass too: the drawn panel passes only when "
        "both pass.  Printed beside the headline record as replicated, not replicated, repeat not judged or repeat not "
        "run.  The verdict (the lead's fifth list of 17 Nov 2026): replicated counts as passing, not replicated as "
        "failing, and anything else (not run, not judged) as pending (None), so a repeat that has not run or could "
        "not be read never counts (C8).  Only D has a second block, and only a row that gives a verdict is sent to "
        "it (the bar's form): a served stand-in row's (SV0, SV2) D record is reported and never replicated"),
    verdict=(
        "a served stand-in row (SV0, SV2) gives no verdict (design 5.1; the lead's fourth list): its records carry "
        "their reading and the verdict None; no served row is attributed (no relax run, no attribution label).  A "
        "row of any other controller gives its reading's pass as its verdict, the drawn panel's with its repeat.  "
        "Condition 2 holds when the served policy rows of both training seeds pass all 28 judged records (7 settings "
        "x 2 delays x 2 policies) and every drawn panel's repeat; no served policy row exists yet (servo.SV5_HOOK), "
        "so this runner states no verdict on condition 2"),
    supply=(
        "design section 4 (the lead's fifth list of 17 Nov 2026): a run of V- or W1 (W1 holds V-) that did not fall "
        "and left the supply window (a step below the leg drives' 20 V floor or a collapsed step: Layer.readings) "
        "is not counted as stayed up: in the record's counts (both readings) and in the summary it is \"left the "
        "supply window\" (counts' left_supply_window; survived and succeeded leave it out; by_the_bar keeps the bar's "
        "own two numbers).  Both settings are reported only, so no verdict moves.  A run of any other setting that "
        "left the window is reported (servo_readings) and counted as the bar counts it (the lead's ruling of 16 Nov "
        "2026, choice 10)"),
    readings=servo.READINGS_RULE,
    key=(
        "a record's key: the bar's record key (bar.record_key: the body, set, block, the row's description and "
        "sources, the condition, plan, seconds, seeds, omega, crouch, the inputs' sha256 and the key context) with "
        "this runner's schema, the setting, the servo table's sha256 and this runner's own sources; the bar's and "
        "the panels' source sha256 are in it as data (this runner is in neither, so neither moves).  A finished "
        "record of another key is never overwritten: use a new --out"),
    development=(
        "gate 1's servo rules (SERVO_MODEL_DESIGN.md package S5) are not registered: every record of this runner is a "
        "development record (development: true), never a gate 1 record; none is written under stage2/runs/, "
        "stage1/ or the sets' root.  On a look set this runner makes no episode (rule X1: not before its G9-ref has "
        "passed; the look sets' row rule names no served row).  Until gate 1's servo rules are registered every run "
        "of this runner on V0, F25F9 or F25F9cl (or a version of one) is a development run (the lead's fifth list of "
        "17 Nov 2026): it is listed in this runner's own listing, stage2/tb/gate1_dev_runs.json beside rule X1's "
        "stage2/tb/dev_episodes.json (or the file --dev-runs names), with its command, set, body, block, rows, "
        "settings, delays, conditions, seeds and count (the episodes it starts), once before any episode runs and "
        "again when its records are written, keyed by its session"),
    placement=(
        "the bar's placement rule as amendment 5 (STAGE2_75_PLAN.md, 15 Nov) says it of R2 rows (the lead's fifth "
        "list of 17 Nov 2026): on a set that is not one of the 75 cm round's three (V0, F25F9, F25F9cl) or a version "
        "of one (bar.registered_name: a scratch or fixture set) a served R2 row (SV2, R2 inside) is refused, by the "
        "run and by the dry run, before anything is read or written; SV0 rows (R0 inside) run there"),
    cap=(
        "a run takes an episode cap (--max-episodes, required) and refuses a plan over it before anything is written "
        "or run: the episodes it would start (every seed still to run of every selected record, a failed episode run "
        "again included) must not exceed the cap.  The dry run states each block's count against a cap it is given.  "
        "Package S4 sets the rates; a budget ledger comes with gate 1's servo rules (the lead's fifth list)"),
    cut=(
        "--cut-s cuts every episode of a run to that many seconds: for tests only.  A cut condition's episode is the "
        "served row's episode alone, never judged (the bar's judge is not called: it reads no cut walk), recorded "
        "with judged false and no success, never as an episode that failed to run, so a later run of the same key "
        "keeps it; its record is read by no rule here"),
    cost=(
        "the dry run's estimate (it runs no episode and writes nothing): each record's planned simulated seconds x "
        "the episodes x the inner row's committed rate (the bar's COST.md: R0's for SV0, R2's for SV2; the largest "
        "for another row), and that x 1.10 for the layer's own cost (design section 6, an estimate: package S4 "
        "measures it).  A second-block record is counted apart: it runs only if its headline record passes"),
)


class Gate1Error(s2io.S2Error):
    """A run this runner refuses (``RULES``)."""


# ------------------------------------------------------------------------------------------------
# What a run takes
# ------------------------------------------------------------------------------------------------
def settings_of(names) -> list[str]:
    """Setting names and groups (``SETTING_GROUPS``) as settings, in order, each once."""
    out: list[str] = []
    for n in names:
        for s in SETTING_GROUPS.get(n, (n,)):
            servo.setting_params(s, 0 if s == DRAWN else None)  # (a known setting, or ValueError)
            if s not in out:
                out.append(s)
    return out


def row_names(bases, settings, delays_ms) -> list[tuple[str, str, int]]:
    """``(<base>/<setting>@<ms>ms, setting, ms)`` for every served row base (``SV0``, ``SV2/<seed>``), setting and
    delay; ``ValueError`` for a name ``servo.NAME_RULE`` does not give."""
    out = []
    for b in bases:
        for s in settings:
            name = f"{b}/{s}"
            ctrl = servo.parse_name(name)[0]
            if ctrl not in servo.SERVED_CONTROLLERS:
                raise ValueError(f"{name}: {servo.SV5_HOOK}")
            for ms in delays_ms:
                bar.ms_steps(int(ms), name)
                out.append((f"{name}@{int(ms)}ms", s, int(ms)))
    return out


def base_of(row: str) -> str:
    """A record's row base: ``SV2/42/D@10ms`` -> ``SV2/42``."""
    inner, _ = bar.parse_row(row)
    ctrl, inner_row, _ = servo.parse_name(inner)
    return ctrl if ctrl == servo.SERVED_R0 else f"{ctrl}/{inner_row.split('/', 1)[1]}"


def gate1_reading_applies(cond_key: str, delay_ms: int | None, cut: bool) -> bool:
    return cond_key == CONDITION and delay_ms in DELAYS_MS and not cut


def check_out(out, root=setio.ROOT) -> None:
    """``RULES["development"]``: where a run may write."""
    places = [(s2io.STUDY / "stage1", "stage1/"), (setio.ROOT, "stage2/sets/"),
              (setio._anchored(root), "the sets' root")]
    if not RULES_REGISTERED:
        places.append((s2io.STUDY / "stage2" / "runs", "stage2/runs/ (gate 1's servo rules are not registered)"))
    for parent, what in places:
        if bar._inside(out, parent):
            raise Gate1Error(f"--out {out} lies under {what}: this runner writes no record there "
                             f"({RULES['development']})")


def registered_set(set_name: str) -> str | None:
    """The registered set a run's set stands for (``bar.registered_name``: V0, F25F9, F25F9cl or a version of one): a
    run on it is a development run, listed (``RULES["development"]``), and SV2 rows run there
    (``RULES["placement"]``); None for a scratch or fixture set."""
    return bar.registered_name(set_name)


def check_rows(set_name: str, bases) -> None:
    """``RULES["placement"]``: no served R2 row (``SV2``) on a set that is not registered (a scratch or fixture set),
    as amendment 5 says of R2 rows."""
    if registered_set(set_name) is not None:
        return
    bad = [b for b in bases if str(b).split("/", 1)[0] == servo.SERVED_R2]
    if bad:
        raise Gate1Error(f"{set_name} is a scratch or fixture set (not V0, F25F9, F25F9cl or a version of one): "
                         f"{', '.join(bad)} serve an R2 row, which never runs there ({RULES['placement']})")


def look_set(set_name: str) -> str | None:
    from . import lookbar  # (only to name the look sets)

    return lookbar.look_set_of(set_name)


def record_path(out, set_name: str, body: str, block: str, row: str, cond) -> Path:
    """``<out>/<set>/<body>/<block>/gate1/<row>/<behavior>/<condition>.json`` (a row ``SV2/42/D@10ms`` is three
    directories)."""
    return Path(out) / set_name / body / block / "gate1" / row / cond.behavior / f"{cond.name}.json"


def partial_dir(out, set_name: str, body: str, block: str, row: str, cond) -> Path:
    return Path(out) / set_name / body / block / "partial" / row / cond.behavior / cond.name


def gate1_sources() -> dict:
    """This runner's own sources by sha256 (the served row's module is in each row's sources)."""
    p = Path(__file__).resolve()
    return {s2io.portable(str(p)): s2io.sha256_file(p)}


def runner_sources() -> dict:
    """The servo model's module and this runner by sha256: what ``GateWorker`` checks has not changed during a
    session (the lead's fifth list of 17 Nov 2026), beside the bar's and the panels' sources ``bar.BarWorker``
    checks."""
    files = sorted({Path(servo.__file__).resolve(), Path(__file__).resolve()})
    return {s2io.portable(str(f)): s2io.sha256_file(f) for f in files}


def plan(set_name: str, body: str, bases, settings=("@judged",), delays_ms=DELAYS_MS, conditions=(CONDITION,),
         block: str = HEADLINE, episodes: int = bar.RUNS, cut_s: float | None = None, root=setio.ROOT,
         out=None) -> dict:
    """The records a run of ``block`` makes (module docstring): ``{set_docs, planner, run_dir, units}``, a unit per
    row and condition: ``{row, base, setting, delay_ms, condition, cond, resolved, why, seeds, selected, reason,
    cut}``.  On the second block only the drawn panel's rows are planned: a row that gives a verdict is selected when
    its headline record under ``out`` has a reading that passed, and a served stand-in row never is
    (``RULES["repeat"]``)."""
    if block not in bar.BLOCKS:
        raise ValueError(f"block {block!r}: choose from {', '.join(bar.BLOCKS)}")
    seeds = [int(s) for s in bar.BLOCKS[block][: int(episodes)]]
    sets = settings_of(settings)
    names = row_names(bases, sets, delays_ms)
    run_dir = setio.set_dir(set_name, body, root)
    set_docs = setio.load(run_dir)
    if set_docs["set.json"].get("set") != set_name:
        raise Gate1Error(f"{run_dir}: set.json names the set {set_docs['set.json'].get('set')!r}")
    bar.check_manifest(set_docs["set.json"])
    planner = bar.BarPlanner(run_dir, set_docs["refs.json"]["references"],
                             set_docs["transitions.json"]["transitions"], set_docs["set.json"])
    units = []
    for c in bar.choose(set_docs, block, list(conditions)):
        cc = c if cut_s is None else dataclasses.replace(c, seconds=float(cut_s))
        why = planner.missing(c)
        res = None if why is not None else planner.resolve(cc)
        for name, s, ms in names:
            u = dict(row=name, base=base_of(name), setting=s, delay_ms=ms, condition=c.key, cond=cc, resolved=res,
                     why=why, seeds=seeds, selected=True, reason=None,
                     cut=res is not None and res.get("cut_from_s") is not None)
            if block == SECOND:
                if s != DRAWN:
                    continue  # (only the drawn panel has a second block)
                if bar.controller_of(name) in servo.NO_VERDICT_CONTROLLERS:
                    u["selected"], u["reason"] = False, NO_REPEAT
                else:
                    u["selected"], u["reason"] = headline_passed(out, set_name, body, name, cc)
            units.append(u)
    return dict(set_docs=set_docs, planner=planner, run_dir=run_dir, units=units, seeds=seeds, block=block,
                settings=sets)


def headline_passed(out, set_name: str, body: str, row: str, cond) -> tuple[bool | None, str]:
    """Whether the headline record of ``row`` and ``cond`` under ``out`` has a reading that passed
    (``RULES["repeat"]``): (True, why) selects the second block's record; (False, why) does not; (None, why) when
    ``out`` is not given (the dry run's "if the headline record passes")."""
    if out is None:
        return None, "runs only if the headline record's reading passes (P41)"
    rec = bar._read(record_path(out, set_name, body, HEADLINE, row, cond))
    if not rec or rec.get("status") != "done":
        return False, "no finished headline record"
    r = rec.get("reading") or {}
    if r.get("pass_") is True:
        return True, "the headline record's reading passed"
    return False, ("the headline record's reading did not pass" if r.get("pass_") is False
                   else "the headline record's reading is not judged")


# ------------------------------------------------------------------------------------------------
# The pass rule, the verdict, the readings
# ------------------------------------------------------------------------------------------------
def rule_of(setting: str) -> str:
    """``corner`` (stay up only), ``whole`` (the bar's whole rule: the drawn panel) or ``reported``."""
    return "corner" if setting in CORNERS else "whole" if setting == DRAWN else "reported"


def left_supply_window(e: dict) -> bool:
    """Whether an episode's run left the supply window (``Layer.readings``; False without readings)."""
    return bool(((e.get("servo") or {}).get("readings") or {}).get("left_supply_window"))


def counts(setting: str, episodes: list[dict], seeds, which: str = "success") -> dict:
    """The bar's counts of a record in one reading (``bar.counts``), with ``RULES["supply"]``: for ``V-`` and ``W1``
    a run that did not fall and left the supply window is not counted as stayed up (nor as meeting every limit) but
    as ``left_supply_window``; ``by_the_bar`` keeps the bar's own two numbers.  Every other setting's counts are the
    bar's, with ``left_supply_window`` 0."""
    c = bar.counts(episodes, seeds, which)
    if setting not in SUPPLY_SETTINGS:
        return dict(c, left_supply_window=0)
    skey = bar.READINGS[which][0]
    left = [e for e in episodes if not e.get("error") and not e["fell"] and left_supply_window(e)]
    n, surv = c["episodes"], c["survived"] - len(left)
    succ = c["succeeded"] - int(sum(bool(e.get(skey)) for e in left))
    return dict(c, survived=surv, succeeded=succ, left_supply_window=len(left),
                survival_lcb=panels.lcb(surv, n), success_lcb=panels.lcb(succ, n),
                pass_=bool(surv == bar.RUNS and succ >= bar.MIN_SUCCEED) if c["judged"] else None,
                by_the_bar=dict(survived=c["survived"], succeeded=c["succeeded"]))


def reading(setting: str, episodes: list[dict], seeds, applies: bool = True) -> dict:
    """Condition 2's reading of a record (``RULES["pass_"]``): ``pass_`` True or False when judged, else None.
    ``applies`` is False for a record off gate 1's walk, floor and delays, or of a cut condition."""
    c = counts(setting, episodes, seeds, "success")
    rule = rule_of(setting)
    judged = bool(c["judged"]) and applies and rule != "reported"
    if not judged:
        pass_ = None
    elif rule == "corner":
        pass_ = bool(c["survived"] == bar.RUNS)
    else:
        pass_ = bool(c["pass_"])
    why = None
    if rule == "reported":
        why = f"{setting} is no setting of condition 2's seven: read by no rule here"
    elif not applies:
        why = "not gate 1's walk on the grippy floor at 4 or 10 ms, or a cut condition"
    elif not c["judged"]:
        why = "the record does not hold a block's 20 seeds in order, or an episode failed to run"
    return dict(setting=setting, rule=rule, judged=judged, pass_=pass_, reason=why, runs=c["episodes"],
                survived=c["survived"], succeeded=c["succeeded"], errors=c["errors"],
                left_supply_window=c["left_supply_window"],
                limits=("recorded, not judged" if rule == "corner" else "judged" if rule == "whole" else "recorded"))


def verdict(row: str, rd: dict) -> dict:
    """``RULES["verdict"]``: the record's verdict (None for a served stand-in row) and why."""
    ctrl = bar.controller_of(row)
    if ctrl in servo.NO_VERDICT_CONTROLLERS:
        return dict(value=None, reason=f"{ctrl} is a served stand-in row: it gives no verdict (design 5.1)")
    return dict(value=rd["pass_"], reason=rd["reason"])


def served_of(row) -> servo.Served:
    inner = getattr(row, "inner", None)
    if not isinstance(inner, servo.Served):
        raise Gate1Error(f"{getattr(row, 'name', row)!r} is not a delayed served row (bar.Delayed around servo.Served)")
    return inner


# ------------------------------------------------------------------------------------------------
# The worker: the bar's, with the servo readings of every episode
# ------------------------------------------------------------------------------------------------
def cut_episode(row, ctx, cond, seed: int) -> dict:
    """``RULES["cut"]``: the episode of a cut condition (``--cut-s``, tests only): the row's episode alone, never
    judged (the bar's judge reads no cut walk); no success, no limits, no trace, and no error."""
    t0 = time.perf_counter()
    b, out, rec = row.episode(ctx, cond, int(seed), ())
    return dict(seed=int(seed), fell=bool(out.get("fell")), fall_time_s=out.get("fall_time_s"),
                fall_cause=out.get("fall_cause"), fell_before_command=None, fall_candidates=[],
                survival_s=float(rec.t[-1]) if rec.rows else 0.0, cut=True, judged=False,
                not_judged="a cut condition (--cut-s, tests only): the episode is never judged (RULES['cut'])",
                succeeded=None, succeeded_but_A1=None, failed_terms=[], failed_terms_but_A1=[], relaxed=[],
                wall_s=time.perf_counter() - t0)


class GateWorker(bar.BarWorker):
    """``bar.BarWorker`` (its checks of the bar's and the panels' sources, the set and the inputs), whose episode
    record also holds the served row's parameters and the layer's readings of the run (``servo``).  It also refuses
    to start when the servo model's module or this runner differ from the session's (``runner``: ``runner_sources``;
    the lead's fifth list of 17 Nov 2026), and runs a cut condition's episode unjudged (``cut_episode``)."""

    def __init__(self, *a, runner: dict | None = None, **kw):
        if runner is not None and runner_sources() != runner:
            raise tune.PoolError("the servo model's module or gate 1's runner (servo.py, gate1.py) changed during the "
                                 "session: restore them and run again")
        super().__init__(*a, **kw)
        self.runner = runner

    def evaluate(self, job, seed) -> dict:
        row, cond, relax, pdir, key_sha256, cut = job
        t0 = time.perf_counter()
        r = self.rows[row]
        try:
            sv = served_of(r)
            if cut:
                e = cut_episode(r, self.ctx, cond, int(seed))
            elif self.traces is None:
                e = bar.run_episode(r, self.ctx, cond, int(seed), tuple(relax))
            else:
                why = self.not_kept(row)
                e = bar.run_episode(r, self.ctx, cond, int(seed), tuple(relax), trace_root=self.traces,
                                    keep_trace=why is None, not_kept_reason=why)
            e["servo"] = dict(setting=sv.params.name, readings=sv.readings)
        except Exception:
            e = dict(bar.failed_episode(cond, int(seed), traceback.format_exc(), relax, time.perf_counter() - t0),
                     servo=None)
        if pdir is not None:
            s2io.write_json(Path(pdir) / f"{int(seed)}.json",
                            dict(schema=PARTIAL_SCHEMA, key_sha256=key_sha256, seed=int(seed), relax=list(relax),
                                 session=self.session, episode=e, written_at=s2io.now_utc()))
        return e


def _worker(body, run_dir, mode, rate, expected_inputs=None, **kw):
    """``tune.Pool``'s factory (a top-level function of this importable module)."""
    return GateWorker(body, run_dir, mode, rate, expected_inputs, **kw)


def _lost(job, reason: str) -> dict:
    (row, cond, relax, pdir, key_sha256, cut), seed = job
    return dict(bar.failed_episode(cond, seed, f"worker lost: {reason}", relax), servo=None)


# ------------------------------------------------------------------------------------------------
# Keys and records
# ------------------------------------------------------------------------------------------------
def record_key(body: str, set_name: str, block: str, row, unit: dict, inputs: dict, context: dict, sources: dict,
               omega: float, crouch: float) -> dict:
    """``RULES["key"]``."""
    k = bar.record_key(body, set_name, block, row, unit["cond"], unit["resolved"], unit["seeds"], inputs, context,
                       sources, omega, crouch)
    return bar._plain_json(dict(k, schema=SCHEMA, setting=unit["setting"], delay_ms=unit["delay_ms"],
                                table_sha256=servo.TABLE_SHA256, runner_sources=gate1_sources()))


def condition_record(body: str, set_name: str, block: str, row, unit: dict, episodes: list[dict], key: dict,
                     prov: dict, inputs: dict, sessions: list, set_docs: dict, repeat_of: dict | None = None) -> dict:
    """A record of this runner (``SCHEMA``): the counts in the bar's two readings, condition 2's reading, the verdict,
    the servo readings of the record and of every run, the episodes, the key and the rules."""
    seeds, cond = unit["seeds"], unit["cond"]
    cut = unit["resolved"].get("cut_from_s") is not None
    rd = reading(unit["setting"], episodes, seeds, gate1_reading_applies(cond.key, unit["delay_ms"], cut))
    rec = dict(
        schema=SCHEMA, development=not RULES_REGISTERED, status="done", body=body, set=set_name, block=block,
        row=row.name, base=unit["base"], controller=bar.controller_of(row.name), setting=unit["setting"],
        delay_ms=unit["delay_ms"], behavior=cond.behavior, condition=cond.name, key_name=cond.key,
        gate1_condition=cond.key == CONDITION, cut=cut, seeds=list(seeds), plan=key["plan"], seconds=key["seconds"],
        counts=counts(unit["setting"], episodes, seeds, "success"),
        counts_but_A1=counts(unit["setting"], episodes, seeds, "but_A1"),
        reading=rd, verdict=verdict(row.name, rd),
        attribution=dict(label="not attributed: no served row is attributed (design 5.1)", episodes=0),
        falls=bar.fall_split(episodes), success_label=bar.success_label(episodes),
        servo_readings=servo.record_readings([(e.get("servo") or {}).get("readings") for e in episodes]),
        servo_settings=sorted({(e.get("servo") or {}).get("setting") for e in episodes} - {None}),
        repeat_of=repeat_of, controller_description=s2io.portable(row.describe()),
        definition=bar.describe_condition(cond),
        episodes=episodes, rules=dict(RULES, bar=bar.rules_of(cond, set_docs)), key=key,
        key_sha256=s2io.sha256_json(key), provenance=prov,
        inputs=dict(dir=s2io.portable(inputs.get("dir")), files=key["inputs"]), set_freeze=inputs.get("freeze"),
        sessions=list(sessions), episode_wall_s=float(sum(e.get("wall_s") or 0.0 for e in episodes)),
        simulated_s=float(sum(e.get("survival_s") or 0.0 for e in episodes)), ended_at=s2io.now_utc())
    return bar._plain_json(rec)


# ------------------------------------------------------------------------------------------------
# The dry run
# ------------------------------------------------------------------------------------------------
def rate_of(rates: dict, row: str) -> tuple[float, str]:
    """(wall seconds per simulated second, which committed rate) of a served row: its inner row's controller's."""
    ctrl = INNER_CONTROLLER.get(bar.controller_of(row))
    r = rates.get(ctrl) if ctrl else None
    if r is None:
        k = max(rates, key=lambda c: float(rates[c]["wall_per_sim_s"]))
        return float(rates[k]["wall_per_sim_s"]), f"{k} (the largest measured)"
    return float(r["wall_per_sim_s"]), ctrl


def dry_run(set_name: str, body: str, bases, settings=("@judged",), delays_ms=DELAYS_MS, conditions=(CONDITION,),
            blocks=(HEADLINE, SECOND), episodes: int = bar.RUNS, cut_s: float | None = None, root=setio.ROOT,
            rates: dict | None = None, out=None, max_episodes: int | None = None, log=print) -> dict:
    """Every record a run would make on ``blocks``, its planned seconds and its cost (``RULES["cost"]``) from the
    committed rates (``rates``: a ``bar.read_cost`` document; default ``COST_MD``'s), and each block's episodes
    against the cap ``max_episodes`` when given (``RULES["cap"]``).  Refuses SV2 rows on a set that is not
    registered (``RULES["placement"]``).  Runs no episode and writes nothing; ``text`` is logged."""
    check_rows(set_name, bases)
    cost = rates if rates is not None else (bar.read_cost(COST_MD) if COST_MD.is_file() else None)
    lines, per_block = [], {b: Counter() for b in blocks}
    totals = Counter({f"{k}_{x}": 0 for k in ("planned", "conditional")
                      for x in ("records", "episodes", "simulated_s", "core_h")})
    for block in blocks:
        p = plan(set_name, body, bases, settings, delays_ms, conditions, block, episodes, cut_s, root, out)
        for u in p["units"]:
            res, n = u["resolved"], len(u["seeds"])
            if u["why"] is not None:
                status, n = f"not run: {u['why']['reason']}", 0
            elif u["selected"] is None:
                status = f"if selected: {u['reason']}"
            elif not u["selected"]:
                status, n = f"not selected: {u['reason']}", 0
            else:
                old = None if out is None else bar._read(record_path(out, set_name, body, block, u["row"], u["cond"]))
                status = "recorded" if old and old.get("status") == "done" else "to run"
            sec = None if res is None else float(res["seconds"])
            rate, which = (None, None) if cost is None else rate_of(cost["rates"], u["row"])
            core_h = None if rate is None or sec is None else n * sec * rate / 3600.0
            lines.append(dict(block=block, row=u["row"], setting=u["setting"], delay_ms=u["delay_ms"],
                              condition=u["condition"], rule=rule_of(u["setting"]), seconds=sec, episodes=n,
                              status=status, rate=which, core_h=core_h,
                              core_h_with_layer=None if core_h is None else core_h * LAYER_COST))
            key = "conditional" if status.startswith("if selected") else "planned"
            per_block[block][key] += n if status == "to run" or key == "conditional" else 0
            totals[f"{key}_records"] += 1 if n else 0
            totals[f"{key}_episodes"] += n
            totals[f"{key}_simulated_s"] += 0.0 if sec is None else n * sec
            totals[f"{key}_core_h"] += core_h or 0.0
    text = ["# Gate 1's served panels: plan and cost (a dry run)", "",
            f"Set `{set_name}`, body `{body}`, rows {', '.join(f'`{b}`' for b in bases)}, blocks "
            f"{', '.join(blocks)}, {episodes} episodes per record; made {s2io.now_utc()} by `designlab.stage2.gate1 "
            "dry-run`.  It runs no episode and writes nothing: an estimate, not a result.  Simulation only; every "
            "record of this runner is a development record (gate 1's servo rules are not registered).", ""]
    if cost is not None:
        mo = cost.get("measured_on") or {}
        text += [f"Rates: the bar's committed `COST.md` (`{s2io.portable(str(COST_MD))}` unless given), measured on "
                 f"`{mo.get('condition')}` of `{mo.get('body')}` without the servo layer: "
                 + ", ".join(f"{k} {float(v['wall_per_sim_s']):.4f}" for k, v in cost["rates"].items())
                 + " wall s per simulated s.  The layer's own cost is not in them: the last column adds the design's "
                 f"estimate (x {LAYER_COST:g}, section 6; package S4 measures it).", ""]
    else:
        text += ["No committed rates: no cost.", ""]
    text += ["| Block | Row | Rule | Condition | Episodes | Planned s each | Core-h | Core-h with layer | Status |",
             "|---|---|---|---|---|---|---|---|---|"]
    for x in lines:
        text.append(f"| {x['block']} | `{x['row']}` | {x['rule']} | `{x['condition']}` | {x['episodes']} | "
                    + ("-" if x["seconds"] is None else f"{x['seconds']:.2f}") + " | "
                    + ("-" if x["core_h"] is None else f"{x['core_h']:.4f}") + " | "
                    + ("-" if x["core_h_with_layer"] is None else f"{x['core_h_with_layer']:.4f}")
                    + f" | {x['status']} |")
    text += ["", f"Planned: {totals['planned_records']} records, {totals['planned_episodes']} episodes, "
                 f"{totals['planned_simulated_s'] / 3600.0:.3f} simulated hours, {totals['planned_core_h']:.3f} "
                 f"core-hours ({totals['planned_core_h'] * LAYER_COST:.3f} with the layer's estimate).",
             f"Second block, only for a drawn-panel record whose headline reading passes (P41): up to "
             f"{totals['conditional_records']} records, {totals['conditional_episodes']} episodes, "
             f"{totals['conditional_core_h']:.3f} core-hours ({totals['conditional_core_h'] * LAYER_COST:.3f} with "
             "the layer's estimate).", ""]
    cap = {}
    for b, c in per_block.items():
        most = c["planned"] + c["conditional"]
        cap[b] = dict(to_run=c["planned"], up_to=most, max_episodes=max_episodes,
                      within=None if max_episodes is None else most <= int(max_episodes))
        text.append(f"Episodes a `run --block {b}` would start: {c['planned']}"
                    + (f" (up to {most} if every conditional record is selected)" if c["conditional"] else "")
                    + ("; no cap given (`run` requires --max-episodes)." if max_episodes is None else
                       f"; cap {int(max_episodes)}: " + ("within it." if most <= int(max_episodes) else
                                                         "over it, so `run` refuses it (RULES['cap'])."))
                    )
    text.append("")
    doc = dict(set=set_name, body=body, bases=list(bases), blocks=list(blocks), episodes=int(episodes),
               cost=cost, records=lines, totals=dict(totals), layer_cost=LAYER_COST, cap=cap, text="\n".join(text))
    log(doc["text"])
    return doc


# ------------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------------
def command_of(set_name, body, out, bases, settings, delays_ms, conditions, block, procs, episodes, cut_s, traces,
               root, max_episodes, dev_runs) -> str:
    """The command line that makes the run ``run``'s arguments name (``RULES["development"]``'s listing)."""
    def j(xs):
        return ",".join(str(x) for x in xs)

    parts = ["python -m designlab.stage2.gate1 run", f"--set {set_name}", f"--body {body}", f"--rows {j(bases)}",
             f"--settings {j(settings)}", f"--delays {j(int(x) for x in delays_ms)}", f"--conditions {j(conditions)}",
             f"--episodes {int(episodes)}", f"--block {block}", f"--procs {int(procs)}",
             f"--max-episodes {int(max_episodes)}", f"--root {s2io.portable(str(root))}",
             f"--out {s2io.portable(str(out))}", f"--dev-runs {s2io.portable(str(dev_runs))}"]
    if cut_s is not None:
        parts.append(f"--cut-s {float(cut_s):g}")
    if traces:
        parts.append("--traces")
    return " ".join(parts)


def list_dev_run(path, entry: dict) -> dict:
    """``RULES["development"]``: list one run on a registered set in ``path`` (``DEV_RUNS``), keyed by its session: a
    later call for the same session updates the entry (its state ``done`` after ``started``).  Whatever else the
    file holds is kept."""
    path = Path(path)
    doc = bar._read(path) if path.is_file() else None
    doc = doc or dict(schema=DEV_RUNS_SCHEMA, rule=RULES["development"],
                      note="gate 1's runner of the served panels: its development runs on V0, F25F9 and F25F9cl "
                           "(simulation only; no record of these runs is a gate 1 record)", runs=[])
    runs = doc.setdefault("runs", [])
    for i, e in enumerate(runs):
        if e.get("session") == entry["session"]:
            runs[i] = dict(e, **entry)
            break
    else:
        runs.append(dict(entry))
    s2io.write_json(path, bar._plain_json(doc))
    return doc


def run(set_name: str, body: str, out, bases, settings=("@judged",), delays_ms=DELAYS_MS, conditions=(CONDITION,),
        block: str = HEADLINE, procs: int = 1, episodes: int = bar.RUNS, cut_s: float | None = None,
        traces: bool = False, root=setio.ROOT, stage1_run=bar.STAGE1_RUN, timeout_s: float = tune.EVAL_TIMEOUT_S,
        log=print, *, max_episodes: int, dev_runs=DEV_RUNS, command: str | None = None) -> dict:
    """Run the plan's records (``plan``) into ``out`` (``record_path``); resumes per episode (partial files), keeps a
    finished record of the same key and never overwrites one of another key (``Gate1Error``).  ``traces`` false has
    every episode's trace hashed and not kept.  Refuses SV2 rows on a set that is not registered
    (``RULES["placement"]``) and a plan that would start more than ``max_episodes`` episodes (``RULES["cap"]``),
    before anything is written; a run on V0, F25F9 or F25F9cl is listed in ``dev_runs`` (``RULES["development"]``)
    with ``command`` (default: the command line of these arguments).  Returns ``summarize``'s document with the
    session."""
    s2io.require_one_thread()
    out = Path(out)
    check_out(out, root)
    if look_set(set_name) is not None:
        raise Gate1Error(f"{set_name} is a look set: {RULES['development']}")
    check_rows(set_name, bases)
    if isinstance(max_episodes, bool) or int(max_episodes) != max_episodes or int(max_episodes) < 0:
        raise ValueError(f"max_episodes {max_episodes!r}: a whole number of episodes, 0 or more ({RULES['cap']})")
    max_episodes = int(max_episodes)
    dev = registered_set(set_name)
    if command is None:
        command = command_of(set_name, body, out, bases, settings, delays_ms, conditions, block, procs, episodes, cut_s,
                             traces, root, max_episodes, dev_runs)
    t_start = time.perf_counter()
    p = plan(set_name, body, bases, settings, delays_ms, conditions, block, episodes, cut_s, root, out)
    set_docs, run_dir, seeds = p["set_docs"], p["run_dir"], p["seeds"]
    todo_units = [u for u in p["units"] if u["selected"]]
    row_objs = bar.rows(body, sorted({u["row"] for u in todo_units}), stage1_run)
    files = sorted({f for u in todo_units if u["resolved"] is not None for f in u["resolved"]["files"]})
    crouch = bar.set_crouch(set_docs["set.json"], set_docs["refs.json"]["references"], bar.walks_of(files))
    inputs, _ = setio.read_inputs(run_dir, files, require_frozen=True)
    omega = bar.omega_of(body, run_dir, crouch, p["planner"].home_walk) if files else None
    with s2io.Lock(out / ".lock"):
        for d in (out / set_name,):
            if d.is_dir():
                s2io.sweep_tmp(d)
        src, bsrc, kctx = panels.panel_sources(), bar.bar_sources(), panels.key_context(body)
        key_sources = dict(panels=src["sha256"], bar=bsrc["sha256"])
        git = s2io.git_state(own_outputs=[str(out.resolve())] + ([str(Path(dev_runs).resolve())] if dev else []))
        rsrc = runner_sources()  # (taken once: every worker checks its own against these, GateWorker)
        prov = dict(plant.provenance(), git=git, panel_sources=src, bar_sources=bsrc, runner_sources=gate1_sources())
        started_at = s2io.now_utc()
        sess = dict(id=f"{started_at}/{os.getpid()}", started_at=started_at, commit=git["commit"], dirty=git["dirty"],
                    versions=panels.library_versions(), blas_threads=s2io.blas_threads(), procs=int(procs),
                    block=block, traces=bool(traces))
        session = dict(sess, set=set_name, body=body, rows=sorted(row_objs), conditions=list(conditions),
                       seeds=seeds, episodes_run=0, skipped=[], reused_partial=[], rerun_failed=[], not_run=[],
                       not_selected=[f"{u['row']} {u['condition']}: {u['reason']}" for u in p["units"]
                                     if not u["selected"]], incomplete=[], incidents=[])
        trace_root = out / set_name / body / block / "gate1" / "traces"
        items, mismatched, not_run = [], [], []
        for u in todo_units:
            row = row_objs[u["row"]]
            path = record_path(out, set_name, body, block, u["row"], u["cond"])
            if u["why"] is not None:
                not_run.append((u, path))
                continue
            key = record_key(body, set_name, block, row, u, inputs, kctx, key_sources, omega, crouch)
            kh = s2io.sha256_json(key)
            old = bar._read(path)
            if path.exists() and (old is None or old.get("key") != key):
                mismatched.append((path, bar.key_differences((old or {}).get("key"), key)))
                continue
            it = dict(u, row_obj=row, path=path, key=key, kh=kh, old=old, run=[], reused={},
                      pdir=partial_dir(out, set_name, body, block, u["row"], u["cond"]) / kh[:16])
            if old is not None and old.get("status") == "done":
                it["run"] = [e["seed"] for e in old["episodes"] if e.get("error")]
                (session["rerun_failed"] if it["run"] else session["skipped"]).append(f"{u['row']} {u['condition']}")
            else:
                it["reused"] = bar.read_partials(it["pdir"], kh, seeds)
                it["run"] = [s for s in seeds if s not in it["reused"]]
                if it["reused"]:
                    session["reused_partial"].append(f"{u['row']} {u['condition']}: seeds {sorted(it['reused'])}")
            items.append(it)
        if mismatched:
            raise Gate1Error("\n".join(f"{q}: a finished record holds another key:\n    " + "\n    ".join(d[:12])
                                       for q, d in mismatched)
                             + "\nA finished record is never overwritten: use a new --out.")
        jobs = [((it["row"], it["cond"], (), str(it["pdir"]), it["kh"], bool(it["cut"])), s) for it in items
                for s in it["run"]]
        if len(jobs) > max_episodes:
            raise Gate1Error(f"the plan would start {len(jobs)} episodes, over the cap of {max_episodes} "
                             f"(--max-episodes; {RULES['cap']}): nothing was run or written")
        for u, path in not_run:
            if not path.exists():
                s2io.write_json(path, bar._plain_json(dict(
                    schema=SCHEMA, development=not RULES_REGISTERED, status="not run", body=body, set=set_name,
                    block=block, row=u["row"], setting=u["setting"], delay_ms=u["delay_ms"],
                    key_name=u["cond"].key, why=u["why"], rules=RULES, ended_at=s2io.now_utc())))
            session["not_run"].append(f"{u['row']} {u['condition']}: {u['why']['reason']}")
        entry = None
        if dev is not None:  # listed before any episode runs: a killed run is still on the list
            entry = dict(session=sess["id"], state="started", command=command, set=set_name, registered_set=dev,
                         body=body, block=block, rows=sorted(row_objs), bases=list(bases),
                         settings=list(p["settings"]), delays_ms=[int(x) for x in delays_ms],
                         conditions=list(conditions), seeds=list(seeds), count=len(jobs), max_episodes=max_episodes,
                         cut_s=cut_s, out=s2io.portable(str(out.resolve())), commit=git["commit"],
                         started_at=started_at)
            list_dev_run(dev_runs, entry)
        pool = None
        try:
            if jobs:
                pool = tune.Pool(body, run_dir, procs, mode="bar", rate=bar.RATE, timeout_s=timeout_s,
                                 factory=functools.partial(
                                     _worker, set_name=set_name, files=files,
                                     rows=[row_objs[r].spec() for r in sorted({it["row"] for it in items})],
                                     sources=panels.fingerprint(src["sha256"], kctx), bar=bsrc["sha256"],
                                     traces=str(trace_root), keep_traces=bool(traces), crouch_frac=None,
                                     session=sess, runner=rsrc))
                theirs = (pool.inputs or {}).get("files") or {}
                diff = [f for f in files if theirs.get(f) != inputs["files"].get(f)]
                if diff or (pool.inputs or {}).get("freeze") != inputs.get("freeze"):
                    raise Gate1Error(f"{run_dir}: the workers read other inputs than the session's: {diff}")
            results = pool.map(jobs, failed=_lost) if jobs else []
            session["episodes_run"] = len(jobs)
            if pool is not None:
                session["incidents"] = list(pool.incidents)
        finally:
            if pool is not None:
                pool.close()
        i = 0
        for it in items:
            new = dict(zip(it["run"], results[i: i + len(it["run"])], strict=True))
            i += len(it["run"])
            old = it["old"]
            if old is not None and old.get("status") == "done" and not new:
                continue
            sessions = list((old or {}).get("sessions") or [])
            if new:
                sessions.append(dict(sess, ended_at=s2io.now_utc(), seeds_run=sorted(new),
                                     wall_s=float(sum(e.get("wall_s") or 0.0 for e in new.values()))))
            have = {e["seed"]: e for e in (old or {}).get("episodes") or []}
            have.update({s: doc["episode"] for s, doc in it["reused"].items()})
            have.update(new)
            rep = None
            if block == SECOND:
                rep = dict(block=HEADLINE, record=os.path.relpath(record_path(out, set_name, body, HEADLINE, it["row"],
                                                                              it["cond"]), out))
            rec = condition_record(body, set_name, block, it["row_obj"], it, [have[s] for s in seeds], it["key"], prov,
                                   inputs, sessions, set_docs, rep)
            s2io.write_json(it["path"], rec)
            bar.drop_partials(it["pdir"].parent)
            c, rd, sr = rec["counts"], rec["reading"], rec["servo_readings"]
            if c["errors"]:
                session["incomplete"].append(f"{it['row']} {it['condition']}: {c['errors']} episodes failed to run")
            log(f"{set_name} {body} {block} {it['row']} {it['condition']}: {c['survived']}/{c['episodes']} stay up"
                + (f" ({c['left_supply_window']} more left the supply window, not counted as up)"
                   if c["left_supply_window"] else "")
                + f", {c['succeeded']} meet every limit; reading {rd['rule']}: "
                + ("not judged" if rd["pass_"] is None else "pass" if rd["pass_"] else "FAIL")
                + f"; verdict {rec['verdict']['value']}; {sr['runs_left_supply_window']} runs left the supply window")
        session.update(ended_at=s2io.now_utc(), wall_s=time.perf_counter() - t_start)
        if entry is not None:
            list_dev_run(dev_runs, dict(entry, state="done", episodes_run=session["episodes_run"],
                                        incomplete=list(session["incomplete"]), ended_at=session["ended_at"]))
    return dict(summarize(out, set_name, body), session=session)


# ------------------------------------------------------------------------------------------------
# The summary
# ------------------------------------------------------------------------------------------------
def records(out, set_name: str, body: str) -> list[dict]:
    """Every finished or "not run" record of this runner under ``out`` for the set and body."""
    root = Path(out) / set_name / body
    docs = []
    for path in sorted(root.glob("*/gate1/**/*.json")):
        if "traces" in path.parts:
            continue
        d = bar._read(path)
        if d and d.get("schema") == SCHEMA:
            docs.append(dict(d, path=os.path.relpath(path, out)))
    return docs


#: ``RULES["repeat"]``: a passing D record's verdict by its repeat's word; any other word is pending (None).
REPEAT_VERDICT = {"replicated": True, "not replicated": False}


def repeat_word(head: dict, second: dict | None) -> str | None:
    """``RULES["repeat"]``'s word for a drawn-panel headline record."""
    if head.get("setting") != DRAWN:
        return None
    if bar.controller_of(head["row"]) in servo.NO_VERDICT_CONTROLLERS:
        return f"no repeat ({NO_REPEAT})"
    if (head.get("reading") or {}).get("pass_") is not True:
        return "no repeat (the headline reading did not pass)"
    if not second or second.get("status") != "done":
        return "repeat not run"
    r = (second.get("reading") or {}).get("pass_")
    return "replicated" if r is True else "not replicated" if r is False else "repeat not judged"


def summarize(out, set_name: str, body: str, write: bool = True) -> dict:
    """The records of the set and body (``records``), each with its reading, verdict and repeat; whether condition 2
    holds (``RULES["verdict"]``: None while no row gives a verdict).  Written to ``<out>/<set>/<body>/
    gate1_summary.json`` unless ``write`` is false."""
    recs = records(out, set_name, body)
    second = {(r["row"], r.get("key_name")): r for r in recs if r.get("block") == SECOND}
    rows = []
    for r in recs:
        if r.get("status") != "done":
            rows.append(dict(block=r["block"], row=r["row"], condition=r.get("key_name"), status=r.get("status")))
            continue
        word = repeat_word(r, second.get((r["row"], r["key_name"]))) if r["block"] == HEADLINE else None
        v = r["verdict"]["value"]
        if v is True and r["setting"] == DRAWN and r["block"] == HEADLINE:
            v = REPEAT_VERDICT.get(word)  # (P41: the repeat must pass too; anything else is pending, C8)
        sr = r["servo_readings"]
        rows.append(dict(block=r["block"], row=r["row"], base=r["base"], setting=r["setting"], delay_ms=r["delay_ms"],
                         condition=r["key_name"], status="done", survived=r["counts"]["survived"],
                         succeeded=r["counts"]["succeeded"], runs=r["counts"]["episodes"], rule=r["reading"]["rule"],
                         reading=r["reading"]["pass_"], verdict=v, repeat=word,
                         not_up_left_supply_window=int(r["counts"].get("left_supply_window") or 0),
                         left_supply_window=sr["runs_left_supply_window"],
                         inside_play_share_leg_mean=sr["inside_play_share_leg_mean"], development=r["development"],
                         path=r["path"]))
    verdict_rows = [x for x in rows if x["status"] == "done" and bar.controller_of(x["row"])
                    not in servo.NO_VERDICT_CONTROLLERS]
    holds, why = None, ("no row of these records gives a verdict: SV0 and SV2 are served stand-in rows, and no served "
                        "policy row exists yet (servo.SV5_HOOK)")
    if verdict_rows:
        bases = sorted({x["base"] for x in verdict_rows})
        need = [(s, ms) for s in servo.JUDGED for ms in DELAYS_MS]
        per = {}
        for b in bases:
            got = {(x["setting"], x["delay_ms"]): x["verdict"] for x in verdict_rows
                   if x["base"] == b and x["block"] == HEADLINE and x["condition"] == CONDITION}
            per[b] = None if any(k not in got or got[k] is None for k in need) else all(got[k] for k in need)
        if any(v is False for v in per.values()):
            holds, why = False, f"a judged record does not pass: {per}"
        elif len(bases) >= 2 and all(v is True for v in per.values()):
            holds, why = True, f"every judged record and every repeat passes for {bases}"
        else:
            why = f"not every judged record of two policies is there and judged: {per}"
    read = dict(holds_if_registered=holds, reason_if_registered=why)
    if not RULES_REGISTERED:  # (RULES["development"]: no record of this runner is a gate 1 record)
        holds = None
        why = "development records: gate 1's servo rules are not registered (package S5); " + why
    doc = dict(schema=SCHEMA + "-summary", set=set_name, body=body, development=not RULES_REGISTERED, records=rows,
               condition2=dict(holds=holds, reason=why, **read), rules=RULES, at=s2io.now_utc())
    if write and recs:
        s2io.write_json(Path(out) / set_name / body / "gate1_summary.json", doc)
    return doc


def summary_text(doc: dict) -> str:
    """The summary as a table.  "Up" leaves out a V- or W1 run that left the supply window (``RULES["supply"]``):
    "Left supply window" then names it as such ("not up"); for every other setting that column is reported only."""
    lines = [f"Gate 1's served panels on {doc['set']}/{doc['body']} (development records; simulation only)",
             "| Block | Row | Rule | Up | Limits | Reading | Verdict | Repeat | Left supply window |",
             "|---|---|---|---|---|---|---|---|---|"]
    for x in doc["records"]:
        if x["status"] != "done":
            lines.append(f"| {x['block']} | `{x['row']}` | - | - | - | {x['status']} | - | - | - |")
            continue
        left = (f"{x['left_supply_window']}, of them {x['not_up_left_supply_window']} not up"
                if x["setting"] in SUPPLY_SETTINGS else f"{x['left_supply_window']} (reported)")
        lines.append(f"| {x['block']} | `{x['row']}` | {x['rule']} | {x['survived']}/{x['runs']} | {x['succeeded']} | "
                     f"{x['reading']} | {x['verdict']} | {x['repeat'] or '-'} | {left} |")
    lines.append(f"Condition 2: {doc['condition2']['holds']} ({doc['condition2']['reason']})")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------
def _csv(text: str | None) -> list[str]:
    return [x for x in (text or "").split(",") if x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.gate1", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dry-run", "run", "summary"):
        a = sub.add_parser(name)
        a.add_argument("--set", required=True)
        a.add_argument("--body", default=servo.BODY)
        a.add_argument("--root", default=str(setio.ROOT), help="the sets' root (default: the study's stage2/sets)")
        if name == "summary":
            a.add_argument("--out", required=True)
            continue
        a.add_argument("--rows", default=servo.SERVED_R0, help="served row bases: SV0, SV2/<seed> (comma-separated)")
        a.add_argument("--settings", default="@judged", help="settings and groups " + ", ".join(SETTING_GROUPS))
        a.add_argument("--delays", default=",".join(str(x) for x in DELAYS_MS), help="ms, comma-separated")
        a.add_argument("--conditions", default=CONDITION)
        a.add_argument("--episodes", type=int, default=bar.RUNS)
        a.add_argument("--cut-s", type=float, default=None,
                       help="for tests only: cut every episode to this many seconds (a smoke check; a cut episode "
                            "is never judged and its record is read by no rule)")
        if name == "dry-run":
            a.add_argument("--blocks", default=f"{HEADLINE},{SECOND}")
            a.add_argument("--cost", default=None, help="a COST.md with the bar's rates (default: the committed one)")
            a.add_argument("--out", default=None, help="a run directory whose records the plan reads (optional)")
            a.add_argument("--max-episodes", type=int, default=None,
                           help="a cap to state each block's episodes against (run refuses a plan over it)")
        else:
            a.add_argument("--out", required=True)
            a.add_argument("--block", default=HEADLINE)
            a.add_argument("--procs", type=int, default=1)
            a.add_argument("--traces", action="store_true", help="keep the episodes' traces")
            a.add_argument("--max-episodes", type=int, required=True,
                           help="the episode cap: a plan that would start more episodes is refused")
            a.add_argument("--dev-runs", default=str(DEV_RUNS),
                           help="the listing of development runs on V0, F25F9 and F25F9cl (default: "
                                "stage2/tb/gate1_dev_runs.json)")
    args = ap.parse_args(argv)
    if args.cmd == "summary":
        print(summary_text(summarize(args.out, args.set, args.body)))
        return 0
    kw = dict(settings=_csv(args.settings), delays_ms=[int(x) for x in _csv(args.delays)],
              conditions=_csv(args.conditions), episodes=args.episodes, cut_s=args.cut_s, root=args.root)
    if args.cmd == "dry-run":
        dry_run(args.set, args.body, _csv(args.rows), blocks=_csv(args.blocks),
                rates=None if args.cost is None else bar.read_cost(args.cost), out=args.out,
                max_episodes=args.max_episodes, **kw)
        return 0
    command = " ".join(["python -m designlab.stage2.gate1"] + [str(x) for x in (sys.argv[1:] if argv is None else argv)])
    doc = run(args.set, args.body, args.out, _csv(args.rows), block=args.block, procs=args.procs, traces=args.traces,
              max_episodes=args.max_episodes, dev_runs=args.dev_runs, command=command, **kw)
    print(summary_text(doc))
    print(json.dumps(dict(session=doc["session"]), default=str)[:4000], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONDITION", "CORNERS", "DELAYS_MS", "DEV_RUNS", "DRAWN", "GateWorker", "NO_REPEAT", "REPEAT_VERDICT", "RULES",
           "SCHEMA", "SETTING_GROUPS", "SLIPPERY", "SUPPLY_SETTINGS", "check_out", "check_rows", "counts", "cut_episode",
           "dry_run", "headline_passed", "list_dev_run", "plan", "reading", "record_path", "records", "repeat_word",
           "registered_set", "row_names", "run", "runner_sources", "settings_of", "summarize", "summary_text", "verdict"]
