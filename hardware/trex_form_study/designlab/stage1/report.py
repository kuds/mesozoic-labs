"""Stage 1 report tables (``STAGE1_PLAN.md`` section 8; section 3 step 9; section 7): machine-made tables from
whatever records a run directory holds, for the narrative report (written separately) to cite.

Run from hardware/trex_form_study::

    python -m designlab.stage1 report --out stage1/runs/2026-11-08 [--dest DIR] [--body ...]

It reads ``--out`` and writes only ``<dest>/STAGE1_TABLES.md`` and ``<dest>/stage1_tables.json`` (``--dest`` default:
``--out``).  Per body (``collect``):

* the verdict grid (``GRID_RULE``): behaviour x condition x the rows R0, R1, R2/42, R2/43, R3/42, R3/43 and R4/42,
  each cell the panels' record (``panels.record_path``; the policy rows' R3 and R4 records are ``policy_rows``'): pass,
  the survival and success counts with their Clopper-Pearson lower bounds, the failure label, gated or reported for
  that row, the checkpoint a policy row ran, and the interim flags (``FLAGS_RULE``); one table per behaviour group,
  and under the grid each row's provenance (``PROVENANCE_RULE``);
* the verdicts per controller (``VERDICTS_RULE``: ``panels.verdict`` on R2's and R3's two seeds' records, the controller
  the summary shows by ``panels.verdict_controller`` unless the body's R3 is judged with R2, ``train.R2_JUDGED_RULE``;
  R4's from its one seed by ``policy_rows.R4_VERDICT_RULE``, at most provisional) and gate G7 (``eval/g7.json``);
* the looks (``LOOKS_RULE``): sway against 0.12 H and head and tail motion against BEHAVIOR_SPEC's bands, per row on
  every walk and carpet walk (plan section 4: reported, not gated; decision 9);
* the margin ledger (``ledger``'s records, ``sweeps/<row>/<margin>.json``) for its rows (R2 of the body-level G5
  selection and R3/42, and any other row with records), with plan section 5's Stage 0 values beside
  (``LEDGER_TABLE_RULE``);
* each PPO run's G6 results with their plan consequences, budget used and judged checkpoint (``ppo/<seed>/run.json``,
  ``policy_only/42/run.json``; ``PPO_RULE``), the fit and G5 (``tune.json``, ``tune/<seed>/cma.json``; ``TUNE_RULE``),
  and the core-hours per step from the records' wall and CPU times (``COST_RULE``).

A record that does not exist shows as "not run"; a condition whose floor has no settled stance as "no stance"; a row a
condition does not apply to (R0 and R1 off the hard floor) as "n/a".  The JSON holds every value the tables show and
more; the Markdown keeps each table to one screen where it can.  ``stage1_tables.json`` is the body-level summary plan
section 7 lists as ``<body>/summary.json`` (STAGE1_PLAN.md decision 34): it holds every body's tables from every step.
"""

from __future__ import annotations

import datetime
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from . import BODIES, panels, tune

SCHEMA = 2
ROWS = ("R0", "R1", "R2/42", "R2/43", "R3/42", "R3/43", "R4/42")
HARD_ONLY = ("R0", "R1")
POLICY_CONTROLLERS = ("R3", "R4")
#: The controllers with verdicts from two seeds (``panels.VERDICT_ROWS``) and R4's one seed (``policy_rows.R4_ROWS``).
VERDICT_CONTROLLERS = tuple(panels.VERDICT_ROWS)
SINGLE_SEED = {"R4": ("R4/42",)}
#: Every controller the grid gives verdicts (``VERDICTS_RULE``).
CONTROLLERS = VERDICT_CONTROLLERS + tuple(SINGLE_SEED)
PPO_RUNS = (("R3", 42), ("R3", 43), ("R4", 42))
#: policy_rows.R4_FALLS_VERDICT (R4's settled verdict on falls; not imported here: policy_rows registers its rows).
R4_FALLS = "fails by falls (R4: not attributed)"
VERDICTS = ("shown", "provisional", "body-limited", "controller-limited", "not attributed", R4_FALLS, "incomplete",
            "not run")
BEHAVIOR_TITLES = {"stand": "Stand", "walk": "Walk", "turn": "Turn", "start_stop": "Start and stop", "push": "Push",
                   "carpet": "Carpet"}
MARKDOWN = "STAGE1_TABLES.md"
JSON_NAME = "stage1_tables.json"
#: A running PPO job writes run.json at every checkpoint (train.CHECKPOINT_EVERY steps): unwritten for this many
#: checkpoint intervals at its own rate (and at least STALE_MIN_S) it is marked 'stopped?' (``PPO_RULE``).
STALE_INTERVALS = 3.0
STALE_MIN_S = 3600.0
CARPET_NOTE = ("The tracker (R0-R2) was never fitted on carpet (decision 23: the fit and G5 ran on hard floor only, "
               "and R2 fell in every start tried on robot_floor_carpet and low pile): the carpet walk is reported for "
               "R0-R2, with its fall attribution, and gated for R3 and R4.")

GRID_RULE = (
    "one cell per row and condition from the panels' record <out>/<body>/eval/<row>/<behavior>/<condition>.json: "
    "'pass' or 'FAIL' (panels.PASS_RULE: 20/20 survive and at least 17/20 succeed) or 'unjudged' (fewer episodes, or "
    "an episode that failed to run), then survived,succeeded of the episodes and their one-sided 95 % Clopper-Pearson "
    "lower bounds (panels.LCB_RULE), as the record holds them; 'not run' if no record exists, 'no stance' if the "
    "floor has no settled stance for the body (panels.NOT_RUN_RULE: a result, not a missing record), 'n/a' where the "
    "row does not run (R0 and R1 run the hard floor only); a condition is gated or reported per row "
    "(panels.GATING_RULE: decision 23 reports the carpet walk for R0-R2); a failing record's label "
    "(panels.FAILURE_LABEL_RULE) is listed under the table")
FLAGS_RULE = (
    "a cell is flagged interim ('*') when its record is not judged (fewer than 20 episodes or an episode that failed "
    "to run), its falls wait for decision 20's attribution (eval --attribute: R2 and R3 only, panels.ATTRIBUTED; the "
    "falls of R0, R1 and R4 are not attributed by design, which is final and not flagged), its key's panel sources "
    "differ from the sources read now (panels.PANEL_SOURCES_RULE: made by other code), or its row is a policy row "
    "(R3, R4) whose checkpoint is not final: final when the run's run.json holds a final judged checkpoint "
    "(train.JUDGED_RULE, 'judge --final' or every run at its budget) at the step count the record names "
    "(policy_rows.CHECKPOINT_SELECTION_RULE: the record's key names the checkpoint by its steps and files' sha256, not "
    "by whether it was final); the cell names the checkpoint's steps")
VERDICTS_RULE = (
    "per condition, panels.verdict on each controller's two seeds' records (panels.VERDICT_ROWS: R2 from R2/42 and "
    "R2/43, R3 from R3/42 and R3/43; panels.VERDICT_RULE), 'not run' when neither record exists; the counts are over "
    "the conditions gated for the controller's rows; the controller the grid's verdict column shows is "
    "panels.verdict_controller's (R3 once its records exist, else R2, interim), except that a body whose R3 is judged "
    "with R2 (train.R2_JUDGED_RULE: seed 42 failed G6c, plan section 3 step 7, or FIX_RULE stopped one of its R3 runs, "
    "decision 32) shows R2's, flagged, and its R3 column reads 'not judged'; until seed 42 reaches G6c the note says "
    "'G6c not reached'; R4 trains one seed (decisions 3 and 28), so its verdict is policy_rows.R4_VERDICT_RULE's on "
    "R4/42's record (panels.verdict with 'shown' read as 'provisional': at most provisional; falls, never attributed "
    "for R4, read as a settled 'fails by falls'; 'not run' when no record exists), counted over the conditions gated "
    "for R4/42, and its records' results on those conditions are counted beside")
PROVENANCE_RULE = (
    "under the grid, one line per row with records: the controller its records name (panels.ROWS_RULE: R0 feedforward "
    "and PD; R1 the body-level G5 selection; R2/s CMA-ES seed s's G5 selection; policy_rows.POLICY_ROWS_RULE: R3/s the "
    "body-level selection plus PPO seed s's residual policy, R4/s PPO seed s's policy-only policy) and, for a policy "
    "row, each checkpoint its records ran with final or provisional (FLAGS_RULE)")
LOOKS_RULE = (
    "per row and walk or carpet walk with a record: panels.LOOK_RULE's readings over the episodes' judged windows "
    "(reported, not gated: plan section 4, decision 9): sway, the mean of the episodes' mean per-stride peak to peak "
    "and the largest of their largest, / H, against BEHAVIOR_SPEC's 0.12 H; head (neck_pitch) and tail (tail_yaw) "
    "amplitude, the mean and range over the episodes, and how many lie in BEHAVIOR_SPEC's band (head 0.05-0.15 rad, "
    "tail 0.10-0.20 rad); episodes that fell before their window have no reading")
LEDGER_TABLE_RULE = (
    "per margin and search (ledger.RULES), each ledger row's record <out>/<body>/sweeps/<row>/<margin>.json: the "
    "threshold on success ('works': the last value at which no seed falls or fails a term it passed at nominal, "
    "ledger.WORKS_RULE) and on survival ('survives'), each with the nearest value that fails and the label at it "
    "(ledger.LABEL_RULE, ledger.LIFT_RULE), or 'fails at nominal', 'works across the interval' (to its far end), 'not "
    "run'; points show works (and outright) and survives; the trunk-mass rows the whole robot's mass change beside the "
    "factor (ledger.MASS_RULE); plan section 5's Stage 0 values beside (ledger.STAGE0_RULE: the drives' with the value "
    "the ledger's own limit compares); a record is flagged '*' when its search is not finished, its key names other "
    "ledger sources than ledger.py now (ledger.SOURCES_RULE), it was made on other panel sources or key context than "
    "now, or (a tracker row) its panel keys differ from the ones computed now, or when a search fails at nominal on "
    "a condition the panels' 20-seed record passes; a policy row's results (R3/42) name the checkpoints its records "
    "ran, listed under the table and marked provisional when one is not final (FLAGS_RULE)")
PPO_RULE = (
    "per run (R3 seeds 42 and 43, R4 seed 42) its run.json: status, env steps trained of the budget "
    "(train.BUDGET_RULE), the last checkpoint, the judged checkpoint (train.JUDGED_RULE: steps, final or provisional; "
    "train.judged_checkpoint's current value marked 'not yet recorded' when run.json holds none), G6a (zero residual "
    "identical; env steps per wall second against 600), G6b, G6b after the one fix and G6c (train.G6_RULE, "
    "train.FIX_RULE: at what step, pass, the monitor's mean survival and the conditions R3 beats R2 on), each FAIL "
    "with the plan's consequence (G6a: the steps cut equally, decision 26; G6b: make one fix, then stop, decision 32; "
    "G6c: skip seed 43 and judge with R2), the job-hours of its sessions (the sum of their wall seconds, finished or "
    "to run.json's last write) and run.json's last write, a run whose status is 'running' marked 'stopped?' when "
    f"run.json has not been written for {STALE_INTERVALS:g} checkpoint intervals at the run's own rate (its last "
    "session's steps over that session's time to the last write; else its first rollouts' rate) and at least "
    f"{STALE_MIN_S / 60:g} min: run.json is written at checkpoints, not at every rollout; per body the elapsed wall "
    "hours, the union of every run's session intervals (plan section 6: wall clock per body; the runs share the "
    "cores, so their job-hours overlap)")
TUNE_RULE = (
    "tune.json: per CMA-ES seed the fit's status, evaluations and final sigma, the candidate G5 selected and its "
    "validation counts per condition (successes, survived of 20), the seed's G5 pass and label, the body's G5 and the "
    "body-level selection (env.GAINS_RULE: the CMA-ES seed R1 and R3 use)")
COST_RULE = (
    "core-hours per step from the records: prepare (refs.json and transitions.json seconds: wall, one process), gates "
    "(gates.json G1-G4 seconds: wall, one process), the fit (each seed's cma.json sessions, end - start x workers; "
    "evals.jsonl's evaluation wall seconds beside), G5 (tune.json seconds; its worker count is not recorded: one "
    "process assumed, a lower bound), PPO (run.json throughput: each finished session's CPU seconds; a session still "
    "running has no total yet: its wall to the record's last write x its first rollouts' CPU per wall second, an "
    "estimate; its wall hours the elapsed union of the sessions, its job-hours beside), the panels (each record's "
    "episode wall seconds and its attribution's: one process per episode, an upper bound on CPU) and the ledger (the "
    "CPU seconds of each evaluation once, by its key: an evaluation several margins share is in each of their "
    "records); core-hours are CPU hours where recorded, else wall hours x processes")
RULES = dict(grid=GRID_RULE, flags=FLAGS_RULE, verdicts=VERDICTS_RULE, provenance=PROVENANCE_RULE, looks=LOOKS_RULE,
             ledger=LEDGER_TABLE_RULE, ppo=PPO_RULE, tune=TUNE_RULE, cost=COST_RULE)


def _read(path: Path) -> dict | None:
    return panels._read(path)


def _now() -> str:
    return tune._now()


def _f(x, digits: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def _g(x) -> str:
    return "-" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


def _hours(s) -> float | None:
    return None if s is None else float(s) / 3600.0


def _parse_time(s: str | None):
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s)
    except ValueError:
        return None


def _mtime(path: Path) -> datetime.datetime | None:
    try:
        return datetime.datetime.fromtimestamp(os.stat(path).st_mtime, datetime.timezone.utc)
    except OSError:
        return None


def checkpoint_final(steps, judged: dict | None) -> bool:
    """``FLAGS_RULE``: a record's checkpoint is final when its run's run.json holds a final judged checkpoint at the
    same step count (as ``policy_rows.checkpoint_final``)."""
    return bool(steps is not None and judged and judged.get("final") and judged.get("steps") == steps)


# ------------------------------------------------------------------------------------------------
# The verdict grid
# ------------------------------------------------------------------------------------------------
def _policy_runs(out: Path, body: str) -> dict:
    """Each policy row's run.json judged checkpoint (``FLAGS_RULE``)."""
    out_ = {}
    for mode, seed in PPO_RUNS:
        doc = _read(run_json(out, body, mode, seed)) or {}
        jc = doc.get("judged_checkpoint")
        out_[f"{mode}/{seed}"] = dict(judged_checkpoint=jc, steps=(jc or {}).get("steps"),
                                      final=bool(jc and jc.get("final")))
    return out_


def _policy_checkpoint(controller: dict | None) -> dict | None:
    """The checkpoint a policy row's record ran, as ``policy_rows`` describes it (the controller's policy's
    checkpoint: steps, model and VecNormalize with their sha256), or None."""
    if not isinstance(controller, dict):
        return None
    pol = controller.get("policy")
    ck = pol.get("checkpoint") if isinstance(pol, dict) else None
    return ck if isinstance(ck, dict) else None


def _checkpoint_of(controller: dict | None):
    """A policy row's checkpoint as its record's controller description names it: the step count of
    ``policy_rows``' checkpoint, else any key with 'checkpoint' that holds a value."""
    ck = _policy_checkpoint(controller)
    if ck is not None:
        return ck.get("steps")
    if not isinstance(controller, dict):
        return None
    for d in (controller, controller.get("policy") if isinstance(controller.get("policy"), dict) else {}):
        for k, v in d.items():
            if "checkpoint" in k and not isinstance(v, (dict, list)):
                return v
    return None


def cell(rec: dict | None, row: str, cond: panels.Condition, sources_now: str | None, policy_runs: dict) -> dict:
    """``GRID_RULE``'s cell of one row on one condition, with its flags (``FLAGS_RULE``)."""
    gated = panels.gated_for(row, cond)
    if row in HARD_ONLY and cond.floor != "hard":
        return dict(status="n/a", gated=gated, text="n/a")
    if rec is None:
        return dict(status="not run", gated=gated, text="not run")
    if rec.get("status") == "not run":
        return dict(status="no stance", gated=gated, text="no stance", reason="no settled stance (panels.NOT_RUN_RULE)")
    c = rec.get("counts") or {}
    f = rec.get("failure") or {}
    ctrl = panels.controller_of(row)
    flags = []
    if not c.get("judged"):
        flags.append("not judged")
    if c.get("errors"):
        flags.append(f"{c['errors']} failed to run")
    unattributed = (f.get("label") or "").startswith("not attributed") or (f.get("falls_label") or "").startswith(
        "not attributed")
    by_design = bool(unattributed and ctrl not in panels.ATTRIBUTED)
    if unattributed and not by_design:
        flags.append("not attributed")
    if sources_now is not None and (rec.get("key") or {}).get("sources") not in (None, sources_now):
        flags.append("other panel sources")
    steps = _checkpoint_of(rec.get("controller")) if ctrl in POLICY_CONTROLLERS else None
    final = None
    if ctrl in POLICY_CONTROLLERS:
        final = checkpoint_final(steps, (policy_runs.get(row) or {}).get("judged_checkpoint"))
        if not final:
            flags.append("checkpoint provisional")
    result = "unjudged" if c.get("pass_") is None else ("pass" if c["pass_"] else "FAIL")
    n, surv, succ = c.get("episodes"), c.get("survived"), c.get("succeeded")
    text = (f"{result} {surv},{succ}" + (f" of {n}" if n != panels.RUNS else "")
            + f" ({_f(c.get('survival_lcb'))},{_f(c.get('success_lcb'))})" + (" *" if flags else ""))
    return dict(status="done", gated=gated, record_gated=rec.get("gated"), result=result, episodes=n, survived=surv,
                succeeded=succ, errors=c.get("errors"), survival_lcb=c.get("survival_lcb"),
                success_lcb=c.get("success_lcb"), judged=c.get("judged"), pass_=c.get("pass_"), label=f.get("label"),
                kind=f.get("kind"), falls_label=f.get("falls_label"), success_label=f.get("success_label"),
                untestable=f.get("untestable") or None, flags=flags, attribution_by_design=by_design or None,
                text=text, checkpoint=steps, checkpoint_final=final,
                wall_s=rec.get("wall_s"), episode_wall_s=rec.get("episode_wall_s"),
                attribution_wall_s=(rec.get("attribution") or {}).get("wall_s"))


def shown_controller(out: Path, body: str, found) -> tuple[str, str, dict]:
    """``VERDICTS_RULE``: the controller whose verdicts the grid shows, why, and the body's R2_JUDGED_RULE reading."""
    from . import train

    jw = train.judged_with_r2(out, body)
    if jw["with_r2"]:
        return "R2", f"R2: {body}'s R3 is judged with R2 ({jw['reason']}; train.R2_JUDGED_RULE)", jw
    shown, note = panels.verdict_controller(found)
    g = jw["g6c"]
    if g["state"] == "passed":
        note += f"; G6c passed at {g['at_steps']} steps"
    else:
        note += "; G6c not reached (plan section 3 step 7: R3's verdicts stand only if seed 42 passes it)"
    return shown, note, jw


def _provenance(row: str, recs: list[dict], policy_runs: dict) -> dict:
    """``PROVENANCE_RULE``: what a row's records name as its controller, and a policy row's checkpoints."""
    ctrls = Counter()
    steps = set()
    for rec in recs:
        c = rec.get("controller") or {}
        text = c.get("controller") or c.get("row") or row
        cand = (c.get("gains_info") or {}).get("candidate")
        if cand and cand not in text:
            text += f" ({cand})"
        ctrls[text] += 1
        s = _checkpoint_of(c) if panels.controller_of(row) in POLICY_CONTROLLERS else None
        if s is not None:
            steps.add(s)
    jc = (policy_runs.get(row) or {}).get("judged_checkpoint")
    cks = [dict(steps=s, final=checkpoint_final(s, jc)) for s in sorted(steps, key=str)]
    return dict(controllers=dict(ctrls), checkpoints=cks, records=len(recs))


def grid(out: Path, body: str, sources_now: str | None = None) -> dict:
    """The verdict grid of a body (``GRID_RULE``, ``VERDICTS_RULE``, ``PROVENANCE_RULE``)."""
    from .policy_rows import r4_verdict

    edir = Path(out) / body / "eval"
    found = sorted({"/".join(q.relative_to(edir).parts[:-2]) for q in edir.glob("**/*.json")
                    if len(q.relative_to(edir).parts) >= 3 and q.relative_to(edir).parts[0] != "traces"})
    rows = list(ROWS) + [r for r in found if r and r not in ROWS]
    shown, note, jw = shown_controller(out, body, found)
    runs = _policy_runs(out, body)
    behaviors, counts, single = {}, {c: Counter() for c in CONTROLLERS}, {c: Counter() for c in SINGLE_SEED}
    by_row: dict[str, list] = {}
    for cond in panels.conditions():
        recs = {r: _read(panels.record_path(out, body, r, cond)) for r in rows}
        for r, x in recs.items():
            if x is not None and x.get("status") == "done":
                by_row.setdefault(r, []).append(x)
        cells = {r: cell(recs[r], r, cond, sources_now, runs) for r in rows}
        verdicts = {}
        for ctrl, pair in panels.VERDICT_ROWS.items():
            rs = [recs.get(r) if r in recs else _read(panels.record_path(out, body, r, cond)) for r in pair]
            g = panels.gated_for(pair[0], cond)
            v = dict(verdict="not run") if all(x is None for x in rs) else panels.verdict(rs)
            judged = not (ctrl == "R3" and jw["with_r2"])
            verdicts[ctrl] = dict(verdict=v["verdict"], gated=g, labels=v.get("labels"), passes=v.get("passes"),
                                  untestable=v.get("untestable") or None, judged=judged)
            if g and judged:
                counts[ctrl][v["verdict"]] += 1
        for ctrl, (r,) in SINGLE_SEED.items():
            x = cells[r]
            v = dict(verdict="not run") if recs[r] is None else r4_verdict([recs[r]])
            verdicts[ctrl] = dict(verdict=v["verdict"], gated=x["gated"], labels=v.get("labels"),
                                  passes=v.get("passes"), untestable=v.get("untestable") or None,
                                  read_from=v.get("read_from"), judged=True)
            if x["gated"]:
                single[ctrl][x.get("result") or x["status"]] += 1
                counts[ctrl][v["verdict"]] += 1
        behaviors.setdefault(cond.behavior, {})[cond.name] = dict(
            key=cond.key, plan_gated=cond.gated, floor=cond.floor, friction=cond.friction,
            gated_for=[r for r in ROWS if panels.gated_for(r, cond)], cells=cells, verdicts=verdicts,
            shown=verdicts[shown]["verdict"])
    provenance = {r: _provenance(r, by_row[r], runs) for r in rows if by_row.get(r)}
    return dict(rows=rows, found=found, verdict_controller=shown, verdict_controller_note=note, judged_with_r2=jw,
                behaviors=behaviors, gated_verdicts={c: dict(v) for c, v in counts.items()},
                single_seed={c: dict(v) for c, v in single.items()}, policy_runs=runs, provenance=provenance)


def g7(out: Path, body: str) -> dict:
    """Gate G7 as the panels recorded it (``eval/g7.json``), or not run."""
    doc = _read(Path(out) / body / "eval" / "g7.json")
    if not doc:
        return dict(status="not run")
    return dict(status="done", pass_=doc.get("pass_"), measured=doc.get("measured"), stage0=doc.get("stage0"),
                plan=doc.get("plan"), difference=doc.get("difference"), within=doc.get("within"),
                plan_rounding=doc.get("plan_rounding"))


# ------------------------------------------------------------------------------------------------
# The looks (reported, not gated)
# ------------------------------------------------------------------------------------------------
def looks(out: Path, body: str, rows=None) -> dict:
    """``LOOKS_RULE``: sway, head and tail per row on every walk and carpet walk with a record."""
    rows = list(rows or ROWS)
    out_ = {}
    for cond in panels.conditions():
        if cond.behavior not in ("walk", "carpet"):
            continue
        for r in rows:
            rec = _read(panels.record_path(out, body, r, cond))
            if not rec or rec.get("status") != "done":
                continue
            lk = [((e.get("measures") or {}).get("look") or {}) for e in rec.get("episodes") or ()]
            sway = [x["sway"] for x in lk if (x.get("sway") or {}).get("mean_H") is not None]
            entry = dict(episodes=len(lk), sway=dict(
                read=len(sway), mean_H=float(np.mean([s["mean_H"] for s in sway])) if sway else None,
                max_H=float(max(s["max_H"] for s in sway)) if sway else None, spec_H=panels.SWAY_SPEC_H))
            for part, (lo, hi) in panels.LOOK_BANDS.items():
                amps = [x[part]["amplitude_rad"] for x in lk if x.get(part)]
                entry[part] = dict(read=len(amps), mean_rad=float(np.mean(amps)) if amps else None,
                                   min_rad=float(min(amps)) if amps else None,
                                   max_rad=float(max(amps)) if amps else None,
                                   in_band=int(sum(lo <= a <= hi for a in amps)), band_rad=[lo, hi])
            out_.setdefault(cond.key, {})[r] = entry
    return dict(rule=LOOKS_RULE, conditions=out_)


# ------------------------------------------------------------------------------------------------
# The margin ledger
# ------------------------------------------------------------------------------------------------
def ledger_rows(out: Path, body: str) -> list[str]:
    """The ledger's rows of a body: R2 of the body-level G5 selection and R3/42 (``ledger.ROWS_RULE``), then any other
    row with records."""
    from . import env, ledger

    rows = []
    try:
        _, info = env.tuned_gains(Path(out) / body, body)
        if info.get("cma_seed") is not None:
            rows.append(f"R2/{int(info['cma_seed'])}")
    except (OSError, ValueError, KeyError):
        pass
    rows.append(ledger.R3_ROW)
    sdir = Path(out) / body / "sweeps"
    for q in sorted(sdir.glob("*/**/*.json")):
        rec = _read(q)
        if rec and rec.get("row") and rec["row"] not in rows:
            rows.append(rec["row"])
    return rows


def current_panel_keys(out: Path, body: str, row: str, cond_keys, cut_s: float | None = None) -> dict | None:
    """``LEDGER_TABLE_RULE``: the ledger's panel keys of a tracker row (R0, R1, R2/s) on these conditions (cut at
    ``cut_s``, as a smoke sweep cuts them) as ``ledger.Session`` would compute them now, or None when they cannot be (a
    policy row, or the inputs not readable)."""
    import dataclasses

    from . import ledger

    if panels.controller_of(row) not in ("R0", "R1", "R2"):
        return None
    try:
        run_dir = Path(out) / body
        inputs, docs = tune.read_inputs(run_dir, panels.panel_files())
        planner = panels.Planner(run_dir, docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
        r = panels.tracker_rows(body, run_dir, (row,))[row]
        by_key = {c.key: c if cut_s is None else dataclasses.replace(c, seconds=float(cut_s))
                  for c in panels.conditions()}
        src, ctx = panels.panel_sources()["sha256"], panels.key_context(body)
        return {k: ledger._sha(panels.record_key(body, r, by_key[k], planner.resolve(by_key[k]), ledger.SEEDS, inputs,
                                                 src, ctx)) for k in cond_keys}
    except (OSError, ValueError, KeyError, tune.InputError):
        return None


def _record_flags(rec: dict, now: dict, panel_keys_now: dict | None) -> list[str]:
    """``LEDGER_TABLE_RULE``: why a ledger record may not be current."""
    flags = []
    key = rec.get("key") or {}
    if key.get("ledger") not in (None, now["ledger"]):
        flags.append("other ledger sources")
    if rec.get("panel_sources") not in (None, now["panel_sources"]):
        flags.append("other panel sources")
    if rec.get("key_context") not in (None, now["key_context"]):
        flags.append("other key context")
    if panel_keys_now is not None and key.get("panel_keys") not in (None, panel_keys_now):
        flags.append("other panel keys")
    return flags


def ledger_table(out: Path, body: str) -> dict:
    """``LEDGER_TABLE_RULE``: per margin, each ledger row's results and Stage 0's values, with each record's flags."""
    from . import ledger, train

    rows = ledger_rows(out, body)
    runs = _policy_runs(out, body)
    ms = ledger.margins(body)
    recs = {(r, m.name): _read(ledger.record_path(out, body, r, m.name)) for m in ms for r in rows}
    now, keys_now = None, {}
    if any(recs.values()):  # (only with a record: the key context reads the plant's round-two files)
        now = dict(ledger=ledger.ledger_sources()["sha256"], panel_sources=None, key_context=None)
        try:
            now["panel_sources"] = panels.panel_sources()["sha256"]
            now["key_context"] = panels.fingerprint(now["panel_sources"], panels.key_context(body))
        except (OSError, ValueError):
            pass
        for r in rows:  # each row's panel keys now, once over every condition its records name (per cut)
            for cut in {rec.get("cut_s") for (rr, _), rec in recs.items() if rr == r and rec}:
                conds = sorted({k for (rr, _), rec in recs.items() if rr == r and rec and rec.get("cut_s") == cut
                                for k in ((rec.get("key") or {}).get("panel_keys") or {})})
                got = current_panel_keys(out, body, r, conds, cut) if conds else None
                keys_now[(r, cut)] = got or {k: None for k in conds}
    margins, notes = {}, []
    for m in ms:
        per_row = {}
        for r in rows:
            rec = recs[(r, m.name)]
            if rec is None:
                per_row[r] = dict(status="not run")
                continue
            ck = _policy_checkpoint(rec.get("controller"))
            if ck is not None:
                ck = dict(steps=ck.get("steps"),
                          final=checkpoint_final(ck.get("steps"), (runs.get(r) or {}).get("judged_checkpoint")))
            now_r = keys_now[(r, rec.get("cut_s"))]
            pk_now = {k: now_r.get(k) for k in ((rec.get("key") or {}).get("panel_keys") or {})}
            flags = _record_flags(rec, now, None if any(v is None for v in pk_now.values()) else pk_now)
            searches = (rec.get("summary") or {}).get("searches") or {}
            for sname, s in searches.items():
                for crit in ("success", "survival"):
                    b = (s or {}).get(crit) or {}
                    if b.get("flag"):
                        notes.append(f"{r} {m.name} {sname} ({crit}): {b['flag']}")
            if flags:
                notes.append(f"{r} {m.name}: " + ", ".join(flags))
            per_row[r] = dict(status=rec.get("status"), summary=rec.get("summary"), cut_s=rec.get("cut_s"),
                              halvings=rec.get("halvings"), cost=rec.get("cost"), flags=flags,
                              not_run=rec.get("not_run"), reference_stop=rec.get("reference_stop"),
                              panel_reading=rec.get("panel_reading"), checkpoint=ck, masses=rec.get("masses"))
        margins[m.name] = dict(title=m.title, swept=m.swept, definition=m.describe(),
                               stage0=ledger.stage0(body, m.name), rows=per_row)
    checkpoints = {}
    for r in rows:
        cks = {(x["checkpoint"]["steps"], x["checkpoint"]["final"]) for m in margins.values()
               for x in [m["rows"][r]] if x.get("checkpoint")}
        if cks:
            checkpoints[r] = [dict(steps=n, final=f) for n, f in sorted(cks, key=lambda t: (t[0] is None, t))]
    jw = train.judged_with_r2(out, body)
    return dict(rows=rows, margins=margins, checkpoints=checkpoints, notes=notes, current=now,
                r3_judged_with_r2=jw["reason"] if jw["with_r2"] else None)


# ------------------------------------------------------------------------------------------------
# PPO, the fit, costs
# ------------------------------------------------------------------------------------------------
def run_json(out: Path, body: str, mode: str, seed: int) -> Path:
    """A PPO run's record (``train.run_dir_of``)."""
    return Path(out) / body / ("ppo" if mode == "R3" else "policy_only") / str(int(seed)) / "run.json"


def _intervals(doc: dict, path: Path) -> list[tuple]:
    """A run's session intervals (start, end): a finished session's end, else run.json's last write."""
    t_last = _mtime(path)
    out = []
    for s in doc.get("sessions") or ():
        t0 = _parse_time(s.get("started_at"))
        t1 = _parse_time(s.get("ended_at")) or t_last
        if t0 is not None and t1 is not None and t1 > t0:
            out.append((t0, t1))
    return out


def _union_s(intervals) -> float:
    """The length of the union of time intervals (seconds)."""
    total, end = 0.0, None
    for t0, t1 in sorted(intervals):
        if end is None or t0 > end:
            total += (t1 - t0).total_seconds()
            end = t1
        elif t1 > end:
            total += (t1 - end).total_seconds()
            end = t1
    return total


def _ppo_times(doc: dict, path: Path) -> dict:
    """Wall and CPU seconds of a run's finished sessions, and the wall of a running one to the record's last write."""
    done = [t for t in doc.get("throughput") or () if not t.get("first_rollouts")]
    wall = float(sum(t.get("wall_s") or 0.0 for t in done))
    cpu = float(sum(t.get("cpu_s") or 0.0 for t in done))
    finished = {t.get("session") for t in done}
    running_wall = 0.0
    for s in doc.get("sessions") or ():
        if s.get("index") in finished or s.get("ended_at"):
            continue
        t0 = _parse_time(s.get("started_at"))
        t1 = _mtime(path)
        if t0 is not None and t1 is not None:
            running_wall += max(0.0, (t1 - t0).total_seconds())
    first = [t for t in doc.get("throughput") or () if t.get("first_rollouts")]
    f0 = first[0] if first else None
    return dict(finished_wall_s=wall, finished_cpu_s=cpu, running_wall_s=running_wall,
                env_processes=(f0 or {}).get("env_processes"),
                cpu_per_wall_first_rollouts=(f0["cpu_s"] / f0["wall_s"]) if f0 and f0.get("wall_s") else None,
                env_steps_per_wall_s=(first[-1].get("env_steps_per_wall_s") if first else None))


def _stale(doc: dict, path: Path, now: datetime.datetime) -> dict:
    """``PPO_RULE``: run.json's last write and whether a 'running' run looks stopped."""
    from . import train

    t = _mtime(path)
    if t is None:
        return dict(last_write=None, idle_s=None, stopped=None)
    idle = (now - t).total_seconds()
    rate = None
    last = (doc.get("sessions") or [None])[-1] or {}
    t0 = _parse_time(last.get("started_at"))
    if t0 is not None and doc.get("num_timesteps") is not None and (t - t0).total_seconds() > 0:
        done = int(doc["num_timesteps"]) - int(last.get("resumed_from") or 0)
        rate = done / (t - t0).total_seconds() if done > 0 else None
    rate = rate or _ppo_times(doc, path)["env_steps_per_wall_s"]
    every = int((doc.get("settings") or {}).get("checkpoint_every") or train.CHECKPOINT_EVERY)
    limit = max(STALE_MIN_S, STALE_INTERVALS * every / rate) if rate else max(STALE_MIN_S, 3 * 3600.0)
    return dict(last_write=t.isoformat(timespec="seconds"), idle_s=idle, rate_env_steps_s=rate, limit_s=limit,
                stopped=bool(doc.get("status") == "running" and idle > limit))


def ppo(out: Path, body: str, now: datetime.datetime | None = None) -> dict:
    """``PPO_RULE`` per run."""
    from . import train

    now = now or datetime.datetime.now(datetime.timezone.utc)
    rows = {}
    current = {}
    for mode, seed in PPO_RUNS:
        path = run_json(out, body, mode, seed)
        doc = _read(path)
        name = f"{mode}/{seed}"
        if doc is None:
            rows[name] = dict(status="not run")
            continue
        g6 = doc.get("g6") or {}
        zr, rate, b, c, bf = (g6.get(k) or {} for k in ("G6a_zero_residual", "G6a_rate", "G6b", "G6c",
                                                         "G6b_after_fix"))
        ck = doc.get("checkpoints") or []
        jc = doc.get("judged_checkpoint") or {}
        if not jc and mode not in current:
            try:
                current[mode] = train.judged_checkpoint(Path(out), mode)
            except (OSError, ValueError):
                current[mode] = None
        mon_b = (b.get("monitor") or {}).get("mean_survival_s") or {}
        mon_bf = (bf.get("monitor") or {}).get("mean_survival_s") or {}
        fix = doc.get("fix") or {}
        rows[name] = dict(
            status=doc.get("status"), mode=mode, seed=seed, steps=doc.get("num_timesteps"), budget=doc.get("budget"),
            checkpoints=len(ck), last_checkpoint=ck[-1] if ck else None,
            judged_checkpoint=dict(steps=jc.get("steps"), final=jc.get("final"), budget=jc.get("budget"),
                                   recorded=True) if jc else
            (dict(steps=current[mode]["steps"], final=current[mode]["final"], budget=current[mode]["budget"],
                  recorded=False) if current.get(mode) else None),
            g6a_zero_residual=zr.get("identical") if zr else None,
            g6a_rate=dict(env_steps_per_wall_s=rate.get("env_steps_per_wall_s"), pass_=rate.get("pass_"),
                          threshold=rate.get("threshold"), consequence=rate.get("consequence_if_failed"))
            if rate else None,
            g6b=dict(at_steps=b.get("at_steps"), pass_=b.get("pass_"), mean_survival_s=mon_b,
                     consequence=b.get("consequence_if_failed")) if b else None,
            fix=dict(from_checkpoint=fix.get("from_checkpoint"), overrides=fix.get("overrides")) if fix else None,
            g6b_after_fix=dict(at_steps=bf.get("at_steps"), pass_=bf.get("pass_"), mean_survival_s=mon_bf,
                               consequence=bf.get("consequence_if_failed")) if bf else None,
            stopped=doc.get("stopped"),
            g6c=dict(at_steps=c.get("at_steps"), pass_=c.get("pass_"), consequence=c.get("consequence_if_failed"),
                     beats_on=(c.get("monitor") or {}).get("policy_beats_zero_on")) if c else None,
            gains=(doc.get("gains") or {}).get("candidate"), times=_ppo_times(doc, path),
            job_wall_s=float(sum((t1 - t0).total_seconds() for t0, t1 in _intervals(doc, path))),
            write=_stale(doc, path, now))
    return rows


def ppo_wall(out: Path, body: str) -> dict:
    """``PPO_RULE``: a body's elapsed PPO wall time (the union of every run's session intervals) and its job-hours."""
    ivs, job = [], 0.0
    for mode, seed in PPO_RUNS:
        path = run_json(out, body, mode, seed)
        doc = _read(path)
        if doc:
            iv = _intervals(doc, path)
            ivs += iv
            job += sum((t1 - t0).total_seconds() for t0, t1 in iv)
    return dict(elapsed_s=_union_s(ivs), job_s=job)


def fit(out: Path, body: str) -> dict:
    """``TUNE_RULE``: the fit and G5 of a body."""
    from . import env

    doc = _read(Path(out) / body / "tune.json")
    if doc is None:
        return dict(status="not run")
    seeds = {}
    for s, v in (doc.get("seeds") or {}).items():
        run = (doc.get("runs") or {}).get(s) or {}
        seeds[s] = dict(fit_status=run.get("status"), evals=run.get("evals"), generations=run.get("generations"),
                        sigma=run.get("sigma"), selected=v.get("selected"), pass_=v.get("pass_"), label=v.get("label"),
                        refused=v.get("refused"),
                        conditions={k: dict(successes=x.get("successes"), survived=x.get("survived"),
                                            runs=x.get("runs"), pass_=x.get("pass_"))
                                    for k, x in (v.get("conditions") or {}).items()})
    try:
        _, info = env.tuned_gains(Path(out) / body, body)
        body_level = dict(cma_seed=info.get("cma_seed"), candidate=info.get("candidate"), status=info.get("status"))
    except (OSError, ValueError, KeyError) as exc:
        body_level = dict(error=str(exc))
    return dict(status="done", pass_=doc.get("pass_"), seeds_passing=doc.get("seeds_passing"), seeds=seeds,
                body_level=body_level, g5_seconds=doc.get("seconds"),
                objective=(doc.get("objective") or {}).get("name"))


def costs(out: Path, body: str) -> dict:
    """``COST_RULE``: wall and core hours per step of a body."""
    from . import ledger

    bdir = Path(out) / body
    steps = {}
    refs, trs = _read(bdir / "refs.json") or {}, _read(bdir / "transitions.json") or {}
    w = float(refs.get("seconds") or 0.0) + float(trs.get("seconds") or 0.0)
    steps["prepare"] = dict(wall_h=_hours(w), core_h=_hours(w), basis="wall, one process") if w else None
    gates = _read(bdir / "gates.json") or {}
    w = float(sum((gates.get(g) or {}).get("seconds") or 0.0 for g in ("G1", "G2", "G3", "G4")))
    steps["gates"] = dict(wall_h=_hours(w), core_h=_hours(w), basis="wall, one process") if w else None
    fit_wall = fit_core = ev_wall = 0.0
    for cma in sorted(bdir.glob("tune/*/cma.json")):
        doc = _read(cma) or {}
        for s in doc.get("sessions") or ():
            t0, t1 = _parse_time(s.get("started_at")), _parse_time(s.get("ended_at"))
            if t0 and t1:
                dt = (t1 - t0).total_seconds()
                fit_wall += dt
                fit_core += dt * int(s.get("workers") or 1)
        ev = cma.with_name("evals.jsonl")
        if ev.exists():
            for line in ev.read_text().splitlines():
                try:
                    ev_wall += float(json.loads(line).get("wall_s") or 0.0)
                except ValueError:
                    continue
    steps["fit"] = dict(wall_h=_hours(fit_wall), core_h=_hours(fit_core), evaluation_wall_h=_hours(ev_wall),
                        basis="sessions' end - start x workers") if fit_wall else None
    tj = _read(bdir / "tune.json") or {}
    steps["g5"] = (dict(wall_h=_hours(tj["seconds"]), core_h=_hours(tj["seconds"]),
                        basis="tune.json seconds, one process assumed (workers not recorded): a lower bound")
                   if tj.get("seconds") else None)
    p = ppo(out, body)
    cpu = sum(r["times"]["finished_cpu_s"] for r in p.values() if r.get("times"))
    job = sum(r["times"]["finished_wall_s"] + r["times"]["running_wall_s"] for r in p.values() if r.get("times"))
    running = [n for n, r in p.items() if r.get("times") and r["times"]["running_wall_s"]]
    est = sum(r["times"]["running_wall_s"] * (r["times"]["cpu_per_wall_first_rollouts"] or 1.0)
              for r in p.values() if r.get("times") and r["times"]["running_wall_s"])
    wall = ppo_wall(out, body)
    basis = ("finished sessions' CPU, and for a running session its wall to the record's last write x the CPU per wall "
             "second of its first rollouts (an estimate)" if running else "sessions' CPU")
    steps["ppo"] = dict(wall_h=_hours(wall["elapsed_s"]), job_h=_hours(job), core_h=_hours(cpu + est),
                        finished_cpu_h=_hours(cpu), running_estimate_h=_hours(est) if running else None,
                        running=running,
                        basis=basis + "; wall: the elapsed union of the sessions, job-hours beside") if job else None
    ep = at = 0.0
    n_rec = 0
    for q in (bdir / "eval").glob("**/*.json") if (bdir / "eval").exists() else ():
        rel = q.relative_to(bdir / "eval").parts
        if rel[0] == "traces" or len(rel) < 3:
            continue
        rec = _read(q) or {}
        ep += float(rec.get("episode_wall_s") or 0.0)
        at += float((rec.get("attribution") or {}).get("wall_s") or 0.0)
        n_rec += 1
    sm = _read(bdir / "eval" / "summary.json") or {}
    sess_wall = float(sum(s.get("wall_s") or 0.0 for s in sm.get("sessions") or ()))
    steps["panels"] = dict(wall_h=_hours(sess_wall) if sess_wall else None, core_h=_hours(ep + at), records=n_rec,
                           basis="episodes' and attributions' wall seconds (one process each): an upper bound on CPU"
                           ) if n_rec else None
    seen: dict[str, dict] = {}
    n_l = 0
    for q in (bdir / "sweeps").glob("*/**/*.json") if (bdir / "sweeps").exists() else ():
        rec = _read(q) or {}
        if not rec.get("margin"):
            continue
        n_l += 1
        evals = rec.get("evaluations")
        if evals is None:  # a record without its evaluations: its own cost, once
            seen[f"record:{q}"] = dict(cpu_s=(rec.get("cost") or {}).get("cpu_s"),
                                       wall_s=(rec.get("cost") or {}).get("wall_s"))
            continue
        for i, e in evals.items():
            seen.setdefault(e.get("valid") or f"{rec.get('row')}:{i}", e)
    lc = float(sum(float(e.get("cpu_s") or 0.0) for e in seen.values()))
    lw = float(sum(float(e.get("wall_s") or 0.0) for e in seen.values()))
    ls = _read(bdir / "sweeps" / "summary.json") or {}
    l_wall = float(sum(s.get("wall_s") or 0.0 for s in ls.get("sessions") or ()))
    steps["ledger"] = dict(wall_h=_hours(l_wall) if l_wall else None, core_h=_hours(lc), episode_wall_h=_hours(lw),
                           records=n_l, evaluations=len(seen),
                           basis="episodes' CPU seconds, each evaluation once (by its key) however many margins use it",
                           rule=ledger.COST_RULE) if n_l else None
    total = sum((s or {}).get("core_h") or 0.0 for s in steps.values())
    return dict(steps=steps, core_h=total)


# ------------------------------------------------------------------------------------------------
# The document
# ------------------------------------------------------------------------------------------------
def collect(out: Path, bodies=None) -> dict:
    """Every table's data for ``bodies`` (default: every body with a directory in ``out``)."""
    out = Path(out)
    bodies = [b for b in (bodies or BODIES) if (out / b).is_dir()] if bodies is None else list(bodies)
    try:
        sources_now = panels.panel_sources()["sha256"]
    except (OSError, ValueError):
        sources_now = None
    doc = dict(schema=SCHEMA, generated_at=_now(), out=str(out), rules=RULES, panel_sources=sources_now, bodies={})
    for b in bodies:
        doc["bodies"][b] = dict(grid=grid(out, b, sources_now), looks=looks(out, b), g7=g7(out, b),
                                ledger=ledger_table(out, b), ppo=ppo(out, b), ppo_wall=ppo_wall(out, b),
                                fit=fit(out, b), costs=costs(out, b))
    doc["core_h"] = float(sum(x["costs"]["core_h"] for x in doc["bodies"].values()))
    return doc


def _cell_label(c: dict) -> str | None:
    if c.get("status") != "done" or c.get("pass_") is not False and not c.get("flags"):
        return None
    parts = []
    if c.get("pass_") is False:
        parts.append((c.get("label") or "no label") + (" (by design)" if c.get("attribution_by_design")
                                                       and (c.get("label") or "").startswith("not attributed")
                                                       else ""))
        other = c.get("success_label") if c.get("kind") == "falls" else c.get("falls_label")
        if other and other != c.get("label"):
            parts.append(f"also {other}")
    if c.get("flags"):
        parts.append("flags: " + ", ".join(c["flags"]))
    if c.get("checkpoint") is not None:
        parts.append(f"checkpoint {c['checkpoint']}")
    return "; ".join(parts)


def _grid_md(g: dict) -> list[str]:
    rows = [r for r in g["rows"]]
    shown = g["verdict_controller"]
    jw = g.get("judged_with_r2") or {}
    lines = [f"Verdicts shown: {g['verdict_controller_note']}.  Cells: result, survived,succeeded of 20 (their "
             "one-sided 95 % Clopper-Pearson lower bounds); '(rep.)' reported, not gated, for that row; '*' interim "
             "(listed below each table); 'not run' no record; 'no stance' the floor has no settled stance for the body "
             "(panels.NOT_RUN_RULE); 'n/a' the row does not run there; R4's verdict is at most provisional (one seed), "
             "and its falls are never attributed (by design: decision 20's attribution is for R2 and R3).", ""]

    def head_of(c):
        if c == "R3" and jw.get("with_r2"):
            return "R3 verdict (not judged: " + ("G6c" if jw["g6c"]["state"] == "failed" else "stopped") + ")"
        return f"{c} verdict" + (" (shown)" if c == shown else "")

    for beh, conds in g["behaviors"].items():
        lines += [f"#### {BEHAVIOR_TITLES.get(beh, beh)}", ""]
        head = ["Condition", "Gated for"] + rows + [head_of(c) for c in CONTROLLERS]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        notes = []
        for name, c in conds.items():
            gated = c["gated_for"]
            gtxt = "all" if len(gated) == len(ROWS) else ("none (reported)" if not gated else ", ".join(gated))
            cells = []
            for r in rows:
                x = c["cells"].get(r) or dict(text="not run")
                t = x["text"]
                if x.get("status") == "done" and not x.get("gated"):
                    t += " (rep.)"
                cells.append(t)
                lab = _cell_label(x)
                if lab:
                    notes.append(f"- {r} {beh}/{name}: {lab}")
            vs = [c["verdicts"][v]["verdict"] + ("" if c["verdicts"][v]["gated"] else " (rep.)")
                  for v in CONTROLLERS]
            lines.append("| " + " | ".join([name, gtxt] + cells + vs) + " |")
        lines.append("")
        if beh == "carpet":
            lines += [CARPET_NOTE, ""]
        if notes:
            lines += notes + [""]
    if g.get("provenance"):
        lines += ["Provenance of each row (PROVENANCE_RULE):", ""]
        for r, p in g["provenance"].items():
            ctrl = "; ".join(f"{t}" + (f" ({n} records)" if len(p["controllers"]) > 1 else "")
                             for t, n in p["controllers"].items())
            cks = ", ".join(f"{_g(c['steps'])} steps ({'final' if c['final'] else 'provisional'})"
                            for c in p["checkpoints"])
            lines.append(f"- {r} = {ctrl}" + (f"; checkpoint {cks}" if cks else ""))
        lines.append("")
    return lines


def _looks_md(lk: dict) -> list[str]:
    conds = lk.get("conditions") or {}
    if not conds:
        return ["No walk or carpet record yet.", ""]
    lines = [f"Reported, not gated (LOOKS_RULE): sway against {panels.SWAY_SPEC_H:g} H, head and tail amplitude "
             f"against BEHAVIOR_SPEC's bands (head {panels.LOOK_BANDS['head'][0]:g}-{panels.LOOK_BANDS['head'][1]:g} "
             f"rad, tail {panels.LOOK_BANDS['tail'][0]:g}-{panels.LOOK_BANDS['tail'][1]:g} rad).", "",
             "| Condition | Row | Sway mean / max (H) | Head amplitude mean [range] (rad), in band | Tail amplitude "
             "mean [range] (rad), in band |", "|---|---|---|---|---|"]
    for key, rows in conds.items():
        for r, x in rows.items():
            s = x["sway"]
            sway = "-" if s["mean_H"] is None else f"{s['mean_H']:.3f} / {s['max_H']:.3f}"

            def amp(p):
                return ("-" if p["mean_rad"] is None else
                        f"{p['mean_rad']:.3f} [{p['min_rad']:.3f}-{p['max_rad']:.3f}], {p['in_band']}/{p['read']}")

            lines.append(f"| {key} | {r} | {sway} | {amp(x['head'])} | {amp(x['tail'])} |")
    return lines + [""]


def _ledger_cell(x: dict | None, search: str, kind: str | None, masses: dict | None = None) -> tuple[str, str]:
    """(works, survives) text of one search of one row."""
    if not x or x.get("status") == "not run":
        return "not run", "not run"
    s = ((x.get("summary") or {}).get("searches") or {}).get(search)
    if s is None:
        return "not run", "not run"
    if s.get("status") == "not run":
        return "n/a", "n/a"
    out = []
    for crit in ("success", "survival"):
        b = s.get(crit) or {}
        st = b.get("status")
        if st == "bisected":
            t = f"{_value(b.get('threshold'), kind, masses)} (fails {_value(b.get('fails'), kind)}"
            t += f"; {b['label']})" if b.get("label") else ")"
        elif st == "fails at nominal":
            t = "fails at nominal" + (f" ({b['label']})" if b.get("label") else "")
            pr = b.get("panels")
            if pr:
                t += (f" [panels: {'pass' if pr.get('pass_') else 'FAIL'} {pr.get('survived')},{pr.get('succeeded')}"
                      + (f"; {pr['label']}" if pr.get("label") else "") + "]")
        elif st == "works across the interval":
            t = f"works to {_value(b.get('threshold'), kind, masses)}"
        else:
            t = st or "not run"
        if b.get("search") and search in ("turn",):
            t += f" [{b['search']}]"
        if b.get("flag"):
            t += " *"
        out.append(t)
    if x.get("status") == "partial" or x.get("flags"):
        out = [t + " *" for t in out]
    return out[0], out[1]


def _value(x, kind: str | None, masses: dict | None = None) -> str:
    if x is None:
        return "?"
    if kind == "latency_steps":
        return f"{int(x)} steps ({2 * int(x)} ms)"
    if kind == "mass" and masses and masses.get("trunk_share"):
        return f"{x:.4g} (robot {100 * (x - 1) * masses['trunk_share']:+.1f} %)"
    return f"{x:.4g}"


def _stage0_text(st: dict, search: str) -> str:
    v = st.get("values") or {}
    if search in v:
        t = _g(v[search])
        if (st.get("without_margin") or {}).get(search) is not None:
            t += f" ({_g(st['without_margin'][search])} without the 15 % margin)"
        if (st.get("walk_rms") or {}).get(search) is not None:
            t += f" (walk RMS {_g(st['walk_rms'][search])})"
        return t
    if st.get("forward_bound_m_s") is not None:
        return f"(forward {_g(st['forward_bound_m_s'])})"
    if len(v) == 1:
        return _g(next(iter(v.values())))
    return "-"


def _ledger_md(lt: dict) -> list[str]:
    rows = lt["rows"]
    head = ["Margin", "Search"] + [f"{r} works" for r in rows] + [f"{r} survives" for r in rows] + ["Stage 0"]
    lines = ["Cells: the threshold (the last value at which all 3 seeds survive and none fails a term it passed at "
             "nominal, or survive) and the nearest value that fails with the label there (falls labelled by lifting "
             "each candidate limit on the 3 seeds, ledger.LIFT_RULE); '*' the search is not finished or the record is "
             "flagged (listed below).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    extra = []
    if lt.get("r3_judged_with_r2"):
        extra.append(f"- R3 is judged with R2 for this body ({lt['r3_judged_with_r2']}): its ledger row is not run "
                     "(ledger.ROWS_RULE)")
    for name, m in lt["margins"].items():
        d = m["definition"]
        kinds = {s["name"]: s["kind"] for s in d["searches"]}
        if d.get("not_run"):
            cells = ["not run (plan deviation, decision 33)"] * (2 * len(rows))
            lines.append("| " + " | ".join([m["title"], "-"] + cells + [_g((m["stage0"].get("values") or {}).get(
                "one_foot_m_s2"))]) + " |")
            ref = next(((x.get("reference_stop") or {}) for x in m["rows"].values()
                        if (x.get("reference_stop") or {}).get("peak_deceleration_m_s2") is not None), {})
            extra.append(f"- {m['title']}: {d['not_run']}"
                         + (f"; the existing stop reference decelerates at {ref['peak_deceleration_m_s2']:.3f} m/s^2 "
                            "(1 Hz low-pass)" if ref else ""))
            for r in rows:
                pr = (m["rows"].get(r) or {}).get("panel_reading")
                if pr and pr.get("acceleration_peak_max_m_s2") is not None:
                    extra.append(f"  - {r}'s start-and-stop panel: stops' acceleration peak at most "
                                 f"{pr['acceleration_peak_max_m_s2']:.3f} m/s^2, {pr['stopped_within']}/"
                                 f"{pr['stops']} stopped within 1.5 strides")
            continue
        names = [s["name"] for s in d["searches"]] + [w[0] for w in d.get("worse") or ()]
        for sname in names:
            kind = kinds.get(sname) or next(iter(kinds.values()), None)
            works, surv = zip(*[_ledger_cell(m["rows"].get(r), sname, kind, (m["rows"].get(r) or {}).get("masses"))
                                for r in rows], strict=True)
            label = f"{sname} (worse of {', '.join(dict(d.get('worse') or ())[sname])})" if sname in dict(
                d.get("worse") or ()) else sname
            lines.append("| " + " | ".join([m["title"], label] + list(works) + list(surv)
                                           + [_stage0_text(m["stage0"], sname)]) + " |")
        for p in d.get("points") or ():
            cells_w, cells_s = [], []
            for r in rows:
                x = m["rows"].get(r) or {}
                pt = (((x.get("summary") or {}).get("points") or {}).get(p["name"])
                      if x.get("status") != "not run" else None)
                if not pt or pt.get("status") in (None, "not run"):
                    cells_w.append("not run")
                    cells_s.append("not run")
                    continue
                if pt.get("status") != "done":
                    cells_w.append(pt.get("status"))
                    cells_s.append(pt.get("status"))
                    continue
                lab = f" ({pt['label']})" if pt.get("label") else ""
                same = " = nominal plant" if pt.get("same_as_nominal") else ""
                verdict = "works" if pt.get("works") else "fails"
                outright = "" if pt.get("works_outright") is None or pt.get("works_outright") == pt.get("works") else (
                    f"; outright {'works' if pt['works_outright'] else 'fails'}")
                cells_w.append(f"{verdict} {pt.get('succeeded')}/3{same}{outright}" + ("" if pt.get("works") else lab))
                cells_s.append(f"{'survives' if pt.get('survives') else 'falls'} {pt.get('survived')}/3")
            title = p["name"]
            if p.get("kind") == "mass":
                ms = next(((m["rows"].get(r) or {}).get("masses") for r in rows if (m["rows"].get(r) or {}).get(
                    "masses")), None)
                if ms:
                    title += f" (trunk x{p['value']:g}: robot {100 * (p['value'] - 1) * ms['trunk_share']:+.1f} %)"
            lines.append("| " + " | ".join([m["title"], title] + cells_w + cells_s + ["-"]) + " |")
        if m["stage0"].get("text") and name in ("voltage", "drive_speed", "drive_torque", "grip"):
            extra.append(f"- {m['title']}: Stage 0: {m['stage0']['text']}")
    for r, cks in (lt.get("checkpoints") or {}).items():
        extra.append(f"- {r} ran checkpoint " + ", ".join(
            f"{_g(c['steps'])} ({'final' if c['final'] else 'provisional'})" for c in cks)
            + ("" if all(c["final"] for c in cks) else " *: not the final judged checkpoint (FLAGS_RULE)"))
    for n in lt.get("notes") or ():
        extra.append(f"- * {n}")
    lines.append("")
    return lines + extra + ([""] if extra else [])


def _gate(x: dict | None, what: str) -> str:
    if not x:
        return "-"
    if x.get("pass_") is None:
        return "-"
    t = f"{'pass' if x['pass_'] else 'FAIL'} at {_g(x.get('at_steps'))}"
    surv = x.get("mean_survival_s") or {}
    if surv:
        t += " (survival " + ", ".join(f"{k} {v:.2f} s" for k, v in surv.items()) + ")"
    if what == "c" and x.get("beats_on"):
        t += f" (beats R2 on {', '.join(x['beats_on'])})"
    if not x["pass_"] and x.get("consequence"):
        t += f": {x['consequence']}"
    return t


def _ppo_md(p: dict, wall: dict | None = None) -> list[str]:
    head = ["Run", "Status", "Steps / budget", "Last checkpoint", "Judged checkpoint", "G6a zero residual",
            "G6a env steps/s", "G6b", "Fix; G6b after it", "G6c", "Job h", "CPU h", "run.json written"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in p.items():
        if r.get("status") == "not run":
            lines.append("| " + " | ".join([name, "not run"] + ["-"] * (len(head) - 2)) + " |")
            continue
        jc = r.get("judged_checkpoint")
        jtxt = "-" if not jc else (f"{_g(jc.get('steps'))} ({'final' if jc.get('final') else 'provisional'}"
                                   + ("" if jc.get("recorded", True) else ", not yet recorded") + ")")
        rate = r.get("g6a_rate") or {}
        fx = r.get("fix")
        ov = "; ".join(f"{k} " + (",".join(f"{x:g}" for x in v) if isinstance(v, list) else str(v))
                       for k, v in ((fx or {}).get("overrides") or {}).items())
        ftxt = "-" if not fx else f"from {_g(fx.get('from_checkpoint'))} ({ov}); " + _gate(r.get("g6b_after_fix"), "b")
        t = r.get("times") or {}
        w = r.get("write") or {}
        status = str(r.get("status")) + (" (stopped?)" if w.get("stopped") else "")
        lines.append("| " + " | ".join([
            name, status, f"{_g(r.get('steps'))} / {_g(r.get('budget'))}", _g(r.get("last_checkpoint")),
            jtxt, _f(r.get("g6a_zero_residual")) if r.get("g6a_zero_residual") is not None else "-",
            "-" if not rate else
            f"{_f(rate.get('env_steps_per_wall_s'), 0)} ({'pass' if rate.get('pass_') else 'FAIL'}"
            + (f": {rate['consequence']}, decision 26" if rate.get("pass_") is False and rate.get("consequence")
               else "") + ")",
            _gate(r.get("g6b"), "b"), ftxt, _gate(r.get("g6c"), "c"),
            _f(_hours(r.get("job_wall_s"))) + (" (running)" if t.get("running_wall_s") else ""),
            _f(_hours(t.get("finished_cpu_s"))) if not t.get("running_wall_s") else "running",
            (w.get("last_write") or "-")]) + " |")
    if wall:
        lines += ["", f"Elapsed PPO wall clock (the union of every run's sessions): {_f(_hours(wall['elapsed_s']))} h; "
                      f"job-hours (their sum: the runs share the cores) {_f(_hours(wall['job_s']))} h."]
    return lines + [""]


def _fit_md(f: dict) -> list[str]:
    if f.get("status") != "done":
        return ["Fit and G5: not run.", ""]
    bl = f.get("body_level") or {}
    lines = [f"G5 {'pass' if f.get('pass_') else 'FAIL'} (seeds passing: {f.get('seeds_passing')}); body-level "
             f"selection (env.GAINS_RULE): CMA-ES seed {bl.get('cma_seed')}, {bl.get('candidate')} "
             f"({bl.get('status')}).",
             "", "| CMA-ES seed | Fit | Evaluations | Sigma | Selected | Walk | Walking turn | In-place turn | G5 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s, v in f["seeds"].items():
        cs = v.get("conditions") or {}

        def c(k):
            x = cs.get(k) or {}
            return f"{_g(x.get('successes'))}/{_g(x.get('runs'))} (surv {_g(x.get('survived'))})" if x else "-"

        lines.append("| " + " | ".join([s, str(v.get("fit_status")), _g(v.get("evals")), _f(v.get("sigma"), 4),
                                         str(v.get("selected")), c("walk"), c("walking_turn"), c("in_place_turn"),
                                         str(v.get("label"))]) + " |")
    return lines + [""]


def _cost_md(c: dict) -> list[str]:
    lines = ["| Step | Wall h | Core-h | Basis |", "|---|---|---|---|"]
    for step, s in c["steps"].items():
        if not s:
            lines.append(f"| {step} | - | - | no record |")
            continue
        wall = _f(s.get("wall_h")) + (f" (job-hours {_f(s['job_h'])})" if s.get("job_h") is not None else "")
        lines.append(f"| {step} | {wall} | {_f(s.get('core_h'))} | {s.get('basis')} |")
    lines.append(f"| total | | {_f(c['core_h'])} | core-hours recorded so far |")
    return lines + [""]


def markdown(doc: dict) -> str:
    """The tables as Markdown (one table per body per behaviour group, and per body the verdicts, G7, the looks, the
    ledger, the PPO runs, the fit and the costs)."""
    lines = ["# Stage 1 tables", "",
             f"Machine-made by `python -m designlab.stage1 report` at {doc['generated_at']} from the records in "
             f"`{doc['out']}` (designlab/stage1/report.py; the rules are in stage1_tables.json, which is also each "
             "body's summary, plan section 7's <body>/summary.json, decision 34).  Missing records show as 'not run'.  "
             "The narrative report is STAGE1_PROBE.md.", ""]
    for body, b in doc["bodies"].items():
        g = b["grid"]
        lines += [f"## {body}", "", "### Verdicts by controller", ""]
        head = ["Controller"] + list(VERDICTS)
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        jw = g.get("judged_with_r2") or {}
        for ctrl, cnt in g["gated_verdicts"].items():
            tag = " (shown)" if ctrl == g["verdict_controller"] else ""
            if ctrl == "R3" and jw.get("with_r2"):
                tag = " (not judged: judged with R2)"
            lines.append("| " + " | ".join([ctrl + tag] + [str(cnt.get(v, 0)) for v in VERDICTS]) + " |")
        lines += ["", "Counts over the conditions gated for each controller's rows."]
        for ctrl, cnt in g["single_seed"].items():
            lines.append(f"{ctrl} has one seed (decisions 3 and 28): its verdicts are at most provisional "
                         "(policy_rows.R4_VERDICT_RULE); its record on each gated condition: "
                         + (", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) or "none") + ".")
        if jw.get("with_r2"):
            lines.append(f"{body}'s R3 is judged with R2 ({jw['reason']}): the grid shows R2's verdicts "
                         "(train.R2_JUDGED_RULE).")
        lines.append("")
        x = b["g7"]
        if x["status"] == "done":
            ms, s0, pl = x.get("measured") or {}, x.get("stage0") or {}, x.get("plan") or {}
            lines += [f"G7 {'pass' if x.get('pass_') else 'FAIL'}: sway {_f(ms.get('sway_H'), 4)} H (Stage 0 "
                      f"{_f(s0.get('sway_H'), 4)}, plan {_g(pl.get('sway_H'))}), cadence "
                      f"{_f(ms.get('cadence_steps_s'), 4)} steps/s (Stage 0 {_f(s0.get('cadence_steps_s'), 4)}, plan "
                      f"{_g(pl.get('cadence_steps_s'))}).", ""]
        else:
            lines += ["G7: not run (no eval/g7.json).", ""]
        lines += ["### Verdict grid", ""] + _grid_md(g)
        lines += ["### Looks (sway, head, tail)", ""] + _looks_md(b.get("looks") or {})
        lines += ["### Margin ledger", ""] + _ledger_md(b["ledger"])
        lines += ["### PPO runs", ""] + _ppo_md(b["ppo"], b.get("ppo_wall"))
        lines += ["### Fit and G5", ""] + _fit_md(b["fit"])
        lines += ["### Core-hours per step", ""] + _cost_md(b["costs"])
    lines += [f"Core-hours recorded over the bodies reported: {_f(doc['core_h'])}.", ""]
    return "\n".join(lines)


def write(out: Path, dest: Path | None = None, bodies=None) -> dict:
    """The tables of ``out``'s records into ``<dest>/STAGE1_TABLES.md`` and ``<dest>/stage1_tables.json`` (``dest``
    default: ``out``); nothing else is written."""
    out = Path(out)
    dest = out if dest is None else Path(dest)
    doc = collect(out, bodies)
    dest.mkdir(parents=True, exist_ok=True)
    md, js = dest / MARKDOWN, dest / JSON_NAME
    doc["paths"] = dict(markdown=str(md), json=str(js))
    tmp = md.with_name(md.name + ".tmp")
    tmp.write_text(markdown(doc))
    os.replace(tmp, md)
    tmp = js.with_name(js.name + ".tmp")
    tmp.write_text(json.dumps(panels._jsonable(doc), indent=1))
    os.replace(tmp, js)
    doc["bodies"] = list(doc["bodies"])
    return doc


__all__ = ["ROWS", "RULES", "cell", "collect", "costs", "fit", "g7", "grid", "ledger_table", "looks", "markdown", "ppo",
           "ppo_wall", "write"]
