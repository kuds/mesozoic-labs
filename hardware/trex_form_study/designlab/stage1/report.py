"""Stage 1 report tables (``STAGE1_PLAN.md`` section 8; section 3 step 9; section 7): machine-made tables from
whatever records a run directory holds, for the narrative report (written separately) to cite.

Run from hardware/trex_form_study::

    python -m designlab.stage1 report --out stage1/runs/2026-11-08 [--dest DIR] [--body ...]

It reads ``--out`` and writes only ``<dest>/STAGE1_TABLES.md`` and ``<dest>/stage1_tables.json`` (``--dest`` default:
``--out``).  Per body (``collect``):

* the verdict grid (``GRID_RULE``): behaviour x condition x the rows R0, R1, R2/42, R2/43, R3/42, R3/43 and R4/42,
  each cell the panels' record (``panels.record_path``; the policy rows' R3 and R4 records are ``policy_rows``'): pass,
  the survival and success counts with their Clopper-Pearson lower bounds, the failure label, gated or reported for
  that row, the checkpoint a policy row ran, and the interim flags (``FLAGS_RULE``); one table per behaviour group;
* the verdicts per controller (``VERDICTS_RULE``: ``panels.verdict`` on R2's and R3's two seeds' records, the controller
  the summary shows by ``panels.verdict_controller``; R4's from its one seed by ``policy_rows.R4_VERDICT_RULE``, at most
  provisional) and gate G7 (``eval/g7.json``);
* the margin ledger (``ledger``'s records, ``sweeps/<row>/<margin>.json``) for its rows (R2 of the body-level G5
  selection and R3/42, and any other row with records), with plan section 5's Stage 0 values beside
  (``LEDGER_TABLE_RULE``);
* each PPO run's G6 results, budget used and judged checkpoint (``ppo/<seed>/run.json``, ``policy_only/42/run.json``;
  ``PPO_RULE``), the fit and G5 (``tune.json``, ``tune/<seed>/cma.json``; ``TUNE_RULE``), and the core-hours per step
  from the records' wall and CPU times (``COST_RULE``).

A record that does not exist shows as "not run"; a row a condition does not apply to (R0 and R1 off the hard floor)
as "n/a".  The JSON holds every value the tables show and more; the Markdown keeps each table to one screen.
"""

from __future__ import annotations

import datetime
import json
import os
from collections import Counter
from pathlib import Path

from . import BODIES, panels, tune

SCHEMA = 1
ROWS = ("R0", "R1", "R2/42", "R2/43", "R3/42", "R3/43", "R4/42")
HARD_ONLY = ("R0", "R1")
#: The controllers with verdicts from two seeds (``panels.VERDICT_ROWS``) and R4's one seed (``policy_rows.R4_ROWS``).
VERDICT_CONTROLLERS = tuple(panels.VERDICT_ROWS)
SINGLE_SEED = {"R4": ("R4/42",)}
#: Every controller the grid gives verdicts (``VERDICTS_RULE``).
CONTROLLERS = VERDICT_CONTROLLERS + tuple(SINGLE_SEED)
PPO_RUNS = (("R3", 42), ("R3", 43), ("R4", 42))
VERDICTS = ("shown", "provisional", "body-limited", "controller-limited", "not attributed", "incomplete", "not run")
BEHAVIOR_TITLES = {"stand": "Stand", "walk": "Walk", "turn": "Turn", "start_stop": "Start and stop", "push": "Push",
                   "carpet": "Carpet"}
MARKDOWN = "STAGE1_TABLES.md"
JSON_NAME = "stage1_tables.json"

GRID_RULE = (
    "one cell per row and condition from the panels' record <out>/<body>/eval/<row>/<behavior>/<condition>.json: "
    "'pass' or 'FAIL' (panels.PASS_RULE: 20/20 survive and at least 17/20 succeed) or 'unjudged' (fewer episodes, or "
    "an episode that failed to run), then survived,succeeded of the episodes and their one-sided 95 % Clopper-Pearson "
    "lower bounds (panels.LCB_RULE), as the record holds them; 'not run' if no record exists (or the floor has no "
    "settled stance, panels.NOT_RUN_RULE), 'n/a' where the row does not run (R0 and R1 run the hard floor only); a "
    "condition is gated or reported per row (panels.GATING_RULE: decision 23 reports the carpet walk for R0-R2); a "
    "failing record's label (panels.FAILURE_LABEL_RULE) is listed under the table")
FLAGS_RULE = (
    "a cell is flagged interim when its record is not judged (fewer than 20 episodes or an episode that failed to "
    "run), its falls are not attributed yet (decision 20, eval --attribute), its key's panel sources differ from the "
    "sources read now (panels.PANEL_SOURCES_RULE: made by other code), or its row is a policy row (R3, R4) that ran a "
    "checkpoint that is not final: the checkpoint its record names (the controller's policy checkpoint, "
    "policy_rows.CHECKPOINT_SELECTION_RULE: an interim one, STAGE1_POLICY_CHECKPOINT, or the judged one before every "
    "run reached its budget or 'judge --final' recorded it), or, for a record that names none, its runs' judged "
    "checkpoint in run.json (train.JUDGED_RULE: judged_checkpoint not final); the cell names the checkpoint's steps")
VERDICTS_RULE = (
    "per condition, panels.verdict on each controller's two seeds' records (panels.VERDICT_ROWS: R2 from R2/42 and "
    "R2/43, R3 from R3/42 and R3/43; panels.VERDICT_RULE), 'not run' when neither record exists; the counts are over "
    "the conditions gated for the controller's rows; the controller the grid's verdict column shows is "
    "panels.verdict_controller's (R3 once its records exist, else R2, interim); R4 trains one seed (decisions 3 and "
    "28), so its verdict is policy_rows.R4_VERDICT_RULE's on R4/42's record (panels.verdict with 'shown' read as "
    "'provisional': at most provisional; 'not run' when no record exists), counted over the conditions gated for "
    "R4/42, and its records' results on those conditions are counted beside")
LEDGER_TABLE_RULE = (
    "per margin and search (ledger.RULES), each ledger row's record <out>/<body>/sweeps/<row>/<margin>.json: the "
    "threshold on success ('works': the last value at which all 3 seeds succeed) and on survival ('survives'), each "
    "with the nearest value that fails and the label at it (ledger.LABEL_RULE), or 'fails at nominal', 'works across "
    "the interval' (to its far end), 'not run'; points show works and survives; plan section 5's Stage 0 value "
    "beside (ledger.STAGE0); a policy row's results (R3/42, policy_rows) name the checkpoints its records ran, "
    "listed under the table and marked provisional when one is not final (FLAGS_RULE)")
PPO_RULE = (
    "per run (R3 seeds 42 and 43, R4 seed 42) its run.json: status, env steps trained of the budget "
    "(train.BUDGET_RULE), the last checkpoint, the judged checkpoint (train.JUDGED_RULE: steps, final or provisional), "
    "G6a (zero residual identical; env steps per wall second against 600), G6b and G6c (train.G6_RULE: at what step, "
    "pass, the monitor's mean survival and the conditions R3 beats R2 on), and the wall and CPU seconds of its "
    "finished sessions (run.json's throughput)")
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
    "estimate), the panels (each record's "
    "episode wall seconds and its attribution's: one process per episode, an upper bound on CPU) and the ledger (each "
    "record's CPU seconds); core-hours are CPU hours where recorded, else wall hours x processes")
RULES = dict(grid=GRID_RULE, flags=FLAGS_RULE, verdicts=VERDICTS_RULE, ledger=LEDGER_TABLE_RULE, ppo=PPO_RULE,
             tune=TUNE_RULE, cost=COST_RULE)


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


# ------------------------------------------------------------------------------------------------
# The verdict grid
# ------------------------------------------------------------------------------------------------
def _policy_runs(out: Path, body: str) -> dict:
    """Each policy row's run.json judged checkpoint (``FLAGS_RULE``)."""
    out_ = {}
    for mode, seed in PPO_RUNS:
        doc = _read(run_json(out, body, mode, seed)) or {}
        jc = doc.get("judged_checkpoint")
        out_[f"{mode}/{seed}"] = dict(judged_checkpoint=jc, final=bool(jc and jc.get("final")))
    return out_


def _policy_checkpoint(controller: dict | None) -> dict | None:
    """The checkpoint a policy row's record ran, as ``policy_rows`` describes it (the controller's policy's
    checkpoint: steps, selection, final, ...), or None."""
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
        return dict(status="not run", gated=gated, text="not run", reason="no settled stance (panels.NOT_RUN_RULE)")
    c = rec.get("counts") or {}
    f = rec.get("failure") or {}
    flags = []
    if not c.get("judged"):
        flags.append("not judged")
    if c.get("errors"):
        flags.append(f"{c['errors']} failed to run")
    if (f.get("label") or "").startswith("not attributed") or (f.get("falls_label") or "").startswith("not attributed"):
        flags.append("not attributed")
    if sources_now is not None and (rec.get("key") or {}).get("sources") not in (None, sources_now):
        flags.append("other panel sources")
    ctrl = panels.controller_of(row)
    ck = _policy_checkpoint(rec.get("controller")) if ctrl in ("R3", "R4") else None
    final = None
    if ctrl in ("R3", "R4"):
        final = bool(ck.get("final")) if ck is not None else bool((policy_runs.get(row) or {}).get("final"))
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
                untestable=f.get("untestable") or None, flags=flags, text=text,
                checkpoint=_checkpoint_of(rec.get("controller")) if ctrl in ("R3", "R4") else None,
                checkpoint_selection=ck.get("selection") if ck is not None else None, checkpoint_final=final,
                wall_s=rec.get("wall_s"), episode_wall_s=rec.get("episode_wall_s"),
                attribution_wall_s=(rec.get("attribution") or {}).get("wall_s"))


def grid(out: Path, body: str, sources_now: str | None = None) -> dict:
    """The verdict grid of a body (``GRID_RULE``, ``VERDICTS_RULE``)."""
    from .policy_rows import r4_verdict

    edir = Path(out) / body / "eval"
    found = sorted({"/".join(q.relative_to(edir).parts[:-2]) for q in edir.glob("**/*.json")
                    if len(q.relative_to(edir).parts) >= 3 and q.relative_to(edir).parts[0] != "traces"})
    rows = list(ROWS) + [r for r in found if r and r not in ROWS]
    shown, note = panels.verdict_controller(found)
    runs = _policy_runs(out, body)
    behaviors, counts, single = {}, {c: Counter() for c in CONTROLLERS}, {c: Counter() for c in SINGLE_SEED}
    for cond in panels.conditions():
        recs = {r: _read(panels.record_path(out, body, r, cond)) for r in rows}
        cells = {r: cell(recs[r], r, cond, sources_now, runs) for r in rows}
        verdicts = {}
        for ctrl, pair in panels.VERDICT_ROWS.items():
            rs = [recs.get(r) if r in recs else _read(panels.record_path(out, body, r, cond)) for r in pair]
            g = panels.gated_for(pair[0], cond)
            v = dict(verdict="not run") if all(x is None for x in rs) else panels.verdict(rs)
            verdicts[ctrl] = dict(verdict=v["verdict"], gated=g, labels=v.get("labels"), passes=v.get("passes"),
                                  untestable=v.get("untestable") or None)
            if g:
                counts[ctrl][v["verdict"]] += 1
        for ctrl, (r,) in SINGLE_SEED.items():
            x = cells[r]
            v = dict(verdict="not run") if recs[r] is None else r4_verdict([recs[r]])
            verdicts[ctrl] = dict(verdict=v["verdict"], gated=x["gated"], labels=v.get("labels"),
                                  passes=v.get("passes"), untestable=v.get("untestable") or None,
                                  read_from=v.get("read_from"))
            if x["gated"]:
                single[ctrl][x.get("result") or x["status"]] += 1
                counts[ctrl][v["verdict"]] += 1
        behaviors.setdefault(cond.behavior, {})[cond.name] = dict(
            key=cond.key, plan_gated=cond.gated, floor=cond.floor, friction=cond.friction,
            gated_for=[r for r in ROWS if panels.gated_for(r, cond)], cells=cells, verdicts=verdicts,
            shown=verdicts[shown]["verdict"])
    return dict(rows=rows, found=found, verdict_controller=shown, verdict_controller_note=note, behaviors=behaviors,
                gated_verdicts={c: dict(v) for c, v in counts.items()},
                single_seed={c: dict(v) for c, v in single.items()}, policy_runs=runs)


def g7(out: Path, body: str) -> dict:
    """Gate G7 as the panels recorded it (``eval/g7.json``), or not run."""
    doc = _read(Path(out) / body / "eval" / "g7.json")
    if not doc:
        return dict(status="not run")
    return dict(status="done", pass_=doc.get("pass_"), measured=doc.get("measured"), stage0=doc.get("stage0"),
                plan=doc.get("plan"), difference=doc.get("difference"), within=doc.get("within"),
                plan_rounding=doc.get("plan_rounding"))


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


def ledger_table(out: Path, body: str) -> dict:
    """``LEDGER_TABLE_RULE``: per margin, each ledger row's results and Stage 0's values."""
    from . import ledger

    rows = ledger_rows(out, body)
    margins = {}
    for m in ledger.margins(body):
        per_row = {}
        for r in rows:
            rec = _read(ledger.record_path(out, body, r, m.name))
            if rec is None:
                per_row[r] = dict(status="not run")
                continue
            ck = _policy_checkpoint(rec.get("controller"))
            if ck is not None:
                ck = dict(steps=ck.get("steps"), final=bool(ck.get("final")), selection=ck.get("selection"))
            per_row[r] = dict(status=rec.get("status"), summary=rec.get("summary"), cut_s=rec.get("cut_s"),
                              halvings=rec.get("halvings"), cost=rec.get("cost"),
                              not_run=rec.get("not_run"), reference_stop=rec.get("reference_stop"),
                              panel_reading=rec.get("panel_reading"), checkpoint=ck)
        margins[m.name] = dict(title=m.title, swept=m.swept, definition=m.describe(),
                               stage0=ledger.stage0(body, m.name), rows=per_row)
    checkpoints = {}
    for r in rows:
        cks = {(x["checkpoint"]["steps"], x["checkpoint"]["final"]) for m in margins.values()
               for x in [m["rows"][r]] if x.get("checkpoint")}
        if cks:
            checkpoints[r] = [dict(steps=n, final=f) for n, f in sorted(cks, key=lambda t: (t[0] is None, t))]
    return dict(rows=rows, margins=margins, checkpoints=checkpoints)


# ------------------------------------------------------------------------------------------------
# PPO, the fit, costs
# ------------------------------------------------------------------------------------------------
def run_json(out: Path, body: str, mode: str, seed: int) -> Path:
    """A PPO run's record (``train.run_dir_of``)."""
    return Path(out) / body / ("ppo" if mode == "R3" else "policy_only") / str(int(seed)) / "run.json"


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
        if t0 is not None and path.exists():
            t1 = datetime.datetime.fromtimestamp(os.stat(path).st_mtime, datetime.timezone.utc)
            running_wall += max(0.0, (t1 - t0).total_seconds())
    first = next((t for t in doc.get("throughput") or () if t.get("first_rollouts")), None)
    return dict(finished_wall_s=wall, finished_cpu_s=cpu, running_wall_s=running_wall,
                env_processes=(first or {}).get("env_processes"),
                cpu_per_wall_first_rollouts=(first["cpu_s"] / first["wall_s"]) if first and first.get("wall_s")
                else None)


def ppo(out: Path, body: str) -> dict:
    """``PPO_RULE`` per run."""
    rows = {}
    for mode, seed in PPO_RUNS:
        path = run_json(out, body, mode, seed)
        doc = _read(path)
        name = f"{mode}/{seed}"
        if doc is None:
            rows[name] = dict(status="not run")
            continue
        g6 = doc.get("g6") or {}
        zr, rate, b, c = (g6.get(k) or {} for k in ("G6a_zero_residual", "G6a_rate", "G6b", "G6c"))
        ck = doc.get("checkpoints") or []
        jc = doc.get("judged_checkpoint") or {}
        mon_b = (b.get("monitor") or {}).get("mean_survival_s") or {}
        rows[name] = dict(
            status=doc.get("status"), mode=mode, seed=seed, steps=doc.get("num_timesteps"), budget=doc.get("budget"),
            checkpoints=len(ck), last_checkpoint=ck[-1] if ck else None,
            judged_checkpoint=dict(steps=jc.get("steps"), final=jc.get("final"), budget=jc.get("budget"))
            if jc else None,
            g6a_zero_residual=zr.get("identical") if zr else None,
            g6a_rate=dict(env_steps_per_wall_s=rate.get("env_steps_per_wall_s"), pass_=rate.get("pass_"),
                          threshold=rate.get("threshold")) if rate else None,
            g6b=dict(at_steps=b.get("at_steps"), pass_=b.get("pass_"), mean_survival_s=mon_b) if b else None,
            g6c=dict(at_steps=c.get("at_steps"), pass_=c.get("pass_"),
                     beats_on=(c.get("monitor") or {}).get("policy_beats_zero_on")) if c else None,
            gains=(doc.get("gains") or {}).get("candidate"), times=_ppo_times(doc, path))
    return rows


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
    wall = sum(r["times"]["finished_wall_s"] + r["times"]["running_wall_s"] for r in p.values() if r.get("times"))
    running = [n for n, r in p.items() if r.get("times") and r["times"]["running_wall_s"]]
    est = sum(r["times"]["running_wall_s"] * (r["times"]["cpu_per_wall_first_rollouts"] or 1.0)
              for r in p.values() if r.get("times") and r["times"]["running_wall_s"])
    steps["ppo"] = dict(wall_h=_hours(wall), core_h=_hours(cpu + est), finished_cpu_h=_hours(cpu),
                        running_estimate_h=_hours(est) if running else None, running=running,
                        basis="finished sessions' CPU, and for a running session its wall to the record's last write x "
                              "the CPU per wall second of its first rollouts (an estimate)" if running else
                        "sessions' CPU") if wall else None
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
    lw = lc = 0.0
    n_l = 0
    for q in (bdir / "sweeps").glob("*/**/*.json") if (bdir / "sweeps").exists() else ():
        rec = _read(q) or {}
        if not rec.get("margin"):
            continue
        c = rec.get("cost") or {}
        lw += float(c.get("wall_s") or 0.0)
        lc += float(c.get("cpu_s") or 0.0)
        n_l += 1
    ls = _read(bdir / "sweeps" / "summary.json") or {}
    l_wall = float(sum(s.get("wall_s") or 0.0 for s in ls.get("sessions") or ()))
    steps["ledger"] = dict(wall_h=_hours(l_wall) if l_wall else None, core_h=_hours(lc), episode_wall_h=_hours(lw),
                           records=n_l, basis="episodes' CPU seconds (each evaluation counted in every record that "
                                              "uses it)", rule=ledger.COST_RULE) if n_l else None
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
        doc["bodies"][b] = dict(grid=grid(out, b, sources_now), g7=g7(out, b), ledger=ledger_table(out, b),
                                ppo=ppo(out, b), fit=fit(out, b), costs=costs(out, b))
    doc["core_h"] = float(sum(x["costs"]["core_h"] for x in doc["bodies"].values()))
    return doc


def _cell_label(c: dict) -> str | None:
    if c.get("status") != "done" or c.get("pass_") is not False and not c.get("flags"):
        return None
    parts = []
    if c.get("pass_") is False:
        parts.append(c.get("label") or "no label")
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
    lines = [f"Verdicts shown: {g['verdict_controller_note']}.  Cells: result, survived,succeeded of 20 (their "
             "one-sided 95 % Clopper-Pearson lower bounds); '(rep.)' reported, not gated, for that row; '*' interim "
             "(listed below each table); 'not run' no record; 'n/a' the row does not run there; R4's verdict is at "
             "most provisional (one seed).", ""]
    for beh, conds in g["behaviors"].items():
        lines += [f"#### {BEHAVIOR_TITLES.get(beh, beh)}", ""]
        head = ["Condition", "Gated for"] + rows + [f"{c} verdict" + (" (shown)" if c == shown else "")
                                                    for c in CONTROLLERS]
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
        if notes:
            lines += notes + [""]
    return lines


def _ledger_cell(x: dict | None, search: str, kind: str | None) -> tuple[str, str]:
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
            t = f"{_value(b.get('threshold'), kind)} (fails {_value(b.get('fails'), kind)}"
            t += f"; {b['label']})" if b.get("label") else ")"
        elif st == "fails at nominal":
            t = "fails at nominal" + (f" ({b['label']})" if b.get("label") else "")
        elif st == "works across the interval":
            t = f"works to {_value(b.get('threshold'), kind)}"
        else:
            t = st or "not run"
        if b.get("search") and search in ("turn",):
            t += f" [{b['search']}]"
        out.append(t)
    if x.get("status") == "partial":
        out = [t + " *" for t in out]
    return out[0], out[1]


def _value(x, kind: str | None) -> str:
    if x is None:
        return "?"
    if kind == "latency_steps":
        return f"{int(x)} steps ({2 * int(x)} ms)"
    return f"{x:.4g}"


def _stage0_text(st: dict, search: str) -> str:
    v = st.get("values") or {}
    if search in v:
        return _g(v[search])
    if st.get("forward_bound_m_s") is not None:
        return f"(forward {_g(st['forward_bound_m_s'])})"
    if len(v) == 1:
        return _g(next(iter(v.values())))
    return "-"


def _ledger_md(lt: dict) -> list[str]:
    rows = lt["rows"]
    head = ["Margin", "Search"] + [f"{r} works" for r in rows] + [f"{r} survives" for r in rows] + ["Stage 0"]
    lines = ["Cells: the threshold (the last value at which all 3 seeds succeed, or survive) and the nearest value "
             "that fails with the label there; '*' the search is not finished.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    extra = []
    for name, m in lt["margins"].items():
        d = m["definition"]
        kinds = {s["name"]: s["kind"] for s in d["searches"]}
        if d.get("not_run"):
            cells = ["not run"] * (2 * len(rows))
            lines.append("| " + " | ".join([m["title"], "-"] + cells + [_g((m["stage0"].get("values") or {}).get(
                "one_foot_m_s2"))]) + " |")
            for r in rows:
                x = m["rows"].get(r) or {}
                ref = x.get("reference_stop") or {}
                if ref.get("peak_deceleration_m_s2") is not None:
                    extra.append(f"- {m['title']}: {d['not_run']}; the existing stop reference decelerates at "
                                 f"{ref['peak_deceleration_m_s2']:.3f} m/s^2 (1 Hz low-pass)")
                    pr = x.get("panel_reading")
                    if pr and pr.get("acceleration_peak_max_m_s2") is not None:
                        extra.append(f"  - {r}'s start-and-stop panel: stops' acceleration peak at most "
                                     f"{pr['acceleration_peak_max_m_s2']:.3f} m/s^2, {pr['stopped_within']}/"
                                     f"{pr['stops']} stopped within 1.5 strides")
                    break
            continue
        names = [s["name"] for s in d["searches"]] + [w[0] for w in d.get("worse") or ()]
        for sname in names:
            kind = kinds.get(sname) or next(iter(kinds.values()), None)
            works, surv = zip(*[_ledger_cell(m["rows"].get(r), sname, kind) for r in rows], strict=True)
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
                cells_w.append(f"{verdict} {pt.get('succeeded')}/3{same}" + ("" if pt.get("works") else lab))
                cells_s.append(f"{'survives' if pt.get('survives') else 'falls'} {pt.get('survived')}/3")
            lines.append("| " + " | ".join([m["title"], f"{p['name']}"] + cells_w + cells_s + ["-"]) + " |")
        if m["stage0"].get("text") and name == "voltage":
            extra.append(f"- {m['title']}: Stage 0: {m['stage0']['text']}")
    for r, cks in (lt.get("checkpoints") or {}).items():
        extra.append(f"- {r} ran checkpoint " + ", ".join(
            f"{_g(c['steps'])} ({'final' if c['final'] else 'provisional'})" for c in cks)
            + ("" if all(c["final"] for c in cks) else " *: not the final judged checkpoint (FLAGS_RULE)"))
    lines.append("")
    return lines + extra + ([""] if extra else [])


def _ppo_md(p: dict) -> list[str]:
    head = ["Run", "Status", "Steps / budget", "Last checkpoint", "Judged checkpoint", "G6a zero residual",
            "G6a env steps/s", "G6b", "G6c", "Wall h", "CPU h"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in p.items():
        if r.get("status") == "not run":
            lines.append("| " + " | ".join([name, "not run"] + ["-"] * (len(head) - 2)) + " |")
            continue
        jc = r.get("judged_checkpoint")
        jtxt = "-" if not jc else f"{_g(jc.get('steps'))} ({'final' if jc.get('final') else 'provisional'})"
        rate = r.get("g6a_rate") or {}
        b, c = r.get("g6b"), r.get("g6c")
        btxt = "-" if not b else (f"{'pass' if b['pass_'] else 'FAIL'} at {_g(b['at_steps'])} (survival "
                                  + ", ".join(f"{k} {v:.2f} s" for k, v in (b.get("mean_survival_s") or {}).items())
                                  + ")")
        ctxt = "-" if not c else (f"{'pass' if c['pass_'] else 'FAIL'} at {_g(c['at_steps'])}"
                                  + (f" (beats R2 on {', '.join(c['beats_on'])})" if c.get("beats_on") else ""))
        t = r.get("times") or {}
        lines.append("| " + " | ".join([
            name, str(r.get("status")), f"{_g(r.get('steps'))} / {_g(r.get('budget'))}", _g(r.get("last_checkpoint")),
            jtxt, _f(r.get("g6a_zero_residual")) if r.get("g6a_zero_residual") is not None else "-",
            "-" if not rate else
            f"{_f(rate.get('env_steps_per_wall_s'), 0)} ({'pass' if rate.get('pass_') else 'FAIL'})",
            btxt, ctxt, _f(_hours((t.get("finished_wall_s") or 0.0) + (t.get("running_wall_s") or 0.0)))
            + (" (running)" if t.get("running_wall_s") else ""),
            _f(_hours(t.get("finished_cpu_s"))) if not t.get("running_wall_s") else "running"]) + " |")
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
        lines.append(f"| {step} | {_f(s.get('wall_h'))} | {_f(s.get('core_h'))} | {s.get('basis')} |")
    lines.append(f"| total | | {_f(c['core_h'])} | core-hours recorded so far |")
    return lines + [""]


def markdown(doc: dict) -> str:
    """The tables as Markdown (one table per body per behaviour group, and per body the verdicts, G7, the ledger,
    the PPO runs, the fit and the costs)."""
    lines = ["# Stage 1 tables", "",
             f"Machine-made by `python -m designlab.stage1 report` at {doc['generated_at']} from the records in "
             f"`{doc['out']}` (designlab/stage1/report.py; the rules are in stage1_tables.json).  Missing records show "
             "as 'not run'.  The narrative report is STAGE1_PROBE.md.", ""]
    for body, b in doc["bodies"].items():
        g = b["grid"]
        lines += [f"## {body}", "", "### Verdicts by controller", ""]
        head = ["Controller"] + list(VERDICTS)
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for ctrl, cnt in g["gated_verdicts"].items():
            lines.append("| " + " | ".join([ctrl + (" (shown)" if ctrl == g["verdict_controller"] else "")]
                                           + [str(cnt.get(v, 0)) for v in VERDICTS]) + " |")
        lines += ["", "Counts over the conditions gated for each controller's rows."]
        for ctrl, cnt in g["single_seed"].items():
            lines.append(f"{ctrl} has one seed (decisions 3 and 28): its verdicts are at most provisional "
                         "(policy_rows.R4_VERDICT_RULE); its record on each gated condition: "
                         + (", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) or "none") + ".")
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
        lines += ["### Margin ledger", ""] + _ledger_md(b["ledger"])
        lines += ["### PPO runs", ""] + _ppo_md(b["ppo"])
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


__all__ = ["ROWS", "RULES", "cell", "collect", "costs", "fit", "g7", "grid", "ledger_table", "markdown", "ppo",
           "write"]
