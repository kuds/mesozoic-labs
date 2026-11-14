"""Stage 1b's reading of the validation rows (``STAGE1B_PLAN.md`` section 4): per body and condition each speed row's
class by plan section 4, the failing limits with their counts, the falls' candidates and decision 20's attribution,
the baseline beside, the trades, and the answers to Q1 and Q2 (the speed walks at 0 ms) and Q3 (at 10 ms).

Run from hardware/trex_form_study::

    python -m designlab.stage1 speedreport --out stage1/runs/2026-11-10-speed [--stage1 stage1/runs/2026-11-08]
                                           [--body B ...] [--dest DIR]

It reads records only (it runs nothing) and writes ``<dest>/STAGE1B_TABLES.md`` and ``<dest>/stage1b_tables.json``
(``--dest`` default: ``--out``).  A body is reported when ``<out>/<body>/speedfit/start.json`` exists.  No module the
speed fit or the rows run imports this one, so it can change during the round without changing their sources or their
records' keys (``speedfit.SOURCES_RULE``, ``speedfit.DelayedRow.sources``).  The panels' own ``eval/summary.json``
reads Stage 1's verdict rule (R2/42 with R2/43), which does not apply to these rows (``speedfit.ROWS_RULE``).
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path

from . import panels, speedfit

SCHEMA = 1
MARKDOWN = "STAGE1B_TABLES.md"
JSON_NAME = "stage1b_tables.json"
SPEED_WALKS = ("walk/brisk", "walk/fr05")
#: The conditions a speed fit must not lose at 0 ms (plan section 4's trade).
KEPT = ("walk/nominal", "turn/walking_0.35_left", "turn/walking_0.35_right", "turn/in_place_0.50_left",
        "turn/in_place_0.50_right")
CLASSES = ("shown", "survives only", "fails", "not judged", "not run")

READING_RULE = (
    "plan section 4 on one row's panel record of one condition (<out>/<body>/eval/<row>/<behavior>/<condition>.json, "
    "panels.record_path): 'shown' if the record is judged (20 episodes on seeds 3042-3061, none failed to run, "
    f"panels.counts) with {panels.RUNS}/{panels.RUNS} up and at least {panels.MIN_SUCCEED}/{panels.RUNS} meeting every "
    f"limit (Stage 1 section 4); 'survives only' with {panels.RUNS}/{panels.RUNS} up and fewer meeting every limit; "
    "'fails' with fewer up; 'not judged' for a record that is not (incomplete, an episode that failed to run, not run "
    "for want of a stance); 'not run' without a record; the failing limits are named with counts: the success terms "
    "the episodes that stayed up failed (their failed_terms), and for the falls their causes, their candidates "
    "(control.CANDIDATE_RULE) and decision 20's label (the record's failure.falls_label, 'not attributed' until eval "
    "--attribute has run, speedfit.ATTRIBUTION_RULE); a shown record's few failures are listed too")
QUESTIONS_RULE = (
    "Q1 (arm S, a body that is its own pick) and Q2 (arm K, a faster-knee body: speedfit.pick_of names another) per "
    "body: walk/brisk and walk/fr05 of R2s/<seed> (0 ms), each beside its baseline: on arm S Stage 1's R2 row of the "
    "start's CMA-ES seed on the same body (<stage1>/<body>/eval/R2/<seed>/; Stage 1's panels, same seeds and judge), on "
    "arm K R2@0ms on the body (the pick's gains without a refit), with the pick's Stage 1 R2 row shown as well (another "
    "body); Q3: the same conditions of R2s/<seed>@10ms beside R2@10ms (the start at 10 ms), the other five conditions "
    "at 10 ms reported; speed G5's pooled counts (0-10 ms) are not an answer (speedfit.SELECTION_RULE)")
TRADE_RULE = (
    "plan section 4: a speed fit that loses the nominal walk or a target turn at 0 ms is a trade, not a success: per "
    f"condition of {', '.join(KEPT)}, a trade when the 0 ms baseline (QUESTIONS_RULE) shows it and R2s/<seed> does "
    "not; 'not lost' when neither shows it (the baseline did not either), 'kept' when both do, 'gained' when only "
    "R2s does, and 'unknown' while either record is not judged")
BASELINE_NOTE = (
    "Stage 1's R2 records were made by Stage 1's code: their keys name its panel sources, and the round's code differs "
    "(this round's edits and the faster-knee registry, STAGE1B_PLAN.md section 6), so Stage 1's tables are read from "
    "its own commit; their episodes reproduce on the round's code (the review's check on brisk and Fr 0.05 episodes, "
    "and speedfit's test on a recorded G5 episode)")
RULES = dict(reading=READING_RULE, questions=QUESTIONS_RULE, trade=TRADE_RULE, baseline=BASELINE_NOTE)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _conditions() -> dict:
    return {c.key: c for c in panels.select(panels.conditions(), names=speedfit.ROW_CONDITIONS)}


def classify(rec: dict | None) -> str:
    """``READING_RULE``'s class of one record."""
    if rec is None:
        return "not run"
    c = rec.get("counts") or {}
    if rec.get("status") != "done" or not c.get("judged"):
        return "not judged"
    if c["survived"] == panels.RUNS:
        return "shown" if c["succeeded"] >= panels.MIN_SUCCEED else "survives only"
    return "fails"


def cell(rec: dict | None) -> dict:
    """One row's reading of one condition (``READING_RULE``)."""
    cls = classify(rec)
    if rec is None:
        return dict(cls=cls)
    c = rec.get("counts") or {}
    eps = rec.get("episodes") or []
    up = [e for e in eps if not e.get("fell") and not e.get("error")]
    falls = [e for e in eps if e.get("fell") and not e.get("error")]
    f = rec.get("failure") or {}
    a = rec.get("attribution") or {}
    return dict(cls=cls, status=rec.get("status"), judged=bool(c.get("judged")), episodes=c.get("episodes"),
                survived=c.get("survived"), succeeded=c.get("succeeded"), errors=c.get("errors"),
                failed_terms=dict(Counter(t for e in up for t in e.get("failed_terms") or ()).most_common()),
                falls=len(falls), fall_causes=dict(Counter(str(e.get("fall_cause")) for e in falls)),
                fall_candidates=dict(Counter(x for e in falls for x in e.get("fall_candidates") or ()).most_common()),
                falls_label=f.get("falls_label") or (a.get("label") if falls else None),
                attribution_episodes=a.get("episodes"), label=f.get("label"), kind=f.get("kind"),
                panel_sources=(rec.get("key") or {}).get("sources"), row_sources=(rec.get("key") or {}).get(
                    "row_sources"), controller=(rec.get("controller") or {}).get("controller"))


def text(x: dict) -> str:
    """A cell as one short line: class, survived and succeeded of 20, the failing limits."""
    if x["cls"] == "not run":
        return "not run"
    s = f"{x['cls']} {x['survived']},{x['succeeded']}"
    if x.get("errors"):
        s += f" ({x['errors']} failed to run)"
    why = []
    if x["failed_terms"]:
        why.append(", ".join(f"{t} {n}" for t, n in x["failed_terms"].items()))
    if x["falls"]:
        cands = ", ".join(f"{k} {n}" for k, n in x["fall_candidates"].items()) or "no candidate"
        why.append(f"falls {x['falls']} ({cands}; {x['falls_label'] or 'not attributed'})")
    return s + (f": {'; '.join(why)}" if why else "")


def rows_of(out: Path, body: str) -> list[str]:
    """The speed rows with records under ``<out>/<body>/eval`` (R2s/<s>, R2s/<s>@<ms>ms, R2@<ms>ms), in that order."""
    edir = Path(out) / body / "eval"
    found = set()
    for q in edir.glob("*/**/*.json"):
        parts = q.relative_to(edir).parts
        if len(parts) >= 3 and parts[0] != "traces":
            found.add("/".join(parts[:-2]))

    def order(r):
        m = speedfit._ROW_NAME.match(r)
        if m:
            return (0, int(m.group(1)), int(m.group(2) or 0))
        return (1, 0, int(r[3:-2])) if r.startswith("R2@") and r.endswith("ms") else (2, 0, 0)

    return sorted((r for r in found if panels.controller_of(r) in speedfit.SPEED_CONTROLLERS), key=order)


def compare(base: str, speed: str) -> str:
    """``TRADE_RULE`` on one condition: the 0 ms baseline's class and the speed row's."""
    judged = ("shown", "survives only", "fails")
    if base not in judged or speed not in judged:
        return "unknown"
    if base == "shown":
        return "kept" if speed == "shown" else "trade"
    return "gained" if speed == "shown" else "not lost"


def body_report(out: Path, body: str, stage1: Path = speedfit.STAGE1_RUN) -> dict:
    """One body's tables (module docstring)."""
    start = _read(speedfit.start_path(out, body)) or {}
    pick = speedfit.pick_of(body)
    arm = "S" if pick == body else "K"
    conds = _conditions()
    rows = rows_of(out, body)
    grid = {k: {r: cell(_read(panels.record_path(out, body, r, c))) for r in rows} for k, c in conds.items()}
    s1_seed = start.get("cma_seed")
    s1_row = None if s1_seed is None else f"R2/{int(s1_seed)}"
    s1_body = start.get("start_from") or pick
    stage1_cells = {k: cell(_read(panels.record_path(stage1, s1_body, s1_row, c))) if s1_row else dict(cls="not run")
                    for k, c in conds.items()}
    g5 = _read(Path(out) / body / "speedg5.json") or {}
    seeds = sorted({int(speedfit._ROW_NAME.match(r).group(1)) for r in rows if speedfit._ROW_NAME.match(r)}
                   | {int(s) for s in g5.get("seeds_selected") or ()})
    base0 = "R2@0ms" if arm == "K" else None  # (None: Stage 1's R2 row on the body itself)
    b10 = f"R2@{speedfit.ROW_DELAY_MS}ms"

    def get(row, k):
        return (grid[k].get(row) or dict(cls="not run")) if row else stage1_cells[k]

    q = "Q1" if arm == "S" else "Q2"
    answers, trades = {}, {}
    for s in seeds:
        r0, r10 = f"R2s/{s}", f"R2s/{s}@{speedfit.ROW_DELAY_MS}ms"
        answers[str(s)] = {
            q: {k: dict(row=r0, speed=get(r0, k), baseline_row=base0 or f"Stage 1 {s1_row}", baseline=get(base0, k),
                        **({"pick_stage1": stage1_cells[k]} if arm == "K" else {})) for k in SPEED_WALKS},
            "Q3": {k: dict(row=r10, speed=get(r10, k), baseline_row=b10, baseline=get(b10, k),
                           answer=k in SPEED_WALKS) for k in conds}}
        trades[str(s)] = {k: dict(baseline=get(base0, k)["cls"], speed=get(r0, k)["cls"],
                                  reading=compare(get(base0, k)["cls"], get(r0, k)["cls"])) for k in KEPT}
    sel = {s: {k: v.get(k) for k in ("selected", "judged", "partial", "judged_at_generation", "reason")}
           for s, v in (g5.get("seeds") or {}).items()}
    return dict(body=body, arm=arm, pick=pick, start={k: start.get(k) for k in ("start_from", "candidate", "cma_seed",
                                                                                 "sha256", "file")},
                rows=rows, grid=grid, stage1_baseline=dict(row=s1_row, body=s1_body, cells=stage1_cells),
                speed_g5=sel, answers=answers, trades=trades,
                trade_conditions=sorted({k for t in trades.values() for k, v in t.items() if v["reading"] == "trade"}))


def collect(out: Path, stage1: Path = speedfit.STAGE1_RUN, bodies=None) -> dict:
    out = Path(out)
    if bodies is None:
        bodies = sorted(b for b in speedfit.BODIES if speedfit.start_path(out, b).exists())
    return dict(schema=SCHEMA, rules=RULES, out=str(out), stage1=str(stage1), bodies=list(bodies),
                reports={b: body_report(out, b, stage1) for b in bodies}, written_at=datetime.datetime.now(
                    datetime.timezone.utc).isoformat(timespec="seconds"))


def markdown(doc: dict) -> str:
    lines = ["# Stage 1b speed round: the validation rows (machine-made)", "",
             f"Plan section 4's reading (`speedreport.READING_RULE`); baselines and questions (`QUESTIONS_RULE`); "
             f"trades (`TRADE_RULE`). Each cell: class, survived and succeeded of {panels.RUNS}, then the failing "
             "limits with counts and the falls' candidates and attribution.", "", f"_{BASELINE_NOTE}._", ""]
    for b in doc["bodies"]:
        r = doc["reports"][b]
        st = r["start"]
        lines += [f"## {b} (arm {r['arm']}{'' if r['arm'] == 'S' else ', pick ' + r['pick']})", "",
                  f"Start: {st.get('start_from')} {st.get('candidate')} (Stage 1 G5 selection). Speed G5: "
                  + ("; ".join(f"seed {s} {v.get('selected')} at generation {v.get('judged_at_generation')}"
                               + ("" if v.get("judged") else f" (not judged: {v.get('reason')})")
                               for s, v in r["speed_g5"].items()) or "not run") + ".", ""]
        base = r["stage1_baseline"]
        head = ["condition"] + r["rows"] + [f"Stage 1 {base['row']} ({base['body']})"]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for k in r["grid"]:
            lines.append("| " + " | ".join([k] + [text(r["grid"][k][x]) for x in r["rows"]]
                                           + [text(base["cells"][k])]) + " |")
        lines.append("")
        for s, a in r["answers"].items():
            q = "Q1" if "Q1" in a else "Q2"
            for k, v in a[q].items():
                lines.append(f"- {q}, seed {s}, {k} at 0 ms: **{v['speed']['cls']}** ({text(v['speed'])}); baseline "
                             f"{v['baseline_row']}: {text(v['baseline'])}")
            for k in SPEED_WALKS:
                v = a["Q3"][k]
                lines.append(f"- Q3, seed {s}, {k} at {speedfit.ROW_DELAY_MS} ms: **{v['speed']['cls']}** "
                             f"({text(v['speed'])}); baseline {v['baseline_row']}: {text(v['baseline'])}")
            t = r["trades"][s]
            lost = [k for k, v in t.items() if v["reading"] == "trade"]
            lines.append(f"- Trades at 0 ms, seed {s}: " + (", ".join(lost) if lost else "none") + " ("
                         + "; ".join(f"{k} {v['reading']}" for k, v in t.items()) + ")")
        lines.append("")
    return "\n".join(lines) + "\n"


def write(out: Path, dest: Path | None = None, stage1: Path = speedfit.STAGE1_RUN, bodies=None) -> dict:
    doc = collect(out, stage1, bodies)
    dest = Path(dest or out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / JSON_NAME).write_text(json.dumps(doc, indent=1))
    (dest / MARKDOWN).write_text(markdown(doc))
    return doc


__all__ = ["RULES", "body_report", "cell", "classify", "collect", "compare", "markdown", "rows_of", "write"]
