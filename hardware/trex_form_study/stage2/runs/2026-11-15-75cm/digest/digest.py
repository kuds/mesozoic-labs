"""A short digest of this run's records, for the written report.  It gives no verdict of its own.

Every number is read from the committed records under this run directory (simulation).  The full machine-made
tables are ``BAR_TABLES.md`` (``designlab.stage2.barreport``); this file only lays a few of the same counts and
medians side by side so that the report can quote them.  Only the standard library is used and nothing of the
study is imported.

    python stage2/runs/2026-11-15-75cm/digest/digest.py

It writes ``digest.json`` and ``DIGEST.md`` beside itself.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REC = re.compile(r"^(?P<set>[^/]+)/(?P<body>[^/]+)/(?P<block>seeds\d+)/eval/(?P<row>R2/\d+|R0)@(?P<delay>\d+)ms/"
                 r"(?P<behavior>[^/]+)/(?P<cond>[^/]+)\.json$")
SETS = ("V0", "F25F9", "F25F9cl")
ROWS = ("R2/42", "R2/43", "R0")
DELAYS = (0, 2, 4, 6, 8, 10)
NAMES = {"V0": "V0 (control: today's walk)", "F25F9": "F25F9 (control: the pull request's walk)",
         "F25F9cl": "F25F9cl (the candidate)"}


def med(values) -> float | None:
    v = [float(x) for x in values if x is not None]
    return statistics.median(v) if v else None


def get(d, *path):
    for k in path:
        if isinstance(k, int):
            if not isinstance(d, list) or len(d) <= k:
                return None
            d = d[k]
        elif not isinstance(d, dict) or d.get(k) is None:
            return None
        else:
            d = d[k]
    return d


def load() -> dict:
    out = {}
    for p in sorted(RUN.glob("*/*/seeds*/eval/**/*.json")):
        rel = p.relative_to(RUN).as_posix()
        m = REC.match(rel)
        if not m or "/traces/" in rel or "/partial/" in rel:
            continue
        raw = p.read_bytes()
        r = json.loads(raw)
        key = (m["set"], m["block"], m["row"], int(m["delay"]), m["behavior"], m["cond"])
        out[key] = dict(rel=rel, sha256=hashlib.sha256(raw).hexdigest(), r=r)
    return out


def digest_record(x: dict) -> dict:
    r = x["r"]
    d = dict(record=x["rel"], sha256=x["sha256"], status=r.get("status"))
    if r.get("status") != "done":
        return dict(d, reason=r.get("reason"))
    eps = r["episodes"]
    c, c3 = r["counts"], r["counts_but_A1"]
    fails: dict = {}
    for e in eps:
        for t in e["failed_terms"]:
            fails[t] = fails.get(t, 0) + 1
    win = [e for e in eps if get(e, "measures", "window", "complete")]
    bar = [get(e, "measures", "bar") or {} for e in eps]
    d.update(episodes=c["episodes"], up=c["survived"], every_limit=c["succeeded"], but_A1=c3["succeeded"],
             pass_every_limit=c.get("pass_"), pass_but_A1=c3.get("pass_"), falls=r.get("falls"),
             failed_terms=dict(sorted(fails.items(), key=lambda kv: (-kv[1], kv[0]))),
             n_window=len(win),
             speed_m_s=med(get(e, "measures", "window", "speed_m_s") for e in win),
             toe_mm=(lambda v: None if v is None else 1000.0 * v)(
                 med(b.get("toe_clearance_own_swing_m") for b in bar)),
             n_toe=sum(1 for b in bar if b.get("toe_clearance_own_swing_m") is not None),
             toe_limit_mm=(lambda v: None if v is None else 1000.0 * v)(
                 med(get(b, "toe", "limit_m") for b in bar)),
             duty=med(statistics.mean(b["duty"]) for b in bar if b.get("duty")),
             tail_ratio=med(get(b, "drives", "window", "ratios", "tail_yaw_servo") for b in bar),
             neck_ratio=med(get(b, "drives", "window", "ratios", "neck_pitch_servo") for b in bar),
             leg_ratio=med(get(b, "drives", "window", "worst_leg", 1) for b in bar),
             toe_stage1_mm=(lambda v: None if v is None else 1000.0 * v)(
                 med(get(e, "measures", "window", "toe_clearance_m") for e in win)),
             a1_over=bool(r.get("a1_over")),
             fall_segments=_count(e.get("segment_reached") for e in eps if e["fell"]),
             plan=[x[1] for x in (r.get("plan") or [])])
    beh = r["behavior"]
    if beh == "turn":
        d.update(heading_error_at_deg=med(b.get("heading_error_at_deg") for b in bar),
                 turned_own_deg=med(b.get("turned_own_deg") for b in bar),
                 time_to_within_10deg_s=med(b.get("time_to_within_10deg_s") for b in bar),
                 n_heading=sum(1 for b in bar if b.get("heading_error_at_deg") is not None))
    if beh == "nudge":
        rec = [get(e, "measures", "recovery") or {} for e in eps]
        d.update(recovered=sum(1 for q in rec if q.get("recovered")),
                 after_s=med(q.get("after_s") for q in rec if q.get("recovered")))
    if beh == "stop":
        d.update(touchdowns_after_command=med(b.get("touchdowns_after_command") for b in bar),
                 forward_stopped_strides=med(b.get("forward_stopped_strides") for b in bar),
                 distance_after_command_m=med(b.get("distance_after_command_m") for b in bar),
                 n_stop=sum(1 for b in bar if b.get("forward_stopped_strides") is not None))
    return d


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def cell(d: dict | None) -> str:
    if d is None:
        return "-"
    if d["status"] != "done":
        return "not run"
    if d.get("a1_over"):  # H3: reading 2 of a turn frozen with a1_over is recorded and gives no pass
        return f"{d['up']} / n.r. / {d['but_A1']}"
    return f"{d['up']} / {d['every_limit']} / {d['but_A1']}"


GATED = (("walk", "v0.4"), ("turn", "q90_left"), ("turn", "q90_right"), ("nudge", "forward"), ("nudge", "backward"),
         ("nudge", "left"), ("nudge", "right"), ("stop", "two_step"))
GATED = GATED + tuple((b, c + "_mu0.54") for b, c in GATED)
SEGMENT = {"start": "the start from standing", "start_v0.4": "the start from standing (straight into 0.40 m/s)",
           "nominal": "the slow stride after the start", "nominal_to_v0.4": "the change of speed to 0.40 m/s",
           "v0.4": "the 0.40 m/s walk itself"}


def meets(d: dict | None, key: str) -> bool:
    return bool(d and d["status"] == "done" and d["up"] == 20 and d[key] >= 17)


def num(v, fmt="{:.2f}") -> str:
    return "-" if v is None else fmt.format(v)


def markdown(doc: dict) -> str:
    R = doc["records"]

    def at(s, row, delay, beh, cond, block="seeds3042"):
        return R.get("|".join((s, block, row, str(delay), beh, cond)))

    L = ["# Digest of the 75 cm run's records (`2026-11-15-75cm`)", "",
         "Simulation only; no number here was measured on hardware. Every number is read from the committed records "
         "of this run by `digest/digest.py` (standard library only). It is a digest for the written report: the "
         "full machine-made tables, with every label and every record path, are `BAR_TABLES.md`. A dash is a cell "
         "the registered order does not hold; \"not run\" is a record the runner wrote as not run.", "",
         "Cells of the count tables: **episodes that stayed up / episodes that met every limit / episodes that met "
         "every limit but for A1** (the neck and tail servos' sustained load), each of 20, seeds 3042 to 3061. A "
         "record passes with 20 up and at least 17 (H1). Rows `R2/42` and `R2/43` are the two fitted balance "
         "controllers (fitted with no servo delay); `R0` has no balance feedback and gives no verdict. \"n.r.\" in "
         "the middle place: not read. That turn's reference is frozen with its tail servo over the planning limit, "
         "so only the reading but for A1 can pass (plan, H3).", ""]
    for grip, suffix in (("grippy floor (mu 0.9)", ""), ("slippery floor (mu 0.54)", "_mu0.54")):
        L += [f"## 1. The walk at 0.40 m/s, {grip}: counts by servo delay", "",
              "| Set | Row | " + " | ".join(f"{d} ms" for d in DELAYS) + " |", "|---|---|" + "---|" * len(DELAYS)]
        for s in SETS:
            for row in ROWS:
                L.append(f"| {NAMES[s]} | `{row}` | "
                         + " | ".join(cell(at(s, row, d, "walk", "v0.4" + suffix)) for d in DELAYS) + " |")
        L.append("")
    L += ["## 2. The walk at 0.40 m/s, grippy floor: measured medians", "",
          "Medians over the episodes that give the value (n in brackets). Speed over the last 10 s. Toe clearance "
          "as the bar judges it (limit 20.07 mm), and beside it as Stage 1 reads it (reported; no limit of the bar). "
          "Duty as the mean of the two feet (limits 0.55 to 0.65). Tail servo and neck servo: each one's "
          "sustained load over its rating in the window (limit 1.0 for each; the pair is A1). Leg servo: the "
          "largest leg drive's ratio (limit 1.0).", "",
          "| Set | Row | Delay | Speed (m/s) | Toe clearance (mm) | Toe clearance, Stage 1's reading (mm) | Duty | "
          "Tail servo | Neck servo | Leg servo | Failed terms (episodes of 20) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        for row in ROWS:
            for d in DELAYS:
                x = at(s, row, d, "walk", "v0.4")
                if x is None or x["status"] != "done":
                    continue
                ft = ", ".join(f"`{k}` {v}" for k, v in x["failed_terms"].items()) or "none"
                L.append(f"| {s} | `{row}` | {d} ms | {num(x['speed_m_s'], '{:.3f}')} ({x['n_window']}) | "
                         f"{num(x['toe_mm'], '{:.1f}')} ({x['n_toe']}) | {num(x['toe_stage1_mm'], '{:.1f}')} | "
                         f"{num(x['duty'], '{:.3f}')} | {num(x['tail_ratio'])} | {num(x['neck_ratio'])} | "
                         f"{num(x['leg_ratio'])} | {ft} |")
    L.append("")
    rungs = ("nominal", "v0.3", "v0.35", "v0.4", "v0.45", "v0.5")
    L += ["## 3. The speed ladder, grippy floor: counts by rung and delay", "",
          "`walk/nominal` is the set's slow walk (V0 and the candidate: 0.243 m/s). The ladder is read at 0, 4, 6 "
          "and 10 ms under the two fitted rows; the 0.40 m/s rung is section 1's record.", "",
          "| Set | Row | Delay | " + " | ".join(f"`{g}`" for g in rungs) + " |",
          "|---|---|---|" + "---|" * len(rungs)]
    for s in SETS:
        for d in (0, 4, 6, 10):
            for row in ("R2/42", "R2/43"):
                cells = [cell(at(s, row, d, "walk", g)) for g in rungs]
                if all(c == "-" for c in cells):
                    continue
                L.append(f"| {s} | `{row}` | {d} ms | " + " | ".join(cells) + " |")
    L.append("")
    rest = sorted({(k.split("|")[4], k.split("|")[5]) for k in R if k.split("|")[4] != "walk"})
    L += ["## 4. Turns, nudges and stops: counts by delay", "",
          "Each of these episodes walks at 0.40 m/s first; the command or push comes after about 13 s. Falls are "
          "split into before and after the command (b / a).", "",
          "| Set | Condition | Row | " + " | ".join(f"{d} ms" for d in (0, 4, 6, 10)) + " |", "|---|---|---|---|---|---|---|"]
    for s in SETS:
        for beh, cond in rest:
            for row in ROWS:
                cells = []
                for d in (0, 4, 6, 10):
                    x = at(s, row, d, beh, cond)
                    if x is None:
                        cells.append("-")
                    elif x["status"] != "done":
                        cells.append("not run")
                    else:
                        f = x.get("falls") or {}
                        cells.append(f"{cell(x)} (falls {f.get('before_command', 0)} b / {f.get('after_command', 0)} a)")
                if all(c == "-" for c in cells):
                    continue
                L.append(f"| {s} | `{beh}/{cond}` | `{row}` | " + " | ".join(cells) + " |")
    L.append("")
    L += ["## 5. Turns: measured medians where the turn was reached", "",
          "Heading error at the judged instant (limit 10 degrees), the angle the trunk turned (commanded 90), the "
          "time to come within 10 degrees, and the toe clearance over the turn's window (limit 20.07 mm).", "",
          "| Set | Condition | Row | Delay | Up | Heading error (deg) | Turned (deg) | Time to within 10 deg (s) | "
          "Toe clearance (mm) | Failed terms (episodes of 20) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        for beh, cond in rest:
            if beh != "turn":
                continue
            for row in ROWS:
                for d in (0, 4, 6, 10):
                    x = at(s, row, d, beh, cond)
                    if x is None or x["status"] != "done" or not x.get("n_heading"):
                        continue
                    ft = ", ".join(f"`{k}` {v}" for k, v in x["failed_terms"].items()) or "none"
                    L.append(f"| {s} | `{beh}/{cond}` | `{row}` | {d} ms | {x['up']} | "
                             f"{num(x['heading_error_at_deg'], '{:.1f}')} ({x['n_heading']}) | "
                             f"{num(x['turned_own_deg'], '{:.1f}')} | {num(x['time_to_within_10deg_s'])} | "
                             f"{num(x['toe_mm'], '{:.1f}')} ({x['n_toe']}) | {ft} |")
    L.append("")
    L += ["## 6. Nudges and stops: measured", "",
          "Nudges: episodes that came back onto the command within 3 s (of 20), and the median time for those. "
          "Stops: median touchdowns after the command, strides until forward motion stopped, and distance covered "
          "after the command.", "",
          "| Set | Condition | Row | Delay | Up | Recovered | Median time back (s) | Touchdowns | Strides to stop | "
          "Distance (m) | Failed terms (episodes of 20) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        for beh, cond in rest:
            if beh == "turn":
                continue
            for row in ROWS:
                for d in (0, 4, 6, 10):
                    x = at(s, row, d, beh, cond)
                    if x is None or x["status"] != "done":
                        continue
                    ft = ", ".join(f"`{k}` {v}" for k, v in x["failed_terms"].items()) or "none"
                    L.append(f"| {s} | `{beh}/{cond}` | `{row}` | {d} ms | {x['up']} | "
                             f"{x.get('recovered', '-')} | {num(x.get('after_s'))} | "
                             f"{num(x.get('touchdowns_after_command'), '{:.0f}')} | "
                             f"{num(x.get('forward_stopped_strides'))} | {num(x.get('distance_after_command_m'))} | "
                             f"{ft} |")
    L.append("")
    L += ["## 7. Where the falls come: the walk at 0.40 m/s, fitted rows", "",
          "Each episode starts from standing, takes the set's lead-in, and then walks at 0.40 m/s (V0 and the "
          "candidate: the start, one slow stride, the change of speed; F25F9: straight from the start). The table "
          "counts the falls of the two fitted rows on both floors together (80 episodes per cell) by the part of "
          "the episode the robot had reached.", "",
          "| Set | Delay | Falls of 80 | Where (falls) |", "|---|---|---|---|"]
    order = list(SEGMENT)
    for s in SETS:
        for d in DELAYS:
            seg: dict = {}
            n = 0
            for row in ("R2/42", "R2/43"):
                for cond in ("v0.4", "v0.4_mu0.54"):
                    x = at(s, row, d, "walk", cond)
                    if x is None or x["status"] != "done":
                        continue
                    for k, v in x["fall_segments"].items():
                        seg[k] = seg.get(k, 0) + v
                        n += v
            where = "; ".join(f"{SEGMENT.get(k, k)}: {seg[k]}" for k in order + sorted(set(seg) - set(order))
                              if k in seg) or "-"
            L.append(f"| {s} | {d} ms | {n} | {where} |")
    L.append("")
    L += ["## 8. The bar's 16 conditions that meet the pass rule under both fitted rows, per delay", "",
          "The pass rule at a delay other than 10 ms gives no verdict (H2); the count is given for scale. The turns, "
          "pushes and stop were run at 0, 4, 6 and 10 ms; on F25F9 only the two walk conditions were run.", "",
          "| Set | Delay | Every limit | Every limit but for A1 | Which (but for A1) |", "|---|---|---|---|---|"]
    for s in SETS:
        for d in (0, 4, 6, 10):
            run = [(b, c) for b, c in GATED if at(s, "R2/42", d, b, c) and at(s, "R2/43", d, b, c)]
            p2 = [f"{b}/{c}" for b, c in run if all(meets(at(s, r, d, b, c), "every_limit")
                                                   and not at(s, r, d, b, c).get("a1_over") for r in ("R2/42", "R2/43"))]
            p3 = [f"{b}/{c}" for b, c in run if all(meets(at(s, r, d, b, c), "but_A1") for r in ("R2/42", "R2/43"))]
            L.append(f"| {s} | {d} ms | {len(p2)} of {len(run)} run | {len(p3)} of {len(run)} run | "
                     f"{', '.join('`' + x + '`' for x in p3) or 'none'} |")
    L.append("")
    if doc.get("controls"):
        L += ["## 9. The registered control: omega at Stage 1's crouch", "",
              "The candidate's walk at 0.40 m/s under `R2/42` with the controller's pendulum frequency computed as "
              "Stage 1 computed it (plan B5). 20 episodes each; a control, no verdict.", "",
              "| Record | Stayed up | Every limit | But for A1 |", "|---|---|---|---|"]
        for k, x in sorted(doc["controls"].items()):
            L.append(f"| `{k}` | {x['up']} | {x['every_limit']} | {x['but_A1']} |")
        L.append("")
    b = doc["budget"]
    L += ["## 10. Compute", "", "| Task | Budget (core-hours) | Used (core-hours) | Units started | Units cut |",
          "|---|---|---|---|---|"]
    for t, v in sorted(b.items()):
        L.append(f"| {t} | {v['cap_core_h']:.0f} | {v['used_core_h']:.2f} | {v['units']} | {v['cut']} |")
    L += ["", f"Records read: {doc['counts']['records']} ({doc['counts']['done']} done, "
              f"{doc['counts']['not_run']} written as not run)."]
    return "\n".join(L)


def main() -> None:
    recs = load()
    records = {"|".join(map(str, k)): digest_record(x) for k, x in recs.items()}
    budget = {}
    bj = RUN / "budget.json"
    if bj.exists():
        for t, v in json.loads(bj.read_text())["tasks"].items():
            units = v.get("units") or []
            budget[t] = dict(cap_core_h=v["cap_core_h"], units=len(units), cut=len(v.get("cut") or []),
                             used_core_h=sum(u.get("used_core_h") or 0.0 for u in units))
    controls = {}
    for p in sorted((RUN / "controls").glob("**/eval/**/*.json")) if (RUN / "controls").exists() else ():
        rel = p.relative_to(RUN).as_posix()
        if "/traces/" in rel or rel.endswith("summary.json"):
            continue
        raw = p.read_bytes()
        controls[rel] = digest_record(dict(rel=rel, sha256=hashlib.sha256(raw).hexdigest(), r=json.loads(raw)))
    doc = dict(schema="digest-1", run=RUN.name, records=records, budget=budget, controls=controls,
               counts=dict(records=len(records), done=sum(1 for d in records.values() if d["status"] == "done"),
                           not_run=sum(1 for d in records.values() if d["status"] != "done")))
    (HERE / "digest.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (HERE / "DIGEST.md").write_text(markdown(doc) + "\n")
    print(f"records read: {doc['counts']['records']} ({doc['counts']['done']} done, {doc['counts']['not_run']} not run)")


if __name__ == "__main__":
    main()
