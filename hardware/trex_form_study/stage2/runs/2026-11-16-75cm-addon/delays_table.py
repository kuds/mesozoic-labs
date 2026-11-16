"""Today's walk (set V0, body 75-M-H-10) against servo delay: the tables of STAGE2_75_ADDON_PLAN.md section A.

It gives no verdict.  Every number is read from committed records (simulation): the registered records of the
75 cm panels under ``stage2/runs/2026-11-15-75cm/`` at 0 to 10 ms, and this add-on's records under ``A/`` beside
this file.  The words (up, down, mixed; the patterns; the mirror estimate's "met" or "not met") are those of rule
A-6, written before any record of the add-on existed.  Only the standard library is used and nothing of the study
is imported.  This file is in no record's key.

    python stage2/runs/2026-11-16-75cm-addon/delays_table.py

It writes ``DELAYS_TABLE.md`` and ``delays_table.json`` beside itself.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[2]
REGISTERED = HERE.parent / "2026-11-15-75cm"
ADDON = HERE / "A"
BASE = "V0/75-M-H-10/seeds3042/eval"
DELAYS = tuple(range(0, 21, 2))
FITS = ("R2/42", "R2/43")
FAST = ("v0.4", "v0.4_mu0.54")
FLOOR = {"v0.4": "grippy floor", "v0.4_mu0.54": "slippery floor", "nominal": "grippy floor"}
UP_AT_LEAST, DOWN_AT_MOST = 17, 3            # A-6: the reading words
JUDGED_MS = 10
MIRROR = {12: "down", 14: "up", 16: "up", 18: "up", 20: "up"}   # A-6: the design estimate's five predictions
LABEL = ("simulation; stand-in controller fitted with no delay; a delay that is not the judged 10 ms; no verdict; "
         "added after the 75 cm results")
UNIT_LABEL = {"A1": "(P37: 14, 16 and 20 ms)", "A2": "the lead's addition, not among the delays the owner approved",
              "A3": "the lead's addition, not among the delays the owner approved"}
SEGMENT = {"start": "start from standing", "nominal": "slow stride", "nominal_to_v0.4": "change of speed",
           "v0.4": "0.40 m/s walk itself"}


def med(values):
    v = [float(x) for x in values if x is not None]
    return statistics.median(v) if v else None


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def unit_of(ms: int, cond: str, source: str) -> str | None:
    if source != "add-on":
        return None
    if ms in (14, 16, 20):
        return "A1"
    if ms in (12, 18):
        return "A2"
    return "A3" if (ms == 8 and cond == "nominal") else "?"


def load(row: str, ms: int, cond: str, reference: dict | None) -> dict | None:
    """One record's reading, or None when no record exists.  A record is read only when it is done, holds 20
    episodes none of which failed to run, and carries the registered code state (A-3, A-4)."""
    rel = f"{BASE}/{row}@{ms}ms/walk/{cond}.json"
    found = [(src, root / rel) for src, root in (("registered", REGISTERED), ("add-on", ADDON)) if (root / rel).exists()]
    if not found:
        return None
    if len(found) > 1:
        raise SystemExit(f"{rel}: a record under both run directories")
    source, path = found[0]
    raw = path.read_bytes()
    r = json.loads(raw)
    d = dict(row=row, ms=ms, condition=cond, source=source, unit=unit_of(ms, cond, source),
             path=path.relative_to(STUDY).as_posix(), sha256=hashlib.sha256(raw).hexdigest(), status=r.get("status"),
             read=False, not_read=None)
    if r.get("status") != "done":
        return dict(d, not_read=f"status {r.get('status')}")
    eps = r.get("episodes") or []
    c, c3 = r.get("counts") or {}, r.get("counts_but_A1") or {}
    if len(eps) != 20 or c.get("errors") or not c.get("judged"):
        return dict(d, not_read=f"an episode failed to run ({len(eps)} episodes, errors {c.get('errors')})")
    key = r.get("key") or {}
    if reference is not None and (key.get("sources") != reference["sources"] or key.get("context") != reference["context"]):
        return dict(d, not_read="another code state")
    steps = get(r, "controller", "latency_steps")
    if steps is None:
        steps = get(r, "key", "row", "latency_steps")
    up = [e for e in eps if not e["fell"]]
    down = [e for e in eps if e["fell"]]
    fails: dict = {}
    for e in up:
        for t in e.get("failed_terms") or []:
            fails[t] = fails.get(t, 0) + 1
    segs: dict = {}
    for e in down:
        segs[str(e.get("segment_reached"))] = segs.get(str(e.get("segment_reached")), 0) + 1
    speed_ratio = max(((e["worst_speed_ratio"][1], e["worst_speed_ratio"][0]) for e in up
                       if e.get("worst_speed_ratio")), default=None)
    n_up = len(up)
    along = sum(1 for e in up if get(e, "success", "along"))
    word = "up" if n_up >= UP_AT_LEAST else "down" if n_up <= DOWN_AT_MOST else "mixed"
    second = None
    if word == "up":
        second = "up and covering the distance" if along >= UP_AT_LEAST else "up, short of the distance"
    d.update(read=True, episodes=len(eps), up=n_up, along=along, word=word, second_word=second,
             stage1_speed=(sum(1 for e in up if get(e, "measures", "stage1", "success", "speed"))
                           if cond == "nominal" else None),
             every_limit=c.get("succeeded"), but_A1=c3.get("succeeded"),
             pass_every_limit=bool(c.get("pass_")), pass_but_A1=bool(c3.get("pass_")),
             latency_steps=steps, latency_ok=(steps == ms // 2),
             fall_segments=dict(sorted(segs.items())), fall_time_median_s=med(e.get("fall_time_s") for e in down),
             up_speed_median_m_s=med(get(e, "measures", "window", "speed_m_s") for e in up
                                     if get(e, "measures", "window", "complete")),
             up_toe_median_mm=(lambda v: None if v is None else 1000.0 * v)(
                 med(get(e, "measures", "bar", "toe_clearance_own_swing_m") for e in up)),
             up_toe_limit_mm=(lambda v: None if v is None else 1000.0 * v)(
                 med(get(e, "measures", "bar", "toe", "limit_m") for e in up)),
             up_failed_terms=dict(sorted(fails.items(), key=lambda kv: (-kv[1], kv[0]))),
             up_speed_ratio_max=(None if speed_ratio is None else dict(ratio=speed_ratio[0], drive=speed_ratio[1])),
             up_grf_peak_median_bw=(med(get(e, "measures", "window", "grf_peak_bw") for e in up)
                                    if cond == "nominal" else None))
    return d


def delay_word(records: list, flagged: bool) -> dict:
    """A delay's word from its R2 records (four for the 0.40 m/s walk, two for the slow walk), by A-6."""
    if all(r is None for r in records):
        return dict(word=None, why="no record")
    if any(r is None or not r["read"] for r in records):
        return dict(word=None, why="not read")
    if flagged:
        return dict(word=None, why="flagged: an R0 record of the 0.40 m/s walk under 17")
    words = {r["word"] for r in records}
    word = "up" if words == {"up"} else "down" if words == {"down"} else "mixed"
    return dict(word=word, why=None, counts=[r["up"] for r in records],
                covering=all(r["second_word"] == "up and covering the distance" for r in records) if word == "up" else None)


def pattern(words: dict, name: str) -> dict:
    """The pattern of one walk from its delays' words (A-6).  ``words``: delay -> delay_word(...)."""
    above = {ms: w for ms, w in words.items() if ms > JUDGED_MS and w["word"]}
    if not above:
        return dict(pattern=None, sentence=f"No delay above 10 ms has a word for {name}: no pattern is named.")
    ups = sorted(ms for ms, w in above.items() if w["word"] == "up")
    if all(w["word"] == "down" for w in above.values()):
        return dict(pattern="delay simply gets worse", delays=sorted(above),
                    sentence=(f"Delay simply gets worse for {name}: every delay run above 10 ms "
                              f"({', '.join(map(str, sorted(above)))} ms) reads down."))
    if ups:
        if name.startswith("the 0.40") and not all(words[m]["word"] == "down" for m in (8, 10)):
            return dict(pattern=None, sentence=f"8 and 10 ms do not both read down for {name}: no pattern is named.")
        parts = []
        for ms in ups:
            parts.append(f"{ms} ms: " + ("up and covering the distance" if above[ms]["covering"]
                                         else "a window of staying up, not of walking"))
        return dict(pattern="a window", lowest_up_ms=ups[0], up=ups,
                    sentence=(f"A window for {name}: the lowest delay above 10 ms that reads up is {ups[0]} ms. "
                              + "; ".join(parts) + "."))
    return dict(pattern="neither", sentence=(f"Neither phrase for {name}: no delay above 10 ms reads up and at least "
                                             f"one is mixed."))


def runner_word(pair: list, ms: int, key: str) -> str:
    """H2's word for the two fits of one condition at a delay that is not the judged one (bar.py, the pair's word)."""
    if any(r is None for r in pair):
        return "-"
    if any(not r["read"] for r in pair):
        return "not read"
    if ms == JUDGED_MS:
        return "judged delay: see BAR_TABLES.md"
    p = [r[key] for r in pair]
    return (f"passes at {ms} ms" if all(p) else f"one of two passes at {ms} ms" if any(p) else f"does not pass at {ms} ms")


def num(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def cell(r, slow=False) -> str:
    if r is None:
        return "-"
    if not r["read"]:
        return f"not read: {r['not_read']}"
    s = f"{r['up']} · {r['along']} · {r['every_limit']} · {r['but_A1']}"
    return s + (f" (Stage 1 speed {r['stage1_speed']})" if slow else "")


def main() -> None:
    ref_path = REGISTERED / f"{BASE}/R2/42@10ms/walk/v0.4.json"
    ref_key = json.loads(ref_path.read_text())["key"]
    reference = dict(sources=ref_key["sources"], context=ref_key["context"])
    recs = {(row, ms, cond): load(row, ms, cond, reference)
            for ms in DELAYS for row in FITS + ("R0",) for cond in FAST + ("nominal",)}
    present = [r for r in recs.values() if r is not None]
    flags = {ms: [r for cond in FAST for r in [recs[("R0", ms, cond)]] if r and r["read"] and r["up"] < UP_AT_LEAST]
             for ms in DELAYS}
    fast = {ms: delay_word([recs[(f, ms, c)] for c in FAST for f in FITS], bool(flags[ms])) for ms in DELAYS}
    slow = {ms: delay_word([recs[(f, ms, "nominal")] for f in FITS], bool(flags[ms])) for ms in DELAYS}
    p_fast, p_slow = pattern(fast, "the 0.40 m/s walk"), pattern(slow, "the slow walk")
    run_above = [ms for ms in MIRROR if fast[ms]["why"] != "no record"]
    differ = [ms for ms in run_above if fast[ms]["word"] != MIRROR[ms]]
    mirror = dict(predictions=MIRROR, delays_run=run_above, differ=differ,
                  read={ms: (fast[ms]["word"] or fast[ms]["why"]) for ms in run_above},
                  result=(None if not run_above else "met" if not differ else "not met"))
    addon = [r for r in present if r["source"] == "add-on"]
    exp = dict(
        e1_R0_fast={ms: [r["up"] for c in FAST for r in [recs[("R0", ms, c)]] if r and r["read"]] for ms in DELAYS},
        e1_flags=[ms for ms in DELAYS if flags[ms]],
        e1_latency_steps_ok=all(r["latency_ok"] for r in present if r["read"]),
        e2_slow_R2_43_20ms=(lambda r: None if r is None else (r["word"] if r["read"] else "not read"))(recs[("R2/43", 20, "nominal")]),
        e4_records_meeting_every_limit_17=[r["path"] for r in addon if r["read"] and r["every_limit"] >= UP_AT_LEAST])
    doc = dict(schema="addon-delays-table-1", rule="STAGE2_75_ADDON_PLAN.md A-5, A-6, A-8, A-10", label=LABEL,
               reference=dict(record=ref_path.relative_to(STUDY).as_posix(), sources=reference["sources"]),
               records={f"{row}@{ms}ms|{cond}": r for (row, ms, cond), r in recs.items() if r is not None},
               fast_words=fast, slow_words=slow, pattern_fast=p_fast, pattern_slow=p_slow, mirror=mirror,
               expectations=exp)
    (HERE / "delays_table.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    def src(ms, conds):
        s = {r["source"] + (f" {r['unit']}" if r["unit"] else "") for c in conds for row in FITS + ("R0",)
             for r in [recs[(row, ms, c)]] if r}
        return ", ".join(sorted(s)) or "-"

    def word(w):
        return w["word"] or f"no word ({w['why']})"

    L = ["# Today's walk against servo delay (set V0, body 75-M-H-10)", "",
         f"**{LABEL}.** Made by `stage2/runs/2026-11-16-75cm-addon/delays_table.py` from the committed records; the",
         "words and patterns are those of `STAGE2_75_ADDON_PLAN.md` A-6, written before any add-on record existed.",
         "Records at 14, 16 and 20 ms: (P37: 14, 16 and 20 ms). Records at 12 and 18 ms and the slow walk at 8 ms:",
         "the lead's addition, not among the delays the owner approved. The slow walk and the `R0` rows at 14, 16 and",
         "20 ms are the lead's reading of \"today's walk\". Records at 0 to 10 ms are the registered records of the",
         "75 cm panels. 10 ms is the judged delay; nothing here changes its verdict.", "",
         "A record is **up** when at least 17 of its 20 runs stay up, **down** when at most 3 do, otherwise **mixed**.",
         "Each cell reads: runs that stayed up · of those, runs that also covered the distance · runs that met every",
         "limit · runs that met every limit but for A1 (the neck and tail servo planning limit). All of 20.", "",
         "## 1. The 0.40 m/s walk", "",
         "| Delay | Records | Fit 42, grippy | Fit 43, grippy | Fit 42, slippery | Fit 43, slippery | No feedback (`R0`): up, grippy · slippery | The delay reads |",
         "|---|---|---|---|---|---|---|---|"]
    for ms in DELAYS:
        r0 = " · ".join("-" if r is None else (str(r["up"]) if r["read"] else "not read")
                        for r in (recs[("R0", ms, c)] for c in FAST))
        L.append(f"| {ms} ms | {src(ms, FAST)} | " + " | ".join(cell(recs[(f, ms, c)]) for c in FAST for f in FITS)
                 + f" | {r0}{' **flag**' if flags[ms] else ''} | {word(fast[ms])} |")
    L += ["", p_fast["sentence"], "", "The runner's word for the two fits together (H2), per floor and reading:", "",
          "| Delay | Grippy, every limit | Grippy, but for A1 | Slippery, every limit | Slippery, but for A1 |",
          "|---|---|---|---|---|"]
    for ms in DELAYS:
        L.append(f"| {ms} ms | " + " | ".join(runner_word([recs[(f, ms, c)] for f in FITS], ms, k)
                                              for c in FAST for k in ("pass_every_limit", "pass_but_A1")) + " |")
    L += ["", "## 2. The slow walk (its own reading; not merged with section 1)", "",
          "| Delay | Records | Fit 42 | Fit 43 | No feedback (`R0`): up | The delay reads |", "|---|---|---|---|---|---|"]
    for ms in DELAYS:
        r0 = recs[("R0", ms, "nominal")]
        L.append(f"| {ms} ms | {src(ms, ('nominal',))} | " + " | ".join(cell(recs[(f, ms, 'nominal')], slow=True) for f in FITS)
                 + f" | {'-' if r0 is None else r0['up'] if r0['read'] else 'not read'} | {word(slow[ms])} |")
    L += ["", p_slow["sentence"], "",
          "`R0` on the slow walk is the lead's addition; it is printed and raises no flag.", "",
          "## 3. The mirror estimate (a design estimate, not a result)", "",
          "Prediction: the 0.40 m/s walk reads down at 12 ms and up at 14, 16, 18 and 20 ms.", "",
          "| Delay | Predicted | Read |", "|---|---|---|"]
    for ms in MIRROR:
        L.append(f"| {ms} ms | {MIRROR[ms]} | {mirror['read'].get(ms, 'not run')} |")
    L += ["", ("No delay above 10 ms was run." if mirror["result"] is None else
               f"**The five predictions are {mirror['result']}** on the delays that were run"
               + ("." if not differ else f": {', '.join(f'{m} ms' for m in differ)} differ" + "."))
          + " No other centre is fitted after the result.", "",
          "## 4. Where the falls come", "",
          "Classified by the runs' `segment_reached` alone. Fall times are printed and classify nothing.", "",
          "| Delay | Walk | Fit and floor | Fell | Where (runs) | Median fall time |", "|---|---|---|---|---|---|"]
    for ms in DELAYS:
        for cond in FAST + ("nominal",):
            for f in FITS:
                r = recs[(f, ms, cond)]
                if r is None or not r["read"] or r["up"] == 20:
                    continue
                where = ", ".join(f"{SEGMENT.get(k, k)} {v}" for k, v in r["fall_segments"].items())
                L.append(f"| {ms} ms | {'slow' if cond == 'nominal' else '0.40 m/s'} | {f[3:]}, {FLOOR[cond]} | "
                         f"{20 - r['up']} | {where} | {num(r['fall_time_median_s'], '{:.1f}')} s |")
    L += ["", "## 5. Of the runs that stayed up (not judged)", "",
          "| Delay | Walk | Row and floor | Up | Median speed | Median toe clearance (limit) | Failed limits (runs) | Largest drive speed ratio | Median peak ground force |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ms in DELAYS:
        for cond in FAST + ("nominal",):
            for row in FITS + ("R0",):
                r = recs[(row, ms, cond)]
                if r is None or not r["read"] or r["up"] == 0:
                    continue
                ft = ", ".join(f"{k} {v}" for k, v in r["up_failed_terms"].items()) or "none"
                sr = r["up_speed_ratio_max"]
                L.append(f"| {ms} ms | {'slow' if cond == 'nominal' else '0.40 m/s'} | {row}, {FLOOR[cond]} | {r['up']} | "
                         f"{num(r['up_speed_median_m_s'], '{:.3f}')} m/s | {num(r['up_toe_median_mm'], '{:.1f}')} mm "
                         f"({num(r['up_toe_limit_mm'], '{:.1f}')}) | {ft} | "
                         f"{'-' if sr is None else '{:.2f} ({})'.format(sr['ratio'], sr['drive'])} | "
                         f"{num(r['up_grf_peak_median_bw'])}{' body weights' if r['up_grf_peak_median_bw'] is not None else ''} |")
    e1 = sorted({v for ms in DELAYS for v in exp["e1_R0_fast"][ms]})
    L += ["", "## 6. The expectations of A-6, beside the result", "",
          f"1. `R0` on the 0.40 m/s walk, expected 18 to 20 of 20 up at every delay: read {e1[0] if e1 else '-'} to "
          f"{e1[-1] if e1 else '-'}. Flags (an `R0` record under 17): "
          f"{', '.join(f'{m} ms' for m in exp['e1_flags']) or 'none'}. Every read record holds "
          f"`latency_steps` = ms / 2: {'yes' if exp['e1_latency_steps_ok'] else 'NO'}.",
          f"2. The slow walk under fit 43 at 20 ms, expected up: read {exp['e2_slow_R2_43_20ms'] or 'not run'}.",
          "3. No expectation was stated for fit 42 at 20 ms, for 12 to 18 ms, or for the 0.40 m/s walk at 20 ms.",
          f"4. No add-on record was expected to meet every limit in 17 of 20: "
          f"{len(exp['e4_records_meeting_every_limit_17'])} do"
          + ("." if not exp["e4_records_meeting_every_limit_17"] else ": " + ", ".join(exp["e4_records_meeting_every_limit_17"]) + "."),
          "", "## 7. The records", "", "| Record | From | sha256 (first 12) |", "|---|---|---|"]
    for r in sorted(present, key=lambda r: (r["ms"], r["condition"], r["row"])):
        frm = r["source"] + (f" {r['unit']}: {UNIT_LABEL.get(r['unit'], '')}" if r["unit"] else "")
        L.append(f"| `{r['path']}` | {frm} | `{r['sha256'][:12]}` |")
    (HERE / "DELAYS_TABLE.md").write_text("\n".join(L) + "\n")
    print(f"{len(present)} records ({len(addon)} of the add-on); 0.40 m/s: {p_fast['pattern']}; slow: {p_slow['pattern']}; "
          f"mirror: {mirror['result']}")


if __name__ == "__main__":
    main()
