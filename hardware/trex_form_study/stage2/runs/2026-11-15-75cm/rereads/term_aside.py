"""A re-reading of this run's records with one success term set aside.  NOT a verdict.

Label (STAGE2_75_PLAN.md, closing amendments, item 11): made on 15 Nov 2026 AFTER the walk records were read.
The registered rules are unchanged and the registered counts are the records' own; this file only counts, from
the same records' episodes, how many episodes fail nothing but the named term.  It runs no episode and writes
nothing into a record.

    python stage2/runs/2026-11-15-75cm/rereads/term_aside.py [--term joint_range]

It writes ``<term>_aside.json`` and ``<TERM>_ASIDE.md`` beside itself.  Only the standard library is used, and
nothing of the study is imported, so the numbers do not depend on the runner's code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REC = re.compile(r"^(?P<set>[^/]+)/(?P<body>[^/]+)/(?P<block>seeds\d+)/eval/(?P<row>R2/\d+|R0)@(?P<delay>\d+)ms/"
                 r"(?P<behavior>[^/]+)/(?P<cond>[^/]+)\.json$")
NEED_UP, NEED_OK = 20, 17  # H1: 20 of 20 up and at least 17 of 20 succeed


def read(term: str) -> dict:
    rows = []
    for p in sorted(RUN.glob("*/*/seeds*/eval/**/*.json")):
        rel = p.relative_to(RUN).as_posix()
        m = REC.match(rel)
        if not m or "/traces/" in rel or "/partial/" in rel:
            continue
        raw = p.read_bytes()
        r = json.loads(raw)
        if r.get("status") != "done":
            continue
        eps = r["episodes"]
        up = sum(1 for e in eps if not e["fell"])
        ok2 = sum(1 for e in eps if not e["failed_terms"])
        ok3 = sum(1 for e in eps if not e["failed_terms_but_A1"])
        aside2 = sum(1 for e in eps if not set(e["failed_terms"]) - {term})
        aside3 = sum(1 for e in eps if not set(e["failed_terms_but_A1"]) - {term})
        # Stage 1's reading of a walk (C4: printed for the comparison with JPS 2.3; it gives no verdict)
        st = [((e.get("measures") or {}).get("stage1") or None) for e in eps]
        has_st = any(x is not None for x in st)
        st_ok = sum(1 for x in st if x and x.get("succeeded")) if has_st else None
        st_aside = (sum(1 for e, x in zip(eps, st) if x and not e["fell"] and not set(x.get("failed_terms") or ()) - {term})
                    if has_st else None)
        fails_term = sum(1 for e in eps if term in e["failed_terms"])
        only_term = sum(1 for e in eps if set(e["failed_terms"]) == {term})
        margins = sorted(e["joint_margin_min_rad"] for e in eps if not e["fell"])
        c, c3 = r["counts"], r["counts_but_A1"]
        if (c["episodes"], c["survived"], c["succeeded"], c3["succeeded"]) != (len(eps), up, ok2, ok3):
            raise SystemExit(f"{rel}: the record's counts do not recompute from its episodes")
        rows.append(dict(m.groupdict(), delay_ms=int(m["delay"]), record=rel, sha256=hashlib.sha256(raw).hexdigest(),
                         episodes=len(eps), up=up, succeeded=ok2, succeeded_but_A1=ok3,
                         fails_term=fails_term, fails_only_term=only_term,
                         succeeded_term_aside=aside2, succeeded_but_A1_term_aside=aside3,
                         stage1_succeeded=st_ok, stage1_succeeded_term_aside=st_aside,
                         joint_margin_min_rad=([margins[0], margins[-1]] if margins else None)))
    for x in rows:
        del x["delay"]
    return dict(schema="term-aside-1", term=term, run=RUN.name,
                label="re-reading after a result (STAGE2_75_PLAN.md, closing amendments, item 11); no verdict; "
                      "no rule changed",
                rule=f"registered pass (H1): {NEED_UP} of 20 up and at least {NEED_OK} succeed; 'meets' below is "
                     f"that arithmetic on the counts with the term set aside, and is not a pass",
                records=rows)


def pairs(doc: dict) -> list[dict]:
    """Per set, block, behaviour, condition and delay: the two fitted rows side by side, and R0."""
    by: dict = {}
    for x in doc["records"]:
        by.setdefault((x["set"], x["block"], x["behavior"], x["cond"], x["delay_ms"]), {})[x["row"]] = x
    out = []
    for (s, block, beh, cond, d), rows in sorted(by.items()):
        r2 = [rows.get(k) for k in ("R2/42", "R2/43")]

        def meets(key: str) -> bool | None:
            if any(x is None for x in r2):
                return None
            return all(x["up"] == NEED_UP and x[key] >= NEED_OK for x in r2)

        out.append(dict(set=s, block=block, behavior=beh, cond=cond, delay_ms=d, rows=rows,
                        both_r2_registered=meets("succeeded"), both_r2_registered_but_A1=meets("succeeded_but_A1"),
                        both_r2_meet_term_aside=meets("succeeded_term_aside"),
                        both_r2_meet_but_A1_term_aside=meets("succeeded_but_A1_term_aside")))
    return out


def word(v: bool | None) -> str:
    return "-" if v is None else ("yes" if v else "no")


def markdown(doc: dict) -> str:
    term = doc["term"]
    L = [f"# The records of `{doc['run']}` with the term `{term}` set aside", "",
         f"**{doc['label']}.**", "",
         "Simulation only. Every count is taken from the committed records' own episodes; nothing was run for this "
         "file. The first three count columns are the records' registered counts. The columns after them count the "
         f"episodes that fail no term other than `{term}`. {doc['rule'][0].upper() + doc['rule'][1:]}.", "",
         "Cells: episodes up / succeed on every limit / succeed but for A1, of 20; then the same two success counts "
         f"with `{term}` set aside; then Stage 1's reading of a walk (the reading the pull request's report names; "
         f"it gives no verdict here) as recorded and with `{term}` set aside; then how many episodes fail "
         f"`{term}`, and the smallest joint margin over the episode among the episodes that stayed up (rad, lowest "
         "to highest).", ""]
    sets = sorted({x["set"] for x in doc["records"]})
    for s in sets:
        L += [f"## {s}", "",
              f"| Block | Condition | Delay | Row | Up | Every limit | But for A1 | Every limit, `{term}` aside | "
              f"But for A1, `{term}` aside | Stage 1's reading | Stage 1's reading, `{term}` aside | "
              f"Fail `{term}` | Smallest joint margin (rad) | "
              f"Both fitted rows meet 20 up and 17, `{term}` aside (every limit / but for A1) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in pairs(doc):
            if g["set"] != s:
                continue
            first = True
            for row in ("R2/42", "R2/43", "R0"):
                x = g["rows"].get(row)
                if x is None:
                    continue
                jm = x["joint_margin_min_rad"]
                both = (f"{word(g['both_r2_meet_term_aside'])} / {word(g['both_r2_meet_but_A1_term_aside'])}"
                        if first else "")
                L.append(f"| {g['block']} | `{g['behavior']}/{g['cond']}` | {g['delay_ms']} ms | {row} | {x['up']} | "
                         f"{x['succeeded']} | {x['succeeded_but_A1']} | {x['succeeded_term_aside']} | "
                         f"{x['succeeded_but_A1_term_aside']} | "
                         f"{'-' if x['stage1_succeeded'] is None else x['stage1_succeeded']} | "
                         f"{'-' if x['stage1_succeeded_term_aside'] is None else x['stage1_succeeded_term_aside']} | "
                         f"{x['fails_term']} | "
                         f"{'-' if jm is None else f'{jm[0]:.4f} to {jm[1]:.4f}'} | {both} |")
                first = False
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="joint_range")
    a = ap.parse_args()
    doc = read(a.term)
    doc["conditions"] = [{k: v for k, v in g.items() if k != "rows"} for g in pairs(doc)]
    (HERE / f"{a.term}_aside.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (HERE / f"{a.term.upper()}_ASIDE.md").write_text(markdown(doc) + "\n")
    n = len(doc["records"])
    touched = sum(1 for x in doc["records"] if x["fails_term"])
    print(f"records read: {n}; records in which `{a.term}` fails in at least one episode: {touched}")
    for s in sorted({x['set'] for x in doc['records']}):
        k = sum(1 for x in doc["records"] if x["set"] == s and x["fails_term"])
        print(f"  {s}: {k} of {sum(1 for x in doc['records'] if x['set'] == s)} records")


if __name__ == "__main__":
    main()
