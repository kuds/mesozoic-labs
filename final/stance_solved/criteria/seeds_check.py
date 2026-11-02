"""Read-only: list the seed values in each PHASE final's M3 panel CSV and the M3 summary's seed fields."""
import csv, json, sys
base = "/home/user/work/pc_study/round/panels/stance7_r10LTSBpc_%s/final/m3/"
for s in ("s60", "s42", "s80"):
    p = base % s + "stance_heading_panel_selected.csv"
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = [c for c in rows[0].keys() if "seed" in c.lower()]
    out = {}
    for c in cols:
        vals = sorted({r[c] for r in rows})
        out[c] = (len(vals), vals[:2], vals[-2:])
    heads = sorted({r.get("heading_deg", r.get("heading", "?")) for r in rows})
    print(s, "rows", len(rows), "seed cols", out, "headings", heads)
