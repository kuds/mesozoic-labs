"""Per policy: the tail_3 window command mean against the sagittal posture (window from 5 s), Spearman rank correlation.

python -B rail_posture.py RUN_DIR  (reads episodes.jsonl only; no simulation)
"""
import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

rows = defaultdict(list)
for line in open(f"{sys.argv[1]}/episodes.jsonl"):
    r = json.loads(line)
    rows[r["who"]].append(r)
keys = ("root_pitch_mean_deg", "com_fore_aft_mean_m", "com_height_mean_m", "com_margin_min_m", "com_margin_fore_aft_min_m")
tab = {}
for who, rs in rows.items():
    if who == "statue":
        continue
    tab[who] = {"tail3": float(np.mean([r["cmd_mean_from_2s"][14] for r in rs]))}
    for k in keys:
        tab[who][k] = float(np.median([r["metrics"]["5.0"][k] for r in rs]))
print(f"{'who':12s} tail_3   " + " ".join(f"{k[:18]:>19s}" for k in keys))
for who, d in tab.items():
    print(f"{who:12s} {d['tail3']:+.3f}  " + " ".join(f"{d[k] * (1000 if k.endswith('_m') else 1):+19.3f}" for k in keys))
for label, ws in (("all 26", list(tab)), ("s42 start + ends (13)", [w for w in tab if "s42" in w]), ("s80 start + ends (13)", [w for w in tab if "s80" in w])):
    x = [tab[w]["tail3"] for w in ws]
    out = []
    for k in keys:
        rho, p = spearmanr(x, [tab[w][k] for w in ws])
        out.append(f"{k}: rho {rho:+.2f} p {p:.3f}")
    print(label, "; ".join(out))
