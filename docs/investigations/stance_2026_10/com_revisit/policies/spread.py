"""Per-episode lean spread by group: each policy's SD over the seeds of the window-mean CoM lateral offset and roll
(window from 5 s), the groups compared (exact Mann-Whitney), and how a policy's lean follows the spawn: Spearman of
the policy's per-seed value against the statue's at the same spawn, and of policies against each other per seed.

python -B spread.py RUN_DIR  (episodes.jsonl only; no simulation)
"""
import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

W = "5.0"
rows = defaultdict(dict)
for line in open(f"{sys.argv[1]}/episodes.jsonl"):
    r = json.loads(line)
    rows[r["who"]][r["seed"]] = r["metrics"][W]
seeds = sorted(rows["statue"])
for key, sc, u in (("com_lateral_mean_m", 1000, "mm"), ("root_roll_mean_deg", 1, "deg"), ("left_load_share_mean", 1, "")):
    sd = {w: float(np.std([rows[w][s][key] for s in seeds], ddof=1)) for w in rows}
    a = [sd[w] for w in rows if w.startswith("s42")]
    b = [sd[w] for w in rows if w.startswith("s80")]
    p = mannwhitneyu(a, b, method="exact").pvalue
    print(f"{key}: SD over seeds, statue {sd['statue'] * sc:.4f} {u}; start_s42 {sd['start_s42'] * sc:.4f}; start_s80 {sd['start_s80'] * sc:.4f};"
          f" s42 ends median {np.median(a) * sc:.4f} [{min(a) * sc:.4f}, {max(a) * sc:.4f}]; s80 ends median {np.median(b) * sc:.4f}"
          f" [{min(b) * sc:.4f}, {max(b) * sc:.4f}]; s42 vs s80 p {p:.4f}")
    st = [rows["statue"][s][key] for s in seeds]
    for g in ("start_s42", "start_s80", "s42", "s80"):
        ws = [w for w in rows if (w == g if g.startswith("start") else w.startswith(g + "_"))]
        rhos = [spearmanr(st, [rows[w][s][key] for s in seeds])[0] for w in ws]
        # the group's per-seed median against the statue
        med = [np.median([rows[w][s][key] for w in ws]) for s in seeds]
        rho_m, p_m = spearmanr(st, med)
        # agreement between the group's policies: mean pairwise Spearman over seeds
        pair = [spearmanr([rows[x][s][key] for s in seeds], [rows[y][s][key] for s in seeds])[0] for i, x in enumerate(ws) for y in ws[i + 1:]]
        print(f"   {g:10s} vs statue at the same spawn: per-policy rho median {np.median(rhos):+.2f}; group per-seed median rho"
              f" {rho_m:+.2f} p {p_m:.3f}; policies with each other: mean pairwise rho {np.mean(pair) if pair else float('nan'):+.2f}")
