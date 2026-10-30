"""Verifier: POL-4, each lineage's start-to-end shift (12 ends against their own start), from recompute.json.
python -I -B lineage.py recompute.json"""
import json, math, sys
import numpy as np
from scipy.stats import spearmanr
D = json.load(open(sys.argv[1]))["mine"]["run"]
M1 = [str(s) for s in range(3042, 3062)]
def med(w, k): return float(np.median([D[w][s]["5.0"][k] for s in M1]))
def tail3(w): return float(np.mean([D[w][s]["cmd_from_2s"][14] for s in M1]))
def p_sign(a, b):
    n = a + b; k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
for lin in ("s42", "s80"):
    ends = [w for w in D if w.startswith(lin + "_")]
    st = f"start_{lin}"
    print(f"{lin}: start tail_3 {tail3(st):+.3f}; ends tail_3 {min(tail3(w) for w in ends):+.3f}..{max(tail3(w) for w in ends):+.3f}")
    for k in ("pitch_deg", "com_fa_mm", "com_lat_mm", "roll_deg"):
        e = np.array([med(w, k) for w in ends]); s0 = med(st, k)
        up, dn = int((e > s0).sum()), int((e < s0).sum())
        print(f"   {k:10s} start {s0:+8.3f}  ends median {np.median(e):+8.3f}  shift {np.median(e) - s0:+7.3f}  ends above/below start {up}/{dn} (sign p {p_sign(up, dn):.4f})")
    x = [tail3(w) for w in ends + [st]]
    for k in ("pitch_deg", "com_fa_mm"):
        r, p = spearmanr(x, [med(w, k) for w in ends + [st]])
        print(f"   tail_3 vs {k} within lineage (13): rho {r:+.2f} p {p:.3f}")
for k in ("pitch_deg", "com_fa_mm"):
    print(f"gap s42 - s80: starts {med('start_s42', k) - med('start_s80', k):+.3f}; ends (median of medians) "
          f"{np.median([med(w, k) for w in D if w.startswith('s42_')]) - np.median([med(w, k) for w in D if w.startswith('s80_')]):+.3f}")
