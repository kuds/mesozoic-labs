"""Verifier: POL-5's power numbers re-derived from recompute.json (my own reduction).  python -I -B power_check.py recompute.json"""
import json, math, sys
import numpy as np
D = json.load(open(sys.argv[1]))["mine"]["run"]
M1 = [str(s) for s in range(3042, 3062)]
def p_sign(a, b):
    n = a + b; k = min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
def flag(v, fl): return p_sign(int((v > 0).sum()), int((v < 0).sum())) <= 0.005 and abs(np.median(v)) >= fl
pols = [w for w in D if w != "statue"]
for key, fl, bs in (("com_lat_mm", 1.0, (0.5, 1.0, 1.5, 2.0, 3.0)), ("roll_deg", 0.10, (0.05, 0.10, 0.15, 0.2))):
    cells = []
    for b in bs:
        ag = al = 0
        for w in pols:
            v = np.array([D[w][s]["5.0"][key] for s in M1]); sgn = -np.sign(np.median(v)) or 1.0
            ag += flag(v + sgn * b, fl); al += flag(v - sgn * b, fl)
        cells.append(f"b {b:g}: {ag / len(pols):.2f}/{al / len(pols):.2f}")
    print(key, "; ".join(cells))
