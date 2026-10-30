"""Verifier: POL-6's proposed target (per-seed SD of the CoM lateral offset < 1 mm, or no episode > 5 mm) on today's
policies, M1 seeds (and M1 + held-out 30 seeds).  python -I -B target_check.py recompute.json"""
import json, sys
import numpy as np
D = json.load(open(sys.argv[1]))["mine"]
M1 = [str(s) for s in range(3042, 3062)]
H = [str(s) for s in range(3062, 3072)]
pols = [w for w in D["run"] if w != "statue"]
for label, seeds in (("M1 20 seeds", None), ("M1 + held-out 30 seeds", True)):
    under1, no5, both = [], [], []
    for w in pols:
        v = [D["run"][w][s]["5.0"]["com_lat_mm"] for s in M1] + ([D["run_held"][w][s]["5.0"]["com_lat_mm"] for s in H] if seeds else [])
        sd = float(np.std(v, ddof=1)); mx = float(np.max(np.abs(v)))
        if sd < 1.0: under1.append(f"{w} {sd:.2f}")
        if mx <= 5.0: no5.append(w)
        if sd < 1.0 and mx <= 5.0: both.append(w)
    print(f"{label}: policies with SD < 1 mm: {len(under1)}/26 {under1}")
    print(f"   policies with no episode over 5 mm: {len(no5)}/26 (s42 {sum(w.startswith('s42') for w in no5)}/12, s80 {sum(w.startswith('s80') for w in no5)}/12, starts {sum(w.startswith('start') for w in no5)}/2)")
    print(f"   both: {len(both)}/26")
sd_st = float(np.std([D["run"]["statue"][s]["5.0"]["com_lat_mm"] for s in M1], ddof=1))
print(f"statue SD {sd_st:.3f} mm")
