"""Exact stratified (conditional) test from the per-start tables (a/6, b/6), my own enumeration with Fractions."""
from fractions import Fraction as F
from math import comb
def dist(nA, nB, m):  # P(a = k | m positives in the stratum), hypergeometric
    return {k: F(comb(nA, k) * comb(nB, m - k), comb(nA + nB, m)) for k in range(max(0, m - nB), min(nA, m) + 1)}
tabs = {"s42": (6, 6, 6, 6), "s80": (1, 6, 1, 6)}
conv = {0: F(1)}
for a, nA, b, nB in tabs.values():
    d = dist(nA, nB, a + b); new = {}
    for x, p in conv.items():
        for k, q in d.items(): new[x + k] = new.get(x + k, 0) + p * q
    conv = new
x = sum(t[0] for t in tabs.values())
pu = sum(p for k, p in conv.items() if k >= x); pd = sum(p for k, p in conv.items() if k <= x)
print("X =", x, "P_UP =", pu, float(pu), "P_DOWN =", pd, float(pd), "line:", "1" if pu <= F(1, 40) else ("2" if pd <= F(1, 40) else "3"))
for s, (a, nA, b, nB) in tabs.items():
    d = dist(nA, nB, a + b)
    print(s, "Fisher A greater", sum(p for k, p in d.items() if k >= a), "B greater", sum(p for k, p in d.items() if k <= a),
          "notes:", ("LOW ARM-A RATE" if a < nA / 2 else "") + (" ARM A AT %d OF %d" % (a, a) if a == nA else ""))
