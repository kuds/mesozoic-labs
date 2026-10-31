"""Recompute B_S and the parent's per-actuator means at S from the F2 probe (read only)."""
import sys
from fractions import Fraction
import numpy as np
p = sys.argv[1]
with np.load(p, allow_pickle=False) as z:
    keys = list(z.keys())
    ut = z["update_timesteps"]
    mu = z["rollout_mu_mean"]
print("keys", keys)
print("rows", len(ut), "mu shape", mu.shape, mu.dtype)
steps = list(range(2_039_808, 2_097_152 + 1, 8192))
idx = [int(np.flatnonzero(ut == s)[0]) for s in steps]
print("rows idx", idx)
vals = [Fraction(float(mu[i, 14])) for i in idx]
b = sum(vals) / 8
print("B_S exact", b, float(b), "den", b.denominator)
print("135930855/2^30", Fraction(135930855, 2**30) == b)
t2 = sum(Fraction(float(mu[i, 13])) for i in idx) / 8
print("tail2 at S", float(t2))
# primary rows for PS window
S, E = 2_097_152, 3_538_944
prim = [S + 8192 * k for k in range(168, 176)]
print("primary steps", prim, "k 168-175 ->", prim[0], prim[-1], "E-8192", E - 8192)
