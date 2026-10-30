"""Fixer's independent Monte Carlo check of design findings D1 (size with unequal arm SDs) and D2 (two-state null
given the gate).  Rise units (B_S = 0).  REDUCED = filled p <= 3/70 with p = share of the 70 4-subsets whose sum is
>= the control sum; informative = median of the 4 control values >= 0.20.  Seeded; no file read."""
import itertools
import sys

import numpy as np

rng = np.random.default_rng(20261030)
N = 200_000
SUBS = np.array(list(itertools.combinations(range(8), 4)))       # 70 x 4


def reduced_and_gate(x):
    obs = x[:, :4].sum(1)
    sums = x[:, SUBS].sum(2)                                      # N x 70
    count = (sums >= obs[:, None] - 1e-12).sum(1)
    red = count <= 3
    ctrl = np.sort(x[:, :4], 1)
    med = (ctrl[:, 1] + ctrl[:, 2]) / 2
    return red, med


print("D1: equal means, unequal SDs (P(REDUCED) overall; given median >= 0.20)")
for m, sc, sb in ((0.579, 0.137, 0.137), (0.579, 0.137, 0.03), (0.579, 0.2, 0.05), (0.579, 0.34, 0.05),
                  (0.579, 0.05, 0.2), (0.29, 0.34, 0.05), (0.29, 0.137, 0.03)):
    x = np.concatenate([rng.normal(m, sc, (N, 4)), rng.normal(m, sb, (N, 4))], 1)
    red, med = reduced_and_gate(x)
    inf = med >= 0.20
    print(f"  m {m:.3f} sc {sc:<5} sb {sb:<5}: P(REDUCED) {red.mean():.4f}  P(REDUCED | informative) "
          f"{red[inf].mean():.4f}  P(not informative) {1 - inf.mean():.3f}")

print("D2: two-state null (each continuation climbs N(0.579, 0.10^2) with prob q, else stays N(0.04, 0.04^2))")
for q in (0.75, 0.6, 0.5, 0.4, 0.25):
    climb = rng.random((N, 8)) < q
    x = np.where(climb, rng.normal(0.579, 0.10, (N, 8)), rng.normal(0.04, 0.04, (N, 8)))
    red, med = reduced_and_gate(x)
    inf = med >= 0.20
    hi = med >= 0.29
    print(f"  q {q:<4}: P(REDUCED) {red.mean():.4f}  | informative {red[inf].mean():.4f}  | median rise >= 0.29 "
          f"{red[hi].mean():.4f}  P(not informative) {1 - inf.mean():.3f}")
print("numpy", np.__version__, "python", sys.version.split()[0])
