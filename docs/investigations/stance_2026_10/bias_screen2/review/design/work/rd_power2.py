"""Design reviewer, follow-up to rd_power.py P2: under equal means and unequal arm SDs, the size of the pre-registered
test (permutation of the difference of means) against a studentized (Welch t) permutation test, and their power at a
half reduction; no gate (the gate passes almost surely at m = R).  Also the sign of the effect: P(the control sum is
among the 3 LOWEST of 70) to show the inflation is two-sided (a spread effect, not a location effect).
Usage: PY -I -B rd_power2.py > rd_power2.out
"""
import itertools

import numpy as np

R = 0.5787834012880921
IDX = np.array(list(itertools.combinations(range(8), 4)))
OBS = 0
assert tuple(IDX[OBS]) == (0, 1, 2, 3)
MASK = np.zeros((70, 8), dtype=bool)
for i, c in enumerate(IDX):
    MASK[i, c] = True


def stats(x):
    """x (n, 8): first 4 control.  Returns (diff-of-means count >= obs, count <= obs, welch count >= obs)."""
    sums = x[:, IDX].sum(axis=2)                                      # (n, 70) control-labelled sums
    tot = x.sum(axis=1, keepdims=True)
    d = sums / 4 - (tot - sums) / 4
    ge = (d >= d[:, :1]).sum(axis=1)
    le = (d <= d[:, :1]).sum(axis=1)
    a = x[:, IDX]                                                     # (n, 70, 4) labelled control
    other = np.stack([x[:, ~MASK[i]] for i in range(70)], axis=1)     # (n, 70, 4)
    va, vb = a.var(axis=2, ddof=1), other.var(axis=2, ddof=1)
    t = (a.mean(axis=2) - other.mean(axis=2)) / np.sqrt(va / 4 + vb / 4 + 1e-300)
    tge = (t >= t[:, :1]).sum(axis=1)
    return ge, le, tge


def run(mc, sc, mb, sb, n=200_000, seed=0, chunk=20_000):
    rng = np.random.default_rng(seed)
    acc = np.zeros(3)
    done = 0
    while done < n:
        k = min(chunk, n - done)
        x = np.concatenate([rng.normal(mc, sc, (k, 4)), rng.normal(mb, sb, (k, 4))], axis=1)
        ge, le, tge = stats(x)
        acc += [(ge <= 3).sum(), (le <= 3).sum(), (tge <= 3).sum()]
        done += k
    return acc / n


print("cols: P(REDUCED) diff-of-means (pre-registered) | P(control sum among the 3 LOWEST) | studentized permutation")
for sc, sb in ((0.137, 0.137), (0.137, 0.03), (0.20, 0.05), (0.34, 0.05), (0.05, 0.20)):
    r0 = run(R, sc, R, sb, seed=int(sc * 1000) + int(sb * 10000))
    r5 = run(R, sc, R - 0.5 * R, sb, seed=7 + int(sc * 1000) + int(sb * 10000))
    print(f"  sc {sc:<5} sb {sb:<5} H0: {r0[0]:.4f} | {r0[1]:.4f} | {r0[2]:.4f}    half reduction: "
          f"{r5[0]:.3f} | - | {r5[2]:.3f}")
print("END")
