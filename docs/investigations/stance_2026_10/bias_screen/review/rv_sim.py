"""Reviewer's Monte Carlo checks of the design (normal model; no training).

    cd /tmp && PY -I -B <BS>/review/rv_sim.py

(1) power cells against WORK/power_bs.out; (2) size under H0 overall and conditional on the informativeness gate;
(3) size under H0 when paired seeds correlate (pair effect tau): the unpaired test's real size.
"""
import itertools

import numpy as np

rng = np.random.default_rng(20261030)
SUBS = np.array([[1 if i in s else 0 for i in range(8)] for s in itertools.combinations(range(8), 4)])   # 70 x 8
B_S = -0.03157372563146055


def pvals(ctrl, blind):
    """counts (1..70) of the exact one-sided test for arrays (n, 4) and (n, 4); control is the first 4 of the pool."""
    pool = np.concatenate([ctrl, blind], axis=1)            # (n, 8)
    sums = pool @ SUBS.T                                    # (n, 70)
    obs = ctrl.sum(axis=1, keepdims=True)
    return (sums >= obs - 1e-12).sum(axis=1)


def reading(ctrl, blind):
    cnt = pvals(ctrl, blind)
    med = np.median(ctrl, axis=1)
    inf = (med - B_S) >= 0.10
    red = inf & (cnt <= 3)
    return red, inf


N = 40000
print("(1) POWER P(REDUCED), control rise 0.231 from B_S, blind lower by delta")
for delta in (0.05, 0.10, 0.231):
    row = []
    for sd in (0.04, 0.10):
        c = B_S + 0.231 + rng.normal(0, sd, (N, 4))
        b = B_S + 0.231 - delta + rng.normal(0, sd, (N, 4))
        red, _ = reading(c, b)
        row.append(f"sd {sd}: {red.mean():.3f}")
    print(f"   delta {delta}: " + "  ".join(row))

print("(2) SIZE UNDER H0 (all 8 iid), overall and given the informativeness gate")
for rise in (0.08, 0.10, 0.12, 0.15, 0.231):
    for sd in (0.04, 0.10):
        c = B_S + rise + rng.normal(0, sd, (N, 4))
        b = B_S + rise + rng.normal(0, sd, (N, 4))
        red, inf = reading(c, b)
        cond = red.sum() / max(inf.sum(), 1)
        print(f"   true rise {rise:.3f} sd {sd}: P(informative) {inf.mean():.3f}  P(REDUCED) {red.mean():.4f}  "
              f"P(REDUCED | informative) {cond:.4f}")

print("(3) SIZE UNDER H0 WITH PAIRED SEEDS: value = pair effect (sd tau) + own noise (sd 0.06)")
for tau in (0.0, 0.03, 0.06, 0.12):
    pair = rng.normal(0, tau, (N, 4))
    c = B_S + 0.231 + pair + rng.normal(0, 0.06, (N, 4))
    b = B_S + 0.231 + pair + rng.normal(0, 0.06, (N, 4))
    red, _ = reading(c, b)
    print(f"   tau {tau}: P(REDUCED) {red.mean():.4f} (nominal 3/70 = {3 / 70:.4f})")
print("(3b) POWER WITH PAIRED SEEDS (delta 0.10, own sd 0.06)")
for tau in (0.0, 0.06):
    pair = rng.normal(0, tau, (N, 4))
    c = B_S + 0.231 + pair + rng.normal(0, 0.06, (N, 4))
    b = B_S + 0.131 + pair + rng.normal(0, 0.06, (N, 4))
    red, _ = reading(c, b)
    print(f"   tau {tau}: P(REDUCED) {red.mean():.3f}")
