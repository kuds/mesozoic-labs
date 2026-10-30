"""Fixer's own check of review findings R1 and R2 (normal model; no training; no tool imported).

    cd /tmp && PY -I -B <BS>/review/fixer/fx_sim.py

Independent of review/rv_sim.py: another seed, the test written as "rank of the observed control sum among the 70"
(strict ">" count + 1, ties have probability 0 here), and the informativeness gate on the control median.
(R1) P(REDUCED) under no effect, overall and given the gate passed, by the control arm's true rise.
(R2) P(REDUCED) at a reduction of half the parent's rise (0.10 and 0.1156) with and without a pair effect.
"""
import itertools

import numpy as np

rng = np.random.default_rng(5150)
IDX = np.array(list(itertools.combinations(range(8), 4)))          # 70 x 4; row 0 = (0,1,2,3) = the observed split
B_S = -0.03157372563146055
N = 60000


def reduced(c, b):
    pool = np.concatenate([c, b], axis=1)
    sums = pool[:, IDX].sum(axis=2)                                 # (n, 70)
    obs = sums[:, :1]
    count = (sums >= obs).sum(axis=1)                               # observed split included
    gate = (np.median(c, axis=1) - B_S) >= 0.10
    return gate & (count <= 3), gate


print("(R1) no effect: all 8 iid N(B_S + rise, sd)")
for rise in (0.10, 0.12, 0.15, 0.20, 0.231):
    for sd in (0.04, 0.06, 0.10):
        c = B_S + rise + rng.normal(0, sd, (N, 4))
        b = B_S + rise + rng.normal(0, sd, (N, 4))
        red, gate = reduced(c, b)
        print(f"  rise {rise:.3f} sd {sd:.2f}: P(gate) {gate.mean():.3f}  P(REDUCED) {red.mean():.4f}  "
              f"P(REDUCED | gate) {red.sum() / max(gate.sum(), 1):.4f}")

print("(R2) power at a reduction delta, control rise 0.231; value = pair effect (sd tau) + own (sd)")
for delta in (0.10, 0.1156):
    for sd in (0.04, 0.06, 0.10):
        for tau in (0.0, 0.06):
            pair = rng.normal(0, tau, (N, 4))
            c = B_S + 0.231 + pair + rng.normal(0, sd, (N, 4))
            b = B_S + 0.231 - delta + pair + rng.normal(0, sd, (N, 4))
            red, _ = reduced(c, b)
            print(f"  delta {delta:.4f} sd {sd:.2f} tau {tau:.2f}: P(REDUCED) {red.mean():.3f}")
