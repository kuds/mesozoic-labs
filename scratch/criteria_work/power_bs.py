"""Power of the bias screen's pre-registered test (4 control vs 4 critic_blind_tail3, the exact one-sided permutation
test on the difference of arm means, REDUCED when p <= 0.05, i.e. at most 3 of the 70 relabelings at or above the
observed): Monte Carlo under a normal model with between-continuation SD sigma and a true reduction delta.  Also the
exact size under the null (continuous, exchangeable values) and the informativeness rule's chance of failing.
Deterministic (seeded).  [inferred: the normal model and sigma are assumptions; sigma's range is from ecs_spread.out]"""
import itertools
import random
import statistics as st

N = 4
SUBSETS = list(itertools.combinations(range(2 * N), N))


def p_value(control, blind):
    vals = list(control) + list(blind)
    obs = sum(control)
    return sum(1 for s in SUBSETS if sum(vals[i] for i in s) >= obs - 1e-15) / len(SUBSETS)


def power(delta, sigma, sims, rng):
    hit = 0
    for _ in range(sims):
        c = [rng.gauss(0.0, sigma) for _ in range(N)]
        b = [rng.gauss(-delta, sigma) for _ in range(N)]
        hit += p_value(c, b) <= 0.05
    return hit / sims


rng = random.Random(20261030)
print("exact size under the null (continuous values): P(p <= 0.05) = 3/70 =", round(3 / 70, 4))
print("minimum attainable p = 1/70 =", round(1 / 70, 4), "; attainable p values <= 0.05: 1/70, 2/70, 3/70")
sims = 4000
print(f"P(REDUCED) by Monte Carlo ({sims} draws per cell; MC SE <= 0.008):")
print("  delta \\ sigma " + "".join(f"{s:>8}" for s in (0.04, 0.06, 0.08, 0.10)))
for delta in (0.05, 0.10, 0.15, 0.20, 0.231):
    row = [power(delta, s, sims, rng) for s in (0.04, 0.06, 0.08, 0.10)]
    print(f"  {delta:>12} " + "".join(f"{x:>8.3f}" for x in row))
print("delta 0.231 = the parent's whole rise over the window (the blind arm held at its value at S)")
# informativeness: control median rise >= 0.10 fails with probability ...
print("P(NOT INFORMATIVE) = P(median of 4 control rises < 0.10), true mean rise m, SD sigma:")
for m in (0.10, 0.15, 0.20, 0.231):
    row = []
    for s in (0.04, 0.06, 0.08, 0.10):
        k = sum(st.median([rng.gauss(m, s) for _ in range(N)]) < 0.10 for _ in range(sims))
        row.append(k / sims)
    print(f"  m {m:>5}: " + "".join(f"{x:>8.3f}" for x in row))
