"""Report-only arithmetic on counts already recorded (no data files read).

Sources of the counts are named in NOTE.md. Exact binomial / beta arithmetic, standard library only.
"""
from math import comb


def cp_interval(k, n, alpha=0.05):
    """Two-sided Clopper-Pearson interval by bisection on the binomial tail."""
    def tail_ge(p):  # P(X >= k)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):  # P(X <= k)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def bisect(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else bisect(tail_ge, alpha / 2, True)
    upper = 1.0 if k == n else bisect(tail_le, alpha / 2, False)
    return lower, upper


print("A. PHASE: 3 of 3 runs met line 4 (no rail/park, gate PASS, no saturating habit elsewhere)")
print(f"   one-sided 95% lower bound on the true success rate: 0.05**(1/3) = {0.05 ** (1 / 3):.3f}")
lo, hi = cp_interval(3, 3)
print(f"   two-sided 95% Clopper-Pearson interval: [{lo:.3f}, {hi:.3f}]")
print(f"   rule of succession (uniform prior), chance the next seed also succeeds: {4 / 5:.3f}")
print(f"   Jeffreys prior posterior mean: {3.5 / 4:.3f}")
print(f"   chance that 3 more new seeds all succeed if the true rate were 0.80: {0.8 ** 3:.3f}; 0.90: {0.9 ** 3:.3f}")

print("B. Plug-in chance of 0 habits in 3 PHASE runs if the phase input changed nothing, at a base rate q:")
for label, q in [("rail only, OFF 1 of 3", 1 / 3), ("rail or park, OFF 2 of 3", 2 / 3),
                 ("any habit incl. OFF s80 neck, 3 of 3", 3 / 3), ("LTSB r9 rails 2 of 5", 2 / 5),
                 ("r9 L+T family rails 4 of 10", 4 / 10), ("pooled rails r9 4 of 10 + r10 1 of 3 = 5 of 13", 5 / 13),
                 ("pooled rail-or-park 6 of 13 (r9 parks not classified)", 6 / 13)]:
    print(f"   {label:55s} q = {q:.3f}: (1-q)^3 = {(1 - q) ** 3:.3f}")
lo, hi = cp_interval(2, 3)
print(f"   OFF rail-or-park 2 of 3, 95% CP interval for q: [{lo:.3f}, {hi:.3f}] -> (1-q)^3 from {(1 - lo) ** 3:.3f} "
      f"to {(1 - hi) ** 3:.2e}")
lo, hi = cp_interval(5, 13)
print(f"   pooled rails 5 of 13, 95% CP interval for q: [{lo:.3f}, {hi:.3f}] -> (1-q)^3 from {(1 - lo) ** 3:.3f} "
      f"to {(1 - hi) ** 3:.3f}")

print("C. Fisher one-sided, 3 vs 3, as a check of SCORE line 363")
print(f"   0 of 3 vs 2 of 3: C(3,2)/C(6,2) = {comb(3, 2) / comb(6, 2):.3f}; 0 of 3 vs 3 of 3: "
      f"1/C(6,3) = {1 / comb(6, 3):.3f}")
print("D. Smallest exact p, 3 + 3 unpaired permutation: 1/C(6,3) =", f"{1 / comb(6, 3):.3f}",
      "; paired sign-flip, 3 pairs: 1/2**3 =", 1 / 2**3)
