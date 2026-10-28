"""Verifier's power check (report only): exact one-sided Fisher p and its power for two-arm designs."""
from math import comb

def fisher_one_sided(a, n1, b, n2):
    """P(X >= a) for X = arm-1 successes given total a+b, hypergeometric (arm 1 higher)."""
    K = a + b; N = n1 + n2
    return sum(comb(K, x) * comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / comb(N, n1)

def binom(k, n, p):
    return comb(n, k) * p**k * (1 - p) ** (n - k)

def power(n, p1, p2, alpha=0.05):
    tot = 0.0
    for a in range(n + 1):
        for b in range(n + 1):
            if fisher_one_sided(a, n, b, n) <= alpha:
                tot += binom(a, n, p1) * binom(b, n, p2)
    return tot

print("p at outcomes (RO's figures): 4/6 vs 0/6 %.4f; 5/8 vs 0/8 %.4f; 4/8 vs 0/8 %.4f" % (
    fisher_one_sided(4, 6, 0, 6), fisher_one_sided(5, 8, 0, 8), fisher_one_sided(4, 8, 0, 8)))
for n in (6, 8):
    for p1, p2, lab in ((2/3, 0.0, "2/3 vs 0 (complete removal)"), (2/3, 1/3, "2/3 vs 1/3 (halved)"),
                        (0.4, 0.0, "r9 0.4 vs 0"), (0.4, 0.2, "0.4 vs 0.2")):
        print("n=%d per arm, true %s: power at one-sided alpha 0.05 = %.3f" % (n, lab, power(n, p1, p2)))
# Option 1's figures
print("Option1: 4/4 vs 0/4 %.4f; 3/4 vs 0/4 %.4f" % (fisher_one_sided(4, 4, 0, 4), fisher_one_sided(3, 4, 0, 4)))
print("Option5: 2/5 vs 2/5 (r10 lower) %.4f; 0/5 vs 2/5 %.4f; 2/3 vs 2/5 %.4f" % (
    fisher_one_sided(2, 5, 2, 5), fisher_one_sided(2, 5, 0, 5), fisher_one_sided(2, 5, 2, 3)))
# bonus value
for t in (3145728, 6000000, 6291456):
    print("ent_coef at %d: %.6f" % (t, 0.005 * max(0.0, 1 - t / 7e6)))
