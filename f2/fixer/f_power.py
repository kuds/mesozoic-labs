# Exact one-sided Fisher power (arm B lower), my own re-derivation for the fixer.
from math import comb
def fisher_p(ka, kb, n):  # P(arm A count >= ka | total ka+kb, n each)
    k = ka + kb
    tot = comb(2*n, k)
    return sum(comb(n, x) * comb(n, k - x) for x in range(ka, min(n, k) + 1)) / tot
def power(n, pa, pb, alpha=0.05):
    s = 0.0
    for ka in range(n + 1):
        for kb in range(n + 1):
            if fisher_p(ka, kb, n) <= alpha + 1e-12:
                s += comb(n, ka) * pa**ka * (1-pa)**(n-ka) * comb(n, kb) * pb**kb * (1-pb)**(n-kb)
    return s
print("p at outcomes: 4/6 vs 0/6 %.4f; 5/8 vs 0/8 %.4f; 4/8 vs 0/8 %.4f" % (fisher_p(4,0,6), fisher_p(5,0,8), fisher_p(4,0,8)))
for n in (6, 8):
    for pa, pb, lab in ((2/3, 0, "2/3 vs 0 (complete cure)"), (2/3, 1/3, "2/3 vs 1/3 (halved)"), (0.4, 0, "0.4 vs 0"), (0.4, 0.2, "0.4 vs 0.2")):
        print("n=%d per arm, %s: power %.3f" % (n, lab, power(n, pa, pb)))
print("Option 1: 4/4 vs 0/4 %.4f; 3/4 vs 0/4 %.4f" % (fisher_p(4,0,4), fisher_p(3,0,4)))
for t in (1048576, 3145728, 6000000, 6291456):
    print("ent_coef at %d: %.6f" % (t, 0.005*(1 - t/7e6)))
# TABLE E upper bound for 2 of 3: exact one-sided binomial vs uniform-prior Beta(3,2)
print("exact one-sided upper bound 2 of 3: %.3f" % (0.95**(1/3)))
lo, hi = 0.0, 1.0
for _ in range(60):
    m = (lo+hi)/2
    if 4*m**3 - 3*m**4 < 0.95: lo = m
    else: hi = m
print("Beta(3,2) 95%% quantile: %.3f" % lo)
