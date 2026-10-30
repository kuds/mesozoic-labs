"""Re-derive the scan's intervals and Fisher p-values (no simulation)."""
from scipy.stats import beta, fisher_exact, binomtest
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return 100 * lo, 100 * hi
for k, n in ((0, 280), (0, 200), (0, 480), (0, 960), (10, 280), (16, 480), (6, 200), (46, 200), (68, 200), (0, 202)):
    lo, hi = cp(k, n); print(f"CP {k}/{n}: {lo:.2f}-{hi:.2f} %")
print("binomtest check 0/280:", binomtest(0, 280).proportion_ci(0.95, 'exact'))
for (a, n1), (b, n2), label in (((0, 280), (10, 280), "model 0.02 vs robot same seeds 0.02"),
                               ((0, 480), (16, 480), "model pooled (0.02 + own train noise) vs robot 0.02 (480)"),
                               ((0, 280), (16, 480), "matched noise: model 0.02 (280) vs robot 0.02 (480)"),
                               ((0, 200), (0, 202), "compy 0.01 vs robot 0.01 (verifier M2)"),
                               ((0, 480), (16, 682), "model pooled vs robot all measured (0.02 480 + 0.01 202)")):
    p = fisher_exact([[a, n1 - a], [b, n2 - b]])[1]
    print(f"Fisher {label}: {a}/{n1} vs {b}/{n2}: p = {p:.3g}")
# one-sided: what incidence would make 0/280 surprising
print("P(0 of 280 | p=3.6%) =", (1 - 10/280) ** 280, "; P(0/280 | 1%) =", 0.99 ** 280)
