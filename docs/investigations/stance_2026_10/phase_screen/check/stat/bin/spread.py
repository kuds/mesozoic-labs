"""PS statistic check: the spread caveat (report only; the pre-registered p decides).

Input: this checker's own out/stat_check.json (primaries as exact rationals, re-derived from the probe files).
(1) arm sample SDs and their ratio; (2) under a normal model with equal means and this SD ratio, P(complete separation:
all 4 control above all 4 phase), by quadrature and by Monte Carlo, at the observed ratio and over a range, and the
ratio at which it reaches 0.05; (3) under the same model, P(filled p <= 3/70) = P(REDUCED) (Monte Carlo of the exact
70-split test); (4) a post hoc Welch t, its Welch-Satterthwaite df and one-sided p (two implementations);
(5) the lowest control minus the highest phase primary in each arm's SD; (6) a 90% interval for the SD ratio
(F(3, 3)), to show how loosely 4 + 4 determine it."""
import itertools
import json
import math
import sys
from fractions import Fraction

import numpy as np
from scipy import integrate, optimize, stats

mine = json.load(open(sys.argv[1]))["mine"]
prim = {k: Fraction(v) for k, v in mine["primaries"].items()}
ctrl = [prim[f"control_{s}"] for s in (2901, 2911, 2921, 2931)]
phase = [prim[f"critic_phase_{s}"] for s in (2901, 2911, 2921, 2931)]


def sd(xs):
    m = sum(xs, Fraction(0)) / len(xs)
    return math.sqrt(sum(((x - m) ** 2 for x in xs), Fraction(0)) / (len(xs) - 1))


sc, sp = sd(ctrl), sd(phase)
mc, mp = float(sum(ctrl, Fraction(0)) / 4), float(sum(phase, Fraction(0)) / 4)
ratio = sc / sp
print(f"arm sample SDs (n-1): control {sc:.6f}, critic_phase {sp:.6f}; ratio control/phase {ratio:.4f}")
print(f"arm means: control {mc:+.6f}, critic_phase {mp:+.6f}; difference {mc - mp:+.6f}")


def p_sep_quad(r):
    # phase ~ N(0, 1), control ~ N(0, r^2), 4 each: P(min control > max phase)
    f = lambda z: 4.0 * stats.norm.pdf(z) * stats.norm.cdf(z) ** 3 * stats.norm.sf(z / r) ** 4
    return integrate.quad(f, -12, 12, epsabs=1e-13, epsrel=1e-11, limit=400)[0]


def p_sep_grid(r, n=400001):
    # independent check: trapezoid on a fine grid with math.erf
    z = np.linspace(-12, 12, n)
    phi = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + np.array([math.erf(x / math.sqrt(2)) for x in z]))
    sfr = 0.5 * (1 - np.array([math.erf(x / r / math.sqrt(2)) for x in z]))
    return float(np.trapezoid(4 * phi * Phi ** 3 * sfr ** 4, z))


rng = np.random.default_rng(20261031)


def p_sep_mc(r, n=4_000_000):
    a = rng.standard_normal((n, 4))
    b = rng.standard_normal((n, 4)) * r
    return float(np.mean(b.min(axis=1) > a.max(axis=1)))


print("\nP(complete separation, all 4 control above all 4 phase | equal means, normal, this SD ratio):")
pq = p_sep_quad(ratio)
pg = p_sep_grid(ratio)
pm = p_sep_mc(ratio)
print(f"  at the observed ratio {ratio:.3f}: quadrature {pq:.5f}; grid trapezoid {pg:.5f}; Monte Carlo (4e6) {pm:.5f}")
print(f"  at ratio 1 (equal SDs): quadrature {p_sep_quad(1.0):.5f} (exact 1/70 = {1/70:.5f})")
for r in (2, 3, 4, 4.6, 5, 6, 7, 8, 10, 15, 20):
    print(f"  ratio {r:>4}: {p_sep_quad(r):.5f}")
r05 = optimize.brentq(lambda r: p_sep_quad(r) - 0.05, 5.0, 40.0, xtol=1e-6)
print(f"  P reaches 0.05 at an SD ratio of {r05:.2f}")

# 90% interval for the SD ratio: (s_c/s_p)^2 / F ~ F(3, 3)
f95 = stats.f.ppf(0.95, 3, 3)
print(f"\n90% interval for the true SD ratio given 4 + 4 (F(3,3) 0.95 quantile {f95:.3f}): "
      f"{ratio / math.sqrt(f95):.2f} to {ratio * math.sqrt(f95):.2f}; P(separation) at the ends "
      f"{p_sep_quad(ratio / math.sqrt(f95)):.4f} / {p_sep_quad(ratio * math.sqrt(f95)):.4f}")

# P(REDUCED) = P(filled p <= 3/70) under equal means at this ratio (Monte Carlo of the exact 70-split test)
subs = np.array([[1 if i in s else 0 for i in range(8)] for s in itertools.combinations(range(8), 4)], dtype=float)
n = 400_000
a = rng.standard_normal((n, 4)) * ratio      # control
b = rng.standard_normal((n, 4))              # phase
x = np.concatenate([a, b], axis=1)
sums = x @ subs.T                            # (n, 70) sums of the 4 labelled "control"
obs = a.sum(axis=1)
cnt = (sums >= obs[:, None] - 1e-12).sum(axis=1)
for k in (1, 2, 3):
    print(f"P(p <= {k}/70 | equal means, ratio {ratio:.2f}) Monte Carlo {np.mean(cnt <= k):.4f} (n {n:,})")
a = rng.standard_normal((n, 4))
b = rng.standard_normal((n, 4))
x = np.concatenate([a, b], axis=1)
cnt1 = ((x @ subs.T) >= a.sum(axis=1)[:, None] - 1e-12).sum(axis=1)
print(f"  check, equal SDs: P(p <= 3/70) {np.mean(cnt1 <= 3):.4f} (exact 3/70 = {3/70:.4f})")

# Welch t (post hoc, report only)
v1, v2 = sc ** 2 / 4, sp ** 2 / 4
t = (mc - mp) / math.sqrt(v1 + v2)
df = (v1 + v2) ** 2 / (v1 ** 2 / 3 + v2 ** 2 / 3)
p1 = stats.t.sf(t, df)
tt = stats.ttest_ind([float(v) for v in ctrl], [float(v) for v in phase], equal_var=False, alternative="greater")
print(f"\nWelch t (post hoc, report only): t {t:.4f}, df {df:.3f}, one-sided p {p1:.3g}; scipy ttest_ind(Welch, "
      f"greater): t {tt.statistic:.4f} p {tt.pvalue:.3g}")
gap = float(min(ctrl) - max(phase))
print(f"lowest control {float(min(ctrl)):+.5f} minus highest phase {float(max(phase)):+.5f} = {gap:.5f} = "
      f"{gap / sp:.1f} phase SDs = {gap / sc:.1f} control SDs")
json.dump({"sd_control": sc, "sd_phase": sp, "ratio": ratio, "p_sep_quad": pq, "p_sep_grid": pg, "p_sep_mc": pm,
           "ratio_p05": r05, "welch_t": t, "welch_df": df, "welch_p_one_sided": p1, "gap": gap,
           "p_reduced_mc": float(np.mean(cnt <= 3)), "ratio_ci90": [ratio / math.sqrt(f95), ratio * math.sqrt(f95)]},
          open(sys.argv[2], "w"), indent=1)
