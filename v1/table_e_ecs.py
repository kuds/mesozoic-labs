"""ECS criteria v1, REFERENCE TABLE E: what each result of the one statistic prints, and its power and size.

    cd /tmp && S2/venv/bin/python -I -B S2/r10round/ecs/v1/table_e_ecs.py > S2/r10round/ecs/v1/table_e_ecs.out

Pure arithmetic, standard library only. No file is read; nothing is simulated or trained.

The statistic (criteria_ecs_v1.txt, THE ONE PRE-REGISTERED STATISTIC): per start s, a_s of nA_s arm-A continuations
and b_s of nB_s arm-B continuations are POSITIVE (an ANY-flagged evaluation in (S, E]). T = sum of a_s. Exact
stratified test (exact conditional test of a common odds ratio, "exact Cochran-Mantel-Haenszel"): conditional on each
start's total m_s = a_s + b_s, a_s is hypergeometric (nA_s + nB_s, m_s, nA_s) under the null, and T is their
convolution. P_UP = P(T >= t) (A higher, i.e. B lower), P_DOWN = P(T <= t). Two-sided at 0.025 per side:
B LOWER when P_UP <= 0.025; B HIGHER when P_DOWN <= 0.025; otherwise NO DIFFERENCE DETECTED. p-values are exact
rationals (fractions), so a result on the 0.025 line is decided exactly.
"""
from fractions import Fraction
from functools import lru_cache
from math import comb

SIDE = Fraction(1, 40)  # 0.025


@lru_cache(maxsize=None)
def stratum_null(nA, nB, m):
    tot = comb(nA + nB, m)
    return tuple((a, Fraction(comb(nA, a) * comb(nB, m - a), tot))
                 for a in range(max(0, m - nB), min(nA, m) + 1))


def null_T(strata):
    """strata: tuple of (nA, nB, m). Exact null distribution of T as {t: Fraction}."""
    d = {0: Fraction(1)}
    for nA, nB, m in strata:
        nd = {}
        for t, p in d.items():
            for a, q in stratum_null(nA, nB, m):
                nd[t + a] = nd.get(t + a, Fraction(0)) + p * q
        d = nd
    return d


@lru_cache(maxsize=None)
def pvals(cells):
    """cells: tuple of (a, nA, b, nB) per start. Returns (P_UP, P_DOWN) as Fractions."""
    strata = tuple((nA, nB, a + b) for a, nA, b, nB in cells)
    t = sum(a for a, _nA, _b, _nB in cells)
    d = null_T(strata)
    up = sum((p for v, p in d.items() if v >= t), Fraction(0))
    down = sum((p for v, p in d.items() if v <= t), Fraction(0))
    return up, down


def reading(cells):
    up, down = pvals(cells)
    if up <= SIDE:
        return "B LOWER", up, down
    if down <= SIDE:
        return "B HIGHER", up, down
    return "NO DIFFERENCE DETECTED", up, down


def binom(n, k, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


def power(k, rates):
    """P(B LOWER) and P(B HIGHER) with k per arm per start; rates = [(pA_s, pB_s)] per start (floats)."""
    lower = higher = 0.0

    def rec(i, cells, prob):
        nonlocal lower, higher
        if i == len(rates):
            r = reading(tuple(cells))[0]
            if r == "B LOWER":
                lower += prob
            elif r == "B HIGHER":
                higher += prob
            return
        pA, pB = rates[i]
        for a in range(k + 1):
            pa = binom(k, a, pA)
            if pa == 0:
                continue
            for b in range(k + 1):
                pb = binom(k, b, pB)
                if pb == 0:
                    continue
                rec(i + 1, cells + [(a, k, b, k)], prob * pa * pb)

    rec(0, [], 1.0)
    return lower, higher


def f(x):
    return "%.4f" % float(x)


def main():
    out = []
    p = out.append
    p("ECS REFERENCE TABLE E (criteria_ecs_v1.txt): exact stratified test, two-sided, 0.025 per side")
    p("")
    p("1. THE SAME TABLE AT BOTH STARTS, k = 6 PER ARM PER START (a = arm-A positives of 6, b = arm-B positives of 6)")
    p("   L = B LOWER, H = B HIGHER, . = NO DIFFERENCE DETECTED")
    p("        b=0 b=1 b=2 b=3 b=4 b=5 b=6")
    for a in range(7):
        row = []
        for b in range(7):
            r = reading(((a, 6, b, 6), (a, 6, b, 6)))[0]
            row.append({"B LOWER": "L", "B HIGHER": "H"}.get(r, "."))
        p("   a=%d   %s" % (a, "   ".join(row)))
    p("   p-values at selected tables (P_UP, P_DOWN):")
    for a, b in ((6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (5, 0), (5, 1), (5, 2), (5, 3), (4, 0), (4, 1), (4, 2), (3, 0),
                 (3, 1), (2, 0), (3, 3), (6, 6), (0, 0)):
        r, up, down = reading(((a, 6, b, 6), (a, 6, b, 6)))
        p("     %d/6 vs %d/6 at both starts: %-22s P_UP %s  P_DOWN %s" % (a, b, r, f(up), f(down)))
    p("")
    p("2. ONE START DECIDES (the other start uninformative: 0 or 12 positives in all, so it adds a constant)")
    p("        b=0 b=1 b=2 b=3 b=4 b=5 b=6")
    for a in range(7):
        row = []
        for b in range(7):
            r = reading(((a, 6, b, 6), (6, 6, 6, 6)))[0]
            row.append({"B LOWER": "L", "B HIGHER": "H"}.get(r, "."))
        p("   a=%d   %s" % (a, "   ".join(row)))
    for a, b in ((6, 0), (6, 1), (6, 2), (5, 0), (5, 1), (4, 0)):
        r, up, down = reading(((a, 6, b, 6), (0, 6, 0, 6)))
        p("     %d/6 vs %d/6 at one start, 0/6 vs 0/6 at the other: %-22s P_UP %s" % (a, b, r, f(up)))
    p("")
    p("3. MIXED EXAMPLES (start 1 ; start 2)")
    for c in (((6, 6, 0, 6), (3, 6, 3, 6)), ((5, 6, 1, 6), (3, 6, 1, 6)), ((4, 6, 1, 6), (4, 6, 1, 6)),
              ((6, 6, 2, 6), (2, 6, 0, 6)), ((5, 6, 2, 6), (4, 6, 2, 6)), ((6, 6, 3, 6), (6, 6, 3, 6)),
              ((2, 6, 2, 6), (1, 6, 1, 6)), ((0, 6, 4, 6), (1, 6, 4, 6))):
        r, up, down = reading(c)
        p("     %d/6 vs %d/6 ; %d/6 vs %d/6: %-22s P_UP %s  P_DOWN %s" % (c[0][0], c[0][2], c[1][0], c[1][2], r,
                                                                       f(up), f(down)))
    p("   With a LOST or censored continuation the arms are unequal; the same test applies with nA_s, nB_s as counted:")
    for c in (((6, 6, 0, 5), (6, 6, 0, 6)), ((5, 5, 0, 6), (4, 6, 1, 6)), ((6, 6, 1, 5), (3, 6, 2, 6))):
        r, up, down = reading(c)
        p("     %d/%d vs %d/%d ; %d/%d vs %d/%d: %-22s P_UP %s  P_DOWN %s" % (c[0][0], c[0][1], c[0][2], c[0][3],
                                                                         c[1][0], c[1][1], c[1][2], c[1][3], r,
                                                                         f(up), f(down)))
    p("")
    p("4. POWER: P(B LOWER) with the same rates at both starts (exact enumeration over binomial outcomes)")
    for k in (6, 4):
        p("   k = %d per arm per start:" % k)
        p("      pA \\ pB      0.00     0.10     0.25     0.40     0.50")
        for pA in (0.6, 0.75, 0.9, 1.0):
            cells = []
            for pB in (0.0, 0.1, 0.25, 0.4, 0.5):
                lo, _hi = power(k, [(pA, pB), (pA, pB)])
                cells.append("%.3f" % lo)
            p("      %.2f       %s" % (pA, "    ".join(cells)))
    p("   Unequal starts, k = 6: s42-like (0.9 vs 0.1) with s80-like (0.6 vs 0.1): %.3f;" % power(6, [(0.9, 0.1),
                                                                                                    (0.6, 0.1)])[0])
    p("      an effect at one start only (0.9 vs 0.1 ; 0.9 vs 0.9): %.3f;" % power(6, [(0.9, 0.1), (0.9, 0.9)])[0])
    p("      one start never fires (0.9 vs 0.2 ; 0 vs 0): %.3f." % power(6, [(0.9, 0.2), (0.0, 0.0)])[0])
    p("   Reverse direction, P(B HIGHER), k = 6: pA 0.1 vs pB 0.9: %.3f; pA 0.25 vs pB 0.9: %.3f; pA 0.5 vs pB 0.9: %.3f"
      % (power(6, [(0.1, 0.9)] * 2)[1], power(6, [(0.25, 0.9)] * 2)[1], power(6, [(0.5, 0.9)] * 2)[1]))
    p("      (an arm-A rate near 0.9 leaves little room to show B HIGHER: pA 0.9 vs pB 1.0: %.3f)"
      % power(6, [(0.9, 1.0)] * 2)[1])
    p("")
    p("5. SIZE UNDER THE NULL (pA = pB at each start), k = 6: largest P(B LOWER) and P(B HIGHER) over every pair of")
    p("   null rates on the grid 0.05, 0.10, ..., 0.95 (the test is exact: each is at most 0.025)")
    grid = [i / 20 for i in range(1, 20)]
    best_lo = best_hi = (0.0, None)
    for r1 in grid:
        for r2 in grid:
            lo, hi = power(6, [(r1, r1), (r2, r2)])
            if lo > best_lo[0]:
                best_lo = (lo, (r1, r2))
            if hi > best_hi[0]:
                best_hi = (hi, (r1, r2))
    p("   P(B LOWER) at most %.4f (at %s); P(B HIGHER) at most %.4f (at %s); sum at most %.4f"
      % (best_lo[0], best_lo[1], best_hi[0], best_hi[1], best_lo[0] + best_hi[0]))
    p("")
    p("6. CROSS-CHECK against S2/r10round/ecs/power_ecs.out section 6c (one side at 0.025):")
    for pA, pB, ref in ((0.9, 0.1, "0.993"), (0.9, 0.25, "0.897"), (0.8, 0.2, "0.812")):
        p("   k = 6, %.2f vs %.2f: %.3f (power_ecs.out %s)" % (pA, pB, power(6, [(pA, pB)] * 2)[0], ref))
    for pA, pB, ref in ((0.9, 0.1, "0.932"), (0.9, 0.25, "0.700"), (0.8, 0.2, "0.599")):
        p("   k = 4, %.2f vs %.2f: %.3f (power_ecs.out %s)" % (pA, pB, power(4, [(pA, pB)] * 2)[0], ref))
    print("\n".join(out))


if __name__ == "__main__":
    main()
