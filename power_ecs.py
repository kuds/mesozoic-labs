"""Entropy continuation screen (ECS): power and false-positive arithmetic.  Pure arithmetic; standard library only.

    PY -I -B ECS/power_ecs.py > ECS/power_ecs.out        (PY = S2/venv/bin/python; cwd /tmp)

No physics, no training, no files read.  Every figure is exact enumeration over binomial outcomes except section 7
(Monte Carlo of a statistical model, seeded, labelled as such).

Model: S strata (the start states), k continuations per arm per stratum, a binary outcome per continuation (e.g. an
ANY-flagged evaluation inside the continuation).  Arm A (recorded ent_coef) positive with probability pA_s, arm B
(ent_coef 0) with pB_s, independent.  H1 (one-sided): B lower.

Tests:
  CMHX  exact conditional test of a common odds ratio (stratified; the "exact Cochran-Mantel-Haenszel" test).
        Statistic T = total positives in arm A; null distribution = convolution over strata of the hypergeometric
        (k, k, m_s), conditional on each stratum's total m_s.  p = P(T >= t_obs).  With equal k per arm in every stratum
        this is identical to a stratified permutation test (labels permuted within strata) on the sum over strata of
        the A-minus-B difference, since within a stratum the difference is 2 a_s - m_s.
  FISH  one-sided Fisher exact test on the strata pooled (ignores the strata).
  SIGN  exact sign (McNemar) test on discordant pairs, when continuation seed j is run in both arms (paired seeds).
"""
from __future__ import annotations

import random
from functools import lru_cache
from math import comb

ALPHA = 0.05
EPS = 1e-12


def binom(n, k, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


@lru_cache(maxsize=None)
def hyper(k, m):
    """Null pmf of the A count in a stratum with k per arm and m positives in all: list over a."""
    tot = comb(2 * k, m)
    return tuple(comb(k, a) * comb(k, m - a) / tot if 0 <= m - a <= k else 0.0 for a in range(k + 1))


def conv(d1, d2):
    out = {}
    for t1, p1 in d1.items():
        for t2, p2 in d2.items():
            out[t1 + t2] = out.get(t1 + t2, 0.0) + p1 * p2
    return out


@lru_cache(maxsize=None)
def null_T(k, ms):
    d = {0: 1.0}
    for m in ms:
        d = conv(d, {a: p for a, p in enumerate(hyper(k, m)) if p > 0})
    return tuple(sorted(d.items()))


def cmhx_p(k, cells):
    """cells: tuple of (a_s, b_s).  One-sided p, A higher."""
    ms = tuple(a + b for a, b in cells)
    t = sum(a for a, _ in cells)
    return sum(p for v, p in null_T(k, ms) if v >= t)


def fisher_p(a, b, nA, nB):
    """One-sided Fisher exact, A higher: P(X >= a | total a + b), X ~ hypergeometric(nA, nB)."""
    m = a + b
    tot = comb(nA + nB, m)
    return sum(comb(nA, x) * comb(nB, m - x) for x in range(a, min(nA, m) + 1) if 0 <= m - x <= nB) / tot


def sign_p(n10, n01):
    """Exact one-sided sign test on discordant pairs: n10 = pairs A+ B-, n01 = pairs A- B+."""
    n = n10 + n01
    if n == 0:
        return 1.0
    return sum(comb(n, x) for x in range(n10, n + 1)) / 2 ** n


def outcomes(k, pA, pB):
    for a in range(k + 1):
        for b in range(k + 1):
            yield a, b, binom(k, a, pA) * binom(k, b, pB)


def power_cmhx(k, rates, alpha=ALPHA):
    """rates: list of (pA_s, pB_s) per stratum.  Returns (P(reject), P(p <= alpha) computed exactly)."""
    def rec(i, cells, prob):
        if i == len(rates):
            return prob if cmhx_p(k, tuple(cells)) <= alpha + EPS else 0.0
        s = 0.0
        pA, pB = rates[i]
        for a, b, pr in outcomes(k, pA, pB):
            if pr > 0:
                s += rec(i + 1, cells + [(a, b)], prob * pr)
        return s
    return rec(0, [], 1.0)


def power_fisher_pooled(k, rates, alpha=ALPHA):
    S = len(rates)
    def rec(i, a, b, prob):
        if i == S:
            return prob if fisher_p(a, b, k * S, k * S) <= alpha + EPS else 0.0
        s = 0.0
        pA, pB = rates[i]
        for x, y, pr in outcomes(k, pA, pB):
            if pr > 0:
                s += rec(i + 1, a + x, b + y, prob * pr)
        return s
    return rec(0, 0, 0, 1.0)


def power_fisher_single(n, pA, pB, alpha=ALPHA):
    return sum(pr for a, b, pr in outcomes(n, pA, pB) if fisher_p(a, b, n, n) <= alpha + EPS)


def power_sign_paired(k, rates, rho=0.0, alpha=ALPHA):
    """Paired seeds: per pair, joint (A, B) with marginals pA, pB and correlation rho (rho 0 = independent)."""
    pairs = []
    for pA, pB in rates:
        sd = (pA * (1 - pA) * pB * (1 - pB)) ** 0.5
        cov = rho * sd
        if sd > 0:  # clamp to the largest correlation the two marginals allow
            cov = min(cov, min(pA, pB) - pA * pB)
            cov = max(cov, max(0.0, pA + pB - 1) - pA * pB)
        p11 = pA * pB + cov
        p10 = pA - p11
        p01 = pB - p11
        p00 = 1 - p11 - p10 - p01
        assert min(p11, p10, p01, p00) >= -1e-12, (pA, pB, rho)
        pairs.append((max(p11, 0), max(p10, 0), max(p01, 0), max(p00, 0)))
    n = k * len(rates)
    # distribution of (n10, n01) over all n pairs
    dist = {(0, 0): 1.0}
    for p11, p10, p01, p00 in pairs:
        for _ in range(k):
            nd = {}
            for (x, y), pr in dist.items():
                for dx, dy, q in ((0, 0, p11 + p00), (1, 0, p10), (0, 1, p01)):
                    if q > 0:
                        nd[(x + dx, y + dy)] = nd.get((x + dx, y + dy), 0.0) + pr * q
            dist = nd
    assert n == sum(1 for _ in range(n))
    return sum(pr for (x, y), pr in dist.items() if sign_p(x, y) <= alpha + EPS)


def size_cmhx_paired(k, rates_null, rho, alpha=ALPHA):
    """Size of CMHX when the k pairs per stratum share seeds and are correlated (rho) under H0 (pA = pB = p_s)."""
    def stratum_dist(p):
        cov = rho * p * (1 - p)
        p11 = p * p + cov
        p10 = p01 = p - p11
        p00 = 1 - p11 - 2 * p10
        d = {(0, 0): 1.0}
        for _ in range(k):
            nd = {}
            for (a, b), pr in d.items():
                for da, db, q in ((1, 1, p11), (1, 0, p10), (0, 1, p01), (0, 0, p00)):
                    if q > 0:
                        nd[(a + da, b + db)] = nd.get((a + da, b + db), 0.0) + pr * q
            d = nd
        return d
    dists = [stratum_dist(p) for p in rates_null]
    def rec(i, cells, prob):
        if i == len(dists):
            return prob if cmhx_p(k, tuple(cells)) <= alpha + EPS else 0.0
        return sum(rec(i + 1, cells + [ab], prob * pr) for ab, pr in dists[i].items())
    return rec(0, [], 1.0)


def line(s=""):
    print(s)


def main():
    line("ECS POWER AND FALSE-POSITIVE ARITHMETIC (exact enumeration unless marked; one-sided alpha 0.05, B lower)")
    line("")
    # ------------------------------------------------------------------------------------------- 1. checks
    line("1. CHECK OF THE FIGURES IN NEXT_STEPS AND REVIEW 4.5")
    line("   Option 1 (NEXT_STEPS RISKS):")
    line("     4/4 vs 0/4, one stratum, Fisher one-sided: %.4f   (NEXT_STEPS 0.014)" % fisher_p(4, 0, 4, 4))
    line("     3/4 vs 0/4, one stratum, Fisher one-sided: %.4f   (NEXT_STEPS 0.071)" % fisher_p(3, 0, 4, 4))
    line("     3/4 vs 0/4 at each of 2 starts, CMHX:       %.4f   (NEXT_STEPS 0.005)" % cmhx_p(4, ((3, 0), (3, 0))))
    line("     the same table pooled (6/8 vs 0/8), Fisher:  %.4f   (pooling ignores the starts; see section 3)" %
         fisher_p(6, 0, 8, 8))
    line("   Option 3 (NEXT_STEPS COST, REVIEW 4.5; from-scratch arms, one stratum, unconditional power of the one-sided")
    line("   Fisher exact test at alpha 0.05):")
    for n in (6, 8):
        for pa, pb, lab in ((2 / 3, 0.0, "2/3 vs 0 (cure)"), (2 / 3, 1 / 3, "2/3 vs 1/3 (halved)"),
                            (0.4, 0.0, "0.4 vs 0 (r9 rate vs cure)"), (0.4, 0.2, "0.4 vs 0.2")):
            line("     n = %d per arm, %-26s power %.3f" % (n, lab, power_fisher_single(n, pa, pb)))
    line("     (NEXT_STEPS / REVIEW: 0.68 and 0.91 for the cure; 0.18 at 6 or 8 for halved; 0.18 and 0.41 for 0.4 vs 0)")
    line("     p at fixed outcomes: 4/6 vs 0/6 %.4f; 5/8 vs 0/8 %.4f; 4/8 vs 0/8 %.4f (REVIEW 0.030 / 0.013 / 0.038)" % (
        fisher_p(4, 0, 6, 6), fisher_p(5, 0, 8, 8), fisher_p(4, 0, 8, 8)))
    def lower_p(a, nA, b, nB):  # one-sided Fisher, first group lower: P(X <= a | total a + b)
        m = a + b
        return sum(comb(nA, x) * comb(nB, m - x) for x in range(0, a + 1) if 0 <= m - x <= nB) / comb(nA + nB, m)
    line("   Option 5 and the round (first group lower): 2/5 vs 2/5 %.4f (NEXT_STEPS 0.74); 0/5 vs 2/5 %.4f (0.22); "
         "the round's 2/3 vs 2/5 %.4f (SCORE 0.9286)" % (lower_p(2, 5, 2, 5), lower_p(0, 5, 2, 5), lower_p(2, 3, 2, 5)))
    line("")
    # ------------------------------------------------------------------------------------------- 2. attainable p
    line("2. WHAT EACH DESIGN CAN REACH (CMHX, 2 strata; smallest p and the tables that reach p <= 0.05)")
    for k in (3, 4, 6):
        best = cmhx_p(k, ((k, 0), (k, 0)))
        one = fisher_p(k, 0, k, k)
        line("   k = %d per arm per start (%d continuations in all): all A positive, all B negative at both starts p = %.5f;"
             " at one start only (other start uninformative, e.g. 0/k vs 0/k) p = %.4f" % (k, 4 * k, best,
                                                                                         cmhx_p(k, ((k, 0), (0, 0)))))
        # symmetric tables a/k vs b/k at both starts
        cells = []
        for a in range(k + 1):
            for b in range(a):
                p = cmhx_p(k, ((a, b), (a, b)))
                if p <= ALPHA + EPS:
                    cells.append("%d/%d vs %d/%d (p %.4f)" % (a, k, b, k, p))
        line("      same table at both starts, reaching 0.05: " + ("; ".join(cells) if cells else "none"))
        line("      single start only: smallest p %.4f%s" % (one, "" if one <= ALPHA + EPS else " (cannot reach 0.05)"))
        single = ["%d/%d vs %d/%d (p %.4f)" % (a, k, b, k, fisher_p(a, b, k, k))
                  for a in range(k + 1) for b in range(a) if fisher_p(a, b, k, k) <= ALPHA + EPS]
        line("      single-start tables reaching 0.05 (Fisher): " + ("; ".join(single) if single else "none"))
    line("")
    # ------------------------------------------------------------------------------------------- 3. power grid
    line("3. POWER, CMHX, 2 STRATA WITH THE SAME RATES (and, in brackets, Fisher on the pooled table)")
    pAs = (0.6, 0.75, 0.8, 0.9, 1.0)
    pBs = (0.0, 0.1, 0.25, 0.4, 0.5)
    for k in (3, 4, 6):
        line("   k = %d per arm per start:" % k)
        line("      pA \\ pB  " + "  ".join("%13.2f" % pb for pb in pBs))
        for pa in pAs:
            cells = []
            for pb in pBs:
                pw = power_cmhx(k, [(pa, pb), (pa, pb)])
                pf = power_fisher_pooled(k, [(pa, pb), (pa, pb)])
                cells.append("%5.3f [%5.3f]" % (pw, pf))
            line("      %5.2f     " % pa + "  ".join(cells))
    line("")
    line("4. POWER, CMHX, UNEQUAL STARTS (s42-like start with a sure arm-A onset, s80-like start with a doubtful one)")
    cases = [((0.9, 0.1), (0.6, 0.1)), ((0.9, 0.25), (0.6, 0.25)), ((0.9, 0.5), (0.6, 0.3)),
             ((0.9, 0.1), (0.9, 0.9)), ((1.0, 0.0), (0.5, 0.0)), ((0.9, 0.2), (0.0, 0.0))]
    for k in (3, 4, 6):
        for c in cases:
            line("   k = %d  s42 (pA %.2f, pB %.2f)  s80 (pA %.2f, pB %.2f): power %.3f" % (
                k, c[0][0], c[0][1], c[1][0], c[1][1], power_cmhx(k, list(c))))
    line("   (the case s80 (0.9, 0.9) is an effect at one start only; (0.0, 0.0) is a start whose outcome never fires)")
    line("")
    line("5. ONE START ALONE (a single stratum: e.g. add-on C at s60, or the core if only one start is informative)")
    for k in (3, 4, 6):
        cells = ["pA %.2f pB %.2f: %.3f" % (pa, pb, power_fisher_single(k, pa, pb))
                 for pa, pb in ((0.9, 0.1), (0.9, 0.25), (0.8, 0.2), (1.0, 0.0))]
        line("   k = %d: " % k + "; ".join(cells))
    line("")
    # ------------------------------------------------------------------------------------------- 6. size
    line("6. FALSE-POSITIVE RATE UNDER THE NULL (pA = pB in each stratum), CMHX at nominal 0.05")
    nulls = [(0.1, 0.1), (0.3, 0.3), (0.5, 0.5), (0.7, 0.7), (0.9, 0.9), (0.9, 0.5), (0.9, 0.1), (1.0, 0.5)]
    for k in (3, 4, 6):
        cells = ["p_s42 %.1f p_s80 %.1f: %.4f" % (p1, p2, power_cmhx(k, [(p1, p1), (p2, p2)])) for p1, p2 in nulls]
        line("   k = %d: " % k + "; ".join(cells))
    line("   (exact conditional test: the size is <= 0.05 at every null rate; the shortfall is the price of discreteness)")
    grid = [i / 20 for i in range(1, 20)]
    for k in (3, 4, 6):
        best = max((power_cmhx(k, [(p1, p1), (p2, p2)]), p1, p2) for p1 in grid for p2 in grid if p2 >= p1)
        line("   largest CMHX size over every pair of null rates on the grid 0.05, 0.10, ..., 0.95, k = %d: %.4f (at %.2f, %.2f)"
             % (k, best[0], best[1], best[2]))
    for k in (3, 4, 6):
        cells = ["p_s42 %.1f p_s80 %.1f: %.4f" % (p1, p2, power_fisher_pooled(k, [(p1, p1), (p2, p2)]))
                 for p1, p2 in nulls]
        line("   pooled Fisher, k = %d: " % k + "; ".join(cells))
    line("   Several outcomes or add-ons each tested at 0.05: the chance that at least one reads p <= 0.05 by chance is at")
    line("   most the sum of their sizes (e.g. 4 readings at a size of 0.03 each: <= 0.12), which is why one primary.")
    line("")
    line("6b. PAIRED SEEDS (continuation seed j run in both arms). CMHX treats the 2k continuations as exchangeable; with")
    line("    positively correlated pairs it stays valid (conservative). Size of CMHX at nominal 0.05, pair correlation rho:")
    for k in (3, 4, 6):
        cells = []
        for p in (0.3, 0.5, 0.7, 0.9):
            for rho in (0.0, 0.5, -0.2):
                cells.append("p %.1f rho %+.1f: %.4f" % (p, rho, size_cmhx_paired(k, (p, p), rho)))
        line("   k = %d: " % k + "; ".join(cells))
    line("    Power of the exact sign test on discordant pairs (all 2k pairs pooled over both starts), for comparison")
    line("    (rho 0.3 is clamped to the largest correlation the two marginals allow, e.g. 0.11 at 0.9 vs 0.1):")
    for k in (3, 4, 6):
        cells = []
        for (pa, pb) in ((0.9, 0.1), (0.9, 0.25), (0.8, 0.2), (0.75, 0.5)):
            cells.append("pA %.2f pB %.2f: sign rho 0 %.3f, rho 0.3 %.3f | CMHX %.3f" % (
                pa, pb, power_sign_paired(k, [(pa, pb), (pa, pb)], 0.0),
                power_sign_paired(k, [(pa, pb), (pa, pb)], 0.3), power_cmhx(k, [(pa, pb), (pa, pb)])))
        line("   k = %d: " % k + "; ".join(cells))
    line("")
    line("6c. TWO-SIDED READING (0.025 each side; detects either direction, e.g. removing the bonus making the rail worse)")
    for k in (3, 4, 6):
        cells = ["pA %.2f pB %.2f: %.3f" % (pa, pb, power_cmhx(k, [(pa, pb), (pa, pb)], alpha=0.025))
                 for pa, pb in ((0.9, 0.1), (0.9, 0.25), (0.8, 0.2), (1.0, 0.0))]
        line("   k = %d: " % k + "; ".join(cells))
    line("")
    # ------------------------------------------------------------------------------------------- 7. continuous
    line("7. A GRADED READING INSTEAD OF A YES/NO ONE [Monte Carlo of a statistical model, seed 20261028, 4,000 draws per")
    line("   cell]: stratified permutation test (exact over within-stratum relabelings) on a continuous per-continuation")
    line("   number (e.g. the share of the continuation's evaluations that are ANY-flagged, or the end-trace |CMD_a|),")
    line("   normal with a shift of d between-continuation SDs between the arms in both strata.")
    rng = random.Random(20261028)
    from itertools import combinations
    for k in (3, 4, 6):
        subsets = list(combinations(range(2 * k), k))
        cells = []
        for d in (1.0, 1.5, 2.0, 3.0):
            hits = 0
            for _ in range(4000):
                sums = []
                obs = 0.0
                for _s in range(2):
                    xa = [rng.gauss(d, 1.0) for _ in range(k)]
                    xb = [rng.gauss(0.0, 1.0) for _ in range(k)]
                    allx = xa + xb
                    obs += sum(xa)
                    sums.append(sorted(sum(allx[i] for i in sub) for sub in subsets))
                s1, s2 = sums
                import bisect
                cnt = 0
                for v in s1:
                    cnt += len(s2) - bisect.bisect_left(s2, obs - v - 1e-12)
                p = cnt / (len(s1) * len(s2))
                if p <= ALPHA + EPS:
                    hits += 1
            cells.append("d %.1f: %.3f" % (d, hits / 4000))
        line("   k = %d: " % k + "; ".join(cells))
    line("   (a graded reading has more power for a partial effect, but needs its between-continuation SD, which no")
    line("   record measures yet; the binary reading needs no such number)")
    line("")
    cost()


def cost():
    """Steps and CPU-hours per design [inferred rates: D-D43 44-46 min of wall per 1,048,576 steps per continuation,
    3 at a time, OMP 1 (RUNLOG l567-l572), i.e. about 0.73-0.77 CPU-h per 1,048,576, rounded up to 0.75-0.80 as in
    NEXT_STEPS, plus about 10% for the in-run ledger (10 deterministic 1,000-step episodes per 131,072 steps)]."""
    M = 1048576
    lo, hi = 0.75 * 1.10, 0.80 * 1.10
    designs = {
        "D1 NEXT_STEPS: s42 3,145,728 + 1,572,864; s80 3,145,728 + 2,359,296": (1572864, 2359296),
        "D2 s42 3,145,728 + 1,572,864; s80 4,194,304 + 1,572,864 (both F2-held, both 655,360 before the first flag)":
            (1572864, 1572864),
        "D3 s42 3,145,728 + 1,572,864; s80 3,145,728 + 3,145,728 (s80 to 6,291,456)": (1572864, 3145728),
        "D4 earlier: s42 2,097,152 + 2,621,440; s80 2,097,152 + 3,407,872 (same end steps as D1)": (2621440, 3407872),
        "D5 earliest: s42 1,048,576 + 3,670,016; s80 1,048,576 + 4,456,448 (same end steps as D1)": (3670016, 4456448),
        "C add-on: s60 6,291,456 + 2,097,152 (2 arms)": (2097152,),
    }
    line("8. COST ARITHMETIC [inferred rate %.3f-%.3f CPU-h per 1,048,576 steps incl. the in-run ledger; wall = CPU-h / 3"
         " at 3 at a time, before wave packing]" % (lo, hi))
    for name, Ls in designs.items():
        for k in (3, 4, 6):
            steps = 2 * k * sum(Ls)
            line("   %-100s k %d: %3d continuations, %11s steps, %5.1f-%5.1f CPU-h, about %4.1f-%4.1f h wall" % (
                name, k, 2 * k * len(Ls), f"{steps:,}", steps / M * lo, steps / M * hi, steps / M * lo / 3,
                steps / M * hi / 3))


if __name__ == "__main__":
    main()
