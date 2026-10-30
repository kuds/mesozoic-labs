"""Reviewer's independent re-derivation of the BS test (no training, no simulation).

    cd /tmp && PY -I -B <BS>/review/rv_rederive.py

Two synthetic sets, worked by hand in integer units of 1/1024 (binary exact, so the doubles' exact rationals are the
integers / 1024), then compared with score_bs.perm_test / decide.  Then the fill by brute-force supremum over a grid
of completions, and a third set in decimals (exact rationals of the doubles computed here with Fraction(float)).
"""
import importlib.util
import itertools
import sys
from fractions import Fraction

BS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen"
spec = importlib.util.spec_from_file_location("score_bs", BS + "/tools/score_bs.py")
SB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SB)
B_S = Fraction(SB.BASELINE_S)


def my_p(control_int, blind_int):
    """One-sided exact permutation p on integers: share of 4-subsets of the pooled 8 whose sum >= the control sum."""
    pooled = list(control_int) + list(blind_int)
    obs = sum(control_int)
    subs = list(itertools.combinations(range(len(pooled)), len(control_int)))
    return sum(1 for s in subs if sum(pooled[i] for i in s) >= obs), len(subs)


def as_pairs(xs):
    return [(0, Fraction(x)) for x in xs]


def run_set(label, c_int, b_int, expect_count):
    c = [Fraction(x, 1024) for x in c_int]
    b = [Fraction(x, 1024) for x in b_int]
    cnt, tot = my_p(c_int, b_int)
    p, count, total = SB.perm_test(as_pairs(c), as_pairs(b))
    names = {}
    classes = []
    for seed, cv, bv in zip(SB.SEEDS, c, b, strict=True):
        for arm, v in ((SB.CONTROL, cv), (SB.INTERVENTION, bv)):
            n = f"{arm}_{seed}"
            names[n] = v
            classes.append({"name": n, "arm": arm, "seed": seed})
    dec = SB.decide(classes, names)
    med = (sorted(c)[1] + sorted(c)[2]) / 2
    rise = med - B_S
    print(f"{label}: control {c_int} blind {b_int} (units 1/1024)")
    print(f"  by hand: {expect_count}/70; my enumeration {cnt}/{tot}; scorer {count}/{total} p={float(p):.4f}")
    print(f"  my control median {float(med):+.5f} rise {float(rise):+.5f} informative {rise >= Fraction(1, 10)}; "
          f"scorer rise {float(dec['control_rise']):+.5f} informative {dec['informative']} line {dec['line']} "
          f"{dec['key']} additions {dec['additions']}")
    assert cnt == count == expect_count and tot == total == 70
    my_line = 1 if rise < Fraction(1, 10) else (2 if Fraction(cnt, 70) <= Fraction(1, 20) else 3)
    assert dec["line"] == my_line, (dec["line"], my_line)
    return c, b


# Set A (boundary, worked by hand): pooled sorted 320 > 280 > 230 > 190 > 150 > 120 > 90 > 50.
# Control {320, 280, 230, 120} sum 950.  4-subsets with sum >= 950: {320,280,230,190} 1020, {320,280,230,150} 980,
# {320,280,230,120} 950 (observed); next {320,280,190,150} 940 < 950, {320,280,230,90} 920.  -> 3/70 = 0.0429: REDUCED.
cA, bA = run_set("SET A", [320, 120, 280, 230], [50, 190, 90, 150], 3)
# Set A' : swap 120 -> 110 in control: observed 940; subsets >= 940: 1020, 980, 950? no (950 contains 120 which is now
# in blind: {320,280,230,120} is a subset with 120 a blind value) -> {320,280,230,190}=1020, {320,280,230,150}=980,
# {320,280,230,120}=950, {320,280,190,150}=940 (tie: counts), {320,280,230,110}=940 (observed) -> 5/70: NO REDUCTION.
run_set("SET A'", [320, 110, 280, 230], [50, 190, 120, 150], 5)
# Set B (ties and a not-informative control): control {0, 40, 80, 10} (rise from B_S about -0.0316 + 0.0244 < 0.10)
# blind {-200, -150, -180, -190}: control is the top-4 -> 1/70, but line 1 (not informative).
run_set("SET B", [0, 40, 80, 10], [-200, -150, -180, -190], 1)
# Set C (every value equal): 70/70.
run_set("SET C", [100] * 4, [100] * 4, 70)

# THE FILL: a missing blind value; supremum over a grid of completions equals the filled p.
c = as_pairs([Fraction(x, 1024) for x in (320, 120, 280, 230)])
b3 = as_pairs([Fraction(x, 1024) for x in (50, 190, 90)])
filled = SB.perm_test(c, b3 + [SB.INF_BLIND])
grid = [Fraction(k, 1024) for k in range(-4000, 4001, 7)] + [Fraction(10**9)]
sup = max(SB.perm_test(c, b3 + [(0, v)])[1] for v in grid)
print(f"FILL blind missing: filled {filled[1]}/70, supremum over {len(grid)} completions {sup}/70")
assert filled[1] == sup
# a missing control value and a missing blind value together
c3 = as_pairs([Fraction(x, 1024) for x in (320, 280, 230)])
filled2 = SB.perm_test(c3 + [SB.INF_CONTROL], b3 + [SB.INF_BLIND])
sup2 = max(SB.perm_test(c3 + [(0, u)], b3 + [(0, v)])[1]
           for u in [Fraction(k, 1024) for k in range(-3000, 3001, 97)] + [Fraction(-10**9)]
           for v in [Fraction(k, 1024) for k in range(-3000, 3001, 97)] + [Fraction(10**9)])
print(f"FILL control + blind missing: filled {filled2[1]}/70, grid supremum {sup2}/70")
assert filled2[1] >= sup2
# minimum p and the attainable values <= 0.05
attain = sorted({Fraction(k, 70) for k in range(1, 71)})
print("min p", attain[0], "attainable <= 1/20:", [str(a) for a in attain if a <= Fraction(1, 20)])

# A decimal set: exact rationals of the recorded doubles (no tolerance), computed independently here.
cd = [0.2113, 0.1874, 0.2391, 0.1650]
bd = [0.0921, 0.1120, 0.1874, 0.1402]
pooled = [Fraction(x) for x in cd + bd]
obs = sum(Fraction(x) for x in cd)
mine = sum(1 for s in itertools.combinations(range(8), 4) if sum(pooled[i] for i in s) >= obs)
p = SB.perm_test(as_pairs(cd), as_pairs(bd))
print(f"DECIMAL SET: mine {mine}/70, scorer {p[1]}/70")
assert mine == p[1]
print("ALL RE-DERIVATIONS AGREE")
sys.exit(0)
