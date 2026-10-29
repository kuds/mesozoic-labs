"""ECS criteria v2: arithmetic behind the numbers v2 adds to v1 (standard library only; nothing read but v1's
table_e_ecs.py, which is loaded by absolute path and not changed; nothing simulated or trained).

    cd /tmp && S2/venv/bin/python -I -B S2/r10round/ecs/v2/check_v2.py > S2/r10round/ecs/v2/check_v2.out

1. The worst-case fill (criteria v2, THE STATISTIC): P_UP never falls when an arm-A POSITIVE becomes NEGATIVE or an
   arm-B NEGATIVE becomes POSITIVE (and P_DOWN never falls under the reverse flips), checked on every table with
   0-6 per arm at each start. So a line read on the filled table is never easier than on the complete data.
2. Power figures quoted in v2 (k = 6, two-sided 0.025 per side; and one-sided 0.05 for D6).
3. How often arm A shows the outcome at fewer than 3 of 6 at one or both starts.
4. Clopper-Pearson bounds for 3 of 3.
5. The seed lists of v2 (T1): every derived seed disjoint from every other and from M1's and the runs' own.
6. Steps, evaluations, updates, probe rows and the D3 start options' costs.
7. Arm A's applied ent_coef at the first and last update of each window (float64 and float32).
8. Worked examples of the fill.
"""
import importlib.util
import struct
from fractions import Fraction
from math import comb

V1 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/v1/table_e_ecs.py"
spec = importlib.util.spec_from_file_location("table_e_ecs_v1", V1)
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)

K = 6
SIDE = Fraction(1, 40)
ONE_SIDED = Fraction(1, 20)
out = []


def p(x):
    return f"{float(x):.4f}"


# 1. monotonicity of the fill
viol = 0
checks = 0
tables = [(a1, b1, a2, b2) for a1 in range(K + 1) for b1 in range(K + 1) for a2 in range(K + 1) for b2 in range(K + 1)]
for a1, b1, a2, b2 in tables:
    up, down = te.pvals(((a1, K, b1, K), (a2, K, b2, K)))
    cells = [a1, b1, a2, b2]
    for i in range(4):
        is_a = i % 2 == 0
        # flips that make the table less favourable to B LOWER: A pos -> neg, B neg -> pos
        if (is_a and cells[i] >= 1) or (not is_a and cells[i] <= K - 1):
            c = list(cells)
            c[i] += -1 if is_a else 1
            up2, _ = te.pvals(((c[0], K, c[1], K), (c[2], K, c[3], K)))
            checks += 1
            viol += up2 < up
        # flips that make it less favourable to B HIGHER: A neg -> pos, B pos -> neg
        if (is_a and cells[i] <= K - 1) or (not is_a and cells[i] >= 1):
            c = list(cells)
            c[i] += 1 if is_a else -1
            _, down2 = te.pvals(((c[0], K, c[1], K), (c[2], K, c[3], K)))
            checks += 1
            viol += down2 < down
out.append("1. WORST-CASE FILL: MONOTONICITY OF P_UP AND P_DOWN (k = 6 per arm per start, all 2,401 tables)")
out.append(f"   single flips checked: {checks}; flips that made a line easier: {viol}")
out.append("   So filling a LOST or censored continuation against a line gives a p at least as large as the complete")
out.append("   data would have, and the size per side stays at most that of the complete-data test (0.0116, TE 5).")

# 2. power
def enum_power(rates, rule):
    tot = 0.0
    for a1 in range(K + 1):
        for b1 in range(K + 1):
            for a2 in range(K + 1):
                for b2 in range(K + 1):
                    pr = (te.binom(K, a1, rates[0][0]) * te.binom(K, b1, rates[0][1])
                          * te.binom(K, a2, rates[1][0]) * te.binom(K, b2, rates[1][1]))
                    if pr == 0:
                        continue
                    up, down = te.pvals(((a1, K, b1, K), (a2, K, b2, K)))
                    if rule(up, down):
                        tot += pr
    return tot


def lower2(up, down):
    return up <= SIDE


def higher2(up, down):
    return down <= SIDE


def lower1(up, down):
    return up <= ONE_SIDED


out.append("")
out.append("2. POWER, k = 6, the same rates at both starts")
for pa in (0.6, 0.75, 0.9):
    row = []
    for pb in (0.1, 0.25, 0.5):
        row.append(f"B {pb}: {enum_power([(pa, pb), (pa, pb)], lower2):.3f}")
    out.append(f"   P(B LOWER) two-sided, A {pa}: " + "; ".join(row))
for pa, pb in ((0.9, 1.0), (0.75, 1.0), (0.6, 1.0)):
    out.append(f"   P(B HIGHER) two-sided, A {pa} vs B {pb}: {enum_power([(pa, pb), (pa, pb)], higher2):.3f}")
for pa, pb in ((0.9, 0.1), (0.9, 0.25), (0.9, 0.5), (0.75, 0.25)):
    one = enum_power([(pa, pb), (pa, pb)], lower1)
    two = enum_power([(pa, pb), (pa, pb)], lower2)
    out.append(f"   B lower, A {pa} vs B {pb}: one-sided 0.05 {one:.3f}; two-sided (0.025 per side) {two:.3f}")

# 3. arm A below 3 of 6
out.append("")
out.append("3. ARM A POSITIVE IN FEWER THAN 3 OF 6 (the LOW ARM-A RATE note)")
for pa in (0.5, 0.6, 0.75, 0.9):
    one = sum(te.binom(K, a, pa) for a in range(3))
    out.append(f"   pA {pa}: at one start {one:.3f}; at both starts {one * one:.4f}")

# 4. Clopper-Pearson for 3 of 3
out.append("")
out.append("4. CLOPPER-PEARSON FOR 3 OF 3 (D-D43's r9 CONTROL at s80)")
out.append(f"   95% two-sided lower bound 0.025**(1/3) = {0.025 ** (1 / 3):.3f}; one-sided 95% lower bound"
           f" 0.05**(1/3) = {0.05 ** (1 / 3):.3f}")

# 5. seeds
SEEDS = {
    ("s42", "A"): [1101, 1111, 1121, 1131, 1141, 1151],
    ("s42", "B"): [1201, 1211, 1221, 1231, 1241, 1251],
    ("s80", "A"): [1301, 1311, 1321, 1331, 1341, 1351],
    ("s80", "B"): [1401, 1411, 1421, 1431, 1441, 1451],
}
used = {}
clash = []
for key, lst in SEEDS.items():
    for s in lst:
        for d in [s + r for r in range(4)] + [s + 1000]:
            if d in used:
                clash.append((d, key, used[d]))
            used[d] = key
reserved = set(range(3042, 3062))
for run in (42, 60, 80):
    reserved |= {run + r for r in range(4)} | {run + 1000}
hit = sorted(d for d in used if d in reserved)
out.append("")
out.append("5. SEEDS (model seed s, training envs s + 0..3, evaluation env s + 1000)")
for key, lst in SEEDS.items():
    out.append(f"   {key[0]} arm {key[1]}: {lst}")
out.append(f"   derived seeds: {len(used)}; clashes among them: {len(clash)}; overlaps with M1 3042-3061 or the runs'"
           f" own (42, 60, 80 and their +0..3, +1000): {len(hit)}")

# 6. steps and costs
G = 131072
R = 8192
U = 1048576
starts = {"s42": (3145728, 3801088), "s80": (4194304, 4849664)}
L = 1572864
out.append("")
out.append("6. STEPS")
for name, (S, first) in starts.items():
    E = S + L
    out.append(f"   {name}: S {S:,}, E {E:,}; evaluations in (S, E] {L // G}; rollouts {L // R}, updates {L // R - 1}, "
               f"probe rows in (S, E) {L // R - 1}; original first ANY-flag lead {first - S:,}, margin to E {E - first:,}")
base = 2 * L
opts = {
    "(a) s42 3,145,728 and s80 4,194,304 to E": 2 * L,
    "(b) s80 from 3,145,728 to 5,767,168": L + (5767168 - 3145728),
    "(c) both from 2,097,152 to the same E": (4718592 - 2097152) + (5767168 - 2097152),
    "(d) both from 1,048,576 to the same E": (4718592 - 1048576) + (5767168 - 1048576),
}
for k, v in opts.items():
    steps24 = v * 12
    lo, hi = steps24 / U * 43 / 60, steps24 / U * 53 / 60
    out.append(f"   D3 {k}: {v:,} steps per arm-and-seed pair of starts, x{v / base:.2f}; at k = 6 {steps24:,} steps, "
               f"{lo:.1f}-{hi:.1f} CPU-h, {lo / 3:.1f}-{hi / 3:.1f} h at 3 at a time")
for name, S in (("s42 1,048,576", 1048576), ("s80 1,048,576", 1048576), ("s42 2,097,152", 2097152),
                ("s80 2,097,152", 2097152)):
    first = 3801088 if name.startswith("s42") else 4849664
    out.append(f"   lead from {name} to the original first ANY flag: {first - S:,}")

# 7. ent_coef
def sched(t):
    return 0.005 + min(1.0, t / 7000000) * (0.0 - 0.005)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


out.append("")
out.append("7. ARM A's APPLIED ent_coef (0.005 to 0 linearly by 7,000,000 on the cumulative step)")
for name, (S, _first) in starts.items():
    E = S + L
    for t in (S, S + R, E - R):
        out.append(f"   {name} at {t:,}: {sched(t):.10f} (float32 {f32(sched(t)):.10f})")
    mean = sum(sched(S + R * i) for i in range(1, L // R)) / (L // R - 1)
    out.append(f"   {name} mean over the 191 updates: {mean:.6f}")

# 8. examples
out.append("")
out.append("8. THE FILL, WORKED (as counted -> filled against the line; '-' marks a LOST or censored continuation)")
ex = [
    ("6/6 vs 0/5(-1) ; 6/6 vs 0/6", ((6, 6, 0, 5), (6, 6, 0, 6)), ((6, 6, 1, 6), (6, 6, 0, 6))),
    ("5/5(-1) vs 0/6 ; 4/6 vs 1/6", ((5, 5, 0, 6), (4, 6, 1, 6)), ((5, 6, 0, 6), (4, 6, 1, 6))),
    ("4/5(-1) vs 1/6 ; 4/6 vs 1/6", ((4, 5, 1, 6), (4, 6, 1, 6)), ((4, 6, 1, 6), (4, 6, 1, 6))),
    ("6/6 vs 1/5(-1) ; 6/6 vs 2/6", ((6, 6, 1, 5), (6, 6, 2, 6)), ((6, 6, 2, 6), (6, 6, 2, 6))),
    ("6/6 vs 3/5(-1) ; 6/6 vs 3/6", ((6, 6, 3, 5), (6, 6, 3, 6)), ((6, 6, 4, 6), (6, 6, 3, 6))),
]
for label, counted, filled in ex:
    cu, _ = te.pvals(counted)
    fu, _ = te.pvals(filled)
    out.append(f"   {label}: as counted P_UP {p(cu)} ({'L' if cu <= SIDE else '.'}); filled P_UP {p(fu)} "
               f"({'L' if fu <= SIDE else '.'})")

print("\n".join(out))
