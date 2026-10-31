"""Power of BS2's pre-registered reading (4 control vs 4 critic_blind_tail3 from r10 stance6 s42's 2,097,152 state to
E = 3,538,944; the exact one-sided unpaired permutation test on the difference of arm means, blind lower, REDUCED when
p <= 0.05, i.e. the observed split among the 3 highest of the 70 control-arm sums, ties against; the informativeness
gate: the median of the control primaries read (midpoint of the two middle values for 4) minus B_S >= GATE).

Every value is a RISE from B_S (the test and the gate are translation-invariant, so B_S itself does not enter).
Deterministic: numpy default_rng with fixed seeds.  Sections:
  A  exact facts: attainable p, the size, the fill bounds (enumerated)
  B  the test alone, normal location model, by the standardised effect d = delta / sigma (Monte Carlo)
  C  the gate alone: P(informative) by the control arm's true mean rise m and SD sigma, for each candidate gate
  D  the three lines at the chosen gate, normal model: m in {R, 3R/4, R/2, R/4}, sigma 0.05-0.34, reduction f*m
  E  the gate and the test share the control values: P(REDUCED | informative) under no effect, per gate
  F  departures: blind arm with a small SD; a positive pair (shared-seed) effect
  G  the "chance" (two-state) model BS suggested: each control climbs like the parent with probability q
  H  EXACT enumeration on BS's own rises (scaled to BS2's window, and unscaled): 4^4 x 4^4 equally likely draws
Inputs [measured]: B_S and R from parent_window.out (the parent's F2 probe); BS's per-continuation rises from
BS check/stat/recompute.out; BS's arm SDs from BS score/score.txt; the ECS spread from BS scratch/criteria_work/
ecs_spread.out.  The normal and two-state models and every sigma are ASSUMPTIONS [inferred], stated per section.
Usage: PY -I -B power_bs2.py > power_bs2.out
"""
import itertools
import math
from fractions import Fraction

import numpy as np

B_S = 0.1265954738482833            # parent_window.out: 135930855/1073741824
R = 0.5787834012880921              # parent's primary at E (0.7053788751363754) minus B_S
BS_R = 0.23113293549977243          # BS's parent rise over (1,048,576, 2,490,368]
BS_CONTROL = (0.00055768666788935661, 0.057248691737186164, 0.12643680511973798, 0.31430359347723424)
BS_BLIND = (0.060257027624174953, 0.0031250197207555175, 0.068467637058347464, 0.047205831331666559)
GATES = (0.10, 0.15, 0.20, 0.25, 0.29)
GATE = 0.20                          # the chosen gate (criteria_v1.txt THE INFORMATIVENESS CONDITION)
SIGMAS = (0.05, 0.10, 0.137, 0.20, 0.34)
N = 4
COMBOS = list(itertools.combinations(range(2 * N), N))
assert COMBOS[0] == (0, 1, 2, 3)
SUB = np.array([[1.0 if i in c else 0.0 for i in range(2 * N)] for c in COMBOS])      # 70 x 8
DRAWS = 200_000
CHUNK = 50_000


def reduced_mask(ctrl, blind):
    """True where the observed control-sum ranks among the 3 highest of the 70 (ties count against)."""
    x = np.concatenate([ctrl, blind], axis=1)
    sums = x @ SUB.T
    count = (sums >= sums[:, :1]).sum(axis=1)
    return count <= 3


def median4(ctrl):
    s = np.sort(ctrl, axis=1)
    return 0.5 * (s[:, 1] + s[:, 2])


def lines(gen, gates, draws=DRAWS, seed=0):
    """gen(rng, n) -> (ctrl (n,4), blind (n,4)).  Returns {gate: (P1, P2, P3, P(REDUCED | informative))} and
    P(p <= 0.05) ignoring the gate."""
    rng = np.random.default_rng(seed)
    acc = {g: [0, 0, 0] for g in gates}
    red_all = 0
    done = 0
    while done < draws:
        n = min(CHUNK, draws - done)
        c, b = gen(rng, n)
        red = reduced_mask(c, b)
        med = median4(c)
        red_all += int(red.sum())
        for g in gates:
            inf = med >= g
            acc[g][0] += int((~inf).sum())
            acc[g][1] += int((inf & red).sum())
            acc[g][2] += int((inf & ~red).sum())
        done += n
    out = {}
    for g in gates:
        p1, p2, p3 = (x / draws for x in acc[g])
        out[g] = (p1, p2, p3, (p2 / (p2 + p3)) if p2 + p3 > 0 else float("nan"))
    return out, red_all / draws


def normal_gen(m, sc, delta, sb=None, tau=0.0):
    """control rise ~ N(m, sc^2), blind ~ N(m - delta, sb^2); optional shared pair effect N(0, tau^2) per seed pair
    (added to both members of a pair; the arms' own SDs are then sqrt(sc^2 - tau^2) so the total SD stays sc)."""
    sb = sc if sb is None else sb

    def gen(rng, n):
        if tau > 0:
            own_c = math.sqrt(max(sc * sc - tau * tau, 0.0))
            own_b = math.sqrt(max(sb * sb - tau * tau, 0.0))
            pair = rng.normal(0.0, tau, size=(n, N))
            c = m + pair + rng.normal(0.0, own_c, size=(n, N))
            b = m - delta + pair + rng.normal(0.0, own_b, size=(n, N))
        else:
            c = rng.normal(m, sc, size=(n, N))
            b = rng.normal(m - delta, sb, size=(n, N))
        return c, b
    return gen


def two_state_gen(q_c, q_b, climb=(R, 0.10), stay=(0.04, 0.04)):
    """each continuation climbs (rise ~ N(climb)) with probability q, else stays (rise ~ N(stay))."""
    def gen(rng, n):
        def arm(q):
            up = rng.random((n, N)) < q
            return np.where(up, rng.normal(climb[0], climb[1], (n, N)), rng.normal(stay[0], stay[1], (n, N)))
        return arm(q_c), arm(q_b)
    return gen


def f3(x):
    return f"{x:.3f}"


# ------------------------------------------------------------------------------------------------ A exact facts
print("A. EXACT FACTS (enumeration over the C(8,4) = 70 relabelings)")
print(f"  attainable p <= 0.05: 1/70 = {1 / 70:.4f}, 2/70 = {2 / 70:.4f}, 3/70 = {3 / 70:.4f}; 4/70 = {4 / 70:.4f} is not")
print(f"  size under exchangeable continuous values: P(p <= 0.05) = 3/70 = {3 / 70:.4f} (the observed split's rank among"
      " the 70 sums is uniform)")


def exact_p_with_inf(ctrl, blind):
    """values as (infinite part, Fraction); the scorer's rule (score_bs.perm_test)."""
    vals = list(ctrl) + list(blind)
    obs = (sum(v[0] for v in ctrl), sum((v[1] for v in ctrl), Fraction(0)))
    cnt = 0
    for c in COMBOS:
        s = (sum(vals[i][0] for i in c), sum((vals[i][1] for i in c), Fraction(0)))
        cnt += s >= obs
    return Fraction(cnt, 70)


# the smallest filled p over many finite configurations with one / two MISSING values (fill against REDUCED)
rng = np.random.default_rng(1)
worst = {}
for case in ("one control missing", "one blind missing", "one missing in each arm", "two control missing"):
    best = Fraction(1)
    for _ in range(400):
        c = [(0, Fraction(int(v))) for v in rng.integers(1000, 2000, 4)]
        b = [(0, Fraction(int(v))) for v in rng.integers(0, 1000, 4)]       # complete separation: p = 1/70 if full
        if case == "one control missing":
            c[0] = (-1, Fraction(0))
        elif case == "one blind missing":
            b[0] = (1, Fraction(0))
        elif case == "one missing in each arm":
            c[0], b[0] = (-1, Fraction(0)), (1, Fraction(0))
        else:
            c[0], c[1] = (-1, Fraction(0)), (-1, Fraction(0))
        best = min(best, exact_p_with_inf(c, b))
    worst[case] = best
    print(f"  fill: {case}: the smallest filled p over 400 completely separated sets = {best} = {float(best):.4f}")
print("  (so one MISSING continuation, in either arm, makes REDUCED unreadable, as in BS)")

# ------------------------------------------------------------------------------------------------ B test alone
print("\nB. THE TEST ALONE (no gate), normal location model, equal SDs, independent continuations "
      f"({DRAWS:,} draws per cell; MC SE <= {0.5 / math.sqrt(DRAWS):.4f})")
print("  d = delta / sigma:  " + "".join(f"{d:>7}" for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)))
row = []
for k, d in enumerate((0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)):
    _, pr = lines(normal_gen(0.0, 1.0, d), (-1e9,), seed=100 + k)
    row.append(pr)
print("  P(p <= 0.05):       " + "".join(f"{x:>7.3f}" for x in row))
_, pr0 = lines(normal_gen(0.0, 1.0, 0.0), (-1e9,), seed=99)
print(f"  check, d = 0: {pr0:.4f} (exact 3/70 = {3 / 70:.4f})")

# ------------------------------------------------------------------------------------------------ C gate alone
print("\nC. THE GATE ALONE: P(informative) = P(median of 4 control rises >= gate); control rise ~ N(m, sigma^2)")
ms = (0.05, 0.10, 0.145, 0.20, 0.29, 0.434, R)
for g in GATES:
    print(f"  gate {g:.2f}:  m \\ sigma " + "".join(f"{s:>8}" for s in SIGMAS))
    for i, m in enumerate(ms):
        vals = []
        for j, s in enumerate(SIGMAS):
            res, _ = lines(normal_gen(m, s, 0.0), (g,), draws=60_000, seed=1000 + 100 * i + j)
            vals.append(1.0 - res[g][0])
        print(f"            m {m:>6.3f}  " + "".join(f"{x:>8.3f}" for x in vals))

# ------------------------------------------------------------------------------------------------ D lines
print(f"\nD. THE THREE LINES AT THE CHOSEN GATE {GATE:.2f}: control rise ~ N(m, sigma^2), blind ~ N(m - f*m, sigma^2), "
      "independent")
print("   cell = P(1 NOT INFORMATIVE) / P(2 REDUCED) / P(3 NO REDUCTION DETECTED)")
for i, (mlab, m) in enumerate((("R (parent)", R), ("3R/4", 0.75 * R), ("R/2", 0.5 * R), ("R/4", 0.25 * R))):
    print(f"  m = {mlab} = {m:.3f}")
    print("    f \\ sigma   " + "".join(f"{s:>20}" for s in SIGMAS))
    for k, f in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        cells = []
        for j, s in enumerate(SIGMAS):
            res, _ = lines(normal_gen(m, s, f * m), (GATE,), seed=20000 + 1000 * i + 100 * k + j)
            p1, p2, p3, _ = res[GATE]
            cells.append(f"{p1:.3f}/{p2:.3f}/{p3:.3f}")
        print(f"    f {f:<5}     " + "".join(f"{c:>20}" for c in cells))

# ------------------------------------------------------------------------------------------------ E conditional size
print("\nE. NO EFFECT (f = 0): P(REDUCED) overall and P(REDUCED | informative), per gate; control and blind ~ N(m, sigma^2)")
for i, m in enumerate((0.10, 0.15, 0.20, 0.25, 0.29, 0.434, R)):
    for j, s in enumerate((0.10, 0.137, 0.20)):
        res, pr = lines(normal_gen(m, s, 0.0), GATES, seed=30000 + 100 * i + j)
        txt = "  ".join(f"g{g:.2f}: {res[g][1]:.3f} | {res[g][3]:.3f}" for g in GATES)
        print(f"  m {m:.3f} sigma {s:.3f}: (overall | given informative)  {txt}")

# ------------------------------------------------------------------------------------------------ F departures
print(f"\nF. DEPARTURES at the chosen gate {GATE:.2f} (P1 / P2 / P3)")
for j, s in enumerate((0.10, 0.137, 0.20, 0.34)):
    for k, f in enumerate((0.5, 1.0)):
        res, _ = lines(normal_gen(R, s, f * R, sb=0.05), (GATE,), seed=40000 + 10 * j + k)
        p1, p2, p3, _ = res[GATE]
        print(f"  blind SD 0.05 (BS's blind SD 0.029): m R, control sigma {s:.3f}, f {f}: {p1:.3f}/{p2:.3f}/{p3:.3f}")
for j, (s, tau) in enumerate(((0.137, 0.06), (0.137, 0.10), (0.20, 0.10), (0.20, 0.14))):
    for k, f in enumerate((0.0, 0.5, 1.0)):
        res, _ = lines(normal_gen(R, s, f * R, tau=tau), (GATE,), seed=41000 + 10 * j + k)
        p1, p2, p3, _ = res[GATE]
        print(f"  pair effect SD {tau:.2f} within total SD {s:.3f}: m R, f {f}: {p1:.3f}/{p2:.3f}/{p3:.3f}")

# ------------------------------------------------------------------------------------------------ G two-state
print("\nG. TWO-STATE ('chance') MODEL: each continuation climbs (rise ~ N(R, 0.10^2)) with probability q, else stays "
      "(rise ~ N(0.04, 0.04^2): BS's non-rising runs 0.001-0.068); blind climbs with probability q (1 - f)")
print("   per gate: P1 / P2 / P3")
for i, q in enumerate((1.0, 0.9, 0.75, 0.5, 0.25)):
    for k, f in enumerate((0.0, 0.5, 1.0)):
        res, _ = lines(two_state_gen(q, q * (1 - f)), GATES, seed=50000 + 10 * i + k)
        txt = "  ".join(f"g{g:.2f} {res[g][0]:.3f}/{res[g][1]:.3f}/{res[g][2]:.3f}" for g in GATES)
        print(f"  q {q:<4} f {f:<4}: {txt}")

# ------------------------------------------------------------------------------------------------ H exact enumeration
print("\nH. EXACT ENUMERATION on BS's own rises (each continuation's rise drawn uniformly from an arm's 4 values; all "
      "4^4 x 4^4 = 65,536 equally likely draws; values in integer units of 1e-9, ties exact and against REDUCED)")


def exact_lines(cvals, bvals, gates):
    ci = [round(v * 1e9) for v in cvals]
    bi = [round(v * 1e9) for v in bvals]
    tally = {g: [0, 0, 0] for g in gates}
    red_total = 0
    ctups = list(itertools.product(ci, repeat=N))
    btups = list(itertools.product(bi, repeat=N))
    ca = np.array(ctups, dtype=np.int64)
    ba = np.array(btups, dtype=np.int64)
    subi = SUB.astype(np.int64)
    cs = np.sort(ca, axis=1)
    med2 = cs[:, 1] + cs[:, 2]                               # 2 x median, integer
    for bi_row in ba:
        x = np.concatenate([ca, np.broadcast_to(bi_row, ca.shape)], axis=1)
        sums = x @ subi.T
        red = (sums >= sums[:, :1]).sum(axis=1) <= 3
        red_total += int(red.sum())
        for g in gates:
            inf = med2 >= round(2 * g * 1e9)
            tally[g][0] += int((~inf).sum())
            tally[g][1] += int((inf & red).sum())
            tally[g][2] += int((inf & ~red).sum())
    tot = len(ctups) * len(btups)
    return {g: tuple(Fraction(t, tot) for t in tally[g]) for g in gates}, Fraction(red_total, tot)


scale = R / BS_R
print(f"  scale = R / BS's parent rise = {R:.6f} / {BS_R:.6f} = {scale:.4f}")
cases = (
    ("H1 scaled, no effect: both arms from BS control rises x scale", [v * scale for v in BS_CONTROL],
     [v * scale for v in BS_CONTROL]),
    ("H2 scaled, BS's arms: control from BS control x scale, blind from BS blind x scale",
     [v * scale for v in BS_CONTROL], [v * scale for v in BS_BLIND]),
    ("H3 unscaled (BS2's rise only as large as BS's), no effect: both arms from BS control rises", list(BS_CONTROL),
     list(BS_CONTROL)),
    ("H4 unscaled, BS's arms: control from BS control, blind from BS blind", list(BS_CONTROL), list(BS_BLIND)),
)
for label, cv, bv in cases:
    res, red = exact_lines(cv, bv, GATES)
    print(f"  {label}")
    print(f"     control values {[round(v, 4) for v in cv]}; blind values {[round(v, 4) for v in bv]}")
    print(f"     P(p <= 0.05) ignoring the gate = {red} = {float(red):.4f}")
    for g in GATES:
        p1, p2, p3 = res[g]
        print(f"     gate {g:.2f}: P1 {float(p1):.4f}  P2 {float(p2):.4f}  P3 {float(p3):.4f}   (exact: {p1}, {p2}, {p3})")
print("\nEND")
