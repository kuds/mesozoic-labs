"""Power and size of PS's pre-registered reading (4 control vs 4 critic_phase from r10 stance6 s42's 2,097,152 state
to E = 3,538,944): the exact one-sided unpaired permutation test on the difference of arm means (critic_phase lower;
REDUCED when p <= 0.05, i.e. the observed control-sum among the 3 highest of the 70, ties against) and the
informativeness gate (the median of the control primaries read, midpoint of the two middle values for 4, minus
B_S >= 0.20), both BS2's unchanged; and the four pre-registered secondaries.

Every primary value is a RISE from B_S (the test and the gate are translation-invariant).  Deterministic: numpy
default_rng with fixed seeds.  Sections:
  A  exact facts (as BS2): attainable p, size, the fill bounds, the multiplicity of 3 + 1 secondaries
  B  the gate with BS2's OBSERVED control arm (new input): normal model at BS2's control mean / SD, and EXACT
     enumeration resampling BS2's four control rises
  C  the three lines, normal model, control at BS2's observed mean rise and at the parent's, control SD 0.10 / 0.166
     (BS2's) / 0.25, phase arm SD equal to control's or BS2's blind SD 0.036, reduction f of the control rise
  D  EXACT enumeration on BS2's own arms: control from BS2's control rises; critic_phase from BS2's blind rises (a
     phase arm that works like blinding), from (1 - f) x BS2's control rises (a partial effect), and from BS2's
     control rises (no effect)
  E  no location effect, unequal spread (BS2 S-3): P(REDUCED) with equal means and SDs 0.166 / 0.036
  F  two-state model (each continuation climbs like BS2's controls with probability q, else stays like BS2's blind)
  G  the secondaries: SEC-MECH (the critic's on-policy tail slope at E) with BS2's control slopes; SEC-CMD and
     SEC-MARGIN with BS2's values; SEC-TAIL2 with BS2's tail_2 values (control vs blind, post hoc)
Inputs [measured]: B_S and R from BS2 criteria.txt (lines 180-192; BS2 criteria_work/parent_window.out); BS2's
primaries, SEC-CMD and SEC-MARGIN values from BS2 score/score.json (classes, secondaries); BS2's on-policy critic slopes
at E from BS2 score/score.txt lines 93, 105, 117, 129, 141, 153, 165, 177; BS2's tail_2 rises from
other_actuators_bs2.out (this folder).  The normal and two-state models and every SD are ASSUMPTIONS [inferred].
Usage: cd /tmp && nice -n 10 PY -I -B power_ps.py > power_ps.out 2> power_ps.err
"""
import itertools
import math
from fractions import Fraction

import numpy as np

B_S = 0.1265954738482833
R = 0.5787834012880921                         # the parent's rise over the window
BS2_CONTROL_PRIMARY = (0.6700147166848183, 0.7922730892896652, 0.5108136124908924, 0.41937798261642456)
BS2_BLIND_PRIMARY = (0.14282535295933485, 0.12485168222337961, 0.13809500634670258, 0.06516436021775007)
C_RISE = tuple(v - B_S for v in BS2_CONTROL_PRIMARY)
B_RISE = tuple(v - B_S for v in BS2_BLIND_PRIMARY)
BS2_C_MEAN = float(np.mean(C_RISE))
BS2_C_SD = float(np.std(C_RISE, ddof=1))
BS2_B_SD = float(np.std(B_RISE, ddof=1))
# BS2 secondaries (score.json), seeds 2501, 2511, 2521, 2531
CMD_C = (0.8840056845359504, 0.9553721153587104, 0.5973017146326602, 0.4753267549201846)
CMD_B = (0.18058780032768845, 0.12518754258332773, 0.14573265082156286, 0.06805177289620042)
MARG_C = (-0.02957710278034209, -0.05954820799827576, 0.4039530179500579, 0.35182105660438534)
MARG_B = (0.9826406861543655, 1.1091616152524948, 1.3006147063970566, 1.193611593425274)
# on-policy critic slope dV/dq3 per degree at E (score.txt; control effective = raw; blind raw, the unused value MLP)
SLOPE_C = (0.06177, 0.05324, 0.02260, 0.04147)
SLOPE_B_RAW = (0.03062, 0.01557, 0.00331, 0.03012)
# tail_2 on-policy rise over the window (other_actuators_bs2.out; post hoc)
T2_C = (0.0236, -0.0073, -0.0320, -0.0112)
T2_B = (0.0105, 0.1204, 0.0171, 0.0158)
GATE = 0.20
N = 4
COMBOS = list(itertools.combinations(range(2 * N), N))
SUB = np.array([[1.0 if i in c else 0.0 for i in range(2 * N)] for c in COMBOS])      # 70 x 8
DRAWS = 100_000
CHUNK = 50_000


def reduced_mask(ctrl, other):
    """True where the control arm's sum ranks among the 3 highest of the 70 (ties count against)."""
    x = np.concatenate([ctrl, other], axis=1)
    sums = x @ SUB.T
    return (sums >= sums[:, :1]).sum(axis=1) <= 3


def median4(ctrl):
    s = np.sort(ctrl, axis=1)
    return 0.5 * (s[:, 1] + s[:, 2])


def lines(gen, draws=DRAWS, seed=0, gate=GATE):
    rng = np.random.default_rng(seed)
    p1 = p2 = p3 = red_all = done = 0
    while done < draws:
        n = min(CHUNK, draws - done)
        c, b = gen(rng, n)
        red = reduced_mask(c, b)
        inf = median4(c) >= gate
        red_all += int(red.sum())
        p1 += int((~inf).sum())
        p2 += int((inf & red).sum())
        p3 += int((inf & ~red).sum())
        done += n
    return p1 / draws, p2 / draws, p3 / draws, red_all / draws


def normal_gen(m, sc, delta, sp=None):
    sp = sc if sp is None else sp

    def gen(rng, n):
        return rng.normal(m, sc, size=(n, N)), rng.normal(m - delta, sp, size=(n, N))
    return gen


def exact_perm_p(hi, lo):
    """exact one-sided p that the 'hi' arm's mean is higher (ties against): share of the 70 relabelings whose 'hi' sum
    is >= the observed one, in exact rationals."""
    x = [Fraction(v) for v in hi] + [Fraction(v) for v in lo]
    obs = sum(x[:4])
    return Fraction(sum(1 for c in COMBOS if sum(x[i] for i in c) >= obs), 70)


def exact_lines(cvals, pvals, gate=GATE):
    """each continuation's rise drawn uniformly from an arm's 4 values: 4^4 x 4^4 equally likely draws; integer
    units of 1e-9; ties exact and against REDUCED."""
    ci = [round(v * 1e9) for v in cvals]
    pi = [round(v * 1e9) for v in pvals]
    ca = np.array(list(itertools.product(ci, repeat=N)), dtype=np.int64)
    pa = np.array(list(itertools.product(pi, repeat=N)), dtype=np.int64)
    subi = SUB.astype(np.int64)
    cs = np.sort(ca, axis=1)
    inf = (cs[:, 1] + cs[:, 2]) >= round(2 * gate * 1e9)
    t1 = t2 = t3 = 0
    for row in pa:
        x = np.concatenate([ca, np.broadcast_to(row, ca.shape)], axis=1)
        sums = x @ subi.T
        red = (sums >= sums[:, :1]).sum(axis=1) <= 3
        t1 += int((~inf).sum())
        t2 += int((inf & red).sum())
        t3 += int((inf & ~red).sum())
    tot = len(ca) * len(pa)
    return Fraction(t1, tot), Fraction(t2, tot), Fraction(t3, tot)


# ------------------------------------------------------------------------------------------------ A
print("A. EXACT FACTS (as BS2; enumeration over the C(8,4) = 70 relabelings)")
print(f"  attainable p <= 0.05: 1/70 = {1 / 70:.4f}, 2/70 = {2 / 70:.4f}, 3/70 = {3 / 70:.4f}; 4/70 = {4 / 70:.4f} is not")
print("  size under exchangeable continuous values: 3/70 = 0.0429; the fill (BS2 power_bs2.out A, same rule): one "
      "MISSING -> filled p >= 36/70, one per arm >= 51/70, two controls >= 56/70")
print(f"  multiplicity: 3 efficacy secondaries (SEC-CMD, SEC-MARGIN, SEC-MECH) at 3/70 each: P(any SUPPORTED | global "
      f"null, exchangeable) <= 9/70 = {9 / 70:.4f} (Bonferroni bound); Bonferroni level 0.05/3 = {0.05 / 3:.4f} >= 1/70 "
      f"= {1 / 70:.4f}, so only p = 1/70 survives")
print(f"  with the guard secondary SEC-TAIL2 too (4 at 3/70): <= 12/70 = {12 / 70:.4f}; 0.05/4 = {0.05 / 4:.4f} < 1/70: "
      "no p could survive a Bonferroni over 4, which is why the guard is kept out of the efficacy family")

# ------------------------------------------------------------------------------------------------ B
print("\nB. THE GATE WITH BS2's OBSERVED CONTROL ARM (BS2 score.json: rises "
      + ", ".join(f"{v:+.5f}" for v in C_RISE) + f"; mean {BS2_C_MEAN:.4f}, SD {BS2_C_SD:.4f})")
for m, s, lab in ((BS2_C_MEAN, BS2_C_SD, "BS2's mean and SD"), (BS2_C_MEAN, 0.25, "BS2's mean, SD 0.25"),
                  (0.5 * BS2_C_MEAN, BS2_C_SD, "half BS2's mean, BS2's SD"), (R, BS2_C_SD, "the parent's rise, BS2's SD")):
    p1, _, _, _ = lines(normal_gen(m, s, 0.0), seed=11 + int(1000 * m) + int(100 * s))
    print(f"  normal model, {lab} (m {m:.4f}, SD {s:.4f}): P(informative) = {1 - p1:.4f}")
e1, _, _ = exact_lines(C_RISE, C_RISE)
print(f"  exact: resampling BS2's four control rises (4^4 draws): P(not informative) = {e1} (every rise >= 0.29)")
half = tuple(0.5 * v for v in C_RISE)
e1h, _, _ = exact_lines(half, half)
print(f"  exact: resampling HALF of BS2's control rises ({', '.join(f'{v:.3f}' for v in half)}): P(not informative) "
      f"= {e1h} = {float(e1h):.4f}")

# ------------------------------------------------------------------------------------------------ C
print("\nC. THE THREE LINES AT THE GATE 0.20, normal model: control rise ~ N(m, sc^2), critic_phase ~ N((1 - f) m, sp^2)")
print("   cell = P(1 NOT INFORMATIVE) / P(2 REDUCED) / P(3 NO REDUCTION DETECTED)")
for i, (mlab, m) in enumerate(((f"BS2 control mean {BS2_C_MEAN:.3f}", BS2_C_MEAN), (f"parent R {R:.3f}", R))):
    for j, sc in enumerate((0.10, BS2_C_SD, 0.25)):
        for k, sp_lab in enumerate(("equal", "0.036")):
            cells = []
            for q, f in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
                sp = sc if sp_lab == "equal" else BS2_B_SD
                p1, p2, p3, _ = lines(normal_gen(m, sc, f * m, sp), seed=20000 + 1000 * i + 100 * j + 10 * k + q)
                cells.append(f"f{f:.2f} {p1:.3f}/{p2:.3f}/{p3:.3f}")
            print(f"  m {mlab}, control SD {sc:.3f}, phase SD {sp_lab:<5}: " + "  ".join(cells))

# ------------------------------------------------------------------------------------------------ D
print("\nD. EXACT ENUMERATION ON BS2's OWN ARMS (each continuation's rise drawn uniformly from 4 values; 65,536 draws)")
cases = [("D1 phase arm like BS2's blinded arm", C_RISE, B_RISE),
         ("D2 phase arm = 0.25 x BS2's control rises (f 0.75)", C_RISE, tuple(0.25 * v for v in C_RISE)),
         ("D3 phase arm = 0.50 x BS2's control rises (f 0.50)", C_RISE, tuple(0.50 * v for v in C_RISE)),
         ("D4 phase arm = 0.75 x BS2's control rises (f 0.25)", C_RISE, tuple(0.75 * v for v in C_RISE)),
         ("D5 no effect: both arms from BS2's control rises", C_RISE, C_RISE)]
for lab, cv, pv in cases:
    p1, p2, p3 = exact_lines(cv, pv)
    print(f"  {lab}: P1 {float(p1):.4f}  P2 REDUCED {float(p2):.4f}  P3 {float(p3):.4f}   (exact {p1}, {p2}, {p3})")

# ------------------------------------------------------------------------------------------------ E
print("\nE. NO LOCATION EFFECT, UNEQUAL SPREAD (BS2 S-3): equal means, control SD 0.166, phase SD 0.036 (BS2's blind)")
for i, m in enumerate((BS2_C_MEAN, R, 0.29)):
    p1, p2, p3, pr = lines(normal_gen(m, BS2_C_SD, 0.0, BS2_B_SD), draws=200_000, seed=30000 + i)
    print(f"  m {m:.3f}: P(p <= 0.05) {pr:.4f}; P(REDUCED) at the gate {p2:.4f}; given informative "
          f"{p2 / (p2 + p3):.4f}")
p1, p2, p3, pr = lines(normal_gen(BS2_C_MEAN, BS2_C_SD, 0.0), draws=200_000, seed=30010)
print(f"  check, equal SDs: P(p <= 0.05) {pr:.4f} (exact 3/70 = {3 / 70:.4f})")

# ------------------------------------------------------------------------------------------------ F
print("\nF. TWO-STATE MODEL: each continuation climbs (rise ~ N(BS2 control mean, BS2 control SD^2)) with probability "
      "q, else stays (rise ~ N(0, 0.04^2), BS2's blinded rises -0.061..+0.016); critic_phase climbs with q (1 - f)")


def two_state_gen(qc, qp):
    def gen(rng, n):
        def arm(q):
            up = rng.random((n, N)) < q
            return np.where(up, rng.normal(BS2_C_MEAN, BS2_C_SD, (n, N)), rng.normal(0.0, 0.04, (n, N)))
        return arm(qc), arm(qp)
    return gen


for i, q in enumerate((1.0, 0.9, 0.75, 0.5)):
    cells = []
    for k, f in enumerate((0.0, 0.5, 0.75, 1.0)):
        p1, p2, p3, _ = lines(two_state_gen(q, q * (1 - f)), seed=50000 + 10 * i + k)
        cells.append(f"f{f:.2f} {p1:.3f}/{p2:.3f}/{p3:.3f}")
    print(f"  q {q:<4}: " + "  ".join(cells))

# ------------------------------------------------------------------------------------------------ G
print("\nG. THE SECONDARIES")
sm, ss = float(np.mean(SLOPE_C)), float(np.std(SLOPE_C, ddof=1))
print(f"  SEC-MECH input: BS2's control on-policy critic slopes at E {SLOPE_C} (mean {sm:.4f}, SD {ss:.4f} per deg); "
      f"BS2's blinded arm's RAW (unused) value-MLP slopes {SLOPE_B_RAW} (mean {np.mean(SLOPE_B_RAW):.4f})")
for k, (lab, mp, sp) in enumerate((("phase arm slope ~ N(0, SD_c)", 0.0, ss),
                                   ("phase arm slope ~ N(0, 0.010)", 0.0, 0.010),
                                   ("phase arm slope ~ N(mean_c/2, SD_c)", sm / 2, ss),
                                   ("phase arm slope ~ N(0.0204, 0.0140) (like BS2's blind raw MLP)",
                                    float(np.mean(SLOPE_B_RAW)), float(np.std(SLOPE_B_RAW, ddof=1))))):
    _, _, _, pr = lines(normal_gen(sm, ss, sm - mp, sp), seed=60000 + k, gate=-1e9)
    print(f"    {lab}: P(SUPPORTED) = {pr:.3f}")
p1, p2, p3 = exact_lines(SLOPE_C, tuple(0.0 * v for v in SLOPE_C), gate=-1e9)
print(f"    exact, control from BS2's slopes, phase arm 0 in all four: P(SUPPORTED) = {float(p2):.4f}")
p1, p2, p3 = exact_lines(SLOPE_C, SLOPE_B_RAW, gate=-1e9)
print(f"    exact, control from BS2's slopes, phase arm from BS2's blind raw slopes: P(SUPPORTED) = {float(p2):.4f}")
print(f"  SEC-CMD with BS2's own values (control higher): exact p = {exact_perm_p(CMD_C, CMD_B)} (BS2 read 1/70)")
print(f"  SEC-MARGIN with BS2's own values (blind higher): exact p = {exact_perm_p(MARG_B, MARG_C)} (BS2 read 1/70)")
print(f"  SEC-TAIL2 with BS2's own tail_2 rises (POST HOC; blind HIGHER): exact p = {exact_perm_p(T2_B, T2_C)}")
p1, p2, p3 = exact_lines(T2_B, T2_C, gate=-1e9)
print(f"    exact resampling, 'phase' arm from BS2's blind tail_2 rises, control from BS2's control: P(SUPPORTED) = "
      f"{float(p2):.4f}")
one_mig = (T2_C[0], T2_C[1], T2_C[2], 0.12)
print(f"    a single migrating continuation (+0.12) among control-like values {one_mig}: exact p vs BS2's controls = "
      f"{exact_perm_p(one_mig, T2_C)}")
print("\nEND")
