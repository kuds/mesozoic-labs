"""Design reviewer: independent power / size computations for BS2's reading (not derived from power_bs2.py's code).

The reading: 4 control and 4 blind primaries (as rises from B_S); line 1 if median(control) < 0.20 (midpoint of the two
middle values); else line 2 (REDUCED) if the exact one-sided permutation p of the control-arm sum <= 0.05 (observed split
among the top 3 of the 70 sums, ties counted against), else line 3.
Sections:
  P1 reproduce designer cells (normal model; D, E, G) with other seeds and another code path
  P2 size under H0 with UNEQUAL arm SDs (equal means): unconditional and given informative
  P3 two-state ('chance') NULL: P(REDUCED | informative) and P(REDUCED | control median rise >= 0.29)
  P4 timing model: every continuation climbs like the parent but with a random delay (no effect / an effect)
  P5 exact enumeration on BS's own rises (H1 / H3 of the designer), own integer code
Usage: PY -I -B rd_power.py > rd_power.out
"""
import itertools
from fractions import Fraction

import numpy as np

R = 0.5787834012880921          # the parent's rise over the window [measured, work/rd_parent.out]
GATE = 0.20
IDX = np.array(list(itertools.combinations(range(8), 4)))      # 70 x 4
OBS_ROW = int(np.nonzero((IDX == np.array([0, 1, 2, 3])).all(axis=1))[0][0])
PARENT_T = np.array([0.0, 0.02855, 0.04199, 0.06888, 0.06359, 0.11478, 0.16517, 0.20113, 0.30514, 0.48692, 0.56476,
                     0.57878])                                     # rise at S, then the 11 evaluation steps (rd_parent)


def reading(c, b, gate=GATE):
    """c, b: (n, 4) rises.  Returns (informative, reduced) boolean arrays."""
    x = np.concatenate([c, b], axis=1)
    sums = x[:, IDX].sum(axis=2)                                   # (n, 70)
    obs = sums[:, OBS_ROW:OBS_ROW + 1]
    count = (sums >= obs).sum(axis=1)
    reduced = count <= 3
    cs = np.sort(c, axis=1)
    med = 0.5 * (cs[:, 1] + cs[:, 2])
    return med >= gate, reduced, med


def run(gen, n=200_000, seed=0, gate=GATE, chunk=25_000):
    rng = np.random.default_rng(seed)
    parts = []
    done = 0
    while done < n:
        k = min(chunk, n - done)
        c, b = gen(rng, k)
        parts.append(reading(c, b, gate))
        done += k
    inf = np.concatenate([p[0] for p in parts])
    red = np.concatenate([p[1] for p in parts])
    med = np.concatenate([p[2] for p in parts])
    p1 = 1 - inf.mean()
    p2 = (inf & red).mean()
    p3 = (inf & ~red).mean()
    cond = p2 / (p2 + p3) if p2 + p3 > 0 else float("nan")
    hi = med >= 0.29
    cond_hi = (hi & red).sum() / max(hi.sum(), 1)
    return p1, p2, p3, cond, red.mean(), cond_hi


def normal(mc, sc, mb, sb):
    return lambda rng, n: (rng.normal(mc, sc, (n, 4)), rng.normal(mb, sb, (n, 4)))


def two_state(qc, qb, climb=(R, 0.10), stay=(0.04, 0.04)):
    def gen(rng, n):
        def arm(q):
            up = rng.random((n, 4)) < q
            return np.where(up, rng.normal(*climb, (n, 4)), rng.normal(*stay, (n, 4)))
        return arm(qc), arm(qb)
    return gen


def delay(mean_delay, sd_delay, shrink_blind=0.0, noise=0.03):
    """each continuation follows the parent's rise curve shifted later by d ~ N(mean, sd) evaluations (>= 0),
    read at E (index 11), plus noise; the blind arm's curve is scaled by (1 - shrink_blind)."""
    steps = np.arange(12)

    def at(d):
        t = np.clip(11 - d, 0, 11)
        return np.interp(t, steps, PARENT_T)

    def gen(rng, n):
        dc = np.clip(rng.normal(mean_delay, sd_delay, (n, 4)), 0, None)
        db = np.clip(rng.normal(mean_delay, sd_delay, (n, 4)), 0, None)
        c = at(dc) + rng.normal(0, noise, (n, 4))
        b = (1 - shrink_blind) * at(db) + rng.normal(0, noise, (n, 4))
        return c, b
    return gen


def f3(t):
    return "/".join(f"{x:.3f}" for x in t)


print("P1 REPRODUCTION of designer cells (P1/P2/P3; designer power_bs2.out D, G; E as P(REDUCED | informative))")
for sig in (0.05, 0.10, 0.137, 0.20, 0.34):
    for f in (0.0, 0.5, 1.0):
        r = run(normal(R, sig, R - f * R, sig), seed=int(sig * 1000) + int(f * 10))
        print(f"  D m=R sigma {sig:<5} f {f}: {f3(r[:3])}")
for sig in (0.10, 0.137, 0.20, 0.34):
    r = run(normal(0.5 * R, sig, 0.5 * R - 0.5 * R, sig), seed=900 + int(sig * 1000))
    print(f"  D m=R/2 sigma {sig:<5} f 1: {f3(r[:3])}")
for m in (0.10, 0.15, 0.20, 0.25, 0.29):
    for sig in (0.10, 0.137, 0.20):
        r = run(normal(m, sig, m, sig), seed=2000 + int(m * 100) + int(sig * 1000))
        print(f"  E no effect m {m:.3f} sigma {sig:.3f}: P(REDUCED) {r[4]:.4f}  P(REDUCED | informative) {r[3]:.3f}")
for q in (1.0, 0.9, 0.75, 0.5, 0.25):
    for f in (0.0, 0.5, 1.0):
        r = run(two_state(q, q * (1 - f)), seed=3000 + int(q * 100) + int(f * 10))
        print(f"  G q {q:<4} f {f}: {f3(r[:3])}")

print("\nP2 SIZE UNDER H0 WITH UNEQUAL ARM SDs (equal means m; control SD sc, blind SD sb)")
print("   columns: P(REDUCED) overall, P(REDUCED | informative), P1")
for m in (R, 0.434, 0.29):
    for sc, sb in ((0.137, 0.03), (0.20, 0.05), (0.34, 0.05), (0.34, 0.10), (0.05, 0.20), (0.03, 0.137), (0.10, 0.34)):
        r = run(normal(m, sc, m, sb), n=400_000, seed=4000 + int(m * 1000) + int(sc * 100) + int(sb * 1000))
        print(f"  m {m:.3f} sc {sc:<5} sb {sb:<5}: {r[4]:.4f}  {r[3]:.4f}  P1 {r[0]:.3f}")

print("\nP3 TWO-STATE NULL (both arms climb with the same q; climb ~ N(R, 0.10^2), stay ~ N(0.04, 0.04^2))")
print("   columns: P1, P(REDUCED) overall, P(REDUCED | informative), P(REDUCED | control median rise >= 0.29)")
for q in (0.9, 0.75, 0.6, 0.5, 0.4, 0.25):
    r = run(two_state(q, q), n=400_000, seed=5000 + int(q * 100))
    print(f"  q {q:<4}: P1 {r[0]:.3f}  {r[4]:.4f}  {r[3]:.4f}  {r[5]:.4f}")
print("  same with the stayers at the parent's early drift (stay ~ N(0.07, 0.04^2)) and climbers N(R, 0.15^2):")
for q in (0.75, 0.5, 0.25):
    r = run(two_state(q, q, climb=(R, 0.15), stay=(0.07, 0.04)), n=400_000, seed=5500 + int(q * 100))
    print(f"  q {q:<4}: P1 {r[0]:.3f}  {r[4]:.4f}  {r[3]:.4f}  {r[5]:.4f}")

print("\nP4 TIMING MODEL (each continuation = the parent's rise curve delayed by d evaluations, d ~ N(mu, sd) >= 0, "
      "noise 0.03; read at E)")
print("   columns: P1 / P2 / P3, P(REDUCED | informative)")
for mu_d, sd_d in ((0, 1), (1, 1), (2, 1.5), (3, 2)):
    for shrink in (0.0, 0.5, 1.0):
        r = run(delay(mu_d, sd_d, shrink), seed=6000 + mu_d * 10 + int(shrink * 3))
        print(f"  delay mean {mu_d} sd {sd_d} blind shrink {shrink}: {f3(r[:3])}  cond {r[3]:.3f}")

print("\nP5 EXACT ENUMERATION on BS's rises (designer H1 / H3), own integer code (units 1e-12)")
BS_C = (0.00055768666788935661, 0.057248691737186164, 0.12643680511973798, 0.31430359347723424)
SCALE = R / 0.23113293549977243


def exact(cv, bv, gate):
    ci = [round(v * 1e12) for v in cv]
    bi = [round(v * 1e12) for v in bv]
    g2 = round(2 * gate * 1e12)
    n1 = n2 = n3 = 0
    ctu = np.array(list(itertools.product(ci, repeat=4)), dtype=np.int64)
    btu = np.array(list(itertools.product(bi, repeat=4)), dtype=np.int64)
    cs = np.sort(ctu, axis=1)
    inf = (cs[:, 1] + cs[:, 2]) >= g2
    for brow in btu:
        x = np.concatenate([ctu, np.broadcast_to(brow, ctu.shape)], axis=1)
        sums = x[:, IDX].sum(axis=2)
        red = (sums >= sums[:, OBS_ROW:OBS_ROW + 1]).sum(axis=1) <= 3
        n1 += int((~inf).sum())
        n2 += int((inf & red).sum())
        n3 += int((inf & ~red).sum())
    tot = len(ctu) * len(btu)
    return Fraction(n1, tot), Fraction(n2, tot), Fraction(n3, tot)


for label, cv in (("H1 scaled x%.4f" % SCALE, [v * SCALE for v in BS_C]), ("H3 unscaled", list(BS_C))):
    for gate in (0.10, 0.15, 0.20, 0.25):
        p1, p2, p3 = exact(cv, cv, gate)
        print(f"  {label} no effect, gate {gate:.2f}: P1 {p1} ({float(p1):.4f})  P2 {p2} ({float(p2):.4f})  "
              f"P3 {float(p3):.4f}  P(REDUCED | informative) {float(p2 / (p2 + p3)) if p2 + p3 else float('nan'):.4f}")
print("END")
