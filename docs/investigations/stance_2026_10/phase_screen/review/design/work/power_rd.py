"""Independent power check for PS (design review). Own code; not derived from power_ps.py.
Model: 4 control + 4 phase primaries (rises from B_S); REDUCED iff gate (control median rise >= 0.20, midpoint of the
two middle values) and exact one-sided permutation p <= 0.05 (control sum rank among the 70 4-subsets, ties against).
"""
import itertools
import numpy as np

SUBS = np.array([[1 if i in c else 0 for i in range(8)] for c in itertools.combinations(range(8), 4)], dtype=float)
OBS = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
assert SUBS.shape == (70, 8)


def reading(ctrl, phase):
    """ctrl, phase: (N, 4). returns (p NOT INFORMATIVE, p REDUCED, p NO REDUCTION)"""
    x = np.concatenate([ctrl, phase], axis=1)
    sums = x @ SUBS.T                      # (N, 70)
    obs = (x @ OBS)[:, None]
    p = (sums >= obs - 1e-12).sum(axis=1) / 70.0
    srt = np.sort(ctrl, axis=1)
    med = (srt[:, 1] + srt[:, 2]) / 2
    inf = med >= 0.20
    red = inf & (p <= 0.05)
    return float((~inf).mean()), float(red.mean()), float((inf & ~red).mean()), p


def main():
    rng = np.random.default_rng(20261031)
    n = 200_000
    m, sc = 0.47152, 0.16583   # BS2 control rises mean, SD (from score.json values below)
    bs2c = np.array([0.54342, 0.66568, 0.38422, 0.29278])
    bs2b = None
    print("BS2 control rises", bs2c, "mean %.4f sd %.4f" % (bs2c.mean(), bs2c.std(ddof=1)))
    print("== C normal model (f = fraction of the control rise removed) ==")
    for mm, lab in ((m, "BS2 mean"), (0.57878, "parent rise")):
        for s in (0.10, sc, 0.25):
            for sp in ("eq", 0.036):
                row = []
                for f in (0, 0.25, 0.5, 0.75, 1.0):
                    spv = s if sp == "eq" else sp
                    c = rng.normal(mm, s, (n, 4))
                    ph = rng.normal((1 - f) * mm, spv, (n, 4))
                    a, b, cc, _ = reading(c, ph)
                    row.append("f%.2f %.3f/%.3f/%.3f" % (f, a, b, cc))
                print("%s sd %.3f phase sd %s: " % (lab, s, sp) + "  ".join(row))
    print("== E spread only (equal means) ==")
    for mm in (0.29, m, 0.57878):
        c = rng.normal(mm, sc, (n, 4)); ph = rng.normal(mm, 0.036, (n, 4))
        a, b, cc, p = reading(c, ph)
        print("m %.3f: P(p<=0.05) %.4f  P(REDUCED) %.4f  P(RED|inf) %.4f" % (mm, float((p <= 0.05).mean()), b, b / (1 - a)))
    c = rng.normal(m, sc, (n, 4)); ph = rng.normal(m, sc, (n, 4))
    a, b, cc, p = reading(c, ph)
    print("equal SDs: P(p<=0.05) %.4f (3/70 = %.4f)" % (float((p <= 0.05).mean()), 3 / 70))
    print("== F two-state: climb ~ N(m, sc) w.p. q, else N(0, 0.04); phase climbs w.p. q(1-f) ==")
    for q in (1.0, 0.9, 0.75, 0.5):
        row = []
        for f in (0, 0.5, 0.75, 1.0):
            def draw(qq):
                climb = rng.random((n, 4)) < qq
                return np.where(climb, rng.normal(m, sc, (n, 4)), rng.normal(0, 0.04, (n, 4)))
            a, b, cc, _ = reading(draw(q), draw(q * (1 - f)))
            row.append("f%.2f %.3f/%.3f/%.3f" % (f, a, b, cc))
        print("q %.2f: " % q + "  ".join(row))
    print("== D exact resampling on BS2's control rises (4^8 = 65,536 combos) ==")
    combos = np.array(list(itertools.product(range(4), repeat=8)))
    for g, lab in ((0.25, "0.25 x control (f .75)"), (0.5, "0.5 x control (f .5)"), (0.75, "0.75 x (f .25)"), (1.0, "no effect")):
        c = bs2c[combos[:, :4]]; ph = g * bs2c[combos[:, 4:]]
        a, b, cc, _ = reading(c, ph)
        print("%s: %.4f/%.4f/%.4f" % (lab, a, b, cc))
    print("== G SEC-MECH (no gate): control slope ~ N(0.0448, 0.0170), phase ~ ... ==")
    cs = np.array([0.06177, 0.05324, 0.0226, 0.04147])
    mc, sdc = cs.mean(), cs.std(ddof=1)
    for mp, sp, lab in ((0, sdc, "N(0, sdc)"), (0, 0.010, "N(0, .010)"), (mc / 2, sdc, "N(mc/2, sdc)"), (0.0214, sdc, "N(+0.0214 = slope at S, sdc)"), (0.0214, 0.005, "N(+0.0214, .005)")):
        c = rng.normal(mc, sdc, (n, 4)); ph = rng.normal(mp, sp, (n, 4))
        x = np.concatenate([c, ph], axis=1); sums = x @ SUBS.T; obs = (x @ OBS)[:, None]
        p = (sums >= obs - 1e-12).sum(axis=1) / 70.0
        print("%s: P(SUPPORTED) %.3f" % (lab, float((p <= 0.05).mean())))
    print("== line-3 clause audit: P(REDUCED) at f = 0.5 over all normal cells above; two-state at f 0.5 ==")


if __name__ == "__main__":
    main()
