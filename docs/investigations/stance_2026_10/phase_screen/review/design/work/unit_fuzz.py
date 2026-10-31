"""Direct fuzz of score_bs.decide / secondary_test / sign_flip_test with exact Fractions (gate boundary included)
against my_score.perm_p / median (independent)."""
import importlib.util, itertools, random, sys
from fractions import Fraction
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/work/phase_screen/review/design/work")
import my_score as MY
spec = importlib.util.spec_from_file_location("sbu", "/home/user/work/phase_screen/tools/score_bs.py")
SB = importlib.util.module_from_spec(spec); spec.loader.exec_module(SB)
rng = random.Random(7)
B = Fraction(SB.BASELINE_S)
assert B == MY.B_S
bad = 0
N = 20000
for it in range(N):
    classes, prim = [], {}
    for seed in SB.SEEDS:
        for arm in SB.ARM_ORDER:
            name = f"{arm}_{seed}"
            classes.append({"name": name, "arm": arm, "seed": seed, "cls": "COUNTED"})
            r = rng.random()
            if r < 0.1:
                prim[name] = None
            elif r < 0.4:   # the gate boundary, exactly
                prim[name] = B + Fraction(1, 5) + rng.choice([Fraction(0), Fraction(1, 10**9), -Fraction(1, 10**9)])
            else:
                prim[name] = B + Fraction(rng.randint(-20, 60), 100)
    d = SB.decide(classes, prim)
    c = [prim[f"control_{s}"] for s in SB.SEEDS]; p = [prim[f"critic_phase_{s}"] for s in SB.SEEDS]
    cnt, tot = MY.perm_p(c, p)
    cr = [x for x in c if x is not None]
    med = MY.median(cr)
    inf = med is not None and med - B >= Fraction(1, 5)
    line = 1 if not inf else (2 if Fraction(cnt, tot) <= Fraction(1, 20) else 3)
    if (d["filled"][1], d["filled"][2]) != (cnt, tot) or d["informative"] != inf or d["line"] != line:
        bad += 1
        if bad < 5:
            print("DIFF", it, d["filled"], (cnt, tot), d["informative"], inf, d["line"], line)
    diffs = [a - b for a, b in zip(c, p) if a is not None and b is not None]
    if diffs:
        obs = sum(diffs)
        my = sum(1 for s in itertools.product((1, -1), repeat=len(diffs)) if sum(x * y for x, y in zip(s, diffs)) >= obs)
        if SB.sign_flip_test(diffs)[1] != my:
            bad += 1
print(f"{N} cases, {bad} differences (decide filled p, gate incl. exact boundary, line, sign-flip)")
