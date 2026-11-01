"""Re-reviewer's independent check of pc_common.one_test (fixer I1): exact rational oracle vs the tool, random cases with
near ties (nextafter of comparators), equal values, +infinity entries; and the k == 1 <=> max(PC) < min(comp) identity."""
import importlib.util, itertools, math, random, sys
from fractions import Fraction
spec = importlib.util.spec_from_file_location("pc_common", "/home/user/work/pc_study/tools/pc/pc_common.py")
P = importlib.util.module_from_spec(spec); sys.modules["pc_common"] = P; spec.loader.exec_module(P)
COMP = [0.38822738704353094, 0.6353434591453841, 0.15433899719014801]
def oracle(pc, comp):
    v = pc + comp
    def s(ix):
        xs = [v[i] for i in ix]
        return math.inf if any(math.isinf(x) for x in xs) else sum(Fraction(x) for x in xs)
    obs = s((0, 1, 2))
    return sum(1 for ix in itertools.combinations(range(6), 3) if s(ix) <= obs)
rng = random.Random(20261101)
n = bad = sep_bad = 0
for _ in range(30000):
    pc = []
    for _j in range(3):
        r = rng.random()
        if r < 0.05: pc.append(math.inf)
        elif r < 0.45: pc.append(math.nextafter(rng.choice(COMP), rng.choice((0.0, 1.0))))
        elif r < 0.55: pc.append(rng.choice(COMP))
        else: pc.append(rng.uniform(0, 1))
    k = P.one_test(pc, COMP)["k"]; ko = oracle(pc, COMP); n += 1
    bad += k != ko
    sep_bad += (k == 1) != (max(pc) < min(COMP))
print(f"cases {n}; tool k != exact oracle k: {bad}; (k == 1) != (max PC < min comparator): {sep_bad}")
lo = min(COMP); near = math.nextafter(lo, 0.0)
print("one ulp below s80 thrice:", P.one_test([near, near, near], COMP)["p"], "; equal to s80:", P.one_test([lo, 0.0, 0.0], COMP)["p"])
print("NaN refused:", end=" ")
try:
    P.one_test([float("nan"), 0, 0], COMP); print("no")
except ValueError as e:
    print("yes", e)
