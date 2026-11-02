"""Re-review: the pinned one_test (tools/pc/pc_common.py, loaded read only) with the three OFF values missing
(-infinity, as NOT BLIND runs enter): k over random and edge TD values, each PHASE value finite or missing."""
import importlib.util
import itertools
import math
import random
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("pc_common_ro", "/home/user/work/pc_study/tools/pc/pc_common.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
rng = random.Random(20261102)
edge = [0.0, 5e-324, 1e-300, 0.1, 0.5, 1.0, 1e300, 1.7976931348623157e308]
ks, n = {}, 0
for trial in range(20000):
    ph = [rng.choice(edge) if rng.random() < 0.3 else rng.uniform(0, 2) for _ in range(3)]
    for mask in itertools.product((False, True), repeat=3):
        phase = [math.inf if m else v for v, m in zip(ph, mask)]
        r = P.one_test(phase, [-math.inf] * 3)
        ks[r["k"]] = ks.get(r["k"], 0) + 1
        n += 1
print(f"{n} calls (20000 random/edge PHASE TD triples x 8 PHASE-missing patterns), OFF all -infinity: k counts {ks}")
# for contrast: one OFF missing only
ks1 = {}
for trial in range(2000):
    ph = [rng.uniform(0, 2) for _ in range(3)]
    of = [rng.uniform(0, 2) for _ in range(2)] + [-math.inf]
    k = P.one_test(ph, of)["k"]
    ks1[k] = ks1.get(k, 0) + 1
print(f"contrast, one OFF missing, finite others: k values seen {sorted(ks1)}")
