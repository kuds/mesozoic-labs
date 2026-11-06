"""Fix lane: the scorer's recomputed_verdicts against pilot4_straight.py's own part_a / part_b / part_c on random
synthetic values (near the bars, with inf and short lists), JSON round-tripped as the tool writes them. Imports the two
tools from TOOLS (no repository module); writes nothing but its printout. Nothing trains.

usage: python -I -B cross_check_verdicts.py TOOLS N
"""
import json
import math
import random
import sys

sys.path.insert(0, sys.argv[1])
import pilot4_score as sc  # noqa: E402
import pilot4_straight as st  # noqa: E402

rng = random.Random(3042)
N = int(sys.argv[2])
bad = []
counts = {}


def near(bar):
    return rng.choice([bar, bar + 1e-9, bar - 1e-9, bar * rng.uniform(0, 2), math.inf])


for i in range(N):
    n_full = rng.choice([0, 1, 30, 39, 40])
    yaws = [rng.choice([near(5.0), -near(5.0), near(15.0), -near(15.0), rng.uniform(-3, 3)]) for _ in range(n_full)]
    yaws = [y for y in yaws if math.isfinite(y)]
    D = [rng.choice([11.5, 11.5 + 1e-9, 11.5 - 1e-9, rng.uniform(0, 23)]) for _ in range(len(yaws))]  # finite, as a full episode D
    W0 = [d + rng.choice([7.0, 7.0 + 1e-9, rng.uniform(0, 14)]) for d in D]
    readings, calm = [], rng.random() < 0.4  # calm: every r near or under the median bar (so (c) can pass)
    for _ in range(40):
        counting = rng.random() < rng.choice([0.6, 0.8, 1.0])
        r = (rng.choice([5.0, 5.0 - 1e-9, rng.uniform(0, 5), rng.uniform(0, 11.5)]) if calm else rng.choice([near(5.0), near(11.5), rng.uniform(0, 5), math.inf]))
        readings.append({"counting": counting, "reached_nudge": True, "passed_prey_at_650": counting,
                         "r_deg": r if counting else None, "r_reason": None, "angle_deg": 15.0, "share_removed": None})
    try:
        parts = {"a": st.part_a(yaws), "b": st.part_b(D, W0), "c": st.part_c(readings)}
    except ValueError:
        continue
    rec = json.loads(json.dumps(st.jsonable(parts)))
    want = {k: rec[k]["verdict"] for k in ("a", "b", "c")}
    got = sc.recomputed_verdicts(rec)
    key = tuple(want.values())
    counts[key] = counts.get(key, 0) + 1
    if got != want:
        bad.append({"i": i, "want": want, "got": got})
print(json.dumps({"trials": N, "disagreements": len(bad), "first": bad[:3],
                  "verdict_combinations_seen": {"/".join(k): v for k, v in sorted(counts.items())}}, indent=1))
sys.exit(1 if bad else 0)
