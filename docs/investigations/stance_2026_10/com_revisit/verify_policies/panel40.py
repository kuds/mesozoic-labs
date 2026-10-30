"""Verifier: the lens's proposed gate rule (P3 b / POL-10) on the full 40-seed gate panel 3042-3081.

python -I -B panel40.py recompute.json sim_panel/panel.jsonl
Seeds 3042-3071 from the lens's tracks (my re-reduction, recompute.json); 3072-3081 from my own main-tree rollouts
(sim_panel/panel.jsonl); r9: 3042-3061 from the lens's run_r9, 3062-3081 mine.  Rule: a lean key fails when 30 or
more of 40 episodes share its sign (two-sided p 0.0022) and |median| >= 1 mm (CoM) or 0.10 deg (roll); creep fails
on 2 or more episodes at >= 1 mm/s.  Also CoM-P1's n = 10 reading on the new seeds alone.
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

D = json.load(open(sys.argv[1]))["mine"]
new = defaultdict(dict)
for line in open(sys.argv[2]):
    r = json.loads(line)
    new[r["who"]][r["seed"]] = r


def p_sign(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


stood_new = sum(r["stood"] for w in new for r in new[w].values())
print(f"new episodes: {sum(len(v) for v in new.values())}, stood {stood_new}")
rows = []
for who in sorted(new):
    if who == "r9":
        old = {int(s): ep for s, ep in D["run_r9"]["r9_4194304"].items()}
    else:
        old = {**{int(s): ep for s, ep in D["run"][who].items()}, **{int(s): ep for s, ep in D["run_held"][who].items()}}
    ok = [s for s in sorted(new[who]) if new[who][s]["stood"]]
    fell = [s for s in sorted(new[who]) if not new[who][s]["stood"]]
    com = [old[s]["5.0"]["com_lat_mm"] for s in sorted(old)] + [1000 * new[who][s]["com_lateral_mean_m"] for s in ok]
    roll = [old[s]["5.0"]["roll_deg"] for s in sorted(old)] + [new[who][s]["root_roll_mean_deg"] for s in ok]
    creep = [old[s]["5.0"]["creep_mmps"] for s in sorted(old)] + [new[who][s]["creep_mmps"] for s in ok]
    out = [who + (f" (FELL on {fell}: CoM-P1 'not read'; read here on the {len(com)} that stood)" if fell else "")]
    fail = False
    for name, v, fl in (("com", np.array(com), 1.0), ("roll", np.array(roll), 0.10)):
        a, b = int((v > 0).sum()), int((v < 0).sum())
        f = max(a, b) >= (30 if len(v) == 40 else 29) and abs(np.median(v)) >= fl  # 29 of 39: two-sided p 0.0034
        fail |= f
        out.append(f"{name} {a:2d}+/{b:2d}- p {p_sign(a, b):.4f} med {np.median(v):+.3f}{' FAIL' if f else ''}")
    nc = sum(c >= 1.0 for c in creep)
    fail |= nc >= 2
    out.append(f"creep eps {nc} max {max(creep):.3f} mm/s")
    # the new 10 alone (CoM-P1 n = 10 form)
    v10c = np.array([1000 * new[who][s]["com_lateral_mean_m"] for s in ok][-10:])
    v10r = np.array([new[who][s]["root_roll_mean_deg"] for s in ok][-10:])
    out.append(f"new10 com {int((v10c > 0).sum())}+/{int((v10c < 0).sum())}- roll {int((v10r > 0).sum())}+/{int((v10r < 0).sum())}-")
    rows.append((fail, " | ".join(out)))
for f, line in rows:
    print(("FAIL " if f else "pass ") + line)
print(f"failing: {[r.split(' | ')[0] for f, r in rows if f]}")
pols = [w for w in new if w != "r9"]
mx = max(new[w][s]["creep_mmps"] for w in pols for s in new[w] if new[w][s]["stood"])
big = sum(abs(new[w][s]["com_lateral_mean_m"]) > 0.005 for w in pols if w.startswith("s80_") for s in new[w] if new[w][s]["stood"])
print(f"new seeds 3072-3081, 26 policies: max feet creep {mx:.3f} mm/s; s80 episodes |CoM lat| > 5 mm: {big}/120")
hr = [sum(new[w][s]["hiproll_cmd_r_l"]) for w in pols for s in new[w] if new[w][s]["stood"]]
print(f"hip-roll r + l per episode (2-10 s mean), new seeds: max |.| {max(abs(x) for x in hr):.4f}")
