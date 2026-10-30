"""How large a same-side lean CoM-P1 would catch on these policies' own spread (semi-synthetic; no simulation).

python -B power.py RUN_DIR
For each of the 26 policies, its 20 per-seed window means (from 5 s) of the CoM lateral offset and the roll are
shifted by a bias b of the sign that works against the policy's own median (the harder case) and of the other sign,
and CoM-P1's key rule is applied (p <= 0.005 and |median| >= floor).  Reported: the share of policies flagged.
Also the fore-aft creep direction pooled over the statue's 120 episodes (fore-aft is not mirror-symmetric).
"""
import json, sys
from collections import defaultdict
import numpy as np
from environments.shared.tests.stance_track_helpers import sign_test_p

FLOORS = {"com_lateral_mean_m": 1e-3, "root_roll_mean_deg": 0.10}
vals = defaultdict(lambda: defaultdict(list))
for line in open(f"{sys.argv[1]}/episodes.jsonl"):
    r = json.loads(line)
    if r["who"] == "statue":
        continue
    for k in FLOORS:
        vals[r["who"]][k].append(r["metrics"]["5.0"][k])
def flag(v, floor):
    v = np.asarray(v)
    return sign_test_p(int((v > 0).sum()), int((v < 0).sum())) <= 0.005 and abs(np.median(v)) >= floor
for k, biases, sc, u in (("com_lateral_mean_m", (0.5e-3, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3), 1000, "mm"),
                         ("root_roll_mean_deg", (0.05, 0.1, 0.15, 0.2, 0.3), 1, "deg")):
    out = []
    for b in biases:
        both = []
        for w, d in vals.items():
            v = np.array(d[k])
            s = -np.sign(np.median(v)) or 1.0
            both.append((flag(v + s * b, FLOORS[k]), flag(v - s * b, FLOORS[k])))
        against = np.mean([x[0] for x in both]); along = np.mean([x[1] for x in both])
        out.append(f"b {b * sc:g}{u}: {against:.2f} / {along:.2f}")
    print(f"{k}: share of the 26 policies flagged with a bias b added (against / along the policy's own median sign): " + "; ".join(out))
