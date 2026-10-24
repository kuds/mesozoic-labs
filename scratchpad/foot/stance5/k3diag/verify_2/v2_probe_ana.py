"""verify_2: reduce probe/*.npz (numpy only).  Window = steps 200 .. L-1 (survivors) or 200 .. L-151 (falls).
Jq = open-loop d(mu_r_hp - mu_l_hp)/d(q_r_hp - q_l_hp) [per rad]; Jv = same per rad/s of the velocity difference.
Closed-loop regressions on the same rollout: d_mu on stagger + dq (k3gain's spec) and d_mu on dq alone."""
import glob
import json
import os
import sys

import numpy as np

P = sys.argv[1]
W0 = 200
rows = []
for f in sorted(glob.glob(os.path.join(P, "*.npz"))):
    d = np.load(f)
    meta = json.loads(str(d["meta"]))
    L = d["length"]
    sel = {k: [] for k in ("Jq", "Jv", "Jq_common", "dmu", "dq", "stagger")}
    X, Y, X1 = [], [], []
    for e in range(len(L)):
        hi = int(L[e]) if L[e] >= 1000 else int(L[e]) - 150
        if hi <= W0 + 60:
            continue
        sl = slice(W0, hi)
        for k in sel:
            sel[k].append(d[k][e, sl].astype(np.float64))
        s, q, m = (d[k][e, sl].astype(np.float64) for k in ("stagger", "dq", "dmu"))
        X.append(np.stack([s - s.mean(), q - q.mean()], 1))
        Y.append(m - m.mean())
    if not X:
        continue
    X, Y = np.concatenate(X), np.concatenate(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    c1 = float(X[:, 1] @ Y / (X[:, 1] @ X[:, 1]))
    cat = {k: np.concatenate(v) for k, v in sel.items()}
    Jq = cat["Jq"]
    r = {**{k: meta[k] for k in ("run", "steps", "cond", "falls")}, "matches_judge": meta.get("matches_judge"),
         "Jq_median": round(float(np.median(Jq)), 3), "Jq_mean": round(float(Jq.mean()), 3),
         "Jq_p10_p90": [round(float(np.percentile(Jq, 10)), 3), round(float(np.percentile(Jq, 90)), 3)],
         "Jq_ep_medians_range": None,
         "Jv_median": round(float(np.median(cat["Jv"])), 4),
         "Jq_common_median": round(float(np.median(cat["Jq_common"])), 3),
         "reg_dq_given_stagger": round(float(coef[1]), 3), "reg_stagger_given_dq": round(float(coef[0]), 3),
         "reg_dq_only": round(c1, 3),
         "stagger_sd_mm": round(1000 * float(np.std(cat["stagger"])), 1),
         "dq_sd_mrad": round(1000 * float(np.std(cat["dq"])), 1),
         "own_sigma_hp": round(meta["own_sigma_hp"], 4)}
    epm = [float(np.median(v)) for v in sel["Jq"]]
    r["Jq_ep_medians_range"] = [round(min(epm), 3), round(max(epm), 3)]
    rows.append(r)
    print(json.dumps(r))
json.dump(rows, open(os.path.join(os.path.dirname(P.rstrip("/")), "v2_probe.json"), "w"), indent=1)
