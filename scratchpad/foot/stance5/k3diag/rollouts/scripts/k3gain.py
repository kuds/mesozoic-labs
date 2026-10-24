"""K3 (REPORT ONLY): the policy's differential hip-pitch response under the joint noise, per checkpoint ("full" condition).
Over pre-failure window steps (200 .. onset or 999; onset from tables/episodes.csv), least squares of
d_mu(t) = mu_r_hip_pitch - mu_l_hip_pitch (the raw policy mean, before any noise) on stagger(t) and d_q(t) = q_r - q_l
(hip pitch angles), all episode-demeaned; also the same with stagger lagged 10 steps.  Units: normalised command per m of
stagger and per rad of hip-pitch difference.  Descriptive: closed-loop data, not an identified controller.

    python3 k3gain.py ROLLOUTS_DIR
"""
import csv
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
ep = {(r["run"], r["steps"], r["cond"], r["seed"]): r for r in csv.DictReader(open(os.path.join(R, "tables", "episodes.csv")))}
out = []
for f in sorted(glob.glob(os.path.join(R, "npz", "*__full.npz"))):
    run, st, cond = os.path.basename(f)[:-4].split("__")
    d = np.load(f)
    m = json.loads(str(d["meta_json"]))
    c = d["core"].astype(np.float64)
    pa = d["per_act"].astype(np.float64)
    stg = c[:, :, m["core_cols"].index("i_foot_stagger_m")]
    X, Y = [], []
    for e in range(len(d["length"])):
        r = ep[(run, st, "full", str(int(d["seed"][e])))]
        L = int(r["length"])
        hi = int(r["onset"]) if r["fell"] == "1" else L
        if hi <= 260:
            continue
        sl = slice(200, hi)
        dmu = pa[e, sl, 0, 3] - pa[e, sl, 0, 7]
        dq = pa[e, sl, 1, 3] - pa[e, sl, 1, 7]
        s = stg[e, sl]
        ok = np.isfinite(dmu) & np.isfinite(dq) & np.isfinite(s)
        X.append(np.stack([s[ok] - s[ok].mean(), dq[ok] - dq[ok].mean()], 1))
        Y.append(dmu[ok] - dmu[ok].mean())
    X, Y = np.concatenate(X), np.concatenate(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    r2 = 1 - float(((Y - pred) ** 2).sum() / (Y ** 2).sum())
    cs = float((X[:, 0] @ Y) / (X[:, 0] @ X[:, 0]))
    out.append({"run": run, "steps": st, "falls": int((d["length"] < 1000).sum()), "gain_dmu_per_m_stagger": round(float(coef[0]), 3),
                "gain_dmu_per_rad_dq": round(float(coef[1]), 3), "r2": round(r2, 3), "simple_slope_dmu_on_stagger": round(cs, 3),
                "dmu_sd": round(float(Y.std()), 4), "stagger_sd_m": round(float(X[:, 0].std()), 4)})
with open(os.path.join(R, "tables", "policy_gain.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(json.dumps(r))
