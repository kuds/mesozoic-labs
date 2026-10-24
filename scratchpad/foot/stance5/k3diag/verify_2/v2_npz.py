"""verify_2 (REPORT ONLY, numpy only): re-derive the K3 claim's numbers from the k3roll npz files, with my own window
definitions, alternative regression specs and episode-bootstrap CIs.

    python3 v2_npz.py NPZ_DIR OUT_JSON
"""
import glob
import json
import os
import sys

import numpy as np

NPZ, OUT = sys.argv[1], sys.argv[2]
W0 = 200
RHP, LHP = 3, 7
SLOTS = ["4194304", "5242880", "6291456", "7340032", "8388608", "10485760", "final"]
RUNS = [("stance5_LTSB_s42", s) for s in SLOTS] + [("stance5_LTSB_s52", "final"), ("stance5_LTSB_s60", "final"),
                                                   ("stance4_LTS_s42", "final"), ("stance4_LTS_s52", "final"),
                                                   ("stance4_LTS_s60", "final")]
rng = np.random.default_rng(0)


def load(run, st, cond):
    p = os.path.join(NPZ, f"{run}__{st}__{cond}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=False)
    m = json.loads(str(d["meta_json"]))
    out = {"meta": m, "length": d["length"].astype(int), "seed": d["seed"],
           "core": {k: d["core"][:, :, i].astype(np.float64) for i, k in enumerate(m["core_cols"])}}
    if "per_act" in d.files:
        out["pa"] = d["per_act"].astype(np.float64)
    return out


def window_hi(D, e, mode):
    """my pre-failure window end: 'cut150' = drop the last 150 steps of a fall; 'tilt' = first step |roll|>5 or
    |pitch-ref|>5 deg after which the episode never returns (computed here, not k3analyze's onset)."""
    L = int(D["length"][e])
    if L >= 1000:
        return L
    if mode == "cut150":
        return max(L - 150, 0)
    roll = D["core"]["roll_deg"][e, :L]
    pitch = D["core"]["pitch_deg"][e, :L]
    pref = np.nanmedian(D["core"]["pitch_deg"][e, W0:min(L, 400)]) if L > 260 else np.nanmedian(pitch)
    good = (np.abs(roll) <= 5) & (np.abs(pitch - pref) <= 5)
    idx = np.flatnonzero(good)
    return int(idx[-1] + 1) if idx.size else 0


def design(D, mode):
    pa, stg = D["pa"], D["core"]["i_foot_stagger_m"]
    rows = []
    for e in range(len(D["length"])):
        hi = window_hi(D, e, mode)
        if hi <= W0 + 60:
            continue
        sl = slice(W0, hi)
        dmu = pa[e, sl, 0, RHP] - pa[e, sl, 0, LHP]
        dq = pa[e, sl, 1, RHP] - pa[e, sl, 1, LHP]
        dqd = np.gradient(pa[e, :, 1, RHP] - pa[e, :, 1, LHP])[sl] / 0.01
        s = stg[e, sl]
        ok = np.isfinite(dmu) & np.isfinite(dq) & np.isfinite(s) & np.isfinite(dqd)
        rows.append({"dmu": dmu[ok], "dq": dq[ok], "dqd": dqd[ok], "s": s[ok], "e": e})
    return rows


def fit(rows, cols):
    X = np.concatenate([np.stack([r[c] - r[c].mean() for c in cols], 1) for r in rows])
    Y = np.concatenate([r["dmu"] - r["dmu"].mean() for r in rows])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r2 = 1 - float(((Y - X @ coef) ** 2).sum() / (Y ** 2).sum())
    return coef, r2


def boot(rows, cols, k, B=400):
    vals = []
    n = len(rows)
    for _ in range(B):
        pick = [rows[i] for i in rng.integers(0, n, n)]
        try:
            vals.append(fit(pick, cols)[0][k])
        except Exception:
            pass
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


res = []
for run, st in RUNS:
    D = load(run, st, "full")
    if D is None:
        continue
    r = {"run": run, "steps": st, "falls_full": int((D["length"] < 1000).sum()),
         "verified_vs_judge": D["meta"]["verified"]["lengths_terms_equal"] if D["meta"]["verified"] else None,
         "own_sigma_hp": round(float(np.mean([D["meta"]["own_sigma"][RHP], D["meta"]["own_sigma"][LHP]])), 4),
         "own_sigma_mean": round(float(np.mean(D["meta"]["own_sigma"])), 4)}
    for mode in ("cut150", "tilt"):
        rows = design(D, mode)
        c2, r2 = fit(rows, ["s", "dq"])
        c1, r21 = fit(rows, ["dq"])
        c3, r23 = fit(rows, ["dq", "dqd"])
        c4, r24 = fit(rows, ["s", "dq", "dqd"])
        S = np.concatenate([x["s"] - x["s"].mean() for x in rows])
        Q = np.concatenate([x["dq"] - x["dq"].mean() for x in rows])
        Sraw = np.concatenate([x["s"] for x in rows])
        r[mode] = {"n_ep": len(rows), "n_steps": int(S.size),
                   "gain_dq_given_stagger": round(float(c2[1]), 3), "gain_stagger_given_dq": round(float(c2[0]), 3), "r2": round(r2, 3),
                   "gain_dq_given_stagger_ci": boot(rows, ["s", "dq"], 1),
                   "gain_dq_only": round(float(c1[0]), 3), "r2_dq_only": round(r21, 3), "gain_dq_only_ci": boot(rows, ["dq"], 0),
                   "gain_dq_with_dqdot": [round(float(x), 3) for x in c3], "r2_dq_dqdot": round(r23, 3),
                   "gain_s_dq_dqdot": [round(float(x), 3) for x in c4],
                   "corr_stagger_dq": round(float(np.corrcoef(S, Q)[0, 1]), 3),
                   "stagger_sd_raw_mm": round(1000 * float(np.std(Sraw)), 1),
                   "stagger_sd_demeaned_mm": round(1000 * float(np.std(S)), 1),
                   "dq_sd_mrad": round(1000 * float(np.std(Q)), 1)}
    # tail habit: share of window steps (200..L-1) with |t3 command| >= 0.9, from the raw mean (cmd = clip(mu))
    for cond in ("noise0", "full"):
        Dc = D if cond == "full" else load(run, st, cond)
        if Dc is None:
            continue
        sh, mus = [], []
        for e in range(len(Dc["length"])):
            L = int(Dc["length"][e])
            if L <= W0:
                continue
            t3 = np.clip(Dc["core"]["t3_mu"][e, W0:L], -1, 1)
            sh.append(np.abs(t3) >= 0.9)
            mus.append(Dc["core"]["t3_mu"][e, W0:L])
        sh = np.concatenate(sh)
        r[f"t3_share_ge09_{cond}"] = round(float(sh.mean()), 3)
        r[f"t3_mu_mean_{cond}"] = round(float(np.nanmean(np.concatenate(mus))), 3)
    for cond in ("hp_anti", "hp_sym", "hip_pitch", "t3hold0_full", "notail", "tail", "legs_no_hippitch", "pulse_anti"):
        Dc = load(run, st, cond)
        if Dc is not None:
            r[f"falls_{cond}"] = int((Dc["length"] < 1000).sum())
    res.append(r)
    print(json.dumps(r), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
