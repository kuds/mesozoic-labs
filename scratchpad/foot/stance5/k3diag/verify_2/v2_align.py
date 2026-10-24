"""verify_2: k3gain's regression on k3's own npz + onset windows, with q as stored (post-step, q(t+1)) vs shifted one
step back (the angle the policy saw when it computed mu(t)); plus a scipy-free Spearman summary."""
import csv, json, os, sys
import numpy as np
R = sys.argv[1]
ep = {(r["run"], r["steps"], r["cond"], r["seed"]): r for r in csv.DictReader(open(os.path.join(R, "tables", "episodes.csv")))}
out = {}
for st in ["4194304", "5242880", "6291456", "7340032", "8388608", "10485760", "final"]:
    run = "stance5_LTSB_s42"
    d = np.load(os.path.join(R, "npz", f"{run}__{st}__full.npz"))
    m = json.loads(str(d["meta_json"]))
    c = d["core"].astype(np.float64); pa = d["per_act"].astype(np.float64)
    stg = c[:, :, m["core_cols"].index("i_foot_stagger_m")]
    res = {}
    for shift in (0, 1):
        X, Y = [], []
        for e in range(len(d["length"])):
            r = ep[(run, st, "full", str(int(d["seed"][e])))]
            L = int(r["length"]); hi = int(r["onset"]) if r["fell"] == "1" else L
            if hi <= 260: continue
            sl = slice(200, hi); sl0 = slice(200 - shift, hi - shift)
            dmu = pa[e, sl, 0, 3] - pa[e, sl, 0, 7]
            dq = pa[e, sl0, 1, 3] - pa[e, sl0, 1, 7]
            s = stg[e, sl]
            ok = np.isfinite(dmu) & np.isfinite(dq) & np.isfinite(s)
            X.append(np.stack([s[ok] - s[ok].mean(), dq[ok] - dq[ok].mean()], 1)); Y.append(dmu[ok] - dmu[ok].mean())
        X, Y = np.concatenate(X), np.concatenate(Y)
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res[f"shift{shift}"] = [round(float(coef[1]), 3), round(1 - float(((Y - X @ coef) ** 2).sum() / (Y ** 2).sum()), 3)]
    out[st] = res
    print(st, res)
json.dump(out, open(sys.argv[2], "w"))
