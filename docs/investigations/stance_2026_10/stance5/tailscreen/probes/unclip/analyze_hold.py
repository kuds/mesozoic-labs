"""Reduce roll_hold.py: per held a, the policy's own tail_3 mu (not applied) vs q3 over window 200..L-1."""
import json
import sys
import numpy as np
for pre in sys.argv[1:]:
    d = json.load(open(pre + ".json")); z = np.load(pre + ".npz")
    print(f"\n#### {d['label']}  zip {d['zip_sha256'][:12]}")
    print("hold_a  falls  q3_mean   q3_last100  mu_mean  mu_q05  mu_q50  mu_q95  mu-a(mean)  eps_mu>a  F_mean  ctrl  win_return")
    for mode, rows in d["runs"].items():
        a = float(mode.split("_")[1])
        mus = np.concatenate([z[f"{mode}_{i}_mu"][200:r["length"]] for i, r in enumerate(rows)])
        q3 = np.concatenate([z[f"{mode}_{i}_q3"][200:r["length"]] for i, r in enumerate(rows)])
        q3l = np.mean([z[f"{mode}_{i}_q3"][max(200, r["length"] - 100):r["length"]].mean() for i, r in enumerate(rows)])
        fo = np.concatenate([z[f"{mode}_{i}_force"][200:r["length"]] for i, r in enumerate(rows)])
        ct = np.concatenate([z[f"{mode}_{i}_ctrl"][200:r["length"]] for i, r in enumerate(rows)])
        epm = [z[f"{mode}_{i}_mu"][200:r["length"]].mean() for i, r in enumerate(rows)]
        print(f"{a:6.3f}  {sum(r['fell'] for r in rows):2d}    {q3.mean():+.5f}  {q3l:+.5f}    {mus.mean():.4f}  "
              f"{np.quantile(mus,.05):.4f}  {np.quantile(mus,.5):.4f}  {np.quantile(mus,.95):.4f}  {mus.mean()-a:+.4f}"
              f"      {sum(m > a for m in epm):2d}/20  {fo.mean():+.2f}  {ct.mean():.4f}  "
              f"{np.mean([r['window_return'] for r in rows]):.1f}")
    print("per-episode window-mean mu by hold level (rows = episodes):")
    modes = list(d["runs"])
    print("ep  " + "  ".join(f"{m.split('_')[1]:>6}" for m in modes))
    for i in range(20):
        print(f"{i:2d}  " + "  ".join(f"{z[f'{m}_{i}_mu'][200:].mean():6.3f}" for m in modes))
