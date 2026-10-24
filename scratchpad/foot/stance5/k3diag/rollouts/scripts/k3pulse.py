"""K3 recovery probe readout (REPORT ONLY): the deterministic hip-pitch pulse conditions of k3roll.py (pulse_anti: right
+0.3 / left -0.3, pulse_sym: both +0.3, normalised command, steps 400-419) against the same checkpoint's noise0 episodes
(same spawns).  Per episode: foot stagger, COM fore-aft and pelvis pitch relative to the unperturbed episode; the peak
response in 400-699, the residual at steps 500, 700 and 999, and the settle time (last step at which |delta stagger| >
5 mm, minus 400, in s).  A fall counts as not settled.

    python3 k3pulse.py NPZ_DIR OUT_CSV
"""
import csv
import warnings
import glob
import json
import os
import sys

import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
src, dst = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(src, "*__pulse_*.npz"))):
    run, st, cond = os.path.basename(f)[:-4].split("__")
    P = np.load(f)
    B = np.load(os.path.join(src, f"{run}__{st}__noise0.npz"))
    mp, mb = json.loads(str(P["meta_json"])), json.loads(str(B["meta_json"]))
    cp, cb = P["core"].astype(np.float64), B["core"].astype(np.float64)
    ip, ib = mp["core_cols"], mb["core_cols"]
    get = lambda c, i, k: c[:, :, i.index(k)]
    ds = get(cp, ip, "i_foot_stagger_m") - get(cb, ib, "i_foot_stagger_m")
    dc = get(cp, ip, "com_fa") - get(cb, ib, "com_fa")
    dp = get(cp, ip, "pitch_deg") - get(cb, ib, "pitch_deg")
    dr = get(cp, ip, "roll_deg") - get(cb, ib, "roll_deg")
    L = P["length"]
    peak = np.nanmax(np.abs(ds[:, 400:700]), axis=1)
    settle = []
    for e in range(len(L)):
        if L[e] < 1000:
            settle.append(np.nan)
            continue
        big = np.flatnonzero(np.abs(ds[e, 400:]) > 0.005)
        settle.append(0.0 if big.size == 0 else (big[-1] + 1) * 0.01)
    settle = np.asarray(settle)
    r = {"run": run, "steps": st, "cond": cond, "falls": int((L < 1000).sum()),
         "stagger_peak_mm_med": 1000 * float(np.nanmedian(peak)), "stagger_peak_mm_max": 1000 * float(np.nanmax(peak)),
         "stagger_resid_500_mm_med": 1000 * float(np.nanmedian(np.abs(ds[:, 500]))),
         "stagger_resid_700_mm_med": 1000 * float(np.nanmedian(np.abs(ds[:, 700]))),
         "stagger_resid_999_mm_med": 1000 * float(np.nanmedian(np.abs(ds[:, 999]))),
         "stagger_resid_999_mm_max": 1000 * float(np.nanmax(np.abs(ds[:, 999]))),
         "settle_s_med": float(np.nanmedian(settle)) if np.isfinite(settle).any() else float("nan"),
         "settled_by_6s": int(np.sum(np.isfinite(settle) & (settle < 5.99))),
         "com_fa_peak_mm_med": 1000 * float(np.nanmedian(np.nanmax(np.abs(dc[:, 400:700]), axis=1))),
         "com_fa_resid_999_mm_med": 1000 * float(np.nanmedian(np.abs(dc[:, 999]))),
         "pitch_peak_deg_med": float(np.nanmedian(np.nanmax(np.abs(dp[:, 400:700]), axis=1))),
         "roll_peak_deg_med": float(np.nanmedian(np.nanmax(np.abs(dr[:, 400:700]), axis=1)))}
    if "per_act" in P.files and "per_act" in B.files:  # the policy's own differential hip-pitch response (raw mean)
        mp_, mb_ = P["per_act"].astype(np.float64), B["per_act"].astype(np.float64)
        dmu = (mp_[:, :, 0, 3] - mp_[:, :, 0, 7]) - (mb_[:, :, 0, 3] - mb_[:, :, 0, 7])
        dq = (mp_[:, :, 1, 3] - mp_[:, :, 1, 7]) - (mb_[:, :, 1, 3] - mb_[:, :, 1, 7])
        r["dmu_diff_during_med"] = float(np.nanmedian(np.nanmean(dmu[:, 400:420], axis=1)))
        r["dmu_diff_after_100_med"] = float(np.nanmedian(np.nanmean(dmu[:, 420:520], axis=1)))
        r["dq_diff_peak_rad_med"] = float(np.nanmedian(np.nanmax(np.abs(dq[:, 400:520]), axis=1)))
    rows.append(r)
keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(dst, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
