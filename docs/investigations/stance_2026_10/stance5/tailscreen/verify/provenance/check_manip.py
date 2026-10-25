import numpy as np, csv, sys, math
S = sys.argv[1]
names = [f"{st}_{a}_r{sd}" for st in ("s42", "s80") for a in ("control", "b90", "b60") for sd in (101, 202, 303)]
agg = {}
for n in names:
    st, a, r = n.split("_")
    rows = list(csv.DictReader(open(f"{S}/runs/{n}/progress.csv")))
    tr = [x for x in rows if x.get("train/tail3_bound_penalty", "") != ""]
    pen = np.array([float(x["train/tail3_bound_penalty"]) for x in tr])
    loss = np.array([float(x["train/tail3_bound_loss"]) for x in tr])
    coef = set(float(x["train/tail3_bound_coef"]) for x in tr)
    bb = set(float(x["train/tail3_bound_b"]) for x in tr)
    bf = np.array([float(x["train/tail3_bound_beyond_fraction"]) for x in tr])
    ts = [int(x["time/total_timesteps"]) for x in rows if x["time/total_timesteps"]]
    z = np.load(f"{S}/runs/{n}/telemetry.npz", allow_pickle=False)
    g90 = z["tail3_mu_gt_0p90"]; g60 = z["tail3_mu_gt_0p60"]; pb = z["tail3_bound_penalty_buffer"]
    ppen = z["prev_train_tail3_bound_penalty"]; pc = z["prev_train_tail3_bound_coef"]
    gb = g60 if a == "b60" else g90
    print(f"{n:18s} prog_rows={len(rows)} train_rows={len(tr)} ts0={ts[0]} tsN={ts[-1]} pen_min={pen.min():.3g} pen_max={pen.max():.3g} pen_nonzero={int((pen!=0).sum())} loss_nonzero={int((loss!=0).sum())} coef={coef} b={bb} beyond_mean={bf.mean():.3f} "
          f"tel_n={len(g90)} tel_coef={set(pc[~np.isnan(pc)].tolist())} tel_prevpen_nz={int((np.nan_to_num(ppen)!=0).sum())} buf_pen_nz={int((pb!=0).sum())} "
          f"share>0.9 mean={g90.mean():.3f} last16={g90[-16:].mean():.3f} share>0.6 mean={g60.mean():.3f} last16={g60[-16:].mean():.3f}")
    agg.setdefault((st, a), []).append((g90.mean(), g60.mean(), g90[-16:].mean(), g60[-16:].mean(), gb.mean()))
print()
for k, v in sorted(agg.items()):
    v = np.array(v); print(k, "mean share>0.90 %.3f  >0.60 %.3f  last16 >0.90 %.3f  >0.60 %.3f  per-rep share>own b" % tuple(v[:, :4].mean(0)), np.round(v[:, 4], 3))
