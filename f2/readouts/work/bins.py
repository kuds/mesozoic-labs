# Binned (1,048,576-step) series of M6 shares, std and ledger over-bar rows / TEL command means, for the entropy check.
import json, sys, csv, pathlib, math
import numpy as np
W = pathlib.Path(sys.argv[1])
ver = json.loads((W / "verified_copies.json").read_text())
ACT = ("neck_pitch_act","neck_yaw_act","head_pitch_act","r_hip_pitch_act","r_hip_roll_act","r_knee_act","r_ankle_act",
       "l_hip_pitch_act","l_hip_roll_act","l_knee_act","l_ankle_act","tail_1_pitch_act","tail_1_yaw_act","tail_2_pitch_act","tail_3_pitch_act")
out = {}
for rid, sr, focus in (("stance6_r10LTSB_s60","s60",[(14,"hi")]),("stance6_r10LTSB_s42","s42",[(14,"hi")]),("stance6_r10LTSB_s80","s80",[(0,"lo"),(14,"hi")])):
    p = ver[rid]["paths"]["01_stance/policy_probe.npz"]["verified"]
    with np.load(p, allow_pickle=False) as z:
        T = z["update_timesteps"]; STD = z["std_per_actuator"]; ENT = z["ent_coef"]
        HI = z["rollout_mu_beyond_high_share"]; LO = z["rollout_mu_beyond_low_share"]; MU = z["rollout_mu_mean"]
    led = ver[rid]["paths"]["01_stance/eval_episodes.csv"]["verified"]
    rows = list(csv.DictReader(open(led, newline="")))
    res = []
    edges = list(range(0, 12582912 + 1, 1048576))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (T > lo) & (T <= hi)
        if not m.any():
            continue
        d = {"bin": f"({lo},{hi}]", "ent_mean": round(float(ENT[m].mean()), 6)}
        for i, side in focus:
            a = (HI if side == "hi" else LO)[m, i]
            d[f"{ACT[i]}_mu_beyond_{side}_mean"] = round(float(a.mean()), 4)
            d[f"{ACT[i]}_mu_beyond_{side}_ge05"] = int((a >= 0.05).sum())
            d[f"{ACT[i]}_mu_mean"] = round(float(MU[m, i].mean()), 3)
            d[f"{ACT[i]}_std_end"] = round(float(STD[m, i][-1]), 4)
        sel = [x for x in rows if lo < int(float(x["timesteps"])) <= hi]
        over = [x for x in sel if (float(x["saturated_actuator_fraction"]) if x["saturated_actuator_fraction"] not in ("", "nan") else 0) > 0.10]
        d["eval_rows"] = len(sel); d["over_bar_rows"] = len(over)
        for i, side in focus:
            vals = [float(x[f"{ACT[i]}_command_mean"]) for x in sel if x[f"{ACT[i]}_command_mean"] not in ("", "nan") and float(x["length"]) >= 1000]
            d[f"{ACT[i]}_cmd_mean"] = round(sum(vals) / len(vals), 3) if vals else None
        res.append(d)
    last = {}
    for i, side in focus:
        a = (HI if side == "hi" else LO)[:, i]
        idx = np.where(a >= 0.05)[0]
        last[f"{ACT[i]}_{side}_last_ge05"] = int(T[idx[-1]]) if len(idx) else None
        last[f"{ACT[i]}_{side}_n_ge05"] = int(len(idx))
    out[sr] = {"bins": res, "last": last}
(W / "bins.json").write_text(json.dumps(out, indent=1) + "\n")
for sr, v in out.items():
    print("==", sr, v["last"])
    for d in v["bins"]:
        print("  ", d)
