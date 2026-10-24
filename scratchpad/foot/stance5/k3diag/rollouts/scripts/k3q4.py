"""K3 question 4, rollout side (REPORT ONLY): stance5 s42's joint-noise falls per checkpoint beside its tail habit and the
other candidates (policy std, leg posture, COM), from tables/conditions.csv (k3analyze.py), the npz metadata (own sigma) and
the earlier satroll outputs (cross-check of the tail share).  Spearman rank correlations over the 7 checkpoints are
descriptive only (n = 7).

    python3 k3q4.py ROLLOUTS_DIR
"""
import csv
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
SP = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
STEPS = ["4194304", "5242880", "6291456", "7340032", "8388608", "10485760", "final"]
rows = list(csv.DictReader(open(os.path.join(R, "tables", "conditions.csv"))))
idx = {(r["run"], r["steps"], r["cond"]): r for r in rows}


def f(run, st, cond, k):
    r = idx.get((run, st, cond))
    if r is None or r.get(k, "") in ("", "nan"):
        return float("nan")
    return float(r[k])


def own_sigma(run, st):
    p = os.path.join(R, "npz", f"{run}__{st}__noise0.npz")
    d = np.load(p)
    m = json.loads(str(d["meta_json"]))
    return np.asarray(m["own_sigma"])


def satroll(st):
    p = f"{SP}/foot/stance5/tools/satroll/stance5_LTSB_s42_{st}.json"
    if not os.path.exists(p):
        return float("nan")
    d = json.load(open(p))
    return float(d["checkpoints"][st]["named"]["frac90"]) if d["checkpoints"][st]["named"]["actuator"] == "tail_3_pitch_act" else float("nan")


def judge(run, st):
    root = "stance5" if run.startswith("stance5") else "stance4"
    d = json.load(open(f"{SP}/foot/{root}/judged/{run}/judge_{st}.json"))
    return d["noise_matched_control_joint"]["falls"], d["noise_matched_own"]["falls"], d["panel"]["n_clean"]


def table(run, steps):
    out = []
    for st in steps:
        if (run, st, "full") not in idx:
            continue
        sig = own_sigma(run, st)
        jf, jo, clean = judge(run, st)
        out.append({
            "run": run, "steps": st, "judge_joint_falls": jf, "judge_own_falls": jo, "panel_clean_of_40": clean,
            "falls_full": f(run, st, "full", "falls"), "falls_legs": f(run, st, "legs", "falls"),
            "falls_notail": f(run, st, "notail", "falls"), "falls_tail": f(run, st, "tail", "falls"),
            "falls_neckhead": f(run, st, "neckhead", "falls"),
            "det_t3_ge09_share": f(run, st, "noise0", "t3_cmd_ge09_share"), "satroll_t3_frac90": satroll(st) if run == "stance5_LTSB_s42" else float("nan"),
            "det_t3_cmd_mean": f(run, st, "noise0", "t3_cmd_mean"), "noise_t3_ge09_share": f(run, st, "full", "t3_cmd_ge09_share"),
            "noise_t3_cmd_mean": f(run, st, "full", "t3_cmd_mean"),
            "own_sigma_mean": float(sig.mean()), "own_sigma_legs": float(sig[3:11].mean()), "own_sigma_hiproll": float(sig[[4, 8]].mean()),
            "own_sigma_ankle": float(sig[[6, 10]].mean()), "own_sigma_tail3": float(sig[14]),
            "det_com_fa_mm": 1000 * f(run, st, "noise0", "com_fa_mean"), "det_pitch_deg": f(run, st, "noise0", "pitch_deg_mean"),
            "det_wcop_med": f(run, st, "noise0", "wcop_med"),
            "det_q_hip_pitch": f(run, st, "noise0", "q_hip_pitch_mean_rad"), "det_q_hip_roll": f(run, st, "noise0", "q_hip_roll_mean_rad"),
            "det_q_knee": f(run, st, "noise0", "q_knee_mean_rad"), "det_q_ankle": f(run, st, "noise0", "q_ankle_mean_rad"),
            "noise_com_fa_mm": 1000 * f(run, st, "full", "com_fa_mean"), "noise_wcop_med": f(run, st, "full", "wcop_med"),
            "noise_toe_share": f(run, st, "full", "wcop_toe_share"), "noise_heel_share": f(run, st, "full", "wcop_heel_share"),
            "noise_roll_sd": f(run, st, "full", "roll_sd"), "noise_ar50_roll": f(run, st, "full", "ar50_roll_deg"),
            "noise_ar50fh_roll": f(run, st, "full", "ar50fh_roll_deg"), "noise_ar50_com_lat": f(run, st, "full", "ar50_com_lat"),
            "noise_ar50fh_com_lat": f(run, st, "full", "ar50fh_com_lat"), "noise_ar50_com_fa": f(run, st, "full", "ar50_com_fa"),
            "noise_ar50fh_com_fa": f(run, st, "full", "ar50fh_com_fa"),
            "noise_unload_share": f(run, st, "full", "unload_step_share"), "noise_unload_events": f(run, st, "full", "unload_events_ge10"),
            "noise_recovered_share": f(run, st, "full", "recovered_share"),
            "noise_stagger_sd_mm": 1000 * f(run, st, "full", "stagger_sd_m"), "noise_hip_pitch_diff_sd": f(run, st, "full", "hip_pitch_diff_sd_rad"),
            "noise_split_share": f(run, st, "full", "split_share"),
            "falls_hip_pitch": f(run, st, "hip_pitch", "falls"), "falls_legs_no_hippitch": f(run, st, "legs_no_hippitch", "falls"),
            "falls_hp_anti": f(run, st, "hp_anti", "falls"), "falls_hp_sym": f(run, st, "hp_sym", "falls"),
            "t3hold0_falls": f(run, st, "t3hold0_full", "falls"), "t3hold0_det_com_fa_mm": 1000 * f(run, st, "t3hold0_noise0", "com_fa_mean"),
        })
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    # average ranks for ties
    for arr, src in ((rx, x[ok]), (ry, y[ok])):
        for v in np.unique(src):
            m = src == v
            arr[m] = arr[m].mean()
    return float(np.corrcoef(rx, ry)[0, 1])


s42 = table("stance5_LTSB_s42", STEPS)
others = table("stance5_LTSB_s52", ["final"]) + table("stance5_LTSB_s60", ["final"]) + table("stance4_LTS_s42", ["final"]) \
    + table("stance4_LTS_s52", ["final"]) + table("stance4_LTS_s60", ["final"])
allrows = s42 + others
keys = list(allrows[0].keys())
with open(os.path.join(R, "tables", "q4_trend.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in allrows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
falls = [r["falls_full"] for r in s42]
corr = {k: spearman(falls, [r[k] for r in s42]) for k in keys if k not in ("run", "steps", "falls_full", "judge_joint_falls")}
with open(os.path.join(R, "tables", "q4_spearman_s42.json"), "w") as fh:
    json.dump({"n_checkpoints": len(s42), "steps": [r["steps"] for r in s42], "falls_full": falls,
               "spearman_vs_falls": {k: (None if not np.isfinite(v) else round(v, 3)) for k, v in corr.items()}}, fh, indent=1)
for r in allrows:
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
print(json.dumps({k: round(v, 2) for k, v in corr.items() if np.isfinite(v)}))
