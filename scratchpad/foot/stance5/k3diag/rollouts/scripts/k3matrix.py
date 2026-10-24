"""K3 ablation matrix (REPORT ONLY): falls of 20 per checkpoint and noise condition, from tables/conditions.csv.

    python3 k3matrix.py ROLLOUTS_DIR
"""
import csv
import os
import sys

R = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(R, "tables", "conditions.csv"))))
CONDS = ["full", "noise0", "legs", "tail", "neckhead", "notail", "hip_pitch", "hip_roll", "knee", "ankle", "legs_no_hippitch",
         "legs_no_hiproll", "legs_no_knee", "legs_no_ankle", "hp_sym", "hp_anti", "t3hold0_full", "t3hold0_noise0"]
ORDER = [("stance5_LTSB_s42", s) for s in ("4194304", "5242880", "6291456", "7340032", "8388608", "10485760", "final")] + \
        [("stance5_LTSB_s52", "final"), ("stance5_LTSB_s60", "final"), ("stance4_LTS_s42", "final"), ("stance4_LTS_s52", "final"),
         ("stance4_LTS_s60", "final")]
idx = {(r["run"], r["steps"], r["cond"]): r for r in rows}
out = []
for run, st in ORDER:
    rec = {"run": run, "steps": st}
    for c in CONDS:
        r = idx.get((run, st, c))
        rec[c] = "" if r is None else r["falls"]
    out.append(rec)
with open(os.path.join(R, "tables", "ablation_matrix.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["run", "steps"] + CONDS)
    w.writeheader()
    w.writerows(out)
print("run steps | " + " ".join(f"{c[:9]:>9s}" for c in CONDS))
for rec in out:
    print(f"{rec['run'][:16]:16s} {rec['steps']:>8s} | " + " ".join(f"{rec[c]:>9s}" for c in CONDS))
