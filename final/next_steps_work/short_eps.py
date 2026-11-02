"""Report-only: episodes shorter than the 1000-step horizon in each run's F2 eval_episodes.csv copy1."""
import csv
ROOT = "/home/user/work/pc_study/round/fetch/F2"
for run in ["stance7_r10LTSBpc_s60", "stance7_r10LTSBpc_s42", "stance7_r10LTSBpc_s80",
            "stance7_r10LTSBoff_s60", "stance7_r10LTSBoff_s42", "stance7_r10LTSBoff_s80"]:
    rows = list(csv.DictReader(open(f"{ROOT}/{run}/copies/01_stance/eval_episodes.csv.copy1", newline="")))
    short = [(int(float(r["timesteps"])), int(float(r["length"]))) for r in rows if int(float(r["length"])) < 1000]
    print(run, "short episodes:", len(short), "of", len(rows), short)
