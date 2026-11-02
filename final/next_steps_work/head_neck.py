"""Report-only: head-tip height and neck/head commands per run from the F2 copies of eval_episodes.csv (copy1)."""
import csv, math, sys
ROOT = "/home/user/work/pc_study/round/fetch/F2"
RUNS = ["stance7_r10LTSBpc_s60", "stance7_r10LTSBoff_s60", "stance7_r10LTSBpc_s42", "stance7_r10LTSBoff_s42",
        "stance7_r10LTSBpc_s80", "stance7_r10LTSBoff_s80"]
COLS = ["head_tip_height_mean_m", "neck_pitch_act_command_mean", "head_pitch_act_command_mean"]
def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None
for run in RUNS:
    path = f"{ROOT}/{run}/copies/01_stance/eval_episodes.csv.copy1"
    by_t = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(float(row["timesteps"]))
            by_t.setdefault(t, []).append(row)
    ts = sorted(by_t)
    def evmean(t, c):
        vals = [f(r[c]) for r in by_t[t]]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    out = [run, f"evals={len(ts)}", f"rows={sum(len(v) for v in by_t.values())}"]
    for c in COLS:
        per = [(t, evmean(t, c)) for t in ts]
        per = [(t, v) for t, v in per if v is not None]
        last8 = [v for _, v in per[-8:]]
        absmax_t, absmax_v = max(per, key=lambda p: abs(p[1]))
        out.append(f"{c}: last8_mean={sum(last8)/len(last8):+.3f} max|eval mean|={absmax_v:+.3f}@{absmax_t}")
    print(" | ".join(out))
