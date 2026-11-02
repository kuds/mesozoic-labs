"""Report-only, read-only: per-evaluation mean commands (tail, neck, head) from each run's F2 ledger copy 1.

For each run and actuator: the largest |evaluation-mean command| and its step, the first step where it reaches
0.30 and 0.60, the mean over three windows (0-4M, 4-8M, 8M-end), and a least-squares slope over evaluations at or
after 8,000,000 steps (per 1,000,000 steps). Reads only; writes to stdout.
"""
import csv
import sys

ROOT = "/home/user/work/pc_study/round/fetch/F2"
RUNS = ["stance7_r10LTSBpc_s60", "stance7_r10LTSBpc_s42", "stance7_r10LTSBpc_s80",
        "stance7_r10LTSBoff_s60", "stance7_r10LTSBoff_s42", "stance7_r10LTSBoff_s80"]
ACTS = ["tail_3_pitch_act", "tail_2_pitch_act", "tail_1_pitch_act", "tail_1_yaw_act",
        "neck_pitch_act", "neck_yaw_act", "head_pitch_act"]


def slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx if sxx else float("nan")


for run in RUNS:
    path = f"{ROOT}/{run}/copies/01_stance/eval_episodes.csv.copy1"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_t = {}
    for r in rows:
        by_t.setdefault(int(float(r["timesteps"])), []).append(r)
    steps = sorted(by_t)
    print(f"{run}: {len(rows)} episode rows, {len(steps)} evaluations, {steps[0]}..{steps[-1]}")
    for a in ACTS:
        col = f"{a}_command_mean"
        series = []
        for t in steps:
            vals = [float(r[col]) for r in by_t[t] if r.get(col, "") not in ("", "nan")]
            if vals:
                series.append((t, sum(vals) / len(vals)))
        mx_t, mx_v = max(series, key=lambda p: abs(p[1]))
        f30 = next((t for t, v in series if abs(v) >= 0.30), None)
        f60 = next((t for t, v in series if abs(v) >= 0.60), None)
        w = []
        for lo, hi in ((0, 4_000_000), (4_000_000, 8_000_000), (8_000_000, 10**9)):
            vs = [v for t, v in series if lo <= t < hi]
            w.append(sum(vs) / len(vs) if vs else float("nan"))
        late = [(t / 1e6, v) for t, v in series if t >= 8_000_000]
        sl = slope([p[0] for p in late], [p[1] for p in late])
        print(f"  {a:18s} max|cmd| {mx_v:+.3f} at {mx_t}; first |cmd|>=0.30 {f30}; >=0.60 {f60}; "
              f"window means {w[0]:+.3f} / {w[1]:+.3f} / {w[2]:+.3f}; slope >=8M {sl:+.4f} per 1M ({len(late)} evals)")
sys.stdout.flush()
