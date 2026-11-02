"""Read-only: per run, every *_command_mean column: max over evaluations of the evaluation mean (signed) and of
the evaluation mean of |episode mean|; saturated_actuator_fraction max; head_pitch first >=0.30 / >=0.60."""
import csv
ROOT = "/home/user/work/pc_study/round/fetch/F2"
RUNS = ["stance7_r10LTSBpc_s60", "stance7_r10LTSBpc_s42", "stance7_r10LTSBpc_s80",
        "stance7_r10LTSBoff_s60", "stance7_r10LTSBoff_s42", "stance7_r10LTSBoff_s80"]
for run in RUNS:
    rows = list(csv.DictReader(open(f"{ROOT}/{run}/copies/01_stance/eval_episodes.csv.copy1", newline="")))
    cols = [c for c in rows[0] if c.endswith("_command_mean")]
    by = {}
    for r in rows:
        by.setdefault(int(float(r["timesteps"])), []).append(r)
    steps = sorted(by)
    out = []
    for c in cols:
        sm, am = [], []
        for t in steps:
            v = [float(r[c]) for r in by[t] if r[c] not in ("", "nan")]
            if v:
                sm.append((t, sum(v) / len(v)))
                am.append((t, sum(abs(x) for x in v) / len(v)))
        ms = max(sm, key=lambda p: abs(p[1]))
        ma = max(am, key=lambda p: p[1])
        f30 = next((t for t, v in sm if abs(v) >= 0.30), None)
        f60 = next((t for t, v in sm if abs(v) >= 0.60), None)
        out.append((ma[1], c, ms, ma, f30, f60))
    out.sort(reverse=True)
    sat = max(float(r["saturated_actuator_fraction"] or 0) for r in rows)
    print(f"{run}: evals {len(steps)}; max saturated_actuator_fraction {sat:.3f}")
    for ma1, c, ms, ma, f30, f60 in out[:5]:
        print(f"   {c:32s} max|evalmean| {ms[1]:+.3f}@{ms[0]}  max evalmean|ep| {ma[1]:.3f}@{ma[0]}  first>=.30 {f30} >=.60 {f60}")
