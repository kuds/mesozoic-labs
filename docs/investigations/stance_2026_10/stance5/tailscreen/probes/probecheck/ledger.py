"""Per evaluation: episodes whose screen_v2_reasons include max_actuator_saturation_fraction (ledger eval_episodes.csv)."""
import csv, sys
from collections import defaultdict
S = sys.argv[1]
paths = {r: f"{S}/runs/stance5_LTSB_{r}/01_stance/eval_episodes.csv" for r in ("s42", "s52", "s60")}
paths.update({r: f"{S}/replication/runs/stance5_LTSB_{r}/01_stance/eval_episodes.csv" for r in ("s70", "s80")})
for run, p in paths.items():
    sat = defaultdict(int); n = defaultdict(int)
    for row in csv.DictReader(open(p)):
        t = int(row["timesteps"]); n[t] += 1
        if "max_actuator_saturation_fraction" in (row["screen_v2_reasons"] or ""):
            sat[t] += 1
    ts = sorted(n)
    flagged = [(t, sat[t]) for t in ts if sat[t]]
    print(f"== {run}: {len(ts)} evaluations {ts[0]:,}..{ts[-1]:,}; evaluations with any saturation episode: {len(flagged)}; with >=5: {sum(1 for t,c in flagged if c>=5)}")
    print("   ", [(f"{t:,}", c) for t, c in flagged][:60])
