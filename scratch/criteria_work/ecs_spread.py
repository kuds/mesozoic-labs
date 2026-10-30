"""Between-continuation spread of the on-policy tail_3 mean (probe rollout_mu_mean[:, 14], mean over the last 8 rows
before E) in the ECS's finished continuations (read only; a different phase: s42 from 3,145,728, s80 from 4,194,304).
Used only as an order-of-magnitude input to the power figures of the bias screen's criteria [inferred there]."""
import statistics as st
import sys
from pathlib import Path

import numpy as np

ECS = Path(sys.argv[1])
WIN = {"s42": (3145728, 4718592), "s80": (4194304, 5767168)}
out = {}
for run in sorted((ECS / "runs").iterdir()):
    name = run.name
    if name.count("_") != 2:
        continue
    start, arm, _seed = name.split("_")
    S, E = WIN[start]
    with np.load(run / "01_stance" / "policy_probe.npz", allow_pickle=False) as z:
        u = z["update_timesteps"].astype(np.int64)
        m = z["rollout_mu_mean"][:, 14].astype(np.float64)
    want = [E - 8192 * k for k in range(8, 0, -1)]
    idx = [int(np.flatnonzero(u == w)[0]) for w in want if (u == w).sum() == 1]
    if len(idx) != 8:
        print(name, "rows", len(idx), "(stopped early?)")
        continue
    first = [int(np.flatnonzero(u == S + 8192 * k)[0]) for k in range(1, 9)]
    out.setdefault((start, arm), []).append((name, float(m[idx].mean()), float(m[first].mean())))
for key, rows in sorted(out.items()):
    last = [r[1] for r in rows]
    rise = [r[1] - r[2] for r in rows]
    print(key, "n", len(rows), "last8 mean", round(st.mean(last), 4), "SD", round(st.stdev(last), 4),
          "| rise from own first 8 rows: mean", round(st.mean(rise), 4), "SD", round(st.stdev(rise), 4))
    for r in rows:
        print("   ", r[0], round(r[1], 4), round(r[2], 4))
