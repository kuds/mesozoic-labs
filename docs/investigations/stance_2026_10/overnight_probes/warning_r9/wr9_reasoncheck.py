"""r10 only: does 'screen_v2_reasons lists max_actuator_saturation_fraction' equal saturated_actuator_fraction > 0.10?"""
import numpy as np, pandas as pd
F2 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2"
for r in ("s42", "s60", "s80"):
    d = pd.read_csv(f"{F2}/stance6_r10LTSB_{r}/copies/01_stance/eval_episodes.csv.copy1",
                    usecols=["timesteps", "screen_v2_reasons", "saturated_actuator_fraction", "saturated_actuator"])
    a = d["screen_v2_reasons"].fillna("").str.contains("max_actuator_saturation_fraction")
    b = d["saturated_actuator_fraction"] > 0.10
    c = d["saturated_actuator_fraction"] >= 0.10
    print(f"{r}: episodes {len(d)}; reason {int(a.sum())}; frac>0.10 {int(b.sum())}; frac>=0.10 {int(c.sum())}; "
          f"disagree(>) {int((a != b).sum())}; disagree(>=) {int((a != c).sum())}; "
          f"actuators of reason episodes {sorted(set(d.loc[a, 'saturated_actuator'].astype(str)))}")
    if (a != b).any():
        print(d.loc[a != b, ["timesteps", "saturated_actuator_fraction", "saturated_actuator"]].head(10).to_string())
