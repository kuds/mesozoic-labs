"""Action std (mean, min, max over the 15 actuators) at the last probe row of each run's F2 policy_probe.npz copy.

Read only; report only.  Pilot 1's parent had a mean of 0.095 (BRP D-D36 amendment of 2026-10-18).
"""
import sys
import numpy as np
root = "/home/user/work/pc_study/round/fetch/F2"
for run in sys.argv[1:]:
    p = f"{root}/{run}/copies/01_stance/policy_probe.npz.copy1"
    with np.load(p, allow_pickle=False) as z:
        t = z["update_timesteps"]
        s = z["std_per_actuator"]
        e = z["ent_coef"]
        i = int(np.argmax(t))
        row = s[i]
        print(f"{run} last_update_timesteps={int(t[i])} std_mean={row.mean():.4f} std_min={row.min():.4f} "
              f"std_max={row.max():.4f} ent_coef={float(e[i]):.6g}")
