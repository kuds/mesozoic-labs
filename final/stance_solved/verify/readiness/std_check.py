"""Independent re-read of the action std at the last probe row (read only)."""
import sys
import numpy as np
root = "/home/user/work/pc_study/round/fetch/F2"
for run in sys.argv[1:]:
    p = f"{root}/{run}/copies/01_stance/policy_probe.npz.copy1"
    with np.load(p, allow_pickle=False) as z:
        t = z["update_timesteps"]; s = z["std_per_actuator"]; e = z["ent_coef"]
        order = np.argsort(t)
        last = order[-1]
        print(run, int(t[last]), round(float(s[last].mean()), 4), "rows", len(t), "max_t", int(t.max()),
              "ent_last", float(e[last]), "std_first", round(float(s[order[0]].mean()), 4), "t_first", int(t[order[0]]))
