"""List the arrays in each PHASE/OFF run's F2 policy_probe.npz copy (read only)."""
import sys
import numpy as np
root = "/home/user/work/pc_study/round/fetch/F2"
for run in sys.argv[1:]:
    p = f"{root}/{run}/copies/01_stance/policy_probe.npz.copy1"
    with np.load(p, allow_pickle=False) as z:
        print(run, {k: z[k].shape for k in z.files})
