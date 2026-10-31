"""Verifier: my own forward pass on the probe bank vs the probe's recorded bank_action_now (network only, no sim).
    python -B vew_probecheck.py > vew_probecheck.txt"""
import numpy as np
from vew_measure import MyNet, paths, F2
for run in ("s42", "s60", "s80"):
    with np.load(F2 / f"stance6_r10LTSB_{run}/copies/01_stance/policy_probe.npz.copy1") as d:
        t = d["update_timesteps"]; ba = d["bank_action_now"]; bank = d["bank_raw_obs"]
    for steps in (1048576, 3145728, 7340032, 11534336, 12001280):
        net = MyNet(*paths(run, steps))
        i = int(np.flatnonzero(t == steps)[0])
        mine = net.mu(bank)
        print(f"{run} {steps}: max|mine - probe row {t[i]}| all 15 actions {np.max(np.abs(mine - ba[i])):.2e}; tail_3 only {np.max(np.abs(mine[:, 14] - ba[i][:, 14])):.2e}"
              f"; vs previous row {t[i-1]} {np.max(np.abs(mine[:, 14] - ba[i-1][:, 14])):.2e}")
