"""Noise structure of the probe's per-update bank tail_3 changes (read only): lag autocorrelation of the single-update
differences and the variance ratio Var(k-update change) / (k Var(1-update change)), per run and window."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
import eb_common as E
for run, (a0, a1) in (("s42", (0.5e6, 2.2e6)), ("s80", (0.5e6, 2.2e6)), ("s60", (5.0e6, 7.6e6)), ("s42", (2.2e6, 3.7e6))):
    pr = E.probe(run)
    t = pr["update_timesteps"]
    sel = (t >= a0) & (t <= a1)
    for name, y in (("bank_mean", pr["bank_action_now"][sel][:, :, 14].mean(1)),
                    ("statue_settled", pr["bank_action_now"][sel][:, [0, 5], 14].mean(1)),
                    ("rollout_mu_mean", pr["rollout_mu_mean"][sel, 14])):
        d = np.diff(y.astype(np.float64))
        ac = [np.corrcoef(d[:-k], d[k:])[0, 1] for k in (1, 2, 3)]
        vr = {k: np.var(y[k:] - y[:-k]) / (k * np.var(d)) for k in (4, 16, 64)}
        print(f"{run} [{a0/1e6:.1f},{a1/1e6:.1f}]M {name:15s} n={len(d)} sd1={d.std():.5f} mean1={d.mean():+.6f} "
              f"acf1..3={ac[0]:+.2f},{ac[1]:+.2f},{ac[2]:+.2f} VR4={vr[4]:.3f} VR16={vr[16]:.3f} VR64={vr[64]:.3f}")
