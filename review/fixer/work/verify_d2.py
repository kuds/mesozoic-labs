"""Fixer: re-derive design D2: P(SEC-MECH SUPPORTED) when the phase arm's critic slope stays at its S value."""
import itertools
import numpy as np
rng = np.random.default_rng(20261031)
splits = list(itertools.combinations(range(8), 4))
def p_control_higher(ctrl, ph):
    v = np.concatenate([ctrl, ph]); obs = ctrl.sum()
    return sum(v[list(s)].sum() >= obs - 1e-15 for s in splits) / 70
ctrl_mean = np.mean([0.0618, 0.0532, 0.0226, 0.0415]); ctrl_sd = np.std([0.0618, 0.0532, 0.0226, 0.0415], ddof=1)
print("BS2 controls' slope at E: mean %.4f SD %.4f" % (ctrl_mean, ctrl_sd))
for ph_sd in (0.017, 0.005):
    n, hit = 20000, 0
    for _ in range(n):
        c = rng.normal(ctrl_mean, ctrl_sd, 4); p = rng.normal(0.02143, ph_sd, 4)
        hit += p_control_higher(c, p) <= 0.05
    print("phase slope stays at S (+0.02143), phase SD %.3f: P(SEC-MECH SUPPORTED) = %.3f" % (ph_sd, hit / n))
