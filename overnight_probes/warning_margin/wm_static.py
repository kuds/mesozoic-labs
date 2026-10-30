"""Static (network-only, no simulation) curve for the pre-registered secondary predictor S.  Run BEFORE any nudged
closed-loop episode.  States: the early-warning's own settled states (early_warning/states/<run>_<steps>.npz, seeds
5001-5010, steps 500-1000, 60 per checkpoint).  obs[6] (tail_3 angle) is set to theta = 0..12 deg by 0.25 (the
normaliser clips it at the checkpoint's cap), everything else fixed; the clipped deterministic tail_3 command, median
over states.  Writes wm_static.txt and wm_static.json."""
from __future__ import annotations

import json
import math

import numpy as np
import torch

import wm_common as C

THETAS = np.round(np.arange(0.0, 12.0001, 0.25), 4)
T_RAIL = 5.19  # deg reached at command +1 (clamp/NOTES.txt section 1: +5.19; s80 forced per-policy medians +5.17..+5.22)

factory, _ = C.make_factory()
out = {}
lines = []
for run, steps in C.CKPTS:
    pol = C.Policy(run, steps, factory)
    X = np.load(C.SCR / f"com_revisit/notes/early_warning/states/{run}_{steps}.npz")["X"]
    curves = []
    for th in THETAS:
        Y = np.array(X, dtype=np.float64, copy=True)
        Y[:, C.I_T3] = math.radians(th)
        nb = pol.norm.normalize_obs(Y.astype(np.float32))
        with torch.no_grad():
            mu = pol.model.policy.get_distribution(torch.as_tensor(np.asarray(nb, dtype=np.float32))).distribution.mean
        curves.append(np.clip(mu.numpy()[:, C.A_T3], -1.0, 1.0))
    curves = np.array(curves)  # (n_theta, n_states)
    med = np.median(curves, axis=1)
    c_rail = float(np.interp(T_RAIL, THETAS, med))
    hit = np.nonzero(med >= 0.99)[0]
    theta_need = float(THETAS[hit[0]]) if len(hit) else None
    out[f"{run}_{steps}"] = {"n_states": int(len(X)), "t3_clip_deg": pol.t3_clip_deg, "thetas": THETAS.tolist(),
                             "median_cmd": med.round(5).tolist(), "c_at_rail_angle": c_rail,
                             "bstar_S": 1.0 - c_rail, "max_cmd_any_theta": float(med.max()),
                             "theta_need_S": theta_need,
                             "delta_star_S": (theta_need - T_RAIL) if theta_need is not None else None}
    show = {t: round(float(med[i]), 3) for i, t in enumerate(THETAS) if t in (0, 1, 2, 3, 4, 5, 5.25, 6, 7, 8, 9, 10, 12)}
    lines.append(f"{run} {steps}: n {len(X)} cap {pol.t3_clip_deg:.2f} deg | c(5.19) {c_rail:+.3f} b*_S {1 - c_rail:.3f} | "
                 f"max over theta {med.max():+.3f} | theta_need(c>=0.99) {theta_need} | curve {show}")
(C.HERE / "wm_static.json").write_text(json.dumps(out, indent=1))
(C.HERE / "wm_static.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
