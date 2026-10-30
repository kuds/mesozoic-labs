"""PHASE PROBE, POST HOC (not pre-registered): the same push measure as posthoc_push.py for two other actuated joints
that drift with the phase: tail_2_pitch (action 13, observation 5) and neck_pitch (action 0, observation 0).
Offline only (recorded rollouts; GAE recomputed; linear value fits in-sample); no policy is trained.
Per checkpoint and joint: the Gaussian-mean command by bin, the angle by bin, and the push
mean A x (a_j - mu_j) / sigma_j^2 under the run's critic (crit), a phase-aware linear value (phase) and Monte Carlo
returns minus that value (mc), with the episode-clustered bootstrap (400, seed 12345).
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PAIRS = (("tail_2_pitch", 13, 5), ("neck_pitch", 0, 0))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    A_ = load("phase_analyze")
    PP = load("posthoc_push")
    out, lines = {}, []
    for lab in A_.LABELS:
        f = HERE / "rollouts" / f"{lab}.npz"
        if not f.is_file():
            continue
        d = A_.Data(lab)
        X2 = A_.design(d, "M2")
        b2 = A_.EpOLS(X2, d.G, d.rows).solve()
        vphase = X2 @ b2
        last = np.array([r[-1] for r in d.rows])
        advs = {"crit": PP.gae(d, d.V, d.ep_vterm), "phase": PP.gae(d, vphase, vphase[last]), "mc": d.G - vphase}
        W = A_.boot_weights(d.n_ep)
        bins = [d.D[:, i] > 0 for i in range(4)]
        res = {}
        for name, act, obs in PAIRS:
            sig = math.exp(float(d.meta["log_std"][act]))
            score = (d.act[:, act] - d.mu[:, act]) / sig ** 2
            ang = np.degrees(d.raw[:, obs])
            r = {"sigma": sig, "mu_by_bin": [float(d.mu[b, act].mean()) for b in bins],
                 "angle_by_bin": [float(ang[b].mean()) for b in bins]}
            for nm, adv in advs.items():
                prod = adv * score
                sums = np.array([prod[rr].sum() for rr in d.rows])
                ns = np.array([len(rr) for rr in d.rows], dtype=float)
                bs = np.array([(w * sums).sum() / (w * ns).sum() for w in W])
                r[nm] = {"push": float(prod.mean()), "ci": A_.ci(bs)}
            res[name] = r
            lines.append(f"{lab:<5} {name:<13} mu by bin " + " ".join(f"{x:+.3f}" for x in r["mu_by_bin"])
                         + "  angle by bin " + " ".join(f"{x:+.2f}" for x in r["angle_by_bin"])
                         + "  push: " + "; ".join(f"{nm} {r[nm]['push']:+.5f} {A_.fmt_ci(r[nm]['ci'])}"
                                                  for nm in ("crit", "phase", "mc")))
        out[lab] = res
    print("\n".join(lines))
    (HERE / "posthoc_other.json").write_text(json.dumps(out, indent=1))
    (HERE / "posthoc_other.txt").write_text("POST HOC (not pre-registered)\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
