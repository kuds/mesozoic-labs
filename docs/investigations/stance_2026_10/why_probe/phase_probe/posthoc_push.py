"""PHASE PROBE, POST HOC (not pre-registered): the policy-gradient push on the tail_3 command mean under different value
baselines, from the recorded rollouts.  Offline only: GAE is recomputed from recorded rewards and values; the
alternative value functions are linear fits on the same rollouts (in-sample); no policy is trained or updated.

push_X = mean over states of A_X(s_k) * (a14_k - mu14_k) / sigma14^2  (the score-function gradient of the Gaussian mean
of action 14, per unit advantage; SB3's per-minibatch advantage normalisation rescales it but keeps its sign),
with the episode-clustered bootstrap (400, seed 12345).  Advantages:
  crit      GAE(0.98, 0.95) with the run's own (effective) critic V and SB3's truncation bootstrap V(terminal);
  phase     GAE with V_phase = the M2 linear fit of G on Z + bins + settle ramp (sees the phase);
  noq       GAE with V_noq = a linear fit of G on Z without the tail_3 column (q3-blind, phase-blind);
  lin       GAE with V_lin = the M0 linear fit of G on Z (phase-blind, sees q3);
  mc        G - V_phase (Monte Carlo returns: what the realised rewards say).
The linear fits' value at a truncated episode's terminal state is taken as their value at the last state.
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
LAM = 0.95


def load_analyze():
    spec = importlib.util.spec_from_file_location("phase_analyze", HERE / "phase_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_analyze"] = mod
    spec.loader.exec_module(mod)
    return mod


def gae(d, Vs, vterm):
    A = np.zeros(len(d.k))
    g = d.gamma
    for j, r in enumerate(d.rows):
        T = len(r)
        nxt_adv = 0.0
        for t in range(T - 1, -1, -1):
            i = r[t]
            if t == T - 1:
                vn = vterm[j] if d.ep_trunc[j] else 0.0
            else:
                vn = Vs[r[t + 1]]
            delta = d.rn[i] + g * vn - Vs[i]
            nxt_adv = delta + g * LAM * nxt_adv
            A[i] = nxt_adv
    return A


def main():
    A_ = load_analyze()
    labels = sys.argv[1].split(",") if len(sys.argv) > 1 else [x for x in A_.LABELS if x != "N80"]
    out, lines = {}, []
    W = None
    for lab in labels:
        f = HERE / "rollouts" / f"{lab}.npz"
        if not f.is_file():
            continue
        d = A_.Data(lab)
        sig = math.exp(float(d.meta["log_std"][14]))
        eps = d.act[:, 14] - d.mu[:, 14]
        score = eps / sig ** 2
        X0 = A_.design(d, "M0")
        X2 = A_.design(d, "M2")
        keep = [c for c in range(X0.shape[1]) if c != 1 + d.zq]
        Xn = X0[:, keep]
        fits = {}
        for nm, X in (("phase", X2), ("noq", Xn), ("lin", X0)):
            b = A_.EpOLS(X, d.G, d.rows).solve()
            fits[nm] = X @ b
        last = np.array([r[-1] for r in d.rows])
        advs = {"crit": gae(d, d.V, d.ep_vterm)}
        for nm in ("phase", "noq", "lin"):
            advs[nm] = gae(d, fits[nm], fits[nm][last])
        advs["mc"] = d.G - fits["phase"]
        W = A_.boot_weights(d.n_ep)
        res = {"sigma14": sig, "n": int(len(d.k))}
        for nm, adv in advs.items():
            prod = adv * score
            sums = np.array([prod[r].sum() for r in d.rows])
            ns = np.array([len(r) for r in d.rows], dtype=float)
            est = float(prod.mean())
            bs = np.array([(w * sums).sum() / (w * ns).sum() for w in W])
            # the same per unit-variance advantage (SB3 normalises advantages): est / sd(adv)
            res[nm] = {"push": est, "ci": A_.ci(bs), "push_per_sd_adv": est / float(adv.std()),
                       "adv_sd": float(adv.std())}
            # settled-only and pre-settle-only pushes
            for part, msk in (("settled", d.k >= 200), ("pre", d.k < 200)):
                res[nm][f"push_{part}"] = float(prod[msk].mean())
        res["q3_coef_per_deg"] = {
            "critic_fd_all": float(d.fd.mean()),
            "V_phase_fit": A_.per_degree(np.linalg.lstsq(X2, fits["phase"], rcond=None)[0][1 + d.zq], d.sd_q),
            "V_lin_fit": A_.per_degree(np.linalg.lstsq(X0, fits["lin"], rcond=None)[0][1 + d.zq], d.sd_q)}
        out[lab] = res
        lines.append(f"{lab}: sigma14 {sig:.4f}; tail_3 push (mean A x (a14-mu14)/sigma^2, 95% CI; per SD of A):")
        for nm in ("crit", "lin", "noq", "phase", "mc"):
            r = res[nm]
            lines.append(f"    {nm:<5} {r['push']:+.5f} {A_.fmt_ci(r['ci'])}  per-SD {r['push_per_sd_adv']:+.4f}  "
                         f"(settled {r['push_settled']:+.5f}, pre-settle {r['push_pre']:+.5f})")
        print("\n".join(lines[-6:]), flush=True)
    (HERE / "posthoc_push.json").write_text(json.dumps(out, indent=1))
    (HERE / "posthoc_push.txt").write_text("POST HOC (not pre-registered)\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
