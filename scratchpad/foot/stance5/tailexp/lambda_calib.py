"""TRACK TAIL (scratch): pick ONE lambda for arm P1 from the policy-gradient scale on tail_3's mean at the 4,194,304
checkpoint of stance5_LTSB_s42.

    S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
    T=$S/foot/stance5/tailexp
    env ISO_WT=$S/fix/scratch_tailfix nice -n 19 python3 $S/foot/stance3/tools/iso.py $T/lambda_calib.py --rollouts 4

The model and its training env come from run_continuation.build (the repository's resume path, as the arms train).
R rollouts of 8,192 steps are collected with SB3's own collect_rollouts (stochastic policy, VecNormalize updating, GAE
computed), with NO update in between (the parameters stay the checkpoint's).  For each rollout, one pass of 64
mini-batches of 128 (rollout_buffer.get, as train() draws them) at ratio 1, the first epoch's situation:

  G_pg   = d(clipped surrogate, advantages normalised per mini-batch) / d(action_net.bias[tail_3])
         = sum over the mini-batch's rows of dL/dmu_b,tail_3, the pull on tail_3's mean as a parameter;
  G_kl   = the same for kappa * the mirror loss (the other term that reaches the actor's mean);
  |g|    = the norm of the full loss's gradient (pg + ent_coef * entropy + vf_coef * mirrored value loss + kappa * KL),
           against max_grad_norm 0.5 (the clip scales every term alike, so it does not change the ratio below);
  G_pen  = d(L_B) / d(bias[tail_3]) for lambda = 1 on the actual rows, L_B = mean over rows and the 15 actuators of
           relu(|mu| - 1)^2.

On a mini-batch whose tail_3 means all sit at +1.11 the penalty's pull is lambda * 2 * 0.11 / 15 (independent of the
batch size).  Rule: lambda = RMS(G_pg) / (2 * 0.11 / 15), rounded to two significant figures, i.e. the penalty's
consistent pull on a mean parked at +1.11 equals the typical size of the policy-gradient pull on that mean.  With
Adam, whose step normalises each parameter by its gradient RMS, that makes the bound's drift on tail_3's bias the same
order as the step size, against a policy gradient whose mean on it is (expected) far below its RMS.  Writes
tailexp/lambda.json (refused if it exists) and prints the numbers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

TAILEXP = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/tailexp"
REF_MU = 1.11
N_ACT = 15


def load_rc():
    spec = importlib.util.spec_from_file_location("run_continuation", os.path.join(TAILEXP, "run_continuation.py"))
    rc = importlib.util.module_from_spec(spec)
    sys.modules["run_continuation"] = rc
    spec.loader.exec_module(rc)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(TAILEXP, "lambda.json"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    rc = load_rc()
    git, origin = rc.check_isolation(False)
    import numpy as np
    import torch
    from torch.nn import functional as F

    from environments.shared import mirror_kl_ppo as MK

    t0 = time.monotonic()
    built = rc.build(out_dir=None)
    model = built["model"]
    policy = model.policy
    tail = rc.TAIL3
    bias = policy.action_net.bias
    params = [p for p in policy.parameters() if p.requires_grad]
    state = model._mirror_state()
    _ = model._mirror_begin_update()  # builds the mirror tensors
    tens = state.tensors
    from stable_baselines3.common.logger import Logger

    model.set_logger(Logger(folder=None, output_formats=[]))  # the checkpoint names a Colab TensorBoard dir; log nothing
    total, callback = model._setup_learn(built["checks"]["run_timesteps"] - rc.START_STEPS, None, False, "calib", False)
    model._mirror_install()
    clip_range = float(model.clip_range(model._current_progress_remaining))
    rows = []
    per_rollout = []
    for r in range(a.rollouts):
        ok = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
        assert ok
        buf_mu14 = []
        stats = {"G_pg": [], "G_kl": [], "G_pen1": [], "grad_norm": [], "frac_pinned": [], "mu14_mean": [],
                 "G_pg_pinned_rows": [], "G_pen1_pinned_rows": [], "dL_dmu14_row_abs_mean": []}
        for data in model.rollout_buffer.get(model.batch_size):
            obs, actions = data.observations, data.actions
            adv = data.advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            dist = policy.get_distribution(obs)
            mu = dist.distribution.mean
            log_prob = dist.log_prob(actions)
            ratio = torch.exp(log_prob - data.old_log_prob)
            pg = -torch.min(adv * ratio, adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
            entropy = dist.entropy()
            values = policy.predict_values(obs).flatten()
            obs_m = obs[:, tens["obs_perm"]] * tens["obs_sign"]
            value_loss = 0.5 * (F.mse_loss(data.returns, values) + F.mse_loss(data.returns, policy.predict_values(obs_m).flatten()))
            mu_m = policy.get_distribution(obs_m).mode()
            kl, _ = MK.mirror_kl_loss(mu_m, mu.detach(), policy.log_std, tens["act_perm"], tens["act_sign"],
                                      state.config.sigma_floor)
            pen = MK.mean_bound_penalty(mu, 1.0)
            loss = pg + float(model.ent_coef) * (-entropy.mean()) + model.vf_coef * value_loss + state.config.kappa * kl
            g_pg_mu = torch.autograd.grad(pg, mu, retain_graph=True)[0][:, tail]
            g_pg_b = torch.autograd.grad(pg, bias, retain_graph=True)[0][tail].item()
            g_kl_b = torch.autograd.grad(state.config.kappa * kl, bias, retain_graph=True)[0][tail].item()
            g_pen_mu = torch.autograd.grad(pen, mu, retain_graph=True)[0][:, tail]
            g_pen_b = torch.autograd.grad(pen, bias, retain_graph=True)[0][tail].item()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            gnorm = math.sqrt(sum(float((g ** 2).sum()) for g in grads if g is not None))
            mu14 = mu.detach()[:, tail]
            pinned = mu14 > 1.0
            stats["G_pg"].append(g_pg_b)
            stats["G_kl"].append(g_kl_b)
            stats["G_pen1"].append(g_pen_b)
            stats["grad_norm"].append(gnorm)
            stats["frac_pinned"].append(float(pinned.float().mean()))
            stats["mu14_mean"].append(float(mu14.mean()))
            stats["G_pg_pinned_rows"].append(float(g_pg_mu[pinned].sum()))
            stats["G_pen1_pinned_rows"].append(float(g_pen_mu[pinned].sum()))
            stats["dL_dmu14_row_abs_mean"].append(float(g_pg_mu.abs().mean()))
            buf_mu14.append(mu14.numpy())
            assert abs(g_pg_b - float(g_pg_mu.sum())) <= 1e-5 * max(1.0, abs(g_pg_b)), (g_pg_b, float(g_pg_mu.sum()))
        allmu = np.concatenate(buf_mu14)
        acts = model.rollout_buffer.actions.reshape(-1, N_ACT)[:, tail]
        per_rollout.append({"rollout": r, "num_timesteps": int(model.num_timesteps), "minibatches": len(stats["G_pg"]),
                            "mu14_mean": float(allmu.mean()), "mu14_frac_gt1": float((allmu > 1).mean()),
                            "mu14_mean_on_gt1": float(allmu[allmu > 1].mean()) if (allmu > 1).any() else None,
                            "mu14_q": np.quantile(allmu, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                            "sampled_tail3_ge_1": float((acts >= 1.0).mean()),
                            "G_pg_rms": float(np.sqrt(np.mean(np.square(stats["G_pg"])))),
                            "G_pg_mean": float(np.mean(stats["G_pg"]))})
        rows.append(stats)
        print(json.dumps(per_rollout[-1]), flush=True)
    cat = {k: np.concatenate([np.asarray(s[k], dtype=np.float64) for s in rows]) for k in rows[0]}
    n = len(cat["G_pg"])
    g_rms = float(np.sqrt(np.mean(cat["G_pg"] ** 2)))
    g_mean = float(np.mean(cat["G_pg"]))
    g_sem = float(np.std(cat["G_pg"], ddof=1) / math.sqrt(n))
    unit = 2.0 * (REF_MU - 1.0) / N_ACT  # d L_B / d bias_tail3 on a batch all at +1.11, lambda = 1
    raw = g_rms / unit
    lam = float(f"{raw:.2g}")
    sigma = float(np.exp(policy.log_std.detach().numpy()[tail]))
    B = model.batch_size
    theory_rms = 1.0 / (sigma * math.sqrt(B))
    out = {
        "experiment": "TRACK TAIL (scratch, maintainer decision 2026-10-22)",
        "lambda": lam,
        "rule": ("lambda = RMS over mini-batches of G_pg / (2 * (1.11 - 1) / 15), two significant figures: on a "
                 "mini-batch whose tail_3 means all sit at +1.11 the penalty's pull on tail_3's output bias, "
                 "lambda * 2 * 0.11 / 15, equals the RMS policy-gradient pull on it at the 4,194,304 checkpoint"),
        "numbers": {
            "minibatches": n, "batch_size": B, "rollouts": a.rollouts, "clip_range": clip_range,
            "G_pg_rms": g_rms, "G_pg_mean": g_mean, "G_pg_sem": g_sem,
            "G_pg_abs_median": float(np.median(np.abs(cat["G_pg"]))),
            "G_pg_mean_over_rms": g_mean / g_rms,
            "G_kl_rms": float(np.sqrt(np.mean(cat["G_kl"] ** 2))),
            "penalty_pull_per_lambda_at_1p11": unit,
            "lambda_unrounded": raw,
            "penalty_pull_at_1p11_with_lambda": lam * unit,
            "theory_G_pg_rms_1_over_sigma_sqrtB": theory_rms,
            "tail3_sigma": sigma,
            "per_row_dL_dmu14_abs_mean": float(np.mean(cat["dL_dmu14_row_abs_mean"])),
            "per_row_penalty_at_1p11_with_lambda_over_B": lam * unit / B,
            "frac_rows_mu14_gt1_mean": float(np.mean(cat["frac_pinned"])),
            "G_pg_pinned_rows_rms": float(np.sqrt(np.mean(cat["G_pg_pinned_rows"] ** 2))),
            "G_pen_pinned_rows_with_lambda_mean": float(lam * np.mean(cat["G_pen1_pinned_rows"])),
            "G_pen_actual_with_lambda_mean": float(lam * np.mean(cat["G_pen1"])),
            "grad_norm_median": float(np.median(cat["grad_norm"])),
            "grad_norm_q10_q90": np.quantile(cat["grad_norm"], [0.1, 0.9]).tolist(),
            "clip_factor_median": float(np.median(np.minimum(1.0, 0.5 / cat["grad_norm"]))),
            "learning_rate_at_start": float(model.lr_schedule(model._current_progress_remaining)),
        },
        "per_rollout": per_rollout,
        "notes": [
            "G_pg is d/d bias[tail_3] of the clipped surrogate at ratio 1 (the first epoch); later epochs move the "
            "ratio within the clip and early-stop at approx_kl > 0.045 (target_kl 0.03).",
            "Parameters stay the checkpoint's through all rollouts (no update); the VecNormalize statistics update "
            "as in training, and the envs continue their episodes across rollouts (resets only at episode ends).",
            "The penalty touches every actuator, but only means beyond +-1 pay; tail_3 is the only one the "
            "4,194,304 rollouts put there in numbers (see mu beyond fractions in the arms' telemetry).",
        ],
        "clone": rc.CLONE, "git": git, "modules_environments": origin, "start_model": rc.START_ZIP,
        "start_model_sha256": rc.sha256_file(rc.START_ZIP), "seconds": round(time.monotonic() - t0, 1),
    }
    with open(a.out, "x") as fh:
        json.dump(rc.norm(out), fh, indent=1)
    print(json.dumps({"lambda": lam, **{k: out["numbers"][k] for k in ("G_pg_rms", "G_pg_mean", "G_pg_sem",
                                                                      "lambda_unrounded", "tail3_sigma",
                                                                      "theory_G_pg_rms_1_over_sigma_sqrtB",
                                                                      "frac_rows_mu14_gt1_mean", "grad_norm_median")}}),
          flush=True)
    rc.modules_under_clone()
    return 0


if __name__ == "__main__":
    sys.exit(main())
