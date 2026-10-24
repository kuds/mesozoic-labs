"""Tail_3 bound screen: V2, the coef calibration (BUILD_SPEC section 7).  NO optimizer step, NO parameter change.

    PY -I SCREEN/tools/calib_tailscreen.py [--out PATH]

Writes SCREEN/calib/coef.json once (refused if it exists, or if anything exists under SCREEN/runs/).  --out is for a
dry run only and must lie outside SCREEN/calib.

Procedure (P1's lambda_calib.py, without its all-actuator mean_bound_penalty and its hard-coded seed), per start state
(s42, s80):
  1. run_tailscreen.build(state, rep_seed=101, out_dir=None), the logger set to Logger(folder=None, output_formats=[]);
  2. _setup_learn(12,000,000 - 4,194,304, reset_num_timesteps=False) and _mirror_install();
  3. clip_range at the current progress;
  4. the digest of the parameters and the Adam state;
  5. four times: collect_rollouts (SB3's own: stochastic policy, VecNormalize statistics updating in memory, GAE),
     then for each of the 64 mini-batches of rollout_buffer.get(128), at ratio 1 with the advantages normalised as SB3
     2.9.0 train() does:
        G_pg      = d(clipped surrogate) / d(action_net.bias[14])
        G_pen[b]  = d(tail3_upper_bound_penalty(mu, b)) / d(action_net.bias[14]),   b in (0.90, 0.60)
        share[b]  = mean(mu[:, 14] > b)
     plus mu14 quantiles, sigma14 and the norm of the stock loss's full gradient (as P1);  gradients come from
     torch.autograd.grad and are discarded;
  6. the parameter/Adam digest is asserted unchanged.
Rule: rms = sqrt(mean(G_pg^2)) over the 512 pooled mini-batches; COEF_b = float(format(rms / (2 x 0.11), ".2g"));
realised_pull[state][b] = COEF_b x mean(G_pen[b]) over that state's 256 mini-batches.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOLS = Path(os.path.realpath(__file__)).parent


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(TOOLS / f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RT = _load("run_tailscreen")
C = RT.C
BOUNDS = (0.90, 0.60)
CALIB_SEED = 101
ROLLOUTS = 4
BATCH = 128
DEFAULT_OUT = Path(C.SCREEN) / "calib" / "coef.json"
RULE = ("COEF_b = RMS(G_pg)/(2*0.11) over 512 pooled mini-batches, format '.2g'; G_pg = d clipped surrogate "
        "(ratio 1, per-minibatch normalised advantages) / d action_net.bias[14]")


def _bkey(b: float) -> str:
    return f"{b:.2f}"


def refusal(out: Path) -> str | None:
    calib_dir = os.path.realpath(DEFAULT_OUT.parent)
    real_out = os.path.realpath(out)
    if real_out != os.path.realpath(DEFAULT_OUT) and (real_out + os.sep).startswith(calib_dir + os.sep):
        return f"--out {out} lies inside {DEFAULT_OUT.parent}; only the default may"
    if os.path.lexists(out):
        return f"{out} exists (coef.json is written once)"
    runs = Path(C.SCREEN) / "runs"
    if runs.is_dir() and any(runs.iterdir()):
        return f"{runs} is not empty: coef.json must be written before any training step"
    return None


def params_adam_digest(model: Any) -> dict:
    full = RT.state_digest(model, model.get_vec_normalize_env())
    return {"params": full["params"], "adam": full["adam"], "n_updates": full["n_updates"]}


def calibrate_state(state: str) -> tuple[dict, dict]:
    """One start state: the per-state record (no coef yet) and the raw per-mini-batch arrays."""
    import numpy as np
    import torch
    from stable_baselines3.common.logger import Logger
    from torch.nn import functional as F

    from environments.shared import mirror_kl_ppo as MK

    built = RT.build(state, CALIB_SEED, None)
    model = built["model"]
    policy = model.policy
    tail = C.TAIL3
    bias = policy.action_net.bias
    params = [p for p in policy.parameters() if p.requires_grad]
    assert model._tail3_bound is None  # the calibration measures the stock gradients
    model.set_logger(Logger(folder=None, output_formats=[]))  # the checkpoint names a Colab TensorBoard dir
    _, callback = model._setup_learn(RT.TOTAL_TIMESTEPS - C.START_STEPS, None, False, "calib", False)
    model._mirror_install()
    mirror = model._mirror_state()
    _ = model._mirror_begin_update()  # builds the mirror tensors (no update follows)
    tens = mirror.tensors
    clip_range = float(model.clip_range(model._current_progress_remaining))
    learning_rate = float(model.lr_schedule(model._current_progress_remaining))
    digest_before = params_adam_digest(model)
    if model.batch_size != BATCH or model.n_steps * model.n_envs != C.ROLLOUT:
        raise SystemExit(f"batch {model.batch_size} / rollout {model.n_steps * model.n_envs} != {BATCH} / {C.ROLLOUT}")
    raw: dict[str, list] = {"G_pg": [], "grad_norm": [], **{f"G_pen_{_bkey(b)}": [] for b in BOUNDS},
                            **{f"share_{_bkey(b)}": [] for b in BOUNDS}}
    per_rollout = []
    for r in range(ROLLOUTS):
        ok = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
        assert ok, "collect_rollouts stopped early"
        mu14_all = []
        g_pg_rollout = []
        for data in model.rollout_buffer.get(BATCH):
            obs, actions = data.observations, data.actions
            adv = data.advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # SB3 2.9.0 train(), normalize_advantage
            dist = policy.get_distribution(obs)
            mu = dist.distribution.mean
            log_prob = dist.log_prob(actions)
            ratio = torch.exp(log_prob - data.old_log_prob)
            pg = -torch.min(adv * ratio, adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
            entropy = dist.entropy()
            values = policy.predict_values(obs).flatten()
            obs_m = obs[:, tens["obs_perm"]] * tens["obs_sign"]
            value_loss = 0.5 * (F.mse_loss(data.returns, values)
                                + F.mse_loss(data.returns, policy.predict_values(obs_m).flatten()))
            mu_m = policy.get_distribution(obs_m).mode()
            kl, _ = MK.mirror_kl_loss(mu_m, mu.detach(), policy.log_std, tens["act_perm"], tens["act_sign"],
                                      mirror.config.sigma_floor)
            loss = pg + float(model.ent_coef) * (-entropy.mean()) + model.vf_coef * value_loss + mirror.config.kappa * kl
            g_pg_mu = torch.autograd.grad(pg, mu, retain_graph=True)[0][:, tail]
            g_pg = torch.autograd.grad(pg, bias, retain_graph=True)[0][tail].item()
            assert abs(g_pg - float(g_pg_mu.sum())) <= 1e-5 * max(1.0, abs(g_pg)), (g_pg, float(g_pg_mu.sum()))
            mu14 = mu.detach()[:, tail]
            for b in BOUNDS:
                g_pen = torch.autograd.grad(MK.tail3_upper_bound_penalty(mu, b), bias, retain_graph=True)[0]
                assert torch.all(g_pen[:tail] == 0.0)  # scope: only bias[14] is pulled
                raw[f"G_pen_{_bkey(b)}"].append(g_pen[tail].item())
                raw[f"share_{_bkey(b)}"].append(float((mu14 > b).float().mean()))
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            raw["grad_norm"].append(math.sqrt(sum(float((g**2).sum()) for g in grads if g is not None)))
            raw["G_pg"].append(g_pg)
            g_pg_rollout.append(g_pg)
            mu14_all.append(mu14.numpy().astype(np.float64))
        allmu = np.concatenate(mu14_all)
        acts = model.rollout_buffer.actions.reshape(-1, model.action_space.shape[0])[:, tail]
        per_rollout.append({
            "rollout": r, "num_timesteps": int(model.num_timesteps), "minibatches": len(g_pg_rollout),
            "mu14_mean": float(allmu.mean()), "mu14_q": np.quantile(allmu, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
            "mu14_frac_gt": {_bkey(b): float((allmu > b).mean()) for b in (*BOUNDS, 1.0)},
            "sampled_tail3_ge_1": float((acts >= 1.0).mean()),
            "G_pg_rms": float(np.sqrt(np.mean(np.square(g_pg_rollout)))), "G_pg_mean": float(np.mean(g_pg_rollout)),
        })
        print(json.dumps({"state": state, **per_rollout[-1]}), flush=True)
    digest_after = params_adam_digest(model)
    assert digest_after["params"] == digest_before["params"] and digest_after["adam"] == digest_before["adam"], (
        "the calibration changed the parameters or the Adam state"
    )
    g = np.asarray(raw["G_pg"], dtype=np.float64)
    n = len(g)
    if n != ROLLOUTS * (C.ROLLOUT // BATCH):
        raise SystemExit(f"{state}: {n} mini-batches, expected {ROLLOUTS * (C.ROLLOUT // BATCH)}")
    sigma = float(np.exp(policy.log_std.detach().numpy()[tail]))
    record = {
        "zip_sha256": RT.sha256_file(built["start_zip"]), "pkl_sha256": RT.sha256_file(built["start_pkl"]),
        "rep_seed": CALIB_SEED, "minibatches": n, "clip_range": clip_range, "learning_rate": learning_rate,
        "G_pg_rms": float(np.sqrt(np.mean(g**2))), "G_pg_mean": float(np.mean(g)),
        "G_pg_sem": float(np.std(g, ddof=1) / math.sqrt(n)), "G_pg_abs_median": float(np.median(np.abs(g))),
        "tail3_sigma": sigma, "theory_rms": 1.0 / (sigma * math.sqrt(BATCH)),
        "grad_norm_median": float(np.median(raw["grad_norm"])),
        "digest_before": digest_before, "digest_after": digest_after, "per_rollout": per_rollout,
    }
    model.env.close()
    return record, raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="dry run only: a path outside SCREEN/calib")
    a = ap.parse_args(argv)
    out = Path(a.out)
    why = refusal(out)
    if why:
        print(json.dumps({"calib": "refused", "reason": why}), flush=True)
        return 2
    problems = C.check_stack()
    if problems:
        print(json.dumps({"calib": "refused", "reason": f"stack: {problems}"}), flush=True)
        return 2
    t0 = time.monotonic()
    utc = RT.utc_now()
    RT.bootstrap_clone()
    import numpy as np

    git = RT.clone_state(C.CLONE)
    states: dict[str, dict] = {}
    raws: dict[str, dict] = {}
    for state in ("s42", "s80"):
        states[state], raws[state] = calibrate_state(state)
    pooled = np.concatenate([np.asarray(raws[s]["G_pg"], dtype=np.float64) for s in ("s42", "s80")])
    pooled_rms = float(np.sqrt(np.mean(pooled**2)))
    coefs = {b: RT.coef_by_rule(pooled_rms) for b in BOUNDS}
    for state, record in states.items():
        g_mean = record["G_pg_mean"]
        record["per_bound"] = {}
        for b in BOUNDS:
            pen = float(np.mean(raws[state][f"G_pen_{_bkey(b)}"]))
            pull = coefs[b] * pen
            record["per_bound"][_bkey(b)] = {
                "share_rows_gt_b": float(np.mean(raws[state][f"share_{_bkey(b)}"])), "G_pen_per_coef_mean": pen,
                "realised_pull": pull, "realised_pull_over_rms": pull / record["G_pg_rms"],
                "realised_pull_over_abs_G_pg_mean": pull / abs(g_mean) if g_mean != 0 else float("inf"),
            }
    out_json = {
        "schema": RT.COEF_SCHEMA, "rule": RULE, "reference_excess": RT.REFERENCE_EXCESS,
        "unit_pull_per_coef": RT.UNIT_PULL, "seed": CALIB_SEED, "rollouts_per_state": ROLLOUTS, "batch_size": BATCH,
        "states": states,
        "pooled": {"minibatches": int(len(pooled)), "G_pg_rms": pooled_rms, "coef_unrounded": pooled_rms / RT.UNIT_PULL},
        "COEF_90": coefs[0.90], "COEF_60": coefs[0.60], "coef_equal_by_formula": coefs[0.90] == coefs[0.60],
        "clone": {"head": git["head"], "status": git["status"], "branch": git["branch"]},
        "stack": C.stack_versions(), "utc": utc, "seconds": round(time.monotonic() - t0, 1),
    }
    out_json = RT.norm(out_json)
    problems = RT.coef_rule_problems(out_json)
    if problems:
        raise SystemExit(f"the coef.json just computed breaks the rule: {problems}")
    RT.modules_under_clone()
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "x") as fh:
        json.dump(out_json, fh, indent=1)
    print(json.dumps({"written": str(out), "sha256": RT.sha256_file(out), "COEF_90": coefs[0.90],
                      "COEF_60": coefs[0.60], "pooled_G_pg_rms": pooled_rms,
                      "realised_pull": {s: {k: v["realised_pull"] for k, v in states[s]["per_bound"].items()}
                                        for s in states},
                      "seconds": out_json["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
