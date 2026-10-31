"""EARLY-BIAS PROBE, one rollout: rebuild one PPO update's inputs at a checkpoint and read what that update would do
to the deterministic tail_3 mean (mu14).  NO TRAINING: no learn(), no optimizer step, no parameter written.

    PYTHONPATH=<r10round/ecs/code> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 venv/bin/python -B eb_update.py RUN STEPS SEED

1. Load (run, STEPS) as eb_common.build does (code copy's config, VecNormalize from the checkpoint pkl FROZEN,
   norm_reward on, MirrorKLPPO enabled), set ent_coef to the probe's value for the coming train().
2. Collect ONE rollout with SB3's own collect_rollouts (n_envs 4 x n_steps 2048, stochastic actions, envs reset
   with seeds SEED + rank, torch / numpy seeded SEED); GAE with the checkpoint's critic, gamma 0.98, lambda 0.95
   (SB3's compute_returns_and_advantage).  A recording callback keeps every step's per-term rewards (info
   reward_*), the raw and normalised reward, the pre-step raw observation and the episode step.
3. Split the advantage exactly: GAE is linear in the rewards and the values, so A = sum_k A(r_k, V=0) + A_critic,
   with A_critic = A(time-limit bootstraps, V) (the critic's baseline and bootstraps; not attributable per term).
4. One epoch of mini-batches exactly as MirrorKLPPO.train() draws them (rollout_buffer.get(128), np seeded), at the
   checkpoint's parameters (ratio 1, the first epoch): per mini-batch the full loss (clipped surrogate + ent_coef *
   entropy loss + vf_coef * mirrored value loss + kappa * mirror KL), its gradient norm and the clip factor
   c = min(1, 0.5 / (|g| + 1e-6)), and the gradient of each component; summed c * g per component = the epoch's SGD
   direction.  Adam preconditioning P = 1 / (sqrt(v_hat) + eps) from the checkpoint's own exp_avg_sq.
5. delta mu = J . step on the probe's bank (bank_raw_obs, normalised with the checkpoint's statistics): for every
   bank state and actuator, step = -lr * sum_mb c g (plain) and -lr * P (.) sum_mb c g (Adam), per component.
6. Per-row influence u = (P J_target) . grad log pi(a|s) by a forward-mode JVP of the actor mean, so that the
   policy term's pull splits exactly over rows: delta = lr * sum_rows (c_mb / B) * A_norm_row * u_row; and the
   mirror term's: -lr * kappa * sum_rows (c_mb / B) * sum_i dev_i / sigma_i^2 * JVP(mu_i(o_m)).  Targets: the bank
   mean mu14, the two statue_settled rows' mean mu14, and the mean mu14 over this rollout's rows.
7. r9-style read (lambda_calib.py): G_pg = d(clipped surrogate)/d(action_net.bias[14]) per mini-batch.
Writes eb_out/<run>_<steps>_<seed>.json and .npz (per-row arrays).
"""
from __future__ import annotations

import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
sys.path.insert(0, str(HERE))
import eb_common as E  # noqa: E402

OUTDIR = HERE / "eb_out2"
TAIL = 14
ACTOR_PREFIX = ("mlp_extractor.policy_net.", "action_net.")
#: the per-row influence u = sum_i (a_i - mu_i) / sigma_i^2 * JVP(mu_i), split by the actuator whose noise carries it
GROUPS = {"tail3": [14], "tail12": [11, 12, 13], "legs": list(range(3, 11)), "neckhead": [0, 1, 2]}
GROUP_NAMES = ("tail3", "tail12", "legs", "neckhead")


def gae(rewards, values, last_values, last_dones, episode_starts, gamma, lam):
    """SB3 2.9.0 RolloutBuffer.compute_returns_and_advantage, in float64 numpy, on (n_steps, n_envs) arrays."""
    n = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1])
    for step in reversed(range(n)):
        if step == n - 1:
            nnt = 1.0 - last_dones.astype(np.float64)
            nv = last_values
        else:
            nnt = 1.0 - episode_starts[step + 1]
            nv = values[step + 1]
        delta = rewards[step] + gamma * nv * nnt - values[step]
        last = delta + gamma * lam * nnt * last
        adv[step] = last
    return adv


def flat(a):
    """SB3's swap_and_flatten: (n_steps, n_envs, ...) -> (n_envs * n_steps, ...)."""
    a = np.asarray(a)
    sh = a.shape
    if len(sh) < 3:
        sh = (*sh, 1)
    return a.swapaxes(0, 1).reshape(sh[0] * sh[1], *sh[2:])


def main() -> int:
    run, steps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    OUTDIR.mkdir(exist_ok=True)
    stem = OUTDIR / f"{run}_{steps}_{seed}"
    t0 = time.time()
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger

    from environments.shared.mirror_kl_ppo import mirror_kl_loss

    b = E.build(run, steps, env_seed=seed)
    pr = E.probe(run)
    sched = E.set_probe_schedule(b, pr)
    lr = sched["learning_rate"]
    m, vn, policy = b.model, b.vn, b.model.policy
    vn.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m.set_logger(Logger(folder=None, output_formats=[]))
    m._last_obs = None
    gamma, lam = float(m.gamma), float(m.gae_lambda)
    ret_scale = math.sqrt(float(vn.ret_rms.var) + float(vn.epsilon))

    class Recorder(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.rows = []

        def _on_rollout_start(self) -> None:
            self.prev_raw = vn.get_original_obs().copy()
            self.t_ep = np.zeros(vn.num_envs, dtype=np.int64)

        def _on_step(self) -> bool:
            L = self.locals
            infos, dones = L["infos"], np.asarray(L["dones"]).copy()
            row = {
                "raw_obs": self.prev_raw.copy(),
                "t_ep": self.t_ep.copy(),
                "rew_norm": np.asarray(L["rewards"], dtype=np.float64).copy(),
                "rew_raw": np.asarray(vn.get_original_reward(), dtype=np.float64).copy(),
                "dones": dones,
                "trunc": np.array([bool(i.get("TimeLimit.truncated", False)) for i in infos]),
                "term_obs": {e: np.asarray(i["terminal_observation"], dtype=np.float32).copy() for e, i in enumerate(infos)
                             if dones[e] and i.get("TimeLimit.truncated", False) and i.get("terminal_observation") is not None},
                "terms": [{k: float(v) for k, v in i.items() if k.startswith("reward_") and k != "reward_total"}
                          for i in infos],
                "ep": [(i["episode"]["r"], i["episode"]["l"], i.get("termination_reason", "truncated"))
                       if "episode" in i else None for i in infos],
            }
            self.rows.append(row)
            self.prev_raw = vn.get_original_obs().copy()
            self.t_ep = np.where(dones, 0, self.t_ep + 1)
            return True

        def _on_rollout_end(self) -> None:
            self.last_values = self.locals["values"].detach().cpu().numpy().ravel().astype(np.float64)
            self.last_dones = np.asarray(self.locals["dones"]).copy()

    rec = Recorder()
    _total, callback = m._setup_learn(m.n_steps * m.n_envs, rec, False, "eb", False)
    ok = m.collect_rollouts(m.env, callback, m.rollout_buffer, n_rollout_steps=m.n_steps)
    assert ok
    t_collect = time.time() - t0
    buf = m.rollout_buffer
    nS, nE = buf.buffer_size, buf.n_envs
    # ---------------------------------------------------------------- per-term rewards and the exact GAE split
    keys = sorted({k for r in rec.rows for d in r["terms"] for k in d})
    R = np.zeros((nS, nE, len(keys)))
    for s, r in enumerate(rec.rows):
        for e in range(nE):
            for j, k in enumerate(keys):
                R[s, e, j] = r["terms"][e].get(k, 0.0)
    rew_raw = np.array([r["rew_raw"] for r in rec.rows])
    rew_norm = np.array([r["rew_norm"] for r in rec.rows])
    fall = rew_raw - R.sum(-1)
    fall_ok = bool(np.all((np.abs(fall) < 1e-6) | (np.abs(fall + 100.0) < 1e-6)))
    # terms whose weight is 0 in this stage: checked identically zero, then dropped from the split
    zero_terms = ("reward_forward", "reward_backward", "reward_bite", "reward_approach", "reward_head_proximity",
                  "reward_gait", "reward_idle", "reward_lateral")
    keep_k = [j for j, k in enumerate(keys) if k not in zero_terms]
    zero_ok = bool(all(np.all(R[..., j] == 0) for j, k in enumerate(keys) if k in zero_terms))
    keys = [keys[j] for j in keep_k] + ["fall_penalty"]
    R = np.concatenate([R[..., keep_k], fall[..., None]], axis=-1)
    clip_rows = int((np.abs(rew_raw / ret_scale) > vn.clip_reward).sum())
    norm_err = float(np.abs(rew_raw / ret_scale - rew_norm).max())
    Rn = R / ret_scale
    boot = buf.rewards.astype(np.float64) - rew_norm  # gamma * V(terminal obs) on time-limit truncations
    ep_starts = buf.episode_starts.astype(np.float64)
    values = buf.values.astype(np.float64)
    A_terms = np.stack([gae(Rn[..., j], np.zeros((nS, nE)), np.zeros(nE), rec.last_dones, ep_starts, gamma, lam)
                        for j in range(len(keys))], axis=-1)
    A_crit = gae(boot, values, rec.last_values, rec.last_dones, ep_starts, gamma, lam)
    A_buf = buf.advantages.astype(np.float64)
    split_err = float(np.abs(A_terms.sum(-1) + A_crit - A_buf).max())
    # per-term lambda=1 discounted returns-to-go inside the rollout (a check that does not use the critic)
    G_terms = np.stack([gae(Rn[..., j], np.zeros((nS, nE)), np.zeros(nE), rec.last_dones, ep_starts, gamma, 1.0)
                        for j in range(len(keys))], axis=-1)
    # ---------------------------------------------------------------- per-term baselines: V_k = ridge readout of the
    # critic's own last hidden layer, fitted to per-term discounted returns-to-go (gamma) on rows >= 200 steps from any
    # cut (time limit or rollout end), cross-fitted by env pairs (fit on envs {0,1}, apply to {2,3}, and back).
    # A_k = GAE(r_k + gamma V_k(terminal) on truncations, V_k); the critic residual A_res = A - sum_k A_k (exact).
    trunc = np.array([r["trunc"] for r in rec.rows])
    dones_arr = np.array([r["dones"] for r in rec.rows])
    with torch.no_grad():
        def feats(o):
            return policy.mlp_extractor.value_net(torch.as_tensor(np.asarray(o, dtype=np.float32))).numpy().astype(np.float64)
        Hs = feats(buf.observations.reshape(nS * nE, -1)).reshape(nS, nE, -1)
        H_last = feats(np.asarray(m._last_obs))
        term_idx = [(s, e) for s, r in enumerate(rec.rows) for e in r["term_obs"]]
        H_term = feats(np.stack([rec.rows[s]["term_obs"][e] for s, e in term_idx])) if term_idx else np.zeros((0, Hs.shape[-1]))
        vw = policy.value_net.weight.detach().numpy().astype(np.float64).ravel()
        vb = float(policy.value_net.bias.detach().numpy()[0])
    v_from_h_err = float(np.abs(Hs @ vw + vb - values).max())
    nK0 = len(keys)
    Gmc = np.zeros((nS, nE, nK0))
    cut = np.zeros((nS, nE))
    nxt = np.zeros((nE, nK0))
    ncut = np.zeros(nE)
    for s in reversed(range(nS)):
        last = s == nS - 1
        fell = dones_arr[s] & ~trunc[s]
        cont = (~dones_arr[s]).astype(np.float64)
        Gmc[s] = Rn[s] + gamma * cont[:, None] * (0.0 if last else nxt)
        ncut = np.where(trunc[s] | last, 0.0, np.where(fell, 1e9, ncut + 1.0))
        cut[s] = ncut
        nxt = Gmc[s]
    fit_mask = cut >= 200
    # low-dimensional features (robust across envs): [1, V(s), the first 8 principal components of the critic's hidden layer]
    Hflat = Hs.reshape(nS * nE, -1)
    Hm = Hflat[fit_mask.ravel()].mean(0)
    _u, _sv, Vt = np.linalg.svd(Hflat[fit_mask.ravel()] - Hm, full_matrices=False)
    PCS = Vt[:8].T

    def X_of(Hx, Vx):
        return np.concatenate([np.ones((len(Hx), 1)), Vx[:, None], (Hx - Hm) @ PCS], axis=1)

    Xall = X_of(Hflat, values.ravel()).reshape(nS, nE, -1)
    v_last = H_last @ vw + vb
    v_term = H_term @ vw + vb if len(term_idx) else np.zeros(0)
    X_last = X_of(H_last, v_last)
    X_term = X_of(H_term, v_term) if len(term_idx) else np.zeros((0, Xall.shape[-1]))
    Vk = np.zeros((nS, nE, nK0))
    Vk_last = np.zeros((nE, nK0))
    Vk_term = np.zeros((len(term_idx), nK0))
    r2_hold = []
    for fit_envs, app_envs in (([0, 1], [2, 3]), ([2, 3], [0, 1])):
        X = Xall[:, fit_envs][fit_mask[:, fit_envs]]
        Y = Gmc[:, fit_envs][fit_mask[:, fit_envs]]
        alpha = 1e-4 * np.trace(X.T @ X) / X.shape[1]
        beta_k = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
        Vk[:, app_envs] = Xall[:, app_envs] @ beta_k
        Vk_last[app_envs] = X_last[app_envs] @ beta_k
        for j, (s_, e_) in enumerate(term_idx):
            if e_ in app_envs:
                Vk_term[j] = X_term[j] @ beta_k
        Xa = Xall[:, app_envs][fit_mask[:, app_envs]]
        Ya = Gmc[:, app_envs][fit_mask[:, app_envs]]
        res = Ya - Xa @ beta_k
        r2_hold.append(1 - (res ** 2).sum(0) / np.maximum(((Ya - Ya.mean(0)) ** 2).sum(0), 1e-30))
    r2_hold = np.mean(r2_hold, axis=0)
    boot_k = np.zeros((nS, nE, nK0))
    for j, (s_, e_) in enumerate(term_idx):
        boot_k[s_, e_] = gamma * Vk_term[j]
    A_base = np.stack([gae(Rn[..., j] + boot_k[..., j], Vk[..., j], Vk_last[:, j], rec.last_dones, ep_starts, gamma, lam)
                       for j in range(nK0)], axis=-1)
    A_res = A_buf - A_base.sum(-1)
    # lambda = 1 per-term returns with the (cross-fitted, state-only) baseline: unbiased per-term attribution of the
    # Monte Carlo policy gradient, on rows >= 200 steps from any cut
    MCb = np.where(fit_mask[..., None], Gmc - Vk, 0.0)
    # The exact GAE split with STATE-ONLY baselines (unbiased in expectation, the GAE weighting unchanged):
    # A = sum_k R_k + C - V(s), R_k = GAE(r_k, V=0) (the gamma*lambda-discounted reward-to-go of term k), C = A_crit + V
    # (the critic's bootstraps).  Each part minus a cross-fitted b(s) has the same expected pull; the parts' pulls sum
    # to the actual pull in expectation (V(s) - sum b has none).
    Ctgt = A_crit + values
    SBt = np.concatenate([A_terms, Ctgt[..., None]], axis=-1)  # (nS, nE, nK0 + 1)
    SBb = np.zeros_like(SBt)
    for fit_envs, app_envs in (([0, 1], [2, 3]), ([2, 3], [0, 1])):
        X = Xall[:, fit_envs].reshape(-1, Xall.shape[-1])
        Y = SBt[:, fit_envs].reshape(-1, SBt.shape[-1])
        alpha = 1e-4 * np.trace(X.T @ X) / X.shape[1]
        beta_sb = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
        SBb[:, app_envs] = Xall[:, app_envs] @ beta_sb
    SBres = SBt - SBb
    vsum_r = float(np.corrcoef(Vk.sum(-1).ravel(), values.ravel())[0, 1])
    vsum_mae = float(np.abs(Vk.sum(-1) - values).mean())
    baseline_info = {"v_from_h_err": v_from_h_err, "fit_rows": int(fit_mask.sum()), "trunc_rows": len(term_idx),
                     "features": "1, V(s), 8 PCs of the critic hidden layer; ridge; cross-fitted by env pairs",
                     "r2_holdout": dict(zip(keys, r2_hold.tolist())), "sumVk_vs_V_r": vsum_r, "sumVk_vs_V_mae": vsum_mae,
                     "V_mean": float(values.mean()), "A_res_sd": float(A_res.std()), "A_sd": float(A_buf.std()),
                     "A_base_sd": dict(zip(keys, A_base.reshape(-1, nK0).std(0).tolist())),
                     "A_raw_sd": dict(zip(keys, A_terms.reshape(-1, nK0).std(0).tolist())),
                     "MC_raw_sd": dict(zip(keys, np.where(fit_mask[..., None], Gmc, 0).reshape(-1, nK0).std(0).tolist())),
                     "MC_base_sd": dict(zip(keys, MCb.reshape(-1, nK0).std(0).tolist()))}
    # ---------------------------------------------------------------- flat (training order) per-row arrays
    Ab_f = flat(A_base)
    Ar_f = flat(A_res).ravel()
    MCb_f = flat(MCb)
    SB_f = flat(SBres)  # (N, nK + 1): per-term then the critic bootstrap part, each minus its state baseline
    fitm_f = flat(fit_mask).ravel()
    raw_obs = np.array([r["raw_obs"] for r in rec.rows])  # (nS, nE, 63)
    t_ep = np.array([r["t_ep"] for r in rec.rows])
    obs_f = flat(buf.observations).astype(np.float32)
    act_f = flat(buf.actions).astype(np.float32)
    olp_f = flat(buf.log_probs).ravel()
    adv_f = flat(A_buf).ravel()
    At_f = flat(A_terms)
    Ac_f = flat(A_crit).ravel()
    Gt_f = flat(G_terms)
    raw_f = flat(raw_obs)
    tep_f = flat(t_ep).ravel()
    # remaining steps in the rollout for each row (for the lambda=1 truncation check)
    rem_f = flat(np.repeat((nS - 1 - np.arange(nS))[:, None], nE, axis=1)).ravel()
    renorm_err = float(np.abs(vn.normalize_obs(raw_f.astype(np.float32)) - obs_f).max())
    N = len(adv_f)
    # ---------------------------------------------------------------- one epoch of mini-batches, as train() draws them
    names = [n for n, _ in policy.named_parameters()]
    params = [p for _, p in policy.named_parameters()]
    opt = policy.optimizer
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in policy.parameters()]
    beta1, beta2 = opt.param_groups[0]["betas"]
    eps = opt.param_groups[0]["eps"]
    P, Mhat = [], []
    for p in params:
        st = opt.state[p]
        t = float(st["step"])
        vhat = st["exp_avg_sq"] / (1 - beta2 ** t)
        P.append(1.0 / (torch.sqrt(vhat) + eps))
        Mhat.append(st["exp_avg"] / (1 - beta1 ** t))
    adam_steps = float(opt.state[params[0]]["step"])
    policy.set_training_mode(True)
    clip_range = float(m.clip_range(m._current_progress_remaining))
    state = m._mirror_state()
    mirror = m._mirror_begin_update()
    tens = state.tensors
    kappa, sfloor = float(state.config.kappa), float(state.config.sigma_floor)
    perm_seed = seed + 1_000_003
    np.random.seed(perm_seed)
    perm = np.random.permutation(N)
    np.random.seed(perm_seed)
    comps = ("full", "pg", "ent", "vf", "kl")
    acc = {c: [torch.zeros_like(p) for p in params] for c in comps}
    acc_unclipped = {c: [torch.zeros_like(p) for p in params] for c in comps}
    c_row = np.zeros(N)
    mb_id = np.zeros(N, dtype=np.int16)
    mbA_mean = np.zeros(N)
    mbA_std = np.zeros(N)
    mb_rec = []
    bias = policy.action_net.bias
    ib = names.index("action_net.bias")
    B = int(m.batch_size)
    for k, data in enumerate(buf.get(B)):
        idx = perm[k * B:(k + 1) * B]
        if k == 0:
            assert np.array_equal(data.observations.numpy(), obs_f[idx]), "permutation mismatch"
        obs, actions = data.observations, data.actions
        vals, log_prob, entropy = policy.evaluate_actions(obs, actions)
        vals = vals.flatten()
        a = data.advantages
        an = (a - a.mean()) / (a.std() + 1e-8)
        ratio = torch.exp(log_prob - data.old_log_prob)
        pg = -torch.min(an * ratio, an * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        vl = F.mse_loss(data.returns, vals)
        vl = m._mirror_value_loss(mirror, data, vl)
        ent = -torch.mean(entropy)
        base = pg + m.ent_coef * ent + m.vf_coef * vl
        full = m._mirror_loss(mirror, data, base)
        klterm = full - base
        g = {"full": torch.autograd.grad(full, params, retain_graph=True, allow_unused=True)}
        norm = math.sqrt(sum(float((x ** 2).sum()) for x in g["full"] if x is not None))
        c = min(1.0, float(m.max_grad_norm) / (norm + 1e-6))
        for name, term in (("pg", pg), ("ent", m.ent_coef * ent), ("vf", m.vf_coef * vl), ("kl", klterm)):
            g[name] = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        for name in comps:
            for i, x in enumerate(g[name]):
                if x is not None:
                    acc[name][i] += c * x
                    acc_unclipped[name][i] += x
        c_row[idx] = c
        mb_id[idx] = k
        mbA_mean[idx] = float(a.mean())
        mbA_std[idx] = float(a.std())
        mb_rec.append({"c": c, "norm": norm, "G_pg_b14": float(g["pg"][ib][TAIL]), "G_kl_b14": float(g["kl"][ib][TAIL]),
                       "ratio_dev": float((ratio.detach() - 1).abs().max()),
                       "vf_actor_grad_max": max((float(g["vf"][i].abs().max()) for i, n in enumerate(names)
                                                 if n.startswith(ACTOR_PREFIX) and g["vf"][i] is not None),
                                                default=0.0),
                       "ent_actor_grad_max": max((float(g["ent"][i].abs().max()) for i, n in enumerate(names)
                                                  if n.startswith(ACTOR_PREFIX) and g["ent"][i] is not None),
                                                 default=0.0)})
    assert k == N // B - 1
    # sum over components equals the full gradient (sanity)
    sum_err = max(float((acc["full"][i] - acc["pg"][i] - acc["ent"][i] - acc["vf"][i] - acc["kl"][i]).abs().max())
                  for i in range(len(params)))
    # ---------------------------------------------------------------- the bank and the targets' Jacobians
    bank_raw = pr["bank_raw_obs"]
    bank_n = torch.as_tensor(np.asarray(vn.normalize_obs(bank_raw.astype(np.float32)), dtype=np.float32))
    actor_idx = [i for i, n in enumerate(names) if n.startswith(ACTOR_PREFIX)]
    actor_names = [names[i] for i in actor_idx]
    actor_params = {names[i]: params[i] for i in actor_idx}

    def _mean(pdict, x):
        h = torch.tanh(F.linear(x, pdict["mlp_extractor.policy_net.0.weight"], pdict["mlp_extractor.policy_net.0.bias"]))
        h = torch.tanh(F.linear(h, pdict["mlp_extractor.policy_net.2.weight"], pdict["mlp_extractor.policy_net.2.bias"]))
        return F.linear(h, pdict["action_net.weight"], pdict["action_net.bias"])

    with torch.no_grad():
        mu_bank = _mean(actor_params, bank_n)
        mode_bank = policy.get_distribution(bank_n).mode()
    mean_fn_err = float((mu_bank - mode_bank).abs().max())
    bank_probe_err = float(np.abs(mu_bank.numpy() - pr["bank_action_now"][b.probe_row]).max())
    # J for every bank state x actuator (150 rows) over the actor parameters
    Jb = []
    for bi in range(bank_n.shape[0]):
        for ai in range(15):
            out = _mean(actor_params, bank_n[bi:bi + 1])[0, ai]
            gr = torch.autograd.grad(out, [actor_params[n] for n in actor_names])
            Jb.append(torch.cat([x.reshape(-1) for x in gr]))
    Jb = torch.stack(Jb)  # (150, n_actor)

    def vec(lst, which=None):
        return torch.cat([lst[i].reshape(-1) for i in actor_idx])

    Pv = vec(P)
    stepv = {}
    for cname in comps:
        sv = vec(acc[cname])
        stepv[f"plain_{cname}"] = -lr * sv
        stepv[f"adam_{cname}"] = -lr * Pv * sv
        stepv[f"plainNC_{cname}"] = -lr * vec(acc_unclipped[cname])
    stepv["adam_momentum_1step"] = -lr * Pv * vec(Mhat)
    S = torch.stack(list(stepv.values()))
    D = (Jb @ S.T).T.reshape(len(stepv), bank_n.shape[0], 15).numpy()  # per step-vector: (bank, actuator)
    bank_delta = {k: D[i].tolist() for i, k in enumerate(stepv)}
    # ---------------------------------------------------------------- per-row influence by JVP
    from torch.func import jvp

    obs_all = torch.as_tensor(obs_f)
    act_all = torch.as_tensor(act_f)
    obs_m = obs_all[:, tens["obs_perm"]] * tens["obs_sign"]
    with torch.no_grad():
        mu_all = _mean(actor_params, obs_all)
        mu_m_all = _mean(actor_params, obs_m)
        sigma = torch.exp(policy.log_std.detach())
        sig_kl = sigma.clamp_min(sfloor)
        target = (mu_all[:, tens["act_perm"]] * tens["act_sign"])
        dev = mu_m_all - target
    ss_rows = [i for i, lab in enumerate(pr["bank_labels"]) if lab == "statue_settled"]
    targets = {"bank_mean": lambda pd: _mean(pd, bank_n)[:, TAIL].mean(),
               "statue_settled": lambda pd: _mean(pd, bank_n[ss_rows])[:, TAIL].mean(),
               "rollout_mean": lambda pd: _mean(pd, obs_all)[:, TAIL].mean()}
    U, K = {}, {}
    pnames = actor_names
    for tname, fn in targets.items():
        pd = {n: actor_params[n] for n in pnames}
        gr = torch.autograd.grad(fn(pd), [pd[n] for n in pnames])
        for pre, scale in (("plain", None), ("adam", P)):
            d = {}
            for j, n in enumerate(pnames):
                d[n] = gr[j] * (scale[names.index(n)] if scale is not None else 1.0)
            prim = {n: actor_params[n].detach() for n in pnames}
            _, dmu = jvp(lambda pdd: _mean(pdd, obs_all), (prim,), (d,))
            _, dmu_m = jvp(lambda pdd: _mean(pdd, obs_m), (prim,), (d,))
            per_act = (((act_all - mu_all) / sigma ** 2) * dmu).detach().numpy()  # (N, 15): J.P.grad log pi
            U[f"{pre}_{tname}"] = np.stack([per_act[:, GROUPS[g]].sum(-1) for g in GROUP_NAMES], axis=-1)
            K[f"{pre}_{tname}"] = ((dev / sig_kl ** 2) * dmu_m).sum(-1).detach().numpy()
    # check: the u-based policy-term sum equals J . step from the accumulators (bank_mean target, Adam)
    An = (adv_f - mbA_mean) / (mbA_std + 1e-8)
    w = lr * c_row / B
    pred_u = {k: float((w * An * U[k].sum(-1)).sum()) for k in U}
    pred_kl_u = {k: float((-w * kappa * K[k]).sum()) for k in K}
    jt = {}
    for tname, fn in targets.items():
        pd = {n: actor_params[n] for n in pnames}
        gr = torch.autograd.grad(fn(pd), [pd[n] for n in pnames])
        jv = torch.cat([x.reshape(-1) for x in gr])
        for pre in ("plain", "adam"):
            for cname in comps:
                jt[f"{pre}_{tname}_{cname}"] = float(jv @ stepv[f"{pre}_{cname}"])
            if pre == "adam":
                jt[f"{pre}_{tname}_momentum1"] = float(jv @ stepv["adam_momentum_1step"])
    # ---------------------------------------------------------------- the r9-style read
    Gpg = np.array([r["G_pg_b14"] for r in mb_rec])
    Gkl = np.array([r["G_kl_b14"] for r in mb_rec])
    # ---------------------------------------------------------------- the split by reward term and by state region
    nK, nmb = len(keys), N // B
    cnt = np.bincount(mb_id, minlength=nmb).astype(np.float64)
    mbk = np.stack([np.bincount(mb_id, weights=At_f[:, j], minlength=nmb) / cnt for j in range(nK)], -1)
    mbc = np.bincount(mb_id, weights=Ac_f, minlength=nmb) / cnt
    den = mbA_std + 1e-8
    Ank = (At_f - mbk[mb_id]) / den[:, None]  # A_norm = sum_k Ank + Anc exactly
    Anc = (Ac_f - mbc[mb_id]) / den
    mbb = np.stack([np.bincount(mb_id, weights=Ab_f[:, j], minlength=nmb) / cnt for j in range(nK)], -1)
    mbr = np.bincount(mb_id, weights=Ar_f, minlength=nmb) / cnt
    Anb = (Ab_f - mbb[mb_id]) / den[:, None]  # with per-term baselines: A_norm = sum_k Anb + Anr exactly
    Anr = (Ar_f - mbr[mb_id]) / den
    anb_err = float(np.abs(Anb.sum(-1) + Anr - An).max())
    cntm = np.maximum(np.bincount(mb_id, weights=fitm_f.astype(np.float64), minlength=nmb), 1.0)
    mbm = np.stack([np.bincount(mb_id, weights=MCb_f[:, j], minlength=nmb) / cntm for j in range(nK)], -1)
    AnMb = np.where(fitm_f[:, None], (MCb_f - mbm[mb_id]) / den[:, None], 0.0)
    mbs = np.stack([np.bincount(mb_id, weights=SB_f[:, j], minlength=nmb) / cnt for j in range(SB_f.shape[1])], -1)
    AnSB = (SB_f - mbs[mb_id]) / den[:, None]
    ank_err = float(np.abs(Ank.sum(-1) + Anc - An).max())
    keep = rem_f >= 200  # lambda = 1 check: rows whose discounted return-to-go is not cut by the rollout's end
    cntk = np.maximum(np.bincount(mb_id, weights=keep.astype(np.float64), minlength=nmb), 1.0)
    Gk = np.where(keep[:, None], Gt_f, 0.0)
    mbG = np.stack([np.bincount(mb_id, weights=Gk[:, j], minlength=nmb) / cntk for j in range(nK)], -1)
    AnG = np.where(keep[:, None], (Gt_f - mbG[mb_id]) / den[:, None], 0.0)
    tail_deg = np.degrees(raw_f[:, E.OBS_TAIL])
    pitch_deg = np.degrees(np.arcsin(np.clip(-raw_f[:, 42], -1.0, 1.0)))
    nfeet = (raw_f[:, 54] >= 42.0).astype(int) + (raw_f[:, 55] >= 42.0).astype(int)
    regions = {"tail3_deg": (tail_deg, [-1.0, 0.0, 1.0, 2.0, 3.0]),
               "pitch_deg": (pitch_deg, [-1.5, -1.0, -0.5, 0.0]),
               "feet_loaded": (nfeet, [0.5, 1.5]),
               "ep_step": (tep_f, [50, 200, 500])}
    split = {}
    for tk in U:
        ug = U[tk]
        u = ug.sum(-1)
        wu = w * u
        d = {"pg": float((wu * An).sum()),
             "pg_by_noise_group": dict(zip(GROUP_NAMES, ((w * An)[:, None] * ug).sum(0).tolist())),
             "kl": float((-w * kappa * K[tk]).sum()),
             "terms_gae": dict(zip(keys, (wu[:, None] * Ank).sum(0).tolist())),
             "critic_gae": float((wu * Anc).sum()),
             "terms_base": dict(zip(keys, (wu[:, None] * Anb).sum(0).tolist())),
             "critic_residual": float((wu * Anr).sum()),
             "terms_mc_base": dict(zip(keys, (wu[:, None] * AnMb).sum(0).tolist())),
             "terms_gae_sb": dict(zip(keys + ["CRITIC_BOOTSTRAP"], (wu[:, None] * AnSB).sum(0).tolist())),
             # like-for-like with MC-b: the actual (GAE) pull and its GAE-sb parts on the same rows (>= 200 steps from a cut)
             "pg_on_mc_rows": float((wu * An)[fitm_f].sum()),
             "gae_sb_on_mc_rows": dict(zip(keys + ["CRITIC_BOOTSTRAP"], (wu[fitm_f][:, None] * AnSB[fitm_f]).sum(0).tolist())),
             "terms_gae_sb_by_group": {g: dict(zip(keys + ["CRITIC_BOOTSTRAP"], ((w * ug[:, gi])[:, None] * AnSB).sum(0).tolist()))
                                       for gi, g in enumerate(GROUP_NAMES)},
             # reward term x whose sampled noise carries the pull (tail3 = tail_3's own exploration noise)
             "terms_base_by_group": {g: dict(zip(keys, ((w * ug[:, gi])[:, None] * Anb).sum(0).tolist()))
                                     for gi, g in enumerate(GROUP_NAMES)},
             "terms_mc_base_by_group": {g: dict(zip(keys, ((w * ug[:, gi])[:, None] * AnMb).sum(0).tolist()))
                                        for gi, g in enumerate(GROUP_NAMES)},
             "critic_residual_by_group": {g: float((w * ug[:, gi] * Anr).sum()) for gi, g in enumerate(GROUP_NAMES)},
             "terms_mc": dict(zip(keys, (wu[:, None] * AnG).sum(0).tolist())),
             "regions": {}}
        for rn, (x, inner) in regions.items():
            bi = np.digitize(x, inner)
            rr = {}
            for j in range(len(inner) + 1):
                sel = bi == j
                rr[str(j)] = {"n": int(sel.sum()), "pg": float((wu * An)[sel].sum()),
                              "kl": float((-w * kappa * K[tk])[sel].sum()),
                              "critic_residual": float((wu * Anr)[sel].sum()),
                              "terms": dict(zip(keys, (wu[sel][:, None] * Anb[sel]).sum(0).tolist())) if tk.startswith("adam") else None,
                              "terms_mc_base": dict(zip(keys, (wu[sel][:, None] * AnMb[sel]).sum(0).tolist())) if tk.startswith("adam") else None,
                              "terms_gae_sb": dict(zip(keys + ["CRITIC_BOOTSTRAP"], (wu[sel][:, None] * AnSB[sel]).sum(0).tolist())) if tk.startswith("adam") else None}
            d["regions"][rn] = {"inner_edges": inner, "bins": rr}
        split[tk] = d
    # the critic's error against realised discounted returns (gamma 0.98, rows >= 200 steps from a cut), by tail angle
    Gtot_f = flat(Gmc.sum(-1)).ravel()
    V_f = flat(values).ravel()
    resid = (Gtot_f - V_f)[fitm_f]
    td = tail_deg[fitm_f]
    Xr = np.stack([np.ones_like(td), td], 1)
    slope = float(np.linalg.lstsq(Xr, resid, rcond=None)[0][1]) if len(td) > 10 else float("nan")
    critic_check = {"resid_mean": float(resid.mean()), "resid_vs_tail_deg_slope": slope, "n": int(len(td)),
                    "tail_deg_sd": float(td.std()),
                    "resid_by_tail_bin": {str(j): [float(resid[np.digitize(td, [-1.0, 0.0, 1.0, 2.0, 3.0]) == j].mean())
                                                   if (np.digitize(td, [-1.0, 0.0, 1.0, 2.0, 3.0]) == j).any() else None,
                                                   int((np.digitize(td, [-1.0, 0.0, 1.0, 2.0, 3.0]) == j).sum())]
                                          for j in range(6)}}
    region_means = {"tail3_deg_mean": float(tail_deg.mean()), "pitch_deg_mean": float(pitch_deg.mean()),
                    "feet2_share": float((nfeet == 2).mean())}
    # ---------------------------------------------------------------- rollout validation numbers
    eps_done = [e for r in rec.rows for e in r["ep"] if e is not None]
    mu14 = mu_all[:, TAIL].numpy()
    res = {
        "run": run, "steps": int(b.steps), "seed": seed, "perm_seed": perm_seed,
        "env_seeds": [seed + i for i in range(nE)], "torch_seed": seed,
        "lr": lr, "ent_coef": float(m.ent_coef), "probe_row": b.probe_row, "clip_range": clip_range,
        "gamma": gamma, "gae_lambda": lam, "vf_coef": float(m.vf_coef), "max_grad_norm": float(m.max_grad_norm),
        "kappa": kappa, "sigma_floor": sfloor, "batch_size": B, "n_epochs": int(m.n_epochs),
        "n_updates_ckpt": int(m._n_updates), "adam_steps": adam_steps, "ret_scale": ret_scale,
        "config_problems": b.config_problems,
        "checks": {"fall_terms_ok": fall_ok, "zero_weight_terms_zero": zero_ok, "reward_clip_rows": clip_rows, "reward_norm_err": norm_err,
                   "gae_split_max_err": split_err, "renormalise_obs_err": renorm_err,
                   "component_sum_err": sum_err, "mean_fn_err": mean_fn_err, "bank_vs_probe_err": bank_probe_err,
                   "ratio_dev_max": max(r["ratio_dev"] for r in mb_rec),
                   "vf_grad_on_actor_max": max(r["vf_actor_grad_max"] for r in mb_rec),
                   "ent_grad_on_actor_max": max(r["ent_actor_grad_max"] for r in mb_rec),
                   "u_vs_J_adam_bank_pg": [pred_u["adam_bank_mean"], jt["adam_bank_mean_pg"]],
                   "u_vs_J_plain_rollout_pg": [pred_u["plain_rollout_mean"], jt["plain_rollout_mean_pg"]],
                   "u_vs_J_adam_bank_kl": [pred_kl_u["adam_bank_mean"], jt["adam_bank_mean_kl"]]},
        "reward_keys": keys,
        "rollout": {"rows": int(N), "mu14_mean": float(mu14.mean()), "mu14_q50": float(np.median(mu14)),
                    "mu14_q95": float(np.quantile(mu14, 0.95)), "mu14_beyond_high": float((mu14 > 1).mean()),
                    "a14_clip_high": float((act_f[:, TAIL] > 1).mean()),
                    "a14_sat": float((np.abs(np.clip(act_f[:, TAIL], -1, 1)) >= 0.9).mean()),
                    "std14": float(sigma[TAIL]),
                    "episodes_done": len(eps_done),
                    "ep_return_mean": float(np.mean([e[0] for e in eps_done])) if eps_done else None,
                    "ep_length_mean": float(np.mean([e[1] for e in eps_done])) if eps_done else None,
                    "ep_falls": int(sum(1 for e in eps_done if e[1] < 1000)),
                    "episodes": eps_done,
                    "reward_term_means_raw": dict(zip(keys, R.reshape(-1, len(keys)).mean(0).tolist())),
                    "rew_raw_mean": float(rew_raw.mean()),
                    "tail3_deg_mean": float(np.degrees(raw_f[:, E.OBS_TAIL]).mean())},
        "mb": {"c_median": float(np.median([r["c"] for r in mb_rec])),
               "norm_median": float(np.median([r["norm"] for r in mb_rec])),
               "G_pg_b14_mean": float(Gpg.mean()), "G_pg_b14_sem": float(Gpg.std(ddof=1) / math.sqrt(len(Gpg))),
               "G_pg_b14_rms": float(np.sqrt((Gpg ** 2).mean())), "G_kl_b14_mean": float(Gkl.mean()),
               "G_kl_b14_rms": float(np.sqrt((Gkl ** 2).mean())), "n_mb": len(mb_rec)},
        "bank_delta_per_epoch": bank_delta,  # step name -> (10 bank, 15 actuators)
        "target_delta_per_epoch": jt,
        "pred_u": pred_u, "pred_kl_u": pred_kl_u,
        "split_per_epoch": split, "region_means": region_means, "critic_check": critic_check,
        "bank_mu_ckpt": mu_bank.numpy().tolist(),
        "seconds": {"collect": t_collect, "total": time.time() - t0},
    }
    res["checks"]["term_split_norm_err"] = ank_err
    res["checks"]["base_split_norm_err"] = anb_err
    res["baseline_fit"] = baseline_info
    with open(str(stem) + ".json", "w") as fh:
        json.dump(res, fh, default=float)
    if os.environ.get("EB_NPZ") == "1":
        np.savez_compressed(
        str(stem) + ".npz", keys=np.array(keys), A_terms=At_f.astype(np.float32), A_crit=Ac_f.astype(np.float32),
        adv=adv_f.astype(np.float32), mbA_mean=mbA_mean.astype(np.float32),
        mbA_std=mbA_std.astype(np.float32), c_row=c_row.astype(np.float32), mb_id=mb_id, rem=rem_f.astype(np.int16),
        t_ep=tep_f.astype(np.int16), tail3=raw_f[:, E.OBS_TAIL].astype(np.float32),
        tail3_vel=raw_f[:, 27].astype(np.float32), gx=raw_f[:, 42].astype(np.float32),
        foot_r=raw_f[:, 54].astype(np.float32), foot_l=raw_f[:, 55].astype(np.float32),
        mu14=mu14.astype(np.float32), a14=act_f[:, TAIL].astype(np.float32),
        **{f"U_{k}": v.astype(np.float32) for k, v in U.items()}, **{f"K_{k}": v.astype(np.float32) for k, v in K.items()},
        Gpg=Gpg.astype(np.float32), Gkl=Gkl.astype(np.float32))
    print(json.dumps({"stem": stem.name, "checks": res["checks"], "seconds": res["seconds"],
                      "rollout": {k: v for k, v in res["rollout"].items() if k != "episodes"}}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
