"""VERIFY EARLY-BIAS (adversarial verifier's own script; does not import the probe's code).  NO TRAINING: no learn(),
no optimizer.step(); parameters are never written (finite differences use a float64 deep copy of the actor).

    PYTHONPATH=<r10round/ecs/code> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 venv/bin/python -B vb_update.py \
        RUN STEPS SEED0 NSEEDS [ZIP PKL]

For each seed: load the checkpoint with stock SB3 PPO.load (the zip's optimizer state included) on a DummyVecEnv of 4
Monitor-wrapped stance envs (env kwargs from the code copy's stage config) under the checkpoint's VecNormalize, FROZEN
(training False, norm_reward True).  Collect one 4 x 2048 rollout with SB3's collect_rollouts (GAE inside).  Then one
epoch of mini-batches (own permutation, own mini-batch loop) of MY re-implementation of MirrorKLPPO.train's loss at the
checkpoint's parameters (ratio 1): pg + ent_coef * (-H) + vf_coef * 0.5 (mse(R, V(o)) + mse(R, V(o_m))) + kappa * L_K,
loss.backward() into .grad, the log_std tie hook as training registers it, torch's own clip_grad_norm_ (0.5); the
clipped gradients summed over the epoch.  Step = -lr * P * sum (Adam preconditioner from the zip's exp_avg_sq) and
-lr * sum (plain).  Read on the deterministic tail_3 mean (mu14) of: the rollout's rows (mean), the probe bank (mean), the
bank's two statue_settled rows (mean).  Two routes: (A) J . step with J by autograd; (B) per-row influence by a central
finite difference of the actor mean (float64 copy) along P*J, so that the policy-term pull splits over rows exactly:
pull = lr * sum_rows (c_mb / B) * A_norm * u, u = sum_i (a_i - mu_i)/sigma_i^2 * dmu_i.  Advantage sources (raw, no
fitted baselines): A = sum_k GAE(r_k, V=0) + A_boot (exact remainder: the critic's baseline, bootstraps and truncation
values).  Critic-free test: A_mc = discounted (0.98, lambda 1) return - V(s) on rows >= 200 steps from any cut (time
limit or rollout end), against the actual A on the same rows, both scaled by the mini-batch's A std.
Writes vb_out/<run>_<steps>_<seed>.json.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
OUT = HERE / "vb_out"
TAIL, OBS_TAIL = 14, 6
GAMMA_MC = 0.98


def run_dir(run):
    return F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance"


def gae_np(r, v, v_last, done_last, ep_start, gamma, lam):
    """GAE written from the definition (not copied): delta_t = r_t + gamma * V_{t+1} * (1 - start_{t+1}) - V_t."""
    n, e = r.shape
    out = np.zeros((n, e))
    acc = np.zeros(e)
    for t in range(n - 1, -1, -1):
        if t == n - 1:
            nonterm = 1.0 - done_last
            vn = v_last
        else:
            nonterm = 1.0 - ep_start[t + 1]
            vn = v[t + 1]
        acc = r[t] + gamma * vn * nonterm - v[t] + gamma * lam * nonterm * acc
        out[t] = acc
    return out


def main():
    run, steps, seed0, nseeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    zpath = sys.argv[5] if len(sys.argv) > 5 else str(run_dir(run) / "models" / f"stage1_{steps}_steps.zip.copy1")
    ppath = sys.argv[6] if len(sys.argv) > 6 else str(run_dir(run) / "models" / f"stage1_vecnormalize_{steps}_steps.pkl.copy1")
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from torch.func import functional_call

    from environments.shared.config import load_stage_config
    from environments.shared.mirror_kl_ppo import mirror_kl_loss
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class
    from environments.shared.symmetry import MirrorMap

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    cfg = load_stage_config(mid, "stance")
    env_kwargs = dict(cfg["env_kwargs"])
    rec = json.load(open(run_dir(run) / "stage_config.json.copy1"))
    hp = rec["hyperparameters"]
    pr = dict(np.load(run_dir(run) / "policy_probe.npz.copy1"))
    row = int(np.flatnonzero(pr["update_timesteps"] == steps)[0])
    lr = float(pr["learning_rate"][row])
    ent_coef = float(pr["ent_coef"][row])
    mm = MirrorMap.load(mid, rec["plant_identity"])
    obs_perm = torch.as_tensor(np.asarray(mm.obs.perm))
    obs_sign = torch.as_tensor(np.asarray(mm.obs.sign), dtype=torch.float32)
    act_perm = torch.as_tensor(np.asarray(mm.action.perm))
    act_sign = torch.as_tensor(np.asarray(mm.action.sign), dtype=torch.float32)
    kappa, sfloor = float(hp["symmetry_kappa"]), float(hp["symmetry_sigma_floor"])
    assert hp["symmetry_mirror_critic"] is True
    cls = env_class(mid)

    for seed in range(seed0, seed0 + nseeds):
        stem = OUT / f"{run}_{steps}_{seed}.json"
        if stem.exists():
            continue
        t0 = time.time()

        def mk(rank):
            def _f():
                set_random_seed(seed + rank)
                e = Monitor(cls(**env_kwargs))
                e.reset(seed=seed + rank)
                return e
            return _f

        venv = DummyVecEnv([mk(i) for i in range(4)])
        vn = VecNormalize.load(ppath, venv)
        vn.training = False
        vn.norm_reward = True
        const = lambda _: 0.0  # noqa: E731
        model = PPO.load(zpath, env=vn, device="cpu",
                         custom_objects={"learning_rate": lr, "lr_schedule": (lambda _: lr), "clip_range": (lambda _: 0.2)})
        policy = model.policy
        # hyperparameters as loaded vs the run's record
        hp_check = {k: (getattr(model, k), hp[k]) for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "target_kl")}
        hp_check["vf_coef"] = (model.vf_coef, 0.5)
        hp_check["max_grad_norm"] = (model.max_grad_norm, 0.5)
        hp_check["normalize_advantage"] = (model.normalize_advantage, True)
        hp_check["clip_range_vf"] = (model.clip_range_vf, None)
        hp_bad = {k: v for k, v in hp_check.items() if v[0] != v[1]}
        # tie hook as MirrorKLPPO registers it (affects only log_std's gradient, hence the clip norm)
        perm_t = act_perm.clone()
        policy.log_std.register_hook(lambda g: (g + g[perm_t]) / 2)
        opt = policy.optimizer
        pg0 = opt.param_groups[0]
        assert not pg0.get("amsgrad", False) and pg0.get("weight_decay", 0) == 0
        assert len(opt.state) == len(list(policy.parameters())), "optimizer state missing"
        names = [n for n, _ in policy.named_parameters()]
        params = [p for _, p in policy.named_parameters()]
        b1, b2 = pg0["betas"]
        eps = pg0["eps"]
        Pc = {}
        for n, p in zip(names, params):
            st = opt.state[p]
            t = float(st["step"])
            Pc[n] = 1.0 / (st["exp_avg_sq"].sqrt() / math.sqrt(1 - b2 ** t) + eps)
        adam_t = float(opt.state[params[0]]["step"])

        set_random_seed(seed)
        vn.seed(seed)
        model.set_logger(Logger(folder=None, output_formats=[]))
        model._last_obs = None

        class Rec(BaseCallback):
            def __init__(self):
                super().__init__(0)
                self.raw, self.terms, self.rraw, self.dones, self.trunc, self.tep, self.eps = [], [], [], [], [], [], []

            def _on_rollout_start(self):
                self.prev = vn.get_original_obs().copy()
                self.t = np.zeros(4, dtype=np.int64)

            def _on_step(self):
                infos = self.locals["infos"]
                d = np.asarray(self.locals["dones"]).copy()
                self.raw.append(self.prev.copy())
                self.tep.append(self.t.copy())
                self.terms.append([{k: float(v) for k, v in i.items() if k.startswith("reward_") and k != "reward_total"} for i in infos])
                self.rraw.append(np.asarray(vn.get_original_reward(), dtype=np.float64).copy())
                self.dones.append(d)
                self.trunc.append(np.array([bool(i.get("TimeLimit.truncated", False)) for i in infos]))
                for i in infos:
                    if "episode" in i:
                        self.eps.append((float(i["episode"]["r"]), int(i["episode"]["l"])))
                self.prev = vn.get_original_obs().copy()
                self.t = np.where(d, 0, self.t + 1)
                return True

            def _on_rollout_end(self):
                self.last_dones = np.asarray(self.locals["dones"]).astype(np.float64)
                self.last_values = self.locals["values"].detach().numpy().ravel().astype(np.float64)

        rc = Rec()
        _tot, cb = model._setup_learn(model.n_steps * model.n_envs, rc, False, "vb", False)
        assert model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=model.n_steps)
        t_col = time.time() - t0
        buf = model.rollout_buffer
        nS, nE = buf.buffer_size, buf.n_envs
        keys = sorted({k for r in rc.terms for d in r for k in d})
        R = np.array([[[d.get(k, 0.0) for k in keys] for d in r] for r in rc.terms])  # (nS, nE, K)
        rraw = np.array(rc.rraw)
        fall = rraw - R.sum(-1)
        fall_ok = bool(np.all((np.abs(fall) < 1e-5) | (np.abs(fall + 100.0) < 1e-5)))
        nz = [j for j in range(len(keys)) if np.any(R[..., j] != 0)]
        keys = [keys[j] for j in nz] + ["fall"]
        R = np.concatenate([R[..., nz], fall[..., None]], -1)
        scale = math.sqrt(float(vn.ret_rms.var) + float(vn.epsilon))
        rew_norm_rec = buf.rewards.astype(np.float64)  # includes gamma V(terminal) on truncations
        ep_start = buf.episode_starts.astype(np.float64)
        V = buf.values.astype(np.float64)
        A = buf.advantages.astype(np.float64)
        # my own GAE of the buffer's rewards must reproduce SB3's advantages
        A_me = gae_np(rew_norm_rec, V, rc.last_values, rc.last_dones, ep_start, model.gamma, model.gae_lambda)
        gae_err = float(np.abs(A_me - A).max())
        Rn = R / scale
        clip_rows = int((np.abs(rraw / scale) > vn.clip_reward).sum())
        A_terms = np.stack([gae_np(Rn[..., j], np.zeros((nS, nE)), np.zeros(nE), rc.last_dones, ep_start, model.gamma,
                                   model.gae_lambda) for j in range(len(keys))], -1)
        A_boot = A - A_terms.sum(-1)
        # critic-free discounted return (gamma 0.98, lambda 1) and the rows >= 200 steps from any cut
        dones = np.array(rc.dones)
        trunc = np.array(rc.trunc)
        G = np.zeros((nS, nE))
        Gk = np.zeros((nS, nE, len(keys)))
        dist = np.zeros((nS, nE))
        nxt = np.zeros(nE)
        nxtk = np.zeros((nE, len(keys)))
        dcut = np.zeros(nE)
        for t in range(nS - 1, -1, -1):
            last = t == nS - 1
            cont = 0.0 if last else 1.0
            G[t] = rraw[t] / scale + GAMMA_MC * (1.0 - dones[t]) * cont * nxt
            Gk[t] = Rn[t] + GAMMA_MC * ((1.0 - dones[t]) * cont)[:, None] * nxtk
            nxtk = Gk[t]
            # distance to the next cut: rollout end or a time-limit truncation resets it; a fall is a true end (no cut)
            dcut = np.where(last | trunc[t], 0.0, np.where(dones[t] & ~trunc[t], 1e9, dcut + 1.0))
            dist[t] = dcut
            nxt = G[t]
        keep = dist >= 200
        A_mc = G - V

        def fl(x):
            x = np.asarray(x)
            return x.swapaxes(0, 1).reshape(nS * nE, *x.shape[2:])

        obs = torch.as_tensor(fl(buf.observations), dtype=torch.float32)
        act = torch.as_tensor(fl(buf.actions), dtype=torch.float32)
        oldlp = torch.as_tensor(fl(buf.log_probs).ravel(), dtype=torch.float32)
        A_f = fl(A).ravel()
        At_f = fl(A_terms)
        Ab_f = fl(A_boot).ravel()
        Amc_f = fl(A_mc).ravel()
        keep_f = fl(keep).ravel()
        Gk_f = fl(Gk)
        V_f = fl(V).ravel()
        gk_err = float(np.abs(Gk.sum(-1) - G).max())
        ret_f = torch.as_tensor(fl(buf.returns).ravel(), dtype=torch.float32)
        raw_f = fl(np.array(rc.raw))
        tep_f = fl(np.array(rc.tep)).ravel()
        N = len(A_f)
        Bsz = int(model.batch_size)
        rng = np.random.default_rng(seed + 424242)
        order = rng.permutation(N)
        acc = {c: [torch.zeros_like(p) for p in params] for c in ("full", "pg", "kl", "ent", "vf")}
        c_row = np.zeros(N)
        mbm = np.zeros(N)
        mbs = np.zeros(N)
        mbid = np.zeros(N, dtype=np.int64)
        norms, G14, kl_chk, comp_norm = [], [], [], {"pg": [], "kl": [], "vf": [], "ent": []}
        policy.set_training_mode(True)
        ib = names.index("action_net.bias")
        for k in range(N // Bsz):
            idx = order[k * Bsz:(k + 1) * Bsz]
            o, a, olp, ret = obs[idx], act[idx], oldlp[idx], ret_f[idx]
            adv = torch.as_tensor(A_f[idx], dtype=torch.float32)
            vals, logp, ent = policy.evaluate_actions(o, a)
            vals = vals.flatten()
            an = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(logp - olp)
            pg = -torch.min(an * ratio, an * torch.clamp(ratio, 0.8, 1.2)).mean()
            om = o[:, obs_perm] * obs_sign
            vl = 0.5 * (F.mse_loss(ret, vals) + F.mse_loss(ret, policy.predict_values(om).flatten()))
            el = -ent.mean()
            mu = policy.get_distribution(o).distribution.mean
            mu_m = policy.get_distribution(om).distribution.mean
            sig = policy.log_std.exp().detach().clamp_min(sfloor)
            tgt = (mu[:, act_perm] * act_sign).detach()
            kl = ((mu_m - tgt) ** 2 / (2 * sig ** 2)).sum(-1).mean()
            if k == 0:
                kl_code, _ = mirror_kl_loss(mu_m, mu, policy.log_std, act_perm, act_sign, sfloor)
                kl_chk.append(abs(float(kl_code) - float(kl)))
            loss = pg + ent_coef * el + model.vf_coef * vl + kappa * kl
            comps = {"pg": pg, "kl": kappa * kl, "ent": ent_coef * el, "vf": model.vf_coef * vl}
            gc = {c: torch.autograd.grad(v, params, retain_graph=True, allow_unused=True) for c, v in comps.items()}
            for p in params:
                p.grad = None
            loss.backward()
            tn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm))
            c = min(1.0, model.max_grad_norm / (tn + 1e-6))
            norms.append(tn)
            for i, p in enumerate(params):
                acc["full"][i] += p.grad.detach().clone()
                for cname in comps:
                    g = gc[cname][i]
                    if g is not None:
                        if cname != "full" and names[i] == "log_std":
                            g = (g + g[perm_t]) / 2  # the hook acts on .grad, not on autograd.grad outputs
                        acc[cname][i] += c * g
            for cname in comps:
                comp_norm[cname].append(math.sqrt(sum(float((g ** 2).sum()) for g in gc[cname] if g is not None)))
            G14.append(float(gc["pg"][ib][TAIL]))
            c_row[idx] = c
            mbm[idx] = float(adv.mean())
            mbs[idx] = float(adv.std())
            mbid[idx] = k
            for p in params:
                p.grad = None
        sum_err = max(float((acc["full"][i] - acc["pg"][i] - acc["kl"][i] - acc["ent"][i] - acc["vf"][i]).abs().max())
                      for i in range(len(params)))
        actor = [i for i, n in enumerate(names) if n.startswith(("mlp_extractor.policy_net", "action_net"))]
        anames = [names[i] for i in actor]
        # ------------------------------------------------ targets and J by autograd
        bank_raw = pr["bank_raw_obs"].astype(np.float32)
        bank = torch.as_tensor(vn.normalize_obs(bank_raw), dtype=torch.float32)
        ss = [i for i, l in enumerate(pr["bank_labels"]) if l == "statue_settled"]
        with torch.no_grad():
            mu_bank = policy.get_distribution(bank).mode()
        bank_err = float(np.abs(mu_bank.numpy() - pr["bank_action_now"][row]).max())

        def mean_fn(pd, x):
            lat = functional_call(policy.mlp_extractor.policy_net, {k.split("policy_net.")[1]: v for k, v in pd.items() if "policy_net." in k}, (x,))
            return functional_call(policy.action_net, {k.split("action_net.")[1]: v for k, v in pd.items() if k.startswith("action_net.")}, (lat,))

        pd0 = {n: dict(policy.named_parameters())[n] for n in anames}
        with torch.no_grad():
            fn_err = float((mean_fn(pd0, obs[:512]) - policy.get_distribution(obs[:512]).mode()).abs().max())
        targets = {"roll": obs, "bank": bank, "ss": bank[ss]}
        J = {}
        for tn_, x in targets.items():
            gr = torch.autograd.grad(mean_fn(pd0, x)[:, TAIL].mean(), [pd0[n] for n in anames])
            J[tn_] = dict(zip(anames, gr))
        res_J = {}
        for tn_ in targets:
            for pre in ("adam", "plain"):
                for cname in ("full", "pg", "kl", "ent", "vf"):
                    s = 0.0
                    for i in actor:
                        n = names[i]
                        stepv = -lr * acc[cname][i] * (Pc[n] if pre == "adam" else 1.0)
                        s += float((J[tn_][n] * stepv).sum())
                    res_J[f"{pre}_{tn_}_{cname}"] = s
        # ------------------------------------------------ route B: per-row influence by finite differences (float64 copy)
        pol64 = {n: pd0[n].detach().double().clone() for n in anames}
        x64 = obs.double()
        sig64 = policy.log_std.detach().exp().double()
        with torch.no_grad():
            mu64 = mean_fn(pol64, x64)
        dev = (act.double() - mu64) / sig64 ** 2  # d log pi / d mu
        U = {}
        fd_lin = {}
        for tn_ in targets:
            for pre in ("adam", "plain"):
                dirv = {n: (J[tn_][n] * (Pc[n] if pre == "adam" else 1.0)).double() for n in anames}
                dn = math.sqrt(sum(float((v ** 2).sum()) for v in dirv.values()))
                h = 1e-3 / dn
                with torch.no_grad():
                    mp = mean_fn({n: pol64[n] + h * dirv[n] for n in anames}, x64)
                    mn = mean_fn({n: pol64[n] - h * dirv[n] for n in anames}, x64)
                    dmu = (mp - mn) / (2 * h)
                    # linearity: half step
                    mp2 = mean_fn({n: pol64[n] + 0.5 * h * dirv[n] for n in anames}, x64)
                    mn2 = mean_fn({n: pol64[n] - 0.5 * h * dirv[n] for n in anames}, x64)
                    dmu2 = (mp2 - mn2) / h
                fd_lin[f"{pre}_{tn_}"] = float((dmu - dmu2).abs().max() / (dmu.abs().max() + 1e-30))
                U[f"{pre}_{tn_}"] = (dev * dmu).numpy()  # (N, 15) per-actuator
        An = (A_f - mbm) / (mbs + 1e-8)
        w = lr * c_row / Bsz
        out_split = {}
        for key, u15 in U.items():
            u = u15.sum(-1)
            d = {"pg": float((w * An * u).sum()), "pg_tail3_noise": float((w * An * u15[:, TAIL]).sum()),
                 "pg_legs_noise": float((w * An * u15[:, 3:11].sum(-1)).sum()),
                 "pg_other_noise": float((w * An * (u - u15[:, TAIL] - u15[:, 3:11].sum(-1))).sum())}
            # raw source split (exact, mini-batch mean of each part removed, the actual A std)
            cnt = np.bincount(mbid, minlength=N // Bsz)
            parts = {}
            for j, kname in enumerate(keys):
                x = At_f[:, j]
                xm = np.bincount(mbid, weights=x, minlength=N // Bsz) / cnt
                parts[kname] = float((w * u * (x - xm[mbid]) / (mbs + 1e-8)).sum())
            xm = np.bincount(mbid, weights=Ab_f, minlength=N // Bsz) / cnt
            parts["BOOT"] = float((w * u * (Ab_f - xm[mbid]) / (mbs + 1e-8)).sum())
            d["src"] = parts
            d["src_sum_err"] = abs(sum(parts.values()) - d["pg"])
            # critic-free test on rows >= 200 steps from a cut (same rows, same scaling, kept-row means removed)
            kc = np.maximum(np.bincount(mbid, weights=keep_f.astype(float), minlength=N // Bsz), 1)
            for nm, x in (("gae_keep", A_f), ("mc_keep", Amc_f)):
                xm = np.bincount(mbid, weights=np.where(keep_f, x, 0.0), minlength=N // Bsz) / kc
                d[nm] = float((np.where(keep_f, w * u * (x - xm[mbid]) / (mbs + 1e-8), 0.0)).sum())
            # tail-3-noise share of the critic-free test
            u3 = u15[:, TAIL]
            for nm, x in (("gae_keep_t3", A_f), ("mc_keep_t3", Amc_f)):
                xm = np.bincount(mbid, weights=np.where(keep_f, x, 0.0), minlength=N // Bsz) / kc
                d[nm] = float((np.where(keep_f, w * u3 * (x - xm[mbid]) / (mbs + 1e-8), 0.0)).sum())
            # per-term splits for tail_3's own noise (u3) and for all carriers (u): raw GAE terms and critic-free MC terms
            for lab, uu in (("t3", u3), ("all", u)):
                pt = {}
                for j, kname in enumerate(keys):
                    x = At_f[:, j]
                    xm = np.bincount(mbid, weights=x, minlength=N // Bsz) / cnt
                    pt[kname] = float((w * uu * (x - xm[mbid]) / (mbs + 1e-8)).sum())
                xm = np.bincount(mbid, weights=Ab_f, minlength=N // Bsz) / cnt
                pt["BOOT"] = float((w * uu * (Ab_f - xm[mbid]) / (mbs + 1e-8)).sum())
                d[f"gae_terms_{lab}"] = pt
                pm = {}
                for j, kname in enumerate(keys):
                    x = Gk_f[:, j]
                    xm = np.bincount(mbid, weights=np.where(keep_f, x, 0.0), minlength=N // Bsz) / kc
                    pm[kname] = float(np.where(keep_f, w * uu * (x - xm[mbid]) / (mbs + 1e-8), 0.0).sum())
                xm = np.bincount(mbid, weights=np.where(keep_f, -V_f, 0.0), minlength=N // Bsz) / kc
                pm["MINUS_V"] = float(np.where(keep_f, w * uu * (-V_f - xm[mbid]) / (mbs + 1e-8), 0.0).sum())
                d[f"mc_terms_{lab}"] = pm
            out_split[key] = d
        # ------------------------------------------------ critic slope on the observed tail angle (per degree)
        z = torch.as_tensor(vn.normalize_obs(raw_f.astype(np.float32)), dtype=torch.float32).requires_grad_(True)
        vsum = policy.predict_values(z).sum()
        gz = torch.autograd.grad(vsum, z)[0][:, OBS_TAIL].numpy()
        z2 = torch.as_tensor(vn.normalize_obs(raw_f.astype(np.float32)), dtype=torch.float32).requires_grad_(True)
        gm = torch.autograd.grad(policy.get_distribution(z2).mode()[:, TAIL].sum(), z2)[0][:, OBS_TAIL].numpy()
        sc = math.radians(1.0) / math.sqrt(float(vn.obs_rms.var[OBS_TAIL]) + float(vn.epsilon))
        unclipped = np.abs(z.detach().numpy()[:, OBS_TAIL]) < vn.clip_obs
        with torch.no_grad():
            mu_roll = policy.get_distribution(obs).mode()[:, TAIL].numpy()
        res = {
            "run": run, "steps": steps, "seed": seed, "zip": zpath, "pkl": ppath, "lr": lr, "ent_coef": ent_coef,
            "n_updates_ckpt": int(model._n_updates), "adam_t": adam_t, "hp_mismatch": {k: [str(v[0]), str(v[1])] for k, v in hp_bad.items()},
            "checks": {"fall_ok": fall_ok, "gae_err": gae_err, "clip_rows": clip_rows, "comp_sum_err": sum_err,
                       "bank_vs_probe_err": bank_err, "mean_fn_err": fn_err, "kl_vs_code": max(kl_chk), "fd_nonlin": fd_lin,
                       "src_sum_err": max(v["src_sum_err"] for v in out_split.values()), "gk_err": gk_err,
                       "mc_terms_sum_err": max(abs(sum(v["mc_terms_all"].values()) - v["mc_keep"]) for v in out_split.values()),
                       "routeA_vs_B_adam_roll": [res_J["adam_roll_pg"], out_split["adam_roll"]["pg"]],
                       "routeA_vs_B_adam_bank": [res_J["adam_bank_pg"], out_split["adam_bank"]["pg"]],
                       "routeA_vs_B_plain_ss": [res_J["plain_ss_pg"], out_split["plain_ss"]["pg"]]},
            "J": res_J, "split": out_split, "keys": keys,
            "mb": {"norm_med": float(np.median(norms)), "c_med": float(np.median(c_row)),
                   "comp_norm_med": {k: float(np.median(v)) for k, v in comp_norm.items()},
                   "G14_mean": float(np.mean(G14)), "G14_sd": float(np.std(G14, ddof=1)), "n": len(G14)},
            "rollout": {"mu14_mean": float(mu_roll.mean()), "tail_deg_mean": float(np.degrees(raw_f[:, OBS_TAIL]).mean()),
                        "eps": rc.eps, "keep_rows": int(keep_f.sum()), "std14": float(sig64[TAIL]),
                        "dV_dq3_deg": float((gz * sc)[unclipped].mean()), "dmu_dq3_deg": float((gm * sc)[unclipped].mean()),
                        "corr_tail_tep": float(np.corrcoef(raw_f[:, OBS_TAIL], tep_f)[0, 1])},
            "sec": {"collect": t_col, "total": time.time() - t0},
        }
        stem.write_text(json.dumps(res, default=float))
        print(json.dumps({"seed": seed, "checks": res["checks"], "pg_adam_roll": out_split["adam_roll"]["pg"],
                          "sec": res["sec"], "hp_mismatch": res["hp_mismatch"]}, default=float), flush=True)
        vn.close()


if __name__ == "__main__":
    main()
