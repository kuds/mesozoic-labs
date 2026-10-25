"""What the tail-up posture is worth under the policy's own noise (report only, no training).

    python -I roll_value.py CKPT_ZIP VECNORM_PKL LABEL OUT_JSON

The env is the stance stage of the cf80ecb1 code copy in ./code (arm LTSB [env] as the stance5 judge loads it, via
judge_stance5.Ctx("LTSB")); the checkpoint loads through judge_stance5.load_checkpoint (SB3 load + frozen VecNormalize).
Conditions: tail_3_pitch_act free ("free"), its command overridden to 0 ("hold0") or 0.9 ("hold09") every step.
Stochastic: N=40 seeds 3042..3081, a = clip(mu + exp(log_std) * z) on all 15 actuators, z from numpy
default_rng(episode seed) drawn for all 15 every step (identical stream in every condition; the judge's own-sigma
noise test arithmetic), then the override. Deterministic: N=20 seeds 3042..3061, a = clip(mu), then the override.
Window = step indices 200..L-1. Reward = the env's own reward r from env.step.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
CODE = os.path.join(HERE, "code")
SEED0 = 3042
WIN0 = 200
T3NAME = "tail_3_pitch_act"
CONDS = {"free": None, "hold0": 0.0, "hold09": 0.9}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    zip_path, pkl_path, label, out_path = sys.argv[1:5]
    spec = importlib.util.spec_from_file_location("judge_stance5", os.path.join(HERE, "judge_stance5.py"))
    J = importlib.util.module_from_spec(spec)
    sys.modules["judge_stance5"] = J
    spec.loader.exec_module(J)
    modules = J.isolate(CODE)
    import mujoco
    import numpy as np
    import torch

    torch.set_num_threads(1)
    ctx = J.Ctx("LTSB")
    model, normalizer = J.load_checkpoint(ctx, zip_path, pkl_path)
    pol = model.policy
    assert not getattr(pol, "use_sde", False)
    sigma = np.exp(pol.log_std.detach().cpu().numpy().astype(np.float64))
    vmean, vvar = np.asarray(normalizer.obs_rms.mean), np.asarray(normalizer.obs_rms.var)
    eps, clip_obs = normalizer.epsilon, normalizer.clip_obs

    env = ctx.env_class(**ctx.env_kwargs)
    low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
    m = env.unwrapped.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    t3 = names.index(T3NAME)
    q3adr = int(m.jnt_qposadr[m.joint("tail_3_pitch").id])
    from stable_baselines3.common.policies import BaseModel

    def mean_action(obs):
        o = np.clip((obs - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
        with torch.no_grad():
            f = BaseModel.extract_features(pol, torch.as_tensor(o[None]), pol.pi_features_extractor)
            return pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy()[0].astype(np.float64)

    def episode(seed, stochastic, override):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        L, ret, win = 0, 0.0, 0.0
        terms_win = {}
        t3cmd, t3mu, q3 = [], [], []
        term_reason = None
        while True:
            mu = mean_action(obs)
            if stochastic:
                act = np.clip(mu + sigma * rng.standard_normal(mu.shape), low, high)
            else:
                act = np.clip(mu, low, high)
            if override is not None:
                act[t3] = override
            obs, r, term, trunc, info = env.step(act)
            r = float(r)
            ret += r
            if L >= WIN0:
                win += r
                t3cmd.append(act[t3])
                t3mu.append(mu[t3])
                q3.append(float(env.unwrapped.data.qpos[q3adr]))
                for k, v in info.items():
                    if k.startswith("reward_") and np.isscalar(v):
                        terms_win[k] = terms_win.get(k, 0.0) + float(v)
            L += 1
            if term or trunc:
                term_reason = str(info.get("termination_reason", ""))
                break
        t3cmd = np.asarray(t3cmd)
        return {"seed": seed, "length": L, "fell": L < ctx.horizon, "terminated": bool(term),
                "termination_reason": term_reason, "return": ret, "window_return": win, "window_steps": len(t3cmd),
                "t3_frac99": float(np.mean(np.abs(t3cmd) >= 0.99)) if len(t3cmd) else None,
                "t3_frac90": float(np.mean(np.abs(t3cmd) >= 0.9)) if len(t3cmd) else None,
                "t3_cmd_mean": float(t3cmd.mean()) if len(t3cmd) else None,
                "t3_mu_mean": float(np.mean(t3mu)) if len(t3mu) else None,
                "t3_mu_gt1": float(np.mean(np.asarray(t3mu) > 1.0)) if len(t3mu) else None,
                "q3_mean": float(np.mean(q3)) if q3 else None,
                "window_terms": terms_win}

    out = {"label": label, "zip": zip_path, "zip_sha256": sha(zip_path), "vecnorm": pkl_path,
           "vecnorm_sha256": sha(pkl_path), "num_timesteps": int(model.num_timesteps), "modules": modules,
           "env_kwargs": J.jsonable(ctx.env_kwargs), "horizon": ctx.horizon, "actuators": names, "t3_index": t3,
           "sigma": sigma.tolist(), "sigma_t3": float(sigma[t3]), "window": [WIN0, "L-1"], "runs": {}}
    t0 = time.monotonic()
    for mode, n in (("det", 20), ("stoch", 40)):
        for cname, ov in CONDS.items():
            rows = [episode(SEED0 + i, mode == "stoch", ov) for i in range(n)]
            out["runs"][f"{mode}_{cname}"] = rows
            print(f"{label} {mode} {cname}: ret {np.mean([r['return'] for r in rows]):.2f} "
                  f"win {np.mean([r['window_return'] for r in rows]):.2f} falls {sum(r['fell'] for r in rows)} "
                  f"t3frac99 {np.mean([r['t3_frac99'] for r in rows if r['t3_frac99'] is not None]):.4f} "
                  f"[{time.monotonic() - t0:.0f}s]", flush=True)
    env.close()
    out["seconds"] = time.monotonic() - t0
    with open(out_path, "x") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
