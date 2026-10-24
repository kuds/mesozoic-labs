"""verify_3 (REPORT ONLY): independent re-roll of stance5 s42's joint-space noise test, its deterministic tail_3 share and
the tail_3-held-at-0 counterfactual, per checkpoint.

Run: ISO_WT=<clone> nice -n 19 python3 iso.py v3roll.py --arm LTSB --run stance5_LTSB_s42 --steps ... --out OUT.jsonl --npz-dir D

Loading: the judge file's own isolate / Ctx / checkpoint_paths / load_checkpoint (judge imported, never edited); the
loop is judge.noise_matched (space "joint") written out again: rng = default_rng(episode seed), obs normalised with the
checkpoint's VecNormalize stats, mu = action_net(forward_actor(extract_features(o))), cmd = clip(mu),
a = clip(cmd + sigma * mask * z) with z = rng.standard_normal(15) drawn every step (so every condition sees the same z).
The judge's SubstepContactRecorder is attached exactly as the judge does (so the env step path is the judge's).
Conditions:
  det     mask 0 (deterministic: a = clip(mu))
  full    mask 1 (the judge's test; must reproduce the judge file's lengths, terminations and rewards)
  t3hold0 cmd[14] = 0 before the noise (tail_3 command mean held at 0; noise kept on all 15 channels)
  t3hold0det  cmd[14] = 0, mask 0
  t3hold0nz   cmd[14] = 0 and no noise on tail_3 (a[14] = 0 exactly), noise kept on the other 14
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
JUDGES = {"LTSB": SCRATCH + "/foot/stance5/judge/judge_stance5.py", "LTS": SCRATCH + "/foot/stance4/judge/judge_stance4.py"}
ROOTS = {"LTSB": SCRATCH + "/foot/stance5", "LTS": SCRATCH + "/foot/stance4"}
SEED0, E, W0 = 3042, 20, 200


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", default=["det", "full", "t3hold0"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz-dir", required=True)
    a = ap.parse_args()
    name = "judge_stance5" if a.arm == "LTSB" else "judge_stance4"
    spec = importlib.util.spec_from_file_location(name, JUDGES[a.arm])
    J = importlib.util.module_from_spec(spec)
    sys.modules[name] = J
    spec.loader.exec_module(J)
    import torch
    from stable_baselines3.common.policies import BaseModel

    torch.set_num_threads(1)
    arm = J.ARMS[a.arm]
    J.isolate(arm["clone"])
    git = J.git_state(arm["clone"])
    assert git["head"] == arm["head"] and not git["status"], git
    ctx = J.Ctx(a.arm)
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder

    seed = int(a.run.rsplit("_s", 1)[1])
    with open(J.FROZEN_PARAMS, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == J.FROZEN_PARAMS_SHA256
    sigma = np.asarray(J.read_json(J.FROZEN_PARAMS)["criteria"]["H4"]["control_sigma"][str(seed)]["vector"], dtype=np.float64)
    stage_dir = os.path.join(ROOTS[a.arm], "runs", a.run, "01_stance")
    os.makedirs(a.npz_dir, exist_ok=True)
    for st in a.steps:
        steps = J.parse_steps(st)
        _, zp, vp = J.checkpoint_paths(os.path.join(stage_dir, "models"), steps)
        jd = J.read_json(os.path.join(ROOTS[a.arm], "judged", a.run, f"judge_{st}.json"))
        assert jd["model_sha256"] == sha(zp) and jd["vecnorm_sha256"] == sha(vp)
        jn = jd["noise_matched_control_joint"]
        assert np.array_equal(np.asarray(jn["sigma_vector"]), sigma)
        model, normalizer = J.load_checkpoint(ctx, zp, vp)
        pol = model.policy
        own = np.exp(pol.log_std.detach().cpu().numpy().astype(np.float64))
        vmean, vvar = np.asarray(normalizer.obs_rms.mean), np.asarray(normalizer.obs_rms.var)
        eps, clip_obs = normalizer.epsilon, normalizer.clip_obs
        for cond in a.conds:
            mask = 0.0 if cond in ("det", "t3hold0det") else 1.0
            if cond == "t3hold0nz":  # tail_3 held at exactly 0: mean 0 and no tail_3 noise (z still drawn for all 15)
                mask = np.ones(15)
                mask[14] = 0.0
            hold = cond.startswith("t3hold0")
            t0 = time.monotonic()
            env = ctx.env_class(**ctx.env_kwargs)
            u = env.unwrapped
            m = u.model
            low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
            import mujoco
            assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 14) == "tail_3_pitch_act"
            q14 = int(m.jnt_qposadr[int(m.actuator_trnid[14, 0])])
            rec = SubstepContactRecorder(env, Morphology.from_env(env, ctx.mid)).attach()
            mu_t3 = np.full((E, 1000), np.nan)
            act_t3 = np.full((E, 1000), np.nan)
            q_t3 = np.full((E, 1000), np.nan)
            lengths, rewards, terms = [], [], []
            try:
                for i in range(E):
                    s = SEED0 + i
                    rng = np.random.default_rng(s)
                    obs, _ = env.reset(seed=s)
                    n, R = 0, 0.0
                    while True:
                        o = np.clip((obs - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
                        with torch.no_grad():
                            f = BaseModel.extract_features(pol, torch.as_tensor(o[None]), pol.pi_features_extractor)
                            mu = pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy()[0].astype(np.float64)
                        cmd = np.clip(mu, low, high)
                        if hold:
                            cmd[14] = 0.0
                        z = rng.standard_normal(cmd.shape)
                        act = np.clip(cmd + (sigma * mask) * z, low, high) if np.any(mask) else cmd
                        obs, r, term, trunc, info = env.step(act)
                        rec.end_step(act, float(r))
                        mu_t3[i, n] = mu[14]
                        act_t3[i, n] = act[14]
                        q_t3[i, n] = float(u.data.qpos[q14])
                        R += float(r)
                        n += 1
                        if term or trunc:
                            tr = rec.end_episode(terminated=bool(term), truncated=bool(trunc),
                                                 termination_reason=str(info.get("termination_reason", "")))
                            break
                    lengths.append(n), rewards.append(R), terms.append(str(getattr(tr, "termination_reason", "")))
            finally:
                rec.detach()
                env.close()
            L = np.asarray(lengths)
            fall = L < ctx.horizon
            win = np.zeros((E, 1000), bool)
            pre = np.zeros((E, 1000), bool)
            for e in range(E):
                win[e, W0:L[e]] = True
                pre[e, W0:max(W0, L[e] - 100 if fall[e] else L[e])] = True
            cm = np.clip(mu_t3, -1, 1)

            def share(x, msk, thr, absval=False):
                v = x[msk]
                v = np.abs(v) if absval else v
                return float((v >= thr).mean()) if v.size else float("nan")

            row = {"run": a.run, "steps": st, "cond": cond, "falls": int(fall.sum()),
                   "fall_seeds": [SEED0 + e for e in range(E) if fall[e]], "lengths": L.tolist(),
                   "terms": terms, "rewards": rewards,
                   "t3_mu_ge09_win": share(cm, win, 0.9), "t3_absmu_ge09_win": share(cm, win, 0.9, True),
                   "t3_mu_ge099_win": share(cm, win, 0.99), "t3_mu_ge09_pre": share(cm, pre, 0.9),
                   "t3_act_ge09_win": share(act_t3, win, 0.9), "t3_absact_ge09_win": share(act_t3, win, 0.9, True),
                   "t3_mu_mean_win": float(np.nanmean(cm[win])), "t3_rawmu_mean_win": float(np.nanmean(mu_t3[win])),
                   "t3_q_mean_win": float(np.nanmean(q_t3[win])),
                   "t3_mu_ge09_win_fullH_only": share(cm, win & ~fall[:, None], 0.9),
                   "t3_mu_ge09_win_falls_only": share(cm, win & fall[:, None], 0.9),
                   "own_sigma_mean": float(own.mean()), "own_sigma_t3": float(own[14]),
                   "seconds": round(time.monotonic() - t0, 1)}
            if cond == "full":
                jl = [(r["seed"], r["length"], r["termination_reason"]) for r in jn["episodes_detail"]]
                ml = [(SEED0 + e, int(L[e]), terms[e]) for e in range(E)]
                rd = max(abs(r["reward"] - x) for r, x in zip(jn["episodes_detail"], rewards))
                row["judge_match"] = {"lengths_terms_equal": jl == ml, "reward_max_abs_diff": rd,
                                      "judge_falls": jn["falls"], "judge_fall_seeds": jn["fall_seeds"]}
            np.savez_compressed(os.path.join(a.npz_dir, f"{a.run}__{st}__{cond}.npz"), mu_t3=mu_t3.astype(np.float32),
                                act_t3=act_t3.astype(np.float32), q_t3=q_t3.astype(np.float32), length=L)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("run", "steps", "cond", "falls", "fall_seeds", "t3_mu_ge09_win", "seconds")}),
                  flush=True)
        del model, normalizer
    J.modules_under(arm["clone"])


if __name__ == "__main__":
    main()
