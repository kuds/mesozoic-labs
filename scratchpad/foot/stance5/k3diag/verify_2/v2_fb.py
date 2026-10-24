"""verify_2 COUNTERFACTUAL copy of v2_probe.py (REPORT ONLY): adds an external differential hip-pitch feedback
fb = -kfb (dq - dqref) - kv d(dq)/dt to the policy mean (r += fb/2, l -= fb/2) before the command clip; kfb = kv = 0
reproduces v2_probe.py.
 the judge's joint-space noise loop (judge_stance5/4.noise_matched, space "joint", sigma =
frozen control_sigma[seed], spawns 3042..3061, rng default_rng(episode seed), a = clip(clip(mu) + sigma z)), plus a
read-only probe of the policy itself at every visited state: the open-loop sensitivity of the hip-pitch command
difference d_mu = mu_r_hp - mu_l_hp to a differential hip-pitch observation (raw obs q_r + h, q_l - h, and -h; central
difference; per rad of q_r - q_l), and the same for the hip-pitch velocity channels (per rad/s).  The probe passes do
not touch the env or the rng.  Checkpoints are loaded only via the judge's J.load_checkpoint (VecNormalize applied, the
deterministic mean).  Conditions: full (lengths/terms/rewards must equal the judge file), noise0, hpA (z_l = -z_r, hip
pitch only: k3roll's hp_anti), hpD (the actual hip-pitch noise projected on its differential mode, hip pitch only).

    iso.py v2_probe.py --arm LTSB --run stance5_LTSB_s42 --steps 6291456 --conds full noise0 hpA hpD --out-dir OUT
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
RHP, LHP = 3, 7
H_POS, H_VEL = 0.01, 0.05


def load_judge(arm):
    name = "judge_stance5" if arm == "LTSB" else "judge_stance4"
    spec = importlib.util.spec_from_file_location(name, JUDGES[arm])
    J = importlib.util.module_from_spec(spec)
    sys.modules[name] = J
    spec.loader.exec_module(J)
    return J


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
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kfb", type=float, default=0.0)   # added differential gain: d(cmd_r - cmd_l)/d(q_r - q_l) += -kfb
    ap.add_argument("--kv", type=float, default=0.0)    # added differential damping per rad/s
    ap.add_argument("--dqref", type=float, default=0.0)  # reference q_r - q_l (rad)
    a = ap.parse_args()
    J = load_judge(a.arm)
    import mujoco
    import torch
    from stable_baselines3.common.policies import BaseModel

    torch.set_num_threads(1)
    spec = J.ARMS[a.arm]
    clone = spec["clone"]
    J.isolate(clone)
    git = J.git_state(clone)
    assert git["head"] == spec["head"] and not git["status"], git
    ctx = J.Ctx(a.arm)
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder

    seed = int(a.run.rsplit("_s", 1)[1])
    stage_dir = os.path.join(ROOTS[a.arm], "runs", a.run, "01_stance")
    with open(J.FROZEN_PARAMS, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == J.FROZEN_PARAMS_SHA256
    sigma = np.asarray(J.read_json(J.FROZEN_PARAMS)["criteria"]["H4"]["control_sigma"][str(seed)]["vector"], dtype=np.float64)
    os.makedirs(a.out_dir, exist_ok=True)
    for st in a.steps:
        steps = J.parse_steps(st)
        _, zp, vp = J.checkpoint_paths(os.path.join(stage_dir, "models"), steps)
        jd = J.read_json(os.path.join(ROOTS[a.arm], "judged", a.run, f"judge_{st}.json"))
        jn = jd["noise_matched_control_joint"]
        assert jd["model_sha256"] == sha(zp) and jd["vecnorm_sha256"] == sha(vp)
        assert np.array_equal(np.asarray(jn["sigma_vector"]), sigma)
        model, normalizer = J.load_checkpoint(ctx, zp, vp)
        pol = model.policy
        own = np.exp(pol.log_std.detach().cpu().numpy().astype(np.float64))
        vmean, vvar = np.asarray(normalizer.obs_rms.mean), np.asarray(normalizer.obs_rms.var)
        eps, clip_obs = normalizer.epsilon, normalizer.clip_obs

        def mu_of(raw_batch):
            o = np.clip((raw_batch - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
            with torch.no_grad():
                f = BaseModel.extract_features(pol, torch.as_tensor(o), pol.pi_features_extractor)
                return pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy().astype(np.float64)

        for cond in a.conds:
            env = ctx.env_class(**ctx.env_kwargs)
            u = env.unwrapped
            m = u.model
            low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
            jid = [int(m.actuator_trnid[i, 0]) for i in (RHP, LHP)]
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in (RHP, LHP)]
            assert names == ["r_hip_pitch_act", "l_hip_pitch_act"], names
            qi = [int(m.jnt_qposadr[j]) - 7 for j in jid]
            vi = [(m.nq - 7) + int(m.jnt_dofadr[j]) - 6 for j in jid]
            rec = SubstepContactRecorder(env, Morphology.from_env(env, ctx.mid)).attach()
            E, T = 20, 1000
            arr = {k: np.full((E, T), np.nan, dtype=np.float32) for k in ("Jq", "Jv", "Jq_common", "dmu", "dq", "stagger", "roll", "dmu_act")}
            lengths, terms, rewards = [], [], []
            t0 = time.monotonic()
            try:
                for i in range(E):
                    s = 3042 + i
                    rng = np.random.default_rng(s)
                    obs, _ = env.reset(seed=s)
                    # sanity: the obs indices are the hip-pitch joint angles
                    assert abs(float(obs[qi[0]]) - float(u.data.qpos[qi[0] + 7])) < 1e-5
                    n, rew = 0, 0.0
                    while True:
                        raw = np.asarray(obs, dtype=np.float64)
                        B = np.repeat(raw[None], 7, 0)
                        B[1, qi[0]] += H_POS; B[1, qi[1]] -= H_POS
                        B[2, qi[0]] -= H_POS; B[2, qi[1]] += H_POS
                        B[3, vi[0]] += H_VEL; B[3, vi[1]] -= H_VEL
                        B[4, vi[0]] -= H_VEL; B[4, vi[1]] += H_VEL
                        B[5, qi[0]] += H_POS; B[5, qi[1]] += H_POS
                        B[6, qi[0]] -= H_POS; B[6, qi[1]] -= H_POS
                        MU = mu_of(B.astype(np.float32).astype(np.float64))
                        # the judge's own single-row mean (bit-identical path) for the step itself
                        o = np.clip((obs - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
                        with torch.no_grad():
                            f = BaseModel.extract_features(pol, torch.as_tensor(o[None]), pol.pi_features_extractor)
                            mu = pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy()[0].astype(np.float64)
                        d = MU[:, RHP] - MU[:, LHP]
                        arr["Jq"][i, n] = (d[1] - d[2]) / (4 * H_POS)
                        arr["Jv"][i, n] = (d[3] - d[4]) / (4 * H_VEL)
                        arr["Jq_common"][i, n] = (d[5] - d[6]) / (2 * H_POS)
                        arr["dmu"][i, n] = mu[RHP] - mu[LHP]
                        arr["dq"][i, n] = raw[qi[0]] - raw[qi[1]]
                        fb = -a.kfb * (raw[qi[0]] - raw[qi[1]] - a.dqref) - a.kv * (raw[vi[0]] - raw[vi[1]])
                        mu_fb = mu.copy()
                        mu_fb[RHP] += fb / 2
                        mu_fb[LHP] -= fb / 2
                        cmd = np.clip(mu_fb, low, high)
                        z = rng.standard_normal(cmd.shape)
                        if cond == "full":
                            nz = sigma * z
                        elif cond == "noise0":
                            nz = np.zeros(15)
                        elif cond == "hpA":
                            nz = np.zeros(15)
                            nz[RHP], nz[LHP] = sigma[RHP] * z[RHP], -sigma[LHP] * z[RHP]
                        elif cond == "hpD":
                            n_ = sigma * z
                            dd = (n_[RHP] - n_[LHP]) / 2
                            nz = np.zeros(15)
                            nz[RHP], nz[LHP] = dd, -dd
                        else:
                            raise SystemExit(cond)
                        act = np.clip(cmd + nz, low, high)
                        arr["dmu_act"][i, n] = act[RHP] - act[LHP]
                        obs, r, term, trunc, info = env.step(act)
                        rec.end_step(act, float(r))
                        arr["stagger"][i, n] = float(info.get("foot_stagger_m", np.nan))
                        qq = u.data.qpos[3:7]
                        arr["roll"][i, n] = float(np.degrees(np.arcsin(np.clip(2 * (qq[2] * qq[3] + qq[0] * qq[1]), -1, 1))))
                        rew += float(r)
                        n += 1
                        if term or trunc:
                            rec.end_episode(terminated=bool(term), truncated=bool(trunc),
                                            termination_reason=str(info.get("termination_reason", "")))
                            terms.append(str(info.get("termination_reason", "")))
                            break
                    lengths.append(n)
                    rewards.append(rew)
            finally:
                rec.detach()
                env.close()
            L = np.asarray(lengths)
            row = {"run": a.run, "steps": st, "cond": cond, "kfb": a.kfb, "kv": a.kv, "dqref": a.dqref, "falls": int((L < ctx.horizon).sum()),
                   "fall_seeds": [3042 + k for k in range(E) if L[k] < ctx.horizon], "seconds": round(time.monotonic() - t0, 1),
                   "own_sigma_mean": float(own.mean()), "own_sigma_hp": float(own[[RHP, LHP]].mean())}
            if cond == "full":
                jl = [(r["length"], r["termination_reason"]) for r in jn["episodes_detail"]]
                rd = max(abs(r["reward"] - x) for r, x in zip(jn["episodes_detail"], rewards))
                row["matches_judge"] = bool(jl == list(zip(lengths, terms)) and rd < 1e-6)
                row["judge_falls"] = jn["falls"]
                row["reward_max_abs_diff"] = rd
            np.savez_compressed(os.path.join(a.out_dir, f"{a.run}__{st}__{cond}_k{a.kfb:g}_kv{a.kv:g}.npz"), length=L, **arr,
                                meta=np.array(json.dumps(row)))
            print(json.dumps(row), flush=True)
        del model, normalizer


if __name__ == "__main__":
    main()
