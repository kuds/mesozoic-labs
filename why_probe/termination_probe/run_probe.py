"""Training-like rollouts of one checkpoint pair, recording what the termination-risk lens needs.  No training: the
checkpoint is opened read only, no optimizer exists, no gradient step is taken (autograd is used only to read the
critic's local slope).

    PY -I -B run_probe.py --label LABEL --seeds 7001:7129 --out DIR [--hold V] [--smoke]

THE POLICY (training-like, as SB3's collect_rollouts acts): the observation is normalised with the pair's frozen
VecNormalize statistics (training False), the policy's Gaussian is SAMPLED (distribution.get_actions(deterministic
False)), the sample is clipped to the action box and stepped.  torch.manual_seed(seed) at every episode start, so an
episode is reproducible on its own.  Fresh env per condition, env.reset(seed=seed) per episode; the env is the stance
config's own constructor (env_class(mid)(**load_stage_config(mid, "stance")["env_kwargs"])), the reader's.
--hold V: the clamp test's override (com_revisit/notes/clamp/run_clamp.py): entry 14 (tail_3_pitch_act) of the
stepped action is replaced by V every step (the env then filters, scales and prices the held command).

PER STEP (state s_t before the action at step index t = 0..T-1; "all" = every step, "sub" = t % 4 == 0):
  all: ep, t, r (raw reward of the transition from s_t), V (the critic's value of s_t: the EFFECTIVE critic, i.e. with
       normalised obs[6] zeroed for the critic_blind_tail3 run, bs_hooks.critic_blind_class's form), Vraw (blind only:
       the value MLP without the zeroing), dV3 (finite difference of V in the tail_3 angle, +-0.25 deg on the RAW
       observation, all else fixed, normalised with the frozen statistics; per degree; effective critic), dV3raw
       (blind only), q3 (tail_3 angle, deg), tip (tail_tip site z, m), low (lowest point of tail_3/4/5 capsules, m),
       pelvis (pelvis z, m), fwdz (pelvis forward-axis z: < 0 nose down), tilt (rad), cmd (stepped action 14), own (the
       policy's own clipped sampled action 14), mu (the Gaussian mean of action 14), headz (head tip z), the reward
       terms reward_tail_home_pose, reward_action_saturation, reward_energy, reward_alive (raw).
  sub: z (normalised observation, float16), g (autograd dV/dz at G_IDX, effective critic), graw (blind only),
       low3 (each capsule's lowest point), and forward-kinematics sensitivities at the state's own pose (scratch MjData,
       +-0.5 deg): dtip_dq1/2/3 and dlow_dq1/2/3 in m per degree (q1 = tail_1_pitch, q2 = tail_2_pitch, q3 =
       tail_3_pitch).
  terminated episodes: their whole raw observation sequence (float32) as term_obs_<ep>.
PER EPISODE (episodes.jsonl): seed, steps, terminated, truncated, reason, return (raw), V_final (critic value of the
final observation, for the truncation bootstrap SB3 applies), reward-term sums, hold, wall seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_common as T  # noqa: E402

TERMS = ("reward_tail_home_pose", "reward_action_saturation", "reward_energy", "reward_alive")
DELTA_DEG = 0.25
FK_DEG = 0.5
SUB = 4
# The autograd gradient is kept for these observation indices only (disk): the four tail joint angles (3 tail_1_pitch,
# 4 tail_1_yaw, 5 tail_2_pitch, 6 tail_3_pitch), their velocities (24-27) and projected gravity (42-44).
G_IDX = (3, 4, 5, 6, 24, 25, 26, 27, 42, 43, 44)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=sorted(T.CKPT))
    ap.add_argument("--seeds", required=True, help="A:B, half-open")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", type=float, default=None)
    ap.add_argument("--smoke", action="store_true", help="print timing and shapes only")
    ap.add_argument("--lite", action="store_true", help="omit sub_z and sub_g (the clamp arms)")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.seeds.split(":"))
    seeds = list(range(lo, hi))
    out = Path(a.out).resolve()
    if T.ME.resolve() not in out.parents:
        raise SystemExit(f"{out} is not under {T.ME}")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.label}_{'free' if a.hold is None else 'hold' + str(a.hold)}"
    origin = T.setup_simulation(tag)

    import mujoco
    import numpy as np
    import torch

    factory = T.env_factory()
    env = factory()
    u = env.unwrapped
    m, d = u.model, u.data
    assert m.actuator(T.ACT_TAIL3).name == "tail_3_pitch_act"
    assert m.joint("tail_3_pitch").qposadr[0] - 7 == T.OBS_TAIL3
    model, norm = T.load_pair(a.label, factory)
    policy = model.policy
    policy.set_training_mode(False)
    blind = T.CKPT[a.label][3] == "blind"
    mean = np.asarray(norm.obs_rms.mean, dtype=np.float64)
    std = np.sqrt(np.asarray(norm.obs_rms.var, dtype=np.float64) + float(norm.epsilon))
    clip = float(norm.clip_obs)
    q_obs = {1: int(m.joint("tail_1_pitch").qposadr[0]) - 7, 2: int(m.joint("tail_2_pitch").qposadr[0]) - 7,
             3: int(m.joint("tail_3_pitch").qposadr[0]) - 7}
    q_adr = {k: v + 7 for k, v in q_obs.items()}
    tail_geoms = [u.tail_3_geom_id, u.tail_4_geom_id, u.tail_5_geom_id]
    tip_site, pelvis_id, head_site = u.tail_tip_site_id, u.pelvis_id, u.head_tip_site_id
    dk = mujoco.MjData(m)

    def zfun(X):
        return np.clip((np.asarray(X, dtype=np.float64) - mean) / std, -clip, clip).astype(np.float32)

    def values(Z, effective=True):
        Zt = torch.as_tensor(np.array(Z, dtype=np.float32, copy=True))
        if effective and blind:
            Zt[..., T.OBS_TAIL3] = 0.0
        with torch.no_grad():
            return policy.predict_values(Zt)[:, 0].numpy().astype(np.float64)

    def grads(Z, effective=True):
        Zt = torch.as_tensor(np.array(Z, dtype=np.float32, copy=True)).requires_grad_(True)
        Zin = Zt
        if effective and blind:
            mask = torch.ones_like(Zt)
            mask[..., T.OBS_TAIL3] = 0.0
            Zin = Zt * mask
        v = policy.predict_values(Zin).sum()
        return torch.autograd.grad(v, Zt)[0].numpy()

    def fd_q3(X, effective=True):
        Xp = np.array(X, dtype=np.float64, copy=True)
        Xm = np.array(X, dtype=np.float64, copy=True)
        Xp[:, T.OBS_TAIL3] += math.radians(DELTA_DEG)
        Xm[:, T.OBS_TAIL3] -= math.radians(DELTA_DEG)
        return (values(zfun(Xp), effective) - values(zfun(Xm), effective)) / (2 * DELTA_DEG)

    def fk_sens(qpos_full):
        res = []
        for k in (1, 2, 3):
            vals = []
            for sgn in (-1.0, 1.0):
                dk.qpos[:] = qpos_full
                dk.qpos[q_adr[k]] += sgn * math.radians(FK_DEG)
                mujoco.mj_kinematics(m, dk)
                vals.append((float(dk.site_xpos[tip_site][2]), min(T.tail_capsule_lowest(m, dk, tail_geoms))))
            res.append(((vals[1][0] - vals[0][0]) / (2 * FK_DEG), (vals[1][1] - vals[0][1]) / (2 * FK_DEG)))
        return res   # [(dtip/dq, dlow/dq) for q1, q2, q3], m per degree

    allf = {k: [] for k in ("ep", "t", "r", "V", "Vraw", "dV3", "dV3raw", "q3", "tip", "low", "pelvis", "fwdz", "tilt",
                            "cmd", "own", "mu", "headz", *TERMS)}
    subf = {k: [] for k in ("ep", "t", "z", "g", "graw", "low3", "dtip_dq1", "dtip_dq2", "dtip_dq3", "dlow_dq1",
                            "dlow_dq2", "dlow_dq3")}
    term_obs = {}
    eps_out = []
    t_start = time.time()
    for ei, seed in enumerate(seeds):
        t0 = time.time()
        obs, _ = env.reset(seed=seed)
        torch.manual_seed(seed)
        X, rec = [], {k: [] for k in allf if k not in ("ep", "t", "V", "Vraw", "dV3", "dV3raw")}
        qposs = []
        terminated = truncated = False
        reason, ret, k = None, 0.0, 0
        term_sums = {}
        V_final = None
        while True:
            X.append(np.array(obs, dtype=np.float64, copy=True))
            qpos = d.qpos.copy()
            qposs.append(qpos)
            quat = d.sensordata[u._sensor_quat_start:u._sensor_quat_start + 4].copy()
            rec["q3"].append(math.degrees(float(qpos[q_adr[3]])))
            rec["tip"].append(float(d.site_xpos[tip_site][2]))
            rec["low"].append(min(T.tail_capsule_lowest(m, d, tail_geoms)))
            rec["pelvis"].append(float(d.xpos[pelvis_id][2]))
            rec["fwdz"].append(float(u._quat_to_forward_z(quat)))
            rec["tilt"].append(float(u._quat_to_tilt(quat)))
            rec["headz"].append(float(d.site_xpos[head_site][2]))
            zt = torch.as_tensor(zfun(obs[None]))
            with torch.no_grad():
                dist = policy.get_distribution(zt)
                mu = dist.distribution.mean[0].numpy()
                act = dist.get_actions(deterministic=False)[0].numpy()
            stepped = np.clip(act, -1.0, 1.0)
            own14 = float(stepped[T.ACT_TAIL3])
            if a.hold is not None:
                stepped = stepped.copy()
                stepped[T.ACT_TAIL3] = a.hold
            rec["cmd"].append(float(stepped[T.ACT_TAIL3]))
            rec["own"].append(own14)
            rec["mu"].append(float(mu[T.ACT_TAIL3]))
            obs, r, terminated, truncated, info = env.step(stepped)
            k += 1
            ret += float(r)
            rec["r"].append(float(r))
            for key in TERMS:
                rec[key].append(float(info.get(key, float("nan"))))
            for key, val in info.items():
                if key.startswith("reward_") and isinstance(val, (int, float, np.floating)):
                    term_sums[key] = term_sums.get(key, 0.0) + float(val)
            if terminated:
                reason = str(info.get("termination_reason", "unknown"))
            if terminated or truncated:
                break
        if truncated and not terminated:
            V_final = float(values(zfun(np.asarray(obs, dtype=np.float64)[None]))[0])
        Xa = np.asarray(X)
        Z = zfun(Xa)
        n = len(Xa)
        allf["ep"].append(np.full(n, ei, np.int16))
        allf["t"].append(np.arange(n, dtype=np.int16))
        allf["V"].append(values(Z).astype(np.float32))
        allf["dV3"].append(fd_q3(Xa).astype(np.float32))
        if blind:
            allf["Vraw"].append(values(Z, effective=False).astype(np.float32))
            allf["dV3raw"].append(fd_q3(Xa, effective=False).astype(np.float32))
        for key in rec:
            dt = np.float16 if key in ("cmd", "own", "mu", *TERMS) else np.float32
            allf[key].append(np.asarray(rec[key], dtype=dt))
        idx = np.arange(0, n, SUB)
        subf["ep"].append(np.full(len(idx), ei, np.int16))
        subf["t"].append(idx.astype(np.int16))
        if not a.lite:
            subf["z"].append(Z[idx].astype(np.float16))
            subf["g"].append(grads(Z[idx])[:, list(G_IDX)].astype(np.float32))
            if blind:
                subf["graw"].append(grads(Z[idx], effective=False)[:, list(G_IDX)].astype(np.float32))
        low3, sens = [], []
        for i in idx:
            dk.qpos[:] = qposs[i]
            mujoco.mj_kinematics(m, dk)
            low3.append(T.tail_capsule_lowest(m, dk, tail_geoms))
            sens.append(fk_sens(qposs[i]))
        subf["low3"].append(np.asarray(low3, dtype=np.float32))
        sens = np.asarray(sens, dtype=np.float32)   # (len, 3 joints, 2)
        for j, q in enumerate((1, 2, 3)):
            subf[f"dtip_dq{q}"].append(sens[:, j, 0])
            subf[f"dlow_dq{q}"].append(sens[:, j, 1])
        if terminated:
            term_obs[f"term_obs_{ei}"] = Xa.astype(np.float32)
        eps_out.append({"ep": ei, "seed": seed, "steps": k, "terminated": bool(terminated),
                        "truncated": bool(truncated), "reason": reason if terminated else ("truncated" if truncated
                                                                                            else "open"),
                        "return": ret, "V_final": V_final, "hold": a.hold, "label": a.label,
                        "term_sums": {kk: round(vv, 6) for kk, vv in sorted(term_sums.items())},
                        "wall_s": round(time.time() - t0, 2)})
        if a.smoke:
            print(f"smoke ep {ei} seed {seed}: steps {k} wall {time.time() - t0:.1f}s; arrays ok", flush=True)
    wall = time.time() - t_start
    if a.smoke:
        print(f"smoke total wall {wall:.1f}s for {len(seeds)} episodes")
    arrays = {}
    for key, lst in allf.items():
        if lst:
            arrays["all_" + key] = np.concatenate(lst)
    for key, lst in subf.items():
        if lst:
            arrays["sub_" + key] = np.concatenate(lst)
    arrays.update(term_obs)
    z, p, cfg, form = T.CKPT[a.label]
    meta = {"label": a.label, "zip": z, "pkl": p, "zip_sha256": T.sha256_file(z), "pkl_sha256": T.sha256_file(p),
            "stage_config": cfg, "critic_form": form, "hold": a.hold, "seeds": [lo, hi],
            "num_timesteps": int(model.num_timesteps), "obs_rms_mean": mean.tolist(), "obs_rms_std": std.tolist(),
            "ret_var": float(norm.ret_rms.var), "epsilon": float(norm.epsilon), "clip_obs": clip,
            "clip_reward": float(norm.clip_reward), "gamma": T.GAMMA, "delta_deg": DELTA_DEG, "fk_deg": FK_DEG,
            "sub": SUB, "g_idx": list(G_IDX), "lite": bool(a.lite), "origin": origin, "wall_s": round(wall, 1), "log_std": policy.log_std.detach().numpy().tolist(),
            "version": "run_probe 1.0"}
    stem = out / tag
    np.savez_compressed(f"{stem}.npz", **arrays)
    Path(f"{stem}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    with open(f"{stem}.episodes.jsonl", "w") as fh:
        for e in eps_out:
            fh.write(json.dumps(e) + "\n")
    env.close()
    print(f"{tag}: {len(seeds)} episodes, wall {wall:.1f}s -> {stem}.npz", flush=True)


if __name__ == "__main__":
    main()
