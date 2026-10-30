"""VERIFY TESTS 2-3: my own simulation (no training).  One process; run phases in order.

PYTHONPATH=<policies/code_copy> python -B vp_sim.py PHASE
  layout   live layout and illusion checks (obs indices, actuator 14, forward axis, rotated-quaternion obs)
  roll     deterministic rollouts of the named policies on MY seeds (5001-5010), 1,000 steps: settled states
           (obs + qpos at steps 500..1000 by 100), per-step actions, window means (steps 501-1000)
  illus    closed-loop pitch illusion (+1.1 deg nose-up seen; gravity, accelerometer, prey direction) on 3 s42 ends
           and -1.1 deg on one s80 end, same seeds; plant untouched
  plant    test 3 check on one s42 end (s42_A_1111) and one s80 end (s80_A_1311) and the statue:
           (a) closed loop, entry 14 clamped to a in {-1,-0.5,0,0.5,1}, policy drives the rest;
           (b) open loop, the policy's own recorded actions replayed with entry 14 replaced by a.
Everything is deterministic (model.predict(normalize_obs(obs), deterministic=True)), stance config constructor
(reset noise 0.05).  Checkpoints opened read only.  Written by the verifier; no code imported from posture_policy.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.heading_observation import heading_free_root_segments
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
RUNS = SCR / "r10round/ecs/runs"
OUT = SCR / "com_revisit/notes/verify_posture"
MID = ModelId("tyrannosaurus_rex", "anatomical")
CTOR = dict(load_stage_config(MID, "stance")["env_kwargs"])
SEEDS = list(range(5001, 5011))
POL = {
    "start_s42": ("s42_A_1101", 3145728),
    "s42_A_1111": ("s42_A_1111", 4718592),
    "s42_B_1211": ("s42_B_1211", 4718592),
    "s42_A_1151": ("s42_A_1151", 4718592),
    "s42_B_1241": ("s42_B_1241", 4718592),
    "s80_A_1311": ("s80_A_1311", 5767168),
    "s80_B_1421": ("s80_B_1421", 5767168),
    "start_s80": ("s80_A_1301", 4194304),
}

for _n in [f"s42_A_{n}" for n in range(1101, 1152, 10)] + [f"s42_B_{n}" for n in range(1201, 1252, 10)]:
    POL.setdefault(_n, (_n, 4718592))
for _n in [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)]:
    POL.setdefault(_n, (_n, 5767168))


def factory():
    return env_class(MID)(**CTOR)


def load(who):
    run, steps = POL[who]
    m = RUNS / run / "01_stance" / "models"
    model, norm, _ = load_sb3_checkpoint(str(m / f"stage1_{steps}_steps.zip"), str(m / f"stage1_vecnormalize_{steps}_steps.pkl"), factory)
    return model, norm


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def pitch_up_deg_from_quat(q):
    """Nose-up pitch of the pelvis +x axis above the horizontal (deg), from the root wxyz quaternion."""
    w, x, y, z = q / np.linalg.norm(q)
    fx_z = 2.0 * (x * z - w * y)  # R[2,0]
    return math.degrees(math.asin(max(-1.0, min(1.0, fx_z))))


def rot_obs_nose_up(obs, deg, channels=(42, 51, 56)):
    """Analytic: body rotated by -deg about its own +y (nose up by deg); body-frame world vectors v' = Ry(a)^T v."""
    a = math.radians(-deg)
    c, s = math.cos(a), math.sin(a)
    RyT = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    out = np.array(obs, dtype=np.float64, copy=True)
    for i0 in channels:
        out[..., i0:i0 + 3] = out[..., i0:i0 + 3] @ RyT.T
    return out.astype(np.float32)


def jaddr(m, name):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])


# --------------------------------------------------------------------------------------------------- layout
def phase_layout():
    env = factory()
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    m, d = u.model, u.data
    L = []
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    adr = [int(m.jnt_qposadr[j]) for j in range(1, m.njnt)]
    L.append(f"joints in qpos[7:] order (obs index = qpos adr - 7): " + ", ".join(f"{n}:{a-7}" for n, a in zip(jn, adr)))
    an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    L.append("actuators: " + ", ".join(f"{i}:{n}" for i, n in enumerate(an)))
    L.append(f"actuator 14 = {an[14]}; its joint = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[14,0]))}"
             f"; gain/bias {m.actuator_gainprm[14,:3]} {m.actuator_biasprm[14,:3]}; ctrlrange {m.actuator_ctrlrange[14]}")
    obs, _ = env.reset(seed=5001)
    for _ in range(300):
        obs, *_ = env.step(np.zeros(15, dtype=np.float32))
    mujoco.mj_kinematics(m, d)
    # obs joint channels vs qpos
    L.append(f"obs[0:21] == qpos[7:28]: max diff {np.abs(obs[:21] - d.qpos[7:28]).max():.2e}; obs[6] - qpos[tail_3_pitch] "
             f"{obs[6] - d.qpos[jaddr(m, 'tail_3_pitch')]:.2e}")
    # forward axis: head (skull) position in the pelvis frame
    pel = u.pelvis_id
    R = d.xmat[pel].reshape(3, 3)
    for b in ("skull", "tail_5", "r_thigh"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        L.append(f"  {b} in pelvis frame: {np.round(R.T @ (d.xpos[bid] - d.xpos[pel]), 3).tolist()}")
    q = d.qpos[3:7].copy()
    L.append(f"pelvis nose-up pitch from quaternion {pitch_up_deg_from_quat(q):+.4f} deg; asin(-obs g_x) "
             f"{math.degrees(math.asin(-obs[42])):+.4f} deg; obs g = {np.round(obs[42:45], 5).tolist()}")
    # rotated quaternion check: rebuild gravity and prey channels with the env's own code and compare to analytic
    for deg in (+1.1, -1.1, +2.0):
        a = math.radians(-deg)
        qy = np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])
        qpos2 = d.qpos.copy()
        qpos2[3:7] = qmul(q, qy)
        prey_rel = d.mocap_pos[0] - d.xpos[pel]
        g2, _hv, p2 = heading_free_root_segments(qpos2, d.qvel[0:3], prey_rel / np.linalg.norm(prey_rel))
        ana = rot_obs_nose_up(obs, deg)
        L.append(f"  nose-up {deg:+.1f}: quat pitch {pitch_up_deg_from_quat(qpos2[3:7]) - pitch_up_deg_from_quat(q):+.4f} deg;"
                 f" |g_env - g_analytic| {np.abs(g2 - ana[42:45]).max():.1e}; |prey_env - prey_analytic| "
                 f"{np.abs(p2 - ana[56:59]).max():.1e}; obs pitch change {math.degrees(math.asin(-ana[42])) - math.degrees(math.asin(-obs[42])):+.4f}")
    # accelerometer at rest ~ -9.81 g ?
    L.append(f"accelerometer / -9.81 vs g: {np.round(obs[51:54] / -9.81, 4).tolist()} vs {np.round(obs[42:45], 4).tolist()}")
    L.append(f"prey direction obs {np.round(obs[56:59], 4).tolist()}, distance {obs[59]:.3f}")
    txt = "\n".join(L)
    (OUT / "vp_layout.txt").write_text(txt + "\n")
    print(txt)


# --------------------------------------------------------------------------------------------------- rollouts
def episode(env, seed, act_fn, steps=1000, record=False):
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    m, d = u.model, u.data
    a3 = jaddr(m, "tail_3_pitch")
    obs, _ = env.reset(seed=seed)
    rec = dict(obs=[], qpos=[], acts=[], pitch=[], q3=[], ret=0.0, steps=0, term=False)
    for k in range(1, steps + 1):
        a = act_fn(obs, k)
        rec["acts"].append(np.asarray(a, dtype=np.float64).copy())
        obs, r, term, trunc, info = env.step(a)
        rec["ret"] += float(r)
        rec["steps"] = k
        rec["pitch"].append(pitch_up_deg_from_quat(d.qpos[3:7].copy()))
        rec["q3"].append(math.degrees(d.qpos[a3]))
        if record and k % 100 == 0 and k >= 500:
            rec["obs"].append(obs.copy())
            rec["qpos"].append(d.qpos.copy())
        if term or trunc:
            rec["term"] = bool(term)
            break
    rec["acts"] = np.array(rec["acts"])
    return rec


def summarize(rec):
    a = np.clip(rec["acts"], -1, 1)
    w = slice(500, 1000)
    return dict(steps=rec["steps"], term=rec["term"], ret=rec["ret"], cmd14=float(a[w, 14].mean()),
                railed=bool(a[w, 14].mean() >= 0.99), q3=float(np.mean(rec["q3"][500:1000])),
                pitch=float(np.mean(rec["pitch"][500:1000])))


def policy_fn(model, norm, illusion=0.0, clamp=None):
    def f(obs, k):
        o = obs if illusion == 0.0 else rot_obs_nose_up(obs, illusion)
        a = model.predict(norm.normalize_obs(o), deterministic=True)[0]
        a = np.array(a, dtype=np.float32, copy=True)
        if clamp is not None:
            a[14] = clamp
        return a
    return f


def phase_roll(whos):
    env = factory()
    out = open(OUT / "vp_roll.jsonl", "a")
    for who in whos:
        model, norm = load(who)
        obs_all, qpos_all = [], []
        acts = {}
        t0 = time.time()
        for seed in SEEDS:
            rec = episode(env, seed, policy_fn(model, norm), record=True)
            s = summarize(rec)
            s.update(who=who, seed=seed, arm="base")
            out.write(json.dumps(s) + "\n")
            out.flush()
            if rec["steps"] == 1000:
                obs_all += rec["obs"]
                qpos_all += rec["qpos"]
            acts[str(seed)] = rec["acts"].astype(np.float32)
        np.savez_compressed(OUT / f"settled_{who}.npz", obs=np.array(obs_all), qpos=np.array(qpos_all), **{f"acts_{k}": v for k, v in acts.items()})
        print(who, f"{time.time()-t0:.1f}s", flush=True)
    out.close()


def phase_illus():
    env = factory()
    out = open(OUT / "vp_illus.jsonl", "a")
    for who, deg in (("s42_A_1111", 1.1), ("s42_B_1211", 1.1), ("s42_B_1241", 1.1), ("s42_A_1151", 1.1),
                     ("s42_A_1151", -1.1), ("s80_A_1311", -1.1)):
        model, norm = load(who)
        for seed in SEEDS:
            rec = episode(env, seed, policy_fn(model, norm, illusion=deg))
            s = summarize(rec)
            s.update(who=who, seed=seed, arm=f"illusion{deg:+.1f}")
            out.write(json.dumps(s) + "\n")
            out.flush()
        print(who, deg, flush=True)
    out.close()


def phase_plant():
    env = factory()
    out = open(OUT / "vp_plant.jsonl", "a")
    seeds = SEEDS[:5]
    for who in ("s42_A_1111", "s80_A_1311", "statue"):
        if who != "statue":
            model, norm = load(who)
            st = np.load(OUT / f"settled_{who}.npz")
        for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for seed in seeds:
                if who == "statue":
                    def f(obs, k, a=a):
                        z = np.zeros(15, dtype=np.float32)
                        z[14] = a
                        return z
                    rec = episode(env, seed, f)
                    s = summarize(rec)
                    s.update(who=who, seed=seed, arm="statue_clamp", a=a)
                    out.write(json.dumps(s) + "\n")
                    continue
                rec = episode(env, seed, policy_fn(model, norm, clamp=a))
                s = summarize(rec)
                s.update(who=who, seed=seed, arm="closed_clamp", a=a)
                out.write(json.dumps(s) + "\n")
                base_acts = st[f"acts_{seed}"].astype(np.float32)

                def g(obs, k, base_acts=base_acts, a=a):
                    x = base_acts[min(k - 1, len(base_acts) - 1)].copy()
                    x[14] = a
                    return x
                rec = episode(env, seed, g)
                s = summarize(rec)
                s.update(who=who, seed=seed, arm="open_replay", a=a)
                out.write(json.dumps(s) + "\n")
                out.flush()
        # unreplaced replay check: must reproduce the base return exactly
        if who != "statue":
            for seed in seeds[:2]:
                base_acts = st[f"acts_{seed}"].astype(np.float32)
                rec = episode(env, seed, lambda obs, k, b=base_acts: b[min(k - 1, len(b) - 1)].copy())
                s = summarize(rec)
                s.update(who=who, seed=seed, arm="replay_unchanged")
                out.write(json.dumps(s) + "\n")
        print(who, flush=True)
    out.close()


if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "layout":
        phase_layout()
    elif ph == "roll":
        phase_roll(sys.argv[2].split(","))
    elif ph == "illus":
        phase_illus()
    elif ph == "plant":
        phase_plant()
