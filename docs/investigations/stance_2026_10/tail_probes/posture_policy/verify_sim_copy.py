"""Verifier's own clamp rollouts (simulation only; no training; checkpoints opened read only).

PYTHONPATH=<policies/code_copy> python -B verify_sim.py OUT.jsonl --who NAME[,NAME] [--force V] [--seeds a,b,c]
                                                         [--steps N] [--zip Z --pkl P]

Written from scratch; does not import the clamp runner's scripts.  Independent of its loading path too:
the policy is PPO.load(zip, device="cpu") and the observation statistics are the VecNormalize pickle unpickled
directly (pickle.load), normalised by hand as SB3's normalize_obs does (clip((obs-mean)/sqrt(var+eps)), float32).
The override is applied here as D-9 applied it: the policy's own deterministic action, entry 14 replaced by V,
then env.step.  After every step the script asserts what the env actually did with entry 14: the filter state
(_action_filter_state[14]) and the applied servo control (data.ctrl[14]) must equal V's mapping exactly
(V=0 -> 0.0 rad; V=+1 -> ctrlrange max).  Any mismatch is counted and reported.

Per episode it writes the PR #650 reduction at a 5 s window (stance_track_quality, the instrument the runner used)
plus quantities computed here directly from env.data over steps 500-999 (independent of the reduction):
pelvis forward-axis z (pitch proxy, + = nose up), whole-body (root subtree) CoM height above the floor plane,
head_pitch / neck_pitch / tail_3 joint angles, the policy's own command means, and the policy's raw (unclipped)
Gaussian mean for entry 14.  For episodes longer than 1,000 steps it also reduces the first 1,001 samples
(the 5-10 s window) and windows from 10 s and 15 s.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.species_registry import env_class
from environments.shared.tests.com_stance_helpers import EnvStanceSampler, Morphology
from environments.shared.tests.stance_track_helpers import stance_track_quality

RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/runs")
END = {"s42": 4_718_592, "s80": 5_767_168}
START = {"start_s42": ("s42_A_1101", 3_145_728), "start_s80": ("s80_A_1301", 4_194_304)}
MID = ModelId("tyrannosaurus_rex", "anatomical")


def paths(who: str) -> tuple[Path, Path]:
    if who in START:
        run, steps = START[who]
    else:
        run, steps = who, END[who[:3]]
    m = RUNS / run / "01_stance" / "models"
    return m / f"stage1_{steps}_steps.zip", m / f"stage1_vecnormalize_{steps}_steps.pkl"


class Policy:
    def __init__(self, zip_path: Path, pkl_path: Path):
        self.model = PPO.load(str(zip_path), device="cpu")
        with open(pkl_path, "rb") as fh:
            vn = pickle.load(fh)
        self.mean = np.asarray(vn.obs_rms.mean, dtype=np.float64)
        self.var = np.asarray(vn.obs_rms.var, dtype=np.float64)
        self.eps = float(vn.epsilon)
        self.clip = float(vn.clip_obs)
        assert vn.norm_obs, "expected observation normalisation"

    def norm(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(self.norm(obs), deterministic=True)[0]

    def raw_mean(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(self.norm(obs)[None, :])
            return self.model.policy.get_distribution(t).distribution.mean.numpy()[0].astype(np.float64)


def episode(env, pol: Policy, seed: int, force: float | None, horizon: int) -> dict:
    u = env.unwrapped
    m, d = u.model, u.data
    sampler = EnvStanceSampler(env, Morphology.from_env(env, MID))
    root = sampler.root
    floor_z = sampler.floor_z
    jq = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in ("head_pitch", "neck_pitch", "tail_3_pitch")}
    assert m.actuator(14).name == "tail_3_pitch_act"
    assert int(m.actuator_trnid[14, 0]) == m.joint("tail_3_pitch").id
    ctrl_expect = None
    if force is not None:
        lo, hi = m.actuator_ctrlrange[14]
        home = float(u._home_ctrl[14])
        f = float(np.clip(force, -1, 1))
        ctrl_expect = home + (f * (home - lo) if f < 0 else f * (hi - home))
    obs, _ = env.reset(seed=seed)
    sampler.sample(env)
    rewards, own14, raw14, tail3, fwdz, comz, head, neck, own_all = [], [], [], [], [], [], [], [], []
    bad_ctrl = bad_filter = 0
    max_dctrl = max_dfilt = 0.0
    terminated = truncated = False
    step = 0
    for step in range(1, horizon + 1):
        own = pol.act(obs)
        r14 = pol.raw_mean(obs)[14] if step > 500 and (step % 10 == 0) else float("nan")
        action = np.array(own, copy=True)
        if force is not None:
            action[14] = force
        obs, reward, terminated, truncated, info = env.step(action)
        sampler.sample(env)
        if force is not None:
            bad_ctrl += int(d.ctrl[14] != ctrl_expect)
            bad_filter += int(u._action_filter_state[14] != force)
            max_dctrl = max(max_dctrl, abs(float(d.ctrl[14]) - ctrl_expect))
            max_dfilt = max(max_dfilt, abs(float(u._action_filter_state[14]) - force))
        rewards.append(float(reward))
        own14.append(float(own[14]))
        raw14.append(float(r14))
        own_all.append(np.asarray(own, dtype=np.float64))
        tail3.append(float(d.qpos[jq["tail_3_pitch"]]))
        head.append(float(d.qpos[jq["head_pitch"]]))
        neck.append(float(d.qpos[jq["neck_pitch"]]))
        fwdz.append(float(d.xmat[root].reshape(3, 3)[2, 0]))
        comz.append(float(d.subtree_com[root][2] - floor_z))
        if terminated or truncated:
            break
    tr = sampler.track()
    stood = (not terminated) and step == horizon
    w = slice(500, 1000)
    row = {"seed": seed, "force": force, "steps": step, "stood": stood, "terminated": bool(terminated),
           "return": float(np.sum(rewards)), "return_1000": float(np.sum(rewards[:1000])),
           "window_return_200_1000": float(np.sum(rewards[200:1000])),
           "bad_ctrl_steps": bad_ctrl, "bad_filter_steps": bad_filter,
           "max_abs_ctrl_dev": max_dctrl, "max_abs_filter_dev": max_dfilt, "ctrl_expect": ctrl_expect,
           "own14_mean_5_10": float(np.mean(own14[w])), "raw14_mean_5_10": float(np.nanmean(raw14[w])),
           "own_mean_5_10": np.mean(own_all[w], axis=0).round(6).tolist(),
           "tail3_deg_5_10": math.degrees(np.mean(tail3[w])), "head_deg_5_10": math.degrees(np.mean(head[w])),
           "neck_deg_5_10": math.degrees(np.mean(neck[w])), "fwdz_5_10": float(np.mean(fwdz[w])),
           "pitch_proxy_deg_5_10": math.degrees(math.asin(float(np.mean(fwdz[w])))),
           "comz_mm_5_10": 1000 * float(np.mean(comz[w])),
           "own14_bins_025s": [float(np.mean(own14[i:i + 25])) for i in range(0, 400, 25)],
           "tail3_deg_bins_025s": [math.degrees(float(np.mean(tail3[i:i + 25]))) for i in range(0, 400, 25)],
           "metrics": {}}
    if stood:
        cut = tr
        if horizon > 1000:
            cut = replace(tr, t=tr.t[:1001], com=tr.com[:1001], root_quat=tr.root_quat[:1001],
                          foot_pos=tr.foot_pos[:1001],
                          foot_load=None if tr.foot_load is None else tr.foot_load[:1001],
                          foot_heading=None if tr.foot_heading is None else tr.foot_heading[:1001],
                          support=None if tr.support is None else tr.support[:1001],
                          com_vel=None if tr.com_vel is None else tr.com_vel[:1001])
        row["metrics"]["5.0"] = stance_track_quality(cut, window_start_s=5.0)
        if horizon > 1000:
            for ws in (10.0, 15.0):
                row["metrics"][str(ws)] = stance_track_quality(tr, window_start_s=ws)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--who", required=True)
    ap.add_argument("--force", type=float, default=None)
    ap.add_argument("--seeds", default="3042,3046,3050,3055,3061")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    ctor = dict(load_stage_config(MID, "stance")["env_kwargs"])
    if a.steps != 1000:
        ctor["max_episode_steps"] = a.steps
    with open(a.out, "a") as out:
        for who in a.who.split(","):
            z, p = (Path(a.zip), Path(a.pkl)) if a.zip else paths(who)
            pol = Policy(z, p)
            t0 = time.time()
            for seed in seeds:
                env = env_class(MID)(**ctor)
                row = episode(env, pol, seed, a.force, a.steps)
                env.close()
                row.update({"who": who, "tag": a.tag, "zip": str(z)})
                out.write(json.dumps(row) + "\n")
                out.flush()
            print(f"{who} force={a.force} steps={a.steps}: {len(seeds)} episodes {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
