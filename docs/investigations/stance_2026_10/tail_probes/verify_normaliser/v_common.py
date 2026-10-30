"""Verifier's own policy evaluator (independent of the probe's probe_common.py and of the harness loader).

Network: the actor is read straight from the checkpoint zip's policy.pth (state dict, weights_only) and evaluated as
63 -> Linear 512 -> tanh -> Linear 256 -> tanh -> Linear 15 (SB3 MlpPolicy default activation, net_arch [512, 256]),
float32 via torch.nn.functional.linear so the arithmetic is the same op as SB3's nn.Linear.  Deterministic action =
clip(mean, -1, 1) (Box action space [-1, 1], squash_output None).
Normaliser: z = clip((obs - mean) / sqrt(var + eps), -clip, clip) cast to float32, with (mean, var, eps, clip) read by
v_stats.Reader (own restricted unpickler).  No SB3 object is built.
Rollout: fresh env from the stance config's env_kwargs per seed, env.reset(seed), the own deterministic action each
step, 1,000 steps, no override.  The loop gain is the verifier's definition re-implemented: at the observations
returned by env.step at steps 500, 600, ..., 1000, set obs[6] to theta = 0..6 deg, fit the slope of the raw mean of
action 14 per deg per state, median over states, x 5.183 deg per unit command.  No training.
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v_stats import RUNS, Reader

PLANT = 5.183
THETAS = np.arange(0.0, 6.01, 1.0)
SWEEP_AT = (500, 600, 700, 800, 900, 1000)
SE = {"s42": (3_145_728, 4_718_592), "s80": (4_194_304, 5_767_168)}
START = {"s42": "s42_A_1101", "s80": "s80_A_1301"}
TAIL8 = (3, 4, 5, 6, 24, 25, 26, 27)


def paths(run, step):
    m = RUNS / run / "01_stance" / "models"
    return m / f"stage1_{step}_steps.zip", m / f"stage1_vecnormalize_{step}_steps.pkl"


class Net:
    def __init__(self, run, step):
        z = zipfile.ZipFile(paths(run, step)[0])
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        self.L = [(sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"]),
                  (sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"]),
                  (sd["action_net.weight"], sd["action_net.bias"])]
        self.tag = f"{run}@{step}"

    def graft(self, other: "Net", layer: int) -> "Net":
        n = object.__new__(Net)
        n.L = list(self.L)
        n.L[layer] = other.L[layer]
        n.tag = f"{self.tag}[L{layer}<-{other.tag}]"
        return n

    def mean(self, z: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(z, dtype=np.float32))
        with torch.no_grad():
            h = torch.tanh(F.linear(x, *self.L[0]))
            h = torch.tanh(F.linear(h, *self.L[1]))
            return F.linear(h, *self.L[2]).numpy().astype(np.float64)


class Norm:
    def __init__(self, run, step):
        vn = Reader(io.BytesIO(paths(run, step)[1].read_bytes())).load()
        self.mean = np.asarray(vn.obs_rms.mean, dtype=np.float64).copy()
        self.var = np.asarray(vn.obs_rms.var, dtype=np.float64).copy()
        self.eps, self.clip = float(vn.epsilon), float(vn.clip_obs)
        self.tag = f"{run}@{step}"

    def mixed(self, donor: "Norm", dims=None, part="both") -> "Norm":
        n = object.__new__(Norm)
        n.eps, n.clip = self.eps, self.clip
        n.mean, n.var = self.mean.copy(), self.var.copy()
        d = list(range(63)) if dims is None else list(dims)
        if part in ("both", "mean"):
            n.mean[d] = donor.mean[d]
        if part in ("both", "var"):
            n.var[d] = donor.var[d]
        n.tag = f"{self.tag}[{part}{'' if dims is None else list(dims)}<-{donor.tag}]"
        return n

    def __call__(self, obs):
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)

    def scale_deg(self, i=6):
        return math.radians(1) / math.sqrt(self.var[i] + self.eps)


def sweep_one(net: Net, norm: Norm, obs: np.ndarray, thetas=THETAS, j=14):
    b = np.repeat(np.asarray(obs, dtype=np.float32)[None, :], len(thetas), axis=0)
    b[:, 6] = np.radians(thetas).astype(np.float32)
    mu = net.mean(np.stack([norm(o) for o in b]))
    return float(np.polyfit(thetas, mu[:, j], 1)[0]), mu[:, j]


def gain(net, norm, states, thetas=THETAS):
    return PLANT * float(np.median([sweep_one(net, norm, s, thetas)[0] for s in states]))


def make_env():
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class
    mid = ModelId("tyrannosaurus_rex", "anatomical")
    kw = dict(load_stage_config(mid, "stance")["env_kwargs"])
    return env_class(mid)(**kw)


def rollout(net: Net, norm: Norm, seed: int, horizon=1000, record_z=False):
    env = make_env()
    u = env.unwrapped
    m, d = u.model, u.data
    q3 = int(m.jnt_qposadr[m.joint("tail_3_pitch").id])
    obs, _ = env.reset(seed=seed)
    ret, own14, t3, states, zlog = 0.0, [], [], [], []
    zmax = 0.0
    term = trunc = False
    k = 0
    for k in range(1, horizon + 1):
        z = norm(obs)
        zmax = max(zmax, float(np.max(np.abs(z))))
        if record_z:
            zlog.append((float(obs[6]), float(z[6])))
        a = np.clip(net.mean(z[None, :])[0], -1.0, 1.0).astype(np.float32)
        own14.append(float(a[14]))
        obs, r, term, trunc, info = env.step(a)
        ret += float(r)
        t3.append(float(d.qpos[q3]))
        if k in SWEEP_AT:
            states.append(np.array(obs, copy=True))
        if term or trunc:
            break
    env.close()
    w = slice(500, 1000)
    row = {"seed": seed, "steps": k, "terminated": bool(term), "return": ret,
           "own14_5_10": float(np.mean(own14[w])), "rail_5_10": float(np.mean(np.asarray(own14[w]) >= 0.99)),
           "tail3_deg_5_10": math.degrees(float(np.mean(t3[w]))), "zmax": zmax}
    return row, np.asarray(states, dtype=np.float32), zlog
