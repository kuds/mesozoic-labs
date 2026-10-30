"""Shared pieces of the normaliser probe (com_revisit/notes/probe).  Simulation and network evaluation only; no training.

Loading is the harness's (policies/run_policies.py, run_clamp.py): load_sb3_checkpoint(zip, pkl, factory) gives the
model and a frozen VecNormalize; the action is model.predict(normalizer.normalize_obs(obs), deterministic=True)[0].
A statistics swap replaces normalizer.obs_rms.mean / .var (float64 copies) with another pickle's values, whole or
for chosen observation dimensions only; eps (1e-8) and clip_obs (10) are the same in every pickle (stats.txt).

The rollout is run_clamp.py's roll loop reduced to what this probe keeps: a fresh env per seed from the stance
config's constructor, env.reset(seed), the policy's own deterministic action each step, no override.  The loop-gain
sweep is the verifier's (verify_clamp/verify_loop.py): at steps 500, 600, ..., 1000 the observation is copied, only
its tail_3_pitch angle entry (obs index 6) is set to theta = 0..6 deg, and the slope of the policy's raw Gaussian mean
(pre-clip) for action 14 against theta is fitted per state; the plant's gain, 5.183 deg of tail_3 per unit command
(verify_clamp/loop_summary.txt, the statue's equilibria), turns the slope into the loop gain.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import torch

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class

RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/runs")
MID = ModelId("tyrannosaurus_rex", "anatomical")
NAMES = (
    [f"s42_A_{n}" for n in range(1101, 1152, 10)]
    + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
    + [f"s80_A_{n}" for n in range(1301, 1352, 10)]
    + [f"s80_B_{n}" for n in range(1401, 1452, 10)]
)
SE = {"s42": (3_145_728, 4_718_592), "s80": (4_194_304, 5_767_168)}
START_RUN = {"s42": "s42_A_1101", "s80": "s80_A_1301"}  # the start pair is byte-identical in all 12 runs of a lineage
STEP = 262_144
T3Q = 6
TAIL_DIMS = (3, 4, 5, 6, 24, 25, 26, 27)  # tail_1_pitch, tail_1_yaw, tail_2_pitch, tail_3_pitch: angle, velocity
PLANT_GAIN_DEG = 5.183  # deg of tail_3 per unit tail_3 command (verify_clamp/loop_summary.txt)
THETAS_DEG = np.arange(0.0, 6.01, 1.0)
SWEEP_STEPS = (500, 600, 700, 800, 900, 1000)


def steps_of(lin: str) -> list[int]:
    s, e = SE[lin]
    return list(range(s, e + 1, STEP))


def ckpt(run: str, step: int) -> tuple[Path, Path]:
    m = RUNS / run / "01_stance" / "models"
    return m / f"stage1_{step}_steps.zip", m / f"stage1_vecnormalize_{step}_steps.pkl"


def label(run: str, step: int) -> str:
    lin = run[:3]
    return f"start_{lin}" if step == SE[lin][0] else f"{run}@{step}"


def ctor() -> dict:
    return dict(load_stage_config(MID, "stance")["env_kwargs"])


def stats(run: str, step: int) -> tuple[np.ndarray, np.ndarray]:
    with open(ckpt(run, step)[1], "rb") as fh:
        vn = pickle.load(fh)
    return np.array(vn.obs_rms.mean, dtype=np.float64), np.array(vn.obs_rms.var, dtype=np.float64)


class Agent:
    """The harness's model + frozen normaliser, with swappable observation statistics."""

    def __init__(self, run: str, step: int, factory):
        z, p = ckpt(run, step)
        self.model, self.normalizer, _ = load_sb3_checkpoint(str(z), str(p), factory)
        try:
            self.normalizer.venv.close()
        except Exception:  # noqa: BLE001
            pass
        assert not self.normalizer.training and self.normalizer.norm_obs
        self.own_mean = np.array(self.normalizer.obs_rms.mean, dtype=np.float64)
        self.own_var = np.array(self.normalizer.obs_rms.var, dtype=np.float64)
        self.stats_tag = "own"

    def set_stats(self, mean: np.ndarray | None = None, var: np.ndarray | None = None, dims=None, tag: str = ""):
        """None -> own statistics.  dims=None -> all 63 from (mean, var); else only those dims, the rest own."""
        m, v = self.own_mean.copy(), self.own_var.copy()
        if mean is not None:
            if dims is None:
                m, v = mean.copy(), var.copy()
            else:
                d = list(dims)
                m[d], v[d] = mean[d], var[d]
        self.normalizer.obs_rms.mean = m
        self.normalizer.obs_rms.var = v
        self.stats_tag = tag or ("own" if mean is None else "swap")

    def norm(self, obs: np.ndarray) -> np.ndarray:
        return self.normalizer.normalize_obs(obs)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(self.norm(obs), deterministic=True)[0]

    def raw_mean(self, obs_batch: np.ndarray) -> np.ndarray:
        nb = np.stack([self.norm(o) for o in obs_batch]).astype(np.float32)
        with torch.no_grad():
            return self.model.policy.get_distribution(torch.as_tensor(nb)).distribution.mean.numpy().astype(np.float64)


def sweep(agent: Agent, obs: np.ndarray, idx: int = T3Q) -> dict:
    """Verifier's observation-only sweep of the tail_3 angle on one state."""
    batch = np.repeat(np.asarray(obs, dtype=np.float32)[None, :], len(THETAS_DEG), axis=0)
    batch[:, idx] = np.radians(THETAS_DEG).astype(np.float32)
    mean = agent.raw_mean(batch)
    out = {}
    for j, name in ((14, "tail3"), (2, "head_pitch")):
        slope = float(np.polyfit(THETAS_DEG, mean[:, j], 1)[0])
        out[name] = {"slope_per_deg": slope, "at0": float(mean[0, j]), "at5": float(mean[5, j])}
    return out


def sweep_states(agent: Agent, states: np.ndarray) -> dict:
    sw = [sweep(agent, s) for s in states]
    res = {}
    for name in ("tail3", "head_pitch"):
        res[name] = {k: float(np.median([x[name][k] for x in sw])) for k in ("slope_per_deg", "at0", "at5")}
    res["loop_gain"] = res["tail3"]["slope_per_deg"] * PLANT_GAIN_DEG
    return res


def episode(env, agent: Agent, seed: int, horizon: int = 1000) -> tuple[dict, np.ndarray]:
    """One deterministic unforced episode; returns the summary row and the observations at SWEEP_STEPS."""
    u = env.unwrapped
    m, d = u.model, u.data
    q3 = int(m.jnt_qposadr[m.joint("tail_3_pitch").id])
    obs, _ = env.reset(seed=seed)
    rewards, own14, tail3, raw14 = [], [], [], []
    terms: dict[str, float] = {}
    states = []
    terminated = truncated = False
    step = 0
    for step in range(1, horizon + 1):
        a = agent.act(obs)
        own14.append(float(a[14]))
        obs, r, terminated, truncated, info = env.step(a)
        rewards.append(float(r))
        tail3.append(float(d.qpos[q3]))
        if step in SWEEP_STEPS:  # the observation returned by env.step at these steps, as verify_loop.py collects it
            states.append(np.array(obs, copy=True))
        for k, v in info.items():
            if k.startswith("reward_") and isinstance(v, (int, float, np.number)):
                terms[k] = terms.get(k, 0.0) + float(v)
        if terminated or truncated:
            break
    w = slice(500, 1000)
    row = {"seed": seed, "steps": step, "terminated": bool(terminated), "return": float(np.sum(rewards)),
           "window_return_200_1000": float(np.sum(rewards[200:1000])),
           "own14_mean_5_10": float(np.mean(own14[w])) if step >= 600 else float("nan"),
           "own14_sat_5_10": float(np.mean(np.asarray(own14[w]) >= 0.99)) if step >= 600 else float("nan"),
           "own14_mean_0_2": float(np.mean(own14[:200])),
           "tail3_deg_5_10": math.degrees(float(np.mean(tail3[w]))) if step >= 600 else float("nan"),
           "terms": {k: round(v, 6) for k, v in sorted(terms.items())}}
    return row, np.asarray(states, dtype=np.float32)


def make_factory():
    kw = ctor()
    return lambda: env_class(MID)(**kw)  # noqa: E731
