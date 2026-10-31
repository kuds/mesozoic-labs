"""MARGIN-TO-RAIL TEST (com_revisit, role warning_margin): shared definitions.  No training; checkpoints read only.

Closed loop, deterministic: env = env_class(mid)(**load_stage_config(mid, "stance")["env_kwargs"]) (the stance
config's own constructor, as early_warning/ew_measure.py and clamp/run_clamp.py), env.reset(seed=s), 1,000 steps,
action = model.predict(normalizer.normalize_obs(obs_seen), deterministic=True)[0] every step.

Two constant nudges, applied separately, every step from step 1:
  BIAS (a): the clamp test's override path (clamp/run_clamp.py --force; D-9's episode_trace holds): the policy's own
            action, with entry 14 (tail_3_pitch_act) replaced by own[14] + b before env.step.  The env clips it to
            [-1, 1], low-passes it and maps it; nothing else changes.
  OBS  (b): obs_seen = the true observation with entry 6 (tail_3_pitch joint angle, rad) + radians(delta) before the
            normaliser (which clips at z = +-10); the env and the applied action are the policy's own.
"command" = the pre-filter tail_3 command the env receives, clipped to [-1, 1]: clip(own14 + b) for BIAS, own14 for
OBS.  Window = steps 501-1000.  RAILED step: command >= 0.99.  PARKED episode: more than 250 railed steps in the
window (i.e. "command 1.0 for most of the window"; a terminated episode counts its missing steps as not railed).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
HERE = SCR / "com_revisit/notes/warning_margin"
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
I_T3 = 6
A_T3 = 14
HORIZON = 1000
WIN = (500, 1000)  # 0-based step indices 500..999 = steps 501..1000
RAIL = 0.99
PARK_COUNT = 250  # parked iff railed steps in the window > 250 (of 500)
OVERBAR_FRAC = 0.10  # secondary: the over-bar analogue (frac99 > 0.10)

CKPTS = [("s80", 10_485_760), ("s80", 11_534_336), ("s80", 12_001_280),
         ("s60", 6_291_456), ("s60", 7_340_032), ("s42", 2_097_152), ("s42", 3_145_728)]

torch.set_num_threads(1)


def ckpt_paths(run: str, steps: int) -> tuple[Path, Path]:
    m = F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance" / "models"
    if steps == 12_001_280:
        z, p = m / "stage1_final.zip.copy1", m / "stage1_final_vecnorm.pkl.copy1"
    else:
        z, p = m / f"stage1_{steps}_steps.zip.copy1", m / f"stage1_vecnormalize_{steps}_steps.pkl.copy1"
    assert z.exists() and p.exists(), (z, p)
    return z, p


def make_factory():
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    return (lambda: env_class(mid)(**ctor)), ctor


class Policy:
    def __init__(self, run: str, steps: int, factory):
        from environments.shared.policy_loading import load_sb3_checkpoint

        z, p = ckpt_paths(run, steps)
        self.model, self.norm, _ = load_sb3_checkpoint(str(z), str(p), factory)
        rms = self.norm.obs_rms
        self.mean = np.asarray(rms.mean, dtype=np.float64)
        self.std = np.sqrt(np.asarray(rms.var, dtype=np.float64) + float(self.norm.epsilon))
        self.clip = float(self.norm.clip_obs)
        # the observed tail_3 angle (deg) at which the normaliser clip (z = +clip) is reached
        self.t3_clip_deg = math.degrees(self.mean[I_T3] + self.clip * self.std[I_T3])

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]


def episode(env, pol: Policy, seed: int, kind: str, value: float) -> dict:
    """One deterministic closed-loop episode with a constant nudge.  kind in {"none", "bias", "obs"}."""
    u = env.unwrapped
    t0 = time.process_time()
    obs, _ = env.reset(seed=seed)
    off = math.radians(value) if kind == "obs" else 0.0
    cmd, own, t3, pit = [], [], [], []
    term = trunc = False
    k = 0
    for k in range(1, HORIZON + 1):
        if kind == "obs":
            seen = np.array(obs, copy=True)
            seen[I_T3] = seen[I_T3] + off
            a = pol.act(seen)
        else:
            a = pol.act(obs)
        o14 = float(a[A_T3])
        if kind == "bias":
            a = np.array(a, copy=True)
            a[A_T3] = o14 + value
        own.append(o14)
        cmd.append(float(np.clip(a[A_T3], -1.0, 1.0)))
        obs, _, term, trunc, _ = env.step(a)
        t3.append(float(obs[I_T3]))
        q = np.array(u.data.qpos[3:7], dtype=np.float64)
        w, x, y, z = q / np.linalg.norm(q)
        pit.append(math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (x * z - w * y))))))
        if term or trunc:
            break
    cmd_a, own_a = np.array(cmd), np.array(own)
    lo, hi = WIN
    w_cmd, w_own = cmd_a[lo:hi], own_a[lo:hi]
    n_rail = int(np.sum(w_cmd >= RAIL))
    return {"seed": seed, "kind": kind, "value": float(value), "steps": k, "terminated": bool(term),
            "n_rail_win": n_rail, "frac99_win": n_rail / (hi - lo), "parked": bool(n_rail > PARK_COUNT),
            "cmd_win": float(np.mean(w_cmd)) if len(w_cmd) else None,
            "own_win": float(np.mean(w_own)) if len(w_own) else None,
            "tail3_deg_win": math.degrees(float(np.mean(t3[lo:hi]))) if len(t3) > lo else None,
            "pitch_deg_win": float(np.mean(pit[lo:hi])) if len(pit) > lo else None,
            "cmd_bins50": [round(float(np.mean(cmd_a[i:i + 50])), 4) for i in range(0, len(cmd_a), 50)],
            "cpu_s": round(time.process_time() - t0, 3)}
