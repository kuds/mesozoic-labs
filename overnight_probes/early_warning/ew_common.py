"""EARLY-WARNING CURVE (role early_warning): shared definitions.  Read only on every input; nothing here trains.

One fixed measurement, applied to every checkpoint:
  - load: the harness loader load_sb3_checkpoint (sealed code copy r10round/ecs/code), CPU; its normaliser frozen.
  - tail_3 command = entry 14 (tail_3_pitch_act) of the policy's raw Gaussian mean (pre-clip) unless "own"
    (the clipped deterministic command the env received).
  - observation index 6 = tail_3_pitch joint angle (rad); 42-44 gravity, 51-53 accelerometer, 56-58 prey
    direction in pelvis coordinates (the three pitch-carrying channels; posture_policy/pp_common.py).
  - PRIMARY tail sweep: obs[6] set to 0, 0.25, 0.5, 0.75, 1.0 deg, everything else fixed.  Slope = least squares
    per state, median over states.  G = slope x PLANT (5.16 deg of tail_3 per unit command, the r10 plant measured
    in the policies' own postures, posture_policy test 3).  c0 = cmd14 at obs[6] = 0 (direct), median over states.
    Fixed point FP = c0 / (1 - G) (inf when G >= 1).  The 0-1 deg range keeps the tail_3 input's z at or below +3.4
    under every checkpoint's normaliser measured (ew_stats.txt; clip_obs 10 is reached at 2.97 deg at the earliest
    s60 / s80 statistics), so no swept point is clipped; linearity is checked by the slope over 0-0.5 against
    0.5-1.0 deg.
  - LEGACY tail sweep: 0..6 deg by 1 (verify_loop.py / posture_policy / VPOST definition), reported for
    comparison only; it is clipped (z = 10) beyond 2.97-3.78 deg at every pre-rise checkpoint and in all of s80.
  - PITCH: pitch illusion (rotate gravity, accelerometer, prey direction about the pelvis y axis) by -0.5, -0.25,
    0, +0.25, +0.5 deg nose-up; least-squares slope of cmd14 per deg nose-up, median over states.
  - Two state sets: REF = the 10 fixed raw observations of the policy probe's bank (bank_raw_obs, byte-identical
    in all 27 runs: sha1 767bab20e8; statue-based states, seeds 3042/3043), and OWN = the checkpoint's own
    settled states: deterministic rollouts, seeds 5001-5010, 1,000 steps (10 s), raw observations after steps
    500, 600, ..., 1000 (60 states).
"""
from __future__ import annotations

import glob
import hashlib
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import torch

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
HERE = SCR / "com_revisit/notes/early_warning"
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
ECS = SCR / "r10round/ecs/runs"
PLANT = 5.16
I_T3 = 6
A_T3 = 14
I_GRAV, I_ACC, I_PREY = [42, 43, 44], [51, 52, 53], [56, 57, 58]
TAIL_PRIMARY = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
TAIL_LEGACY = np.arange(0.0, 6.01, 1.0)
PITCH_DEG = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
SEEDS = list(range(5001, 5011))
HORIZON = 1000
SNAP_STEPS = (500, 600, 700, 800, 900, 1000)
R10_RUNS = ("s42", "s60", "s80")
ECS_RUNS = sorted(p.name for p in ECS.iterdir() if p.is_dir())


def r10_dir(run: str) -> Path:
    return F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance"


def ecs_dir(run: str) -> Path:
    return ECS / run / "01_stance"


def checkpoints(run: str) -> list[tuple[int, Path, Path]]:
    """(steps, zip, pkl) for every numbered checkpoint of a run, ascending."""
    if run in R10_RUNS:
        m = r10_dir(run) / "models"
        zs = glob.glob(str(m / "stage1_*_steps.zip.copy1"))
        out = []
        for z in zs:
            n = int(re.search(r"stage1_(\d+)_steps", z).group(1))
            out.append((n, Path(z), m / f"stage1_vecnormalize_{n}_steps.pkl.copy1"))
        # the run's final save (num_timesteps 12,001,280 in all three r10 runs; distinct weights from 11,534,336)
        out.append((12_001_280, m / "stage1_final.zip.copy1", m / "stage1_final_vecnorm.pkl.copy1"))
    else:
        m = ecs_dir(run) / "models"
        out = []
        for z in glob.glob(str(m / "stage1_*_steps.zip")):
            n = int(re.search(r"stage1_(\d+)_steps", z).group(1))
            out.append((n, Path(z), m / f"stage1_vecnormalize_{n}_steps.pkl"))
    out.sort()
    for n, z, p in out:
        assert z.exists() and p.exists(), (z, p)
    return out


def ckpt_key(z: Path, p: Path) -> str:
    """Identity of a checkpoint: sha1 of the actor/critic weights (policy.pth) and of the normaliser pickle."""
    with zipfile.ZipFile(z) as zf:
        h1 = hashlib.sha1(zf.read("policy.pth")).hexdigest()
    h2 = hashlib.sha1(p.read_bytes()).hexdigest()
    return h1[:16] + ":" + h2[:16]


def bank() -> tuple[np.ndarray, np.ndarray]:
    with np.load(r10_dir("s42") / "policy_probe.npz.copy1", allow_pickle=False) as d:
        return np.asarray(d["bank_raw_obs"], dtype=np.float64), np.asarray(d["bank_labels"])


def rot_y(alpha: float) -> np.ndarray:
    c, s = math.cos(alpha), math.sin(alpha)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def pitch_illusion(obs: np.ndarray, nose_up_deg: float) -> np.ndarray:
    """posture_policy/pp_common.pitch_illusion (all three channels): the observation as if the pelvis were pitched
    nose_up_deg further nose-up about its own y axis."""
    out = np.array(obs, dtype=np.float64, copy=True)
    Rt = rot_y(-math.radians(nose_up_deg)).T
    for idx in (I_GRAV, I_ACC, I_PREY):
        out[..., idx] = out[..., idx] @ Rt.T
    return out


def pitch_up_deg(q: np.ndarray) -> float:
    w, x, y, z = q / np.linalg.norm(q)
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (x * z - w * y)))))


def slope_per_state(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xc = xs - xs.mean()
    return (xc[:, None] * (ys - ys.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()


class Net:
    def __init__(self, z: Path, p: Path, factory):
        from environments.shared.policy_loading import load_sb3_checkpoint

        self.model, self.norm, _ = load_sb3_checkpoint(str(z), str(p), factory)
        self.p = self.model.policy
        rms = self.norm.obs_rms
        self.mean = np.asarray(rms.mean, dtype=np.float64)
        self.std = np.sqrt(np.asarray(rms.var, dtype=np.float64) + float(self.norm.epsilon))
        self.clip = float(self.norm.clip_obs)

    def z(self, obs: np.ndarray) -> np.ndarray:
        return (np.asarray(obs, dtype=np.float64) - self.mean) / self.std

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]

    def mu(self, obs_batch: np.ndarray) -> np.ndarray:
        nb = self.norm.normalize_obs(np.asarray(obs_batch, dtype=np.float32))
        with torch.no_grad():
            d = self.p.get_distribution(torch.as_tensor(np.asarray(nb, dtype=np.float32)))
            return d.distribution.mean.numpy().astype(np.float64)


def tail_sweep(net: Net, X: np.ndarray, degs: np.ndarray) -> dict:
    ys = []
    zmax = []
    for d in degs:
        Y = np.array(X, dtype=np.float64, copy=True)
        Y[:, I_T3] = math.radians(d)
        ys.append(net.mu(Y)[:, A_T3])
        zmax.append(float(np.max(net.z(Y)[:, I_T3])))
    ys = np.array(ys)  # (k, n)
    sl = slope_per_state(degs, ys)
    out = {"slope_per_deg": float(np.median(sl)), "slope_q25": float(np.quantile(sl, 0.25)),
           "slope_q75": float(np.quantile(sl, 0.75)), "c0": float(np.median(ys[0])),
           "c0_q25": float(np.quantile(ys[0], 0.25)), "c0_q75": float(np.quantile(ys[0], 0.75)),
           "z_tail_at_max": max(zmax), "n_points_clipped": int(sum(z >= net.clip for z in zmax)),
           "curve_median": np.median(ys, axis=1).round(5).tolist()}
    G = out["slope_per_deg"] * PLANT
    out["G"] = G
    out["FP"] = out["c0"] / (1.0 - G) if G < 1.0 else float("inf")
    # per-state fixed points, median (robust to state mix)
    Gs = sl * PLANT
    fps = np.where(Gs < 1.0, ys[0] / np.maximum(1.0 - Gs, 1e-9), np.inf)
    out["FP_state_median"] = float(np.median(fps))
    if len(degs) == 5:  # primary: linearity check, halves 0-0.5 and 0.5-1.0 deg
        lo = np.median((ys[2] - ys[0]) / (degs[2] - degs[0]))
        hi = np.median((ys[4] - ys[2]) / (degs[4] - degs[2]))
        out["slope_lo_half"], out["slope_hi_half"] = float(lo), float(hi)
    return out


def pitch_sweep(net: Net, X: np.ndarray) -> dict:
    ys, zs = [], []
    for d in PITCH_DEG:
        Y = pitch_illusion(X, d)
        ys.append(net.mu(Y)[:, A_T3])
        zs.append(float(np.max(np.abs(net.z(Y)[:, I_GRAV + I_ACC + I_PREY]))))
    ys = np.array(ys)
    sl = slope_per_state(PITCH_DEG, ys)
    return {"pitch_slope": float(np.median(sl)), "pitch_q25": float(np.quantile(sl, 0.25)),
            "pitch_q75": float(np.quantile(sl, 0.75)), "z_pitch_absmax": max(zs),
            "n_points_clipped": int(sum(z >= net.clip for z in zs))}


def measure_states(net: Net, X: np.ndarray) -> dict:
    return {"n": int(len(X)), "mu14_at_states": float(np.median(net.mu(X)[:, A_T3])),
            "tail3_deg_at_states": float(np.degrees(np.median(X[:, I_T3]))),
            "primary": tail_sweep(net, X, TAIL_PRIMARY), "legacy": tail_sweep(net, X, TAIL_LEGACY),
            "pitch": pitch_sweep(net, X)}
