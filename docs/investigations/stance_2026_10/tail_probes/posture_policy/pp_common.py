"""Shared helpers for the posture_policy role (tests 2 and 3).  Read only; no training.

Imported by run_base.py, sweep.py, run_plant.py (this folder).  PYTHONPATH must hold the policies lens's code copy
(com_revisit/notes/policies/code_copy) and this folder.  Checkpoints are opened read only.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
RUNS = SCR / "r10round/ecs/runs"
HERE = SCR / "com_revisit/notes/posture_policy"
MID = ModelId("tyrannosaurus_rex", "anatomical")
NAMES = (
    [f"s42_A_{n}" for n in range(1101, 1152, 10)]
    + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
    + [f"s80_A_{n}" for n in range(1301, 1352, 10)]
    + [f"s80_B_{n}" for n in range(1401, 1452, 10)]
)
END = {"s42": 4_718_592, "s80": 5_767_168}
START = {"start_s42": ("s42_A_1101", 3_145_728), "start_s80": ("s80_A_1301", 4_194_304)}
ALL_WHO = ["statue", "start_s42", "start_s80"] + NAMES
TAIL3 = 14  # tail_3_pitch_act; checked against the model in check_layout()

# Observation layout (bipedal-target-heading-free/v1, 63 values), from _get_obs and probe/layout.txt, re-checked live
# in probe_obs.py.
JOINTS = ["neck_pitch", "neck_yaw", "head_pitch", "tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch",
          "r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle", "r_toe_d2_joint", "r_toe_d3_joint", "r_toe_d4_joint",
          "l_hip_pitch", "l_hip_roll", "l_knee", "l_ankle", "l_toe_d2_joint", "l_toe_d3_joint", "l_toe_d4_joint"]
OBS_NAMES = ([f"{j}:q" for j in JOINTS] + [f"{j}:v" for j in JOINTS]
             + ["grav_x", "grav_y", "grav_z", "gyro_x", "gyro_y", "gyro_z", "hvel_fwd", "hvel_left", "hvel_up",
                "acc_x", "acc_y", "acc_z", "r_foot_contact", "l_foot_contact", "prey_bx", "prey_by", "prey_bz",
                "prey_dist", "cmd_vx", "cmd_vy", "cmd_yaw"])
assert len(OBS_NAMES) == 63
I_TAIL3 = 6
I_GRAV = [42, 43, 44]
I_GYRO = [45, 46, 47]
I_HVEL = [48, 49, 50]
I_ACC = [51, 52, 53]
I_FOOT = [54, 55]
I_PREY = [56, 57, 58]
I_HEAD = [0, 1, 2]
I_TAIL12 = [3, 4, 5]
I_LEGS = list(range(7, 21))
I_JVEL = list(range(21, 42))
# Transplant groups (test 2).  "posture" = body orientation + head/neck + legs + foot loads; tail_1/tail_2 angles and
# all velocities are kept separate; tail_3's own angle is never transplanted.
GROUPS = {
    "orient(grav+acc+prey)": I_GRAV + I_ACC + I_PREY,
    "grav": I_GRAV,
    "acc": I_ACC,
    "head_neck_q": I_HEAD,
    "legs_q": I_LEGS,
    "foot_contact": I_FOOT,
    "tail12_q": I_TAIL12,
    "rates(gyro+hvel+jvel)": I_GYRO + I_HVEL + I_JVEL,
    "POSTURE(orient+head+legs+feet)": I_GRAV + I_ACC + I_PREY + I_HEAD + I_LEGS + I_FOOT,
    "ALL_but_tail3": [i for i in range(63) if i != I_TAIL3],
}


def stance_ctor() -> dict:
    return dict(load_stage_config(MID, "stance")["env_kwargs"])


def make_env(ctor: dict | None = None):
    return env_class(MID)(**(ctor or stance_ctor()))


def ckpt(who: str) -> tuple[Path, Path]:
    if who in START:
        run, steps = START[who]
    else:
        run, steps = who, END[who[:3]]
    m = RUNS / run / "01_stance" / "models"
    return m / f"stage1_{steps}_steps.zip", m / f"stage1_vecnormalize_{steps}_steps.pkl"


def lineage(who: str) -> str:
    if who == "statue":
        return "statue"
    if who.startswith("start_"):
        return who
    return who[:3]


class Pol:
    """The lens's loader (load_sb3_checkpoint) plus the raw Gaussian mean for batches."""

    def __init__(self, who: str, factory):
        z, p = ckpt(who)
        self.zip, self.pkl = z, p
        self.model, self.normalizer, _ = load_sb3_checkpoint(str(z), str(p), factory)
        rms = self.normalizer.obs_rms
        self.std = np.sqrt(np.asarray(rms.var, dtype=np.float64) + float(self.normalizer.epsilon))
        self.mu = np.asarray(rms.mean, dtype=np.float64)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(self.normalizer.normalize_obs(obs), deterministic=True)[0]

    def raw_mean(self, obs_batch: np.ndarray) -> np.ndarray:
        """Pre-clip Gaussian mean, shape (n, 15), for raw (un-normalised) observations (n, 63)."""
        nb = self.normalizer.normalize_obs(np.asarray(obs_batch, dtype=np.float32))
        with torch.no_grad():
            mean = self.model.policy.get_distribution(torch.as_tensor(np.asarray(nb, dtype=np.float32)))
            return mean.distribution.mean.numpy().astype(np.float64)


def rot_y(alpha: float) -> np.ndarray:
    c, s = math.cos(alpha), math.sin(alpha)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(alpha: float) -> np.ndarray:
    c, s = math.cos(alpha), math.sin(alpha)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def pitch_illusion(obs: np.ndarray, nose_up_deg: float, channels=("grav", "acc", "prey")) -> np.ndarray:
    """The observation as if the pelvis were pitched nose_up_deg further nose-up (about its own y axis), all else equal.

    A body-frame world vector is v_b = R^T v_w.  Pitching the body by alpha about its own y axis (R' = R Ry(alpha))
    gives v_b' = Ry(alpha)^T v_b.  With x forward and z up, a positive rotation about +y is nose-DOWN, so nose-up by
    theta is alpha = -theta.  Applied to projected gravity, the accelerometer (the imu site has the pelvis's frame;
    at rest it reads -9.81 g) and the prey direction in pelvis coordinates.  Heading velocity is in the
    gravity-aligned heading frame (independent of pitch) and is left alone, as are the gyro, joints and contacts.
    Checked against the env's own heading_free_root_segments on a rotated quaternion in probe_obs.py.
    """
    out = np.array(obs, dtype=np.float64, copy=True)
    Rt = rot_y(-math.radians(nose_up_deg)).T
    idx = {"grav": I_GRAV, "acc": I_ACC, "prey": I_PREY}
    for ch in channels:
        out[..., idx[ch]] = out[..., idx[ch]] @ Rt.T
    return out.astype(np.float32)


def roll_illusion(obs: np.ndarray, right_up_deg: float, channels=("grav", "acc", "prey")) -> np.ndarray:
    """As pitch_illusion, for a rotation about the pelvis x axis by right_up_deg (positive = +x rotation)."""
    out = np.array(obs, dtype=np.float64, copy=True)
    Rt = rot_x(math.radians(right_up_deg)).T
    idx = {"grav": I_GRAV, "acc": I_ACC, "prey": I_PREY}
    for ch in channels:
        out[..., idx[ch]] = out[..., idx[ch]] @ Rt.T
    return out.astype(np.float32)


def pitch_deg_from_obs(obs: np.ndarray) -> np.ndarray:
    """Pelvis pitch (deg, + nose up) as the observation's projected gravity encodes it: g_x = -forward_z."""
    return np.degrees(np.arcsin(np.clip(-np.asarray(obs, dtype=np.float64)[..., 42], -1, 1)))
