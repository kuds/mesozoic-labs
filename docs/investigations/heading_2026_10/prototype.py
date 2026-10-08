"""Research-only heading-free observation candidates; not a training contract.

The production environments and their hashes are deliberately not changed.
Use only on the anatomical T. rex and Compsognathus stance tasks in this study.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np


def rotation(quaternion: np.ndarray) -> np.ndarray:
    """Body-to-world matrix, invariant to the sign of a wxyz quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("expected a finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_observation(obs: np.ndarray, joint_width: int, *, candidate: str = "gravity") -> np.ndarray:
    """Replace world quaternion/velocity/target; leave local and scalar channels.

    gravity: three-component unit down vector in the pelvis frame.
    quat: four-component yaw-removed quaternion, canonical sign (comparison).
    Linear velocity uses the gravity-aligned heading frame; target uses the
    full pelvis frame. Command channels are already heading-relative.
    """
    if candidate not in {"gravity", "quat"}:
        raise ValueError(candidate)
    old = np.asarray(obs, dtype=np.float64)
    if old.shape != (joint_width + 22,):
        raise ValueError("expected the current two-foot bipedal-target/v1 layout")
    q = old[joint_width : joint_width + 4]
    q = q / np.linalg.norm(q)
    matrix = rotation(q)
    # Defined over healthy stances. A forward axis exactly parallel to gravity
    # has no yaw frame; fail explicitly rather than select a world direction.
    if np.hypot(matrix[0, 0], matrix[1, 0]) < 1e-8:
        raise ValueError("heading frame is undefined for a vertical forward axis")
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    world_to_heading = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    if candidate == "gravity":
        orientation = -matrix[2, :]
    else:
        # Left multiply q by the inverse world-z yaw quaternion.
        ch, sh = math.cos(yaw / 2), math.sin(yaw / 2)
        w, x, y, z = q
        orientation = np.array([ch * w + sh * z, ch * x + sh * y, ch * y - sh * x, ch * z - sh * w])
        if orientation[0] < 0:
            orientation *= -1
    return np.concatenate(
        [
            old[:joint_width],
            orientation,
            old[joint_width + 4 : joint_width + 7],  # local gyro
            world_to_heading @ old[joint_width + 7 : joint_width + 10],
            old[joint_width + 10 : joint_width + 15],  # local accel, two contact forces
            matrix.T @ old[joint_width + 15 : joint_width + 18],
            old[joint_width + 18 :],  # distance, three command inputs
        ]
    ).astype(np.float32)


class HeadingObservation(gym.ObservationWrapper):
    """Short-study wrapper. Its checkpoints must never be used as repository trunks."""

    def __init__(self, env: gym.Env, candidate: str = "gravity") -> None:
        super().__init__(env)
        self.joint_width = (env.unwrapped.model.nq - 7) + (env.unwrapped.model.nv - 6)
        self.candidate = candidate
        width = env.observation_space.shape[0] - (candidate == "gravity")
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return transform_observation(observation, self.joint_width, candidate=self.candidate)
