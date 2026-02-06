"""
Brachiosaurus Gymnasium Environment

A quadrupedal sauropod locomotion environment with food-reaching behavior.

Observation space:
    - Joint positions (qpos) excluding root freejoint
    - Joint velocities (qvel) excluding root freejoint
    - Torso orientation (quaternion)
    - Torso angular velocity
    - Torso linear velocity
    - Foot contact states (4 feet)
    - Food relative position
    - Food distance

Action space:
    - Continuous control for all actuators [-1, 1] normalized
    - 22 actuators: 6 neck + 16 legs (4 per leg)

Reward components:
    - Forward velocity toward food
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Gait stability (encourage coordinated quadrupedal gait)
    - Food reach bonus (when head gets close to food)
    - Approach shaping (distance to food)
"""

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from typing import Optional, Dict, Any, Tuple

# Make shared module importable
_env_root = str(Path(__file__).resolve().parent.parent.parent)
if _env_root not in sys.path:
    sys.path.insert(0, _env_root)

from shared.base_env import BaseDinoEnv


class BrachioEnv(BaseDinoEnv):
    """Brachiosaurus quadrupedal locomotion and food-reaching environment."""

    _camera_distance = 8.0
    _camera_azimuth = 135
    _camera_elevation = -20

    def __init__(
        self,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        # Reward weights
        forward_vel_weight: float = 1.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        gait_stability_weight: float = 0.05,
        food_reach_bonus: float = 500.0,
        food_reach_threshold: float = 0.5,
        food_approach_weight: float = 0.3,
        # Environment settings
        food_distance_range: Tuple[float, float] = (3.0, 8.0),
        food_lateral_range: Tuple[float, float] = (-2.0, 2.0),
        food_height_range: Tuple[float, float] = (2.0, 4.0),
        healthy_z_range: Tuple[float, float] = (1.0, 3.5),
    ):
        model_path = str(
            Path(__file__).parent.parent / "assets" / "brachiosaurus.xml"
        )

        # Brachio-specific reward weights
        self.gait_stability_weight = gait_stability_weight
        self.food_reach_bonus = food_reach_bonus
        self.food_reach_threshold = food_reach_threshold
        self.food_approach_weight = food_approach_weight

        # Brachio-specific env settings
        self.food_distance_range = food_distance_range
        self.food_lateral_range = food_lateral_range
        self.food_height_range = food_height_range

        super().__init__(
            model_path=model_path,
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            forward_vel_weight=forward_vel_weight,
            alive_bonus=alive_bonus,
            energy_penalty_weight=energy_penalty_weight,
            fall_penalty=fall_penalty,
            healthy_z_range=healthy_z_range,
        )

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""
        # Body IDs
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # Geom IDs for contact detection
        self.food_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "food_geom"
        )
        self.torso_main_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_main"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self.head_tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tip"
        )
        self.tail_tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip"
        )

        # Foot site IDs
        self.foot_site_ids = {
            "fr": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "fr_foot_contact"
            ),
            "fl": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot_contact"
            ),
            "rr": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "rr_foot_contact"
            ),
            "rl": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "rl_foot_contact"
            ),
        }

        # Sensor indices (order matches MJCF definition)
        self._sensor_gyro_start = 0
        self._sensor_accel_start = 3
        self._sensor_quat_start = 6
        self._sensor_fr_foot = 10
        self._sensor_fl_foot = 11
        self._sensor_rr_foot = 12
        self._sensor_rl_foot = 13

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Joint positions (exclude root freejoint: first 7 values)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values)
        qvel = self.data.qvel[6:].copy()

        # Torso state from sensors
        torso_gyro = self.data.sensordata[
            self._sensor_gyro_start:self._sensor_gyro_start + 3
        ].copy()
        torso_accel = self.data.sensordata[
            self._sensor_accel_start:self._sensor_accel_start + 3
        ].copy()
        torso_quat = self.data.sensordata[
            self._sensor_quat_start:self._sensor_quat_start + 4
        ].copy()

        # Torso linear velocity (from root freejoint)
        torso_linvel = self.data.qvel[0:3].copy()

        # Foot contacts (from touch sensors)
        foot_contact = np.array([
            self.data.sensordata[self._sensor_fr_foot],
            self.data.sensordata[self._sensor_fl_foot],
            self.data.sensordata[self._sensor_rr_foot],
            self.data.sensordata[self._sensor_rl_foot],
        ])

        # Food info (relative to torso)
        torso_pos = self.data.xpos[self.torso_id]
        food_pos = self.data.mocap_pos[0]
        food_rel = food_pos - torso_pos
        food_distance = np.linalg.norm(food_rel)

        # Normalize food direction
        food_direction = food_rel / (food_distance + 1e-8)

        obs = np.concatenate([
            qpos,                   # Joint positions
            qvel,                   # Joint velocities
            torso_quat,             # Orientation (quaternion)
            torso_gyro,             # Angular velocity
            torso_linvel,           # Linear velocity
            torso_accel,            # Accelerometer
            foot_contact,           # Foot contacts (4)
            food_direction,         # Direction to food (unit vector)
            [food_distance],        # Distance to food (scalar)
        ]).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        # 1. Forward velocity reward (toward food)
        torso_pos = self.data.xpos[self.torso_id]
        food_pos = self.data.mocap_pos[0]
        food_dir_2d = food_pos[:2] - torso_pos[:2]
        food_dist_2d = np.linalg.norm(food_dir_2d)
        if food_dist_2d > 1e-6:
            food_dir_2d = food_dir_2d / food_dist_2d

        # Project velocity onto food direction
        vel_2d = self.data.qvel[0:2]
        forward_vel = np.dot(vel_2d, food_dir_2d)
        info["forward_vel"] = forward_vel
        reward_forward = self.forward_vel_weight * forward_vel
        info["reward_forward"] = reward_forward

        # 2. Alive bonus
        reward_alive = self.alive_bonus
        info["reward_alive"] = reward_alive

        # 3. Energy penalty (encourage efficiency)
        energy = np.sum(np.square(action))
        reward_energy = -self.energy_penalty_weight * energy
        info["reward_energy"] = reward_energy

        # 4. Gait stability (penalize high angular velocity of torso)
        torso_angvel = self.data.qvel[3:6]
        gait_instability = np.linalg.norm(torso_angvel)
        reward_gait = -self.gait_stability_weight * gait_instability
        info["gait_instability"] = gait_instability
        info["reward_gait"] = reward_gait

        # 5. Food reach bonus (head tip close to food)
        head_tip_pos = self.data.site_xpos[self.head_tip_site_id]
        head_food_dist = np.linalg.norm(head_tip_pos - food_pos)
        info["head_food_distance"] = head_food_dist

        food_reward = 0.0
        if head_food_dist < self.food_reach_threshold:
            food_reward = self.food_reach_bonus
            info["food_reached"] = 1.0
        else:
            info["food_reached"] = 0.0

        reward_food = food_reward
        info["reward_food"] = reward_food

        # 6. Approach shaping (smooth gradient toward food for head)
        reward_approach = -self.food_approach_weight * head_food_dist
        info["reward_approach"] = reward_approach

        # Total reward
        total_reward = (
            reward_forward + reward_alive + reward_energy
            + reward_gait + reward_food + reward_approach
        )
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        info = {}

        # Get torso height
        torso_z = self.data.xpos[self.torso_id, 2]
        info["torso_height"] = torso_z

        # Termination: torso too low (fallen)
        if torso_z < self.healthy_z_range[0]:
            info["termination_reason"] = "fallen"
            return True, info

        # Termination: torso too high (shouldn't happen, safety check)
        if torso_z > self.healthy_z_range[1]:
            info["termination_reason"] = "too_high"
            return True, info

        # Check for torso-ground contact (fallen over)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if (geom1 == self.torso_main_geom_id and geom2 == self.floor_geom_id) or \
               (geom2 == self.torso_main_geom_id and geom1 == self.floor_geom_id):
                info["termination_reason"] = "torso_contact"
                return True, info

        return False, info

    def _spawn_target(self):
        """Spawn food at random location (elevated, since Brachiosaurus browses high)."""
        if self.np_random is not None:
            distance = self.np_random.uniform(*self.food_distance_range)
            lateral = self.np_random.uniform(*self.food_lateral_range)
            height = self.np_random.uniform(*self.food_height_range)
        else:
            distance = np.mean(self.food_distance_range)
            lateral = 0.0
            height = np.mean(self.food_height_range)

        food_pos = np.array([distance, lateral, height])
        self.data.mocap_pos[0] = food_pos


# Register with Gymnasium
gym.register(
    id="Brachio-v0",
    entry_point="envs.brachio_env:BrachioEnv",
    max_episode_steps=1000,
)
