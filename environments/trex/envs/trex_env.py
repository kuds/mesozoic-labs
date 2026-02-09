"""
Tyrannosaurus Rex Gymnasium Environment

A large bipedal predator with a massive skull, powerful jaws, and
vestigial forelimbs.  The T-Rex hunts by sprinting toward prey and
delivering a bite with its jaw.

Observation space:
    - Joint positions (qpos) excluding root freejoint
    - Joint velocities (qvel) excluding root freejoint
    - Pelvis orientation (quaternion)
    - Pelvis angular velocity
    - Pelvis linear velocity
    - Foot contact states (2 feet)
    - Prey relative direction
    - Prey distance

Action space:
    - Continuous control for all actuators [-1, 1] normalized
    - 14 actuators: 3 neck/head + 1 jaw + 5 per leg

Reward components:
    - Forward velocity (toward prey)
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Bite bonus (jaw contacts prey)
    - Approach shaping (distance to prey)
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from typing import Optional, Dict, Any, Tuple

from environments.shared.base_env import BaseDinoEnv


class TRexEnv(BaseDinoEnv):
    """Tyrannosaurus Rex bipedal locomotion and bite-attack environment."""

    _camera_distance = 5.0
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
        tail_stability_weight: float = 0.05,
        bite_bonus: float = 500.0,
        bite_approach_weight: float = 0.5,
        # Environment settings
        prey_distance_range: Tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: Tuple[float, float] = (-2.0, 2.0),
        healthy_z_range: Tuple[float, float] = (0.4, 1.6),
    ):
        model_path = str(
            Path(__file__).parent.parent / "assets" / "trex.xml"
        )

        # T-Rex-specific reward weights
        self.tail_stability_weight = tail_stability_weight
        self.bite_bonus = bite_bonus
        self.bite_approach_weight = bite_approach_weight

        # T-Rex-specific env settings
        self.prey_distance_range = prey_distance_range
        self.prey_lateral_range = prey_lateral_range

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
        self.pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        # Geom IDs for contact detection
        self.prey_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom"
        )
        self.jaw_bite_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "jaw_bite"
        )
        self.torso_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self.r_foot_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "r_foot"
        )
        self.l_foot_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "l_foot"
        )
        self.tail_tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip"
        )
        self.head_tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tip"
        )

        # Sensor indices (order matches MJCF sensor definition)
        # pelvis_gyro(3), pelvis_accel(3), pelvis_orientation(4),
        # r_foot_touch(1), l_foot_touch(1)
        self._sensor_gyro_start = 0
        self._sensor_accel_start = 3
        self._sensor_quat_start = 6
        self._sensor_r_foot = 10
        self._sensor_l_foot = 11

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Joint positions (exclude root freejoint: first 7 values)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values)
        qvel = self.data.qvel[6:].copy()

        # Pelvis state from sensors
        pelvis_gyro = self.data.sensordata[
            self._sensor_gyro_start:self._sensor_gyro_start + 3
        ].copy()
        pelvis_accel = self.data.sensordata[
            self._sensor_accel_start:self._sensor_accel_start + 3
        ].copy()
        pelvis_quat = self.data.sensordata[
            self._sensor_quat_start:self._sensor_quat_start + 4
        ].copy()

        # Pelvis linear velocity (from root freejoint)
        pelvis_linvel = self.data.qvel[0:3].copy()

        # Foot contact (from touch sensors)
        foot_contact = np.array([
            self.data.sensordata[self._sensor_r_foot],
            self.data.sensordata[self._sensor_l_foot],
        ])

        # Prey info (relative to pelvis)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        prey_rel = prey_pos - pelvis_pos
        prey_distance = np.linalg.norm(prey_rel)

        # Normalize prey direction
        prey_direction = prey_rel / (prey_distance + 1e-8)

        obs = np.concatenate([
            qpos,                   # Joint positions
            qvel,                   # Joint velocities
            pelvis_quat,            # Orientation (quaternion)
            pelvis_gyro,            # Angular velocity
            pelvis_linvel,          # Linear velocity
            pelvis_accel,           # Accelerometer
            foot_contact,           # Foot contacts (2)
            prey_direction,         # Direction to prey (unit vector)
            [prey_distance],        # Distance to prey (scalar)
        ]).astype(np.float32)

        return obs

    def _get_reward_info(
        self, action: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        # 1. Forward velocity reward (toward prey)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        prey_dir_2d = prey_pos[:2] - pelvis_pos[:2]
        prey_dist_2d = np.linalg.norm(prey_dir_2d)
        if prey_dist_2d > 1e-6:
            prey_dir_2d = prey_dir_2d / prey_dist_2d

        vel_2d = self.data.qvel[0:2]
        forward_vel = np.dot(vel_2d, prey_dir_2d)
        info["forward_vel"] = forward_vel
        reward_forward = self.forward_vel_weight * forward_vel
        info["reward_forward"] = reward_forward

        # 2. Alive bonus
        reward_alive = self.alive_bonus
        info["reward_alive"] = reward_alive

        # 3. Energy penalty
        energy = np.sum(np.square(action))
        reward_energy = -self.energy_penalty_weight * energy
        info["reward_energy"] = reward_energy

        # 4. Tail stability (penalize high angular velocity at tail tip)
        tail_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_SITE, self.tail_tip_site_id,
            tail_vel, 0,
        )
        tail_tip_angvel = tail_vel[3:6]
        tail_instability = np.linalg.norm(tail_tip_angvel)
        reward_tail = -self.tail_stability_weight * tail_instability
        info["tail_instability"] = tail_instability
        info["reward_tail"] = reward_tail

        # 5. Bite bonus (check jaw_bite-prey contact)
        bite_reward = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if (geom1 == self.jaw_bite_geom_id and geom2 == self.prey_geom_id) or \
               (geom2 == self.jaw_bite_geom_id and geom1 == self.prey_geom_id):
                bite_reward = self.bite_bonus
                info["bite_success"] = 1.0
                break
        else:
            info["bite_success"] = 0.0

        reward_bite = bite_reward
        info["reward_bite"] = reward_bite

        # 6. Approach shaping (smooth gradient toward prey)
        prey_distance = np.linalg.norm(prey_pos - pelvis_pos)
        reward_approach = -self.bite_approach_weight * prey_distance
        info["prey_distance"] = prey_distance
        info["reward_approach"] = reward_approach

        # Total reward
        total_reward = (
            reward_forward + reward_alive + reward_energy
            + reward_tail + reward_bite + reward_approach
        )
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        info = {}

        # Get pelvis height
        pelvis_z = self.data.xpos[self.pelvis_id, 2]
        info["pelvis_height"] = pelvis_z

        # Termination: pelvis too low (fallen)
        if pelvis_z < self.healthy_z_range[0]:
            info["termination_reason"] = "fallen"
            return True, info

        # Termination: pelvis too high (safety check)
        if pelvis_z > self.healthy_z_range[1]:
            info["termination_reason"] = "too_high"
            return True, info

        # Check for torso-ground contact (fallen over)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if (geom1 == self.torso_geom_id and geom2 == self.floor_geom_id) or \
               (geom2 == self.torso_geom_id and geom1 == self.floor_geom_id):
                info["termination_reason"] = "torso_contact"
                return True, info

        return False, info

    def _spawn_target(self):
        """Spawn prey at random location ahead of T-Rex."""
        if self.np_random is not None:
            distance = self.np_random.uniform(*self.prey_distance_range)
            lateral = self.np_random.uniform(*self.prey_lateral_range)
        else:
            distance = np.mean(self.prey_distance_range)
            lateral = 0.0

        prey_pos = np.array([distance, lateral, 0.5])
        self.data.mocap_pos[0] = prey_pos


# Register with Gymnasium (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/TRex-v0",
    entry_point="environments.trex.envs.trex_env:TRexEnv",
    max_episode_steps=1000,
)
