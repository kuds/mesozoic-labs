"""
Velociraptor Gymnasium Environment

A bipedal dinosaur locomotion environment with predatory strike behavior.

Observation space:
    - Joint positions (qpos) excluding root freejoint
    - Joint velocities (qvel) excluding root freejoint
    - Pelvis orientation (quaternion)
    - Pelvis angular velocity
    - Pelvis linear velocity
    - Foot contact states
    - Prey relative position
    - Prey distance

Action space:
    - Continuous control for all actuators [-1, 1] normalized

Reward components:
    - Forward velocity
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Strike bonus (when claw contacts prey)
    - Approach shaping (distance to prey)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv


class RaptorEnv(BaseDinoEnv):
    """Velociraptor locomotion and strike environment."""

    _camera_distance = 2.0
    _camera_azimuth = 135
    _camera_elevation = -20
    _camera_track_body = "pelvis"

    def __init__(
        self,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        # Reward weights (tune these!)
        forward_vel_weight: float = 1.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        tail_stability_weight: float = 0.05,
        strike_bonus: float = 500.0,
        strike_approach_weight: float = 0.5,
        # Environment settings
        prey_distance_range: Tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: Tuple[float, float] = (-2.0, 2.0),
        healthy_z_range: Tuple[float, float] = (0.3, 1.0),
    ):
        model_path = str(Path(__file__).parent.parent / "assets" / "raptor.xml")

        # Raptor-specific reward weights
        self.tail_stability_weight = tail_stability_weight
        self.strike_bonus = strike_bonus
        self.strike_approach_weight = strike_approach_weight

        # Raptor-specific env settings
        self.prey_distance_range = prey_distance_range
        self.prey_lateral_range = prey_lateral_range

        # Approach tracking for delta-based reward shaping
        self._prev_prey_distance: float | None = None

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
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        # Geom IDs for contact detection
        self.prey_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")
        self.r_claw_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_claw_geom")
        self.l_claw_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_claw_geom")
        self.torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso")
        self.neck_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "neck")
        self.head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Geoms that should terminate the episode on ground contact
        self._body_ground_geoms = {self.torso_geom_id, self.neck_geom_id, self.head_geom_id}

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.r_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_foot")
        self.l_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_foot")
        self.tail_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")

        # Mocap body for prey
        self.prey_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prey")

        # Sensor indices (order matches MJCF definition)
        self._sensor_gyro_start = 0
        self._sensor_accel_start = 3
        self._sensor_quat_start = 6
        self._sensor_r_foot = 10
        self._sensor_l_foot = 11

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Joint positions (exclude root freejoint: first 7 values are pos + quat)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values are lin + ang vel)
        qvel = self.data.qvel[6:].copy()

        # Pelvis state from sensors
        pelvis_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3].copy()
        pelvis_accel = self.data.sensordata[self._sensor_accel_start : self._sensor_accel_start + 3].copy()
        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4].copy()

        # Pelvis linear velocity (from root freejoint)
        pelvis_linvel = self.data.qvel[0:3].copy()

        # Foot contact (from touch sensors)
        foot_contact = np.array(
            [
                self.data.sensordata[self._sensor_r_foot],
                self.data.sensordata[self._sensor_l_foot],
            ]
        )

        # Prey info (relative to pelvis)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]  # First (and only) mocap body
        prey_rel = prey_pos - pelvis_pos
        prey_distance = np.linalg.norm(prey_rel)

        # Normalize prey direction
        prey_direction = prey_rel / (prey_distance + 1e-8)

        obs = np.concatenate(
            [
                qpos,  # Joint positions
                qvel,  # Joint velocities
                pelvis_quat,  # Orientation (quaternion)
                pelvis_gyro,  # Angular velocity
                pelvis_linvel,  # Linear velocity
                pelvis_accel,  # Accelerometer
                foot_contact,  # Foot contacts
                prey_direction,  # Direction to prey (unit vector)
                [prey_distance],  # Distance to prey (scalar)
            ]
        ).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        # 1. Forward velocity reward
        forward_vel = self.data.qvel[0]  # x-velocity of root
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

        # 4. Tail stability (penalize high angular velocity at tail tip)
        tail_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.tail_tip_site_id, tail_vel, 0)
        tail_tip_angvel = tail_vel[0:3]  # Angular velocity (first 3 elements, rot:lin order)
        tail_instability = np.linalg.norm(tail_tip_angvel)
        reward_tail = -self.tail_stability_weight * tail_instability
        info["tail_instability"] = tail_instability
        info["reward_tail"] = reward_tail

        # 5. Strike bonus (check claw-prey contact)
        strike_reward = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Check if either claw touched prey
            claw_geoms = {self.r_claw_geom_id, self.l_claw_geom_id}
            if (geom1 in claw_geoms and geom2 == self.prey_geom_id) or (
                geom2 in claw_geoms and geom1 == self.prey_geom_id
            ):
                strike_reward = self.strike_bonus
                info["strike_success"] = 1.0
                break
        else:
            info["strike_success"] = 0.0

        reward_strike = strike_reward
        info["reward_strike"] = reward_strike

        # 6. Approach shaping (reward closing distance to prey, penalise retreating)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        prey_distance = float(np.linalg.norm(prey_pos - pelvis_pos))

        if self._prev_prey_distance is not None:
            approach_delta = self._prev_prey_distance - prey_distance
        else:
            approach_delta = 0.0
        self._prev_prey_distance = prey_distance

        reward_approach = self.strike_approach_weight * approach_delta
        info["prey_distance"] = prey_distance
        info["approach_delta"] = approach_delta
        info["reward_approach"] = reward_approach

        # Total reward
        total_reward = reward_forward + reward_alive + reward_energy + reward_tail + reward_strike + reward_approach
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        info = {}

        # Get pelvis height
        pelvis_z = self.data.xpos[self.pelvis_id, 2]
        info["pelvis_height"] = pelvis_z

        # Compute tilt angle from pelvis orientation
        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        tilt_angle = self._quat_to_tilt(pelvis_quat)
        info["tilt_angle"] = tilt_angle

        # Termination: pelvis too low (fallen)
        if pelvis_z < self.healthy_z_range[0]:
            info["termination_reason"] = "fallen"
            return True, info

        # Termination: pelvis too high (shouldn't happen, but safety check)
        if pelvis_z > self.healthy_z_range[1]:
            info["termination_reason"] = "too_high"
            return True, info

        # Termination: excessive tilt (about to fall)
        if tilt_angle > self.max_tilt_angle:
            info["termination_reason"] = "excessive_tilt"
            return True, info

        # Check for body-ground contact (torso, neck, or head touching floor)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if geom2 == self.floor_geom_id and geom1 in self._body_ground_geoms:
                info["termination_reason"] = "body_contact"
                return True, info
            if geom1 == self.floor_geom_id and geom2 in self._body_ground_geoms:
                info["termination_reason"] = "body_contact"
                return True, info

        return False, info

    def _spawn_target(self):
        """Spawn prey at random location ahead of raptor."""
        if self.np_random is not None:
            distance = self.np_random.uniform(*self.prey_distance_range)
            lateral = self.np_random.uniform(*self.prey_lateral_range)
        else:
            distance = np.mean(self.prey_distance_range)
            lateral = 0.0

        # Prey position (relative to origin, raptor starts at origin)
        prey_pos = np.array([distance, lateral, 0.3])
        self.data.mocap_pos[0] = prey_pos

        # Reset approach tracking (first step will produce zero delta)
        self._prev_prey_distance = None


# Register with Gymnasium (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/Raptor-v0",
    entry_point="environments.velociraptor.envs.raptor_env:RaptorEnv",
    max_episode_steps=1000,
)
