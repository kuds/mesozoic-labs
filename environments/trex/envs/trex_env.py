"""
Tyrannosaurus Rex Gymnasium Environment

A large bipedal predator with a massive skull and vestigial forelimbs.
The T-Rex hunts by sprinting toward prey and delivering a bite with
its head.

Observation space (83 dims):
    - Joint positions (qpos[7:]) — 33 (25 hinge + 2x4 ball shoulders)
    - Joint velocities (qvel[6:]) — 31 (25 hinge + 2x3 ball shoulders)
    - Pelvis orientation (quaternion) — 4
    - Pelvis angular velocity (gyroscope) — 3
    - Pelvis linear velocity — 3
    - Pelvis acceleration — 3
    - Foot contact states (2 feet, sensed on central digit 3) — 2
    - Prey direction (unit vector) — 3
    - Prey distance (scalar) — 1

Action space (21 dims):
    - Neck/head: neck pitch, neck yaw, head pitch (3)
    - Right leg: hip pitch/roll, knee, ankle, toe d2/d3/d4 (7)
    - Left leg: hip pitch/roll, knee, ankle, toe d2/d3/d4 (7)
    - Tail: pitch 1, yaw 1, pitch 2, pitch 3 (4)

Reward components:
    - Forward velocity (toward prey)
    - Backward velocity penalty
    - Drift penalty (horizontal displacement from spawn)
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Bite bonus (head contacts prey)
    - Approach shaping (distance to prey)
    - Head proximity shaping (reward for positioning head near prey)
    - Posture (continuous tilt penalty)
    - Nosedive penalty
    - Height maintenance
    - Gait symmetry (alternating foot contacts)
    - Action smoothness (penalize jerky action changes)
    - Spin penalty (penalize pelvis angular velocity)
    - Heading alignment (facing toward prey)
    - Lateral velocity penalty (anti crab-walk)
    - Speed penalty (penalise absolute speed above threshold)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv


class TRexEnv(BaseDinoEnv):
    """Tyrannosaurus Rex bipedal locomotion and bite-attack environment."""

    _camera_distance = 3.0
    _camera_azimuth = 135
    _camera_elevation = -20
    _camera_track_body = "pelvis"

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        # Reward weights
        forward_vel_weight: float = 1.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        tail_stability_weight: float = 0.05,
        bite_bonus: float = 10.0,
        bite_approach_weight: float = 1.0,
        bite_head_proximity_weight: float = 0.0,
        posture_weight: float = 0.2,
        nosedive_weight: float = 0.0,
        natural_pitch: float = 0.17,
        height_weight: float = 0.0,
        gait_symmetry_weight: float = 0.0,
        smoothness_weight: float = 0.05,
        heading_weight: float = 0.0,
        lateral_penalty_weight: float = 0.0,
        backward_vel_penalty_weight: float = 0.0,
        drift_penalty_weight: float = 0.0,
        spin_penalty_weight: float = 0.0,
        speed_penalty_weight: float = 0.0,
        speed_penalty_threshold: float = 0.10,
        idle_penalty_weight: float = 0.0,
        idle_velocity_threshold: float = 0.05,
        forward_vel_max: float = 8.0,
        # Environment settings
        prey_distance_range: tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: tuple[float, float] = (-2.0, 2.0),
        healthy_z_range: tuple[float, float] = (0.5, 1.6),
        reset_noise_scale: float = 0.01,
    ):
        model_path = str(Path(__file__).parent.parent / "assets" / "trex.xml")

        # T-Rex-specific reward weights
        self.tail_stability_weight = tail_stability_weight
        self.bite_bonus = bite_bonus
        self.bite_approach_weight = bite_approach_weight
        self.bite_head_proximity_weight = bite_head_proximity_weight
        self.posture_weight = posture_weight
        self.nosedive_weight = nosedive_weight
        self.height_weight = height_weight
        self.gait_symmetry_weight = gait_symmetry_weight
        self.smoothness_weight = smoothness_weight
        self.heading_weight = heading_weight
        self.lateral_penalty_weight = lateral_penalty_weight
        self.backward_vel_penalty_weight = backward_vel_penalty_weight
        self.drift_penalty_weight = drift_penalty_weight
        self.spin_penalty_weight = spin_penalty_weight
        self.speed_penalty_weight = speed_penalty_weight
        self.speed_penalty_threshold = speed_penalty_threshold
        self.idle_penalty_weight = idle_penalty_weight
        self.idle_velocity_threshold = idle_velocity_threshold
        self.forward_vel_max = forward_vel_max

        # Natural forward pitch (~10°). The nosedive penalty and termination
        # are measured relative to this angle so the T-Rex isn't punished for
        # its biomechanically correct forward lean.
        self._natural_forward_z = -np.sin(natural_pitch)

        # T-Rex-specific env settings
        self.prey_distance_range = prey_distance_range
        self.prey_lateral_range = prey_lateral_range

        # State tracking for delta-based rewards
        self._prev_prey_distance: float | None = None
        self._prev_action: np.ndarray | None = None

        # Gait symmetry: track foot touchdown events for alternation reward
        self._init_gait_state()

        # Cached initial direction to prey (set in _spawn_target).
        # Used by forward-velocity and heading rewards so the "forward"
        # reference direction stays fixed for the whole episode, preventing
        # the reward from flipping sign when the T-Rex passes the prey.
        self._initial_prey_dir_2d: np.ndarray = np.array([1.0, 0.0])

        # Cached initial pelvis position (set in _spawn_target).
        # Used by the drift penalty to discourage horizontal displacement.
        self._initial_pos_2d: np.ndarray = np.array([0.0, 0.0])

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
            reset_noise_scale=reset_noise_scale,
        )

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""
        # Body IDs
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        # Geom IDs for contact detection
        self.prey_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")
        self.head_bite_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head_bite")
        self.torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Head/skull geom IDs (collision-enabled geoms that should terminate on ground contact)
        # Note: neck_geom and brow_ridge have contype=0, so they never produce floor contacts
        self.skull_upper_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "skull_upper")
        self.snout_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "snout")

        # Tail geom IDs (distal segments that should not contact floor)
        self.tail_3_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_3_geom")
        self.tail_4_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_4_geom")
        self.tail_5_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_5_geom")

        # Geoms that should terminate the episode on ground contact
        self._body_ground_geoms = {
            self.torso_geom_id,
            self.skull_upper_geom_id,
            self.snout_geom_id,
            self.tail_3_geom_id,
            self.tail_4_geom_id,
            self.tail_5_geom_id,
        }
        self._head_ground_geoms = {self.skull_upper_geom_id, self.snout_geom_id}
        self._tail_ground_geoms = {self.tail_3_geom_id, self.tail_4_geom_id, self.tail_5_geom_id}

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.r_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_foot")
        self.l_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_foot")
        self.tail_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")
        self.head_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tip")

        # Prey mocap body
        self.prey_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prey")

        # Sensor indices (order matches MJCF sensor definition)
        # pelvis_gyro(3), pelvis_accel(3), pelvis_orientation(4),
        # r_foot_touch(1), l_foot_touch(1)
        # _sensor_gyro_start, _sensor_accel_start, _sensor_quat_start
        # are inherited from BaseDinoEnv (0, 3, 6 respectively).
        self._sensor_r_foot = 10
        self._sensor_l_foot = 11

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Joint positions (exclude root freejoint: first 7 values)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values)
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
        prey_pos = self.data.mocap_pos[0]
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
                foot_contact,  # Foot contacts (2)
                prey_direction,  # Direction to prey (unit vector)
                [prey_distance],  # Distance to prey (scalar)
            ]
        ).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        forward_ref_2d = self._initial_prey_dir_2d
        vel_2d = self.data.qvel[0:2]

        # 1. Forward velocity reward (toward prey)
        reward_forward, forward_vel = self._compute_forward_velocity(
            vel_2d, forward_ref_2d, self.forward_vel_max, self.forward_vel_weight
        )
        info["forward_vel"] = forward_vel
        info["reward_forward"] = reward_forward

        # 1b. Backward velocity penalty
        reward_backward, backward_vel = self._compute_backward_penalty(
            forward_vel, self.forward_vel_max, self.backward_vel_penalty_weight
        )
        info["backward_vel"] = backward_vel
        info["reward_backward"] = reward_backward

        # 1c. Drift penalty
        reward_drift, drift_dist = self._compute_drift_penalty(
            pelvis_pos[:2], self._initial_pos_2d, self.drift_penalty_weight
        )
        info["drift_distance"] = drift_dist
        info["reward_drift"] = reward_drift

        # 2. Alive bonus (shared helper)
        reward_alive = self._reward_alive()
        info["reward_alive"] = reward_alive

        # 3. Energy penalty (shared helper)
        reward_energy = self._reward_energy(action)
        info["reward_energy"] = reward_energy

        # 4. Tail stability
        reward_tail, tail_instability = self._compute_tail_stability(self.tail_tip_site_id, self.tail_stability_weight)
        info["tail_instability"] = tail_instability
        info["reward_tail"] = reward_tail

        # 5. Bite bonus (check head_bite-prey contact)
        bite_reward = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if (geom1 == self.head_bite_geom_id and geom2 == self.prey_geom_id) or (
                geom2 == self.head_bite_geom_id and geom1 == self.prey_geom_id
            ):
                bite_reward = self.bite_bonus
                info["bite_success"] = 1.0
                break
        else:
            info["bite_success"] = 0.0

        reward_bite = bite_reward
        info["reward_bite"] = reward_bite

        # 6. Approach shaping
        prey_distance = float(np.linalg.norm(prey_pos - pelvis_pos))
        reward_approach, approach_delta = self._compute_approach_shaping(
            prey_distance, self._prev_prey_distance, self.bite_approach_weight, 8.0
        )
        self._prev_prey_distance = prey_distance
        info["prey_distance"] = prey_distance
        info["approach_delta"] = approach_delta
        info["reward_approach"] = reward_approach

        # 6b. Head-to-prey proximity shaping
        # Uses the head tip position (not the pelvis) to give the agent a
        # gradient for aiming its head toward the prey.  Analogous to the
        # raptor's claw proximity reward.
        head_tip_pos = self.data.site_xpos[self.head_tip_site_id]
        head_prey_dist = float(np.linalg.norm(prey_pos - head_tip_pos))
        head_proximity_max_dist = 3.0
        head_proximity = max(0.0, 1.0 - head_prey_dist / head_proximity_max_dist)
        reward_head_proximity = self.bite_head_proximity_weight * head_proximity
        info["head_prey_distance"] = head_prey_dist
        info["head_proximity"] = head_proximity
        info["reward_head_proximity"] = reward_head_proximity

        # 7. Continuous posture reward
        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        reward_posture, tilt_angle = self._compute_posture_reward(pelvis_quat, self.posture_weight)
        info["tilt_angle"] = tilt_angle
        info["reward_posture"] = reward_posture

        # 8. Nosedive penalty
        reward_nosedive, forward_z = self._compute_nosedive_penalty(
            pelvis_quat, self.nosedive_weight, self._natural_forward_z
        )
        info["forward_z"] = forward_z
        info["reward_nosedive"] = reward_nosedive

        # 8b. Pelvis height (for LocomotionMetrics tracking)
        pelvis_height = float(self.data.xpos[self.pelvis_id, 2])
        info["pelvis_height"] = pelvis_height

        # 8c. Height maintenance reward (smooth gradient toward staying upright)
        min_z = self.healthy_z_range[0]
        target_z = 0.90
        height_frac = float(np.clip((pelvis_height - min_z) / (target_z - min_z), 0.0, 1.0))
        reward_height = self.height_weight * height_frac
        info["reward_height"] = reward_height

        # 9. Gait symmetry (reward alternating foot contacts, shared helper)
        r_contact = self.data.sensordata[self._sensor_r_foot]
        l_contact = self.data.sensordata[self._sensor_l_foot]
        info["r_foot_contact"] = float(r_contact)
        info["l_foot_contact"] = float(l_contact)

        reward_gait, alternation_ratio = self._compute_gait_symmetry(
            float(r_contact), float(l_contact), self.gait_symmetry_weight
        )
        info["alternation_ratio"] = alternation_ratio
        info["contact_asymmetry"] = alternation_ratio  # backward compat with metrics
        info["reward_gait"] = reward_gait

        # 10. Action smoothness (shared helper)
        reward_smoothness, action_delta = self._reward_action_smoothness(action)
        info["action_delta"] = action_delta
        info["reward_smoothness"] = reward_smoothness

        # 11. Heading alignment
        body_forward_2d = self._quat_to_forward_2d(pelvis_quat)
        reward_heading, heading_alignment = self._compute_heading_alignment(
            body_forward_2d, forward_ref_2d, self.heading_weight
        )
        info["heading_alignment"] = heading_alignment
        info["reward_heading"] = reward_heading

        # 12. Lateral velocity penalty
        reward_lateral, lateral_vel = self._compute_lateral_velocity_penalty(
            vel_2d, body_forward_2d, self.lateral_penalty_weight
        )
        info["lateral_vel"] = lateral_vel
        info["reward_lateral"] = reward_lateral

        # Pelvis angular velocity (for spinning detection in shared diagnostics)
        pelvis_angular_vel, pelvis_yaw_vel = self._compute_pelvis_diagnostics()
        info["pelvis_angular_vel"] = pelvis_angular_vel
        info["pelvis_yaw_vel"] = pelvis_yaw_vel

        # 13. Spin penalty
        reward_spin, spin_instability = self._compute_angular_velocity_penalty(self.spin_penalty_weight)
        info["spin_instability"] = spin_instability
        info["reward_spin"] = reward_spin

        # 14. Speed penalty (penalise absolute speed above threshold)
        reward_speed, abs_speed = self._compute_speed_penalty(
            vel_2d, self.speed_penalty_weight, self.speed_penalty_threshold
        )
        info["abs_speed"] = abs_speed
        info["reward_speed"] = reward_speed

        # 14b. Idle penalty (penalise standing still / barely moving)
        reward_idle, idle_speed = self._compute_idle_penalty(
            vel_2d, self.idle_penalty_weight, self.idle_velocity_threshold
        )
        info["reward_idle"] = reward_idle

        # Total reward
        total_reward = (
            reward_forward
            + reward_backward
            + reward_drift
            + reward_alive
            + reward_energy
            + reward_tail
            + reward_bite
            + reward_approach
            + reward_head_proximity
            + reward_posture
            + reward_nosedive
            + reward_height
            + reward_gait
            + reward_smoothness
            + reward_heading
            + reward_lateral
            + reward_spin
            + reward_speed
            + reward_idle
        )
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        """Check if episode should terminate."""
        info = {}

        pelvis_z = self.data.xpos[self.pelvis_id, 2]
        info["pelvis_height"] = pelvis_z

        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        tilt_angle = self._quat_to_tilt(pelvis_quat)
        info["tilt_angle"] = tilt_angle

        # Height/tilt termination (shared)
        terminated, reason = self._check_height_tilt_termination(pelvis_z, tilt_angle)
        if terminated:
            info["termination_reason"] = reason
            return True, info

        # Nosedive termination
        forward_z = self._quat_to_forward_z(pelvis_quat)
        info["forward_z"] = forward_z
        if forward_z < self._natural_forward_z - 0.5:
            info["termination_reason"] = "nosedive"
            return True, info

        # Check contacts: head-prey (success)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if self.bite_bonus > 0 and (
                (geom1 == self.head_bite_geom_id and geom2 == self.prey_geom_id)
                or (geom2 == self.head_bite_geom_id and geom1 == self.prey_geom_id)
            ):
                info["termination_reason"] = "bite_success"
                info["success"] = True
                return True, info

        # Floor contact termination (shared)
        terminated, reason = self._check_floor_contact(
            self._body_ground_geoms,
            self.floor_geom_id,
            geom_categories={
                "tail": self._tail_ground_geoms,
                "head": self._head_ground_geoms,
                "torso": {self.torso_geom_id},
            },
        )
        if terminated:
            info["termination_reason"] = reason
            return True, info

        return False, info

    def _spawn_target(self):
        """Spawn prey at random location ahead of T-Rex."""
        prey_pos = self._spawn_target_2d(self.prey_distance_range, self.prey_lateral_range, 0.5)
        self._initial_prey_dir_2d = self._compute_initial_direction_2d(prey_pos)
        self._initial_pos_2d = self.data.qpos[0:2].copy()

        # Reset delta-based tracking (first step will produce zero deltas)
        self._prev_prey_distance = None
        self._prev_action = None

        # Reset gait symmetry tracking
        self._reset_gait_state()


# Register with Gymnasium (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/TRex-v0",
    entry_point="environments.trex.envs.trex_env:TRexEnv",
    max_episode_steps=1000,
)
