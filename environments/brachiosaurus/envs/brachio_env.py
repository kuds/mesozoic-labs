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
    - 26 actuators: 6 neck + 20 legs (5 per leg: hip pitch/roll, knee, ankle, toe)

Reward components:
    - Forward velocity toward food
    - Backward velocity penalty
    - Drift penalty (horizontal displacement from spawn)
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Gait stability (penalize torso angular velocity)
    - Gait symmetry (alternating foot contacts)
    - Action smoothness (penalize jerky action changes)
    - Posture (continuous tilt penalty)
    - Nosedive penalty (excessive forward pitch)
    - Height maintenance (reward maintaining target torso height)
    - Heading alignment (facing toward food)
    - Lateral velocity penalty (anti crab-walk)
    - Spin penalty (penalize torso angular velocity)
    - Speed penalty (penalise absolute speed above threshold)
    - Tail stability (penalize tail tip angular velocity)
    - Food reach bonus (when head gets close to food)
    - Approach shaping (distance to food)
    - Head proximity shaping (continuous head-to-food distance reward)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv


class BrachioEnv(BaseDinoEnv):
    """Brachiosaurus quadrupedal locomotion and food-reaching environment."""

    _camera_distance = 5.0
    _camera_azimuth = 135
    _camera_elevation = -20
    _camera_track_body = "torso"

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
        gait_stability_weight: float = 0.05,
        gait_symmetry_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        posture_weight: float = 0.0,
        nosedive_weight: float = 0.0,
        natural_pitch: float = 0.0,
        height_weight: float = 0.0,
        heading_weight: float = 0.0,
        lateral_penalty_weight: float = 0.0,
        backward_vel_penalty_weight: float = 0.0,
        drift_penalty_weight: float = 0.0,
        spin_penalty_weight: float = 0.0,
        tail_stability_weight: float = 0.0,
        speed_penalty_weight: float = 0.0,
        speed_penalty_threshold: float = 0.10,
        idle_penalty_weight: float = 0.0,
        idle_velocity_threshold: float = 0.05,
        forward_vel_max: float = 1.0,
        food_reach_bonus: float = 10.0,
        food_reach_threshold: float = 0.5,
        food_approach_weight: float = 1.0,
        food_head_proximity_weight: float = 0.0,
        # Environment settings
        food_distance_range: tuple[float, float] = (3.0, 8.0),
        food_lateral_range: tuple[float, float] = (-2.0, 2.0),
        food_height_range: tuple[float, float] = (2.0, 4.0),
        healthy_z_range: tuple[float, float] = (1.0, 3.5),
        reset_noise_scale: float = 0.01,
    ):
        model_path = str(Path(__file__).parent.parent / "assets" / "brachiosaurus.xml")

        # Brachio-specific reward weights
        self.gait_stability_weight = gait_stability_weight
        self.gait_symmetry_weight = gait_symmetry_weight
        self.smoothness_weight = smoothness_weight
        self.posture_weight = posture_weight
        self.nosedive_weight = nosedive_weight
        self.height_weight = height_weight
        self.heading_weight = heading_weight
        self.lateral_penalty_weight = lateral_penalty_weight
        self.backward_vel_penalty_weight = backward_vel_penalty_weight
        self.drift_penalty_weight = drift_penalty_weight
        self.spin_penalty_weight = spin_penalty_weight
        self.tail_stability_weight = tail_stability_weight
        self.speed_penalty_weight = speed_penalty_weight
        self.speed_penalty_threshold = speed_penalty_threshold
        self.idle_penalty_weight = idle_penalty_weight
        self.idle_velocity_threshold = idle_velocity_threshold
        self.forward_vel_max = forward_vel_max
        self.food_reach_bonus = food_reach_bonus
        self.food_reach_threshold = food_reach_threshold
        self.food_approach_weight = food_approach_weight
        self.food_head_proximity_weight = food_head_proximity_weight

        # Natural forward pitch. The nosedive penalty is measured relative to
        # this angle so the Brachiosaurus isn't punished for its natural stance.
        # Default 0.0 (upright quadruped); override if the model has a natural lean.
        self._natural_forward_z = -np.sin(natural_pitch)

        # Brachio-specific env settings
        self.food_distance_range = food_distance_range
        self.food_lateral_range = food_lateral_range
        self.food_height_range = food_height_range

        # State tracking for delta-based rewards
        self._prev_head_food_distance: float | None = None
        self._prev_action: np.ndarray | None = None

        # Gait symmetry: track diagonal pair alternation for quadrupedal gait.
        # Diagonal-A = FR+RL, Diagonal-B = FL+RR.  A proper walk/trot
        # alternates these pairs.
        self._init_quadruped_gait_state()

        # Cached initial direction to food (set in _spawn_target).
        # Used by forward-velocity and heading rewards so the "forward"
        # reference direction stays fixed for the whole episode.
        self._initial_food_dir_2d: np.ndarray = np.array([1.0, 0.0])

        # Cached initial torso position (set in _spawn_target).
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
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # Geom IDs for contact detection
        self.food_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "food_geom")
        self.torso_main_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_main")
        self.belly_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "belly")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Head geom ID — head_geom has default contype=1 so it CAN produce
        # floor contacts.  Neck geoms (neck_1 through neck_4) have contype=0
        # and cannot collide with the floor.
        self.head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head_geom")

        # Tail geom IDs — distal segments (tail_3, tail_4) that should not
        # contact floor.  Proximal tail_1/tail_2 may touch during walking.
        self.tail_3_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_3_geom")
        self.tail_4_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_4_geom")

        # Geoms that should terminate the episode on ground contact.
        # Only includes geoms with contype != 0 (capable of floor collision).
        self._body_ground_geoms = {
            self.torso_main_geom_id,
            self.belly_geom_id,
            self.head_geom_id,
            self.tail_3_geom_id,
            self.tail_4_geom_id,
        }
        self._head_ground_geoms = {self.head_geom_id}
        self._tail_ground_geoms = {self.tail_3_geom_id, self.tail_4_geom_id}
        self._torso_ground_geoms = {self.torso_main_geom_id, self.belly_geom_id}

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.head_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tip")
        self.tail_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")

        # Metatarsal geom IDs (digitigrade stance — metacarpus/metatarsus)
        self.fr_meta_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fr_meta_geom")
        self.fl_meta_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fl_meta_geom")
        self.rr_meta_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rr_meta_geom")
        self.rl_meta_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rl_meta_geom")

        # Foot site IDs
        self.foot_site_ids = {
            "fr": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fr_foot_contact"),
            "fl": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot_contact"),
            "rr": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rr_foot_contact"),
            "rl": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rl_foot_contact"),
        }

        # Sensor indices (order matches MJCF definition)
        # _sensor_gyro_start, _sensor_accel_start, _sensor_quat_start
        # are inherited from BaseDinoEnv (0, 3, 6 respectively).
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
        torso_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3].copy()
        torso_accel = self.data.sensordata[self._sensor_accel_start : self._sensor_accel_start + 3].copy()
        torso_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4].copy()

        # Torso linear velocity (from root freejoint)
        torso_linvel = self.data.qvel[0:3].copy()

        # Foot contacts (from touch sensors)
        foot_contact = np.array(
            [
                self.data.sensordata[self._sensor_fr_foot],
                self.data.sensordata[self._sensor_fl_foot],
                self.data.sensordata[self._sensor_rr_foot],
                self.data.sensordata[self._sensor_rl_foot],
            ]
        )

        # Food info (relative to torso)
        torso_pos = self.data.xpos[self.torso_id]
        food_pos = self.data.mocap_pos[0]
        food_rel = food_pos - torso_pos
        food_distance = np.linalg.norm(food_rel)

        # Normalize food direction
        food_direction = food_rel / (food_distance + 1e-8)

        obs = np.concatenate(
            [
                qpos,  # Joint positions
                qvel,  # Joint velocities
                torso_quat,  # Orientation (quaternion)
                torso_gyro,  # Angular velocity
                torso_linvel,  # Linear velocity
                torso_accel,  # Accelerometer
                foot_contact,  # Foot contacts (4)
                food_direction,  # Direction to food (unit vector)
                [food_distance],  # Distance to food (scalar)
            ]
        ).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        torso_pos = self.data.xpos[self.torso_id]
        food_pos = self.data.mocap_pos[0]
        forward_ref_2d = self._initial_food_dir_2d
        vel_2d = self.data.qvel[0:2]

        # 1. Forward velocity reward (toward food)
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
            torso_pos[:2], self._initial_pos_2d, self.drift_penalty_weight
        )
        info["drift_distance"] = drift_dist
        info["reward_drift"] = reward_drift

        # 2. Alive bonus (shared helper)
        reward_alive = self._reward_alive()
        info["reward_alive"] = reward_alive

        # 3. Energy penalty (shared helper)
        reward_energy = self._reward_energy(action)
        info["reward_energy"] = reward_energy

        # 4. Gait stability (penalize high angular velocity of torso)
        reward_gait_stability, gait_instability = self._compute_angular_velocity_penalty(
            self.gait_stability_weight, max_angvel=5.0
        )
        info["gait_instability"] = gait_instability
        info["reward_gait"] = reward_gait_stability

        # 5. Tail stability
        reward_tail, tail_instability = self._compute_tail_stability(self.tail_tip_site_id, self.tail_stability_weight)
        info["tail_instability"] = tail_instability
        info["reward_tail"] = reward_tail

        # 6. Posture reward (continuous tilt penalty)
        torso_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        reward_posture, tilt_angle = self._compute_posture_reward(torso_quat, self.posture_weight)
        info["tilt_angle"] = tilt_angle
        info["reward_posture"] = reward_posture

        # 7. Nosedive penalty
        reward_nosedive, forward_z = self._compute_nosedive_penalty(
            torso_quat, self.nosedive_weight, self._natural_forward_z
        )
        info["forward_z"] = forward_z
        info["reward_nosedive"] = reward_nosedive

        # 8. Height maintenance reward
        torso_height = float(torso_pos[2])
        info["pelvis_height"] = torso_height
        min_z = self.healthy_z_range[0]
        target_z = 1.2  # Brachiosaurus natural standing torso height (from XML keyframe)
        height_frac = float(np.clip((torso_height - min_z) / (target_z - min_z), 0.0, 1.0))
        reward_height = self.height_weight * height_frac
        info["reward_height"] = reward_height

        # 9. Gait symmetry (reward alternating diagonal pair contacts)
        fr_contact = self.data.sensordata[self._sensor_fr_foot]
        fl_contact = self.data.sensordata[self._sensor_fl_foot]
        rr_contact = self.data.sensordata[self._sensor_rr_foot]
        rl_contact = self.data.sensordata[self._sensor_rl_foot]
        info["r_foot_contact"] = float(fr_contact)
        info["l_foot_contact"] = float(fl_contact)
        info["rr_foot_contact"] = float(rr_contact)
        info["rl_foot_contact"] = float(rl_contact)

        reward_gait_sym, alternation_ratio = self._compute_quadruped_gait_symmetry(
            float(fr_contact),
            float(fl_contact),
            float(rr_contact),
            float(rl_contact),
            self.gait_symmetry_weight,
        )
        info["alternation_ratio"] = alternation_ratio
        info["contact_asymmetry"] = alternation_ratio
        info["reward_gait_symmetry"] = reward_gait_sym

        # 10. Action smoothness (shared helper)
        reward_smoothness, action_delta = self._reward_action_smoothness(action)
        info["action_delta"] = action_delta
        info["reward_smoothness"] = reward_smoothness

        # 11. Heading alignment
        body_forward_2d = self._quat_to_forward_2d(torso_quat)
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

        # Torso angular velocity (for spinning detection in shared diagnostics)
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

        # 15. Food reach bonus (head tip close to food)
        head_tip_pos = self.data.site_xpos[self.head_tip_site_id]
        head_food_dist = float(np.linalg.norm(head_tip_pos - food_pos))
        info["head_food_distance"] = head_food_dist
        info["prey_distance"] = float(head_food_dist)  # alias for LocomotionMetrics

        food_reward = 0.0
        if head_food_dist < self.food_reach_threshold:
            food_reward = self.food_reach_bonus
            info["food_reached"] = 1.0
        else:
            info["food_reached"] = 0.0

        reward_food = food_reward
        info["reward_food"] = reward_food

        # 16. Approach shaping (delta-based)
        head_food_dist_f = float(head_food_dist)
        reward_approach, approach_delta = self._compute_approach_shaping(
            head_food_dist_f, self._prev_head_food_distance, self.food_approach_weight, 3.0
        )
        self._prev_head_food_distance = head_food_dist_f
        info["approach_delta"] = approach_delta
        info["reward_approach"] = reward_approach

        # 17. Head-to-food proximity shaping (continuous distance reward)
        head_proximity_max_dist = 5.0  # Larger range than T-Rex due to long neck
        head_proximity = max(0.0, 1.0 - head_food_dist_f / head_proximity_max_dist)
        reward_head_proximity = self.food_head_proximity_weight * head_proximity
        info["head_proximity"] = head_proximity
        info["head_food_proximity_distance"] = head_food_dist_f
        info["reward_head_proximity"] = reward_head_proximity

        # Total reward
        total_reward = (
            reward_forward
            + reward_backward
            + reward_drift
            + reward_alive
            + reward_energy
            + reward_gait_stability
            + reward_tail
            + reward_posture
            + reward_nosedive
            + reward_height
            + reward_gait_sym
            + reward_smoothness
            + reward_heading
            + reward_lateral
            + reward_spin
            + reward_speed
            + reward_idle
            + reward_food
            + reward_approach
            + reward_head_proximity
        )
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        """Check if episode should terminate."""
        info = {}

        torso_z = self.data.xpos[self.torso_id, 2]
        info["torso_height"] = torso_z

        torso_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        tilt_angle = self._quat_to_tilt(torso_quat)
        info["tilt_angle"] = tilt_angle

        # Height/tilt termination (shared)
        terminated, reason = self._check_height_tilt_termination(torso_z, tilt_angle)
        if terminated:
            info["termination_reason"] = reason
            return True, info

        # Success: head reached food
        head_tip_pos = self.data.site_xpos[self.head_tip_site_id]
        food_pos = self.data.mocap_pos[0]
        head_food_dist = float(np.linalg.norm(head_tip_pos - food_pos))
        if head_food_dist < self.food_reach_threshold:
            info["termination_reason"] = "food_reached"
            info["success"] = True
            return True, info

        # Floor contact termination (shared) — categorized by body part
        terminated, reason = self._check_floor_contact(
            self._body_ground_geoms,
            self.floor_geom_id,
            geom_categories={
                "head": self._head_ground_geoms,
                "tail": self._tail_ground_geoms,
                "torso": self._torso_ground_geoms,
            },
        )
        if terminated:
            info["termination_reason"] = reason
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

        # Cache initial direction and position for fixed-reference rewards
        self._initial_food_dir_2d = self._compute_initial_direction_2d(food_pos)
        self._initial_pos_2d = self.data.qpos[0:2].copy()

        # Reset delta-based tracking (first step will produce zero deltas)
        self._prev_head_food_distance = None
        self._prev_action = None

        # Reset gait symmetry tracking (quadrupedal diagonal pairs)
        self._reset_quadruped_gait_state()


# Register with Gymnasium (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/Brachio-v0",
    entry_point="environments.brachiosaurus.envs.brachio_env:BrachioEnv",
    max_episode_steps=1000,
)
