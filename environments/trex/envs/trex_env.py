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
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Bite bonus (head contacts prey)
    - Approach shaping (distance to prey)
    - Posture (continuous tilt penalty)
    - Nosedive penalty
    - Height maintenance
    - Gait symmetry (alternating foot contacts)
    - Action smoothness (penalize jerky action changes)
    - Heading alignment (facing toward prey)
    - Lateral velocity penalty (anti crab-walk)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        render_mode: Optional[str] = None,
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
        posture_weight: float = 0.2,
        nosedive_weight: float = 0.0,
        natural_pitch: float = 0.17,
        height_weight: float = 0.0,
        gait_symmetry_weight: float = 0.0,
        smoothness_weight: float = 0.05,
        heading_weight: float = 0.0,
        lateral_penalty_weight: float = 0.0,
        forward_vel_max: float = 8.0,
        # Environment settings
        prey_distance_range: Tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: Tuple[float, float] = (-2.0, 2.0),
        healthy_z_range: Tuple[float, float] = (0.5, 1.6),
    ):
        model_path = str(Path(__file__).parent.parent / "assets" / "trex.xml")

        # T-Rex-specific reward weights
        self.tail_stability_weight = tail_stability_weight
        self.bite_bonus = bite_bonus
        self.bite_approach_weight = bite_approach_weight
        self.posture_weight = posture_weight
        self.nosedive_weight = nosedive_weight
        self.height_weight = height_weight
        self.gait_symmetry_weight = gait_symmetry_weight
        self.smoothness_weight = smoothness_weight
        self.heading_weight = heading_weight
        self.lateral_penalty_weight = lateral_penalty_weight
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

    def _get_reward_info(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        # 1. Forward velocity reward (toward prey) — normalized
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        prey_dir_2d = prey_pos[:2] - pelvis_pos[:2]
        prey_dist_2d = np.linalg.norm(prey_dir_2d)
        if prey_dist_2d > 1e-6:
            prey_dir_2d = prey_dir_2d / prey_dist_2d

        vel_2d = self.data.qvel[0:2]
        forward_vel = np.dot(vel_2d, prey_dir_2d)
        forward_vel_norm = np.clip(forward_vel / self.forward_vel_max, -1.0, 1.0)
        info["forward_vel"] = forward_vel
        reward_forward = self.forward_vel_weight * forward_vel_norm
        info["reward_forward"] = reward_forward

        # 2. Alive bonus
        reward_alive = self.alive_bonus
        info["reward_alive"] = reward_alive

        # 3. Energy penalty (normalized by number of actuators)
        energy = np.sum(np.square(action))
        assert self.action_space.shape is not None
        n_actuators = self.action_space.shape[0]
        energy_norm = energy / n_actuators
        reward_energy = -self.energy_penalty_weight * energy_norm
        info["reward_energy"] = reward_energy

        # 4. Tail stability (penalize high angular velocity at tail tip) — normalized
        tail_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.tail_tip_site_id,
            tail_vel,
            0,
        )
        tail_tip_angvel = tail_vel[0:3]  # Angular velocity (first 3 elements, rot:lin order)
        tail_instability = float(np.linalg.norm(tail_tip_angvel))
        # Normalize assuming max angular vel ~10.0 rad/s
        tail_instability_norm = min(tail_instability / 10.0, 1.0)
        reward_tail = -self.tail_stability_weight * tail_instability_norm
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

        # 6. Approach shaping (reward closing distance to prey) — normalized
        prey_distance = float(np.linalg.norm(prey_pos - pelvis_pos))

        if self._prev_prey_distance is not None:
            approach_delta = self._prev_prey_distance - prey_distance
        else:
            approach_delta = 0.0
        self._prev_prey_distance = prey_distance

        # Max approach speed ~8m/s. dt is frame_skip * model.opt.timestep
        dt = self.frame_skip * self.model.opt.timestep
        max_delta = 8.0 * dt
        approach_delta_norm = np.clip(approach_delta / max_delta, -1.0, 1.0)

        reward_approach = self.bite_approach_weight * approach_delta_norm
        info["prey_distance"] = prey_distance
        info["approach_delta"] = approach_delta
        info["reward_approach"] = reward_approach

        # 7. Continuous posture reward (quadratic tilt penalty)
        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        tilt_angle = self._quat_to_tilt(pelvis_quat)
        tilt_angle_norm = min(tilt_angle / self.max_tilt_angle, 1.0)
        reward_posture = -self.posture_weight * (tilt_angle_norm**2)
        info["tilt_angle"] = tilt_angle
        info["reward_posture"] = reward_posture

        # 8. Nosedive penalty (excessive forward pitch beyond natural lean)
        w, x, y, z = pelvis_quat
        # Z-component of body's local X-axis (head direction) in world frame
        forward_z = 2.0 * (x * z - w * y)
        nosedive_excess = max(0.0, -(forward_z - self._natural_forward_z))
        reward_nosedive = -self.nosedive_weight * nosedive_excess
        info["forward_z"] = forward_z
        info["reward_nosedive"] = reward_nosedive

        # 8b. Pelvis height (for LocomotionMetrics tracking)
        pelvis_height = float(self.data.xpos[self.pelvis_id, 2])
        info["pelvis_height"] = pelvis_height

        # 8c. Height maintenance reward (smooth gradient toward staying upright)
        min_z = self.healthy_z_range[0]  # 0.5m (termination threshold)
        target_z = 0.90  # Initial standing height from keyframe
        height_frac = np.clip((pelvis_height - min_z) / (target_z - min_z), 0.0, 1.0)
        reward_height = self.height_weight * height_frac
        info["reward_height"] = reward_height

        # 9. Gait symmetry (reward alternating foot contacts)
        r_contact = self.data.sensordata[self._sensor_r_foot]
        l_contact = self.data.sensordata[self._sensor_l_foot]
        info["r_foot_contact"] = float(r_contact)
        info["l_foot_contact"] = float(l_contact)
        contact_sum = r_contact + l_contact + 1e-6
        contact_asymmetry = abs(r_contact - l_contact) / contact_sum
        reward_gait = self.gait_symmetry_weight * contact_asymmetry
        info["contact_asymmetry"] = contact_asymmetry
        info["reward_gait"] = reward_gait

        # 10. Action smoothness (penalize large action changes between steps)
        if self._prev_action is not None:
            action_delta = float(np.sum(np.square(action - self._prev_action)))
            assert self.action_space.shape is not None
            max_action_delta = self.action_space.shape[0] * 4.0
            action_delta_norm = action_delta / max_action_delta
            reward_smoothness = -self.smoothness_weight * action_delta_norm
        else:
            action_delta = 0.0
            reward_smoothness = 0.0
        self._prev_action = action.copy()
        info["action_delta"] = action_delta
        info["reward_smoothness"] = reward_smoothness

        # 11. Heading alignment (reward facing toward prey)
        pelvis_quat_h = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        wh, xh, yh, zh = pelvis_quat_h
        body_forward_x = 1.0 - 2.0 * (yh * yh + zh * zh)
        body_forward_y = 2.0 * (xh * yh + wh * zh)
        body_forward_2d = np.array([body_forward_x, body_forward_y])
        body_forward_len = np.linalg.norm(body_forward_2d)
        if body_forward_len > 1e-6:
            body_forward_2d = body_forward_2d / body_forward_len
        heading_alignment = float(np.dot(body_forward_2d, prey_dir_2d))
        reward_heading = self.heading_weight * heading_alignment
        info["heading_alignment"] = heading_alignment
        info["reward_heading"] = reward_heading

        # 12. Lateral velocity penalty (penalize crab-walking)
        lateral_vel = abs(vel_2d[0] * body_forward_2d[1] - vel_2d[1] * body_forward_2d[0])
        lateral_vel_norm = float(np.clip(lateral_vel / 5.0, 0.0, 1.0))
        reward_lateral = -self.lateral_penalty_weight * lateral_vel_norm
        info["lateral_vel"] = float(lateral_vel)
        info["reward_lateral"] = reward_lateral

        # Total reward
        total_reward = (
            reward_forward
            + reward_alive
            + reward_energy
            + reward_tail
            + reward_bite
            + reward_approach
            + reward_posture
            + reward_nosedive
            + reward_height
            + reward_gait
            + reward_smoothness
            + reward_heading
            + reward_lateral
        )
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

        # Termination: pelvis too high (safety check)
        if pelvis_z > self.healthy_z_range[1]:
            info["termination_reason"] = "too_high"
            return True, info

        # Termination: excessive tilt (about to fall)
        if tilt_angle > self.max_tilt_angle:
            info["termination_reason"] = "excessive_tilt"
            return True, info

        # Termination: nosedive (forward pitch exceeds natural lean by > 30°)
        w, x, y, z = pelvis_quat
        forward_z = 2.0 * (x * z - w * y)
        info["forward_z"] = forward_z
        if forward_z < self._natural_forward_z - 0.5:
            info["termination_reason"] = "nosedive"
            return True, info

        # Check contacts: body-ground (failure) and head-prey (success)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Success: head_bite contacted prey
            if (geom1 == self.head_bite_geom_id and geom2 == self.prey_geom_id) or (
                geom2 == self.head_bite_geom_id and geom1 == self.prey_geom_id
            ):
                info["termination_reason"] = "bite_success"
                info["success"] = True
                return True, info

            # Failure: body part contacted floor
            floor_contact_geom = None
            if geom2 == self.floor_geom_id and geom1 in self._body_ground_geoms:
                floor_contact_geom = geom1
            elif geom1 == self.floor_geom_id and geom2 in self._body_ground_geoms:
                floor_contact_geom = geom2

            if floor_contact_geom is not None:
                if floor_contact_geom in self._tail_ground_geoms:
                    info["termination_reason"] = "tail_contact"
                elif floor_contact_geom in self._head_ground_geoms:
                    info["termination_reason"] = "head_contact"
                else:
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

        # Reset delta-based tracking (first step will produce zero deltas)
        self._prev_prey_distance = None
        self._prev_action = None


# Register with Gymnasium (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/TRex-v0",
    entry_point="environments.trex.envs.trex_env:TRexEnv",
    max_episode_steps=1000,
)
