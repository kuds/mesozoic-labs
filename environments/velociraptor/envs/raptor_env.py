"""
Velociraptor Gymnasium Environment

A bipedal dinosaur locomotion environment with predatory strike behavior.

Observation space (67 dims):
    - Joint positions (qpos[7:]) — 24 hinge joints excluding root freejoint
    - Joint velocities (qvel[6:]) — 24 hinge joints excluding root freejoint
    - Pelvis orientation (quaternion) — 4
    - Pelvis angular velocity (gyroscope) — 3
    - Pelvis linear velocity — 3
    - Pelvis acceleration — 3
    - Foot contact states — 2
    - Prey direction (unit vector) — 3
    - Prey distance (scalar) — 1

Action space (22 dims):
    - Right leg: hip pitch/roll, knee, ankle, toe d3/d4 (6)
    - Right sickle claw (1)
    - Left leg: hip pitch/roll, knee, ankle, toe d3/d4 (6)
    - Left sickle claw (1)
    - Tail: pitch 1, yaw 1, pitch 2, pitch 3 (4)
    - Right arm: shoulder pitch/roll (2)
    - Left arm: shoulder pitch/roll (2)

Reward components:
    - Forward velocity
    - Backward velocity penalty
    - Drift penalty (horizontal displacement from spawn)
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Strike bonus (when claw contacts prey)
    - Approach shaping (distance to prey)
    - Proximity bonus (continuous reward for being close to prey)
    - Claw proximity shaping (reward for positioning claw tip near prey)
    - Posture (continuous tilt penalty)
    - Nosedive penalty
    - Gait symmetry (alternating foot contacts)
    - Action smoothness (penalize jerky action changes)
    - Spin penalty (penalize pelvis angular velocity)
    - Heading alignment (facing toward prey)
    - Lateral velocity penalty (anti crab-walk)
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
        forward_vel_max: float = 10.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        tail_stability_weight: float = 0.05,
        strike_bonus: float = 10.0,
        strike_approach_weight: float = 1.0,
        strike_proximity_weight: float = 0.0,
        strike_claw_proximity_weight: float = 0.0,
        posture_weight: float = 0.2,
        nosedive_weight: float = 0.0,
        natural_pitch: float = 0.35,
        gait_symmetry_weight: float = 0.0,
        smoothness_weight: float = 0.05,
        heading_weight: float = 0.0,
        lateral_penalty_weight: float = 0.0,
        backward_vel_penalty_weight: float = 0.0,
        drift_penalty_weight: float = 0.0,
        spin_penalty_weight: float = 0.0,
        # Environment settings
        prey_distance_range: Tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: Tuple[float, float] = (-2.0, 2.0),
        healthy_z_range: Tuple[float, float] = (0.3, 1.0),
        reset_noise_scale: float = 0.01,
    ):
        model_path = str(Path(__file__).parent.parent / "assets" / "raptor.xml")

        # Raptor-specific reward weights
        self.forward_vel_max = forward_vel_max
        self.tail_stability_weight = tail_stability_weight
        self.strike_bonus = strike_bonus
        self.strike_approach_weight = strike_approach_weight
        self.strike_proximity_weight = strike_proximity_weight
        self.strike_claw_proximity_weight = strike_claw_proximity_weight
        self.posture_weight = posture_weight
        self.nosedive_weight = nosedive_weight
        self.gait_symmetry_weight = gait_symmetry_weight
        self.smoothness_weight = smoothness_weight
        self.heading_weight = heading_weight
        self.lateral_penalty_weight = lateral_penalty_weight
        self.backward_vel_penalty_weight = backward_vel_penalty_weight
        self.drift_penalty_weight = drift_penalty_weight
        self.spin_penalty_weight = spin_penalty_weight

        # Natural forward pitch (~20°). The nosedive penalty and termination
        # are measured relative to this angle so the raptor isn't punished for
        # its biomechanically correct forward lean.
        self._natural_forward_z = -np.sin(natural_pitch)

        # Raptor-specific env settings
        self.prey_distance_range = prey_distance_range
        self.prey_lateral_range = prey_lateral_range

        # State tracking for delta-based rewards
        self._prev_prey_distance: float | None = None
        self._prev_action: np.ndarray | None = None

        # Gait symmetry: track foot touchdown events for alternation reward
        self._contact_threshold = 0.1  # Newtons, matches metrics.py onset detection
        self._prev_r_in_contact = False
        self._prev_l_in_contact = False
        self._touchdown_sequence: list = []  # Recent 'L'/'R' touchdown events
        self._max_touchdown_history = 20

        # Cached initial direction to prey (set in _spawn_target).
        # Used by forward-velocity and heading rewards so the "forward"
        # reference direction stays fixed for the whole episode, preventing
        # the reward from flipping sign when the raptor passes the prey.
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
        self.r_claw_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_claw_geom")
        self.l_claw_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_claw_geom")
        self.torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso")
        self.neck_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "neck")
        self.head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Tail geom IDs (distal segments that should not contact floor)
        self.tail_3_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_3_geom")
        self.tail_4_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_4_geom")
        self.tail_5_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_5_geom")

        # Geoms that should terminate the episode on ground contact
        self._body_ground_geoms = {
            self.torso_geom_id,
            self.neck_geom_id,
            self.head_geom_id,
            self.tail_3_geom_id,
            self.tail_4_geom_id,
            self.tail_5_geom_id,
        }
        self._tail_ground_geoms = {self.tail_3_geom_id, self.tail_4_geom_id, self.tail_5_geom_id}

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.r_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_foot")
        self.l_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_foot")
        self.tail_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")

        # Mocap body for prey
        self.prey_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prey")

        # Claw tip site IDs (for claw-to-prey proximity shaping)
        self.r_claw_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_claw_tip")
        self.l_claw_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_claw_tip")

        # Sensor indices (order matches MJCF definition)
        # _sensor_gyro_start, _sensor_accel_start, _sensor_quat_start
        # are inherited from BaseDinoEnv (0, 3, 6 respectively).
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

        # 1. Forward velocity reward (toward prey)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]

        # Use the cached initial direction so the "forward" reference stays
        # fixed for the whole episode.  This prevents the reward from flipping
        # sign when the raptor overshoots and passes the prey.
        forward_ref_2d = self._initial_prey_dir_2d

        # Keep the live direction available for rewards that genuinely need
        # the current relative vector (approach shaping, observations).
        prey_dir_2d = prey_pos[:2] - pelvis_pos[:2]
        prey_dist_2d = np.linalg.norm(prey_dir_2d)
        if prey_dist_2d > 1e-6:
            prey_dir_2d = prey_dir_2d / prey_dist_2d

        vel_2d = self.data.qvel[0:2]
        forward_vel = np.dot(vel_2d, forward_ref_2d)
        forward_vel_norm = np.clip(forward_vel / self.forward_vel_max, -1.0, 1.0)
        info["forward_vel"] = forward_vel
        reward_forward = self.forward_vel_weight * forward_vel_norm
        info["reward_forward"] = reward_forward

        # 1b. Backward velocity penalty (penalize drifting backward without
        #     rewarding forward movement — useful in balance stages where
        #     forward_vel_weight is zero)
        backward_vel = max(0.0, -forward_vel)
        backward_vel_norm = min(backward_vel / self.forward_vel_max, 1.0)
        reward_backward = -self.backward_vel_penalty_weight * backward_vel_norm
        info["backward_vel"] = backward_vel
        info["reward_backward"] = reward_backward

        # 1c. Drift penalty (penalize horizontal displacement from spawn point;
        #     quadratic so small wobbles are cheap but sustained drift is costly)
        drift_2d = pelvis_pos[:2] - self._initial_pos_2d
        drift_dist = float(np.linalg.norm(drift_2d))
        # Normalize: 1.0 at 2m drift, capped at 1.0
        drift_norm = min(drift_dist / 2.0, 1.0)
        reward_drift = -self.drift_penalty_weight * (drift_norm ** 2)
        info["drift_distance"] = drift_dist
        info["reward_drift"] = reward_drift

        # 2. Alive bonus (shared helper)
        reward_alive = self._reward_alive()
        info["reward_alive"] = reward_alive

        # 3. Energy penalty (shared helper)
        reward_energy = self._reward_energy(action)
        info["reward_energy"] = reward_energy

        # 4. Tail stability (penalize high angular velocity at tail tip)
        tail_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.tail_tip_site_id, tail_vel, 0)
        tail_tip_angvel = tail_vel[0:3]  # Angular velocity (first 3 elements, rot:lin order)
        tail_instability = float(np.linalg.norm(tail_tip_angvel))
        # Normalize assuming max angular vel ~10.0 rad/s
        tail_instability_norm = min(tail_instability / 10.0, 1.0)
        reward_tail = -self.tail_stability_weight * tail_instability_norm
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
        prey_distance = float(np.linalg.norm(prey_pos - pelvis_pos))

        if self._prev_prey_distance is not None:
            approach_delta = self._prev_prey_distance - prey_distance
        else:
            approach_delta = 0.0
        self._prev_prey_distance = prey_distance

        # Max approach speed ~10m/s. dt is frame_skip * model.opt.timestep (typically 5 * 0.002 = 0.01s)
        dt = self.frame_skip * self.model.opt.timestep
        max_delta = 10.0 * dt
        approach_delta_norm = np.clip(approach_delta / max_delta, -1.0, 1.0)

        reward_approach = self.strike_approach_weight * approach_delta_norm
        info["prey_distance"] = prey_distance
        info["approach_delta"] = approach_delta
        info["reward_approach"] = reward_approach

        # 6b. Proximity bonus (continuous reward for being close to prey)
        # Provides a smooth basin of attraction that complements the noisy
        # delta-based approach reward.  Linearly scales from 0 at max spawn
        # distance to 1 at the prey location.
        max_prey_dist = max(self.prey_distance_range[1], 1.0)
        proximity = max(0.0, 1.0 - prey_distance / max_prey_dist)
        reward_proximity = self.strike_proximity_weight * proximity
        info["proximity"] = proximity
        info["reward_proximity"] = reward_proximity

        # 6c. Claw-to-prey proximity shaping
        # Uses the actual claw tip positions (not the pelvis) to give the agent
        # a gradient for positioning its weapon near the prey.  Takes the min
        # distance of the two claw tips so the agent is rewarded for whichever
        # claw is closest.  Activates only when the pelvis is already within
        # max_prey_dist (outer approach is handled by the pelvis-based rewards).
        r_claw_pos = self.data.site_xpos[self.r_claw_tip_site_id]
        l_claw_pos = self.data.site_xpos[self.l_claw_tip_site_id]
        r_claw_dist = float(np.linalg.norm(prey_pos - r_claw_pos))
        l_claw_dist = float(np.linalg.norm(prey_pos - l_claw_pos))
        min_claw_dist = min(r_claw_dist, l_claw_dist)
        # Scale: 1.0 when claw touches prey, 0.0 at claw_proximity_max_dist away.
        # Use a tighter range than the pelvis proximity since this reward is
        # meant to guide the final strike positioning.
        claw_proximity_max_dist = 2.0
        claw_proximity = max(0.0, 1.0 - min_claw_dist / claw_proximity_max_dist)
        reward_claw_proximity = self.strike_claw_proximity_weight * claw_proximity
        info["min_claw_prey_distance"] = min_claw_dist
        info["claw_proximity"] = claw_proximity
        info["reward_claw_proximity"] = reward_claw_proximity

        # 7. Continuous posture reward (quadratic tilt penalty — stronger gradient near upright)
        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        tilt_angle = self._quat_to_tilt(pelvis_quat)
        # Normalize by max_tilt_angle, then square for stronger upright incentive
        tilt_angle_norm = min(tilt_angle / self.max_tilt_angle, 1.0)
        reward_posture = -self.posture_weight * (tilt_angle_norm**2)
        info["tilt_angle"] = tilt_angle
        info["reward_posture"] = reward_posture

        # 8. Nosedive penalty (excessive forward pitch beyond the natural lean)
        w, x, y, z = pelvis_quat
        # Z-component of body's local X-axis (head direction) in world frame
        forward_z = 2.0 * (x * z - w * y)
        # Only penalize pitch beyond the natural forward lean (~20°).
        # _natural_forward_z is negative (≈ -0.34), so we penalize when
        # forward_z drops further below that baseline.
        nosedive_excess = max(0.0, -(forward_z - self._natural_forward_z))
        reward_nosedive = -self.nosedive_weight * nosedive_excess
        info["forward_z"] = forward_z
        info["reward_nosedive"] = reward_nosedive

        # 8b. Pelvis height (for LocomotionMetrics tracking)
        info["pelvis_height"] = float(self.data.xpos[self.pelvis_id, 2])

        # 8c. Pelvis angular velocity (for spinning detection in eval metrics)
        pelvis_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3]
        info["pelvis_angular_vel"] = float(np.linalg.norm(pelvis_gyro))
        info["pelvis_yaw_vel"] = float(pelvis_gyro[2])  # Z-axis rotation = yaw = spinning

        # 8d. Spin penalty (penalize pelvis angular velocity, cf. Brachiosaurus gait_stability)
        pelvis_angvel = self.data.qvel[3:6]
        spin_instability = float(np.linalg.norm(pelvis_angvel))
        spin_instability_norm = min(spin_instability / 10.0, 1.0)
        reward_spin = -self.spin_penalty_weight * spin_instability_norm
        info["spin_instability"] = spin_instability
        info["reward_spin"] = reward_spin

        # 9. Gait symmetry (reward alternating foot contacts)
        # Track touchdown events (off→on transitions) and reward when
        # consecutive touchdowns alternate feet: L→R→L = 1.0, L→L→R = 0.5.
        r_contact = self.data.sensordata[self._sensor_r_foot]
        l_contact = self.data.sensordata[self._sensor_l_foot]
        info["r_foot_contact"] = float(r_contact)
        info["l_foot_contact"] = float(l_contact)

        r_in_contact = r_contact > self._contact_threshold
        l_in_contact = l_contact > self._contact_threshold
        r_touchdown = r_in_contact and not self._prev_r_in_contact
        l_touchdown = l_in_contact and not self._prev_l_in_contact
        self._prev_r_in_contact = r_in_contact
        self._prev_l_in_contact = l_in_contact

        if r_touchdown:
            self._touchdown_sequence.append("R")
        if l_touchdown:
            self._touchdown_sequence.append("L")
        if len(self._touchdown_sequence) > self._max_touchdown_history:
            self._touchdown_sequence = self._touchdown_sequence[-self._max_touchdown_history :]

        n_touchdowns = len(self._touchdown_sequence)
        if n_touchdowns > 1:
            alternations = sum(
                1 for i in range(1, n_touchdowns) if self._touchdown_sequence[i] != self._touchdown_sequence[i - 1]
            )
            alternation_ratio = alternations / (n_touchdowns - 1)
        else:
            alternation_ratio = 0.0

        reward_gait = self.gait_symmetry_weight * alternation_ratio
        info["alternation_ratio"] = alternation_ratio
        info["contact_asymmetry"] = alternation_ratio  # backward compat with metrics
        info["reward_gait"] = reward_gait

        # 10. Action smoothness (shared helper)
        reward_smoothness, action_delta = self._reward_action_smoothness(action)
        info["action_delta"] = action_delta
        info["reward_smoothness"] = reward_smoothness

        # 11. Heading alignment (reward facing toward prey)
        # Extract body forward direction (+X local axis) projected into world XY
        pelvis_quat_h = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        wh, xh, yh, zh = pelvis_quat_h
        body_forward_x = 1.0 - 2.0 * (yh * yh + zh * zh)
        body_forward_y = 2.0 * (xh * yh + wh * zh)
        body_forward_2d = np.array([body_forward_x, body_forward_y])
        body_forward_len = np.linalg.norm(body_forward_2d)
        if body_forward_len > 1e-6:
            body_forward_2d = body_forward_2d / body_forward_len
        # Dot product: +1 when facing the initial prey direction, -1 when facing away.
        # Uses the cached direction so heading reward doesn't flip after overshoot.
        heading_alignment = float(np.dot(body_forward_2d, forward_ref_2d))
        reward_heading = self.heading_weight * heading_alignment
        info["heading_alignment"] = heading_alignment
        info["reward_heading"] = reward_heading

        # 12. Lateral velocity penalty (penalize crab-walking)
        # Component of velocity perpendicular to body's facing direction
        lateral_vel = abs(vel_2d[0] * body_forward_2d[1] - vel_2d[1] * body_forward_2d[0])
        lateral_vel_norm = float(np.clip(lateral_vel / 5.0, 0.0, 1.0))
        reward_lateral = -self.lateral_penalty_weight * lateral_vel_norm
        info["lateral_vel"] = float(lateral_vel)
        info["reward_lateral"] = reward_lateral

        # Total reward
        total_reward = (
            reward_forward
            + reward_backward
            + reward_drift
            + reward_alive
            + reward_energy
            + reward_tail
            + reward_strike
            + reward_approach
            + reward_proximity
            + reward_claw_proximity
            + reward_posture
            + reward_nosedive
            + reward_spin
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

        # Termination: pelvis too high (shouldn't happen, but safety check)
        if pelvis_z > self.healthy_z_range[1]:
            info["termination_reason"] = "too_high"
            return True, info

        # Termination: excessive tilt (about to fall)
        if tilt_angle > self.max_tilt_angle:
            info["termination_reason"] = "excessive_tilt"
            return True, info

        # Termination: nosedive (forward pitch exceeds natural lean by > 30°)
        # _natural_forward_z ≈ -0.34 for 20° lean; subtracting 0.5 gives ≈ -0.84,
        # which corresponds to ~30° of additional pitch beyond the natural pose.
        w, x, y, z = pelvis_quat
        forward_z = 2.0 * (x * z - w * y)
        info["forward_z"] = forward_z
        if forward_z < self._natural_forward_z - 0.5:
            info["termination_reason"] = "nosedive"
            return True, info

        # Check contacts: body-ground (failure) and claw-prey (success)
        claw_geoms = {self.r_claw_geom_id, self.l_claw_geom_id}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Success: sickle claw contacted prey (only terminate when striking is rewarded)
            if self.strike_bonus > 0 and (
                (geom1 in claw_geoms and geom2 == self.prey_geom_id)
                or (geom2 in claw_geoms and geom1 == self.prey_geom_id)
            ):
                info["termination_reason"] = "strike_success"
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
                else:
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

        # Cache initial 2D direction to prey for stable reward computation.
        # The raptor starts near the origin, so prey_pos[:2] ≈ the direction.
        dir_2d = prey_pos[:2].copy()
        dir_len = np.linalg.norm(dir_2d)
        if dir_len > 1e-6:
            dir_2d /= dir_len
        self._initial_prey_dir_2d = dir_2d

        # Cache initial pelvis position for drift penalty (qpos[0:2] is the
        # root freejoint x,y — valid here before mj_forward).
        self._initial_pos_2d = self.data.qpos[0:2].copy()

        # Reset delta-based tracking (first step will produce zero deltas)
        self._prev_prey_distance = None
        self._prev_action = None

        # Reset gait symmetry tracking
        self._prev_r_in_contact = False
        self._prev_l_in_contact = False
        self._touchdown_sequence = []


# Register with Gymnasium (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/Raptor-v0",
    entry_point="environments.velociraptor.envs.raptor_env:RaptorEnv",
    max_episode_steps=1000,
)
