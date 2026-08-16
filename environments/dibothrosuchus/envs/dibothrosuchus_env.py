"""
Dibothrosuchus elaphros Gymnasium Environment

A small, gracile sphenosuchian crocodylomorph with erect (parasagittal) limbs
and hindlimbs longer than its forelimbs.  The Stage 3 task uses contact between
a fixed snout-tip geom and small prey as a "snap" proxy; the model has no
articulated jaw.

Observation space (total dimension is generated in the public species catalog):
    - Joint positions (qpos[7:]) — 28 hinge
    - Joint velocities (qvel[6:]) — 28 hinge
    - Trunk orientation (quaternion) — 4
    - Trunk angular velocity (gyroscope) — 3
    - Trunk linear velocity — 3
    - Trunk acceleration — 3
    - Foot contact forces (4 pad touch sensors) — 4
    - Prey direction (unit vector) — 3
    - Prey distance (scalar) — 1

Action space (total dimension is generated in the public species catalog):
    - Neck/skull: neck pitch, neck yaw, head pitch (3)
    - Front right / front left leg: hip pitch/roll, knee, ankle, toe (5 each)
    - Rear right / rear left leg: hip pitch/roll, knee, ankle, toe (5 each)
    - Tail: pitch 1, yaw 1, pitch 2, pitch 3 (4)

Reward components:
    - Forward velocity (toward prey)
    - Backward velocity penalty
    - Drift penalty (horizontal displacement from spawn)
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Gait stability (penalize trunk angular velocity)
    - Tail stability
    - Posture (continuous tilt penalty)
    - Nosedive penalty
    - Height maintenance
    - Gait symmetry (alternating diagonal foot pairs)
    - Action smoothness (penalize jerky action changes)
    - Heading alignment (facing toward prey)
    - Lateral velocity penalty (anti crab-walk)
    - Spin penalty (penalize trunk angular velocity)
    - Speed penalty (penalise absolute speed above threshold)
    - Idle penalty (penalise standing still)
    - Snap-proxy bonus (fixed snout geom contacts prey)
    - Approach shaping (snout-to-prey distance)
    - Snout proximity shaping (reward aiming the snout at the prey)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv


class DibothrosuchusEnv(BaseDinoEnv):
    """Dibothrosuchus quadrupedal locomotion and snout-snap environment."""

    action_mapping = "home-keyframe-residual/v1"
    _camera_distance = 1.6
    _camera_azimuth = 135
    _camera_elevation = -15
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
        tail_stability_weight: float = 0.05,
        smoothness_weight: float = 0.05,
        posture_weight: float = 0.2,
        nosedive_weight: float = 0.0,
        natural_pitch: float = 0.0,
        height_weight: float = 0.0,
        heading_weight: float = 0.0,
        lateral_penalty_weight: float = 0.0,
        backward_vel_penalty_weight: float = 0.0,
        drift_penalty_weight: float = 0.0,
        spin_penalty_weight: float = 0.0,
        speed_penalty_weight: float = 0.0,
        speed_penalty_threshold: float = 0.10,
        idle_penalty_weight: float = 0.0,
        idle_velocity_threshold: float = 0.05,
        forward_vel_max: float = 3.0,
        snap_bonus: float = 10.0,
        snap_approach_weight: float = 1.0,
        snap_snout_proximity_weight: float = 0.0,
        # JAX-only: the MJX reward path gates its alive bonus on foot contact
        # and pays a per-step grounded bonus.  The Gymnasium path gets the same
        # effect from snout-contact and body-contact termination, so these are
        # accepted (the stage TOMLs are shared between backends) and stored,
        # but not read by _get_reward_info.
        foot_contact_weight: float = 0.0,
        foot_contact_gate: float = 0.0,
        nosedive_termination_threshold: float = 0.55,
        # Environment settings
        prey_distance_range: tuple[float, float] = (1.5, 4.0),
        prey_lateral_range: tuple[float, float] = (-1.0, 1.0),
        healthy_z_range: tuple[float, float] = (0.18, 0.55),
        max_tilt_angle: float = 0.9,
        reset_noise_scale: float = 0.01,
        # STATE-INERT: the ground settle overwrites root height from the
        # sampled joint pose, so this scale no longer reaches the post-reset
        # state.  Kept (with its historical 0.03) only so the reset's RNG
        # draw count is unchanged and seeded baselines stay anchored; remove
        # with the rest of the height channel at the next policy-interface
        # revision.  It was originally the metre-scale decoupling of root
        # height from the radian-scale joint jitter.
        reset_height_noise_scale: float | None = 0.03,
        perturbation_capture_velocity_multiple: float = 0.0,
        perturbation_interval: float = 2.0,
        perturbation_jitter: float = 0.5,
        perturbation_duration: float = 0.20,
        perturbation_direction: str = "uniform_horizontal",
    ):
        model_path = str(Path(__file__).parent.parent / "assets" / "dibothrosuchus.xml")

        # Dibothrosuchus-specific reward weights
        self.gait_stability_weight = gait_stability_weight
        self.gait_symmetry_weight = gait_symmetry_weight
        self.tail_stability_weight = tail_stability_weight
        self.smoothness_weight = smoothness_weight
        self.posture_weight = posture_weight
        self.nosedive_weight = nosedive_weight
        self.height_weight = height_weight
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
        self.snap_bonus = snap_bonus
        self.snap_approach_weight = snap_approach_weight
        self.snap_snout_proximity_weight = snap_snout_proximity_weight
        self.foot_contact_weight = foot_contact_weight
        self.foot_contact_gate = foot_contact_gate
        self.nosedive_termination_threshold = nosedive_termination_threshold

        # Natural forward pitch.  Measured: holding the home control vector for
        # 1500 steps leaves the trunk frame at forward_z +0.003, i.e. level, so
        # the default of 0.0 is the plant's real neutral rather than a guess.
        # Both the nosedive penalty and the posture reward stay centred on
        # world vertical; the MJX path matches by leaving
        # posture_target_forward_z unset for this species.
        self._natural_forward_z = -np.sin(natural_pitch)

        # Dibothrosuchus-specific env settings
        self.prey_distance_range = prey_distance_range
        self.prey_lateral_range = prey_lateral_range

        # State tracking for delta-based rewards
        self._prev_snout_prey_distance: float | None = None
        self._prev_action: np.ndarray | None = None

        # Gait symmetry: quadrupedal diagonal pair alternation.
        # Diagonal-A = FR+RL, Diagonal-B = FL+RR.
        self._init_quadruped_gait_state()

        # Cached initial direction to prey (set in _spawn_target).  Used by the
        # forward-velocity and heading rewards so the "forward" reference stays
        # fixed for the whole episode and does not flip sign if the animal
        # overshoots the prey.
        self._initial_prey_dir_2d: np.ndarray = np.array([1.0, 0.0])

        # Cached initial trunk position (set in _spawn_target), used by the
        # drift penalty.
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
            max_tilt_angle=max_tilt_angle,
            reset_noise_scale=reset_noise_scale,
            reset_height_noise_scale=reset_height_noise_scale,
            perturbation_capture_velocity_multiple=perturbation_capture_velocity_multiple,
            perturbation_interval=perturbation_interval,
            perturbation_jitter=perturbation_jitter,
            perturbation_duration=perturbation_duration,
            perturbation_direction=perturbation_direction,
        )

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""
        # Policies command residuals around the complete XML home control
        # vector, so Gymnasium reset and action zero share one nominal state.
        self.home_keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.home_keyframe_id < 0:
            raise ValueError("Dibothrosuchus model must define a named 'home' keyframe")
        self._reset_keyframe_id = self.home_keyframe_id
        self._home_ctrl = self.model.key_ctrl[self.home_keyframe_id].copy()

        # Body IDs
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.skull_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "skull")

        # Geom IDs for contact detection
        self.prey_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")
        self.snout_snap_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "snout_snap")
        self.torso_main_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_main")
        self.belly_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "belly")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Skull geoms that can collide with the floor.  neck_geom and
        # sagittal_crest have contype=0, so they never produce floor contacts.
        self.skull_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "skull_geom")
        self.snout_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "snout_geom")

        # Distal tail geoms that should not contact the floor.  The croc high
        # walk carries the whole tail clear of the ground.
        self.tail_3_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_3_geom")
        self.tail_4_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_4_geom")

        # Geoms that should terminate the episode on ground contact
        self._body_ground_geoms = {
            self.torso_main_geom_id,
            self.belly_geom_id,
            self.skull_geom_id,
            self.snout_geom_id,
            self.tail_3_geom_id,
            self.tail_4_geom_id,
        }
        self._head_ground_geoms = {self.skull_geom_id, self.snout_geom_id}
        self._tail_ground_geoms = {self.tail_3_geom_id, self.tail_4_geom_id}
        self._torso_ground_geoms = {self.torso_main_geom_id, self.belly_geom_id}

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.snout_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "snout_tip")
        self.tail_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")
        # Snout-height termination sampled at every physics substep by the
        # base step loop; _is_terminated reads the substep MIN.
        self._substep_height_checks = (("site", self.snout_tip_site_id),)

        # Foot site IDs
        self.foot_site_ids = {
            "fr": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fr_foot_contact"),
            "fl": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot_contact"),
            "rr": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rr_foot_contact"),
            "rl": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rl_foot_contact"),
        }

        # Prey mocap body
        self.prey_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prey")

        # Sensor indices (order matches the MJCF sensor definition)
        # torso_gyro(3), torso_accel(3), torso_orientation(4),
        # fr/fl/rr/rl_foot_touch(1 each)
        # _sensor_gyro_start, _sensor_accel_start, _sensor_quat_start
        # are inherited from BaseDinoEnv (0, 3, 6 respectively).
        self._sensor_fr_foot = 10
        self._sensor_fl_foot = 11
        self._sensor_rr_foot = 12
        self._sensor_rl_foot = 13
        # Per-foot groups for the base class's substep MIN aggregation --
        # mirrors mjx_config's sensor_foot_indices (no aux sensors here).
        self._foot_sensor_groups = (
            (self._sensor_fr_foot,),
            (self._sensor_fl_foot,),
            (self._sensor_rr_foot,),
            (self._sensor_rl_foot,),
        )

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized residual actions around the XML home controls.

        The two halves are scaled independently so the mapping preserves the
        actuator endpoints while making action zero exactly the named home
        control, even when that control is not the range midpoint.
        """
        residual = np.clip(action, -1.0, 1.0)
        ctrl_range = self.model.actuator_ctrlrange
        ctrl_min = ctrl_range[:, 0]
        ctrl_max = ctrl_range[:, 1]

        below_home = residual * (self._home_ctrl - ctrl_min)
        above_home = residual * (ctrl_max - self._home_ctrl)
        scaled = self._home_ctrl + np.where(residual < 0.0, below_home, above_home)
        return np.asarray(scaled)

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Joint positions (exclude root freejoint: first 7 values)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values)
        qvel = self.data.qvel[6:].copy()

        # Trunk state from sensors
        torso_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3].copy()
        torso_accel = self.data.sensordata[self._sensor_accel_start : self._sensor_accel_start + 3].copy()
        torso_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4].copy()

        # Trunk linear velocity (from root freejoint)
        torso_linvel = self.data.qvel[0:3].copy()

        # Foot contact (from touch sensors)
        foot_contact = np.array(
            [
                self.data.sensordata[self._sensor_fr_foot],
                self.data.sensordata[self._sensor_fl_foot],
                self.data.sensordata[self._sensor_rr_foot],
                self.data.sensordata[self._sensor_rl_foot],
            ]
        )

        # Prey info (relative to trunk)
        torso_pos = self.data.xpos[self.torso_id]
        prey_pos = self.data.mocap_pos[0]
        prey_rel = prey_pos - torso_pos
        prey_distance = np.linalg.norm(prey_rel)

        # Normalize prey direction
        prey_direction = prey_rel / (prey_distance + 1e-8)

        obs = np.concatenate(
            [
                qpos,  # Joint positions
                qvel,  # Joint velocities
                torso_quat,  # Orientation (quaternion)
                torso_gyro,  # Angular velocity
                torso_linvel,  # Linear velocity
                torso_accel,  # Accelerometer
                foot_contact,  # Foot contacts (4)
                prey_direction,  # Direction to prey (unit vector)
                [prey_distance],  # Distance to prey (scalar)
            ]
        ).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        torso_pos = self.data.xpos[self.torso_id]
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

        # 4. Gait stability (penalize high trunk angular velocity)
        reward_gait_stability, gait_instability = self._compute_angular_velocity_penalty(
            self.gait_stability_weight, max_angvel=5.0
        )
        info["gait_instability"] = gait_instability
        info["reward_gait"] = reward_gait_stability

        # 5. Tail stability
        reward_tail, tail_instability = self._compute_tail_stability(self.tail_tip_site_id, self.tail_stability_weight)
        info["tail_instability"] = tail_instability
        info["reward_tail"] = reward_tail

        # 6. Snap bonus (check snout-prey contact)
        snap_reward = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if (geom1 == self.snout_snap_geom_id and geom2 == self.prey_geom_id) or (
                geom2 == self.snout_snap_geom_id and geom1 == self.prey_geom_id
            ):
                snap_reward = self.snap_bonus
                info["snap_success"] = 1.0
                break
        else:
            info["snap_success"] = 0.0

        reward_snap = snap_reward
        info["reward_snap"] = reward_snap

        # 7. Approach shaping.  Measured from the SNOUT rather than the trunk:
        # the snout leads the strike on a 0.16m skull carried on a mobile neck,
        # so trunk distance would keep rewarding approach after the snout has
        # already overshot.
        snout_tip_pos = self.data.site_xpos[self.snout_tip_site_id]
        snout_prey_dist = float(np.linalg.norm(prey_pos - snout_tip_pos))
        reward_approach, approach_delta = self._compute_approach_shaping(
            snout_prey_dist, self._prev_snout_prey_distance, self.snap_approach_weight, 3.0
        )
        self._prev_snout_prey_distance = snout_prey_dist
        info["snout_prey_distance"] = snout_prey_dist
        info["prey_distance"] = snout_prey_dist  # alias for LocomotionMetrics
        info["approach_delta"] = approach_delta
        info["reward_approach"] = reward_approach

        # 7b. Snout-to-prey proximity shaping (continuous aiming gradient)
        snout_proximity_max_dist = 1.5
        snout_proximity = max(0.0, 1.0 - snout_prey_dist / snout_proximity_max_dist)
        reward_snout_proximity = self.snap_snout_proximity_weight * snout_proximity
        info["snout_proximity"] = snout_proximity
        info["reward_snout_proximity"] = reward_snout_proximity

        # 8. Continuous posture reward
        torso_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        reward_posture, tilt_angle = self._compute_posture_reward(torso_quat, self.posture_weight)
        info["tilt_angle"] = tilt_angle
        info["reward_posture"] = reward_posture

        # 9. Nosedive penalty
        reward_nosedive, forward_z = self._compute_nosedive_penalty(
            torso_quat, self.nosedive_weight, self._natural_forward_z
        )
        info["forward_z"] = forward_z
        info["reward_nosedive"] = reward_nosedive

        # 10. Trunk height (for LocomotionMetrics tracking) and maintenance reward
        torso_height = float(torso_pos[2])
        info["torso_height"] = torso_height
        info["pelvis_height"] = torso_height  # alias for LocomotionMetrics
        min_z = self.healthy_z_range[0]
        target_z = 0.3129  # settled trunk height under the home controller
        height_frac = float(np.clip((torso_height - min_z) / (target_z - min_z), 0.0, 1.0))
        reward_height = self.height_weight * height_frac
        info["reward_height"] = reward_height

        # 11. Gait symmetry (reward alternating diagonal pair touchdowns).
        # Substep-MIN aggregated -- see BaseDinoEnv._aggregated_foot_contact_forces.
        fr_contact, fl_contact, rr_contact, rl_contact = self._aggregated_foot_contact_forces()
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
        info["contact_asymmetry"] = alternation_ratio  # backward compat with metrics
        info["reward_gait_symmetry"] = reward_gait_sym

        # 12. Action smoothness (shared helper)
        reward_smoothness, action_delta = self._reward_action_smoothness(action)
        info["action_delta"] = action_delta
        info["reward_smoothness"] = reward_smoothness

        # 13. Heading alignment
        body_forward_2d = self._quat_to_forward_2d(torso_quat)
        reward_heading, heading_alignment = self._compute_heading_alignment(
            body_forward_2d, forward_ref_2d, self.heading_weight
        )
        info["heading_alignment"] = heading_alignment
        info["reward_heading"] = reward_heading

        # 14. Lateral velocity penalty
        reward_lateral, lateral_vel = self._compute_lateral_velocity_penalty(
            vel_2d, body_forward_2d, self.lateral_penalty_weight
        )
        info["lateral_vel"] = lateral_vel
        info["reward_lateral"] = reward_lateral

        # Trunk angular velocity (for spinning detection in shared diagnostics)
        pelvis_angular_vel, pelvis_yaw_vel = self._compute_pelvis_diagnostics()
        info["pelvis_angular_vel"] = pelvis_angular_vel
        info["pelvis_yaw_vel"] = pelvis_yaw_vel

        # 15. Spin penalty
        reward_spin, spin_instability = self._compute_angular_velocity_penalty(self.spin_penalty_weight)
        info["spin_instability"] = spin_instability
        info["reward_spin"] = reward_spin

        # 16. Speed penalty (penalise absolute speed above threshold)
        reward_speed, abs_speed = self._compute_speed_penalty(
            vel_2d, self.speed_penalty_weight, self.speed_penalty_threshold
        )
        info["abs_speed"] = abs_speed
        info["reward_speed"] = reward_speed

        # 16b. Idle penalty (penalise standing still / barely moving)
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
            + reward_gait_stability
            + reward_tail
            + reward_snap
            + reward_approach
            + reward_snout_proximity
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

        # Height/tilt termination (shared).  The lower bound is the sprawl
        # gate: a modern-crocodilian belly-down sprawl drops the trunk well
        # below the erect stance this species is being trained to hold.
        terminated, reason = self._check_height_tilt_termination(torso_z, tilt_angle)
        if terminated:
            info["termination_reason"] = reason
            return True, info

        # Nosedive termination
        forward_z = self._quat_to_forward_z(torso_quat)
        info["forward_z"] = forward_z
        if forward_z < self._natural_forward_z - self.nosedive_termination_threshold:
            info["termination_reason"] = "nosedive"
            return True, info

        # Site-height termination: the snout tip must stay off the ground.
        # Catches snout-propping that geom contact detection may miss.  Reads
        # the substep MIN so a dip that recovers between control-boundary
        # samples still terminates; the info key keeps the boundary sample.
        snout_tip_z = self.data.site_xpos[self.snout_tip_site_id, 2]
        info["snout_tip_z"] = snout_tip_z
        if self._aggregated_min_height(0, snout_tip_z) < 0.04:
            info["termination_reason"] = "head_contact"
            return True, info

        # Success: snout snap geom contacted the prey.  Gated on snap_bonus so
        # stages 1-2 don't end episodes on incidental contact.
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if self.snap_bonus > 0 and (
                (geom1 == self.snout_snap_geom_id and geom2 == self.prey_geom_id)
                or (geom2 == self.snout_snap_geom_id and geom1 == self.prey_geom_id)
            ):
                info["termination_reason"] = "snap_success"
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
        """Spawn prey at a random location ahead of the animal."""
        # 0.15m: the prey sphere's underside then sits at 0.09m, clear of the
        # 0.04m snout-prop termination gate, and the snout reaches 0.08m at
        # full nose-down, so the snap is inside the neck/skull work envelope.
        prey_pos = self._spawn_target_2d(self.prey_distance_range, self.prey_lateral_range, 0.15)
        self._initial_prey_dir_2d = self._compute_initial_direction_2d(prey_pos)
        self._initial_pos_2d = self.data.qpos[0:2].copy()

        # Reset delta-based tracking (first step will produce zero deltas)
        self._prev_snout_prey_distance = None
        self._prev_action = None

        # Reset gait symmetry tracking (quadrupedal diagonal pairs)
        self._reset_quadruped_gait_state()


# Register with Gymnasium (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/Dibothrosuchus-v0",
    entry_point="environments.dibothrosuchus.envs.dibothrosuchus_env:DibothrosuchusEnv",
    max_episode_steps=1000,
)
