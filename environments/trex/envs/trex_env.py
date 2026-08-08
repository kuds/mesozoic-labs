"""
Tyrannosaurus Rex Gymnasium Environment

A large bipedal predator with a massive skull and vestigial forelimbs.
The Stage 3 task uses contact between a fixed head geom and prey as a
"bite" proxy; the model has no articulated jaw.

Observation space (total dimension is generated in the public species catalog):
    - Joint positions (qpos[7:]) — 21 (21 hinge)
    - Joint velocities (qvel[6:]) — 21 (21 hinge)
    - Pelvis orientation (quaternion) — 4
    - Pelvis angular velocity (gyroscope) — 3
    - Pelvis linear velocity — 3
    - Pelvis acceleration — 3
    - Foot contact forces (2 plantar-pad touch sensors) — 2
    - Prey direction (unit vector) — 3
    - Prey distance (scalar) — 1

Action space (total dimension is generated in the public species catalog):
    - Neck/head: neck pitch, neck yaw, head pitch (3)
    - Right leg: hip pitch/roll, knee, ankle (4)
    - Left leg: hip pitch/roll, knee, ankle (4)
    - Tail: pitch 1, yaw 1, pitch 2, pitch 3 (4)

The six toe hinges are passive (spring-loaded at the home pose); they stay
in qpos/qvel so the observation still senses toe posture.

Reward components:
    - Forward velocity (toward prey)
    - Backward velocity penalty
    - Drift penalty (horizontal displacement from spawn)
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Bite-proxy bonus (fixed head geom contacts prey)
    - Approach shaping (distance to prey)
    - Head proximity shaping (reward for positioning head near prey)
    - Posture (continuous tilt penalty)
    - Nosedive penalty
    - Height maintenance
    - Bilateral foot support and load balance
    - Home leg-pose retention
    - Head clearance and neck posture
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
from environments.shared.reward_functions import (
    reward_bilateral_support as _reward_bilateral_support_pure,
)
from environments.shared.reward_functions import (
    reward_foot_load_balance as _reward_foot_load_balance_pure,
)
from environments.shared.reward_functions import (
    reward_head_clearance as _reward_head_clearance_pure,
)
from environments.shared.reward_functions import (
    reward_soft_home_pose as _reward_soft_home_pose_pure,
)
from environments.shared.reward_functions import (
    reward_target_centered_height as _reward_target_centered_height_pure,
)


class TRexEnv(BaseDinoEnv):
    """Tyrannosaurus Rex bipedal locomotion and bite-attack environment."""

    action_mapping = "home-keyframe-residual/v1"
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
        natural_pitch: float = 0.027,
        height_weight: float = 0.0,
        height_target_tolerance: float = 0.0,
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
        foot_contact_weight: float = 0.0,
        foot_contact_gate: float = 0.0,
        bilateral_support_weight: float = 0.0,
        foot_contact_saturation_force: float = 100.0,
        foot_load_balance_weight: float = 0.0,
        foot_load_balance_min_support_force: float = 0.0,
        foot_load_balance_airborne_penalty: float = 0.0,
        action_jerk_weight: float = 0.0,
        support_conditioned_alive_fraction: float = 0.0,
        leg_home_pose_weight: float = 0.0,
        leg_home_pose_tolerance: float = 0.35,
        head_clearance_weight: float = 0.0,
        head_clearance_target: float = 0.60,
        head_clearance_tolerance: float = 0.48,
        neck_posture_weight: float = 0.0,
        neck_posture_tolerance: float = 0.35,
        nosedive_termination_threshold: float = 0.62,
        # Environment settings
        prey_distance_range: tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: tuple[float, float] = (-2.0, 2.0),
        # Matches the MJX registration, which is the value the evidence
        # supports.  Both bounds moved down by 0.05 with the theropod stance:
        # the flexed home keyframe settles at 0.9260 m where the columnar one
        # settled at 0.9757 (-0.0497).  Translating the whole band by the
        # measured stance drop is what preserves the three measurements the
        # old (0.75, 1.6) was sized on, rather than re-guessing them:
        #   * The tail is the real backstop.  tail_3/4/5 are unconditional
        #     termination geoms and the tail reaches the floor at a pelvis
        #     height of ~0.55-0.57 in a level squat, so the plant is already
        #     lying down below that.  The tail's pose is fixed relative to the
        #     pelvis and does not move with the legs, so that figure is
        #     unchanged and 0.70 still clears it by ~0.13.
        #   * Healthy full-horizon episodes bottomed out 0.134 m above the
        #     floor (0.884 against 0.75).  0.834 against 0.70 is the same
        #     margin under a stance that sits 0.0497 lower.
        #   * The MJX alive bonus scales by (z - floor) / (ceiling - floor),
        #     which is 0.266 at the settled stance before and after -- that is
        #     why the ceiling moves too -- against the velociraptor's 0.275.
        # The spawn tail is preserved too: 2000 resets at reset_noise_scale
        # = 0.10 put 0.75% below the floor on both plants (0.70 on this one,
        # 0.75 on the columnar one).  The root-height jitter is a 0.10 m
        # normal and dominates the joint noise on either stance, so the
        # translated floor keeps the same share of unrecoverable spawns.
        healthy_z_range: tuple[float, float] = (0.70, 1.55),
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
        self.height_target_tolerance = height_target_tolerance
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
        self.foot_contact_weight = foot_contact_weight
        self.foot_contact_gate = foot_contact_gate
        self.bilateral_support_weight = bilateral_support_weight
        self.foot_contact_saturation_force = foot_contact_saturation_force
        self.foot_load_balance_weight = foot_load_balance_weight
        self.foot_load_balance_min_support_force = foot_load_balance_min_support_force
        self.foot_load_balance_airborne_penalty = foot_load_balance_airborne_penalty
        self.action_jerk_weight = action_jerk_weight
        self.support_conditioned_alive_fraction = support_conditioned_alive_fraction
        self.leg_home_pose_weight = leg_home_pose_weight
        self.leg_home_pose_tolerance = leg_home_pose_tolerance
        self.head_clearance_weight = head_clearance_weight
        self.head_clearance_target = head_clearance_target
        self.head_clearance_tolerance = head_clearance_tolerance
        self.neck_posture_weight = neck_posture_weight
        self.neck_posture_tolerance = neck_posture_tolerance
        self.nosedive_termination_threshold = nosedive_termination_threshold

        if self.height_target_tolerance < 0.0:
            raise ValueError("height_target_tolerance must be non-negative")
        if self.foot_contact_saturation_force <= 0.0:
            raise ValueError("foot_contact_saturation_force must be positive")
        if not 0.0 <= self.support_conditioned_alive_fraction <= 1.0:
            raise ValueError("support_conditioned_alive_fraction must be in [0, 1]")
        if self.leg_home_pose_tolerance <= 0.0:
            raise ValueError("leg_home_pose_tolerance must be positive")
        if self.head_clearance_tolerance <= 0.0:
            raise ValueError("head_clearance_tolerance must be positive")
        if self.neck_posture_tolerance <= 0.0:
            raise ValueError("neck_posture_tolerance must be positive")

        # Natural forward pitch (~1.55°), measured: the pelvis frame at the home
        # keyframe is level, and under the home controller the plant settles at
        # forward_z ≈ -0.027.  The nosedive penalty and termination are measured
        # relative to that neutral pose.  The former 0.17 rad (9.7°) described
        # the torso capsule's built-in slope, not the pelvis frame the reward
        # actually reads, so it granted ~7 deg of unpenalized nose-down pitch.
        # The theropod stance halved the residual: a flexed limb settles closer
        # to its commanded pose than a columnar one does, so the plant now
        # settles 1.55° nose-down where the columnar stance settled 2.9°.
        # nosedive_termination_threshold in configs/trex/stage1_balance.toml is
        # calibrated against this number and moved with it.
        # Unlike the raptor, the T-Rex posture reward stays centred on world
        # vertical (see _get_reward_info); the MJX path matches by leaving
        # posture_target_forward_z unset for this species.
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
        # T-Rex policies command residuals around the complete XML home
        # control vector. Cache the named keyframe so Gymnasium reset and
        # action zero share one nominal state.
        self.home_keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.home_keyframe_id < 0:
            raise ValueError("T-Rex model must define a named 'home' keyframe")
        self._reset_keyframe_id = self.home_keyframe_id
        self._home_ctrl = self.model.key_ctrl[self.home_keyframe_id].copy()

        # Body IDs
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.skull_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "skull")

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
        # Height terminations sampled at every physics substep by the base
        # step loop; _is_terminated reads the per-check MIN so a between-
        # samples head dip terminates like the MJX height emulation does.
        # Order is consumed by index there.
        self._substep_height_checks = (
            ("site", self.head_tip_site_id),
            ("body", self.skull_body_id),
        )

        # Prey mocap body
        self.prey_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prey")

        # Sensor indices (order matches MJCF sensor definition)
        # pelvis_gyro(3), pelvis_accel(3), pelvis_orientation(4),
        # r_foot_touch(1), l_foot_touch(1)
        # _sensor_gyro_start, _sensor_accel_start, _sensor_quat_start
        # are inherited from BaseDinoEnv (0, 3, 6 respectively).
        self._sensor_r_foot = 10
        self._sensor_l_foot = 11
        # The pad sensors above see the plantar box only: a touch sensor sums
        # contacts on geoms of its site's own body, and the three digits are
        # child bodies on their own passive hinges.  Their own sensors are
        # appended after the tail block, so pad + digits is the force the foot
        # actually transmits -- at the home keyframe 388.4 N + 112.0 N against
        # a measured 500.4 N of floor contact.
        self._sensor_r_foot_digits = (24, 25, 26)
        self._sensor_l_foot_digits = (27, 28, 29)
        # Per-foot groups for the base class's substep MIN aggregation --
        # mirrors mjx_config's sensor_foot_indices + sensor_foot_aux_indices
        # so both backends aggregate the same sensors.
        self._foot_sensor_groups = (
            (self._sensor_r_foot, *self._sensor_r_foot_digits),
            (self._sensor_l_foot, *self._sensor_l_foot_digits),
        )

        # Stage-1 pose targets come from the named home keyframe instead of
        # duplicating angles in Python.  This keeps the reward centred on the
        # p5 theropod stance if the XML's authored equilibrium is recalibrated.
        leg_home_joint_names = (
            "r_hip_pitch",
            "r_hip_roll",
            "r_knee",
            "r_ankle",
            "l_hip_pitch",
            "l_hip_roll",
            "l_knee",
            "l_ankle",
        )
        neck_home_joint_names = ("neck_pitch", "neck_yaw", "head_pitch")
        self._leg_home_qpos_indices = self._joint_qpos_indices(leg_home_joint_names)
        self._neck_home_qpos_indices = self._joint_qpos_indices(neck_home_joint_names)
        home_qpos = self.model.key_qpos[self.home_keyframe_id]
        self._leg_home_qpos = home_qpos[self._leg_home_qpos_indices].copy()
        self._neck_home_qpos = home_qpos[self._neck_home_qpos_indices].copy()

    def _joint_qpos_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        """Resolve scalar hinge-joint qpos addresses, failing on model drift."""
        indices = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"T-Rex model must define joint {name!r}")
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(indices, dtype=np.int32)

    def _foot_contact_forces(self) -> tuple[float, float]:
        """Total floor contact force under each foot: plantar pad and digits."""
        sensordata = self.data.sensordata
        right = sensordata[self._sensor_r_foot] + sum(sensordata[index] for index in self._sensor_r_foot_digits)
        left = sensordata[self._sensor_l_foot] + sum(sensordata[index] for index in self._sensor_l_foot_digits)
        return float(right), float(left)

    def _bilateral_support_quality(self, right_force: float, left_force: float) -> float:
        """Return bounded support quality, requiring load on both feet."""
        _, quality = _reward_bilateral_support_pure(
            np.asarray((right_force, left_force)),
            self.foot_contact_saturation_force,
            1.0,
        )
        return float(quality)

    def _foot_load_balance(self, right_force: float, left_force: float) -> tuple[float, float]:
        """Return ``(reward, normalized load imbalance)`` for the foot pair.

        An unsupported pair reports imbalance ``1.0``, not ``0.0`` -- airborne
        is maximally imbalanced, not perfectly balanced -- and additionally
        pays ``foot_load_balance_airborne_penalty`` so that airborne is
        strictly worse than honest single support rather than tied with it.
        """
        reward, imbalance = _reward_foot_load_balance_pure(
            np.asarray((right_force, left_force)),
            self.foot_load_balance_weight,
            self.foot_load_balance_min_support_force,
            self.foot_load_balance_airborne_penalty,
        )
        return float(reward), float(imbalance)

    def _foot_load_imbalance(self, right_force: float, left_force: float) -> float:
        """Backwards-compatible diagnostic accessor: the imbalance alone."""
        return self._foot_load_balance(right_force, left_force)[1]

    def _home_pose_quality(
        self,
        qpos_indices: np.ndarray,
        target_qpos: np.ndarray,
        tolerance: float,
    ) -> tuple[float, float]:
        """Return RMS joint error and a smooth bounded home-pose quality."""
        _, rms_error, quality = _reward_soft_home_pose_pure(
            self.data.qpos[qpos_indices],
            target_qpos,
            tolerance,
            1.0,
        )
        return float(rms_error), float(quality)

    def _head_clearance_quality(self, head_tip_z: float) -> float:
        """Reward safe clearance smoothly from zero to the target height."""
        _, quality = _reward_head_clearance_pure(
            np.asarray(head_tip_z),
            self.head_clearance_target,
            self.head_clearance_tolerance,
            1.0,
        )
        return float(quality)

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

        # Pelvis state from sensors
        pelvis_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3].copy()
        pelvis_accel = self.data.sensordata[self._sensor_accel_start : self._sensor_accel_start + 3].copy()
        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4].copy()

        # Pelvis linear velocity (from root freejoint)
        pelvis_linvel = self.data.qvel[0:3].copy()

        # Foot contact (from touch sensors)
        foot_contact = np.array(self._foot_contact_forces())

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

        # 1d. Bilateral support and load balance.  Touch sensors report the
        # full plantar-plus-digit load for each foot.  Saturation prevents
        # impact spikes from being more valuable than quiet support, while the
        # minimum requires both feet to carry load.
        # AGGREGATED across the control step's physics substeps (per-foot MIN):
        # the last-substep read let the seed-43 bounce unload for 4 of every 5
        # substeps and still collect full support reward, and the same sample
        # feeds the r/l_foot_contact info keys the stance-duty gate certifies
        # from.  Falls back to the instantaneous read on un-stepped states.
        r_contact, l_contact = self._aggregated_foot_contact_forces()
        info["r_foot_contact"] = r_contact
        info["l_foot_contact"] = l_contact

        bilateral_support_quality = self._bilateral_support_quality(r_contact, l_contact)
        reward_bilateral_support = self.bilateral_support_weight * bilateral_support_quality
        info["bilateral_support_quality"] = bilateral_support_quality
        info["reward_bilateral_support"] = reward_bilateral_support

        # Take the pure function's REWARD, not just its diagnostic: the
        # airborne penalty that makes the ordering monotone lives in the
        # reward, so recomputing `-weight * imbalance` here would silently drop
        # it (and did, for the imbalance-only version this replaces).
        reward_foot_load_balance, foot_load_imbalance = self._foot_load_balance(r_contact, l_contact)
        info["foot_load_imbalance"] = foot_load_imbalance
        info["reward_foot_load_balance"] = reward_foot_load_balance

        # 2. Alive bonus (shared helper), optionally conditioned in part on
        # bilateral support.  A fraction below 1 leaves recovery headroom
        # after a noisy reset; the zero default is exactly the legacy bonus.
        raw_alive = self._reward_alive()
        alive_fraction = self.support_conditioned_alive_fraction
        alive_gate = (1.0 - alive_fraction) + alive_fraction * bilateral_support_quality
        reward_alive = raw_alive * alive_gate
        info["raw_alive"] = raw_alive
        info["alive_gate"] = alive_gate
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

        # 6c. Head clearance and neck posture.  Clearance is a smoothstep:
        # zero one configured tolerance below the target and saturated at the
        # target so lifting the head ever higher cannot farm reward.  The neck
        # target is the authored home keyframe, not a duplicated angle list.
        head_tip_z = float(head_tip_pos[2])
        head_clearance_quality = self._head_clearance_quality(head_tip_z)
        reward_head_clearance = self.head_clearance_weight * head_clearance_quality
        info["head_tip_z"] = head_tip_z
        info["head_pelvis_rel_z"] = head_tip_z - float(pelvis_pos[2])
        info["head_clearance_quality"] = head_clearance_quality
        info["reward_head_clearance"] = reward_head_clearance

        neck_posture_error, neck_posture_quality = self._home_pose_quality(
            self._neck_home_qpos_indices,
            self._neck_home_qpos,
            self.neck_posture_tolerance,
        )
        reward_neck_posture = self.neck_posture_weight * neck_posture_quality
        info["neck_posture_error"] = neck_posture_error
        info["neck_posture_quality"] = neck_posture_quality
        info["reward_neck_posture"] = reward_neck_posture

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
        # Settled pelvis height under the home controller, measured (zero action,
        # zero reset noise, 600 steps: 0.9260 m, sd 0.00001).  This must track the
        # plant: the previous 0.90 was the root height of the PRE-repair plant,
        # and after the July home-equilibrium fix raised the stance to 0.9757 the
        # term saturated at 1.0 everywhere in the healthy band -- a flat constant
        # worth 39% of stage-1 and 10% of stage-2 return that shaped nothing.
        # ``TestHeightTargetTracksStance`` pins it to the measured stance, which
        # is how the 0.9757 -> 0.9260 move under the theropod stance correction
        # was caught rather than left to drift.  Keep this and
        # ``target_standing_z`` in environments/trex/mjx_config.py equal.
        target_z = 0.9260
        if self.height_target_tolerance > 0.0:
            reward_height, height_error, height_quality = _reward_target_centered_height_pure(
                np.asarray(pelvis_height),
                target_z,
                self.height_target_tolerance,
                self.height_weight,
            )
            reward_height = float(reward_height)
            height_error = float(height_error)
            height_quality = float(height_quality)
        else:
            # Legacy one-sided reward: retain it exactly unless the explicit
            # target-centred tolerance switch is enabled.
            height_error = abs(pelvis_height - target_z)
            height_quality = float(np.clip((pelvis_height - min_z) / (target_z - min_z), 0.0, 1.0))
            reward_height = self.height_weight * height_quality
        info["height_error"] = height_error
        info["height_quality"] = height_quality
        info["reward_height"] = reward_height

        # 8d. Soft leg-pose retention around the authored p5 home stance.
        # Gaussian quality gives corrective signal without hard-locking a
        # joint, and averaging prevents one joint from dominating the term.
        leg_home_pose_error, leg_home_pose_quality = self._home_pose_quality(
            self._leg_home_qpos_indices,
            self._leg_home_qpos,
            self.leg_home_pose_tolerance,
        )
        reward_leg_home_pose = self.leg_home_pose_weight * leg_home_pose_quality
        info["leg_home_pose_error"] = leg_home_pose_error
        info["leg_home_pose_quality"] = leg_home_pose_quality
        info["reward_leg_home_pose"] = reward_leg_home_pose

        # 9. Gait symmetry (reward alternating foot contacts, shared helper)
        reward_gait, alternation_ratio = self._compute_gait_symmetry(
            float(r_contact), float(l_contact), self.gait_symmetry_weight
        )
        info["alternation_ratio"] = alternation_ratio
        info["contact_asymmetry"] = alternation_ratio  # backward compat with metrics
        info["reward_gait"] = reward_gait

        # 10. Action smoothness (shared helper)
        # Jerk BEFORE smoothness: _reward_action_smoothness rotates the action
        # history, so calling it first would leave the jerk term reading this
        # step's own action as its first lag.
        reward_action_jerk, action_jerk = self._reward_action_jerk(action)
        info["action_jerk"] = action_jerk
        info["reward_action_jerk"] = reward_action_jerk

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
            + reward_head_clearance
            + reward_neck_posture
            + reward_posture
            + reward_nosedive
            + reward_height
            + reward_bilateral_support
            + reward_foot_load_balance
            + reward_leg_home_pose
            + reward_gait
            + reward_smoothness
            + reward_action_jerk
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
        if forward_z < self._natural_forward_z - self.nosedive_termination_threshold:
            info["termination_reason"] = "nosedive"
            return True, info

        # Site-height termination: snout tip must stay above threshold
        # This catches nose-balancing that geom contact detection may miss.
        # The check reads the substep MIN (indices match the
        # _substep_height_checks declaration) so a dip that recovers between
        # control-boundary samples still terminates; the info key keeps the
        # boundary sample.
        head_tip_z = self.data.site_xpos[self.head_tip_site_id, 2]
        info["head_tip_z"] = head_tip_z
        if self._aggregated_min_height(0, head_tip_z) < 0.12:
            info["termination_reason"] = "head_contact"
            return True, info

        # Body-height termination: skull origin must stay above threshold
        skull_z = self.data.xpos[self.skull_body_id, 2]
        if self._aggregated_min_height(1, skull_z) < 0.45:
            info["termination_reason"] = "skull_low"
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
        self._prev_prev_action = None

        # Reset gait symmetry tracking
        self._reset_gait_state()


# Register with Gymnasium (MesozoicLabs namespace)
gym.register(
    id="MesozoicLabs/TRex-v0",
    entry_point="environments.trex.envs.trex_env:TRexEnv",
    max_episode_steps=1000,
)
