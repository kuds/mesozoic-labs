"""
Brachiosaurus Gymnasium Environment

A quadrupedal sauropod locomotion environment.  The scene holds the animal
and the floor alone: the food sphere and its observation channels (direction
and distance) left in physics r7 / policy interface r12 with the hunt
(decision D-D59), and the forward reference of the progress and heading terms
is the spawn's own heading (``_reference_heading_2d``, set by
BaseDinoEnv.reset), or under a live command the commanded heading (decision
D-D62).

Observation space (quadrupedal-heading-free/v1 since policy interface r12;
81 values; the total dimension is generated in the public species catalog):
    - Joint positions (qpos[7:]) — 31 hinge joints, excluding the root freejoint
    - Joint velocities (qvel[6:]) — the same 31 joints
    - Torso projected gravity (unit down vector in torso coordinates) — 3
    - Torso angular velocity (gyroscope, its own site frame) — 3
    - Torso linear velocity in the gravity-aligned heading frame (forward, left, up) — 3
    - Torso acceleration (accelerometer, its own site frame) — 3
    - Foot contact forces (pad + meta touch sensors per leg) — 4
    - Body-relative command (v_x_cmd, v_y_cmd, yaw_rate_cmd; zeros under command_mode = "none") — 3

No channel carries the world heading: the two root blocks that replaced the
world-frame torso quaternion and linear velocity in policy interface r12 come
from the free root's own quaternion through
environments/shared/heading_observation.py
(docs/HEADING_INVARIANCE_DESIGN_2026_10.md), the bipeds' layout (D-D31) on
the torso, so turning the whole scene about vertical leaves the observation
unchanged.  Policy interface r11 and earlier observed the torso quaternion
(4), the world-frame torso linear velocity (3) and the food direction and
distance (4), 86 values.

Action space:
    - Continuous control for all actuators [-1, 1] normalized
    - Neck, four-leg, and tail actuator groups; the public catalog derives the current total

Reward components:
    - Forward velocity (along the forward reference: the spawn's forward, or the commanded heading)
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
    - Heading alignment (facing the forward reference)
    - Lateral velocity penalty (anti crab-walk)
    - Spin penalty (penalize torso angular velocity)
    - Speed penalty (penalise absolute speed above threshold)
    - Tail stability (penalize tail tip angular velocity)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.heading_observation import heading_free_root_segments


class BrachiosaurusAltithoraxEnv(BaseDinoEnv):
    """Brachiosaurus quadrupedal locomotion environment.

    The species base (decision D-D33): the task, the rewards and the hashed
    policy-interface methods (``_get_obs``, ``_scale_action``,
    ``_cache_ids``), shared by every embodiment of the species.  An
    embodiment subclass (``anatomical/env.py``) sets ``embodiment`` and
    ``MODEL_PATH``; this class is never instantiated itself.
    """

    #: Set by each embodiment subclass: its embodiment and its MJCF.
    embodiment: str
    MODEL_PATH: Path
    #: The species' training labels (species_registry.get_species_config).
    HEIGHT_LABEL = "Torso height"
    #: No success event: the food reach, its only one, was removed with the
    #: hunt (decision D-D59).
    SUCCESS_KEYS: tuple[str, ...] = ()

    # SB3 only since policy interface r11 (configs/plant_versions.toml note
    # 23): the frozen MJX interface core (D-D17) was removed with this
    # declaration, as the velociraptor (note 14) and the T-Rex (note 18) left
    # it.  The plant contract now hashes _cache_ids
    # (interface_implementations.sb3_sensor_cache) as it does for every
    # SB3-only species.
    supported_training_backends = ("stable-baselines3",)
    action_mapping = "home-keyframe-residual/v1"
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
        # Environment settings
        healthy_z_range: tuple[float, float] = (1.0, 3.5),
        reset_noise_scale: float = 0.01,
        perturbation_capture_velocity_multiple: float = 0.0,
        perturbation_interval: float = 2.0,
        perturbation_jitter: float = 0.5,
        perturbation_duration: float = 0.20,
        perturbation_direction: str = "uniform_horizontal",
        command_mode: str = "none",
        command_config: DirectionCommandConfig | None = None,
    ):
        model_path = str(self.MODEL_PATH)

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

        # Natural forward pitch. The nosedive penalty is measured relative to
        # this angle so the Brachiosaurus isn't punished for its natural stance.
        # Default 0.0 (upright quadruped); override if the model has a natural lean.
        self._natural_forward_z = -np.sin(natural_pitch)

        # Action history for the smoothness term (cleared by
        # _reset_episode_state).
        self._prev_action: np.ndarray | None = None

        # Gait symmetry: track diagonal pair alternation for quadrupedal gait.
        # Diagonal-A = FR+RL, Diagonal-B = FL+RR.  A proper walk/trot
        # alternates these pairs.
        self._init_quadruped_gait_state()

        # The progress and heading terms' forward reference and the drift
        # penalty's anchor are BaseDinoEnv's _forward_reference_2d and
        # _drift_anchor_2d: under command_mode "none" the spawn's
        # _reference_heading_2d (the unit forward of the spawn's root
        # quaternion) and _initial_pos_2d (the spawn's position), both set by
        # BaseDinoEnv.__init__ and every reset; under a live command the
        # command's heading and its event's start (decision D-D62).

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
            perturbation_capture_velocity_multiple=perturbation_capture_velocity_multiple,
            perturbation_interval=perturbation_interval,
            perturbation_jitter=perturbation_jitter,
            perturbation_duration=perturbation_duration,
            perturbation_direction=perturbation_direction,
            command_mode=command_mode,
            command_config=command_config,
        )

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""
        # Policies command residuals around the complete XML home control
        # vector, so Gymnasium reset and action zero share one nominal state.
        self._cache_home_keyframe("Brachiosaurus")

        # Body IDs
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # Geom IDs for contact detection
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

        # Site IDs for sensors.  head_tip carries the head_tip_pos framepos
        # sensor and the posture telemetry's head height; no reward reads it.
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.head_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tip")
        self.tail_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")

        # Metapodial collision proxies: fore metacarpus and hind metatarsus.
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
        # are inherited from BaseDinoEnv (0, 3, 6 respectively): torso_gyro,
        # torso_accel and torso_orientation.  The observation reads the first
        # two; the posture, nosedive and heading terms read the third.
        self._sensor_fr_foot = 10
        self._sensor_fl_foot = 11
        self._sensor_rr_foot = 12
        self._sensor_rl_foot = 13
        # The pad sensors above see the foot-pad ellipsoid only: a touch
        # sensor sums contacts on geoms of its site's own body, and each
        # metacarpus/metatarsus is the pad's parent body and can contact
        # terrain independently of the sole.  The meta sensors are appended after the tail block, so
        # pad + meta is the force the leg actually transmits (see
        # FOOT_SENSOR_VERIFICATION.md §4 — before this repair all four feet
        # read exactly 0.0 N while carrying 98.8% of body weight).
        self._sensor_fr_meta = 26
        self._sensor_fl_meta = 27
        self._sensor_rr_meta = 28
        self._sensor_rl_meta = 29
        # Per-foot groups for the base class's substep MIN aggregation: each
        # foot's pad sensor and its appended meta sensor.
        self._foot_sensor_groups = (
            (self._sensor_fr_foot, self._sensor_fr_meta),
            (self._sensor_fl_foot, self._sensor_fl_meta),
            (self._sensor_rr_foot, self._sensor_rr_meta),
            (self._sensor_rl_foot, self._sensor_rl_meta),
        )

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized residual actions around the XML home controls.

        The two halves are scaled independently so the mapping preserves the
        actuator endpoints while making action zero exactly the named home
        control, even when that control is not the range midpoint.  Under the
        inherited midpoint mapping, "do nothing" dragged all four knees up to
        0.35 rad away from the standing pose — the statue that §6 of
        PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md measured falling on 40 of 40
        episodes was never commanding the home stance at all.
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
        """Construct the quadrupedal-heading-free/v1 observation vector (81 values since policy interface r12).

        No channel carries the world heading: the torso orientation is
        projected gravity and the torso linear velocity is expressed in the
        gravity-aligned heading frame, both from the free root's own
        quaternion qpos[3:7] (environments/shared/heading_observation.py),
        never from the orientation sensor, which is still zero when the
        constructor calls this before any forward pass.  The bipeds' layout
        (D-D31) on the torso, with the four per-leg foot channels.  Policy
        interface r11 and earlier observed the torso quaternion, the
        world-frame torso linear velocity and the food direction and
        distance instead (86 values).
        """
        # Joint positions (exclude root freejoint: first 7 values are pos + quat)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values are lin + ang vel)
        qvel = self.data.qvel[6:].copy()

        # Torso state from sensors, already in their own site frame (never rotated again)
        torso_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3].copy()
        torso_accel = self.data.sensordata[self._sensor_accel_start : self._sensor_accel_start + 3].copy()

        # Foot contacts: per-leg pad + meta totals, keeping the observation
        # width at four values while reporting the force each leg transmits.
        foot_contact = np.array(self._foot_contact_forces())

        # Heading-free root block: projected gravity and the root's world
        # linear velocity (free joint qvel[0:3]) in the heading frame.
        gravity, heading_velocity = heading_free_root_segments(self.data.qpos, self.data.qvel[0:3])

        obs = np.concatenate(
            [
                qpos,  # Joint positions
                qvel,  # Joint velocities
                gravity,  # Projected gravity (unit down vector, torso frame)
                torso_gyro,  # Angular velocity
                heading_velocity,  # Linear velocity (forward, left, up)
                torso_accel,  # Accelerometer
                foot_contact,  # Foot contacts (4)
                self._command,  # Body-relative command (v_x, v_y, yaw_rate), pre-scaled; zeros unless command_mode != "none"
            ]
        ).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info: dict[str, Any] = {}

        torso_pos = self.data.xpos[self.torso_id]
        # The forward reference: the spawn's heading under command_mode "none",
        # the commanded heading under a live command (BaseDinoEnv._forward_reference_2d).
        forward_ref_2d = self._forward_reference_2d()
        vel_2d = self.data.qvel[0:2]

        # 1-1c. Forward velocity (along the forward reference), backward velocity penalty and drift penalty
        reward_forward, reward_backward, reward_drift = self._progress_terms(
            info, vel_2d, forward_ref_2d, torso_pos[:2]
        )

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
        reward_nosedive = self._nosedive_term(info, torso_quat)

        # 8. Height maintenance reward
        torso_height = self._clearance(torso_pos)
        info["pelvis_height"] = torso_height
        min_z = self.healthy_z_range[0]
        target_z = 1.2  # Brachiosaurus natural standing torso height (from XML keyframe)
        height_frac = float(np.clip((torso_height - min_z) / (target_z - min_z), 0.0, 1.0))
        reward_height = self.height_weight * height_frac
        info["reward_height"] = reward_height

        # 9. Gait symmetry (reward alternating diagonal pair contacts).
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
        info["contact_asymmetry"] = alternation_ratio
        info["reward_gait_symmetry"] = reward_gait_sym

        # 10. Action smoothness (shared helper)
        reward_smoothness, action_delta = self._reward_action_smoothness(action)
        info["action_delta"] = action_delta
        info["reward_smoothness"] = reward_smoothness

        # 11-12. Heading alignment and lateral velocity penalty
        reward_heading, reward_lateral = self._heading_terms(info, torso_quat, forward_ref_2d, vel_2d)

        # Torso angular velocity (for spinning detection in shared diagnostics)
        pelvis_angular_vel, pelvis_yaw_vel = self._compute_pelvis_diagnostics()
        info["pelvis_angular_vel"] = pelvis_angular_vel
        info["pelvis_yaw_vel"] = pelvis_yaw_vel

        # 13. Spin penalty
        reward_spin, spin_instability = self._compute_angular_velocity_penalty(self.spin_penalty_weight)
        info["spin_instability"] = spin_instability
        info["reward_spin"] = reward_spin

        # 14-14b. Speed penalty above its threshold, idle penalty below its own
        reward_speed, reward_idle = self._speed_terms(info, vel_2d)

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
        )
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        """Check if episode should terminate."""
        # Height/tilt termination (shared)
        terminated, info = self._root_termination(self.torso_id, "torso_height")
        if terminated:
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

    def _reset_episode_state(self) -> None:
        """Clear the per-episode state: the action history and the gait tracking.

        BaseDinoEnv.reset has already set the drift anchor and the reference
        heading from the spawn; nothing here draws a random number.
        """
        # The smoothness term reads no lag on an episode's first step (the
        # second lag is cleared with it, though no brachiosaurus term reads
        # it: the species carries no jerk term).
        self._prev_action = None
        self._prev_prev_action = None

        # Reset gait symmetry tracking (quadrupedal diagonal pairs)
        self._reset_quadruped_gait_state()
