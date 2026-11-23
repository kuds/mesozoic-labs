"""
Dibothrosuchus elaphros Gymnasium Environment

A small, gracile sphenosuchian crocodylomorph with erect (parasagittal) limbs
and hindlimbs longer than its forelimbs.  The scene holds the animal and the
floor alone: the prey sphere, its observation channels (direction and
distance) and the snout_snap contact geom left in physics r4 / policy
interface r11 with the hunt (decision D-D59), and the forward reference of the
progress and heading terms is the spawn's own heading
(``_reference_heading_2d``, set by BaseDinoEnv.reset).

Observation space (quadrupedal-heading-free/v1 since policy interface r11;
75 values; the total dimension is generated in the public species catalog):
    - Joint positions (qpos[7:]) — 28 hinge
    - Joint velocities (qvel[6:]) — 28 hinge
    - Trunk projected gravity (unit down vector in torso coordinates) — 3
    - Trunk angular velocity (gyroscope, its own site frame) — 3
    - Trunk linear velocity in the gravity-aligned heading frame (forward, left, up) — 3
    - Trunk acceleration (accelerometer, its own site frame) — 3
    - Foot contact forces (4 pad touch sensors) — 4
    - Body-relative command (v_x_cmd, v_y_cmd, yaw_rate_cmd; zeros under command_mode = "none") — 3

No channel carries the world heading: the two root blocks that replaced the
world-frame trunk quaternion and linear velocity in policy interface r11 come
from the free root's own quaternion through
environments/shared/heading_observation.py
(docs/HEADING_INVARIANCE_DESIGN_2026_10.md), the bipeds' layout (D-D31) on
the torso, so turning the whole scene about vertical leaves the observation
unchanged.  Policy interface r10 and earlier observed the trunk quaternion
(4), the world-frame trunk linear velocity (3) and the prey direction and
distance (4), 80 values.

Action space (total dimension is generated in the public species catalog):
    - Neck/skull: neck pitch, neck yaw, head pitch (3)
    - Front right / front left leg: hip pitch/roll, knee, ankle, toe (5 each)
    - Rear right / rear left leg: hip pitch/roll, knee, ankle, toe (5 each)
    - Tail: pitch 1, yaw 1, pitch 2, pitch 3 (4)

Reward components:
    - Forward velocity (along the reference heading, the spawn's forward)
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
    - Heading alignment (facing the reference heading, ``_reference_heading_2d``)
    - Lateral velocity penalty (anti crab-walk)
    - Spin penalty (penalize trunk angular velocity)
    - Speed penalty (penalise absolute speed above threshold)
    - Idle penalty (penalise standing still)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.heading_observation import heading_free_root_segments


class DibothrosuchusElaphrosEnv(BaseDinoEnv):
    """Dibothrosuchus quadrupedal locomotion environment.

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
    HEIGHT_LABEL = "Trunk height"
    #: No success event: the snap, its only one, was removed with the hunt
    #: (decision D-D59).
    SUCCESS_KEYS: tuple[str, ...] = ()

    # SB3 only since policy interface r10 (configs/plant_versions.toml note
    # 23): the frozen MJX interface core (D-D17) was removed with this
    # declaration, as the velociraptor (note 14) and the T-Rex (note 18) left
    # it.  The plant contract now hashes _cache_ids
    # (interface_implementations.sb3_sensor_cache) as it does for every
    # SB3-only species.
    supported_training_backends = ("stable-baselines3",)
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
        # Read only by the retired MJX reward, which gated its alive bonus on
        # foot contact and paid a per-step grounded bonus; this env gets the
        # same effect from snout-contact and body-contact termination.  They
        # are accepted and stored (stance.toml sets them, and
        # task_sha256 hashes them) but not read by _get_reward_info.
        foot_contact_weight: float = 0.0,
        foot_contact_gate: float = 0.0,
        nosedive_termination_threshold: float = 0.55,
        # Environment settings
        healthy_z_range: tuple[float, float] = (0.18, 0.55),
        max_tilt_angle: float = 0.9,
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
        self.foot_contact_weight = foot_contact_weight
        self.foot_contact_gate = foot_contact_gate
        self.nosedive_termination_threshold = nosedive_termination_threshold

        # Natural forward pitch.  Measured: holding the home control vector for
        # 1500 steps leaves the trunk frame at forward_z +0.003, i.e. level, so
        # the default of 0.0 is the plant's real neutral rather than a guess.
        # Both the nosedive penalty and the posture reward stay centred on
        # world vertical.
        self._natural_forward_z = -np.sin(natural_pitch)

        # Action history for the smoothness term (cleared by
        # _reset_episode_state).
        self._prev_action: np.ndarray | None = None

        # Gait symmetry: quadrupedal diagonal pair alternation.
        # Diagonal-A = FR+RL, Diagonal-B = FL+RR.
        self._init_quadruped_gait_state()

        # The progress and heading terms' forward reference and the drift
        # penalty's anchor are BaseDinoEnv's _reference_heading_2d (the unit
        # forward of the spawn's root quaternion) and _initial_pos_2d (the
        # spawn's position), both set by BaseDinoEnv.__init__ and every reset.

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
        self._cache_home_keyframe("Dibothrosuchus")

        # Body IDs
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.skull_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "skull")

        # Geom IDs for contact detection
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

        # Site IDs for sensors.  snout_tip is the snout-prop termination's
        # height probe and carries the snout_tip_pos framepos sensor.
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

        # Sensor indices (order matches the MJCF sensor definition)
        # torso_gyro(3), torso_accel(3), torso_orientation(4),
        # fr/fl/rr/rl_foot_touch(1 each)
        # _sensor_gyro_start, _sensor_accel_start, _sensor_quat_start
        # are inherited from BaseDinoEnv (0, 3, 6 respectively).  The
        # observation reads the first two; the posture, nosedive and heading
        # terms read the third.
        self._sensor_fr_foot = 10
        self._sensor_fl_foot = 11
        self._sensor_rr_foot = 12
        self._sensor_rl_foot = 13
        # Per-foot groups for the base class's substep MIN aggregation: one
        # touch sensor per foot (no meta sensors on this plant).
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
        """Construct the quadrupedal-heading-free/v1 observation vector (75 values since policy interface r11).

        No channel carries the world heading: the trunk orientation is
        projected gravity and the trunk linear velocity is expressed in the
        gravity-aligned heading frame, both from the free root's own
        quaternion qpos[3:7] (environments/shared/heading_observation.py),
        never from the orientation sensor, which is still zero when the
        constructor calls this before any forward pass.  The bipeds' layout
        (D-D31) on the torso, with one touch channel per foot.  Policy
        interface r10 and earlier observed the trunk quaternion, the
        world-frame trunk linear velocity and the prey direction and distance
        instead (80 values).
        """
        # Joint positions (exclude root freejoint: first 7 values are pos + quat)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values are lin + ang vel)
        qvel = self.data.qvel[6:].copy()

        # Trunk state from sensors, already in their own site frame (never rotated again)
        torso_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3].copy()
        torso_accel = self.data.sensordata[self._sensor_accel_start : self._sensor_accel_start + 3].copy()

        # Foot contact: one pad touch sensor per foot (_foot_sensor_groups, FR, FL, RR, RL).
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
        # The forward reference: the spawn's heading, fixed for the episode.
        forward_ref_2d = self._reference_heading_2d
        vel_2d = self.data.qvel[0:2]

        # 1-1c. Forward velocity (along the reference heading), backward velocity penalty and drift penalty
        reward_forward, reward_backward, reward_drift = self._progress_terms(
            info, vel_2d, forward_ref_2d, torso_pos[:2]
        )

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

        # 8. Continuous posture reward
        torso_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        reward_posture, tilt_angle = self._compute_posture_reward(torso_quat, self.posture_weight)
        info["tilt_angle"] = tilt_angle
        info["reward_posture"] = reward_posture

        # 9. Nosedive penalty
        reward_nosedive = self._nosedive_term(info, torso_quat)

        # 10. Trunk height (for LocomotionMetrics tracking) and maintenance reward
        torso_height = self._clearance(torso_pos)
        info["torso_height"] = torso_height
        info["pelvis_height"] = torso_height  # alias for LocomotionMetrics
        min_z = self.healthy_z_range[0]
        # Settled trunk height under the home controller on physics r1; since r2 (the elliptic cone) the trunk
        # settles at 0.3132, above it, where the clipped term already pays in full.
        target_z = 0.3129
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

        # 13-14. Heading alignment and lateral velocity penalty
        reward_heading, reward_lateral = self._heading_terms(info, torso_quat, forward_ref_2d, vel_2d)

        # Trunk angular velocity (for spinning detection in shared diagnostics)
        pelvis_angular_vel, pelvis_yaw_vel = self._compute_pelvis_diagnostics()
        info["pelvis_angular_vel"] = pelvis_angular_vel
        info["pelvis_yaw_vel"] = pelvis_yaw_vel

        # 15. Spin penalty
        reward_spin, spin_instability = self._compute_angular_velocity_penalty(self.spin_penalty_weight)
        info["spin_instability"] = spin_instability
        info["reward_spin"] = reward_spin

        # 16-16b. Speed penalty above its threshold, idle penalty below its own
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
        # Height/tilt, then nosedive termination (shared).  The lower height
        # bound is the sprawl gate: a modern-crocodilian belly-down sprawl
        # drops the trunk well below the erect stance this species is being
        # trained to hold.
        terminated, info = self._root_termination(self.torso_id, "torso_height", self.nosedive_termination_threshold)
        if terminated:
            return True, info

        # Site-height termination: the snout tip must stay off the ground.
        # Catches snout-propping that geom contact detection may miss.  Reads
        # the substep MIN so a dip that recovers between control-boundary
        # samples still terminates; the info key keeps the boundary sample.
        snout_tip_z = self._clearance(self.data.site_xpos[self.snout_tip_site_id])
        info["snout_tip_z"] = snout_tip_z
        if self._aggregated_min_height(0, snout_tip_z) < 0.04:
            info["termination_reason"] = "head_contact"
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
        # second lag is cleared with it, though no dibothrosuchus term reads
        # it: the species carries no jerk term).
        self._prev_action = None
        self._prev_prev_action = None

        # Reset gait symmetry tracking (quadrupedal diagonal pairs)
        self._reset_quadruped_gait_state()
