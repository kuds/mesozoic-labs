"""
Velociraptor Gymnasium Environment

A bipedal dinosaur locomotion environment with predatory strike behavior.

Observation space (total dimension is generated in the public species catalog):
    - Joint positions (qpos[7:]) — 24 hinge joints excluding root freejoint
    - Joint velocities (qvel[6:]) — 24 hinge joints excluding root freejoint
    - Pelvis orientation (quaternion) — 4
    - Pelvis angular velocity (gyroscope) — 3
    - Pelvis linear velocity — 3
    - Pelvis acceleration — 3
    - Foot contact — 2 (per-foot sum of the toe d3, metatarsus and toe d4 touch sensors)
    - Prey direction (unit vector) — 3
    - Prey distance (scalar) — 1
    - Body-relative command (v_x_cmd, v_y_cmd, yaw_rate_cmd; zeros under command_mode = "none") — 3

Action space (total dimension is generated in the public species catalog):
    - Actions are residuals around the named XML ``home`` keyframe controls
    - Zero commands home; -1/+1 command each actuator's lower/upper limit
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
    - Alive bonus (optionally conditioned in part on bilateral support)
    - Bilateral support (load on the weaker-loaded foot)
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Strike bonus (when claw contacts prey)
    - Approach shaping (distance to prey)
    - Proximity bonus (continuous reward for being close to prey)
    - Claw proximity shaping (reward for positioning claw tip near prey)
    - Posture (continuous deviation from the natural forward lean)
    - Nosedive penalty
    - Gait symmetry (alternating foot contacts)
    - Action smoothness (penalize jerky action changes)
    - Action jerk (penalize the second difference of actions, i.e. chatter)
    - Action saturation (penalize commands parked at their range limits)
    - Leg home pose (soft retention of the home keyframe leg stance)
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
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.reward_functions import (
    reward_action_saturation as _reward_action_saturation_pure,
)
from environments.shared.reward_functions import (
    reward_bilateral_support as _reward_bilateral_support_pure,
)
from environments.shared.reward_functions import (
    reward_soft_home_pose as _reward_soft_home_pose_pure,
)


class RaptorEnv(BaseDinoEnv):
    """Velociraptor locomotion and strike environment."""

    # SB3-only since policy-interface revision 11 (plant_versions note 14).
    # The foot observation sums three touch sensors per foot, and the frozen
    # MJX registration (D-D17) reads the toe-d3 sensor alone and is never
    # edited, so the two backends would observe different feet; KNOWN_ISSUES
    # and CLEANUP_PLAN_2026_09 section 4.1 prescribe exactly this exit at the
    # species' next policy-interface revision.
    supported_training_backends = ("stable-baselines3",)
    action_mapping = "home-keyframe-residual/v1"
    _camera_distance = 2.0
    _camera_azimuth = 135
    _camera_elevation = -20
    _camera_track_body = "pelvis"

    def __init__(
        self,
        render_mode: str | None = None,
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
        speed_penalty_weight: float = 0.0,
        speed_penalty_threshold: float = 0.10,
        idle_penalty_weight: float = 0.0,
        idle_velocity_threshold: float = 0.05,
        # Stance-quality terms, named as on TRexEnv.  Every default is inert:
        # a zero weight (or alive fraction) reproduces the reward that
        # predates them bit for bit.
        bilateral_support_weight: float = 0.0,
        foot_contact_saturation_force: float = 50.0,
        support_conditioned_alive_fraction: float = 0.0,
        action_jerk_weight: float = 0.0,
        action_saturation_weight: float = 0.0,
        action_saturation_threshold: float = 0.9,
        leg_home_pose_weight: float = 0.0,
        leg_home_pose_tolerance: float = 0.35,
        # Environment settings
        prey_distance_range: tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: tuple[float, float] = (-2.0, 2.0),
        healthy_z_range: tuple[float, float] = (0.3, 1.0),
        reset_noise_scale: float = 0.01,
        perturbation_capture_velocity_multiple: float = 0.0,
        perturbation_interval: float = 2.0,
        perturbation_jitter: float = 0.5,
        perturbation_duration: float = 0.20,
        perturbation_direction: str = "uniform_horizontal",
        command_mode: str = "none",
        command_config: DirectionCommandConfig | None = None,
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
        self.speed_penalty_weight = speed_penalty_weight
        self.speed_penalty_threshold = speed_penalty_threshold
        self.idle_penalty_weight = idle_penalty_weight
        self.idle_velocity_threshold = idle_velocity_threshold
        self.bilateral_support_weight = bilateral_support_weight
        self.foot_contact_saturation_force = foot_contact_saturation_force
        self.support_conditioned_alive_fraction = support_conditioned_alive_fraction
        self.action_jerk_weight = action_jerk_weight
        self.action_saturation_weight = action_saturation_weight
        self.action_saturation_threshold = action_saturation_threshold
        self.leg_home_pose_weight = leg_home_pose_weight
        self.leg_home_pose_tolerance = leg_home_pose_tolerance

        if self.foot_contact_saturation_force <= 0.0:
            raise ValueError("foot_contact_saturation_force must be positive")
        if not 0.0 <= self.support_conditioned_alive_fraction <= 1.0:
            raise ValueError("support_conditioned_alive_fraction must be in [0, 1]")
        if not 0.0 <= self.action_saturation_threshold < 1.0:
            raise ValueError("action_saturation_threshold must be in [0, 1)")
        if self.leg_home_pose_tolerance <= 0.0:
            raise ValueError("leg_home_pose_tolerance must be positive")

        # Natural forward pitch (~20°). Posture shaping, the nosedive penalty,
        # and nosedive termination are measured relative to this angle so the
        # raptor is not rewarded for abandoning its biomechanically supported
        # forward lean.  Measured, not authored: the physics-r3 statue settles
        # at 20.10° (0.3508 rad over 40 seeds at reset noise 0.05), where the
        # r2 plant settled at 24.0° and the statue paid -95 per episode of
        # nosedive charge for standing still.  TestNaturalPitchTracksStance
        # pins the default to the settled stance.
        self._natural_forward_z = -np.sin(natural_pitch)

        # Raptor-specific env settings
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
            perturbation_capture_velocity_multiple=perturbation_capture_velocity_multiple,
            perturbation_interval=perturbation_interval,
            perturbation_jitter=perturbation_jitter,
            perturbation_duration=perturbation_duration,
            perturbation_direction=perturbation_direction,
            command_mode=command_mode,
            command_config=command_config,
        )

        # Leg-pose target for the leg_home_pose term, hips to toes: the named
        # home keyframe, which is the standing equilibrium since physics
        # revision 3 (springs anchored there, servos preloaded against
        # gravity) -- the noise-free statue settles within 0.1 deg of it on
        # every leg joint.  Resolved here rather than in _cache_ids, whose
        # tokens are part of this SB3-only species' policy-interface digest:
        # a reward target is not part of what a checkpoint observes.
        leg_home_joint_names = (
            "r_hip_pitch",
            "r_hip_roll",
            "r_knee",
            "r_ankle",
            "r_toe_d3_joint",
            "r_toe_d4_joint",
            "l_hip_pitch",
            "l_hip_roll",
            "l_knee",
            "l_ankle",
            "l_toe_d3_joint",
            "l_toe_d4_joint",
        )
        self._leg_home_qpos_indices = self._joint_qpos_indices(leg_home_joint_names)
        self._leg_home_qpos = self.model.key_qpos[self.home_keyframe_id][self._leg_home_qpos_indices].copy()

    def _joint_qpos_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        """Resolve scalar hinge-joint qpos addresses, failing on model drift."""
        indices = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Velociraptor model must define joint {name!r}")
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(indices, dtype=np.int32)

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""
        # The raptor policy commands residuals around the biomechanically
        # balanced XML home pose.  Cache the controls once so action zero can
        # preserve that pose without a keyframe lookup on every environment
        # step.
        self._cache_home_keyframe("Velociraptor")

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
        # The toe-d3 sensors above see that digit only: a touch sensor sums
        # contacts on geoms of its site's own body, and the metatarsal head
        # and digit 4 are other bodies.  Their own sensors are appended after
        # the tail block, so d3 + metatarsus + d4 is the force the foot
        # actually transmits -- at the settled stance 36.0 + 18.7 + 11.6 N
        # against 66.2 N of floor contact, where d3 alone reads 54%.
        self._sensor_r_foot_aux = (27, 28)
        self._sensor_l_foot_aux = (29, 30)
        # Per-foot groups for the base class's substep MIN aggregation, d3
        # first (the pinned group[0] + sum(rest) order).  They feed the
        # observation, the contact-shaped rewards and the r/l_foot_contact
        # info keys alike.
        self._foot_sensor_groups = (
            (self._sensor_r_foot, *self._sensor_r_foot_aux),
            (self._sensor_l_foot, *self._sensor_l_foot_aux),
        )

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized residual actions around the XML home controls.

        The home pose is not generally the midpoint of an actuator's control
        range.  A piecewise-linear mapping therefore preserves all three
        policy-interface anchors:

        * ``-1`` maps to the actuator minimum,
        * ``0`` maps exactly to the home-keyframe control, and
        * ``+1`` maps to the actuator maximum.

        Reward terms continue to receive the normalized residual ``action``
        from :meth:`step`; only the command sent to MuJoCo is transformed.
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

        # Foot contact (from touch sensors: d3 + metatarsus + d4 per foot)
        foot_contact = np.array(self._foot_contact_forces())

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
                self._command,  # Body-relative command (v_x, v_y, yaw_rate), pre-scaled; zeros unless command_mode != "none"
            ]
        ).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info: dict[str, Any] = {}

        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        forward_ref_2d = self._initial_prey_dir_2d
        vel_2d = self.data.qvel[0:2]

        # 1-1c. Forward velocity (toward prey), backward velocity penalty and drift penalty
        reward_forward, reward_backward, reward_drift = self._progress_terms(
            info, vel_2d, forward_ref_2d, pelvis_pos[:2]
        )

        # 1d. Bilateral support.  Since physics revision 3 the per-foot touch
        # sum (d3 + metatarsus + d4) IS the floor force under that foot, so
        # the weaker-loaded foot's fraction of the saturation force is a
        # floor-true support quality; saturating it keeps an impact spike
        # from being worth more than quiet support.  Substep-MIN aggregated:
        # the info keys feed the stance diagnostics, and a touchdown that
        # unloads between control-boundary samples must not read as
        # continuous support.
        r_contact, l_contact = self._aggregated_foot_contact_forces()
        info["r_foot_contact"] = float(r_contact)
        info["l_foot_contact"] = float(l_contact)

        _, support_quality = _reward_bilateral_support_pure(
            np.asarray((r_contact, l_contact)),
            self.foot_contact_saturation_force,
            1.0,
        )
        bilateral_support_quality = float(support_quality)
        reward_bilateral_support = self.bilateral_support_weight * bilateral_support_quality
        info["bilateral_support_quality"] = bilateral_support_quality
        info["reward_bilateral_support"] = reward_bilateral_support

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

        # 5. Strike bonus (either claw contacts the prey)
        struck = self._contact_geom({self.r_claw_geom_id, self.l_claw_geom_id}, self.prey_geom_id) is not None
        info["strike_success"] = 1.0 if struck else 0.0
        reward_strike = self.strike_bonus if struck else 0.0
        info["reward_strike"] = reward_strike

        # 6. Approach shaping
        prey_distance = float(np.linalg.norm(prey_pos - pelvis_pos))
        reward_approach, approach_delta = self._compute_approach_shaping(
            prey_distance, self._prev_prey_distance, self.strike_approach_weight, 10.0
        )
        self._prev_prey_distance = prey_distance
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

        # 7. Continuous posture reward centred on the natural forward lean.
        # The returned tilt remains absolute (relative to world-up) for
        # diagnostics and termination parity.
        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        reward_posture, tilt_angle = self._compute_lean_aware_posture_reward(
            pelvis_quat,
            self.posture_weight,
            self._natural_forward_z,
        )
        info["tilt_angle"] = tilt_angle
        info["reward_posture"] = reward_posture

        # 8. Nosedive penalty
        reward_nosedive = self._nosedive_term(info, pelvis_quat)

        # 8b. Pelvis height (for LocomotionMetrics tracking)
        info["pelvis_height"] = self._clearance(self.data.xpos[self.pelvis_id])

        # 8b2. Soft leg-pose retention around the home keyframe, hips to toes.
        # The mean per-joint Gaussian gives corrective signal without
        # hard-locking a joint, and averaging keeps one joint from dominating.
        _, leg_rms_error, leg_quality = _reward_soft_home_pose_pure(
            self.data.qpos[self._leg_home_qpos_indices],
            self._leg_home_qpos,
            self.leg_home_pose_tolerance,
            1.0,
        )
        leg_home_pose_error = float(leg_rms_error)
        leg_home_pose_quality = float(leg_quality)
        reward_leg_home_pose = self.leg_home_pose_weight * leg_home_pose_quality
        info["leg_home_pose_error"] = leg_home_pose_error
        info["leg_home_pose_quality"] = leg_home_pose_quality
        info["reward_leg_home_pose"] = reward_leg_home_pose

        # 8c. Pelvis angular velocity (for spinning detection in eval metrics)
        pelvis_angular_vel, pelvis_yaw_vel = self._compute_pelvis_diagnostics()
        info["pelvis_angular_vel"] = pelvis_angular_vel
        info["pelvis_yaw_vel"] = pelvis_yaw_vel

        # 8d. Spin penalty
        reward_spin, spin_instability = self._compute_angular_velocity_penalty(self.spin_penalty_weight)
        info["spin_instability"] = spin_instability
        info["reward_spin"] = reward_spin

        # 9. Gait symmetry (reward alternating foot contacts, shared helper),
        # on the substep-MIN foot forces read in 1d.
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

        # 10b. Saturation cost.  A command pinned at a range limit is
        # invisible to the smoothness/jerk penalties above -- it cannot
        # oscillate -- so parking joints at stops was the cheapest way to be
        # smooth.  Price the parked fraction directly.
        reward_action_saturation, action_saturation = _reward_action_saturation_pure(
            action, self.action_saturation_weight, self.action_saturation_threshold
        )
        info["action_saturation"] = float(action_saturation)
        info["reward_action_saturation"] = float(reward_action_saturation)

        # 11-12. Heading alignment and lateral velocity penalty
        reward_heading, reward_lateral = self._heading_terms(info, pelvis_quat, forward_ref_2d, vel_2d)

        # 13. Speed penalty above its threshold, idle penalty below its own
        reward_speed, reward_idle = self._speed_terms(info, vel_2d)

        # Total reward
        total_reward = (
            reward_forward
            + reward_backward
            + reward_drift
            + reward_alive
            + reward_bilateral_support
            + reward_energy
            + reward_tail
            + reward_strike
            + reward_approach
            + reward_proximity
            + reward_claw_proximity
            + reward_posture
            + reward_nosedive
            + reward_leg_home_pose
            + reward_spin
            + reward_gait
            + reward_smoothness
            + reward_action_jerk
            + reward_action_saturation
            + reward_heading
            + reward_lateral
            + reward_speed
            + reward_idle
        )
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        """Check if episode should terminate."""
        # Height/tilt, then nosedive termination (shared).  The 0.5 nosedive
        # margin stays a literal: a constructor knob would add its default to
        # the effective config that task_sha256 hashes.
        terminated, info = self._root_termination(self.pelvis_id, "pelvis_height", 0.5)
        if terminated:
            return True, info

        # Success: sickle claw contacted prey (only terminate when striking is rewarded)
        claw_geoms = {self.r_claw_geom_id, self.l_claw_geom_id}
        if self.strike_bonus > 0 and self._contact_geom(claw_geoms, self.prey_geom_id) is not None:
            info["termination_reason"] = "strike_success"
            info["success"] = True
            return True, info

        # Floor contact termination (shared)
        terminated, reason = self._check_floor_contact(
            self._body_ground_geoms,
            self.floor_geom_id,
            geom_categories={"tail": self._tail_ground_geoms},
        )
        if terminated:
            info["termination_reason"] = reason
            return True, info

        return False, info

    def _spawn_target(self):
        """Spawn prey at random location ahead of raptor."""
        prey_pos = self._spawn_target_2d(self.prey_distance_range, self.prey_lateral_range, 0.3)
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
    id="MesozoicLabs/Raptor-v0",
    entry_point="environments.velociraptor.envs.raptor_env:RaptorEnv",
    max_episode_steps=1000,
)
