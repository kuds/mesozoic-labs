"""
Velociraptor Gymnasium Environment

A bipedal dinosaur locomotion environment with predatory strike behavior.

Observation space (bipedal-target-heading-free/v1 since policy interface r13, the
maintainer's decision of 2026-11-06; the life-size plant of physics r5 and
policy interface r14, the maintainer's decision of 2026-11-08, 71 values; the
total dimension is generated in the public species catalog):
    - Joint positions (qpos[7:]) — 23 hinge joints excluding root freejoint, in
      qpos order: neck pitch, neck yaw, head pitch; tail 1 pitch, tail 1 yaw,
      tail 2 pitch, tail 3 pitch; then per leg (right, left) hip pitch, hip roll,
      knee, ankle, digit II (MTP-II, passive), sickle claw, digit III and digit IV
      (MTP, passive)
    - Joint velocities (qvel[6:]) — the same 23 joints
    - Pelvis projected gravity (unit down vector in pelvis coordinates) — 3
    - Pelvis angular velocity (gyroscope, its own site frame) — 3
    - Pelvis linear velocity in the gravity-aligned heading frame (forward, left, up) — 3
    - Pelvis acceleration (accelerometer, its own site frame) — 3
    - Foot contact — 6, one channel per load-bearing site since policy interface
      r14: the right foot's MTP pad, digit III and digit IV touch sensors, then
      the left foot's (the per-foot sums they replace are their totals)
    - Prey direction in pelvis coordinates (unit vector) — 3
    - Prey distance (scalar) — 1
    - Body-relative command (v_x_cmd, v_y_cmd, yaw_rate_cmd; zeros under command_mode = "none") — 3

No channel carries the world heading: the three root blocks come from the
free root's own quaternion through environments/shared/heading_observation.py
(docs/HEADING_INVARIANCE_DESIGN_2026_10.md), as on the T-Rex and the
anatomical Compsognathus (D-D31), so turning the whole scene about vertical
leaves the observation unchanged.

Action space (17 position servos since physics r5; the total dimension is
generated in the public species catalog):
    - Actions are residuals around the named XML ``home`` keyframe controls
    - Zero commands home; -1/+1 command each actuator's lower/upper limit
    - Neck: pitch, yaw; head pitch (3)
    - Right leg: hip pitch/roll, knee, ankle (4)
    - Right sickle claw (1, bounded at 4 N.m)
    - Left leg: hip pitch/roll, knee, ankle (4)
    - Left sickle claw (1)
    - Tail: pitch 1, yaw 1, pitch 2, pitch 3 (4)
    The forelimbs are welded passive bodies (no joints, no actuators), and
    digits II-IV hinge passively on springs, as on the T-Rex.

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
    - Posture (continuous deviation from the natural pitch, level since physics r5)
    - Nosedive penalty
    - Gait symmetry (alternating foot contacts)
    - Action smoothness (penalize jerky action changes)
    - Action jerk (penalize the second difference of actions, i.e. chatter)
    - Action saturation (penalize commands parked at their range limits)
    - Leg home pose (soft retention of the home keyframe leg stance)
    - Tail home pose (the tail joints around the statue's settled tail; off by default)
    - Neck home pose (the neck and head joints around the statue's settled
      carriage; off by default)
    - Floor impact (summed foot force above a body-weight multiple, the excess
      capped) and airborne substeps, read per physics substep (off by default)
    - Spin penalty (penalize pelvis angular velocity)
    - Heading alignment (facing toward prey)
    - Lateral velocity penalty (anti crab-walk)
    - Speed penalty (penalise absolute speed above threshold)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.heading_observation import heading_free_root_segments
from environments.shared.reward_functions import (
    reward_action_saturation as _reward_action_saturation_pure,
)
from environments.shared.reward_functions import (
    reward_bilateral_support as _reward_bilateral_support_pure,
)
from environments.shared.reward_functions import (
    reward_soft_home_pose as _reward_soft_home_pose_pure,
)


class VelociraptorMongoliensisEnv(BaseDinoEnv):
    """Velociraptor locomotion and strike environment.

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
    HEIGHT_LABEL = "Pelvis height"
    BEHAVIOR_SECTION_LABEL = "Hunting"
    # The raptor env only emits strike_success (bite_success was
    # impossible and was removed).
    SUCCESS_KEYS = ("strike_success",)

    # SB3-only since policy-interface revision 11 (plant_versions note 14).
    # The foot observation summed three touch sensors per foot, and the frozen
    # MJX registration (D-D17) reads the toe-d3 sensor alone and is never
    # edited, so the two backends would observe different feet; KNOWN_ISSUES
    # and CLEANUP_PLAN_2026_09 section 4.1 prescribe exactly this exit at the
    # species' next policy-interface revision.  The heading-free observation
    # (r13) and the per-site foot channels (r14) have no MJX builder either.
    supported_training_backends = ("stable-baselines3",)
    action_mapping = "home-keyframe-residual/v1"
    # 2.0 m on the r4 plant, scaled by the footprint radius (every animal
    # geom's horizontal reach from the root plus its bounding radius, at the
    # home keyframe: 0.776 -> 0.901 m on the life-size plant of physics r5),
    # rounded: the whole animal stays in frame.
    _camera_distance = 2.3
    _camera_azimuth = 135
    _camera_elevation = -20
    _camera_track_body = "pelvis"
    #: Opt-in task options carved out of the task fingerprint and of the
    #: effective config ``stage_config.json`` records while off: each entry is
    #: (switch, off value, the kwargs the switch makes inert).  Off, they change
    #: no reward, info key, trajectory or reset draw, so a stage that leaves
    #: them off keeps its task_sha256 (task_fingerprint.inert_opt_in_env_kwargs,
    #: the T-Rex's lateral-lean pattern).  The stance stage turns the first
    #: three on (the maintainer's decision of 2026-11-06); locomotion, strike and the behaviors leave them off.
    #: The neck home pose joined them with the life-size plant (physics r5,
    #: the maintainer's decision of 2026-11-08).
    OPT_IN_TASK_KWARGS = (
        ("tail_home_pose_weight", 0.0, ("tail_home_pose_weight", "tail_home_pose_tolerance")),
        ("floor_impact_weight", 0.0, ("floor_impact_weight", "floor_impact_threshold_bw")),
        ("airborne_substep_weight", 0.0, ("airborne_substep_weight",)),
        ("neck_home_pose_weight", 0.0, ("neck_home_pose_weight", "neck_home_pose_tolerance")),
    )
    #: The tail joints the tail_home_pose term prices, and their target: the
    #: noise-free statue's post-settle pose (seed 0, steps 200-999;
    #: TestSettledTailTargetTracksTheStatue re-measures it).  Since physics r5
    #: each tail pitch joint carries its gravity torque in a springref preload
    #: (the T-Rex's tail since its physics r10, D-D44), so the servos hold
    #: about 0 N.m at home and the statue settles on the keyframe zeros
    #: instead of drooping (r4: 4.5 deg down at tail_1, -0.07851 / 0.0 /
    #: -0.04639 / -0.02307 rad).  Measured on r5 by the protocol above (env
    #: defaults, reset noise 0, seed 0): every joint within 4.3e-7 rad of
    #: zero, and the 5000-step roll's last 1000 steps within 6.9e-7; each
    #: joint 0.209 rad (tail_3: 0.175) clear of its stops.
    #: PROVISIONAL until the physics-r5 statue re-measurement that moves
    #: statue_constants_physics_revision to 5 (the reset-noise panels).
    _TAIL_HOME_JOINTS = ("tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch")
    _TAIL_SETTLED_QPOS = (0.0, 0.0, 0.0, 0.0)
    #: The neck and head joints the neck_home_pose term prices (the T-Rex's
    #: neck_posture joints), and their target: the noise-free statue's
    #: post-settle carriage, measured the way _TAIL_SETTLED_QPOS is.  The neck
    #: and head servos carry a gravity preload in the home ctrl (ctrl - qpos =
    #: tau_g / kp, -4.665 / -4.628 deg on neck_pitch / head_pitch), so the
    #: statue holds the reconstructed carriage, where the T-Rex's unpreloaded
    #: neck settles 12 deg low (its _NECK_SETTLED_QPOS): -2.78e-5 / 0.0 /
    #: 1.69e-5 rad (steps 200-999; steps 4000-4999 of a 5000-step roll agree
    #: to 7e-7), 0.35 rad or more clear of every stop.
    #: PROVISIONAL, as _TAIL_SETTLED_QPOS above.
    _NECK_HOME_JOINTS = ("neck_pitch", "neck_yaw", "head_pitch")
    _NECK_SETTLED_QPOS = (-0.00003, 0.0, 0.00002)
    #: The observation's per-site foot channels (BaseDinoEnv declares the
    #: attribute, None on every other species); set in _cache_ids.
    _foot_observation_groups: tuple[tuple[int, ...], ...]
    # The neck never collides since physics r5 (contype/conaffinity 0, as the
    # T-Rex's); the behavior env ends an episode when either neck capsule
    # penetrates the terrain.  Inert in this env.
    _terrain_contact_probe_geoms = ("neck_geom", "neck_upper_geom")
    #: A foot (or the pair) reads as airborne at or below this touch force, in
    #: N: the floor-truth library's contact threshold (gait.constants
    #: CONTACT_THRESHOLD_N, which a species env may not import).
    _AIRBORNE_TOUCH_N = 0.1
    #: The floor-impact term charges a step's peak at most this far above its
    #: threshold, in body weights (the anatomical Compsognathus's cap): a hop
    #: landing reads 11-12.6 body weights on this plant (the r3 pilots), and an
    #: uncapped charge would out-price the -50 fall penalty within a few
    #: landings, paying the policy to fall sooner rather than land softer.
    _FLOOR_IMPACT_EXCESS_CAP_BW = 1.0

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
        natural_pitch: float = 0.00234,
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
        # Stance-quality terms, named as on TyrannosaurusRexEnv.  Every default is inert:
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
        # The T-Rex's D-D27 contact terms and its tail pose term (
        # the maintainer's decision of 2026-11-06), each off by default and carved out of the task fingerprint
        # while off (OPT_IN_TASK_KWARGS): a zero weight reproduces the reward
        # that predates them bit for bit and emits none of their info keys.
        tail_home_pose_weight: float = 0.0,
        tail_home_pose_tolerance: float = 0.05,
        # The neck and head carriage term of the life-size plant (physics r5,
        # the maintainer's decision of 2026-11-08), shaped as the tail term and
        # likewise off by default and carved out while off.
        neck_home_pose_weight: float = 0.0,
        neck_home_pose_tolerance: float = 0.05,
        floor_impact_weight: float = 0.0,
        floor_impact_threshold_bw: float = 3.0,
        airborne_substep_weight: float = 0.0,
        # Environment settings
        prey_distance_range: tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: tuple[float, float] = (-2.0, 2.0),
        # The pelvis-height fall / launch bounds: the r4 plant's (0.3, 1.0) m,
        # 0.593 / 1.978 of its 0.5055 m statue, scaled with the life-size hip
        # height (0.5488 m at the physics-r5 keyframe, x 1.0856) and rounded to
        # the centimetre.
        healthy_z_range: tuple[float, float] = (0.33, 1.09),
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
        self.tail_home_pose_weight = tail_home_pose_weight
        self.tail_home_pose_tolerance = tail_home_pose_tolerance
        self.neck_home_pose_weight = neck_home_pose_weight
        self.neck_home_pose_tolerance = neck_home_pose_tolerance
        self.floor_impact_weight = floor_impact_weight
        self.floor_impact_threshold_bw = floor_impact_threshold_bw
        self.airborne_substep_weight = airborne_substep_weight

        if not 0.0 <= self.tail_home_pose_weight < np.inf:
            raise ValueError("tail_home_pose_weight must be finite and non-negative")
        if not 0.0 < self.tail_home_pose_tolerance < np.inf:
            raise ValueError("tail_home_pose_tolerance must be a positive, finite angle")
        if not 0.0 <= self.neck_home_pose_weight < np.inf:
            raise ValueError("neck_home_pose_weight must be finite and non-negative")
        if not 0.0 < self.neck_home_pose_tolerance < np.inf:
            raise ValueError("neck_home_pose_tolerance must be a positive, finite angle")
        if not 0.0 <= self.floor_impact_weight < np.inf:
            raise ValueError("floor_impact_weight must be finite and non-negative")
        if not 0.0 < self.floor_impact_threshold_bw < np.inf:
            raise ValueError("floor_impact_threshold_bw must be a positive, finite multiple of body weight")
        if not 0.0 <= self.airborne_substep_weight < np.inf:
            raise ValueError("airborne_substep_weight must be finite and non-negative")
        if self.foot_contact_saturation_force <= 0.0:
            raise ValueError("foot_contact_saturation_force must be positive")
        if not 0.0 <= self.support_conditioned_alive_fraction <= 1.0:
            raise ValueError("support_conditioned_alive_fraction must be in [0, 1]")
        if not 0.0 <= self.action_saturation_threshold < 1.0:
            raise ValueError("action_saturation_threshold must be in [0, 1)")
        if self.leg_home_pose_tolerance <= 0.0:
            raise ValueError("leg_home_pose_tolerance must be positive")

        # Natural pitch of the pelvis frame, nose-down positive.  Posture
        # shaping, the nosedive penalty and nosedive termination are measured
        # relative to this angle, so the statue is not charged for the stance
        # it holds.  Measured, not authored: the r2 plant settled at 24.0 deg
        # and its statue paid -95 per episode of nosedive charge for standing
        # still against the 0.35 rad of the time; the r3 / r4 statues leaned
        # 20.10 / 20.02 deg (0.3508 / 0.3494 rad) and 0.35 was right.  The
        # life-size plant of physics r5 authors the trunk level (sacrum
        # horizontal, its home keyframe) and its noise-free statue settles
        # 0.134 deg nose-down, 0.00234 rad (TestNaturalPitchTracksStance's
        # protocol: env defaults, reset noise 0, seed 0, 600 zero-action
        # steps).  PROVISIONAL until the physics-r5 statue re-measurement.
        # With it the 0.5 nosedive margin of _is_terminated ends an episode at
        # forward_z < -0.502, 30.1 deg nose-down, where the T-Rex stance's
        # 0.503 ends at 31.3 deg.  TestNaturalPitchTracksStance pins the
        # default to the settled stance.
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

        # Leg-pose target for the leg_home_pose term: the named home keyframe,
        # which is the standing equilibrium since physics revision 3 (springs
        # anchored there, servos preloaded against gravity) -- the noise-free
        # physics-r5 statue settles within 0.003 deg of it.  The eight
        # servoed leg joints, hips to ankles, as on the T-Rex: since physics r5
        # digits II-IV are passive springs whose angle the ground sets (r4
        # priced its servoed digits III and IV here too), and the sickle claw
        # is the strike's.  Resolved here rather than in _cache_ids, whose
        # tokens are part of this SB3-only species' policy-interface digest:
        # a reward target is not part of what a checkpoint observes.
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
        self._leg_home_qpos_indices = self._joint_qpos_indices(leg_home_joint_names)
        self._leg_home_qpos = self.model.key_qpos[self.home_keyframe_id][self._leg_home_qpos_indices].copy()
        # The tail_home_pose term's joints and its statue-measured target
        # (_TAIL_SETTLED_QPOS), resolved here for the same reason as the leg
        # target above.  The term prices JOINT positions: tail_stability reads
        # only the tail-tip angular velocity, which a tail parked motionless
        # against a stop satisfies perfectly (the T-Rex's 2026-08
        # narrow-tolerance run parked three tail joints that way, and this
        # species' r2 audit stance held tail_1_pitch at -15 deg, its stop, on
        # 95% of steps).
        self._tail_home_qpos_indices = self._joint_qpos_indices(self._TAIL_HOME_JOINTS)
        self._tail_home_qpos = np.asarray(self._TAIL_SETTLED_QPOS, dtype=np.float64)
        # The neck_home_pose term's joints and its statue-measured target
        # (_NECK_SETTLED_QPOS), resolved here for the same reason.  The
        # posture term reads only the pelvis frame, so a policy could carry the
        # head anywhere for free; this prices the carriage itself, centred on
        # what the statue holds so the statue keeps full quality.
        self._neck_home_qpos_indices = self._joint_qpos_indices(self._NECK_HOME_JOINTS)
        self._neck_home_qpos = np.asarray(self._NECK_SETTLED_QPOS, dtype=np.float64)
        # The animal's weight in N, for the floor-impact term: the pelvis
        # subtree is the whole animal (the prey is a separate mocap body), so
        # this is the 15.0 kg x 9.81 = 147.15 N (physics r5; r4 13.5 kg,
        # 132.4 N) the floor-truth library and the stance gate's peak-force
        # bars divide by.
        self._body_weight_n = float(self.model.body_subtreemass[self.pelvis_id] * abs(self.model.opt.gravity[2]))

    def _joint_qpos_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        """Resolve scalar hinge-joint qpos addresses, failing on model drift."""
        indices = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Velociraptor model must define joint {name!r}")
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(indices, dtype=np.int32)

    def _required_id(self, kind: mujoco.mjtObj, name: str) -> int:
        """Resolve a named MuJoCo element, failing on model drift.

        ``mj_name2id`` returns -1 for a missing name and nothing downstream
        notices: a renamed claw geom silently disables the strike, a renamed
        head geom silently drops out of the termination set, and
        ``site_xpos[-1]`` silently reads the model's LAST site.  Every name
        ``_cache_ids`` reads goes through here.
        """
        element_id = mujoco.mj_name2id(self.model, kind, name)
        if element_id < 0:
            raise ValueError(f"Velociraptor model must define {kind.name.removeprefix('mjOBJ_').lower()} {name!r}")
        return int(element_id)

    def _require_sensor_address(self, name: str, address: int) -> None:
        """Fail unless the named sensor starts at *address* of sensordata (the shared IMU block's positions)."""
        actual = int(self.model.sensor_adr[self._required_id(mujoco.mjtObj.mjOBJ_SENSOR, name)])
        if actual != address:
            raise ValueError(f"Velociraptor sensor {name!r} must start at sensordata {address}, not {actual}")

    def _touch_sensor_address(self, name: str) -> int:
        """The sensordata address of the named touch sensor, failing on model drift."""
        sensor_id = self._required_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        if int(self.model.sensor_type[sensor_id]) != int(mujoco.mjtSensor.mjSENS_TOUCH):
            raise ValueError(f"Velociraptor sensor {name!r} must be a touch sensor")
        return int(self.model.sensor_adr[sensor_id])

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, sites and sensors, every one by name and failing closed."""
        # The raptor policy commands residuals around the biomechanically
        # balanced XML home pose.  Cache the controls once so action zero can
        # preserve that pose without a keyframe lookup on every environment
        # step.
        self._cache_home_keyframe("Velociraptor")
        body, geom, site = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE

        # Body IDs
        self.pelvis_id = self._required_id(body, "pelvis")
        self.skull_body_id = self._required_id(body, "skull")

        # Geom IDs for contact detection.  The sickle claws are contype /
        # conaffinity 2/2 against the floor's 1/1: they meet only the prey.
        self.prey_geom_id = self._required_id(geom, "prey_geom")
        self.r_claw_geom_id = self._required_id(geom, "r_claw_geom")
        self.l_claw_geom_id = self._required_id(geom, "l_claw_geom")
        self.floor_geom_id = self._required_id(geom, "floor")
        # The trunk since physics r5: the dorsal torso, the low belly and the
        # pectoral ribcage capsules (r4 had the torso alone).
        self.torso_geom_id = self._required_id(geom, "torso")
        self.belly_geom_id = self._required_id(geom, "belly")
        self.ribcage_geom_id = self._required_id(geom, "ribcage")
        # The head's colliding geoms, on the separate skull body since physics
        # r5 (the T-Rex's names); the neck capsules and the mandible never
        # collide (contype/conaffinity 0), so they produce no floor contact.
        self.skull_upper_geom_id = self._required_id(geom, "skull_upper")
        self.snout_geom_id = self._required_id(geom, "snout")

        # Tail geom IDs (distal segments that should not contact floor)
        self.tail_3_geom_id = self._required_id(geom, "tail_3_geom")
        self.tail_4_geom_id = self._required_id(geom, "tail_4_geom")
        self.tail_5_geom_id = self._required_id(geom, "tail_5_geom")

        # Geoms that should terminate the episode on ground contact
        self._body_ground_geoms = {
            self.torso_geom_id,
            self.belly_geom_id,
            self.ribcage_geom_id,
            self.skull_upper_geom_id,
            self.snout_geom_id,
            self.tail_3_geom_id,
            self.tail_4_geom_id,
            self.tail_5_geom_id,
        }
        self._head_ground_geoms = {self.skull_upper_geom_id, self.snout_geom_id}
        self._tail_ground_geoms = {self.tail_3_geom_id, self.tail_4_geom_id, self.tail_5_geom_id}

        # Site IDs for sensors
        self.imu_site_id = self._required_id(site, "imu")
        self.r_foot_site_id = self._required_id(site, "r_foot")
        self.l_foot_site_id = self._required_id(site, "l_foot")
        self.tail_tip_site_id = self._required_id(site, "tail_tip")
        self.head_tip_site_id = self._required_id(site, "head_tip")
        # Height terminations sampled at every physics substep by the base
        # step loop (the T-Rex's): _is_terminated reads the per-check MIN so a
        # head dip between the control-step samples still terminates.  Order
        # is consumed by index there.
        self._substep_height_checks = (
            ("site", self.head_tip_site_id),
            ("body", self.skull_body_id),
        )

        # Mocap body for prey
        self.prey_mocap_id = self._required_id(body, "prey")

        # Claw tip site IDs (for claw-to-prey proximity shaping)
        self.r_claw_tip_site_id = self._required_id(site, "r_claw_tip")
        self.l_claw_tip_site_id = self._required_id(site, "l_claw_tip")

        # Sensor addresses, resolved by NAME since physics r5 (r4 hard-coded
        # them by position).  The shared IMU block stays at the addresses
        # BaseDinoEnv's _sensor_gyro_start, _sensor_accel_start and
        # _sensor_quat_start (0, 3, 6) declare, which shared code reads.
        self._require_sensor_address("pelvis_gyro", self._sensor_gyro_start)
        self._require_sensor_address("pelvis_accel", self._sensor_accel_start)
        self._require_sensor_address("pelvis_orientation", self._sensor_quat_start)
        # One touch sensor per load-bearing body (a touch sensor sums only the
        # contacts on geoms of its site's own body): the MTP pad on the
        # metatarsus (the r/l_foot site), digit III and digit IV.  At the
        # settled stance their sum is the floor force under the foot (the
        # metatarsus shaft and the raised digit II never touch).
        self._sensor_r_foot = self._touch_sensor_address("r_foot_touch")
        self._sensor_l_foot = self._touch_sensor_address("l_foot_touch")
        self._sensor_r_foot_digits = (
            self._touch_sensor_address("r_toe_d3_touch"),
            self._touch_sensor_address("r_toe_d4_touch"),
        )
        self._sensor_l_foot_digits = (
            self._touch_sensor_address("l_toe_d3_touch"),
            self._touch_sensor_address("l_toe_d4_touch"),
        )
        # Per-foot groups for the base class's substep MIN aggregation, pad
        # first (the pinned group[0] + sum(rest) order).  They feed the
        # contact-shaped rewards and the r/l_foot_contact info keys.
        self._foot_sensor_groups = (
            (self._sensor_r_foot, *self._sensor_r_foot_digits),
            (self._sensor_l_foot, *self._sensor_l_foot_digits),
        )
        # The observation's foot channels since policy interface r14: one per
        # touch sensor, in the per-foot groups' order (right pad, d3, d4, then
        # left), so a policy sees where under each foot its load sits.  #608's
        # per-site channels, which plant_versions note 21 deferred to the next
        # change that strands every checkpoint; the plant contract records each
        # channel's sensor in the foot_contact segment.
        self._foot_observation_groups = tuple((address,) for group in self._foot_sensor_groups for address in group)

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
        """Construct the bipedal-target-heading-free/v1 observation vector (71 values since policy interface r14).

        Since policy interface r13 (the maintainer's decision of 2026-11-06) no channel carries the
        world heading: the pelvis orientation is projected gravity, the pelvis
        linear velocity is expressed in the gravity-aligned heading frame and
        the prey direction in pelvis coordinates, all three from the free
        root's own quaternion qpos[3:7] (environments/shared/heading_observation.py),
        never from the orientation sensor, which is still zero when the
        constructor calls this before any forward pass.  The T-Rex's layout
        (D-D31), on this species' joints and feet, with one foot-contact
        channel per load-bearing site (``_foot_observation_groups``) since
        policy interface r14 (the maintainer's decision of 2026-11-08).
        """
        # Joint positions (exclude root freejoint: first 7 values are pos + quat)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values are lin + ang vel)
        qvel = self.data.qvel[6:].copy()

        # Pelvis state from sensors, already in their own site frame (never rotated again)
        pelvis_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3].copy()
        pelvis_accel = self.data.sensordata[self._sensor_accel_start : self._sensor_accel_start + 3].copy()

        # Foot contact, one channel per touch sensor (MTP pad, digit III,
        # digit IV; right foot first), each its group's sum in the pinned
        # group[0] + sum(rest) order of BaseDinoEnv._foot_contact_forces.
        sensordata = self.data.sensordata
        foot_contact = np.array(
            [
                sensordata[group[0]] + sum(sensordata[index] for index in group[1:])
                for group in self._foot_observation_groups
            ]
        )

        # Prey info (relative to pelvis)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]  # First (and only) mocap body
        prey_rel = prey_pos - pelvis_pos
        prey_distance = np.linalg.norm(prey_rel)

        # Normalize prey direction (world frame; turned into pelvis coordinates below)
        prey_direction = prey_rel / (prey_distance + 1e-8)

        # Heading-free root block: projected gravity, the root's world linear
        # velocity (free joint qvel[0:3]) in the heading frame, the prey
        # direction in pelvis coordinates.
        gravity, heading_velocity, prey_direction_body = heading_free_root_segments(
            self.data.qpos, self.data.qvel[0:3], prey_direction
        )

        obs = np.concatenate(
            [
                qpos,  # Joint positions
                qvel,  # Joint velocities
                gravity,  # Projected gravity (unit down vector, pelvis frame)
                pelvis_gyro,  # Angular velocity
                heading_velocity,  # Linear velocity (forward, left, up)
                pelvis_accel,  # Accelerometer
                foot_contact,  # Foot contacts (6: per site)
                prey_direction_body,  # Direction to prey (unit vector, pelvis frame)
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
        # sum (the MTP pad + d3 + d4 since physics r5; r3 / r4 d3 + metatarsus
        # + d4) IS the floor force under that foot, so
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

        # 7. Continuous posture reward centred on the natural pitch (the
        # statue's settled pelvis pitch; 0.134 deg nose-down since physics r5).
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

        # 8b3. Tail home pose (off unless tail_home_pose_weight > 0; its info
        # keys exist only then).  The same soft Gaussian as the leg term, on
        # the four servoed tail joints around the statue's settled tail
        # (_TAIL_SETTLED_QPOS).  The tail stability term (4) reads only the
        # tail-tip gyro, so a tail frozen against a stop is optimal for it;
        # this prices the joint positions themselves (the T-Rex's term, 2026-11-06).
        reward_tail_home_pose = 0.0
        if self.tail_home_pose_weight > 0.0:
            _, tail_rms_error, tail_quality = _reward_soft_home_pose_pure(
                self.data.qpos[self._tail_home_qpos_indices],
                self._tail_home_qpos,
                self.tail_home_pose_tolerance,
                1.0,
            )
            reward_tail_home_pose = self.tail_home_pose_weight * float(tail_quality)
            info["tail_home_pose_error"] = float(tail_rms_error)
            info["tail_home_pose_quality"] = float(tail_quality)
            info["reward_tail_home_pose"] = reward_tail_home_pose

        # 8b4. Neck home pose (off unless neck_home_pose_weight > 0; its info
        # keys exist only then).  The same soft Gaussian as the tail term, on
        # the neck and head joints around the statue's settled carriage
        # (_NECK_SETTLED_QPOS; the maintainer's decision of 2026-11-08).
        reward_neck_home_pose = 0.0
        if self.neck_home_pose_weight > 0.0:
            _, neck_rms_error, neck_quality = _reward_soft_home_pose_pure(
                self.data.qpos[self._neck_home_qpos_indices],
                self._neck_home_qpos,
                self.neck_home_pose_tolerance,
                1.0,
            )
            reward_neck_home_pose = self.neck_home_pose_weight * float(neck_quality)
            info["neck_home_pose_error"] = float(neck_rms_error)
            info["neck_home_pose_quality"] = float(neck_quality)
            info["reward_neck_home_pose"] = reward_neck_home_pose

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

        # 9b. Floor impact and airborne substeps (each off unless its weight is
        # positive; its info keys exist only then), from every physics
        # substep's per-foot touch (BaseDinoEnv.step keeps the block beside the
        # MIN the support terms read).  Since physics r3 the per-foot touch sum
        # IS the floor force under that foot, and nothing but the feet touches
        # the floor while the animal stands, so the summed touch is the
        # animal's floor force.  The impact term prices the step's PEAK above
        # floor_impact_threshold_bw body weights, the excess capped at
        # _FLOOR_IMPACT_EXCESS_CAP_BW; the airborne term the share of the
        # step's substeps on which every foot is unloaded.  Both run over the
        # whole episode, settle included, where the support terms' step-level
        # MIN and the gate's window let a stomp or a two-foot hop go unpriced
        # (the T-Rex's D-D27 terms, here in the capped form of the anatomical
        # Compsognathus; the maintainer's decision of 2026-11-06).
        reward_floor_impact = 0.0
        reward_airborne_substeps = 0.0
        if self.floor_impact_weight > 0.0 or self.airborne_substep_weight > 0.0:
            foot_force_block = self._substep_foot_force_block()
            if self.floor_impact_weight > 0.0:
                peak_foot_force_bw = float(foot_force_block.sum(axis=1).max()) / self._body_weight_n
                impact_excess = min(
                    max(0.0, peak_foot_force_bw - self.floor_impact_threshold_bw), self._FLOOR_IMPACT_EXCESS_CAP_BW
                )
                reward_floor_impact = -self.floor_impact_weight * impact_excess
                info["peak_foot_force_bw"] = peak_foot_force_bw
                info["reward_floor_impact"] = reward_floor_impact
            if self.airborne_substep_weight > 0.0:
                airborne_substeps = int(np.count_nonzero(np.all(foot_force_block <= self._AIRBORNE_TOUCH_N, axis=1)))
                reward_airborne_substeps = -self.airborne_substep_weight * airborne_substeps / foot_force_block.shape[0]
                info["airborne_substeps"] = float(airborne_substeps)
                info["reward_airborne_substeps"] = reward_airborne_substeps

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
        # Added only when enabled, so the legacy sum keeps its exact order.
        if self.tail_home_pose_weight > 0.0:
            total_reward += reward_tail_home_pose
        if self.neck_home_pose_weight > 0.0:
            total_reward += reward_neck_home_pose
        if self.floor_impact_weight > 0.0:
            total_reward += reward_floor_impact
        if self.airborne_substep_weight > 0.0:
            total_reward += reward_airborne_substeps
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

        # Head-height terminations (the T-Rex's, since physics r5 gave the
        # raptor its own skull body and head_tip site), each read at the
        # substep MIN (indices match _substep_height_checks) so a dip that
        # recovers between control-boundary samples still terminates; the info
        # key keeps the boundary sample.  The thresholds are the T-Rex's 0.12 m
        # and 0.45 m at the same fraction of the home height they guard
        # (0.098 of its 1.227 m head tip, 0.367 of its 1.227 m skull origin):
        # 0.098 x 0.724 m = 0.071 and 0.367 x 0.756 m = 0.278 on this plant,
        # rounded.  At 0.07 m with the head level the snout sits 0.047 m and
        # the visual mandible 0.032 m above the floor.
        head_tip_z = self._clearance(self.data.site_xpos[self.head_tip_site_id])
        info["head_tip_z"] = head_tip_z
        if self._aggregated_min_height(0, head_tip_z) < 0.07:
            info["termination_reason"] = "head_contact"
            return True, info
        skull_z = self._clearance(self.data.xpos[self.skull_body_id])
        if self._aggregated_min_height(1, skull_z) < 0.28:
            info["termination_reason"] = "skull_low"
            return True, info

        # Success: sickle claw contacted prey (only terminate when striking is rewarded)
        claw_geoms = {self.r_claw_geom_id, self.l_claw_geom_id}
        if self.strike_bonus > 0 and self._contact_geom(claw_geoms, self.prey_geom_id) is not None:
            info["termination_reason"] = "strike_success"
            info["success"] = True
            return True, info

        # Floor contact termination (shared): tail_contact, head_contact for
        # the skull's geoms (the T-Rex's category), body_contact for the trunk.
        terminated, reason = self._check_floor_contact(
            self._body_ground_geoms,
            self.floor_geom_id,
            geom_categories={"tail": self._tail_ground_geoms, "head": self._head_ground_geoms},
        )
        if terminated:
            info["termination_reason"] = reason
            return True, info

        return False, info

    def _spawn_target(self):
        """Spawn prey at random location ahead of raptor."""
        # The 0.15 m sphere's centre at 0.3 m, as on the r4 plant: the strike is
        # claw-to-sphere, and the life-size claw rests where r4's did (tip
        # 0.068 m high at home, r4 0.071) and reaches as high (0.76 m over its
        # leg's joint ranges, r4 0.80), so the sphere's lower edge stays 0.08 m
        # above the resting claw.
        prey_pos = self._spawn_target_2d(self.prey_distance_range, self.prey_lateral_range, 0.3)
        self._initial_prey_dir_2d = self._compute_initial_direction_2d(prey_pos)
        self._initial_pos_2d = self.data.qpos[0:2].copy()

        # Reset delta-based tracking (first step will produce zero deltas)
        self._prev_prey_distance = None
        self._prev_action = None
        self._prev_prev_action = None

        # Reset gait symmetry tracking
        self._reset_gait_state()
