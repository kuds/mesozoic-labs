"""
Base Gymnasium environment for MuJoCo dinosaur simulations.

Provides the common lifecycle (init, step, reset, render, close) shared
across all dinosaur species. Subclasses override species-specific methods:
  - _cache_ids()
  - _get_obs()
  - _get_reward_info()
  - _is_terminated()
  - _spawn_target()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .constants import SENSOR_ACCEL_START, SENSOR_GYRO_START, SENSOR_QUAT_START, TAIL_ANGULAR_VEL_MAX
from .reward_functions import check_height_tilt_termination as _check_height_tilt_pure
from .reward_functions import quat_to_forward_2d as _quat_to_forward_2d_pure
from .reward_functions import quat_to_forward_z as _quat_to_forward_z_pure
from .reward_functions import quat_to_tilt as _quat_to_tilt_pure
from .reward_functions import reward_action_jerk as _reward_action_jerk_pure
from .reward_functions import reward_action_smoothness as _reward_action_smoothness_pure
from .reward_functions import reward_alive as _reward_alive_pure
from .reward_functions import reward_angular_velocity_penalty as _reward_angular_velocity_penalty_pure
from .reward_functions import reward_approach_shaping as _reward_approach_shaping_pure
from .reward_functions import reward_backward_penalty as _reward_backward_penalty_pure
from .reward_functions import reward_drift_penalty as _reward_drift_penalty_pure
from .reward_functions import reward_energy as _reward_energy_pure
from .reward_functions import reward_forward_velocity as _reward_forward_velocity_pure
from .reward_functions import reward_heading_alignment as _reward_heading_alignment_pure
from .reward_functions import reward_idle_penalty as _reward_idle_penalty_pure
from .reward_functions import reward_lateral_velocity_penalty as _reward_lateral_velocity_penalty_pure
from .reward_functions import reward_lean_aware_posture as _reward_lean_aware_posture_pure
from .reward_functions import reward_nosedive as _reward_nosedive_pure
from .reward_functions import reward_posture as _reward_posture_pure
from .reward_functions import reward_speed_penalty as _reward_speed_penalty_pure

# Clearance in METRES kept between a reset spawn and the nearer end of
# healthy_z_range.  Reset must never generate an already-terminal state: an
# episode that ends on step 1 regardless of the action is not a policy failure,
# and counting it as one puts an unreachable ceiling on any reliability gate.
# See BaseDinoEnv._bounded_reset_height_delta.
_RESET_HEIGHT_TERMINATION_MARGIN = 0.02

# Fallback settle target in METRES, used only when a species has no keyframe to
# read an authored contact depth from.  Reset normally settles to the home
# keyframe's OWN clearance so the noise-free reset stays bit-identical and each
# species keeps the resting contact depth its MJCF was authored with.
_RESET_GROUND_CLEARANCE = 0.0

# Upper bound in METRES on the body-to-floor distance probe.  Only the sign and
# small magnitudes matter for settling, so this just has to exceed any plausible
# spawn offset; mj_geomDistance saturates beyond it.
_GROUND_PROBE_DISTANCE = 10.0


class BaseDinoEnv(gym.Env, ABC):
    """Abstract base class for dinosaur locomotion environments.

    Species hook with a base implementation driven by ``_foot_sensor_groups``:

    ``_foot_contact_forces() -> tuple[float, ...]``
        INSTANTANEOUS total floor-contact force under each foot from the
        current ``data.sensordata``, in a stable per-species order, summing
        every sensor that sees that foot (T-Rex: plantar pad + three digits;
        brachiosaurus: pad + meta).  **The arity is the number of feet** -- 2
        on the bipeds, 4 on the quadrupeds -- so callers must either sum it
        or check ``len`` before unpacking.  Species declare their sensor
        groups via ``_foot_sensor_groups`` in ``_cache_ids``; a species with
        no groups gets an empty tuple, which is why consumers check the
        arity.  Kept an overridable METHOD (tests monkeypatch it to steer
        the contact-shaped rewards), and it stays the per-substep primitive:
        :meth:`step` calls it once per physics substep to build the
        aggregated value below.

    ``_aggregated_foot_contact_forces() -> tuple[float, ...]``
        The per-foot MIN across the ``frame_skip`` physics substeps of the
        current control step -- what the contact-shaped rewards, the
        ``*_foot_contact`` info keys, and therefore the stance-duty gate
        consume.  Physics runs at 1/``model.opt.timestep`` Hz while control
        runs ``frame_skip`` times slower; reading only the final substep let
        a control-clock-locked hop unload between samples and read as
        continuous support (the seed-43 stage-1 bounce measured exactly one
        unloaded sample per 5).  MIN over per-substep per-foot SUMS -- never
        a sum of per-sensor minima, which under-reports when load shifts
        between pad and digits within a control step.  Falls back to the
        instantaneous read when no step has run (reset, direct calls, the
        SB3/MJX single-state parity tests)."""

    # Ground-settling caches, all derived from the model and so fixed for the
    # lifetime of the instance.  Declared here rather than assigned via getattr
    # so their types are visible; see _settle_root_on_ground.
    _root_subtree_geom_ids: "np.ndarray | None" = None
    _static_floor_geom_ids: "np.ndarray | None" = None
    _home_ground_clearance_m: float | None = None

    # Per-foot touch-sensor groups (sensordata indices), declared per species
    # in _cache_ids; () means "no foot sensors".  Mirrors the MJX registry's
    # sensor_foot_indices + sensor_foot_aux_indices so both backends aggregate
    # the same sensors.
    _foot_sensor_groups: "tuple[tuple[int, ...], ...]" = ()

    # Substep aggregation state, populated by step()'s frame-skip loop.
    # Tagged with the _step_count it was measured at rather than cleared in
    # reset(): reset() zeroes _step_count anyway, which invalidates the tag,
    # and reset()'s SOURCE is fingerprinted by the plant contract's
    # home_reset interface -- touching it would bump every species' policy
    # interface for a bookkeeping detail.  None / a stale tag both mean "no
    # aggregate for the current step", and consumers fall back to the
    # instantaneous read (__init__ and reset call _get_obs() before any step
    # has run, and several tests score hand-posed states directly).
    _substep_min_foot_forces: "np.ndarray | None" = None
    _substep_floor_hit_geom: "int | None" = None
    _substep_contact_step: int = -1
    _ground_geom_array: "np.ndarray | None" = None

    # Site/body height checks aggregated per substep, declared per species in
    # _cache_ids as ("site" | "body", entity_id) pairs; () means "none".  The
    # step loop records each entity's MINIMUM z across the substeps so the
    # species' height terminations (trex head_tip/skull, dibothrosuchus
    # snout_tip) fire on a between-samples dip exactly like the MJX
    # height-emulation checks, which became any-substep with the contact
    # aggregation.  Info keys keep reporting the boundary sample.
    _substep_height_checks: "tuple[tuple[str, int], ...]" = ()
    _substep_min_heights: "np.ndarray | None" = None

    # Optional zero-argument callable invoked after EVERY physics substep in
    # step().  Exists so stance_duty_validation.py and the aggregation
    # regression tests can record per-substep kinematic ground truth through
    # the REAL step path instead of replicating the loop on a shadow env
    # that can silently drift from it.  None (the default) costs one
    # comparison per substep.
    _substep_probe_hook: "Callable[[], None] | None" = None

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # Subclasses override for camera positioning
    _camera_distance: float = 3.0
    _camera_azimuth: float = 135
    _camera_elevation: float = -20
    _camera_track_body: str | None = None  # Body name to track, or None for fixed

    def __init__(
        self,
        model_path: str,
        render_mode: str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        forward_vel_weight: float = 1.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        healthy_z_range: tuple[float, float] = (0.25, 1.0),
        max_tilt_angle: float = 1.047,
        reset_noise_scale: float = 0.01,
        reset_height_noise_scale: float | None = None,
    ):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # Distance tracking (cumulative XY path length)
        self._prev_pos_2d: np.ndarray = np.zeros(2)
        self._distance_traveled: float = 0.0

        # Common reward weights
        self.forward_vel_weight = forward_vel_weight
        self.alive_bonus = alive_bonus
        self.energy_penalty_weight = energy_penalty_weight
        self.fall_penalty = fall_penalty

        # Environment settings
        self.healthy_z_range = healthy_z_range
        self.max_tilt_angle = max_tilt_angle
        self.reset_noise_scale = reset_noise_scale
        # Root-height jitter at reset, in METRES.  ``reset_noise_scale`` is a
        # joint-angle scale in RADIANS, and reusing it for a length is only
        # harmless while the species is roughly a metre tall: on a 0.31 m
        # stance a 0.14 rad joint jitter becomes a 0.14 m height jitter, i.e.
        # 45% of standing height, which spawns a quarter of episodes already
        # outside healthy_z_range.  ``None`` keeps the historical coupled
        # behaviour so existing species' reset distributions are unchanged.
        self.reset_height_noise_scale = reset_height_noise_scale

        # Cache body/geom/site IDs (species-specific)
        self._cache_ids()

        # Define action space (normalized to [-1, 1])
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # Define observation space from initial obs
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        # Rendering
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None
        self._camera = None

    @staticmethod
    def _quat_to_tilt(quat: np.ndarray) -> float:
        """Compute tilt angle (radians) between body up-axis and world up.

        Args:
            quat: MuJoCo quaternion (w, x, y, z).

        Returns:
            Angle in radians between the body's Z-axis and world Z-axis.
            0 means perfectly upright, pi/2 means horizontal.
        """
        return float(_quat_to_tilt_pure(quat))

    # ------------------------------------------------------------------
    # Abstract methods: subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector."""

    @abstractmethod
    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute reward and breakdown dict for logging."""

    @abstractmethod
    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        """Check species-specific termination conditions."""

    @abstractmethod
    def _spawn_target(self):
        """Randomize the target (prey/food) position on reset."""

    # ------------------------------------------------------------------
    # Common sensor layout (overridable by subclasses)
    # ------------------------------------------------------------------
    _sensor_gyro_start: int = SENSOR_GYRO_START
    _sensor_accel_start: int = SENSOR_ACCEL_START
    _sensor_quat_start: int = SENSOR_QUAT_START

    # ------------------------------------------------------------------
    # Shared reward helpers
    # ------------------------------------------------------------------

    def _reward_alive(self) -> float:
        """Return the alive bonus. Identical across all species."""
        return _reward_alive_pure(self.alive_bonus)

    # Subclass attributes used by shared helpers.  Declared here for type
    # checking; actual values are set in subclass ``__init__``.
    smoothness_weight: float
    _prev_action: "np.ndarray | None"
    # Second-most-recent action, for the frequency-aware jerk term.  Separate
    # from _prev_action because the jerk needs two lags; both are cleared on
    # reset so an episode never charges jerk across an episode boundary.
    _prev_prev_action: "np.ndarray | None" = None
    action_jerk_weight: float = 0.0

    def _reward_energy(self, action: np.ndarray) -> float:
        """Compute normalised energy penalty. Identical across all species.

        Energy is ``sum(action**2) / n_actuators``, so it ranges [0, 1]
        when actions are in [-1, 1].
        """
        n_actuators: int = self.action_space.shape[0]  # type: ignore[index]
        return float(_reward_energy_pure(action, n_actuators, self.energy_penalty_weight))

    def _reward_action_smoothness(self, action: np.ndarray) -> tuple[float, float]:
        """Compute action-smoothness penalty and raw action delta.

        Returns ``(reward, action_delta)`` where *action_delta* is the
        sum of squared differences from the previous action.  Callers
        must set ``self._prev_action`` before the first call.
        """
        n_actuators: int = self.action_space.shape[0]  # type: ignore[index]
        reward, action_delta = _reward_action_smoothness_pure(
            action, self._prev_action, n_actuators, self.smoothness_weight
        )
        self._prev_prev_action = None if self._prev_action is None else self._prev_action.copy()
        self._prev_action = action.copy()
        return reward, action_delta

    def _reward_action_jerk(self, action: np.ndarray) -> tuple[float, float]:
        """Frequency-aware smoothness penalty and raw action jerk.

        MUST be called BEFORE :meth:`_reward_action_smoothness` in a step, so
        it sees the two prior actions rather than this step's own action as
        its first lag.  Returns ``(reward, action_jerk)``.
        """
        n_actuators: int = self.action_space.shape[0]  # type: ignore[index]
        reward, jerk = _reward_action_jerk_pure(
            action, self._prev_action, self._prev_prev_action, n_actuators, self.action_jerk_weight
        )
        return float(reward), float(jerk)

    # ------------------------------------------------------------------
    # Consolidated reward helpers (extracted from species envs)
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_forward_2d(quat: np.ndarray) -> np.ndarray:
        """Extract body forward direction (+X local axis) projected into XY plane.

        Args:
            quat: MuJoCo quaternion (w, x, y, z).

        Returns:
            Normalised 2D forward direction vector.
        """
        return np.asarray(_quat_to_forward_2d_pure(quat))

    @staticmethod
    def _quat_to_forward_z(quat: np.ndarray) -> float:
        """Compute the Z-component of the body's local X-axis (head direction) in world frame.

        Used for nosedive detection: negative values mean the head is
        pointing downward.

        Args:
            quat: MuJoCo quaternion (w, x, y, z).

        Returns:
            Scalar Z-component of forward direction.
        """
        return float(_quat_to_forward_z_pure(quat))

    def _compute_posture_reward(self, quat: np.ndarray, weight: float) -> tuple[float, float]:
        """Compute quadratic tilt penalty.

        Args:
            quat: Pelvis/torso quaternion from sensor data.
            weight: Posture reward weight.

        Returns:
            (reward, tilt_angle) tuple.
        """
        return _reward_posture_pure(quat, self.max_tilt_angle, weight)

    def _compute_lean_aware_posture_reward(
        self,
        quat: np.ndarray,
        weight: float,
        natural_forward_z: float,
    ) -> tuple[float, float]:
        """Compute posture penalty relative to a natural forward lean.

        Returns the absolute tilt angle alongside the target-relative reward
        so termination and diagnostics remain relative to world-up.
        """
        return _reward_lean_aware_posture_pure(
            quat,
            self.max_tilt_angle,
            weight,
            natural_forward_z,
        )

    def _compute_nosedive_penalty(
        self, quat: np.ndarray, weight: float, natural_forward_z: float
    ) -> tuple[float, float]:
        """Compute nosedive penalty (excessive forward pitch beyond natural lean).

        Args:
            quat: Pelvis/torso quaternion from sensor data.
            weight: Nosedive penalty weight.
            natural_forward_z: Baseline forward_z for species' natural lean.

        Returns:
            (reward, forward_z) tuple.
        """
        return _reward_nosedive_pure(quat, weight, natural_forward_z)

    def _compute_angular_velocity_penalty(
        self, weight: float, max_angvel: float = TAIL_ANGULAR_VEL_MAX
    ) -> tuple[float, float]:
        """Compute angular velocity (spin) penalty from root freejoint.

        Args:
            weight: Penalty weight.
            max_angvel: Normalisation ceiling (rad/s).

        Returns:
            (reward, instability_magnitude) tuple.
        """
        angvel = self.data.qvel[3:6]
        return _reward_angular_velocity_penalty_pure(angvel, weight, max_angvel)

    def _compute_tail_stability(
        self, tail_tip_site_id: int, weight: float, max_angvel: float = TAIL_ANGULAR_VEL_MAX
    ) -> tuple[float, float]:
        """Compute tail tip angular velocity penalty.

        Args:
            tail_tip_site_id: MuJoCo site ID for the tail tip.
            weight: Tail stability weight.
            max_angvel: Normalisation ceiling (rad/s).

        Returns:
            (reward, tail_instability_magnitude) tuple.
        """
        tail_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, tail_tip_site_id, tail_vel, 0)
        tail_tip_angvel = tail_vel[0:3]
        instability = float(np.linalg.norm(tail_tip_angvel))
        instability_norm = min(instability / max_angvel, 1.0)
        reward = -weight * instability_norm
        return reward, instability

    def _compute_approach_shaping(
        self,
        current_distance: float,
        prev_distance: "float | None",
        weight: float,
        max_speed: float,
    ) -> tuple[float, float]:
        """Compute approach shaping reward (reward closing distance, penalise retreating).

        Args:
            current_distance: Current distance to target.
            prev_distance: Previous step's distance (None on first step).
            weight: Approach shaping weight.
            max_speed: Maximum expected approach speed (m/s) for normalisation.

        Returns:
            (reward, approach_delta) tuple.
        """
        dt = self.frame_skip * self.model.opt.timestep
        return _reward_approach_shaping_pure(current_distance, prev_distance, weight, max_speed, dt)

    def _compute_forward_velocity(
        self, vel_2d: np.ndarray, forward_ref_2d: np.ndarray, vel_max: float, weight: float
    ) -> tuple[float, float]:
        """Compute forward velocity reward along a reference direction.

        Args:
            vel_2d: 2D velocity vector (qvel[0:2]).
            forward_ref_2d: Unit reference direction in XY plane.
            vel_max: Maximum velocity for normalisation.
            weight: Reward weight.

        Returns:
            (reward, raw_forward_vel) tuple.
        """
        return _reward_forward_velocity_pure(vel_2d, forward_ref_2d, vel_max, weight)

    def _compute_backward_penalty(self, forward_vel: float, vel_max: float, weight: float) -> tuple[float, float]:
        """Compute backward velocity penalty.

        Args:
            forward_vel: Forward velocity (negative means backward).
            vel_max: Normalisation ceiling.
            weight: Penalty weight.

        Returns:
            (reward, backward_vel) tuple.
        """
        return _reward_backward_penalty_pure(forward_vel, vel_max, weight)

    def _compute_drift_penalty(
        self, current_pos_2d: np.ndarray, initial_pos_2d: np.ndarray, weight: float
    ) -> tuple[float, float]:
        """Compute quadratic drift penalty (horizontal displacement from spawn).

        Args:
            current_pos_2d: Current XY position.
            initial_pos_2d: Spawn XY position.
            weight: Penalty weight.

        Returns:
            (reward, drift_distance) tuple.
        """
        return _reward_drift_penalty_pure(current_pos_2d, initial_pos_2d, weight)

    def _compute_heading_alignment(
        self, body_forward_2d: np.ndarray, forward_ref_2d: np.ndarray, weight: float
    ) -> tuple[float, float]:
        """Compute heading alignment reward (reward facing toward target).

        Args:
            body_forward_2d: Body's forward direction in XY plane.
            forward_ref_2d: Reference direction to target in XY plane.
            weight: Reward weight.

        Returns:
            (reward, heading_alignment_cos) tuple.
        """
        return _reward_heading_alignment_pure(body_forward_2d, forward_ref_2d, weight)

    def _compute_lateral_velocity_penalty(
        self, vel_2d: np.ndarray, body_forward_2d: np.ndarray, weight: float
    ) -> tuple[float, float]:
        """Compute lateral (crab-walk) velocity penalty.

        Args:
            vel_2d: 2D velocity vector.
            body_forward_2d: Body's forward direction in XY plane.
            weight: Penalty weight.

        Returns:
            (reward, lateral_vel) tuple.
        """
        return _reward_lateral_velocity_penalty_pure(vel_2d, body_forward_2d, weight)

    def _compute_speed_penalty(
        self, vel_2d: np.ndarray, weight: float, threshold: float = 0.10, max_excess: float = 1.0
    ) -> tuple[float, float]:
        """Penalise absolute 2D speed exceeding a threshold.

        Args:
            vel_2d: 2D velocity vector (qvel[0:2]).
            weight: Penalty weight.
            threshold: Speed (m/s) below which no penalty applies.
            max_excess: Speed above threshold at which penalty saturates.

        Returns:
            (reward, absolute_speed) tuple.
        """
        return _reward_speed_penalty_pure(vel_2d, weight, threshold, max_excess)

    def _compute_idle_penalty(self, vel_2d: np.ndarray, weight: float, threshold: float = 0.05) -> tuple[float, float]:
        """Penalise low 2D speed (standing still / barely moving).

        Applies a penalty that is strongest at zero speed and linearly
        decreases to zero when speed reaches *threshold*.

        Args:
            vel_2d: 2D velocity vector (qvel[0:2]).
            weight: Penalty weight (positive value; returned reward is negative).
            threshold: Speed (m/s) at or above which no penalty applies.

        Returns:
            (reward, absolute_speed) tuple.
        """
        return _reward_idle_penalty_pure(vel_2d, weight, threshold)

    def _init_gait_state(
        self,
        contact_threshold: float = 0.1,
        max_touchdown_history: int = 20,
    ) -> None:
        """Initialise gait symmetry tracking state.

        Call this in the subclass ``__init__`` (before ``super().__init__``
        is fine) to enable :meth:`_compute_gait_symmetry`.

        Args:
            contact_threshold: Force (N) above which a foot is considered
                in contact.  Matches ``metrics.py`` onset detection default.
            max_touchdown_history: Maximum number of touchdown events to keep
                in the sliding window.
        """
        self._contact_threshold = contact_threshold
        self._max_touchdown_history = max_touchdown_history
        self._prev_r_in_contact = False
        self._prev_l_in_contact = False
        self._touchdown_sequence: list[str] = []

    def _reset_gait_state(self) -> None:
        """Reset gait symmetry tracking for a new episode.

        Call this from the subclass ``_spawn_target`` / reset path.
        """
        self._prev_r_in_contact = False
        self._prev_l_in_contact = False
        self._touchdown_sequence = []

    def _compute_gait_symmetry(
        self,
        r_contact_force: float,
        l_contact_force: float,
        weight: float,
    ) -> tuple[float, float]:
        """Compute gait symmetry reward based on foot touchdown alternation.

        Tracks off→on transitions (touchdowns) and rewards when consecutive
        touchdowns alternate feet: L→R→L = 1.0, L→L→R = 0.5.

        Requires :meth:`_init_gait_state` to have been called.

        Args:
            r_contact_force: Right foot contact sensor reading (N).
            l_contact_force: Left foot contact sensor reading (N).
            weight: Gait symmetry reward weight.

        Returns:
            (reward, alternation_ratio) tuple.
        """
        r_in_contact = r_contact_force > self._contact_threshold
        l_in_contact = l_contact_force > self._contact_threshold
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

        reward = weight * alternation_ratio
        return reward, alternation_ratio

    # ── quadrupedal gait symmetry ──────────────────────────────────────

    def _init_quadruped_gait_state(
        self,
        contact_threshold: float = 0.1,
        max_touchdown_history: int = 20,
    ) -> None:
        """Initialise quadrupedal gait symmetry tracking state.

        For quadrupedal animals, gait symmetry is measured by diagonal pair
        alternation.  Diagonal-A = front-right + rear-left, Diagonal-B =
        front-left + rear-right.  A proper walk or trot produces alternating
        diagonal pair touchdowns (A→B→A→B).

        Call this in the subclass ``__init__`` (before ``super().__init__``
        is fine) to enable :meth:`_compute_quadruped_gait_symmetry`.

        Also initialises the bipedal gait state so
        :meth:`_compute_gait_symmetry` remains callable (e.g. for front-pair
        only analysis).

        Args:
            contact_threshold: Force (N) above which a foot is considered
                in contact.
            max_touchdown_history: Maximum number of diagonal touchdown
                events to keep in the sliding window.
        """
        # Reuse bipedal state init (keeps _compute_gait_symmetry working)
        self._init_gait_state(contact_threshold, max_touchdown_history)

        # Quadrupedal diagonal pair tracking
        self._quad_contact_threshold = contact_threshold
        self._quad_max_touchdown_history = max_touchdown_history
        self._prev_diag_a_in_contact = False  # FR + RL
        self._prev_diag_b_in_contact = False  # FL + RR
        self._quad_touchdown_sequence: list[str] = []

    def _reset_quadruped_gait_state(self) -> None:
        """Reset quadrupedal gait symmetry tracking for a new episode.

        Call this from the subclass ``_spawn_target`` / reset path.
        """
        self._reset_gait_state()
        self._prev_diag_a_in_contact = False
        self._prev_diag_b_in_contact = False
        self._quad_touchdown_sequence = []

    def _compute_quadruped_gait_symmetry(
        self,
        fr_contact_force: float,
        fl_contact_force: float,
        rr_contact_force: float,
        rl_contact_force: float,
        weight: float,
    ) -> tuple[float, float]:
        """Compute quadrupedal gait symmetry based on diagonal pair alternation.

        Diagonal pairs for a proper walk/trot:
            - Diagonal A: front-right (FR) + rear-left (RL)
            - Diagonal B: front-left (FL) + rear-right (RR)

        A good quadrupedal gait alternates these diagonal pairs.  This method
        tracks off→on transitions of each diagonal pair (at least one foot in
        the pair touching down while the pair was previously airborne) and
        rewards alternation between A and B touchdowns.

        Args:
            fr_contact_force: Front-right foot contact sensor reading (N).
            fl_contact_force: Front-left foot contact sensor reading (N).
            rr_contact_force: Rear-right foot contact sensor reading (N).
            rl_contact_force: Rear-left foot contact sensor reading (N).
            weight: Gait symmetry reward weight.

        Returns:
            (reward, alternation_ratio) tuple.
        """
        threshold = self._quad_contact_threshold

        # Diagonal pair contact: either foot in the pair is grounded
        diag_a_in_contact = fr_contact_force > threshold or rl_contact_force > threshold
        diag_b_in_contact = fl_contact_force > threshold or rr_contact_force > threshold

        # Detect off→on transitions for each diagonal pair
        diag_a_touchdown = diag_a_in_contact and not self._prev_diag_a_in_contact
        diag_b_touchdown = diag_b_in_contact and not self._prev_diag_b_in_contact

        self._prev_diag_a_in_contact = diag_a_in_contact
        self._prev_diag_b_in_contact = diag_b_in_contact

        if diag_a_touchdown:
            self._quad_touchdown_sequence.append("A")
        if diag_b_touchdown:
            self._quad_touchdown_sequence.append("B")
        if len(self._quad_touchdown_sequence) > self._quad_max_touchdown_history:
            self._quad_touchdown_sequence = self._quad_touchdown_sequence[-self._quad_max_touchdown_history :]

        n_touchdowns = len(self._quad_touchdown_sequence)
        if n_touchdowns > 1:
            alternations = sum(
                1
                for i in range(1, n_touchdowns)
                if self._quad_touchdown_sequence[i] != self._quad_touchdown_sequence[i - 1]
            )
            alternation_ratio = alternations / (n_touchdowns - 1)
        else:
            alternation_ratio = 0.0

        reward = weight * alternation_ratio
        return reward, alternation_ratio

    def _compute_pelvis_diagnostics(self) -> tuple[float, float]:
        """Compute pelvis angular velocity metrics for spinning detection.

        Returns:
            (pelvis_angular_vel_magnitude, pelvis_yaw_vel) tuple.
        """
        gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3]
        return float(np.linalg.norm(gyro)), float(gyro[2])

    # ------------------------------------------------------------------
    # Consolidated termination helpers
    # ------------------------------------------------------------------

    def _check_height_tilt_termination(self, body_z: float, tilt_angle: float) -> "tuple[bool, str | None]":
        """Check common height and tilt termination conditions.

        Args:
            body_z: Height of root body (pelvis/torso).
            tilt_angle: Tilt angle in radians.

        Returns:
            (terminated, reason) where reason is None if not terminated.
        """
        return _check_height_tilt_pure(body_z, tilt_angle, self.healthy_z_range, self.max_tilt_angle)

    def _foot_contact_forces(self) -> "tuple[float, ...]":
        """Instantaneous per-foot touch-force sums from the current sensordata.

        See the class docstring for the contract.  This is the per-substep
        primitive; the contact-shaped rewards and info keys consume
        :meth:`_aggregated_foot_contact_forces` instead.
        """
        sensordata = self.data.sensordata
        return tuple(float(sum(sensordata[index] for index in group)) for group in self._foot_sensor_groups)

    def _aggregated_foot_contact_forces(self) -> "tuple[float, ...]":
        """Per-foot MIN force across the current control step's substeps.

        Falls back to the instantaneous read when no aggregate exists for the
        CURRENT step count (before any step, or after reset zeroes the count)
        so those callers keep today's semantics on both backends.  The tag
        cannot see EXTERNAL state mutation: hand-posing ``self.data`` after a
        completed step and scoring directly reads that step's minima, not the
        posed state -- call :meth:`_invalidate_substep_aggregates` after
        out-of-band mutation.
        """
        if self._substep_min_foot_forces is None or self._substep_contact_step != self._step_count:
            return self._foot_contact_forces()
        return tuple(float(value) for value in self._substep_min_foot_forces)

    def _aggregated_min_height(self, check_index: int, instantaneous: float) -> float:
        """MIN z of a ``_substep_height_checks`` entry across the last step.

        Returns ``min(aggregate, instantaneous)`` when a fresh aggregate
        exists so a hand-posed-lower state can still terminate, and the bare
        instantaneous value otherwise (before any step, after reset, or for a
        species that declares no checks).  Termination reads this; info keys
        keep the boundary sample.
        """
        if self._substep_min_heights is None or self._substep_contact_step != self._step_count:
            return instantaneous
        return float(min(self._substep_min_heights[check_index], instantaneous))

    def _invalidate_substep_aggregates(self) -> None:
        """Drop the last step's contact aggregates and strike latch.

        The step-count tag cannot detect EXTERNAL state mutation: a caller
        that hand-poses ``self.data`` after a completed step (keyframe reset,
        qpos surgery + ``mj_forward``) and then scores the state directly
        would otherwise read the pre-mutation step's minima and latch.  Call
        this after mutating the state out-of-band; ordinary ``step``/``reset``
        flows never need it (reset zeroes ``_step_count``, which the tag
        already fails against).
        """
        self._substep_min_foot_forces = None
        self._substep_floor_hit_geom = None
        self._substep_min_heights = None
        self._substep_contact_step = -1

    def _check_floor_contact(
        self, body_ground_geoms: set, floor_geom_id: int, geom_categories: "dict[str, set] | None" = None
    ) -> "tuple[bool, str | None]":
        """Check if any body geom contacts the floor.

        Consults the step loop's ANY-substep latch first: contacts are
        recomputed by every physics substep, so a strike during substeps
        1..N-1 is invisible to a scan of the final substep's ``data.contact``
        -- a tail slap that resolves within 8 ms used to go unterminated.
        The live scan remains as the fallback for callers outside step()
        (tests, hand-posed states).

        Args:
            body_ground_geoms: Set of geom IDs that should terminate on floor contact.
            floor_geom_id: Floor geom ID.
            geom_categories: Optional mapping of category names to geom ID sets
                for more specific termination reasons (e.g. {"tail": tail_geoms, "head": head_geoms}).
                Falls back to "body_contact" if geom not found in any category.

        Returns:
            (terminated, reason) where reason is None if not terminated.
        """
        latched = self._substep_floor_hit_geom if self._substep_contact_step == self._step_count else None
        if latched is not None and latched in body_ground_geoms:
            if geom_categories:
                for category_name, category_geoms in geom_categories.items():
                    if latched in category_geoms:
                        return True, f"{category_name}_contact"
            return True, "body_contact"
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            floor_contact_geom = None
            if geom2 == floor_geom_id and geom1 in body_ground_geoms:
                floor_contact_geom = geom1
            elif geom1 == floor_geom_id and geom2 in body_ground_geoms:
                floor_contact_geom = geom2

            if floor_contact_geom is not None:
                if geom_categories:
                    for category_name, category_geoms in geom_categories.items():
                        if floor_contact_geom in category_geoms:
                            return True, f"{category_name}_contact"
                return True, "body_contact"
        return False, None

    # ------------------------------------------------------------------
    # Consolidated target spawning helpers
    # ------------------------------------------------------------------

    def _spawn_target_2d(
        self,
        distance_range: "tuple[float, float]",
        lateral_range: "tuple[float, float]",
        target_z: float,
    ) -> np.ndarray:
        """Spawn target (prey/food) at a random 2D location with fixed height.

        Sets ``self.data.mocap_pos[0]`` and returns the target position.

        Args:
            distance_range: (min, max) forward distance.
            lateral_range: (min, max) lateral offset.
            target_z: Fixed Z height for target.

        Returns:
            Target position as (3,) numpy array.
        """
        if self.np_random is not None:
            distance = self.np_random.uniform(*distance_range)
            lateral = self.np_random.uniform(*lateral_range)
        else:
            distance = float(np.mean(distance_range))
            lateral = 0.0

        target_pos = np.array([distance, lateral, target_z])
        self.data.mocap_pos[0] = target_pos
        return target_pos

    @staticmethod
    def _compute_initial_direction_2d(target_pos: np.ndarray) -> np.ndarray:
        """Compute normalised initial 2D direction from origin to target.

        Args:
            target_pos: Target position (3D).

        Returns:
            Normalised 2D direction vector.
        """
        dir_2d = np.array(target_pos[:2], dtype=np.float64)
        dir_len = float(np.linalg.norm(dir_2d))
        if dir_len > 1e-6:
            dir_2d /= dir_len
        return dir_2d

    # ------------------------------------------------------------------
    # Shared methods
    # ------------------------------------------------------------------

    def set_reward_weight(self, name: str, value: float) -> None:
        """Dynamically update a reward weight attribute.

        Used by :class:`RewardRampCallback` to gradually introduce new reward
        components during curriculum stage transitions.
        """
        if not hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")
        setattr(self, name, value)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        # Scale action from [-1, 1] to actuator control ranges
        ctrl = self._scale_action(action)
        self.data.ctrl[:] = ctrl

        # Step physics, aggregating contact across the substeps: each mj_step
        # recomputes sensordata and data.contact, so after the loop only the
        # final substep survives -- and a control-clock-locked hop can unload
        # (or a tail can strike the floor) entirely between control-boundary
        # samples.  MIN per-foot force feeds the contact-shaped rewards and
        # the stance-duty gate; the first floor strike is latched for
        # _check_floor_contact.  The MJX step_fn carries the same aggregates
        # through its fori_loop -- keep the two in lockstep.
        min_forces: "np.ndarray | None" = None
        min_heights: "np.ndarray | None" = None
        self._substep_floor_hit_geom = None
        track_feet = bool(self._foot_sensor_groups)
        height_checks = self._substep_height_checks
        track_strikes = getattr(self, "_body_ground_geoms", None)
        floor_geom_id = getattr(self, "floor_geom_id", None)
        ground_geoms: "np.ndarray | None" = None
        if track_strikes and floor_geom_id is not None:
            if self._ground_geom_array is None:
                # Cached sorted array of the terminating geoms for the
                # vectorized per-substep scan below -- a Python loop over
                # data.contact costs ~50 us per substep on the training hot
                # path.
                self._ground_geom_array = np.fromiter(sorted(track_strikes), dtype=np.int64)
            ground_geoms = self._ground_geom_array
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._substep_probe_hook is not None:
                self._substep_probe_hook()
            if track_feet:
                forces = np.asarray(self._foot_contact_forces(), dtype=np.float64)
                min_forces = forces if min_forces is None else np.minimum(min_forces, forces)
            if height_checks:
                heights = np.fromiter(
                    (
                        self.data.site_xpos[entity_id, 2] if kind == "site" else self.data.xpos[entity_id, 2]
                        for kind, entity_id in height_checks
                    ),
                    dtype=np.float64,
                    count=len(height_checks),
                )
                min_heights = heights if min_heights is None else np.minimum(min_heights, heights)
            if ground_geoms is not None and self._substep_floor_hit_geom is None:
                pairs = self.data.contact.geom
                if len(pairs):
                    g1, g2 = pairs[:, 0], pairs[:, 1]
                    hits = ((g2 == floor_geom_id) & np.isin(g1, ground_geoms)) | (
                        (g1 == floor_geom_id) & np.isin(g2, ground_geoms)
                    )
                    hit_indices = np.flatnonzero(hits)
                    if hit_indices.size:
                        # First matching contact, matching the live scan's
                        # iteration order for categorization ties.
                        first = int(hit_indices[0])
                        struck = g1[first] if g2[first] == floor_geom_id else g2[first]
                        self._substep_floor_hit_geom = int(struck)
        self._step_count += 1
        # Tag the aggregates with the step they were measured at; the tag is
        # what invalidates them across reset (which zeroes _step_count).
        self._substep_min_foot_forces = min_forces
        self._substep_min_heights = min_heights
        self._substep_contact_step = self._step_count

        # Update cumulative distance traveled (XY path length)
        current_pos_2d = self.data.qpos[0:2].copy()
        self._distance_traveled += float(np.linalg.norm(current_pos_2d - self._prev_pos_2d))
        self._prev_pos_2d = current_pos_2d

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward, reward_info = self._get_reward_info(action)

        # Check termination
        terminated, term_info = self._is_terminated()
        if terminated:
            if not term_info.get("success"):
                reward += self.fall_penalty
            reward_info["reward_total"] = reward

        truncated = self._is_truncated()

        # Combine info
        info = {**reward_info, **term_info}
        info["step"] = self._step_count
        info["distance_traveled"] = self._distance_traveled

        # SB3 convention: report episode success at episode end so
        # EvalCallback records per-episode success rates (evaluations.npz
        # "successes" array). CurriculumCallback reads these for gating.
        if terminated or truncated:
            info["is_success"] = bool(term_info.get("success", False))

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [-1, 1] to actuator control range.

        Clips the incoming action to [-1, 1] first: SB3 already clips
        before stepping, but direct callers (custom scripts, raw Gaussian
        policy outputs) would otherwise command out-of-range ctrl values.
        """
        action = np.clip(action, -1.0, 1.0)
        ctrl_range = self.model.actuator_ctrlrange
        ctrl_min = ctrl_range[:, 0]
        ctrl_max = ctrl_range[:, 1]

        # Linear interpolation from [-1, 1] to [min, max]
        scaled = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        return np.asarray(scaled)

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)."""
        return self._step_count >= self.max_episode_steps

    def _bounded_reset_height_delta(self, base_z: float, delta: float) -> float:
        """Bound a reset height perturbation so the spawn is never terminal.

        SUPERSEDED by ``_settle_root_on_ground``, which overwrites the root
        height as a pure function of the sampled joint pose, so the bounded
        delta this returns no longer reaches the post-reset state (verified to
        one ULP).  The draw and this bound are retained only to keep the reset
        deterministic in its seed: removing the draw would shift every
        subsequent RNG draw and re-anchor all seeded baselines.  Remove both at
        the next policy-interface revision.

        The historical rationale, for the record: the root-height jitter is
        the only UNBOUNDED term in the reset — every
        other one is a bounded uniform — and an unbounded Gaussian will
        eventually place the root outside ``healthy_z_range``.  On T-Rex it did
        so often enough to matter: home pelvis 0.926 m against a 0.70 m floor
        is a 0.226 m margin, and at sigma 0.10 m that is only 2.26 sigma, so
        1.19% of spawns landed below the floor.  Measured over seeds 3042-5041,
        18/2000 spawned sub-floor and 16 of those terminated on the first step
        whatever the policy did — capping any reliability measurement at ~99%
        for reasons that have nothing to do with the policy.

        The bound is the distance to the nearer end of ``healthy_z_range``, less
        a small margin, and is applied symmetrically so the mean spawn height is
        unchanged.  Clipping rather than resampling keeps the number of RNG
        draws per reset fixed and the reset deterministic; the cost is a small
        point mass at each bound (~2% per side on T-Rex).  Species with ample
        headroom are effectively unaffected, since the bound then sits far out
        in the tail.
        """
        low, high = self.healthy_z_range
        headroom = min(base_z - low, high - base_z) - _RESET_HEIGHT_TERMINATION_MARGIN
        bound = max(headroom, 0.0)
        return float(np.clip(delta, -bound, bound))

    def _root_subtree_geoms(self) -> "np.ndarray":
        """Geom IDs belonging to the animal, i.e. the free-joint root's subtree.

        Everything the reset translates vertically moves together, and nothing
        else does: prey, food and other spawned props hang off the world or
        their own joints, so they must not take part in ground settling.

        NON-COLLIDING geoms are excluded.  A geom with ``contype == 0`` and
        ``conaffinity == 0`` generates no contacts, so it can never rest on the
        floor -- settling to one would hold the animal's real feet above the
        ground, which is the hover this method exists to prevent.  Every
        species carries some: cosmetic surface detail (``brow_ridge``,
        ``crest``, ``sagittal_crest``, dibothrosuchus' twelve ``scute``\\s) and
        the necks that are deliberately non-collidable on every species except
        velociraptor.  None of them is currently the lowest geom at any shipped
        noise level, so this filter is behaviour-preserving today; it stops the
        probe from disagreeing with its own definition of "the ground".
        """
        if self._root_subtree_geom_ids is not None:
            return self._root_subtree_geom_ids
        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if not free:
            ids = np.empty(0, dtype=np.int32)
        else:
            root = int(self.model.jnt_bodyid[free[0]])
            bodies = {root}
            for b in range(root + 1, self.model.nbody):
                if int(self.model.body_parentid[b]) in bodies:
                    bodies.add(b)
            ids = np.array(
                [
                    g
                    for g in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[g]) in bodies
                    and (int(self.model.geom_contype[g]) != 0 or int(self.model.geom_conaffinity[g]) != 0)
                ],
                dtype=np.int32,
            )
        self._root_subtree_geom_ids = ids
        return ids

    def _static_floor_geoms(self) -> "np.ndarray":
        """Geom IDs of the static ground the animal is expected to stand on.

        HFIELD is accepted here so the probe still reports clearance on one,
        but ``_settle_root_on_ground``'s single-shift exactness argument holds
        only for a horizontal PLANE — over a heightfield the nearest-distance
        pair can change as the root translates, so a species standing on one
        would need an iterative settle.  Every current species floors on a
        plane at z=0.

        Assumptions this shares with the MJX settle, stated so the two stay
        comparable: exactly ONE horizontal floor, and a spawn over it.  This
        side takes the minimum ``mj_geomDistance`` over every floor geom, which
        respects finite plane extents; MJX takes the highest floor's z and
        treats the plane as infinite.  With one 100x100 plane at z=0 and a
        spawn near the origin the two agree to ~1e-9, but multiple floors at
        different heights, a tilted plane, or a spawn beyond the plane's extent
        would diverge.  MJX raises on a heightfield rather than mis-settling.
        """
        if self._static_floor_geom_ids is not None:
            return self._static_floor_geom_ids
        ids = np.array(
            [
                g
                for g in range(self.model.ngeom)
                if int(self.model.geom_bodyid[g]) == 0
                and self.model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
            ],
            dtype=np.int32,
        )
        self._static_floor_geom_ids = ids
        return ids

    def lowest_ground_clearance(self, data: "mujoco.MjData | None" = None) -> float:
        """Signed distance from the animal's lowest geom to the ground, in METRES.

        Negative means the pose is interpenetrating the floor.  Requires the
        caller to have run ``mj_forward`` (or ``mj_kinematics``) first.
        """
        data = self.data if data is None else data
        body_geoms, floor_geoms = self._root_subtree_geoms(), self._static_floor_geoms()
        if not len(body_geoms) or not len(floor_geoms):
            return float("inf")
        worst = float("inf")
        for g in body_geoms:
            for f in floor_geoms:
                d = mujoco.mj_geomDistance(self.model, data, int(g), int(f), _GROUND_PROBE_DISTANCE, None)
                worst = min(worst, float(d))
        return worst

    def home_ground_clearance(self) -> float:
        """The ground clearance the unperturbed home keyframe was authored with.

        Species author a deliberate resting contact depth into the keyframe --
        T-Rex's plantar pad and digits sit 0.5 mm into the floor, for instance.
        Settling every reset back to *this* value rather than to an arbitrary
        constant keeps the noise-free reset bit-identical, preserves each
        species' authored contact, and still removes the pose-dependent error
        that joint jitter introduces.
        """
        if self._home_ground_clearance_m is not None:
            return self._home_ground_clearance_m
        scratch = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            keyframe = int(getattr(self, "_reset_keyframe_id", 0))
            if not 0 <= keyframe < self.model.nkey:
                keyframe = 0
            mujoco.mj_resetDataKeyframe(self.model, scratch, keyframe)
        else:
            mujoco.mj_resetData(self.model, scratch)
        mujoco.mj_forward(self.model, scratch)
        value = self.lowest_ground_clearance(scratch)
        if not np.isfinite(value):
            value = _RESET_GROUND_CLEARANCE
        self._home_ground_clearance_m = float(value)
        return self._home_ground_clearance_m

    def _settle_root_on_ground(self, clearance: float | None = None) -> float:
        """Translate the root vertically so the animal starts *on* the ground.

        The reset perturbs joint angles and root height independently, but the
        two are not independent in the world: bending the legs changes how far
        the feet sit below the root, so any fixed root height is wrong for most
        sampled poses.  Left uncorrected on T-Rex this spawned the model up to
        0.198 m inside the floor, and the contact solver answered with ~19x body
        weight, launching it 0.5 m into the air to tumble ballistically for ~60
        steps before landing nose-down — an "episode" whose outcome no policy
        could influence.  The same jitter spawned other seeds 0.18 m *above* the
        floor, opening the episode with a free fall instead.

        Because the ground is a horizontal plane, translating the root changes
        every body-to-floor distance by exactly the same amount, so a single
        shift settles the pose exactly — no iteration, no extra RNG draws, and
        the reset stays deterministic in the seed.

        The settle target is the home keyframe's own clearance, so a noise-free
        reset is a no-op and each species keeps its authored resting contact.

        Returns the applied shift in metres (positive = raised).
        """
        target = self.home_ground_clearance() if clearance is None else clearance
        mujoco.mj_forward(self.model, self.data)
        worst = self.lowest_ground_clearance()
        if not np.isfinite(worst):
            return 0.0
        shift = target - worst
        self.data.qpos[2] += shift
        mujoco.mj_forward(self.model, self.data)
        return float(shift)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset MuJoCo state using the species-selected keyframe if available,
        # otherwise the historical first keyframe. Species with a named
        # nominal pose can cache its ID during initialization.
        if self.model.nkey > 0:
            reset_keyframe_id = int(getattr(self, "_reset_keyframe_id", 0))
            if not 0 <= reset_keyframe_id < self.model.nkey:
                raise ValueError("reset keyframe ID is outside the model keyframe range")
            mujoco.mj_resetDataKeyframe(self.model, self.data, reset_keyframe_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbation to initial pose
        if self.np_random is not None:
            noise_scale = self.reset_noise_scale
            self.data.qpos[7:] += self.np_random.uniform(-noise_scale, noise_scale, size=self.data.qpos[7:].shape)
            self.data.qvel[:] += self.np_random.uniform(-noise_scale, noise_scale, size=self.data.qvel.shape)
            # STATE-INERT since _settle_root_on_ground below, which overwrites
            # the root height as a pure function of the sampled joint pose
            # (verified to one ULP across height scales).  The draw is kept so
            # the reset stays deterministic in its seed: dropping it would
            # shift every subsequent draw and re-anchor all seeded baselines.
            # Remove the whole height channel — this draw, reset_height_noise_scale
            # and _bounded_reset_height_delta — at the next policy-interface
            # revision.
            height_scale = 0.0
            if noise_scale > 0.0:
                height_scale = noise_scale if self.reset_height_noise_scale is None else self.reset_height_noise_scale
            base_z = float(self.data.qpos[2])
            height_delta = self.np_random.normal(0, height_scale)
            self.data.qpos[2] = base_z + self._bounded_reset_height_delta(base_z, height_delta)

        # Randomize target position (species-specific)
        self._spawn_target()

        # Place the animal ON the ground.  This has to come after BOTH the joint
        # jitter and the height jitter, since either one changes how far the
        # lowest geom sits below the root.  _bounded_reset_height_delta above
        # keeps the spawn inside healthy_z_range, but that is a termination
        # predicate on the ROOT and says nothing about foot-to-floor geometry —
        # a pelvis at 0.739 m is "healthy" with the toes 0.198 m underground.
        self._settle_root_on_ground()

        # Forward pass to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_pos_2d = self.data.qpos[0:2].copy()
        self._distance_traveled = 0.0

        obs = self._get_obs()
        info = {"step": 0}

        return obs, info

    def _make_camera(self) -> mujoco.MjvCamera:
        """Create a configured MjvCamera for rendering."""
        camera = mujoco.MjvCamera()
        if self._camera_track_body is not None:
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._camera_track_body)
        camera.distance = self._camera_distance
        camera.azimuth = self._camera_azimuth
        camera.elevation = self._camera_elevation
        return camera

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                cam = self._viewer.cam
                if self._camera_track_body is not None:
                    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._camera_track_body)
                cam.distance = self._camera_distance
                cam.azimuth = self._camera_azimuth
                cam.elevation = self._camera_elevation
            self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
                self._camera = self._make_camera()
            self._renderer.update_scene(self.data, self._camera)
            return self._renderer.render()

    def close(self):
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
