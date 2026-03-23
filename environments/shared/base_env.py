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
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .constants import SENSOR_ACCEL_START, SENSOR_GYRO_START, SENSOR_QUAT_START, TAIL_ANGULAR_VEL_MAX
from .reward_functions import check_height_tilt_termination as _check_height_tilt_pure
from .reward_functions import quat_to_forward_2d as _quat_to_forward_2d_pure
from .reward_functions import quat_to_forward_z as _quat_to_forward_z_pure
from .reward_functions import quat_to_tilt as _quat_to_tilt_pure
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
from .reward_functions import reward_nosedive as _reward_nosedive_pure
from .reward_functions import reward_posture as _reward_posture_pure
from .reward_functions import reward_speed_penalty as _reward_speed_penalty_pure


class BaseDinoEnv(gym.Env, ABC):
    """Abstract base class for dinosaur locomotion environments."""

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
        return _quat_to_tilt_pure(quat)

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

    def _reward_energy(self, action: np.ndarray) -> float:
        """Compute normalised energy penalty. Identical across all species.

        Energy is ``sum(action**2) / n_actuators``, so it ranges [0, 1]
        when actions are in [-1, 1].
        """
        n_actuators: int = self.action_space.shape[0]  # type: ignore[index]
        return _reward_energy_pure(action, n_actuators, self.energy_penalty_weight)

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
        self._prev_action = action.copy()
        return reward, action_delta

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
        return _quat_to_forward_z_pure(quat)

    def _compute_posture_reward(self, quat: np.ndarray, weight: float) -> tuple[float, float]:
        """Compute quadratic tilt penalty.

        Args:
            quat: Pelvis/torso quaternion from sensor data.
            weight: Posture reward weight.

        Returns:
            (reward, tilt_angle) tuple.
        """
        return _reward_posture_pure(quat, self.max_tilt_angle, weight)

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

    def _check_floor_contact(
        self, body_ground_geoms: set, floor_geom_id: int, geom_categories: "dict[str, set] | None" = None
    ) -> "tuple[bool, str | None]":
        """Check if any body geom contacts the floor.

        Args:
            body_ground_geoms: Set of geom IDs that should terminate on floor contact.
            floor_geom_id: Floor geom ID.
            geom_categories: Optional mapping of category names to geom ID sets
                for more specific termination reasons (e.g. {"tail": tail_geoms, "head": head_geoms}).
                Falls back to "body_contact" if geom not found in any category.

        Returns:
            (terminated, reason) where reason is None if not terminated.
        """
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

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

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

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [-1, 1] to actuator control range."""
        ctrl_range = self.model.actuator_ctrlrange
        ctrl_min = ctrl_range[:, 0]
        ctrl_max = ctrl_range[:, 1]

        # Linear interpolation from [-1, 1] to [min, max]
        scaled = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        return np.asarray(scaled)

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)."""
        return self._step_count >= self.max_episode_steps

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset MuJoCo state using keyframe if available, otherwise default
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbation to initial pose
        if self.np_random is not None:
            noise_scale = self.reset_noise_scale
            self.data.qpos[7:] += self.np_random.uniform(-noise_scale, noise_scale, size=self.data.qpos[7:].shape)
            self.data.qvel[:] += self.np_random.uniform(-noise_scale, noise_scale, size=self.data.qvel.shape)
            # Slightly vary starting height to improve policy robustness
            self.data.qpos[2] += self.np_random.normal(0, noise_scale)

        # Randomize target position (species-specific)
        self._spawn_target()

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
