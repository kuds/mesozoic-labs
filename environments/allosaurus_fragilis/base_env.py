"""Level-ground standing and forward walking for Allosaurus fragilis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.heading_observation import heading_free_root_segments
from environments.shared.stance_diagnostics import derive_stance_info


class AllosaurusFragilisEnv(BaseDinoEnv):
    """Control hindlimb joint torques on a free-root, digitigrade biped.

    The observation follows ``bipedal-heading-free/v1`` (41 values since
    policy interface r2; r1 had 45, with a target direction and distance):
    hinge positions (12) and velocities (12), pelvis projected gravity (3),
    the pelvis gyro (3), the pelvis velocity in its heading frame (3), the
    pelvis accelerometer (3), one touch reading per foot (2) and the command
    (3, last).  Nothing in it reads the world heading.
    """

    MODEL_PATH: Path
    HEIGHT_LABEL = "Pelvis height"
    SUCCESS_KEYS = ("course_complete",)
    #: The walking course: the pelvis completes it once its displacement along
    #: the spawn heading reaches this length less COURSE_TOLERANCE_M, with a
    #: foot bearing weight.  Until policy interface r2 a massless mocap target
    #: sat this far ahead of the spawn, and the course ended within the
    #: tolerance of it.
    COURSE_LENGTH_M = 8.0
    COURSE_TOLERANCE_M = 0.35
    supported_training_backends = ("stable-baselines3",)
    action_mapping = "home-keyframe-residual/v1"
    _camera_distance = 10.2
    _camera_azimuth = 105
    _camera_elevation = -9
    _camera_track_body = "pelvis"

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int | None = None,
        forward_vel_weight: float = 0.0,
        drift_penalty_weight: float = 0.4,
        energy_penalty_weight: float = 0.03,
        alive_bonus: float = 1.0,
        fall_penalty: float = -50.0,
        healthy_z_range: tuple[float, float] = (1.25, 2.35),
        max_tilt_angle: float = 0.65,
        reset_noise_scale: float = 0.005,
        command_mode: str = "none",
        command_config: DirectionCommandConfig | None = None,
    ):
        # Gymnasium consumes max_episode_steps for its TimeLimit wrapper.
        # Keep the native walking horizon long enough to reach the course end.
        if max_episode_steps is None:
            max_episode_steps = 3000 if forward_vel_weight > 0 else 1000
        self.drift_penalty_weight = drift_penalty_weight
        super().__init__(
            model_path=str(self.MODEL_PATH),
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            forward_vel_weight=forward_vel_weight,
            energy_penalty_weight=energy_penalty_weight,
            alive_bonus=alive_bonus,
            fall_penalty=fall_penalty,
            healthy_z_range=healthy_z_range,
            max_tilt_angle=max_tilt_angle,
            reset_noise_scale=reset_noise_scale,
            command_mode=command_mode,
            command_config=command_config,
        )
        self.metadata = {**self.metadata, "render_fps": round(1 / self.dt)}

    def _cache_ids(self) -> None:
        self._cache_home_keyframe("Allosaurus fragilis")
        self.pelvis_id = self.model.body("pelvis").id
        self.floor_geom_id = self.model.geom("floor").id
        self._sensor_gyro_start = int(self.model.sensor("pelvis_gyro").adr[0])
        self._sensor_accel_start = int(self.model.sensor("pelvis_accel").adr[0])
        self._sensor_quat_start = int(self.model.sensor("pelvis_orientation").adr[0])
        self._foot_sensor_groups = tuple(
            (int(self.model.sensor(name).adr[0]),) for name in ("r_foot_touch", "l_foot_touch")
        )
        self.target_standing_z = float(self.model.key_qpos[self._reset_keyframe_id, 2])
        self.body_mass = float(self.model.body_subtreemass[self.pelvis_id])
        self._contact_threshold = 0.04 * self.body_mass * abs(float(self.model.opt.gravity[2]))
        foot_geoms = {
            self.model.geom(side + suffix).id
            for side in ("l", "r")
            for suffix in ("_pad", "_pad_d2", "_pad_d3", "_pad_d4")
        }
        self._body_ground_geoms = set(self._root_subtree_geoms()) - foot_geoms

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        residual = np.clip(action, -1.0, 1.0)
        lo, hi = self.model.actuator_ctrlrange.T
        span = np.where(residual < 0, self._home_ctrl - lo, hi - self._home_ctrl)
        return cast(np.ndarray, self._home_ctrl + residual * span)

    def _get_obs(self) -> np.ndarray:
        gravity, velocity = heading_free_root_segments(self.data.qpos, self.data.qvel[:3])
        sensors = self.data.sensordata
        return np.asarray(
            np.concatenate(
                (
                    self.data.qpos[7:],
                    self.data.qvel[6:],
                    gravity,
                    sensors[self._sensor_gyro_start : self._sensor_gyro_start + 3],
                    velocity,
                    sensors[self._sensor_accel_start : self._sensor_accel_start + 3],
                    self._foot_contact_forces(),
                    self._command,
                )
            ),
            dtype=np.float32,
        )

    def _reset_episode_state(self) -> None:
        """Nothing per-episode to clear.

        The course is the spawn's drift anchor and reference heading, which
        reset() sets before this hook, and no reward term keeps an action
        history or gait state.
        """

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        pelvis = self.data.xpos[self.pelvis_id]
        delta = pelvis[:2] - self._initial_pos_2d
        forward = float(self.data.qvel[:2] @ self._reference_heading_2d)
        drift = float(np.linalg.norm(delta))
        right, left = self._aggregated_foot_contact_forces()
        supported = max(right, left) > self._contact_threshold
        tilt = self._quat_to_tilt(self.data.qpos[3:7])
        height_error = (self._clearance(pelvis) - self.target_standing_z) / self.target_standing_z
        posture = float(np.exp(-((tilt / 0.30) ** 2 + (height_error / 0.18) ** 2)))
        components = {
            "reward_alive": self.alive_bonus * supported,
            "reward_posture": posture * supported,
            "reward_forward": self.forward_vel_weight * float(np.clip(forward, -1.5, 1.5)),
            "reward_drift": -self.drift_penalty_weight * drift,
            "reward_energy": -self.energy_penalty_weight * float(np.mean(np.square(action))),
        }
        reward = float(sum(components.values()))
        info = {
            **components,
            "reward_total": reward,
            "forward_vel": forward,
            "pelvis_height": self._clearance(pelvis),
            "tilt_angle": tilt,
            "r_foot_contact": right,
            "l_foot_contact": left,
            "drift_distance": drift,
            "course_complete": 0.0,
        }
        info.update(derive_stance_info(info, contact_threshold=self._contact_threshold))
        return float(reward), info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            return True, {"termination_reason": "nonfinite_state"}
        terminated, info = self._root_termination(self.pelvis_id, "pelvis_height")
        if terminated:
            return True, info
        contact, reason = self._check_floor_contact(self._body_ground_geoms, self.floor_geom_id)
        if contact:
            return True, {**info, "termination_reason": reason}
        if self.forward_vel_weight > 0:
            # The course ends at a line across the spawn heading, not at a
            # point: lateral offset neither prevents nor causes completion.
            along = float((self.data.xpos[self.pelvis_id, :2] - self._initial_pos_2d) @ self._reference_heading_2d)
            if (
                along >= self.COURSE_LENGTH_M - self.COURSE_TOLERANCE_M
                and max(self._aggregated_foot_contact_forces()) > self._contact_threshold
            ):
                return True, {**info, "termination_reason": "course_complete", "course_complete": 1.0, "success": True}
        return False, info
