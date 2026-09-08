"""SB3 environments for the unchanged Compsognathus MuJoCo models.

Actions are normalized residuals about the gravity-preloaded ``home`` controls.
The 53-dimensional anatomical and 43-dimensional robot observations use the
repository's privileged bipedal state/target layout. Camera pixels are available
separately; these MLP policies are simulation baselines, not onboard policies.
"""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import mujoco
import numpy as np

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.model import robot_body_ids
from environments.shared.base_env import BaseDinoEnv
from environments.shared.stance_diagnostics import derive_stance_info


class CompsognathusEnv(BaseDinoEnv):
    """Balance, locomotion and non-contact target reaching on the anatomical model."""

    variant = "biological"
    supported_training_backends = ("stable-baselines3",)
    action_mapping = "home-keyframe-residual/v1"
    _camera_distance = 1.15
    _camera_azimuth = 135
    _camera_elevation = -18
    _camera_track_body = "pelvis"

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 1000,
        forward_vel_weight: float = 0.0,
        forward_vel_max: float = 0.25,
        alive_bonus: float = 1.0,
        energy_penalty_weight: float = 0.02,
        fall_penalty: float = -25.0,
        posture_weight: float = 1.0,
        height_weight: float = 1.0,
        home_pose_weight: float = 0.15,
        smoothness_weight: float = 0.02,
        spin_penalty_weight: float = 0.02,
        drift_penalty_weight: float = 0.5,
        lateral_penalty_weight: float = 0.1,
        heading_weight: float = 0.0,
        gait_symmetry_weight: float = 0.0,
        target_approach_weight: float = 0.0,
        target_reach_bonus: float = 0.0,
        target_radius: float = 0.08,
        target_max_speed: float = 0.10,
        prey_distance_range: tuple[float, float] = (0.6, 1.0),
        prey_lateral_range: tuple[float, float] = (-0.2, 0.2),
        healthy_z_range: tuple[float, float] | None = None,
        max_tilt_angle: float = 0.7,
        reset_noise_scale: float = 0.01,
        reset_height_noise_scale: float = 0.0,
    ):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode!r}")
        if not isinstance(frame_skip, int) or isinstance(frame_skip, bool) or frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer")
        if not isinstance(max_episode_steps, int) or isinstance(max_episode_steps, bool) or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        for name, value in (
            ("forward_vel_max", forward_vel_max),
            ("target_radius", target_radius),
            ("target_max_speed", target_max_speed),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(reset_noise_scale) or not 0 <= reset_noise_scale <= 0.1:
            raise ValueError("reset_noise_scale must be between 0 and 0.1 radians")
        if not np.isfinite(max_tilt_angle) or not 0 < max_tilt_angle < np.pi / 2:
            raise ValueError("max_tilt_angle must be between 0 and pi/2 radians")
        for name, bounds in (("prey_distance_range", prey_distance_range), ("prey_lateral_range", prey_lateral_range)):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must contain two ordered finite bounds")
        if prey_distance_range[0] <= target_radius:
            raise ValueError("Targets must spawn outside target_radius")

        self.forward_vel_max = forward_vel_max
        self.posture_weight = posture_weight
        self.height_weight = height_weight
        self.home_pose_weight = home_pose_weight
        self.smoothness_weight = smoothness_weight
        self.spin_penalty_weight = spin_penalty_weight
        self.drift_penalty_weight = drift_penalty_weight
        self.lateral_penalty_weight = lateral_penalty_weight
        self.heading_weight = heading_weight
        self.gait_symmetry_weight = gait_symmetry_weight
        self.target_approach_weight = target_approach_weight
        self.target_reach_bonus = target_reach_bonus
        self.target_radius = target_radius
        self.target_max_speed = target_max_speed
        self.prey_distance_range = (prey_distance_range[0], prey_distance_range[1])
        self.prey_lateral_range = (prey_lateral_range[0], prey_lateral_range[1])
        self._initial_pos_2d = np.zeros(2)
        self._initial_prey_dir_2d = np.array([1.0, 0.0])
        self._prev_prey_distance: float | None = None
        self._prev_action: np.ndarray | None = None
        nominal_height = 0.218 if self.variant == "robot" else 0.244391440454
        height_range = healthy_z_range or (0.60 * nominal_height, 1.45 * nominal_height)
        if len(height_range) != 2 or not 0 < height_range[0] < nominal_height < height_range[1]:
            raise ValueError("healthy_z_range must enclose the model's home pelvis height")

        super().__init__(
            str(MODEL_PATHS[self.variant]),
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            forward_vel_weight=forward_vel_weight,
            alive_bonus=alive_bonus,
            energy_penalty_weight=energy_penalty_weight,
            fall_penalty=fall_penalty,
            healthy_z_range=height_range,
            max_tilt_angle=max_tilt_angle,
            reset_noise_scale=reset_noise_scale,
            reset_height_noise_scale=reset_height_noise_scale,
        )
        self.metadata = {**self.metadata, "render_fps": round(1 / self.dt)}

    def _cache_ids(self) -> None:
        self.pelvis_id = self.model.body("pelvis").id
        self.prey_id = self.model.body("prey").id
        self._target_mocap_id = int(self.model.body_mocapid[self.prey_id])
        self.floor_geom_id = self.model.geom("floor").id
        self._reset_keyframe_id = self.model.key("home").id
        self._home_ctrl = self.model.key_ctrl[self._reset_keyframe_id].copy()
        self._home_qpos = self.model.key_qpos[self._reset_keyframe_id].copy()
        self.target_standing_z = float(self._home_qpos[2])
        self._sensor_gyro_start = int(self.model.sensor("pelvis_gyro").adr[0])
        self._sensor_accel_start = int(self.model.sensor("pelvis_accel").adr[0])
        self._sensor_quat_start = int(self.model.sensor("diagnostic_pelvis_quat").adr[0])
        self._foot_sensor_groups = tuple(
            (int(self.model.sensor(name).adr[0]),) for name in ("r_foot_touch", "l_foot_touch")
        )
        foot_bodies = {
            int(self.model.site_bodyid[int(self.model.sensor(name).objid[0])])
            for name in ("r_foot_touch", "l_foot_touch")
        }
        dynamic_bodies = set(robot_body_ids(self.model))
        self._body_ground_geoms = {
            i
            for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) in dynamic_bodies - foot_bodies
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        }
        self.body_mass = float(self.model.body_mass[list(dynamic_bodies)].sum())
        self._contact_threshold = self.body_mass * abs(float(self.model.opt.gravity[2])) * 0.04
        self._actuated_qpos = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]].copy()
        self._init_gait_state(contact_threshold=self._contact_threshold)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        residual = np.clip(action, -1.0, 1.0)
        minimum, maximum = self.model.actuator_ctrlrange.T
        return cast(
            np.ndarray,
            self._home_ctrl
            + np.where(residual < 0, residual * (self._home_ctrl - minimum), residual * (maximum - self._home_ctrl)),
        )

    def _get_obs(self) -> np.ndarray:
        relative = self.data.mocap_pos[self._target_mocap_id] - self.data.xpos[self.pelvis_id]
        distance = np.linalg.norm(relative)
        sensors = self.data.sensordata
        return np.asarray(
            np.concatenate(
                [
                    self.data.qpos[7:],
                    self.data.qvel[6:],
                    sensors[self._sensor_quat_start : self._sensor_quat_start + 4],
                    sensors[self._sensor_gyro_start : self._sensor_gyro_start + 3],
                    self.data.qvel[:3],
                    sensors[self._sensor_accel_start : self._sensor_accel_start + 3],
                    self._foot_contact_forces(),
                    relative / (distance + 1e-8),
                    [distance],
                ]
            ),
            dtype=np.float32,
        )

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        position = self.data.xpos[self.pelvis_id]
        velocity = self.data.qvel[:2]
        forward = float(np.dot(velocity, self._initial_prey_dir_2d))
        lateral = float(np.dot(velocity, [-self._initial_prey_dir_2d[1], self._initial_prey_dir_2d[0]]))
        quaternion = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        tilt = self._quat_to_tilt(quaternion)
        heading = float(np.dot(self._quat_to_forward_2d(quaternion), self._initial_prey_dir_2d))
        height_error = (float(position[2]) - self.target_standing_z) / self.target_standing_z
        joint_error = self.data.qpos[self._actuated_qpos] - self._home_qpos[self._actuated_qpos]
        distance = float(np.linalg.norm(self.data.mocap_pos[self._target_mocap_id, :2] - position[:2]))
        progress = 0.0 if self._prev_prey_distance is None else (self._prev_prey_distance - distance) / self.dt
        self._prev_prey_distance = distance
        right, left = self._aggregated_foot_contact_forces()
        support = float(right + left > self._contact_threshold)
        _, status = self._is_terminated()
        success = float(status.get("success", False))
        smoothness, _ = self._reward_action_smoothness(action)
        gait, _ = self._compute_gait_symmetry(right, left, self.gait_symmetry_weight)
        components = {
            "reward_alive": self.alive_bonus * support,
            "reward_forward": self.forward_vel_weight
            * float(np.clip(forward, -self.forward_vel_max, self.forward_vel_max)),
            "reward_energy": -self.energy_penalty_weight * float(np.mean(np.square(action))),
            "reward_posture": self.posture_weight * float(np.exp(-np.square(tilt / 0.25))) * support,
            "reward_height": self.height_weight * float(np.exp(-np.square(height_error / 0.15))) * support,
            "reward_home_pose": -self.home_pose_weight * float(np.mean(np.square(joint_error))),
            "reward_smoothness": smoothness,
            "reward_spin": -self.spin_penalty_weight * float(np.dot(self.data.qvel[3:6], self.data.qvel[3:6])),
            "reward_drift": -self.drift_penalty_weight * float(np.linalg.norm(position[:2] - self._initial_pos_2d)),
            "reward_lateral": -self.lateral_penalty_weight * lateral**2,
            "reward_heading": self.heading_weight * heading,
            "reward_gait": gait,
            "reward_approach": self.target_approach_weight
            * float(np.clip(progress, -self.forward_vel_max, self.forward_vel_max)),
            "reward_target": self.target_reach_bonus * success,
        }
        reward = float(sum(components.values()))
        info = {
            **components,
            "reward_total": reward,
            "forward_vel": forward,
            "pelvis_height": float(position[2]),
            "tilt_angle": tilt,
            "prey_distance": distance,
            "heading_alignment": heading,
            "r_foot_contact": right,
            "l_foot_contact": left,
            "pelvis_angular_vel": float(np.linalg.norm(self.data.qvel[3:6])),
            "pelvis_yaw_vel": float(abs(self.data.qvel[5])),
            "drift_distance": float(np.linalg.norm(position[:2] - self._initial_pos_2d)),
            "target_success": success,
            "is_success": bool(success),
        }
        info.update(derive_stance_info(info))
        return reward, info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return True, {"termination_reason": "nonfinite_state", "success": False}
        height = float(self.data.xpos[self.pelvis_id, 2])
        quaternion = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        terminated, reason = self._check_height_tilt_termination(height, self._quat_to_tilt(quaternion))
        if terminated:
            return True, {"termination_reason": reason, "success": False}
        terminated, reason = self._check_floor_contact(self._body_ground_geoms, self.floor_geom_id)
        if terminated:
            return True, {"termination_reason": reason, "success": False}
        distance = float(
            np.linalg.norm(self.data.mocap_pos[self._target_mocap_id, :2] - self.data.xpos[self.pelvis_id, :2])
        )
        if (
            self.target_reach_bonus > 0
            and distance <= self.target_radius
            and np.linalg.norm(self.data.qvel[:2]) <= self.target_max_speed
        ):
            return True, {"termination_reason": "target_reached", "success": True}
        return False, {"success": False}

    def _spawn_target(self) -> None:
        position = self._spawn_target_2d(self.prey_distance_range, self.prey_lateral_range, 0.025)
        self._initial_pos_2d = self.data.qpos[:2].copy()
        direction = position[:2] - self._initial_pos_2d
        self._initial_prey_dir_2d = direction / (np.linalg.norm(direction) + 1e-8)
        self._prev_prey_distance = float(np.linalg.norm(direction))
        self._prev_action = None
        self._prev_prev_action = None
        self._reset_gait_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected a finite action with shape {self.action_space.shape}")
        return super().step(np.clip(action, -1.0, 1.0))

    def render_head_camera(self) -> np.ndarray:
        """Return the single 640×480 RGB camera image, separate from MLP state."""
        renderer: Any = self._renderer
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer = renderer
        renderer.update_scene(self.data, camera="head_camera")
        return np.asarray(renderer.render()).copy()


class CompsognathusRobotEnv(CompsognathusEnv):
    """The 12-servo robot, with the original fixed head and unpowered tail."""

    variant = "robot"
    _camera_distance = 0.9


gym.register(
    id="MesozoicLabs/Compsognathus-v0",
    entry_point="environments.compsognathus.envs:CompsognathusEnv",
)
gym.register(
    id="MesozoicLabs/CompsognathusRobot-v0",
    entry_point="environments.compsognathus.envs:CompsognathusRobotEnv",
)
