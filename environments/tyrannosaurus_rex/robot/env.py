"""The mini T. rex skeleton on the shared SB3 training pipeline.

The species' heading-free observation, residual actions and action filter are
reused. Robot rewards and terminations use this chassis' dimensions; anatomical
head, tail and settled-pose constants have no meaning on the fixed-load skeleton.
State velocity and foot forces are privileged simulation observations. This is
not a hardware driver or a validated sim-to-real policy interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.tyrannosaurus_rex.base_env import TyrannosaurusRexEnv


class TyrannosaurusRexRobotEnv(TyrannosaurusRexEnv):
    """Torso A, ten full-size servo envelopes, forward-folding legs, fixed loads."""

    embodiment = "robot"
    MODEL_PATH = ModelId("tyrannosaurus_rex", "robot").asset_path(ENVIRONMENTS_ROOT)
    nominal_pelvis_height = 0.113
    _camera_distance = 0.65
    # The species base infers fixed-length tuple types for anatomical data.
    _terrain_contact_probe_geoms = ()  # type: ignore[assignment]
    OPT_IN_TASK_KWARGS = ()  # type: ignore[assignment]
    BEHAVIOR_SECTION_LABEL = "Target reaching"
    SUCCESS_KEYS = ("target_reach",)  # type: ignore[assignment]
    FOOT_SUPPORT = {
        "support": ("{side}_flat_pad_collision",),
        "sole": "{side}_flat_pad_collision",
        "foot_site": "{side}_sole_center",
        "ankle": "{side}_ankle",
        "heel_line": -0.028,
        "toe_line": 0.042,
    }
    REWARD_PROBE = (0.01, "body", "pelvis")

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 1000,
        forward_vel_weight: float = 0.0,
        forward_vel_max: float = 0.10,
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
        target_radius: float = 0.04,
        target_max_speed: float = 0.05,
        prey_distance_range: tuple[float, float] = (0.5, 0.8),
        prey_lateral_range: tuple[float, float] = (0.0, 0.0),
        healthy_z_range: tuple[float, float] = (0.065, 0.16),
        max_tilt_angle: float = 0.7,
        reset_noise_scale: float = 0.005,
        reset_height_noise_scale: float = 0.0,
        perturbation_capture_velocity_multiple: float = 0.0,
        perturbation_interval: float = 2.0,
        perturbation_jitter: float = 0.25,
        perturbation_duration: float = 0.20,
        perturbation_direction: str = "uniform_horizontal",
        command_mode: str = "none",
        command_config: DirectionCommandConfig | None = None,
    ):
        # Explicit defaults are part of the task fingerprint. Do not hide them
        # behind **kwargs or inherit the anatomical robot's metre-scale defaults.
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
        self.prey_distance_range = prey_distance_range
        self.prey_lateral_range = prey_lateral_range
        self._initial_pos_2d = np.zeros(2)
        self._initial_prey_dir_2d = np.array([1.0, 0.0])
        self._prev_prey_distance: float | None = None
        self._prev_action: np.ndarray | None = None
        self._policy_action: np.ndarray | None = None
        self._policy_action_step = -1
        if not 0 < healthy_z_range[0] < self.nominal_pelvis_height < healthy_z_range[1]:
            raise ValueError("healthy_z_range must enclose the skeleton's home pelvis height")
        if forward_vel_max <= 0 or target_radius <= 0 or target_max_speed <= 0:
            raise ValueError("speed and target limits must be positive")
        # Initialise the shared engine directly: the species constructor sets
        # anatomical reward state that the skeleton deliberately does not use.
        BaseDinoEnv.__init__(
            self,
            str(self.MODEL_PATH),
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
            command_mode=command_mode,
            command_config=command_config,
        )
        self.metadata = {**self.metadata, "render_fps": round(1 / self.dt)}

    def _cache_ids(self) -> None:
        self._cache_home_keyframe("Mini T-Rex skeleton")
        self._home_qpos = self.model.key_qpos[self.home_keyframe_id].copy()
        self.target_standing_z = float(self._home_qpos[2])
        self.pelvis_id = self.model.body("pelvis").id
        self.floor_geom_id = self.model.geom("floor").id
        self.imu_site_id = self.model.site("imu").id
        self.r_foot_site_id = self.model.site("r_foot").id
        self.l_foot_site_id = self.model.site("l_foot").id
        self._foot_sensor_groups = tuple(
            (int(self.model.sensor_adr[self.model.sensor(name).id]),) for name in ("r_foot_touch", "l_foot_touch")
        )
        feet = {self.model.body(side + "_foot").id for side in ("left", "right")}
        self._body_ground_geoms = {
            i
            for i in range(self.model.ngeom)
            if self.model.body_rootid[self.model.geom_bodyid[i]] == self.pelvis_id
            and self.model.geom_bodyid[i] not in feet
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        }
        self.body_mass = float(self.model.body_subtreemass[self.pelvis_id])
        self._contact_threshold = self.body_mass * abs(float(self.model.opt.gravity[2])) * 0.04
        self._actuated_qpos = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]].copy()
        self._init_gait_state(contact_threshold=self._contact_threshold)

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        position = self.data.xpos[self.pelvis_id]
        velocity = self.data.qvel[:2]
        forward = float(np.dot(velocity, self._initial_prey_dir_2d))
        lateral = float(np.dot(velocity, [-self._initial_prey_dir_2d[1], self._initial_prey_dir_2d[0]]))
        quaternion = self.data.qpos[3:7]
        tilt = self._quat_to_tilt(quaternion)
        heading = float(np.dot(self._quat_to_forward_2d(quaternion), self._initial_prey_dir_2d))
        height = self._clearance(position)
        height_error = (height - self.target_standing_z) / self.target_standing_z
        joint_error = self.data.qpos[self._actuated_qpos] - self._home_qpos[self._actuated_qpos]
        distance = float(np.linalg.norm(self.data.mocap_pos[0, :2] - position[:2]))
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
            "pelvis_height": height,
            "tilt_angle": tilt,
            "prey_distance": distance,
            "r_foot_contact": right,
            "l_foot_contact": left,
            "target_reach": success,
        }
        return reward, info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        terminated, info = self._root_termination(self.pelvis_id, "pelvis_height")
        if terminated:
            return True, info
        hit, reason = self._check_floor_contact(self._body_ground_geoms, self.floor_geom_id)
        if hit:
            info["termination_reason"] = reason
            return True, info
        distance = np.linalg.norm(self.data.mocap_pos[0, :2] - self.data.xpos[self.pelvis_id, :2])
        if (
            self.target_reach_bonus > 0
            and distance < self.target_radius
            and np.linalg.norm(self.data.qvel[:2]) < self.target_max_speed
        ):
            info.update(success=True, termination_reason="target_reach")
            return True, info
        return False, info

    def _spawn_target(self) -> None:
        target = self._spawn_target_2d(self.prey_distance_range, self.prey_lateral_range, self.nominal_pelvis_height)
        self._initial_prey_dir_2d = self._compute_initial_direction_2d(target)
        self._initial_pos_2d = self.data.qpos[:2].copy()
        self._prev_prey_distance = None
        self._prev_action = None
        self._policy_action = None
        self._policy_action_step = -1
        self._reset_gait_state()
