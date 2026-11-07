"""The independent 60 cm robot embodiment. The walking actor sees only sensor packets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.intermediate_drive import InstalledDrive
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.shared.robot_floor import apply_surface
from environments.shared.robot_walking import pack_sensors, reference_angles, update_gravity
from environments.tyrannosaurus_rex.base_env import TyrannosaurusRexEnv


class TyrannosaurusRexIntermediateEnv(TyrannosaurusRexEnv):
    embodiment = "intermediate"
    MODEL_PATH = ModelId("tyrannosaurus_rex", "intermediate").asset_path(ENVIRONMENTS_ROOT)
    supported_training_backends = ("stable-baselines3",)
    action_mapping = "periodic-joint-residual/v1"
    action_filter_cutoff_hz = 0.0  # drive owns command transport and filtering
    OPT_IN_TASK_KWARGS = ()  # type: ignore[assignment]
    nominal_pelvis_height = 0.36
    HEIGHT_LABEL = "Pelvis height"
    BEHAVIOR_SECTION_LABEL = "Command tracking"
    SUCCESS_KEYS = ("command_tracking",)  # type: ignore[assignment]
    REWARD_PROBE = (0.01, "body", "pelvis")
    FOOT_SUPPORT = {
        "support": ("{side}_sole",),
        "sole": "{side}_sole",
        "foot_site": "{side}_sole_center",
    }
    _camera_distance = 1.65

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 360,
        task: str = "locomotion",
        surface: str = "hard",
        reset_noise_scale: float = 0.006,
        scene_xml: str | Path | None = None,
    ):
        if task not in ("stance", "locomotion") or surface not in ("hard", "carpet", "mixed"):
            raise ValueError("Unknown walking task or surface")
        self.task = task
        self.surface = surface
        self.expression = np.zeros(2)
        self.phase = 0.0
        self.amplitude = 0.0
        self.last_action = np.zeros(14)
        self.up = np.array([0.0, 0.0, 1.0])
        self.substep_callback = None
        self._sensor_noise = np.zeros(36)
        self.external_command = False
        self.external_expression = False
        BaseDinoEnv.__init__(
            self,
            str(scene_xml or self.MODEL_PATH),
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            healthy_z_range=(0.20, 0.50),
            max_tilt_angle=0.65,
            reset_noise_scale=reset_noise_scale,
            reset_height_noise_scale=0.0,
            fall_penalty=-12.0,
        )
        self.drive = InstalledDrive(self.model)
        self.metadata = {**self.metadata, "render_fps": round(1 / self.dt)}

    def _cache_ids(self) -> None:
        self._cache_home_keyframe("Intermediate T. rex")
        self.pelvis_id = self.model.body("pelvis").id
        self.floor_geom_id = self.model.geom("floor").id
        self._sensor_quat_start = self.model.sensor("pelvis_orientation").adr[0]
        self.targets = self.model.actuator_ctrlrange.copy()
        joint_ids = self.model.actuator_trnid[:, 0]
        self.enc_q = self.model.jnt_qposadr[joint_ids]
        self.enc_v = self.model.jnt_dofadr[joint_ids]
        self.gyro_adr = self.model.sensor("pelvis_gyro").adr[0]
        self.accel_adr = self.model.sensor("pelvis_accelerometer").adr[0]
        self.load_adr = [self.model.sensor(s + "_contact").adr[0] for s in ("left", "right")]
        self._foot_sensor_groups = tuple((int(i),) for i in reversed(self.load_adr))
        self._ground_mask = np.zeros(self.model.ngeom, dtype=bool)
        self.soles = [self.model.site(s + "_sole_center").id for s in ("left", "right")]
        self.r_foot_site_id, self.l_foot_site_id = reversed(self.soles)
        self._body_ground_geoms = {
            i
            for i, body in enumerate(self.model.geom_bodyid)
            if self.model.body_rootid[body] == self.pelvis_id
            and body not in (self.model.body("left_foot").id, self.model.body("right_foot").id)
        }

        self._ground_mask[list(self._body_ground_geoms)] = True

    def observe(self) -> np.ndarray:
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        sensors = self.data.sensordata
        return pack_sensors(
            self.data.qpos[self.enc_q] + self._sensor_noise[:14],
            self.data.qvel[self.enc_v] + self._sensor_noise[14:28],
            self.up,
            sensors[self.gyro_adr : self.gyro_adr + 3] + self._sensor_noise[28:31],
            np.maximum(0, sensors[self.load_adr] + self._sensor_noise[34:36]),
            self.phase,
            self.expression,
            self.last_action,
            self._command,
        )

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        reference = reference_angles(self.phase, self._command, self.expression, self.amplitude)
        residual = np.clip(action, -1, 1) * np.array([0.09, 0.13, 0.16, 0.20, 0.16, 0.13] * 2 + [0.04, 0.05])
        return np.asarray(np.clip(reference + residual, self.targets[:, 0], self.targets[:, 1]))

    def set_velocity_command(self, forward_mps: float, yaw_rps: float) -> None:
        if not np.isfinite([forward_mps, yaw_rps]).all():
            raise ValueError("Velocity command must be finite")
        self.external_command = True
        self._command[:] = (np.clip(forward_mps, 0, 0.20), 0.0, np.clip(yaw_rps, -0.5, 0.5))

    def set_expression(self, neck_rad: float, tail_rad: float) -> None:
        if not np.isfinite([neck_rad, tail_rad]).all():
            raise ValueError("Expression command must be finite")
        self.external_expression = True
        self.expression[:] = np.clip([neck_rad, tail_rad], self.targets[-2:, 0], self.targets[-2:, 1])

    def _spawn_target(self) -> None:
        self.external_command = False
        self.external_expression = False

    def _draw_walk_command(self) -> None:
        if self.task == "stance" or self.np_random.random() < 0.10:
            self._command[:] = 0.0
        else:
            self._command[:] = (self.np_random.uniform(0.07, 0.20), 0.0, self.np_random.uniform(-0.50, 0.50))

    def reset(self, *, seed=None, options=None):
        _, info = BaseDinoEnv.reset(self, seed=seed, options=options)
        surface = self.np_random.choice(["hard", "carpet"]) if self.surface == "mixed" else self.surface
        apply_surface(self.model, surface)
        self.drive.reset(self.data)
        # The floor material and restored drive limits affect derived sensor
        # values. Recompute them before emitting the first encoder/IMU/load packet.
        mujoco.mj_forward(self.model, self.data)
        self.phase = self.amplitude = 0.0
        self.last_action.fill(0.0)
        self.expression.fill(0.0)
        self.up[:] = (0, 0, 1)  # calibrated upright at initialization
        self._sensor_noise.fill(0.0)
        self._draw_walk_command()
        info["surface"] = str(surface)
        return self._get_obs(), info

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        # Privileged evaluation signals are confined to reward and termination.
        body = self.data.xmat[self.pelvis_id].reshape(3, 3)
        velocity = body.T @ self.data.qvel[:3]
        yaw = self.data.sensordata[self.gyro_adr + 2]
        dv, dw = velocity[0] - self._command[0], yaw - self._command[2]
        force = self.data.sensordata[self.load_adr]
        swing = math.sin(self.phase)
        lifting = int(swing < 0)
        height = self.data.site_xpos[self.soles[lifting], 2]
        support = force[1 - lifting] > 25.0 and force[lifting] < 10.0
        swing_score = float(support) * min(1.0, max(0.0, height / 0.018)) * abs(swing)
        reward = (
            0.6 * float(body[2, 2])
            + 1.4 * math.exp(-((dv / 0.09) ** 2))
            + 0.65 * math.exp(-((dw / 0.22) ** 2))
            + self.amplitude * (1.4 * swing_score + 3.0 * np.clip(velocity[0], -0.15, 0.24))
            - 4.0 * velocity[1] ** 2
            - 0.10 * np.mean((action - self.last_action) ** 2)
        )
        return float(reward), {
            "reward_total": float(reward),
            "forward_vel": float(velocity[0]),
            "command_tracking": float(abs(dv) < 0.06 and abs(dw) < 0.15),
            "r_foot_contact": float(force[1]),
            "l_foot_contact": float(force[0]),
            "swing_score": swing_score,
        }

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        ended, info = self._root_termination(self.pelvis_id, "pelvis_height")
        if not ended:
            ended, reason = self._check_floor_contact(self._body_ground_geoms, self.floor_geom_id)
            if ended:
                info["termination_reason"] = reason
        return ended, info

    def step(self, action: np.ndarray):
        target = self._scale_action(action)
        self._substep_floor_hit_geom = None
        force_block = np.empty((self.frame_skip, len(self._foot_sensor_groups)))
        for substep in range(self.frame_skip):
            self.drive.step(self.data, target)
            if self._substep_probe_hook is not None:
                self._substep_probe_hook()
            if self.data.ncon and self._substep_floor_hit_geom is None:
                pairs = self.data.contact.geom
                ground_hits = ((pairs[:, 0] == self.floor_geom_id) & self._ground_mask[pairs[:, 1]]) | (
                    (pairs[:, 1] == self.floor_geom_id) & self._ground_mask[pairs[:, 0]]
                )
                if np.any(ground_hits):
                    pair = pairs[np.flatnonzero(ground_hits)[0]]
                    self._substep_floor_hit_geom = int(pair[0] if pair[1] == self.floor_geom_id else pair[1])
            force_block[substep] = self._foot_contact_forces()
            self._accumulate_substep(substep)
            if self.substep_callback is not None:
                self.substep_callback()
        self._step_count += 1
        self._substep_min_foot_forces = force_block.min(axis=0)
        self._substep_foot_forces = force_block
        self._substep_contact_step = self._step_count
        reward, info = self._get_reward_info(action)
        terminated, status = self._is_terminated()
        truncated = self._is_truncated()
        if terminated:
            reward += self.fall_penalty
        self.last_action = np.clip(action, -1, 1).copy()
        self._sensor_noise = self.np_random.standard_normal(36) * (
            [0.001] * 14 + [0.015] * 14 + [0.004] * 3 + [0.08] * 3 + [0.4] * 2
        )
        sensors = self.data.sensordata
        self.up = update_gravity(
            self.up,
            sensors[self.gyro_adr : self.gyro_adr + 3] + self._sensor_noise[28:31],
            sensors[self.accel_adr : self.accel_adr + 3] + self._sensor_noise[31:34],
            self.dt,
        )
        self.phase += self.dt * 2.0 * math.pi * 1.40
        moving = min(1.0, max(self._command[0] / 0.08, abs(self._command[2]) / 0.20))
        self.amplitude += np.clip(moving - self.amplitude, -2.5 * self.dt, 2.5 * self.dt)
        if not self.external_command and self._step_count % 120 == 0:
            self._draw_walk_command()
        if not self.external_expression and self.task == "locomotion":
            self.expression[:] = (0.09 * math.sin(0.23 * self.phase), 0.13 * math.sin(0.45 * self.phase))
        self._distance_traveled += float(np.linalg.norm(self.data.qpos[:2] - self._prev_pos_2d))
        self._prev_pos_2d = self.data.qpos[:2].copy()
        info.update(status, step=self._step_count, distance_traveled=self._distance_traveled, reward_total=reward)
        if terminated or truncated:
            info["is_success"] = bool(truncated and not terminated and info["command_tracking"])
        return self._get_obs(), reward, terminated, truncated, info
