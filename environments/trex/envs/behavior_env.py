"""Opt-in SB3 direction tracking on reproducible terrain.

This pilot keeps the r13 observation order and articulated animal, but has its
own task identity. It must not be certified by the canonical locomotion gates.
The ordinary TRexEnv and its seeded reset contract are unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.direction_commands import (
    DirectionCommandConfig,
    DirectionCommandController,
    DirectionCommandState,
    gaussian_tracking_reward,
    tracking_metrics,
    wrap_angle,
)
from environments.shared.plant_contract import current_plant_identity, validate_compiled_plant
from environments.shared.reward_functions import reward_head_clearance, reward_target_centered_height
from environments.shared.terrain import (
    TerrainConfig,
    TerrainRealization,
    apply_terrain,
    build_terrain_model,
    generate_terrain,
)
from environments.trex.envs.trex_env import TRexEnv


class TRexBehaviorEnv(TRexEnv):
    """Heading/speed commands with optional terrain regenerated on each reset.

    ``reset(seed=S)`` repeats the same pose, commands and terrain. Unseeded
    subsequent resets advance a separate terrain/command episode stream.
    Different ``run_seed`` values generate different courses. Terrain queries
    are used for scoring and safety, never appended as privileged policy input.
    """

    def __init__(
        self,
        *,
        commands: DirectionCommandConfig | None = None,
        terrain: TerrainConfig | None = None,
        run_seed: int = 42,
        tracking_weight: float = 2.5,
        course_distance: float = 10.0,
        flat_probability: float = 0.0,
        **env_kwargs: Any,
    ):
        if run_seed < 0 or not np.isfinite(tracking_weight) or tracking_weight <= 0:
            raise ValueError("run_seed must be nonnegative and tracking_weight finite and positive")
        if not np.isfinite(course_distance) or course_distance <= 0:
            raise ValueError("course_distance must be finite and positive")
        if not np.isfinite(flat_probability) or not 0 <= flat_probability <= 1:
            raise ValueError("flat_probability must be in [0, 1]")
        if env_kwargs.get("command_mode", "none") != "none":
            raise ValueError("Use the commands argument for this behavior environment")
        self.direction_controller = DirectionCommandController(commands or DirectionCommandConfig())
        self.terrain_config = terrain
        self.run_seed = int(run_seed)
        self._episode_index = 0
        self._episode_seed = 0
        self.tracking_weight = float(tracking_weight)
        self.course_distance = float(course_distance)
        self.flat_probability = float(flat_probability)
        self.terrain: TerrainRealization | None = None
        self._clearance_min: np.ndarray | None = None
        self._neck_hit = False
        self._heading_before = 0.0
        self._command_state: DirectionCommandState | None = None
        self._command_metrics: dict[str, Any] = {}
        self._tracking_dwell_s = 0.0
        self._max_radius = 0.0
        # These objectives would pay for ignoring a requested turn or stop.
        for name in (
            "forward_vel_weight",
            "heading_weight",
            "lateral_penalty_weight",
            "backward_vel_penalty_weight",
            "drift_penalty_weight",
            "spin_penalty_weight",
            "speed_penalty_weight",
            "idle_penalty_weight",
            "bite_bonus",
            "bite_approach_weight",
            "bite_head_proximity_weight",
        ):
            env_kwargs[name] = 0.0
        env_kwargs.setdefault("max_episode_steps", 2500)
        super().__init__(**env_kwargs)
        self.parent_plant_identity = current_plant_identity("trex")
        validate_compiled_plant(self.model, self.parent_plant_identity)
        # Keep separately compiled contact models. Even a perfectly level
        # heightfield has different narrow-phase contacts from the authored
        # plane, so retention episodes must use the plane itself.
        self._home_ground_clearance_m = self.home_ground_clearance()
        self._spawn_model = copy.copy(self.model)
        self._spawn_data = mujoco.MjData(self._spawn_model)
        self._plane_model, self._plane_data = self.model, self.data
        self._terrain_model: mujoco.MjModel | None = None
        self._terrain_data: mujoco.MjData | None = None
        self._terrain_probe_model: mujoco.MjModel | None = None
        self._terrain_probe_data: mujoco.MjData | None = None
        path = Path(__file__).parent.parent / "assets" / "trex.xml"
        initial_terrain = None
        if terrain is not None:
            initial_terrain = generate_terrain(terrain, run_seed=self.run_seed, episode_index=0)
            self._terrain_model = build_terrain_model(path, initial_terrain)
            self._assert_same_animal(self._plane_model, self._terrain_model)
            self._assert_matching_ids(self._plane_model, self._terrain_model)
            self._terrain_data = mujoco.MjData(self._terrain_model)
        # Retain the parent's unrelated prey observation distribution, but the
        # prop cannot physically obstruct a non-hunting route.
        for model in (self._plane_model, self._terrain_model):
            if model is not None:
                model.geom_contype[self.prey_geom_id] = 0
                model.geom_conaffinity[self.prey_geom_id] = 0
        self._neck_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "neck_geom")
        # A separate collision-only model detects neck/surface intersections
        # without making the non-colliding neck support the trained animal.
        # mj_geomDistance does not return surface distance for hfields.
        # Compile masks before building the broad-phase body/BVH caches.
        # Flipping geom masks on a copy of an already-compiled model misses
        # bodies that had no colliding geoms in the canonical model.
        probe_spec = mujoco.MjSpec.from_file(str(path))
        probe_spec.geom("neck_geom").contype = 1
        probe_spec.geom("neck_geom").conaffinity = 1
        self._plane_probe_model = probe_spec.compile()
        self._assert_matching_ids(self._plane_model, self._plane_probe_model)
        self._plane_probe_data = mujoco.MjData(self._plane_probe_model)
        if initial_terrain is not None:
            self._terrain_probe_model = build_terrain_model(probe_spec, initial_terrain)
            self._assert_matching_ids(self._plane_model, self._terrain_probe_model)
            self._terrain_probe_data = mujoco.MjData(self._terrain_probe_model)
        self._probe_model, self._probe_data = self._plane_probe_model, self._plane_probe_data
        self._substep_probe_hook = self._probe_ground_clearance
        parameters = {
            key: value.default
            for key, value in inspect.signature(TRexEnv.__init__).parameters.items()
            if key != "self" and value.default is not inspect.Parameter.empty
        }
        parameters.update(env_kwargs)
        parameters.pop("render_mode", None)
        self._behavior_parameters = parameters

    @staticmethod
    def _assert_matching_ids(parent: mujoco.MjModel, child: mujoco.MjModel) -> None:
        """Pools share all IDs consumed by state, sensor and contact code."""
        for field in ("nq", "nv", "nmocap", "nsensordata"):
            if getattr(parent, field) != getattr(child, field):
                raise ValueError(f"Behavior model pool changed {field}")
        for field, kind in (
            ("nbody", mujoco.mjtObj.mjOBJ_BODY),
            ("njnt", mujoco.mjtObj.mjOBJ_JOINT),
            ("ngeom", mujoco.mjtObj.mjOBJ_GEOM),
            ("nsite", mujoco.mjtObj.mjOBJ_SITE),
            ("nsensor", mujoco.mjtObj.mjOBJ_SENSOR),
            ("nu", mujoco.mjtObj.mjOBJ_ACTUATOR),
            ("nkey", mujoco.mjtObj.mjOBJ_KEY),
        ):
            count = getattr(parent, field)
            if count != getattr(child, field) or any(
                mujoco.mj_id2name(parent, kind, i) != mujoco.mj_id2name(child, kind, i) for i in range(count)
            ):
                raise ValueError(f"Behavior model pool changed {field} IDs")

    def _select_contact_model(self, *, use_terrain: bool) -> None:
        """Select a precompiled scene and discard episode-dependent caches."""
        if use_terrain:
            assert self._terrain_model is not None and self._terrain_data is not None
            assert self._terrain_probe_model is not None and self._terrain_probe_data is not None
            model, data = self._terrain_model, self._terrain_data
            probe_model, probe_data = self._terrain_probe_model, self._terrain_probe_data
        else:
            model, data = self._plane_model, self._plane_data
            probe_model, probe_data = self._plane_probe_model, self._plane_probe_data
        if self.model is not model:
            # Renderer/viewer instances retain model pointers. Recreate them
            # lazily on render, never retarget a live instance or geom type.
            self.close()
            self._camera = None
        self.model, self.data = model, data
        self._probe_model, self._probe_data = probe_model, probe_data
        self._cache_ids()
        self._root_subtree_geom_ids = None
        self._static_floor_geom_ids = None
        self._ground_geom_array = None
        self._invalidate_substep_aggregates()
        self._action_filter_state = None
        mujoco.mj_resetData(self._probe_model, self._probe_data)

    @staticmethod
    def _assert_same_animal(parent: mujoco.MjModel, child: mujoco.MjModel) -> None:
        """Reject accidental changes outside the declared static floor asset."""
        if (parent.nq, parent.nv, parent.nu, parent.ngeom) != (child.nq, child.nv, child.nu, child.ngeom):
            raise ValueError("Terrain changed the articulated model layout")
        # Compare all exposed model arrays for the animal's kinematics,
        # dynamics, actuation, sensing, constraints and reset keyframes.
        prefixes = ("body_", "jnt_", "dof_", "actuator_", "sensor_", "site_", "key_", "eq_", "pair_", "exclude_")
        for name in dir(parent):
            if name.startswith(prefixes):
                a, b = getattr(parent, name), getattr(child, name)
                if isinstance(a, np.ndarray) and not np.array_equal(a, b):
                    raise ValueError(f"Terrain changed animal field {name}")
        floor = mujoco.mj_name2id(parent, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        animal = np.arange(parent.ngeom) != floor
        for name in dir(parent):
            if name.startswith("geom_"):
                a, b = getattr(parent, name), getattr(child, name)
                if isinstance(a, np.ndarray) and a.shape and a.shape[0] == parent.ngeom:
                    if not np.array_equal(a[animal], b[animal]):
                        raise ValueError(f"Terrain changed animal field {name}")
        for name in dir(parent.opt):
            if not name.startswith("_") and not callable(value := getattr(parent.opt, name)):
                if not np.array_equal(value, getattr(child.opt, name)):
                    raise ValueError(f"Terrain changed physics option {name}")

    @property
    def behavior_identity(self) -> dict[str, Any]:
        """Configuration/source identity; episode seeds live in run manifests."""
        source_paths = (
            Path(__file__),
            Path(inspect.getfile(TRexEnv)),
            Path(inspect.getfile(DirectionCommandController)),
            Path(inspect.getfile(TerrainConfig)),
        )
        identity = {
            "schema": "mesozoic.trex-command-terrain/v1",
            "backend": "stable-baselines3",
            "parent_plant": self.parent_plant_identity.to_dict(),
            "commands": asdict(self.direction_controller.config),
            "terrain": asdict(self.terrain_config) if self.terrain_config is not None else None,
            "env": self._behavior_parameters,
            "tracking_weight": self.tracking_weight,
            "course_distance": self.course_distance,
            "flat_probability": self.flat_probability,
            "prey_collision": False,
            "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        }
        return dict(json.loads(json.dumps(identity)))

    def command_manifest(self) -> dict[str, Any]:
        return self.direction_controller.manifest()

    def _heading(self) -> float:
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _clearance(self, xyz: np.ndarray) -> float:
        ground = 0.0 if self.terrain is None else self.terrain.height_at(float(xyz[0]), float(xyz[1]))
        if not np.isfinite(ground):
            # The boundary check terminates this state; avoid propagating NaN
            # into reward/observation before it runs.
            ground = 0.0
        return float(xyz[2] - ground)

    def _current_clearances(self) -> np.ndarray:
        return np.array(
            [
                self._clearance(self.data.xpos[self.pelvis_id]),
                self._clearance(self.data.site_xpos[self.head_tip_site_id]),
                self._clearance(self.data.xpos[self.skull_body_id]),
            ]
        )

    def _probe_ground_clearance(self) -> None:
        heights = self._current_clearances()
        self._clearance_min = heights if self._clearance_min is None else np.minimum(self._clearance_min, heights)
        self._probe_data.qpos[:] = self.data.qpos
        self._probe_data.mocap_pos[:] = self.data.mocap_pos
        self._probe_data.mocap_quat[:] = self.data.mocap_quat
        mujoco.mj_kinematics(self._probe_model, self._probe_data)
        mujoco.mj_collision(self._probe_model, self._probe_data)
        pairs = self._probe_data.contact.geom
        if len(pairs):
            hits = ((pairs[:, 0] == self._neck_geom_id) & (pairs[:, 1] == self.floor_geom_id)) | (
                (pairs[:, 1] == self._neck_geom_id) & (pairs[:, 0] == self.floor_geom_id)
            )
            self._neck_hit |= bool(np.any(self._probe_data.contact.dist[hits] < 0.0))

    def _settle_root_on_ground(self, clearance: float | None = None) -> float:
        # Every generated surface is exactly flat under the complete spawn
        # pose; the canonical single vertical translation remains exact here.
        if self.terrain is None:
            return super()._settle_root_on_ground(clearance)
        else:
            mujoco.mj_forward(self.model, self.data)
            geoms = self._root_subtree_geoms()
            xy = self.data.geom_xpos[geoms, :2]
            radii = self.model.geom_rbound[geoms]
            if np.any(np.linalg.norm(xy, axis=1) + radii >= self.terrain.config.apron_radius):
                raise ValueError("Spawn apron does not contain the complete animal")
            self._spawn_data.qpos[:] = self.data.qpos
            mujoco.mj_forward(self._spawn_model, self._spawn_data)
            worst = min(
                float(
                    mujoco.mj_geomDistance(self._spawn_model, self._spawn_data, int(g), self.floor_geom_id, 10.0, None)
                )
                for g in geoms
            )
            target = self.home_ground_clearance() if clearance is None else clearance
            shift = target - worst
            self.data.qpos[2] += shift
            mujoco.mj_forward(self.model, self.data)
            # Contact penetration is checked through the actual hfield narrow
            # phase, independently of the plane placement calculation.
            contacts = self.data.contact
            floor_hits = np.any(contacts.geom == self.floor_geom_id, axis=1)
            if np.any(contacts.dist[floor_hits] < target - 0.002):
                raise ValueError("Terrain reset produced excessive ground penetration")
            return float(shift)

    def lowest_ground_clearance(self, data: mujoco.MjData | None = None) -> float:
        if self.terrain is not None:
            raise NotImplementedError(
                "General geom distance is invalid for heightfields; use terrain-relative body clearances and contacts"
            )
        return super().lowest_ground_clearance(data)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            if seed < 0:
                raise ValueError("seed must be nonnegative")
            self._episode_seed = int(seed)
            self._episode_index = 0
        # Independent streams keep command draws from changing a terrain map
        # or the parent's pose-noise sequence.
        terrain_seed = int(np.random.SeedSequence([self.run_seed, self._episode_seed, 0x7E22]).generate_state(1)[0])
        command_seed = int(
            np.random.SeedSequence([self.run_seed, self._episode_seed, self._episode_index, 0xC044]).generate_state(1)[
                0
            ]
        )
        use_terrain = False
        if self.terrain_config is not None:
            draw = np.random.default_rng(
                np.random.SeedSequence([self.run_seed, self._episode_seed, self._episode_index, 0xF1A7])
            ).random()
            use_terrain = bool(draw >= self.flat_probability)
        self._select_contact_model(use_terrain=use_terrain)
        self.terrain = None
        if use_terrain:
            assert self.terrain_config is not None
            self.terrain = generate_terrain(
                self.terrain_config, run_seed=terrain_seed, episode_index=self._episode_index
            )
            apply_terrain(self.model, self.terrain, self.data)
            apply_terrain(self._probe_model, self.terrain, self._probe_data)
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            if self._viewer is not None:
                self._viewer.update_hfield(0)
        self._clearance_min = None
        self._neck_hit = False
        self._tracking_dwell_s = 0.0
        self._max_radius = 0.0
        _, info = super().reset(seed=seed, options=options)
        self._command_state = self.direction_controller.reset(np.random.default_rng(command_seed), self._heading())
        self._command = self._command_state.normalized.copy()
        info.update(
            {
                "run_seed": self.run_seed,
                "episode_seed": self._episode_seed,
                "episode_index": self._episode_index,
                "command_seed": command_seed,
                "terrain": self.terrain.manifest() if self.terrain is not None else {"family": "flat_plane"},
            }
        )
        self._episode_index += 1
        return self._get_obs(), info

    def set_direction(self, heading: float, speed: float) -> np.ndarray:
        """Set a persistent world heading (radians) and speed (m/s).

        Returns the updated observation to use for the next policy action.
        """
        self.direction_controller.set_target(heading, speed, time_s=self._step_count * self.dt)
        self._command_state = self.direction_controller.update(self._step_count * self.dt, self._heading())
        self._command = self._command_state.normalized.copy()
        self._tracking_dwell_s = 0.0
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._command_state is None:
            raise RuntimeError("reset must be called before step")
        self._heading_before = self._heading()
        self._clearance_min = None
        self._neck_hit = False
        _, reward, terminated, truncated, info = super().step(action)
        info.update(self._command_metrics)
        radius = float(np.linalg.norm(self.data.qpos[:2] - self._initial_pos_2d))
        self._max_radius = max(self._max_radius, radius)
        # This is a traversal diagnostic, not an automatic training gate. A
        # complete route evaluator additionally checks heading/speed and falls.
        info["course_progress_m"] = radius
        info["course_reached"] = bool(self._max_radius >= self.course_distance)
        info["tracking_dwell_s"] = self._tracking_dwell_s
        info["terrain_height_m"] = float(
            self.data.xpos[self.pelvis_id, 2] - self._clearance(self.data.xpos[self.pelvis_id])
        )
        # Keep the base is_success=False: radial displacement alone cannot
        # certify command following or a terrain route.
        previous_event = self._command_state.event_id
        self._command_state = self.direction_controller.update(self._step_count * self.dt, self._heading())
        self._command = self._command_state.normalized.copy()
        if previous_event != self._command_state.event_id:
            self._tracking_dwell_s = 0.0
        return self._get_obs(), reward, terminated, truncated, info

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        reward, info = super()._get_reward_info(action)
        pelvis = self._clearance(self.data.xpos[self.pelvis_id])
        target = 0.9260
        error = abs(pelvis - target)
        if self.height_target_tolerance > 0:
            new_height_reward, error, quality = map(
                float,
                reward_target_centered_height(
                    np.asarray(pelvis), target, self.height_target_tolerance, self.height_weight
                ),
            )
        else:
            quality = float(np.clip((pelvis - self.healthy_z_range[0]) / (target - self.healthy_z_range[0]), 0, 1))
            new_height_reward = self.height_weight * quality
        reward += new_height_reward - info["reward_height"]
        info.update(
            pelvis_clearance=pelvis, height_error=error, height_quality=quality, reward_height=new_height_reward
        )
        # Head clearance is also measured against the surface below the snout.
        head = self._clearance(self.data.site_xpos[self.head_tip_site_id])
        head_reward, head_quality = reward_head_clearance(
            np.asarray(head), self.head_clearance_target, self.head_clearance_tolerance, self.head_clearance_weight
        )
        if "reward_head_clearance" in info:
            replacement = float(head_reward)
            reward += replacement - info["reward_head_clearance"]
            info["reward_head_clearance"] = replacement
            info["head_clearance_quality"] = float(head_quality)
        info["head_clearance_m"] = head
        if self._command_state is not None:
            yaw_rate = float(wrap_angle(self._heading() - self._heading_before) / self.dt)
            metrics = tracking_metrics(
                self._command_state, self.data.qvel[:2], yaw_rate, current_heading=self._heading()
            )
            self._command_metrics = {**self._command_state.as_info(), **metrics}
            self._command_metrics["actual_heading"] = self._heading()
            self._command_metrics["heading_error_rad"] = (
                float(wrap_angle(self._command_state.desired_heading - self._heading()))
                if self._command_state.heading_active
                else 0.0
            )
            tracking = gaussian_tracking_reward(metrics["tracking_error_v"], metrics["tracking_error_yaw"])
            reward += self.tracking_weight * tracking
            info["reward_tracking"] = self.tracking_weight * tracking
            self._tracking_dwell_s = self._tracking_dwell_s + self.dt if metrics["tracking_in_tolerance"] else 0.0
        info["reward_total"] = reward
        return reward, info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        clearances = self._current_clearances()
        minima = clearances if self._clearance_min is None else self._clearance_min
        quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        tilt = self._quat_to_tilt(quat)
        info: dict[str, Any] = {"pelvis_clearance": float(clearances[0]), "tilt_angle": tilt}
        terminated, reason = self._check_height_tilt_termination(float(minima[0]), tilt)
        if (
            not terminated
            and self._quat_to_forward_z(quat) < self._natural_forward_z - self.nosedive_termination_threshold
        ):
            terminated, reason = True, "nosedive"
        if not terminated and minima[1] < 0.12:
            terminated, reason = True, "head_contact"
        if not terminated and minima[2] < 0.45:
            terminated, reason = True, "skull_low"
        if not terminated and self._neck_hit:
            terminated, reason = True, "neck_ground_contact"
        if not terminated:
            terminated, reason = self._check_floor_contact(self._body_ground_geoms, self.floor_geom_id)
        if not terminated and self.terrain is not None:
            # Keep the complete animal away from the finite heightfield edge.
            if np.any(np.abs(self.data.qpos[:2]) >= self.terrain.config.extent - 3.0):
                terminated, reason = True, "terrain_boundary"
        if terminated:
            info["termination_reason"] = reason
        return terminated, info
