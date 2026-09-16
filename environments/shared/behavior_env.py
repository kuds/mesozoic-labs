"""Direction and terrain behavior environments for every registered species.

The canonical animal, observation ordering and action mapping remain unchanged.
Task rewards and safety clearances use the actual surface under each body/site.
The original T. rex implementation is retained for saved-bundle compatibility.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.direction_commands import (
    DirectionCommandConfig,
    DirectionCommandController,
    DirectionCommandState,
    gaussian_tracking_reward,
    tracking_metrics,
    wrap_angle,
)
from environments.shared.plant_contract import REPOSITORY_ROOT, current_plant_identity, validate_compiled_plant
from environments.shared.species_names import resolve_species_id
from environments.shared.species_registry import get_species_config
from environments.shared.terrain import (
    TerrainConfig,
    TerrainRealization,
    apply_terrain,
    build_terrain_model,
    generate_terrain,
)
from environments.trex.envs.behavior_env import TRexBehaviorEnv


def canonical_env_parameters(species: str) -> dict[str, Any]:
    """Constructor defaults, including inherited robot-species parameters."""
    env_class = get_species_config(species).env_class
    parameters: dict[str, Any] = {}
    for cls in reversed(env_class.__mro__):
        if cls is BaseDinoEnv:
            continue
        for key, value in inspect.signature(getattr(cls, "__init__")).parameters.items():
            if key != "self" and value.default is not inspect.Parameter.empty:
                parameters[key] = value.default
    return parameters


class SpeciesBehaviorMixin(BaseDinoEnv):
    """Compose command tracking with each species' canonical balance behavior."""

    species: str
    _canonical_env_class: type[BaseDinoEnv]
    floor_geom_id: int
    target_standing_z: float
    height_weight: float
    _initial_pos_2d: np.ndarray

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
        if not isinstance(run_seed, (int, np.integer)) or isinstance(run_seed, bool) or not 0 <= run_seed <= 2**32 - 1:
            raise ValueError("run_seed must be an integer in [0, 2**32 - 1]")
        if not np.isfinite(tracking_weight) or tracking_weight <= 0:
            raise ValueError("tracking_weight must be finite and positive")
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
        self._clearance_max: np.ndarray | None = None
        self._heading_before = 0.0
        self._command_state: DirectionCommandState | None = None
        self._command_metrics: dict[str, Any] = {}
        self._tracking_dwell_s = 0.0
        self._max_radius = 0.0
        # Tolerances are fractions of the command's physical speed scale;
        # millimetre-scale robots must not pass a walk command while standing.
        speed_ratio = self.direction_controller.config.speed_scale / 1.5
        self.tracking_tolerances = {
            "velocity_tolerance": 0.2 * speed_ratio,
            "stop_speed_tolerance": 0.1 * speed_ratio,
        }
        self.tracking_velocity_sigma = 0.25 * speed_ratio
        parameters = canonical_env_parameters(self.species)
        # Fixed target, straight-line and stationary objectives conflict with
        # requested turns/stops. Only pass parameters the species supports.
        for name in (
            "forward_vel_weight",
            "heading_weight",
            "lateral_penalty_weight",
            "backward_vel_penalty_weight",
            "drift_penalty_weight",
            "spin_penalty_weight",
            "speed_penalty_weight",
            "idle_penalty_weight",
            "strike_bonus",
            "strike_approach_weight",
            "strike_proximity_weight",
            "strike_claw_proximity_weight",
            "food_reach_bonus",
            "food_approach_weight",
            "food_head_proximity_weight",
            "snap_bonus",
            "snap_approach_weight",
            "snap_snout_proximity_weight",
            "target_reach_bonus",
            "target_approach_weight",
        ):
            if name in parameters:
                env_kwargs[name] = 0.0
        env_kwargs.setdefault("max_episode_steps", 2500)
        super().__init__(**env_kwargs)
        self.parent_plant_identity = current_plant_identity(self.species)
        validate_compiled_plant(self.model, self.parent_plant_identity)
        self._home_ground_clearance_m = self.home_ground_clearance()
        self._spawn_model = copy.copy(self.model)
        self._spawn_data = mujoco.MjData(self._spawn_model)
        self._plane_model, self._plane_data = self.model, self.data
        self._terrain_model: mujoco.MjModel | None = None
        self._terrain_data: mujoco.MjData | None = None
        path = REPOSITORY_ROOT / self.parent_plant_identity.model_path
        if terrain is not None:
            initial = generate_terrain(terrain, run_seed=self.run_seed, episode_index=0)
            self._terrain_model = build_terrain_model(path, initial)
            self._assert_same_animal(self._plane_model, self._terrain_model)
            self._assert_matching_ids(self._plane_model, self._terrain_model)
            self._terrain_data = mujoco.MjData(self._terrain_model)
        # Preserve target observation channels, while making target props
        # unable to obstruct a route. Includes descendants of mocap props.
        prop_bodies: set[int] = set()
        for body in range(1, self.model.nbody):
            if self.model.body_mocapid[body] >= 0 or int(self.model.body_parentid[body]) in prop_bodies:
                prop_bodies.add(body)
        prop_geoms = np.isin(self.model.geom_bodyid, list(prop_bodies))
        for model in (self._plane_model, self._terrain_model):
            if model is not None:
                model.geom_contype[prop_geoms] = 0
                model.geom_conaffinity[prop_geoms] = 0
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self._behavior_root_id = int(self.model.jnt_bodyid[free[0]])
        self._substep_probe_hook = self._probe_ground_clearance
        parameters.update(env_kwargs)
        parameters.pop("render_mode", None)
        self._behavior_parameters = parameters

    _assert_matching_ids = staticmethod(TRexBehaviorEnv._assert_matching_ids)
    _assert_same_animal = staticmethod(TRexBehaviorEnv._assert_same_animal)

    def _select_contact_model(self, *, use_terrain: bool) -> None:
        if use_terrain:
            assert self._terrain_model is not None and self._terrain_data is not None
            model, data = self._terrain_model, self._terrain_data
        else:
            model, data = self._plane_model, self._plane_data
        if self.model is not model:
            self.close()
            self._camera = None
        self.model, self.data = model, data
        self._cache_ids()
        self._root_subtree_geom_ids = None
        self._static_floor_geom_ids = None
        self._ground_geom_array = None
        self._invalidate_substep_aggregates()
        self._action_filter_state = None

    @property
    def behavior_identity(self) -> dict[str, Any]:
        source_paths = (
            Path(__file__),
            Path(inspect.getfile(self._canonical_env_class)),
            Path(inspect.getfile(TRexBehaviorEnv)),
            Path(inspect.getfile(DirectionCommandController)),
            Path(inspect.getfile(TerrainConfig)),
        )
        identity = {
            "schema": "mesozoic.command-terrain/v1",
            "species": self.species,
            "backend": "stable-baselines3",
            "parent_plant": self.parent_plant_identity.to_dict(),
            "commands": asdict(self.direction_controller.config),
            "terrain": asdict(self.terrain_config) if self.terrain_config is not None else None,
            "env": self._behavior_parameters,
            "tracking_weight": self.tracking_weight,
            "tracking_tolerances": self.tracking_tolerances,
            "tracking_velocity_sigma": self.tracking_velocity_sigma,
            "course_distance": self.course_distance,
            "flat_probability": self.flat_probability,
            "prey_collision": False,
            "sources": {
                str(p.relative_to(REPOSITORY_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
            },
        }
        return dict(json.loads(json.dumps(identity)))

    def _heading(self) -> float:
        rotation = self.data.xmat[self._behavior_root_id].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _current_clearances(self) -> np.ndarray:
        points = [self.data.xpos[self._behavior_root_id]]
        for kind, index in self._substep_height_checks:
            points.append(self.data.site_xpos[index] if kind == "site" else self.data.xpos[index])
        return np.array([self._clearance(point) for point in points])

    def _probe_ground_clearance(self) -> None:
        clearances = self._current_clearances()
        self._clearance_min = clearances if self._clearance_min is None else np.minimum(self._clearance_min, clearances)
        self._clearance_max = clearances if self._clearance_max is None else np.maximum(self._clearance_max, clearances)

    def _check_height_tilt_termination(self, body_z: float, tilt_angle: float) -> tuple[bool, str | None]:
        clearances = self._current_clearances()
        minimum = clearances[0] if self._clearance_min is None else self._clearance_min[0]
        maximum = clearances[0] if self._clearance_max is None else self._clearance_max[0]
        terminated, reason = super()._check_height_tilt_termination(float(minimum), tilt_angle)
        if not terminated:
            terminated, reason = super()._check_height_tilt_termination(float(maximum), tilt_angle)
        return terminated, reason

    def _aggregated_min_height(self, check_index: int, instantaneous: float) -> float:
        clearances = self._current_clearances() if self._clearance_min is None else self._clearance_min
        return float(clearances[check_index + 1])

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        # Keep each canonical species' non-finite, tilt, nosedive, snout and
        # categorized contact checks. The two helpers above only change the
        # height reference from world zero to the measured terrain surface.
        terminated, info = self._canonical_env_class._is_terminated(self)
        info["pelvis_clearance"] = self._clearance(self.data.xpos[self._behavior_root_id])
        if not terminated and self.terrain is not None:
            geoms = self._root_subtree_geoms()
            bounds = np.abs(self.data.geom_xpos[geoms, :2]) + self.model.geom_rbound[geoms, None]
            if np.any(bounds >= self.terrain.config.extent):
                terminated = True
                info["termination_reason"] = "terrain_boundary"
        return terminated, info

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        reward, info = self._canonical_env_class._get_reward_info(self, action)
        clearance = self._clearance(self.data.xpos[self._behavior_root_id])
        info["pelvis_clearance"] = clearance
        if "reward_height" in info:
            if self.species.startswith("compsognathus"):
                target = self.target_standing_z
                error = (clearance - target) / target
                contacts = self._aggregated_foot_contact_forces()
                support = float(sum(contacts) > self._contact_threshold)
                height = self.height_weight * float(np.exp(-np.square(error / 0.15))) * support
            else:
                target = 1.2 if self.species == "brachiosaurus" else 0.3129
                minimum = self.healthy_z_range[0]
                height = self.height_weight * float(np.clip((clearance - minimum) / (target - minimum), 0, 1))
            reward += height - info["reward_height"]
            info["reward_height"] = height
        if self._command_state is not None:
            yaw_rate = float(wrap_angle(self._heading() - self._heading_before) / self.dt)
            metrics = tracking_metrics(
                self._command_state,
                self.data.qvel[:2],
                yaw_rate,
                current_heading=self._heading(),
                **self.tracking_tolerances,
            )
            self._command_metrics = {**self._command_state.as_info(), **metrics}
            self._command_metrics["heading_error_rad"] = (
                float(wrap_angle(self._command_state.desired_heading - self._heading()))
                if self._command_state.heading_active
                else 0.0
            )
            tracking = gaussian_tracking_reward(
                metrics["tracking_error_v"],
                metrics["tracking_error_yaw"],
                velocity_sigma=self.tracking_velocity_sigma,
            )
            reward += self.tracking_weight * tracking
            info["reward_tracking"] = self.tracking_weight * tracking
            self._tracking_dwell_s = self._tracking_dwell_s + self.dt if metrics["tracking_in_tolerance"] else 0.0
        info["reward_total"] = reward
        return reward, info

    def command_manifest(self) -> dict[str, Any]:
        return self.direction_controller.manifest()

    def _clearance(self, xyz: np.ndarray) -> float:
        ground = 0.0 if self.terrain is None else self.terrain.height_at(float(xyz[0]), float(xyz[1]))
        if not np.isfinite(ground):
            # The boundary check terminates this state; avoid propagating NaN
            # into reward/observation before it runs.
            ground = 0.0
        return float(xyz[2] - ground)

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
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            if self._viewer is not None:
                self._viewer.update_hfield(0)
        self._clearance_min = None
        self._clearance_max = None
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
        self._clearance_max = None
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
            self.data.xpos[self._behavior_root_id, 2] - self._clearance(self.data.xpos[self._behavior_root_id])
        )
        # Keep the base is_success=False: radial displacement alone cannot
        # certify command following or a terrain route.
        previous_event = self._command_state.event_id
        self._command_state = self.direction_controller.update(self._step_count * self.dt, self._heading())
        self._command = self._command_state.normalized.copy()
        if previous_event != self._command_state.event_id:
            self._tracking_dwell_s = 0.0
        return self._get_obs(), reward, terminated, truncated, info


@lru_cache(maxsize=None)
def get_behavior_env_class(species: str) -> type[Any]:
    """Return a behavior subclass of the selected canonical species environment."""
    species = resolve_species_id(species)
    if species == "trex":
        return TRexBehaviorEnv
    canonical = get_species_config(species).env_class
    return type(
        f"{canonical.__name__.removesuffix('Env')}BehaviorEnv",
        (SpeciesBehaviorMixin, canonical),
        {"species": species, "_canonical_env_class": canonical, "__module__": __name__},
    )
