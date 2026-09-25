"""Direction and terrain behavior environments for every registered species.

The canonical animal, observation ordering and action mapping remain unchanged.
Task rewards and safety clearances use the actual surface under each body/site:
the species code reads heights through ``BaseDinoEnv._clearance`` and this env
overrides only ``_ground_height_at``.
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
        self._probe_hit_geom: int | None = None
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
            "bite_bonus",
            "bite_approach_weight",
            "bite_head_proximity_weight",
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
        initial = None
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
        if self._terrain_contact_probe_geoms:
            # A separate collision-only model detects probe-geom/surface
            # intersections without letting the non-colliding geom support the
            # trained animal; mj_geomDistance does not return surface distance
            # for hfields. Masks are set on the spec before compiling: flipping
            # them on a compiled copy misses bodies that had no colliding geoms
            # in the canonical broad-phase caches.
            probe_spec = mujoco.MjSpec.from_file(str(path))
            for name in self._terrain_contact_probe_geoms:
                probe_spec.geom(name).contype = 1
                probe_spec.geom(name).conaffinity = 1
            self._plane_probe_model = probe_spec.compile()
            self._assert_matching_ids(self._plane_model, self._plane_probe_model)
            self._plane_probe_data = mujoco.MjData(self._plane_probe_model)
            self._terrain_probe_model: mujoco.MjModel | None = None
            self._terrain_probe_data: mujoco.MjData | None = None
            if initial is not None:
                self._terrain_probe_model = build_terrain_model(probe_spec, initial)
                self._assert_matching_ids(self._plane_model, self._terrain_probe_model)
                self._terrain_probe_data = mujoco.MjData(self._terrain_probe_model)
            self._probe_model, self._probe_data = self._plane_probe_model, self._plane_probe_data
            self._probe_geom_ids = np.array(
                [
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                    for name in self._terrain_contact_probe_geoms
                ]
            )
            self._substep_probe_hook = self._probe_terrain_contacts
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self._behavior_root_id = int(self.model.jnt_bodyid[free[0]])
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

    def _select_contact_model(self, *, use_terrain: bool) -> None:
        """Select a precompiled scene and discard episode-dependent caches."""
        if use_terrain:
            assert self._terrain_model is not None and self._terrain_data is not None
            model, data = self._terrain_model, self._terrain_data
        else:
            model, data = self._plane_model, self._plane_data
        if self.model is not model:
            # Renderer/viewer instances retain model pointers. Recreate them
            # lazily on render, never retarget a live instance or geom type.
            self.close()
            self._camera = None
        self.model, self.data = model, data
        self._cache_ids()
        self._root_subtree_geom_ids = None
        self._static_floor_geom_ids = None
        self._ground_geom_array = None
        self._invalidate_substep_aggregates()
        self._action_filter_state = None
        if self._terrain_contact_probe_geoms:
            if use_terrain:
                assert self._terrain_probe_model is not None and self._terrain_probe_data is not None
                self._probe_model, self._probe_data = self._terrain_probe_model, self._terrain_probe_data
            else:
                self._probe_model, self._probe_data = self._plane_probe_model, self._plane_probe_data
            mujoco.mj_resetData(self._probe_model, self._probe_data)

    @property
    def behavior_identity(self) -> dict[str, Any]:
        source_paths = (
            Path(__file__),
            Path(inspect.getfile(self._canonical_env_class)),
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

    def _probe_terrain_contacts(self) -> None:
        """Substep hook: latch the first probe geom that penetrates the floor."""
        self._probe_data.qpos[:] = self.data.qpos
        self._probe_data.mocap_pos[:] = self.data.mocap_pos
        self._probe_data.mocap_quat[:] = self.data.mocap_quat
        mujoco.mj_kinematics(self._probe_model, self._probe_data)
        mujoco.mj_collision(self._probe_model, self._probe_data)
        pairs = self._probe_data.contact.geom
        if self._probe_hit_geom is None and len(pairs):
            g1, g2 = pairs[:, 0], pairs[:, 1]
            hits = ((g2 == self.floor_geom_id) & np.isin(g1, self._probe_geom_ids)) | (
                (g1 == self.floor_geom_id) & np.isin(g2, self._probe_geom_ids)
            )
            hit_indices = np.flatnonzero(hits & (self._probe_data.contact.dist < 0.0))
            if hit_indices.size:
                first = int(hit_indices[0])
                self._probe_hit_geom = int(g1[first] if g2[first] == self.floor_geom_id else g2[first])

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        # Keep each canonical species' non-finite, height/tilt, nosedive,
        # snout and categorized contact checks; their heights are clearances
        # above the surface through _ground_height_at below.
        terminated, info = self._canonical_env_class._is_terminated(self)
        info["pelvis_clearance"] = self._clearance(self.data.xpos[self._behavior_root_id])
        if not terminated and self._probe_hit_geom is not None:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self._probe_hit_geom)
            terminated = True
            info["termination_reason"] = f"{name.removesuffix('_geom')}_ground_contact"
        if not terminated and self.terrain is not None:
            geoms = self._root_subtree_geoms()
            bounds = np.abs(self.data.geom_xpos[geoms, :2]) + self.model.geom_rbound[geoms, None]
            if np.any(bounds >= self.terrain.config.extent):
                terminated = True
                info["termination_reason"] = "terrain_boundary"
        return terminated, info

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        reward, info = self._canonical_env_class._get_reward_info(self, action)
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

    def _ground_height_at(self, xy: np.ndarray) -> float:
        if self.terrain is None:
            return 0.0
        ground = float(self.terrain.height_at(float(xy[0]), float(xy[1])))
        # Off the map: the boundary check terminates this state; avoid
        # propagating NaN into reward/observation before it runs.
        return ground if np.isfinite(ground) else 0.0

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
            if self._terrain_contact_probe_geoms:
                apply_terrain(self._probe_model, self.terrain, self._probe_data)
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            if self._viewer is not None:
                self._viewer.update_hfield(0)
        self._probe_hit_geom = None
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
        self._probe_hit_geom = None
        _, reward, terminated, truncated, info = super().step(action)
        info.update(self._command_metrics)
        radius = float(np.linalg.norm(self.data.qpos[:2] - self._initial_pos_2d))
        self._max_radius = max(self._max_radius, radius)
        # This is a traversal diagnostic, not an automatic training gate. A
        # complete route evaluator additionally checks heading/speed and falls.
        info["course_progress_m"] = radius
        info["course_reached"] = bool(self._max_radius >= self.course_distance)
        info["tracking_dwell_s"] = self._tracking_dwell_s
        info["terrain_height_m"] = self._ground_height_at(self.data.xpos[self._behavior_root_id, :2])
        # Keep the base is_success=False: radial displacement alone cannot
        # certify command following or a terrain route.
        previous_event = self._command_state.event_id
        self._command_state = self.direction_controller.update(self._step_count * self.dt, self._heading())
        self._command = self._command_state.normalized.copy()
        if previous_event != self._command_state.event_id:
            self._tracking_dwell_s = 0.0
        return self._get_obs(), reward, terminated, truncated, info


@lru_cache(maxsize=None)
def _behavior_env_class(species: str) -> type[Any]:
    canonical = get_species_config(species).env_class
    return type(
        f"{canonical.__name__.removesuffix('Env')}BehaviorEnv",
        (SpeciesBehaviorMixin, canonical),
        {"species": species, "_canonical_env_class": canonical, "__module__": __name__},
    )


def get_behavior_env_class(species: str) -> type[Any]:
    """Return a behavior subclass of the selected canonical species environment."""
    return _behavior_env_class(resolve_species_id(species))
