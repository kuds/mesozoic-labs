"""Direction and terrain behavior environments for every registered model.

The canonical animal, observation ordering and action mapping remain unchanged.
Task rewards and safety clearances use the actual surface under each body/site:
the species code reads heights through ``BaseDinoEnv._clearance`` and this env
overrides only ``_ground_height_at``.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import asdict, replace
from functools import lru_cache
from typing import Any, cast

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.command_frame import COMMAND_STREAM_DOMAIN
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import REPOSITORY_ROOT, current_plant_identity, validate_compiled_plant
from environments.shared.species_registry import get_species_config
from environments.shared.task_fingerprint import FINGERPRINT_BACKEND, compute_task_fingerprint
from environments.shared.terrain import (
    TerrainConfig,
    TerrainRealization,
    apply_terrain,
    build_terrain_model,
    generate_terrain,
)
from environments.shared.terrain_sampling import (
    TerrainSamplerConfig,
    TerrainTemplate,
    select_terrain_family,
)

#: The task fingerprint's ``stage`` for every recipe env.  Versioned like
#: ``task_fingerprint.SCHEDULE_IMPLEMENTATION``, in place of the source hashes
#: the retired ``behavior_identity`` carried: bump it when this module,
#: terrain.py or terrain_sampling.py changes what a recipe's task means, not
#: merely its shape (direction_commands.py is versioned by its manifest).  It
#: is never a stage id (``STAGE_ID_PATTERN``), so no manifest node takes a
#: recipe checkpoint as its parent (decision D-D9).
RECIPE_TASK_STAGE = "command-terrain/v2"


class SpeciesBehaviorMixin(BaseDinoEnv):
    """Compose command tracking with each species' canonical balance behavior.

    The tracking reward, metrics, info keys and dwell are ``BaseDinoEnv``'s,
    live under the ``heading_and_speed`` mode this env always runs (decision
    D-D60).  The env adds terrain, the course diagnostics and three things that
    keep every recipe's task as it was: its own command stream (run seed,
    episode seed and episode index, reported in the reset info), its
    ``tracking_weight`` kwarg, recorded in the task's env section, and the
    reward rules before the live walk (decision D-D62,
    :meth:`_rewards_follow_command` is False): its terms read the spawn's
    heading and position, never the command's, its idle term is computed at
    a zero command too and the Allosaurus' course is the "none" one.  A
    recipe zeroes every forward, heading, lateral, drift and idle weight, so
    the live rules would move no recipe reward, only diagnostics such as
    ``forward_vel`` and ``drift_distance`` (and the Allosaurus' course
    events), which the pilot rollout golden pins.  A recipe's command config
    may declare no stand block, no cruise ramp and no tracking window
    (decision D-D64): the recipe pipeline runs none of them (the follow
    nodes that supersede it will), and the behavior certificate judges the
    per-step signal.
    """

    model_id: ModelId
    _canonical_env_class: type[BaseDinoEnv]
    floor_geom_id: int
    _initial_pos_2d: np.ndarray

    def __init__(
        self,
        *,
        commands: DirectionCommandConfig | None = None,
        terrain: TerrainConfig | None = None,
        terrain_sampler: TerrainSamplerConfig | None = None,
        run_seed: int = 42,
        tracking_weight: float = 2.5,
        course_distance: float = 10.0,
        **env_kwargs: Any,
    ):
        if not isinstance(run_seed, (int, np.integer)) or isinstance(run_seed, bool) or not 0 <= run_seed <= 2**32 - 1:
            raise ValueError("run_seed must be an integer in [0, 2**32 - 1]")
        if not np.isfinite(tracking_weight) or tracking_weight <= 0:
            raise ValueError("tracking_weight must be finite and positive")
        if not np.isfinite(course_distance) or course_distance <= 0:
            raise ValueError("course_distance must be finite and positive")
        commands = commands or DirectionCommandConfig()
        unrecorded_window = DirectionCommandConfig.UNRECORDED_VALUES["tracking_window_s"]
        if commands.stand_weights or commands.has_cruise_ramp or commands.tracking_window_s != unrecorded_window:
            raise ValueError(
                "a behavior recipe's commands declare no stand_weights, no cruise ramp and no tracking window: the "
                "recipe pipeline runs none of them, and the behavior certificate reads the per-step signal "
                "(decisions D-D62, D-D64)"
            )
        unrecorded_weight = DirectionCommandConfig.UNRECORDED_VALUES["tracking_weight"]
        if commands.tracking_weight != unrecorded_weight:
            raise ValueError(
                f"commands.tracking_weight is {commands.tracking_weight!r}: a behavior env weights tracking by its "
                f"tracking_weight kwarg ([env] tracking_weight in a recipe), so leave the command config's at "
                f"{unrecorded_weight}"
            )
        self.terrain_config = terrain
        self.terrain_sampler = terrain_sampler
        if terrain_sampler is not None:
            if not isinstance(terrain_sampler, TerrainSamplerConfig):
                raise ValueError("terrain_sampler must be a TerrainSamplerConfig")
            if terrain is None:
                raise ValueError("terrain_sampler requires an enabled terrain configuration")
            # Build every enabled family's surface before any model is compiled: a valid slope
            # profile may otherwise have a grid too coarse for the requested bump radius.
            for family in terrain_sampler.families:
                self._family_terrain(family)
        self.run_seed = int(run_seed)
        self._episode_index = 0
        self._episode_seed = 0
        self._command_seed = 0
        self.course_distance = float(course_distance)
        self.terrain: TerrainRealization | None = None
        self._probe_hit_geom: int | None = None
        self._max_radius = 0.0
        parameters = inspect.signature(self._canonical_env_class).parameters
        # Straight-line and stationary objectives (along the spawn's reference
        # heading, or holding the spawn) conflict with requested turns/stops.
        # Only pass parameters the species supports.  The hunt and target
        # kwargs left with their code (decision D-D59).
        for name in (
            "forward_vel_weight",
            "heading_weight",
            "lateral_penalty_weight",
            "backward_vel_penalty_weight",
            "drift_penalty_weight",
            "spin_penalty_weight",
            "speed_penalty_weight",
            "idle_penalty_weight",
        ):
            if name in parameters:
                env_kwargs[name] = 0.0
        env_kwargs.setdefault("max_episode_steps", 2500)
        super().__init__(command_mode="heading_and_speed", command_config=commands, **env_kwargs)
        # The base read the command config's weight; this env's is its own kwarg (read live, like the
        # reward and the task fingerprint read it).  The sigmas and tolerances stay the config's.
        self.tracking_weight = float(tracking_weight)
        self.parent_plant_identity = current_plant_identity(self.model_id)
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
        # The task's constructor kwargs (run_seed is a seed, not task); task_fingerprint adds the
        # subclass's own four, read live as reset, the reward and the step read them.
        self._task_env_kwargs = {**env_kwargs, "command_mode": self.command_mode, "command_config": commands}

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
        # Every pool model back to nominal before the swap and _cache_ids,
        # which reads masses (the species' weight thresholds): the opt-in
        # physics randomisation (episode_physics.py) writes the next episode's
        # values only when it begins.  Nothing while it is off.
        self._restore_episode_physics()
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
    def task_fingerprint(self) -> dict[str, Any]:
        """This task's identity: plant, effective constructor kwargs (its own four read live) and command manifest."""
        return compute_task_fingerprint(
            model_id=self.model_id,
            stage=RECIPE_TASK_STAGE,
            backend=FINGERPRINT_BACKEND,
            env_kwargs={
                **self._task_env_kwargs,
                "terrain": self.terrain_config,
                "terrain_sampler": self.terrain_sampler,
                "tracking_weight": float(self.tracking_weight),
                "course_distance": float(self.course_distance),
            },
            plant_identity=self.parent_plant_identity.to_dict(),
            perturbation_manifest=self.perturbation_manifest(),
            command_manifest=self.command_manifest(),
        )

    @property
    def terrain_families(self) -> tuple[str, ...]:
        """The families this task's episodes visit: the sampler's, else its one surface (the plane without terrain)."""
        if self.terrain_sampler is not None:
            return self.terrain_sampler.families
        if self.terrain_config is None:
            return ("flat",)
        return ("terrain_contact" if self.terrain_config.mode == "flat" else self.terrain_config.template,)

    def _family_terrain(self, family: str) -> TerrainConfig | None:
        """A family's surface on the terrain's map: the original plane (None), zero heights, or a gentle template."""
        if family == "flat":
            return None
        if self.terrain_config is None:
            raise ValueError(f"terrain family {family!r} requires an enabled terrain configuration")
        if family == "terrain_contact":
            return replace(self.terrain_config, mode="flat")
        return replace(self.terrain_config, mode="gentle", template=cast(TerrainTemplate, family))

    def _probe_terrain_contacts(self) -> None:
        """Substep hook: latch the first probe geom that penetrates the floor."""
        self._probe_data.qpos[:] = self.data.qpos
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
        info["pelvis_clearance"] = self._clearance(self.data.xpos[self._root_body_id])
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

    def _rewards_follow_command(self) -> bool:
        """Always False: a recipe task keeps the reward rules before the live walk (see the class docstring)."""
        return False

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
        # Refused before the episode stream moves: a seed that is not a nonnegative integer, a family
        # override (options={"terrain_family": ...}, evaluation's round robin) outside this task, or a
        # family whose surface cannot be built (a sampler reassigned after construction).  First, a reset
        # inside a nominal_physics() block (episode_physics.py), before the episode counters move or a pool
        # model is selected; nothing while the randomisation is off.
        self._check_episode_may_begin()
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            seed = int(seed)
        options = dict(options) if options is not None else None
        forced_family = None if options is None else options.pop("terrain_family", None)
        if forced_family is not None and forced_family not in self.terrain_families:
            raise ValueError(f"terrain_family must be an enabled family: {', '.join(self.terrain_families)}")
        episode_seed = self._episode_seed if seed is None else seed
        episode_index = self._episode_index if seed is None else 0
        sampler, selection = self.terrain_sampler, None
        if sampler is not None:
            selection = select_terrain_family(
                sampler,
                run_seed=self.run_seed,
                episode_seed=episode_seed,
                episode_index=episode_index,
            )
        if forced_family is not None:
            family = forced_family
        else:
            family = selection.family if selection is not None else self.terrain_families[0]
        surface = self._family_terrain(family)
        self._episode_seed, self._episode_index = episode_seed, episode_index
        # Independent streams keep command draws from changing a terrain map
        # or the parent's pose-noise sequence.
        terrain_seed = int(np.random.SeedSequence([self.run_seed, self._episode_seed, 0x7E22]).generate_state(1)[0])
        command_seed = int(
            np.random.SeedSequence(
                [self.run_seed, self._episode_seed, self._episode_index, COMMAND_STREAM_DOMAIN]
            ).generate_state(1)[0]
        )
        self._select_contact_model(use_terrain=surface is not None)
        self.terrain = None
        if surface is not None:
            self.terrain = generate_terrain(surface, run_seed=terrain_seed, episode_index=self._episode_index)
            apply_terrain(self.model, self.terrain, self.data)
            if self._terrain_contact_probe_geoms:
                apply_terrain(self._probe_model, self.terrain, self._probe_data)
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            if self._viewer is not None:
                self._viewer.update_hfield(0)
        self._probe_hit_geom = None
        self._max_radius = 0.0
        self._command_seed = command_seed
        obs, info = super().reset(seed=seed, options=options)
        info.update(
            {
                "run_seed": self.run_seed,
                "episode_seed": self._episode_seed,
                "episode_index": self._episode_index,
                "command_seed": command_seed,
                "terrain": self.terrain.manifest() if self.terrain is not None else {"family": "flat_plane"},
            }
        )
        if sampler is not None and selection is not None:
            info["terrain_sampling"] = {
                **asdict(selection),
                "family": family,
                "mode": "balanced_shuffle" if forced_family is None else "evaluation_override",
                "weights": asdict(sampler),
            }
        self._episode_index += 1
        return obs, info

    def _command_rng(self) -> np.random.Generator:
        # The recipe's own stream (run seed, episode seed, episode index; see reset), which the base's
        # reset-seed stream replaces for manifest stages: either way commands never shift the parent's reset
        # draws or a terrain map, and this one keeps every recipe's command schedules as they were.
        return np.random.default_rng(self._command_seed)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._probe_hit_geom = None
        obs, reward, terminated, truncated, info = super().step(action)
        radius = float(np.linalg.norm(self.data.qpos[:2] - self._initial_pos_2d))
        self._max_radius = max(self._max_radius, radius)
        # This is a traversal diagnostic, not an automatic training gate. A
        # complete route evaluator additionally checks heading/speed and falls.
        # Keep the base is_success=False: radial displacement alone cannot
        # certify command following or a terrain route.
        info["course_progress_m"] = radius
        info["course_reached"] = bool(self._max_radius >= self.course_distance)
        # The base step reports the dwell; a recipe's info has always carried it after the course diagnostics.
        info["tracking_dwell_s"] = info.pop("tracking_dwell_s")
        info["terrain_height_m"] = self._ground_height_at(self.data.xpos[self._root_body_id, :2])
        return obs, reward, terminated, truncated, info


@lru_cache(maxsize=None)
def _behavior_env_class(model_id: ModelId) -> type[Any]:
    canonical = get_species_config(model_id).env_class
    return type(
        f"{model_id.pascal}BehaviorEnv",
        (SpeciesBehaviorMixin, canonical),
        {"model_id": model_id, "_canonical_env_class": canonical, "__module__": __name__},
    )


def get_behavior_env_class(model_id: ModelId) -> type[Any]:
    """Return a behavior subclass of the model's canonical environment (``<Pascal s><Pascal e>BehaviorEnv``)."""
    return _behavior_env_class(model_id)
