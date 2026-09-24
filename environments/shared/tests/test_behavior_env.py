"""Real-model integration checks for the shared species behavior environments."""

from __future__ import annotations

import copy
import inspect
from functools import cache
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environments.shared import behavior_env as behavior_module
from environments.shared.behavior_env import SpeciesBehaviorMixin, canonical_env_parameters, get_behavior_env_class
from environments.shared.direction_commands import DirectionCommandConfig, gaussian_tracking_reward
from environments.shared.plant_contract import REPOSITORY_ROOT, current_plant_identity, validate_compiled_plant
from environments.shared.species_names import species_display_name
from environments.shared.species_registry import get_species_config
from environments.shared.terrain import TerrainConfig, TerrainRealization, apply_terrain

SPECIES = ("trex", "velociraptor", "brachiosaurus", "dibothrosuchus", "compsognathus", "compsognathus_robot")
SPEEDS = {
    "velociraptor": 2.0,
    "brachiosaurus": 0.75,
    "dibothrosuchus": 0.9,
    "compsognathus": 0.08,
    "compsognathus_robot": 0.04,
    "trex": 1.05,
}
RECIPES = Path(__file__).resolve().parents[3] / "configs"
# Height-derived reward/info keys; every species reports at least pelvis_height.
HEIGHT_KEYS = (
    "reward_height",
    "height_error",
    "height_quality",
    "pelvis_height",
    "torso_height",
    "reward_head_clearance",
    "head_clearance_quality",
    "head_tip_z",
)


@pytest.fixture(scope="module", autouse=True)
def _one_plant_identity_per_species():
    """Fingerprint each species' plant once per module instead of once per env (seconds per construction)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(behavior_module, "current_plant_identity", cache(current_plant_identity))
        yield


def commands(species, **kwargs):
    speed = SPEEDS[species]
    return DirectionCommandConfig(cruise_speed=speed, speed_scale=speed * 1.5, **kwargs)


@pytest.fixture
def make_env():
    instances = []

    def factory(species, **kwargs):
        kwargs.setdefault("commands", commands(species))
        env = get_behavior_env_class(species)(**kwargs)
        instances.append(env)
        return env

    yield factory
    for env in instances:
        env.close()


def terrain(**kwargs):
    return TerrainConfig(extent=8.0, nrow=81, ncol=81, apron_radius=4.0, **kwargs)


def zeros(env):
    return np.zeros(env.action_space.shape)


def _translate_surface_and_animal(env, height):
    realization = TerrainRealization(
        env.terrain_config, 999, 0, np.full((env.terrain_config.nrow, env.terrain_config.ncol), height)
    )
    env.terrain = realization
    apply_terrain(env.model, realization, env.data)
    if env._terrain_contact_probe_geoms:
        apply_terrain(env._probe_model, realization, env._probe_data)
    env.data.qpos[2] += realization.height_at(0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize("species", SPECIES)
def test_factory_preserves_canonical_inheritance(species):
    behavior = get_behavior_env_class(species)
    canonical = get_species_config(species).env_class
    assert issubclass(behavior, canonical) and issubclass(behavior, SpeciesBehaviorMixin)
    assert behavior.__mro__[1:3] == (SpeciesBehaviorMixin, canonical)
    assert "forward_vel_weight" in canonical_env_parameters(species)
    assert behavior is get_behavior_env_class(species) is get_behavior_env_class(species_display_name(species))
    # The species declaration must not be hidden by a mixin default earlier in the MRO.
    assert behavior._terrain_contact_probe_geoms == (("neck_geom",) if species == "trex" else ())


@pytest.mark.parametrize("species", SPECIES)
def test_species_has_real_surface_command_inputs_and_reproducible_rollout(make_env, species):
    env = make_env(
        species,
        commands=commands(species, switch_interval_s=0.02, straight_probability=0.0),
        terrain=terrain(template="mixed"),
        run_seed=17,
        reset_noise_scale=0.01,
    )
    canonical = get_species_config(species).env_class(reset_noise_scale=0.01)
    actions = np.random.default_rng(912).uniform(-0.01, 0.01, (8, *env.action_space.shape))

    def replay(seed):
        observation, info = env.reset(seed=seed)
        initial = (observation.copy(), env.data.qpos.copy(), env.data.qvel.copy(), env.model.hfield_data.copy())
        rows = []
        for action in actions:
            obs, reward, terminated, truncated, step_info = env.step(action)
            rows.append(
                (obs.copy(), reward, terminated, truncated, step_info["command_event_id"], step_info["desired_heading"])
            )
        return initial, info, rows

    try:
        (observation, qpos, qvel, surface), info, first = replay(1729)
        canonical.reset(seed=1729)
        assert observation.shape == canonical.observation_space.shape
        assert env.action_space == canonical.action_space
        np.testing.assert_allclose(qpos, canonical.data.qpos, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(qvel, canonical.data.qvel)
        assert observation[-3] > 0
        assert env.behavior_identity["parent_plant"]["species"] == species
        assert env.model.nhfield == 1
        assert info["terrain"]["template"] == "mixed"
        props = np.flatnonzero(env.model.body_mocapid >= 0)
        prop_geoms = np.isin(env.model.geom_bodyid, props)
        assert not np.any(env.model.geom_contype[prop_geoms])
        assert not np.any(env.model.geom_conaffinity[prop_geoms])
        assert np.isfinite(first[0][1]) and not first[0][2]
        assert first[-1][4] > 0  # the switch schedule fired during the replay
        second_initial, second_info, second = replay(1729)
        assert second_info == info
        for a, b in zip((observation, qpos, qvel, surface), second_initial, strict=True):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:] == b[1:]
        env.reset(seed=1730)
        assert not np.array_equal(env.data.qpos, qpos)
        assert not np.array_equal(env.model.hfield_data, surface)
    finally:
        canonical.close()


@pytest.mark.parametrize("species", SPECIES)
def test_plane_and_terrain_free_behavior_env_score_identically(make_env, species):
    """Consolidation PR-7's mitigation: on the plane the ground-height hook is ``z - 0.0``.

    The canonical env built with the behavior env's own parameters (zeroed
    objectives, 2,500-step horizon) sees the same states, reward terms,
    terminations and reasons; the behavior env only adds command tracking.
    """
    env = make_env(species, terrain=None, reset_noise_scale=0.05)
    plane = get_species_config(species).env_class(**env._behavior_parameters)
    try:
        terminations = 0
        for seed in range(4):
            env.reset(seed=seed)
            plane.reset(seed=seed)
            rng = np.random.default_rng(seed)
            scale = 1.0 if seed % 2 else 0.3
            for _ in range(150):
                action = rng.uniform(-scale, scale, env.action_space.shape)
                _, reward, terminated, truncated, info = env.step(action)
                _, plane_reward, plane_terminated, plane_truncated, plane_info = plane.step(action)
                np.testing.assert_array_equal(env.data.qpos, plane.data.qpos)
                for key, value in plane_info.items():
                    if key.startswith("reward_") and key != "reward_total":
                        assert info[key] == value, key
                assert reward - info["reward_tracking"] == pytest.approx(plane_reward, rel=0, abs=1e-9)
                assert (terminated, truncated) == (plane_terminated, plane_truncated)
                assert info.get("termination_reason") == plane_info.get("termination_reason")
                if terminated or truncated:
                    terminations += terminated
                    break
        assert terminations  # the full-scale episodes exercise the termination chain
    finally:
        plane.close()


@pytest.mark.parametrize("height", [0.8, -1.5])
@pytest.mark.parametrize("species", SPECIES)
def test_rewards_and_terminations_follow_a_translated_surface(make_env, species, height):
    """A raised or lowered surface leaves every height term and termination where it was.

    Lowering by 1.5 m puts trex's head tip and skull (and the dibothrosuchus
    snout) below their world-z thresholds, so a species check that went back
    to world z would terminate the standing animal.
    """
    extra = (
        dict(height_weight=0.3, height_target_tolerance=0.1, head_clearance_weight=0.35, head_clearance_target=1.0)
        if species == "trex"
        else {}
    )
    env = make_env(species, terrain=terrain(mode="flat"), reset_noise_scale=0.0, **extra)
    env.reset(seed=42)
    action = zeros(env)
    reward, original = env._get_reward_info(action)
    root = env._behavior_root_id
    original_clearance = env._clearance(env.data.xpos[root])
    _translate_surface_and_animal(env, height)
    assert env._clearance(env.data.xpos[root]) == pytest.approx(original_clearance, abs=1e-12)
    _, translated = env._get_reward_info(action)
    compared = [key for key in HEIGHT_KEYS if key in original]
    assert "pelvis_height" in compared
    for key in compared:
        assert translated[key] == pytest.approx(original[key], abs=1e-12), key
    assert np.isfinite(reward)
    assert not env._is_terminated()[0]
    env.data.qpos[2] -= original_clearance - (env.healthy_z_range[0] - 0.01)
    mujoco.mj_forward(env.model, env.data)
    if height > 0:
        assert env.data.xpos[root, 2] > env.healthy_z_range[0]  # World z would not terminate.
    terminated, info = env._is_terminated()
    assert terminated
    assert info["pelvis_clearance"] == pytest.approx(env.healthy_z_range[0] - 0.01, abs=1e-9)
    assert info["termination_reason"] == "fallen"


@pytest.mark.parametrize("species", ["trex", "dibothrosuchus"])
def test_substep_height_minima_are_surface_relative(make_env, species):
    """BaseDinoEnv.step aggregates head/snout clearance, not world z, on a raised surface."""
    level, raised = (make_env(species, terrain=terrain(mode="flat"), reset_noise_scale=0.0) for _ in range(2))
    level.reset(seed=42)
    raised.reset(seed=42)
    _translate_surface_and_animal(raised, 0.8)
    for _ in range(3):
        level_step = level.step(zeros(level))
        raised_step = raised.step(zeros(raised))
        # Contact solves on the two surfaces differ in the last bits; world z would differ by 0.8 m.
        np.testing.assert_allclose(raised._substep_min_heights, level._substep_min_heights, rtol=0, atol=1e-4)
        assert not raised_step[2] and not level_step[2]
    kind, entity = raised._substep_height_checks[0]
    world_z = (raised.data.site_xpos if kind == "site" else raised.data.xpos)[entity, 2]
    assert raised._substep_min_heights[0] < world_z - 0.7
    raised._substep_min_heights[0] = 0.03  # a dip below both species' head-contact thresholds
    terminated, info = raised._is_terminated()
    assert terminated and info["termination_reason"] == "head_contact"


@pytest.mark.parametrize("species", SPECIES)
def test_small_species_cannot_pass_a_walk_command_while_stationary(make_env, species):
    env = make_env(species, reset_noise_scale=0.0)
    env.reset(seed=42)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env._get_reward_info(zeros(env))
    assert not env._command_metrics["tracking_in_tolerance"]
    assert env._command_metrics["tracking_error_requested_speed"] == pytest.approx(SPEEDS[species])
    assert env.behavior_identity["tracking_tolerances"] == env.tracking_tolerances
    assert env.tracking_velocity_sigma == pytest.approx(env.direction_controller.config.speed_scale / 6)


@pytest.mark.parametrize("on_terrain", [False, True])
def test_neck_probe_detects_penetration_without_adding_physical_support(make_env, on_terrain):
    env = make_env("trex", terrain=terrain() if on_terrain else None, reset_noise_scale=0.0)
    env.reset(seed=1)
    (neck,) = env._probe_geom_ids
    assert neck == mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "neck_geom")
    assert env._substep_probe_hook == env._probe_terrain_contacts
    assert env.model.geom_contype[neck] == 0
    assert env.model.geom_conaffinity[neck] == 0
    env._probe_terrain_contacts()
    assert env._probe_hit_geom is None
    standing = env.data.qpos.copy()
    rotation = env.data.geom_xmat[neck].reshape(3, 3)
    lowest_z = env.data.geom_xpos[neck, 2] - abs(rotation[2, 2]) * env.model.geom_size[neck, 1]
    lowest_z -= env.model.geom_size[neck, 0]
    env.data.qpos[2] -= lowest_z + 0.03
    mujoco.mj_forward(env.model, env.data)
    env._probe_terrain_contacts()
    assert env._probe_hit_geom == neck
    assert not np.any(env.data.contact.geom == neck)
    assert env.model.geom_contype[neck] == 0
    assert env.model.geom_conaffinity[neck] == 0
    assert env._probe_model.geom_contype[neck] == 1
    # The latch outlives a recovery within the step and ends the episode once
    # the canonical checks pass.
    env.data.qpos[:] = standing
    mujoco.mj_forward(env.model, env.data)
    terminated, info = env._is_terminated()
    assert terminated and info["termination_reason"] == "neck_ground_contact"


@pytest.mark.parametrize("species", SPECIES)
def test_different_run_seeds_change_surface_without_perturbing_pose_stream(make_env, species):
    left = make_env(species, terrain=terrain(), run_seed=42, reset_noise_scale=0.05)
    right = make_env(species, terrain=terrain(), run_seed=44, reset_noise_scale=0.05)
    left.reset(seed=2042)
    right.reset(seed=2042)
    np.testing.assert_array_equal(left.data.qpos, right.data.qpos)
    np.testing.assert_array_equal(left.data.qvel, right.data.qvel)
    assert not np.array_equal(left.model.hfield_data, right.model.hfield_data)
    first_surface = left.model.hfield_data.copy()
    _, info = left.reset()
    assert info["episode_index"] == 1
    assert not np.array_equal(first_surface, left.model.hfield_data)


@pytest.mark.parametrize("species", SPECIES)
def test_hfield_spawn_uses_canonical_pose_and_does_not_inject_motion(make_env, species):
    env = make_env(species, terrain=terrain(), reset_noise_scale=0.05)
    canonical = get_species_config(species).env_class(reset_noise_scale=0.05)
    try:
        for seed in (42, 43, 44, 1042):
            env.reset(seed=seed)
            canonical.reset(seed=seed)
            np.testing.assert_allclose(env.data.qpos, canonical.data.qpos, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(env.data.qvel, canonical.data.qvel)
            assert env.model.nhfield == 1
            assert not np.any(env.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            geoms = env._root_subtree_geoms()
            floor_heights = env.terrain.height_at(env.data.geom_xpos[geoms, 0], env.data.geom_xpos[geoms, 1])
            np.testing.assert_array_equal(floor_heights, 0.0)
            contacts = env.data.contact
            floor_hits = np.any(contacts.geom == env.floor_geom_id, axis=1)
            assert np.all(contacts.dist[floor_hits] >= env.home_ground_clearance() - 0.002)
            assert not env._is_terminated()[0]
            for _ in range(10):
                _, reward, terminated, _, _ = env.step(zeros(env))
                assert np.isfinite(reward)
                assert np.all(np.isfinite(env.data.qpos))
                assert not terminated
        for field in (
            "timestep",
            "gravity",
            "solver",
            "integrator",
            "iterations",
            "ls_iterations",
            "tolerance",
            "disableflags",
            "enableflags",
            "cone",
            "impratio",
        ):
            np.testing.assert_array_equal(getattr(env.model.opt, field), getattr(canonical.model.opt, field))
    finally:
        canonical.close()


@pytest.mark.parametrize("species", SPECIES)
def test_behavior_pools_leave_an_independent_canonical_environment_unchanged(make_env, species):
    canonical = get_species_config(species).env_class(reset_noise_scale=0.05)
    identity = behavior_module.current_plant_identity(species)
    try:
        first, _ = canonical.reset(seed=1042)
        qpos, qvel = canonical.data.qpos.copy(), canonical.data.qvel.copy()
        arrays = {
            name: getattr(canonical.model, name).copy()
            for name in (
                "body_mass",
                "body_inertia",
                "geom_type",
                "geom_contype",
                "geom_conaffinity",
                "actuator_gainprm",
            )
        }
        env = make_env(species, terrain=terrain(), flat_probability=0.5, reset_noise_scale=0.05)
        assert (env._substep_probe_hook is not None) == hasattr(env, "_probe_model") == (species == "trex")
        seeds = _mixed_surface_seeds(env)
        props = np.flatnonzero(env.model.body_mocapid >= 0)
        for is_plane in (True, False, True):
            env.reset(seed=seeds[is_plane])
            assert env.model is not canonical.model and env.data is not canonical.data
            prop_geoms = np.isin(env.model.geom_bodyid, props)
            assert not np.any(env.model.geom_contype[prop_geoms])
            assert not np.any(env.model.geom_conaffinity[prop_geoms])
            for _ in range(3):
                env.step(zeros(env))
            validate_compiled_plant(canonical.model, identity)
            np.testing.assert_array_equal(canonical.data.qpos, qpos)
            np.testing.assert_array_equal(canonical.data.qvel, qvel)
            for name, value in arrays.items():
                np.testing.assert_array_equal(getattr(canonical.model, name), value)
        repeated, _ = canonical.reset(seed=1042)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(repeated[-3:], 0.0)
        assert canonical.model.nhfield == 0
        assert canonical._substep_probe_hook is None
    finally:
        canonical.close()


@pytest.mark.parametrize("species", SPECIES)
def test_switch_rewards_executed_command_and_exposes_next_command(make_env, species):
    env = make_env(
        species,
        commands=commands(species, switch_interval_s=0.02, straight_probability=0.0, turn_increment_max=0.6),
        reset_noise_scale=0.0,
    )
    observation, _ = env.reset(seed=123)
    executed = env._command_state
    assert executed.event_id == 0
    for _ in range(5):
        np.testing.assert_array_equal(observation[-3:], executed.normalized)
        observation, reward, terminated, truncated, info = env.step(zeros(env))
        assert np.isfinite(reward) and not terminated and not truncated
        assert info["command_event_id"] == executed.event_id
        assert info["desired_heading"] == executed.desired_heading
        assert info["command_v_x"] == float(executed.physical[0])
        assert info["command_yaw_rate"] == float(executed.physical[2])
        assert info["command_time_s"] == executed.time_s
        assert info["actual_heading"] == pytest.approx(env._heading())
        expected_reward = env.tracking_weight * gaussian_tracking_reward(
            info["tracking_error_v"], info["tracking_error_yaw"], velocity_sigma=env.tracking_velocity_sigma
        )
        assert info["reward_tracking"] == pytest.approx(expected_reward)
        executed = env._command_state
        np.testing.assert_array_equal(observation[-3:], executed.normalized)
    assert executed.event_id == 5 // round(0.02 / env.dt)
    assert len(env.direction_controller.events) == executed.event_id + 1


@pytest.mark.parametrize("species", SPECIES)
def test_external_direction_takes_effect_on_next_action_and_masks_stops(make_env, species):
    env = make_env(species, commands=commands(species, switch_interval_s=0.02), reset_noise_scale=0.0)
    env.reset(seed=17)
    target_heading = env._heading() + np.pi / 4
    observation = env.set_direction(target_heading, SPEEDS[species])
    np.testing.assert_array_equal(observation[-3:], env._command_state.normalized)
    assert observation[-1] > 0.0
    external_event = env._command_state.event_id
    for _ in range(5):
        _, _, _, _, info = env.step(zeros(env))
        assert info["command_event_id"] == external_event
        assert info["command_target_source"] == "external"
        assert info["desired_heading"] == pytest.approx(target_heading)
    observation = env.set_direction(target_heading, 0.0)
    np.testing.assert_array_equal(observation[-3:], 0.0)
    _, _, _, _, info = env.step(zeros(env))
    assert not info["tracking_heading_active"]
    assert info["tracking_error_heading"] == 0.0
    assert info["desired_speed"] == 0.0


@pytest.mark.parametrize("species", SPECIES)
def test_identity_binds_task_options_but_not_episode_draws(make_env, species):
    env = make_env(species, terrain=terrain(), run_seed=42)
    first = copy.deepcopy(env.behavior_identity)
    assert first["schema"] == "mesozoic.command-terrain/v1" and first["species"] == species
    assert set(first["sources"]) == {
        "environments/shared/behavior_env.py",
        str(Path(inspect.getfile(get_species_config(species).env_class)).relative_to(REPOSITORY_ROOT)),
        "environments/shared/direction_commands.py",
        "environments/shared/terrain.py",
    }
    env.reset(seed=1)
    env.step(zeros(env))
    env.reset(seed=2)
    assert env.behavior_identity == first
    changed = make_env(species, terrain=terrain(max_slope_degrees=1.0), run_seed=44)
    assert changed.behavior_identity != first


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("flat_probability", [0.0, 1.0])
def test_flat_retention_uses_plane_and_flat_terrain_keeps_hfield(make_env, species, flat_probability):
    env = make_env(species, terrain=terrain(mode="flat"), flat_probability=flat_probability, reset_noise_scale=0.0)
    observation, info = env.reset(seed=1042)
    is_plane = flat_probability == 1.0
    expected_type = mujoco.mjtGeom.mjGEOM_PLANE if is_plane else mujoco.mjtGeom.mjGEOM_HFIELD
    expected_model = env._plane_model if is_plane else env._terrain_model
    expected_data = env._plane_data if is_plane else env._terrain_data
    assert env.model is expected_model and env.data is expected_data
    assert env.model.geom_type[env.floor_geom_id] == expected_type
    assert env.model.nhfield == (0 if is_plane else 1)
    if env._terrain_contact_probe_geoms:
        expected_probe_model = env._plane_probe_model if is_plane else env._terrain_probe_model
        expected_probe_data = env._plane_probe_data if is_plane else env._terrain_probe_data
        assert env._probe_model is expected_probe_model and env._probe_data is expected_probe_data
        assert env._probe_model.geom_type[env.floor_geom_id] == expected_type
        assert env._probe_model.nhfield == env.model.nhfield
    assert (env.terrain is None) == is_plane
    if is_plane:
        assert info["terrain"] == {"family": "flat_plane"}
        assert env.lowest_ground_clearance() == pytest.approx(env.home_ground_clearance(), abs=1e-12)
    else:
        assert env.terrain.config.mode == "flat"
        np.testing.assert_array_equal(env.terrain.heights, 0.0)
        with pytest.raises(NotImplementedError, match="heightfields"):
            env.lowest_ground_clearance()
    assert observation.shape == env.observation_space.shape
    for _ in range(5):
        observation, reward, terminated, truncated, _ = env.step(zeros(env))
        assert np.all(np.isfinite(observation)) and np.isfinite(reward)
        assert not terminated and not truncated


def _mixed_surface_seeds(env):
    seeds = {}
    for seed in range(32):
        env.reset(seed=seed)
        seeds.setdefault(env.terrain is None, seed)
        if len(seeds) == 2:
            return seeds
    pytest.fail("Mixed retention did not produce both plane and heightfield episodes")


def _pools(env, is_plane):
    names = ("model", "data", "probe_model", "probe_data") if env._terrain_contact_probe_geoms else ("model", "data")
    return tuple(getattr(env, f"_{'plane' if is_plane else 'terrain'}_{name}") for name in names)


@pytest.mark.parametrize("species", SPECIES)
def test_mixed_resets_reuse_compiled_pairs_and_replay_both_surfaces(make_env, species):
    env = make_env(species, terrain=terrain(), flat_probability=0.5, run_seed=17, reset_noise_scale=0.05)
    pairs = {is_plane: _pools(env, is_plane) for is_plane in (True, False)}
    for plane_member, terrain_member in zip(pairs[True], pairs[False], strict=True):
        assert plane_member is not terrain_member
    models = {key: pair[::2] for key, pair in pairs.items()}
    geom_types = {key: [model.geom_type.copy() for model in value] for key, value in models.items()}
    identity = copy.deepcopy(env.behavior_identity)
    actions = np.random.default_rng(921).uniform(-0.01, 0.01, (3, *env.action_space.shape))

    def replay():
        rows = []
        for episode_index in range(12):
            observation, info = env.reset(seed=1042 if episode_index == 0 else None)
            is_plane = env.terrain is None
            current = (env.model, env.data, getattr(env, "_probe_model", None), getattr(env, "_probe_data", None))
            for active, pooled in zip(current, pairs[is_plane], strict=False):
                assert active is pooled
            if not is_plane and env._terrain_contact_probe_geoms:
                # The probe copy collides with this episode's surface, not episode 0's.
                np.testing.assert_array_equal(env._probe_model.hfield_data, env.model.hfield_data)
            assert info["episode_index"] == episode_index
            initial = (observation.copy(), env.data.qpos.copy(), env.data.qvel.copy(), env.model.hfield_data.copy())
            rollout = []
            for action in actions:
                obs, reward, terminated, truncated, step_info = env.step(action)
                rollout.append((obs.copy(), reward, terminated, truncated, step_info["desired_heading"]))
            rows.append((is_plane, info, initial, rollout))
            for key, value in models.items():
                for model, geom_type in zip(value, geom_types[key], strict=True):
                    np.testing.assert_array_equal(model.geom_type, geom_type)
        return rows

    first, second = replay(), replay()
    assert {row[0] for row in first} == {False, True}
    assert sum(a[0] != b[0] for a, b in zip(first, first[1:])) >= 2
    for a, b in zip(first, second, strict=True):
        assert a[:2] == b[:2]
        for first_array, second_array in zip(a[2], b[2], strict=True):
            np.testing.assert_array_equal(first_array, second_array)
        for first_step, second_step in zip(a[3], b[3], strict=True):
            np.testing.assert_array_equal(first_step[0], second_step[0])
            assert first_step[1:] == second_step[1:]
    assert env.behavior_identity == identity


@pytest.mark.parametrize("species", SPECIES)
def test_switching_surfaces_clears_reused_physics_and_contact_caches(make_env, species):
    kwargs = dict(terrain=terrain(), flat_probability=0.5, run_seed=17, reset_noise_scale=0.05)
    env, fresh = make_env(species, **kwargs), make_env(species, **kwargs)
    seeds = _mixed_surface_seeds(env)
    probes = bool(env._terrain_contact_probe_geoms)
    for is_plane in (False, True, False):
        env.reset(seed=seeds[not is_plane])
        for data in _pools(env, is_plane)[1::2]:
            data.qpos[2] = -10.0
            data.qvel[:] = 37.0
            data.ctrl[:] = 19.0
            data.qacc_warmstart[:] = 23.0
            data.xfrc_applied[:] = 7.0
            data.qfrc_applied[:] = 9.0
            data.time = 123.0
        env._probe_hit_geom = 0
        env._substep_min_foot_forces = np.full(2, -99.0)
        env._substep_floor_hit_geom = int(env._root_subtree_geoms()[0])
        env._substep_min_heights = np.full(2, -99.0)
        env._substep_contact_step = 0
        env._root_subtree_geom_ids = np.empty(0, dtype=np.int32)
        env._static_floor_geom_ids = np.empty(0, dtype=np.int32)
        env._ground_geom_array = np.empty(0, dtype=np.int32)
        observation, info = env.reset(seed=seeds[is_plane])
        fresh_observation, fresh_info = fresh.reset(seed=seeds[is_plane])
        np.testing.assert_array_equal(observation, fresh_observation)
        assert info == fresh_info
        assert env._probe_hit_geom is None
        assert env._substep_min_foot_forces is None
        assert env._substep_floor_hit_geom is None
        assert env._substep_min_heights is None
        assert env._substep_contact_step == -1
        assert env._ground_geom_array is None
        np.testing.assert_array_equal(env._root_subtree_geoms(), fresh._root_subtree_geoms())
        np.testing.assert_array_equal(env._static_floor_geoms(), fresh._static_floor_geoms())
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "xfrc_applied", "qfrc_applied", "sensordata"):
            np.testing.assert_array_equal(getattr(env.data, name), getattr(fresh.data, name))
        assert env.data.time == 0.0
        if probes:
            assert env._probe_data.time == 0.0
            for name in ("qvel", "xfrc_applied", "qfrc_applied"):
                np.testing.assert_array_equal(getattr(env._probe_data, name), 0.0)
            env._probe_terrain_contacts()
            fresh._probe_terrain_contacts()
            np.testing.assert_array_equal(env._probe_data.contact.geom, fresh._probe_data.contact.geom)
            np.testing.assert_array_equal(env._probe_data.contact.dist, fresh._probe_data.contact.dist)
        assert not env._is_terminated()[0]
        for _ in range(3):
            actual = env.step(zeros(env))
            expected = fresh.step(zeros(fresh))
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:4] == expected[1:4]


def test_model_switch_closes_rendering_resources_bound_to_old_model(make_env):
    from unittest.mock import Mock

    env = make_env("trex", terrain=terrain(), flat_probability=0.5)
    seeds = _mixed_surface_seeds(env)
    env.reset(seed=seeds[True])
    for is_plane in (False, True):
        renderer, viewer = Mock(), Mock()
        env._renderer, env._viewer = renderer, viewer
        env.reset(seed=seeds[is_plane])
        renderer.close.assert_called_once_with()
        viewer.close.assert_called_once_with()
        viewer.update_hfield.assert_not_called()
        assert env._renderer is None and env._viewer is None


def test_surface_boundary_protects_complete_animal(make_env):
    env = make_env("brachiosaurus", terrain=terrain(), reset_noise_scale=0.0)
    env.reset(seed=42)
    env.data.qpos[0] = 7.0
    mujoco.mj_forward(env.model, env.data)
    terminated, info = env._is_terminated()
    assert terminated
    assert info["termination_reason"] == "terrain_boundary"


def test_wrong_compiled_animal_is_rejected(monkeypatch):
    original = behavior_module.build_terrain_model

    def changed_model(*args, **kwargs):
        model = copy.copy(original(*args, **kwargs))
        model.body_mass[1] *= 2
        return model

    monkeypatch.setattr(behavior_module, "build_terrain_model", changed_model)
    with pytest.raises(ValueError, match="animal field body_mass"):
        get_behavior_env_class("velociraptor")(terrain=terrain())


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize(
    "name",
    [
        "follow_direction",
        "follow_direction_speed",
        "terrain_contact",
        "sloped_terrain",
        "bumps_terrain",
        "depressions_terrain",
        "mixed_terrain",
        "combined_terrain",
    ],
)
def test_each_committed_recipe_instantiates_and_steps(species, name):
    from environments.shared.train_behaviors import read_recipe

    recipe, recipe_commands, recipe_terrain, kwargs = read_recipe(RECIPES / species / "behaviors" / f"{name}.toml")
    assert recipe["behavior"]["species"] == species and recipe["behavior"]["name"] == name
    assert recipe["behavior"]["timesteps"] > 0
    env = get_behavior_env_class(species)(commands=recipe_commands, terrain=recipe_terrain, run_seed=42, **kwargs)
    try:
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
        next_obs, reward, terminated, truncated, step_info = env.step(zeros(env))
        assert np.all(np.isfinite(next_obs)) and np.isfinite(reward)
        assert not terminated and not truncated
        assert step_info["command_event_id"] == 0
        assert step_info["heading_error_rad"] == pytest.approx(
            ((step_info["desired_heading"] - step_info["actual_heading"] + np.pi) % (2 * np.pi) - np.pi)
            if step_info["heading_active"]
            else 0.0
        )
        if env.terrain is None:
            assert np.isfinite(env.lowest_ground_clearance())
        else:
            expected_schema = (
                "mesozoic.gentle-terrain/v1" if recipe_terrain.template == "sloped" else "mesozoic.terrain-templates/v2"
            )
            assert info["terrain"]["schema"] == expected_schema
            with pytest.raises(NotImplementedError, match="heightfield"):
                env.lowest_ground_clearance()
    finally:
        env.close()
