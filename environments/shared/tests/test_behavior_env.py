"""Real-model integration checks for the shared species behavior environments."""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from environments.shared.behavior_env import canonical_env_parameters, get_behavior_env_class
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.species_registry import get_species_config
from environments.shared.terrain import TerrainConfig, TerrainRealization, apply_terrain
from environments.trex.envs.behavior_env import TRexBehaviorEnv

SPECIES = ("trex", "velociraptor", "brachiosaurus", "dibothrosuchus", "compsognathus", "compsognathus_robot")
NEW_SPECIES = SPECIES[1:]
SPEEDS = {
    "velociraptor": 2.0,
    "brachiosaurus": 0.75,
    "dibothrosuchus": 0.9,
    "compsognathus": 0.08,
    "compsognathus_robot": 0.04,
    "trex": 1.05,
}


@pytest.fixture
def make_env():
    instances = []

    def factory(species, **kwargs):
        speed = SPEEDS[species]
        kwargs.setdefault("commands", DirectionCommandConfig(cruise_speed=speed, speed_scale=speed * 1.5))
        env = get_behavior_env_class(species)(**kwargs)
        instances.append(env)
        return env

    yield factory
    for env in instances:
        env.close()


def terrain(**kwargs):
    return TerrainConfig(extent=8.0, nrow=81, ncol=81, apron_radius=4.0, **kwargs)


@pytest.mark.parametrize("species", SPECIES)
def test_factory_preserves_canonical_inheritance(species):
    assert issubclass(get_behavior_env_class(species), get_species_config(species).env_class)
    assert "forward_vel_weight" in canonical_env_parameters(species)
    assert get_behavior_env_class(species) is get_behavior_env_class(species)
    assert get_behavior_env_class("Tyrannosaurus Rex") is TRexBehaviorEnv


@pytest.mark.parametrize("species", SPECIES)
def test_species_has_real_surface_command_inputs_and_reproducible_rollout(make_env, species):
    env = make_env(species, terrain=terrain(template="mixed"), reset_noise_scale=0.01)
    canonical = get_species_config(species).env_class(reset_noise_scale=0.01)
    try:
        observation, info = env.reset(seed=1729)
        canonical.reset(seed=1729)
        assert observation.shape == canonical.observation_space.shape
        assert env.action_space == canonical.action_space
        np.testing.assert_allclose(env.data.qpos, canonical.data.qpos, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(env.data.qvel, canonical.data.qvel)
        np.testing.assert_array_equal(observation[-3:], env._command)
        assert observation[-3] > 0
        assert env.behavior_identity["parent_plant"]["species"] == species
        assert env.model.nhfield == 1
        props = np.flatnonzero(env.model.body_mocapid >= 0)
        prop_geoms = np.isin(env.model.geom_bodyid, props)
        assert not np.any(env.model.geom_contype[prop_geoms])
        assert not np.any(env.model.geom_conaffinity[prop_geoms])
        surface = env.model.hfield_data.copy()
        action = np.zeros(env.action_space.shape)
        first = env.step(action)
        assert np.isfinite(first[1])
        assert first[4]["reward_tracking"] >= 0
        assert not first[2]
        env.reset(seed=1729)
        np.testing.assert_array_equal(env.model.hfield_data, surface)
        second = env.step(action)
        np.testing.assert_array_equal(first[0], second[0])
        assert first[1:4] == second[1:4]
        assert info["terrain"]["template"] == "mixed"
        env.reset(seed=1730)
        assert not np.array_equal(env.model.hfield_data, surface)
    finally:
        canonical.close()


@pytest.mark.parametrize("species", NEW_SPECIES)
def test_surface_relative_height_reward_and_termination(make_env, species):
    env = make_env(species, terrain=terrain(mode="flat"), reset_noise_scale=0.0)
    env.reset(seed=42)
    action = np.zeros(env.action_space.shape)
    reward, original_info = env._get_reward_info(action)
    original_height = env._current_clearances().copy()
    shift = -0.5
    env.terrain = TerrainRealization(env.terrain_config, 999, 0, np.full((81, 81), shift))
    apply_terrain(env.model, env.terrain, env.data)
    env.data.qpos[2] += shift
    mujoco.mj_forward(env.model, env.data)
    np.testing.assert_allclose(env._current_clearances(), original_height, atol=1e-12, rtol=0)
    assert not env._is_terminated()[0]
    _, info = env._get_reward_info(action)
    if "reward_height" in info:
        assert info["reward_height"] == pytest.approx(original_info["reward_height"])
    assert np.isfinite(reward)


@pytest.mark.parametrize("species", NEW_SPECIES)
def test_substep_low_root_is_not_hidden_by_recovery(make_env, species):
    env = make_env(species, reset_noise_scale=0.0)
    env.reset(seed=42)
    original = env.data.qpos.copy()
    env.data.qpos[2] = env.healthy_z_range[0] * 0.5
    mujoco.mj_forward(env.model, env.data)
    env._probe_ground_clearance()
    env.data.qpos[:] = original
    mujoco.mj_forward(env.model, env.data)
    env._probe_ground_clearance()
    terminated, info = env._is_terminated()
    assert terminated
    assert info["termination_reason"] == "fallen"


@pytest.mark.parametrize("species", NEW_SPECIES)
def test_small_species_cannot_pass_a_walk_command_while_stationary(make_env, species):
    env = make_env(species, reset_noise_scale=0.0)
    env.reset(seed=42)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env._get_reward_info(np.zeros(env.action_space.shape))
    assert not env._command_metrics["tracking_in_tolerance"]
    assert env._command_metrics["tracking_error_requested_speed"] == pytest.approx(SPEEDS[species])
    assert env.behavior_identity["tracking_tolerances"] == env.tracking_tolerances
    assert env.tracking_velocity_sigma == pytest.approx(env.direction_controller.config.speed_scale / 6)


def test_dibothrosuchus_snout_minimum_is_surface_relative(make_env):
    env = make_env("dibothrosuchus", terrain=terrain(mode="flat"), reset_noise_scale=0.0)
    env.reset(seed=42)
    original = env.data.site_xpos[env.snout_tip_site_id].copy()
    env.data.site_xpos[env.snout_tip_site_id, 2] = 0.03
    env._probe_ground_clearance()
    env.data.site_xpos[env.snout_tip_site_id] = original
    env._probe_ground_clearance()
    terminated, info = env._is_terminated()
    assert terminated
    assert info["termination_reason"] == "head_contact"


def test_plane_retention_uses_original_contact_geometry(make_env):
    env = make_env("velociraptor", terrain=terrain(), flat_probability=1.0, reset_noise_scale=0.0)
    _, info = env.reset(seed=42)
    assert env.terrain is None
    assert info["terrain"]["family"] == "flat_plane"
    assert env.model.nhfield == 0
    assert env.model.geom_type[env.floor_geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
    env.flat_probability = 0.0
    env.reset(seed=42)
    assert env.model.nhfield == 1


def test_surface_boundary_protects_complete_animal(make_env):
    env = make_env("brachiosaurus", terrain=terrain(), reset_noise_scale=0.0)
    env.reset(seed=42)
    env.data.qpos[0] = 7.0
    mujoco.mj_forward(env.model, env.data)
    terminated, info = env._is_terminated()
    assert terminated
    assert info["termination_reason"] == "terrain_boundary"


def test_wrong_compiled_animal_is_rejected(monkeypatch):
    from environments.shared import behavior_env

    original = behavior_env.build_terrain_model

    def changed_model(*args, **kwargs):
        model = copy.copy(original(*args, **kwargs))
        model.body_mass[1] *= 2
        return model

    monkeypatch.setattr(behavior_env, "build_terrain_model", changed_model)
    with pytest.raises(ValueError, match="animal field body_mass"):
        get_behavior_env_class("velociraptor")(terrain=terrain())
