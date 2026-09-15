"""Balanced schedules and actual contact-model transitions across families."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pytest

from environments.shared.behavior_env import get_behavior_env_class
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.terrain import TerrainConfig
from environments.shared.terrain_sampling import (
    TERRAIN_FAMILIES,
    TerrainSamplerConfig,
    get_sampled_behavior_env_class,
    sampler_source_identity,
    select_terrain_family,
)


@pytest.mark.parametrize("bad_weight", [-1, True, 1.0, "1", None])
def test_sampler_rejects_invalid_weights(bad_weight):
    with pytest.raises(ValueError, match="nonnegative integer"):
        TerrainSamplerConfig(bumps=bad_weight)


@pytest.mark.parametrize("weights", [dict.fromkeys(TERRAIN_FAMILIES, 0), {"flat": 1000}])
def test_sampler_rejects_empty_or_oversized_blocks(weights):
    with pytest.raises(ValueError, match="total weight"):
        TerrainSamplerConfig(**weights)


def test_every_shuffled_block_has_exact_configured_coverage():
    config = TerrainSamplerConfig(flat=2, sloped=1, bumps=3, depressions=0, mixed=1)
    assert config.families == ("flat", "sloped", "bumps", "mixed")
    expected = Counter({family: count for family, count in asdict(config).items() if count})
    orders = []
    for block in range(8):
        selections = [
            select_terrain_family(config, run_seed=23, episode_seed=99, episode_index=block * config.block_size + i)
            for i in range(config.block_size)
        ]
        assert Counter(selection.family for selection in selections) == expected
        assert {selection.block_index for selection in selections} == {block}
        assert [selection.block_position for selection in selections] == list(range(config.block_size))
        assert len({selection.selection_seed for selection in selections}) == 1
        orders.append(tuple(selection.family for selection in selections))
    assert len(set(orders)) > 1


def test_schedule_is_random_access_reproducible_and_seeded():
    config = TerrainSamplerConfig()

    def schedule(run_seed, episode_seed):
        return [
            select_terrain_family(config, run_seed=run_seed, episode_seed=episode_seed, episode_index=i)
            for i in range(30)
        ]

    first = schedule(23, 99)
    assert first == schedule(23, 99)
    assert [item.family for item in first] != [item.family for item in schedule(24, 99)]
    assert [item.family for item in first] != [item.family for item in schedule(23, 100)]
    assert first[19] == select_terrain_family(config, run_seed=23, episode_seed=99, episode_index=19)


@pytest.mark.parametrize("name", ["run_seed", "episode_seed", "episode_index"])
@pytest.mark.parametrize("bad_value", [-1, True, 0.5])
def test_schedule_rejects_invalid_seed_or_index(name, bad_value):
    values = dict(run_seed=42, episode_seed=17, episode_index=0)
    values[name] = bad_value
    with pytest.raises(ValueError, match=name):
        select_terrain_family(TerrainSamplerConfig(), **values)


@pytest.mark.parametrize(
    "species", ["trex", "velociraptor", "brachiosaurus", "dibothrosuchus", "compsognathus", "compsognathus_robot"]
)
def test_factory_keeps_existing_species_behavior_class(species):
    sampled = get_sampled_behavior_env_class(species)
    assert issubclass(sampled, get_behavior_env_class(species))
    assert sampled is get_sampled_behavior_env_class(species)
    assert get_sampled_behavior_env_class("Tyrannosaurus Rex") is get_sampled_behavior_env_class("trex")


def terrain(**kwargs):
    return TerrainConfig(extent=8.0, nrow=81, ncol=81, apron_radius=4.0, template="mixed", **kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"terrain_sampler": {}}, "TerrainSamplerConfig"),
        ({"terrain": None}, "gentle terrain"),
        ({"terrain": terrain(mode="flat")}, "gentle terrain"),
        ({"flat_probability": 0.25}, "flat_probability"),
        ({"flat_probability": float("nan")}, "flat_probability"),
        ({"terrain": replace(terrain(), template="sloped", feature_radius_min=0.2)}, "three samples"),
    ],
)
def test_invalid_mixture_is_rejected_before_model_creation(kwargs, message):
    arguments = dict(terrain_sampler=TerrainSamplerConfig(), terrain=terrain())
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        get_sampled_behavior_env_class("trex")(**arguments)


@pytest.fixture(params=["trex", "velociraptor"])
def sampled_env(request):
    env = get_sampled_behavior_env_class(request.param)(
        terrain_sampler=TerrainSamplerConfig(),
        terrain=terrain(),
        run_seed=123,
        reset_noise_scale=0.01,
        max_episode_steps=10,
        commands=DirectionCommandConfig(
            speed_range=(0.5, 1.05), stop_probability=0.15, turn_increment_max=0.5, straight_probability=0.25
        ),
    )
    yield env
    env.close()


def test_resets_balance_families_without_changing_task_identity(sampled_env):
    env = sampled_env
    identity = env.behavior_identity
    selected = []
    for episode in range(10):
        _, info = env.reset(seed=29 if episode == 0 else None)
        selection = info["terrain_sampling"]
        selected.append(selection["family"])
        assert selection["mode"] == "balanced_shuffle"
        assert selection["block_index"] == episode // 5
        assert selection["block_position"] == episode % 5
        assert selection["weights"] == asdict(TerrainSamplerConfig())
        assert env.behavior_identity == identity
        assert env.terrain_config.template == "mixed"
        if selection["family"] == "flat":
            assert env.terrain is None
            assert env.model.nhfield == 0
            assert info["terrain"]["family"] == "flat_plane"
        else:
            assert env.model.nhfield == 1
            assert env.terrain.config.template == selection["family"]
            assert info["terrain"]["template"] == selection["family"]
        if hasattr(env, "_probe_model"):
            assert env._probe_model.nhfield == env.model.nhfield
            np.testing.assert_array_equal(env._probe_model.hfield_data, env.model.hfield_data)
    assert Counter(selected[:5]) == Counter(TERRAIN_FAMILIES)
    assert Counter(selected[5:]) == Counter(TERRAIN_FAMILIES)
    assert identity["sampler_sources"] == sampler_source_identity()
    base = super(type(env).__mro__[1], env).behavior_identity
    assert {key: value for key, value in identity.items() if key not in ("terrain_sampler", "sampler_sources")} == base


def test_explicit_seed_repeats_course_and_commands_while_unseeded_reset_changes_layout(sampled_env):
    env = sampled_env
    options = {"terrain_family": "bumps"}
    observation, info = env.reset(seed=71, options=options)
    field = env.model.hfield_data.copy()
    qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
    env.step(np.zeros(env.action_space.shape))
    second_observation, second_info = env.reset(seed=71, options=options)
    np.testing.assert_array_equal(env.model.hfield_data, field)
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.qvel, qvel)
    np.testing.assert_array_equal(second_observation, observation)
    assert info == second_info
    env.reset(options=options)
    assert not np.array_equal(env.model.hfield_data, field)
    assert options == {"terrain_family": "bumps"}


def test_family_override_does_not_change_commands_or_persist(sampled_env):
    env = sampled_env
    command_streams = []
    for family in TERRAIN_FAMILIES:
        observation, info = env.reset(seed=71, options={"terrain_family": family})
        command_streams.append((observation[-3:].copy(), info["command_seed"], env.command_manifest()))
        assert info["terrain_sampling"]["family"] == family
        assert info["terrain_sampling"]["mode"] == "evaluation_override"
    for commands, command_seed, manifest in command_streams[1:]:
        np.testing.assert_array_equal(commands, command_streams[0][0])
        assert command_seed == command_streams[0][1]
        assert manifest == command_streams[0][2]
    _, info = env.reset()
    assert info["terrain_sampling"]["mode"] == "balanced_shuffle"
    expected = select_terrain_family(env.terrain_sampler, run_seed=123, episode_seed=71, episode_index=1)
    assert info["terrain_sampling"]["family"] == expected.family


def test_course_and_identity_stay_fixed_during_episode(sampled_env):
    env = sampled_env
    env.reset(seed=81, options={"terrain_family": "depressions"})
    field = env.model.hfield_data.copy()
    identity = env.behavior_identity
    manifestation = env.terrain.manifest()
    for _ in range(3):
        env.step(np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(env.model.hfield_data, field)
        assert env.terrain.manifest() == manifestation
        assert env.behavior_identity == identity


def test_invalid_override_does_not_advance_episode(sampled_env):
    env = sampled_env
    env.reset(seed=9)
    before = env._episode_index
    with pytest.raises(ValueError, match="enabled family"):
        env.reset(options={"terrain_family": "unknown"})
    assert env._episode_index == before
    env.terrain_sampler = TerrainSamplerConfig(bumps=0)
    with pytest.raises(ValueError, match="enabled family"):
        env.reset(options={"terrain_family": "bumps"})


def test_base_settings_are_restored_when_reset_fails(sampled_env, monkeypatch):
    env = sampled_env
    identity = env.behavior_identity

    def failed_reset(*args, **kwargs):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(get_behavior_env_class(env.species), "reset", failed_reset)
    with pytest.raises(RuntimeError, match="reset failed"):
        env.reset(seed=9, options={"terrain_family": "sloped"})
    assert env.terrain_config.template == "mixed"
    assert env.flat_probability == 0
    assert env.behavior_identity == identity
