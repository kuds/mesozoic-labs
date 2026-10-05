"""Balanced schedules, and each episode's terrain family through the behavior env's one selector."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from environments.shared.behavior_env import get_behavior_env_class
from environments.shared.behavior_evaluation import terrain_family_from_reset
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.species_registry import get_species_config
from environments.shared.terrain import TerrainConfig
from environments.shared.terrain_sampling import (
    TERRAIN_FAMILIES,
    TerrainSamplerConfig,
    select_terrain_family,
)


def only(**weights):
    """A sampler that enables exactly the named families."""
    return TerrainSamplerConfig(**{**dict.fromkeys(TERRAIN_FAMILIES, 0), **weights})


@pytest.mark.parametrize("family", TERRAIN_FAMILIES)
@pytest.mark.parametrize("bad_weight", [-1, True, 1.0, "1", None])
def test_sampler_rejects_invalid_weights(family, bad_weight):
    with pytest.raises(ValueError, match=f"weight {family} must be a nonnegative integer"):
        TerrainSamplerConfig(**{family: bad_weight})


@pytest.mark.parametrize("weights", [dict.fromkeys(TERRAIN_FAMILIES, 0), {"flat": 1000}])
def test_sampler_rejects_empty_or_oversized_blocks(weights):
    with pytest.raises(ValueError, match="total weight"):
        TerrainSamplerConfig(**weights)


def test_terrain_contact_is_the_last_family_and_off_by_default():
    """Consolidation PR-8: appended with weight 0, so TerrainSamplerConfig() is still the five-family block."""
    assert TERRAIN_FAMILIES == ("flat", "sloped", "bumps", "depressions", "mixed", "terrain_contact")
    assert [field.name for field in fields(TerrainSamplerConfig)] == list(TERRAIN_FAMILIES)
    assert TerrainSamplerConfig().families == TERRAIN_FAMILIES[:5]
    assert TerrainSamplerConfig().block_size == 5


def test_every_shuffled_block_has_exact_configured_coverage():
    config = TerrainSamplerConfig(flat=2, sloped=1, bumps=3, depressions=0, mixed=1, terrain_contact=2)
    assert config.families == ("flat", "sloped", "bumps", "mixed", "terrain_contact")
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


def test_terrain_contact_takes_the_plane_slots_of_a_same_shaped_template_block():
    """A one-plane-in-four block puts the plane in the same episodes whatever its other family is, so the
    terrain_contact recipes share the template recipes' schedule; listing "flat" first fixes it."""
    for run_seed, episode_seed in ((0, 1042), (23, 99), (7, 0)):
        planes = {
            family: [
                select_terrain_family(
                    only(flat=1, **{family: 3}), run_seed=run_seed, episode_seed=episode_seed, episode_index=i
                ).family
                == "flat"
                for i in range(40)
            ]
            for family in TERRAIN_FAMILIES[1:]
        }
        assert all(planes[family] == planes["sloped"] for family in planes)
        assert sum(planes["sloped"]) == 10


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


def terrain(**kwargs):
    return TerrainConfig(extent=8.0, nrow=81, ncol=81, apron_radius=4.0, **{"template": "mixed", **kwargs})


ENV_KWARGS = dict(
    run_seed=123,
    reset_noise_scale=0.01,
    max_episode_steps=10,
    commands=DirectionCommandConfig(
        speed_range=(0.5, 1.05), stop_probability=0.15, turn_increment_max=0.5, straight_probability=0.25
    ),
)


@pytest.fixture
def canonical_init_forbidden(monkeypatch):
    """A refusal must come before the canonical env compiles its model."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the canonical model was built before the refusal")

    monkeypatch.setattr(get_species_config("trex").env_class, "__init__", forbidden)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"terrain_sampler": {}}, "TerrainSamplerConfig"),
        ({"terrain": None}, "terrain_sampler requires an enabled terrain"),
        ({"terrain": replace(terrain(), template="sloped", feature_radius_min=0.2)}, "three samples"),
    ],
)
def test_invalid_mixture_is_rejected_before_model_creation(canonical_init_forbidden, kwargs, message):
    arguments = {"terrain_sampler": TerrainSamplerConfig(), "terrain": terrain(), **kwargs}
    with pytest.raises(ValueError, match=message):
        get_behavior_env_class("trex")(**arguments)


def test_flat_probability_is_no_longer_a_behavior_env_argument():
    with pytest.raises(TypeError, match="flat_probability"):
        get_behavior_env_class("trex")(terrain=terrain(), terrain_sampler=TerrainSamplerConfig(), flat_probability=0.25)


@pytest.fixture(params=["trex", "velociraptor"])
def sampled_env(request):
    env = get_behavior_env_class(request.param)(terrain_sampler=TerrainSamplerConfig(), terrain=terrain(), **ENV_KWARGS)
    yield env
    env.close()


def test_resets_balance_families_without_changing_task_identity(sampled_env):
    env = sampled_env
    base = env.terrain_config
    identity = env.task_fingerprint
    selected = []
    for episode in range(10):
        _, info = env.reset(seed=29 if episode == 0 else None)
        selection = info["terrain_sampling"]
        selected.append(selection["family"])
        assert selection == {
            **asdict(select_terrain_family(env.terrain_sampler, run_seed=123, episode_seed=29, episode_index=episode)),
            "mode": "balanced_shuffle",
            "weights": asdict(TerrainSamplerConfig()),
        }
        assert env.task_fingerprint == identity
        assert env.terrain_config is base
        assert terrain_family_from_reset(info) == selection["family"]
        if selection["family"] == "flat":
            assert env.terrain is None
            assert env.model.nhfield == 0
            assert info["terrain"]["family"] == "flat_plane"
        else:
            assert env.model.nhfield == 1
            assert env.terrain.config == replace(base, template=selection["family"])
            assert info["terrain"]["template"] == selection["family"]
        if hasattr(env, "_probe_model"):
            assert env._probe_model.nhfield == env.model.nhfield
            np.testing.assert_array_equal(env._probe_model.hfield_data, env.model.hfield_data)
    assert Counter(selected[:5]) == Counter(TerrainSamplerConfig().families)
    assert Counter(selected[5:]) == Counter(TerrainSamplerConfig().families)
    assert identity["env"]["terrain_sampler"] == asdict(TerrainSamplerConfig())
    fixed = get_behavior_env_class(env.species)(terrain=terrain(), **ENV_KWARGS)
    try:
        fixed_identity = fixed.task_fingerprint
    finally:
        fixed.close()
    assert fixed_identity["env"]["terrain_sampler"] is None and "flat_probability" not in fixed_identity["env"]

    def without_sampler(fingerprint):
        env_section = {key: value for key, value in fingerprint["env"].items() if key != "terrain_sampler"}
        return {**{key: value for key, value in fingerprint.items() if key != "task_sha256"}, "env": env_section}

    assert without_sampler(identity) == without_sampler(fixed_identity)
    assert identity["task_sha256"] != fixed_identity["task_sha256"]


def test_explicit_seed_repeats_course_and_commands_while_unseeded_reset_changes_layout(sampled_env):
    env = sampled_env
    options = {"terrain_family": "bumps"}
    observation, info = env.reset(seed=71, options=options)
    field = env.model.hfield_data.copy()
    qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
    env.step(np.zeros(env.action_space.shape))
    second_observation, second_info = env.reset(seed=np.int64(71), options=options)
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
    for family in env.terrain_families:
        observation, info = env.reset(seed=71, options={"terrain_family": family})
        command_streams.append((observation[-3:].copy(), info["command_seed"], env.command_manifest()))
        assert info["terrain_sampling"]["family"] == family
        assert info["terrain_sampling"]["mode"] == "evaluation_override"
        assert terrain_family_from_reset(info) == family
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
    identity = env.task_fingerprint
    manifestation = env.terrain.manifest()
    for _ in range(3):
        env.step(np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(env.model.hfield_data, field)
        assert env.terrain.manifest() == manifestation
        assert env.task_fingerprint == identity


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


@pytest.mark.parametrize(
    "reset_kwargs, message",
    [
        ({"seed": 5, "options": {"terrain_family": "terrain_contact"}}, "enabled family: flat, sloped, bumps"),
        ({"seed": -1}, "seed must be a nonnegative integer"),
        ({"seed": True}, "seed must be a nonnegative integer"),
        ({"seed": 1.5}, "seed must be a nonnegative integer"),
    ],
)
def test_refused_reset_does_not_advance_or_reseed_the_episode(sampled_env, reset_kwargs, message):
    env = sampled_env
    env.reset(seed=9)
    before = (env._episode_seed, env._episode_index, env.model)
    with pytest.raises(ValueError, match=message):
        env.reset(**reset_kwargs)
    assert (env._episode_seed, env._episode_index, env.model) == before


def test_a_family_whose_surface_cannot_be_built_is_refused_before_the_episode_stream_moves():
    """A sampler reassigned after construction is checked at reset: a family the terrain map cannot carry, or any
    terrain family on an env without terrain, is refused with seed, index and model unchanged (as the sampled env
    left them for a family its map could not carry)."""
    coarse = get_behavior_env_class("trex")(
        terrain=replace(terrain(), template="sloped", feature_radius_min=0.2),
        terrain_sampler=only(flat=1, sloped=1),
        **ENV_KWARGS,
    )
    plane = get_behavior_env_class("trex")(**ENV_KWARGS)
    try:
        for env, sampler, message in (
            (coarse, only(bumps=1), "three samples"),
            (plane, only(sloped=1), "terrain family 'sloped' requires an enabled terrain configuration"),
        ):
            env.reset(seed=9)
            env.reset()
            before = (env._episode_seed, env._episode_index, env.model)
            env.terrain_sampler = sampler
            for seed in (33, None):
                with pytest.raises(ValueError, match=message):
                    env.reset(seed=seed)
                assert (env._episode_seed, env._episode_index, env.model) == before
    finally:
        coarse.close()
        plane.close()


@pytest.mark.parametrize("species", ["trex", "brachiosaurus"])
@pytest.mark.parametrize("base_mode", ["gentle", "flat"])
def test_each_family_sets_its_own_surface_on_the_terrain_map(species, base_mode):
    """terrain_contact, which the sampler could not express before consolidation PR-8, is the zero-height
    heightfield on either base, and a template family is that gentle template on either base. The reset record
    carries this sampler's own weights."""
    base = terrain(mode=base_mode)
    sampler = only(flat=1, sloped=1, terrain_contact=1)
    env = get_behavior_env_class(species)(terrain=base, terrain_sampler=sampler, run_seed=5, reset_noise_scale=0.0)
    try:
        assert env.terrain_families == ("flat", "sloped", "terrain_contact")
        _, info = env.reset(seed=3, options={"terrain_family": "terrain_contact"})
        assert info["terrain_sampling"] == {
            **asdict(select_terrain_family(sampler, run_seed=5, episode_seed=3, episode_index=0)),
            "family": "terrain_contact",
            "mode": "evaluation_override",
            "weights": asdict(sampler),
        }
        assert env.model is env._terrain_model and env.model.nhfield == 1
        assert env.terrain.config == replace(base, mode="flat")
        np.testing.assert_array_equal(env.terrain.heights, 0.0)
        assert terrain_family_from_reset(info) == "terrain_contact"
        with pytest.raises(NotImplementedError, match="heightfields"):
            env.lowest_ground_clearance()
        _, info = env.reset(seed=3, options={"terrain_family": "sloped"})
        assert env.terrain.config == replace(base, mode="gentle", template="sloped")
        assert np.ptp(env.terrain.heights) > 0
        assert terrain_family_from_reset(info) == "sloped"
        _, info = env.reset(seed=3, options={"terrain_family": "flat"})
        assert env.model is env._plane_model and env.model.nhfield == 0
        assert terrain_family_from_reset(info) == "flat"
        assert env.terrain_config == base
        for _ in range(3):
            observation, reward, terminated, _, _ = env.step(np.zeros(env.action_space.shape))
            assert np.isfinite(observation).all() and np.isfinite(reward) and not terminated
    finally:
        env.close()


@pytest.mark.parametrize("species", ["trex", "velociraptor"])
def test_the_task_fingerprint_follows_a_reassigned_sampler(species):
    """Reset reads the sampler, the reward the tracking weight and the step the course distance as they stand, so
    the task fingerprint does too: after a reassignment it is the fingerprint of an env built with the new values."""
    env_class = get_behavior_env_class(species)
    env = env_class(terrain=terrain(), terrain_sampler=only(flat=1, sloped=1), **ENV_KWARGS)
    fresh = env_class(terrain=terrain(), terrain_sampler=only(bumps=1), **ENV_KWARGS)
    retuned = env_class(
        terrain=terrain(), terrain_sampler=only(bumps=1), tracking_weight=7, course_distance=12, **ENV_KWARGS
    )
    try:
        before = env.task_fingerprint["task_sha256"]
        env.terrain_sampler = only(bumps=1)
        _, info = env.reset(seed=9)
        assert info["terrain_sampling"]["family"] == "bumps"
        assert env.task_fingerprint["task_sha256"] != before
        assert env.task_fingerprint == fresh.task_fingerprint
        env.tracking_weight, env.course_distance = 7, 12
        assert env.task_fingerprint == retuned.task_fingerprint != fresh.task_fingerprint
    finally:
        env.close()
        fresh.close()
        retuned.close()


def test_a_task_without_a_sampler_has_one_family_and_no_sampling_record():
    """The plane, or the terrain's own surface on every episode, as before; only its own family is accepted."""
    plane = get_behavior_env_class("trex")(**ENV_KWARGS)
    contact = get_behavior_env_class("trex")(terrain=terrain(mode="flat"), **ENV_KWARGS)
    try:
        assert plane.terrain_families == ("flat",)
        assert contact.terrain_families == ("terrain_contact",)
        for env in (plane, contact):
            assert env.task_fingerprint["env"]["terrain_sampler"] is None
            assert "flat_probability" not in env.task_fingerprint["env"]
        for index in range(3):
            _, info = plane.reset(seed=3 if index == 0 else None, options={"terrain_family": "flat"})
            assert plane.terrain is None and "terrain_sampling" not in info
            assert info["terrain"] == {"family": "flat_plane"}
            _, info = contact.reset(seed=3 if index == 0 else None)
            assert contact.terrain.config == terrain(mode="flat") and "terrain_sampling" not in info
            assert terrain_family_from_reset(info) == "terrain_contact"
        with pytest.raises(ValueError, match="enabled family: flat$"):
            plane.reset(options={"terrain_family": "terrain_contact"})
        with pytest.raises(ValueError, match="enabled family: terrain_contact$"):
            contact.reset(options={"terrain_family": "flat"})
    finally:
        plane.close()
        contact.close()
