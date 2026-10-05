"""Keep the supported behavior recipes complete and physically species-scaled, and refuse malformed ones."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.species_names import species_display_names
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.terrain import TerrainConfig, generate_terrain
from environments.shared.terrain_sampling import TERRAIN_FAMILIES, TerrainSamplerConfig, select_terrain_family

ROOT = Path(__file__).resolve().parents[3]
SPECIES = tuple(species_display_names(backend="stable-baselines3"))
BEHAVIORS = (
    "difficult_terrain",
    "follow_direction_difficult_terrain",
    "follow_direction",
    "follow_direction_speed",
    "terrain_contact",
    "sloped_terrain",
    "bumps_terrain",
    "depressions_terrain",
    "mixed_terrain",
    "combined_terrain",
    "combined_mixed_terrain",
)
SAMPLER_BEHAVIORS = {"difficult_terrain", "follow_direction_difficult_terrain"}
PLANE_BEHAVIORS = {"follow_direction", "follow_direction_speed"}
MODEL_PATHS = {
    "velociraptor": "velociraptor/assets/raptor.xml",
    "trex": "trex/assets/trex.xml",
    "brachiosaurus": "brachiosaurus/assets/brachiosaurus.xml",
    "dibothrosuchus": "dibothrosuchus/assets/dibothrosuchus.xml",
    "compsognathus": "compsognathus/assets/compsognathus.xml",
    "compsognathus_robot": "compsognathus/assets/compsognathus_robot.xml",
}


def _recipe(species, behavior):
    with (ROOT / "configs" / species / "behaviors" / f"{behavior}.toml").open("rb") as source:
        return tomllib.load(source)


def _terrain(recipe):
    values = dict(recipe["terrain"])
    assert values.pop("enabled")
    return TerrainConfig(**values)


@pytest.mark.parametrize("species", SPECIES)
def test_every_registered_sb3_species_has_all_supported_behaviors(species):
    paths = ROOT / "configs" / species / "behaviors"
    assert {path.stem for path in paths.glob("*.toml")} == set(BEHAVIORS)


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_recipe_resolves_its_own_locomotion_parent_and_valid_task(species, behavior):
    recipe = _recipe(species, behavior)
    sections = {"behavior", "commands", "terrain", "env", "ppo"}
    if behavior not in PLANE_BEHAVIORS:
        sections.add("terrain_sampler")
    assert set(recipe) == sections
    assert recipe["behavior"]["species"] == species
    assert recipe["behavior"]["parent"] == "locomotion"
    assert recipe["behavior"]["timesteps"] > 0
    assert recipe["behavior"]["name"]
    parent = load_stage_manifest(species).resolve(recipe["behavior"]["parent"])
    assert parent.legacy_number == 2
    assert (ROOT / "configs" / species / parent.config_file).is_file()

    commands = DirectionCommandConfig(**recipe["commands"])
    turning = behavior.startswith(("follow_", "combined_"))
    assert (commands.turn_increment_max > 0) == turning
    variable_speed = behavior in {
        "follow_direction_speed",
        "combined_terrain",
        "combined_mixed_terrain",
        "follow_direction_difficult_terrain",
    }
    assert (commands.speed_range is not None) == variable_speed
    assert (commands.stop_probability > 0) == variable_speed

    if behavior in {"follow_direction", "follow_direction_speed"}:
        assert recipe["terrain"] == {"enabled": False}
    else:
        terrain = _terrain(recipe)
        assert (terrain.mode == "flat") == (behavior == "terrain_contact")
        expected_template = {
            "bumps_terrain": "bumps",
            "depressions_terrain": "depressions",
            "mixed_terrain": "mixed",
            "combined_mixed_terrain": "mixed",
            "difficult_terrain": "mixed",
            "follow_direction_difficult_terrain": "mixed",
        }.get(behavior, "sloped")
        assert terrain.template == expected_template
        assert "flat_probability" not in recipe["env"]
        own = "terrain_contact" if terrain.mode == "flat" else terrain.template
        # Consolidation PR-8: the former flat_probability = 0.25 is one plane episode in each block of four.
        expected_weights = (
            {**dict.fromkeys(TERRAIN_FAMILIES, 1), "terrain_contact": 0}
            if behavior in SAMPLER_BEHAVIORS
            else {**dict.fromkeys(TERRAIN_FAMILIES, 0), "flat": 1, own: 3}
        )
        assert recipe["terrain_sampler"] == expected_weights
        assert list(recipe["terrain_sampler"]) == list(TERRAIN_FAMILIES)
        assert terrain.episode_variation > 0


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize(
    "behavior,diagnostic",
    [("difficult_terrain", "mixed_terrain"), ("follow_direction_difficult_terrain", "combined_mixed_terrain")],
)
def test_unified_terrain_recipes_cover_all_families_and_preserve_species_profiles(species, behavior, diagnostic):
    recipe = _recipe(species, behavior)
    focused_recipe = _recipe(species, diagnostic)
    assert recipe["terrain_sampler"] == {
        "flat": 1,
        "sloped": 1,
        "bumps": 1,
        "depressions": 1,
        "mixed": 1,
        "terrain_contact": 0,
    }
    assert recipe["behavior"]["name"] == behavior
    assert recipe["behavior"]["timesteps"] == 3_000_000
    assert recipe["commands"] == focused_recipe["commands"]
    assert recipe["terrain"] == focused_recipe["terrain"]
    assert recipe["env"] == focused_recipe["env"]


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("behavior", sorted(set(BEHAVIORS) - PLANE_BEHAVIORS - SAMPLER_BEHAVIORS))
def test_single_template_recipe_keeps_exactly_one_plane_episode_per_block_of_four(species, behavior):
    """The section 8 distribution change: a Bernoulli draw (plane with p = 0.25) became balanced blocks."""
    sampler = TerrainSamplerConfig(**_recipe(species, behavior)["terrain_sampler"])
    assert sampler.block_size == 4
    for run_seed, episode_seed in ((0, 1042), (42, 7)):
        schedule = [
            select_terrain_family(sampler, run_seed=run_seed, episode_seed=episode_seed, episode_index=i).family
            for i in range(100)
        ]
        assert [schedule[i : i + 4].count("flat") for i in range(0, 100, 4)] == [1] * 25


@pytest.mark.parametrize("species", SPECIES)
def test_species_recipes_share_the_command_contract_for_adaptation(species):
    contracts = set()
    for behavior in BEHAVIORS:
        commands = DirectionCommandConfig(**_recipe(species, behavior)["commands"])
        contracts.add(
            (
                commands.speed_scale,
                commands.lateral_speed_scale,
                commands.yaw_rate_scale,
                commands.yaw_gain,
                commands.yaw_rate_max,
                commands.turn_slowdown,
                commands.minimum_turn_speed_fraction,
            )
        )
    assert len(contracts) == 1


@pytest.mark.parametrize("species", SPECIES)
def test_terrain_scale_fits_the_actual_animal_and_25_second_course(species):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "environments" / MODEL_PATHS[species]))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    # Include visual geoms as a conservative footprint; exclude floor and mocap target.
    animal = np.array(
        [body != 0 and model.body_mocapid[body] < 0 and model.body_rootid[body] != 0 for body in model.geom_bodyid]
    )
    radius = float(
        (np.linalg.norm(data.geom_xpos[animal, :2] - data.qpos[:2], axis=1) + model.geom_rbound[animal]).max()
    )
    recipe = _recipe(species, "combined_mixed_terrain")
    terrain = _terrain(recipe)
    frame_skip = 10 if species.startswith("compsognathus") else 5
    duration = recipe["env"]["max_episode_steps"] * frame_skip * model.opt.timestep
    assert duration == pytest.approx(25.0)
    assert terrain.apron_radius > radius + terrain.cell_diagonal
    assert terrain.extent > recipe["commands"]["cruise_speed"] * duration + radius
    assert 0.01 < terrain.feature_height / data.qpos[2] < 0.03
    assert terrain.feature_radius_min >= 3 * max(terrain.dx, terrain.dy)
    assert recipe["env"]["course_distance"] < recipe["commands"]["cruise_speed"] * duration


@pytest.mark.parametrize("species", SPECIES)
def test_mixed_recipe_repeats_exactly_but_varies_between_runs_and_episodes(species):
    config = _terrain(_recipe(species, "combined_mixed_terrain"))
    first = generate_terrain(config, run_seed=42, episode_index=0)
    repeated = generate_terrain(config, run_seed=42, episode_index=0)
    next_episode = generate_terrain(config, run_seed=42, episode_index=1)
    next_run = generate_terrain(config, run_seed=43, episode_index=0)
    np.testing.assert_array_equal(first.normalized_heights, repeated.normalized_heights)
    assert not np.array_equal(first.normalized_heights, next_episode.normalized_heights)
    assert not np.array_equal(first.normalized_heights, next_run.normalized_heights)
    assert any(feature.height > 0 for feature in first.features)
    assert any(feature.height < 0 for feature in first.features)
    assert first.height_at(0, 0) == pytest.approx(0, abs=1e-6)


# Every bad-recipe case sits under a valid [behavior] header: since consolidation PR-6 a recipe without one is
# refused for that reason alone, which would let every case below pass without testing its named defect.
_BEHAVIOR_HEADER = '[behavior]\nspecies = "trex"\nname = "bad"\nparent = "locomotion"\n'


@pytest.mark.parametrize(
    "content",
    [
        "[unknown]\nx=1\n",
        "[commands]\nswitch_inteval_s=1.0\n",
        "[terrain]\nroughnes_amplitude=0.01\n",
        "[ppo]\nlearn_rate=0.0001\n",
        "[env]\nmax_episode_step=1000\n",
        "[terrain]\nenabled=1\n",
        "timesteps=1.5\n",
        "timesteps=-1\n",
        "[ppo]\nent_coef=nan\n",
        "[ppo]\nent_coef=-0.1\n",
        "[ppo]\ntarget_kl=-0.1\n",
        "[ppo]\nlearning_rate=0.0\n",
        "[ppo]\nwarmup_timesteps=1.5\n",
        "[ppo]\nwarmup_clip_range=0.5\n",
    ],
)
def test_bad_recipe_refuses_before_environment_or_policy_loading(tmp_path, content):
    from environments.shared.train_behaviors import read_recipe

    path = tmp_path / "bad.toml"
    path.write_text(_BEHAVIOR_HEADER + content)
    with pytest.raises((ValueError, TypeError)):
        read_recipe(path)


@pytest.mark.parametrize(
    "content,refused",
    [
        ('[env]\ncommand_mode = "heading"\n', "command_mode"),
        ("[env]\ncommand_config = { cruise_speed = 0.5 }\n", "command_config"),
        ("[env]\ndrift_penalty_weight = 0.25\n", None),
    ],
)
def test_env_takes_the_species_signature_but_not_the_command_keys(tmp_path, content, refused):
    """Consolidation PR-9: [env] takes the species constructor's keys (canonical_env_parameters is gone), here a
    T. rex reward weight BaseDinoEnv does not declare, but not command_mode or command_config: [commands] states
    the command task."""
    from environments.shared.train_behaviors import read_recipe

    path = tmp_path / "recipe.toml"
    path.write_text(_BEHAVIOR_HEADER + content)
    if refused is None:
        assert read_recipe(path)[3]["drift_penalty_weight"] == 0.25
    else:
        with pytest.raises(ValueError, match=rf"Unknown env fields: \['{refused}'\]"):
            read_recipe(path)


@pytest.mark.parametrize("key", ["command_mode", "command_config"])
def test_create_behavior_env_refuses_the_command_kwargs_by_name(key):
    """The behavior env runs "heading_and_speed" on its commands; either kwarg is refused by name, not as a duplicate."""
    from environments.shared.train_behaviors import create_behavior_env

    with pytest.raises(ValueError, match=rf"\['{key}'\]: a behavior env runs command_mode 'heading_and_speed'"):
        create_behavior_env("trex", commands=DirectionCommandConfig(), terrain=None, run_seed=0, **{key: None})


def _sampler(**weights):
    weights = {**dict.fromkeys(TERRAIN_FAMILIES, 0), **weights}
    return "[terrain_sampler]\n" + "".join(f"{family}={weight}\n" for family, weight in weights.items())


_TERRAIN = "[terrain]\nenabled=true\n"


@pytest.mark.parametrize(
    "content,message",
    [
        ("[env]\nflat_probability=0.25\n", "env.flat_probability is retired: \\[terrain_sampler\\] states"),
        (_TERRAIN + _sampler(flat=1, sloped=3) + "[env]\nflat_probability=0.0\n", "flat_probability is retired"),
        ("[env]\nterrain_sampler=1\n", "Unknown env fields: \\['terrain_sampler'\\]"),
        (_sampler(flat=1, sloped=3), "terrain_sampler requires enabled terrain"),
        ("[terrain]\nenabled=false\n" + _sampler(flat=1, sloped=3), "terrain_sampler requires enabled terrain"),
        (
            _TERRAIN + "[terrain_sampler]\nflat=1\nsloped=3\n",
            "state every family's episodes \\(0 disables one\\); missing \\['bumps', 'depressions', 'mixed', "
            "'terrain_contact'\\]",
        ),
        (_TERRAIN + _sampler(flat=1, sloped=3) + "bumpps=1\n", "Unknown terrain_sampler fields: \\['bumpps'\\]"),
        (_TERRAIN + _sampler(flat=-1, sloped=3), "weight flat must be a nonnegative integer"),
        (_TERRAIN + _sampler(), "total weight"),
        (_TERRAIN + _sampler(flat=1, sloped=3) + "[terrain_sampler.extra]\n", "Unknown terrain_sampler fields"),
    ],
)
def test_terrain_recipe_refusals_name_their_fix(tmp_path, content, message):
    from environments.shared.train_behaviors import read_recipe

    path = tmp_path / "bad.toml"
    path.write_text(_BEHAVIOR_HEADER + content)
    with pytest.raises(ValueError, match=message):
        read_recipe(path)


@pytest.mark.parametrize("mode, own", [("flat", "terrain_contact"), ("gentle", "sloped")])
def test_terrain_sampler_recipe_injects_the_sampler_and_admits_terrain_contact(tmp_path, mode, own):
    from environments.shared.train_behaviors import read_recipe

    path = tmp_path / "ok.toml"
    path.write_text(_BEHAVIOR_HEADER + _TERRAIN + f'mode="{mode}"\n' + _sampler(flat=1, **{own: 3}))
    _, _, terrain, kwargs = read_recipe(path)
    assert terrain.mode == mode
    assert kwargs["terrain_sampler"] == TerrainSamplerConfig(
        **{**dict.fromkeys(TERRAIN_FAMILIES, 0), "flat": 1, own: 3}
    )


def test_terrain_without_a_sampler_stays_one_fixed_surface(tmp_path):
    """As before PR-8 for a recipe without flat_probability: every episode on the [terrain] surface."""
    from environments.shared.train_behaviors import read_recipe

    path = tmp_path / "fixed.toml"
    path.write_text(_BEHAVIOR_HEADER + _TERRAIN)
    _, _, terrain, kwargs = read_recipe(path)
    assert terrain is not None and "terrain_sampler" not in kwargs and "flat_probability" not in kwargs


@pytest.mark.parametrize(
    "content,message",
    [
        ("", "behavior requires species, name and parent"),
        ("[commands]\ncruise_speed = 1.0\n", "behavior requires species, name and parent"),
        ('[pilot]\nname = "T-Rex F1"\ntimesteps = 100\n', "behavior requires species, name and parent"),
        (_BEHAVIOR_HEADER + "[pilot]\ntimesteps = 100\n", "Unknown recipe sections"),
    ],
)
def test_recipe_without_behavior_table_is_refused_instead_of_defaulting_to_trex(tmp_path, content, message):
    """Consolidation PR-6: the [pilot] dialect is gone; a recipe names its species in [behavior] or is refused."""
    from environments.shared.train_behaviors import read_recipe

    path = tmp_path / "recipe.toml"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_recipe(path)
