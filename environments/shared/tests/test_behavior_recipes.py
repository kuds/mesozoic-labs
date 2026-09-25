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
    if behavior in SAMPLER_BEHAVIORS:
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
        assert recipe["env"]["flat_probability"] == (0.0 if behavior in SAMPLER_BEHAVIORS else 0.25)
        assert terrain.episode_variation > 0


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize(
    "behavior,diagnostic",
    [("difficult_terrain", "mixed_terrain"), ("follow_direction_difficult_terrain", "combined_mixed_terrain")],
)
def test_unified_terrain_recipes_cover_all_families_and_preserve_species_profiles(species, behavior, diagnostic):
    recipe = _recipe(species, behavior)
    focused_recipe = _recipe(species, diagnostic)
    assert recipe["terrain_sampler"] == {"flat": 1, "sloped": 1, "bumps": 1, "depressions": 1, "mixed": 1}
    assert recipe["behavior"]["name"] == behavior
    assert recipe["behavior"]["timesteps"] == 3_000_000
    assert recipe["commands"] == focused_recipe["commands"]
    assert recipe["terrain"] == focused_recipe["terrain"]
    assert recipe["env"] == {**focused_recipe["env"], "flat_probability": 0.0}


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
        "[terrain_sampler]\nflat=1\n",
        '[terrain]\nenabled=true\nmode="flat"\n[terrain_sampler]\nflat=1\n',
        '[terrain]\nenabled=true\nmode="gentle"\n[terrain_sampler]\nbumpps=1\n',
        '[terrain]\nenabled=true\nmode="gentle"\n[terrain_sampler]\nflat=-1\n',
        '[terrain]\nenabled=true\nmode="gentle"\n[terrain_sampler]\nflat=1\n[env]\nflat_probability=0.25\n',
    ],
)
def test_bad_recipe_refuses_before_environment_or_policy_loading(tmp_path, content):
    from environments.shared.train_behaviors import read_recipe

    path = tmp_path / "bad.toml"
    path.write_text(_BEHAVIOR_HEADER + content)
    with pytest.raises((ValueError, TypeError)):
        read_recipe(path)


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
