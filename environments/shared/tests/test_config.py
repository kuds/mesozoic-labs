"""Tests for the TOML config loader."""

import pytest

from environments.shared.config import (
    load_all_stages,
    load_stage_config,
)

SPECIES = ["velociraptor", "brachiosaurus", "trex"]


class TestLoadStageConfig:
    """Test loading individual stage configs."""

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_loads_successfully(self, species, stage):
        config = load_stage_config(species, stage)
        assert isinstance(config, dict)

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_has_required_keys(self, species, stage):
        config = load_stage_config(species, stage)
        assert "name" in config
        assert "description" in config
        assert "env_kwargs" in config
        assert "ppo_kwargs" in config
        assert "sac_kwargs" in config

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_env_kwargs_has_common_keys(self, species, stage):
        config = load_stage_config(species, stage)
        env_kw = config["env_kwargs"]
        assert "forward_vel_weight" in env_kw
        assert "alive_bonus" in env_kw
        assert "energy_penalty_weight" in env_kw
        assert "max_episode_steps" in env_kw

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_ppo_kwargs_has_common_keys(self, species, stage):
        config = load_stage_config(species, stage)
        ppo_kw = config["ppo_kwargs"]
        assert "learning_rate" in ppo_kw
        assert "batch_size" in ppo_kw
        assert "gamma" in ppo_kw

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_range_params_are_tuples(self, species, stage):
        """TOML lists should be converted to tuples for range parameters."""
        config = load_stage_config(species, stage)
        env_kw = config["env_kwargs"]
        for key, value in env_kw.items():
            if key.endswith("_range"):
                assert isinstance(value, tuple), f"{key} should be a tuple, got {type(value)}"

    def test_missing_species_raises(self):
        with pytest.raises(FileNotFoundError):
            load_stage_config("stegosaurus", 1)

    def test_explicit_config_path(self, tmp_path):
        toml_content = b"""
[stage]
name = "test"
description = "test stage"

[env]
forward_vel_weight = 1.0
alive_bonus = 0.5
energy_penalty_weight = 0.001
max_episode_steps = 100

[ppo]
learning_rate = 3e-4
batch_size = 64
gamma = 0.99
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_bytes(toml_content)

        config = load_stage_config("ignored", 1, config_path=str(config_file))
        assert config["name"] == "test"
        assert config["env_kwargs"]["forward_vel_weight"] == 1.0


class TestLoadAllStages:
    """Test loading all stages for a species."""

    @pytest.mark.parametrize("species", SPECIES)
    def test_returns_three_stages(self, species):
        stages = load_all_stages(species)
        assert set(stages.keys()) == {1, 2, 3}

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage1_is_balance(self, species):
        stages = load_all_stages(species)
        assert stages[1]["name"] == "balance"

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage1_no_forward_reward(self, species):
        """Stage 1 (balance) should have zero forward velocity weight."""
        stages = load_all_stages(species)
        assert stages[1]["env_kwargs"]["forward_vel_weight"] == 0.0

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage_progression_alive_bonus(self, species):
        """Alive bonus should decrease across stages (less reliance on survival)."""
        stages = load_all_stages(species)
        assert stages[1]["env_kwargs"]["alive_bonus"] > stages[2]["env_kwargs"]["alive_bonus"]
        assert stages[2]["env_kwargs"]["alive_bonus"] > stages[3]["env_kwargs"]["alive_bonus"]

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage_progression_learning_rate(self, species):
        """Learning rate should decrease across stages (finer tuning)."""
        stages = load_all_stages(species)
        assert stages[1]["ppo_kwargs"]["learning_rate"] > stages[2]["ppo_kwargs"]["learning_rate"]
        assert stages[2]["ppo_kwargs"]["learning_rate"] > stages[3]["ppo_kwargs"]["learning_rate"]


class TestCurriculumInvariants:
    """Test that curriculum configs maintain expected invariants."""

    def test_velociraptor_strike_bonus_only_stage3(self):
        stages = load_all_stages("velociraptor")
        assert stages[1]["env_kwargs"]["strike_bonus"] == 0.0
        assert stages[2]["env_kwargs"]["strike_bonus"] == 0.0
        assert stages[3]["env_kwargs"]["strike_bonus"] > 0.0

    def test_brachiosaurus_food_reach_only_stage3(self):
        stages = load_all_stages("brachiosaurus")
        assert stages[1]["env_kwargs"]["food_reach_bonus"] == 0.0
        assert stages[2]["env_kwargs"]["food_reach_bonus"] == 0.0
        assert stages[3]["env_kwargs"]["food_reach_bonus"] > 0.0

    def test_trex_bite_bonus_only_stage3(self):
        stages = load_all_stages("trex")
        assert stages[1]["env_kwargs"]["bite_bonus"] == 0.0
        assert stages[2]["env_kwargs"]["bite_bonus"] == 0.0
        assert stages[3]["env_kwargs"]["bite_bonus"] > 0.0

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage1_shorter_episodes(self, species):
        """Stage 1 should have shorter episodes than later stages."""
        stages = load_all_stages(species)
        assert stages[1]["env_kwargs"]["max_episode_steps"] < stages[2]["env_kwargs"]["max_episode_steps"]
