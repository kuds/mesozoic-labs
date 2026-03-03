"""Tests for the CLI entry point (cli.py)."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.cli import _apply_overrides, _cast_value, main


class TestCastValue:
    """Test _cast_value auto-casting (covers cli.py's copy of the function)."""

    def test_int_string(self):
        assert _cast_value("42") == 42

    def test_float_string(self):
        assert _cast_value("3.14") == pytest.approx(3.14)

    def test_float_encoded_int(self):
        assert _cast_value("128.0") == 128
        assert isinstance(_cast_value("128.0"), int)

    def test_plain_string(self):
        assert _cast_value("hello") == "hello"

    def test_scientific_notation(self):
        assert _cast_value("1e-4") == pytest.approx(1e-4)


class TestApplyOverrides:
    """Test _apply_overrides for both global and per-stage overrides."""

    @pytest.fixture()
    def configs(self):
        return {
            1: {
                "env_kwargs": {"alive_bonus": 2.0, "forward_vel_weight": 0.0},
                "ppo_kwargs": {"learning_rate": 1e-3, "batch_size": 256},
            },
            2: {
                "env_kwargs": {"alive_bonus": 1.5, "forward_vel_weight": 1.0},
                "ppo_kwargs": {"learning_rate": 5e-4, "batch_size": 128},
            },
        }

    def test_none_overrides(self, configs):
        _apply_overrides(configs, None)
        assert configs[1]["env_kwargs"]["alive_bonus"] == 2.0

    def test_empty_overrides(self, configs):
        _apply_overrides(configs, [])
        assert configs[1]["env_kwargs"]["alive_bonus"] == 2.0

    def test_global_override_applies_to_all_stages(self, configs):
        _apply_overrides(configs, ["ppo.learning_rate=1e-4"])
        assert configs[1]["ppo_kwargs"]["learning_rate"] == pytest.approx(1e-4)
        assert configs[2]["ppo_kwargs"]["learning_rate"] == pytest.approx(1e-4)

    def test_stage_scoped_override(self, configs):
        _apply_overrides(configs, ["2.ppo.learning_rate=5e-5"])
        assert configs[1]["ppo_kwargs"]["learning_rate"] == pytest.approx(1e-3)
        assert configs[2]["ppo_kwargs"]["learning_rate"] == pytest.approx(5e-5)

    def test_env_section_override(self, configs):
        _apply_overrides(configs, ["env.alive_bonus=5.0"])
        assert configs[1]["env_kwargs"]["alive_bonus"] == pytest.approx(5.0)
        assert configs[2]["env_kwargs"]["alive_bonus"] == pytest.approx(5.0)


class TestMainDispatch:
    """Test main() argument parsing and dispatch."""

    @pytest.fixture
    def species_cfg(self):
        cfg = MagicMock()
        cfg.species = "velociraptor"
        cfg.stage_descriptions = "1=balance, 2=locomotion, 3=strike"
        return cfg

    def test_train_command(self, species_cfg):
        """main() with 'train' should call train()."""
        mock_train = MagicMock()
        mock_load = MagicMock(
            return_value={
                1: {"env_kwargs": {}, "ppo_kwargs": {}},
                2: {"env_kwargs": {}, "ppo_kwargs": {}},
                3: {"env_kwargs": {}, "ppo_kwargs": {}},
            }
        )
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train", mock_train),
            patch("sys.argv", ["prog", "train", "--stage", "1", "--timesteps", "1000"]),
        ):
            main(species_cfg)
            mock_train.assert_called_once()

    def test_curriculum_command(self, species_cfg):
        """main() with 'curriculum' should call train_curriculum()."""
        mock_curriculum = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train_curriculum", mock_curriculum),
            patch("sys.argv", ["prog", "curriculum"]),
        ):
            main(species_cfg)
            mock_curriculum.assert_called_once()

    def test_eval_command(self, species_cfg):
        """main() with 'eval' should call evaluate()."""
        mock_eval = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.evaluation.evaluate", mock_eval),
            patch("sys.argv", ["prog", "eval", "/tmp/model.zip", "--episodes", "5"]),
        ):
            main(species_cfg)
            mock_eval.assert_called_once()

    def test_no_command_defaults_to_train(self, species_cfg):
        """main() with no subcommand should default to train."""
        mock_train = MagicMock()
        mock_load = MagicMock(
            return_value={
                1: {"env_kwargs": {}, "ppo_kwargs": {}},
                2: {"env_kwargs": {}, "ppo_kwargs": {}},
                3: {"env_kwargs": {}, "ppo_kwargs": {}},
            }
        )
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train", mock_train),
            patch("sys.argv", ["prog"]),
        ):
            main(species_cfg)
            mock_train.assert_called_once()

    def test_train_with_overrides(self, species_cfg):
        """main() should pass overrides through _apply_overrides."""
        mock_train = MagicMock()
        mock_load = MagicMock(
            return_value={
                1: {"env_kwargs": {}, "ppo_kwargs": {"learning_rate": 1e-3}},
                2: {"env_kwargs": {}, "ppo_kwargs": {"learning_rate": 5e-4}},
                3: {"env_kwargs": {}, "ppo_kwargs": {"learning_rate": 3e-4}},
            }
        )
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train", mock_train),
            patch("sys.argv", ["prog", "train", "--override", "ppo.learning_rate=1e-4"]),
        ):
            main(species_cfg)
            mock_train.assert_called_once()

    def test_eval_with_algorithm(self, species_cfg):
        """main() eval command should pass algorithm correctly."""
        mock_eval = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.evaluation.evaluate", mock_eval),
            patch("sys.argv", ["prog", "eval", "/tmp/model.zip", "--algorithm", "sac"]),
        ):
            main(species_cfg)
            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["algorithm"] == "sac"
