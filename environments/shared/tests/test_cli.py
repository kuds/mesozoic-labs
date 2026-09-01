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

    def test_train_forwards_explicit_legacy_plant_override(self, species_cfg):
        mock_train = MagicMock()
        mock_load = MagicMock(
            return_value={
                1: {"env_kwargs": {}, "ppo_kwargs": {}, "curriculum_kwargs": {"timesteps": 100}},
                2: {},
                3: {},
            }
        )
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train", mock_train),
            patch(
                "sys.argv",
                ["prog", "train", "--stage", "1", "--load", "/tmp/legacy.zip", "--allow-legacy-plant"],
            ),
        ):
            main(species_cfg)

        assert mock_train.call_args.kwargs["allow_legacy_plant"] is True

    def test_eval_forwards_explicit_legacy_plant_override(self, species_cfg):
        mock_eval = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.evaluation.evaluate", mock_eval),
            patch("sys.argv", ["prog", "eval", "/tmp/legacy.zip", "--allow-legacy-plant"]),
        ):
            main(species_cfg)

        assert mock_eval.call_args.kwargs["allow_legacy_plant"] is True


class TestApplyOverridesStageScoping:
    """Semantic stage ids resolve; unknown stages/sections raise (TC10/CI6).

    A typo'd override used to train the full multi-hour stage budget at
    unmodified hyperparameters with nothing in the log, and the semantic
    ``recovery`` stage — the active training focus — could not be targeted
    by ``--override`` at all (bare KeyError at launch).
    """

    def _configs(self):
        return {
            1: {"env_kwargs": {}, "ppo_kwargs": {}, "curriculum_kwargs": {}},
            "recovery": {"env_kwargs": {}, "ppo_kwargs": {}, "curriculum_kwargs": {}},
            2: {"env_kwargs": {}, "ppo_kwargs": {}, "curriculum_kwargs": {}},
        }

    def test_semantic_config_key_scopes_to_that_stage_only(self):
        configs = self._configs()
        _apply_overrides(configs, ["recovery.env.push_interval_steps=250"])
        assert configs["recovery"]["env_kwargs"] == {"push_interval_steps": 250}
        assert configs[1]["env_kwargs"] == {}
        assert configs[2]["env_kwargs"] == {}

    def test_manifest_stage_id_resolves_through_the_species_manifest(self):
        configs = self._configs()
        _apply_overrides(configs, ["locomotion.ppo.learning_rate=0.0002"], "trex")
        assert configs[2]["ppo_kwargs"] == {"learning_rate": 0.0002}
        assert configs[1]["ppo_kwargs"] == {}

    def test_unknown_numeric_stage_raises_instead_of_silently_no_opping(self):
        with pytest.raises(ValueError, match="unknown stage '7'"):
            _apply_overrides(self._configs(), ["7.ppo.learning_rate=0.0001"])

    def test_typoed_semantic_stage_gets_the_stage_error_with_choices(self):
        # 'recvery.env.x' must not be misattributed to a config section named
        # 'recvery' — the middle token IS a real section, so the head token
        # was meant as a stage and the error must list the available stages.
        with pytest.raises(ValueError, match="unknown stage 'recvery'.*available"):
            _apply_overrides(self._configs(), ["recvery.env.alive_bonus=1.0"], "trex")

    def test_unknown_section_raises_instead_of_bare_keyerror(self):
        with pytest.raises(ValueError, match="unknown config section"):
            _apply_overrides(self._configs(), ["recovery.badsection.x=1"])

    def test_unknown_all_stages_section_raises(self):
        with pytest.raises(ValueError, match="unknown config section 'environmnt'"):
            _apply_overrides(self._configs(), ["environmnt.x=1"])

    def test_all_stages_form_still_applies_everywhere(self):
        configs = self._configs()
        _apply_overrides(configs, ["env.alive_bonus=0.5"])
        assert all(configs[key]["env_kwargs"] == {"alive_bonus": 0.5} for key in configs)
