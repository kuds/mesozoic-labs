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

    def _run_curriculum(self, species_cfg, argv):
        """Dispatch ``curriculum`` with the trainer replaced; returns the mock."""
        mock_curriculum = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train_curriculum", mock_curriculum),
            patch("sys.argv", ["prog", "curriculum", *argv]),
        ):
            main(species_cfg)
        return mock_curriculum

    def test_curriculum_forwards_retrain_from_with_its_trunk(self, species_cfg, tmp_path):
        """Decision D-A19: ``--retrain-from`` resolves like ``--stage`` (id or legacy number) and
        reaches train_curriculum beside the trunk it applies to."""
        mock = self._run_curriculum(species_cfg, ["--trunk-from", str(tmp_path), "--retrain-from", "locomotion"])
        kwargs = mock.call_args.kwargs
        assert kwargs["trunk_from"] == str(tmp_path) and kwargs["retrain_from"] == "locomotion"

        mock = self._run_curriculum(species_cfg, ["--trunk-from", str(tmp_path), "--retrain-from", "2"])
        assert mock.call_args.kwargs["retrain_from"] == 2

        mock = self._run_curriculum(species_cfg, ["--trunk-from", str(tmp_path)])
        assert mock.call_args.kwargs["retrain_from"] is None

    def test_curriculum_accepts_auto_as_the_trunk(self, species_cfg):
        """Decision D-A25: the literal ``auto`` is not a directory and reaches train_curriculum verbatim,
        with ``--retrain-from`` allowed beside it."""
        mock = self._run_curriculum(species_cfg, ["--trunk-from", "auto", "--retrain-from", "locomotion"])
        assert mock.call_args.kwargs["trunk_from"] == "auto"
        assert mock.call_args.kwargs["retrain_from"] == "locomotion"

    def test_retrain_from_without_a_trunk_is_a_usage_error(self, species_cfg, capsys):
        """Without --trunk-from every stage is trained here already; the knob is refused rather
        than silently ignored."""
        with pytest.raises(SystemExit) as excinfo:
            self._run_curriculum(species_cfg, ["--retrain-from", "locomotion"])
        assert excinfo.value.code == 2
        assert "--retrain-from 'locomotion' requires --trunk-from" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("ref", "expected"),
        [("recovery", "non-advancing stage 'recovery'"), ("no_such_node", "has no stage 'no_such_node'")],
    )
    def test_retrain_from_must_name_an_advancing_stage(self, species_cfg, tmp_path, capsys, ref, expected):
        """A non-advancing id (trex's recovery) or an unknown one is a parser.error naming the
        advancing ids, before train_curriculum is reached."""
        species_cfg.species = "trex"
        with pytest.raises(SystemExit) as excinfo:
            self._run_curriculum(species_cfg, ["--trunk-from", str(tmp_path), "--retrain-from", ref])
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert expected in err and "['stance', 'locomotion', 'behavior']" in err

    def test_curriculum_forwards_target_as_a_label_an_id_or_a_legacy_number(self, species_cfg, tmp_path):
        """Decision D-A24: ``--target`` resolves like ``--stage`` (digits are a legacy number) and
        reaches train_curriculum as ``target=``; absent, it is None (the whole ladder)."""
        assert self._run_curriculum(species_cfg, ["--target", "walk"]).call_args.kwargs["target"] == "walk"
        assert self._run_curriculum(species_cfg, ["--target", "locomotion"]).call_args.kwargs["target"] == "locomotion"
        assert self._run_curriculum(species_cfg, ["--target", "2"]).call_args.kwargs["target"] == 2
        assert self._run_curriculum(species_cfg, []).call_args.kwargs["target"] is None
        # Beside a trunk and a retrain-from on its chain, all three are forwarded.
        mock = self._run_curriculum(
            species_cfg, ["--trunk-from", str(tmp_path), "--target", "walk", "--retrain-from", "locomotion"]
        )
        kwargs = mock.call_args.kwargs
        assert (kwargs["trunk_from"], kwargs["target"], kwargs["retrain_from"]) == (str(tmp_path), "walk", "locomotion")

    @pytest.mark.parametrize(
        ("species", "target", "expected"),
        [
            (
                "trex",
                "stand",
                "--target 'stand' resolves to 'recovery', whose chain ['stance', 'recovery'] runs through the "
                "non-advancing stage(s) ['recovery']; the command-line curriculum walks only the advancing stages "
                "['stance', 'locomotion', 'behavior'] (the CurriculumManager is integer-keyed, decision D-A7). "
                "Train that chain through the notebook's BEHAVIOR knob (BEHAVIOR = 'stand')",
            ),
            (
                "velociraptor",
                "fly",
                "--target 'fly' does not name a behavior of velociraptor: velociraptor has no behavior 'fly'; "
                "recipe labels: ['stand', 'walk', 'hunt'], deliverable ids: ['stance', 'locomotion', 'behavior']",
            ),
        ],
    )
    def test_target_must_resolve_to_a_chain_of_advancing_stages(self, species_cfg, target, species, expected, capsys):
        """A chain through a non-advancing node (trex's stand runs through recovery) is a parser.error
        naming the notebook's BEHAVIOR knob; an unknown label lists the labels — both before
        train_curriculum is reached."""
        species_cfg.species = species
        with pytest.raises(SystemExit) as excinfo:
            self._run_curriculum(species_cfg, ["--target", target])
        assert excinfo.value.code == 2
        assert expected in capsys.readouterr().err

    def test_a_target_whose_chain_skips_a_ladder_stage_is_a_usage_error(
        self, species_cfg, tmp_path, monkeypatch, capsys
    ):
        """The chain must be a prefix of the advancing ladder: with velociraptor's behavior edge rewired
        onto stance (a manifest the loader accepts), ``--target hunt`` is a parser.error naming the
        skipped locomotion stage, before train_curriculum is reached."""
        import shutil

        from environments.shared import stage_manifest

        configs = tmp_path / "configs"
        shutil.copytree(stage_manifest._CONFIGS_DIR / "velociraptor", configs / "velociraptor")
        manifest_path = configs / "velociraptor" / "stages.toml"
        text = manifest_path.read_text(encoding="utf-8")
        assert text.count('warm_start_from = "locomotion"') == 1
        manifest_path.write_text(text.replace('warm_start_from = "locomotion"', 'warm_start_from = "stance"'))
        monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", configs)

        with pytest.raises(SystemExit) as excinfo:
            self._run_curriculum(species_cfg, ["--target", "hunt"])
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert (
            "--target 'hunt' resolves to 'behavior', whose chain ['stance', 'behavior'] skips the advancing stage(s) ['locomotion']"
            in err
        )
        assert "Train that chain through the notebook's BEHAVIOR knob (BEHAVIOR = 'hunt')" in err

    def test_target_help_says_how_each_form_resolves(self, species_cfg, capsys):
        """Labels and ids resolve as the notebook's BEHAVIOR knob does; a legacy number resolves as
        ``--stage`` does (the notebook accepts no number) — the help must not conflate the two."""
        with pytest.raises(SystemExit) as excinfo, patch("sys.argv", ["prog", "curriculum", "--help"]):
            main(species_cfg)
        assert excinfo.value.code == 0
        help_text = " ".join(capsys.readouterr().out.split())
        assert (
            "resolved as the notebook's BEHAVIOR knob resolves them, or a legacy number (2), resolved as --stage resolves it"
            in help_text
        )
        assert "gate_verdict.json passed under the current gate configuration" in help_text
        assert "the command line and the notebook's TRUNK_FROM both follow those records" in help_text

    def test_retrain_from_outside_the_targets_chain_is_a_usage_error(self, species_cfg, tmp_path, capsys):
        """``--retrain-from behavior`` names an advancing node a ``--target walk`` run never walks."""
        with pytest.raises(SystemExit) as excinfo:
            self._run_curriculum(
                species_cfg, ["--trunk-from", str(tmp_path), "--target", "walk", "--retrain-from", "behavior"]
            )
        assert excinfo.value.code == 2
        assert (
            "--retrain-from 'behavior' names 'behavior', which is not on the chain this run walks to its target "
            "'locomotion': ['stance', 'locomotion']; it must name one of those"
        ) in capsys.readouterr().err

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

    def test_label_reaches_train_and_train_curriculum(self, species_cfg, tmp_path):
        """Decision D-A21: ``--label TEXT`` on both commands is forwarded verbatim as ``label=``;
        without it the trainers receive ``None`` (nothing recorded)."""
        mock_train = MagicMock()
        mock_load = MagicMock(return_value={n: {"env_kwargs": {}, "ppo_kwargs": {}} for n in (1, 2, 3)})
        for argv, expected in ((["--label", "lr-sweep-a"], "lr-sweep-a"), ([], None)):
            with (
                patch("environments.shared.config.load_all_stages", mock_load),
                patch("environments.shared.train_base.train", mock_train),
                patch("sys.argv", ["prog", "train", "--stage", "1", "--timesteps", "1000", *argv]),
            ):
                main(species_cfg)
            assert mock_train.call_args.kwargs["label"] == expected

        mock = self._run_curriculum(species_cfg, ["--label", "lr-sweep-a"])
        assert mock.call_args.kwargs["label"] == "lr-sweep-a"
        assert self._run_curriculum(species_cfg, []).call_args.kwargs["label"] is None

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
