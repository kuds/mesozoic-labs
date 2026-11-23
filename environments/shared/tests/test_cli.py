"""Tests for the CLI entry point (cli.py)."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.cli import _apply_overrides, _cast_value, main
from environments.shared.model_id import ModelId

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")


class TestCastValue:
    """An override value is parsed as the TOML value it spells."""

    def test_int_string(self):
        assert _cast_value("42") == 42
        assert isinstance(_cast_value("42"), int)

    def test_float_string(self):
        assert _cast_value("3.14") == pytest.approx(3.14)

    @pytest.mark.parametrize("raw", ["1.0", "128.0", "0.0", "-2.0"])
    def test_a_float_with_an_integral_value_stays_a_float(self, raw):
        """TOML's ``x = 1.0`` is a float, so ``--override ppo.x=1.0`` must be one too: the int this used to
        return gave the stage a different hyperparameters_sha256 than the TOML's own value."""
        value = _cast_value(raw)
        assert isinstance(value, float) and value == float(raw)

    def test_plain_string(self):
        assert _cast_value("hello") == "hello"

    def test_a_bare_word_that_is_not_toml_stays_the_raw_string(self):
        assert _cast_value("MlpPolicy") == "MlpPolicy"
        assert _cast_value("") == ""

    def test_scientific_notation(self):
        assert _cast_value("1e-4") == pytest.approx(1e-4)

    @pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False), ("True", True), ("FALSE", False)])
    def test_booleans_in_any_case(self, raw, expected):
        assert _cast_value(raw) is expected

    @pytest.mark.parametrize(("raw", "expected"), [("'MlpPolicy'", "MlpPolicy"), ('"a b"', "a b"), ("'1.0'", "1.0")])
    def test_quoted_strings_are_strings(self, raw, expected):
        assert _cast_value(raw) == expected

    def test_arrays_and_inline_tables(self):
        assert _cast_value("[512, 256]") == [512, 256]
        assert _cast_value("[1.0, 2]") == [1.0, 2]
        assert _cast_value("{pi = [64, 64], vf = [64]}") == {"pi": [64, 64], "vf": [64]}

    @pytest.mark.parametrize(("raw", "expected"), [(".5", 0.5), ("007", 7), ("5.", 5.0)])
    def test_a_number_toml_refuses_is_still_a_number(self, raw, expected):
        value = _cast_value(raw)
        assert value == expected and type(value) is type(expected)

    def test_an_override_types_its_value_as_the_stage_toml_does(self):
        """``--override`` of a value with the TOML's own text leaves the stage's hyperparameters_sha256 unchanged."""
        from environments.shared.config import hyperparameters_sha256, load_all_stages

        configs = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)
        stance = configs[1]["ppo_kwargs"]
        # The committed values these overrides restate: a float with an integral value, and a nested one.
        assert stance["ent_coef_end"] == 0.0 and isinstance(stance["ent_coef_end"], float)
        assert stance["policy_kwargs"]["log_std_init"] == -2.30
        digest = hyperparameters_sha256(configs[1], "ppo")

        _apply_overrides(configs, ["1.ppo.ent_coef_end=0.0", "1.ppo.policy_kwargs.log_std_init=-2.30"])
        assert hyperparameters_sha256(configs[1], "ppo") == digest

        # The int the old cast produced is a different recipe.
        _apply_overrides(configs, ["1.ppo.ent_coef_end=0"])
        assert hyperparameters_sha256(configs[1], "ppo") != digest


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
        cfg.model_id = VELOCIRAPTOR_MONGOLIENSIS
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
        species_cfg.model_id = TYRANNOSAURUS_REX
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
        ("model_id", "target", "expected"),
        [
            (
                TYRANNOSAURUS_REX,
                "recover",
                "--target 'recover' resolves to 'recovery', whose chain ['stance', 'recovery'] runs through the "
                "non-advancing stage(s) ['recovery']; the command-line curriculum walks only the advancing stages "
                "['stance', 'locomotion', 'behavior'] (the CurriculumManager is integer-keyed, decision D-A7). "
                "Train that chain through the notebook's BEHAVIOR knob (BEHAVIOR = 'recover')",
            ),
            (
                VELOCIRAPTOR_MONGOLIENSIS,
                "fly",
                "--target 'fly' does not name a behavior of velociraptor_mongoliensis/anatomical: "
                "velociraptor_mongoliensis/anatomical has no behavior 'fly'; "
                "recipe labels: ['stance', 'walk', 'hunt'], deliverable ids: ['stance', 'locomotion', 'behavior']",
            ),
            (
                VELOCIRAPTOR_MONGOLIENSIS,
                "recover",
                "--target 'recover' does not name a behavior of velociraptor_mongoliensis/anatomical: "
                "velociraptor_mongoliensis/anatomical has no behavior 'recover'; "
                "recipe labels: ['stance', 'walk', 'hunt'], deliverable ids: ['stance', 'locomotion', 'behavior']",
            ),
        ],
        ids=[
            "tyrannosaurus_rex-anatomical-recover",
            "velociraptor_mongoliensis-anatomical-fly",
            "velociraptor_mongoliensis-anatomical-recover",
        ],
    )
    def test_target_must_resolve_to_a_chain_of_advancing_stages(self, species_cfg, target, model_id, expected, capsys):
        """A chain through a non-advancing node (trex's recover runs through recovery) is a parser.error
        naming the notebook's BEHAVIOR knob; an unknown label lists the labels — both before
        train_curriculum is reached."""
        species_cfg.model_id = model_id
        with pytest.raises(SystemExit) as excinfo:
            self._run_curriculum(species_cfg, ["--target", target])
        assert excinfo.value.code == 2
        assert expected in capsys.readouterr().err

    @pytest.mark.parametrize(
        "model_id",
        [
            TYRANNOSAURUS_REX,
            ModelId("compsognathus_longipes", "anatomical"),
            ModelId("compsognathus_longipes", "robot"),
            VELOCIRAPTOR_MONGOLIENSIS,
            ModelId("brachiosaurus_altithorax", "anatomical"),
            ModelId("dibothrosuchus_elaphros", "anatomical"),
        ],
        ids=lambda m: f"{m.species}-{m.embodiment}",
    )
    def test_target_stance_is_the_stance_stage_on_every_model(self, species_cfg, model_id):
        """Decisions D-D35 and D-D38: ``stance``, the stance stage's id and its recipe label, is the
        stance stage alone on every model, so ``--target stance`` is a stance-only run (D-D35 made the
        label, then ``stand``, the stance stage on the three recovery models too; ``recover`` took the
        chain through recovery and its refusal)."""
        from environments.shared.stage_manifest import load_stage_manifest
        from environments.shared.train_base import _resolve_curriculum_target

        species_cfg.model_id = model_id
        assert self._run_curriculum(species_cfg, ["--target", "stance"]).call_args.kwargs["target"] == "stance"
        entry, chain = _resolve_curriculum_target(load_stage_manifest(model_id), "stance", flag="--target")
        assert entry.id == "stance" and [node.id for node in chain] == ["stance"]

    @pytest.mark.parametrize(
        "model_id",
        [
            TYRANNOSAURUS_REX,
            ModelId("compsognathus_longipes", "anatomical"),
            ModelId("compsognathus_longipes", "robot"),
            VELOCIRAPTOR_MONGOLIENSIS,
            ModelId("brachiosaurus_altithorax", "anatomical"),
            ModelId("dibothrosuchus_elaphros", "anatomical"),
        ],
        ids=lambda m: f"{m.species}-{m.embodiment}",
    )
    def test_target_stand_is_refused_naming_stance(self, species_cfg, model_id, capsys):
        """Decision D-D38: ``stand``, the stance label until then, is retired and does not resolve, so
        ``--target stand`` is a usage error pointing at ``stance`` on every model, before
        train_curriculum is reached."""
        species_cfg.model_id = model_id
        with pytest.raises(SystemExit) as excinfo:
            self._run_curriculum(species_cfg, ["--target", "stand"])
        assert excinfo.value.code == 2
        assert (
            f"--target 'stand' does not name a behavior of {model_id}: {model_id} has no behavior 'stand': "
            "'stand' was renamed to 'stance' (decision D-D38); use 'stance'"
        ) in capsys.readouterr().err

    def test_a_target_whose_chain_skips_a_ladder_stage_is_a_usage_error(
        self, species_cfg, tmp_path, monkeypatch, capsys
    ):
        """The chain must be a prefix of the advancing ladder: with velociraptor's behavior edge rewired
        onto stance (a manifest the loader accepts), ``--target hunt`` is a parser.error naming the
        skipped locomotion stage, before train_curriculum is reached."""
        import shutil

        from environments.shared import stage_manifest

        configs = tmp_path / "configs"
        shutil.copytree(
            VELOCIRAPTOR_MONGOLIENSIS.path(stage_manifest._CONFIGS_DIR), VELOCIRAPTOR_MONGOLIENSIS.path(configs)
        )
        manifest_path = VELOCIRAPTOR_MONGOLIENSIS.path(configs) / "stages.toml"
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

    @pytest.mark.parametrize("flag", ["--gcs-bucket", "--gcs-project"])
    def test_retired_gcs_flags_are_usage_errors(self, species_cfg, capsys, flag):
        """D-D17 retired GCS upload (cleanup PR-A2): the flags are gone, so argparse refuses them before
        train_curriculum is reached, and the curriculum is called with no gcs_* keyword."""
        with pytest.raises(SystemExit) as excinfo:
            self._run_curriculum(species_cfg, [flag, "my-bucket"])
        assert excinfo.value.code == 2
        assert f"unrecognized arguments: {flag} my-bucket" in capsys.readouterr().err
        kwargs = self._run_curriculum(species_cfg, []).call_args.kwargs
        assert not {"gcs_bucket", "gcs_project"} & set(kwargs)

    @pytest.mark.parametrize("command", ["train", "curriculum"])
    @pytest.mark.parametrize("raw", ["0", "-50000", "1e4"])
    def test_a_non_positive_eval_freq_is_a_usage_error(self, species_cfg, capsys, command, raw):
        """``--eval-freq 0`` used to reach the trainer as an evaluation interval EvalCallback never reaches."""
        mock_load = MagicMock(return_value={n: {"env_kwargs": {}, "ppo_kwargs": {}} for n in (1, 2, 3)})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train") as mock_train,
            patch("environments.shared.train_base.train_curriculum") as mock_curriculum,
            patch("sys.argv", ["prog", command, f"--eval-freq={raw}"]),
            pytest.raises(SystemExit) as excinfo,
        ):
            main(species_cfg)
        assert excinfo.value.code == 2
        assert "argument --eval-freq: expected a positive number of steps, got" in capsys.readouterr().err
        mock_train.assert_not_called()
        mock_curriculum.assert_not_called()

    def test_a_positive_eval_freq_reaches_both_trainers(self, species_cfg):
        mock_train = MagicMock()
        mock_load = MagicMock(return_value={n: {"env_kwargs": {}, "ppo_kwargs": {}} for n in (1, 2, 3)})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train", mock_train),
            patch("sys.argv", ["prog", "train", "--timesteps", "1000", "--eval-freq", "1"]),
        ):
            main(species_cfg)
        assert mock_train.call_args.kwargs["eval_freq"] == 1
        assert self._run_curriculum(species_cfg, ["--eval-freq", "2048"]).call_args.kwargs["eval_freq"] == 2048
        assert self._run_curriculum(species_cfg, []).call_args.kwargs["eval_freq"] == 50000

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
        _apply_overrides(configs, ["locomotion.ppo.learning_rate=0.0002"], TYRANNOSAURUS_REX)
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
            _apply_overrides(self._configs(), ["recvery.env.alive_bonus=1.0"], TYRANNOSAURUS_REX)

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

    def test_a_key_with_no_section_raises(self):
        with pytest.raises(ValueError, match="'alive_bonus' names no key"):
            _apply_overrides(self._configs(), ["alive_bonus=1.0"])


class TestNestedOverrides:
    """A dotted key deep-sets into a nested table, in the all-stages and the STAGE forms.

    ``env.a.b=v`` used to fall to ``key.partition('.')`` and write the flat literal key ``'a.b'``
    into env_kwargs, which failed only later, at env construction.  The committed stage TOMLs nest
    tables only as ``[ppo.policy_kwargs]`` / ``[sac.policy_kwargs]`` (their ``[env]`` is flat), so a
    nested key may enter a table the stage config already has, or create an algorithm block's
    ``policy_kwargs``; anything else is refused at apply time, naming the override.
    """

    @staticmethod
    def _configs():
        return {
            1: {
                "env_kwargs": {"alive_bonus": 2.0, "zones": {"width": 1.0, "depth": 2.0}},
                "ppo_kwargs": {"learning_rate": 1e-3, "policy_kwargs": {"net_arch": [256, 256], "log_std_init": -1.0}},
                "sac_kwargs": {},
            },
            2: {
                "env_kwargs": {"alive_bonus": 1.5, "zones": {"width": 3.0}},
                "ppo_kwargs": {"learning_rate": 5e-4},
                "sac_kwargs": {},
            },
        }

    def test_all_stages_nested_env_key_deep_sets_into_the_table(self):
        configs = self._configs()
        _apply_overrides(configs, ["env.zones.width=0.5"])
        assert configs[1]["env_kwargs"]["zones"] == {"width": 0.5, "depth": 2.0}
        assert configs[2]["env_kwargs"]["zones"] == {"width": 0.5}
        assert all("zones.width" not in configs[stage]["env_kwargs"] for stage in configs)

    def test_stage_scoped_nested_key_sets_that_stage_only(self):
        configs = self._configs()
        _apply_overrides(configs, ["1.ppo.policy_kwargs.log_std_init=-1.5"])
        assert configs[1]["ppo_kwargs"]["policy_kwargs"] == {"net_arch": [256, 256], "log_std_init": -1.5}
        assert "policy_kwargs" not in configs[2]["ppo_kwargs"]

    def test_manifest_stage_id_resolves_for_a_nested_key(self):
        configs = self._configs()
        _apply_overrides(configs, ["stance.ppo.policy_kwargs.net_arch=[64, 64]"], TYRANNOSAURUS_REX)
        assert configs[1]["ppo_kwargs"]["policy_kwargs"]["net_arch"] == [64, 64]
        assert "policy_kwargs" not in configs[2]["ppo_kwargs"]

    def test_policy_kwargs_is_created_where_the_stage_declares_none(self):
        configs = self._configs()
        _apply_overrides(configs, ["ppo.policy_kwargs.log_std_init=-1.5", "sac.policy_kwargs.net_arch=[64]"])
        assert configs[1]["ppo_kwargs"]["policy_kwargs"] == {"net_arch": [256, 256], "log_std_init": -1.5}
        assert configs[2]["ppo_kwargs"]["policy_kwargs"] == {"log_std_init": -1.5}
        assert configs[2]["sac_kwargs"] == {"policy_kwargs": {"net_arch": [64]}}

    def test_a_table_the_stage_does_not_declare_is_refused_and_nothing_is_written(self):
        configs = self._configs()
        with pytest.raises(ValueError, match=r"'env\.a\.b' sets a key inside 'env\.a', a table stage 1's config"):
            _apply_overrides(configs, ["env.a.b=1"])
        assert configs == self._configs()

    def test_an_all_stages_key_one_stage_cannot_take_writes_no_stage(self):
        """Stage 2's zones has no ``depth`` table: the override is refused before stage 1 is written."""
        configs = self._configs()
        configs[1]["env_kwargs"]["zones"]["depth"] = {"min": 0.1}
        with pytest.raises(ValueError, match=r"inside 'env\.zones\.depth', a table stage 2's config does not declare"):
            _apply_overrides(configs, ["env.zones.depth.min=0.5"])
        assert configs[1]["env_kwargs"]["zones"]["depth"] == {"min": 0.1}

    def test_a_key_inside_a_value_that_is_not_a_table_is_refused(self):
        with pytest.raises(
            ValueError, match=r"'env\.alive_bonus\.x' sets a key inside 'env\.alive_bonus', which is not"
        ):
            _apply_overrides(self._configs(), ["env.alive_bonus.x=1"])
        with pytest.raises(
            ValueError, match=r"inside 'ppo\.policy_kwargs\.net_arch', which is not a table .*\[256, 256\]"
        ):
            _apply_overrides(self._configs(), ["1.ppo.policy_kwargs.net_arch.pi=[64]"])

    def test_only_policy_kwargs_itself_is_created(self):
        """A created policy_kwargs is empty, so a key below a table inside it is refused."""
        with pytest.raises(ValueError, match=r"inside 'ppo\.policy_kwargs\.net_arch', a table stage 2's config"):
            _apply_overrides(self._configs(), ["2.ppo.policy_kwargs.net_arch.pi=[64]"])
        with pytest.raises(ValueError, match=r"inside 'env\.policy_kwargs', a table stage 1's config"):
            _apply_overrides(self._configs(), ["env.policy_kwargs.x=1"])

    def test_a_typoed_stage_with_a_nested_key_gets_the_stage_error(self):
        with pytest.raises(ValueError, match="unknown stage 'stnce'.*available"):
            _apply_overrides(self._configs(), ["stnce.ppo.policy_kwargs.net_arch=[64]"], TYRANNOSAURUS_REX)

    def test_an_env_array_becomes_a_tuple_as_the_stage_loader_makes_it(self):
        configs = self._configs()
        _apply_overrides(configs, ["env.prey_distance_range=[1.0, 2.5]", "ppo.policy_kwargs.net_arch=[32, 32]"])
        assert configs[1]["env_kwargs"]["prey_distance_range"] == (1.0, 2.5)
        assert configs[1]["ppo_kwargs"]["policy_kwargs"]["net_arch"] == [32, 32]

    def test_the_committed_stage_configs_take_a_nested_policy_kwargs_key(self):
        """Every stage of a real model: [ppo.policy_kwargs] is a table, and [env] has none to enter."""
        from environments.shared.config import load_all_stages

        configs = load_all_stages(TYRANNOSAURUS_REX)
        _apply_overrides(configs, ["ppo.policy_kwargs.log_std_init=-1.25"], TYRANNOSAURUS_REX)
        assert all(cfg["ppo_kwargs"]["policy_kwargs"]["log_std_init"] == -1.25 for cfg in configs.values())
        with pytest.raises(ValueError, match="which is not a table"):
            _apply_overrides(configs, ["env.max_episode_steps.x=1"], TYRANNOSAURUS_REX)
