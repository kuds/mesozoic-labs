"""Semantic-stage tooling regressions (review §3.5, findings F9-F13).

The recovery stage has no legacy number, so every tool that assumed an
integer stage reference silently dropped it: figure generation crashed
inside stage_artifacts' broad try/except (F9), the CLI raised a raw
KeyError before its friendly unknown-stage error (F10), eval stage
detection misread recovery checkpoints and its --stage could not override
the misread (F11), and the W&B run name minted "trex-stagerecovery" (F13).
These tests pin the restored behavior for BOTH generations of stage
references: legacy integers keep their exact historical behavior, and
semantic ids work everywhere integers do.
"""

import zlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared.cli import _parse_stage_ref, _resolve_stage_ref, main
from environments.shared.evaluation import detect_stage_from_path
from environments.shared.visualization import _stage_style_index


class TestStageStyleIndex:
    """One helper maps every stage reference to a stable small int."""

    def test_integer_stages_pass_through_unchanged(self):
        # Bit-for-bit the old `stage_num % ...` inputs, so stages 1-3 keep
        # their exact historical colors and linestyles.
        for stage in (1, 2, 3):
            assert _stage_style_index(stage) == stage
            assert _stage_style_index(stage, species="trex") == stage

    def test_semantic_id_uses_manifest_position(self):
        # configs/trex/stages.toml places recovery at position 2.
        assert _stage_style_index("recovery", species="trex") == 2

    def test_unresolvable_id_gets_stable_crc32_fallback(self):
        # crc32, not hash(): hash() varies per process, which would shuffle
        # colors between figure regenerations of the same run.
        expected = zlib.crc32(b"recovery") % 1000
        assert _stage_style_index("recovery") == expected
        assert _stage_style_index("recovery", species="no_such_species") == expected
        assert _stage_style_index("recovery") != _stage_style_index("behavior")


class TestRecoveryStageFigures:
    """The four figures F9 cost every recovery run must render again."""

    @pytest.fixture()
    def recovery_stage_dir(self, tmp_path):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=np.array([[10.0, 12.0], [20.0, 22.0]]),
            ep_lengths=np.array([[100, 110], [200, 210]]),
            timesteps=np.array([50000, 100000]),
        )
        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=np.array([50000, 100000]),
            tilt_angle=np.array([0.1, 0.05]),
            forward_vel=np.array([0.5, 1.0]),
            pelvis_height=np.array([0.8, 0.85]),
            reward_energy=np.array([-0.1, -0.2]),
            reward_alive=np.array([1.0, 1.0]),
            r_foot_contact=np.array([0.5, 0.8]),
            l_foot_contact=np.array([0.3, 0.6]),
            bilateral_support_duty=np.array([0.5, 0.9]),
            single_support_duty=np.array([0.5, 0.1]),
            foot_load_imbalance=np.array([0.3, 0.1]),
            drift_distance=np.array([0.2, 0.1]),
        )
        return tmp_path

    _CONFIGS = {"recovery": {"name": "Recovery", "curriculum_kwargs": {}, "env_kwargs": {"alive_bonus": 1.0}}}

    def test_training_curves_render_for_stage_recovery(self, recovery_stage_dir):
        import matplotlib.pyplot as plt

        from environments.shared.visualization import plot_training_curves

        save_path = recovery_stage_dir / "training_curves.png"
        fig = plot_training_curves(
            [("recovery", recovery_stage_dir)],
            self._CONFIGS,
            species="trex",
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )
        plt.close(fig)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_diagnostics_graphs_render_for_stage_recovery(self, recovery_stage_dir):
        # Regression (F9): "recovery" % 10 raised TypeError here, and
        # stage_artifacts' broad try/except swallowed it — one figure where
        # stance got five.
        from environments.shared.visualization import plot_diagnostics_graphs

        plot_diagnostics_graphs(
            [("recovery", recovery_stage_dir)],
            self._CONFIGS,
            species="trex",
            algorithm="ppo",
            save_dir=recovery_stage_dir,
            show=False,
        )
        assert (recovery_stage_dir / "locomotion_health.png").exists()
        assert (recovery_stage_dir / "behavioral_metrics.png").exists()

    def test_foot_contacts_render_for_stage_recovery(self, recovery_stage_dir):
        from environments.shared.visualization import plot_foot_contacts

        save_path = recovery_stage_dir / "foot_contacts.png"
        fig = plot_foot_contacts(
            [("recovery", recovery_stage_dir)],
            self._CONFIGS,
            species="trex",
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )
        assert save_path.exists()
        labels = [line.get_label() for line in fig.axes[0].get_lines()]
        assert any("recovery" in label for label in labels)

    def test_stance_diagnostics_render_for_stage_recovery(self, recovery_stage_dir):
        from environments.shared.visualization import plot_stance_diagnostics

        save_path = recovery_stage_dir / "stance_diagnostics.png"
        fig = plot_stance_diagnostics(
            [("recovery", recovery_stage_dir)],
            self._CONFIGS,
            species="trex",
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )
        assert fig is not None
        assert save_path.exists()


class TestDetectStageFromPathSemanticLayouts:
    """Stage inference across every run-directory generation (F11)."""

    def test_single_stage_run_dir_with_trailing_timestamp(self):
        # train()'s default log dir is f"{stage_dirname(...)}_{timestamp}" —
        # this exact layout was misread as stage 1 in the 5M pilot.
        assert detect_stage_from_path("/logs/trex/02_recovery_20260821_142144/models/best_model.zip") == "recovery"

    def test_bare_semantic_dir_from_20260819_layout(self):
        assert detect_stage_from_path("/logs/trex/ppo_20260819_120000/recovery/models/best_model.zip") == "recovery"

    def test_stage1b_component_does_not_read_as_stage1(self):
        # "stage1" matched as a raw substring of "stage1b" and outranked the
        # explicit recovery dir deeper in the path.
        assert detect_stage_from_path("/runs/stage1b/02_recovery/models/best_model.zip") == "recovery"

    def test_stage10_component_does_not_read_as_stage1(self):
        assert detect_stage_from_path("/runs/stage10/03_locomotion/models/best_model.zip") == 2

    def test_deepest_evidence_wins_over_outer_stage_folder(self):
        assert detect_stage_from_path("/logs/stage1_experiments/02_recovery/models/best_model.zip") == "recovery"

    def test_existing_stage2_layouts_stay_pinned(self):
        assert detect_stage_from_path("/runs/20260817/stage2/models/best_model.zip") == 2
        assert detect_stage_from_path("/runs/20260817/stage2/stage2_final.zip") == 2

    def test_unrecognized_paths_keep_the_historical_default(self):
        assert detect_stage_from_path("/tmp/some_model.zip") == 1

    def test_an_unreserved_nn_component_keeps_the_historical_default_species_free(self):
        # Manifest v2 opened the id vocabulary, but a species-free reader
        # recognises only the reserved ids (decision D-A12): an NN_<word>
        # directory on a checkpoint path cannot claim to be a stage.
        assert detect_stage_from_path("/runs/x/05_experiments/models/best_model.zip") == 1
        assert detect_stage_from_path("/runs/x/05_follow_direction/models/best_model.zip") == 1


class TestCliStageResolution:
    """--stage resolution runs before ANY stage_configs lookup (F10)."""

    def test_parse_stage_ref_coerces_digits_only(self):
        assert _parse_stage_ref("2") == 2
        assert isinstance(_parse_stage_ref("2"), int)
        assert _parse_stage_ref("recovery") == "recovery"

    def test_legacy_semantic_id_resolves_to_historical_number(self):
        configs = {1: {}, 2: {"curriculum_kwargs": {"timesteps": 123_456}}, 3: {}}
        resolved = _resolve_stage_ref("locomotion", configs, "trex")
        assert resolved == 2
        # The default-timesteps lookup that used to raise a raw KeyError.
        assert configs[resolved]["curriculum_kwargs"]["timesteps"] == 123_456

    def test_semantic_only_id_passes_through_as_config_key(self):
        assert _resolve_stage_ref("recovery", {1: {}, "recovery": {}}, "trex") == "recovery"

    def test_unknown_refs_return_none_for_the_friendly_error(self):
        configs = {1: {}, 2: {}, 3: {}}
        assert _resolve_stage_ref("cartwheel", configs, "trex") is None
        assert _resolve_stage_ref(5, configs, "trex") is None
        # A species whose manifest has no recovery stage fails closed.
        assert _resolve_stage_ref("recovery", configs, "velociraptor") is None

    @pytest.fixture
    def species_cfg(self):
        cfg = MagicMock()
        cfg.species = "velociraptor"
        cfg.stage_descriptions = "1=balance, 2=locomotion, 3=strike"
        return cfg

    def test_train_stage_locomotion_without_timesteps_uses_stage_config(self, species_cfg):
        mock_train = MagicMock()
        mock_load = MagicMock(
            return_value={
                1: {"env_kwargs": {}, "ppo_kwargs": {}},
                2: {"env_kwargs": {}, "ppo_kwargs": {}, "curriculum_kwargs": {"timesteps": 777_000}},
                3: {"env_kwargs": {}, "ppo_kwargs": {}},
            }
        )
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train", mock_train),
            patch("sys.argv", ["prog", "train", "--stage", "locomotion"]),
        ):
            main(species_cfg)

        assert mock_train.call_args.kwargs["stage"] == 2
        assert mock_train.call_args.kwargs["total_timesteps"] == 777_000

    def test_train_unknown_stage_reaches_the_friendly_parser_error(self, species_cfg):
        mock_train = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}})
        for bad_stage in ("cartwheel", "5"):
            with (
                patch("environments.shared.config.load_all_stages", mock_load),
                patch("environments.shared.train_base.train", mock_train),
                patch("sys.argv", ["prog", "train", "--stage", bad_stage]),
                pytest.raises(SystemExit),
            ):
                main(species_cfg)
        mock_train.assert_not_called()

    def test_eval_stage_recovery_reaches_evaluate(self):
        # F11: eval --stage was int choices=[1, 2, 3], so a misdetected
        # recovery checkpoint could not be corrected from the CLI.
        species_cfg = MagicMock()
        species_cfg.species = "trex"
        species_cfg.stage_descriptions = "1=stance, 2=locomotion, 3=behavior"
        mock_eval = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}, "recovery": {}})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.evaluation.evaluate", mock_eval),
            patch("sys.argv", ["prog", "eval", "/tmp/model.zip", "--stage", "recovery"]),
        ):
            main(species_cfg)

        assert mock_eval.call_args.kwargs["stage"] == "recovery"

    def test_eval_integer_stage_keeps_current_behavior(self, species_cfg):
        mock_eval = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.evaluation.evaluate", mock_eval),
            patch("sys.argv", ["prog", "eval", "/tmp/model.zip", "--stage", "2"]),
        ):
            main(species_cfg)

        assert mock_eval.call_args.kwargs["stage"] == 2

    def test_eval_unknown_stage_reaches_the_friendly_parser_error(self, species_cfg):
        mock_eval = MagicMock()
        mock_load = MagicMock(return_value={1: {}, 2: {}, 3: {}})
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.evaluation.evaluate", mock_eval),
            patch("sys.argv", ["prog", "eval", "/tmp/model.zip", "--stage", "warpdrive"]),
            pytest.raises(SystemExit),
        ):
            main(species_cfg)
        mock_eval.assert_not_called()


class TestWandbRunName:
    """Run names follow stage_label: no more "trex-stagerecovery" (F13)."""

    @pytest.fixture
    def mock_wandb(self):
        with patch("environments.shared.wandb_integration.wandb") as mock:
            yield mock

    def test_integer_stage_keeps_historical_run_name(self, mock_wandb):
        from environments.shared.wandb_integration import init_wandb

        init_wandb("trex", 1, {"name": "stance"})
        assert mock_wandb.init.call_args.kwargs["name"] == "trex-stage1"

    def test_semantic_stage_named_by_id(self, mock_wandb):
        from environments.shared.wandb_integration import init_wandb

        init_wandb("trex", "recovery", {"name": "recovery"})
        assert mock_wandb.init.call_args.kwargs["name"] == "trex-recovery"
