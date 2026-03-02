"""Tests for the hyperparameter sweep tool's pure functions."""

from unittest.mock import MagicMock

import pytest

from environments.shared.scripts.sweep import (
    NET_ARCH_PRESETS,
    _best_trial_model_path,
    _collect_trial_results,
    _hpt_arg_to_override,
    _is_per_stage,
    _resolve_search_space,
    _search_space_for_stage,
    _settings_for_stage,
    _split_stage_block,
    write_results_csv,
)

# ── _hpt_arg_to_override ────────────────────────────────────────────────────


class TestHptArgToOverride:
    """Conversion of Vertex AI HPT arg names to --override dot notation."""

    def test_ppo_prefix(self):
        assert _hpt_arg_to_override("ppo_learning_rate", "0.0003") == "ppo.learning_rate=0.0003"

    def test_sac_prefix(self):
        assert _hpt_arg_to_override("sac_batch_size", "256") == "sac.batch_size=256"

    def test_env_prefix(self):
        assert _hpt_arg_to_override("env_alive_bonus", "2.0") == "env.alive_bonus=2.0"

    def test_curriculum_prefix(self):
        assert (
            _hpt_arg_to_override("curriculum_warmup_timesteps", "50000")
            == "curriculum.warmup_timesteps=50000"
        )

    def test_unknown_prefix_passthrough(self):
        assert _hpt_arg_to_override("unknown_param", "42") == "unknown_param=42"

    def test_net_arch_preset(self):
        assert _hpt_arg_to_override("ppo_net_arch", "medium") == "ppo.net_arch=medium"

    def test_multi_underscore_param(self):
        """Only the first underscore after the prefix becomes the dot separator."""
        assert _hpt_arg_to_override("ppo_clip_range", "0.2") == "ppo.clip_range=0.2"

    def test_env_multi_word_param(self):
        assert (
            _hpt_arg_to_override("env_forward_vel_weight", "1.5")
            == "env.forward_vel_weight=1.5"
        )


# ── _is_per_stage / _split_stage_block / _search_space_for_stage ─────────


class TestPerStageDetection:
    def test_flat_config(self):
        flat = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}
        assert _is_per_stage(flat) is False

    def test_per_stage_config(self):
        per_stage = {
            "stage1": {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}},
            "stage2": {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 1e-4}},
        }
        assert _is_per_stage(per_stage) is True

    def test_partial_stage_keys(self):
        """Even having just stage1 makes it per-stage."""
        partial = {"stage1": {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}}
        assert _is_per_stage(partial) is True


class TestSplitStageBlock:
    def test_separates_search_space_and_settings(self):
        block = {
            "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
            "trials": 30,
            "timesteps": 1000000,
            "parallel": 5,
        }
        search_space, settings = _split_stage_block(block)
        assert "ppo_learning_rate" in search_space
        assert search_space["ppo_learning_rate"]["type"] == "double"
        assert settings == {"trials": 30, "timesteps": 1000000, "parallel": 5}

    def test_all_search_params(self):
        block = {
            "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4},
            "ppo_ent_coef": {"type": "double", "min": 0.001, "max": 0.05},
        }
        search_space, settings = _split_stage_block(block)
        assert len(search_space) == 2
        assert len(settings) == 0

    def test_all_settings(self):
        block = {"trials": 10, "timesteps": 500000}
        search_space, settings = _split_stage_block(block)
        assert len(search_space) == 0
        assert len(settings) == 2

    def test_empty_block(self):
        search_space, settings = _split_stage_block({})
        assert search_space == {}
        assert settings == {}


class TestSearchSpaceForStage:
    def test_flat_returns_as_is(self):
        flat = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}
        assert _search_space_for_stage(flat, 1) == flat
        assert _search_space_for_stage(flat, 2) == flat

    def test_per_stage_extracts_correct_stage(self):
        per_stage = {
            "stage1": {
                "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4},
                "trials": 20,
            },
            "stage2": {
                "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 1e-4},
            },
            "stage3": {
                "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 5e-5},
            },
        }
        space1 = _search_space_for_stage(per_stage, 1)
        # Should have only the search space param, not the "trials" setting
        assert "ppo_learning_rate" in space1
        assert "trials" not in space1

        space2 = _search_space_for_stage(per_stage, 2)
        assert space2["ppo_learning_rate"]["max"] == 1e-4

    def test_missing_stage_key_exits(self):
        per_stage = {"stage1": {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}}
        with pytest.raises(SystemExit):
            _search_space_for_stage(per_stage, 3)


class TestSettingsForStage:
    def test_flat_returns_empty(self):
        flat = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}
        assert _settings_for_stage(flat, 1) == {}

    def test_per_stage_extracts_settings(self):
        per_stage = {
            "stage1": {
                "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4},
                "trials": 30,
                "timesteps": 1000000,
            },
        }
        settings = _settings_for_stage(per_stage, 1)
        assert settings == {"trials": 30, "timesteps": 1000000}

    def test_missing_stage_returns_empty(self):
        per_stage = {"stage1": {"trials": 10}}
        assert _settings_for_stage(per_stage, 3) == {}


# ── _resolve_search_space ────────────────────────────────────────────────


class TestResolveSearchSpace:
    def test_inline_json_takes_priority(self):
        inline = '{"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 1e-3}}'
        result = _resolve_search_space(inline, None, "ppo")
        assert "ppo_learning_rate" in result
        assert result["ppo_learning_rate"]["max"] == 1e-3

    def test_invalid_json_exits(self):
        with pytest.raises(SystemExit):
            _resolve_search_space("{bad json", None, "ppo")

    def test_default_ppo_space(self):
        result = _resolve_search_space(None, None, "ppo")
        assert "ppo_learning_rate" in result
        assert "ppo_ent_coef" in result
        assert "ppo_batch_size" in result

    def test_default_sac_space(self):
        result = _resolve_search_space(None, None, "sac")
        assert "sac_learning_rate" in result
        assert "sac_batch_size" in result

    def test_unknown_algorithm_falls_back_to_ppo(self):
        result = _resolve_search_space(None, None, "unknown_algo")
        assert "ppo_learning_rate" in result


# ── _collect_trial_results ───────────────────────────────────────────────


def _make_mock_trial(trial_id, params=None, metrics=None):
    """Helper to build a mock Vertex AI trial object."""
    trial = MagicMock()
    trial.id = trial_id
    trial.parameters = []
    if params:
        for pid, val in params.items():
            p = MagicMock()
            p.parameter_id = pid
            p.value = val
            trial.parameters.append(p)

    if metrics:
        measurement = MagicMock()
        metric_objs = []
        for mid, val in metrics.items():
            m = MagicMock()
            m.metric_id = mid
            m.value = val
            metric_objs.append(m)
        measurement.metrics = metric_objs
        trial.final_measurement = measurement
    else:
        trial.final_measurement = None

    return trial


class TestCollectTrialResults:
    def test_passing_trial(self):
        trial = _make_mock_trial(
            "1",
            params={"ppo_learning_rate": 0.0003},
            metrics={"best_mean_reward": 150.0, "best_mean_episode_length": 500.0},
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}

        rows = _collect_trial_results(job, 1, stage_config)
        assert len(rows) == 1
        assert rows[0]["stage_passed"] is True
        assert rows[0]["best_mean_reward"] == 150.0

    def test_failing_trial_below_threshold(self):
        trial = _make_mock_trial(
            "2",
            metrics={"best_mean_reward": 50.0},
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}

        rows = _collect_trial_results(job, 1, stage_config)
        assert rows[0]["stage_passed"] is False

    def test_crashed_trial_no_metrics(self):
        trial = _make_mock_trial("3")
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {}}

        rows = _collect_trial_results(job, 1, stage_config)
        assert rows[0]["stage_passed"] is False
        assert rows[0]["best_mean_reward"] is None

    def test_no_thresholds_passes_with_valid_reward(self):
        trial = _make_mock_trial("4", metrics={"best_mean_reward": 10.0})
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {}}

        rows = _collect_trial_results(job, 3, stage_config)
        assert rows[0]["stage_passed"] is True

    def test_ep_length_threshold_checked(self):
        trial = _make_mock_trial(
            "5",
            metrics={"best_mean_reward": 200.0, "best_mean_episode_length": 100.0},
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {
            "curriculum_kwargs": {
                "min_avg_reward": 100.0,
                "min_avg_episode_length": 500.0,
            }
        }

        rows = _collect_trial_results(job, 1, stage_config)
        assert rows[0]["stage_passed"] is False

    def test_forward_vel_threshold_checked(self):
        trial = _make_mock_trial(
            "6",
            metrics={
                "best_mean_reward": 200.0,
                "best_mean_forward_vel": 0.5,
            },
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {
            "curriculum_kwargs": {
                "min_avg_reward": 100.0,
                "min_avg_forward_vel": 1.0,
            }
        }

        rows = _collect_trial_results(job, 2, stage_config)
        assert rows[0]["stage_passed"] is False


# ── _best_trial_model_path ───────────────────────────────────────────────


class TestBestTrialModelPath:
    def test_selects_highest_reward_among_passed(self):
        rows = [
            {"trial_id": "1", "stage_passed": True, "best_mean_reward": 100.0},
            {"trial_id": "2", "stage_passed": True, "best_mean_reward": 200.0},
            {"trial_id": "3", "stage_passed": False, "best_mean_reward": 300.0},
        ]
        path, best_row = _best_trial_model_path(rows, "my-bucket", "velociraptor", 1)
        assert best_row["trial_id"] == "2"
        assert "my-bucket" in path
        assert "stage1" in path
        assert "/2/" in path

    def test_exits_when_no_trials_pass(self):
        rows = [
            {"trial_id": "1", "stage_passed": False, "best_mean_reward": 50.0},
        ]
        with pytest.raises(SystemExit):
            _best_trial_model_path(rows, "bucket", "trex", 2)

    def test_gcs_path_format(self):
        rows = [{"trial_id": "7", "stage_passed": True, "best_mean_reward": 150.0}]
        path, _ = _best_trial_model_path(rows, "my-bucket", "velociraptor", 2)
        assert path == "/gcs/my-bucket/sweeps/velociraptor/stage2/7/models/stage2_final.zip"


# ── write_results_csv ────────────────────────────────────────────────────


class TestWriteResultsCsv:
    def test_writes_csv(self, tmp_path):
        rows = [
            {
                "trial_id": "1",
                "stage": 1,
                "ppo_learning_rate": 0.0003,
                "best_mean_reward": 100.0,
                "best_mean_episode_length": 500.0,
                "last_mean_reward": 95.0,
                "last_mean_episode_length": 480.0,
                "reward_threshold": 80.0,
                "ep_length_threshold": None,
                "forward_vel_threshold": None,
                "success_rate_threshold": None,
                "stage_passed": True,
            }
        ]
        csv_path = tmp_path / "results.csv"
        result = write_results_csv(rows, csv_path)
        assert result == csv_path
        assert csv_path.exists()

        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            written = list(reader)
        assert len(written) == 1
        assert written[0]["trial_id"] == "1"
        assert written[0]["ppo_learning_rate"] == "0.0003"

    def test_empty_rows_skipped(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        write_results_csv([], csv_path)
        assert not csv_path.exists()


# ── NET_ARCH_PRESETS ─────────────────────────────────────────────────────


class TestNetArchPresets:
    def test_all_presets_are_lists_of_ints(self):
        for name, arch in NET_ARCH_PRESETS.items():
            assert isinstance(arch, list), f"Preset {name} should be a list"
            assert all(isinstance(x, int) for x in arch), f"Preset {name} should contain ints"

    def test_expected_presets_exist(self):
        expected = {"small", "medium", "large", "deep", "tapered", "deep_tapered"}
        assert set(NET_ARCH_PRESETS.keys()) == expected
