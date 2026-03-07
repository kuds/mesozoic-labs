"""Tests for the hyperparameter sweep tool's pure functions."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.scripts.sweep import (
    NET_ARCH_PRESETS,
    SweepStageError,
    _best_trial_model_path,
    _collect_trial_results,
    _hpt_arg_to_override,
    _is_per_stage,
    _is_retryable_gcp_error,
    _load_sweep_state,
    _normalize_accelerator_type,
    _parse_hpt_extra_args,
    _resolve_search_space,
    _save_sweep_state,
    _search_space_for_stage,
    _settings_for_stage,
    _split_stage_block,
    _sweep_state_local_path,
    _SweepJobFailed,
    _validate_machine_type,
    collect_results_from_disk,
    write_results_csv,
)

# ── _normalize_accelerator_type ──────────────────────────────────────────────


class TestNormalizeAcceleratorType:
    """Short aliases like 'T4' are expanded to full Vertex AI enum labels."""

    def test_short_alias_t4(self):
        assert _normalize_accelerator_type("T4") == "NVIDIA_TESLA_T4"

    def test_short_alias_case_insensitive(self):
        assert _normalize_accelerator_type("t4") == "NVIDIA_TESLA_T4"
        assert _normalize_accelerator_type("v100") == "NVIDIA_TESLA_V100"

    def test_full_name_passthrough(self):
        assert _normalize_accelerator_type("NVIDIA_TESLA_T4") == "NVIDIA_TESLA_T4"

    def test_none_string_returns_none(self):
        assert _normalize_accelerator_type("None") is None
        assert _normalize_accelerator_type("none") is None
        assert _normalize_accelerator_type("NONE") is None

    def test_unknown_type_passthrough(self):
        assert _normalize_accelerator_type("TPU_V3") == "TPU_V3"

    def test_l4_alias(self):
        assert _normalize_accelerator_type("L4") == "NVIDIA_L4"

    def test_a100_alias(self):
        assert _normalize_accelerator_type("A100") == "NVIDIA_TESLA_A100"


# ── _validate_machine_type ───────────────────────────────────────────────────


class TestValidateMachineType:
    """Unsupported machine families are rejected before job submission."""

    @pytest.mark.parametrize(
        "mt",
        [
            "n1-standard-8",
            "n2-standard-4",
            "e2-standard-16",
            "c2-standard-8",
            "c3-standard-4",
            "a2-highgpu-1g",
            "g2-standard-4",
        ],
    )
    def test_supported_types_pass(self, mt):
        _validate_machine_type(mt)  # should not raise

    @pytest.mark.parametrize("mt", ["c4-standard-8", "c4-highcpu-16", "z1-standard-4"])
    def test_unsupported_types_raise(self, mt):
        with pytest.raises(ValueError, match="not supported by Vertex AI"):
            _validate_machine_type(mt)


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
        assert _hpt_arg_to_override("curriculum_warmup_timesteps", "50000") == "curriculum.warmup_timesteps=50000"

    def test_unknown_prefix_passthrough(self):
        assert _hpt_arg_to_override("unknown_param", "42") == "unknown_param=42"

    def test_net_arch_preset(self):
        assert _hpt_arg_to_override("ppo_net_arch", "medium") == "ppo.net_arch=medium"

    def test_multi_underscore_param(self):
        """Only the first underscore after the prefix becomes the dot separator."""
        assert _hpt_arg_to_override("ppo_clip_range", "0.2") == "ppo.clip_range=0.2"

    def test_env_multi_word_param(self):
        assert _hpt_arg_to_override("env_forward_vel_weight", "1.5") == "env.forward_vel_weight=1.5"


# ── _parse_hpt_extra_args ────────────────────────────────────────────────────


class TestParseHptExtraArgs:
    """Parsing of HPT-injected extra CLI args into override strings."""

    def test_equals_format(self):
        """Vertex AI HPT uses --key=value format."""
        extra = ["--ppo_learning_rate=0.0001", "--ppo_ent_coef=0.005"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001", "ppo.ent_coef=0.005"]

    def test_space_separated_format(self):
        """Fallback --key value format."""
        extra = ["--ppo_learning_rate", "0.0001", "--ppo_ent_coef", "0.005"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001", "ppo.ent_coef=0.005"]

    def test_mixed_formats(self):
        """Both formats can appear in the same arg list."""
        extra = ["--ppo_learning_rate=0.0001", "--ppo_ent_coef", "0.005"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001", "ppo.ent_coef=0.005"]

    def test_empty(self):
        assert _parse_hpt_extra_args([]) == []

    def test_boolean_flags_skipped(self):
        """Boolean flags (no value) are skipped."""
        extra = ["--some_flag", "--ppo_learning_rate=0.0001"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001"]

    def test_discrete_values(self):
        """Discrete HPT params like batch_size come as floats from Vertex AI."""
        extra = ["--ppo_batch_size=256.0", "--ppo_n_steps=2048.0"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.batch_size=256.0", "ppo.n_steps=2048.0"]

    def test_multiple_params_equals_format(self):
        """Realistic Vertex AI HPT injection with 5 parameters."""
        extra = [
            "--ppo_learning_rate=0.0001",
            "--ppo_ent_coef=0.005",
            "--ppo_batch_size=128.0",
            "--ppo_gamma=0.985",
            "--ppo_n_steps=2048.0",
        ]
        result = _parse_hpt_extra_args(extra)
        assert len(result) == 5
        assert "ppo.learning_rate=0.0001" in result
        assert "ppo.batch_size=128.0" in result

    def test_non_flag_tokens_skipped(self):
        """Bare tokens without -- prefix are skipped."""
        extra = ["stray_token", "--ppo_learning_rate=0.0001"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001"]


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

    def test_unknown_algorithm_exits_with_error(self):
        with pytest.raises(SystemExit):
            _resolve_search_space(None, None, "unknown_algo")


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


def _write_trial_metrics(output_base, trial_id, metrics_dict):
    """Write a metrics.json sidecar for a mock trial."""
    import json

    trial_dir = Path(output_base) / str(trial_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f)


class TestCollectTrialResults:
    def test_passing_trial(self, tmp_path):
        trial = _make_mock_trial(
            "1",
            params={"ppo_learning_rate": 0.0003},
            metrics={"best_mean_reward": 150.0},
        )
        _write_trial_metrics(
            tmp_path,
            "1",
            {
                "best_mean_reward": 150.0,
                "best_mean_episode_length": 500.0,
            },
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}

        rows = _collect_trial_results(job, 1, stage_config, output_base=str(tmp_path))
        assert len(rows) == 1
        assert rows[0]["stage_passed"] is True
        assert rows[0]["best_mean_reward"] == 150.0
        assert rows[0]["best_mean_episode_length"] == 500.0

    def test_failing_trial_below_threshold(self, tmp_path):
        trial = _make_mock_trial(
            "2",
            metrics={"best_mean_reward": 50.0},
        )
        _write_trial_metrics(tmp_path, "2", {"best_mean_reward": 50.0})
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}

        rows = _collect_trial_results(job, 1, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is False

    def test_crashed_trial_no_metrics(self, tmp_path):
        trial = _make_mock_trial("3")
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {}}

        rows = _collect_trial_results(job, 1, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is False
        assert rows[0]["best_mean_reward"] is None

    def test_no_thresholds_passes_with_valid_reward(self, tmp_path):
        trial = _make_mock_trial("4", metrics={"best_mean_reward": 10.0})
        _write_trial_metrics(tmp_path, "4", {"best_mean_reward": 10.0})
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {}}

        rows = _collect_trial_results(job, 3, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is True

    def test_ep_length_threshold_checked(self, tmp_path):
        trial = _make_mock_trial(
            "5",
            metrics={"best_mean_reward": 200.0},
        )
        _write_trial_metrics(
            tmp_path,
            "5",
            {
                "best_mean_reward": 200.0,
                "best_mean_episode_length": 100.0,
            },
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {
            "curriculum_kwargs": {
                "min_avg_reward": 100.0,
                "min_avg_episode_length": 500.0,
            }
        }

        rows = _collect_trial_results(job, 1, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is False

    def test_forward_vel_threshold_checked(self, tmp_path):
        trial = _make_mock_trial(
            "6",
            metrics={"best_mean_reward": 200.0},
        )
        _write_trial_metrics(
            tmp_path,
            "6",
            {
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

        rows = _collect_trial_results(job, 2, stage_config, output_base=str(tmp_path))
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

    def test_raises_when_no_trials_pass(self):
        rows = [
            {"trial_id": "1", "stage_passed": False, "best_mean_reward": 50.0},
        ]
        with pytest.raises(SweepStageError):
            _best_trial_model_path(rows, "bucket", "trex", 2)

    def test_gcs_path_format(self):
        rows = [{"trial_id": "7", "stage_passed": True, "best_mean_reward": 150.0}]
        path, _ = _best_trial_model_path(rows, "my-bucket", "velociraptor", 2)
        assert path == "/gcs/my-bucket/sweeps/velociraptor/stage2/7/models/best_model.zip"


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


# ── SweepStageError ─────────────────────────────────────────────────────


class TestSweepStageError:
    def test_is_exception(self):
        assert issubclass(SweepStageError, Exception)

    def test_message(self):
        err = SweepStageError("stage 2 failed")
        assert "stage 2 failed" in str(err)


# ── _is_retryable_gcp_error ────────────────────────────────────────────


class TestIsRetryableGcpError:
    def test_retryable_by_name(self):
        for name in ("ResourceExhausted", "ServiceUnavailable", "GoogleAPICallError", "TooManyRequests"):
            # Create a dynamically-named exception class
            exc_cls = type(name, (Exception,), {})
            assert _is_retryable_gcp_error(exc_cls("quota exceeded")) is True

    def test_non_retryable(self):
        assert _is_retryable_gcp_error(ValueError("bad")) is False
        assert _is_retryable_gcp_error(RuntimeError("crash")) is False


# ── _save_sweep_state / _load_sweep_state ───────────────────────────────


class TestSaveLoadSweepState:
    def test_round_trip_local(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = {
            "species": "velociraptor",
            "algorithm": "ppo",
            "started_at": "2026-01-01T00:00:00Z",
            "stages": {
                "1": {"status": "completed", "best_model_path": "/gcs/bucket/path.zip"},
            },
        }
        _save_sweep_state(state, "velociraptor", "ppo")
        loaded = _load_sweep_state("velociraptor", "ppo")
        assert loaded is not None
        assert loaded["species"] == "velociraptor"
        assert loaded["stages"]["1"]["status"] == "completed"

    def test_load_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _load_sweep_state("velociraptor", "ppo") is None

    def test_load_returns_none_on_species_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = {"species": "trex", "algorithm": "ppo", "stages": {}}
        _save_sweep_state(state, "trex", "ppo")
        # Try to load with different species
        loaded = _load_sweep_state("velociraptor", "ppo")
        assert loaded is None

    def test_load_returns_none_on_algorithm_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = {"species": "velociraptor", "algorithm": "ppo", "stages": {}}
        _save_sweep_state(state, "velociraptor", "ppo")
        loaded = _load_sweep_state("velociraptor", "sac")
        assert loaded is None

    def test_local_path_format(self):
        path = _sweep_state_local_path("velociraptor", "ppo")
        assert path.name == "sweep_state_velociraptor_ppo.json"

    def test_gcs_upload_failure_is_non_fatal(self, tmp_path, monkeypatch):
        """GCS failures should log a warning but not raise."""
        monkeypatch.chdir(tmp_path)
        state = {"species": "velociraptor", "algorithm": "ppo", "stages": {}}
        # Mock google.cloud.storage to raise on Client() so the GCS upload fails
        mock_storage = MagicMock()
        mock_storage.Client.side_effect = Exception("connection refused")
        with patch.dict(
            "sys.modules", {"google.cloud.storage": mock_storage, "google.cloud": MagicMock(), "google": MagicMock()}
        ):
            # Should not raise even if GCS upload fails
            _save_sweep_state(state, "velociraptor", "ppo", bucket="my-bucket")
        # Local file should still be written
        assert _sweep_state_local_path("velociraptor", "ppo").exists()


# ── Retry logic in _submit_stage_sweep ──────────────────────────────────


class TestSubmitStageRetry:
    def test_retries_on_resource_exhausted(self):
        """hpt_job.run() is retried on retryable errors."""
        from environments.shared.scripts.sweep import _submit_stage_sweep

        mock_aiplatform = MagicMock()
        mock_hpt = MagicMock()

        # Make the HPT job's run method fail twice then succeed
        ResourceExhausted = type("ResourceExhausted", (Exception,), {})
        mock_job = MagicMock()
        mock_job.run.side_effect = [
            ResourceExhausted("quota"),
            ResourceExhausted("quota"),
            None,  # success on 3rd attempt
        ]
        mock_aiplatform.HyperparameterTuningJob.return_value = mock_job
        mock_aiplatform.CustomJob.return_value = MagicMock()

        search_space = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}}

        with patch("time.sleep"):
            result = _submit_stage_sweep(
                aiplatform=mock_aiplatform,
                hpt_module=mock_hpt,
                species="velociraptor",
                stage=1,
                algorithm="ppo",
                timesteps=100000,
                n_envs=4,
                trials=5,
                parallel=2,
                bucket="test-bucket",
                image="test-image",
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                search_space=search_space,
            )

        assert mock_job.run.call_count == 3
        assert result is mock_job

    def test_raises_after_all_retries_exhausted(self):
        """After all retry attempts are exhausted, _SweepJobFailed is raised with the hpt_job."""
        from environments.shared.scripts.sweep import _submit_stage_sweep

        mock_aiplatform = MagicMock()
        mock_hpt = MagicMock()

        ResourceExhausted = type("ResourceExhausted", (Exception,), {})
        mock_job = MagicMock()
        mock_job.run.side_effect = ResourceExhausted("quota")
        mock_aiplatform.HyperparameterTuningJob.return_value = mock_job
        mock_aiplatform.CustomJob.return_value = MagicMock()

        search_space = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}}

        with patch("time.sleep"), pytest.raises(_SweepJobFailed) as exc_info:
            _submit_stage_sweep(
                aiplatform=mock_aiplatform,
                hpt_module=mock_hpt,
                species="velociraptor",
                stage=1,
                algorithm="ppo",
                timesteps=100000,
                n_envs=4,
                trials=5,
                parallel=2,
                bucket="test-bucket",
                image="test-image",
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                search_space=search_space,
            )

        # 1 initial + 3 retries = 4 total attempts
        assert mock_job.run.call_count == 4
        # The hpt_job is attached so callers can collect partial trial results
        assert exc_info.value.hpt_job is mock_job
        assert isinstance(exc_info.value.__cause__, ResourceExhausted)

    def test_non_retryable_error_raised_immediately(self):
        """Non-retryable errors are wrapped in _SweepJobFailed without retry."""
        from environments.shared.scripts.sweep import _submit_stage_sweep

        mock_aiplatform = MagicMock()
        mock_hpt = MagicMock()

        mock_job = MagicMock()
        mock_job.run.side_effect = ValueError("bad parameter")
        mock_aiplatform.HyperparameterTuningJob.return_value = mock_job
        mock_aiplatform.CustomJob.return_value = MagicMock()

        search_space = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}}

        with patch("time.sleep"), pytest.raises(_SweepJobFailed) as exc_info:
            _submit_stage_sweep(
                aiplatform=mock_aiplatform,
                hpt_module=mock_hpt,
                species="velociraptor",
                stage=1,
                algorithm="ppo",
                timesteps=100000,
                n_envs=4,
                trials=5,
                parallel=2,
                bucket="test-bucket",
                image="test-image",
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                search_space=search_space,
            )

        assert mock_job.run.call_count == 1
        assert exc_info.value.hpt_job is mock_job
        assert isinstance(exc_info.value.__cause__, ValueError)


# ── _SweepJobFailed ────────────────────────────────────────────────────


class TestSweepJobFailed:
    def test_is_sweep_stage_error(self):
        assert issubclass(_SweepJobFailed, SweepStageError)

    def test_carries_hpt_job(self):
        mock_job = MagicMock()
        exc = _SweepJobFailed("job failed", hpt_job=mock_job)
        assert exc.hpt_job is mock_job
        assert "job failed" in str(exc)

    def test_hpt_job_defaults_to_none(self):
        exc = _SweepJobFailed("no job")
        assert exc.hpt_job is None


# ── _best_trial_model_path with pre-computed model_path ───────────────


class TestBestTrialModelPathWithModelPath:
    def test_uses_precomputed_model_path(self):
        rows = [
            {
                "trial_id": "5",
                "stage_passed": True,
                "best_mean_reward": 200.0,
                "model_path": "/gcs/bucket/sweeps/velociraptor/stage2_r1/5/models/best_model.zip",
            },
        ]
        path, best_row = _best_trial_model_path(rows, "bucket", "velociraptor", 2)
        assert path == "/gcs/bucket/sweeps/velociraptor/stage2_r1/5/models/best_model.zip"

    def test_falls_back_to_constructed_path(self):
        """Rows without model_path still get the default constructed path."""
        rows = [
            {"trial_id": "3", "stage_passed": True, "best_mean_reward": 100.0},
        ]
        path, _ = _best_trial_model_path(rows, "my-bucket", "trex", 1)
        assert path == "/gcs/my-bucket/sweeps/trex/stage1/3/models/best_model.zip"

    def test_mixed_rows_picks_best_with_correct_path(self):
        """When rows from different runs are merged, the best trial's model_path is used."""
        rows = [
            # From original run (no model_path — constructed by default)
            {"trial_id": "1", "stage_passed": True, "best_mean_reward": 100.0},
            # From resumed run (has model_path)
            {
                "trial_id": "1",
                "stage_passed": True,
                "best_mean_reward": 250.0,
                "model_path": "/gcs/bucket/sweeps/velociraptor/stage1_r1/1/models/best_model.zip",
            },
        ]
        path, best_row = _best_trial_model_path(rows, "bucket", "velociraptor", 1)
        assert best_row["best_mean_reward"] == 250.0
        assert path == "/gcs/bucket/sweeps/velociraptor/stage1_r1/1/models/best_model.zip"


# ── Partial trial recovery (save/load state) ──────────────────────────


class TestPartialTrialRecovery:
    def test_partial_state_round_trip(self, tmp_path, monkeypatch):
        """Partial stage data survives a save/load cycle."""
        monkeypatch.chdir(tmp_path)
        state = {
            "species": "velociraptor",
            "algorithm": "ppo",
            "started_at": "2026-01-01T00:00:00Z",
            "stages": {
                "1": {"status": "completed", "best_model_path": "/gcs/bucket/path.zip"},
                "2": {
                    "status": "partial",
                    "trial_rows": [
                        {"trial_id": "1", "best_mean_reward": 100.0, "stage_passed": True},
                        {"trial_id": "2", "best_mean_reward": 80.0, "stage_passed": False},
                    ],
                    "fixed_trial_args": ["--ppo_net_arch", "medium"],
                    "resume_run": 0,
                },
            },
        }
        _save_sweep_state(state, "velociraptor", "ppo")
        loaded = _load_sweep_state("velociraptor", "ppo")
        assert loaded is not None
        assert loaded["stages"]["1"]["status"] == "completed"
        assert loaded["stages"]["2"]["status"] == "partial"
        assert len(loaded["stages"]["2"]["trial_rows"]) == 2
        assert loaded["stages"]["2"]["resume_run"] == 0

    def test_partial_rows_accumulate_across_failures(self, tmp_path, monkeypatch):
        """Multiple failures for the same stage accumulate partial rows."""
        monkeypatch.chdir(tmp_path)
        # First failure saved 3 rows
        state = {
            "species": "velociraptor",
            "algorithm": "ppo",
            "stages": {
                "1": {
                    "status": "partial",
                    "trial_rows": [
                        {
                            "trial_id": "1",
                            "best_mean_reward": 100.0,
                            "stage_passed": True,
                            "model_path": "/gcs/b/s/stage1/1/models/best_model.zip",
                        },
                        {
                            "trial_id": "2",
                            "best_mean_reward": 80.0,
                            "stage_passed": False,
                            "model_path": "/gcs/b/s/stage1/2/models/best_model.zip",
                        },
                        {
                            "trial_id": "3",
                            "best_mean_reward": 120.0,
                            "stage_passed": True,
                            "model_path": "/gcs/b/s/stage1/3/models/best_model.zip",
                        },
                    ],
                    "resume_run": 0,
                },
            },
        }
        _save_sweep_state(state, "velociraptor", "ppo")

        # Simulate resume: load, find partial, note resume_run should be 1
        loaded = _load_sweep_state("velociraptor", "ppo")
        partial_data = loaded["stages"]["1"]
        assert partial_data["status"] == "partial"
        partial_rows = [r for r in partial_data["trial_rows"] if r.get("best_mean_reward") is not None]
        assert len(partial_rows) == 3
        resume_run = partial_data.get("resume_run", 0) + 1
        assert resume_run == 1


# ── resume_run output directory isolation ─────────────────────────────


class TestResumeRunOutputDir:
    def test_resume_run_changes_output_base(self):
        """resume_run > 0 appends _r{N} to the output directory."""
        from environments.shared.scripts.sweep import _submit_stage_sweep

        mock_aiplatform = MagicMock()
        mock_hpt = MagicMock()

        mock_job = MagicMock()
        mock_aiplatform.HyperparameterTuningJob.return_value = mock_job
        mock_aiplatform.CustomJob.return_value = MagicMock()

        search_space = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}}

        _submit_stage_sweep(
            aiplatform=mock_aiplatform,
            hpt_module=mock_hpt,
            species="velociraptor",
            stage=2,
            algorithm="ppo",
            timesteps=100000,
            n_envs=4,
            trials=5,
            parallel=2,
            bucket="test-bucket",
            image="test-image",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            search_space=search_space,
            resume_run=1,
        )

        # Check that the trial args include the resumed output dir
        call_args = mock_aiplatform.CustomJob.call_args
        worker_specs = call_args[1]["worker_pool_specs"]
        trial_args = worker_specs[0]["container_spec"]["args"]
        output_idx = trial_args.index("--output-dir")
        output_dir = trial_args[output_idx + 1]
        assert output_dir == "/gcs/test-bucket/sweeps/velociraptor/stage2_r1"

    def test_resume_run_zero_uses_default_path(self):
        """resume_run=0 (default) uses the standard output directory."""
        from environments.shared.scripts.sweep import _submit_stage_sweep

        mock_aiplatform = MagicMock()
        mock_hpt = MagicMock()

        mock_job = MagicMock()
        mock_aiplatform.HyperparameterTuningJob.return_value = mock_job
        mock_aiplatform.CustomJob.return_value = MagicMock()

        search_space = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}}

        _submit_stage_sweep(
            aiplatform=mock_aiplatform,
            hpt_module=mock_hpt,
            species="velociraptor",
            stage=2,
            algorithm="ppo",
            timesteps=100000,
            n_envs=4,
            trials=5,
            parallel=2,
            bucket="test-bucket",
            image="test-image",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            search_space=search_space,
        )

        call_args = mock_aiplatform.CustomJob.call_args
        worker_specs = call_args[1]["worker_pool_specs"]
        trial_args = worker_specs[0]["container_spec"]["args"]
        output_idx = trial_args.index("--output-dir")
        output_dir = trial_args[output_idx + 1]
        assert output_dir == "/gcs/test-bucket/sweeps/velociraptor/stage2"


# ── GCS state key includes algorithm ───────────────────────────────


class TestGcsStateKeyIncludesAlgorithm:
    """The GCS state key must include the algorithm to avoid collisions."""

    def test_save_uses_algorithm_in_gcs_key(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = {"species": "velociraptor", "algorithm": "ppo", "stages": {}}

        mock_storage = MagicMock()
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket

        mock_gc = MagicMock()
        mock_gc.storage = mock_storage

        with patch.dict(
            "sys.modules",
            {"google.cloud.storage": mock_storage, "google.cloud": mock_gc, "google": MagicMock()},
        ):
            _save_sweep_state(state, "velociraptor", "ppo", bucket="my-bucket")

        # The blob key should contain the algorithm
        mock_bucket.blob.assert_called_once_with("sweeps/velociraptor/_sweep_state_ppo.json")

    def test_load_uses_algorithm_in_gcs_key(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        mock_storage = MagicMock()
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = json.dumps(
            {"species": "velociraptor", "algorithm": "sac", "stages": {}}
        )
        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        mock_gc = MagicMock()
        mock_gc.storage = mock_storage

        with patch.dict(
            "sys.modules",
            {"google.cloud.storage": mock_storage, "google.cloud": mock_gc, "google": MagicMock()},
        ):
            _load_sweep_state("velociraptor", "sac", bucket="my-bucket")

        mock_bucket.blob.assert_called_once_with("sweeps/velociraptor/_sweep_state_sac.json")


# ── _resolve_search_space logging ──────────────────────────────────


class TestResolveSearchSpaceLogging:
    def test_inline_json_logs_source(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            _resolve_search_space('{"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 1e-3}}', None, "ppo")
        assert "inline --search-space JSON" in caplog.text

    def test_default_space_logs_algorithm(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            _resolve_search_space(None, None, "sac")
        assert "default sac search space" in caplog.text


# ── _best_trial_model_path for stage 3 (no chain-forward needed) ──


# ── restart_job_on_worker_restart ──────────────────────────────────


class TestRestartJobOnWorkerRestart:
    def test_restart_passed_to_run(self):
        """restart_job_on_worker_restart=True is forwarded to hpt_job.run()."""
        from environments.shared.scripts.sweep import _submit_stage_sweep

        mock_aiplatform = MagicMock()
        mock_hpt = MagicMock()

        mock_job = MagicMock()
        mock_aiplatform.HyperparameterTuningJob.return_value = mock_job
        mock_aiplatform.CustomJob.return_value = MagicMock()

        search_space = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}}

        _submit_stage_sweep(
            aiplatform=mock_aiplatform,
            hpt_module=mock_hpt,
            species="velociraptor",
            stage=1,
            algorithm="ppo",
            timesteps=100000,
            n_envs=4,
            trials=5,
            parallel=2,
            bucket="test-bucket",
            image="test-image",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            search_space=search_space,
            restart_job_on_worker_restart=True,
        )

        mock_job.run.assert_called_once_with(sync=True, restart_job_on_worker_restart=True)

    def test_restart_not_passed_when_false(self):
        """restart_job_on_worker_restart=False (default) does not add the kwarg."""
        from environments.shared.scripts.sweep import _submit_stage_sweep

        mock_aiplatform = MagicMock()
        mock_hpt = MagicMock()

        mock_job = MagicMock()
        mock_aiplatform.HyperparameterTuningJob.return_value = mock_job
        mock_aiplatform.CustomJob.return_value = MagicMock()

        search_space = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}}

        _submit_stage_sweep(
            aiplatform=mock_aiplatform,
            hpt_module=mock_hpt,
            species="velociraptor",
            stage=1,
            algorithm="ppo",
            timesteps=100000,
            n_envs=4,
            trials=5,
            parallel=2,
            bucket="test-bucket",
            image="test-image",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            search_space=search_space,
        )

        mock_job.run.assert_called_once_with(sync=True)

    def test_cli_arg_parsed(self):
        """--restart-job-on-worker-restart is parsed correctly."""
        from environments.shared.scripts.sweep.__main__ import _build_parser

        parser = _build_parser()

        # launch mode
        args = parser.parse_args(
            [
                "launch",
                "--species",
                "velociraptor",
                "--project",
                "test",
                "--bucket",
                "b",
                "--image",
                "img",
                "--restart-job-on-worker-restart",
            ]
        )
        assert args.restart_job_on_worker_restart is True

        # launch mode with --no- prefix
        args = parser.parse_args(
            [
                "launch",
                "--species",
                "velociraptor",
                "--project",
                "test",
                "--bucket",
                "b",
                "--image",
                "img",
                "--no-restart-job-on-worker-restart",
            ]
        )
        assert args.restart_job_on_worker_restart is False

        # launch-all mode
        args = parser.parse_args(
            [
                "launch-all",
                "--species",
                "velociraptor",
                "--project",
                "test",
                "--bucket",
                "b",
                "--image",
                "img",
                "--restart-job-on-worker-restart",
            ]
        )
        assert args.restart_job_on_worker_restart is True

    def test_cli_default_is_false(self):
        """Default value for --restart-job-on-worker-restart is False."""
        from environments.shared.scripts.sweep.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "launch",
                "--species",
                "velociraptor",
                "--project",
                "test",
                "--bucket",
                "b",
                "--image",
                "img",
            ]
        )
        assert args.restart_job_on_worker_restart is False


class TestBestTrialModelPathStage3:
    def test_stage3_best_trial_identified(self):
        rows = [
            {"trial_id": "1", "stage_passed": True, "best_mean_reward": 100.0},
            {"trial_id": "2", "stage_passed": True, "best_mean_reward": 200.0},
        ]
        path, best_row = _best_trial_model_path(rows, "bucket", "velociraptor", 3)
        assert best_row["trial_id"] == "2"
        assert "stage3" in path


# ── collect_results_from_disk ───────────────────────────────────────────


def _setup_sweep_dir(base, stage, trials):
    """Create a sweep-style directory with metrics.json per trial.

    Args:
        base: Root path (tmp_path).
        stage: Stage number.
        trials: Dict mapping trial_id -> metrics dict.

    Returns:
        The stage directory path.
    """
    stage_dir = base / f"stage{stage}"
    for trial_id, metrics in trials.items():
        trial_dir = stage_dir / str(trial_id)
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)
    return stage_dir


class TestCollectResultsFromDisk:
    def test_sweep_layout(self, tmp_path):
        """Collects results from sweep-style trial directories."""
        _setup_sweep_dir(
            tmp_path,
            1,
            {
                "1": {"best_mean_reward": 150.0, "best_mean_episode_length": 500.0},
                "2": {"best_mean_reward": 80.0, "best_mean_episode_length": 300.0},
            },
        )
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 2
        rewards = {r["trial_id"]: r["best_mean_reward"] for r in rows}
        assert rewards["1"] == 150.0
        assert rewards["2"] == 80.0

    def test_curriculum_layout(self, tmp_path):
        """Collects results from curriculum-style single-trial directories."""
        for stage in (1, 2):
            stage_dir = tmp_path / f"stage{stage}"
            stage_dir.mkdir(parents=True)
            with open(stage_dir / "metrics.json", "w") as f:
                json.dump({"best_mean_reward": 100.0 * stage}, f)
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 2
        assert rows[0]["stage"] == 1
        assert rows[1]["stage"] == 2

    def test_stage_filter(self, tmp_path):
        """Only collects from specified stages."""
        _setup_sweep_dir(tmp_path, 1, {"1": {"best_mean_reward": 100.0}})
        _setup_sweep_dir(tmp_path, 2, {"1": {"best_mean_reward": 200.0}})
        rows = collect_results_from_disk(tmp_path, stages=[2])
        assert len(rows) == 1
        assert rows[0]["stage"] == 2

    def test_thresholds_from_stage_config(self, tmp_path):
        """Loads curriculum thresholds from stage_config.json."""
        stage_dir = _setup_sweep_dir(
            tmp_path,
            1,
            {
                "1": {"best_mean_reward": 50.0},
            },
        )
        cfg = {"curriculum": {"min_avg_reward": 100.0}}
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 1
        assert rows[0]["stage_passed"] is False
        assert rows[0]["reward_threshold"] == 100.0

    def test_passing_trial_with_threshold(self, tmp_path):
        """Trial passes when reward exceeds threshold."""
        stage_dir = _setup_sweep_dir(
            tmp_path,
            1,
            {
                "1": {"best_mean_reward": 150.0},
            },
        )
        cfg = {"curriculum": {"min_avg_reward": 100.0}}
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        assert rows[0]["stage_passed"] is True

    def test_empty_directory(self, tmp_path):
        """Returns empty list for directory with no stage sub-dirs."""
        rows = collect_results_from_disk(tmp_path)
        assert rows == []

    def test_nonexistent_directory(self, tmp_path):
        """Returns empty list for nonexistent directory."""
        rows = collect_results_from_disk(tmp_path / "does_not_exist")
        assert rows == []

    def test_multi_stage_collection(self, tmp_path):
        """Collects across multiple stages into a single list."""
        _setup_sweep_dir(
            tmp_path,
            1,
            {
                "1": {"best_mean_reward": 100.0},
                "2": {"best_mean_reward": 120.0},
            },
        )
        _setup_sweep_dir(
            tmp_path,
            2,
            {
                "1": {"best_mean_reward": 200.0},
            },
        )
        _setup_sweep_dir(
            tmp_path,
            3,
            {
                "1": {"best_mean_reward": 300.0},
            },
        )
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 4
        stages = [r["stage"] for r in rows]
        assert stages.count(1) == 2
        assert stages.count(2) == 1
        assert stages.count(3) == 1

    def test_hyperparameters_from_stage_config(self, tmp_path):
        """Includes algorithm hyperparameters and reward weights from stage_config.json."""
        stage_dir = _setup_sweep_dir(
            tmp_path,
            1,
            {"1": {"best_mean_reward": 100.0, "best_mean_episode_length": 400.0}},
        )
        cfg = {
            "algorithm": "PPO",
            "hyperparameters": {
                "learning_rate": 0.0003,
                "batch_size": 128,
                "n_epochs": 10,
                "policy_kwargs": {"net_arch": [256, 256]},
            },
            "reward_weights": {
                "alive_bonus": 2.0,
                "energy_penalty_weight": 0.05,
            },
            "curriculum": {"min_avg_reward": 50.0},
        }
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        # Algorithm hyperparameters prefixed with algorithm name
        assert row["ppo_learning_rate"] == 0.0003
        assert row["ppo_batch_size"] == 128
        assert row["ppo_n_epochs"] == 10
        # Nested policy_kwargs flattened
        assert row["ppo_policy_kwargs_net_arch"] == [256, 256]
        # Reward weights prefixed with env_
        assert row["env_alive_bonus"] == 2.0
        assert row["env_energy_penalty_weight"] == 0.05

    def test_hyperparameters_from_per_trial_config(self, tmp_path):
        """Falls back to per-trial stage_config.json for hyperparameters."""
        stage_dir = tmp_path / "stage1"
        trial_dir = stage_dir / "1"
        trial_dir.mkdir(parents=True)
        with open(trial_dir / "metrics.json", "w") as f:
            json.dump({"best_mean_reward": 100.0}, f)
        cfg = {
            "algorithm": "SAC",
            "hyperparameters": {"learning_rate": 0.001, "gamma": 0.99},
            "reward_weights": {"alive_bonus": 1.5},
            "curriculum": {"min_avg_reward": 50.0},
        }
        with open(trial_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["sac_learning_rate"] == 0.001
        assert row["sac_gamma"] == 0.99
        assert row["env_alive_bonus"] == 1.5

    def test_hyperparameters_in_csv_output(self, tmp_path):
        """Hyperparameters appear as columns in the written CSV."""
        stage_dir = _setup_sweep_dir(
            tmp_path,
            1,
            {"1": {"best_mean_reward": 100.0}},
        )
        cfg = {
            "algorithm": "PPO",
            "hyperparameters": {"learning_rate": 0.0003, "batch_size": 256},
            "reward_weights": {"alive_bonus": 2.0},
            "curriculum": {"min_avg_reward": 50.0},
        }
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        csv_path = tmp_path / "collected_results.csv"
        write_results_csv(rows, csv_path)
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 1
        assert "ppo_learning_rate" in csv_rows[0]
        assert "ppo_batch_size" in csv_rows[0]
        assert "env_alive_bonus" in csv_rows[0]
        assert csv_rows[0]["ppo_learning_rate"] == "0.0003"
        assert csv_rows[0]["ppo_batch_size"] == "256"
        assert csv_rows[0]["env_alive_bonus"] == "2.0"


# ── collect_results_from_disk with gs:// URIs ──────────────────────────


def _make_mock_gcs_blobs(file_map):
    """Build mock GCS blobs from a dict of {blob_name: content_bytes}.

    Returns a list of mock blob objects whose ``download_to_filename``
    writes the content to the requested local path.
    """
    blobs = []
    for name, content in file_map.items():
        blob = MagicMock()
        blob.name = name

        def _download(local_path, _content=content):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(_content)

        blob.download_to_filename = _download
        blobs.append(blob)
    return blobs


class TestCollectResultsFromDiskGCS:
    def _gcs_modules(self, mock_storage):
        """Build sys.modules dict for mocking google.cloud.storage."""
        mock_gc = MagicMock()
        mock_gc.storage = mock_storage
        return {
            "google": MagicMock(),
            "google.cloud": mock_gc,
            "google.cloud.storage": mock_storage,
        }

    def test_gs_uri_downloads_and_collects(self):
        """gs:// URI triggers GCS download and produces correct rows."""
        metrics_1 = json.dumps({"best_mean_reward": 150.0}).encode()
        metrics_2 = json.dumps({"best_mean_reward": 200.0}).encode()
        stage_cfg = json.dumps({"curriculum": {"min_avg_reward": 100.0}}).encode()

        file_map = {
            "sweeps/velociraptor/stage1/1/metrics.json": metrics_1,
            "sweeps/velociraptor/stage1/stage_config.json": stage_cfg,
            "sweeps/velociraptor/stage2/1/metrics.json": metrics_2,
            # Also include a non-JSON file that should be skipped
            "sweeps/velociraptor/stage1/1/best_model.zip": b"fake_model",
        }
        mock_blobs = _make_mock_gcs_blobs(file_map)

        mock_storage = MagicMock()
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = mock_blobs

        with patch.dict("sys.modules", self._gcs_modules(mock_storage)):
            rows = collect_results_from_disk(
                "gs://my-bucket/sweeps/velociraptor",
                species="velociraptor",
            )

        assert len(rows) == 2
        stage_nums = {r["stage"] for r in rows}
        assert stage_nums == {1, 2}
        # Stage 1 has a stage_config.json with threshold
        s1_row = [r for r in rows if r["stage"] == 1][0]
        assert s1_row["reward_threshold"] == 100.0
        assert s1_row["stage_passed"] is True

    def test_gs_uri_cleans_up_tempdir_on_success(self):
        """Temp directory is cleaned up after successful collection."""
        metrics = json.dumps({"best_mean_reward": 100.0}).encode()
        file_map = {"prefix/stage1/1/metrics.json": metrics}
        mock_blobs = _make_mock_gcs_blobs(file_map)

        mock_storage = MagicMock()
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = mock_blobs

        created_tmpdirs = []
        original_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        with (
            patch.dict("sys.modules", self._gcs_modules(mock_storage)),
            patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp),
        ):
            rows = collect_results_from_disk("gs://bucket/prefix")

        assert len(rows) == 1
        # Temp dir should have been cleaned up
        assert len(created_tmpdirs) == 1
        assert not Path(created_tmpdirs[0]).exists()

    def test_gs_uri_cleans_up_tempdir_on_error(self):
        """Temp directory is cleaned up even if GCS download fails."""
        mock_storage = MagicMock()
        mock_storage.Client.side_effect = Exception("auth failed")

        created_tmpdirs = []
        original_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        with (
            patch.dict("sys.modules", self._gcs_modules(mock_storage)),
            patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp),
        ):
            with pytest.raises(Exception, match="auth failed"):
                collect_results_from_disk("gs://bucket/prefix")

        assert len(created_tmpdirs) == 1
        assert not Path(created_tmpdirs[0]).exists()
