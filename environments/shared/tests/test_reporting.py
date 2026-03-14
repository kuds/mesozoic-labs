"""Tests for the training reporting utilities."""

import csv
import json

import pytest

from environments.shared.reporting import (
    CSV_METRIC_COLUMNS,
    _compute_fieldnames,
    build_stage_results_from_eval_data,
    format_duration,
    format_duration_hms,
    save_results_json,
    write_results_csv,
    write_stage_summary,
    write_training_summary,
)

# ── CSV_METRIC_COLUMNS ──────────────────────────────────────────────────


class TestCsvMetricColumns:
    """Tests for CSV_METRIC_COLUMNS schema."""

    def test_contains_distance_traveled(self):
        assert "mean_distance_traveled" in CSV_METRIC_COLUMNS

    def test_contains_forward_vel(self):
        assert "mean_forward_vel" in CSV_METRIC_COLUMNS


# ── _compute_fieldnames ─────────────────────────────────────────────────


class TestComputeFieldnames:
    """Tests for _compute_fieldnames column ordering."""

    def test_fixed_columns_come_first(self):
        rows = [{"trial_id": "1", "stage": 1, "best_mean_reward": 10.0}]
        cols = _compute_fieldnames(rows, fixed_columns=["trial_id", "stage"])
        assert cols[0] == "trial_id"
        assert cols[1] == "stage"

    def test_hparams_sorted_between_fixed_and_metrics(self):
        rows = [{"trial_id": "1", "stage": 1, "ppo_lr": 0.001, "env_bonus": 1.0}]
        cols = _compute_fieldnames(rows, fixed_columns=["trial_id", "stage"])
        hparam_start = 2  # after fixed cols
        metric_start = cols.index("best_mean_reward")
        assert cols[hparam_start:metric_start] == ["env_bonus", "ppo_lr"]

    def test_eval_columns_come_last(self):
        rows = [{"trial_id": "1", "stage": 1, "eval_spin": 0.1, "eval_heading": 0.5}]
        cols = _compute_fieldnames(rows, fixed_columns=["trial_id", "stage"])
        assert cols[-2:] == ["eval_heading", "eval_spin"]

    def test_no_fixed_columns(self):
        rows = [{"best_mean_reward": 10.0, "ppo_lr": 0.001}]
        cols = _compute_fieldnames(rows)
        assert cols[0] == "ppo_lr"  # hparam sorted first
        assert "best_mean_reward" in cols


# ── write_results_csv ───────────────────────────────────────────────────


class TestWriteResultsCsv:
    """Tests for the unified write_results_csv writer."""

    def test_batch_write_creates_file(self, tmp_path):
        rows = [{"trial_id": "t1", "stage": 1, "best_mean_reward": 50.0}]
        path = write_results_csv(rows, tmp_path / "out.csv", fixed_columns=["trial_id", "stage"])
        assert path.exists()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            result = list(reader)
        assert len(result) == 1
        assert result[0]["trial_id"] == "t1"
        assert result[0]["best_mean_reward"] == "50.0"

    def test_column_ordering(self, tmp_path):
        rows = [
            {
                "trial_id": "t1",
                "stage": 1,
                "ppo_lr": 0.001,
                "env_bonus": 1.0,
                "best_mean_reward": 50.0,
                "stage_passed": True,
                "eval_spin": 0.1,
            }
        ]
        path = write_results_csv(rows, tmp_path / "out.csv", fixed_columns=["trial_id", "stage"])
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        # Fixed columns first
        assert fieldnames[0] == "trial_id"
        assert fieldnames[1] == "stage"
        # Hparams sorted next
        hparam_end = fieldnames.index("best_mean_reward")
        assert sorted(fieldnames[2:hparam_end]) == fieldnames[2:hparam_end]
        # Metric columns in canonical order
        metric_section = fieldnames[hparam_end : hparam_end + len(CSV_METRIC_COLUMNS)]
        assert metric_section == CSV_METRIC_COLUMNS
        # eval_* at the end
        assert fieldnames[-1] == "eval_spin"

    def test_empty_rows_returns_path_without_creating(self, tmp_path):
        path = write_results_csv([], tmp_path / "out.csv")
        assert not (tmp_path / "out.csv").exists()
        assert path == tmp_path / "out.csv"

    def test_append_creates_new_file(self, tmp_path):
        rows = [{"stage": 1, "best_mean_reward": 10.0}]
        path = write_results_csv(rows, tmp_path / "out.csv", append=True)
        assert path.exists()
        with open(path, newline="") as f:
            result = list(csv.DictReader(f))
        assert len(result) == 1

    def test_append_adds_rows(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        write_results_csv([{"stage": 1, "best_mean_reward": 10.0}], csv_path, append=True)
        write_results_csv([{"stage": 2, "best_mean_reward": 20.0}], csv_path, append=True)
        with open(csv_path, newline="") as f:
            result = list(csv.DictReader(f))
        assert len(result) == 2
        assert result[0]["stage"] == "1"
        assert result[1]["stage"] == "2"

    def test_append_expands_header(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        write_results_csv([{"stage": 1, "best_mean_reward": 10.0}], csv_path, append=True)
        write_results_csv(
            [{"stage": 2, "best_mean_reward": 20.0, "ppo_lr": 0.001}],
            csv_path,
            append=True,
        )
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            assert "ppo_lr" in reader.fieldnames
            result = list(reader)
        assert len(result) == 2
        # First row should have empty value for the new column
        assert result[0]["ppo_lr"] == ""
        assert result[1]["ppo_lr"] == "0.001"

    def test_append_gcs_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Append mode is not supported"):
            write_results_csv([{"stage": 1}], "gs://bucket/file.csv", append=True)

    def test_creates_parent_directories(self, tmp_path):
        rows = [{"stage": 1, "best_mean_reward": 10.0}]
        path = write_results_csv(rows, tmp_path / "a" / "b" / "out.csv")
        assert path.exists()

    def test_multiple_rows(self, tmp_path):
        rows = [
            {"trial_id": "t1", "stage": 1, "best_mean_reward": 50.0},
            {"trial_id": "t2", "stage": 1, "best_mean_reward": 60.0},
            {"trial_id": "t3", "stage": 2, "best_mean_reward": 70.0},
        ]
        path = write_results_csv(rows, tmp_path / "out.csv", fixed_columns=["trial_id", "stage"])
        with open(path, newline="") as f:
            result = list(csv.DictReader(f))
        assert len(result) == 3

    def test_extrasaction_ignore(self, tmp_path):
        """Extra keys not in fieldnames should be silently ignored."""
        rows = [{"stage": 1, "best_mean_reward": 10.0, "_internal_flag": True}]
        # _internal_flag is not a known column type, so it becomes a hparam
        path = write_results_csv(rows, tmp_path / "out.csv")
        assert path.exists()


# ── format_duration ──────────────────────────────────────────────────────


class TestFormatDuration:
    """Tests for format_duration (human-readable duration)."""

    def test_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_duration(130) == "2m 10s"

    def test_hours_minutes_seconds(self):
        assert format_duration(3661) == "1h 1m 1s"

    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_exact_hour(self):
        assert format_duration(3600) == "1h 0m 0s"

    def test_exact_minute(self):
        assert format_duration(60) == "1m 0s"

    def test_fractional_seconds(self):
        # Fractional seconds should be truncated (int conversion)
        assert format_duration(59.9) == "59s"


# ── format_duration_hms ─────────────────────────────────────────────────


class TestFormatDurationHms:
    """Tests for format_duration_hms (H:MM:SS format)."""

    def test_zero(self):
        assert format_duration_hms(0) == "0:00:00"

    def test_seconds_only(self):
        assert format_duration_hms(45) == "0:00:45"

    def test_minutes_and_seconds(self):
        assert format_duration_hms(130) == "0:02:10"

    def test_hours_minutes_seconds(self):
        assert format_duration_hms(3661) == "1:01:01"

    def test_large_value(self):
        assert format_duration_hms(36000) == "10:00:00"


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_stage_result(stage=1, **overrides):
    """Build a minimal stage result dict for testing."""
    result = {
        "stage": stage,
        "name": f"Stage {stage}",
        "description": f"Description for stage {stage}",
        "timesteps": 100_000 * stage,
        "duration_seconds": 300.0 * stage,
        "mean_reward": 50.0 + stage * 10,
        "std_reward": 5.0,
        "mean_episode_length": 200.0,
        "std_episode_length": 20.0,
        "mean_forward_vel": 0.5 * stage,
        "std_forward_vel": 0.1,
        "mean_success_rate": 0.0,
        "model_path": f"/tmp/stage{stage}/best_model",
        "vecnorm_path": f"/tmp/stage{stage}/vecnorm.pkl",
        "sim_dt": 0.01,
        "gate_passed": True,
        "gate_failures": [],
    }
    result.update(overrides)
    return result


# ── write_stage_summary ──────────────────────────────────────────────────


class TestWriteStageSummary:
    """Tests for write_stage_summary (per-stage text files)."""

    def test_creates_summary_file(self, tmp_path):
        result = _make_stage_result(1)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        assert path.exists()
        assert path.name == "stage_summary.txt"

    def test_returns_path(self, tmp_path):
        result = _make_stage_result(1)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        assert path == tmp_path / "stage_summary.txt"

    def test_contains_species(self, tmp_path):
        result = _make_stage_result(1)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Velociraptor" in text

    def test_contains_algorithm(self, tmp_path):
        result = _make_stage_result(1)
        path = write_stage_summary(tmp_path, result, "trex", "SAC")
        text = path.read_text()
        assert "SAC" in text

    def test_contains_stage_number(self, tmp_path):
        result = _make_stage_result(2)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Stage 2" in text

    def test_contains_reward(self, tmp_path):
        result = _make_stage_result(1, mean_reward=123.45, std_reward=6.78)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "123.45" in text
        assert "6.78" in text

    def test_contains_forward_velocity(self, tmp_path):
        result = _make_stage_result(1, mean_forward_vel=1.23, std_forward_vel=0.45)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "1.23" in text

    def test_includes_best_eval_when_present(self, tmp_path):
        result = _make_stage_result(1, best_eval_reward=99.5, best_eval_std=3.2, best_eval_timestep=50000)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "99.5" in text
        assert "50,000 steps" in text

    def test_includes_best_model_section(self, tmp_path):
        result = _make_stage_result(
            1,
            best_model_reward=88.0,
            best_model_std_reward=2.0,
            best_model_length=180.5,
            best_model_std_length=15.0,
            best_model_fwd_vel=1.1,
            best_model_std_fwd_vel=0.2,
            best_model_success_rate=0.75,
        )
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Best Model Evaluation" in text
        assert "88.0" in text
        assert "75%" in text

    def test_no_best_eval_when_empty(self, tmp_path):
        result = _make_stage_result(1, best_eval_reward="", best_eval_std="")
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Best eval:" not in text

    def test_includes_best_eval_length(self, tmp_path):
        result = _make_stage_result(
            1,
            best_eval_reward=80.0,
            best_eval_std=3.0,
            best_eval_timestep=40000,
            best_eval_length=250.0,
            best_eval_std_length=10.0,
        )
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Best ep length:" in text


# ── write_training_summary ───────────────────────────────────────────────


class TestWriteTrainingSummary:
    """Tests for write_training_summary (overall training text file)."""

    def test_creates_summary_file(self, tmp_path):
        results = [_make_stage_result(1), _make_stage_result(2)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        assert path.exists()
        assert path.name == "training_summary.txt"

    def test_returns_path(self, tmp_path):
        results = [_make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        assert path == tmp_path / "training_summary.txt"

    def test_contains_species(self, tmp_path):
        results = [_make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "trex", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Trex" in text

    def test_contains_algorithm(self, tmp_path):
        results = [_make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "velociraptor", "SAC", seed=42, n_envs=4)
        text = path.read_text()
        assert "SAC" in text

    def test_contains_seed(self, tmp_path):
        results = [_make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=123, n_envs=4)
        text = path.read_text()
        assert "123" in text

    def test_contains_all_stages(self, tmp_path):
        results = [_make_stage_result(1), _make_stage_result(2), _make_stage_result(3)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Stage 1" in text
        assert "Stage 2" in text
        assert "Stage 3" in text

    def test_contains_total_training_time(self, tmp_path):
        results = [_make_stage_result(1, duration_seconds=60.0), _make_stage_result(2, duration_seconds=120.0)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Total training time:" in text
        assert "3m" in text

    def test_contains_quick_test_flag(self, tmp_path):
        results = [_make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4, quick_test=True)
        text = path.read_text()
        assert "True" in text

    def test_includes_best_eval_when_present(self, tmp_path):
        results = [_make_stage_result(1, best_eval_reward=95.0, best_eval_std=2.5, best_eval_timestep=80000)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Best eval:" in text
        assert "95.0" in text

    def test_no_best_eval_when_empty_string(self, tmp_path):
        results = [_make_stage_result(1, best_eval_reward="")]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Best eval:" not in text


# ── save_results_json ────────────────────────────────────────────────────


class TestSaveResultsJson:
    """Tests for save_results_json (machine-readable JSON output)."""

    def test_creates_json_file(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        assert path.exists()
        assert path.name == "summary.json"

    def test_returns_path(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        assert path == tmp_path / "summary.json"

    def test_valid_json(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_contains_species(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["species"] == "velociraptor"

    def test_contains_algorithm(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "SAC", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["algorithm"] == "SAC"

    def test_contains_seed(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=123, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["seed"] == 123

    def test_contains_date(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert "date" in data

    def test_stages_match_input(self, tmp_path):
        results = [_make_stage_result(1), _make_stage_result(2), _make_stage_result(3)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert set(data["stages"].keys()) == {"1", "2", "3"}

    def test_total_timesteps(self, tmp_path):
        results = [_make_stage_result(1, timesteps=100_000), _make_stage_result(2, timesteps=200_000)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["total_timesteps"] == 300_000

    def test_total_training_time(self, tmp_path):
        results = [_make_stage_result(1, duration_seconds=100.0), _make_stage_result(2, duration_seconds=200.0)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["total_training_time_seconds"] == 300.0

    def test_final_avg_reward(self, tmp_path):
        results = [_make_stage_result(1, mean_reward=50.0), _make_stage_result(2, mean_reward=75.123)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["final_avg_reward"] == 75.12

    def test_forward_vel_in_stage_data(self, tmp_path):
        results = [_make_stage_result(1, mean_forward_vel=1.5)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["stages"]["1"]["avg_forward_vel"] == 1.5

    def test_creates_results_dir_if_needed(self, tmp_path):
        nested_dir = tmp_path / "a" / "b" / "c"
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=nested_dir)
        assert path.exists()

    def test_accepts_string_path(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=str(tmp_path))
        assert path.exists()

    def test_stage_data_has_training_time(self, tmp_path):
        results = [_make_stage_result(1, duration_seconds=3661.0)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        stage = data["stages"]["1"]
        assert "training_time" in stage
        assert "training_time_seconds" in stage
        assert stage["training_time"] == "1:01:01"


# ── build_stage_results_from_eval_data ──────────────────────────────────


class TestBuildStageResultsFromEvalData:
    """Tests for build_stage_results_from_eval_data."""

    def test_builds_from_evaluations_npz(self, tmp_path):
        import numpy as np

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        rewards = np.array([[10.0, 12.0], [20.0, 22.0], [15.0, 17.0]])
        lengths = np.array([[100, 110], [200, 210], [150, 160]])
        timesteps = np.array([50000, 100000, 150000])
        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=rewards,
            ep_lengths=lengths,
            timesteps=timesteps,
        )

        config = {
            "name": "Balance",
            "description": "Stand up",
            "env_kwargs": {"sim_dt": 0.02},
        }

        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=1,
            stage_config=config,
            timesteps=150_000,
        )

        assert result["stage"] == 1
        assert result["name"] == "Balance"
        assert result["timesteps"] == 150_000
        assert result["sim_dt"] == 0.02
        # Best eval is at index 1 (mean 21.0)
        assert result["best_eval_reward"] == 21.0
        assert result["best_eval_timestep"] == 100000
        # Last eval used as final metrics (mean 16.0)
        assert result["mean_reward"] == 16.0
        # Forward vel defaults to 0 (requires live eval)
        assert result["mean_forward_vel"] == 0.0

    def test_reads_duration_from_metrics_json(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        (tmp_path / "metrics.json").write_text(json.dumps({"training_duration_seconds": 123.4}))

        config = {"name": "Loco", "description": "Walk", "env_kwargs": {}}
        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=2,
            stage_config=config,
            timesteps=50_000,
        )
        assert result["duration_seconds"] == 123.4

    def test_explicit_duration_overrides_metrics_json(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        (tmp_path / "metrics.json").write_text(json.dumps({"training_duration_seconds": 123.4}))

        config = {"name": "Loco", "description": "Walk", "env_kwargs": {}}
        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=2,
            stage_config=config,
            timesteps=50_000,
            duration_seconds=999.0,
        )
        assert result["duration_seconds"] == 999.0

    def test_no_eval_data_returns_defaults(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        config = {"name": "Balance", "description": "Stand", "env_kwargs": {}}
        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=1,
            stage_config=config,
            timesteps=100_000,
        )
        assert result["mean_reward"] == 0.0
        assert result["best_eval_reward"] == ""
        assert result["duration_seconds"] == 0.0
