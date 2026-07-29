"""Tests for the training reporting utilities."""

import csv
import json

import pytest

from environments.shared.plant_contract import MODEL_IDENTITY_ATTRIBUTE, PlantIdentity
from environments.shared.reporting import (
    CSV_METRIC_COLUMNS,
    _compute_fieldnames,
    build_stage_results_from_eval_data,
    evaluate_recorded_gate,
    format_duration,
    format_duration_hms,
    parse_optional_bool,
    save_jax_stage_artifacts,
    save_results_csv,
    save_results_json,
    write_results_csv,
    write_stage_summary,
    write_training_summary,
)


def _plant_identity():
    return PlantIdentity(
        species="velociraptor",
        model_path="environments/velociraptor/assets/raptor.xml",
        physics_revision=1,
        policy_interface_revision=1,
        visual_revision=1,
        source_closure_sha256="sha256:" + "1" * 64,
        policy_interface_sha256="sha256:" + "2" * 64,
        physics_sha256="sha256:" + "3" * 64,
        visual_sha256="sha256:" + "4" * 64,
        nq=31,
        nv=30,
        nu=22,
        observation_dim=67,
        action_dim=22,
    )


class TestStrictRecordedGate:
    def test_requires_every_enabled_metric_and_consecutive_windows(self):
        curriculum = {
            "min_avg_reward": 100.0,
            "min_avg_forward_vel": 2.0,
            "min_eval_episodes": 10,
            "required_consecutive": 2,
        }
        incomplete = [
            {"mean_reward": 120.0, "n_episodes": 10},
            {"mean_reward": 130.0, "n_episodes": 10},
        ]
        assert evaluate_recorded_gate(curriculum, incomplete) is None

        complete = [
            {"mean_reward": 120.0, "mean_forward_vel": 2.1, "n_episodes": 10},
            {"mean_reward": 130.0, "mean_forward_vel": 2.2, "n_episodes": 10},
        ]
        assert evaluate_recorded_gate(curriculum, complete) is True

    def test_returns_false_when_complete_history_never_passes(self):
        curriculum = {"min_avg_reward": 100.0, "required_consecutive": 2}
        evaluations = [
            {"mean_reward": 120.0, "n_episodes": 10},
            {"mean_reward": 90.0, "n_episodes": 10},
        ]
        assert evaluate_recorded_gate(curriculum, evaluations) is False

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, True), (False, False), ("true", True), ("False", False), (1, True), (0, False), ("", None)],
    )
    def test_parses_serialized_booleans_strictly(self, value, expected):
        assert parse_optional_bool(value) is expected


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
        "publication_gate_passed": True,
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

    def test_provenance_records_repository_commit(self, tmp_path):
        results = [_make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        prov = json.loads(path.read_text())["provenance"]
        assert set(prov) == {
            "model_revision_status",
            "verification_status",
            "evaluation_episodes",
            "repository_commit",
            "model_hash",
            "config_hash",
        }
        # Repository commit is auto-populated; a fresh run stays conservatively uncertified.
        assert isinstance(prov["repository_commit"], str) and prov["repository_commit"]
        assert prov["model_revision_status"] == "historical"
        assert prov["verification_status"] == "unverified"

    def test_provenance_overrides_are_applied(self, tmp_path):
        results = [_make_stage_result(1)]
        override = {
            "model_hash": "sha256:abc",
            "config_hash": "sha256:def",
            "evaluation_episodes": 30,
            "model_revision_status": "current",
            "verification_status": "verified",
        }
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path, provenance=override)
        prov = json.loads(path.read_text())["provenance"]
        assert prov["model_hash"] == "sha256:abc"
        assert prov["config_hash"] == "sha256:def"
        assert prov["evaluation_episodes"] == 30
        assert prov["model_revision_status"] == "current"
        assert prov["verification_status"] == "verified"
        assert prov["repository_commit"]  # still auto-filled when not overridden

    def test_backend_fields_present(self, tmp_path):
        results = [_make_stage_result(1)]
        data = json.loads(save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path).read_text())
        assert data["backend"] == "stable-baselines3"
        assert "backend_version" in data

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

    def test_includes_plant_identity_from_final_stage(self, tmp_path):
        identity = _plant_identity().to_dict()
        results = [_make_stage_result(1, plant_identity=identity)]

        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)

        assert json.loads(path.read_text())["plant_identity"] == identity

    def test_standard_csv_flattens_plant_provenance(self, tmp_path):
        identity = _plant_identity().to_dict()
        results = [_make_stage_result(1, plant_identity=identity)]
        configs = {
            1: {
                "env_kwargs": {},
                "ppo_kwargs": {},
                "curriculum_kwargs": {},
            }
        }

        path = save_results_csv(results, configs, "velociraptor", "PPO", 42, tmp_path)

        with path.open() as source:
            row = next(csv.DictReader(source))
        assert row["plant_physics_sha256"] == identity["physics_sha256"]
        assert row["plant_policy_interface_sha256"] == identity["policy_interface_sha256"]


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

    def test_reads_plant_identity_from_metrics_json(self, tmp_path):
        (tmp_path / "models").mkdir()
        identity = _plant_identity().to_dict()
        (tmp_path / "metrics.json").write_text(json.dumps({"plant_identity": identity}))

        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=1,
            stage_config={"name": "Balance", "description": "Stand", "env_kwargs": {}},
            timesteps=100,
        )

        assert result["plant_identity"] == identity

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


# ── save_jax_stage_artifacts ─────────────────────────────────────────────


class _FakeEvalResults:
    """Minimal stand-in for jax_eval.EvalResults."""

    def __init__(self):
        self.rewards = [50.0, 55.0, 60.0]
        self.lengths = [200, 210, 220]
        self.forward_vels = [0.5, 0.6, 0.55]
        self.distances = [1.0, 1.2, 1.1]
        self.successes = [True, True, True]
        self.diag_tilt = [0.1, 0.2, 0.15]
        self.diag_fwd_vel = [0.5, 0.6, 0.55]
        self.diag_pelvis_h = [0.7, 0.72, 0.71]
        self.diag_energy = [0.01, 0.02, 0.015]
        self.diag_l_foot = [1.0, 0.0, 1.0]
        self.diag_r_foot = [0.0, 1.0, 0.0]
        self.diag_reward_components = {
            "forward": [0.3, 0.4, 0.35],
            "alive": [0.1, 0.1, 0.1],
        }
        self.diag_reward_diagnostics = {
            "bilateral_support_quality": [0.4, 0.6, 0.8],
            "alive_gate": [0.7, 0.8, 0.9],
        }


class _FakeFinalEvalResults:
    """Terminal-policy evidence deliberately different from the selected policy."""

    def __init__(self):
        self.rewards = [10.0, 15.0, 20.0]
        self.lengths = [100, 110, 120]
        self.forward_vels = [0.1, 0.2, 0.3]
        self.distances = [0.5, 1.0, 1.5]
        self.successes = [False, False, True]


class TestSaveJaxStageArtifacts:
    """Tests for save_jax_stage_artifacts."""

    @pytest.fixture(autouse=True)
    def _clean_repository_state(self, monkeypatch):
        from environments.shared import result_bundle

        monkeypatch.setattr(
            result_bundle,
            "_repository_state",
            lambda _root: {
                "repository_url": "https://github.com/kuds/mesozoic-labs.git",
                "repository_commit": "a" * 40,
                "repository_dirty": False,
                "repository_patch_sha256": None,
            },
        )

    def _call(self, tmp_path, stage=1, species="velociraptor", **overrides):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        stage_dir = run_dir / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_config = {
            "name": f"Stage {stage}",
            "description": f"Curriculum stage {stage}",
            "env_kwargs": {"forward_vel_weight": float(stage)},
            "jax_kwargs": {"learning_rate": 3e-4},
            "curriculum_kwargs": {"min_avg_reward": 50.0},
        }
        stage_results = _make_stage_result(
            stage=stage,
            model_path=str(stage_dir / "models" / "best_model.pkl"),
            best_eval_reward=55.0,
            best_eval_timestep=50000,
            mean_distance_traveled=2.5,
        )

        kwargs = dict(
            species=species,
            stage=stage,
            stage_config=stage_config,
            stage_results=stage_results,
            stage_dir=stage_dir,
            run_dir=run_dir,
            eval_results=_FakeEvalResults(),
            params={"dense": [1.0, 2.0]},
            obs_rms=None,
            seed=42,
            num_envs=2048,
            reward_cfg={"forward_vel_weight": 1.0},
            best_params={"dense": [3.0, 4.0]},
            best_reward=55.0,
            best_update=10,
            plant_identity=_plant_identity(),
        )
        kwargs.update(overrides)
        return save_jax_stage_artifacts(**kwargs), stage_dir, run_dir

    def test_returns_all_artifact_paths(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        expected_keys = {
            "stage_summary",
            "stage_config",
            "evaluation_episodes",
            "final_evaluation_episodes",
            "collected_results_csv",
            "diagnostics",
            "best_model",
            "final_model",
            "stage_result",
            "training_summary",
            "provenance",
            "artifact_manifest",
        }
        assert set(paths.keys()) == expected_keys

    def test_all_files_exist(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        for name, path in paths.items():
            assert path.exists(), f"{name} not found at {path}"

    def test_stage_summary_content(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        text = paths["stage_summary"].read_text()
        assert "Velociraptor" in text
        assert "Stage 1" in text

    def test_stage_config_json(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        data = json.loads(paths["stage_config"].read_text())
        assert data["species"] == "velociraptor"
        assert data["stage"] == 1
        assert data["algorithm"] == "JAX_PPO"
        assert data["plant_identity"] == _plant_identity().to_dict()
        assert (paths["stage_config"].parent / "plant_identity.json").exists()

    def test_collected_results_csv(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        with open(paths["collected_results_csv"]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["species"] == "velociraptor"
        assert rows[0]["algorithm"] == "PPO"
        assert rows[0]["backend"] == "jax-mjx"
        assert rows[0]["plant_physics_sha256"] == _plant_identity().physics_sha256

    def test_diagnostics_npz(self, tmp_path):
        import numpy as np

        paths, _, _ = self._call(tmp_path)
        data = np.load(paths["diagnostics"])
        assert "tilt_angle" in data
        assert "timesteps" in data
        assert "forward_vel" in data
        assert "l_foot_contact" in data
        assert "reward_forward" in data
        assert "bilateral_support_quality" in data
        assert "alive_gate" in data
        assert all(len(data[key]) == len(data["timesteps"]) for key in data.files)

    def test_trex_stage1_diagnostics_derive_support_duty_and_load_share(self, tmp_path):
        from dataclasses import replace

        import numpy as np

        trex_identity = replace(
            _plant_identity(),
            species="trex",
            model_path="environments/trex/assets/trex.xml",
            physics_revision=5,
            policy_interface_revision=7,
            nq=28,
            nv=27,
            nu=21,
            observation_dim=61,
            action_dim=21,
        )
        paths, _, _ = self._call(
            tmp_path,
            species="trex",
            plant_identity=trex_identity,
        )
        data = np.load(paths["diagnostics"])

        np.testing.assert_array_equal(data["r_foot_contact_duty"], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(data["l_foot_contact_duty"], [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(data["bilateral_support_duty"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(data["single_support_duty"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(data["unsupported_duty"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data["r_foot_load_share"], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(data["l_foot_load_share"], [1.0, 0.0, 1.0])
        np.testing.assert_allclose(data["foot_load_balance"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data["foot_load_asymmetry"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(data["foot_load_imbalance"], [1.0, 1.0, 1.0])

    def test_diagnostics_npz_no_foot_data(self, tmp_path):
        import numpy as np

        eval_results = _FakeEvalResults()
        eval_results.diag_l_foot = []
        eval_results.diag_r_foot = []

        paths, _, _ = self._call(tmp_path, eval_results=eval_results)
        data = np.load(paths["diagnostics"])
        assert "tilt_angle" in data
        assert "l_foot_contact" not in data

    def test_best_model_checkpoint(self, tmp_path):
        import pickle

        paths, _, _ = self._call(tmp_path)
        with open(paths["best_model"], "rb") as f:
            ckpt = pickle.load(f)  # noqa: S301
        assert ckpt["params"] == {"dense": [3.0, 4.0]}
        assert ckpt["best_reward"] == 55.0
        assert ckpt[MODEL_IDENTITY_ATTRIBUTE] == _plant_identity().to_dict()

    def test_final_model_checkpoint(self, tmp_path):
        import pickle

        paths, _, _ = self._call(tmp_path)
        with open(paths["final_model"], "rb") as f:
            ckpt = pickle.load(f)  # noqa: S301
        assert ckpt["params"] == {"dense": [1.0, 2.0]}
        assert ckpt[MODEL_IDENTITY_ATTRIBUTE] == _plant_identity().to_dict()

    def test_training_summary_content(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        text = paths["training_summary"].read_text()
        assert "Velociraptor" in text
        assert "JAX/MJX PPO" in text
        assert (paths["training_summary"].parent / "plant_identity.json").exists()

    def test_partial_bundle_has_no_public_summary(self, tmp_path):
        paths, _, run_dir = self._call(tmp_path)
        from environments.shared.result_bundle import validate_result_bundle

        assert "summary" not in paths
        assert not (run_dir / "summary.json").exists()
        assert validate_result_bundle(run_dir, require_complete=False)["status"] == "partial"

    def test_csv_upserts_across_stages(self, tmp_path):
        """Each stage is represented once in the regenerated canonical CSV."""
        _, _, run_dir = self._call(tmp_path)
        self._call(tmp_path, stage=2)
        with open(run_dir / "collected_results.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["stage"] == "1"
        assert rows[1]["stage"] == "2"

    def test_repeated_stage_save_is_idempotent(self, tmp_path):
        _, _, run_dir = self._call(tmp_path)
        self._call(tmp_path)

        with open(run_dir / "collected_results.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["stage"] == "1"

    def test_rejects_a_skipped_prior_stage_before_writing(self, tmp_path):
        from environments.shared.result_bundle import ResultBundleError

        with pytest.raises(ResultBundleError, match="contiguous prefix"):
            self._call(tmp_path, stage=2)
        assert not (tmp_path / "run" / "provenance.json").exists()

    def test_preserves_preinitialized_colab_hardware_identity(self, tmp_path):
        from environments.shared.result_bundle import initialize_result_bundle

        run_dir = tmp_path / "run"
        initialize_result_bundle(
            run_dir,
            species="velociraptor",
            algorithm="JAX_PPO",
            backend="jax-mjx",
            seed=42,
            evaluation_seeds=[42],
            evaluation_episodes=3,
            parallel_envs=2048,
            hardware="Google Colab (NVIDIA A100-SXM4-40GB)",
            plant_identity=_plant_identity().to_dict(),
            run_id="preinitialized-jax-run",
        )

        paths, _, _ = self._call(tmp_path)
        provenance = json.loads(paths["provenance"].read_text())
        assert provenance["hardware"] == "Google Colab (NVIDIA A100-SXM4-40GB)"

    def test_three_stage_run_emits_valid_jax_summary(self, tmp_path):
        from environments.shared.result_bundle import validate_result_bundle
        from environments.shared.result_schema import validate_result_summary

        self._call(tmp_path, stage=1, backend_version="0.11.0")
        self._call(tmp_path, stage=2, backend_version="0.11.0")
        paths, _, run_dir = self._call(tmp_path, stage=3, backend_version="0.11.0")

        summary = json.loads(paths["summary"].read_text())
        validate_result_summary(
            summary,
            expected_species="velociraptor",
            require_complete=True,
            canonical_provenance=True,
        )
        assert summary["algorithm"] == "PPO"
        assert summary["backend"] == "jax-mjx"
        assert summary["backend_version"] == "0.11.0"
        assert set(summary["stages"]) == {"1", "2", "3"}
        assert validate_result_bundle(run_dir)["status"] == "canonical-valid"

    def test_selected_and_terminal_jax_metrics_remain_distinct(self, tmp_path):
        for stage in (1, 2, 3):
            self._call(
                tmp_path,
                stage=stage,
                backend_version="0.11.0",
                final_eval_results=_FakeFinalEvalResults(),
            )

        summary = json.loads((tmp_path / "run" / "summary.json").read_text())
        assert summary["stages"]["3"]["selected_model_reward"] == 55.0
        assert summary["stages"]["3"]["final_eval_reward"] == 15.0

    def test_completed_jax_bundle_rejects_stage_rewrite_before_mutation(self, tmp_path):
        from environments.shared.result_bundle import ResultBundleError

        self._call(tmp_path, stage=1, backend_version="0.11.0")
        self._call(tmp_path, stage=2, backend_version="0.11.0")
        _, _, run_dir = self._call(tmp_path, stage=3, backend_version="0.11.0")
        before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}

        with pytest.raises(ResultBundleError, match="completed result bundle is immutable"):
            self._call(tmp_path, stage=3, backend_version="0.11.0")

        after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
        assert after == before
