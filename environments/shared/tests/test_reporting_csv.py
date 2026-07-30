"""Tests for environments.shared.reporting.csv_output."""

import csv

import pytest

from environments.shared.reporting import CSV_METRIC_COLUMNS, write_results_csv
from environments.shared.reporting.csv_output import _compute_fieldnames


class TestCsvMetricColumns:
    """Tests for CSV_METRIC_COLUMNS schema."""

    def test_contains_distance_traveled(self):
        assert "mean_distance_traveled" in CSV_METRIC_COLUMNS

    def test_contains_forward_vel(self):
        assert "mean_forward_vel" in CSV_METRIC_COLUMNS


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
