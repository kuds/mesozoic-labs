"""Tests for sweep results.py — trial result collection, CSV export, model path resolution."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.scripts.sweep import (
    SweepStageError,
    _best_trial_model_path,
    _collect_trial_results,
    collect_results_from_disk,
    write_results_csv,
)

# ── Helpers ──────────────────────────────────────────────────────────


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
    trial_dir = Path(output_base) / str(trial_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f)


def _setup_sweep_dir(base, stage, trials):
    """Create a sweep-style directory with metrics.json per trial."""
    stage_dir = base / f"stage{stage}"
    for trial_id, metrics in trials.items():
        trial_dir = stage_dir / str(trial_id)
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)
    return stage_dir


def _make_mock_gcs_blobs(file_map):
    """Build mock GCS blobs from a dict of {blob_name: content_bytes}."""
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


# ── _collect_trial_results ───────────────────────────────────────────────


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


class TestBestTrialModelPathStage3:
    def test_stage3_best_trial_identified(self):
        rows = [
            {"trial_id": "1", "stage_passed": True, "best_mean_reward": 100.0},
            {"trial_id": "2", "stage_passed": True, "best_mean_reward": 200.0},
        ]
        path, best_row = _best_trial_model_path(rows, "bucket", "velociraptor", 3)
        assert best_row["trial_id"] == "2"
        assert "stage3" in path


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


# ── collect_results_from_disk ───────────────────────────────────────────


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
