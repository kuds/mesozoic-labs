"""Tests for sweep state.py — sweep state persistence (save/load)."""

import json
from unittest.mock import MagicMock, patch

from environments.shared.scripts.sweep import (
    _load_sweep_state,
    _save_sweep_state,
    _sweep_state_local_path,
)

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
