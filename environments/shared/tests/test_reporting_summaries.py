"""Tests for environments.shared.reporting.summaries."""

import csv
import json

import pytest

from environments.shared.reporting import save_results_csv, save_results_json

from .reporting_helpers import make_stage_result, plant_identity


class TestSaveResultsJson:
    """Tests for save_results_json (machine-readable JSON output)."""

    def test_creates_json_file(self, tmp_path):
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        assert path.exists()
        assert path.name == "summary.json"

    def test_returns_path(self, tmp_path):
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        assert path == tmp_path / "summary.json"

    def test_valid_json(self, tmp_path):
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_contains_species(self, tmp_path):
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["species"] == "velociraptor"

    def test_contains_algorithm(self, tmp_path):
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "SAC", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["algorithm"] == "SAC"

    def test_contains_seed(self, tmp_path):
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=123, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["seed"] == 123

    def test_contains_date(self, tmp_path):
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert "date" in data

    def test_stages_match_input(self, tmp_path):
        results = [make_stage_result(1), make_stage_result(2), make_stage_result(3)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert set(data["stages"].keys()) == {"1", "2", "3"}

    def test_total_timesteps(self, tmp_path):
        results = [make_stage_result(1, timesteps=100_000), make_stage_result(2, timesteps=200_000)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["total_timesteps"] == 300_000

    def test_total_training_time(self, tmp_path):
        results = [make_stage_result(1, duration_seconds=100.0), make_stage_result(2, duration_seconds=200.0)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["total_training_time_seconds"] == 300.0

    def test_final_avg_reward(self, tmp_path):
        results = [make_stage_result(1, mean_reward=50.0), make_stage_result(2, mean_reward=75.123)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["final_avg_reward"] == 75.12

    def test_provenance_records_repository_commit(self, tmp_path):
        results = [make_stage_result(1)]
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
        results = [make_stage_result(1)]
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
        results = [make_stage_result(1)]
        data = json.loads(save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path).read_text())
        assert data["backend"] == "stable-baselines3"
        assert "backend_version" in data

    def test_forward_vel_in_stage_data(self, tmp_path):
        results = [make_stage_result(1, mean_forward_vel=1.5)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        assert data["stages"]["1"]["avg_forward_vel"] == 1.5

    def test_creates_results_dir_if_needed(self, tmp_path):
        nested_dir = tmp_path / "a" / "b" / "c"
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=nested_dir)
        assert path.exists()

    def test_accepts_string_path(self, tmp_path):
        results = [make_stage_result(1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=str(tmp_path))
        assert path.exists()

    def test_stage_data_has_training_time(self, tmp_path):
        results = [make_stage_result(1, duration_seconds=3661.0)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        data = json.loads(path.read_text())
        stage = data["stages"]["1"]
        assert "training_time" in stage
        assert "training_time_seconds" in stage
        assert stage["training_time"] == "1:01:01"

    def test_includes_plant_identity_from_final_stage(self, tmp_path):
        identity = plant_identity().to_dict()
        results = [make_stage_result(1, plant_identity=identity)]

        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)

        assert json.loads(path.read_text())["plant_identity"] == identity

    def test_standard_csv_flattens_plant_provenance(self, tmp_path):
        identity = plant_identity().to_dict()
        results = [make_stage_result(1, plant_identity=identity)]
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


class TestGateKindProvenance:
    """Review SS5: a stage_passed travels with the gate kind it was earned under."""

    def test_gate_kind_travels_with_the_verdict(self, tmp_path):
        results = [make_stage_result(1, gate_kind="reward_and_length/v1", gate_schema_version=1)]
        path = save_results_json(results, "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        stage = json.loads(path.read_text())["stages"]["1"]
        assert stage["stage_passed"] is True
        assert stage["gate_kind"] == "reward_and_length/v1"
        assert stage["gate_schema_version"] == 1

    def test_a_producer_without_the_key_omits_it(self, tmp_path):
        """Absent, not null: "unrecorded" must stay distinguishable from "declared none"."""
        path = save_results_json([make_stage_result(1)], "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        stage = json.loads(path.read_text())["stages"]["1"]
        assert "gate_kind" not in stage
        assert "gate_schema_version" not in stage


class TestSaveResultsJsonIsAtomic:
    def test_a_crash_mid_publish_keeps_the_previous_summary(self, tmp_path, monkeypatch):
        """Review ER5: the same lost-mount treatment every bundle writer gets."""
        import os

        path = save_results_json([make_stage_result(1)], "velociraptor", "PPO", seed=42, results_dir=tmp_path)
        before = path.read_bytes()

        def lost_mount(src, dst):
            raise OSError("mount went away")

        monkeypatch.setattr(os, "replace", lost_mount)
        with pytest.raises(OSError, match="mount went away"):
            save_results_json(
                [make_stage_result(1, mean_reward=999.0)], "velociraptor", "PPO", seed=42, results_dir=tmp_path
            )

        assert path.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]
