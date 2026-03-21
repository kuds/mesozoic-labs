"""Tests for the TOML config loader."""

import csv
import json
from unittest.mock import patch

import pytest

from environments.shared.config import (
    _detect_gpu_info,
    _detect_gpu_info_nvidia_smi,
    _find_stage_file,
    _upload_to_gcs,
    append_stage_result_csv,
    load_all_stages,
    load_stage_config,
    save_stage_config,
    upload_curriculum_artifacts,
)

SPECIES = ["velociraptor", "brachiosaurus", "trex"]


class TestLoadStageConfig:
    """Test loading individual stage configs."""

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_loads_successfully(self, species, stage):
        config = load_stage_config(species, stage)
        assert isinstance(config, dict)

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_has_required_keys(self, species, stage):
        config = load_stage_config(species, stage)
        assert "name" in config
        assert "description" in config
        assert "env_kwargs" in config
        assert "ppo_kwargs" in config
        assert "sac_kwargs" in config

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_env_kwargs_has_common_keys(self, species, stage):
        config = load_stage_config(species, stage)
        env_kw = config["env_kwargs"]
        assert "forward_vel_weight" in env_kw
        assert "alive_bonus" in env_kw
        assert "energy_penalty_weight" in env_kw
        assert "max_episode_steps" in env_kw

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_ppo_kwargs_has_common_keys(self, species, stage):
        config = load_stage_config(species, stage)
        ppo_kw = config["ppo_kwargs"]
        assert "learning_rate" in ppo_kw
        assert "batch_size" in ppo_kw
        assert "gamma" in ppo_kw

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_range_params_are_tuples(self, species, stage):
        """TOML lists should be converted to tuples for range parameters."""
        config = load_stage_config(species, stage)
        env_kw = config["env_kwargs"]
        for key, value in env_kw.items():
            if key.endswith("_range"):
                assert isinstance(value, tuple), f"{key} should be a tuple, got {type(value)}"

    def test_missing_species_raises(self):
        with pytest.raises(FileNotFoundError):
            load_stage_config("stegosaurus", 1)

    def test_explicit_config_path(self, tmp_path):
        toml_content = b"""
[stage]
name = "test"
description = "test stage"

[env]
forward_vel_weight = 1.0
alive_bonus = 0.5
energy_penalty_weight = 0.001
max_episode_steps = 100

[ppo]
learning_rate = 3e-4
batch_size = 64
gamma = 0.99
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_bytes(toml_content)

        config = load_stage_config("ignored", 1, config_path=str(config_file))
        assert config["name"] == "test"
        assert config["env_kwargs"]["forward_vel_weight"] == 1.0


class TestLoadAllStages:
    """Test loading all stages for a species."""

    @pytest.mark.parametrize("species", SPECIES)
    def test_returns_three_stages(self, species):
        stages = load_all_stages(species)
        assert set(stages.keys()) == {1, 2, 3}

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage1_is_balance(self, species):
        stages = load_all_stages(species)
        assert stages[1]["name"] == "balance"

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage1_no_forward_reward(self, species):
        """Stage 1 (balance) should have zero forward velocity weight."""
        stages = load_all_stages(species)
        assert stages[1]["env_kwargs"]["forward_vel_weight"] == 0.0

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage_progression_alive_bonus(self, species):
        """Alive bonus should not increase across stages (less reliance on survival)."""
        stages = load_all_stages(species)
        assert stages[1]["env_kwargs"]["alive_bonus"] >= stages[2]["env_kwargs"]["alive_bonus"]
        assert stages[2]["env_kwargs"]["alive_bonus"] >= stages[3]["env_kwargs"]["alive_bonus"]

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage_progression_learning_rate(self, species):
        """Learning rate should decrease across stages (finer tuning)."""
        stages = load_all_stages(species)
        assert stages[1]["ppo_kwargs"]["learning_rate"] > stages[2]["ppo_kwargs"]["learning_rate"]
        assert stages[2]["ppo_kwargs"]["learning_rate"] >= stages[3]["ppo_kwargs"]["learning_rate"]


class TestCurriculumInvariants:
    """Test that curriculum configs maintain expected invariants."""

    def test_velociraptor_strike_bonus_only_stage3(self):
        stages = load_all_stages("velociraptor")
        assert stages[1]["env_kwargs"]["strike_bonus"] == 0.0
        assert stages[2]["env_kwargs"]["strike_bonus"] == 0.0
        assert stages[3]["env_kwargs"]["strike_bonus"] > 0.0

    def test_brachiosaurus_food_reach_only_stage3(self):
        stages = load_all_stages("brachiosaurus")
        assert stages[1]["env_kwargs"]["food_reach_bonus"] == 0.0
        assert stages[2]["env_kwargs"]["food_reach_bonus"] == 0.0
        assert stages[3]["env_kwargs"]["food_reach_bonus"] > 0.0

    def test_trex_bite_bonus_only_stage3(self):
        stages = load_all_stages("trex")
        assert stages[1]["env_kwargs"]["bite_bonus"] == 0.0
        assert stages[2]["env_kwargs"]["bite_bonus"] == 0.0
        assert stages[3]["env_kwargs"]["bite_bonus"] > 0.0

    @pytest.mark.parametrize("species", SPECIES)
    def test_consistent_episode_length(self, species):
        """All stages should use the same max_episode_steps for consistent return horizons."""
        stages = load_all_stages(species)
        assert stages[1]["env_kwargs"]["max_episode_steps"] == stages[2]["env_kwargs"]["max_episode_steps"]


class TestCatastrophicForgettingMitigation:
    """Regression tests for curriculum configs that prevent catastrophic forgetting.

    Stage 2 must relax balance-centric reward weights below Stage 1 values.
    Keeping them equal (or higher) traps the agent in a standing posture and
    prevents locomotion learning — the "inverse forgetting" pattern documented
    in TRAINING_REVIEW.md.
    """

    def test_velociraptor_stage2_reduces_posture_weight(self):
        """Stage 2 posture_weight must be strictly lower than Stage 1."""
        stages = load_all_stages("velociraptor")
        assert stages[2]["env_kwargs"]["posture_weight"] < stages[1]["env_kwargs"]["posture_weight"], (
            "Stage 2 posture_weight must be reduced from Stage 1 to allow forward lean during locomotion"
        )

    def test_velociraptor_stage2_reduces_nosedive_weight(self):
        """Stage 2 nosedive_weight must be strictly lower than Stage 1."""
        stages = load_all_stages("velociraptor")
        assert stages[2]["env_kwargs"]["nosedive_weight"] < stages[1]["env_kwargs"]["nosedive_weight"], (
            "Stage 2 nosedive_weight must be reduced from Stage 1 to allow natural walking lean"
        )

    def test_velociraptor_stage2_reduces_alive_bonus(self):
        """Stage 2 alive_bonus must be lower than Stage 1."""
        stages = load_all_stages("velociraptor")
        assert stages[2]["env_kwargs"]["alive_bonus"] < stages[1]["env_kwargs"]["alive_bonus"], (
            "Stage 2 alive_bonus must be reduced so forward velocity reward can dominate"
        )

    def test_trex_stage2_reduces_posture_weight(self):
        """T-Rex Stage 2 posture_weight must be strictly lower than Stage 1."""
        stages = load_all_stages("trex")
        assert stages[2]["env_kwargs"]["posture_weight"] < stages[1]["env_kwargs"]["posture_weight"], (
            "T-Rex Stage 2 posture_weight must be reduced to allow locomotion"
        )

    def test_trex_stage2_reduces_nosedive_weight(self):
        """T-Rex Stage 2 nosedive_weight must be strictly lower than Stage 1."""
        stages = load_all_stages("trex")
        assert stages[2]["env_kwargs"]["nosedive_weight"] < stages[1]["env_kwargs"]["nosedive_weight"], (
            "T-Rex Stage 2 nosedive_weight must be reduced to allow head-forward walking"
        )

    def test_trex_stage2_reduces_alive_bonus(self):
        """T-Rex Stage 2 alive_bonus must be lower than Stage 1."""
        stages = load_all_stages("trex")
        assert stages[2]["env_kwargs"]["alive_bonus"] < stages[1]["env_kwargs"]["alive_bonus"], (
            "T-Rex Stage 2 alive_bonus must be reduced so forward velocity reward can dominate"
        )

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage2_has_warmup_config(self, species):
        """Stage 2 must have warmup configuration to stabilise critic during transition."""
        stages = load_all_stages(species)
        cur = stages[2].get("curriculum_kwargs", {})
        assert "warmup_timesteps" in cur, f"{species} Stage 2 missing warmup_timesteps"
        assert cur["warmup_timesteps"] > 0

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage2_has_reward_ramp_config(self, species):
        """Stage 2 must have reward ramp config to gradually introduce forward_vel_weight."""
        stages = load_all_stages(species)
        cur = stages[2].get("curriculum_kwargs", {})
        assert "ramp_timesteps" in cur, f"{species} Stage 2 missing ramp_timesteps"
        assert cur["ramp_timesteps"] > 0
        assert "ramp_start_value" in cur, f"{species} Stage 2 missing ramp_start_value"
        assert 0 < cur["ramp_start_value"] < 1.0

    @pytest.mark.parametrize("species", SPECIES)
    def test_stage2_learning_rate_lower_than_stage1(self, species):
        """Stage 2 LR must be lower than Stage 1 to prevent overwriting balance knowledge."""
        stages = load_all_stages(species)
        assert stages[2]["ppo_kwargs"]["learning_rate"] < stages[1]["ppo_kwargs"]["learning_rate"], (
            f"{species}: Stage 2 LR must be lower than Stage 1 to mitigate catastrophic forgetting"
        )


class TestSaveStageConfig:
    """Test saving stage config to JSON."""

    def test_saves_ppo_config(self, tmp_path):
        stage_config = {
            "name": "balance",
            "description": "Stand upright",
            "env_kwargs": {"alive_bonus": 2.0, "prey_distance_range": (5.0, 10.0)},
            "ppo_kwargs": {"learning_rate": 3e-4, "batch_size": 64},
            "sac_kwargs": {"learning_rate": 1e-4},
            "curriculum_kwargs": {"min_avg_reward": 10.0},
        }
        out = save_stage_config(tmp_path / "stage1", 1, stage_config, "PPO", species="velociraptor")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["species"] == "velociraptor"
        assert data["stage"] == 1
        assert data["name"] == "balance"
        assert data["algorithm"] == "PPO"
        assert data["hyperparameters"]["learning_rate"] == 3e-4
        # Tuples should be converted to lists for JSON
        assert data["reward_weights"]["prey_distance_range"] == [5.0, 10.0]
        assert data["curriculum"]["min_avg_reward"] == 10.0

    def test_saves_sac_config(self, tmp_path):
        stage_config = {
            "name": "locomotion",
            "description": "Walk forward",
            "env_kwargs": {"forward_vel_weight": 1.0},
            "ppo_kwargs": {},
            "sac_kwargs": {"learning_rate": 1e-4, "batch_size": 256},
            "curriculum_kwargs": {},
        }
        out = save_stage_config(tmp_path / "stage2", 2, stage_config, "SAC")
        data = json.loads(out.read_text())
        assert data["algorithm"] == "SAC"
        assert data["hyperparameters"]["learning_rate"] == 1e-4

    def test_saves_with_extra_metadata(self, tmp_path):
        stage_config = {
            "name": "test",
            "env_kwargs": {},
            "ppo_kwargs": {},
            "sac_kwargs": {},
            "curriculum_kwargs": {},
        }
        extra = {"seed": 42, "n_envs": 4}
        out = save_stage_config(tmp_path / "run", 1, stage_config, "PPO", extra=extra)
        data = json.loads(out.read_text())
        assert data["run"]["seed"] == 42
        assert data["run"]["n_envs"] == 4

    def test_creates_nested_directories(self, tmp_path):
        stage_config = {"name": "t", "env_kwargs": {}, "ppo_kwargs": {}, "sac_kwargs": {}, "curriculum_kwargs": {}}
        out = save_stage_config(tmp_path / "a" / "b" / "c", 1, stage_config, "PPO")
        assert out.exists()

    def test_includes_gpu_info_when_available(self, tmp_path):
        stage_config = {"name": "t", "env_kwargs": {}, "ppo_kwargs": {}, "sac_kwargs": {}, "curriculum_kwargs": {}}
        fake_gpu = {
            "gpu_model": "A100",
            "gpu_full_name": "NVIDIA A100-SXM4-40GB",
            "gpu_memory_gb": 40.0,
            "cuda_version": "12.1",
        }
        with patch("environments.shared.config._detect_gpu_info", return_value=fake_gpu):
            out = save_stage_config(tmp_path / "gpu_run", 1, stage_config, "PPO")
        data = json.loads(out.read_text())
        assert data["gpu"]["gpu_model"] == "A100"
        assert data["gpu"]["gpu_memory_gb"] == 40.0
        assert data["gpu"]["cuda_version"] == "12.1"

    def test_omits_gpu_key_when_no_gpu(self, tmp_path):
        stage_config = {"name": "t", "env_kwargs": {}, "ppo_kwargs": {}, "sac_kwargs": {}, "curriculum_kwargs": {}}
        with patch("environments.shared.config._detect_gpu_info", return_value={}):
            out = save_stage_config(tmp_path / "cpu_run", 1, stage_config, "PPO")
        data = json.loads(out.read_text())
        assert "gpu" not in data


class TestDetectGpuInfo:
    """Tests for GPU detection with torch and nvidia-smi fallback."""

    def test_falls_back_to_nvidia_smi_when_torch_unavailable(self):
        fake_smi = {
            "gpu_model": "T4",
            "gpu_full_name": "Tesla T4",
            "gpu_memory_gb": 15.0,
            "driver_version": "535.104.05",
        }
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("environments.shared.config._detect_gpu_info_nvidia_smi", return_value=fake_smi),
        ):
            result = _detect_gpu_info()
        assert result["gpu_model"] == "T4"
        assert result["gpu_memory_gb"] == 15.0

    def test_nvidia_smi_parses_output(self):
        import subprocess

        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Tesla T4, 15360, 535.104.05\n", stderr=""
        )
        with patch("subprocess.run", return_value=fake_result):
            result = _detect_gpu_info_nvidia_smi()
        assert result["gpu_model"] == "T4"
        assert result["gpu_full_name"] == "Tesla T4"
        assert result["gpu_memory_gb"] == 15.0
        assert result["driver_version"] == "535.104.05"

    def test_nvidia_smi_returns_empty_on_failure(self):
        import subprocess

        fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="not found")
        with patch("subprocess.run", return_value=fake_result):
            result = _detect_gpu_info_nvidia_smi()
        assert result == {}

    def test_nvidia_smi_returns_empty_when_not_installed(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _detect_gpu_info_nvidia_smi()
        assert result == {}


class TestAppendStageResultCsv:
    """Test CSV append helper."""

    def test_creates_new_csv(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        data = {"stage": 1, "reward": 10.5, "passed": True}
        result = append_stage_result_csv(csv_path, data)
        assert result == csv_path
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["stage"] == "1"
        assert rows[0]["reward"] == "10.5"

    def test_appends_to_existing_csv(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        append_stage_result_csv(csv_path, {"stage": 1, "reward": 10.0})
        append_stage_result_csv(csv_path, {"stage": 2, "reward": 20.0})
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[1]["stage"] == "2"

    def test_expands_header_for_new_keys(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        append_stage_result_csv(csv_path, {"stage": 1, "reward": 10.0})
        append_stage_result_csv(csv_path, {"stage": 2, "reward": 20.0, "velocity": 1.5})
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            assert "velocity" in reader.fieldnames
            rows = list(reader)
        assert len(rows) == 2
        assert rows[1]["velocity"] == "1.5"


class TestFindStageFile:
    """Test edge cases in _find_stage_file."""

    def test_no_matching_file_raises(self, tmp_path):
        species_dir = tmp_path / "configs" / "unknown_species"
        species_dir.mkdir(parents=True)
        with patch("environments.shared.config._CONFIGS_DIR", tmp_path / "configs"):
            with pytest.raises(FileNotFoundError, match="No config file matching"):
                _find_stage_file("unknown_species", 1)

    def test_multiple_matching_files_raises(self, tmp_path):
        species_dir = tmp_path / "configs" / "multi"
        species_dir.mkdir(parents=True)
        (species_dir / "stage1_a.toml").write_text("")
        (species_dir / "stage1_b.toml").write_text("")
        with patch("environments.shared.config._CONFIGS_DIR", tmp_path / "configs"):
            with pytest.raises(ValueError, match="Multiple config files"):
                _find_stage_file("multi", 1)


class TestUploadToGcs:
    """Test GCS upload."""

    def test_returns_false_for_missing_file(self, tmp_path):
        result = _upload_to_gcs(tmp_path / "nonexistent.csv", "bucket", "path.csv")
        assert result is False

    def test_returns_false_when_gcs_import_fails(self, tmp_path):
        local_file = tmp_path / "data.csv"
        local_file.write_text("a,b\n1,2\n")
        # The GCS import is inside the function, so we mock the import mechanism
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "google.cloud":
                raise ImportError("no google-cloud-storage")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _upload_to_gcs(local_file, "bucket", "data.csv")
        assert result is False


class TestUploadCurriculumArtifacts:
    """Test upload_curriculum_artifacts."""

    def test_noop_when_no_bucket(self, tmp_path):
        """Should do nothing when bucket is None."""
        upload_curriculum_artifacts(tmp_path, "velociraptor", "ppo", bucket=None)

    def test_uploads_artifacts_with_bucket(self, tmp_path):
        """Should call upload_to_gcs for existing artifacts."""
        # Create a fake run directory structure
        base = tmp_path / "curriculum_20240228_150000"
        base.mkdir()
        (base / "curriculum_results.csv").write_text("stage,reward\n1,10\n")
        (base / "training_summary.txt").write_text("summary")
        stage1 = base / "stage1"
        stage1_models = stage1 / "models"
        stage1_models.mkdir(parents=True)
        (stage1 / "stage_summary.txt").write_text("stage 1 summary")
        (stage1 / "velociraptor_ppo_stage1_best.mp4").write_bytes(b"vid1")
        (stage1 / "velociraptor_ppo_stage1_final.mp4").write_bytes(b"vid2")
        (stage1_models / "best_model.zip").write_bytes(b"fake")
        (stage1_models / "stage1_final.zip").write_bytes(b"fake")
        (stage1_models / "stage1_final_vecnorm.pkl").write_bytes(b"fake")

        with patch("environments.shared.config._upload_to_gcs", return_value=True) as mock_upload:
            upload_curriculum_artifacts(base, "velociraptor", "ppo", bucket="test-bucket", project="test-project")

        # CSV + training_summary + stage_summary + 2 videos + best_model + final + vecnorm = 8
        assert mock_upload.call_count == 8

        # Verify the GCS paths for the new artifact types
        uploaded_paths = [call.args[2] for call in mock_upload.call_args_list]
        run = "curriculum_20240228_150000"
        assert f"training/velociraptor/{run}/training_summary.txt" in uploaded_paths
        assert f"training/velociraptor/{run}/stage1/stage_summary.txt" in uploaded_paths
        assert f"training/velociraptor/{run}/stage1/velociraptor_ppo_stage1_best.mp4" in uploaded_paths
        assert f"training/velociraptor/{run}/stage1/velociraptor_ppo_stage1_final.mp4" in uploaded_paths
