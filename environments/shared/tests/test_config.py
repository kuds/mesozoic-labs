"""Tests for the TOML config loader."""

import csv
import json
import logging
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.config import (
    _CONFIGS_DIR,
    LOAD_LINEAGE_KEYS,
    _detect_gpu_info,
    _detect_gpu_info_nvidia_smi,
    _upload_to_gcs,
    append_stage_result_csv,
    get_git_commit,
    load_all_stages,
    load_stage_config,
    save_stage_config,
    upload_curriculum_artifacts,
)

from .reporting_helpers import make_plant_identity as _plant_identity

SPECIES = ["velociraptor", "brachiosaurus", "trex"]
#: Every species with committed stage configs, manifest-less ones included.
COMMITTED_SPECIES = sorted(path.name for path in _CONFIGS_DIR.iterdir() if path.is_dir())


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

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_net_arch_only_under_policy_kwargs(self, species, stage):
        """net_arch must live under [<alg>.policy_kwargs], never in [<alg>].

        A lost [ppo.policy_kwargs] table header silently drops net_arch
        into [ppo], which is forwarded verbatim to PPO.__init__ and
        crashes at model construction — hours into a Colab run instead of
        in CI (run 20260712_185931).
        """
        config = load_stage_config(species, stage)
        for alg_key in ("ppo_kwargs", "sac_kwargs", "jax_kwargs"):
            kwargs = config[alg_key]
            assert "net_arch" not in kwargs, (
                f"{species} stage {stage}: net_arch is a top-level [{alg_key[:-7]}] key — "
                f"a [{alg_key[:-7]}.policy_kwargs] header is missing in the TOML."
            )
        assert "net_arch" in config["ppo_kwargs"].get("policy_kwargs", {}), (
            f"{species} stage {stage}: [ppo.policy_kwargs] must define net_arch."
        )

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_ppo_kwargs_accepted_by_sb3(self, species, stage):
        """Every top-level [ppo] key must be a PPO constructor kwarg or a harness schedule key."""
        sb3 = pytest.importorskip("stable_baselines3")
        import inspect

        # Popped by _prepare_alg_kwargs in train_base.py before PPO() is called.
        harness_keys = {
            "ent_coef_end",
            "ent_coef_decay_timesteps",
            "learning_rate_end",
            "lr_schedule",
            "clip_range_end",
        }
        accepted = set(inspect.signature(sb3.PPO.__init__).parameters) | harness_keys
        config = load_stage_config(species, stage)
        unknown = set(config["ppo_kwargs"]) - accepted
        assert not unknown, (
            f"{species} stage {stage}: [ppo] keys {sorted(unknown)} are neither PPO "
            "constructor kwargs nor harness schedule keys — they would crash "
            "PPO.__init__ at run time."
        )

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_env_kwargs_accepted_by_env_constructor(self, species, stage):
        """[env] is passed verbatim to the species env constructor; unknown keys crash at reset."""
        import inspect

        from environments.shared.species_registry import get_species_config

        env_class = get_species_config(species).env_class
        accepted = set(inspect.signature(env_class.__init__).parameters)
        config = load_stage_config(species, stage)
        unknown = set(config["env_kwargs"]) - accepted
        assert not unknown, (
            f"{species} stage {stage}: [env] keys {sorted(unknown)} are not "
            f"{env_class.__name__}.__init__ parameters — they would crash env "
            "construction at run time."
        )

    def test_missing_species_raises(self):
        from environments.shared.stage_manifest import StageManifestError

        with pytest.raises(StageManifestError, match="config directory not found"):
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
    def test_returns_every_manifest_stage(self, species):
        stages = load_all_stages(species)
        # The legacy numbers are present for every species; a species whose
        # manifest declares semantic-only stages (trex: recovery) carries
        # them under their IDs without moving the integers.
        assert {1, 2, 3} <= set(stages.keys())
        extra = set(stages.keys()) - {1, 2, 3}
        assert extra == ({"recovery"} if species == "trex" else set())

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
        """Later stages should use moderate LRs (Stage 1 can be lower due to sweep-tuned warmup in S2)."""
        stages = load_all_stages(species)
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
    in docs/investigations/TRAINING_REVIEW.md.
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
    def test_stage2_has_warmup_for_catastrophic_forgetting(self, species):
        """Stage 2 must have warmup to mitigate catastrophic forgetting (replaces LR-drop requirement)."""
        stages = load_all_stages(species)
        cur = stages[2].get("curriculum_kwargs", {})
        assert cur.get("warmup_timesteps", 0) > 0, (
            f"{species}: Stage 2 needs warmup_timesteps > 0 to mitigate catastrophic forgetting"
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

    def test_embeds_plant_identity_and_writes_sidecar(self, tmp_path):
        stage_dir = tmp_path / "stage1"
        identity = _plant_identity()
        stage_config = {
            "name": "test",
            "env_kwargs": {},
            "ppo_kwargs": {},
            "sac_kwargs": {},
            "curriculum_kwargs": {},
        }

        out = save_stage_config(stage_dir, 1, stage_config, "PPO", plant_identity=identity)

        assert json.loads(out.read_text())["plant_identity"] == identity.to_dict()
        assert json.loads((stage_dir / "plant_identity.json").read_text()) == identity.to_dict()

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

    def test_torch_reports_24gb_using_total_memory(self):
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        cuda.get_device_properties.return_value = SimpleNamespace(total_memory=24_000_000_000)
        fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.1"))

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("environments.shared.config._detect_gpu_info_nvidia_smi") as mock_fallback,
        ):
            result = _detect_gpu_info()

        assert result == {
            "gpu_model": "RTX",
            "gpu_full_name": "NVIDIA RTX 4090",
            "gpu_memory_gb": 24.0,
            "cuda_version": "12.1",
        }
        mock_fallback.assert_not_called()

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

    def test_device_property_lookup_failure_falls_back_to_nvidia_smi(self):
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        cuda.get_device_properties.side_effect = RuntimeError("CUDA device query failed")
        fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.1"))
        fake_smi = {
            "gpu_model": "RTX",
            "gpu_full_name": "NVIDIA RTX 4090",
            "gpu_memory_gb": 24.0,
            "driver_version": "555.42",
        }

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("environments.shared.config._detect_gpu_info_nvidia_smi", return_value=fake_smi),
        ):
            result = _detect_gpu_info()

        assert result == fake_smi

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


class TestGetGitCommit:
    """Git-commit capture for stage-config reproducibility."""

    def test_returns_commit_hash(self):
        import subprocess

        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123def456\n", stderr="")
        with patch("subprocess.run", return_value=fake):
            assert get_git_commit() == "abc123def456"

    def test_falls_back_to_github_sha(self, monkeypatch):
        import subprocess

        monkeypatch.setenv("GITHUB_SHA", "ci-sha-789")
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="not a git repo")
        with patch("subprocess.run", return_value=fake):
            assert get_git_commit() == "ci-sha-789"

    def test_unknown_when_no_git_and_no_env(self, monkeypatch):
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert get_git_commit() == "unknown"


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


class TestStageFileResolution:
    """Integer stages resolve through the manifest; its synthesizer keeps
    the old edge-case guarantees for manifest-less species."""

    def test_no_config_files_raises(self, tmp_path):
        from environments.shared.stage_manifest import StageManifestError, load_stage_manifest

        species_dir = tmp_path / "configs" / "unknown_species"
        species_dir.mkdir(parents=True)
        with pytest.raises(StageManifestError, match="no stage config files"):
            load_stage_manifest("unknown_species", configs_dir=tmp_path / "configs")

    def test_multiple_matching_files_raises(self, tmp_path):
        from environments.shared.stage_manifest import StageManifestError, load_stage_manifest

        species_dir = tmp_path / "configs" / "multi"
        species_dir.mkdir(parents=True)
        (species_dir / "stage1_a.toml").write_text("")
        (species_dir / "stage1_b.toml").write_text("")
        with pytest.raises(StageManifestError, match="multiple stage-1 config files"):
            load_stage_manifest("multi", configs_dir=tmp_path / "configs")


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
        (base / "plant_identity.json").write_text("{}")
        stage1 = base / "stage1"
        stage1_models = stage1 / "models"
        stage1_models.mkdir(parents=True)
        (stage1 / "stage_summary.txt").write_text("stage 1 summary")
        (stage1 / "plant_identity.json").write_text("{}")
        (stage1 / "velociraptor_ppo_stage1_best.mp4").write_bytes(b"vid1")
        (stage1 / "velociraptor_ppo_stage1_final.mp4").write_bytes(b"vid2")
        (stage1_models / "best_model.zip").write_bytes(b"fake")
        (stage1_models / "stage1_final.zip").write_bytes(b"fake")
        (stage1_models / "stage1_final_vecnorm.pkl").write_bytes(b"fake")

        with patch("environments.shared.config._upload_to_gcs", return_value=True) as mock_upload:
            upload_curriculum_artifacts(base, "velociraptor", "ppo", bucket="test-bucket", project="test-project")

        # Run CSV/summary/identity + stage summary/identity + 2 videos + 3 models = 10
        assert mock_upload.call_count == 10

        # Verify the GCS paths for the new artifact types
        uploaded_paths = [call.args[2] for call in mock_upload.call_args_list]
        run = "curriculum_20240228_150000"
        assert f"training/velociraptor/{run}/training_summary.txt" in uploaded_paths
        assert f"training/velociraptor/{run}/plant_identity.json" in uploaded_paths
        assert f"training/velociraptor/{run}/stage1/stage_summary.txt" in uploaded_paths
        assert f"training/velociraptor/{run}/stage1/plant_identity.json" in uploaded_paths
        assert f"training/velociraptor/{run}/stage1/velociraptor_ppo_stage1_best.mp4" in uploaded_paths
        assert f"training/velociraptor/{run}/stage1/velociraptor_ppo_stage1_final.mp4" in uploaded_paths

    @staticmethod
    def _uploaded_keys(base, species):
        with patch("environments.shared.config._upload_to_gcs", return_value=True) as mock_upload:
            upload_curriculum_artifacts(base, species, "ppo", bucket="test-bucket")
        return [call.args[2] for call in mock_upload.call_args_list]

    def test_nn_id_and_bare_id_dirs_upload_and_unrelated_dirs_do_not(self, tmp_path):
        """The stage-dir filter is stage_ref_from_dirname with the species in hand."""
        base = tmp_path / "curriculum_20260901_120000"
        for name in ("02_recovery", "recovery", "stage1", "models", "replays", "stage4", "ancestors"):
            (base / name).mkdir(parents=True)
            (base / name / "stage_config.json").write_text("{}")
        keys = self._uploaded_keys(base, "trex")
        run = base.name
        for name in ("02_recovery", "recovery", "stage1"):
            assert f"training/trex/{run}/{name}/stage_config.json" in keys
        for name in ("models", "replays", "stage4", "ancestors"):
            assert not [key for key in keys if f"/{name}/" in key], name

    def test_an_open_id_dir_uploads_only_when_the_manifest_declares_it(self, tmp_path, monkeypatch):
        import shutil

        from environments.shared import stage_manifest

        configs = tmp_path / "configs"
        shutil.copytree(stage_manifest._CONFIGS_DIR / "trex", configs / "trex")
        species_dir = configs / "pilot"
        species_dir.mkdir(parents=True)
        for name in ("stance.toml", "follow_direction.toml"):
            (species_dir / name).write_text("[stage]\nname = 'x'\n")
        (species_dir / "stages.toml").write_text(
            f'schema = "{stage_manifest.STAGE_MANIFEST_SCHEMA_V2}"\n'
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = true\n'
            '[[stages]]\nid = "follow_direction"\nconfig = "follow_direction.toml"\nwarm_start_from = "stance"\n'
            "deliverable = true\n"
        )
        monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", configs)
        base = tmp_path / "run"
        for name in ("01_stance", "02_follow_direction", "02_sprint"):
            (base / name).mkdir(parents=True)
            (base / name / "stage_config.json").write_text("{}")
        keys = self._uploaded_keys(base, "pilot")
        assert "training/pilot/run/01_stance/stage_config.json" in keys
        assert "training/pilot/run/02_follow_direction/stage_config.json" in keys
        assert not [key for key in keys if "02_sprint" in key]
        # A species whose manifest does not declare the id skips the directory.
        assert not [key for key in self._uploaded_keys(base, "trex") if "follow_direction" in key]


class TestLoadStageConfigTableValidation:
    """Review CF4: a misspelled top-level table must fail, never load as empty.

    ``[environment]`` for ``[env]`` used to yield ``env_kwargs == {}`` with no
    warning, and the stage trained on constructor defaults — a different
    experiment — while ``save_stage_config`` back-filled those defaults into
    ``stage_config.json`` and hid the omission.
    """

    @staticmethod
    def _write(tmp_path, body):
        path = tmp_path / "stage.toml"
        path.write_text(body)
        return path

    def test_misspelled_env_table_is_rejected_by_name(self, tmp_path):
        path = self._write(
            tmp_path,
            '[stage]\nname = "x"\n\n[environment]\nalive_bonus = 1.0\n\n[ppo]\nlearning_rate = 1e-4\n',
        )
        with pytest.raises(ValueError) as excinfo:
            load_stage_config("ignored", 1, config_path=str(path))
        message = str(excinfo.value)
        assert str(path) in message
        assert "['environment']" in message
        assert "'env'" in message and "'curriculum'" in message

    def test_misspelled_algorithm_table_is_rejected(self, tmp_path):
        path = self._write(
            tmp_path,
            '[stage]\nname = "x"\n\n[env]\nalive_bonus = 1.0\n\n[ppo_config]\nlearning_rate = 1e-4\n',
        )
        with pytest.raises(ValueError, match=r"\['ppo_config'\]"):
            load_stage_config("ignored", 1, config_path=str(path))

    def test_empty_env_table_warns(self, tmp_path, caplog):
        path = self._write(tmp_path, '[stage]\nname = "x"\n\n[ppo]\nlearning_rate = 1e-4\n')
        with caplog.at_level(logging.WARNING, logger="environments.shared.config"):
            config = load_stage_config("ignored", 1, config_path=str(path))
        assert config["env_kwargs"] == {}
        assert any("no [env] table" in record.message for record in caplog.records)

    def test_missing_algorithm_tables_warn(self, tmp_path, caplog):
        path = self._write(tmp_path, '[stage]\nname = "x"\n\n[env]\nalive_bonus = 1.0\n')
        with caplog.at_level(logging.WARNING, logger="environments.shared.config"):
            load_stage_config("ignored", 1, config_path=str(path))
        assert any("no algorithm table" in record.message for record in caplog.records)

    @pytest.mark.parametrize("species", COMMITTED_SPECIES)
    def test_every_committed_stage_config_still_loads_quietly(self, species, caplog):
        from environments.shared.stage_manifest import load_stage_manifest

        with caplog.at_level(logging.WARNING, logger="environments.shared.config"):
            for entry in load_stage_manifest(species).stages:
                config = load_stage_config(species, entry.reference)
                assert config["env_kwargs"], f"{species}/{entry.config_file} loaded an empty [env]"
        assert not [r for r in caplog.records if "no [env] table" in r.message or "no algorithm table" in r.message]


def _sb3_style_zip(path, data):
    """An SB3 checkpoint archive's shape: a JSON ``data`` member beside the weights."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data))
        archive.writestr("policy.pth", b"weights")
    return path


class TestSaveStageConfigLoadLineage:
    """Review RP4: the run block records where a stage's weights came from.

    The audit reads exactly ``load_path`` / ``load_mode`` /
    ``parent_checkpoint_sha256`` / ``parent_task_sha256`` /
    ``parent_run_id`` and audits each key when present, so a from-scratch
    stage writes none of them, an unfingerprinted parent leaves the task
    digest out rather than null, and a parent from this run leaves
    ``parent_run_id`` out (BEHAVIOR_RECIPES_PLAN §4.2: it is written only
    for a certified ancestor reused from ANOTHER run).
    """

    STAGE_CONFIG = {"name": "t", "env_kwargs": {}, "ppo_kwargs": {}, "sac_kwargs": {}, "curriculum_kwargs": {}}

    def _run_block(self, tmp_path, **kwargs):
        out = save_stage_config(tmp_path / "run", 1, self.STAGE_CONFIG, "PPO", extra={"seed": 1}, **kwargs)
        return json.loads(out.read_text())["run"]

    def test_from_scratch_records_no_lineage_keys(self, tmp_path):
        run = self._run_block(tmp_path)
        assert run == {"seed": 1}
        assert not set(LOAD_LINEAGE_KEYS) & set(run)

    def test_from_scratch_without_extra_writes_no_run_block(self, tmp_path):
        out = save_stage_config(tmp_path / "run", 1, self.STAGE_CONFIG, "PPO")
        assert "run" not in json.loads(out.read_text())

    def test_loaded_checkpoint_records_the_lineage_keys(self, tmp_path):
        from environments.shared.result_bundle import sha256_file
        from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE

        digest = "sha256:" + "a" * 64
        parent = _sb3_style_zip(tmp_path / "best_model.zip", {MODEL_TASK_ATTRIBUTE: {"task_sha256": digest}})
        run = self._run_block(tmp_path, load_path=str(parent), load_mode="initialize_next_stage")
        assert run["seed"] == 1
        assert run["load_path"] == str(parent)
        assert run["load_mode"] == "initialize_next_stage"
        assert run["parent_checkpoint_sha256"] == sha256_file(parent)
        assert run["parent_task_sha256"] == digest
        # A same-run handoff (and a plain --load) never names a parent run.
        assert "parent_run_id" not in run

    def test_a_cross_run_parent_records_its_run_id(self, tmp_path):
        parent = _sb3_style_zip(tmp_path / "best_model.zip", {})
        run = self._run_block(
            tmp_path, load_path=str(parent), load_mode="initialize_next_stage", parent_run_id="20260901_120000"
        )
        assert run["parent_run_id"] == "20260901_120000"
        for absent in ("", None):
            run = self._run_block(
                tmp_path, load_path=str(parent), load_mode="initialize_next_stage", parent_run_id=absent
            )
            assert "parent_run_id" not in run
        # Without a load there is no lineage at all, whatever else is passed.
        assert "parent_run_id" not in self._run_block(tmp_path, parent_run_id="20260901_120000")

    def test_load_lineage_keys_cover_parent_run_id(self, tmp_path):
        from environments.shared.result_bundle import audit

        assert LOAD_LINEAGE_KEYS[-1] == "parent_run_id"
        assert set(audit._LOAD_LINEAGE_KEYS) == set(LOAD_LINEAGE_KEYS)
        # The audit's rule for the new key: present means a non-empty string.
        base = {"load_path": "01_stance/models/best_model.zip", "load_mode": "initialize_next_stage"}
        _, problems = audit._audit_load_lineage(
            {**base, "parent_run_id": "20260901_120000"}, stage=2, run_path=tmp_path, declared_hashes={}
        )
        assert problems == []
        for bad in ("", "  ", 5, None):
            _, problems = audit._audit_load_lineage(
                {**base, "parent_run_id": bad}, stage=2, run_path=tmp_path, declared_hashes={}
            )
            assert problems == ["stage 2 config run.parent_run_id must be a non-empty string"], bad

    def test_stem_load_path_is_recorded_as_the_zip_it_hashes(self, tmp_path):
        """SB3 loads ``<stem>.zip``; the manifest hashes the ``.zip``.

        A stem recorded as given never matched a manifest key, so the audit's
        parent-hash cross-check never fired for any real SB3 producer
        (train_curriculum and the notebook both hand SB3 a stem).
        """
        from environments.shared.result_bundle import sha256_file

        parent = _sb3_style_zip(tmp_path / "stage1_final.zip", {})
        run = self._run_block(tmp_path, load_path=str(tmp_path / "stage1_final"), load_mode="resume_same_stage")
        assert run["load_path"] == str(parent)
        assert run["parent_checkpoint_sha256"] == sha256_file(parent)

    def test_a_relative_stem_stays_relative(self, tmp_path, monkeypatch):
        (tmp_path / "stage1" / "models").mkdir(parents=True)
        _sb3_style_zip(tmp_path / "stage1" / "models" / "best_model.zip", {})
        monkeypatch.chdir(tmp_path)
        run = self._run_block(tmp_path, load_path="stage1/models/best_model", load_mode="initialize_next_stage")
        assert run["load_path"] == "stage1/models/best_model.zip"

    def test_an_explicit_zip_path_is_recorded_verbatim(self, tmp_path):
        parent = _sb3_style_zip(tmp_path / "best_model.zip", {})
        run = self._run_block(tmp_path, load_path=str(parent), load_mode="resume_same_stage")
        assert run["load_path"] == str(parent)

    def test_unfingerprinted_parent_omits_the_task_digest(self, tmp_path):
        parent = _sb3_style_zip(tmp_path / "old.zip", {"n_steps": 2048})
        run = self._run_block(tmp_path, load_path=str(parent), load_mode="resume_same_stage")
        assert "parent_task_sha256" not in run
        assert run["parent_checkpoint_sha256"].startswith("sha256:")

    def test_non_sb3_parent_records_the_file_hash_only(self, tmp_path):
        from environments.shared.result_bundle import sha256_file

        parent = tmp_path / "best_model.pkl"
        parent.write_bytes(b"not a zip")
        run = self._run_block(tmp_path, load_path=str(parent), load_mode="initialize_next_stage")
        assert run["parent_checkpoint_sha256"] == sha256_file(parent)
        assert "parent_task_sha256" not in run

    def test_missing_checkpoint_fails_closed(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            self._run_block(tmp_path, load_path=str(tmp_path / "nope.zip"), load_mode="resume_same_stage")

    def test_the_notebook_records_lineage_from_a_resolved_load_mode(self):
        """The notebook saves its config itself; it must pass the same keys."""
        repo_root = Path(__file__).resolve().parents[3]
        notebook = json.loads((repo_root / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code"]
        cell = next(c for c in cells if "def train_stage(" in c)
        save_call = cell.index("cfg_path = save_stage_config(")
        call_text = cell[save_call : cell.index(")", save_call)]
        assert "load_path=load_path" in call_text
        assert "load_mode=task_load_mode if load_path else None" in call_text
        # Inferred AFTER the save, the recorded mode would be null for every
        # per-stage cell (they leave task_load_mode=None to be inferred).
        assert cell.index("if task_load_mode is None:") < save_call

    def test_the_notebook_binds_evidence_to_the_vecnormalize_it_ran_under(self):
        """Every evidence write names the sidecar ``_eval_forward_vel`` evaluated with.

        The binding used to cover the model file only, so a sidecar write
        landing between evaluation and bundle save was hashed into
        provenance unbound: final and fallback evaluate under the final
        model's statistics, the selected checkpoint under its matched ones.
        """
        repo_root = Path(__file__).resolve().parents[3]
        notebook = json.loads((repo_root / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code"]
        cell = next(c for c in cells if "def train_stage(" in c)
        bindings = []
        for match in re.finditer(r"_lib_save_evaluation_episodes\(", cell):
            call_text = cell[match.start() : cell.index(")", match.start())]
            label = re.search(r'checkpoint_label="(\w+)"', call_text)
            checkpoint = re.search(r"checkpoint_path=(.+),", call_text)
            normalization = re.search(r"normalization_path=(\w+),", call_text)
            assert label and checkpoint and normalization, call_text
            bindings.append((label[1], checkpoint[1], normalization[1]))
        assert bindings == [
            ("final", 'f"{final_path}.zip"', "final_vecnorm_path"),
            ("selected", "best_model_zip", "vecnorm_save_path"),
            ("selected", 'f"{final_path}.zip"', "final_vecnorm_path"),
        ]
        # The sidecars named are the ones the rollouts were run with.
        assert "_eval_forward_vel(\n        model,\n        stage,\n        final_vecnorm_path," in cell
        assert "_eval_forward_vel(\n            model,\n            stage,\n            vecnorm_save_path," in cell
