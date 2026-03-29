"""Tests for sweep ray_search_space.py — build_search_space and save_search_space."""

import json
from pathlib import Path

from environments.shared.scripts.sweep import (
    build_search_space,
    load_resume_settings,
    save_search_space,
)

# ── build_search_space (JSON config) ──────────────────────────────────────


class TestBuildSearchSpace:
    """build_search_space loads from JSON config files."""

    def _write_config(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "sweep.json"
        p.write_text(json.dumps(data))
        return p

    def test_loads_stage_from_json(self, tmp_path):
        config = {
            "stage1": {
                "trials": 10,
                "timesteps": 1000,
                "ppo_lr": {"type": "double", "min": 1e-5, "max": 1e-3, "scale": "log"},
                "env_bonus": {"type": "double", "min": 0.1, "max": 5.0, "scale": "linear"},
            },
            "stage2": {
                "trials": 20,
                "ppo_lr": {"type": "double", "min": 1e-4, "max": 1e-2, "scale": "log"},
            },
        }
        path = self._write_config(tmp_path, config)
        result = build_search_space("velociraptor", 1, "ppo", config_path=path)
        # Should contain only parameter specs, not job settings
        assert "ppo_lr" in result
        assert "env_bonus" in result
        assert "trials" not in result
        assert "timesteps" not in result

    def test_loads_different_stage(self, tmp_path):
        config = {
            "stage1": {"ppo_lr": {"type": "double", "min": 1e-5, "max": 1e-3, "scale": "log"}},
            "stage2": {"ppo_gamma": {"type": "double", "min": 0.95, "max": 0.999, "scale": "linear"}},
        }
        path = self._write_config(tmp_path, config)
        result = build_search_space("trex", 2, "ppo", config_path=path)
        assert "ppo_gamma" in result
        assert "ppo_lr" not in result

    def test_default_config_path_species_ppo(self):
        """Default config path points to species-specific sweep config."""
        from environments.shared.scripts.sweep.ray_search_space import _default_config_path

        path = _default_config_path("ppo", "brachiosaurus")
        assert path.name == "sweep_ppo.json"
        assert "brachiosaurus" in str(path)
        assert path.exists(), f"Expected config at {path}"

    def test_default_config_path_species_sac(self):
        from environments.shared.scripts.sweep.ray_search_space import _default_config_path

        path = _default_config_path("sac", "velociraptor")
        assert path.name == "sweep_sac.json"
        assert "velociraptor" in str(path)
        assert path.exists(), f"Expected config at {path}"

    def test_default_config_path_missing_raises(self):
        """Missing species config raises FileNotFoundError."""
        from environments.shared.scripts.sweep.ray_search_space import _default_config_path

        with pytest.raises(FileNotFoundError):
            _default_config_path("ppo", "nonexistent_species")

    def test_default_config_path_no_species_raises(self):
        """Omitting species raises FileNotFoundError."""
        from environments.shared.scripts.sweep.ray_search_space import _default_config_path

        with pytest.raises(FileNotFoundError):
            _default_config_path("ppo")

    def test_loads_real_ppo_config_stage1(self):
        """Smoke test: load the actual species sweep config for stage 1."""
        result = build_search_space("brachiosaurus", 1, "ppo")
        assert "ppo_learning_rate" in result
        assert "trials" not in result

    def test_loads_real_sac_config_stage1(self):
        result = build_search_space("velociraptor", 1, "sac")
        assert "sac_learning_rate" in result
        assert "trials" not in result


# ── save_search_space ─────────────────────────────────────────────────────


class TestSaveSearchSpace:
    """save_search_space writes the resolved search space to disk."""

    def test_writes_json_file(self, tmp_path):
        space = {
            "ppo_lr": {"type": "double", "min": 1e-5, "max": 1e-3, "scale": "log"},
            "env_bonus": {"type": "double", "min": 0.1, "max": 5.0, "scale": "linear"},
        }
        result_path = save_search_space(space, tmp_path, species="velociraptor", stage=1, algorithm="ppo")
        assert result_path.exists()
        assert result_path.name == "search_space_stage1_ppo.json"

        data = json.loads(result_path.read_text())
        assert data["species"] == "velociraptor"
        assert data["stage"] == 1
        assert data["algorithm"] == "ppo"
        assert data["num_parameters"] == 2
        assert "ppo_lr" in data["parameters"]
        assert "env_bonus" in data["parameters"]

    def test_writes_runtime_section(self, tmp_path):
        space = {"ppo_lr": {"type": "double", "min": 1e-5, "max": 1e-3, "scale": "log"}}
        result_path = save_search_space(
            space,
            tmp_path,
            species="velociraptor",
            stage=1,
            algorithm="ppo",
            max_concurrent=3,
            n_envs=8,
            timesteps_per_trial=6_000_000,
            num_trials=50,
            eval_freq=50_000,
            seed=42,
        )
        data = json.loads(result_path.read_text())
        assert "runtime" in data
        rt = data["runtime"]
        assert "gpu_model" not in rt
        assert rt["max_concurrent"] == 3
        assert rt["n_envs"] == 8
        assert rt["timesteps_per_trial"] == 6_000_000
        assert rt["num_trials"] == 50
        assert rt["eval_freq"] == 50_000
        assert rt["seed"] == 42
        assert rt["use_asha"] is True

    def test_runtime_includes_use_asha_even_with_no_other_fields(self, tmp_path):
        space = {"x": {"type": "double", "min": 0, "max": 1, "scale": "linear"}}
        result_path = save_search_space(space, tmp_path, species="trex", stage=2, algorithm="sac")
        data = json.loads(result_path.read_text())
        # use_asha is always written (defaults to True), so runtime is present
        assert data["runtime"] == {"use_asha": True}

    def test_runtime_partial_fields(self, tmp_path):
        space = {"x": {"type": "double", "min": 0, "max": 1, "scale": "linear"}}
        result_path = save_search_space(space, tmp_path, species="trex", stage=1, algorithm="ppo", n_envs=4, seed=7)
        data = json.loads(result_path.read_text())
        assert "runtime" in data
        rt = data["runtime"]
        assert rt["n_envs"] == 4
        assert rt["seed"] == 7
        assert rt["use_asha"] is True
        assert "gpu_model" not in rt
        assert "max_concurrent" not in rt

    def test_runtime_use_asha_false(self, tmp_path):
        space = {"x": {"type": "double", "min": 0, "max": 1, "scale": "linear"}}
        result_path = save_search_space(
            space,
            tmp_path,
            species="trex",
            stage=1,
            algorithm="ppo",
            use_asha=False,
            n_envs=4,
        )
        data = json.loads(result_path.read_text())
        assert data["runtime"]["use_asha"] is False

    def test_writes_gpu_info_section(self, tmp_path):
        space = {"ppo_lr": {"type": "double", "min": 1e-5, "max": 1e-3, "scale": "log"}}
        gpu = {
            "gpu_model": "A100",
            "gpu_full_name": "NVIDIA A100-SXM4-40GB",
            "gpu_memory_gb": 40.0,
            "cuda_version": "12.1",
        }
        result_path = save_search_space(
            space,
            tmp_path,
            species="velociraptor",
            stage=1,
            algorithm="ppo",
            gpu_info=gpu,
            max_concurrent=3,
        )
        data = json.loads(result_path.read_text())
        # Full GPU info at top level (like stage_config.json)
        assert "gpu" in data
        assert data["gpu"]["gpu_model"] == "A100"
        assert data["gpu"]["gpu_full_name"] == "NVIDIA A100-SXM4-40GB"
        assert data["gpu"]["gpu_memory_gb"] == 40.0
        assert data["gpu"]["cuda_version"] == "12.1"
        # gpu_model should NOT be in runtime (lives in gpu section only)
        assert "gpu_model" not in data["runtime"]

    def test_gpu_info_absent_when_not_provided(self, tmp_path):
        space = {"x": {"type": "double", "min": 0, "max": 1, "scale": "linear"}}
        result_path = save_search_space(space, tmp_path, species="trex", stage=1, algorithm="ppo")
        data = json.loads(result_path.read_text())
        assert "gpu" not in data

    def test_creates_dest_dir(self, tmp_path):
        dest = tmp_path / "nested" / "dir"
        save_search_space({"x": {"type": "double", "min": 0, "max": 1, "scale": "linear"}}, dest)
        assert dest.exists()

    def test_fallback_filename_without_metadata(self, tmp_path):
        path = save_search_space({"x": {"type": "double", "min": 0, "max": 1, "scale": "linear"}}, tmp_path)
        assert path.name == "search_space.json"

    def test_writes_resume_section(self, tmp_path):
        space = {"ppo_lr": {"type": "double", "min": 1e-5, "max": 1e-3, "scale": "log"}}
        result_path = save_search_space(
            space,
            tmp_path,
            species="brachiosaurus",
            stage=2,
            algorithm="ppo",
            config_path="/some/custom/config.json",
        )
        data = json.loads(result_path.read_text())
        assert "resume" in data
        assert data["resume"]["sweep_dir"] == str(tmp_path)
        assert data["resume"]["config_path"] == "/some/custom/config.json"

    def test_resume_section_without_config_path(self, tmp_path):
        space = {"x": {"type": "double", "min": 0, "max": 1, "scale": "linear"}}
        result_path = save_search_space(space, tmp_path, species="trex", stage=1, algorithm="ppo")
        data = json.loads(result_path.read_text())
        assert "resume" in data
        assert data["resume"]["sweep_dir"] == str(tmp_path)
        assert data["resume"]["config_path"] == ""


# ── load_resume_settings ─────────────────────────────────────────────────


class TestLoadResumeSettings:
    """load_resume_settings reads back settings from a saved search space file."""

    def test_loads_all_settings(self, tmp_path):
        space = {
            "ppo_lr": {"type": "double", "min": 1e-5, "max": 1e-3, "scale": "log"},
            "env_bonus": {"type": "double", "min": 0.1, "max": 5.0, "scale": "linear"},
        }
        saved_path = save_search_space(
            space,
            tmp_path,
            species="brachiosaurus",
            stage=2,
            algorithm="ppo",
            max_concurrent=3,
            n_envs=8,
            timesteps_per_trial=16_000_000,
            num_trials=50,
            eval_freq=50_000,
            seed=42,
            use_asha=True,
            grace_period=40,
            reduction_factor=2,
            collapse_min_evals=8,
            collapse_patience=5,
            load_path="/some/model/path",
            config_path="/some/config.json",
        )
        settings = load_resume_settings(saved_path)
        assert settings["species"] == "brachiosaurus"
        assert settings["stage"] == 2
        assert settings["algorithm"] == "ppo"
        assert settings["resume"] is True
        assert settings["resume_sweep_dir"] == str(tmp_path)
        assert settings["sweep_config_path"] == "/some/config.json"
        assert settings["max_concurrent"] == 3
        assert settings["n_envs"] == 8
        assert settings["timesteps_per_trial"] == 16_000_000
        assert settings["num_trials"] == 50
        assert settings["eval_freq"] == 50_000
        assert settings["seed"] == 42
        assert settings["use_asha"] is True
        assert settings["grace_period"] == 40
        assert settings["reduction_factor"] == 2
        assert settings["collapse_min_evals"] == 8
        assert settings["collapse_patience"] == 5
        assert settings["load_path"] == "/some/model/path"

    def test_defaults_config_path_to_file_itself(self, tmp_path):
        space = {"x": {"type": "double", "min": 0, "max": 1, "scale": "linear"}}
        saved_path = save_search_space(space, tmp_path, species="trex", stage=1, algorithm="ppo")
        settings = load_resume_settings(saved_path)
        # When no config_path was stored, falls back to the file itself
        assert settings["sweep_config_path"] == str(saved_path)

    def test_file_not_found(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_resume_settings(tmp_path / "nonexistent.json")

    def test_build_search_space_from_saved_file(self, tmp_path):
        """Saved search space JSON can be used as config_path for build_search_space."""
        space = {
            "ppo_lr": {"type": "double", "min": 1e-5, "max": 1e-3, "scale": "log"},
            "env_bonus": {"type": "double", "min": 0.1, "max": 5.0, "scale": "linear"},
        }
        saved_path = save_search_space(space, tmp_path, species="velociraptor", stage=1, algorithm="ppo")
        # Loading the saved file back via build_search_space should work
        result = build_search_space("velociraptor", 1, "ppo", config_path=saved_path)
        assert "ppo_lr" in result
        assert "env_bonus" in result
        assert "species" not in result
        assert "runtime" not in result
