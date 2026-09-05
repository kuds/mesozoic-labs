"""Focused plant-identity tests for the Ray Tune sweep backend."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from environments.shared.plant_contract import (
    PlantCompatibilityError,
    PlantIdentity,
    attach_plant_identity,
    write_plant_identity,
)
from environments.shared.scripts.sweep import ray_orchestration, ray_tune

from .reporting_helpers import make_plant_identity as _plant_identity


def _install_fake_promotion_loaders(monkeypatch, current: PlantIdentity) -> dict[str, PlantIdentity]:
    identities = {"model": current, "vecnorm": current}

    class FakeArtifact:
        def __init__(self, identity: PlantIdentity, payload: bytes):
            self.payload = payload
            attach_plant_identity(self, identity)

        def save(self, path):
            Path(path).write_bytes(self.payload)

    class FakeAlgorithm:
        @classmethod
        def load(cls, _path, device="auto"):
            assert device == "cpu"
            return FakeArtifact(identities["model"], b"model")

    save_util_module = ModuleType("stable_baselines3.common.save_util")
    setattr(save_util_module, "load_from_pkl", lambda _path: FakeArtifact(identities["vecnorm"], b"stats"))
    common_module = ModuleType("stable_baselines3.common")
    setattr(common_module, "save_util", save_util_module)
    sb3_module = ModuleType("stable_baselines3")
    setattr(sb3_module, "PPO", FakeAlgorithm)
    setattr(sb3_module, "SAC", FakeAlgorithm)
    setattr(sb3_module, "common", common_module)
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_module)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common_module)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.save_util", save_util_module)
    return identities


def test_sweep_identity_sidecar_fails_closed_and_legacy_override_is_missing_only(tmp_path):
    current = _plant_identity()
    identity_path = tmp_path / "plant_identity.json"

    with pytest.raises(PlantCompatibilityError, match="has no plant identity"):
        ray_orchestration._validate_plant_identity_sidecar(
            identity_path,
            current,
            artifact="Ray sweep state",
        )

    ray_orchestration._validate_plant_identity_sidecar(
        identity_path,
        current,
        artifact="Ray sweep state",
        allow_legacy_plant=True,
    )

    write_plant_identity(identity_path, _plant_identity(physics_revision=2))
    with pytest.raises(PlantCompatibilityError, match="physics_revision"):
        ray_orchestration._validate_plant_identity_sidecar(
            identity_path,
            current,
            artifact="Ray sweep state",
            allow_legacy_plant=True,
        )


def test_ray_report_checkpoint_packages_plant_identity(monkeypatch):
    current = _plant_identity()
    reports: list[tuple[dict, dict]] = []

    class FakeBaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.num_timesteps = 123

    callbacks_module = ModuleType("stable_baselines3.common.callbacks")
    callbacks_module.BaseCallback = FakeBaseCallback
    common_module = ModuleType("stable_baselines3.common")
    common_module.callbacks = callbacks_module
    sb3_module = ModuleType("stable_baselines3")
    sb3_module.common = common_module
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_module)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common_module)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.callbacks", callbacks_module)

    class FakeCheckpoint:
        @classmethod
        def from_directory(cls, directory):
            identity = json.loads((Path(directory) / ray_tune.PLANT_IDENTITY_FILENAME).read_text())
            return {"plant_identity": identity}

    tune_module = ModuleType("ray.tune")
    tune_module.report = lambda metrics, checkpoint: reports.append((metrics, checkpoint))
    train_module = ModuleType("ray.train")
    train_module.Checkpoint = FakeCheckpoint
    ray_module = ModuleType("ray")
    ray_module.tune = tune_module
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.tune", tune_module)
    monkeypatch.setitem(sys.modules, "ray.train", train_module)

    class Saver:
        def save(self, path):
            Path(path).write_text("artifact")

    eval_callback = SimpleNamespace(
        evaluations_timesteps=[1],
        last_mean_reward=1.5,
        best_mean_reward=2.5,
    )
    callback = ray_tune.RayTuneReportCallback(
        eval_callback,
        Saver(),
        [Saver()],
        "ppo",
        1,
        current,
    )

    assert callback._on_step() is True
    assert reports[0][0]["timesteps"] == 123
    assert reports[0][1]["plant_identity"] == current.to_dict()


def test_trial_metadata_sync_includes_plant_identity_by_default(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "drive"
    source.mkdir()
    write_plant_identity(source / ray_tune.PLANT_IDENTITY_FILENAME, _plant_identity())

    ray_tune._sync_trial_metadata(source, destination)

    assert json.loads((destination / ray_tune.PLANT_IDENTITY_FILENAME).read_text()) == _plant_identity().to_dict()


def test_export_best_trial_validates_source_and_writes_promoted_identity(tmp_path, monkeypatch):
    current = _plant_identity()
    monkeypatch.setattr(ray_orchestration, "current_plant_identity", lambda _species: current)
    embedded_identities = _install_fake_promotion_loaders(monkeypatch, current)

    local_trials = tmp_path / "local_trials"
    trial_dir = local_trials / "trial-1"
    model_dir = trial_dir / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "best_model.zip").write_bytes(b"model")
    (model_dir / "best_model_vecnorm.pkl").write_bytes(b"stats")
    write_plant_identity(trial_dir / ray_orchestration.PLANT_IDENTITY_FILENAME, current)

    best_result = SimpleNamespace(
        config={"ppo_learning_rate": 3e-4, "_species": "velociraptor"},
        metrics={"trial_id": "trial-1", "best_mean_reward": 10.0},
    )
    sweep_dir = tmp_path / "sweep"
    promoted = ray_orchestration.export_best_trial(
        best_result,
        sweep_dir,
        species="velociraptor",
        algorithm="ppo",
        stage=1,
        timesteps=100,
        num_trials=1,
        local_trials_dir=local_trials,
    )

    assert (promoted / "best_model.zip").read_bytes() == b"model"
    assert json.loads((promoted / ray_orchestration.PLANT_IDENTITY_FILENAME).read_text()) == current.to_dict()
    config = json.loads((sweep_dir / "best_trial_config.json").read_text())
    assert config["plant_identity"] == current.to_dict()

    # A current directory sidecar must not mask an incompatible identity
    # embedded in the actual SB3 model payload being promoted.
    embedded_identities["model"] = replace(current, physics_revision=2)
    with pytest.raises(PlantCompatibilityError, match="physics_revision"):
        ray_orchestration.export_best_trial(
            best_result,
            tmp_path / "second_sweep",
            species="velociraptor",
            algorithm="ppo",
            stage=1,
            timesteps=100,
            num_trials=1,
            local_trials_dir=local_trials,
            allow_legacy_plant=True,
        )

    embedded_identities["model"] = current
    embedded_identities["vecnorm"] = replace(current, policy_interface_revision=2)
    with pytest.raises(PlantCompatibilityError, match="policy_interface_revision"):
        ray_orchestration.export_best_trial(
            best_result,
            tmp_path / "third_sweep",
            species="velociraptor",
            algorithm="ppo",
            stage=1,
            timesteps=100,
            num_trials=1,
            local_trials_dir=local_trials,
        )


def test_run_ray_sweep_rejects_incompatible_resume_state_before_restore(tmp_path, monkeypatch):
    current = _plant_identity()
    monkeypatch.setattr(ray_orchestration, "current_plant_identity", lambda _species: current)

    restore_calls: list[dict] = []

    class FakeTuner:
        @classmethod
        def restore(cls, **kwargs):
            restore_calls.append(kwargs)
            return cls()

        def fit(self):
            return "results"

    tune_module = ModuleType("ray.tune")
    tune_module.with_resources = lambda train_fn, _resources: train_fn
    tune_module.Tuner = FakeTuner
    ray_module = ModuleType("ray")
    ray_module.tune = tune_module
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.tune", tune_module)

    local_ray_dir = tmp_path / "ray"
    (local_ray_dir / "velociraptor_stage1_ppo").mkdir(parents=True)
    sweep_dir = tmp_path / "sweep"
    write_plant_identity(
        sweep_dir / ray_orchestration.PLANT_IDENTITY_FILENAME,
        replace(current, policy_interface_revision=2),
    )
    run_kwargs = {
        "train_fn": lambda _config: None,
        "search_space": {},
        "species": "velociraptor",
        "algorithm": "ppo",
        "stage": 1,
        "timesteps": 100,
        "n_envs": 1,
        "seed": 1,
        "eval_freq": 10,
        "num_trials": 1,
        "max_concurrent": 1,
        "local_ray_dir": local_ray_dir,
        "local_trials_dir": tmp_path / "trials",
        "sweep_dir": sweep_dir,
        "resume": True,
        "allow_legacy_plant": True,
    }

    with pytest.raises(PlantCompatibilityError, match="policy_interface_revision"):
        ray_orchestration.run_ray_sweep(**run_kwargs)

    assert restore_calls == []

    # The same explicit override may migrate genuinely untagged state, and
    # the resumed Tuner receives the override for any legacy warm-starts.
    (sweep_dir / ray_orchestration.PLANT_IDENTITY_FILENAME).unlink()
    assert ray_orchestration.run_ray_sweep(**run_kwargs) == "results"
    assert restore_calls[0]["param_space"]["_allow_legacy_plant"] is True
    assert json.loads((sweep_dir / ray_orchestration.PLANT_IDENTITY_FILENAME).read_text()) == current.to_dict()
    experiment_identity = (
        sweep_dir / "ray_results" / "velociraptor_stage1_ppo" / ray_orchestration.PLANT_IDENTITY_FILENAME
    )
    assert json.loads(experiment_identity.read_text()) == current.to_dict()
