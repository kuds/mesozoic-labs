"""Behavior transfer preserves the walker while separating experimental artifacts."""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")
import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared.behavior_checkpoint import (  # noqa: E402
    BEHAVIOR_IDENTITY_SCHEMA,
    COMMAND_LAYERS,
    BehaviorCheckpointError,
    BehaviorVecNormalize,
    load_behavior_checkpoint,
    prepare_behavior_checkpoint,
)
from environments.shared.plant_contract import (  # noqa: E402
    MODEL_IDENTITY_ATTRIBUTE,
    PlantCompatibilityError,
    attach_plant_identity,
    current_plant_identity,
    validate_model_plant,
)
from environments.shared.result_bundle import sha256_file  # noqa: E402
from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE  # noqa: E402

BEHAVIOR = {"schema": "test-behavior/v1", "terrain": "flat", "command_speed": 1.0}


class CommandEnv(gym.Env):
    """Cheap dynamics with the canonical tensor shape and explicit command inputs."""

    def __init__(self, live=False):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(64,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(15,), dtype=np.float32)
        self.live = live
        self.steps = 0

    def _obs(self):
        obs = self.np_random.normal(0.0, 0.1, 64).astype(np.float32)
        obs[-3:] = (0.7, -0.2, 0.4) if self.live else (0.0, 0.0, 0.0)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        reward = float(1.0 - np.square(action - 0.1).mean())
        return self._obs(), reward, False, self.steps >= 8, {}


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    root = tmp_path_factory.mktemp("behavior-parent")
    normalizer = VecNormalize(DummyVecEnv([CommandEnv]))
    model = PPO(
        "MlpPolicy",
        normalizer,
        seed=7,
        device="cpu",
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"net_arch": [16, 16]},
    )
    model.learn(32)
    identity = current_plant_identity("trex")
    for obj in (model, normalizer):
        attach_plant_identity(obj, identity)
    setattr(model, MODEL_TASK_ATTRIBUTE, {"schema": "mesozoic.task-fingerprint/v2", "species": "trex", "stage": 2})
    # Nonzero synthetic moments make accidental carryover into the newly live
    # command connections observable, without changing the parent's function.
    for name in COMMAND_LAYERS:
        param = model.policy.get_submodule(name).weight
        for value in model.policy.optimizer.state[param].values():
            if torch.is_tensor(value) and value.shape == param.shape:
                value[:, -3:] = 0.123
    normalizer.obs_rms.var[-3:] = 1e-11
    model_path, vecnorm_path = root / "walker.zip", root / "walker_vecnorm.pkl"
    model.save(model_path)
    normalizer.save(str(vecnorm_path))
    yield model_path, vecnorm_path, model, normalizer, identity
    normalizer.close()


def _metadata_copy(source: Path, target: Path, key: str, value):
    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(target, "w") as out:
        for info in archive.infolist():
            data = archive.read(info.filename)
            if info.filename == "data":
                metadata = json.loads(data)
                if value is None:
                    metadata.pop(key, None)
                else:
                    metadata[key] = value
                data = json.dumps(metadata).encode()
            out.writestr(info, data)


def test_prepare_preserves_body_weights_stats_and_clears_command_moments(parent):
    model_path, vecnorm_path, original, stats, identity = parent
    source_hashes = (sha256_file(model_path), sha256_file(vecnorm_path))
    prepared, normalizer, report = prepare_behavior_checkpoint(
        model_path,
        vecnorm_path,
        CommandEnv(live=True),
        behavior_identity=BEHAVIOR,
    )
    try:
        assert isinstance(normalizer, BehaviorVecNormalize)
        assert report["max_action_delta"] <= 1e-6
        assert report["max_value_delta"] <= 1e-6
        assert report["parent_training_timesteps"] == 32
        for name, value in original.policy.state_dict().items():
            actual = prepared.policy.state_dict()[name]
            if name in {f"{layer}.weight" for layer in COMMAND_LAYERS}:
                torch.testing.assert_close(actual[:, :-3], value[:, :-3], rtol=0, atol=0)
                assert torch.count_nonzero(actual[:, -3:]) == 0
            else:
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
        for name in COMMAND_LAYERS:
            old_param = original.policy.get_submodule(name).weight
            new_param = prepared.policy.get_submodule(name).weight
            for key, old_value in original.policy.optimizer.state[old_param].items():
                new_value = prepared.policy.optimizer.state[new_param][key]
                if torch.is_tensor(old_value) and old_value.shape == old_param.shape:
                    torch.testing.assert_close(new_value[:, :-3], old_value[:, :-3], rtol=0, atol=0)
                    assert torch.count_nonzero(new_value[:, -3:]) == 0
                else:
                    torch.testing.assert_close(new_value, old_value, rtol=0, atol=0)
        np.testing.assert_array_equal(normalizer.obs_rms.mean[:-3], stats.obs_rms.mean[:-3])
        np.testing.assert_array_equal(normalizer.obs_rms.var[:-3], stats.obs_rms.var[:-3])
        np.testing.assert_array_equal(normalizer.obs_rms.mean[-3:], 0)
        np.testing.assert_array_equal(normalizer.obs_rms.var[-3:], 1)
        assert normalizer.obs_rms.count == stats.obs_rms.count
        assert normalizer.training and not normalizer.norm_reward
        assert getattr(prepared, MODEL_IDENTITY_ATTRIBUTE)["schema"] == BEHAVIOR_IDENTITY_SCHEMA
        for artifact in (prepared, normalizer):
            for allow_legacy in (False, True):
                with pytest.raises(PlantCompatibilityError, match="unsupported plant identity schema"):
                    validate_model_plant(artifact, identity, allow_legacy=allow_legacy)
        assert source_hashes == (sha256_file(model_path), sha256_file(vecnorm_path))
    finally:
        normalizer.close()


def test_commands_bypass_statistics_clipping_and_survive_saved_reload(parent, tmp_path):
    model_path, vecnorm_path, *_ = parent
    model, normalizer, _ = prepare_behavior_checkpoint(
        model_path,
        vecnorm_path,
        CommandEnv(live=True),
        behavior_identity=BEHAVIOR,
    )
    try:
        normalizer.clip_obs = 0.1
        normalizer.obs_rms.mean[-3:] = 500
        normalizer.obs_rms.var[-3:] = 1e-20
        raw = np.ones((2, 64), dtype=np.float32)
        raw[:, -3:] = [0.7, -0.2, 0.4]
        normalized = normalizer.normalize_obs(raw)
        np.testing.assert_array_equal(normalized[:, -3:], raw[:, -3:])
        np.testing.assert_array_equal(normalizer.unnormalize_obs(normalized)[:, -3:], raw[:, -3:])
        assert np.max(np.abs(normalized[:, :-3])) <= 0.1
        model.learn(32)
        assert all(torch.isfinite(t).all() for t in model.policy.state_dict().values())
        assert any(torch.count_nonzero(model.policy.get_submodule(name).weight[:, -3:]) for name in COMMAND_LAYERS)
        state = copy.deepcopy(model.policy.state_dict())
        moments = copy.deepcopy(model.policy.optimizer.state_dict())
        rms = copy.deepcopy(normalizer.obs_rms)
        model.save(tmp_path / "behavior.zip")
        normalizer.save(str(tmp_path / "behavior.pkl"))
    finally:
        normalizer.close()
    resumed, loaded, report = load_behavior_checkpoint(
        tmp_path / "behavior.zip",
        tmp_path / "behavior.pkl",
        CommandEnv(live=True),
        behavior_identity=BEHAVIOR,
    )
    try:
        for name, value in state.items():
            torch.testing.assert_close(resumed.policy.state_dict()[name], value, rtol=0, atol=0)
        for key, saved_state in moments["state"].items():
            for field, value in saved_state.items():
                torch.testing.assert_close(resumed.policy.optimizer.state_dict()["state"][key][field], value)
        np.testing.assert_array_equal(loaded.obs_rms.mean, rms.mean)
        np.testing.assert_array_equal(loaded.obs_rms.var, rms.var)
        assert loaded.obs_rms.count == rms.count
        np.testing.assert_array_equal(loaded.normalize_obs(raw)[:, -3:], raw[:, -3:])
        assert report["resume_checkpoint_sha256"] == sha256_file(tmp_path / "behavior.zip")
        resumed.learn(16, reset_num_timesteps=False)
        assert resumed.num_timesteps == 48
    finally:
        loaded.close()
    with pytest.raises(BehaviorCheckpointError, match="configuration/source identity differs"):
        load_behavior_checkpoint(
            tmp_path / "behavior.zip",
            tmp_path / "behavior.pkl",
            CommandEnv(),
            behavior_identity={"terrain": "other"},
        )


@pytest.mark.parametrize("identity_kind", ["missing", "old", "behavior"])
def test_parent_identity_is_checked_before_policy_load(parent, tmp_path, identity_kind):
    model_path, vecnorm_path, _, _, identity = parent
    raw = identity.to_dict()
    if identity_kind == "missing":
        raw = None
    elif identity_kind == "old":
        raw["policy_interface_revision"] -= 1
    else:
        raw = {"schema": BEHAVIOR_IDENTITY_SCHEMA}
    source = tmp_path / "incompatible.zip"
    _metadata_copy(model_path, source, MODEL_IDENTITY_ATTRIBUTE, raw)
    with pytest.raises(PlantCompatibilityError):
        prepare_behavior_checkpoint(source, vecnorm_path, CommandEnv())


def test_refuses_wrong_task_missing_sidecar_and_existing_normalization(parent, tmp_path):
    model_path, vecnorm_path, *_ = parent
    wrong_task = tmp_path / "stance.zip"
    _metadata_copy(model_path, wrong_task, MODEL_TASK_ATTRIBUTE, {"species": "trex", "stage": 1})
    with pytest.raises(BehaviorCheckpointError, match="locomotion task"):
        prepare_behavior_checkpoint(wrong_task, vecnorm_path, CommandEnv())
    with pytest.raises(BehaviorCheckpointError, match="must exist"):
        prepare_behavior_checkpoint(model_path, tmp_path / "absent.pkl", CommandEnv())
    wrapped = VecNormalize(DummyVecEnv([CommandEnv]))
    try:
        with pytest.raises(BehaviorCheckpointError, match="unnormalized"):
            prepare_behavior_checkpoint(model_path, vecnorm_path, wrapped)
    finally:
        wrapped.close()


@pytest.fixture(scope="module")
def learned_behavior(parent, tmp_path_factory):
    from dataclasses import asdict

    from environments.shared.direction_commands import DirectionCommandConfig

    model_path, vecnorm_path, _, _, identity = parent
    task = {
        "schema": "mesozoic.trex-command-terrain/v1",
        "backend": "stable-baselines3",
        "parent_plant": identity.to_dict(),
        "sources": {"behavior_env.py": "fixed-source-digest"},
        "commands": asdict(DirectionCommandConfig()),
        "env": {"frame_skip": 5, "height_weight": 0.3, "healthy_z_range": [0.7, 1.55]},
        "terrain": None,
        "tracking_weight": 2.5,
        "flat_probability": 0.0,
    }
    model, normalizer, _ = prepare_behavior_checkpoint(
        model_path,
        vecnorm_path,
        CommandEnv(live=True),
        behavior_identity=task,
    )
    model.learn(32)
    root = tmp_path_factory.mktemp("learned-behavior")
    model_path, vecnorm_path = root / "heading.zip", root / "heading.pkl"
    model.save(model_path)
    normalizer.save(str(vecnorm_path))
    yield model_path, vecnorm_path, model, normalizer, task
    normalizer.close()


@pytest.mark.parametrize("sample_terrain", [False, True])
def test_adaptation_keeps_learned_commands_and_records_new_stage(learned_behavior, tmp_path, sample_terrain):
    from dataclasses import asdict

    from environments.shared.behavior_checkpoint import adapt_behavior_checkpoint
    from environments.shared.terrain_sampling import TerrainSamplerConfig, sampler_source_identity

    model_path, vecnorm_path, parent_model, parent_stats, previous = learned_behavior
    requested = copy.deepcopy(previous)
    requested["commands"]["speed_range"] = [0.4, 1.2]
    requested["commands"]["switch_interval_s"] = 2.0
    requested["terrain"] = {"mode": "gentle", "max_slope_degrees": 3.0}
    requested["flat_probability"] = 0.25
    if sample_terrain:
        requested["flat_probability"] = 0.0
        requested["terrain_sampler"] = asdict(TerrainSamplerConfig())
        requested["sampler_sources"] = sampler_source_identity()
    requested["tracking_weight"] = 3.0
    requested["env"]["height_weight"] = 0.6
    adapted, normalizer, report = adapt_behavior_checkpoint(
        model_path,
        vecnorm_path,
        CommandEnv(live=True),
        behavior_identity=requested,
    )
    try:
        assert any(
            torch.count_nonzero(parent_model.policy.get_submodule(name).weight[:, -3:]) for name in COMMAND_LAYERS
        )
        for name, value in parent_model.policy.state_dict().items():
            torch.testing.assert_close(adapted.policy.state_dict()[name], value, rtol=0, atol=0)
        for key, saved_state in parent_model.policy.optimizer.state_dict()["state"].items():
            for field, value in saved_state.items():
                torch.testing.assert_close(
                    adapted.policy.optimizer.state_dict()["state"][key][field], value, rtol=0, atol=0
                )
        np.testing.assert_array_equal(normalizer.obs_rms.mean, parent_stats.obs_rms.mean)
        np.testing.assert_array_equal(normalizer.obs_rms.var, parent_stats.obs_rms.var)
        assert normalizer.obs_rms.count == parent_stats.obs_rms.count
        transition = report["transitions"][-1]
        assert transition["parent_checkpoint_sha256"] == sha256_file(model_path)
        assert transition["parent_normalization_sha256"] == sha256_file(vecnorm_path)
        assert transition["previous_behavior_identity"]["terrain"] is None
        assert transition["behavior_identity"] == requested
        assert getattr(adapted, MODEL_IDENTITY_ATTRIBUTE)["behavior_identity"] == requested
        adapted.learn(16, reset_num_timesteps=False)
        assert adapted.num_timesteps == 48
        adapted.save(tmp_path / "adapted.zip")
        normalizer.save(str(tmp_path / "adapted.pkl"))
    finally:
        normalizer.close()
    resumed, loaded, _ = load_behavior_checkpoint(
        tmp_path / "adapted.zip",
        tmp_path / "adapted.pkl",
        CommandEnv(live=True),
        behavior_identity=requested,
    )
    try:
        assert resumed.num_timesteps == 48
    finally:
        loaded.close()
    with pytest.raises(BehaviorCheckpointError, match="configuration/source identity differs"):
        load_behavior_checkpoint(
            tmp_path / "adapted.zip",
            tmp_path / "adapted.pkl",
            CommandEnv(live=True),
            behavior_identity=previous,
        )


@pytest.mark.parametrize("change", ["stale_source", "unknown_source", "missing_source", "missing_config", "bad_weight"])
@pytest.mark.parametrize("source_side", [False, True])
def test_sampler_transition_refuses_unverified_sources_and_invalid_weights(learned_behavior, change, source_side):
    from dataclasses import asdict

    from environments.shared.behavior_checkpoint import _validate_behavior_transition
    from environments.shared.terrain_sampling import TerrainSamplerConfig, sampler_source_identity

    previous = copy.deepcopy(learned_behavior[-1])
    sampled = copy.deepcopy(previous)
    sampled["terrain_sampler"] = asdict(TerrainSamplerConfig())
    sampled["sampler_sources"] = sampler_source_identity()
    if change == "stale_source":
        key = next(iter(sampled["sampler_sources"]))
        sampled["sampler_sources"][key] = "stale"
    elif change == "unknown_source":
        sampled["sampler_sources"]["unknown.py"] = "unverified"
    elif change == "missing_source":
        del sampled["sampler_sources"]
    elif change == "missing_config":
        del sampled["terrain_sampler"]
    else:
        sampled["terrain_sampler"]["flat"] = -1
    with pytest.raises(BehaviorCheckpointError, match="terrain sampler"):
        _validate_behavior_transition(sampled, previous) if source_side else _validate_behavior_transition(
            previous, sampled
        )


def test_sampler_transition_allows_verified_reweighting_and_return_to_fixed_terrain(learned_behavior):
    from dataclasses import asdict

    from environments.shared.behavior_checkpoint import _validate_behavior_transition
    from environments.shared.terrain_sampling import TerrainSamplerConfig, sampler_source_identity

    fixed = copy.deepcopy(learned_behavior[-1])
    sampled = {
        **fixed,
        "terrain_sampler": asdict(TerrainSamplerConfig()),
        "sampler_sources": sampler_source_identity(),
    }
    reweighted = {**sampled, "terrain_sampler": asdict(TerrainSamplerConfig(flat=2, mixed=0))}
    _validate_behavior_transition(sampled, reweighted)
    _validate_behavior_transition(sampled, fixed)


@pytest.mark.parametrize(
    "field", ["command_scale", "frame_skip", "health_threshold", "source", "parent_plant", "backend"]
)
def test_adaptation_refuses_interface_source_or_dynamics_changes(learned_behavior, field):
    from environments.shared.behavior_checkpoint import adapt_behavior_checkpoint

    model_path, vecnorm_path, _, _, previous = learned_behavior
    requested = copy.deepcopy(previous)
    if field == "command_scale":
        requested["commands"]["speed_scale"] = 2.0
    elif field == "frame_skip":
        requested["env"]["frame_skip"] = 10
    elif field == "health_threshold":
        requested["env"]["healthy_z_range"] = [0.1, 9.0]
    elif field == "source":
        requested["sources"]["behavior_env.py"] = "changed-source"
    elif field == "parent_plant":
        requested["parent_plant"]["physics_revision"] += 1
    elif field == "backend":
        requested["backend"] = "jax-mjx"
    with pytest.raises(BehaviorCheckpointError, match="Incompatible behavior transition fields"):
        adapt_behavior_checkpoint(model_path, vecnorm_path, CommandEnv(), behavior_identity=requested)
