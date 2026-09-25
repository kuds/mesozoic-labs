"""Behavior transfer preserves the walker while separating experimental artifacts.

The train_behaviors CLI checks live here too, beside the CommandEnv they drive.
"""

from __future__ import annotations

import copy
import hashlib
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

from environments.shared import train_behaviors  # noqa: E402
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
PRESETS = Path(__file__).parents[3] / "configs" / "trex" / "behaviors"


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
        "schema": "mesozoic.command-terrain/v1",
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


def test_retired_trex_identity_schema_cannot_start_a_transition(learned_behavior):
    """Consolidation PR-7 deleted its only emitter (TRexBehaviorEnv); D-D9 keeps those bundles evaluation-only."""
    from environments.shared.behavior_checkpoint import _validate_behavior_transition

    previous = {**copy.deepcopy(learned_behavior[-1]), "schema": "mesozoic.trex-command-terrain/v1"}
    with pytest.raises(BehaviorCheckpointError, match="Only supported command/terrain tasks"):
        _validate_behavior_transition(previous, learned_behavior[-1])


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


def _bundle(tmp_path):
    model, normalizer = tmp_path / "model.zip", tmp_path / "vecnormalize.pkl"
    model.write_bytes(b"selected-model")
    normalizer.write_bytes(b"paired-stats")
    manifest = {
        "schema": "mesozoic.behavior-bundle/v1",
        "model": model.name,
        "normalizer": normalizer.name,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "normalizer_sha256": hashlib.sha256(normalizer.read_bytes()).hexdigest(),
        "canonical_certification": False,
    }
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    return model, normalizer


@pytest.mark.parametrize("changed", ["model", "normalizer", "manifest"])
def test_resume_refuses_missing_or_altered_pair(tmp_path, changed):
    model, normalizer = _bundle(tmp_path)
    train_behaviors._verify_bundle(model, normalizer)
    if changed == "model":
        model.write_bytes(b"other-model")
    elif changed == "normalizer":
        normalizer.write_bytes(b"other-stats")
    else:
        (tmp_path / "bundle.json").unlink()
    with pytest.raises(ValueError, match="bundle|hash"):
        train_behaviors._verify_bundle(model, normalizer)


@pytest.fixture
def stubbed_cli(monkeypatch, tmp_path):
    """Exercise actual CLI orchestration with an explicit no-optimizer model."""
    from environments.shared import behavior_checkpoint

    calls = {"learn": [], "loads": [], "interrupt_after": None, "rollout_starts": []}

    class FakeModel:
        def __init__(self, state=None):
            self.num_timesteps = 8_000_000
            self._n_updates = 1_000
            self.n_steps = 64
            self.batch_size = 16
            self.n_epochs = 1
            self.gamma = 0.99
            self.gae_lambda = 0.95
            self.device = "cpu"
            self.__dict__.update(state or {})

        def set_random_seed(self, seed):
            self.seed = seed

        def set_logger(self, logger):
            pass

        def learn(self, *, total_timesteps, reset_num_timesteps, callback):
            assert not reset_num_timesteps
            calls["learn"].append(total_timesteps)
            if calls["interrupt_after"] is not None:
                self.num_timesteps += calls["interrupt_after"]
                calls["interrupt_after"] = None
                raise KeyboardInterrupt
            start = self.num_timesteps
            for elapsed in calls["rollout_starts"]:
                self.num_timesteps = start + elapsed
                for current_callback in callback:
                    current_callback.model = self
                    current_callback._on_rollout_start()
            self.num_timesteps = start + total_timesteps
            self._n_updates += 1
            return self

        def save(self, path):
            Path(path).write_text(json.dumps(self.__dict__))

    class FakeNormalizer:
        def __init__(self, env):
            self.env = env
            self.norm_reward = False

        def save(self, path):
            Path(path).write_text("paired-normalizer")

        def close(self):
            self.env.close()

    def loader(mode):
        def load(model_path, vecnorm_path, env, **kwargs):
            calls["loads"].append(mode)
            state = json.loads(Path(model_path).read_text())
            return FakeModel(state), FakeNormalizer(env), {"loader": mode}

        return load

    for mode, function in (
        ("prepare", "prepare_behavior_checkpoint"),
        ("resume", "load_behavior_checkpoint"),
        ("adapt", "adapt_behavior_checkpoint"),
    ):
        monkeypatch.setattr(behavior_checkpoint, function, loader(mode))
    parent, normalization = tmp_path / "parent.zip", tmp_path / "parent.pkl"
    parent.write_text(json.dumps({"num_timesteps": 8_000_000}))
    normalization.write_text("parent-normalizer")
    return calls, parent, normalization


def _args(config, checkpoint, normalization, output):
    return [
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--vecnormalize",
        str(normalization),
        "--output",
        str(output),
        "--seed",
        "42",
        "--eval-episodes",
        "0",
    ]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_cli_seed_refuses_before_loading_or_creating_output(tmp_path, capsys, seed):
    output = tmp_path / "output"
    args = _args(tmp_path / "missing.toml", tmp_path / "model.zip", tmp_path / "stats.pkl", output)
    with pytest.raises(SystemExit) as error:
        train_behaviors.main(args + ["--seed", str(seed)])
    assert error.value.code == 2
    assert "--seed must be between 0 and 2**32 - 1" in capsys.readouterr().err
    assert not output.exists()


def test_eval_only_never_learns_or_overwrites_parent(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    original = parent.read_bytes(), normalization.read_bytes()
    output = tmp_path / "evaluation"
    train_behaviors.main(_args(PRESETS / "follow_direction.toml", parent, normalization, output) + ["--eval-only"])
    assert calls["learn"] == []
    assert (parent.read_bytes(), normalization.read_bytes()) == original
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete"
    assert run["training"]["actual_additional_steps"] == 0
    assert run["training"]["requested_additional_steps"] == 0
    train_behaviors._verify_bundle(output / "model.zip", output / "vecnormalize.pkl")


@pytest.mark.parametrize("problem", ["missing_encoder", "excessive_fps"])
def test_unavailable_requested_replay_fails_before_loading_or_learning(stubbed_cli, tmp_path, monkeypatch, problem):
    from environments.shared import behavior_replay

    calls, parent, normalization = stubbed_cli

    def preflight():
        if problem == "missing_encoder":
            raise behavior_replay.ReplayExportError("The requested video encoder is unavailable")

    monkeypatch.setattr(behavior_replay, "require_replay_dependencies", preflight)
    args = _args(PRESETS / "mixed_terrain.toml", parent, normalization, tmp_path / "unavailable-replay")
    args[args.index("--eval-episodes") + 1] = "1"
    args += ["--record-video", "--video-fps", "101" if problem == "excessive_fps" else "25"]
    with pytest.raises((SystemExit, ValueError, behavior_replay.ReplayExportError)):
        train_behaviors.main(args)
    assert calls["loads"] == []
    assert calls["learn"] == []


def test_resume_refuses_changed_ppo_recipe_before_loading(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    source = tmp_path / "first"
    config = PRESETS / "follow_direction.toml"
    train_behaviors.main(_args(config, parent, normalization, source) + ["--steps", "0"])
    changed = tmp_path / "changed.toml"
    changed.write_text(config.read_text().replace("ent_coef = 0.005", "ent_coef = 0.1"))
    previous_loads = list(calls["loads"])
    with pytest.raises((ValueError, SystemExit), match="recipe|PPO|resume|settings"):
        train_behaviors.main(
            _args(changed, source / "model.zip", source / "vecnormalize.pkl", tmp_path / "resumed") + ["--resume"]
        )
    assert calls["loads"] == previous_loads
    assert calls["learn"] == []


def test_resume_finishes_original_budget_without_repeating_full_training(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    source, resumed = tmp_path / "interrupted", tmp_path / "resumed"
    config = tmp_path / "short_recipe.toml"
    config.write_text((PRESETS / "follow_direction.toml").read_text().replace("timesteps = 3000000", "timesteps = 100"))
    calls["interrupt_after"] = 40
    train_behaviors.main(_args(config, parent, normalization, source))
    first = json.loads((source / "run.json").read_text())
    assert first["status"] == "interrupted"
    assert first["training"]["actual_additional_steps"] == 40
    train_behaviors.main(_args(config, source / "model.zip", source / "vecnormalize.pkl", resumed) + ["--resume"])
    assert calls["learn"] == [100, 60]
    run = json.loads((resumed / "run.json").read_text())
    assert run["training"]["actual_additional_steps"] == 60
    assert json.loads((resumed / "bundle.json").read_text())["num_timesteps"] == 8_000_100


def test_cli_rejects_conflicting_load_modes_and_nonempty_output(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    args = _args(PRESETS / "follow_direction.toml", parent, normalization, tmp_path / "unused")
    with pytest.raises(SystemExit):
        train_behaviors.main(args + ["--resume", "--adapt"])
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep.txt"
    sentinel.write_text("preserved")
    with pytest.raises(SystemExit):
        train_behaviors.main(_args(PRESETS / "follow_direction.toml", parent, normalization, occupied))
    assert sentinel.read_text() == "preserved"
    assert calls["loads"] == [] and calls["learn"] == []


def test_periodic_rollout_checkpoints_are_complete_matched_bundles(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    calls["rollout_starts"] = [100_000, 200_000]
    output = tmp_path / "periodic"
    train_behaviors.main(
        _args(PRESETS / "follow_direction.toml", parent, normalization, output) + ["--steps", "250000"]
    )
    checkpoints = sorted((output / "checkpoints").iterdir())
    assert [path.name for path in checkpoints] == ["step-8100000", "step-8200000"]
    for checkpoint in checkpoints:
        train_behaviors._verify_bundle(checkpoint / "model.zip", checkpoint / "vecnormalize.pkl")
        metadata = json.loads((checkpoint / "bundle.json").read_text())
        assert metadata["training_recipe"]["ppo"]["ent_coef"] == 0.005
    latest = json.loads((output / "latest_checkpoint.json").read_text())
    assert latest["num_timesteps"] == 8_200_000
    assert Path(latest["directory"]) == checkpoints[-1]


def test_interrupted_evaluation_keeps_trained_bundle_and_records_status(stubbed_cli, tmp_path, monkeypatch):
    from environments.shared import behavior_evaluation

    calls, parent, normalization = stubbed_cli
    output = tmp_path / "interrupted-evaluation"
    saved = {}

    def interrupt(*args, **kwargs):
        saved.update({name: (output / name).read_bytes() for name in ("model.zip", "vecnormalize.pkl", "bundle.json")})
        raise KeyboardInterrupt

    monkeypatch.setattr(behavior_evaluation, "evaluate_behavior", interrupt)
    args = _args(PRESETS / "follow_direction.toml", parent, normalization, output)
    args[args.index("--eval-episodes") + 1] = "1"
    train_behaviors.main(args + ["--steps", "100"])
    assert calls["learn"] == [100]
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "interrupted"
    assert run["training"]["actual_additional_steps"] == 100
    assert "evaluation" not in run
    assert saved == {name: (output / name).read_bytes() for name in saved}
    train_behaviors._verify_bundle(output / "model.zip", output / "vecnormalize.pkl")


def test_interrupted_bundle_save_finishes_a_resumable_pair(stubbed_cli, tmp_path, monkeypatch):
    _, parent, normalization = stubbed_cli
    output = tmp_path / "interrupted-save"
    save = train_behaviors._save_bundle
    attempts = []

    def interrupt_once(model, normalizer, directory, identity, recipe):
        attempts.append(directory)
        if len(attempts) == 1:
            model.save(str(directory / "model.zip"))
            raise KeyboardInterrupt
        save(model, normalizer, directory, identity, recipe)

    monkeypatch.setattr(train_behaviors, "_save_bundle", interrupt_once)
    train_behaviors.main(_args(PRESETS / "follow_direction.toml", parent, normalization, output) + ["--steps", "100"])
    assert attempts == [output, output]
    assert json.loads((output / "run.json").read_text())["status"] == "interrupted"
    train_behaviors._verify_bundle(output / "model.zip", output / "vecnormalize.pkl")


def test_real_ppo_cli_resume_preserves_recipe_and_releases_stage_warmup(tmp_path, monkeypatch):
    """Actual PPO updates exercise the loader/runner boundary and persisted anchor."""
    torch.set_num_threads(1)
    parent_normalizer = VecNormalize(DummyVecEnv([CommandEnv]), gamma=0.97)
    parent = PPO(
        "MlpPolicy",
        parent_normalizer,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        gamma=0.97,
        gae_lambda=0.91,
        policy_kwargs={"net_arch": [16, 16]},
        seed=7,
        device="cpu",
    )
    checkpoint, stats = tmp_path / "parent.zip", tmp_path / "parent.pkl"
    try:
        parent.learn(32)
        for artifact in (parent, parent_normalizer):
            attach_plant_identity(artifact, current_plant_identity("trex"))
        setattr(parent, MODEL_TASK_ATTRIBUTE, {"species": "trex", "stage": 2})
        parent.save(checkpoint)
        parent_normalizer.save(str(stats))
    finally:
        parent_normalizer.close()

    class PilotEnv(CommandEnv):
        def __init__(self, **kwargs):
            super().__init__(live=True)
            self.behavior_identity = {"schema": "runner-test/v1", "task": "live-commands"}

    monkeypatch.setattr(train_behaviors, "get_behavior_env_class", lambda species: PilotEnv)
    config = tmp_path / "short.toml"
    config.write_text(
        '[behavior]\nspecies = "trex"\nname = "short"\nparent = "locomotion"\ntimesteps = 48\n[ppo]\nwarmup_timesteps = 24\n'
    )
    updates = []
    train = PPO.train

    def record_update(model):
        updates.append((model.num_timesteps, model.clip_range(1.0), model.lr_schedule(1.0), model.ent_coef))
        return train(model)

    monkeypatch.setattr(PPO, "train", record_update)
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    train_behaviors.main(_args(config, checkpoint, stats, first) + ["--steps", "16"])
    train_behaviors.main(_args(config, first / "model.zip", first / "vecnormalize.pkl", resumed) + ["--resume"])
    assert updates == [(48, 0.02, 5e-5, 0.005), (64, 0.2, 5e-5, 0.005), (80, 0.2, 5e-5, 0.005)]
    saved = PPO.load(resumed / "model.zip", device="cpu")
    assert saved.num_timesteps == 80
    assert saved.mesozoic_behavior_stage_start == 32
    assert (saved.n_steps, saved.batch_size, saved.n_epochs) == (16, 8, 2)
    assert (saved.gamma, saved.gae_lambda) == (0.97, 0.91)
    assert saved.target_kl == 0.03
    assert all(torch.isfinite(value).all() for value in saved.policy.state_dict().values())
    normalizer = VecNormalize.load(str(resumed / "vecnormalize.pkl"), DummyVecEnv([PilotEnv]))
    try:
        raw = np.ones((1, 64), dtype=np.float32)
        raw[:, -3:] = [0.7, -0.2, 0.4]
        np.testing.assert_array_equal(normalizer.normalize_obs(raw)[:, -3:], raw[:, -3:])
        assert normalizer.norm_reward and normalizer.gamma == 0.97
    finally:
        normalizer.close()
    run = json.loads((resumed / "run.json").read_text())
    assert run["training"]["requested_additional_steps"] == 32
    assert run["training"]["actual_additional_steps"] == 32
    train_behaviors._verify_bundle(resumed / "model.zip", resumed / "vecnormalize.pkl")
