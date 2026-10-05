"""Behavior transfer preserves the walker while separating experimental artifacts.

The train_behaviors CLI checks live here too, beside the CommandEnv they drive.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
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
    COMMAND_LAYERS,
    BehaviorCheckpointError,
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
from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE, MODEL_TASK_LINEAGE_ATTRIBUTE  # noqa: E402

BEHAVIOR = {"schema": "test-behavior/v1", "species": "trex", "env": {"terrain": "flat"}, "task_sha256": "sha256:b"}
#: The stamp every behavior checkpoint carried before consolidation PR-9 (D-D9: evaluation-only).
OLD_MARKER = {
    "schema": "mesozoic.behavior-artifact/v1",
    "species": "trex",
    "behavior_identity": {"schema": "mesozoic.command-terrain/v1"},
    "canonical_certification": False,
}
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
    setattr(
        model,
        MODEL_TASK_ATTRIBUTE,
        {"schema": "mesozoic.task-fingerprint/v2", "species": "trex", "stage": 2, "task_sha256": "sha256:walker"},
    )
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
        task_fingerprint=BEHAVIOR,
    )
    try:
        assert type(normalizer) is VecNormalize
        assert report["command_stats_reseeded"] is True
        assert report["command_normalization"] == (
            "reseeded to mean 0 / variance 1, count kept; statistics keep updating"
        )
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
        # Consolidation PR-9: the canonical stamps -- the plant, the task fingerprint, the lineage.
        for artifact in (prepared, normalizer):
            validate_model_plant(artifact, identity)
            assert getattr(artifact, MODEL_IDENTITY_ATTRIBUTE) == identity.to_dict()
            assert getattr(artifact, MODEL_TASK_ATTRIBUTE) == BEHAVIOR
            assert getattr(artifact, MODEL_TASK_LINEAGE_ATTRIBUTE) == report
            assert not hasattr(artifact, "mesozoic_behavior_preparation")
        assert report["mode"] == "initialize_next_stage" and report["task_fingerprint"] == BEHAVIOR
        assert (report["parent_task_sha256"], report["child_task_sha256"]) == ("sha256:walker", "sha256:b")
        assert report["schema"] == "mesozoic.behavior-preparation/v2"
        assert source_hashes == (sha256_file(model_path), sha256_file(vecnorm_path))
    finally:
        normalizer.close()


def test_commands_follow_reseeded_statistics_and_survive_saved_reload(parent, tmp_path):
    """Decision D-D3: invariant 8 is the one rule; no command passthrough survives preparation or a reload."""
    model_path, vecnorm_path, *_ = parent
    model, normalizer, _ = prepare_behavior_checkpoint(
        model_path,
        vecnorm_path,
        CommandEnv(live=True),
        task_fingerprint=BEHAVIOR,
    )
    raw = np.ones((2, 64), dtype=np.float32)
    raw[:, -3:] = [0.7, -0.2, 0.4]
    try:
        # Reseeded to mean 0 / variance 1: a command c enters as c / sqrt(1 + eps), O(1) from the first step.
        count = normalizer.obs_rms.count
        expected = raw[:, -3:] / np.sqrt(1.0 + normalizer.epsilon)
        np.testing.assert_allclose(normalizer.normalize_obs(raw)[:, -3:], expected, rtol=1e-6)
        normalizer.clip_obs = 0.1
        np.testing.assert_allclose(normalizer.normalize_obs(raw)[:, -3:], np.clip(expected, -0.1, 0.1), rtol=1e-6)
        normalizer.clip_obs = 10.0
        model.learn(32)
        # The command statistics keep updating: the live commands pull the reseeded mean off zero.
        assert normalizer.obs_rms.count > count
        assert np.all(normalizer.obs_rms.mean[-3:] * np.array([0.7, -0.2, 0.4]) > 0)
        assert all(torch.isfinite(t).all() for t in model.policy.state_dict().values())
        assert any(torch.count_nonzero(model.policy.get_submodule(name).weight[:, -3:]) for name in COMMAND_LAYERS)
        state = copy.deepcopy(model.policy.state_dict())
        moments = copy.deepcopy(model.policy.optimizer.state_dict())
        rms = copy.deepcopy(normalizer.obs_rms)
        normalized = normalizer.normalize_obs(raw)
        model.save(tmp_path / "behavior.zip")
        normalizer.save(str(tmp_path / "behavior.pkl"))
    finally:
        normalizer.close()
    assert b"BehaviorVecNormalize" not in (tmp_path / "behavior.pkl").read_bytes()
    resumed, loaded, report = load_behavior_checkpoint(
        tmp_path / "behavior.zip",
        tmp_path / "behavior.pkl",
        CommandEnv(live=True),
        task_fingerprint=BEHAVIOR,
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
        assert type(loaded) is VecNormalize
        np.testing.assert_array_equal(loaded.normalize_obs(raw), normalized)
        assert report["resume_checkpoint_sha256"] == sha256_file(tmp_path / "behavior.zip")
        # The preparation report travels as the task lineage and comes back on resume.
        assert report["schema"] == "mesozoic.behavior-preparation/v2" and report["task_fingerprint"] == BEHAVIOR
        resumed.learn(16, reset_num_timesteps=False)
        assert resumed.num_timesteps == 48
    finally:
        loaded.close()
    with pytest.raises(BehaviorCheckpointError, match=r"differs from the saved task in \['env'\]"):
        load_behavior_checkpoint(
            tmp_path / "behavior.zip",
            tmp_path / "behavior.pkl",
            CommandEnv(),
            task_fingerprint={**BEHAVIOR, "env": {"terrain": "other"}},
        )


def test_a_normalizer_stamped_for_another_task_is_refused(parent, tmp_path):
    """The normalizer's task stamp must be the model's: a sidecar prepared for another task is not this pair."""
    model_path, vecnorm_path, *_ = parent
    for name, task in (("this", BEHAVIOR), ("other", {**BEHAVIOR, "task_sha256": "sha256:other"})):
        model, normalizer, _ = prepare_behavior_checkpoint(
            model_path, vecnorm_path, CommandEnv(live=True), task_fingerprint=task
        )
        try:
            model.save(tmp_path / f"{name}.zip")
            normalizer.save(str(tmp_path / f"{name}.pkl"))
        finally:
            normalizer.close()
    with pytest.raises(BehaviorCheckpointError, match="model and normalization identities disagree"):
        load_behavior_checkpoint(
            tmp_path / "this.zip", tmp_path / "other.pkl", CommandEnv(live=True), task_fingerprint=BEHAVIOR
        )


@pytest.mark.parametrize("mode", ["resume", "adapt"])
def test_a_behavior_checkpoint_from_before_the_task_fingerprint_is_refused_by_name(parent, tmp_path, mode):
    """Decision D-D9: every older behavior bundle is evaluation-only; nothing loads one as a parent."""
    from environments.shared.behavior_checkpoint import adapt_behavior_checkpoint

    model_path, vecnorm_path, *_ = parent
    model, normalizer, _ = prepare_behavior_checkpoint(
        model_path, vecnorm_path, CommandEnv(live=True), task_fingerprint=BEHAVIOR
    )
    try:
        model.save(tmp_path / "new.zip")
        normalizer.save(str(tmp_path / "new.pkl"))
    finally:
        normalizer.close()
    # Before PR-9 the marker was both stamps: the plant identity and the task fingerprint.
    _metadata_copy(tmp_path / "new.zip", tmp_path / "half.zip", MODEL_IDENTITY_ATTRIBUTE, OLD_MARKER)
    _metadata_copy(tmp_path / "half.zip", tmp_path / "old.zip", MODEL_TASK_ATTRIBUTE, OLD_MARKER)
    load = load_behavior_checkpoint if mode == "resume" else adapt_behavior_checkpoint
    requested = OLD_MARKER if mode == "resume" else BEHAVIOR
    with pytest.raises(BehaviorCheckpointError, match="records no task fingerprint.*D-D9.*git_commit"):
        load(tmp_path / "old.zip", tmp_path / "new.pkl", CommandEnv(live=True), task_fingerprint=requested)


@pytest.mark.parametrize("name", ["BehaviorVecNormalize", "OtherVecNormalize"])
def test_sidecar_pickling_the_deleted_passthrough_class_is_refused(parent, tmp_path, monkeypatch, name):
    """A behavior sidecar saved before PR-8 pickles BehaviorVecNormalize: refused by name (D-D3), never loaded as a
    plain VecNormalize and never a bare unpickling AttributeError. Only that class: any other class missing from
    the module stays Python's own AttributeError."""
    from environments.shared import behavior_checkpoint

    model_path, vecnorm_path, *_ = parent
    model, normalizer, _ = prepare_behavior_checkpoint(
        model_path, vecnorm_path, CommandEnv(live=True), task_fingerprint=BEHAVIOR
    )
    retired = type(name, (VecNormalize,), {"__module__": behavior_checkpoint.__name__})
    try:
        with monkeypatch.context() as patch:
            patch.setattr(behavior_checkpoint, name, retired, raising=False)
            normalizer.__class__ = retired
            model.save(tmp_path / "old.zip")
            normalizer.save(str(tmp_path / "old.pkl"))
    finally:
        normalizer.close()
    assert not hasattr(behavior_checkpoint, name)
    assert b"environments.shared.behavior_checkpoint" in (tmp_path / "old.pkl").read_bytes()
    assert name.encode() in (tmp_path / "old.pkl").read_bytes()
    refusal = (
        pytest.raises(BehaviorCheckpointError, match="command-passthrough class BehaviorVecNormalize.*D-D3")
        if name == "BehaviorVecNormalize"
        else pytest.raises(AttributeError, match=name)
    )
    with refusal:
        load_behavior_checkpoint(
            tmp_path / "old.zip", tmp_path / "old.pkl", CommandEnv(live=True), task_fingerprint=BEHAVIOR
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
        raw = OLD_MARKER
    source = tmp_path / "incompatible.zip"
    _metadata_copy(model_path, source, MODEL_IDENTITY_ATTRIBUTE, raw)
    with pytest.raises(PlantCompatibilityError):
        prepare_behavior_checkpoint(source, vecnorm_path, CommandEnv(), task_fingerprint=BEHAVIOR)


@pytest.mark.parametrize("fingerprint", [None, {}, {"schema": "test-behavior/v1", "species": "trex"}])
def test_preparation_requires_the_recipe_task_fingerprint(parent, fingerprint):
    """An artifact stamped without the recipe env's task_sha256 could never be resumed or adapted (the loaders would
    refuse it as one from before the task fingerprint, D-D9), so preparation refuses it before loading anything."""
    model_path, vecnorm_path, *_ = parent
    if fingerprint is None:
        with pytest.raises(TypeError, match="task_fingerprint"):
            prepare_behavior_checkpoint(model_path, vecnorm_path, CommandEnv(live=True))
        return
    message = r"needs the recipe env's task fingerprint \(env\.task_fingerprint\) with its task_sha256: .*adapted$"
    with pytest.raises(BehaviorCheckpointError, match=message):
        prepare_behavior_checkpoint(model_path, vecnorm_path, None, task_fingerprint=fingerprint)


@pytest.mark.parametrize("task", ["locomotion", "stage_id"])
def test_preparation_refuses_a_task_a_manifest_node_could_resume(parent, task):
    """D-D9 under the canonical stamps: a stage id stamped on a behavior artifact would let canonical train_base
    resume or warm-start from it, so preparation refuses one before loading anything (the recipe env's stage never
    is one): the canonical locomotion task's own fingerprint, and a stage-id string."""
    from environments.shared.task_fingerprint import stage_task_fingerprint

    model_path, vecnorm_path, *_ = parent
    fingerprint = {**BEHAVIOR, "stage": "behavior"}
    if task == "locomotion":
        fingerprint = stage_task_fingerprint("trex", "locomotion")
    with pytest.raises(BehaviorCheckpointError, match="whose stage is never a stage id"):
        prepare_behavior_checkpoint(model_path, vecnorm_path, None, task_fingerprint=fingerprint)


def test_a_behavior_checkpoint_is_not_a_walker_parent(parent, tmp_path):
    """Consolidation PR-9: its plant stamp is now the plant identity, so preparation refuses it by its task."""
    from environments.shared.behavior_env import RECIPE_TASK_STAGE

    model_path, vecnorm_path, *_ = parent
    model, normalizer, _ = prepare_behavior_checkpoint(
        model_path, vecnorm_path, CommandEnv(live=True), task_fingerprint={**BEHAVIOR, "stage": RECIPE_TASK_STAGE}
    )
    try:
        model.save(tmp_path / "behavior.zip")
        normalizer.save(str(tmp_path / "behavior.pkl"))
    finally:
        normalizer.close()
    with pytest.raises(BehaviorCheckpointError, match="must record the trex locomotion task"):
        prepare_behavior_checkpoint(
            tmp_path / "behavior.zip", tmp_path / "behavior.pkl", CommandEnv(), task_fingerprint=BEHAVIOR
        )


def test_refuses_wrong_task_missing_sidecar_and_existing_normalization(parent, tmp_path):
    model_path, vecnorm_path, *_ = parent
    wrong_task = tmp_path / "stance.zip"
    _metadata_copy(model_path, wrong_task, MODEL_TASK_ATTRIBUTE, {"species": "trex", "stage": 1})
    with pytest.raises(BehaviorCheckpointError, match="locomotion task"):
        prepare_behavior_checkpoint(wrong_task, vecnorm_path, CommandEnv(), task_fingerprint=BEHAVIOR)
    with pytest.raises(BehaviorCheckpointError, match="must exist"):
        prepare_behavior_checkpoint(model_path, tmp_path / "absent.pkl", CommandEnv(), task_fingerprint=BEHAVIOR)
    wrapped = VecNormalize(DummyVecEnv([CommandEnv]))
    try:
        with pytest.raises(BehaviorCheckpointError, match="unnormalized"):
            prepare_behavior_checkpoint(model_path, vecnorm_path, wrapped, task_fingerprint=BEHAVIOR)
    finally:
        wrapped.close()


@pytest.fixture(scope="module")
def learned_behavior(parent, tmp_path_factory):
    from environments.shared.behavior_env import RECIPE_TASK_STAGE
    from environments.shared.direction_commands import DirectionCommandConfig, DirectionCommandController
    from environments.shared.task_fingerprint import compute_task_fingerprint

    model_path, vecnorm_path, _, _, identity = parent
    config = DirectionCommandConfig()
    task = compute_task_fingerprint(
        species="trex",
        stage=RECIPE_TASK_STAGE,
        backend="stable-baselines3",
        env_kwargs={
            "height_weight": 0.3,
            "command_mode": "heading_and_speed",
            "command_config": config,
            "terrain": None,
            "terrain_sampler": None,
            "tracking_weight": 2.5,
            "course_distance": 10.0,
        },
        plant_identity=identity.to_dict(),
        perturbation_manifest=None,
        command_manifest=DirectionCommandController(config).manifest(),
    )
    model, normalizer, _ = prepare_behavior_checkpoint(
        model_path,
        vecnorm_path,
        CommandEnv(live=True),
        task_fingerprint=task,
    )
    model.learn(32)
    root = tmp_path_factory.mktemp("learned-behavior")
    model_path, vecnorm_path = root / "heading.zip", root / "heading.pkl"
    model.save(model_path)
    normalizer.save(str(vecnorm_path))
    yield model_path, vecnorm_path, model, normalizer, task
    normalizer.close()


@pytest.mark.parametrize(
    "weights",
    [
        None,
        {"flat": 1, "sloped": 3, "bumps": 0, "depressions": 0, "mixed": 0, "terrain_contact": 0},
        {"flat": 1, "sloped": 1, "bumps": 1, "depressions": 1, "mixed": 1, "terrain_contact": 0},
    ],
)
def test_adaptation_keeps_learned_commands_and_records_new_stage(learned_behavior, parent, tmp_path, weights):
    from dataclasses import asdict

    from environments.shared.behavior_checkpoint import adapt_behavior_checkpoint
    from environments.shared.terrain_sampling import TerrainSamplerConfig

    model_path, vecnorm_path, parent_model, parent_stats, previous = learned_behavior
    requested = copy.deepcopy(previous)
    requested["command"]["config"]["speed_range"] = [0.4, 1.2]
    requested["command"]["config"]["switch_interval_s"] = 2.0
    requested["env"]["command_config"] = copy.deepcopy(requested["command"]["config"])
    requested["env"]["terrain"] = {"mode": "gentle", "max_slope_degrees": 3.0}
    if weights is not None:
        requested["env"]["terrain_sampler"] = asdict(TerrainSamplerConfig(**weights))
    requested["env"]["tracking_weight"] = 3.0
    requested["env"]["height_weight"] = 0.6
    requested["task_sha256"] = "sha256:requested"
    adapted, normalizer, report = adapt_behavior_checkpoint(
        model_path,
        vecnorm_path,
        CommandEnv(live=True),
        task_fingerprint=requested,
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
        assert transition["previous_task_fingerprint"] == previous
        assert transition["task_fingerprint"] == requested
        assert transition["schema"] == "mesozoic.behavior-transition/v2"
        assert (report["task_fingerprint"], report["child_task_sha256"]) == (requested, "sha256:requested")
        for artifact in (adapted, normalizer):
            assert getattr(artifact, MODEL_TASK_ATTRIBUTE) == requested
            assert getattr(artifact, MODEL_IDENTITY_ATTRIBUTE) == parent[4].to_dict()
            assert getattr(artifact, MODEL_TASK_LINEAGE_ATTRIBUTE) == report
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
        task_fingerprint=requested,
    )
    try:
        assert resumed.num_timesteps == 48
    finally:
        loaded.close()
    with pytest.raises(BehaviorCheckpointError, match="task fingerprint differs from the saved task"):
        load_behavior_checkpoint(
            tmp_path / "adapted.zip",
            tmp_path / "adapted.pkl",
            CommandEnv(live=True),
            task_fingerprint=previous,
        )


def test_the_lineage_names_the_task_an_adaptation_stamps(learned_behavior):
    """After an adaptation the lineage agrees with the stamps: its task fingerprint and child_task_sha256 name the
    requested task, its parent stays the walker, and the step from the prepared task is its transition."""
    from environments.shared.behavior_checkpoint import adapt_behavior_checkpoint

    model_path, vecnorm_path, _, _, previous = learned_behavior
    requested = copy.deepcopy(previous)
    requested["command"]["config"]["switch_interval_s"] = 2.0
    requested["env"]["command_config"] = copy.deepcopy(requested["command"]["config"])
    requested["task_sha256"] = "sha256:requested"
    adapted, normalizer, report = adapt_behavior_checkpoint(
        model_path, vecnorm_path, CommandEnv(live=True), task_fingerprint=requested
    )
    try:
        lineage = getattr(adapted, MODEL_TASK_LINEAGE_ATTRIBUTE)
        assert lineage == report == getattr(normalizer, MODEL_TASK_LINEAGE_ATTRIBUTE)
        assert getattr(adapted, MODEL_TASK_ATTRIBUTE)["task_sha256"] == "sha256:requested"
        assert lineage["task_fingerprint"] == requested and lineage["child_task_sha256"] == "sha256:requested"
        assert (lineage["mode"], lineage["parent_task_sha256"]) == ("initialize_next_stage", "sha256:walker")
        assert [step["previous_task_fingerprint"] for step in lineage["transitions"]] == [previous]
    finally:
        normalizer.close()


@pytest.mark.parametrize("fingerprint", ["empty", "allowed_transition"])
def test_adaptation_requires_the_recipe_task_fingerprint(learned_behavior, fingerprint):
    """As for preparation: the adapted artifacts are stamped with the requested fingerprint, and one without its
    task_sha256 could never be resumed or adapted again (the loaders would refuse it as one from before the task
    fingerprint, D-D9), so adaptation refuses it before anything is loaded or stamped, even for a transition that
    is otherwise allowed."""
    from environments.shared.behavior_checkpoint import adapt_behavior_checkpoint

    model_path, vecnorm_path, _, _, previous = learned_behavior
    requested = {} if fingerprint == "empty" else copy.deepcopy(previous)
    if fingerprint == "allowed_transition":
        requested["command"]["config"]["switch_interval_s"] = 2.0
        requested["env"]["command_config"] = copy.deepcopy(requested["command"]["config"])
        del requested["task_sha256"]
    message = (
        r"^Adaptation needs the recipe env's task fingerprint \(env\.task_fingerprint\) with its task_sha256: "
        r"an artifact stamped without one could never be resumed or adapted$"
    )
    with pytest.raises(BehaviorCheckpointError, match=message):
        adapt_behavior_checkpoint(model_path, vecnorm_path, None, task_fingerprint=requested)


@pytest.mark.parametrize("field", ["stage", "command.schema", "command.adapter"])
def test_transition_refuses_another_implementation_version(learned_behavior, field):
    """Versioned strings replace the source hashes (consolidation PR-9): the recipe stage names this module's,
    terrain.py's and terrain_sampling.py's implementation, the command manifest direction_commands.py's."""
    from environments.shared.behavior_checkpoint import _validate_behavior_transition

    previous = copy.deepcopy(learned_behavior[-1])
    requested = copy.deepcopy(previous)
    if field == "stage":
        requested["stage"] = "command-terrain/v3"
    else:
        requested["command"][field.split(".")[1]] += "+next"
    with pytest.raises(BehaviorCheckpointError, match=f"Incompatible behavior transition fields: {re.escape(field)}$"):
        _validate_behavior_transition(previous, requested)


@pytest.mark.parametrize("change", ["bad_weight", "not_a_table", "unknown_family"])
@pytest.mark.parametrize("source_side", [False, True])
def test_sampler_transition_refuses_an_invalid_sampler_table(learned_behavior, change, source_side):
    """A transition may add, remove or reweight the sampling layer, never carry an invalid table on either side:
    consolidation PR-9 deleted the source hashes beside the table, not the table's own check."""
    from dataclasses import asdict

    from environments.shared.behavior_checkpoint import _validate_behavior_transition
    from environments.shared.terrain_sampling import TerrainSamplerConfig

    previous = copy.deepcopy(learned_behavior[-1])
    sampled = copy.deepcopy(previous)
    sampled["env"]["terrain_sampler"] = asdict(TerrainSamplerConfig())
    if change == "bad_weight":
        sampled["env"]["terrain_sampler"]["flat"] = -1
    elif change == "not_a_table":
        sampled["env"]["terrain_sampler"] = "not-a-table"
    else:
        sampled["env"]["terrain_sampler"]["lava"] = 1
    with pytest.raises(BehaviorCheckpointError, match="^Invalid terrain sampler configuration$"):
        _validate_behavior_transition(sampled, previous) if source_side else _validate_behavior_transition(
            previous, sampled
        )


def test_sampler_transition_allows_verified_reweighting_and_return_to_fixed_terrain(learned_behavior):
    from dataclasses import asdict

    from environments.shared.behavior_checkpoint import _validate_behavior_transition
    from environments.shared.terrain_sampling import TerrainSamplerConfig

    fixed = copy.deepcopy(learned_behavior[-1])
    sampled = {**fixed, "env": {**fixed["env"], "terrain_sampler": asdict(TerrainSamplerConfig())}}
    reweighted = {**sampled, "env": {**fixed["env"], "terrain_sampler": asdict(TerrainSamplerConfig(flat=2, mixed=0))}}
    _validate_behavior_transition(sampled, reweighted)
    _validate_behavior_transition(sampled, fixed)


@pytest.mark.parametrize("requested_value", [None, 0.0, 0.25])
def test_flat_probability_is_no_longer_a_transition_setting(learned_behavior, requested_value):
    """Consolidation PR-8 removed it from the behavior identity; an identity that carries it is another task."""
    from environments.shared.behavior_checkpoint import _validate_behavior_transition

    previous = copy.deepcopy(learned_behavior[-1])
    previous["env"]["flat_probability"] = 0.25
    requested = copy.deepcopy(learned_behavior[-1])
    if requested_value is not None:
        requested["env"]["flat_probability"] = requested_value
    if requested_value == 0.25:
        _validate_behavior_transition(previous, requested)
    else:
        with pytest.raises(
            BehaviorCheckpointError, match="Incompatible behavior transition fields: env.flat_probability"
        ):
            _validate_behavior_transition(previous, requested)


@pytest.mark.parametrize("schema", ["mesozoic.trex-command-terrain/v1", "mesozoic.command-terrain/v1"])
def test_retired_identity_schemas_cannot_start_a_transition(learned_behavior, schema):
    """Consolidation PR-7 deleted the trex emitter and PR-9 the shared one; D-D9 keeps those bundles
    evaluation-only, and a canonical stage fingerprint (no command section) is not a behavior task either."""
    from environments.shared.behavior_checkpoint import _validate_behavior_transition

    retired = {
        "schema": schema,
        "backend": "stable-baselines3",
        "commands": learned_behavior[-1]["command"]["config"],
        "env": learned_behavior[-1]["env"],
        "sources": {"behavior_env.py": "fixed-source-digest"},
    }
    canonical = {key: value for key, value in learned_behavior[-1].items() if key != "command"}
    for previous in (retired, canonical):
        with pytest.raises(BehaviorCheckpointError, match="Only supported command/terrain tasks"):
            _validate_behavior_transition(previous, learned_behavior[-1])


@pytest.mark.parametrize(
    "field", ["command_scale", "frame_skip", "health_threshold", "implementation", "parent_plant", "backend"]
)
def test_adaptation_refuses_interface_source_or_dynamics_changes(learned_behavior, field):
    from environments.shared.behavior_checkpoint import adapt_behavior_checkpoint

    model_path, vecnorm_path, _, _, previous = learned_behavior
    requested = copy.deepcopy(previous)
    if field == "command_scale":
        requested["command"]["config"]["speed_scale"] = 2.0
    elif field == "frame_skip":
        requested["env"]["frame_skip"] = 10
    elif field == "health_threshold":
        requested["env"]["healthy_z_range"] = [0.1, 9.0]
    elif field == "implementation":
        requested["stage"] = "command-terrain/v3"
    elif field == "parent_plant":
        requested["plant"]["physics_sha256"] = "sha256:other"
    elif field == "backend":
        requested["backend"] = "jax-mjx"
    with pytest.raises(BehaviorCheckpointError, match="Incompatible behavior transition fields"):
        adapt_behavior_checkpoint(model_path, vecnorm_path, CommandEnv(), task_fingerprint=requested)


def _bundle(tmp_path):
    model, normalizer = tmp_path / "model.zip", tmp_path / "vecnormalize.pkl"
    model.write_bytes(b"selected-model")
    normalizer.write_bytes(b"paired-stats")
    manifest = {
        "schema": train_behaviors.BUNDLE_SCHEMA,
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


def test_resume_refuses_a_bundle_saved_before_the_task_fingerprint_by_name(tmp_path):
    """Decision D-D9: a v1 bundle.json (saved before consolidation PR-9) is refused before anything is loaded."""
    model, normalizer = _bundle(tmp_path)
    manifest = json.loads((tmp_path / "bundle.json").read_text())
    (tmp_path / "bundle.json").write_text(json.dumps({**manifest, "schema": "mesozoic.behavior-bundle/v1"}))
    with pytest.raises(ValueError, match=r"'mesozoic.behavior-bundle/v1'.*evaluation-only \(decision D-D9\)"):
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
            self.task_fingerprint = {"schema": "runner-test/v1", "task": "live-commands", "task_sha256": "sha256:r"}

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
        assert type(normalizer) is VecNormalize
        raw = np.ones((1, 64), dtype=np.float32)
        raw[:, -3:] = [0.7, -0.2, 0.4]
        rms = normalizer.obs_rms
        # D-D3: the saved command statistics moved with training and normalise the commands.
        assert np.all(rms.mean[-3:] * raw[0, -3:] > 0)
        expected = np.clip((raw - rms.mean) / np.sqrt(rms.var + normalizer.epsilon), -10.0, 10.0)
        np.testing.assert_allclose(normalizer.normalize_obs(raw), expected, rtol=1e-6)
        assert normalizer.norm_reward and normalizer.gamma == 0.97
    finally:
        normalizer.close()
    run = json.loads((resumed / "run.json").read_text())
    assert run["training"]["requested_additional_steps"] == 32
    assert run["training"]["actual_additional_steps"] == 32
    assert run["training"]["observation_normalization"] == (
        "every input; command statistics reseeded at preparation (D-D3)"
    )
    train_behaviors._verify_bundle(resumed / "model.zip", resumed / "vecnormalize.pkl")
