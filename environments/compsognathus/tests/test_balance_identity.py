"""Research checkpoint separation and fail-closed bundle verification."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.compsognathus.experiments import balance_identity as identity_module
from environments.compsognathus.experiments.balance_env import make_balance_env
from environments.compsognathus.experiments.balance_identity import (
    CHECKPOINT_SCHEMA,
    STUDY_IDENTITY_ATTRIBUTE,
    attach_study_identity,
    build_study_identity,
    load_study_checkpoint,
    save_study_checkpoint,
    validate_study_environment,
    validate_study_identity,
)
from environments.shared.plant_contract import (
    MODEL_IDENTITY_ATTRIBUTE,
    PlantCompatibilityError,
    PlantIdentity,
    validate_model_plant,
)


@pytest.fixture
def env():
    instance = make_balance_env("A")
    yield instance
    instance.close()


@pytest.fixture
def identity(env):
    return build_study_identity(env, {"arm": "A", "seed": 42, "ppo": {"n_steps": 16}})


def test_study_identity_detaches_config_and_metadata(env):
    config = {"arm": "A", "ppo": {"n_steps": 16}}
    identity = build_study_identity(env, config)
    artifact = SimpleNamespace()
    attach_study_identity(artifact, identity)
    config["ppo"]["n_steps"] = 32
    assert identity["config"]["ppo"]["n_steps"] == 16
    identity["config"]["ppo"]["n_steps"] = 64
    assert getattr(artifact, STUDY_IDENTITY_ATTRIBUTE)["config"]["ppo"]["n_steps"] == 16


@pytest.mark.parametrize("other_arm", ["B", "C", "D"])
def test_equal_dimensions_do_not_allow_wrong_arm(identity, other_arm):
    other = make_balance_env(other_arm)
    try:
        other_identity = build_study_identity(other, {**identity["config"], "arm": other_arm})
        assert other_identity["plant_identity"]["action_dim"] == identity["plant_identity"]["action_dim"]
        assert other_identity["plant_identity"]["observation_dim"] == identity["plant_identity"]["observation_dim"]
        with pytest.raises(PlantCompatibilityError, match="identity mismatch"):
            validate_study_identity(other_identity, identity)
    finally:
        other.close()


def test_canonical_loader_rejects_study_even_with_identical_dimensions(identity):
    artifact = SimpleNamespace()
    attach_study_identity(artifact, identity)
    canonical = replace(PlantIdentity.from_mapping(identity["plant_identity"]), species="compsognathus")
    with pytest.raises(PlantCompatibilityError, match="species"):
        validate_model_plant(artifact, canonical)
    canonical_artifact = SimpleNamespace(**{MODEL_IDENTITY_ATTRIBUTE: canonical.to_dict()})
    with pytest.raises(PlantCompatibilityError, match="refusing to relabel"):
        attach_study_identity(canonical_artifact, identity)


def test_config_and_corrupted_metadata_fail_closed(env, identity):
    changed = build_study_identity(env, {**identity["config"], "seed": 43})
    with pytest.raises(PlantCompatibilityError, match="config"):
        validate_study_identity(changed, identity)
    corrupted = copy.deepcopy(identity)
    corrupted["config"]["seed"] = 43
    with pytest.raises(PlantCompatibilityError, match="digest mismatch"):
        validate_study_identity(corrupted, identity)
    with pytest.raises(PlantCompatibilityError, match="missing or invalid"):
        validate_study_identity(None, identity)


@pytest.mark.parametrize("mutation", ["reward", "physics", "filter_alpha"])
def test_live_environment_mutation_is_rejected(env, identity, mutation):
    if mutation == "reward":
        env.alive_bonus += 0.2
    elif mutation == "physics":
        env.model.body_mass[env.pelvis_id] *= 1.01
    else:
        env._action_filter_alpha += 0.1
    with pytest.raises(PlantCompatibilityError, match="identity mismatch|compiled model differs"):
        validate_study_environment(env, identity)


def test_source_change_is_rejected_before_loading(env, identity, monkeypatch):
    changed_sources = {**identity["source_files"], "injected-source.py": "sha256:" + "f" * 64}
    monkeypatch.setattr(identity_module, "_source_files", lambda: changed_sources)
    with pytest.raises(PlantCompatibilityError, match="source_files"):
        validate_study_environment(env, identity)


def test_canonical_xml_change_cannot_be_retagged(env, monkeypatch):
    original = identity_module._file_digest
    canonical = identity_module.MODEL_PATHS["biological"]
    monkeypatch.setattr(
        identity_module,
        "_file_digest",
        lambda path: "sha256:" + "f" * 64 if path == canonical else original(path),
    )
    with pytest.raises(PlantCompatibilityError, match="canonical model source changed"):
        build_study_identity(env, {"arm": "A"})


def _fake_bundle(tmp_path, identity):
    prefix = tmp_path / "step_16"
    model, normalizer, manifest = identity_module._paths(prefix)
    model.write_bytes(b"placeholder-model; must not be unpickled")
    normalizer.write_bytes(b"placeholder-normalizer; must not be unpickled")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity,
        "pair_id": "pair-one",
        "model": {"filename": model.name, "sha256": identity_module._file_digest(model)},
        "normalizer": {"filename": normalizer.name, "sha256": identity_module._file_digest(normalizer)},
    }
    manifest.write_text(json.dumps(payload))
    return prefix, model, normalizer, manifest


@pytest.mark.parametrize("tampered", ["model", "normalizer"])
def test_both_file_hashes_are_checked_before_any_pickle_load(tmp_path, identity, tampered, monkeypatch):
    prefix, model_path, norm_path, _ = _fake_bundle(tmp_path, identity)
    (model_path if tampered == "model" else norm_path).write_bytes(b"tampered")

    def forbidden_load(*args, **kwargs):
        pytest.fail("pickle loading began before bundle verification")

    monkeypatch.setattr(VecNormalize, "load", forbidden_load)
    monkeypatch.setattr(PPO, "load", forbidden_load)
    vector = DummyVecEnv([lambda: make_balance_env("A")])
    try:
        with pytest.raises(PlantCompatibilityError, match=f"{tampered} checkpoint hash"):
            load_study_checkpoint(prefix, identity, vector)
    finally:
        vector.close()


def test_matching_hashes_do_not_allow_wrong_embedded_normalizer_pair(tmp_path, identity, monkeypatch):
    prefix, _, _, _ = _fake_bundle(tmp_path, identity)
    wrong_pair_normalizer = SimpleNamespace(_balance_study_checkpoint_pair="pair-two")
    attach_study_identity(wrong_pair_normalizer, identity)
    monkeypatch.setattr(VecNormalize, "load", lambda *args, **kwargs: wrong_pair_normalizer)
    monkeypatch.setattr(PPO, "load", lambda *args, **kwargs: pytest.fail("model loaded after wrong normalizer"))
    vector = DummyVecEnv([lambda: make_balance_env("A")])
    try:
        with pytest.raises(PlantCompatibilityError, match="different checkpoint pair"):
            load_study_checkpoint(prefix, identity, vector)
    finally:
        vector.close()


def test_wrong_task_in_normalizer_is_rejected_before_model_load(tmp_path, identity, monkeypatch):
    prefix, _, _, _ = _fake_bundle(tmp_path, identity)
    wrong_normalizer = SimpleNamespace(_balance_study_checkpoint_pair="pair-one")
    other = make_balance_env("C")
    try:
        attach_study_identity(wrong_normalizer, build_study_identity(other, {"arm": "C"}))
    finally:
        other.close()
    monkeypatch.setattr(VecNormalize, "load", lambda *args, **kwargs: wrong_normalizer)
    monkeypatch.setattr(PPO, "load", lambda *args, **kwargs: pytest.fail("model loaded after wrong task"))
    vector = DummyVecEnv([lambda: make_balance_env("A")])
    try:
        with pytest.raises(PlantCompatibilityError, match="identity mismatch"):
            load_study_checkpoint(prefix, identity, vector)
    finally:
        vector.close()


def test_saving_requires_tagged_artifacts(tmp_path, identity):
    with pytest.raises(PlantCompatibilityError, match="missing or invalid"):
        save_study_checkpoint(SimpleNamespace(), SimpleNamespace(), tmp_path / "unsafe", identity)
    assert not list(tmp_path.iterdir())


def test_canonical_environment_is_not_an_implicit_study():
    canonical = identity_module.CompsognathusEnv()
    try:
        with pytest.raises(PlantCompatibilityError, match="explicit balance-study"):
            build_study_identity(canonical, {"arm": "A"})
    finally:
        canonical.close()
