"""SB3 artifacts on the heading-free observation (plant_versions note 18, D-D31): the SB3 job's half.

The heading-free migration takes the T-Rex to policy interface r14 (63-wide)
and the anatomical Compsognathus to r4 (55-wide).  These tests run genuine
Stable-Baselines3 artifacts through the repository's own loaders on the new
production environments:

* a PPO model and its VecNormalize statistics save, load
  (``load_sb3_checkpoint``) and evaluate (``evaluation.evaluate``) with the
  plant check on, acting exactly as before the save, the statistics 63 / 55
  wide;
* an interrupted run resumes on the new environment
  (``train_base._load_vecnorm_into_envs`` then ``_create_or_load_model``, as
  ``_train_stage_body`` runs them), carrying its step count and return
  statistics;
* a pre-migration pair -- 64 / 56 wide, stamped with the committed r13 / r3
  identities -- is refused by plant, naming ``policy_interface_revision``
  and ``observation_dim``, on every path that could otherwise reuse it:
  the trainer's warm start and resume, its statistics loader,
  ``load_sb3_checkpoint`` (with the old model or only the old statistics),
  ``evaluation.evaluate``, the certified-ancestor rule behind a manual
  ``TRUNK_FROM`` and ``TRUNK_FROM = "auto"``, the widen tool's identity gate
  and the behavior preparation and resume (with the old walker, or a current
  walker beside only the old statistics).

The old pairs are synthetic (a 64- / 56-wide stand-in env) but carry the
real pre-migration identities: what every loader reads is the identity, and
the old width is what used to make SB3 fail first with a shape error.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.plant_contract import (  # noqa: E402
    PlantCompatibilityError,
    PlantIdentity,
    attach_plant_identity,
    current_plant_identity,
)
from environments.shared.species_registry import get_species_config  # noqa: E402

from .ancestors_helpers import STANCE_CURRICULUM, STANCE_TASK, build_trunk_run  # noqa: E402

SPECIES = ("trex", "compsognathus")
#: The pre-migration policy interfaces, as the committed plant manifest recorded them before note 18.
PRE_MIGRATION: dict[str, dict[str, Any]] = {
    "trex": {
        "policy_interface_revision": 13,
        "observation_dim": 64,
        "policy_interface_sha256": "sha256:0b43de19860b3b6e8ea7d010d8b7069b265b14040d57ece7fffab49a9cb51252",
    },
    "compsognathus": {
        "policy_interface_revision": 3,
        "observation_dim": 56,
        "policy_interface_sha256": "sha256:8253fcf4ef8bc65e809cbe3caf552f897b30e5f6b1fae8af56ba66da0edf1abd",
    },
}
#: Current widths: the 4-wide quaternion became 3-wide projected gravity.
CURRENT_WIDTH = {"trex": 63, "compsognathus": 55}
PPO_KWARGS: dict[str, Any] = {
    "n_steps": 64,
    "batch_size": 32,
    "n_epochs": 1,
    "seed": 0,
    "device": "cpu",
    "policy_kwargs": {"net_arch": [16]},
}


def _factory(species: str):
    env_class = get_species_config(species).env_class
    kwargs = load_stage_config(species, "stance")["env_kwargs"]
    return lambda: env_class(**kwargs)


def _old_identity(species: str) -> PlantIdentity:
    return dataclasses.replace(current_plant_identity(species), **PRE_MIGRATION[species])


def _refused(excinfo, species: str) -> None:
    message = str(excinfo.value)
    old = PRE_MIGRATION[species]
    current = current_plant_identity(species)
    assert (
        f"policy_interface_revision: checkpoint={old['policy_interface_revision']}, "
        f"current={current.policy_interface_revision}" in message
    ), message
    assert f"observation_dim: checkpoint={old['observation_dim']}, current={current.observation_dim}" in message


@pytest.fixture(scope="module", params=SPECIES)
def species(request) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def trained(species, tmp_path_factory):
    """A briefly trained PPO pair on the current stance env, saved the way a run saves its handoff pair."""
    root = tmp_path_factory.mktemp(f"heading-free-{species}") / "01_stance" / "models"
    root.mkdir(parents=True)
    identity = current_plant_identity(species)
    venv = VecNormalize(DummyVecEnv([_factory(species)]), norm_reward=True)
    model = PPO("MlpPolicy", venv, **PPO_KWARGS)
    attach_plant_identity(model, identity)
    attach_plant_identity(venv, identity)
    model.learn(128)
    model.save(root / "robust_best_model")
    venv.save(str(root / "robust_best_model_vecnorm.pkl"))
    yield {"species": species, "root": root, "model": model, "venv": venv, "identity": identity}
    venv.close()


@pytest.fixture(scope="module")
def pre_migration(species, tmp_path_factory):
    """A pre-migration pair: the old width, stamped with the committed r13 (trex) / r3 (compsognathus) identity."""
    width = PRE_MIGRATION[species]["observation_dim"]
    actions = current_plant_identity(species).action_dim

    class _OldInterfaceEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(actions,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(width, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(width, dtype=np.float32), 0.0, False, False, {}

    root = tmp_path_factory.mktemp(f"pre-migration-{species}")
    old = _old_identity(species)
    venv = VecNormalize(DummyVecEnv([_OldInterfaceEnv]))
    model = PPO("MlpPolicy", venv, **PPO_KWARGS)
    attach_plant_identity(model, old)
    attach_plant_identity(venv, old)
    model.save(root / "robust_best_model")
    venv.save(str(root / "robust_best_model_vecnorm.pkl"))
    venv.close()
    return {"species": species, "root": root, "identity": old}


def test_ppo_save_load_evaluate_and_normalizer_round_trip(trained, caplog):
    from environments.shared.evaluation import evaluate
    from environments.shared.policy_loading import load_sb3_checkpoint

    species, root, identity = trained["species"], trained["root"], trained["identity"]
    factory = _factory(species)
    loaded, normalizer, resolved = load_sb3_checkpoint(
        str(root / "robust_best_model.zip"), None, factory, plant_identity=identity
    )
    try:
        assert resolved == str(root / "robust_best_model_vecnorm.pkl")
        assert normalizer.obs_rms.mean.shape == (CURRENT_WIDTH[species],) == (identity.observation_dim,)
        np.testing.assert_array_equal(normalizer.obs_rms.mean, trained["venv"].obs_rms.mean)
        np.testing.assert_array_equal(normalizer.obs_rms.var, trained["venv"].obs_rms.var)
        env = factory()
        try:
            obs, _ = env.reset(seed=3042)
            for _ in range(50):
                expected = trained["model"].predict(trained["venv"].normalize_obs(obs), deterministic=True)[0]
                action = loaded.predict(normalizer.normalize_obs(obs), deterministic=True)[0]
                np.testing.assert_array_equal(action, expected)
                obs, *_ = env.step(action)
        finally:
            env.close()
    finally:
        normalizer.close()
    # The CLI evaluator: the sidecar is found, both artifacts pass the plant check, one episode is scored.
    with caplog.at_level("INFO", logger="environments.shared.evaluation"):
        evaluate(
            species_cfg=get_species_config(species),
            stage_configs={"stance": load_stage_config(species, "stance")},
            model_path=str(root / "robust_best_model.zip"),
            n_episodes=1,
            render=False,
            stage="stance",
        )
    assert any("Evaluation Results (1 episodes)" in record.message for record in caplog.records)


def test_interrupted_run_resumes_on_the_new_env(trained):
    from environments.shared import train_base

    species, root, identity = trained["species"], trained["root"], trained["identity"]
    stem = str(root / "robust_best_model")
    train_env = VecNormalize(DummyVecEnv([_factory(species)]), norm_reward=True)
    eval_env = VecNormalize(DummyVecEnv([_factory(species)]), norm_reward=False)
    try:
        train_base._load_vecnorm_into_envs(
            stem, train_env, eval_env, plant_identity=identity, task_load_mode="resume_same_stage"
        )
        np.testing.assert_array_equal(train_env.obs_rms.mean, trained["venv"].obs_rms.mean)
        np.testing.assert_array_equal(train_env.ret_rms.var, trained["venv"].ret_rms.var)
        sb3 = {"PPO": PPO, "SAC": None}
        kwargs = {key: value for key, value in PPO_KWARGS.items() if key not in ("seed", "device", "policy_kwargs")}
        model = train_base._create_or_load_model(sb3, "ppo", kwargs, train_env, stem, plant_identity=identity)
        assert model.num_timesteps == trained["model"].num_timesteps == 128
        model.learn(64, reset_num_timesteps=False)
        assert model.num_timesteps == 192
        assert model.observation_space.shape == (CURRENT_WIDTH[species],)
    finally:
        train_env.close()
        eval_env.close()


def test_pre_migration_pair_is_refused_by_the_trainer(species, pre_migration):
    from environments.shared import train_base

    stem = str(pre_migration["root"] / "robust_best_model")
    current = current_plant_identity(species)
    train_env = VecNormalize(DummyVecEnv([_factory(species)]))
    eval_env = VecNormalize(DummyVecEnv([_factory(species)]))
    try:
        for mode in ("resume_same_stage", "initialize_next_stage"):
            with pytest.raises(PlantCompatibilityError) as excinfo:
                train_base._load_vecnorm_into_envs(
                    stem, train_env, eval_env, plant_identity=current, task_load_mode=mode
                )
            _refused(excinfo, species)
            # Warm start and resume alike refuse the model before SB3 rebuilds it around the new env.
            with pytest.raises(PlantCompatibilityError) as excinfo:
                train_base._create_or_load_model(
                    {"PPO": PPO, "SAC": None},
                    "ppo",
                    {"n_steps": 64, "batch_size": 32},
                    train_env,
                    stem,
                    plant_identity=current,
                    task_load_mode=mode,
                )
            _refused(excinfo, species)
    finally:
        train_env.close()
        eval_env.close()


def test_pre_migration_pair_is_refused_by_the_loaders(species, pre_migration, trained):
    from environments.shared.evaluation import evaluate
    from environments.shared.policy_loading import load_sb3_checkpoint

    old_root = pre_migration["root"]
    current = current_plant_identity(species)
    factory = _factory(species)
    with pytest.raises(PlantCompatibilityError) as excinfo:
        load_sb3_checkpoint(str(old_root / "robust_best_model.zip"), None, factory, plant_identity=current)
    _refused(excinfo, species)
    # A current model beside pre-migration statistics: refused by plant, not as a damaged file.
    with pytest.raises(PlantCompatibilityError) as excinfo:
        load_sb3_checkpoint(
            str(trained["root"] / "robust_best_model.zip"),
            str(old_root / "robust_best_model_vecnorm.pkl"),
            factory,
            plant_identity=current,
        )
    _refused(excinfo, species)
    with pytest.raises(PlantCompatibilityError) as excinfo:
        evaluate(
            species_cfg=get_species_config(species),
            stage_configs={"stance": load_stage_config(species, "stance")},
            model_path=str(old_root / "robust_best_model.zip"),
            n_episodes=1,
            render=False,
            stage="stance",
        )
    _refused(excinfo, species)


def test_pre_migration_pair_is_refused_as_a_trunk_and_by_the_widen_gate(pre_migration, tmp_path, monkeypatch):
    from environments.shared.ancestors import AncestorReuseError, find_certified_ancestor, select_trunk
    from environments.shared.scripts.widen_checkpoint import identity_gate_errors
    from environments.shared.stage_manifest import load_stage_manifest

    species, old = pre_migration["species"], pre_migration["identity"]
    current = current_plant_identity(species)
    run = tmp_path / "20261009_155723"
    run.mkdir()
    build_trunk_run(run, plant=old, species=species)
    stance = load_stage_manifest(species).by_id("stance")
    for follow_records in (False, True):  # the chain loop's own run, and a manual TRUNK_FROM
        with pytest.raises(AncestorReuseError) as excinfo:
            find_certified_ancestor(
                run,
                species=species,
                entry=stance,
                current_task_sha256=STANCE_TASK,
                plant_identity=current,
                current_gate_config=STANCE_CURRICULUM,
                follow_records=follow_records,
            )
        _refused(excinfo, species)

    # TRUNK_FROM = "auto": the run is scanned and refused at its root.
    from environments.shared import task_fingerprint as task_fingerprint_module

    monkeypatch.setattr(
        task_fingerprint_module, "derive_stage_task_fingerprint", lambda **kwargs: {"task_sha256": STANCE_TASK}
    )
    manifest = load_stage_manifest(species)
    chain = manifest.chain_for("locomotion")
    stage_configs = {
        entry.reference: {"env_kwargs": {}, "curriculum_kwargs": STANCE_CURRICULUM, "name": entry.id} for entry in chain
    }
    selection = select_trunk(
        tmp_path, species=species, chain=chain, stage_configs=stage_configs, plant_identity=current
    )
    assert selection.selected is None
    ((candidate),) = selection.candidates
    assert candidate.coverage == 0 and candidate.refusal is not None and candidate.refusal[0] == "stance"
    assert "policy_interface_revision" in candidate.refusal[1] and "observation_dim" in candidate.refusal[1]

    # Widening cannot migrate a reframed observation: the width moved by -1, not by a command segment.
    problems = identity_gate_errors(old, current)
    assert f"observation_dim: parent={old.observation_dim} + 3 != current={current.observation_dim}" in problems


def test_pre_migration_walker_is_refused_by_the_behavior_preparation(pre_migration):
    from environments.shared.behavior_checkpoint import prepare_behavior_checkpoint

    species, root = pre_migration["species"], pre_migration["root"]
    env = _factory(species)()
    try:
        with pytest.raises(PlantCompatibilityError) as excinfo:
            prepare_behavior_checkpoint(
                root / "robust_best_model.zip",
                root / "robust_best_model_vecnorm.pkl",
                env,
                task_fingerprint={"schema": "test-behavior/v1", "species": species, "task_sha256": "sha256:b"},
                species=species,
            )
        _refused(excinfo, species)
    finally:
        env.close()


def test_pre_migration_statistics_beside_a_current_walker_are_refused_by_the_behavior_paths(
    species, pre_migration, tmp_path
):
    from environments.shared.behavior_checkpoint import load_behavior_checkpoint, prepare_behavior_checkpoint
    from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE

    current = current_plant_identity(species)
    # A current walker that passes every check before the statistics load: its plant and its locomotion task.
    walker_task = {"schema": "test-locomotion/v1", "species": species, "stage": "locomotion", "task_sha256": "sha256:a"}
    venv = DummyVecEnv([_factory(species)])
    try:
        walker = PPO("MlpPolicy", venv, **PPO_KWARGS)
        attach_plant_identity(walker, current)
        setattr(walker, MODEL_TASK_ATTRIBUTE, walker_task)
        walker.save(tmp_path / "walker")
    finally:
        venv.close()
    old_stats = pre_migration["root"] / "robust_best_model_vecnorm.pkl"
    env = _factory(species)()
    try:
        # Refused by plant, not by SB3's "spaces must have the same shape" as the statistics bind to the env.
        with pytest.raises(PlantCompatibilityError) as excinfo:
            prepare_behavior_checkpoint(
                tmp_path / "walker.zip",
                old_stats,
                env,
                task_fingerprint={"schema": "test-behavior/v1", "species": species, "task_sha256": "sha256:b"},
                species=species,
            )
        _refused(excinfo, species)
        with pytest.raises(PlantCompatibilityError) as excinfo:
            load_behavior_checkpoint(
                tmp_path / "walker.zip", old_stats, env, task_fingerprint=walker_task, species=species
            )
        _refused(excinfo, species)
    finally:
        env.close()
