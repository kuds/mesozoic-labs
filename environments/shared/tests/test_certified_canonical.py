"""Actual SB3 artifact verification and portable canonical library integration."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared import certified_canonical as canonical  # noqa: E402
from environments.shared.ancestors import AncestorReuseError, find_certified_ancestor, record_ancestor  # noqa: E402
from environments.shared.certified_library import resolve_recommended  # noqa: E402
from environments.shared.config import hyperparameters_sha256  # noqa: E402
from environments.shared.curriculum.gate_schema import gate_config_view  # noqa: E402
from environments.shared.plant_contract import MODEL_IDENTITY_ATTRIBUTE, PlantCompatibilityError  # noqa: E402
from environments.shared.result_bundle import write_gate_verdict  # noqa: E402
from environments.shared.result_bundle.hashing import canonical_json_sha256  # noqa: E402
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname  # noqa: E402
from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE, TASK_FINGERPRINT_SCHEMA  # noqa: E402

from .reporting_helpers import make_plant_identity  # noqa: E402


class TinyCanonicalEnv(gym.Env):
    """Real SB3 execution with a deterministic, fully measured tiny environment."""

    metadata = {}
    missing_field = None
    nonfinite_field = None
    ending = None
    step_hook = None

    def __init__(self, max_episode_steps=3):
        self.observation_space = gym.spaces.Box(-10, 10, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self.dt = 0.02
        self._contact_threshold = 0.1
        self.length = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.length = 0
        return np.zeros(4, np.float32), {}

    def step(self, action):
        self.length += 1
        info = {
            "r_foot_contact": 1.0,
            "l_foot_contact": 1.0,
            "forward_vel": 0.1,
            "tilt_angle": 0.03,
            "strike_success": self.length == 1,
        }
        if self.missing_field:
            info.pop(self.missing_field)
        if self.nonfinite_field:
            info[self.nonfinite_field] = float("nan")
        terminated = self.ending is not None and self.length == 2
        if terminated:
            info["termination_reason"] = self.ending
        hook = type(self).step_hook
        if hook:
            hook()
        return np.zeros(4, np.float32), 1000.0, terminated, self.length >= self.max_episode_steps, info


def fingerprint(*, species, stage, backend, env_kwargs, plant_identity):
    return {
        "schema": TASK_FINGERPRINT_SCHEMA,
        "species": species,
        "stage": stage,
        "backend": backend,
        "task_sha256": canonical_json_sha256({"stage": stage, "env": env_kwargs, "plant": plant_identity}),
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    plant = make_plant_identity(observation_dim=4, action_dim=2, nu=2)
    manifest = load_stage_manifest("velociraptor")
    gates = {
        entry.id: {
            "gate_kind": "task_success/v1" if entry.id == "behavior" else "stance_quality/v1",
            "gate_schema_version": 1,
            "certification_seeds": 1,
        }
        for entry in manifest.stages
    }
    env_kwargs = {"max_episode_steps": 3}
    hyperparameters = {"n_steps": 2, "batch_size": 2, "n_epochs": 1}
    monkeypatch.setattr(
        canonical,
        "get_species_config",
        lambda species: SimpleNamespace(env_class=TinyCanonicalEnv, success_keys=["strike_success"]),
    )
    monkeypatch.setattr(canonical, "derive_stage_task_fingerprint", fingerprint)
    monkeypatch.setattr(canonical, "current_plant_identity", lambda species: plant)
    monkeypatch.setattr(
        canonical,
        "load_stage_config",
        lambda species, stage: {
            "env_kwargs": env_kwargs,
            "curriculum_kwargs": gates[manifest.resolve(stage).id],
            "ppo_kwargs": hyperparameters,
        },
    )
    monkeypatch.setattr(TinyCanonicalEnv, "missing_field", None)
    monkeypatch.setattr(TinyCanonicalEnv, "nonfinite_field", None)
    monkeypatch.setattr(TinyCanonicalEnv, "ending", None)
    monkeypatch.setattr(TinyCanonicalEnv, "step_hook", None)

    def context(stage="stance", parent=None):
        entry = manifest.by_id(stage)
        task = fingerprint(
            species="velociraptor",
            stage=entry.reference,
            backend="stable-baselines3",
            env_kwargs=env_kwargs,
            plant_identity=plant.to_dict(),
        )
        parent_normalization = "sha256:" + "8" * 64 if parent else None
        if parent:
            for candidate in tmp_path.rglob("robust_best_model.zip"):
                if canonical.sha256_file(candidate) == parent:
                    parent_normalization = canonical.sha256_file(candidate.with_name("robust_best_model_vecnorm.pkl"))
                    break
        return dict(
            species="velociraptor",
            algorithm="ppo",
            entry=entry,
            current_task_sha256=task["task_sha256"],
            plant_identity=plant,
            current_gate_config=gates[stage],
            parent_model_sha256=parent,
            parent_normalization_sha256=parent_normalization,
        )

    def write_verdict(stage_dir, passed=True):
        config = json.loads((stage_dir / "stage_config.json").read_text())
        entry = manifest.resolve(config["stage"])
        write_gate_verdict(
            stage_dir,
            species="velociraptor",
            stage=entry.reference,
            stage_id=entry.id,
            gate_kind=gates[entry.id]["gate_kind"],
            gate_schema_version=1,
            passed=passed,
            failures=[] if passed else ["fixture failed"],
            task_sha256=config["task_fingerprint"]["task_sha256"],
            judged_by="test-canonical",
            checkpoint=stage_dir / "models" / "robust_best_model.zip",
            normalization=stage_dir / "models" / "robust_best_model_vecnorm.pkl",
            gate_config=gate_config_view(gates[entry.id]),
        )

    def build(name="run-a", stage="stance", seed=1, parent=None, passed=True):
        run = tmp_path / name
        entry = manifest.by_id(stage)
        directory = run / stage_dirname("velociraptor", entry.reference)
        (directory / "models").mkdir(parents=True)
        (run / "provenance.json").write_text(json.dumps({"run_id": name}))
        vector = VecNormalize(DummyVecEnv([lambda: TinyCanonicalEnv(**env_kwargs)]))
        model = PPO("MlpPolicy", vector, seed=seed, device="cpu", policy_kwargs={"net_arch": [8]}, **hyperparameters)
        task = fingerprint(
            species="velociraptor",
            stage=entry.reference,
            backend="stable-baselines3",
            env_kwargs=env_kwargs,
            plant_identity=plant.to_dict(),
        )
        setattr(model, MODEL_IDENTITY_ATTRIBUTE, plant.to_dict())
        setattr(model, MODEL_TASK_ATTRIBUTE, task)
        setattr(vector, MODEL_IDENTITY_ATTRIBUTE, plant.to_dict())
        parent_normalization = None
        if parent:
            for candidate in tmp_path.rglob("robust_best_model.zip"):
                if canonical.sha256_file(candidate) == parent:
                    parent_normalization = canonical.sha256_file(candidate.with_name("robust_best_model_vecnorm.pkl"))
                    break
        canonical.stamp_canonical_training(
            model,
            seed=seed,
            task_sha256=task["task_sha256"],
            load_mode="initialize_next_stage" if parent else None,
            parent_checkpoint_sha256=parent,
            parent_normalization_sha256=parent_normalization,
        )
        model.learn(2)
        model.save(directory / "models" / "robust_best_model")
        vector.save(directory / "models" / "robust_best_model_vecnorm.pkl")
        vector.close()
        recipe = hyperparameters_sha256({"ppo_kwargs": hyperparameters, "curriculum_kwargs": gates[stage]}, "ppo")
        run_info = {"seed": seed, "hyperparameters_sha256": recipe}
        if parent:
            run_info.update(
                load_mode="initialize_next_stage",
                parent_checkpoint_sha256=parent,
                load_path="/old/machine/deleted/parent.zip",
            )
        config = {
            "species": "velociraptor",
            "stage": entry.reference,
            "algorithm": "PPO",
            "reward_weights": env_kwargs,
            "hyperparameters": hyperparameters,
            "curriculum": gates[stage],
            "task_fingerprint": task,
            "plant_identity": plant.to_dict(),
            "run": run_info,
        }
        (directory / "stage_config.json").write_text(json.dumps(config))
        (directory / "task_fingerprint.json").write_text(json.dumps(task))
        (directory / "plant_identity.json").write_text(json.dumps(plant.to_dict()))
        (directory / "figures").mkdir()
        (directory / "figures" / "evidence.txt").write_text("Complete evidence stays with the model.")
        write_verdict(directory, passed)
        return run, directory

    def find(run, **kwargs):
        args = context(**kwargs)
        args.pop("algorithm")
        args.pop("parent_normalization_sha256")
        return find_certified_ancestor(run, **args, follow_records=True)

    return SimpleNamespace(
        plant=plant,
        manifest=manifest,
        gates=gates,
        build=build,
        context=context,
        find=find,
        write_verdict=write_verdict,
        library=tmp_path / "library",
        tmp=tmp_path,
    )


def test_publish_default_fifty_and_copy_complete_artifacts(fixture):
    f = fixture
    run, stage = f.build()
    report = canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())
    assert report["recommended"]
    selected = resolve_recommended(f.library, canonical.canonical_library_key(**f.context()))
    assert len(selected["comparison"]["protocol"]["episode_seeds"]) == 50
    assert [metric["name"] for metric in selected["comparison"]["metrics"]] == ["survival", "support", "tilt_radians"]
    assert all("reward" not in metric["name"] for metric in selected["comparison"]["metrics"])
    copied = canonical.resolve_canonical_parent(f.library, run_dir=f.tmp / "next", **f.context())
    assert copied.run_id == "run-a"
    assert copied.stage_dir != stage
    assert (copied.stage_dir / "figures" / "evidence.txt").read_bytes() == (
        stage / "figures" / "evidence.txt"
    ).read_bytes()
    shutil.rmtree(run)
    shutil.rmtree(f.library)
    model, normalizer = canonical._load_pair(copied, species="velociraptor", algorithm="ppo", plant=f.plant)
    assert model.observation_space.shape == (4,)
    normalizer.close()


def test_no_comparison_never_automatically_recommended(fixture):
    f = fixture
    run, _ = f.build()
    report = canonical.publish_canonical_stage(f.library, run_dir=run, benchmark=False, **f.context())
    assert not report["recommended"]
    with pytest.raises(LookupError):
        canonical.resolve_canonical_parent(f.library, run_dir=f.tmp / "next", **f.context())


def test_failed_outcome_stored_without_benchmark(fixture, monkeypatch):
    f = fixture
    run, _ = f.build(passed=False)
    monkeypatch.setattr(canonical, "benchmark_canonical_stage", lambda *a, **k: pytest.fail("failed gate benchmarked"))
    report = canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())
    assert not report["eligible"] and not report["recommended"]
    with pytest.raises(LookupError):
        resolve_recommended(f.library, canonical.canonical_library_key(**f.context()))


def test_independent_seed_requirement_and_stricter_current_bar(fixture):
    f = fixture
    f.gates["stance"]["certification_seeds"] = 2
    first, _ = f.build()
    report = canonical.publish_canonical_stage(f.library, run_dir=first, comparison_episodes=2, **f.context())
    assert not report["eligible"]
    second, _ = f.build("run-b", seed=2)
    report = canonical.publish_canonical_stage(f.library, run_dir=second, comparison_episodes=2, **f.context())
    assert report["eligible"] and report["distinct_seeds"] == 2
    f.gates["stance"]["certification_seeds"] = 3
    with pytest.raises(LookupError, match="provisional"):
        canonical.resolve_canonical_parent(f.library, run_dir=f.tmp / "next", **f.context())


@pytest.mark.parametrize("artifact", ["robust_best_model.zip", "robust_best_model_vecnorm.pkl"])
def test_changed_judged_pair_is_rejected(fixture, artifact):
    f = fixture
    run, stage = f.build()
    with (stage / "models" / artifact).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises((AncestorReuseError, canonical.CanonicalLibraryError)):
        canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())


def test_actual_normalizer_identity_is_checked_even_when_verdict_rehashed(fixture):
    f = fixture
    run, stage = f.build()
    path = stage / "models" / "robust_best_model_vecnorm.pkl"
    normalizer = VecNormalize.load(path, DummyVecEnv([TinyCanonicalEnv]))
    setattr(normalizer, MODEL_IDENTITY_ATTRIBUTE, replace(f.plant, physics_sha256="sha256:" + "9" * 64).to_dict())
    normalizer.save(path)
    normalizer.close()
    f.write_verdict(stage)
    with pytest.raises(PlantCompatibilityError):
        canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())


def test_recorded_environment_must_reproduce_task(fixture):
    f = fixture
    run, stage = f.build()
    path = stage / "stage_config.json"
    config = json.loads(path.read_text())
    config["reward_weights"]["max_episode_steps"] = 4
    path.write_text(json.dumps(config))
    with pytest.raises(canonical.CanonicalLibraryError, match="environment settings"):
        canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())


def test_recipe_digest_recomputed(fixture):
    f = fixture
    run, stage = f.build()
    path = stage / "stage_config.json"
    config = json.loads(path.read_text())
    config["hyperparameters"]["n_steps"] = 10
    path.write_text(json.dumps(config))
    with pytest.raises(canonical.CanonicalLibraryError, match="recipe digest"):
        canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())


def build_linked_child(f):
    root, _ = f.build()
    ancestor = f.find(root)
    child, stage = f.build("child", "locomotion", parent=ancestor.model_sha256)
    record_ancestor(child, ancestor)
    return root, ancestor, child, stage


def test_linked_parent_complete_snapshot_and_locomotion_lookup(fixture):
    f = fixture
    root, ancestor, child, _ = build_linked_child(f)
    canonical.publish_canonical_stage(f.library, run_dir=root, comparison_episodes=2, **f.context())
    canonical.publish_canonical_stage(
        f.library, run_dir=child, comparison_episodes=2, **f.context("locomotion", ancestor.model_sha256)
    )
    copied = canonical.resolve_canonical_locomotion(f.library, run_dir=f.tmp / "next", species="velociraptor")
    assert copied.stage_id == "locomotion" and copied.run_id == "child"
    shutil.rmtree(root)
    shutil.rmtree(child)
    model, normalizer = canonical._load_pair(copied, species="velociraptor", algorithm="ppo", plant=f.plant)
    normalizer.close()
    parent = f.find(copied.source_run_dir)
    assert parent.model_sha256 == ancestor.model_sha256


def test_different_parent_rejected(fixture):
    f = fixture
    _, _, child, _ = build_linked_child(f)
    with pytest.raises(canonical.CanonicalLibraryError, match="exact selected parent"):
        canonical.publish_canonical_stage(f.library, run_dir=child, **f.context("locomotion", "sha256:" + "9" * 64))


def test_manual_copy_preserves_lineage_and_all_files(fixture):
    f = fixture
    _, ancestor, child, stage = build_linked_child(f)
    selected = f.find(child, stage="locomotion", parent=ancestor.model_sha256)
    copied = canonical.copy_canonical_ancestor(selected, f.tmp / "next")
    assert copied.run_id == "child"
    assert copied.model_sha256 == selected.model_sha256
    assert (copied.stage_dir / "figures" / "evidence.txt").read_bytes() == (
        stage / "figures" / "evidence.txt"
    ).read_bytes()
    assert f.find(copied.source_run_dir).model_sha256 == ancestor.model_sha256
    (copied.stage_dir / "figures" / "evidence.txt").write_text("tampered")
    with pytest.raises(canonical.CanonicalLibraryError, match="artifact changed"):
        canonical.copy_canonical_ancestor(selected, f.tmp / "next")


def test_changed_comparison_count_rebenchmarks_incumbent(fixture, monkeypatch):
    f = fixture
    first, _ = f.build()
    canonical.publish_canonical_stage(f.library, run_dir=first, comparison_episodes=2, **f.context())
    second, _ = f.build("run-b", seed=2)
    actual = canonical.benchmark_canonical_stage
    calls = []

    def tracked(ancestor, **kwargs):
        calls.append((ancestor.run_id, kwargs["comparison_episodes"]))
        return actual(ancestor, **kwargs)

    monkeypatch.setattr(canonical, "benchmark_canonical_stage", tracked)
    canonical.publish_canonical_stage(f.library, run_dir=second, comparison_episodes=3, **f.context())
    assert calls == [("run-b", 3), ("run-a", 3)]
    selected = resolve_recommended(f.library, canonical.canonical_library_key(**f.context()))
    assert len(selected["comparison"]["protocol"]["episode_seeds"]) == 3


@pytest.mark.parametrize("field", ["r_foot_contact", "l_foot_contact", "tilt_angle"])
@pytest.mark.parametrize("defect", ["missing_field", "nonfinite_field"])
def test_comparison_requires_real_finite_metrics(fixture, monkeypatch, field, defect):
    f = fixture
    run, _ = f.build()
    monkeypatch.setattr(TinyCanonicalEnv, defect, field)
    with pytest.raises(canonical.CanonicalLibraryError, match="finite measured"):
        canonical.benchmark_canonical_stage(
            f.find(run), species="velociraptor", algorithm="ppo", plant_identity=f.plant, comparison_episodes=2
        )


@pytest.mark.parametrize("reason, safe", [("fallen", 0.0), ("strike_success", 1.0)])
def test_past_success_does_not_hide_fall(fixture, monkeypatch, reason, safe):
    f = fixture
    run, _ = f.build()
    ancestor = replace(f.find(run), verdict={"gate_kind": "task_success/v1"})
    monkeypatch.setattr(TinyCanonicalEnv, "ending", reason)
    result = canonical.benchmark_canonical_stage(
        ancestor, species="velociraptor", algorithm="ppo", plant_identity=f.plant, comparison_episodes=2
    )
    assert all(episode["survival"] == 0 for episode in result["episodes"])
    assert all(episode["safe_completion"] == safe for episode in result["episodes"])


def test_checkpoint_change_during_scoring_refused(fixture, monkeypatch):
    f = fixture
    run, stage = f.build()
    ancestor = f.find(run)
    path = stage / "models" / "robust_best_model.zip"

    def change():
        with path.open("ab") as stream:
            stream.write(b"changed during scoring")

    monkeypatch.setattr(TinyCanonicalEnv, "step_hook", change)
    with pytest.raises(canonical.CanonicalLibraryError, match="changed during scoring"):
        canonical.benchmark_canonical_stage(
            ancestor, species="velociraptor", algorithm="ppo", plant_identity=f.plant, comparison_episodes=2
        )


def test_checkpoint_embedded_task_refused_even_with_new_gate_hash(fixture):
    f = fixture
    run, stage = f.build()
    path = stage / "models" / "robust_best_model.zip"
    model = PPO.load(path, device="cpu")
    setattr(model, MODEL_TASK_ATTRIBUTE, {"task_sha256": "sha256:" + "9" * 64})
    model.save(path)
    f.write_verdict(stage)
    with pytest.raises(canonical.CanonicalLibraryError, match="task fingerprints disagree"):
        canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())


def test_source_change_during_snapshot_refused(fixture, monkeypatch):
    f = fixture
    run, _ = f.build()
    actual = canonical._snapshot

    def changed_snapshot(chain, directory, species):
        metadata = actual(chain, directory, species)
        with (directory / metadata["normalization_path"]).open("ab") as stream:
            stream.write(b"concurrent artifact change")
        return metadata

    monkeypatch.setattr(canonical, "_snapshot", changed_snapshot)
    with pytest.raises(AncestorReuseError, match="normalization"):
        canonical.publish_canonical_stage(f.library, run_dir=run, comparison_episodes=2, **f.context())
    with pytest.raises(LookupError):
        resolve_recommended(f.library, canonical.canonical_library_key(**f.context()))


def test_existing_bundle_audit_failure_refuses_publication(fixture, monkeypatch):
    from environments.shared.result_bundle import audit

    f = fixture
    run, _ = f.build()
    (run / "artifact_manifest.json").write_text("{}")
    monkeypatch.setattr(audit, "audit_result_bundle", lambda *a, **k: {"errors": ["missing judged evidence"]})
    with pytest.raises(canonical.CanonicalLibraryError, match="missing judged evidence"):
        canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())


def test_locomotion_requires_measured_velocity(fixture, monkeypatch):
    f = fixture
    _, parent, child, _ = build_linked_child(f)
    ancestor = f.find(child, stage="locomotion", parent=parent.model_sha256)
    monkeypatch.setattr(TinyCanonicalEnv, "missing_field", "forward_vel")
    with pytest.raises(canonical.CanonicalLibraryError, match="forward_vel"):
        canonical.benchmark_canonical_stage(
            ancestor, species="velociraptor", algorithm="ppo", plant_identity=f.plant, comparison_episodes=2
        )


def test_zero_optimizer_updates_cannot_publish(fixture):
    f = fixture
    run, stage = f.build()
    path = stage / "models" / "robust_best_model.zip"
    model = PPO.load(path, device="cpu")
    # A rollout can advance the step counter without completing any update.
    canonical.stamp_canonical_training(model, seed=5, task_sha256=f.context()["current_task_sha256"], load_mode=None)
    model.num_timesteps += 100
    model.save(path)
    f.write_verdict(stage)
    with pytest.raises(canonical.CanonicalLibraryError, match="optimizer updates"):
        canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())


def test_same_stage_resume_keeps_origin_seed_and_does_not_add_replica(fixture):
    f = fixture
    f.gates["stance"]["certification_seeds"] = 2
    run, stage = f.build()
    canonical.publish_canonical_stage(f.library, run_dir=run, comparison_episodes=2, **f.context())
    resumed = f.tmp / "resumed"
    shutil.copytree(run, resumed)
    resumed_stage = resumed / stage.name
    path = resumed_stage / "models" / "robust_best_model.zip"
    vector = VecNormalize.load(
        resumed_stage / "models" / "robust_best_model_vecnorm.pkl", DummyVecEnv([TinyCanonicalEnv])
    )
    model = PPO.load(path, env=vector, device="cpu")
    marker = canonical.stamp_canonical_training(
        model, seed=999, task_sha256=f.context()["current_task_sha256"], load_mode="resume_same_stage"
    )
    assert marker["seed"] == 1
    model.learn(2, reset_num_timesteps=False)
    model.save(path)
    vector.save(resumed_stage / "models" / "robust_best_model_vecnorm.pkl")
    vector.close()
    config_path = resumed_stage / "stage_config.json"
    config = json.loads(config_path.read_text())
    config["run"]["seed"] = 999
    config_path.write_text(json.dumps(config))
    (resumed / "provenance.json").write_text(json.dumps({"run_id": "resumed"}))
    f.write_verdict(resumed_stage)
    report = canonical.publish_canonical_stage(f.library, run_dir=resumed, comparison_episodes=2, **f.context())
    assert report["distinct_seeds"] == 1
    assert not report["eligible"]


def test_legacy_origin_can_manual_copy_but_cannot_auto_publish(fixture):
    f = fixture
    run, stage = f.build()
    path = stage / "models" / "robust_best_model.zip"
    model = PPO.load(path, device="cpu")
    delattr(model, canonical._TRAINING_ATTRIBUTE)
    assert (
        canonical.stamp_canonical_training(
            model, seed=999, task_sha256=f.context()["current_task_sha256"], load_mode="resume_same_stage"
        )
        is None
    )
    model.save(path)
    f.write_verdict(stage)
    copied = canonical.copy_canonical_ancestor(f.find(run), f.tmp / "next")
    assert copied.model_zip.is_file()
    with pytest.raises(canonical.CanonicalLibraryError, match="training origin"):
        canonical.publish_canonical_stage(f.library, run_dir=run, **f.context())


def test_training_origin_binds_exact_parent_normalization(fixture):
    f = fixture
    _, ancestor, child, stage = build_linked_child(f)
    path = stage / "models" / "robust_best_model.zip"
    model = PPO.load(path, device="cpu")
    getattr(model, canonical._TRAINING_ATTRIBUTE)["parent_normalization_sha256"] = "sha256:" + "9" * 64
    model.save(path)
    f.write_verdict(stage)
    with pytest.raises(canonical.CanonicalLibraryError, match="exact task and parent pair"):
        canonical.publish_canonical_stage(f.library, run_dir=child, **f.context("locomotion", ancestor.model_sha256))


def test_copying_a_snapshot_preserves_each_original_provenance(fixture):
    f = fixture
    root, ancestor, child, _ = build_linked_child(f)
    (root / "provenance.json").write_text(json.dumps({"run_id": "run-a", "seed": 11, "hardware": "gpu-a"}))
    (child / "provenance.json").write_text(json.dumps({"run_id": "child", "seed": 22, "hardware": "gpu-b"}))
    selected = f.find(child, stage="locomotion", parent=ancestor.model_sha256)
    first = canonical.copy_canonical_ancestor(selected, f.tmp / "first-copy")
    second = canonical.copy_canonical_ancestor(first, f.tmp / "second-copy")
    records = second.source_run_dir.parent / "source_records"
    assert json.loads((records / "stance" / "provenance.json").read_text()) == {
        "run_id": "run-a",
        "seed": 11,
        "hardware": "gpu-a",
    }
    assert json.loads((records / "locomotion" / "provenance.json").read_text()) == {
        "run_id": "child",
        "seed": 22,
        "hardware": "gpu-b",
    }
    provenance = json.loads((second.source_run_dir / "provenance.json").read_text())
    assert provenance["original_sources"]["stance"]["run_id"] == "run-a"
    assert provenance["original_sources"]["locomotion"]["run_id"] == "child"


def test_parent_normalization_is_part_of_exact_lookup_key(fixture):
    f = fixture
    _, parent, child, _ = build_linked_child(f)
    kwargs = f.context("locomotion", parent.model_sha256)
    canonical.publish_canonical_stage(f.library, run_dir=child, comparison_episodes=2, **kwargs)
    different = {**kwargs, "parent_normalization_sha256": "sha256:" + "9" * 64}
    with pytest.raises(LookupError):
        canonical.resolve_canonical_parent(f.library, run_dir=f.tmp / "next", **different)
    with pytest.raises(canonical.CanonicalLibraryError, match="exact selected parent pair"):
        canonical.publish_canonical_stage(f.library, run_dir=child, comparison_episodes=2, **different)


def test_benchmark_seed_panel_is_disjoint_from_recorded_certification(fixture):
    f = fixture
    run, _ = f.build()
    (run / "provenance.json").write_text(json.dumps({"run_id": "run-a", "evaluation_seeds": [1_700_000]}))
    with pytest.raises(canonical.CanonicalLibraryError, match="overlap recorded"):
        canonical.benchmark_canonical_stage(
            f.find(run), species="velociraptor", algorithm="ppo", plant_identity=f.plant, comparison_episodes=2
        )
