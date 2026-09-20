"""Publish and copy verified canonical stage policies through the shared library.

The library is an additional immutable copy. Original runs and gate evidence
remain unchanged. A selected input includes its complete verified parent chain,
so loading it never depends on the original machine's checkpoint paths.
"""

from __future__ import annotations

import inspect
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from environments.shared.ancestors import CertifiedAncestor, find_certified_ancestor, run_id_for
from environments.shared.config import hyperparameters_sha256, load_stage_config
from environments.shared.curriculum.gate_schema import (
    declared_certification_seeds,
    gate_config_sha256,
    gate_config_view,
)
from environments.shared.plant_contract import (
    PlantIdentity,
    current_plant_identity,
    validate_model_plant,
    validate_recorded_identity,
)
from environments.shared.result_bundle import read_gate_verdict, sha256_file
from environments.shared.result_bundle.hashing import canonical_json_sha256
from environments.shared.species_names import resolve_species_id
from environments.shared.species_registry import get_species_config
from environments.shared.stage_manifest import StageEntry, load_stage_manifest, stage_dir_candidates, stage_dirname
from environments.shared.task_fingerprint import (
    MODEL_TASK_ATTRIBUTE,
    derive_stage_task_fingerprint,
    read_checkpoint_attribute,
)

_BENCHMARK_SEED_START = 1_700_000
_TRAINING_ATTRIBUTE = "mesozoic_canonical_training"
_ROOT_RECORDS = (
    "provenance.json",
    "summary.json",
    "artifact_manifest.json",
    "collected_results.csv",
    "plant_identity.json",
)


class CanonicalLibraryError(RuntimeError):
    """A claimed canonical policy or its evidence cannot be verified."""


def stamp_canonical_training(
    model: Any,
    *,
    seed: int,
    task_sha256: str,
    load_mode: str | None,
    parent_checkpoint_sha256: str | None = None,
    parent_normalization_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Bind independent stage origin before learning; resume never mints a seed.

    Older same-stage checkpoints remain trainable but cannot automatically
    become independently certified replicas without recorded origin evidence.
    The optimizer-update anchor works even when SB3 resets its step counter.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CanonicalLibraryError("Canonical training seed must be a nonnegative integer")
    if load_mode == "resume_same_stage":
        marker = getattr(model, _TRAINING_ATTRIBUTE, None)
        if marker is None:
            return None
        if not isinstance(marker, dict) or marker.get("task_sha256") != task_sha256:
            raise CanonicalLibraryError("Resumed canonical training origin describes another task")
        return dict(marker)
    if load_mode not in {None, "initialize_next_stage"}:
        raise CanonicalLibraryError("Unknown canonical training load mode")
    if load_mode == "initialize_next_stage" and not (parent_checkpoint_sha256 and parent_normalization_sha256):
        raise CanonicalLibraryError("A new canonical stage must identify both selected parent artifacts")
    if load_mode is None and (parent_checkpoint_sha256 or parent_normalization_sha256):
        raise CanonicalLibraryError("Fresh canonical training cannot declare parent artifacts")
    marker = {
        "version": "mesozoic.canonical-training/v1",
        "seed": seed,
        "task_sha256": task_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_normalization_sha256": parent_normalization_sha256,
        "starting_updates": int(model._n_updates),
    }
    model.seed = seed
    model.set_random_seed(seed)
    setattr(model, _TRAINING_ATTRIBUTE, marker)
    return dict(marker)


def _training_origin(ancestor: CertifiedAncestor, parent: CertifiedAncestor | None) -> dict[str, Any]:
    marker = read_checkpoint_attribute(ancestor.model_zip, _TRAINING_ATTRIBUTE)
    updates = read_checkpoint_attribute(ancestor.model_zip, "_n_updates")
    steps = read_checkpoint_attribute(ancestor.model_zip, "num_timesteps")
    if not isinstance(marker, dict) or marker.get("version") != "mesozoic.canonical-training/v1":
        raise CanonicalLibraryError("Automatic publication requires recorded canonical training origin")
    seed, anchor = marker.get("seed"), marker.get("starting_updates")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(anchor, bool)
        or not isinstance(anchor, int)
        or anchor < 0
        or not isinstance(updates, int)
        or updates <= anchor
        or not isinstance(steps, int)
        or steps <= 0
    ):
        raise CanonicalLibraryError("Canonical candidate has no verified optimizer updates since stage entry")
    if (
        marker.get("task_sha256") != ancestor.task_sha256
        or marker.get("parent_checkpoint_sha256") != (parent.model_sha256 if parent else None)
        or marker.get("parent_normalization_sha256") != (parent.normalization_sha256 if parent else None)
    ):
        raise CanonicalLibraryError("Canonical training origin does not match the exact task and parent pair")
    return marker


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CanonicalLibraryError(f"Expected a JSON object: {path}")
    return value


def _algorithm(value: str) -> str:
    algorithm = value.lower()
    if algorithm not in {"ppo", "sac"}:
        raise CanonicalLibraryError("Canonical policy library supports SB3 PPO and SAC")
    return algorithm


def canonical_library_key(
    *,
    species: str,
    algorithm: str,
    entry: StageEntry,
    current_task_sha256: str,
    plant_identity: PlantIdentity,
    current_gate_config: Mapping[str, Any],
    parent_model_sha256: str | None = None,
    parent_normalization_sha256: str | None = None,
) -> dict[str, Any]:
    """Exact compatibility key; algorithm recipes remain separate seed groups."""
    species = resolve_species_id(species)
    if plant_identity.species != species:
        raise CanonicalLibraryError("Plant identity belongs to another species")
    if load_stage_manifest(species).by_id(entry.id) != entry:
        raise CanonicalLibraryError("Stage does not match the selected species manifest")
    if bool(entry.warm_start_from) != bool(parent_model_sha256) or bool(entry.warm_start_from) != bool(
        parent_normalization_sha256
    ):
        raise CanonicalLibraryError("A non-root stage requires both selected parent artifact hashes")
    return {
        "species": species,
        "algorithm": _algorithm(algorithm),
        "behavior": entry.id,
        "backend": "stable-baselines3",
        "task_sha256": current_task_sha256,
        "gate_sha256": gate_config_sha256(gate_config_view(current_gate_config)),
        "plant_sha256": canonical_json_sha256(plant_identity.to_dict()),
        "parent_model_sha256": parent_model_sha256,
        "parent_normalization_sha256": parent_normalization_sha256,
    }


def _current_context(species: str, entry: StageEntry, plant: PlantIdentity) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_stage_config(species, entry.reference)
    fingerprint = derive_stage_task_fingerprint(
        species=species,
        stage=entry.reference,
        backend="stable-baselines3",
        env_kwargs=config["env_kwargs"],
        plant_identity=plant.to_dict(),
    )
    return config, fingerprint


def _locate_stage(run_dir: Path, species: str, entry: StageEntry, seen: tuple[Path, ...] = ()) -> tuple[Path, Path]:
    root = run_dir.resolve()
    if root in seen or len(seen) >= 8:
        raise CanonicalLibraryError("Canonical ancestor records contain a cycle or too many hops")
    for name in stage_dir_candidates(species, entry.reference):
        stage = root / name
        if stage.is_dir():
            return root, stage
    record_path = root / "ancestors" / entry.id / "ancestor.json"
    if not record_path.is_file():
        raise CanonicalLibraryError(f"Missing stage or ancestor evidence for {entry.id}: {root}")
    record = _read(record_path)
    source = Path(record["source_run_dir"])
    if not source.is_dir():
        source = root.parent / source.name
    if not source.is_dir():
        raise CanonicalLibraryError(f"Cannot locate original certified ancestor: {record_path}")
    return _locate_stage(source, species, entry, (*seen, root))


def _load_pair(ancestor: CertifiedAncestor, *, species: str, algorithm: str, plant: PlantIdentity) -> tuple[Any, Any]:
    """Load both actual artifacts, with safe constant evaluation schedules."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.policy_loading import _checkpoint_algorithm, load_sb3_model

    config = _read(ancestor.stage_dir / "stage_config.json")
    if config.get("species") != species:
        raise CanonicalLibraryError("Recorded stage belongs to another species")
    validate_recorded_identity(config.get("plant_identity"), plant, artifact="recorded stage", allow_legacy=False)
    if str(config.get("algorithm", "")).lower() != algorithm:
        raise CanonicalLibraryError("Recorded stage algorithm differs from the selected algorithm")
    cls = _checkpoint_algorithm(str(ancestor.model_zip))
    if cls.__name__.lower() != algorithm:
        raise CanonicalLibraryError("Checkpoint optimizer metadata differs from the recorded algorithm")
    fingerprint = config.get("task_fingerprint")
    model_fingerprint = read_checkpoint_attribute(ancestor.model_zip, MODEL_TASK_ATTRIBUTE)
    if not isinstance(fingerprint, dict) or not isinstance(model_fingerprint, dict):
        raise CanonicalLibraryError("Both stage and checkpoint must record the canonical task")
    if (
        model_fingerprint.get("task_sha256") != ancestor.task_sha256
        or fingerprint.get("task_sha256") != ancestor.task_sha256
    ):
        raise CanonicalLibraryError("Checkpoint, stage and verdict task fingerprints disagree")
    env_class = get_species_config(species).env_class
    env_kwargs = dict(config.get("reward_weights", {}))
    env_kwargs.pop("render_mode", None)
    derived = derive_stage_task_fingerprint(
        species=species,
        stage=load_stage_manifest(species).by_id(ancestor.stage_id).reference,
        backend="stable-baselines3",
        env_kwargs=env_kwargs,
        plant_identity=plant.to_dict(),
    )
    if derived["task_sha256"] != ancestor.task_sha256:
        raise CanonicalLibraryError("Recorded environment settings no longer produce the certified task fingerprint")
    vector = DummyVecEnv([lambda: env_class(**env_kwargs)])
    normalizer = None
    try:
        normalizer = VecNormalize.load(str(ancestor.normalization_path), vector)
        normalizer.training = False
        normalizer.norm_reward = False
        validate_model_plant(normalizer, plant, artifact="certified normalization", allow_legacy=False)
        # Serialized Python schedules from another runtime can be unsafe to
        # execute; inference never needs the original training schedules.
        custom = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0}
        if algorithm == "ppo":
            custom.update(clip_range=lambda _: 0.2, clip_range_vf=None)
        model = load_sb3_model(
            str(ancestor.model_zip), algorithm=cls, env=normalizer, device="cpu", custom_objects=custom
        )
        validate_model_plant(model, plant, artifact="certified checkpoint", allow_legacy=False)
        if normalizer.observation_space.shape != (plant.observation_dim,) or model.action_space.shape != (
            plant.action_dim,
        ):
            raise CanonicalLibraryError("Loaded policy/normalizer dimensions differ from the current plant")
        return model, normalizer
    except BaseException:
        (normalizer if normalizer is not None else vector).close()
        raise


def _failed_handoff(
    run_dir: Path,
    *,
    species: str,
    entry: StageEntry,
    current_task_sha256: str,
    current_gate_config: Mapping[str, Any],
    parent_model_sha256: str | None,
    parent_normalization_sha256: str | None = None,
) -> CertifiedAncestor:
    """Validate a failed artifact pair for the outcome ledger, never for reuse.

    The existing handoff record is used as a container. Its verdict remains
    false, and no resolver or manual-copy API returns this failed candidate.
    """
    from environments.shared.ancestors import _check_chain
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint

    root, stage = _locate_stage(run_dir, species, entry)
    verdict = read_gate_verdict(stage)
    if verdict is None or verdict.get("passed") is not False:
        raise CanonicalLibraryError("Failed publication requires an explicit failed gate verdict")
    if verdict.get("species") != species or verdict.get("stage_id") != entry.id:
        raise CanonicalLibraryError("Failed verdict belongs to another species or stage")
    record = _read(stage / "stage_config.json")
    task = record.get("task_fingerprint", {})
    if task.get("task_sha256") != current_task_sha256 or verdict.get("task_sha256") != current_task_sha256:
        raise CanonicalLibraryError("Failed stage and verdict do not describe the current task")
    expected_gate = gate_config_sha256(gate_config_view(current_gate_config))
    if verdict.get("gate_sha256") != expected_gate:
        raise CanonicalLibraryError("Failed verdict used another gate")
    _check_chain(stage, entry=entry, parent_model_sha256=parent_model_sha256)
    selected = select_handoff_checkpoint(stage / "models")
    if selected is None:
        raise CanonicalLibraryError("Failed stage has no complete judged checkpoint/normalization pair")
    name, stem, normalization = selected
    model, normalizer = Path(stem + ".zip"), Path(normalization)
    if sha256_file(model) != verdict.get("checkpoint_sha256") or sha256_file(normalizer) != verdict.get(
        "normalization_sha256"
    ):
        raise CanonicalLibraryError("Failed handoff artifacts differ from the pair judged by its gate")
    return CertifiedAncestor(
        stage_id=entry.id,
        stage_key=entry.key,
        run_id=run_id_for(root),
        source_run_dir=root,
        stage_dir=stage,
        handoff_name=name,
        model_stem=stem,
        model_zip=model,
        model_sha256=verdict["checkpoint_sha256"],
        normalization_path=normalizer,
        normalization_sha256=verdict["normalization_sha256"],
        task_sha256=current_task_sha256,
        judged_by=verdict["judged_by"],
        verdict=verdict,
        gate_sha256=expected_gate,
    )


def _verified_chain(
    run_dir: Path,
    *,
    species: str,
    algorithm: str,
    entry: StageEntry,
    current_task_sha256: str,
    plant_identity: PlantIdentity,
    current_gate_config: Mapping[str, Any],
    parent_model_sha256: str | None,
    parent_normalization_sha256: str | None = None,
    allow_failed_target: bool = False,
) -> list[CertifiedAncestor]:
    manifest = load_stage_manifest(species)
    source_root, _ = _locate_stage(run_dir, species, entry)
    chain: list[CertifiedAncestor] = []
    if entry.warm_start_from is not None:
        parent = manifest.by_id(entry.warm_start_from)
        parent_config, parent_task = _current_context(species, parent, plant_identity)
        _, parent_stage = _locate_stage(source_root, species, parent)
        parent_record = _read(parent_stage / "stage_config.json")
        grandparent_hash = (
            parent_record.get("run", {}).get("parent_checkpoint_sha256") if parent.warm_start_from else None
        )
        chain = _verified_chain(
            source_root,
            species=species,
            algorithm=algorithm,
            entry=parent,
            current_task_sha256=parent_task["task_sha256"],
            plant_identity=plant_identity,
            current_gate_config=parent_config["curriculum_kwargs"],
            parent_model_sha256=grandparent_hash,
        )
        if chain[-1].model_sha256 != parent_model_sha256 or (
            parent_normalization_sha256 is not None and chain[-1].normalization_sha256 != parent_normalization_sha256
        ):
            raise CanonicalLibraryError("Candidate does not descend from the exact selected parent pair")
    _, source_stage = _locate_stage(run_dir, species, entry)
    verdict = read_gate_verdict(source_stage)
    if allow_failed_target and verdict is not None and verdict.get("passed") is False:
        ancestor = _failed_handoff(
            run_dir,
            species=species,
            entry=entry,
            current_task_sha256=current_task_sha256,
            current_gate_config=current_gate_config,
            parent_model_sha256=parent_model_sha256,
            parent_normalization_sha256=parent_normalization_sha256,
        )
    else:
        ancestor = find_certified_ancestor(
            run_dir,
            species=species,
            entry=entry,
            current_task_sha256=current_task_sha256,
            plant_identity=plant_identity,
            current_gate_config=current_gate_config,
            parent_model_sha256=parent_model_sha256,
            follow_records=True,
        )
    if (ancestor.source_run_dir / "artifact_manifest.json").is_file():
        from environments.shared.result_bundle.audit import audit_result_bundle

        audit = audit_result_bundle(ancestor.source_run_dir, reject_unlisted=False)
        if audit["errors"]:
            raise CanonicalLibraryError("Source bundle audit failed: " + "; ".join(audit["errors"]))
    model, normalizer = _load_pair(ancestor, species=species, algorithm=algorithm, plant=plant_identity)
    del model
    normalizer.close()
    provenance_path = ancestor.source_run_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = _read(provenance_path)
        if provenance.get("canonical_library_copy") is True:
            original = provenance.get("original_sources", {}).get(entry.id, {})
            if (
                original.get("model_sha256") != ancestor.model_sha256
                or original.get("normalization_sha256") != ancestor.normalization_sha256
                or not isinstance(original.get("run_id"), str)
            ):
                raise CanonicalLibraryError("Copied ancestry provenance does not match its artifact pair")
            ancestor = replace(ancestor, run_id=original["run_id"])
    chain.append(ancestor)
    return chain


def benchmark_canonical_stage(
    ancestor: CertifiedAncestor,
    *,
    species: str,
    algorithm: str,
    plant_identity: PlantIdentity,
    comparison_episodes: int = 50,
) -> dict[str, Any]:
    """A separate paired-seed benchmark, never a substitute for the stage gate."""
    if isinstance(comparison_episodes, bool) or not isinstance(comparison_episodes, int) or comparison_episodes < 2:
        raise CanonicalLibraryError("comparison_episodes must be an integer of at least 2")
    config = _read(ancestor.stage_dir / "stage_config.json")
    seeds = list(range(_BENCHMARK_SEED_START, _BENCHMARK_SEED_START + comparison_episodes))
    from environments.shared.constants import PUBLICATION_SEED_START

    origin = read_checkpoint_attribute(ancestor.model_zip, _TRAINING_ATTRIBUTE)
    training_seeds = [config.get("run", {}).get("seed")]
    if isinstance(origin, dict):
        training_seeds.append(origin.get("seed"))
    used = {PUBLICATION_SEED_START}
    parallel_envs = int(config.get("run", {}).get("n_envs", 1))
    for training_seed in training_seeds:
        if isinstance(training_seed, int):
            used.update(range(training_seed, training_seed + parallel_envs))
            used.add(training_seed + 1000)  # canonical in-training evaluation
    provenance_path = ancestor.source_run_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = _read(provenance_path)
        if provenance.get("canonical_library_copy") is True:
            provenance_path = ancestor.source_run_dir.parent / "source_records" / ancestor.stage_id / "provenance.json"
            provenance = _read(provenance_path) if provenance_path.is_file() else {}
        used.update(value for value in provenance.get("evaluation_seeds", []) if isinstance(value, int))
        used.update(value for value in provenance.get("seed_roles", {}).values() if isinstance(value, int))
    if used.intersection(seeds):
        raise CanonicalLibraryError("Comparison seeds overlap recorded training or certification evaluation seeds")
    expected = (ancestor.model_sha256, ancestor.normalization_sha256)

    def check_artifacts() -> None:
        if (sha256_file(ancestor.model_zip), sha256_file(ancestor.normalization_path)) != expected:
            raise CanonicalLibraryError("Comparison checkpoint or normalization changed during scoring")

    check_artifacts()
    model, normalizer = _load_pair(ancestor, species=species, algorithm=algorithm, plant=plant_identity)
    from environments.shared.base_env import BaseDinoEnv

    raw = normalizer.venv.envs[0].unwrapped
    sources = {Path(__file__), Path(inspect.getfile(BaseDinoEnv)), Path(inspect.getfile(type(raw)))}
    source_hashes = {path.name: sha256_file(path) for path in sorted(sources)}
    horizon, dt = int(raw.max_episode_steps), float(raw.dt)
    episodes = []
    success_keys = get_species_config(species).success_keys
    task_success = ancestor.stage_id == "behavior" or ancestor.verdict.get("gate_kind") == "task_success/v1"
    successful_reasons = set(success_keys) | {"target_reached"}
    contact_keys = ["r_foot_contact", "l_foot_contact"]
    if species in {"brachiosaurus", "dibothrosuchus"}:
        contact_keys += ["rr_foot_contact", "rl_foot_contact"]
    threshold = float(raw._contact_threshold)

    def measured(info: Mapping[str, Any], name: str) -> float:
        value = info.get(name)
        if not isinstance(value, (int, float, np.number)) or not np.isfinite(float(value)):
            raise CanonicalLibraryError(f"Comparison requires finite measured {name}")
        return float(value)

    try:
        check_artifacts()
        for seed in seeds:
            normalizer.seed(seed)
            observation = normalizer.reset()
            supported = 0.0
            progress = 0.0
            tilts = []
            success = False
            length = 0
            terminated = False
            reason = None
            for length in range(1, horizon + 1):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, dones, infos = normalizer.step(action)
                info = infos[0]
                if not task_success:
                    forces = [measured(info, name) for name in contact_keys]
                    supported += float(sum(forces) > threshold)
                    if ancestor.stage_id == "locomotion":
                        progress += measured(info, "forward_vel") * dt
                    else:
                        tilts.append(measured(info, "tilt_angle"))
                if task_success:
                    available = [key for key in ("is_success", *success_keys) if key in info]
                    if not available:
                        raise CanonicalLibraryError("Comparison requires measured task success")
                    outcomes = [measured(info, key) for key in available]
                    if any(value not in {0.0, 1.0} for value in outcomes):
                        raise CanonicalLibraryError("Task success must be a boolean or zero/one")
                    success |= any(outcomes)
                if dones[0]:
                    terminated = not bool(info.get("TimeLimit.truncated", False))
                    reason = info.get("termination_reason")
                    break
            survived = length == horizon and not terminated
            # A successful task termination is distinct from full-horizon
            # survival. A past target touch must never erase a later fall.
            completed_safely = task_success and success and terminated and reason in successful_reasons
            episodes.append(
                {
                    "seed": seed,
                    "survival": float(survived),
                    "safe_completion": float(survived or completed_safely),
                    "support": supported / horizon if not task_success else None,
                    "progress_m": progress if ancestor.stage_id == "locomotion" else None,
                    "tilt_radians": float(np.mean(tilts)) if tilts else None,
                    "success": float(success),
                    "completion_time_s": length * dt if completed_safely else horizon * dt,
                    "termination_reason": reason,
                }
            )
        check_artifacts()
        if {path.name: sha256_file(path) for path in sorted(sources)} != source_hashes:
            raise CanonicalLibraryError("Comparison implementation changed during scoring")
    finally:
        normalizer.close()
    # Ordering gives each stage a primary objective. Every other metric is a
    # non-regression guard, so greater speed cannot compensate for more falls.
    if task_success:
        metrics = [("success", "higher", 0.0), ("safe_completion", "higher", 0.0), ("completion_time_s", "lower", 0.05)]
    elif ancestor.stage_id == "locomotion":
        metrics = [("survival", "higher", 0.0), ("progress_m", "higher", 0.01), ("support", "higher", 0.01)]
    else:
        metrics = [("survival", "higher", 0.0), ("support", "higher", 0.01), ("tilt_radians", "lower", 0.01)]
    from importlib.metadata import version

    return {
        "protocol": {
            "version": "mesozoic.canonical-library-comparison/v1",
            "episode_seeds": seeds,
            "species": species,
            "algorithm": algorithm,
            "stage": ancestor.stage_id,
            "task_sha256": ancestor.task_sha256,
            "plant_sha256": canonical_json_sha256(plant_identity.to_dict()),
            "dt": dt,
            "horizon_steps": horizon,
            "deterministic": True,
            "normalization_statistics_frozen": True,
            "reward_normalization": False,
            "source_sha256": source_hashes,
            "runtime_versions": {package: version(package) for package in ("stable-baselines3", "mujoco", "numpy")},
        },
        "metrics": [
            {"name": name, "values": [episode[name] for episode in episodes], "direction": direction, "margin": margin}
            for name, direction, margin in metrics
        ],
        "episodes": episodes,
        "model_sha256": ancestor.model_sha256,
        "normalization_sha256": ancestor.normalization_sha256,
    }


def _copy_tree(source: Path, destination: Path) -> None:
    for item in source.rglob("*"):
        if item.is_symlink():
            raise CanonicalLibraryError(f"Artifact snapshots refuse symlinks: {item}")
    shutil.copytree(source, destination)


def _snapshot(chain: list[CertifiedAncestor], directory: Path, species: str) -> dict[str, Any]:
    """Flatten actual ancestry into portable stage directories, retaining originals."""
    root = directory / "snapshot"
    root.mkdir()
    records = {}
    for ancestor in chain:
        entry = load_stage_manifest(species).by_id(ancestor.stage_id)
        stage = root / stage_dirname(species, entry.reference)
        _copy_tree(ancestor.stage_dir, stage)
        records[entry.id] = {
            "run_id": ancestor.run_id,
            "model_sha256": ancestor.model_sha256,
            "normalization_sha256": ancestor.normalization_sha256,
        }
        archive = directory / "source_records" / entry.id
        archive.mkdir(parents=True)
        provenance_path = ancestor.source_run_dir / "provenance.json"
        copied_source = provenance_path.is_file() and _read(provenance_path).get("canonical_library_copy") is True
        records_root = (
            ancestor.source_run_dir.parent / "source_records" / entry.id if copied_source else ancestor.source_run_dir
        )
        for name in _ROOT_RECORDS:
            original = records_root / name
            if original.is_symlink():
                raise CanonicalLibraryError(f"Artifact snapshots refuse symlinks: {original}")
            if original.is_file():
                shutil.copy2(original, archive / name)
    leaf = chain[-1]
    (root / "provenance.json").write_text(
        json.dumps({"run_id": leaf.run_id, "canonical_library_copy": True, "original_sources": records}, indent=2)
        + "\n"
    )
    leaf_dir = root / stage_dirname(species, load_stage_manifest(species).by_id(leaf.stage_id).reference)
    return {
        "snapshot_root": "snapshot",
        "original_run_id": leaf.run_id,
        "stage_id": leaf.stage_id,
        "sources": records,
        "model_path": (leaf_dir / "models" / leaf.model_zip.name).relative_to(directory).as_posix(),
        "normalization_path": (leaf_dir / "models" / leaf.normalization_path.name).relative_to(directory).as_posix(),
    }


def publish_canonical_stage(
    library: str | Path,
    *,
    run_dir: str | Path,
    species: str,
    algorithm: str,
    entry: StageEntry,
    current_task_sha256: str,
    plant_identity: PlantIdentity,
    current_gate_config: Mapping[str, Any],
    parent_model_sha256: str | None = None,
    parent_normalization_sha256: str | None = None,
    benchmark: bool = True,
    comparison_episodes: int = 50,
) -> dict[str, Any]:
    """Preserve judged outcomes; only verified gate passes can be recommended."""
    from environments.shared.certified_library import publish_candidate

    species, algorithm = resolve_species_id(species), _algorithm(algorithm)
    key = canonical_library_key(
        species=species,
        algorithm=algorithm,
        entry=entry,
        current_task_sha256=current_task_sha256,
        plant_identity=plant_identity,
        current_gate_config=current_gate_config,
        parent_model_sha256=parent_model_sha256,
        parent_normalization_sha256=parent_normalization_sha256,
    )
    chain = _verified_chain(
        Path(run_dir),
        species=species,
        algorithm=algorithm,
        entry=entry,
        current_task_sha256=current_task_sha256,
        plant_identity=plant_identity,
        current_gate_config=current_gate_config,
        parent_model_sha256=parent_model_sha256,
        parent_normalization_sha256=parent_normalization_sha256,
        allow_failed_target=True,
    )
    ancestor = chain[-1]
    config = _read(ancestor.stage_dir / "stage_config.json")
    origin = _training_origin(ancestor, chain[-2] if len(chain) > 1 else None)
    seed = origin["seed"]
    recipe = hyperparameters_sha256(
        {f"{algorithm}_kwargs": config.get("hyperparameters", {}), "curriculum_kwargs": config.get("curriculum", {})},
        algorithm,
    )
    recorded_recipe = config.get("run", {}).get("hyperparameters_sha256")
    if recorded_recipe is not None and recorded_recipe != recipe:
        raise CanonicalLibraryError("Recorded recipe digest disagrees with actual training settings")
    comparison = (
        benchmark_canonical_stage(
            ancestor,
            species=species,
            algorithm=algorithm,
            plant_identity=plant_identity,
            comparison_episodes=comparison_episodes,
        )
        if benchmark and ancestor.verdict["passed"]
        else None
    )
    incumbent = None
    incumbent_comparison = None
    if comparison is not None:
        try:
            from environments.shared.certified_library import copy_recommended

            # The incumbent is an input to this run's comparison and must
            # remain inspectable even after the shared library disappears.
            incumbent = copy_recommended(Path(library), key, Path(run_dir))
        except LookupError:
            incumbent = None
        if incumbent is not None and incumbent["comparison"]["protocol"] != comparison["protocol"]:
            old = _verified_chain(
                Path(incumbent["directory"]) / "snapshot",
                species=species,
                algorithm=algorithm,
                entry=entry,
                current_task_sha256=current_task_sha256,
                plant_identity=plant_identity,
                current_gate_config=current_gate_config,
                parent_model_sha256=parent_model_sha256,
                parent_normalization_sha256=parent_normalization_sha256,
            )[-1]
            incumbent_comparison = benchmark_canonical_stage(
                old,
                species=species,
                algorithm=algorithm,
                plant_identity=plant_identity,
                comparison_episodes=comparison_episodes,
            )
    with tempfile.TemporaryDirectory(prefix="canonical-publication-") as temporary:
        staging = Path(temporary).resolve()
        metadata = _snapshot(chain, staging, species)
        # Revalidate the actual copied bytes. A training process that changes
        # its source artifacts while publication is copying must not publish a
        # certificate for a different model or normalization file.
        _verified_chain(
            staging / "snapshot",
            species=species,
            algorithm=algorithm,
            entry=entry,
            current_task_sha256=current_task_sha256,
            plant_identity=plant_identity,
            current_gate_config=current_gate_config,
            parent_model_sha256=parent_model_sha256,
            parent_normalization_sha256=parent_normalization_sha256,
            allow_failed_target=True,
        )
        if comparison is not None:
            (staging / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
        if incumbent_comparison is not None:
            (staging / "incumbent_comparison.json").write_text(
                json.dumps(incumbent_comparison, indent=2, allow_nan=False) + "\n"
            )
        files = {path.relative_to(staging).as_posix(): path for path in staging.rglob("*") if path.is_file()}
        publication = publish_candidate(
            Path(library),
            key=key,
            recipe_sha256=canonical_json_sha256(
                {
                    "recipe": recipe,
                    "parent_checkpoint_sha256": origin["parent_checkpoint_sha256"],
                    "parent_normalization_sha256": origin["parent_normalization_sha256"],
                }
            ),
            training_seed=seed,
            source_run_id=ancestor.run_id,
            files=files,
            model_path=metadata["model_path"],
            normalization_path=metadata["normalization_path"],
            certificate={
                "passed": bool(ancestor.verdict["passed"]),
                "gate_sha256": ancestor.gate_sha256,
                "gate_verdict": ancestor.verdict,
                "complete_parent_chain_verified": True,
            },
            comparison={key: value for key, value in comparison.items() if key != "episodes"} if comparison else None,
            incumbent_comparison={key: value for key, value in incumbent_comparison.items() if key != "episodes"}
            if incumbent_comparison
            else None,
            required_seeds=declared_certification_seeds(current_gate_config, stage=entry.id),
            metadata={"canonical": metadata, "training_origin": origin},
        )
    if comparison is not None:
        from environments.shared.certified_comparison import save_head_to_head

        publication["comparison_artifacts"] = save_head_to_head(
            Path(run_dir) / "comparison" / entry.id,
            candidate=comparison,
            incumbent=incumbent_comparison
            if incumbent_comparison is not None
            else (incumbent["comparison"] if incumbent is not None else None),
            publication=publication,
            incumbent_version=incumbent["version"] if incumbent is not None else None,
            incumbent_reused=incumbent is not None and incumbent_comparison is None,
        )
    return publication


def resolve_canonical_parent(
    library: str | Path,
    *,
    run_dir: str | Path,
    species: str,
    algorithm: str,
    entry: StageEntry,
    current_task_sha256: str,
    plant_identity: PlantIdentity,
    current_gate_config: Mapping[str, Any],
    parent_model_sha256: str | None = None,
    parent_normalization_sha256: str | None = None,
) -> CertifiedAncestor:
    """Copy the recommendation into this run, then independently validate its chain."""
    from environments.shared.certified_library import copy_recommended, resolve_recommended

    key = canonical_library_key(
        species=species,
        algorithm=algorithm,
        entry=entry,
        current_task_sha256=current_task_sha256,
        plant_identity=plant_identity,
        current_gate_config=current_gate_config,
        parent_model_sha256=parent_model_sha256,
        parent_normalization_sha256=parent_normalization_sha256,
    )
    selected = resolve_recommended(Path(library), key)
    required = declared_certification_seeds(current_gate_config, stage=entry.id)
    if selected["distinct_seeds"] < required:
        raise LookupError(f"Recommended {entry.id} is provisional under the current {required}-seed requirement")
    candidate = copy_recommended(Path(library), key, Path(run_dir))
    if candidate["distinct_seeds"] < required:
        raise LookupError(f"Copied {entry.id} does not meet the current {required}-seed requirement")
    root = Path(candidate["directory"]) / "snapshot"
    return _verified_chain(
        root,
        species=resolve_species_id(species),
        algorithm=_algorithm(algorithm),
        entry=entry,
        current_task_sha256=current_task_sha256,
        plant_identity=plant_identity,
        current_gate_config=current_gate_config,
        parent_model_sha256=parent_model_sha256,
        parent_normalization_sha256=parent_normalization_sha256,
    )[-1]


def resolve_canonical_locomotion(
    library: str | Path,
    *,
    run_dir: str | Path,
    species: str,
    algorithm: str = "ppo",
) -> CertifiedAncestor:
    """Resolve and copy a coherent current locomotion chain without folder paths."""
    species = resolve_species_id(species)
    manifest, plant = load_stage_manifest(species), current_plant_identity(species)
    parent = None
    for entry in manifest.chain_for("locomotion"):
        config, fingerprint = _current_context(species, entry, plant)
        parent = resolve_canonical_parent(
            library,
            run_dir=run_dir,
            species=species,
            algorithm=algorithm,
            entry=entry,
            current_task_sha256=fingerprint["task_sha256"],
            plant_identity=plant,
            current_gate_config=config["curriculum_kwargs"],
            parent_model_sha256=parent.model_sha256 if parent else None,
            parent_normalization_sha256=parent.normalization_sha256 if parent else None,
        )
    assert parent is not None
    return parent


def copy_canonical_ancestor(ancestor: CertifiedAncestor, run_dir: str | Path) -> CertifiedAncestor:
    """Materialize an explicit TRUNK_FROM choice with all artifacts and ancestry."""
    config = _read(ancestor.stage_dir / "stage_config.json")
    species, algorithm = resolve_species_id(config["species"]), _algorithm(config["algorithm"])
    plant = current_plant_identity(species)
    entry = load_stage_manifest(species).by_id(ancestor.stage_id)
    current_config, task = _current_context(species, entry, plant)
    parent_hash = config.get("run", {}).get("parent_checkpoint_sha256") if entry.warm_start_from else None
    chain = _verified_chain(
        ancestor.source_run_dir,
        species=species,
        algorithm=algorithm,
        entry=entry,
        current_task_sha256=task["task_sha256"],
        plant_identity=plant,
        current_gate_config=current_config["curriculum_kwargs"],
        parent_model_sha256=parent_hash,
    )
    parent = Path(run_dir) / "certified_inputs" / entry.id
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".manual-", dir=parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        _snapshot(chain, staging, species)
        files = {
            path.relative_to(staging).as_posix(): sha256_file(path) for path in staging.rglob("*") if path.is_file()
        }
        payload = {"files": files}
        version = canonical_json_sha256(payload).removeprefix("sha256:")
        destination = parent / ("manual-" + version)
        (staging / "copy_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
        if not destination.exists():
            staging.rename(destination)
    manifest = _read(destination / "copy_manifest.json")
    if canonical_json_sha256(manifest).removeprefix("sha256:") != version:
        raise CanonicalLibraryError("Copied manual ancestor manifest changed")
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual != set(files) | {"copy_manifest.json"}:
        raise CanonicalLibraryError("Copied manual ancestor artifact inventory changed")
    for relative, expected in files.items():
        path = destination / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise CanonicalLibraryError(f"Copied manual ancestor artifact changed: {relative}")
    copied = _verified_chain(
        destination / "snapshot",
        species=species,
        algorithm=algorithm,
        entry=entry,
        current_task_sha256=task["task_sha256"],
        plant_identity=plant,
        current_gate_config=current_config["curriculum_kwargs"],
        parent_model_sha256=parent_hash,
    )[-1]
    return replace(copied, run_id=ancestor.run_id, via=ancestor.via)
