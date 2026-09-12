"""Shared builders for the result-bundle test modules.

``test_result_bundle_*.py`` each cover one submodule of
``environments.shared.result_bundle``; these builders assemble the on-disk
run directories they all operate on.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypedDict

from environments.shared.reporting import save_evaluation_episodes, save_result_bundle
from environments.shared.result_bundle import (
    ANCESTOR_RECORD_NAME,
    ANCESTOR_RECORD_SCHEMA,
    ANCESTORS_DIRNAME,
    sha256_file,
    write_gate_verdict,
)
from environments.shared.stage_manifest import load_stage_manifest, stage_label

from .reporting_helpers import make_plant_identity

_COMMIT = "a" * 40
#: The task fingerprint a reused ancestor carries in every record of it.
_ANCESTOR_TASK_SHA256 = "sha256:" + "7" * 64
#: The run the reused trunk came from (its provenance run_id).
_TRUNK_RUN_ID = "velociraptor-stable-baselines3-ppo-trunk"


class _InitializeResultBundleKwargs(TypedDict):
    species: str
    algorithm: str
    backend: str
    seed: int
    evaluation_seeds: list[int]
    evaluation_episodes: int
    parallel_envs: int
    plant_identity: dict[str, Any]
    run_id: str
    captured_at: str


def _plant_identity(species: str = "velociraptor") -> dict[str, Any]:
    model_paths = {
        "velociraptor": "environments/velociraptor/assets/raptor.xml",
        "trex": "environments/trex/assets/trex.xml",
    }
    return make_plant_identity(species=species, model_path=model_paths[species]).to_dict()


def _stage_result(
    stage: "int | str",
    *,
    model_path: Path | None = None,
    magnitude: int | None = None,
    species: str = "velociraptor",
) -> dict[str, Any]:
    # ``magnitude`` decouples the synthetic metric values from the stage
    # REFERENCE, which since the manifest migration may be a semantic id
    # (recovery) rather than a legacy number.  Integer stages keep their
    # historical stage-derived values so every existing assertion holds.
    if magnitude is None:
        if not isinstance(stage, int):
            raise ValueError(f"semantic stage {stage!r} needs an explicit magnitude")
        magnitude = stage
    result: dict[str, Any] = {
        "stage": stage,
        "name": f"Stage {stage}",
        "description": f"Description for stage {stage}",
        "timesteps": magnitude * 100_000,
        "duration_seconds": magnitude * 10.0,
        "mean_reward": 50.0 + magnitude,
        "std_reward": 2.0,
        "mean_episode_length": 200.0 + magnitude,
        "std_episode_length": 4.9,
        "mean_forward_vel": 0.25 * magnitude,
        "std_forward_vel": 0.05,
        "mean_distance_traveled": 1.5 * magnitude,
        "mean_success_rate": magnitude / 3,
        "best_eval_reward": 75.0 + magnitude,
        "best_eval_std": 1.5,
        "best_eval_length": 210.0 + magnitude,
        "best_eval_timestep": magnitude * 90_000,
        "best_model_reward": float(magnitude + 1),
        "best_model_std_reward": 0.82,
        "best_model_length": 110.0,
        "best_model_std_length": 8.2,
        "best_model_fwd_vel": 0.2,
        "best_model_std_fwd_vel": 0.082,
        "best_model_distance": 2.0,
        "best_model_success_rate": 0.6667,
        "gate_passed": True,
        "publication_gate_passed": True,
        "plant_identity": _plant_identity(species),
    }
    if model_path is not None:
        result["model_path"] = str(model_path)
    return result


def _stage_config(
    stage: "int | str",
    algorithm: str,
    *,
    curriculum: dict[str, Any] | None = None,
    magnitude: int | None = None,
) -> dict[str, Any]:
    if magnitude is None:
        magnitude = stage if isinstance(stage, int) else 1
    algorithm_key = "jax_kwargs" if algorithm == "JAX_PPO" else f"{algorithm.lower()}_kwargs"
    return {
        "name": f"Stage {stage}",
        "description": f"Description for stage {stage}",
        "env_kwargs": {"forward_vel_weight": float(magnitude)},
        algorithm_key: {"learning_rate": 3e-4},
        "curriculum_kwargs": dict(curriculum) if curriculum is not None else {"min_avg_reward": 1.0},
    }


def _legacy_stage_dirname(stage: "int | str") -> str:
    # stage{N} for legacy numbers; the bare id for semantic stages
    # (stage_manifest.stage_label semantics — "recovery", never a minted
    # "stagerecovery").
    return f"stage{stage}" if isinstance(stage, int) else str(stage)


def _write_stage_configs(
    run_dir: Path,
    stage_configs: "dict[int | str, dict[str, Any]]",
    dirname=_legacy_stage_dirname,
    species: str = "velociraptor",
) -> None:
    for stage, config in stage_configs.items():
        stage_dir = run_dir / dirname(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        if "jax_kwargs" in config:
            algorithm = "JAX_PPO"
        elif "sac_kwargs" in config:
            algorithm = "SAC"
        else:
            algorithm = "PPO"
        saved_config = {
            **config,
            "species": species,
            "stage": stage,
            "algorithm": algorithm,
            "plant_identity": _plant_identity(species),
        }
        (stage_dir / "stage_config.json").write_text(
            json.dumps(saved_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _snapshot_files(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _rewrite_csv_cell(path: Path, *, field: str, value: str) -> None:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    assert rows
    assert field in fieldnames
    rows[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_csv_column(path: Path, *, field: str, value: str | None) -> None:
    """Set *field* on every row, or drop the column entirely when *value* is None."""
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    assert field in fieldnames
    if value is None:
        fieldnames.remove(field)
        for row in rows:
            del row[field]
    else:
        for row in rows:
            row[field] = value
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _complete_bundle_inputs(
    run_dir: Path,
    *,
    algorithm: str,
    dirname=_legacy_stage_dirname,
    species: str = "velociraptor",
    stage_refs: "tuple[int | str, ...]" = (1, 2, 3),
) -> "tuple[list[dict[str, Any]], dict[int | str, dict[str, Any]]]":
    # Metric magnitudes stay keyed to the legacy trio's historical values;
    # a semantic stage (recovery) reuses magnitude 1 so its evaluation
    # evidence below reproduces the same aggregates its stage result claims.
    def _magnitude(stage: "int | str") -> int:
        return stage if isinstance(stage, int) else 1

    stage_configs: "dict[int | str, dict[str, Any]]" = {}
    for stage in stage_refs:
        if stage == "recovery":
            # The pilot's real declaration: a recorded non-advancing gate.
            curriculum = {"gate_kind": "none/v1", "gate_schema_version": 1}
        else:
            curriculum = None
        stage_configs[stage] = _stage_config(stage, algorithm, curriculum=curriculum, magnitude=_magnitude(stage))
    _write_stage_configs(run_dir, stage_configs, dirname=dirname, species=species)
    stage_results = []
    for stage in stage_refs:
        magnitude = _magnitude(stage)
        selected_model = run_dir / dirname(stage) / "models" / "best_model.pkl"
        selected_model.parent.mkdir(parents=True, exist_ok=True)
        selected_model.write_bytes(f"selected model stage {stage}".encode())
        selected_vecnorm = run_dir / dirname(stage) / "models" / "best_model_vecnorm.pkl"
        selected_vecnorm.write_bytes(f"selected normalization stage {stage}".encode())
        stage_result = _stage_result(stage, model_path=selected_model, magnitude=magnitude, species=species)
        stage_result["vecnorm_path"] = str(selected_vecnorm)
        if stage == "recovery":
            # none/v1 refuses to pass; the pilot's honest verdict is False.
            stage_result["gate_passed"] = False
            stage_result["publication_gate_passed"] = False
        stage_results.append(stage_result)
    for stage in stage_refs:
        magnitude = _magnitude(stage)
        final_reward = 50.0 + magnitude
        final_forward_velocity = 0.25 * magnitude
        final_distance = 1.5 * magnitude
        model_dir = run_dir / dirname(stage) / "models"
        final_model = model_dir / f"{stage_label(stage)}_final.pkl"
        final_model.write_bytes(f"final model stage {stage}".encode())
        final_vecnorm = model_dir / f"{stage_label(stage)}_final_vecnorm.pkl"
        final_vecnorm.write_bytes(f"final normalization stage {stage}".encode())
        save_evaluation_episodes(
            run_dir / dirname(stage),
            rewards=[
                final_reward - 2.449489743,
                final_reward,
                final_reward + 2.449489743,
            ],
            lengths=[194 + magnitude, 200 + magnitude, 206 + magnitude],
            forward_velocities=[
                final_forward_velocity - 0.061237244,
                final_forward_velocity,
                final_forward_velocity + 0.061237244,
            ],
            distances=[final_distance - 1.0, final_distance, final_distance + 1.0],
            successes=[episode < magnitude for episode in range(3)],
            evaluation_seed=101,
            checkpoint_label="final",
            checkpoint_path=final_model,
            normalization_path=final_vecnorm,
        )
        save_evaluation_episodes(
            run_dir / dirname(stage),
            rewards=[float(magnitude), float(magnitude + 1), float(magnitude + 2)],
            lengths=[100, 110, 120],
            forward_velocities=[0.1, 0.2, 0.3],
            distances=[1.0, 2.0, 3.0],
            successes=[False, True, True],
            evaluation_seed=101,
            checkpoint_label="selected",
            checkpoint_path=model_dir / "best_model.pkl",
            normalization_path=model_dir / "best_model_vecnorm.pkl",
        )
    return stage_results, stage_configs


def _write_ancestor_record(
    run_dir: Path,
    stage: "int | str",
    *,
    species: str = "velociraptor",
    passed: bool = True,
    parent_run_id: str = _TRUNK_RUN_ID,
    task_sha256: str = _ANCESTOR_TASK_SHA256,
    source_run_dir: Path | None = None,
) -> "tuple[Path, str]":
    """Write ``ancestors/<stage_id>/`` for a node reused from another run.

    Reproduces the ANCESTORS LAYOUT contract by hand — ``ancestor.json`` plus
    verbatim copies of the ancestor stage's ``gate_verdict.json``,
    ``stage_config.json``, ``task_fingerprint.json`` and
    ``plant_identity.json`` — so this helper pins the layout independently
    of the writer in ``environments.shared.ancestors``.  The ancestor's
    handoff pair lives OUTSIDE the bundle (the trunk run's directory); only
    its hashes travel.  Returns the record directory and the handoff
    checkpoint's sha256.
    """
    entry = load_stage_manifest(species).resolve(stage)
    source_root = source_run_dir if source_run_dir is not None else run_dir.parent / "trunk-run"
    source_stage_dir = source_root / _legacy_stage_dirname(stage)
    source_models = source_stage_dir / "models"
    source_models.mkdir(parents=True, exist_ok=True)
    checkpoint = source_models / "best_model.zip"
    if not checkpoint.exists():
        checkpoint.write_bytes(f"trunk checkpoint for {entry.id}".encode())
    normalization = source_models / "best_model_vecnorm.pkl"
    if not normalization.exists():
        normalization.write_bytes(f"trunk normalization for {entry.id}".encode())
    stage_config = {
        **_stage_config(stage, "PPO", magnitude=1),
        "species": species,
        "stage": entry.reference,
        "algorithm": "PPO",
        "plant_identity": _plant_identity(species),
        "task_fingerprint": {"schema": "mesozoic.task-fingerprint/v2", "task_sha256": task_sha256},
    }
    (source_stage_dir / "stage_config.json").write_text(
        json.dumps(stage_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_gate_verdict(
        source_stage_dir,
        species=species,
        stage=entry.reference,
        stage_id=entry.id,
        gate_kind="reward_and_length/v1",
        gate_schema_version=1,
        passed=passed,
        failures=[] if passed else ["min_avg_reward: evidence=0.5 threshold=1.0"],
        task_sha256=task_sha256,
        judged_by="reporting.stage_artifacts.generate_stage_artifacts",
        checkpoint=checkpoint,
        normalization=normalization,
    )

    record_dir = run_dir / ANCESTORS_DIRNAME / entry.id
    record_dir.mkdir(parents=True, exist_ok=True)
    for name in ("gate_verdict.json", "stage_config.json"):
        (record_dir / name).write_bytes((source_stage_dir / name).read_bytes())
    (record_dir / "task_fingerprint.json").write_text(
        json.dumps(stage_config["task_fingerprint"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (record_dir / "plant_identity.json").write_text(
        json.dumps(_plant_identity(species), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = {
        "schema": ANCESTOR_RECORD_SCHEMA,
        "stage_id": entry.id,
        "stage_key": entry.key,
        "parent_run_id": parent_run_id,
        "source_run_dir": str(source_root),
        "source_stage_dir": str(source_stage_dir),
        "handoff": {
            "name": "best_model",
            "model_path": str(checkpoint),
            "model_sha256": sha256_file(checkpoint),
            "normalization_path": str(normalization),
            "normalization_sha256": sha256_file(normalization),
        },
        "task_sha256": task_sha256,
        "judged_by": "reporting.stage_artifacts.generate_stage_artifacts",
        "reused_at": "2026-09-06T12:00:00+00:00",
    }
    (record_dir / ANCESTOR_RECORD_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record_dir, sha256_file(checkpoint)


def _reused_trunk_bundle_inputs(
    run_dir: Path,
    *,
    reused: "tuple[int | str, ...]" = (1,),
    trained: "tuple[int | str, ...]" = (2, 3),
    species: str = "velociraptor",
    reused_passed: bool = True,
) -> "tuple[list[dict[str, Any]], dict[int | str, dict[str, Any]]]":
    """A run that reused *reused* from another run and trained *trained* here.

    Writes an ``ancestors/<id>/`` record per reused node and, on the FIRST
    trained stage, the run-block lineage a reuse leaves behind:
    ``load_path`` outside the bundle, ``load_mode`` initialize_next_stage,
    ``parent_run_id`` and ``parent_checkpoint_sha256`` bound to the record.
    """
    stage_results, stage_configs = _complete_bundle_inputs(
        run_dir, algorithm="PPO", species=species, stage_refs=trained
    )
    hashes: dict[Any, str] = {}
    for stage in reused:
        record_dir, checkpoint_hash = _write_ancestor_record(run_dir, stage, species=species, passed=reused_passed)
        hashes[stage] = checkpoint_hash
    first_trained = trained[0]
    parent = load_stage_manifest(species).parent_of(first_trained)
    if parent is not None and parent.reference in hashes:
        config_path = run_dir / _legacy_stage_dirname(first_trained) / "stage_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["run"] = {
            "seed": 42,
            "n_envs": 4,
            "load_path": str(run_dir.parent / "trunk-run" / _legacy_stage_dirname(parent.reference) / "models"),
            "load_mode": "initialize_next_stage",
            "parent_task_sha256": _ANCESTOR_TASK_SHA256,
            "parent_checkpoint_sha256": hashes[parent.reference],
            "parent_run_id": _TRUNK_RUN_ID,
        }
        config["run"]["load_path"] += "/best_model"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stage_results, stage_configs


def _complete_bundle(
    run_dir: Path,
    *,
    algorithm: str,
    backend: str,
    dirname=_legacy_stage_dirname,
    species: str = "velociraptor",
    stage_refs: "tuple[int | str, ...]" = (1, 2, 3),
) -> "tuple[dict[str, Path], list[dict[str, Any]], dict[int | str, dict[str, Any]]]":
    stage_results, stage_configs = _complete_bundle_inputs(
        run_dir,
        algorithm=algorithm,
        dirname=dirname,
        species=species,
        stage_refs=stage_refs,
    )
    paths = save_result_bundle(
        stage_results,
        stage_configs,
        species,
        algorithm,
        42,
        run_dir,
        backend=backend,
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(species),
        run_id=f"{species}-{backend}-{algorithm.lower()}-test",
    )
    return paths, stage_results, stage_configs
