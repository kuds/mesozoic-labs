"""Idempotent, Drive-portable result bundle publication.

Validates a curriculum run end to end, then writes provenance, CSV, an
artifact manifest, and (only for a complete, passing curriculum) the public
``summary.json``."""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .csv_output import build_results_csv_rows, save_results_csv
from .formatting import parse_optional_bool
from .summaries import _backend_version, build_result_summary


def _resolve_model_artifact(model_path: Any, *, run_dir: Path) -> Path | None:
    if model_path in {None, ""}:
        return None
    candidate = Path(str(model_path))
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidates = [candidate]
    if candidate.suffix == "":
        candidates.extend([candidate.with_suffix(".zip"), candidate.with_suffix(".pkl")])
    for path in candidates:
        if path.is_file():
            return path
    return None


def save_result_bundle(
    stage_results_list: list[dict[str, Any]],
    stage_configs: dict[int, dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    run_dir: str | Path,
    *,
    backend: str | None = None,
    backend_version: str | None = None,
    hardware: str = "Google Colab",
    parallel_envs: int | None = None,
    evaluation_episodes: int = 30,
    evaluation_seeds: Sequence[int] | None = None,
    seed_roles: Mapping[str, int] | None = None,
    plant_identity: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write one idempotent, Drive-portable result bundle.

    Partial curricula receive provenance, CSV, and a ``partial``/``failed``
    manifest.  A public schema-v2 ``summary.json`` is emitted only when all
    three stages are present and the selected checkpoint, resolved configs,
    backend version, and plant identity are available.
    """
    from ..result_bundle import (
        ResultBundleError,
        _normalize_plant_identity,
        _write_json,
        aggregate_file_hash,
        canonical_algorithm,
        canonical_backend,
        compare_summary_to_csv,
        initialize_result_bundle,
        load_provenance,
        sha256_file,
        update_provenance,
        validate_evaluation_evidence,
        validate_result_bundle,
        write_artifact_manifest,
    )
    from ..result_schema import validate_result_summary

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    previous_manifest = run_path / "artifact_manifest.json"

    # Preflight every expected failure before invalidating an existing
    # completion marker. This is especially important on Drive, where a
    # disconnected runtime may not get another chance to rebuild the bundle.
    try:
        ordered_stage_numbers = [int(result["stage"]) for result in stage_results_list]
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultBundleError("every stage result must contain an integer stage") from exc
    stage_numbers = set(ordered_stage_numbers)
    if len(stage_numbers) != len(stage_results_list):
        raise ResultBundleError("stage_results_list contains duplicate stages")
    if not stage_numbers or not stage_numbers <= {1, 2, 3}:
        raise ResultBundleError("stage_results_list must contain stages 1, 2, or 3")
    expected_prefix = set(range(1, max(stage_numbers) + 1))
    if stage_numbers != expected_prefix:
        raise ResultBundleError(
            f"stage_results_list must be a contiguous curriculum prefix; found {sorted(stage_numbers)}"
        )
    gate_values: dict[int, bool] = {}
    for result in stage_results_list:
        stage = int(result["stage"])
        gate_value = parse_optional_bool(result.get("publication_gate_passed"))
        if gate_value is None:
            raise ResultBundleError(f"stage {stage} is missing an explicit boolean publication_gate_passed value")
        gate_values[stage] = gate_value
    has_all_stages = stage_numbers == {1, 2, 3}
    promotion_ready = has_all_stages and all(gate_values.values())
    status = "complete" if promotion_ready else ("failed" if not all(gate_values.values()) else "partial")

    summary_path = run_path / "summary.json"
    if not promotion_ready and summary_path.exists():
        raise ResultBundleError("non-publishable bundle contains a stale summary.json")
    if not evaluation_seeds:
        raise ResultBundleError("result bundle requires at least one recorded publication evaluation seed")
    if not isinstance(parallel_envs, int) or isinstance(parallel_envs, bool) or parallel_envs <= 0:
        raise ResultBundleError("result bundle requires a positive parallel_envs value")

    effective_plant = plant_identity
    if effective_plant is None and stage_results_list:
        final_stage_result = max(stage_results_list, key=lambda item: int(item["stage"]))
        candidate = final_stage_result.get("plant_identity")
        if isinstance(candidate, Mapping):
            effective_plant = candidate
    normalized_plant = _normalize_plant_identity(effective_plant, species=species)
    if normalized_plant is None:
        raise ResultBundleError("result bundle is missing plant identity")
    for result in stage_results_list:
        stage = int(result["stage"])
        stage_plant_value = result.get("plant_identity")
        if not isinstance(stage_plant_value, Mapping):
            raise ResultBundleError(f"stage {stage} is missing plant identity")
        stage_plant = _normalize_plant_identity(stage_plant_value, species=species)
        if stage_plant != normalized_plant:
            raise ResultBundleError(f"stage {stage} plant identity does not match the run identity")

    public_algorithm = canonical_algorithm(algorithm)
    public_backend = canonical_backend(algorithm, backend)
    detected_backend_version = backend_version or _backend_version(
        "JAX_PPO" if public_backend == "jax-mjx" else algorithm
    )
    if promotion_ready and not detected_backend_version:
        raise ResultBundleError("complete bundle requires a recorded backend version")

    config_paths = [run_path / f"stage{stage}" / "stage_config.json" for stage in sorted(stage_numbers)]
    missing_configs = [path for path in config_paths if not path.is_file()]
    if missing_configs:
        raise ResultBundleError(f"result bundle is missing resolved stage configs: {missing_configs}")
    resolved_stage_configs: dict[int, dict[str, Any]] = {}
    for stage, config_path in zip(sorted(stage_numbers), config_paths, strict=True):
        try:
            saved_config = _json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read resolved stage config {config_path}: {exc}") from exc
        if not isinstance(saved_config, Mapping):
            raise ResultBundleError(f"resolved stage config must contain an object: {config_path}")
        if saved_config.get("stage") is not None and int(saved_config["stage"]) != stage:
            raise ResultBundleError(f"resolved stage config is mislabeled for stage {stage}: {config_path}")
        if saved_config.get("species") not in {None, "", species}:
            raise ResultBundleError(f"resolved stage config species mismatch: {config_path}")
        if saved_config.get("algorithm"):
            try:
                saved_algorithm = canonical_algorithm(str(saved_config["algorithm"]))
            except ResultBundleError as exc:
                raise ResultBundleError(f"invalid algorithm in resolved stage config {config_path}") from exc
            if saved_algorithm != public_algorithm:
                raise ResultBundleError(f"resolved stage config algorithm mismatch: {config_path}")
        saved_plant_value = saved_config.get("plant_identity")
        if promotion_ready and saved_plant_value is None:
            raise ResultBundleError(f"complete bundle stage config is missing plant identity: {config_path}")
        if saved_plant_value is not None:
            if not isinstance(saved_plant_value, Mapping):
                raise ResultBundleError(f"invalid plant identity in resolved stage config: {config_path}")
            saved_plant = _normalize_plant_identity(saved_plant_value, species=species)
            if saved_plant != normalized_plant:
                raise ResultBundleError(f"resolved stage config plant identity mismatch: {config_path}")
        reward_weights = saved_config.get("reward_weights", saved_config.get("env_kwargs", {}))
        hyperparameters = saved_config.get(
            "hyperparameters",
            saved_config.get(
                "jax_kwargs"
                if public_backend == "jax-mjx"
                else ("sac_kwargs" if public_algorithm == "SAC" else "ppo_kwargs"),
                {},
            ),
        )
        curriculum = saved_config.get(
            "curriculum",
            saved_config.get("curriculum_kwargs", {}),
        )
        if not all(isinstance(value, Mapping) for value in (reward_weights, hyperparameters, curriculum)):
            raise ResultBundleError(f"resolved stage config sections must be objects: {config_path}")
        algorithm_config_key = (
            "jax_kwargs"
            if public_backend == "jax-mjx"
            else ("sac_kwargs" if public_algorithm == "SAC" else "ppo_kwargs")
        )
        resolved_stage_configs[stage] = {
            "name": str(saved_config.get("name") or f"Stage {stage}"),
            "description": str(saved_config.get("description") or f"Curriculum stage {stage}"),
            "env_kwargs": dict(reward_weights),
            algorithm_config_key: dict(hyperparameters),
            "curriculum_kwargs": dict(curriculum),
        }
    config_hash = aggregate_file_hash(config_paths, root=run_path)

    selected_checkpoints: dict[str, dict[str, Any]] = {}
    for result in stage_results_list:
        stage = int(result["stage"])
        model_artifact = _resolve_model_artifact(result.get("model_path"), run_dir=run_path)
        if promotion_ready and model_artifact is None:
            raise ResultBundleError(f"complete bundle is missing its selected Stage {stage} checkpoint")
        if model_artifact is None:
            continue
        try:
            selected_path = model_artifact.resolve().relative_to(run_path.resolve()).as_posix()
        except ValueError as exc:
            raise ResultBundleError(f"selected checkpoint lies outside run directory: {model_artifact}") from exc
        normalization_path: str | None = None
        normalization_hash: str | None = None
        if public_backend == "stable-baselines3":
            normalization_artifact = _resolve_model_artifact(
                result.get("vecnorm_path"),
                run_dir=run_path,
            )
            if promotion_ready and normalization_artifact is None:
                raise ResultBundleError(
                    f"complete SB3 bundle is missing selected Stage {stage} VecNormalize statistics"
                )
            if normalization_artifact is not None:
                try:
                    normalization_path = normalization_artifact.resolve().relative_to(run_path.resolve()).as_posix()
                except ValueError as exc:
                    raise ResultBundleError(
                        f"selected VecNormalize artifact lies outside run directory: {normalization_artifact}"
                    ) from exc
                normalization_hash = sha256_file(normalization_artifact)
        selected_checkpoints[str(stage)] = {
            "model_path": selected_path,
            "model_hash": sha256_file(model_artifact),
            "normalization_path": normalization_path,
            "normalization_hash": normalization_hash,
        }

    stage3_checkpoint = selected_checkpoints.get("3", {})
    selected_model_path = stage3_checkpoint.get("model_path")
    model_hash = stage3_checkpoint.get("model_hash")

    if promotion_ready:
        for stage in (1, 2, 3):
            for checkpoint_label in ("selected", "final"):
                evidence_path = run_path / f"stage{stage}" / f"evaluation_{checkpoint_label}.csv"
                if not evidence_path.is_file() or evidence_path.stat().st_size == 0:
                    raise ResultBundleError(
                        f"complete bundle is missing {checkpoint_label} evaluation evidence: {evidence_path}"
                    )

    previous_manifest_status: str | None = None
    if previous_manifest.exists():
        try:
            previous_manifest_value = _json.loads(previous_manifest.read_text(encoding="utf-8"))
            if isinstance(previous_manifest_value, Mapping):
                previous_manifest_status = previous_manifest_value.get("status")
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read existing artifact manifest: {exc}") from exc
        if previous_manifest_status == "complete":
            validate_result_bundle(run_path, require_complete=True)

    provenance_path = initialize_result_bundle(
        run_path,
        species=species,
        algorithm=algorithm,
        backend=backend,
        seed=seed,
        evaluation_seeds=evaluation_seeds,
        evaluation_episodes=evaluation_episodes,
        seed_roles=seed_roles,
        parallel_envs=parallel_envs,
        hardware=hardware,
        plant_identity=normalized_plant,
        run_id=run_id,
        repository_root=repository_root,
    )
    captured = load_provenance(run_path)
    finalization = {
        "model_hash": model_hash,
        "config_hash": config_hash,
        "backend_version": detected_backend_version,
        "selected_model_path": selected_model_path,
        "selected_checkpoints": selected_checkpoints,
    }
    finalized_provenance = {**captured, **finalization}
    result_date = str(captured.get("captured_at", "")).split("T", maxsplit=1)[0]

    prospective_summary: dict[str, Any] | None = None
    if promotion_ready:
        prospective_summary = build_result_summary(
            stage_results_list,
            species,
            algorithm,
            seed,
            hardware=str(captured["hardware"]),
            provenance=finalized_provenance,
            backend=public_backend,
            backend_version=detected_backend_version,
            parallel_envs=int(captured["parallel_envs"]),
            run_id=str(captured["run_id"]),
            result_date=result_date,
            plant_identity=normalized_plant,
        )
        validate_result_summary(
            prospective_summary,
            expected_species=species,
            require_complete=True,
            require_canonical_provenance=True,
            result_path=str(summary_path),
        )
        validate_evaluation_evidence(run_path, prospective_summary, finalized_provenance)
        if previous_manifest_status == "complete":
            existing_summary = _json.loads(summary_path.read_text(encoding="utf-8"))
            if existing_summary != prospective_summary:
                raise ResultBundleError("completed result bundle is immutable; use a new run_id for different results")
            return {
                "provenance": provenance_path,
                "collected_results_csv": run_path / "collected_results.csv",
                "summary": summary_path,
                "artifact_manifest": previous_manifest,
            }

    # Exercise CSV construction before removing the previous completion marker.
    build_results_csv_rows(
        stage_results_list,
        resolved_stage_configs,
        species,
        algorithm,
        seed,
        backend=public_backend,
        run_id=str(captured["run_id"]),
        provenance=finalized_provenance,
    )

    # All semantic validation is complete. Remove the old marker immediately
    # before changing derived artifacts, then write the new marker last.
    if previous_manifest.exists():
        previous_manifest.unlink()
    update_provenance(run_path, **finalization)
    captured = load_provenance(run_path)
    _write_json(run_path / "plant_identity.json", normalized_plant)

    csv_path = save_results_csv(
        stage_results_list,
        resolved_stage_configs,
        species,
        algorithm,
        seed,
        run_path,
        backend=public_backend,
        run_id=str(captured["run_id"]),
        provenance=captured,
    )
    paths: dict[str, Path] = {
        "provenance": provenance_path,
        "collected_results_csv": csv_path,
    }

    if promotion_ready:
        assert prospective_summary is not None
        _write_json(summary_path, prospective_summary)
        contradictions = compare_summary_to_csv(prospective_summary, csv_path)
        if contradictions:
            raise ResultBundleError("summary/CSV contradictions: " + "; ".join(contradictions))
        paths["summary"] = summary_path

    manifest_path = write_artifact_manifest(run_path, status=status)
    paths["artifact_manifest"] = manifest_path
    validate_result_bundle(run_path, require_complete=promotion_ready)
    return paths
