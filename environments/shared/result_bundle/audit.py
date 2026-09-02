"""Whole-bundle audit: the gate a run passes before repository promotion.

:func:`audit_result_bundle` reports every problem it finds;
:func:`validate_result_bundle` raises on the first one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..stage_manifest import find_stage_dir
from . import evidence, hashing
from .constants import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_PROVENANCE_NAME,
)
from .errors import ResultBundleError

# Imported by name, not through the module: `manifest` is also used as a local
# variable here, so `manifest.x()` would shadow it. Patch these at this module.
from .manifest import verify_artifact_manifest
from .naming import _normalize_plant_identity, _portable_relative_path
from .provenance import load_provenance

#: Load lineage a stage config's ``run`` block may carry once the trainer
#: persists it (``--load`` today validates the lineage and then forgets it).
#: Audited when present, skipped when absent, so bundles from either side of
#: that change keep auditing.
_LOAD_LINEAGE_KEYS = ("load_path", "load_mode", "parent_task_sha256", "parent_checkpoint_sha256")


def _lineage_parent_keys(load_path: str, run_path: Path) -> list[str]:
    """Manifest keys a recorded parent may live under, in lookup order.

    The path as recorded first, then the ``.zip`` SB3 appends to a stem: the
    manifest only ever hashes the ``.zip``, so a producer that recorded the
    stem it handed SB3 (rather than the file it hashed) must still bind to it.
    Empty for a parent outside the bundle — a Drive path from a previous run
    is not checkable here.
    """
    run_root = run_path.resolve()
    candidates = [load_path] if load_path.endswith(".zip") else [load_path, load_path + ".zip"]
    keys: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = run_root / path
        try:
            keys.append(path.resolve().relative_to(run_root).as_posix())
        except ValueError:
            continue
    return keys


def _audit_load_lineage(
    run_block: Mapping[str, Any],
    *,
    stage: "int | str",
    run_path: Path,
    declared_hashes: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a stage's recorded load lineage and the problems it has, if any."""
    from ..task_fingerprint import LOAD_MODES

    lineage = {key: run_block[key] for key in _LOAD_LINEAGE_KEYS if key in run_block}
    if not lineage:
        return None, []
    problems: list[str] = []
    load_path = lineage.get("load_path")
    if "load_path" in lineage and (not isinstance(load_path, str) or not load_path.strip()):
        problems.append(f"stage {stage} config run.load_path must be a non-empty string")
    if "load_mode" in lineage and lineage["load_mode"] not in LOAD_MODES:
        problems.append(f"stage {stage} config run.load_mode {lineage['load_mode']!r} is not one of {LOAD_MODES}")
    for key in ("parent_task_sha256", "parent_checkpoint_sha256"):
        digest = lineage.get(key)
        if key in lineage and (not isinstance(digest, str) or evidence._SHA256_DIGEST.fullmatch(digest) is None):
            problems.append(f"stage {stage} config run.{key} must be sha256:<64 lowercase hex>")
    parent_hash = lineage.get("parent_checkpoint_sha256")
    if isinstance(load_path, str) and load_path.strip() and isinstance(parent_hash, str):
        # A parent that lives in this bundle is hashed by the manifest; the
        # recorded lineage must agree with it.  A parent elsewhere (a Drive
        # path from a previous run) is not checkable here.
        parent_key = next((key for key in _lineage_parent_keys(load_path, run_path) if key in declared_hashes), None)
        if parent_key is not None and declared_hashes[parent_key] != parent_hash:
            problems.append(
                f"stage {stage} records parent_checkpoint_sha256 {parent_hash} for {parent_key}, "
                f"but the manifest hashes that artifact as {declared_hashes[parent_key]}"
            )
    return lineage, problems


def audit_result_bundle(
    run_dir: str | Path,
    *,
    verify_hashes: bool = True,
    reject_unlisted: bool = True,
    prospective_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a canonical or legacy Drive result directory without mutating it.

    *prospective_manifest* is audited in place of the on-disk
    ``artifact_manifest.json``: the bundle writer validates the manifest it is
    about to write, so a failed validation never leaves a completion marker
    behind.
    """
    from ..result_schema import (
        ResultSchemaError,
        ordered_stage_entries,
        validate_captured_provenance,
        validate_result_summary,
    )

    run_path = Path(run_dir).resolve()
    summary_path = run_path / "summary.json"
    csv_path = run_path / "collected_results.csv"
    provenance_path = run_path / DEFAULT_PROVENANCE_NAME
    manifest_path = run_path / DEFAULT_MANIFEST_NAME
    plant_path = run_path / "plant_identity.json"
    canonical = provenance_path.exists() or manifest_path.exists() or prospective_manifest is not None
    warnings: list[str] = []
    errors: list[str] = []
    lineage: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None

    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ResultBundleError("summary.json must contain an object")
            summary = loaded
            validate_result_summary(
                summary,
                expected_species=summary.get("species"),
                require_complete=True,
                require_canonical_provenance=canonical,
                result_path=str(summary_path),
            )
        except (OSError, json.JSONDecodeError, ResultBundleError, ResultSchemaError) as exc:
            errors.append(str(exc))

    if canonical:
        if provenance_path.exists():
            try:
                provenance = load_provenance(run_path)
                validate_captured_provenance(
                    provenance,
                    result_path=str(provenance_path),
                )
            except (ResultBundleError, ResultSchemaError) as exc:
                errors.append(str(exc))
        else:
            errors.append(f"canonical bundle is missing {DEFAULT_PROVENANCE_NAME}")

    if prospective_manifest is not None or manifest_path.exists():
        try:
            if verify_hashes:
                manifest = verify_artifact_manifest(
                    run_path,
                    reject_unlisted=reject_unlisted,
                    prospective_manifest=prospective_manifest,
                )
            else:
                loaded_manifest: Any = (
                    dict(prospective_manifest)
                    if prospective_manifest is not None
                    else json.loads(manifest_path.read_text(encoding="utf-8"))
                )
                if not isinstance(loaded_manifest, dict):
                    raise ResultBundleError("artifact manifest must contain an object")
                if loaded_manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
                    raise ResultBundleError("unsupported artifact manifest schema")
                if loaded_manifest.get("status") not in {"partial", "failed", "complete"}:
                    raise ResultBundleError("invalid artifact manifest status")
                manifest = loaded_manifest
        except (OSError, json.JSONDecodeError, ResultBundleError) as exc:
            errors.append(str(exc))
    elif canonical:
        errors.append(f"canonical bundle is missing {DEFAULT_MANIFEST_NAME}")

    if summary is not None and csv_path.exists():
        try:
            errors.extend(evidence.compare_summary_to_csv(summary, csv_path))
        except (OSError, ResultBundleError, ValueError) as exc:
            errors.append(str(exc))
    elif summary is None and csv_path.exists():
        warnings.append("CSV-only run cannot be promoted without a canonical summary")

    # The stages a complete bundle must prove are the stages its summary
    # RECORDS — historically the hardcoded trio (1, 2, 3), now resolved
    # through the species' manifest so a recorded recovery stage is audited
    # too (configs, evidence, hashes) rather than treated as foreign.  When
    # the summary is absent or its stage keys do not resolve, the audit
    # falls back to the advancing trio; the summary validation above has
    # already recorded that failure as an error of its own.
    ordered_stage_refs: "list[int | str]" = [1, 2, 3]
    if summary is not None and isinstance(summary.get("stages"), Mapping):
        try:
            summary_stage_entries = ordered_stage_entries(
                summary["stages"],
                species=str(summary.get("species")),
                field="summary stages",
            )
        except ResultSchemaError:
            summary_stage_entries = []
        if summary_stage_entries:
            ordered_stage_refs = [entry.reference for _, entry in summary_stage_entries]

    manifest_status = manifest.get("status") if manifest else None
    declared_hashes: dict[str, Any] = (
        {
            entry["path"]: entry.get("sha256")
            for entry in manifest.get("files", [])
            if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
        }
        if manifest
        else {}
    )
    declared_paths = set(declared_hashes)

    if canonical and provenance is not None:
        provenance_run_id = provenance.get("run_id")
        if manifest is not None and manifest.get("run_id") != provenance_run_id:
            errors.append(f"manifest/provenance run_id mismatch: {manifest.get('run_id')!r} != {provenance_run_id!r}")

        provenance_plant: dict[str, Any] | None = None
        if provenance.get("plant_identity") is not None:
            try:
                provenance_plant = _normalize_plant_identity(
                    provenance["plant_identity"],
                    species=provenance.get("species"),
                )
            except ResultBundleError as exc:
                errors.append(str(exc))

        if summary is not None:
            if summary.get("provenance") != provenance:
                errors.append("summary provenance does not match provenance.json")
            for summary_key, provenance_key in (
                ("run_id", "run_id"),
                ("species", "species"),
                ("algorithm", "algorithm"),
                ("backend", "backend"),
                ("seed", "training_seed"),
                ("hardware", "hardware"),
                ("parallel_envs", "parallel_envs"),
            ):
                if summary.get(summary_key) != provenance.get(provenance_key):
                    errors.append(
                        f"summary/provenance {summary_key} mismatch: "
                        f"{summary.get(summary_key)!r} != {provenance.get(provenance_key)!r}"
                    )
            try:
                summary_plant = _normalize_plant_identity(
                    summary.get("plant_identity"),
                    species=summary.get("species"),
                )
                if summary_plant != provenance_plant:
                    errors.append("summary plant_identity does not match provenance.json")
            except ResultBundleError as exc:
                errors.append(str(exc))

        if plant_path.exists():
            try:
                loaded_plant = json.loads(plant_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_plant, Mapping):
                    raise ResultBundleError("plant_identity.json must contain an object")
                root_plant = _normalize_plant_identity(
                    loaded_plant,
                    species=provenance.get("species"),
                )
                if root_plant != provenance_plant:
                    errors.append("plant_identity.json does not match provenance.json")
            except (OSError, json.JSONDecodeError, ResultBundleError) as exc:
                errors.append(str(exc))

        if manifest is not None:
            required_paths = {DEFAULT_PROVENANCE_NAME, "collected_results.csv"}
            if manifest_status == "complete":
                required_paths.update({"summary.json", "plant_identity.json"})
                # Per-stage requirements resolve to whatever the run actually
                # named its stage directories (stage{N} historically, NN_id
                # from 2026-08-20 on, the bare id in between for semantic
                # stages) — a complete bundle in any layout must audit,
                # never wedge as canonical-conflict.
                for stage in ordered_stage_refs:
                    stage_dir_name = find_stage_dir(run_path, stage).name
                    required_paths.update(
                        {
                            f"{stage_dir_name}/stage_config.json",
                            f"{stage_dir_name}/evaluation_final.csv",
                            f"{stage_dir_name}/evaluation_selected.csv",
                        }
                    )
            missing_declared = sorted(required_paths - declared_paths)
            if missing_declared:
                errors.append(f"manifest is missing required bundle artifacts: {missing_declared}")
            missing_files = sorted(path for path in required_paths if not (run_path / path).is_file())
            if missing_files:
                errors.append(f"bundle is missing required artifacts: {missing_files}")

        if summary is not None and manifest_status != "complete":
            errors.append("summary.json is only valid with a complete artifact manifest")

        if manifest_status == "complete":
            if summary is None:
                errors.append("complete bundle is missing summary.json")
            else:
                try:
                    evidence.validate_evaluation_evidence(run_path, summary, provenance, warnings=warnings)
                except (OSError, ResultBundleError) as exc:
                    errors.append(str(exc))
            if not plant_path.is_file():
                errors.append("complete bundle is missing plant_identity.json")

            # Manifest position order — the same order save_result_bundle
            # hashed the configs in (identical to sorted-by-number for
            # integer-only runs, so historical hashes still verify).
            config_paths = [find_stage_dir(run_path, stage) / "stage_config.json" for stage in ordered_stage_refs]
            if all(path.is_file() for path in config_paths):
                actual_config_hash = hashing.aggregate_file_hash(config_paths, root=run_path)
                if actual_config_hash != provenance.get("config_hash"):
                    errors.append(
                        "resolved stage config hash does not match provenance: "
                        f"{actual_config_hash!r} != {provenance.get('config_hash')!r}"
                    )
                for stage, config_path in zip(ordered_stage_refs, config_paths, strict=True):
                    try:
                        config_value = json.loads(config_path.read_text(encoding="utf-8"))
                        if not isinstance(config_value, Mapping):
                            raise ResultBundleError(f"{config_path} must contain an object")
                        config_plant = config_value.get("plant_identity")
                        if config_plant is None:
                            errors.append(f"stage {stage} config is missing plant_identity")
                        else:
                            normalized_config_plant = _normalize_plant_identity(
                                config_plant,
                                species=provenance.get("species"),
                            )
                            if normalized_config_plant != provenance_plant:
                                errors.append(f"stage {stage} config plant_identity does not match provenance.json")
                        run_block = config_value.get("run")
                        if isinstance(run_block, Mapping):
                            stage_lineage, lineage_problems = _audit_load_lineage(
                                run_block,
                                stage=stage,
                                run_path=run_path,
                                declared_hashes=declared_hashes,
                            )
                            errors.extend(lineage_problems)
                            if stage_lineage is not None:
                                lineage[str(stage)] = stage_lineage
                    except (OSError, json.JSONDecodeError, ResultBundleError) as exc:
                        errors.append(str(exc))

            try:
                selected_relative = _portable_relative_path(
                    provenance.get("selected_model_path"),
                    field="provenance.selected_model_path",
                )
                selected_model = (run_path / Path(*selected_relative.parts)).resolve()
                selected_model.relative_to(run_path)
                selected_model_key = selected_relative.as_posix()
                if selected_model_key not in declared_paths:
                    errors.append(f"manifest does not declare selected checkpoint: {selected_model_key}")
                if not selected_model.is_file():
                    errors.append(f"selected checkpoint is missing: {selected_model_key}")
                else:
                    actual_model_hash = hashing.sha256_file(selected_model)
                    if actual_model_hash != provenance.get("model_hash"):
                        errors.append(
                            "selected checkpoint hash does not match provenance: "
                            f"{actual_model_hash!r} != {provenance.get('model_hash')!r}"
                        )
            except (OSError, ValueError, ResultBundleError) as exc:
                errors.append(str(exc))

            selected_checkpoints = provenance.get("selected_checkpoints")
            if isinstance(selected_checkpoints, Mapping):
                for stage_key, checkpoint_value in selected_checkpoints.items():
                    if not isinstance(checkpoint_value, Mapping):
                        errors.append(f"selected checkpoint record for stage {stage_key} must be an object")
                        continue
                    for artifact_kind, path_key, hash_key in (
                        ("model", "model_path", "model_hash"),
                        ("normalization", "normalization_path", "normalization_hash"),
                    ):
                        relative_value = checkpoint_value.get(path_key)
                        expected_hash = checkpoint_value.get(hash_key)
                        if relative_value is None and expected_hash is None:
                            continue
                        try:
                            relative = _portable_relative_path(
                                relative_value,
                                field=f"stage {stage_key} selected {artifact_kind} path",
                            )
                            artifact = (run_path / Path(*relative.parts)).resolve()
                            artifact.relative_to(run_path)
                            artifact_key = relative.as_posix()
                            if artifact_key not in declared_paths:
                                errors.append(
                                    f"manifest does not declare stage {stage_key} selected "
                                    f"{artifact_kind}: {artifact_key}"
                                )
                            if not artifact.is_file():
                                errors.append(f"stage {stage_key} selected {artifact_kind} is missing: {artifact_key}")
                            else:
                                actual_hash = hashing.sha256_file(artifact)
                                if actual_hash != expected_hash:
                                    errors.append(
                                        f"stage {stage_key} selected {artifact_kind} hash does not match "
                                        f"provenance: {actual_hash!r} != {expected_hash!r}"
                                    )
                        except (OSError, ValueError, ResultBundleError) as exc:
                            errors.append(str(exc))

    if canonical:
        if errors:
            status = "canonical-conflict"
        elif manifest_status in {"partial", "failed"} and summary is None:
            status = manifest_status
        elif summary is not None and manifest_status == "complete":
            status = "canonical-valid"
        else:
            status = "partial"
    elif errors:
        status = "legacy-conflict"
    else:
        status = "legacy-unverified"
        warnings.append("legacy run has no captured provenance or artifact manifest")

    return {
        "status": status,
        "canonical": canonical,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "manifest_status": manifest_status,
        "errors": errors,
        "warnings": warnings,
        "lineage": lineage,
    }


def validate_result_bundle(
    run_dir: str | Path,
    *,
    require_complete: bool = True,
    reject_unlisted: bool = True,
    prospective_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a canonical Drive bundle and raise on conflicts."""
    report = audit_result_bundle(
        run_dir,
        reject_unlisted=reject_unlisted,
        prospective_manifest=prospective_manifest,
    )
    allowed = {"canonical-valid"} if require_complete else {"canonical-valid", "partial", "failed"}
    if report["status"] not in allowed:
        details = "; ".join([*report["errors"], *report["warnings"]])
        raise ResultBundleError(f"result bundle is {report['status']}: {details}")
    return report
