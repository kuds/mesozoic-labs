"""Idempotent, Drive-portable result bundle publication.

Validates a curriculum run end to end, then writes provenance, CSV, an
artifact manifest, and — whenever at least one deliverable is certified — the
public ``summary.json`` (result schema v4, BEHAVIOR_RECIPES_PLAN §4.3)."""

from __future__ import annotations

import json as _json
import logging
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..result_bundle.constants import DEFAULT_PROVENANCE_NAME
from ..stage_manifest import find_stage_dir
from . import csv_output, summaries
from .formatting import parse_optional_bool

logger = logging.getLogger(__name__)

#: The artifacts :func:`save_result_bundle` itself regenerates on every call —
#: exactly the files its write phase produces.  On re-entry over a complete
#: manifest that no longer verifies, these are the only files allowed to
#: disagree with it: they are rebuilt from the in-memory results, so a
#: disagreement there loses nothing certified.  A disagreement on anything
#: else — checkpoints and sidecars under ``models/``, evaluation evidence,
#: stage configs, diagnostics, gate files, task fingerprints, ancestor
#: records, a file that appeared after publication — is a change to a
#: certified artifact, and a rebuild would re-certify it under new hashes.
_REGENERATED_ARTIFACTS = frozenset(
    {DEFAULT_PROVENANCE_NAME, "plant_identity.json", "collected_results.csv", "summary.json"}
)


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
    stage_configs: "dict[int | str, dict[str, Any]]",
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
    target_deliverable: "int | str | None" = None,
) -> dict[str, Path]:
    """Write one idempotent, Drive-portable result bundle.

    Publication is per DELIVERABLE (result schema v4).  A deliverable node
    present in the run is *certified* when its own gate passed and every
    ``warm_start_from`` ancestor is present with a passed gate — trained in
    this run or carried by an ``ancestors/<stage_id>/`` record of a node
    reused from another run.  The bundle status is target-aware (decision
    D-A1): *target_deliverable* is the node the run aimed at — the manifest's
    last deliverable by default (the behavior node for every committed
    species, i.e. the historical terminal stage; the notebook passes its
    BEHAVIOR's node) — and the status is ``complete`` iff the target is
    present and certified AND every present deliverable is certified,
    ``partial`` iff at least one deliverable is certified, ``failed``
    otherwise.  A public ``summary.json`` is written whenever at least one
    deliverable is certified, so a failed leaf publishes its certified
    trunk and a walk-only run writes a valid bundle; its
    ``selected_model_path`` / ``model_hash`` are the PRIMARY deliverable's —
    the target when certified, else the deepest certified one.  A stance-only
    run targeting hunt is ``partial``, never ``complete``, so the bundle stays
    writable for the chain's next node; a ``complete`` marker is immutable.

    Every present stage's declared parent must be present or carried by an
    ancestor record.  Partial and failed curricula receive provenance, CSV
    and a manifest; a publishable one additionally requires the selected
    checkpoint (and SB3 sidecar), resolved config with plant identity, and
    evaluation evidence for every PRESENT stage, plus a recorded backend
    version.  Under a v1 or synthesized stage manifest — one deliverable, the
    last advancing node — all of this collapses to the pre-Phase-A rule.
    """
    from ..result_bundle import (
        ResultBundleError,
        _normalize_plant_identity,
        aggregate_file_hash,
        build_artifact_manifest,
        canonical_algorithm,
        canonical_backend,
        compare_summary_to_csv,
        hashing,
        initialize_result_bundle,
        load_ancestor_records,
        load_provenance,
        manifest_disagreements,
        project_ancestor_records,
        sha256_file,
        update_provenance,
        validate_evaluation_evidence,
        validate_result_bundle,
    )
    from ..result_bundle.audit import _audit_load_lineage, _lineage_parent_keys
    from ..result_schema import (
        ResultSchemaError,
        bundle_status_for,
        certified_deliverables,
        primary_deliverable_key,
        validate_result_summary,
    )
    from ..stage_manifest import StageManifestError, load_stage_manifest
    from .summaries import _stage_reference

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    previous_manifest = run_path / "artifact_manifest.json"

    # Preflight every expected failure before invalidating an existing
    # completion marker. This is especially important on Drive, where a
    # disconnected runtime may not get another chance to rebuild the bundle.
    #
    # Stage references resolve through the species' manifest and everything
    # downstream orders by manifest POSITION — never by int() on the
    # reference, which the semantic recovery stage has none of.
    try:
        stage_manifest = load_stage_manifest(species)
    except StageManifestError as exc:
        raise ResultBundleError(f"cannot load the stage manifest for {species!r}: {exc}") from exc
    keyed_results: list[tuple[Any, dict[str, Any]]] = []
    for result in stage_results_list:
        try:
            entry = stage_manifest.resolve(_stage_reference(result["stage"]))
        except (KeyError, TypeError, ValueError, StageManifestError) as exc:
            raise ResultBundleError(f"every stage result must name a stage the {species} manifest declares") from exc
        keyed_results.append((entry, result))
    keyed_results.sort(key=lambda pair: pair[0].position)
    present_ids = [entry.id for entry, _ in keyed_results]
    if len(set(present_ids)) != len(present_ids):
        raise ResultBundleError("stage_results_list contains duplicate stages")
    if not keyed_results:
        raise ResultBundleError("stage_results_list must contain at least one stage")
    if not stage_manifest.deliverables:
        raise ResultBundleError(f"the {species} stage manifest declares no deliverable node to publish")
    if target_deliverable is None:
        target_entry = stage_manifest.deliverables[-1]
    else:
        try:
            target_entry = stage_manifest.resolve(_stage_reference(target_deliverable))
        except (TypeError, ValueError, StageManifestError) as exc:
            raise ResultBundleError(
                f"target_deliverable {target_deliverable!r} is not a stage the {species} manifest declares"
            ) from exc
        if not target_entry.deliverable:
            raise ResultBundleError(
                f"target_deliverable {target_deliverable!r} is not a deliverable of the {species} manifest"
            )
    # Nodes reused from another run (BEHAVIOR_RECIPES_PLAN §4.2): their
    # records stand in for the stages this run did not train.  Every
    # malformation fails closed here, before anything is written.
    ancestor_records = load_ancestor_records(run_path, species=species)
    present_id_set = set(present_ids)
    reused_and_trained = [
        record["stage_id"] for record in ancestor_records.values() if record["stage_id"] in present_id_set
    ]
    if reused_and_trained:
        raise ResultBundleError(
            f"stages {reused_and_trained} are recorded both as trained in this run and as reused ancestors"
        )
    # The historical "contiguous curriculum prefix" rule, restated over the
    # manifest's EDGES: a stage's record is only coherent if the parent it
    # warm-started from was trained in this run or reused as an ancestor.
    # Transitively that is the old rule for every legacy chain; a node
    # nobody warm-starts from (recovery today) may be absent anywhere.
    for entry, _ in keyed_results:
        parent = stage_manifest.parent_of(entry.id)
        if parent is None or parent.id in present_id_set or parent.key in ancestor_records:
            continue
        raise ResultBundleError(
            f"stage_results_list must be a contiguous curriculum prefix over the manifest's edges; stage "
            f"{entry.reference} is recorded without its declared parent {parent.reference} "
            f"(warm_start_from = {parent.id!r}); neither trained in this run nor recorded under ancestors/"
        )
    gate_values: dict[str, bool] = {}
    for entry, result in keyed_results:
        gate_value = parse_optional_bool(result.get("publication_gate_passed"))
        if gate_value is None:
            raise ResultBundleError(
                f"stage {entry.reference} is missing an explicit boolean publication_gate_passed value"
            )
        gate_values[entry.key] = gate_value
    # Certification per deliverable: its own gate plus every chain ancestor's,
    # in this run or in an ancestor record.  Recovery's honest False verdict
    # (gate_kind none/v1 refuses to pass) leaves recovery uncertified and
    # everything else untouched — it neither completes nor fails the run.
    try:
        certified = certified_deliverables(
            [(entry.key, entry) for entry, _ in keyed_results],
            gate_values,
            ancestor_records,
            species=species,
        )
        status = bundle_status_for(certified, species=species, target=target_entry.key, stages=gate_values)
    except ResultSchemaError as exc:
        raise ResultBundleError(str(exc)) from exc
    publishable = any(certified.values())

    summary_path = run_path / "summary.json"
    if not publishable and summary_path.exists():
        raise ResultBundleError("non-publishable bundle contains a stale summary.json")
    if not evaluation_seeds:
        raise ResultBundleError("result bundle requires at least one recorded publication evaluation seed")
    if not isinstance(parallel_envs, int) or isinstance(parallel_envs, bool) or parallel_envs <= 0:
        raise ResultBundleError("result bundle requires a positive parallel_envs value")

    effective_plant = plant_identity
    if effective_plant is None and keyed_results:
        # keyed_results is in manifest position order, so [-1] is the run's
        # latest curriculum stage (the historical max-by-int, generalized).
        candidate = keyed_results[-1][1].get("plant_identity")
        if isinstance(candidate, Mapping):
            effective_plant = candidate
    normalized_plant = _normalize_plant_identity(effective_plant, species=species)
    if normalized_plant is None:
        raise ResultBundleError("result bundle is missing plant identity")
    for entry, result in keyed_results:
        stage_plant_value = result.get("plant_identity")
        if not isinstance(stage_plant_value, Mapping):
            raise ResultBundleError(f"stage {entry.reference} is missing plant identity")
        stage_plant = _normalize_plant_identity(stage_plant_value, species=species)
        if stage_plant != normalized_plant:
            raise ResultBundleError(f"stage {entry.reference} plant identity does not match the run identity")

    public_algorithm = canonical_algorithm(algorithm)
    public_backend = canonical_backend(algorithm, backend)
    detected_backend_version = backend_version or summaries._backend_version(
        "JAX_PPO" if public_backend == "jax-mjx" else algorithm
    )
    if publishable and not detected_backend_version:
        raise ResultBundleError("publishable bundle requires a recorded backend version")

    # Position order, matching the audit's recomputation: the aggregate
    # config hash is order-sensitive, and for integer-only runs position
    # order IS the historical sorted-by-number order, so old hashes hold.
    ordered_refs = [entry.reference for entry, _ in keyed_results]
    config_paths = [find_stage_dir(run_path, ref) / "stage_config.json" for ref in ordered_refs]
    missing_configs = [path for path in config_paths if not path.is_file()]
    if missing_configs:
        raise ResultBundleError(f"result bundle is missing resolved stage configs: {missing_configs}")
    resolved_stage_configs: dict[Any, dict[str, Any]] = {}
    run_blocks: dict[Any, Any] = {}
    for (entry, _), config_path in zip(keyed_results, config_paths, strict=True):
        stage = entry.reference
        try:
            saved_config = _json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read resolved stage config {config_path}: {exc}") from exc
        if not isinstance(saved_config, Mapping):
            raise ResultBundleError(f"resolved stage config must contain an object: {config_path}")
        if saved_config.get("stage") is not None:
            try:
                saved_stage = _stage_reference(saved_config["stage"])
            except ValueError as exc:
                raise ResultBundleError(
                    f"resolved stage config is mislabeled for stage {stage}: {config_path}"
                ) from exc
            if saved_stage != stage:
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
        if publishable and saved_plant_value is None:
            raise ResultBundleError(f"publishable bundle stage config is missing plant identity: {config_path}")
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
        run_blocks[stage] = saved_config.get("run")
        resolved_stage_configs[stage] = {
            "name": str(saved_config.get("name") or f"Stage {stage}"),
            "description": str(saved_config.get("description") or f"Curriculum stage {stage}"),
            "env_kwargs": dict(reward_weights),
            algorithm_config_key: dict(hyperparameters),
            "curriculum_kwargs": dict(curriculum),
        }
    config_hash = aggregate_file_hash(config_paths, root=run_path)

    selected_checkpoints: dict[str, dict[str, Any]] = {}
    for entry, result in keyed_results:
        stage = entry.reference
        model_artifact = _resolve_model_artifact(result.get("model_path"), run_dir=run_path)
        if publishable and model_artifact is None:
            raise ResultBundleError(f"publishable bundle is missing its selected Stage {stage} checkpoint")
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
            if publishable and normalization_artifact is None:
                raise ResultBundleError(
                    f"publishable SB3 bundle is missing selected Stage {stage} VecNormalize statistics"
                )
            if normalization_artifact is not None:
                try:
                    normalization_path = normalization_artifact.resolve().relative_to(run_path.resolve()).as_posix()
                except ValueError as exc:
                    raise ResultBundleError(
                        f"selected VecNormalize artifact lies outside run directory: {normalization_artifact}"
                    ) from exc
                normalization_hash = sha256_file(normalization_artifact)
        selected_checkpoints[entry.key] = {
            "model_path": selected_path,
            "model_hash": sha256_file(model_artifact),
            "normalization_path": normalization_path,
            "normalization_hash": normalization_hash,
        }

    if publishable:
        # Every RECORDED stage — the failed leaf and the optional recovery
        # pilot included — must carry its evaluation evidence; a stage in
        # the bundle without evidence would publish unverifiable numbers.
        for ref in ordered_refs:
            for checkpoint_label in ("selected", "final"):
                evidence_path = find_stage_dir(run_path, ref) / f"evaluation_{checkpoint_label}.csv"
                if not evidence_path.is_file() or evidence_path.stat().st_size == 0:
                    raise ResultBundleError(
                        f"publishable bundle is missing {checkpoint_label} evaluation evidence: {evidence_path}"
                    )

    previous_manifest_status: str | None = None
    previous_manifest_value: Any = None
    # A complete marker that still verifies makes this call a read-only
    # re-export (identical results return early below; different results
    # are refused).  One that no longer verifies is neither downgraded nor
    # unlinked here: nothing on disk changes until the rebuild has passed
    # every check the final validation will apply.  A partial marker — a
    # chain whose target has not certified yet — is simply rebuilt over.
    previous_manifest_verified = False
    if previous_manifest.exists():
        try:
            previous_manifest_value = _json.loads(previous_manifest.read_text(encoding="utf-8"))
            if isinstance(previous_manifest_value, Mapping):
                previous_manifest_status = previous_manifest_value.get("status")
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read existing artifact manifest: {exc}") from exc
        if previous_manifest_status == "complete":
            try:
                validate_result_bundle(run_path, require_complete=True)
                previous_manifest_verified = True
            except ResultBundleError as exc:
                # A complete marker that no longer verifies may be rebuilt
                # over ONLY when every file it disagrees with is one this
                # function regenerates anyway (older saves wrote the marker
                # before their final validation, and a derived artifact may
                # have been damaged since; or the audit itself has grown a
                # rule, which the preflight below re-applies).  A certified
                # artifact that changed after publication is exactly what
                # the marker exists to catch: rebuilding would re-certify it
                # under new hashes, so that fails as loudly as the marker's
                # own verification does.
                disagreements = manifest_disagreements(run_path, previous_manifest_value)
                changed = sorted(set(disagreements) - _REGENERATED_ARTIFACTS)
                if changed:
                    raise ResultBundleError(
                        "completed result bundle is immutable, but certified artifact(s) changed after "
                        f"publication: {changed}; {exc}"
                    ) from exc
                logger.warning(
                    "existing complete artifact manifest in %s no longer verifies (%s); only regenerated "
                    "artifacts disagree with it (%s), so the bundle is rebuilt over it once the rebuild has "
                    "passed every check",
                    run_path,
                    exc,
                    disagreements or "none",
                )

    # From here on, stage results travel in manifest position order so the
    # CSV rows (and any other order-sensitive artifact) come out identical
    # no matter how the caller ordered its list.
    ordered_stage_results = [result for _, result in keyed_results]

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
    # The deliverables map (schema v4): one record per present deliverable
    # whose selected checkpoint exists — every present stage's, once the
    # bundle is publishable.  gate_kind is the verdict's own record, else
    # the resolved config's declaration, else null (unrecorded, like the
    # stage rows).  Phase A replication is this run alone.
    deliverables: dict[str, dict[str, Any]] = {}
    for entry, result in keyed_results:
        checkpoint = selected_checkpoints.get(entry.key)
        if not entry.deliverable or checkpoint is None:
            continue
        gate_kind = result.get("gate_kind")
        if gate_kind is None:
            gate_kind = resolved_stage_configs[entry.reference]["curriculum_kwargs"].get("gate_kind")
        deliverables[entry.key] = {
            "model_path": checkpoint["model_path"],
            "model_hash": checkpoint["model_hash"],
            "normalization_hash": checkpoint["normalization_hash"],
            "gate_kind": gate_kind if isinstance(gate_kind, str) and gate_kind.strip() else None,
            "certified": certified[entry.key],
            "replication": {"count": 1, "runs": [{"run_id": str(captured["run_id"]), "training_seed": seed}]},
        }
    # The published model is the PRIMARY deliverable's checkpoint: the target
    # when certified, else the deepest certified deliverable — never a failed
    # leaf, never a checkpoint of an uncertified node.
    try:
        primary_key = primary_deliverable_key(deliverables, species=species, target=target_entry.key)
    except ResultSchemaError as exc:
        raise ResultBundleError(str(exc)) from exc
    primary_checkpoint = selected_checkpoints.get(primary_key, {}) if primary_key is not None else {}
    selected_model_path = primary_checkpoint.get("model_path")
    model_hash = primary_checkpoint.get("model_hash")
    finalization = {
        "model_hash": model_hash,
        "config_hash": config_hash,
        "backend_version": detected_backend_version,
        "selected_model_path": selected_model_path,
        "selected_checkpoints": selected_checkpoints,
        "deliverables": deliverables,
        "primary_deliverable": primary_key,
        "target_deliverable": target_entry.key,
        "ancestors": project_ancestor_records(ancestor_records),
    }
    finalized_provenance = {**captured, **finalization}
    result_date = str(captured.get("captured_at", "")).split("T", maxsplit=1)[0]

    prospective_summary: dict[str, Any] | None = None
    if publishable:
        prospective_summary = summaries.build_result_summary(
            ordered_stage_results,
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
        if prospective_summary["bundle_status"] != status:
            raise ResultBundleError(
                f"bundle status {status!r} disagrees with the summary's {prospective_summary['bundle_status']!r}"
            )
        try:
            validate_result_summary(
                prospective_summary,
                expected_species=species,
                require_complete=status == "complete",
                require_publishable=True,
                require_canonical_provenance=True,
                result_path=str(summary_path),
            )
        except ResultSchemaError as exc:
            raise ResultBundleError(str(exc)) from exc
        validate_evaluation_evidence(run_path, prospective_summary, finalized_provenance)
        # The audit's one remaining rule that nothing above has applied: a
        # recorded load lineage must agree with the hash of the parent it
        # names when that parent lives in this bundle — or, for a parent
        # reused from another run, with the ancestor record that carries it.
        # Hashed from disk here, which is what the manifest below will
        # declare.
        for stage, run_block in run_blocks.items():
            if not isinstance(run_block, Mapping):
                continue
            parent_hashes: dict[str, str] = {}
            load_path = run_block.get("load_path")
            if isinstance(load_path, str) and load_path.strip():
                for parent_key in _lineage_parent_keys(load_path, run_path):
                    parent_file = run_path / parent_key
                    if parent_file.is_file():
                        parent_hashes[parent_key] = sha256_file(parent_file)
            _, lineage_problems = _audit_load_lineage(
                run_block,
                stage=stage,
                run_path=run_path.resolve(),
                declared_hashes=parent_hashes,
                ancestor_records=ancestor_records,
            )
            if lineage_problems:
                raise ResultBundleError("; ".join(lineage_problems))
        if previous_manifest_verified:
            existing_summary = _json.loads(summary_path.read_text(encoding="utf-8"))
            if existing_summary != prospective_summary:
                raise ResultBundleError("completed result bundle is immutable; use a new run_id for different results")
            return {
                "provenance": provenance_path,
                "collected_results_csv": run_path / "collected_results.csv",
                "summary": summary_path,
                "artifact_manifest": previous_manifest,
            }

    # Exercise the CSV — and, for a publishable bundle, its agreement with the
    # prospective summary — in scratch, so the run directory is untouched
    # until every check the final validation applies has passed.
    with tempfile.TemporaryDirectory(prefix="result-bundle-preflight-") as scratch:
        prospective_csv = csv_output.save_results_csv(
            ordered_stage_results,
            resolved_stage_configs,
            species,
            algorithm,
            seed,
            scratch,
            backend=public_backend,
            run_id=str(captured["run_id"]),
            provenance=finalized_provenance,
        )
        if prospective_summary is not None:
            contradictions = compare_summary_to_csv(prospective_summary, prospective_csv)
            if contradictions:
                raise ResultBundleError("summary/CSV contradictions: " + "; ".join(contradictions))

    # All semantic validation is complete.  The derived artifacts are
    # rewritten in place; the previous marker — complete or not — is never
    # unlinked, only overwritten atomically by the validated replacement at
    # the end, so no reader ever finds the bundle marker-less, and a rewrite
    # interrupted here leaves a marker that disagrees only on regenerated
    # artifacts (which the next re-entry is allowed to rebuild over).
    update_provenance(run_path, **finalization)
    captured = load_provenance(run_path)
    hashing._write_json(run_path / "plant_identity.json", normalized_plant)

    csv_path = csv_output.save_results_csv(
        ordered_stage_results,
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

    if publishable:
        assert prospective_summary is not None
        hashing._write_json(summary_path, prospective_summary)
        contradictions = compare_summary_to_csv(prospective_summary, csv_path)
        if contradictions:
            raise ResultBundleError("summary/CSV contradictions: " + "; ".join(contradictions))
        paths["summary"] = summary_path

    # Validate against the manifest before it exists on disk: a completion
    # marker is only ever written for a bundle that has already passed, and
    # the previous marker is replaced only by one that has.
    manifest = build_artifact_manifest(run_path, status=status)
    validate_result_bundle(
        run_path,
        require_complete=status == "complete",
        require_publishable=publishable,
        prospective_manifest=manifest,
    )
    paths["artifact_manifest"] = hashing._write_json(previous_manifest, manifest)
    return paths
