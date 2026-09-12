"""Shared validation for curated and in-progress training result summaries.

The public species catalog and the training exporters both consume this
result-summary schema.  Keeping the contract here prevents the writer and
reader from silently drifting apart.

Schema v3 (2026-08-23, stage-manifest migration's final part): stage keys in
``stages`` and ``provenance.selected_checkpoints`` are stage REFERENCES
resolved through the species' stage manifest — the decimal string of a legacy
number (``"1"``/``"2"``/``"3"``, their historical meaning) or a semantic
stage id (``"recovery"``), so the recovery stage joins result bundles.  The
v2 completeness rule "exactly stages 1, 2, and 3" meant "a complete
curriculum recorded every advancing stage"; v3 preserves that meaning
exactly — every advancing stage is still required — while non-advancing
semantic stages are optional.  v2 summaries are a strict subset of v3, so
every committed historical artifact validates unchanged.

Schema v4 (2026-09-06, BEHAVIOR_RECIPES_PLAN §4.3, Phase A): publication is
per DELIVERABLE, not per advancing trio.  ``provenance.deliverables`` maps
each deliverable stage key the run recorded to ``{model_path, model_hash,
normalization_hash, gate_kind, certified, replication}``;
``provenance.ancestors`` mirrors the on-disk ``ancestors/<stage_id>/``
records of nodes reused from another run; ``primary_deliverable`` and
``target_deliverable`` name the published model and the node the run aimed
at.  A deliverable is *certified* when its own gate passed and every
transitive ``warm_start_from`` ancestor is present with a passed gate — in
``stages`` or as an ancestor record.  The primary deliverable is the target
when it is certified, else the deepest certified deliverable in manifest
order; ``selected_model_path`` / ``model_hash`` are the primary's, so a v3
reader still sees one model.  ``bundle_status`` is ``complete`` when the
target and every present deliverable are certified, ``partial`` when at
least one is, ``failed`` when none is — and a summary exists whenever at
least one is.

The rules are VERSION-GATED: every rule for a schema below 4 runs verbatim
(the four committed ``results/**/summary.json`` are the bit-identity pins),
and a provenance block without a ``deliverables`` key is read under the v3
rules when no version is given.  Under a v1 or synthesized stage manifest
the loader derives exactly one deliverable (the last advancing node) with
edges to the previous advancing node, so the v4 rules collapse to the v3
"every advancing stage present and passed / terminal = last advancing"
rule.  All three versions are accepted here; writers emit
:data:`RESULT_SCHEMA_VERSION`.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, cast

RESULT_SCHEMA_VERSION = 4
#: Versions this reader accepts.  v2 is the integer-stage schema every
#: committed summary under results/ carries; v3 widened the stage-key
#: vocabulary and v4 added per-deliverable certification (module docstring)
#: without changing any earlier-valid artifact.
SUPPORTED_RESULT_SCHEMA_VERSIONS = frozenset({2, 3, RESULT_SCHEMA_VERSION})
ALLOWED_BUNDLE_STATUSES = frozenset({"complete", "partial", "failed"})
#: Exactly the fields of one ``provenance.deliverables`` record (schema v4).
#: ``replication`` is ``{count, runs: [{run_id, training_seed}]}`` — Phase A
#: writes count 1 with the run itself; Phase B's seed replication extends it.
DELIVERABLE_RECORD_FIELDS = (
    "model_path",
    "model_hash",
    "normalization_hash",
    "gate_kind",
    "certified",
    "replication",
)
#: Exactly the fields of one ``provenance.ancestors`` record (schema v4): the
#: summary-side projection of ``ancestors/<stage_id>/`` on disk.
ANCESTOR_RECORD_FIELDS = (
    "run_id",
    "model_hash",
    "normalization_hash",
    "gate_kind",
    "passed",
    "task_sha256",
)
ALLOWED_MODEL_REVISION_STATUSES = frozenset({"current", "historical"})
ALLOWED_VERIFICATION_STATUSES = frozenset({"verified", "unverified"})
ALLOWED_TRAINING_BACKENDS = frozenset({"stable-baselines3", "jax-mjx"})
PROVENANCE_IDENTIFIERS = ("repository_commit", "model_hash", "config_hash")
REQUIRED_PROVENANCE_FIELDS = (
    "model_revision_status",
    "verification_status",
    "evaluation_episodes",
    *PROVENANCE_IDENTIFIERS,
)
CANONICAL_RUNTIME_PROVENANCE_FIELDS = (
    "run_id",
    "captured_at",
    "species",
    "algorithm",
    "backend",
    "backend_version",
    "repository_dirty",
    "repository_patch_sha256",
    "training_seed",
    "seed_roles",
    "evaluation_protocols",
    "evaluation_seeds",
    "parallel_envs",
    "hardware",
    "python_version",
    "platform",
    "dependency_versions",
    "plant_identity",
    "selected_checkpoints",
    "selected_model_path",
)
#: The v4 canonical runtime fields.  Deliberately a NEW tuple rather than an
#: extension of the v3 one: the v3 tuple is what a schema-2/3 summary is
#: validated against, and it must not grow.
CANONICAL_RUNTIME_PROVENANCE_FIELDS_V4 = CANONICAL_RUNTIME_PROVENANCE_FIELDS + (
    "deliverables",
    "primary_deliverable",
    "target_deliverable",
)

_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class ResultSchemaError(ValueError):
    """Raised when a result summary is incomplete or contradictory."""


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultSchemaError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResultSchemaError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_nonempty_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field=field)


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ResultSchemaError(f"{field} must be a positive integer")
    return value


def _optional_number(value: Any, *, field: str) -> int | float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ResultSchemaError(f"{field} must be a number or null")
    if not math.isfinite(float(value)):
        raise ResultSchemaError(f"{field} must be finite or null")
    return value


def _canonical_plant_identity(value: Any, *, species: str, field: str) -> dict[str, Any]:
    value = _require_mapping(value, field=field)
    try:
        from .plant_contract import PlantContractError, PlantIdentity

        identity = PlantIdentity.from_mapping(value)
    except (PlantContractError, KeyError, TypeError, ValueError) as exc:
        raise ResultSchemaError(f"{field} is invalid: {exc}") from exc
    if identity.species != species:
        raise ResultSchemaError(f"{field} species mismatch: expected {species!r}, found {identity.species!r}")
    normalized = identity.to_dict()
    if value != normalized:
        raise ResultSchemaError(f"{field} must use canonical v1 field types and values")
    return normalized


def seed_role_collisions(seed_roles: Mapping[str, Any]) -> list[str]:
    """Describe seed-role assignments that would bias the published numbers.

    The publication evaluation must run on a seed that neither training nor
    any checkpoint-*selection* role saw: evaluating on the training seed
    replays the reset sequence the policy fitted, and reusing a selection
    seed publishes the maximum of noisy draws — the winner's curse the
    ``seed_roles`` record exists to rule out.  Returns one message per
    collision, naming the colliding roles; empty when the roles are distinct
    or no publication role is assigned.
    """
    publication_seed = seed_roles.get("publication_evaluation")
    if publication_seed is None:
        return []
    collisions: list[str] = []
    if seed_roles.get("training") == publication_seed:
        collisions.append(f"publication_evaluation reuses the training seed {publication_seed}")
    for role in sorted(seed_roles):
        if "selection" in role and seed_roles[role] == publication_seed:
            collisions.append(f"publication_evaluation reuses the {role} seed {publication_seed}")
    return collisions


def _algorithm_slug(algorithm: str) -> str:
    """Return the storage slug for an algorithm label.

    JAX is a backend, not a distinct reinforcement-learning algorithm.  Legacy
    labels such as ``JAX_PPO`` are nevertheless normalized so older Drive
    artifacts remain addressable.
    """

    slug = re.sub(r"[^a-z0-9]+", "_", algorithm.lower()).strip("_")
    for prefix in ("jax_mjx_", "jax_"):
        if slug.startswith(prefix):
            slug = slug.removeprefix(prefix)
            break
    if not slug:
        raise ResultSchemaError("algorithm must contain at least one letter or number")
    return slug


def ordered_stage_entries(stage_keys: Any, *, species: str, field: str) -> "list[tuple[str, Any]]":
    """Resolve serialized stage keys against the species' manifest, in order.

    Returns ``(key, StageEntry)`` pairs sorted by manifest position — never
    by ``int()`` on the key, which would crash on semantic ids and, worse,
    silently read positions into legacy numbers.  Fail-closed: an unknown
    reference, a non-string key, and two spellings of the same stage
    (``"2"`` beside ``"locomotion"``) are all fatal, as is a species whose
    manifest cannot be loaded — "we could not resolve it" must never read
    as "it is a stage".
    """
    from .stage_manifest import StageManifestError, load_stage_manifest

    try:
        manifest = load_stage_manifest(species)
    except StageManifestError as exc:
        raise ResultSchemaError(f"{field}: cannot load the stage manifest for species {species!r}: {exc}") from exc

    entries: list[tuple[str, Any]] = []
    spelling_by_id: dict[str, str] = {}
    for stage_key in stage_keys:
        if not isinstance(stage_key, str) or not stage_key.strip():
            raise ResultSchemaError(f"{field} keys must be non-empty strings: {stage_key!r}")
        try:
            # The serialized-key rule of stage_manifest.resolve_stage_key,
            # against the manifest loaded once above: decimal spellings are
            # legacy numbers, everything else is a semantic id.
            entry = manifest.by_legacy_number(int(stage_key)) if stage_key.isdigit() else manifest.by_id(stage_key)
        except StageManifestError as exc:
            raise ResultSchemaError(f"{field} contains an invalid stage reference {stage_key!r}: {exc}") from exc
        if entry.id in spelling_by_id:
            raise ResultSchemaError(
                f"{field} references stage {entry.id!r} twice (as {spelling_by_id[entry.id]!r} and {stage_key!r})"
            )
        spelling_by_id[entry.id] = stage_key
        entries.append((stage_key, entry))
    entries.sort(key=lambda pair: pair[1].position)
    return entries


def _missing_advancing_stages(present_entries: "list[tuple[str, Any]]", *, species: str) -> "list[Any]":
    """Advancing stages the species declares that *present_entries* omit."""
    from .stage_manifest import load_stage_manifest

    present_ids = {entry.id for _, entry in present_entries}
    return [entry for entry in load_stage_manifest(species).advancing_stages if entry.id not in present_ids]


def _load_manifest_for(species: str, *, field: str) -> Any:
    """The species' stage manifest, with a load failure reported as a schema error."""
    from .stage_manifest import StageManifestError, load_stage_manifest

    try:
        return load_stage_manifest(species)
    except StageManifestError as exc:
        raise ResultSchemaError(f"{field}: cannot load the stage manifest for species {species!r}: {exc}") from exc


def _resolve_stage_ref(ref: Any, manifest: Any, *, field: str) -> Any:
    """Resolve an in-memory or serialized stage reference against *manifest*."""
    from .stage_manifest import StageManifestError

    if isinstance(ref, bool) or not isinstance(ref, (int, str)) or (isinstance(ref, str) and not ref.strip()):
        raise ResultSchemaError(f"{field} must be a stage reference, not {ref!r}")
    try:
        if isinstance(ref, str) and ref.isdigit():
            return manifest.by_legacy_number(int(ref))
        return manifest.resolve(ref)
    except StageManifestError as exc:
        raise ResultSchemaError(f"{field} is not a stage of {manifest.species!r}: {exc}") from exc


def _recorded_verdict(value: Any, *, key: str, field: str) -> bool:
    """A bool, or a stage-summary / deliverable-record mapping's verdict, or fail closed."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        for verdict_key in ("stage_passed", "certified", "passed"):
            if verdict_key in value:
                verdict = value[verdict_key]
                if isinstance(verdict, bool):
                    return verdict
                raise ResultSchemaError(f"{field}.{key}.{verdict_key} must be a boolean")
    raise ResultSchemaError(f"{field}.{key} must be a boolean verdict or a record carrying one")


def deliverable_chain(entry: Any, manifest: Any) -> "tuple[Any, ...]":
    """*entry*'s transitive ``warm_start_from`` ancestors, root first (manifest order).

    The loader only accepts edges to earlier entries, so the chain is finite
    and its order is the manifest's.  Under a v1 or synthesized manifest the
    derived edge is "the previous advancing entry", so the chain of the last
    advancing node is exactly ``advancing_stages[:-1]``.
    """
    return tuple(manifest.ancestors(entry.id))


def certified_deliverables(
    stage_entries: "list[tuple[str, Any]]",
    stages: Mapping[str, Any],
    ancestors: Mapping[str, Any] | None,
    *,
    species: str,
) -> dict[str, bool]:
    """Which recorded deliverables are certified, keyed as *stage_entries* spells them.

    One entry per manifest deliverable present in *stage_entries* (the
    ``(key, StageEntry)`` pairs :func:`ordered_stage_entries` returns).  A
    deliverable is certified iff its own recorded verdict is True and every
    chain ancestor is present in *stages* with a True verdict or in
    *ancestors* with ``passed`` True.  An ancestor in neither is absent, and
    absent never reads as passed.  *stages* and *ancestors* map stage keys
    to a bool or to a record carrying ``stage_passed`` / ``certified`` /
    ``passed``; a stage in both is a contradiction (trained here AND reused)
    and fails closed, as does any key outside the species' vocabulary.
    """
    manifest = _load_manifest_for(species, field="stages")
    passed_by_id: dict[str, bool] = {}
    for key, entry in stage_entries:
        if key not in stages:
            raise ResultSchemaError(f"stages is missing recorded stage {key!r}")
        passed_by_id[entry.id] = _recorded_verdict(stages[key], key=key, field="stages")
    ancestor_passed_by_id: dict[str, bool] = {}
    if ancestors:
        for key, entry in ordered_stage_entries(ancestors, species=species, field="ancestors"):
            if entry.id in passed_by_id:
                raise ResultSchemaError(
                    f"stage {entry.key!r} is recorded both as a trained stage and as a reused ancestor"
                )
            ancestor_passed_by_id[entry.id] = _recorded_verdict(ancestors[key], key=key, field="ancestors")
    certified: dict[str, bool] = {}
    for key, entry in stage_entries:
        if not entry.deliverable:
            continue
        verdicts = [passed_by_id[entry.id]]
        for ancestor in deliverable_chain(entry, manifest):
            if ancestor.id in passed_by_id:
                verdicts.append(passed_by_id[ancestor.id])
            elif ancestor.id in ancestor_passed_by_id:
                verdicts.append(ancestor_passed_by_id[ancestor.id])
            else:
                verdicts.append(False)
        certified[key] = all(verdicts)
    return certified


def uncertified_chain_members(
    stage_key: str,
    stage_entries: "list[tuple[str, Any]]",
    stages: Mapping[str, Any],
    ancestors: Mapping[str, Any] | None,
    *,
    species: str,
) -> list[str]:
    """Why *stage_key* is not certified: each chain member that is absent or failed, by name."""
    manifest = _load_manifest_for(species, field="stages")
    entry = next((entry for key, entry in stage_entries if key == stage_key), None)
    if entry is None:
        raise ResultSchemaError(f"stages does not record {stage_key!r}")
    passed_by_id = {entry.id: _recorded_verdict(stages[key], key=key, field="stages") for key, entry in stage_entries}
    ancestor_passed_by_id: dict[str, bool] = {}
    if ancestors:
        for key, ancestor_entry in ordered_stage_entries(ancestors, species=species, field="ancestors"):
            ancestor_passed_by_id[ancestor_entry.id] = _recorded_verdict(ancestors[key], key=key, field="ancestors")
    reasons: list[str] = []
    for member in (*deliverable_chain(entry, manifest), entry):
        if member.id in passed_by_id:
            if not passed_by_id[member.id]:
                reasons.append(f"stage {member.key} failed its gate")
        elif member.id in ancestor_passed_by_id:
            if not ancestor_passed_by_id[member.id]:
                reasons.append(f"ancestor record {member.key} did not pass")
        else:
            reasons.append(f"ancestor {member.key} is absent (neither recorded in stages nor as an ancestor record)")
    return reasons


def primary_deliverable_key(
    deliverables: Mapping[str, Any],
    *,
    species: str,
    target: "int | str | None" = None,
) -> str | None:
    """The key of the published model: the target when certified, else the deepest certified.

    *deliverables* maps stage keys to a bool or to a record carrying
    ``certified``.  Manifest order is topological, so the LAST certified
    deliverable is the deepest along its chain.  ``None`` when nothing is
    certified — a bundle with no primary has no ``summary.json``.
    """
    manifest = _load_manifest_for(species, field="provenance.deliverables")
    entries = ordered_stage_entries(deliverables, species=species, field="provenance.deliverables")
    certified = {
        key: _recorded_verdict(deliverables[key], key=key, field="provenance.deliverables") for key, _ in entries
    }
    if target is not None:
        target_entry = _resolve_stage_ref(target, manifest, field="target_deliverable")
        for key, entry in entries:
            if entry.id == target_entry.id and certified[key]:
                return key
    for key, _ in reversed(entries):
        if certified[key]:
            return key
    return None


def bundle_status_for(
    deliverables: Mapping[str, Any],
    *,
    species: str,
    target: "int | str | None",
    stages: Mapping[str, Any] | None = None,
) -> str:
    """``complete`` / ``partial`` / ``failed`` for a run (decision D-A1, target-aware).

    ``complete`` iff *target* is present in *deliverables* and certified AND
    every present deliverable is certified; ``partial`` iff at least one is
    certified; ``failed`` iff none is.  *stages* — the verdicts of every
    stage the run recorded (a bool or a record with ``stage_passed`` per
    key) — refines ``failed``: a run in which nothing is certified because
    no DELIVERABLE is present yet, while every recorded stage passed, is
    still in progress and reads ``partial``.  That is the pre-Phase-A rule
    for a v1 / synthesized manifest, whose only deliverable is the last
    advancing node (a passing stance-only run was ``partial`` without a
    summary).  Under the committed v2 manifests every node is a
    deliverable, so the refinement never changes a status.
    """
    manifest = _load_manifest_for(species, field="provenance.deliverables")
    entries = ordered_stage_entries(deliverables, species=species, field="provenance.deliverables")
    certified = {
        key: _recorded_verdict(deliverables[key], key=key, field="provenance.deliverables") for key, _ in entries
    }
    if certified and all(certified.values()) and target is not None:
        target_entry = _resolve_stage_ref(target, manifest, field="target_deliverable")
        if any(entry.id == target_entry.id for _, entry in entries):
            return "complete"
    if any(certified.values()):
        return "partial"
    if stages is not None and not certified:
        stage_entries = ordered_stage_entries(stages, species=species, field="stages")
        verdicts = [_recorded_verdict(stages[key], key=key, field="stages") for key, _ in stage_entries]
        if verdicts and all(verdicts):
            return "partial"
    return "failed"


def expected_result_directory(algorithm: str, backend: str) -> str:
    """Return the backend-aware directory name for a curated result."""

    if backend not in ALLOWED_TRAINING_BACKENDS:
        raise ResultSchemaError(f"invalid backend: {backend}")
    algorithm_slug = _algorithm_slug(algorithm)
    return algorithm_slug if backend == "stable-baselines3" else f"jax_{algorithm_slug}"


def validate_result_path(
    relative_path: str | Path,
    *,
    species: str,
    algorithm: str,
    backend: str,
) -> None:
    """Validate the conventional ``results/<species>/<run>/summary.json`` path."""

    path = Path(relative_path)
    if path.name != "summary.json" or len(path.parents) < 2:
        raise ResultSchemaError(f"result path must end in a species result summary: {relative_path}")
    path_species = path.parents[1].name
    if path_species != species:
        raise ResultSchemaError(
            f"result species/path mismatch in {relative_path}: expected {species}, found {path_species}"
        )
    expected_directory = expected_result_directory(algorithm, backend)
    actual_directory = path.parent.name
    if actual_directory != expected_directory:
        raise ResultSchemaError(
            f"result directory mismatch in {relative_path}: "
            f"expected {expected_directory} for {backend}, found {actual_directory}"
        )


def _require_relative_posix_path(value: Any, *, field: str) -> str:
    text = _require_nonempty_string(value, field=field)
    portable = PurePosixPath(text)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise ResultSchemaError(f"{field} must be a normalized relative POSIX path")
    return text


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ResultSchemaError(f"{field} must be sha256:<64 lowercase hex>")
    return value


def _validate_deliverable_records(
    deliverables_value: Any,
    *,
    species: str,
    backend: str,
    result_path: str,
) -> "tuple[list[tuple[str, Any]], dict[str, dict[str, Any]]]":
    """Shape-check ``provenance.deliverables`` (v4): keys, record fields, replication."""
    field = f"provenance.deliverables in {result_path}"
    deliverables_value = _require_mapping(deliverables_value, field=field)
    entries = ordered_stage_entries(deliverables_value, species=species, field=field)
    records: dict[str, dict[str, Any]] = {}
    for key, entry in entries:
        if not entry.deliverable:
            raise ResultSchemaError(
                f"{field} names stage {key!r}, which the {species} manifest does not flag as a deliverable"
            )
        record = _require_mapping(deliverables_value[key], field=f"{field}.{key}")
        if set(record) != set(DELIVERABLE_RECORD_FIELDS):
            raise ResultSchemaError(
                f"{field}.{key} must carry exactly the fields {list(DELIVERABLE_RECORD_FIELDS)}; found {sorted(record)}"
            )
        prefix = f"{field}.{key}"
        model_path = _require_relative_posix_path(record["model_path"], field=f"{prefix}.model_path")
        model_hash = _require_sha256(record["model_hash"], field=f"{prefix}.model_hash")
        normalization_hash = record["normalization_hash"]
        if backend == "stable-baselines3":
            normalization_hash = _require_sha256(normalization_hash, field=f"{prefix}.normalization_hash")
        elif normalization_hash is not None:
            raise ResultSchemaError(f"JAX {prefix}.normalization_hash must be null")
        gate_kind = _optional_nonempty_string(record["gate_kind"], field=f"{prefix}.gate_kind")
        if not isinstance(record["certified"], bool):
            raise ResultSchemaError(f"{prefix}.certified must be a boolean")
        replication = _require_mapping(record["replication"], field=f"{prefix}.replication")
        if set(replication) != {"count", "runs"}:
            raise ResultSchemaError(f"{prefix}.replication must carry exactly count and runs")
        count = _require_positive_int(replication["count"], field=f"{prefix}.replication.count")
        runs = replication["runs"]
        if not isinstance(runs, list) or len(runs) != count:
            raise ResultSchemaError(f"{prefix}.replication.runs must be a list of {count} run records")
        normalized_runs: list[dict[str, Any]] = []
        for index, run_value in enumerate(runs):
            run = _require_mapping(run_value, field=f"{prefix}.replication.runs[{index}]")
            if set(run) != {"run_id", "training_seed"}:
                raise ResultSchemaError(
                    f"{prefix}.replication.runs[{index}] must carry exactly run_id and training_seed"
                )
            run_id = _require_nonempty_string(run["run_id"], field=f"{prefix}.replication.runs[{index}].run_id")
            training_seed = run["training_seed"]
            if not isinstance(training_seed, int) or isinstance(training_seed, bool) or training_seed < 0:
                raise ResultSchemaError(
                    f"{prefix}.replication.runs[{index}].training_seed must be a non-negative integer"
                )
            normalized_runs.append({"run_id": run_id, "training_seed": training_seed})
        records[key] = {
            "model_path": model_path,
            "model_hash": model_hash,
            "normalization_hash": normalization_hash,
            "gate_kind": gate_kind,
            "certified": record["certified"],
            "replication": {"count": count, "runs": normalized_runs},
        }
    return entries, records


def _validate_ancestor_records(
    ancestors_value: Any,
    *,
    species: str,
    result_path: str,
) -> "tuple[list[tuple[str, Any]], dict[str, dict[str, Any]]]":
    """Shape-check ``provenance.ancestors`` (v4)."""
    field = f"provenance.ancestors in {result_path}"
    ancestors_value = _require_mapping(ancestors_value, field=field)
    entries = ordered_stage_entries(ancestors_value, species=species, field=field)
    records: dict[str, dict[str, Any]] = {}
    for key, _entry in entries:
        record = _require_mapping(ancestors_value[key], field=f"{field}.{key}")
        if set(record) != set(ANCESTOR_RECORD_FIELDS):
            raise ResultSchemaError(
                f"{field}.{key} must carry exactly the fields {list(ANCESTOR_RECORD_FIELDS)}; found {sorted(record)}"
            )
        prefix = f"{field}.{key}"
        normalization_hash = record["normalization_hash"]
        if normalization_hash is not None:
            normalization_hash = _require_sha256(normalization_hash, field=f"{prefix}.normalization_hash")
        if not isinstance(record["passed"], bool):
            raise ResultSchemaError(f"{prefix}.passed must be a boolean")
        records[key] = {
            "run_id": _require_nonempty_string(record["run_id"], field=f"{prefix}.run_id"),
            "model_hash": _require_sha256(record["model_hash"], field=f"{prefix}.model_hash"),
            "normalization_hash": normalization_hash,
            "gate_kind": _optional_nonempty_string(record["gate_kind"], field=f"{prefix}.gate_kind"),
            "passed": record["passed"],
            "task_sha256": _require_sha256(record["task_sha256"], field=f"{prefix}.task_sha256"),
        }
    return entries, records


def _optional_stage_key(value: Any, *, species: str, field: str) -> str | None:
    """A serialized stage key of the species, or null."""
    if value is None:
        return None
    key = _require_nonempty_string(value, field=field)
    ordered_stage_entries([key], species=species, field=field)
    return key


def validate_provenance(
    provenance: Any,
    *,
    result_path: str = "result summary",
    canonical: bool = False,
    schema_version: int | None = None,
) -> dict[str, Any]:
    """Validate result-summary provenance (schema v2/v3/v4).

    Historical summaries may retain unknown identifiers as ``null``.  Canonical
    mode is for newly exported bundles and requires complete, well-formed
    identity fields regardless of the result's current/historical label.

    *schema_version* selects the rule set: below 4 the advancing-trio /
    terminal-key rules run verbatim; from 4 on the deliverable rules
    (module docstring) replace them.  ``None`` infers it from the block —
    v4 iff it carries a ``deliverables`` key — so a reader of the summary's
    provenance alone (the species catalog) needs no version in hand and
    every schema-2/3 artifact keeps its rules.
    """

    provenance = _require_mapping(provenance, field=f"provenance in {result_path}")
    v4 = schema_version >= 4 if schema_version is not None else "deliverables" in provenance
    missing_fields = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in provenance]
    if missing_fields:
        raise ResultSchemaError(f"provenance in {result_path} is missing fields: {missing_fields}")

    model_revision_status = provenance["model_revision_status"]
    verification_status = provenance["verification_status"]
    if model_revision_status not in ALLOWED_MODEL_REVISION_STATUSES:
        raise ResultSchemaError(f"invalid model_revision_status in {result_path}: {model_revision_status}")
    if verification_status not in ALLOWED_VERIFICATION_STATUSES:
        raise ResultSchemaError(f"invalid verification_status in {result_path}: {verification_status}")

    evaluation_episodes = provenance["evaluation_episodes"]
    if evaluation_episodes is not None and (
        not isinstance(evaluation_episodes, int) or isinstance(evaluation_episodes, bool) or evaluation_episodes <= 0
    ):
        raise ResultSchemaError(f"evaluation_episodes must be null or a positive integer in {result_path}")

    identifiers = {
        key: _optional_nonempty_string(provenance[key], field=f"provenance.{key} in {result_path}")
        for key in PROVENANCE_IDENTIFIERS
    }
    claims_certified = model_revision_status == "current" or verification_status == "verified"
    if canonical or claims_certified:
        missing_identifiers = [key for key, value in identifiers.items() if value is None]
        if evaluation_episodes is None:
            missing_identifiers.append("evaluation_episodes")
        if missing_identifiers:
            qualifier = "canonical" if canonical else "current or verified"
            raise ResultSchemaError(f"{qualifier} result {result_path} is missing identifiers: {missing_identifiers}")

    # Optional, additive (no schema bump): one entry per process that touched
    # the run — started_at for the creating session, resumed_at for resumes.
    # Older bundles without the field remain valid.
    sessions = provenance.get("sessions")
    if sessions is not None:
        if not isinstance(sessions, list):
            raise ResultSchemaError(f"provenance.sessions in {result_path} must be a list when present")
        for index, session_value in enumerate(sessions):
            session = _require_mapping(session_value, field=f"provenance.sessions[{index}] in {result_path}")
            _require_nonempty_string(
                session.get("session_token"),
                field=f"provenance.sessions[{index}].session_token in {result_path}",
            )
            timestamps = [key for key in ("started_at", "resumed_at") if key in session]
            if len(timestamps) != 1:
                raise ResultSchemaError(
                    f"provenance.sessions[{index}] in {result_path} must record exactly one of started_at or resumed_at"
                )
            _require_nonempty_string(
                session.get(timestamps[0]),
                field=f"provenance.sessions[{index}].{timestamps[0]} in {result_path}",
            )

    canonical_runtime: dict[str, Any] = {}
    if canonical:
        repository_commit = identifiers["repository_commit"]
        if repository_commit is None or _GIT_COMMIT_PATTERN.fullmatch(repository_commit) is None:
            raise ResultSchemaError(
                f"provenance.repository_commit in {result_path} must be a 40-character lowercase Git SHA"
            )
        for key in ("model_hash", "config_hash"):
            value = identifiers[key]
            if value is None or _SHA256_PATTERN.fullmatch(value) is None:
                raise ResultSchemaError(f"provenance.{key} in {result_path} must be sha256:<64 lowercase hex>")

        runtime_fields = CANONICAL_RUNTIME_PROVENANCE_FIELDS_V4 if v4 else CANONICAL_RUNTIME_PROVENANCE_FIELDS
        missing_runtime_fields = [field for field in runtime_fields if field not in provenance]
        if missing_runtime_fields:
            raise ResultSchemaError(
                f"canonical provenance in {result_path} is missing fields: {missing_runtime_fields}"
            )

        run_id = _require_nonempty_string(provenance["run_id"], field=f"provenance.run_id in {result_path}")
        repository_dirty = provenance["repository_dirty"]
        if not isinstance(repository_dirty, bool):
            raise ResultSchemaError(f"provenance.repository_dirty in {result_path} must be a boolean")
        if repository_dirty:
            raise ResultSchemaError(
                f"canonical promotion requires a clean repository in {result_path}; "
                "start a new run from a committed revision"
            )
        repository_patch_sha256 = provenance["repository_patch_sha256"]
        if repository_dirty and repository_patch_sha256 is None:
            raise ResultSchemaError(
                f"provenance.repository_patch_sha256 in {result_path} is required for a dirty repository"
            )
        if not repository_dirty and repository_patch_sha256 is not None:
            raise ResultSchemaError(
                f"provenance.repository_patch_sha256 in {result_path} must be null for a clean repository"
            )
        if repository_patch_sha256 is not None and (
            not isinstance(repository_patch_sha256, str) or _SHA256_PATTERN.fullmatch(repository_patch_sha256) is None
        ):
            raise ResultSchemaError(
                f"provenance.repository_patch_sha256 in {result_path} must be null or sha256:<64 lowercase hex>"
            )

        captured_at = _require_nonempty_string(
            provenance["captured_at"],
            field=f"provenance.captured_at in {result_path}",
        )
        try:
            captured_datetime = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ResultSchemaError(f"provenance.captured_at in {result_path} must be an ISO-8601 timestamp") from exc
        if captured_datetime.tzinfo is None or captured_datetime.utcoffset() is None:
            raise ResultSchemaError(f"provenance.captured_at in {result_path} must include a UTC offset")

        training_seed = provenance["training_seed"]
        if not isinstance(training_seed, int) or isinstance(training_seed, bool) or training_seed < 0:
            raise ResultSchemaError(f"provenance.training_seed in {result_path} must be a non-negative integer")

        evaluation_seeds = provenance["evaluation_seeds"]
        if not isinstance(evaluation_seeds, list) or not evaluation_seeds:
            raise ResultSchemaError(f"provenance.evaluation_seeds in {result_path} must be a non-empty list")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in evaluation_seeds):
            raise ResultSchemaError(
                f"provenance.evaluation_seeds in {result_path} must contain only non-negative integers"
            )
        if len(evaluation_seeds) != len(set(evaluation_seeds)):
            raise ResultSchemaError(f"provenance.evaluation_seeds in {result_path} must contain unique seeds")

        seed_roles = _require_mapping(
            provenance["seed_roles"],
            field=f"provenance.seed_roles in {result_path}",
        )
        if not seed_roles:
            raise ResultSchemaError(f"provenance.seed_roles in {result_path} must not be empty")
        if any(not isinstance(role, str) or not role.strip() for role in seed_roles):
            raise ResultSchemaError(f"provenance.seed_roles in {result_path} must use non-empty string role names")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seed_roles.values()):
            raise ResultSchemaError(f"provenance.seed_roles in {result_path} must contain non-negative integer seeds")
        if seed_roles.get("training") != training_seed:
            raise ResultSchemaError(f"provenance.seed_roles.training in {result_path} must match training_seed")
        publication_evaluation_seed = seed_roles.get("publication_evaluation")
        if publication_evaluation_seed not in evaluation_seeds:
            raise ResultSchemaError(
                f"provenance.seed_roles.publication_evaluation in {result_path} must be listed in evaluation_seeds"
            )
        evaluation_role_seeds = {seed for role, seed in seed_roles.items() if "evaluation" in role}
        if set(evaluation_seeds) != evaluation_role_seeds:
            raise ResultSchemaError(
                f"provenance.evaluation_seeds in {result_path} must exactly match seeds assigned to evaluation roles"
            )
        collisions = seed_role_collisions(seed_roles)
        if collisions:
            raise ResultSchemaError(
                f"provenance.seed_roles in {result_path} must keep the publication seed distinct: "
                + "; ".join(collisions)
            )
        evaluation_protocols = _require_mapping(
            provenance["evaluation_protocols"],
            field=f"provenance.evaluation_protocols in {result_path}",
        )
        evaluation_roles = {role for role in seed_roles if "evaluation" in role}
        if set(evaluation_protocols) != evaluation_roles:
            raise ResultSchemaError(
                f"provenance.evaluation_protocols in {result_path} must exactly match evaluation seed roles"
            )
        for role, protocol_value in evaluation_protocols.items():
            protocol = _require_mapping(
                protocol_value,
                field=f"provenance.evaluation_protocols.{role} in {result_path}",
            )
            if protocol.get("seed") != seed_roles[role]:
                raise ResultSchemaError(
                    f"provenance.evaluation_protocols.{role}.seed in {result_path} must match the seed role"
                )
            if protocol.get("episodes") != evaluation_episodes:
                raise ResultSchemaError(
                    f"provenance.evaluation_protocols.{role}.episodes in {result_path} must match evaluation_episodes"
                )
            if protocol.get("deterministic") is not True:
                raise ResultSchemaError(
                    f"provenance.evaluation_protocols.{role}.deterministic in {result_path} must be true"
                )

        species = _require_nonempty_string(
            provenance["species"],
            field=f"provenance.species in {result_path}",
        )
        algorithm = _require_nonempty_string(
            provenance["algorithm"],
            field=f"provenance.algorithm in {result_path}",
        )
        backend = provenance["backend"]
        if backend not in ALLOWED_TRAINING_BACKENDS:
            raise ResultSchemaError(f"invalid provenance.backend in {result_path}: {backend}")
        provenance_backend_version = _require_nonempty_string(
            provenance["backend_version"],
            field=f"provenance.backend_version in {result_path}",
        )
        parallel_envs = _require_positive_int(
            provenance["parallel_envs"],
            field=f"provenance.parallel_envs in {result_path}",
        )
        hardware = _require_nonempty_string(
            provenance["hardware"],
            field=f"provenance.hardware in {result_path}",
        )
        plant_identity = _canonical_plant_identity(
            provenance["plant_identity"],
            species=species,
            field=f"provenance.plant_identity in {result_path}",
        )
        selected_model_path = _require_nonempty_string(
            provenance["selected_model_path"],
            field=f"provenance.selected_model_path in {result_path}",
        )
        portable_model_path = PurePosixPath(selected_model_path)
        if portable_model_path.is_absolute() or any(part in {"", ".", ".."} for part in portable_model_path.parts):
            raise ResultSchemaError(
                f"provenance.selected_model_path in {result_path} must be a normalized relative POSIX path"
            )
        selected_checkpoints_value = _require_mapping(
            provenance["selected_checkpoints"],
            field=f"provenance.selected_checkpoints in {result_path}",
        )
        checkpoint_entries = ordered_stage_entries(
            selected_checkpoints_value,
            species=species,
            field=f"provenance.selected_checkpoints in {result_path}",
        )
        if not v4:
            # v2 required exactly {"1", "2", "3"}, which enforced "a complete
            # curriculum recorded a handoff per advancing stage".  Preserved
            # exactly for schema < 4: every advancing stage is still
            # required; non-advancing semantic stages (recovery) may add a
            # checkpoint but never replace one, and any key outside the
            # species' vocabulary fails closed.
            missing_advancing = _missing_advancing_stages(checkpoint_entries, species=species)
            if missing_advancing:
                raise ResultSchemaError(
                    f"provenance.selected_checkpoints in {result_path} must record a checkpoint for every "
                    f"advancing stage; missing {[entry.key for entry in missing_advancing]}"
                )
            if not any(entry.legacy_number is not None for _, entry in checkpoint_entries):
                raise ResultSchemaError(
                    f"provenance.selected_checkpoints in {result_path} records no advancing-stage checkpoint"
                )
        selected_checkpoints: dict[str, dict[str, Any]] = {}
        for stage_key, _checkpoint_entry in checkpoint_entries:
            checkpoint = _require_mapping(
                selected_checkpoints_value[stage_key],
                field=f"provenance.selected_checkpoints.{stage_key} in {result_path}",
            )
            checkpoint_model_path = _require_nonempty_string(
                checkpoint.get("model_path"),
                field=f"provenance.selected_checkpoints.{stage_key}.model_path in {result_path}",
            )
            portable_checkpoint_path = PurePosixPath(checkpoint_model_path)
            if portable_checkpoint_path.is_absolute() or any(
                part in {"", ".", ".."} for part in portable_checkpoint_path.parts
            ):
                raise ResultSchemaError(
                    f"provenance.selected_checkpoints.{stage_key}.model_path in {result_path} "
                    "must be a normalized relative POSIX path"
                )
            checkpoint_model_hash = checkpoint.get("model_hash")
            if not isinstance(checkpoint_model_hash, str) or _SHA256_PATTERN.fullmatch(checkpoint_model_hash) is None:
                raise ResultSchemaError(
                    f"provenance.selected_checkpoints.{stage_key}.model_hash in {result_path} "
                    "must be sha256:<64 lowercase hex>"
                )
            normalization_path = checkpoint.get("normalization_path")
            normalization_hash = checkpoint.get("normalization_hash")
            if backend == "stable-baselines3":
                normalization_path = _require_nonempty_string(
                    normalization_path,
                    field=(f"provenance.selected_checkpoints.{stage_key}.normalization_path in {result_path}"),
                )
                portable_normalization_path = PurePosixPath(normalization_path)
                if portable_normalization_path.is_absolute() or any(
                    part in {"", ".", ".."} for part in portable_normalization_path.parts
                ):
                    raise ResultSchemaError(
                        f"provenance.selected_checkpoints.{stage_key}.normalization_path "
                        f"in {result_path} must be a normalized relative POSIX path"
                    )
                if not isinstance(normalization_hash, str) or _SHA256_PATTERN.fullmatch(normalization_hash) is None:
                    raise ResultSchemaError(
                        f"provenance.selected_checkpoints.{stage_key}.normalization_hash "
                        f"in {result_path} must be sha256:<64 lowercase hex>"
                    )
            elif normalization_path is not None or normalization_hash is not None:
                raise ResultSchemaError(
                    f"JAX provenance.selected_checkpoints.{stage_key} in {result_path} "
                    "must not declare separate normalization artifacts"
                )
            selected_checkpoints[stage_key] = {
                "model_path": checkpoint_model_path,
                "model_hash": checkpoint_model_hash,
                "normalization_path": normalization_path,
                "normalization_hash": normalization_hash,
            }
        deliverable_runtime: dict[str, Any] = {}
        if not v4:
            # The published model is the terminal ADVANCING stage's checkpoint
            # — "3" (behavior) for every current species.  Recovery, at an
            # earlier manifest position and non-advancing, can never be
            # terminal here.
            terminal_key = next(key for key, entry in reversed(checkpoint_entries) if entry.legacy_number is not None)
            if selected_checkpoints[terminal_key]["model_path"] != selected_model_path:
                raise ResultSchemaError(
                    f"provenance.selected_model_path in {result_path} must match the terminal advancing "
                    f"stage {terminal_key} selected checkpoint"
                )
            if selected_checkpoints[terminal_key]["model_hash"] != identifiers["model_hash"]:
                raise ResultSchemaError(
                    f"provenance.model_hash in {result_path} must match the terminal advancing "
                    f"stage {terminal_key} selected checkpoint"
                )
        else:
            # v4: the published model is the PRIMARY deliverable's checkpoint
            # — the target when certified, else the deepest certified one —
            # and every recorded deliverable carries the checkpoint it names.
            deliverable_entries, deliverables = _validate_deliverable_records(
                provenance["deliverables"],
                species=species,
                backend=backend,
                result_path=result_path,
            )
            checkpoint_id_to_key = {entry.id: key for key, entry in checkpoint_entries}
            ancestor_entries: list[tuple[str, Any]] = []
            ancestors: dict[str, dict[str, Any]] = {}
            if provenance.get("ancestors") is not None:
                ancestor_entries, ancestors = _validate_ancestor_records(
                    provenance["ancestors"],
                    species=species,
                    result_path=result_path,
                )
                reused_and_trained = [key for key, entry in ancestor_entries if entry.id in checkpoint_id_to_key]
                if reused_and_trained:
                    raise ResultSchemaError(
                        f"provenance.ancestors in {result_path} names stages that also carry a selected "
                        f"checkpoint (a stage cannot be both trained here and reused): {reused_and_trained}"
                    )
            manifest = _load_manifest_for(species, field=f"provenance.deliverables in {result_path}")
            ancestor_ids = {entry.id for _, entry in ancestor_entries}
            for key, entry in deliverable_entries:
                checkpoint_key = checkpoint_id_to_key.get(entry.id)
                if checkpoint_key is None:
                    raise ResultSchemaError(
                        f"provenance.deliverables in {result_path}: deliverable {key} has no selected checkpoint"
                    )
                checkpoint = selected_checkpoints[checkpoint_key]
                for hash_field in ("model_path", "model_hash", "normalization_hash"):
                    if deliverables[key][hash_field] != checkpoint[hash_field]:
                        raise ResultSchemaError(
                            f"provenance.deliverables.{key}.{hash_field} in {result_path} does not match "
                            f"provenance.selected_checkpoints.{checkpoint_key}.{hash_field}"
                        )
                if deliverables[key]["certified"]:
                    missing_chain = [
                        ancestor.key
                        for ancestor in deliverable_chain(entry, manifest)
                        if ancestor.id not in checkpoint_id_to_key and ancestor.id not in ancestor_ids
                    ]
                    if missing_chain:
                        raise ResultSchemaError(
                            f"provenance.deliverables in {result_path}: certified deliverable {key} has no "
                            f"selected checkpoint or ancestor record for its chain ancestors {missing_chain}"
                        )
            target_deliverable = _optional_stage_key(
                provenance["target_deliverable"],
                species=species,
                field=f"provenance.target_deliverable in {result_path}",
            )
            primary_deliverable = _optional_stage_key(
                provenance["primary_deliverable"],
                species=species,
                field=f"provenance.primary_deliverable in {result_path}",
            )
            if not any(record["certified"] for record in deliverables.values()):
                raise ResultSchemaError(
                    f"canonical provenance in {result_path} records no certified deliverable; a publishable "
                    "result certifies at least one"
                )
            expected_primary = primary_deliverable_key(deliverables, species=species, target=target_deliverable)
            if primary_deliverable != expected_primary:
                raise ResultSchemaError(
                    f"provenance.primary_deliverable in {result_path} must be {expected_primary!r} (the target "
                    f"deliverable when certified, else the deepest certified deliverable); found "
                    f"{primary_deliverable!r}"
                )
            assert primary_deliverable is not None
            if not deliverables[primary_deliverable]["certified"]:
                raise ResultSchemaError(
                    f"provenance.primary_deliverable in {result_path} names {primary_deliverable!r}, "
                    "which is not certified"
                )
            if deliverables[primary_deliverable]["model_path"] != selected_model_path:
                raise ResultSchemaError(
                    f"provenance.selected_model_path in {result_path} must match the primary deliverable "
                    f"{primary_deliverable} checkpoint"
                )
            if deliverables[primary_deliverable]["model_hash"] != identifiers["model_hash"]:
                raise ResultSchemaError(
                    f"provenance.model_hash in {result_path} must match the primary deliverable "
                    f"{primary_deliverable} checkpoint"
                )
            deliverable_runtime = {
                "deliverables": deliverables,
                "ancestors": ancestors,
                "primary_deliverable": primary_deliverable,
                "target_deliverable": target_deliverable,
            }

        python_version = _require_nonempty_string(
            provenance["python_version"], field=f"provenance.python_version in {result_path}"
        )
        platform_name = _require_nonempty_string(
            provenance["platform"],
            field=f"provenance.platform in {result_path}",
        )
        dependency_versions = _require_mapping(
            provenance["dependency_versions"],
            field=f"provenance.dependency_versions in {result_path}",
        )
        for dependency, dependency_version in dependency_versions.items():
            _require_nonempty_string(
                dependency, field=f"dependency name in provenance.dependency_versions in {result_path}"
            )
            _optional_nonempty_string(
                dependency_version,
                field=f"version for provenance.dependency_versions.{dependency} in {result_path}",
            )

        canonical_runtime = {
            "run_id": run_id,
            "captured_at": captured_at,
            "species": species,
            "algorithm": algorithm,
            "backend": backend,
            "backend_version": provenance_backend_version,
            "repository_dirty": repository_dirty,
            "repository_patch_sha256": repository_patch_sha256,
            "training_seed": training_seed,
            "seed_roles": dict(seed_roles),
            "evaluation_protocols": {role: dict(protocol) for role, protocol in evaluation_protocols.items()},
            "evaluation_seeds": list(evaluation_seeds),
            "parallel_envs": parallel_envs,
            "hardware": hardware,
            "python_version": python_version,
            "platform": platform_name,
            "dependency_versions": dict(dependency_versions),
            "plant_identity": plant_identity,
            "selected_checkpoints": selected_checkpoints,
            "selected_model_path": selected_model_path,
            **deliverable_runtime,
        }
    elif v4:
        # A non-canonical v4 block (a partial or failed run's captured
        # provenance, a compat-wrapper summary) is not cross-checked against
        # selected checkpoints, but what it does record must be well-formed.
        noncanonical_species = provenance.get("species")
        noncanonical_backend = provenance.get("backend")
        if isinstance(noncanonical_species, str) and noncanonical_species.strip():
            if provenance.get("deliverables") is not None and noncanonical_backend in ALLOWED_TRAINING_BACKENDS:
                _, canonical_runtime["deliverables"] = _validate_deliverable_records(
                    provenance["deliverables"],
                    species=noncanonical_species,
                    backend=noncanonical_backend,
                    result_path=result_path,
                )
            if provenance.get("ancestors") is not None:
                _, canonical_runtime["ancestors"] = _validate_ancestor_records(
                    provenance["ancestors"],
                    species=noncanonical_species,
                    result_path=result_path,
                )
            for key in ("primary_deliverable", "target_deliverable"):
                if key in provenance:
                    canonical_runtime[key] = _optional_stage_key(
                        provenance[key],
                        species=noncanonical_species,
                        field=f"provenance.{key} in {result_path}",
                    )

    return {
        "model_revision_status": model_revision_status,
        "verification_status": verification_status,
        "evaluation_episodes": evaluation_episodes,
        **identifiers,
        **canonical_runtime,
    }


def validate_captured_provenance(
    provenance: Any,
    *,
    result_path: str = "provenance.json",
) -> dict[str, Any]:
    """Validate immutable capture-time fields for partial or failed bundles.

    Final checkpoint/config identifiers may still be null, but every value
    needed to continue the same experiment must already be present and valid.

    The block is passed through canonical validation with placeholders for
    the finalization fields it lacks.  A block that already carries a
    ``deliverables`` map with a certified entry (a finalized v4 partial run)
    keeps its real deliverables, checkpoints and primary; otherwise ONE
    placeholder deliverable is synthesized, keyed on the species' last
    manifest deliverable (``"3"`` for every current species), and stripped
    from the return again.
    """
    provenance = _require_mapping(provenance, field=f"provenance in {result_path}")
    validate_provenance(provenance, result_path=result_path, canonical=False)
    captured_fields = set(CANONICAL_RUNTIME_PROVENANCE_FIELDS) - {
        "backend_version",
        "selected_checkpoints",
        "selected_model_path",
    }
    missing_fields = sorted(captured_fields - provenance.keys())
    if missing_fields:
        raise ResultSchemaError(f"captured provenance in {result_path} is missing fields: {missing_fields}")

    candidate = dict(provenance)
    dirty = candidate.get("repository_dirty")
    patch_hash = candidate.get("repository_patch_sha256")
    if dirty is True:
        if not isinstance(patch_hash, str) or _SHA256_PATTERN.fullmatch(patch_hash) is None:
            raise ResultSchemaError(
                f"provenance.repository_patch_sha256 in {result_path} is required and must be "
                "sha256:<64 lowercase hex> for a dirty repository"
            )
        # Canonical promotion rejects dirty state, while capture-time auditing
        # accepts it so the interrupted run remains inspectable.
        candidate["repository_dirty"] = False
        candidate["repository_patch_sha256"] = None
    if candidate.get("model_hash") is None:
        candidate["model_hash"] = "sha256:" + "0" * 64
    if candidate.get("config_hash") is None:
        candidate["config_hash"] = "sha256:" + "0" * 64
    if candidate.get("backend_version") is None:
        candidate["backend_version"] = "pending"
    recorded_deliverables = candidate.get("deliverables")
    finalized = isinstance(recorded_deliverables, Mapping) and any(
        isinstance(record, Mapping) and record.get("certified") is True for record in recorded_deliverables.values()
    )
    synthesized_fields: set[str] = set()
    if not finalized:
        # No certified deliverable was finalized (a capture-time block, a
        # pre-Phase-A partial bundle, or a finalized FAILED run): stand in
        # one pending deliverable so the canonical rules can run.
        species = _require_nonempty_string(candidate.get("species"), field=f"provenance.species in {result_path}")
        manifest = _load_manifest_for(species, field=f"provenance in {result_path}")
        if not manifest.deliverables:
            raise ResultSchemaError(f"provenance in {result_path}: the {species} manifest declares no deliverable")
        placeholder_entry = manifest.deliverables[-1]
        placeholder_key = placeholder_entry.key
        if candidate.get("selected_model_path") is None:
            candidate["selected_model_path"] = "pending/model.bin"
        normalization_required = candidate.get("backend") == "stable-baselines3"
        # The placeholder's chain ancestors get pending checkpoints too —
        # for every current species exactly the historical {"1","2","3"}
        # trio — except those the block already carries as reused ancestors.
        recorded_ancestors = candidate.get("ancestors")
        reused_ids = set()
        if isinstance(recorded_ancestors, Mapping):
            for ancestor_key in recorded_ancestors:
                reused_ids.add(_resolve_stage_ref(ancestor_key, manifest, field="provenance.ancestors").id)
        candidate["selected_checkpoints"] = {
            ancestor.key: {
                "model_path": f"pending/{ancestor.id}/model.bin",
                "model_hash": "sha256:" + "0" * 64,
                "normalization_path": f"pending/{ancestor.id}/vecnormalize.pkl" if normalization_required else None,
                "normalization_hash": "sha256:" + "0" * 64 if normalization_required else None,
            }
            for ancestor in manifest.ancestors(placeholder_entry.id)
            if ancestor.id not in reused_ids
        }
        candidate["selected_checkpoints"][placeholder_key] = {
            "model_path": candidate["selected_model_path"],
            "model_hash": candidate["model_hash"],
            "normalization_path": "pending/vecnormalize.pkl" if normalization_required else None,
            "normalization_hash": "sha256:" + "0" * 64 if normalization_required else None,
        }
        candidate["deliverables"] = {
            placeholder_key: {
                "model_path": candidate["selected_model_path"],
                "model_hash": candidate["model_hash"],
                "normalization_hash": "sha256:" + "0" * 64 if normalization_required else None,
                "gate_kind": "pending",
                "certified": True,
                "replication": {
                    "count": 1,
                    "runs": [{"run_id": candidate.get("run_id"), "training_seed": candidate.get("training_seed")}],
                },
            }
        }
        candidate["primary_deliverable"] = placeholder_key
        candidate["target_deliverable"] = placeholder_key
        synthesized_fields = {"deliverables", "primary_deliverable", "target_deliverable"}
    validated = validate_provenance(
        candidate,
        result_path=result_path,
        canonical=True,
        schema_version=RESULT_SCHEMA_VERSION,
    )
    return {
        key: value
        for key, value in validated.items()
        if (key not in {"backend_version", "selected_model_path"} or provenance.get(key) is not None)
        and key not in synthesized_fields
    }


def validate_result_summary(
    summary: Any,
    *,
    expected_species: str | None = None,
    relative_path: str | Path | None = None,
    result_path: str | Path | None = None,
    require_complete: bool = True,
    require_publishable: bool = False,
    canonical_provenance: bool = False,
    require_canonical_provenance: bool | None = None,
) -> dict[str, Any]:
    """Validate a schema v2/v3/v4 result summary.

    Below schema 4, complete public summaries contain every advancing stage
    the species declares (1, 2, and 3 — non-advancing semantic stages such
    as recovery are optional), and ``require_publishable`` is a synonym of
    ``require_complete``.  From schema 4 on (decision D-A2):

    * ``require_publishable`` — ``provenance.deliverables`` is present, at
      least one deliverable is certified, every certified deliverable's
      chain is present (in ``stages`` or ``provenance.ancestors``), each
      recorded ``certified`` flag equals the recomputation from the recorded
      verdicts, and ``bundle_status`` is ``complete`` or ``partial`` and
      equals :func:`bundle_status_for` of the deliverables and the recorded
      target;
    * ``require_complete`` — publishable AND ``bundle_status == "complete"``;
    * neither — any non-empty valid stage subset, so a Colab run can be
      checked after each node without being eligible for publication yet.

    Canonical provenance implies the publishable rules and additionally
    requires the headline ``final_avg_reward`` to be the primary
    deliverable's ``final_eval_reward``.
    """

    if require_canonical_provenance is not None:
        canonical_provenance = require_canonical_provenance
    label_source = result_path if result_path is not None else relative_path
    label = str(label_source) if label_source is not None else "result summary"
    summary = _require_mapping(summary, field=f"result summary {label}")
    schema_version = summary.get("schema_version")
    if schema_version not in SUPPORTED_RESULT_SCHEMA_VERSIONS:
        raise ResultSchemaError(
            f"result summary schema_version must be one of {sorted(SUPPORTED_RESULT_SCHEMA_VERSIONS)}: {label}"
        )
    v4 = int(schema_version) >= 4
    if not v4:
        # The v3 flags are synonyms: "publishable" meant "every advancing
        # stage present and passed", which is also what "complete" meant.
        require_complete = require_complete or require_publishable

    species = _require_nonempty_string(summary.get("species"), field=f"species in {label}")
    if expected_species is not None and species != expected_species:
        raise ResultSchemaError(f"result species mismatch in {label}: {summary.get('species')}")
    algorithm = _require_nonempty_string(summary.get("algorithm"), field=f"algorithm in {label}")
    backend = summary.get("backend")
    if backend not in ALLOWED_TRAINING_BACKENDS:
        raise ResultSchemaError(f"invalid backend in {label}: {backend}")
    backend = cast(str, backend)
    backend_version = _optional_nonempty_string(summary.get("backend_version"), field=f"backend_version in {label}")
    if relative_path is not None:
        validate_result_path(
            relative_path,
            species=species,
            algorithm=algorithm,
            backend=backend,
        )

    result_date = _require_nonempty_string(summary.get("date"), field=f"date in {label}")
    try:
        date.fromisoformat(result_date)
    except ValueError as exc:
        raise ResultSchemaError(f"date in {label} must use ISO YYYY-MM-DD format") from exc

    _optional_nonempty_string(summary.get("hardware"), field=f"hardware in {label}")
    seed = summary.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ResultSchemaError(f"seed in {label} must be null or a non-negative integer")
    parallel_envs = summary.get("parallel_envs")
    if parallel_envs is not None:
        _require_positive_int(parallel_envs, field=f"parallel_envs in {label}")

    raw_stages = _require_mapping(summary.get("stages"), field=f"stages in {label}")
    # Stage keys are references into the species' manifest — legacy decimal
    # spellings and semantic ids — validated and ORDERED there (never by
    # int() on the key, which the recovery id would crash and a position
    # would silently corrupt).
    stage_entries = ordered_stage_entries(raw_stages, species=species, field=f"stages in {label}")
    if not stage_entries:
        raise ResultSchemaError(f"stages in {label} must record at least one stage")
    if require_complete and not v4:
        missing_advancing = _missing_advancing_stages(stage_entries, species=species)
        if missing_advancing:
            raise ResultSchemaError(
                f"stages in {label} must contain every advancing stage; "
                f"missing {[entry.key for entry in missing_advancing]}"
            )

    stage_timesteps = 0
    for stage_key, stage_entry in stage_entries:
        raw_stage = _require_mapping(raw_stages[stage_key], field=f"stage {stage_key} in {label}")
        prefix = f"stage {stage_key} in {label}"
        _require_nonempty_string(raw_stage.get("name"), field=f"name for {prefix}")
        _require_nonempty_string(raw_stage.get("description"), field=f"description for {prefix}")
        stage_timesteps += _require_positive_int(raw_stage.get("timesteps"), field=f"timesteps for {prefix}")
        for metric_key in (
            "best_eval_reward",
            "final_eval_reward",
            "avg_forward_vel",
            "avg_episode_length",
        ):
            if metric_key not in raw_stage:
                raise ResultSchemaError(f"{metric_key} is required for {prefix}")
            metric_value = _optional_number(raw_stage[metric_key], field=f"{metric_key} for {prefix}")
            if canonical_provenance and metric_value is None:
                raise ResultSchemaError(f"{metric_key} must be a finite number for canonical {prefix}")
        for metric_key in (
            "selected_model_reward",
            "selected_model_reward_std",
            "selected_model_episode_length",
            "selected_model_episode_length_std",
            "selected_model_forward_vel",
            "selected_model_forward_vel_std",
            "selected_model_distance",
            "selected_model_success_rate",
        ):
            if metric_key not in raw_stage:
                if canonical_provenance:
                    raise ResultSchemaError(f"{metric_key} is required for canonical {prefix}")
                continue
            metric_value = _optional_number(raw_stage[metric_key], field=f"{metric_key} for {prefix}")
            if canonical_provenance and metric_value is None:
                raise ResultSchemaError(f"{metric_key} must be a finite number for canonical {prefix}")
        for metric_key in (
            "final_eval_std",
            "avg_episode_length_std",
            "avg_forward_vel_std",
            "mean_distance_traveled",
            "mean_success_rate",
        ):
            if metric_key not in raw_stage:
                if canonical_provenance:
                    raise ResultSchemaError(f"{metric_key} is required for canonical {prefix}")
                continue
            metric_value = _optional_number(raw_stage[metric_key], field=f"{metric_key} for {prefix}")
            if canonical_provenance and metric_value is None:
                raise ResultSchemaError(f"{metric_key} must be a finite number for canonical {prefix}")
        for metric_key in ("best_eval_std", "best_eval_step", "training_time_seconds"):
            if metric_key in raw_stage:
                _optional_number(raw_stage[metric_key], field=f"{metric_key} for {prefix}")
        success_rate = _optional_number(raw_stage.get("mean_success_rate"), field=f"mean_success_rate for {prefix}")
        if success_rate is not None and not 0.0 <= success_rate <= 1.0:
            raise ResultSchemaError(f"mean_success_rate for {prefix} must be between 0 and 1")
        selected_success_rate = _optional_number(
            raw_stage.get("selected_model_success_rate"),
            field=f"selected_model_success_rate for {prefix}",
        )
        if selected_success_rate is not None and not 0.0 <= selected_success_rate <= 1.0:
            raise ResultSchemaError(f"selected_model_success_rate for {prefix} must be between 0 and 1")
        _optional_nonempty_string(raw_stage.get("training_time"), field=f"training_time for {prefix}")
        if not isinstance(raw_stage.get("stage_passed"), bool):
            raise ResultSchemaError(f"stage_passed for {prefix} must be a boolean")
        if canonical_provenance and not isinstance(raw_stage.get("publication_gate_passed"), bool):
            raise ResultSchemaError(f"publication_gate_passed for canonical {prefix} must be a boolean")
        if canonical_provenance and raw_stage.get("publication_gate_passed") != raw_stage.get("stage_passed"):
            raise ResultSchemaError(f"publication_gate_passed for canonical {prefix} must match stage_passed")
        # Below schema 4 only ADVANCING stages must have passed for a
        # canonical bundle: that is what publication certified.  A
        # non-advancing pilot (recovery) records its verdict honestly —
        # necessarily False under gate_kind none/v1 — without blocking the
        # curriculum's publication or being laundered into a pass.  From
        # schema 4 on every stage records its verdict honestly and the
        # per-deliverable certification below decides what is published.
        if (
            canonical_provenance
            and not v4
            and stage_entry.legacy_number is not None
            and raw_stage["stage_passed"] is not True
        ):
            raise ResultSchemaError(f"stage_passed for canonical {prefix} must be true")

    total_timesteps = _require_positive_int(summary.get("total_timesteps"), field=f"total_timesteps in {label}")
    if total_timesteps != stage_timesteps:
        raise ResultSchemaError(
            f"total_timesteps in {label} is {total_timesteps}, but stage totals sum to {stage_timesteps}"
        )
    _optional_number(
        summary.get("total_training_time_seconds"),
        field=f"total_training_time_seconds in {label}",
    )
    _optional_nonempty_string(summary.get("total_training_time"), field=f"total_training_time in {label}")
    _optional_number(summary.get("final_avg_reward"), field=f"final_avg_reward in {label}")

    provenance = validate_provenance(
        summary.get("provenance"),
        result_path=label,
        canonical=canonical_provenance,
        schema_version=int(schema_version),
    )
    stage_key_by_id = {entry.id: key for key, entry in stage_entries}
    primary_stage_key: str | None = None
    if v4:
        bundle_status = summary.get("bundle_status")
        if bundle_status is not None and bundle_status not in ALLOWED_BUNDLE_STATUSES:
            raise ResultSchemaError(f"bundle_status in {label} must be one of {sorted(ALLOWED_BUNDLE_STATUSES)}")
        if require_publishable or require_complete or canonical_provenance:
            deliverables = provenance.get("deliverables")
            if deliverables is None:
                raise ResultSchemaError(
                    f"{label} is a schema-{schema_version} result without provenance.deliverables, so it "
                    "certifies nothing and is not publishable"
                )
            ancestors = provenance.get("ancestors") or {}
            recomputed = certified_deliverables(stage_entries, raw_stages, ancestors, species=species)
            deliverable_entries = ordered_stage_entries(
                deliverables,
                species=species,
                field=f"provenance.deliverables in {label}",
            )
            recorded_ids = {entry.id for _, entry in deliverable_entries}
            missing_deliverables = [
                key for key, entry in stage_entries if entry.deliverable and entry.id not in recorded_ids
            ]
            if missing_deliverables:
                raise ResultSchemaError(
                    f"provenance.deliverables in {label} must record every deliverable stage the summary "
                    f"records; missing {missing_deliverables}"
                )
            for key, entry in deliverable_entries:
                if entry.id not in stage_key_by_id:
                    raise ResultSchemaError(
                        f"provenance.deliverables in {label} names stage {key!r}, which the summary does not record"
                    )
                claimed = _recorded_verdict(deliverables[key], key=key, field=f"provenance.deliverables in {label}")
                actual = recomputed[stage_key_by_id[entry.id]]
                if claimed and not actual:
                    reasons = uncertified_chain_members(
                        stage_key_by_id[entry.id], stage_entries, raw_stages, ancestors, species=species
                    )
                    raise ResultSchemaError(
                        f"provenance.deliverables.{key} in {label} is recorded as certified, but its recorded "
                        f"verdicts do not certify it: {'; '.join(reasons)}"
                    )
                if actual and not claimed:
                    raise ResultSchemaError(
                        f"provenance.deliverables.{key} in {label} is recorded as uncertified although its "
                        "gate and every chain ancestor's gate passed"
                    )
            if not any(recomputed.values()):
                raise ResultSchemaError(f"{label} records no certified deliverable, so it is not publishable")
            target = provenance.get("target_deliverable")
            expected_status = bundle_status_for(deliverables, species=species, target=target, stages=raw_stages)
            if bundle_status != expected_status:
                raise ResultSchemaError(
                    f"bundle_status in {label} must be {expected_status!r} for its deliverables and target "
                    f"{target!r}; found {bundle_status!r}"
                )
            if require_complete and bundle_status != "complete":
                raise ResultSchemaError(
                    f"{label} has bundle_status {bundle_status!r}; a complete result certifies its target "
                    f"deliverable {target!r} and every deliverable it records"
                )
            primary = provenance.get("primary_deliverable")
            if primary is None:
                primary = primary_deliverable_key(deliverables, species=species, target=target)
            if primary is not None:
                primary_entry = next(entry for key, entry in deliverable_entries if key == primary)
                primary_stage_key = stage_key_by_id[primary_entry.id]
    if canonical_provenance:
        if not v4 and summary.get("bundle_status") != "complete":
            raise ResultSchemaError(f"canonical result {label} must have bundle_status='complete'")
        run_id = _require_nonempty_string(summary.get("run_id"), field=f"run_id in {label}")
        if provenance["run_id"] != run_id:
            raise ResultSchemaError(
                f"run_id in {label} does not match provenance.run_id: {run_id!r} != {provenance['run_id']!r}"
            )
        if seed is None:
            raise ResultSchemaError(f"seed in {label} is required for canonical provenance")
        if provenance["training_seed"] != seed:
            raise ResultSchemaError(
                f"seed in {label} does not match provenance.training_seed: {seed!r} != {provenance['training_seed']!r}"
            )
        if provenance["species"] != species:
            raise ResultSchemaError(
                f"species in {label} does not match provenance.species: {species!r} != {provenance['species']!r}"
            )
        if provenance["algorithm"] != algorithm:
            raise ResultSchemaError(
                f"algorithm in {label} does not match provenance.algorithm: "
                f"{algorithm!r} != {provenance['algorithm']!r}"
            )
        if provenance["backend"] != backend:
            raise ResultSchemaError(
                f"backend in {label} does not match provenance.backend: {backend!r} != {provenance['backend']!r}"
            )
        if backend_version is not None and provenance["backend_version"] != backend_version:
            raise ResultSchemaError(
                f"backend_version in {label} does not match provenance.backend_version: "
                f"{backend_version!r} != {provenance['backend_version']!r}"
            )
        captured_date = datetime.fromisoformat(provenance["captured_at"].replace("Z", "+00:00")).date().isoformat()
        if result_date != captured_date:
            raise ResultSchemaError(
                f"date in {label} does not match provenance.captured_at: {result_date!r} != {captured_date!r}"
            )
        if parallel_envs != provenance["parallel_envs"]:
            raise ResultSchemaError(f"parallel_envs in {label} does not match provenance.parallel_envs")
        if summary.get("hardware") != provenance["hardware"]:
            raise ResultSchemaError(f"hardware in {label} does not match provenance.hardware")
        summary_plant = _canonical_plant_identity(
            summary.get("plant_identity"),
            species=species,
            field=f"plant_identity in {label}",
        )
        if summary_plant != provenance["plant_identity"]:
            raise ResultSchemaError(f"plant_identity in {label} does not match provenance.plant_identity")
        # The run's headline reward: below schema 4 the terminal ADVANCING
        # stage's ("3", behavior, for every current species) — a trailing
        # non-advancing stage must not redefine it; from schema 4 on the
        # PRIMARY deliverable's, the same checkpoint selected_model_path
        # names (a failed leaf never headlines the run).
        if v4:
            final_stage_key = primary_stage_key
        else:
            final_stage_key = next(
                (key for key, entry in reversed(stage_entries) if entry.legacy_number is not None),
                None,
            )
        if final_stage_key is None:
            raise ResultSchemaError(f"canonical result {label} records no advancing stage")
        final_stage_reward = raw_stages[final_stage_key].get("final_eval_reward")
        if summary.get("final_avg_reward") != final_stage_reward:
            raise ResultSchemaError(
                f"final_avg_reward in {label} does not match stage {final_stage_key} final_eval_reward"
            )
        # Every recorded stage — recovery included — trained on the clock.
        stage_duration_total = round(
            sum(float(raw_stages[key].get("training_time_seconds") or 0.0) for key, _ in stage_entries),
            1,
        )
        if summary.get("total_training_time_seconds") != stage_duration_total:
            raise ResultSchemaError(f"total_training_time_seconds in {label} does not match the stage total")

    claims_certified = (
        provenance["model_revision_status"] == "current" or provenance["verification_status"] == "verified"
    )
    if (canonical_provenance or claims_certified) and backend_version is None:
        qualifier = "canonical" if canonical_provenance else "current or verified"
        raise ResultSchemaError(f"{qualifier} result {label} is missing backend_version")

    return cast(dict[str, Any], summary)
