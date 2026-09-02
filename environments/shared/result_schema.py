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
every committed historical artifact validates unchanged and both versions
are accepted here; writers emit :data:`RESULT_SCHEMA_VERSION`.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, cast

RESULT_SCHEMA_VERSION = 3
#: Versions this reader accepts.  v2 is the integer-stage schema every
#: committed summary under results/ carries; v3 widened the stage-key
#: vocabulary (module docstring) without changing any v2-valid artifact.
SUPPORTED_RESULT_SCHEMA_VERSIONS = frozenset({2, RESULT_SCHEMA_VERSION})
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


def validate_provenance(
    provenance: Any,
    *,
    result_path: str = "result summary",
    canonical: bool = False,
) -> dict[str, Any]:
    """Validate result-summary provenance (schema v2/v3).

    Historical summaries may retain unknown identifiers as ``null``.  Canonical
    mode is for newly exported bundles and requires complete, well-formed
    identity fields regardless of the result's current/historical label.
    """

    provenance = _require_mapping(provenance, field=f"provenance in {result_path}")
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

        missing_runtime_fields = [field for field in CANONICAL_RUNTIME_PROVENANCE_FIELDS if field not in provenance]
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
        # v2 required exactly {"1", "2", "3"}, which enforced "a complete
        # curriculum recorded a handoff per advancing stage".  Preserved
        # exactly: every advancing stage is still required; non-advancing
        # semantic stages (recovery) may add a checkpoint but never replace
        # one, and any key outside the species' vocabulary fails closed.
        checkpoint_entries = ordered_stage_entries(
            selected_checkpoints_value,
            species=species,
            field=f"provenance.selected_checkpoints in {result_path}",
        )
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
        # The published model is the terminal ADVANCING stage's checkpoint —
        # "3" (behavior) for every current species.  Recovery, at an earlier
        # manifest position and non-advancing, can never be terminal here.
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
        }

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
    if candidate.get("selected_model_path") is None:
        candidate["selected_model_path"] = "pending/model.bin"
    normalization_required = candidate.get("backend") == "stable-baselines3"
    candidate["selected_checkpoints"] = {
        stage: {
            "model_path": candidate["selected_model_path"] if stage == "3" else f"pending/stage{stage}/model.bin",
            "model_hash": candidate["model_hash"],
            "normalization_path": f"pending/stage{stage}/vecnormalize.pkl" if normalization_required else None,
            "normalization_hash": "sha256:" + "0" * 64 if normalization_required else None,
        }
        for stage in ("1", "2", "3")
    }
    validated = validate_provenance(
        candidate,
        result_path=result_path,
        canonical=True,
    )
    return {
        key: value
        for key, value in validated.items()
        if key not in {"backend_version", "selected_model_path"} or provenance.get(key) is not None
    }


def validate_result_summary(
    summary: Any,
    *,
    expected_species: str | None = None,
    relative_path: str | Path | None = None,
    result_path: str | Path | None = None,
    require_complete: bool = True,
    canonical_provenance: bool = False,
    require_canonical_provenance: bool | None = None,
) -> dict[str, Any]:
    """Validate a schema v2/v3 result summary.

    Complete public summaries contain every advancing stage the species
    declares (1, 2, and 3 — non-advancing semantic stages such as recovery
    are optional).  Partial validation accepts any non-empty valid stage
    subset so a Colab run can be checked after each stage without being
    eligible for publication yet.
    """

    if require_canonical_provenance is not None:
        canonical_provenance = require_canonical_provenance
    label_source = result_path if result_path is not None else relative_path
    label = str(label_source) if label_source is not None else "result summary"
    summary = _require_mapping(summary, field=f"result summary {label}")
    if summary.get("schema_version") not in SUPPORTED_RESULT_SCHEMA_VERSIONS:
        raise ResultSchemaError(
            f"result summary schema_version must be one of {sorted(SUPPORTED_RESULT_SCHEMA_VERSIONS)}: {label}"
        )

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
    if require_complete:
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
        # Only ADVANCING stages must have passed for a canonical bundle: that
        # is what publication certifies.  A non-advancing pilot (recovery)
        # records its verdict honestly — today necessarily False, since its
        # placeholder gate_kind none/v1 refuses to pass — without blocking
        # the curriculum's publication or being laundered into a pass.
        if canonical_provenance and stage_entry.legacy_number is not None and raw_stage["stage_passed"] is not True:
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
    )
    if canonical_provenance:
        if summary.get("bundle_status") != "complete":
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
        # The run's headline reward is the terminal ADVANCING stage's ("3",
        # behavior, for every current species) — same rule as the selected
        # model above; a trailing non-advancing stage must not redefine it.
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
