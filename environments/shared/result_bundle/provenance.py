"""Run provenance: capture it when the run starts, update it as it finishes.

Records the repository state, seed roles, and dependency versions up front so
that a run whose runtime later disconnects still says where it came from."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from . import constants, hashing
from .constants import (
    _DEPENDENCY_PACKAGES,
    _FINALIZATION_PROVENANCE_FIELDS,
    DEFAULT_PROVENANCE_NAME,
    PROVENANCE_SCHEMA_VERSION,
)
from .errors import ResultBundleError
from .hashing import canonical_json_sha256
from .naming import _normalize_plant_identity, canonical_algorithm, canonical_backend

_logger = logging.getLogger(__name__)

#: Identifies this Python process in the provenance ``sessions`` record. A
#: repeated :func:`initialize_result_bundle` call from the same process is a
#: retry, not a resume, so it must stay a byte-identical no-op; a call from a
#: fresh process (a new Colab runtime resuming the run) appends a session entry.
_PROCESS_SESSION_TOKEN = uuid.uuid4().hex


def _git_command(repository_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _sanitize_repository_url(remote: str | None) -> str | None:
    """Remove credentials and URL secrets before provenance is published."""
    if remote is None:
        return None
    value = remote.strip()
    if not value:
        return None
    if Path(value).expanduser().is_absolute() or value.startswith(("./", "../", "~")):
        return None
    if "://" in value:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if not hostname:
            return None
        netloc = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        try:
            if parsed.port is not None:
                netloc = f"{netloc}:{parsed.port}"
        except ValueError:
            return None
        # Query strings and fragments can also carry access tokens.
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    # SCP-style Git remotes normally use ``git@host:path``. Drop the user
    # component as it is unnecessary provenance and can itself be a token.
    if "@" in value and ":" in value.partition("@")[2]:
        value = value.partition("@")[2]
    return value


def _repository_state(repository_root: Path) -> dict[str, Any]:
    commit = _git_command(repository_root, "rev-parse", "HEAD") or os.environ.get("GITHUB_SHA") or "unknown"
    status = _git_command(repository_root, "status", "--porcelain", "--untracked-files=all")
    diff = _git_command(repository_root, "diff", "--binary", "HEAD")
    untracked_output = _git_command(repository_root, "ls-files", "--others", "--exclude-standard", "-z")
    remote = _git_command(repository_root, "config", "--get", "remote.origin.url")
    dirty = None if status is None else bool(status)
    patch_sha256 = None
    if dirty and diff is not None and untracked_output is not None:
        untracked: list[dict[str, str]] = []
        for relative in sorted(path for path in untracked_output.split("\0") if path):
            candidate = (repository_root / relative).resolve()
            try:
                candidate.relative_to(repository_root.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                untracked.append({"path": relative, "sha256": hashing.sha256_file(candidate)})
        patch_sha256 = canonical_json_sha256(
            {
                "tracked_diff_sha256": f"sha256:{hashlib.sha256(diff.encode('utf-8')).hexdigest()}",
                "untracked": untracked,
            }
        )
    return {
        "repository_url": _sanitize_repository_url(remote),
        "repository_commit": commit,
        "repository_dirty": dirty,
        "repository_patch_sha256": patch_sha256,
    }


def _dependency_versions() -> dict[str, str | None]:
    installed: dict[str, str | None] = {}
    for label, package in _DEPENDENCY_PACKAGES.items():
        try:
            installed[label] = version(package)
        except PackageNotFoundError:
            installed[label] = None
        except Exception:
            installed[label] = None
    return installed


def initialize_result_bundle(
    run_dir: str | Path,
    *,
    species: str,
    algorithm: str,
    seed: int,
    backend: str | None = None,
    evaluation_seeds: Sequence[int] | None = None,
    evaluation_episodes: int = 30,
    seed_roles: Mapping[str, int] | None = None,
    parallel_envs: int | None = None,
    hardware: str | None = None,
    plant_identity: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    repository_root: str | Path | None = None,
    captured_at: str | None = None,
) -> Path:
    """Capture immutable run identity before training starts.

    Repeated calls from the same process are byte-identical no-ops.  A reused
    run directory is compared against the captured provenance in two tiers:

    - **Identity** fields (species, algorithm, backend, training seed, seed
      roles, evaluation protocols/seeds/episodes, parallel envs, plant
      identity, and ``run_id`` when provided) must match exactly; a mismatch
      is rejected instead of silently mixing artifacts.
    - **Environment** fields (``python_version``, ``platform``,
      ``dependency_versions``, ``repository_commit``, ``repository_dirty``,
      ``repository_patch_sha256``, ``hardware``) may drift between sessions —
      a resumed Colab runtime rebuilds them from an unpinned clone and
      install.  The top-level fields keep the original session's values; the
      drift is recorded in the appended ``sessions`` entry instead.

    Every run records a ``sessions`` list: the creating process appends
    ``{"session_token", "started_at"}``, and each later process that resumes
    the run appends ``{"session_token", "resumed_at"}`` plus an
    ``environment_drift`` mapping when its environment differs from the one
    originally captured.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ResultBundleError("training seed must be a non-negative integer")
    if not isinstance(evaluation_episodes, int) or isinstance(evaluation_episodes, bool) or evaluation_episodes <= 0:
        raise ResultBundleError("evaluation_episodes must be a positive integer")
    if parallel_envs is not None and (
        not isinstance(parallel_envs, int) or isinstance(parallel_envs, bool) or parallel_envs <= 0
    ):
        raise ResultBundleError("parallel_envs must be a positive integer when provided")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ResultBundleError("run_id must be a non-empty string when provided")
    captured_timestamp = captured_at or datetime.now(timezone.utc).isoformat()
    try:
        captured_datetime = datetime.fromisoformat(captured_timestamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ResultBundleError("captured_at must be an ISO-8601 timestamp") from exc
    if captured_datetime.tzinfo is None or captured_datetime.utcoffset() is None:
        raise ResultBundleError("captured_at must include a UTC offset")

    public_algorithm = canonical_algorithm(algorithm)
    public_backend = canonical_backend(algorithm, backend)
    evaluation_seed_list = list(evaluation_seeds or [])
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in evaluation_seed_list):
        raise ResultBundleError("evaluation_seeds must contain non-negative integers")
    if len(evaluation_seed_list) != len(set(evaluation_seed_list)):
        raise ResultBundleError("evaluation_seeds must not contain duplicates")
    default_seed_roles = {"training": seed}
    if evaluation_seed_list:
        default_seed_roles["publication_evaluation"] = evaluation_seed_list[0]
        for index, evaluation_seed in enumerate(evaluation_seed_list[1:], start=2):
            default_seed_roles[f"additional_evaluation_{index}"] = evaluation_seed
    normalized_seed_roles = dict(seed_roles or default_seed_roles)
    if any(not isinstance(key, str) or not key.strip() for key in normalized_seed_roles):
        raise ResultBundleError("seed_roles keys must be non-empty strings")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in normalized_seed_roles.values()
    ):
        raise ResultBundleError("seed_roles values must be non-negative integers")
    if normalized_seed_roles.get("training") != seed:
        raise ResultBundleError("seed_roles.training must equal the training seed")
    evaluation_role_seeds = {role_seed for role, role_seed in normalized_seed_roles.items() if "evaluation" in role}
    if set(evaluation_seed_list) != evaluation_role_seeds:
        raise ResultBundleError("evaluation_seeds must exactly match the seeds assigned to evaluation roles")
    from ..result_schema import seed_role_collisions

    collisions = seed_role_collisions(normalized_seed_roles)
    if collisions:
        raise ResultBundleError("seed_roles must keep the publication seed distinct: " + "; ".join(collisions))
    evaluation_protocols = {
        role: {
            "seed": role_seed,
            "episodes": evaluation_episodes,
            "deterministic": True,
        }
        for role, role_seed in normalized_seed_roles.items()
        if "evaluation" in role
    }
    normalized_plant_identity = _normalize_plant_identity(plant_identity, species=species)

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    provenance_path = run_path / DEFAULT_PROVENANCE_NAME
    if repository_root is None:
        repository = constants.REPOSITORY_ROOT
    else:
        repository = Path(repository_root).resolve()
    current_repository_state = _repository_state(repository)
    current_python_version = platform.python_version()
    current_platform = platform.platform()
    current_dependency_versions = _dependency_versions()

    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_identity = {
            "species": species,
            "algorithm": public_algorithm,
            "backend": public_backend,
            "training_seed": seed,
            "seed_roles": normalized_seed_roles,
            "evaluation_protocols": evaluation_protocols,
            "evaluation_seeds": evaluation_seed_list,
            "evaluation_episodes": evaluation_episodes,
            "parallel_envs": parallel_envs,
            "plant_identity": normalized_plant_identity,
        }
        if run_id is not None:
            expected_identity["run_id"] = run_id
        mismatches = {
            key: (existing.get(key), value) for key, value in expected_identity.items() if existing.get(key) != value
        }
        if mismatches:
            raise ResultBundleError(f"run directory already belongs to a different run: {mismatches}")

        recorded_sessions = existing.get("sessions")
        if not isinstance(recorded_sessions, list):
            recorded_sessions = []
        last_session = recorded_sessions[-1] if recorded_sessions else None
        last_token = last_session.get("session_token") if isinstance(last_session, Mapping) else None
        if last_token == _PROCESS_SESSION_TOKEN:
            # A retry from the process already on record: keep the file
            # byte-identical.
            return provenance_path

        manifest_path = run_path / constants.DEFAULT_MANIFEST_NAME
        if manifest_path.exists():
            try:
                manifest_status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
            except (OSError, ValueError):
                manifest_status = None
            if manifest_status == "complete":
                # The run is finished: a fresh process touching it is a
                # read-only re-export, not a resumed session.  Mutating the
                # provenance here would stale the complete manifest's hash and
                # the summary's embedded copy, wedging the bundle as
                # canonical-conflict.  Identity was already validated above.
                _logger.info(
                    "run %s is already complete; not recording a new session for this process",
                    existing.get("run_id"),
                )
                return provenance_path

        # A fresh process resuming the run.  Environment fields may legitimately
        # differ (unpinned clone and reinstall in a new runtime); the top-level
        # values keep the original capture, and the drift is recorded on this
        # session's entry.
        current_environment = {
            "python_version": current_python_version,
            "platform": current_platform,
            "dependency_versions": current_dependency_versions,
            "repository_commit": current_repository_state["repository_commit"],
            "repository_dirty": current_repository_state["repository_dirty"],
            "repository_patch_sha256": current_repository_state["repository_patch_sha256"],
            "hardware": hardware,
        }
        environment_drift = {
            field: {"was": existing.get(field), "now": value}
            for field, value in current_environment.items()
            if existing.get(field) != value
        }
        session_entry: dict[str, Any] = {
            "session_token": _PROCESS_SESSION_TOKEN,
            "resumed_at": captured_timestamp,
        }
        if environment_drift:
            session_entry["environment_drift"] = environment_drift
            _logger.warning(
                "resuming run %s with a drifted environment (%s); keeping the originally captured values "
                "and recording the drift in the provenance sessions record",
                existing.get("run_id"),
                ", ".join(sorted(environment_drift)),
            )
        existing["sessions"] = [*recorded_sessions, session_entry]
        return hashing._write_json(provenance_path, existing)

    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_id": run_id or f"{species}-{public_backend}-{public_algorithm.lower()}-{uuid.uuid4().hex[:12]}",
        "captured_at": captured_timestamp,
        "sessions": [{"session_token": _PROCESS_SESSION_TOKEN, "started_at": captured_timestamp}],
        "species": species,
        "algorithm": public_algorithm,
        "backend": public_backend,
        "training_seed": seed,
        "seed_roles": normalized_seed_roles,
        "evaluation_protocols": evaluation_protocols,
        "evaluation_seeds": evaluation_seed_list,
        "evaluation_episodes": evaluation_episodes,
        "parallel_envs": parallel_envs,
        "hardware": hardware,
        "python_version": current_python_version,
        "platform": current_platform,
        "dependency_versions": current_dependency_versions,
        "plant_identity": normalized_plant_identity,
        "model_revision_status": "historical",
        "verification_status": "unverified",
        "model_hash": None,
        "config_hash": None,
        **current_repository_state,
    }
    return hashing._write_json(provenance_path, provenance)


def load_provenance(run_dir: str | Path) -> dict[str, Any]:
    """Load the captured provenance for *run_dir*."""
    path = Path(run_dir) / DEFAULT_PROVENANCE_NAME
    if not path.exists():
        raise ResultBundleError(f"missing {DEFAULT_PROVENANCE_NAME}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultBundleError(f"cannot read provenance file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultBundleError(f"provenance file must contain an object: {path}")
    return value


def update_provenance(run_dir: str | Path, **updates: Any) -> Path:
    """Update finalization fields while preserving the run-time identity."""
    run_path = Path(run_dir)
    provenance = load_provenance(run_path)
    forbidden = set(updates) - _FINALIZATION_PROVENANCE_FIELDS
    if forbidden:
        raise ResultBundleError(
            f"only finalization provenance fields may be updated; cannot replace captured fields: {sorted(forbidden)}"
        )
    provenance.update(updates)
    return hashing._write_json(run_path / DEFAULT_PROVENANCE_NAME, provenance)
