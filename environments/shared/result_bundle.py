"""Portable result-bundle provenance, hashing, and validation helpers.

Training runs are written to Google Drive by the public Colab notebooks and
later promoted into the repository.  This module keeps that hand-off
deterministic: provenance is captured when a run starts, artifact paths are
relative to the run directory, and a manifest written last detects incomplete
copies or later modification.

The result schema itself lives in :mod:`environments.shared.result_schema`.
Drive bundles may be partial, but repository promotion requires a complete
three-stage summary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_NAME = "artifact_manifest.json"
DEFAULT_PROVENANCE_NAME = "provenance.json"

_FINALIZATION_PROVENANCE_FIELDS = frozenset(
    {
        "backend_version",
        "config_hash",
        "model_hash",
        "model_revision_status",
        "selected_checkpoints",
        "selected_model_path",
        "verification_status",
    }
)

_DEPENDENCY_PACKAGES = {
    "mesozoic_labs": "mesozoic-labs",
    "mujoco": "mujoco",
    "mujoco_mjx": "mujoco-mjx",
    "gymnasium": "gymnasium",
    "numpy": "numpy",
    "stable_baselines3": "stable-baselines3",
    "jax": "jax",
    "flax": "flax",
    "optax": "optax",
}


class ResultBundleError(ValueError):
    """Raised when a result bundle is incomplete or internally inconsistent."""


def canonical_algorithm(algorithm: str) -> str:
    """Return the public algorithm label independent of training backend."""
    normalized = algorithm.strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    if normalized in {"ppo", "jax_ppo", "jax_mjx_ppo"}:
        return "PPO"
    if normalized == "sac":
        return "SAC"
    raise ResultBundleError(f"unsupported training algorithm: {algorithm!r}")


def canonical_backend(algorithm: str, backend: str | None = None) -> str:
    """Return the canonical training-backend identifier."""
    if backend is not None:
        normalized = backend.strip().lower().replace("_", "-")
        aliases = {
            "jax": "jax-mjx",
            "jax-mjx": "jax-mjx",
            "stable-baselines3": "stable-baselines3",
            "sb3": "stable-baselines3",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ResultBundleError(f"unsupported training backend: {backend!r}") from exc
    return "jax-mjx" if "jax" in algorithm.lower() else "stable-baselines3"


def _normalize_plant_identity(
    value: Mapping[str, Any] | None,
    *,
    species: str | None = None,
) -> dict[str, Any] | None:
    """Return a canonical plant-identity mapping or raise a bundle error."""
    if value is None:
        return None
    try:
        from .plant_contract import PlantContractError, PlantIdentity

        identity = PlantIdentity.from_mapping(value)
    except (PlantContractError, KeyError, TypeError, ValueError) as exc:
        raise ResultBundleError(f"invalid plant identity: {exc}") from exc
    if species is not None and identity.species != species:
        raise ResultBundleError(f"plant identity species mismatch: expected {species!r}, found {identity.species!r}")
    normalized = identity.to_dict()
    if dict(value) != normalized:
        raise ResultBundleError("plant identity must use the canonical v1 field types and values")
    return normalized


def _portable_relative_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ResultBundleError(f"{field} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ResultBundleError(f"{field} must be a normalized relative POSIX path: {value!r}")
    return path


def sha256_file(path: str | Path) -> str:
    """Return a portable ``sha256:<hex>`` digest for *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value independently of formatting."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Atomically write a JSON mapping with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


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
                untracked.append({"path": relative, "sha256": sha256_file(candidate)})
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

    Repeated calls are idempotent when the identifying inputs match.  A reused
    run directory with a different species, algorithm, backend, or training
    seed is rejected instead of silently mixing artifacts.
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
        repository = Path(__file__).resolve().parents[2]
    else:
        repository = Path(repository_root).resolve()
    current_repository_state = _repository_state(repository)
    current_python_version = platform.python_version()
    current_platform = platform.platform()
    current_dependency_versions = _dependency_versions()

    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected = {
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
            "plant_identity": normalized_plant_identity,
            "python_version": current_python_version,
            "platform": current_platform,
            "dependency_versions": current_dependency_versions,
            "repository_commit": current_repository_state["repository_commit"],
            "repository_dirty": current_repository_state["repository_dirty"],
            "repository_patch_sha256": current_repository_state["repository_patch_sha256"],
        }
        if run_id is not None:
            expected["run_id"] = run_id
        mismatches = {key: (existing.get(key), value) for key, value in expected.items() if existing.get(key) != value}
        if mismatches:
            raise ResultBundleError(f"run directory already belongs to a different run: {mismatches}")
        return provenance_path

    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_id": run_id or f"{species}-{public_backend}-{public_algorithm.lower()}-{uuid.uuid4().hex[:12]}",
        "captured_at": captured_timestamp,
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
    return _write_json(provenance_path, provenance)


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
    return _write_json(run_path / DEFAULT_PROVENANCE_NAME, provenance)


def aggregate_file_hash(paths: Iterable[str | Path], *, root: str | Path) -> str | None:
    """Return a deterministic aggregate hash for existing files under *root*."""
    root_path = Path(root).resolve()
    entries: list[dict[str, str]] = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = root_path / path
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ResultBundleError(f"artifact lies outside run directory: {resolved}") from exc
        entries.append({"path": relative, "sha256": sha256_file(resolved)})
    if not entries:
        return None
    entries.sort(key=lambda item: item["path"])
    return canonical_json_sha256(entries)


def write_artifact_manifest(
    run_dir: str | Path,
    *,
    status: str,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> Path:
    """Hash every file in a run and write the completion marker last."""
    if status not in {"partial", "failed", "complete"}:
        raise ResultBundleError(f"invalid bundle status: {status!r}")
    run_path = Path(run_dir).resolve()
    provenance = load_provenance(run_path)
    manifest_path = run_path / manifest_name
    entries: list[dict[str, Any]] = []
    for path in sorted(run_path.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.startswith("."):
            continue
        relative = path.relative_to(run_path).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "run_id": provenance.get("run_id"),
        "status": status,
        "files": entries,
    }
    return _write_json(manifest_path, manifest)


def verify_artifact_manifest(
    run_dir: str | Path,
    *,
    reject_unlisted: bool = True,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> dict[str, Any]:
    """Verify all declared artifacts and optionally reject undeclared files."""
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / manifest_name
    if not manifest_path.exists():
        raise ResultBundleError(f"missing {manifest_name}: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultBundleError(f"cannot read artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ResultBundleError(f"unsupported artifact manifest schema: {manifest_path}")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"].strip():
        raise ResultBundleError(f"artifact manifest run_id must be a non-empty string: {manifest_path}")
    if manifest.get("status") not in {"partial", "failed", "complete"}:
        raise ResultBundleError(f"invalid artifact manifest status: {manifest.get('status')!r}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ResultBundleError(f"artifact manifest files must be a list: {manifest_path}")

    declared: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ResultBundleError("artifact manifest entries must be objects")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or relative in declared:
            raise ResultBundleError(f"invalid or duplicate manifest path: {relative!r}")
        portable_path = PurePosixPath(relative)
        if portable_path.is_absolute() or any(part in {"", ".", ".."} for part in portable_path.parts):
            raise ResultBundleError(f"manifest path must be a normalized relative POSIX path: {relative!r}")
        path = (run_path / relative).resolve()
        try:
            path.relative_to(run_path)
        except ValueError as exc:
            raise ResultBundleError(f"manifest path escapes run directory: {relative}") from exc
        if not path.is_file():
            raise ResultBundleError(f"manifest artifact is missing: {relative}")
        if path.stat().st_size != entry.get("size_bytes"):
            raise ResultBundleError(f"artifact size mismatch: {relative}")
        if sha256_file(path) != entry.get("sha256"):
            raise ResultBundleError(f"artifact hash mismatch: {relative}")
        declared.add(relative)

    if reject_unlisted:
        actual = {
            path.relative_to(run_path).as_posix()
            for path in run_path.rglob("*")
            if path.is_file() and path != manifest_path and not path.name.startswith(".")
        }
        undeclared = sorted(actual - declared)
        if undeclared:
            raise ResultBundleError(f"undeclared bundle artifacts: {undeclared}")
    return manifest


def _optional_csv_number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ResultBundleError(f"invalid numeric CSV value: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ResultBundleError(f"non-finite numeric CSV value: {value!r}")
    return numeric


def _optional_csv_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ResultBundleError(f"invalid boolean CSV value: {value!r}")


def compare_summary_to_csv(
    summary: Mapping[str, Any],
    csv_path: str | Path,
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """Return contradictions between canonical summary and derived CSV."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    rows_by_stage: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for row in rows:
        stage = (row.get("stage") or "").strip()
        if stage in rows_by_stage:
            problems.append(f"CSV contains duplicate stage {stage}")
        rows_by_stage[stage] = row

    summary_stages = summary.get("stages")
    if not isinstance(summary_stages, Mapping):
        return ["summary stages must be an object"]
    if set(rows_by_stage) != set(summary_stages):
        problems.append(f"stage sets differ: summary={sorted(summary_stages)} csv={sorted(rows_by_stage)}")

    numeric_fields = {
        "timesteps": "timesteps",
        "best_eval_reward": "best_mean_reward",
        "best_eval_std": "best_mean_reward_std",
        "best_eval_step": "best_eval_step",
        "final_eval_reward": "last_mean_reward",
        "final_eval_std": "last_mean_reward_std",
        "avg_episode_length": "last_mean_episode_length",
        "avg_episode_length_std": "last_mean_episode_length_std",
        "avg_forward_vel": "mean_forward_vel",
        "avg_forward_vel_std": "std_forward_vel",
        "mean_distance_traveled": "mean_distance_traveled",
        "mean_success_rate": "mean_success_rate",
        "training_time_seconds": "training_duration_seconds",
        "selected_model_reward": "selected_model_mean_reward",
        "selected_model_reward_std": "selected_model_reward_std",
        "selected_model_episode_length": "selected_model_mean_episode_length",
        "selected_model_episode_length_std": "selected_model_episode_length_std",
        "selected_model_forward_vel": "selected_model_mean_forward_vel",
        "selected_model_forward_vel_std": "selected_model_mean_forward_vel_std",
        "selected_model_distance": "selected_model_mean_distance",
        "selected_model_success_rate": "selected_model_success_rate",
    }
    for stage, stage_summary_value in summary_stages.items():
        if stage not in rows_by_stage or not isinstance(stage_summary_value, Mapping):
            continue
        row = rows_by_stage[stage]
        for summary_key, csv_key in numeric_fields.items():
            expected = stage_summary_value.get(summary_key)
            actual = _optional_csv_number(row.get(csv_key))
            if expected is None and actual is None:
                continue
            if expected is None or actual is None or abs(float(expected) - actual) > tolerance:
                problems.append(f"stage {stage} {summary_key}/{csv_key} differs: summary={expected!r} csv={actual!r}")
        expected_passed = stage_summary_value.get("stage_passed")
        actual_passed = _optional_csv_bool(row.get("stage_passed"))
        if expected_passed != actual_passed:
            problems.append(f"stage {stage} stage_passed differs: summary={expected_passed!r} csv={actual_passed!r}")
        expected_publication_gate = stage_summary_value.get("publication_gate_passed")
        actual_publication_gate = _optional_csv_bool(row.get("publication_gate_passed"))
        if expected_publication_gate != actual_publication_gate:
            problems.append(
                f"stage {stage} publication_gate_passed differs: "
                f"summary={expected_publication_gate!r} csv={actual_publication_gate!r}"
            )

        provenance = summary.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        identities: dict[str, Any] = {
            "schema_version": summary.get("schema_version"),
            "species": summary.get("species"),
            "algorithm": summary.get("algorithm"),
            "backend": summary.get("backend"),
            "seed": summary.get("seed"),
            "run_id": summary.get("run_id"),
            "repository_commit": provenance.get("repository_commit"),
            "config_hash": provenance.get("config_hash"),
            "model_hash": provenance.get("model_hash"),
        }
        plant_identity = summary.get("plant_identity")
        if isinstance(plant_identity, Mapping):
            identities.update({f"plant_{key}": value for key, value in plant_identity.items()})
        for key, expected in identities.items():
            if expected is None:
                continue
            if key not in row:
                problems.append(f"stage {stage} CSV is missing required column {key}")
                continue
            actual_value: Any = row[key]
            if key in {"schema_version", "seed"} and actual_value != "":
                try:
                    actual_value = int(actual_value)
                except ValueError as exc:
                    raise ResultBundleError(f"invalid integer CSV value for {key}: {actual_value!r}") from exc
            if str(actual_value).lower() != str(expected).lower():
                problems.append(f"stage {stage} {key} differs: summary={expected!r} csv={actual_value!r}")
    return problems


def _evaluation_evidence_aggregates(
    evidence_path: Path,
    *,
    checkpoint_label: str,
    expected_episodes: int,
    evaluation_seeds: Sequence[int],
    stage: int,
) -> dict[str, float]:
    """Parse one fixed-seed episode file and return publication aggregates."""
    required_columns = {
        "episode",
        "evaluation_seed",
        "checkpoint",
        "reward",
        "length",
        "mean_forward_velocity",
        "distance_traveled",
        "success",
    }
    if not evidence_path.is_file():
        raise ResultBundleError(f"missing {checkpoint_label} evaluation evidence: {evidence_path}")
    with evidence_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        raise ResultBundleError(
            f"{checkpoint_label} evaluation evidence for stage {stage} is missing columns: {missing_columns}"
        )
    if len(rows) != expected_episodes:
        raise ResultBundleError(
            f"{checkpoint_label} evaluation evidence for stage {stage} has "
            f"{len(rows)} rows; expected {expected_episodes}"
        )

    rewards: list[float] = []
    lengths: list[float] = []
    forward_velocities: list[float] = []
    distances: list[float] = []
    successes: list[float] = []
    recorded_seed: int | None = None
    for expected_episode, row in enumerate(rows, start=1):
        try:
            episode = int(row.get("episode", ""))
            seed = int(row.get("evaluation_seed", ""))
            length = int(row.get("length", ""))
        except (TypeError, ValueError) as exc:
            raise ResultBundleError(
                f"invalid integer in {checkpoint_label} evaluation evidence for stage {stage}"
            ) from exc
        if episode != expected_episode:
            raise ResultBundleError(
                f"{checkpoint_label} evaluation evidence for stage {stage} must number episodes consecutively"
            )
        if seed not in evaluation_seeds:
            raise ResultBundleError(
                f"{checkpoint_label} evaluation seed {seed} for stage {stage} is not recorded in provenance"
            )
        if recorded_seed is None:
            recorded_seed = seed
        elif seed != recorded_seed:
            raise ResultBundleError(f"{checkpoint_label} evaluation evidence for stage {stage} mixes evaluation seeds")
        if row.get("checkpoint") != checkpoint_label:
            raise ResultBundleError(
                f"{checkpoint_label} evaluation evidence for stage {stage} has the wrong checkpoint label"
            )
        if length <= 0:
            raise ResultBundleError(f"{checkpoint_label} evaluation episode length for stage {stage} must be positive")
        reward = _optional_csv_number(row.get("reward"))
        forward_velocity = _optional_csv_number(row.get("mean_forward_velocity"))
        distance = _optional_csv_number(row.get("distance_traveled"))
        success = _optional_csv_bool(row.get("success"))
        if reward is None or forward_velocity is None or distance is None or success is None:
            raise ResultBundleError(f"{checkpoint_label} evaluation evidence for stage {stage} contains blank values")
        rewards.append(reward)
        lengths.append(float(length))
        forward_velocities.append(forward_velocity)
        distances.append(distance)
        successes.append(float(success))

    def _mean(values: Sequence[float]) -> float:
        return math.fsum(values) / len(values)

    def _population_std(values: Sequence[float]) -> float:
        mean_value = _mean(values)
        return math.sqrt(math.fsum((value - mean_value) ** 2 for value in values) / len(values))

    assert recorded_seed is not None
    return {
        "evaluation_seed": float(recorded_seed),
        "reward": _mean(rewards),
        "reward_std": _population_std(rewards),
        "episode_length": _mean(lengths),
        "episode_length_std": _population_std(lengths),
        "forward_vel": _mean(forward_velocities),
        "forward_vel_std": _population_std(forward_velocities),
        "distance": _mean(distances),
        "success_rate": _mean(successes),
    }


def _compare_evaluation_aggregates(
    stage_summary: Mapping[str, Any],
    aggregates: Mapping[str, float],
    *,
    stage: int,
    label: str,
    metric_map: Mapping[str, tuple[str, float]],
) -> None:
    for summary_metric, (aggregate_metric, tolerance) in metric_map.items():
        expected = stage_summary.get(summary_metric)
        if expected is None:
            raise ResultBundleError(
                f"summary is missing {label} evaluation aggregate {summary_metric} for stage {stage}"
            )
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            raise ResultBundleError(f"summary {summary_metric} for stage {stage} must be numeric")
        actual = aggregates[aggregate_metric]
        if not math.isfinite(float(expected)) or abs(float(expected) - actual) > tolerance:
            raise ResultBundleError(
                f"{label} evaluation aggregate for stage {stage} {summary_metric} differs: "
                f"summary={expected!r} evidence={actual!r}"
            )


def validate_evaluation_evidence(
    run_dir: str | Path,
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Bind selected and terminal-policy episode rows to canonical claims."""
    run_path = Path(run_dir).resolve()
    expected_episodes = provenance.get("evaluation_episodes")
    evaluation_seeds = provenance.get("evaluation_seeds")
    if not isinstance(expected_episodes, int) or isinstance(expected_episodes, bool) or expected_episodes <= 0:
        raise ResultBundleError("provenance evaluation_episodes must be a positive integer")
    if not isinstance(evaluation_seeds, list) or not evaluation_seeds:
        raise ResultBundleError("provenance evaluation_seeds must be a non-empty list")
    seed_roles = provenance.get("seed_roles")
    if not isinstance(seed_roles, Mapping):
        raise ResultBundleError("provenance seed_roles must be an object")
    publication_evaluation_seed = seed_roles.get("publication_evaluation")
    if (
        not isinstance(publication_evaluation_seed, int)
        or isinstance(publication_evaluation_seed, bool)
        or publication_evaluation_seed < 0
    ):
        raise ResultBundleError("provenance publication_evaluation seed role must be a non-negative integer")
    stages = summary.get("stages")
    if not isinstance(stages, Mapping):
        raise ResultBundleError("summary stages must be an object")

    selected_metric_map = {
        "selected_model_reward": ("reward", 0.0051),
        "selected_model_reward_std": ("reward_std", 0.0051),
        "selected_model_episode_length": ("episode_length", 0.0501),
        "selected_model_episode_length_std": ("episode_length_std", 0.0501),
        "selected_model_forward_vel": ("forward_vel", 0.0051),
        "selected_model_forward_vel_std": ("forward_vel_std", 0.0051),
        "selected_model_distance": ("distance", 0.0051),
        "selected_model_success_rate": ("success_rate", 0.0051),
    }
    final_metric_map = {
        "final_eval_reward": ("reward", 0.0051),
        "final_eval_std": ("reward_std", 0.0051),
        "avg_episode_length": ("episode_length", 0.0501),
        "avg_episode_length_std": ("episode_length_std", 0.0501),
        "avg_forward_vel": ("forward_vel", 0.0051),
        "avg_forward_vel_std": ("forward_vel_std", 0.0051),
        "mean_distance_traveled": ("distance", 0.0051),
        "mean_success_rate": ("success_rate", 0.0051),
    }
    for stage in (1, 2, 3):
        stage_summary = stages.get(str(stage))
        if not isinstance(stage_summary, Mapping):
            raise ResultBundleError(f"summary is missing stage {stage}")
        selected_aggregates = _evaluation_evidence_aggregates(
            run_path / f"stage{stage}" / "evaluation_selected.csv",
            checkpoint_label="selected",
            expected_episodes=expected_episodes,
            evaluation_seeds=evaluation_seeds,
            stage=stage,
        )
        if int(selected_aggregates["evaluation_seed"]) != publication_evaluation_seed:
            raise ResultBundleError(f"stage {stage} selected evidence does not use the publication_evaluation seed")
        _compare_evaluation_aggregates(
            stage_summary,
            selected_aggregates,
            stage=stage,
            label="selected",
            metric_map=selected_metric_map,
        )
        config_path = run_path / f"stage{stage}" / "stage_config.json"
        try:
            config_value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read publication gate config for stage {stage}: {exc}") from exc
        if not isinstance(config_value, Mapping):
            raise ResultBundleError(f"publication gate config for stage {stage} must be an object")
        curriculum = config_value.get("curriculum", config_value.get("curriculum_kwargs", {}))
        if not isinstance(curriculum, Mapping):
            raise ResultBundleError(f"publication gate config for stage {stage} must contain a curriculum object")
        publication_thresholds = {
            "min_avg_reward": "reward",
            "min_avg_episode_length": "episode_length",
            "min_avg_forward_vel": "forward_vel",
            "min_success_rate": "success_rate",
        }
        for threshold_name, aggregate_name in publication_thresholds.items():
            threshold_value = curriculum.get(threshold_name)
            if threshold_value is None:
                continue
            if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
                raise ResultBundleError(
                    f"publication gate threshold {threshold_name} for stage {stage} must be numeric"
                )
            threshold = float(threshold_value)
            if not math.isfinite(threshold):
                raise ResultBundleError(f"publication gate threshold {threshold_name} for stage {stage} must be finite")
            actual = selected_aggregates[aggregate_name]
            if actual < threshold:
                raise ResultBundleError(
                    f"stage {stage} publication gate fails {threshold_name}: "
                    f"evidence={actual:.6g} threshold={threshold:.6g}"
                )
        final_aggregates = _evaluation_evidence_aggregates(
            run_path / f"stage{stage}" / "evaluation_final.csv",
            checkpoint_label="final",
            expected_episodes=expected_episodes,
            evaluation_seeds=evaluation_seeds,
            stage=stage,
        )
        if int(final_aggregates["evaluation_seed"]) != publication_evaluation_seed:
            raise ResultBundleError(f"stage {stage} final evidence does not use the publication_evaluation seed")
        _compare_evaluation_aggregates(
            stage_summary,
            final_aggregates,
            stage=stage,
            label="final",
            metric_map=final_metric_map,
        )


def audit_result_bundle(
    run_dir: str | Path,
    *,
    verify_hashes: bool = True,
    reject_unlisted: bool = True,
) -> dict[str, Any]:
    """Classify a canonical or legacy Drive result directory without mutating it."""
    from .result_schema import (
        ResultSchemaError,
        validate_captured_provenance,
        validate_result_summary,
    )

    run_path = Path(run_dir).resolve()
    summary_path = run_path / "summary.json"
    csv_path = run_path / "collected_results.csv"
    provenance_path = run_path / DEFAULT_PROVENANCE_NAME
    manifest_path = run_path / DEFAULT_MANIFEST_NAME
    plant_path = run_path / "plant_identity.json"
    canonical = provenance_path.exists() or manifest_path.exists()
    warnings: list[str] = []
    errors: list[str] = []
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

    if manifest_path.exists():
        try:
            if verify_hashes:
                manifest = verify_artifact_manifest(run_path, reject_unlisted=reject_unlisted)
            else:
                loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
            errors.extend(compare_summary_to_csv(summary, csv_path))
        except (OSError, ResultBundleError, ValueError) as exc:
            errors.append(str(exc))
    elif summary is None and csv_path.exists():
        warnings.append("CSV-only run cannot be promoted without a canonical summary")

    manifest_status = manifest.get("status") if manifest else None
    declared_paths = (
        {
            entry.get("path")
            for entry in manifest.get("files", [])
            if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
        }
        if manifest
        else set()
    )

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
                required_paths.update(
                    {
                        "summary.json",
                        "plant_identity.json",
                        "stage1/stage_config.json",
                        "stage1/evaluation_final.csv",
                        "stage1/evaluation_selected.csv",
                        "stage2/stage_config.json",
                        "stage2/evaluation_final.csv",
                        "stage2/evaluation_selected.csv",
                        "stage3/stage_config.json",
                        "stage3/evaluation_final.csv",
                        "stage3/evaluation_selected.csv",
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
                    validate_evaluation_evidence(run_path, summary, provenance)
                except (OSError, ResultBundleError) as exc:
                    errors.append(str(exc))
            if not plant_path.is_file():
                errors.append("complete bundle is missing plant_identity.json")

            config_paths = [run_path / f"stage{stage}" / "stage_config.json" for stage in (1, 2, 3)]
            if all(path.is_file() for path in config_paths):
                actual_config_hash = aggregate_file_hash(config_paths, root=run_path)
                if actual_config_hash != provenance.get("config_hash"):
                    errors.append(
                        "resolved stage config hash does not match provenance: "
                        f"{actual_config_hash!r} != {provenance.get('config_hash')!r}"
                    )
                for stage, config_path in enumerate(config_paths, start=1):
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
                    actual_model_hash = sha256_file(selected_model)
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
                                actual_hash = sha256_file(artifact)
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
    }


def validate_result_bundle(
    run_dir: str | Path,
    *,
    require_complete: bool = True,
    reject_unlisted: bool = True,
) -> dict[str, Any]:
    """Validate a canonical Drive bundle and raise on conflicts."""
    report = audit_result_bundle(run_dir, reject_unlisted=reject_unlisted)
    allowed = {"canonical-valid"} if require_complete else {"canonical-valid", "partial", "failed"}
    if report["status"] not in allowed:
        details = "; ".join([*report["errors"], *report["warnings"]])
        raise ResultBundleError(f"result bundle is {report['status']}: {details}")
    return report
