"""The artifact manifest, written last and verified first.

Because the manifest is the final write, its presence means the copy finished;
re-hashing against it detects a truncated transfer or a later edit."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    DEFAULT_MANIFEST_NAME,
)
from .errors import ResultBundleError
from .hashing import _write_json, sha256_file
from .provenance import load_provenance


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
