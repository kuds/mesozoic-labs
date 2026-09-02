"""The artifact manifest, written last and verified first.

Because the manifest is the final write, its presence means the copy finished;
re-hashing against it detects a truncated transfer or a later edit."""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from . import hashing
from .constants import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    DEFAULT_MANIFEST_NAME,
)
from .errors import ResultBundleError

# Imported by name, not through the module: `provenance` is also used as a local
# variable here, so `provenance.x()` would shadow it. Patch these at this module.
from .provenance import load_provenance

_logger = logging.getLogger(__name__)

#: OS and tooling litter that carries no bundle content: Finder's
#: ``.DS_Store``, AppleDouble ``._<name>`` resource forks, NFS silly-rename
#: ``.nfs*`` leftovers, Explorer's ``Thumbs.db``, KDE's ``.directory``, and
#: anything under Jupyter's ``.ipynb_checkpoints/``.  Both the writer and the
#: verifier ignore these, so opening a bundle in a file browser can neither
#: fail its save nor turn a complete bundle into canonical-conflict.  Every
#: OTHER dot-file is content: declared and hashed when the bundle is built,
#: undeclared if it appears later — except the ``.<name>.tmp`` temporaries of
#: an interrupted atomic write, which the writer discards before hashing and
#: the verifier rejects.
_IGNORED_LITTER_NAMES = (".DS_Store", "._*", ".nfs*", "Thumbs.db", ".directory")
_IGNORED_LITTER_DIRECTORIES = (".ipynb_checkpoints",)
_STALE_TEMPORARY_NAME = ".*.tmp"


def _is_ignored_litter(path: Path, run_path: Path) -> bool:
    """Whether *path* (under *run_path*) is litter neither hashed nor rejected."""
    relative = path.relative_to(run_path)
    if any(part in _IGNORED_LITTER_DIRECTORIES for part in relative.parts[:-1]):
        return True
    return any(fnmatch.fnmatchcase(relative.name, pattern) for pattern in _IGNORED_LITTER_NAMES)


def _discard_stale_temporaries(run_path: Path) -> list[Path]:
    """Remove ``.<name>.tmp`` leftovers from interrupted :func:`hashing._write_json` calls.

    A runtime that dies between the temporary write and the rename leaves the
    ``.tmp`` beside the artifact it was replacing.  Verification rejects it as
    an undeclared file, so it is cleared here — before hashing — rather than
    failing every later audit of an otherwise sound bundle.
    """
    removed: list[Path] = []
    for path in sorted(run_path.rglob(_STALE_TEMPORARY_NAME)):
        if not path.is_file():
            continue
        path.unlink()
        removed.append(path)
        _logger.warning("discarded stale temporary file from an interrupted write: %s", path)
    return removed


def build_artifact_manifest(
    run_dir: str | Path,
    *,
    status: str,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> dict[str, Any]:
    """Hash every file in a run into a manifest mapping without writing it.

    Every file is listed except the manifest itself and the
    :data:`_IGNORED_LITTER_NAMES` litter; the writer's own stale ``.*.tmp``
    temporaries are cleared first rather than hashed, so
    :func:`verify_artifact_manifest` rejects any that appear later.
    """
    if status not in {"partial", "failed", "complete"}:
        raise ResultBundleError(f"invalid bundle status: {status!r}")
    run_path = Path(run_dir).resolve()
    provenance = load_provenance(run_path)
    manifest_path = run_path / manifest_name
    _discard_stale_temporaries(run_path)
    entries: list[dict[str, Any]] = []
    for path in sorted(run_path.rglob("*")):
        if not path.is_file() or path == manifest_path or _is_ignored_litter(path, run_path):
            continue
        relative = path.relative_to(run_path).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": hashing.sha256_file(path),
            }
        )
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "run_id": provenance.get("run_id"),
        "status": status,
        "files": entries,
    }


def write_artifact_manifest(
    run_dir: str | Path,
    *,
    status: str,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> Path:
    """Hash every file in a run and write the completion marker last."""
    manifest = build_artifact_manifest(run_dir, status=status, manifest_name=manifest_name)
    return hashing._write_json(Path(run_dir).resolve() / manifest_name, manifest)


def manifest_disagreements(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> list[str]:
    """Relative paths on which *manifest* and the run directory disagree, mutating neither.

    Declared artifacts that are missing or differ in size or hash, plus files
    the manifest does not declare — ignoring the litter and the stale
    ``.*.tmp`` temporaries exactly as :func:`build_artifact_manifest` and
    :func:`verify_artifact_manifest` do — so the result is what a rebuild
    would have to re-hash.  The bundle writer uses it to decide whether a
    complete manifest that no longer verifies may be rebuilt over (only its
    own regenerated artifacts disagree) or certifies a changed artifact.
    """
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / manifest_name
    files = manifest.get("files")
    entries = [entry for entry in files if isinstance(entry, Mapping)] if isinstance(files, list) else []
    declared: set[str] = set()
    disagreements: set[str] = set()
    for entry in entries:
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            continue
        declared.add(relative)
        path = (run_path / relative).resolve()
        try:
            path.relative_to(run_path)
        except ValueError:
            disagreements.add(relative)
            continue
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("size_bytes")
            or hashing.sha256_file(path) != entry.get("sha256")
        ):
            disagreements.add(relative)
    for path in run_path.rglob("*"):
        if not path.is_file() or path == manifest_path or _is_ignored_litter(path, run_path):
            continue
        if fnmatch.fnmatchcase(path.name, _STALE_TEMPORARY_NAME):
            continue
        relative = path.relative_to(run_path).as_posix()
        if relative not in declared:
            disagreements.add(relative)
    return sorted(disagreements)


def verify_artifact_manifest(
    run_dir: str | Path,
    *,
    reject_unlisted: bool = True,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    prospective_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify all declared artifacts and optionally reject undeclared files.

    *prospective_manifest* is verified in place of the on-disk file: the
    bundle writer validates the manifest it is about to write, so a failed
    validation never leaves a completion marker behind.
    """
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / manifest_name
    manifest: Any
    if prospective_manifest is not None:
        manifest = dict(prospective_manifest)
    else:
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
        if hashing.sha256_file(path) != entry.get("sha256"):
            raise ResultBundleError(f"artifact hash mismatch: {relative}")
        declared.add(relative)

    if reject_unlisted:
        # Everything the writer would have hashed must be declared, so a
        # leftover such as a crashed write's `.summary.json.tmp` (never
        # hashed) or a file added later surfaces here rather than riding
        # along inside an "immutable" bundle.  Litter is ignored on both
        # sides, so it can neither be declared nor be undeclared.
        actual = {
            path.relative_to(run_path).as_posix()
            for path in run_path.rglob("*")
            if path.is_file() and path != manifest_path and not _is_ignored_litter(path, run_path)
        }
        undeclared = sorted(actual - declared)
        if undeclared:
            raise ResultBundleError(f"undeclared bundle artifacts: {undeclared}")
    return manifest
