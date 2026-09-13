"""Content hashes and deterministic JSON writing.

Every hash here is over bytes or canonical JSON, so the same bundle copied
through Drive hashes identically on the other side."""

from __future__ import annotations

import hashlib
import json
import numbers
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import ResultBundleError


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


def gate_config_sha256(view: Mapping[str, Any]) -> str:
    """Hash a gate-configuration view (decision D-A22) with its numerics normalised.

    *view* is the ``curriculum.gate_schema.gate_config_view`` projection —
    ``{gate_kind, gate_schema_version, thresholds}``.  Every real number
    under ``thresholds`` (bools excluded; numpy scalars included, since a
    sweep override or a loaded ``evaluations.npz`` hands those in and they
    are neither ``int`` nor ``float`` subclasses) is hashed as its
    ``float()``, so a cosmetic TOML retype (``100`` to ``100.0``) leaves the
    digest unchanged and cannot refuse a reuse (decision D-B7); everything
    else is hashed verbatim through :func:`canonical_json_sha256`.  Pure — no
    curriculum import — so ``result_bundle`` keeps its hashing-and-file-io-only
    promise.
    """
    normalised = dict(view)
    thresholds = view.get("thresholds")
    if isinstance(thresholds, Mapping):
        normalised["thresholds"] = {
            key: float(value) if isinstance(value, numbers.Real) and not isinstance(value, bool) else value
            for key, value in thresholds.items()
        }
    return canonical_json_sha256(normalised)


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Atomically write a JSON mapping with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


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
