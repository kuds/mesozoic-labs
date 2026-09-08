"""Presentation names and stable species IDs from the versioned manifest.

Full names are display labels. Config paths, checkpoints and result schemas
continue to use the existing IDs (for example, ``trex``).
"""

from __future__ import annotations

import re
import tomllib
from functools import cache
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "species_manifest.toml"


@cache
def _manifest() -> dict[str, Any]:
    with _MANIFEST_PATH.open("rb") as source:
        return tomllib.load(source)


def species_display_names(*, include_prototypes: bool = False, backend: str | None = None) -> dict[str, str]:
    """Return IDs and full labels in manifest display order.

    By default only registered training species are returned. Model-only
    prototypes can be included for presentation without making them trainable.
    """
    entries = sorted(_manifest()["species"], key=lambda entry: entry["display_order"])
    if backend is not None:
        if backend not in ("stable-baselines3", "jax-mjx"):
            raise ValueError(f"Unknown training backend: {backend!r}")
        entries = [
            entry for entry in entries if backend in entry.get("training_backends", ["stable-baselines3", "jax-mjx"])
        ]
    if include_prototypes:
        entries += _manifest().get("model_prototypes", [])
    return {entry["id"]: entry["display_name"] for entry in entries}


def _normalize(name: str) -> str:
    return re.sub(r"[\s_-]+", "", name.casefold())


def resolve_species_id(name: str) -> str:
    """Resolve a full name, ID or legacy alias, ignoring case and separators."""
    key = _normalize(name)
    for entry in [*_manifest()["species"], *_manifest().get("model_prototypes", [])]:
        names = [entry["id"], entry["display_name"], *entry.get("aliases", [])]
        if any(key == _normalize(candidate) for candidate in names):
            return str(entry["id"])
    available = ", ".join(species_display_names(include_prototypes=True).values())
    raise ValueError(f"Unknown species {name!r}. Available names: {available}")


def species_display_name(name: str) -> str:
    """Expand known species labels, preserving custom labels in generic plots."""
    try:
        species_id = resolve_species_id(name)
    except ValueError:
        return name
    return species_display_names(include_prototypes=True)[species_id]
