"""Loading reviewed plant revisions and species-manifest entries from ``configs/``, by model."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Mapping

from ..model_id import ModelId
from ..species_names import _read_manifest as _read_species_manifest
from . import constants
from .constants import FINGERPRINT_TOOL_VERSION, PLANT_MANIFEST_SCHEMA
from .errors import PlantContractError
from .identity import PlantVersion

#: ``configs/plant_versions.toml`` schema: 2 since decision D-D33, one
#: ``[plants.<species>.<embodiment>]`` table per model.
PLANT_VERSIONS_SCHEMA_VERSION = 2
_REVISION_KEYS = ("physics_revision", "policy_interface_revision", "visual_revision", "observation_schema")


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PlantContractError(f"cannot read {path.relative_to(constants.REPOSITORY_ROOT)}: {exc}") from exc


def _load_generated_manifest(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = (
            constants.GENERATED_MANIFEST_PATH
            if constants.GENERATED_MANIFEST_PATH.is_file()
            else constants.BUNDLED_MANIFEST_PATH
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlantContractError(f"cannot read generated plant manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != PLANT_MANIFEST_SCHEMA:
        raise PlantContractError(f"generated plant manifest must use schema {PLANT_MANIFEST_SCHEMA}")
    return value


def _generated_plants(generated: Mapping[str, Any]) -> dict[ModelId, Mapping[str, Any]]:
    """A generated manifest's plants by model: ``plants[<species>][<embodiment>]`` (schema v2)."""
    plants = generated.get("plants", {})
    if not isinstance(plants, Mapping):
        raise PlantContractError("generated plant manifest has no plants table")
    keyed: dict[ModelId, Mapping[str, Any]] = {}
    for species, embodiments in plants.items():
        if not isinstance(embodiments, Mapping):
            raise PlantContractError(f"generated plant manifest species {species!r} is not a table of embodiments")
        for embodiment, entry in embodiments.items():
            try:
                model_id = ModelId(species, embodiment)
            except ValueError as exc:
                raise PlantContractError(f"generated plant manifest key {species!r}.{embodiment!r}: {exc}") from exc
            if not isinstance(entry, Mapping):
                raise PlantContractError(f"generated plant manifest entry for {model_id} is not a table")
            keyed[model_id] = entry
    return keyed


def _species_entries() -> dict[ModelId, dict[str, Any]]:
    """Each model's species-manifest entry (``env_entrypoint``, ``model_path``, ``training_backends``).

    Read from :data:`constants.SPECIES_MANIFEST_PATH` (schema 3) through the
    one manifest reader, ``species_names``; an installed package without
    ``configs/`` falls back to the bundled plant manifest.
    """
    if not constants.SPECIES_MANIFEST_PATH.is_file():
        generated = _load_generated_manifest()
        # The generated manifest records no training backends, so a bundled
        # entry declares none (None): the environment's own
        # supported_training_backends then stands, which is what lets an
        # SB3-only model resolve its identity from the bundle.
        bundled_entries = {
            model_id: {
                "model_path": entry.get("model_path"),
                "env_entrypoint": entry.get("env_entrypoint"),
                "training_backends": None,
            }
            for model_id, entry in _generated_plants(generated).items()
        }
        if not bundled_entries or any(
            not entry.get("model_path") or not entry.get("env_entrypoint") for entry in bundled_entries.values()
        ):
            raise PlantContractError("bundled plant manifest lacks runtime model pointers")
        return bundled_entries

    try:
        raw = _read_species_manifest(constants.SPECIES_MANIFEST_PATH)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise PlantContractError(
            f"cannot read {constants.SPECIES_MANIFEST_PATH.relative_to(constants.REPOSITORY_ROOT)}: {exc}"
        ) from exc
    return {
        ModelId(species, embodiment): dict(entry)
        for species, table in raw["species"].items()
        for embodiment, entry in table["embodiments"].items()
    }


def load_plant_versions() -> tuple[str, dict[ModelId, PlantVersion]]:
    """Load human revision counters and require complete model coverage.

    ``plant_versions.toml`` schema 2 (decision D-D33) holds one table per
    model, ``[plants.<species>.<embodiment>]``; a species table holding
    revision keys itself (the schema-1 shape) is refused.
    """
    if not constants.PLANT_VERSIONS_PATH.is_file():
        generated = _load_generated_manifest()
        canonical_mujoco = str(generated.get("generated_with", {}).get("mujoco", ""))
        bundled_versions: dict[ModelId, PlantVersion] = {}
        for model_id, entry in _generated_plants(generated).items():
            policy = entry.get("policy_interface", {})
            physics = entry.get("physics", {})
            visual = entry.get("visual", {})
            try:
                bundled_versions[model_id] = PlantVersion(
                    model_id=model_id,
                    physics_revision=int(physics["revision"]),
                    policy_interface_revision=int(policy["revision"]),
                    visual_revision=int(visual["revision"]),
                    observation_schema=str(policy["observation_schema"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PlantContractError(f"invalid bundled plant version for {model_id}: {exc}") from exc
        if not canonical_mujoco or not bundled_versions:
            raise PlantContractError("bundled plant manifest lacks canonical version metadata")
        return canonical_mujoco, bundled_versions

    raw = _read_toml(constants.PLANT_VERSIONS_PATH)
    if raw.get("schema_version") != PLANT_VERSIONS_SCHEMA_VERSION:
        raise PlantContractError(f"plant_versions.toml schema_version must be {PLANT_VERSIONS_SCHEMA_VERSION}")
    if raw.get("fingerprint_tool_version") != FINGERPRINT_TOOL_VERSION:
        raise PlantContractError(f"plant_versions.toml fingerprint_tool_version must be {FINGERPRINT_TOOL_VERSION}")
    canonical_mujoco = str(raw.get("canonical_mujoco_version", ""))
    if not canonical_mujoco:
        raise PlantContractError("plant_versions.toml must define canonical_mujoco_version")

    versions: dict[ModelId, PlantVersion] = {}
    for species, embodiments in raw.get("plants", {}).items():
        revision_keys = sorted(set(embodiments) & set(_REVISION_KEYS)) if isinstance(embodiments, dict) else []
        if not isinstance(embodiments, dict) or revision_keys:
            raise PlantContractError(
                f"[plants.{species}] must hold one [plants.{species}.<embodiment>] table per model, "
                f"not revision keys {revision_keys} (plant_versions.toml schema 2, decision D-D33)"
            )
        for embodiment, value in embodiments.items():
            try:
                model_id = ModelId(species, embodiment)
            except ValueError as exc:
                raise PlantContractError(f"invalid plant table [plants.{species}.{embodiment}]: {exc}") from exc
            try:
                version = PlantVersion(
                    model_id=model_id,
                    physics_revision=int(value["physics_revision"]),
                    policy_interface_revision=int(value["policy_interface_revision"]),
                    visual_revision=int(value["visual_revision"]),
                    observation_schema=str(value["observation_schema"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PlantContractError(f"invalid version entry for {model_id}: {exc}") from exc
            if min(version.physics_revision, version.policy_interface_revision, version.visual_revision) < 1:
                raise PlantContractError(f"all plant revisions must be positive for {model_id}")
            if not version.observation_schema:
                raise PlantContractError(f"observation_schema must be non-empty for {model_id}")
            versions[model_id] = version

    models = set(_species_entries())
    if set(versions) != models:
        raise PlantContractError(
            f"plant version coverage mismatch: missing={sorted(map(str, models - set(versions)))}, "
            f"unknown={sorted(map(str, set(versions) - models))}"
        )
    return canonical_mujoco, versions


def _resolve_repo_path(relative_path: str, *, field: str) -> Path:
    path = (constants.REPOSITORY_ROOT / relative_path).resolve()
    try:
        path.relative_to(constants.REPOSITORY_ROOT)
    except ValueError as exc:
        raise PlantContractError(f"{field} must stay inside the repository: {relative_path}") from exc
    if not path.is_file():
        raise PlantContractError(f"{field} does not exist: {relative_path}")
    return path
