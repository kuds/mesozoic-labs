"""Loading reviewed plant revisions and species entries from ``configs/``."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from . import constants
from .constants import FINGERPRINT_TOOL_VERSION, PLANT_MANIFEST_SCHEMA
from .errors import PlantContractError
from .identity import PlantVersion


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


def _species_entries() -> dict[str, dict[str, Any]]:
    if not constants.SPECIES_MANIFEST_PATH.is_file():
        generated = _load_generated_manifest()
        bundled_entries = {
            str(species): {
                "id": str(species),
                "model_path": entry.get("model_path"),
                "env_entrypoint": entry.get("env_entrypoint"),
            }
            for species, entry in generated.get("plants", {}).items()
        }
        if not bundled_entries or any(
            not entry.get("model_path") or not entry.get("env_entrypoint") for entry in bundled_entries.values()
        ):
            raise PlantContractError("bundled plant manifest lacks runtime species pointers")
        return bundled_entries

    raw = _read_toml(constants.SPECIES_MANIFEST_PATH)
    entries: dict[str, dict[str, Any]] = {}
    for entry in raw.get("species", []):
        species = str(entry.get("id", ""))
        if not species or species in entries:
            raise PlantContractError(
                f"invalid or duplicate species id in {constants.SPECIES_MANIFEST_PATH.name}: {species!r}"
            )
        entries[species] = dict(entry)
    if not entries:
        raise PlantContractError("species manifest contains no species")
    return entries


def load_plant_versions() -> tuple[str, dict[str, PlantVersion]]:
    """Load human revision counters and require complete species coverage."""
    if not constants.PLANT_VERSIONS_PATH.is_file():
        generated = _load_generated_manifest()
        canonical_mujoco = str(generated.get("generated_with", {}).get("mujoco", ""))
        bundled_versions: dict[str, PlantVersion] = {}
        for species, entry in generated.get("plants", {}).items():
            policy = entry.get("policy_interface", {})
            physics = entry.get("physics", {})
            visual = entry.get("visual", {})
            try:
                bundled_versions[str(species)] = PlantVersion(
                    species=str(species),
                    physics_revision=int(physics["revision"]),
                    policy_interface_revision=int(policy["revision"]),
                    visual_revision=int(visual["revision"]),
                    observation_schema=str(policy["observation_schema"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PlantContractError(f"invalid bundled plant version for {species}: {exc}") from exc
        if not canonical_mujoco or not bundled_versions:
            raise PlantContractError("bundled plant manifest lacks canonical version metadata")
        return canonical_mujoco, bundled_versions

    raw = _read_toml(constants.PLANT_VERSIONS_PATH)
    if raw.get("schema_version") != 1:
        raise PlantContractError("plant_versions.toml schema_version must be 1")
    if raw.get("fingerprint_tool_version") != FINGERPRINT_TOOL_VERSION:
        raise PlantContractError(f"plant_versions.toml fingerprint_tool_version must be {FINGERPRINT_TOOL_VERSION}")
    canonical_mujoco = str(raw.get("canonical_mujoco_version", ""))
    if not canonical_mujoco:
        raise PlantContractError("plant_versions.toml must define canonical_mujoco_version")

    versions: dict[str, PlantVersion] = {}
    for species, value in raw.get("plants", {}).items():
        try:
            version = PlantVersion(
                species=str(species),
                physics_revision=int(value["physics_revision"]),
                policy_interface_revision=int(value["policy_interface_revision"]),
                visual_revision=int(value["visual_revision"]),
                observation_schema=str(value["observation_schema"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlantContractError(f"invalid version entry for {species}: {exc}") from exc
        if min(version.physics_revision, version.policy_interface_revision, version.visual_revision) < 1:
            raise PlantContractError(f"all plant revisions must be positive for {species}")
        if not version.observation_schema:
            raise PlantContractError(f"observation_schema must be non-empty for {species}")
        versions[species] = version

    species_ids = set(_species_entries())
    if set(versions) != species_ids:
        raise PlantContractError(
            f"plant version coverage mismatch: missing={sorted(species_ids - set(versions))}, "
            f"unknown={sorted(set(versions) - species_ids)}"
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
