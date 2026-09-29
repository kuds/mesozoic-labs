"""Generated plant manifest: build, render, write, and check.

``configs/plant_manifest.generated.json`` is the committed baseline that makes
an unreviewed physics or policy-interface change fail CI."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, cast

import mujoco
import numpy as np

from . import constants, physics_layer, policy_layer, source_layer, visual_layer
from .constants import (
    FINGERPRINT_TOOL_VERSION,
    PHYSICS_SCHEMA,
    PLANT_MANIFEST_SCHEMA,
    POLICY_INTERFACE_SCHEMA,
    PORTABLE_FLOAT_SIGNIFICANT_DIGITS,
    SOURCE_SCHEMA,
    VISUAL_SCHEMA,
)
from .digests import _semantic_digest
from .errors import PlantContractError
from .identity import PlantIdentity, PlantVersion

# Imported by name, not through the module: `versions` is also used as a local
# variable here, so `versions.x()` would shadow it. Patch these at this module.
from .versions import (
    _load_generated_manifest,
    _resolve_repo_path,
    _species_entries,
    load_plant_versions,
)

_logger = logging.getLogger(__name__)


def _load_environment(entrypoint: str) -> type[Any]:
    module_name, separator, class_name = entrypoint.partition(":")
    if not separator:
        raise PlantContractError(f"invalid environment entrypoint: {entrypoint}")
    try:
        return cast(type[Any], getattr(importlib.import_module(module_name), class_name))
    except (ImportError, AttributeError) as exc:
        raise PlantContractError(f"cannot import environment entrypoint {entrypoint}: {exc}") from exc


def fingerprint_model_layers(
    model: mujoco.MjModel,
    env: Any,
    version: PlantVersion,
    *,
    require_backend_parity: bool = False,
) -> dict[str, str]:
    """Fingerprint a compiled model's policy, physics, and visual layers.

    This small public seam also supports mutation tests and model-authoring
    tools without exposing the large canonical payloads as persisted APIs.
    """
    return {
        "policy_interface_sha256": _semantic_digest(
            POLICY_INTERFACE_SCHEMA,
            policy_layer._policy_interface_payload(
                model,
                env,
                version,
                require_backend_parity=require_backend_parity,
            ),
        ),
        "physics_sha256": _semantic_digest(PHYSICS_SCHEMA, physics_layer._physics_payload(model, version)),
        "visual_sha256": _semantic_digest(VISUAL_SCHEMA, visual_layer._visual_payload(model, version)),
    }


def _manifest_entry_for_identity(
    species: str,
    entry: Mapping[str, Any],
    version: PlantVersion,
) -> tuple[dict[str, Any], PlantIdentity]:
    model_relative_path = str(entry["model_path"])
    model_path = _resolve_repo_path(model_relative_path, field=f"{species} model")
    env_class = _load_environment(str(entry["env_entrypoint"]))
    env = env_class(reset_noise_scale=0.0)
    try:
        declared_backends = set(entry.get("training_backends", ("stable-baselines3", "jax-mjx")))
        implemented_backends = set(getattr(env, "supported_training_backends", ("stable-baselines3", "jax-mjx")))
        if declared_backends != implemented_backends:
            raise PlantContractError(f"{species} manifest/environment training backends differ")
        model = env.model
        source_payload = source_layer._source_payload(model_path)
        source_digest = _semantic_digest(SOURCE_SCHEMA, source_payload)
        layer_digests = fingerprint_model_layers(model, env, version, require_backend_parity=True)
        declared_model = mujoco.MjModel.from_xml_path(str(model_path))
        declared_digests = fingerprint_model_layers(declared_model, env, version, require_backend_parity=True)
        if declared_digests != layer_digests:
            raise PlantContractError(
                f"{species} environment model does not match the declared source {model_relative_path}"
            )
        policy_digest = layer_digests["policy_interface_sha256"]
        physics_digest = layer_digests["physics_sha256"]
        visual_digest = layer_digests["visual_sha256"]
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        identity = PlantIdentity(
            species=species,
            model_path=model_relative_path,
            physics_revision=version.physics_revision,
            policy_interface_revision=version.policy_interface_revision,
            visual_revision=version.visual_revision,
            source_closure_sha256=source_digest,
            policy_interface_sha256=policy_digest,
            physics_sha256=physics_digest,
            visual_sha256=visual_digest,
            nq=int(model.nq),
            nv=int(model.nv),
            nu=int(model.nu),
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
    finally:
        env.close()

    bundle_digest = _semantic_digest(
        PLANT_MANIFEST_SCHEMA,
        {
            "species": species,
            "source": source_digest,
            "policy_interface": policy_digest,
            "physics": physics_digest,
            "visual": visual_digest,
        },
    )
    manifest_entry = {
        "species": species,
        "model_path": model_relative_path,
        "env_entrypoint": str(entry["env_entrypoint"]),
        "source": {"schema": SOURCE_SCHEMA, **source_payload, "closure_sha256": source_digest},
        "policy_interface": {
            "schema": POLICY_INTERFACE_SCHEMA,
            "revision": version.policy_interface_revision,
            "observation_schema": version.observation_schema,
            "sha256": policy_digest,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
        },
        "physics": {
            "schema": PHYSICS_SCHEMA,
            "revision": version.physics_revision,
            "sha256": physics_digest,
            "nq": identity.nq,
            "nv": identity.nv,
            "nu": identity.nu,
        },
        "visual": {"schema": VISUAL_SCHEMA, "revision": version.visual_revision, "sha256": visual_digest},
        "bundle_sha256": bundle_digest,
    }
    return manifest_entry, identity


def build_plant_manifest(*, require_canonical_mujoco: bool = False) -> dict[str, Any]:
    """Build the current generated plant manifest from executable sources."""
    canonical_mujoco, versions = load_plant_versions()
    if require_canonical_mujoco and mujoco.__version__ != canonical_mujoco:
        raise PlantContractError(
            f"plant manifest generation requires MuJoCo {canonical_mujoco}, found {mujoco.__version__}"
        )
    entries = _species_entries()
    plants = {
        species: _manifest_entry_for_identity(species, entries[species], versions[species])[0]
        for species in sorted(entries)
    }
    return {
        "schema": PLANT_MANIFEST_SCHEMA,
        "fingerprint_tool_version": FINGERPRINT_TOOL_VERSION,
        "generated_with": {
            "mujoco": mujoco.__version__,
            "float_significant_digits": PORTABLE_FLOAT_SIGNIFICANT_DIGITS,
        },
        "plants": plants,
    }


def render_plant_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize the generated manifest deterministically for source control."""
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _enforce_revision_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    old_plants = old.get("plants", {})
    new_plants = new.get("plants", {})
    for species in sorted(set(old_plants) & set(new_plants)):
        old_entry = old_plants[species]
        new_entry = new_plants[species]
        for layer, revision_field in (
            ("policy_interface", "policy_interface_revision"),
            ("physics", "physics_revision"),
            ("visual", "visual_revision"),
        ):
            old_layer = old_entry[layer]
            new_layer = new_entry[layer]
            old_revision = int(old_layer["revision"])
            new_revision = int(new_layer["revision"])
            if new_revision < old_revision:
                raise PlantContractError(
                    f"{species} {revision_field} cannot decrease from {old_revision} to {new_revision}"
                )
            if old_layer["sha256"] == new_layer["sha256"]:
                continue
            if new_revision <= old_revision:
                raise PlantContractError(
                    f"{species} {layer} fingerprint changed without increasing {revision_field} "
                    f"above {old_layer['revision']} in {constants.PLANT_VERSIONS_PATH.relative_to(constants.REPOSITORY_ROOT)}"
                )


def write_plant_manifest() -> Path:
    """Regenerate the committed manifest after enforcing revision increments."""
    if not constants.SPECIES_MANIFEST_PATH.is_file() or not constants.PLANT_VERSIONS_PATH.is_file():
        raise PlantContractError("plant manifest generation requires a source checkout with configs/")
    new_manifest = build_plant_manifest(require_canonical_mujoco=True)
    if constants.GENERATED_MANIFEST_PATH.exists():
        _enforce_revision_changes(_load_generated_manifest(), new_manifest)
    rendered = render_plant_manifest(new_manifest)
    for output_path in (constants.GENERATED_MANIFEST_PATH, constants.BUNDLED_MANIFEST_PATH):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".json.tmp")
        temp_path.write_text(rendered, encoding="utf-8")
        temp_path.replace(output_path)
    return constants.GENERATED_MANIFEST_PATH


def check_plant_manifest(*, baseline_path: Path | None = None) -> dict[str, Any]:
    """Fail when the committed generated manifest is stale."""
    expected = render_plant_manifest(build_plant_manifest(require_canonical_mujoco=True))
    source_path = (
        constants.GENERATED_MANIFEST_PATH
        if constants.GENERATED_MANIFEST_PATH.is_file()
        else constants.BUNDLED_MANIFEST_PATH
    )
    try:
        actual = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlantContractError(f"generated plant manifest is missing: {exc}") from exc
    if actual != expected:
        raise PlantContractError(
            "generated plant manifest is stale; if semantic layers changed, bump affected revisions in "
            "configs/plant_versions.toml, then run `python -m environments.shared.plant_contract --write`"
        )
    if constants.GENERATED_MANIFEST_PATH.is_file():
        try:
            bundled = constants.BUNDLED_MANIFEST_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PlantContractError(f"bundled plant manifest is missing: {exc}") from exc
        if bundled != actual:
            raise PlantContractError("bundled plant manifest is stale; regenerate with `--write`")
    current = cast(dict[str, Any], json.loads(actual))
    if baseline_path is not None:
        _enforce_revision_changes(_load_generated_manifest(baseline_path), current)
    return current


# One slot per species: the identity build (an environment, two compiled
# models and their layer fingerprints; seconds for the mesh-heavy robot) and
# the key it was built under.  Only a build equal to the committed manifest
# entry is stored, so a stale or modified plant is always rebuilt.  The
# manifest build and check below never read it: CI's --check recomputes.
_IDENTITY_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any], PlantIdentity]] = {}


def clear_plant_identity_cache() -> None:
    """Forget every memoised identity; the next call per species rebuilds it."""
    _IDENTITY_CACHE.clear()


def _identity_cache_key(species: str, entry: Mapping[str, Any], version: PlantVersion) -> tuple[Any, ...] | None:
    """The inputs of ``_manifest_entry_for_identity`` that can change in a process.

    Resolved in the build's own order.  ``None`` (uncacheable) when the source
    closure cannot be read, so the build raises its own error in its own order.
    Python code is not part of the key; every hit re-runs the policy layer on a
    fresh environment instead (:func:`_policy_interface_unchanged`).
    """
    model_path = _resolve_repo_path(str(entry["model_path"]), field=f"{species} model")
    env_class = _load_environment(str(entry["env_entrypoint"]))
    try:
        source_digest = _semantic_digest(SOURCE_SCHEMA, source_layer._source_payload(model_path))
    except (PlantContractError, OSError):
        return None
    return (
        json.dumps(entry, sort_keys=True, default=str),
        version,
        constants.REPOSITORY_ROOT,
        model_path,
        source_digest,
        env_class,
        mujoco.__version__,
    )


def _is_committed_entry(species: str, manifest_entry: Mapping[str, Any]) -> bool:
    try:
        plants = _load_generated_manifest().get("plants")
    except (PlantContractError, OSError, ValueError):  # ValueError: an undecodable file
        return False
    return isinstance(plants, dict) and plants.get(species) == manifest_entry


def _policy_interface_unchanged(entry: Mapping[str, Any], version: PlantVersion, identity: PlantIdentity) -> bool:
    """Re-run the cheap policy layer on a fresh env (Python code is not in the key)."""
    default = ("stable-baselines3", "jax-mjx")
    try:
        env = _load_environment(str(entry["env_entrypoint"]))(reset_noise_scale=0.0)
    except Exception:
        return False
    try:
        payload = policy_layer._policy_interface_payload(env.model, env, version, require_backend_parity=True)
        return (
            _semantic_digest(POLICY_INTERFACE_SCHEMA, payload) == identity.policy_interface_sha256
            and int(np.prod(env.observation_space.shape)) == identity.observation_dim
            and int(np.prod(env.action_space.shape)) == identity.action_dim
            and set(entry.get("training_backends", default))
            == set(getattr(env, "supported_training_backends", default))
        )
    except Exception:
        return False
    finally:
        env.close()


def current_plant_identity(species: str, *, verify_generated: bool = True) -> PlantIdentity:
    """Return the executable identity for one species and optionally verify it.

    Memoised per process and species.  The configs, the source closure, the
    policy layer and the generated-manifest check are re-run on every call; the
    build is reused while its key (:func:`_identity_cache_key`) is unchanged.
    """
    _, versions = load_plant_versions()
    entries = _species_entries()
    if species not in entries:
        raise PlantContractError(f"unknown species: {species}")
    key = _identity_cache_key(species, entries[species], versions[species])
    cached = _IDENTITY_CACHE.get(species)
    if (
        key is not None
        and cached is not None
        and cached[0] == key
        and _policy_interface_unchanged(entries[species], versions[species], cached[2])
    ):
        _, manifest_entry, identity = cached
    else:
        manifest_entry, identity = _manifest_entry_for_identity(species, entries[species], versions[species])
        # key[4] is the closure digest read before the build; storing only when the build
        # hashed the same bytes keeps a file that changed mid-build from keying a stale entry.
        if (
            key is not None
            and key[4] == manifest_entry["source"]["closure_sha256"]
            and _is_committed_entry(species, manifest_entry)
        ):
            _IDENTITY_CACHE[species] = (key, manifest_entry, identity)
    if verify_generated:
        generated = _load_generated_manifest().get("plants", {}).get(species)
        if generated != manifest_entry:
            raise PlantContractError(
                f"generated plant manifest is stale for {species}; run the plant-contract check before training"
            )
    return identity
