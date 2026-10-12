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

from ..model_id import ModelId
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
    _generated_plants,
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
    model_id: ModelId,
    entry: Mapping[str, Any],
    version: PlantVersion,
) -> tuple[dict[str, Any], PlantIdentity]:
    model_relative_path = str(entry["model_path"])
    model_path = _resolve_repo_path(model_relative_path, field=f"{model_id} model")
    env_class = _load_environment(str(entry["env_entrypoint"]))
    env = env_class(reset_noise_scale=0.0)
    try:
        implemented_backends = set(getattr(env, "supported_training_backends", ("stable-baselines3", "jax-mjx")))
        # Required on every species-manifest entry (review L5); None: a
        # bundled-manifest entry, which carries no declaration to check.
        declared = entry["training_backends"]
        declared_backends = implemented_backends if declared is None else set(declared)
        if declared_backends != implemented_backends:
            raise PlantContractError(f"{model_id} manifest/environment training backends differ")
        model = env.model
        source_payload = source_layer._source_payload(model_path)
        source_digest = _semantic_digest(SOURCE_SCHEMA, source_payload)
        layer_digests = fingerprint_model_layers(model, env, version, require_backend_parity=True)
        declared_model = mujoco.MjModel.from_xml_path(str(model_path))
        declared_digests = fingerprint_model_layers(declared_model, env, version, require_backend_parity=True)
        if declared_digests != layer_digests:
            raise PlantContractError(
                f"{model_id} environment model does not match the declared source {model_relative_path}"
            )
        policy_digest = layer_digests["policy_interface_sha256"]
        physics_digest = layer_digests["physics_sha256"]
        visual_digest = layer_digests["visual_sha256"]
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        identity = PlantIdentity(
            species=model_id.species,
            embodiment=model_id.embodiment,
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
            "species": model_id.species,
            "embodiment": model_id.embodiment,
            "source": source_digest,
            "policy_interface": policy_digest,
            "physics": physics_digest,
            "visual": visual_digest,
        },
    )
    manifest_entry = {
        "species": model_id.species,
        "embodiment": model_id.embodiment,
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
    # Nested by species, then embodiment (schema v2, decision D-D33).
    plants: dict[str, dict[str, Any]] = {}
    for model_id in sorted(entries):
        entry, _ = _manifest_entry_for_identity(model_id, entries[model_id], versions[model_id])
        plants.setdefault(model_id.species, {})[model_id.embodiment] = entry
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


# Decision D-D33 re-keyed every plant from its species to (species, embodiment)
# and moved the manifest from v1 to v2.  The revision check reads a v1
# baseline only against a v2 current manifest, the one transition it bridges;
# runtime reads stay strict (versions._load_generated_manifest).
_D_D33_BASELINE_SCHEMA = "mesozoic.plant-manifest/v1"
_D_D33_CURRENT_SCHEMA = "mesozoic.plant-manifest/v2"


def _load_baseline_manifest(path: Path) -> dict[str, Any]:
    """Read a revision-check baseline: the committed manifest, or CI's copy of the PR base.

    Its schema is left to :func:`_enforce_revision_changes`, which accepts the
    current schema and, against a v2 current manifest, the v1 baseline of
    decision D-D33.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlantContractError(f"cannot read baseline plant manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlantContractError(f"baseline plant manifest {path} is not a JSON object")
    return value


def _check_manifest_schemas(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    current_schema = constants.PLANT_MANIFEST_SCHEMA
    if new.get("schema") != current_schema:
        raise PlantContractError(
            f"the current plant manifest must use schema {current_schema}, not {new.get('schema')!r}"
        )
    baseline_schema = old.get("schema")
    if baseline_schema == current_schema or (
        (baseline_schema, current_schema) == (_D_D33_BASELINE_SCHEMA, _D_D33_CURRENT_SCHEMA)
    ):
        return
    raise PlantContractError(
        f"baseline plant manifest schema {baseline_schema!r} cannot be checked against {current_schema}; "
        f"only {_D_D33_BASELINE_SCHEMA} against {_D_D33_CURRENT_SCHEMA} crosses schemas (decision D-D33)"
    )


def _keyed_plants(manifest: Mapping[str, Any], role: str) -> dict[str | ModelId, Mapping[str, Any]]:
    """The manifest's plants by identity key.

    v1 keys a plant by its species string; v2 nests it as
    ``plants[species][embodiment]`` and keys it by its :class:`ModelId`.
    """
    plants = manifest.get("plants", {})
    if not isinstance(plants, Mapping):
        raise PlantContractError(f"{role} plant manifest has no plants table")
    if manifest.get("schema") == _D_D33_BASELINE_SCHEMA:
        return {str(species): entry for species, entry in plants.items()}
    keyed: dict[str | ModelId, Mapping[str, Any]] = {}
    for species, embodiments in plants.items():
        if not isinstance(embodiments, Mapping):
            raise PlantContractError(f"{role} plant manifest species {species!r} is not a table of embodiments")
        for embodiment, entry in embodiments.items():
            try:
                keyed[ModelId(species, embodiment)] = entry
            except ValueError as exc:
                raise PlantContractError(f"{role} plant manifest key {species!r}.{embodiment!r}: {exc}") from exc
    return keyed


def _physics_sha256(key: str | ModelId, entry: Mapping[str, Any]) -> str:
    try:
        return str(entry["physics"]["sha256"])
    except (KeyError, TypeError) as exc:
        raise PlantContractError(f"plant {key} records no physics digest") from exc


def _pair_plants(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    """Pair every baseline plant with a current plant: by key, then by physics digest.

    A plant re-keyed under the same physics (decision D-D33's rename, or a
    later epithet correction) keeps its counters through the digest; a
    baseline plant with neither a key nor a digest in the current manifest is
    refused, because a plant leaves the manifest only by an explicit decision.
    Current plants left unpaired are new.  Returns (label, baseline entry,
    current entry) per pair, in baseline key order.
    """
    old_plants = _keyed_plants(old, "baseline")
    new_plants = _keyed_plants(new, "current")
    pairs = [(str(key), old_plants[key], new_plants[key]) for key in sorted(old_plants, key=str) if key in new_plants]
    # The plants left over on each side, by physics digest.
    old_left = {key: _physics_sha256(key, entry) for key, entry in old_plants.items() if key not in new_plants}
    new_left = {key: _physics_sha256(key, entry) for key, entry in new_plants.items() if key not in old_plants}
    for old_key, digest in sorted(old_left.items(), key=lambda item: str(item[0])):
        twins = sorted(str(key) for key, value in old_left.items() if value == digest)
        matches = [key for key, value in new_left.items() if value == digest]
        if len(twins) > 1 or len(matches) > 1:
            raise PlantContractError(
                f"physics digest {digest} is shared by more than one re-keyed plant (baseline: {', '.join(twins)}; "
                f"current: {', '.join(sorted(map(str, matches)))}), so they cannot be paired"
            )
        if not matches:
            raise PlantContractError(
                f"{old_key} left the plant manifest: no current plant has its key or its physics digest; "
                "a plant is retired only by an explicit decision, and a re-keyed plant keeps its physics "
                "in the change that re-keys it"
            )
        (new_key,) = matches
        _logger.info("paired baseline plant %s with %s by physics digest", old_key, new_key)
        pairs.append((f"{old_key} (now {new_key})", old_plants[old_key], new_plants[new_key]))
    return pairs


def _enforce_revision_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    """Refuse a lowered revision, or a changed layer whose revision did not increase.

    ``old`` is the baseline (the committed manifest, or the PR base in CI) and
    ``new`` the current manifest; their plants are paired by
    :func:`_pair_plants`, so re-keying or removing a plant cannot bypass the
    rules.
    """
    _check_manifest_schemas(old, new)
    for label, old_entry, new_entry in _pair_plants(old, new):
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
                    f"{label} {revision_field} cannot decrease from {old_revision} to {new_revision}"
                )
            if old_layer["sha256"] == new_layer["sha256"]:
                continue
            if new_revision <= old_revision:
                raise PlantContractError(
                    f"{label} {layer} fingerprint changed without increasing {revision_field} "
                    f"above {old_layer['revision']} in {constants.PLANT_VERSIONS_PATH.relative_to(constants.REPOSITORY_ROOT)}"
                )


def write_plant_manifest() -> Path:
    """Regenerate the committed manifest after enforcing revision increments."""
    if not constants.SPECIES_MANIFEST_PATH.is_file() or not constants.PLANT_VERSIONS_PATH.is_file():
        raise PlantContractError("plant manifest generation requires a source checkout with configs/")
    new_manifest = build_plant_manifest(require_canonical_mujoco=True)
    if constants.GENERATED_MANIFEST_PATH.exists():
        _enforce_revision_changes(_load_baseline_manifest(constants.GENERATED_MANIFEST_PATH), new_manifest)
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
        _enforce_revision_changes(_load_baseline_manifest(baseline_path), current)
    return current


# One slot per model: the identity build (an environment, two compiled
# models and their layer fingerprints; seconds for the mesh-heavy robot) and
# the key it was built under.  Only a build equal to the committed manifest
# entry is stored, so a stale or modified plant is always rebuilt.  The
# manifest build and check below never read it: CI's --check recomputes.
_IDENTITY_CACHE: dict[ModelId, tuple[tuple[Any, ...], dict[str, Any], PlantIdentity]] = {}


def clear_plant_identity_cache() -> None:
    """Forget every memoised identity; the next call per model rebuilds it."""
    _IDENTITY_CACHE.clear()


def _identity_cache_key(model_id: ModelId, entry: Mapping[str, Any], version: PlantVersion) -> tuple[Any, ...] | None:
    """The inputs of ``_manifest_entry_for_identity`` that can change in a process.

    Resolved in the build's own order.  ``None`` (uncacheable) when the source
    closure cannot be read, so the build raises its own error in its own order.
    Python code is not part of the key; every hit re-runs the policy layer on a
    fresh environment instead (:func:`_policy_interface_unchanged`).
    """
    model_path = _resolve_repo_path(str(entry["model_path"]), field=f"{model_id} model")
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


def _is_committed_entry(model_id: ModelId, manifest_entry: Mapping[str, Any]) -> bool:
    try:
        plants = _generated_plants(_load_generated_manifest())
    except (PlantContractError, OSError, ValueError):  # ValueError: an undecodable file
        return False
    return plants.get(model_id) == manifest_entry


def _policy_interface_unchanged(entry: Mapping[str, Any], version: PlantVersion, identity: PlantIdentity) -> bool:
    """Re-run the cheap policy layer on a fresh env (Python code is not in the key)."""
    default = ("stable-baselines3", "jax-mjx")
    declared = entry["training_backends"]
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
            and (declared is None or set(declared) == set(getattr(env, "supported_training_backends", default)))
        )
    except Exception:
        return False
    finally:
        env.close()


def current_plant_identity(model_id: ModelId, *, verify_generated: bool = True) -> PlantIdentity:
    """Return the executable identity for one model and optionally verify it.

    Memoised per process and model.  The configs, the source closure, the
    policy layer and the generated-manifest check are re-run on every call; the
    build is reused while its key (:func:`_identity_cache_key`) is unchanged.
    """
    _, versions = load_plant_versions()
    entries = _species_entries()
    if model_id not in entries:
        raise PlantContractError(f"unknown model: {model_id}")
    key = _identity_cache_key(model_id, entries[model_id], versions[model_id])
    cached = _IDENTITY_CACHE.get(model_id)
    if (
        key is not None
        and cached is not None
        and cached[0] == key
        and _policy_interface_unchanged(entries[model_id], versions[model_id], cached[2])
    ):
        _, manifest_entry, identity = cached
    else:
        manifest_entry, identity = _manifest_entry_for_identity(model_id, entries[model_id], versions[model_id])
        # key[4] is the closure digest read before the build; storing only when the build
        # hashed the same bytes keeps a file that changed mid-build from keying a stale entry.
        if (
            key is not None
            and key[4] == manifest_entry["source"]["closure_sha256"]
            and _is_committed_entry(model_id, manifest_entry)
        ):
            _IDENTITY_CACHE[model_id] = (key, manifest_entry, identity)
    if verify_generated:
        generated = _generated_plants(_load_generated_manifest()).get(model_id)
        if generated != manifest_entry:
            raise PlantContractError(
                f"generated plant manifest is stale for {model_id}; run the plant-contract check before training"
            )
    return identity
