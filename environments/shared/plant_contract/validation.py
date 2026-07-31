"""Enforcement: validate compiled models, live envs, and checkpoints.

These are the call sites training and evaluation code uses to fail closed
before a policy is trained, replayed, or promoted against the wrong plant."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from . import physics_layer
from .constants import (
    _ACTION_MAPPING_MIDPOINT,
    _MISSING_IDENTITY,
    MODEL_IDENTITY_ATTRIBUTE,
    PHYSICS_SCHEMA,
)
from .digests import _semantic_digest
from .errors import PlantCompatibilityError
from .identity import PlantIdentity

# Imported by name, not through the module: `manifest` is also used as a local
# variable here, so `manifest.x()` would shadow it. Patch these at this module.
from .manifest import fingerprint_model_layers
from .versions import load_plant_versions

_logger = logging.getLogger(__name__)


def validate_compiled_plant(
    model: mujoco.MjModel,
    current: PlantIdentity,
    *,
    artifact: str = "runtime MuJoCo model",
) -> None:
    """Prove that a backend's compiled model matches the tagged plant.

    This is intentionally independent of Gymnasium and JAX.  It prevents a
    backend from loading a different MJCF and then stamping checkpoints with
    the identity computed from the canonical species entrypoint.
    """
    _, versions = load_plant_versions()
    version = versions.get(current.species)
    if version is None:
        raise PlantCompatibilityError(f"{artifact} uses unknown species {current.species!r}")
    actual_physics = _semantic_digest(PHYSICS_SCHEMA, physics_layer._physics_payload(model, version))
    errors = []
    for field_name in ("nq", "nv", "nu"):
        actual = int(getattr(model, field_name))
        expected = int(getattr(current, field_name))
        if actual != expected:
            errors.append(f"{field_name}: runtime={actual}, current={expected}")
    if actual_physics != current.physics_sha256:
        errors.append(f"physics_sha256: runtime={actual_physics!r}, current={current.physics_sha256!r}")
    if errors:
        raise PlantCompatibilityError(
            f"{artifact} is not the current {current.species} plant:\n- " + "\n- ".join(errors)
        )


def validate_environment_plant(
    env: Any,
    current: PlantIdentity,
    *,
    artifact: str = "runtime environment",
) -> None:
    """Validate the actual Gymnasium environment before tagging artifacts."""
    model = getattr(env, "model", None)
    if not isinstance(model, mujoco.MjModel):
        raise PlantCompatibilityError(f"{artifact} exposes no MuJoCo model for plant validation")
    _, versions = load_plant_versions()
    version = versions.get(current.species)
    if version is None:
        raise PlantCompatibilityError(f"{artifact} uses unknown species {current.species!r}")
    layers = fingerprint_model_layers(model, env, version, require_backend_parity=True)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    errors = []
    checks = {
        "physics_sha256": layers["physics_sha256"],
        "policy_interface_sha256": layers["policy_interface_sha256"],
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
    }
    for field_name, actual in checks.items():
        expected = getattr(current, field_name)
        if actual != expected:
            errors.append(f"{field_name}: runtime={actual!r}, current={expected!r}")
    if errors:
        raise PlantCompatibilityError(
            f"{artifact} is not the current {current.species} plant:\n- " + "\n- ".join(errors)
        )


def validate_mjx_environment_plant(
    env: Any,
    current: PlantIdentity,
    *,
    artifact: str = "runtime MJX environment",
) -> None:
    """Bind an actual MJX model and observation/control config to identity."""
    model = getattr(env, "mj_model", None)
    if not isinstance(model, mujoco.MjModel):
        raise PlantCompatibilityError(f"{artifact} exposes no compiled mj_model for plant validation")
    validate_compiled_plant(model, current, artifact=artifact)

    try:
        importlib.import_module(f"environments.{current.species}.mjx_config")
        mjx_env_module = importlib.import_module("environments.shared.mjx_env")
        expected = mjx_env_module._SPECIES_CONFIGS[current.species]
    except (ImportError, KeyError) as exc:
        raise PlantCompatibilityError(f"{artifact} has no canonical MJX registration: {exc}") from exc
    config = getattr(env, "config", None)
    if config is None:
        raise PlantCompatibilityError(f"{artifact} exposes no MJX config for plant validation")

    fields = (
        "frame_skip",
        "body_ids",
        "sensor_foot_indices",
        "sensor_foot_aux_indices",
        "sensor_gyro_start",
        "sensor_accel_start",
        "sensor_quat_start",
        "action_mapping",
    )
    errors = []
    if str(getattr(config, "species", "")) != current.species:
        errors.append(f"species: runtime={getattr(config, 'species', None)!r}, current={current.species!r}")
    for field_name in fields:
        actual_value = getattr(config, field_name, None)
        expected_value = (
            expected.get(field_name, _ACTION_MAPPING_MIDPOINT)
            if field_name == "action_mapping"
            else expected.get(field_name)
        )
        if actual_value != expected_value:
            errors.append(f"{field_name}: runtime={actual_value!r}, registered={expected_value!r}")
    action_dim = int(getattr(env, "action_dim", -1))
    if action_dim != current.action_dim:
        errors.append(f"action_dim: runtime={action_dim}, current={current.action_dim}")
    if errors:
        raise PlantCompatibilityError(
            f"{artifact} is not the current {current.species} MJX interface:\n- " + "\n- ".join(errors)
        )


def write_plant_identity(path: str | Path, identity: PlantIdentity) -> Path:
    """Atomically write a checkpoint/run identity sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)
    return path


def attach_plant_identity(model: Any, identity: PlantIdentity) -> None:
    """Attach identity metadata that Stable-Baselines3 persists in its ZIP."""
    setattr(model, MODEL_IDENTITY_ATTRIBUTE, identity.to_dict())


def validate_recorded_identity(
    recorded: Mapping[str, Any] | None,
    current: PlantIdentity,
    *,
    artifact: str = "checkpoint",
    allow_legacy: bool = False,
) -> None:
    """Validate persisted metadata, failing closed unless legacy use is explicit."""
    if recorded is None:
        if allow_legacy:
            _logger.warning(
                "%s has no plant identity; loading only because allow_legacy_plant was explicitly enabled",
                artifact,
            )
            return
        raise PlantCompatibilityError(
            f"{artifact} has no plant identity. It predates the plant-safety contract and cannot be loaded "
            "silently. Re-run with allow_legacy_plant=True only for deliberate historical migration/evaluation."
        )
    checkpoint_identity = PlantIdentity.from_mapping(recorded)
    errors = current.compatibility_errors(checkpoint_identity)
    if errors:
        raise PlantCompatibilityError(
            f"{artifact} is incompatible with the current {current.species} plant:\n- " + "\n- ".join(errors)
        )


def validate_model_plant(
    model: Any,
    current: PlantIdentity,
    *,
    artifact: str = "checkpoint",
    allow_legacy: bool = False,
) -> None:
    """Validate identity embedded in a loaded SB3 model."""
    raw = getattr(model, MODEL_IDENTITY_ATTRIBUTE, _MISSING_IDENTITY)
    if raw is _MISSING_IDENTITY:
        validate_recorded_identity(None, current, artifact=artifact, allow_legacy=allow_legacy)
        return
    if not isinstance(raw, Mapping):
        raise PlantCompatibilityError(
            f"{artifact} contains invalid plant identity metadata of type {type(raw).__name__}"
        )
    validate_recorded_identity(raw, current, artifact=artifact, allow_legacy=allow_legacy)
