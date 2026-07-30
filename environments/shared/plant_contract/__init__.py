"""Versioned, layered safety contract for the MuJoCo physics plants.

The contract separates four concerns that used to be conflated as "the XML":

* source closure -- exact MJCF and referenced asset bytes;
* policy interface -- ordered observations, actions, sensors, and controls;
* physics -- compiled dynamics, contacts, actuators, and reset state; and
* visual presentation -- render geometry, materials, cameras, and lights.

This separation lets a render-only model improvement invalidate videos without
silently invalidating a policy, while any control or physics change fails closed
until its human revision and generated manifest are deliberately updated.

Those four concerns are one submodule each, over a shared base of digesting and
version loading, with manifest generation and enforcement on top.  Import from
this package rather than from a submodule:

* :mod:`~environments.shared.plant_contract.constants` — paths and schema ids;
  the single patch point for the ``configs/`` locations
* :mod:`~environments.shared.plant_contract.errors` — the two exceptions
* :mod:`~environments.shared.plant_contract.digests` — portable
  canonicalisation and semantic digests
* :mod:`~environments.shared.plant_contract.identity` — :class:`PlantVersion`
  and :class:`PlantIdentity`
* :mod:`~environments.shared.plant_contract.versions` — reviewed revisions,
  species entries, and the committed manifest, read from disk
* :mod:`~environments.shared.plant_contract.introspection` — shared MuJoCo
  model-introspection helpers
* ``source_layer`` / ``policy_layer`` / ``physics_layer`` / ``visual_layer`` —
  one payload builder per contract layer
* :mod:`~environments.shared.plant_contract.manifest` — build, render, write,
  and check the generated manifest
* :mod:`~environments.shared.plant_contract.validation` — fail-closed checks
  for compiled models, live environments, and checkpoints

Run ``python -m environments.shared.plant_contract --check`` to verify the
committed manifest.
"""

from __future__ import annotations

from . import constants
from .constants import (
    BUNDLED_MANIFEST_PATH,
    FINGERPRINT_TOOL_VERSION,
    GENERATED_MANIFEST_PATH,
    MODEL_IDENTITY_ATTRIBUTE,
    PHYSICS_SCHEMA,
    PLANT_IDENTITY_SCHEMA,
    PLANT_MANIFEST_SCHEMA,
    PLANT_VERSIONS_PATH,
    POLICY_INTERFACE_SCHEMA,
    PORTABLE_FLOAT_SIGNIFICANT_DIGITS,
    REPOSITORY_ROOT,
    SOURCE_SCHEMA,
    SPECIES_MANIFEST_PATH,
    VISUAL_SCHEMA,
)
from .errors import PlantCompatibilityError, PlantContractError
from .identity import PlantIdentity, PlantVersion
from .manifest import (
    build_plant_manifest,
    check_plant_manifest,
    current_plant_identity,
    fingerprint_model_layers,
    render_plant_manifest,
    write_plant_manifest,
)
from .validation import (
    attach_plant_identity,
    validate_compiled_plant,
    validate_environment_plant,
    validate_mjx_environment_plant,
    validate_model_plant,
    validate_recorded_identity,
    write_plant_identity,
)
from .versions import load_plant_versions

__all__ = [
    "BUNDLED_MANIFEST_PATH",
    "FINGERPRINT_TOOL_VERSION",
    "GENERATED_MANIFEST_PATH",
    "MODEL_IDENTITY_ATTRIBUTE",
    "PHYSICS_SCHEMA",
    "PLANT_IDENTITY_SCHEMA",
    "PLANT_MANIFEST_SCHEMA",
    "PLANT_VERSIONS_PATH",
    "POLICY_INTERFACE_SCHEMA",
    "PORTABLE_FLOAT_SIGNIFICANT_DIGITS",
    "REPOSITORY_ROOT",
    "SOURCE_SCHEMA",
    "SPECIES_MANIFEST_PATH",
    "VISUAL_SCHEMA",
    "PlantCompatibilityError",
    "PlantContractError",
    "PlantIdentity",
    "PlantVersion",
    "attach_plant_identity",
    "build_plant_manifest",
    "check_plant_manifest",
    "constants",
    "current_plant_identity",
    "fingerprint_model_layers",
    "load_plant_versions",
    "render_plant_manifest",
    "validate_compiled_plant",
    "validate_environment_plant",
    "validate_mjx_environment_plant",
    "validate_model_plant",
    "validate_recorded_identity",
    "write_plant_identity",
    "write_plant_manifest",
]
