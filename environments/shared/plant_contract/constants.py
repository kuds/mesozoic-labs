"""Paths, schema identifiers, and tuning constants for the plant contract."""

from __future__ import annotations

from pathlib import Path

_SHARED_ROOT = Path(__file__).resolve().parent.parent

REPOSITORY_ROOT = _SHARED_ROOT.parents[1]
SPECIES_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "species_manifest.toml"
PLANT_VERSIONS_PATH = REPOSITORY_ROOT / "configs" / "plant_versions.toml"
GENERATED_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "plant_manifest.generated.json"
BUNDLED_MANIFEST_PATH = _SHARED_ROOT / "data" / "plant_manifest.generated.json"

PLANT_MANIFEST_SCHEMA = "mesozoic.plant-manifest/v1"
PLANT_IDENTITY_SCHEMA = "mesozoic.plant-identity/v1"
POLICY_INTERFACE_SCHEMA = "mesozoic.policy-interface/v1"
PHYSICS_SCHEMA = "mesozoic.mujoco-physics/v1"
VISUAL_SCHEMA = "mesozoic.mujoco-visual/v1"
SOURCE_SCHEMA = "mesozoic.source-closure/v1"
FINGERPRINT_TOOL_VERSION = 2
PORTABLE_FLOAT_SIGNIFICANT_DIGITS = 12
MODEL_IDENTITY_ATTRIBUTE = "_mesozoic_plant_identity"
_MISSING_IDENTITY = object()

_ACTION_MAPPING_MIDPOINT = "midpoint/v1"
_ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL = "home-keyframe-residual/v1"
_MIDPOINT_ACTION_MAPPING_DESCRIPTION = "clip[-1,1]-then-affine-to-ordered-ctrlrange/v1"
