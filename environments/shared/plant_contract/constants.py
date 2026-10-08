"""Paths, schema identifiers, and tuning constants for the plant contract."""

from __future__ import annotations

# Bound as this module's own attributes: it stays the plant contract's patch
# point (tests patch ``constants.REPOSITORY_ROOT``; source_layer, versions,
# manifest and ``__main__`` read it here at call time), while the value comes
# from the one computation in ``environments/shared/paths.py``.
from ..paths import REPOSITORY_ROOT as REPOSITORY_ROOT
from ..paths import SHARED_ROOT as _SHARED_ROOT

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
# home-keyframe-residual/v1 with the clipped residual of selected actuators
# shaped to b = s*a + (1 - s)*a**3 before the same piecewise-affine span: the
# origin (b(0) = 0) and both ctrlrange endpoints (b(+-1) = +-1) are unchanged
# and the slope at home is s.  SB3-only (no JAX implementation); the species
# declares s as ``residual_linear_slope`` and the shaped actuators as the
# per-actuator boolean ``_shaped_residual_mask`` (compsognathus, D-D26).
_ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL_SOFTCUBIC = "home-keyframe-residual-softcubic/v1"
_HOME_KEYFRAME_ACTION_MAPPINGS = (
    _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL,
    _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL_SOFTCUBIC,
)
_MIDPOINT_ACTION_MAPPING_DESCRIPTION = "clip[-1,1]-then-affine-to-ordered-ctrlrange/v1"
