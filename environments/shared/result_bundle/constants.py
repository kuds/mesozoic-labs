"""Schema versions, filenames, and the recorded dependency set."""

from __future__ import annotations

ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_NAME = "artifact_manifest.json"
DEFAULT_PROVENANCE_NAME = "provenance.json"

_FINALIZATION_PROVENANCE_FIELDS = frozenset(
    {
        "backend_version",
        "config_hash",
        "model_hash",
        "model_revision_status",
        "selected_checkpoints",
        "selected_model_path",
        "verification_status",
    }
)

_DEPENDENCY_PACKAGES = {
    "mesozoic_labs": "mesozoic-labs",
    "mujoco": "mujoco",
    "mujoco_mjx": "mujoco-mjx",
    "gymnasium": "gymnasium",
    "numpy": "numpy",
    "stable_baselines3": "stable-baselines3",
    "jax": "jax",
    "flax": "flax",
    "optax": "optax",
}
