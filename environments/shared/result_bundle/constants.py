"""Schema versions, filenames, paths, and the recorded dependency set."""

from __future__ import annotations

from pathlib import Path

_SHARED_ROOT = Path(__file__).resolve().parent.parent
"""``environments/shared`` — the anchor every repository-relative path derives from.

Expressed via a named anchor rather than a bare ``parents[N]`` count because a
raw count silently changes meaning when a module moves to a different depth, and
stays byte-identical while doing so.  ``test_repository_root_resolves_to_the_repository``
pins the result.
"""

REPOSITORY_ROOT = _SHARED_ROOT.parents[1]

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
