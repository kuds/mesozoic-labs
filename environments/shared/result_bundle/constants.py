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

#: Certified ancestors reused from another run (BEHAVIOR_RECIPES_PLAN §4.2):
#: ``<run_dir>/ancestors/<stage_id>/ancestor.json`` plus verbatim copies of
#: the ancestor stage's verdict, config, fingerprint and plant identity —
#: never a checkpoint.  Written by ``environments.shared.ancestors``, read by
#: :mod:`.ancestors`.
ANCESTORS_DIRNAME = "ancestors"
ANCESTOR_RECORD_NAME = "ancestor.json"
ANCESTOR_RECORD_SCHEMA = "mesozoic.ancestor-record/v1"

#: The fields :func:`.provenance.update_provenance` may write after capture.
#: ``deliverables`` / ``primary_deliverable`` / ``target_deliverable`` /
#: ``ancestors`` are FINALIZATION fields (decision D-A16), written by
#: ``save_result_bundle`` — never identity fields captured at
#: ``initialize_result_bundle``, so re-running the notebook's setup cell
#: under the same run id with a different BEHAVIOR is not a rejected run.
_FINALIZATION_PROVENANCE_FIELDS = frozenset(
    {
        "ancestors",
        "backend_version",
        "config_hash",
        "deliverables",
        "model_hash",
        "model_revision_status",
        "primary_deliverable",
        "selected_checkpoints",
        "selected_model_path",
        "target_deliverable",
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
