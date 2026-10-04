"""Portable result-bundle provenance, hashing, and validation helpers.

Training runs are written to Google Drive by the public Colab notebooks and
later promoted into the repository.  This package keeps that hand-off
deterministic: provenance is captured when a run starts, artifact paths are
relative to the run directory, and a manifest written last detects incomplete
copies or later modification.

The result schema itself lives in :mod:`environments.shared.result_schema`.
Drive bundles may be partial or failed; a ``summary.json`` exists whenever
at least one deliverable is certified, and repository promotion requires a
publishable (canonical-valid or canonical-partial) bundle.

Submodules layer bottom-up; import from this package rather than from one of
them:

* :mod:`~environments.shared.result_bundle.constants` — schema versions,
  filenames, and the recorded dependency set
* :mod:`~environments.shared.result_bundle.errors` — ``ResultBundleError``
* :mod:`~environments.shared.result_bundle.naming` — canonical algorithm,
  backend, and path spellings
* :mod:`~environments.shared.result_bundle.hashing` — content hashes and
  deterministic JSON writing
* :mod:`~environments.shared.result_bundle.provenance` — capture and update
  run provenance
* :mod:`~environments.shared.result_bundle.manifest` — write and verify the
  artifact manifest
* :mod:`~environments.shared.result_bundle.evidence` — cross-check the summary
  against the CSV and per-episode evaluation evidence
* :mod:`~environments.shared.result_bundle.audit` — the whole-bundle audit run
  before repository promotion
* :mod:`~environments.shared.result_bundle.gate_verdict` — the per-node
  ``gate_verdict.json`` record a stage directory carries beside its handoff
* :mod:`~environments.shared.result_bundle.ancestors` — the reader of the
  ``ancestors/<stage_id>/`` records of nodes reused from another run
* :mod:`~environments.shared.result_bundle.trunk_record` — the run-level
  ``trunk_run.json`` record of the trunk run a notebook session resolved
* :mod:`~environments.shared.result_bundle.reentry` — what a notebook session
  that re-enters an existing run directory is refused before it writes
  anything (a complete bundle, a widened root's seed, a root widened beside
  a reused record of it, a resume under another trunk than the recorded
  one, a node judged ahead of the trunk off another parent; and, for a
  notebook copy older than cleanup ROW-4/6, a trunk over an unjudged
  widened root)
"""

from __future__ import annotations

from .ancestors import load_ancestor_records, project_ancestor_records
from .audit import audit_result_bundle, validate_result_bundle
from .constants import (
    ANCESTOR_RECORD_NAME,
    ANCESTOR_RECORD_SCHEMA,
    ANCESTORS_DIRNAME,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_PROVENANCE_NAME,
    PROVENANCE_SCHEMA_VERSION,
    TRUNK_RECORD_NAME,
    TRUNK_RECORD_SCHEMA,
)
from .errors import ResultBundleError
from .evidence import compare_summary_to_csv, validate_evaluation_evidence
from .gate_verdict import (
    GATE_VERDICT_FILENAME,
    GATE_VERDICT_SCHEMA,
    GateVerdictError,
    read_gate_verdict,
    verdict_is_reusable,
    write_gate_verdict,
)
from .hashing import _write_json, aggregate_file_hash, canonical_json_sha256, sha256_file
from .manifest import (
    build_artifact_manifest,
    manifest_disagreements,
    read_bundle_status,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from .naming import _normalize_plant_identity, canonical_algorithm, canonical_backend
from .provenance import initialize_result_bundle, load_provenance, update_provenance
from .reentry import (
    refuse_complete_run_session,
    refuse_judging_off_the_resolved_parent,
    refuse_trunk_other_than_recorded,
    refuse_trunk_over_unjudged_widened_root,
    refuse_widened_seed_mismatch,
    refuse_write_into_complete_run,
    unjudged_stage_dir,
)
from .trunk_record import read_trunk_record, record_trunk_run, trunk_from_value

__all__ = [
    "ANCESTOR_RECORD_NAME",
    "ANCESTOR_RECORD_SCHEMA",
    "ANCESTORS_DIRNAME",
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "DEFAULT_MANIFEST_NAME",
    "DEFAULT_PROVENANCE_NAME",
    "GATE_VERDICT_FILENAME",
    "GATE_VERDICT_SCHEMA",
    "PROVENANCE_SCHEMA_VERSION",
    "TRUNK_RECORD_NAME",
    "TRUNK_RECORD_SCHEMA",
    "GateVerdictError",
    "ResultBundleError",
    "_normalize_plant_identity",
    "_write_json",
    "aggregate_file_hash",
    "audit_result_bundle",
    "build_artifact_manifest",
    "canonical_algorithm",
    "canonical_backend",
    "canonical_json_sha256",
    "compare_summary_to_csv",
    "initialize_result_bundle",
    "load_ancestor_records",
    "load_provenance",
    "manifest_disagreements",
    "project_ancestor_records",
    "read_bundle_status",
    "read_gate_verdict",
    "read_trunk_record",
    "record_trunk_run",
    "refuse_complete_run_session",
    "refuse_judging_off_the_resolved_parent",
    "refuse_trunk_other_than_recorded",
    "refuse_trunk_over_unjudged_widened_root",
    "refuse_widened_seed_mismatch",
    "refuse_write_into_complete_run",
    "sha256_file",
    "trunk_from_value",
    "unjudged_stage_dir",
    "update_provenance",
    "validate_evaluation_evidence",
    "validate_result_bundle",
    "verdict_is_reusable",
    "verify_artifact_manifest",
    "write_artifact_manifest",
    "write_gate_verdict",
]
