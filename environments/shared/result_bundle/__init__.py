"""Portable result-bundle provenance, hashing, and validation helpers.

Training runs are written to Google Drive by the public Colab notebooks and
later promoted into the repository.  This package keeps that hand-off
deterministic: provenance is captured when a run starts, artifact paths are
relative to the run directory, and a manifest written last detects incomplete
copies or later modification.

The result schema itself lives in :mod:`environments.shared.result_schema`.
Drive bundles may be partial, but repository promotion requires a complete
three-stage summary.

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
"""

from __future__ import annotations

from .audit import audit_result_bundle, validate_result_bundle
from .constants import (
    ANCESTOR_RECORD_NAME,
    ANCESTOR_RECORD_SCHEMA,
    ANCESTORS_DIRNAME,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_PROVENANCE_NAME,
    PROVENANCE_SCHEMA_VERSION,
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
    verify_artifact_manifest,
    write_artifact_manifest,
)
from .naming import _normalize_plant_identity, canonical_algorithm, canonical_backend
from .provenance import initialize_result_bundle, load_provenance, update_provenance

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
    "load_provenance",
    "manifest_disagreements",
    "read_gate_verdict",
    "sha256_file",
    "update_provenance",
    "validate_evaluation_evidence",
    "validate_result_bundle",
    "verdict_is_reusable",
    "verify_artifact_manifest",
    "write_artifact_manifest",
    "write_gate_verdict",
]
