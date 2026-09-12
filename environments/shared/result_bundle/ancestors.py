"""Read the ``ancestors/<stage_id>/`` records of nodes reused from another run.

BEHAVIOR_RECIPES_PLAN §4.2 (Phase A): a node may be satisfied by an existing
certified checkpoint instead of trained.  The child run then carries, under
``<run_dir>/ancestors/<stage_id>/``, an ``ancestor.json`` record plus
verbatim copies of the ancestor stage's ``gate_verdict.json``,
``stage_config.json`` (and ``task_fingerprint.json`` / ``plant_identity.json``
when the stage had them) — small files, never the checkpoint.  The writer
lives in ``environments.shared.ancestors`` (the orchestration side); this is
the reader the bundle writer and audit use.

Every rule here fails CLOSED: an undeclared stage id, a missing file, a
verdict whose hashes disagree with the record's, a task fingerprint that
disagrees between the three files — none of it reads as "passed".  Absence
of a record is simply an empty result; absence of a file inside a record is
an error, never a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .constants import ANCESTOR_RECORD_NAME, ANCESTOR_RECORD_SCHEMA, ANCESTORS_DIRNAME
from .errors import ResultBundleError
from .gate_verdict import GATE_VERDICT_FILENAME, GateVerdictError, read_gate_verdict
from .manifest import _is_ignored_litter

#: What one loaded record carries: the six ``provenance.ancestors`` fields
#: (``result_schema.ANCESTOR_RECORD_FIELDS``) plus the id and the judge.
ANCESTOR_RECORD_KEYS = (
    "run_id",
    "model_hash",
    "normalization_hash",
    "gate_kind",
    "passed",
    "task_sha256",
    "stage_id",
    "judged_by",
)


def _read_json_object(path: Path, *, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise ResultBundleError(f"ancestor record is missing its {what}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultBundleError(f"ancestor record {what} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultBundleError(f"ancestor record {what} must hold a JSON object: {path}")
    return value


def _digest_or_none(value: Any, *, field: str, path: Path) -> str | None:
    from .evidence import _SHA256_DIGEST

    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_DIGEST.fullmatch(value) is None:
        raise ResultBundleError(f"{path}: {field} must be sha256:<64 lowercase hex> or null")
    return value


def _digest(value: Any, *, field: str, path: Path) -> str:
    digest = _digest_or_none(value, field=field, path=path)
    if digest is None:
        raise ResultBundleError(f"{path}: {field} must be sha256:<64 lowercase hex>")
    return digest


def load_ancestor_records(run_dir: str | Path, *, species: str) -> dict[str, dict[str, Any]]:
    """Load and cross-check every ``ancestors/<stage_id>/`` record of *run_dir*.

    Returns ``{stage_key: record}`` with the :data:`ANCESTOR_RECORD_KEYS`
    fields, keyed the way the summary keys stages (the legacy number's
    decimal spelling, else the id).  Empty when the run has no ``ancestors``
    directory.  Raises :class:`ResultBundleError` when a record directory
    names a stage the species' manifest does not declare, when
    ``ancestor.json``, ``gate_verdict.json`` or ``stage_config.json`` is
    absent or malformed, when the verdict's handoff digests disagree with
    the record's, or when the task fingerprint disagrees between the record,
    the verdict and the config.
    """
    from ..stage_manifest import StageManifestError, load_stage_manifest

    run_path = Path(run_dir)
    root = run_path / ANCESTORS_DIRNAME
    if not root.is_dir():
        return {}
    try:
        manifest = load_stage_manifest(species)
    except StageManifestError as exc:
        raise ResultBundleError(f"cannot load the stage manifest for {species!r}: {exc}") from exc

    records: dict[str, dict[str, Any]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            if _is_ignored_litter(child, run_path):
                continue
            raise ResultBundleError(f"{root} may only hold ancestor record directories, found {child.name!r}")
        stage_id = child.name
        try:
            entry = manifest.by_id(stage_id)
        except StageManifestError as exc:
            raise ResultBundleError(
                f"ancestors/{stage_id} names a stage the {species} manifest does not declare: {exc}"
            ) from exc

        record_path = child / ANCESTOR_RECORD_NAME
        record = _read_json_object(record_path, what=ANCESTOR_RECORD_NAME)
        if record.get("schema") != ANCESTOR_RECORD_SCHEMA:
            raise ResultBundleError(
                f"{record_path} declares schema {record.get('schema')!r}; expected {ANCESTOR_RECORD_SCHEMA!r}"
            )
        if record.get("stage_id") != stage_id:
            raise ResultBundleError(
                f"{record_path} records stage_id {record.get('stage_id')!r} but lives under ancestors/{stage_id}"
            )
        if record.get("stage_key") != entry.key:
            raise ResultBundleError(
                f"{record_path} records stage_key {record.get('stage_key')!r}; the {species} manifest keys "
                f"{stage_id!r} as {entry.key!r}"
            )
        parent_run_id = record.get("parent_run_id")
        if not isinstance(parent_run_id, str) or not parent_run_id.strip():
            raise ResultBundleError(f"{record_path}: parent_run_id must be a non-empty string")
        handoff = record.get("handoff")
        if not isinstance(handoff, Mapping):
            raise ResultBundleError(f"{record_path}: handoff must be an object")
        model_hash = _digest(handoff.get("model_sha256"), field="handoff.model_sha256", path=record_path)
        normalization_hash = _digest_or_none(
            handoff.get("normalization_sha256"),
            field="handoff.normalization_sha256",
            path=record_path,
        )
        task_sha256 = _digest(record.get("task_sha256"), field="task_sha256", path=record_path)
        judged_by = record.get("judged_by")
        if not isinstance(judged_by, str) or not judged_by.strip():
            raise ResultBundleError(f"{record_path}: judged_by must name the verdict's producer")

        try:
            verdict = read_gate_verdict(child)
        except GateVerdictError as exc:
            raise ResultBundleError(str(exc)) from exc
        verdict_path = child / GATE_VERDICT_FILENAME
        if verdict is None:
            raise ResultBundleError(
                f"ancestor record ancestors/{stage_id} has no {GATE_VERDICT_FILENAME}; absence is never a pass"
            )
        if verdict.get("stage_id") != stage_id:
            raise ResultBundleError(f"{verdict_path} judges stage {verdict.get('stage_id')!r}, not {stage_id!r}")
        if verdict.get("species") != species:
            raise ResultBundleError(f"{verdict_path} judges species {verdict.get('species')!r}, not {species!r}")
        if verdict.get("checkpoint_sha256") != model_hash:
            raise ResultBundleError(
                f"{verdict_path} hashes the handoff checkpoint as {verdict.get('checkpoint_sha256')!r}, "
                f"but {ANCESTOR_RECORD_NAME} records {model_hash!r}"
            )
        if verdict.get("normalization_sha256") != normalization_hash:
            raise ResultBundleError(
                f"{verdict_path} hashes the normalization sidecar as {verdict.get('normalization_sha256')!r}, "
                f"but {ANCESTOR_RECORD_NAME} records {normalization_hash!r}"
            )
        if verdict.get("task_sha256") != task_sha256:
            raise ResultBundleError(
                f"{verdict_path} records task_sha256 {verdict.get('task_sha256')!r}, "
                f"but {ANCESTOR_RECORD_NAME} records {task_sha256!r}"
            )
        gate_kind = verdict.get("gate_kind")
        if gate_kind is not None and (not isinstance(gate_kind, str) or not gate_kind.strip()):
            raise ResultBundleError(f"{verdict_path}: gate_kind must be a non-empty string or null")

        config_path = child / "stage_config.json"
        config = _read_json_object(config_path, what="stage_config.json")
        fingerprint = config.get("task_fingerprint")
        recorded_task = fingerprint.get("task_sha256") if isinstance(fingerprint, Mapping) else None
        if recorded_task is None:
            raise ResultBundleError(f"{config_path} records no task fingerprint, so the reused task is unprovable")
        if recorded_task != task_sha256:
            raise ResultBundleError(
                f"{config_path} records task_sha256 {recorded_task!r}, but {ANCESTOR_RECORD_NAME} records {task_sha256!r}"
            )

        records[entry.key] = {
            "run_id": parent_run_id,
            "model_hash": model_hash,
            "normalization_hash": normalization_hash,
            "gate_kind": gate_kind,
            "passed": verdict["passed"],
            "task_sha256": task_sha256,
            "stage_id": stage_id,
            "judged_by": judged_by,
        }
    return records


def project_ancestor_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The ``provenance.ancestors`` projection of loaded records (schema v4 fields only)."""
    from ..result_schema import ANCESTOR_RECORD_FIELDS

    return {key: {field: record[field] for field in ANCESTOR_RECORD_FIELDS} for key, record in records.items()}
