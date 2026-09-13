"""The per-node gate verdict record: ``gate_verdict.json`` in a stage directory.

BEHAVIOR_RECIPES_PLAN §4.2 (adopted 2026-09-05): a node may be satisfied by
an existing certified checkpoint instead of trained, and the rule needs a
verdict that lives with the checkpoint it judged.  Today the verdict lives
only in run-level ``collected_results.csv`` / ``summary.json``; this file
puts it beside the handoff pair and hash-binds it to them.

Where it lives: the STAGE DIRECTORY ROOT beside ``stage_config.json`` —
never under ``models/`` (retention pruning must never touch it) and never
in the staged ``figures/`` / ``replays/`` tree
(``stage_layout.iter_generated_artifacts`` must never enumerate it).

Who writes it (Phase A, decision D-A5): ``generate_stage_artifacts``
(post-stage, evidence-backed) and ``train_curriculum``'s in-training
``CurriculumManager`` verdict, each recording ``judged_by``; a backfill tool
re-derives it for pre-Phase-A directories with ``judged_by = "backfill"``.
``train()`` and ``save_jax_stage_artifacts`` do not write it.  **Absence
never reads as a pass**: :func:`read_gate_verdict` returns None for a
missing file and raises on a malformed one.

Decision D-A22 (Phase B): the verdict also records the gate configuration
it was judged under — ``gate`` (the ``curriculum.gate_schema.gate_config_view``
projection: kind, schema version and the thresholds the kind consumes) and
``gate_sha256`` (:func:`.hashing.gate_config_sha256` over it) — so reuse
rule 7 can refuse a candidate judged under a different gate, naming the
thresholds that differ.  Every writer passes the block it judged under
(``write_gate_verdict(gate_config=...)`` is required).  A file written
before D-A22 carries neither field and still reads (the schema is
unchanged); reuse refuses it until it is re-judged.  When both fields are
present they must agree — ``gate_sha256`` is the digest of ``gate`` — which
:func:`read_gate_verdict` enforces, so a verdict edited after judging is
refused everywhere the reader is used (reuse, the ancestor records, the
bundle audit).

This module lives under ``result_bundle`` because ``result_bundle`` never
imports ``reporting``; it depends only on :func:`.hashing.sha256_file`,
:func:`.hashing.gate_config_sha256` and :func:`..file_io.atomic_write_text`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..file_io import atomic_write_text
from .errors import ResultBundleError
from .hashing import gate_config_sha256, sha256_file

GATE_VERDICT_FILENAME = "gate_verdict.json"
GATE_VERDICT_SCHEMA = "mesozoic.gate-verdict/v1"

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

#: The keys of the gate-configuration view a verdict records (D-A22): what
#: ``curriculum.gate_schema.gate_config_view`` returns, and nothing else.
_GATE_CONFIG_KEYS = frozenset({"gate_kind", "gate_schema_version", "thresholds"})

#: The ``stage_result`` projection: the keys ``save_jax_stage_artifacts``
#: persists into ``stage_result.json`` plus the SB3-side handoff and gate
#: fields, so a verdict carries the numbers the gate was judged on without
#: dragging in per-episode arrays or callbacks.
_PERSISTED_STAGE_RESULT_KEYS = (
    "stage",
    "name",
    "description",
    "timesteps",
    "duration_seconds",
    "mean_reward",
    "std_reward",
    "mean_episode_length",
    "std_episode_length",
    "mean_forward_vel",
    "std_forward_vel",
    "mean_distance_traveled",
    "mean_success_rate",
    "best_eval_reward",
    "best_eval_std",
    "best_eval_length",
    "best_eval_std_length",
    "best_eval_timestep",
    "selection_training_return",
    "selection_training_update",
    "gate_passed",
    "publication_gate_passed",
    "gate_failures",
    "gate_kind",
    "gate_schema_version",
    "best_model_reward",
    "best_model_std_reward",
    "best_model_length",
    "best_model_std_length",
    "best_model_fwd_vel",
    "best_model_std_fwd_vel",
    "best_model_distance",
    "best_model_success_rate",
    "model_path",
    "vecnorm_path",
    "plant_identity",
)


class GateVerdictError(ResultBundleError):
    """A gate verdict file exists but cannot be trusted."""


def _json_default(value: Any) -> Any:
    """Serialise the odd non-JSON scalar a stage-result dict carries."""
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"gate verdict cannot serialise {type(value).__name__}")


def _gate_record(gate_config: Any) -> dict[str, Any]:
    """The ``gate`` block as it will be written: the view, JSON-normalised, validated.

    Normalised through the same serialisation the file gets, so the digest
    recorded beside it is the digest of the block on disk (a numpy scalar
    in a threshold serialises to its item, and is hashed as such).
    """
    if (
        not isinstance(gate_config, Mapping)
        or set(gate_config) != _GATE_CONFIG_KEYS
        or not isinstance(gate_config["thresholds"], Mapping)
    ):
        raise GateVerdictError(
            "gate verdict 'gate_config' must be the curriculum.gate_schema.gate_config_view projection "
            "({gate_kind, gate_schema_version, thresholds})"
        )
    normalised = json.loads(json.dumps(dict(gate_config), default=_json_default))
    return {
        "gate_kind": normalised["gate_kind"],
        "gate_schema_version": normalised["gate_schema_version"],
        "thresholds": dict(normalised["thresholds"]),
    }


def _handoff_record(stage_dir: Path, path: "Path | None") -> tuple[str | None, str | None]:
    """The stage-dir-relative POSIX path of a handoff file and its digest."""
    if path is None:
        return None, None
    file_path = Path(path)
    if not file_path.is_file():
        raise GateVerdictError(f"gate verdict cannot hash a missing handoff file: {file_path}")
    try:
        recorded = file_path.resolve().relative_to(stage_dir.resolve()).as_posix()
    except ValueError:
        recorded = file_path.resolve().as_posix()
    return recorded, sha256_file(file_path)


def write_gate_verdict(
    stage_dir: "str | Path",
    *,
    species: str,
    stage: "int | str",
    stage_id: str,
    gate_kind: str,
    gate_schema_version: "int | str | None",
    passed: bool,
    failures: "list[str] | tuple[str, ...]",
    task_sha256: "str | None",
    judged_by: str,
    checkpoint: "Path | None",
    normalization: "Path | None",
    gate_config: Mapping[str, Any],
    stage_result: "Mapping[str, Any] | None" = None,
) -> Path:
    """Atomically write ``gate_verdict.json`` into *stage_dir*'s root.

    *stage* is the entry's canonical reference (int for a legacy stage, the
    id otherwise — the same value ``stage_config.json['stage']`` and the
    task fingerprint carry); *stage_id* is always the id.  *checkpoint* and
    *normalization* are the handoff pair the verdict is about: each is
    hashed and recorded stage-dir-relative, or recorded as null when None
    (a stage with no handoff), which makes the verdict unreusable.  A path
    that is given but missing is an error — a verdict must never describe
    a file that was not there.  *gate_config* is the
    ``curriculum.gate_schema.gate_config_view`` of the block the verdict was
    judged under (decision D-A22) — required, with no default, so a writer
    that forgets it cannot write; it is recorded as ``gate`` beside its
    digest ``gate_sha256``.
    """
    if isinstance(passed, bool) is False:
        raise GateVerdictError(f"gate verdict 'passed' must be a bool, not {passed!r}")
    if not isinstance(judged_by, str) or not judged_by.strip():
        raise GateVerdictError("gate verdict 'judged_by' must name the producer")
    gate = _gate_record(gate_config)
    root = Path(stage_dir)
    checkpoint_path, checkpoint_sha256 = _handoff_record(root, checkpoint)
    normalization_path, normalization_sha256 = _handoff_record(root, normalization)
    projected = None
    if stage_result is not None:
        projected = {key: stage_result[key] for key in _PERSISTED_STAGE_RESULT_KEYS if key in stage_result}
    record: dict[str, Any] = {
        "schema": GATE_VERDICT_SCHEMA,
        "species": species,
        "stage": stage,
        "stage_id": stage_id,
        "gate_kind": gate_kind,
        "gate_schema_version": gate_schema_version,
        "passed": passed,
        "failures": [str(failure) for failure in failures],
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "normalization": normalization_path,
        "normalization_sha256": normalization_sha256,
        "task_sha256": task_sha256,
        "gate": gate,
        "gate_sha256": gate_config_sha256(gate),
        "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judged_by": judged_by,
        "stage_result": projected,
    }
    text = json.dumps(record, indent=2, sort_keys=True, default=_json_default) + "\n"
    target = root / GATE_VERDICT_FILENAME
    atomic_write_text(target, text)
    return target


def read_gate_verdict(stage_dir: "str | Path") -> "dict[str, Any] | None":
    """Read a stage's verdict: None iff the file is absent, error if malformed.

    A missing file is the honest state of every pre-Phase-A stage directory
    and of a node trained without artifacts; it is never a pass.  ``gate``
    and ``gate_sha256`` (D-A22) may be absent — a file judged before the
    decision — but when both are present ``gate_sha256`` must be the digest
    of ``gate``: a verdict whose recorded gate was edited after judging is
    malformed, not a verdict for some other gate.
    """
    path = Path(stage_dir) / GATE_VERDICT_FILENAME
    if not path.is_file():
        return None
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateVerdictError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(verdict, dict):
        raise GateVerdictError(f"{path} must hold a JSON object")
    if verdict.get("schema") != GATE_VERDICT_SCHEMA:
        raise GateVerdictError(f"{path} declares schema {verdict.get('schema')!r}; expected {GATE_VERDICT_SCHEMA!r}")
    if not isinstance(verdict.get("passed"), bool):
        raise GateVerdictError(f"{path}: 'passed' must be a JSON boolean")
    if not isinstance(verdict.get("failures"), list):
        raise GateVerdictError(f"{path}: 'failures' must be a JSON list")
    for key in ("checkpoint_sha256", "normalization_sha256", "task_sha256", "gate_sha256"):
        digest = verdict.get(key)
        if digest is not None and (not isinstance(digest, str) or _SHA256_DIGEST.fullmatch(digest) is None):
            raise GateVerdictError(f"{path}: {key} must be sha256:<64 lowercase hex> or null")
    gate = verdict.get("gate")
    if gate is not None and not isinstance(gate, dict):
        raise GateVerdictError(f"{path}: 'gate' must be a JSON object")
    recorded_digest = verdict.get("gate_sha256")
    if gate is not None and recorded_digest is not None and gate_config_sha256(gate) != recorded_digest:
        raise GateVerdictError(
            f"{path}: gate_sha256 {recorded_digest} is not the digest of the recorded gate block "
            f"({gate_config_sha256(gate)}); the verdict was edited after judging"
        )
    return verdict


def verdict_is_reusable(verdict: "Mapping[str, Any] | None") -> bool:
    """Whether a verdict certifies a handoff pair another run may reuse.

    True only for a passed verdict that hashes BOTH handoff files; a
    missing verdict (None), a failed one, or one without a hashed
    checkpoint or normalization sidecar is not reusable.
    """
    if verdict is None or verdict.get("passed") is not True:
        return False
    return all(
        isinstance(verdict.get(key), str) and bool(verdict.get(key))
        for key in ("checkpoint_sha256", "normalization_sha256")
    )
