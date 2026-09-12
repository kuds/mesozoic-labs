"""Certified-ancestor reuse across runs (BEHAVIOR_RECIPES_PLAN §4.2).

A node may be satisfied by an existing certified checkpoint instead of
trained.  ``train_curriculum --trunk-from <run_dir>`` and the notebook's
chain loop both apply the rule this module owns, in this order, refusing
on the first failure with a reason naming it:

1. the candidate run has a stage directory for the node, in any layout
   generation (``stage{N}``, ``NN_{id}``, bare id);
2. that directory carries a ``gate_verdict.json`` that PASSED, hashes both
   files of its handoff pair, and judged this node's id;
3. the verdict's ``task_sha256`` equals the fingerprint derived from the
   CURRENT stage config and the stage directory's own recorded task — exact
   equality, the same check ``resume_same_stage`` applies (the schema-v1
   fingerprint valve is deliberately not extended to reuse);
4. the chain: a non-root node's candidate must record, in its
   ``stage_config.json`` run block, an ``initialize_next_stage`` load whose
   ``parent_checkpoint_sha256`` equals the digest of the checkpoint the
   caller resolved for the node's declared parent (``parent_model_sha256``),
   so a certified walk is reused only on top of the very stance it was
   trained from; a root's candidate must not have entered from a parent at
   all (a ``resume_same_stage`` load is not a parent).  Reuse therefore
   proceeds root-first: a child is reusable only once its parent is;
5. the handoff pair the directory selects NOW re-hashes to the verdict's
   digests, so a checkpoint rewritten after judging is refused;
6. the checkpoint's recorded plant identity validates against the current
   plant with no legacy allowance.

Two things the rule never does.  It never reuses a run's TARGET node — the
node the run exists to certify is always trained; an earlier run's certified
target is that run's deliverable and is published from there — which is why
``train_curriculum`` consults the rule for ancestors of the target only.
And it never treats the checkpoint hashes as replaceable by ids: two runs
that both certified ``stance`` produced two different checkpoints, and a
walk descends from exactly one of them.

On reuse the child run records the ancestor under
``<run_dir>/ancestors/<stage_id>/``: ``ancestor.json`` plus verbatim copies
of the ancestor stage's ``gate_verdict.json``, ``stage_config.json``,
``task_fingerprint.json`` and ``plant_identity.json`` — small records only,
never the checkpoint pair (plan A10) — and the child's lineage names the
ancestor's run as ``parent_run_id``.  A stage judged before Phase A has no
verdict file and is refused by rule 2 until it is re-judged
(``generate_stage_artifacts`` or ``scripts/backfill_gate_verdict.py``).

Lives beside ``task_fingerprint.py`` rather than under ``curriculum/`` so it
can reach ``reporting.gates`` lazily without closing the reporting <->
curriculum import cycle that module documents.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .plant_contract import PlantIdentity
    from .stage_manifest import StageEntry

logger = logging.getLogger(__name__)

#: The ancestor stage's records copied verbatim beside ``ancestor.json`` —
#: JSON sidecars only; a ``.zip`` or ``.pkl`` is never among them.
ANCESTOR_COPIED_FILES = (
    "gate_verdict.json",
    "stage_config.json",
    "task_fingerprint.json",
    "plant_identity.json",
)


class AncestorReuseError(RuntimeError):
    """A candidate stage directory cannot be reused as a certified ancestor."""


@dataclass(frozen=True)
class CertifiedAncestor:
    """A stage directory that passed every reuse rule, ready to be loaded from."""

    stage_id: str
    stage_key: str
    run_id: str
    source_run_dir: Path
    stage_dir: Path
    handoff_name: str
    #: The stem the trainer is handed (SB3 appends ``.zip`` itself).
    model_stem: str
    model_zip: Path
    model_sha256: str
    normalization_path: Path
    normalization_sha256: str
    task_sha256: str
    judged_by: str
    verdict: dict[str, Any]


def run_id_for(run_dir: "str | Path") -> str:
    """The id a run is named by as a parent: its provenance ``run_id``, else its directory name.

    ``train_curriculum`` runs never write ``provenance.json`` (no bundle is
    initialised there), so the directory name is the honest fallback for
    them.  A provenance file that exists but is unreadable or carries no
    string ``run_id`` is an error: naming that run by its directory would
    contradict its own record.
    """
    from .result_bundle import DEFAULT_PROVENANCE_NAME, ResultBundleError, load_provenance

    path = Path(run_dir)
    provenance_path = path / DEFAULT_PROVENANCE_NAME
    if not provenance_path.is_file():
        return path.name
    try:
        provenance = load_provenance(path)
    except ResultBundleError as exc:
        raise AncestorReuseError(
            f"{provenance_path} is unreadable, so the run cannot be named as a parent: {exc}"
        ) from exc
    run_id = provenance.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise AncestorReuseError(f"{provenance_path} records no run_id, so the run cannot be named as a parent")
    return run_id


def _recorded_load_lineage(stage_dir: Path) -> dict[str, Any]:
    """The load-lineage keys the stage's ``stage_config.json`` run block records.

    Empty for a stage trained from scratch or one whose config is missing
    or unreadable: rule 4 then reads "no recorded parent", which refuses a
    non-root candidate and accepts a root, the same fail-closed reading the
    bundle audit gives an absent lineage.
    """
    from .config import LOAD_LINEAGE_KEYS

    path = stage_dir / "stage_config.json"
    if not path.is_file():
        return {}
    try:
        record: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    run_block = record.get("run") if isinstance(record, Mapping) else None
    if not isinstance(run_block, Mapping):
        return {}
    return {key: run_block[key] for key in LOAD_LINEAGE_KEYS if key in run_block}


def _check_chain(stage_dir: Path, *, entry: "StageEntry", parent_model_sha256: "str | None") -> None:
    """Rule 4: the candidate's recorded parent is the checkpoint resolved for its declared parent."""
    lineage = _recorded_load_lineage(stage_dir)
    load_mode = lineage.get("load_mode")
    recorded_parent = lineage.get("parent_checkpoint_sha256")
    if entry.warm_start_from is None:
        if parent_model_sha256 is not None:
            raise ValueError(f"{entry.id!r} is a root node; no parent checkpoint digest applies to it")
        if load_mode == "initialize_next_stage":
            raise AncestorReuseError(
                f"{stage_dir} entered from a parent checkpoint ({recorded_parent}) under initialize_next_stage, "
                f"but {entry.id!r} is a root node in the current manifest"
            )
        return
    if not isinstance(parent_model_sha256, str) or not parent_model_sha256:
        raise AncestorReuseError(
            f"{entry.id!r} warm-starts from {entry.warm_start_from!r}, which has no resolved certified checkpoint "
            "in this run, so the candidate's chain cannot be verified (reuse proceeds root-first)"
        )
    if load_mode != "initialize_next_stage" or not isinstance(recorded_parent, str) or not recorded_parent:
        raise AncestorReuseError(
            f"{stage_dir}/stage_config.json records no initialize_next_stage load "
            f"(load_mode={load_mode!r}, parent_checkpoint_sha256={recorded_parent!r}), so which "
            f"{entry.warm_start_from!r} checkpoint it descends from is unknown"
        )
    if recorded_parent != parent_model_sha256:
        raise AncestorReuseError(
            f"{stage_dir} descends from {entry.warm_start_from!r} checkpoint {recorded_parent}, not the "
            f"{parent_model_sha256} resolved for {entry.warm_start_from!r} in this run; a certified node is "
            "reusable only on top of the parent it was trained from"
        )


def find_certified_ancestor(
    run_dir: "str | Path",
    *,
    species: str,
    entry: "StageEntry",
    current_task_sha256: "str | None",
    plant_identity: "PlantIdentity",
    parent_model_sha256: "str | None" = None,
) -> CertifiedAncestor:
    """Apply the §4.2 reuse rule to *run_dir*'s directory for *entry*.

    Raises :class:`AncestorReuseError` naming the first rule that failed
    (module docstring, rules 1-6); returns the ancestor otherwise.
    *current_task_sha256* is the digest derived from the CURRENT stage
    config (``derive_stage_task_fingerprint``), which the verdict and the
    directory's own record must both equal exactly.  *parent_model_sha256*
    is the digest of the checkpoint resolved for *entry*'s declared parent
    (a reused ancestor's ``model_sha256``): required for a non-root node,
    which is refused when it is None because an unresolved parent leaves the
    chain unverifiable; it must be None for a root.
    """
    from .curriculum.checkpoints import select_handoff_checkpoint
    from .plant_contract import MODEL_IDENTITY_ATTRIBUTE, PlantCompatibilityError, validate_recorded_identity
    from .reporting.gates import _current_task_sha256
    from .result_bundle import GateVerdictError, read_gate_verdict, sha256_file, verdict_is_reusable
    from .stage_manifest import stage_dir_candidates
    from .task_fingerprint import read_checkpoint_attribute

    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise AncestorReuseError(f"{run_path} is not a run directory")

    # (1) The stage directory, in any layout generation.
    candidates = stage_dir_candidates(species, entry.reference)
    stage_dir = next((run_path / name for name in candidates if (run_path / name).is_dir()), None)
    if stage_dir is None:
        raise AncestorReuseError(
            f"{run_path} has no stage directory for {entry.id!r} (looked for {', '.join(candidates)})"
        )

    # (2) A passed, reusable verdict for this node.
    try:
        verdict = read_gate_verdict(stage_dir)
    except GateVerdictError as exc:
        raise AncestorReuseError(f"{stage_dir}: {exc}") from exc
    if verdict is None:
        raise AncestorReuseError(
            f"{stage_dir} has no gate_verdict.json, so its gate verdict is unknown: a stage judged before "
            "Phase A must be re-judged (generate_stage_artifacts, or scripts/backfill_gate_verdict.py) "
            "before it can be reused"
        )
    if verdict.get("passed") is not True:
        failures = verdict.get("failures") or []
        raise AncestorReuseError(
            f"{stage_dir}/gate_verdict.json records a FAILED gate"
            + (f": {'; '.join(str(failure) for failure in failures)}" if failures else "")
        )
    if not verdict_is_reusable(verdict):
        raise AncestorReuseError(
            f"{stage_dir}/gate_verdict.json hashes no complete handoff pair (checkpoint_sha256="
            f"{verdict.get('checkpoint_sha256')!r}, normalization_sha256={verdict.get('normalization_sha256')!r}), "
            "so it certifies no checkpoint another run could load"
        )
    if verdict.get("stage_id") != entry.id:
        raise AncestorReuseError(
            f"{stage_dir}/gate_verdict.json judged stage {verdict.get('stage_id')!r}, not {entry.id!r}"
        )

    # (3) The task: verdict == current config == the directory's own record.
    recorded_task = verdict.get("task_sha256")
    if not isinstance(current_task_sha256, str) or not current_task_sha256:
        raise AncestorReuseError(f"no current task fingerprint to compare {stage_dir} against")
    if recorded_task != current_task_sha256:
        raise AncestorReuseError(
            f"{stage_dir}/gate_verdict.json was judged under task {recorded_task}, but {entry.id!r} is "
            f"configured as task {current_task_sha256} now; a certified checkpoint proves only the task it "
            "was judged on (exact equality, the schema-v1 fingerprint valve does not apply to reuse)"
        )
    own_task = _current_task_sha256(stage_dir)
    if own_task != current_task_sha256:
        raise AncestorReuseError(
            f"{stage_dir} records task {own_task} in its own stage_config.json / task_fingerprint.json, "
            f"which disagrees with the current task {current_task_sha256}"
        )

    # (4) The chain: the candidate descends from the checkpoint resolved for
    # the node's declared parent, or from nothing when the node is a root.
    _check_chain(stage_dir, entry=entry, parent_model_sha256=parent_model_sha256)

    # (5) The handoff pair selected NOW re-hashes to the verdict's digests.
    handoff = select_handoff_checkpoint(stage_dir / "models")
    if handoff is None:
        raise AncestorReuseError(
            f"{stage_dir}/models has no complete handoff pair (a checkpoint with its matched _vecnorm.pkl)"
        )
    handoff_name, model_stem, normalization = handoff
    model_zip = Path(model_stem + ".zip")
    normalization_path = Path(normalization)
    model_sha256 = sha256_file(model_zip)
    if model_sha256 != verdict["checkpoint_sha256"]:
        raise AncestorReuseError(
            f"{model_zip} does not hash to the verdict's checkpoint_sha256 (verdict judged "
            f"{verdict.get('checkpoint')!r}): the checkpoint was rewritten after judging, or a different "
            "handoff is selected now"
        )
    normalization_sha256 = sha256_file(normalization_path)
    if normalization_sha256 != verdict["normalization_sha256"]:
        raise AncestorReuseError(
            f"{normalization_path} does not hash to the verdict's normalization_sha256 (verdict judged "
            f"{verdict.get('normalization')!r}): the VecNormalize sidecar was rewritten after judging"
        )

    # (6) The plant, with no legacy allowance: an untagged ancestor is refused.
    raw_identity = read_checkpoint_attribute(model_zip, MODEL_IDENTITY_ATTRIBUTE)
    if raw_identity is not None and not isinstance(raw_identity, Mapping):
        raise AncestorReuseError(
            f"{model_zip} contains invalid plant identity metadata of type {type(raw_identity).__name__}"
        )
    try:
        validate_recorded_identity(raw_identity, plant_identity, artifact=str(model_zip), allow_legacy=False)
    except PlantCompatibilityError as exc:
        raise AncestorReuseError(str(exc)) from exc

    return CertifiedAncestor(
        stage_id=entry.id,
        stage_key=entry.key,
        run_id=run_id_for(run_path),
        source_run_dir=run_path,
        stage_dir=stage_dir,
        handoff_name=handoff_name,
        model_stem=model_stem,
        model_zip=model_zip,
        model_sha256=model_sha256,
        normalization_path=normalization_path,
        normalization_sha256=normalization_sha256,
        task_sha256=recorded_task,
        judged_by=str(verdict.get("judged_by")),
        verdict=dict(verdict),
    )


def _ancestor_record(ancestor: CertifiedAncestor) -> dict[str, Any]:
    from .result_bundle import ANCESTOR_RECORD_SCHEMA

    return {
        "schema": ANCESTOR_RECORD_SCHEMA,
        "stage_id": ancestor.stage_id,
        "stage_key": ancestor.stage_key,
        "parent_run_id": ancestor.run_id,
        "source_run_dir": str(ancestor.source_run_dir),
        "source_stage_dir": str(ancestor.stage_dir),
        "handoff": {
            "name": ancestor.handoff_name,
            "model_path": str(ancestor.model_zip.resolve()),
            "model_sha256": ancestor.model_sha256,
            "normalization_path": str(ancestor.normalization_path.resolve()),
            "normalization_sha256": ancestor.normalization_sha256,
        },
        "task_sha256": ancestor.task_sha256,
        "judged_by": ancestor.judged_by,
        "reused_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _without_timestamp(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "reused_at"}


def record_ancestor(child_run_dir: "str | Path", ancestor: CertifiedAncestor) -> Path:
    """Write ``<child_run_dir>/ancestors/<stage_id>/`` for a reused ancestor.

    Copies the ancestor stage's JSON records (:data:`ANCESTOR_COPIED_FILES`,
    those that exist) and then writes ``ancestor.json`` atomically, last, so
    an interrupted write leaves no record.  Never copies the checkpoint or
    its sidecar.  Recording the same ancestor again is a no-op; an existing
    record for the same node that differs in anything but its timestamp is
    an error, because a run cannot have two parents for one node.
    """
    from .file_io import atomic_write_text
    from .result_bundle import ANCESTOR_RECORD_NAME, ANCESTORS_DIRNAME

    target = Path(child_run_dir) / ANCESTORS_DIRNAME / ancestor.stage_id
    record = _ancestor_record(ancestor)
    record_path = target / ANCESTOR_RECORD_NAME
    if record_path.is_file():
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AncestorReuseError(f"{record_path} exists but is unreadable: {exc}") from exc
        if isinstance(existing, Mapping) and _without_timestamp(existing) == _without_timestamp(record):
            return target
        raise AncestorReuseError(
            f"{record_path} already records a different ancestor for {ancestor.stage_id!r}; a run cannot "
            "reuse two parents for one node"
        )
    target.mkdir(parents=True, exist_ok=True)
    for name in ANCESTOR_COPIED_FILES:
        source = ancestor.stage_dir / name
        if source.is_file():
            shutil.copyfile(source, target / name)
    atomic_write_text(record_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    logger.info("Recorded certified ancestor %r from run %s under %s", ancestor.stage_id, ancestor.run_id, target)
    return target
