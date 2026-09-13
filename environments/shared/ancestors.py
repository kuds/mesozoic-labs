"""Certified-ancestor reuse across runs (BEHAVIOR_RECIPES_PLAN §4.2).

A node may be satisfied by an existing certified checkpoint instead of
trained.  ``train_curriculum --trunk-from <run_dir>`` and the notebook's
chain loop both apply the rule this module owns, in this order, refusing
on the first failure with a reason naming it (the list is in evaluation
order; rule 7 was appended by decision D-A22 and keeps its number, which
is why it sits between 3 and 4):

1. the candidate run has a stage directory for the node, in any layout
   generation (``stage{N}``, ``NN_{id}``, bare id) — preferred whenever it
   exists; else the candidate holds ``ancestors/<stage_id>/ancestor.json``
   for the node (it REUSED the node itself) and the record is FOLLOWED
   (decision D-A23): its ``source_run_dir`` is resolved — as recorded, else
   the sibling of the candidate with the same name, since Colab, Drive and
   bucket layouts keep runs side by side under ``LOG_BASE/<species>/<algo>/``
   — and the rule is applied at the source, every rule below included, whose
   handoff pair must still hash to the ``handoff.model_sha256`` /
   ``handoff.normalization_sha256`` the record bound the reuse to.  At most
   :data:`ANCESTOR_RECORD_HOP_LIMIT` records are followed, a record that
   points back at a run already on the path is a cycle, and every refusal
   met on the followed path is re-raised prefixed with the hop taken.  The
   result describes the SOURCE (its run id, run directory and stage
   directory) with the followed runs in ``via``, so a child trunked from a
   run that reused a node records the run that certified it.  Following is
   OPT-IN (``follow_records=True``, which ``train_curriculum --trunk-from``
   passes, and the notebook's chain loop passes for its ``TRUNK_DIR``
   candidate only): a caller that looks in its OWN run directory first — the
   notebook tries ``RUN_DIR`` before ``TRUNK_DIR`` — must leave it off there,
   because the record in its own directory is the reuse it made itself, and
   following it would present the trunk's node as this run's own (its results
   re-entered as trained here, its lineage lost); with the default the
   pre-D-A23 refusal stands;
2. that directory carries a ``gate_verdict.json`` that PASSED, hashes both
   files of its handoff pair, and judged this node's id;
3. the verdict's ``task_sha256`` equals the fingerprint derived from the
   CURRENT stage config and the stage directory's own recorded task — exact
   equality, the same check ``resume_same_stage`` applies (the schema-v1
   fingerprint valve is deliberately not extended to reuse);
7. the gate (decision D-A22): the verdict's ``gate_sha256`` — the digest of
   the gate configuration it was judged under (``gate``: kind, schema
   version and the thresholds the kind consumes, numerics normalised;
   ``curriculum.gate_schema.gate_config_view`` / ``gate_config_sha256``) —
   equals the digest of the block the CURRENT stage config declares
   (``current_gate_config``, required: None refuses).  A verdict without
   the field was judged before D-A22 and is refused until re-judged (the
   notebook's JUDGE branch / ``generate_stage_artifacts``, or
   ``scripts/backfill_gate_verdict.py --force [--gate current]``); a
   differing digest is refused naming every threshold that differs
   (``gate_config_differences``).  Only the gate the verdict was judged
   under counts, never the block the directory's own ``stage_config.json``
   recorded: re-judging a directory under an edited gate is the intended
   path (edit a threshold, re-judge, never retrain; decision D-B8), so a
   re-judged directory is reusable under the gate it was re-judged under,
   and ``generate_stage_artifacts`` logs a warning at judge time when the
   two blocks differ.  The verdict must agree with itself — ``gate_sha256``
   is the digest of ``gate`` — which the reader enforces, so a verdict
   edited after judging is refused by rule 2.  Checked here, after the
   task and before the chain, because it is cheaper than hashing the
   handoff pair and validating the plant;
4. the chain: a non-root node's candidate must record, in its
   ``stage_config.json`` run block, an ``initialize_next_stage`` load whose
   ``parent_checkpoint_sha256`` equals the digest of the checkpoint the
   caller resolved for the node's declared parent (``parent_model_sha256``),
   so a certified walk is reused only on top of the very stance it was
   trained from; a root's candidate must not have entered from a parent at
   all (a ``resume_same_stage`` load is not a parent).  A same-stage resume
   of a node that entered from its parent keeps that edge in the run block
   (``save_stage_config`` records the continued-from checkpoint under
   ``config.RESUME_LINEAGE_KEYS`` instead), so a resumed-then-judged node
   still chains.  Reuse therefore proceeds root-first: a child is reusable
   only once its parent is;
5. the handoff pair the directory selects NOW re-hashes to the verdict's
   digests, so a checkpoint rewritten after judging is refused;
6. the checkpoint's recorded plant identity validates against the current
   plant with no legacy allowance.

Trunks therefore compose on the command line: a run that reused stance and
trained walk serves as a ``--trunk-from`` for both, and a run trunked from
it names the run that certified stance as stance's ``parent_run_id`` — one
machine-visible run directory away, never a copy of the checkpoint.  The
record names the source by ABSOLUTE path (``source_run_dir`` and
``source_stage_dir`` are resolved when written, as the handoff paths are),
so a record made from a relative ``--trunk-from logs/<run>`` follows from
any working directory, and the sibling fallback is for a moved layout only.
The notebook's chain loop opts in for its ``TRUNK_DIR`` candidate only, so
``TRUNK_FROM`` composes the same way.

Two things the rule never does.  It never reuses a run's TARGET node — the
node the run exists to certify is always trained; an earlier run's certified
target is that run's deliverable and is published from there — which is why
``train_curriculum`` consults the rule for ancestors of the target only —
and ``--retrain-from <stage_id>`` (the notebook's ``RETRAIN_FROM``, decision
D-A19) generalises that: the rule is consulted for the certified ancestors
strictly above the named node, while the node and every descendant are
trained in this run.  And it never treats the checkpoint hashes as replaceable by ids: two runs
that both certified ``stance`` produced two different checkpoints, and a
walk descends from exactly one of them.

On reuse the child run records the ancestor under
``<run_dir>/ancestors/<stage_id>/``: ``ancestor.json`` plus verbatim copies
of the ancestor stage's ``gate_verdict.json``, ``stage_config.json``,
``task_fingerprint.json`` and ``plant_identity.json`` — small records only,
never the checkpoint pair (plan A10) — and the child's lineage names the
ancestor's run as ``parent_run_id``.  A stage judged before Phase A has no
verdict file and is refused by rule 2, and a verdict without ``gate_sha256``
by rule 7, until it is re-judged (``generate_stage_artifacts`` or
``scripts/backfill_gate_verdict.py``).  The copied ``gate_verdict.json``
carries ``gate`` / ``gate_sha256`` verbatim, so the digest travels with the
record; ``ancestor.json`` itself is unchanged.

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


#: How many ``ancestors/<stage_id>/ancestor.json`` records rule 1 follows
#: before refusing (decision D-A23).  A real lineage is one hop deep — a
#: followed record names the run that certified the node, so the child's
#: own record names that run again — and a chain this long is a layout
#: rewritten by hand, which the cycle guard alone would not bound.
ANCESTOR_RECORD_HOP_LIMIT = 8


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
    #: The digest of the gate configuration the verdict was judged under
    #: (rule 7, decision D-A22): the verdict's ``gate_sha256``.
    gate_sha256: str
    #: The runs whose ``ancestors/<stage_id>/ancestor.json`` rule 1 followed
    #: to reach ``source_run_dir``, outermost first; empty for a direct hit.
    #: Logged, never persisted: ``ancestor.json`` records the source alone.
    via: tuple[Path, ...] = ()


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
    current_gate_config: "Mapping[str, Any] | None",
    parent_model_sha256: "str | None" = None,
    follow_records: bool = False,
) -> CertifiedAncestor:
    """Apply the §4.2 reuse rule to *run_dir*'s directory for *entry*.

    Raises :class:`AncestorReuseError` naming the first rule that failed
    (module docstring, rules 1-7); returns the ancestor otherwise.
    *current_task_sha256* is the digest derived from the CURRENT stage
    config (``derive_stage_task_fingerprint``), which the verdict and the
    directory's own record must both equal exactly.  *current_gate_config*
    is the CURRENT stage config's ``[curriculum]`` block
    (``curriculum_kwargs``), whose gate-configuration digest the verdict's
    ``gate_sha256`` must equal (rule 7, decision D-A22); required, with no
    default, and None refuses — like *current_task_sha256*, a missing
    comparison must not read as a match.  *parent_model_sha256*
    is the digest of the checkpoint resolved for *entry*'s declared parent
    (a reused ancestor's ``model_sha256``): required for a non-root node,
    which is refused when it is None because an unresolved parent leaves the
    chain unverifiable; it must be None for a root.

    With *follow_records* a *run_dir* that holds no stage directory for
    *entry* but an ``ancestors/<stage_id>/ancestor.json`` for it is followed
    to the run that certified the node (rule 1, decision D-A23); the result
    then describes that source and lists the followed runs in ``via``.  It
    is off by default because only a caller that knows *run_dir* is ANOTHER
    run may follow: ``train_curriculum`` passes it for ``--trunk-from``,
    while a caller that tries its own run directory first (the notebook's
    chain loop) must not, since the record it finds there is the reuse it
    made itself and following it would present the trunk's node as this
    run's own.  Without it such a *run_dir* is refused as one with no stage
    directory, naming the record it holds.
    """
    return _find_certified_ancestor(
        Path(run_dir),
        species=species,
        entry=entry,
        current_task_sha256=current_task_sha256,
        plant_identity=plant_identity,
        current_gate_config=current_gate_config,
        parent_model_sha256=parent_model_sha256,
        follow_records=follow_records,
        via=(),
    )


def _load_ancestor_record(record_path: Path, *, entry: "StageEntry") -> dict[str, Any]:
    """The ``ancestor.json`` a run keeps for *entry*, with every key the follow needs — fail closed.

    A local loader rather than ``result_bundle.ancestors.load_ancestor_records``:
    that reader loads a whole run's records against the species manifest and
    projects them to the provenance fields, dropping ``source_run_dir`` — the
    one field the follow is about — and the source is re-verified in full
    (every rule, at the source) rather than trusted from its copied sidecars.
    """
    from .result_bundle import ANCESTOR_RECORD_SCHEMA

    try:
        record: Any = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AncestorReuseError(f"{record_path} is not readable JSON, so it cannot be followed: {exc}") from exc
    if not isinstance(record, Mapping):
        raise AncestorReuseError(f"{record_path} must hold a JSON object, so it cannot be followed")
    if record.get("schema") != ANCESTOR_RECORD_SCHEMA:
        raise AncestorReuseError(
            f"{record_path} declares schema {record.get('schema')!r}, not {ANCESTOR_RECORD_SCHEMA!r}, so it cannot "
            "be followed"
        )
    if record.get("stage_id") != entry.id:
        raise AncestorReuseError(
            f"{record_path} records stage_id {record.get('stage_id')!r}, not {entry.id!r}, so it cannot be followed"
        )
    source = record.get("source_run_dir")
    if not isinstance(source, str) or not source.strip():
        raise AncestorReuseError(f"{record_path} records no source_run_dir, so it cannot be followed")
    handoff = record.get("handoff")
    if not isinstance(handoff, Mapping):
        raise AncestorReuseError(f"{record_path} records no handoff object, so it cannot be followed")
    for key in ("model_sha256", "normalization_sha256"):
        digest = handoff.get(key)
        if not isinstance(digest, str) or not digest:
            raise AncestorReuseError(
                f"{record_path} records no handoff.{key} (found {digest!r}), so the reuse it made is bound to "
                "no checkpoint and it cannot be followed"
            )
    return dict(record)


def _resolve_source_run_dir(run_path: Path, record_path: Path, recorded: str) -> Path:
    """Where the followed record's source run is on THIS machine: as recorded, else beside *run_path*."""
    recorded_path = Path(recorded)
    if recorded_path.is_dir():
        return recorded_path
    sibling = run_path.parent / recorded_path.name
    if sibling.is_dir():
        logger.info(
            "The ancestor record %s names source run %s, which is not here; using its sibling %s beside %s",
            record_path,
            recorded_path,
            sibling,
            run_path,
        )
        return sibling
    raise AncestorReuseError(
        f"{record_path} was made against a run directory this machine cannot see: neither the recorded "
        f"source {recorded_path} nor its sibling {sibling} beside {run_path} is a directory"
    )


def _follow_ancestor_record(
    run_path: Path,
    record_path: Path,
    *,
    species: str,
    entry: "StageEntry",
    current_task_sha256: "str | None",
    plant_identity: "PlantIdentity",
    current_gate_config: "Mapping[str, Any] | None",
    parent_model_sha256: "str | None",
    via: tuple[Path, ...],
) -> CertifiedAncestor:
    """Rule 1's second branch: apply the rule at the run the record names, bound to the record's digests."""
    record = _load_ancestor_record(record_path, entry=entry)
    if len(via) >= ANCESTOR_RECORD_HOP_LIMIT:
        raise AncestorReuseError(
            f"{record_path} would be ancestor record number {len(via) + 1} followed for {entry.id!r}, past the "
            f"limit of {ANCESTOR_RECORD_HOP_LIMIT} (path so far: {' -> '.join(str(hop) for hop in via)})"
        )
    source = _resolve_source_run_dir(run_path, record_path, record["source_run_dir"])
    prefix = f"followed the ancestor record in {run_path} to {source}: "
    path = (*via, run_path)
    visited = {hop.resolve() for hop in path}
    if source.resolve() in visited:
        raise AncestorReuseError(
            f"{prefix}the record points back at a run already on the followed path "
            f"({' -> '.join(str(hop) for hop in path)}), which is a cycle"
        )
    logger.info(
        "Following the ancestor record in %s for %r to run %s (parent_run_id %s)",
        run_path,
        entry.id,
        source,
        record.get("parent_run_id"),
    )
    try:
        ancestor = _find_certified_ancestor(
            source,
            species=species,
            entry=entry,
            current_task_sha256=current_task_sha256,
            plant_identity=plant_identity,
            current_gate_config=current_gate_config,
            parent_model_sha256=parent_model_sha256,
            follow_records=True,
            via=path,
        )
    except AncestorReuseError as exc:
        raise AncestorReuseError(f"{prefix}{exc}") from exc
    # The record binds the reuse to one checkpoint pair: a source rewritten
    # and re-judged since is a different checkpoint, which the run holding
    # the record never descended from.
    handoff = record["handoff"]
    if ancestor.model_sha256 != handoff["model_sha256"]:
        raise AncestorReuseError(
            f"{prefix}{ancestor.model_zip} hashes to {ancestor.model_sha256} now, but {record_path} bound the "
            f"reuse to handoff.model_sha256 {handoff['model_sha256']}; the source was rewritten since"
        )
    if ancestor.normalization_sha256 != handoff["normalization_sha256"]:
        raise AncestorReuseError(
            f"{prefix}{ancestor.normalization_path} hashes to {ancestor.normalization_sha256} now, but "
            f"{record_path} bound the reuse to handoff.normalization_sha256 {handoff['normalization_sha256']}; "
            "the source was rewritten since"
        )
    return ancestor


def _find_certified_ancestor(
    run_path: Path,
    *,
    species: str,
    entry: "StageEntry",
    current_task_sha256: "str | None",
    plant_identity: "PlantIdentity",
    current_gate_config: "Mapping[str, Any] | None",
    parent_model_sha256: "str | None",
    follow_records: bool,
    via: tuple[Path, ...],
) -> CertifiedAncestor:
    """:func:`find_certified_ancestor` with the followed path carried through the recursion."""
    from .curriculum.checkpoints import select_handoff_checkpoint
    from .curriculum.gate_schema import gate_config_differences, gate_config_sha256, gate_config_view
    from .plant_contract import MODEL_IDENTITY_ATTRIBUTE, PlantCompatibilityError, validate_recorded_identity
    from .reporting.gates import _current_task_sha256
    from .result_bundle import (
        ANCESTOR_RECORD_NAME,
        ANCESTORS_DIRNAME,
        GateVerdictError,
        read_gate_verdict,
        sha256_file,
        verdict_is_reusable,
    )
    from .stage_manifest import stage_dir_candidates
    from .task_fingerprint import read_checkpoint_attribute

    if not run_path.is_dir():
        raise AncestorReuseError(f"{run_path} is not a run directory")

    # (1) The stage directory, in any layout generation — else, for a caller
    # that opted in, the record of a reuse, followed to the run that
    # certified the node (D-A23).
    candidates = stage_dir_candidates(species, entry.reference)
    stage_dir = next((run_path / name for name in candidates if (run_path / name).is_dir()), None)
    if stage_dir is None:
        record_name = f"{ANCESTORS_DIRNAME}/{entry.id}/{ANCESTOR_RECORD_NAME}"
        record_path = run_path / ANCESTORS_DIRNAME / entry.id / ANCESTOR_RECORD_NAME
        reason = f"{run_path} has no stage directory for {entry.id!r} (looked for {', '.join(candidates)})"
        if not record_path.is_file():
            raise AncestorReuseError(reason + (f" and no {record_name} to follow" if follow_records else ""))
        if not follow_records:
            raise AncestorReuseError(
                f"{reason}; it holds {record_name} for a reuse it made itself, which is not followed here "
                "(follow_records=False: only another run's record may stand in for its stage directory)"
            )
        return _follow_ancestor_record(
            run_path,
            record_path,
            species=species,
            entry=entry,
            current_task_sha256=current_task_sha256,
            plant_identity=plant_identity,
            current_gate_config=current_gate_config,
            parent_model_sha256=parent_model_sha256,
            via=via,
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

    # (7) The gate: the verdict was judged under the gate configuration the
    # current config declares (D-A22).  Only the judged-under gate counts
    # (D-B8): a directory re-judged under an edited gate is reusable under
    # that gate.  Before the chain, the hashing and the plant: cheapest.
    if current_gate_config is None:
        raise AncestorReuseError(f"no current gate configuration to compare {stage_dir} against")
    recorded_digest = verdict.get("gate_sha256")
    if not isinstance(recorded_digest, str) or not recorded_digest:
        raise AncestorReuseError(
            f"{stage_dir}/gate_verdict.json records no gate_sha256 (judged before decision D-A22), so the gate "
            "it certifies under is unknown: re-judge it (the notebook JUDGE branch / generate_stage_artifacts, "
            "or scripts/backfill_gate_verdict.py --force [--gate current]) before it can be reused"
        )
    current_view = gate_config_view(current_gate_config)
    current_digest = gate_config_sha256(current_view)
    if recorded_digest != current_digest:
        raw_gate = verdict.get("gate")
        recorded_gate: Mapping[str, Any] = raw_gate if isinstance(raw_gate, Mapping) else {}
        raw_thresholds = recorded_gate.get("thresholds")
        recorded_thresholds: Mapping[str, Any] = raw_thresholds if isinstance(raw_thresholds, Mapping) else {}
        if not isinstance(raw_gate, Mapping):
            # The reader allows the digest without the block; without the
            # block there is nothing to name, and "judged at None" for every
            # key would read as thresholds that were unset, not unrecorded.
            named = "not nameable (the verdict records its gate digest but no gate block)"
        else:
            differences = gate_config_differences(recorded_thresholds, current_view)
            named = ", ".join(differences) or "none nameable (gate kind or schema version differs)"
        raise AncestorReuseError(
            f"{stage_dir}/gate_verdict.json was judged under gate {recorded_digest} "
            f"({recorded_gate.get('gate_kind', verdict.get('gate_kind'))}), but {entry.id!r} declares gate "
            f"{current_digest} ({current_view['gate_kind']}) now; differing thresholds: {named}; a certified "
            "checkpoint proves only the gate it was judged against — re-judge it under the current gate (the "
            "notebook JUDGE branch / generate_stage_artifacts, or scripts/backfill_gate_verdict.py --force "
            "--gate current) to reuse it"
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
        gate_sha256=recorded_digest,
        via=via,
    )


def _ancestor_record(ancestor: CertifiedAncestor) -> dict[str, Any]:
    from .result_bundle import ANCESTOR_RECORD_SCHEMA

    return {
        "schema": ANCESTOR_RECORD_SCHEMA,
        "stage_id": ancestor.stage_id,
        "stage_key": ancestor.stage_key,
        "parent_run_id": ancestor.run_id,
        # Absolute, like the handoff paths below: a record made from a
        # relative ``--trunk-from logs/<run>`` is followed (D-A23) from any
        # working directory, not only the one it was written from.
        "source_run_dir": str(Path(ancestor.source_run_dir).resolve()),
        "source_stage_dir": str(Path(ancestor.stage_dir).resolve()),
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
    logger.info(
        "Recorded certified ancestor %r from run %s under %s%s",
        ancestor.stage_id,
        ancestor.run_id,
        target,
        f" (resolved via {' -> '.join(str(hop) for hop in ancestor.via)})" if ancestor.via else "",
    )
    return target
