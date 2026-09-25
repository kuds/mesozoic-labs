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
from typing import TYPE_CHECKING, Any, Mapping, Sequence

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


# ---------------------------------------------------------------------------
# Automatic trunk selection (decision D-A25)
# ---------------------------------------------------------------------------

#: The ``TRUNK_FROM`` / ``--trunk-from`` value that asks for :func:`select_trunk`
#: instead of naming a run.
AUTO_TRUNK = "auto"

#: How many refused runs :meth:`TrunkSelection.describe` names before it counts the rest;
#: every run scanned is in :attr:`TrunkSelection.candidates` whatever the count.
_DESCRIBE_REFUSALS = 20


@dataclass(frozen=True)
class TrunkCandidate:
    """One run under the log directory and how much of the chain it satisfies root-first."""

    run_dir: Path
    run_id: str
    #: The ancestors the run satisfied, root-first, each chained onto the one before.
    covered: tuple[CertifiedAncestor, ...]
    #: ``(stage_id, reason)`` for the first considered node the run could not
    #: satisfy; None when it covered every considered node.
    refusal: "tuple[str, str] | None"

    @property
    def coverage(self) -> int:
        return len(self.covered)


@dataclass(frozen=True)
class NodeSupport:
    """What a selected trunk's covered node rests on: its handoff, seed and replication."""

    stage_id: str
    run_id: str
    handoff_name: str
    training_seed: "int | None"
    judged_at: "str | None"
    #: Distinct passing training seeds of this node's task and gate among the
    #: source run's siblings, this run included (decision D-B16).
    distinct_seeds: int
    #: The ``certification_seeds`` the node's current ``[curriculum]`` block declares.
    required_seeds: int

    @property
    def provisional(self) -> bool:
        return self.distinct_seeds < self.required_seeds


@dataclass(frozen=True)
class OlderInterfaceParent:
    """A run whose ROOT node passed its gate under an older policy interface.

    Never a trunk (rule 3 refuses it: its task digest carries the old plant),
    but the run the command-line ``python -m
    environments.shared.scripts.widen_checkpoint`` (with ``--max-revision-gap``
    set to :attr:`revision_gap`) can widen into a new run's root
    (BEHAVIOR_RECIPES_PLAN §4.6, decisions D-C13, D-C17 and D-D14).
    """

    run_dir: Path
    run_id: str
    stage_id: str
    policy_interface_revision: int
    current_policy_interface_revision: int

    @property
    def revision_gap(self) -> int:
        return self.current_policy_interface_revision - self.policy_interface_revision


@dataclass(frozen=True)
class TrunkSelection:
    """The outcome of :func:`select_trunk`: which run, if any, the chain reuses its ancestors from."""

    log_dir: Path
    #: The chain nodes consulted: the target's ancestors root-first, above ``retrain_from``.
    considered: tuple[str, ...]
    #: Every run scanned, newest first (by directory name), with its coverage.
    candidates: tuple[TrunkCandidate, ...]
    selected: "TrunkCandidate | None"
    support: tuple[NodeSupport, ...]
    older_interface: tuple[OlderInterfaceParent, ...]
    #: Why no run was consulted at all (nothing to reuse, or the log directory could not be listed), else None.
    skipped: "str | None" = None

    @property
    def run_dir(self) -> "Path | None":
        """The trunk run to reuse from — what the notebook's ``TRUNK_DIR`` / the CLI's ``trunk_from`` become."""
        return None if self.selected is None else self.selected.run_dir

    def describe(self) -> str:
        """The operator-facing account: the choice, what it rests on, and what was refused and why."""
        nodes = ", ".join(self.considered) or "nothing (the root is trained here)"
        lines = [f"Trunk (auto) under {self.log_dir}: scanned {len(self.candidates)} run(s) for {nodes}"]
        if self.skipped:
            lines.append(f"  skipped: {self.skipped}")
            return "\n".join(lines)
        if self.selected is not None:
            covered = [ancestor.stage_id for ancestor in self.selected.covered]
            rest = [stage_id for stage_id in self.considered if stage_id not in covered]
            trained = f"; {', '.join(rest)} train here" if rest else ""
            lines.append(
                f"  selected run {self.selected.run_id} ({self.selected.run_dir}): covers {', '.join(covered)}{trained}"
            )
            for node in self.support:
                seed = f"seed {node.training_seed}" if node.training_seed is not None else "seed unrecorded"
                judged = f" judged {node.judged_at}" if node.judged_at else ""
                state = " - provisional" if node.provisional else ""
                lines.append(
                    f"    {node.stage_id}: {node.handoff_name} from run {node.run_id} ({seed}{judged}); "
                    f"replication {node.distinct_seeds} of {node.required_seeds} required seed(s){state}"
                )
            if self.selected.refusal is not None:
                stage_id, reason = self.selected.refusal
                lines.append(f"    {stage_id}: not covered: {reason}")
            others = [c for c in self.candidates if c is not self.selected and c.coverage]
            for candidate in others:
                covered_ids = ", ".join(ancestor.stage_id for ancestor in candidate.covered)
                lines.append(f"  also usable: run {candidate.run_id} ({candidate.run_dir.name}) covers {covered_ids}")
        elif self.considered:
            lines.append(f"  no run covers the root {self.considered[0]!r}: every node trains here")
        refused = [c for c in self.candidates if not c.coverage]
        for candidate in refused[:_DESCRIBE_REFUSALS]:
            stage_id, reason = candidate.refusal or ("?", "no node consulted")
            lines.append(f"  refused {candidate.run_dir.name}: {stage_id}: {reason}")
        if len(refused) > _DESCRIBE_REFUSALS:
            lines.append(
                f"  ... and {len(refused) - _DESCRIBE_REFUSALS} more refused run(s); TRUNK_SELECTION.candidates lists every run"
            )
        for parent in self.older_interface:
            lines.append(
                f"  older-interface parent: run {parent.run_id} ({parent.run_dir.name}) passed {parent.stage_id!r} under "
                f"policy interface r{parent.policy_interface_revision}; this checkout is "
                f"r{parent.current_policy_interface_revision}. Widen it into a new run instead of retraining the "
                f"root: python -m environments.shared.scripts.widen_checkpoint --max-revision-gap {parent.revision_gap} "
                "(its --help gives the recipe)."
            )
        lines.append("  Pin another run with TRUNK_FROM = '<run id>'; TRUNK_FROM = '' trains every node here.")
        return "\n".join(lines)


def _read_json_mapping(path: Path) -> "dict[str, Any] | None":
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _recorded_seed(stage_config: "Mapping[str, Any] | None") -> "int | None":
    run_block = stage_config.get("run") if isinstance(stage_config, Mapping) else None
    seed = run_block.get("seed") if isinstance(run_block, Mapping) else None
    return seed if isinstance(seed, int) and not isinstance(seed, bool) else None


def _recorded_plant_identity(stage_dir: Path, run_dir: Path) -> "dict[str, Any] | None":
    """The plant identity a stage recorded: its sidecar, the run's, else the handoff checkpoint's attribute."""
    identity = _read_json_mapping(stage_dir / "plant_identity.json") or _read_json_mapping(
        run_dir / "plant_identity.json"
    )
    if identity is not None:
        return identity
    from .curriculum.checkpoints import select_handoff_checkpoint
    from .plant_contract import MODEL_IDENTITY_ATTRIBUTE
    from .task_fingerprint import read_checkpoint_attribute

    try:
        handoff = select_handoff_checkpoint(stage_dir / "models")
    except OSError:
        return None
    if handoff is None:
        return None
    attribute = read_checkpoint_attribute(Path(handoff[1] + ".zip"), MODEL_IDENTITY_ATTRIBUTE)
    return dict(attribute) if isinstance(attribute, Mapping) else None


def _older_interface_root(
    run_dir: Path, run_id: str, *, species: str, root: "StageEntry", plant_identity: "PlantIdentity"
) -> "OlderInterfaceParent | None":
    """*run_dir*'s root node passed under an older policy interface that ``widen_checkpoint`` can bridge, else None.

    The hint is only offered when every field the widen tool's identity gate
    checks besides the revision agrees with the current plant: same species,
    ``physics_sha256``, ``nq``/``nv``/``nu`` and ``action_dim``, and an
    observation exactly ``COMMAND_WIDTH`` narrower (decision D-C17).  A run
    behind a physics bump is an older plant, not a widen candidate.
    """
    from .command_frame import COMMAND_WIDTH
    from .plant_contract import PlantIdentity
    from .result_bundle import GateVerdictError, read_gate_verdict
    from .stage_manifest import stage_dir_candidates

    stage_dir = next(
        (run_dir / name for name in stage_dir_candidates(species, root.reference) if (run_dir / name).is_dir()), None
    )
    if stage_dir is None:
        return None
    try:
        verdict = read_gate_verdict(stage_dir)
    except (GateVerdictError, OSError):
        return None
    if verdict is None or verdict.get("passed") is not True or verdict.get("stage_id") != root.id:
        return None
    identity = _recorded_plant_identity(stage_dir, run_dir)
    if identity is None:
        return None
    try:
        recorded = PlantIdentity.from_mapping(identity)
    except (KeyError, TypeError, ValueError):
        return None
    current = int(plant_identity.policy_interface_revision)
    if (
        recorded.species != plant_identity.species
        or int(recorded.policy_interface_revision) >= current
        or recorded.physics_sha256 != plant_identity.physics_sha256
        or (recorded.nq, recorded.nv, recorded.nu) != (plant_identity.nq, plant_identity.nv, plant_identity.nu)
        or recorded.action_dim != plant_identity.action_dim
        or recorded.observation_dim + COMMAND_WIDTH != plant_identity.observation_dim
    ):
        return None
    return OlderInterfaceParent(
        run_dir=run_dir,
        run_id=run_id,
        stage_id=root.id,
        policy_interface_revision=int(recorded.policy_interface_revision),
        current_policy_interface_revision=current,
    )


def _node_support(
    ancestor: CertifiedAncestor,
    *,
    species: str,
    entry: "StageEntry",
    plant_identity: "PlantIdentity",
    current_gate_config: "Mapping[str, Any]",
) -> NodeSupport:
    """Replication of a covered node among its SOURCE run's siblings (decision D-B16), this run counted."""
    from .config import recorded_hyperparameters_sha256
    from .curriculum.gate_schema import declared_certification_seeds
    from .replication import discover_replicates

    stage_config = _read_json_mapping(ancestor.stage_dir / "stage_config.json")
    seed = _recorded_seed(stage_config)
    recipe = recorded_hyperparameters_sha256(stage_config) if stage_config is not None else None
    distinct = 1
    if recipe is not None:
        # Siblings are counted in the LOGS tree: for an ancestor reached through a
        # followed record (D-A23) that is the outermost followed run, not the
        # source, which may be a snapshot under another run's certified_inputs/.
        sibling_of = ancestor.via[0] if ancestor.via else ancestor.source_run_dir
        replicates = discover_replicates(
            sibling_of,
            species=species,
            entry=entry,
            task_sha256=ancestor.task_sha256,
            plant_identity=plant_identity,
            gate_sha256=ancestor.gate_sha256,
            hyperparameters_sha256=recipe,
            training_seed=seed,
        )
        distinct = 1 + len(replicates)
    judged_at = ancestor.verdict.get("judged_at")
    return NodeSupport(
        stage_id=ancestor.stage_id,
        run_id=ancestor.run_id,
        handoff_name=ancestor.handoff_name,
        training_seed=seed,
        judged_at=judged_at if isinstance(judged_at, str) else None,
        distinct_seeds=distinct,
        required_seeds=declared_certification_seeds(current_gate_config, stage=entry.id),
    )


def _canonical_algorithm_label(value: Any) -> "str | None":
    from .result_bundle import ResultBundleError, canonical_algorithm

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return canonical_algorithm(value)
    except ResultBundleError:
        return value.strip().lower()


def _identity_mismatch(
    run_dir: Path, *, root: "StageEntry", species: str, algorithm: "str | None", backend: str
) -> "str | None":
    """Why *run_dir* is another species', algorithm's or backend's run, from its own records; None when it is not.

    ``provenance.json`` (a notebook run) names all three; a CLI curriculum run
    has none, so its root ``stage_config.json`` ``algorithm`` is read instead.
    Records that are missing or unreadable decide nothing: the seven rules
    still apply.
    """
    from .result_bundle import DEFAULT_PROVENANCE_NAME, ResultBundleError, load_provenance
    from .stage_manifest import stage_dir_candidates

    wanted = _canonical_algorithm_label(algorithm)
    if (run_dir / DEFAULT_PROVENANCE_NAME).is_file():
        try:
            provenance = load_provenance(run_dir)
        except (ResultBundleError, OSError, ValueError):
            provenance = None
        if provenance is not None:
            recorded_species = provenance.get("species")
            if isinstance(recorded_species, str) and recorded_species != species:
                return f"{run_dir} was trained as species {recorded_species!r}, not {species!r}"
            recorded_backend = provenance.get("backend")
            if isinstance(recorded_backend, str) and recorded_backend != backend:
                return f"{run_dir} was trained on backend {recorded_backend!r}, not {backend!r}"
            recorded = _canonical_algorithm_label(provenance.get("algorithm"))
            if wanted is not None and recorded is not None and recorded != wanted:
                return f"{run_dir} was trained with {recorded}, not {wanted}"
    if wanted is None:
        return None
    stage_dir = next(
        (run_dir / name for name in stage_dir_candidates(species, root.reference) if (run_dir / name).is_dir()), None
    )
    if stage_dir is None:
        return None
    stage_config = _read_json_mapping(stage_dir / "stage_config.json")
    recorded = _canonical_algorithm_label(stage_config.get("algorithm")) if stage_config else None
    if recorded is not None and recorded != wanted:
        return f"{stage_dir} records algorithm {recorded}, not {wanted}"
    return None


def select_trunk(
    log_dir: "str | Path",
    *,
    species: str,
    chain: "Sequence[StageEntry]",
    stage_configs: "Mapping[Any, Mapping[str, Any]]",
    plant_identity: "PlantIdentity",
    exclude: "Sequence[str | Path]" = (),
    retrain_from: "StageEntry | None" = None,
    algorithm: "str | None" = None,
    backend: str = "stable-baselines3",
    limit: "int | None" = None,
) -> TrunkSelection:
    """Choose the trunk run for *chain* from the runs under *log_dir* (decision D-A25).

    *chain* is the target's chain root-first, target last (``StageManifest.chain_for``);
    *stage_configs* is keyed by each entry's ``reference`` (``load_all_stages``).  The
    nodes consulted are the target's ancestors — the target is never reused
    across runs (D-A18) — above *retrain_from* when given (D-A19: it and its
    descendants train here).  For every run directory under *log_dir* except
    *exclude* (the run being started), newest first by directory name (run ids
    are timestamps), the §4.2 rule is applied node by node root-first with
    ``follow_records=True`` (another run's record is followed to the run that
    certified the node, D-A23), each child chained onto the ancestor found
    for its parent; the run stops at its first refusal.  The run covering the
    MOST consecutive nodes wins, the greatest directory name on a tie (the
    newest for timestamp run ids; a custom name such as ``my_run`` sorts after
    every timestamp): a whole coherent trunk, never one node from one run and
    its child from another, which rule 4 would refuse anyway.  The task
    digest and gate block per node are
    derived exactly as the chain loop and ``train_base.train`` derive them, so a
    run this selects is one the loop's own ``find_certified_ancestor`` call
    accepts as ``TRUNK_DIR``.

    Nothing is copied or written.  The result names every run scanned with
    its coverage or first refusal, the replication each covered node rests
    on (informative — a provisional ancestor is reused and labelled, never
    refused; §4.5 makes replication provenance, not a gate), and any run
    whose root passed under an older policy interface (a candidate for the
    command-line ``widen_checkpoint``, D-C13/D-C17/D-D14).  With
    *retrain_from* naming the root there is nothing to reuse.

    A run whose ``provenance.json`` names another species, *algorithm* or
    *backend*, or whose root ``stage_config.json`` records another
    algorithm, is refused before the rules run (the identity check a pinned
    ``TRUNK_FROM`` makes; the seven rules never read the algorithm, and the
    CLI's default layout keeps PPO and SAC curricula side by side).  A run
    without ``provenance.json`` is named by its directory and judged by its
    stage records alone.  No malformed or unreadable neighbour raises out of
    the scan: it is listed as refused with the error, so one bad folder on a
    mounted drive never blocks a session.
    """
    from .curriculum.gate_schema import declared_certification_seeds
    from .task_fingerprint import derive_stage_task_fingerprint

    chain = tuple(chain)
    if not chain:
        raise ValueError("select_trunk needs the target's chain (root-first, target last)")
    root_dir = Path(log_dir)
    considered = list(chain[:-1])
    if retrain_from is not None:
        ids = [entry.id for entry in chain]
        if retrain_from.id not in ids:
            raise ValueError(f"retrain_from {retrain_from.id!r} is not on the chain {ids}")
        considered = considered[: ids.index(retrain_from.id)]
    considered_ids = tuple(entry.id for entry in considered)
    if not considered:
        skipped = (
            f"RETRAIN_FROM={retrain_from.id!r} covers the root: every node trains here"
            if retrain_from is not None
            else "the target is the root of its chain and is never reused across runs: it trains here"
        )
        return TrunkSelection(
            log_dir=root_dir,
            considered=considered_ids,
            candidates=(),
            selected=None,
            support=(),
            older_interface=(),
            skipped=skipped,
        )

    context: list[tuple[StageEntry, str, Mapping[str, Any]]] = []
    for entry in considered:
        config = stage_configs[entry.reference]
        fingerprint = derive_stage_task_fingerprint(
            species=species,
            stage=entry.reference,
            backend=backend,
            env_kwargs=config.get("env_kwargs", {}),
            plant_identity=plant_identity.to_dict(),
        )
        context.append((entry, fingerprint["task_sha256"], config.get("curriculum_kwargs", {})))

    excluded = {Path(path).resolve() for path in exclude}
    runs: list[Path] = []
    try:
        if root_dir.is_dir():
            runs = sorted(
                (path for path in root_dir.iterdir() if path.is_dir() and not path.name.startswith(".")),
                key=lambda path: path.name,
                reverse=True,
            )
            runs = [path for path in runs if path.resolve() not in excluded]
    except OSError as exc:
        return TrunkSelection(
            log_dir=root_dir,
            considered=considered_ids,
            candidates=(),
            selected=None,
            support=(),
            older_interface=(),
            skipped=f"{root_dir} could not be listed: {exc}",
        )
    if limit is not None:
        runs = runs[:limit]
    root_entry = context[0][0] if context else None

    candidates: list[TrunkCandidate] = []
    older: list[OlderInterfaceParent] = []
    for run_dir in runs:
        try:
            run_id = run_id_for(run_dir)
        except AncestorReuseError:
            run_id = run_dir.name
        covered: list[CertifiedAncestor] = []
        refusal: "tuple[str, str] | None" = None
        parent_sha256: "str | None" = None
        mismatch = (
            _identity_mismatch(run_dir, root=root_entry, species=species, algorithm=algorithm, backend=backend)
            if root_entry is not None
            else None
        )
        if mismatch is not None:
            refusal = (context[0][0].id, mismatch)
        else:
            for entry, task_sha256, gate_config in context:
                try:
                    ancestor = find_certified_ancestor(
                        run_dir,
                        species=species,
                        entry=entry,
                        current_task_sha256=task_sha256,
                        plant_identity=plant_identity,
                        current_gate_config=gate_config,
                        parent_model_sha256=parent_sha256,
                        follow_records=True,
                    )
                except AncestorReuseError as exc:
                    refusal = (entry.id, str(exc))
                    break
                except OSError as exc:
                    # Rule 5 hashes the handoff pair; a checkpoint the mount cannot
                    # read (a half-synced upload, a placeholder) refuses this run only.
                    refusal = (entry.id, f"unreadable: {exc}")
                    break
                covered.append(ancestor)
                parent_sha256 = ancestor.model_sha256
        candidates.append(TrunkCandidate(run_dir=run_dir, run_id=run_id, covered=tuple(covered), refusal=refusal))
        if not covered and root_entry is not None and mismatch is None:
            try:
                hint = _older_interface_root(
                    run_dir, run_id, species=species, root=root_entry, plant_identity=plant_identity
                )
            except OSError:
                hint = None
            if hint is not None:
                older.append(hint)

    selected = max(
        (candidate for candidate in candidates if candidate.coverage),
        key=lambda candidate: (candidate.coverage, candidate.run_dir.name),
        default=None,
    )
    support: list[NodeSupport] = []
    if selected is not None:
        for ancestor, (entry, _, gate_config) in zip(selected.covered, context):
            try:
                node = _node_support(
                    ancestor,
                    species=species,
                    entry=entry,
                    plant_identity=plant_identity,
                    current_gate_config=gate_config,
                )
            except OSError as exc:
                # Replication is informative; an unreadable neighbour costs the count, never the trunk.
                logger.info("replication of %r from %s not counted: %s", entry.id, ancestor.run_id, exc)
                node = NodeSupport(
                    stage_id=ancestor.stage_id,
                    run_id=ancestor.run_id,
                    handoff_name=ancestor.handoff_name,
                    training_seed=None,
                    judged_at=None,
                    distinct_seeds=1,
                    required_seeds=declared_certification_seeds(gate_config, stage=entry.id),
                )
            support.append(node)
    selection = TrunkSelection(
        log_dir=root_dir,
        considered=considered_ids,
        candidates=tuple(candidates),
        selected=selected,
        support=tuple(support),
        older_interface=tuple(older),
    )
    logger.info("%s", selection.describe())
    return selection
