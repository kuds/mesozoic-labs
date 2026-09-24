"""Refusals for a notebook session that re-enters an existing run directory.

The SB3 notebook names its run directory with ``RUN_ID`` (configuration cell):
``""`` keeps the run the storage cell resolved last in the runtime (a fresh
timestamped run on the first pass), a new id starts a fresh run, and an existing
run's id re-enters that run in place.  What the run directory already holds
constrains such a session.  Each check here reads the directory (and the chain
the resolve cell resolved), writes nothing, and raises :class:`ResultBundleError`
naming the way out:

* **A complete bundle is immutable** (``docs/RESULT_BUNDLES.md``).  Once
  ``artifact_manifest.json`` records ``status: "complete"`` the run may be
  re-entered to REUSE its certified nodes in place, never to judge or train a
  node into it: ``reporting.bundles.save_result_bundle`` refuses the next
  bundle write ("certified artifact(s) changed after publication"), and the
  chain loop reaches that write only after the node has trained.
  :func:`refuse_complete_run_session` (the resolve cell) predicts, per chain
  node, what the chain loop will do and refuses the session when any node
  would be judged or trained; :func:`refuse_write_into_complete_run` is the
  one-write form the chain loop (before a node's first write, for what the
  prediction cannot see without the reuse rule), the manual and the resume
  cells call.
* **A root widened on the command line** (``widen_checkpoint --to-stage-dir
  <run>/<stage_dirname(species, root)>``; decision D-D14 removed the notebook's
  widen cell) records ``widened_from_run_id`` in its ``stage_config.json`` run
  block and keeps its parent run's ``seed``, while the provenance the storage
  cell mints publishes ``training_seed = SEED`` (decision D-C14):
  :func:`refuse_widened_seed_mismatch` (the storage cell, before
  ``initialize_result_bundle``) refuses another ``SEED``.  Until the chain
  loop has judged that root, a trunk run would satisfy the root from another
  run and bypass it (decision D-C13):
  :func:`refuse_trunk_over_unjudged_widened_root` (the resolve cell).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ..stage_manifest import stage_dir_candidates, stage_dirname
from .constants import ANCESTOR_RECORD_NAME, ANCESTORS_DIRNAME, DEFAULT_PROVENANCE_NAME
from .errors import ResultBundleError
from .gate_verdict import GATE_VERDICT_FILENAME
from .manifest import read_bundle_status

if TYPE_CHECKING:
    from ..stage_manifest import StageEntry

#: The run-block key that marks a stage directory as a root ``widen_checkpoint``
#: wrote (one of ``config.WIDEN_LINEAGE_KEYS``, pinned by
#: ``test_result_bundle_reentry.py``; ``result_schema`` keys its widened-root
#: rules on the same field).
WIDENED_FROM_RUN_ID = "widened_from_run_id"


def _fresh_run_remedy(run_path: Path, fresh_run_id: "str | None") -> str:
    fresh = fresh_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        "Nothing has been trained or written. Work in a fresh run instead: in the configuration cell set "
        f'RUN_ID = "{fresh}" (any id no run uses yet; RUN_ID = "" would re-enter this run through this runtime\'s '
        f'_ACTIVE_RUN_ID memo) and TRUNK_FROM = "{run_path.name}", which reuses this run\'s certified nodes '
        "across runs, then run the notebook again from the configuration cell."
    )


def complete_run_writes(
    run_dir: "str | Path",
    *,
    species: str,
    chain: "Sequence[StageEntry]",
    target: "StageEntry",
    trunk_dir: "str | Path | None",
) -> tuple[str, ...]:
    """The ids of the *chain* nodes the SB3 notebook's chain loop would judge or train into *run_dir*.

    Read from the directory alone, node by node, without applying the reuse
    rule (``RETRAIN_FROM`` is :func:`refuse_complete_run_session`'s concern):

    * a node whose loop directory ``stage_dirname(species, ref)`` holds a
      ``gate_verdict.json`` writes nothing: the loop reuses it in place, or its
      verdict branch raises before any write (a verdict the reuse rule refuses
      is never judged or trained over);
    * a node other than *target* that the run holds only as an
      ``ancestors/<id>/ancestor.json`` record (no stage directory in any layout
      generation) writes nothing when *trunk_dir* is set: the loop re-resolves
      it through the trunk, and ``record_ancestor`` is a no-op for the same
      record and refuses a different one before it writes;
    * every other node — no directory, a directory without a verdict (JUDGE,
      or the interrupted-node refusal the RESUME cell then follows), or the
      target without one, which the loop looks for in this run only — would be
      judged or trained.

    Conservative on purpose: a verdict in a legacy ``stage{N}`` or bare-id
    directory does not count, because the loop judges and trains only the
    ``stage_dirname`` directory.  What this cannot see without applying the
    reuse rule — a trunk that no longer certifies a record-held node, or a
    run's own verdict refused while the trunk's copy is accepted (a record the
    run lacks) — the chain loop refuses with
    :func:`refuse_write_into_complete_run` before that node's first write.
    """
    run_path = Path(run_dir)
    writes: list[str] = []
    for entry in chain:
        if (run_path / stage_dirname(species, entry.reference) / GATE_VERDICT_FILENAME).is_file():
            continue
        held_as_record = (
            entry.id != target.id
            and trunk_dir is not None
            and (run_path / ANCESTORS_DIRNAME / entry.id / ANCESTOR_RECORD_NAME).is_file()
            and not any((run_path / name).is_dir() for name in stage_dir_candidates(species, entry.reference))
        )
        if not held_as_record:
            writes.append(entry.id)
    return tuple(writes)


def refuse_complete_run_session(
    run_dir: "str | Path",
    *,
    species: str,
    chain: "Sequence[StageEntry]",
    target: "StageEntry",
    retrain_from: "StageEntry | None",
    trunk_dir: "str | Path | None",
    fresh_run_id: "str | None" = None,
) -> None:
    """Refuse a session that would judge or train a node into a run whose bundle is complete.

    Returns without reading anything else when *run_dir* holds no manifest or a
    ``partial`` / ``failed`` one (the writer rebuilds over those, so such a run
    is continued in place).  On a ``complete`` one it raises
    :class:`ResultBundleError` when *retrain_from* is set (every node it covers
    trains, into an occupied directory that D-A20 refuses or into a new one the
    bundle write then refuses) or when :func:`complete_run_writes` names any
    node; a session that only reuses the run's certified nodes passes.  The
    message names the remedy: a fresh ``RUN_ID`` (*fresh_run_id*, else a
    timestamp taken now) with ``TRUNK_FROM`` set to this run.  The SB3
    notebook's storage cell writes nothing into a complete run
    (``initialize_result_bundle`` records no session there), so the end of the
    resolve cell, where this is called, comes before any write.
    """
    if read_bundle_status(run_dir) != "complete":
        return
    run_path = Path(run_dir)
    header = (
        f"This session re-enters run {run_path.name!r} ({run_path}), whose result bundle is complete "
        '(artifact_manifest.json records status "complete"). A complete bundle is immutable: a node judged or '
        'trained into it makes the next bundle write fail ("certified artifact(s) changed after publication") '
        "after the node has trained."
    )
    if retrain_from is not None:
        raise ResultBundleError(
            f"{header} RETRAIN_FROM = {retrain_from.id!r} trains {retrain_from.id!r} and every node below it here, "
            "and a variant is a new run (decisions D-A19, D-A20); keep RETRAIN_FROM in the fresh run. "
            + _fresh_run_remedy(run_path, fresh_run_id)
        )
    nodes = complete_run_writes(run_path, species=species, chain=chain, target=target, trunk_dir=trunk_dir)
    if nodes:
        raise ResultBundleError(
            f"{header} The chain loop would judge or train {', '.join(repr(node) for node in nodes)} here. "
            + _fresh_run_remedy(run_path, fresh_run_id)
        )


def refuse_write_into_complete_run(
    run_dir: "str | Path",
    *,
    what: str,
    fresh_run_id: "str | None" = None,
) -> None:
    """Refuse one write (*what*: "Resuming 2", "The manual node 'recovery'") into a run whose bundle is complete.

    The one-write form of :func:`refuse_complete_run_session`, called right
    before the first write it guards; a run without a manifest, or with a
    ``partial`` / ``failed`` one, passes.
    """
    if read_bundle_status(run_dir) != "complete":
        return
    run_path = Path(run_dir)
    raise ResultBundleError(
        f"{what} would write into run {run_path.name!r} ({run_path}), whose result bundle is complete "
        '(artifact_manifest.json records status "complete"), and a complete bundle is immutable. '
        + _fresh_run_remedy(run_path, fresh_run_id)
    )


def _run_block(stage_dir: Path) -> "dict[str, Any] | None":
    """The ``run`` block of *stage_dir*'s ``stage_config.json``; None when it is absent or unreadable."""
    try:
        config = json.loads((stage_dir / "stage_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    run = config.get("run") if isinstance(config, dict) else None
    return run if isinstance(run, dict) else None


def _provenance_training_seed(run_path: Path) -> Any:
    """The ``training_seed`` *run_path*'s ``provenance.json`` records; None when it is absent or unreadable."""
    try:
        provenance = json.loads((run_path / DEFAULT_PROVENANCE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return provenance.get("training_seed") if isinstance(provenance, dict) else None


def refuse_widened_seed_mismatch(run_dir: "str | Path", *, seed: int) -> None:
    """Refuse a *seed* other than the ``run.seed`` a widened root of *run_dir* kept from its parent (D-C14).

    Every stage directory of the run whose run block records
    ``widened_from_run_id`` is checked; a run with none (every fresh run, and
    every run the notebook trained itself) passes, and so does a directory
    that does not exist yet.  Called by the storage cell before
    ``initialize_result_bundle``, so a refused pass writes nothing.  The
    storage cell may have minted the run's ``provenance.json`` on an earlier
    pass, before the root was widened into the run: one that records another
    ``training_seed`` can never publish the widened root whatever *seed* is,
    and is refused with the remedy that works (widen again into a new run id).
    """
    run_path = Path(run_dir)
    if not run_path.is_dir():
        return
    minted = _provenance_training_seed(run_path)
    for config_path in sorted(run_path.glob("*/stage_config.json")):
        run = _run_block(config_path.parent)
        if run is None or WIDENED_FROM_RUN_ID not in run:
            continue
        recorded = run.get("seed")
        held = f"{config_path.parent.name} in {run_path} holds a root widened from run {run[WIDENED_FROM_RUN_ID]!r}"
        if not isinstance(recorded, int) or isinstance(recorded, bool):
            raise ResultBundleError(
                f"{held} without an integer run seed ({recorded!r}): a widened root keeps its parent's run facts, "
                "so this run's training seed cannot be checked against it (decision D-C14). The storage cell "
                "refused before writing anything."
            )
        if minted is not None and minted != recorded:
            raise ResultBundleError(
                f"{held} that keeps that run's seed {recorded}, but {run_path / DEFAULT_PROVENANCE_NAME} records "
                f"training_seed {minted!r}: it was minted under another SEED before the root was widened into this "
                "run, so this run can never publish the widened root (decision D-C14). The storage cell refused "
                "before writing anything, but no SEED fixes this run: widen the parent again into a run id no run "
                f"uses yet (widen_checkpoint --to-stage-dir <LOG_BASE>/<species>/<algo>/<new run id>/"
                f"{config_path.parent.name}), then set RUN_ID to that id and SEED = {recorded}."
            )
        if recorded != seed:
            raise ResultBundleError(
                f"SEED = {seed}, but {held} that keeps that run's seed {recorded}: this run's provenance "
                "publishes training_seed = SEED and seed replication counts it, so the two must agree (decision "
                f"D-C14). The storage cell refused before writing anything: set SEED = {recorded} in the "
                "configuration cell and run the notebook again from there."
            )


def refuse_trunk_over_unjudged_widened_root(
    run_dir: "str | Path",
    *,
    species: str,
    chain: "Sequence[StageEntry]",
    target: "StageEntry",
    retrain_from: "StageEntry | None",
    trunk_dir: "str | Path | None",
) -> None:
    """Refuse a trunk while the chain's root in *run_dir* is a widened root without a verdict (D-C13).

    The chain loop tries this run's root directory first; a widened root holds
    no ``gate_verdict.json`` until the loop's JUDGE branch writes one, so the
    loop would take the root from the trunk instead and never judge the
    widened copy.  The resolve cell calls this once the trunk is known.
    Passes when there is no trunk, when the root is the target (the loop
    never consults a trunk for it, D-A18) or is named by ``RETRAIN_FROM`` (it
    then trains, and D-A20 refuses the occupied directory before any write),
    and once the root holds a verdict.
    """
    chain = tuple(chain)
    if trunk_dir is None or not chain:
        return
    root = chain[0]
    if root.id == target.id or (retrain_from is not None and retrain_from.id == root.id):
        return
    stage_dir = Path(run_dir) / stage_dirname(species, root.reference)
    run = _run_block(stage_dir)
    if run is None or WIDENED_FROM_RUN_ID not in run or (stage_dir / GATE_VERDICT_FILENAME).is_file():
        return
    raise ResultBundleError(
        f"{stage_dir} holds {root.id!r} widened from run {run[WIDENED_FROM_RUN_ID]!r} and not judged yet, and this "
        f"session has a trunk run ({trunk_dir}): the chain loop would reuse {root.id!r} from the trunk and never "
        "judge the widened root (decision D-C13). Nothing has been trained or judged; this run's provenance.json "
        "(minted by the storage cell) already fixes SEED, N_ENVS and the plant for this run, so change only "
        'TRUNK_FROM: set TRUNK_FROM = "" in the configuration cell for the session that judges it (no node below a '
        "widened root is reusable from another run anyway: reuse chains on the parent's checkpoint digest) and run "
        "the notebook again from there."
    )
