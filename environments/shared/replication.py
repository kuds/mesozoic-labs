"""Seed replication as provenance: discovering a deliverable's replicate runs (BEHAVIOR_RECIPES_PLAN §4.5).

A run trains one seed, so replication is a property of a SET of runs.  A
replicate of a deliverable is a sibling run of the SAME recipe (decision
D-B16): under the same ``LOG_BASE/<species>/<algo>/`` directory (the layout
Colab, Drive and bucket runs keep side by side, which decision D-A23's
record-following relies on too), whose stage directory for the node holds a
passed, reusable ``gate_verdict.json`` with the same ``task_sha256`` and the
same ``gate_sha256`` (the verdict's own record of the gate it was judged
under, decision D-A22), whose ``stage_config.json`` records the same plant
identity and digests to the same ``hyperparameters_sha256`` (decision
D-A21), and which trained a DIFFERENT seed.  Two seeds of different recipes
— another gate, another algorithm block — are not replication of one
deliverable, which is the byte-identical-configuration reading
``docs/KNOWN_ISSUES.md``'s seed 42/43/44 record uses.

The count is WRITER-RECORDED (decision D-B10): ``reporting.save_result_bundle``
takes the replicates :func:`discover_replicates_for_run` finds when the
publication cell runs and records them in each deliverable's ``replication``
record, first this run, then the replicates; the catalog validates the
record and re-derives only the provisional label from the CURRENT
``certification_seeds``, never aggregating across bundles.  A replicate that
finishes later is counted when the run's publication cell is re-run with
the sibling present: a partial bundle is rebuilt, and a complete bundle —
immutable in every certified artifact and result — regenerates its derived
artifacts when the replication record is the only thing that changed.

Two pre-Phase-B shapes are handled deliberately differently (D-B16): a
sibling whose run block records no ``hyperparameters_sha256`` (saved before
D-A21) has it DERIVED from its recorded blocks
(:func:`environments.shared.config.recorded_hyperparameters_sha256`), so it
is never skipped for the missing field alone; a sibling whose verdict
records no ``gate_sha256`` (judged before D-A22) IS skipped, consistent with
reuse rule 7 and decision D-B6, and joins the count once re-judged
(``scripts/backfill_gate_verdict.py --force``).

Discovery never raises on a malformed sibling: every skip is logged at INFO
with its reason, because a neighbouring run's broken files must not stop
this run from publishing.  The RECORD is what fails closed — the schema
and the audit refuse a malformed or inconsistent replication record.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

_logger = logging.getLogger(__name__)

#: Exactly the fields of one ``replication.runs`` entry (result schema v4).
REPLICATE_RUN_FIELDS = ("run_id", "training_seed")


def _plant_mapping(plant_identity: Any) -> Mapping[str, Any] | None:
    """A plant identity as the mapping the bundle normaliser reads (a ``PlantIdentity`` or its dict)."""
    if plant_identity is None:
        return None
    to_dict = getattr(plant_identity, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(plant_identity, Mapping):
        return plant_identity
    raise TypeError(f"plant_identity must be a PlantIdentity or a mapping, not {type(plant_identity).__name__}")


def _read_stage_config(stage_dir: Path) -> dict[str, Any] | None:
    path = stage_dir / "stage_config.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _recorded_training_seed(stage_config: Mapping[str, Any]) -> int | None:
    run_block = stage_config.get("run")
    if not isinstance(run_block, Mapping):
        return None
    seed = run_block.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        return None
    return seed


def _stage_dir_or_none(run_dir: Path, reference: "int | str") -> Path | None:
    """The run's stage directory for *reference*, in any layout generation, or None."""
    from .stage_manifest import StageManifestError, find_stage_dir

    try:
        stage_dir = find_stage_dir(run_dir, reference)
    except StageManifestError:
        return None
    return stage_dir if stage_dir.is_dir() else None


def discover_replicates(
    run_dir: "str | Path",
    *,
    species: str,
    entry: Any,
    task_sha256: str,
    plant_identity: Any,
    gate_sha256: str,
    hyperparameters_sha256: str,
    training_seed: int | None = None,
) -> list[dict[str, Any]]:
    """The replicate runs of one deliverable node among *run_dir*'s siblings (decision D-B16).

    Scans the siblings of *run_dir* (``Path(run_dir).resolve().parent``),
    never *run_dir* itself nor a sibling that carries this run's own id,
    and keeps every sibling whose stage directory for *entry* holds a
    passed, reusable verdict judging ``entry.id`` under the same
    *task_sha256* and *gate_sha256*, whose ``stage_config.json`` records the
    same *plant_identity* and digests to *hyperparameters_sha256*, and whose
    run block records a training seed other than *training_seed* (when
    given) and other than any replicate already kept.  Returns
    ``[{"run_id", "training_seed"}]`` sorted by run id, one entry per run
    id.  Every skipped sibling is logged at INFO with its reason; nothing
    here raises on a malformed neighbour.
    """
    from .ancestors import AncestorReuseError, run_id_for
    from .config import recorded_hyperparameters_sha256
    from .result_bundle import (
        GateVerdictError,
        ResultBundleError,
        _normalize_plant_identity,
        read_gate_verdict,
        verdict_is_reusable,
    )

    run_path = Path(run_dir).resolve()
    try:
        own_run_id: str | None = run_id_for(run_path)
    except AncestorReuseError:
        own_run_id = None
    current_plant = _normalize_plant_identity(_plant_mapping(plant_identity), species=species)
    if current_plant is None:
        raise ResultBundleError("replicate discovery needs the run's plant identity")

    def _skip(sibling: Path, reason: str) -> None:
        _logger.info("replicates of %s for %r: skipping %s: %s", run_path.name, entry.id, sibling.name, reason)

    found: dict[str, dict[str, Any]] = {}
    seeds_taken: dict[int, str] = {}
    parent = run_path.parent
    if not parent.is_dir():
        return []
    for sibling in sorted(path for path in parent.iterdir() if path.is_dir()):
        if sibling.resolve() == run_path:
            continue
        stage_dir = _stage_dir_or_none(sibling, entry.reference)
        if stage_dir is None:
            _skip(sibling, f"no stage directory for {entry.id!r}")
            continue
        try:
            verdict = read_gate_verdict(stage_dir)
        except GateVerdictError as exc:
            _skip(sibling, f"unreadable gate_verdict.json: {exc}")
            continue
        if verdict is None:
            _skip(sibling, "no gate_verdict.json (judged before Phase A, or never judged)")
            continue
        if verdict.get("passed") is not True:
            _skip(sibling, "its gate verdict is FAILED")
            continue
        if not verdict_is_reusable(verdict):
            _skip(sibling, "its verdict hashes no complete handoff pair")
            continue
        if verdict.get("stage_id") != entry.id:
            _skip(sibling, f"its verdict judged stage {verdict.get('stage_id')!r}, not {entry.id!r}")
            continue
        if verdict.get("task_sha256") != task_sha256:
            _skip(sibling, f"judged under task {verdict.get('task_sha256')}, not {task_sha256}")
            continue
        recorded_gate = verdict.get("gate_sha256")
        if not isinstance(recorded_gate, str) or not recorded_gate:
            _skip(
                sibling,
                "its verdict records no gate_sha256 (judged before decision D-A22); re-judge it with "
                "scripts/backfill_gate_verdict.py --force before it can count as a replicate (decision D-B6)",
            )
            continue
        if recorded_gate != gate_sha256:
            _skip(sibling, f"judged under gate {recorded_gate}, not {gate_sha256}: another recipe")
            continue
        stage_config = _read_stage_config(stage_dir)
        if stage_config is None:
            _skip(sibling, "no readable stage_config.json")
            continue
        plant_value = stage_config.get("plant_identity")
        if not isinstance(plant_value, Mapping):
            # The normaliser reads a mapping; anything else (a string, a
            # list, a number) would raise out of PlantIdentity.from_mapping
            # rather than as a bundle error.
            _skip(sibling, "its recorded plant identity is not an object")
            continue
        try:
            sibling_plant = _normalize_plant_identity(plant_value, species=species)
        except ResultBundleError as exc:
            _skip(sibling, f"invalid recorded plant identity: {exc}")
            continue
        if sibling_plant != current_plant:
            _skip(sibling, "its recorded plant identity differs")
            continue
        recorded_recipe = recorded_hyperparameters_sha256(stage_config)
        if recorded_recipe is None:
            _skip(sibling, "its stage_config.json records no algorithm to digest the recipe under")
            continue
        if recorded_recipe != hyperparameters_sha256:
            _skip(sibling, f"recipe digest {recorded_recipe} differs from {hyperparameters_sha256}: another recipe")
            continue
        seed = _recorded_training_seed(stage_config)
        if seed is None:
            _skip(sibling, "its stage_config.json run block records no training seed")
            continue
        if training_seed is not None and seed == training_seed:
            _skip(sibling, f"it trained this run's own seed {seed}: a replicate is a different seed")
            continue
        try:
            run_id = run_id_for(sibling)
        except AncestorReuseError as exc:
            _skip(sibling, str(exc))
            continue
        if own_run_id is not None and run_id == own_run_id:
            _skip(sibling, f"it carries this run's own id {run_id!r}")
            continue
        if run_id in found:
            _skip(sibling, f"run id {run_id!r} is already counted")
            continue
        if seed in seeds_taken:
            _skip(sibling, f"it repeats training seed {seed} of replicate {seeds_taken[seed]!r}")
            continue
        found[run_id] = {"run_id": run_id, "training_seed": seed}
        seeds_taken[seed] = run_id
        _logger.info(
            "replicates of %s for %r: counting %s (run %r, seed %d)",
            run_path.name,
            entry.id,
            sibling.name,
            run_id,
            seed,
        )
    return [found[run_id] for run_id in sorted(found)]


def discover_replicates_for_run(
    run_dir: "str | Path",
    *,
    species: str,
    plant_identity: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Replicates of every deliverable node *run_dir* holds a certifiable verdict for, keyed by stage key.

    For each deliverable of the species' manifest whose stage directory in
    *run_dir* holds a passed, reusable ``gate_verdict.json`` carrying
    ``task_sha256`` and ``gate_sha256`` and a ``stage_config.json`` with a
    training seed and a recipe digest (recorded, else derived — D-B16),
    :func:`discover_replicates` is run against the run's siblings.  The
    result is what ``reporting.save_result_bundle(replicates=...)`` records;
    a node with no such verdict (never judged, failed, judged before
    D-A22) contributes no key, and its record then counts this run alone.
    """
    from .config import recorded_hyperparameters_sha256
    from .result_bundle import GateVerdictError, read_gate_verdict, verdict_is_reusable
    from .stage_manifest import load_stage_manifest

    run_path = Path(run_dir).resolve()
    manifest = load_stage_manifest(species)
    replicates: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest.deliverables:
        stage_dir = _stage_dir_or_none(run_path, entry.reference)
        if stage_dir is None:
            continue
        try:
            verdict = read_gate_verdict(stage_dir)
        except GateVerdictError as exc:
            _logger.info(
                "replicates of %s: %r has an unreadable verdict, counting this run alone: %s",
                run_path.name,
                entry.id,
                exc,
            )
            continue
        if verdict is None or verdict.get("passed") is not True or not verdict_is_reusable(verdict):
            continue
        task_sha256 = verdict.get("task_sha256")
        gate_sha256 = verdict.get("gate_sha256")
        if not isinstance(task_sha256, str) or not task_sha256:
            _logger.info(
                "replicates of %s: %r's verdict records no task_sha256, counting this run alone",
                run_path.name,
                entry.id,
            )
            continue
        if not isinstance(gate_sha256, str) or not gate_sha256:
            _logger.info(
                "replicates of %s: %r's verdict records no gate_sha256 (judged before decision D-A22), so its "
                "replicates cannot be matched; counting this run alone until it is re-judged",
                run_path.name,
                entry.id,
            )
            continue
        stage_config = _read_stage_config(stage_dir)
        if stage_config is None:
            _logger.info(
                "replicates of %s: %r has no readable stage_config.json, counting this run alone",
                run_path.name,
                entry.id,
            )
            continue
        recipe = recorded_hyperparameters_sha256(stage_config)
        if recipe is None:
            _logger.info(
                "replicates of %s: %r records no recipe digest, counting this run alone", run_path.name, entry.id
            )
            continue
        replicates[entry.key] = discover_replicates(
            run_path,
            species=species,
            entry=entry,
            task_sha256=task_sha256,
            plant_identity=plant_identity,
            gate_sha256=gate_sha256,
            hyperparameters_sha256=recipe,
            training_seed=_recorded_training_seed(stage_config),
        )
    return replicates
