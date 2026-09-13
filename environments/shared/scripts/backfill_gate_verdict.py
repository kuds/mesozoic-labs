"""Backfill ``gate_verdict.json`` for a stage directory judged before Phase A.

Decision D-A6 (BEHAVIOR_RECIPES_PLAN §4.2): a stage directory written before
the per-node verdict record existed — the certified trex stance run
``20260810_145546``, every sweep trial — carries its verdict only in the
run-level ``collected_results.csv`` / ``summary.json``, so ancestor reuse
(``--trunk-from``, the notebook's ``TRUNK_FROM``) refuses it with a reason
naming the missing file.  This tool re-derives the verdict from the
evidence the directory already holds, through the one shared judge
``reporting.gates.evaluate_stage_gate``, and writes it with
``judged_by = "backfill"``:

* ``stance_quality/v1`` — from ``stance_gate_report.json`` (the same input
  ``generate_stage_artifacts`` certifies from), whose recorded ``thresholds``
  must state the gate being judged under: the judge reads the report's
  verdict off the panel it scored and re-derives nothing, so a report
  scored under other thresholds proves nothing about this gate and is
  refused (a stance directory whose rail moved is re-judged through the
  notebook JUDGE branch / ``generate_stage_artifacts``, which measures a
  fresh panel under the current gate — never through ``--gate current``);
* ``reward_and_length/v1`` — from the selected checkpoint's
  ``evaluation_selected.csv`` when its rows are hash-bound to the handoff
  checkpoint, else from ``evaluations.npz`` / ``metrics.json`` (the
  historical ``best_eval_*`` fallback the judge already accepts);
* ``recovery_quality/v1`` — refused: the policy panel's per-seed successes
  the frozen ``gate_resolution.json`` is paired against were never
  persisted, so there is no evidence to re-judge from (re-judge through
  the notebook chain, which rolls the panel);
* ``task_success/v1`` — from the selected checkpoint's
  ``evaluation_selected.csv`` ONLY, hash-bound to the handoff checkpoint:
  the judge forms the exact binomial lower bound from the per-episode
  ``task_success`` column, and a rounded ``mean_success_rate`` cannot
  recover ``k/n``, so a directory without that file (a CLI ``curriculum``
  hunt, whose only verdict is the in-training one) is refused.

Every missing input is a refusal, never a default: a verdict re-derived
from nothing would be exactly the pass-by-absence the record exists to
prevent.  The verdict binds to the handoff pair ``select_handoff_checkpoint``
picks NOW (robust_best_model, then best_model); the stance report records
no checkpoint path, so the tool cannot prove the report scored that pair —
it says so in the log.  An existing verdict is never overwritten without
``--force``.

Decision D-A22 (Phase B): the verdict records the gate configuration it was
judged under (``gate`` / ``gate_sha256``), and reuse rule 7 compares it
against the gate the reusing run declares.  ``--gate recorded`` (the
default) judges under the block ``stage_config.json`` recorded — what the
directory actually ran under — and digests that block; ``--gate current``
judges under the checkout's ``load_all_stages(species)[stage]`` block, the
re-judge-after-a-threshold-edit path for a directory that was never
re-judged through the notebook, and digests that one.  Either way the
verdict's ``gate`` names the block used, and either way the evidence must
be re-scorable under that block: ``reward_and_length/v1`` is (the episode
rows are re-aggregated), a stance report is not (see above).  Pre-D-A22
verdicts carry no ``gate_sha256`` and are refused by reuse rule 7;
``--force`` re-derives them.  Recovery stays refused, so a recovery
threshold edit forces a notebook re-roll of the node.

Run: ``python -m environments.shared.scripts.backfill_gate_verdict <stage_dir>
[--species trex] [--stage 1] [--force] [--gate recorded|current]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

BACKFILL_JUDGED_BY = "backfill"

#: The evidence CSV the verdict is re-derived from for reward gates.
SELECTED_EVIDENCE_CSV = "evaluation_selected.csv"

#: Which ``[curriculum]`` block the verdict is judged under and digests
#: (D-A22): the one the directory's ``stage_config.json`` recorded, or the
#: checkout's current one for the stage.
GATE_SOURCES = ("recorded", "current")

#: The thresholds ``reporting.stance_report`` records the panel was scored
#: under.  A stance verdict is read off the report, so every one of these
#: must state the gate the verdict is judged under.
_STANCE_REPORT_THRESHOLD_KEYS = (
    "min_full_horizon_fraction",
    "max_unsupported_duty",
    "max_unsupported_duty_ucb",
    "min_avg_reward",
    "min_eval_episodes",
)


class BackfillError(RuntimeError):
    """The stage directory lacks the evidence a verdict could be re-derived from."""


def _load_json(path: Path, *, what: str) -> Any:
    if not path.is_file():
        raise BackfillError(f"{what} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackfillError(f"{what} is unreadable: {path}: {exc}") from exc


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(statistics.fmean(values)), float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def selected_evidence_metrics(
    stage_dir: Path,
    *,
    model_zip: Path,
    normalization: Path,
) -> dict[str, Any] | None:
    """Selected-checkpoint metrics aggregated from ``evaluation_selected.csv``.

    ``None`` when the file is absent.  Rows that carry no
    ``checkpoint_sha256`` (evidence written before it recorded the evaluated
    checkpoint's hash) or that hash a different checkpoint or sidecar than
    the handoff pair are refused: evidence that cannot be bound to the
    checkpoint the verdict names is not evidence for it.
    """
    from environments.shared.reporting.formatting import parse_optional_bool
    from environments.shared.result_bundle import sha256_file

    path = stage_dir / SELECTED_EVIDENCE_CSV
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise BackfillError(f"{path} holds no episode rows")
    recorded = {row.get("checkpoint_sha256") for row in rows}
    if recorded != {sha256_file(model_zip)}:
        raise BackfillError(
            f"{path} is not bound to the selected handoff checkpoint {model_zip.name} "
            f"(rows record checkpoint_sha256 {sorted(str(digest) for digest in recorded)}); evidence written "
            "before the hash binding, or for another checkpoint, cannot back this verdict"
        )
    normalization_digests = {row.get("normalization_sha256") for row in rows if row.get("normalization_sha256")}
    if normalization_digests and normalization_digests != {sha256_file(normalization)}:
        raise BackfillError(
            f"{path} records VecNormalize digests {sorted(str(digest) for digest in normalization_digests)} "
            "that do not match the selected "
            f"sidecar {normalization.name}"
        )
    try:
        rewards = [float(row["reward"]) for row in rows]
        lengths = [float(row["length"]) for row in rows]
        velocities = [float(row["mean_forward_velocity"]) for row in rows]
        distances = [float(row["distance_traveled"]) for row in rows]
        successes = [parse_optional_bool(row["task_success"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise BackfillError(f"{path} is malformed: {exc}") from exc
    if any(success is None for success in successes):
        raise BackfillError(f"{path} carries a non-boolean task_success")
    from environments.shared.curriculum.recovery_gate import binomial_lcb

    reward_mean, reward_std = _mean_std(rewards)
    length_mean, length_std = _mean_std(lengths)
    velocity_mean, velocity_std = _mean_std(velocities)
    success_count = sum(1 for success in successes if success)
    return {
        "best_model_reward": reward_mean,
        "best_model_std_reward": reward_std,
        "best_model_length": length_mean,
        "best_model_std_length": length_std,
        "best_model_fwd_vel": velocity_mean,
        "best_model_std_fwd_vel": velocity_std,
        "best_model_distance": float(statistics.fmean(distances)),
        "best_model_success_rate": float(statistics.fmean(1.0 if success else 0.0 for success in successes)),
        # task_success/v1's judged numbers (plan §4.4), so the backfilled
        # verdict carries the count and bound the judge re-derives.
        "best_model_success_count": success_count,
        "best_model_n_episodes": len(successes),
        "best_model_success_lcb": binomial_lcb(success_count, len(successes)),
    }


def backfill_gate_verdict(
    stage_dir: "str | Path",
    *,
    species: "str | None" = None,
    stage: "int | str | None" = None,
    force: bool = False,
    gate: str = "recorded",
) -> Path:
    """Re-derive and write *stage_dir*'s ``gate_verdict.json`` from its evidence.

    *species* and *stage* default to what the directory's
    ``stage_config.json`` records.  *gate* names the block the verdict is
    judged under and digests (:data:`GATE_SOURCES`): ``"recorded"``, the
    directory's own ``stage_config.json`` block, or ``"current"``, the
    checkout's block for the stage.  Raises :class:`BackfillError` on any
    missing evidence; returns the written path.
    """
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint
    from environments.shared.curriculum.gate_schema import gate_config_view
    from environments.shared.curriculum.recovery_gate import RECOVERY_GATE_KIND
    from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND
    from environments.shared.curriculum.task_success_gate import TASK_SUCCESS_GATE_KIND
    from environments.shared.reporting import build_stage_results_from_eval_data, evaluate_stage_gate
    from environments.shared.reporting.gates import _current_task_sha256
    from environments.shared.result_bundle import GATE_VERDICT_FILENAME, write_gate_verdict
    from environments.shared.stage_manifest import StageManifestError, load_stage_manifest

    if gate not in GATE_SOURCES:
        raise ValueError(f"gate must be one of {GATE_SOURCES}, not {gate!r}")
    stage_path = Path(stage_dir)
    if not stage_path.is_dir():
        raise BackfillError(f"{stage_path} is not a stage directory")
    verdict_path = stage_path / GATE_VERDICT_FILENAME
    if verdict_path.is_file() and not force:
        raise BackfillError(f"{verdict_path} already exists; pass --force to re-derive it from the evidence")

    record = _load_json(stage_path / "stage_config.json", what="stage_config.json")
    if not isinstance(record, Mapping):
        raise BackfillError(f"{stage_path / 'stage_config.json'} must hold a JSON object")
    resolved_species = species or record.get("species")
    if not isinstance(resolved_species, str) or not resolved_species:
        raise BackfillError("stage_config.json records no species; pass --species")
    stage_ref: Any = stage if stage is not None else record.get("stage")
    if isinstance(stage_ref, bool) or not isinstance(stage_ref, (int, str)):
        raise BackfillError(f"stage_config.json records no usable stage ({stage_ref!r}); pass --stage")
    try:
        entry = load_stage_manifest(resolved_species).resolve(stage_ref)
    except StageManifestError as exc:
        raise BackfillError(str(exc)) from exc
    recorded_block = record.get("curriculum", record.get("curriculum_kwargs"))
    if not isinstance(recorded_block, Mapping) or not recorded_block:
        raise BackfillError("stage_config.json records no curriculum block, so the gate the stage ran under is unknown")
    curriculum: Mapping[str, Any] = recorded_block
    if gate == "current":
        curriculum = _current_curriculum_block(resolved_species, entry.reference)
        logger.info(
            "Judging %s under the checkout's current %s %r gate, not the block its stage_config.json recorded",
            stage_path,
            resolved_species,
            entry.id,
        )
    # Recorded as declared (null when undeclared; the judge then fails it).
    gate_kind: Any = curriculum.get("gate_kind")
    # The results builder is typed for the legacy integer but reads the
    # reference only to label the record; a semantic id is fine.
    stage_for_results: Any = entry.reference

    handoff = select_handoff_checkpoint(stage_path / "models")
    if handoff is None:
        raise BackfillError(
            f"{stage_path / 'models'} has no complete handoff pair (a checkpoint with its matched _vecnorm.pkl)"
        )
    handoff_name, model_stem, normalization = handoff
    model_zip = Path(model_stem + ".zip")
    normalization_path = Path(normalization)

    if gate_kind == RECOVERY_GATE_KIND:
        raise BackfillError(
            f"{entry.id!r} declares {gate_kind}: the policy panel's per-seed successes the frozen "
            "gate_resolution.json is paired against were never persisted, so the verdict cannot be re-derived "
            "from evidence; re-judge the node through the notebook chain, which rolls the panel"
        )
    stance_report: dict[str, Any] | None = None
    if gate_kind == STANCE_GATE_KIND:
        stance_report = _load_json(stage_path / "stance_gate_report.json", what="stance_gate_report.json")
        if not isinstance(stance_report, Mapping):
            raise BackfillError(f"{stage_path / 'stance_gate_report.json'} must hold a JSON object")
        stance_report = dict(stance_report)
        _require_stance_report_scored_this_gate(
            stance_report, curriculum, report_path=stage_path / "stance_gate_report.json", stage_id=entry.id, gate=gate
        )
        logger.warning(
            "The stance report records no checkpoint path; the verdict binds to the handoff pair selected now "
            "(%s). Confirm the report scored that checkpoint before trusting the backfill.",
            handoff_name,
        )

    stage_config = {
        "name": record.get("name", ""),
        "description": record.get("description", ""),
        "env_kwargs": dict(record.get("reward_weights") or {}),
        "curriculum_kwargs": dict(curriculum),
    }
    run_block = record.get("run") if isinstance(record.get("run"), Mapping) else {}
    timesteps = run_block.get("timesteps", 0) if isinstance(run_block, Mapping) else 0
    stage_results: dict[str, Any]
    if (stage_path / "evaluations.npz").is_file():
        stage_results = build_stage_results_from_eval_data(
            stage_path, stage_for_results, stage_config, timesteps=int(timesteps or 0)
        )
    else:
        stage_results = {
            "stage": entry.reference,
            "name": stage_config["name"],
            "description": stage_config["description"],
            "timesteps": int(timesteps or 0),
            "model_path": model_stem,
            "vecnorm_path": normalization,
        }
    selected = selected_evidence_metrics(stage_path, model_zip=model_zip, normalization=normalization_path)
    if selected is not None:
        stage_results.update(selected)
    elif gate_kind == TASK_SUCCESS_GATE_KIND:
        # The judge below would refuse anyway ("absent"); name the reason
        # here so the tool exits before touching the directory.
        raise BackfillError(
            f"{entry.id!r} declares task_success/v1: no {SELECTED_EVIDENCE_CSV} hash-bound to the handoff, "
            "so the verdict cannot be re-derived"
        )
    elif stance_report is None and not (stage_path / "evaluations.npz").is_file():
        raise BackfillError(
            f"{stage_path} holds neither {SELECTED_EVIDENCE_CSV} nor evaluations.npz, so there is no measurement "
            f"to judge {gate_kind!r} on"
        )

    passed, failures = evaluate_stage_gate(
        curriculum,
        stage_results,
        stage=entry.reference,
        stance_report=stance_report,
        stage_dir=stage_path,
    )
    stage_results["gate_kind"] = gate_kind
    stage_results["gate_schema_version"] = curriculum.get("gate_schema_version")
    stage_results["gate_passed"] = passed
    stage_results["publication_gate_passed"] = passed
    stage_results["gate_failures"] = failures
    written = write_gate_verdict(
        stage_path,
        species=resolved_species,
        stage=entry.reference,
        stage_id=entry.id,
        gate_kind=gate_kind,
        gate_schema_version=curriculum.get("gate_schema_version"),
        passed=passed,
        failures=failures,
        task_sha256=_current_task_sha256(stage_path),
        judged_by=BACKFILL_JUDGED_BY,
        checkpoint=model_zip,
        normalization=normalization_path,
        # D-A22: the verdict digests the block it was judged under — the
        # recorded one by default, the checkout's under --gate current.
        gate_config=gate_config_view(curriculum),
        stage_result=stage_results,
    )
    logger.info(
        "Backfilled %s: %s gate %s (%s), judged under the %s gate configuration",
        written,
        gate_kind,
        "PASS" if passed else "FAIL",
        "; ".join(failures) if failures else "no failures",
        gate,
    )
    return written


def _require_stance_report_scored_this_gate(
    stance_report: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    *,
    report_path: Path,
    stage_id: str,
    gate: str,
) -> None:
    """Refuse a stance report that was not scored under the gate being judged under.

    ``evaluate_stage_gate`` reads a ``stance_quality/v1`` verdict off the
    report's ``passed`` and re-derives nothing, so the thresholds handed to
    it are never consulted: a report scored under a lower rail would be
    minted as a pass under the current one.  The report records the
    thresholds it scored (:data:`_STANCE_REPORT_THRESHOLD_KEYS`); every one
    must state what the judged block declares, under either ``--gate``.
    A report without them cannot be shown to have scored any gate.
    """
    from environments.shared.curriculum.gate_schema import same_threshold
    from environments.shared.curriculum.stance_gate import StanceGateThresholds

    recorded = stance_report.get("thresholds")
    if not isinstance(recorded, Mapping):
        raise BackfillError(
            f"{report_path} records no thresholds, so the gate its verdict scored is unknown and cannot be "
            "shown to be the one being judged under; re-judge the node through the notebook JUDGE branch / "
            "generate_stage_artifacts, which measures a fresh panel under the current gate"
        )
    try:
        declared = StanceGateThresholds.from_curriculum(curriculum)
    except (KeyError, TypeError, ValueError) as exc:
        raise BackfillError(
            f"the {gate} {stage_id!r} block does not declare a complete stance_quality/v1 gate to judge under "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    disagreements = [
        f"{key}: report scored at {recorded.get(key)!r}, judging under {getattr(declared, key)!r} now"
        for key in _STANCE_REPORT_THRESHOLD_KEYS
        if not same_threshold(recorded.get(key), getattr(declared, key))
    ]
    if disagreements:
        raise BackfillError(
            f"{report_path} was scored under a gate that differs from the {gate} {stage_id!r} block "
            f"({'; '.join(disagreements)}): its verdict certifies only the thresholds it scored, and the judge "
            "re-derives nothing from the report, so it cannot be re-judged under this gate from the report alone; "
            "re-judge the node through the notebook JUDGE branch / generate_stage_artifacts, which measures a "
            "fresh panel under the current gate"
        )


def _current_curriculum_block(species: str, stage_ref: "int | str") -> Mapping[str, Any]:
    """The checkout's ``[curriculum]`` block for *species*' *stage_ref* (``--gate current``)."""
    from environments.shared.config import load_all_stages

    try:
        configs = load_all_stages(species)
    except Exception as exc:  # noqa: BLE001 - any loader failure is "no current gate", never a default
        raise BackfillError(f"the current {species} stage configs cannot be loaded: {exc}") from exc
    current = configs.get(stage_ref)
    if not isinstance(current, Mapping):
        raise BackfillError(
            f"the {species} checkout declares no stage {stage_ref!r} (known: {sorted(map(str, configs))}), "
            "so there is no current gate to judge under"
        )
    block = current.get("curriculum_kwargs")
    if not isinstance(block, Mapping) or not block:
        raise BackfillError(f"the current {species} stage {stage_ref!r} config declares no [curriculum] block")
    return block


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("stage_dir", help="the stage directory to write gate_verdict.json into")
    parser.add_argument("--species", default=None, help="override the species stage_config.json records")
    parser.add_argument(
        "--stage",
        default=None,
        help="override the stage reference stage_config.json records (a legacy number or a stage id)",
    )
    parser.add_argument("--force", action="store_true", help="re-derive even when a gate_verdict.json exists")
    parser.add_argument(
        "--gate",
        choices=GATE_SOURCES,
        default="recorded",
        help=(
            "which [curriculum] block to judge under and record (decision D-A22): the block the directory's "
            "stage_config.json recorded (default), or the checkout's current block for the stage — the "
            "re-judge path after a threshold edit"
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stage: "int | str | None" = None
    if args.stage is not None:
        stage = int(args.stage) if args.stage.isdigit() else args.stage
    try:
        written = backfill_gate_verdict(
            args.stage_dir, species=args.species, stage=stage, force=args.force, gate=args.gate
        )
    except BackfillError as exc:
        logger.error("Refusing to backfill %s: %s", args.stage_dir, exc)
        return 1
    print(written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
