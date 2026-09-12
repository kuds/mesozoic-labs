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
  ``generate_stage_artifacts`` certifies from);
* ``reward_and_length/v1`` — from the selected checkpoint's
  ``evaluation_selected.csv`` when its rows are hash-bound to the handoff
  checkpoint, else from ``evaluations.npz`` / ``metrics.json`` (the
  historical ``best_eval_*`` fallback the judge already accepts);
* ``recovery_quality/v1`` — refused: the policy panel's per-seed successes
  the frozen ``gate_resolution.json`` is paired against were never
  persisted, so there is no evidence to re-judge from (re-judge through
  the notebook chain, which rolls the panel).

Every missing input is a refusal, never a default: a verdict re-derived
from nothing would be exactly the pass-by-absence the record exists to
prevent.  The verdict binds to the handoff pair ``select_handoff_checkpoint``
picks NOW (robust_best_model, then best_model); the stance report records
no checkpoint path, so the tool cannot prove the report scored that pair —
it says so in the log.  An existing verdict is never overwritten without
``--force``.

Run: ``python -m environments.shared.scripts.backfill_gate_verdict <stage_dir>
[--species trex] [--stage 1] [--force]``.
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
    reward_mean, reward_std = _mean_std(rewards)
    length_mean, length_std = _mean_std(lengths)
    velocity_mean, velocity_std = _mean_std(velocities)
    return {
        "best_model_reward": reward_mean,
        "best_model_std_reward": reward_std,
        "best_model_length": length_mean,
        "best_model_std_length": length_std,
        "best_model_fwd_vel": velocity_mean,
        "best_model_std_fwd_vel": velocity_std,
        "best_model_distance": float(statistics.fmean(distances)),
        "best_model_success_rate": float(statistics.fmean(1.0 if success else 0.0 for success in successes)),
    }


def backfill_gate_verdict(
    stage_dir: "str | Path",
    *,
    species: "str | None" = None,
    stage: "int | str | None" = None,
    force: bool = False,
) -> Path:
    """Re-derive and write *stage_dir*'s ``gate_verdict.json`` from its evidence.

    *species* and *stage* default to what the directory's
    ``stage_config.json`` records.  Raises :class:`BackfillError` on any
    missing evidence; returns the written path.
    """
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint
    from environments.shared.curriculum.recovery_gate import RECOVERY_GATE_KIND
    from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND
    from environments.shared.reporting import build_stage_results_from_eval_data, evaluate_stage_gate
    from environments.shared.reporting.gates import _current_task_sha256
    from environments.shared.result_bundle import GATE_VERDICT_FILENAME, write_gate_verdict
    from environments.shared.stage_manifest import StageManifestError, load_stage_manifest

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
    curriculum = record.get("curriculum")
    if not isinstance(curriculum, Mapping) or not curriculum:
        raise BackfillError("stage_config.json records no curriculum block, so the gate the stage ran under is unknown")
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
        stage_result=stage_results,
    )
    logger.info(
        "Backfilled %s: %s gate %s (%s)",
        written,
        gate_kind,
        "PASS" if passed else "FAIL",
        "; ".join(failures) if failures else "no failures",
    )
    return written


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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stage: "int | str | None" = None
    if args.stage is not None:
        stage = int(args.stage) if args.stage.isdigit() else args.stage
    try:
        written = backfill_gate_verdict(args.stage_dir, species=args.species, stage=stage, force=args.force)
    except BackfillError as exc:
        logger.error("Refusing to backfill %s: %s", args.stage_dir, exc)
        return 1
    print(written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
