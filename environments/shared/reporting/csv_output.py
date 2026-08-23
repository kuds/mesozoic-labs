"""CSV result writers.

Holds the canonical ``collected_results.csv`` schema plus the single shared
CSV writer used by single-run training, sweep collection, and CLI curriculum
training, and the per-episode evaluation-evidence writer."""

from __future__ import annotations

import csv as _csv
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .formatting import _optional_metric, parse_optional_bool

logger = logging.getLogger(__name__)


# Canonical metric column order for collected_results.csv.  Both the
# single-run notebook (``save_results_csv``) and the sweep result
# collector (``sweep/results.write_results_csv``) reference this list
# so that all CSVs share a consistent schema.
CSV_METRIC_COLUMNS: list[str] = [
    "best_mean_reward",
    "best_mean_episode_length",
    "last_mean_reward",
    "last_mean_episode_length",
    "mean_forward_vel",
    "std_forward_vel",
    "mean_distance_traveled",
    "mean_success_rate",
    "training_duration_seconds",
    "reward_threshold",
    "ep_length_threshold",
    "forward_vel_threshold",
    "success_rate_threshold",
    "stage_passed",
    "quality_score",
    "quality_rank",
]


def _compute_fieldnames(
    rows: list[dict[str, Any]],
    fixed_columns: list[str] | None = None,
) -> list[str]:
    """Derive ordered fieldnames from *rows*.

    Column order: *fixed_columns* → hyperparameter columns (sorted) →
    ``CSV_METRIC_COLUMNS`` → ``eval_*`` columns (sorted).

    Any key in a row dict that is not in *fixed_columns*,
    ``CSV_METRIC_COLUMNS``, or prefixed with ``eval_`` is treated as a
    hyperparameter column.
    """
    if fixed_columns is None:
        fixed_columns = []
    eval_cols: list[str] = sorted({k for row in rows for k in row if k.startswith("eval_")})
    all_known = set(fixed_columns) | set(CSV_METRIC_COLUMNS) | set(eval_cols)
    hparam_cols: list[str] = sorted({k for row in rows for k in row if k not in all_known})
    return fixed_columns + hparam_cols + CSV_METRIC_COLUMNS + eval_cols


def write_results_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    fixed_columns: list[str] | None = None,
    append: bool = False,
) -> Path:
    """Write (or append) result rows to a CSV file.

    This is the single shared CSV writer used by single-run training,
    sweep result collection, and CLI curriculum training.  All callers
    build a flat row dict with prefixed hyperparameter keys (``ppo_*``,
    ``env_*``, …), canonical metric keys from :data:`CSV_METRIC_COLUMNS`,
    and optional ``eval_*`` quality-metric keys, then delegate the actual
    file I/O to this function.

    Args:
        rows: Flat result dicts (one per trial/stage).
        path: Output CSV path.  ``gs://`` URIs are supported for batch
            writes (the file is written locally first, then uploaded).
        fixed_columns: Column names that appear first in the header, in
            the order given.  Remaining non-metric, non-eval keys are
            treated as hyperparameter columns and sorted alphabetically.
            When *None*, all non-metric/non-eval keys are sorted.
        append: When *True*, rows are appended to an existing file.  If
            the file does not yet exist it is created with a header.  If
            new keys appear that were not in the original header the file
            is rewritten with the expanded column set.  Append mode does
            not support ``gs://`` URIs.

    Returns:
        Path to the written CSV file.
    """
    import tempfile

    path_str = str(path)
    is_gcs = path_str.startswith("gs://")

    if append and is_gcs:
        raise ValueError("Append mode is not supported for gs:// URIs")

    if not rows:
        if not append:
            logger.warning("No result rows to write — skipping CSV")
        return Path(path_str)

    # ── Append mode ────────────────────────────────────────────────────
    if append:
        local_path = Path(path_str)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():
            fieldnames = _compute_fieldnames(rows, fixed_columns)
            with open(local_path, "w", newline="") as f:
                writer = _csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(local_path, "r", newline="") as f:
                reader = _csv.DictReader(f)
                existing_fieldnames: list[str] = list(reader.fieldnames or [])
                existing_rows = list(reader)

            new_keys = [k for row in rows for k in row if k not in existing_fieldnames]
            if new_keys:
                # Rewrite with canonical column ordering so new keys land
                # in the correct position.
                all_rows = existing_rows + list(rows)
                fieldnames = _compute_fieldnames(all_rows, fixed_columns)
                with open(local_path, "w", newline="") as f:
                    writer = _csv.DictWriter(
                        f,
                        fieldnames=fieldnames,
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(existing_rows)
                    writer.writerows(rows)
            else:
                with open(local_path, "a", newline="") as f:
                    writer = _csv.DictWriter(
                        f,
                        fieldnames=existing_fieldnames,
                        extrasaction="ignore",
                    )
                    writer.writerows(rows)

        logger.info("Results CSV updated: %s", local_path)
        return local_path

    # ── Batch mode ─────────────────────────────────────────────────────
    if is_gcs:
        local_path = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
    else:
        local_path = Path(path_str)
        local_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _compute_fieldnames(rows, fixed_columns)
    with open(local_path, "w", newline="") as f:
        writer = _csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    if is_gcs:
        from google.cloud import storage

        without_scheme = path_str[len("gs://") :]
        bucket_name, _, blob_name = without_scheme.partition("/")
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            bucket.blob(blob_name).upload_from_filename(str(local_path))
        finally:
            local_path.unlink(missing_ok=True)

    logger.info("Results CSV written to: %s", path_str)
    return Path(path_str)


def build_results_csv_rows(
    stage_results_list: Sequence[Mapping[str, Any]],
    stage_configs: "Mapping[int | str, Mapping[str, Any]]",
    species: str,
    algorithm: str,
    seed: int,
    *,
    backend: str | None = None,
    run_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build CSV rows from the same stage-result values used by JSON export."""
    from ..result_bundle import canonical_algorithm, canonical_backend
    from ..result_schema import RESULT_SCHEMA_VERSION
    from .summaries import _stage_reference

    public_algorithm = canonical_algorithm(algorithm)
    public_backend = canonical_backend(algorithm, backend)
    algorithm_key = public_algorithm.lower()
    rows: list[dict[str, Any]] = []
    for r in stage_results_list:
        # A stage reference, not int(): legacy stages keep their historical
        # numbers in the ``stage`` column and semantic stages (recovery)
        # write their id, matching the load_all_stages keying the
        # *stage_configs* mapping uses.
        stage = _stage_reference(r["stage"])
        cfg = stage_configs[stage]

        row: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": run_id or "",
            "species": species,
            "algorithm": public_algorithm,
            "backend": public_backend,
            "seed": seed,
            "stage": stage,
            "timesteps": int(r["timesteps"]),
        }
        if provenance:
            for key in ("repository_commit", "config_hash", "model_hash"):
                row[key] = provenance.get(key) or ""
        plant_identity = r.get("plant_identity")
        if isinstance(plant_identity, Mapping):
            row.update({f"plant_{key}": value for key, value in plant_identity.items()})

        # ── Hyperparameters (mirroring sweep CSV key names) ─────────
        for key, val in cfg.get("env_kwargs", {}).items():
            row[f"env_{key}"] = val
        if public_backend == "jax-mjx":
            algo_key = "jax_kwargs"
        else:
            algo_key = "sac_kwargs" if algorithm_key == "sac" else "ppo_kwargs"
        for key, val in cfg.get(algo_key, {}).items():
            if key == "policy_kwargs":
                # Flatten net_arch to a string like the sweep CSV does
                net_arch = val.get("net_arch", [])
                row[f"{algorithm_key}_net_arch"] = str(net_arch)
            elif key == "verbose":
                continue
            else:
                row[f"{algorithm_key}_{key}"] = val
        for key, val in cfg.get("curriculum_kwargs", {}).items():
            row[f"curriculum_{key}"] = val

        # ── Metrics ─────────────────────────────────────────────────
        row["best_mean_reward"] = _optional_metric(r.get("best_eval_reward"), digits=2)
        row["best_mean_reward_std"] = _optional_metric(r.get("best_eval_std"), digits=2)
        row["best_eval_step"] = _optional_metric(
            r.get("best_eval_timestep", r.get("best_eval_step")),
        )
        row["best_mean_episode_length"] = _optional_metric(r.get("best_eval_length"), digits=1)
        row["last_mean_reward"] = _optional_metric(r.get("mean_reward"), digits=2)
        row["last_mean_reward_std"] = _optional_metric(r.get("std_reward"), digits=2)
        row["last_mean_episode_length"] = _optional_metric(r.get("mean_episode_length"), digits=1)
        row["last_mean_episode_length_std"] = _optional_metric(r.get("std_episode_length"), digits=1)
        row["mean_forward_vel"] = _optional_metric(r.get("mean_forward_vel"), digits=3)
        row["std_forward_vel"] = _optional_metric(r.get("std_forward_vel"), digits=3)
        row["mean_distance_traveled"] = _optional_metric(r.get("mean_distance_traveled"), digits=3)
        row["mean_success_rate"] = _optional_metric(r.get("mean_success_rate"), digits=4)
        row["training_duration_seconds"] = round(r.get("duration_seconds", 0.0), 1)
        row["selected_model_mean_reward"] = _optional_metric(r.get("best_model_reward"), digits=2)
        row["selected_model_reward_std"] = _optional_metric(r.get("best_model_std_reward"), digits=2)
        row["selected_model_mean_episode_length"] = _optional_metric(r.get("best_model_length"), digits=1)
        row["selected_model_episode_length_std"] = _optional_metric(r.get("best_model_std_length"), digits=1)
        row["selected_model_mean_forward_vel"] = _optional_metric(r.get("best_model_fwd_vel"), digits=3)
        row["selected_model_mean_forward_vel_std"] = _optional_metric(
            r.get("best_model_std_fwd_vel"),
            digits=3,
        )
        row["selected_model_mean_distance"] = _optional_metric(r.get("best_model_distance"), digits=3)
        row["selected_model_success_rate"] = _optional_metric(r.get("best_model_success_rate"), digits=4)

        # Curriculum thresholds
        cur = cfg.get("curriculum_kwargs", {})
        row["reward_threshold"] = cur.get("min_avg_reward", "")
        row["ep_length_threshold"] = cur.get("min_avg_episode_length", "")
        row["forward_vel_threshold"] = cur.get("min_avg_forward_vel", "")
        row["success_rate_threshold"] = cur.get("min_success_rate", "")
        # stance_quality/v1. Without these a stance-gated stage exports only
        # its reward RAIL, reading as though reward were the gate.
        row["gate_kind"] = cur.get("gate_kind", "")
        row["full_horizon_fraction_threshold"] = cur.get("min_full_horizon_fraction", "")
        row["unsupported_duty_ceiling"] = cur.get("max_unsupported_duty", "")
        row["unsupported_duty_ucb_ceiling"] = cur.get("max_unsupported_duty_ucb", "")
        row["stage_passed"] = r.get("publication_gate_passed", "")
        row["publication_gate_passed"] = r.get("publication_gate_passed", "")

        # Quality evaluation metrics (eval_* keys from quality eval)
        for key, val in r.items():
            if key.startswith("eval_"):
                row[key] = val

        rows.append(row)
    return rows


def save_results_csv(
    stage_results_list: list[dict[str, Any]],
    stage_configs: "dict[int | str, dict[str, Any]]",
    species: str,
    algorithm: str,
    seed: int,
    run_dir: "str | Path",
    *,
    backend: str | None = None,
    run_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save canonical stage rows to ``collected_results.csv``.

    Hyperparameters are retained for sweep analysis, while every metric shared
    with ``summary.json`` is derived from the same in-memory stage results.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    rows = build_results_csv_rows(
        stage_results_list,
        stage_configs,
        species,
        algorithm,
        seed,
        backend=backend,
        run_id=run_id,
        provenance=provenance,
    )

    plant_columns = sorted({key for row in rows for key in row if key.startswith("plant_")})
    return write_results_csv(
        rows,
        run_path / "collected_results.csv",
        fixed_columns=[
            "schema_version",
            "run_id",
            "species",
            "algorithm",
            "backend",
            "seed",
            "stage",
            "timesteps",
            "repository_commit",
            "config_hash",
            "model_hash",
            *plant_columns,
        ],
    )


def save_evaluation_episodes(
    stage_dir: str | Path,
    *,
    rewards: Sequence[Any],
    lengths: Sequence[Any],
    forward_velocities: Sequence[Any],
    distances: Sequence[Any],
    successes: Sequence[Any],
    evaluation_seed: int,
    checkpoint_label: str,
) -> Path:
    """Persist per-episode evaluation evidence instead of only aggregates."""
    if not isinstance(evaluation_seed, int) or isinstance(evaluation_seed, bool) or evaluation_seed < 0:
        raise ValueError("evaluation_seed must be a non-negative integer")
    if not checkpoint_label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in checkpoint_label
    ):
        raise ValueError("checkpoint_label must use lowercase letters, digits, '_' or '-'")
    counts = {
        len(rewards),
        len(lengths),
        len(forward_velocities),
        len(distances),
        len(successes),
    }
    if len(counts) != 1:
        raise ValueError("evaluation metric sequences must have equal lengths")
    if counts == {0}:
        raise ValueError("evaluation evidence must contain at least one episode")

    output = Path(stage_dir) / f"evaluation_{checkpoint_label}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (reward, length, forward_velocity, distance, success) in enumerate(
        zip(rewards, lengths, forward_velocities, distances, successes, strict=True),
        start=1,
    ):
        parsed_success = parse_optional_bool(success)
        if parsed_success is None:
            raise ValueError(f"evaluation success at episode {index} is not boolean")
        numeric_values = {
            "reward": float(reward),
            "forward velocity": float(forward_velocity),
            "distance": float(distance),
        }
        nonfinite = [name for name, value in numeric_values.items() if not math.isfinite(value)]
        if nonfinite:
            raise ValueError(f"evaluation episode {index} contains non-finite values for: {', '.join(nonfinite)}")
        parsed_length = int(length)
        if isinstance(length, bool) or parsed_length <= 0 or float(length) != parsed_length:
            raise ValueError(f"evaluation length at episode {index} must be a positive integer")
        rows.append(
            {
                "episode": index,
                "evaluation_seed": evaluation_seed,
                "checkpoint": checkpoint_label,
                "reward": numeric_values["reward"],
                "length": parsed_length,
                "mean_forward_velocity": numeric_values["forward velocity"],
                "distance_traveled": numeric_values["distance"],
                # "task_success", not "success": this is the stage's TASK event
                # (bite/strike/food-reached), which a stance-gated stage can
                # never emit, so it reads False in every stage-1 row by
                # construction. Under the bare name it was repeatedly misread
                # as the gate verdict — that lives in stance_gate_report.{txt,
                # json} and stage_summary.txt, never here (gate-pass
                # postmortem, follow-up 1).
                "task_success": parsed_success,
            }
        )
    fieldnames = [
        "episode",
        "evaluation_seed",
        "checkpoint",
        "reward",
        "length",
        "mean_forward_velocity",
        "distance_traveled",
        "task_success",
    ]
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = _csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output
