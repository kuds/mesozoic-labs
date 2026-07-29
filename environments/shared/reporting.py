"""Training result reporting utilities.

Provides functions to write human-readable stage and training summaries as
well as machine-readable JSON result files.  These were originally defined
inline in the Colab training notebook and are now shared so that both the
notebook and CLI training scripts can produce consistent output.
"""

from __future__ import annotations

import csv as _csv
import json as _json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .plant_contract import PlantIdentity

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


def parse_optional_bool(value: Any) -> bool | None:
    """Parse a serialized boolean without treating non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        if value == 1:
            return True
        if value == 0:
            return False
    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        try:
            scalar = scalar_item()
        except (TypeError, ValueError):
            return None
        if scalar is not value:
            return parse_optional_bool(scalar)
    return None


def evaluate_recorded_gate(
    curriculum: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> bool | None:
    """Evaluate a curriculum gate only when every enabled metric is recorded.

    Evaluations must be chronological. ``None`` means the available records
    cannot prove either a pass or a failure—for example, when a velocity or
    success-rate gate is enabled but that metric was not saved.
    """
    criteria: list[tuple[str, float]] = []
    if curriculum.get("min_avg_reward") is not None:
        criteria.append(("mean_reward", float(curriculum["min_avg_reward"])))
    if curriculum.get("min_avg_episode_length") is not None:
        criteria.append(("mean_episode_length", float(curriculum["min_avg_episode_length"])))
    if float(curriculum.get("min_avg_forward_vel") or 0.0) > 0.0:
        criteria.append(("mean_forward_vel", float(curriculum["min_avg_forward_vel"])))
    if float(curriculum.get("min_success_rate") or 0.0) > 0.0:
        criteria.append(("mean_success_rate", float(curriculum["min_success_rate"])))
    if not criteria or not evaluations:
        return None

    min_eval_episodes = int(curriculum.get("min_eval_episodes", 10))
    required_consecutive = int(curriculum.get("required_consecutive", 3))
    consecutive = 0
    incomplete = False
    for evaluation in evaluations:
        required_values = [evaluation.get(key) for key, _ in criteria]
        n_episodes = evaluation.get("n_episodes")
        if any(value is None for value in required_values) or n_episodes is None:
            incomplete = True
            consecutive = 0
            continue

        passes = int(n_episodes) >= min_eval_episodes and all(
            float(evaluation[key]) >= threshold for key, threshold in criteria
        )
        consecutive = consecutive + 1 if passes else 0
        if consecutive >= required_consecutive:
            return True

    return None if incomplete else False


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


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string (e.g. ``2h 15m 30s``)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def format_duration_hms(seconds: float) -> str:
    """Format seconds as ``H:MM:SS``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def write_stage_summary(
    stage_dir,
    results_dict: dict[str, Any],
    species: str,
    algorithm: str,
) -> Path:
    """Write a text summary for a single completed stage to its directory.

    Returns the path to the written summary file.
    """
    from .config import get_library_version

    summary_path = Path(stage_dir) / "stage_summary.txt"
    mean_len = results_dict.get("mean_episode_length", 0)
    std_len = results_dict.get("std_episode_length", 0)
    sim_dt = results_dict.get("sim_dt", 0.01)
    avg_duration_s = mean_len * sim_dt
    mean_vel = results_dict.get("mean_forward_vel", 0.0)
    std_vel = results_dict.get("std_forward_vel", 0.0)
    lines = [
        f"Mesozoic Labs: Stage {results_dict['stage']} Summary",
        "=" * 50,
        "",
        f"Species:        {species.title()}",
        f"Stage:          {results_dict['stage']} ({results_dict['name']})",
        f"Description:    {results_dict['description']}",
        f"Algorithm:      {algorithm}",
        f"Version:        {get_library_version()}",
        f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timesteps:      {results_dict['timesteps']:,}",
        f"Duration:       {format_duration(results_dict['duration_seconds'])}",
        f"Final eval:     {results_dict['mean_reward']:.2f} +/- {results_dict['std_reward']:.2f}",
        f"Avg ep length:  {mean_len:.1f} +/- {std_len:.1f} steps ({avg_duration_s:.2f}s sim time)",
        f"Avg fwd vel:    {mean_vel:.2f} +/- {std_vel:.2f} m/s",
    ]
    best_r = results_dict.get("best_eval_reward", "")
    if best_r != "":
        best_s = results_dict.get("best_eval_std", "")
        best_ts = results_dict.get("best_eval_timestep", "")
        best_len = results_dict.get("best_eval_length", "")
        best_len_s = results_dict.get("best_eval_std_length", "")
        ts_label = f"  (at {best_ts:,} steps)" if isinstance(best_ts, int) else ""
        lines.append(f"Best eval:      {best_r} +/- {best_s}{ts_label}")
        if best_len != "":
            best_dur_s = best_len * sim_dt
            lines.append(f"Best ep length: {best_len} +/- {best_len_s} steps ({best_dur_s:.2f}s sim time)")
    bm_r = results_dict.get("best_model_reward", "")
    if bm_r != "":
        bm_s = results_dict.get("best_model_std_reward", "")
        bm_len = results_dict.get("best_model_length", "")
        bm_len_s = results_dict.get("best_model_std_length", "")
        bm_vel = results_dict.get("best_model_fwd_vel", "")
        bm_vel_s = results_dict.get("best_model_std_fwd_vel", "")
        bm_sr = results_dict.get("best_model_success_rate", "")
        bm_n_episodes = results_dict.get("best_model_n_episodes", 30)
        lines.append("")
        lines.append(f"Best Model Evaluation ({bm_n_episodes} episodes)")
        lines.append("-" * 40)
        lines.append(f"  Reward:       {bm_r} +/- {bm_s}")
        if bm_len != "":
            bm_dur_s = bm_len * sim_dt
            lines.append(f"  Ep length:    {bm_len} +/- {bm_len_s} steps ({bm_dur_s:.2f}s sim time)")
        if bm_vel != "":
            lines.append(f"  Fwd vel:      {bm_vel} +/- {bm_vel_s} m/s")
        if bm_sr != "":
            lines.append(f"  Success rate: {bm_sr:.0%}")
    _model_path = results_dict.get("model_path", "")
    if _model_path:
        _model_path = str(_model_path)
        if not Path(_model_path).suffix:
            _model_path += ".zip"
        lines.append(f"Best model:     {_model_path}")
        if "vecnorm_path" in results_dict:
            lines.append(f"VecNormalize:   {results_dict['vecnorm_path']}")
        lines.append("")
    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    return summary_path


def write_training_summary(
    run_dir,
    stage_results_list: list[dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    n_envs: int,
    quick_test: bool = False,
) -> Path:
    """Write a training summary text file to the run directory.

    Returns the path to the written summary file.
    """
    summary_path = Path(run_dir) / "training_summary.txt"
    total_duration = sum(r["duration_seconds"] for r in stage_results_list)

    lines = [
        "Mesozoic Labs Training Summary",
        "=" * 50,
        "",
        f"Species:        {species.title()}",
        f"Algorithm:      {algorithm}",
        f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seed:           {seed}",
        f"Quick test:     {quick_test}",
        f"Parallel envs:  {n_envs}",
        f"Run directory:  {run_dir}",
        "",
    ]

    for r in stage_results_list:
        mean_len = r.get("mean_episode_length", 0)
        std_len = r.get("std_episode_length", 0)
        sim_dt = r.get("sim_dt", 0.01)
        mean_vel = r.get("mean_forward_vel", 0.0)
        std_vel = r.get("std_forward_vel", 0.0)
        lines.extend(
            [
                f"Stage {r['stage']}: {r['name']}",
                f"  Description:    {r['description']}",
                f"  Timesteps:      {r['timesteps']:,}",
                f"  Duration:       {format_duration(r['duration_seconds'])}",
                f"  Final eval:     {r['mean_reward']:.2f} +/- {r['std_reward']:.2f}",
                f"  Avg ep length:  {mean_len:.1f} +/- {std_len:.1f} steps ({mean_len * sim_dt:.2f}s sim time)",
                f"  Avg fwd vel:    {mean_vel:.2f} +/- {std_vel:.2f} m/s",
            ]
        )
        best_r = r.get("best_eval_reward", "")
        if best_r != "":
            best_s = r.get("best_eval_std", "")
            best_ts = r.get("best_eval_timestep", "")
            ts_label = f"  (at {best_ts:,} steps)" if isinstance(best_ts, int) else ""
            lines.append(f"  Best eval:      {best_r} +/- {best_s}{ts_label}")
        model_path = r.get("model_path")
        if model_path is not None:
            model_path_str = str(model_path)
            if not Path(model_path_str).suffix:
                model_path_str += ".zip"
            lines.append(f"  Best model:     {model_path_str}")
        lines.append("")

    lines.extend(
        [
            "-" * 50,
            f"Total training time: {format_duration(total_duration)}",
        ]
    )

    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    return summary_path


def _backend_name(algorithm: str) -> str:
    """Training-backend identifier for a summary's ``backend`` field."""
    from .result_bundle import canonical_backend

    return canonical_backend(algorithm)


def _backend_version(algorithm: str) -> "str | None":
    """Installed version of the training backend, or ``None`` if undetectable."""
    package = "jax" if algorithm.upper().startswith("JAX") else "stable_baselines3"
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def _run_provenance(overrides: "Mapping[str, Any] | None" = None) -> dict[str, Any]:
    """Build a summary ``provenance`` block, recording the repository commit.

    Defaults are conservative: a freshly generated summary is ``historical`` /
    ``unverified``. Per the results contract, ``current`` or ``verified`` may
    only be claimed once the model hash, config hash, backend version, and
    evaluation-episode count are all recorded. The repository commit is filled
    automatically so a generated summary ties back to the exact code revision;
    callers with full identity can pass ``overrides`` to complete and certify it.
    """
    from .config import get_git_commit

    provenance: dict[str, Any] = {
        "model_revision_status": "historical",
        "verification_status": "unverified",
        "evaluation_episodes": None,
        "repository_commit": get_git_commit(),
        "model_hash": None,
        "config_hash": None,
    }
    if overrides:
        provenance.update(overrides)
    return provenance


def _optional_metric(value: Any, *, digits: int | None = None) -> int | float | None:
    """Normalize an optional numeric metric for JSON output."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"result metric must be numeric or null, got {value!r}")
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"result metric must be numeric or null, got {value!r}") from exc
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if digits is None and isinstance(value, int):
        return int(value)
    return round(numeric, digits) if digits is not None else numeric


def _canonical_stage_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one in-memory stage result to the public schema-v2 shape."""
    stage = int(result["stage"])
    duration = float(result.get("duration_seconds", 0.0))
    passed = parse_optional_bool(result.get("publication_gate_passed"))
    if passed is None:
        raise ValueError(f"stage {stage} is missing an explicit boolean publication_gate_passed value")

    stage_summary: dict[str, Any] = {
        "name": str(result.get("name") or f"Stage {stage}"),
        "description": str(result.get("description") or f"Curriculum stage {stage}"),
        "timesteps": int(result["timesteps"]),
        "best_eval_reward": _optional_metric(result.get("best_eval_reward"), digits=2),
        "best_eval_std": _optional_metric(result.get("best_eval_std"), digits=2),
        "best_eval_step": _optional_metric(
            result.get("best_eval_timestep", result.get("best_eval_step")),
        ),
        "final_eval_reward": _optional_metric(result.get("mean_reward"), digits=2),
        "final_eval_std": _optional_metric(result.get("std_reward"), digits=2),
        "avg_episode_length": _optional_metric(result.get("mean_episode_length"), digits=1),
        "avg_episode_length_std": _optional_metric(result.get("std_episode_length"), digits=1),
        "avg_forward_vel": _optional_metric(result.get("mean_forward_vel"), digits=3),
        "avg_forward_vel_std": _optional_metric(result.get("std_forward_vel"), digits=3),
        "mean_distance_traveled": _optional_metric(result.get("mean_distance_traveled"), digits=3),
        "mean_success_rate": _optional_metric(result.get("mean_success_rate"), digits=4),
        "training_time_seconds": round(duration, 1),
        "training_time": format_duration_hms(duration),
        "stage_passed": passed,
        "publication_gate_passed": passed,
    }
    selected_metrics = {
        "selected_model_reward": ("best_model_reward", 2),
        "selected_model_reward_std": ("best_model_std_reward", 2),
        "selected_model_episode_length": ("best_model_length", 1),
        "selected_model_episode_length_std": ("best_model_std_length", 1),
        "selected_model_forward_vel": ("best_model_fwd_vel", 3),
        "selected_model_forward_vel_std": ("best_model_std_fwd_vel", 3),
        "selected_model_distance": ("best_model_distance", 3),
        "selected_model_success_rate": ("best_model_success_rate", 4),
    }
    for output_key, (result_key, digits) in selected_metrics.items():
        if result_key in result:
            stage_summary[output_key] = _optional_metric(result.get(result_key), digits=digits)
    return stage_summary


def build_result_summary(
    stage_results_list: Sequence[Mapping[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    *,
    hardware: str = "Google Colab",
    provenance: Mapping[str, Any] | None = None,
    backend: str | None = None,
    backend_version: str | None = None,
    parallel_envs: int | None = None,
    run_id: str | None = None,
    result_date: str | None = None,
    plant_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one backend-independent result summary from normalized stage data."""
    from .result_bundle import canonical_algorithm, canonical_backend

    if not stage_results_list:
        raise ValueError("at least one stage result is required")
    stages: dict[str, dict[str, Any]] = {}
    for result in stage_results_list:
        stage_key = str(int(result["stage"]))
        if stage_key in stages:
            raise ValueError(f"duplicate stage result: {stage_key}")
        stages[stage_key] = _canonical_stage_summary(result)

    total_duration = sum(float(stage["training_time_seconds"] or 0.0) for stage in stages.values())
    total_timesteps = sum(int(result["timesteps"]) for result in stage_results_list)
    final_stage_key = max(stages, key=int)
    public_algorithm = canonical_algorithm(algorithm)
    public_backend = canonical_backend(algorithm, backend)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "bundle_status": "complete" if set(stages) == {"1", "2", "3"} else "partial",
        "species": species,
        "algorithm": public_algorithm,
        "backend": public_backend,
        "backend_version": backend_version
        if backend_version is not None
        else _backend_version("JAX_PPO" if public_backend == "jax-mjx" else public_algorithm),
        "hardware": hardware,
        "seed": seed,
        "date": result_date or datetime.now().strftime("%Y-%m-%d"),
        "stages": stages,
        "total_timesteps": total_timesteps,
        "total_training_time_seconds": round(total_duration, 1),
        "total_training_time": format_duration_hms(total_duration),
        "final_avg_reward": stages[final_stage_key]["final_eval_reward"],
        "provenance": _run_provenance(provenance),
    }
    if parallel_envs is not None:
        summary["parallel_envs"] = parallel_envs
    if run_id is not None:
        summary["run_id"] = run_id
    effective_plant_identity = plant_identity
    if effective_plant_identity is None:
        candidate = stage_results_list[-1].get("plant_identity")
        if isinstance(candidate, Mapping):
            effective_plant_identity = candidate
    if effective_plant_identity is not None:
        summary["plant_identity"] = dict(effective_plant_identity)
    return summary


def save_results_json(
    stage_results_list: list[dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    results_dir: "str | Path",
    hardware: str = "Google Colab",
    provenance: "Mapping[str, Any] | None" = None,
    *,
    backend: str | None = None,
    backend_version: str | None = None,
    parallel_envs: int | None = None,
    run_id: str | None = None,
    result_date: str | None = None,
    plant_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Save a schema-v2 ``summary.json`` to *results_dir*.

    This compatibility wrapper can write partial summaries.  The canonical
    :func:`save_result_bundle` workflow only publishes ``summary.json`` after
    all three stages exist.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    summary = build_result_summary(
        stage_results_list,
        species,
        algorithm,
        seed,
        hardware=hardware,
        provenance=provenance,
        backend=backend,
        backend_version=backend_version,
        parallel_envs=parallel_envs,
        run_id=run_id,
        result_date=result_date,
        plant_identity=plant_identity,
    )
    summary_path = results_path / "summary.json"
    summary_path.write_text(_json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path


def build_results_csv_rows(
    stage_results_list: Sequence[Mapping[str, Any]],
    stage_configs: Mapping[int, Mapping[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    *,
    backend: str | None = None,
    run_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build CSV rows from the same stage-result values used by JSON export."""
    from .result_bundle import canonical_algorithm, canonical_backend

    public_algorithm = canonical_algorithm(algorithm)
    public_backend = canonical_backend(algorithm, backend)
    algorithm_key = public_algorithm.lower()
    rows: list[dict[str, Any]] = []
    for r in stage_results_list:
        stage = int(r["stage"])
        cfg = stage_configs[stage]

        row: dict[str, Any] = {
            "schema_version": 2,
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
    stage_configs: dict[int, dict[str, Any]],
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
                "success": parsed_success,
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
        "success",
    ]
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = _csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _resolve_model_artifact(model_path: Any, *, run_dir: Path) -> Path | None:
    if model_path in {None, ""}:
        return None
    candidate = Path(str(model_path))
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidates = [candidate]
    if candidate.suffix == "":
        candidates.extend([candidate.with_suffix(".zip"), candidate.with_suffix(".pkl")])
    for path in candidates:
        if path.is_file():
            return path
    return None


def save_result_bundle(
    stage_results_list: list[dict[str, Any]],
    stage_configs: dict[int, dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    run_dir: str | Path,
    *,
    backend: str | None = None,
    backend_version: str | None = None,
    hardware: str = "Google Colab",
    parallel_envs: int | None = None,
    evaluation_episodes: int = 30,
    evaluation_seeds: Sequence[int] | None = None,
    seed_roles: Mapping[str, int] | None = None,
    plant_identity: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write one idempotent, Drive-portable result bundle.

    Partial curricula receive provenance, CSV, and a ``partial``/``failed``
    manifest.  A public schema-v2 ``summary.json`` is emitted only when all
    three stages are present and the selected checkpoint, resolved configs,
    backend version, and plant identity are available.
    """
    from .result_bundle import (
        ResultBundleError,
        _normalize_plant_identity,
        _write_json,
        aggregate_file_hash,
        canonical_algorithm,
        canonical_backend,
        compare_summary_to_csv,
        initialize_result_bundle,
        load_provenance,
        sha256_file,
        update_provenance,
        validate_evaluation_evidence,
        validate_result_bundle,
        write_artifact_manifest,
    )
    from .result_schema import validate_result_summary

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    previous_manifest = run_path / "artifact_manifest.json"

    # Preflight every expected failure before invalidating an existing
    # completion marker. This is especially important on Drive, where a
    # disconnected runtime may not get another chance to rebuild the bundle.
    try:
        ordered_stage_numbers = [int(result["stage"]) for result in stage_results_list]
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultBundleError("every stage result must contain an integer stage") from exc
    stage_numbers = set(ordered_stage_numbers)
    if len(stage_numbers) != len(stage_results_list):
        raise ResultBundleError("stage_results_list contains duplicate stages")
    if not stage_numbers or not stage_numbers <= {1, 2, 3}:
        raise ResultBundleError("stage_results_list must contain stages 1, 2, or 3")
    expected_prefix = set(range(1, max(stage_numbers) + 1))
    if stage_numbers != expected_prefix:
        raise ResultBundleError(
            f"stage_results_list must be a contiguous curriculum prefix; found {sorted(stage_numbers)}"
        )
    gate_values: dict[int, bool] = {}
    for result in stage_results_list:
        stage = int(result["stage"])
        gate_value = parse_optional_bool(result.get("publication_gate_passed"))
        if gate_value is None:
            raise ResultBundleError(f"stage {stage} is missing an explicit boolean publication_gate_passed value")
        gate_values[stage] = gate_value
    has_all_stages = stage_numbers == {1, 2, 3}
    promotion_ready = has_all_stages and all(gate_values.values())
    status = "complete" if promotion_ready else ("failed" if not all(gate_values.values()) else "partial")

    summary_path = run_path / "summary.json"
    if not promotion_ready and summary_path.exists():
        raise ResultBundleError("non-publishable bundle contains a stale summary.json")
    if not evaluation_seeds:
        raise ResultBundleError("result bundle requires at least one recorded publication evaluation seed")
    if not isinstance(parallel_envs, int) or isinstance(parallel_envs, bool) or parallel_envs <= 0:
        raise ResultBundleError("result bundle requires a positive parallel_envs value")

    effective_plant = plant_identity
    if effective_plant is None and stage_results_list:
        final_stage_result = max(stage_results_list, key=lambda item: int(item["stage"]))
        candidate = final_stage_result.get("plant_identity")
        if isinstance(candidate, Mapping):
            effective_plant = candidate
    normalized_plant = _normalize_plant_identity(effective_plant, species=species)
    if normalized_plant is None:
        raise ResultBundleError("result bundle is missing plant identity")
    for result in stage_results_list:
        stage = int(result["stage"])
        stage_plant_value = result.get("plant_identity")
        if not isinstance(stage_plant_value, Mapping):
            raise ResultBundleError(f"stage {stage} is missing plant identity")
        stage_plant = _normalize_plant_identity(stage_plant_value, species=species)
        if stage_plant != normalized_plant:
            raise ResultBundleError(f"stage {stage} plant identity does not match the run identity")

    public_algorithm = canonical_algorithm(algorithm)
    public_backend = canonical_backend(algorithm, backend)
    detected_backend_version = backend_version or _backend_version(
        "JAX_PPO" if public_backend == "jax-mjx" else algorithm
    )
    if promotion_ready and not detected_backend_version:
        raise ResultBundleError("complete bundle requires a recorded backend version")

    config_paths = [run_path / f"stage{stage}" / "stage_config.json" for stage in sorted(stage_numbers)]
    missing_configs = [path for path in config_paths if not path.is_file()]
    if missing_configs:
        raise ResultBundleError(f"result bundle is missing resolved stage configs: {missing_configs}")
    resolved_stage_configs: dict[int, dict[str, Any]] = {}
    for stage, config_path in zip(sorted(stage_numbers), config_paths, strict=True):
        try:
            saved_config = _json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read resolved stage config {config_path}: {exc}") from exc
        if not isinstance(saved_config, Mapping):
            raise ResultBundleError(f"resolved stage config must contain an object: {config_path}")
        if saved_config.get("stage") is not None and int(saved_config["stage"]) != stage:
            raise ResultBundleError(f"resolved stage config is mislabeled for stage {stage}: {config_path}")
        if saved_config.get("species") not in {None, "", species}:
            raise ResultBundleError(f"resolved stage config species mismatch: {config_path}")
        if saved_config.get("algorithm"):
            try:
                saved_algorithm = canonical_algorithm(str(saved_config["algorithm"]))
            except ResultBundleError as exc:
                raise ResultBundleError(f"invalid algorithm in resolved stage config {config_path}") from exc
            if saved_algorithm != public_algorithm:
                raise ResultBundleError(f"resolved stage config algorithm mismatch: {config_path}")
        saved_plant_value = saved_config.get("plant_identity")
        if promotion_ready and saved_plant_value is None:
            raise ResultBundleError(f"complete bundle stage config is missing plant identity: {config_path}")
        if saved_plant_value is not None:
            if not isinstance(saved_plant_value, Mapping):
                raise ResultBundleError(f"invalid plant identity in resolved stage config: {config_path}")
            saved_plant = _normalize_plant_identity(saved_plant_value, species=species)
            if saved_plant != normalized_plant:
                raise ResultBundleError(f"resolved stage config plant identity mismatch: {config_path}")
        reward_weights = saved_config.get("reward_weights", saved_config.get("env_kwargs", {}))
        hyperparameters = saved_config.get(
            "hyperparameters",
            saved_config.get(
                "jax_kwargs"
                if public_backend == "jax-mjx"
                else ("sac_kwargs" if public_algorithm == "SAC" else "ppo_kwargs"),
                {},
            ),
        )
        curriculum = saved_config.get(
            "curriculum",
            saved_config.get("curriculum_kwargs", {}),
        )
        if not all(isinstance(value, Mapping) for value in (reward_weights, hyperparameters, curriculum)):
            raise ResultBundleError(f"resolved stage config sections must be objects: {config_path}")
        algorithm_config_key = (
            "jax_kwargs"
            if public_backend == "jax-mjx"
            else ("sac_kwargs" if public_algorithm == "SAC" else "ppo_kwargs")
        )
        resolved_stage_configs[stage] = {
            "name": str(saved_config.get("name") or f"Stage {stage}"),
            "description": str(saved_config.get("description") or f"Curriculum stage {stage}"),
            "env_kwargs": dict(reward_weights),
            algorithm_config_key: dict(hyperparameters),
            "curriculum_kwargs": dict(curriculum),
        }
    config_hash = aggregate_file_hash(config_paths, root=run_path)

    selected_checkpoints: dict[str, dict[str, Any]] = {}
    for result in stage_results_list:
        stage = int(result["stage"])
        model_artifact = _resolve_model_artifact(result.get("model_path"), run_dir=run_path)
        if promotion_ready and model_artifact is None:
            raise ResultBundleError(f"complete bundle is missing its selected Stage {stage} checkpoint")
        if model_artifact is None:
            continue
        try:
            selected_path = model_artifact.resolve().relative_to(run_path.resolve()).as_posix()
        except ValueError as exc:
            raise ResultBundleError(f"selected checkpoint lies outside run directory: {model_artifact}") from exc
        normalization_path: str | None = None
        normalization_hash: str | None = None
        if public_backend == "stable-baselines3":
            normalization_artifact = _resolve_model_artifact(
                result.get("vecnorm_path"),
                run_dir=run_path,
            )
            if promotion_ready and normalization_artifact is None:
                raise ResultBundleError(
                    f"complete SB3 bundle is missing selected Stage {stage} VecNormalize statistics"
                )
            if normalization_artifact is not None:
                try:
                    normalization_path = normalization_artifact.resolve().relative_to(run_path.resolve()).as_posix()
                except ValueError as exc:
                    raise ResultBundleError(
                        f"selected VecNormalize artifact lies outside run directory: {normalization_artifact}"
                    ) from exc
                normalization_hash = sha256_file(normalization_artifact)
        selected_checkpoints[str(stage)] = {
            "model_path": selected_path,
            "model_hash": sha256_file(model_artifact),
            "normalization_path": normalization_path,
            "normalization_hash": normalization_hash,
        }

    stage3_checkpoint = selected_checkpoints.get("3", {})
    selected_model_path = stage3_checkpoint.get("model_path")
    model_hash = stage3_checkpoint.get("model_hash")

    if promotion_ready:
        for stage in (1, 2, 3):
            for checkpoint_label in ("selected", "final"):
                evidence_path = run_path / f"stage{stage}" / f"evaluation_{checkpoint_label}.csv"
                if not evidence_path.is_file() or evidence_path.stat().st_size == 0:
                    raise ResultBundleError(
                        f"complete bundle is missing {checkpoint_label} evaluation evidence: {evidence_path}"
                    )

    previous_manifest_status: str | None = None
    if previous_manifest.exists():
        try:
            previous_manifest_value = _json.loads(previous_manifest.read_text(encoding="utf-8"))
            if isinstance(previous_manifest_value, Mapping):
                previous_manifest_status = previous_manifest_value.get("status")
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read existing artifact manifest: {exc}") from exc
        if previous_manifest_status == "complete":
            validate_result_bundle(run_path, require_complete=True)

    provenance_path = initialize_result_bundle(
        run_path,
        species=species,
        algorithm=algorithm,
        backend=backend,
        seed=seed,
        evaluation_seeds=evaluation_seeds,
        evaluation_episodes=evaluation_episodes,
        seed_roles=seed_roles,
        parallel_envs=parallel_envs,
        hardware=hardware,
        plant_identity=normalized_plant,
        run_id=run_id,
        repository_root=repository_root,
    )
    captured = load_provenance(run_path)
    finalization = {
        "model_hash": model_hash,
        "config_hash": config_hash,
        "backend_version": detected_backend_version,
        "selected_model_path": selected_model_path,
        "selected_checkpoints": selected_checkpoints,
    }
    finalized_provenance = {**captured, **finalization}
    result_date = str(captured.get("captured_at", "")).split("T", maxsplit=1)[0]

    prospective_summary: dict[str, Any] | None = None
    if promotion_ready:
        prospective_summary = build_result_summary(
            stage_results_list,
            species,
            algorithm,
            seed,
            hardware=str(captured["hardware"]),
            provenance=finalized_provenance,
            backend=public_backend,
            backend_version=detected_backend_version,
            parallel_envs=int(captured["parallel_envs"]),
            run_id=str(captured["run_id"]),
            result_date=result_date,
            plant_identity=normalized_plant,
        )
        validate_result_summary(
            prospective_summary,
            expected_species=species,
            require_complete=True,
            require_canonical_provenance=True,
            result_path=str(summary_path),
        )
        validate_evaluation_evidence(run_path, prospective_summary, finalized_provenance)
        if previous_manifest_status == "complete":
            existing_summary = _json.loads(summary_path.read_text(encoding="utf-8"))
            if existing_summary != prospective_summary:
                raise ResultBundleError("completed result bundle is immutable; use a new run_id for different results")
            return {
                "provenance": provenance_path,
                "collected_results_csv": run_path / "collected_results.csv",
                "summary": summary_path,
                "artifact_manifest": previous_manifest,
            }

    # Exercise CSV construction before removing the previous completion marker.
    build_results_csv_rows(
        stage_results_list,
        resolved_stage_configs,
        species,
        algorithm,
        seed,
        backend=public_backend,
        run_id=str(captured["run_id"]),
        provenance=finalized_provenance,
    )

    # All semantic validation is complete. Remove the old marker immediately
    # before changing derived artifacts, then write the new marker last.
    if previous_manifest.exists():
        previous_manifest.unlink()
    update_provenance(run_path, **finalization)
    captured = load_provenance(run_path)
    _write_json(run_path / "plant_identity.json", normalized_plant)

    csv_path = save_results_csv(
        stage_results_list,
        resolved_stage_configs,
        species,
        algorithm,
        seed,
        run_path,
        backend=public_backend,
        run_id=str(captured["run_id"]),
        provenance=captured,
    )
    paths: dict[str, Path] = {
        "provenance": provenance_path,
        "collected_results_csv": csv_path,
    }

    if promotion_ready:
        assert prospective_summary is not None
        _write_json(summary_path, prospective_summary)
        contradictions = compare_summary_to_csv(prospective_summary, csv_path)
        if contradictions:
            raise ResultBundleError("summary/CSV contradictions: " + "; ".join(contradictions))
        paths["summary"] = summary_path

    manifest_path = write_artifact_manifest(run_path, status=status)
    paths["artifact_manifest"] = manifest_path
    validate_result_bundle(run_path, require_complete=promotion_ready)
    return paths


def build_stage_results_from_eval_data(
    stage_dir: "str | Path",
    stage: int,
    stage_config: dict[str, Any],
    timesteps: int,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build a ``stage_results`` dict from on-disk evaluation artifacts.

    Reads ``evaluations.npz`` (written by SB3's ``EvalCallback``) and
    ``metrics.json`` to reconstruct the same results dict that the
    training notebook's ``train_stage`` produces.  This allows sweep
    trials and any other post-hoc consumers to build a consistent
    results dict without re-running evaluation.

    If *duration_seconds* is 0 and a ``metrics.json`` exists, the duration
    is read from ``training_duration_seconds`` in that file.

    Fields that require a live policy evaluation (``mean_forward_vel``,
    ``mean_success_rate``, ``best_model_*``) default to ``0.0`` / ``""``
    and can be updated by the caller after running ``eval_policy``.
    """
    import numpy as _np

    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"

    # ── Parse evaluations.npz ───────────────────────────────────────────
    eval_npz = stage_dir / "evaluations.npz"
    mean_reward = 0.0
    std_reward = 0.0
    mean_length = 0.0
    std_length = 0.0
    best_eval_reward: float | str = ""
    best_eval_std: float | str = ""
    best_eval_length: float | str = ""
    best_eval_std_length: float | str = ""
    best_eval_timestep: int | str = ""

    if eval_npz.exists():
        eval_data = _np.load(str(eval_npz))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        eval_timesteps = eval_data["timesteps"]

        mean_per_eval = eval_rewards.mean(axis=1)
        best_idx = int(mean_per_eval.argmax())

        best_eval_reward = round(float(mean_per_eval[best_idx]), 2)
        best_eval_std = round(float(eval_rewards[best_idx].std()), 2)
        best_eval_length = round(float(eval_lengths[best_idx].mean()), 1)
        best_eval_std_length = round(float(eval_lengths[best_idx].std()), 1)
        best_eval_timestep = int(eval_timesteps[best_idx])

        # Use last eval as "final" metrics
        mean_reward = float(mean_per_eval[-1])
        std_reward = float(eval_rewards[-1].std())
        mean_length = float(eval_lengths[-1].mean())
        std_length = float(eval_lengths[-1].std())

    # ── Duration and provenance from sidecars ───────────────────────────
    metrics: dict[str, Any] = {}
    metrics_path = stage_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _json.loads(metrics_path.read_text())
        if duration_seconds == 0.0:
            duration_seconds = metrics.get("training_duration_seconds", 0.0)
    plant_identity = metrics.get("plant_identity")
    if not isinstance(plant_identity, Mapping):
        saved_config_path = stage_dir / "stage_config.json"
        if saved_config_path.exists():
            saved_config = _json.loads(saved_config_path.read_text())
            plant_identity = saved_config.get("plant_identity")

    best_model_path = model_dir / "best_model"
    vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
    sim_dt = stage_config.get("env_kwargs", {}).get("sim_dt", 0.01)

    result = {
        "stage": stage,
        "name": stage_config["name"],
        "description": stage_config["description"],
        "timesteps": timesteps,
        "duration_seconds": duration_seconds,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        "mean_forward_vel": 0.0,
        "std_forward_vel": 0.0,
        "mean_success_rate": 0.0,
        "best_eval_reward": best_eval_reward,
        "best_eval_std": best_eval_std,
        "best_eval_length": best_eval_length,
        "best_eval_std_length": best_eval_std_length,
        "best_eval_timestep": best_eval_timestep,
        "sim_dt": sim_dt,
        "model_path": str(best_model_path),
        "vecnorm_path": vecnorm_path,
    }
    if isinstance(plant_identity, Mapping):
        result["plant_identity"] = dict(plant_identity)
    return result


def generate_stage_artifacts(
    species_cfg,
    stage_config: dict[str, Any],
    stage: int,
    algorithm: str,
    stage_dir: "str | Path",
    seed: int,
    stage_results: dict[str, Any] | None = None,
    timesteps: int = 0,
    record_videos: bool = True,
    generate_graphs: bool = True,
    allow_legacy_plant: bool = False,
) -> dict[str, Any]:
    """Write stage summary, record replay videos, and generate training graphs.

    This is the single shared entry-point for generating post-training
    artifacts.  Both the training notebook and the sweep trial worker
    call this function so that the artifacts are always consistent.

    When *stage_results* is ``None``, a results dict is built from on-disk
    eval data via :func:`build_stage_results_from_eval_data`.  Callers
    that already have richer metrics (e.g. the notebook, which runs a
    full 30-episode eval) should pass their own *stage_results*.

    When *generate_graphs* is ``True`` (the default), training curves and
    diagnostic graphs are saved to the stage directory.  Requires
    ``matplotlib``.

    Returns the (possibly enriched) *stage_results* dict.
    """
    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"
    species = species_cfg.species

    if stage_results is None:
        stage_results = build_stage_results_from_eval_data(
            stage_dir,
            stage,
            stage_config,
            timesteps=timesteps,
        )

    write_stage_summary(stage_dir, stage_results, species, algorithm)
    logger.info("Stage summary written to: %s", stage_dir / "stage_summary.txt")

    # ── Generate training graphs ────────────────────────────────────────
    if generate_graphs:
        try:
            from environments.shared.visualization import (
                plot_diagnostics_graphs,
                plot_foot_contacts,
                plot_stance_diagnostics,
                plot_training_curves,
            )

            stage_dirs = [(stage, stage_dir)]
            stage_configs = {stage: stage_config}

            plot_training_curves(
                stage_dirs,
                stage_configs,
                species,
                algorithm,
                save_path=stage_dir / "training_curves.png",
                show=False,
            )
            plot_diagnostics_graphs(
                stage_dirs,
                stage_configs,
                species,
                algorithm,
                save_dir=stage_dir,
                show=False,
            )
            plot_foot_contacts(
                stage_dirs,
                stage_configs,
                species,
                algorithm,
                save_path=stage_dir / "foot_contacts.png",
                show=False,
            )
            plot_stance_diagnostics(
                stage_dirs,
                stage_configs,
                species,
                algorithm,
                save_path=stage_dir / "stance_diagnostics.png",
                show=False,
            )
        except ImportError:
            logger.warning("Skipping graph generation (matplotlib not installed).")
        except Exception:
            logger.warning("Graph generation failed.", exc_info=True)

    if not record_videos:
        return stage_results

    # ── Record replay videos for best and final models ──────────────────
    from .plant_contract import PlantCompatibilityError, current_plant_identity, validate_model_plant

    try:
        from environments.shared.evaluation import TREX_STAGE1_CAMERA_VIEWS, record_stage_video
        from environments.shared.train_base import _ensure_sb3

        sb3 = _ensure_sb3()
        env_kwargs = stage_config["env_kwargs"].copy()
        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
        plant_identity = current_plant_identity(species)

        best_model_path = model_dir / "best_model"
        vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
        final_path = model_dir / f"stage{stage}_final"
        final_vecnorm_path = str(final_path) + "_vecnorm.pkl"
        replay_diagnostics = species.lower() == "trex" and stage == 1
        replay_camera_views = TREX_STAGE1_CAMERA_VIEWS if replay_diagnostics else None

        if (model_dir / "best_model.zip").exists():
            best_model = alg_cls.load(str(best_model_path))
            validate_model_plant(
                best_model,
                plant_identity,
                artifact=str(best_model_path) + ".zip",
                allow_legacy=allow_legacy_plant,
            )
            record_stage_video(
                best_model,
                env_class=species_cfg.env_class,
                env_kwargs=env_kwargs,
                stage=stage,
                stage_dir=stage_dir,
                species=species,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=vecnorm_path,
                label="best",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )

        if (Path(str(final_path) + ".zip")).exists():
            final_model = alg_cls.load(str(final_path))
            validate_model_plant(
                final_model,
                plant_identity,
                artifact=str(final_path) + ".zip",
                allow_legacy=allow_legacy_plant,
            )
            record_stage_video(
                final_model,
                env_class=species_cfg.env_class,
                env_kwargs=env_kwargs,
                stage=stage,
                stage_dir=stage_dir,
                species=species,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=final_vecnorm_path,
                label="final",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )
    except PlantCompatibilityError:
        raise
    except Exception:
        logger.warning("Video recording failed.", exc_info=True)

    return stage_results


def save_jax_stage_artifacts(
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_results: dict[str, Any],
    stage_dir: "str | Path",
    run_dir: "str | Path",
    eval_results: Any,
    params: Any,
    obs_rms: Any,
    *,
    final_eval_results: Any | None = None,
    seed: int = 42,
    num_envs: int = 2048,
    reward_cfg: dict[str, float] | None = None,
    best_params: Any | None = None,
    best_reward: float = 0.0,
    best_update: int = 0,
    evaluation_seed: int = 42,
    backend_version: str | None = None,
    plant_identity: PlantIdentity | None = None,
) -> dict[str, Path]:
    """Save all post-training artifacts for a JAX/MJX training stage.

    Orchestrates the same artifact generation that the SB3 path performs
    via :func:`generate_stage_artifacts`, but using JAX-native checkpoint
    formats and without requiring an SB3 ``SpeciesConfig``.

    Artifacts saved:

    * ``stage_summary.txt`` — human-readable stage summary
    * ``stage_config.json`` — frozen config snapshot
    * ``collected_results.csv`` — one row per stage (append-safe)
    * ``diagnostics.npz`` — per-step evaluation diagnostics
    * ``best_model.pkl`` — best checkpoint (params + obs stats)
    * ``stage{N}_final.pkl`` — final checkpoint
    * ``training_summary.txt`` — run-level summary

    Args:
        species: Species identifier (e.g. ``"velociraptor"``).
        stage: Curriculum stage number (1, 2, or 3).
        stage_config: Config dict from :func:`config.load_stage_config`.
        stage_results: Results dict with keys like ``mean_reward``,
            ``timesteps``, ``duration_seconds``, ``model_path``, etc.
        stage_dir: Directory for this stage's output files.
        run_dir: Parent run directory (for CSV and training summary).
        eval_results: :class:`jax_eval.EvalResults` instance with
            selected-checkpoint episode and per-step diagnostic data.
        final_eval_results: Episode evidence for the terminal parameters.
            Defaults to *eval_results* only for backward-compatible callers
            whose selected and terminal parameters are identical.
        params: Final JAX network parameters.
        obs_rms: Observation normalisation statistics.
        seed: Random seed used for training.
        num_envs: Number of parallel environments.
        reward_cfg: Reward weight dict (included in config snapshot).
        best_params: Best-performing parameters (falls back to *params*).
        best_reward: Best evaluation reward achieved during training.
        best_update: Update number at which *best_params* was recorded.
        evaluation_seed: Seed used for the fixed publication evaluation.
        backend_version: Optional explicit JAX version for portable tests or
            environments where package metadata cannot be detected.
        plant_identity: Optional precomputed plant identity.  The current
            identity is resolved and verified when omitted.

    Returns:
        Dict mapping artifact name to its file path.
    """
    import numpy as _np

    from .config import save_stage_config
    from .jax_checkpoint import save_checkpoint
    from .plant_contract import current_plant_identity, write_plant_identity
    from .result_bundle import (
        ResultBundleError,
        initialize_result_bundle,
        load_provenance,
        validate_result_bundle,
    )

    stage_dir = Path(stage_dir)
    run_dir = Path(run_dir)
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    if stage_dir.parent.resolve() != run_dir.resolve():
        raise ValueError("stage_dir must be run_dir/stage<N> for a portable result bundle")
    if stage_dir.name != f"stage{stage}":
        raise ValueError(f"stage_dir must be named stage{stage}")
    if plant_identity is None:
        plant_identity = current_plant_identity(species)

    manifest_path = run_dir / "artifact_manifest.json"
    if manifest_path.exists():
        try:
            existing_manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read existing artifact manifest: {exc}") from exc
        if isinstance(existing_manifest, Mapping) and existing_manifest.get("status") == "complete":
            validate_result_bundle(run_dir, require_complete=True)
            raise ResultBundleError("completed result bundle is immutable; use a new run_id to rewrite a stage")

    existing_stages: set[int] = set()
    for existing_result_path in sorted(run_dir.glob("stage*/stage_result.json")):
        try:
            saved_result = _json.loads(existing_result_path.read_text(encoding="utf-8"))
            saved_stage = int(saved_result["stage"])
        except (OSError, _json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ResultBundleError(f"cannot read prior JAX stage record {existing_result_path}") from exc
        if existing_result_path.parent.name != f"stage{saved_stage}" or saved_stage not in {1, 2, 3}:
            raise ResultBundleError(f"mislabeled prior JAX stage record: {existing_result_path}")
        if saved_stage in existing_stages:
            raise ResultBundleError(f"duplicate prior JAX stage record for stage {saved_stage}")
        existing_stages.add(saved_stage)
    combined_stages = existing_stages | {stage}
    expected_stages = set(range(1, max(combined_stages) + 1))
    if combined_stages != expected_stages or max(combined_stages) > stage:
        raise ResultBundleError(f"JAX stages must be saved as a contiguous prefix; found {sorted(combined_stages)}")

    episode_fields = ("rewards", "lengths", "forward_vels", "distances", "successes")
    has_episode_evidence = all(hasattr(eval_results, field) for field in episode_fields)
    terminal_eval_results = final_eval_results if final_eval_results is not None else eval_results
    has_final_evidence = all(hasattr(terminal_eval_results, field) for field in episode_fields)
    evaluation_episode_count = len(getattr(eval_results, "rewards", [])) if has_episode_evidence else 30
    if not has_episode_evidence:
        raise ResultBundleError("JAX selected evaluation evidence is missing episode arrays")
    if evaluation_episode_count <= 0:
        raise ResultBundleError("JAX selected evaluation evidence must contain at least one episode")
    if not has_final_evidence:
        raise ResultBundleError("JAX terminal evaluation evidence is missing episode arrays")
    for label, evidence in (("selected", eval_results), ("final", terminal_eval_results)):
        evidence_lengths = {len(getattr(evidence, field)) for field in episode_fields}
        if evidence_lengths != {evaluation_episode_count}:
            raise ResultBundleError(f"JAX {label} evaluation evidence sequences must have equal lengths")
    selected_rewards = _np.asarray(eval_results.rewards, dtype=float)
    selected_lengths = _np.asarray(eval_results.lengths, dtype=float)
    selected_forward_velocities = _np.asarray(eval_results.forward_vels, dtype=float)
    selected_distances = _np.asarray(eval_results.distances, dtype=float)
    selected_successes = _np.asarray(eval_results.successes, dtype=float)
    final_rewards = _np.asarray(terminal_eval_results.rewards, dtype=float)
    final_lengths = _np.asarray(terminal_eval_results.lengths, dtype=float)
    final_forward_velocities = _np.asarray(terminal_eval_results.forward_vels, dtype=float)
    final_distances = _np.asarray(terminal_eval_results.distances, dtype=float)
    final_successes = _np.asarray(terminal_eval_results.successes, dtype=float)
    stage_results.update(
        {
            "mean_reward": round(float(final_rewards.mean()), 2),
            "std_reward": round(float(final_rewards.std()), 2),
            "mean_episode_length": round(float(final_lengths.mean()), 1),
            "std_episode_length": round(float(final_lengths.std()), 1),
            "mean_forward_vel": round(float(final_forward_velocities.mean()), 3),
            "std_forward_vel": round(float(final_forward_velocities.std()), 3),
            "mean_distance_traveled": round(float(final_distances.mean()), 3),
            "mean_success_rate": round(float(final_successes.mean()), 4),
            "best_model_reward": round(float(selected_rewards.mean()), 2),
            "best_model_std_reward": round(float(selected_rewards.std()), 2),
            "best_model_length": round(float(selected_lengths.mean()), 1),
            "best_model_std_length": round(float(selected_lengths.std()), 1),
            "best_model_fwd_vel": round(float(selected_forward_velocities.mean()), 3),
            "best_model_std_fwd_vel": round(float(selected_forward_velocities.std()), 3),
            "best_model_distance": round(float(selected_distances.mean()), 3),
            "best_model_success_rate": round(float(selected_successes.mean()), 4),
        }
    )

    try:
        captured_provenance = load_provenance(run_dir)
    except ResultBundleError:
        captured_provenance = {}
    seed_roles = captured_provenance.get("seed_roles") or {
        "training": seed,
        "publication_evaluation": evaluation_seed,
    }
    initialize_result_bundle(
        run_dir,
        species=species,
        algorithm="JAX_PPO",
        backend="jax-mjx",
        seed=seed,
        evaluation_seeds=[evaluation_seed],
        evaluation_episodes=evaluation_episode_count,
        seed_roles=seed_roles,
        parallel_envs=num_envs,
        hardware=str(captured_provenance.get("hardware") or "Google Colab"),
        plant_identity=plant_identity.to_dict(),
        run_id=captured_provenance.get("run_id"),
    )
    captured_provenance = load_provenance(run_dir)
    if manifest_path.exists():
        manifest_path.unlink()

    model_dir = stage_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    write_plant_identity(run_dir / "plant_identity.json", plant_identity)
    stage_results["plant_identity"] = plant_identity.to_dict()

    # 1. Stage summary text file
    summary_path = write_stage_summary(stage_dir, stage_results, species, "JAX/MJX PPO")
    paths["stage_summary"] = summary_path
    logger.info("Stage summary saved: %s", summary_path)

    # 2. Stage config snapshot
    config_path = save_stage_config(
        stage_dir,
        stage=stage,
        stage_config=stage_config,
        algorithm="jax_ppo",
        species=species,
        extra={
            "seed": seed,
            "num_envs": num_envs,
            "reward_cfg": reward_cfg or {},
        },
        plant_identity=plant_identity,
    )
    paths["stage_config"] = config_path
    logger.info("Stage config saved: %s", config_path)

    # 3. Per-episode evidence for both the selected and terminal parameters.
    evaluation_path = save_evaluation_episodes(
        stage_dir,
        rewards=eval_results.rewards,
        lengths=eval_results.lengths,
        forward_velocities=eval_results.forward_vels,
        distances=eval_results.distances,
        successes=eval_results.successes,
        evaluation_seed=evaluation_seed,
        checkpoint_label="selected",
    )
    paths["evaluation_episodes"] = evaluation_path
    final_evaluation_path = save_evaluation_episodes(
        stage_dir,
        rewards=terminal_eval_results.rewards,
        lengths=terminal_eval_results.lengths,
        forward_velocities=terminal_eval_results.forward_vels,
        distances=terminal_eval_results.distances,
        successes=terminal_eval_results.successes,
        evaluation_seed=evaluation_seed,
        checkpoint_label="final",
    )
    paths["final_evaluation_episodes"] = final_evaluation_path

    # 4. Diagnostics NPZ from eval results
    diag_data: dict[str, Any] = {
        # JAX diagnostics are a contiguous evaluation trace rather than
        # rollout snapshots.  Expose its step axis under the same key the
        # shared stance dashboard consumes for SB3 artifacts.
        "timesteps": _np.arange(len(eval_results.diag_tilt), dtype=int),
        "tilt_angle": _np.array(eval_results.diag_tilt),
        "forward_vel": _np.array(eval_results.diag_fwd_vel),
        "pelvis_height": _np.array(eval_results.diag_pelvis_h),
        "energy": _np.array(eval_results.diag_energy),
    }
    if eval_results.diag_l_foot:
        diag_data["l_foot_contact"] = _np.array(eval_results.diag_l_foot)
        diag_data["r_foot_contact"] = _np.array(eval_results.diag_r_foot)
    for comp_name, comp_vals in eval_results.diag_reward_components.items():
        diag_data[f"reward_{comp_name}"] = _np.array(comp_vals)
    for diagnostic_name, diagnostic_vals in getattr(eval_results, "diag_reward_diagnostics", {}).items():
        diag_data[diagnostic_name] = _np.array(diagnostic_vals)
    if species.lower() == "trex" and stage == 1 and eval_results.diag_l_foot and eval_results.diag_r_foot:
        from .stance_diagnostics import derive_stance_info

        derived_rows = [
            derive_stance_info(
                {
                    "r_foot_contact": right_force,
                    "l_foot_contact": left_force,
                }
            )
            for right_force, left_force in zip(
                eval_results.diag_r_foot,
                eval_results.diag_l_foot,
                strict=True,
            )
        ]
        if derived_rows:
            for diagnostic_name in derived_rows[0]:
                diag_data[diagnostic_name] = _np.array(
                    [row[diagnostic_name] for row in derived_rows],
                    dtype=float,
                )

    diag_path = stage_dir / "diagnostics.npz"
    _np.savez(diag_path, **diag_data)
    paths["diagnostics"] = diag_path
    logger.info("Diagnostics saved: %s", diag_path)

    # 5. Model checkpoints (best + final)
    effective_best = best_params if best_params is not None else params
    best_model_path = model_dir / "best_model.pkl"
    save_checkpoint(
        best_model_path,
        effective_best,
        obs_rms=obs_rms,
        extra={"best_reward": best_reward, "best_update": best_update},
        plant_identity=plant_identity,
    )
    paths["best_model"] = best_model_path

    final_model_path = model_dir / f"stage{stage}_final.pkl"
    save_checkpoint(final_model_path, params, obs_rms=obs_rms, plant_identity=plant_identity)
    paths["final_model"] = final_model_path
    logger.info("Models saved: %s, %s", best_model_path, final_model_path)

    # 6. Persist one idempotent stage record for cross-session curricula.
    stage_results["model_path"] = best_model_path.resolve().relative_to(run_dir.resolve()).as_posix()
    persisted_keys = (
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
        "best_model_reward",
        "best_model_std_reward",
        "best_model_length",
        "best_model_std_length",
        "best_model_fwd_vel",
        "best_model_std_fwd_vel",
        "best_model_distance",
        "best_model_success_rate",
        "model_path",
        "plant_identity",
    )
    persisted_result = {key: stage_results[key] for key in persisted_keys if key in stage_results}
    stage_result_path = stage_dir / "stage_result.json"
    stage_result_path.write_text(_json.dumps(persisted_result, indent=2, sort_keys=True) + "\n")
    paths["stage_result"] = stage_result_path

    accumulated_results: list[dict[str, Any]] = []
    accumulated_configs: dict[int, dict[str, Any]] = {}
    for existing_result_path in sorted(run_dir.glob("stage*/stage_result.json")):
        saved_result = _json.loads(existing_result_path.read_text())
        saved_stage = int(saved_result["stage"])
        accumulated_results.append(saved_result)
        saved_config_path = existing_result_path.parent / "stage_config.json"
        saved_config = _json.loads(saved_config_path.read_text())
        accumulated_configs[saved_stage] = {
            "name": saved_config.get("name", f"Stage {saved_stage}"),
            "description": saved_config.get("description", f"Curriculum stage {saved_stage}"),
            "env_kwargs": saved_config.get("reward_weights", {}),
            "jax_kwargs": saved_config.get("hyperparameters", {}),
            "curriculum_kwargs": saved_config.get("curriculum", {}),
        }
    accumulated_results.sort(key=lambda result: int(result["stage"]))

    # 7. Training summary (run-level, regenerated from every saved stage)
    training_summary_path = write_training_summary(
        run_dir,
        accumulated_results,
        species,
        algorithm="JAX/MJX PPO",
        seed=seed,
        n_envs=num_envs,
    )
    paths["training_summary"] = training_summary_path
    logger.info("Training summary saved: %s", training_summary_path)

    # 8. Canonical CSV/provenance/manifest; summary.json appears at Stage 3.
    bundle_paths = save_result_bundle(
        accumulated_results,
        accumulated_configs,
        species,
        "JAX_PPO",
        seed,
        run_dir,
        backend="jax-mjx",
        backend_version=backend_version,
        parallel_envs=num_envs,
        hardware=str(captured_provenance.get("hardware") or "Google Colab"),
        evaluation_episodes=evaluation_episode_count,
        evaluation_seeds=[evaluation_seed],
        seed_roles=captured_provenance.get("seed_roles"),
        plant_identity=plant_identity.to_dict(),
        run_id=captured_provenance.get("run_id"),
    )
    paths.update(bundle_paths)
    logger.info("Canonical JAX result bundle saved: %s", run_dir)

    return paths
