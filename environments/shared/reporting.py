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
from datetime import datetime
from pathlib import Path
from typing import Any

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
    lines.extend(
        [
            f"Best model:     {results_dict['model_path']}.zip",
            f"VecNormalize:   {results_dict['vecnorm_path']}",
            "",
        ]
    )
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
        lines.extend(
            [
                f"  Best model:     {r['model_path']}.zip",
                "",
            ]
        )

    lines.extend(
        [
            "-" * 50,
            f"Total training time: {format_duration(total_duration)}",
        ]
    )

    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    return summary_path


def save_results_json(
    stage_results_list: list[dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    results_dir: "str | Path",
    hardware: str = "Google Colab T4 GPU",
) -> Path:
    """Save a ``summary.json`` to *results_dir*.

    Creates a machine-readable record of the training run that can be
    used to auto-generate the README results table and website content.

    Args:
        hardware: Description of the training hardware (e.g.
            ``"Vertex AI n1-standard-8 + T4"``).  Defaults to
            ``"Google Colab T4 GPU"`` for notebook usage.

    Returns the path to the written JSON file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total_duration = sum(r["duration_seconds"] for r in stage_results_list)
    total_timesteps = sum(r["timesteps"] for r in stage_results_list)
    final_result = stage_results_list[-1]

    stages = {}
    for r in stage_results_list:
        stage_data = {
            "name": r["name"],
            "timesteps": r["timesteps"],
            "avg_reward": round(r["mean_reward"], 2),
            "std_reward": round(r["std_reward"], 2),
            "training_time_seconds": round(r["duration_seconds"], 1),
            "training_time": format_duration_hms(r["duration_seconds"]),
        }
        if "mean_forward_vel" in r:
            stage_data["avg_forward_vel"] = round(r["mean_forward_vel"], 2)
        stages[str(r["stage"])] = stage_data

    summary = {
        "species": species,
        "algorithm": algorithm,
        "hardware": hardware,
        "seed": seed,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stages": stages,
        "total_timesteps": total_timesteps,
        "total_training_time_seconds": round(total_duration, 1),
        "total_training_time": format_duration_hms(total_duration),
        "final_avg_reward": round(final_result["mean_reward"], 2),
    }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(_json.dumps(summary, indent=2) + "\n")
    return summary_path


def save_results_csv(
    stage_results_list: list[dict[str, Any]],
    stage_configs: dict[int, dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    run_dir: "str | Path",
) -> Path:
    """Save a ``collected_results.csv`` to *run_dir*.

    Produces a CSV with the same column structure as the sweep's
    ``collected_results.csv`` so that single-run notebook results can be
    analysed with the same downstream tooling (plotting, comparison
    scripts, etc.).

    Each stage produces one row containing:

    * **Fixed columns** — ``species``, ``algorithm``, ``seed``, ``stage``
    * **Hyperparameter columns** — all ``env_kwargs``, ``ppo_kwargs``/
      ``sac_kwargs``, and ``curriculum_kwargs`` values from the stage
      config, prefixed with ``env_``, ``ppo_``/``sac_``, or
      ``curriculum_`` respectively (matching sweep CSV conventions).
    * **Metric columns** — ``best_mean_reward``,
      ``best_mean_episode_length``, ``last_mean_reward``,
      ``last_mean_episode_length``, ``mean_forward_vel``,
      ``mean_success_rate``, ``training_duration_seconds``,
      curriculum thresholds, and ``stage_passed``.

    Returns the path to the written CSV file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for r in stage_results_list:
        stage = r["stage"]
        cfg = stage_configs[stage]

        row: dict[str, Any] = {
            "species": species,
            "algorithm": algorithm,
            "seed": seed,
            "stage": stage,
        }

        # ── Hyperparameters (mirroring sweep CSV key names) ─────────
        for key, val in cfg.get("env_kwargs", {}).items():
            row[f"env_{key}"] = val
        algo_key = "sac_kwargs" if algorithm.lower() == "sac" else "ppo_kwargs"
        for key, val in cfg.get(algo_key, {}).items():
            if key == "policy_kwargs":
                # Flatten net_arch to a string like the sweep CSV does
                net_arch = val.get("net_arch", [])
                row[f"{algorithm.lower()}_net_arch"] = str(net_arch)
            elif key == "verbose":
                continue
            else:
                row[f"{algorithm.lower()}_{key}"] = val
        for key, val in cfg.get("curriculum_kwargs", {}).items():
            row[f"curriculum_{key}"] = val

        # ── Metrics ─────────────────────────────────────────────────
        best_model_reward = r.get("best_model_reward", "")
        row["best_mean_reward"] = float(best_model_reward) if best_model_reward != "" else r.get("best_eval_reward", "")
        best_model_length = r.get("best_model_length", "")
        row["best_mean_episode_length"] = (
            float(best_model_length) if best_model_length != "" else r.get("best_eval_length", "")
        )
        row["last_mean_reward"] = round(r.get("mean_reward", 0.0), 2)
        row["last_mean_episode_length"] = round(r.get("mean_episode_length", 0.0), 1)
        row["mean_forward_vel"] = round(r.get("mean_forward_vel", 0.0), 2)
        row["std_forward_vel"] = round(r.get("std_forward_vel", 0.0), 2)
        row["mean_distance_traveled"] = round(r.get("mean_distance_traveled", 0.0), 2)
        row["mean_success_rate"] = round(r.get("mean_success_rate", 0.0), 4)
        row["training_duration_seconds"] = round(r.get("duration_seconds", 0.0), 1)

        # Curriculum thresholds
        cur = cfg.get("curriculum_kwargs", {})
        row["reward_threshold"] = cur.get("min_avg_reward", "")
        row["ep_length_threshold"] = cur.get("min_avg_episode_length", "")
        row["forward_vel_threshold"] = cur.get("min_avg_forward_vel", "")
        row["success_rate_threshold"] = cur.get("min_success_rate", "")
        row["stage_passed"] = r.get("gate_passed", "")

        # Quality evaluation metrics (eval_* keys from quality eval)
        for key, val in r.items():
            if key.startswith("eval_"):
                row[key] = val

        rows.append(row)

    return write_results_csv(
        rows,
        run_dir / "collected_results.csv",
        fixed_columns=["species", "algorithm", "seed", "stage"],
    )


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

    # ── Duration from metrics.json fallback ─────────────────────────────
    if duration_seconds == 0.0:
        metrics_path = stage_dir / "metrics.json"
        if metrics_path.exists():
            metrics = _json.loads(metrics_path.read_text())
            duration_seconds = metrics.get("training_duration_seconds", 0.0)

    best_model_path = model_dir / "best_model"
    vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
    sim_dt = stage_config.get("env_kwargs", {}).get("sim_dt", 0.01)

    return {
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
        except ImportError:
            logger.warning("Skipping graph generation (matplotlib not installed).")
        except Exception:
            logger.warning("Graph generation failed.", exc_info=True)

    if not record_videos:
        return stage_results

    # ── Record replay videos for best and final models ──────────────────
    try:
        from environments.shared.evaluation import record_stage_video
        from environments.shared.train_base import _ensure_sb3

        sb3 = _ensure_sb3()
        env_kwargs = stage_config["env_kwargs"].copy()
        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]

        best_model_path = model_dir / "best_model"
        vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
        final_path = model_dir / f"stage{stage}_final"
        final_vecnorm_path = str(final_path) + "_vecnorm.pkl"

        if (model_dir / "best_model.zip").exists():
            best_model = alg_cls.load(str(best_model_path))
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
            )

        if (Path(str(final_path) + ".zip")).exists():
            final_model = alg_cls.load(str(final_path))
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
            )
    except Exception:
        logger.warning("Video recording failed.", exc_info=True)

    return stage_results
