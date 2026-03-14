"""Trial result collection, CSV export, and visualisation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from environments.shared.reporting import CSV_METRIC_COLUMNS
from environments.shared.reporting import write_results_csv as _write_results_csv

from .constants import SweepStageError

logger = logging.getLogger(__name__)


def _evaluate_curriculum_gate(
    best_reward: float | None,
    aux_metrics: dict[str, float],
    reward_threshold: float | None,
    ep_length_threshold: float | None,
    forward_vel_threshold: float | None,
    success_rate_threshold: float | None,
) -> tuple[bool, list[str]]:
    """Evaluate whether a trial meets all curriculum advancement criteria.

    Returns:
        A tuple of ``(passed, fail_reasons)`` where *passed* is ``True``
        when all criteria are met and *fail_reasons* lists human-readable
        explanations of which criteria were not met.
    """
    passed = best_reward is not None
    fail_reasons: list[str] = []

    if best_reward is None:
        fail_reasons.append("no reward reported (trial may have crashed)")
    if reward_threshold is not None and (best_reward is None or best_reward < reward_threshold):
        passed = False
        fail_reasons.append(f"reward {best_reward} < threshold {reward_threshold}")
    if ep_length_threshold is not None:
        best_ep_length = aux_metrics.get("best_mean_episode_length")
        if best_ep_length is None or best_ep_length < ep_length_threshold:
            passed = False
            fail_reasons.append(f"ep_length {best_ep_length} < threshold {ep_length_threshold}")
    if forward_vel_threshold is not None:
        trial_fwd_vel = aux_metrics.get("best_mean_forward_vel")
        if trial_fwd_vel is None or trial_fwd_vel < forward_vel_threshold:
            passed = False
            fail_reasons.append(f"forward_vel {trial_fwd_vel} < threshold {forward_vel_threshold}")
    if success_rate_threshold is not None:
        trial_success_rate = aux_metrics.get("best_mean_success_rate")
        if trial_success_rate is None or trial_success_rate < success_rate_threshold:
            passed = False
            fail_reasons.append(f"success_rate {trial_success_rate} < threshold {success_rate_threshold}")

    return passed, fail_reasons


def _extract_thresholds(config: dict) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract curriculum thresholds from a stage config dict.

    Supports both runtime TOML configs (key: ``curriculum_kwargs``) and
    serialized JSON configs (key: ``curriculum``).

    Returns:
        A tuple of ``(reward, ep_length, forward_vel, success_rate)`` thresholds.
    """
    cur = config.get("curriculum_kwargs") or config.get("curriculum") or {}
    return (
        cur.get("min_avg_reward"),
        cur.get("min_avg_episode_length"),
        cur.get("min_avg_forward_vel"),
        cur.get("min_success_rate"),
    )


def _load_trial_metrics(output_base: str, trial_id: str) -> dict[str, float]:
    """Load auxiliary metrics from a trial's ``metrics.json`` sidecar.

    Supports both local paths and ``/gcs/``-prefixed paths.  When the
    path starts with ``/gcs/`` and the FUSE mount is not available, the
    file is fetched directly from GCS using the ``google-cloud-storage``
    client library.

    The training code writes all computed metrics (episode lengths,
    forward velocity, success rate, etc.) to ``<trial_dir>/metrics.json``
    so they can be collected without being declared in the HPT
    ``metric_spec``.

    Returns an empty dict if the file is missing (e.g. trial crashed).
    """
    metrics_path = Path(f"{output_base}/{trial_id}/metrics.json")

    # Try local / FUSE-mounted path first.
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                return dict(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("  Trial %s: failed to read metrics.json: %s", trial_id, exc)
            return {}

    # Fall back to reading from GCS when the path looks like a FUSE path
    # but the mount is not available (e.g. orchestrator running locally).
    path_str = str(metrics_path)
    if path_str.startswith("/gcs/"):
        # Convert /gcs/<bucket>/... → gs://<bucket>/...
        gcs_uri = "gs://" + path_str[len("/gcs/") :]
        try:
            from google.cloud import storage as _gcs

            without_scheme = gcs_uri[len("gs://") :]
            bucket_name, _, blob_name = without_scheme.partition("/")
            client = _gcs.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if not blob.exists():
                logger.warning("  Trial %s: metrics.json not found at %s", trial_id, gcs_uri)
                return {}
            data = json.loads(blob.download_as_text())
            return dict(data)
        except ImportError:
            logger.warning(
                "  Trial %s: metrics.json not found locally and google-cloud-storage "
                "is not installed for GCS fallback: %s",
                trial_id,
                path_str,
            )
            return {}
        except json.JSONDecodeError as exc:
            logger.warning("  Trial %s: failed to parse metrics.json from GCS: %s", trial_id, exc)
            return {}
        except OSError as exc:
            logger.warning("  Trial %s: failed to read metrics.json from GCS: %s", trial_id, exc)
            return {}

    logger.warning("  Trial %s: metrics.json not found at %s", trial_id, metrics_path)
    return {}


def _collect_trial_results(hpt_job: Any, stage: int, stage_config: dict, output_base: str) -> list[dict]:
    """Extract per-trial hyperparameters and outcomes from a completed HPT job.

    The primary optimisation metric (``best_mean_reward``) is read from
    the HPT trial's ``final_measurement``.  All auxiliary metrics are
    read from the ``metrics.json`` sidecar written by the training code
    to each trial's GCS output directory.

    Each returned dict contains:

    * ``trial_id`` — Vertex AI trial identifier
    * ``stage`` — curriculum stage number
    * one key per hyperparameter (e.g. ``ppo_learning_rate``)
    * ``best_mean_reward`` — primary HPT metric
    * ``best_mean_episode_length`` — from ``metrics.json``
    * ``last_mean_reward`` — from ``metrics.json``
    * ``last_mean_episode_length`` — from ``metrics.json``
    * ``reward_threshold`` — ``min_avg_reward`` from the stage TOML config
    * ``ep_length_threshold`` — ``min_avg_episode_length`` from config
    * ``forward_vel_threshold`` — ``min_avg_forward_vel`` from config
    * ``success_rate_threshold`` — ``min_success_rate`` from config
    * ``stage_passed`` — ``True`` when all curriculum criteria are met
    """
    reward_threshold, ep_length_threshold, forward_vel_threshold, success_rate_threshold = _extract_thresholds(
        stage_config
    )

    rows: list[dict] = []
    for trial in hpt_job.trials:
        row: dict = {"trial_id": trial.id, "stage": stage}
        if hasattr(trial, "parameters") and trial.parameters:
            for param in trial.parameters:
                row[param.parameter_id] = param.value

        # Primary metric from HPT final_measurement
        best_reward: float | None = None
        if trial.final_measurement and trial.final_measurement.metrics:
            for metric in trial.final_measurement.metrics:
                if metric.metric_id == "best_mean_reward":
                    best_reward = metric.value

        # Auxiliary metrics from the JSON sidecar
        aux = _load_trial_metrics(output_base, trial.id)
        # Prefer HPT value for best_mean_reward (authoritative), fall
        # back to sidecar if HPT somehow didn't capture it.
        if best_reward is None:
            best_reward = aux.get("best_mean_reward")

        row["best_mean_reward"] = best_reward
        row["best_mean_episode_length"] = aux.get("best_mean_episode_length")
        row["last_mean_reward"] = aux.get("last_mean_reward")
        row["last_mean_episode_length"] = aux.get("last_mean_episode_length")
        row["mean_forward_vel"] = aux.get("mean_forward_vel")
        row["std_forward_vel"] = aux.get("std_forward_vel")
        row["mean_distance_traveled"] = aux.get("mean_distance_traveled")
        row["mean_success_rate"] = aux.get("mean_success_rate")
        row["training_duration_seconds"] = aux.get("training_duration_seconds")

        # Quality evaluation metrics (spinning detection, heading, reward breakdown)
        for key, value in aux.items():
            if key.startswith("eval_"):
                row[key] = value

        row["reward_threshold"] = reward_threshold
        row["ep_length_threshold"] = ep_length_threshold
        row["forward_vel_threshold"] = forward_vel_threshold
        row["success_rate_threshold"] = success_rate_threshold

        passed, fail_reasons = _evaluate_curriculum_gate(
            best_reward, aux, reward_threshold, ep_length_threshold, forward_vel_threshold, success_rate_threshold
        )
        row["stage_passed"] = passed

        # Log per-trial diagnostic summary
        if passed:
            logger.info(
                "  Trial %s stage %d: PASSED (reward=%.2f)",
                trial.id,
                stage,
                best_reward,
            )
        else:
            logger.warning(
                "  Trial %s stage %d: FAILED — %s",
                trial.id,
                stage,
                "; ".join(fail_reasons),
            )

        rows.append(row)
    return rows


def write_results_csv(rows: list[dict], path: str | Path) -> Path:
    """Write sweep trial results to a CSV file.

    Thin wrapper around :func:`environments.shared.reporting.write_results_csv`
    with sweep-specific fixed columns (``trial_id``, ``stage``).

    Each row records the trial ID, stage, all hyperparameter values,
    performance metrics, curriculum thresholds, and whether the trial
    met all stage advancement criteria.

    Args:
        rows: List of result dicts from :func:`_collect_trial_results`.
        path: Output CSV path (parent directories are created as needed).
            Supports ``gs://`` URIs.

    Returns:
        Path to the written CSV file.
    """
    return _write_results_csv(
        rows,
        path,
        fixed_columns=["trial_id", "stage"],
    )


def plot_sweep_results(csv_path: str | Path, species: str, algorithm: str, save_dir: str | Path | None = None) -> None:
    """Generate visualisation graphs from sweep results CSV.

    Produces two PNG figures saved alongside the CSV (or to *save_dir*):

    **sweep_trial_metrics.png** — 2x2 grid:
      - [0,0] Best Mean Reward per trial (grouped by stage)
      - [0,1] Best Mean Episode Length per trial (grouped by stage)
      - [1,0] Best vs Last Mean Reward (training stability)
      - [1,1] Stage Pass/Fail summary (stacked bar)

    **sweep_hyperparameter_analysis.png** — Nx1 column:
      - One scatter plot per hyperparameter vs best_mean_reward (colour = stage)

    Args:
        csv_path: Path to the sweep results CSV.
        species: Species name for titles.
        algorithm: Algorithm name for titles.
        save_dir: Directory to save PNGs. Defaults to the CSV's parent directory.
    """
    import csv as _csv

    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend for headless environments
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib or numpy not installed — skipping sweep visualisations")
        return

    csv_path_str = str(csv_path)
    _temp_csv: Path | None = None

    # Support gs:// URIs by downloading to a temp file.
    if csv_path_str.startswith("gs://"):
        import tempfile

        try:
            from google.cloud import storage as _gcs

            without_scheme = csv_path_str[len("gs://") :]
            bucket_name, _, blob_name = without_scheme.partition("/")
            client = _gcs.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if not blob.exists():
                logger.warning("Sweep CSV not found at %s — skipping visualisations", csv_path_str)
                return
            _temp_csv = Path(tempfile.mktemp(suffix=".csv"))
            blob.download_to_filename(str(_temp_csv))
            csv_path = _temp_csv
        except ImportError:
            logger.warning("google-cloud-storage not installed — cannot read %s", csv_path_str)
            return
        except Exception as exc:
            logger.warning("Failed to download %s: %s — skipping visualisations", csv_path_str, exc)
            return
    else:
        csv_path = Path(csv_path_str)

    if not csv_path.exists():
        logger.warning("Sweep CSV not found: %s — skipping visualisations", csv_path)
        return

    if save_dir is None:
        save_dir = Path.cwd() if _temp_csv is not None else csv_path.parent
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV into list of dicts
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    # Clean up temp file if we downloaded from GCS.
    if _temp_csv is not None:
        _temp_csv.unlink(missing_ok=True)

    if not rows:
        logger.warning("Sweep CSV is empty — skipping visualisations")
        return

    # Parse numeric values
    def _float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    stages = sorted({int(r["stage"]) for r in rows if r.get("stage")})
    stage_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    # ── Figure 1: Trial Metrics (2x2) ────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    title = f"{species.capitalize()} {algorithm.upper()} Sweep"
    fig1.suptitle(title, fontsize=14, fontweight="bold")

    # Build combined x-axis labels: "S{stage}_{trial_id}" for each trial
    all_reward_labels: list[str] = []
    all_reward_values: list[float] = []
    all_reward_colors: list[str] = []
    all_ep_labels: list[str] = []
    all_ep_values: list[float] = []
    all_ep_colors: list[str] = []

    for stage in stages:
        stage_rows = [r for r in rows if int(r["stage"]) == stage]
        trial_ids = [r["trial_id"] for r in stage_rows]
        label = f"Stage {stage}"
        color = stage_colors.get(stage, "#333333")

        # [0,0] Best Mean Reward
        for tid, r in zip(trial_ids, stage_rows):
            rw = _float(r.get("best_mean_reward"))
            if rw is not None:
                all_reward_labels.append(f"S{stage}_{tid}")
                all_reward_values.append(rw)
                all_reward_colors.append(color)

        # Draw threshold line if available
        if stage_rows:
            threshold = _float(stage_rows[0].get("reward_threshold"))
            if threshold is not None:
                axes1[0, 0].axhline(y=threshold, color=color, linestyle="--", alpha=0.5, label=f"S{stage} threshold")

        # [0,1] Best Mean Episode Length
        for tid, r in zip(trial_ids, stage_rows):
            el = _float(r.get("best_mean_episode_length"))
            if el is not None:
                all_ep_labels.append(f"S{stage}_{tid}")
                all_ep_values.append(el)
                all_ep_colors.append(color)

        # Draw episode length threshold line once per stage (outside trial loop)
        if stage_rows:
            ep_threshold = _float(stage_rows[0].get("ep_length_threshold"))
            if ep_threshold is not None:
                axes1[0, 1].axhline(y=ep_threshold, color=color, linestyle="--", alpha=0.5, label=f"S{stage} threshold")

        # [1,0] Best vs Last Mean Reward (training stability)
        best_last = [(_float(r.get("best_mean_reward")), _float(r.get("last_mean_reward"))) for r in stage_rows]
        valid_bl = [(b, last) for b, last in best_last if b is not None and last is not None]
        if valid_bl:
            bests, lasts = zip(*valid_bl)
            axes1[1, 0].scatter(bests, lasts, color=color, alpha=0.7, label=label, edgecolors="white", s=50)

    # Render reward bars with non-overlapping x positions
    if all_reward_values:
        x_rw = np.arange(len(all_reward_values))
        axes1[0, 0].bar(x_rw, all_reward_values, color=all_reward_colors, alpha=0.7)
        axes1[0, 0].set_xticks(x_rw)
        axes1[0, 0].set_xticklabels(all_reward_labels, rotation=45, fontsize=7, ha="right")

    # Render episode length bars with non-overlapping x positions
    if all_ep_values:
        x_ep = np.arange(len(all_ep_values))
        axes1[0, 1].bar(x_ep, all_ep_values, color=all_ep_colors, alpha=0.7)
        axes1[0, 1].set_xticks(x_ep)
        axes1[0, 1].set_xticklabels(all_ep_labels, rotation=45, fontsize=7, ha="right")

    axes1[0, 0].set_xlabel("Trial")
    axes1[0, 0].set_ylabel("Best Mean Reward")
    axes1[0, 0].set_title("Best Mean Reward per Trial")
    axes1[0, 0].legend()
    axes1[0, 0].grid(True, alpha=0.3)

    axes1[0, 1].set_xlabel("Trial")
    axes1[0, 1].set_ylabel("Best Mean Episode Length")
    axes1[0, 1].set_title("Best Mean Episode Length per Trial")
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)

    axes1[1, 0].set_xlabel("Best Mean Reward")
    axes1[1, 0].set_ylabel("Last Mean Reward")
    axes1[1, 0].set_title("Training Stability (Best vs Last Reward)")
    # Draw y=x reference line
    all_rewards = [_float(r.get("best_mean_reward")) for r in rows]
    all_rewards = [v for v in all_rewards if v is not None]
    if all_rewards:
        lo, hi = min(all_rewards), max(all_rewards)
        axes1[1, 0].plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y=x")
    axes1[1, 0].legend()
    axes1[1, 0].grid(True, alpha=0.3)

    # [1,1] Stage Pass/Fail Summary
    ax_pf = axes1[1, 1]
    pass_counts = []
    fail_counts = []
    stage_labels = []
    for stage in stages:
        stage_rows = [r for r in rows if int(r["stage"]) == stage]
        passed = sum(1 for r in stage_rows if str(r.get("stage_passed", "")).lower() == "true")
        failed = len(stage_rows) - passed
        pass_counts.append(passed)
        fail_counts.append(failed)
        stage_labels.append(f"Stage {stage}")

    x_pf = np.arange(len(stages))
    ax_pf.bar(x_pf, pass_counts, color="#2ca02c", alpha=0.8, label="Passed")
    ax_pf.bar(x_pf, fail_counts, bottom=pass_counts, color="#d62728", alpha=0.8, label="Failed")
    ax_pf.set_xticks(x_pf)
    ax_pf.set_xticklabels(stage_labels)
    ax_pf.set_ylabel("Number of Trials")
    ax_pf.set_title("Stage Pass/Fail Summary")
    ax_pf.legend()
    ax_pf.grid(True, alpha=0.3, axis="y")

    fig1.tight_layout()
    fig1_path = save_dir / "sweep_trial_metrics.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info("Sweep trial metrics graph saved to: %s", fig1_path)

    # ── Figure 2: Hyperparameter Analysis ─────────────────────────────────────
    # Identify hyperparameter columns (not fixed or metric columns)
    fixed_cols = {"trial_id", "stage"} | set(CSV_METRIC_COLUMNS)
    # Exclude eval_* quality metrics from hyperparameter analysis
    hparam_cols = [k for k in rows[0].keys() if k not in fixed_cols and not k.startswith("eval_")]
    # Filter to columns with numeric, varying values
    numeric_hparams = []
    for col in hparam_cols:
        vals = [_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2 and len(set(vals)) > 1:
            numeric_hparams.append(col)

    if numeric_hparams:
        n_params = len(numeric_hparams)
        fig2, axes2 = plt.subplots(n_params, 1, figsize=(10, 4 * n_params), squeeze=False)
        fig2.suptitle(f"{title} — Hyperparameter vs Reward", fontsize=14, fontweight="bold")

        for idx, hparam in enumerate(numeric_hparams):
            ax = axes2[idx, 0]
            for stage in stages:
                stage_rows = [r for r in rows if int(r["stage"]) == stage]
                xs = [_float(r.get(hparam)) for r in stage_rows]
                ys = [_float(r.get("best_mean_reward")) for r in stage_rows]
                valid_hp = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
                if valid_hp:
                    hx, hy = zip(*valid_hp)
                    ax.scatter(
                        hx,
                        hy,
                        color=stage_colors.get(stage, "#333"),
                        alpha=0.7,
                        label=f"Stage {stage}",
                        edgecolors="white",
                        s=50,
                    )
            ax.set_xlabel(hparam)
            ax.set_ylabel("Best Mean Reward")
            ax.set_title(f"{hparam} vs Best Mean Reward")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig2.tight_layout()
        fig2_path = save_dir / "sweep_hyperparameter_analysis.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info("Sweep hyperparameter analysis graph saved to: %s", fig2_path)
    else:
        logger.info("No varying numeric hyperparameters found — skipping hyperparameter analysis graph")


def _download_gcs_results(gcs_uri: str, local_dir: Path) -> Path:
    """Download ``metrics.json`` and ``stage_config.json`` from a GCS URI.

    Mirrors the remote directory structure into *local_dir* so the
    local-path logic in :func:`collect_results_from_disk` can process
    the files without modification.

    Args:
        gcs_uri: A ``gs://bucket/prefix`` URI pointing at the root
            directory containing ``stage*`` sub-directories.
        local_dir: Local directory to download files into.

    Returns:
        Path to the local mirror root (same as *local_dir*).

    Raises:
        ImportError: If ``google-cloud-storage`` is not installed.
        ValueError: If *gcs_uri* is not a valid ``gs://`` URI.
    """
    from google.cloud import storage as _gcs

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {gcs_uri}")

    # Parse gs://bucket/prefix
    without_scheme = gcs_uri[len("gs://") :]
    slash_idx = without_scheme.find("/")
    if slash_idx == -1:
        bucket_name = without_scheme
        prefix = ""
    else:
        bucket_name = without_scheme[:slash_idx]
        prefix = without_scheme[slash_idx + 1 :].rstrip("/")

    client = _gcs.Client()
    bucket = client.bucket(bucket_name)

    blob_prefix = f"{prefix}/" if prefix else ""
    target_filenames = {"metrics.json", "stage_config.json"}
    downloaded = 0

    for blob in bucket.list_blobs(prefix=blob_prefix):
        # Only download the JSON files we care about.
        basename = blob.name.rsplit("/", 1)[-1] if "/" in blob.name else blob.name
        if basename not in target_filenames:
            continue

        # Compute the relative path under the prefix.
        rel_path = blob.name[len(blob_prefix) :] if blob_prefix else blob.name
        local_path = local_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        downloaded += 1

    logger.info(
        "Downloaded %d file(s) from gs://%s/%s to %s",
        downloaded,
        bucket_name,
        prefix,
        local_dir,
    )
    return local_dir


def collect_results_from_disk(
    output_dir: str | Path,
    species: str | None = None,
    stages: list[int] | None = None,
) -> list[dict]:
    """Scan trial directories on disk and collect results into row dicts.

    This is the offline equivalent of :func:`_collect_trial_results` — it
    reads ``metrics.json`` sidecars directly from the filesystem without
    needing a Vertex AI HPT job object.

    Supports both local paths and ``gs://`` URIs.  When a ``gs://`` URI
    is provided, the relevant JSON files are downloaded to a temporary
    directory which is cleaned up automatically.

    Expected directory layout (one of)::

        # Sweep layout: output_dir = .../sweeps/<species>
        output_dir/stage1/<trial_id>/metrics.json
        output_dir/stage2/<trial_id>/metrics.json

        # Curriculum layout: output_dir = .../curriculum_<timestamp>
        output_dir/stage1/metrics.json
        output_dir/stage2/metrics.json

    Each trial's ``stage_config.json`` (if present) is used to populate
    curriculum thresholds and evaluate pass/fail status.

    Args:
        output_dir: Root directory containing stage sub-directories, or a
            ``gs://bucket/prefix`` URI.
        species: Species name (for logging only; optional).
        stages: Restrict collection to these stage numbers.  Defaults to
            all ``stage*`` sub-directories found.

    Returns:
        List of result dicts compatible with :func:`write_results_csv`.
    """
    import tempfile

    # Handle gs:// URIs by downloading to a temp directory.
    cleanup_tempdir = None
    output_dir_str = str(output_dir)
    if output_dir_str.startswith("gs://"):
        tmpdir = tempfile.mkdtemp(prefix="sweep_collect_")
        cleanup_tempdir = tmpdir
        try:
            output_dir = _download_gcs_results(output_dir_str, Path(tmpdir))
        except Exception:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
    else:
        output_dir = Path(output_dir)

    try:
        return _collect_results_local(output_dir, species, stages)
    finally:
        if cleanup_tempdir is not None:
            import shutil

            shutil.rmtree(cleanup_tempdir, ignore_errors=True)


def _extract_hyperparameters(config: dict) -> dict[str, Any]:
    """Extract hyperparameter settings from a stage_config.json dict.

    Converts the nested ``hyperparameters``, ``reward_weights``, and
    ``curriculum`` sections into flat keys using the same
    ``<prefix>_<param>`` naming convention used by the Vertex AI HPT
    parameter injection (e.g. ``ppo_learning_rate``, ``env_alive_bonus``,
    ``curriculum_timesteps``).

    Nested dicts (like ``policy_kwargs``) are flattened with an underscore
    separator (e.g. ``ppo_policy_kwargs_net_arch``).

    Returns:
        A flat dict of ``{param_id: value}`` pairs.
    """
    params: dict[str, Any] = {}
    algorithm = (config.get("algorithm") or "ppo").lower()

    # Algorithm hyperparameters (e.g. learning_rate → ppo_learning_rate)
    for key, value in (config.get("hyperparameters") or {}).items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                params[f"{algorithm}_{key}_{sub_key}"] = sub_value
        else:
            params[f"{algorithm}_{key}"] = value

    # Environment / reward weights (e.g. alive_bonus → env_alive_bonus)
    for key, value in (config.get("reward_weights") or {}).items():
        params[f"env_{key}"] = value

    return params


def _collect_results_local(
    output_dir: Path,
    species: str | None = None,
    stages: list[int] | None = None,
) -> list[dict]:
    """Collect results from a local directory tree.

    This is the internal implementation used by :func:`collect_results_from_disk`
    after any GCS download has been completed.
    """
    if not output_dir.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return []

    # Discover stage directories
    stage_dirs: list[tuple[int, Path]] = []
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and child.name.startswith("stage"):
            try:
                stage_num = int(child.name.replace("stage", "").split("_")[0])
            except ValueError:
                continue
            if stages and stage_num not in stages:
                continue
            stage_dirs.append((stage_num, child))

    if not stage_dirs:
        logger.warning("No stage directories found under %s", output_dir)
        return []

    rows: list[dict] = []
    for stage_num, stage_path in stage_dirs:
        # Load stage_config.json for thresholds and hyperparameters (may be
        # at the stage level or inside each trial directory — check the
        # stage level first).
        stage_config_path = stage_path / "stage_config.json"
        stage_cfg: dict = {}
        if stage_config_path.exists():
            try:
                with open(stage_config_path) as f:
                    stage_cfg = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read %s: %s", stage_config_path, exc)

        reward_threshold, ep_length_threshold, forward_vel_threshold, success_rate_threshold = _extract_thresholds(
            stage_cfg
        )
        stage_hparams = _extract_hyperparameters(stage_cfg)

        # Check if this is a single-trial (curriculum) layout where
        # metrics.json sits directly in the stage directory.
        direct_metrics = stage_path / "metrics.json"
        if direct_metrics.exists():
            trial_dirs = [(stage_path.name, stage_path)]
        else:
            # Sweep layout: each child is a trial directory.
            trial_dirs = []
            for td in sorted(stage_path.iterdir()):
                if td.is_dir() and (td / "metrics.json").exists():
                    trial_dirs.append((td.name, td))

        for trial_id, trial_path in trial_dirs:
            metrics = _load_trial_metrics(str(trial_path.parent), trial_path.name)
            if not metrics:
                logger.warning("Stage %d trial %s: no valid metrics.json", stage_num, trial_id)
                continue

            # Load per-trial stage_config.json if the stage-level one wasn't found.
            trial_reward_th = reward_threshold
            trial_ep_th = ep_length_threshold
            trial_fwd_th = forward_vel_threshold
            trial_sr_th = success_rate_threshold
            trial_hparams = stage_hparams
            if not stage_cfg:
                per_trial_cfg_path = trial_path / "stage_config.json"
                if per_trial_cfg_path.exists():
                    try:
                        with open(per_trial_cfg_path) as f:
                            trial_cfg = json.load(f)
                        trial_reward_th, trial_ep_th, trial_fwd_th, trial_sr_th = _extract_thresholds(trial_cfg)
                        trial_hparams = _extract_hyperparameters(trial_cfg)
                    except (json.JSONDecodeError, OSError):
                        pass

            best_reward = metrics.get("best_mean_reward")
            row: dict = {
                "trial_id": trial_id,
                "stage": stage_num,
            }

            # Add hyperparameter settings from stage_config.json
            row.update(trial_hparams)

            # Include any extra keys from metrics.json (e.g. hyperparameters,
            # auxiliary metrics) that aren't in the fixed/metric column sets.
            _fixed_metric_keys = {"trial_id", "stage"} | set(CSV_METRIC_COLUMNS)
            for mk, mv in metrics.items():
                if mk not in _fixed_metric_keys:
                    row[mk] = mv
            row.update(
                {
                    "best_mean_reward": best_reward,
                    "best_mean_episode_length": metrics.get("best_mean_episode_length"),
                    "last_mean_reward": metrics.get("last_mean_reward"),
                    "last_mean_episode_length": metrics.get("last_mean_episode_length"),
                    "mean_forward_vel": metrics.get("mean_forward_vel"),
                    "std_forward_vel": metrics.get("std_forward_vel"),
                    "mean_distance_traveled": metrics.get("mean_distance_traveled"),
                    "mean_success_rate": metrics.get("mean_success_rate"),
                    "training_duration_seconds": metrics.get("training_duration_seconds"),
                    "reward_threshold": trial_reward_th,
                    "ep_length_threshold": trial_ep_th,
                    "forward_vel_threshold": trial_fwd_th,
                    "success_rate_threshold": trial_sr_th,
                }
            )

            passed, _ = _evaluate_curriculum_gate(
                best_reward, metrics, trial_reward_th, trial_ep_th, trial_fwd_th, trial_sr_th
            )
            row["stage_passed"] = passed

            rows.append(row)

        logger.info(
            "Stage %d: collected %d trial(s) from %s",
            stage_num,
            len([r for r in rows if r["stage"] == stage_num]),
            stage_path,
        )

    species_label = species or output_dir.name
    logger.info("Collected %d total rows for %s", len(rows), species_label)
    return rows


def _best_trial_model_path(stage_rows: list[dict], bucket: str, species: str, stage: int) -> tuple[str, dict]:
    """Return the GCS path of the best trial's best model and its result row.

    Each trial saves its best model (highest eval reward) to::

        /gcs/<bucket>/sweeps/<species>/stage<N>/<trial_id>/models/best_model.zip

    alongside a matched VecNormalize snapshot at ``best_model_vecnorm.pkl``.

    This function inspects the completed trial rows, identifies the trial with
    the highest ``best_mean_reward`` among those that passed the stage, and
    returns the best model checkpoint path so the next stage's sweep can
    warm-start from it.

    Returns:
        A tuple of ``(model_path, best_row)`` where ``best_row`` is the full
        result dict for the winning trial.  This allows callers to inspect
        hyperparameters (e.g. ``net_arch``) that must be propagated forward.
    """
    best_row: dict | None = None
    best_value = float("-inf")

    for row in stage_rows:
        if row.get("stage_passed"):
            best_reward = row.get("best_mean_reward")
            if best_reward is not None and best_reward > best_value:
                best_value = best_reward
                best_row = row

    if best_row is None:
        raise SweepStageError(
            f"No trials passed stage {stage} criteria. Re-run with adjusted thresholds or more trials."
        )

    best_trial_id = best_row["trial_id"]
    logger.info("Best passing trial for stage %d: id=%s  best_mean_reward=%.4f", stage, best_trial_id, best_value)

    # Use a pre-computed model_path if available (e.g. from a partial-resume
    # merge where trials come from different GCS output directories).
    if "model_path" in best_row:
        path = best_row["model_path"]
    else:
        path = f"/gcs/{bucket}/sweeps/{species}/stage{stage}/{best_trial_id}/models/best_model.zip"
    return path, best_row


def _best_trial_model_path_any(stage_rows: list[dict], bucket: str, species: str, stage: int) -> tuple[str, dict]:
    """Return the GCS path of the best trial's model, ignoring pass/fail status.

    Like :func:`_best_trial_model_path` but selects the trial with the highest
    ``best_mean_reward`` regardless of whether it met the curriculum gate.
    Used by ``--force-continue`` to chain stages even when no trial passes.

    Raises:
        SweepStageError: If no trial reported a valid reward.
    """
    best_row: dict | None = None
    best_value = float("-inf")

    for row in stage_rows:
        best_reward = row.get("best_mean_reward")
        if best_reward is not None and best_reward > best_value:
            best_value = best_reward
            best_row = row

    if best_row is None:
        raise SweepStageError(f"No trials reported a valid reward for stage {stage}. All trials may have crashed.")

    best_trial_id = best_row["trial_id"]
    passed_str = "PASSED" if best_row.get("stage_passed") else "FAILED gate"
    logger.info(
        "Best trial for stage %d (force-continue): id=%s  best_mean_reward=%.4f (%s)",
        stage,
        best_trial_id,
        best_value,
        passed_str,
    )

    if "model_path" in best_row:
        path = best_row["model_path"]
    else:
        path = f"/gcs/{bucket}/sweeps/{species}/stage{stage}/{best_trial_id}/models/best_model.zip"
    return path, best_row
