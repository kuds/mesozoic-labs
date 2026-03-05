"""Trial result collection, CSV export, and visualisation."""

import logging
from pathlib import Path
from typing import Any

from .constants import SweepStageError

logger = logging.getLogger(__name__)


def _collect_trial_results(hpt_job: Any, stage: int, stage_config: dict) -> list[dict]:
    """Extract per-trial hyperparameters and outcomes from a completed HPT job.

    Each returned dict contains:

    * ``trial_id`` — Vertex AI trial identifier
    * ``stage`` — curriculum stage number
    * one key per hyperparameter (e.g. ``ppo_learning_rate``)
    * ``best_mean_reward`` — final metric reported by the trial
    * ``best_mean_episode_length`` — episode length from the best eval
    * ``reward_threshold`` — ``min_avg_reward`` from the stage TOML config
    * ``ep_length_threshold`` — ``min_avg_episode_length`` from config
    * ``forward_vel_threshold`` — ``min_avg_forward_vel`` from config
    * ``success_rate_threshold`` — ``min_success_rate`` from config
    * ``stage_passed`` — ``True`` when all curriculum criteria are met
    """
    cur = stage_config.get("curriculum_kwargs", {})
    reward_threshold = cur.get("min_avg_reward")
    ep_length_threshold = cur.get("min_avg_episode_length")
    forward_vel_threshold = cur.get("min_avg_forward_vel")
    success_rate_threshold = cur.get("min_success_rate")

    rows: list[dict] = []
    for trial in hpt_job.trials:
        row: dict = {"trial_id": trial.id, "stage": stage}
        if hasattr(trial, "parameters") and trial.parameters:
            for param in trial.parameters:
                row[param.parameter_id] = param.value

        # Extract all reported metrics from the trial
        metrics: dict[str, float | None] = {}
        if trial.final_measurement and trial.final_measurement.metrics:
            for metric in trial.final_measurement.metrics:
                metrics[metric.metric_id] = metric.value

        best_reward = metrics.get("best_mean_reward")
        best_ep_length = metrics.get("best_mean_episode_length")
        last_reward = metrics.get("last_mean_reward")
        last_ep_length = metrics.get("last_mean_episode_length")
        trial_seed = metrics.get("seed")

        row["seed"] = int(trial_seed) if trial_seed is not None else None
        row["best_mean_reward"] = best_reward
        row["best_mean_episode_length"] = best_ep_length
        row["last_mean_reward"] = last_reward
        row["last_mean_episode_length"] = last_ep_length
        row["reward_threshold"] = reward_threshold
        row["ep_length_threshold"] = ep_length_threshold
        row["forward_vel_threshold"] = forward_vel_threshold
        row["success_rate_threshold"] = success_rate_threshold

        # Check all curriculum criteria.  When no thresholds are defined
        # (e.g. Stage 3 hunting stages), the trial passes by default as
        # long as it produced a valid reward.
        passed = best_reward is not None
        fail_reasons: list[str] = []
        if best_reward is None:
            fail_reasons.append("no reward reported (trial may have crashed)")
        if reward_threshold is not None and (best_reward is None or best_reward < reward_threshold):
            passed = False
            fail_reasons.append(f"reward {best_reward} < threshold {reward_threshold}")
        if passed and ep_length_threshold is not None:
            if best_ep_length is None or best_ep_length < ep_length_threshold:
                passed = False
                fail_reasons.append(f"ep_length {best_ep_length} < threshold {ep_length_threshold}")
        if passed and forward_vel_threshold is not None:
            trial_fwd_vel = metrics.get("best_mean_forward_vel")
            if trial_fwd_vel is None or trial_fwd_vel < forward_vel_threshold:
                passed = False
                fail_reasons.append(f"forward_vel {trial_fwd_vel} < threshold {forward_vel_threshold}")
        if passed and success_rate_threshold is not None:
            trial_success_rate = metrics.get("best_mean_success_rate")
            if trial_success_rate is None or trial_success_rate < success_rate_threshold:
                passed = False
                fail_reasons.append(f"success_rate {trial_success_rate} < threshold {success_rate_threshold}")
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

    Each row records the trial ID, stage, all hyperparameter values,
    performance metrics, curriculum thresholds, and whether the trial
    met all stage advancement criteria.

    Args:
        rows: List of result dicts from :func:`_collect_trial_results`.
        path: Output CSV path (parent directories are created as needed).

    Returns:
        Path to the written CSV file.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        logger.warning("No trial rows to write — skipping CSV")
        return path

    fixed_cols = ["trial_id", "stage", "seed"]
    metric_cols = [
        "best_mean_reward",
        "best_mean_episode_length",
        "last_mean_reward",
        "last_mean_episode_length",
        "reward_threshold",
        "ep_length_threshold",
        "forward_vel_threshold",
        "success_rate_threshold",
        "stage_passed",
    ]
    # Collect all hyperparameter column names across all rows (union, sorted)
    hparam_cols: list[str] = sorted({k for row in rows for k in row if k not in fixed_cols + metric_cols})
    fieldnames = fixed_cols + hparam_cols + metric_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Sweep results written to: %s", path)
    return path


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

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("Sweep CSV not found: %s — skipping visualisations", csv_path)
        return

    if save_dir is None:
        save_dir = csv_path.parent
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV into list of dicts
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

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

    for stage in stages:
        stage_rows = [r for r in rows if int(r["stage"]) == stage]
        trial_ids = [r["trial_id"] for r in stage_rows]
        label = f"Stage {stage}"
        color = stage_colors.get(stage, "#333333")

        # [0,0] Best Mean Reward
        rewards = [_float(r.get("best_mean_reward")) for r in stage_rows]
        valid = [(tid, rw) for tid, rw in zip(trial_ids, rewards) if rw is not None]
        if valid:
            tids, rws = zip(*valid)
            x = np.arange(len(tids))
            axes1[0, 0].bar(x, rws, color=color, alpha=0.7, label=label)
            axes1[0, 0].set_xticks(x)
            axes1[0, 0].set_xticklabels(tids, rotation=45, fontsize=7)
            # Draw threshold line if available
            threshold = _float(stage_rows[0].get("reward_threshold"))
            if threshold is not None:
                axes1[0, 0].axhline(y=threshold, color=color, linestyle="--", alpha=0.5)

        # [0,1] Best Mean Episode Length
        ep_lengths = [_float(r.get("best_mean_episode_length")) for r in stage_rows]
        valid_ep = [(tid, el) for tid, el in zip(trial_ids, ep_lengths) if el is not None]
        if valid_ep:
            tids_ep, els = zip(*valid_ep)
            x = np.arange(len(tids_ep))
            axes1[0, 1].bar(x, els, color=color, alpha=0.7, label=label)
            axes1[0, 1].set_xticks(x)
            axes1[0, 1].set_xticklabels(tids_ep, rotation=45, fontsize=7)
            ep_threshold = _float(stage_rows[0].get("ep_length_threshold"))
            if ep_threshold is not None:
                axes1[0, 1].axhline(y=ep_threshold, color=color, linestyle="--", alpha=0.5)

        # [1,0] Best vs Last Mean Reward (training stability)
        best_last = [(_float(r.get("best_mean_reward")), _float(r.get("last_mean_reward"))) for r in stage_rows]
        valid_bl = [(b, last) for b, last in best_last if b is not None and last is not None]
        if valid_bl:
            bests, lasts = zip(*valid_bl)
            axes1[1, 0].scatter(bests, lasts, color=color, alpha=0.7, label=label, edgecolors="white", s=50)

    axes1[0, 0].set_xlabel("Trial ID")
    axes1[0, 0].set_ylabel("Best Mean Reward")
    axes1[0, 0].set_title("Best Mean Reward per Trial")
    axes1[0, 0].legend()
    axes1[0, 0].grid(True, alpha=0.3)

    axes1[0, 1].set_xlabel("Trial ID")
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
    fixed_cols = {
        "trial_id",
        "stage",
        "seed",
        "best_mean_reward",
        "best_mean_episode_length",
        "last_mean_reward",
        "last_mean_episode_length",
        "reward_threshold",
        "ep_length_threshold",
        "forward_vel_threshold",
        "success_rate_threshold",
        "stage_passed",
    }
    hparam_cols = [k for k in rows[0].keys() if k not in fixed_cols]
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
