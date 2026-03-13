"""
Weights & Biases integration for experiment tracking.

Provides a Stable-Baselines3 callback that logs per-component rewards,
evaluation metrics, curriculum stage, and hyperparameters to W&B.
Supports video recording of evaluation episodes.

Usage:
    from environments.shared.wandb_integration import WandbCallback, init_wandb

    run = init_wandb(
        species="velociraptor",
        stage=1,
        config=stage_config,
    )

    wandb_callback = WandbCallback(video_env=eval_env, video_freq=50000)

    model.learn(
        total_timesteps=500_000,
        callback=CallbackList([eval_callback, wandb_callback]),
    )

    run.finish()

Requires: pip install wandb
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    BaseCallback = object  # type: ignore[misc,assignment]


def is_available() -> bool:
    """Check whether wandb is installed."""
    return wandb is not None


def init_wandb(
    species: str,
    stage: int,
    config: dict[str, Any],
    project: str = "mesozoic-labs",
    tags: list | None = None,
    notes: str | None = None,
) -> Any:
    """Initialize a W&B run for a training session.

    Args:
        species: Species name (e.g. "velociraptor").
        stage: Curriculum stage number.
        config: Full stage config dict (from ``load_stage_config``).
        project: W&B project name.
        tags: Optional list of tags.
        notes: Optional run notes.

    Returns:
        The ``wandb.Run`` object, or ``None`` if wandb is not installed.
    """
    if not is_available():
        logger.warning("wandb not installed. Skipping W&B initialization.")
        return None

    run_name = f"{species}-stage{stage}"

    # Collect git info
    git_hash = _get_git_hash()

    flat_config = {
        "species": species,
        "stage": stage,
        "stage_name": config.get("name", ""),
        "git_hash": git_hash,
    }

    # Flatten env and algorithm kwargs into the config
    for key, value in config.get("env_kwargs", {}).items():
        flat_config[f"env/{key}"] = value
    for key, value in config.get("ppo_kwargs", {}).items():
        flat_config[f"ppo/{key}"] = value
    for key, value in config.get("sac_kwargs", {}).items():
        flat_config[f"sac/{key}"] = value

    all_tags = [species, f"stage{stage}"]
    if tags:
        all_tags.extend(tags)

    run = wandb.init(
        project=project,
        name=run_name,
        config=flat_config,
        tags=all_tags,
        notes=notes,
        reinit=True,
    )

    logger.info("W&B run initialized: %s (%s)", run.name, run.url)

    # Configure metric display and dashboard panels
    setup_wandb_metrics(stage)
    create_wandb_dashboard(stage)

    return run


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class WandbCallback(BaseCallback):
    """Stable-Baselines3 callback that logs training metrics to W&B.

    Logs per-component reward breakdowns, episode statistics, and
    learning rate at each rollout end. Optionally records evaluation
    episode videos at a configurable frequency. Designed to work with
    the info dicts produced by ``BaseDinoEnv``.

    Args:
        log_freq: Log metrics every N training steps.
        video_env: Optional evaluation VecEnv with ``render_mode="rgb_array"``
            for recording videos. If ``None``, video recording is disabled.
        video_freq: Record a video every N training steps.
        video_length: Maximum number of steps per recorded episode.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        log_freq: int = 1000,
        video_env: Any = None,
        video_freq: int = 50000,
        video_length: int = 500,
        verbose: int = 0,
    ):
        if BaseCallback is not object:
            super().__init__(verbose)
        else:
            super().__init__()
        self.log_freq = log_freq
        self.video_env = video_env
        self.video_freq = video_freq
        self.video_length = video_length
        self._last_video_step = 0

    def _on_step(self) -> bool:
        if not is_available() or wandb.run is None:
            return True

        if self.num_timesteps % self.log_freq != 0:
            return True

        metrics: dict[str, Any] = {
            "train/timesteps": self.num_timesteps,
        }

        # Log info from the most recent environment steps.
        # Re-use the canonical key lists from DiagnosticsCallback so all
        # tracking backends (TensorBoard, W&B) stay in sync automatically.
        if self.locals.get("infos"):
            from .diagnostics import DiagnosticsCallback as _DC

            info_keys = (
                list(_DC.REWARD_KEYS)
                + list(_DC.INFO_KEYS)
                + [
                    # Extra keys not tracked by DiagnosticsCallback but useful
                    # for W&B dashboards (species-specific raw signals).
                    "reward_total",
                    "backward_vel",
                    "drift_distance",
                    "spin_instability",
                    "head_food_distance",
                    "bite_success",
                    "food_reached",
                    "torso_height",
                    "reward_neck",
                    "reward_food_reach",
                    "jaw_distance",
                ]
            )
            for key in info_keys:
                values = [info[key] for info in self.locals["infos"] if key in info]
                if values:
                    metrics[f"reward/{key}"] = float(sum(values) / len(values))

        # Log learning rate (use current progress for scheduled LR)
        if hasattr(self.model, "learning_rate"):
            lr = self.model.learning_rate
            if callable(lr):
                progress = getattr(self.model, "_current_progress_remaining", 1.0)
                lr = lr(progress)
            metrics["train/learning_rate"] = lr

        wandb.log(metrics, step=self.num_timesteps)

        # Record video periodically
        if self.video_env is not None and (self.num_timesteps - self._last_video_step) >= self.video_freq:
            self._last_video_step = self.num_timesteps
            self._record_video()

        return True

    def _record_video(self) -> None:
        """Record a single evaluation episode and log as a W&B video."""
        if self.video_env is None or not is_available() or wandb.run is None:
            return
        if np is None:
            return

        frames: list[Any] = []
        obs = self.video_env.reset()
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = self.video_env.step(action)

            # Collect rendered frame
            try:
                frame = self.video_env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                break

            if dones[0]:
                break

        if frames and np is not None:
            # Stack frames into (T, H, W, C) array and log
            video_array = np.array(frames)
            # wandb.Video expects (T, C, H, W) for numpy arrays
            if video_array.ndim == 4:
                video_array = np.transpose(video_array, (0, 3, 1, 2))
            wandb.log(
                {"eval/video": wandb.Video(video_array, fps=30)},
                step=self.num_timesteps,
            )
            logger.info(
                "Recorded evaluation video (%d frames) at step %d",
                len(frames),
                self.num_timesteps,
            )

    def _on_rollout_end(self) -> None:
        if not is_available() or wandb.run is None:
            return

        # Log rollout-level stats from the logger
        if hasattr(self.model, "logger") and self.model.logger is not None:
            name_to_value = getattr(self.model.logger, "name_to_value", {})
            rollout_metrics = {}
            for key, value in name_to_value.items():
                if isinstance(value, (int, float)):
                    safe_key = key.replace("/", "_")
                    rollout_metrics[f"rollout/{safe_key}"] = value
            if rollout_metrics:
                wandb.log(rollout_metrics, step=self.num_timesteps)


def log_eval_metrics(
    eval_results: dict[str, Any],
    stage: int,
    step: int | None = None,
):
    """Log evaluation metrics to W&B.

    Handles numeric metrics as scalar logs and termination_counts as
    individual per-reason metrics (for stacked area charts).

    Args:
        eval_results: Aggregated dict from ``LocomotionMetrics.aggregate_episodes``.
        stage: Current curriculum stage.
        step: Training step number for x-axis alignment.
    """
    if not is_available() or wandb.run is None:
        return

    metrics: dict[str, Any] = {"eval/stage": float(stage)}

    for key, value in eval_results.items():
        if key == "termination_counts" and isinstance(value, dict):
            # Log each termination reason as a separate metric for stacked area
            for reason, count in value.items():
                metrics[f"eval/termination/{reason}"] = float(count)
        elif isinstance(value, (int, float)):
            metrics[f"eval/{key}"] = float(value)

    wandb.log(metrics, step=step)


def setup_wandb_metrics(stage: int) -> None:
    """Configure W&B metric display properties.

    Defines metric groupings and x-axis relationships so W&B
    auto-creates well-organised panels. Call once after ``init_wandb``.

    Args:
        stage: Current curriculum stage number.
    """
    if not is_available() or wandb.run is None:
        return

    # All eval metrics use training timesteps as x-axis
    wandb.define_metric("eval/*", step_metric="train/timesteps")
    wandb.define_metric("reward/*", step_metric="train/timesteps")

    # ---- Master panels (all stages) ----
    # Termination reasons: individual reason metrics for stacked area
    wandb.define_metric("eval/termination/*", step_metric="train/timesteps")
    # Cost of transport
    wandb.define_metric("eval/mean_cost_of_transport", step_metric="train/timesteps")

    # ---- Stage 1: Balance ----
    wandb.define_metric("reward/pelvis_height", step_metric="train/timesteps")
    # reward_alive, reward_posture, reward_nosedive already under reward/*

    # ---- Stage 2: Locomotion ----
    wandb.define_metric("eval/mean_gait_symmetry", step_metric="train/timesteps")
    wandb.define_metric("eval/mean_stride_frequency", step_metric="train/timesteps")
    wandb.define_metric("reward/heading_alignment", step_metric="train/timesteps")

    # ---- Stage 3: Strike / Bite / Food ----
    wandb.define_metric("reward/prey_distance", step_metric="train/timesteps")
    wandb.define_metric("eval/mean_min_prey_distance", step_metric="train/timesteps")
    wandb.define_metric("reward/strike_success", step_metric="train/timesteps")
    wandb.define_metric("reward/bite_success", step_metric="train/timesteps")
    wandb.define_metric("reward/food_reached", step_metric="train/timesteps")
    wandb.define_metric("eval/mean_success_rate", step_metric="train/timesteps")
    wandb.define_metric("eval/mean_heading_alignment", step_metric="train/timesteps")


def create_wandb_dashboard(
    stage: int,
    entity: str | None = None,
    project: str = "mesozoic-labs",
) -> None:
    """Create a W&B workspace with the training curve panel layout.

    Uses the ``wandb-workspaces`` library to programmatically build a
    workspace with 4 sections (master + one per stage), each containing
    2 panels arranged in a 2-column grid. Falls back to saving the
    panel configuration as run metadata if ``wandb-workspaces`` is not
    installed.

    Args:
        stage: Current curriculum stage number.
        entity: W&B entity (team/user). Auto-detected from run if None.
        project: W&B project name.
    """
    if not is_available() or wandb.run is None:
        return

    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError:
        logger.info(
            "wandb-workspaces not installed. Install with: pip install wandb-workspaces. "
            "Saving panel config as run metadata instead."
        )
        _save_dashboard_config_fallback(stage)
        return

    if entity is None:
        entity = wandb.run.entity

    x_axis = "train/timesteps"

    # Known termination reasons across all species
    termination_keys = [
        "eval/termination/fallen",
        "eval/termination/excessive_tilt",
        "eval/termination/nosedive",
        "eval/termination/body_contact",
        "eval/termination/tail_contact",
        "eval/termination/too_high",
        "eval/termination/truncated",
    ]

    sections = [
        # ---- Master (all stages) ----
        ws.Section(
            name="Master — All Stages",
            is_open=True,
            panels=[
                wr.LinePlot(
                    title="Termination Reasons",
                    x=x_axis,
                    y=termination_keys,
                    plot_type="stacked-area",
                    title_x="Training Steps",
                    title_y="Count per Eval",
                ),
                wr.LinePlot(
                    title="Cost of Transport",
                    x=x_axis,
                    y=["eval/mean_cost_of_transport"],
                    title_x="Training Steps",
                    title_y="CoT (lower = more efficient)",
                ),
            ],
        ),
        # ---- Stage 1: Balance ----
        ws.Section(
            name="Stage 1 — Balance",
            is_open=(stage == 1),
            panels=[
                wr.LinePlot(
                    title="Pelvis Height",
                    x=x_axis,
                    y=["reward/pelvis_height"],
                    title_x="Training Steps",
                    title_y="Height (m)",
                ),
                wr.LinePlot(
                    title="Reward Decomposition (alive / posture / nosedive)",
                    x=x_axis,
                    y=[
                        "reward/reward_alive",
                        "reward/reward_posture",
                        "reward/reward_nosedive",
                    ],
                    plot_type="stacked-area",
                    title_x="Training Steps",
                    title_y="Reward Component",
                ),
            ],
        ),
        # ---- Stage 2: Locomotion ----
        ws.Section(
            name="Stage 2 — Locomotion",
            is_open=(stage == 2),
            panels=[
                wr.LinePlot(
                    title="Gait Symmetry + Stride Frequency",
                    x=x_axis,
                    y=[
                        "eval/mean_gait_symmetry",
                        "eval/mean_stride_frequency",
                    ],
                    title_x="Training Steps",
                    title_y="Value",
                ),
                wr.LinePlot(
                    title="Heading Alignment",
                    x=x_axis,
                    y=["reward/heading_alignment"],
                    title_x="Training Steps",
                    title_y="Alignment (-1 to +1)",
                ),
            ],
        ),
        # ---- Stage 3: Strike / Bite / Food ----
        ws.Section(
            name="Stage 3 — Strike / Bite / Food",
            is_open=(stage == 3),
            panels=[
                wr.LinePlot(
                    title="Prey / Food Distance (mean + min)",
                    x=x_axis,
                    y=[
                        "reward/prey_distance",
                        "reward/head_food_distance",
                        "eval/mean_min_prey_distance",
                    ],
                    title_x="Training Steps",
                    title_y="Distance (m)",
                ),
                wr.LinePlot(
                    title="Strike / Bite / Food Success Rate",
                    x=x_axis,
                    y=[
                        "reward/strike_success",
                        "reward/bite_success",
                        "reward/food_reached",
                        "eval/mean_success_rate",
                    ],
                    title_x="Training Steps",
                    title_y="Success (0 or 1)",
                ),
            ],
        ),
    ]

    try:
        workspace = ws.Workspace(
            name=f"Locomotion Dashboard — Stage {stage}",
            entity=entity,
            project=project,
            sections=sections,
        )
        workspace.save()
        logger.info("W&B workspace dashboard created for stage %d", stage)
    except Exception as e:
        logger.warning("Failed to create W&B workspace: %s. Saving config as fallback.", e)
        _save_dashboard_config_fallback(stage)


def _save_dashboard_config_fallback(stage: int) -> None:
    """Store dashboard panel config as run metadata (fallback)."""
    panel_config = {
        "master": [
            {"title": "Termination Reasons", "metrics": ["eval/termination/*"], "type": "stacked_area"},
            {"title": "Cost of Transport", "metrics": ["eval/mean_cost_of_transport"], "type": "line"},
        ],
        "stage1_balance": [
            {"title": "Pelvis Height", "metrics": ["reward/pelvis_height"], "type": "line"},
            {
                "title": "Reward Decomposition",
                "metrics": ["reward/reward_alive", "reward/reward_posture", "reward/reward_nosedive"],
                "type": "stacked_area",
            },
        ],
        "stage2_locomotion": [
            {
                "title": "Gait Symmetry + Stride Frequency",
                "metrics": ["eval/mean_gait_symmetry", "eval/mean_stride_frequency"],
                "type": "dual_axis",
            },
            {"title": "Heading Alignment", "metrics": ["reward/heading_alignment"], "type": "line"},
        ],
        "stage3_strike": [
            {
                "title": "Prey / Food Distance (mean + min)",
                "metrics": ["reward/prey_distance", "reward/head_food_distance", "eval/mean_min_prey_distance"],
                "type": "multi_line",
            },
            {
                "title": "Strike / Bite / Food Success Rate",
                "metrics": [
                    "reward/strike_success",
                    "reward/bite_success",
                    "reward/food_reached",
                    "eval/mean_success_rate",
                ],
                "type": "multi_line",
            },
        ],
    }
    wandb.config.update({"dashboard_panels": panel_config}, allow_val_change=True)
    logger.info("W&B dashboard panel config saved as run metadata for stage %d", stage)
