"""
Weights & Biases integration for experiment tracking.

Provides a Stable-Baselines3 callback that logs per-component rewards,
evaluation metrics, curriculum stage, and hyperparameters to W&B.

Usage:
    from environments.shared.wandb_integration import WandbCallback, init_wandb

    run = init_wandb(
        species="velociraptor",
        stage=1,
        config=stage_config,
    )

    wandb_callback = WandbCallback()

    model.learn(
        total_timesteps=500_000,
        callback=CallbackList([eval_callback, wandb_callback]),
    )

    run.finish()

Requires: pip install wandb
"""

import logging
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

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
    config: Dict[str, Any],
    project: str = "mesozoic-labs",
    tags: Optional[list] = None,
    notes: Optional[str] = None,
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
    return run


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class WandbCallback(BaseCallback):
    """Stable-Baselines3 callback that logs training metrics to W&B.

    Logs per-component reward breakdowns, episode statistics, and
    learning rate at each rollout end. Designed to work with the
    info dicts produced by ``BaseDinoEnv``.

    Args:
        log_freq: Log metrics every N training steps.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if not is_available() or wandb.run is None:
            return True

        if self.num_timesteps % self.log_freq != 0:
            return True

        metrics: Dict[str, Any] = {
            "train/timesteps": self.num_timesteps,
        }

        # Log info from the most recent environment steps
        if self.locals.get("infos"):
            info_keys = [
                "reward_forward", "reward_alive", "reward_energy",
                "reward_tail", "reward_strike", "reward_approach",
                "reward_total", "forward_vel", "prey_distance",
                "strike_success", "tail_instability",
                "reward_neck", "reward_food_reach",
                "reward_bite", "jaw_distance",
            ]
            for key in info_keys:
                values = [
                    info[key]
                    for info in self.locals["infos"]
                    if key in info
                ]
                if values:
                    metrics[f"reward/{key}"] = float(sum(values) / len(values))

        # Log learning rate
        if hasattr(self.model, "learning_rate"):
            lr = self.model.learning_rate
            if callable(lr):
                lr = lr(1.0)
            metrics["train/learning_rate"] = lr

        wandb.log(metrics, step=self.num_timesteps)
        return True

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
    eval_results: Dict[str, float],
    stage: int,
    step: Optional[int] = None,
):
    """Log evaluation metrics to W&B.

    Args:
        eval_results: Dict of metric name to value (from LocomotionMetrics.compute).
        stage: Current curriculum stage.
        step: Training step number for x-axis alignment.
    """
    if not is_available() or wandb.run is None:
        return

    metrics = {"eval/stage": stage}
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            metrics[f"eval/{key}"] = value

    wandb.log(metrics, step=step)
