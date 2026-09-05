"""Built-in training hooks for the JAX training loop.

Provides ready-to-use :class:`~jax_trainer.TrainingHook` implementations
for common concerns: console logging, checkpointing, and stability
monitoring.

Usage::

    from environments.shared.jax_hooks import LoggingHook, CheckpointHook, StabilityHook

    trainer = JaxTrainer(
        env=env, network=network, optimizer=optimizer, ppo_config=config,
        hooks=[
            LoggingHook(interval=10),
            CheckpointHook(directory="checkpoints", interval=50),
            StabilityHook(),
        ],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .jax_checkpoint import CheckpointManager
from .jax_trainer import StopTraining, TrainerState
from .jax_training_utils import StabilityMonitor
from .plant_contract import PlantIdentity

_logger = logging.getLogger(__name__)


class LoggingHook:
    """Logs training metrics to the console at a fixed interval.

    Args:
        interval: Log every N updates.
        num_updates: Total number of updates (for progress display).
    """

    def __init__(self, interval: int = 10, num_updates: int = 0):
        self.interval = interval
        self.num_updates = num_updates

    def on_update_end(self, state: TrainerState, metrics: dict[str, float]) -> None:
        update = int(metrics.get("update", state.update))
        if update % self.interval != 0:
            return
        _logger.info(
            "Update %4d/%d | steps=%s reward=%.2f fps=%.0f",
            update,
            self.num_updates,
            f"{state.total_steps:,}",
            metrics.get("mean_reward", 0.0),
            metrics.get("fps", 0.0),
        )


class CheckpointHook:
    """Saves rotating checkpoints at a fixed interval.

    Args:
        directory: Directory for checkpoint files.
        prefix: Filename prefix.
        interval: Save every N updates.
        max_keep: Maximum checkpoints to retain.
    """

    def __init__(
        self,
        directory: str,
        prefix: str = "checkpoint",
        interval: int = 50,
        max_keep: int = 5,
        plant_identity: PlantIdentity | Mapping[str, Any] | None = None,
    ):
        self.interval = interval
        self._manager = CheckpointManager(
            directory=directory,
            prefix=prefix,
            max_keep=max_keep,
            plant_identity=plant_identity,
        )

    # Checkpoints record the number of updates COMPLETED (state.update is the
    # index of the one just finished), the convention the functional trainer
    # and restore_train_state share: a resume starts at that count and trains
    # the budget's remainder.  Recording the index instead made every resume
    # repeat one update and re-number from one too low.
    def on_update_end(self, state: TrainerState, metrics: dict[str, float]) -> None:
        completed = state.update + 1
        if completed % self.interval != 0:
            return
        self._manager.save(
            params=state.params,
            update=completed,
            obs_rms=state.obs_stats,
            opt_state=state.opt_state,
        )

    def on_train_end(self, state: TrainerState) -> None:
        self._manager.save(
            params=state.params,
            update=state.update + 1,
            obs_rms=state.obs_stats,
            opt_state=state.opt_state,
        )

    @property
    def manager(self) -> CheckpointManager:
        """Access the underlying checkpoint manager."""
        return self._manager


class StabilityHook:
    """Monitors training stability and halts on divergence.

    Wraps :class:`~jax_training_utils.StabilityMonitor` as a training
    hook.  Raises :class:`~jax_trainer.StopTraining` when the monitor
    detects catastrophic instability.

    Args:
        kl_warn: Warn threshold for approx KL.
        kl_halt: Halt threshold for approx KL.
        grad_warn: Warn threshold for gradient norm.
        loss_warn: Warn threshold for total loss.
        max_warnings: Halt after this many consecutive warnings.
    """

    def __init__(self, **kwargs: Any):
        self._monitor = StabilityMonitor(**kwargs)

    def on_update_end(self, state: TrainerState, metrics: dict[str, float]) -> None:
        approx_kl = metrics.get("approx_kl", 0.0)
        grad_norm = metrics.get("grad_norm", 0.0)
        loss = metrics.get("total_loss", 0.0)

        should_halt, is_unstable, message = self._monitor.check(
            approx_kl=approx_kl,
            grad_norm=grad_norm,
            loss=loss,
            update=state.update,
        )

        if message:
            _logger.warning(message)

        if should_halt:
            raise StopTraining(message)

    @property
    def monitor(self) -> StabilityMonitor:
        """Access the underlying stability monitor."""
        return self._monitor


class CSVLoggingHook:
    """Writes per-update metrics to a CSV file.

    Provides a persistent training log compatible with notebook
    ``TrainingCSVLogger`` but driven by the hook system.

    Args:
        path: Path to the CSV file.
        extra_fields: Optional extra field names to include.
    """

    def __init__(self, path: str | Any, extra_fields: list[str] | None = None):
        from pathlib import Path as _Path

        from .jax_training_utils import TrainingCSVLogger

        self._logger = TrainingCSVLogger(_Path(path) if not isinstance(path, _Path) else path)
        self._extra_fields = extra_fields or []

    def on_update_end(self, state: TrainerState, metrics: dict[str, float]) -> None:
        row = {
            "update": int(metrics.get("update", state.update)),
            "reward_per_step": f"{metrics.get('mean_reward', 0.0):.4f}",
            "episode_return": f"{metrics.get('episode_return', float('nan')):.2f}",
            "episode_length": f"{metrics.get('episode_length', float('nan')):.1f}",
            "total_loss": f"{metrics.get('total_loss', 0.0):.4f}",
            "policy_loss": f"{metrics.get('policy_loss', 0.0):.4f}",
            "value_loss": f"{metrics.get('value_loss', 0.0):.4f}",
            "explained_variance": f"{metrics.get('explained_variance', float('nan')):.4f}",
            "entropy": f"{metrics.get('entropy', 0.0):.4f}",
            "approx_kl": f"{metrics.get('approx_kl', 0.0):.6f}",
            "clip_fraction": f"{metrics.get('clip_fraction', 0.0):.4f}",
            "grad_norm": f"{metrics.get('grad_norm', 0.0):.4f}",
            "mean_std": f"{metrics.get('mean_std', 0.0):.4f}",
            "steps": int(metrics.get("total_steps", state.total_steps)),
            "sps": f"{metrics.get('fps', 0.0):.0f}",
            "fall_rate": f"{metrics.get('fall_rate', 0.0):.4f}",
            "elapsed": f"{metrics.get('elapsed', 0.0):.1f}",
        }
        for field in self._extra_fields:
            if field in metrics:
                row[field] = metrics[field]
        self._logger.log(row)

    def on_train_end(self, state: TrainerState) -> None:
        self._logger.close()

    @property
    def path(self) -> Any:
        """Path to the CSV file."""
        return self._logger.path


class BestModelHook:
    """Tracks the best model parameters during training.

    Saves a copy of params whenever the tracked metric improves.
    Provides ``best_params``, ``best_reward``, and ``best_update``
    attributes after training.

    Args:
        metric_key: Metrics dict key to track (default: ``"mean_reward"``).
    """

    def __init__(self, metric_key: str = "mean_reward"):
        self.metric_key = metric_key
        self.best_reward: float = -float("inf")
        self.best_params: Any = None
        self.best_update: int = -1

    def on_update_end(self, state: TrainerState, metrics: dict[str, float]) -> None:
        value = metrics.get(self.metric_key, -float("inf"))
        if value > self.best_reward:
            self.best_reward = value
            try:
                import jax

                self.best_params = jax.device_get(state.params)
            except ImportError:
                self.best_params = state.params
            self.best_update = int(metrics.get("update", state.update))
