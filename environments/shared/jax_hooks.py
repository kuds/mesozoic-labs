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
from typing import Any

from .jax_checkpoint import CheckpointManager
from .jax_trainer import StopTraining, TrainerState
from .jax_training_utils import StabilityMonitor

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
    ):
        self.interval = interval
        self._manager = CheckpointManager(
            directory=directory, prefix=prefix, max_keep=max_keep,
        )

    def on_update_end(self, state: TrainerState, metrics: dict[str, float]) -> None:
        if state.update % self.interval != 0:
            return
        self._manager.save(
            params=state.params,
            update=state.update,
            obs_rms=state.obs_stats,
        )

    def on_train_end(self, state: TrainerState) -> None:
        self._manager.save(
            params=state.params,
            update=state.update,
            obs_rms=state.obs_stats,
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
