"""Utility functions for the JAX/MJX training loop.

Contains helpers extracted from the Colab training notebook to keep the
notebook lean while making the logic reusable by the CLI training path.

- **Stability monitoring**: KL divergence, gradient norm, and loss watchdog
- **Episode tracking**: Reconstruct per-episode returns from per-step data
- **CSV logging**: Per-update metric writer
- **Profiling**: Per-operation timing within rollout steps
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stability monitoring
# ---------------------------------------------------------------------------


@dataclass
class StabilityMonitor:
    """Watches training metrics for instability and triggers warnings/halts.

    Tracks consecutive stability warnings and halts training when thresholds
    are exceeded.  Compatible with both notebook and CLI training loops.

    Args:
        kl_warn: Warn when approx KL exceeds this value.
        kl_halt: Halt training when approx KL exceeds this value.
        grad_warn: Warn when gradient norm exceeds this value.
        loss_warn: Warn when total loss exceeds this value.
        max_warnings: Halt after this many consecutive warnings.
    """

    kl_warn: float = 100.0
    kl_halt: float = 1e6
    grad_warn: float = 1000.0
    loss_warn: float = 100.0
    max_warnings: int = 5

    # Internal state
    _consecutive_warnings: int = field(default=0, init=False, repr=False)
    _total_warnings: int = field(default=0, init=False, repr=False)

    def check(
        self,
        approx_kl: float,
        grad_norm: float,
        loss: float,
        update: int,
    ) -> tuple[bool, bool, str]:
        """Check metrics and return (should_halt, is_unstable, message).

        Returns:
            ``(should_halt, is_unstable, message)`` tuple.  ``message`` is
            empty when everything is stable.
        """
        # KL explosion → immediate halt
        if approx_kl > self.kl_halt:
            msg = (
                f"HALTING: KL divergence exploded to {approx_kl:.2e} at update {update}. "
                f"Catastrophic policy collapse — resume from last checkpoint with lower LR."
            )
            return True, True, msg

        # Check individual thresholds
        reasons = []
        if approx_kl > self.kl_warn:
            reasons.append(f"kl={approx_kl:.2e}")
        if grad_norm > self.grad_warn:
            reasons.append(f"grad={grad_norm:.1f}")
        if loss > self.loss_warn:
            reasons.append(f"loss={loss:.1f}")

        if reasons:
            self._consecutive_warnings += 1
            self._total_warnings += 1
            msg = f"WARNING #{self._consecutive_warnings} at update {update}: {', '.join(reasons)}"

            if self._consecutive_warnings >= self.max_warnings:
                msg += (
                    f"\nHALTING: {self.max_warnings} consecutive stability warnings. "
                    f"Training is diverging — resume from last checkpoint with lower LR."
                )
                return True, True, msg

            return False, True, msg

        # Stable — reset consecutive counter
        self._consecutive_warnings = 0
        return False, False, ""

    @property
    def total_warnings(self) -> int:
        """Total number of stability warnings raised."""
        return self._total_warnings


# ---------------------------------------------------------------------------
# Episode return tracking
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStatsAccumulator:
    """Persistent state for tracking episode returns across rollouts.

    In JAX/MJX training the rollout length (e.g. 64 steps) is typically much
    shorter than the maximum episode length (e.g. 1000 steps).  Without
    carrying accumulator state across rollouts, only episodes that happen to
    complete within a single rollout are captured — biasing ``ep_ret`` and
    ``ep_length`` toward short (often failed) episodes.

    Create one instance before the training loop and pass it to
    :func:`compute_episode_stats` on every update.
    """

    run_ret: np.ndarray | None = None
    run_len: np.ndarray | None = None

    def _init_if_needed(self, num_envs: int) -> None:
        if self.run_ret is None:
            self.run_ret = np.zeros(num_envs)
            self.run_len = np.zeros(num_envs, dtype=np.int32)


def compute_episode_stats(
    rewards: np.ndarray,
    dones: np.ndarray,
    accumulator: EpisodeStatsAccumulator | None = None,
) -> tuple[list[float], list[int]]:
    """Reconstruct per-episode returns and lengths from rollout data.

    Walks through timesteps to accumulate returns per environment, then
    harvests completed episodes when ``dones`` is True.

    Args:
        rewards: Array of shape ``(T, num_envs)`` — per-step rewards.
        dones: Array of shape ``(T, num_envs)`` — done flags (>0.5 = done).
        accumulator: Optional persistent state that carries running returns
            and lengths across rollouts.  When provided, episodes spanning
            multiple rollouts are tracked correctly.  When ``None``, the
            function behaves as before (only episodes completing within this
            rollout are captured).

    Returns:
        ``(completed_returns, completed_lengths)`` — lists of floats/ints
        for each completed episode in the rollout.
    """
    T, num_envs = rewards.shape

    if accumulator is not None:
        accumulator._init_if_needed(num_envs)
        assert accumulator.run_ret is not None and accumulator.run_len is not None
        run_ret = accumulator.run_ret
        run_len = accumulator.run_len
    else:
        run_ret = np.zeros(num_envs)
        run_len = np.zeros(num_envs, dtype=np.int32)

    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    for t in range(T):
        run_ret += rewards[t]
        run_len += 1
        done_mask = dones[t] > 0.5
        if done_mask.any():
            completed_returns.extend(run_ret[done_mask].tolist())
            completed_lengths.extend(run_len[done_mask].tolist())
            run_ret[done_mask] = 0.0
            run_len[done_mask] = 0

    return completed_returns, completed_lengths


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

# Default field names for the per-update training CSV log.
DEFAULT_CSV_FIELDS: list[str] = [
    "update",
    "reward_per_step",
    "episode_return",
    "episode_length",
    "total_loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "mean_std",
    "steps",
    "sps",
    "fall_rate",
    "learning_rate",
    "elapsed",
    "t_rollout",
    "t_ppo",
]


class TrainingCSVLogger:
    """Per-update CSV logger for JAX training metrics.

    Opens a CSV file on creation, writes a header row, and provides a
    ``log()`` method to append one row per PPO update.  Call ``close()``
    when training is done (or use as a context manager).

    Args:
        path: Path to the CSV file.
        fields: Column names.  Defaults to :data:`DEFAULT_CSV_FIELDS`.
    """

    def __init__(
        self,
        path: str | Path,
        fields: list[str] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = fields or list(DEFAULT_CSV_FIELDS)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields)
        self._writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        """Write a single row.  Extra keys not in ``fields`` are ignored."""
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Per-operation profiling
# ---------------------------------------------------------------------------


@dataclass
class RolloutProfiler:
    """Lightweight per-operation profiler for rollout steps.

    Measures wall-clock time for individual operations (obs extraction,
    policy forward pass, physics step, reward computation, reset) within
    the rollout loop.  Only active when ``interval > 0`` and the current
    update is a profiling update.

    Args:
        interval: Profile every N updates.  0 disables profiling.
    """

    interval: int = 50
    _results: list[dict[str, float]] = field(default_factory=list, init=False)

    def should_profile(self, update: int) -> bool:
        """Return True if this update should be profiled."""
        return self.interval > 0 and update % self.interval == 0

    def record(self, update: int, op_times: dict[str, float]) -> None:
        """Store timing results for one profiled update."""
        self._results.append({"update": update, **op_times})

    def summary(self) -> str:
        """Return a human-readable summary of averaged per-op timings."""
        if not self._results:
            return ""
        ops = [k for k in self._results[0] if k != "update"]
        avg = {k: np.mean([r[k] for r in self._results]) for k in ops}
        total = sum(avg.values())
        lines = [f"Per-operation breakdown (averaged over {len(self._results)} profiled updates):"]
        for k, v in avg.items():
            pct = 100 * v / total if total > 0 else 0
            lines.append(f"  {k:8s}: {v:6.2f}s ({pct:4.1f}%)")
        lines.append(f"  {'total':8s}: {total:6.2f}s")
        return "\n".join(lines)

    @property
    def results(self) -> list[dict[str, float]]:
        """Raw per-update timing results."""
        return self._results
