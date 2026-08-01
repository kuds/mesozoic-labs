"""Public types for the JAX/PPO trainer's hook-based library API.

Extracted from :mod:`environments.shared.jax_trainer` so that hook
implementations (see :mod:`environments.shared.jax_hooks`) can import
these primitives without pulling in the full JIT-compiled trainer —
useful for tests and for tools that only build hooks.

``jax_trainer`` re-exports everything here, so existing
``from environments.shared.jax_trainer import TrainerState`` imports
continue to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TrainerState:
    """All mutable state for a training run.

    Passed to hooks so they can inspect (but should not mutate) training
    progress.  The trainer itself updates state between steps.
    """

    params: Any
    opt_state: Any
    obs_stats: Any  # RunningMeanStd
    env_states: Any  # EnvState (batched)
    rng: Any  # jax PRNGKey
    update: int = 0
    total_steps: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    episode_return_history: list[float] = field(default_factory=list)
    # Tracked alongside the return so the curriculum gate can enforce
    # min_avg_episode_length.  Without it the JAX path could only check
    # min_avg_reward, silently ignoring half of the reward_and_length/v1 gate
    # that the SB3 path enforces in full.
    episode_length_history: list[float] = field(default_factory=list)
    t_rollout_cumulative: float = 0.0
    t_ppo_cumulative: float = 0.0


@runtime_checkable
class TrainingHook(Protocol):
    """Lifecycle hook for the JAX training loop.

    All methods are optional — implement only the ones you need.
    The default implementations are no-ops.
    """

    def on_train_start(self, state: TrainerState) -> None: ...
    def on_rollout_end(self, state: TrainerState, rollout_metrics: dict[str, float]) -> None: ...
    def on_update_end(self, state: TrainerState, update_metrics: dict[str, float]) -> None: ...
    def on_train_end(self, state: TrainerState) -> None: ...


class StopTraining(Exception):
    """Raise from a hook to halt training early."""

    def __init__(self, reason: str = ""):
        self.reason = reason
        super().__init__(reason)
