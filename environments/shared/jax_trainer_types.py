"""Public types for the JAX/PPO trainer's hook-based library API.

Extracted from :mod:`environments.shared.jax_trainer` so that hook
implementations (see :mod:`environments.shared.jax_hooks`) can import
these primitives without pulling in the full JIT-compiled trainer —
useful for tests and for tools that only build hooks.

Also home to the two loop helpers BOTH PPO loops share
(:func:`_bootstrap_truncations`, :func:`_build_batched_value`): the
notebook-path :mod:`environments.shared.jax_train_fn` and the library
:class:`~environments.shared.jax_trainer.JaxTrainer` import them from here,
so neither loop module has to import the other.  Both defer any JAX import
to call time, so this module stays import-light.

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


# ---------------------------------------------------------------------------
# Loop helpers shared by both trainers (jax_train_fn.train and JaxTrainer)
# ---------------------------------------------------------------------------


def _bootstrap_truncations(rewards, gae_dones, full_dones, final_values, gamma: float):
    """SB3-style per-step time-limit bootstrap for GAE.

    Both trainers collect ``final_obs`` (the pre-auto-reset obs) for EVERY
    step but used it only for the rollout's last step.  A truncation at a
    non-final rollout position therefore bootstrapped
    ``gamma * V(post-auto-reset obs)`` — the next episode's first state —
    and, because the GAE mask was termination-only, let the lambda carry
    leak the next episode's advantages into the ended one.  With
    synchronized resets, ``max_episode_steps=1000`` and ``rollout_len=64``
    (1000 mod 64 = 40) the whole fleet truncates mid-rollout at once as soon
    as a policy reaches the horizon — precisely the regime the stance gate
    selects for.

    Where a step truncated without terminating, fold ``gamma * V(final_obs)``
    into that step's reward and mask it as done, so the step bootstraps its
    true successor value and the carry stops at the episode boundary.
    Terminations are unchanged: no bootstrap term, and they were already
    masked.

    Args:
        rewards: ``(T, N)`` per-step rewards.
        gae_dones: ``(T, N)`` float mask, 1 where the step TERMINATED.
        full_dones: ``(T, N)`` float mask, 1 where the step terminated OR
            truncated (``gae_dones <= full_dones`` elementwise).
        final_values: ``(T, N)`` value of each step's pre-reset ``final_obs``
            under the rollout policy and rollout-time obs stats.
        gamma: Discount factor.

    Returns:
        ``(rewards_with_bootstrap, gae_mask)`` — feed both to ``compute_gae``
        in place of the raw rewards and the termination-only mask.
    """
    truncated_only = full_dones - gae_dones
    return rewards + gamma * final_values * truncated_only, full_dones


def _build_batched_value(network):
    """JIT value head over a flat ``(B, obs_dim)`` batch of normalized obs."""
    import jax

    @jax.jit
    def batched_value(params, obs_batch):
        _, _, values = network.apply(params, obs_batch)
        return values

    return batched_value
