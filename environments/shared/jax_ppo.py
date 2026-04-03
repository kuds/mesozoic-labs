"""Reusable PPO implementation in JAX (Flax + Optax).

Provides the actor-critic network, GAE computation, and PPO update step
as pure functions compatible with ``jax.jit``.  This module is the JAX
equivalent of what Stable-Baselines3 provides for the SB3 training path.

All JAX/Flax/Optax imports are lazy — this module can be safely imported
even when only the ``[train]`` (SB3) extras are installed.

Usage::

    from environments.shared.jax_ppo import ActorCritic, ppo_update, compute_gae

    network = ActorCritic(action_dim=21)
    params = network.init(rng, dummy_obs)
    ...
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .mjx_utils import check_jax


class PPOConfig(NamedTuple):
    """Hyperparameters for a PPO training run."""

    learning_rate: float = 3e-4
    learning_rate_end: float | None = None  # Final LR for linear decay (None = constant)
    clip_range: float = 0.2
    vf_clip_range: float | None = None  # Clip value function updates (None = no clipping)
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    n_minibatches: int = 4
    target_kl: float | None = 0.05  # Early-stop epochs when approx KL exceeds this
    total_updates: int = 500  # Total training updates (for LR schedule)


class Transition(NamedTuple):
    """Single-step transition stored during rollout collection."""

    obs: Any
    action: Any
    log_prob: Any
    value: Any
    reward: Any
    done: Any


def make_actor_critic(action_dim: int, hidden_dims: tuple[int, ...] = (512, 256)):
    """Create a Flax actor-critic network.

    Returns:
        A Flax ``nn.Module`` with ``__call__`` returning
        ``(action_mean, action_log_std, value)``.
    """
    check_jax()
    import flax.linen as nn
    import jax.numpy as jnp

    class ActorCritic(nn.Module):
        """Shared-backbone actor-critic for continuous control."""

        action_dim: int
        hidden_dims: tuple[int, ...] = (512, 256)

        @nn.compact
        def __call__(self, obs):
            # Shared backbone
            x = obs
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.tanh(x)

            # Actor head (Gaussian policy)
            action_mean = nn.Dense(self.action_dim)(x)
            action_log_std = self.param(
                "log_std",
                nn.initializers.zeros,
                (self.action_dim,),
            )

            # Critic head
            value = nn.Dense(1)(x)
            value = jnp.squeeze(value, axis=-1)

            return action_mean, action_log_std, value

    return ActorCritic(action_dim=action_dim, hidden_dims=hidden_dims)


def sample_action(params, network, obs, rng):
    """Sample an action from the policy and return (action, log_prob, value).

    Args:
        params: Network parameters.
        network: Flax ActorCritic module.
        obs: Observation array.
        rng: JAX PRNGKey.

    Returns:
        (action, log_prob, value) tuple.
    """
    check_jax()
    import jax
    import jax.numpy as jnp

    action_mean, action_log_std, value = network.apply(params, obs)
    action_std = jnp.exp(action_log_std)

    # Sample from Gaussian
    noise = jax.random.normal(rng, shape=action_mean.shape)
    raw_action = action_mean + action_std * noise

    # Log probability computed on the *unclipped* action so the Gaussian
    # PDF is correct (clipping squashes probability mass at the boundaries,
    # biasing the PPO ratio if log_prob is computed after clipping).
    log_prob = -0.5 * jnp.sum(
        jnp.square((raw_action - action_mean) / (action_std + 1e-8)) + 2.0 * action_log_std + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )

    # Clamp to [-1, 1] after computing log_prob
    action = jnp.clip(raw_action, -1.0, 1.0)

    return action, log_prob, value


def compute_gae(
    rewards,
    values,
    dones,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Array of shape ``(T, num_envs)``.
        values: Array of shape ``(T+1, num_envs)`` (includes bootstrap).
        dones: Array of shape ``(T, num_envs)``.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        (advantages, returns) tuple, each of shape ``(T, num_envs)``.
    """
    check_jax()
    import jax
    import jax.numpy as jnp

    T = rewards.shape[0]

    def body_fn(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return gae, gae

    # Scan backwards through time
    indices = jnp.arange(T - 1, -1, -1)
    _, advantages = jax.lax.scan(
        body_fn,
        jnp.zeros_like(values[0]),
        indices,
    )
    # Reverse to get chronological order
    advantages = advantages[::-1]
    returns = advantages + values[:T]

    return advantages, returns


def ppo_loss(params, network, batch, config: PPOConfig):
    """Compute PPO clipped surrogate loss.

    Args:
        params: Network parameters.
        network: Flax ActorCritic module.
        batch: Dict with keys ``obs``, ``action``, ``old_log_prob``,
            ``advantage``, ``return_``.
        config: PPO hyperparameters.

    Returns:
        (total_loss, info_dict) tuple.
    """
    check_jax()
    import jax.numpy as jnp

    action_mean, action_log_std, value = network.apply(params, batch["obs"])
    action_std = jnp.exp(action_log_std)

    # New log probability
    log_prob = -0.5 * jnp.sum(
        jnp.square((batch["action"] - action_mean) / (action_std + 1e-8))
        + 2.0 * action_log_std
        + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )

    # Policy loss (clipped surrogate)
    ratio = jnp.exp(log_prob - batch["old_log_prob"])
    advantage = batch["advantage"]
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    pg_loss1 = -advantage * ratio
    pg_loss2 = -advantage * jnp.clip(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    # Value loss (with optional clipping to prevent catastrophic value updates)
    if config.vf_clip_range is not None and "old_value" in batch:
        value_clipped = batch["old_value"] + jnp.clip(
            value - batch["old_value"],
            -config.vf_clip_range,
            config.vf_clip_range,
        )
        vf_loss1 = jnp.square(value - batch["return_"])
        vf_loss2 = jnp.square(value_clipped - batch["return_"])
        value_loss = 0.5 * jnp.mean(jnp.maximum(vf_loss1, vf_loss2))
    else:
        value_loss = 0.5 * jnp.mean(jnp.square(value - batch["return_"]))

    # Entropy bonus
    entropy = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * jnp.e * action_std**2), axis=-1)
    entropy_loss = -jnp.mean(entropy)

    total_loss = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss

    info = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": jnp.mean(entropy),
        "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio)),
        "clip_fraction": jnp.mean((jnp.abs(ratio - 1.0) > config.clip_range).astype(jnp.float32)),
        "mean_std": jnp.mean(action_std),
    }

    return total_loss, info


def make_optimizer(config: PPOConfig):
    """Create an Optax optimizer for PPO training.

    When ``config.learning_rate_end`` is set, the learning rate is
    linearly decayed from ``learning_rate`` to ``learning_rate_end``
    over ``config.total_updates * config.n_epochs`` gradient steps.

    Returns:
        An ``optax.GradientTransformation``.
    """
    check_jax()
    import optax

    if config.learning_rate_end is not None and config.learning_rate_end != config.learning_rate:
        total_steps = config.total_updates * config.n_epochs
        lr_schedule = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=config.learning_rate_end,
            transition_steps=total_steps,
        )
    else:
        lr_schedule = config.learning_rate

    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr_schedule),
    )


def ppo_update(params, opt_state, optimizer, network, batch, config: PPOConfig):
    """Perform one PPO update step.

    Args:
        params: Network parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        network: Flax ActorCritic module.
        batch: Training batch dict.
        config: PPO hyperparameters.

    Returns:
        (new_params, new_opt_state, loss_info) tuple.
    """
    check_jax()
    import jax
    import optax

    grad_fn = jax.grad(ppo_loss, has_aux=True)
    grads, info = grad_fn(params, network, batch, config)
    info["grad_norm"] = optax.global_norm(grads)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, info
