"""High-level JAX/MJX training entry point for Mesozoic Labs.

This module is the JAX equivalent of ``train_base.py``.  It loads the
species + stage configuration from TOML files, creates an MJX batched
environment, runs a JIT-compiled PPO training loop, and optionally logs
to Weights & Biases.

Usage::

    python -m environments.shared.jax_training --species trex --stage 1

Or from Python::

    from environments.shared.jax_training import train_jax
    params, history = train_jax("trex", stage=1, num_envs=2048)
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from .mjx_utils import check_jax


def train_jax(
    species: str,
    stage: int = 1,
    num_envs: int = 2048,
    num_updates: int = 500,
    rollout_len: int = 64,
    seed: int = 42,
    checkpoint_dir: str | None = None,
    wandb_project: str | None = None,
    init_params: Any | None = None,
    learning_rate: float = 3e-4,
    clip_range: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Any, dict[str, float]]:
    """Train a species with JAX/MJX PPO.

    Loads config from TOML, creates MJX env, runs PPO training loop.

    Args:
        species: One of ``"trex"``, ``"velociraptor"``, ``"brachiosaurus"``.
        stage: Curriculum stage (1, 2, or 3).
        num_envs: Number of parallel environments.
        num_updates: Number of PPO update iterations.
        rollout_len: Number of steps per rollout.
        seed: Random seed.
        checkpoint_dir: Optional directory for saving checkpoints.
        wandb_project: Optional W&B project name for logging.
        init_params: Optional initial network parameters (for curriculum).
        learning_rate: PPO learning rate.
        clip_range: PPO clip range.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(params, eval_metrics)`` tuple.
    """
    check_jax()

    import jax
    import jax.numpy as jnp

    from .jax_normalization import RunningMeanStd, normalize_obs, update_running_stats
    from .jax_ppo import PPOConfig, compute_gae, make_actor_critic, make_optimizer, ppo_update, sample_action
    from .mjx_env import MJXDinoEnv

    # Import species config to trigger registration
    _import_species_config(species)

    print(f"[JAX Training] species={species} stage={stage} num_envs={num_envs}")
    print(f"[JAX Training] device: {jax.devices()[0]}")

    # Create environment
    env = MJXDinoEnv(species, stage=stage, num_envs=num_envs)

    # PPO config
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        clip_range=clip_range,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # Create network and optimizer
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    network = make_actor_critic(env.action_dim)
    dummy_obs = jnp.zeros(env.mj_model.nq - 7 + env.mj_model.nv - 6 + 17)  # approximate obs dim
    params = network.init(init_rng, dummy_obs) if init_params is None else init_params
    optimizer = make_optimizer(ppo_config)
    opt_state = optimizer.init(params)

    # Running observation normalisation
    obs_dim = dummy_obs.shape[0]
    obs_stats = RunningMeanStd.create(obs_dim)

    # Reset environments
    rng, reset_rng = jax.random.split(rng)
    states = env.reset(reset_rng)

    # Training loop
    total_steps = 0
    history: list[dict[str, float]] = []

    t0 = time.time()
    for update in range(num_updates):
        # Collect rollout
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []

        for step in range(rollout_len):
            rng, action_rng, step_rng = jax.random.split(rng, 3)

            obs = normalize_obs(states.obs, obs_stats)
            action, log_prob, value = jax.vmap(
                lambda o, r: sample_action(params, network, o, r),
            )(obs, jax.random.split(action_rng, num_envs))

            new_states, rewards, terminated, truncated = env.step(states, action, step_rng)
            dones = terminated | truncated

            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_values.append(value)
            rollout_rewards.append(rewards)
            rollout_dones.append(dones.astype(jnp.float32))

            # Update obs stats
            obs_stats = update_running_stats(obs_stats, states.obs)

            states = new_states
            total_steps += num_envs

        # Bootstrap value
        final_obs = normalize_obs(states.obs, obs_stats)
        _, _, bootstrap_value = jax.vmap(
            lambda o, r: sample_action(params, network, o, r),
        )(final_obs, jax.random.split(rng, num_envs))

        # Stack rollout
        rollout_rewards_arr = jnp.stack(rollout_rewards)
        rollout_values_arr = jnp.concatenate([jnp.stack(rollout_values), bootstrap_value[None]], axis=0)
        rollout_dones_arr = jnp.stack(rollout_dones)

        # Compute GAE
        advantages, returns = compute_gae(
            rollout_rewards_arr, rollout_values_arr, rollout_dones_arr, ppo_config.gamma, ppo_config.gae_lambda
        )

        # Flatten rollout for minibatch updates
        batch = {
            "obs": jnp.concatenate([jnp.stack(rollout_obs).reshape(-1, obs_dim)]),
            "action": jnp.concatenate([jnp.stack(rollout_actions).reshape(-1, env.action_dim)]),
            "old_log_prob": jnp.concatenate([jnp.stack(rollout_log_probs).reshape(-1)]),
            "advantage": advantages.reshape(-1),
            "return_": returns.reshape(-1),
        }

        # PPO update epochs
        for epoch in range(ppo_config.n_epochs):
            params, opt_state, loss_info = ppo_update(params, opt_state, optimizer, network, batch, ppo_config)

        # Logging
        mean_reward = float(jnp.mean(rollout_rewards_arr))
        elapsed = time.time() - t0
        fps = total_steps / elapsed if elapsed > 0 else 0

        step_info = {
            "update": update,
            "total_steps": total_steps,
            "mean_reward": mean_reward,
            "fps": fps,
        }
        history.append(step_info)

        if update % 10 == 0:
            print(f"[Update {update:4d}/{num_updates}] steps={total_steps:,} reward={mean_reward:.2f} fps={fps:.0f}")

    elapsed = time.time() - t0
    print(f"[JAX Training] Done. {total_steps:,} steps in {elapsed:.1f}s ({total_steps / elapsed:.0f} fps)")

    eval_metrics = {
        "mean_reward": float(jnp.mean(jnp.array([h["mean_reward"] for h in history[-10:]]))),
        "total_steps": total_steps,
    }

    return params, eval_metrics


def _import_species_config(species: str) -> None:
    """Import the species MJX config module to trigger registration."""
    import importlib

    module_map = {
        "trex": "environments.trex.mjx_config",
        "velociraptor": "environments.velociraptor.mjx_config",
        "brachiosaurus": "environments.brachiosaurus.mjx_config",
    }
    module_name = module_map.get(species)
    if module_name:
        importlib.import_module(module_name)


def main():
    """CLI entry point for JAX/MJX training."""
    parser = argparse.ArgumentParser(description="Train dinosaur locomotion with JAX/MJX PPO")
    parser.add_argument("--species", type=str, required=True, choices=["trex", "velociraptor", "brachiosaurus"])
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--rollout-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--curriculum", action="store_true", help="Run full 3-stage curriculum")

    args = parser.parse_args()

    if args.curriculum:
        from .jax_curriculum import run_curriculum

        results = run_curriculum(
            species=args.species,
            train_fn=train_jax,
            num_envs=args.num_envs,
            num_updates=args.num_updates,
            rollout_len=args.rollout_len,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        for stage, (params, metrics) in results.items():
            print(f"Stage {stage}: reward={metrics['mean_reward']:.2f}")
    else:
        train_jax(
            species=args.species,
            stage=args.stage,
            num_envs=args.num_envs,
            num_updates=args.num_updates,
            rollout_len=args.rollout_len,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            wandb_project=args.wandb_project,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
