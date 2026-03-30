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
import logging
import time
from typing import Any

from .mjx_utils import check_jax

_logger = logging.getLogger(__name__)


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

    _logger.info("species=%s stage=%d num_envs=%d", species, stage, num_envs)
    _logger.info("device: %s", jax.devices()[0])

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

    # ---- JIT-compiled rollout collector (params & obs_stats as args, not closures) ----
    @jax.jit
    def _collect_rollout(states, rng, params, obs_stats_arg):
        def step_fn(carry, _):
            states, rng = carry
            rng, action_rng = jax.random.split(rng)

            obs = normalize_obs(states.obs, obs_stats_arg)
            action, log_prob, value = jax.vmap(
                sample_action,
                in_axes=(None, None, 0, 0),
            )(params, network, obs, jax.random.split(action_rng, num_envs))

            rng, step_rng = jax.random.split(rng)
            new_states, rewards, terminated, truncated = env.step(states, action, step_rng)
            dones = (terminated | truncated).astype(jnp.float32)

            # Return raw observations so the caller can update RunningMeanStd
            # correctly (normalised obs would corrupt the running statistics).
            return (new_states, rng), (states.obs, action, log_prob, value, rewards, dones)

        return jax.lax.scan(step_fn, (states, rng), None, length=rollout_len)

    # ---- JIT-compiled PPO updater ----
    @jax.jit
    def _scan_ppo_update(params, opt_state, batch, rng):
        def epoch_fn(carry, _):
            params, opt_state, rng = carry
            params, opt_state, loss_info = ppo_update(params, opt_state, optimizer, network, batch, ppo_config)
            return (params, opt_state, rng), loss_info

        (params, opt_state, _), _ = jax.lax.scan(epoch_fn, (params, opt_state, rng), None, length=ppo_config.n_epochs)
        return params, opt_state

    # Training loop
    total_steps = 0
    history: list[dict[str, float]] = []

    t0 = time.time()
    _logger.info("Compiling scan functions (first update only)...")
    for update in range(num_updates):
        # Collect rollout via jax.lax.scan (fully on-device, compiled once)
        rng, collect_rng = jax.random.split(rng)
        (states, _), rollout_data = _collect_rollout(states, collect_rng, params, obs_stats)
        rollout_obs, rollout_actions, rollout_log_probs, rollout_values, rollout_rewards, rollout_dones = rollout_data

        total_steps += num_envs * rollout_len

        # Update obs stats with raw (unnormalized) observations
        obs_stats = update_running_stats(obs_stats, rollout_obs.reshape(-1, obs_dim))

        # Normalize raw observations for PPO training
        rollout_obs_norm = normalize_obs(rollout_obs.reshape(-1, obs_dim), obs_stats)

        # Bootstrap value
        rng, bootstrap_rng = jax.random.split(rng)
        final_obs = normalize_obs(states.obs, obs_stats)
        _, _, bootstrap_value = jax.vmap(
            lambda o, r: sample_action(params, network, o, r),
        )(final_obs, jax.random.split(bootstrap_rng, num_envs))

        # Stack and compute GAE
        rollout_values_arr = jnp.concatenate([rollout_values, bootstrap_value[None]], axis=0)

        advantages, returns = compute_gae(
            rollout_rewards, rollout_values_arr, rollout_dones, ppo_config.gamma, ppo_config.gae_lambda
        )

        # Flatten rollout for minibatch updates
        batch = {
            "obs": rollout_obs_norm,
            "action": rollout_actions.reshape(-1, env.action_dim),
            "old_log_prob": rollout_log_probs.reshape(-1),
            "advantage": advantages.reshape(-1),
            "return_": returns.reshape(-1),
        }

        # PPO update epochs (scanned)
        rng, ppo_rng = jax.random.split(rng)
        params, opt_state = _scan_ppo_update(params, opt_state, batch, ppo_rng)

        # Logging
        mean_reward = float(jnp.mean(rollout_rewards))
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
            _logger.info(
                "Update %4d/%d | steps=%s reward=%.2f fps=%.0f",
                update,
                num_updates,
                f"{total_steps:,}",
                mean_reward,
                fps,
            )

    elapsed = time.time() - t0
    _logger.info("Done. %s steps in %.1fs (%.0f fps)", f"{total_steps:,}", elapsed, total_steps / elapsed)

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
            _logger.info("Stage %d: reward=%.2f", stage, metrics["mean_reward"])
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
