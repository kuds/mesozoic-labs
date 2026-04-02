"""High-level JAX/MJX training entry point for Mesozoic Labs.

This module is the JAX equivalent of ``train_base.py``.  It loads the
species + stage configuration from TOML files, creates an MJX batched
environment, runs a JIT-compiled PPO training loop via
:class:`~jax_trainer.JaxTrainer`, and optionally logs to Weights & Biases.

Usage::

    python -m environments.shared.jax_training --species trex --stage 1

Or from Python::

    from environments.shared.jax_training import train_jax
    params, history = train_jax("trex", stage=1, num_envs=2048)
"""

from __future__ import annotations

import argparse
import logging
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
    learning_rate_end: float | None = None,
    clip_range: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    n_epochs: int = 10,
    n_minibatches: int = 4,
    target_kl: float | None = 0.05,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train a species with JAX/MJX PPO.

    Loads config from TOML, creates MJX env, runs PPO training loop
    via :class:`~jax_trainer.JaxTrainer`.

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
        learning_rate_end: Final LR for linear decay (None = constant LR).
        clip_range: PPO clip range.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        ent_coef: Entropy coefficient.
        max_grad_norm: Max gradient norm for clipping.
        n_epochs: PPO gradient epochs per update.
        n_minibatches: Number of minibatches per epoch.
        target_kl: Early-stop PPO epochs when approx KL exceeds this
            threshold.  Set to ``None`` to disable.
        env_kwargs: Stage-specific environment kwargs (reward weights etc.)
            from the TOML ``[env]`` section, overlaid onto species defaults.

    Returns:
        ``(params, eval_metrics)`` tuple.
    """
    check_jax()

    from .jax_hooks import CheckpointHook, LoggingHook
    from .jax_ppo import PPOConfig, make_actor_critic, make_optimizer
    from .jax_trainer import JaxTrainer
    from .mjx_env import MJXDinoEnv

    # Import species config to trigger registration
    _import_species_config(species)

    _logger.info("species=%s stage=%d num_envs=%d", species, stage, num_envs)

    # Create environment with TOML-derived reward weights
    env = MJXDinoEnv(species, stage=stage, num_envs=num_envs, env_kwargs=env_kwargs)

    # PPO config
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        learning_rate_end=learning_rate_end,
        clip_range=clip_range,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        n_minibatches=n_minibatches,
        target_kl=target_kl,
        total_updates=num_updates,
    )

    # Create network and optimizer
    network = make_actor_critic(env.action_dim)
    optimizer = make_optimizer(ppo_config)

    # Assemble hooks
    hooks: list[Any] = [LoggingHook(interval=10, num_updates=num_updates)]

    if checkpoint_dir:
        hooks.append(
            CheckpointHook(
                directory=checkpoint_dir,
                prefix=f"{species}_s{stage}",
                interval=50,
            )
        )

    # Create trainer and run
    trainer = JaxTrainer(
        env=env,
        network=network,
        optimizer=optimizer,
        ppo_config=ppo_config,
        num_envs=num_envs,
        rollout_len=rollout_len,
        hooks=hooks,
    )

    return trainer.train(
        num_updates=num_updates,
        seed=seed,
        init_params=init_params,
    )


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
        # Load TOML config for single-stage runs so that reward weights
        # and JAX-specific hyperparameters are applied correctly.
        from .config import load_stage_config

        stage_config = load_stage_config(args.species, args.stage)
        jax_kwargs = stage_config.get("jax_kwargs", {})
        env_kwargs = stage_config.get("env_kwargs", {})

        # Override fall_penalty / noise from [jax] section
        # Use direct assignment — setdefault is a no-op when [env] already
        # defines the key, which silently ignores the JAX-specific override.
        if "fall_penalty" in jax_kwargs:
            env_kwargs["fall_penalty"] = jax_kwargs["fall_penalty"]
        for noise_key in ("reset_noise_scale", "init_qpos_noise", "init_yaw_noise"):
            if noise_key in jax_kwargs:
                env_kwargs[noise_key] = jax_kwargs[noise_key]

        train_jax(
            species=args.species,
            stage=args.stage,
            num_envs=args.num_envs,
            num_updates=jax_kwargs.get("num_updates", args.num_updates),
            rollout_len=jax_kwargs.get("rollout_len", args.rollout_len),
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            wandb_project=args.wandb_project,
            learning_rate=jax_kwargs.get("learning_rate", args.learning_rate),
            learning_rate_end=jax_kwargs.get("learning_rate_end"),
            gamma=jax_kwargs.get("gamma", 0.99),
            gae_lambda=jax_kwargs.get("gae_lambda", 0.95),
            clip_range=jax_kwargs.get("clip_range", 0.2),
            ent_coef=jax_kwargs.get("ent_coef", 0.01),
            max_grad_norm=jax_kwargs.get("max_grad_norm", 0.5),
            n_epochs=jax_kwargs.get("ppo_epochs", 10),
            target_kl=jax_kwargs.get("target_kl", 0.05),
            env_kwargs=env_kwargs,
        )


if __name__ == "__main__":
    main()
