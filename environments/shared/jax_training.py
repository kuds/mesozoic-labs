"""High-level JAX/MJX training entry point for Mesozoic Labs.

This module is the JAX equivalent of ``train_base.py``.  It loads the
species + stage configuration from TOML files, creates an MJX batched
environment, runs a JIT-compiled PPO training loop via
:class:`~jax_trainer.JaxTrainer`, and optionally logs to Weights & Biases.

Usage::

    python -m environments.shared.jax_training --species trex --stage 1

Or from Python::

    from environments.shared.jax_training import train_jax
    params, eval_metrics, obs_stats = train_jax("trex", stage=1, num_envs=2048)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
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
    init_params: Any | None = None,
    init_obs_stats: Any | None = None,
    learning_rate: float = 3e-4,
    learning_rate_end: float | None = None,
    clip_range: float = 0.2,
    vf_clip_range: float | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    n_epochs: int = 10,
    n_minibatches: int = 4,
    minibatch_size: int | None = None,
    target_kl: float | None = 0.05,
    env_kwargs: dict[str, Any] | None = None,
    warmup_updates: int = 0,
    warmup_clip_range: float = 0.02,
    warmup_ent_coef: float = 0.02,
    ramp_updates: int = 0,
    ramp_start_fraction: float = 0.1,
) -> tuple[Any, dict[str, float], Any]:
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
        init_params: Optional initial network parameters (for curriculum).
        init_obs_stats: Optional observation RunningMeanStd to carry over
            (for curriculum — must match *init_params*' normalization).
        learning_rate: PPO learning rate.
        learning_rate_end: Final LR for linear decay (None = constant LR).
        clip_range: PPO clip range.
        vf_clip_range: Clip value function updates to prevent catastrophic
            value loss spikes.  Set to ``None`` to disable (default).
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        ent_coef: Entropy coefficient.
        vf_coef: Value-loss coefficient (TOML ``vf_coef``).
        max_grad_norm: Max gradient norm for clipping.
        n_epochs: PPO gradient epochs per update.
        n_minibatches: Number of minibatches per epoch.
        target_kl: Early-stop PPO epochs when approx KL exceeds this
            threshold.  Set to ``None`` to disable.
        minibatch_size: TOML-style minibatch size; when given it overrides
            *n_minibatches* via ``num_envs * rollout_len // minibatch_size``.
        env_kwargs: Stage-specific environment kwargs (reward weights etc.)
            from the TOML ``[env]`` section, overlaid onto species defaults.
        warmup_updates: Constrain policy updates for the first N updates
            (TOML ``warmup_updates``).  0 disables.
        warmup_clip_range: Clip range during warm-up.
        warmup_ent_coef: Entropy coefficient during warm-up.
        ramp_updates: Ramp forward_vel_weight scale over the first N
            updates (TOML ``ramp_updates``).  0 disables.
        ramp_start_fraction: Starting fraction of forward_vel_weight.

    Returns:
        ``(params, eval_metrics, obs_stats)`` tuple.  *obs_stats* is the
        final observation ``RunningMeanStd`` — feed it (with *params*) into
        the next curriculum stage so the policy keeps seeing inputs scaled
        the way it was trained.
    """
    check_jax()

    from .jax_hooks import BestModelHook, CheckpointHook, CSVLoggingHook, LoggingHook
    from .jax_ppo import PPOConfig, make_actor_critic, make_optimizer
    from .jax_trainer import JaxTrainer
    from .mjx_env import MJXDinoEnv

    # Import species config to trigger registration
    _import_species_config(species)

    _logger.info("species=%s stage=%d num_envs=%d", species, stage, num_envs)

    # Create environment with TOML-derived reward weights
    env = MJXDinoEnv(species, stage=stage, num_envs=num_envs, env_kwargs=env_kwargs)

    # TOML configs specify minibatch_size; PPOConfig wants a count.
    if minibatch_size is not None:
        n_minibatches = max(1, (num_envs * rollout_len) // int(minibatch_size))

    # PPO config
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        learning_rate_end=learning_rate_end,
        clip_range=clip_range,
        vf_clip_range=vf_clip_range,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        n_minibatches=n_minibatches,
        target_kl=target_kl,
        total_updates=num_updates,
    )

    # Create network and optimizer
    network = make_actor_critic(env.action_dim)
    optimizer = make_optimizer(ppo_config)

    # Assemble hooks.  When a checkpoint dir is given, the headless run also
    # produces the durable artifacts the notebook path gets: a per-update
    # training CSV, best-model tracking, and rotating + final checkpoints.
    hooks: list[Any] = [LoggingHook(interval=10, num_updates=num_updates)]

    best_hook: Any = None
    if checkpoint_dir:
        _ckpt_dir = Path(checkpoint_dir)
        hooks.append(
            CheckpointHook(
                directory=checkpoint_dir,
                prefix=f"{species}_s{stage}",
                interval=50,
            )
        )
        hooks.append(CSVLoggingHook(_ckpt_dir / f"{species}_s{stage}_training_log.csv"))
        best_hook = BestModelHook(metric_key="episode_return")
        hooks.append(best_hook)

    # Create trainer and run
    trainer = JaxTrainer(
        env=env,
        network=network,
        optimizer=optimizer,
        ppo_config=ppo_config,
        num_envs=num_envs,
        rollout_len=rollout_len,
        hooks=hooks,
        warmup_updates=warmup_updates,
        warmup_clip_range=warmup_clip_range,
        warmup_ent_coef=warmup_ent_coef,
        ramp_updates=ramp_updates,
        ramp_start_fraction=ramp_start_fraction,
    )

    params, eval_metrics, _state = trainer.train(
        num_updates=num_updates,
        seed=seed,
        init_params=init_params,
        init_obs_stats=init_obs_stats,
    )

    # Persist the best-return checkpoint alongside the rotating ones so the
    # strongest policy survives late-training regressions.
    if best_hook is not None and best_hook.best_params is not None:
        from .jax_checkpoint import save_checkpoint

        save_checkpoint(
            Path(checkpoint_dir) / f"{species}_s{stage}_best.pkl",
            params=best_hook.best_params,
            obs_rms=_state.obs_stats,
            update=best_hook.best_update,
            extra={"best_episode_return": best_hook.best_reward},
        )
        eval_metrics["best_episode_return"] = best_hook.best_reward

    return params, eval_metrics, _state.obs_stats


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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--curriculum", action="store_true", help="Run full 3-stage curriculum")

    args = parser.parse_args()

    if args.curriculum:
        from .jax_curriculum import run_curriculum

        curriculum_kwargs: dict[str, Any] = dict(
            num_envs=args.num_envs,
            num_updates=args.num_updates,
            rollout_len=args.rollout_len,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        # Stage-prefixed checkpoints/CSVs keep the stages apart in one dir.
        if args.checkpoint_dir:
            curriculum_kwargs["checkpoint_dir"] = args.checkpoint_dir

        results = run_curriculum(
            species=args.species,
            train_fn=train_jax,
            **curriculum_kwargs,
        )
        for stage, (params, metrics) in results.items():
            _logger.info(
                "Stage %d: episode return=%.2f (per-step reward=%.2f)",
                stage,
                metrics.get("mean_episode_return", 0.0),
                metrics.get("mean_reward", 0.0),
            )
    else:
        # Load TOML config for single-stage runs so that reward weights
        # and JAX-specific hyperparameters are applied correctly.
        from .config import load_stage_config

        stage_config = load_stage_config(args.species, args.stage)
        jax_kwargs = stage_config.get("jax_kwargs", {})
        env_kwargs = stage_config.get("env_kwargs", {})

        # Override fall_penalty / reset noise from [jax] section if specified.
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
            num_envs=jax_kwargs.get("num_envs", args.num_envs),
            num_updates=jax_kwargs.get("num_updates", args.num_updates),
            rollout_len=jax_kwargs.get("rollout_len", args.rollout_len),
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            learning_rate=jax_kwargs.get("learning_rate", args.learning_rate),
            learning_rate_end=jax_kwargs.get("learning_rate_end"),
            gamma=jax_kwargs.get("gamma", 0.99),
            gae_lambda=jax_kwargs.get("gae_lambda", 0.95),
            clip_range=jax_kwargs.get("clip_range", 0.2),
            vf_clip_range=jax_kwargs.get("vf_clip_range"),
            ent_coef=jax_kwargs.get("ent_coef", 0.01),
            vf_coef=jax_kwargs.get("vf_coef", 0.5),
            max_grad_norm=jax_kwargs.get("max_grad_norm", 0.5),
            n_epochs=jax_kwargs.get("ppo_epochs", 10),
            target_kl=jax_kwargs.get("target_kl", 0.05),
            # Tuned TOML keys the single-stage path previously dropped
            # (warmup/ramp were silently inert; minibatch_size fell back
            # to the default 4 minibatches).
            minibatch_size=jax_kwargs.get("minibatch_size"),
            warmup_updates=jax_kwargs.get("warmup_updates", 0),
            warmup_clip_range=jax_kwargs.get("warmup_clip_range", 0.02),
            warmup_ent_coef=jax_kwargs.get("warmup_ent_coef", 0.02),
            ramp_updates=jax_kwargs.get("ramp_updates", 0),
            ramp_start_fraction=jax_kwargs.get("ramp_start_fraction", 0.1),
            env_kwargs=env_kwargs,
        )


if __name__ == "__main__":
    main()
