"""JAX-compatible curriculum manager.

Mirrors the stage-gating logic from ``curriculum.py`` but works with
the JAX training path.  Stage configs are loaded from the same TOML
files used by the SB3 path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .config import load_stage_config

_logger = logging.getLogger(__name__)


def check_stage_gate(
    eval_metrics: dict[str, float],
    stage_config: dict[str, Any],
) -> bool:
    """Check if curriculum gate thresholds are met.

    Args:
        eval_metrics: Evaluation metrics dict (keys like ``"mean_reward"``).
        stage_config: Stage configuration dict from TOML (must contain
            ``[curriculum]`` section with ``min_avg_reward``).

    Returns:
        ``True`` if the gate is passed and training should advance.
    """
    curriculum = stage_config.get("curriculum", {})
    min_reward = curriculum.get("min_avg_reward", float("inf"))
    return bool(eval_metrics.get("mean_reward", 0.0) >= min_reward)


def run_curriculum(
    species: str,
    train_fn: Callable,
    stages: tuple[int, ...] = (1, 2, 3),
    **train_kwargs: Any,
) -> dict[int, Any]:
    """Run full curriculum: train each stage, evaluate gate, advance.

    Args:
        species: Species name (``"trex"``, ``"velociraptor"``, ``"brachiosaurus"``).
        train_fn: Training function with signature
            ``train_fn(species, stage, **kwargs) -> (params, eval_metrics)``.
        stages: Tuple of stage numbers to train through.
        **train_kwargs: Extra keyword arguments forwarded to ``train_fn``.

    Returns:
        Dict mapping stage number to final ``(params, eval_metrics)``.
    """
    results: dict[int, Any] = {}

    params = None
    for stage in stages:
        stage_config = load_stage_config(species, stage)

        # Pass previous stage params as init for next stage
        if params is not None:
            train_kwargs["init_params"] = params

        # Merge TOML [jax] and [env] sections into train_kwargs so that
        # stage-specific hyperparameters and reward weights reach train_jax.
        stage_train_kwargs = dict(train_kwargs)
        jax_kwargs = stage_config.get("jax_kwargs", {})
        env_kwargs = stage_config.get("env_kwargs", {})

        # Map TOML [jax] keys to train_jax parameter names
        _JAX_KEY_MAP = {
            "num_envs": "num_envs",
            "rollout_len": "rollout_len",
            "num_updates": "num_updates",
            "learning_rate": "learning_rate",
            "learning_rate_end": "learning_rate_end",
            "max_grad_norm": "max_grad_norm",
            "gamma": "gamma",
            "gae_lambda": "gae_lambda",
            "clip_range": "clip_range",
            "ent_coef": "ent_coef",
            "ppo_epochs": "n_epochs",
            "target_kl": "target_kl",
        }
        for toml_key, param_name in _JAX_KEY_MAP.items():
            if toml_key in jax_kwargs and param_name not in stage_train_kwargs:
                stage_train_kwargs[param_name] = jax_kwargs[toml_key]

        # Always pass env_kwargs so reward weights reach the MJX env
        stage_train_kwargs["env_kwargs"] = env_kwargs

        # Override fall_penalty from [jax] section if specified
        # Use direct assignment — setdefault is a no-op when [env] already
        # defines the key, which silently ignores the JAX-specific override.
        if "fall_penalty" in jax_kwargs:
            env_kwargs["fall_penalty"] = jax_kwargs["fall_penalty"]

        # Carry over reset noise settings from [jax] section
        for noise_key in ("reset_noise_scale", "init_qpos_noise", "init_yaw_noise"):
            if noise_key in jax_kwargs:
                env_kwargs[noise_key] = jax_kwargs[noise_key]

        params, eval_metrics = train_fn(species=species, stage=stage, **stage_train_kwargs)
        results[stage] = (params, eval_metrics)

        # Check gate (skip for last stage)
        if stage != stages[-1]:
            if not check_stage_gate(eval_metrics, stage_config):
                _logger.warning(
                    "Stage %d gate NOT passed (reward=%.1f). Stopping early.",
                    stage,
                    eval_metrics.get("mean_reward", 0.0),
                )
                break
            _logger.info("Stage %d gate passed. Advancing to stage %d.", stage, stage + 1)

    return results
