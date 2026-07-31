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
from .curriculum.gate_schema import GateSchemaError, validate_gate_config

_logger = logging.getLogger(__name__)


def check_stage_gate(
    eval_metrics: dict[str, float],
    stage_config: dict[str, Any],
) -> bool:
    """Check if curriculum gate thresholds are met.

    Args:
        eval_metrics: Evaluation metrics dict (keys like ``"mean_reward"``).
        stage_config: Stage configuration dict as returned by
            :func:`~environments.shared.config.load_stage_config` (the TOML
            ``[curriculum]`` section lives under ``"curriculum_kwargs"``).

    Returns:
        ``True`` if the gate is passed and training should advance.

    Raises:
        GateSchemaError: If the stage's gate declaration is missing, unknown,
            or malformed. This used to log a warning and return ``True``, so a
            stage with no reward threshold advanced unconditionally — the same
            fail-open behaviour the SB3 path had, reached by a different route.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    validate_gate_config(stage_config.get("stage", "?"), curriculum, advancement_enabled=True)
    min_reward = curriculum.get("min_avg_reward")
    if min_reward is None:
        raise GateSchemaError(
            "stage config declares an advancement gate but sets no "
            "min_avg_reward, so there is nothing for the JAX path to check. "
            'Declare gate_kind = "none/v1" for a non-advancing pilot instead of '
            "leaving the threshold out."
        )
    # min_avg_reward is an EPISODE-level threshold (shared with the SB3
    # TOMLs, e.g. 100.0).  The trainer's "mean_reward" is the mean PER-STEP
    # rollout reward (~0.5-2 for a good policy), so gating on it would fail
    # every well-trained policy.
    episode_return = eval_metrics.get("mean_episode_return")
    if episode_return is None:
        _logger.warning(
            "eval_metrics has no mean_episode_return — falling back to per-step "
            "mean_reward, which is NOT comparable to TOML min_avg_reward."
        )
        episode_return = eval_metrics.get("mean_reward", 0.0)
    return bool(episode_return >= min_reward)


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
            ``train_fn(species, stage, **kwargs) -> (params, eval_metrics, obs_stats)``
            (a legacy 2-tuple without *obs_stats* is also accepted).
        stages: Tuple of stage numbers to train through.
        **train_kwargs: Extra keyword arguments forwarded to ``train_fn``.

    Returns:
        Dict mapping stage number to final ``(params, eval_metrics)``.
    """
    results: dict[int, Any] = {}

    params = None
    obs_stats = None
    for stage in stages:
        stage_config = load_stage_config(species, stage)

        # Pass previous stage's params AND observation-normalization stats to
        # the next stage — carrying only the weights would feed the policy
        # freshly re-scaled inputs it was never trained on (the SB3 path
        # carries obs_rms across stages for the same reason).
        if params is not None:
            train_kwargs["init_params"] = params
        if obs_stats is not None:
            train_kwargs["init_obs_stats"] = obs_stats

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
            "vf_clip_range": "vf_clip_range",
            "ent_coef": "ent_coef",
            "vf_coef": "vf_coef",
            "ppo_epochs": "n_epochs",
            "target_kl": "target_kl",
            "minibatch_size": "minibatch_size",
            "warmup_updates": "warmup_updates",
            "warmup_clip_range": "warmup_clip_range",
            "warmup_ent_coef": "warmup_ent_coef",
            "ramp_updates": "ramp_updates",
            "ramp_start_fraction": "ramp_start_fraction",
        }
        for toml_key, param_name in _JAX_KEY_MAP.items():
            if toml_key in jax_kwargs:
                stage_train_kwargs[param_name] = jax_kwargs[toml_key]

        # Always pass env_kwargs so reward weights reach the MJX env
        stage_train_kwargs["env_kwargs"] = env_kwargs

        # Override fall_penalty / reset noise from [jax] section if specified.
        # Use direct assignment — setdefault is a no-op when [env] already
        # defines the key, which silently ignores the JAX-specific override.
        if "fall_penalty" in jax_kwargs:
            env_kwargs["fall_penalty"] = jax_kwargs["fall_penalty"]
        for noise_key in ("reset_noise_scale", "init_qpos_noise", "init_yaw_noise"):
            if noise_key in jax_kwargs:
                env_kwargs[noise_key] = jax_kwargs[noise_key]

        result = train_fn(species=species, stage=stage, **stage_train_kwargs)
        if len(result) == 3:
            params, eval_metrics, obs_stats = result
        else:  # legacy 2-tuple train_fn
            params, eval_metrics = result
            obs_stats = None
        results[stage] = (params, eval_metrics)

        # Check gate (skip for last stage)
        if stage != stages[-1]:
            if not check_stage_gate(eval_metrics, stage_config):
                _logger.warning(
                    "Stage %d gate NOT passed (episode return=%.1f). Stopping early.",
                    stage,
                    eval_metrics.get("mean_episode_return", eval_metrics.get("mean_reward", 0.0)),
                )
                break
            _logger.info("Stage %d gate passed. Advancing to stage %d.", stage, stage + 1)

    return results
