"""JAX-compatible curriculum manager.

Mirrors the stage-gating logic from ``curriculum.py`` but works with
the JAX training path.  Stage configs are loaded from the same TOML
files used by the SB3 path.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from .config import load_stage_config
from .curriculum.gate_schema import GateSchemaError, validate_gate_config
from .curriculum.stance_gate import (
    STANCE_GATE_KIND,
    StanceGateThresholds,
    StancePanel,
    evaluate_stance_gate,
)

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
    gate_kind = validate_gate_config(stage_config.get("stage", "?"), curriculum, advancement_enabled=True)

    if gate_kind == STANCE_GATE_KIND:
        return _check_stance_gate(eval_metrics, curriculum)

    # reward_and_length/v1 requires min_avg_reward, so a validated config of
    # that kind always carries it; a KeyError here would mean the schema and
    # this branch fell out of sync.
    min_reward = curriculum["min_avg_reward"]
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
    if not bool(episode_return >= min_reward):
        return False

    # The length half of reward_and_length/v1.  This used to be ignored here
    # while the SB3 CurriculumManager enforced it, so a stage carrying
    # min_avg_episode_length was gated on reward alone under JAX -- the exact
    # "one backend silently ignores a gate the other enforces" divergence the
    # gate schema exists to prevent.  It matters more since stage 1 began
    # encoding its full-horizon floor in this field (PLANT_VALIDATION §12).
    min_length = curriculum.get("min_avg_episode_length")
    if min_length is None:
        return True
    episode_length = eval_metrics.get("mean_episode_length")
    if episode_length is None:
        raise GateSchemaError(
            "stage config sets min_avg_episode_length but eval_metrics carries "
            "no mean_episode_length, so the length half of the gate cannot be "
            "checked. Passing on reward alone would silently half-enforce the "
            "gate; emit mean_episode_length from the trainer instead."
        )
    return bool(episode_length >= min_length)


#: Metrics the MJX trainer must emit for ``stance_quality/v1``.  Named here so
#: the error message can say exactly what is missing.
_STANCE_METRIC_KEYS = (
    "n_eval_episodes",
    "full_horizon_fraction",
    "n_duty_episodes",
    "mean_unsupported_duty",
    "unsupported_duty_ucb",
)


def _check_stance_gate(eval_metrics: dict[str, float], curriculum: dict[str, Any]) -> bool:
    """Evaluate ``stance_quality/v1`` on the JAX path.

    Runs the *same* :func:`~environments.shared.curriculum.stance_gate.evaluate_stance_gate`
    the SB3 ``CurriculumManager`` runs, on a panel the MJX trainer must supply
    already reduced -- so the two backends cannot disagree about the criteria.

    Raises:
        GateSchemaError: If the trainer emitted none of the stance metrics.
            Deliberately loud rather than a warning-and-pass: a stage that
            declares this kind and is then gated on nothing is precisely the
            "one backend silently ignores a gate the other enforces"
            divergence the versioned schema exists to prevent.
    """
    missing = [key for key in _STANCE_METRIC_KEYS if eval_metrics.get(key) is None]
    if missing:
        raise GateSchemaError(
            f"stage declares gate_kind {STANCE_GATE_KIND!r} but eval_metrics is missing "
            f"{missing}, so stance quality cannot be checked. Passing on the metrics that "
            "are present would half-enforce the gate. The MJX evaluator must reduce its "
            "rollout into a StancePanel with stance_gate.summarize_stance_panel() -- "
            "reconstructing episode boundaries from cumsum(lengths) over the per-step "
            "diag_* arrays -- and emit these keys. Note that reconstruction is valid for "
            "BIPEDS only: see the diag_r_foot/diag_l_foot interleaving defect in "
            "KNOWN_ISSUES, which must be fixed before this gate can be used for "
            "brachiosaurus or dibothrosuchus."
        )

    panel = StancePanel(
        n_episodes=int(eval_metrics["n_eval_episodes"]),
        full_horizon_fraction=float(eval_metrics["full_horizon_fraction"]),
        mean_reward=float(eval_metrics.get("mean_episode_return", eval_metrics.get("mean_reward", -math.inf))),
        n_duty_episodes=int(eval_metrics["n_duty_episodes"]),
        mean_unsupported_duty=float(eval_metrics["mean_unsupported_duty"]),
        unsupported_duty_ucb=float(eval_metrics["unsupported_duty_ucb"]),
    )
    thresholds = StanceGateThresholds(
        min_full_horizon_fraction=float(curriculum["min_full_horizon_fraction"]),
        max_unsupported_duty=float(curriculum["max_unsupported_duty"]),
        max_unsupported_duty_ucb=float(curriculum["max_unsupported_duty_ucb"]),
        settle_steps=int(curriculum.get("settle_steps", 0)),
        min_eval_episodes=int(curriculum.get("min_eval_episodes", 40)),
        min_avg_reward=float(curriculum.get("min_avg_reward", -math.inf)),
        required_consecutive=int(curriculum.get("required_consecutive", 3)),
    )
    passed, failures = evaluate_stance_gate(panel, thresholds)
    if not passed:
        _logger.info("Stance gate not met: %s", "; ".join(failures))
    return passed


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
