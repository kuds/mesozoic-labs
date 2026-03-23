"""JAX-compatible curriculum manager.

Mirrors the stage-gating logic from ``curriculum.py`` but works with
the JAX training path.  Stage configs are loaded from the same TOML
files used by the SB3 path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import load_stage_config


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

        params, eval_metrics = train_fn(species=species, stage=stage, **train_kwargs)
        results[stage] = (params, eval_metrics)

        # Check gate (skip for last stage)
        if stage != stages[-1]:
            if not check_stage_gate(eval_metrics, stage_config):
                print(
                    f"[Curriculum] Stage {stage} gate NOT passed "
                    f"(reward={eval_metrics.get('mean_reward', 0.0):.1f}). "
                    f"Stopping early."
                )
                break
            print(f"[Curriculum] Stage {stage} gate passed. Advancing to stage {stage + 1}.")

    return results
