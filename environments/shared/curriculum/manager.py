"""Stage thresholds and the backend-independent curriculum manager.

:class:`CurriculumManager` owns the advancement decision and per-stage config
loading; it has no SB3 dependency, so it can be driven from a JAX loop or a
notebook just as well as from :class:`~.advancement.CurriculumCallback`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from environments.shared.config import load_all_stages

from .gate_schema import validate_gate_config

logger = logging.getLogger(__name__)


@dataclass
class StageThreshold:
    """Performance thresholds that must be met to advance past a stage."""

    min_avg_reward: float = -np.inf
    min_avg_episode_length: float = 0.0
    min_avg_forward_vel: float = 0.0
    min_success_rate: float = 0.0
    min_eval_episodes: int = 10
    required_consecutive: int = 3


class CurriculumManager:
    """Manages automated progression through curriculum stages.

    Tracks evaluation results and determines when a training stage's
    performance thresholds have been met, signalling that it is time
    to advance to the next stage.

    Args:
        species: Species name used to load TOML configs.
        stage_thresholds: Mapping from stage number to threshold dict.
            Keys in each dict should match ``StageThreshold`` fields.
        start_stage: Initial curriculum stage (default 1).
        total_stages: Total number of stages (default 3).
    """

    def __init__(
        self,
        species: str,
        stage_thresholds: dict[int, dict[str, Any]] | None = None,
        start_stage: int = 1,
        total_stages: int = 3,
    ):
        self.species = species
        self.total_stages = total_stages
        self._current_stage = start_stage
        self._configs = load_all_stages(species)

        # Build threshold objects per stage
        self._thresholds: dict[int, StageThreshold] = {}
        for stage in range(1, total_stages + 1):
            if stage_thresholds and stage in stage_thresholds:
                self._thresholds[stage] = StageThreshold(**stage_thresholds[stage])
            else:
                self._thresholds[stage] = StageThreshold()

        # History of evaluation results per stage
        self._eval_history: dict[int, list[dict[str, float]]] = {s: [] for s in range(1, total_stages + 1)}
        self._consecutive_passes: dict[int, int] = {s: 0 for s in range(1, total_stages + 1)}

        logger.info(
            "CurriculumManager initialized for %s: stage %d/%d",
            species,
            start_stage,
            total_stages,
        )

    @property
    def current_stage(self) -> int:
        """Current curriculum stage number."""
        return self._current_stage

    @property
    def current_threshold(self) -> StageThreshold:
        """Effective threshold for the current stage, including overrides."""

        return self._thresholds[self._current_stage]

    @property
    def is_final_stage(self) -> bool:
        """Whether the manager is on the last stage."""
        return self._current_stage >= self.total_stages

    def current_config(self) -> dict[str, Any]:
        """Return the TOML config dict for the current stage."""
        return self._configs[self._current_stage]

    def record_eval(
        self,
        rewards: list[float],
        episode_lengths: list[float],
        forward_velocities: list[float] | None = None,
        success_rates: list[float] | None = None,
    ) -> dict[str, float]:
        """Record evaluation results for the current stage.

        Args:
            rewards: List of episode total rewards from evaluation.
            episode_lengths: List of episode lengths from evaluation.
            forward_velocities: Optional list of mean forward velocities
                per episode (m/s). Used for locomotion stage gating.
            success_rates: Optional list of per-episode success flags
                (1.0 if prey contact / food reached, 0.0 otherwise).

        Returns:
            Summary dict with mean/std statistics.
        """
        summary = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
            "n_episodes": len(rewards),
        }
        if forward_velocities is not None:
            summary["mean_forward_vel"] = float(np.mean(forward_velocities))
        if success_rates is not None:
            summary["mean_success_rate"] = float(np.mean(success_rates))
        self._eval_history[self._current_stage].append(summary)

        vel_str = ""
        if "mean_forward_vel" in summary:
            vel_str = f", fwd_vel={summary['mean_forward_vel']:.2f} m/s"
        success_str = ""
        if "mean_success_rate" in summary:
            success_str = f", success={summary['mean_success_rate']:.0%}"
        logger.info(
            "Stage %d eval: reward=%.2f +/- %.2f, length=%.1f +/- %.1f%s%s (%d eps)",
            self._current_stage,
            summary["mean_reward"],
            summary["std_reward"],
            summary["mean_length"],
            summary["std_length"],
            vel_str,
            success_str,
            summary["n_episodes"],
        )
        return summary

    def should_advance(
        self,
        rewards: list[float] | None = None,
        episode_lengths: list[float] | None = None,
        forward_velocities: list[float] | None = None,
        success_rates: list[float] | None = None,
    ) -> bool:
        """Check whether performance thresholds are met for advancement.

        If ``rewards`` and ``episode_lengths`` are provided they are
        recorded first via :meth:`record_eval`.

        Args:
            rewards: Per-episode total rewards.
            episode_lengths: Per-episode step counts.
            forward_velocities: Per-episode mean forward velocities (m/s).
            success_rates: Per-episode success flags (1.0 if prey contact
                / food reached, 0.0 otherwise).

        Returns:
            True if the current stage thresholds have been met for the
            required number of consecutive evaluations.
        """
        if rewards is not None and episode_lengths is not None:
            self.record_eval(rewards, episode_lengths, forward_velocities, success_rates)

        threshold = self._thresholds[self._current_stage]
        history = self._eval_history[self._current_stage]

        if not history:
            return False

        latest = history[-1]

        passes = (
            latest["mean_reward"] >= threshold.min_avg_reward
            and latest["mean_length"] >= threshold.min_avg_episode_length
            and latest["n_episodes"] >= threshold.min_eval_episodes
        )

        # Forward velocity gate (only checked when threshold is > 0)
        if threshold.min_avg_forward_vel > 0.0:
            mean_vel = latest.get("mean_forward_vel", 0.0)
            passes = passes and mean_vel >= threshold.min_avg_forward_vel

        # Success rate gate (only checked when threshold is > 0)
        if threshold.min_success_rate > 0.0:
            mean_success = latest.get("mean_success_rate", 0.0)
            passes = passes and mean_success >= threshold.min_success_rate

        if passes:
            self._consecutive_passes[self._current_stage] += 1
        else:
            self._consecutive_passes[self._current_stage] = 0

        met = self._consecutive_passes[self._current_stage] >= threshold.required_consecutive

        if met:
            logger.info(
                "Stage %d thresholds met (%d consecutive passes). Ready to advance.",
                self._current_stage,
                threshold.required_consecutive,
            )

        return met

    def advance(self) -> int:
        """Advance to the next curriculum stage.

        Returns:
            The new stage number.

        Raises:
            RuntimeError: If already on the final stage.
        """
        if self.is_final_stage:
            raise RuntimeError(f"Cannot advance past final stage {self.total_stages}")

        prev = self._current_stage
        self._current_stage += 1

        logger.info(
            "Advanced from stage %d to stage %d (%s -> %s)",
            prev,
            self._current_stage,
            self._configs[prev]["name"],
            self._configs[self._current_stage]["name"],
        )

        return self._current_stage

    def summary(self) -> dict[str, Any]:
        """Return a summary of the curriculum state for logging/serialization."""
        return {
            "species": self.species,
            "current_stage": self._current_stage,
            "total_stages": self.total_stages,
            "eval_history": dict(self._eval_history),
            "consecutive_passes": dict(self._consecutive_passes),
        }


def thresholds_from_configs(
    configs: dict[int, dict[str, Any]],
    *,
    advancement_enabled: bool = True,
) -> dict[int, dict[str, Any]]:
    """Extract stage thresholds from loaded TOML configs.

    Reads the ``curriculum_kwargs`` section from each stage config and
    returns a dict suitable for passing to ``CurriculumManager``.

    Args:
        configs: Dict mapping stage number to config dict (from ``load_all_stages``).
        advancement_enabled: Whether this run may advance between stages. Passed
            through to :func:`~environments.shared.curriculum.gate_schema.validate_gate_config`,
            which tolerates a missing gate declaration only when it is ``False``.

    Returns:
        Dict mapping stage number to threshold kwargs.

    Raises:
        GateSchemaError: If any stage's gate declaration is missing, unknown,
            or malformed.
    """
    thresholds: dict[int, dict[str, Any]] = {}
    for stage, cfg in configs.items():
        cur = cfg.get("curriculum_kwargs", {})
        # Fail closed. Silently dropping an unrecognised key here is what let a
        # composite-only gate config produce no thresholds at all, after which
        # StageThreshold's permissive defaults advanced the stage on any
        # evaluation whatsoever. See gate_schema for the full failure mode.
        validate_gate_config(stage, cur, advancement_enabled=advancement_enabled)
        threshold_fields: dict[str, Any] = {}
        if "min_avg_reward" in cur:
            threshold_fields["min_avg_reward"] = cur["min_avg_reward"]
        if "min_avg_episode_length" in cur:
            threshold_fields["min_avg_episode_length"] = cur["min_avg_episode_length"]
        if "min_avg_forward_vel" in cur:
            threshold_fields["min_avg_forward_vel"] = cur["min_avg_forward_vel"]
        if "min_success_rate" in cur:
            threshold_fields["min_success_rate"] = cur["min_success_rate"]
        if "min_eval_episodes" in cur:
            threshold_fields["min_eval_episodes"] = cur["min_eval_episodes"]
        if "required_consecutive" in cur:
            threshold_fields["required_consecutive"] = cur["required_consecutive"]
        if threshold_fields:
            thresholds[stage] = threshold_fields
    return thresholds
