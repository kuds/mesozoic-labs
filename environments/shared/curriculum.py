"""
Automated curriculum manager for multi-stage training.

Monitors evaluation metrics and automatically advances through curriculum
stages when performance thresholds are met. Supports both PPO and SAC
algorithms with per-stage hyperparameter loading from TOML configs.

Usage:
    from environments.shared.curriculum import CurriculumManager

    manager = CurriculumManager(
        species="velociraptor",
        stage_thresholds={
            1: {"min_avg_reward": 50.0, "min_avg_episode_length": 400},
            2: {"min_avg_reward": 100.0, "min_avg_episode_length": 800},
        },
    )

    # In training loop or as a callback:
    if manager.should_advance(eval_rewards, eval_lengths):
        manager.advance()
        new_config = manager.current_config()
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from environments.shared.config import load_all_stages

logger = logging.getLogger(__name__)


@dataclass
class StageThreshold:
    """Performance thresholds that must be met to advance past a stage."""

    min_avg_reward: float = -np.inf
    min_avg_episode_length: float = 0.0
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
        stage_thresholds: Optional[Dict[int, Dict[str, Any]]] = None,
        start_stage: int = 1,
        total_stages: int = 3,
    ):
        self.species = species
        self.total_stages = total_stages
        self._current_stage = start_stage
        self._configs = load_all_stages(species)

        # Build threshold objects per stage
        self._thresholds: Dict[int, StageThreshold] = {}
        for stage in range(1, total_stages + 1):
            if stage_thresholds and stage in stage_thresholds:
                self._thresholds[stage] = StageThreshold(**stage_thresholds[stage])
            else:
                self._thresholds[stage] = StageThreshold()

        # History of evaluation results per stage
        self._eval_history: Dict[int, List[Dict[str, float]]] = {
            s: [] for s in range(1, total_stages + 1)
        }
        self._consecutive_passes: Dict[int, int] = {
            s: 0 for s in range(1, total_stages + 1)
        }

        logger.info(
            "CurriculumManager initialized for %s: stage %d/%d",
            species, start_stage, total_stages,
        )

    @property
    def current_stage(self) -> int:
        """Current curriculum stage number."""
        return self._current_stage

    @property
    def is_final_stage(self) -> bool:
        """Whether the manager is on the last stage."""
        return self._current_stage >= self.total_stages

    def current_config(self) -> Dict[str, Any]:
        """Return the TOML config dict for the current stage."""
        return self._configs[self._current_stage]

    def record_eval(
        self,
        rewards: List[float],
        episode_lengths: List[float],
    ) -> Dict[str, float]:
        """Record evaluation results for the current stage.

        Args:
            rewards: List of episode total rewards from evaluation.
            episode_lengths: List of episode lengths from evaluation.

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
        self._eval_history[self._current_stage].append(summary)

        logger.info(
            "Stage %d eval: reward=%.2f +/- %.2f, length=%.1f +/- %.1f (%d eps)",
            self._current_stage,
            summary["mean_reward"], summary["std_reward"],
            summary["mean_length"], summary["std_length"],
            summary["n_episodes"],
        )
        return summary

    def should_advance(
        self,
        rewards: Optional[List[float]] = None,
        episode_lengths: Optional[List[float]] = None,
    ) -> bool:
        """Check whether performance thresholds are met for advancement.

        If ``rewards`` and ``episode_lengths`` are provided they are
        recorded first via :meth:`record_eval`.

        Returns:
            True if the current stage thresholds have been met for the
            required number of consecutive evaluations.
        """
        if self.is_final_stage:
            return False

        if rewards is not None and episode_lengths is not None:
            self.record_eval(rewards, episode_lengths)

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
            raise RuntimeError(
                f"Cannot advance past final stage {self.total_stages}"
            )

        prev = self._current_stage
        self._current_stage += 1

        logger.info(
            "Advanced from stage %d to stage %d (%s -> %s)",
            prev, self._current_stage,
            self._configs[prev]["name"],
            self._configs[self._current_stage]["name"],
        )

        return self._current_stage

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the curriculum state for logging/serialization."""
        return {
            "species": self.species,
            "current_stage": self._current_stage,
            "total_stages": self.total_stages,
            "eval_history": dict(self._eval_history),
            "consecutive_passes": dict(self._consecutive_passes),
        }
