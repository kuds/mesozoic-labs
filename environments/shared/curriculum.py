"""
Automated curriculum manager for multi-stage training.

Monitors evaluation metrics and automatically advances through curriculum
stages when performance thresholds are met. Supports both PPO and SAC
algorithms with per-stage hyperparameter loading from TOML configs.

Usage with CurriculumManager directly::

    from environments.shared.curriculum import CurriculumManager

    manager = CurriculumManager(species="velociraptor")

    if manager.should_advance(eval_rewards, eval_lengths):
        manager.advance()
        new_config = manager.current_config()

Usage with CurriculumCallback (SB3 integration)::

    from environments.shared.curriculum import CurriculumManager, CurriculumCallback

    manager = CurriculumManager(species="velociraptor")
    curriculum_cb = CurriculumCallback(manager, eval_env, eval_freq=10000)

    model.learn(total_timesteps=500_000, callback=curriculum_cb)

    if curriculum_cb.ready_to_advance:
        manager.advance()
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from environments.shared.config import load_all_stages

try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv

    _SB3_AVAILABLE = True
except ImportError:
    BaseCallback = object  # type: ignore[misc,assignment]
    VecEnv = object  # type: ignore[misc,assignment]
    _SB3_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StageThreshold:
    """Performance thresholds that must be met to advance past a stage."""

    min_avg_reward: float = -np.inf
    min_avg_episode_length: float = 0.0
    min_avg_forward_vel: float = 0.0
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
        self._eval_history: Dict[int, List[Dict[str, float]]] = {s: [] for s in range(1, total_stages + 1)}
        self._consecutive_passes: Dict[int, int] = {s: 0 for s in range(1, total_stages + 1)}

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
        forward_velocities: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Record evaluation results for the current stage.

        Args:
            rewards: List of episode total rewards from evaluation.
            episode_lengths: List of episode lengths from evaluation.
            forward_velocities: Optional list of mean forward velocities
                per episode (m/s). Used for locomotion stage gating.

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
        self._eval_history[self._current_stage].append(summary)

        vel_str = ""
        if "mean_forward_vel" in summary:
            vel_str = f", fwd_vel={summary['mean_forward_vel']:.2f} m/s"
        logger.info(
            "Stage %d eval: reward=%.2f +/- %.2f, length=%.1f +/- %.1f%s (%d eps)",
            self._current_stage,
            summary["mean_reward"],
            summary["std_reward"],
            summary["mean_length"],
            summary["std_length"],
            vel_str,
            summary["n_episodes"],
        )
        return summary

    def should_advance(
        self,
        rewards: Optional[List[float]] = None,
        episode_lengths: Optional[List[float]] = None,
        forward_velocities: Optional[List[float]] = None,
    ) -> bool:
        """Check whether performance thresholds are met for advancement.

        If ``rewards`` and ``episode_lengths`` are provided they are
        recorded first via :meth:`record_eval`.

        Args:
            rewards: Per-episode total rewards.
            episode_lengths: Per-episode step counts.
            forward_velocities: Per-episode mean forward velocities (m/s).

        Returns:
            True if the current stage thresholds have been met for the
            required number of consecutive evaluations.
        """
        if self.is_final_stage:
            return False

        if rewards is not None and episode_lengths is not None:
            self.record_eval(rewards, episode_lengths, forward_velocities)

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

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the curriculum state for logging/serialization."""
        return {
            "species": self.species,
            "current_stage": self._current_stage,
            "total_stages": self.total_stages,
            "eval_history": dict(self._eval_history),
            "consecutive_passes": dict(self._consecutive_passes),
        }


def thresholds_from_configs(
    configs: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Extract stage thresholds from loaded TOML configs.

    Reads the ``curriculum_kwargs`` section from each stage config and
    returns a dict suitable for passing to ``CurriculumManager``.

    Args:
        configs: Dict mapping stage number to config dict (from ``load_all_stages``).

    Returns:
        Dict mapping stage number to threshold kwargs.
    """
    thresholds: Dict[int, Dict[str, Any]] = {}
    for stage, cfg in configs.items():
        cur = cfg.get("curriculum_kwargs", {})
        threshold_fields: Dict[str, Any] = {}
        if "min_avg_reward" in cur:
            threshold_fields["min_avg_reward"] = cur["min_avg_reward"]
        if "min_avg_episode_length" in cur:
            threshold_fields["min_avg_episode_length"] = cur["min_avg_episode_length"]
        if "min_avg_forward_vel" in cur:
            threshold_fields["min_avg_forward_vel"] = cur["min_avg_forward_vel"]
        if "required_consecutive" in cur:
            threshold_fields["required_consecutive"] = cur["required_consecutive"]
        if threshold_fields:
            thresholds[stage] = threshold_fields
    return thresholds


class CurriculumCallback(BaseCallback):  # type: ignore[misc]
    """SB3 callback that monitors evaluation and signals stage advancement.

    Periodically evaluates the policy and feeds results to a
    :class:`CurriculumManager`. When thresholds are met, the callback
    stops the current ``model.learn()`` call by returning ``False``
    from ``_on_step``. The caller can then check :attr:`ready_to_advance`
    and advance to the next stage.

    Args:
        curriculum_manager: The manager tracking stage progress.
        eval_env: Vectorized evaluation environment.
        eval_freq: Evaluate every N training steps.
        n_eval_episodes: Number of episodes per evaluation.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        eval_env: Any,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: int = 0,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for CurriculumCallback. Install with: pip install stable-baselines3"
            )
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.ready_to_advance = False
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.curriculum_manager.is_final_stage:
            return True

        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        # Run evaluation episodes
        rewards: List[float] = []
        lengths: List[float] = []
        forward_vels: List[float] = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            ep_forward_vels: List[float] = []
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.eval_env.step(action)
                episode_reward += float(reward[0])
                episode_length += 1
                if "forward_vel" in infos[0]:
                    ep_forward_vels.append(float(infos[0]["forward_vel"]))
                done = bool(dones[0])
            rewards.append(episode_reward)
            lengths.append(float(episode_length))
            if ep_forward_vels:
                forward_vels.append(float(np.mean(ep_forward_vels)))

        fwd_vel_arg = forward_vels if forward_vels else None
        if self.curriculum_manager.should_advance(rewards, lengths, fwd_vel_arg):
            self.ready_to_advance = True
            logger.info(
                "CurriculumCallback: stage %d thresholds met at step %d. Stopping training for stage advancement.",
                self.curriculum_manager.current_stage,
                self.num_timesteps,
            )
            return False  # Stop model.learn()

        return True
