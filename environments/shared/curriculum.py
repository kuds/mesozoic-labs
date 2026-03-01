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
from environments.shared.metrics import LocomotionMetrics
from environments.shared.wandb_integration import log_eval_metrics

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
        success_rates: Optional[List[float]] = None,
    ) -> Dict[str, float]:
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
        rewards: Optional[List[float]] = None,
        episode_lengths: Optional[List[float]] = None,
        forward_velocities: Optional[List[float]] = None,
        success_rates: Optional[List[float]] = None,
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
        if "min_success_rate" in cur:
            threshold_fields["min_success_rate"] = cur["min_success_rate"]
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

    When an ``eval_callback`` is provided, this callback piggybacks on
    its evaluation results (reward / episode length) instead of running
    a redundant full eval pass.  A small supplementary eval still runs
    to collect forward velocity and success rate from the info dicts,
    which ``EvalCallback`` does not capture.

    Args:
        curriculum_manager: The manager tracking stage progress.
        eval_env: Vectorized evaluation environment.
        eval_freq: Evaluate every N training steps.
        n_eval_episodes: Number of episodes per evaluation (used only
            when no *eval_callback* is provided).
        eval_callback: Optional ``EvalCallback`` to read results from.
            When set, the callback reads reward/length from the
            evaluations.npz and only runs a short supplementary eval
            for forward velocity and success rate.
        supplementary_episodes: Number of episodes for the supplementary
            eval when *eval_callback* is provided (default 5).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        eval_env: Any,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        eval_callback: Any = None,
        supplementary_episodes: int = 5,
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
        self.eval_callback = eval_callback
        self.supplementary_episodes = supplementary_episodes
        self.ready_to_advance = False
        self._last_eval_step = 0
        self._last_seen_n_evals = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        if self.eval_callback is not None:
            return self._on_step_with_eval_callback()
        return self._on_step_standalone()

    def _read_latest_eval(self) -> tuple:
        """Read the latest per-episode rewards/lengths from EvalCallback's npz.

        Returns ``(rewards, lengths, n_evals)`` or ``(None, None, 0)``
        if no new evaluation data is available.
        """
        log_path = getattr(self.eval_callback, "log_path", None)
        if log_path is None:
            return None, None, 0

        from pathlib import Path

        npz_path = Path(log_path) / "evaluations.npz"
        if not npz_path.exists():
            return None, None, 0

        data = np.load(str(npz_path))
        eval_rewards = data["results"]  # (n_evals, n_episodes)
        eval_lengths = data["ep_lengths"]

        n_evals = eval_rewards.shape[0]
        if n_evals <= self._last_seen_n_evals:
            return None, None, n_evals  # No new eval

        self._last_seen_n_evals = n_evals
        latest_rewards = eval_rewards[-1].tolist()
        latest_lengths = eval_lengths[-1].tolist()
        return latest_rewards, latest_lengths, n_evals

    def _run_supplementary_eval(self) -> tuple:
        """Run a small eval pass to collect forward velocity and success rate.

        Returns ``(forward_vels, success_flags, episode_reports)``.
        """
        forward_vels: List[float] = []
        success_flags: List[float] = []
        episode_reports: List[Dict[str, Any]] = []

        for _ in range(self.supplementary_episodes):
            obs = self.eval_env.reset()
            metrics = LocomotionMetrics()
            ep_forward_vels: List[float] = []
            ep_success = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.eval_env.step(action)
                step_reward = float(reward[0])
                metrics.record_step(infos[0], step_reward)
                if "forward_vel" in infos[0]:
                    ep_forward_vels.append(float(infos[0]["forward_vel"]))
                for key in ("bite_success", "strike_success", "food_reached"):
                    if infos[0].get(key):
                        ep_success = 1.0
                        break
                done = bool(dones[0])
            if ep_forward_vels:
                forward_vels.append(float(np.mean(ep_forward_vels)))
            success_flags.append(ep_success)
            episode_reports.append(metrics.compute())

        return forward_vels, success_flags, episode_reports

    def _log_locomotion_metrics(self, episode_reports: List[Dict[str, Any]]) -> None:
        """Log aggregated locomotion metrics from episode reports."""
        if not episode_reports:
            return

        agg = LocomotionMetrics.aggregate_episodes(episode_reports)
        stage = self.curriculum_manager.current_stage

        metric_keys = [
            "mean_forward_velocity",
            "mean_total_distance",
            "mean_cost_of_transport",
            "mean_gait_symmetry",
            "mean_stride_frequency",
            "mean_pelvis_height",
            "mean_mean_tilt_angle",
            "mean_velocity_consistency",
        ]
        metric_parts = []
        for k in metric_keys:
            if k in agg:
                short_name = k.replace("mean_", "")
                metric_parts.append(f"{short_name}={agg[k]:.3f}")
        if metric_parts:
            logger.info(
                "Stage %d locomotion: %s",
                stage,
                ", ".join(metric_parts),
            )

        term_counts = agg.get("termination_counts")
        if term_counts:
            n_eps = len(episode_reports)
            parts = [f"{reason}={count}" for reason, count in sorted(term_counts.items())]
            logger.info(
                "Stage %d terminations (%d eps): %s",
                stage,
                n_eps,
                ", ".join(parts),
            )

        log_eval_metrics(agg, stage, step=self.num_timesteps)

    def _on_step_with_eval_callback(self) -> bool:
        """Advancement check using EvalCallback results + supplementary eval."""
        rewards, lengths, _n_evals = self._read_latest_eval()
        if rewards is None:
            return True  # EvalCallback hasn't produced new results yet

        # Run supplementary eval for forward_vel / success_rate
        forward_vels, success_flags, episode_reports = self._run_supplementary_eval()
        self._log_locomotion_metrics(episode_reports)

        fwd_vel_arg = forward_vels if forward_vels else None
        success_arg = success_flags if success_flags else None
        if self.curriculum_manager.should_advance(rewards, lengths, fwd_vel_arg, success_arg):
            self.ready_to_advance = True
            logger.info(
                "CurriculumCallback: stage %d thresholds met at step %d. Stopping training for stage advancement.",
                self.curriculum_manager.current_stage,
                self.num_timesteps,
            )
            return False

        return True

    def _on_step_standalone(self) -> bool:
        """Full standalone evaluation (backward-compatible path)."""
        rewards: List[float] = []
        lengths: List[float] = []
        forward_vels: List[float] = []
        success_flags: List[float] = []
        episode_reports: List[Dict[str, Any]] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            metrics = LocomotionMetrics()
            episode_reward = 0.0
            episode_length = 0
            ep_forward_vels: List[float] = []
            ep_success = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.eval_env.step(action)
                step_reward = float(reward[0])
                episode_reward += step_reward
                episode_length += 1
                metrics.record_step(infos[0], step_reward)
                if "forward_vel" in infos[0]:
                    ep_forward_vels.append(float(infos[0]["forward_vel"]))
                for key in ("bite_success", "strike_success", "food_reached"):
                    if infos[0].get(key):
                        ep_success = 1.0
                        break
                done = bool(dones[0])
            rewards.append(episode_reward)
            lengths.append(float(episode_length))
            if ep_forward_vels:
                forward_vels.append(float(np.mean(ep_forward_vels)))
            success_flags.append(ep_success)
            episode_reports.append(metrics.compute())

        self._log_locomotion_metrics(episode_reports)

        fwd_vel_arg = forward_vels if forward_vels else None
        success_arg = success_flags if success_flags else None
        if self.curriculum_manager.should_advance(rewards, lengths, fwd_vel_arg, success_arg):
            self.ready_to_advance = True
            logger.info(
                "CurriculumCallback: stage %d thresholds met at step %d. Stopping training for stage advancement.",
                self.curriculum_manager.current_stage,
                self.num_timesteps,
            )
            return False

        return True


def load_vecnorm_stats(vecnorm_path: str, train_env, eval_env=None) -> bool:
    """Load VecNormalize running statistics from a previous stage into new envs.

    This preserves observation/reward normalization across curriculum stage
    transitions, preventing the policy from receiving scrambled inputs when
    the environment wrapper is re-created for a new stage.

    Args:
        vecnorm_path: Path to a ``_vecnorm.pkl`` file saved by a previous stage.
        train_env: The new stage's training ``VecNormalize`` wrapper.
            ``training`` is left ``True`` so stats keep updating.
        eval_env: Optional evaluation ``VecNormalize`` wrapper.
            ``training`` is set to ``False``; ``norm_reward`` is disabled.

    Returns:
        ``True`` if stats were loaded, ``False`` if the file was not found.
    """
    from pathlib import Path as _Path

    if not _SB3_AVAILABLE:
        logger.warning("stable-baselines3 not available; skipping VecNormalize load.")
        return False

    from stable_baselines3.common.vec_env import VecNormalize

    path = _Path(vecnorm_path)
    if not path.exists():
        logger.debug("VecNormalize file not found: %s", vecnorm_path)
        return False

    logger.info("Loading VecNormalize stats from: %s", vecnorm_path)
    prev_norm = VecNormalize.load(str(path), train_env.venv)

    train_env.obs_rms = prev_norm.obs_rms
    train_env.ret_rms = prev_norm.ret_rms
    train_env.training = True
    train_env.norm_reward = True

    if eval_env is not None:
        eval_env.obs_rms = prev_norm.obs_rms.copy()
        eval_env.ret_rms = prev_norm.ret_rms.copy()
        eval_env.training = False
        eval_env.norm_reward = False

    return True
