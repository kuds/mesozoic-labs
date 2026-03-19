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
    curriculum_cb = CurriculumCallback(manager, eval_env, eval_freq=50000)

    model.learn(total_timesteps=500_000, callback=curriculum_cb)

    if curriculum_cb.ready_to_advance:
        manager.advance()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

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


class _ConstantSchedule:
    """Picklable callable that returns a constant value.

    Replaces inline lambdas (e.g. ``lambda _: 0.02``) which capture the
    notebook cell's ``__globals__`` and fail to pickle in Colab/Jupyter
    because of ``zmq.Context`` objects in that namespace.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, _progress: float) -> float:
        return self.value


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
) -> dict[int, dict[str, Any]]:
    """Extract stage thresholds from loaded TOML configs.

    Reads the ``curriculum_kwargs`` section from each stage config and
    returns a dict suitable for passing to ``CurriculumManager``.

    Args:
        configs: Dict mapping stage number to config dict (from ``load_all_stages``).

    Returns:
        Dict mapping stage number to threshold kwargs.
    """
    thresholds: dict[int, dict[str, Any]] = {}
    for stage, cfg in configs.items():
        cur = cfg.get("curriculum_kwargs", {})
        threshold_fields: dict[str, Any] = {}
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
        eval_freq: int = 50000,
        n_eval_episodes: int = 30,
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

        The eval env's ``VecNormalize`` flags are temporarily set to
        ``training=False`` and ``norm_reward=False`` so that running
        statistics are not contaminated by evaluation episodes and
        rewards are returned in the original (unnormalised) scale.
        """
        # Disable VecNormalize stat updates during evaluation
        old_training = getattr(self.eval_env, "training", None)
        old_norm_reward = getattr(self.eval_env, "norm_reward", None)
        if old_training is not None:
            self.eval_env.training = False
        if old_norm_reward is not None:
            self.eval_env.norm_reward = False

        forward_vels: list[float] = []
        success_flags: list[float] = []
        episode_reports: list[dict[str, Any]] = []

        try:
            for _ in range(self.supplementary_episodes):
                obs = self.eval_env.reset()
                metrics = LocomotionMetrics()
                ep_forward_vels: list[float] = []
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
        finally:
            if old_training is not None:
                self.eval_env.training = old_training
            if old_norm_reward is not None:
                self.eval_env.norm_reward = old_norm_reward

        return forward_vels, success_flags, episode_reports

    def _log_locomotion_metrics(self, episode_reports: list[dict[str, Any]]) -> None:
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
        """Full standalone evaluation (backward-compatible path).

        Temporarily sets ``training=False`` and ``norm_reward=False`` on the
        eval env's ``VecNormalize`` wrapper so running statistics are not
        contaminated and rewards are in the original scale.
        """
        # Disable VecNormalize stat updates during evaluation
        old_training = getattr(self.eval_env, "training", None)
        old_norm_reward = getattr(self.eval_env, "norm_reward", None)
        if old_training is not None:
            self.eval_env.training = False
        if old_norm_reward is not None:
            self.eval_env.norm_reward = False

        rewards: list[float] = []
        lengths: list[float] = []
        forward_vels: list[float] = []
        success_flags: list[float] = []
        episode_reports: list[dict[str, Any]] = []

        try:
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                metrics = LocomotionMetrics()
                episode_reward = 0.0
                episode_length = 0
                ep_forward_vels: list[float] = []
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
        finally:
            if old_training is not None:
                self.eval_env.training = old_training
            if old_norm_reward is not None:
                self.eval_env.norm_reward = old_norm_reward

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


class StageWarmupCallback(BaseCallback):  # type: ignore[misc]
    """Constrain policy updates during the first N timesteps of a new stage.

    When transitioning between curriculum stages, the value function (critic)
    must adapt to the new reward landscape before the policy (actor) should
    change significantly.  This callback temporarily:

    **PPO mode:**

    * Reduces ``clip_range`` to a small value (default 0.02) so PPO's clipped
      surrogate objective barely moves the policy per update while the critic
      adapts via its own loss.
    * Increases ``ent_coef`` (default 0.02) so the policy maintains
      exploration breadth during the transition instead of committing to
      stale stage-1 action patterns.

    **SAC mode:**

    * Reduces the actor ``learning_rate`` to ``warmup_lr_scale`` × original
      (default 0.1×) so gradient updates to the actor are small while the
      twin critics adapt to the new reward landscape.  The critic learning
      rate is unchanged so Q-values converge quickly.
    * Temporarily fixes ``ent_coef`` to ``warmup_ent_coef`` (overriding
      ``"auto"``) to maintain exploration breadth during the transition.

    After ``warmup_timesteps`` have elapsed the original values are restored.

    Args:
        warmup_timesteps: Number of timesteps for the warm-up period.
        warmup_clip_range: Clip range during warm-up (PPO only).
        warmup_ent_coef: Entropy coefficient during warm-up (PPO and SAC).
        warmup_lr_scale: Factor to scale actor LR during warm-up (SAC only).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        warmup_timesteps: int = 100_000,
        warmup_clip_range: float = 0.02,
        warmup_ent_coef: float = 0.02,
        warmup_lr_scale: float = 0.1,
        verbose: int = 0,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for StageWarmupCallback.")
        super().__init__(verbose)
        self.warmup_timesteps = warmup_timesteps
        self.warmup_clip_range = warmup_clip_range
        self.warmup_ent_coef = warmup_ent_coef
        self.warmup_lr_scale = warmup_lr_scale
        self._original_clip_range = None
        self._original_ent_coef = None
        self._original_lr_schedule: Optional[Callable[[float], float]] = None
        self._original_log_ent_coef = None
        self._is_sac = False
        self._warmup_done = False

    def _on_training_start(self) -> None:
        self._is_sac = hasattr(self.model, "log_ent_coef")

        if self._is_sac:
            # SAC warmup: reduce LR and fix entropy coefficient.
            # Store the original lr_schedule callable (not the raw learning_rate
            # float) so we can restore it exactly after warmup.
            self._original_lr_schedule = self.model.lr_schedule
            original_lr = self._original_lr_schedule(1.0)
            warmup_lr = original_lr * self.warmup_lr_scale
            self.model.lr_schedule = lambda _: warmup_lr
            # Override auto-entropy by fixing ent_coef to a constant
            self._original_ent_coef = self.model.ent_coef
            self._original_log_ent_coef = self.model.log_ent_coef.item()
            self.model.ent_coef = self.warmup_ent_coef
            logger.info(
                "StageWarmupCallback [SAC]: warm-up active for %d timesteps (lr=%.2e → %.2e, ent_coef=%.3f)",
                self.warmup_timesteps,
                original_lr,
                warmup_lr,
                self.warmup_ent_coef,
            )
        elif hasattr(self.model, "clip_range"):
            # PPO warmup: reduce clip_range and set entropy coefficient
            self._original_clip_range = self.model.clip_range
            self._original_ent_coef = self.model.ent_coef
            self.model.clip_range = _ConstantSchedule(self.warmup_clip_range)
            self.model.ent_coef = self.warmup_ent_coef
            logger.info(
                "StageWarmupCallback [PPO]: warm-up active for %d timesteps (clip_range=%.3f, ent_coef=%.3f)",
                self.warmup_timesteps,
                self.warmup_clip_range,
                self.warmup_ent_coef,
            )
        else:
            self._warmup_done = True

    def _on_step(self) -> bool:
        if self._warmup_done:
            return True
        if self.num_timesteps >= self.warmup_timesteps:
            if self._is_sac:
                self.model.lr_schedule = self._original_lr_schedule
                self.model.ent_coef = self._original_ent_coef
                # Restore the learned log_ent_coef so auto-tuning resumes
                self.model.log_ent_coef.data.fill_(self._original_log_ent_coef)
            else:
                self.model.clip_range = self._original_clip_range
                self.model.ent_coef = self._original_ent_coef
            self._warmup_done = True
            logger.info(
                "StageWarmupCallback: warm-up complete at step %d. Restored original parameters.",
                self.num_timesteps,
            )
        return True


class RewardRampCallback(BaseCallback):  # type: ignore[misc]
    """Gradually ramp a reward weight from a starting value to the target.

    Instead of abruptly switching ``forward_vel_weight`` from 0.0 to its
    full stage-2 value, this callback linearly increases it over
    ``ramp_timesteps``.  This gives the policy time to adapt to the new
    reward signal without catastrophic gradient updates that overwrite
    previously learned balance behaviours.

    Works with both ``DummyVecEnv`` and ``SubprocVecEnv`` via
    ``env_method`` on the underlying VecEnv.

    Args:
        attr_name: Name of the reward-weight attribute on the environment
            (e.g. ``"forward_vel_weight"``).
        start_value: Initial value at the beginning of training.
        end_value: Target value at the end of the ramp.
        ramp_timesteps: Number of timesteps over which to ramp.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        attr_name: str = "forward_vel_weight",
        start_value: float = 0.1,
        end_value: float = 1.0,
        ramp_timesteps: int = 500_000,
        verbose: int = 0,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RewardRampCallback.")
        super().__init__(verbose)
        self.attr_name = attr_name
        self.start_value = start_value
        self.end_value = end_value
        self.ramp_timesteps = ramp_timesteps
        self._last_set_value: float | None = None

    def _set_env_attr(self, value: float) -> None:
        """Set the reward weight on all underlying envs."""
        vec_norm = self.model.get_env()
        # Access the inner VecEnv through VecNormalize
        inner_venv = getattr(vec_norm, "venv", vec_norm)
        inner_venv.env_method("set_reward_weight", self.attr_name, value)
        self._last_set_value = value

    def _on_training_start(self) -> None:
        self._set_env_attr(self.start_value)
        logger.info(
            "RewardRampCallback: ramping %s from %.3f to %.3f over %d timesteps",
            self.attr_name,
            self.start_value,
            self.end_value,
            self.ramp_timesteps,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.ramp_timesteps:
            if self._last_set_value != self.end_value:
                self._set_env_attr(self.end_value)
                logger.info(
                    "RewardRampCallback: ramp complete — %s = %.3f",
                    self.attr_name,
                    self.end_value,
                )
            return True

        progress = self.num_timesteps / self.ramp_timesteps
        current = self.start_value + progress * (self.end_value - self.start_value)

        # Only update every 10k steps to avoid overhead
        quantised = round(current, 3)
        if quantised != self._last_set_value:
            self._set_env_attr(quantised)

        return True


class EvalCollapseEarlyStopCallback(BaseCallback):  # type: ignore[misc]
    """Stop training if eval reward drops significantly from the peak.

    Monitors evaluation results and stops training when the mean reward
    drops below ``(1 - drop_fraction) * peak_reward`` for
    ``patience`` consecutive evaluations.  This prevents wasting compute
    on a policy that has already collapsed past recovery.

    Args:
        eval_callback: The ``EvalCallback`` whose ``evaluations.npz``
            to monitor.
        drop_fraction: Fractional drop from peak that triggers early
            stopping (default 0.3 = 30% drop).
        patience: Number of consecutive below-threshold evaluations
            before stopping (default 3).
        min_evals: Minimum number of evaluations before early stopping
            can activate (default 5).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        eval_callback: Any,
        drop_fraction: float = 0.3,
        patience: int = 3,
        min_evals: int = 5,
        verbose: int = 0,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for EvalCollapseEarlyStopCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.drop_fraction = drop_fraction
        self.patience = patience
        self.min_evals = min_evals
        self._peak_reward = -np.inf
        self._consecutive_drops = 0
        self._last_seen_n_evals = 0

    def _on_step(self) -> bool:
        log_path = getattr(self.eval_callback, "log_path", None)
        if log_path is None:
            return True

        from pathlib import Path

        npz_path = Path(log_path) / "evaluations.npz"
        if not npz_path.exists():
            return True

        data = np.load(str(npz_path))
        eval_rewards = data["results"]  # (n_evals, n_episodes)
        n_evals = eval_rewards.shape[0]

        if n_evals <= self._last_seen_n_evals:
            return True  # No new eval yet
        self._last_seen_n_evals = n_evals

        if n_evals < self.min_evals:
            return True

        mean_rewards = eval_rewards.mean(axis=1)
        current_peak = float(mean_rewards.max())
        self._peak_reward = max(self._peak_reward, current_peak)

        latest_mean = float(mean_rewards[-1])
        threshold = (1.0 - self.drop_fraction) * self._peak_reward

        if self._peak_reward > 0 and latest_mean < threshold:
            self._consecutive_drops += 1
            logger.warning(
                "EvalCollapseEarlyStop: reward %.1f < %.1f (%.0f%% of peak %.1f), consecutive drops: %d/%d",
                latest_mean,
                threshold,
                100 * (1 - self.drop_fraction),
                self._peak_reward,
                self._consecutive_drops,
                self.patience,
            )
            if self._consecutive_drops >= self.patience:
                logger.warning(
                    "EvalCollapseEarlyStop: stopping training at step %d — reward collapsed from peak %.1f to %.1f",
                    self.num_timesteps,
                    self._peak_reward,
                    latest_mean,
                )
                return False
        else:
            self._consecutive_drops = 0

        return True


class SaveVecNormalizeCallback(BaseCallback):  # type: ignore[misc]
    """Save VecNormalize stats whenever triggered.

    Intended for use as ``callback_on_new_best`` in SB3's ``EvalCallback``
    so that the VecNormalize wrapper is saved alongside ``best_model.zip``.
    This ensures the observation normalization statistics match the policy
    weights when the best model is loaded for evaluation or next-stage
    curriculum training.

    Example::

        save_vecnorm_cb = SaveVecNormalizeCallback(
            save_path=str(model_dir / "best_model_vecnorm.pkl"),
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=save_vecnorm_cb,
            ...
        )

    Args:
        save_path: Destination path for the VecNormalize ``.pkl`` file.
        verbose: Verbosity level.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for SaveVecNormalizeCallback.")
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(self.save_path)
            logger.info("VecNormalize saved to: %s", self.save_path)
        return True


def load_vecnorm_stats(vecnorm_path: str, train_env, eval_env=None) -> bool:
    """Load VecNormalize running statistics from a previous stage into new envs.

    Only observation normalization (``obs_rms``) is carried forward.  Return
    normalization (``ret_rms``) is deliberately **reset** because the reward
    distribution changes between curriculum stages (new reward components,
    different weight magnitudes).  Carrying stale ``ret_rms`` produces badly
    scaled normalized rewards that destabilise policy gradients during the
    critical first updates of a new stage.

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

    # Carry forward observation statistics — the observation space is identical
    # across stages, so the running mean/var remain valid.
    train_env.obs_rms = prev_norm.obs_rms
    # Reset ret_rms: reward distribution changes between stages, so stale
    # return statistics would produce incorrectly scaled normalised rewards.
    train_env.training = True
    train_env.norm_reward = True
    logger.info("obs_rms carried forward; ret_rms reset (reward distribution changed)")

    if eval_env is not None:
        eval_env.obs_rms = prev_norm.obs_rms.copy()
        eval_env.training = False
        eval_env.norm_reward = False

    return True
