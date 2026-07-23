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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from environments.shared.config import load_all_stages
from environments.shared.metrics import LocomotionMetrics, env_dt
from environments.shared.wandb_integration import log_eval_metrics

if TYPE_CHECKING:
    from environments.shared.plant_contract import PlantIdentity

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
        if "min_eval_episodes" in cur:
            threshold_fields["min_eval_episodes"] = cur["min_eval_episodes"]
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
    a redundant full eval pass. ``StageAwareEvalCallback`` also supplies
    per-episode forward velocity and task success from that same sample.
    A small supplementary eval remains for the richer locomotion report and
    as a compatibility fallback for a plain SB3 ``EvalCallback``.

    Args:
        curriculum_manager: The manager tracking stage progress.
        eval_env: Vectorized evaluation environment.
        eval_freq: Evaluate every N training steps.
        n_eval_episodes: Number of episodes per evaluation (used only
            when no *eval_callback* is provided).
        eval_callback: Optional ``EvalCallback`` to read results from.
            When set, the callback reads reward/length (and per-episode
            successes, when the env reports ``info["is_success"]``) from
            evaluations.npz. A ``StageAwareEvalCallback`` additionally
            provides forward velocity in memory.
        supplementary_episodes: Number of episodes for the supplementary
            eval when *eval_callback* is provided (default 10).  Drives the
            locomotion report and the compatibility fallback samples.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        eval_env: Any,
        eval_freq: int = 50000,
        n_eval_episodes: int = 30,
        eval_callback: Any = None,
        supplementary_episodes: int = 10,
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

    def _success_rates_for_stage(
        self,
        preferred: list[float] | None,
        fallback: list[float] | None,
    ) -> list[float] | None:
        """Select success samples only when the current stage gates on them."""

        if self.curriculum_manager.current_threshold.min_success_rate <= 0.0:
            return None
        return preferred if preferred else fallback

    def _forward_velocities_for_eval(
        self,
        n_evals: int,
        fallback: list[float] | None,
    ) -> list[float] | None:
        """Use the main evaluation sample for gating, with legacy fallback."""

        histories = getattr(self.eval_callback, "evaluations_forward_velocities", None)
        if histories is not None and n_evals > 0 and len(histories) >= n_evals:
            latest = [float(value) for value in histories[n_evals - 1] if np.isfinite(value)]
            if latest:
                return latest
        return fallback

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        if self.eval_callback is not None:
            return self._on_step_with_eval_callback()
        return self._on_step_standalone()

    def _read_latest_eval(self) -> tuple:
        """Read the latest per-episode rewards/lengths/successes from EvalCallback's npz.

        Returns ``(rewards, lengths, successes, n_evals)`` where *successes*
        is ``None`` when the npz has no ``successes`` array (envs that never
        emit ``info["is_success"]``), or ``(None, None, None, 0)`` if no new
        evaluation data is available.
        """
        log_path = getattr(self.eval_callback, "log_path", None)
        if log_path is None:
            return None, None, None, 0

        from pathlib import Path

        # SB3's EvalCallback stores log_path as the "<dir>/evaluations"
        # file prefix (np.savez appends ".npz"), not the directory.
        npz_path = Path(str(log_path) + ".npz")
        if not npz_path.exists():
            return None, None, None, 0

        data = np.load(str(npz_path))
        eval_rewards = data["results"]  # (n_evals, n_episodes)
        eval_lengths = data["ep_lengths"]

        n_evals = eval_rewards.shape[0]
        if n_evals <= self._last_seen_n_evals:
            return None, None, None, n_evals  # No new eval

        self._last_seen_n_evals = n_evals
        latest_rewards = eval_rewards[-1].tolist()
        latest_lengths = eval_lengths[-1].tolist()
        # SB3 saves a "successes" array when the env reports
        # info["is_success"] at episode end (BaseDinoEnv does).  Using it
        # gives the success-rate gate the full n_eval_episodes sample
        # instead of the small supplementary eval.
        latest_successes = None
        if "successes" in data.files and data["successes"].shape[0] == n_evals:
            latest_successes = [float(s) for s in data["successes"][-1]]
        return latest_rewards, latest_lengths, latest_successes, n_evals

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
            eval_dt = env_dt(self.eval_env)
            for _ in range(self.supplementary_episodes):
                obs = self.eval_env.reset()
                metrics = LocomotionMetrics(dt=eval_dt)
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
        rewards, lengths, npz_successes, n_evals = self._read_latest_eval()
        if rewards is None:
            return True  # EvalCallback hasn't produced new results yet

        # Run supplementary eval for forward_vel / locomotion metrics
        forward_vels, success_flags, episode_reports = self._run_supplementary_eval()
        self._log_locomotion_metrics(episode_reports)

        fwd_vel_arg = self._forward_velocities_for_eval(n_evals, forward_vels or None)
        # Prefer the EvalCallback's per-episode successes (full
        # n_eval_episodes sample) over the small supplementary eval, but only
        # when task success is an active gate for this stage.  Incidental
        # target contacts in balance/locomotion are not stage success.
        success_arg = self._success_rates_for_stage(npz_successes, success_flags or None)
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
            eval_dt = env_dt(self.eval_env)
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                metrics = LocomotionMetrics(dt=eval_dt)
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
        success_arg = self._success_rates_for_stage(None, success_flags or None)
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

    * Reduces the ``learning_rate`` to ``warmup_lr_scale`` × original
      (default 0.1×).  Note SB3 applies ``lr_schedule`` to **all** SAC
      optimizers (actor, critics, and entropy coefficient alike), so the
      whole agent updates slowly while adapting to the new reward landscape.
    * Re-seeds the auto-tuned entropy coefficient at ``warmup_ent_coef`` by
      writing ``log_ent_coef`` directly (SAC's ``train()`` reads
      ``log_ent_coef``, not the ``ent_coef`` attribute, so setting the
      attribute alone has no effect).  Auto-tuning continues from that
      value — at the reduced warm-up LR — and is NOT rolled back when the
      warm-up ends, so tuning progress made during warm-up is kept.

    After ``warmup_timesteps`` have elapsed the original learning rate
    schedule (and, for PPO, clip range / entropy coefficient) is restored.

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
        self._is_sac = False
        self._warmup_done = False

    def _on_training_start(self) -> None:
        self._is_sac = hasattr(self.model, "log_ent_coef")

        if self._is_sac:
            # SAC warmup: reduce LR and re-seed the entropy coefficient.
            # Store the original lr_schedule callable (not the raw learning_rate
            # float) so we can restore it exactly after warmup.
            self._original_lr_schedule = self.model.lr_schedule
            original_lr = self._original_lr_schedule(1.0)
            warmup_lr = original_lr * self.warmup_lr_scale
            self.model.lr_schedule = _ConstantSchedule(warmup_lr)
            # SAC's train() reads log_ent_coef (auto mode), NOT the ent_coef
            # attribute — write the warm-up value into the learned tensor so
            # it actually takes effect.  Auto-tuning then continues from
            # there (at the reduced warm-up LR) and is deliberately not
            # rolled back at the end of the warm-up: clobbering it would
            # discard the tuning progress made during the warm-up window.
            import math as _math

            self.model.log_ent_coef.data.fill_(_math.log(self.warmup_ent_coef))
            logger.info(
                "StageWarmupCallback [SAC]: warm-up active for %d timesteps (lr=%.2e → %.2e, ent_coef seeded at %.3f)",
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
                # log_ent_coef is intentionally left at its current
                # (auto-tuned) value — see _on_training_start.
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
        self._last_update_bucket = -1

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

        # Only update on 10k-step boundaries: each update is an env_method
        # RPC round-trip to every SubprocVecEnv worker, so a value-based
        # check would broadcast every few dozen steps on a long ramp.
        bucket = self.num_timesteps // 10_000
        if bucket == self._last_update_bucket:
            return True
        self._last_update_bucket = bucket

        progress = self.num_timesteps / self.ramp_timesteps
        current = self.start_value + progress * (self.end_value - self.start_value)

        quantised = round(current, 3)
        if quantised != self._last_set_value:
            self._set_env_attr(quantised)

        return True


class EvalCollapseEarlyStopCallback(BaseCallback):  # type: ignore[misc]
    """Stop training if the eval reward collapses from a robust peak.

    The peak candidate is the median of each full trailing window of
    per-evaluation mean rewards. The historical maximum of those candidates
    is robust to a single variance-inflated evaluation, unlike a raw or
    moving-mean peak. Detection waits until both ``min_evals`` and one full
    ``smoothing_window`` are available.

    Once armed, each new evaluation contributes exactly one patience
    observation: its raw per-evaluation mean is compared with
    ``(1 - drop_fraction) * peak_rolling_median``. Using the raw current mean
    prevents one low outlier from being counted repeatedly through several
    overlapping windows. It also preserves the healthy central tendency of
    a bimodal gait-transition evaluation, where ``mean - std`` can collapse
    even though the episode mean remains healthy.

    Detection stays disarmed until the rolling-median peak clears
    ``peak_floor`` (wired to the stage's ``min_avg_reward`` curriculum gate
    by ``_build_core_callbacks``): a run can only "collapse" from a level
    that was actually good, so the pre-convergence grind cannot trip the
    backstop.

    Args:
        eval_callback: The ``EvalCallback`` whose ``evaluations.npz``
            to monitor.
        drop_fraction: Fractional drop from the rolling-median peak that
            counts as a collapse evaluation (default 0.3 = 30% drop).
        patience: Number of consecutive below-threshold evaluations
            before stopping (default 3).
        min_evals: Minimum number of evaluations before early stopping
            can activate (default 5).
        peak_floor: Minimum rolling-median peak before collapse detection
            arms (default 0.0 — arm on any positive peak).
        verbose: Verbosity level.
        smoothing_window: Number of per-evaluation means in each full
            rolling-median peak window (default 5). Keyword-only so the
            historical positional slot for ``verbose`` remains stable.
    """

    def __init__(
        self,
        eval_callback: Any,
        drop_fraction: float = 0.3,
        patience: int = 3,
        min_evals: int = 5,
        peak_floor: float = 0.0,
        verbose: int = 0,
        *,
        smoothing_window: int = 5,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for EvalCollapseEarlyStopCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.drop_fraction = drop_fraction
        self.patience = patience
        self.min_evals = min_evals
        self.peak_floor = peak_floor
        self.smoothing_window = max(1, int(smoothing_window))
        self._peak_score = -np.inf
        self._consecutive_drops = 0
        self._last_seen_n_evals = 0

    def _on_step(self) -> bool:
        # EvalCallback keeps per-eval episode rewards in memory (populated
        # whenever its log_path is set); using it avoids re-opening
        # evaluations.npz — a GCS FUSE file on Vertex AI — on every single
        # training step.
        results = getattr(self.eval_callback, "evaluations_results", None)
        if not results:
            return True

        n_evals = len(results)

        if n_evals <= self._last_seen_n_evals:
            return True  # No new eval yet
        self._last_seen_n_evals = n_evals

        window = self.smoothing_window
        if n_evals < max(self.min_evals, window):
            return True

        # A full-window median prevents one high outlier from setting the
        # reference peak. Only the peak is windowed: the latest raw eval mean
        # supplies one (and only one) patience observation.
        eval_means = np.array([float(np.mean(r)) for r in results])
        rolling_medians = np.array(
            [float(np.median(eval_means[i - window + 1 : i + 1])) for i in range(window - 1, len(eval_means))]
        )
        self._peak_score = float(rolling_medians.max())

        latest_mean = float(eval_means[-1])
        threshold = (1.0 - self.drop_fraction) * self._peak_score
        armed = self._peak_score > 0 and self._peak_score >= self.peak_floor

        if armed and latest_mean < threshold:
            self._consecutive_drops += 1
            logger.warning(
                "EvalCollapseEarlyStop: eval mean %.1f < %.1f (%.0f%% of rolling-median peak %.1f), "
                "consecutive drops: %d/%d",
                latest_mean,
                threshold,
                100 * (1 - self.drop_fraction),
                self._peak_score,
                self._consecutive_drops,
                self.patience,
            )
            if self._consecutive_drops >= self.patience:
                logger.warning(
                    "EvalCollapseEarlyStop: stopping training at step %d — eval mean collapsed "
                    "from peak %.1f to %.1f",
                    self.num_timesteps,
                    self._peak_score,
                    latest_mean,
                )
                return False
        else:
            self._consecutive_drops = 0

        return True


def build_eval_collapse_early_stop_callback(
    eval_callback: Any,
    curriculum_kwargs: dict[str, Any],
    *,
    verbose: int = 0,
) -> EvalCollapseEarlyStopCallback:
    """Build the shared eval-collapse backstop from curriculum settings."""
    return EvalCollapseEarlyStopCallback(
        eval_callback=eval_callback,
        min_evals=int(curriculum_kwargs.get("collapse_min_evals", 12)),
        patience=int(curriculum_kwargs.get("collapse_patience", 8)),
        drop_fraction=float(curriculum_kwargs.get("collapse_drop_fraction", 0.4)),
        peak_floor=float(curriculum_kwargs.get("collapse_peak_floor", curriculum_kwargs.get("min_avg_reward", 0.0))),
        verbose=verbose,
        smoothing_window=int(curriculum_kwargs.get("collapse_smoothing_window", 5)),
    )


class RobustBestModelCallback(BaseCallback):  # type: ignore[misc]
    """Save the checkpoint with the best risk-adjusted evaluation score.

    SB3's ``EvalCallback`` saves ``best_model.zip`` by mean eval reward.
    In run 20260709_185946 that selected the 800k-step checkpoint whose
    eval distribution was already bimodal — a high mean propped up by
    27/30 good episodes while the catastrophic-fall fraction was growing
    — when the genuinely robust checkpoint (~450k steps, zero failures)
    scored a slightly lower mean. This callback scores each evaluation as
    ``mean - risk_coef * std`` and saves ``robust_best_model.zip`` (plus
    its matching VecNormalize stats) whenever the score improves, so
    next-stage training and gating can prefer consistently-good policies
    over great-on-average ones with a fat failure tail.

    Place after the ``EvalCallback`` in the callback list.

    Args:
        eval_callback: The ``EvalCallback`` whose in-memory eval results
            to score.
        model_dir: Directory to save ``robust_best_model.zip`` into.
        risk_coef: Weight on the per-episode std in the score
            (default 1.0, i.e. one standard deviation below the mean).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        eval_callback: Any,
        model_dir: "str | Path",
        risk_coef: float = 1.0,
        verbose: int = 0,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RobustBestModelCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.model_dir = Path(model_dir)
        self.risk_coef = risk_coef
        self.best_score = -np.inf
        self._last_seen_n = 0

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None) or []
        n = len(results)
        if n <= self._last_seen_n:
            return True
        self._last_seen_n = n

        latest = np.asarray(results[-1], dtype=float)
        score = float(latest.mean() - self.risk_coef * latest.std())
        if score <= self.best_score:
            return True
        self.best_score = score

        path = self.model_dir / "robust_best_model"
        self.model.save(str(path))
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(str(path) + "_vecnorm.pkl")
        logger.info(
            "RobustBestModel: new best risk-adjusted score %.1f (mean %.1f - %.1f*std %.1f) at step %d",
            score,
            latest.mean(),
            self.risk_coef,
            latest.std(),
            self.num_timesteps,
        )
        return True


class EntCoefDecayCallback(BaseCallback):  # type: ignore[misc]
    """Linearly decay PPO's entropy coefficient during a stage.

    In run 20260709_185946 the policy's action std grew 1.18 → 1.49 under
    a constant ``ent_coef`` while the eval failure fraction climbed: the
    entropy bonus keeps paying for exploration noise long after the gait
    needs consolidation. This callback decays ``model.ent_coef`` from its
    initial value to ``end_value`` over ``decay_timesteps``, then holds.

    PPO only — SAC's auto-tuned entropy already adapts on its own.

    Args:
        end_value: Final entropy coefficient.
        decay_timesteps: Timesteps over which to decay (typically the
            stage budget).
        verbose: Verbosity level.
    """

    def __init__(self, end_value: float = 0.0005, decay_timesteps: int = 4_000_000, verbose: int = 0):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for EntCoefDecayCallback.")
        super().__init__(verbose)
        self.end_value = end_value
        self.decay_timesteps = max(1, decay_timesteps)
        self._initial: float | None = None

    def _on_training_start(self) -> None:
        self._initial = float(self.model.ent_coef)
        logger.info(
            "EntCoefDecay: ent_coef %.4f → %.4f over %d timesteps",
            self._initial,
            self.end_value,
            self.decay_timesteps,
        )

    def _on_step(self) -> bool:
        if self._initial is None:
            return True
        frac = min(1.0, self.num_timesteps / self.decay_timesteps)
        self.model.ent_coef = self._initial + frac * (self.end_value - self._initial)
        return True


class PublishEvalArtifactsCallback(BaseCallback):  # type: ignore[misc]
    """Atomically publish EvalCallback's ``evaluations.npz`` to the stage dir.

    The paired ``EvalCallback`` writes its npz to fast local scratch;
    this callback copies it to the (possibly Drive/GCS-FUSE mounted)
    stage directory after every new evaluation and again at training
    end, via copy-to-temp + rename, so the published file is never
    observed truncated. Place it *after* the ``EvalCallback`` in the
    callback list so it runs on the same step a new eval completes.

    Args:
        eval_callback: The ``EvalCallback`` whose npz to publish.
        publish_dir: Directory to publish ``evaluations.npz`` into
            (typically the stage directory).
        verbose: Verbosity level.
    """

    def __init__(self, eval_callback: Any, publish_dir: "str | Path", verbose: int = 0):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PublishEvalArtifactsCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.publish_dir = Path(publish_dir)
        self._last_published_n = 0

    def _source_npz(self) -> "Path | None":
        # SB3 stores log_path as the "<dir>/evaluations" file prefix.
        prefix = getattr(self.eval_callback, "log_path", None)
        return Path(str(prefix) + ".npz") if prefix else None

    def _publish(self) -> None:
        from .file_io import atomic_copy

        src = self._source_npz()
        if src is not None and src.exists():
            atomic_copy(src, self.publish_dir / "evaluations.npz")

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None) or []
        if len(results) > self._last_published_n:
            self._last_published_n = len(results)
            self._publish()
        return True

    def _on_training_end(self) -> None:
        self._publish()


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


def load_vecnorm_stats(
    vecnorm_path: str,
    train_env,
    eval_env=None,
    *,
    current_plant: PlantIdentity | None = None,
    allow_legacy_plant: bool = False,
    unsafe_skip_plant_validation: bool = False,
) -> bool:
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
            Its existing ``training`` / ``norm_reward`` flags are preserved
            (set by ``create_vec_env`` based on the algorithm — SAC keeps
            ``norm_reward=False`` to avoid replay-buffer reward-scale drift).
        eval_env: Optional evaluation ``VecNormalize`` wrapper.
            ``training`` is set to ``False``; ``norm_reward`` is disabled.
        current_plant: Expected plant identity. Required unless the explicit
            unsafe inspection-only escape hatch is enabled.
        allow_legacy_plant: Explicitly allow a pre-contract normalization
            artifact with no identity.  Incompatible tagged artifacts are
            always rejected.
        unsafe_skip_plant_validation: Deliberately load without checking the
            plant. This is for low-level inspection/tests only and cannot be
            combined with either validation option.

    Returns:
        ``True`` if stats were loaded, ``False`` if the file was not found.
    """
    from pathlib import Path as _Path

    from environments.shared.plant_contract import PlantCompatibilityError

    if unsafe_skip_plant_validation:
        if current_plant is not None or allow_legacy_plant:
            raise ValueError("unsafe_skip_plant_validation cannot be combined with current_plant or allow_legacy_plant")
        logger.warning(
            "UNSAFE: loading %s without plant compatibility validation; "
            "do not use these statistics for training or evaluation until verified",
            vecnorm_path,
        )
    elif current_plant is None:
        raise PlantCompatibilityError(
            f"refusing to load {vecnorm_path} without current_plant; pass the current PlantIdentity "
            "or use unsafe_skip_plant_validation=True only for deliberate low-level inspection"
        )

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
    if not unsafe_skip_plant_validation:
        from environments.shared.plant_contract import validate_model_plant

        assert current_plant is not None
        # Validate before copying obs_rms so an incompatible artifact cannot
        # partially mutate either destination environment.
        validate_model_plant(
            prev_norm,
            current_plant,
            artifact=str(path),
            allow_legacy=allow_legacy_plant,
        )

    # Carry forward observation statistics — the observation space is identical
    # across stages, so the running mean/var remain valid.  ret_rms is
    # intentionally NOT copied: reward distribution changes between stages, so
    # stale return statistics would produce incorrectly scaled normalised
    # rewards for PPO.  train_env.training / norm_reward are left as configured
    # by create_vec_env (algorithm-aware).
    train_env.obs_rms = prev_norm.obs_rms
    logger.info("obs_rms carried forward; ret_rms reset (reward distribution changed)")

    if eval_env is not None:
        eval_env.obs_rms = prev_norm.obs_rms.copy()
        eval_env.training = False
        eval_env.norm_reward = False

    return True
