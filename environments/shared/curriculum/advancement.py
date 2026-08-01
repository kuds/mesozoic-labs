"""SB3 callbacks that drive stage entry, shaping, and advancement.

* :class:`CurriculumCallback` evaluates the stage gate and reports readiness
* :class:`StageWarmupCallback` softens the first updates after a stage change
* :class:`RewardRampCallback` ramps stage reward weights in over time"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from environments.shared.metrics import LocomotionMetrics, env_dt
from environments.shared.stance_diagnostics import derive_stance_info
from environments.shared.wandb_integration import log_eval_metrics

from . import sb3_compat
from .manager import CurriculumManager
from .sb3_compat import BaseCallback
from .schedules import _ConstantSchedule
from .stance_gate import STANCE_GATE_KIND, StancePanel, stance_panel_from_episode_duties

logger = logging.getLogger(__name__)


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
        if not sb3_compat._SB3_AVAILABLE:
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

    def _stance_panel_for_eval(
        self,
        n_evals: int,
        rewards: list[float],
        lengths: list[float],
    ) -> StancePanel | None:
        """Build this evaluation's stance panel, or ``None`` if unavailable.

        Read from ``StageAwareEvalCallback``'s per-episode capture over the
        *main* evaluation pass, so the bound is formed at the panel size it is
        specified for rather than at the 10-episode supplementary sample.
        Returns ``None`` when the stage does not gate on stance or the
        callback did not record duties, which makes the gate fail closed
        rather than advance on evidence nobody collected.
        """
        if self.curriculum_manager.current_threshold.gate_kind != STANCE_GATE_KIND:
            return None

        histories = getattr(self.eval_callback, "evaluations_unsupported_duties", None)
        if histories is None or n_evals <= 0 or len(histories) < n_evals:
            return None
        duties = list(histories[n_evals - 1])
        if not duties:
            return None

        return stance_panel_from_episode_duties(
            episode_lengths=lengths,
            episode_duties=duties,
            episode_rewards=rewards,
            horizon=self._eval_horizon(),
        )

    def _eval_horizon(self) -> int:
        """Episode-step count that counts as reaching the horizon."""
        env_kwargs = self.curriculum_manager.current_config().get("env_kwargs", {})
        return int(env_kwargs.get("max_episode_steps", 1000))

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
        stance_panel = self._stance_panel_for_eval(n_evals, rewards, lengths)
        if self.curriculum_manager.should_advance(rewards, lengths, fwd_vel_arg, success_arg, stance_panel):
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
        # Per-episode unsupported duty, so this path can drive
        # stance_quality/v1 too. Without it a stage on that gate would fail
        # closed forever here — correct, but a silent dead end.
        settle_steps = self.curriculum_manager.current_threshold.settle_steps
        episode_duties: list[float] = []

        try:
            eval_dt = env_dt(self.eval_env)
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                metrics = LocomotionMetrics(dt=eval_dt)
                episode_reward = 0.0
                episode_length = 0
                ep_forward_vels: list[float] = []
                ep_success = 0.0
                unsupported_steps = 0
                measured_steps = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, dones, infos = self.eval_env.step(action)
                    step_reward = float(reward[0])
                    episode_reward += step_reward
                    metrics.record_step(infos[0], step_reward)
                    if "forward_vel" in infos[0]:
                        ep_forward_vels.append(float(infos[0]["forward_vel"]))
                    if episode_length >= settle_steps:
                        stance = derive_stance_info(infos[0])
                        if stance:
                            measured_steps += 1
                            unsupported_steps += int(stance["unsupported_duty"])
                    episode_length += 1
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
                episode_duties.append(unsupported_steps / measured_steps if measured_steps else float("nan"))
                episode_reports.append(metrics.compute())
        finally:
            if old_training is not None:
                self.eval_env.training = old_training
            if old_norm_reward is not None:
                self.eval_env.norm_reward = old_norm_reward

        self._log_locomotion_metrics(episode_reports)

        fwd_vel_arg = forward_vels if forward_vels else None
        success_arg = self._success_rates_for_stage(None, success_flags or None)
        stance_panel = None
        if self.curriculum_manager.current_threshold.gate_kind == STANCE_GATE_KIND:
            stance_panel = stance_panel_from_episode_duties(
                episode_lengths=lengths,
                episode_duties=episode_duties,
                episode_rewards=rewards,
                horizon=self._eval_horizon(),
            )
        if self.curriculum_manager.should_advance(rewards, lengths, fwd_vel_arg, success_arg, stance_panel):
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
        if not sb3_compat._SB3_AVAILABLE:
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
        if not sb3_compat._SB3_AVAILABLE:
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
