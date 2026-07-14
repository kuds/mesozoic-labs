"""Stage-aware Stable-Baselines3 evaluation diagnostics.

The rollout diagnostics in :mod:`environments.shared.diagnostics` describe
what the training workers are doing.  Plateau decisions belong here instead:
they use deterministic evaluation episodes and the same configured metrics
that define each curriculum stage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
    from stable_baselines3.common.callbacks import EvalCallback as _EvalCallback

    _SB3_AVAILABLE = True
except ImportError:
    _BaseCallback = object  # type: ignore[misc,assignment]
    _EvalCallback = object  # type: ignore[misc,assignment]
    _SB3_AVAILABLE = False


def success_metric_applicable(stage_config: dict[str, Any]) -> bool:
    """Return whether task success is an active gate for *stage_config*.

    Stage number alone is deliberately not used: the TOML curriculum config
    is the authority, and a genuine zero remains meaningful whenever a
    positive success-rate threshold is configured.
    """

    curriculum = stage_config.get("curriculum_kwargs", {})
    return float(curriculum.get("min_success_rate", 0.0)) > 0.0


class StageAwareEvalCallback(_EvalCallback):  # type: ignore[misc]
    """SB3 ``EvalCallback`` with stage-aware success and velocity capture.

    Stable-Baselines3 prints a success rate whenever an environment emits
    terminal ``info["is_success"]``.  The dinosaur environments intentionally
    keep emitting that standard field, but success is only an evaluation
    metric in stages whose config has a positive success-rate gate.  This
    callback suppresses collection in other stages and prints an explicit
    ``N/A`` instead of a misleading ``0.00%``.

    Mean forward velocity is also captured for every evaluation episode so
    the plateau diagnostic can follow the Stage 2 locomotion gate without
    launching another evaluation pass.
    """

    def __init__(
        self,
        eval_env: Any,
        *,
        stage: int,
        success_applicable: bool,
        **kwargs: Any,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for StageAwareEvalCallback. Install with: pip install stable-baselines3"
            )
        super().__init__(eval_env, **kwargs)
        self.stage = stage
        self.success_applicable = success_applicable
        self.evaluations_forward_velocities: list[list[float]] = []
        self._episode_forward_sums: dict[int, float] = {}
        self._episode_forward_counts: dict[int, int] = {}
        self._current_eval_forward_velocities: list[float] = []

    def _reset_forward_velocity_capture(self) -> None:
        self._episode_forward_sums.clear()
        self._episode_forward_counts.clear()
        self._current_eval_forward_velocities = []

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """Capture evaluation velocity and conditionally delegate success."""

        info = locals_["info"]
        env_index = int(locals_.get("i", 0))
        forward_vel = info.get("forward_vel")
        if forward_vel is not None:
            value = float(forward_vel)
            if math.isfinite(value):
                self._episode_forward_sums[env_index] = self._episode_forward_sums.get(env_index, 0.0) + value
                self._episode_forward_counts[env_index] = self._episode_forward_counts.get(env_index, 0) + 1

        if locals_["done"]:
            count = self._episode_forward_counts.pop(env_index, 0)
            total = self._episode_forward_sums.pop(env_index, 0.0)
            self._current_eval_forward_velocities.append(total / count if count else float("nan"))

        if self.success_applicable:
            super()._log_success_callback(locals_, globals_)

    def _on_step(self) -> bool:
        evaluation_due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluation_due:
            self._reset_forward_velocity_capture()

        continue_training = super()._on_step()

        if evaluation_due:
            self.evaluations_forward_velocities.append(list(self._current_eval_forward_velocities))
            if not self.success_applicable and self.verbose >= 1:
                print(f"Success rate: N/A (not an active Stage {self.stage} gate)")

        return bool(continue_training)


@dataclass(frozen=True)
class _GateMetric:
    key: str
    label: str
    threshold: float
    value: float | None
    sample_count: int
    percent: bool = False


class StageGatePlateauCallback(_BaseCallback):  # type: ignore[misc]
    """Warn once when the currently blocking evaluation gate stops moving.

    A plateau is a small range over the latest ``plateau_window`` deterministic
    evaluations, normalized by that metric's configured gate.  The callback
    follows the most behaviorally specific failing gate: task success, forward
    velocity, episode length, then reward.  It disarms after meaningful
    movement or a gate pass and can warn again only after a later plateau.

    This callback is diagnostic only.  It never stops training and is separate
    from both curriculum advancement and collapse early stopping.
    """

    _METRIC_PRIORITY = (
        "mean_success_rate",
        "mean_forward_vel",
        "mean_episode_length",
        "mean_reward",
    )

    _GUIDANCE = {
        "mean_success_rate": (
            "Inspect target reachability, contact/proximity signals, and the Stage 3 reward ramp before tuning the optimizer."
        ),
        "mean_forward_vel": (
            "Inspect gait stability, actuator saturation, and locomotion reward balance before tuning the optimizer."
        ),
        "mean_episode_length": (
            "Inspect termination reasons, posture, and balance stability before tuning the optimizer."
        ),
        "mean_reward": (
            "Inspect reward components and policy diagnostics to identify which behavior has stopped improving."
        ),
    }

    def __init__(
        self,
        eval_callback: StageAwareEvalCallback,
        *,
        stage: int,
        curriculum_kwargs: dict[str, Any],
        plateau_window: int = 5,
        min_relative_variation: float = 0.05,
        verbose: int = 0,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for StageGatePlateauCallback. "
                "Install with: pip install stable-baselines3"
            )
        if plateau_window < 2:
            raise ValueError("plateau_window must be at least 2 evaluations")
        if min_relative_variation < 0.0:
            raise ValueError("min_relative_variation must be non-negative")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.stage = stage
        self.curriculum_kwargs = dict(curriculum_kwargs)
        self.plateau_window = plateau_window
        self.min_relative_variation = min_relative_variation
        self._last_seen_n_evals = 0
        self._histories: dict[str, list[float]] = {key: [] for key in self._METRIC_PRIORITY}
        self._plateau_active = False
        self._plateau_metric: str | None = None

    @staticmethod
    def _finite_mean(values: Any) -> float | None:
        array = np.asarray(values, dtype=float).reshape(-1)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return None
        return float(np.mean(finite))

    @staticmethod
    def _finite_count(values: Any) -> int:
        array = np.asarray(values, dtype=float).reshape(-1)
        return int(np.count_nonzero(np.isfinite(array)))

    def _metrics_for_evaluation(self, index: int) -> list[_GateMetric]:
        rewards = self.eval_callback.evaluations_results
        lengths = self.eval_callback.evaluations_length
        velocities = self.eval_callback.evaluations_forward_velocities
        successes = self.eval_callback.evaluations_successes

        samples: dict[str, Any | None] = {
            "mean_reward": rewards[index] if index < len(rewards) else None,
            "mean_episode_length": lengths[index] if index < len(lengths) else None,
            "mean_forward_vel": velocities[index] if index < len(velocities) else None,
            "mean_success_rate": successes[index] if index < len(successes) else None,
        }
        values = {key: self._finite_mean(sample) if sample is not None else None for key, sample in samples.items()}
        counts = {key: self._finite_count(sample) if sample is not None else 0 for key, sample in samples.items()}

        configured = (
            _GateMetric(
                "mean_success_rate",
                "success rate",
                float(self.curriculum_kwargs.get("min_success_rate", 0.0)),
                values["mean_success_rate"],
                counts["mean_success_rate"],
                percent=True,
            ),
            _GateMetric(
                "mean_forward_vel",
                "forward velocity",
                float(self.curriculum_kwargs.get("min_avg_forward_vel", 0.0)),
                values["mean_forward_vel"],
                counts["mean_forward_vel"],
            ),
            _GateMetric(
                "mean_episode_length",
                "episode length",
                float(self.curriculum_kwargs.get("min_avg_episode_length", 0.0)),
                values["mean_episode_length"],
                counts["mean_episode_length"],
            ),
            _GateMetric(
                "mean_reward",
                "mean reward",
                float(self.curriculum_kwargs.get("min_avg_reward", -math.inf)),
                values["mean_reward"],
                counts["mean_reward"],
            ),
        )
        return [
            metric
            for metric in configured
            if math.isfinite(metric.threshold) and (metric.key == "mean_reward" or metric.threshold > 0.0)
        ]

    def _clear_plateau(self, reason: str) -> None:
        if self._plateau_active:
            logger.info(
                "Stage %d evaluation plateau cleared at step %d: %s.",
                self.stage,
                self.num_timesteps,
                reason,
            )
        self._plateau_active = False
        self._plateau_metric = None

    @staticmethod
    def _format_value(metric: _GateMetric, value: float) -> str:
        if metric.percent:
            return f"{value:.1%}"
        if metric.key == "mean_forward_vel":
            return f"{value:.3f} m/s"
        if metric.key == "mean_episode_length":
            return f"{value:.1f} steps"
        return f"{value:.3f}"

    @staticmethod
    def _gate_scale(threshold: float) -> float:
        """Return the gate magnitude, with a stable scale for a zero gate."""

        return abs(threshold) if threshold != 0.0 else 1.0

    def _process_evaluation(self, index: int) -> None:
        metrics = self._metrics_for_evaluation(index)
        if not metrics:
            return

        min_eval_episodes = int(self.curriculum_kwargs.get("min_eval_episodes", 10))
        reward_metric = next((metric for metric in metrics if metric.key == "mean_reward"), None)
        if reward_metric is not None:
            self.logger.record("diagnostics/eval_episode_count", reward_metric.sample_count)
        if any(metric.value is None or metric.sample_count < min_eval_episodes for metric in metrics):
            self.logger.record("diagnostics/eval_gate_data_complete", 0.0)
            return

        self.logger.record("diagnostics/eval_gate_data_complete", 1.0)
        for metric in metrics:
            assert metric.value is not None
            self._histories[metric.key].append(metric.value)

        failing = [metric for metric in metrics if metric.value is not None and metric.value < metric.threshold]
        if not failing:
            self.logger.record("diagnostics/eval_gate_met", 1.0)
            self.logger.record("diagnostics/eval_plateau_active", 0.0)
            self._clear_plateau("all active stage gates are currently met")
            return

        self.logger.record("diagnostics/eval_gate_met", 0.0)
        blocking = next(metric for key in self._METRIC_PRIORITY for metric in failing if metric.key == key)
        assert blocking.value is not None

        if self._plateau_metric is not None and self._plateau_metric != blocking.key:
            self._clear_plateau(f"the blocking gate changed to {blocking.label}")
        self._plateau_metric = blocking.key

        history = self._histories[blocking.key]
        if len(history) < self.plateau_window:
            self.logger.record("diagnostics/eval_plateau_active", 0.0)
            return

        recent = history[-self.plateau_window :]
        absolute_range = max(recent) - min(recent)
        gate_scale = self._gate_scale(blocking.threshold)
        relative_range = absolute_range / gate_scale
        self.logger.record("diagnostics/eval_plateau_relative_range", relative_range)
        relative_margin = (blocking.value - blocking.threshold) / gate_scale
        self.logger.record("diagnostics/eval_gate_relative_margin", relative_margin)

        if relative_range >= self.min_relative_variation:
            self._clear_plateau(
                f"{blocking.label} moved by {relative_range:.1%} of its gate over the evaluation window"
            )
        elif not self._plateau_active:
            logger.warning(
                "STAGE %d EVALUATION PLATEAU: %s stayed within a %s range (%s of its gate) "
                "across the last %d evaluations; latest %s, required %s, step %d. %s "
                "Training continues; this diagnostic is not the curriculum gate or collapse early-stop.",
                self.stage,
                blocking.label,
                self._format_value(blocking, absolute_range),
                f"{relative_range:.1%}",
                self.plateau_window,
                self._format_value(blocking, blocking.value),
                self._format_value(blocking, blocking.threshold),
                self.num_timesteps,
                self._GUIDANCE[blocking.key],
            )
            self._plateau_active = True

        self.logger.record("diagnostics/eval_plateau_active", float(self._plateau_active))

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None)
        if results is None:
            return True

        n_evals = len(results)
        if n_evals < self._last_seen_n_evals:
            self._last_seen_n_evals = 0
            self._histories = {key: [] for key in self._METRIC_PRIORITY}
            self._clear_plateau("evaluation history was reset")

        for index in range(self._last_seen_n_evals, n_evals):
            self._process_evaluation(index)
        self._last_seen_n_evals = n_evals
        return True


def build_stage_evaluation_callbacks(
    eval_env: Any,
    *,
    stage: int,
    stage_config: dict[str, Any],
    diagnostics_verbose: int = 0,
    **eval_callback_kwargs: Any,
) -> tuple[StageAwareEvalCallback, StageGatePlateauCallback]:
    """Build the paired evaluation and stage-gate diagnostic callbacks."""

    curriculum_kwargs = stage_config.get("curriculum_kwargs", {})
    eval_callback = StageAwareEvalCallback(
        eval_env,
        stage=stage,
        success_applicable=success_metric_applicable(stage_config),
        **eval_callback_kwargs,
    )
    plateau_callback = StageGatePlateauCallback(
        eval_callback,
        stage=stage,
        curriculum_kwargs=curriculum_kwargs,
        plateau_window=int(curriculum_kwargs.get("diagnostics_plateau_window", 5)),
        min_relative_variation=float(curriculum_kwargs.get("diagnostics_plateau_min_relative_variation", 0.05)),
        verbose=diagnostics_verbose,
    )
    return eval_callback, plateau_callback
