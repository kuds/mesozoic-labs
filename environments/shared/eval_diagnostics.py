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

from .stance_diagnostics import derive_stance_info

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

    Per-episode **unsupported duty** is captured the same way, for the
    ``stance_quality/v1`` advancement gate.  It has to come from this pass
    rather than from the small supplementary evaluation: the gate's bound is
    specified at the full evaluation panel size, and reducing it to the
    10-episode supplementary sample would throw away most of the power the
    bound exists to provide.

    The duty signal is :func:`~environments.shared.stance_diagnostics.derive_stance_info`'s
    ``unsupported_duty`` at its default 0.1 N contact threshold -- deliberately
    *not* the reward's 42 N ``min_support_force``.  Every calibration point the
    gate's ceiling rests on is measured at 0.1 N: the statue's 0.000 and run
    ``20260801_021545``'s 0.319.  Switching thresholds here would silently
    shift duty upward and invalidate the 0.02 ceiling.
    """

    def __init__(
        self,
        eval_env: Any,
        *,
        stage: int,
        success_applicable: bool,
        settle_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for StageAwareEvalCallback. Install with: pip install stable-baselines3"
            )
        super().__init__(eval_env, **kwargs)
        self.stage = stage
        self.success_applicable = success_applicable
        self.settle_steps = int(settle_steps)
        self.evaluations_forward_velocities: list[list[float]] = []
        self.evaluations_unsupported_duties: list[list[float]] = []
        # Not a gate criterion -- captured so a run GENERATES the data needed
        # to set one honestly. The three stance shares sum to 1, so a falling
        # unsupported duty says nothing about whether the feet are being
        # planted; that time can land in single support instead. Bounding
        # bilateral duty is the obvious fix, but the only reference point
        # available today is the statue's 0.998, and the statue never shifts
        # weight. A competent active policy legitimately spends some time in
        # single support, and how much is unmeasured. Recording it every
        # evaluation is what turns that into evidence instead of a guess.
        self.evaluations_bilateral_duties: list[list[float]] = []
        self._episode_forward_sums: dict[int, float] = {}
        self._episode_forward_counts: dict[int, int] = {}
        self._current_eval_forward_velocities: list[float] = []
        self._episode_steps: dict[int, int] = {}
        self._episode_unsupported: dict[int, int] = {}
        self._episode_bilateral: dict[int, int] = {}
        self._episode_measured: dict[int, int] = {}
        self._current_eval_unsupported_duties: list[float] = []
        self._current_eval_bilateral_duties: list[float] = []

    def _reset_forward_velocity_capture(self) -> None:
        self._episode_forward_sums.clear()
        self._episode_forward_counts.clear()
        self._current_eval_forward_velocities = []
        self._episode_steps.clear()
        self._episode_unsupported.clear()
        self._episode_bilateral.clear()
        self._episode_measured.clear()
        self._current_eval_unsupported_duties = []
        self._current_eval_bilateral_duties = []

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """Capture evaluation velocity and duty, conditionally delegate success."""

        info = locals_["info"]
        env_index = int(locals_.get("i", 0))
        forward_vel = info.get("forward_vel")
        if forward_vel is not None:
            value = float(forward_vel)
            if math.isfinite(value):
                self._episode_forward_sums[env_index] = self._episode_forward_sums.get(env_index, 0.0) + value
                self._episode_forward_counts[env_index] = self._episode_forward_counts.get(env_index, 0) + 1

        # Duty is accumulated only after the settling window, because a policy
        # that corrects reset randomisation may legitimately move a foot doing
        # it -- and settling is the capability stage 1a exists to certify.
        step_index = self._episode_steps.get(env_index, 0)
        self._episode_steps[env_index] = step_index + 1
        if step_index >= self.settle_steps:
            stance = derive_stance_info(info)
            if stance:
                self._episode_measured[env_index] = self._episode_measured.get(env_index, 0) + 1
                self._episode_unsupported[env_index] = self._episode_unsupported.get(env_index, 0) + int(
                    stance["unsupported_duty"]
                )
                self._episode_bilateral[env_index] = self._episode_bilateral.get(env_index, 0) + int(
                    stance["bilateral_support_duty"]
                )

        if locals_["done"]:
            count = self._episode_forward_counts.pop(env_index, 0)
            total = self._episode_forward_sums.pop(env_index, 0.0)
            self._current_eval_forward_velocities.append(total / count if count else float("nan"))

            measured = self._episode_measured.pop(env_index, 0)
            unsupported = self._episode_unsupported.pop(env_index, 0)
            bilateral = self._episode_bilateral.pop(env_index, 0)
            self._episode_steps.pop(env_index, None)
            # NaN for an episode with no measurable tail (shorter than the
            # settling window, or an env that reports no foot contacts). The
            # panel builder drops those rather than scoring them as zero.
            self._current_eval_unsupported_duties.append(unsupported / measured if measured else float("nan"))
            self._current_eval_bilateral_duties.append(bilateral / measured if measured else float("nan"))

        if self.success_applicable:
            super()._log_success_callback(locals_, globals_)

    def _on_step(self) -> bool:
        evaluation_due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluation_due:
            self._reset_forward_velocity_capture()

        continue_training = super()._on_step()

        if evaluation_due:
            self.evaluations_forward_velocities.append(list(self._current_eval_forward_velocities))
            self.evaluations_unsupported_duties.append(list(self._current_eval_unsupported_duties))
            self.evaluations_bilateral_duties.append(list(self._current_eval_bilateral_duties))
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
    #: True when the threshold is an upper bound the metric must stay under
    #: (``stance_quality/v1``'s duty ceilings) rather than a floor it must
    #: clear. Without the distinction a ceiling read as a floor inverts:
    #: a policy chattering at duty 0.3 against a 0.02 ceiling would be
    #: reported as comfortably passing.
    ceiling: bool = False

    def is_failing(self) -> bool:
        if self.value is None:
            return False
        return self.value > self.threshold if self.ceiling else self.value < self.threshold


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

    # Ordered most behaviourally specific first. The stance metrics sit above
    # episode length and reward because on a stance-gated stage they ARE the
    # gate: without them the callback would follow mean_reward and hand out
    # reward-tuning guidance for a stage whose blocking criterion is foot
    # contact, which is worse than silence.
    _METRIC_PRIORITY = (
        "mean_success_rate",
        "mean_forward_vel",
        "mean_unsupported_duty",
        "full_horizon_fraction",
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
        "mean_unsupported_duty": (
            "Inspect foot contact forces, the action smoothness and jerk penalties, and exploration noise: "
            "the policy is spending too much time with neither foot loaded."
        ),
        "full_horizon_fraction": (
            "Inspect termination reasons and the reset distribution before tuning the optimizer: "
            "episodes are ending early rather than degrading in quality."
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
        horizon: int = 1000,
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
        self.horizon = int(horizon)
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

    def _series(self, attribute: str, index: int) -> Any | None:
        """One evaluation's per-episode series from the capture callback."""
        history = getattr(self.eval_callback, attribute, None) or []
        return history[index] if index < len(history) else None

    def _stance_panel(self, index: int) -> Any | None:
        """This evaluation's stance panel, or ``None`` if it cannot be built.

        Built with the same :mod:`~environments.shared.curriculum.stance_gate`
        reduction the curriculum runs, so what this callback plots is what the
        gate decides. Reducing the duties here independently is how the
        TensorBoard curve came to disagree with the gate by 8x: the scalar
        averaged every episode while the gate averages the full-horizon ones
        only, and on a panel with 5 early failures at duty 0.60 among 35 clean
        episodes at 0.01 that reads 0.084 against a 0.02 ceiling the policy
        was actually clearing.
        """
        from .curriculum.stance_gate import stance_panel_from_episode_duties

        duties = self._series("evaluations_unsupported_duties", index)
        lengths = self._series("evaluations_length", index)
        rewards = self._series("evaluations_results", index)
        if duties is None or lengths is None or rewards is None:
            return None
        if not len(duties) or len(duties) != len(lengths) or len(duties) != len(rewards):
            # Positional alignment is the whole contract; a mismatch pairs each
            # duty with the wrong episode's length.
            return None
        try:
            return stance_panel_from_episode_duties(
                episode_lengths=[float(v) for v in lengths],
                episode_duties=[float(v) for v in duties],
                episode_rewards=[float(v) for v in rewards],
                horizon=self.horizon,
            )
        except (TypeError, ValueError):
            return None

    def _full_horizon_mean(self, attribute: str, index: int) -> float | None:
        """Mean of a per-episode series over full-horizon episodes only.

        The set the gate's duty is measured over.  Averaging a stance share
        over a different set than the gated one is what makes the three
        shares stop summing to 1 and the pair stop being readable together.
        """
        series = self._series(attribute, index)
        lengths = self._series("evaluations_length", index)
        if series is None or lengths is None or len(series) != len(lengths):
            return None
        kept = [
            float(value)
            for value, length in zip(series, lengths)
            if float(length) >= self.horizon and math.isfinite(float(value))
        ]
        return float(np.mean(kept)) if kept else None

    def _metrics_for_evaluation(self, index: int, panel: Any | None = None) -> list[_GateMetric]:
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

        # Duty comes from the panel, not from a second reduction, so the
        # metric the plateau follows is the number the gate compares.
        if panel is not None and panel.n_duty_episodes:
            values["mean_unsupported_duty"] = panel.mean_unsupported_duty
            counts["mean_unsupported_duty"] = panel.n_duty_episodes
        else:
            values["mean_unsupported_duty"] = None
            counts["mean_unsupported_duty"] = 0

        # Derived, not sampled: the fraction of episodes reaching the horizon.
        # It comes from the same per-episode lengths the length gate uses, so
        # no extra capture is needed.
        episode_lengths = samples["mean_episode_length"]
        if episode_lengths is not None and len(episode_lengths):
            finite = [float(v) for v in episode_lengths if math.isfinite(float(v))]
            horizon_fraction = sum(1.0 for v in finite if v >= self.horizon) / len(finite) if finite else None
            horizon_count = len(finite)
        else:
            horizon_fraction, horizon_count = None, 0

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
                "mean_unsupported_duty",
                "unsupported duty",
                # inf when absent, so the isfinite filter drops it for stages
                # that do not gate on stance. A ceiling cannot use 0.0 as its
                # "absent" default the way a floor does -- 0.0 would be the
                # strictest possible bound, not the loosest.
                float(self.curriculum_kwargs.get("max_unsupported_duty", math.inf)),
                values["mean_unsupported_duty"],
                counts["mean_unsupported_duty"],
                ceiling=True,
            ),
            _GateMetric(
                "full_horizon_fraction",
                "full-horizon episodes",
                float(self.curriculum_kwargs.get("min_full_horizon_fraction", 0.0)),
                horizon_fraction,
                horizon_count,
                percent=True,
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
        if metric.key == "mean_unsupported_duty":
            return f"{value:.4f}"
        return f"{value:.3f}"

    @staticmethod
    def _gate_scale(threshold: float) -> float:
        """Return the gate magnitude, with a stable scale for a zero gate."""

        return abs(threshold) if threshold != 0.0 else 1.0

    def _record_stance_scalars(self, index: int, panel: Any | None) -> None:
        """Emit the quantities the stance gate actually decides on.

        All three gating numbers used to be absent from TensorBoard and W&B
        entirely — only a raw all-episode duty mean was published — so a
        stance-gated stage gave no way to tell whether it was blocked on the
        mean or on panel variance driving the bound, which is the single most
        useful thing to know.

        ``eval_duty_episodes`` is recorded even when it is zero: that is the
        early-training signal that no episode survived the settling window,
        and it reads identically to "not evaluated" if it is simply omitted.
        """
        if panel is None:
            return
        self.logger.record("diagnostics/eval_full_horizon_fraction", panel.full_horizon_fraction)
        self.logger.record("diagnostics/eval_duty_episodes", panel.n_duty_episodes)
        if not panel.n_duty_episodes:
            return
        self.logger.record("diagnostics/eval_unsupported_duty", panel.mean_unsupported_duty)
        if math.isfinite(panel.unsupported_duty_ucb):
            self.logger.record("diagnostics/eval_unsupported_duty_ucb", panel.unsupported_duty_ucb)
        # Not gated. Measured over the SAME full-horizon episodes as the duty
        # above so the two are comparable; averaging it over every episode is
        # what made the three stance shares stop summing to 1.
        bilateral = self._full_horizon_mean("evaluations_bilateral_duties", index)
        if bilateral is not None:
            self.logger.record("diagnostics/eval_bilateral_support_duty", bilateral)

    def _process_evaluation(self, index: int) -> None:
        panel = self._stance_panel(index)
        metrics = self._metrics_for_evaluation(index, panel)
        if not metrics:
            return

        min_eval_episodes = int(self.curriculum_kwargs.get("min_eval_episodes", 10))
        reward_metric = next((metric for metric in metrics if metric.key == "mean_reward"), None)
        if reward_metric is not None:
            self.logger.record("diagnostics/eval_episode_count", reward_metric.sample_count)

        self._record_stance_scalars(index, panel)

        # Follow the metrics this evaluation can actually support, rather than
        # abandoning the whole diagnostic when one of them is unmeasurable.
        #
        # This used to `return` if ANY configured metric was short of samples.
        # Adding unsupported duty to the priority list then switched the entire
        # callback off for exactly the case it is most needed: early stage-1
        # training, where episodes end before settle_steps and every duty is
        # NaN. Measured — a flat, dying run produced one "episode length
        # plateaued" warning under reward_and_length/v1 and ZERO under the
        # stance gate, with eval_gate_met and eval_plateau_active never
        # recorded at all.
        usable = [metric for metric in metrics if metric.value is not None and metric.sample_count >= min_eval_episodes]
        complete = len(usable) == len(metrics)
        self.logger.record("diagnostics/eval_gate_data_complete", 1.0 if complete else 0.0)
        if not usable:
            return

        for metric in usable:
            assert metric.value is not None
            self._histories[metric.key].append(metric.value)

        failing = [metric for metric in usable if metric.is_failing()]
        if not failing:
            # "Met" only when every configured criterion was checkable. A
            # partial panel that happens to clear the criteria it could
            # measure has not met the gate, and publishing 1.0 here would say
            # it had.
            self.logger.record("diagnostics/eval_gate_met", 1.0 if complete else 0.0)
            self.logger.record("diagnostics/eval_plateau_active", 0.0)
            if complete:
                self._clear_plateau("all active stage gates are currently met")
            return

        self.logger.record("diagnostics/eval_gate_met", 0.0)
        blocking = next(metric for key in self._METRIC_PRIORITY for metric in failing if metric.key == key)
        assert blocking.value is not None

        # `blocking` is the most behaviourally specific failing metric AMONG
        # THOSE MEASURABLE. When the panel is partial, a more specific
        # criterion may be the real blocker and simply have no data, so the
        # warning has to say which ones went unmeasured rather than implying
        # the named metric is the whole story.
        unmeasured = [metric.label for metric in metrics if metric not in usable]
        caveat = (
            ""
            if complete
            else (
                f" NOTE: {', '.join(unmeasured)} could not be measured this evaluation, "
                "so a more specific criterion may be the real blocker."
            )
        )

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
                "across the last %d evaluations; latest %s, required %s, step %d. %s%s "
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
                caveat,
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
        # Taken from the stage's own gate declaration so the duty this
        # callback measures uses the same settling window the gate compares
        # it against. Absent (any non-stance gate) it is 0, which measures the
        # whole episode and costs nothing.
        settle_steps=int(curriculum_kwargs.get("settle_steps", 0)),
        **eval_callback_kwargs,
    )
    plateau_callback = StageGatePlateauCallback(
        eval_callback,
        stage=stage,
        curriculum_kwargs=curriculum_kwargs,
        plateau_window=int(curriculum_kwargs.get("diagnostics_plateau_window", 5)),
        min_relative_variation=float(curriculum_kwargs.get("diagnostics_plateau_min_relative_variation", 0.05)),
        # From [env], not [curriculum]: the horizon defines what "full" means
        # for the full-horizon fraction, and it is an environment property.
        horizon=int(stage_config.get("env_kwargs", {}).get("max_episode_steps", 1000)),
        verbose=diagnostics_verbose,
    )
    return eval_callback, plateau_callback
