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
from pathlib import Path
from typing import Any

import numpy as np

from .curriculum.stance_gate import required_duty_episodes
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
        stage: "int | str",
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
        # Action statistics for the DETERMINISTIC policy, which is what the
        # gate scores. `diagnostics.npz` records actions from training
        # rollouts, so every number in it carries exploration noise; the
        # quantities that matter here -- how far the commanded target sits from
        # the home control (action = 0), and how much the policy shakes about it -- are
        # properties of the mean action and are unrecoverable from a noisy
        # sample. Issue #489 had to invert them out of per-episode reward
        # totals under a narrowband assumption. These measure them.
        #
        # Per actuator, not per group: grouping is an analysis decision, and
        # resolving joint names through the VecEnv wrappers is exactly the kind
        # of best-effort lookup that silently returns nothing. Recording the
        # full vector keeps the grouping offline and impossible to get wrong
        # here.
        self.evaluations_action_dc: list[list[float]] = []
        self.evaluations_action_ac_rms: list[list[float]] = []
        self.evaluations_action_delta: list[float] = []
        self.evaluations_action_jerk: list[float] = []
        # Per-episode reward decomposition. Only `mean_reward` was kept, so
        # comparing two runs meant comparing their final checkpoints and
        # nothing in between -- there was no way to ask WHEN a policy adopted
        # the pose it ended up with.
        self.evaluations_reward_terms: list[dict[str, float]] = []
        self._episode_forward_sums: dict[int, float] = {}
        self._episode_forward_counts: dict[int, int] = {}
        self._current_eval_forward_velocities: list[float] = []
        self._episode_steps: dict[int, int] = {}
        self._episode_unsupported: dict[int, int] = {}
        self._episode_bilateral: dict[int, int] = {}
        self._episode_measured: dict[int, int] = {}
        self._current_eval_unsupported_duties: list[float] = []
        self._current_eval_bilateral_duties: list[float] = []
        # Action accumulators, over the post-settle window so the reset
        # transient does not inflate the AC term -- the same window the duty
        # uses, and for the same reason.
        self._action_sum: Any = None
        self._action_sq_sum: Any = None
        self._action_delta_sum = 0.0
        self._action_jerk_sum = 0.0
        self._action_count = 0
        # Counted separately from the samples: a difference needs two and a
        # second difference three, and episode boundaries reset the history.
        # Dividing all three by the sample count biases their ratio, and that
        # ratio is what becomes a frequency.
        self._action_delta_count = 0
        self._action_jerk_count = 0
        self._prev_action: dict[int, Any] = {}
        self._prev_prev_action: dict[int, Any] = {}
        # Reward terms accumulate over the WHOLE episode: they are episode
        # returns, and truncating them to the post-settle window would make
        # them incomparable with `stance_gate_report.py`'s breakdown.
        self._reward_term_sums: dict[str, float] = {}
        self._reward_term_episodes = 0

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
        self._action_sum = None
        self._action_sq_sum = None
        self._action_delta_sum = 0.0
        self._action_jerk_sum = 0.0
        self._action_count = 0
        self._action_delta_count = 0
        self._action_jerk_count = 0
        self._prev_action.clear()
        self._prev_prev_action.clear()
        self._reward_term_sums.clear()
        self._reward_term_episodes = 0

    def _capture_action(self, locals_: dict[str, Any], env_index: int, after_settle: bool) -> None:
        """Accumulate deterministic-policy action statistics for one step.

        Wrapped defensively because ``actions`` is a local of SB3's
        ``evaluate_policy``, not part of any documented callback contract: a
        version that renames it must cost this diagnostic, not the run.
        """
        try:
            actions = locals_.get("actions")
            if actions is None:
                return
            action = np.asarray(actions)
            action = action[env_index] if action.ndim > 1 else action
            action = np.asarray(action, dtype=np.float64).ravel()
            if not action.size:
                return
            prev = self._prev_action.get(env_index)
            prev_prev = self._prev_prev_action.get(env_index)
            if after_settle:
                if self._action_sum is None or self._action_sum.shape != action.shape:
                    self._action_sum = np.zeros_like(action)
                    self._action_sq_sum = np.zeros_like(action)
                self._action_sum += action
                self._action_sq_sum += action * action
                self._action_count += 1
                # Both differences are summed over actuators, matching the
                # `Sum` in `reward_action_smoothness` / `reward_action_jerk`
                # so the numbers invert straight back through those formulas.
                if prev is not None and prev.shape == action.shape:
                    self._action_delta_sum += float(np.sum((action - prev) ** 2))
                    self._action_delta_count += 1
                    if prev_prev is not None and prev_prev.shape == action.shape:
                        self._action_jerk_sum += float(np.sum((action - 2.0 * prev + prev_prev) ** 2))
                        self._action_jerk_count += 1
            self._prev_prev_action[env_index] = prev
            self._prev_action[env_index] = action
        except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
            return

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """Capture evaluation velocity and duty, conditionally delegate success."""

        info = locals_["info"]
        env_index = int(locals_.get("i", 0))
        for key, value in info.items():
            if not key.startswith("reward_"):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                self._reward_term_sums[key] = self._reward_term_sums.get(key, 0.0) + number
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
        self._capture_action(locals_, env_index, after_settle=step_index >= self.settle_steps)
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
            self._reward_term_episodes += 1
            self._prev_action.pop(env_index, None)
            self._prev_prev_action.pop(env_index, None)

        if self.success_applicable:
            super()._log_success_callback(locals_, globals_)

    def seed_prior_history(self, n_evals: int) -> None:
        """Pad the parallel per-eval capture series for *n_evals* seeded evals.

        On a same-stage resume, ``curriculum.seed_resume_eval_state`` restores
        SB3's ``evaluations_*`` lists from the stage's published
        ``evaluations.npz`` — but the prior session's per-eval captures
        (duties, velocities, action statistics, reward terms) are not in that
        file.  ``StageGatePlateauCallback`` resolves these series by the
        ABSOLUTE index into ``evaluations_results``; without padding, every
        post-resume evaluation's index points past the end of each series,
        the stance panel resolves to ``None``, and the gate telemetry
        (``gate_progress.npz`` appends, ``diagnostics/eval_*`` scalars)
        silently stops for the rest of the session.  The placeholders are the
        exact values ``_on_step``/``_publish_action_statistics`` append for an
        unmeasured evaluation, which every consumer already treats as
        "no data for this eval".
        """
        for _ in range(n_evals):
            self.evaluations_forward_velocities.append([])
            self.evaluations_unsupported_duties.append([])
            self.evaluations_bilateral_duties.append([])
            self.evaluations_action_dc.append([])
            self.evaluations_action_ac_rms.append([])
            self.evaluations_action_delta.append(float("nan"))
            self.evaluations_action_jerk.append(float("nan"))
            self.evaluations_reward_terms.append({})

    def _publish_action_statistics(self) -> None:
        """Reduce one evaluation's accumulators into the published series.

        The DC/AC split is the whole point. ``mean(a)`` per actuator is the
        static target the policy commands -- the distance from the home
        control, which under ``home-keyframe-residual/v1`` is exactly
        ``action = 0`` (the home keyframe's control targets, not necessarily
        its joint pose: the trex ankles carry an authored preload).
        ``sqrt(mean(a^2) - mean(a)^2)`` is what it does *around* that pose. The
        two answer different questions and a pooled standard deviation over
        actuators and time (which is what ``diagnostics.action_std`` computes)
        cannot separate them.
        """
        count = self._action_count
        if count and self._action_sum is not None:
            mean = self._action_sum / count
            var = np.maximum(self._action_sq_sum / count - mean * mean, 0.0)
            self.evaluations_action_dc.append([float(v) for v in mean])
            self.evaluations_action_ac_rms.append([float(v) for v in np.sqrt(var)])
            # Per differencing opportunity, so they invert directly through
            # the reward formulas without an edge-effect bias.
            self.evaluations_action_delta.append(
                self._action_delta_sum / self._action_delta_count if self._action_delta_count else float("nan")
            )
            self.evaluations_action_jerk.append(
                self._action_jerk_sum / self._action_jerk_count if self._action_jerk_count else float("nan")
            )
        else:
            self.evaluations_action_dc.append([])
            self.evaluations_action_ac_rms.append([])
            self.evaluations_action_delta.append(float("nan"))
            self.evaluations_action_jerk.append(float("nan"))
        episodes = self._reward_term_episodes
        self.evaluations_reward_terms.append(
            {key: value / episodes for key, value in self._reward_term_sums.items()} if episodes else {}
        )

    def _on_step(self) -> bool:
        evaluation_due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluation_due:
            self._reset_forward_velocity_capture()

        continue_training = super()._on_step()

        if evaluation_due:
            self.evaluations_forward_velocities.append(list(self._current_eval_forward_velocities))
            self.evaluations_unsupported_duties.append(list(self._current_eval_unsupported_duties))
            self.evaluations_bilateral_duties.append(list(self._current_eval_bilateral_duties))
            self._publish_action_statistics()
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
    #: Samples this metric needs before it can be followed.  ``None`` means
    #: the stage's ``min_eval_episodes``.  The stance duty overrides it: the
    #: gate certifies on ``ceil(min_eval_episodes x min_full_horizon_fraction)``
    #: duty episodes, so requiring the full panel size here made a 39-of-40
    #: panel report ``eval_gate_met = 0`` while the curriculum advanced on it.
    min_samples: int | None = None
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

    #: Declared at class scope so an instance built without ``__init__``
    #: still has them. The suite constructs this callback with
    #: ``object.__new__`` to exercise the plateau logic without
    #: stable-baselines3, and a new instance attribute only ``__init__`` set
    #: would turn every one of those tests into an AttributeError.
    _gate_progress_dir: "Path | None" = None
    _gate_progress_timesteps: list[int] = []
    _gate_progress: dict[str, list[float]] = {}
    _warned_gate_progress_unwritable = False

    def __init__(
        self,
        eval_callback: StageAwareEvalCallback,
        *,
        stage: "int | str",
        curriculum_kwargs: dict[str, Any],
        plateau_window: int = 5,
        min_relative_variation: float = 0.05,
        horizon: int = 1000,
        gate_progress_dir: "str | Path | None" = None,
        control_dt: float = 0.01,
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
        # Seconds per control step: `timestep * frame_skip`, needed to turn the
        # jerk/delta ratio into a frequency rather than a bare ratio.
        self.control_dt = float(control_dt) if control_dt and control_dt > 0 else 0.01
        # Per-evaluation gate criteria, persisted to `gate_progress.npz`.
        self._gate_progress_dir = None if gate_progress_dir is None else Path(gate_progress_dir)
        self._gate_progress_timesteps: list[int] = []
        self._gate_progress: dict[str, list[float]] = {}
        self._warned_gate_progress_unwritable = False

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
                # The gate's own rule, not the panel size -- see _GateMetric.
                min_samples=required_duty_episodes(
                    int(self.curriculum_kwargs.get("min_eval_episodes", 10)),
                    float(self.curriculum_kwargs.get("min_full_horizon_fraction", 0.0)),
                ),
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
        bilateral = self._full_horizon_mean("evaluations_bilateral_duties", index)
        # Recorded to TensorBoard AND accumulated for `gate_progress.npz`.
        # These are the deterministic, panel-estimator gate criteria -- the
        # same numbers the stance gate tests -- and until now they existed
        # only in TensorBoard. Post-hoc analysis of run 20260804_143747 had to
        # substitute the TRAINING-rollout duty from `diagnostics.npz`, which is
        # contaminated by exploration noise; it happened to agree to three
        # decimals there, but that was luck, not a property worth relying on.
        # Also to the SB3 logger, so TensorBoard and W&B carry them live
        # rather than only the npz. A diagnostic that exists solely in a file
        # nobody opens mid-run is how the stance gate criteria stayed invisible
        # for the whole of run 20260804_143747.
        action_stats = self._action_statistics(index)
        for key, value in action_stats.items():
            if math.isfinite(value):
                self.logger.record(f"diagnostics/eval_{key}", value)
        for key, value in self._reward_term_scalars(index).items():
            self.logger.record(f"reward_terms/eval_{key[len('term_') :]}", value)
        self._record_gate_progress(
            full_horizon_fraction=panel.full_horizon_fraction,
            duty_episodes=panel.n_duty_episodes,
            unsupported_duty=panel.mean_unsupported_duty if panel.n_duty_episodes else float("nan"),
            unsupported_duty_ucb=(
                panel.unsupported_duty_ucb
                if panel.n_duty_episodes and math.isfinite(panel.unsupported_duty_ucb)
                else float("nan")
            ),
            bilateral_support_duty=float("nan") if bilateral is None else bilateral,
            mean_reward=panel.mean_reward,
            **action_stats,
            **self._reward_term_scalars(index),
        )
        if not panel.n_duty_episodes:
            return
        self.logger.record("diagnostics/eval_unsupported_duty", panel.mean_unsupported_duty)
        if math.isfinite(panel.unsupported_duty_ucb):
            self.logger.record("diagnostics/eval_unsupported_duty_ucb", panel.unsupported_duty_ucb)
        # Not gated. Measured over the SAME full-horizon episodes as the duty
        # above so the two are comparable; averaging it over every episode is
        # what made the three stance shares stop summing to 1.
        if bilateral is not None:
            self.logger.record("diagnostics/eval_bilateral_support_duty", bilateral)

    def _eval_series(self, name: str, index: int) -> Any:
        series = getattr(self.eval_callback, name, None)
        if not series or index >= len(series):
            return None
        return series[index]

    def _action_statistics(self, index: int) -> dict[str, float]:
        """Scalar summaries of the deterministic policy's action.

        ``action_dc_rms`` is the distance from the home control (``action = 0``) and
        ``action_ac_rms`` is the tremor about it; keeping them apart is the
        point, because they have different causes and different fixes
        (issue #489). The per-actuator vectors go to `gate_progress.npz`
        separately -- these scalars exist so the series is readable without
        reducing a matrix.
        """
        out: dict[str, float] = {}
        dc = self._eval_series("evaluations_action_dc", index)
        ac = self._eval_series("evaluations_action_ac_rms", index)
        out["action_dc_rms"] = float(np.sqrt(np.mean(np.square(dc)))) if dc else float("nan")
        out["action_ac_rms"] = float(np.sqrt(np.mean(np.square(ac)))) if ac else float("nan")
        for key, name in (("action_delta", "evaluations_action_delta"), ("action_jerk", "evaluations_action_jerk")):
            series = getattr(self.eval_callback, name, None)
            value = series[index] if series and index < len(series) else float("nan")
            out[key] = float(value)
        # The frequency the two differences imply, for a narrowband action
        # signal: jerk/delta = (2 sin(pi f dt))^2, blind to any DC offset. A
        # value at the white-noise limit means the ratio is saturated and the
        # signal is broadband, not that it sits at that frequency.
        delta, jerk = out["action_delta"], out["action_jerk"]
        if math.isfinite(delta) and math.isfinite(jerk) and delta > 0.0:
            gain = math.sqrt(jerk / delta)
            out["action_freq_hz"] = math.asin(min(1.0, gain / 2.0)) / (math.pi * self.control_dt)
        else:
            out["action_freq_hz"] = float("nan")
        return out

    def _reward_term_scalars(self, index: int) -> dict[str, float]:
        """Per-episode mean of each reward term, flattened into the npz."""
        terms = self._eval_series("evaluations_reward_terms", index)
        if not terms:
            return {}
        return {f"term_{key[len('reward_') :]}": float(value) for key, value in terms.items()}

    def _record_gate_progress(self, **values: float) -> None:
        """Append one evaluation's gate criteria and rewrite ``gate_progress.npz``.

        A separate file rather than columns in ``diagnostics.npz`` because the
        two are on different clocks: diagnostics is per training rollout, this
        is per evaluation. Merging them would force one series to be padded
        with NaN against the other's timeline, which is exactly the alignment
        trap ``_history_algo`` already has to work around.
        """
        # Bind instance-local containers before appending: the class-scope
        # defaults above exist only so `object.__new__` instances have the
        # attribute, and appending to them would share state across every
        # instance in the process.
        if "_gate_progress_timesteps" not in self.__dict__:
            self._gate_progress_timesteps = []
            self._gate_progress = {}
        self._gate_progress_timesteps.append(int(self.num_timesteps))
        for key, value in values.items():
            self._gate_progress.setdefault(key, []).append(float(value))
        if self._gate_progress_dir is None:
            return
        try:
            from .file_io import atomic_savez

            n = len(self._gate_progress_timesteps)
            payload = {"timesteps": np.array(self._gate_progress_timesteps)}
            # Per-actuator DC and AC as (n_evals, n_actuators) matrices. Which
            # actuators carry the offset decides whether a leg-pose weight can
            # reach it at all -- `leg_home_pose` covers only the leg joints, so
            # a displacement living in the tail or neck is invisible to it and
            # unfixable by it (issue #489). Scalar summaries alone cannot answer
            # that, and the grouping is left to analysis rather than guessed at
            # here.
            # `getattr` on self, not `self.eval_callback` directly: these
            # matrices are an enhancement, and an AttributeError here would be
            # caught by the outer handler and take the ENTIRE file down with
            # it -- losing the gate criteria to a failure in a side channel.
            source = getattr(self, "eval_callback", None)
            for key, name in (
                ("action_dc_per_actuator", "evaluations_action_dc"),
                ("action_ac_rms_per_actuator", "evaluations_action_ac_rms"),
            ):
                series = getattr(source, name, None) or []
                rows = [row for row in series[:n] if row]
                # Ragged rows would need padding, which invents values. A run
                # whose actuator count changed mid-flight is not a thing that
                # happens; a run with some empty evaluations is.
                if len(rows) == n and len({len(row) for row in rows}) == 1:
                    payload[key] = np.array(rows, dtype=float)
            # Length-guarded against `timesteps`, like every optional series in
            # `diagnostics.npz`. Today one call site supplies every key on every
            # evaluation, so nothing can diverge -- but a future caller that
            # supplies a key conditionally would otherwise write a short array
            # that silently reads as the FIRST n evaluations rather than the
            # ones it came from. Dropping it is recoverable; misaligning it is
            # the kind of error this file exists to stop.
            for key, series in self._gate_progress.items():
                if len(series) == n:
                    payload[key] = np.array(series)
            atomic_savez(Path(self._gate_progress_dir) / "gate_progress.npz", **payload)
        except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
            # Once. An unwritable directory fails identically on all ~200
            # evaluations of a 10M run, and 200 stack traces would bury the
            # warnings this run is actually meant to surface.
            if not self._warned_gate_progress_unwritable:
                self._warned_gate_progress_unwritable = True
                logger.warning("Could not write gate_progress.npz", exc_info=True)

    def _process_evaluation(self, index: int) -> None:
        panel = self._stance_panel(index)
        # Record BEFORE the no-metrics early-out below: a stage whose gate
        # declares none of the legacy scalar thresholds (the recovery stage's
        # recovery_quality/v1 — judged post-stage from its frozen resolution)
        # has no plateau metrics to follow, but its gate_progress.npz appends
        # and diagnostics/eval_* scalars are exactly the mid-run visibility
        # the stage needs.  Gating the recording on the plateau metrics made
        # every recovery run train dark (gap review EE1).
        self._record_stance_scalars(index, panel)

        metrics = self._metrics_for_evaluation(index, panel)
        if not metrics:
            # Nothing to follow for plateau/gate-met tracking; the recording
            # above already happened.
            return

        min_eval_episodes = int(self.curriculum_kwargs.get("min_eval_episodes", 10))
        reward_metric = next((metric for metric in metrics if metric.key == "mean_reward"), None)
        if reward_metric is not None:
            self.logger.record("diagnostics/eval_episode_count", reward_metric.sample_count)

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
        usable = [
            metric
            for metric in metrics
            if metric.value is not None
            and metric.sample_count >= (min_eval_episodes if metric.min_samples is None else metric.min_samples)
        ]
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
    stage: "int | str",
    stage_config: dict[str, Any],
    diagnostics_verbose: int = 0,
    gate_progress_dir: "str | Path | None" = None,
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
        gate_progress_dir=gate_progress_dir,
        # Also from [env]: one control step is `timestep * frame_skip`, and
        # without it the jerk/delta ratio is a bare number rather than a
        # frequency. Defaults match the MuJoCo/repo defaults so a config that
        # declares neither still reports a sane 100 Hz rather than nothing.
        control_dt=(
            float(stage_config.get("env_kwargs", {}).get("timestep", 0.002))
            * int(stage_config.get("env_kwargs", {}).get("frame_skip", 5))
        ),
        verbose=diagnostics_verbose,
    )
    return eval_callback, plateau_callback
