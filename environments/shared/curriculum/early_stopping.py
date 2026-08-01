"""Early stop on evaluation collapse.

Aborts a stage whose evaluation return has fallen far enough below its own
best that the remaining budget is not worth spending."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)


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
    ``peak_floor``: a run can only "collapse" from a level that was actually
    good, so the pre-convergence grind cannot trip the backstop. Each stage
    config sets its own ``collapse_peak_floor``; the value is deliberately not
    inherited from ``min_avg_reward``, which would couple this backstop to an
    unrelated advancement threshold.

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
        if not sb3_compat._SB3_AVAILABLE:
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
                    "EvalCollapseEarlyStop: stopping training at step %d — eval mean collapsed from peak %.1f to %.1f",
                    self.num_timesteps,
                    self._peak_score,
                    latest_mean,
                )
                return False
        else:
            self._consecutive_drops = 0

        return True


def collapse_settings_from_config(curriculum_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve the eval-collapse backstop's settings from a stage config.

    Kept separate from the callback so the resolution can be tested without
    stable-baselines3 installed — it is plain arithmetic on a config dict, and
    the one rule worth pinning does not need SB3 to state.

    That rule: ``collapse_peak_floor`` is deliberately **not** inherited from
    ``min_avg_reward``.  It used to chain ``collapse_peak_floor`` ->
    ``min_avg_reward`` -> ``0.0``, which coupled an early-stop backstop to an
    unrelated advancement threshold: removing the reward gate from a stage --
    as a state-capability gate would -- silently dropped the floor to ``0.0``,
    arming collapse detection after *any* positive robust peak.  A missing
    floor now means "never arm" instead, because a backstop that is not
    configured should not abort a run; every stage config sets the value it
    actually wants.  See docs/STAGE1_SPLIT_PLAN.md section 7.4.

    **The floor should be expressed RELATIVELY.** An absolute number cannot
    survive a reward-function edit, and has now failed to arm twice for the
    same reason.  Run ``20260731_132102``: floor 2200 calibrated against a run
    whose peak was 2496, reward scale shifted, next run peaked at 1934.1, the
    detector never armed and watched a -59% collapse.  Run ``20260801_021545``:
    floor 2450 re-derived as 0.75x the zero-action statue (3271.8), the run's
    best evaluation was 2347.67 -- still below it -- so the detector never
    armed again and watched eval degrade 2347.67 -> 1666.33.  Deriving the
    floor from the statue is only half a fix: the statue bounds what is
    *achievable*, not what a *learning* policy passes through.

    ``collapse_peak_floor_fraction`` therefore sets the floor as a fraction of
    ``collapse_peak_floor_reference`` -- the zero-action standing baseline for
    the species and stage, which is measurable before training and re-anchors
    automatically whenever the reward changes.  An explicit
    ``collapse_peak_floor`` still wins when present, so existing configs are
    untouched.
    """
    peak_floor = curriculum_kwargs.get("collapse_peak_floor")
    if peak_floor is None:
        fraction = curriculum_kwargs.get("collapse_peak_floor_fraction")
        reference = curriculum_kwargs.get("collapse_peak_floor_reference")
        if fraction is not None and reference is not None:
            peak_floor = float(fraction) * float(reference)
        else:
            peak_floor = float("inf")
    return {
        "min_evals": int(curriculum_kwargs.get("collapse_min_evals", 12)),
        "patience": int(curriculum_kwargs.get("collapse_patience", 8)),
        "drop_fraction": float(curriculum_kwargs.get("collapse_drop_fraction", 0.4)),
        "peak_floor": float(peak_floor),
        "smoothing_window": int(curriculum_kwargs.get("collapse_smoothing_window", 5)),
    }


def build_eval_collapse_early_stop_callback(
    eval_callback: Any,
    curriculum_kwargs: dict[str, Any],
    *,
    verbose: int = 0,
) -> EvalCollapseEarlyStopCallback:
    """Build the shared eval-collapse backstop from curriculum settings.

    See :func:`collapse_settings_from_config` for how the settings resolve, in
    particular why ``collapse_peak_floor`` does not inherit ``min_avg_reward``.
    """
    settings = collapse_settings_from_config(curriculum_kwargs)
    return EvalCollapseEarlyStopCallback(
        eval_callback=eval_callback,
        min_evals=settings["min_evals"],
        patience=settings["patience"],
        drop_fraction=settings["drop_fraction"],
        peak_floor=settings["peak_floor"],
        verbose=verbose,
        smoothing_window=settings["smoothing_window"],
    )
