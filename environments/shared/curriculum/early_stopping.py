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

    ``peak_warmup_timesteps`` excludes early evaluations from *peak candidacy*,
    which is a different axis from the floor and exists because on stage 1 no
    floor value can work. With ``home-keyframe-residual/v1`` and
    ``log_std_init = 0``, action = 0 commands the nominal stance, so the
    UNTRAINED policy already scores near the reward optimum: run
    ``20260803_012355`` peaked at 2469.4 on its second evaluation (100k steps),
    75% of the 3271.8 zero-action baseline. Its floor of 0.45 x 3271.8 = 1472.3
    therefore armed on initialisation, and the ordinary exploration dip that
    follows -- the documented "dips before it gets better" pattern, which the
    6M run at the same seed passed through on its way to new bests at 2.8M,
    3.3M and 4.55M -- read as a collapse. It stopped at 1.45M of a 10M budget,
    14.5%, and stages 2 and 3 then trained 9 1/4 hours on the near-statue
    checkpoint it left.

    No floor fixes that. Set it above initialisation (> 0.75x baseline) and it
    lands above what a learning policy passes through, so it never arms -- the
    two historical failures :func:`collapse_settings_from_config` documents.
    Set it lower and it arms on initialisation. The two are the same number on
    this stage. A warm-up separates them, and it is expressed in TIMESTEPS
    rather than reward, so unlike an absolute floor it survives a
    reward-function edit.

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
        peak_warmup_timesteps: Windows containing any evaluation before this
            step do not set the peak (default 0.0 — every window counts, the
            historical behaviour). While no window qualifies the backstop stays
            disarmed, which is the fail-safe direction for a backstop.
    """

    #: Log-once latches, declared at class scope so an instance built without
    #: ``__init__`` still has them. The test suite constructs this callback
    #: with ``object.__new__`` to exercise ``_on_step`` without SB3 installed,
    #: and a new instance attribute that only ``__init__`` sets would turn
    #: every one of those tests into an AttributeError.
    _warned_unreadable_timesteps = False
    _announced_armed = False

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
        peak_warmup_timesteps: float = 0.0,
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
        self.peak_warmup_timesteps = float(peak_warmup_timesteps)
        self._peak_score = -np.inf
        self._consecutive_drops = 0
        self._last_seen_n_evals = 0
        # Both are "say it once, not once per evaluation": a backstop that
        # repeats the same line for the rest of a 10M-step run buries whatever
        # else the log had to say.
        self._warned_unreadable_timesteps = False
        self._announced_armed = False

    def _windows_after_warmup(self, n_windows: int, window: int) -> np.ndarray | None:
        """Mask of rolling-median windows that start at or after the warm-up.

        ``None`` when the evaluation timesteps are unavailable or do not line
        up with the results — the caller then stays disarmed, because a
        backstop that cannot tell early evaluations from late ones must not be
        the thing that aborts a multi-hour run.
        """
        timesteps = getattr(self.eval_callback, "evaluations_timesteps", None)
        # The two causes need different fixes, so they get different messages:
        # "absent" means the EvalCallback was built without a log_path,
        # "misaligned" means its two arrays disagree, which is a bug rather
        # than a configuration mistake. Written as two early returns rather
        # than a shared `reason` branch so the narrowing survives to the
        # indexing below.
        if timesteps is None:
            self._warn_disarmed("the EvalCallback records no evaluation timesteps (is its log_path set?)")
            return None
        expected = n_windows + window - 1
        if len(timesteps) < expected:
            self._warn_disarmed(
                f"only {len(timesteps)} evaluation timesteps for {expected} recorded "
                "evaluations, so windows cannot be dated"
            )
            return None
        starts: np.ndarray = np.asarray(timesteps[:n_windows], dtype=np.float64)
        return starts >= self.peak_warmup_timesteps

    def _warn_disarmed(self, reason: str) -> None:
        """Explain the disarm once, then stay quiet for the rest of the stage."""
        if not self._warned_unreadable_timesteps:
            self._warned_unreadable_timesteps = True
            logger.warning(
                "EvalCollapseEarlyStop: peak_warmup_timesteps=%.0f is set but %s. "
                "Staying disarmed for the rest of this stage — a backstop that cannot "
                "tell early evaluations from late ones must not abort the run.",
                self.peak_warmup_timesteps,
                reason,
            )

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

        # Window w covers eval_means[w : w + window], so requiring
        # timesteps[w] >= warmup drops every window that CONTAINS a
        # pre-warm-up evaluation, not merely those ending in one. A median
        # is robust to one contaminating sample out of five, not to three.
        if self.peak_warmup_timesteps > 0.0:
            eligible = self._windows_after_warmup(rolling_medians.size, window)
            if eligible is None or not eligible.any():
                # Nothing has yet been learned that could be destroyed, so
                # there is no peak to collapse from. Stay disarmed.
                self._consecutive_drops = 0
                return True
            rolling_medians = rolling_medians[eligible]

        self._peak_score = float(rolling_medians.max())

        latest_mean = float(eval_means[-1])
        threshold = (1.0 - self.drop_fraction) * self._peak_score
        armed = self._peak_score > 0 and self._peak_score >= self.peak_floor

        # The single line that makes an early stop diagnosable from the log
        # alone. Reconstructing whether run 20260803_012355's backstop armed
        # on its second evaluation took replaying the whole evaluation series
        # against the config, because nothing recorded the transition. Said
        # once, at the step the backstop becomes able to end the run.
        if armed and not self._announced_armed:
            self._announced_armed = True
            logger.info(
                "EvalCollapseEarlyStop: armed at step %d — rolling-median peak %.1f "
                "reached the floor %.1f; training now stops after %d consecutive "
                "evaluations below %.1f (%.0f%% of peak).",
                self.num_timesteps,
                self._peak_score,
                self.peak_floor,
                self.patience,
                threshold,
                100 * (1 - self.drop_fraction),
            )

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

    **Do NOT tighten ``drop_fraction`` / ``patience`` to make this fire more
    often.**  PLANT_VALIDATION section 14 item 3 asked for that, and simulating
    the detector against run ``20260801_021545``'s real 120-evaluation series
    refutes it.  Stage-1 evaluation reward on this task is enormously noisy:
    45 of 94 pre-peak evaluations sit below HALF their running rolling-median
    peak, and 52 below 70%.  The run's endgame dip is milder than 53 separate
    mid-training excursions, so any threshold that catches the endgame fires
    dozens of times earlier -- every (drop_fraction, patience) pair tried,
    from 0.5/10 down to 0.2/3, first stops the run somewhere between
    evaluation 66 and 103, all of them BEFORE the best model at evaluation
    114.  Tightening does not detect collapse sooner; it aborts healthy runs.
    Episode length behaves the same way (48 of 94 below 70%), so switching
    signals does not rescue it either.

    Nor was there anything to catch: the final evaluation of that run (1630.7)
    is HIGHER than 76 of its own 119 preceding evaluations, and its late-run
    spread (618-2348) matches its mid-run spread (625-2312).  The gap between
    the best checkpoint and the last one is ordinary evaluation variance --
    which is what ``robust_best_model`` exists to absorb -- not a collapse.
    The defaults 0.5/10 are the only setting tested that does NOT abort that
    run, and declining to fire on it was correct behaviour.
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
        # Defaults to 0.0 = every window may set the peak, the behaviour every
        # existing config already has. Only a stage that declares the key
        # changes, so this cannot silently disarm a backstop elsewhere.
        "peak_warmup_timesteps": float(curriculum_kwargs.get("collapse_peak_warmup_timesteps", 0.0)),
    }


def build_eval_collapse_early_stop_callback(
    eval_callback: Any,
    curriculum_kwargs: dict[str, Any],
    *,
    verbose: int = 0,
    total_timesteps: int | None = None,
) -> EvalCollapseEarlyStopCallback:
    """Build the shared eval-collapse backstop from curriculum settings.

    See :func:`collapse_settings_from_config` for how the settings resolve, in
    particular why ``collapse_peak_floor`` does not inherit ``min_avg_reward``.

    ``total_timesteps`` is the cumulative step count the session actually
    trains to; the warm-up sanity log below compares against it rather than
    the TOML budget, which a ``--timesteps`` override or a shortened resume
    makes wrong (review TC9).  ``None`` falls back to the TOML value.
    """
    settings = collapse_settings_from_config(curriculum_kwargs)

    # State the resolved settings once, at construction. Whether a run stopped
    # early -- and whether it *could* have -- turns entirely on these five
    # numbers, and none of them appeared anywhere in a run's log: diagnosing
    # runs 20260802_203215 and 20260803_012355 meant reading the TOML at the
    # run's commit against the artifacts, because the log never said which
    # floor was in force.
    warmup = settings["peak_warmup_timesteps"]
    logger.info(
        "EvalCollapseEarlyStop: min_evals=%d patience=%d drop_fraction=%.2f "
        "peak_floor=%.1f smoothing_window=%d peak_warmup_timesteps=%.0f",
        settings["min_evals"],
        settings["patience"],
        settings["drop_fraction"],
        settings["peak_floor"],
        settings["smoothing_window"],
        warmup,
    )

    # A warm-up at or past the stage's own budget disarms the backstop for the
    # entire run, and does it silently -- the run simply never stops early and
    # nothing says why. It is a plausible typo (an extra zero) on a key whose
    # value is a step count in the millions, so say so rather than leave the
    # protection quietly switched off.
    budget = curriculum_kwargs.get("timesteps") if total_timesteps is None else total_timesteps
    if warmup > 0.0 and budget is not None and warmup >= float(budget):
        logger.warning(
            "EvalCollapseEarlyStop: collapse_peak_warmup_timesteps=%.0f is at or beyond this "
            "stage's timesteps=%.0f, so no window can ever set the peak and the collapse "
            "backstop is disabled for the whole run.",
            warmup,
            float(budget),
        )

    return EvalCollapseEarlyStopCallback(
        eval_callback=eval_callback,
        min_evals=settings["min_evals"],
        patience=settings["patience"],
        drop_fraction=settings["drop_fraction"],
        peak_floor=settings["peak_floor"],
        verbose=verbose,
        smoothing_window=settings["smoothing_window"],
        peak_warmup_timesteps=settings["peak_warmup_timesteps"],
    )
