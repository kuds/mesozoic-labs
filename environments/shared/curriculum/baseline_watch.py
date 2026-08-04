"""Report training progress against the do-nothing policy.

Run ``20260804_143747`` trained T-Rex stage 1 for its full 10,002,432 steps,
8h 13m, and finished **below the zero-action statue on every major reward
term** -- 2686.9 against the statue's 3271.8, a 17.9% shortfall (issue #486).
Nothing in the run said so.  Every signal that was logged looked healthy:
reward climbing, 100% full-horizon, evaluation standard deviation down to 5,
the collapse backstop correctly not firing.  The run only turned out to be in
a worse-than-trivial local optimum once someone scored ``action = 0`` by hand
afterwards.

That comparison is nearly free.  ``zero_action_baseline.json`` is already
written into the run directory before training starts, and the number needed
is one float in it.  This module reads that float and reports it against each
evaluation, so "the policy has not yet beaten doing nothing" is visible at
100k steps instead of after the budget is spent.

Deliberately **advisory only**.  It logs and warns; it never stops a run.
Being under the baseline is normal for most of a stage -- the statue is the
stage-1 reward optimum by construction (STAGE1_SPLIT_PLAN §2.3.1), so a
learning policy climbs toward it from below and may legitimately sit under it
for millions of steps.  Turning that into an abort would kill healthy runs for
the same reason tightening the collapse detector does
(:func:`~environments.shared.curriculum.early_stopping.collapse_settings_from_config`).
The warning is about *elapsed budget with no crossing*, not about the gap
itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)

#: Fraction of the stage budget after which still being below the baseline is
#: worth a warning.  0.35 is chosen so the warning lands with most of the
#: budget still unspent: on the 10M stage that is 3.5M steps, ~2.9 hours in on
#: run ``20260804_143747``'s hardware, with ~5.5 hours still to save.  It is
#: also past the point where that run's duty had already fallen below 0.43 and
#: its reward trend had flattened, so a policy still short of the statue there
#: is not merely early.
DEFAULT_WARN_AFTER_BUDGET_FRACTION = 0.35


def read_zero_action_baseline(run_dir: "str | Path", species: str) -> float | None:
    """The statue's mean reward for *species* from a run's captured baseline.

    ``None`` when the file is absent, unreadable, or does not cover this
    species -- the caller then skips the comparison rather than inventing a
    reference.  A wrong baseline is worse than none: it would either mute a
    real warning or cry wolf for a whole run.
    """
    path = Path(run_dir) / "zero_action_baseline.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Zero-action baseline at %s could not be read; skipping the comparison.", path)
        return None
    entry = (payload.get("results") or {}).get(species)
    if not isinstance(entry, dict):
        return None
    value: Any = entry.get("reward_mean")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


class BaselineProgressCallback(BaseCallback):  # type: ignore[misc]
    """Log each evaluation against the zero-action baseline.

    Args:
        eval_callback: The ``EvalCallback`` whose ``evaluations_results`` to
            read.  Same source as the collapse backstop, so no extra rollouts.
        baseline_reward: The statue's mean reward for this species and stage.
        total_timesteps: The stage's configured budget, used to decide when a
            still-unbeaten baseline is worth warning about.
        warn_after_budget_fraction: Fraction of the budget after which the
            warning fires.
        duty_ceiling: The stage's ``max_unsupported_duty`` when it declares
            one, reported alongside so the gate's binding criterion is visible
            in the same line.  ``None`` for stages with no stance gate.
    """

    #: Log-once latch, at class scope so an instance built without ``__init__``
    #: (as the tests do, to exercise ``_on_step`` without stable-baselines3)
    #: still has it.
    _warned_below_baseline = False

    def __init__(
        self,
        eval_callback: Any,
        baseline_reward: float,
        *,
        total_timesteps: int,
        warn_after_budget_fraction: float = DEFAULT_WARN_AFTER_BUDGET_FRACTION,
        duty_ceiling: float | None = None,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for BaselineProgressCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.baseline_reward = float(baseline_reward)
        self.total_timesteps = int(total_timesteps)
        self.warn_after_budget_fraction = float(warn_after_budget_fraction)
        self.duty_ceiling = None if duty_ceiling is None else float(duty_ceiling)
        self._last_seen_n_evals = 0
        self._warned_below_baseline = False
        self._ever_beat_baseline = False

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None)
        if not results:
            return True
        n_evals = len(results)
        if n_evals <= self._last_seen_n_evals:
            return True
        self._last_seen_n_evals = n_evals

        latest = float(np.mean(results[-1]))
        share = latest / self.baseline_reward if self.baseline_reward else float("nan")
        if latest >= self.baseline_reward:
            self._ever_beat_baseline = True

        ceiling = "" if self.duty_ceiling is None else f", duty ceiling {self.duty_ceiling:.4f}"
        logger.info(
            "BaselineProgress: eval mean %.1f vs zero-action baseline %.1f (%.1f%% of baseline)%s",
            latest,
            self.baseline_reward,
            100.0 * share,
            ceiling,
        )

        # Once, and only once the budget spent makes "still climbing" an
        # unconvincing explanation. Suppressed entirely for a run that has
        # beaten the baseline at any point, because falling back under it
        # afterwards is what the collapse backstop is for.
        if (
            not self._warned_below_baseline
            and not self._ever_beat_baseline
            and self.total_timesteps > 0
            and self.num_timesteps >= self.warn_after_budget_fraction * self.total_timesteps
        ):
            self._warned_below_baseline = True
            logger.warning(
                "BaselineProgress: %d of %d steps (%.0f%% of budget) and the policy has never "
                "beaten the zero-action baseline — latest eval %.1f vs %.1f (%.1f%%). On a stage "
                "whose reward optimum IS the do-nothing policy this means the run is in a "
                "worse-than-trivial local optimum, not merely early. Run 20260804_143747 spent "
                "its whole 10M budget in exactly this state and failed its gate (issue #486).",
                self.num_timesteps,
                self.total_timesteps,
                100.0 * self.num_timesteps / self.total_timesteps,
                latest,
                self.baseline_reward,
                100.0 * share,
            )
        return True


def build_baseline_progress_callback(
    eval_callback: Any,
    *,
    run_dir: "str | Path | None",
    species: str,
    stage_config: dict[str, Any],
    verbose: int = 0,
) -> BaselineProgressCallback | None:
    """Build the watcher, or ``None`` when there is no baseline to compare to.

    Returning ``None`` rather than a no-op callback keeps the absence visible
    in the callback list, and means a run without a captured baseline pays
    nothing for a comparison it cannot make.
    """
    if run_dir is None:
        return None
    baseline = read_zero_action_baseline(run_dir, species)
    if baseline is None:
        logger.info(
            "No zero-action baseline for %s in %s; skipping the below-baseline watch.",
            species,
            run_dir,
        )
        return None
    curriculum = stage_config.get("curriculum_kwargs", {})
    duty_ceiling = curriculum.get("max_unsupported_duty")
    return BaselineProgressCallback(
        eval_callback,
        baseline,
        total_timesteps=int(curriculum.get("timesteps", 0)),
        warn_after_budget_fraction=float(
            curriculum.get("baseline_warn_after_budget_fraction", DEFAULT_WARN_AFTER_BUDGET_FRACTION)
        ),
        duty_ceiling=None if duty_ceiling is None else float(duty_ceiling),
        verbose=verbose,
    )
