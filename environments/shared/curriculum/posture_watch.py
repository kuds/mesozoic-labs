"""Warn when the evaluation panel keeps standing over the stage's fore-aft centre-of-pressure bar.

The T. rex physics-r10 screen of 2026-10-16 trained three seeds of one
config.  Its mid-run telemetry at 2.3 to 2.5M steps showed seed 42 standing
clean while seeds 52 and 60 drifted onto their toes (whole-foot centre of
pressure 1.1 to 1.25 span half-lengths, digit share 0.7 to 0.9, negative toe
margins, seed 60 nosediving), much further than the physics-r9 runs had
drifted at the same step.  ``gate_progress.npz`` held the numbers
(``StageGatePlateauCallback``'s over-bar fraction of each declared rail,
#620 and #622), but nothing in the training output said so: the drift was
found by reading the file by hand.

This callback reads that over-bar fraction after each evaluation and, when
it is at least ``posture_drift_warn_over_bar_fraction`` (0.5) on
``posture_drift_warn_consecutive_evals`` (2) evaluations in a row, prints
one warning line in the training output, which the notebook shows, and
records it for ``progress.json`` (:mod:`.progress_report`).  It repeats at
most once per ``posture_drift_warn_repeat_evals`` (10) evaluations while the
drift lasts.  Two in a row, not three, to warn earlier: replayed on the r10
screen's own whole-foot series (to 6.16M steps), two first warn on seeds 52
and 60 at 917,504 steps and never warn on seed 42 (through 6.03M), while
three first warn on seed 52 at 2,359,296 steps, 1.44M steps later, and on
seed 60 one evaluation later.  Over the 69 pad-rail series of 3M steps or
more from the earlier T. rex stance screens, every series two warn on, three
warn on too, 1 to 10 evaluations later, and none warns under two only.

An evaluation with no full-horizon episode has a NaN over-bar fraction
(nothing measured the posture), and NaN ends the streak of drifting
evaluations.  A seed whose evaluation episodes all fall therefore goes quiet
here exactly when it is worst; its falls show in ``progress.json``'s
``evaluation.falls`` and ``gate_progress.npz``'s ``full_horizon_fraction``.

Deliberately **advisory only**, like the below-baseline warning
(:mod:`.baseline_watch`): it never stops or changes training.  The rail is
the post-stage judge's, applied there to its own certification panel; a
posture over it mid-run may still settle, and the hop watch
(:mod:`.hop_watch`) is the opt-in stop for the one failure that was shown not
to.  A stage that declares neither fore-aft bar (``max_sole_cop_fore_aft``,
``max_foot_cop_fore_aft``) gets no watch.  The three keys configure a
diagnostic, like ``baseline_warn_after_budget_fraction``
(``gate_schema._DIAGNOSTIC_KEYS``), and enter no gate, recipe, task or
hyperparameter digest.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Mapping
from typing import Any

from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)

#: The ``[curriculum]`` keys this callback reads (``gate_schema._DIAGNOSTIC_KEYS``).
POSTURE_DRIFT_KEYS = (
    "posture_drift_warn_over_bar_fraction",
    "posture_drift_warn_consecutive_evals",
    "posture_drift_warn_repeat_evals",
)
#: The over-bar fraction at or above which an evaluation counts as drifting.
DEFAULT_OVER_BAR_FRACTION = 0.5
#: Drifting evaluations in a row that raise the warning.
DEFAULT_CONSECUTIVE_EVALS = 2
#: Evaluations after a warning before the next one may be printed.
DEFAULT_REPEAT_EVALS = 10

#: The fore-aft rails a stage may declare: ``(gate key, what it reads, the per-foot signed field that says which
#: edge)``.  Each rail's over-bar fraction is ``gate_progress.npz``'s ``<gate key>_over_bar_fraction``.
FORE_AFT_RAILS: tuple[tuple[str, str, str], ...] = (
    ("max_sole_cop_fore_aft", "pad centre of pressure", "sole_cop_fore_aft"),
    ("max_foot_cop_fore_aft", "whole-foot centre of pressure", "foot_cop_fore_aft"),
)


def posture_drift_settings(curriculum: Mapping[str, Any]) -> tuple[float, int, int]:
    """``(over_bar_fraction, consecutive_evals, repeat_evals)`` from a ``[curriculum]`` table, defaults filled in.

    Strict: a fraction outside ``(0, 1]`` and a count that is not a whole
    number of at least 1 raise ``ValueError`` naming the key, because a
    warning the reader silently disabled would read as "no drift" for a
    whole run.
    """
    fraction = curriculum.get("posture_drift_warn_over_bar_fraction", DEFAULT_OVER_BAR_FRACTION)
    if isinstance(fraction, bool) or not isinstance(fraction, numbers.Real) or not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"posture_drift_warn_over_bar_fraction must be a fraction in (0, 1], not {fraction!r}")
    counts = []
    for key, default in (
        ("posture_drift_warn_consecutive_evals", DEFAULT_CONSECUTIVE_EVALS),
        ("posture_drift_warn_repeat_evals", DEFAULT_REPEAT_EVALS),
    ):
        value = curriculum.get(key, default)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{key} must be a whole number of evaluations of at least 1, not {value!r}")
        counts.append(int(value))
    return float(fraction), counts[0], counts[1]


def declared_fore_aft_rails(curriculum: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    """``(gate key, bar)`` of each fore-aft rail the stage declares, in :data:`FORE_AFT_RAILS` order."""
    rails = []
    for key, _what, _signed in FORE_AFT_RAILS:
        bar = curriculum.get(key)
        if isinstance(bar, bool) or not isinstance(bar, numbers.Real) or not math.isfinite(float(bar)):
            continue
        rails.append((key, float(bar)))
    return tuple(rails)


class PostureDriftWatchCallback(BaseCallback):  # type: ignore[misc]
    """Warn, at most once per *repeat_evals* evaluations, while a fore-aft rail's over-bar fraction stays high.

    Args:
        plateau_callback: The ``StageGatePlateauCallback`` to read: its
            ``gate_progress_series`` for each rail's
            ``<key>_over_bar_fraction``, and its ``gate_progress_row`` for
            the latest row's per-foot signed CoP, which the warning line
            names.  It runs earlier in the same callback list, so each new
            evaluation is already recorded when this callback sees the step.
        rails: ``(gate key, bar)`` per declared fore-aft rail
            (:func:`declared_fore_aft_rails`).
        over_bar_fraction: An evaluation drifts when a rail's over-bar
            fraction is at least this (NaN, an evaluation with no
            full-horizon episode, never does, and ends the streak of
            drifting ones: a seed whose episodes all fall goes quiet here,
            and its falls show in ``progress.json``).
        consecutive_evals: Drifting evaluations in a row that warn.
        repeat_evals: Evaluations after a warning before the next may print.
    """

    #: Class-scope defaults so an instance built without ``__init__`` (as the
    #: tests do, to exercise ``_on_step`` without stable-baselines3) has them.
    _last_seen = 0
    _last_warned_row: "int | None" = None

    def __init__(
        self,
        plateau_callback: Any,
        *,
        rails: "tuple[tuple[str, float], ...]",
        over_bar_fraction: float = DEFAULT_OVER_BAR_FRACTION,
        consecutive_evals: int = DEFAULT_CONSECUTIVE_EVALS,
        repeat_evals: int = DEFAULT_REPEAT_EVALS,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PostureDriftWatchCallback.")
        super().__init__(verbose)
        self.plateau_callback = plateau_callback
        self.rails = tuple((str(key), float(bar)) for key, bar in rails)
        self.over_bar_fraction = float(over_bar_fraction)
        self.consecutive_evals = int(consecutive_evals)
        self.repeat_evals = int(repeat_evals)
        self._last_seen = 0
        self._last_warned_row = None
        self._streaks: dict[str, list[tuple[int, float]]] = {}
        #: The warnings raised, for ``progress.json``: ``{"timesteps", "source", "rail", "message"}``.
        self.progress_warnings: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        series = {
            key: self.plateau_callback.gate_progress_series(f"{key}_over_bar_fraction") for key, _bar in self.rails
        }
        if not series:
            return True
        timesteps = next(iter(series.values()))[0]
        start, self._last_seen = self._last_seen, len(timesteps)
        streaks = self.__dict__.setdefault("_streaks", {})
        for row in range(start, len(timesteps)):
            drifting = []
            for key, bar in self.rails:
                fraction = series[key][1][row]
                run = streaks.setdefault(key, [])
                if math.isfinite(fraction) and fraction >= self.over_bar_fraction:
                    run.append((int(timesteps[row]), float(fraction)))
                    del run[: -self.consecutive_evals]
                else:
                    run.clear()
                if len(run) >= self.consecutive_evals:
                    drifting.append((key, bar, list(run)))
            last = self.__dict__.get("_last_warned_row")
            if drifting and (last is None or row - last >= self.repeat_evals):
                self._last_warned_row = row
                self._warn(int(timesteps[row]), drifting)
        return True

    def _feet(self, signed_field: str) -> str:
        """The latest evaluation's signed per-foot CoP of one rail, as ``"r +1.12 (toe side), l +0.95 (toe side)"``."""
        accessor = getattr(self.plateau_callback, "gate_progress_row", None)
        row: Mapping[str, float] = accessor() if callable(accessor) else {}
        parts = []
        for key, value in row.items():
            if not key.endswith(f"_{signed_field}") or not math.isfinite(value):
                continue
            side = "toe side" if value > 0 else "heel side" if value < 0 else "centred"
            parts.append(f"{key[: -len(signed_field) - 1]} {value:+.2f} ({side})")
        return ", ".join(parts)

    def _warn(self, timestep: int, drifting: "list[tuple[str, float, list[tuple[int, float]]]]") -> None:
        what = {key: (reads, signed) for key, reads, signed in FORE_AFT_RAILS}
        clauses = []
        for key, bar, run in drifting:
            reads, signed = what.get(key, (key, ""))
            feet = self._feet(signed) if signed else ""
            clauses.append(
                f"{key} ({reads}) over its {bar:.2f} bar on "
                f"{', '.join(f'{fraction:.2f}' for _step, fraction in run)} of the full-horizon panel episodes "
                f"at {run[0][0]:,} to {run[-1][0]:,} steps"
                + (f"; latest signed per foot, in half-lengths: {feet}" if feet else "")
            )
        message = (
            f"PostureDrift at {timestep:,} steps: {'; '.join(clauses)} (warns at {self.over_bar_fraction:.2f} or more "
            f"on {self.consecutive_evals} evaluations in a row). The post-stage judge refuses an episode over this "
            "rail. Advisory only: training continues unchanged; progress.json and gate_progress.npz hold the series."
        )
        logger.warning("%s", message)
        self.__dict__.setdefault("progress_warnings", []).append(
            {
                "timesteps": timestep,
                "source": "posture_drift",
                "rails": [key for key, _bar, _run in drifting],
                "message": message,
            }
        )


def build_posture_drift_watch_callback(
    plateau_callback: Any,
    stage_config: Mapping[str, Any],
    *,
    verbose: int = 0,
) -> PostureDriftWatchCallback | None:
    """Build the watch, or ``None`` when the stage declares no fore-aft centre-of-pressure bar.

    ``None`` rather than a no-op keeps the absence visible in the callback
    list.  Malformed knobs raise (:func:`posture_drift_settings`), whether or
    not a bar is declared.
    """
    curriculum = stage_config.get("curriculum_kwargs", {}) or {}
    fraction, consecutive, repeat = posture_drift_settings(curriculum)
    rails = declared_fore_aft_rails(curriculum)
    if not rails or plateau_callback is None:
        return None
    return PostureDriftWatchCallback(
        plateau_callback,
        rails=rails,
        over_bar_fraction=fraction,
        consecutive_evals=consecutive,
        repeat_evals=repeat,
        verbose=verbose,
    )
