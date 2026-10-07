"""Stop a stance run that is still hopping when its exploration has run out.

The two T. rex physics-r8 stance runs (``20261006_185343`` seed 42 and
``20261006_185704`` seed 44) trained one byte-identical config for 11M steps
and split on a coin: both entered a hop regime in the first million steps
(evaluation unsupported duty 0.13 at 1M, 0.32-0.42 at 4M), seed 44 escaped
it between 5.0M and 6.0M while the entropy bonus was still non-zero (duty
under 0.05 from 5.35M, 0.0001 at 6M, 0.013 at 7M and at most 0.018 from 7M
on, 2.8x under the T. rex stance's 0.05 bar), and seed 42 locked into a
12.5 Hz two-foot micro-hop just before the bonus reached zero at 7M
(``ent_coef_decay_timesteps``) and held exactly 0.375 at every evaluation
from 6.5M to 11M.  Nothing in the run said so; the stance_quality/v2 report
refused the checkpoint at the end (0/40 clean), four million steps later.

This callback reads the evaluation unsupported duty the plateau diagnostics
already record into ``gate_progress.npz`` (``StageGatePlateauCallback``, the
full-horizon panel mean of ``derive_stance_info``'s duty at 0.1 N after the
settle) and, at the first evaluation at or after ``after_timesteps`` whose
duty is above the bar, warns once, writes ``hop_watch.json`` into the stage
directory and -- when ``stop`` -- ends training.  The stage then runs its
normal post-stage report on the handoff pair: a stop certifies nothing and
skips no judgement, and ``hop_watch.json`` records why the run ended early
(the remedy is a new seed, not a longer run).

Opt-in, per stage: :func:`build_hop_watch_callback` returns ``None`` unless the
stage's ``[curriculum]`` declares ``hop_watch_max_unsupported_duty``, so every
other stage trains exactly as before.  The three keys configure early
stopping, like the collapse keys, and enter no gate, recipe or task digest.
"""

from __future__ import annotations

import json
import logging
import math
import numbers
from pathlib import Path
from typing import Any

from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)

#: The ``[curriculum]`` keys this callback reads (``gate_schema._COLLAPSE_KEYS``).
HOP_WATCH_KEYS = ("hop_watch_max_unsupported_duty", "hop_watch_after_timesteps", "hop_watch_stop")

#: The file a tripped watch writes into the stage directory.
HOP_WATCH_FILENAME = "hop_watch.json"


def hop_watch_settings(curriculum: dict[str, Any]) -> tuple[float, int, bool] | None:
    """``(max_unsupported_duty, after_timesteps, stop)`` from a ``[curriculum]`` table, or ``None`` when undeclared.

    Strict once declared: a duty bar outside ``[0, 1]``, a negative or
    fractional ``hop_watch_after_timesteps`` and a non-boolean
    ``hop_watch_stop`` raise ``ValueError`` naming the key, because a guard
    the reader silently disabled would read as "no hop" for a whole run.
    ``hop_watch_after_timesteps`` defaults to 0 and ``hop_watch_stop`` to
    ``False`` (warn and record only); the two may not be declared without
    the bar.
    """
    if "hop_watch_max_unsupported_duty" not in curriculum:
        stray = sorted(key for key in HOP_WATCH_KEYS[1:] if key in curriculum)
        if stray:
            raise ValueError(f"{stray} configure a hop watch, but hop_watch_max_unsupported_duty is not declared")
        return None
    bar = curriculum["hop_watch_max_unsupported_duty"]
    if isinstance(bar, bool) or not isinstance(bar, numbers.Real) or not 0.0 <= float(bar) <= 1.0:
        raise ValueError(f"hop_watch_max_unsupported_duty must be a duty in [0, 1], not {bar!r}")
    after = curriculum.get("hop_watch_after_timesteps", 0)
    if isinstance(after, bool) or not isinstance(after, numbers.Integral) or after < 0:
        raise ValueError(f"hop_watch_after_timesteps must be a whole number of steps of at least 0, not {after!r}")
    stop = curriculum.get("hop_watch_stop", False)
    if not isinstance(stop, bool):
        raise ValueError(f"hop_watch_stop must be true or false, not {stop!r}")
    return float(bar), int(after), stop


class HopRegimeWatchCallback(BaseCallback):  # type: ignore[misc]
    """Warn, record and optionally stop when the evaluation unsupported duty stays above a bar late in training.

    Args:
        plateau_callback: The ``StageGatePlateauCallback`` whose
            ``gate_progress_series("unsupported_duty")`` to read.  It runs
            earlier in the same callback list, so each new evaluation is
            already recorded when this callback sees the step.
        max_unsupported_duty: The bar; an evaluation above it trips the watch.
        after_timesteps: Evaluations before this cumulative step are ignored.
        stop: End training when the watch trips (``False``: warn and record only).
        stage_dir: Where ``hop_watch.json`` goes; ``None`` writes nothing.
    """

    #: Class-scope defaults so an instance built without ``__init__`` (as the
    #: tests do, to exercise ``_on_step`` without stable-baselines3) has them.
    _tripped = False
    _last_seen = 0

    def __init__(
        self,
        plateau_callback: Any,
        *,
        max_unsupported_duty: float,
        after_timesteps: int,
        stop: bool,
        stage_dir: "str | Path | None",
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for HopRegimeWatchCallback.")
        super().__init__(verbose)
        self.plateau_callback = plateau_callback
        self.max_unsupported_duty = float(max_unsupported_duty)
        self.after_timesteps = int(after_timesteps)
        self.stop = bool(stop)
        self.stage_dir = None if stage_dir is None else Path(stage_dir)
        self._tripped = False
        self._last_seen = 0

    def _on_step(self) -> bool:
        if self._tripped:
            return not self.stop
        # Nothing before the window can trip, so the series is not even read there (this runs every step).
        if getattr(self, "num_timesteps", self.after_timesteps) < self.after_timesteps:
            return True
        timesteps, duties = self.plateau_callback.gate_progress_series("unsupported_duty")
        start, self._last_seen = self._last_seen, len(timesteps)
        for timestep, duty in zip(timesteps[start:], duties[start:]):
            # NaN (no full-horizon episode to measure) never trips: a panel
            # that falls is the collapse backstop's to judge, not this one's.
            if timestep < self.after_timesteps or not math.isfinite(duty) or duty <= self.max_unsupported_duty:
                continue
            self._trip(int(timestep), float(duty))
            return not self.stop
        return True

    def _trip(self, timestep: int, duty: float) -> None:
        self._tripped = True
        action = "stopping training" if self.stop else "training continues (hop_watch_stop = false)"
        logger.warning(
            "HopWatch: evaluation unsupported duty %.4f > %.4f at %d steps (watching from %d); %s. A stance still "
            "hopping when its entropy bonus has run out did not escape the hop regime on the T. rex physics-r8 "
            "runs; the post-stage report judges the handoff pair as usual, and the remedy is a new seed.",
            duty,
            self.max_unsupported_duty,
            timestep,
            self.after_timesteps,
            action,
        )
        if self.stage_dir is None:
            return
        payload = {
            "schema": "mesozoic.hop-watch/v1",
            "timesteps": timestep,
            "unsupported_duty": duty,
            "max_unsupported_duty": self.max_unsupported_duty,
            "after_timesteps": self.after_timesteps,
            "action": "stopped" if self.stop else "warned",
        }
        try:
            from ..file_io import atomic_write_text

            atomic_write_text(self.stage_dir / HOP_WATCH_FILENAME, json.dumps(payload, indent=2, sort_keys=True))
        except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
            logger.warning("Could not write %s", self.stage_dir / HOP_WATCH_FILENAME, exc_info=True)


def build_hop_watch_callback(
    plateau_callback: Any,
    stage_config: dict[str, Any],
    *,
    stage_dir: "str | Path | None",
    verbose: int = 0,
) -> HopRegimeWatchCallback | None:
    """Build the watch, or ``None`` when the stage declares no ``hop_watch_max_unsupported_duty``.

    ``None`` rather than a no-op keeps the absence visible in the callback
    list.  A declared but malformed block raises (:func:`hop_watch_settings`).
    """
    settings = hop_watch_settings(stage_config.get("curriculum_kwargs", {}))
    if settings is None or plateau_callback is None:
        return None
    bar, after, stop = settings
    return HopRegimeWatchCallback(
        plateau_callback,
        max_unsupported_duty=bar,
        after_timesteps=after,
        stop=stop,
        stage_dir=stage_dir,
        verbose=verbose,
    )
