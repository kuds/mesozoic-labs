"""The floor-truth down state and its debounced events, with every window defined in SECONDS.

Pure numpy.  A leg is DOWN on a control step when its whole-limb floor force
exceeds :data:`.constants.CONTACT_THRESHOLD_N` on at least
:data:`.constants.DOWN_SUBSTEP_FRACTION` of the step's substeps
(:func:`down_mask` over ``EpisodeTrace.leg_down_frac``).  Touchdowns and
lift-offs are the edges of that state after :func:`debounce`, whose run
length comes from :data:`.constants.DEBOUNCE_S` through :func:`steps_for`:
2 steps at dt 0.01, but 1 step -- the identity -- at dt 0.02, so a 20 ms
flicker means the same thing on every species (GAIT_QUALITY_PLAN_2026_09.md
§3.1 "events.py").  The locomotion work's simultaneous-touchdown merge (the
plan's ``B`` events) belongs here too when it lands.
"""

from __future__ import annotations

import numpy as np

from .constants import DOWN_SUBSTEP_FRACTION


def steps_for(seconds: float, dt: float) -> int:
    """Whole control steps covering ``seconds`` at step ``dt``: ``max(1, round(seconds / dt))``."""
    return max(1, int(round(seconds / dt)))


def down_mask(leg_down_frac: np.ndarray) -> np.ndarray:
    """Per step (and leg), whether the leg is down: ``leg_down_frac >= DOWN_SUBSTEP_FRACTION``."""
    return np.asarray(leg_down_frac, dtype=np.float64) >= DOWN_SUBSTEP_FRACTION


def _runs(flags: np.ndarray) -> list[tuple[bool, int, int]]:
    """Run-length encoding ``[(value, start, length), ...]`` of a boolean series."""
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return []
    change = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [flags.size]])
    return [(bool(flags[start]), int(start), int(end - start)) for start, end in zip(starts, ends)]


def debounce(flags: np.ndarray, min_run_steps: int) -> np.ndarray:
    """``flags`` with its INTERIOR runs shorter than ``min_run_steps`` removed: short gaps filled first, then short blips dropped.

    The audit probe's ``debounce`` (``docs/investigations/gait_2026_09/gait_probe.py``),
    unchanged, so the calibration it produced transfers.  A run touching
    either end of the series is kept whatever its length -- the episode's
    first and last states are what they are.  Filling gaps before dropping
    blips means a foot that chatters down-up-down inside the window reads as
    one stance, not as a blip.  ``min_run_steps <= 1`` is the identity.
    """
    out = np.asarray(flags, dtype=bool).copy()
    if min_run_steps <= 1:
        return out
    for value, start, length in _runs(out):
        if not value and length < min_run_steps and start > 0 and start + length < out.size:
            out[start : start + length] = True
    for value, start, length in _runs(out):
        if value and length < min_run_steps and start > 0 and start + length < out.size:
            out[start : start + length] = False
    return out


def rising_edges(flags: np.ndarray) -> np.ndarray:
    """Indices ``k >= 1`` with ``flags[k]`` set and ``flags[k - 1]`` clear (touchdowns)."""
    return np.flatnonzero(np.diff(np.asarray(flags, dtype=np.int8)) == 1) + 1


def falling_edges(flags: np.ndarray) -> np.ndarray:
    """Indices ``k >= 1`` with ``flags[k]`` clear and ``flags[k - 1]`` set (lift-offs)."""
    return np.flatnonzero(np.diff(np.asarray(flags, dtype=np.int8)) == -1) + 1


def count_runs(flags: np.ndarray) -> int:
    """The number of maximal runs of set flags."""
    return sum(1 for value, _start, _length in _runs(flags) if value)
