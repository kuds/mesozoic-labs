"""Finite-window, time-based contact events without simulator dependencies."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from environments.shared.gait.types import ContactEvents, GaitProtocol


def circular_distance(phase: ArrayLike, target: float) -> NDArray[np.float64]:
    """Shortest distance on a unit-period phase circle."""
    return cast(NDArray[np.float64], np.abs((np.asarray(phase, dtype=float) - target + 0.5) % 1.0 - 0.5))


def contact_events(
    time_s: ArrayLike,
    force_n: ArrayLike,
    *,
    body_weight_n: float,
    protocol: GaitProtocol,
) -> ContactEvents:
    """Detect support with Schmitt hysteresis and seconds-based dwell.

    Once a transition's dwell is confirmed, its timestamp is attributed to the
    beginning of that dwell. Short unload/reload chatter produces no steps.
    A complete cycle is an observed touchdown, exactly one liftoff, then the
    next observed touchdown. No stride is extrapolated across either boundary.
    """
    time = np.asarray(time_s, dtype=float)
    force = np.asarray(force_n, dtype=float)
    if time.ndim != 1 or force.shape != time.shape or len(time) < 2:
        raise ValueError("contact events require matching one-dimensional arrays with at least two samples")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("contact event timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(force)) or np.any(force < 0.0):
        raise ValueError("contact force must be finite and nonnegative")
    if not np.isfinite(body_weight_n) or body_weight_n <= 0.0:
        raise ValueError("body weight must be positive and finite")

    on = protocol.contact_on_bw * body_weight_n
    off = protocol.contact_off_bw * body_weight_n
    raw: NDArray[np.bool_] = np.empty(len(time), dtype=bool)
    raw[0] = force[0] >= on
    for index in range(1, len(time)):
        raw[index] = force[index] > off if raw[index - 1] else force[index] >= on

    loaded: NDArray[np.bool_] = np.empty(len(time), dtype=bool)
    state = bool(raw[0])
    loaded[0] = state
    candidate_start: int | None = None
    touchdowns: list[int] = []
    liftoffs: list[int] = []
    for index in range(1, len(time)):
        loaded[index] = state
        if bool(raw[index]) == state:
            candidate_start = None
            continue
        if candidate_start is None:
            candidate_start = index
        dwell = protocol.min_stance_s if raw[index] else protocol.min_swing_s
        if time[index] - time[candidate_start] + 1e-12 < dwell:
            continue
        state = bool(raw[index])
        loaded[candidate_start : index + 1] = state
        (touchdowns if state else liftoffs).append(candidate_start)
        candidate_start = None

    cycles: list[tuple[int, int, int]] = []
    for start, end in zip(touchdowns, touchdowns[1:], strict=False):
        between = [index for index in liftoffs if start < index < end]
        if len(between) == 1:
            cycles.append((start, between[0], end))
    return ContactEvents(loaded, tuple(touchdowns), tuple(liftoffs), tuple(cycles))
