"""Contact segmentation and continuous limb phase, without simulator dependencies.

Segmentation (per limb, floor normal force only):

1. A sample is loaded when the force exceeds the protocol's contact threshold.
2. An unload between two loads is merged back into stance when it is
   sub-dwell chatter, a scuff (never clears the ground and barely moves the
   foot, or is short) or an impact bounce (short and low relative to the
   limb's own reference swing). The reference swing is the upper quartile of
   the limb's clear unloads, so it stays on a real swing even when bounces are
   nearly half of them.
3. Loads shorter than the blip time are dropped only *after* merging, so a
   touchdown is the first contact of a stance and an impact bounce never
   delays it.

The continuous phase of a limb rises linearly from 0 to 1 between consecutive
events (touchdowns, or mid-stances for the diagnostic anchor). A stride longer
than ``pause_factor`` times its *local reference* is a pause: the phase is
undefined for its whole duration. The local reference of a stride is the
cadence of its neighbouring strides of the same limb on either side
(``stride_references``), not the whole-episode median, so a genuine change of
cadence (a walk-to-run transition halves the stride) is not mistaken for a
pause, while a stop between normal strides still is. Before the first and after the last
event the phase is extrapolated with the nearest stride's reference for as long
as a normal stride may last (``pause_factor`` references) and is undefined
beyond, because the next event cannot be observed there.

The clock is any nondecreasing time base: the measurement passes the moving
clock, which stops while the trunk stands, so a stop does not stretch the
stride around it.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from environments.shared.gait.types import GaitProtocol, LimbContacts


def circular_distance(phase: ArrayLike, target: float) -> NDArray[np.float64]:
    """Shortest distance on a unit-period phase circle."""
    return cast(NDArray[np.float64], np.abs((np.asarray(phase, dtype=float) - target + 0.5) % 1.0 - 0.5))


def boolean_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Maximal runs of true samples as ``[start, end)`` index pairs."""
    padded = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1).tolist()
    ends = np.flatnonzero(changes == -1).tolist()
    return list(zip(starts, ends, strict=True))


def segment_limb(
    time_s: NDArray[np.float64],
    force_n: NDArray[np.float64],
    clearance_m: NDArray[np.float64],
    foot_xy_m: NDArray[np.float64],
    *,
    threshold_n: float,
    leg_length_m: float,
    protocol: GaitProtocol,
) -> LimbContacts:
    """Merge chatter, scuffs and impact bounces into stance; drop blips."""
    count = len(time_s)
    runs = boolean_runs(force_n > threshold_n)
    gaps: list[tuple[float, float, float]] = []
    for (_, unload), (start, _) in zip(runs[:-1], runs[1:], strict=True):
        lifted = float(np.max(clearance_m[unload:start])) if start > unload else 0.0
        dx, dy = (foot_xy_m[start] - foot_xy_m[unload - 1]).tolist()
        gaps.append((float(time_s[start] - time_s[unload - 1]), lifted, math.hypot(dx, dy)))
    scuff_height = protocol.scuff_clearance_over_leg * leg_length_m
    clear = [
        (duration, lifted)
        for duration, lifted, _ in gaps
        if lifted >= scuff_height and duration > protocol.chatter_fill_s
    ]
    reference_duration = float(np.percentile([d for d, _ in clear], 75)) if len(clear) >= 3 else None
    reference_clearance = float(np.percentile([c for _, c in clear], 75)) if len(clear) >= 3 else None
    merged: list[list[int]] = []
    merged_unloads = 0
    for index, (start, end) in enumerate(runs):
        if merged:
            duration, lifted, moved = gaps[index - 1]
            short = reference_duration is not None and duration < protocol.bounce_swing_fraction * reference_duration
            chatter = duration <= protocol.chatter_fill_s + 1e-9
            scuff = lifted < scuff_height and (moved < protocol.scuff_travel_over_leg * leg_length_m or short)
            bounce = (
                short
                and reference_clearance is not None
                and lifted < protocol.bounce_clearance_fraction * reference_clearance
            )
            if chatter or scuff or bounce:
                merged[-1][1] = end
                merged_unloads += 1
                continue
        merged.append([start, end])
    kept = [
        (start, end)
        for start, end in merged
        if start == 0 or end == count or (time_s[min(end, count - 1)] - time_s[start]) >= protocol.blip_s - 1e-9
    ]
    stance = np.zeros(count, dtype=bool)
    for start, end in kept:
        stance[start:end] = True
    touchdowns: list[int] = []
    swing_clearance: list[float] = []
    swing_travel: list[float] = []
    previous_end: int | None = None
    for start, end in kept:
        if start > 0:
            begin = previous_end if previous_end is not None else 0
            touchdowns.append(start)
            swing_clearance.append(float(np.max(clearance_m[begin:start])) if start > begin else 0.0)
            dx, dy = (foot_xy_m[start] - foot_xy_m[max(begin - 1, 0)]).tolist()
            swing_travel.append(math.hypot(dx, dy))
        previous_end = end
    valid = tuple(
        bool(
            lifted >= protocol.valid_swing_clearance_over_leg * leg_length_m
            and moved >= protocol.valid_swing_travel_over_leg * leg_length_m
        )
        for lifted, moved in zip(swing_clearance, swing_travel, strict=True)
    )
    return LimbContacts(
        stance=stance,
        stances=tuple(kept),
        touchdowns=tuple(touchdowns),
        swing_clearance_m=tuple(swing_clearance),
        swing_travel_m=tuple(swing_travel),
        swing_valid=valid,
        merged_unloads=merged_unloads,
    )


def stride_references(periods_s: ArrayLike, neighbours: int) -> NDArray[np.float64]:
    """Local cadence reference of each stride, from its neighbours on either side.

    Each side's cadence is the median of up to ``neighbours`` strides of the
    same limb on that side; the reference is the larger of the two. The
    stride itself is excluded, so a stop cannot raise its own reference, and
    a stride is a pause only when it is long against the cadence on *both*
    sides: at a walk-to-run transition every stride matches the cadence of
    its own side. A side median of three strides still sees through a second
    stop next to the first. The first and last strides have one side only;
    there the stride itself joins that side's median, so a cadence change
    right after the first stride is not a pause while a stop followed by
    normal strides still is. A limb with a single stride has no neighbours;
    its reference is the stride itself (never a pause).
    """
    periods = np.asarray(periods_s, dtype=float)
    count = len(periods)
    references = np.empty(count)
    for k in range(count):
        before, after = periods[max(0, k - neighbours) : k], periods[k + 1 : k + 1 + neighbours]
        if len(before) and len(after):
            references[k] = max(float(np.median(before)), float(np.median(after)))
        elif len(before) or len(after):
            references[k] = float(np.median(np.concatenate((before, periods[k : k + 1], after))))
        else:
            references[k] = float(periods[k])
    return references


def stride_is_normal(periods_s: ArrayLike, references_s: ArrayLike, pause_factor: float) -> NDArray[np.bool_]:
    """Strides that are not pauses: positive and no longer than ``pause_factor`` local references."""
    periods = np.asarray(periods_s, dtype=float)
    return cast(NDArray[np.bool_], (periods > 0.0) & (periods <= pause_factor * np.asarray(references_s, dtype=float)))


def limb_phase(
    time_s: NDArray[np.float64],
    event_times_s: ArrayLike,
    reference_period_s: float | ArrayLike,
    pause_factor: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Continuous phase in ``[0, 1)`` on every sample and its definedness mask.

    ``reference_period_s`` is either one reference for every stride or the
    per-stride local references (``stride_references``, one per interval
    between consecutive events).
    """
    count = len(time_s)
    phase = np.zeros(count)
    defined = np.zeros(count, dtype=bool)
    events = np.asarray(event_times_s, dtype=float)
    if len(events) == 0:
        return phase, defined
    reference = np.asarray(reference_period_s, dtype=float)
    if reference.ndim == 0:
        references = np.full(max(len(events) - 1, 0), float(reference))
        first_reference = last_reference = float(reference)
    else:
        if reference.shape != (max(len(events) - 1, 0),):
            raise ValueError("per-stride references must have one value per interval between events")
        if len(reference) == 0:
            return phase, defined
        references = reference
        first_reference, last_reference = float(reference[0]), float(reference[-1])
    stride = np.searchsorted(events, time_s, side="right") - 1
    if len(events) >= 2:
        inside = (stride >= 0) & (stride < len(events) - 1)
        clipped = np.clip(stride, 0, len(events) - 2)
        period = events[clipped + 1] - events[clipped]
        # a stride of zero length on the clock (two touchdowns while the
        # moving clock stands still) has no phase
        normal = inside & (period > 0.0) & (period <= pause_factor * references[clipped])
        phase[normal] = (time_s[normal] - events[clipped[normal]]) / period[normal]
        defined |= normal
    if last_reference > 0.0:
        elapsed = time_s - events[-1]
        after = (stride >= len(events) - 1) & (elapsed < pause_factor * last_reference)
        phase[after] = elapsed[after] / last_reference
        defined |= after
    if first_reference > 0.0:
        lead = events[0] - time_s
        before = (stride < 0) & (lead <= pause_factor * first_reference)
        phase[before] = 1.0 - lead[before] / first_reference
        defined |= before
    return np.mod(phase, 1.0), defined
