"""Serializable physical gait metrics from immutable physics-sample telemetry.

Gaits are measured as phase-locked patterns of limb oscillators (Collins and
Stewart 1993; Golubitsky et al. 1999). Each limb's continuous phase rises from
0 to 1 between its touchdowns; for every limb pair the relative phase is
summarised by its circular mean and mean resultant length (the pair's
phase-locking index), by a centred one-stride sliding estimate and by
Hildebrand's discrete footfall estimate pooled in both directions. Every
walk-first profile templates its contralateral pairs at anti-phase (left and
right alternate, on four legs within each girdle; no limb-phase partition).
The off-gait fraction is one persistence budget: the time a limb is not
cycling, a contralateral pair is locked well away from anti-phase (a hop, a
bound, a skip, a gallop), the pairs are uncoordinated for two strides or
more, or a step does not step through. Physical rails measure limb and
girdle participation (whole-window and stride by stride), weight-bearing
stance, swing validity, stride and step length, swing-phase floor contact,
sliding and gliding stances, non-foot support, foot-on-foot support, trunk
height and flight.

Footprints are load-weighted stance centres (force x time), so a light toe
touch or a slide during stance does not move them. Step length (the
step-through test) compares each footprint with the contralateral foot's
previous footprint along the trunk's own axis, the root body's x axis from
the recorded root quaternion averaged over one stride, so neither a crabbing
trunk nor a heading drift changes it. Stride length and skid travel are
measured along the local travel heading (the trunk displacement over a
centred three-stride window). Only the progress rail (``mean_speed_mps``) is
projected on the declared task direction.

Only recorder keys are read; the root quaternion is required telemetry, so a
trace without it fails closed. Every reduction that reaches a stored metric is
order-deterministic: ``math.fsum`` for global sums, sequential ``np.cumsum``
for sliding windows, scalar ``math`` trigonometry, and IEEE-exact elementwise
arithmetic for local on-template tests (no BLAS reductions and no SIMD
transcendental ufuncs). Stored values are rounded to six decimals, and the
gate judges only the stored values, so a reader re-judging ``metrics_json``
and the in-process verdict always agree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from environments.shared.gait.events import (
    boolean_runs,
    limb_phase,
    segment_limb,
    stride_is_normal,
    stride_references,
)
from environments.shared.gait.labels import circular_mean, stride_labels
from environments.shared.gait.types import GaitProtocol, LimbContacts

GRAVITY_MPS2 = 9.81
DIGITS = 6
BIPED_FEET = ("r", "l")
QUADRUPED_FEET = ("fr", "fl", "rr", "rl")
CONTRALATERAL_PAIRS: dict[int, tuple[tuple[str, str], ...]] = {
    2: (("r", "l"),),
    4: (("fr", "fl"), ("rr", "rl")),
}
#: The template every walk-first profile reads: contralateral anti-phase.
ALTERNATION_TEMPLATE = "alternation"
_REQUIRED_FIELDS = (
    "time_s",
    "floor_force_n",
    "foot_position_m",
    "foot_clearance_m",
    "slip_speed_mps",
    "body_floor_force_n",
    "foot_foot_force_n",
    "root_position_m",
    "root_quat_wxyz",
)
#: Minimum stance-to-trunk travel denominator of the skid fraction, in L.
_SKID_MIN_TRAVEL_OVER_LEG = 0.05
#: A stance resolves its own typical slip speed (glide) only with at least this many samples.
_GLIDE_MIN_SAMPLES = 3
#: Duty-normalised stance overlap is undefined when the duty factors leave
#: less than this between its anti-phase and in-phase extremes.
_OVERLAP_MIN_SPAN = 0.10
#: Alternation-index touchdown merge window, in pooled strides.
_ALTERNATION_MERGE_STRIDES = 0.05
#: A root quaternion shorter than this is not an orientation.
_MIN_QUATERNION_NORM = 1e-6

Target = tuple[float, float]


# ---------------------------------------------------------------------------
# deterministic numeric helpers
# ---------------------------------------------------------------------------


def _r(value: Any, digits: int = DIGITS) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number, digits) + 0.0 if math.isfinite(number) else None


def _fsum(values: Any) -> float:
    return math.fsum(np.asarray(values, dtype=float).ravel().tolist())


def _cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros(len(values) + 1)
    np.cumsum(values, out=out[1:])
    return out


def _unit_circle(phase: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """cos/sin of 2*pi*phase with scalar libm calls (no SIMD-dispatched ufuncs)."""
    angles = (2.0 * math.pi * phase).tolist()
    return np.array([math.cos(a) for a in angles], dtype=float), np.array([math.sin(a) for a in angles], dtype=float)


def _circular_stats(mean_cos: float, mean_sin: float) -> tuple[float | None, float]:
    length = math.hypot(mean_cos, mean_sin)
    if length < 1e-12:
        return None, 0.0
    mean = math.atan2(mean_sin, mean_cos) / (2.0 * math.pi)
    return mean - math.floor(mean), length


def _cdist(phase: float, target: float) -> float:
    d = (phase - target) % 1.0
    return min(d, 1.0 - d)


def _median(values: Sequence[float] | NDArray[np.float64]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.median(array)) if array.size else math.nan


def _empty(errors: list[str], foot_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "telemetry_valid": False,
        "telemetry_errors": errors,
        "n_limbs": len(foot_names),
        "duration_s": None,
        "recorded_span_s": None,
        "max_sample_interval_s": None,
        "mean_speed_mps": None,
        "flight_fraction": None,
        "body_support_fraction": None,
        "foot_foot_contact_fraction": None,
        "per_foot": {},
        "contralateral": {},
        "pair_phase": {},
        "templates": {},
        "gait_label": None,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def episode_gait_metrics(
    trace: Mapping[str, Any],
    *,
    body_weight_n: float,
    leg_length_m: float,
    foot_names: tuple[str, ...],
    protocol: GaitProtocol,
    settle_s: float = 0.0,
    direction_xy: tuple[float, float] = (1.0, 0.0),
) -> dict[str, Any]:
    """Measure a single episode; missing/NaN/invalid telemetry fails closed.

    Array fields follow the recorder contract. ``touch_force_n`` is optional
    diagnostic telemetry, and is validated when present. Integration holds
    each sample until the next one over the analysis window, which starts
    ``settle_s`` after the first sample. Arrays are never mutated.
    """
    foot_names = tuple(foot_names)
    missing = [name for name in _REQUIRED_FIELDS if name not in trace]
    if missing:
        return _empty([f"missing telemetry: {name}" for name in missing], foot_names)
    try:
        arrays = {name: np.asarray(trace[name], dtype=float) for name in _REQUIRED_FIELDS}
        if "touch_force_n" in trace:
            arrays["touch_force_n"] = np.asarray(trace["touch_force_n"], dtype=float)
    except (TypeError, ValueError) as error:
        return _empty([f"non-numeric telemetry: {error}"], foot_names)
    time = arrays["time_s"]
    if time.ndim != 1 or len(time) < 2:
        return _empty(["time_s must contain at least two one-dimensional samples"], foot_names)
    count, feet = len(time), len(foot_names)
    shapes = {
        "floor_force_n": (count, feet),
        "foot_position_m": (count, feet, 3),
        "foot_clearance_m": (count, feet),
        "slip_speed_mps": (count, feet),
        "body_floor_force_n": (count,),
        "foot_foot_force_n": (count,),
        "root_position_m": (count, 3),
        "root_quat_wxyz": (count, 4),
        "touch_force_n": (count, feet),
    }
    errors = [
        f"invalid shape: {name}" for name, shape in shapes.items() if name in arrays and arrays[name].shape != shape
    ]
    errors += [f"non-finite telemetry: {name}" for name, array in arrays.items() if not np.all(np.isfinite(array))]
    if not np.all(np.diff(time) > 0.0):
        errors.append("time_s must increase strictly; reset boundaries require separate episodes")
    if sorted(foot_names) not in (sorted(BIPED_FEET), sorted(QUADRUPED_FEET)):
        errors.append("foot_names must be the distinct biped feet (r, l) or quadruped feet (fr, fl, rr, rl)")
    if not np.isfinite(body_weight_n) or body_weight_n <= 0.0:
        errors.append("body_weight_n must be positive and finite")
    if not np.isfinite(leg_length_m) or leg_length_m <= 0.0:
        errors.append("leg_length_m must be positive and finite")
    if not np.isfinite(settle_s) or settle_s < 0.0:
        errors.append("settle_s must be nonnegative and finite")
    direction = np.asarray(direction_xy, dtype=float).copy()
    if direction.shape != (2,) or not np.all(np.isfinite(direction)) or math.hypot(*direction.tolist()) <= 0.0:
        errors.append("direction_xy must be a finite, nonzero two-dimensional direction")
    for name in ("floor_force_n", "body_floor_force_n", "foot_foot_force_n", "slip_speed_mps", "touch_force_n"):
        if name in arrays and np.any(arrays[name] < 0.0):
            errors.append(f"negative telemetry: {name}")
    quaternion = arrays["root_quat_wxyz"]
    if quaternion.shape == (count, 4) and np.all(np.isfinite(quaternion)):
        if np.any(np.sqrt(np.sum(quaternion * quaternion, axis=1)) < _MIN_QUATERNION_NORM):
            errors.append("degenerate telemetry: root_quat_wxyz is not a rotation")
    if errors:
        return _empty(errors, foot_names)
    if time[-1] - (time[0] + settle_s) <= 0.0:
        return _empty(["settling exclusion leaves no measured duration"], foot_names)
    norm = math.hypot(float(direction[0]), float(direction[1]))
    return _measure(
        arrays,
        foot_names=foot_names,
        body_weight_n=float(body_weight_n),
        leg=float(leg_length_m),
        settle_s=float(settle_s),
        direction=(float(direction[0]) / norm, float(direction[1]) / norm),
        protocol=protocol,
    )


class _Pair:
    """Continuous, local and event relative-phase statistics of one ordered limb pair."""

    def __init__(
        self,
        mask: NDArray[np.bool_],
        mean: float | None,
        locking: float,
        coverage: float,
        local_cos: NDArray[np.float64] | None,
        local_sin: NDArray[np.float64] | None,
        local_norm: NDArray[np.float64] | None,
        locked: NDArray[np.bool_],
        events: list[float],
        overlap: float | None,
    ) -> None:
        self.mask = mask
        self.mean = mean
        self.locking = locking
        self.coverage = coverage
        self.local_cos = local_cos
        self.local_sin = local_sin
        self.local_norm = local_norm
        self.locked = locked
        self.events = events
        self.event_mean, self.event_locking = circular_mean(events)
        self.overlap = overlap

    def local_within(self, targets: Sequence[Target], widening: float) -> NDArray[np.bool_]:
        """Samples whose local mean lies within ``targets`` widened by ``widening`` cycles.

        cdist(local mean, c) <= w  <=>  <v, u_c> >= |v| cos(2 pi w), with v the
        local resultant: exact elementwise arithmetic, no inverse tangent.
        """
        inside = np.zeros(len(self.locked), dtype=bool)
        if self.local_cos is None or self.local_sin is None or self.local_norm is None:
            return inside
        for centre, tolerance in targets:
            projection = self.local_cos * math.cos(2.0 * math.pi * centre) + self.local_sin * math.sin(
                2.0 * math.pi * centre
            )
            inside |= projection >= self.local_norm * math.cos(2.0 * math.pi * (tolerance + widening))
        return inside & self.locked


def _measure(
    arrays: dict[str, NDArray[np.float64]],
    *,
    foot_names: tuple[str, ...],
    body_weight_n: float,
    leg: float,
    settle_s: float,
    direction: tuple[float, float],
    protocol: GaitProtocol,
) -> dict[str, Any]:
    time = arrays["time_s"]
    count = len(time)
    feet = len(foot_names)
    canon = BIPED_FEET if feet == 2 else QUADRUPED_FEET
    order = [foot_names.index(name) for name in canon]
    index = {name: position for position, name in enumerate(canon)}
    force = arrays["floor_force_n"][:, order]
    position = arrays["foot_position_m"][:, order]
    clearance = arrays["foot_clearance_m"][:, order]
    slip = arrays["slip_speed_mps"][:, order]
    body = arrays["body_floor_force_n"]
    foot_foot = arrays["foot_foot_force_n"]
    root = arrays["root_position_m"]
    dx_dir, dy_dir = direction

    # sample-and-hold weights restricted to the analysis window [start, end]
    start, end = float(time[0]) + settle_s, float(time[-1])
    span = end - start
    following = np.concatenate((time[1:], time[-1:]))
    held = following - time
    weights = np.clip(np.minimum(following, end) - np.maximum(time, start), 0.0, None)
    in_window = weights > 0.0

    threshold = protocol.contact_force_bw_per_limb * body_weight_n / feet
    limbs = [
        segment_limb(
            time,
            force[:, i],
            clearance[:, i],
            position[:, i, :2],
            threshold_n=threshold,
            leg_length_m=leg,
            protocol=protocol,
        )
        for i in range(feet)
    ]
    stance = np.stack([limb.stance for limb in limbs], axis=1)
    # Footprints: the load-weighted centre (force x time) of every stance, in
    # position and time. A light toe touch ahead of the loaded foot, or a
    # foot sliding through stance, does not move it.
    footprints = [_footprints(time, held, force[:, i], position[:, i, :2], limbs[i]) for i in range(feet)]

    # ---- progress ---------------------------------------------------------
    first = max(int(np.searchsorted(time, start, side="right")) - 1, 0)
    fraction = (start - time[first]) / max(following[first] - time[first], 1e-12) if first + 1 < count else 0.0
    origin = root[first, :2] + (root[min(first + 1, count - 1), :2] - root[first, :2]) * fraction
    displacement = root[-1, :2] - origin
    speed = float(displacement[0] * dx_dir + displacement[1] * dy_dir) / span

    # ---- occupancy and load -------------------------------------------------
    foot_impulse = [_fsum(weights * force[:, i]) for i in range(feet)]
    total_foot = math.fsum(foot_impulse)
    body_impulse = _fsum(weights * body)
    load_share = [value / total_foot if total_foot > 0.0 else 0.0 for value in foot_impulse]
    duty = [_fsum(weights * stance[:, i]) / span for i in range(feet)]
    body_share = body_impulse / (body_impulse + total_foot) if body_impulse > 0.0 else 0.0
    foot_foot_fraction = _fsum(weights * (foot_foot > protocol.foot_foot_force_bw * body_weight_n)) / span
    support = np.sum(stance, axis=1)
    airborne = support == 0
    flight = _fsum(weights * airborne) / span
    airborne_time = _cumulative(weights * airborne)
    max_flight = max((float(airborne_time[e] - airborne_time[s]) for s, e in boolean_runs(airborne)), default=0.0)
    windowed_height = root[in_window, 2]
    trunk_height = float(np.median(windowed_height)) / leg if len(windowed_height) else math.nan

    # ---- strides and continuous phase -----------------------------------------
    touchdown_times = [time[np.asarray(limb.touchdowns, dtype=int)] for limb in limbs]
    median_period: list[float] = []
    stride_normal: list[NDArray[np.bool_]] = []
    phase: list[NDArray[np.float64]] = []
    defined: list[NDArray[np.bool_]] = []
    complete_cycles: list[int] = []
    for i in range(feet):
        events = touchdown_times[i]
        periods = np.diff(events)
        overlapping = (events[1:] > start) & (events[:-1] < end) if len(events) >= 2 else np.zeros(0, dtype=bool)
        median_period.append(_median(periods[overlapping]) if np.any(overlapping) else math.nan)
        # Pause rule against the neighbouring strides of the same limb (a stop
        # between normal strides), not the episode median (a cadence change).
        references = stride_references(periods, protocol.pause_neighbour_strides)
        normal = stride_is_normal(periods, references, protocol.pause_factor)
        stride_normal.append(normal)
        values, ok = limb_phase(time, events, references, protocol.pause_factor)
        phase.append(values)
        defined.append(ok & in_window)
        inside = (events[:-1] >= start) & (events[1:] <= end) if len(events) >= 2 else np.zeros(0, dtype=bool)
        complete_cycles.append(int(np.count_nonzero(inside & normal)))
    finite_periods = [value for value in median_period if math.isfinite(value)]
    pooled_period = _median(finite_periods) if finite_periods else math.nan
    phase_coverage = [_fsum(weights * defined[i]) / span for i in range(feet)]
    unit = [_unit_circle(phase[i]) for i in range(feet)]

    # ---- local travel frame and trunk frame ------------------------------------------
    # Stride length and skid travel are measured along the trunk's own travel
    # heading, not the declared task direction, so a heading drift or a rigid
    # rotation of the episode leaves them unchanged. Step length is measured
    # along the trunk's body axis (root quaternion), so a crabbing trunk,
    # whose stance width projects onto its travel heading, cannot fake it.
    min_travel = protocol.heading_min_travel_over_leg * leg
    window_heading = _unit_or(float(displacement[0]), float(displacement[1]), min_travel, (dx_dir, dy_dir))
    heading_x, heading_y = _travel_heading(
        time,
        root[:, :2],
        protocol.heading_window_strides * pooled_period if math.isfinite(pooled_period) else math.nan,
        min_travel,
        window_heading,
    )
    axis_x, axis_y = _trunk_axis(
        time,
        held,
        arrays["root_quat_wxyz"],
        protocol.trunk_axis_window_strides * pooled_period if math.isfinite(pooled_period) else math.nan,
        heading_x,
        heading_y,
    )

    # ---- swings, slip, skid, stride length ---------------------------------------
    valid_fraction: list[float] = []
    clearance_median: list[float] = []
    skid: list[float] = []
    slip_median: list[float] = []
    slip_p90: list[float] = []
    swing_ground: list[float] = []
    swing_slip: list[float] = []
    glide: list[float] = []
    light_stance: list[float] = []
    foot_stride: list[float | None] = []
    ground_height = protocol.swing_ground_clearance_over_leg * leg
    light_load = protocol.light_load_bw_per_limb * body_weight_n / feet
    in_place = np.zeros(count, dtype=bool)
    for i, limb in enumerate(limbs):
        selected = [k for k, td in enumerate(limb.touchdowns) if start <= time[td] <= end]
        valid_fraction.append(sum(limb.swing_valid[k] for k in selected) / len(selected) if selected else 0.0)
        clearance_median.append(_median([limb.swing_clearance_m[k] for k in selected]) / leg if selected else 0.0)
        slipped = _cumulative(held * slip[:, i])
        distances: list[float] = []
        travels: list[float] = []
        durations: list[float] = []
        gliding: list[float] = []
        for s, e in limb.stances:
            if s > 0 and start <= time[s] <= end:
                distances.append(float(slipped[e] - slipped[s]))
                last = min(e, count - 1)
                moved = root[last, :2] - root[s, :2]
                middle = (s + last) // 2
                travels.append(abs(float(moved[0] * heading_x[middle] + moved[1] * heading_y[middle])))
                elapsed = float(time[last] - time[s])
                if e - s >= _GLIDE_MIN_SAMPLES and elapsed > 0.0:
                    # Glide: the foot's typical (median) slip speed over the
                    # stance exceeds ``glide_skid_ratio`` of the trunk's speed
                    # during it. A touchdown skid is a brief spike that leaves
                    # the median alone; a skating foot slides all stance long.
                    durations.append(elapsed)
                    trunk_speed = max(travels[-1], _SKID_MIN_TRAVEL_OVER_LEG * leg) / elapsed
                    if float(np.median(slip[s:e, i])) > protocol.glide_skid_ratio * trunk_speed:
                        gliding.append(elapsed)
        skid.append(
            math.fsum(distances) / max(math.fsum(travels), _SKID_MIN_TRAVEL_OVER_LEG * leg) if distances else 0.0
        )
        # Share of stance time in gliding stances: skating bouts concentrate
        # sliding in whole stances, which a whole-window ratio averages away.
        glide.append(math.fsum(gliding) / math.fsum(durations) if math.fsum(durations) > 0.0 else 0.0)
        # Light stance: share of stance time (merged unloads included) with the
        # foot carrying less than ``light_load_bw_per_limb``: a hovering
        # retouch or a phantom contact that times a stance without bearing weight.
        stance_weights = weights * stance[:, i]
        stance_time = _fsum(stance_weights)
        light_stance.append(
            _fsum(stance_weights * (force[:, i] < light_load)) / stance_time if stance_time > 0.0 else 0.0
        )
        slip_median.append(_median(distances) / leg if distances else 0.0)
        slip_p90.append(float(np.percentile(distances, 90)) / leg if distances else 0.0)
        # Swing-phase floor contact: time of the swings (between two stances)
        # with any floor force or the foot below the ground band, and the
        # distance slid on the floor during swing over the swing travel.
        between = np.zeros(count, dtype=bool)
        for (_, lift), (land, _) in zip(limb.stances[:-1], limb.stances[1:], strict=True):
            between[lift:land] = True
        swing_weights = weights * between
        swing_time = _fsum(swing_weights)
        grounded = (force[:, i] > 0.0) | (clearance[:, i] < ground_height)
        swing_ground.append(_fsum(swing_weights * grounded) / swing_time if swing_time > 0.0 else 0.0)
        slid = _fsum(swing_weights * slip[:, i])
        swing_travel = math.fsum(limb.swing_travel_m[k] for k in selected)
        swing_slip.append(slid / swing_travel if swing_travel > 0.0 else (1.0 if slid > 0.0 else 0.0))
        # Stride length: footprint-to-footprint advance of the same foot along
        # the local heading, over complete (non-pause) strides in the window.
        # Stances that begin after the first sample are the touchdown stances.
        offset = 1 if limb.stances and limb.stances[0][0] == 0 else 0
        # In-place strides: consecutive footprints of this foot that advance
        # less than ``in_place_stride_over_leg`` along the local heading
        # (marking time, tapping or freezing in place) are off-gait time.
        for before, after in zip(footprints[i][:-1], footprints[i][1:], strict=True):
            if after[2] < start or before[2] > end:
                continue
            middle = min(int(np.searchsorted(time, 0.5 * (before[2] + after[2]))), count - 1)
            advance = (after[0] - before[0]) * heading_x[middle] + (after[1] - before[1]) * heading_y[middle]
            if advance < protocol.in_place_stride_over_leg * leg:
                low = int(np.searchsorted(time, before[2], side="left"))
                high = int(np.searchsorted(time, after[2], side="right"))
                in_place[low:high] = True
        advances: list[float] = []
        touchdowns = limb.touchdowns
        for k, (s0, s1) in enumerate(zip(touchdowns[:-1], touchdowns[1:], strict=True)):
            if time[s0] < start or time[s1] > end or not stride_normal[i][k]:
                continue
            middle = (s0 + s1) // 2
            before, after = footprints[i][offset + k], footprints[i][offset + k + 1]
            dx, dy = after[0] - before[0], after[1] - before[1]
            advances.append((dx * heading_x[middle] + dy * heading_y[middle]) / leg)
        foot_stride.append(_median(advances) if advances else None)

    # ---- step-through: each footprint ahead of the other foot's last one ----------
    # Step length: a footprint's advance past the contralateral foot's latest
    # footprint (centred no later than this one), along the trunk axis at the
    # step's mid-time. A step-to gait
    # (the trailing foot lands level with or behind the leader) has a short or
    # negative step on one side; the gate judges each foot's median step and
    # the share of strides in which both feet step through.
    steps: list[list[float]] = [[] for _ in range(feet)]
    step_spans: list[list[tuple[float, float]]] = [[] for _ in range(feet)]
    crab: list[float] = []
    for first_name, second_name in CONTRALATERAL_PAIRS[feet]:
        a, b = index[first_name], index[second_name]
        for own, other in ((a, b), (b, a)):
            other_times = np.asarray([footprint[2] for footprint in footprints[other]], dtype=float)
            for x_own, y_own, t_own in footprints[own]:
                if not start <= t_own <= end:
                    continue
                previous = int(np.searchsorted(other_times, t_own, side="right")) - 1
                if previous < 0:
                    continue
                x_other, y_other, t_other = footprints[other][previous]
                middle = min(int(np.searchsorted(time, 0.5 * (t_own + t_other))), count - 1)
                ux, uy = float(axis_x[middle]), float(axis_y[middle])
                step = ((x_own - x_other) * ux + (y_own - y_other) * uy) / leg
                steps[own].append(step)
                step_spans[own].append((t_other, t_own))
                crab.append(
                    math.degrees(
                        math.atan2(
                            float(heading_x[middle]) * uy - float(heading_y[middle]) * ux,
                            float(heading_x[middle]) * ux + float(heading_y[middle]) * uy,
                        )
                    )
                )
    # Step-to bouts (diagnostic, never a verdict input): at least
    # ``step_to_bout_steps`` consecutive steps of one foot that do not step
    # through. A genuine walker crabbing in its trunk frame (compsognathus
    # 0921) has such bouts of its weaker foot, so they are not off-gait time.
    not_through = np.zeros(count, dtype=bool)
    for i in range(feet):
        failing = np.asarray([value < protocol.step_through_min_over_leg for value in steps[i]], dtype=bool)
        for s, e in boolean_runs(failing):
            if e - s < protocol.step_to_bout_steps:
                continue
            for t_other, t_own in step_spans[i][s:e]:
                low = int(np.searchsorted(time, t_other, side="left"))
                high = int(np.searchsorted(time, t_own, side="right"))
                not_through[low:high] = True
    not_through &= in_window
    step_median = [_median(values) if values else None for values in steps]
    step_p25 = [float(np.percentile(values, 25)) if values else None for values in steps]
    step_through = [
        sum(value >= protocol.step_through_min_over_leg for value in values) / len(values) if values else 0.0
        for values in steps
    ]
    # Step-through strides: consecutive steps of the two feet of a pair (one
    # stride, a step of each foot) in which both feet step through. A step-to
    # gait fails one step of every stride whichever foot leads, so switching
    # the leader every few strides cannot hide it behind each foot's median.
    stride_through: dict[tuple[int, int], float | None] = {}
    for first_name, second_name in CONTRALATERAL_PAIRS[feet]:
        a, b = index[first_name], index[second_name]
        ordered = sorted(
            [(times[1], a, value) for times, value in zip(step_spans[a], steps[a], strict=True)]
            + [(times[1], b, value) for times, value in zip(step_spans[b], steps[b], strict=True)]
        )
        through = [
            earlier[2] >= protocol.step_through_min_over_leg and later[2] >= protocol.step_through_min_over_leg
            for earlier, later in zip(ordered[:-1], ordered[1:], strict=True)
            if earlier[1] != later[1]
        ]
        stride_through[(a, b)] = sum(through) / len(through) if through else None

    # ---- pairwise relative phase ------------------------------------------------
    local_width = protocol.local_window_strides * pooled_period if math.isfinite(pooled_period) else math.nan
    low_index = high_index = None
    if math.isfinite(local_width) and local_width > 0.0:
        low_index = np.searchsorted(time, time - local_width / 2.0, side="left")
        high_index = np.searchsorted(time, time + local_width / 2.0, side="right")
    cache: dict[tuple[int, int], _Pair] = {}

    def pair(i: int, j: int) -> _Pair:
        if (i, j) in cache:
            return cache[(i, j)]
        mask = defined[i] & defined[j]
        (cos_i, sin_i), (cos_j, sin_j) = unit[i], unit[j]
        rel_cos = cos_i * cos_j + sin_i * sin_j
        rel_sin = sin_i * cos_j - cos_i * sin_j
        masked = weights * mask
        total = _fsum(masked)
        if total > 0.0:
            mean, locking = _circular_stats(_fsum(masked * rel_cos) / total, _fsum(masked * rel_sin) / total)
        else:
            mean, locking = None, 0.0
        local_cos = local_sin = local_norm = None
        locked = np.zeros(count, dtype=bool)
        if low_index is not None and high_index is not None:
            cw, cc, cs = _cumulative(masked), _cumulative(masked * rel_cos), _cumulative(masked * rel_sin)
            summed = cw[high_index] - cw[low_index]
            local_cos = cc[high_index] - cc[low_index]
            local_sin = cs[high_index] - cs[low_index]
            local_norm = np.sqrt(local_cos * local_cos + local_sin * local_sin)
            good = summed > 0.5 * local_width
            resultant = np.where(good, local_norm / np.where(good, summed, 1.0), 0.0)
            locked = mask & good & (local_norm > 0.0) & (resultant >= protocol.local_min_locking)
        # Hildebrand footfall phases pooled in both directions: phi_i at each
        # touchdown of j (lag of j behind i) and 1 - phi_j at each touchdown of i.
        events: list[float] = []
        for source, target, sign in ((i, j, 1.0), (j, i, -1.0)):
            for sample in limbs[target].touchdowns:
                if start <= time[sample] <= end and defined[source][sample]:
                    events.append(float((sign * phase[source][sample]) % 1.0))
        both = _fsum(weights * (stance[:, i] & stance[:, j])) / span
        anti, sync = max(0.0, duty[i] + duty[j] - 1.0), min(duty[i], duty[j])
        overlap = min(1.0, max(0.0, (both - anti) / (sync - anti))) if sync - anti >= _OVERLAP_MIN_SPAN else None
        result = _Pair(mask, mean, locking, total / span, local_cos, local_sin, local_norm, locked, events, overlap)
        cache[(i, j)] = result
        return result

    pair_phase: dict[str, Any] = {}
    for a in range(feet):
        for b in range(a + 1, feet):
            stats = pair(a, b)
            pair_phase[f"{canon[a]}>{canon[b]}"] = {
                "mean_phase": _r(stats.mean),
                "locking": _r(stats.locking),
                "coverage": _r(stats.coverage),
                "mean_phase_event": _r(stats.event_mean),
                "locking_event": _r(stats.event_locking),
                "n_events": len(stats.events),
                "overlap_index": _r(stats.overlap),
            }

    # ---- contralateral symmetry --------------------------------------------------
    contralateral: dict[str, Any] = {}
    for first_name, second_name in CONTRALATERAL_PAIRS[feet]:
        a, b = index[first_name], index[second_name]
        heavier = max(foot_impulse[a], foot_impulse[b])
        longer = max(duty[a], duty[b])
        stats = pair(a, b)
        anchor_offset: dict[str, float] = {}
        for anchor in ("touchdown", "midstance"):
            if anchor == "touchdown":
                mean, event_mean = stats.mean, stats.event_mean
            else:
                mean, event_mean = _midstance_means(time, limbs, a, b, start, end, weights, protocol)
            anchor_offset[anchor] = max(
                _cdist(mean, 0.5) if mean is not None else 0.5,
                _cdist(event_mean, 0.5) if event_mean is not None else 0.5,
            )
        merged = _alternation_index(touchdown_times[a], touchdown_times[b], start, end, pooled_period)
        contralateral[f"{first_name}|{second_name}"] = {
            "load_ratio": _r(min(foot_impulse[a], foot_impulse[b]) / heavier if heavier > 0.0 else 0.0),
            "duty_ratio": _r(min(duty[a], duty[b]) / longer if longer > 0.0 else 0.0),
            "alternation_phase_offset_touchdown": _r(anchor_offset["touchdown"]),
            "alternation_phase_offset_midstance": _r(anchor_offset["midstance"]),
            "alternation_index": _r(merged),
            "pci_percent": _r(_phase_coordination_index(stats.events)),
            "step_through_stride_fraction": _r(stride_through[(a, b)]),
        }

    # ---- per-foot records -------------------------------------------------------
    per_foot: dict[str, Any] = {}
    for i, name in enumerate(canon):
        per_foot[name] = {
            "duty_factor": _r(duty[i]),
            "load_share": _r(load_share[i]),
            "relative_load_share": _r(load_share[i] * feet),
            "phase_coverage": _r(phase_coverage[i]),
            "touchdowns": int(sum(start <= time[td] <= end for td in limbs[i].touchdowns)),
            "complete_cycles": complete_cycles[i],
            "median_stride_s": _r(median_period[i]),
            "valid_swing_fraction": _r(valid_fraction[i]),
            "median_swing_clearance_over_leg": _r(clearance_median[i]),
            "skid_fraction": _r(skid[i]),
            "slip_over_leg_median": _r(slip_median[i]),
            "slip_over_leg_p90": _r(slip_p90[i]),
            "steps": len(steps[i]),
            "step_length_over_leg_median": _r(step_median[i]),
            "step_length_over_leg_p25": _r(step_p25[i]),
            "step_through_fraction": _r(step_through[i]),
            "stride_length_over_leg_median": _r(foot_stride[i]),
            "swing_ground_fraction": _r(swing_ground[i]),
            "swing_slip_fraction": _r(swing_slip[i]),
            "glide_stance_fraction": _r(glide[i]),
            "light_stance_fraction": _r(light_stance[i]),
            "merged_unloads": limbs[i].merged_unloads,
        }

    # ---- alternation template and the off-gait budget ----------------------------------
    # Every walk-first profile templates its contralateral pairs at anti-phase,
    # anchored at touchdown or at mid-stance: with unequal duty factors evenly
    # spaced mid-stances put the touchdown lag of j behind i at
    # 0.5 + (d_i - d_j) / 2. Either anchor may be the symmetric one; the duty
    # asymmetry itself is judged by the pair duty ratio.
    bout_width = protocol.off_gait_bout_strides * pooled_period if math.isfinite(pooled_period) else math.inf
    tolerance = protocol.template_alternation_tolerance
    joint = in_window.copy()
    undefined = np.zeros(count, dtype=bool)
    unlocked = np.zeros(count, dtype=bool)
    off_template = np.zeros(count, dtype=bool)
    pairs: dict[str, Any] = {}
    locking_values: list[float] = []
    offsets: list[float] = []
    overlaps: list[float] = []
    for first_name, second_name in CONTRALATERAL_PAIRS[feet]:
        i, j = index[first_name], index[second_name]
        stats = pair(i, j)
        targets: tuple[Target, ...] = ((0.5, tolerance), ((0.5 + 0.5 * (duty[i] - duty[j])) % 1.0, tolerance))
        on = stats.local_within(targets, 0.0)
        near = stats.local_within(targets, protocol.off_gait_extra_tolerance)
        joint &= on
        undefined |= in_window & ~stats.mask
        unlocked |= in_window & stats.mask & ~stats.locked
        # Locked well away from anti-phase: a hop, bound or pronk (in phase),
        # a skip, a gallop or a staggered hop, whatever the lag.
        off_template |= in_window & stats.locked & ~near
        locking_values.extend((stats.locking, stats.event_locking))
        estimates = (stats.mean, stats.event_mean)
        # the better anchor, judged by both estimators
        offsets.append(min(max(_cdist(m, c) if m is not None else 0.5 for m in estimates) for c, _ in targets))
        if stats.overlap is not None:
            overlaps.append(stats.overlap)
        event_on = (
            sum(any(_cdist(x, c) <= t for c, t in targets) for x in stats.events) / len(stats.events)
            if stats.events
            else 0.0
        )
        pairs[f"{first_name}>{second_name}"] = {
            "midstance_target_phase": _r(targets[-1][0]),
            "mean_phase": _r(stats.mean),
            "locking": _r(stats.locking),
            "mean_phase_event": _r(stats.event_mean),
            "locking_event": _r(stats.event_locking),
            "on_template_coverage": _r(_fsum(weights * on) / span),
            "event_on_template_fraction": _r(event_on),
            "n_events": len(stats.events),
            "overlap_index": _r(stats.overlap),
        }
    # One off-gait budget: a limb not cycling (undefined phase, e.g. a pause or
    # a stand), a pair locked off template, a sustained bout of uncoordinated
    # stepping (any of the three, unbroken for at least
    # ``off_gait_bout_strides`` pooled strides), the trunk standing, a stride
    # in place, or (four legs) an asymmetrical stride. Shorter unlocked
    # moments (a stumble, a double step) are not off-gait time.
    incoherent = undefined | off_template | unlocked
    incoherent_time = _cumulative(weights * incoherent)
    bouts = np.zeros(count, dtype=bool)
    for s, e in boolean_runs(incoherent):
        if float(incoherent_time[e] - incoherent_time[s]) >= bout_width:
            bouts[s:e] = True
    # Standing: the trunk barely moves (a pause, a freeze, a standing start
    # inside the window), whatever the feet do.
    standing = in_window & _standing(
        time,
        root[:, :2],
        protocol.standing_window_strides * pooled_period if math.isfinite(pooled_period) else math.nan,
        in_window,
        protocol.standing_speed_fraction,
    )
    in_place &= in_window
    # On four legs, strides in which both contralateral pairs land off
    # template at once: an asymmetrical stride (canter, gallop, half-bound),
    # however briefly. One pair off by itself is a stumble.
    asymmetric = np.zeros(count, dtype=bool)
    if feet == 4:
        off_pairs = []
        for first_name, second_name in CONTRALATERAL_PAIRS[feet]:
            off_pairs.append(
                _off_template_strides(
                    time, limbs, stride_normal, index[first_name], index[second_name], duty, start, end, tolerance
                )
            )
        asymmetric = in_window & off_pairs[0] & off_pairs[1]
    off_gait = undefined | off_template | bouts | standing | in_place | asymmetric
    off_gait_time = _cumulative(weights * off_gait)
    longest = max((float(off_gait_time[e] - off_gait_time[s]) for s, e in boolean_runs(off_gait)), default=0.0)
    templates = {
        ALTERNATION_TEMPLATE: {
            "pairs": pairs,
            "phase_locking_min": _r(min(locking_values)),
            "alternation_phase_offset_max": _r(max(offsets)),
            "alternating_overlap_index_max": _r(max(overlaps)) if overlaps else None,
            "template_coverage": _r(_fsum(weights * joint) / span),
            "off_gait_fraction": _r(_fsum(weights * off_gait) / span),
            "off_gait_strides": _r(_fsum(weights * off_gait) / pooled_period)
            if math.isfinite(pooled_period) and pooled_period > 0.0
            else None,
            "undefined_phase_fraction": _r(_fsum(weights * undefined) / span),
            "off_template_locked_fraction": _r(_fsum(weights * off_template) / span),
            "uncoordinated_bout_fraction": _r(_fsum(weights * (bouts & ~undefined & ~off_template)) / span),
            "standing_fraction": _r(_fsum(weights * standing) / span),
            "in_place_stride_fraction": _r(_fsum(weights * in_place) / span),
            "asymmetric_stride_fraction": _r(_fsum(weights * asymmetric) / span) if feet == 4 else None,
            "step_to_bout_fraction": _r(_fsum(weights * not_through) / span),
            "longest_off_gait_s": _r(longest),
        }
    }

    # ---- girdle participation (quadrupeds) -------------------------------------------
    # Share of the foot impulse carried by the lighter girdle (fore pair or
    # hind pair) and the ratio of the girdles' mean duty factors: a rearing,
    # wheelbarrow or token-limb gait leaves one girdle nearly unloaded.
    fore_share = girdle_share = girdle_duty = girdle_unloaded = None
    if feet == 4:
        fore_share = load_share[index["fr"]] + load_share[index["fl"]]
        hind_share = load_share[index["rr"]] + load_share[index["rl"]]
        girdle_share = min(fore_share, hind_share) if total_foot > 0.0 else 0.0
        fore_duty = 0.5 * (duty[index["fr"]] + duty[index["fl"]])
        hind_duty = 0.5 * (duty[index["rr"]] + duty[index["rl"]])
        longer_girdle = max(fore_duty, hind_duty)
        girdle_duty = min(fore_duty, hind_duty) / longer_girdle if longer_girdle > 0.0 else 0.0
        # Time-local participation: the share of the window in which the lighter
        # girdle carries less than ``girdle_local_min_share`` of the foot impulse
        # over a centred one-stride window. Whole-window shares let time spent
        # quadrupedal pay for time spent rearing or wheelbarrowing.
        if low_index is not None and high_index is not None:
            girdle_window = []
            for names in (("fr", "fl"), ("rr", "rl")):
                carried = _cumulative(held * (force[:, index[names[0]]] + force[:, index[names[1]]]))
                girdle_window.append(carried[high_index] - carried[low_index])
            fore_window, hind_window = girdle_window
            both = fore_window + hind_window
            lighter = np.where(both > 0.0, np.minimum(fore_window, hind_window) / np.where(both > 0.0, both, 1.0), 0.0)
            girdle_unloaded = _fsum(weights * (in_window & (lighter < protocol.girdle_local_min_share))) / span

    # ---- diagnostics ---------------------------------------------------------------
    labels = stride_labels(
        time, limbs, root[:, 2], window=(start, end), median_periods=median_period, protocol=protocol
    )
    froude = max(speed, 0.0) ** 2 / (GRAVITY_MPS2 * leg)
    measured_strides = [value for value in foot_stride if value is not None]
    stride_length = _median(measured_strides) * leg if measured_strides else math.nan
    alexander = 2.3 * froude**0.3 * leg if froude > 0.0 else math.nan
    step_medians = [value for value in step_median if value is not None]
    step_lower = [value for value in step_p25 if value is not None]
    pair_records = list(contralateral.values())

    def aggregate(values: list[float], operation: str) -> float | None:
        return _r(min(values) if operation == "min" else max(values)) if values else None

    stride_fractions = [value for value in stride_through.values() if value is not None]

    return {
        "telemetry_valid": True,
        "telemetry_errors": [],
        "n_limbs": feet,
        "duration_s": _r(span),
        "recorded_span_s": _r(float(time[-1] - time[0])),
        "max_sample_interval_s": _r(float(np.max(np.diff(time)))),
        "mean_speed_mps": _r(speed),
        "progress_over_leg": _r(speed * span / leg),
        "froude": _r(froude),
        "stride_period_s": _r(pooled_period),
        "stride_frequency_hz": _r(1.0 / pooled_period) if math.isfinite(pooled_period) and pooled_period > 0 else None,
        "stride_length_over_leg": _r(stride_length / leg),
        "stride_over_alexander_prediction": _r(stride_length / alexander),
        "flight_fraction": _r(flight),
        "max_flight_s": _r(max_flight),
        "hop_flight_fraction": _r(_hop_flight(time, limbs, weights, airborne, pooled_period, protocol, span))
        if feet == 2
        else None,
        "body_support_fraction": _r(body_share),
        "foot_foot_contact_fraction": _r(foot_foot_fraction),
        "weight_support_ratio": _r((total_foot + body_impulse) / (body_weight_n * span)),
        "support_count_fraction": {str(n): _r(_fsum(weights * (support == n)) / span) for n in range(feet + 1)},
        "trunk_height_over_leg_median": _r(trunk_height),
        "trunk_crab_angle_deg_median": _r(_median(crab)) if crab else None,
        "limb_phase_coverage_min": aggregate(phase_coverage, "min"),
        "limb_duty_min": aggregate(duty, "min"),
        "relative_limb_load_share_min": aggregate([share * feet for share in load_share], "min"),
        "complete_cycles_min": min(complete_cycles),
        "valid_swing_fraction_min": aggregate(valid_fraction, "min"),
        "median_swing_clearance_over_leg_min": aggregate(clearance_median, "min"),
        "skid_fraction_max": aggregate(skid, "max"),
        "swing_ground_fraction_max": aggregate(swing_ground, "max"),
        "swing_slip_fraction_max": aggregate(swing_slip, "max"),
        "stride_length_over_leg_min": aggregate(measured_strides, "min") if len(measured_strides) == feet else None,
        "step_length_over_leg_min": aggregate(step_medians, "min") if len(step_medians) == feet else None,
        "step_length_p25_over_leg_min": aggregate(step_lower, "min") if len(step_lower) == feet else None,
        "step_through_fraction_min": aggregate(step_through, "min"),
        "step_through_stride_fraction_min": aggregate(stride_fractions, "min")
        if len(stride_fractions) == len(stride_through)
        else None,
        "fore_load_share": _r(fore_share),
        "girdle_load_share_min": _r(girdle_share),
        "girdle_duty_ratio": _r(girdle_duty),
        "girdle_unloaded_fraction": _r(girdle_unloaded),
        "glide_stance_fraction_max": aggregate(glide, "max"),
        "light_stance_fraction_max": aggregate(light_stance, "max"),
        "pair_load_ratio_min": aggregate([record["load_ratio"] for record in pair_records], "min"),
        "pair_duty_ratio_min": aggregate([record["duty_ratio"] for record in pair_records], "min"),
        "per_foot": {name: per_foot[name] for name in foot_names},
        "contralateral": contralateral,
        "pair_phase": pair_phase,
        "templates": templates,
        "gait_label": labels["gait_label"],
        "label_distribution": {
            key: _r(value, 4) for key, value in labels["label_distribution"].items() if value > 5e-4
        },
        "family_distribution": {
            key: _r(value, 4) for key, value in labels["family_distribution"].items() if value > 5e-4
        },
        "limb_phase_mean": _r(labels["limb_phase_mean"]),
        "limb_phase_concentration": _r(labels["limb_phase_concentration"]),
        "hind_duty_median": _r(labels["hind_duty_median"]),
        "non_stepping_limbs": [canon[i] for i in range(feet) if phase_coverage[i] < 0.5],
    }


def _standing(
    time: NDArray[np.float64],
    root_xy: NDArray[np.float64],
    width: float,
    in_window: NDArray[np.bool_],
    fraction: float,
) -> NDArray[np.bool_]:
    """Samples at which the trunk stands: its horizontal speed over a centred ``width`` window (at
    least one sample interval either side) is below ``fraction`` of the upper quartile of that speed
    over the analysis window.

    The upper quartile stays on the walking speed even when a quarter of the
    window is spent standing; a genuine slow segment (a creep at a tenth of
    the cruise speed, a slow start) is above a twentieth of it.
    """
    count = len(time)
    if not (math.isfinite(width) and width > 0.0) or not np.any(in_window):
        return np.zeros(count, dtype=bool)
    # at least one sample interval on either side, so coarse sampling never
    # leaves a window without elapsed time
    samples = np.arange(count)
    low = np.clip(np.minimum(np.searchsorted(time, time - width / 2.0, side="left"), samples - 1), 0, count - 1)
    high = np.clip(np.maximum(np.searchsorted(time, time + width / 2.0, side="right") - 1, samples + 1), 0, count - 1)
    elapsed = time[high] - time[low]
    dx = root_xy[high, 0] - root_xy[low, 0]
    dy = root_xy[high, 1] - root_xy[low, 1]
    speed = np.where(elapsed > 0.0, np.sqrt(dx * dx + dy * dy) / np.where(elapsed > 0.0, elapsed, 1.0), 0.0)
    reference = float(np.percentile(speed[in_window], 75.0))
    if not reference > 0.0:
        return in_window.copy()
    return speed < fraction * reference


def _off_template_strides(
    time: NDArray[np.float64],
    limbs: Sequence[LimbContacts],
    stride_normal: Sequence[NDArray[np.bool_]],
    a: int,
    b: int,
    duty: Sequence[float],
    start: float,
    end: float,
    tolerance: float,
) -> NDArray[np.bool_]:
    """Samples inside a stride whose contralateral footfall lands beyond ``tolerance`` of both anchors.

    For each normal stride of either limb that holds exactly one touchdown of
    the other, the footfall phase is compared with anti-phase anchored at
    touchdown (0.5) and at mid-stance (0.5 + (d_i - d_j) / 2); the stride is
    marked when it is farther than ``tolerance`` from both.
    """
    count = len(time)
    marked = np.zeros(count, dtype=bool)
    for i, j in ((a, b), (b, a)):
        own = time[np.asarray(limbs[i].touchdowns, dtype=int)]
        other = time[np.asarray(limbs[j].touchdowns, dtype=int)]
        shifted = (0.5 + 0.5 * (duty[i] - duty[j])) % 1.0
        for k in range(len(own) - 1):
            t0, t1 = float(own[k]), float(own[k + 1])
            if t1 < start or t0 > end or not stride_normal[i][k]:
                continue
            inside = other[(other >= t0) & (other < t1)]
            if len(inside) != 1:
                continue
            phase = (float(inside[0]) - t0) / (t1 - t0)
            if min(_cdist(phase, 0.5), _cdist(phase, shifted)) > tolerance:
                marked[int(np.searchsorted(time, t0, side="left")) : int(np.searchsorted(time, t1, side="left"))] = True
    return marked


def _footprints(
    time: NDArray[np.float64],
    held: NDArray[np.float64],
    force_n: NDArray[np.float64],
    foot_xy: NDArray[np.float64],
    limb: LimbContacts,
) -> list[tuple[float, float, float]]:
    """Load-weighted centre ``(x, y, t)`` of every stance (force x held time).

    A stance that carries no load over its samples (a lone final sample) falls
    back to its unweighted centre.
    """
    out: list[tuple[float, float, float]] = []
    for s, e in limb.stances:
        load = held[s:e] * force_n[s:e]
        total = _fsum(load)
        if total > 0.0:
            out.append(
                (
                    _fsum(load * foot_xy[s:e, 0]) / total,
                    _fsum(load * foot_xy[s:e, 1]) / total,
                    _fsum(load * time[s:e]) / total,
                )
            )
        else:
            out.append(
                (
                    _fsum(foot_xy[s:e, 0]) / (e - s),
                    _fsum(foot_xy[s:e, 1]) / (e - s),
                    0.5 * float(time[s] + time[e - 1]),
                )
            )
    return out


def _trunk_axis(
    time: NDArray[np.float64],
    held: NDArray[np.float64],
    quaternion: NDArray[np.float64],
    width: float,
    fallback_x: NDArray[np.float64],
    fallback_y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit trunk axis per sample: the root body's x axis projected on the floor.

    The projection of the body x axis of the (normalised) root quaternion
    ``(w, x, y, z)`` is ``(1 - 2 (y^2 + z^2), 2 (x y + w z))``; it is summed
    over a centred ``width`` window (time-weighted) to remove the
    stride-periodic yaw wobble. A trunk standing vertically for the whole
    window has no horizontal axis there and takes the travel heading.
    Exact elementwise arithmetic and sequential sums only.
    """
    count = len(time)
    w, x, y, z = (quaternion[:, k] for k in range(4))
    norm = w * w + x * x + y * y + z * z
    forward_x = 1.0 - 2.0 * (y * y + z * z) / norm
    forward_y = 2.0 * (x * y + w * z) / norm
    if math.isfinite(width) and width > 0.0:
        low = np.searchsorted(time, time - width / 2.0, side="left")
        high = np.searchsorted(time, time + width / 2.0, side="right")
        cw = _cumulative(held)
        cx = _cumulative(held * forward_x)
        cy = _cumulative(held * forward_y)
        total = cw[high] - cw[low]
        sum_x = cx[high] - cx[low]
        sum_y = cy[high] - cy[low]
    else:
        total = np.ones(count)
        sum_x, sum_y = forward_x, forward_y
    length = np.sqrt(sum_x * sum_x + sum_y * sum_y)
    horizontal = (length > 1e-6 * total) & (length > 0.0)
    safe = np.where(horizontal, length, 1.0)
    return (
        np.where(horizontal, sum_x / safe, fallback_x),
        np.where(horizontal, sum_y / safe, fallback_y),
    )


def _unit_or(dx: float, dy: float, min_length: float, fallback: tuple[float, float]) -> tuple[float, float]:
    """``(dx, dy)`` normalised, or ``fallback`` when it is shorter than ``min_length``."""
    length = math.hypot(dx, dy)
    if length < min_length or length <= 0.0:
        return fallback
    return dx / length, dy / length


def _travel_heading(
    time: NDArray[np.float64],
    root_xy: NDArray[np.float64],
    width: float,
    min_travel: float,
    fallback: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit travel heading per sample: trunk displacement over a centred ``width`` window.

    Samples whose window displacement is shorter than ``min_travel`` (a stop,
    a stand, the truncated window at a trace edge of a slow walker) take the
    ``fallback`` heading. Exact elementwise arithmetic only.
    """
    count = len(time)
    heading_x = np.full(count, fallback[0])
    heading_y = np.full(count, fallback[1])
    if not (math.isfinite(width) and width > 0.0):
        return heading_x, heading_y
    low = np.clip(np.searchsorted(time, time - width / 2.0, side="left"), 0, count - 1)
    high = np.clip(np.searchsorted(time, time + width / 2.0, side="right") - 1, 0, count - 1)
    dx = root_xy[high, 0] - root_xy[low, 0]
    dy = root_xy[high, 1] - root_xy[low, 1]
    length = np.sqrt(dx * dx + dy * dy)
    moving = (length >= min_travel) & (length > 0.0)
    safe = np.where(moving, length, 1.0)
    heading_x[moving] = (dx / safe)[moving]
    heading_y[moving] = (dy / safe)[moving]
    return heading_x, heading_y


def _midstance_means(
    time: NDArray[np.float64],
    limbs: Sequence[LimbContacts],
    a: int,
    b: int,
    start: float,
    end: float,
    weights: NDArray[np.float64],
    protocol: GaitProtocol,
) -> tuple[float | None, float | None]:
    """Continuous and event pair phase anchored at mid-stance (diagnostic).

    Anchoring at mid-stance removes the shift that unequal stance durations
    impose on the touchdown-anchored phase (left-heavy walkers).
    """
    count = len(time)
    anchors = []
    for limb in limbs[a], limbs[b]:
        anchors.append(
            np.asarray([0.5 * (time[s] + time[e]) for s, e in limb.stances if s > 0 and e < count], dtype=float)
        )
    phases = []
    for anchor in anchors:
        references = stride_references(np.diff(anchor), protocol.pause_neighbour_strides)
        value, ok = limb_phase(time, anchor, references, protocol.pause_factor)
        phases.append((value, ok & (weights > 0.0)))
    (phase_a, ok_a), (phase_b, ok_b) = phases
    mask = ok_a & ok_b
    masked = weights * mask
    total = _fsum(masked)
    mean = None
    if total > 0.0:
        cos_a, sin_a = _unit_circle(phase_a)
        cos_b, sin_b = _unit_circle(phase_b)
        mean, _ = _circular_stats(
            _fsum(masked * (cos_a * cos_b + sin_a * sin_b)) / total,
            _fsum(masked * (sin_a * cos_b - cos_a * sin_b)) / total,
        )
    events: list[float] = []
    for (source_phase, source_ok), target_events, sign in (
        ((phase_a, ok_a), anchors[1], 1.0),
        ((phase_b, ok_b), anchors[0], -1.0),
    ):
        for when in target_events.tolist():
            sample = min(int(np.searchsorted(time, when)), count - 1)
            if start <= when <= end and source_ok[sample]:
                events.append(float((sign * source_phase[sample]) % 1.0))
    event_mean, _ = circular_mean(events)
    return mean, event_mean


def _alternation_index(
    first: NDArray[np.float64], second: NDArray[np.float64], start: float, end: float, stride: float
) -> float | None:
    """Fraction of consecutive touchdowns that switch foot (near-simultaneous pairs merged)."""
    merge = _ALTERNATION_MERGE_STRIDES * stride if math.isfinite(stride) else 0.0
    events = sorted(
        [(float(t), 0) for t in first.tolist() if start <= t <= end]
        + [(float(t), 1) for t in second.tolist() if start <= t <= end]
    )
    sequence: list[int] = []
    k = 0
    while k < len(events):
        if k + 1 < len(events) and events[k + 1][1] != events[k][1] and events[k + 1][0] - events[k][0] <= merge:
            sequence.append(2)
            k += 2
        else:
            sequence.append(events[k][1])
            k += 1
    if len(sequence) < 2:
        return None
    switches = sum(1 for x, y in zip(sequence, sequence[1:], strict=False) if x != y and 2 not in (x, y))
    return switches / (len(sequence) - 1)


def _phase_coordination_index(events: list[float]) -> float | None:
    """Plotnik et al. (2007) PCI in percent from the pooled footfall phases."""
    if len(events) < 3:
        return None
    mean = math.fsum(events) / len(events)
    if mean <= 0.0:
        return None
    deviation = math.sqrt(math.fsum((x - mean) ** 2 for x in events) / (len(events) - 1))
    absolute = math.fsum(abs(x - 0.5) for x in events) / len(events)
    return 100.0 * (deviation / mean + absolute / 0.5)


def _hop_flight(
    time: NDArray[np.float64],
    limbs: Sequence[LimbContacts],
    weights: NDArray[np.float64],
    airborne: NDArray[np.bool_],
    stride: float,
    protocol: GaitProtocol,
    span: float,
) -> float:
    """Biped flight not bracketed by one foot lifting off and the other landing."""
    count = len(time)
    window = protocol.hop_flight_window_strides * stride if math.isfinite(stride) else 0.0
    liftoffs = [time[[e for _, e in limb.stances if e < count]] for limb in limbs]
    touchdowns = [time[np.asarray(limb.touchdowns, dtype=int)] for limb in limbs]
    hop = 0.0
    for s, e in boolean_runs(airborne):
        if s == 0 or e >= count:
            continue
        duration = _fsum(weights[s:e])
        if duration <= 0.0:
            continue
        begin, finish = float(time[s]), float(time[e])
        openers = [
            f for f, lifts in enumerate(liftoffs) if np.any((lifts >= begin - window - 1e-9) & (lifts <= begin + 1e-9))
        ]
        closers = [
            f
            for f, lands in enumerate(touchdowns)
            if np.any((lands >= finish - 1e-9) & (lands <= finish + window + 1e-9))
        ]
        if not (len(openers) == 1 and len(closers) == 1 and openers[0] != closers[0]):
            hop += duration
    return hop / span
