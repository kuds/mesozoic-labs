"""Serializable physical gait metrics from immutable physics-sample telemetry.

Gaits are measured as phase-locked patterns of limb oscillators (Collins and
Stewart 1993; Golubitsky et al. 1999). Each limb's continuous phase rises from
0 to 1 between its touchdowns; for every limb pair the relative phase is
summarised by its circular mean and mean resultant length (the pair's
phase-locking index), by a centred one-stride sliding estimate and by
Hildebrand's discrete footfall estimate pooled in both directions. A declared
gait is a template: limb pairs with target relative-phase sets. Template
coverage and persistence measure how much of the analysis window every
templated pair is locally on its target, so a stop, a limb that stops
cycling, or a switch to another gait costs coverage in proportion to its
duration. Physical rails measure limb participation, swing validity, sliding,
non-foot support, foot-on-foot support and flight.

Only recorder keys are read. Every reduction that reaches a stored metric is
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

from environments.shared.gait.events import boolean_runs, limb_phase, segment_limb
from environments.shared.gait.labels import circular_mean, stride_labels
from environments.shared.gait.types import GaitProtocol, LimbContacts

GRAVITY_MPS2 = 9.81
DIGITS = 6
BIPED_FEET = ("r", "l")
QUADRUPED_FEET = ("fr", "fl", "rr", "rl")
PROFILES_BY_FEET: dict[int, tuple[str, ...]] = {
    2: ("biped_alternating",),
    4: ("quadruped_walk", "quadruped_trot", "quadruped_pace"),
}
CONTRALATERAL_PAIRS: dict[int, tuple[tuple[str, str], ...]] = {
    2: (("r", "l"),),
    4: (("fr", "fl"), ("rr", "rl")),
}
_REQUIRED_FIELDS = (
    "time_s",
    "floor_force_n",
    "foot_position_m",
    "foot_clearance_m",
    "slip_speed_mps",
    "body_floor_force_n",
    "foot_foot_force_n",
    "root_position_m",
)
#: Minimum stance-to-trunk travel denominator of the skid fraction, in L.
_SKID_MIN_TRAVEL_OVER_LEG = 0.05
#: Duty-normalised stance overlap is undefined when the duty factors leave
#: less than this between its anti-phase and in-phase extremes.
_OVERLAP_MIN_SPAN = 0.10
#: Alternation-index touchdown merge window, in pooled strides.
_ALTERNATION_MERGE_STRIDES = 0.05

Target = tuple[float, float]
TemplatePair = tuple[str, str, tuple[Target, ...], str]


def gait_templates(protocol: GaitProtocol, feet: int) -> dict[str, tuple[TemplatePair, ...]]:
    """Profile -> templated pairs ``(i, j, targets, kind)``; phase is the lag of j behind i.

    Limb phase is the forefoot's lag behind the ipsilateral hindfoot
    (Hildebrand 1976). The 1/8 partition of the limb-phase circle makes the
    quadruped profiles mutually exclusive: pace within 1/8 of 0, trot within
    1/8 of 1/2, lateral- and diagonal-sequence single-foot walks between.
    """
    alternation = ((0.5, protocol.template_alternation_tolerance),)
    synchrony = ((0.0, protocol.template_synchrony_tolerance),)
    centre = 0.5 * (protocol.template_walk_band_low + protocol.template_walk_band_high)
    half = 0.5 * (protocol.template_walk_band_high - protocol.template_walk_band_low)
    band = ((centre, half), (1.0 - centre, half))
    if feet == 2:
        return {"biped_alternating": (("r", "l", alternation, "alternation"),)}
    contralateral: tuple[TemplatePair, ...] = (
        ("fr", "fl", alternation, "alternation"),
        ("rr", "rl", alternation, "alternation"),
    )
    return {
        "quadruped_walk": contralateral + (("rr", "fr", band, "limb_phase"), ("rl", "fl", band, "limb_phase")),
        "quadruped_trot": contralateral + (("fr", "rl", synchrony, "synchrony"), ("fl", "rr", synchrony, "synchrony")),
        "quadruped_pace": contralateral + (("fr", "rr", synchrony, "synchrony"), ("fl", "rl", synchrony, "synchrony")),
    }


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

    # ---- strides and continuous phase -----------------------------------------
    touchdown_times = [time[np.asarray(limb.touchdowns, dtype=int)] for limb in limbs]
    median_period: list[float] = []
    phase: list[NDArray[np.float64]] = []
    defined: list[NDArray[np.bool_]] = []
    complete_cycles: list[int] = []
    for i in range(feet):
        events = touchdown_times[i]
        periods = np.diff(events)
        overlapping = (events[1:] > start) & (events[:-1] < end) if len(events) >= 2 else np.zeros(0, dtype=bool)
        median = _median(periods[overlapping]) if np.any(overlapping) else math.nan
        median_period.append(median)
        values, ok = limb_phase(time, events, median, protocol.pause_factor)
        phase.append(values)
        defined.append(ok & in_window)
        inside = (events[:-1] >= start) & (events[1:] <= end) if len(events) >= 2 else np.zeros(0, dtype=bool)
        complete_cycles.append(
            int(np.count_nonzero(inside & (periods <= protocol.pause_factor * median))) if math.isfinite(median) else 0
        )
    finite_periods = [value for value in median_period if math.isfinite(value)]
    pooled_period = _median(finite_periods) if finite_periods else math.nan
    phase_coverage = [_fsum(weights * defined[i]) / span for i in range(feet)]
    unit = [_unit_circle(phase[i]) for i in range(feet)]

    # ---- swings, slip, skid ---------------------------------------------------
    valid_fraction: list[float] = []
    clearance_median: list[float] = []
    skid: list[float] = []
    slip_median: list[float] = []
    slip_p90: list[float] = []
    for i, limb in enumerate(limbs):
        selected = [k for k, td in enumerate(limb.touchdowns) if start <= time[td] <= end]
        valid_fraction.append(sum(limb.swing_valid[k] for k in selected) / len(selected) if selected else 0.0)
        clearance_median.append(_median([limb.swing_clearance_m[k] for k in selected]) / leg if selected else 0.0)
        slipped = _cumulative(held * slip[:, i])
        distances: list[float] = []
        travels: list[float] = []
        for s, e in limb.stances:
            if s > 0 and start <= time[s] <= end:
                distances.append(float(slipped[e] - slipped[s]))
                moved = root[min(e, count - 1), :2] - root[s, :2]
                travels.append(abs(float(moved[0] * dx_dir + moved[1] * dy_dir)))
        skid.append(
            math.fsum(distances) / max(math.fsum(travels), _SKID_MIN_TRAVEL_OVER_LEG * leg) if distances else 0.0
        )
        slip_median.append(_median(distances) / leg if distances else 0.0)
        slip_p90.append(float(np.percentile(distances, 90)) / leg if distances else 0.0)

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

    # ---- contralateral symmetry, lead exchange, step length --------------------
    hysteresis = protocol.lead_hysteresis_over_leg * leg
    contralateral: dict[str, Any] = {}
    step_length: dict[str, float | None] = {}
    for first_name, second_name in CONTRALATERAL_PAIRS[feet]:
        a, b = index[first_name], index[second_name]
        key = f"{first_name}|{second_name}"
        ahead = (position[:, b, 0] - position[:, a, 0]) * dx_dir + (position[:, b, 1] - position[:, a, 1]) * dy_dir
        flags: list[bool] = []
        for reference in (a, b):
            events = limbs[reference].touchdowns
            for s0, s1 in zip(events[:-1], events[1:], strict=False):
                if (
                    time[s0] < start
                    or time[s1] > end
                    or not (time[s1] - time[s0]) <= protocol.pause_factor * median_period[reference]
                ):
                    continue
                segment = ahead[s0 : s1 + 1]
                flags.append(bool(np.max(segment) > hysteresis and np.min(segment) < -hysteresis))
        exchange = sum(flags) / len(flags) if flags else 0.0
        windowed = ahead[in_window]
        if len(windowed) >= 4:
            p5, p50, p95 = np.percentile(windowed, [5, 50, 95]).tolist()
            amplitude = 0.5 * (p95 - p5)
            offset_index = abs(p50) / amplitude if amplitude > 1e-9 else 1.0
        else:
            offset_index = 1.0
        for name, own, sign in ((first_name, a, -1.0), (second_name, b, 1.0)):
            # Step length: touchdown foot ahead of the contralateral foot, along travel.
            steps = [
                sign * float(ahead[sample]) / leg for sample in limbs[own].touchdowns if start <= time[sample] <= end
            ]
            step_length[name] = _median(steps) if steps else None
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
        contralateral[key] = {
            "load_ratio": _r(min(foot_impulse[a], foot_impulse[b]) / heavier if heavier > 0.0 else 0.0),
            "duty_ratio": _r(min(duty[a], duty[b]) / longer if longer > 0.0 else 0.0),
            "lead_exchange_fraction": _r(exchange),
            "lead_exchange_strides": len(flags),
            "lead_offset_index": _r(offset_index),
            "alternation_phase_offset_touchdown": _r(anchor_offset["touchdown"]),
            "alternation_phase_offset_midstance": _r(anchor_offset["midstance"]),
            "alternation_index": _r(merged),
            "pci_percent": _r(_phase_coordination_index(stats.events)),
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
            "step_length_over_leg_median": _r(step_length.get(name)),
            "merged_unloads": limbs[i].merged_unloads,
        }

    # ---- templates ----------------------------------------------------------------
    thirds = [start + span * k / 3.0 for k in range(4)]
    templates: dict[str, Any] = {}
    for profile, template in gait_templates(protocol, feet).items():
        joint = in_window.copy()
        gross = np.zeros(count, dtype=bool)
        wrong = np.zeros(count, dtype=bool)
        pairs: dict[str, Any] = {}
        locking_values: list[float] = []
        offsets_by_kind: dict[str, list[float]] = {"alternation": [], "synchrony": [], "limb_phase": []}
        overlaps_by_kind: dict[str, list[float]] = {"alternation": [], "synchrony": []}
        for first_name, second_name, targets, kind in template:
            stats = pair(index[first_name], index[second_name])
            on = stats.local_within(targets, protocol.template_local_extra_tolerance)
            near = stats.local_within(
                targets, protocol.template_local_extra_tolerance + protocol.template_gross_extra_tolerance
            )
            joint &= on
            gross |= in_window & ~near
            wrong |= in_window & stats.locked & ~on
            locking_values.extend((stats.locking, stats.event_locking))
            estimates = (stats.mean, stats.event_mean)
            if kind == "alternation":
                offsets = [_cdist(m, 0.5) if m is not None else 0.5 for m in estimates]
            elif kind == "synchrony":
                offsets = [_cdist(m, 0.0) if m is not None else 0.5 for m in estimates]
            else:
                # folded limb phase: distance from pace (0) on the limb-phase circle
                offsets = [_cdist(m, 0.0) if m is not None else 0.0 for m in estimates]
            offsets_by_kind[kind].extend(offsets)
            if kind in overlaps_by_kind and stats.overlap is not None:
                overlaps_by_kind[kind].append(stats.overlap)
            event_on = (
                sum(
                    any(_cdist(x, c) <= t + protocol.template_local_extra_tolerance for c, t in targets)
                    for x in stats.events
                )
                / len(stats.events)
                if stats.events
                else 0.0
            )
            pairs[f"{first_name}>{second_name}"] = {
                "kind": kind,
                "mean_phase": _r(stats.mean),
                "locking": _r(stats.locking),
                "mean_phase_event": _r(stats.event_mean),
                "locking_event": _r(stats.event_locking),
                "on_template_coverage": _r(_fsum(weights * on) / span),
                "event_on_template_fraction": _r(event_on),
                "n_events": len(stats.events),
                "overlap_index": _r(stats.overlap),
            }
        segments = []
        for k in range(3):
            lo, hi = thirds[k], thirds[k + 1]
            part = np.clip(np.minimum(following, hi) - np.maximum(time, lo), 0.0, None)
            segments.append(_fsum(part * (in_window & ~gross)) / (hi - lo))
        gross_time = _cumulative(weights * gross)
        longest = max((float(gross_time[e] - gross_time[s]) for s, e in boolean_runs(gross)), default=0.0)
        limb_phase_values = offsets_by_kind["limb_phase"]
        templates[profile] = {
            "pairs": pairs,
            "phase_locking_min": _r(min(locking_values)),
            "alternation_phase_offset_max": _r(max(offsets_by_kind["alternation"]))
            if offsets_by_kind["alternation"]
            else None,
            "synchrony_phase_offset_max": _r(max(offsets_by_kind["synchrony"]))
            if offsets_by_kind["synchrony"]
            else None,
            "walk_limb_phase_min": _r(min(limb_phase_values)) if limb_phase_values else None,
            "walk_limb_phase_max": _r(max(limb_phase_values)) if limb_phase_values else None,
            "alternating_overlap_index_max": _r(max(overlaps_by_kind["alternation"]))
            if overlaps_by_kind["alternation"]
            else None,
            "synchronous_overlap_index_min": _r(min(overlaps_by_kind["synchrony"]))
            if overlaps_by_kind["synchrony"]
            else None,
            "template_coverage": _r(_fsum(weights * joint) / span),
            "segment_coverage": [_r(value) for value in segments],
            "min_segment_coverage": _r(min(segments)),
            "gross_off_template_fraction": _r(_fsum(weights * gross) / span),
            "wrong_locked_fraction": _r(_fsum(weights * wrong) / span),
            "longest_off_template_s": _r(longest),
            "longest_off_template_fraction": _r(longest / span),
            "longest_off_template_strides": _r(longest / pooled_period)
            if math.isfinite(pooled_period) and pooled_period > 0.0
            else None,
        }

    # ---- diagnostics ---------------------------------------------------------------
    labels = stride_labels(
        time, limbs, root[:, 2], window=(start, end), median_periods=median_period, protocol=protocol
    )
    froude = max(speed, 0.0) ** 2 / (GRAVITY_MPS2 * leg)
    stride_length = abs(speed) * pooled_period if math.isfinite(pooled_period) else math.nan
    alexander = 2.3 * froude**0.3 * leg if froude > 0.0 else math.nan
    step_values = [value for value in step_length.values() if value is not None]
    pair_records = list(contralateral.values())

    def aggregate(values: list[float], operation: str) -> float | None:
        return _r(min(values) if operation == "min" else max(values)) if values else None

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
        "limb_phase_coverage_min": aggregate(phase_coverage, "min"),
        "limb_duty_min": aggregate(duty, "min"),
        "relative_limb_load_share_min": aggregate([share * feet for share in load_share], "min"),
        "complete_cycles_min": min(complete_cycles),
        "valid_swing_fraction_min": aggregate(valid_fraction, "min"),
        "median_swing_clearance_over_leg_min": aggregate(clearance_median, "min"),
        "skid_fraction_max": aggregate(skid, "max"),
        "pair_load_ratio_min": aggregate([record["load_ratio"] for record in pair_records], "min"),
        "pair_duty_ratio_min": aggregate([record["duty_ratio"] for record in pair_records], "min"),
        "lead_exchange_fraction_min": aggregate([record["lead_exchange_fraction"] for record in pair_records], "min"),
        "step_length_over_leg_min": aggregate(step_values, "min") if len(step_values) == feet else None,
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
        periods = np.diff(anchor)
        overlapping = (anchor[1:] > start) & (anchor[:-1] < end) if len(anchor) >= 2 else np.zeros(0, dtype=bool)
        median = _median(periods[overlapping]) if np.any(overlapping) else math.nan
        value, ok = limb_phase(time, anchor, median, protocol.pause_factor)
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
