"""Serializable gait diagnostics from immutable physics-sample telemetry.

Contact pattern, foot clearance/repositioning and contact-point slip are
measured independently. Duty factor does not assert walking versus running;
grounded running and species-dependent mechanics require additional evidence.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import numpy as np

from environments.shared.gait.events import circular_distance, contact_events
from environments.shared.gait.types import ContactEvents, GaitProtocol

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


def _number(values: list[float], operation: str) -> float | None:
    if not values:
        return None
    return float(getattr(np, operation)(values))


def _empty(errors: list[str], foot_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "telemetry_valid": False,
        "telemetry_errors": errors,
        "duration_s": None,
        "mean_speed_mps": None,
        "progress_over_leg": None,
        "flight_fraction": None,
        "max_flight_s": None,
        "body_support_fraction": None,
        "body_contact_fraction": None,
        "body_loaded_fraction": None,
        "foot_foot_contact_fraction": None,
        "slip_over_leg": None,
        "max_slip_distance_over_leg": None,
        "loaded_slip_speed_mps": None,
        "total_stance_slip_distance_over_leg": None,
        "mean_stance_slip_distance_over_leg": None,
        "phase_match_fraction": None,
        "simultaneous_fraction": None,
        "diagonal_phase_match_fraction": None,
        "ipsilateral_phase_match_fraction": None,
        "four_beat_fraction": None,
        "per_foot": {name: {"complete_cycles": 0, "valid_cycles": 0, "cycles": []} for name in foot_names},
        "pair_phase": {},
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive/exclusive sample runs; final sample has zero integration time."""
    changes = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist(), strict=True))


def _pair_metrics(
    time: np.ndarray,
    first: ContactEvents,
    second: ContactEvents,
    cutoff: float,
    protocol: GaitProtocol,
) -> dict[str, Any]:
    phases: list[float] = []
    periods: list[float] = []
    eligible_cycles = 0
    used_touchdowns: set[int] = set()
    resolution = float(np.max(np.diff(time)))
    for start, _, end in first.complete_cycles:
        if time[start] < cutoff:
            continue
        eligible_cycles += 1
        period = float(time[end] - time[start])
        # Nearest touchdown within half a stride handles synchronous events
        # jittering across phase zero. Exact half-cycle ties prefer the later
        # event. One event cannot be reused to hide a missing opposite step.
        candidates = [
            index for index in second.touchdowns if abs(time[index] - time[start]) <= period / 2.0 + resolution + 1e-12
        ]
        if not candidates:
            continue
        if len(candidates) > 1 and not all(
            abs(abs(time[index] - time[start]) - period / 2.0) <= resolution + 1e-12 for index in candidates
        ):
            continue
        candidates = [index for index in candidates if index not in used_touchdowns]
        if not candidates:
            continue
        paired = min(candidates, key=lambda index: (round(abs(time[index] - time[start]), 12), -time[index]))
        used_touchdowns.add(paired)
        phases.append(float((time[paired] - time[start]) / period % 1.0))
        periods.append(period)
    if not phases:
        return {
            "samples": 0,
            "eligible_cycles": eligible_cycles,
            "mean_phase": None,
            "concentration": None,
            "alternation_match_fraction": 0.0,
            "synchrony_match_fraction": 0.0,
            "simultaneous_touchdown_fraction": 0.0,
            "phases": [],
        }
    array = np.asarray(phases)
    resultant = np.mean(np.exp(2j * np.pi * array))
    mean = float(np.angle(resultant) / (2.0 * np.pi) % 1.0)
    if mean > 1.0 - 1e-12:
        mean = 0.0
    synchrony = circular_distance(array, 0.0)
    return {
        "samples": len(phases),
        "eligible_cycles": eligible_cycles,
        "mean_phase": mean,
        "concentration": float(abs(resultant)),
        "alternation_match_fraction": float(
            np.count_nonzero(circular_distance(array, 0.5) <= protocol.phase_tolerance + 1e-12) / eligible_cycles
        ),
        "synchrony_match_fraction": float(
            np.count_nonzero(synchrony <= protocol.phase_tolerance + 1e-12) / eligible_cycles
        ),
        "simultaneous_touchdown_fraction": float(
            np.count_nonzero(synchrony * np.asarray(periods) <= protocol.simultaneous_s + 1e-12) / eligible_cycles
        ),
        "phases": phases,
    }


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
    diagnostic telemetry, and is validated when present. Integration uses
    elapsed seconds, with each sample held until the next sample. The reset
    sample supplies initial conditions only. Arrays are never mutated.
    """
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
    if feet not in (2, 4) or len(set(foot_names)) != feet:
        errors.append("foot_names must identify two or four distinct feet")
    if not np.isfinite(body_weight_n) or body_weight_n <= 0.0:
        errors.append("body_weight_n must be positive and finite")
    if not np.isfinite(leg_length_m) or leg_length_m <= 0.0:
        errors.append("leg_length_m must be positive and finite")
    if not np.isfinite(settle_s) or settle_s < 0.0:
        errors.append("settle_s must be nonnegative and finite")
    direction = np.asarray(direction_xy, dtype=float).copy()
    if direction.shape != (2,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 0.0:
        errors.append("direction_xy must be a finite, nonzero two-dimensional direction")
    for name in ("floor_force_n", "body_floor_force_n", "foot_foot_force_n", "slip_speed_mps", "touch_force_n"):
        if name in arrays and np.any(arrays[name] < 0.0):
            errors.append(f"negative telemetry: {name}")
    if errors:
        return _empty(errors, foot_names)
    cutoff = float(time[0] + settle_s)
    duration = float(time[-1] - cutoff)
    if duration <= 0.0:
        return _empty(["settling exclusion leaves no measured duration"], foot_names)
    weights = np.maximum(0.0, time[1:] - np.maximum(time[:-1], cutoff))
    events = [
        contact_events(time, arrays["floor_force_n"][:, index], body_weight_n=body_weight_n, protocol=protocol)
        for index in range(feet)
    ]
    loaded = np.stack([event.loaded for event in events], axis=1)
    support_count = np.sum(loaded, axis=1)
    flight_runs = [float(np.sum(weights[start : min(end, len(weights))])) for start, end in _runs(support_count == 0)]
    # Slip and non-foot support use every positive physical ground force;
    # detector thresholds must not make subthreshold dragging disappear.
    force_weight = arrays["floor_force_n"][:-1]
    denominator = float(np.sum(force_weight * weights[:, None]))
    rms_slip = (
        float(np.sqrt(np.sum(force_weight * arrays["slip_speed_mps"][:-1] ** 2 * weights[:, None]) / denominator))
        if denominator > 0.0
        else None
    )
    direction /= np.linalg.norm(direction)
    origin = np.array([np.interp(cutoff, time, arrays["root_position_m"][:, dimension]) for dimension in range(2)])
    progress = float(np.dot(arrays["root_position_m"][-1, :2] - origin, direction))
    body_impulse = float(np.dot(weights, arrays["body_floor_force_n"][:-1]))
    ground_impulse = body_impulse + denominator
    result: dict[str, Any] = {
        "telemetry_valid": True,
        "telemetry_errors": [],
        "duration_s": duration,
        "mean_speed_mps": progress / duration,
        "progress_over_leg": progress / leg_length_m,
        "flight_fraction": float(np.dot(weights, support_count[:-1] == 0) / duration),
        "max_flight_s": max(flight_runs, default=0.0),
        "body_support_fraction": body_impulse / ground_impulse if ground_impulse > 0.0 else 0.0,
        "body_contact_fraction": float(np.dot(weights, arrays["body_floor_force_n"][:-1] > 0.0) / duration),
        "body_loaded_fraction": float(
            np.dot(weights, arrays["body_floor_force_n"][:-1] >= protocol.contact_on_bw * body_weight_n) / duration
        ),
        "foot_foot_contact_fraction": float(
            np.dot(weights, arrays["foot_foot_force_n"][:-1] >= protocol.contact_on_bw * body_weight_n) / duration
        ),
        "loaded_slip_speed_mps": rms_slip,
        "support_count_fraction": {
            str(number): float(np.dot(weights, support_count[:-1] == number) / duration) for number in range(feet + 1)
        },
        "per_foot": {},
        "pair_phase": {},
    }
    stance_slips: list[float] = []
    for index, name in enumerate(foot_names):
        slips = []
        for start, end in _runs(loaded[:, index] | (arrays["floor_force_n"][:, index] > 0.0)):
            stop = min(end, len(weights))
            if float(np.sum(weights[start:stop])) <= 0.0:
                continue
            measured_load = arrays["floor_force_n"][start:stop, index] > 0.0
            slips.append(
                float(
                    np.dot(weights[start:stop], arrays["slip_speed_mps"][start:stop, index] * measured_load)
                    / leg_length_m
                )
            )
        stance_slips.extend(slips)
        cycles = []
        for start, lift, end in events[index].complete_cycles:
            if time[start] < cutoff:
                continue
            period = float(time[end] - time[start])
            clearance = float(np.max(arrays["foot_clearance_m"][lift : end + 1, index]) / leg_length_m)
            reposition = float(
                np.linalg.norm(arrays["foot_position_m"][end, index, :2] - arrays["foot_position_m"][lift, index, :2])
                / leg_length_m
            )
            duty = float((time[lift] - time[start]) / period)
            cycles.append(
                {
                    "touchdown_s": float(time[start]),
                    "liftoff_s": float(time[lift]),
                    "next_touchdown_s": float(time[end]),
                    "period_s": period,
                    "duty_factor": duty,
                    "clearance_over_leg": clearance,
                    "reposition_over_leg": reposition,
                    "valid": bool(
                        clearance + 1e-12 >= protocol.min_clearance_over_leg
                        and reposition + 1e-12 >= protocol.min_reposition_over_leg
                    ),
                }
            )
        valid = sum(cycle["valid"] for cycle in cycles)
        periods = [cycle["period_s"] for cycle in cycles]
        result["per_foot"][name] = {
            "complete_cycles": len(cycles),
            "valid_cycles": valid,
            "cycle_valid_fraction": valid / len(cycles) if cycles else None,
            "duty_factor_mean": _number([cycle["duty_factor"] for cycle in cycles], "mean"),
            "duty_factor_min": _number([cycle["duty_factor"] for cycle in cycles], "min"),
            "duty_factor_max": _number([cycle["duty_factor"] for cycle in cycles], "max"),
            "stride_period_s_mean": _number(periods, "mean"),
            "cadence_hz": _number([1.0 / period for period in periods], "mean"),
            "clearance_over_leg_min": _number([cycle["clearance_over_leg"] for cycle in cycles], "min"),
            "reposition_over_leg_min": _number([cycle["reposition_over_leg"] for cycle in cycles], "min"),
            "stance_slip_over_leg_max": max(slips, default=0.0),
            "cycles": cycles,
        }
    result["slip_over_leg"] = result["max_slip_distance_over_leg"] = max(stance_slips, default=0.0)
    result["total_stance_slip_distance_over_leg"] = float(sum(stance_slips))
    result["mean_stance_slip_distance_over_leg"] = float(np.mean(stance_slips)) if stance_slips else None
    for first, second in combinations(range(feet), 2):
        result["pair_phase"][f"{foot_names[first]}|{foot_names[second]}"] = _pair_metrics(
            time, events[first], events[second], cutoff, protocol
        )

    def pair_fraction(first_name: str, second_name: str, field: str) -> float | None:
        for key in (f"{first_name}|{second_name}", f"{second_name}|{first_name}"):
            pair = result["pair_phase"].get(key)
            if pair is not None and pair["eligible_cycles"] > 0:
                return float(pair[field])
        return None

    def combined(pairs: tuple[tuple[str, str], ...], field: str, operation: str = "min") -> float | None:
        values = [pair_fraction(first, second, field) for first, second in pairs]
        return (
            _number([value for value in values if value is not None], operation)
            if all(value is not None for value in values)
            else None
        )

    contralateral = ((foot_names[0], foot_names[1]),) if feet == 2 else (("fr", "fl"), ("rr", "rl"))
    result["phase_match_fraction"] = combined(contralateral, "alternation_match_fraction")
    result["simultaneous_fraction"] = combined(contralateral, "simultaneous_touchdown_fraction", "max")
    result["diagonal_phase_match_fraction"] = (
        combined((("fr", "rl"), ("fl", "rr")), "synchrony_match_fraction") if feet == 4 else None
    )
    result["ipsilateral_phase_match_fraction"] = (
        combined((("fr", "rr"), ("fl", "rl")), "synchrony_match_fraction") if feet == 4 else None
    )
    four_beat = []
    if feet == 4:
        for start, _, end in events[0].complete_cycles:
            if time[start] < cutoff:
                continue
            contacts = [[index for index in event.touchdowns if start <= index < end] for event in events]
            if any(len(indices) != 1 for indices in contacts):
                four_beat.append(False)
                continue
            ordered = np.sort([time[indices[0]] for indices in contacts])
            gaps = np.diff(np.concatenate((ordered, [time[end]])))
            four_beat.append(bool(np.all(gaps > protocol.simultaneous_s + 1e-12)))
    result["four_beat_fraction"] = float(np.mean(four_beat)) if four_beat else None
    return result
