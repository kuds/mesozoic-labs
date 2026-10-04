"""Per-stride gait labels on Hildebrand's gait plot (diagnostic only).

Every stride of every stepping limb is labelled from its footfall timing:
the touchdown phase of each other limb inside a one-stride window centred on
that limb's local mean phase (Hildebrand's convention, so a jittered
synchronous foot is not lost across phase zero), the stride's duty factors
and its flight fraction.

* Bipeds: walk, grounded run (duty <= 0.5 or Cavagna-style bouncing trunk) or
  aerial run when the pair alternates; hop, staggered hop, skip otherwise;
  one-leg and double-step strides when the other foot lands 0 or 2+ times.
* Quadrupeds (Hildebrand 1976, 1977): with both girdles alternating, the limb
  phase (fore lag behind the ipsilateral hind) gives pace, trot or a lateral-
  or diagonal-sequence walk / amble, prefixed walking / running / flying;
  otherwise pronk, bound, half-bound, canter, transverse or rotary gallop.

Labels never enter a verdict. They describe a certified or failed episode in
the vocabulary of comparative biomechanics. Strides longer than the pause
factor are ``pause``; shorter than half the median stride, ``stutter``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from environments.shared.gait.types import GaitProtocol, LimbContacts

#: Descriptive bands, in cycles (labels only, never a verdict input).
PAIR_SYMMETRY_BAND = 0.15
PAIR_SYNC_BAND = 0.15
PAIR_TOGETHER_BAND = 0.05
FOUR_BEAT_MIN_GAP = 0.0625
AERIAL_FLIGHT_FRACTION = 0.02
STUTTER_FACTOR = 0.5

FAMILY = {
    "walk": "alternating",
    "grounded_run": "alternating",
    "aerial_run": "alternating",
    "hop": "hop",
    "staggered_hop": "hop",
    "skip": "skip",
    "LS_walk": "walk",
    "DS_walk": "walk",
    "LS_amble": "walk",
    "DS_amble": "walk",
    "LS_flying_amble": "walk",
    "DS_flying_amble": "walk",
    "walking_trot": "trot",
    "running_trot": "trot",
    "flying_trot": "trot",
    "walking_pace": "pace",
    "running_pace": "pace",
    "flying_pace": "pace",
    "transverse_gallop": "gallop",
    "rotary_gallop": "gallop",
    "canter": "gallop",
    "half_bound": "bound",
    "bound": "bound",
    "pronk": "pronk",
}


def _circular(x: float) -> float:
    return x - math.floor(x)


def _cdist(a: float, b: float) -> float:
    d = _circular(a - b)
    return min(d, 1.0 - d)


def circular_mean(phases: Sequence[float]) -> tuple[float | None, float]:
    """Circular mean in ``[0, 1)`` and mean resultant length, order-free sums."""
    if not phases:
        return None, 0.0
    c = math.fsum(math.cos(2.0 * math.pi * x) for x in phases) / len(phases)
    s = math.fsum(math.sin(2.0 * math.pi * x) for x in phases) / len(phases)
    r = math.hypot(c, s)
    if r < 1e-12:
        return None, 0.0
    return _circular(math.atan2(s, c) / (2.0 * math.pi)), r


def _bouncing(stance: NDArray[np.bool_], root_z: NDArray[np.float64], start: int, end: int) -> bool:
    """Trunk lower at mid-stance than at touchdown/lift-off (spring-mass bouncing, Cavagna et al. 1977)."""
    off = np.flatnonzero(~stance[start:end])
    lift = start + int(off[0]) if len(off) else end
    if lift - start < 3:
        return False
    middle = (start + lift) // 2
    return bool(root_z[middle] < 0.5 * (root_z[start] + root_z[lift - 1]))


def _label_biped(count: int, phase: float | None, duty: list[float], flight: float, bouncing: bool) -> str:
    if count == 0 or phase is None:
        return "one_leg"
    if count >= 2:
        return "double_step"
    if _cdist(phase, 0.5) <= PAIR_SYMMETRY_BAND:
        if flight > AERIAL_FLIGHT_FRACTION:
            return "aerial_run"
        if 0.5 * (duty[0] + duty[1]) <= 0.5 or bouncing:
            return "grounded_run"
        return "walk"
    if _cdist(phase, 0.0) <= PAIR_TOGETHER_BAND:
        return "hop"
    if _cdist(phase, 0.0) <= PAIR_SYNC_BAND:
        return "staggered_hop"
    return "skip"


def _label_quad(
    counts: dict[int, int], phases: dict[int, float], duty: list[float], flight: float, protocol: GaitProtocol
) -> tuple[str, float | None, float | None]:
    """Canonical limbs 0 fr, 1 fl, 2 rr, 3 rl; returns (label, limb phase, hind duty)."""
    if any(count == 0 for count in counts.values()):
        return "three_legged", None, None
    if any(count >= 2 for count in counts.values()):
        return "irregular", None, None
    fr, fl, rr, rl = (phases[index] for index in range(4))
    hind = _circular(rl - rr)
    fore = _circular(fl - fr)
    hind_duty = 0.5 * (duty[2] + duty[3])
    if _cdist(hind, 0.5) <= PAIR_SYMMETRY_BAND and _cdist(fore, 0.5) <= PAIR_SYMMETRY_BAND:
        limb_phase, _ = circular_mean([_circular(fr - rr), _circular(fl - rl)])
        if limb_phase is None:
            return "irregular", None, None
        aerial = flight > AERIAL_FLIGHT_FRACTION
        folded = _cdist(limb_phase, 0.0)
        if folded < protocol.template_walk_band_low:
            kind = "pace"
        elif folded > protocol.template_walk_band_high:
            kind = "trot"
        else:
            ordered = sorted([fr, fl, rr, rl])
            gaps = [ordered[i + 1] - ordered[i] for i in range(3)] + [ordered[0] + 1.0 - ordered[3]]
            if min(gaps) < FOUR_BEAT_MIN_GAP:
                return "irregular", limb_phase, hind_duty
            sequence = "LS" if limb_phase < 0.5 else "DS"
            if aerial:
                return f"{sequence}_flying_amble", limb_phase, hind_duty
            return (f"{sequence}_walk" if hind_duty > 0.5 else f"{sequence}_amble"), limb_phase, hind_duty
        if aerial:
            return f"flying_{kind}", limb_phase, hind_duty
        return (f"walking_{kind}" if hind_duty > 0.5 else f"running_{kind}"), limb_phase, hind_duty
    hind_together = _cdist(hind, 0.0) <= PAIR_TOGETHER_BAND
    fore_together = _cdist(fore, 0.0) <= PAIR_TOGETHER_BAND
    if hind_together and fore_together:
        return ("pronk" if _cdist(fr - rr, 0.0) <= PAIR_TOGETHER_BAND else "bound"), None, hind_duty
    if hind_together or fore_together:
        return "half_bound", None, hind_duty
    if _cdist(fr - rl, 0.0) <= PAIR_TOGETHER_BAND or _cdist(fl - rr, 0.0) <= PAIR_TOGETHER_BAND:
        return "canter", None, hind_duty
    hind_lead = hind < 0.5
    fore_lead = fore < 0.5
    return ("transverse_gallop" if hind_lead == fore_lead else "rotary_gallop"), None, hind_duty


def stride_labels(
    time_s: NDArray[np.float64],
    contacts: Sequence[LimbContacts],
    root_z: NDArray[np.float64],
    *,
    window: tuple[float, float],
    median_periods: Sequence[float],
    protocol: GaitProtocol,
) -> dict[str, Any]:
    """Time-weighted label and family distributions over the analysis window."""
    start_s, end_s = window
    span = end_s - start_s
    feet = len(contacts)
    held = np.diff(np.concatenate((time_s, time_s[-1:])))
    stance = np.stack([limb.stance for limb in contacts], axis=1)
    stance_time = [np.concatenate(([0.0], np.cumsum(held * stance[:, f]))) for f in range(feet)]
    flight_time = np.concatenate(([0.0], np.cumsum(held * ~np.any(stance, axis=1))))
    touchdown_times = [time_s[np.asarray(limb.touchdowns, dtype=int)] for limb in contacts]
    seconds: dict[str, float] = defaultdict(float)
    limb_phases: list[float] = []
    hind_duties: list[float] = []
    stepping = 0
    for reference in range(feet):
        events = list(contacts[reference].touchdowns)
        strides = [
            (first, second)
            for first, second in zip(events[:-1], events[1:], strict=True)
            if time_s[second] > start_s and time_s[first] < end_s
        ]
        if not strides:
            continue
        stepping += 1
        starts = np.asarray([time_s[first] for first, _ in strides])
        periods = np.asarray([time_s[second] - time_s[first] for first, second in strides])
        others = [other for other in range(feet) if other != reference]
        contained: dict[int, list[list[float]]] = {}
        for other in others:
            lists: list[list[float]] = [[] for _ in strides]
            if len(touchdown_times[other]):
                slot = np.searchsorted(starts, touchdown_times[other], side="right") - 1
                for when, k in zip(touchdown_times[other].tolist(), slot.tolist(), strict=True):
                    if 0 <= k < len(strides) and when < starts[k] + periods[k]:
                        lists[k].append((when - starts[k]) / periods[k])
            contained[other] = lists
        median_period = median_periods[reference]
        for k, (first, second) in enumerate(strides):
            t0, period = float(starts[k]), float(periods[k])
            weight = max(0.0, min(t0 + period, end_s) - max(t0, start_s))
            if math.isfinite(median_period) and period > protocol.pause_factor * median_period:
                seconds["pause"] += weight
                continue
            if math.isfinite(median_period) and period < STUTTER_FACTOR * median_period:
                seconds["stutter"] += weight
                continue
            counts: dict[int, int] = {}
            phases: dict[int, float] = {reference: 0.0}
            for other in others:
                near = [p for j in range(max(0, k - 2), min(len(strides), k + 3)) for p in contained[other][j]]
                centre, locking = circular_mean(near)
                c = centre if centre is not None and locking >= 0.5 else 0.5
                low, high = t0 + (c - 0.5) * period, t0 + (c + 0.5) * period
                found = touchdown_times[other][(touchdown_times[other] >= low) & (touchdown_times[other] < high)]
                counts[other] = len(found)
                if len(found) == 1:
                    phases[other] = _circular((float(found[0]) - t0) / period)
            duty = [float(stance_time[f][second] - stance_time[f][first]) / period for f in range(feet)]
            flight = float(flight_time[second] - flight_time[first]) / period
            if feet == 2:
                other = others[0]
                label = _label_biped(
                    counts[other],
                    phases.get(other),
                    duty,
                    flight,
                    _bouncing(contacts[reference].stance, root_z, first, second),
                )
            else:
                label, limb_phase_value, hind_duty = _label_quad(counts, phases, duty, flight, protocol)
                if limb_phase_value is not None and hind_duty is not None:
                    limb_phases.append(limb_phase_value)
                    hind_duties.append(hind_duty)
            seconds[label] += weight
    distribution: dict[str, float] = defaultdict(float)
    if stepping and span > 0:
        for label, value in seconds.items():
            distribution[label] += value / span / stepping
    covered = math.fsum(distribution.values())
    if covered < 1.0:
        distribution["no_complete_stride"] += 1.0 - covered
    families: dict[str, float] = defaultdict(float)
    for label, value in distribution.items():
        families[FAMILY.get(label, label)] += value
    limb_phase_mean, limb_phase_concentration = circular_mean(limb_phases)
    ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    family_ranked = sorted(families.items(), key=lambda item: (-item[1], item[0]))
    if family_ranked:
        top_family, top_share = family_ranked[0]
        members = [item for item in ranked if FAMILY.get(item[0], item[0]) == top_family]
        name = f"{members[0][0]} ({100.0 * top_share:.0f}% {top_family})"
    else:
        name = "no steps"
    return {
        "label_distribution": dict(ranked),
        "family_distribution": dict(family_ranked),
        "gait_label": name,
        "stepping_limbs": stepping,
        "limb_phase_mean": limb_phase_mean,
        "limb_phase_concentration": limb_phase_concentration if limb_phases else None,
        "hind_duty_median": float(np.median(hind_duties)) if hind_duties else None,
    }
