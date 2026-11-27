"""Per-episode locomotion gait metrics on a floor-truth :class:`.recorder.EpisodeTrace` (pure numpy).

:func:`episode_locomotion_metrics` reduces one recorded episode to a
:class:`LocomotionEpisodeMetrics` row: the separators that
GAIT_QUALITY_PLAN_2026_09.md GQ-8 (a) enforces first on bipeds, the floors
GQ-9 (a) enforces on quadrupeds, both as §3.3 defines them, and the
report-only numbers beside them.  The ``command_tracking/v1`` gait clause
(``curriculum/command_tracking_gate.py``, decision D-D61) reads them, so a
command-following gate judges the footfalls, not the pelvis velocity alone
(GQ-18).  No mujoco import (the trace class is imported for
its type only), so the pure gate module can import this one.

The measurement
---------------
A limb is DOWN on a control step under :func:`.events.down_mask` (its
whole-limb floor force above the contact threshold on at least half of the
step's substeps, the stance gate's floor truth).  Touchdowns are the rising
edges of each limb's down state after :func:`.events.debounce` at
:data:`.constants.DEBOUNCE_S`.  A left/right pair's touchdowns merge into one
simultaneous event ``B`` when they fall within :data:`SIMULTANEOUS_S`
(:func:`.events.pair_touchdown_events`).  Pairs: (l, r) on a biped; fore
(fl, fr) and hind (rl, rr) on a quadruped.  The WINDOW is every step from
:data:`GAIT_WINDOW_S` on (§3.3: "after 1 s for locomotion"), so the reset
transient and the first step off the spawn pose are not judged; touchdowns
are found on the whole episode and counted in the window.

* ``flight_fraction``: window steps with no limb down / window steps.
* ``hop_flight_fraction``: window flight steps in flight intervals NOT
  opened by exactly one limb lifting and closed by exactly one OTHER limb
  landing, two limbs lifting (landing) within :data:`SIMULTANEOUS_S` of each
  other counting as lifting (landing) together, / window steps.  An
  alternating run's aerial phase (one foot pushes off, the other lands) is
  not hop flight; a two-footed hop's is.  An interval still open when the
  episode ends has no observed landing and is left out (its steps still
  count in ``flight_fraction``).
* ``alternation_index`` (bipeds; NaN on quadrupeds): on the merged
  touchdown sequence, consecutive pairs that switch limb (``L R`` or
  ``R L``) / pairs; NaN with fewer than two events.
* ``pair_alternation_index``: the same index for each left/right pair, the
  LOWER of the fore and hind pairs' on a quadruped (GQ-9's "fore and hind
  pair alternation"); on a biped it equals ``alternation_index``.
* ``simultaneous_fraction``: ``B`` events / events (NaN with none); the
  HIGHER of the two pairs' on a quadruped.
* ``lead_swaps_per_stride``: changes of the leading foot per stride (NaN
  without a stride); the LOWER of the two pairs' on a quadruped.  The lead
  is read from the fore-aft offset ``x_left - x_right`` of the pair's foot
  reference sites along the root's heading of the step, through a DEAD BAND
  of :data:`LEAD_DEAD_BAND_L` leg lengths: the left foot leads once the
  offset exceeds ``+LEAD_DEAD_BAND_L * leg_length_m``, the right once it
  falls below ``-LEAD_DEAD_BAND_L * leg_length_m``, and in between the lead
  stays what it was (a Schmitt trigger); the lead series is then debounced
  at :data:`.constants.DEBOUNCE_S` and its changes counted.  A lead the
  offset never takes leaves no change (0 swaps, not NaN).  A walk swaps the
  leading foot twice a stride; a skip or a staggered scoot keeps one foot
  ahead; a two-footed hop with the feet side by side never moves its offset
  past the band, so it reads 0 however its footfalls are staggered.  The
  band is what ``locomotion-gait/v2`` changed: under v1 (the probe's
  definition, a plain sign change) the offset of feet side by side
  flickered about zero on the real plants, so the scripted T. rex hop read
  2.5 lead swaps a stride and a hop whose feet landed more than 20 ms apart
  passed every biped separator.
* ``leg_length_m``: the scale the dead band is in, the spawn root's height
  above the mean height of the spawn foot reference sites (the probe's leg
  length L is the home keyframe's hip height; the two agree to within the
  foot sites' height on every model).  NaN when it is not positive, which
  leaves the lead swaps unmeasured.
* ``strides``: window seconds x the stride frequency, the mean over limbs of
  1 / the limb's mean touchdown-to-touchdown interval (limbs with fewer than
  two window touchdowns left out; NaN when none has two).
* ``min_limb_duty``: the lowest over limbs of the fraction of window steps
  down.
* ``foot_contact_fraction``: the window mean of the fraction of substeps
  with any leg-on-leg contact (a sole resting on the other foot), the stance
  metrics' ``foot_on_foot_fraction`` over this window.
* ``phantom_support_fraction``: window steps on which some limb's touch
  sensors read above the contact threshold while the floor says the limb is
  up, the stance metrics' definition over this window.

Report-only (no threshold key reads them): ``all_limb_support_fraction``
(the slide's signature: every limb down), ``touchdown_rate`` (touchdowns
per limb per second), ``stride_frequency_hz``, ``window_displacement_m`` and
``displacement_speed_mps``, ``leg_length_m``, ``lead_amplitude_m`` (half the
5th-to-95th percentile range of the fore-aft offset over the window, the
probe's ``lead_amplitude_m``; the LOWER of the two pairs' on a quadruped),
the quadruped pairs' own indices (``fore_*``, ``hind_*``), and:

* ``left_right_phase`` (bipeds; NaN on a quadruped): the right foot's first
  touchdown in each left stride (from one left touchdown to the next), as a
  fraction of that stride, circular mean in [0, 1) (§3.3's left/right phase
  φ).  A walk or a run lies near 0.5; a two-footed hop near 0, its stagger
  over its period when its footfalls are staggered.  §4.4 has
  ``|φ - 0.5| <= 0.15`` separating the walkers from the hops but measured on
  eight traces only, so it is not a bar yet.
* ``stance_slip_speed_mps``: the mean horizontal speed of a limb's foot
  reference site over the window steps on which that limb is down on the
  step and the one before (floor truth, not debounced), over every limb;
  NaN when no limb stays down two steps.  A planted foot reads near 0, a
  skid at the body's speed; GQ-8 (a) and GQ-9 (a) keep slip report-only (the
  velociraptor's walk reads 0.9-1.1 m/s from its toe roll, §4.4), so a skid
  broken by brief alternating lifts is measured here and judged by no bar.
* ``ipsilateral_phase`` (Hildebrand; quadrupeds, NaN on a biped): the
  forefoot's first touchdown after the same side's hind touchdown, as a
  fraction of that hind stride, circular mean over both sides (a
  lateral-sequence walk lies between 0 and 0.5, a trot at 0.5, a pace at 0).
  §4.3 B's band ([0.10, 0.45] brachiosaurus, [0.10, 0.60] dibothrosuchus) is
  report-only under GQ-9 (a): it was measured on no passing quadruped.

The definitions are the audit probe's
(``docs/investigations/gait_2026_09/gait_probe.py``, the source of §4.3's
calibration) on the library's floor-truth channel, with every window in
seconds, except the lead swaps' dead band (above).  NaN means UNMEASURED, and a gate bar never passes NaN.  Changing a
definition here changes what an adopted bar means: bump
:data:`LOCOMOTION_METRICS_VERSION`, which every reduced panel episode records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from . import events
from .constants import CONTACT_THRESHOLD_N, DEBOUNCE_S, MEASUREMENT_VERSION

if TYPE_CHECKING:
    from .recorder import EpisodeTrace

NAN = float("nan")

#: The version of the definitions above.  Bump it whenever one changes what a
#: metric means; :data:`GAIT_MEASUREMENT` joins it to the floor-truth
#: measurement version the down state is read under.
LOCOMOTION_METRICS_VERSION = "locomotion-gait/v2"
#: Everything that fixes what a locomotion metric means: the floor-truth
#: constants' version and these definitions'.  A panel episode records it, and
#: the gate refuses one measured under another.
GAIT_MEASUREMENT = f"{MEASUREMENT_VERSION}+{LOCOMOTION_METRICS_VERSION}"

#: The locomotion window starts this long after reset (seconds): §3.3.
GAIT_WINDOW_S = 1.0
#: Left and right touchdowns this close are one simultaneous event (seconds): §3.1's merge window, the same 20 ms as
#: the debounce.
SIMULTANEOUS_S = DEBOUNCE_S

#: The lead swaps' dead band, in leg lengths (``leg_length_m``): the fore-aft offset must pass this far beyond zero
#: for its sign to name the leading foot.  The repository's reference walks swing the offset 0.09 (allosaurus) to
#: 0.44 (T. rex) leg lengths either side of zero (``lead_amplitude_m``), the scripted hops of the zero-action statues
#: less than 0.003; at 0.03 every reference walk keeps at least 1.58 swaps a stride and every scripted hop reads 0.
LEAD_DEAD_BAND_L = 0.03

#: The limb labels a biped and a quadruped trace carry (``Morphology.labels``).
BIPED_LABELS = frozenset({"l", "r"})
QUADRUPED_LABELS = frozenset({"fl", "fr", "rl", "rr"})


@dataclass(frozen=True)
class LocomotionEpisodeMetrics:
    """One episode's locomotion gait metrics; the module docstring defines each."""

    limbs: int
    window_start_step: int
    window_steps: int
    # --- support ---
    flight_fraction: float
    hop_flight_fraction: float
    min_limb_duty: float
    all_limb_support_fraction: float
    # --- footfall sequence ---
    touchdown_events: float
    alternation_index: float
    pair_alternation_index: float
    simultaneous_fraction: float
    lead_swaps_per_stride: float
    strides: float
    stride_frequency_hz: float
    touchdown_rate: float
    leg_length_m: float
    lead_amplitude_m: float
    left_right_phase: float
    # --- contact faults ---
    foot_contact_fraction: float
    phantom_support_fraction: float
    # --- kinematics ---
    window_displacement_m: float
    displacement_speed_mps: float
    stance_slip_speed_mps: float
    # --- quadruped pairs and limb phase (NaN on a biped) ---
    fore_alternation_index: float = NAN
    hind_alternation_index: float = NAN
    fore_simultaneous_fraction: float = NAN
    hind_simultaneous_fraction: float = NAN
    fore_lead_swaps_per_stride: float = NAN
    hind_lead_swaps_per_stride: float = NAN
    ipsilateral_phase: float = NAN

    def as_gait_metrics(self) -> dict[str, float]:
        """Every metric by name, as floats: the ``gait_metrics`` a panel episode record carries."""
        return {spec.name: float(getattr(self, spec.name)) for spec in fields(self)}


#: The field names of :class:`LocomotionEpisodeMetrics`, in declaration order.
LOCOMOTION_METRIC_FIELDS: tuple[str, ...] = tuple(spec.name for spec in fields(LocomotionEpisodeMetrics))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if np.size(values) else NAN


def _yaw(quat: np.ndarray) -> np.ndarray:
    """(T,4) MuJoCo (w, x, y, z) quaternions -> (T,) yaw of the body +x axis on the ground plane."""
    w, x, y, z = (quat[:, index] for index in range(4))
    heading: np.ndarray = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return heading


def _pairs(labels: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """``[(name, left index, right index), ...]``: ``[("", l, r)]`` on a biped, fore then hind on a quadruped."""
    index = {label: position for position, label in enumerate(labels)}
    if set(labels) == BIPED_LABELS and len(labels) == 2:
        return [("", index["l"], index["r"])]
    if set(labels) == QUADRUPED_LABELS and len(labels) == 4:
        return [("fore", index["fl"], index["fr"]), ("hind", index["rl"], index["rr"])]
    raise ValueError(
        f"the trace's limbs {list(labels)} are neither a biped's (l, r) nor a quadruped's (fl, fr, rl, rr); the "
        "locomotion metrics pair the left and right limbs by label"
    )


def _sequence_indices(sequence: list[tuple[int, str]]) -> tuple[float, float]:
    """(alternation index, simultaneous fraction) of a merged pair sequence; NaN where undefined."""
    labels = [label for _, label in sequence]
    pairs = len(labels) - 1
    switching = sum(
        1 for first, second in zip(labels, labels[1:]) if {first, second} == {events.LEFT_EVENT, events.RIGHT_EVENT}
    )
    alternation = switching / pairs if pairs > 0 else NAN
    simultaneous = labels.count(events.BOTH_EVENT) / len(labels) if labels else NAN
    return alternation, simultaneous


def _hop_flight_steps(down: np.ndarray, window_start: int, tolerance: int) -> int:
    """Window flight steps in intervals not opened by one limb lifting and closed by one other landing.

    A flight interval runs from step ``start`` (no limb down) to ``end`` (a
    limb down again).  Its OPENERS are the limbs that lifted off at most
    *tolerance* steps before ``start`` (down on some step of
    ``[start - tolerance - 1, start - 1]``), its CLOSERS the limbs that
    touched down at most *tolerance* steps after ``end`` (down on some step
    of ``[end, end + tolerance]``): the same window as the touchdown merge,
    so two limbs that lift or land within it count as lifting or landing
    together.  An interval still open when the episode ends has no observed
    landing and is left out.
    """
    flight = ~np.any(down, axis=1)
    steps = flight.size
    if steps == 0:
        return 0
    change = np.flatnonzero(np.diff(flight.astype(np.int8))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [steps]])
    counted = 0
    for start, end in zip(starts.tolist(), ends.tolist()):
        if not flight[start] or end >= steps:
            continue
        judged = end - max(start, window_start)
        if judged <= 0:
            continue
        openers = set(np.flatnonzero(down[max(0, start - tolerance - 1) : start].any(axis=0)).tolist())
        closers = set(np.flatnonzero(down[end : end + tolerance + 1].any(axis=0)).tolist())
        if not (len(openers) == 1 and len(closers) == 1 and openers != closers):
            counted += judged
    return counted


def _leading(offset: np.ndarray, band: float) -> np.ndarray:
    """The leading foot of each step through the dead band: True left, False right, from the first step past it.

    The offset names the lead where it lies beyond ``+band`` (left) or below
    ``-band`` (right); in between the lead stays what it was.  Steps before
    the offset first leaves the band are dropped (the lead is not yet
    known), so an offset that never leaves it returns an empty series.
    """
    mark = np.where(offset > band, 1, np.where(offset < -band, -1, 0))
    named = np.flatnonzero(mark)
    if named.size == 0:
        return np.zeros(0, dtype=bool)
    last = np.where(mark != 0, np.arange(mark.size), 0)
    np.maximum.accumulate(last, out=last)
    held: np.ndarray = mark[last][named[0] :] > 0
    return held


def _circular_phase(phases: list[float]) -> float:
    """The circular mean of *phases* (fractions of a cycle) in [0, 1); NaN for none or a mean of no direction."""
    if not phases:
        return NAN
    mean = np.exp(2j * np.pi * np.asarray(phases)).mean()
    if abs(mean) < 1e-12:
        return NAN
    return float((np.angle(mean) / (2.0 * np.pi)) % 1.0)


def _stride_phases(reference: np.ndarray, other: np.ndarray) -> list[float]:
    """Per stride of *reference* (one touchdown to the next), *other*'s first touchdown in it as a fraction of it."""
    phases: list[float] = []
    for start, end in zip(reference[:-1].tolist(), reference[1:].tolist()):
        inside = other[(other >= start) & (other < end)]
        if inside.size:
            phases.append((float(inside[0]) - start) / (end - start))
    return phases


def _ipsilateral_phase(touchdowns: dict[str, np.ndarray]) -> float:
    """The forefoot's touchdown after the same side's hind one per hind stride, circular mean in [0, 1); NaN if none."""
    phases: list[float] = []
    for side in ("l", "r"):
        phases.extend(_stride_phases(touchdowns["r" + side], touchdowns["f" + side]))
    return _circular_phase(phases)


def _stance_slip_speed(foot_xy: np.ndarray, down: np.ndarray, dt: float) -> float:
    """Mean horizontal speed of each limb's site over the steps it is down on and down on the step before; NaN if none."""
    if foot_xy.shape[0] < 2:
        return NAN
    speed = np.linalg.norm(np.diff(foot_xy, axis=0), axis=2) / dt
    planted = down[1:] & down[:-1]
    return _mean(speed[planted])


def _leg_length(trace: "EpisodeTrace") -> float:
    """The spawn root's height above the mean height of the spawn foot sites (m); NaN unless positive and finite."""
    length = float(trace.spawn_root_pos[2]) - float(np.mean(np.asarray(trace.spawn_foot_pos)[:, 2]))
    return length if math.isfinite(length) and length > 0.0 else NAN


def episode_locomotion_metrics(
    trace: "EpisodeTrace", *, window_start_s: float = GAIT_WINDOW_S
) -> LocomotionEpisodeMetrics:
    """The locomotion gait metrics of one recorded episode, over the window from *window_start_s* on.

    Raises ``ValueError`` for a trace with no steps, a negative window start,
    and limbs that are neither a biped's nor a quadruped's.
    """
    if not (math.isfinite(window_start_s) and window_start_s >= 0.0):
        raise ValueError(f"window_start_s must be a non-negative number of seconds, got {window_start_s!r}")
    steps, n_limbs = trace.leg_floor_mean.shape
    if steps == 0:
        raise ValueError("the trace has no steps")
    pairs = _pairs(tuple(trace.labels))
    dt = float(trace.dt)
    start = int(round(window_start_s / dt))
    window = slice(start, steps)
    window_steps = max(steps - start, 0)
    window_s = window_steps * dt
    down = events.down_mask(trace.leg_down_frac)
    debounce_steps = events.steps_for(DEBOUNCE_S, dt)
    merge_steps = events.steps_for(SIMULTANEOUS_S, dt)
    debounced = np.stack([events.debounce(down[:, limb], debounce_steps) for limb in range(n_limbs)], axis=1)
    touchdowns = [events.rising_edges(debounced[:, limb]) for limb in range(n_limbs)]
    in_window = [steps_down[steps_down >= start] for steps_down in touchdowns]

    if window_steps == 0:
        return LocomotionEpisodeMetrics(
            limbs=int(n_limbs),
            window_start_step=start,
            window_steps=0,
            flight_fraction=NAN,
            hop_flight_fraction=NAN,
            min_limb_duty=NAN,
            all_limb_support_fraction=NAN,
            touchdown_events=NAN,
            alternation_index=NAN,
            pair_alternation_index=NAN,
            simultaneous_fraction=NAN,
            lead_swaps_per_stride=NAN,
            strides=NAN,
            stride_frequency_hz=NAN,
            touchdown_rate=NAN,
            leg_length_m=NAN,
            lead_amplitude_m=NAN,
            left_right_phase=NAN,
            foot_contact_fraction=NAN,
            phantom_support_fraction=NAN,
            window_displacement_m=NAN,
            displacement_speed_mps=NAN,
            stance_slip_speed_mps=NAN,
        )

    down_window = down[window]
    touch_up = np.asarray(trace.touch[window]) > CONTACT_THRESHOLD_N

    frequencies = [1.0 / (float(np.mean(np.diff(limb))) * dt) for limb in in_window if limb.size >= 2]
    stride_frequency = float(np.mean(frequencies)) if frequencies else NAN
    strides = window_s * stride_frequency if math.isfinite(stride_frequency) else NAN

    heading = _yaw(np.asarray(trace.root_quat[window], dtype=np.float64))
    forward = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    relative = np.asarray(trace.foot_pos[window, :, :2], dtype=np.float64) - np.asarray(
        trace.root_pos[window, None, :2]
    )
    along = np.einsum("kfi,ki->kf", relative, forward)

    leg_length = _leg_length(trace)
    band = LEAD_DEAD_BAND_L * leg_length
    alternation: dict[str, float] = {}
    simultaneous: dict[str, float] = {}
    lead_swaps: dict[str, float] = {}
    lead_amplitude: dict[str, float] = {}
    n_events = 0
    for name, left, right in pairs:
        sequence = events.pair_touchdown_events(in_window[left], in_window[right], merge_steps)
        n_events += len(sequence)
        alternation[name], simultaneous[name] = _sequence_indices(sequence)
        offset = along[:, left] - along[:, right]
        lead_amplitude[name] = float((np.percentile(offset, 95) - np.percentile(offset, 5)) / 2.0)
        leading = events.debounce(_leading(offset, band), debounce_steps)
        swaps = int(np.count_nonzero(np.diff(leading.astype(np.int8))))
        measured = math.isfinite(strides) and strides > 0.0 and math.isfinite(band)
        lead_swaps[name] = swaps / strides if measured else NAN

    def lowest(values: dict[str, float]) -> float:
        listed = list(values.values())
        return min(listed) if all(math.isfinite(value) for value in listed) else NAN

    def highest(values: dict[str, float]) -> float:
        listed = list(values.values())
        return max(listed) if all(math.isfinite(value) for value in listed) else NAN

    root_before = trace.root_pos[start - 1, :2] if start >= 1 else trace.spawn_root_pos[:2]
    displacement = float(np.linalg.norm(np.asarray(trace.root_pos[steps - 1, :2]) - np.asarray(root_before)))
    quadruped = n_limbs == 4
    extra: dict[str, float] = {}
    if quadruped:
        extra = {
            "fore_alternation_index": alternation["fore"],
            "hind_alternation_index": alternation["hind"],
            "fore_simultaneous_fraction": simultaneous["fore"],
            "hind_simultaneous_fraction": simultaneous["hind"],
            "fore_lead_swaps_per_stride": lead_swaps["fore"],
            "hind_lead_swaps_per_stride": lead_swaps["hind"],
            "ipsilateral_phase": _ipsilateral_phase(
                {label: in_window[position] for position, label in enumerate(trace.labels)}
            ),
        }
    return LocomotionEpisodeMetrics(
        limbs=int(n_limbs),
        window_start_step=start,
        window_steps=int(window_steps),
        flight_fraction=_mean(~np.any(down_window, axis=1)),
        hop_flight_fraction=_hop_flight_steps(down, start, merge_steps) / window_steps,
        min_limb_duty=float(np.min(np.mean(down_window, axis=0))),
        all_limb_support_fraction=_mean(np.all(down_window, axis=1)),
        touchdown_events=float(n_events),
        alternation_index=NAN if quadruped else alternation[""],
        pair_alternation_index=lowest(alternation),
        simultaneous_fraction=highest(simultaneous),
        lead_swaps_per_stride=lowest(lead_swaps),
        strides=strides,
        stride_frequency_hz=stride_frequency,
        touchdown_rate=sum(limb.size for limb in in_window) / (n_limbs * window_s),
        leg_length_m=leg_length,
        lead_amplitude_m=lowest(lead_amplitude),
        left_right_phase=NAN
        if quadruped
        else _circular_phase(_stride_phases(in_window[pairs[0][1]], in_window[pairs[0][2]])),
        foot_contact_fraction=_mean(np.asarray(trace.interleg_substep_frac[window])),
        phantom_support_fraction=_mean(np.any(touch_up & ~down_window, axis=1)),
        window_displacement_m=displacement,
        displacement_speed_mps=displacement / window_s,
        stance_slip_speed_mps=_stance_slip_speed(
            np.asarray(trace.foot_pos[window, :, :2], dtype=np.float64), down_window, dt
        ),
        **extra,
    )
