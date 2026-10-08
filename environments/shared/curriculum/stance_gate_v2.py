"""The floor-truth stance gate statistic: ``stance_quality/v2`` (decision D-D23).

What it replaces, and why
-------------------------
``stance_quality/v1`` certifies a stance on two numbers: the full-horizon
fraction and an upper bound on mean unsupported duty read off the touch
sensors at 0.1 N.  The 2026-10 stance-hack audit rolled every certified and
near-certified stance on the 40-episode panel and found that both numbers
can be met by policies that are not standing: the trex seed-42 stance
(``20260914``) clears every GAIT plan §4.3 C criterion yet lands each episode
at 1.55-2.86 body weights and stands on a rolled pad edge, and the
velociraptor ``20260922`` stance saturates its hip pitch and knee for the
whole episode while one toe is never loaded.  Touch reads the sensor sites,
not the floor, and the per-substep MIN drops the very contacts a hop makes;
a duty mean averages the hop away.

v2 judges the episode instead of a panel mean.  Every episode of the panel
is classified on FLOOR TRUTH -- the floor normal force on each leg's geoms,
decoded per physics substep exactly as ``mj_contactForce`` reports it, by
:mod:`environments.shared.gait` -- and the gate certifies a one-sided 95%
lower confidence bound on the probability that an episode is clean.

The statistic
-------------
An episode is **clean** when it reaches the horizon AND every declared
per-episode criterion holds on a FINITE metric.  NaN means unmeasured (a
window with no steps, a sole metric on a foot with no sole, saturation with
no recorded action), and an unmeasured criterion fails its episode: the
alternative is the fail-open ``nan < bar is False`` that let NaN rewards
clear the v1 rail (``gate_schema.finite_gate_metric``).

The panel **passes** when:

* it holds at least ``min_eval_episodes`` episodes, every one measured with
  the declared ``settle_steps``;
* :func:`~environments.shared.curriculum.recovery_gate.binomial_lcb` (exact
  one-sided Clopper-Pearson, the bound ``recovery_quality/v1`` and
  ``task_success/v1`` certify with) of ``k`` clean of ``n`` clears
  ``min_clean_stance_lcb``;
* the declared panel rails hold: the absolute ``min_avg_reward``, the
  statue-relative ``min_avg_reward_statue_ratio``, the optional
  ``min_full_horizon_fraction`` and the optional
  ``max_hop_or_fall_episodes`` (below).

The hop-or-fall rail
--------------------
The bound admits a panel with a few unclean episodes whatever they failed:
at 37/40 it cannot tell three pad near-misses from three whole-episode hops
or falls.  ``max_hop_or_fall_episodes`` caps the episodes that end before
the horizon or fail one of :data:`HOP_OR_FALL_KEYS` -- the criteria a hop, a
chatter limit cycle or a fall fails (support, touchdowns, drift, command
saturation, and the window hop pair below where declared) -- and leaves the
pad, load-share and settle bars to the bound,
as D-D23 sized them.  It counts from the classification reasons, which a
recorded panel CSV re-derives, so the judge, publication and backfill need
no new input.  Declared by trex since D-D27, where the seed-44 physics-r8
stance kept a whole-episode hop mode on about 4% of its resets inside a
37/40 bound.

Sizing at the 40-episode panel (seeds 3042-3081): 40/40 bounds at 0.928,
39 at 0.887, 38 at 0.851, 37 at 0.817, 36 at 0.786, 35 at 0.755 -- so a bar
of 0.80 admits 37/40 and refuses 36/40 (GAIT plan GQ-7, taken for stance by
D-D23).  v1 rejected a binary count because ITS candidate bar, 0.90, needs a
clean 40/40, which a statue at the pooled 119/120 = 99.17% full-horizon rate
scores only 71.6% of the time.  At 0.80 the same statue clears >= 37/40 with
probability 0.9997, and on the floor-truth criteria the statue classified
clean 40/40 on all six species (floor map §5.4) while every audit hack
episode failed two or more independent criteria.  Each adopting TOML still
has to show its statue clears >= 37/40 under its own declared bars.

Statue-relative criteria
------------------------
``min_avg_reward_statue_ratio`` and ``min_foot_load_share_statue_ratio`` are
judged against a :class:`StatueReference` reduced from a zero-action panel
rolled in the SAME report -- the same env kwargs, the same seeds, the same
recorder -- so they recalibrate themselves when the reward or the plant
changes and add no absolute statue constant for
``test_statue_constant_freshness.py`` to police.  The reference is a
panel-level mean over the statue's full-horizon episodes, not a per-seed
pairing, so one statue fall cannot turn a policy episode unmeasured.  A
declared statue-relative criterion with no reference fails closed, by
name; the in-training screen (``in_training=True``, a later tier) skips
them explicitly and says which it skipped.

The settle window
-----------------
``settle_steps`` splits the episode: the settle-window criteria
(``max_settle_*``) read ``[g, settle_steps)``, after the spawn grace
``g = gait.stance_metrics.spawn_grace_steps(dt)`` (0.1 s: the reset pop is a
plant property, floor map §5.3), and every other criterion reads
``[settle_steps, T)``.  A ``settle_steps`` at or below the grace leaves the
settle window empty and every settle metric NaN, so
:meth:`StanceV2Thresholds.validate_settle_window` and the gate itself refuse
one whenever the control step is known.

The window hop pair
-------------------
A leg is down on a step when it carries load on at least half of the step's
substeps, so the window's support, flight and touchdown criteria read a hop
only when its flight phase lasts half a control step or more.  On a light
plant it can be shorter: the anatomical compsognathus (2 ms substeps, ten to
a 0.02 s step) hops on both feet at 10 Hz with 1-3 airborne substeps a step
and reads both legs down on every step, no touchdown and no flight.  The
optional ``max_window_airborne_substeps`` and
``max_window_peak_floor_force_bw`` read the settle window's own hop and
impact detectors over the window instead (``feet_airborne_substeps`` and
``total_floor_max`` of the trace, summed and maxed over ``[settle_steps,
T)``), so such a hop fails its episode wherever in the episode it starts if
it flies longer, or lands harder, than the declared bars; declared by the
compsognathus since D-D26.  Neither sees a foot lifted for less than half a
step while the other stays down, and a soft enough bounce passes both: on
that plant both certify (KNOWN_ISSUES).  Undeclared, neither is applied,
and neither needs a statue panel.  Both are metrics added later (below).

The pad's centre of pressure
----------------------------
The sole bars read a pad's tilt, its corner lift and its loaded contact
points, and a pad standing on its front edge with its digits unloaded can
meet all three: the physics-r8 seed-44 trex stance holds its left pad 0.7
degrees toe-down on two loaded corners, which ``min_sole_contacts`` 1.5
admits and cannot be raised against (the statue reaches 2.086).  The
optional ``max_sole_cop_fore_aft`` reads where along the pad the floor
pushes: the window mean of the per-step ``|fore-aft CoP| / half-length``
(``StanceEpisodeMetrics.max_sole_cop_fore_aft``: 0 centred, 1 on an edge
or with no loaded contact), which separates that foot (0.85-1.00) from the
statue (at most 0.48).  Box soles only, and two-sided, because the
compsognathus statue stands heel-side.  Declared by trex since D-D28;
undeclared, it is not applied.

Metrics added later
-------------------
A report row or panel CSV recorded before a
:data:`~environments.shared.gait.stance_metrics.STANCE_METRIC_LATER_FIELDS`
metric existed lacks it, and the readers (``StanceEpisodeMetrics.from_row``,
:func:`read_stance_v2_panel`) read it as unmeasured: every earlier report
keeps its verdict under a gate that does not declare the key.  Under one
that does, the episode would fail as unmeasured though nothing failed to
measure it, so such a panel is refused, never failed: publication
(``result_bundle.evidence``), which re-derives from the CSV, refuses it by
name (:func:`unrecorded_criteria`); the judge (``reporting.gates``, and
backfill through it) refuses a report of that age first because the
thresholds it records lack the key, and by name when they match.
Re-rolling the panel on the same handoff pair measures the metric.

This module is pure: the standard library,
:func:`~environments.shared.curriculum.recovery_gate.binomial_lcb` and the
pure numpy row type :class:`~environments.shared.gait.stance_metrics.StanceEpisodeMetrics`.
``reporting`` and ``result_bundle`` both import it -- the post-stage judge
(``reporting.gates``), the report that rolls the panel
(``reporting.stance_report``), publication (``result_bundle.evidence``) and
the backfill tool share this one implementation -- so it must not reach back
into either.
"""

from __future__ import annotations

import csv
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ..gait.stance_metrics import (
    STANCE_METRIC_FIELDS,
    STANCE_METRIC_FOOT_FIELDS,
    STANCE_METRIC_LATER_FIELDS,
    StanceEpisodeMetrics,
    spawn_grace_steps,
)
from .recovery_gate import binomial_lcb

#: The gate kind this module evaluates.  Imported by ``gate_schema`` so the
#: registry and the implementation cannot drift apart.
STANCE_GATE_V2_KIND = "stance_quality/v2"

#: The per-episode criteria: ``(threshold key, StanceEpisodeMetrics field,
#: direction)``.  ``"min"`` is a floor the metric must reach, ``"max"`` a
#: ceiling it must not exceed.  The key names state the bar; the field names
#: are the library's (``max_foot_contact_fraction`` reads the foot-on-foot
#: fraction, ``max_window_displacement_m`` the root's window travel).
EPISODE_CRITERIA: tuple[tuple[str, str, str], ...] = (
    ("min_all_feet_support", "all_feet_support", "min"),
    ("max_touchdown_rate", "touchdown_rate", "max"),
    ("max_window_displacement_m", "window_displacement_m", "max"),
    ("min_foot_load_share", "min_foot_load_share", "min"),
    ("max_actuator_saturation_fraction", "max_actuator_saturation_fraction", "max"),
    ("max_settle_airborne_substeps", "settle_airborne_substeps", "max"),
    ("max_settle_peak_floor_force_bw", "settle_peak_floor_force_bw", "max"),
    ("min_foot_load_share_windowed", "min_foot_load_share_windowed", "min"),
    ("max_foot_contact_fraction", "foot_on_foot_fraction", "max"),
    ("max_phantom_support_fraction", "phantom_support_fraction", "max"),
    ("max_nonfoot_load_fraction", "nonfoot_load_fraction", "max"),
    ("max_settle_stance_width_change_m", "settle_stance_width_change_m", "max"),
    ("min_support_geom_duty", "min_support_geom_duty", "min"),
    ("min_support_geom_coverage", "min_support_geom_coverage", "min"),
    ("max_sole_tilt_deg", "max_sole_tilt_deg", "max"),
    ("max_sole_tilt_excess_deg", "max_sole_tilt_excess_deg", "max"),
    ("max_sole_corner_lift_m", "max_sole_corner_lift_m", "max"),
    ("min_sole_contacts", "min_sole_contacts", "min"),
    ("max_episode_yaw_change_deg", "episode_yaw_change_deg", "max"),
    ("max_settle_touchdowns", "settle_touchdowns", "max"),
    ("max_sole_cop_fore_aft", "max_sole_cop_fore_aft", "max"),
    ("max_window_airborne_substeps", "window_airborne_substeps", "max"),
    ("max_window_peak_floor_force_bw", "window_peak_floor_force_bw", "max"),
)

#: Optional keys whose bar must lie strictly inside ``(low, high)``, ``(key,
#: low, high, why)``: a bar outside is a typo, not a strict gate
#: (:meth:`StanceV2Thresholds.from_curriculum`, and ``gate_schema`` at config
#: load, so the typo stops the run before it trains).
CRITERION_BAR_RANGES: tuple[tuple[str, float, float, str], ...] = (
    (
        "max_sole_cop_fore_aft",
        0.0,
        1.0,
        "it is a fraction of the pad's half-length: 0 refuses every loaded pad, and a pad on its edge reads 1",
    ),
)

#: The window hop pair's floors, ``(key, least valid bar, why)``: a bar below
#: either refuses every episode, so it is a typo, not a strict gate
#: (:meth:`StanceV2Thresholds.from_curriculum`).
WINDOW_HOP_KEY_FLOORS: tuple[tuple[str, float, str], ...] = (
    ("max_window_airborne_substeps", 0.0, "it counts substeps"),
    (
        "max_window_peak_floor_force_bw",
        1.0,
        "a supported window's floor force averages one body weight, so no substep peak sits below it",
    ),
)

#: The criteria whose failure, with an early end (:data:`HORIZON_REASON`),
#: marks an episode as a hop or a fall for the ``max_hop_or_fall_episodes``
#: panel rail: every leg down, debounced touchdowns, root drift and command
#: saturation over the window, and the window hop pair where a gate declares
#: it (D-D26: on a plant whose hop flights are shorter than half a control
#: step, the pair is what reads a hop over the window, so it counts as one).
#: A whole-episode hop fails several of them at once (the seed-44 physics-r8
#: trex hop episodes fail support and displacement); the pad, load-share,
#: phantom, yaw and settle bars are not in it, so the rail does not tighten
#: them.
HOP_OR_FALL_KEYS: tuple[str, ...] = (
    "min_all_feet_support",
    "max_touchdown_rate",
    "max_window_displacement_m",
    "max_actuator_saturation_fraction",
    "max_window_airborne_substeps",
    "max_window_peak_floor_force_bw",
)

#: The keys judged against the statue panel rolled in the same report.
STATUE_RELATIVE_KEYS: tuple[str, ...] = ("min_avg_reward_statue_ratio", "min_foot_load_share_statue_ratio")

#: The ``policy`` a v3 report's statue block records: the zero command (the
#: named home control) every step -- the reference the statue-relative keys
#: are defined against, and the text ``reporting.stance_report`` describes a
#: zero-action panel with.  The judge refuses a statue block that records
#: any other.
STATUE_POLICY = "zero action (do-nothing reference)"

#: Every threshold key the kind requires (``gate_schema._REQUIRED_THRESHOLD_KEYS``).
#: The panel size, the bar and the settle window define the statistic; the
#: seven criteria are the ones every audit hack fails at least one of and
#: every statue clears on all six species (floor map §5.4): support, chatter,
#: drift, load balance, actuation, and the hop and the stomp of the settle
#: window.  The species-specific flatness and coverage keys stay optional,
#: because the right one depends on the foot (a box pad's corner lift on
#: trex, support-geom duty on the padless velociraptor).
STANCE_V2_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "min_eval_episodes",
        "min_clean_stance_lcb",
        "settle_steps",
        "min_all_feet_support",
        "max_touchdown_rate",
        "max_window_displacement_m",
        "min_foot_load_share",
        "max_actuator_saturation_fraction",
        "max_settle_airborne_substeps",
        "max_settle_peak_floor_force_bw",
    }
)

#: Every threshold key the kind consumes (``gate_schema.GATE_KINDS``): the
#: required set, the optional per-episode criteria, the statue-relative
#: ratios, the panel rails, and ``required_consecutive`` -- allowed, like every
#: advancing kind's copy of it (decision D-B3), as the hysteresis of the
#: in-training screen a later tier adds; the manager refuses the kind today.
STANCE_V2_THRESHOLD_KEYS: frozenset[str] = frozenset(
    STANCE_V2_REQUIRED_KEYS
    | {key for key, _, _ in EPISODE_CRITERIA}
    | set(STATUE_RELATIVE_KEYS)
    | {"min_full_horizon_fraction", "min_avg_reward", "max_hop_or_fall_episodes", "required_consecutive"}
)

#: The criteria an episode is scored on beyond the horizon, in report order:
#: the per-episode table above, then the statue-relative share ratio.
EPISODE_CRITERION_KEYS: tuple[str, ...] = (
    *(key for key, _, _ in EPISODE_CRITERIA),
    "min_foot_load_share_statue_ratio",
)

#: The reason key of an episode that did not reach the horizon.
HORIZON_REASON = "horizon"

#: The schema of the stance gate report a v2 panel is scored into
#: (``reporting.stance_report``).  Defined here, beside the panel CSV
#: contract below, because the judge and the backfill tool check it without
#: importing the report module (which pulls the species registry); the
#: report's own revision, distinct from the v1 report's
#: ``mesozoic.stance-gate-report/v2`` so "report v2" is never "kind v2".
STANCE_V2_REPORT_SCHEMA = "mesozoic.stance-gate-report/v3"


def _finite_number(value: Any, *, key: str) -> float:
    """*value* as a finite float, or ``ValueError`` naming *key*.

    A real NUMBER only: a string that would parse (``"0.98"``) is refused,
    because the gate digest hashes a string threshold verbatim and
    ``gate_schema.same_threshold`` would then never match it against the
    number a report records -- every v2 key must be numeric in the TOML.
    """
    if value is None or isinstance(value, bool):
        raise ValueError(f"{STANCE_GATE_V2_KIND} threshold {key} is missing or not numeric ({value!r})")
    if not isinstance(value, numbers.Real):
        raise ValueError(f"{STANCE_GATE_V2_KIND} threshold {key} is not numeric ({value!r})")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{STANCE_GATE_V2_KIND} threshold {key} must be finite, not {value!r}")
    return number


def _whole_number(value: float, *, key: str, minimum: int) -> int:
    if value != int(value) or value < minimum:
        raise ValueError(
            f"{STANCE_GATE_V2_KIND} threshold {key} must be a whole number of at least {minimum}, not {value!r}"
        )
    return int(value)


@dataclass(frozen=True)
class StanceV2Thresholds:
    """The ``stance_quality/v2`` criteria, as a stage's ``[curriculum]`` declares them.

    The required fields have no default: :meth:`from_curriculum` refuses a
    block without them rather than inventing a bar.  An optional criterion
    is ``None`` when undeclared and is then not applied at all -- never a
    permissive ``+inf`` ceiling, which would read as "declared, no limit".
    """

    min_eval_episodes: int
    min_clean_stance_lcb: float
    settle_steps: int
    min_all_feet_support: float
    max_touchdown_rate: float
    max_window_displacement_m: float
    min_foot_load_share: float
    max_actuator_saturation_fraction: float
    max_settle_airborne_substeps: float
    max_settle_peak_floor_force_bw: float
    min_full_horizon_fraction: float | None = None
    min_avg_reward: float | None = None
    min_avg_reward_statue_ratio: float | None = None
    min_foot_load_share_statue_ratio: float | None = None
    min_foot_load_share_windowed: float | None = None
    max_foot_contact_fraction: float | None = None
    max_phantom_support_fraction: float | None = None
    max_nonfoot_load_fraction: float | None = None
    max_settle_stance_width_change_m: float | None = None
    min_support_geom_duty: float | None = None
    min_support_geom_coverage: float | None = None
    max_sole_tilt_deg: float | None = None
    max_sole_tilt_excess_deg: float | None = None
    max_sole_corner_lift_m: float | None = None
    min_sole_contacts: float | None = None
    max_episode_yaw_change_deg: float | None = None
    max_settle_touchdowns: float | None = None
    max_sole_cop_fore_aft: float | None = None
    max_hop_or_fall_episodes: int | None = None
    max_window_airborne_substeps: float | None = None
    max_window_peak_floor_force_bw: float | None = None
    required_consecutive: int = 3

    @classmethod
    def from_curriculum(cls, curriculum: Mapping[str, Any]) -> StanceV2Thresholds:
        """Read the thresholds a stage's ``[curriculum]`` table declares.

        Strict, like ``TaskSuccessGateThresholds``: a required key that is
        missing, non-numeric or non-finite raises ``ValueError`` naming it,
        and so does a DECLARED optional key that is not a finite number -- a
        criterion the reader silently dropped would be the disabled-by-typo
        gate the schema exists to prevent.  ``min_eval_episodes`` must be a
        positive whole number, ``settle_steps`` a non-negative one, and
        ``min_clean_stance_lcb`` strictly inside (0, 1): a bar of 0 is met by
        ``binomial_lcb(0, n) = 0``, a panel with no clean episode, and no
        panel of 40 bounds above 0.928, so a bar at or above 1 is a typo
        that would refuse every policy; :data:`CRITERION_BAR_RANGES` bounds
        the keys that read a fraction the same way, and
        :data:`WINDOW_HOP_KEY_FLOORS` the window hop pair from below:
        ``max_window_airborne_substeps`` must not be negative, and
        ``max_window_peak_floor_force_bw`` must be at least 1, since a
        supported window's floor force averages the body weight, so its
        substep peak cannot sit below it.  Keys outside the kind (schedule,
        diagnostic, publication) are ignored; ``gate_schema`` validates them.
        """
        values: dict[str, Any] = {}
        for key in sorted(STANCE_V2_REQUIRED_KEYS):
            values[key] = _finite_number(curriculum.get(key), key=key)
        for key in sorted(STANCE_V2_THRESHOLD_KEYS - STANCE_V2_REQUIRED_KEYS):
            if key in curriculum:
                values[key] = _finite_number(curriculum[key], key=key)
        values["min_eval_episodes"] = _whole_number(values["min_eval_episodes"], key="min_eval_episodes", minimum=1)
        values["settle_steps"] = _whole_number(values["settle_steps"], key="settle_steps", minimum=0)
        if "required_consecutive" in values:
            values["required_consecutive"] = _whole_number(
                values["required_consecutive"], key="required_consecutive", minimum=1
            )
        if "max_hop_or_fall_episodes" in values:
            values["max_hop_or_fall_episodes"] = _whole_number(
                values["max_hop_or_fall_episodes"], key="max_hop_or_fall_episodes", minimum=0
            )
        for key, low, high, why in CRITERION_BAR_RANGES:
            if key in values and not low < values[key] < high:
                raise ValueError(
                    f"{STANCE_GATE_V2_KIND} threshold {key} must lie strictly between {low:g} and {high:g}, not "
                    f"{values[key]!r}: {why}"
                )
        for key, floor, why in WINDOW_HOP_KEY_FLOORS:
            if key in values and values[key] < floor:
                raise ValueError(
                    f"{STANCE_GATE_V2_KIND} threshold {key} must be at least {floor:g}, not {values[key]!r}: {why}"
                )
        lcb = values["min_clean_stance_lcb"]
        if not 0.0 < lcb < 1.0:
            raise ValueError(
                f"{STANCE_GATE_V2_KIND} threshold min_clean_stance_lcb must lie strictly between 0 and 1, not "
                f"{lcb!r}: 0 certifies a panel with no clean episode, and no finite panel bounds at 1"
            )
        return cls(**values)

    def declared(self) -> dict[str, float]:
        """The declared criteria by key (the required ones and every optional one not ``None``)."""
        return {
            spec.name: getattr(self, spec.name)
            for spec in fields(self)
            if spec.name != "required_consecutive" and getattr(self, spec.name) is not None
        }

    def declares_statue_criteria(self) -> bool:
        """Whether a statue panel must be rolled to judge this gate."""
        return any(getattr(self, key) is not None for key in STATUE_RELATIVE_KEYS)

    def validate_settle_window(self, control_dt: float) -> None:
        """Refuse a ``settle_steps`` that leaves no settle window after the spawn grace at *control_dt*.

        At or below :func:`~environments.shared.gait.stance_metrics.spawn_grace_steps`
        (10 steps at dt 0.01, 5 at 0.02) every settle metric is NaN, so
        every episode would fail ``max_settle_airborne_substeps`` and
        ``max_settle_peak_floor_force_bw`` as unmeasured -- a gate nothing
        can pass, declared by mistake.  Raises ``ValueError``.
        """
        grace = spawn_grace_steps(float(control_dt))
        if self.settle_steps <= grace:
            raise ValueError(
                f"{STANCE_GATE_V2_KIND} settle_steps {self.settle_steps} leaves no settle window after the "
                f"{grace}-step spawn grace at control dt {float(control_dt):g} s, so every settle metric "
                "would be unmeasured"
            )


# A dataclass field added to the thresholds without a schema key (or the
# reverse) would be read by nothing or validated by nothing; make the drift
# unmissable at import.  A raise, not an assert, for the reason gate_schema
# gives: `python -O` strips asserts.
if {spec.name for spec in fields(StanceV2Thresholds)} != STANCE_V2_THRESHOLD_KEYS:
    raise RuntimeError("StanceV2Thresholds fields must be exactly STANCE_V2_THRESHOLD_KEYS")


@dataclass(frozen=True)
class StatueReference:
    """The zero-action panel the statue-relative criteria are judged against.

    ``mean_reward`` and ``foot_load_share`` (per foot, in the recorder's
    ``_foot_sensor_groups`` order) are means over the statue's FULL-HORIZON
    episodes only, and NaN when it had none -- which then fails every
    statue-relative criterion as unmeasured rather than dividing by a fall.
    """

    n_episodes: int
    n_full_horizon: int
    mean_reward: float
    foot_load_share: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "n_full_horizon": self.n_full_horizon,
            "mean_reward": self.mean_reward,
            "foot_load_share": list(self.foot_load_share),
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> StatueReference:
        """Rebuild from :meth:`as_dict` output, read back from JSON (``null`` is NaN).

        Raises ``ValueError`` on a missing or malformed field: a reference
        that cannot be read proves nothing about the statue.
        """
        try:
            shares = mapping["foot_load_share"]
            if isinstance(shares, (str, bytes)) or not isinstance(shares, Sequence):
                raise TypeError(f"foot_load_share is not a list ({shares!r})")
            return cls(
                n_episodes=int(mapping["n_episodes"]),
                n_full_horizon=int(mapping["n_full_horizon"]),
                mean_reward=_nan_if_none(mapping["mean_reward"]),
                foot_load_share=tuple(_nan_if_none(value) for value in shares),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"the statue reference is unreadable: {exc}") from exc


def _nan_if_none(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        raise TypeError(f"{value!r} is a boolean, not a number")
    return float(value)


def _mean(values: Sequence[float]) -> float:
    """The mean, NaN for no values or when any is not finite (never a mean over the finite subset)."""
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return math.fsum(values) / len(values)


def statue_reference(episodes: Iterable[StanceEpisodeMetrics], *, horizon: int) -> StatueReference:
    """Reduce a zero-action panel's metrics to the reference the statue-relative criteria divide by.

    Full-horizon episodes only (``length >= horizon``).  A foot whose share
    is unmeasured on any of them, or a panel whose episodes disagree on the
    number of feet, gives NaN shares: the ratio is then unmeasured, never
    computed against a partial statue.
    """
    rows = list(episodes)
    full = [row for row in rows if row.length >= horizon]
    feet = {len(row.foot_load_share) for row in full}
    if len(feet) == 1:
        n_feet = feet.pop()
        shares = tuple(_mean([row.foot_load_share[foot] for row in full]) for foot in range(n_feet))
    else:
        shares = tuple(math.nan for _ in range(max(feet, default=0)))
    return StatueReference(
        n_episodes=len(rows),
        n_full_horizon=len(full),
        mean_reward=_mean([row.reward for row in full]),
        foot_load_share=shares,
    )


def _criterion_failure(key: str, metric: str, direction: str, value: float, bar: float) -> str | None:
    """The reason *value* fails the bar, or ``None`` when it holds (NaN never holds)."""
    if not math.isfinite(value):
        return f"{key}: {metric} is unmeasured ({value!r})"
    if direction == "min" and value < bar:
        return f"{key}: {metric} {value:.6g} < {bar:.6g}"
    if direction == "max" and value > bar:
        return f"{key}: {metric} {value:.6g} > {bar:.6g}"
    return None


def _statue_share_failure(episode: StanceEpisodeMetrics, bar: float, statue: StatueReference | None) -> str | None:
    """The ``min_foot_load_share_statue_ratio`` reason for *episode*, or ``None`` when it holds."""
    key = "min_foot_load_share_statue_ratio"
    if statue is None:
        return f"{key}: no statue panel was rolled, so the ratio to the statue's share is unmeasured"
    policy, reference = episode.foot_load_share, statue.foot_load_share
    if not reference or len(policy) != len(reference):
        return f"{key}: the episode measures {len(policy)} feet and the statue reference {len(reference)}"
    ratios: list[float] = []
    for foot, (share, statue_share) in enumerate(zip(policy, reference)):
        if not math.isfinite(statue_share) or statue_share <= 0.0:
            return f"{key}: the statue's share on foot {foot} is {statue_share!r}, which no ratio can divide by"
        ratios.append(share / statue_share)
    # Checked before min(): Python's min() keeps a NaN only when it comes
    # first (every comparison with NaN is False), so ``(0.5, nan)`` would
    # otherwise read as the finite 0.5 / share and pass.
    unmeasured = next((ratio for ratio in ratios if not math.isfinite(ratio)), None)
    if unmeasured is not None:
        return _criterion_failure(key, "min foot share / statue share", "min", unmeasured, bar)
    return _criterion_failure(key, "min foot share / statue share", "min", min(ratios), bar)


def classify_stance_episode(
    episode: StanceEpisodeMetrics,
    thresholds: StanceV2Thresholds,
    *,
    horizon: int,
    statue: StatueReference | None = None,
    apply_statue_criteria: bool = True,
) -> tuple[str, ...]:
    """Every reason *episode* is not clean, in :data:`EPISODE_CRITERION_KEYS` order; empty when it is clean.

    Each reason starts ``"<key>: "`` (``"horizon: "`` for an episode that
    ended early), so a reader -- and :func:`evaluate_stance_v2_gate`'s
    per-criterion counts -- can tell which bar failed.  Only declared
    criteria are applied.  ``apply_statue_criteria=False`` (the in-training
    screen) skips the statue-relative ratio instead of failing it.
    """
    reasons: list[str] = []
    if episode.length < horizon:
        reasons.append(f"{HORIZON_REASON}: length {episode.length} < {horizon}")
    for key, metric, direction in EPISODE_CRITERIA:
        bar = getattr(thresholds, key)
        if bar is None:
            continue
        failure = _criterion_failure(key, metric, direction, float(getattr(episode, metric)), float(bar))
        if failure is not None:
            reasons.append(failure)
    share_bar = thresholds.min_foot_load_share_statue_ratio
    if share_bar is not None and apply_statue_criteria:
        failure = _statue_share_failure(episode, float(share_bar), statue)
        if failure is not None:
            reasons.append(failure)
    return tuple(reasons)


def unrecorded_criteria(thresholds: StanceV2Thresholds, recorded: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """The declared criteria whose metric a panel recorded before it existed lacks: ``(key, metric)`` pairs.

    *recorded* is the panel's metric names (a CSV header, a report row's
    keys).  Only a :data:`~environments.shared.gait.stance_metrics.STANCE_METRIC_LATER_FIELDS`
    metric can be absent from a readable panel; such a panel was never
    measured on the criterion, so the judge and publication refuse it by
    name -- re-roll it -- instead of failing every episode as unmeasured.
    """
    present = set(recorded)
    return tuple(
        (key, metric)
        for key, metric, _ in EPISODE_CRITERIA
        if metric in STANCE_METRIC_LATER_FIELDS and metric not in present and getattr(thresholds, key) is not None
    )


def reason_key(reason: str) -> str:
    """The criterion key a :func:`classify_stance_episode` reason names."""
    return reason.split(":", 1)[0]


def is_hop_or_fall(reasons: Iterable[str]) -> bool:
    """Whether an episode's reasons mark a hop or a fall: an early end or a :data:`HOP_OR_FALL_KEYS` failure."""
    return any(reason_key(reason) in {HORIZON_REASON, *HOP_OR_FALL_KEYS} for reason in reasons)


@dataclass(frozen=True)
class StanceV2Result:
    """One panel's verdict and the numbers it was reached on.

    ``episode_reasons`` is positionally aligned with the episodes judged;
    ``criterion_failures`` counts, per criterion (``"horizon"`` first, then
    :data:`EPISODE_CRITERION_KEYS` order, declared ones only), the episodes
    that failed it -- an unclean episode usually fails several, so these do
    not sum to ``n_episodes - n_clean``.  ``skipped_criteria`` names the
    declared criteria this evaluation did not apply (the statue-relative
    ones under ``in_training``).
    """

    passed: bool
    failures: tuple[str, ...]
    n_episodes: int
    n_clean: int
    clean_fraction: float
    clean_lcb: float
    n_full_horizon: int
    full_horizon_fraction: float
    mean_reward: float
    statue_mean_reward: float | None
    episode_reasons: tuple[tuple[str, ...], ...] = field(default=())
    criterion_failures: tuple[tuple[str, int], ...] = field(default=())
    skipped_criteria: tuple[str, ...] = field(default=())
    hop_or_fall_episodes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """The panel-level numbers, JSON-ready (the per-episode reasons travel with the episode rows).

        ``hop_or_fall_episodes`` appears only when the gate declares
        ``max_hop_or_fall_episodes``, so a panel judged without the rail
        serialises exactly as it did before the rail existed.
        """
        payload: dict[str, Any] = {
            "passed": self.passed,
            "failures": list(self.failures),
            "n_episodes": self.n_episodes,
            "n_clean": self.n_clean,
            "clean_fraction": self.clean_fraction,
            "clean_lcb": self.clean_lcb,
            "n_full_horizon": self.n_full_horizon,
            "full_horizon_fraction": self.full_horizon_fraction,
            "mean_reward": self.mean_reward,
            "statue_mean_reward": self.statue_mean_reward,
            "criterion_failures": dict(self.criterion_failures),
            "skipped_criteria": list(self.skipped_criteria),
        }
        if self.hop_or_fall_episodes is not None:
            payload["hop_or_fall_episodes"] = self.hop_or_fall_episodes
        return payload


def evaluate_stance_v2_gate(
    episodes: Iterable[StanceEpisodeMetrics],
    thresholds: StanceV2Thresholds,
    *,
    horizon: int,
    statue: StatueReference | None = None,
    control_dt: float | None = None,
    in_training: bool = False,
) -> StanceV2Result:
    """Judge one panel's per-episode metrics against the gate, naming every failure.

    ``horizon`` is the stage's ``max_episode_steps``: an episode is clean
    only if it reached it.  ``statue`` is the reference the statue-relative
    criteria divide by; ``None`` with such a criterion declared fails it,
    by name.  ``control_dt`` (the stage's ``timestep * frame_skip``), when
    known, lets the gate refuse a settle window inside the spawn grace
    (:meth:`StanceV2Thresholds.validate_settle_window`).  ``in_training=True``
    skips the statue-relative criteria and lists them in
    ``skipped_criteria``; it is for a screen, never for a certificate.

    Fails closed throughout: a panel smaller than ``min_eval_episodes``, an
    episode measured under another ``settle_steps``, a NaN bound or rail, a
    statue reference that cannot be divided by, and a non-positive horizon
    are failures, never passes.
    """
    rows = list(episodes)
    failures: list[str] = []
    n = len(rows)
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        failures.append(f"horizon {horizon!r} is not a positive whole number of steps, so no episode can reach it")
    if control_dt is not None:
        try:
            thresholds.validate_settle_window(control_dt)
        except ValueError as exc:
            failures.append(str(exc))
    misaligned = [index for index, row in enumerate(rows) if row.settle_steps != thresholds.settle_steps]
    if misaligned:
        shown = ", ".join(f"episode {index} at {rows[index].settle_steps}" for index in misaligned[:3])
        failures.append(
            f"{len(misaligned)} of {n} episodes were measured with another settle window than the declared "
            f"settle_steps {thresholds.settle_steps} ({shown}{', ...' if len(misaligned) > 3 else ''}); their "
            "metrics answer a different question"
        )

    apply_statue = not in_training
    skipped = tuple(key for key in STATUE_RELATIVE_KEYS if getattr(thresholds, key) is not None and in_training)
    reasons = tuple(
        classify_stance_episode(
            row, thresholds, horizon=int(horizon), statue=statue, apply_statue_criteria=apply_statue
        )
        for row in rows
    )
    k = sum(1 for reason in reasons if not reason)

    if n < thresholds.min_eval_episodes:
        failures.append(
            f"n_episodes {n} < min_eval_episodes {thresholds.min_eval_episodes} (the bound's power is "
            "specified at this panel size)"
        )
    clean_lcb = binomial_lcb(k, n) if n else math.nan
    if not math.isfinite(clean_lcb) or clean_lcb < thresholds.min_clean_stance_lcb:
        failures.append(
            f"clean_stance_lcb {clean_lcb:.4f} < {thresholds.min_clean_stance_lcb:.4f} ({k}/{n} episodes clean)"
        )

    n_full = sum(1 for row in rows if row.length >= horizon)
    full_fraction = n_full / n if n else math.nan
    if thresholds.min_full_horizon_fraction is not None and not (
        math.isfinite(full_fraction) and full_fraction >= thresholds.min_full_horizon_fraction
    ):
        failures.append(f"full_horizon_fraction {full_fraction:.4f} < {thresholds.min_full_horizon_fraction:.4f}")

    hop_or_fall: int | None = None
    if thresholds.max_hop_or_fall_episodes is not None:
        hop_or_fall = sum(1 for episode_reasons in reasons if is_hop_or_fall(episode_reasons))
        if hop_or_fall > thresholds.max_hop_or_fall_episodes:
            failures.append(
                f"hop_or_fall_episodes {hop_or_fall} > {thresholds.max_hop_or_fall_episodes} (episodes ending early "
                f"or failing {', '.join(HOP_OR_FALL_KEYS)}; panel rail)"
            )

    mean_reward = _mean([row.reward for row in rows])
    if thresholds.min_avg_reward is not None and not (
        math.isfinite(mean_reward) and mean_reward >= thresholds.min_avg_reward
    ):
        failures.append(f"mean_reward {mean_reward:.1f} < {thresholds.min_avg_reward:.1f} (rail)")

    ratio = thresholds.min_avg_reward_statue_ratio
    statue_mean = None if statue is None else statue.mean_reward
    if ratio is not None and apply_statue:
        if statue is None:
            failures.append(
                "min_avg_reward_statue_ratio is declared but no statue panel was rolled, so the statue-relative "
                "reward rail is unmeasured"
            )
        elif not math.isfinite(statue.mean_reward) or statue.mean_reward <= 0.0:
            failures.append(
                f"the statue's full-horizon mean reward is {statue.mean_reward!r} over {statue.n_full_horizon} "
                "episodes, so min_avg_reward_statue_ratio has no positive reference to scale"
            )
        elif not (math.isfinite(mean_reward) and mean_reward >= ratio * statue.mean_reward):
            failures.append(
                f"mean_reward {mean_reward:.1f} < {ratio:g} x statue {statue.mean_reward:.1f} = "
                f"{ratio * statue.mean_reward:.1f} (statue-relative rail)"
            )

    counted = [HORIZON_REASON, *(key for key in EPISODE_CRITERION_KEYS if getattr(thresholds, key) is not None)]
    tally = {key: 0 for key in counted}
    for episode_reasons in reasons:
        for key in {reason_key(reason) for reason in episode_reasons}:
            tally[key] = tally.get(key, 0) + 1

    return StanceV2Result(
        passed=not failures,
        failures=tuple(failures),
        n_episodes=n,
        n_clean=k,
        clean_fraction=k / n if n else math.nan,
        clean_lcb=clean_lcb,
        n_full_horizon=n_full,
        full_horizon_fraction=full_fraction,
        mean_reward=mean_reward,
        statue_mean_reward=statue_mean,
        episode_reasons=reasons,
        criterion_failures=tuple(tally.items()),
        skipped_criteria=skipped,
        hop_or_fall_episodes=hop_or_fall,
    )


# ── the per-episode panel evidence (``stance_panel_selected.csv``) ──────────

#: Columns every row stamps with one value for the whole panel: the handoff
#: pair the panel was rolled on, the task it was rolled under (the report's
#: ``task_sha256``), the measurement definition, and the statue reference.
#: Each must be single-valued across a file -- rows that disagree are
#: evidence for no one panel.  ``measurement_definition_sha256`` is the
#: digest of the manifest's model-free part
#: (``gait.constants.measurement_definition_sha256``), the one publication
#: can recompute without a model.
STANCE_V2_PANEL_HANDOFF_COLUMNS: tuple[str, ...] = ("checkpoint_sha256", "normalization_sha256")
STANCE_V2_PANEL_TASK_COLUMNS: tuple[str, ...] = ("task_sha256",)
STANCE_V2_PANEL_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "measurement_version",
    "measurement_sha256",
    "measurement_definition_sha256",
)
STANCE_V2_PANEL_STATUE_COLUMNS: tuple[str, ...] = (
    "statue_n_episodes",
    "statue_n_full_horizon",
    "statue_mean_reward",
    "statue_foot_load_share",
)

#: Column order of a ``stance_quality/v2`` panel's ``stance_panel_selected.csv``.
#: v1's columns first (its touch duties stay, report-only, so the two panels
#: read side by side), then the classification, every
#: :data:`~environments.shared.gait.stance_metrics.STANCE_METRIC_FIELDS` scalar
#: and per-foot field (``StanceEpisodeMetrics.from_row`` reads a row back
#: exactly), and the panel-wide stamps.  Separate from v1's
#: ``STANCE_PANEL_FIELDNAMES`` on purpose, and each publication reader
#: refuses the other's file: :func:`read_stance_v2_panel` on the missing v2
#: columns, the v1 reader (``result_bundle.evidence``) on the
#: ``measurement_version`` column v1's lack.  The second needs saying: v1's
#: columns are a SUBSET of these (``length`` and ``reward`` are metric fields
#: here), so a reader that checked only for its own columns would score a v2
#: file as a v1 panel.
STANCE_V2_PANEL_FIELDNAMES: tuple[str, ...] = (
    "episode",
    "panel_seed",
    "reached_horizon",
    "unsupported_duty",
    "bilateral_support_duty",
    "single_support_duty",
    "clean",
    "reasons",
    *STANCE_METRIC_FIELDS,
    *STANCE_METRIC_FOOT_FIELDS,
    *STANCE_V2_PANEL_HANDOFF_COLUMNS,
    *STANCE_V2_PANEL_TASK_COLUMNS,
    *STANCE_V2_PANEL_MEASUREMENT_COLUMNS,
    *STANCE_V2_PANEL_STATUE_COLUMNS,
)


@dataclass(frozen=True)
class StanceV2PanelEvidence:
    """One v2 panel CSV read back: the episodes, their seeds and the panel-wide stamps.

    ``recorded_clean`` / ``reached_horizon`` are the writer's own columns,
    kept so a consumer can refuse a file whose recorded classification
    disagrees with the one it re-derives.  A stamp is ``None`` when its
    column is blank on every row; ``statue`` is ``None`` when the panel was
    scored without a statue reference.  ``absent_metrics`` names the
    :data:`~environments.shared.gait.stance_metrics.STANCE_METRIC_LATER_FIELDS`
    columns the file lacks (read as unmeasured), in field order.
    """

    episodes: list[StanceEpisodeMetrics]
    panel_seeds: list[int]
    recorded_clean: list[bool]
    reached_horizon: list[bool]
    checkpoint_sha256: str | None
    normalization_sha256: str | None
    task_sha256: str | None
    measurement_version: str | None
    measurement_sha256: str | None
    measurement_definition_sha256: str | None
    statue: StatueReference | None
    absent_metrics: tuple[str, ...] = ()


def _cell_bool(value: str | None, *, column: str, index: int, path: Path) -> bool:
    text = (value or "").strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{path} episode {index} has an unparsable {column} cell {value!r}")


def _single_value(rows: list[dict[str, str]], column: str, *, path: Path) -> str | None:
    values = {(row.get(column) or "").strip() for row in rows}
    if len(values) != 1:
        raise ValueError(f"{path} mixes {column} values across its rows: {sorted(values)}")
    value = values.pop()
    return value or None


def read_stance_v2_panel(csv_path: "str | Path") -> StanceV2PanelEvidence:
    """Read a ``stance_quality/v2`` panel's ``stance_panel_selected.csv`` back into gate inputs.

    Strict, because every consumer -- publication above all -- re-derives a
    verdict from what this returns: a file with no rows, a column of
    :data:`STANCE_V2_PANEL_FIELDNAMES` missing (a v1 panel file among them),
    an unparsable metric, seed or flag cell, a panel-wide stamp that differs
    between rows, or a statue stamp that is present on some columns and not
    others is a ``ValueError``.  An empty metric cell is NaN -- unmeasured,
    which fails its criterion -- never zero.  The one tolerated missing
    column is a :data:`~environments.shared.gait.stance_metrics.STANCE_METRIC_LATER_FIELDS`
    metric, which a panel recorded before it existed lacks: it reads as
    unmeasured, like an empty cell, and ``absent_metrics`` names it so a
    consumer can refuse the panel under a gate that declares its key
    (:func:`unrecorded_criteria`).
    """
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [
        column
        for column in STANCE_V2_PANEL_FIELDNAMES
        if column not in columns and column not in STANCE_METRIC_LATER_FIELDS
    ]
    if missing:
        raise ValueError(f"{path} is not a {STANCE_GATE_V2_KIND} panel: it lacks the column(s) {missing}")
    if not rows:
        raise ValueError(f"{path} holds no episode rows")
    episodes: list[StanceEpisodeMetrics] = []
    seeds: list[int] = []
    clean: list[bool] = []
    reached: list[bool] = []
    for index, row in enumerate(rows):
        try:
            episodes.append(StanceEpisodeMetrics.from_row(row))
        except ValueError as exc:
            raise ValueError(f"{path} episode {index}: {exc}") from exc
        seed_text = (row.get("panel_seed") or "").strip()
        if not seed_text.isdigit():
            raise ValueError(f"{path} episode {index} has an unparsable panel_seed cell {seed_text!r}")
        seeds.append(int(seed_text))
        clean.append(_cell_bool(row.get("clean"), column="clean", index=index, path=path))
        reached.append(_cell_bool(row.get("reached_horizon"), column="reached_horizon", index=index, path=path))
    stamps = {
        column: _single_value(rows, column, path=path)
        for column in (
            *STANCE_V2_PANEL_HANDOFF_COLUMNS,
            *STANCE_V2_PANEL_TASK_COLUMNS,
            *STANCE_V2_PANEL_MEASUREMENT_COLUMNS,
            *STANCE_V2_PANEL_STATUE_COLUMNS,
        )
    }
    statue_cells = [stamps[column] for column in STANCE_V2_PANEL_STATUE_COLUMNS]
    statue: StatueReference | None = None
    if any(cell is not None for cell in statue_cells):
        if any(cell is None for cell in statue_cells[:2]):
            raise ValueError(
                f"{path} records a partial statue reference ({dict(zip(STANCE_V2_PANEL_STATUE_COLUMNS, statue_cells))})"
            )
        shares_text = (stamps["statue_foot_load_share"] or "").strip()
        if shares_text[:1] in "[(" and shares_text[-1:] in "])":
            shares_text = shares_text[1:-1]
        statue = StatueReference.from_dict(
            {
                "n_episodes": stamps["statue_n_episodes"],
                "n_full_horizon": stamps["statue_n_full_horizon"],
                "mean_reward": stamps["statue_mean_reward"] or None,
                "foot_load_share": [item for item in shares_text.split(",") if item.strip()],
            }
        )
    return StanceV2PanelEvidence(
        episodes=episodes,
        panel_seeds=seeds,
        recorded_clean=clean,
        reached_horizon=reached,
        checkpoint_sha256=stamps["checkpoint_sha256"],
        normalization_sha256=stamps["normalization_sha256"],
        task_sha256=stamps["task_sha256"],
        measurement_version=stamps["measurement_version"],
        measurement_sha256=stamps["measurement_sha256"],
        measurement_definition_sha256=stamps["measurement_definition_sha256"],
        statue=statue,
        absent_metrics=tuple(name for name in STANCE_METRIC_FIELDS if name not in columns),
    )
