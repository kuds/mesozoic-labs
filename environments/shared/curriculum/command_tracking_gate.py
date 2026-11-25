"""The command-tracking gate statistic: ``command_tracking/v1`` (walk and follow profiles).

What it certifies, and why
--------------------------
A stage whose body-relative command is live (a ``[commands]`` table,
decision D-D60) is judged on whether the animal does what the command asks,
on a fixed panel of seeded episodes rolled with deterministic actions.  The
walk node's ``reward_and_length/v1`` floor on mean forward speed cannot do
that once a walk trains whole-episode stand commands: a stand episode
dilutes the mean, and a mean says nothing about heading or stopping.  The
behavior certificate (``behavior_certification.judge_behavior_panel``) has
the per-event statistics, but it gates no pipeline step and reads its bars
from ``configs/behavior_certification.toml`` at judge time.  This kind takes
those statistics into a manifest gate: every bar is a threshold key of the
stage's ``[curriculum]`` table, so the gate digest covers it, and nothing is
read from the certificate's file when a stage is judged.

One kind, two profiles, selected by the ``profile`` threshold key:

* ``walk`` judges the walk node's narrow distribution (a forward cruise with
  a small initial heading offset, no mid-episode switch, and whole-episode
  stand commands drawn at reset).  Every episode is in one FAMILY by the
  command its reset draws: ``cruise`` (a non-zero speed) or ``stand`` (an
  exactly zero command).
* ``follow`` judges the later follow nodes' widened distribution with the
  behavior certificate's per-episode statistics, plus an optional paired
  comparison against a reference policy on the same seeds.

The walk profile
----------------
A CRUISE episode succeeds when all of these hold:

* it reaches the horizon (no fall, no termination);
* its first command event settles: the first in-tolerance run starts within
  ``SETTLE_BASE_S + |initial heading error| / yaw_rate_max`` and lasts
  ``DWELL_S`` (``behavior_evaluation.summarize_episode``'s event logic, with
  the episode's own initial heading error measured against the spawn
  heading);
* its in-tolerance fraction over the steps after their event's settling
  allowance reaches ``min_cruise_tracking_fraction``;
* its mean speed along the desired heading over the active steps after
  ``SPEED_SETTLE_S``, divided by the mean requested speed over the same
  steps, reaches ``min_cruise_speed_ratio``;
* it passes the gait clause (below).

The cruise family passes when the exact one-sided 95% Clopper-Pearson lower
bound (:func:`~.recovery_gate.binomial_lcb`) on the episodes that reached
the horizon clears ``min_cruise_survival_lcb`` and the bound on successful
episodes clears ``min_cruise_success_lcb``, the certificate's two bounds.

A STAND episode succeeds when it reaches the horizon, its stop event settles
(the same event logic: allowance ``SETTLE_BASE_S``, a stop has no heading
error) and its in-tolerance fraction over the steps after ``STAND_SETTLE_S``
reaches ``min_stand_hold_fraction``.  At a zero command the env's
in-tolerance test is exactly the stop criterion: speed at most the stop-speed
tolerance and ``|yaw rate|`` at most the stop yaw-rate tolerance
(``direction_commands.tracking_metrics``).  The stand family passes when at
most ``max_stand_falls`` of its episodes end before the horizon and the
fraction of successful episodes reaches ``min_stand_success_fraction`` (the
certificate judges its stop commands on a fraction, not a bound).

The gate fails closed when either family holds fewer than its minimum
(``min_cruise_episodes``, ``min_stand_episodes``), and when any episode
plans more than one command event within the horizon: a walk command is a
whole-episode command, and a schedule that switches mid-episode is judged by
the follow profile.  The panel runner
(:mod:`environments.shared.command_tracking_panel`) picks the panel's reset
seeds so that each family holds exactly its minimum, by construction.

The follow profile
------------------
The behavior certificate's statistics for one flat family, read off the same
``summarize_episode`` output: per episode, survival (the horizon reached and
no fall), the whole-episode in-tolerance fraction, the settled fraction of
the eligible events, of the eligible stops and of the eligible turns (an
active event whose requested heading moved more than
:data:`TURN_HEADING_CHANGE_RAD` from the previous event's), and, when
declared, the radial course fraction.  An episode succeeds when it survives
and clears every bar (a stop or turn bar only when the stage's schedule
draws stops or turns and the episode was exposed to one) and the gait
clause (below).  The panel passes
when it holds ``min_follow_episodes`` episodes of a horizon of at least
``min_follow_horizon_s``, the survival and success bounds clear their bars,
and, for each of stops and turns the schedule draws, at least
``min_follow_command_episodes`` episodes were exposed and the fraction of
them that survived and settled theirs reaches the bar.  With
``min_follow_reference_delta_lcb`` declared, a reference policy (the walker
the node continues) is rolled on the same seeds and the one-sided 95% lower
bound on the mean per-seed success difference (policy minus reference,
:func:`~.recovery_gate.paired_difference_lcb`) must clear the bar.  A
declared comparison without its reference panel fails closed.  The pairing
is on whole-episode FOLLOW success on the follow panel, so the statistic is
a non-inferiority bound on the follow task (the follower is about as good as
the walker at following, or better), not a pure test that walking was not
forgotten: the walker fails the turn and stop episodes, which raises the
difference, and gains on turns can hide a loss of straight walking.  For
the strict reading, a follow node also rolls both policies on the walk panel
(or on the follow panel's straight, walk-command episodes) and pairs them on
walk success through :func:`paired_success_differences`; the follow nodes,
not yet built, choose that.

The gait clause
---------------
Command tracking reads the pelvis, so on its own it would certify a hop or a
slide that keeps the requested speed and heading (GAIT_QUALITY_PLAN_2026_09.md
GQ-18).  Every cruise and follow episode is therefore also judged on its
footfalls, measured on floor truth by the panel runner
(:func:`~environments.shared.gait.locomotion_metrics.episode_locomotion_metrics`,
the window from 1 s on) and recorded in the episode's ``gait_metrics``.  The
clause is required in both profiles: ``gait_limbs`` (2 or 4) names the
animal, and with it the bars that must be declared
(:data:`GAIT_REQUIRED_KEYS`).

* A biped (``gait_limbs = 2``: the velociraptor, the T. rex, both
  compsognathus and the allosaurus) is judged on GQ-8 (a)'s separators:
  ``min_alternation_index``, ``max_simultaneous_fraction``,
  ``max_hop_flight_fraction``, ``min_lead_swaps_per_stride``,
  ``max_foot_contact_fraction`` and ``max_phantom_support_fraction``, with
  ``max_flight_fraction`` optional (declared where the commanded speed is
  below Froude 0.5).  A two-footed hop lands both feet together (no
  alternation, simultaneous touchdowns, flight not bracketed by alternating
  footfalls); a slide, with both feet down while the body moves, makes no
  touchdowns, so its alternation and lead swaps are unmeasured and fail.  A
  hop whose feet land more than 20 ms apart reads as alternating footfalls
  and its flight as alternating flight under the plan's 20 ms definitions,
  so of the required separators only the lead swaps refuse it: they count a
  change of the leading foot only once the fore-aft offset has passed a
  dead band of ``gait.locomotion_metrics.LEAD_DEAD_BAND_L`` (0.03) leg
  lengths (``locomotion-gait/v2``), which feet side by side never do.  (A
  plain sign change, the probe's and v1's, flickered about zero on the
  real plants, so the T. rex hop with its feet landing 30 ms apart passed
  every required biped separator.)
* A quadruped (``gait_limbs = 4``: the brachiosaurus and the dibothrosuchus)
  is judged on GQ-9 (a)'s floors: ``max_flight_fraction``,
  ``min_limb_duty`` and ``min_pair_alternation_index`` (the lower of the fore
  and hind pairs').  A walk, a trot and a pace meet them, and a skid with
  every foot down, a pronk or bound that flies, and a pronk or bound whose
  pair feet land within the 20 ms merge do not; the lateral-sequence or trot match
  (the Hildebrand limb phase) is measured and report-only.  The floor-truth
  tooling computes every one of these on four limbs, and both repository
  reference walks clear them, so the clause is enforced rather than
  report-only.

What the clause does not refuse.  A staggered biped hop whose feet swing
their lead past the dead band about once a stride or more clears every
required biped separator; only the optional flight cap refuses it, where it
is declared (a biped above Froude 0.5, the velociraptor at its cruise,
declares none).  A skid broken by brief alternating lifts makes alternating
touchdowns on either animal and is not refused: slip is report-only under
GQ-8 (a) and GQ-9 (a).  A flightless quadruped bound whose pair feet land
more than 20 ms apart reads as alternating pairs and clears every required
quadruped floor (the Hildebrand phase reads it as a trot's 0.5).  The
optional ``min_lead_swaps_per_stride`` refuses it only while each pair's
feet stay side by side, within the dead band, where they never swap: a
bound whose pair feet swing their lead past the band reads about two lead
swaps a stride and clears every quadruped bar, that one included, so the
bar narrows the gap and does not close it.  The metrics record the first
two, report-only:
``left_right_phase`` (near 0.5 on a walk, near 0 or 1 on a hop, staggered or
not) and ``stance_slip_speed_mps`` (near 0 under a planted foot, the body's
speed under a skidding one).

The other separators are optional on either animal.  An episode measured on
another number of limbs fails ``gait_limbs``, and a panel holding one is
refused.  The stand family's gait is not judged, so stepping in place within
the stop tolerances (the stop criterion reads the pelvis only) passes it;
every stand record carries its gait metrics, and the stand family's recorded
alternation and ``touchdown_rate`` are the report-only signal.  The
starting bars are PROVISIONAL (:data:`PROVISIONAL_BIPED_GAIT_THRESHOLDS`,
:data:`PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS`), copied from the plan's §4.3.

Tolerances and the measurement
------------------------------
The in-tolerance test is the env's own (``tracking_metrics``), with the
tolerances the stage's command config resolves
(``DirectionCommandConfig.tracking_parameters``); a ``gate_*_tolerance`` key
replaces one for the gate.  Each episode record carries the
:class:`PanelMeasurement` it was reduced under (the protocol, the five
tolerances, ``yaw_rate_max`` and the gait metrics' definitions), and the
gate refuses a record measured under another one: its fractions answer a
different question.  The timing
constants are part of the protocol (:data:`COMMAND_PANEL_PROTOCOL`), not
threshold keys; another settle or dwell is another protocol.

Provisional values
------------------
Every committed walk declares the kind, under the walk profile, since
decision D-D62; no committed stage declares the follow profile yet.
:data:`PROVISIONAL_WALK_THRESHOLDS` and
:data:`PROVISIONAL_FOLLOW_THRESHOLDS` are the documented starting values,
copied from ``configs/behavior_certification.toml``, and the gait clause's
come from GAIT_QUALITY_PLAN_2026_09.md §4.3 (a stage declares a profile's
block with its animal's gait block); every one is
PROVISIONAL, to be frozen from the first pilot's panel, and a verdict minted
under a provisional bar is re-judged, never retrained, when the bar moves
(D-B7/D-B8).

This module is pure: the standard library, numpy through
``behavior_evaluation.summarize_episode`` and the pure
``gait.locomotion_metrics`` (its measurement version and field names),
``direction_commands`` (``DirectionCommandConfig`` and ``wrap_angle``) and
the binomial helpers of :mod:`~.recovery_gate`.  It never builds an env; the
panel runner does.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from ..behavior_evaluation import summarize_episode
from ..direction_commands import DirectionCommandConfig, wrap_angle
from ..gait.locomotion_metrics import GAIT_MEASUREMENT, LOCOMOTION_METRIC_FIELDS
from .recovery_gate import binomial_lcb, paired_difference_lcb

#: The gate kind this module evaluates.  Imported by ``gate_schema`` so the
#: registry and the implementation cannot drift apart.
COMMAND_TRACKING_GATE_KIND = "command_tracking/v1"

#: The threshold key that selects the profile, and its two values.
PROFILE_KEY = "profile"
WALK_PROFILE = "walk"
FOLLOW_PROFILE = "follow"
PROFILES: tuple[str, ...] = (WALK_PROFILE, FOLLOW_PROFILE)

#: The walk profile's two families, by the command an episode's reset draws.
CRUISE_FAMILY = "cruise"
STAND_FAMILY = "stand"
FAMILIES: tuple[str, ...] = (CRUISE_FAMILY, STAND_FAMILY)

#: The measurement protocol every episode record is reduced under: the event
#: logic's settle base and dwell, the walk speed window and the stand hold
#: window below, and the certificate's turn test.  Another value of any of
#: them is another protocol.
COMMAND_PANEL_PROTOCOL = "command_panel/v1"

#: ``summarize_episode``'s settling base: an event's settling allowance is
#: this plus ``|initial heading error| / yaw_rate_max`` (seconds).
SETTLE_BASE_S = 1.0
#: ``summarize_episode``'s dwell: a settled event's in-tolerance run lasts this long (seconds).
DWELL_S = 1.0
#: The walk speed ratio reads the active steps that end after this time (seconds).
SPEED_SETTLE_S = 2.0
#: The stand hold fraction reads the steps that end after this time (seconds).
STAND_SETTLE_S = 1.0
#: A turn is an active event whose requested heading differs from the previous
#: event's by more than this (radians): the certificate's test.
TURN_HEADING_CHANGE_RAD = 0.05

#: The reason key of an episode that did not reach the horizon (or fell).
HORIZON_REASON = "horizon"
#: The reason key of a cruise episode whose first event (its heading) did not settle.
HEADING_SETTLE_REASON = "heading_settle"
#: The reason key of a stand episode whose stop did not settle.
STOP_SETTLE_REASON = "stop_settle"

#: The gate keys that replace one of the command config's tracking tolerances
#: for the gate: ``(threshold key, tracking_metrics keyword)``.
TOLERANCE_OVERRIDE_KEYS: tuple[tuple[str, str], ...] = (
    ("gate_velocity_tolerance", "velocity_tolerance"),
    ("gate_yaw_rate_tolerance", "yaw_rate_tolerance"),
    ("gate_heading_tolerance", "heading_tolerance"),
    ("gate_stop_speed_tolerance", "stop_speed_tolerance"),
    ("gate_stop_yaw_rate_tolerance", "stop_yaw_rate_tolerance"),
)
#: The five ``tracking_metrics`` tolerance keywords, in that function's order.
TOLERANCE_NAMES: tuple[str, ...] = tuple(name for _, name in TOLERANCE_OVERRIDE_KEYS)

#: The gait clause's per-episode criteria (GAIT_QUALITY_PLAN_2026_09 GQ-8 (a),
#: GQ-9 (a) and GQ-18): ``(threshold key, gait metric, direction)``, ``"min"``
#: a floor and ``"max"`` a ceiling, read off an episode's ``gait_metrics``
#: (:class:`~environments.shared.gait.locomotion_metrics.LocomotionEpisodeMetrics`,
#: which defines every metric).  Each is a threshold key of both profiles,
#: parsed into the thresholds' :class:`GaitClause`, and fails a cruise (walk)
#: or follow episode whose metric misses it or is unmeasured
#: (:func:`gait_failures`).  The panel runner measures the metrics on floor
#: truth.  Which are required depends on the animal (:data:`GAIT_REQUIRED_KEYS`).
GAIT_CRITERIA: tuple[tuple[str, str, str], ...] = (
    ("min_alternation_index", "alternation_index", "min"),
    ("min_pair_alternation_index", "pair_alternation_index", "min"),
    ("max_simultaneous_fraction", "simultaneous_fraction", "max"),
    ("max_hop_flight_fraction", "hop_flight_fraction", "max"),
    ("min_lead_swaps_per_stride", "lead_swaps_per_stride", "min"),
    ("max_foot_contact_fraction", "foot_contact_fraction", "max"),
    ("max_phantom_support_fraction", "phantom_support_fraction", "max"),
    ("max_flight_fraction", "flight_fraction", "max"),
    ("min_limb_duty", "min_limb_duty", "min"),
)
GAIT_CRITERION_KEYS: frozenset[str] = frozenset(key for key, _, _ in GAIT_CRITERIA)
#: The gait clause's required selector: the number of limbs the clause judges,
#: 2 (a biped's clause) or 4 (a quadruped's).  It picks the required bars, and
#: the gate refuses an episode that measured another number of limbs.
GAIT_LIMBS_KEY = "gait_limbs"
#: The gait metric that records how many limbs an episode was measured on.
GAIT_LIMBS_METRIC = "limbs"
BIPED_LIMBS = 2
QUADRUPED_LIMBS = 4
GAIT_THRESHOLD_KEYS: frozenset[str] = frozenset(GAIT_CRITERION_KEYS | {GAIT_LIMBS_KEY})

#: The bars a gait clause must declare, by its ``gait_limbs``.  A biped's are
#: GQ-8 (a)'s separators, each at least 4x apart between the genuine walkers
#: and the hops of the 2026-09-28 replays (§4.3 A); the flight cap is optional,
#: declared where the commanded speed is below Froude 0.5 (a velociraptor run
#: may fly, and hop-flight confines its flight to alternating footfalls).  A
#: quadruped's are GQ-9 (a)'s floors, which a walk, a trot and a pace all meet
#: and a statue, a skid and a pronk or bound that flies or lands its pair feet
#: within the 20 ms merge do not (a flightless bound whose pair feet land
#: further apart clears them; the optional lead-swap bar refuses it only while
#: each pair's feet stay side by side within the dead band, and a bound whose
#: pair feet swing their lead past the band clears it too).  The lateral-sequence or
#: trot match (the Hildebrand limb phase, ``ipsilateral_phase``) is measured
#: and report-only: §4.3 B measured its band on no passing quadruped.
GAIT_REQUIRED_KEYS: dict[int, frozenset[str]] = {
    BIPED_LIMBS: frozenset(
        {
            "min_alternation_index",
            "max_simultaneous_fraction",
            "max_hop_flight_fraction",
            "min_lead_swaps_per_stride",
            "max_foot_contact_fraction",
            "max_phantom_support_fraction",
        }
    ),
    QUADRUPED_LIMBS: frozenset({"max_flight_fraction", "min_limb_duty", "min_pair_alternation_index"}),
}
#: The bars a gait clause may declare, by its ``gait_limbs``: every criterion
#: but the other animal's alternation key (a biped's one pair is
#: ``min_alternation_index``; a quadruped has two pairs and no single
#: left-right index, so it declares ``min_pair_alternation_index``).
GAIT_ALLOWED_KEYS: dict[int, frozenset[str]] = {
    BIPED_LIMBS: frozenset(GAIT_CRITERION_KEYS - {"min_pair_alternation_index"}),
    QUADRUPED_LIMBS: frozenset(GAIT_CRITERION_KEYS - {"min_alternation_index"}),
}
if any(not GAIT_REQUIRED_KEYS[limbs] <= GAIT_ALLOWED_KEYS[limbs] for limbs in GAIT_REQUIRED_KEYS):
    raise RuntimeError("every required gait bar must be an allowed one")
if not {metric for _, metric, _ in GAIT_CRITERIA} <= set(LOCOMOTION_METRIC_FIELDS) or (
    GAIT_LIMBS_METRIC not in LOCOMOTION_METRIC_FIELDS
):
    raise RuntimeError("every gait criterion must read a LocomotionEpisodeMetrics field")

#: The keys both profiles accept: the tolerance overrides and
#: ``required_consecutive`` (allowed, like every advancing kind's copy of it,
#: decision D-B3, as the hysteresis of an in-training screen; the panel
#: verdict does not read it).
SHARED_OPTIONAL_KEYS: frozenset[str] = frozenset({key for key, _ in TOLERANCE_OVERRIDE_KEYS} | {"required_consecutive"})

#: The walk profile's bars; all required.
WALK_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "min_cruise_episodes",
        "min_stand_episodes",
        "min_cruise_survival_lcb",
        "min_cruise_success_lcb",
        "min_cruise_tracking_fraction",
        "min_cruise_speed_ratio",
        "min_stand_hold_fraction",
        "min_stand_success_fraction",
        "max_stand_falls",
    }
)

#: The follow profile's bars from the behavior certificate; all required.
FOLLOW_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "min_follow_episodes",
        "min_follow_horizon_s",
        "min_follow_survival_lcb",
        "min_follow_success_lcb",
        "min_follow_tracking_fraction",
        "min_follow_event_settle_fraction",
        "min_follow_stop_settle_fraction",
        "min_follow_turn_settle_fraction",
        "min_follow_command_episodes",
    }
)

#: The follow profile's optional criteria: the course pair (declared together:
#: the fraction is of ``follow_course_distance_m``, a species' length scale)
#: and the paired reference comparison.
FOLLOW_OPTIONAL_KEYS: frozenset[str] = frozenset(
    {"min_follow_course_fraction", "follow_course_distance_m", "min_follow_reference_delta_lcb"}
)

#: Every threshold key each profile consumes.
WALK_THRESHOLD_KEYS: frozenset[str] = frozenset(
    {PROFILE_KEY} | WALK_REQUIRED_KEYS | SHARED_OPTIONAL_KEYS | GAIT_THRESHOLD_KEYS
)
FOLLOW_THRESHOLD_KEYS: frozenset[str] = frozenset(
    {PROFILE_KEY} | FOLLOW_REQUIRED_KEYS | FOLLOW_OPTIONAL_KEYS | SHARED_OPTIONAL_KEYS | GAIT_THRESHOLD_KEYS
)
#: Every threshold key the kind consumes (``gate_schema.GATE_KINDS``): both
#: profiles' keys.  A stage declaring one profile may carry only that
#: profile's keys (:func:`thresholds_from_curriculum`).
COMMAND_TRACKING_THRESHOLD_KEYS: frozenset[str] = frozenset(WALK_THRESHOLD_KEYS | FOLLOW_THRESHOLD_KEYS)
#: The key the kind always requires (``gate_schema._REQUIRED_THRESHOLD_KEYS``);
#: the profile's own required keys are enforced by :func:`thresholds_from_curriculum`,
#: which ``gate_schema.validate_gate_config`` calls for this kind.
COMMAND_TRACKING_REQUIRED_KEYS: frozenset[str] = frozenset({PROFILE_KEY})

#: How each numeric key is validated: ``(rule, why)``.  ``count`` is a whole
#: number of at least the rule's minimum; ``bound`` lies strictly inside
#: (0, 1) (a bound of 0 is met by a panel with no success, and no finite panel
#: bounds at 1); ``fraction`` lies in (0, 1] (0 admits every episode);
#: ``ratio`` lies in (0, 1] (above 1 demands overshooting the command);
#: ``positive`` is above 0; ``difference`` lies strictly inside (-1, 1)
#: (a mean difference of two success indicators); ``ceiling`` lies in [0, 1)
#: (a gait fraction's cap: 0 allows none, and 1 admits every episode);
#: ``limbs`` is 2 or 4.
_COUNT_MINIMUMS: dict[str, int] = {
    "min_cruise_episodes": 1,
    "min_stand_episodes": 1,
    "max_stand_falls": 0,
    "min_follow_episodes": 1,
    "min_follow_command_episodes": 1,
    "required_consecutive": 1,
}
_BOUND_KEYS: frozenset[str] = frozenset(
    {"min_cruise_survival_lcb", "min_cruise_success_lcb", "min_follow_survival_lcb", "min_follow_success_lcb"}
)
_FRACTION_KEYS: frozenset[str] = frozenset(
    {
        "min_cruise_tracking_fraction",
        "min_stand_hold_fraction",
        "min_stand_success_fraction",
        "min_follow_tracking_fraction",
        "min_follow_event_settle_fraction",
        "min_follow_stop_settle_fraction",
        "min_follow_turn_settle_fraction",
        "min_follow_course_fraction",
        "min_alternation_index",
        "min_pair_alternation_index",
        "min_limb_duty",
    }
)
_RATIO_KEYS: frozenset[str] = frozenset({"min_cruise_speed_ratio"})
_POSITIVE_KEYS: frozenset[str] = frozenset(
    {"min_follow_horizon_s", "follow_course_distance_m", "min_lead_swaps_per_stride"}
    | {key for key, _ in TOLERANCE_OVERRIDE_KEYS}
)
_DIFFERENCE_KEYS: frozenset[str] = frozenset({"min_follow_reference_delta_lcb"})
_CEILING_KEYS: frozenset[str] = frozenset(
    {
        "max_simultaneous_fraction",
        "max_hop_flight_fraction",
        "max_foot_contact_fraction",
        "max_phantom_support_fraction",
        "max_flight_fraction",
    }
)
_LIMB_KEYS: frozenset[str] = frozenset({GAIT_LIMBS_KEY})

# Every numeric key has exactly one rule; a key added to a profile without one
# would be read by nothing.  A raise, not an assert: `python -O` strips asserts.
_RULES = (_BOUND_KEYS, _FRACTION_KEYS, _RATIO_KEYS, _POSITIVE_KEYS, _DIFFERENCE_KEYS, _CEILING_KEYS, _LIMB_KEYS)
_RULED = [*_COUNT_MINIMUMS, *(key for rule in _RULES for key in rule)]
if len(_RULED) != len(set(_RULED)) or set(_RULED) != set(COMMAND_TRACKING_THRESHOLD_KEYS - {PROFILE_KEY}):
    raise RuntimeError("every command_tracking/v1 numeric threshold key needs exactly one validation rule")

#: PROVISIONAL starting values of the walk profile, to be frozen from the
#: first walk pilot's panel (decision D-D61).  Each is copied from, or
#: restates, a bar of ``configs/behavior_certification.toml``
#: (``direction-terrain-certification/v1``), named per line; nothing reads
#: that file when a stage is judged.  A stage declares these in its
#: ``[curriculum]`` table, so its gate digest records them.
PROVISIONAL_WALK_THRESHOLDS: dict[str, Any] = {
    PROFILE_KEY: WALK_PROFILE,
    # episodes_per_family = 20: the certificate's panel size per family.
    "min_cruise_episodes": 20,
    # minimum_command_episodes = 5: the certificate's least number of episodes exposed to a stop.
    "min_stand_episodes": 5,
    # minimum_survival_lcb = 0.80: at 20 episodes only 20/20 clears it (LCB 0.861; 19/20 bounds at 0.784).
    "min_cruise_survival_lcb": 0.80,
    # minimum_success_lcb = 0.60: at 20 episodes 17/20 clears it (0.656) and 16/20 does not (0.599).
    "min_cruise_success_lcb": 0.60,
    # minimum_tracking_fraction = 0.60: the certificate's per-episode in-tolerance bar, read after settle here.
    "min_cruise_tracking_fraction": 0.60,
    # minimum_course_fraction = 0.50: the certificate's progress bar, restated as progress along the commanded
    # heading per commanded speed.  A placeholder, looser than the design draft's 0.9 x cruise: the per-step tracking
    # bar already bounds the speed error (the velocity tolerance), and the pilot's panel sets the value.
    "min_cruise_speed_ratio": 0.50,
    # minimum_tracking_fraction = 0.60: at a zero command the env's in-tolerance test is the stop criterion.  The
    # design's "held after a 1 s settle" read as a FRACTION of the steps after 1 s (with the stop event settled), so at
    # 0.60 a stand may spend 40% of them moving; the pilot is to freeze it near 1.
    "min_stand_hold_fraction": 0.60,
    # minimum_stop_settle_fraction = 0.60: the certificate's fraction of stop-exposed episodes that stop.
    "min_stand_success_fraction": 0.60,
    # The design (command_tracking/v1, D-D61): a stand episode does not fall.  Not from the certificate.
    "max_stand_falls": 0,
}

#: PROVISIONAL starting values of the follow profile, to be frozen from the
#: first follow pilot's panel (decision D-D61): the bars of
#: ``configs/behavior_certification.toml`` under the gate's own keys, named per
#: line.  The course pair is left out because ``follow_course_distance_m`` is a
#: species' length scale (each follow_direction recipe's ``course_distance``);
#: a stage that declares it starts ``min_follow_course_fraction`` at
#: :data:`PROVISIONAL_FOLLOW_COURSE_FRACTION`.  The reference comparison has no
#: starting value: the follow nodes that pair against the walker choose it.
PROVISIONAL_FOLLOW_THRESHOLDS: dict[str, Any] = {
    PROFILE_KEY: FOLLOW_PROFILE,
    "min_follow_episodes": 20,  # episodes_per_family = 20
    "min_follow_horizon_s": 20.0,  # minimum_horizon_s = 20.0
    "min_follow_survival_lcb": 0.80,  # minimum_survival_lcb = 0.80
    "min_follow_success_lcb": 0.60,  # minimum_success_lcb = 0.60
    "min_follow_tracking_fraction": 0.60,  # minimum_tracking_fraction = 0.60
    "min_follow_event_settle_fraction": 0.60,  # minimum_event_settle_fraction = 0.60
    "min_follow_stop_settle_fraction": 0.60,  # minimum_stop_settle_fraction = 0.60
    "min_follow_turn_settle_fraction": 0.60,  # minimum_turn_settle_fraction = 0.60
    "min_follow_command_episodes": 5,  # minimum_command_episodes = 5
}
#: PROVISIONAL: ``minimum_course_fraction = 0.50`` of the certificate, for a
#: stage that declares the course pair.
PROVISIONAL_FOLLOW_COURSE_FRACTION = 0.50

#: PROVISIONAL starting values of a biped's gait clause (both profiles), to be
#: frozen from the first walk pilot's panel (decision D-D61): the bars of
#: GAIT_QUALITY_PLAN_2026_09.md §4.3 A (GQ-8 (a)), each named with the per-episode
#: range it separates on the 2026-09-28 CPU replays of the genuine walkers
#: (compsognathus `20260921_203149`, velociraptor `20260922_125248`, certified and
#: final checkpoints) and the hops (Tyrannosaurus rex seeds 42 and 44, the
#: compsognathus robot).  Every repository reference walk (the Tyrannosaurus
#: rex's, the velociraptor's and the allosaurus's) clears them
#: (``test_command_tracking_gait.py``).
PROVISIONAL_BIPED_GAIT_THRESHOLDS: dict[str, Any] = {
    GAIT_LIMBS_KEY: BIPED_LIMBS,
    # §4.3 A: walkers 0.857-1.0 (the velociraptor's 0.857 the tightest margin), hops 0-0.058.
    "min_alternation_index": 0.75,
    # §4.3 A: walkers 0-0.043, hops 0.811-1.0.
    "max_simultaneous_fraction": 0.15,
    # §4.3 A: walkers 0-0.069, hops 0.323-0.49 (the hops' on representative traces).
    "max_hop_flight_fraction": 0.15,
    # §4.3 A: walkers 1.38-2.16 (the compsognathus's 1.38 the tightest margin), hops 0-0.05, both as the probe's plain
    # sign change.  locomotion-gait/v2 counts through a dead band of 0.03 leg lengths, which can only remove swaps:
    # the repository's reference walks read 1.58-2.02 under it, and the scripted statue hops 0, staggered or not.
    "min_lead_swaps_per_stride": 1.0,
    # §4.3 A: 0 on every walker and every Tyrannosaurus rex hop; the robot's stacked feet 0.949-0.966.
    "max_foot_contact_fraction": 0.02,
    # §4.3 A: 0 on every walker and every Tyrannosaurus rex hop; the robot 0.526-0.592.
    "max_phantom_support_fraction": 0.05,
}
#: PROVISIONAL: §4.3 A's biped flight cap, ``max_flight_fraction``, which the
#: plan declares where the stage's commanded speed is below Froude 0.5 (the
#: Tyrannosaurus rex, both compsognathus and, since decision D-D62 (12), the
#: allosaurus at Froude 0.007; walkers 0-0.007, hops 0.324-0.497) and leaves
#: off a biped that may run (the velociraptor at Froude 0.82).
PROVISIONAL_BIPED_FLIGHT_FRACTION = 0.05
#: PROVISIONAL starting values of a quadruped's gait clause (both profiles), to
#: be frozen from the first quadruped walk pilot's panel (decision D-D61): GQ-9
#: (a)'s floors (§4.3 B), which no certified quadruped has met yet.  The July
#: brachiosaurus bound passes the flight and duty floors and fails pair
#: alternation (hind 30 of 30 episodes, fore 29 of 30); the dibothrosuchus
#: statue and skid fail all three.  Both repository reference walks clear them
#: (``test_command_tracking_gait.py``).
PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS: dict[str, Any] = {
    GAIT_LIMBS_KEY: QUADRUPED_LIMBS,
    # §4.3 B: the July bound 0.013-0.041, the skid 0.15-0.46.
    "max_flight_fraction": 0.05,
    # §4.3 B: the July bound 0.25-0.42, the skid 0-0.12.
    "min_limb_duty": 0.20,
    # §4.3 B: the July bound's fore 0.18-0.87 and hind 0.19-0.53, the skid 0-0.53.
    "min_pair_alternation_index": 0.75,
}


# ── thresholds ───────────────────────────────────────────────────────────────


def _number(value: Any, *, key: str) -> float:
    """*value* as a finite float, or ``ValueError`` naming *key*.

    A real NUMBER only: a string that would parse is refused, because the
    gate digest hashes a string verbatim and ``gate_schema.same_threshold``
    would then never match it against the number a report records.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{COMMAND_TRACKING_GATE_KIND} threshold {key} is missing or not a number ({value!r})")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must be finite, not {value!r}")
    return number


def _validated(key: str, value: Any) -> float | int:
    """*value* checked against *key*'s rule, as an ``int`` for a count and a ``float`` otherwise."""
    number = _number(value, key=key)
    if key in _COUNT_MINIMUMS:
        minimum = _COUNT_MINIMUMS[key]
        if number != int(number) or number < minimum:
            raise ValueError(
                f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must be a whole number of at least {minimum}, "
                f"not {value!r}"
            )
        return int(number)
    if key in _BOUND_KEYS and not 0.0 < number < 1.0:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must lie strictly between 0 and 1, not {value!r}: 0 "
            "certifies a panel with no success, and no finite panel bounds at 1"
        )
    if key in _FRACTION_KEYS and not 0.0 < number <= 1.0:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must lie in (0, 1], not {value!r}: it is a fraction, "
            "and a bar of 0 admits every episode"
        )
    if key in _RATIO_KEYS and not 0.0 < number <= 1.0:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must lie in (0, 1], not {value!r}: it is the achieved "
            "fraction of the commanded speed, and above 1 demands overshooting the command"
        )
    if key in _POSITIVE_KEYS and not number > 0.0:
        raise ValueError(f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must be positive, not {value!r}")
    if key in _DIFFERENCE_KEYS and not -1.0 < number < 1.0:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must lie strictly between -1 and 1, not {value!r}: it "
            "bounds a mean difference of two success indicators"
        )
    if key in _CEILING_KEYS and not 0.0 <= number < 1.0:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must lie in [0, 1), not {value!r}: it caps a fraction, "
            "and a cap of 1 admits every episode"
        )
    if key in _LIMB_KEYS:
        if number not in (BIPED_LIMBS, QUADRUPED_LIMBS):
            raise ValueError(
                f"{COMMAND_TRACKING_GATE_KIND} threshold {key} must be {BIPED_LIMBS} (a biped's gait clause) or "
                f"{QUADRUPED_LIMBS} (a quadruped's), not {value!r}"
            )
        return int(number)
    return number


@dataclass(frozen=True)
class GaitClause:
    """A stage's gait clause: the limbs it judges and its declared bars.

    ``bars`` holds ``(threshold key, value)`` pairs in :data:`GAIT_CRITERIA`
    order: every bar :data:`GAIT_REQUIRED_KEYS` names for ``limbs`` and each
    optional one the stage declares.
    """

    limbs: int
    bars: tuple[tuple[str, float], ...]

    def bar(self, key: str) -> float | None:
        """The declared bar of *key*, ``None`` when undeclared."""
        return dict(self.bars).get(key)

    def declared(self) -> dict[str, Any]:
        """The clause as ``[curriculum]`` keys."""
        return {GAIT_LIMBS_KEY: self.limbs, **dict(self.bars)}


def gait_clause_from_curriculum(curriculum: Mapping[str, Any]) -> GaitClause:
    """Read the gait clause a ``command_tracking/v1`` block declares, strictly (``ValueError`` naming the key).

    ``gait_limbs`` is required and selects which bars are
    (:data:`GAIT_REQUIRED_KEYS`: a biped's for ``gait_limbs = 2``, a
    quadruped's for ``gait_limbs = 4``).  A bar the selected animal cannot be judged on
    (:data:`GAIT_ALLOWED_KEYS`: a biped's ``min_pair_alternation_index``, a
    quadruped's ``min_alternation_index``) is refused, as is every value
    outside its rule.
    """
    if GAIT_LIMBS_KEY not in curriculum:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} is missing required threshold(s) ['{GAIT_LIMBS_KEY}']: the gait clause is "
            "part of every profile, so a panel is never certified on the pelvis velocity alone"
        )
    limbs = int(_validated(GAIT_LIMBS_KEY, curriculum[GAIT_LIMBS_KEY]))
    animal = "biped" if limbs == BIPED_LIMBS else "quadruped"
    declared = set(curriculum) & GAIT_CRITERION_KEYS
    foreign = sorted(declared - GAIT_ALLOWED_KEYS[limbs])
    if foreign:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} gait clause for {limbs} limbs (a {animal}) does not judge {foreign}: "
            "min_alternation_index is a biped's left-right index, and a quadruped declares "
            "min_pair_alternation_index for its fore and hind pairs"
        )
    missing = sorted(GAIT_REQUIRED_KEYS[limbs] - declared)
    if missing:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} gait clause for {limbs} limbs (a {animal}) is missing required "
            f"threshold(s) {missing}"
        )
    bars = tuple((key, float(_validated(key, curriculum[key]))) for key, _, _ in GAIT_CRITERIA if key in curriculum)
    return GaitClause(limbs=limbs, bars=bars)


def _shared_values(curriculum: Mapping[str, Any]) -> dict[str, Any]:
    """The tolerance overrides, the gait clause and ``required_consecutive`` a block declares, validated."""
    overrides = tuple(
        (name, float(_validated(key, curriculum[key]))) for key, name in TOLERANCE_OVERRIDE_KEYS if key in curriculum
    )
    values: dict[str, Any] = {"tolerance_overrides": overrides, "gait": gait_clause_from_curriculum(curriculum)}
    if "required_consecutive" in curriculum:
        values["required_consecutive"] = _validated("required_consecutive", curriculum["required_consecutive"])
    return values


@dataclass(frozen=True)
class WalkThresholds:
    """The walk profile's bars, as a stage's ``[curriculum]`` declares them (every bar required).

    ``gait`` is the gait clause (:class:`GaitClause`, required: its bars
    judge the cruise family); ``tolerance_overrides`` holds the declared
    ``gate_*_tolerance`` keys as ``(tracking_metrics keyword, value)`` pairs
    in their table's order; ``required_consecutive`` is ``None`` when
    undeclared.
    """

    min_cruise_episodes: int
    min_stand_episodes: int
    min_cruise_survival_lcb: float
    min_cruise_success_lcb: float
    min_cruise_tracking_fraction: float
    min_cruise_speed_ratio: float
    min_stand_hold_fraction: float
    min_stand_success_fraction: float
    max_stand_falls: int
    gait: GaitClause
    tolerance_overrides: tuple[tuple[str, float], ...] = ()
    required_consecutive: int | None = None

    profile: ClassVar[str] = WALK_PROFILE
    required_keys: ClassVar[frozenset[str]] = WALK_REQUIRED_KEYS
    threshold_keys: ClassVar[frozenset[str]] = WALK_THRESHOLD_KEYS

    def declared(self) -> dict[str, Any]:
        """The thresholds as ``[curriculum]`` keys: the profile, every bar, the gait clause and each declared override."""
        return _declared(self)


@dataclass(frozen=True)
class FollowThresholds:
    """The follow profile's bars, as a stage's ``[curriculum]`` declares them.

    The certificate's bars and the gait clause (:class:`GaitClause`) are
    required.  ``min_follow_course_fraction`` and ``follow_course_distance_m``
    are declared together or not at all, and
    ``min_follow_reference_delta_lcb`` activates the paired comparison; an
    undeclared optional criterion is ``None`` and not applied (and an
    undeclared ``required_consecutive`` is ``None``).
    """

    min_follow_episodes: int
    min_follow_horizon_s: float
    min_follow_survival_lcb: float
    min_follow_success_lcb: float
    min_follow_tracking_fraction: float
    min_follow_event_settle_fraction: float
    min_follow_stop_settle_fraction: float
    min_follow_turn_settle_fraction: float
    min_follow_command_episodes: int
    gait: GaitClause
    min_follow_course_fraction: float | None = None
    follow_course_distance_m: float | None = None
    min_follow_reference_delta_lcb: float | None = None
    tolerance_overrides: tuple[tuple[str, float], ...] = ()
    required_consecutive: int | None = None

    profile: ClassVar[str] = FOLLOW_PROFILE
    required_keys: ClassVar[frozenset[str]] = FOLLOW_REQUIRED_KEYS
    threshold_keys: ClassVar[frozenset[str]] = FOLLOW_THRESHOLD_KEYS

    def declared(self) -> dict[str, Any]:
        """The thresholds as ``[curriculum]`` keys: the profile, every declared bar, the gait clause and each override."""
        return _declared(self)


CommandTrackingThresholds = WalkThresholds | FollowThresholds

_OVERRIDE_KEY_BY_NAME = {name: key for key, name in TOLERANCE_OVERRIDE_KEYS}


def _declared(thresholds: CommandTrackingThresholds) -> dict[str, Any]:
    values: dict[str, Any] = {PROFILE_KEY: thresholds.profile}
    for spec in fields(thresholds):
        value = getattr(thresholds, spec.name)
        if spec.name == "tolerance_overrides":
            values.update({_OVERRIDE_KEY_BY_NAME[name]: bar for name, bar in value})
        elif spec.name == "gait":
            values.update(value.declared())
        elif value is not None:
            values[spec.name] = value
    return values


# A dataclass field added without a schema key (or the reverse) would be read
# by nothing or validated by nothing; make the drift unmissable at import.
def _check_fields(cls: type[WalkThresholds] | type[FollowThresholds]) -> None:
    named = {spec.name for spec in fields(cls)} - {"tolerance_overrides", "gait"}
    expected = (
        set(cls.threshold_keys) - {PROFILE_KEY} - GAIT_THRESHOLD_KEYS - {key for key, _ in TOLERANCE_OVERRIDE_KEYS}
    )
    if named != expected:
        raise RuntimeError(f"{cls.__name__} fields must be exactly its profile's threshold keys")


_check_fields(WalkThresholds)
_check_fields(FollowThresholds)


def thresholds_from_curriculum(curriculum: Mapping[str, Any]) -> CommandTrackingThresholds:
    """Read the ``command_tracking/v1`` thresholds a stage's ``[curriculum]`` table declares.

    Strict, raising ``ValueError`` naming the key: ``profile`` must be
    ``"walk"`` or ``"follow"``; every required key of that profile must be
    present; a key of the OTHER profile is refused (it would read as a bar
    nobody applies); any key that is neither this profile's threshold key nor
    a non-threshold ``[curriculum]`` key (``gate_schema``'s schedule,
    collapse, diagnostic, retention, warm-start, provenance, publication and
    schema keys) is refused, so a misspelled optional bar cannot disable
    itself; and every value must satisfy its rule (whole counts, bounds
    strictly inside (0, 1), fractions and the speed ratio in (0, 1], positive
    tolerances, horizon and course distance, a reference delta inside
    (-1, 1), gait caps in [0, 1)).  The course pair is declared together or
    not at all.  The gait clause is required in both profiles
    (:func:`gait_clause_from_curriculum`: ``gait_limbs`` and the bars it
    selects).
    ``gate_schema.validate_gate_config`` calls this for the kind, so a bad
    block stops the run at config load.
    """
    from .gate_schema import NON_THRESHOLD_CURRICULUM_KEYS

    profile = curriculum.get(PROFILE_KEY)
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} threshold {PROFILE_KEY} must be one of {list(PROFILES)}, not {profile!r}"
        )
    cls: type[WalkThresholds] | type[FollowThresholds] = WalkThresholds if profile == WALK_PROFILE else FollowThresholds
    other = FOLLOW_THRESHOLD_KEYS if profile == WALK_PROFILE else WALK_THRESHOLD_KEYS
    foreign = sorted((set(curriculum) & other) - cls.threshold_keys)
    if foreign:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} profile {profile!r} does not consume {foreign}: they belong to the other "
            "profile, and a bar the declared profile never applies implies a gate that is not enforced"
        )
    unknown = sorted(set(curriculum) - cls.threshold_keys - NON_THRESHOLD_CURRICULUM_KEYS)
    if unknown:
        raise ValueError(
            f"{COMMAND_TRACKING_GATE_KIND} profile {profile!r} does not know the key(s) {unknown}; unknown keys are "
            "refused rather than ignored, because a misspelled bar would disable itself"
        )
    missing = sorted(cls.required_keys - set(curriculum))
    if missing:
        raise ValueError(f"{COMMAND_TRACKING_GATE_KIND} profile {profile!r} is missing required threshold(s) {missing}")
    values: dict[str, Any] = {key: _validated(key, curriculum[key]) for key in sorted(cls.required_keys)}
    if cls is FollowThresholds:
        for key in sorted(FOLLOW_OPTIONAL_KEYS):
            if key in curriculum:
                values[key] = _validated(key, curriculum[key])
        if ("min_follow_course_fraction" in values) != ("follow_course_distance_m" in values):
            raise ValueError(
                f"{COMMAND_TRACKING_GATE_KIND} declares min_follow_course_fraction and follow_course_distance_m "
                "together or not at all: the fraction is of the distance"
            )
    values.update(_shared_values(curriculum))
    return cls(**values)


# ── the measurement ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PanelMeasurement:
    """What a panel's episodes were reduced under: the protocol, the five tolerances, ``yaw_rate_max`` and the gait metrics.

    ``tolerances`` is ``(tracking_metrics keyword, value)`` pairs in
    :data:`TOLERANCE_NAMES` order.  Two records answer the same question
    only when their measurements are equal.
    """

    tolerances: tuple[tuple[str, float], ...]
    yaw_rate_max: float
    protocol: str = COMMAND_PANEL_PROTOCOL
    #: What the episode's ``gait_metrics`` mean: the floor-truth measurement
    #: version and the locomotion metrics' (``gait.locomotion_metrics.GAIT_MEASUREMENT``).
    gait_measurement: str = GAIT_MEASUREMENT

    def tolerance(self, name: str) -> float:
        return dict(self.tolerances)[name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "tolerances": dict(self.tolerances),
            "yaw_rate_max": self.yaw_rate_max,
            "gait_measurement": self.gait_measurement,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> PanelMeasurement:
        """The inverse of :meth:`as_dict`, strict: exactly its keys, the five tolerances, finite numbers.

        A missing or extra key, a tolerance set other than :data:`TOLERANCE_NAMES`,
        a non-finite or non-numeric value or a non-string protocol is a
        ``ValueError``: a measurement read back is compared field by field
        against the gate's, and a reading that guessed a field would compare
        equal to nothing or to the wrong thing.
        """
        if not isinstance(payload, Mapping) or set(payload) != _MEASUREMENT_KEYS:
            shown = sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__
            raise ValueError(f"a panel measurement records exactly {sorted(_MEASUREMENT_KEYS)}, not {shown}")
        tolerances = payload["tolerances"]
        if not isinstance(tolerances, Mapping) or set(tolerances) != set(TOLERANCE_NAMES):
            shown = sorted(tolerances) if isinstance(tolerances, Mapping) else tolerances
            raise ValueError(f"a panel measurement records the tolerances {sorted(TOLERANCE_NAMES)}, not {shown}")
        for key in ("protocol", "gait_measurement"):
            if not isinstance(payload[key], str) or not payload[key]:
                raise ValueError(f"a panel measurement's {key} is a non-empty string, not {payload[key]!r}")
        return cls(
            tolerances=tuple(
                (name, _record_number(tolerances[name], field=f"tolerance {name}")) for name in TOLERANCE_NAMES
            ),
            yaw_rate_max=_record_number(payload["yaw_rate_max"], field="yaw_rate_max"),
            protocol=str(payload["protocol"]),
            gait_measurement=str(payload["gait_measurement"]),
        )


_MEASUREMENT_KEYS = frozenset({"protocol", "tolerances", "yaw_rate_max", "gait_measurement"})


def _record_number(value: Any, *, field: str, allow_unmeasured: bool = False) -> float:
    """A recorded float: a finite number, or (with *allow_unmeasured*) ``None`` / NaN read as NaN; else ``ValueError``."""
    if allow_unmeasured and value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field} is a number, not {value!r}")
    number = float(value)
    if math.isnan(number) and allow_unmeasured:
        return number
    if not math.isfinite(number):
        raise ValueError(f"{field} is a finite number, not {value!r}")
    return number


def command_config_from(value: Any) -> DirectionCommandConfig | None:
    """A stage's ``command_config`` env kwarg as the dataclass: as given, rebuilt from a record, or ``None``.

    A ``stage_config.json`` records the config field by field
    (``config.env_kwargs_record``), so a mapping is rebuilt with
    ``DirectionCommandConfig(**mapping)``; ``None`` (a ``"none"`` stage)
    stays ``None``.  Anything else, or a mapping the dataclass refuses,
    raises ``TypeError`` / ``ValueError``.
    """
    if value is None or isinstance(value, DirectionCommandConfig):
        return value
    if isinstance(value, Mapping):
        return DirectionCommandConfig(**dict(value))
    raise TypeError(f"a command config is a DirectionCommandConfig or its record, not {type(value).__name__}")


def panel_measurement(command_config: Any, thresholds: CommandTrackingThresholds) -> PanelMeasurement:
    """The measurement a stage's panel is reduced and judged under.

    The tolerances are the ones the stage's env scores with
    (``command_config.tracking_parameters()["tolerances"]``), each replaced
    by its ``gate_*_tolerance`` key where the thresholds declare one;
    ``yaw_rate_max`` is the command config's (the settling allowance divides
    by it).  *command_config* is a ``direction_commands.DirectionCommandConfig``.
    """
    resolved = dict(command_config.tracking_parameters()["tolerances"])
    resolved.update(dict(thresholds.tolerance_overrides))
    return PanelMeasurement(
        tolerances=tuple((name, float(resolved[name])) for name in TOLERANCE_NAMES),
        yaw_rate_max=float(command_config.yaw_rate_max),
    )


# ── the per-episode record ───────────────────────────────────────────────────


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float:
    """The mean, NaN for no values or when any is not finite (never a mean over the finite subset)."""
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return math.fsum(values) / len(values)


@dataclass(frozen=True)
class CommandTrackingEpisode:
    """One panel episode reduced for the gate: what was commanded, what happened, and under which measurement.

    ``family`` is the walk family the reset draw puts the episode in
    (``stand`` for a zero requested speed, ``cruise`` otherwise; the follow
    profile reads it only as a label).  Fractions are NaN when no step
    qualified (unmeasured, which fails a bar).  Every field is measured on
    every episode; the walk profile reads ``tracking_fraction_after_settle``
    and the speed means on cruise episodes and ``stand_hold_fraction`` on
    stand episodes, and the follow profile the whole-episode
    ``tracking_fraction``, the event counts and the course progress.  The
    event counts are the certificate's: the eligible events, the eligible
    stops (events with no heading) and the eligible turns
    (:data:`TURN_HEADING_CHANGE_RAD`), each
    with how many settled.  ``gait_metrics`` holds the episode's floor-truth
    locomotion metrics (``LocomotionEpisodeMetrics.as_gait_metrics()``, every
    field by name), which the gait clause reads; empty for a record reduced
    without them, whose gait bars then fail as unmeasured.
    """

    episode: int
    seed: int
    family: str
    length: int
    horizon: int
    dt: float
    full_horizon: bool
    fall: bool
    termination_reason: str | None
    reward: float
    desired_speed: float
    planned_event_count: int
    initial_heading_error_rad: float
    first_event_settled: bool
    tracking_fraction: float
    tracking_fraction_after_settle: float
    mean_speed_along_heading_mps: float
    mean_desired_speed_mps: float
    stand_hold_fraction: float
    eligible_events: int
    settled_events: int
    eligible_stops: int
    settled_stops: int
    eligible_turns: int
    settled_turns: int
    max_course_progress_m: float
    measurement: PanelMeasurement
    gait_metrics: tuple[tuple[str, float], ...] = field(default=())

    @property
    def survived(self) -> bool:
        """The certificate's survival: the horizon reached and no fall."""
        return bool(self.full_horizon and not self.fall)

    @property
    def horizon_s(self) -> float:
        return self.horizon * self.dt

    @property
    def speed_ratio(self) -> float:
        """Mean speed along the desired heading per mean requested speed after :data:`SPEED_SETTLE_S`; NaN unmeasured."""
        if not (math.isfinite(self.mean_desired_speed_mps) and self.mean_desired_speed_mps > 0.0):
            return math.nan
        return self.mean_speed_along_heading_mps / self.mean_desired_speed_mps

    @property
    def event_settle_fraction(self) -> float:
        """Settled eligible events per eligible event; 0 with none eligible (the certificate's reading)."""
        return self.settled_events / self.eligible_events if self.eligible_events else 0.0

    @property
    def stop_settle_fraction(self) -> float:
        return self.settled_stops / self.eligible_stops if self.eligible_stops else 0.0

    @property
    def turn_settle_fraction(self) -> float:
        return self.settled_turns / self.eligible_turns if self.eligible_turns else 0.0

    @property
    def stop_exposed(self) -> bool:
        return self.eligible_stops > 0

    @property
    def turn_exposed(self) -> bool:
        return self.eligible_turns > 0

    def as_dict(self) -> dict[str, Any]:
        """The record, JSON-ready (NaN as ``None``), with the measurement and gait metrics as objects."""
        payload: dict[str, Any] = {}
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name == "measurement":
                value = value.as_dict()
            elif spec.name == "gait_metrics":
                value = {key: _json_number(metric) for key, metric in value}
            elif isinstance(value, float):
                value = _json_number(value)
            payload[spec.name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> CommandTrackingEpisode:
        """The inverse of :meth:`as_dict`, strict, so a recorded panel can be re-judged from its rows alone.

        Exactly the record's fields, each of its type: a whole number for a
        count (a float that is whole is read as one, as JSON may write it), a
        boolean for a flag, a string family, a string or ``None``
        termination reason, a number for a float (``None`` reads as NaN, the
        unmeasured value :meth:`as_dict` wrote as ``None``; the control step
        must be finite and positive), the measurement through
        :meth:`PanelMeasurement.from_dict` and the gait metrics as a mapping
        of metric name to number or ``None``, ordered as
        ``LocomotionEpisodeMetrics`` orders them.  Anything else is a
        ``ValueError`` naming the field: a row read with a guessed field
        would be re-judged as an episode nobody rolled.
        """
        if not isinstance(payload, Mapping):
            raise ValueError(f"an episode record is an object, not {type(payload).__name__}")
        expected = {spec.name for spec in fields(cls)}
        if set(payload) != expected:
            missing, extra = sorted(expected - set(payload)), sorted(set(payload) - expected)
            raise ValueError(f"an episode record has exactly the record's fields (missing {missing}, extra {extra})")
        values: dict[str, Any] = {}
        for name in _EPISODE_INT_FIELDS:
            values[name] = _record_whole(payload[name], field=name)
        for name in _EPISODE_BOOL_FIELDS:
            if not isinstance(payload[name], bool):
                raise ValueError(f"{name} is a boolean, not {payload[name]!r}")
            values[name] = payload[name]
        for name in _EPISODE_FLOAT_FIELDS:
            values[name] = _record_number(payload[name], field=name, allow_unmeasured=True)
        dt = _record_number(payload["dt"], field="dt")
        if dt <= 0.0:
            raise ValueError(f"dt is a positive control step, not {payload['dt']!r}")
        values["dt"] = dt
        if not isinstance(payload["family"], str):
            raise ValueError(f"family is a string, not {payload['family']!r}")
        values["family"] = payload["family"]
        reason = payload["termination_reason"]
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"termination_reason is a string or null, not {reason!r}")
        values["termination_reason"] = reason
        values["measurement"] = PanelMeasurement.from_dict(payload["measurement"])
        gait = payload["gait_metrics"]
        if not isinstance(gait, Mapping) or not all(isinstance(key, str) for key in gait):
            raise ValueError(f"gait_metrics maps metric names to numbers, not {gait!r}")
        order = {name: index for index, name in enumerate(LOCOMOTION_METRIC_FIELDS)}
        values["gait_metrics"] = tuple(
            (key, _record_number(gait[key], field=f"gait metric {key}", allow_unmeasured=True))
            for key in sorted(gait, key=lambda key: (order.get(key, len(order)), key))
        )
        return cls(**values)


_EPISODE_INT_FIELDS: tuple[str, ...] = (
    "episode",
    "seed",
    "length",
    "horizon",
    "planned_event_count",
    "eligible_events",
    "settled_events",
    "eligible_stops",
    "settled_stops",
    "eligible_turns",
    "settled_turns",
)
_EPISODE_BOOL_FIELDS: tuple[str, ...] = ("full_horizon", "fall", "first_event_settled")
_EPISODE_FLOAT_FIELDS: tuple[str, ...] = (
    "reward",
    "desired_speed",
    "initial_heading_error_rad",
    "tracking_fraction",
    "tracking_fraction_after_settle",
    "mean_speed_along_heading_mps",
    "mean_desired_speed_mps",
    "stand_hold_fraction",
    "max_course_progress_m",
)


def _record_whole(value: Any, *, field: str) -> int:
    """A recorded count: an int, or a float that is whole (JSON's spelling); never a bool."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field} is a whole number, not {value!r}")
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{field} is a whole number, not {value!r}")
    return int(number)


if {
    *_EPISODE_INT_FIELDS,
    *_EPISODE_BOOL_FIELDS,
    *_EPISODE_FLOAT_FIELDS,
    "dt",
    "family",
    "termination_reason",
    "measurement",
    "gait_metrics",
} != {spec.name for spec in fields(CommandTrackingEpisode)}:  # pragma: no cover - import-time contract
    raise RuntimeError("CommandTrackingEpisode.from_dict must read every field of the record exactly once")


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def sample_in_tolerance(row: Mapping[str, Any], measurement: PanelMeasurement) -> bool:
    """``tracking_metrics``'s in-tolerance test of one step, at *measurement*'s tolerances.

    Active (a positive requested speed): the velocity and yaw-rate errors of
    the executed command within their tolerances, the heading error within
    the heading tolerance and the speed error against the request within the
    velocity tolerance.  At a stop: the speed within the stop-speed
    tolerance and ``|yaw rate|`` within the stop yaw-rate tolerance.  A step
    missing a field the test or ``summarize_episode``'s tracking check reads
    is not in tolerance.
    """
    actual_speed = _finite(row.get("actual_speed"))
    yaw_rate = _finite(row.get("actual_yaw_rate"))
    error_v = _finite(row.get("tracking_error_v"))
    error_yaw = _finite(row.get("tracking_error_yaw"))
    speed = _finite(row.get("desired_speed"))
    if actual_speed is None or yaw_rate is None or error_v is None or error_yaw is None or speed is None:
        return False
    tolerance = measurement.tolerance
    if speed > 0.0:
        heading = _finite(row.get("tracking_error_heading"))
        requested = _finite(row.get("tracking_error_requested_speed"))
        desired, actual = _finite(row.get("desired_heading")), _finite(row.get("actual_heading"))
        if heading is None or requested is None or desired is None or actual is None:
            return False
        return (
            error_v <= tolerance("velocity_tolerance")
            and error_yaw <= tolerance("yaw_rate_tolerance")
            and heading <= tolerance("heading_tolerance")
            and requested <= tolerance("velocity_tolerance")
        )
    return actual_speed <= tolerance("stop_speed_tolerance") and abs(yaw_rate) <= tolerance("stop_yaw_rate_tolerance")


def speed_along_heading(row: Mapping[str, Any]) -> float:
    """The root's planar speed along the requested heading at the end of the step, NaN when unrecorded.

    ``actual_v_x`` / ``actual_v_y`` are the velocity in the heading frame of
    ``actual_heading`` (``tracking_metrics``), so the component along the
    desired heading is ``v_x cos(e) + v_y sin(e)`` with
    ``e = wrap(desired_heading - actual_heading)``.
    """
    v_x, v_y = _finite(row.get("actual_v_x")), _finite(row.get("actual_v_y"))
    desired, actual = _finite(row.get("desired_heading")), _finite(row.get("actual_heading"))
    if v_x is None or v_y is None or desired is None or actual is None:
        return math.nan
    error = wrap_angle(desired - actual)
    return v_x * math.cos(error) + v_y * math.sin(error)


def reduce_command_episode(
    rows: Sequence[Mapping[str, Any]],
    *,
    episode: int,
    seed: int,
    dt: float,
    horizon: int,
    planned_events: Sequence[Mapping[str, Any]],
    initial_heading: float,
    measurement: PanelMeasurement,
    gait_metrics: Mapping[str, float] | None = None,
) -> CommandTrackingEpisode:
    """Reduce one rolled episode's post-step info rows to its :class:`CommandTrackingEpisode`.

    *rows* are the step infos (the live command's ``as_info`` and
    ``tracking_metrics`` keys) with ``time_s`` (the end of the step),
    ``reward``, ``terminated``, ``truncated`` and ``course_progress_m``
    added, as the panel runner records them.  *planned_events* is the
    command schedule planned over the horizon right after reset (its first
    event names the family); *initial_heading* the root's heading at reset,
    which the first event's initial heading error is measured from.  Each
    row's in-tolerance flag is recomputed at *measurement*'s tolerances
    before ``summarize_episode`` reads it, so a gate override applies to
    every statistic.  Raises ``ValueError`` for rows ``summarize_episode``
    refuses or an empty schedule.
    """
    if not planned_events:
        raise ValueError("a panel episode plans at least the command event its reset draws")
    flags = [sample_in_tolerance(row, measurement) for row in rows]
    judged = [{**dict(row), "tracking_in_tolerance": flag} for row, flag in zip(rows, flags)]
    summary = summarize_episode(
        judged,
        dt,
        planned_events=planned_events,
        horizon_steps=horizon,
        initial_heading=initial_heading,
        yaw_rate_max=measurement.yaw_rate_max,
        settling_base_s=SETTLE_BASE_S,
        dwell_s=DWELL_S,
    )
    events = summary["events"]
    by_id = {int(event["event_id"]): event for event in events}

    after_settle: list[bool] = []
    along: list[float] = []
    requested: list[float] = []
    hold: list[bool] = []
    for row, flag in zip(judged, flags):
        time_s = float(row["time_s"])
        event = by_id.get(int(row["command_event_id"]))
        if event is not None and time_s >= event["start_time_s"] + event["settling_allowance_s"]:
            after_settle.append(flag)
        speed = _finite(row.get("desired_speed"))
        if speed is not None and speed > 0.0 and time_s > SPEED_SETTLE_S + 1e-9:
            along.append(speed_along_heading(row))
            requested.append(speed)
        if time_s > STAND_SETTLE_S + 1e-9:
            hold.append(flag)

    eligible = [event for event in events if event["eligible"]]
    stops = [event for event in eligible if not event["heading_active"]]
    turns: list[Mapping[str, Any]] = []
    previous_heading: float | None = None
    for event in events:
        heading = float(event["desired_heading"])
        if (
            previous_heading is not None
            and event["eligible"]
            and event["heading_active"]
            and abs(wrap_angle(heading - previous_heading)) > TURN_HEADING_CHANGE_RAD
        ):
            turns.append(event)
        previous_heading = heading

    first = events[0]
    desired_speed = float(planned_events[0]["desired_speed"])
    tracking = summary["tracking_fraction"]
    reason = summary["termination_reason"]
    return CommandTrackingEpisode(
        episode=int(episode),
        seed=int(seed),
        family=STAND_FAMILY if desired_speed == 0.0 else CRUISE_FAMILY,
        length=int(summary["steps"]),
        horizon=int(horizon),
        dt=float(dt),
        full_horizon=bool(summary["full_horizon"]),
        fall=bool(summary["fall"]),
        termination_reason=None if reason is None else str(reason),
        reward=float(summary["reward"]),
        desired_speed=desired_speed,
        # The events summarize_episode keeps: planned or observed, starting before the horizon.
        planned_event_count=len(events),
        initial_heading_error_rad=float(first["initial_heading_error_rad"]),
        first_event_settled=bool(first["settled"]),
        tracking_fraction=math.nan if tracking is None else float(tracking),
        tracking_fraction_after_settle=_mean([float(flag) for flag in after_settle]),
        mean_speed_along_heading_mps=_mean(along),
        mean_desired_speed_mps=_mean(requested),
        stand_hold_fraction=_mean([float(flag) for flag in hold]),
        eligible_events=len(eligible),
        settled_events=sum(1 for event in eligible if event["settled"]),
        eligible_stops=len(stops),
        settled_stops=sum(1 for event in stops if event["settled"]),
        eligible_turns=len(turns),
        settled_turns=sum(1 for event in turns if event["settled"]),
        max_course_progress_m=float(summary["max_course_progress_m"]),
        measurement=measurement,
        gait_metrics=tuple((str(key), float(value)) for key, value in (gait_metrics or {}).items()),
    )


# ── per-episode classification ───────────────────────────────────────────────


def _bar_failure(key: str, metric: str, direction: str, value: float, bar: float) -> str | None:
    """The reason *value* fails the bar, or ``None`` when it holds (NaN never holds)."""
    if not math.isfinite(value):
        return f"{key}: {metric} is unmeasured ({value!r})"
    if direction == "min" and value < bar:
        return f"{key}: {metric} {value:.6g} < {bar:.6g}"
    if direction == "max" and value > bar:
        return f"{key}: {metric} {value:.6g} > {bar:.6g}"
    return None


def _horizon_failure(episode: CommandTrackingEpisode) -> str | None:
    if episode.survived:
        return None
    ending = episode.termination_reason or ("terminated" if not episode.full_horizon else "fall")
    return f"{HORIZON_REASON}: length {episode.length} < {episode.horizon} or a fall ({ending})"


def gait_failures(episode: CommandTrackingEpisode, clause: GaitClause) -> list[str]:
    """Every reason *episode* fails the gait clause, each ``"<key>: ..."``; empty when it holds.

    The episode must have been measured on the clause's number of limbs (an
    episode with no gait metrics, or measured on another animal, fails
    ``gait_limbs``), and every declared bar must hold with its metric finite:
    an unmeasured metric (no gait metrics at all, or NaN: too few touchdowns
    for an alternation index, no stride for the lead swaps) fails its bar.
    """
    metrics = dict(episode.gait_metrics)
    spec = {key: (metric, direction) for key, metric, direction in GAIT_CRITERIA}
    reasons: list[str] = []
    limbs = metrics.get(GAIT_LIMBS_METRIC, math.nan)
    if not (math.isfinite(limbs) and limbs == clause.limbs):
        reasons.append(
            f"{GAIT_LIMBS_KEY}: the episode was measured on {limbs!r} limbs and the gait clause judges {clause.limbs}"
        )
    for key, bar in clause.bars:
        metric, direction = spec[key]
        failure = _bar_failure(key, metric, direction, float(metrics.get(metric, math.nan)), float(bar))
        if failure is not None:
            reasons.append(failure)
    return reasons


def _gait_limb_failures(episodes: Sequence[CommandTrackingEpisode], clause: GaitClause) -> list[str]:
    """The panel-level refusal of episodes measured on another number of limbs than the gait clause judges.

    Every episode is checked, the stand family's too (whose gait the walk
    profile does not judge): a panel measured on another animal than the
    stage's clause describes answers another question.  An episode with no
    gait metrics is not refused here; its gait bars fail it per episode.
    """
    other = [
        episode.episode
        for episode in episodes
        if GAIT_LIMBS_METRIC in dict(episode.gait_metrics)
        and dict(episode.gait_metrics)[GAIT_LIMBS_METRIC] != clause.limbs
    ]
    if not other:
        return []
    return [
        f"{len(other)} of {len(episodes)} episodes were measured on another number of limbs than the gait clause's "
        f"{GAIT_LIMBS_KEY} = {clause.limbs} (episodes {other[:3]}); the stage declares another animal's clause"
    ]


def classify_cruise_episode(episode: CommandTrackingEpisode, thresholds: WalkThresholds) -> tuple[str, ...]:
    """Every reason a cruise episode is not a success, each ``"<key>: ..."``; empty when it is one."""
    reasons: list[str] = []
    horizon = _horizon_failure(episode)
    if horizon is not None:
        reasons.append(horizon)
    if not episode.first_event_settled:
        allowance = SETTLE_BASE_S + episode.initial_heading_error_rad / episode.measurement.yaw_rate_max
        reasons.append(
            f"{HEADING_SETTLE_REASON}: the initial heading error {episode.initial_heading_error_rad:.4g} rad did "
            f"not settle within {allowance:.3g} s and hold for {DWELL_S:g} s"
        )
    for key, metric, value in (
        ("min_cruise_tracking_fraction", "tracking_fraction_after_settle", episode.tracking_fraction_after_settle),
        ("min_cruise_speed_ratio", "speed_ratio", episode.speed_ratio),
    ):
        failure = _bar_failure(key, metric, "min", value, float(getattr(thresholds, key)))
        if failure is not None:
            reasons.append(failure)
    reasons.extend(gait_failures(episode, thresholds.gait))
    return tuple(reasons)


def classify_stand_episode(episode: CommandTrackingEpisode, thresholds: WalkThresholds) -> tuple[str, ...]:
    """Every reason a stand episode is not a success, each ``"<key>: ..."``; empty when it is one."""
    reasons: list[str] = []
    horizon = _horizon_failure(episode)
    if horizon is not None:
        reasons.append(horizon)
    if not episode.first_event_settled:
        reasons.append(
            f"{STOP_SETTLE_REASON}: the stop did not settle within {SETTLE_BASE_S:g} s and hold for {DWELL_S:g} s"
        )
    failure = _bar_failure(
        "min_stand_hold_fraction",
        "stand_hold_fraction",
        "min",
        episode.stand_hold_fraction,
        thresholds.min_stand_hold_fraction,
    )
    if failure is not None:
        reasons.append(failure)
    return tuple(reasons)


def classify_follow_episode(
    episode: CommandTrackingEpisode, thresholds: FollowThresholds, *, require_stop: bool, require_turn: bool
) -> tuple[str, ...]:
    """Every reason a follow episode is not a success (the certificate's), each ``"<key>: ..."``; empty when it is one.

    A stop (turn) bar applies only when the schedule draws stops (turns),
    *require_stop* (*require_turn*), and the episode was exposed to one.
    """
    reasons: list[str] = []
    horizon = _horizon_failure(episode)
    if horizon is not None:
        reasons.append(horizon)
    checks: list[tuple[str, str, float, float]] = [
        (
            "min_follow_tracking_fraction",
            "tracking_fraction",
            episode.tracking_fraction,
            thresholds.min_follow_tracking_fraction,
        ),
        (
            "min_follow_event_settle_fraction",
            "event_settle_fraction",
            episode.event_settle_fraction,
            thresholds.min_follow_event_settle_fraction,
        ),
    ]
    if thresholds.min_follow_course_fraction is not None and thresholds.follow_course_distance_m is not None:
        progress = min(1.0, episode.max_course_progress_m / thresholds.follow_course_distance_m)
        checks.append(
            ("min_follow_course_fraction", "course_fraction", progress, thresholds.min_follow_course_fraction)
        )
    if require_stop and episode.stop_exposed:
        checks.append(
            (
                "min_follow_stop_settle_fraction",
                "stop_settle_fraction",
                episode.stop_settle_fraction,
                thresholds.min_follow_stop_settle_fraction,
            )
        )
    if require_turn and episode.turn_exposed:
        checks.append(
            (
                "min_follow_turn_settle_fraction",
                "turn_settle_fraction",
                episode.turn_settle_fraction,
                thresholds.min_follow_turn_settle_fraction,
            )
        )
    for key, metric, value, bar in checks:
        failure = _bar_failure(key, metric, "min", value, bar)
        if failure is not None:
            reasons.append(failure)
    reasons.extend(gait_failures(episode, thresholds.gait))
    return tuple(reasons)


def reason_key(reason: str) -> str:
    """The criterion key a classification reason names."""
    return reason.split(":", 1)[0]


# ── the verdict ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FamilyResult:
    """One family's numbers: survival, success and the per-criterion failure counts.

    ``survival_lcb`` and ``success_lcb`` are ``None`` where the family is
    judged on a fraction instead (the stand family).  ``command_exposure``
    (follow) holds ``(label, exposed episodes, successful episodes,
    fraction)`` for stops and turns.
    """

    family: str
    n_episodes: int
    n_survived: int
    n_success: int
    success_fraction: float
    survival_lcb: float | None = None
    success_lcb: float | None = None
    criterion_failures: tuple[tuple[str, int], ...] = ()
    command_exposure: tuple[tuple[str, int, int, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_episodes": self.n_episodes,
            "n_survived": self.n_survived,
            "n_success": self.n_success,
            "success_fraction": _json_number(self.success_fraction),
            "survival_lcb": None if self.survival_lcb is None else _json_number(self.survival_lcb),
            "success_lcb": None if self.success_lcb is None else _json_number(self.success_lcb),
            "criterion_failures": dict(self.criterion_failures),
            "command_exposure": {
                label: {"exposed_episodes": exposed, "successful_episodes": settled, "success_fraction": fraction}
                for label, exposed, settled, fraction in self.command_exposure
            },
        }


@dataclass(frozen=True)
class CommandTrackingResult:
    """One panel's verdict under a profile and the numbers it was reached on.

    ``episode_reasons`` is positionally aligned with the episodes judged
    (empty for a success).  ``paired_delta_lcb`` and ``n_pairs`` are set when
    the follow profile's reference comparison ran.
    """

    passed: bool
    failures: tuple[str, ...]
    profile: str
    n_episodes: int
    families: tuple[FamilyResult, ...]
    episode_reasons: tuple[tuple[str, ...], ...] = ()
    paired_delta_lcb: float | None = None
    n_pairs: int | None = None

    def family(self, name: str) -> FamilyResult:
        return next(result for result in self.families if result.family == name)

    def as_dict(self) -> dict[str, Any]:
        """The panel-level numbers, JSON-ready (the per-episode reasons travel with the episode rows)."""
        payload: dict[str, Any] = {
            "passed": self.passed,
            "failures": list(self.failures),
            "profile": self.profile,
            "n_episodes": self.n_episodes,
            "families": [result.as_dict() for result in self.families],
        }
        if self.n_pairs is not None:
            payload["paired_delta_lcb"] = None if self.paired_delta_lcb is None else _json_number(self.paired_delta_lcb)
            payload["n_pairs"] = self.n_pairs
        return payload


#: The rule a stage's handoff is chosen by, per profile, as the choice record
#: states it: the gate's own verdict and bounds on a panel of each candidate
#: checkpoint, never its reward (:func:`handoff_selection_key`).
HANDOFF_SELECTION_RULES: dict[str, str] = {
    WALK_PROFILE: (
        "passed, then cruise success LCB95, then stand success fraction, then cruise survival LCB95 "
        "(larger first); ties keep the candidate order"
    ),
    FOLLOW_PROFILE: (
        "passed, then follow success LCB95, then follow survival LCB95 (larger first); ties keep the candidate order"
    ),
}


def handoff_selection_key(result: CommandTrackingResult) -> tuple[float, ...]:
    """How a candidate checkpoint's panel ranks for the handoff: by the gated statistic, larger first.

    ``(passed, bound, ...)``: a candidate whose panel passes the gate before
    one that fails it, then the profile's certifying bound (the walk
    profile's cruise success LCB95, the follow profile's success LCB95),
    then the walk stand family's success fraction, then the survival bound.
    A value the panel did not measure (a family that is absent, an empty
    family's NaN) ranks below every measured one.  Never the reward: a
    checkpoint whose mean return is higher but which tracks the command less
    reliably is the one the gate exists to refuse.  The order of the keys is
    :data:`HANDOFF_SELECTION_RULES`.
    """

    def measured(value: float | None) -> float:
        return float(value) if value is not None and math.isfinite(value) else -math.inf

    def family(name: str) -> FamilyResult | None:
        return next((entry for entry in result.families if entry.family == name), None)

    passed = 1.0 if result.passed else 0.0
    if result.profile == WALK_PROFILE:
        cruise, stand = family(CRUISE_FAMILY), family(STAND_FAMILY)
        return (
            passed,
            measured(None if cruise is None else cruise.success_lcb),
            measured(None if stand is None else stand.success_fraction),
            measured(None if cruise is None else cruise.survival_lcb),
        )
    follow = family(FOLLOW_PROFILE)
    return (
        passed,
        measured(None if follow is None else follow.success_lcb),
        measured(None if follow is None else follow.survival_lcb),
    )


def _tally(reasons: Iterable[tuple[str, ...]]) -> tuple[tuple[str, int], ...]:
    tally: dict[str, int] = {}
    for episode_reasons in reasons:
        for key in dict.fromkeys(reason_key(reason) for reason in episode_reasons):
            tally[key] = tally.get(key, 0) + 1
    return tuple(tally.items())


def _lcb(successes: int, trials: int) -> float:
    return binomial_lcb(successes, trials) if trials else math.nan


def _measurement_failures(episodes: Sequence[CommandTrackingEpisode], measurement: PanelMeasurement) -> list[str]:
    """The panel-level refusals every profile shares: the measurement, the seeds and the family labels."""
    failures: list[str] = []
    other = [index for index, episode in enumerate(episodes) if episode.measurement != measurement]
    if other:
        shown = ", ".join(str(index) for index in other[:3])
        failures.append(
            f"{len(other)} of {len(episodes)} episodes were reduced under another measurement than the gate's "
            f"({measurement.as_dict()}): episodes {shown}{', ...' if len(other) > 3 else ''}; their fractions "
            "answer a different question"
        )
    seeds = [episode.seed for episode in episodes]
    if len(set(seeds)) != len(seeds):
        failures.append("the panel repeats a reset seed, so an episode would be counted twice")
    mislabelled = [
        episode.episode
        for episode in episodes
        if episode.family not in FAMILIES or (episode.family == STAND_FAMILY) != (episode.desired_speed == 0.0)
    ]
    if mislabelled:
        failures.append(
            f"episodes {mislabelled[:3]} carry a family that disagrees with their reset command (stand is exactly "
            "a zero requested speed)"
        )
    return failures


def evaluate_walk_profile(
    episodes: Iterable[CommandTrackingEpisode], thresholds: WalkThresholds, *, measurement: PanelMeasurement
) -> CommandTrackingResult:
    """Judge a walk panel: the cruise family's two bounds, the stand family's falls and fraction, both minimums.

    *measurement* is the stage's (:func:`panel_measurement`).  Fails closed,
    naming each failure: a family with fewer episodes than its minimum, an
    episode reduced under another measurement, a repeated seed, a family
    label that disagrees with the reset command, an episode whose schedule
    switches mid-episode, a NaN bound or fraction.
    """
    rows = list(episodes)
    failures = _measurement_failures(rows, measurement)
    failures.extend(_gait_limb_failures(rows, thresholds.gait))
    switching = [episode.episode for episode in rows if episode.planned_event_count != 1]
    if switching:
        failures.append(
            f"{len(switching)} episodes plan more than one command event within the horizon (episodes "
            f"{switching[:3]}): the walk profile judges a whole-episode command, and a schedule that switches "
            "mid-episode is judged by the follow profile"
        )
    reasons: list[tuple[str, ...]] = []
    for episode in rows:
        if episode.family == CRUISE_FAMILY:
            reasons.append(classify_cruise_episode(episode, thresholds))
        elif episode.family == STAND_FAMILY:
            reasons.append(classify_stand_episode(episode, thresholds))
        else:
            reasons.append((f"family: {episode.family!r} is not a walk family",))

    cruise = [index for index, episode in enumerate(rows) if episode.family == CRUISE_FAMILY]
    stand = [index for index, episode in enumerate(rows) if episode.family == STAND_FAMILY]

    n_cruise = len(cruise)
    if n_cruise < thresholds.min_cruise_episodes:
        failures.append(
            f"cruise family: {n_cruise} episodes < min_cruise_episodes {thresholds.min_cruise_episodes} (the family "
            "is short, so the gate fails closed)"
        )
    cruise_survived = sum(1 for index in cruise if rows[index].survived)
    cruise_success = sum(1 for index in cruise if not reasons[index])
    survival_lcb = _lcb(cruise_survived, n_cruise)
    success_lcb = _lcb(cruise_success, n_cruise)
    if not (math.isfinite(survival_lcb) and survival_lcb >= thresholds.min_cruise_survival_lcb):
        failures.append(
            f"cruise family: survival_lcb {survival_lcb:.4f} < {thresholds.min_cruise_survival_lcb:.4f} "
            f"({cruise_survived}/{n_cruise} episodes reached the horizon)"
        )
    if not (math.isfinite(success_lcb) and success_lcb >= thresholds.min_cruise_success_lcb):
        failures.append(
            f"cruise family: success_lcb {success_lcb:.4f} < {thresholds.min_cruise_success_lcb:.4f} "
            f"({cruise_success}/{n_cruise} episodes tracked the command)"
        )

    n_stand = len(stand)
    if n_stand < thresholds.min_stand_episodes:
        failures.append(
            f"stand family: {n_stand} episodes < min_stand_episodes {thresholds.min_stand_episodes} (the family is "
            "short, so the gate fails closed)"
        )
    stand_survived = sum(1 for index in stand if rows[index].survived)
    stand_falls = n_stand - stand_survived
    stand_success = sum(1 for index in stand if not reasons[index])
    stand_fraction = stand_success / n_stand if n_stand else math.nan
    if stand_falls > thresholds.max_stand_falls:
        failures.append(
            f"stand family: {stand_falls} episodes fell or ended before the horizon > max_stand_falls "
            f"{thresholds.max_stand_falls}"
        )
    if not (math.isfinite(stand_fraction) and stand_fraction >= thresholds.min_stand_success_fraction):
        failures.append(
            f"stand family: success_fraction {stand_fraction:.4f} < {thresholds.min_stand_success_fraction:.4f} "
            f"({stand_success}/{n_stand} episodes held the stop)"
        )

    families = (
        FamilyResult(
            family=CRUISE_FAMILY,
            n_episodes=n_cruise,
            n_survived=cruise_survived,
            n_success=cruise_success,
            success_fraction=cruise_success / n_cruise if n_cruise else math.nan,
            survival_lcb=survival_lcb,
            success_lcb=success_lcb,
            criterion_failures=_tally(reasons[index] for index in cruise),
        ),
        FamilyResult(
            family=STAND_FAMILY,
            n_episodes=n_stand,
            n_survived=stand_survived,
            n_success=stand_success,
            success_fraction=stand_fraction,
            criterion_failures=_tally(reasons[index] for index in stand),
        ),
    )
    return CommandTrackingResult(
        passed=not failures,
        failures=tuple(failures),
        profile=WALK_PROFILE,
        n_episodes=len(rows),
        families=families,
        episode_reasons=tuple(reasons),
    )


def paired_success_differences(
    episodes: Sequence[CommandTrackingEpisode],
    reference: Sequence[CommandTrackingEpisode],
    *,
    success: Any,
) -> tuple[list[float], list[str]]:
    """Per-seed ``success(policy) - success(reference)`` over the same seeds, and every refusal.

    *success* maps an episode to a bool (the profile's own per-episode
    success).  The two panels must hold the same seeds, once each, reduced
    under the same measurement, and each seed's two records must agree on
    the recorded reset-draw fields: the family, the requested speed, the
    initial heading error and the planned event count, at the same horizon
    and control step (:func:`_same_paired_task`).  Otherwise the differences
    are empty and the refusals say why.  The differences are in the policy
    panel's order.

    The pairing binds only what the records show.  A schedule difference no
    record carries (another ``turn_increment_max``, which moves the later
    events' headings, or other command-adapter gains) leaves every one of
    those fields equal, so it is bound only by the reference being rolled
    on the stage's own env (the report rolls both panels on the stage's
    config, and the judge binds the reference to the stage's parent
    checkpoint), not by the pairing.
    """
    refusals: list[str] = []
    policy_seeds = [episode.seed for episode in episodes]
    reference_seeds = [episode.seed for episode in reference]
    if len(set(reference_seeds)) != len(reference_seeds):
        refusals.append("the reference panel repeats a reset seed")
    if set(policy_seeds) != set(reference_seeds):
        refusals.append(
            f"the reference panel was rolled on other seeds ({len(set(reference_seeds) - set(policy_seeds))} extra, "
            f"{len(set(policy_seeds) - set(reference_seeds))} missing); a paired difference needs the same seeds"
        )
    measurements = {episode.measurement for episode in (*episodes, *reference)}
    if len(measurements) > 1:
        refusals.append("the reference panel was reduced under another measurement than the policy's")
    by_seed = {episode.seed: episode for episode in reference}
    unpaired = [
        episode.seed
        for episode in episodes
        if episode.seed in by_seed and not _same_paired_task(episode, by_seed[episode.seed])
    ]
    if unpaired:
        refusals.append(
            f"the reference panel ran another task than the policy's on {len(unpaired)} seeds (first {unpaired[0]}): "
            "a pair must share its family, requested speed, initial heading error, planned event count, horizon and "
            "control step"
        )
    if refusals:
        return [], refusals
    return [float(bool(success(episode))) - float(bool(success(by_seed[episode.seed]))) for episode in episodes], []


def _same_paired_task(policy: CommandTrackingEpisode, reference: CommandTrackingEpisode) -> bool:
    """Whether two records of one seed agree on the recorded reset-draw fields.

    The fields compared are the family, the requested speed, the initial
    heading error and the planned event count, with the horizon and the
    control step.  A schedule difference no record shows (``turn_increment_max``,
    the command-adapter gains) passes this test; it is bound only by the
    reference being rolled on the stage's own env
    (:func:`paired_success_differences`), not here.
    """
    return (
        policy.family == reference.family
        and policy.planned_event_count == reference.planned_event_count
        and policy.horizon == reference.horizon
        and abs(policy.dt - reference.dt) <= 1e-9
        and abs(policy.desired_speed - reference.desired_speed) <= 1e-9
        and abs(policy.initial_heading_error_rad - reference.initial_heading_error_rad) <= 1e-9
    )


def evaluate_follow_profile(
    episodes: Iterable[CommandTrackingEpisode],
    thresholds: FollowThresholds,
    *,
    measurement: PanelMeasurement,
    require_stop: bool,
    require_turn: bool,
    reference: Sequence[CommandTrackingEpisode] | None = None,
) -> CommandTrackingResult:
    """Judge a follow panel with the behavior certificate's statistics, and the paired reference when declared.

    *require_stop* / *require_turn* say whether the stage's schedule draws
    stops (``stop_probability > 0``) and turns (``turn_increment_max > 0``),
    as the certificate reads them off the command config.  An episode whose
    horizon is shorter than ``min_follow_horizon_s`` is not a valid sample:
    it is named and left out of the counts.  *reference* is the reference
    policy's panel on the same seeds; with ``min_follow_reference_delta_lcb``
    declared and no reference, the gate fails closed.  The pairing is on
    follow success on this panel: a non-inferiority bound on the follow
    task, not a test that straight walking was kept (the module docstring).
    """
    rows = list(episodes)
    failures = _measurement_failures(rows, measurement)
    failures.extend(_gait_limb_failures(rows, thresholds.gait))
    short = [index for index, episode in enumerate(rows) if episode.horizon_s + 1e-9 < thresholds.min_follow_horizon_s]
    if short:
        failures.append(
            f"{len(short)} episodes have a horizon of {rows[short[0]].horizon_s:g} s < min_follow_horizon_s "
            f"{thresholds.min_follow_horizon_s:g}; they are not valid samples"
        )

    def classify(episode: CommandTrackingEpisode) -> tuple[str, ...]:
        return classify_follow_episode(episode, thresholds, require_stop=require_stop, require_turn=require_turn)

    reasons = [classify(episode) for episode in rows]
    valid = [episode for index, episode in enumerate(rows) if index not in short]
    valid_reasons = [episode_reasons for index, episode_reasons in enumerate(reasons) if index not in short]
    n = len(valid)
    if n < thresholds.min_follow_episodes:
        failures.append(f"{n} valid episodes < min_follow_episodes {thresholds.min_follow_episodes}")
    survived = sum(1 for episode in valid if episode.survived)
    successful = sum(1 for episode_reasons in valid_reasons if not episode_reasons)
    survival_lcb = _lcb(survived, n)
    success_lcb = _lcb(successful, n)
    if not (math.isfinite(survival_lcb) and survival_lcb >= thresholds.min_follow_survival_lcb):
        failures.append(f"survival_lcb {survival_lcb:.4f} < {thresholds.min_follow_survival_lcb:.4f} ({survived}/{n})")
    if not (math.isfinite(success_lcb) and success_lcb >= thresholds.min_follow_success_lcb):
        failures.append(f"success_lcb {success_lcb:.4f} < {thresholds.min_follow_success_lcb:.4f} ({successful}/{n})")

    exposure: list[tuple[str, int, int, float]] = []
    for label, required, bar, exposed_of, fraction_of in (
        ("stop", require_stop, thresholds.min_follow_stop_settle_fraction, "stop_exposed", "stop_settle_fraction"),
        ("turn", require_turn, thresholds.min_follow_turn_settle_fraction, "turn_exposed", "turn_settle_fraction"),
    ):
        exposed = [episode for episode in valid if getattr(episode, exposed_of)]
        settled = sum(1 for episode in exposed if episode.survived and getattr(episode, fraction_of) >= bar)
        fraction = settled / len(exposed) if exposed else 0.0
        exposure.append((label, len(exposed), settled, fraction))
        if required and len(exposed) < thresholds.min_follow_command_episodes:
            failures.append(
                f"{len(exposed)} episodes with an eligible {label} < min_follow_command_episodes "
                f"{thresholds.min_follow_command_episodes}"
            )
        if required and fraction < bar:
            failures.append(f"{label} success fraction {fraction:.4f} < {bar:.4f} ({settled}/{len(exposed)})")

    paired_lcb: float | None = None
    n_pairs: int | None = None
    if thresholds.min_follow_reference_delta_lcb is not None:
        n_pairs = 0
        if reference is None:
            failures.append(
                "the reference panel is missing while min_follow_reference_delta_lcb is declared; the paired "
                "comparison fails closed rather than being skipped"
            )
        else:
            differences, refusals = paired_success_differences(
                rows, list(reference), success=lambda episode: not classify(episode)
            )
            failures.extend(refusals)
            n_pairs = len(differences)
            if not refusals:
                paired_lcb = paired_difference_lcb(differences)
                if not (math.isfinite(paired_lcb) and paired_lcb >= thresholds.min_follow_reference_delta_lcb):
                    failures.append(
                        f"paired_delta_lcb {paired_lcb:.4f} < {thresholds.min_follow_reference_delta_lcb:.4f} "
                        f"(policy minus reference success over {n_pairs} seeds)"
                    )

    family = FamilyResult(
        family=FOLLOW_PROFILE,
        n_episodes=n,
        n_survived=survived,
        n_success=successful,
        success_fraction=successful / n if n else math.nan,
        survival_lcb=survival_lcb,
        success_lcb=success_lcb,
        criterion_failures=_tally(valid_reasons),
        command_exposure=tuple(exposure),
    )
    return CommandTrackingResult(
        passed=not failures,
        failures=tuple(failures),
        profile=FOLLOW_PROFILE,
        n_episodes=len(rows),
        families=(family,),
        episode_reasons=tuple(reasons),
        paired_delta_lcb=paired_lcb,
        n_pairs=n_pairs,
    )


def evaluate_command_tracking_gate(
    episodes: Iterable[CommandTrackingEpisode],
    thresholds: CommandTrackingThresholds,
    *,
    command_config: Any,
    reference: Sequence[CommandTrackingEpisode] | None = None,
) -> CommandTrackingResult:
    """Judge a panel under the profile *thresholds* declares, against the stage's *command_config*.

    *command_config* is the stage's ``DirectionCommandConfig`` (the env's
    ``command_config`` kwarg): it gives the tolerances, ``yaw_rate_max`` and,
    for the follow profile, whether the schedule draws stops and turns.  A
    stage with no live command (``None``) fails closed: there is nothing to
    track.  *reference* is the follow profile's paired reference panel.
    """
    rows = list(episodes)
    try:
        command_config = command_config_from(command_config)
    except (TypeError, ValueError) as exc:
        return CommandTrackingResult(
            passed=False,
            failures=(f"the stage's command config is unreadable: {exc}",),
            profile=thresholds.profile,
            n_episodes=len(rows),
            families=(),
        )
    if command_config is None:
        return CommandTrackingResult(
            passed=False,
            failures=(
                f"{COMMAND_TRACKING_GATE_KIND} judges a live command, and the stage has no command config "
                '(command_mode "none")',
            ),
            profile=thresholds.profile,
            n_episodes=len(rows),
            families=(),
        )
    measurement = panel_measurement(command_config, thresholds)
    if isinstance(thresholds, WalkThresholds):
        return evaluate_walk_profile(rows, thresholds, measurement=measurement)
    return evaluate_follow_profile(
        rows,
        thresholds,
        measurement=measurement,
        require_stop=float(command_config.stop_probability) > 0.0,
        require_turn=float(command_config.turn_increment_max) > 0.0,
        reference=reference,
    )


# ── the descriptive panel summary ────────────────────────────────────────────


def summarize_command_panel(episodes: Iterable[CommandTrackingEpisode]) -> dict[str, Any]:
    """A descriptive summary of a panel, per family and overall: no verdict.

    Counts, survivors, falls with their termination reasons, and the means
    of the tracking fractions, the speed ratio, the stand hold fraction and
    each gait criterion's metric over the episodes that measured them (an
    episode's NaN is left out here, and counted as ``unmeasured``; the gate
    itself fails it).
    """
    rows = list(episodes)

    def block(selected: list[CommandTrackingEpisode]) -> dict[str, Any]:
        def mean_of(values: Iterable[float]) -> dict[str, Any]:
            listed = list(values)
            finite = [value for value in listed if math.isfinite(value)]
            return {
                "mean": _json_number(math.fsum(finite) / len(finite)) if finite else None,
                "unmeasured": len(listed) - len(finite),
            }

        reasons: dict[str, int] = {}
        for episode in selected:
            if not episode.survived:
                label = episode.termination_reason or "ended_early"
                reasons[label] = reasons.get(label, 0) + 1
        return {
            "n_episodes": len(selected),
            "n_survived": sum(1 for episode in selected if episode.survived),
            "termination_reasons": reasons,
            "seeds": [episode.seed for episode in selected],
            "tracking_fraction": mean_of(episode.tracking_fraction for episode in selected),
            "tracking_fraction_after_settle": mean_of(episode.tracking_fraction_after_settle for episode in selected),
            "first_event_settled": sum(1 for episode in selected if episode.first_event_settled),
            "speed_ratio": mean_of(episode.speed_ratio for episode in selected if episode.family == CRUISE_FAMILY),
            "stand_hold_fraction": mean_of(
                episode.stand_hold_fraction for episode in selected if episode.family == STAND_FAMILY
            ),
            "mean_reward": _json_number(_mean([episode.reward for episode in selected])),
            # The gait clause's metrics, by criterion metric: an episode without gait metrics counts as unmeasured.
            "gait": {
                metric: mean_of(float(dict(episode.gait_metrics).get(metric, math.nan)) for episode in selected)
                for metric in dict.fromkeys(metric for _, metric, _ in GAIT_CRITERIA)
            },
        }

    return {
        "protocol": COMMAND_PANEL_PROTOCOL,
        "all": block(rows),
        "families": {family: block([episode for episode in rows if episode.family == family]) for family in FAMILIES},
    }
