"""Physical locomotion certification from a declared, per-episode gait panel.

The reward gate remains unchanged. ``locomotion_gait/v2`` certifies the joint
event of completing the episode, making forward progress, every limb (and, on
four legs, every girdle) really stepping and bearing weight, real strides that
clear the floor and step through, and the declared gait holding over the
analysis window within one off-gait budget, without excess flight, skidding,
body support or foot-on-foot support. Missing telemetry fails the episode;
missing or stale panel evidence fails the gate. This module is pure: it
judges stored metrics only (see ``gait/metrics.py``), so a reader re-judging
``metrics_json`` under today's declared bars reproduces the in-process
verdict exactly. Checkpoint and evidence binding belongs to reporting.

The profiles are walk-first. ``biped_walk`` and ``quadruped_walk`` certify a
walk: contralateral limbs alternate, every limb is down roughly half the
stride or more, the feet carry the body (little or no flight, measured on
contact and on load) and planted feet do not skate; on four legs any
symmetrical walking gait counts (lateral- or diagonal-sequence walks and
diagonal- or lateral-couplet walks alike), with no limb-phase partition, as
long as the girdles step together. ``biped_alternating`` is the lenient
run-allowed alternating profile for stages whose speed bar asks for running.
Every profile requires step-through along the line of progression, strides
whose two steps are not grossly lopsided, and no crabbing step-to. Trot,
pace, gallop, run and jump profiles are deferred; Hildebrand's labels stay
report-only diagnostics.

Every criterion is explicit: there are no silent defaults in a declared gate.
``provisional_gait_criteria`` returns the calibrated development values used
for report-only panels; a certification config must still spell out each bar,
and no bar may be declared more lenient than its ``SCHEMA_LIMITS`` entry, so
a schema-valid declaration cannot switch a profile's defining rails off.
Feet are paired by name ((r, l); (fr, fl), (rr, rl)), never by position.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..record_fields import is_sha256_digest
from .recovery_gate import binomial_lcb

GAIT_GATE_KIND = "locomotion_gait/v2"
GAIT_PROFILES = frozenset({"biped_walk", "biped_alternating", "quadruped_walk"})
#: Foot count each profile is defined for.
PROFILE_FEET = {"biped_walk": 2, "biped_alternating": 2, "quadruped_walk": 4}
#: The stored template each profile reads (``episode["templates"][...]``).
PROFILE_TEMPLATE = {"biped_walk": "alternation", "biped_alternating": "alternation", "quadruped_walk": "alternation"}
#: Duration tolerance in units of the largest recorded sample interval (D2):
#: an accumulated float clock may end a fraction of a step short of a horizon.
DURATION_TOLERANCE_SAMPLES = 1.5
#: Report-only panels without a declared profile: a stage whose speed bar asks
#: for running (Froude number of the bar at or above the walk-run transition,
#: about 0.5; Alexander 1989) defaults to the run-allowed alternating profile.
WALK_RUN_FROUDE = 0.5

_PANEL_KEYS = frozenset(
    {
        "gait_profile",
        "measurement_protocol_sha256",
        "min_eval_episodes",
        "gait_panel_seed_start",
        "min_gait_success_lcb",
    }
)
#: Bars every profile declares.
_COMMON_CRITERIA = frozenset(
    {
        "min_episode_forward_vel",
        "min_episode_duration_s",
        "max_flight_fraction",
        "max_body_support_fraction",
        "max_foot_foot_contact_fraction",
        "max_skid_fraction",
        "max_glide_stance_fraction",
        "min_trunk_height_over_leg",
        "min_limb_phase_coverage",
        "min_limb_duty",
        "min_relative_limb_load_share",
        "max_light_stance_fraction",
        "min_pair_load_ratio",
        "min_pair_duty_ratio",
        "min_complete_cycles_per_foot",
        "min_valid_swing_fraction",
        "min_median_swing_clearance_over_leg",
        "min_stride_length_over_leg",
        "max_swing_ground_fraction",
        "max_swing_slip_fraction",
        "min_step_length_over_leg",
        "min_step_through_stride_fraction",
        "min_step_symmetry",
        "min_body_frame_step_to_symmetry",
        "min_phase_locking",
        "max_alternation_phase_offset",
        "max_alternating_overlap_index",
        "max_off_gait_fraction",
    }
)
#: Bars that only some profiles consume; declaring one for another profile is
#: an error rather than dead config.
_GIRDLE_CRITERIA = frozenset({"min_girdle_load_share", "min_girdle_duty_ratio", "max_girdle_unloaded_fraction"})
_WALK_CRITERIA = frozenset({"min_walking_duty", "max_unloaded_fraction", "max_walk_glide_stance_fraction"})
_PROFILE_CRITERIA: dict[str, frozenset[str]] = {
    "biped_walk": _WALK_CRITERIA,
    "biped_alternating": frozenset(),
    "quadruped_walk": _GIRDLE_CRITERIA | _WALK_CRITERIA,
}
_OPTIONAL_CRITERIA = frozenset({"min_avg_reward"})
GAIT_REQUIRED_KEYS = _PANEL_KEYS | _COMMON_CRITERIA
GAIT_THRESHOLD_KEYS = (
    GAIT_REQUIRED_KEYS | frozenset().union(*_PROFILE_CRITERIA.values()) | _OPTIONAL_CRITERIA | {"required_consecutive"}
)
_INTEGER_KEYS = frozenset({"min_eval_episodes", "min_complete_cycles_per_foot", "gait_panel_seed_start"})
#: Keys bounded to [0, 1] (fractions, ratios, locking, LCB).
_UNIT_KEYS = frozenset(
    {
        "min_gait_success_lcb",
        "max_flight_fraction",
        "max_unloaded_fraction",
        "min_step_symmetry",
        "min_body_frame_step_to_symmetry",
        "max_body_support_fraction",
        "max_foot_foot_contact_fraction",
        "min_limb_phase_coverage",
        "min_limb_duty",
        "min_walking_duty",
        "min_pair_load_ratio",
        "min_pair_duty_ratio",
        "min_girdle_duty_ratio",
        "max_girdle_unloaded_fraction",
        "max_light_stance_fraction",
        "max_glide_stance_fraction",
        "max_walk_glide_stance_fraction",
        "min_valid_swing_fraction",
        "max_swing_ground_fraction",
        "max_swing_slip_fraction",
        "min_step_through_stride_fraction",
        "min_phase_locking",
        "max_alternating_overlap_index",
        "max_off_gait_fraction",
    }
)
#: The lighter girdle's share of the load cannot exceed one half.
_HALF_KEYS = frozenset({"min_girdle_load_share"})
#: Keys that are circular distances, bounded to [0, 0.5] cycles.
_HALF_CYCLE_KEYS = frozenset({"max_alternation_phase_offset"})
#: Keys that may be negative (a step length bar below zero is meaningless but well defined).
_SIGNED_KEYS = frozenset({"min_avg_reward", "min_step_length_over_leg"})

#: Calibrated development criteria (walk-first round 2: bakeoff DEV split,
#: re-recorded replays with the trunk quaternion, and every previous attack
#: set, the walk-first round-1 attacks included, as development cases);
#: docs/GAIT_CERTIFICATION.md tabulates each bar against the worst genuine
#: development value and the nearest pathological one.
_PROVISIONAL_COMMON: dict[str, float | int] = {
    "max_body_support_fraction": 0.01,
    "max_foot_foot_contact_fraction": 0.02,
    "max_skid_fraction": 0.35,
    "max_glide_stance_fraction": 0.10,
    "min_trunk_height_over_leg": 0.50,
    "max_light_stance_fraction": 0.18,
    "min_limb_phase_coverage": 0.80,
    "min_limb_duty": 0.10,
    "min_relative_limb_load_share": 0.36,
    "min_pair_load_ratio": 0.70,
    "min_pair_duty_ratio": 0.70,
    "min_complete_cycles_per_foot": 3,
    "min_valid_swing_fraction": 0.75,
    "min_median_swing_clearance_over_leg": 0.02,
    "min_stride_length_over_leg": 0.20,
    "max_swing_ground_fraction": 0.50,
    "max_swing_slip_fraction": 0.10,
    "min_step_length_over_leg": 0.05,
    "min_step_through_stride_fraction": 0.60,
    "min_step_symmetry": 0.10,
    "min_body_frame_step_to_symmetry": 0.35,
    "min_phase_locking": 0.50,
    "max_alternation_phase_offset": 0.15,
    "max_alternating_overlap_index": 0.50,
    "max_off_gait_fraction": 0.15,
}
#: Quadruped girdle participation: the lighter girdle carries at least this
#: share of the foot impulse, its mean duty factor is at least this fraction
#: of the other girdle's, and it is never unloaded for long.
_PROVISIONAL_GIRDLE: dict[str, float | int] = {
    "min_girdle_load_share": 0.18,
    "min_girdle_duty_ratio": 0.60,
    "max_girdle_unloaded_fraction": 0.05,
}
#: Walking support: little or no flight (on contact and on load), every limb down
#: roughly half the stride, no skating walking stance.
_PROVISIONAL_WALK: dict[str, float | int] = {
    "max_flight_fraction": 0.10,
    "max_unloaded_fraction": 0.15,
    "max_walk_glide_stance_fraction": 0.15,
    "min_walking_duty": 0.35,
}
_PROVISIONAL_PROFILE: dict[str, dict[str, float | int]] = {
    "biped_walk": dict(_PROVISIONAL_WALK),
    # running is allowed (sprint duty ~0.2, Weyand et al. 2000)
    "biped_alternating": {"max_flight_fraction": 0.65},
    # four legs: every limb down at least 0.42 of the stride (a trot or pace
    # below it has two suspensions per stride: a flying trot)
    "quadruped_walk": {**_PROVISIONAL_GIRDLE, **_PROVISIONAL_WALK, "min_walking_duty": 0.42},
}


#: The most lenient value a declared bar may take: beyond it the rail no
#: longer tests what its profile promises (a timing tolerance wide enough to
#: admit an in-phase hop, an off-gait budget larger than the gait, a duty or
#: load floor near zero, a flight cap near one, a confidence bound near zero).
#: Each limit lies between the calibrated provisional bar and the vacuous
#: value; tightening any bar is always allowed. ``(">=", x)`` is a floor's
#: lowest declarable value, ``("<=", x)`` a cap's highest.
SCHEMA_LIMITS: dict[str, tuple[str, float]] = {
    "min_gait_success_lcb": (">=", 0.5),
    "min_limb_phase_coverage": (">=", 0.5),
    "min_limb_duty": (">=", 0.05),
    "min_walking_duty": (">=", 0.3),
    "min_relative_limb_load_share": (">=", 0.15),
    "max_light_stance_fraction": ("<=", 0.4),
    "min_pair_load_ratio": (">=", 0.5),
    "min_pair_duty_ratio": (">=", 0.5),
    "min_complete_cycles_per_foot": (">=", 3),
    "min_valid_swing_fraction": (">=", 0.5),
    "min_median_swing_clearance_over_leg": (">=", 0.005),
    "min_stride_length_over_leg": (">=", 0.1),
    "max_swing_ground_fraction": ("<=", 0.75),
    "max_swing_slip_fraction": ("<=", 0.3),
    "min_step_length_over_leg": (">=", 0.01),
    "min_step_through_stride_fraction": (">=", 0.5),
    "min_step_symmetry": (">=", 0.05),
    "min_body_frame_step_to_symmetry": (">=", 0.2),
    "max_unloaded_fraction": ("<=", 0.3),
    "max_body_support_fraction": ("<=", 0.1),
    "max_foot_foot_contact_fraction": ("<=", 0.1),
    "max_skid_fraction": ("<=", 0.5),
    "max_glide_stance_fraction": ("<=", 0.3),
    "max_walk_glide_stance_fraction": ("<=", 0.3),
    "min_trunk_height_over_leg": (">=", 0.3),
    "min_girdle_load_share": (">=", 0.1),
    "min_girdle_duty_ratio": (">=", 0.4),
    "max_girdle_unloaded_fraction": ("<=", 0.2),
    "min_phase_locking": (">=", 0.3),
    "max_alternation_phase_offset": ("<=", 0.2),
    "max_alternating_overlap_index": ("<=", 0.75),
    "max_off_gait_fraction": ("<=", 0.35),
}
#: Flight is a walking-support bar on the walk profiles and a run allowance
#: on the run-allowed one.
FLIGHT_LIMITS = {"biped_walk": 0.2, "quadruped_walk": 0.2, "biped_alternating": 0.8}


def default_gait_profile(n_feet: int, speed_bar_mps: float, leg_length_m: float) -> str:
    """Report-only default: a walk, unless a biped stage's speed bar asks for running.

    A quadruped defaults to ``quadruped_walk``. A biped defaults to
    ``biped_walk`` unless the bar's Froude number ``v^2 / (g L)`` reaches the
    walk-run transition (``WALK_RUN_FROUDE``), as the velociraptor stage's
    2.0 m/s does (Froude 0.82); then ``biped_alternating``. A certification
    config always declares its profile.
    """
    if n_feet == 4:
        return "quadruped_walk"
    if n_feet != 2:
        raise ValueError(f"{GAIT_GATE_KIND} has no profile for {n_feet} feet")
    if not (math.isfinite(leg_length_m) and leg_length_m > 0.0):
        raise ValueError("leg length must be positive and finite")
    froude = max(float(speed_bar_mps), 0.0) ** 2 / (9.81 * leg_length_m)
    return "biped_alternating" if froude >= WALK_RUN_FROUDE else "biped_walk"


def provisional_gait_criteria(profile: str) -> dict[str, float | int]:
    """Calibrated development bars for one profile (report-only panels, adapters)."""
    if profile not in GAIT_PROFILES:
        raise ValueError(f"{GAIT_GATE_KIND} unknown gait_profile {profile!r}")
    return {**_PROVISIONAL_COMMON, **_PROVISIONAL_PROFILE[profile]}


def _number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{GAIT_GATE_KIND} {key} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class GaitGateThresholds:
    gait_profile: str
    measurement_protocol_sha256: str
    min_eval_episodes: int
    gait_panel_seed_start: int
    min_gait_success_lcb: float
    min_episode_forward_vel: float
    min_episode_duration_s: float
    max_flight_fraction: float
    max_body_support_fraction: float
    max_foot_foot_contact_fraction: float
    max_skid_fraction: float
    max_glide_stance_fraction: float
    min_trunk_height_over_leg: float
    min_limb_phase_coverage: float
    min_limb_duty: float
    min_relative_limb_load_share: float
    max_light_stance_fraction: float
    min_pair_load_ratio: float
    min_pair_duty_ratio: float
    min_complete_cycles_per_foot: int
    min_valid_swing_fraction: float
    min_median_swing_clearance_over_leg: float
    min_stride_length_over_leg: float
    max_swing_ground_fraction: float
    max_swing_slip_fraction: float
    min_step_length_over_leg: float
    min_step_through_stride_fraction: float
    min_step_symmetry: float
    min_body_frame_step_to_symmetry: float
    min_phase_locking: float
    max_alternation_phase_offset: float
    max_alternating_overlap_index: float
    max_off_gait_fraction: float
    min_walking_duty: float | None = None
    max_unloaded_fraction: float | None = None
    max_walk_glide_stance_fraction: float | None = None
    min_girdle_load_share: float | None = None
    min_girdle_duty_ratio: float | None = None
    max_girdle_unloaded_fraction: float | None = None
    min_avg_reward: float | None = None

    @classmethod
    def from_curriculum(cls, curriculum: Mapping[str, Any]) -> GaitGateThresholds:
        """Require an explicit protocol, gait, panel, and every physical criterion."""
        missing = sorted(GAIT_REQUIRED_KEYS - set(curriculum))
        if missing:
            raise ValueError(f"{GAIT_GATE_KIND} missing required thresholds: {missing}")
        profile = curriculum["gait_profile"]
        if not isinstance(profile, str) or profile not in GAIT_PROFILES:
            raise ValueError(f"{GAIT_GATE_KIND} unknown gait_profile {profile!r}")
        needed = _PROFILE_CRITERIA[profile]
        missing = sorted(needed - set(curriculum))
        if missing:
            raise ValueError(f"{GAIT_GATE_KIND} {profile} missing required thresholds: {missing}")
        foreign = sorted((frozenset().union(*_PROFILE_CRITERIA.values()) - needed) & set(curriculum))
        if foreign:
            raise ValueError(f"{GAIT_GATE_KIND} {profile} does not consume thresholds {foreign}")
        digest = curriculum["measurement_protocol_sha256"]
        if not is_sha256_digest(digest):
            raise ValueError(f"{GAIT_GATE_KIND} measurement_protocol_sha256 must be a sha256:<hex> digest")
        values: dict[str, Any] = {"gait_profile": profile, "measurement_protocol_sha256": digest}
        for key in sorted(
            GAIT_THRESHOLD_KEYS - {"gait_profile", "measurement_protocol_sha256", "required_consecutive"}
        ):
            if key not in curriculum:
                continue
            if key in _INTEGER_KEYS:
                # A count is an integer, not a float that equals one: 40.0
                # would validate here and then never match a rolled panel.
                value = curriculum[key]
                minimum = 0 if key == "gait_panel_seed_start" else 1
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{GAIT_GATE_KIND} {key} must be an integer >= {minimum}, got {value!r}")
                values[key] = value
                continue
            number = _number(curriculum[key], key)
            if key not in _SIGNED_KEYS and number < 0:
                raise ValueError(f"{GAIT_GATE_KIND} {key} must be nonnegative")
            if key in _UNIT_KEYS and number > 1:
                raise ValueError(f"{GAIT_GATE_KIND} {key} must be in [0, 1]")
            if key in _HALF_CYCLE_KEYS and number > 0.5:
                raise ValueError(f"{GAIT_GATE_KIND} {key} must be in [0, 0.5] cycles")
            if key in _HALF_KEYS and number > 0.5:
                raise ValueError(f"{GAIT_GATE_KIND} {key} must be in [0, 0.5]")
            values[key] = number
        if values["min_episode_duration_s"] <= 0:
            raise ValueError(f"{GAIT_GATE_KIND} min_episode_duration_s must be positive")
        if values["min_gait_success_lcb"] <= 0:
            raise ValueError(f"{GAIT_GATE_KIND} min_gait_success_lcb must be positive")
        if "required_consecutive" in curriculum:
            consecutive = curriculum["required_consecutive"]
            if isinstance(consecutive, bool) or not isinstance(consecutive, int) or consecutive < 1:
                raise ValueError(f"{GAIT_GATE_KIND} required_consecutive must be a positive integer")
        limits = {**SCHEMA_LIMITS, "max_flight_fraction": ("<=", FLIGHT_LIMITS[profile])}
        for key, (op, limit) in sorted(limits.items()):
            if key not in values:
                continue
            if values[key] < limit if op == ">=" else values[key] > limit:
                raise ValueError(
                    f"{GAIT_GATE_KIND} {profile} {key} {values[key]!r} is vacuous; the declared bar must be "
                    f"{op} {limit:g} for the profile to mean what it says"
                )
        return cls(**values)


def _metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


#: Rail -> physical cause group (reasons read ``group/rail: value op bar``).
RAIL_GROUPS = {
    "telemetry_valid": "episode",
    "completed_horizon": "episode",
    "duration_s": "episode",
    "mean_speed_mps": "episode",
    "limb_phase_coverage_min": "participation",
    "limb_duty_min": "participation",
    "relative_limb_load_share_min": "participation",
    "pair_load_ratio_min": "participation",
    "pair_duty_ratio_min": "participation",
    "girdle_load_share_min": "participation",
    "girdle_duty_ratio": "participation",
    "girdle_unloaded_fraction": "participation",
    "light_stance_fraction_max": "participation",
    "complete_cycles_min": "participation",
    "valid_swing_fraction_min": "stepping",
    "median_swing_clearance_over_leg_min": "stepping",
    "stride_length_over_leg_min": "stepping",
    "swing_ground_fraction_max": "stepping",
    "swing_slip_fraction_max": "stepping",
    "step_length_over_leg_min": "stepping",
    "step_through_stride_fraction_min": "stepping",
    "step_symmetry": "stepping",
    "body_frame_step_to_symmetry": "stepping",
    "walking_duty": "support",
    "flight_fraction": "support",
    "unloaded_fraction": "support",
    "body_support_fraction": "support",
    "foot_foot_contact_fraction": "support",
    "skid_fraction_max": "support",
    "glide_stance_fraction_max": "support",
    "walk_glide_stance_fraction_max": "support",
    "trunk_height_over_leg_p10": "support",
    "phase_locking_min": "coupling",
    "alternation_phase_offset_max": "coupling",
    "alternating_overlap_index_max": "coupling",
    "off_gait_fraction": "persistence",
}
#: Where to look up which limb or pair is worst, for the reason text.
_WORST = {
    "limb_phase_coverage_min": ("per_foot", "phase_coverage", min),
    "limb_duty_min": ("per_foot", "duty_factor", min),
    "relative_limb_load_share_min": ("per_foot", "relative_load_share", min),
    "complete_cycles_min": ("per_foot", "complete_cycles", min),
    "valid_swing_fraction_min": ("per_foot", "valid_swing_fraction", min),
    "median_swing_clearance_over_leg_min": ("per_foot", "median_swing_clearance_over_leg", min),
    "stride_length_over_leg_min": ("per_foot", "stride_length_over_leg_median", min),
    "swing_ground_fraction_max": ("per_foot", "swing_ground_fraction", max),
    "swing_slip_fraction_max": ("per_foot", "swing_slip_fraction", max),
    "skid_fraction_max": ("per_foot", "skid_fraction", max),
    "glide_stance_fraction_max": ("per_foot", "glide_stance_fraction", max),
    "walk_glide_stance_fraction_max": ("per_foot", "walk_glide_stance_fraction", max),
    "light_stance_fraction_max": ("per_foot", "light_stance_fraction", max),
    "step_length_over_leg_min": ("per_foot", "step_length_over_leg_median", min),
    "step_through_stride_fraction_min": ("contralateral", "step_through_stride_fraction", min),
    "pair_load_ratio_min": ("contralateral", "load_ratio", min),
    "pair_duty_ratio_min": ("contralateral", "duty_ratio", min),
}


def rail_id(reason: str) -> str:
    """Stable ``group/rail`` identifier of a failure reason (aggregation key)."""
    return reason.split(":", 1)[0]


def _where(episode: Mapping[str, Any], key: str) -> str:
    spec = _WORST.get(key)
    if spec is None:
        return ""
    table, field, pick = spec
    records = episode.get(table)
    if not isinstance(records, Mapping):
        return ""
    scored = [
        (value, name)
        for name, record in records.items()
        if isinstance(record, Mapping)
        for value in [_metric(record.get(field))]
        if value is not None
    ]
    if not scored:
        return ""
    return f" ({pick(scored)[1]})"


def _check(
    failures: list[str],
    episode: Mapping[str, Any],
    source: Mapping[str, Any],
    key: str,
    threshold: float,
    *,
    floor: bool = False,
    allow_negative: bool = False,
    label: str | None = None,
) -> None:
    name = f"{RAIL_GROUPS.get(label or key, 'other')}/{label or key}"
    value = _metric(source.get(key))
    if value is None:
        failures.append(f"{name}: unmeasured")
    elif (value < 0 and not allow_negative) or (key.endswith("fraction") and value > 1):
        failures.append(f"{name}: {value:.6g} is outside its physical range")
    elif value < threshold if floor else value > threshold:
        failures.append(f"{name}: {value:.6g}{_where(episode, key)} {'<' if floor else '>'} {threshold:.6g}")


def classify_gait_episode(
    episode: Mapping[str, Any], thresholds: GaitGateThresholds, *, foot_names: Sequence[str]
) -> tuple[bool, tuple[str, ...]]:
    """Qualify one episode from its stored metrics. Foot order is (r,l) or (fr,fl,rr,rl).

    Reasons are ``group/rail: value op bar``; ``rail_id`` extracts the stable
    prefix. Coupling and persistence rails read the profile's stored template
    (``episode["templates"][PROFILE_TEMPLATE[profile]]``).
    """
    failures: list[str] = []
    profile = thresholds.gait_profile
    expected = ("r", "l") if PROFILE_FEET[profile] == 2 else ("fr", "fl", "rr", "rl")
    if len(foot_names) != len(expected) or sorted(foot_names) != sorted(expected):
        return False, (f"episode/foot_names: {profile} requires the distinct feet {expected}",)
    if episode.get("telemetry_valid") is not True:
        return False, ("episode/telemetry_valid: telemetry is invalid or missing",)
    if episode.get("completed_horizon") is not True:
        failures.append("episode/completed_horizon: episode did not complete its horizon")
    duration = _metric(episode.get("duration_s"))
    interval = _metric(episode.get("max_sample_interval_s"))
    if duration is None or interval is None:
        failures.append("episode/duration_s: unmeasured")
    elif duration + DURATION_TOLERANCE_SAMPLES * interval < thresholds.min_episode_duration_s:
        failures.append(f"episode/duration_s: {duration:.6g} < {thresholds.min_episode_duration_s:.6g}")
    _check(
        failures,
        episode,
        episode,
        "mean_speed_mps",
        thresholds.min_episode_forward_vel,
        floor=True,
        allow_negative=True,
    )
    for key, bar, floor in (
        ("limb_phase_coverage_min", thresholds.min_limb_phase_coverage, True),
        ("limb_duty_min", thresholds.min_limb_duty, True),
        ("relative_limb_load_share_min", thresholds.min_relative_limb_load_share, True),
        ("light_stance_fraction_max", thresholds.max_light_stance_fraction, False),
        ("pair_load_ratio_min", thresholds.min_pair_load_ratio, True),
        ("pair_duty_ratio_min", thresholds.min_pair_duty_ratio, True),
        ("complete_cycles_min", thresholds.min_complete_cycles_per_foot, True),
        ("valid_swing_fraction_min", thresholds.min_valid_swing_fraction, True),
        ("median_swing_clearance_over_leg_min", thresholds.min_median_swing_clearance_over_leg, True),
        ("stride_length_over_leg_min", thresholds.min_stride_length_over_leg, True),
        ("swing_ground_fraction_max", thresholds.max_swing_ground_fraction, False),
        ("swing_slip_fraction_max", thresholds.max_swing_slip_fraction, False),
        ("flight_fraction", thresholds.max_flight_fraction, False),
        ("body_support_fraction", thresholds.max_body_support_fraction, False),
        ("foot_foot_contact_fraction", thresholds.max_foot_foot_contact_fraction, False),
        ("skid_fraction_max", thresholds.max_skid_fraction, False),
        ("glide_stance_fraction_max", thresholds.max_glide_stance_fraction, False),
        ("trunk_height_over_leg_p10", thresholds.min_trunk_height_over_leg, True),
    ):
        _check(failures, episode, episode, key, bar, floor=floor)
    # Step-through: every foot's median step lands ahead of the other foot's
    # previous footprint along the line of progression (step-to gaits never
    # do), in most strides both feet step through (a step-to gait that
    # switches its leading foot fails one step of every stride), and the two
    # steps of a stride are of comparable length in the travel or the trunk
    # frame (a step-to dressed up by a crab or a trunk yaw is lopsided in both).
    _check(
        failures,
        episode,
        episode,
        "step_length_over_leg_min",
        thresholds.min_step_length_over_leg,
        floor=True,
        allow_negative=True,
    )
    _check(
        failures,
        episode,
        episode,
        "step_through_stride_fraction_min",
        thresholds.min_step_through_stride_fraction,
        floor=True,
    )
    _check(failures, episode, episode, "step_symmetry", thresholds.min_step_symmetry, floor=True)
    if episode.get("body_frame_step_to_symmetry") is not None:
        # A foot lands behind the other along the trunk axis (feet together in
        # the body frame) while the travel frame shows a step-through: only
        # even travel steps make that a crab walk rather than a crabbing step-to.
        _check(
            failures,
            episode,
            episode,
            "body_frame_step_to_symmetry",
            thresholds.min_body_frame_step_to_symmetry,
            floor=True,
        )
    if (
        thresholds.min_walking_duty is not None
        and thresholds.max_unloaded_fraction is not None
        and thresholds.max_walk_glide_stance_fraction is not None
    ):
        # Walking support: every limb down roughly half the stride or more,
        # the feet carry the body (no ballistic phase, light toe contacts
        # bridging one included), and planted feet do not skate.
        _check(
            failures, episode, episode, "limb_duty_min", thresholds.min_walking_duty, floor=True, label="walking_duty"
        )
        _check(failures, episode, episode, "unloaded_fraction", thresholds.max_unloaded_fraction)
        _check(failures, episode, episode, "walk_glide_stance_fraction_max", thresholds.max_walk_glide_stance_fraction)
    if PROFILE_FEET[profile] == 4:
        assert thresholds.min_girdle_load_share is not None and thresholds.min_girdle_duty_ratio is not None
        assert thresholds.max_girdle_unloaded_fraction is not None
        _check(failures, episode, episode, "girdle_load_share_min", thresholds.min_girdle_load_share, floor=True)
        _check(failures, episode, episode, "girdle_duty_ratio", thresholds.min_girdle_duty_ratio, floor=True)
        _check(failures, episode, episode, "girdle_unloaded_fraction", thresholds.max_girdle_unloaded_fraction)
    templates = episode.get("templates")
    name = PROFILE_TEMPLATE[profile]
    template = templates.get(name) if isinstance(templates, Mapping) else None
    if not isinstance(template, Mapping):
        failures.append(f"persistence/template: {name} template is unmeasured")
        return not failures, tuple(failures)
    _check(failures, episode, template, "phase_locking_min", thresholds.min_phase_locking, floor=True)
    _check(failures, episode, template, "alternation_phase_offset_max", thresholds.max_alternation_phase_offset)
    if template.get("alternating_overlap_index_max") is not None:
        # undefined only at extreme duty factors, which other rails judge
        _check(failures, episode, template, "alternating_overlap_index_max", thresholds.max_alternating_overlap_index)
    # Persistence: one budget, a fraction of the window, for every way of not
    # being in the declared gait (a limb not cycling or pausing, a pair locked
    # off template, a sustained uncoordinated bout, a step-to bout).
    _check(failures, episode, template, "off_gait_fraction", thresholds.max_off_gait_fraction)
    return not failures, tuple(failures)


#: Plain-language cause of each failing rail, appended to the gait label.
_CAUSES = {
    "episode/completed_horizon": "incomplete episode",
    "episode/duration_s": "short episode",
    "episode/mean_speed_mps": "too slow",
    "participation/limb_phase_coverage_min": "limb not cycling",
    "participation/limb_duty_min": "limb not bearing weight",
    "participation/relative_limb_load_share_min": "limb not bearing weight",
    "participation/pair_load_ratio_min": "asymmetric loading",
    "participation/pair_duty_ratio_min": "asymmetric stance",
    "participation/girdle_load_share_min": "girdle not bearing weight",
    "participation/girdle_duty_ratio": "girdle barely in stance",
    "participation/girdle_unloaded_fraction": "girdle unloaded for stretches (rearing or wheelbarrowing)",
    "participation/light_stance_fraction_max": "stance without bearing weight",
    "participation/complete_cycles_min": "too few strides",
    "stepping/valid_swing_fraction_min": "shuffle (invalid swings)",
    "stepping/median_swing_clearance_over_leg_min": "shuffle (low swings)",
    "stepping/stride_length_over_leg_min": "shuffle (short strides)",
    "stepping/swing_ground_fraction_max": "foot drag (swing at the floor)",
    "stepping/swing_slip_fraction_max": "foot drag (sliding swing)",
    "stepping/step_length_over_leg_min": "step-to (a foot does not step past the other)",
    "stepping/step_through_stride_fraction_min": "step-to strides (a foot lands level with or behind the other)",
    "stepping/step_symmetry": "lopsided steps (one step much shorter than the other)",
    "stepping/body_frame_step_to_symmetry": "step-to in the body frame on a crabbing path",
    "support/walking_duty": "not walking (short stance)",
    "support/flight_fraction": "too much flight",
    "support/unloaded_fraction": "feet not carrying the body (ballistic phases)",
    "support/body_support_fraction": "body-supported",
    "support/foot_foot_contact_fraction": "feet stacked",
    "support/skid_fraction_max": "skidding",
    "support/glide_stance_fraction_max": "skating stances",
    "support/walk_glide_stance_fraction_max": "skating stances",
    "support/trunk_height_over_leg_p10": "crouched or kneeling trunk",
    "coupling/phase_locking_min": "limbs not phase-locked",
    "coupling/alternation_phase_offset_max": "asymmetric pair timing",
    "coupling/alternating_overlap_index_max": "feet loaded together",
    "persistence/off_gait_fraction": "off-gait bouts",
}


def describe_gait_episode(episode: Mapping[str, Any], failures: Sequence[str] = ()) -> str | None:
    """Gait label (Hildebrand vocabulary) plus the physical causes of any failure."""
    label = episode.get("gait_label")
    if not isinstance(label, str):
        return None
    causes: list[str] = []
    for reason in failures:
        cause = _CAUSES.get(rail_id(reason))
        if cause is not None and cause not in causes:
            causes.append(cause)
    return label if not causes else f"{label}; " + "; ".join(causes)


@dataclass(frozen=True)
class GaitGateResult:
    passed: bool
    failures: tuple[str, ...]
    success_count: int
    n_episodes: int
    success_fraction: float
    success_lcb: float
    episode_failures: tuple[tuple[str, ...], ...]


def evaluate_gait_gate(
    episodes: Sequence[Mapping[str, Any]], thresholds: GaitGateThresholds, *, foot_names: Sequence[str]
) -> GaitGateResult:
    """Certify the joint event at the fixed declared n with an exact 95% LCB.

    Despite its shared legacy name, min_eval_episodes is an exact panel size
    for this kind. Optional stopping or extending a failing panel is refused.
    """
    classified = [classify_gait_episode(episode, thresholds, foot_names=foot_names) for episode in episodes]
    n = len(classified)
    k = sum(passed for passed, _ in classified)
    bound = binomial_lcb(k, n) if n else math.nan
    failures: list[str] = []
    if n != thresholds.min_eval_episodes:
        failures.append(f"n_episodes {n} != declared fixed panel size {thresholds.min_eval_episodes}")
    if not math.isfinite(bound) or bound < thresholds.min_gait_success_lcb:
        failures.append(f"gait_success_lcb {bound:.4f} < {thresholds.min_gait_success_lcb:.4f} ({k}/{n} qualified)")
    if thresholds.min_avg_reward is not None:
        rewards = [_metric(episode.get("reward")) for episode in episodes]
        if not rewards or any(reward is None for reward in rewards):
            failures.append("panel reward rail is unmeasured")
        elif sum(reward for reward in rewards if reward is not None) / n < thresholds.min_avg_reward:
            failures.append(f"panel mean reward < min_avg_reward {thresholds.min_avg_reward:.6g}")
    return GaitGateResult(
        passed=not failures,
        failures=tuple(failures),
        success_count=k,
        n_episodes=n,
        success_fraction=k / n if n else math.nan,
        success_lcb=bound,
        episode_failures=tuple(reasons for _, reasons in classified),
    )
