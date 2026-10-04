"""Physical locomotion certification from a declared, per-episode gait panel.

The reward gate remains unchanged. ``locomotion_gait/v2`` certifies the joint
event of completing the episode, making forward progress, every limb (and, on
four legs, every girdle) really stepping and bearing weight, real strides that
clear the floor, and the declared gait template holding persistently over the
analysis window without excess off-gait time, flight, skidding, body support
or foot-on-foot support. Missing telemetry fails the episode; missing or stale panel evidence
fails the gate. This module is pure: it judges stored metrics only (see
``gait/metrics.py``), so a reader re-judging ``metrics_json`` under today's
declared bars reproduces the in-process verdict exactly. Checkpoint and
evidence binding belongs to reporting.

Every criterion is explicit: there are no silent defaults in a declared gate.
``provisional_gait_criteria`` returns the calibrated development values used
for report-only panels; a certification config must still spell out each bar.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..record_fields import is_sha256_digest
from .recovery_gate import binomial_lcb

GAIT_GATE_KIND = "locomotion_gait/v2"
GAIT_PROFILES = frozenset({"biped_alternating", "quadruped_walk", "quadruped_trot", "quadruped_pace"})
#: Duration tolerance in units of the largest recorded sample interval (D2):
#: an accumulated float clock may end a fraction of a step short of a horizon.
DURATION_TOLERANCE_SAMPLES = 1.5

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
        "min_limb_phase_coverage",
        "min_limb_duty",
        "min_relative_limb_load_share",
        "min_pair_load_ratio",
        "min_pair_duty_ratio",
        "min_complete_cycles_per_foot",
        "min_valid_swing_fraction",
        "min_median_swing_clearance_over_leg",
        "min_stride_length_over_leg",
        "max_swing_ground_fraction",
        "max_swing_slip_fraction",
        "min_lead_exchange_fraction",
        "min_phase_locking",
        "max_alternation_phase_offset",
        "max_alternating_overlap_index",
        "min_template_coverage",
        "max_off_gait_fraction",
    }
)
#: Bars that only some profiles consume; declaring one for another profile is
#: an error rather than dead config.
_GIRDLE_CRITERIA = frozenset({"min_girdle_load_share", "min_girdle_duty_ratio"})
_PROFILE_CRITERIA: dict[str, frozenset[str]] = {
    "biped_alternating": frozenset(),
    "quadruped_walk": _GIRDLE_CRITERIA
    | {"min_walk_limb_phase", "max_walk_limb_phase", "max_off_gait_strides", "max_off_gait_fraction_ceiling"},
    "quadruped_trot": _GIRDLE_CRITERIA | {"max_synchrony_phase_offset", "min_synchronous_overlap_index"},
    "quadruped_pace": _GIRDLE_CRITERIA | {"max_synchrony_phase_offset", "min_synchronous_overlap_index"},
}
_OPTIONAL_CRITERIA = frozenset({"min_step_length_over_leg", "min_avg_reward"})
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
        "max_body_support_fraction",
        "max_foot_foot_contact_fraction",
        "min_limb_phase_coverage",
        "min_limb_duty",
        "min_pair_load_ratio",
        "min_pair_duty_ratio",
        "min_girdle_duty_ratio",
        "min_valid_swing_fraction",
        "max_swing_ground_fraction",
        "max_swing_slip_fraction",
        "min_lead_exchange_fraction",
        "min_phase_locking",
        "max_alternating_overlap_index",
        "min_synchronous_overlap_index",
        "min_template_coverage",
        "max_off_gait_fraction",
        "max_off_gait_fraction_ceiling",
    }
)
#: The lighter girdle's share of the load cannot exceed one half.
_HALF_KEYS = frozenset({"min_girdle_load_share"})
#: Keys that are circular distances, bounded to [0, 0.5] cycles.
_HALF_CYCLE_KEYS = frozenset(
    {"max_alternation_phase_offset", "max_synchrony_phase_offset", "min_walk_limb_phase", "max_walk_limb_phase"}
)

#: Calibrated development criteria (bakeoff DEV split and the verifier's
#: attack traces as negatives, 2026-10 hardening round 1); the derivation of
#: each bar is in docs/GAIT_CERTIFICATION.md.
_PROVISIONAL_COMMON: dict[str, float | int] = {
    "max_body_support_fraction": 0.01,
    "max_foot_foot_contact_fraction": 0.02,
    "max_skid_fraction": 0.35,
    "min_limb_phase_coverage": 0.80,
    "min_limb_duty": 0.10,
    "min_relative_limb_load_share": 0.36,
    "min_pair_load_ratio": 0.70,
    "min_pair_duty_ratio": 0.70,
    "min_complete_cycles_per_foot": 3,
    "min_valid_swing_fraction": 0.75,
    "min_median_swing_clearance_over_leg": 0.02,
    "min_stride_length_over_leg": 0.20,
    "max_swing_ground_fraction": 0.40,
    "max_swing_slip_fraction": 0.10,
    "min_lead_exchange_fraction": 0.15,
    "min_phase_locking": 0.60,
    "max_alternation_phase_offset": 0.09,
    "max_alternating_overlap_index": 0.50,
    "min_template_coverage": 0.70,
    "max_off_gait_fraction": 0.05,
}
#: Quadruped girdle participation: the lighter girdle carries at least this
#: share of the foot impulse, and its mean duty factor is at least this
#: fraction of the other girdle's.
_PROVISIONAL_GIRDLE: dict[str, float | int] = {"min_girdle_load_share": 0.18, "min_girdle_duty_ratio": 0.60}
_PROVISIONAL_PROFILE: dict[str, dict[str, float | int]] = {
    # aerial running is a run, not a defect (sprint duty ~0.2, Weyand et al. 2000)
    "biped_alternating": {"max_flight_fraction": 0.65},
    # walks and ambles have no suspension (Hildebrand 1976); 5 % absorbs jitter-made gaps
    # A walk's competing templates (pace 0, trot 1/2) lie only 0.05 cycles beyond
    # its on-template band, so one mistimed footfall of a long-stride walk can
    # lock there for most of a stride: up to 0.9 stride of off-gait time is
    # forgiven, never more than 15 % of the window.
    "quadruped_walk": {
        **_PROVISIONAL_GIRDLE,
        "max_flight_fraction": 0.05,
        "min_walk_limb_phase": 0.125,
        "max_walk_limb_phase": 0.375,
        "max_off_gait_strides": 0.9,
        "max_off_gait_fraction_ceiling": 0.15,
    },
    # flying trots and paces are trots and paces
    "quadruped_trot": {
        **_PROVISIONAL_GIRDLE,
        "max_flight_fraction": 0.50,
        "max_synchrony_phase_offset": 0.125,
        "min_synchronous_overlap_index": 0.50,
    },
    "quadruped_pace": {
        **_PROVISIONAL_GIRDLE,
        "max_flight_fraction": 0.50,
        "max_synchrony_phase_offset": 0.125,
        "min_synchronous_overlap_index": 0.50,
    },
}


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
    min_limb_phase_coverage: float
    min_limb_duty: float
    min_relative_limb_load_share: float
    min_pair_load_ratio: float
    min_pair_duty_ratio: float
    min_complete_cycles_per_foot: int
    min_valid_swing_fraction: float
    min_median_swing_clearance_over_leg: float
    min_stride_length_over_leg: float
    max_swing_ground_fraction: float
    max_swing_slip_fraction: float
    min_lead_exchange_fraction: float
    min_phase_locking: float
    max_alternation_phase_offset: float
    max_alternating_overlap_index: float
    min_template_coverage: float
    max_off_gait_fraction: float
    min_girdle_load_share: float | None = None
    min_girdle_duty_ratio: float | None = None
    min_walk_limb_phase: float | None = None
    max_walk_limb_phase: float | None = None
    max_off_gait_strides: float | None = None
    max_off_gait_fraction_ceiling: float | None = None
    max_synchrony_phase_offset: float | None = None
    min_synchronous_overlap_index: float | None = None
    min_step_length_over_leg: float | None = None
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
            number = _number(curriculum[key], key)
            if key in _INTEGER_KEYS:
                minimum = 0 if key == "gait_panel_seed_start" else 1
                if number < minimum or number != int(number):
                    raise ValueError(f"{GAIT_GATE_KIND} {key} must be an integer >= {minimum}")
                values[key] = int(number)
                continue
            if key not in {"min_avg_reward", "min_step_length_over_leg"} and number < 0:
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
        if profile == "quadruped_walk" and not values["min_walk_limb_phase"] < values["max_walk_limb_phase"]:
            raise ValueError(f"{GAIT_GATE_KIND} min_walk_limb_phase must be below max_walk_limb_phase")
        if profile == "quadruped_walk" and values["max_off_gait_fraction_ceiling"] < values["max_off_gait_fraction"]:
            raise ValueError(f"{GAIT_GATE_KIND} max_off_gait_fraction_ceiling must be >= max_off_gait_fraction")
        if "required_consecutive" in curriculum:
            consecutive = _number(curriculum["required_consecutive"], "required_consecutive")
            if consecutive < 1 or consecutive != int(consecutive):
                raise ValueError(f"{GAIT_GATE_KIND} required_consecutive must be a positive integer")
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
    "complete_cycles_min": "participation",
    "valid_swing_fraction_min": "stepping",
    "median_swing_clearance_over_leg_min": "stepping",
    "stride_length_over_leg_min": "stepping",
    "swing_ground_fraction_max": "stepping",
    "swing_slip_fraction_max": "stepping",
    "lead_exchange_fraction_min": "stepping",
    "step_length_over_leg_min": "stepping",
    "flight_fraction": "support",
    "body_support_fraction": "support",
    "foot_foot_contact_fraction": "support",
    "skid_fraction_max": "support",
    "phase_locking_min": "coupling",
    "alternation_phase_offset_max": "coupling",
    "synchrony_phase_offset_max": "coupling",
    "walk_limb_phase_min": "coupling",
    "walk_limb_phase_max": "coupling",
    "alternating_overlap_index_max": "coupling",
    "synchronous_overlap_index_min": "coupling",
    "template_coverage": "persistence",
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
    "step_length_over_leg_min": ("per_foot", "step_length_over_leg_median", min),
    "pair_load_ratio_min": ("contralateral", "load_ratio", min),
    "pair_duty_ratio_min": ("contralateral", "duty_ratio", min),
    "lead_exchange_fraction_min": ("contralateral", "lead_exchange_fraction", min),
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
    name = f"{RAIL_GROUPS.get(key, 'other')}/{label or key}"
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
    prefix. Template rails read ``episode["templates"][gait_profile]``.
    """
    failures: list[str] = []
    profile = thresholds.gait_profile
    expected = ("r", "l") if profile == "biped_alternating" else ("fr", "fl", "rr", "rl")
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
        ("pair_load_ratio_min", thresholds.min_pair_load_ratio, True),
        ("pair_duty_ratio_min", thresholds.min_pair_duty_ratio, True),
        ("complete_cycles_min", thresholds.min_complete_cycles_per_foot, True),
        ("valid_swing_fraction_min", thresholds.min_valid_swing_fraction, True),
        ("median_swing_clearance_over_leg_min", thresholds.min_median_swing_clearance_over_leg, True),
        ("stride_length_over_leg_min", thresholds.min_stride_length_over_leg, True),
        ("swing_ground_fraction_max", thresholds.max_swing_ground_fraction, False),
        ("swing_slip_fraction_max", thresholds.max_swing_slip_fraction, False),
        ("lead_exchange_fraction_min", thresholds.min_lead_exchange_fraction, True),
        ("flight_fraction", thresholds.max_flight_fraction, False),
        ("body_support_fraction", thresholds.max_body_support_fraction, False),
        ("foot_foot_contact_fraction", thresholds.max_foot_foot_contact_fraction, False),
        ("skid_fraction_max", thresholds.max_skid_fraction, False),
    ):
        _check(failures, episode, episode, key, bar, floor=floor)
    if profile != "biped_alternating":
        assert thresholds.min_girdle_load_share is not None and thresholds.min_girdle_duty_ratio is not None
        _check(failures, episode, episode, "girdle_load_share_min", thresholds.min_girdle_load_share, floor=True)
        _check(failures, episode, episode, "girdle_duty_ratio", thresholds.min_girdle_duty_ratio, floor=True)
    if thresholds.min_step_length_over_leg is not None:
        _check(
            failures,
            episode,
            episode,
            "step_length_over_leg_min",
            thresholds.min_step_length_over_leg,
            floor=True,
            allow_negative=True,
        )
    templates = episode.get("templates")
    template = templates.get(profile) if isinstance(templates, Mapping) else None
    if not isinstance(template, Mapping):
        failures.append(f"persistence/template: {profile} template is unmeasured")
        return not failures, tuple(failures)
    _check(failures, episode, template, "phase_locking_min", thresholds.min_phase_locking, floor=True)
    _check(failures, episode, template, "alternation_phase_offset_max", thresholds.max_alternation_phase_offset)
    overlap = template.get("alternating_overlap_index_max")
    if overlap is not None:  # undefined only at extreme duty factors, which other rails judge
        _check(failures, episode, template, "alternating_overlap_index_max", thresholds.max_alternating_overlap_index)
    if profile == "quadruped_walk":
        assert thresholds.min_walk_limb_phase is not None and thresholds.max_walk_limb_phase is not None
        _check(failures, episode, template, "walk_limb_phase_min", thresholds.min_walk_limb_phase, floor=True)
        _check(failures, episode, template, "walk_limb_phase_max", thresholds.max_walk_limb_phase)
    if profile in {"quadruped_trot", "quadruped_pace"}:
        assert thresholds.max_synchrony_phase_offset is not None
        assert thresholds.min_synchronous_overlap_index is not None
        _check(failures, episode, template, "synchrony_phase_offset_max", thresholds.max_synchrony_phase_offset)
        if template.get("synchronous_overlap_index_min") is not None:
            _check(
                failures,
                episode,
                template,
                "synchronous_overlap_index_min",
                thresholds.min_synchronous_overlap_index,
                floor=True,
            )
    # Persistence: the declared gait must be on template for most of the
    # window, and the time demonstrably spent off it (a limb not cycling, a
    # pair locked in a competing gait, a sustained uncoordinated bout) has
    # one budget, a fraction of the window. A walk may instead spend up to
    # ``max_off_gait_strides`` strides (one mistimed footfall) below the
    # ceiling: its competing templates sit next to its own.
    _check(failures, episode, template, "template_coverage", thresholds.min_template_coverage, floor=True)
    off_gait = _metric(template.get("off_gait_fraction"))
    off_strides = _metric(template.get("off_gait_strides"))
    if off_gait is None:
        failures.append("persistence/off_gait_fraction: unmeasured")
    elif off_gait < 0 or off_gait > 1:
        failures.append(f"persistence/off_gait_fraction: {off_gait:.6g} is outside its physical range")
    elif off_gait > thresholds.max_off_gait_fraction and not (
        thresholds.max_off_gait_strides is not None
        and thresholds.max_off_gait_fraction_ceiling is not None
        and off_strides is not None
        and off_strides <= thresholds.max_off_gait_strides
        and off_gait <= thresholds.max_off_gait_fraction_ceiling
    ):
        failures.append(
            f"persistence/off_gait_fraction: {off_gait:.6g} of window"
            + (f" ({off_strides:.3g} strides)" if off_strides is not None else "")
            + f" > {thresholds.max_off_gait_fraction:.6g}"
        )
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
    "participation/complete_cycles_min": "too few strides",
    "stepping/valid_swing_fraction_min": "shuffle (invalid swings)",
    "stepping/median_swing_clearance_over_leg_min": "shuffle (low swings)",
    "stepping/stride_length_over_leg_min": "shuffle (short strides)",
    "stepping/swing_ground_fraction_max": "foot drag (swing at the floor)",
    "stepping/swing_slip_fraction_max": "foot drag (sliding swing)",
    "stepping/lead_exchange_fraction_min": "step-to (no lead-limb exchange)",
    "stepping/step_length_over_leg_min": "step-to (foot lands behind)",
    "support/flight_fraction": "too much flight",
    "support/body_support_fraction": "body-supported",
    "support/foot_foot_contact_fraction": "feet stacked",
    "support/skid_fraction_max": "skidding",
    "coupling/phase_locking_min": "limbs not phase-locked",
    "coupling/alternation_phase_offset_max": "asymmetric pair timing",
    "coupling/alternating_overlap_index_max": "feet loaded together",
    "coupling/synchrony_phase_offset_max": "pairs not synchronous",
    "coupling/synchronous_overlap_index_min": "pairs not loaded together",
    "coupling/walk_limb_phase_min": "limb phase outside walk band",
    "coupling/walk_limb_phase_max": "limb phase outside walk band",
    "persistence/template_coverage": "gait not sustained",
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
