"""Physical locomotion certification from a declared, per-episode gait panel.

The reward gate remains unchanged. ``locomotion_gait/v1`` certifies the joint
event of completing the episode, making forward progress, and expressing
the declared gait without excess flight, sliding, or body support. Missing
telemetry fails the episode; missing or stale panel evidence fails the gate.
This module is pure: checkpoint and evidence binding belongs to reporting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..record_fields import is_sha256_digest
from .recovery_gate import binomial_lcb

GAIT_GATE_KIND = "locomotion_gait/v1"
GAIT_PROFILES = frozenset({"biped_alternating", "quadruped_walk", "quadruped_trot", "quadruped_pace"})
GAIT_REQUIRED_KEYS = frozenset(
    {
        "gait_profile",
        "measurement_protocol_sha256",
        "min_eval_episodes",
        "gait_panel_seed_start",
        "min_gait_success_lcb",
        "min_episode_forward_vel",
        "min_episode_duration_s",
        "min_complete_cycles_per_foot",
        "min_phase_match_fraction",
        "max_simultaneous_fraction",
        "max_flight_fraction",
        "max_flight_s",
        "max_slip_distance_over_leg",
        "max_body_support_fraction",
    }
)
GAIT_THRESHOLD_KEYS = GAIT_REQUIRED_KEYS | {
    "max_foot_foot_contact_fraction",
    "min_avg_reward",
    "required_consecutive",
}


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
    min_complete_cycles_per_foot: int
    min_phase_match_fraction: float
    max_simultaneous_fraction: float
    max_flight_fraction: float
    max_flight_s: float
    max_slip_distance_over_leg: float
    max_body_support_fraction: float
    max_foot_foot_contact_fraction: float | None = None
    min_avg_reward: float | None = None

    @classmethod
    def from_curriculum(cls, curriculum: Mapping[str, Any]) -> GaitGateThresholds:
        """Require an explicit protocol, gait, panel, and physical criteria."""
        missing = sorted(GAIT_REQUIRED_KEYS - set(curriculum))
        if missing:
            raise ValueError(f"{GAIT_GATE_KIND} missing required thresholds: {missing}")
        profile = curriculum["gait_profile"]
        if not isinstance(profile, str) or profile not in GAIT_PROFILES:
            raise ValueError(f"{GAIT_GATE_KIND} unknown gait_profile {profile!r}")
        digest = curriculum["measurement_protocol_sha256"]
        if not is_sha256_digest(digest):
            raise ValueError(f"{GAIT_GATE_KIND} measurement_protocol_sha256 must be a sha256:<hex> digest")
        values: dict[str, Any] = {"gait_profile": profile, "measurement_protocol_sha256": digest}
        for key in GAIT_THRESHOLD_KEYS - {"gait_profile", "measurement_protocol_sha256", "required_consecutive"}:
            if key not in curriculum:
                continue
            number = _number(curriculum[key], key)
            if key in {"min_eval_episodes", "min_complete_cycles_per_foot", "gait_panel_seed_start"}:
                minimum = 0 if key == "gait_panel_seed_start" else 1
                if number < minimum or number != int(number):
                    raise ValueError(f"{GAIT_GATE_KIND} {key} must be an integer >= {minimum}")
                values[key] = int(number)
            else:
                if key != "min_avg_reward" and number < 0:
                    raise ValueError(f"{GAIT_GATE_KIND} {key} must be nonnegative")
                if (key.endswith("fraction") or key == "min_gait_success_lcb") and number > 1:
                    raise ValueError(f"{GAIT_GATE_KIND} {key} must be in [0, 1]")
                values[key] = number
        if values["min_episode_duration_s"] <= 0:
            raise ValueError(f"{GAIT_GATE_KIND} min_episode_duration_s must be positive")
        if values["min_gait_success_lcb"] <= 0:
            raise ValueError(f"{GAIT_GATE_KIND} min_gait_success_lcb must be positive")
        if "required_consecutive" in curriculum:
            consecutive = _number(curriculum["required_consecutive"], "required_consecutive")
            if consecutive < 1 or consecutive != int(consecutive):
                raise ValueError(f"{GAIT_GATE_KIND} required_consecutive must be a positive integer")
        return cls(**values)


def _metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _check(failures: list[str], metrics: Mapping[str, Any], key: str, threshold: float, *, floor: bool = False) -> None:
    value = _metric(metrics.get(key))
    if value is None:
        failures.append(f"{key} is unmeasured")
    elif (value < 0 and key != "mean_speed_mps") or (key.endswith("fraction") and value > 1):
        failures.append(f"{key} {value:.6g} is outside its physical range")
    elif value < threshold if floor else value > threshold:
        failures.append(f"{key} {value:.6g} {'<' if floor else '>'} {threshold:.6g}")


def classify_gait_episode(
    episode: Mapping[str, Any], thresholds: GaitGateThresholds, *, foot_names: Sequence[str]
) -> tuple[bool, tuple[str, ...]]:
    """Qualify one episode. Foot order is (r,l) or (fr,fl,rr,rl).

    The quadruped walk requires four distinct beats, the trot diagonal
    synchrony, and the pace ipsilateral synchrony, in addition to fore/hind
    left/right alternation. A synchronous hop cannot qualify as a walk.
    """
    failures: list[str] = []
    expected_feet = 2 if thresholds.gait_profile == "biped_alternating" else 4
    if len(foot_names) != expected_feet or len(set(foot_names)) != expected_feet:
        return False, (f"{thresholds.gait_profile} requires {expected_feet} distinct feet",)
    if episode.get("telemetry_valid") is not True:
        failures.append("telemetry is invalid or missing")
    if episode.get("completed_horizon") is not True:
        failures.append("episode did not complete its horizon")
    _check(failures, episode, "duration_s", thresholds.min_episode_duration_s, floor=True)
    _check(failures, episode, "mean_speed_mps", thresholds.min_episode_forward_vel, floor=True)
    for key, ceiling in (
        ("flight_fraction", thresholds.max_flight_fraction),
        ("max_flight_s", thresholds.max_flight_s),
        ("max_slip_distance_over_leg", thresholds.max_slip_distance_over_leg),
        ("body_support_fraction", thresholds.max_body_support_fraction),
    ):
        _check(failures, episode, key, ceiling)
    if thresholds.max_foot_foot_contact_fraction is not None:
        _check(failures, episode, "foot_foot_contact_fraction", thresholds.max_foot_foot_contact_fraction)
    feet = episode.get("per_foot")
    feet = feet if isinstance(feet, Mapping) else {}
    for name in foot_names:
        foot = feet.get(name)
        foot = foot if isinstance(foot, Mapping) else {}
        for key in ("complete_cycles", "valid_cycles"):
            value = _metric(foot.get(key))
            if value is None or value != int(value) or value < thresholds.min_complete_cycles_per_foot:
                failures.append(f"{name}.{key} {value!r} < {thresholds.min_complete_cycles_per_foot}")
    phases = episode.get("pair_phase")
    phases = phases if isinstance(phases, Mapping) else {}
    for first, second in zip(foot_names[::2], foot_names[1::2], strict=True):
        pair_name = f"{first}|{second}"
        pair = phases.get(pair_name)
        if not isinstance(pair, Mapping):
            failures.append(f"pair_phase {pair_name} is unmeasured")
            continue
        pair_failures: list[str] = []
        samples = _metric(pair.get("samples"))
        if samples is None or samples != int(samples) or samples < thresholds.min_complete_cycles_per_foot:
            pair_failures.append("phase sample count is missing or below the cycle minimum")
        _check(pair_failures, pair, "alternation_match_fraction", thresholds.min_phase_match_fraction, floor=True)
        _check(pair_failures, pair, "simultaneous_touchdown_fraction", thresholds.max_simultaneous_fraction)
        failures.extend(f"{pair_name}.{failure}" for failure in pair_failures)
    profile_metric = {
        "quadruped_walk": "four_beat_fraction",
        "quadruped_trot": "diagonal_phase_match_fraction",
        "quadruped_pace": "ipsilateral_phase_match_fraction",
    }.get(thresholds.gait_profile)
    if profile_metric:
        _check(failures, episode, profile_metric, thresholds.min_phase_match_fraction, floor=True)
    return not failures, tuple(failures)


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
