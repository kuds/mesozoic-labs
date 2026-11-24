"""Tests for the command-tracking gate statistic, ``command_tracking/v1`` (pure; no env).

Every walk panel here starts from one passing cruise record and one passing
stand record, and each clause test edits only the field its defect would
move, so the criterion that refuses it is the one under test.  The follow
profile is checked against the behavior certificate it copies
(``behavior_certification.judge_behavior_panel``) on synthetic step rows run
through both.  The panel runner and its real-env checks are in
``test_command_tracking_panel.py``.
"""

from __future__ import annotations

import ast
import math
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from environments.shared.behavior_certification import RULES_PATH, judge_behavior_panel
from environments.shared.behavior_evaluation import summarize_episode
from environments.shared.config import env_kwargs_record
from environments.shared.curriculum import command_tracking_gate, gate_schema
from environments.shared.curriculum.command_tracking_gate import (
    COMMAND_PANEL_PROTOCOL,
    COMMAND_TRACKING_GATE_KIND,
    COMMAND_TRACKING_THRESHOLD_KEYS,
    CRUISE_FAMILY,
    FOLLOW_REQUIRED_KEYS,
    FOLLOW_THRESHOLD_KEYS,
    PROVISIONAL_BIPED_GAIT_THRESHOLDS,
    PROVISIONAL_FOLLOW_COURSE_FRACTION,
    PROVISIONAL_FOLLOW_THRESHOLDS,
    PROVISIONAL_WALK_THRESHOLDS,
    STAND_FAMILY,
    WALK_REQUIRED_KEYS,
    WALK_THRESHOLD_KEYS,
    CommandTrackingEpisode,
    FollowThresholds,
    WalkThresholds,
    classify_cruise_episode,
    command_config_from,
    evaluate_command_tracking_gate,
    evaluate_follow_profile,
    evaluate_walk_profile,
    panel_measurement,
    reason_key,
    reduce_command_episode,
    sample_in_tolerance,
    summarize_command_panel,
    thresholds_from_curriculum,
)
from environments.shared.curriculum.gate_schema import (
    FROZEN_NULL_GATE_KINDS,
    GATE_KINDS,
    GATE_SCHEMA_VERSION,
    STANCE_GATE_KINDS,
    GateSchemaError,
    gate_config_differences,
    gate_config_sha256,
    gate_config_view,
    same_threshold,
    validate_gate_config,
)
from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.direction_commands import DirectionCommandConfig, wrap_angle

#: Complete blocks: a profile's provisional bars with a biped's provisional gait clause (the clause is
#: required in both profiles; its own tests are in test_command_tracking_gait.py).
WALK_BLOCK: dict[str, Any] = {**PROVISIONAL_WALK_THRESHOLDS, **PROVISIONAL_BIPED_GAIT_THRESHOLDS}
FOLLOW_BLOCK: dict[str, Any] = {**PROVISIONAL_FOLLOW_THRESHOLDS, **PROVISIONAL_BIPED_GAIT_THRESHOLDS}
#: A biped walk's gait metrics that clear every provisional biped bar (the T. rex reference walk's, rounded).
WALKING_GAIT: tuple[tuple[str, float], ...] = (
    ("limbs", 2.0),
    ("flight_fraction", 0.0),
    ("hop_flight_fraction", 0.0),
    ("min_limb_duty", 0.6),
    ("alternation_index", 1.0),
    ("pair_alternation_index", 1.0),
    ("simultaneous_fraction", 0.0),
    ("lead_swaps_per_stride", 2.0),
    ("foot_contact_fraction", 0.0),
    ("phantom_support_fraction", 0.0),
)

#: The design's walk command (heading hold, reset-only stands, no turns).
WALK_CONFIG = DirectionCommandConfig(
    cruise_speed=0.5,
    speed_range=(0.25, 0.5),
    stop_probability=0.15,
    turn_increment_max=0.0,
    switch_interval_s=100.0,
    initial_heading_offset_max=0.26,
)
#: A follow command: turns, speed changes and stops at the switches.
FOLLOW_CONFIG = DirectionCommandConfig(
    speed_range=(0.5, 1.05), stop_probability=0.15, turn_increment_max=math.pi / 6, switch_interval_s=6.0
)


def _walk_thresholds(block: dict[str, Any]) -> WalkThresholds:
    thresholds = thresholds_from_curriculum(block)
    assert isinstance(thresholds, WalkThresholds)
    return thresholds


def _follow_thresholds(block: dict[str, Any]) -> FollowThresholds:
    thresholds = thresholds_from_curriculum(block)
    assert isinstance(thresholds, FollowThresholds)
    return thresholds


WALK = _walk_thresholds(WALK_BLOCK)
#: The certificate's whole rule set, the course pair included (it is mandatory there).
COURSE_DISTANCE_M = 10.0
FOLLOW = _follow_thresholds(
    {
        **FOLLOW_BLOCK,
        "min_follow_course_fraction": PROVISIONAL_FOLLOW_COURSE_FRACTION,
        "follow_course_distance_m": COURSE_DISTANCE_M,
    }
)
WALK_MEASUREMENT = panel_measurement(WALK_CONFIG, WALK)
FOLLOW_MEASUREMENT = panel_measurement(FOLLOW_CONFIG, FOLLOW)


def _block(profile_block: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "gate_kind": COMMAND_TRACKING_GATE_KIND,
        **profile_block,
        **extra,
    }


def _cruise(seed: int = 3042, **overrides: Any) -> CommandTrackingEpisode:
    """A cruise episode that clears every walk bar: the horizon, the settle, tracking and speed."""
    values: dict[str, Any] = dict(
        episode=seed - 3042,
        seed=seed,
        family=CRUISE_FAMILY,
        length=1000,
        horizon=1000,
        dt=0.01,
        full_horizon=True,
        fall=False,
        termination_reason=None,
        reward=120.0,
        desired_speed=0.4,
        planned_event_count=1,
        initial_heading_error_rad=0.12,
        first_event_settled=True,
        tracking_fraction=0.9,
        tracking_fraction_after_settle=0.95,
        mean_speed_along_heading_mps=0.38,
        mean_desired_speed_mps=0.4,
        stand_hold_fraction=0.93,
        eligible_events=1,
        settled_events=1,
        eligible_stops=0,
        settled_stops=0,
        eligible_turns=0,
        settled_turns=0,
        max_course_progress_m=3.6,
        measurement=WALK_MEASUREMENT,
        gait_metrics=WALKING_GAIT,
    )
    values.update(overrides)
    return CommandTrackingEpisode(**values)


def _stand(seed: int = 3060, **overrides: Any) -> CommandTrackingEpisode:
    """A whole-episode stand that holds the stop after its settle."""
    values: dict[str, Any] = dict(
        family=STAND_FAMILY,
        desired_speed=0.0,
        initial_heading_error_rad=0.0,
        tracking_fraction=0.98,
        tracking_fraction_after_settle=1.0,
        mean_speed_along_heading_mps=math.nan,
        mean_desired_speed_mps=math.nan,
        stand_hold_fraction=1.0,
        eligible_stops=1,
        settled_stops=1,
        max_course_progress_m=0.01,
    )
    values.update(overrides)
    return _cruise(seed, **values)


def _walk_panel(n_cruise: int = 20, n_stand: int = 5) -> list[CommandTrackingEpisode]:
    return [_cruise(3042 + index) for index in range(n_cruise)] + [_stand(3100 + index) for index in range(n_stand)]


def _follow(seed: int = 3042, **overrides: Any) -> CommandTrackingEpisode:
    """A 25 s follow episode: four events, one stop and two turns, every one settled."""
    values: dict[str, Any] = dict(
        horizon=2500,
        length=2500,
        desired_speed=0.8,
        planned_event_count=4,
        tracking_fraction=0.8,
        eligible_events=4,
        settled_events=4,
        eligible_stops=1,
        settled_stops=1,
        eligible_turns=2,
        settled_turns=2,
        max_course_progress_m=7.0,
        measurement=FOLLOW_MEASUREMENT,
    )
    values.update(overrides)
    return _cruise(seed, **values)


def _follow_panel(n: int = 20) -> list[CommandTrackingEpisode]:
    return [_follow(3042 + index) for index in range(n)]


def _judge_walk(episodes: list[CommandTrackingEpisode], thresholds: WalkThresholds = WALK):
    return evaluate_walk_profile(episodes, thresholds, measurement=WALK_MEASUREMENT)


def _judge_follow(episodes: list[CommandTrackingEpisode], thresholds: FollowThresholds = FOLLOW, **kwargs: Any):
    return evaluate_follow_profile(
        episodes,
        thresholds,
        measurement=FOLLOW_MEASUREMENT,
        require_stop=kwargs.pop("require_stop", True),
        require_turn=kwargs.pop("require_turn", True),
        **kwargs,
    )


# ── registration ─────────────────────────────────────────────────────────────


class TestRegistration:
    def test_the_kind_is_registered_with_the_modules_own_keys(self):
        assert GATE_KINDS[COMMAND_TRACKING_GATE_KIND] is COMMAND_TRACKING_THRESHOLD_KEYS
        assert COMMAND_TRACKING_THRESHOLD_KEYS == WALK_THRESHOLD_KEYS | FOLLOW_THRESHOLD_KEYS
        assert gate_schema._REQUIRED_THRESHOLD_KEYS[COMMAND_TRACKING_GATE_KIND] == {"profile"}
        assert COMMAND_TRACKING_GATE_KIND not in STANCE_GATE_KINDS
        assert COMMAND_TRACKING_GATE_KIND not in FROZEN_NULL_GATE_KINDS
        assert GATE_SCHEMA_VERSION == 1

    def test_no_key_is_also_a_non_threshold_curriculum_key(self):
        """A key both a threshold and a schedule/diagnostic key would project into a digest it never entered."""
        assert not COMMAND_TRACKING_THRESHOLD_KEYS & gate_schema.NON_THRESHOLD_CURRICULUM_KEYS

    def test_registration_moves_no_committed_stage_gate_view(self):
        """No committed stage declares the kind, and each one's view is its own kind's projection, unchanged."""
        from environments.shared.config import load_all_stages
        from environments.shared.species_names import model_ids

        others = frozenset().union(*(keys for kind, keys in GATE_KINDS.items() if kind != COMMAND_TRACKING_GATE_KIND))
        assert model_ids()
        for model_id in model_ids():
            for stage, cfg in load_all_stages(model_id).items():
                curriculum = cfg["curriculum_kwargs"]
                assert curriculum.get("gate_kind") != COMMAND_TRACKING_GATE_KIND, (model_id, stage)
                assert not set(curriculum) & (COMMAND_TRACKING_THRESHOLD_KEYS - others), (model_id, stage)
                kind = curriculum.get("gate_kind")
                keys = GATE_KINDS[kind] if kind in GATE_KINDS else others
                expected = {key: curriculum[key] for key in sorted(keys) if key in curriculum}
                assert gate_config_view(curriculum)["thresholds"] == expected, (model_id, stage)

    @pytest.mark.parametrize("profile_block", [WALK_BLOCK, FOLLOW_BLOCK])
    def test_the_provisional_blocks_validate_beside_non_threshold_keys(self, profile_block):
        block = _block(profile_block, timesteps=1_000_000, certification_seeds=2, max_checkpoints=3)
        assert validate_gate_config("locomotion", block) == COMMAND_TRACKING_GATE_KIND

    @pytest.mark.parametrize(
        "edit, match",
        [
            ({"profile": None}, r"missing required threshold field\(s\) \['profile'\]"),
            ({"profile": "trot"}, r"profile must be one of \['walk', 'follow'\]"),
            ({"min_cruise_success_lcb": None}, r"missing required threshold\(s\) \['min_cruise_success_lcb'\]"),
            ({"min_follow_episodes": 20}, r"profile 'walk' does not consume \['min_follow_episodes'\]"),
            ({"min_cruise_survival_lcb": 1.0}, r"min_cruise_survival_lcb must lie strictly between 0 and 1"),
            ({"min_avg_forward_vel": 0.5}, r"does not consume threshold field\(s\) \['min_avg_forward_vel'\]"),
        ],
    )
    def test_config_load_refuses_a_bad_block(self, edit, match):
        block = _block(WALK_BLOCK)
        for key, value in edit.items():
            if value is None:
                block.pop(key)
            else:
                block[key] = value
        with pytest.raises(GateSchemaError, match=match):
            validate_gate_config("locomotion", block)

    def test_the_string_profile_compares_as_a_word(self):
        assert same_threshold("walk", "walk")
        assert not same_threshold("walk", "follow")
        assert not same_threshold("walk", 1.0)
        assert same_threshold(20, 20.0)
        view = gate_config_view(_block(WALK_BLOCK))
        assert view["thresholds"]["profile"] == "walk"
        assert gate_config_differences(view["thresholds"], view) == []
        other = gate_config_view(_block(FOLLOW_BLOCK))
        assert "profile: judged at 'walk', configured 'follow' now" in gate_config_differences(
            view["thresholds"], other
        )

    def test_the_gate_digest_is_retype_stable_and_profile_sensitive(self):
        walk = _block(WALK_BLOCK)
        retyped = {**walk, "min_cruise_episodes": 20.0, "max_stand_falls": 0.0}
        assert gate_config_sha256(gate_config_view(walk)) == gate_config_sha256(gate_config_view(retyped))
        tightened = {**walk, "min_cruise_speed_ratio": 0.6}
        assert gate_config_sha256(gate_config_view(walk)) != gate_config_sha256(gate_config_view(tightened))

    def test_the_gate_module_is_pure_and_never_reads_the_certificate(self):
        """stdlib, behavior_evaluation's event logic, direction_commands, the pure locomotion metrics and recovery_gate."""
        tree = ast.parse(Path(command_tracking_gate.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(("." * node.level) + (node.module or ""))
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert imported == {
            "__future__",
            "math",
            "numbers",
            "collections.abc",
            "dataclasses",
            "typing",
            "..behavior_evaluation",
            "..direction_commands",
            "..gait.locomotion_metrics",
            ".recovery_gate",
            ".gate_schema",
        }
        assert "tomllib" not in imported and "..behavior_certification" not in imported


# ── thresholds ───────────────────────────────────────────────────────────────


class TestThresholds:
    def test_the_provisional_blocks_parse_and_round_trip(self):
        assert isinstance(WALK, WalkThresholds) and WALK.profile == "walk"
        assert isinstance(FOLLOW, FollowThresholds) and FOLLOW.profile == "follow"
        assert WALK.declared() == WALK_BLOCK
        assert thresholds_from_curriculum(WALK.declared()) == WALK
        assert thresholds_from_curriculum(FOLLOW.declared()) == FOLLOW
        assert FOLLOW.min_follow_reference_delta_lcb is None

    def test_the_starting_values_are_the_certificates(self):
        """Copied from configs/behavior_certification.toml (read here, in a test, never by the gate)."""
        rules = tomllib.loads(RULES_PATH.read_text())
        copied = {
            "min_follow_episodes": "episodes_per_family",
            "min_follow_horizon_s": "minimum_horizon_s",
            "min_follow_survival_lcb": "minimum_survival_lcb",
            "min_follow_success_lcb": "minimum_success_lcb",
            "min_follow_tracking_fraction": "minimum_tracking_fraction",
            "min_follow_event_settle_fraction": "minimum_event_settle_fraction",
            "min_follow_stop_settle_fraction": "minimum_stop_settle_fraction",
            "min_follow_turn_settle_fraction": "minimum_turn_settle_fraction",
            "min_follow_command_episodes": "minimum_command_episodes",
        }
        assert set(copied) == FOLLOW_REQUIRED_KEYS
        for key, rule in copied.items():
            assert PROVISIONAL_FOLLOW_THRESHOLDS[key] == rules[rule], key
        assert PROVISIONAL_FOLLOW_COURSE_FRACTION == rules["minimum_course_fraction"]
        walk = {
            "min_cruise_episodes": "episodes_per_family",
            "min_stand_episodes": "minimum_command_episodes",
            "min_cruise_survival_lcb": "minimum_survival_lcb",
            "min_cruise_success_lcb": "minimum_success_lcb",
            "min_cruise_tracking_fraction": "minimum_tracking_fraction",
            "min_cruise_speed_ratio": "minimum_course_fraction",
            "min_stand_hold_fraction": "minimum_tracking_fraction",
            "min_stand_success_fraction": "minimum_stop_settle_fraction",
        }
        for key, rule in walk.items():
            assert PROVISIONAL_WALK_THRESHOLDS[key] == rules[rule], key
        assert PROVISIONAL_WALK_THRESHOLDS["max_stand_falls"] == 0

    def test_every_starting_value_is_marked_provisional_with_its_source(self):
        source = Path(command_tracking_gate.__file__).read_text(encoding="utf-8")
        assert "PROVISIONAL starting values of the walk profile" in source
        assert "PROVISIONAL starting values of the follow profile" in source
        for key in WALK_REQUIRED_KEYS | FOLLOW_REQUIRED_KEYS:
            assert f'"{key}":' in source
        for rule in ("episodes_per_family", "minimum_survival_lcb", "minimum_stop_settle_fraction"):
            assert rule in source

    @pytest.mark.parametrize("key", sorted(WALK_REQUIRED_KEYS))
    def test_every_walk_bar_is_required(self, key):
        block = {k: v for k, v in WALK_BLOCK.items() if k != key}
        with pytest.raises(ValueError, match=rf"missing required threshold\(s\) \['{key}'\]"):
            thresholds_from_curriculum(block)

    @pytest.mark.parametrize("key", sorted(FOLLOW_REQUIRED_KEYS))
    def test_every_follow_bar_is_required(self, key):
        block = {k: v for k, v in FOLLOW_BLOCK.items() if k != key}
        with pytest.raises(ValueError, match=rf"missing required threshold\(s\) \['{key}'\]"):
            thresholds_from_curriculum(block)

    @pytest.mark.parametrize("profile", [None, "", "Walk", 1, ["walk"]])
    def test_the_profile_must_be_named(self, profile):
        block = {**WALK_BLOCK, "profile": profile}
        if profile is None:
            del block["profile"]
        with pytest.raises(ValueError, match="profile must be one of"):
            thresholds_from_curriculum(block)

    def test_unknown_keys_are_refused_and_non_threshold_keys_accepted(self):
        with pytest.raises(ValueError, match=r"does not know the key\(s\) \['gate_velocity_tolerence'\]"):
            thresholds_from_curriculum({**WALK_BLOCK, "gate_velocity_tolerence": 0.3})
        with pytest.raises(ValueError, match=r"does not know the key\(s\) \['min_avg_reward'\]"):
            thresholds_from_curriculum({**WALK_BLOCK, "min_avg_reward": 10.0})
        accepted = {
            **WALK_BLOCK,
            "gate_kind": COMMAND_TRACKING_GATE_KIND,
            "gate_schema_version": 1,
            "timesteps": 5_000_000,
            "certification_seeds": 2,
            "collapse_patience": 3,
        }
        assert thresholds_from_curriculum(accepted) == WALK

    def test_the_other_profiles_keys_are_refused(self):
        with pytest.raises(ValueError, match=r"profile 'follow' does not consume \['min_stand_episodes'\]"):
            thresholds_from_curriculum({**FOLLOW_BLOCK, "min_stand_episodes": 5})

    @pytest.mark.parametrize(
        "key, value, match",
        [
            ("min_cruise_episodes", 0, "whole number of at least 1"),
            ("min_cruise_episodes", 19.5, "whole number of at least 1"),
            ("max_stand_falls", -1, "whole number of at least 0"),
            ("min_cruise_survival_lcb", 0.0, "strictly between 0 and 1"),
            ("min_cruise_success_lcb", 1.0, "strictly between 0 and 1"),
            ("min_cruise_tracking_fraction", 0.0, r"in \(0, 1\]"),
            ("min_stand_hold_fraction", 1.01, r"in \(0, 1\]"),
            ("min_cruise_speed_ratio", 1.2, "overshooting the command"),
            ("min_cruise_tracking_fraction", "0.6", "not a number"),
            ("min_cruise_tracking_fraction", True, "not a number"),
            ("min_cruise_tracking_fraction", math.nan, "must be finite"),
            ("gate_velocity_tolerance", 0.0, "must be positive"),
            ("required_consecutive", 0, "whole number of at least 1"),
        ],
    )
    def test_values_outside_their_rule_are_refused(self, key, value, match):
        with pytest.raises(ValueError, match=match):
            thresholds_from_curriculum({**WALK_BLOCK, key: value})

    @pytest.mark.parametrize(
        "key, value, match",
        [
            ("min_follow_horizon_s", 0.0, "must be positive"),
            ("min_follow_reference_delta_lcb", 1.0, "strictly between -1 and 1"),
            ("min_follow_command_episodes", 0, "whole number of at least 1"),
        ],
    )
    def test_follow_values_outside_their_rule_are_refused(self, key, value, match):
        with pytest.raises(ValueError, match=match):
            thresholds_from_curriculum({**FOLLOW_BLOCK, key: value})

    def test_the_course_pair_is_declared_together(self):
        with pytest.raises(ValueError, match="together or not at all"):
            thresholds_from_curriculum({**FOLLOW_BLOCK, "min_follow_course_fraction": 0.5})
        with pytest.raises(ValueError, match="together or not at all"):
            thresholds_from_curriculum({**FOLLOW_BLOCK, "follow_course_distance_m": 10.0})

    def test_the_tolerance_overrides_reach_the_measurement(self):
        thresholds = thresholds_from_curriculum(
            {**WALK_BLOCK, "gate_velocity_tolerance": 0.3, "gate_stop_yaw_rate_tolerance": 0.1}
        )
        assert thresholds.tolerance_overrides == (("velocity_tolerance", 0.3), ("stop_yaw_rate_tolerance", 0.1))
        assert thresholds.declared()["gate_velocity_tolerance"] == 0.3
        measurement = panel_measurement(WALK_CONFIG, thresholds)
        own = WALK_CONFIG.tracking_parameters()["tolerances"]
        assert measurement.tolerance("velocity_tolerance") == 0.3
        assert measurement.tolerance("stop_yaw_rate_tolerance") == 0.1
        assert measurement.tolerance("yaw_rate_tolerance") == own["yaw_rate_tolerance"]
        assert measurement != WALK_MEASUREMENT


# ── the measurement ──────────────────────────────────────────────────────────


class TestMeasurement:
    def test_it_uses_the_envs_own_tolerances(self):
        """The command config's resolved tolerances, the speed ones scaled with speed_scale, and its yaw_rate_max."""
        config = replace(WALK_CONFIG, speed_scale=2.6, cruise_speed=0.5)
        measurement = panel_measurement(config, WALK)
        assert dict(measurement.tolerances) == config.tracking_parameters()["tolerances"]
        assert measurement.tolerance("velocity_tolerance") == pytest.approx(0.2 * 2.6 / 1.5)
        assert measurement.tolerance("stop_speed_tolerance") == pytest.approx(0.1 * 2.6 / 1.5)
        assert measurement.tolerance("stop_yaw_rate_tolerance") == 0.15
        assert measurement.yaw_rate_max == config.yaw_rate_max
        assert measurement.protocol == COMMAND_PANEL_PROTOCOL == "command_panel/v1"

    def test_a_recorded_command_config_rebuilds(self):
        record = env_kwargs_record({"command_config": WALK_CONFIG})["command_config"]
        assert isinstance(record["speed_range"], list)
        assert command_config_from(record) == WALK_CONFIG
        assert command_config_from(WALK_CONFIG) is WALK_CONFIG
        assert command_config_from(None) is None
        with pytest.raises(TypeError):
            command_config_from("heading_and_speed")
        with pytest.raises(TypeError):
            command_config_from({**record, "not_a_field": 1})


# ── the reduction ────────────────────────────────────────────────────────────

DT = 0.1


def _row(
    time_s: float,
    event: dict[str, Any],
    *,
    ok: bool,
    progress: float = 0.0,
    terminated: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """One post-step info row under *event*: on the command when *ok*, well off it otherwise."""
    speed, heading = float(event["desired_speed"]), float(event["desired_heading"])
    active = speed > 0.0
    actual_heading = heading if ok or not active else heading + 0.5
    actual_speed = speed if ok else speed + 1.0
    error_heading = abs(wrap_angle(heading - actual_heading)) if active else 0.0
    return {
        "time_s": time_s,
        "command_event_id": int(event["event_id"]),
        "command_event_start_s": float(event["start_time_s"]),
        "desired_speed": speed,
        "desired_heading": heading,
        "actual_heading": actual_heading,
        "actual_speed": actual_speed,
        "actual_v_x": actual_speed,
        "actual_v_y": 0.0,
        "actual_yaw_rate": 0.0 if ok else 0.5,
        "tracking_error_v": 0.0 if ok else 1.0,
        "tracking_error_yaw": 0.0 if ok else 0.5,
        "tracking_error_heading": error_heading,
        "tracking_error_requested_speed": 0.0 if ok else 1.0,
        "tracking_heading_active": active,
        "tracking_in_tolerance": ok,
        "reward": 1.0,
        "terminated": terminated,
        "truncated": False,
        "termination_reason": reason,
        "course_progress_m": progress,
    }


def _events(*specs: tuple[float, float, float]) -> list[dict[str, Any]]:
    """Planned events from ``(start_time_s, desired_heading, desired_speed)``."""
    return [
        {"event_id": index, "start_time_s": start, "desired_heading": heading, "desired_speed": speed}
        for index, (start, heading, speed) in enumerate(specs)
    ]


def _rows(
    events: list[dict[str, Any]],
    *,
    steps: int,
    ok: Any = lambda time_s, event: time_s >= event["start_time_s"] + 0.5,
    fall_at: float | None = None,
    progress_rate: float = 0.4,
) -> list[dict[str, Any]]:
    """Rows of an episode over *events*: each step's command is the event its START time falls in."""
    rows = []
    for step in range(1, steps + 1):
        time_s = step * DT
        event = [event for event in events if (step - 1) * DT + 1e-9 >= event["start_time_s"]][-1]
        falls = fall_at is not None and time_s >= fall_at - 1e-9
        rows.append(
            _row(
                time_s,
                event,
                ok=bool(ok(time_s, event)),
                progress=progress_rate * time_s,
                terminated=falls,
                reason="fallen" if falls else None,
            )
        )
        if falls:
            break
    return rows


def _reduce(
    rows,
    events,
    *,
    horizon: int,
    seed: int = 3042,
    measurement=WALK_MEASUREMENT,
    initial_heading=0.0,
    gait_metrics=dict(WALKING_GAIT),
):
    """The reduction with a walking gait by default, so a clause under test is the one that decides."""
    return reduce_command_episode(
        rows,
        episode=0,
        seed=seed,
        dt=DT,
        horizon=horizon,
        planned_events=events,
        initial_heading=initial_heading,
        measurement=measurement,
        gait_metrics=gait_metrics,
    )


class TestReduction:
    def test_a_cruise_episode(self):
        """Settled within 1 s + 0.15/0.3 s, tracking after that allowance, speed along the heading after 2 s."""
        events = _events((0.0, 0.15, 0.4))
        rows = _rows(events, steps=50)
        episode = _reduce(rows, events, horizon=50)
        assert episode.family == CRUISE_FAMILY and episode.survived and episode.length == 50
        assert episode.initial_heading_error_rad == pytest.approx(0.15)
        assert episode.first_event_settled
        # Every step from 0.5 s is on the command, so every step after the 1.5 s allowance tracks.
        assert episode.tracking_fraction_after_settle == 1.0
        assert episode.tracking_fraction == pytest.approx(46 / 50)
        assert episode.speed_ratio == pytest.approx(1.0)
        assert episode.planned_event_count == 1
        assert episode.max_course_progress_m == pytest.approx(0.4 * 5.0)

    def test_only_the_events_that_start_before_the_horizon_count(self):
        """A switch planned after the 5 s horizon is no event of the episode: a walk command stays whole-episode."""
        events = _events((0.0, 0.15, 0.4), (100.0, 0.15, 0.4))
        assert _reduce(_rows(events, steps=50), events, horizon=50).planned_event_count == 1
        events = _events((0.0, 0.15, 0.4), (3.0, 0.15, 0.25))
        assert _reduce(_rows(events, steps=50), events, horizon=50).planned_event_count == 2

    def test_the_speed_ratio_reads_the_component_along_the_requested_heading(self):
        events = _events((0.0, 0.0, 0.4))
        rows = _rows(events, steps=50)
        for row in rows:
            row.update(actual_heading=0.3, actual_v_x=0.4, actual_v_y=0.0)
        episode = _reduce(rows, events, horizon=50)
        assert episode.speed_ratio == pytest.approx(math.cos(0.3))
        assert episode.mean_desired_speed_mps == pytest.approx(0.4)

    def test_a_heading_that_settles_too_late_does_not_settle(self):
        events = _events((0.0, 0.15, 0.4))
        late = _rows(events, steps=50, ok=lambda time_s, event: time_s >= 2.0)
        episode = _reduce(late, events, horizon=50)
        assert not episode.first_event_settled
        # The 36 steps ending at or after the 1.5 s allowance; the 31 from 2.0 s are on the command.
        assert episode.tracking_fraction_after_settle == pytest.approx(31 / 36)

    def test_a_stand_episode_holds_after_its_settle(self):
        events = _events((0.0, 0.2, 0.0))
        rows = _rows(events, steps=50, ok=lambda time_s, event: time_s >= 0.5 and not 3.0 <= time_s < 3.5)
        episode = _reduce(rows, events, horizon=50)
        assert episode.family == STAND_FAMILY and episode.desired_speed == 0.0
        assert episode.initial_heading_error_rad == 0.0
        assert episode.first_event_settled
        assert episode.stand_hold_fraction == pytest.approx(35 / 40)
        assert math.isnan(episode.speed_ratio)
        assert episode.eligible_stops == episode.settled_stops == 1

    def test_a_fall_is_recorded(self):
        events = _events((0.0, 0.0, 0.4))
        episode = _reduce(_rows(events, steps=50, fall_at=3.0), events, horizon=50)
        assert episode.fall and not episode.full_horizon and not episode.survived
        assert episode.termination_reason == "fallen" and episode.length == 30

    def test_follow_events_count_stops_and_turns_the_certificates_way(self):
        events = _events((0.0, 0.0, 0.8), (6.0, 0.4, 0.8), (12.0, 0.4, 0.0), (16.0, 0.1, 0.8))
        episode = _reduce(_rows(events, steps=200), events, horizon=200, measurement=FOLLOW_MEASUREMENT)
        assert episode.planned_event_count == 4
        assert episode.eligible_events == episode.settled_events == 4
        assert episode.eligible_stops == episode.settled_stops == 1
        # The first event is never a turn; 6 s turns by 0.4 and 16 s (after the stop) by 0.3.
        assert episode.eligible_turns == episode.settled_turns == 2

    def test_the_in_tolerance_flag_is_recomputed_at_the_gate_tolerances(self):
        """An override tighter than a step's error refuses the step whatever the env's flag said."""
        events = _events((0.0, 0.0, 0.4))
        rows = _rows(events, steps=50)
        for row in rows:
            row["tracking_error_v"] = 0.1 if row["tracking_in_tolerance"] else 1.0
        assert all(sample_in_tolerance(row, WALK_MEASUREMENT) == row["tracking_in_tolerance"] for row in rows)
        tight = panel_measurement(
            WALK_CONFIG, thresholds_from_curriculum({**WALK_BLOCK, "gate_velocity_tolerance": 0.05})
        )
        assert not any(sample_in_tolerance(row, tight) for row in rows)
        episode = _reduce(rows, events, horizon=50, measurement=tight)
        assert episode.tracking_fraction == 0.0 and not episode.first_event_settled
        assert episode.measurement == tight

    def test_a_step_missing_a_tracking_field_is_not_in_tolerance(self):
        events = _events((0.0, 0.0, 0.4))
        row = _rows(events, steps=10)[-1]
        assert sample_in_tolerance(row, WALK_MEASUREMENT)
        for key in ("actual_speed", "tracking_error_v", "tracking_error_heading", "desired_heading"):
            assert not sample_in_tolerance({**row, key: None}, WALK_MEASUREMENT), key
        assert not sample_in_tolerance({**row, "tracking_error_yaw": math.nan}, WALK_MEASUREMENT)

    def test_an_episode_needs_its_reset_event(self):
        with pytest.raises(ValueError, match="plans at least the command event its reset draws"):
            _reduce(_rows(_events((0.0, 0.0, 0.4)), steps=5), [], horizon=5)

    def test_the_record_serialises_with_nan_as_null(self):
        payload = _stand().as_dict()
        assert payload["mean_speed_along_heading_mps"] is None
        assert payload["measurement"] == WALK_MEASUREMENT.as_dict()
        assert payload["gait_metrics"] == dict(WALKING_GAIT)
        assert _stand(gait_metrics=(("hop_flight_fraction", math.nan),)).as_dict()["gait_metrics"] == {
            "hop_flight_fraction": None
        }
        assert set(payload) == set(asdict(_stand()))


# ── the walk profile ─────────────────────────────────────────────────────────


class TestWalkProfile:
    def test_a_full_panel_passes_on_its_exact_bounds(self):
        result = _judge_walk(_walk_panel())
        assert result.passed, result.failures
        cruise, stand = result.family(CRUISE_FAMILY), result.family(STAND_FAMILY)
        assert (cruise.n_episodes, cruise.n_survived, cruise.n_success) == (20, 20, 20)
        assert cruise.survival_lcb == binomial_lcb(20, 20) == pytest.approx(0.8609, abs=5e-5)
        assert cruise.success_lcb == binomial_lcb(20, 20)
        assert (stand.n_episodes, stand.n_success, stand.success_fraction) == (5, 5, 1.0)
        assert stand.survival_lcb is None and stand.success_lcb is None
        assert result.episode_reasons == ((),) * 25

    def test_one_cruise_fall_fails_the_survival_bound(self):
        """19/20 bounds at 0.7839 < 0.80: the certificate's bar admits no fall at 20 episodes."""
        panel = _walk_panel()
        panel[3] = replace(panel[3], full_horizon=False, fall=True, length=400, termination_reason="fallen")
        result = _judge_walk(panel)
        assert not result.passed
        assert result.family(CRUISE_FAMILY).survival_lcb == pytest.approx(0.7839, abs=5e-5)
        assert any(failure.startswith("cruise family: survival_lcb 0.7839 < 0.8000") for failure in result.failures)
        assert result.episode_reasons[3][0].startswith("horizon: length 400 < 1000")

    @pytest.mark.parametrize(
        "defect, key",
        [
            ({"first_event_settled": False}, "heading_settle"),
            ({"tracking_fraction_after_settle": 0.55}, "min_cruise_tracking_fraction"),
            ({"tracking_fraction_after_settle": math.nan}, "min_cruise_tracking_fraction"),
            ({"mean_speed_along_heading_mps": 0.15}, "min_cruise_speed_ratio"),
            ({"mean_speed_along_heading_mps": math.nan}, "min_cruise_speed_ratio"),
        ],
    )
    def test_each_cruise_clause_fails_the_success_bound_at_four_episodes_and_not_at_three(self, defect, key):
        """17/20 bounds at 0.6563 and clears 0.60; 16/20 bounds at 0.5990 and does not."""
        three = _walk_panel()
        for index in range(3):
            three[index] = replace(three[index], **defect)
        result = _judge_walk(three)
        assert result.passed, result.failures
        assert result.family(CRUISE_FAMILY).success_lcb == pytest.approx(0.6563, abs=5e-5)
        assert dict(result.family(CRUISE_FAMILY).criterion_failures) == {key: 3}
        four = list(three)
        four[3] = replace(four[3], **defect)
        result = _judge_walk(four)
        assert not result.passed
        assert result.family(CRUISE_FAMILY).success_lcb == pytest.approx(0.5990, abs=5e-5)
        assert any(failure.startswith("cruise family: success_lcb 0.5990 < 0.6000") for failure in result.failures)
        assert all(reasons[0].startswith(f"{key}: ") for reasons in result.episode_reasons[:4])

    def test_the_cruise_bars_are_the_declared_ones(self):
        strict = replace(WALK, min_cruise_tracking_fraction=0.96, min_cruise_speed_ratio=0.96)
        reasons = classify_cruise_episode(_cruise(), strict)
        assert [reason.split(":", 1)[0] for reason in reasons] == [
            "min_cruise_tracking_fraction",
            "min_cruise_speed_ratio",
        ]
        assert classify_cruise_episode(_cruise(), WALK) == ()

    def test_a_stand_fall_fails_the_no_fall_rail(self):
        panel = _walk_panel()
        panel[-1] = replace(panel[-1], full_horizon=False, fall=True, length=300, termination_reason="fallen")
        result = _judge_walk(panel)
        assert not result.passed
        assert "stand family: 1 episodes fell or ended before the horizon > max_stand_falls 0" in result.failures
        assert result.family(STAND_FAMILY).success_fraction == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "defect, key",
        [
            ({"stand_hold_fraction": 0.5}, "min_stand_hold_fraction"),
            ({"stand_hold_fraction": math.nan}, "min_stand_hold_fraction"),
            ({"first_event_settled": False}, "stop_settle"),
        ],
    )
    def test_each_stand_clause_fails_the_fraction_at_three_of_five_and_not_at_two(self, defect, key):
        panel = _walk_panel()
        for index in (-1, -2):
            panel[index] = replace(panel[index], **defect)
        result = _judge_walk(panel)
        assert result.passed, result.failures
        assert result.family(STAND_FAMILY).success_fraction == pytest.approx(0.6)
        assert dict(result.family(STAND_FAMILY).criterion_failures) == {key: 2}
        panel[-3] = replace(panel[-3], **defect)
        result = _judge_walk(panel)
        assert not result.passed
        assert "stand family: success_fraction 0.4000 < 0.6000 (2/5 episodes held the stop)" in result.failures

    @pytest.mark.parametrize(
        "n_cruise, n_stand, message",
        [
            (19, 5, "cruise family: 19 episodes < min_cruise_episodes 20"),
            (20, 4, "stand family: 4 episodes < min_stand_episodes 5"),
            (20, 0, "stand family: 0 episodes < min_stand_episodes 5"),
            (0, 5, "cruise family: 0 episodes < min_cruise_episodes 20"),
        ],
    )
    def test_a_short_family_fails_closed(self, n_cruise, n_stand, message):
        result = _judge_walk(_walk_panel(n_cruise, n_stand))
        assert not result.passed
        assert any(failure.startswith(message) for failure in result.failures), result.failures

    def test_a_short_family_fails_even_when_its_bar_is_met(self):
        """At a minimum of 3 the 3 cruise episodes bound at 0.3684; the count, not only the bound, refuses 2."""
        small = replace(WALK, min_cruise_episodes=3, min_cruise_survival_lcb=0.3, min_cruise_success_lcb=0.3)
        assert _judge_walk(_walk_panel(3, 5), small).passed
        result = _judge_walk(_walk_panel(2, 5), replace(small, min_cruise_survival_lcb=0.2, min_cruise_success_lcb=0.2))
        assert result.family(CRUISE_FAMILY).survival_lcb == pytest.approx(binomial_lcb(2, 2))
        assert binomial_lcb(2, 2) > 0.2
        assert result.failures == (
            "cruise family: 2 episodes < min_cruise_episodes 3 (the family is short, so the gate fails closed)",
        )

    def test_an_empty_panel_fails_closed_on_every_family(self):
        result = _judge_walk([])
        assert not result.passed and result.n_episodes == 0
        assert len(result.failures) == 5
        assert math.isnan(result.family(CRUISE_FAMILY).survival_lcb)

    def test_a_record_reduced_under_another_measurement_is_refused(self):
        panel = _walk_panel()
        other = panel_measurement(replace(WALK_CONFIG, tracking_heading_tolerance=0.5), WALK)
        assert other != WALK_MEASUREMENT
        panel[0] = replace(panel[0], measurement=other)
        result = _judge_walk(panel)
        assert not result.passed
        assert result.failures[0].startswith("1 of 25 episodes were reduced under another measurement")

    def test_repeated_seeds_and_mislabelled_families_are_refused(self):
        panel = _walk_panel()
        panel[1] = replace(panel[1], seed=panel[0].seed)
        assert "the panel repeats a reset seed, so an episode would be counted twice" in _judge_walk(panel).failures
        panel = _walk_panel()
        panel[0] = replace(panel[0], family=STAND_FAMILY)
        assert any("disagrees with their reset command" in failure for failure in _judge_walk(panel).failures)

    def test_a_schedule_that_switches_mid_episode_is_not_a_walk(self):
        panel = _walk_panel()
        panel[0] = replace(panel[0], planned_event_count=2)
        result = _judge_walk(panel)
        assert not result.passed
        assert any("judged by the follow profile" in failure for failure in result.failures)

    def test_the_dispatcher_reads_the_stage_command_config(self):
        panel = _walk_panel()
        assert evaluate_command_tracking_gate(panel, WALK, command_config=WALK_CONFIG).passed
        record = env_kwargs_record({"command_config": WALK_CONFIG})["command_config"]
        assert evaluate_command_tracking_gate(panel, WALK, command_config=record).passed
        result = evaluate_command_tracking_gate(panel, WALK, command_config=None)
        assert not result.passed and 'command_mode "none"' in result.failures[0]
        assert not evaluate_command_tracking_gate(panel, WALK, command_config="heading").passed
        # Another tolerance in the stage's config is another measurement: the records are refused.
        wider = replace(WALK_CONFIG, tracking_velocity_tolerance=0.5)
        assert not evaluate_command_tracking_gate(panel, WALK, command_config=wider).passed


# ── the follow profile ───────────────────────────────────────────────────────


class TestFollowProfile:
    def test_a_full_panel_passes(self):
        result = _judge_follow(_follow_panel())
        assert result.passed, result.failures
        (family,) = result.families
        assert (family.n_episodes, family.n_survived, family.n_success) == (20, 20, 20)
        assert family.success_lcb == binomial_lcb(20, 20)
        assert dict((label, exposed) for label, exposed, _, _ in family.command_exposure) == {"stop": 20, "turn": 20}
        assert result.n_pairs is None

    @pytest.mark.parametrize(
        "defect, key",
        [
            ({"tracking_fraction": 0.5}, "min_follow_tracking_fraction"),
            ({"settled_events": 2}, "min_follow_event_settle_fraction"),
            ({"max_course_progress_m": 4.0}, "min_follow_course_fraction"),
        ],
    )
    def test_each_episode_clause_fails_the_success_bound(self, defect, key):
        panel = _follow_panel()
        for index in range(4):
            panel[index] = replace(panel[index], **defect)
        result = _judge_follow(panel)
        assert not result.passed
        assert result.family("follow").success_lcb == pytest.approx(0.5990, abs=5e-5)
        assert dict(result.family("follow").criterion_failures) == {key: 4}

    def test_survival_is_bounded(self):
        panel = _follow_panel()
        panel[0] = replace(panel[0], full_horizon=False, fall=True, length=900)
        result = _judge_follow(panel)
        assert any(failure.startswith("survival_lcb 0.7839 < 0.8000 (19/20)") for failure in result.failures)

    def test_stop_and_turn_exposure_and_fractions(self):
        panel = [replace(episode, eligible_stops=0, settled_stops=0) for episode in _follow_panel()]
        for index in range(4):
            panel[index] = replace(panel[index], eligible_stops=1, settled_stops=1)
        result = _judge_follow(panel)
        assert "4 episodes with an eligible stop < min_follow_command_episodes 5" in result.failures
        # A schedule without stops does not ask for any.
        assert _judge_follow(panel, require_stop=False).passed
        panel = _follow_panel()
        for index in range(9):
            panel[index] = replace(panel[index], settled_turns=1)
        result = _judge_follow(panel)
        assert "turn success fraction 0.5500 < 0.6000 (11/20)" in result.failures

    def test_a_stop_that_does_not_settle_fails_its_episode_and_the_panels_stop_fraction(self):
        """The certificate's two stop readings: a per-episode clause, and the fraction over the exposed episodes."""
        panel = _follow_panel()
        panel[0] = replace(panel[0], settled_stops=0)
        result = _judge_follow(panel)
        assert result.passed, result.failures
        assert [reason_key(reason) for reason in result.episode_reasons[0]] == ["min_follow_stop_settle_fraction"]
        exposure = {label: rest for label, *rest in result.family("follow").command_exposure}
        assert exposure["stop"] == [20, 19, 0.95]
        for index in range(1, 9):
            panel[index] = replace(panel[index], settled_stops=0)
        result = _judge_follow(panel)
        assert not result.passed
        assert "stop success fraction 0.5500 < 0.6000 (11/20)" in result.failures
        assert dict(result.family("follow").criterion_failures) == {"min_follow_stop_settle_fraction": 9}
        # A schedule that draws no stop judges neither reading.
        assert _judge_follow(panel, require_stop=False).passed

    @pytest.mark.parametrize("n", [0, 1, 19])
    def test_a_short_follow_panel_fails_closed(self, n):
        result = _judge_follow(_follow_panel(n))
        assert not result.passed
        assert f"{n} valid episodes < min_follow_episodes 20" in result.failures
        assert result.family("follow").n_episodes == n
        if n == 0:
            assert any(failure.startswith("survival_lcb nan") for failure in result.failures), result.failures

    def test_a_short_horizon_is_not_a_sample(self):
        panel = _follow_panel()
        panel[0] = replace(panel[0], horizon=1000, length=1000)
        result = _judge_follow(panel)
        assert (
            "1 episodes have a horizon of 10 s < min_follow_horizon_s 20; they are not valid samples" in result.failures
        )
        assert "19 valid episodes < min_follow_episodes 20" in result.failures

    def test_without_the_course_pair_progress_is_not_judged(self):
        thresholds = thresholds_from_curriculum(FOLLOW_BLOCK)
        measurement = panel_measurement(FOLLOW_CONFIG, thresholds)
        assert measurement == FOLLOW_MEASUREMENT
        panel = [replace(episode, max_course_progress_m=0.0) for episode in _follow_panel()]
        assert evaluate_follow_profile(
            panel, thresholds, measurement=measurement, require_stop=True, require_turn=True
        ).passed

    def test_the_paired_reference_fails_closed_when_missing(self):
        thresholds = replace(FOLLOW, min_follow_reference_delta_lcb=-0.1)
        result = _judge_follow(_follow_panel(), thresholds)
        assert not result.passed and result.n_pairs == 0
        assert any("reference panel is missing" in failure for failure in result.failures)

    def test_the_paired_reference_statistic(self):
        thresholds = replace(FOLLOW, min_follow_reference_delta_lcb=-0.1)
        panel = _follow_panel()
        # The same panel as its own reference: every difference is 0, which bounds at 0 >= -0.1.
        result = _judge_follow(panel, thresholds, reference=panel)
        assert result.passed and result.n_pairs == 20 and result.paired_delta_lcb == 0.0
        # A reference that succeeds where the policy forgot: the policy fails 3 seeds the reference passes.
        forgetful = list(panel)
        for index in range(3):
            forgetful[index] = replace(forgetful[index], tracking_fraction=0.1)
        result = _judge_follow(forgetful, replace(thresholds, min_follow_success_lcb=0.5), reference=panel)
        assert not result.passed
        assert result.paired_delta_lcb is not None and result.paired_delta_lcb < -0.1
        assert any(failure.startswith("paired_delta_lcb") for failure in result.failures)
        # Seeds must pair.
        shifted = [replace(episode, seed=episode.seed + 100) for episode in panel]
        result = _judge_follow(panel, thresholds, reference=shifted)
        assert any("rolled on other seeds" in failure for failure in result.failures)

    @pytest.mark.parametrize(
        "change",
        [
            {"family": "stand", "desired_speed": 0.0},
            {"desired_speed": 0.25},
            {"planned_event_count": 9},
            {"horizon": 1500, "length": 1500},
            {"dt": 0.025},
        ],
        ids=["stand_on_the_same_seeds", "speed", "event_count", "horizon", "dt"],
    )
    def test_the_paired_reference_must_run_the_same_task_on_each_seed(self, change):
        """Same seeds are not enough: a pair whose reference ran another reset draw or clock is refused."""
        thresholds = replace(FOLLOW, min_follow_reference_delta_lcb=-0.1)
        panel = _follow_panel()
        other = [replace(episode, **change) for episode in panel]
        result = _judge_follow(panel, thresholds, reference=other)
        assert not result.passed
        assert result.paired_delta_lcb is None and result.n_pairs == 0
        assert any("ran another task than the policy's on 20 seeds" in failure for failure in result.failures), (
            result.failures
        )
        # One mismatched seed is enough to refuse the whole pairing.
        one = list(panel)
        one[5] = other[5]
        result = _judge_follow(panel, thresholds, reference=one)
        assert any(f"on 1 seeds (first {panel[5].seed})" in failure for failure in result.failures), result.failures

    @pytest.mark.parametrize("scenario", ["clean", "falls", "turns", "stops", "few_stops", "tracking", "course"])
    def test_it_agrees_with_the_behavior_certificate(self, scenario):
        """The same rows through summarize_episode: the certificate's verdict and numbers are the profile's."""
        records, summaries = [], []
        for index in range(20):
            seed = 3042 + index
            stop_episode = index % 3 == 0 if scenario != "few_stops" else index % 7 == 0
            events = _events(
                (0.0, 0.1 * (index % 2), 0.8),
                (6.0, 0.1 * (index % 2) + 0.4, 0.8),
                (12.0, 0.1 * (index % 2) + 0.4, 0.0 if stop_episode else 0.8),
                (16.0, 0.1 * (index % 2) + 0.1, 0.8),
            )

            def ok(time_s: float, event: dict[str, Any], index: int = index) -> bool:
                if time_s < event["start_time_s"] + 0.5:
                    return False
                if scenario == "turns" and index < 9 and event["event_id"] == 1:
                    return False
                if scenario == "stops" and event["desired_speed"] == 0.0 and index < 12:
                    return False
                if scenario == "tracking" and index < 4:
                    return time_s < 9.0 and event["event_id"] < 2
                return True

            rows = _rows(
                events,
                steps=200,
                ok=ok,
                fall_at=8.0 if scenario == "falls" and index < 2 else None,
                progress_rate=0.1 if scenario == "course" and index < 4 else 0.4,
            )
            assert all(sample_in_tolerance(row, FOLLOW_MEASUREMENT) == row["tracking_in_tolerance"] for row in rows)
            records.append(_reduce(rows, events, horizon=200, seed=seed, measurement=FOLLOW_MEASUREMENT))
            summary = summarize_episode(
                rows, DT, planned_events=events, horizon_steps=200, initial_heading=0.0, yaw_rate_max=0.3
            )
            summaries.append({**summary, "episode_seed": seed, "terrain_family": "flat"})
        report = {"terrain_coverage": {"enabled_families": ["flat"], "complete": True}, "episodes": summaries}
        identity = {
            "command": {"config": {"stop_probability": 0.15, "turn_increment_max": math.pi / 6}},
            "env": {"course_distance": COURSE_DISTANCE_M},
        }
        rules = tomllib.loads(RULES_PATH.read_text())
        certificate = judge_behavior_panel(report, identity, rules)
        result = _judge_follow(records)
        flat = certificate["families"]["flat"]
        (family,) = result.families
        assert result.passed == certificate["passed"], (result.failures, certificate["failures"])
        assert (family.n_episodes, family.n_survived, family.n_success) == (
            flat["episodes"],
            flat["survived"],
            flat["successful"],
        )
        assert family.survival_lcb == pytest.approx(flat["survival_lcb"])
        assert family.success_lcb == pytest.approx(flat["success_lcb"])
        for label, exposed, settled, fraction in family.command_exposure:
            expected = flat["commands"][label]
            assert (exposed, settled) == (expected["exposed_episodes"], expected["successful_episodes"]), label
            assert fraction == pytest.approx(expected["success_fraction"]), label
        if scenario != "clean":
            assert not result.passed


# ── the descriptive summary ──────────────────────────────────────────────────


class TestPanelSummary:
    def test_it_counts_each_family_and_leaves_out_unmeasured_values(self):
        panel = _walk_panel(3, 2)
        panel[0] = replace(panel[0], full_horizon=False, fall=True, termination_reason="fallen")
        panel[1] = replace(panel[1], mean_speed_along_heading_mps=math.nan)
        summary = summarize_command_panel(panel)
        assert summary["protocol"] == COMMAND_PANEL_PROTOCOL
        cruise, stand = summary["families"][CRUISE_FAMILY], summary["families"][STAND_FAMILY]
        assert (cruise["n_episodes"], cruise["n_survived"], cruise["termination_reasons"]) == (3, 2, {"fallen": 1})
        assert cruise["speed_ratio"] == {"mean": pytest.approx(0.95), "unmeasured": 1}
        assert stand["stand_hold_fraction"] == {"mean": 1.0, "unmeasured": 0}
        assert summary["all"]["n_episodes"] == 5
        assert cruise["seeds"] == [3042, 3043, 3044]


class TestCatalogHeadline:
    def test_the_profile_names_the_headline(self):
        """The catalog's headline registry knows the kind (an import-time contract); the profile picks the statistic."""
        from environments.shared.species_catalog import _deliverable_headline

        walk = _deliverable_headline(COMMAND_TRACKING_GATE_KIND, {}, dict(PROVISIONAL_WALK_THRESHOLDS))
        assert [entry["key"] for entry in walk] == ["command_cruise_success_lcb", "command_stand_success_fraction"]
        follow = _deliverable_headline(COMMAND_TRACKING_GATE_KIND, {}, dict(PROVISIONAL_FOLLOW_THRESHOLDS))
        assert [entry["key"] for entry in follow] == ["command_follow_success_lcb"]
        assert all(entry["value"] is None for entry in walk + follow)
