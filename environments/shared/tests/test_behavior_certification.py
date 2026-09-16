"""Skill gates, comparable seeded panels, and per-terrain promotion evidence."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from environments.shared.behavior_certification import (
    _panel_env,
    behavior_library_key,
    comparison_from_panel,
    judge_behavior_panel,
    load_certification_rules,
)

ROOT = Path(__file__).resolve().parents[3]


def _identity():
    return {
        "parent_plant": {"species": "trex"},
        "commands": {"stop_probability": 0.15, "turn_increment_max": 0.5},
        "course_distance": 10.0,
        "terrain": {"apron_radius": 3.0, "blend_width": 1.5},
    }


def _report(families=("flat", "bumps"), per_family=20):
    episodes = []
    for i in range(per_family * len(families)):
        events = [
            {"eligible": True, "heading_active": True, "desired_heading": 0.0, "settled": True},
            {"eligible": True, "heading_active": True, "desired_heading": 0.3, "settled": True},
            {"eligible": True, "heading_active": False, "desired_heading": 0.3, "settled": True},
        ]
        episodes.append(
            {
                "episode_seed": 820001 + i,
                "terrain_family": families[i % len(families)],
                "horizon_s": 25.0,
                "full_horizon": True,
                "fall": False,
                "tracking_fraction": 0.9,
                "max_course_progress_m": 10.0,
                "events": events,
            }
        )
    return {
        "protocol": {
            "episode_seeds": [e["episode_seed"] for e in episodes],
            "environment_run_seed": 810001,
            "horizon_steps": 2500,
            "dt": 0.01,
        },
        "terrain_coverage": {"enabled_families": list(families), "complete": True},
        "episodes": episodes,
        "model_sha256": "sha256:" + "a" * 64,
        "normalization_sha256": "sha256:" + "b" * 64,
    }


def test_gate_requires_all_families_and_measured_commands():
    report = _report()
    result = judge_behavior_panel(report, _identity())
    assert result["passed"] and not result["failures"]
    assert result["families"]["bumps"]["survival_lcb"] > 0.8
    for episode in report["episodes"]:
        if episode["terrain_family"] == "bumps":
            episode["tracking_fraction"] = 0.2
    result = judge_behavior_panel(report, _identity())
    assert not result["passed"]
    assert any("bumps: skill-success" in failure for failure in result["failures"])
    assert result["families"]["flat"]["successful"] == 20


@pytest.mark.parametrize(
    "problem",
    ["short", "small_panel", "no_stops", "no_turns", "stationary", "falls", "nan", "duplicate_seed", "missing_family"],
)
def test_incomplete_or_bad_evidence_cannot_be_certified(problem):
    report = _report()
    if problem == "small_panel":
        report = _report(per_family=5)
    elif problem == "duplicate_seed":
        report["episodes"][1]["episode_seed"] = report["episodes"][0]["episode_seed"]
    elif problem == "missing_family":
        report["terrain_coverage"]["enabled_families"].append("depressions")
    else:
        for episode in report["episodes"]:
            if problem == "short":
                episode["horizon_s"] = 1
            elif problem == "no_stops":
                episode["events"] = episode["events"][:2]
            elif problem == "no_turns":
                for event in episode["events"]:
                    event["desired_heading"] = 0
            elif problem == "stationary":
                episode["max_course_progress_m"] = 0
            elif problem == "falls":
                episode.update(full_horizon=False, fall=True)
            elif problem == "nan":
                episode["tracking_fraction"] = float("nan")
    assert not judge_behavior_panel(report, _identity())["passed"]


def test_comparison_default_is_50_and_has_separate_per_family_constraints():
    assert load_certification_rules()["comparison"]["episodes"] == 50
    report = _report(("flat", "sloped", "bumps", "depressions", "mixed"), per_family=10)
    comparison = comparison_from_panel(report, _identity())
    assert len(comparison["protocol"]["episode_seeds"]) == 50
    assert len(comparison["metrics"]) == 30
    assert all(len(metric["values"]) == 10 for metric in comparison["metrics"])
    assert [m["name"] for m in comparison["metrics"]][:5] == [
        f"survival:{f}" for f in report["terrain_coverage"]["enabled_families"]
    ]
    assert comparison["model_sha256"] == report["model_sha256"]


def test_library_key_binds_species_behavior_task_and_judging_rules():
    identity = _identity()
    key = behavior_library_key("trex", "difficult_terrain", identity)
    changed = copy.deepcopy(identity)
    changed["course_distance"] = 5
    assert key != behavior_library_key("trex", "difficult_terrain", changed)
    assert key != behavior_library_key("trex", "follow_direction", identity)


@pytest.mark.parametrize("behavior", ["follow_direction", "mixed_terrain", "difficult_terrain"])
def test_real_panel_env_preserves_identity_and_stratifies_actual_ground(behavior):
    pytest.importorskip("mujoco")
    from environments.shared.behavior_evaluation import terrain_family_from_reset
    from environments.shared.train_behaviors import create_behavior_env, read_recipe

    recipe = ROOT / "configs/trex/behaviors" / f"{behavior}.toml"
    _, commands, terrain, kwargs = read_recipe(recipe)
    original = create_behavior_env("trex", commands=commands, terrain=terrain, run_seed=17, **kwargs)
    panel = _panel_env("trex", recipe, 910001)
    try:
        assert panel.behavior_identity == original.behavior_identity
        hashes = {}
        for family in panel.terrain_families:
            observation, info = panel.reset(seed=920001, options={"terrain_family": family})
            assert terrain_family_from_reset(info) == family
            assert np.isfinite(observation).all()
            assert panel.behavior_identity == original.behavior_identity
            hashes[family] = None if panel.terrain is None else panel.terrain.normalized_heights.copy()
            _, info2 = panel.reset(seed=920001, options={"terrain_family": family})
            assert info == info2
            if hashes[family] is not None:
                np.testing.assert_array_equal(hashes[family], panel.terrain.normalized_heights)
    finally:
        panel.close()
        original.close()


@pytest.mark.parametrize("command", ["stop", "turn"])
def test_unexposed_episodes_cannot_hide_command_specific_failures(command):
    report = _report()
    for family in ("flat", "bumps"):
        rows = [row for row in report["episodes"] if row["terrain_family"] == family]
        for index, episode in enumerate(rows):
            if index >= 5:
                if command == "stop":
                    episode["events"] = episode["events"][:2]
                else:
                    for event in episode["events"]:
                        event["desired_heading"] = 0.0
            elif index < 3:
                episode["events"][-1 if command == "stop" else 1]["settled"] = False
    result = judge_behavior_panel(report, _identity())
    assert not result["passed"]
    for family in ("flat", "bumps"):
        measured = result["families"][family]
        assert measured["successful"] == 17
        assert measured["success_lcb"] > 0.6
        assert measured["commands"][command] == {
            "exposed_episodes": 5,
            "successful_episodes": 2,
            "success_fraction": 0.4,
        }
        assert any(f"{family}: {command} success fraction" in failure for failure in result["failures"])


@pytest.mark.parametrize("distance", [0.2, 0.5, 0.75])
def test_small_species_must_leave_the_flat_apron_and_blend(distance):
    identity = _identity()
    identity["course_distance"] = 0.4
    identity["terrain"] = {"apron_radius": 0.5, "blend_width": 0.25}
    report = _report()
    for episode in report["episodes"]:
        episode["max_course_progress_m"] = distance
    result = judge_behavior_panel(report, identity)
    assert not result["passed"]
    assert result["families"]["flat"]["successful"] == 20
    assert result["families"]["bumps"]["successful"] == 0
    assert result["families"]["bumps"]["terrain_departure_episodes"] == 0
    assert result["families"]["bumps"]["terrain_departure_radius_m"] == 0.75
    for episode in report["episodes"]:
        episode["max_course_progress_m"] = 0.76
    assert judge_behavior_panel(report, identity)["passed"]


@pytest.mark.parametrize("terrain", [None, {}, {"apron_radius": 0.5, "blend_width": float("nan")}])
def test_nonflat_gate_refuses_missing_or_invalid_surface_dimensions(terrain):
    identity = _identity()
    identity["terrain"] = terrain
    result = judge_behavior_panel(_report(), identity)
    assert not result["passed"]
    assert any("apron and blend dimensions" in failure for failure in result["failures"])
