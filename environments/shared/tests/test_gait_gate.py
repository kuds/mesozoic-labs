"""Joint episode qualification, stale evidence refusals, and gate integration."""

from __future__ import annotations

import copy
import csv
import json
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum import CurriculumManager, thresholds_from_configs
from environments.shared.curriculum.gait_gate import (
    GAIT_GATE_KIND,
    GaitGateThresholds,
    classify_gait_episode,
    default_gait_profile,
    describe_gait_episode,
    evaluate_gait_gate,
    provisional_gait_criteria,
    rail_id,
)
from environments.shared.curriculum.gate_schema import (
    GateSchemaError,
    gate_config_differences,
    gate_config_sha256,
    gate_config_view,
    validate_gate_config,
)
from environments.shared.gait.identity import measurement_protocol
from environments.shared.gait.metrics import episode_gait_metrics
from environments.shared.gait.report import REPORT_SCHEMA
from environments.shared.gait.seeds import checkpoint_seed_provenance
from environments.shared.gait.types import GaitProtocol
from environments.shared.plant_contract import current_plant_identity
from environments.shared.reporting.gates import evaluate_recorded_gate, evaluate_stage_gate, gait_statistics
from environments.shared.result_bundle.hashing import canonical_json_sha256, sha256_file
from environments.shared.task_fingerprint import stage_task_fingerprint

from .test_gait_metrics import BW, gait_trace, stepping_trace

TASK_RECORD = stage_task_fingerprint("trex", 2, env_kwargs={})
TASK = TASK_RECORD["task_sha256"]


@lru_cache(maxsize=1)
def _authored_morphology():
    from environments.shared.gait.morphology import GaitMorphology
    from environments.trex.envs.trex_env import TRexEnv

    env = TRexEnv()
    try:
        env.reset(seed=PUBLICATION_SEED_START)
        return GaitMorphology.from_env(env, "trex").describe(), float(env.model.opt.timestep), float(env.dt)
    finally:
        env.close()


def protocol():
    _, physics_dt, control_dt = _authored_morphology()
    return measurement_protocol(
        "trex",
        GaitProtocol(),
        settle_s=1.0,
        direction_xy=(1.0, 0.0),
        horizon=1000,
        physics_dt_s=physics_dt,
        control_dt_s=control_dt,
        episodes=40,
        seed_start=PUBLICATION_SEED_START,
    )


def curriculum(profile="biped_walk", **updates):
    return dict(
        {
            **provisional_gait_criteria(profile),
            "gate_kind": GAIT_GATE_KIND,
            "gate_schema_version": 1,
            "gait_profile": profile,
            "measurement_protocol_sha256": canonical_json_sha256(protocol()),
            "min_eval_episodes": 40,
            "gait_panel_seed_start": PUBLICATION_SEED_START,
            "min_gait_success_lcb": 0.8,
            "min_episode_forward_vel": 0.5,
            "min_episode_duration_s": 9.0,
        },
        **updates,
    )


@lru_cache(maxsize=None)
def _measured(kind):
    """Stored (JSON round-tripped) metrics of a clean synthetic gait, 10 s at 2 ms, 1 s settle."""
    feet = ("r", "l") if kind == "biped" else ("fr", "fl", "rr", "rl")
    phases = {
        "biped": (0.0, 0.5),
        "walk": (0.25, 0.75, 0.0, 0.5),
        "trot": (0.0, 0.5, 0.5, 0.0),
        "pace": (0.0, 0.5, 0.0, 0.5),
    }[kind]
    trace = gait_trace(phases, duty=0.6 if kind == "biped" else 0.7, period=0.8 if kind == "biped" else 1.0)
    measured = episode_gait_metrics(
        trace, body_weight_n=BW, leg_length_m=1.0, foot_names=feet, protocol=GaitProtocol(), settle_s=1.0
    )
    return json.dumps(measured)


def episode(seed=0, kind="biped", **updates):
    record = json.loads(_measured(kind))
    record.update({"seed": seed, "completed_horizon": True, "reward": 1000.0})
    record.update(updates)
    return record


def _with(record, path, value):
    """Copy of an episode with one nested metric replaced (path like 'templates.alternation.x')."""
    result = copy.deepcopy(record)
    target = result
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    return result


QUADRUPED = ("fr", "fl", "rr", "rl")


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("min_eval_episodes", True),
        ("min_eval_episodes", 2.5),
        ("min_eval_episodes", 0),
        ("min_episode_duration_s", 0),
        ("min_gait_success_lcb", 0),
        ("min_episode_forward_vel", "0.5"),
        ("max_flight_fraction", float("nan")),
        ("max_flight_fraction", -0.1),
        ("max_flight_fraction", 1.1),
        ("max_alternation_phase_offset", 0.6),
        ("min_complete_cycles_per_foot", 2.5),
        ("max_off_gait_fraction", 1.5),
        ("max_swing_ground_fraction", -0.1),
        ("max_swing_slip_fraction", 2.0),
        ("min_stride_length_over_leg", float("inf")),
        ("min_step_length_over_leg", float("nan")),
        ("min_step_through_stride_fraction", 1.2),
        ("min_walking_duty", 1.5),
        ("min_trunk_height_over_leg", -0.1),
        ("max_glide_stance_fraction", float("nan")),
        ("max_light_stance_fraction", -0.1),
        ("gait_profile", "automatic"),
        ("gait_profile", "quadruped_trot"),
        ("measurement_protocol_sha256", "old-version"),
        ("required_consecutive", 0),
        ("gait_panel_seed_start", -1),
    ],
)
def test_declared_criteria_are_strict(key, bad):
    with pytest.raises(ValueError, match=key):
        GaitGateThresholds.from_curriculum(curriculum(**{key: bad}))
    with pytest.raises(GateSchemaError, match=key):
        validate_gate_config(2, curriculum(**{key: bad}))


def test_missing_measurement_and_gait_criteria_never_default():
    block = curriculum()
    del block["measurement_protocol_sha256"]
    with pytest.raises(ValueError, match="measurement_protocol_sha256"):
        GaitGateThresholds.from_curriculum(block)
    with pytest.raises(GateSchemaError, match="missing required"):
        validate_gate_config(2, block)
    for key in (
        "max_foot_foot_contact_fraction",
        "max_off_gait_fraction",
        "min_stride_length_over_leg",
        "min_step_length_over_leg",
        "min_step_through_stride_fraction",
        "max_swing_ground_fraction",
        "max_swing_slip_fraction",
        "max_glide_stance_fraction",
        "max_light_stance_fraction",
        "min_trunk_height_over_leg",
        "min_walking_duty",
    ):
        block = curriculum()
        del block[key]
        with pytest.raises(ValueError, match=key):
            GaitGateThresholds.from_curriculum(block)
    for key in ("min_girdle_load_share", "min_girdle_duty_ratio", "max_girdle_unloaded_fraction", "min_walking_duty"):
        block = curriculum("quadruped_walk")
        del block[key]
        with pytest.raises(ValueError, match=key):
            GaitGateThresholds.from_curriculum(block)


@pytest.mark.parametrize(
    "key",
    [
        "max_stall_fraction",
        "max_asymmetric_bout_fraction",
        "min_lead_exchange_fraction",
        "min_template_coverage",
        "min_walk_limb_phase",
        "max_synchrony_phase_offset",
    ],
)
def test_retired_round_two_criteria_are_refused(key):
    """The stall, asymmetric-bout, lead-exchange and walk-band bars were folded into one budget or dropped."""
    with pytest.raises(GateSchemaError, match=key):
        validate_gate_config(2, curriculum(**{key: 0.05}))


def test_profile_specific_criteria_are_required_and_foreign_ones_refused():
    # walking support is walk-only; the run-allowed profile does not consume it
    with pytest.raises(ValueError, match="does not consume"):
        GaitGateThresholds.from_curriculum(curriculum("biped_alternating", min_walking_duty=0.35))
    # girdle bars are quadruped-only
    with pytest.raises(ValueError, match="does not consume"):
        GaitGateThresholds.from_curriculum(curriculum(min_girdle_load_share=0.18))
    with pytest.raises(ValueError, match="does not consume"):
        GaitGateThresholds.from_curriculum(curriculum(max_girdle_unloaded_fraction=0.05))
    with pytest.raises(ValueError, match="min_girdle_load_share"):
        GaitGateThresholds.from_curriculum(curriculum("quadruped_walk", min_girdle_load_share=0.6))
    for profile in ("biped_walk", "biped_alternating", "quadruped_walk"):
        assert GaitGateThresholds.from_curriculum(curriculum(profile)).gait_profile == profile
        assert validate_gate_config(2, curriculum(profile)) == GAIT_GATE_KIND


def test_report_only_default_profile_is_a_walk_unless_the_speed_bar_asks_for_running():
    # (species, feet, speed bar m/s, leg length m) -> owner's per-species recommendation
    assert default_gait_profile(2, 0.08, 0.21) == "biped_walk"  # compsognathus, Froude 0.003
    assert default_gait_profile(2, 1.0, 2.5) == "biped_walk"  # trex
    assert default_gait_profile(2, 2.0, 0.5) == "biped_alternating"  # velociraptor, Froude 0.82
    assert default_gait_profile(4, 0.75, 3.0) == "quadruped_walk"  # brachiosaurus
    assert default_gait_profile(4, 5.0, 0.3) == "quadruped_walk"  # never a quadruped run profile
    with pytest.raises(ValueError, match="no profile"):
        default_gait_profile(3, 1.0, 1.0)
    with pytest.raises(ValueError, match="leg length"):
        default_gait_profile(2, 1.0, 0.0)


def test_arbitrary_seeds_are_development_only_and_strict_config_uses_registered_block():
    development = curriculum(gait_panel_seed_start=0)
    assert GaitGateThresholds.from_curriculum(development).gait_panel_seed_start == 0
    with pytest.raises(GateSchemaError, match="registered certification"):
        validate_gate_config(2, development)


def test_clean_measured_gaits_qualify_and_reasons_carry_stable_rail_ids():
    thresholds = GaitGateThresholds.from_curriculum(curriculum())
    passed, failures = classify_gait_episode(episode(), thresholds, foot_names=("r", "l"))
    assert passed, failures
    hop = _with(episode(), "templates.alternation.alternation_phase_offset_max", 0.48)
    passed, failures = classify_gait_episode(hop, thresholds, foot_names=("r", "l"))
    assert not passed
    assert [rail_id(reason) for reason in failures] == ["coupling/alternation_phase_offset_max"]
    assert failures[0].endswith("> 0.15")
    assert describe_gait_episode(hop, failures) == episode()["gait_label"] + "; asymmetric pair timing"


@pytest.mark.parametrize(
    ("path", "value", "rail"),
    [
        ("telemetry_valid", False, "episode/telemetry_valid"),
        ("completed_horizon", False, "episode/completed_horizon"),
        ("mean_speed_mps", float("nan"), "episode/mean_speed_mps"),
        ("mean_speed_mps", 0.1, "episode/mean_speed_mps"),
        ("duration_s", 8.0, "episode/duration_s"),
        ("skid_fraction_max", 0.8, "support/skid_fraction_max"),
        ("skid_fraction_max", 0.47, "support/skid_fraction_max"),
        ("flight_fraction", 0.7, "support/flight_fraction"),
        ("flight_fraction", 0.2, "support/flight_fraction"),
        ("flight_fraction", -1.0, "support/flight_fraction"),
        ("body_support_fraction", 0.02, "support/body_support_fraction"),
        ("foot_foot_contact_fraction", 0.5, "support/foot_foot_contact_fraction"),
        ("trunk_height_over_leg_p10", 0.4, "support/trunk_height_over_leg_p10"),
        ("trunk_height_over_leg_p10", None, "support/trunk_height_over_leg_p10"),
        ("unloaded_fraction", 0.3, "support/unloaded_fraction"),
        ("unloaded_fraction", None, "support/unloaded_fraction"),
        ("walk_glide_stance_fraction_max", 0.3, "support/walk_glide_stance_fraction_max"),
        ("limb_duty_min", 0.05, "participation/limb_duty_min"),
        ("limb_duty_min", 0.3, "support/walking_duty"),
        ("relative_limb_load_share_min", 0.06, "participation/relative_limb_load_share_min"),
        ("pair_load_ratio_min", 0.62, "participation/pair_load_ratio_min"),
        ("pair_duty_ratio_min", 0.63, "participation/pair_duty_ratio_min"),
        ("limb_phase_coverage_min", 0.4, "participation/limb_phase_coverage_min"),
        ("valid_swing_fraction_min", 0.2, "stepping/valid_swing_fraction_min"),
        ("median_swing_clearance_over_leg_min", 0.012, "stepping/median_swing_clearance_over_leg_min"),
        ("stride_length_over_leg_min", 0.09, "stepping/stride_length_over_leg_min"),
        ("stride_length_over_leg_min", None, "stepping/stride_length_over_leg_min"),
        ("swing_ground_fraction_max", 0.85, "stepping/swing_ground_fraction_max"),
        ("swing_ground_fraction_max", 0.55, "stepping/swing_ground_fraction_max"),
        ("glide_stance_fraction_max", 0.22, "support/glide_stance_fraction_max"),
        ("light_stance_fraction_max", 0.25, "participation/light_stance_fraction_max"),
        ("swing_slip_fraction_max", 0.68, "stepping/swing_slip_fraction_max"),
        ("step_length_over_leg_min", 0.0, "stepping/step_length_over_leg_min"),
        ("step_length_over_leg_min", -0.08, "stepping/step_length_over_leg_min"),
        ("step_length_over_leg_min", None, "stepping/step_length_over_leg_min"),
        ("step_through_stride_fraction_min", 0.5, "stepping/step_through_stride_fraction_min"),
        ("step_through_stride_fraction_min", None, "stepping/step_through_stride_fraction_min"),
        ("step_symmetry", 0.05, "stepping/step_symmetry"),
        ("step_symmetry", None, "stepping/step_symmetry"),
        ("body_frame_step_to_symmetry", 0.2, "stepping/body_frame_step_to_symmetry"),
        ("templates", {}, "persistence/template"),
        ("templates.alternation.phase_locking_min", 0.3, "coupling/phase_locking_min"),
        ("templates.alternation.alternating_overlap_index_max", 0.9, "coupling/alternating_overlap_index_max"),
        ("templates.alternation.off_gait_fraction", 0.16, "persistence/off_gait_fraction"),
        ("templates.alternation.off_gait_fraction", None, "persistence/off_gait_fraction"),
    ],
)
def test_hops_slides_falls_and_unmeasured_episodes_fail_on_their_rail(path, value, rail):
    thresholds = GaitGateThresholds.from_curriculum(curriculum())
    passed, failures = classify_gait_episode(_with(episode(), path, value), thresholds, foot_names=("r", "l"))
    assert not passed
    assert rail in [rail_id(reason) for reason in failures]


def test_step_through_is_required_by_every_profile():
    """A step-to gait (a foot never lands ahead of the other) is refused even by the run-allowed profile."""
    step_to = episode(step_length_over_leg_min=-0.02)
    for profile, kind, feet in (
        ("biped_walk", "biped", ("r", "l")),
        ("biped_alternating", "biped", ("r", "l")),
        ("quadruped_walk", "walk", QUADRUPED),
    ):
        thresholds = GaitGateThresholds.from_curriculum(curriculum(profile))
        record = _with(episode(kind=kind), "step_length_over_leg_min", -0.02)
        passed, failures = classify_gait_episode(record, thresholds, foot_names=feet)
        assert not passed and [rail_id(reason) for reason in failures] == ["stepping/step_length_over_leg_min"]
    assert "step-to" in describe_gait_episode(step_to, ("stepping/step_length_over_leg_min: -0.02 < 0.05",))


def test_a_crabbing_step_to_is_judged_only_when_the_feet_come_together_along_the_trunk():
    """``body_frame_step_to_symmetry`` is null unless a foot lands beside or behind the other along the trunk."""
    thresholds = GaitGateThresholds.from_curriculum(curriculum("biped_alternating"))
    clean = episode()
    assert clean["body_frame_step_to_symmetry"] is None
    assert classify_gait_episode(clean, thresholds, foot_names=("r", "l"))[0]
    even = episode(body_frame_step_to_symmetry=0.5)
    assert classify_gait_episode(even, thresholds, foot_names=("r", "l"))[0]
    lopsided = episode(body_frame_step_to_symmetry=0.22)
    passed, failures = classify_gait_episode(lopsided, thresholds, foot_names=("r", "l"))
    assert not passed and [rail_id(reason) for reason in failures] == ["stepping/body_frame_step_to_symmetry"]
    assert "crabbing path" in describe_gait_episode(lopsided, failures)


def test_light_toe_contacts_cannot_hide_flight_from_the_walk_profiles():
    """Loaded support: a run bridged by toe contact has little contact flight but much unloaded time."""
    bridged = episode(flight_fraction=0.08, unloaded_fraction=0.24)
    walk = GaitGateThresholds.from_curriculum(curriculum("biped_walk"))
    passed, failures = classify_gait_episode(bridged, walk, foot_names=("r", "l"))
    assert not passed and [rail_id(reason) for reason in failures] == ["support/unloaded_fraction"]
    alternating = GaitGateThresholds.from_curriculum(curriculum("biped_alternating"))
    assert classify_gait_episode(bridged, alternating, foot_names=("r", "l"))[0]
    # the run-allowed profile does not consume the walking-support bars
    with pytest.raises(ValueError, match="does not consume"):
        GaitGateThresholds.from_curriculum(curriculum("biped_alternating", max_unloaded_fraction=0.15))


def test_running_is_refused_by_the_walk_profile_and_accepted_by_the_run_allowed_one():
    """Short stances with flight: a run. Only the lenient alternating profile certifies it."""
    run = episode(flight_fraction=0.3, limb_duty_min=0.32, max_sample_interval_s=0.002)
    walk = GaitGateThresholds.from_curriculum(curriculum("biped_walk"))
    passed, failures = classify_gait_episode(run, walk, foot_names=("r", "l"))
    assert not passed
    assert sorted(rail_id(reason) for reason in failures) == ["support/flight_fraction", "support/walking_duty"]
    alternating = GaitGateThresholds.from_curriculum(curriculum("biped_alternating"))
    assert classify_gait_episode(run, alternating, foot_names=("r", "l"))[0]
    # a walking duty just under one half (one foot at 0.49, as compsognathus walks) is still a walk
    assert classify_gait_episode(episode(limb_duty_min=0.49), walk, foot_names=("r", "l"))[0]


def test_duration_floor_tolerates_float_accumulation_but_not_a_short_episode():
    thresholds = GaitGateThresholds.from_curriculum(curriculum(min_episode_duration_s=19.0))
    accumulated = episode(duration_s=18.999999999999794, max_sample_interval_s=0.002)
    assert classify_gait_episode(accumulated, thresholds, foot_names=("r", "l"))[0]
    frame_skip = episode(duration_s=18.99, max_sample_interval_s=0.01)
    assert classify_gait_episode(frame_skip, thresholds, foot_names=("r", "l"))[0]
    short = episode(duration_s=18.9, max_sample_interval_s=0.01)
    assert not classify_gait_episode(short, thresholds, foot_names=("r", "l"))[0]


def test_off_gait_budget_is_one_window_fraction_for_every_profile():
    """Walk-first leniency: one budget (15% of the window) for every way of being off the gait."""
    for profile, kind, names in (
        ("biped_walk", "biped", ("r", "l")),
        ("biped_alternating", "biped", ("r", "l")),
        ("quadruped_walk", "walk", QUADRUPED),
    ):
        thresholds = GaitGateThresholds.from_curriculum(curriculum(profile))
        within = _with(episode(kind=kind), "templates.alternation.off_gait_fraction", 0.14)
        within["templates"]["alternation"]["off_gait_strides"] = 1.5
        assert classify_gait_episode(within, thresholds, foot_names=names)[0]
        beyond = _with(within, "templates.alternation.off_gait_fraction", 0.16)
        passed, failures = classify_gait_episode(beyond, thresholds, foot_names=names)
        assert not passed and [rail_id(reason) for reason in failures] == ["persistence/off_gait_fraction"]
        # its components are diagnostics, never separate rails
        diagnostic = copy.deepcopy(within)
        for component in ("standing_fraction", "in_place_stride_fraction", "asymmetric_stride_fraction"):
            diagnostic["templates"]["alternation"][component] = 0.14
        assert classify_gait_episode(diagnostic, thresholds, foot_names=names)[0]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("girdle_load_share_min", 0.16),
        ("girdle_duty_ratio", 0.36),
        ("girdle_load_share_min", None),
        ("girdle_unloaded_fraction", 0.2),
        ("girdle_unloaded_fraction", None),
    ],
)
def test_quadruped_girdle_participation_rails(key, value):
    thresholds = GaitGateThresholds.from_curriculum(curriculum("quadruped_walk"))
    for kind in ("walk", "trot", "pace"):
        assert classify_gait_episode(episode(kind=kind), thresholds, foot_names=QUADRUPED)[0]
        passed, failures = classify_gait_episode(
            _with(episode(kind=kind), key, value), thresholds, foot_names=QUADRUPED
        )
        assert not passed
        assert [rail_id(reason) for reason in failures] == [f"participation/{key}"]


@pytest.mark.parametrize("kind", ["walk", "trot", "pace"])
def test_quadruped_walk_accepts_any_symmetrical_walking_gait(kind):
    """No limb-phase partition: lateral-sequence walks and walking trots/paces at duty 0.7 all certify."""
    thresholds = GaitGateThresholds.from_curriculum(curriculum("quadruped_walk"))
    record = episode(kind=kind)
    assert classify_gait_episode(record, thresholds, foot_names=QUADRUPED)[0]
    # the Hildebrand label is a report-only diagnostic
    assert describe_gait_episode(record) == record["gait_label"]


def test_foot_registry_must_match_the_profile():
    thresholds = GaitGateThresholds.from_curriculum(curriculum())
    assert not classify_gait_episode(episode(), thresholds, foot_names=("r", "r"))[0]
    assert not classify_gait_episode(episode(), thresholds, foot_names=QUADRUPED)[0]
    quadruped = GaitGateThresholds.from_curriculum(curriculum("quadruped_walk"))
    assert not classify_gait_episode(episode(kind="walk"), quadruped, foot_names=("r", "l"))[0]


def test_panel_joint_success_uses_exact_bound_and_cannot_pool_speed_and_survival():
    thresholds = GaitGateThresholds.from_curriculum(curriculum())
    passing = [episode(i) for i in range(37)]
    failing = [episode(i, completed_horizon=False) for i in range(37, 40)]
    assert evaluate_gait_gate(passing + failing, thresholds, foot_names=("r", "l")).passed
    assert not evaluate_gait_gate(
        passing[:-1] + failing + [episode(99, mean_speed_mps=0.0)], thresholds, foot_names=("r", "l")
    ).passed
    assert not evaluate_gait_gate(passing, thresholds, foot_names=("r", "l")).passed
    assert not evaluate_gait_gate([episode(i) for i in range(41)], thresholds, foot_names=("r", "l")).passed
    empty = evaluate_gait_gate([], thresholds, foot_names=("r", "l"))
    assert not empty.passed and empty.n_episodes == 0


def test_reward_rail_is_same_panel_and_missing_reward_refuses():
    thresholds = GaitGateThresholds.from_curriculum(curriculum(min_avg_reward=900.0))
    assert evaluate_gait_gate([episode(i) for i in range(40)], thresholds, foot_names=("r", "l")).passed
    assert not evaluate_gait_gate(
        [episode(i, reward=800.0) for i in range(40)], thresholds, foot_names=("r", "l")
    ).passed
    assert not evaluate_gait_gate(
        [episode(i, reward=None) for i in range(40)], thresholds, foot_names=("r", "l")
    ).passed


def test_schema_hashes_protocol_and_profile_but_excludes_diagnostic_panel_override():
    block = curriculum()
    assert validate_gate_config(2, block) == GAIT_GATE_KIND
    view = gate_config_view(block)
    assert not gate_config_differences(view["thresholds"], view)
    assert gate_config_sha256(view) == gate_config_sha256(gate_config_view(curriculum(gait_report_episodes=2)))
    changed = curriculum(measurement_protocol_sha256="sha256:" + "c" * 64)
    assert gate_config_sha256(view) != gate_config_sha256(gate_config_view(changed))
    assert "measurement_protocol_sha256" in gate_config_differences(view["thresholds"], gate_config_view(changed))[0]
    with pytest.raises(GateSchemaError, match="does not consume"):
        validate_gate_config(2, dict(block, gate_kind="reward_and_length/v1", min_avg_reward=100.0))


def test_manager_and_history_refuse_ordinary_reward_evidence():
    thresholds = thresholds_from_configs({2: {"curriculum_kwargs": curriculum(required_consecutive=1)}})
    manager = CurriculumManager(species="trex", stage_thresholds=thresholds, start_stage=2)
    for _ in range(3):
        assert not manager.should_advance([1e9] * 40, [1000.0] * 40)
    assert manager.summary()["consecutive_passes"][2] == 0
    assert evaluate_recorded_gate(curriculum(), [{"mean_reward": 1e9, "mean_episode_length": 1000}]) is None
    passed, failures = evaluate_stage_gate(curriculum(), {"best_model_reward": 1e9}, stage=2)
    assert not passed and "stage_dir" in failures[0]


@lru_cache(maxsize=1)
def _stepping_base():
    """A clean 10 s stepping trace at the T. rex timestep, scaled to its body weight."""
    morphology, physics_dt, _ = _authored_morphology()
    trace = stepping_trace(duration=10.0, dt=physics_dt)
    trace["floor_force_n"] *= morphology["body_weight_n"] / BW
    trace["touch_force_n"] *= morphology["body_weight_n"] / BW
    return trace


@lru_cache(maxsize=None)
def _seed_tied(seed: int):
    """``(trace, metrics, reset digest)`` of a stepping episode that starts from ``reset(seed)``.

    The reader re-resets every panel seed, so the trace stores that seed's
    reset state and its first sample is that state; the stepping that
    follows is the synthetic gait (whose analysis window starts at 1 s).
    """
    from environments.shared.gait.morphology import GaitMorphology
    from environments.shared.gait.report import json_safe, reset_state_digest
    from environments.trex.envs.trex_env import TRexEnv

    morphology, _, _ = _authored_morphology()
    trace = {key: np.array(value, copy=True) for key, value in _stepping_base().items()}
    env = TRexEnv()
    try:
        env.reset(seed=seed)
        data = env.data
        morph = GaitMorphology.from_env(env, "trex")
        root = morph.root_qpos_address
        trace["root_position_m"][0] = data.qpos[root : root + 3]
        trace["root_quat_wxyz"][0] = data.qpos[root + 3 : root + 7]
        trace["foot_position_m"][0] = data.site_xpos[list(morph.foot_site_ids)]
        trace["reset_qpos"] = np.array(data.qpos, dtype=np.float64)
        trace["reset_qvel"] = np.array(data.qvel, dtype=np.float64)
        trace["reset_mocap_pos"] = np.array(data.mocap_pos, dtype=np.float64)
    finally:
        env.close()
    trace["physics_diverged"] = np.asarray(False)
    metrics = json_safe(
        episode_gait_metrics(
            trace,
            body_weight_n=morphology["body_weight_n"],
            leg_length_m=morphology["leg_length_m"],
            foot_names=("r", "l"),
            protocol=GaitProtocol(),
            settle_s=1.0,
        )
    )
    digest = reset_state_digest(trace["reset_qpos"], trace["reset_qvel"], trace["reset_mocap_pos"])
    return trace, json.dumps(metrics), digest


def _panel(root: Path, *, episodes=None, trace_for=None):
    """A certifying selected-handoff stage directory whose every trace is tied to its seed.

    ``trace_for(index, seed, trace)`` may replace an episode's trace before
    it is written; the stored metrics are then recomputed from it.
    """
    models = root / "models"
    models.mkdir()
    checkpoint = models / "robust_best_model.zip"
    normalization = models / "robust_best_model_vecnorm.pkl"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({"seed": 42, "n_envs": 4, "mesozoic_task_fingerprint": TASK_RECORD}))
    normalization.write_bytes(b"selected statistics")
    (root / "stage_config.json").write_text(json.dumps({"reward_weights": {}, "run": {"seed": 42, "n_envs": 4}}))
    (root / "task_fingerprint.json").write_text(json.dumps(TASK_RECORD))
    measured = protocol()
    digests = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "normalization_sha256": sha256_file(normalization),
        "task_sha256": TASK,
        "measurement_protocol_sha256": canonical_json_sha256(measured),
    }
    morphology, _, _ = _authored_morphology()
    morphology = copy.deepcopy(morphology)
    metadata = episodes if episodes is not None else [episode(PUBLICATION_SEED_START + i) for i in range(40)]
    rows = []
    traces = []
    for i, recorded in enumerate(metadata):
        trace, derived_json, reset_digest = _seed_tied(recorded["seed"])
        derived = json.loads(derived_json)
        trace = {key: np.array(value, copy=True) for key, value in trace.items()}
        if trace_for is not None:
            trace = trace_for(i, recorded["seed"], trace)
            from environments.shared.gait.report import json_safe

            derived = json_safe(
                episode_gait_metrics(
                    trace,
                    body_weight_n=morphology["body_weight_n"],
                    leg_length_m=morphology["leg_length_m"],
                    foot_names=("r", "l"),
                    protocol=GaitProtocol(),
                    settle_s=1.0,
                )
            )
        rows.append(
            {
                **derived,
                "seed": recorded["seed"],
                "episode": i,
                "length": 1000,
                "completed_horizon": recorded["completed_horizon"],
                "reward": recorded["reward"],
                "terminated": not recorded["completed_horizon"],
                "truncated": recorded["completed_horizon"],
                "physics_diverged": False,
                "reset_state_sha256": reset_digest,
            }
        )
        relative = f"gait_traces/episode_{i:04d}.npz"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **trace)
        traces.append({"path": relative, "sha256": sha256_file(path)})
    panel = root / "gait_panel.csv"
    with panel.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*digests, "metrics_json"])
        writer.writeheader()
        writer.writerows({**digests, "metrics_json": json.dumps(row)} for row in rows)
    report = {
        "schema": REPORT_SCHEMA,
        "species": "trex",
        "stage": 2,
        "plant_identity": current_plant_identity("trex").to_dict(),
        "task_fingerprint": TASK_RECORD,
        "status": "complete",
        "plant_validated": True,
        "task_validated": True,
        "certification_eligible": True,
        "report_only": False,
        "seed_start": PUBLICATION_SEED_START,
        **digests,
        "measurement_protocol": measured,
        "gait_profile": "biped_walk",
        "foot_names": ["r", "l"],
        "morphology": morphology,
        "seed_provenance": checkpoint_seed_provenance(
            checkpoint, seed_start=PUBLICATION_SEED_START, episodes=40, stage=2, species="trex"
        ),
        "traces": traces,
        "episodes": rows,
        "panel_csv": {"path": panel.name, "sha256": sha256_file(panel)},
        "passed": True,
    }
    (root / "gait_report.json").write_text(json.dumps(report))
    return report


def test_selected_handoff_panel_rejudges_and_persists_statistics(tmp_path):
    _panel(tmp_path)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert not failures and stats is not None
    assert stats["selected_gait_success_count"] == stats["selected_gait_n_episodes"] == 40
    assert stats["selected_gait_success_lcb"] > 0.9
    assert evaluate_stage_gate(curriculum(), {}, stage=2, stage_dir=tmp_path)[0]


@pytest.mark.parametrize(
    "mutation",
    [
        "checkpoint",
        "normalization",
        "task",
        "protocol",
        "protocol_payload",
        "csv",
        "episodes",
        "profile",
        "schema",
        "status",
        "plant_validated",
        "task_validated",
        "certification_eligible",
        "report_only",
        "seed_start",
        "foot_names",
    ],
)
def test_changed_evidence_and_stale_identity_refuse(tmp_path, mutation):
    report = _panel(tmp_path)
    if mutation in {"checkpoint", "normalization"}:
        name = "robust_best_model.zip" if mutation == "checkpoint" else "robust_best_model_vecnorm.pkl"
        (tmp_path / "models" / name).write_bytes(b"different")
    elif mutation == "csv":
        with (tmp_path / "gait_panel.csv").open("a") as handle:
            handle.write("tampered")
    elif mutation == "task":
        (tmp_path / "task_fingerprint.json").unlink()
    else:
        report = copy.deepcopy(report)
        if mutation == "protocol":
            report["measurement_protocol_sha256"] = "sha256:" + "c" * 64
        elif mutation == "protocol_payload":
            report["measurement_protocol"]["schema"] = "new"
        elif mutation == "episodes":
            report["episodes"][0]["mean_speed_mps"] = 50.0
        elif mutation == "profile":
            report["gait_profile"] = "quadruped_walk"
        elif mutation == "status":
            report["status"] = "incomplete"
        elif mutation in {"plant_validated", "task_validated", "certification_eligible"}:
            report[mutation] = False
        elif mutation == "report_only":
            report["report_only"] = True
        elif mutation == "seed_start":
            report["seed_start"] = 42
        elif mutation == "foot_names":
            report["foot_names"] = ["l", "r"]
        else:
            report["schema"] = "mesozoic.gait-report/v0"
        (tmp_path / "gait_report.json").write_text(json.dumps(report))
    passed, failures = evaluate_stage_gate(curriculum(), {"best_model_reward": 1e9}, stage=2, stage_dir=tmp_path)
    assert not passed and failures


def test_duplicate_rollouts_and_claimed_pass_never_certify(tmp_path):
    _panel(tmp_path, episodes=[episode(0) for _ in range(40)])
    assert not evaluate_stage_gate(curriculum(), {}, stage=2, stage_dir=tmp_path)[0]
    other = tmp_path / "other"
    other.mkdir()
    _panel(other, episodes=[episode(PUBLICATION_SEED_START + i, completed_horizon=False) for i in range(40)])
    stats, failures = gait_statistics(other, curriculum())
    assert not failures and stats is not None and not stats["passed"]
    assert stats["selected_gait_success_count"] == 0


def test_stage_artifacts_write_selected_gait_statistics_into_verdict_and_summary(tmp_path):
    from environments.shared.reporting.stage_artifacts import _apply_stage_gate
    from environments.shared.reporting.summaries import _canonical_stage_summary

    _panel(tmp_path)
    result = {"stage": 2, "timesteps": 0}
    _apply_stage_gate(
        stage=2,
        stage_config={"curriculum_kwargs": curriculum()},
        stage_results=result,
        stance_report=None,
        stage_dir=tmp_path,
        species="trex",
    )
    assert result["gate_passed"]
    verdict = json.loads((tmp_path / "gate_verdict.json").read_text())
    assert verdict["stage_result"]["selected_gait_success_count"] == 40
    summary = _canonical_stage_summary(result)
    assert summary["selected_gait_success_count"] == summary["selected_gait_n_episodes"] == 40
    assert summary["selected_gait_success_lcb"] > 0.9


def _fake_report_module(monkeypatch, observed):
    from environments.shared.gait import report as real

    module = ModuleType("environments.shared.gait.report")

    def fake_writer(*args, **kwargs):
        observed.append((args, kwargs))
        return {"status": "complete"}

    module.write_gait_report = fake_writer
    module.stage_panel = real.stage_panel
    module.clear_panel_files = real.clear_panel_files
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_report_generation_uses_exact_selected_pair_and_fixed_gate_seed(tmp_path, monkeypatch):
    from environments.shared.reporting.stage_artifacts import _write_gait_report

    _panel(tmp_path)
    observed = []
    _fake_report_module(monkeypatch, observed)
    config = {"curriculum_kwargs": curriculum()}
    result = _write_gait_report(
        species_cfg=SimpleNamespace(species="trex"),
        stage=2,
        stage_config=config,
        stage_dir=tmp_path,
        model_dir=tmp_path / "models",
        algorithm="PPO",
    )
    assert result == {"status": "complete"}
    args, kwargs = observed.pop()
    assert Path(args[2]).name == "robust_best_model.zip"
    assert Path(args[3]).name == "robust_best_model_vecnorm.pkl"
    assert args[1]["_gait_stage"] == 2 and "_gait_stage" not in config
    assert kwargs["episodes"] == 40 and kwargs["seed"] == PUBLICATION_SEED_START
    assert (
        _write_gait_report(
            species_cfg=SimpleNamespace(species="trex"),
            stage=1,
            stage_config={"curriculum_kwargs": {}},
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
            algorithm="PPO",
        )
        is None
    )
    assert not observed


@pytest.mark.parametrize(
    "updates,development_panel,expected",
    [
        ({}, True, (10, 9000)),  # report-only: a short panel on the development block
        ({"gait_report_episodes": 4}, True, (4, 9000)),
        ({"gait_report_episodes": 0}, True, None),
        ({}, False, None),  # development diagnostics switched off (generate_graphs=False)
    ],
)
def test_report_only_panels_roll_the_development_block_never_the_certification_block(
    tmp_path, monkeypatch, updates, development_panel, expected
):
    from environments.shared.constants import DEVELOPMENT_GAIT_SEED_START
    from environments.shared.reporting.stage_artifacts import _write_gait_report

    _panel(tmp_path)
    observed = []
    _fake_report_module(monkeypatch, observed)
    config = {
        "curriculum_kwargs": {
            "gate_kind": "reward_and_length/v1",
            "gate_schema_version": 1,
            "min_avg_reward": 1.0,
            **updates,
        }
    }
    result = _write_gait_report(
        species_cfg=SimpleNamespace(species="trex"),
        stage=2,
        stage_config=config,
        stage_dir=tmp_path,
        model_dir=tmp_path / "models",
        algorithm="PPO",
        development_panel=development_panel,
    )
    if expected is None:
        assert result is None and not observed
    else:
        _, kwargs = observed.pop()
        assert (kwargs["episodes"], kwargs["seed"]) == expected
        assert kwargs["seed"] == DEVELOPMENT_GAIT_SEED_START
        assert not PUBLICATION_SEED_START <= kwargs["seed"] < PUBLICATION_SEED_START + 40
    # Any earlier report is invalidated either way.
    assert not (tmp_path / "gait_report.json").exists()


def test_generate_stage_artifacts_ties_development_panel_to_graphs_unless_told(monkeypatch, tmp_path):
    from environments.shared.reporting import stage_artifacts

    calls = []
    monkeypatch.setattr(stage_artifacts, "_write_stance_gate_report", lambda **kwargs: None)
    monkeypatch.setattr(stage_artifacts, "_write_gait_report", lambda **kwargs: calls.append(kwargs) or None)
    monkeypatch.setattr(stage_artifacts, "_write_task_success_evidence", lambda **kwargs: None)
    monkeypatch.setattr(stage_artifacts, "_apply_stage_gate", lambda **kwargs: None)
    monkeypatch.setattr(stage_artifacts, "_run_stance_probes", lambda **kwargs: None)
    monkeypatch.setattr(stage_artifacts.text_summaries, "write_stage_summary", lambda *args, **kwargs: None)
    for graphs, diagnostics, expected in ((True, None, True), (False, None, False), (False, True, True)):
        stage_artifacts.generate_stage_artifacts(
            SimpleNamespace(species="trex"),
            {"curriculum_kwargs": {}},
            2,
            "PPO",
            tmp_path,
            42,
            stage_results={},
            record_videos=False,
            generate_graphs=graphs,
            gait_diagnostics=diagnostics,
        )
        assert calls.pop()["development_panel"] is expected


@pytest.mark.parametrize(
    "updates",
    [
        {"min_eval_episodes": 40.0},  # a float count validated, then never matched a rolled panel
        {"min_complete_cycles_per_foot": 3.0},
        {"gait_panel_seed_start": 3042.0},
        {"required_consecutive": 1.0},
        {"required_consecutive": True},
        {"min_eval_episodes": True},
    ],
)
def test_integer_criteria_must_be_integers(updates):
    with pytest.raises((GateSchemaError, ValueError), match="integer"):
        validate_gate_config(2, curriculum(**updates))
    with pytest.raises(ValueError):
        GaitGateThresholds.from_curriculum(curriculum(**updates))


@pytest.mark.parametrize("value,valid", [(40, True), (39, False), (0, False), (40.0, False)])
def test_gait_report_episodes_on_a_gait_gate_must_equal_the_declared_panel(value, valid):
    block = curriculum(gait_report_episodes=value)
    if valid:
        assert validate_gate_config(2, block) == GAIT_GATE_KIND
    else:
        with pytest.raises(GateSchemaError, match="gait_report_episodes"):
            validate_gate_config(2, block)


def test_certification_panel_must_lie_inside_the_registered_block():
    with pytest.raises(GateSchemaError, match="registered block"):
        validate_gate_config(2, curriculum(min_eval_episodes=41))


def test_publication_rederives_gait_and_rejects_unbound_or_laundered_claims(tmp_path):
    from environments.shared.result_bundle.errors import ResultBundleError
    from environments.shared.result_bundle.evidence import _validate_gait_evidence

    report = _panel(tmp_path)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert not failures and stats is not None
    params = {
        "certified_hash": report["checkpoint_sha256"],
        "certified_normalization": report["normalization_sha256"],
        "stage": 2,
        "stage_summary": stats,
        "panel_seed_start": PUBLICATION_SEED_START,
    }
    _validate_gait_evidence(tmp_path, curriculum(), **params)
    for updates in (
        {"certified_hash": None},
        {"certified_normalization": "sha256:" + "c" * 64},
        {"panel_seed_start": 1},
        {"stage_summary": dict(stats, selected_gait_success_count=39)},
        {"stage_summary": dict(stats, selected_gait_success_lcb=1.0)},
    ):
        with pytest.raises(ResultBundleError):
            _validate_gait_evidence(tmp_path, curriculum(), **dict(params, **updates))
    report["report_only"] = True
    (tmp_path / "gait_report.json").write_text(json.dumps(report))
    with pytest.raises(ResultBundleError, match="report-only"):
        _validate_gait_evidence(tmp_path, curriculum(), **params)


@pytest.mark.parametrize("reason", ["invalid_gate", "no_handoff", "development_off", "no_development_episodes"])
def test_failed_fresh_generation_invalidates_older_certificate(tmp_path, reason):
    """A skipped or failed panel removes the earlier report with every earlier panel file (PL-7)."""
    from environments.shared.reporting.stage_artifacts import _write_gait_report

    _panel(tmp_path)
    assert (tmp_path / "gait_panel.csv").is_file() and any((tmp_path / "gait_traces").iterdir())
    config = {"curriculum_kwargs": curriculum()}
    development_panel = True
    if reason == "invalid_gate":
        config["curriculum_kwargs"]["gait_report_episodes"] = 0
    elif reason == "no_handoff":
        (tmp_path / "models" / "robust_best_model_vecnorm.pkl").unlink()
    else:
        # A report-only locomotion stage whose development panel is not rolled.
        config["curriculum_kwargs"] = {"gate_kind": "reward_and_length/v1", "min_avg_reward": 1.0}
        if reason == "development_off":
            development_panel = False
        else:
            config["curriculum_kwargs"]["gait_report_episodes"] = 0
    assert (
        _write_gait_report(
            species_cfg=SimpleNamespace(species="trex"),
            stage=2,
            stage_config=config,
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
            algorithm="PPO",
            development_panel=development_panel,
        )
        is None
    )
    assert not (tmp_path / "gait_report.json").exists()
    assert not (tmp_path / "gait_panel.csv").exists() and not (tmp_path / "gait_traces").exists()


def test_immutable_bundle_guard_precedes_certificate_invalidation(tmp_path):
    from environments.shared.reporting.stage_artifacts import _write_gait_report
    from environments.shared.result_bundle.errors import ResultBundleError

    _panel(tmp_path)
    before = (tmp_path / "gait_report.json").read_bytes()
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ResultBundleError, match="immutable"):
        _write_gait_report(
            species_cfg=SimpleNamespace(species="trex"),
            stage=2,
            stage_config={"curriculum_kwargs": curriculum(gait_report_episodes=0)},
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
            algorithm="PPO",
        )
    assert (tmp_path / "gait_report.json").read_bytes() == before


@pytest.mark.parametrize("stale", ["implementation", "registry"])
def test_old_measurement_code_or_registry_cannot_certify_even_with_matching_declared_digest(tmp_path, stale):
    report = _panel(tmp_path)
    if stale == "implementation":
        report["measurement_protocol"]["implementation_sha256"]["environments/shared/gait/metrics.py"] = (
            "sha256:" + "c" * 64
        )
    else:
        report["measurement_protocol"]["foot_geometries"]["r"] = ["unknown-foot"]
    digest = canonical_json_sha256(report["measurement_protocol"])
    report["measurement_protocol_sha256"] = digest
    path = tmp_path / "gait_panel.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    for row in rows:
        row["measurement_protocol_sha256"] = digest
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report["panel_csv"]["sha256"] = sha256_file(path)
    (tmp_path / "gait_report.json").write_text(json.dumps(report))
    stats, failures = gait_statistics(tmp_path, curriculum(measurement_protocol_sha256=digest))
    assert stats is None and "stale" in failures[0]


def _rewrite_panel(root, report):
    bindings = {
        key: report[key]
        for key in (
            "checkpoint_sha256",
            "normalization_sha256",
            "task_sha256",
            "measurement_protocol_sha256",
        )
    }
    path = root / "gait_panel.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*bindings, "metrics_json"])
        writer.writeheader()
        writer.writerows({**bindings, "metrics_json": json.dumps(row)} for row in report["episodes"])
    report["panel_csv"]["sha256"] = sha256_file(path)
    (root / "gait_report.json").write_text(json.dumps(report))


@pytest.mark.parametrize(
    "corruption",
    ["missing", "extra", "changed", "escape", "metrics", "reset", "length", "morphology", "seed_provenance"],
)
def test_raw_physics_and_seed_evidence_are_required_and_reproducible(tmp_path, corruption):
    report = _panel(tmp_path)
    first = tmp_path / "gait_traces" / "episode_0000.npz"
    if corruption == "missing":
        first.unlink()
    elif corruption == "extra":
        (first.parent / "episode_0040.npz").write_bytes(first.read_bytes())
    elif corruption == "changed":
        with first.open("ab") as handle:
            handle.write(b"changed")
    elif corruption == "escape":
        report["traces"][0]["path"] = "../episode_0000.npz"
    elif corruption == "metrics":
        report["episodes"][0]["mean_speed_mps"] = 100.0
    elif corruption == "reset":
        report["episodes"][1]["reset_state_sha256"] = report["episodes"][0]["reset_state_sha256"]
    elif corruption == "length":
        report["episodes"][0]["length"] = 500
    elif corruption == "morphology":
        report["morphology"]["leg_length_m"] *= 2
    else:
        report["seed_provenance"]["training_envs"] = 100
    _rewrite_panel(tmp_path, report)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert stats is None and failures


def test_changed_sampling_protocol_cannot_override_the_authored_plant_timestep(tmp_path):
    report = _panel(tmp_path)
    report["measurement_protocol"]["sampling"]["physics_dt_s"] *= 2
    report["measurement_protocol_sha256"] = canonical_json_sha256(report["measurement_protocol"])
    _rewrite_panel(tmp_path, report)
    stats, failures = gait_statistics(
        tmp_path, curriculum(measurement_protocol_sha256=report["measurement_protocol_sha256"])
    )
    assert stats is None and "timestep" in failures[0]


@pytest.mark.parametrize("updates", [{"min_episode_forward_vel": 0}, {"required_consecutive": 2}])
def test_strict_locomotion_requires_progress_and_one_fixed_panel(updates):
    with pytest.raises(GateSchemaError):
        validate_gate_config(2, curriculum(**updates))


def _float_paths(record, prefix=""):
    """Every (path, value) of a float in a nested metrics record."""
    if isinstance(record, dict):
        for key, value in record.items():
            yield from _float_paths(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(record, float):
        yield prefix, record


def test_reader_accepts_last_digit_differences_and_judges_the_replayed_metrics(tmp_path, monkeypatch):
    """A replay on another machine or BLAS kernel may differ in the last digit of a stored value.

    The stored numbers only have to agree with the replay within one
    six-decimal storage quantum; the verdict is formed from the replayed
    values, so nudging a stored number cannot move it.
    """
    from environments.shared.curriculum import gait_gate

    report = _panel(tmp_path)
    nudged = 0
    for row in report["episodes"]:
        for path, value in list(_float_paths(row)):
            if path in {"reward"} or not value:
                continue
            target = row
            keys = path.split(".")
            for key in keys[:-1]:
                target = target[key]
            # one ulp on some values, one storage quantum (1e-6) on others
            target[keys[-1]] = float(np.nextafter(value, np.inf)) if nudged % 2 else value + 1e-6
            nudged += 1
    assert nudged > 100
    _rewrite_panel(tmp_path, report)
    judged = []
    real = gait_gate.evaluate_gait_gate

    def capture(episodes, *args, **kwargs):
        judged.extend(episodes)
        return real(episodes, *args, **kwargs)

    monkeypatch.setattr(gait_gate, "evaluate_gait_gate", capture)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert not failures and stats is not None and stats["passed"]
    replayed = json.loads(_seed_tied(PUBLICATION_SEED_START)[1])
    assert judged[0]["mean_speed_mps"] == replayed["mean_speed_mps"]
    assert judged[0]["mean_speed_mps"] != report["episodes"][0]["mean_speed_mps"]


@pytest.mark.parametrize(
    "path,value",
    [
        ("complete_cycles_min", 2),  # a count must reproduce exactly
        ("per_foot.r.complete_cycles", 1),
        ("telemetry_valid", False),
        ("gait_label", "pronk"),
        ("mean_speed_mps", 0.5),  # far outside the storage quantum
    ],
)
def test_reader_refuses_a_count_label_or_value_that_does_not_reproduce(tmp_path, path, value):
    report = _panel(tmp_path)
    report["episodes"][3] = _with(report["episodes"][3], path, value)
    _rewrite_panel(tmp_path, report)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert stats is None and "do not reproduce" in failures[0]


def _rewrite_trace(root, report, index, trace):
    path = root / report["traces"][index]["path"]
    np.savez_compressed(path, **trace)
    report["traces"][index]["sha256"] = sha256_file(path)


def _load_trace(root, report, index):
    with np.load(root / report["traces"][index]["path"]) as archive:
        return {key: archive[key] for key in archive.files}


@pytest.mark.parametrize("mutation", ["duplicate", "swap", "fabricated_reset", "moved_first_sample", "copied_reset"])
def test_reader_ties_every_trace_to_its_own_seed(tmp_path, mutation):
    """Duplicated, swapped or relabelled rollouts are not independent seeded episodes."""
    report = _panel(tmp_path)
    first, second = _load_trace(tmp_path, report, 0), _load_trace(tmp_path, report, 1)
    if mutation == "duplicate":
        # episode 1 replaced by a copy of episode 0, its metrics copied too
        _rewrite_trace(tmp_path, report, 1, first)
        report["episodes"][1] = {**report["episodes"][0], "seed": report["episodes"][1]["seed"], "episode": 1}
        report["episodes"][1]["reset_state_sha256"] = canonical_json_sha256({"other": 1})
        match = "duplicates another episode's trace"
    elif mutation == "swap":
        _rewrite_trace(tmp_path, report, 0, second)
        _rewrite_trace(tmp_path, report, 1, first)
        for key in ("reset_state_sha256",):
            report["episodes"][0][key], report["episodes"][1][key] = (
                report["episodes"][1][key],
                report["episodes"][0][key],
            )
        match = "fresh reset of seed"
    elif mutation == "fabricated_reset":
        report["episodes"][2]["reset_state_sha256"] = "sha256:" + "1" * 64
        match = "reset digest"
    elif mutation == "moved_first_sample":
        moved = _load_trace(tmp_path, report, 2)
        moved["root_position_m"][0, 0] += 0.01
        _rewrite_trace(tmp_path, report, 2, moved)
        match = "first sample"
    else:
        # episode 2's stored reset state replaced by episode 3's (and its digest)
        relabelled = _load_trace(tmp_path, report, 2)
        third = _load_trace(tmp_path, report, 3)
        for key in ("reset_qpos", "reset_qvel", "reset_mocap_pos"):
            relabelled[key] = third[key]
        _rewrite_trace(tmp_path, report, 2, relabelled)
        report["episodes"][2]["reset_state_sha256"] = report["episodes"][3]["reset_state_sha256"]
        match = "missing or duplicated"
    _rewrite_panel(tmp_path, report)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert stats is None and match in failures[0]


def test_reader_requires_the_reset_state_and_divergence_record(tmp_path):
    report = _panel(tmp_path)
    trace = _load_trace(tmp_path, report, 0)
    del trace["reset_qvel"]
    _rewrite_trace(tmp_path, report, 0, trace)
    _rewrite_panel(tmp_path, report)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert stats is None and "reset state or divergence record" in failures[0]


def test_reader_accepts_a_diverged_episode_as_a_failed_one(tmp_path):
    """A MuJoCo divergence ends that episode's trace; the panel stays readable and the episode fails."""

    def diverge(index, seed, trace):
        if index != 5:
            return trace
        cut = 2001  # 4 s of 10 s
        trace = {key: (value[:cut] if value.ndim and len(value) == 5001 else value) for key, value in trace.items()}
        trace["physics_diverged"] = np.asarray(True)
        return trace

    report = _panel(tmp_path, trace_for=diverge)
    report["episodes"][5].update(physics_diverged=True, completed_horizon=False)
    _rewrite_panel(tmp_path, report)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert not failures and stats is not None
    assert stats["selected_gait_success_count"] == 39
    assert stats["episode_failures"][5] == ["episode/telemetry_valid: telemetry is invalid or missing"]
    # The divergence flag must agree with the trace.
    report["episodes"][5]["physics_diverged"] = False
    _rewrite_panel(tmp_path, report)
    stats, failures = gait_statistics(tmp_path, curriculum())
    assert stats is None and "divergence record" in failures[0]


@pytest.mark.parametrize(
    "key,value",
    [
        ("min_phase_locking", 0.0),  # disables the template: synchronous hops would qualify
        ("max_alternation_phase_offset", 0.5),
        ("max_off_gait_fraction", 1.0),
        ("min_gait_success_lcb", 1e-9),  # 1/40 qualifying would pass
        ("min_complete_cycles_per_foot", 1),
        ("min_limb_duty", 0.0),
        ("min_step_length_over_leg", 0.0),
        ("min_step_through_stride_fraction", 0.0),
        ("max_body_support_fraction", 1.0),
        ("max_flight_fraction", 1.0),
        ("max_unloaded_fraction", 1.0),
        ("min_walking_duty", 0.0),
    ],
)
def test_schema_refuses_bars_that_switch_a_profile_off(key, value):
    with pytest.raises(GateSchemaError, match="vacuous"):
        validate_gate_config(2, curriculum(**{key: value}))


def test_schema_limits_leave_every_provisional_bar_and_the_run_profile_valid():
    from environments.shared.curriculum.gait_gate import FLIGHT_LIMITS, GAIT_PROFILES, SCHEMA_LIMITS

    for profile in GAIT_PROFILES:
        block = curriculum(profile)
        if profile == "quadruped_walk":
            block["min_episode_forward_vel"] = 0.5
        assert validate_gate_config(2, block) == GAIT_GATE_KIND
        for key, value in provisional_gait_criteria(profile).items():
            op, limit = (
                ("<=", FLIGHT_LIMITS[profile]) if key == "max_flight_fraction" else SCHEMA_LIMITS.get(key, ("", 0))
            )
            assert not op or (value >= limit if op == ">=" else value <= limit), (profile, key)
    # running stays allowed on the run profile
    assert validate_gate_config(2, curriculum("biped_alternating", max_flight_fraction=0.75)) == GAIT_GATE_KIND


@pytest.mark.parametrize("order", [("fr", "rr", "fl", "rl"), ("rl", "rr", "fl", "fr")])
def test_feet_are_paired_by_name_whatever_the_recorded_order(order):
    """A sensor reorder must not turn the contralateral pairs into ipsilateral ones."""
    canonical = ("fr", "fl", "rr", "rl")
    trace = gait_trace((0.0, 0.1, 0.5, 0.6), duty=0.7, period=1.0)  # a transverse gallop
    permuted = {
        key: value[:, [canonical.index(name) for name in order]]
        if key in ("floor_force_n", "foot_position_m", "foot_clearance_m", "slip_speed_mps", "touch_force_n")
        else value
        for key, value in trace.items()
    }
    measured = episode_gait_metrics(
        trace, body_weight_n=BW, leg_length_m=1.0, foot_names=canonical, protocol=GaitProtocol(), settle_s=1.0
    )
    reordered = episode_gait_metrics(
        permuted, body_weight_n=BW, leg_length_m=1.0, foot_names=order, protocol=GaitProtocol(), settle_s=1.0
    )
    assert json.dumps(measured, sort_keys=True) == json.dumps(reordered, sort_keys=True)
    thresholds = GaitGateThresholds.from_curriculum(curriculum("quadruped_walk"))
    record = {**json.loads(json.dumps(reordered)), "completed_horizon": True, "seed": 0}
    passed, reasons = classify_gait_episode(record, thresholds, foot_names=order)
    assert (passed, reasons) == classify_gait_episode(record, thresholds, foot_names=canonical)
    assert not passed
