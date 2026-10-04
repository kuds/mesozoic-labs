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


def curriculum(profile="biped_alternating", **updates):
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
    """Copy of an episode with one nested metric replaced (path like 'templates.biped_alternating.x')."""
    result = copy.deepcopy(record)
    target = result
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    return result


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
        ("min_template_coverage", 1.5),
        ("max_alternation_phase_offset", 0.6),
        ("min_complete_cycles_per_foot", 2.5),
        ("max_off_template_run_fraction_ceiling", 0.05),
        ("gait_profile", "automatic"),
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
    for key in ("max_foot_foot_contact_fraction", "min_lead_exchange_fraction", "max_wrong_locked_fraction"):
        block = curriculum()
        del block[key]
        with pytest.raises(ValueError, match=key):
            GaitGateThresholds.from_curriculum(block)


def test_profile_specific_criteria_are_required_and_foreign_ones_refused():
    walk = curriculum("quadruped_walk")
    del walk["min_walk_limb_phase"]
    with pytest.raises(ValueError, match="min_walk_limb_phase"):
        GaitGateThresholds.from_curriculum(walk)
    with pytest.raises(ValueError, match="does not consume"):
        GaitGateThresholds.from_curriculum(curriculum(max_synchrony_phase_offset=0.125))
    with pytest.raises(ValueError, match="does not consume"):
        GaitGateThresholds.from_curriculum(curriculum("quadruped_trot", min_walk_limb_phase=0.125))
    with pytest.raises(ValueError, match="below"):
        GaitGateThresholds.from_curriculum(curriculum("quadruped_walk", min_walk_limb_phase=0.4))
    for profile in ("biped_alternating", "quadruped_walk", "quadruped_trot", "quadruped_pace"):
        assert GaitGateThresholds.from_curriculum(curriculum(profile)).gait_profile == profile


def test_arbitrary_seeds_are_development_only_and_strict_config_uses_registered_block():
    development = curriculum(gait_panel_seed_start=0)
    assert GaitGateThresholds.from_curriculum(development).gait_panel_seed_start == 0
    with pytest.raises(GateSchemaError, match="registered certification"):
        validate_gate_config(2, development)


def test_clean_measured_gaits_qualify_and_reasons_carry_stable_rail_ids():
    thresholds = GaitGateThresholds.from_curriculum(curriculum())
    passed, failures = classify_gait_episode(episode(), thresholds, foot_names=("r", "l"))
    assert passed, failures
    hop = _with(episode(), "templates.biped_alternating.alternation_phase_offset_max", 0.48)
    passed, failures = classify_gait_episode(hop, thresholds, foot_names=("r", "l"))
    assert not passed
    assert [rail_id(reason) for reason in failures] == ["coupling/alternation_phase_offset_max"]
    assert failures[0].endswith("> 0.09")
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
        ("flight_fraction", 0.7, "support/flight_fraction"),
        ("flight_fraction", -1.0, "support/flight_fraction"),
        ("body_support_fraction", 0.02, "support/body_support_fraction"),
        ("foot_foot_contact_fraction", 0.5, "support/foot_foot_contact_fraction"),
        ("limb_duty_min", 0.05, "participation/limb_duty_min"),
        ("relative_limb_load_share_min", 0.06, "participation/relative_limb_load_share_min"),
        ("limb_phase_coverage_min", 0.4, "participation/limb_phase_coverage_min"),
        ("valid_swing_fraction_min", 0.2, "stepping/valid_swing_fraction_min"),
        ("median_swing_clearance_over_leg_min", 0.012, "stepping/median_swing_clearance_over_leg_min"),
        ("lead_exchange_fraction_min", 0.0, "stepping/lead_exchange_fraction_min"),
        ("templates", {}, "persistence/template"),
        ("templates.biped_alternating.phase_locking_min", 0.3, "coupling/phase_locking_min"),
        ("templates.biped_alternating.alternating_overlap_index_max", 0.9, "coupling/alternating_overlap_index_max"),
        ("templates.biped_alternating.template_coverage", 0.5, "persistence/template_coverage"),
        ("templates.biped_alternating.min_segment_coverage", 0.2, "persistence/min_segment_coverage"),
        ("templates.biped_alternating.wrong_locked_fraction", 0.3, "persistence/wrong_locked_fraction"),
        ("templates.biped_alternating.longest_off_template_fraction", 0.3, "persistence/longest_off_template"),
    ],
)
def test_hops_slides_falls_and_unmeasured_episodes_fail_on_their_rail(path, value, rail):
    thresholds = GaitGateThresholds.from_curriculum(curriculum())
    passed, failures = classify_gait_episode(_with(episode(), path, value), thresholds, foot_names=("r", "l"))
    assert not passed
    assert rail in [rail_id(reason) for reason in failures]


def test_duration_floor_tolerates_float_accumulation_but_not_a_short_episode():
    thresholds = GaitGateThresholds.from_curriculum(curriculum(min_episode_duration_s=19.0))
    accumulated = episode(duration_s=18.999999999999794, max_sample_interval_s=0.002)
    assert classify_gait_episode(accumulated, thresholds, foot_names=("r", "l"))[0]
    frame_skip = episode(duration_s=18.99, max_sample_interval_s=0.01)
    assert classify_gait_episode(frame_skip, thresholds, foot_names=("r", "l"))[0]
    short = episode(duration_s=18.9, max_sample_interval_s=0.01)
    assert not classify_gait_episode(short, thresholds, foot_names=("r", "l"))[0]


def test_long_stride_persistence_allows_a_few_strides_but_never_beyond_the_ceiling():
    thresholds = GaitGateThresholds.from_curriculum(curriculum())
    base = episode()
    two_strides = _with(base, "templates.biped_alternating.longest_off_template_fraction", 0.13)
    two_strides["templates"]["biped_alternating"]["longest_off_template_strides"] = 1.5
    assert classify_gait_episode(two_strides, thresholds, foot_names=("r", "l"))[0]
    many = _with(two_strides, "templates.biped_alternating.longest_off_template_strides", 12.0)
    assert not classify_gait_episode(many, thresholds, foot_names=("r", "l"))[0]
    ceiling = _with(two_strides, "templates.biped_alternating.longest_off_template_fraction", 0.16)
    assert not classify_gait_episode(ceiling, thresholds, foot_names=("r", "l"))[0]


def test_optional_step_length_rail_and_foot_registry():
    thresholds = GaitGateThresholds.from_curriculum(curriculum(min_step_length_over_leg=0.05))
    assert classify_gait_episode(episode(), thresholds, foot_names=("r", "l"))[0]
    behind = episode(step_length_over_leg_min=-0.08)
    assert not classify_gait_episode(behind, thresholds, foot_names=("r", "l"))[0]
    assert classify_gait_episode(behind, GaitGateThresholds.from_curriculum(curriculum()), foot_names=("r", "l"))[0]
    assert not classify_gait_episode(episode(), thresholds, foot_names=("r", "r"))[0]


@pytest.mark.parametrize("kind", ["walk", "trot", "pace"])
def test_quadruped_profiles_are_mutually_exclusive(kind):
    feet = ("fr", "fl", "rr", "rl")
    verdicts = {
        profile: classify_gait_episode(
            episode(kind=kind), GaitGateThresholds.from_curriculum(curriculum(profile)), foot_names=feet
        )[0]
        for profile in ("quadruped_walk", "quadruped_trot", "quadruped_pace")
    }
    assert verdicts == {profile: profile == f"quadruped_{kind}" for profile in verdicts}


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


def _panel(root: Path, *, episodes=None):
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
    morphology, physics_dt, _ = _authored_morphology()
    morphology = copy.deepcopy(morphology)
    trace = stepping_trace(duration=10.0, dt=physics_dt)
    trace["floor_force_n"] *= morphology["body_weight_n"] / BW
    trace["touch_force_n"] *= morphology["body_weight_n"] / BW
    derived = episode_gait_metrics(
        trace,
        body_weight_n=morphology["body_weight_n"],
        leg_length_m=morphology["leg_length_m"],
        foot_names=("r", "l"),
        protocol=GaitProtocol(),
        settle_s=1.0,
    )
    metadata = episodes if episodes is not None else [episode(PUBLICATION_SEED_START + i) for i in range(40)]
    rows = []
    traces = []
    for i, recorded in enumerate(metadata):
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
                "reset_state_sha256": canonical_json_sha256({"synthetic_reset": i}),
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
        "gait_profile": "biped_alternating",
        "foot_names": ["r", "l"],
        "morphology": morphology,
        "seed_provenance": checkpoint_seed_provenance(checkpoint, seed_start=PUBLICATION_SEED_START, episodes=40),
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


def test_report_generation_uses_exact_selected_pair_and_fixed_gate_seed(tmp_path, monkeypatch):
    from environments.shared.reporting.stage_artifacts import _write_gait_report

    _panel(tmp_path)
    observed = []
    module = ModuleType("environments.shared.gait.report")

    def fake_writer(*args, **kwargs):
        observed.append((args, kwargs))
        return {"status": "complete"}

    module.write_gait_report = fake_writer
    monkeypatch.setitem(sys.modules, module.__name__, module)
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


@pytest.mark.parametrize("reason", ["skipped", "no_handoff"])
def test_failed_fresh_generation_invalidates_older_certificate(tmp_path, reason):
    from environments.shared.reporting.stage_artifacts import _write_gait_report

    _panel(tmp_path)
    config = {"curriculum_kwargs": curriculum()}
    if reason == "skipped":
        config["curriculum_kwargs"]["gait_report_episodes"] = 0
    else:
        (tmp_path / "models" / "robust_best_model_vecnorm.pkl").unlink()
    assert (
        _write_gait_report(
            species_cfg=SimpleNamespace(species="trex"),
            stage=2,
            stage_config=config,
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
            algorithm="PPO",
        )
        is None
    )
    assert not (tmp_path / "gait_report.json").exists()


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
