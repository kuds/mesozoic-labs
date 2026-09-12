"""Species calibration must stay bound to its measured task and frozen judge."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from environments.shared import recovery_calibration as calibration
from environments.shared.curriculum.gate_resolver import (
    GateResolutionError,
    build_gate_resolution,
    evaluate_recovery_gate_from_resolution,
    require_gate_resolution,
    write_gate_resolution,
)
from environments.shared.harnesses import freeze_recovery_gate as producer
from environments.shared.recovery_evaluation import EpisodeRecord, RecoveryPanelEvidence


@pytest.fixture
def profile_task(tmp_path, monkeypatch):
    env_kwargs = {
        "frame_skip": 10,
        "max_episode_steps": 1000,
        "perturbation_capture_velocity_multiple": 1.0,
    }
    spec = {
        "gate_kind": "recovery_quality/v1",
        "min_recovery_success_lcb": 0.5,
        "min_paired_success_delta_lcb": 0.1,
        "recovery_t_recover_steps": 40,
        "recovery_dwell_steps": 20,
        "min_eval_episodes": 40,
    }
    identity = {"species": "compsognathus", "physics_sha256": "sha256:measured"}
    profile = {
        "schema": calibration.PROFILE_SCHEMA,
        "species": "compsognathus",
        "calibration_id": "test-fixed-command-reference-v1",
        "control_dt_s": 0.02,
        "height_reference_m": 0.24,
        "safe_set": {
            "height_error_max_m": 0.01,
            "tilt_max_rad": 0.08,
            "planar_speed_max_mps": 0.1,
            "min_foot_force_n": 0,
        },
        "recovery_env_kwargs": env_kwargs,
        "capability_spec": spec,
        "plant_identity": identity,
        "task_sha256": "sha256:measured-task",
        "required_paired_nulls": ["zero_action", "brace"],
    }
    path = tmp_path / "compsognathus" / "recovery_calibration.json"
    path.parent.mkdir()
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(calibration, "CONFIGS_ROOT", tmp_path)
    monkeypatch.setattr(calibration, "current_plant_identity", lambda _: SimpleNamespace(to_dict=lambda: identity))
    monkeypatch.setattr(
        calibration, "derive_stage_task_fingerprint", lambda **_: {"task_sha256": "sha256:measured-task"}
    )
    config = {"env_kwargs": deepcopy(env_kwargs), "curriculum_kwargs": deepcopy(spec)}
    monkeypatch.setattr(
        calibration,
        "load_stage_config",
        lambda *_: deepcopy(config),
    )
    env = SimpleNamespace(**env_kwargs, dt=0.02, healthy_z_range=(0.14, 0.35), close=lambda: None)
    monkeypatch.setattr(calibration, "build_env", lambda *_: env)
    return profile, path, env


def test_profile_loads_physical_judge_not_trex_defaults(profile_task):
    profile, _, env = profile_task
    loaded = calibration.load_recovery_calibration("Compsognathus Longipes", env=env)
    assert loaded.safe_set == profile["safe_set"]
    assert loaded.height_reference_m == 0.24
    assert loaded.thresholds.t_recover_steps * env.dt == pytest.approx(0.8)
    assert loaded.evaluation_spec()["control_dt_s"] == env.dt
    assert loaded.evaluation_spec()["profile_sha256"].startswith("sha256:")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("species", "compsognathus_robot", "schema or species"),
        ("control_dt_s", 0.01, "control timestep"),
        ("height_reference_m", 0.4, "healthy range"),
        ("height_reference_m", False, "finite number"),
        ("safe_set", {"height_error_max_m": 0.01}, "all four"),
        ("plant_identity", {}, "plant identity"),
        ("task_sha256", "sha256:old-implementation", "task fingerprint changed"),
        ("required_paired_nulls", ["zero_action", "zero_action"], "unique zero_action and brace"),
        ("required_paired_nulls", ["zero_action"], "unique zero_action and brace"),
        ("recovery_env_kwargs", {}, "environment changed"),
    ],
)
def test_changed_calibration_is_refused(profile_task, field, value, match):
    profile, path, env = profile_task
    profile[field] = value
    path.write_text(json.dumps(profile))
    with pytest.raises(GateResolutionError, match=match):
        calibration.load_recovery_calibration("compsognathus", env=env)


def test_nonfinite_measurement_is_refused(profile_task):
    profile, path, env = profile_task
    profile["safe_set"]["tilt_max_rad"] = float("nan")
    path.write_text(json.dumps(profile))
    with pytest.raises(GateResolutionError, match="cannot load recovery calibration"):
        calibration.load_recovery_calibration("compsognathus", env=env)


def test_actual_environment_override_requires_recalibration(profile_task):
    _, _, env = profile_task
    env.perturbation_capture_velocity_multiple = 2.0
    with pytest.raises(GateResolutionError, match="actual environment perturbation"):
        calibration.load_recovery_calibration("compsognathus", env=env)


def test_capability_spec_must_agree_with_curriculum(profile_task):
    profile, path, env = profile_task
    profile["capability_spec"]["min_recovery_success_lcb"] = 0.1
    path.write_text(json.dumps(profile))
    with pytest.raises(GateResolutionError, match="disagrees with the stage curriculum"):
        calibration.load_recovery_calibration("compsognathus", env=env)


def test_missing_species_profile_never_falls_back_to_trex(profile_task):
    with pytest.raises(GateResolutionError, match="cannot load recovery calibration"):
        producer._species_calibration("compsognathus_robot", "recovery")


def _evidence(safe_set, *, full_horizon=False, n_pushes=0):
    episode = EpisodeRecord("zero_action", 1, 3042, 1000 if full_horizon else 10, full_horizon, n_pushes, 0, False, 0.0)
    return RecoveryPanelEvidence("zero_action", (episode,), (), safe_set)


def test_complete_judge_is_hashed_and_legacy_records_remain_readable(profile_task, tmp_path):
    loaded = calibration.load_recovery_calibration("compsognathus")
    kwargs = {
        "task_fingerprint": {"task_sha256": "sha256:task"},
        "thresholds": loaded.thresholds,
        "null_evidence": {"zero_action": _evidence(loaded.safe_set)},
        "panel_seed_start": 3042,
    }
    legacy = build_gate_resolution(**kwargs)
    write_gate_resolution(tmp_path, legacy)
    assert "evaluation_spec" not in require_gate_resolution(tmp_path, current_task_sha256="sha256:task")
    frozen = build_gate_resolution(
        **kwargs,
        evaluation_spec=loaded.evaluation_spec(),
        null_provenance={"brace": {"stance_env_kwargs": {"prey_distance_range": (0.8, 1.2)}}},
    )
    assert frozen["resolution_sha256"] != legacy["resolution_sha256"]
    assert frozen["null_provenance"]["brace"]["stance_env_kwargs"]["prey_distance_range"] == [0.8, 1.2]
    write_gate_resolution(tmp_path, frozen)
    assert require_gate_resolution(tmp_path, current_task_sha256="sha256:task") == frozen
    frozen["evaluation_spec"]["height_reference_m"] += 0.01
    write_gate_resolution(tmp_path, frozen)
    with pytest.raises(GateResolutionError, match="integrity hash"):
        require_gate_resolution(tmp_path, current_task_sha256="sha256:task")


@pytest.mark.parametrize("change", ["missing", "height", "timebase", "provenance"])
def test_policy_panel_rejects_changed_or_missing_frozen_judge(profile_task, tmp_path, monkeypatch, change):
    loaded = calibration.load_recovery_calibration("compsognathus")
    judge = loaded.evaluation_spec()
    if change == "height":
        judge["height_reference_m"] += 0.005
    elif change == "timebase":
        judge["control_dt_s"] *= 2
    elif change == "provenance":
        judge["profile_sha256"] = "sha256:different-source"
    frozen = build_gate_resolution(
        task_fingerprint={"task_sha256": "sha256:task"},
        thresholds=loaded.thresholds,
        null_evidence={"zero_action": _evidence(loaded.safe_set)},
        panel_seed_start=3042,
        evaluation_spec=None if change == "missing" else judge,
    )
    write_gate_resolution(tmp_path, frozen)
    monkeypatch.setattr(producer, "stage_task_fingerprint", lambda *_: {"task_sha256": "sha256:task"})
    with pytest.raises(GateResolutionError, match="evaluation_spec"):
        producer.roll_policy_panel(tmp_path, "unopened.zip", "unopened.pkl", species="compsognathus")


def test_full_horizon_without_a_judged_push_cannot_certify():
    producer._require_judged_pushes(_evidence({}))  # An early failure is real evidence.
    producer._require_judged_pushes(_evidence({}, full_horizon=True, n_pushes=1))
    with pytest.raises(GateResolutionError, match="without a judged push"):
        producer._require_judged_pushes(_evidence({}, full_horizon=True))


def test_sac_never_uses_the_ppo_numpy_forward_pass():
    with pytest.raises(ValueError, match="PPO only"):
        producer.policy_controller(
            "unopened.zip", "unopened.pkl", action_space=None, algorithm="sac", inference="numpy"
        )


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_checkpoint_inference_matches_selected_algorithm(algorithm, tmp_path, monkeypatch):
    sb3 = pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    import numpy as np

    env = gym.make("Pendulum-v1")
    try:
        model = getattr(sb3, algorithm.upper())("MlpPolicy", env, policy_kwargs={"net_arch": [16, 16]}, seed=42)
        path = tmp_path / "policy.zip"
        model.save(path)
        stats = {
            "mean": np.array([0.1, -0.2, 0.3]),
            "var": np.array([0.2, 0.3, 0.4]),
            "clip_obs": 10,
            "epsilon": 1e-8,
        }
        monkeypatch.setattr(producer, "load_vecnormalize_obs_stats", lambda _: stats)
        controller = producer.policy_controller(path, "stats.pkl", action_space=env.action_space, algorithm=algorithm)
        obs, _ = env.reset(seed=3042)
        expected, _ = model.predict(producer.normalize_observation(obs, stats).astype(np.float32), deterministic=True)
        np.testing.assert_allclose(controller(obs), expected, atol=1e-7, rtol=0)
    finally:
        env.close()


@pytest.mark.parametrize("fail_step,occupancy", [(None, True), (60, True), (None, False)])
def test_brace_uses_complete_quiet_stance_and_closes_env(monkeypatch, fail_step, occupancy):
    import numpy as np

    class QuietEnv:
        dt = 0.02
        perturbation_capture_velocity_multiple = 0.0
        action_space = SimpleNamespace(shape=(2,))
        closed = False

        def reset(self, *, seed):
            self.step_index = 0
            return np.array([seed]), {}

        def step(self, action):
            self.step_index += 1
            return action, 0, self.step_index == fail_step, False, {}

        def close(self):
            self.closed = True

    env = QuietEnv()

    def build(species, stage):
        assert species == "compsognathus"
        assert stage == "stance"
        return env

    monkeypatch.setattr(producer, "build_env", build)
    monkeypatch.setattr(producer, "load_stage_config", lambda *_: {"env_kwargs": {"frame_skip": 10}})
    monkeypatch.setattr(producer, "_safe_step", lambda *_: occupancy)
    judge = SimpleNamespace(safe_set={"height_error_max_m": 0.01}, height_reference_m=0.24)
    if fail_step:
        with pytest.raises(GateResolutionError, match="complete quiet brace reference"):
            producer._quiet_brace_controller("compsognathus", lambda _: np.array([0.1, -0.2]), calibration=judge)
    elif not occupancy:
        with pytest.raises(GateResolutionError, match="calibrated recovery safe set"):
            producer._quiet_brace_controller("compsognathus", lambda _: np.array([0.1, -0.2]), calibration=judge)
    else:
        controller, metadata = producer._quiet_brace_controller(
            "compsognathus", lambda _: np.array([0.1, -0.2]), calibration=judge
        )
        np.testing.assert_allclose(controller(None), [0.1, -0.2])
        assert metadata["settle_steps"] == metadata["sample_steps"] == 50
        assert metadata["held_action"] == pytest.approx([0.1, -0.2])
        assert metadata["quiet_stance_min_safe_fraction"] == 0.95
        assert set(metadata["quiet_safe_fractions_by_seed"].values()) == {1.0}
        assert not set(metadata["seeds"]) & set(range(3042, 3082))
    assert env.closed


@pytest.mark.parametrize("mutation", [None, "task", "plant", "algorithm"])
def test_checkpoint_source_is_bound_to_plant_task_and_algorithm(tmp_path, monkeypatch, mutation):
    import zipfile

    from environments.shared.plant_contract import (
        MODEL_IDENTITY_ATTRIBUTE,
        PlantCompatibilityError,
        current_plant_identity,
    )
    from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE, TaskFingerprintError

    identity = current_plant_identity("compsognathus")
    task = {"task_sha256": "sha256:stance"}
    metadata = {
        "clip_range": 0.2,
        "n_epochs": 5,
        MODEL_IDENTITY_ATTRIBUTE: identity.to_dict(),
        MODEL_TASK_ATTRIBUTE: deepcopy(task),
    }
    if mutation == "task":
        metadata[MODEL_TASK_ATTRIBUTE]["task_sha256"] = "sha256:other-stage"
    elif mutation == "plant":
        metadata[MODEL_IDENTITY_ATTRIBUTE]["species"] = "compsognathus_robot"
    elif mutation == "algorithm":
        del metadata["clip_range"]
        del metadata["n_epochs"]
        metadata.update(target_entropy="auto", replay_buffer_class="ReplayBuffer")
    path = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(metadata))
    monkeypatch.setattr(producer, "stage_task_fingerprint", lambda *_: task)
    if mutation:
        with pytest.raises((GateResolutionError, PlantCompatibilityError, TaskFingerprintError)):
            producer._validate_checkpoint_source(path, "compsognathus", "stance", "ppo")
    else:
        assert producer._validate_checkpoint_source(path, "compsognathus", "stance", "ppo") == identity


def _outcome_panel(controller_id, outcomes, safe_set):
    episodes = tuple(
        EpisodeRecord(controller_id, index + 1, 3042 + index, 1000, True, 1, int(outcome), outcome, 0.0)
        for index, outcome in enumerate(outcomes)
    )
    return RecoveryPanelEvidence(controller_id, episodes, (), safe_set)


def _comparison_resolution(loaded, *, brace_success=False, include_brace=True, legacy=False, species=None):
    nulls = {"zero_action": _outcome_panel("zero_action", [False] * 40, loaded.safe_set)}
    if include_brace:
        nulls["brace"] = _outcome_panel("brace", [brace_success] * 40, loaded.safe_set)
    return build_gate_resolution(
        task_fingerprint={
            "task_sha256": "sha256:measured-task",
            "species": species or ("trex" if legacy else "compsognathus"),
        },
        thresholds=loaded.thresholds,
        null_evidence=nulls,
        panel_seed_start=3042,
        evaluation_spec=None if legacy else loaded.evaluation_spec(),
    )


@pytest.mark.parametrize("brace_success", [False, True])
def test_recovery_must_beat_both_statue_and_brace(profile_task, tmp_path, brace_success):
    loaded = calibration.load_recovery_calibration("compsognathus")
    resolution = _comparison_resolution(loaded, brace_success=brace_success)
    write_gate_resolution(tmp_path, resolution)
    result = evaluate_recovery_gate_from_resolution(
        tmp_path,
        current_task_sha256="sha256:measured-task",
        policy_successes_by_seed=dict.fromkeys(range(3042, 3082), True),
        null_controller_id="caller-cannot-bypass-the-required-family",
    )
    assert result.passed is not brace_success
    assert result.paired_delta_lcb == pytest.approx(0.0 if brace_success else 1.0)
    if brace_success:
        assert any(failure.startswith("brace: paired_success_delta_lcb") for failure in result.failures)


def test_missing_brace_blocks_direct_gate_and_pretraining_check(profile_task, tmp_path, monkeypatch):
    loaded = calibration.load_recovery_calibration("compsognathus")
    write_gate_resolution(tmp_path, _comparison_resolution(loaded, include_brace=False))
    with pytest.raises(GateResolutionError, match="brace"):
        evaluate_recovery_gate_from_resolution(
            tmp_path,
            current_task_sha256="sha256:measured-task",
            policy_successes_by_seed=dict.fromkeys(range(3042, 3082), True),
        )
    monkeypatch.setattr(producer, "stage_task_fingerprint", lambda *_: {"task_sha256": "sha256:measured-task"})
    with pytest.raises(GateResolutionError, match="brace"):
        producer.validate_recovery_resolution(tmp_path, species="compsognathus")


def test_legacy_gate_keeps_original_single_null_behavior(profile_task, tmp_path):
    loaded = calibration.load_recovery_calibration("compsognathus")
    write_gate_resolution(tmp_path, _comparison_resolution(loaded, brace_success=True, legacy=True))
    result = evaluate_recovery_gate_from_resolution(
        tmp_path,
        current_task_sha256="sha256:measured-task",
        policy_successes_by_seed=dict.fromkeys(range(3042, 3082), True),
    )
    assert result.passed
    assert result.paired_delta_lcb == 1.0


def test_compsognathus_missing_judge_refuses_cached_direct_and_reporting_verdict(profile_task, tmp_path):
    from environments.shared.reporting import evaluate_stage_gate

    loaded = calibration.load_recovery_calibration("compsognathus")
    write_gate_resolution(tmp_path, _comparison_resolution(loaded, legacy=True, species="compsognathus"))
    (tmp_path / "task_fingerprint.json").write_text(
        json.dumps({"task_sha256": "sha256:measured-task", "species": "compsognathus"})
    )
    counts = dict.fromkeys(range(3042, 3082), True)
    with pytest.raises(GateResolutionError, match="missing recovery evaluation_spec"):
        evaluate_recovery_gate_from_resolution(
            tmp_path, current_task_sha256="sha256:measured-task", policy_successes_by_seed=counts
        )
    passed, failures = evaluate_stage_gate(
        {"gate_schema_version": 1, **loaded.profile["capability_spec"]},
        {},
        stage="recovery",
        stage_dir=tmp_path,
        recovery_successes_by_seed=counts,
    )
    assert not passed
    assert any("missing recovery evaluation_spec" in failure for failure in failures)


@pytest.mark.parametrize("identity_source", ["sidecar", "explicit", "unknown"])
def test_historical_record_requires_trex_identity(profile_task, tmp_path, identity_source):
    loaded = calibration.load_recovery_calibration("compsognathus")
    legacy = build_gate_resolution(
        task_fingerprint={"task_sha256": "sha256:old-trex-task"},
        thresholds=loaded.thresholds,
        null_evidence={"zero_action": _outcome_panel("zero_action", [False] * 40, loaded.safe_set)},
        panel_seed_start=3042,
    )
    write_gate_resolution(tmp_path, legacy)
    if identity_source == "sidecar":
        (tmp_path / "task_fingerprint.json").write_text(
            json.dumps({"task_sha256": "sha256:old-trex-task", "species": "trex"})
        )
    kwargs = {
        "current_task_sha256": "sha256:old-trex-task",
        "policy_successes_by_seed": dict.fromkeys(range(3042, 3082), True),
        "expected_species": "trex" if identity_source == "explicit" else None,
    }
    if identity_source == "unknown":
        with pytest.raises(GateResolutionError, match="only an identified historical T-Rex"):
            evaluate_recovery_gate_from_resolution(tmp_path, **kwargs)
    else:
        assert evaluate_recovery_gate_from_resolution(tmp_path, **kwargs).passed


def test_cached_counts_cannot_recertify_after_profile_height_changes(profile_task, tmp_path):
    loaded = calibration.load_recovery_calibration("compsognathus")
    write_gate_resolution(tmp_path, _comparison_resolution(loaded))
    profile, path, _ = profile_task
    profile["height_reference_m"] += 0.005
    path.write_text(json.dumps(profile))
    with pytest.raises(GateResolutionError, match="evaluation_spec differs"):
        evaluate_recovery_gate_from_resolution(
            tmp_path,
            current_task_sha256="sha256:measured-task",
            policy_successes_by_seed=dict.fromkeys(range(3042, 3082), True),
        )


@pytest.mark.parametrize("change", [None, "checkpoint", "vecnormalize", "algorithm"])
def test_pretraining_check_binds_brace_to_current_stance_sources(profile_task, tmp_path, monkeypatch, change):
    loaded = calibration.load_recovery_calibration("compsognathus")
    checkpoint, stats = tmp_path / "stance.zip", tmp_path / "stance.pkl"
    checkpoint.write_bytes(b"stance checkpoint")
    stats.write_bytes(b"matched statistics")
    resolution = _comparison_resolution(loaded)
    resolution["null_provenance"] = {
        "brace": {
            "checkpoint_sha256": producer._file_sha256(checkpoint),
            "vecnormalize_sha256": producer._file_sha256(stats),
            "algorithm": "ppo",
        }
    }
    monkeypatch.setattr(producer, "_validated_recovery_resolution", lambda *_: (resolution, loaded))
    if change == "checkpoint":
        checkpoint.write_bytes(b"a different stance checkpoint")
    elif change == "vecnormalize":
        stats.write_bytes(b"a different statistics file")
    kwargs = {
        "species": "compsognathus",
        "policy_zip": checkpoint,
        "vecnorm": stats,
        "algorithm": "sac" if change == "algorithm" else "ppo",
    }
    if change:
        with pytest.raises(GateResolutionError, match="frozen brace source differs"):
            producer.validate_recovery_resolution(tmp_path, **kwargs)
    else:
        assert producer.validate_recovery_resolution(tmp_path, **kwargs) is resolution
