"""Verify suite ordering, failure isolation and resumability without training."""

import json

import pytest

from environments.compsognathus.experiments import balance_suite
from environments.compsognathus.scripts import train_balance_study
from environments.shared.plant_contract import PlantCompatibilityError


@pytest.fixture
def suite(tmp_path, monkeypatch):
    plan = {
        "arms": {arm: {} for arm in "ABCD"},
        "training_seeds": [42, 43, 44],
        "n_envs": 4,
        "stage_config": {"ppo_kwargs": {"n_steps": 1024}},
        "training_budget": 11_000_000,
        "implementation_sha256": "sha256:original",
    }
    calls = []
    trained = []
    failures = {}

    def run(output, arm, seed, **kwargs):
        tag = f"{arm}_seed{seed}"
        if kwargs.get("probe_updates"):
            tag += f"_probe{kwargs['probe_updates']}"
        calls.append((tag, kwargs))
        assert kwargs["resume"] is True
        progress = json.loads((output / "suite_progress.json").read_text())
        assert next(job for job in progress["jobs"] if job["name"] == tag)["status"] == "running"
        if tag in failures:
            raise failures[tag]
        path = output / "runs" / tag / "run_summary.json"
        if path.exists():
            return json.loads(path.read_text())
        trained.append(tag)
        path.parent.mkdir(parents=True)
        result = {"arm": arm, "seed": seed, "training_seconds": 1.5, "diagnostic_probe": "probe" in tag}
        path.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(balance_suite, "_load_plan", lambda output: plan)
    monkeypatch.setattr(balance_suite, "train_balance_arm", run)
    monkeypatch.setattr(
        balance_suite, "summarize_study", lambda output: {"expected_runs": 12, "source": "validated_trainer"}
    )
    return tmp_path, plan, calls, trained, failures


def test_full_suite_runs_all_smokes_before_twelve_matched_runs(suite):
    output, _, calls, trained, _ = suite
    result = balance_suite.run_balance_suite(output, mode="full")
    expected = [f"{arm}_seed42_probe2" for arm in "ABCD"] + [
        f"{arm}_seed{seed}" for arm in "ABCD" for seed in (42, 43, 44)
    ]
    assert trained == expected
    assert [tag for tag, _ in calls] == expected
    assert all(kwargs == {"resume": True, "probe_updates": 2, "evaluation_episodes": 4} for _, kwargs in calls[:4])
    assert all(kwargs == {"resume": True} for _, kwargs in calls[4:])
    assert result["status"] == "complete" and result["all_jobs_complete"]
    assert result["completed_jobs"] == 16
    assert result["planned_probe_training_steps"] == 32_768
    assert result["planned_full_training_steps"] == 132_022_272
    assert result["planned_training_steps"] == result["completed_training_steps"] == 132_055_040
    assert result["recorded_training_seconds"] == 24.0
    assert result["comparison"]["source"] == "validated_trainer"
    assert json.loads((output / "suite_comparison.json").read_text()) == result
    assert json.loads((output / "suite_progress.json").read_text()) == result


def test_smoke_mode_runs_only_four_short_probes(suite):
    output, _, calls, _, _ = suite
    result = balance_suite.run_balance_suite(output, mode="smoke")
    assert len(calls) == 4
    assert result["status"] == "complete"
    assert result["planned_training_steps"] == 32_768
    assert result["planned_full_training_steps"] == 0
    assert all(job["phase"] == "smoke" for job in result["jobs"])


def test_repeated_suite_delegates_completed_bundle_validation_without_retraining(suite):
    output, _, calls, trained, _ = suite
    balance_suite.run_balance_suite(output, mode="full")
    result = balance_suite.run_balance_suite(output, mode="full")
    assert len(calls) == 32  # Every old result is validated again by the trainer.
    assert len(trained) == 16
    assert result["invocations"] == 2
    assert result["all_jobs_complete"]
    assert all(job["status"] == "skipped" for job in result["jobs"])
    assert all(job["skip_reason"] == "validated_completed_run" for job in result["jobs"])
    assert all(job["completion_validated"] for job in result["jobs"])


def test_smoke_can_be_extended_to_full_without_retraining_validated_probes(suite):
    output, _, calls, trained, _ = suite
    balance_suite.run_balance_suite(output, mode="smoke")
    result = balance_suite.run_balance_suite(output, mode="full")
    assert len(calls) == 20 and len(trained) == 16
    assert all(job["status"] == "skipped" for job in result["jobs"][:4])
    assert all(job["status"] == "complete" for job in result["jobs"][4:])


def test_probe_failure_blocks_full_sweep_but_finishes_other_probes(suite):
    output, _, calls, _, failures = suite
    failures["B_seed42_probe2"] = RuntimeError("nonfinite training result")
    result = balance_suite.run_balance_suite(output, mode="full")
    assert len(calls) == 4
    assert result["status"] == "complete_with_errors"
    assert not result["all_jobs_complete"]
    assert result["failed_jobs"] == 1 and result["prerequisite_blocked_jobs"] == 12
    assert result["jobs"][1]["error"]["message"] == "nonfinite training result"
    assert all(job["skip_reason"] == "prerequisite_failed" for job in result["jobs"][4:])


def test_full_run_failure_does_not_block_other_full_runs(suite):
    output, _, calls, _, failures = suite
    failures["B_seed43"] = RuntimeError("training failed")
    result = balance_suite.run_balance_suite(output, mode="full")
    assert len(calls) == 16
    assert result["status"] == "complete_with_errors"
    assert result["failed_jobs"] == 1 and result["prerequisite_blocked_jobs"] == 0
    assert result["completed_jobs"] == 15
    assert result["jobs"][-1]["status"] == "complete"


def test_interrupt_persists_current_job_then_rerun_validates_finished_work(suite):
    output, _, calls, trained, failures = suite
    failures["B_seed42_probe2"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        balance_suite.run_balance_suite(output, mode="smoke")
    saved = json.loads((output / "suite_progress.json").read_text())
    assert saved["status"] == "interrupted"
    assert saved["jobs"][0]["status"] == "complete"
    assert saved["jobs"][1]["status"] == "interrupted"
    assert saved["jobs"][2]["status"] == "pending"
    del failures["B_seed42_probe2"]
    result = balance_suite.run_balance_suite(output, mode="smoke")
    assert result["all_jobs_complete"]
    assert len(calls) == 6 and len(trained) == 4
    assert result["jobs"][0]["skip_reason"] == "validated_completed_run"


def test_existing_summary_still_fails_when_trainer_rejects_its_identity(suite):
    output, _, _, _, failures = suite
    balance_suite.run_balance_suite(output, mode="smoke")
    failures["A_seed42_probe2"] = ValueError("checkpoint source identity mismatch")
    result = balance_suite.run_balance_suite(output, mode="full")
    assert result["jobs"][0]["status"] == "failed"
    assert result["prerequisite_blocked_jobs"] == 12


def test_changed_source_plan_is_rejected_before_running_any_job(suite):
    output, plan, calls, _, _ = suite
    balance_suite.run_balance_suite(output, mode="smoke")
    plan["implementation_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="different study"):
        balance_suite.run_balance_suite(output, mode="smoke")
    assert len(calls) == 4


def test_summary_failure_is_recorded_without_discarding_finished_job_evidence(suite, monkeypatch):
    output, _, _, _, _ = suite

    def fail_summary(output):
        raise ValueError("inconsistent confirmation report")

    monkeypatch.setattr(balance_suite, "summarize_study", fail_summary)
    result = balance_suite.run_balance_suite(output, mode="smoke")
    assert result["status"] == "complete_with_errors"
    assert result["completed_jobs"] == 4
    assert result["comparison_error"]["type"] == "ValueError"


@pytest.mark.parametrize("mode", ["smoke-only", "unknown", None])
def test_invalid_mode_is_rejected(suite, mode):
    with pytest.raises(ValueError, match="mode"):
        balance_suite.run_balance_suite(suite[0], mode=mode)


def test_full_suite_requires_the_prepared_three_seed_matrix(suite):
    output, plan, calls, _, _ = suite
    plan["training_seeds"] = [42]
    with pytest.raises(ValueError, match="training seeds"):
        balance_suite.run_balance_suite(output)
    assert not calls


def test_comparison_rejects_corrupted_completed_pair_despite_plausible_qualified_summary(tmp_path, monkeypatch):
    """A good-looking summary cannot bypass the trainer's strict pair check."""
    plan = {
        "arms": {"A": {}},
        "training_seeds": [42],
        "n_envs": 4,
        "stage_config": {"ppo_kwargs": {"n_steps": 1024}, "env_kwargs": {}},
        "training_budget": 11_000_000,
    }
    directory = tmp_path / "runs" / "A_seed42"
    directory.mkdir(parents=True)
    snapshot = {"plan": plan, "arm": "A", "seed": 42, "probe_updates": None, "evaluation_episodes": 40}
    (directory / "run_config.json").write_text(json.dumps(snapshot))
    (directory / "study_identity.json").write_text("{}")
    # Config, identity, budget and displayed behavior all appear valid. The
    # actual committed model/normalizer pair is rejected by the strict loader.
    summary = {
        "schema": train_balance_study.SCHEMA,
        "arm": "A",
        "seed": 42,
        "diagnostic_probe": False,
        "training_steps": 11_001_856,
        "learned_balance_qualified": True,
        "selected_checkpoint": "checkpoints/step_11001856",
        "confirmation": {
            "projected_stance_panel": {"full_horizon_fraction": 1.0, "mean_reward": 2990.0},
            "behavior_means": {"physics_bilateral_20pct_weight": 1.0},
        },
    }
    (directory / "run_summary.json").write_text(json.dumps(summary))

    class FakeEnvironment:
        def close(self):
            pass

    monkeypatch.setattr(train_balance_study, "_load_plan", lambda output: plan)
    monkeypatch.setattr(train_balance_study, "make_balance_env", lambda *args, **kwargs: FakeEnvironment())
    monkeypatch.setattr(train_balance_study, "build_study_identity", lambda env, config: {})
    monkeypatch.setattr(train_balance_study, "validate_study_identity", lambda recorded, expected: None)
    validations = []

    def reject_corrupted_pair(output, arm, seed, *, resume):
        assert (output, arm, seed, resume) == (tmp_path, "A", 42, True)
        assert (directory / "run_summary.json").is_file()  # This validation must never start a missing run.
        validations.append((arm, seed))
        raise PlantCompatibilityError("balance-study model checkpoint hash mismatch")

    monkeypatch.setattr(train_balance_study, "train_balance_arm", reject_corrupted_pair)
    with pytest.raises(PlantCompatibilityError, match="checkpoint hash mismatch"):
        train_balance_study.summarize_study(tmp_path)
    assert validations == [("A", 42)]
    assert not (tmp_path / "comparison.json").exists()
    assert not (tmp_path / "comparison.csv").exists()
