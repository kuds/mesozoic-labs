"""Tests for environments.shared.result_bundle.evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from environments.shared.reporting import save_evaluation_episodes, save_result_bundle
from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    compare_summary_to_csv,
    validate_result_bundle,
    write_artifact_manifest,
)

from .result_bundle_helpers import (
    _complete_bundle,
    _complete_bundle_inputs,
    _plant_identity,
    _rewrite_csv_cell,
)


def test_save_evaluation_episodes_preserves_per_episode_evidence(tmp_path: Path) -> None:
    path = save_evaluation_episodes(
        tmp_path / "stage1",
        rewards=[10.5, 20.5],
        lengths=[100, 200],
        forward_velocities=[0.1, 0.2],
        distances=[1.0, 2.0],
        successes=[False, True],
        evaluation_seed=123,
        checkpoint_label="selected",
    )

    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows == [
        {
            "episode": "1",
            "evaluation_seed": "123",
            "checkpoint": "selected",
            "reward": "10.5",
            "length": "100",
            "mean_forward_velocity": "0.1",
            "distance_traveled": "1.0",
            "success": "False",
        },
        {
            "episode": "2",
            "evaluation_seed": "123",
            "checkpoint": "selected",
            "reward": "20.5",
            "length": "200",
            "mean_forward_velocity": "0.2",
            "distance_traveled": "2.0",
            "success": "True",
        },
    ]


def test_save_evaluation_episodes_rejects_incomplete_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        save_evaluation_episodes(
            tmp_path,
            rewards=[1.0, 2.0],
            lengths=[100],
            forward_velocities=[0.1, 0.2],
            distances=[1.0, 2.0],
            successes=[True, False],
            evaluation_seed=1,
            checkpoint_label="selected",
        )


def test_save_evaluation_episodes_rejects_boolean_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        save_evaluation_episodes(
            tmp_path,
            rewards=[1.0],
            lengths=[100],
            forward_velocities=[0.1],
            distances=[1.0],
            successes=[True],
            evaluation_seed=True,
            checkpoint_label="selected",
        )


@pytest.mark.parametrize("invalid_number", ["nan", "inf", "-inf"])
def test_nonfinite_csv_metric_is_classified_as_canonical_conflict(
    tmp_path: Path,
    stable_provenance: None,
    invalid_number: str,
) -> None:
    run_dir = tmp_path / invalid_number.replace("-", "negative-")
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    _rewrite_csv_cell(
        paths["collected_results_csv"],
        field="last_mean_reward",
        value=invalid_number,
    )
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert report["errors"]
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


def test_malformed_csv_seed_is_reported_instead_of_escaping_audit(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    _rewrite_csv_cell(
        paths["collected_results_csv"],
        field="seed",
        value="not-an-integer",
    )
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert report["errors"]
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


def test_summary_csv_contradiction_is_detected(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    csv_path = paths["collected_results_csv"]
    with csv_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        csv_fields = list(reader.fieldnames or [])
    rows[0]["last_mean_reward"] = "9999"
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    problems = compare_summary_to_csv(summary, csv_path)
    assert any("final_eval_reward/last_mean_reward differs" in problem for problem in problems)


def test_complete_bundle_requires_selected_evaluation_evidence(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    evidence_path = run_dir / "stage2" / "evaluation_selected.csv"
    evidence_path.unlink()
    manifest = json.loads(paths["artifact_manifest"].read_text(encoding="utf-8"))
    manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != "stage2/evaluation_selected.csv"]
    paths["artifact_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert audit_result_bundle(run_dir)["status"] == "canonical-conflict"


@pytest.mark.parametrize(("field", "value"), [("evaluation_seed", "999"), ("episode", "2")])
def test_selected_evaluation_evidence_identity_is_validated(
    tmp_path: Path,
    stable_provenance: None,
    field: str,
    value: str,
) -> None:
    run_dir = tmp_path / field
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    evidence_path = run_dir / "stage1" / "evaluation_selected.csv"
    _rewrite_csv_cell(evidence_path, field=field, value=value)
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert report["errors"]


def test_selected_evaluation_aggregate_must_match_summary_and_csv(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["stages"]["1"]["selected_model_reward"] = 999.0
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_csv_cell(
        paths["collected_results_csv"],
        field="selected_model_mean_reward",
        value="999.0",
    )
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert any("evaluation aggregate" in error for error in report["errors"])


def test_publication_gate_is_recomputed_from_frozen_thresholds(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    config_path = run_dir / "stage1" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["curriculum_kwargs"]["min_avg_reward"] = 100.0
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ResultBundleError, match=r"publication gate fails min_avg_reward"):
        save_result_bundle(
            stage_results,
            stage_configs,
            "velociraptor",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="2.7.0",
            parallel_envs=4,
            evaluation_episodes=3,
            evaluation_seeds=[101, 102, 103],
            plant_identity=_plant_identity(),
        )


def test_stance_gated_stage_refuses_publication_rather_than_certifying_on_the_rail(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """A gate this evidence file cannot express must not be half-checked.

    stance_quality/v1 carries min_avg_reward only as a RAIL, set below the
    zero-action statue. The per-episode evidence CSV records reward and
    length but no unsupported duty, so evaluating the legacy thresholds would
    certify the stage on the rail alone -- reintroducing exactly the "a statue
    clears this gate" failure the stance gate was built to remove.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    config_path = run_dir / "stage1" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["curriculum_kwargs"]["gate_kind"] = "stance_quality/v1"
    config["curriculum_kwargs"]["min_full_horizon_fraction"] = 0.95
    config["curriculum_kwargs"]["max_unsupported_duty"] = 0.02
    config["curriculum_kwargs"]["max_unsupported_duty_ucb"] = 0.02
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ResultBundleError, match=r"cannot be checked from the publication evidence"):
        save_result_bundle(
            stage_results,
            stage_configs,
            "velociraptor",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="2.7.0",
            parallel_envs=4,
            evaluation_episodes=3,
            evaluation_seeds=[101, 102, 103],
            plant_identity=_plant_identity(),
        )


def test_final_evaluation_claims_are_bound_to_terminal_episode_evidence(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["stages"]["1"]["final_eval_reward"] = 999.0
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_csv_cell(paths["collected_results_csv"], field="last_mean_reward", value="999.0")
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert any("final evaluation aggregate" in error for error in report["errors"])
