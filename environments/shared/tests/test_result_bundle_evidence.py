"""Tests for environments.shared.result_bundle.evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from environments.shared.reporting import save_evaluation_episodes, save_result_bundle
from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    compare_summary_to_csv,
    sha256_file,
    validate_result_bundle,
    write_artifact_manifest,
)
from environments.shared.scripts.stance_gate_report import STANCE_PANEL_FIELDNAMES

from .result_bundle_helpers import (
    _complete_bundle,
    _complete_bundle_inputs,
    _plant_identity,
    _rewrite_csv_cell,
    _rewrite_csv_column,
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
            "task_success": "False",
        },
        {
            "episode": "2",
            "evaluation_seed": "123",
            "checkpoint": "selected",
            "reward": "20.5",
            "length": "200",
            "mean_forward_velocity": "0.2",
            "distance_traveled": "2.0",
            "task_success": "True",
        },
    ]


def test_save_evaluation_episodes_records_the_evaluated_checkpoint_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_model.zip"
    checkpoint.write_bytes(b"policy weights")
    path = save_evaluation_episodes(
        tmp_path / "stage1",
        rewards=[1.0, 2.0],
        lengths=[10, 20],
        forward_velocities=[0.1, 0.2],
        distances=[1.0, 2.0],
        successes=[True, False],
        evaluation_seed=1,
        checkpoint_label="selected",
        checkpoint_path=checkpoint,
    )

    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert [row["checkpoint_sha256"] for row in rows] == [sha256_file(checkpoint)] * 2

    with pytest.raises(ValueError, match="does not exist"):
        save_evaluation_episodes(
            tmp_path / "stage1",
            rewards=[1.0],
            lengths=[10],
            forward_velocities=[0.1],
            distances=[1.0],
            successes=[True],
            evaluation_seed=1,
            checkpoint_label="selected",
            checkpoint_path=tmp_path / "missing.zip",
        )


def test_legacy_success_header_still_audits(tmp_path: Path, stable_provenance: None) -> None:
    """Bundles written before the 2026-08-15 column rename must keep auditing.

    The per-episode task-event column was renamed ``success`` ->
    ``task_success`` (the bare name was misread as the stage-gate verdict in
    stance-gated stages). The reader accepts either header; this pins the
    legacy side so the fallback cannot be dropped while pre-rename bundles
    exist.
    """
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    evidence_path = run_dir / "stage1" / "evaluation_selected.csv"
    header, rest = evidence_path.read_text(encoding="utf-8").split("\n", 1)
    assert "task_success" in header
    evidence_path.write_text(header.replace("task_success", "success") + "\n" + rest, encoding="utf-8")
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    assert not report["errors"]


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


_STANCE_HORIZON = 1000


def _make_stance_gated(run_dir: Path, stage: int = 1) -> None:
    """Declare stance_quality/v1 on a stage of an otherwise complete bundle."""
    config_path = run_dir / f"stage{stage}" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["curriculum_kwargs"] = {
        "gate_kind": "stance_quality/v1",
        "gate_schema_version": 1,
        "min_full_horizon_fraction": 0.95,
        "max_unsupported_duty": 0.02,
        "max_unsupported_duty_ucb": 0.02,
        "min_eval_episodes": 40,
        "settle_steps": 200,
    }
    config["env_kwargs"] = {**config.get("env_kwargs", {}), "max_episode_steps": _STANCE_HORIZON}
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_stance_panel(run_dir: Path, rows: list[dict[str, Any]], stage: int = 1) -> Path:
    path = run_dir / f"stage{stage}" / "stance_panel_selected.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(STANCE_PANEL_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _stance_panel_rows(
    *,
    episodes: int = 40,
    duty: float = 0.0,
    full_horizon: int | None = None,
    reward: float = 2400.0,
) -> list[dict[str, Any]]:
    reached = episodes if full_horizon is None else full_horizon
    rows = []
    for index in range(episodes):
        is_full = index < reached
        rows.append(
            {
                "episode": index,
                "panel_seed": 3042 + index,
                "length": _STANCE_HORIZON if is_full else 300,
                "reward": reward if is_full else 600.0,
                "reached_horizon": is_full,
                "unsupported_duty": duty,
                "bilateral_support_duty": 1.0 - duty,
                "single_support_duty": 0.0,
            }
        )
    return rows


def _save_bundle(run_dir: Path, stage_results: Any, stage_configs: Any) -> Any:
    return save_result_bundle(
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


def test_stance_gated_stage_refuses_publication_without_duty_evidence(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """A gate whose criteria nothing measured must not be half-checked.

    stance_quality/v1 carries min_avg_reward only as a RAIL, set below the
    zero-action statue. `evaluation_selected.csv` records reward and length
    but no unsupported duty, so evaluating the legacy thresholds would certify
    the stage on the rail alone -- reintroducing exactly the "a statue clears
    this gate" failure the stance gate was built to remove.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _make_stance_gated(run_dir)

    with pytest.raises(ResultBundleError, match=r"stance_panel_selected\.csv is missing"):
        _save_bundle(run_dir, stage_results, stage_configs)


def test_stance_gated_stage_publishes_when_the_panel_evidence_clears_the_gate(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """The milestone must actually be reachable.

    Before the panel evidence existed this refused unconditionally, so a run
    that genuinely cleared the stance gate trained all three stages and then
    failed at bundle finalisation -- observed on run 20260803_012355, whose
    collected_results.csv and artifact_manifest.json stop at stage 2 while
    stage 3's own artifacts were written in full.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _make_stance_gated(run_dir)
    _write_stance_panel(run_dir, _stance_panel_rows(duty=0.0))

    paths = _save_bundle(run_dir, stage_results, stage_configs)
    assert paths["summary"].is_file()
    assert (run_dir / "collected_results.csv").is_file()


def test_stance_gated_stage_fails_on_the_real_duty_breach(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Run 20260802_203215's actual panel: full horizon 1.0000, duty 0.2120.

    That run recorded publication_gate_passed = True beside a stance report
    reading GATE: FAIL at 10.6x the duty ceiling. Re-derived from per-episode
    evidence, the bundle must refuse it.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _make_stance_gated(run_dir)
    _write_stance_panel(run_dir, _stance_panel_rows(duty=0.2120, reward=2295.5))

    with pytest.raises(ResultBundleError, match=r"publication gate fails stance_quality/v1"):
        _save_bundle(run_dir, stage_results, stage_configs)


def test_stance_gated_stage_fails_on_the_real_survival_breach(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Run 20260803_012355's actual panel: duty 0.0000 but only 26 of 40 full.

    The two runs failed on opposite criteria, so both directions need pinning:
    a perfect duty score must not rescue a panel that mostly fell over.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _make_stance_gated(run_dir)
    _write_stance_panel(run_dir, _stance_panel_rows(duty=0.0, full_horizon=26, reward=2319.7))

    with pytest.raises(ResultBundleError, match=r"publication gate fails stance_quality/v1"):
        _save_bundle(run_dir, stage_results, stage_configs)


def test_an_unmeasured_duty_cell_is_not_read_as_a_perfect_score(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Empty must mean "unmeasured", never 0.0.

    Zero duty is the statue's score and the best attainable one, so coercing a
    blank cell to 0.0 would turn missing evidence into a perfect result — the
    fail-open shape this whole gate exists to refuse.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _make_stance_gated(run_dir)
    rows = _stance_panel_rows(duty=0.0)
    for row in rows:
        row["unsupported_duty"] = ""
    _write_stance_panel(run_dir, rows)

    with pytest.raises(ResultBundleError, match=r"publication gate fails stance_quality/v1"):
        _save_bundle(run_dir, stage_results, stage_configs)


def test_a_missing_horizon_refuses_rather_than_assuming_one(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Guessing the horizon is a fail-open with a very quiet failure mode.

    The horizon decides which episodes count as survivals. Defaulting it to
    1000 against a stage whose real horizon is 2000 certifies every episode
    that fell at step 1500 as having reached the horizon. Measured on this
    function while the default was in place: 40 such episodes passed the gate.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _make_stance_gated(run_dir)
    config_path = run_dir / "stage1" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["env_kwargs"].pop("max_episode_steps")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_stance_panel(run_dir, _stance_panel_rows(duty=0.0))

    with pytest.raises(ResultBundleError, match=r"records no max_episode_steps"):
        _save_bundle(run_dir, stage_results, stage_configs)


def test_a_reached_horizon_column_that_contradicts_its_length_is_rejected(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """The auditor scores from `length`; the column must not say otherwise.

    A disagreeing column would mislead a human reading the evidence file
    while the verdict came from a different column of the same row.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _make_stance_gated(run_dir)
    rows = _stance_panel_rows(duty=0.0)
    rows[3]["length"] = 300
    _write_stance_panel(run_dir, rows)

    with pytest.raises(ResultBundleError, match=r"claims reached_horizon=True but its length 300"):
        _save_bundle(run_dir, stage_results, stage_configs)


def test_the_recorded_verdict_is_re_derived_not_trusted(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """A stored `passed: true` must not be able to certify a failing panel.

    The point of an evidence file is that the published claim is reproducible
    from the episodes behind it. Run 20260802_203215 is the counterexample:
    a recorded pass beside measurements that refute it.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _make_stance_gated(run_dir)
    _write_stance_panel(run_dir, _stance_panel_rows(duty=0.2120))
    (run_dir / "stage1" / "stance_gate_report.json").write_text(
        json.dumps({"gate_kind": "stance_quality/v1", "passed": True, "failures": []}),
        encoding="utf-8",
    )

    with pytest.raises(ResultBundleError, match=r"publication gate fails stance_quality/v1"):
        _save_bundle(run_dir, stage_results, stage_configs)


def test_a_checkpoint_rewritten_after_evaluation_cannot_be_published(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """The certified hash must be the checkpoint that produced the certified numbers.

    The bundle hashes the selected checkpoint at save time, hours after the
    evidence was generated; a write landing in between used to yield a
    canonical-valid bundle whose hash-certified checkpoint never produced its
    metrics (arbitrary bytes in best_model passed audit).
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    (run_dir / "stage3" / "models" / "best_model.pkl").write_bytes(b"arbitrary bytes written after evaluation")

    with pytest.raises(
        ResultBundleError,
        match=r"stage 3 selected evaluation evidence was produced by checkpoint sha256:[0-9a-f]{64}, "
        r"not the certified selected checkpoint",
    ):
        _save_bundle(run_dir, stage_results, stage_configs)
    assert not (run_dir / "artifact_manifest.json").exists()


@pytest.mark.parametrize(
    ("recorded_hash", "expected_error"),
    [
        ("sha256:" + "f" * 64, "not the certified selected checkpoint"),
        ("not-a-digest", "records a malformed checkpoint_sha256"),
    ],
)
def test_selected_evidence_checkpoint_hash_is_bound_at_audit(
    tmp_path: Path,
    stable_provenance: None,
    recorded_hash: str,
    expected_error: str,
) -> None:
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    _rewrite_csv_column(run_dir / "stage1" / "evaluation_selected.csv", field="checkpoint_sha256", value=recorded_hash)
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert any(expected_error in error for error in report["errors"])


def test_evidence_without_a_checkpoint_hash_audits_with_a_warning(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Bundles written before the column existed stay valid — visibly unbound."""
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    for label in ("selected", "final"):
        _rewrite_csv_column(run_dir / "stage1" / f"evaluation_{label}.csv", field="checkpoint_sha256", value=None)
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    assert report["errors"] == []
    assert any("stage 1 selected evaluation evidence records no checkpoint_sha256" in w for w in report["warnings"])
    assert any("stage 1 final evaluation evidence records no checkpoint_sha256" in w for w in report["warnings"])
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"


def test_a_sidecar_rewritten_after_evaluation_cannot_be_published(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """The certified checkpoint is the model AND the VecNormalize statistics.

    The binding used to cover the model file only; a sidecar write landing
    between evaluation and bundle save was hashed into
    ``provenance.selected_checkpoints[stage].normalization_hash`` and audited
    canonical-valid.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    (run_dir / "stage3" / "models" / "best_model_vecnorm.pkl").write_bytes(b"statistics written after evaluation")

    with pytest.raises(
        ResultBundleError,
        match=r"stage 3 selected evaluation evidence ran under VecNormalize statistics sha256:[0-9a-f]{64}, "
        r"not the certified selected statistics sha256:[0-9a-f]{64}",
    ):
        _save_bundle(run_dir, stage_results, stage_configs)
    assert not (run_dir / "artifact_manifest.json").exists()


@pytest.mark.parametrize(
    ("recorded_hash", "expected_error"),
    [
        ("sha256:" + "f" * 64, "not the certified selected statistics"),
        ("not-a-digest", "records a malformed normalization_sha256"),
    ],
)
def test_selected_evidence_normalization_hash_is_bound_at_audit(
    tmp_path: Path,
    stable_provenance: None,
    recorded_hash: str,
    expected_error: str,
) -> None:
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    _rewrite_csv_column(
        run_dir / "stage1" / "evaluation_selected.csv", field="normalization_sha256", value=recorded_hash
    )
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert any(expected_error in error for error in report["errors"])


def test_evidence_without_a_normalization_hash_audits_with_a_warning(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """SB3 bundles written before the column existed stay valid — visibly unbound."""
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    for label in ("selected", "final"):
        _rewrite_csv_column(run_dir / "stage1" / f"evaluation_{label}.csv", field="normalization_sha256", value=None)
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    assert report["errors"] == []
    assert any("stage 1 selected evaluation evidence records no normalization_sha256" in w for w in report["warnings"])
    assert any("stage 1 final evaluation evidence records no normalization_sha256" in w for w in report["warnings"])
    assert not any("stage 2" in w and "normalization_sha256" in w for w in report["warnings"])


def test_jax_evidence_has_no_sidecar_to_bind_and_no_warning(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """JAX certifies no VecNormalize statistics: no column, no comparison, no noise."""
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="JAX_PPO", backend="jax-mjx")
    for stage in (1, 2, 3):
        for label in ("selected", "final"):
            _rewrite_csv_column(
                run_dir / f"stage{stage}" / f"evaluation_{label}.csv", field="normalization_sha256", value=None
            )
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    assert not any("normalization_sha256" in warning for warning in report["warnings"])


def _write_stage_budget(run_dir: Path, stage: int, *, curriculum: int | None, run: int | None) -> None:
    config_path = run_dir / f"stage{stage}" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if curriculum is not None:
        config["curriculum_kwargs"]["timesteps"] = curriculum
    if run is not None:
        config["run"] = {"seed": 42, "n_envs": 4, "timesteps": run}
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_short_budget_run_with_a_stale_checkpoint_no_longer_audits_clean(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """The probe: a resumed short-budget run over a longer run's leftovers.

    stage_config.json records the 8M curriculum budget beside the 2.5M
    invocation budget; the summary claims what stage 2 actually trained
    (200k here); and a 5M-step periodic checkpoint from the longer run sits in
    the manifest.  The shortfall is legitimate and must warn.  The checkpoint
    beyond the claimed timesteps but within the 8M budget is indistinguishable
    from a rollback-resume's leftover, so it must warn too — visibly — while a
    checkpoint beyond the budget itself cannot be this stage's and must fail.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _write_stage_budget(run_dir, 2, curriculum=8_000_000, run=2_500_000)
    _save_bundle(run_dir, stage_results, stage_configs)

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    assert any(
        "stage 2 trained 200,000 timesteps, short of its curriculum budget of 8,000,000 "
        "and run budget of 2,500,000" in warning
        for warning in report["warnings"]
    )

    model_dir = run_dir / "stage2" / "models"
    (model_dir / "stage2_5000000_steps.zip").write_bytes(b"checkpoint from the longer run")
    (model_dir / "stage2_vecnormalize_5000000_steps.pkl").write_bytes(b"its statistics")
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    assert report["errors"] == []
    assert any(
        "stage 2 periodic checkpoint models/stage2_5000000_steps.zip records 5,000,000 steps, "
        "beyond the 200,000 timesteps the summary claims for the stage but within its curriculum budget "
        "of 8,000,000: this may be a rollback-resume" in warning
        for warning in report["warnings"]
    )

    (model_dir / "stage2_9000000_steps.zip").write_bytes(b"checkpoint no 8M-budget stage could have written")
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert any(
        "stage 2 periodic checkpoint models/stage2_9000000_steps.zip records 9,000,000 steps, "
        "beyond the 200,000 timesteps the summary claims for the stage and beyond its curriculum budget "
        "of 8,000,000" in error
        for error in report["errors"]
    )
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


def test_save_refuses_a_periodic_checkpoint_beyond_the_curriculum_budget(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _write_stage_budget(run_dir, 1, curriculum=100_000, run=100_000)
    (run_dir / "stage1" / "models" / "stage1_500000_steps.zip").write_bytes(b"stale")

    with pytest.raises(
        ResultBundleError,
        match=r"stage1_500000_steps\.zip records 500,000 steps, beyond the 100,000 .* and beyond its curriculum "
        r"budget of 100,000",
    ):
        _save_bundle(run_dir, stage_results, stage_configs)
    assert not (run_dir / "artifact_manifest.json").exists()


@pytest.mark.parametrize("curriculum_budget", [None, 200_000], ids=["no-budget-recorded", "within-budget"])
def test_a_rollback_resume_publishes_with_a_warning(
    tmp_path: Path,
    stable_provenance: None,
    curriculum_budget: int | None,
) -> None:
    """A legitimate rollback layout: the stage collapsed at 120k, was resumed
    from the 80k checkpoint and finished at 100k; retention kept the
    highest-numbered pre-rollback checkpoint.  The hard error used to make
    such a run unpublishable; it now publishes, with the leftover named.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    if curriculum_budget is not None:
        _write_stage_budget(run_dir, 1, curriculum=curriculum_budget, run=None)
    model_dir = run_dir / "stage1" / "models"
    for steps in (60_000, 80_000, 100_000, 120_000):
        (model_dir / f"stage1_{steps}_steps.zip").write_bytes(b"checkpoint")
        (model_dir / f"stage1_vecnormalize_{steps}_steps.pkl").write_bytes(b"statistics")

    paths = _save_bundle(run_dir, stage_results, stage_configs)

    assert json.loads(paths["artifact_manifest"].read_text(encoding="utf-8"))["status"] == "complete"
    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    leftover_warnings = [warning for warning in report["warnings"] if "may be a rollback-resume" in warning]
    assert len(leftover_warnings) == 2  # the .zip and its vecnormalize sidecar
    assert "models/stage1_120000_steps.zip records 120,000 steps, beyond the 100,000 timesteps" in leftover_warnings[0]
    bound = (
        f"within its curriculum budget of {curriculum_budget:,}"
        if curriculum_budget is not None
        else "with no curriculum budget recorded to bound it"
    )
    assert all(bound in warning for warning in leftover_warnings)


def test_a_full_budget_run_with_in_budget_checkpoints_warns_about_nothing(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _write_stage_budget(run_dir, 1, curriculum=100_000, run=100_000)
    (run_dir / "stage1" / "models" / "stage1_100000_steps.zip").write_bytes(b"final periodic checkpoint")
    _save_bundle(run_dir, stage_results, stage_configs)

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    assert report["warnings"] == []


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
