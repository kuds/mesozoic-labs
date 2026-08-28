"""Semantic stages (recovery) as first-class citizens of the result bundle.

The stage-manifest migration's final part (2026-08-23; review doc §4 item
13): a trex run that trained the recovery pilot must join the run bundle —
CSV row, summary stage, checkpoint hashes, evaluation evidence — pass audit,
and stay fail-closed everywhere a bad reference could slip in.  Built on the
same helpers as ``test_result_bundle_audit.py``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from environments.shared.reporting import save_result_bundle
from environments.shared.reporting.csv_output import build_results_csv_rows, write_results_csv
from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    compare_summary_to_csv,
    validate_result_bundle,
)
from environments.shared.result_schema import validate_result_summary

from .result_bundle_helpers import (
    _complete_bundle,
    _complete_bundle_inputs,
    _plant_identity,
    _snapshot_files,
)

_TREX_STAGES: "tuple[int | str, ...]" = (1, "recovery", 2, 3)


def test_trex_bundle_with_recovery_stage_is_canonical_valid(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """A recovery-bearing bundle publishes, audits clean, and re-validates."""
    run_dir = tmp_path / "trex-run"
    paths, _, _ = _complete_bundle(
        run_dir,
        algorithm="PPO",
        backend="stable-baselines3",
        species="trex",
        stage_refs=_TREX_STAGES,
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert set(summary["stages"]) == {"1", "recovery", "2", "3"}
    assert summary["schema_version"] == 3
    # The pilot's honest verdict is recorded, not laundered.
    assert summary["stages"]["recovery"]["stage_passed"] is False
    # The headline reward stays the terminal advancing stage's.
    assert summary["final_avg_reward"] == summary["stages"]["3"]["final_eval_reward"]
    validate_result_summary(
        summary,
        expected_species="trex",
        require_complete=True,
        canonical_provenance=True,
    )
    assert set(summary["provenance"]["selected_checkpoints"]) == {"1", "recovery", "2", "3"}

    with paths["collected_results_csv"].open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    # CSV rows in manifest order, the recovery row keyed by its id.
    assert [row["stage"] for row in rows] == ["1", "recovery", "2", "3"]
    recovery_row = rows[1]
    assert recovery_row["stage_passed"] == "False"
    assert recovery_row["curriculum_gate_kind"] == "none/v1"
    assert compare_summary_to_csv(summary, paths["collected_results_csv"]) == []

    assert audit_result_bundle(run_dir)["status"] == "canonical-valid"
    assert validate_result_bundle(run_dir, require_complete=True)["status"] == "canonical-valid"


def test_trex_bundle_with_recovery_stage_resaves_byte_identically(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "trex-run"
    paths, stage_results, stage_configs = _complete_bundle(
        run_dir,
        algorithm="PPO",
        backend="stable-baselines3",
        species="trex",
        stage_refs=_TREX_STAGES,
    )
    before = _snapshot_files(run_dir)

    second = save_result_bundle(
        stage_results,
        stage_configs,
        "trex",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity("trex"),
        run_id="trex-stable-baselines3-ppo-test",
    )

    assert second["summary"].is_file()
    assert _snapshot_files(run_dir) == before


def test_recovery_verdict_cannot_be_laundered_into_a_pass(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """gate_kind none/v1 never passes, so stage_passed=True is a
    contradiction the bundle must refuse to publish."""
    run_dir = tmp_path / "trex-run"
    stage_results, stage_configs = _complete_bundle_inputs(
        run_dir,
        algorithm="PPO",
        species="trex",
        stage_refs=_TREX_STAGES,
    )
    recovery_result = next(result for result in stage_results if result["stage"] == "recovery")
    recovery_result["gate_passed"] = True
    recovery_result["publication_gate_passed"] = True

    with pytest.raises(ResultBundleError, match="none/v1.*stage_passed=true"):
        save_result_bundle(
            stage_results,
            stage_configs,
            "trex",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="test-backend-1.0",
            parallel_envs=4,
            evaluation_episodes=3,
            evaluation_seeds=[101, 102, 103],
            plant_identity=_plant_identity("trex"),
            run_id="trex-laundered-recovery",
        )


def test_recovery_without_its_stance_parent_is_rejected(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Recovery warm-starts from stance; a bundle recording it without the
    earlier advancing stage is incoherent (the historical contiguous-prefix
    rule, restated over the manifest)."""
    run_dir = tmp_path / "trex-run"
    stage_results, stage_configs = _complete_bundle_inputs(
        run_dir,
        algorithm="PPO",
        species="trex",
        stage_refs=(1, "recovery"),
    )
    orphaned = [result for result in stage_results if result["stage"] == "recovery"]

    with pytest.raises(ResultBundleError, match="contiguous curriculum prefix"):
        save_result_bundle(
            orphaned,
            {"recovery": stage_configs["recovery"]},
            "trex",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="test-backend-1.0",
            parallel_envs=4,
            evaluation_seeds=[101],
            plant_identity=_plant_identity("trex"),
        )


def test_partial_stance_plus_recovery_bundle_stays_partial(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """The pilot's recorded False verdict must not read as a FAILED
    curriculum: only advancing gates decide the bundle status."""
    run_dir = tmp_path / "trex-run"
    stage_results, stage_configs = _complete_bundle_inputs(
        run_dir,
        algorithm="PPO",
        species="trex",
        stage_refs=(1, "recovery"),
    )

    paths = save_result_bundle(
        stage_results,
        stage_configs,
        "trex",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity("trex"),
        run_id="trex-partial-with-recovery",
    )

    assert "summary" not in paths
    assert json.loads(paths["artifact_manifest"].read_text(encoding="utf-8"))["status"] == "partial"
    with paths["collected_results_csv"].open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert [row["stage"] for row in rows] == ["1", "recovery"]
    assert audit_result_bundle(run_dir)["status"] == "partial"


def test_a_stage_reference_outside_the_manifest_fails_closed(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "trex-run"
    stage_results, stage_configs = _complete_bundle_inputs(
        run_dir,
        algorithm="PPO",
        species="trex",
        stage_refs=(1,),
    )
    stage_results[0]["stage"] = "warp"

    with pytest.raises(ResultBundleError, match="manifest declares"):
        save_result_bundle(
            stage_results,
            stage_configs,
            "trex",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="test-backend-1.0",
            parallel_envs=4,
            evaluation_seeds=[101],
            plant_identity=_plant_identity("trex"),
        )


def test_recovery_collected_results_row_round_trips(tmp_path: Path) -> None:
    """A semantic-stage CSV row is writable and re-readable as itself."""
    stage_results, stage_configs = _complete_bundle_inputs(
        tmp_path / "inputs",
        algorithm="PPO",
        species="trex",
        stage_refs=(1, "recovery"),
    )
    rows = build_results_csv_rows(
        stage_results,
        stage_configs,
        "trex",
        "PPO",
        42,
        backend="stable-baselines3",
        run_id="round-trip",
    )
    assert [row["stage"] for row in rows] == [1, "recovery"]
    assert rows[1]["curriculum_gate_kind"] == "none/v1"

    csv_path = write_results_csv(rows, tmp_path / "collected_results.csv")
    with csv_path.open(newline="", encoding="utf-8") as source:
        reread = list(csv.DictReader(source))
    assert [row["stage"] for row in reread] == ["1", "recovery"]
    assert reread[1]["species"] == "trex"
    assert reread[1]["schema_version"] == "3"
