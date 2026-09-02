"""Integration tests for environments.shared.result_bundle.audit.

These drive a whole run directory through save_result_bundle and then
audit_result_bundle, so they span the package by design."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from environments.shared import result_bundle
from environments.shared.reporting import save_result_bundle
from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    compare_summary_to_csv,
    sha256_file,
    validate_result_bundle,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from environments.shared.result_schema import validate_result_summary

from .result_bundle_helpers import (
    _complete_bundle,
    _complete_bundle_inputs,
    _plant_identity,
    _snapshot_files,
    _stage_config,
    _stage_result,
    _write_stage_configs,
)


def test_partial_bundle_is_valid_but_not_publishable(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "logs" / "velociraptor" / "ppo" / "20260718_120000"
    configs: dict[int | str, dict[str, Any]] = {1: _stage_config(1, "PPO")}
    _write_stage_configs(run_dir, configs)

    paths = save_result_bundle(
        [_stage_result(1)],
        configs,
        "velociraptor",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="2.7.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101],
        plant_identity=_plant_identity(),
        run_id="partial-test",
    )

    assert "summary" not in paths
    assert not (run_dir / "summary.json").exists()
    assert audit_result_bundle(run_dir)["status"] == "partial"
    assert validate_result_bundle(run_dir, require_complete=False)["status"] == "partial"
    with pytest.raises(ResultBundleError, match="result bundle is partial"):
        validate_result_bundle(run_dir, require_complete=True)


@pytest.mark.parametrize(
    ("algorithm", "backend"),
    [
        ("PPO", "stable-baselines3"),
        ("SAC", "stable-baselines3"),
        ("JAX_PPO", "jax-mjx"),
    ],
)
def test_complete_bundle_round_trips_exporter_schema_csv_and_manifest(
    tmp_path: Path,
    stable_provenance: None,
    algorithm: str,
    backend: str,
) -> None:
    run_dir = tmp_path / backend / algorithm.lower()
    paths, stage_results, stage_configs = _complete_bundle(run_dir, algorithm=algorithm, backend=backend)

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    validate_result_summary(
        summary,
        expected_species="velociraptor",
        require_complete=True,
        canonical_provenance=True,
    )
    assert summary["algorithm"] in {"PPO", "SAC"}
    assert summary["backend"] == backend
    assert summary["backend_version"] == "test-backend-1.0"
    assert compare_summary_to_csv(summary, paths["collected_results_csv"]) == []
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"
    verify_artifact_manifest(run_dir)

    first_bytes = {name: path.read_bytes() for name, path in paths.items()}
    second_paths = save_result_bundle(
        stage_results,
        stage_configs,
        "velociraptor",
        algorithm,
        42,
        run_dir,
        backend=backend,
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(),
        run_id=f"velociraptor-{backend}-{algorithm.lower()}-test",
    )
    assert {name: path.read_bytes() for name, path in second_paths.items()} == first_bytes


def test_failed_save_preserves_existing_complete_bundle_byte_for_byte(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    _, stage_results, stage_configs = _complete_bundle(
        run_dir,
        algorithm="PPO",
        backend="stable-baselines3",
    )
    before = _snapshot_files(run_dir)

    with pytest.raises(ResultBundleError):
        save_result_bundle(
            stage_results[:1],
            {1: stage_configs[1]},
            "velociraptor",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="test-backend-1.0",
            parallel_envs=4,
            evaluation_episodes=3,
            evaluation_seeds=[101, 102, 103],
            plant_identity=_plant_identity(),
            run_id="velociraptor-stable-baselines3-ppo-test",
        )

    assert _snapshot_files(run_dir) == before
    assert verify_artifact_manifest(run_dir)["status"] == "complete"
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"


@pytest.mark.parametrize("failed_stage", [1, 2, 3])
def test_any_failed_curriculum_gate_is_not_a_complete_bundle(
    tmp_path: Path,
    stable_provenance: None,
    failed_stage: int,
) -> None:
    run_dir = tmp_path / f"failed-stage-{failed_stage}"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    stage_results[failed_stage - 1]["gate_passed"] = False
    stage_results[failed_stage - 1]["publication_gate_passed"] = False

    paths = save_result_bundle(
        stage_results,
        stage_configs,
        "velociraptor",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(),
        run_id=f"failed-stage-{failed_stage}",
    )

    assert "summary" not in paths
    assert not (run_dir / "summary.json").exists()
    assert json.loads(paths["artifact_manifest"].read_text(encoding="utf-8"))["status"] == "failed"
    assert audit_result_bundle(run_dir)["status"] == "failed"
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir, require_complete=True)


def test_summary_and_provenance_run_ids_must_match(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance["run_id"] = "different-provenance-run"
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


@pytest.mark.parametrize("hash_field", ["model_hash", "config_hash"])
def test_summary_and_provenance_hashes_must_match(
    tmp_path: Path,
    stable_provenance: None,
    hash_field: str,
) -> None:
    run_dir = tmp_path / hash_field
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance[hash_field] = "sha256:" + "f" * 64
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


def test_mixed_stage_plant_identities_are_rejected(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    stage_results[1]["plant_identity"] = _plant_identity() | {"physics_revision": 2}

    with pytest.raises(ResultBundleError, match="plant"):
        save_result_bundle(
            stage_results,
            stage_configs,
            "velociraptor",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="test-backend-1.0",
            parallel_envs=4,
            evaluation_episodes=3,
            evaluation_seeds=[101, 102, 103],
            plant_identity=_plant_identity(),
            run_id="mixed-plant-test",
        )


def test_malformed_plant_identity_is_rejected(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    malformed_plant = _plant_identity()
    malformed_plant["physics_sha256"] = "not-a-sha256"
    for result in stage_results:
        result["plant_identity"] = malformed_plant

    with pytest.raises(ResultBundleError, match="plant"):
        save_result_bundle(
            stage_results,
            stage_configs,
            "velociraptor",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="test-backend-1.0",
            parallel_envs=4,
            evaluation_episodes=3,
            evaluation_seeds=[101, 102, 103],
            plant_identity=malformed_plant,
            run_id="malformed-plant-test",
        )


def test_summary_and_provenance_plant_identities_must_match(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["plant_identity"]["physics_revision"] = 2
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


def test_backend_version_must_match_between_summary_and_provenance(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance["backend_version"] = "different-backend"
    paths["provenance"].write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["provenance"] = provenance
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert any("backend_version" in error for error in report["errors"])


def test_identical_completed_export_is_a_write_free_noop(
    tmp_path: Path,
    stable_provenance: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _, stage_results, stage_configs = _complete_bundle(
        run_dir,
        algorithm="PPO",
        backend="stable-baselines3",
    )
    before = _snapshot_files(run_dir)

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated Drive write failure")

    # Patch the hashing module, not the package re-export: every internal
    # writer calls it through that module, so this one point covers
    # initialize_result_bundle, update_provenance, write_artifact_manifest
    # and save_result_bundle's own writes.
    monkeypatch.setattr(result_bundle.hashing, "_write_json", fail_write)
    paths = save_result_bundle(
        stage_results,
        stage_configs,
        "velociraptor",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(),
        run_id="velociraptor-stable-baselines3-ppo-test",
    )

    assert paths["artifact_manifest"].is_file()
    assert _snapshot_files(run_dir) == before


def test_save_rejects_skipped_curriculum_stages_before_writing(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    config = _stage_config(2, "PPO")
    _write_stage_configs(run_dir, {2: config})

    with pytest.raises(ResultBundleError, match="contiguous curriculum prefix"):
        save_result_bundle(
            [_stage_result(2)],
            {2: config},
            "velociraptor",
            "PPO",
            42,
            run_dir,
            backend="stable-baselines3",
            backend_version="2.7.0",
            parallel_envs=4,
            evaluation_seeds=[101],
            plant_identity=_plant_identity(),
        )
    assert not (run_dir / "provenance.json").exists()


def test_complete_save_requires_plant_identity_in_every_hashed_stage_config(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    config_path = run_dir / "stage2" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["plant_identity"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ResultBundleError, match="stage config is missing plant identity"):
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


def test_csv_hyperparameters_are_derived_from_hashed_stage_configs(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    stage_configs[1]["env_kwargs"]["forward_vel_weight"] = 999.0
    paths = save_result_bundle(
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
    with paths["collected_results_csv"].open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["env_forward_vel_weight"] == "1.0"


def _save_bundle(run_dir: Path, stage_results: Any, stage_configs: Any) -> dict[str, Path]:
    return save_result_bundle(
        stage_results,
        stage_configs,
        "velociraptor",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(),
        run_id="velociraptor-stable-baselines3-ppo-test",
    )


def _write_run_block(run_dir: Path, stage: int, run_block: dict[str, Any]) -> None:
    config_path = run_dir / f"stage{stage}" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["run"] = run_block
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_load_lineage_in_stage_config_is_validated_and_surfaced(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Lineage the trainer persists into the run block is checked and reported.

    A run block without lineage keys (stage 1) is skipped; a parent inside
    the bundle (stage 2) must hash as the manifest says; a parent elsewhere
    (stage 3) is surfaced but not checkable.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _write_run_block(run_dir, 1, {"seed": 42, "n_envs": 4, "timesteps": 100_000})
    in_bundle = {
        "load_path": "stage1/models/best_model.pkl",
        "load_mode": "initialize_next_stage",
        "parent_task_sha256": "sha256:" + "1" * 64,
        "parent_checkpoint_sha256": sha256_file(run_dir / "stage1" / "models" / "best_model.pkl"),
    }
    _write_run_block(run_dir, 2, {"seed": 42, "n_envs": 4, **in_bundle})
    elsewhere = {
        "load_path": "/content/drive/MyDrive/runs/earlier/stage2/models/best_model.zip",
        "load_mode": "resume_same_stage",
        "parent_checkpoint_sha256": "sha256:" + "2" * 64,
    }
    _write_run_block(run_dir, 3, {"seed": 42, "n_envs": 4, **elsewhere})
    _save_bundle(run_dir, stage_results, stage_configs)

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-valid"
    assert report["errors"] == []
    assert report["lineage"] == {"2": in_bundle, "3": elsewhere}


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"load_mode": "warp"}, "run.load_mode 'warp' is not one of"),
        ({"parent_checkpoint_sha256": "sha256:" + "f" * 64}, "but the manifest hashes that artifact as"),
        ({"parent_task_sha256": "not-a-digest"}, "run.parent_task_sha256 must be sha256:<64 lowercase hex>"),
        ({"load_path": ""}, "run.load_path must be a non-empty string"),
    ],
)
def test_inconsistent_load_lineage_fails_before_a_complete_manifest_is_written(
    tmp_path: Path,
    stable_provenance: None,
    override: dict[str, Any],
    expected_error: str,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    lineage = {
        "load_path": "stage1/models/best_model.pkl",
        "load_mode": "initialize_next_stage",
        "parent_task_sha256": "sha256:" + "1" * 64,
        "parent_checkpoint_sha256": sha256_file(run_dir / "stage1" / "models" / "best_model.pkl"),
        **override,
    }
    _write_run_block(run_dir, 2, {"seed": 42, "n_envs": 4, **lineage})

    with pytest.raises(ResultBundleError) as excinfo:
        _save_bundle(run_dir, stage_results, stage_configs)
    assert expected_error in str(excinfo.value)
    assert not (run_dir / "artifact_manifest.json").exists()


def test_failed_final_validation_writes_no_complete_manifest(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """The completion marker is written only after the bundle has passed.

    A dot-file is unlisted by the writer and rejected by verification, so it
    fails the save's final validation the way any post-hash surprise would.
    The marker used to be on disk by then, wedging every retry at the
    re-entry gate; now nothing is marked complete and the retry publishes.
    """
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    stray = run_dir / ".DS_Store"
    stray.write_bytes(b"\x00")

    with pytest.raises(ResultBundleError, match=r"undeclared bundle artifacts: \['\.DS_Store'\]"):
        _save_bundle(run_dir, stage_results, stage_configs)
    assert not (run_dir / "artifact_manifest.json").exists()

    stray.unlink()
    paths = _save_bundle(run_dir, stage_results, stage_configs)
    assert json.loads(paths["artifact_manifest"].read_text(encoding="utf-8"))["status"] == "complete"
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"


def test_a_wedged_complete_bundle_is_downgraded_and_rebuilt(
    tmp_path: Path,
    stable_provenance: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A complete marker that no longer verifies is healed, not fatal.

    Older saves wrote the marker before their final validation; a hashed file
    changing afterwards left the run permanently unpromotable except by
    manual deletion of artifact_manifest.json.  Re-entry now downgrades the
    marker with a logged reason and takes the rebuild path.
    """
    run_dir = tmp_path / "run"
    paths, stage_results, stage_configs = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    pristine = _snapshot_files(run_dir)
    paths["collected_results_csv"].write_bytes(b"corrupted after completion\n")
    assert audit_result_bundle(run_dir)["status"] == "canonical-conflict"

    with caplog.at_level(logging.WARNING, logger="environments.shared.reporting.bundles"):
        rebuilt = _save_bundle(run_dir, stage_results, stage_configs)

    assert any("downgrading it to partial and rebuilding" in record.getMessage() for record in caplog.records)
    assert json.loads(rebuilt["artifact_manifest"].read_text(encoding="utf-8"))["status"] == "complete"
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"
    assert _snapshot_files(run_dir) == pristine


def test_legacy_drive_directory_is_audited_without_mutation(tmp_path: Path) -> None:
    run_dir = tmp_path / "logs" / "velociraptor" / "ppo" / "20260315_120000"
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "collected_results.csv"
    csv_path.write_text("species,algorithm,stage\nvelociraptor,ppo,1\n", encoding="utf-8")
    before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}

    report = audit_result_bundle(run_dir)

    assert report["status"] == "legacy-unverified"
    assert report["canonical"] is False
    assert any("legacy run has no captured provenance" in warning for warning in report["warnings"])
    after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    assert after == before


def test_complete_bundle_in_position_prefixed_layout_audits_clean(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Runs from 2026-08-20 name stage dirs NN_id, not stage{N}.

    Regression: the complete-manifest required-path set hardcoded
    stage{N}/... nine ways, so a new-layout bundle passed every save-side
    check, wrote a status="complete" manifest, then failed its own final
    validate as canonical-conflict — permanently wedged, since retries
    die in preflight against the "complete" status. The required paths
    must resolve to whatever the run actually named its directories.
    """
    layout = {1: "01_stance", 2: "02_locomotion", 3: "03_behavior"}
    run_dir = tmp_path / "sb3" / "ppo"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="sb3", dirname=lambda stage: layout[stage])
    # The save's own final validate already passed, or paths would not
    # exist; auditing again must also be clean, and re-validation must not
    # report canonical-conflict.
    assert paths["summary"].is_file()
    result_bundle.validate_result_bundle(run_dir, require_complete=True)
