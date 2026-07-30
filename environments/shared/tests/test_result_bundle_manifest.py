"""Tests for environments.shared.result_bundle.manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    initialize_result_bundle,
    validate_result_bundle,
    verify_artifact_manifest,
    write_artifact_manifest,
)

from .result_bundle_helpers import (
    _complete_bundle,
)


def test_artifact_manifest_is_root_and_creation_order_independent(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    manifests: list[bytes] = []
    for index, filenames in enumerate((("b.bin", "a.json"), ("a.json", "b.bin"))):
        run_dir = tmp_path / f"copy{index}"
        initialize_result_bundle(
            run_dir,
            species="velociraptor",
            algorithm="PPO",
            seed=42,
            run_id="fixed-run-id",
            captured_at="2026-07-18T12:00:00+00:00",
        )
        payloads = {"a.json": b'{"value": 1}\n', "b.bin": b"\x00\x01\x02"}
        for filename in filenames:
            (run_dir / filename).write_bytes(payloads[filename])
        manifests.append(write_artifact_manifest(run_dir, status="partial").read_bytes())
        verify_artifact_manifest(run_dir)

    assert manifests[0] == manifests[1]


def test_artifact_manifest_detects_post_write_tampering(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    initialize_result_bundle(run_dir, species="velociraptor", algorithm="PPO", seed=42)
    artifact = run_dir / "checkpoint.bin"
    artifact.write_bytes(b"abcd")
    write_artifact_manifest(run_dir, status="partial")
    artifact.write_bytes(b"abce")

    with pytest.raises(ResultBundleError, match="hash mismatch"):
        verify_artifact_manifest(run_dir)


def test_complete_manifest_cannot_omit_required_csv(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    paths["collected_results_csv"].unlink()
    manifest_path = paths["artifact_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != "collected_results.csv"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


def test_canonical_bundle_rejects_unlisted_artifact(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    extra_checkpoint = run_dir / "stage3" / "models" / "unlisted_checkpoint.pkl"
    extra_checkpoint.write_bytes(b"not declared by the completion manifest")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


def test_manifest_and_provenance_run_ids_must_match(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    manifest = json.loads(paths["artifact_manifest"].read_text(encoding="utf-8"))
    manifest["run_id"] = "different-manifest-run"
    paths["artifact_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


@pytest.mark.parametrize(
    ("artifact_path", "replacement"),
    [
        ("stage3/models/best_model.pkl", b"replacement checkpoint"),
        ("stage1/stage_config.json", b'{"changed": true}\n'),
    ],
)
def test_manifest_refresh_cannot_hide_stale_provenance_hash(
    tmp_path: Path,
    stable_provenance: None,
    artifact_path: str,
    replacement: bytes,
) -> None:
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    (run_dir / artifact_path).write_bytes(replacement)
    write_artifact_manifest(run_dir, status="complete")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    with pytest.raises(ResultBundleError):
        validate_result_bundle(run_dir)


def test_summary_is_rejected_under_a_partial_manifest(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    write_artifact_manifest(run_dir, status="partial")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert any("only valid with a complete" in error for error in report["errors"])
