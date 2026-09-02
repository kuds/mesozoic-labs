"""Tests for environments.shared.result_bundle.manifest."""

from __future__ import annotations

import json
import logging
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
    _complete_bundle_inputs,
    _plant_identity,
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


def test_stale_atomic_write_temporary_is_rejected_as_undeclared(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Stale ``.*.tmp`` temporaries are never hashed, so they must not slip past the unlisted scan.

    A crashed ``_write_json`` leaves ``.summary.json.tmp`` beside the file it
    was replacing; an "immutable" bundle carrying it must not audit clean.
    """
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    (run_dir / ".summary.json.tmp").write_bytes(b'{"partial": "write"}')

    with pytest.raises(ResultBundleError, match=r"undeclared bundle artifacts: \['\.summary\.json\.tmp'\]"):
        verify_artifact_manifest(run_dir)
    assert audit_result_bundle(run_dir)["status"] == "canonical-conflict"


#: One of every litter class the writer and verifier ignore (review ER7 fix).
_LITTER = (
    ".DS_Store",
    "stage1/._stage_config.json",
    "stage2/models/.nfs000000000000abcd00000001",
    "Thumbs.db",
    "stage3/.directory",
    ".ipynb_checkpoints/summary-checkpoint.json",
)


def _drop_files(run_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00Bud1")


def test_os_and_tooling_litter_is_ignored_by_writer_and_verifier(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Opening a bundle in Finder / Explorer / Jupyter must not break it.

    Every dot-file used to be rejected as undeclared while the writer refused
    to list any, so ``.DS_Store`` or an AppleDouble fork made every save fail
    its final validation and turned every complete bundle carrying one into
    canonical-conflict.  Litter carries no bundle content: neither hashed nor
    rejected, before or after the manifest is written.
    """
    run_dir = tmp_path / "run"
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    _drop_files(run_dir, _LITTER)

    verify_artifact_manifest(run_dir)
    assert audit_result_bundle(run_dir)["status"] == "canonical-valid"
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"

    manifest = json.loads(write_artifact_manifest(run_dir, status="complete").read_text(encoding="utf-8"))
    declared = {entry["path"] for entry in manifest["files"]}
    assert not declared & set(_LITTER)
    assert all((run_dir / name).exists() for name in _LITTER), "litter is ignored, never deleted"
    assert audit_result_bundle(run_dir)["status"] == "canonical-valid"


def test_a_bundle_saves_with_litter_already_present(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _drop_files(run_dir, _LITTER)

    from environments.shared.reporting import save_result_bundle

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

    assert json.loads(paths["artifact_manifest"].read_text(encoding="utf-8"))["status"] == "complete"
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"


def test_other_dot_files_are_content_declared_at_build_and_undeclared_after(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """A dot-file that is not litter and not a stale temporary is bundle content."""
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    content = {".gitkeep": b"", "stage1/.hidden_notes.json": b'{"note": "kept"}\n'}
    for name, payload in content.items():
        (run_dir / name).write_bytes(payload)

    manifest = json.loads(write_artifact_manifest(run_dir, status="complete").read_text(encoding="utf-8"))
    declared = {entry["path"] for entry in manifest["files"]}
    assert set(content) <= declared
    assert audit_result_bundle(run_dir)["status"] == "canonical-valid"

    (run_dir / "stage1" / ".hidden_notes.json").write_bytes(b'{"note": "edited"}\n')
    with pytest.raises(ResultBundleError, match=r"artifact size mismatch: stage1/\.hidden_notes\.json"):
        verify_artifact_manifest(run_dir)

    (run_dir / "stage1" / ".hidden_notes.json").write_bytes(content["stage1/.hidden_notes.json"])
    (run_dir / ".late_dotfile").write_bytes(b"appeared after the manifest")
    with pytest.raises(ResultBundleError, match=r"undeclared bundle artifacts: \['\.late_dotfile'\]"):
        verify_artifact_manifest(run_dir)
    assert audit_result_bundle(run_dir)["status"] == "canonical-conflict"


def test_write_artifact_manifest_discards_stale_temporaries(
    tmp_path: Path,
    stable_provenance: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The writer self-heals a crashed write instead of failing every later audit."""
    run_dir = tmp_path / "run"
    _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    stale = [run_dir / ".summary.json.tmp", run_dir / "stage1" / ".stage_config.json.tmp"]
    for path in stale:
        path.write_bytes(b"interrupted")

    with caplog.at_level(logging.WARNING, logger="environments.shared.result_bundle.manifest"):
        manifest_path = write_artifact_manifest(run_dir, status="complete")

    assert not any(path.exists() for path in stale)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not any(Path(entry["path"]).name.endswith(".tmp") for entry in manifest["files"])
    assert sum("stale temporary" in record.getMessage() for record in caplog.records) == 2
    assert audit_result_bundle(run_dir)["status"] == "canonical-valid"


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
