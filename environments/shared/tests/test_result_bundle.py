"""Integration tests for portable Colab/Google Drive result bundles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypedDict

import pytest

from environments.shared import result_bundle
from environments.shared.reporting import save_evaluation_episodes, save_result_bundle
from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    compare_summary_to_csv,
    initialize_result_bundle,
    validate_result_bundle,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from environments.shared.result_schema import validate_result_summary

_COMMIT = "a" * 40


class _InitializeResultBundleKwargs(TypedDict):
    species: str
    algorithm: str
    backend: str
    seed: int
    evaluation_seeds: list[int]
    evaluation_episodes: int
    parallel_envs: int
    plant_identity: dict[str, Any]
    run_id: str
    captured_at: str


def _plant_identity() -> dict[str, Any]:
    return {
        "schema": "mesozoic.plant-identity/v1",
        "species": "velociraptor",
        "model_path": "environments/velociraptor/assets/raptor.xml",
        "physics_revision": 1,
        "policy_interface_revision": 1,
        "visual_revision": 1,
        "source_closure_sha256": "sha256:" + "1" * 64,
        "policy_interface_sha256": "sha256:" + "2" * 64,
        "physics_sha256": "sha256:" + "3" * 64,
        "visual_sha256": "sha256:" + "4" * 64,
        "nq": 31,
        "nv": 30,
        "nu": 22,
        "observation_dim": 67,
        "action_dim": 22,
    }


def _stage_result(stage: int, *, model_path: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": stage,
        "name": f"Stage {stage}",
        "description": f"Description for stage {stage}",
        "timesteps": stage * 100_000,
        "duration_seconds": stage * 10.0,
        "mean_reward": 50.0 + stage,
        "std_reward": 2.0,
        "mean_episode_length": 200.0 + stage,
        "std_episode_length": 4.9,
        "mean_forward_vel": 0.25 * stage,
        "std_forward_vel": 0.05,
        "mean_distance_traveled": 1.5 * stage,
        "mean_success_rate": stage / 3,
        "best_eval_reward": 75.0 + stage,
        "best_eval_std": 1.5,
        "best_eval_length": 210.0 + stage,
        "best_eval_timestep": stage * 90_000,
        "best_model_reward": float(stage + 1),
        "best_model_std_reward": 0.82,
        "best_model_length": 110.0,
        "best_model_std_length": 8.2,
        "best_model_fwd_vel": 0.2,
        "best_model_std_fwd_vel": 0.082,
        "best_model_distance": 2.0,
        "best_model_success_rate": 0.6667,
        "gate_passed": True,
        "publication_gate_passed": True,
        "plant_identity": _plant_identity(),
    }
    if model_path is not None:
        result["model_path"] = str(model_path)
    return result


def _stage_config(stage: int, algorithm: str) -> dict[str, Any]:
    algorithm_key = "jax_kwargs" if algorithm == "JAX_PPO" else f"{algorithm.lower()}_kwargs"
    return {
        "name": f"Stage {stage}",
        "description": f"Description for stage {stage}",
        "env_kwargs": {"forward_vel_weight": float(stage)},
        algorithm_key: {"learning_rate": 3e-4},
        "curriculum_kwargs": {"min_avg_reward": 1.0},
    }


@pytest.fixture
def stable_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove host Git/package state from result-bundle assertions."""
    monkeypatch.setattr(
        result_bundle,
        "_repository_state",
        lambda _root: {
            "repository_url": "https://github.com/kuds/mesozoic-labs.git",
            "repository_commit": _COMMIT,
            "repository_dirty": False,
            "repository_patch_sha256": None,
        },
    )
    monkeypatch.setattr(
        result_bundle,
        "_dependency_versions",
        lambda: {
            "mesozoic_labs": "0.3.3.dev0",
            "mujoco": "3.10.0",
            "mujoco_mjx": "3.10.0",
            "gymnasium": "1.2.0",
            "numpy": "2.3.0",
            "stable_baselines3": "2.7.0",
            "jax": "0.11.0",
            "flax": "0.12.7",
            "optax": "0.2.8",
        },
    )


def _write_stage_configs(run_dir: Path, stage_configs: dict[int, dict[str, Any]]) -> None:
    for stage, config in stage_configs.items():
        stage_dir = run_dir / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        if "jax_kwargs" in config:
            algorithm = "JAX_PPO"
        elif "sac_kwargs" in config:
            algorithm = "SAC"
        else:
            algorithm = "PPO"
        saved_config = {
            **config,
            "species": "velociraptor",
            "stage": stage,
            "algorithm": algorithm,
            "plant_identity": _plant_identity(),
        }
        (stage_dir / "stage_config.json").write_text(
            json.dumps(saved_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _snapshot_files(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _rewrite_csv_cell(path: Path, *, field: str, value: str) -> None:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    assert rows
    assert field in fieldnames
    rows[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _complete_bundle_inputs(
    run_dir: Path,
    *,
    algorithm: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    stage_configs = {stage: _stage_config(stage, algorithm) for stage in (1, 2, 3)}
    _write_stage_configs(run_dir, stage_configs)
    stage_results = []
    for stage in (1, 2, 3):
        selected_model = run_dir / f"stage{stage}" / "models" / "best_model.pkl"
        selected_model.parent.mkdir(parents=True, exist_ok=True)
        selected_model.write_bytes(f"selected model stage {stage}".encode())
        selected_vecnorm = run_dir / f"stage{stage}" / "models" / "best_model_vecnorm.pkl"
        selected_vecnorm.write_bytes(f"selected normalization stage {stage}".encode())
        stage_result = _stage_result(stage, model_path=selected_model)
        stage_result["vecnorm_path"] = str(selected_vecnorm)
        stage_results.append(stage_result)
    for stage in (1, 2, 3):
        final_reward = 50.0 + stage
        final_forward_velocity = 0.25 * stage
        final_distance = 1.5 * stage
        save_evaluation_episodes(
            run_dir / f"stage{stage}",
            rewards=[
                final_reward - 2.449489743,
                final_reward,
                final_reward + 2.449489743,
            ],
            lengths=[194 + stage, 200 + stage, 206 + stage],
            forward_velocities=[
                final_forward_velocity - 0.061237244,
                final_forward_velocity,
                final_forward_velocity + 0.061237244,
            ],
            distances=[final_distance - 1.0, final_distance, final_distance + 1.0],
            successes=[episode < stage for episode in range(3)],
            evaluation_seed=101,
            checkpoint_label="final",
        )
        save_evaluation_episodes(
            run_dir / f"stage{stage}",
            rewards=[float(stage), float(stage + 1), float(stage + 2)],
            lengths=[100, 110, 120],
            forward_velocities=[0.1, 0.2, 0.3],
            distances=[1.0, 2.0, 3.0],
            successes=[False, True, True],
            evaluation_seed=101,
            checkpoint_label="selected",
        )
    return stage_results, stage_configs


def _complete_bundle(
    run_dir: Path,
    *,
    algorithm: str,
    backend: str,
) -> tuple[dict[str, Path], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm=algorithm)
    paths = save_result_bundle(
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
    return paths, stage_results, stage_configs


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


def test_initialize_result_bundle_is_idempotent_and_rejects_reuse(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "logs" / "velociraptor" / "ppo" / "20260718_120000"
    kwargs: _InitializeResultBundleKwargs = {
        "species": "velociraptor",
        "algorithm": "PPO",
        "backend": "stable-baselines3",
        "seed": 42,
        "evaluation_seeds": [11, 12],
        "evaluation_episodes": 2,
        "parallel_envs": 4,
        "plant_identity": _plant_identity(),
        "run_id": "fixed-run-id",
        "captured_at": "2026-07-18T12:00:00+00:00",
    }

    first = initialize_result_bundle(run_dir, **kwargs)
    first_bytes = first.read_bytes()
    second = initialize_result_bundle(run_dir, **kwargs)
    assert second == first
    assert second.read_bytes() == first_bytes

    provenance = json.loads(first.read_text(encoding="utf-8"))
    assert provenance["repository_commit"] == _COMMIT
    assert provenance["algorithm"] == "PPO"
    assert provenance["backend"] == "stable-baselines3"
    assert provenance["evaluation_seeds"] == [11, 12]

    changed_kwargs = kwargs.copy()
    changed_kwargs["seed"] = 43
    with pytest.raises(ResultBundleError, match="different run"):
        initialize_result_bundle(run_dir, **changed_kwargs)


def test_initialize_result_bundle_rejects_duplicate_evaluation_seeds(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    with pytest.raises(ResultBundleError, match="must not contain duplicates"):
        initialize_result_bundle(
            tmp_path / "run",
            species="velociraptor",
            algorithm="PPO",
            seed=42,
            evaluation_seeds=[11, 11],
        )


def test_initialize_result_bundle_rejects_an_unlisted_evaluation_role_seed(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    with pytest.raises(ResultBundleError, match="exactly match"):
        initialize_result_bundle(
            tmp_path / "run",
            species="velociraptor",
            algorithm="PPO",
            seed=42,
            evaluation_seeds=[11],
            seed_roles={
                "training": 42,
                "publication_evaluation": 11,
                "checkpoint_selection_evaluation": 999,
            },
        )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("evaluation_seeds", [11, 13]),
        ("evaluation_episodes", 3),
        ("parallel_envs", 8),
    ],
)
def test_initialize_result_bundle_rejects_changed_evaluation_identity(
    tmp_path: Path,
    stable_provenance: None,
    field: str,
    changed_value: Any,
) -> None:
    run_dir = tmp_path / "run"
    kwargs: dict[str, Any] = {
        "species": "velociraptor",
        "algorithm": "PPO",
        "backend": "stable-baselines3",
        "seed": 42,
        "evaluation_seeds": [11, 12],
        "evaluation_episodes": 2,
        "parallel_envs": 4,
        "plant_identity": _plant_identity(),
        "run_id": "fixed-run-id",
        "captured_at": "2026-07-18T12:00:00+00:00",
    }
    provenance_path = initialize_result_bundle(run_dir, **kwargs)
    before = provenance_path.read_bytes()

    with pytest.raises(ResultBundleError, match="different run"):
        initialize_result_bundle(run_dir, **(kwargs | {field: changed_value}))

    assert provenance_path.read_bytes() == before


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


def test_partial_bundle_is_valid_but_not_publishable(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "logs" / "velociraptor" / "ppo" / "20260718_120000"
    configs = {1: _stage_config(1, "PPO")}
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


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        (
            "https://user:secret-token@github.com/kuds/mesozoic-labs.git?token=also-secret",
            "https://github.com/kuds/mesozoic-labs.git",
        ),
        ("git@github.com:kuds/mesozoic-labs.git", "github.com:kuds/mesozoic-labs.git"),
        ("/content/mesozoic-labs", None),
        ("file:///content/mesozoic-labs", None),
    ],
)
def test_repository_url_sanitizer_never_publishes_credentials_or_local_paths(
    remote: str,
    expected: str | None,
) -> None:
    assert result_bundle._sanitize_repository_url(remote) == expected


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


def test_partial_bundle_requires_complete_capture_time_provenance(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    configs = {1: _stage_config(1, "PPO")}
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
    )
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    del provenance["species"]
    paths["provenance"].write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(run_dir, status="partial")

    report = audit_result_bundle(run_dir)
    assert report["status"] == "canonical-conflict"
    assert any("species" in error for error in report["errors"])


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

    monkeypatch.setattr(result_bundle, "_write_json", fail_write)
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


def test_summary_date_is_derived_from_immutable_capture_time(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    run_dir = tmp_path / "run"
    initialize_result_bundle(
        run_dir,
        species="velociraptor",
        algorithm="PPO",
        backend="stable-baselines3",
        seed=42,
        evaluation_seeds=[101, 102, 103],
        evaluation_episodes=3,
        parallel_envs=4,
        hardware="Google Colab",
        plant_identity=_plant_identity(),
        run_id="stable-date",
        captured_at="2020-02-03T23:59:59+00:00",
    )
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
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
        run_id="stable-date",
    )
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["date"] == "2020-02-03"


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
