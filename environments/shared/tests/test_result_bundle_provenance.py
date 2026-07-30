"""Tests for environments.shared.result_bundle.provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from environments.shared.reporting import save_result_bundle
from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    initialize_result_bundle,
    write_artifact_manifest,
)
from environments.shared.result_bundle import provenance as result_bundle_provenance

from .result_bundle_helpers import (
    _COMMIT,
    _complete_bundle_inputs,
    _InitializeResultBundleKwargs,
    _plant_identity,
    _stage_config,
    _stage_result,
    _write_stage_configs,
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
    assert result_bundle_provenance._sanitize_repository_url(remote) == expected


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
