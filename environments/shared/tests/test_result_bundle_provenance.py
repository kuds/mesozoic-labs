"""Tests for environments.shared.result_bundle.provenance."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from environments.shared import result_bundle
from environments.shared.reporting import save_result_bundle
from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    initialize_result_bundle,
    validate_result_bundle,
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


def test_initialize_result_bundle_rejects_a_publication_seed_shared_with_training(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Evaluating on the training seed replays the reset sequence the policy fitted."""
    with pytest.raises(ResultBundleError, match=r"publication_evaluation reuses the training seed 42"):
        initialize_result_bundle(
            tmp_path / "run",
            species="velociraptor",
            algorithm="PPO",
            seed=42,
            evaluation_seeds=[42],
        )
    assert not (tmp_path / "run" / "provenance.json").exists()


def test_initialize_result_bundle_rejects_a_publication_seed_shared_with_selection(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """Publishing on the selection seed reports the maximum of noisy draws (winner's curse)."""
    with pytest.raises(ResultBundleError, match=r"reuses the checkpoint_selection_evaluation seed 1042"):
        initialize_result_bundle(
            tmp_path / "run",
            species="velociraptor",
            algorithm="PPO",
            seed=42,
            evaluation_seeds=[1042],
            seed_roles={
                "training": 42,
                "checkpoint_selection_evaluation": 1042,
                "publication_evaluation": 1042,
            },
        )
    assert not (tmp_path / "run" / "provenance.json").exists()


def test_distinct_seed_roles_publish_canonical_valid(
    tmp_path: Path,
    stable_provenance: None,
) -> None:
    """The notebook's layout — SEED, SEED+1000 for selection, a third for publication."""
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    seed_roles = {"training": 42, "checkpoint_selection_evaluation": 1042, "publication_evaluation": 101}

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
        evaluation_seeds=[1042, 101],
        seed_roles=seed_roles,
        plant_identity=_plant_identity(),
        run_id="distinct-roles",
    )

    assert json.loads(paths["provenance"].read_text(encoding="utf-8"))["seed_roles"] == seed_roles
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"


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


def _resume_run_kwargs() -> dict[str, Any]:
    """Initializer kwargs shared by the multi-session (resume) tests."""
    return {
        "species": "velociraptor",
        "algorithm": "PPO",
        "backend": "stable-baselines3",
        "seed": 42,
        "evaluation_seeds": [11, 12],
        "evaluation_episodes": 2,
        "parallel_envs": 4,
        "hardware": "Google Colab",
        "plant_identity": _plant_identity(),
        "run_id": "resumed-run",
        "captured_at": "2026-07-18T12:00:00+00:00",
    }


def test_fresh_session_resume_appends_a_session_record_without_drift(
    tmp_path: Path,
    stable_provenance: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    kwargs = _resume_run_kwargs()
    first = initialize_result_bundle(run_dir, **kwargs)
    original = json.loads(first.read_text(encoding="utf-8"))
    assert original["sessions"] == [
        {
            "session_token": result_bundle_provenance._PROCESS_SESSION_TOKEN,
            "started_at": "2026-07-18T12:00:00+00:00",
        }
    ]

    monkeypatch.setattr(result_bundle_provenance, "_PROCESS_SESSION_TOKEN", "f" * 32)
    second = initialize_result_bundle(run_dir, **(kwargs | {"captured_at": "2026-07-19T09:00:00+00:00"}))

    assert second == first
    resumed = json.loads(second.read_text(encoding="utf-8"))
    assert resumed["sessions"] == [
        original["sessions"][0],
        {"session_token": "f" * 32, "resumed_at": "2026-07-19T09:00:00+00:00"},
    ]
    assert "environment_drift" not in resumed["sessions"][-1]
    assert {key: value for key, value in resumed.items() if key != "sessions"} == {
        key: value for key, value in original.items() if key != "sessions"
    }


def test_fresh_session_environment_drift_is_absorbed_and_recorded(
    tmp_path: Path,
    stable_provenance: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_dir = tmp_path / "run"
    kwargs = _resume_run_kwargs()
    first = initialize_result_bundle(run_dir, **kwargs)
    original_dependencies = json.loads(first.read_text(encoding="utf-8"))["dependency_versions"]

    drifted_dependencies = original_dependencies | {"numpy": "2.4.1"}
    monkeypatch.setattr(result_bundle_provenance, "_dependency_versions", lambda: drifted_dependencies)
    monkeypatch.setattr(result_bundle_provenance, "_PROCESS_SESSION_TOKEN", "f" * 32)
    with caplog.at_level(logging.WARNING, logger=result_bundle_provenance.__name__):
        second = initialize_result_bundle(run_dir, **(kwargs | {"captured_at": "2026-07-19T09:00:00+00:00"}))

    resumed = json.loads(second.read_text(encoding="utf-8"))
    # The top-level fields keep the original session's environment.
    assert resumed["dependency_versions"] == original_dependencies
    assert resumed["sessions"][-1] == {
        "session_token": "f" * 32,
        "resumed_at": "2026-07-19T09:00:00+00:00",
        "environment_drift": {
            "dependency_versions": {"was": original_dependencies, "now": drifted_dependencies},
        },
    }
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("dependency_versions" in record.getMessage() for record in warnings)


def test_identity_mismatch_on_resume_still_refuses_and_records_no_session(
    tmp_path: Path,
    stable_provenance: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    kwargs = _resume_run_kwargs()
    first = initialize_result_bundle(run_dir, **kwargs)
    before = first.read_bytes()

    monkeypatch.setattr(result_bundle_provenance, "_PROCESS_SESSION_TOKEN", "f" * 32)
    with pytest.raises(ResultBundleError, match="different run"):
        initialize_result_bundle(run_dir, **(kwargs | {"seed": 43}))

    assert first.read_bytes() == before


def test_old_style_provenance_without_sessions_still_audits_and_resumes(
    tmp_path: Path,
    stable_provenance: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
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
        run_id="legacy-no-sessions",
    )
    legacy = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    del legacy["sessions"]
    paths["provenance"].write_text(json.dumps(legacy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(run_dir, status="partial")

    # An older bundle that never recorded sessions keeps validating.
    report = audit_result_bundle(run_dir)
    assert report["status"] == "partial"
    assert report["errors"] == []

    monkeypatch.setattr(result_bundle_provenance, "_PROCESS_SESSION_TOKEN", "f" * 32)
    resumed_path = initialize_result_bundle(
        run_dir,
        species="velociraptor",
        algorithm="PPO",
        backend="stable-baselines3",
        seed=42,
        evaluation_seeds=[101],
        evaluation_episodes=3,
        parallel_envs=4,
        hardware="Google Colab",
        plant_identity=_plant_identity(),
        run_id="legacy-no-sessions",
        captured_at="2026-07-19T09:00:00+00:00",
    )
    resumed = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert resumed["sessions"] == [{"session_token": "f" * 32, "resumed_at": "2026-07-19T09:00:00+00:00"}]

    write_artifact_manifest(run_dir, status="partial")
    report = audit_result_bundle(run_dir)
    assert report["status"] == "partial"
    assert report["errors"] == []


def test_complete_bundle_with_a_resumed_session_audits_canonical_valid(
    tmp_path: Path,
    stable_provenance: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    kwargs = _resume_run_kwargs() | {"evaluation_seeds": [101, 102, 103], "evaluation_episodes": 3}
    initial_token = result_bundle_provenance._PROCESS_SESSION_TOKEN
    initialize_result_bundle(run_dir, **kwargs)
    monkeypatch.setattr(result_bundle_provenance, "_PROCESS_SESSION_TOKEN", "f" * 32)
    initialize_result_bundle(run_dir, **(kwargs | {"captured_at": "2026-07-19T09:00:00+00:00"}))

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
        run_id="resumed-run",
    )

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert [entry["session_token"] for entry in provenance["sessions"]] == [initial_token, "f" * 32]
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["provenance"]["sessions"] == provenance["sessions"]
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"


def test_fresh_process_reexport_of_a_complete_bundle_records_no_session(
    tmp_path: Path,
    stable_provenance: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished run is re-exported read-only, never "resumed".

    Appending a session entry here would stale the complete manifest's
    provenance hash and the summary's embedded copy, wedging the bundle as
    canonical-conflict on every subsequent save (Phase R integration
    finding).  The completion marker gates the append.
    """
    run_dir = tmp_path / "run"
    kwargs = _resume_run_kwargs() | {"evaluation_seeds": [101, 102, 103], "evaluation_episodes": 3}
    initialize_result_bundle(run_dir, **kwargs)
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    save_kwargs: dict[str, Any] = {
        "backend": "stable-baselines3",
        "backend_version": "2.7.0",
        "parallel_envs": 4,
        "evaluation_episodes": 3,
        "evaluation_seeds": [101, 102, 103],
        "plant_identity": _plant_identity(),
        "run_id": "resumed-run",
    }
    paths = save_result_bundle(stage_results, stage_configs, "velociraptor", "PPO", 42, run_dir, **save_kwargs)
    before = paths["provenance"].read_bytes()

    # A fresh runtime re-runs the storage cell (and even a full re-export) on
    # the already-complete run: identity validates, nothing is written.
    monkeypatch.setattr(result_bundle_provenance, "_PROCESS_SESSION_TOKEN", "f" * 32)
    second = initialize_result_bundle(run_dir, **(kwargs | {"captured_at": "2026-07-19T09:00:00+00:00"}))
    assert second.read_bytes() == before

    reexport = save_result_bundle(stage_results, stage_configs, "velociraptor", "PPO", 42, run_dir, **save_kwargs)
    assert reexport["provenance"].read_bytes() == before
    assert validate_result_bundle(run_dir)["status"] == "canonical-valid"


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


def test_repository_root_resolves_to_the_repository(tmp_path: Path) -> None:
    """The default provenance root must be the repository, not a subdirectory.

    Regression: this was `Path(__file__).resolve().parents[2]`, correct while it
    lived in `environments/shared/result_bundle.py` but off by one level once the
    module moved into the package — it resolved to `<repo>/environments`. The
    expression stayed byte-identical through the move, so an AST diff could not
    see it; only evaluating it can.
    """
    assert (result_bundle.constants.REPOSITORY_ROOT / "pyproject.toml").is_file()
    assert (result_bundle.constants.REPOSITORY_ROOT / "environments").is_dir()
    assert result_bundle.constants.REPOSITORY_ROOT.name != "environments"


def test_initializer_defaults_to_the_repository_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`initialize_result_bundle` must hand `_repository_state` the repository root.

    Pinning `constants.REPOSITORY_ROOT` alone is not enough: it fixes the value
    but not the wiring, so re-introducing the original
    `Path(__file__).resolve().parents[2]` inside the initializer left every other
    test in this module green. This spies on the seam the initializer actually
    uses.
    """
    seen: list[Path] = []

    def spy(root: Path) -> dict[str, Any]:
        seen.append(root)
        return {
            "repository_url": "https://github.com/kuds/mesozoic-labs.git",
            "repository_commit": _COMMIT,
            "repository_dirty": False,
            "repository_patch_sha256": None,
        }

    monkeypatch.setattr(result_bundle_provenance, "_repository_state", spy)
    initialize_result_bundle(
        tmp_path / "run",
        species="velociraptor",
        algorithm="PPO",
        backend="stable-baselines3",
        seed=42,
        evaluation_seeds=[11],
        evaluation_episodes=2,
        parallel_envs=4,
        plant_identity=_plant_identity(),
        run_id="default-root-run",
    )

    assert seen == [result_bundle.constants.REPOSITORY_ROOT]


def test_untracked_file_at_repository_root_enters_the_patch_hash(tmp_path: Path) -> None:
    """A root-level untracked file must change `repository_patch_sha256`.

    `git status` reports repo-wide, but `git ls-files --others` is scoped to its
    working directory. Pointed at a subdirectory, the state therefore reports the
    tree dirty while a file above that directory never enters the patch hash —
    two materially different dirty trees produce identical provenance.
    """
    repo = tmp_path / "repo"
    (repo / "environments" / "shared").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "environments" / "shared" / "mod.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    (repo / "environments" / "shared" / "untracked_below.txt").write_text("below\n")

    subdir = repo / "environments"

    def patch_hash(root: Path) -> Any:
        state = result_bundle_provenance._repository_state(root)
        assert state["repository_dirty"] is True
        return state["repository_patch_sha256"]

    root_before, subdir_before = patch_hash(repo), patch_hash(subdir)
    (repo / "root_level_untracked.txt").write_text("at the root\n")
    root_after, subdir_after = patch_hash(repo), patch_hash(subdir)

    assert root_after != root_before, "the repository root must notice a root-level untracked file"
    assert subdir_after == subdir_before, (
        "a too-deep root cannot see it — which is why REPOSITORY_ROOT must be the repository"
    )
