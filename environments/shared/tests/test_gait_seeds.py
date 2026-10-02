"""Safe metadata corroboration and held-out gait seed exclusions."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from environments.shared.gait.seeds import checkpoint_seed_provenance
from environments.shared.result_bundle.hashing import canonical_json_sha256


def _checkpoint(root: Path, *, run_seed=42, run_envs=4, model_seed=42, model_envs=4, models=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage_config.json").write_text(json.dumps({"run": {"seed": run_seed, "n_envs": run_envs}}))
    directory = root / "models" if models else root
    directory.mkdir(exist_ok=True)
    path = directory / "robust_best_model.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"seed": model_seed, "n_envs": model_envs}))
        # A dangerous pickle member must never be read/deserialized by this
        # metadata-only check. It need not even be a valid pickle.
        archive.writestr("policy.pth", b"not executable JSON metadata")
    return path


@pytest.mark.parametrize("models", [True, False])
def test_seed_binding_is_portable_and_ignores_unrelated_config_changes(tmp_path, models):
    model = _checkpoint(tmp_path / "source", models=models)
    evidence = checkpoint_seed_provenance(model, seed_start=3042, episodes=40)
    assert evidence["training_seed"] == 42 and evidence["training_envs"] == 4
    assert evidence["selection_seed"] == 1042
    payload = {key: value for key, value in evidence.items() if key != "seed_provenance_sha256"}
    assert evidence["seed_provenance_sha256"] == canonical_json_sha256(payload)
    shutil.copytree(tmp_path / "source", tmp_path / "copied")
    copied = tmp_path / "copied" / model.relative_to(tmp_path / "source")
    assert checkpoint_seed_provenance(copied, seed_start=3042, episodes=40) == evidence
    config = tmp_path / "copied" / "stage_config.json"
    value = json.loads(config.read_text())
    value["duration_seconds"] = 123
    config.write_text(json.dumps(value, indent=4))
    assert checkpoint_seed_provenance(copied, seed_start=3042, episodes=40) == evidence
    assert str(tmp_path) not in json.dumps(evidence)


@pytest.mark.parametrize(
    "updates",
    [
        {"run_seed": None},
        {"run_seed": True},
        {"run_seed": -1},
        {"run_seed": 42.0},
        {"run_envs": 0},
        {"run_envs": False},
        {"run_envs": 4.0},
        {"model_seed": None},
        {"model_seed": True},
        {"model_seed": -1},
        {"model_seed": 43},
        {"model_envs": 0},
        {"model_envs": False},
        {"model_envs": 5},
    ],
)
def test_missing_invalid_or_mismatched_seed_metadata_refuses(tmp_path, updates):
    model = _checkpoint(tmp_path, **updates)
    with pytest.raises(ValueError, match="seed provenance"):
        checkpoint_seed_provenance(model, seed_start=3042, episodes=40)


@pytest.mark.parametrize("seed_start,episodes", [(True, 40), (-1, 40), (3042, False), (3042, 0), (3042, 1.5)])
def test_invalid_panel_refuses(tmp_path, seed_start, episodes):
    with pytest.raises(ValueError, match="seed provenance"):
        checkpoint_seed_provenance(_checkpoint(tmp_path), seed_start=seed_start, episodes=episodes)


@pytest.mark.parametrize("seed_start,episodes", [(42, 1), (45, 1), (40, 3), (1042, 1), (1040, 3)])
def test_full_panel_training_and_selection_intersections_refuse(tmp_path, seed_start, episodes):
    with pytest.raises(ValueError, match="overlaps"):
        checkpoint_seed_provenance(_checkpoint(tmp_path), seed_start=seed_start, episodes=episodes)


@pytest.mark.parametrize("seed_start,episodes", [(40, 2), (46, 40), (1002, 40), (1043, 40)])
def test_disjoint_boundary_panels_are_allowed(tmp_path, seed_start, episodes):
    assert checkpoint_seed_provenance(_checkpoint(tmp_path), seed_start=seed_start, episodes=episodes)


def test_missing_config_or_archive_metadata_refuses(tmp_path):
    model = _checkpoint(tmp_path)
    (tmp_path / "stage_config.json").unlink()
    with pytest.raises(ValueError, match="stage_config"):
        checkpoint_seed_provenance(model, seed_start=3042, episodes=40)
    model = _checkpoint(tmp_path)
    with zipfile.ZipFile(model, "w") as archive:
        archive.writestr("policy.pth", b"not JSON")
    with pytest.raises(ValueError, match="exactly one"):
        checkpoint_seed_provenance(model, seed_start=3042, episodes=40)
    model.write_bytes(b"not a ZIP")
    with pytest.raises(ValueError, match="checkpoint JSON"):
        checkpoint_seed_provenance(model, seed_start=3042, episodes=40)


def test_optional_seed_roles_bind_known_development_exclusions(tmp_path):
    model = _checkpoint(tmp_path)
    provenance = tmp_path / "provenance.json"
    roles = {
        "training": 42,
        "checkpoint_selection_evaluation": 1042,
        "publication_evaluation": 3042,
        "certification_panel": 3042,
        "gait_development": 104042,
    }
    provenance.write_text(json.dumps({"seed_roles": roles}))
    evidence = checkpoint_seed_provenance(model, seed_start=3042, episodes=40)
    assert "gait_development" in evidence["additional_seed_roles"]
    assert "publication_evaluation" not in evidence["additional_seed_roles"]
    roles["gait_development"] = 3070
    provenance.write_text(json.dumps({"seed_roles": roles}))
    with pytest.raises(ValueError, match="gait_development"):
        checkpoint_seed_provenance(model, seed_start=3042, episodes=40)
    roles["gait_development"] = 104042
    roles["training"] = 43
    provenance.write_text(json.dumps({"seed_roles": roles}))
    with pytest.raises(ValueError, match="training role"):
        checkpoint_seed_provenance(model, seed_start=3042, episodes=40)


def test_later_ordinary_provenance_repeats_existing_exclusions_without_changing_binding(tmp_path):
    model = _checkpoint(tmp_path)
    original = checkpoint_seed_provenance(model, seed_start=3042, episodes=40)
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "seed_roles": {
                    "training": 42,
                    "checkpoint_selection_evaluation": 1042,
                    "publication_evaluation": 3042,
                    "certification_panel": 3042,
                }
            }
        )
    )
    assert checkpoint_seed_provenance(model, seed_start=3042, episodes=40) == original
