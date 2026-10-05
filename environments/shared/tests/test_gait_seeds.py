"""Safe metadata corroboration and held-out gait seed exclusions."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from environments.shared.gait.seeds import checkpoint_seed_provenance, stage_replay_seeds
from environments.shared.result_bundle.hashing import canonical_json_sha256


def _provenance(model, **panel):
    """The trex locomotion panel's binding (legacy stage 2, id ``locomotion``)."""
    return checkpoint_seed_provenance(model, stage=2, species="trex", **panel)


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
    evidence = _provenance(model, seed_start=3042, episodes=40)
    assert evidence["training_seed"] == 42 and evidence["training_envs"] == 4
    assert evidence["selection_seed"] == 1042
    # replay_seed(42, 2) and replay_seed(42, "locomotion"): both spellings of the stage
    assert evidence["replay_seeds"] == [2044, 2236]
    payload = {key: value for key, value in evidence.items() if key != "seed_provenance_sha256"}
    assert evidence["seed_provenance_sha256"] == canonical_json_sha256(payload)
    shutil.copytree(tmp_path / "source", tmp_path / "copied")
    copied = tmp_path / "copied" / model.relative_to(tmp_path / "source")
    assert _provenance(copied, seed_start=3042, episodes=40) == evidence
    config = tmp_path / "copied" / "stage_config.json"
    value = json.loads(config.read_text())
    value["duration_seconds"] = 123
    config.write_text(json.dumps(value, indent=4))
    assert _provenance(copied, seed_start=3042, episodes=40) == evidence
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
        _provenance(model, seed_start=3042, episodes=40)


@pytest.mark.parametrize("seed_start,episodes", [(True, 40), (-1, 40), (3042, False), (3042, 0), (3042, 1.5)])
def test_invalid_panel_refuses(tmp_path, seed_start, episodes):
    with pytest.raises(ValueError, match="seed provenance"):
        _provenance(_checkpoint(tmp_path), seed_start=seed_start, episodes=episodes)


@pytest.mark.parametrize("seed_start,episodes", [(42, 1), (45, 1), (40, 3), (1042, 1), (1040, 3)])
def test_full_panel_training_and_selection_intersections_refuse(tmp_path, seed_start, episodes):
    with pytest.raises(ValueError, match="overlaps"):
        _provenance(_checkpoint(tmp_path), seed_start=seed_start, episodes=episodes)


@pytest.mark.parametrize("seed_start,episodes", [(40, 2), (46, 40), (1002, 40), (1043, 40)])
def test_disjoint_boundary_panels_are_allowed(tmp_path, seed_start, episodes):
    assert _provenance(_checkpoint(tmp_path), seed_start=seed_start, episodes=episodes)


def test_missing_config_or_archive_metadata_refuses(tmp_path):
    model = _checkpoint(tmp_path)
    (tmp_path / "stage_config.json").unlink()
    with pytest.raises(ValueError, match="stage_config"):
        _provenance(model, seed_start=3042, episodes=40)
    model = _checkpoint(tmp_path)
    with zipfile.ZipFile(model, "w") as archive:
        archive.writestr("policy.pth", b"not JSON")
    with pytest.raises(ValueError, match="exactly one"):
        _provenance(model, seed_start=3042, episodes=40)
    model.write_bytes(b"not a ZIP")
    with pytest.raises(ValueError, match="checkpoint JSON"):
        _provenance(model, seed_start=3042, episodes=40)


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
    evidence = _provenance(model, seed_start=3042, episodes=40)
    assert "gait_development" in evidence["additional_seed_roles"]
    assert "publication_evaluation" not in evidence["additional_seed_roles"]
    roles["gait_development"] = 3070
    provenance.write_text(json.dumps({"seed_roles": roles}))
    with pytest.raises(ValueError, match="gait_development"):
        _provenance(model, seed_start=3042, episodes=40)
    roles["gait_development"] = 104042
    roles["training"] = 43
    provenance.write_text(json.dumps({"seed_roles": roles}))
    with pytest.raises(ValueError, match="training role"):
        _provenance(model, seed_start=3042, episodes=40)


def test_later_ordinary_provenance_repeats_existing_exclusions_without_changing_binding(tmp_path):
    model = _checkpoint(tmp_path)
    original = _provenance(model, seed_start=3042, episodes=40)
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
    assert _provenance(model, seed_start=3042, episodes=40) == original


@pytest.mark.parametrize(
    "run_seed,stage",
    [
        (1040, 2),  # replay_seed(1040, 2) = 3042: the stage video is certification episode 0
        (848, "locomotion"),  # replay_seed(848, "locomotion") = 848 + 2000 + 194 = 3042
        (848, 2),  # the same stage spelled by its legacy number: every spelling is excluded
        (1079, 2),  # 3081, the panel's last seed
    ],
)
def test_stage_replay_video_seeds_are_excluded_whatever_the_stage_spelling(tmp_path, run_seed, stage):
    model = _checkpoint(tmp_path, run_seed=run_seed, model_seed=run_seed)
    assert 3042 <= max(seed for seed in stage_replay_seeds(run_seed, stage, species="trex") if seed < 3082)
    with pytest.raises(ValueError, match="replay video seed"):
        checkpoint_seed_provenance(model, seed_start=3042, episodes=40, stage=stage, species="trex")


def test_replay_seed_just_outside_the_panel_is_allowed(tmp_path):
    model = _checkpoint(tmp_path, run_seed=1080, model_seed=1080)  # replay 3082 and 3274
    assert checkpoint_seed_provenance(model, seed_start=3042, episodes=40, stage=2, species="trex")


def _with_roles(tmp_path, roles):
    model = _checkpoint(tmp_path)
    (tmp_path / "provenance.json").write_text(json.dumps({"seed_roles": {"training": 42, **roles}}))
    return model


@pytest.mark.parametrize(
    "roles",
    [
        {"development_panel": {"start": 3000, "episodes": 100}},  # a panel recorded as its whole block
        {"development_panel": {"start": 3081, "episodes": 1}},
        {"threshold_tuning": 3042},  # an unrecognised role is a used seed, whatever its name
        {"evaluation": 3050},
        {"gait_report_viewed": {"start": 3070, "episodes": 5}},
    ],
)
def test_range_valued_and_unknown_roles_are_used_seeds(tmp_path, roles):
    with pytest.raises(ValueError, match="overlaps the recorded"):
        _provenance(_with_roles(tmp_path, roles), seed_start=3042, episodes=40)


@pytest.mark.parametrize(
    "roles",
    [
        {"development_panel": {"start": 2900, "episodes": 142}},  # 2900-3041, adjacent
        {"development_panel": {"start": 3082, "episodes": 10}},
        {"publication_evaluation": 3042, "certification_panel": 3042, "additional_evaluation_2": 3050},
        {"gait_development": {"start": 9000, "episodes": 10}},
    ],
)
def test_disjoint_blocks_and_publication_roles_are_allowed(tmp_path, roles):
    evidence = _provenance(_with_roles(tmp_path, roles), seed_start=3042, episodes=40)
    for role in ("publication_evaluation", "certification_panel", "additional_evaluation_2"):
        assert role not in evidence["additional_seed_roles"]
    if "gait_development" in roles:
        assert evidence["additional_seed_roles"]["gait_development"] == {"start": 9000, "episodes": 10}


@pytest.mark.parametrize(
    "value",
    [
        {"start": 3000},
        {"start": 3000, "episodes": 0},
        {"start": -1, "episodes": 2},
        {"start": 1.5, "episodes": 2},
        "3042",
    ],
)
def test_malformed_role_blocks_refuse(tmp_path, value):
    with pytest.raises(ValueError, match="seed provenance"):
        _provenance(_with_roles(tmp_path, {"development_panel": value}), seed_start=3042, episodes=40)


def test_stage_reference_is_required(tmp_path):
    with pytest.raises(ValueError, match="stage reference"):
        checkpoint_seed_provenance(_checkpoint(tmp_path), seed_start=3042, episodes=40, stage=None)  # type: ignore[arg-type]
