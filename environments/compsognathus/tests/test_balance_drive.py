"""Legacy Drive sidecars, exact pair binding and canonical replay refusal."""

import json
from pathlib import Path

import pytest

from environments.compsognathus.experiments.balance_drive import (
    discover_balance_runs,
    evaluate_saved_baseline,
    save_historical_baselines,
)
from environments.shared.plant_contract import PlantCompatibilityError


@pytest.fixture
def saved_run(tmp_path):
    stage = tmp_path / "logs/compsognathus/ppo/20260909_162812/01_stance"
    models = stage / "models"
    models.mkdir(parents=True)
    (models / "robust_best_model.zip").write_bytes(b"not deserialized in discovery")
    (models / "robust_best_model_vecnorm.pkl").write_bytes(b"paired normalizer")
    (stage / "stage_config.json").write_text(
        json.dumps({"species": "compsognathus", "stage": 1, "reward_weights": {}, "curriculum": {}})
    )
    prefix = "/content/drive/MyDrive/mesozoic-labs/logs/compsognathus/ppo/20260909_162812/01_stance/models/"
    (stage / "stage_summary.txt").write_text(
        "Best model:     " + prefix + "robust_best_model.zip\n"
        "VecNormalize:   " + prefix + "robust_best_model_vecnorm.pkl\n"
    )
    (stage / "stance_gate_report.txt").write_text(
        "panel               40 episodes, seeds 3042-3081\n"
        "reward                    2801.6 +/- 51.2\n"
        "full_horizon_fraction     1.0000   (>= 0.9500)\n"
        "mean_unsupported_duty     0.0131   (<= 0.0200)\n"
        "  bilateral support       0.0140   (statue 0.998, not gated)\n"
        "GATE: PASS\n"
    )
    return tmp_path, stage


def test_reads_existing_layout_and_explicit_pair_without_loading_policy(saved_run, tmp_path):
    root, stage = saved_run
    # A giant source bundle is irrelevant to report discovery and is not read.
    (stage.parent / "source_bundle.json").write_text("invalid JSON deliberately ignored")
    runs = discover_balance_runs(root)
    assert len(runs) == 1
    row = runs[0]
    assert row["status"] == "available"
    assert row["saved_metrics"]["bilateral_control_window_support"] == 0.014
    assert row["saved_metrics"]["n_episodes"] == 40
    assert row["checkpoint_pair"]["model_path"] == str(stage / "models/robust_best_model.zip")
    assert row["checkpoint_pair"]["normalization_path"].endswith("robust_best_model_vecnorm.pkl")
    report = save_historical_baselines(runs, tmp_path / "new-study/references.json")
    assert report["historical_reference_only"]


def test_missing_pair_does_not_guess_another_normalizer(saved_run):
    root, stage = saved_run
    (stage / "models/robust_best_model_vecnorm.pkl").unlink()
    (stage / "models/vec_normalize.pkl").write_bytes(b"wrong candidate")
    assert discover_balance_runs(root)[0]["checkpoint_pair"] is None


def test_declared_pair_cannot_escape_selected_models_directory(saved_run):
    root, stage = saved_run
    (stage / "stage_summary.txt").write_text("Best model: /tmp/other.zip\nVecNormalize: /tmp/other.pkl\n")
    assert discover_balance_runs(root)[0]["checkpoint_pair"] is None


@pytest.mark.parametrize("changed", ["model", "config", "declaration"])
def test_changed_artifact_is_rejected_before_pickle_load(saved_run, monkeypatch, changed):
    root, stage = saved_run
    pair = discover_balance_runs(root)[0]["checkpoint_pair"]
    path = {
        "model": Path(pair["model_path"]),
        "config": stage / "stage_config.json",
        "declaration": stage / "stage_summary.txt",
    }[changed]
    path.write_bytes(path.read_bytes() + b" \n")

    def no_load(*args):
        pytest.fail("Changed artifacts must be rejected before model deserialization")

    monkeypatch.setattr("environments.compsognathus.scripts.probe_feet._load_policy", no_load)
    with pytest.raises(ValueError):
        evaluate_saved_baseline(pair, root / "new-study/replay", seeds=(11042,))


def test_canonical_failure_stays_visible_and_preserves_originals(saved_run, monkeypatch):
    root, stage = saved_run
    pair = discover_balance_runs(root)[0]["checkpoint_pair"]
    before = {p: p.read_bytes() for p in stage.rglob("*") if p.is_file()}

    def incompatible(*args):
        raise PlantCompatibilityError("canonical plant differs")

    monkeypatch.setattr("environments.compsognathus.scripts.probe_feet._load_policy", incompatible)
    destination = root / "new-study/replay"
    with pytest.raises(PlantCompatibilityError, match="canonical plant differs"):
        evaluate_saved_baseline(pair, destination, seeds=(11042,))
    assert "canonical plant differs" in (destination / "failure.json").read_text()
    assert all(p.read_bytes() == data for p, data in before.items())


def test_replay_cannot_write_inside_original_run(saved_run):
    root, stage = saved_run
    pair = discover_balance_runs(root)[0]["checkpoint_pair"]
    with pytest.raises(ValueError, match="existing run"):
        evaluate_saved_baseline(pair, stage / "new-replay", seeds=(11042,))
