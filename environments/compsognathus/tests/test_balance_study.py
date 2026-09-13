"""Exercise the actual study workflow, schedules, artifacts and comparisons."""

import json
import shutil

import pytest

pytest.importorskip("stable_baselines3")

from environments.compsognathus.scripts.train_balance_study import (
    make_study_plan,
    prepare_study,
    summarize_study,
    train_balance_arm,
)


def test_plan_keeps_mechanics_and_training_matched_with_disjoint_panels():
    plan = make_study_plan()
    assert tuple(plan["arms"]) == ("A", "B", "C", "D")
    assert [arm["filter_hz"] for arm in plan["arms"].values()] == [0, 10, 0, 10]
    assert plan["arms"]["A"]["shaping"] == plan["arms"]["B"]["shaping"]
    assert plan["arms"]["C"]["shaping"] == plan["arms"]["D"]["shaping"]
    assert plan["training_budget"] == 11_000_000
    assert not set(plan["screen_seeds"]) & set(plan["confirmation_seeds"])
    assert len(plan["confirmation_seeds"]) == 40
    assert not set(plan["confirmation_seeds"]) & set(plan["probe_confirmation_seeds"])
    assert not set(plan["screen_seeds"]) & set(plan["probe_screen_seeds"])
    worker_seeds = [s for workers in plan["training_environment_seeds"].values() for s in workers]
    assert len(worker_seeds) == len(set(worker_seeds)) == 12


def test_prepared_plan_is_immutable_and_missing_runs_are_reported(tmp_path):
    prepare_study(tmp_path)
    assert prepare_study(tmp_path)["schema"]
    with pytest.raises(ValueError, match="differs"):
        prepare_study(tmp_path, training_seeds=(45,))
    summary = summarize_study(tmp_path)
    assert summary["expected_runs"] == 12
    assert not summary["all_runs_complete"]
    assert all(row["status"] == "not_started" for row in summary["runs"])
    assert not any(row["learned_balance_qualified"] for row in summary["runs"])


@pytest.mark.parametrize("seed", [1510, 1511, 1785, 1795, 2260, 2510])
def test_training_environments_cannot_reuse_evaluation_seeds(seed):
    with pytest.raises(ValueError, match="overlap"):
        make_study_plan(training_seeds=(seed,))


def test_source_edits_at_same_commit_cannot_mix_study_arms(tmp_path, monkeypatch):
    prepare_study(tmp_path)
    monkeypatch.setattr(
        "environments.compsognathus.scripts.train_balance_study.study_source_fingerprint", lambda: "changed-source"
    )
    with pytest.raises(ValueError, match="no longer matches"):
        summarize_study(tmp_path)


def test_real_ppo_update_checkpoint_reload_and_probe_schedule(tmp_path, monkeypatch):
    from environments.compsognathus.experiments.balance_env import CompsognathusBalanceEnv

    resets = []
    original_reset = CompsognathusBalanceEnv.reset

    def record_reset(self, **kwargs):
        resets.append(kwargs.get("seed"))
        return original_reset(self, **kwargs)

    monkeypatch.setattr(CompsognathusBalanceEnv, "reset", record_reset)
    prepare_study(tmp_path, training_seeds=(42,))
    result = train_balance_arm(tmp_path, "D", 42, probe_updates=1, evaluation_episodes=2)
    assert result["training_steps"] == 4096
    assert {168, 169, 170, 171}.issubset(resets)
    assert result["diagnostic_probe"]
    assert not result["learned_balance_qualified"]
    assert not result["production_advancement"]
    run = tmp_path / "runs" / "D_seed42_probe1"
    updates = json.loads((run / "updates.json").read_text())
    assert len(updates) == 1 and updates[0]["finite_parameters"]
    # A probe must not squeeze the 11M/7M schedules into a single update.
    assert updates[0]["learning_rate"] > 2.99e-5
    assert updates[0]["entropy_coefficient"] > 0.00499
    confirmation = json.loads((run / "confirmation.json").read_text())
    assert [row["seed"] for row in confirmation["episodes"]] == [10042, 10043]
    assert not confirmation["summary"]["behavior_qualified"]
    manifest = run / (result["selected_checkpoint"] + ".manifest.json")
    assert manifest.is_file()
    assert (run / "selected_trace.csv").is_file()
    # Probe results never fill a full-training slot in the comparison.
    assert all(row["status"] == "not_started" for row in summarize_study(tmp_path)["runs"])
    # Copying or renaming a probe into a full-run slot must not create a result.
    shutil.copytree(run, tmp_path / "runs" / "D_seed42")
    with pytest.raises(ValueError, match="does not match"):
        summarize_study(tmp_path)


def test_full_runs_cannot_shrink_the_confirmation_panel(tmp_path):
    prepare_study(tmp_path, training_seeds=(42,))
    with pytest.raises(ValueError, match="Only probes"):
        train_balance_arm(tmp_path, "A", 42, evaluation_episodes=4)
    assert not (tmp_path / "runs").exists()
