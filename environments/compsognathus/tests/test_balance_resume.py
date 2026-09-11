"""Checkpoint continuation preserves the declared study and schedule horizons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.compsognathus.experiments.balance_env import make_balance_env
from environments.compsognathus.experiments.balance_identity import (
    attach_study_identity,
    build_study_identity,
    load_study_checkpoint,
    save_study_checkpoint,
)
from environments.compsognathus.scripts import train_balance_study as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_can_restore_the_study_subclass_for_continuation(tmp_path):
    vector = DummyVecEnv([lambda: make_balance_env("D")])
    identity = build_study_identity(vector.envs[0], {"arm": "D", "seed": 42})
    normalizer = VecNormalize(vector)
    model = runner.BalanceStudyPPO(
        "MlpPolicy",
        normalizer,
        n_steps=8,
        batch_size=4,
        policy_kwargs={"net_arch": [8, 8]},
        seed=42,
        device="cpu",
        stop_after_steps=16,
    )
    attach_study_identity(model, identity)
    attach_study_identity(normalizer, identity)
    checkpoint = tmp_path / "step_0"
    save_study_checkpoint(model, normalizer, checkpoint, identity)
    fresh = DummyVecEnv([lambda: make_balance_env("D")])
    loaded_normalizer = None
    try:
        loaded, loaded_normalizer = load_study_checkpoint(
            checkpoint, identity, fresh, model_class=runner.BalanceStudyPPO
        )
        assert type(loaded) is runner.BalanceStudyPPO
        assert loaded.stop_after_steps == 16
        assert loaded.num_timesteps == 0
        assert loaded.get_env() is loaded_normalizer
        assert loaded._last_obs is None
        # A loader remains an evaluation operation. The continuation runner
        # must explicitly restore training flags after strict verification.
        assert not loaded_normalizer.training and not loaded_normalizer.norm_reward
    finally:
        (loaded_normalizer if loaded_normalizer is not None else fresh).close()
        normalizer.close()


def test_incomplete_checkpoint_files_are_preserved_before_prefix_reuse(tmp_path, monkeypatch):
    plan = runner.prepare_study(tmp_path, training_seeds=(42,))
    run = tmp_path / "runs" / "D_seed42_probe3"
    snapshot = {"plan": plan, "arm": "D", "seed": 42, "probe_updates": 3, "evaluation_episodes": 2}
    runner._save(run / "run_config.json", snapshot)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    partials = {"step_4096.model.zip": b"partial model", "step_4096.vecnormalize.pkl": b"partial normalizer"}
    for name, content in partials.items():
        (checkpoints / name).write_bytes(content)

    def stop_before_model_creation(*args, **kwargs):
        raise RuntimeError("stop test before training")

    monkeypatch.setattr(runner, "_prepare_alg_kwargs", stop_before_model_creation)
    with pytest.raises(RuntimeError, match="stop test before training"):
        runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True)
    assert not list(checkpoints.iterdir())
    quarantine = run / "incomplete_checkpoints"
    for name, content in partials.items():
        preserved = list(quarantine.rglob(name))
        assert len(preserved) == 1 and preserved[0].read_bytes() == content


def test_interrupted_completed_update_resumes_without_restarting_schedules(tmp_path, monkeypatch):
    original_plan = runner.make_study_plan

    def every_update_plan(**kwargs):
        plan = original_plan(**kwargs)
        plan["screen_every_steps"] = 4096
        return plan

    monkeypatch.setattr(runner, "make_study_plan", every_update_plan)
    plan = runner.prepare_study(tmp_path, training_seeds=(42,))
    original_save = runner._save
    interrupted = False

    def interrupt_after_committed_screen(path, value):
        nonlocal interrupted
        original_save(path, value)
        if str(path).endswith(".screen.json") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("simulated disconnect after completed checkpoint and screen")

    monkeypatch.setattr(runner, "_save", interrupt_after_committed_screen)
    with pytest.raises(KeyboardInterrupt, match="simulated disconnect"):
        runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2)
    assert interrupted
    run = tmp_path / "runs" / "D_seed42_probe3"
    first_checkpoint = run / "checkpoints" / "step_4096"
    protected_paths = [
        Path(str(first_checkpoint) + suffix) for suffix in (".model.zip", ".vecnormalize.pkl", ".manifest.json")
    ]
    protected_hashes = {path: _sha256(path) for path in protected_paths}
    original_logs = {path: path.read_bytes() for path in (run / "training_logs").rglob("progress.csv")}
    assert original_logs

    monkeypatch.setattr(runner, "_save", original_save)
    result = runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True)
    assert result["training_steps"] == 3 * 4096
    assert result["diagnostic_probe"]
    assert not result["learned_balance_qualified"]
    assert not result["production_advancement"]
    assert {path: _sha256(path) for path in protected_paths} == protected_hashes
    assert all(path.read_bytes() == content for path, content in original_logs.items())
    assert len(list((run / "training_logs").rglob("progress.csv"))) >= 2

    updates = json.loads((run / "updates.json").read_text())
    final = updates[-1]
    assert final["timesteps"] == 3 * 4096
    assert final["finite_parameters"]
    ppo = plan["stage_config"]["ppo_kwargs"]
    expected_lr = ppo["learning_rate"] + (final["timesteps"] / plan["training_budget"]) * (
        ppo["learning_rate_end"] - ppo["learning_rate"]
    )
    expected_entropy = ppo["ent_coef"] + (final["timesteps"] / ppo["ent_coef_decay_timesteps"]) * (
        ppo["ent_coef_end"] - ppo["ent_coef"]
    )
    assert final["learning_rate"] == pytest.approx(expected_lr, rel=1e-10)
    assert final["entropy_coefficient"] == pytest.approx(expected_entropy, rel=1e-10)
    screens = json.loads((run / "screening_history.json").read_text())
    assert [row["timesteps"] for row in screens] == [4096, 8192, 12288]
    # Selection is reconstructed from all completed screens, keeping the
    # earlier checkpoint when physical scores tie.
    selected_row = max(screens, key=lambda row: tuple(row["selection_key"]))
    assert Path(result["selected_checkpoint"]).name == selected_row["checkpoint"]
    resume_path = run / "resume_history.json"
    segments = json.loads(resume_path.read_text())
    assert len(segments) == 2
    assert segments[0]["segment_id"] == 0 and segments[0]["status"] == "interrupted"
    assert segments[1]["segment_id"] == 1 and segments[1]["status"] == "completed"
    assert segments[1]["from_steps"] == 4096
    assert segments[1]["to_steps"] == 12288
    assert Path(segments[1]["from_checkpoint"]).name == "step_4096"
    assert all(segment["simulator_reset"] is True for segment in segments)
    assert all(len(segment["worker_seeds"]) == 4 for segment in segments)

    summary_bytes = (run / "run_summary.json").read_bytes()
    confirmation_bytes = (run / "confirmation.json").read_bytes()
    resume_bytes = resume_path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("a completed resumed run must not train or evaluate again")

    monkeypatch.setattr(runner.BalanceStudyPPO, "learn", forbidden)
    monkeypatch.setattr(runner, "_evaluate", forbidden)
    repeated = runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True)
    assert repeated["training_steps"] == result["training_steps"]
    assert (run / "run_summary.json").read_bytes() == summary_bytes
    assert (run / "confirmation.json").read_bytes() == confirmation_bytes
    assert resume_path.read_bytes() == resume_bytes
    assert {path: _sha256(path) for path in protected_paths} == protected_hashes
    with pytest.raises(ValueError, match="config|snapshot|match"):
        runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=3, resume=True)

    # Matching cached summaries are insufficient: recompute from the actual
    # episode rows so coordinated corruption cannot manufacture a result.
    tampered_confirmation = json.loads(confirmation_bytes)
    tampered_summary = json.loads(summary_bytes)
    tampered_confirmation["summary"]["mean_study_reward"] += 1000
    tampered_summary["confirmation"] = tampered_confirmation["summary"]
    try:
        (run / "confirmation.json").write_text(json.dumps(tampered_confirmation))
        (run / "run_summary.json").write_text(json.dumps(tampered_summary))
        with pytest.raises(ValueError, match="panel|summary|episode|evidence"):
            runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True)
    finally:
        (run / "confirmation.json").write_bytes(confirmation_bytes)
        (run / "run_summary.json").write_bytes(summary_bytes)


@pytest.mark.parametrize("max_seconds", [0.0, -1.0, float("inf"), float("-inf"), float("nan"), True, False])
def test_invalid_session_deadline_is_rejected_before_starting_a_run(tmp_path, max_seconds):
    runner.prepare_study(tmp_path, training_seeds=(42,))
    with pytest.raises(ValueError, match="max_seconds|positive|finite"):
        runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, max_seconds=max_seconds)
    assert not (tmp_path / "runs").exists()


def test_session_deadline_pauses_at_an_update_and_resumes_full_schedules(tmp_path, monkeypatch):
    plan = runner.prepare_study(tmp_path, training_seeds=(42,))
    assert plan["screen_every_steps"] > 3 * 4096
    real_time = runner.time
    original_train = runner.BalanceStudyPPO.train
    original_load = runner.load_study_checkpoint
    original_evaluate = runner._evaluate
    clock = {"now": 0.0}
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))

    def expire_during_first_update(self):
        # Collection and the before-learn check see time zero. The first
        # completed update's callback sees an expired soft session budget.
        clock["now"] = 1.0
        return original_train(self)

    monkeypatch.setattr(runner.BalanceStudyPPO, "train", expire_during_first_update)
    paused = runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, max_seconds=0.001)
    assert paused["status"] == "paused"
    assert paused["training_steps"] == 4096
    assert paused["planned_training_steps"] == 12288
    assert not paused.get("learned_balance_qualified", False)
    run = tmp_path / "runs" / "D_seed42_probe3"
    first_checkpoint = run / paused["checkpoint"]
    assert first_checkpoint.parent.name == "continuations"
    protected_paths = [
        Path(str(first_checkpoint) + suffix) for suffix in (".model.zip", ".vecnormalize.pkl", ".manifest.json")
    ]
    protected_hashes = {path: _sha256(path) for path in protected_paths}
    assert not first_checkpoint.with_suffix(".screen.json").exists()
    assert not list((run / "checkpoints").glob("*.screen.json"))
    assert not (run / "run_summary.json").exists()
    assert not (run / "confirmation.json").exists()
    assert json.loads((run / "resume_history.json").read_text())[-1]["status"] == "paused"

    # A later session can spend its allowance verifying/restoring a cached
    # checkpoint. It must not start another rollout or re-score cached screens.
    clock["now"] = 0.0

    def expire_after_restore(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        if kwargs.get("model_class") is runner.BalanceStudyPPO:
            clock["now"] = 1.0
        return loaded

    def forbidden(*args, **kwargs):
        pytest.fail("no new training or inference is allowed for this cached/deadline result")

    monkeypatch.setattr(runner, "load_study_checkpoint", expire_after_restore)
    monkeypatch.setattr(runner.BalanceStudyPPO, "train", forbidden)
    monkeypatch.setattr(runner, "_evaluate", forbidden)
    cached_pause = runner.train_balance_arm(
        tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True, max_seconds=0.001
    )
    assert cached_pause["status"] == "paused"
    assert cached_pause["training_steps"] == 4096
    assert not cached_pause.get("learned_balance_qualified", False)
    assert {path: _sha256(path) for path in protected_paths} == protected_hashes
    assert not (run / "run_summary.json").exists()
    assert not (run / "confirmation.json").exists()

    monkeypatch.setattr(runner, "time", real_time)
    monkeypatch.setattr(runner, "load_study_checkpoint", original_load)
    monkeypatch.setattr(runner.BalanceStudyPPO, "train", original_train)
    monkeypatch.setattr(runner, "_evaluate", original_evaluate)
    result = runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True)
    assert result["training_steps"] == 12288
    assert result["diagnostic_probe"] and not result["learned_balance_qualified"]
    assert {path: _sha256(path) for path in protected_paths} == protected_hashes
    updates = json.loads((run / "updates.json").read_text())
    assert [row["timesteps"] for row in updates] == [4096, 8192, 12288]
    final = updates[-1]
    ppo = plan["stage_config"]["ppo_kwargs"]
    expected_lr = ppo["learning_rate"] + (12288 / plan["training_budget"]) * (
        ppo["learning_rate_end"] - ppo["learning_rate"]
    )
    expected_entropy = ppo["ent_coef"] + (12288 / ppo["ent_coef_decay_timesteps"]) * (
        ppo["ent_coef_end"] - ppo["ent_coef"]
    )
    assert final["learning_rate"] == pytest.approx(expected_lr, rel=1e-10)
    assert final["entropy_coefficient"] == pytest.approx(expected_entropy, rel=1e-10)
    assert (run / "confirmation.json").is_file()
    summary_bytes = (run / "run_summary.json").read_bytes()
    history_bytes = (run / "resume_history.json").read_bytes()
    monkeypatch.setattr(runner.BalanceStudyPPO, "learn", forbidden)
    monkeypatch.setattr(runner, "_evaluate", forbidden)
    completed = runner.train_balance_arm(
        tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True, max_seconds=1e-12
    )
    assert completed["training_steps"] == 12288
    assert (run / "run_summary.json").read_bytes() == summary_bytes
    assert (run / "resume_history.json").read_bytes() == history_bytes

    # Inspecting a completed run is read-only, including unrelated incomplete
    # files left by an older interrupted attempt.
    orphan = run / "checkpoints" / "step_8192.model.zip"
    orphan.write_bytes(b"uncommitted checkpoint fragment")
    runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True)
    assert orphan.read_bytes() == b"uncommitted checkpoint fragment"
    assert not (run / "incomplete_checkpoints").exists()

    identity_path = run / "study_identity.json"
    identity_bytes = identity_path.read_bytes()
    identity_path.unlink()
    before_missing_identity_check = {path.relative_to(run): _sha256(path) for path in run.rglob("*") if path.is_file()}
    try:
        with pytest.raises(ValueError, match="missing.*identity"):
            runner.train_balance_arm(tmp_path, "D", 42, probe_updates=3, evaluation_episodes=2, resume=True)
        assert not identity_path.exists()
        assert {path.relative_to(run): _sha256(path) for path in run.rglob("*") if path.is_file()} == (
            before_missing_identity_check
        )
    finally:
        identity_path.write_bytes(identity_bytes)
