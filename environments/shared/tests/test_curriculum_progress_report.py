"""Tests for environments.shared.curriculum.progress_report: ``progress.json``, report-only.

The record is built from what the evaluation callbacks already hold (the
evaluation history and the latest ``gate_progress.npz`` row) and rewritten
atomically.  Last, a short real T. rex stance training run on CPU writes it,
and the same run with every addition of this change switched off (the file,
the posture watch, the roll rails and the termination counts) trains and
evaluates exactly the same: the same ``evaluations.npz``, the same
``gate_progress.npz`` values on the keys both write, and the same checkpoints.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from environments.shared.curriculum import progress_report
from environments.shared.curriculum.progress_report import (
    HISTORY_EVALUATIONS,
    PROGRESS_FILENAME,
    PROGRESS_SCHEMA,
    TrainingProgressCallback,
    posture_block,
    termination_counts,
)
from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
NAN = float("nan")

#: A T. rex gate_progress.npz row: both fore-aft rails (the pad's bar declared), both roll rails, a leg DC pair.
ROW = {
    "timesteps": 300.0,
    "full_horizon_fraction": 0.75,
    "mean_reward": 100.0,
    "r_sole_cop_fore_aft": 0.6,
    "r_sole_cop_fore_aft_gate": 0.65,
    "r_digit_share": 0.7,
    "r_foot_cop_fore_aft": 1.12,
    "r_foot_cop_fore_aft_gate": 1.12,
    "r_toe_margin_nm": -4.0,
    "r_heel_margin_nm": 60.0,
    "l_sole_cop_fore_aft": -0.2,
    "l_sole_cop_fore_aft_gate": 0.3,
    "l_digit_share": NAN,
    "l_foot_cop_fore_aft": 0.4,
    "l_foot_cop_fore_aft_gate": 0.45,
    "l_toe_margin_nm": 20.0,
    "l_heel_margin_nm": 30.0,
    "max_sole_cop_fore_aft_mean": 0.66,
    "max_sole_cop_fore_aft_over_bar_fraction": 0.25,
    "max_foot_cop_fore_aft_mean": 1.12,
    "r_sole_tilt_deg": 1.0,
    "r_sole_corner_lift_m": 0.002,
    "l_sole_tilt_deg": 2.5,
    "l_sole_corner_lift_m": 0.001,
    "max_sole_tilt_deg_mean": 2.5,
    "max_sole_tilt_deg_over_bar_fraction": 1.0,
    "max_sole_corner_lift_m_mean": 0.002,
    "max_sole_corner_lift_m_over_bar_fraction": 0.0,
    "leg_dc_l_minus_r_ankle_deg": -3.165,
    "termination_truncated": 3.0,
    "termination_nosedive": 1.0,
    "termination_head_contact": 0.0,
}
CURRICULUM = {"max_sole_cop_fore_aft": 0.8, "max_sole_tilt_deg": 2.0, "max_sole_corner_lift_m": 0.004}


class TestTheBlocks:
    def test_the_posture_block_reports_each_rail_against_its_declared_bar_per_foot(self):
        block = posture_block(ROW, CURRICULUM)
        assert list(block) == [
            "max_sole_cop_fore_aft",
            "max_foot_cop_fore_aft",
            "max_sole_tilt_deg",
            "max_sole_corner_lift_m",
            "leg_dc_deg",
        ]
        assert block["max_sole_cop_fore_aft"] == {
            "bar": 0.8,
            "mean": 0.66,
            "over_bar_fraction": 0.25,
            "per_foot": {
                "r": {"gate": 0.65, "sole_cop_fore_aft": 0.6, "digit_share": 0.7},
                "l": {"gate": 0.3, "sole_cop_fore_aft": -0.2, "digit_share": None},
            },
        }
        # No bar declared for the whole foot: no fraction, and the bar reads null.
        whole = block["max_foot_cop_fore_aft"]
        assert whole["bar"] is None and "over_bar_fraction" not in whole
        assert whole["per_foot"]["r"] == {
            "gate": 1.12,
            "foot_cop_fore_aft": 1.12,
            "toe_margin_nm": -4.0,
            "heel_margin_nm": 60.0,
        }
        assert block["max_sole_tilt_deg"] == {
            "bar": 2.0,
            "mean": 2.5,
            "over_bar_fraction": 1.0,
            "per_foot": {"r": 1.0, "l": 2.5},
        }
        assert block["leg_dc_deg"] == {"l_minus_r_ankle": -3.165}

    def test_a_model_without_a_rail_has_no_entry_for_it(self):
        row = {"timesteps": 1.0, "leg_dc_l_minus_r_knee_deg": 0.0}
        assert posture_block(row, CURRICULUM) == {"leg_dc_deg": {"l_minus_r_knee": 0.0}}
        assert posture_block({"timesteps": 1.0}, {}) == {}

    def test_the_termination_counts_leave_out_the_causes_no_episode_ended_by(self):
        assert termination_counts(ROW) == {"truncated": 3, "nosedive": 1}
        assert termination_counts({"termination_truncated": NAN}) == {}


# ── the record, on stand-in callbacks ─────────────────────────────────────────


def _eval_callback(evaluations: int, *, episodes: int = 4) -> SimpleNamespace:
    rewards = [[100.0 + index + episode for episode in range(episodes)] for index in range(evaluations)]
    lengths = [[1000] * (episodes - 1) + [400] for _ in range(evaluations)]
    return SimpleNamespace(
        evaluations_timesteps=[100 * (index + 1) for index in range(evaluations)],
        evaluations_results=rewards,
        evaluations_length=lengths,
        best_mean_reward=max(float(np.mean(row)) for row in rewards) if rewards else -math.inf,
    )


class _Plateau:
    horizon = 1000

    def __init__(self, rows: list[dict[str, float]]) -> None:
        self.rows = rows

    def gate_progress_row(self) -> dict[str, float]:
        return dict(self.rows[-1]) if self.rows else {}

    def gate_progress_series(self, key: str) -> tuple[list[int], list[float]]:
        return [int(row["timesteps"]) for row in self.rows], [row.get(key, NAN) for row in self.rows]


def _callback(tmp_path: Path, *, evaluations: int = 3, model_id: ModelId | None = TYRANNOSAURUS_REX):
    """A writer built without ``__init__`` (no stable-baselines3), as of its third evaluation by default."""
    stage_dir = tmp_path / "run" / "01_stance"
    (stage_dir / "models").mkdir(parents=True)
    callback = object.__new__(TrainingProgressCallback)
    callback.eval_callback = _eval_callback(evaluations)
    rows = [{**ROW, "timesteps": float(step)} for step in callback.eval_callback.evaluations_timesteps]
    callback.plateau_callback = _Plateau(rows)
    callback.stage_dir = stage_dir
    callback.model_dir = stage_dir / "models"
    callback.stage = "stance"
    callback.model_id = model_id
    callback.seed = 42
    callback.total_timesteps = 1200
    callback.curriculum_kwargs = dict(CURRICULUM)
    callback.warning_sources = []
    callback.robust_callback = SimpleNamespace(risk_coef=1.0, best_score=-math.inf)
    callback.num_timesteps = 100 * evaluations
    return callback


class TestTheRecord:
    def test_identity_budget_and_the_latest_evaluation(self, tmp_path):
        callback = _callback(tmp_path)
        (tmp_path / "run" / "provenance.json").write_text(json.dumps({"run_id": "trex-run-1"}), encoding="utf-8")
        record = callback.record("training")
        assert record["schema"] == PROGRESS_SCHEMA and record["status"] == "training"
        assert (record["species"], record["embodiment"], record["stage"]) == (
            "tyrannosaurus_rex",
            "anatomical",
            "stance",
        )
        assert record["model"] == "Tyrannosaurus Rex (Anatomical)"
        assert (record["run_id"], record["run_dir"], record["stage_dir"]) == ("trex-run-1", "run", "01_stance")
        assert (record["seed"], record["timesteps"], record["budget_timesteps"]) == (42, 300, 1200)
        assert record["fraction_done"] == 0.25
        assert datetime.fromisoformat(record["written_at"].replace("Z", "+00:00")).tzinfo is not None
        # No training start seen: the clock is unknown, not invented.
        assert record["steps_per_s"] is None and record["projected_finish"] is None
        assert record["evaluation"] == {
            "timesteps": 300,
            "episodes": 4,
            "mean_reward": 103.5,
            "std_reward": pytest.approx(float(np.std([102.0, 103.0, 104.0, 105.0])), rel=1e-5),
            "mean_episode_length": 850.0,
            "falls": 1,
            "full_horizon_fraction": 0.75,
            "termination_causes": {"truncated": 3, "nosedive": 1},
        }
        assert record["posture"] == posture_block(ROW, CURRICULUM)
        assert record["zero_action_reward"] is None  # the run captured no baseline

    def test_the_clock_projects_the_finish_from_the_observed_rate(self, tmp_path, monkeypatch):
        callback = _callback(tmp_path)
        monkeypatch.setattr(progress_report.time, "monotonic", lambda: 1000.0)
        callback._session_start = (900.0, datetime.fromisoformat("2026-10-16T20:00:00+00:00"), 100)
        record = callback.record("training")
        assert record["session_started_at"] == "2026-10-16T20:00:00Z"
        assert record["wall_clock_s"] == 100.0 and record["steps_per_s"] == 2.0  # 200 steps in 100 s
        finish = datetime.fromisoformat(record["projected_finish"].replace("Z", "+00:00"))
        written = datetime.fromisoformat(record["written_at"].replace("Z", "+00:00"))
        assert abs((finish - written).total_seconds() - 450.0) <= 1.0  # 900 steps left at 2 steps/s

    def test_the_history_keeps_the_last_evaluations_with_their_posture_statistics(self, tmp_path):
        callback = _callback(tmp_path, evaluations=HISTORY_EVALUATIONS + 2)
        history = callback.record("training")["history"]
        assert [entry["timesteps"] for entry in history] == [100 * (n + 3) for n in range(HISTORY_EVALUATIONS)]
        assert history[-1]["max_sole_cop_fore_aft_over_bar_fraction"] == 0.25
        assert history[-1]["max_foot_cop_fore_aft_mean"] == 1.12 and history[-1]["falls"] == 1
        assert "max_foot_cop_fore_aft_over_bar_fraction" not in history[-1]  # no bar, no fraction in the row

    def test_the_checkpoints_name_the_newest_periodic_pair_and_the_best_evaluations(self, tmp_path):
        callback = _callback(tmp_path)
        for name in ("stance_100_steps.zip", "stance_200_steps.zip", "stance_x_900_steps.zip", "best_model.zip"):
            (callback.model_dir / name).write_bytes(b"")
        callback.robust_callback = SimpleNamespace(
            risk_coef=1.0, best_score=float(103.5 - np.std([102, 103, 104, 105]))
        )
        checkpoints = callback.record("training")["checkpoints"]
        assert checkpoints["latest_periodic_timesteps"] == 200
        assert checkpoints["best_model_timesteps"] == 300 and checkpoints["best_mean_reward"] == 103.5
        assert checkpoints["robust_best_model_timesteps"] == 300

    def test_the_warnings_of_every_source_in_step_order(self, tmp_path):
        callback = _callback(tmp_path)
        callback.warning_sources = [
            SimpleNamespace(progress_warnings=[{"timesteps": 300, "source": "posture_drift", "message": "b"}]),
            SimpleNamespace(progress_warnings=[{"timesteps": 200, "source": "evaluation_plateau", "message": "a"}]),
            SimpleNamespace(),
        ]
        record = callback.record("training")
        assert [warning["message"] for warning in record["warnings"]] == ["a", "b"] and record["warnings_total"] == 2

    def test_only_the_newest_warnings_are_kept(self, tmp_path):
        callback = _callback(tmp_path)
        many = [{"timesteps": step, "source": "posture_drift", "message": str(step)} for step in range(25)]
        callback.warning_sources = [SimpleNamespace(progress_warnings=many)]
        record = callback.record("training")
        assert record["warnings_total"] == 25
        assert [warning["timesteps"] for warning in record["warnings"]] == list(range(15, 25))

    def test_the_zero_action_reward_when_the_run_captured_it(self, tmp_path):
        callback = _callback(tmp_path)
        baseline = {"results": {"tyrannosaurus_rex": {"anatomical": {"reward_mean": 3271.8}}}}
        (tmp_path / "run" / "zero_action_baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
        assert callback.record("training")["zero_action_reward"] == 3271.8

    def test_before_the_first_evaluation_and_without_a_model(self, tmp_path):
        callback = _callback(tmp_path, evaluations=0, model_id=None)
        callback.stage = 1
        record = callback.record("training")
        assert record["evaluation"] is None and record["posture"] == {} and record["history"] == []
        assert record["species"] is None and record["stage"] == "stage1"


class TestTheFile:
    def test_written_as_strict_json_with_nan_as_null(self, tmp_path):
        callback = _callback(tmp_path)
        callback.write("training")
        text = callback.path.read_text(encoding="utf-8")
        assert callback.path == tmp_path / "run" / "01_stance" / PROGRESS_FILENAME
        assert "NaN" not in text and text.endswith("\n")
        record = json.loads(text)
        assert record["posture"]["max_sole_cop_fore_aft"]["per_foot"]["l"]["digit_share"] is None

    def test_rewritten_through_a_temporary_file_in_the_stage_directory_and_os_replace(self, tmp_path, monkeypatch):
        from environments.shared import file_io

        callback = _callback(tmp_path)
        replaced = []
        real_replace = os.replace
        monkeypatch.setattr(
            file_io.os, "replace", lambda src, dst: replaced.append((src, dst)) or real_replace(src, dst)
        )
        callback.write("training")
        ((src, dst),) = replaced
        assert Path(dst) == callback.path
        assert Path(src).parent == callback.path.parent and Path(src).name.startswith(".progress.json.")

    def test_a_failed_rewrite_keeps_the_previous_file_whole_and_warns_once(self, tmp_path, monkeypatch, caplog):
        from environments.shared import file_io

        callback = _callback(tmp_path)
        callback.write("training")
        before = callback.path.read_bytes()

        def refuse(src, dst):
            raise OSError("the mount went away")

        monkeypatch.setattr(file_io.os, "replace", refuse)
        callback.num_timesteps = 400
        with caplog.at_level(logging.WARNING, logger=progress_report.__name__):
            callback.write("training")
            callback.write("training")
        assert callback.path.read_bytes() == before
        assert sorted(path.name for path in callback.path.parent.iterdir()) == ["models", PROGRESS_FILENAME]
        assert sum("Could not write" in message for message in caplog.messages) == 1

    def test_rewritten_after_each_new_evaluation_only(self, tmp_path):
        callback = _callback(tmp_path, evaluations=1)
        writes = []
        callback.write = writes.append  # type: ignore[method-assign]
        callback._on_step()
        callback._on_step()
        callback.eval_callback.evaluations_results.append([1.0])
        callback._on_step()
        callback._on_training_end()
        assert writes == ["training", "training", "ended"]


# ── a short real T. rex stance training run ─────────────────────────────────

EPISODES = 2
STEPS = 768


def _train(root: Path, monkeypatch, *, additions: bool) -> Path:
    """Train the T. rex stance for 768 steps on CPU (an evaluation every 256), into ``root/run/01_stance``.

    ``additions=False`` switches off everything this change adds: the
    progress file, the posture watch, the roll-rail telemetry and the
    termination counts.
    """
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mujoco")
    import torch

    from environments.shared import curriculum, eval_diagnostics, train_base
    from environments.shared.config import load_all_stages
    from environments.shared.gait import pad_telemetry
    from environments.shared.species_registry import get_species_config

    monkeypatch.setattr(train_base, "_report_hpt_metrics", lambda *a, **k: None)
    monkeypatch.setattr(train_base, "_eval_episodes_for_stage", lambda config: EPISODES)
    if not additions:
        monkeypatch.setattr(curriculum, "build_training_progress_callback", lambda *a, **k: None)
        monkeypatch.setattr(curriculum, "build_posture_drift_watch_callback", lambda *a, **k: None)
        monkeypatch.setattr(pad_telemetry.RailLayout, "from_morphology", classmethod(lambda cls, morphology: cls()))
        monkeypatch.setattr(eval_diagnostics.StageGatePlateauCallback, "_termination_scalars", lambda self, index: {})
    configs = load_all_stages(TYRANNOSAURUS_REX)
    stance = configs[1]
    # Past the settle window, so the posture is measured, and short enough for CI.
    stance["env_kwargs"]["max_episode_steps"] = int(stance["curriculum_kwargs"]["settle_steps"]) + 40
    stance["ppo_kwargs"].update(n_steps=256, batch_size=64, n_epochs=2)
    run = root / "run"
    run.mkdir(parents=True)
    (run / "provenance.json").write_text(json.dumps({"run_id": "trex-progress-test"}), encoding="utf-8")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_base.train(
            get_species_config(TYRANNOSAURUS_REX),
            configs,
            1,
            total_timesteps=STEPS,
            n_envs=1,
            seed=11,
            eval_freq=256,
            save_freq=512,
            verbose=0,
            use_tensorboard=False,
            output_dir=str(run / "01_stance"),
        )
    finally:
        torch.set_num_threads(threads)
    return run / "01_stance"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """The run with this change's additions and the run without them, trained once for the module."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        with_additions = _train(tmp_path_factory.mktemp("with"), monkeypatch, additions=True)
    finally:
        monkeypatch.undo()
    monkeypatch = pytest.MonkeyPatch()
    try:
        without = _train(tmp_path_factory.mktemp("without"), monkeypatch, additions=False)
    finally:
        monkeypatch.undo()
    return with_additions, without


def test_a_real_tyrannosaurus_rex_stance_run_writes_its_progress(trained):
    stage_dir, _ = trained
    record = json.loads((stage_dir / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    evaluations = np.load(stage_dir / "evaluations.npz")
    gate = np.load(stage_dir / "gate_progress.npz")
    assert record["schema"] == PROGRESS_SCHEMA and record["status"] == "ended"
    assert (record["species"], record["embodiment"], record["stage"]) == ("tyrannosaurus_rex", "anatomical", "stance")
    assert (record["run_id"], record["run_dir"], record["stage_dir"]) == ("trex-progress-test", "run", "01_stance")
    assert (record["seed"], record["timesteps"], record["budget_timesteps"], record["fraction_done"]) == (
        11,
        STEPS,
        STEPS,
        1.0,
    )
    assert record["steps_per_s"] > 0 and record["wall_clock_s"] > 0 and record["projected_finish"] is not None
    latest = record["evaluation"]
    assert latest["timesteps"] == int(evaluations["timesteps"][-1]) == STEPS
    assert latest["episodes"] == EPISODES
    assert latest["mean_reward"] == pytest.approx(float(np.mean(evaluations["results"][-1])), rel=1e-5)
    assert latest["falls"] == int(np.count_nonzero(evaluations["ep_lengths"][-1] < 240))
    assert sum(latest["termination_causes"].values()) == EPISODES
    assert latest["full_horizon_fraction"] == pytest.approx(float(gate["full_horizon_fraction"][-1]))
    posture = record["posture"]
    assert set(posture) == {
        "max_sole_cop_fore_aft",
        "max_foot_cop_fore_aft",
        "max_sole_tilt_deg",
        "max_sole_corner_lift_m",
        "leg_dc_deg",
    }
    for rail, bar in (
        ("max_sole_cop_fore_aft", 0.8),
        ("max_foot_cop_fore_aft", None),
        ("max_sole_tilt_deg", 2.0),
        ("max_sole_corner_lift_m", 0.004),
    ):
        assert posture[rail]["bar"] == bar and set(posture[rail]["per_foot"]) == {"r", "l"}
        assert ("over_bar_fraction" in posture[rail]) == (bar is not None)
        if math.isfinite(float(gate[f"{rail}_mean"][-1])):
            assert posture[rail]["mean"] == pytest.approx(float(gate[f"{rail}_mean"][-1]), rel=1e-3)
    assert set(posture["leg_dc_deg"]) == {"l_minus_r_hip_pitch", "l_minus_r_knee", "l_minus_r_ankle"}
    assert [entry["timesteps"] for entry in record["history"]] == [int(step) for step in evaluations["timesteps"]]
    checkpoints = record["checkpoints"]
    assert checkpoints["latest_periodic_timesteps"] == 512
    assert checkpoints["best_model_timesteps"] in evaluations["timesteps"].tolist()
    assert checkpoints["robust_best_model_timesteps"] in evaluations["timesteps"].tolist()
    assert isinstance(record["warnings"], list)
    # The termination counts and the roll rails reach the npz on every evaluation.
    counts = sum(gate[key] for key in gate.files if key.startswith("termination_"))
    assert counts.tolist() == [float(EPISODES)] * len(evaluations["timesteps"])
    assert {"r_sole_tilt_deg", "l_sole_corner_lift_m", "max_sole_tilt_deg_over_bar_fraction"} <= set(gate.files)
    # Nothing stranded beside it.
    assert not [path.name for path in stage_dir.iterdir() if path.name.startswith(".")]


def _policy_tensors(path: Path) -> dict[str, np.ndarray]:
    import torch

    with zipfile.ZipFile(path) as archive:
        state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
    return {key: value.numpy() for key, value in state.items()}


def test_the_additions_change_no_evaluation_and_no_checkpoint(trained):
    """Report-only: the same run without this change's additions evaluates and saves exactly the same."""
    with_additions, without = trained
    assert not (without / PROGRESS_FILENAME).exists()
    first, second = np.load(with_additions / "evaluations.npz"), np.load(without / "evaluations.npz")
    assert sorted(first.files) == sorted(second.files)
    for key in first.files:
        assert np.array_equal(first[key], second[key]), key
    gate, plain = np.load(with_additions / "gate_progress.npz"), np.load(without / "gate_progress.npz")
    added = set(gate.files) - set(plain.files)
    assert set(plain.files) <= set(gate.files)
    assert added and all(key.startswith("termination_") or "sole_tilt" in key or "corner_lift" in key for key in added)
    for key in plain.files:
        assert np.array_equal(gate[key], plain[key], equal_nan=True), key
    models = sorted(path.name for path in (without / "models").glob("*.zip"))
    assert models == sorted(path.name for path in (with_additions / "models").glob("*.zip"))
    assert {"best_model.zip", "robust_best_model.zip", "stage1_final.zip", "stage1_512_steps.zip"} <= set(models)
    for name in models:
        ours, theirs = _policy_tensors(with_additions / "models" / name), _policy_tensors(without / "models" / name)
        assert ours.keys() == theirs.keys()
        assert all(np.array_equal(ours[key], theirs[key]) for key in ours), name
