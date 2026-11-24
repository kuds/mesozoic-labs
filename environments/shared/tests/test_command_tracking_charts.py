"""The ``command_tracking/v1`` charts: the tracking history, the chart trace and the two figures (decision D-D61).

* **The tracking fraction over training**: ``DiagnosticsCallback`` records the
  live command's tracking keys as rollout means aligned with ``timesteps``,
  and a run whose env emits none (every ``command_mode = "none"`` stage)
  records exactly what it recorded before, in ``diagnostics.npz`` and in
  TensorBoard.
* **The chart trace**: the chart episodes are each reset family's first seed
  of the certification plan; each step row carries the gate's own per-step
  test and speed along the heading, and whether the re-rolled episode is the
  certified one; the CSV round-trips bit-exactly and is read strictly by
  header.  On a real plant the statue's re-rolled chart episodes are the
  report's records, and the trace's flag is the env's own.
* **The figures**: ``plot_command_tracking`` draws one column per chart
  episode and the tracking fraction over training against the profile's
  bar, and says "not recorded" without one; ``plot_command_gate_progress``
  draws each gated statistic of every handoff candidate (in the order
  training saved them) and of the certified handoff against its bar, and
  the criteria the certification panel's episodes failed.
* **The wiring**: ``stage_artifacts.write_command_tracking_figures`` is a
  no-op for other kinds, rolls the trace only on the pair the report scored,
  never raises, and a failure costs its own file; ``generate_stage_artifacts``
  and the CLI curriculum's judge draw them after the verdict.

The end-to-end figures of a trained stage are in
``test_command_tracking_wiring.py`` (``TestEndToEnd``).
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import shutil
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.shared.command_tracking_panel import CommandPanelPlan, plan_command_panel
from environments.shared.curriculum.checkpoints import write_handoff_choice
from environments.shared.curriculum.command_tracking_gate import (
    CRUISE_FAMILY,
    FOLLOW_PROFILE,
    PROVISIONAL_BIPED_GAIT_THRESHOLDS,
    PROVISIONAL_FOLLOW_THRESHOLDS,
    STAND_FAMILY,
    sample_in_tolerance,
    speed_along_heading,
    thresholds_from_curriculum,
)
from environments.shared.diagnostics import DiagnosticsCallback
from environments.shared.reporting import stage_artifacts
from environments.shared.reporting.command_tracking_report import write_command_tracking_report
from environments.shared.reporting.command_tracking_trace import (
    COMMAND_TRACKING_TRACE_CSV,
    COMMAND_TRACKING_TRACE_FIELDNAMES,
    chart_plan,
    read_command_tracking_trace,
    trace_episodes,
    trace_rows,
    write_command_tracking_trace,
)
from environments.shared.tests.stage_toml_helpers import replace_commands_table

from .test_command_tracking_wiring import (
    COMPSOGNATHUS_LONGIPES,
    MEASUREMENT,
    PLAN,
    WALK_CONFIG,
    WALK_CURRICULUM,
    _choose,
    _panel,
    _record,
    _report,
    _stage_dir,
)

# ── the tracking fraction over training ──────────────────────────────────────


def _callback(log_dir: Path) -> DiagnosticsCallback:
    callback = DiagnosticsCallback(log_dir=str(log_dir), verbose=0)
    model = MagicMock()
    model.logger = MagicMock()
    model.get_env.return_value = MagicMock(spec=[])
    del model.rollout_buffer
    callback.init_callback(model)
    callback.locals = {"infos": []}
    return callback


def _rollout(callback: DiagnosticsCallback, timesteps: int, infos: list[dict[str, Any]]) -> None:
    callback.num_timesteps = timesteps
    for info in infos:
        callback.locals = {"infos": [info]}
        callback._on_step()
    callback._on_rollout_end()


def _recorded_keys(callback: DiagnosticsCallback) -> set[str]:
    logger: Any = callback.logger  # the mock model's logger
    return {call.args[0] for call in logger.record.call_args_list}


class TestTheTrackingHistory:
    def test_a_live_command_run_records_the_tracking_keys_as_rollout_means(self, tmp_path):
        callback = _callback(tmp_path)
        live = {"forward_vel": 0.3, "tracking_error_v": 0.1, "tracking_error_yaw": 0.2, "reward_tracking": 0.5}
        _rollout(callback, 64, [{**live, "tracking_in_tolerance": True}, {**live, "tracking_in_tolerance": False}])
        _rollout(
            callback, 128, [{**live, "tracking_in_tolerance": True}] * 3 + [{**live, "tracking_in_tolerance": False}]
        )
        callback._on_training_end()
        with np.load(tmp_path / "diagnostics.npz") as data:
            assert data["timesteps"].tolist() == [64, 128]
            assert data["tracking_in_tolerance"].tolist() == [0.5, 0.75]
            assert data["tracking_error_v"].tolist() == pytest.approx([0.1, 0.1])
            assert data["reward_tracking"].tolist() == pytest.approx([0.5, 0.5])
        assert {f"diagnostics/{key}" for key in DiagnosticsCallback.COMMAND_TRACKING_KEYS} <= _recorded_keys(callback)

    def test_a_rollout_without_them_reads_nan_on_the_same_clock(self, tmp_path):
        callback = _callback(tmp_path)
        _rollout(callback, 64, [{"forward_vel": 0.3}])
        _rollout(callback, 128, [{"forward_vel": 0.3, "tracking_in_tolerance": True}])
        callback._on_training_end()
        with np.load(tmp_path / "diagnostics.npz") as data:
            assert data["tracking_in_tolerance"].shape == data["timesteps"].shape
            assert math.isnan(data["tracking_in_tolerance"][0]) and data["tracking_in_tolerance"][1] == 1.0

    def test_a_run_whose_env_emits_none_records_exactly_what_it_did(self, tmp_path):
        """A command_mode "none" stage: no tracking key in diagnostics.npz or TensorBoard, every other key as before."""
        plain, live = tmp_path / "plain", tmp_path / "live"
        plain.mkdir()
        live.mkdir()
        without, with_tracking = _callback(plain), _callback(live)
        info = {"forward_vel": 0.3, "reward_alive": 1.0, "pelvis_height": 0.5}
        for timesteps in (64, 128):
            _rollout(without, timesteps, [info, info])
            _rollout(with_tracking, timesteps, [{**info, "tracking_in_tolerance": True}] * 2)
        without._on_training_end()
        with_tracking._on_training_end()
        with np.load(plain / "diagnostics.npz") as data:
            plain_keys = set(data.files)
            plain_values = {key: data[key].tolist() for key in data.files}
        with np.load(live / "diagnostics.npz") as data:
            live_keys = set(data.files)
            shared = {key: data[key].tolist() for key in plain_keys}
        assert not plain_keys & set(DiagnosticsCallback.COMMAND_TRACKING_KEYS)
        assert live_keys - plain_keys == set(DiagnosticsCallback.COMMAND_TRACKING_KEYS)
        assert json.dumps(shared, sort_keys=True) == json.dumps(plain_values, sort_keys=True)
        assert not any("tracking" in key for key in _recorded_keys(without))
        assert _recorded_keys(with_tracking) - _recorded_keys(without) == {
            f"diagnostics/{key}" for key in DiagnosticsCallback.COMMAND_TRACKING_KEYS if key == "tracking_in_tolerance"
        }

    def test_the_tracking_keys_are_the_envs(self):
        """Each key is one ``BaseDinoEnv`` puts in a live step's info (``tracking_metrics`` and its reward term)."""
        from environments.shared.direction_commands import DirectionCommandController, tracking_metrics

        state = DirectionCommandController(WALK_CONFIG).reset(np.random.default_rng(0), 0.0)
        metrics = tracking_metrics(state, np.array([0.3, 0.0]), 0.0, current_heading=0.0)
        source = (Path(__file__).resolve().parents[1] / "base_env.py").read_text()
        for key in DiagnosticsCallback.COMMAND_TRACKING_KEYS:
            assert key in metrics or f'info["{key}"]' in source, key
        assert not set(DiagnosticsCallback.COMMAND_TRACKING_KEYS) & set(DiagnosticsCallback.INFO_KEYS)


# ── the chart trace ──────────────────────────────────────────────────────────


def _step(time_s: float, *, speed: float, cruise: bool = True, yaw_rate: float = 0.0) -> dict[str, Any]:
    """One live step's info row as the panel runner keeps it, with the tracking metrics the gate reads."""
    desired = 0.4 if cruise else 0.0
    return {
        "time_s": time_s,
        "step": int(round(time_s / 0.01)),
        "desired_speed": desired,
        "desired_heading": 0.05,
        "actual_heading": 0.0,
        "heading_error_rad": 0.05 if cruise else 0.0,
        "command_v_x": desired,
        "command_v_y": 0.0,
        "command_yaw_rate": 0.1 if cruise else 0.0,
        "actual_v_x": speed,
        "actual_v_y": 0.0,
        "actual_speed": speed,
        "actual_yaw_rate": yaw_rate,
        "tracking_error_v": abs(desired - speed),
        "tracking_error_yaw": abs((0.1 if cruise else 0.0) - yaw_rate),
        "tracking_error_heading": 0.05 if cruise else 0.0,
        "tracking_error_requested_speed": abs(desired - speed),
        "course_progress_m": speed * time_s,
        "reward": 1.0,
        "terminated": False,
    }


def _steps(*, cruise: bool, n: int = 50) -> list[dict[str, Any]]:
    """A cruise that falls short of the speed for 10 steps then tracks it, or a stand that holds still."""
    if cruise:
        return [_step(0.01 * (k + 1), speed=0.0 if k < 10 else 0.4) for k in range(n)]
    return [_step(0.01 * (k + 1), speed=0.0, cruise=False) for k in range(n)]


def _chart_episodes() -> list[Any]:
    """The walk PLAN's chart records (its first cruise and first stand seed), as the report records them."""
    plan = chart_plan(PLAN)
    return [_record(seed, index, family) for index, (seed, family) in enumerate(zip(plan.seeds, plan.families))]


class TestTheChartTrace:
    def test_the_chart_episodes_are_the_first_seed_of_each_reset_family(self):
        plan = chart_plan(PLAN)
        first_cruise = PLAN.seeds[PLAN.families.index(CRUISE_FAMILY)]
        first_stand = PLAN.seeds[PLAN.families.index(STAND_FAMILY)]
        assert plan.seeds == tuple(sorted((first_cruise, first_stand)))
        assert set(plan.families) == {CRUISE_FAMILY, STAND_FAMILY}
        assert (plan.profile, plan.seed_start) == (PLAN.profile, PLAN.seed_start)
        # A follow plan labels its seeds by the same reset draw.
        follow = thresholds_from_curriculum({**PROVISIONAL_FOLLOW_THRESHOLDS, **PROVISIONAL_BIPED_GAIT_THRESHOLDS})
        follow_plan = plan_command_panel(follow, WALK_CONFIG)
        assert chart_plan(follow_plan).seeds[0] == follow_plan.seeds[0]
        assert chart_plan(follow_plan).profile == FOLLOW_PROFILE
        assert len(chart_plan(CommandPanelPlan("walk", 3042, (3042, 3043), (CRUISE_FAMILY,) * 2, 2)).seeds) == 1

    def test_each_row_carries_the_gates_own_test_and_speed(self):
        cruise, stand = _chart_episodes()
        steps = [_steps(cruise=cruise.family == CRUISE_FAMILY), _steps(cruise=stand.family == CRUISE_FAMILY)]
        rows = trace_rows([cruise, stand], steps, MEASUREMENT)
        assert len(rows) == 100 and tuple(rows[0]) == COMMAND_TRACKING_TRACE_FIELDNAMES
        for row, step in zip(rows, steps[0] + steps[1]):
            assert row["in_tolerance"] is sample_in_tolerance(step, MEASUREMENT)
            assert row["speed_along_heading_mps"] == speed_along_heading(step)
        flags = [row["in_tolerance"] for row in rows[:50]]
        assert flags[:10] == [False] * 10 and all(flags[10:]), "short of the speed, then on it"
        assert all(row["in_tolerance"] for row in rows[50:]), "a stand that holds still"
        assert {row["episode_seed"] for row in rows} == {cruise.seed, stand.seed}

    def test_a_chart_episode_is_the_certified_one_only_when_its_record_is_the_reports(self, tmp_path):
        directory = _stage_dir(tmp_path)
        _choose(directory)
        report = _report(directory, _panel())
        episodes = _chart_episodes()
        steps = [_steps(cruise=episode.family == CRUISE_FAMILY) for episode in episodes]
        # The chart plan numbers its episodes 0 and 1; the record's position is not part of the comparison.
        assert episodes[1].episode != next(
            entry["record"]["episode"]
            for entry in report["episode_evidence"]
            if entry["record"]["seed"] == episodes[1].seed
        )
        assert all(row["record_matches_report"] for row in trace_rows(episodes, steps, MEASUREMENT, report=report))
        edited = [episodes[0], _record(episodes[1].seed, 1, episodes[1].family, reward=1.0)]
        rows = trace_rows(edited, steps, MEASUREMENT, report=report)
        assert {row["episode_seed"]: row["record_matches_report"] for row in rows} == {
            episodes[0].seed: True,
            episodes[1].seed: False,
        }
        assert not any(row["record_matches_report"] for row in trace_rows(episodes, steps, MEASUREMENT))
        with pytest.raises(ValueError, match="chart episodes"):
            trace_rows(episodes, steps[:1], MEASUREMENT)

    def test_the_csv_round_trips_bit_exactly(self, tmp_path):
        episodes = _chart_episodes()
        steps = [_steps(cruise=episode.family == CRUISE_FAMILY) for episode in episodes]
        steps[0][3]["actual_yaw_rate"] = math.nan
        rows = trace_rows(episodes, steps, MEASUREMENT)
        rows[1]["reward"] = 0.1 + 0.2
        path = write_command_tracking_trace(tmp_path, rows)
        assert path == tmp_path / COMMAND_TRACKING_TRACE_CSV
        read = read_command_tracking_trace(path)
        assert read is not None and len(read) == len(rows)
        assert math.isnan(read[3]["actual_yaw_rate"])
        read[3]["actual_yaw_rate"] = rows[3]["actual_yaw_rate"] = 0.0
        assert read == rows
        assert read[1]["reward"] == 0.1 + 0.2
        grouped = trace_episodes(read)
        assert [(seed, family, len(group)) for seed, family, group in grouped] == [
            (episodes[0].seed, episodes[0].family, 50),
            (episodes[1].seed, episodes[1].family, 50),
        ]

    def test_the_reader_refuses_another_layout_and_reads_no_file_as_none(self, tmp_path):
        assert read_command_tracking_trace(tmp_path / COMMAND_TRACKING_TRACE_CSV) is None
        (tmp_path / COMMAND_TRACKING_TRACE_CSV).write_text("episode_seed,step\n3042,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="has the columns"):
            read_command_tracking_trace(tmp_path / COMMAND_TRACKING_TRACE_CSV)


# ── the figures ──────────────────────────────────────────────────────────────


def _checkpoint(models: Path, name: str, steps: int) -> None:
    """A checkpoint pair whose zip records *steps* as SB3's ``data`` member does."""
    with zipfile.ZipFile(models / f"{name}.zip", "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": steps}))
    (models / f"{name}_vecnorm.pkl").write_bytes(name.encode())


def _charted_stage(tmp_path: Path, *, history: bool = True, failing: bool = False) -> Path:
    """A judged command stage: three candidates saved at known steps, a choice ranking them, a report, a trace."""
    directory = _stage_dir(tmp_path)
    models = directory / "models"
    for name, steps in (("robust_best_model", 40_000), ("best_model", 20_000), ("stage2_final", 60_000)):
        _checkpoint(models, name, steps)
    record = _choose(directory, "robust_best_model")
    lcbs = {"robust_best_model": 0.47, "best_model": 0.2, "stage2_final": 0.35}
    for entry in record["candidates"]:
        entry.update(
            passed=entry["name"] == "robust_best_model",
            families=[
                {"family": CRUISE_FAMILY, "survival_lcb": 0.6, "success_lcb": lcbs[entry["name"]]},
                {"family": STAND_FAMILY, "success_fraction": 1.0, "survival_lcb": None, "success_lcb": None},
            ],
        )
    record["selection_seed_start"] = 2042
    write_handoff_choice(models, record)
    overrides: dict[str, Any] = {f"seed_{PLAN.seeds[0]}": {"tracking_fraction_after_settle": 0.1}} if failing else {}
    report = _report(directory, _panel(**overrides))
    write_command_tracking_report(directory, report)
    episodes = _chart_episodes()
    step_rows = [_steps(cruise=episode.family == CRUISE_FAMILY) for episode in episodes]
    write_command_tracking_trace(directory, trace_rows(episodes, step_rows, MEASUREMENT, report=report))
    if history:
        np.savez(
            directory / "diagnostics.npz",
            timesteps=np.array([20_000, 40_000, 60_000]),
            tracking_in_tolerance=np.array([0.2, 0.4, 0.5]),
        )
    return directory


class TestTheCommandTrackingFigure:
    def test_one_column_per_chart_episode_and_the_history_against_the_bar(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_tracking

        directory = _charted_stage(tmp_path)
        out = tmp_path / "command_tracking.png"
        figure = plot_command_tracking(directory, model_id=COMPSOGNATHUS_LONGIPES, save_path=out)
        assert out.stat().st_size > 0
        assert len(figure.axes) == 2 * 3 + 1
        titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
        cruise, stand = _chart_episodes()
        assert any(
            f"cruise episode, seed {cruise.seed}" in title and "the certified episode" in title for title in titles
        )
        assert any(f"stand episode, seed {stand.seed}" in title and "passed" in title for title in titles)
        training = figure.axes[-1]
        labels = [line.get_label() for line in training.get_lines()]
        assert "in-tolerance fraction of training steps" in labels
        assert any(label.startswith("min_cruise_tracking_fraction 0.60") for label in labels)

    def test_a_failing_or_re_rolled_episode_says_so(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_tracking

        directory = _charted_stage(tmp_path, failing=True)
        rows = read_command_tracking_trace(directory / COMMAND_TRACKING_TRACE_CSV)
        assert rows is not None
        write_command_tracking_trace(directory, [{**row, "record_matches_report": False} for row in rows])
        figure = plot_command_tracking(directory)
        titles = " ".join(axis.get_title() for axis in figure.axes)
        assert "failed: min_cruise_tracking_fraction" in titles
        assert "re-rolled; its record is not the report's" in titles

    def test_a_trace_of_another_pair_is_not_drawn(self, tmp_path, caplog):
        """Each trace row names the checkpoint it was rolled on; rows of another than the report's handoff are left
        out, and so is a trace beside no report."""
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_tracking

        directory = _charted_stage(tmp_path)
        rows = read_command_tracking_trace(directory / COMMAND_TRACKING_TRACE_CSV)
        assert rows is not None
        report = json.loads((directory / "command_tracking_report.json").read_text())
        assert {row["checkpoint_sha256"] for row in rows} == {report["handoff"]["checkpoint_sha256"]}
        write_command_tracking_trace(directory, [{**row, "checkpoint_sha256": "sha256:" + "ab" * 32} for row in rows])
        with caplog.at_level(logging.WARNING):
            figure = plot_command_tracking(directory)
        assert len(figure.axes) == 1, "the history only"
        assert "not rolled on the report's handoff checkpoint" in caplog.text
        (directory / "command_tracking_report.json").unlink()
        (directory / "diagnostics.npz").unlink()
        write_command_tracking_trace(directory, rows)
        assert plot_command_tracking(directory) is None

    def test_the_training_panel_marks_the_cruise_ramp_its_recorded_task_declares(self, tmp_path):
        """The training rollouts run the ramp's commanded speeds, not the end state the bar is judged on: the window
        the recorded task's ramp spans is shaded and the title says so; a task without a ramp draws neither."""
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_tracking

        directory = _charted_stage(tmp_path)
        unramped = plot_command_tracking(directory).axes[-1]
        assert "cruise ramp" not in unramped.get_title() and not unramped.patches
        config = {"cruise_ramp_start_fraction": 0.25, "cruise_ramp_timesteps": 40_000}
        (directory / "stage_config.json").write_text(
            json.dumps({"task_fingerprint": {"command": {"schema": "x", "config": config}}}), encoding="utf-8"
        )
        training = plot_command_tracking(directory).axes[-1]
        assert "training rollouts at the cruise ramp's commanded speeds" in training.get_title()
        (shade,) = training.patches
        assert shade.get_label() == "cruise ramp: commanded speeds 0.25 to 1 x the end state"
        assert shade.get_x() == 0.0 and shade.get_width() == pytest.approx(0.04)  # millions of steps

    def test_without_a_history_it_says_not_recorded_and_without_either_it_draws_nothing(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_tracking

        directory = _charted_stage(tmp_path / "a", history=False)
        figure = plot_command_tracking(directory)
        assert any("not recorded" in text.get_text() for text in figure.axes[-1].texts)
        (directory / COMMAND_TRACKING_TRACE_CSV).unlink()
        assert plot_command_tracking(directory) is None
        only_history = _charted_stage(tmp_path / "b")
        (only_history / COMMAND_TRACKING_TRACE_CSV).unlink()
        assert len(plot_command_tracking(only_history).axes) == 1


class TestTheGateProgressFigure:
    def test_each_statistic_per_candidate_in_saving_order_and_the_certified_star(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_gate_progress

        directory = _charted_stage(tmp_path)
        out = tmp_path / "gate.png"
        figure = plot_command_gate_progress(directory, model_id=COMPSOGNATHUS_LONGIPES, save_path=out)
        assert out.stat().st_size > 0
        titles = [axis.get_title() for axis in figure.axes]
        assert titles[:3] == ["cruise survival LCB95", "cruise success LCB95", "stand success fraction"]
        success = figure.axes[1]
        ticks = [label.get_text() for label in success.get_xticklabels()]
        assert ticks == ["best_model\n20,000 steps", "robust_best_model\n40,000 steps", "stage2_final\n60,000 steps"]
        markers = [line for line in success.get_lines() if line.get_marker() == "o"]
        assert [line.get_ydata()[0] for line in markers] == [0.2, 0.47, 0.35]
        assert [line.get_markerfacecolor() for line in markers].count("white") == 2, "two candidates failed"
        (star,) = [line for line in success.get_lines() if line.get_marker() == "*"]
        assert star.get_xdata()[0] == 1, "the certified handoff, at its candidate"
        assert star.get_ydata()[0] == pytest.approx(0.4729, abs=1e-4)
        assert any(line.get_label() == "min_cruise_success_lcb 0.30" for line in success.get_lines())
        assert any("every episode passed every criterion" in text.get_text() for text in figure.axes[3].texts)

    def test_the_failures_panel_counts_each_criterion(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_gate_progress

        figure = plot_command_gate_progress(_charted_stage(tmp_path, failing=True))
        failures = figure.axes[3]
        assert [label.get_text() for label in failures.get_yticklabels()] == ["cruise: min_cruise_tracking_fraction"]
        assert [patch.get_width() for patch in failures.patches] == [1]

    def test_a_follow_gate_with_a_reference_draws_the_paired_bound(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_gate_progress

        directory = _charted_stage(tmp_path)
        report = json.loads((directory / "command_tracking_report.json").read_text())
        report.update(profile="follow")
        report["thresholds"].update(min_follow_success_lcb=0.6, min_follow_reference_delta_lcb=-0.1)
        report["result"].update(
            families=[{"family": "follow", "survival_lcb": 0.9, "success_lcb": 0.7, "criterion_failures": {}}],
            paired_delta_lcb=0.05,
            n_pairs=20,
        )
        (directory / "command_tracking_report.json").write_text(json.dumps(report))
        figure = plot_command_gate_progress(directory)
        titles = [axis.get_title() for axis in figure.axes]
        assert titles[:3] == ["follow survival LCB95", "follow success LCB95", "paired success difference LCB95"]
        (star,) = [line for line in figure.axes[2].get_lines() if line.get_marker() == "*"]
        assert star.get_ydata()[0] == 0.05

    def test_no_report_draws_nothing(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_command_gate_progress

        assert plot_command_gate_progress(tmp_path) is None


# ── the wiring ───────────────────────────────────────────────────────────────


def _write(stage_dir: Path, report: Any, figures: Path, verdict: bool | None = None) -> list[Path]:
    config = {"curriculum_kwargs": WALK_CURRICULUM, "env_kwargs": {}}
    return stage_artifacts.write_command_tracking_figures(
        model_id=COMPSOGNATHUS_LONGIPES,
        stage=2,
        stage_config=config,
        stage_dir=stage_dir,
        figures_out=figures,
        report=report,
        verdict=verdict,
    )


class TestTheWiring:
    def test_another_kind_is_a_no_op(self, tmp_path):
        config = {"curriculum_kwargs": {"gate_kind": "reward_and_length/v1"}, "env_kwargs": {}}
        assert (
            stage_artifacts.write_command_tracking_figures(
                model_id=COMPSOGNATHUS_LONGIPES,
                stage=1,
                stage_config=config,
                stage_dir=tmp_path,
                figures_out=tmp_path / "f",
                report=None,
            )
            == []
        )
        assert not (tmp_path / "f").exists()

    def test_the_trace_is_rolled_on_the_pair_the_report_scored(self, tmp_path, monkeypatch):
        pytest.importorskip("matplotlib")
        from environments.shared.reporting import command_tracking_trace

        directory = _charted_stage(tmp_path)
        (directory / COMMAND_TRACKING_TRACE_CSV).unlink()
        report = json.loads((directory / "command_tracking_report.json").read_text())
        calls: list[dict[str, Any]] = []

        def roll(model_id: Any, stage: Any, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            episodes = _chart_episodes()
            steps = [_steps(cruise=episode.family == CRUISE_FAMILY) for episode in episodes]
            return trace_rows(episodes, steps, MEASUREMENT, report=kwargs["report"])

        monkeypatch.setattr(command_tracking_trace, "roll_command_trace", roll)
        drawn = _write(directory, report, tmp_path / "figures")
        assert [path.name for path in drawn] == ["command_tracking.png", "command_gate_progress.png"]
        assert all(path.stat().st_size > 0 for path in drawn)
        (call,) = calls
        assert call["model_path"] == str(directory / "models" / "robust_best_model.zip")
        assert call["vecnorm_path"] == str(directory / "models" / "robust_best_model_vecnorm.pkl")
        assert (directory / COMMAND_TRACKING_TRACE_CSV).is_file()

    def test_a_pair_rewritten_since_the_report_is_not_charted(self, tmp_path, monkeypatch, caplog):
        pytest.importorskip("matplotlib")
        from environments.shared.reporting import command_tracking_trace

        directory = _charted_stage(tmp_path, history=False)
        (directory / COMMAND_TRACKING_TRACE_CSV).unlink()
        report = json.loads((directory / "command_tracking_report.json").read_text())
        (directory / "models" / "robust_best_model_vecnorm.pkl").write_bytes(b"rewritten")
        monkeypatch.setattr(command_tracking_trace, "roll_command_trace", MagicMock(side_effect=AssertionError))
        with caplog.at_level(logging.WARNING, logger=stage_artifacts.logger.name):
            drawn = _write(directory, report, tmp_path / "figures")
        assert "no longer the pair its report scored" in caplog.text
        assert [path.name for path in drawn] == ["command_gate_progress.png"], "no trace and no history: no chart"
        assert not (directory / COMMAND_TRACKING_TRACE_CSV).exists()

    def test_nothing_raises_and_each_failure_costs_its_own_file(self, tmp_path, monkeypatch, caplog):
        pytest.importorskip("matplotlib")
        from environments.shared import visualization
        from environments.shared.reporting import command_tracking_trace

        directory = _charted_stage(tmp_path)
        report = json.loads((directory / "command_tracking_report.json").read_text())
        monkeypatch.setattr(command_tracking_trace, "roll_command_trace", MagicMock(side_effect=RuntimeError("roll")))
        monkeypatch.setattr(visualization, "plot_command_gate_progress", MagicMock(side_effect=RuntimeError("draw")))
        with caplog.at_level(logging.WARNING, logger=stage_artifacts.logger.name):
            drawn = _write(directory, report, tmp_path / "figures")
        # The trace an earlier judgement left is removed before the roll, so the chart draws the history only.
        assert [path.name for path in drawn] == ["command_tracking.png"]
        assert not (directory / COMMAND_TRACKING_TRACE_CSV).exists()
        assert (
            "command-tracking trace failed" in caplog.text and "command_gate_progress.png figure failed" in caplog.text
        )

    def test_without_this_judgements_report_nothing_is_drawn_and_nothing_stale_is_left(self, tmp_path, monkeypatch):
        """The panel could not be rolled (no report, the gate failed closed) on a stage whose earlier judgement left a
        passing report, its trace and both figures: the figures and the trace go, and nothing is drawn."""
        pytest.importorskip("matplotlib")
        from environments.shared.reporting import command_tracking_trace

        directory = _charted_stage(tmp_path)
        report = json.loads((directory / "command_tracking_report.json").read_text())
        assert report["passed"] is True
        figures = tmp_path / "figures"
        monkeypatch.setattr(command_tracking_trace, "roll_command_trace", self._roll_from_report)
        assert len(_write(directory, report, figures, verdict=True)) == 2
        monkeypatch.setattr(command_tracking_trace, "roll_command_trace", MagicMock(side_effect=AssertionError))
        assert _write(directory, None, figures, verdict=False) == []
        assert not (figures / "command_tracking.png").exists() and not (figures / "command_gate_progress.png").exists()
        assert not (directory / COMMAND_TRACKING_TRACE_CSV).exists()

    def test_the_published_copies_go_too_when_drawing_into_a_staging_directory(self, tmp_path, monkeypatch):
        """The notebook layout: *figures_out* is a staging directory, and the stage directory's figures/ holds an
        earlier judgement's published copies.  Without this judgement's report both sets go."""
        pytest.importorskip("matplotlib")
        from environments.shared.reporting import command_tracking_trace, stage_layout

        directory = _charted_stage(tmp_path)
        report = json.loads((directory / "command_tracking_report.json").read_text())
        published = stage_layout.figures_dir(directory, create=True)
        monkeypatch.setattr(command_tracking_trace, "roll_command_trace", self._roll_from_report)
        assert len(_write(directory, report, published, verdict=True)) == 2
        staging = tmp_path / "staging" / "figures"
        staging.mkdir(parents=True)
        monkeypatch.setattr(command_tracking_trace, "roll_command_trace", MagicMock(side_effect=AssertionError))
        assert _write(directory, None, staging, verdict=False) == []
        for folder in (published, staging):
            assert not (folder / "command_tracking.png").exists()
            assert not (folder / "command_gate_progress.png").exists()
        assert not (directory / COMMAND_TRACKING_TRACE_CSV).exists()

    def test_the_stage_directory_sweep_is_a_no_op_for_another_kind(self, tmp_path):
        from environments.shared.reporting import stage_layout

        published = stage_layout.figures_dir(tmp_path, create=True)
        kept = [published / "command_tracking.png", published / "command_gate_progress.png"]
        for path in kept:
            path.write_bytes(b"png")
        stage_artifacts.remove_stale_command_tracking_outputs(
            stage=1, stage_config={"curriculum_kwargs": {"gate_kind": "reward_and_length/v1"}}, stage_dir=tmp_path
        )
        assert all(path.is_file() for path in kept)

    def test_the_gate_progress_title_is_the_verdict_not_the_reports_record(self, tmp_path, monkeypatch):
        """A report recording a pass that the judge refused (no valid handoff choice, say) is titled FAIL."""
        pytest.importorskip("matplotlib")
        from environments.shared.reporting import command_tracking_trace
        from environments.shared.visualization import plot_command_gate_progress

        directory = _charted_stage(tmp_path)
        report = json.loads((directory / "command_tracking_report.json").read_text())
        monkeypatch.setattr(command_tracking_trace, "roll_command_trace", self._roll_from_report)
        drawn: list[Any] = []
        original = plot_command_gate_progress

        def capture(*args: Any, **kwargs: Any) -> Any:
            drawn.append(original(*args, **kwargs))
            return drawn[-1]

        from environments.shared import visualization

        monkeypatch.setattr(visualization, "plot_command_gate_progress", capture)
        _write(directory, report, tmp_path / "figures", verdict=False)
        (figure,) = drawn
        title = figure.get_suptitle()
        assert "gate progress: FAIL" in title and "PASS" not in title
        alone = plot_command_gate_progress(directory).get_suptitle()
        assert "gate progress: report records PASS" in alone

    @staticmethod
    def _roll_from_report(model_id: Any, stage: Any, **kwargs: Any) -> list[dict[str, Any]]:
        episodes = _chart_episodes()
        steps = [_steps(cruise=episode.family == CRUISE_FAMILY) for episode in episodes]
        return trace_rows(episodes, steps, MEASUREMENT, report=kwargs["report"])

    def test_the_figures_follow_the_verdict_on_both_paths(self):
        from environments.shared import train_base

        source = inspect.getsource(stage_artifacts.generate_stage_artifacts)
        assert (
            source.index("_apply_stage_gate(")
            < source.index("plot_gate_progress(")
            < source.index("write_command_tracking_figures(")
            < source.index("_write_training_history(")
        )
        assert source.index("if generate_graphs:") < source.index("write_command_tracking_figures(")
        # The stage-directory sweep runs before the staged block and outside the graphs switch.
        assert (
            source.index("_apply_stage_gate(")
            < source.index("remove_stale_command_tracking_outputs(")
            < source.index("staged_artifacts(")
            < source.index("if generate_graphs:")
        )
        judge = inspect.getsource(train_base._post_training_command_verdict)
        assert judge.index("evaluate_stage_gate(") < judge.index("write_command_tracking_figures(")
        assert "figures_dir(stage_dir)" in judge


# ── a real plant ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_stage(tmp_path_factory) -> dict[str, Any]:
    """The compsognathus's committed locomotion stage with the design's walk command, a 2 + 1 panel of 3 s episodes."""
    from environments.shared.config import load_stage_config
    from environments.shared.curriculum.gate_schema import NON_THRESHOLD_CURRICULUM_KEYS
    from environments.shared.paths import REPOSITORY_ROOT
    from environments.shared.stage_manifest import load_stage_manifest

    from .test_command_tracking_wiring import WALK_COMMANDS

    root = tmp_path_factory.mktemp("configs")
    directory = root / COMPSOGNATHUS_LONGIPES.species / COMPSOGNATHUS_LONGIPES.embodiment
    shutil.copytree(COMPSOGNATHUS_LONGIPES.path(REPOSITORY_ROOT / "configs"), directory)
    path = directory / load_stage_manifest(COMPSOGNATHUS_LONGIPES).by_id("locomotion").config_file
    replace_commands_table(path, WALK_COMMANDS)
    config = load_stage_config(COMPSOGNATHUS_LONGIPES, "locomotion", config_path=str(path))
    config["env_kwargs"]["max_episode_steps"] = 150
    curriculum = {k: v for k, v in config["curriculum_kwargs"].items() if k in NON_THRESHOLD_CURRICULUM_KEYS}
    curriculum.update({**WALK_CURRICULUM, "min_cruise_episodes": 2, "min_stand_episodes": 1})
    config["curriculum_kwargs"] = curriculum
    return config


class TestARealPlant:
    def test_the_statues_chart_episodes_are_its_certified_ones_under_the_envs_own_flag(self, live_stage):
        from environments.shared.command_tracking_panel import roll_command_panel, zero_action_predict
        from environments.shared.reporting.command_tracking_report import build_command_tracking_report
        from environments.shared.reporting.command_tracking_trace import roll_command_trace
        from environments.shared.species_registry import env_class

        report = build_command_tracking_report(
            COMPSOGNATHUS_LONGIPES, "locomotion", stage_config=live_stage, zero_action=True
        )
        rows = roll_command_trace(
            COMPSOGNATHUS_LONGIPES, "locomotion", stage_config=live_stage, report=report, zero_action=True
        )
        plan = report["plan"]
        expected = chart_plan(
            CommandPanelPlan(**{**plan, "seeds": tuple(plan["seeds"]), "families": tuple(plan["families"])})
        )
        grouped = trace_episodes(rows)
        assert [(seed, family) for seed, family, _ in grouped] == list(zip(expected.seeds, expected.families))
        assert all(row["record_matches_report"] for row in rows), "the re-rolled episodes are the certified ones"
        assert all(len(group) == report["horizon"] for _, _, group in grouped), "the statue stands the horizon"
        # The trace's flag is the env's own (no gate override): the same steps rolled by the runner.
        env = env_class(COMPSOGNATHUS_LONGIPES)(**live_stage["env_kwargs"])
        try:
            statue = zero_action_predict(env.action_space.shape[0])
        finally:
            env.close()
        rollout = roll_command_panel(
            COMPSOGNATHUS_LONGIPES,
            live_stage,
            predict=statue,
            thresholds=thresholds_from_curriculum(live_stage["curriculum_kwargs"]),
            plan=expected,
            keep_steps=True,
        )
        assert rollout.steps is not None
        env_flags = [bool(step["tracking_in_tolerance"]) for steps in rollout.steps for step in steps]
        assert [row["in_tolerance"] for row in rows] == env_flags
