"""Tests for sweep results.py — trial result collection, CSV export, model path resolution."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.scripts.sweep import (
    SweepStageError,
    _best_trial_model_path,
    _best_trial_model_path_any,
    _collect_trial_results,
    _evaluate_curriculum_gate,
    collect_results_from_disk,
    plot_sweep_results,
    write_results_csv,
)
from environments.shared.scripts.sweep.results import _gate_status

# configs/trex/stance.toml's [curriculum] gate declaration: min_avg_reward is
# a collapse RAIL the 3271.8 statue and the 2133.4 chatterer both clear.
STANCE_CURRICULUM = {
    "gate_schema_version": 1,
    "gate_kind": "stance_quality/v1",
    "min_avg_reward": 2100.0,
    "min_full_horizon_fraction": 0.9,
    "max_unsupported_duty": 0.05,
    "max_unsupported_duty_ucb": 0.10,
}
RECOVERY_CURRICULUM = {
    "gate_schema_version": 1,
    "gate_kind": "recovery_quality/v1",
    "min_recovery_success_lcb": 0.8,
    "recovery_t_recover_steps": 100,
    "recovery_dwell_steps": 50,
}
# The hunting gate (plan §4.4) at the provisional bar: 20/30 clears 0.5, 19/30 does not.
TASK_SUCCESS_CURRICULUM = {
    "gate_schema_version": 1,
    "gate_kind": "task_success/v1",
    "min_success_lcb": 0.5,
    "min_eval_episodes": 30,
    "min_avg_reward": 361.0,
}

# ── Helpers ──────────────────────────────────────────────────────────


def _make_mock_trial(trial_id, params=None, metrics=None):
    """Helper to build a mock Vertex AI trial object."""
    trial = MagicMock()
    trial.id = trial_id
    trial.parameters = []
    if params:
        for pid, val in params.items():
            p = MagicMock()
            p.parameter_id = pid
            p.value = val
            trial.parameters.append(p)

    if metrics:
        measurement = MagicMock()
        metric_objs = []
        for mid, val in metrics.items():
            m = MagicMock()
            m.metric_id = mid
            m.value = val
            metric_objs.append(m)
        measurement.metrics = metric_objs
        trial.final_measurement = measurement
    else:
        trial.final_measurement = None

    return trial


def _write_trial_metrics(output_base, trial_id, metrics_dict):
    """Write a metrics.json sidecar for a mock trial."""
    trial_dir = Path(output_base) / str(trial_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f)


def _setup_sweep_dir(base, stage, trials):
    """Create a sweep-style directory with metrics.json per trial."""
    stage_dir = base / f"stage{stage}"
    for trial_id, metrics in trials.items():
        trial_dir = stage_dir / str(trial_id)
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)
    return stage_dir


def _make_mock_gcs_blobs(file_map):
    """Build mock GCS blobs from a dict of {blob_name: content_bytes}."""
    blobs = []
    for name, content in file_map.items():
        blob = MagicMock()
        blob.name = name

        def _download(local_path, _content=content):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(_content)

        blob.download_to_filename = _download
        blobs.append(blob)
    return blobs


# ── _collect_trial_results ───────────────────────────────────────────────


class TestCollectTrialResults:
    def test_passing_trial(self, tmp_path):
        trial = _make_mock_trial(
            "1",
            params={"ppo_learning_rate": 0.0003},
            metrics={"best_mean_reward": 150.0},
        )
        _write_trial_metrics(
            tmp_path,
            "1",
            {
                "best_mean_reward": 150.0,
                "best_mean_episode_length": 500.0,
            },
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 100.0}}

        rows = _collect_trial_results(job, 1, stage_config, output_base=str(tmp_path))
        assert len(rows) == 1
        assert rows[0]["stage_passed"] is True
        assert rows[0]["best_mean_reward"] == 150.0
        assert rows[0]["best_mean_episode_length"] == 500.0

    def test_failing_trial_below_threshold(self, tmp_path):
        trial = _make_mock_trial(
            "2",
            metrics={"best_mean_reward": 50.0},
        )
        _write_trial_metrics(tmp_path, "2", {"best_mean_reward": 50.0})
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 100.0}}

        rows = _collect_trial_results(job, 1, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is False

    def test_crashed_trial_no_metrics(self, tmp_path):
        trial = _make_mock_trial("3")
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {"gate_kind": "reward_and_length/v1"}}

        rows = _collect_trial_results(job, 1, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is False
        assert rows[0]["best_mean_reward"] is None

    def test_no_gate_declared_is_not_a_pass(self, tmp_path):
        """An empty curriculum block used to pass any finite reward vacuously."""
        trial = _make_mock_trial("4", metrics={"best_mean_reward": 10.0})
        _write_trial_metrics(tmp_path, "4", {"best_mean_reward": 10.0})
        job = MagicMock()
        job.trials = [trial]
        stage_config = {"curriculum_kwargs": {}}

        rows = _collect_trial_results(job, 3, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is None
        assert rows[0]["gate_evaluable"] is False
        assert "declares no gate_kind" in rows[0]["gate_reason"]

    def test_ep_length_threshold_checked(self, tmp_path):
        trial = _make_mock_trial(
            "5",
            metrics={"best_mean_reward": 200.0},
        )
        _write_trial_metrics(
            tmp_path,
            "5",
            {
                "best_mean_reward": 200.0,
                "best_mean_episode_length": 100.0,
            },
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {
            "curriculum_kwargs": {
                "gate_kind": "reward_and_length/v1",
                "min_avg_reward": 100.0,
                "min_avg_episode_length": 500.0,
            }
        }

        rows = _collect_trial_results(job, 1, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is False

    def test_forward_vel_threshold_checked(self, tmp_path):
        trial = _make_mock_trial(
            "6",
            metrics={"best_mean_reward": 200.0},
        )
        _write_trial_metrics(
            tmp_path,
            "6",
            {
                "best_mean_reward": 200.0,
                "best_mean_forward_vel": 0.5,
            },
        )
        job = MagicMock()
        job.trials = [trial]
        stage_config = {
            "curriculum_kwargs": {
                "gate_kind": "reward_and_length/v1",
                "min_avg_reward": 100.0,
                "min_avg_forward_vel": 1.0,
            }
        }

        rows = _collect_trial_results(job, 2, stage_config, output_base=str(tmp_path))
        assert rows[0]["stage_passed"] is False


# ── _evaluate_curriculum_gate: non-finite rewards ──────────────────────────


class TestEvaluateCurriculumGateNonFinite:
    """NaN is not None and ``nan < threshold`` is False — it must not pass."""

    @pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf"), None, ""])
    def test_non_finite_reward_fails_with_reason(self, reward):
        passed, reasons = _evaluate_curriculum_gate(reward, {}, 100.0, None, None, None)
        assert passed is False
        assert any("no finite reward" in reason for reason in reasons)

    def test_nan_fails_even_without_thresholds(self):
        passed, reasons = _evaluate_curriculum_gate(float("nan"), {}, None, None, None, None)
        assert passed is False
        assert any("no finite reward" in reason for reason in reasons)

    def test_finite_reward_still_passes(self):
        assert _evaluate_curriculum_gate(150.0, {}, 100.0, None, None, None) == (True, [])

    def test_nan_ray_row_is_not_stage_passed(self):
        """collect_ray_results: a trial that errored before its first tune.report."""
        pd = pytest.importorskip("pandas")
        from environments.shared.scripts.sweep import collect_ray_results

        df = pd.DataFrame(
            [
                {"trial_id": "errored", "best_mean_reward": float("nan")},
                {"trial_id": "ok", "best_mean_reward": 150.0},
            ]
        )
        rows = collect_ray_results(
            df, 1, {"curriculum_kwargs": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 100.0}}
        )
        by_id = {r["trial_id"]: r for r in rows}
        assert by_id["errored"]["stage_passed"] is False
        assert "no finite reward" in by_id["errored"]["gate_reason"]
        assert by_id["ok"]["stage_passed"] is True


# ── gate_kind routing ────────────────────────────────────────────────────────


class TestGateKindRouting:
    """A stage's pass/fail follows its declared gate_kind, never the reward rail alone."""

    @staticmethod
    def _vertex_rows(tmp_path, curriculum, rewards):
        job = MagicMock()
        job.trials = []
        for trial_id, reward in rewards.items():
            job.trials.append(_make_mock_trial(trial_id, metrics={"best_mean_reward": reward}))
            _write_trial_metrics(tmp_path, trial_id, {"best_mean_reward": reward, "best_mean_episode_length": 1000.0})
        return _collect_trial_results(job, 1, {"curriculum_kwargs": curriculum}, output_base=str(tmp_path))

    def test_stance_gate_is_not_evaluable_offline(self, tmp_path):
        """The statue and the chatterer both clear the 2100 rail; neither is a pass."""
        rows = self._vertex_rows(tmp_path, STANCE_CURRICULUM, {"statue": 3271.8, "chatterer": 2133.4})
        assert len(rows) == 2
        for row in rows:
            assert row["stage_passed"] is None
            assert row["gate_evaluable"] is False
            assert row["gate_kind"] == "stance_quality/v1"
            assert "stance_quality/v1" in row["gate_reason"]
            # The rail is still reported as a threshold column, not as the gate.
            assert row["reward_threshold"] == 2100.0

    def test_recovery_gate_is_not_evaluable_offline(self, tmp_path):
        rows = self._vertex_rows(tmp_path, RECOVERY_CURRICULUM, {"1": 500.0})
        assert rows[0]["stage_passed"] is None
        assert rows[0]["gate_evaluable"] is False
        assert "recovery_quality/v1" in rows[0]["gate_reason"]

    def test_reward_and_length_gate_still_evaluates(self, tmp_path):
        curriculum = {"gate_schema_version": 1, "gate_kind": "reward_and_length/v1", "min_avg_reward": 100.0}
        rows = self._vertex_rows(tmp_path, curriculum, {"hi": 150.0, "lo": 50.0})
        by_id = {r["trial_id"]: r for r in rows}
        assert by_id["hi"]["stage_passed"] is True
        assert by_id["hi"]["gate_evaluable"] is True
        assert by_id["hi"]["gate_reason"] == ""
        assert by_id["lo"]["stage_passed"] is False
        assert "threshold" in by_id["lo"]["gate_reason"]

    def test_undeclared_gate_is_not_evaluable(self, tmp_path):
        """Thresholds without a gate_kind are columns, never a verdict — the
        judgement reporting.gates.evaluate_stage_gate gives the same input."""
        rows = self._vertex_rows(tmp_path, {"min_avg_reward": 100.0}, {"hi": 150.0})
        assert rows[0]["stage_passed"] is None
        assert rows[0]["gate_kind"] is None
        assert rows[0]["gate_evaluable"] is False
        assert "declares no gate_kind" in rows[0]["gate_reason"]
        assert rows[0]["reward_threshold"] == 100.0
        with pytest.raises(SweepStageError, match="<undeclared>"):
            _best_trial_model_path(rows, "b", "trex", 1)

    def test_unreadable_stage_config_names_that_cause(self, tmp_path):
        stage_dir = _setup_sweep_dir(tmp_path, 1, {"1": {"best_mean_reward": 150.0}})
        (stage_dir / "stage_config.json").write_text("{not json", encoding="utf-8")
        rows = collect_results_from_disk(tmp_path)
        assert rows[0]["stage_passed"] is None
        assert rows[0]["gate_evaluable"] is False
        assert "stage config unreadable" in rows[0]["gate_reason"]
        assert "stage_config.json" in rows[0]["gate_reason"]

    def test_none_gate_never_passes(self, tmp_path):
        rows = self._vertex_rows(tmp_path, {"gate_schema_version": 1, "gate_kind": "none/v1"}, {"hi": 150.0})
        assert rows[0]["stage_passed"] is False
        assert rows[0]["gate_evaluable"] is True
        assert "none/v1" in rows[0]["gate_reason"]

    def test_unknown_gate_kind_is_not_evaluable(self, tmp_path):
        rows = self._vertex_rows(tmp_path, {"gate_kind": "made_up/v9"}, {"hi": 150.0})
        assert rows[0]["stage_passed"] is None
        assert "unknown gate_kind" in rows[0]["gate_reason"]

    def test_disk_collector_reads_gate_kind_from_stage_config(self, tmp_path):
        stage_dir = _setup_sweep_dir(tmp_path, 1, {"statue": {"best_mean_reward": 3271.8}})
        (stage_dir / "stage_config.json").write_text(json.dumps({"curriculum": STANCE_CURRICULUM}))
        rows = collect_results_from_disk(tmp_path)
        assert rows[0]["stage_passed"] is None
        assert rows[0]["gate_evaluable"] is False
        assert rows[0]["reward_threshold"] == 2100.0

    def test_disk_collector_reads_gate_kind_from_per_trial_config(self, tmp_path):
        trial_dir = tmp_path / "stage1" / "statue"
        trial_dir.mkdir(parents=True)
        (trial_dir / "metrics.json").write_text(json.dumps({"best_mean_reward": 3271.8}))
        (trial_dir / "stage_config.json").write_text(json.dumps({"curriculum": STANCE_CURRICULUM}))
        rows = collect_results_from_disk(tmp_path)
        assert rows[0]["stage_passed"] is None
        assert rows[0]["gate_evaluable"] is False

    def test_ray_collector_routes_through_gate_kind(self):
        pd = pytest.importorskip("pandas")
        from environments.shared.scripts.sweep import collect_ray_results

        df = pd.DataFrame([{"trial_id": "statue", "best_mean_reward": 3271.8}])
        rows = collect_ray_results(df, 1, {"curriculum_kwargs": STANCE_CURRICULUM})
        assert rows[0]["stage_passed"] is None
        assert rows[0]["gate_evaluable"] is False
        assert rows[0]["reward_threshold"] == 2100.0


# ── not-evaluable rows downstream: selection, CSV, plot, state ───────────────


def _unevaluable_row(trial_id, reward, stage=1):
    return {
        "trial_id": trial_id,
        "stage": stage,
        "best_mean_reward": reward,
        "reward_threshold": 2100.0,
        "gate_kind": "stance_quality/v1",
        "gate_evaluable": False,
        "stage_passed": None,
        "gate_reason": "gate_kind 'stance_quality/v1' is judged on evidence a sweep row does not carry",
    }


class TestTaskSuccessRows:
    """Decision D-B12: task_success/v1 is offline-evaluable from the recorded count."""

    @staticmethod
    def _rows(tmp_path, metrics_by_trial):
        job = MagicMock()
        job.trials = []
        for trial_id, metrics in metrics_by_trial.items():
            job.trials.append(_make_mock_trial(trial_id, metrics={"best_mean_reward": metrics["best_mean_reward"]}))
            _write_trial_metrics(tmp_path, trial_id, metrics)
        return {
            row["trial_id"]: row
            for row in _collect_trial_results(
                job, 3, {"curriculum_kwargs": TASK_SUCCESS_CURRICULUM}, output_base=str(tmp_path)
            )
        }

    @staticmethod
    def _panel(count, n=30, *, selected_mean_reward=600.0, best_mean_reward=600.0):
        """The metrics.json entries run_success_panel records for a judged hunt panel."""
        return {
            "best_mean_reward": best_mean_reward,
            "success_count": count,
            "n_success_episodes": n,
            "selected_mean_reward": selected_mean_reward,
            "selected_mean_episode_length": 1000.0,
        }

    def test_task_success_gate_evaluates_from_the_recorded_count(self, tmp_path):
        rows = self._rows(
            tmp_path,
            {
                "twenty": self._panel(20),
                "nineteen": self._panel(19),
                "railed": self._panel(30, selected_mean_reward=200.0),
            },
        )
        assert rows["twenty"]["stage_passed"] is True and rows["twenty"]["gate_evaluable"] is True
        assert rows["twenty"]["gate_reason"] == ""
        assert rows["nineteen"]["stage_passed"] is False and rows["nineteen"]["gate_evaluable"] is True
        assert "task_success_lcb" in rows["nineteen"]["gate_reason"] and "19/30" in rows["nineteen"]["gate_reason"]
        assert rows["railed"]["stage_passed"] is False
        assert "task_success rail" in rows["railed"]["gate_reason"]
        assert rows["twenty"]["success_lcb_threshold"] == 0.5
        assert (rows["twenty"]["success_count"], rows["twenty"]["n_success_episodes"]) == (20, 30)
        assert rows["twenty"]["selected_mean_reward"] == 600.0

    def test_the_rail_reads_the_selected_panel_not_the_eval_callback_best(self, tmp_path):
        """best_mean_reward is the argmax EvalCallback panel of a possibly different checkpoint; the
        rail is judged on the panel the count came from, the one the on-disk judge rails on too."""
        rows = self._rows(
            tmp_path,
            {
                "npz_high": self._panel(30, selected_mean_reward=300.0, best_mean_reward=700.0),
                "npz_low": self._panel(30, selected_mean_reward=600.0, best_mean_reward=100.0),
                "no_panel_reward": {"best_mean_reward": 700.0, "success_count": 30, "n_success_episodes": 30},
            },
        )
        assert rows["npz_high"]["stage_passed"] is False and "task_success rail" in rows["npz_high"]["gate_reason"]
        assert rows["npz_low"]["stage_passed"] is True
        assert rows["no_panel_reward"]["stage_passed"] is None and rows["no_panel_reward"]["gate_evaluable"] is False
        assert "no selected_mean_reward" in rows["no_panel_reward"]["gate_reason"]

    def test_task_success_without_a_count_is_not_evaluable(self, tmp_path):
        """A rounded mean_success_rate cannot recover k/n: never a pass, never a reward substitute."""
        rows = self._rows(tmp_path, {"mean_only": {"best_mean_reward": 1e9, "mean_success_rate": 1.0}})
        row = rows["mean_only"]
        assert row["stage_passed"] is None and row["gate_evaluable"] is False
        assert "no success_count/n_success_episodes" in row["gate_reason"]
        assert _gate_status(row) == "not evaluable"

    def test_a_non_integral_count_is_not_evaluable(self, tmp_path):
        """20.4/30 is not a count; a corrupted or hand-edited sidecar is refused, never truncated to 20/30."""
        rows = self._rows(
            tmp_path,
            {"frac_count": self._panel(20.4), "frac_n": self._panel(22, n=30.9)},
        )
        for row in rows.values():
            assert row["stage_passed"] is None and row["gate_evaluable"] is False
            assert "non-integer success_count/n_success_episodes" in row["gate_reason"]

    def test_success_lcb_threshold_column_is_exported(self, tmp_path):
        from environments.shared.reporting.csv_output import CSV_METRIC_COLUMNS

        # The pin itself: the columns are declared, in order, before the verdict columns.
        for column in ("success_lcb_threshold", "success_count", "n_success_episodes", "selected_mean_reward"):
            assert column in CSV_METRIC_COLUMNS
            assert CSV_METRIC_COLUMNS.index(column) < CSV_METRIC_COLUMNS.index("gate_kind")
        rows = list(self._rows(tmp_path, {"t": self._panel(20)}).values())
        path = write_results_csv(rows, tmp_path / "sweep.csv")
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            written = list(reader)
        assert {"success_lcb_threshold", "success_count", "n_success_episodes"} <= set(fieldnames)
        assert written[0]["success_lcb_threshold"] == "0.5"
        assert (written[0]["success_count"], written[0]["n_success_episodes"]) == ("20", "30")

    def test_collect_ray_results_carries_the_count_and_judges_the_row(self):
        """The Ray DataFrame path copies the count and the panel aggregates, and judges the kind on them."""
        pandas = pytest.importorskip("pandas")
        from environments.shared.scripts.sweep.ray_tune import collect_ray_results

        frame = pandas.DataFrame(
            [
                {"trial_id": "twenty", **self._panel(20)},
                {"trial_id": "nineteen", **self._panel(19)},
                {"trial_id": "unjudged", "best_mean_reward": 700.0},
            ]
        )
        rows = {
            row["trial_id"]: row
            for row in collect_ray_results(frame, 3, {"curriculum_kwargs": dict(TASK_SUCCESS_CURRICULUM)})
        }
        assert rows["twenty"]["stage_passed"] is True and rows["twenty"]["gate_evaluable"] is True
        assert (rows["twenty"]["success_count"], rows["twenty"]["n_success_episodes"]) == (20, 30)
        assert rows["twenty"]["selected_mean_reward"] == 600.0 and rows["twenty"]["success_lcb_threshold"] == 0.5
        assert rows["nineteen"]["stage_passed"] is False and "19/30" in rows["nineteen"]["gate_reason"]
        assert rows["unjudged"]["stage_passed"] is None and rows["unjudged"]["gate_evaluable"] is False


class TestTaskSuccessRowAgreesWithTheVerdict:
    """The sweep row and the trial's on-disk verdict are judged on ONE panel (D-B12 amendment).

    The trial worker (train_base.run_success_panel, shared by train() and
    the Ray Tune worker) writes evaluation_selected.csv from its post-training
    panel and records metrics.json's success_count / selected_mean_reward
    from the same episodes, so generate_stage_artifacts' verdict (through
    _apply_stage_gate) and the collector's row cannot disagree — on the
    bound or on the rail.
    """

    @staticmethod
    def _trial(tmp_path, successes, *, reward=600.0, handoff=True, monkeypatch=None):
        """A trial directory as the worker leaves it: the panel run through run_success_panel."""
        from types import SimpleNamespace

        from environments.shared import train_base
        from environments.shared.curriculum.checkpoints import select_handoff_checkpoint

        trial_dir = tmp_path / "stage3" / "trial"
        models = trial_dir / "models"
        models.mkdir(parents=True)
        if handoff:
            (models / "robust_best_model.zip").write_bytes(b"weights")
            (models / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")
        n = len(successes)
        monkeypatch.setattr(
            train_base,
            "eval_policy",
            lambda *a, n_episodes: ([reward] * n, [1000] * n, [1.0] * n, list(successes), [5.0] * n),
        )
        _, metrics = train_base.run_success_panel(
            object(),
            MagicMock(),
            SimpleNamespace(species="trex", success_keys=["bite_success"]),
            n_episodes=n,
            curriculum=TASK_SUCCESS_CURRICULUM,
            evaluated_handoff=select_handoff_checkpoint(models),
            stage_dir=trial_dir,
        )
        (trial_dir / "metrics.json").write_text(json.dumps({"best_mean_reward": 700.0, **metrics}))
        (trial_dir / "stage_config.json").write_text(
            json.dumps({"species": "trex", "stage": 3, "curriculum": TASK_SUCCESS_CURRICULUM})
        )
        return trial_dir

    @staticmethod
    def _judge(trial_dir):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate
        from environments.shared.result_bundle import read_gate_verdict

        # A sweep trial's stage_results come from evaluations.npz: the argmax
        # EvalCallback panel, not the handoff pair's panel.
        stage_results = {"best_eval_reward": 700.0, "best_eval_length": 1000.0}
        _apply_stage_gate(
            stage=3,
            stage_config={"name": "Hunt", "description": "Hunt", "curriculum_kwargs": dict(TASK_SUCCESS_CURRICULUM)},
            stage_results=stage_results,
            stance_report=None,
            stage_dir=trial_dir,
            species="trex",
        )
        verdict = read_gate_verdict(trial_dir)
        assert verdict is not None
        return verdict, stage_results

    @pytest.mark.parametrize("successes", [[True] * 20 + [False] * 10, [True] * 19 + [False] * 11])
    def test_the_row_verdict_equals_the_on_disk_verdict(self, tmp_path, monkeypatch, successes):
        trial_dir = self._trial(tmp_path, successes, monkeypatch=monkeypatch)
        verdict, stage_results = self._judge(trial_dir)
        rows = collect_results_from_disk(tmp_path)
        assert rows[0]["gate_evaluable"] is True
        assert rows[0]["stage_passed"] == verdict["passed"] == (sum(successes) >= 20)
        assert verdict["stage_result"]["best_model_success_count"] == sum(successes)
        assert verdict["stage_result"]["best_model_n_episodes"] == 30
        # The verdict describes the ONE panel: its reward, not the npz best.
        assert verdict["stage_result"]["best_model_reward"] == 600.0 == stage_results["best_model_reward"]
        assert (rows[0]["success_count"], rows[0]["selected_mean_reward"]) == (sum(successes), 600.0)

    def test_the_rail_agrees_on_the_same_panel(self, tmp_path, monkeypatch):
        """A handoff panel below the rail fails BOTH verdicts although the npz best (700) clears it."""
        trial_dir = self._trial(tmp_path, [True] * 30, reward=300.0, monkeypatch=monkeypatch)
        verdict, _ = self._judge(trial_dir)
        rows = collect_results_from_disk(tmp_path)
        assert verdict["passed"] is False and rows[0]["stage_passed"] is False
        assert any("task_success rail" in failure for failure in verdict["failures"])
        assert "task_success rail" in rows[0]["gate_reason"]

    def test_a_panel_that_did_not_evaluate_the_handoff_pair_is_unjudged_on_both_sides(
        self, tmp_path, monkeypatch, caplog
    ):
        """The smoke-trial shape (no checkpoint saved, final_model evaluated): the worker records no
        count, so the row is 'not evaluable' beside the refused verdict — never PASS beside FAIL."""
        import logging

        with caplog.at_level(logging.WARNING):
            trial_dir = self._trial(tmp_path, [True] * 30, handoff=False, monkeypatch=monkeypatch)
        assert "NOT recorded as a task_success/v1 count" in caplog.text
        metrics = json.loads((trial_dir / "metrics.json").read_text())
        assert "success_count" not in metrics and "selected_mean_reward" not in metrics
        assert not (trial_dir / "evaluation_selected.csv").exists()
        verdict, _ = self._judge(trial_dir)
        assert verdict["passed"] is False
        rows = collect_results_from_disk(tmp_path)
        assert rows[0]["stage_passed"] is None and rows[0]["gate_evaluable"] is False
        assert _gate_status(rows[0]) == "not evaluable"


class TestNotEvaluableConsequences:
    def test_gate_status_labels(self):
        assert _gate_status({"stage_passed": True}) == "passed"
        assert _gate_status({"stage_passed": False}) == "failed"
        assert _gate_status({"stage_passed": None, "gate_evaluable": False}) == "not evaluable"
        assert _gate_status({}) == "not evaluable"
        # CSV cells arrive as strings.
        assert _gate_status({"stage_passed": "True", "gate_evaluable": "True"}) == "passed"
        assert _gate_status({"stage_passed": "False", "gate_evaluable": "True"}) == "failed"
        assert _gate_status({"stage_passed": "", "gate_evaluable": "False"}) == "not evaluable"

    def test_best_trial_model_path_refuses_not_evaluable_rows(self):
        rows = [_unevaluable_row("statue", 3271.8), _unevaluable_row("chatterer", 2133.4)]
        with pytest.raises(SweepStageError, match="cannot evaluate offline"):
            _best_trial_model_path(rows, "b", "trex", 1)

    def test_best_trial_model_path_ignores_not_evaluable_rows_beside_passed_ones(self):
        rows = [_unevaluable_row("statue", 3271.8), {"trial_id": "ok", "stage_passed": True, "best_mean_reward": 10.0}]
        _, best = _best_trial_model_path(rows, "b", "trex", 1)
        assert best["trial_id"] == "ok"

    def test_best_trial_model_path_any_is_reward_ranked(self):
        rows = [_unevaluable_row("chatterer", 2133.4), _unevaluable_row("statue", 3271.8)]
        _, best = _best_trial_model_path_any(rows, "b", "trex", 1)
        assert best["trial_id"] == "statue"

    def test_csv_round_trip_and_plot(self, tmp_path):
        rows = [
            _unevaluable_row("statue", 3271.8),
            {
                "trial_id": "ok",
                "stage": 2,
                "best_mean_reward": 150.0,
                "reward_threshold": 100.0,
                "gate_kind": "reward_and_length/v1",
                "gate_evaluable": True,
                "stage_passed": True,
                "gate_reason": "",
            },
        ]
        csv_path = tmp_path / "results.csv"
        write_results_csv(rows, csv_path)
        with open(csv_path, newline="") as f:
            by_id = {r["trial_id"]: r for r in csv.DictReader(f)}
        assert by_id["statue"]["stage_passed"] == ""
        assert by_id["statue"]["gate_evaluable"] == "False"
        assert _gate_status(by_id["statue"]) == "not evaluable"
        assert _gate_status(by_id["ok"]) == "passed"

        pytest.importorskip("matplotlib")
        plot_sweep_results(csv_path, "trex", "ppo", save_dir=tmp_path)
        assert (tmp_path / "sweep_trial_metrics.png").exists()

    def test_state_json_round_trip(self):
        restored = json.loads(json.dumps([_unevaluable_row("statue", 3271.8)]))
        assert restored[0]["stage_passed"] is None
        assert _gate_status(restored[0]) == "not evaluable"


# ── _best_trial_model_path ───────────────────────────────────────────────


class TestBestTrialModelPath:
    def test_selects_highest_reward_among_passed(self):
        rows = [
            {"trial_id": "1", "stage_passed": True, "best_mean_reward": 100.0},
            {"trial_id": "2", "stage_passed": True, "best_mean_reward": 200.0},
            {"trial_id": "3", "stage_passed": False, "best_mean_reward": 300.0},
        ]
        path, best_row = _best_trial_model_path(rows, "my-bucket", "velociraptor", 1)
        assert best_row["trial_id"] == "2"
        assert "my-bucket" in path
        assert "stage1" in path
        assert "/2/" in path

    def test_raises_when_no_trials_pass(self):
        rows = [
            {"trial_id": "1", "stage_passed": False, "best_mean_reward": 50.0},
        ]
        with pytest.raises(SweepStageError):
            _best_trial_model_path(rows, "bucket", "trex", 2)

    def test_gcs_path_format(self):
        rows = [{"trial_id": "7", "stage_passed": True, "best_mean_reward": 150.0}]
        path, _ = _best_trial_model_path(rows, "my-bucket", "velociraptor", 2)
        assert path == "/gcs/my-bucket/sweeps/velociraptor/stage2/7/models/best_model.zip"


class TestBestTrialModelPathWithModelPath:
    def test_uses_precomputed_model_path(self):
        rows = [
            {
                "trial_id": "5",
                "stage_passed": True,
                "best_mean_reward": 200.0,
                "model_path": "/gcs/bucket/sweeps/velociraptor/stage2_r1/5/models/best_model.zip",
            },
        ]
        path, best_row = _best_trial_model_path(rows, "bucket", "velociraptor", 2)
        assert path == "/gcs/bucket/sweeps/velociraptor/stage2_r1/5/models/best_model.zip"

    def test_falls_back_to_constructed_path(self):
        """Rows without model_path still get the default constructed path."""
        rows = [
            {"trial_id": "3", "stage_passed": True, "best_mean_reward": 100.0},
        ]
        path, _ = _best_trial_model_path(rows, "my-bucket", "trex", 1)
        assert path == "/gcs/my-bucket/sweeps/trex/stage1/3/models/best_model.zip"

    def test_mixed_rows_picks_best_with_correct_path(self):
        """When rows from different runs are merged, the best trial's model_path is used."""
        rows = [
            # From original run (no model_path — constructed by default)
            {"trial_id": "1", "stage_passed": True, "best_mean_reward": 100.0},
            # From resumed run (has model_path)
            {
                "trial_id": "1",
                "stage_passed": True,
                "best_mean_reward": 250.0,
                "model_path": "/gcs/bucket/sweeps/velociraptor/stage1_r1/1/models/best_model.zip",
            },
        ]
        path, best_row = _best_trial_model_path(rows, "bucket", "velociraptor", 1)
        assert best_row["best_mean_reward"] == 250.0
        assert path == "/gcs/bucket/sweeps/velociraptor/stage1_r1/1/models/best_model.zip"


class TestBestTrialModelPathStage3:
    def test_stage3_best_trial_identified(self):
        rows = [
            {"trial_id": "1", "stage_passed": True, "best_mean_reward": 100.0},
            {"trial_id": "2", "stage_passed": True, "best_mean_reward": 200.0},
        ]
        path, best_row = _best_trial_model_path(rows, "bucket", "velociraptor", 3)
        assert best_row["trial_id"] == "2"
        assert "stage3" in path


# ── write_results_csv ────────────────────────────────────────────────────


class TestWriteResultsCsv:
    def test_writes_csv(self, tmp_path):
        rows = [
            {
                "trial_id": "1",
                "stage": 1,
                "ppo_learning_rate": 0.0003,
                "best_mean_reward": 100.0,
                "best_mean_episode_length": 500.0,
                "last_mean_reward": 95.0,
                "last_mean_episode_length": 480.0,
                "reward_threshold": 80.0,
                "ep_length_threshold": None,
                "forward_vel_threshold": None,
                "success_rate_threshold": None,
                "stage_passed": True,
            }
        ]
        csv_path = tmp_path / "results.csv"
        result = write_results_csv(rows, csv_path)
        assert result == csv_path
        assert csv_path.exists()

        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            written = list(reader)
        assert len(written) == 1
        assert written[0]["trial_id"] == "1"
        assert written[0]["ppo_learning_rate"] == "0.0003"

    def test_empty_rows_skipped(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        write_results_csv([], csv_path)
        assert not csv_path.exists()


# ── collect_results_from_disk ───────────────────────────────────────────


class TestCollectResultsFromDisk:
    def test_sweep_layout(self, tmp_path):
        """Collects results from sweep-style trial directories."""
        _setup_sweep_dir(
            tmp_path,
            1,
            {
                "1": {"best_mean_reward": 150.0, "best_mean_episode_length": 500.0},
                "2": {"best_mean_reward": 80.0, "best_mean_episode_length": 300.0},
            },
        )
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 2
        rewards = {r["trial_id"]: r["best_mean_reward"] for r in rows}
        assert rewards["1"] == 150.0
        assert rewards["2"] == 80.0

    def test_curriculum_layout(self, tmp_path):
        """Collects results from curriculum-style single-trial directories."""
        for stage in (1, 2):
            stage_dir = tmp_path / f"stage{stage}"
            stage_dir.mkdir(parents=True)
            with open(stage_dir / "metrics.json", "w") as f:
                json.dump({"best_mean_reward": 100.0 * stage}, f)
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 2
        assert rows[0]["stage"] == 1
        assert rows[1]["stage"] == 2

    def test_stage_filter(self, tmp_path):
        """Only collects from specified stages."""
        _setup_sweep_dir(tmp_path, 1, {"1": {"best_mean_reward": 100.0}})
        _setup_sweep_dir(tmp_path, 2, {"1": {"best_mean_reward": 200.0}})
        rows = collect_results_from_disk(tmp_path, stages=[2])
        assert len(rows) == 1
        assert rows[0]["stage"] == 2

    def test_thresholds_from_stage_config(self, tmp_path):
        """Loads curriculum thresholds from stage_config.json."""
        stage_dir = _setup_sweep_dir(
            tmp_path,
            1,
            {
                "1": {"best_mean_reward": 50.0},
            },
        )
        cfg = {"curriculum": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 100.0}}
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 1
        assert rows[0]["stage_passed"] is False
        assert rows[0]["reward_threshold"] == 100.0

    def test_passing_trial_with_threshold(self, tmp_path):
        """Trial passes when reward exceeds threshold."""
        stage_dir = _setup_sweep_dir(
            tmp_path,
            1,
            {
                "1": {"best_mean_reward": 150.0},
            },
        )
        cfg = {"curriculum": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 100.0}}
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        assert rows[0]["stage_passed"] is True

    def test_empty_directory(self, tmp_path):
        """Returns empty list for directory with no stage sub-dirs."""
        rows = collect_results_from_disk(tmp_path)
        assert rows == []

    def test_nonexistent_directory(self, tmp_path):
        """Returns empty list for nonexistent directory."""
        rows = collect_results_from_disk(tmp_path / "does_not_exist")
        assert rows == []

    def test_multi_stage_collection(self, tmp_path):
        """Collects across multiple stages into a single list."""
        _setup_sweep_dir(
            tmp_path,
            1,
            {
                "1": {"best_mean_reward": 100.0},
                "2": {"best_mean_reward": 120.0},
            },
        )
        _setup_sweep_dir(
            tmp_path,
            2,
            {
                "1": {"best_mean_reward": 200.0},
            },
        )
        _setup_sweep_dir(
            tmp_path,
            3,
            {
                "1": {"best_mean_reward": 300.0},
            },
        )
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 4
        stages = [r["stage"] for r in rows]
        assert stages.count(1) == 2
        assert stages.count(2) == 1
        assert stages.count(3) == 1

    def test_hyperparameters_from_stage_config(self, tmp_path):
        """Includes algorithm hyperparameters and reward weights from stage_config.json."""
        stage_dir = _setup_sweep_dir(
            tmp_path,
            1,
            {"1": {"best_mean_reward": 100.0, "best_mean_episode_length": 400.0}},
        )
        cfg = {
            "algorithm": "PPO",
            "hyperparameters": {
                "learning_rate": 0.0003,
                "batch_size": 128,
                "n_epochs": 10,
                "policy_kwargs": {"net_arch": [256, 256]},
            },
            "reward_weights": {
                "alive_bonus": 2.0,
                "energy_penalty_weight": 0.05,
            },
            "curriculum": {"min_avg_reward": 50.0},
        }
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        # Algorithm hyperparameters prefixed with algorithm name
        assert row["ppo_learning_rate"] == 0.0003
        assert row["ppo_batch_size"] == 128
        assert row["ppo_n_epochs"] == 10
        # Nested policy_kwargs flattened
        assert row["ppo_policy_kwargs_net_arch"] == [256, 256]
        # Reward weights prefixed with env_
        assert row["env_alive_bonus"] == 2.0
        assert row["env_energy_penalty_weight"] == 0.05

    def test_hyperparameters_from_per_trial_config(self, tmp_path):
        """Falls back to per-trial stage_config.json for hyperparameters."""
        stage_dir = tmp_path / "stage1"
        trial_dir = stage_dir / "1"
        trial_dir.mkdir(parents=True)
        with open(trial_dir / "metrics.json", "w") as f:
            json.dump({"best_mean_reward": 100.0}, f)
        cfg = {
            "algorithm": "SAC",
            "hyperparameters": {"learning_rate": 0.001, "gamma": 0.99},
            "reward_weights": {"alive_bonus": 1.5},
            "curriculum": {"min_avg_reward": 50.0},
        }
        with open(trial_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["sac_learning_rate"] == 0.001
        assert row["sac_gamma"] == 0.99
        assert row["env_alive_bonus"] == 1.5

    def test_hyperparameters_in_csv_output(self, tmp_path):
        """Hyperparameters appear as columns in the written CSV."""
        stage_dir = _setup_sweep_dir(
            tmp_path,
            1,
            {"1": {"best_mean_reward": 100.0}},
        )
        cfg = {
            "algorithm": "PPO",
            "hyperparameters": {"learning_rate": 0.0003, "batch_size": 256},
            "reward_weights": {"alive_bonus": 2.0},
            "curriculum": {"min_avg_reward": 50.0},
        }
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump(cfg, f)
        rows = collect_results_from_disk(tmp_path)
        csv_path = tmp_path / "collected_results.csv"
        write_results_csv(rows, csv_path)
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 1
        assert "ppo_learning_rate" in csv_rows[0]
        assert "ppo_batch_size" in csv_rows[0]
        assert "env_alive_bonus" in csv_rows[0]
        assert csv_rows[0]["ppo_learning_rate"] == "0.0003"
        assert csv_rows[0]["ppo_batch_size"] == "256"
        assert csv_rows[0]["env_alive_bonus"] == "2.0"


# ── collect_results_from_disk with gs:// URIs ──────────────────────────


class TestCollectResultsFromDiskGCS:
    def _gcs_modules(self, mock_storage):
        """Build sys.modules dict for mocking google.cloud.storage."""
        mock_gc = MagicMock()
        mock_gc.storage = mock_storage
        return {
            "google": MagicMock(),
            "google.cloud": mock_gc,
            "google.cloud.storage": mock_storage,
        }

    def test_gs_uri_downloads_and_collects(self):
        """gs:// URI triggers GCS download and produces correct rows."""
        metrics_1 = json.dumps({"best_mean_reward": 150.0}).encode()
        metrics_2 = json.dumps({"best_mean_reward": 200.0}).encode()
        stage_cfg = json.dumps({"curriculum": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 100.0}}).encode()

        file_map = {
            "sweeps/velociraptor/stage1/1/metrics.json": metrics_1,
            "sweeps/velociraptor/stage1/stage_config.json": stage_cfg,
            "sweeps/velociraptor/stage2/1/metrics.json": metrics_2,
            # Also include a non-JSON file that should be skipped
            "sweeps/velociraptor/stage1/1/best_model.zip": b"fake_model",
        }
        mock_blobs = _make_mock_gcs_blobs(file_map)

        mock_storage = MagicMock()
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = mock_blobs

        with patch.dict("sys.modules", self._gcs_modules(mock_storage)):
            rows = collect_results_from_disk(
                "gs://my-bucket/sweeps/velociraptor",
                species="velociraptor",
            )

        assert len(rows) == 2
        stage_nums = {r["stage"] for r in rows}
        assert stage_nums == {1, 2}
        # Stage 1 has a stage_config.json with threshold
        s1_row = [r for r in rows if r["stage"] == 1][0]
        assert s1_row["reward_threshold"] == 100.0
        assert s1_row["stage_passed"] is True

    def test_gs_uri_cleans_up_tempdir_on_success(self):
        """Temp directory is cleaned up after successful collection."""
        metrics = json.dumps({"best_mean_reward": 100.0}).encode()
        file_map = {"prefix/stage1/1/metrics.json": metrics}
        mock_blobs = _make_mock_gcs_blobs(file_map)

        mock_storage = MagicMock()
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = mock_blobs

        created_tmpdirs = []
        original_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        with (
            patch.dict("sys.modules", self._gcs_modules(mock_storage)),
            patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp),
        ):
            rows = collect_results_from_disk("gs://bucket/prefix")

        assert len(rows) == 1
        # Temp dir should have been cleaned up
        assert len(created_tmpdirs) == 1
        assert not Path(created_tmpdirs[0]).exists()

    def test_gs_uri_cleans_up_tempdir_on_error(self):
        """Temp directory is cleaned up even if GCS download fails."""
        mock_storage = MagicMock()
        mock_storage.Client.side_effect = Exception("auth failed")

        created_tmpdirs = []
        original_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        with (
            patch.dict("sys.modules", self._gcs_modules(mock_storage)),
            patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp),
        ):
            with pytest.raises(Exception, match="auth failed"):
                collect_results_from_disk("gs://bucket/prefix")

        assert len(created_tmpdirs) == 1
        assert not Path(created_tmpdirs[0]).exists()


class TestPositionPrefixedStageDirs:
    """collect-results must see NN_id stage dirs (run layout of 2026-08-20)."""

    def test_curriculum_layout_collects_from_nn_id_dirs(self, tmp_path):
        import json

        from environments.shared.scripts.sweep.results import collect_results_from_disk

        for name, stage in (("01_stance", 1), ("03_locomotion", 2)):
            stage_dir = tmp_path / name
            stage_dir.mkdir()
            (stage_dir / "metrics.json").write_text(json.dumps({"mean_reward": 10.0 * stage}))
            (stage_dir / "stage_config.json").write_text(json.dumps({"curriculum": {"timesteps": 1000}}))
        rows = collect_results_from_disk(tmp_path)
        # The legacy number comes from the id suffix, never the digits:
        # 03_locomotion is legacy stage 2 at manifest position 3.
        assert sorted(row["stage"] for row in rows) == [1, 2]

    def test_stage_filter_applies_to_nn_id_dirs(self, tmp_path):
        import json

        from environments.shared.scripts.sweep.results import collect_results_from_disk

        for name in ("01_stance", "03_locomotion"):
            stage_dir = tmp_path / name
            stage_dir.mkdir()
            (stage_dir / "metrics.json").write_text(json.dumps({"mean_reward": 1.0}))
        rows = collect_results_from_disk(tmp_path, stages=[2])
        assert [row["stage"] for row in rows] == [2]

    def test_semantic_stage_dirs_collect_under_their_id(self, tmp_path):
        """The recovery stage has no legacy number, so its rows carry the id.

        Both on-disk generations a semantic stage has written must collect:
        the bare id ("recovery", pre-2026-08-20 stage_label layout) and the
        position-prefixed NN_id form — and the reference is always the id,
        never a number minted from the NN position digits.
        """
        import json

        from environments.shared.scripts.sweep.results import collect_results_from_disk

        for name in ("01_stance", "02_recovery"):
            stage_dir = tmp_path / "nn_layout" / name
            stage_dir.mkdir(parents=True)
            (stage_dir / "metrics.json").write_text(json.dumps({"mean_reward": 1.0}))
        rows = collect_results_from_disk(tmp_path / "nn_layout")
        assert sorted(str(row["stage"]) for row in rows) == ["1", "recovery"]

        bare_dir = tmp_path / "bare_layout" / "recovery"
        bare_dir.mkdir(parents=True)
        (bare_dir / "metrics.json").write_text(json.dumps({"mean_reward": 2.0}))
        rows = collect_results_from_disk(tmp_path / "bare_layout")
        assert [row["stage"] for row in rows] == ["recovery"]

    def test_stage_filter_accepts_semantic_references(self, tmp_path):
        import json

        from environments.shared.scripts.sweep.results import collect_results_from_disk

        for name in ("01_stance", "02_recovery"):
            stage_dir = tmp_path / name
            stage_dir.mkdir()
            (stage_dir / "metrics.json").write_text(json.dumps({"mean_reward": 1.0}))
        rows = collect_results_from_disk(tmp_path, stages=["recovery"])
        assert [row["stage"] for row in rows] == ["recovery"]

    def test_an_ancestors_directory_is_not_a_stage(self, tmp_path):
        """A run reusing certified ancestors carries ancestors/<id>/ — records, never results."""
        import json

        from environments.shared.scripts.sweep.results import collect_results_from_disk

        for name in ("01_stance", "ancestors"):
            stage_dir = tmp_path / name
            stage_dir.mkdir()
            (stage_dir / "metrics.json").write_text(json.dumps({"mean_reward": 1.0}))
        (tmp_path / "ancestors" / "stance").mkdir()
        (tmp_path / "ancestors" / "stance" / "metrics.json").write_text(json.dumps({"mean_reward": 9.0}))
        rows = collect_results_from_disk(tmp_path)
        assert [row["stage"] for row in rows] == [1]

    def test_an_unreserved_nn_dir_is_skipped_species_free(self, tmp_path):
        """The collector holds no trustworthy species, so only reserved ids collect (D-A12)."""
        import json

        from environments.shared.scripts.sweep.results import collect_results_from_disk

        for name in ("01_stance", "05_follow_direction", "05_experiments", "models"):
            stage_dir = tmp_path / name
            stage_dir.mkdir()
            (stage_dir / "metrics.json").write_text(json.dumps({"mean_reward": 1.0}))
        rows = collect_results_from_disk(tmp_path)
        assert [row["stage"] for row in rows] == [1]
        assert collect_results_from_disk(tmp_path, stages=["follow_direction"]) == []
