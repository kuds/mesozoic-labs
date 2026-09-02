"""Tests for environments.shared.curriculum.baseline_watch.

Regression for issue #486: run 20260804_143747 trained T-Rex stage 1 for its
full 10,002,432 steps and finished at 2686.9 against the zero-action statue's
3271.8 -- below the do-nothing policy on every major reward term, for the
entire 8h 13m -- and nothing in the run reported it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from environments.shared.curriculum import baseline_watch
from environments.shared.curriculum.baseline_watch import (
    BaselineProgressCallback,
    build_baseline_progress_callback,
    read_zero_action_baseline,
)

#: The real numbers from run 20260804_143747 / its captured baseline.
STATUE = 3271.770493548379
POLICY = 2686.917084553575


def _write_baseline(run_dir: Path, *, species: str = "trex", reward: float = STATUE) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "zero_action_baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema": "mesozoic.zero-action-baseline/v1",
                "stage": 1,
                "results": {species: {"reward_mean": reward, "episodes": 40}},
            }
        ),
        encoding="utf-8",
    )
    return path


class TestReadZeroActionBaseline:
    def test_reads_the_species_reward(self, tmp_path):
        _write_baseline(tmp_path)
        assert read_zero_action_baseline(tmp_path, "trex") == pytest.approx(STATUE)

    def test_missing_file_is_none_not_a_guess(self, tmp_path):
        assert read_zero_action_baseline(tmp_path, "trex") is None

    def test_other_species_is_none(self, tmp_path):
        _write_baseline(tmp_path, species="velociraptor")
        assert read_zero_action_baseline(tmp_path, "trex") is None

    def test_corrupt_file_is_none_rather_than_raising(self, tmp_path):
        (tmp_path / "zero_action_baseline.json").write_text("{not json", encoding="utf-8")
        assert read_zero_action_baseline(tmp_path, "trex") is None

    @pytest.mark.parametrize("value", [None, "abc", float("nan"), float("inf")])
    def test_unusable_reward_is_none(self, tmp_path, value):
        (tmp_path / "zero_action_baseline.json").write_text(
            json.dumps({"results": {"trex": {"reward_mean": value}}}), encoding="utf-8"
        )
        assert read_zero_action_baseline(tmp_path, "trex") is None


class TestBaselineProgressCallback:
    """`object.__new__` throughout so `_on_step` runs without stable-baselines3."""

    @staticmethod
    def _callback(*, total=10_000_000, fraction=0.35, ceiling=0.02):
        cb = object.__new__(BaselineProgressCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = []
        cb.baseline_reward = STATUE
        cb.total_timesteps = total
        cb.warn_after_budget_fraction = fraction
        cb.duty_ceiling = ceiling
        cb._last_seen_n_evals = 0
        cb._warned_below_baseline = False
        cb._ever_beat_baseline = False
        return cb

    @staticmethod
    def _feed(cb, reward, step):
        cb.eval_callback.evaluations_results = cb.eval_callback.evaluations_results + [[reward]]
        cb.num_timesteps = step
        return cb._on_step()

    def test_reports_each_evaluation_against_the_baseline(self, caplog):
        cb = self._callback()
        with caplog.at_level(logging.INFO, logger=baseline_watch.logger.name):
            self._feed(cb, POLICY, 100_000)
        assert any("zero-action baseline" in r.message for r in caplog.records)

    def test_warns_once_past_the_budget_fraction_when_never_beaten(self, caplog):
        """The signal run 20260804_143747 needed and did not have."""
        cb = self._callback()
        with caplog.at_level(logging.WARNING, logger=baseline_watch.logger.name):
            self._feed(cb, POLICY, 1_000_000)  # 10% of budget: too early
            early = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert not early, "warned before the budget fraction"
            self._feed(cb, POLICY, 4_000_000)  # 40%: past 0.35
            self._feed(cb, POLICY, 6_000_000)
            self._feed(cb, POLICY, 9_000_000)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, "must warn exactly once, not on every evaluation"
        assert "never" in warnings[0].message and "worse-than-trivial" in warnings[0].message

    def test_a_run_that_beats_the_baseline_is_never_warned_about(self, caplog):
        """Falling back under it later is the collapse backstop's job, not this."""
        cb = self._callback()
        with caplog.at_level(logging.WARNING, logger=baseline_watch.logger.name):
            self._feed(cb, STATUE + 100.0, 1_000_000)
            self._feed(cb, POLICY, 5_000_000)
            self._feed(cb, POLICY, 9_500_000)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_it_never_stops_the_run(self):
        """Advisory only: the statue is the stage-1 optimum, so a learning
        policy legitimately sits below it for millions of steps."""
        cb = self._callback()
        for step in (1_000_000, 5_000_000, 9_900_000):
            assert self._feed(cb, POLICY, step) is True

    def test_repeated_step_without_a_new_evaluation_is_a_no_op(self, caplog):
        cb = self._callback()
        self._feed(cb, POLICY, 4_000_000)
        caplog.clear()  # discard the setup evaluation's own records
        with caplog.at_level(logging.INFO, logger=baseline_watch.logger.name):
            cb.num_timesteps = 4_000_001
            cb._on_step()
        assert not caplog.records

    def test_no_evaluations_yet_is_a_no_op(self):
        cb = self._callback()
        cb.num_timesteps = 0
        assert cb._on_step() is True


class TestBuilder:
    def test_returns_none_without_a_baseline_file(self, tmp_path):
        built = build_baseline_progress_callback(
            MagicMock(), run_dir=tmp_path, species="trex", stage_config={"curriculum_kwargs": {}}
        )
        assert built is None

    def test_returns_none_without_a_run_dir(self):
        built = build_baseline_progress_callback(
            MagicMock(), run_dir=None, species="trex", stage_config={"curriculum_kwargs": {}}
        )
        assert built is None

    def test_forwards_the_stage_budget_ceiling_and_override(self, tmp_path, monkeypatch):
        captured = {}

        def _capture(eval_callback, baseline_reward, **kwargs):
            captured["baseline"] = baseline_reward
            captured.update(kwargs)
            return "callback"

        monkeypatch.setattr(baseline_watch, "BaselineProgressCallback", _capture)
        _write_baseline(tmp_path)
        built = build_baseline_progress_callback(
            MagicMock(),
            run_dir=tmp_path,
            species="trex",
            stage_config={
                "curriculum_kwargs": {
                    "timesteps": 10_000_000,
                    "max_unsupported_duty": 0.02,
                    "baseline_warn_after_budget_fraction": 0.5,
                }
            },
        )
        assert built == "callback"
        assert captured["baseline"] == pytest.approx(STATUE)
        assert captured["total_timesteps"] == 10_000_000
        assert captured["duty_ceiling"] == 0.02
        assert captured["warn_after_budget_fraction"] == 0.5

    def test_anchors_on_the_budget_actually_trained_not_the_toml(self, tmp_path, monkeypatch):
        """A --timesteps override or a shortened resume budget must move the
        warning with it; re-reading the TOML mis-timed or suppressed it (TC9)."""
        captured = {}

        def _capture(eval_callback, baseline_reward, **kwargs):
            captured.update(kwargs)
            return "callback"

        monkeypatch.setattr(baseline_watch, "BaselineProgressCallback", _capture)
        _write_baseline(tmp_path)
        build_baseline_progress_callback(
            MagicMock(),
            run_dir=tmp_path,
            species="trex",
            stage_config={"curriculum_kwargs": {"timesteps": 10_000_000}},
            total_timesteps=2_000_000,
        )
        assert captured["total_timesteps"] == 2_000_000


def test_every_trainer_call_site_passes_the_actual_budget():
    """`total_timesteps` is optional (TOML fallback), so a call site that omits
    it silently re-inherits the TC9 mis-anchoring. train() must hand over its
    resume-aware cumulative target, not this call's remaining budget."""
    import ast

    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "environments" / "shared" / "train_base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_build_core_callbacks"
    ]
    assert calls, "expected _build_core_callbacks to be called in train_base.py"
    for call in calls:
        assert any(kw.arg == "total_timesteps" for kw in call.keywords), (
            f"_build_core_callbacks call at train_base.py:{call.lineno} omits total_timesteps=, "
            "so the budget-anchored advisories fall back to the TOML budget on that path"
        )
    train_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    train_call = next(
        node
        for node in ast.walk(train_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_build_core_callbacks"
    )
    passed = next(kw.value for kw in train_call.keywords if kw.arg == "total_timesteps")
    assert isinstance(passed, ast.Name) and passed.id == "target_timesteps"


def test_the_config_key_is_accepted_by_the_gate_schema():
    """An unregistered [curriculum] key is fatal, which would make it unusable."""
    from environments.shared.curriculum.gate_schema import validate_gate_config

    validate_gate_config(
        1,
        {
            "gate_schema_version": 1,
            "gate_kind": "stance_quality/v1",
            "min_full_horizon_fraction": 0.95,
            "max_unsupported_duty": 0.02,
            "max_unsupported_duty_ucb": 0.02,
            "baseline_warn_after_budget_fraction": 0.5,
        },
    )


def test_the_notebook_passes_species_so_the_watch_is_not_dead_code():
    """`species` is optional, so omitting it silently disables the watch.

    That is the failure mode this repository keeps rediscovering — a
    mechanism wired into the library but never reached from the trainer.
    """
    repo_root = Path(__file__).resolve().parents[3]
    notebook = json.loads((repo_root / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code"]
    calls = [c for c in cells if "_build_core_callbacks(" in c]
    assert len(calls) == 1, "expected exactly one _build_core_callbacks call"
    assert "species=SPECIES," in calls[0], (
        "sb3_training.ipynb must pass species= to _build_core_callbacks, or the "
        "zero-action baseline watch is never constructed"
    )


def test_every_trainer_call_site_passes_species():
    """The notebook is not the only caller, and pinning only it missed two.

    `train_stage` and `train_curriculum` both shipped omitting `species`,
    which left the watch dead on both library entry points while the
    notebook test above passed — the *same* dead-code failure that test was
    written to prevent, one call site over. So this asserts over every call
    site in `train_base.py` rather than a hand-listed one.
    """
    import ast

    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "environments" / "shared" / "train_base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_build_core_callbacks"
    ]
    assert calls, "expected _build_core_callbacks to be called in train_base.py"
    for call in calls:
        assert any(kw.arg == "species" for kw in call.keywords), (
            f"_build_core_callbacks call at train_base.py:{call.lineno} omits species=, "
            "so the zero-action baseline watch is never constructed on that path"
        )
