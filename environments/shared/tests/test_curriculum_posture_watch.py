"""Tests for environments.shared.curriculum.posture_watch, the posture-drift advisory.

The T. rex physics-r10 screen's mid-run telemetry (2026-10-16): seeds 52 and
60 drifted onto their toes by 2.3 to 2.5M steps while seed 42 stood clean,
and nothing in the training output said so.  The watch reads each declared
fore-aft rail's over-bar fraction from ``gate_progress.npz`` and warns, at
most once per ten evaluations, after three in a row at 0.5 or more; it never
stops or changes training.
"""

from __future__ import annotations

import logging
import math

import pytest

from environments.shared.config import load_stage_config
from environments.shared.curriculum import posture_watch
from environments.shared.curriculum.gate_schema import _DIAGNOSTIC_KEYS, gate_config_view, validate_gate_config
from environments.shared.curriculum.posture_watch import (
    POSTURE_DRIFT_KEYS,
    PostureDriftWatchCallback,
    build_posture_drift_watch_callback,
    declared_fore_aft_rails,
    posture_drift_settings,
)
from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
EVAL = 50_000


class _Plateau:
    """The two accessors the watch reads, fed one evaluation at a time."""

    def __init__(self, key: str = "max_foot_cop_fore_aft_over_bar_fraction") -> None:
        self.key = key
        self.timesteps: list[int] = []
        self.fractions: list[float] = []
        self.row: dict[str, float] = {}

    def add(self, fraction: float) -> None:
        self.timesteps.append(EVAL * (len(self.timesteps) + 1))
        self.fractions.append(fraction)

    def gate_progress_series(self, key: str) -> tuple[list[int], list[float]]:
        if key == self.key:
            return list(self.timesteps), list(self.fractions)
        return list(self.timesteps), [math.nan] * len(self.timesteps)

    def gate_progress_row(self) -> dict[str, float]:
        return dict(self.row)


def _watch(plateau: _Plateau, *, rails=(("max_foot_cop_fore_aft", 0.8),), consecutive=3, repeat=10):
    # Built without __init__ so the test needs no stable-baselines3 model.
    watch = object.__new__(PostureDriftWatchCallback)
    watch.plateau_callback = plateau
    watch.rails = rails
    watch.over_bar_fraction = 0.5
    watch.consecutive_evals = consecutive
    watch.repeat_evals = repeat
    return watch


def _replay(fractions, **kwargs) -> tuple[PostureDriftWatchCallback, list[bool]]:
    plateau = _Plateau()
    watch = _watch(plateau, **kwargs)
    results = []
    for fraction in fractions:
        plateau.add(fraction)
        results.append(watch._on_step())
    return watch, results


def _warned_at(watch: PostureDriftWatchCallback) -> list[int]:
    return [warning["timesteps"] for warning in watch.__dict__.get("progress_warnings", [])]


class TestSettings:
    def test_the_defaults(self):
        assert posture_drift_settings({}) == (0.5, 3, 10)

    def test_the_knobs(self):
        block = {
            "posture_drift_warn_over_bar_fraction": 0.25,
            "posture_drift_warn_consecutive_evals": 2,
            "posture_drift_warn_repeat_evals": 20,
        }
        assert posture_drift_settings(block) == (0.25, 2, 20)

    @pytest.mark.parametrize(
        ("block", "message"),
        [
            ({"posture_drift_warn_over_bar_fraction": 0.0}, "posture_drift_warn_over_bar_fraction"),
            ({"posture_drift_warn_over_bar_fraction": 1.5}, "posture_drift_warn_over_bar_fraction"),
            ({"posture_drift_warn_over_bar_fraction": True}, "posture_drift_warn_over_bar_fraction"),
            ({"posture_drift_warn_over_bar_fraction": "0.5"}, "posture_drift_warn_over_bar_fraction"),
            ({"posture_drift_warn_consecutive_evals": 0}, "posture_drift_warn_consecutive_evals"),
            ({"posture_drift_warn_consecutive_evals": 2.5}, "posture_drift_warn_consecutive_evals"),
            ({"posture_drift_warn_repeat_evals": True}, "posture_drift_warn_repeat_evals"),
        ],
    )
    def test_a_malformed_knob_is_refused(self, block, message):
        with pytest.raises(ValueError, match=message):
            posture_drift_settings(block)

    def test_the_declared_rails(self):
        assert declared_fore_aft_rails({}) == ()
        assert declared_fore_aft_rails({"max_sole_cop_fore_aft": 0.8, "max_foot_cop_fore_aft": 0.75}) == (
            ("max_sole_cop_fore_aft", 0.8),
            ("max_foot_cop_fore_aft", 0.75),
        )
        assert declared_fore_aft_rails({"max_sole_cop_fore_aft": True}) == ()
        stance = load_stage_config(TYRANNOSAURUS_REX, "stance")["curriculum_kwargs"]
        assert declared_fore_aft_rails(stance) == (("max_sole_cop_fore_aft", 0.8),)

    def test_the_knobs_configure_a_diagnostic_in_no_gate_or_hyperparameter_digest(self):
        from environments.shared.config import hyperparameters_sha256

        assert set(POSTURE_DRIFT_KEYS) <= _DIAGNOSTIC_KEYS
        config = load_stage_config(TYRANNOSAURUS_REX, "stance")
        knobs = dict(zip(POSTURE_DRIFT_KEYS, (0.4, 2, 5)))
        widened = {**config, "curriculum_kwargs": {**config["curriculum_kwargs"], **knobs}}
        assert validate_gate_config("stance", widened["curriculum_kwargs"]) == "stance_quality_heading/v1"
        assert gate_config_view(widened["curriculum_kwargs"]) == gate_config_view(config["curriculum_kwargs"])
        for algorithm in ("ppo", "sac"):
            assert hyperparameters_sha256(widened, algorithm) == hyperparameters_sha256(config, algorithm)

    def test_no_committed_stage_sets_a_knob(self):
        from environments.shared.config import load_all_stages
        from environments.shared.species_names import model_ids

        for model_id in model_ids():
            for config in load_all_stages(model_id).values():
                assert not set(POSTURE_DRIFT_KEYS) & set(config.get("curriculum_kwargs", {})), model_id


class TestTheWarning:
    def test_three_drifting_evaluations_in_a_row_warn_once_and_training_continues(self, caplog):
        with caplog.at_level(logging.WARNING, logger=posture_watch.__name__):
            watch, results = _replay([0.2, 0.6, 0.7, 0.8])
        assert all(results)  # never stops training
        assert _warned_at(watch) == [4 * EVAL]
        (line,) = [message for message in caplog.messages if "PostureDrift" in message]
        assert "max_foot_cop_fore_aft" in line and "0.60, 0.70, 0.80" in line and "\n" not in line
        assert "Advisory only" in line
        (warning,) = watch.progress_warnings
        assert warning["source"] == "posture_drift" and warning["rails"] == ["max_foot_cop_fore_aft"]
        assert warning["message"] == line

    def test_a_fraction_at_the_threshold_counts(self):
        watch, _ = _replay([0.5, 0.5, 0.5])
        assert _warned_at(watch) == [3 * EVAL]

    @pytest.mark.parametrize(
        "fractions",
        [
            [0.6, 0.7, 0.49, 0.8, 0.9],  # one evaluation under the threshold breaks the run
            [0.6, 0.7, math.nan, 0.8, 0.9],  # so does one with no full-horizon episode to measure
            [0.9, 0.9],  # two are not three
            [0.0] * 30,
        ],
    )
    def test_no_warning_without_three_in_a_row(self, fractions):
        watch, results = _replay(fractions)
        assert all(results) and _warned_at(watch) == []

    def test_a_lasting_drift_repeats_at_most_once_per_ten_evaluations(self):
        watch, results = _replay([0.9] * 25)
        assert all(results)
        assert _warned_at(watch) == [3 * EVAL, 13 * EVAL, 23 * EVAL]

    def test_a_drift_that_breaks_and_reforms_waits_out_the_ten_evaluations(self):
        fractions = [0.9] * 3 + [0.1] + [0.9] * 3 + [0.1] * 5 + [0.9] * 3
        watch, _ = _replay(fractions)
        # Warned at the 3rd; the run re-forms by the 7th, inside ten; it re-forms again by the 15th, past them.
        assert _warned_at(watch) == [3 * EVAL, 15 * EVAL]

    def test_the_line_names_each_foots_edge(self, caplog):
        plateau = _Plateau()
        plateau.row = {"r_foot_cop_fore_aft": 1.12, "l_foot_cop_fore_aft": -0.31, "r_foot_cop_fore_aft_gate": 1.12}
        watch = _watch(plateau)
        with caplog.at_level(logging.WARNING, logger=posture_watch.__name__):
            for fraction in (0.6, 0.7, 0.8):
                plateau.add(fraction)
                watch._on_step()
        assert "r +1.12 (toe side), l -0.31 (heel side)" in caplog.text

    def test_the_pad_rail_reads_its_own_series(self):
        plateau = _Plateau(key="max_sole_cop_fore_aft_over_bar_fraction")
        watch = _watch(plateau, rails=(("max_sole_cop_fore_aft", 0.8), ("max_foot_cop_fore_aft", 0.8)))
        for _ in range(3):
            plateau.add(0.75)
            watch._on_step()
        assert watch.progress_warnings[0]["rails"] == ["max_sole_cop_fore_aft"]

    def test_a_resumed_watch_starts_past_the_seeded_rows(self):
        plateau = _Plateau()
        for _ in range(5):
            plateau.add(0.9)
        watch = _watch(plateau)
        watch._last_seen = 5  # what seed_resume_eval_state sets
        plateau.add(0.9)
        watch._on_step()
        assert _warned_at(watch) == []


class TestBuild:
    def test_built_only_where_a_fore_aft_bar_is_declared(self):
        assert build_posture_drift_watch_callback(_Plateau(), {"curriculum_kwargs": {}}) is None
        velociraptor = load_stage_config(ModelId("velociraptor_mongoliensis", "anatomical"), "stance")
        assert build_posture_drift_watch_callback(_Plateau(), velociraptor) is None
        assert build_posture_drift_watch_callback(None, load_stage_config(TYRANNOSAURUS_REX, "stance")) is None

    def test_a_malformed_knob_is_refused_where_no_bar_is_declared_too(self):
        with pytest.raises(ValueError, match="posture_drift_warn_repeat_evals"):
            build_posture_drift_watch_callback(
                _Plateau(), {"curriculum_kwargs": {"posture_drift_warn_repeat_evals": 0}}
            )

    def test_the_tyrannosaurus_rex_stance_watches_its_pad_rail_and_trains_with_it(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        from environments.shared.eval_diagnostics import StageGatePlateauCallback
        from environments.shared.tests.test_curriculum_hop_watch import TestTrainerWiring

        watch = build_posture_drift_watch_callback(_Plateau(), load_stage_config(TYRANNOSAURUS_REX, "stance"))
        assert isinstance(watch, PostureDriftWatchCallback)
        assert watch.rails == (("max_sole_cop_fore_aft", 0.8),)
        assert (watch.over_bar_fraction, watch.consecutive_evals, watch.repeat_evals) == (0.5, 3, 10)
        callbacks = TestTrainerWiring._callbacks(tmp_path, monkeypatch, load_stage_config(TYRANNOSAURUS_REX, "stance"))
        kinds = [type(callback) for callback in callbacks]
        assert kinds.count(PostureDriftWatchCallback) == 1
        # After the plateau callback, so each evaluation is recorded before the watch reads it.
        assert kinds.index(StageGatePlateauCallback) < kinds.index(PostureDriftWatchCallback)
        assert (
            callbacks[kinds.index(PostureDriftWatchCallback)].plateau_callback
            is callbacks[kinds.index(StageGatePlateauCallback)]
        )

    def test_a_resume_starts_the_watch_past_the_seeded_gate_progress_rows(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        import numpy as np

        from environments.shared.curriculum.checkpoints import seed_resume_eval_state
        from environments.shared.eval_diagnostics import StageGatePlateauCallback

        np.savez(
            tmp_path / "evaluations.npz",
            timesteps=np.array([EVAL, 2 * EVAL]),
            results=np.array([[1.0], [1.0]]),
            ep_lengths=np.array([[1000], [1000]]),
        )
        np.savez(
            tmp_path / "gate_progress.npz",
            timesteps=np.array([EVAL, 2 * EVAL]),
            max_sole_cop_fore_aft_over_bar_fraction=np.array([0.9, 0.9]),
        )
        plateau = object.__new__(StageGatePlateauCallback)
        watch = _watch(plateau)  # type: ignore[arg-type]
        eval_callback = type("Eval", (), {"n_eval_episodes": 1})()
        seed_resume_eval_state(eval_callback, [plateau, watch], tmp_path / "evaluations.npz")
        assert watch._last_seen == 2
