"""Tests for environments.shared.curriculum.posture_watch, the posture-drift advisory.

The T. rex physics-r10 screen's mid-run telemetry (2026-10-16): seeds 52 and
60 drifted onto their toes by 2.3 to 2.5M steps while seed 42 stood clean,
and nothing in the training output said so.  The watch reads each declared
fore-aft rail's over-bar fraction from ``gate_progress.npz`` and warns, at
most once per ten evaluations, after two in a row at 0.5 or more; it never
stops or changes training.  The screen's own series are replayed below: the
defaults warn on seeds 52 and 60 and stay silent on seed 42, and three in a
row warn on the same seeds later (seed 52 1.44M steps later).
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction

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


def _fractions(text: str) -> list[float]:
    """Whitespace-separated over-bar fractions, each a decimal or ``k/n``, as floats."""
    return [float(Fraction(token)) for token in text.split()]


#: The T. rex physics-r10 screen (2026-10-16): its evaluation interval, its whole-foot rail's bar
#: (``max_foot_cop_fore_aft``), and per seed that rail's over-bar fraction at every evaluation from 131,072 steps,
#: ten evaluations to a line, as each run's ``gate_progress.npz`` held it at 23:49 UTC: through 6,029,312 steps on
#: seed 42 and 6,160,384 on seeds 52 and 60.  ``k/n`` is k of the n panel episodes, of ten, that reached the horizon.
R10_EVAL = 131_072
R10_BAR = 0.8
R10_OVER_BAR = {
    42: _fractions(
        """
        0.0 0.1 0.0 0.1 0.1 0.0 0.0 0.3 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 1/9 0.1 0.1 0.0 2/9 0.1
        0.1 0.0 0.0 0.0 2/9 0.1 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        """
    ),
    52: _fractions(
        """
        0.1 0.0 0.3 0.0 0.2 0.6 0.7 0.2 0.2 0.1
        0.6 0.5 0.4 0.9 0.4 0.8 0.9 0.9 1.0 1.0
        7/8 6/9 6/7 6/9 5/8 0.7 0.9 0.6 0.7 1.0
        1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
        1.0 1.0 1.0 1.0 1.0 1.0 1.0
        """
    ),
    60: _fractions(
        """
        0.0 0.1 0.3 0.1 4/9 7/9 1.0 1.0 1.0 1.0
        1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
        1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
        1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
        1.0 1.0 1.0 1.0 1.0 1.0 1.0
        """
    ),
}


class _Plateau:
    """The two accessors the watch reads, fed one evaluation at a time."""

    def __init__(self, key: str = "max_foot_cop_fore_aft_over_bar_fraction", every: int = EVAL) -> None:
        self.key = key
        self.every = every
        self.timesteps: list[int] = []
        self.fractions: list[float] = []
        self.row: dict[str, float] = {}

    def add(self, fraction: float) -> None:
        self.timesteps.append(self.every * (len(self.timesteps) + 1))
        self.fractions.append(fraction)

    def gate_progress_series(self, key: str) -> tuple[list[int], list[float]]:
        if key == self.key:
            return list(self.timesteps), list(self.fractions)
        return list(self.timesteps), [math.nan] * len(self.timesteps)

    def gate_progress_row(self) -> dict[str, float]:
        return dict(self.row)


def _watch(
    plateau: _Plateau,
    *,
    rails=(("max_foot_cop_fore_aft", 0.8),),
    consecutive=posture_watch.DEFAULT_CONSECUTIVE_EVALS,
    repeat=posture_watch.DEFAULT_REPEAT_EVALS,
):
    # Built without __init__ so the test needs no stable-baselines3 model.
    watch = object.__new__(PostureDriftWatchCallback)
    watch.plateau_callback = plateau
    watch.rails = rails
    watch.over_bar_fraction = posture_watch.DEFAULT_OVER_BAR_FRACTION
    watch.consecutive_evals = consecutive
    watch.repeat_evals = repeat
    return watch


def _replay(fractions, *, every: int = EVAL, **kwargs) -> tuple[PostureDriftWatchCallback, list[bool]]:
    plateau = _Plateau(every=every)
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
        assert posture_drift_settings({}) == (0.5, 2, 10)

    def test_the_knobs(self):
        block = {
            "posture_drift_warn_over_bar_fraction": 0.25,
            "posture_drift_warn_consecutive_evals": 3,
            "posture_drift_warn_repeat_evals": 20,
        }
        assert posture_drift_settings(block) == (0.25, 3, 20)

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
    def test_two_drifting_evaluations_in_a_row_warn_once_and_training_continues(self, caplog):
        with caplog.at_level(logging.WARNING, logger=posture_watch.__name__):
            watch, results = _replay([0.2, 0.6, 0.7, 0.8])
        assert all(results)  # never stops training
        assert _warned_at(watch) == [3 * EVAL]
        (line,) = [message for message in caplog.messages if "PostureDrift" in message]
        assert "max_foot_cop_fore_aft" in line and "on 0.60, 0.70 of" in line and "\n" not in line
        assert "on 2 evaluations in a row" in line and "Advisory only" in line
        (warning,) = watch.progress_warnings
        assert warning["source"] == "posture_drift" and warning["rails"] == ["max_foot_cop_fore_aft"]
        assert warning["message"] == line

    def test_a_fraction_at_the_threshold_counts(self):
        watch, _ = _replay([0.5, 0.5])
        assert _warned_at(watch) == [2 * EVAL]

    @pytest.mark.parametrize(
        "fractions",
        [
            [0.6, 0.49, 0.8, 0.49, 0.9],  # one evaluation under the threshold breaks the run
            # so does one with no full-horizon episode to measure: a seed whose episodes all fall goes quiet
            [0.6, math.nan, 0.8, math.nan, 0.9, math.nan],
            [0.9],  # one is not two
            [0.0] * 30,
        ],
    )
    def test_no_warning_without_two_in_a_row(self, fractions):
        watch, results = _replay(fractions)
        assert all(results) and _warned_at(watch) == []

    def test_a_longer_streak_when_configured(self):
        watch, _ = _replay([0.6, 0.7, 0.4, 0.8, 0.9], consecutive=3)
        assert _warned_at(watch) == []
        watch, _ = _replay([0.6, 0.7, 0.8], consecutive=3)
        assert _warned_at(watch) == [3 * EVAL]

    def test_a_lasting_drift_repeats_at_most_once_per_ten_evaluations(self):
        watch, results = _replay([0.9] * 25)
        assert all(results)
        assert _warned_at(watch) == [2 * EVAL, 12 * EVAL, 22 * EVAL]

    def test_a_drift_that_breaks_and_reforms_waits_out_the_ten_evaluations(self):
        fractions = [0.9] * 2 + [0.1] + [0.9] * 2 + [0.1] * 6 + [0.9] * 2
        watch, _ = _replay(fractions)
        # Warned at the 2nd; the run re-forms by the 5th, inside ten; it re-forms again by the 13th, past them.
        assert _warned_at(watch) == [2 * EVAL, 13 * EVAL]

    def test_the_line_names_each_foots_edge(self, caplog):
        plateau = _Plateau()
        plateau.row = {"r_foot_cop_fore_aft": 1.12, "l_foot_cop_fore_aft": -0.31, "r_foot_cop_fore_aft_gate": 1.12}
        watch = _watch(plateau)
        with caplog.at_level(logging.WARNING, logger=posture_watch.__name__):
            for fraction in (0.6, 0.7):
                plateau.add(fraction)
                watch._on_step()
        assert "r +1.12 (toe side), l -0.31 (heel side)" in caplog.text

    def test_the_pad_rail_reads_its_own_series(self):
        plateau = _Plateau(key="max_sole_cop_fore_aft_over_bar_fraction")
        watch = _watch(plateau, rails=(("max_sole_cop_fore_aft", 0.8), ("max_foot_cop_fore_aft", 0.8)))
        for _ in range(2):
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
        assert _warned_at(watch) == []  # one new drifting evaluation, not two
        plateau.add(0.9)
        watch._on_step()
        assert _warned_at(watch) == [7 * EVAL]


class TestTheR10Screen:
    """The physics-r10 screen's own whole-foot series, replayed: what the watch would have printed in training."""

    @staticmethod
    def _replay(seed: int, **kwargs) -> PostureDriftWatchCallback:
        rails = (("max_foot_cop_fore_aft", R10_BAR),)
        watch, results = _replay(R10_OVER_BAR[seed], every=R10_EVAL, rails=rails, **kwargs)
        assert all(results)  # never stops training
        return watch

    def test_the_series_run_to_6_03m_steps_on_seed_42_and_6_16m_on_seeds_52_and_60(self):
        assert {seed: len(series) * R10_EVAL for seed, series in R10_OVER_BAR.items()} == {
            42: 6_029_312,
            52: 6_160_384,
            60: 6_160_384,
        }

    def test_the_defaults_warn_on_both_drifting_seeds_and_never_on_the_clean_one(self):
        assert _warned_at(self._replay(42)) == []
        # Both first warn at 917,504 steps (seed 52 at 0.6 then 0.7, seed 60 at 7/9 then 1.0), and both are
        # drifting again every ten evaluations to the end of the series.
        clause = "over its 0.80 bar on {} of the full-horizon panel episodes at 786,432 to 917,504 steps"
        for seed, fractions in ((52, "0.60, 0.70"), (60, "0.78, 1.00")):
            watch = self._replay(seed)
            assert _warned_at(watch) == [917_504, 2_228_224, 3_538_944, 4_849_664, 6_160_384], seed
            first = watch.progress_warnings[0]["message"]
            assert first.startswith("PostureDrift at 917,504 steps: max_foot_cop_fore_aft"), seed
            assert clause.format(fractions) in first, seed

    def test_three_in_a_row_warn_on_the_same_seeds_later(self):
        # Seed 52's first three in a row are 0.8, 0.9, 0.9 at 2,097,152 to 2,359,296 steps: eleven evaluations
        # (1,441,792 steps) after two in a row first warn.  Seed 60's are 7/9, 1.0, 1.0: one evaluation after.
        assert _warned_at(self._replay(42, consecutive=3)) == []
        seed_52 = self._replay(52, consecutive=3)
        assert _warned_at(seed_52) == [2_359_296, 3_670_016, 4_980_736]
        first = seed_52.progress_warnings[0]["message"]
        assert "on 0.80, 0.90, 0.90 of the full-horizon panel episodes at 2,097,152 to 2,359,296 steps" in first
        assert _warned_at(self._replay(60, consecutive=3)) == [1_048_576, 2_359_296, 3_670_016, 4_980_736]
        for seed, evaluations_later in ((52, 11), (60, 1)):
            two, three = (_warned_at(self._replay(seed, consecutive=n))[0] for n in (2, 3))
            assert three - two == evaluations_later * R10_EVAL, seed


class TestBuild:
    def test_built_only_where_a_fore_aft_bar_is_declared(self):
        assert build_posture_drift_watch_callback(_Plateau(), {"curriculum_kwargs": {}}) is None
        velociraptor_mongoliensis = load_stage_config(ModelId("velociraptor_mongoliensis", "anatomical"), "stance")
        assert build_posture_drift_watch_callback(_Plateau(), velociraptor_mongoliensis) is None
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
        assert (watch.over_bar_fraction, watch.consecutive_evals, watch.repeat_evals) == (0.5, 2, 10)
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
