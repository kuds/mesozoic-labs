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
    POSTURE_TREND_KEYS,
    PostureDriftWatchCallback,
    PostureTrendWatchCallback,
    build_posture_drift_watch_callback,
    build_posture_trend_watch_callback,
    declared_fore_aft_rails,
    posture_drift_settings,
    posture_trend_settings,
    trend_slope,
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


# ── PostureTrend: the signed CoP moving toward the edge it stands over ───────


def _means(text: str) -> list[float]:
    """Whitespace-separated signed means (half-lengths), as floats."""
    return [float(token) for token in text.split()]


class _TrendPlateau:
    """The accessors the trend watch reads: one rail's over-bar fraction and the per-foot signed CoP, fed per row."""

    def __init__(self, rail: str = "max_sole_cop_fore_aft", signed: str = "sole_cop_fore_aft", every: int = EVAL):
        self.rail, self.signed, self.every = rail, signed, every
        self.timesteps: list[int] = []
        self.fractions: list[float] = []
        self.feet: dict[str, list[float]] = {}

    def add(self, fraction: float, **feet: float) -> None:
        self.timesteps.append(self.every * (len(self.timesteps) + 1))
        self.fractions.append(fraction)
        for label, value in feet.items():
            self.feet.setdefault(label, [math.nan] * (len(self.timesteps) - 1)).append(value)

    def gate_progress_series(self, key: str) -> tuple[list[int], list[float]]:
        if key == f"{self.rail}_over_bar_fraction":
            return list(self.timesteps), list(self.fractions)
        for label, values in self.feet.items():
            if key == f"{label}_{self.signed}":
                return list(self.timesteps), list(values)
        return list(self.timesteps), [math.nan] * len(self.timesteps)

    def gate_progress_row(self) -> dict[str, float]:
        row = {f"{label}_{self.signed}": values[-1] for label, values in self.feet.items()}
        row[f"max_{self.signed}_mean"] = 0.9  # a panel key: never read as a foot
        return row


def _trend_watch(plateau, *, rail=("max_sole_cop_fore_aft", 0.8), **settings):
    # Built without __init__ so the test needs no stable-baselines3 model.
    watch = object.__new__(PostureTrendWatchCallback)
    watch.plateau_callback = plateau
    watch.rails = (rail,)
    fraction, consecutive, window, repeat = posture_trend_settings(settings)
    watch.over_bar_fraction, watch.consecutive_evals, watch.window_evals, watch.repeat_evals = (
        fraction,
        consecutive,
        window,
        repeat,
    )
    watch.drift_over_bar_fraction, watch.drift_consecutive_evals = 0.5, 2
    return watch


def _trend_replay(fractions, means, *, every=EVAL, rail=("max_sole_cop_fore_aft", 0.8), signed="sole_cop_fore_aft"):
    """Feed one row at a time, both feet at the signed mean, and step the watch after each."""
    plateau = _TrendPlateau(rail=rail[0], signed=signed, every=every)
    watch = _trend_watch(plateau, rail=rail)
    results = []
    for fraction, mean in zip(fractions, means, strict=True):
        plateau.add(fraction, r=mean, l=mean)
        results.append(watch._on_step())
    return watch, results


class TestTheTrendSettings:
    def test_the_defaults_and_the_knobs(self):
        assert posture_trend_settings({}) == (0.1, 2, 5, 10)
        block = dict(zip(POSTURE_TREND_KEYS, (0.2, 3, 4, 20)))
        assert posture_trend_settings(block) == (0.2, 3, 4, 20)

    @pytest.mark.parametrize(
        ("block", "message"),
        [
            ({"posture_trend_warn_over_bar_fraction": 0.0}, "posture_trend_warn_over_bar_fraction"),
            ({"posture_trend_warn_over_bar_fraction": 1.5}, "posture_trend_warn_over_bar_fraction"),
            ({"posture_trend_warn_over_bar_fraction": True}, "posture_trend_warn_over_bar_fraction"),
            ({"posture_trend_warn_over_bar_fraction": "0.1"}, "posture_trend_warn_over_bar_fraction"),
            ({"posture_trend_warn_consecutive_evals": 0}, "posture_trend_warn_consecutive_evals"),
            ({"posture_trend_window_evals": 2}, "posture_trend_window_evals"),  # a slope needs three points
            ({"posture_trend_window_evals": 4.0}, "posture_trend_window_evals"),
            ({"posture_trend_warn_repeat_evals": True}, "posture_trend_warn_repeat_evals"),
        ],
    )
    def test_a_malformed_knob_is_refused(self, block, message):
        with pytest.raises(ValueError, match=message):
            posture_trend_settings(block)
        with pytest.raises(ValueError, match=message):
            build_posture_trend_watch_callback(_TrendPlateau(), {"curriculum_kwargs": block})

    def test_the_knobs_configure_a_diagnostic_in_no_gate_or_hyperparameter_digest(self):
        from environments.shared.config import hyperparameters_sha256, load_all_stages
        from environments.shared.species_names import model_ids

        assert set(POSTURE_TREND_KEYS) <= _DIAGNOSTIC_KEYS
        config = load_stage_config(TYRANNOSAURUS_REX, "stance")
        knobs = dict(zip(POSTURE_TREND_KEYS, (0.2, 3, 4, 20)))
        widened = {**config, "curriculum_kwargs": {**config["curriculum_kwargs"], **knobs}}
        assert validate_gate_config("stance", widened["curriculum_kwargs"]) == "stance_quality_heading/v1"
        assert gate_config_view(widened["curriculum_kwargs"]) == gate_config_view(config["curriculum_kwargs"])
        for algorithm in ("ppo", "sac"):
            assert hyperparameters_sha256(widened, algorithm) == hyperparameters_sha256(config, algorithm)
        for model_id in model_ids():
            for stage in load_all_stages(model_id).values():
                assert not set(POSTURE_TREND_KEYS) & set(stage.get("curriculum_kwargs", {})), model_id

    def test_the_slope_is_least_squares_per_million_steps_over_the_finite_points(self):
        assert trend_slope([1e6, 2e6, 3e6], [0.1, 0.3, 0.5]) == pytest.approx(0.2)
        assert trend_slope([1e6, 2e6, 3e6, 4e6], [0.1, math.nan, 0.5, 0.7]) == pytest.approx(0.2)
        assert math.isnan(trend_slope([1e6, 2e6, 3e6], [0.1, math.nan, 0.5]))  # two points are not a trend


class TestTheTrendLine:
    RISING = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]

    def test_two_rows_over_the_bar_with_the_cop_moving_toward_the_toe_warn_once(self, caplog):
        with caplog.at_level(logging.WARNING, logger=posture_watch.__name__):
            watch, results = _trend_replay([0.0, 0.0, 0.1, 0.2, 0.0, 0.0], self.RISING)
        assert all(results)  # never stops training
        assert _warned_at(watch) == [4 * EVAL]
        (line,) = [message for message in caplog.messages if "PostureTrend" in message]
        assert line.startswith("PostureTrend at 200,000 steps: max_sole_cop_fore_aft (pad centre of pressure)")
        assert "over its 0.80 bar on 0.10, 0.20 of the full-horizon panel episodes at 150,000 to 200,000 steps" in line
        assert "mean signed CoP +0.36 half-lengths, moving toward the toe edge at +0.40 per million steps" in line
        assert "over the last 5 evaluations; per foot r +0.36 (toe side), l +0.36 (toe side)" in line
        assert "Earlier and more sensitive than PostureDrift (0.50 on 2 in a row)" in line
        assert line.endswith("Advisory only: training continues unchanged.") and "\n" not in line
        (warning,) = watch.progress_warnings
        assert warning == {
            "timesteps": 4 * EVAL,
            "source": "posture_trend",
            "rails": ["max_sole_cop_fore_aft"],
            "message": line,
        }

    def test_a_cop_moving_back_toward_the_centre_does_not_warn(self):
        falling = list(reversed(self.RISING))
        watch, _ = _trend_replay([0.0, 0.0, 0.1, 0.2, 0.3, 0.3], falling)
        assert _warned_at(watch) == []

    def test_the_heel_edge_reads_a_negative_mean_moving_more_negative(self, caplog):
        with caplog.at_level(logging.WARNING, logger=posture_watch.__name__):
            watch, _ = _trend_replay([0.0, 0.0, 0.1, 0.2], [-0.3, -0.32, -0.34, -0.36])
        assert _warned_at(watch) == [4 * EVAL]
        assert (
            "mean signed CoP -0.36 half-lengths, moving toward the heel edge at -0.40 per million steps" in caplog.text
        )
        assert "per foot r -0.36 (heel side), l -0.36 (heel side)" in caplog.text

    @pytest.mark.parametrize(
        ("fractions", "warned"),
        [
            ([0.0, 0.0, 0.1, 0.1], [4 * EVAL]),  # at the threshold counts
            ([0.0, 0.0, 0.09, 0.2], []),  # under it does not
            ([0.0, 0.0, 0.2, math.nan], []),  # NaN (no full-horizon episode) ends the streak
            ([0.2, 0.2, 0.0, 0.0], []),  # two rows over it, but only two signed means: no slope yet
        ],
    )
    def test_the_threshold_and_the_quiet_rules(self, fractions, warned):
        watch, _ = _trend_replay(fractions, self.RISING[: len(fractions)])
        assert _warned_at(watch) == warned

    def test_a_lasting_trend_repeats_at_most_once_per_ten_evaluations(self):
        rising = [0.3 + 0.01 * row for row in range(25)]
        watch, _ = _trend_replay([0.5] * 25, rising)
        assert _warned_at(watch) == [3 * EVAL, 13 * EVAL, 23 * EVAL]

    def test_each_evaluation_records_the_signed_mean_and_its_trend_to_tensorboard_only(self):
        pytest.importorskip("stable_baselines3")
        from unittest.mock import MagicMock

        plateau = _TrendPlateau()
        watch = _trend_watch(plateau)
        watch.model = MagicMock()  # SB3's callback logger is its model's
        for fraction, mean in zip([0.0, 0.0, 0.0], [0.30, 0.32, 0.34]):
            plateau.add(fraction, r=mean, l=mean + 0.02)
            watch._on_step()
        calls = [(call.args, call.kwargs) for call in watch.logger.record.call_args_list]
        means = [args[1] for args, _ in calls if args[0] == "posture/max_sole_cop_fore_aft_signed_mean"]
        assert means == pytest.approx([0.31, 0.33, 0.35])
        trends = [args[1] for args, _ in calls if args[0] == "posture/max_sole_cop_fore_aft_trend_per_mstep"]
        assert trends == pytest.approx([0.4])  # from the third row, the first with three points
        assert all(kwargs == {"exclude": ("stdout", "log")} for _, kwargs in calls)

    def test_a_resumed_watch_starts_past_the_seeded_rows_and_fits_over_them(self):
        plateau = _TrendPlateau()
        for fraction, mean in zip([0.0, 0.0, 0.2], [0.30, 0.32, 0.34]):
            plateau.add(fraction, r=mean, l=mean)
        watch = _trend_watch(plateau)
        watch._last_seen = 3  # what seed_resume_eval_state sets
        plateau.add(0.2, r=0.36, l=0.36)
        watch._on_step()
        assert _warned_at(watch) == []  # the seeded row's drift is the prior session's: one new row, not two
        plateau.add(0.2, r=0.38, l=0.38)
        watch._on_step()
        assert _warned_at(watch) == [5 * EVAL]


#: The replays at design time (2026-10-17): per series, each evaluation's over-bar fraction of the rail and the
#: mean over the feet of its signed per-foot CoP (``k/n`` is k of the n full-horizon episodes, of ten), ten
#: evaluations to a line, evaluations every 131,072 steps, read at 22:44 UTC from each run's ``gate_progress.npz``:
#: * the r9 arm L symmetry screen's seeds, pad rail (``max_sole_cop_fore_aft``, bar 0.80, 48 evaluations to
#:   6,291,456 steps, through the 6M resume);
#: * the r10 arm L screen's seeds, whole-foot rail (``max_foot_cop_fore_aft``, bar 0.80, 91 evaluations to
#:   11,927,552 steps, the report-only run to 12M).
#: Both screens are written up in ``docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`` (sections 5
#: and 6), with their evidence in ``docs/investigations/stance_2026_10/``.
#: The signed means are rounded to four decimals; the warnings are the same on the unrounded series.
TREND_R9 = {
    42: (
        _fractions(
            """
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.3
            0.3 0.1 0.1 0.1 0.0 0.0 0.1 1/9 1/9 0.2
            0.1 0.3 1/7 0.1 0.3 0.2 0.2 0.0 0.4 0.3
            0.4 0.2 0.1 0.3 0.0 0.3 0.4 0.0
            """
        ),
        _means(
            """
            +0.1097 +0.1546 +0.2113 +0.2795 +0.2059 +0.2697 +0.3216 +0.3137 +0.3347 +0.3483
            +0.3746 +0.3772 +0.3565 +0.3339 +0.3948 +0.4175 +0.4370 +0.4250 +0.4854 +0.5183
            +0.4286 +0.3778 +0.3316 +0.3744 +0.2288 +0.2682 +0.1873 +0.2758 +0.1937 +0.1865
            +0.3167 +0.3193 +0.3232 +0.3225 +0.3391 +0.3184 +0.2351 +0.2178 +0.1571 +0.1902
            +0.2332 +0.2319 +0.3265 +0.3669 +0.2999 +0.3908 +0.4514 +0.4219
            """
        ),
    ),
    52: (
        _fractions(
            """
            0.0 0.0 0.0 0.0 0.0 1/8 0.0 0.0 0.1 0.0
            0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 0.1 0.1 0.3 0.0 1/3 0.1
            0.0 0.0 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            """
        ),
        _means(
            """
            +0.2342 +0.1599 +0.2318 +0.3083 +0.3136 +0.3216 +0.3542 +0.3626 +0.3610 +0.3955
            +0.4183 +0.3006 +0.3453 +0.2768 +0.1898 +0.1018 -0.0047 -0.0327 -0.0415 -0.0895
            -0.1021 -0.1243 -0.1787 -0.1875 -0.1930 -0.2832 -0.3720 -0.1815 -0.3009 -0.2722
            -0.3440 -0.3873 -0.4183 -0.4410 -0.3834 -0.4505 -0.2701 -0.2956 -0.2857 -0.3009
            -0.3426 -0.2862 -0.2802 -0.2694 -0.2774 -0.2482 -0.3307 -0.3319
            """
        ),
    ),
    60: (
        _fractions(
            """
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.0 0.0
            0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            0.0 0.3 1/3 0.4 0.3 4/7 5/9 2/3 0.7 0.9
            0.8 1.0 1.0 0.9 0.8 0.4 0.0 0.0
            """
        ),
        _means(
            """
            +0.2227 +0.1610 +0.2830 +0.3883 +0.4436 +0.3541 +0.4312 +0.4585 +0.5000 +0.5031
            +0.4845 +0.5014 +0.5169 +0.4817 +0.5031 +0.5248 +0.5723 +0.5967 +0.5347 +0.5459
            +0.5954 +0.5861 +0.5264 +0.5264 +0.5375 +0.4775 +0.5087 +0.4820 +0.5035 +0.4155
            +0.5116 +0.5821 +0.5386 +0.5888 +0.5330 +0.7246 +0.7210 +0.7208 +0.7668 +0.8218
            +0.8073 +0.8462 +0.8569 +0.8478 +0.8040 +0.6958 +0.2756 +0.2692
            """
        ),
    ),
}
TREND_R10 = {
    42: (
        _fractions(
            """
            0.0 0.1 0.0 0.1 0.1 0.0 0.0 0.3 0.0 0.0
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 1/9 0.1 0.1 0.0 2/9 0.1
            0.1 0.0 0.0 0.0 2/9 0.1 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.1
            0.5 2/9 0.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            0.1 0.2 0.0 0.0 0.1 0.1 0.4 0.0 0.3 0.4
            0.4 0.4 0.3 0.2 0.1 0.1 0.2 0.1 0.0 0.2
            0.4 0.1 0.6 0.3 0.1 0.0 0.1 0.1 0.2 0.4
            0.2
            """
        ),
        _means(
            """
            +0.1822 +0.5158 +0.5773 +0.5620 +0.6469 +0.5913 +0.5001 +0.5131 +0.4202 +0.3582
            +0.2809 +0.1911 +0.1149 +0.0569 +0.0876 +0.0732 +0.0836 +0.1482 +0.2579 +0.2142
            +0.1499 +0.2434 +0.2299 +0.2434 +0.3354 +0.3147 +0.2724 +0.2040 +0.4090 +0.3385
            +0.2333 +0.2717 +0.1621 +0.1479 +0.3533 +0.4508 +0.3518 +0.3675 +0.3410 +0.4543
            +0.3376 +0.3942 +0.3181 +0.4158 +0.3159 +0.3246 +0.4307 +0.4816 +0.2944 +0.4610
            +0.5730 +0.5696 +0.5462 +0.1877 -0.1362 -0.0723 -0.1386 -0.1195 -0.0989 -0.1133
            +0.0099 +0.0303 -0.0272 +0.0049 +0.0076 +0.0081 +0.1740 +0.1133 +0.3143 +0.3258
            +0.2493 +0.1947 +0.0404 -0.1319 -0.3626 -0.2461 -0.1952 -0.2986 -0.2126 -0.1218
            -0.0316 -0.0307 +0.2184 +0.1300 +0.2094 +0.0877 +0.1027 +0.1226 +0.1667 +0.2282
            +0.2467
            """
        ),
    ),
    52: (
        _fractions(
            """
            0.1 0.0 0.3 0.0 0.2 0.6 0.7 0.2 0.2 0.1
            0.6 0.5 0.4 0.9 0.4 0.8 0.9 0.9 1.0 1.0
            7/8 2/3 6/7 2/3 5/8 0.7 0.9 0.6 0.7 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0
            """
        ),
        _means(
            """
            +0.3328 +0.4714 +0.5848 +0.6540 +0.6383 +0.7380 +0.7963 +0.6840 +0.6682 +0.6872
            +0.7500 +0.7775 +0.7701 +0.9051 +0.8172 +0.8876 +1.0638 +0.8972 +0.9691 +0.9424
            +0.8441 +0.8199 +0.8626 +0.8404 +0.8137 +0.8209 +0.8247 +0.8185 +0.8307 +0.9527
            +0.9284 +0.8650 +0.8576 +0.8779 +0.8953 +0.9279 +0.9196 +0.9553 +0.9160 +0.9567
            +0.9529 +0.9273 +0.9714 +0.9751 +0.9710 +0.9445 +0.9648 +0.9667 +1.0028 +1.0407
            +1.0072 +1.0012 +1.0015 +1.0171 +1.0101 +0.9770 +0.9593 +0.9928 +0.9593 +0.9511
            +0.9741 +0.9580 +0.9767 +0.9977 +0.9892 +1.0618 +1.0876 +1.0369 +1.0116 +1.0292
            +1.0336 +0.9994 +1.0136 +0.9329 +0.9592 +0.9764 +0.9399 +0.9582 +0.9949 +0.9806
            +0.9860 +0.9732 +0.9302 +0.9772 +1.0523 +1.0345 +1.0557 +1.0524 +1.0119 +1.0503
            +1.0266
            """
        ),
    ),
    60: (
        _fractions(
            """
            0.0 0.1 0.3 0.1 4/9 7/9 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0
            """
        ),
        _means(
            """
            +0.3194 +0.4909 +0.5774 +0.5656 +0.7110 +0.7804 +0.9692 +0.8423 +0.9719 +1.0199
            +1.1365 +1.1923 +1.1809 +1.1733 +1.1583 +1.2362 +1.1655 +1.2351 +1.1428 +1.1909
            +1.2064 +1.2346 +1.1746 +1.1444 +1.0964 +1.1470 +1.1549 +1.1911 +1.1604 +1.1360
            +1.1499 +1.0413 +1.0326 +1.0454 +0.9966 +0.9745 +0.9565 +1.0368 +1.0701 +1.0706
            +1.0462 +1.0074 +1.0747 +1.1136 +1.0808 +1.0874 +1.0447 +0.9851 +0.9980 +1.0349
            +1.0393 +1.0399 +1.0154 +1.0042 +1.0244 +1.0490 +1.0483 +1.0371 +0.9917 +1.0097
            +1.0258 +1.0413 +1.0377 +1.0418 +1.0079 +1.0879 +1.0627 +0.9934 +1.0182 +0.9952
            +0.9486 +0.9185 +0.9524 +0.9552 +0.9920 +0.9956 +0.9992 +0.9511 +0.9570 +0.9710
            +0.9141 +0.9538 +0.9934 +0.9292 +0.9257 +0.9193 +0.9179 +0.8222 +0.9577 +0.9332
            +0.9549
            """
        ),
    ),
}


class TestTheReplays:
    """What the trend line would have printed on the r9 and r10 arm L screens, against PostureDrift's 0.5 x 2."""

    @staticmethod
    def _warned(series, seed, rail, signed):
        fractions, means = series[seed]
        watch, results = _trend_replay(fractions, means, every=131_072, rail=(rail, 0.8), signed=signed)
        assert all(results)
        return _warned_at(watch)

    @staticmethod
    def _drift(series, seed):
        watch, _ = _replay(series[seed][0], every=131_072)
        return _warned_at(watch)

    def test_the_series(self):
        assert {seed: (len(f), len(m)) for seed, (f, m) in TREND_R9.items()} == dict.fromkeys((42, 52, 60), (48, 48))
        assert {seed: (len(f), len(m)) for seed, (f, m) in TREND_R10.items()} == dict.fromkeys((42, 52, 60), (91, 91))

    @pytest.mark.parametrize(
        ("seed", "trend", "drift"),
        [
            (60, [4_325_376, 5_636_096], [4_849_664]),
            (52, [3_407_872], []),
            (42, [2_621_440, 4_063_232, 5_505_024], []),
        ],
    )
    def test_the_r9_pad_rail(self, seed, trend, drift):
        assert self._warned(TREND_R9, seed, "max_sole_cop_fore_aft", "sole_cop_fore_aft") == trend
        assert self._drift(TREND_R9, seed) == drift

    @pytest.mark.parametrize(
        ("seed", "trend", "drift"),
        [
            (60, [393_216, 1_703_936, 3_014_656, 5_111_808], [917_504, 2_228_224, 3_538_944, 4_849_664]),
            (52, [786_432, 2_097_152, 3_801_088, 5_111_808], [917_504, 2_228_224, 3_538_944, 4_849_664]),
            (42, [655_360, 3_407_872, 4_718_592, 6_553_600], []),
        ],
    )
    def test_the_r10_whole_foot_rail(self, seed, trend, drift):
        assert self._warned(TREND_R10, seed, "max_foot_cop_fore_aft", "foot_cop_fore_aft")[:4] == trend
        assert self._drift(TREND_R10, seed)[:4] == drift

    def test_on_r9_seed_60_the_line_comes_one_evaluation_into_the_lasting_drift_and_four_before_posture_drift(self):
        fractions, _means_ = TREND_R9[60]
        # The drift that lasted starts at 4,194,304 steps (0.3 of the panel over the bar; the two single episodes
        # over it at 1.05M and 1.44M steps did not repeat), the evaluation before the line.
        assert fractions[31:34] == pytest.approx([0.3, 1 / 3, 0.4]) and max(fractions[11:31]) == 0.0
        warned = self._warned(TREND_R9, 60, "max_sole_cop_fore_aft", "sole_cop_fore_aft")[0]
        assert warned == 4_194_304 + 131_072
        assert self._drift(TREND_R9, 60)[0] == warned + 4 * 131_072


class TestTheTrendWatchIsBuiltAndWired:
    def test_built_only_where_a_fore_aft_bar_is_declared(self):
        assert build_posture_trend_watch_callback(_TrendPlateau(), {"curriculum_kwargs": {}}) is None
        velociraptor_mongoliensis = load_stage_config(ModelId("velociraptor_mongoliensis", "anatomical"), "stance")
        assert build_posture_trend_watch_callback(_TrendPlateau(), velociraptor_mongoliensis) is None
        assert build_posture_trend_watch_callback(None, load_stage_config(TYRANNOSAURUS_REX, "stance")) is None

    def test_the_tyrannosaurus_rex_stance_trains_with_it_after_the_drift_watch_and_reports_it(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("stable_baselines3")
        from environments.shared.curriculum.progress_report import TrainingProgressCallback
        from environments.shared.eval_diagnostics import StageGatePlateauCallback
        from environments.shared.tests.test_curriculum_hop_watch import TestTrainerWiring

        watch = build_posture_trend_watch_callback(_TrendPlateau(), load_stage_config(TYRANNOSAURUS_REX, "stance"))
        assert isinstance(watch, PostureTrendWatchCallback)
        assert watch.rails == (("max_sole_cop_fore_aft", 0.8),)
        assert (watch.over_bar_fraction, watch.consecutive_evals, watch.window_evals, watch.repeat_evals) == (
            0.1,
            2,
            5,
            10,
        )
        callbacks = TestTrainerWiring._callbacks(tmp_path, monkeypatch, load_stage_config(TYRANNOSAURUS_REX, "stance"))
        kinds = [type(callback) for callback in callbacks]
        assert kinds.count(PostureTrendWatchCallback) == 1
        trend = callbacks[kinds.index(PostureTrendWatchCallback)]
        assert kinds.index(PostureDriftWatchCallback) < kinds.index(PostureTrendWatchCallback)
        assert kinds.index(PostureTrendWatchCallback) < kinds.index(TrainingProgressCallback)
        assert trend.plateau_callback is callbacks[kinds.index(StageGatePlateauCallback)]
        assert trend in callbacks[kinds.index(TrainingProgressCallback)].warning_sources

    def test_a_resume_starts_the_trend_watch_past_the_seeded_gate_progress_rows(self, tmp_path):
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
        watch = _trend_watch(plateau)
        eval_callback = type("Eval", (), {"n_eval_episodes": 1})()
        seed_resume_eval_state(eval_callback, [plateau, watch], tmp_path / "evaluations.npz")
        assert watch._last_seen == 2
