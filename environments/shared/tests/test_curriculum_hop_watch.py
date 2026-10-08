"""Tests for environments.shared.curriculum.hop_watch (decision D-D27).

The two T. rex physics-r8 stance runs trained one config and split: seed 44's
evaluation unsupported duty fell under the 0.05 bar from 5.35M (0.0001 at 6M,
0.013 at 7M and at most 0.018 from 7M on), while seed 42 locked into a 12.5 Hz
two-foot hop at exactly 0.375 from 6.5M to 11M.  The series below are those
runs' ``gate_progress.npz`` values at the evaluations named.
"""

from __future__ import annotations

import json
import logging
import math

import pytest

from environments.shared.config import load_stage_config
from environments.shared.curriculum import hop_watch
from environments.shared.curriculum.gate_schema import _COLLAPSE_KEYS, validate_gate_config
from environments.shared.curriculum.hop_watch import (
    HOP_WATCH_FILENAME,
    HOP_WATCH_KEYS,
    HopRegimeWatchCallback,
    build_hop_watch_callback,
    hop_watch_settings,
)

#: Evaluation unsupported duty of the two r8 runs (gate_progress.npz).
SEED_42 = [(4_000_000, 0.418), (5_000_000, 0.365), (6_000_000, 0.372), (6_500_000, 0.375), (7_000_000, 0.375)]
SEED_44 = [
    (4_000_000, 0.321),
    (5_000_000, 0.139),
    (6_000_000, 0.0001),
    (6_500_000, 0.0),
    (7_000_000, 0.013),
    (7_250_000, 0.018),
]


class _Plateau:
    """The one accessor the watch reads, fed one evaluation at a time."""

    def __init__(self) -> None:
        self.timesteps: list[int] = []
        self.duties: list[float] = []

    def add(self, timestep: int, duty: float) -> None:
        self.timesteps.append(timestep)
        self.duties.append(duty)

    def gate_progress_series(self, key: str) -> tuple[list[int], list[float]]:
        assert key == "unsupported_duty"
        return list(self.timesteps), list(self.duties)


def _watch(plateau: _Plateau, *, stop: bool, stage_dir=None, after: int = 7_000_000) -> HopRegimeWatchCallback:
    # Built without __init__ so the test needs no stable-baselines3 model.
    watch = object.__new__(HopRegimeWatchCallback)
    watch.plateau_callback = plateau
    watch.max_unsupported_duty = 0.05
    watch.after_timesteps = after
    watch.stop = stop
    watch.stage_dir = stage_dir
    return watch


def _replay(series, *, stop: bool, stage_dir=None) -> list[bool]:
    plateau = _Plateau()
    watch = _watch(plateau, stop=stop, stage_dir=stage_dir)
    results = []
    for timestep, duty in series:
        plateau.add(timestep, duty)
        results.append(watch._on_step())
    return results


class TestSettings:
    def test_undeclared_is_none(self):
        assert hop_watch_settings({}) is None

    def test_the_trex_stance_block(self):
        curriculum = load_stage_config("trex", "stance")["curriculum_kwargs"]
        assert hop_watch_settings(curriculum) == (0.05, 7_000_000, True)
        # Anchored to the entropy anneal's end: move the two together.
        assert (
            curriculum["hop_watch_after_timesteps"]
            == load_stage_config("trex", "stance")["ppo_kwargs"]["ent_coef_decay_timesteps"]
        )

    def test_defaults_warn_from_the_start(self):
        assert hop_watch_settings({"hop_watch_max_unsupported_duty": 0.1}) == (0.1, 0, False)

    @pytest.mark.parametrize(
        ("block", "message"),
        [
            ({"hop_watch_max_unsupported_duty": 1.5}, "hop_watch_max_unsupported_duty"),
            ({"hop_watch_max_unsupported_duty": True}, "hop_watch_max_unsupported_duty"),
            ({"hop_watch_max_unsupported_duty": "0.05"}, "hop_watch_max_unsupported_duty"),
            ({"hop_watch_max_unsupported_duty": 0.05, "hop_watch_after_timesteps": -1}, "hop_watch_after_timesteps"),
            ({"hop_watch_max_unsupported_duty": 0.05, "hop_watch_after_timesteps": 7e6}, "hop_watch_after_timesteps"),
            ({"hop_watch_max_unsupported_duty": 0.05, "hop_watch_stop": 1}, "hop_watch_stop"),
            ({"hop_watch_stop": True}, "hop_watch_max_unsupported_duty is not declared"),
        ],
    )
    def test_a_malformed_block_is_refused(self, block, message):
        with pytest.raises(ValueError, match=message):
            hop_watch_settings(block)

    def test_the_keys_configure_early_stopping_not_the_gate(self):
        assert set(HOP_WATCH_KEYS) <= _COLLAPSE_KEYS
        curriculum = load_stage_config("trex", "stance")["curriculum_kwargs"]
        assert validate_gate_config("stance", curriculum) == "stance_quality/v2"


class TestReplay:
    def test_seed_42_stops_at_seven_million(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=hop_watch.__name__):
            results = _replay(SEED_42, stop=True, stage_dir=tmp_path)
        assert results == [True, True, True, True, False]
        record = json.loads((tmp_path / HOP_WATCH_FILENAME).read_text(encoding="utf-8"))
        assert record["timesteps"] == 7_000_000 and record["unsupported_duty"] == 0.375
        assert record["action"] == "stopped"
        assert sum("HopWatch" in message for message in caplog.messages) == 1

    def test_seed_44_is_never_touched(self, tmp_path):
        assert _replay(SEED_44, stop=True, stage_dir=tmp_path) == [True] * len(SEED_44)
        assert not (tmp_path / HOP_WATCH_FILENAME).exists()

    def test_warn_only_records_and_keeps_training(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=hop_watch.__name__):
            results = _replay(SEED_42 + [(7_500_000, 0.375)], stop=False, stage_dir=tmp_path)
        assert all(results)
        assert json.loads((tmp_path / HOP_WATCH_FILENAME).read_text(encoding="utf-8"))["action"] == "warned"
        assert sum("HopWatch" in message for message in caplog.messages) == 1

    def test_an_unmeasured_duty_never_trips(self):
        assert _replay([(7_000_000, math.nan), (7_500_000, math.nan)], stop=True) == [True, True]

    def test_a_late_evaluation_after_a_clean_one_still_trips(self):
        assert _replay([(7_000_000, 0.0), (7_500_000, 0.2)], stop=True) == [True, False]

    def test_a_duty_at_the_bar_does_not_trip(self):
        # Strictly above: 0.05 itself is under the watch, as at the gate's own duty bars.
        assert _replay([(7_000_000, 0.05), (7_500_000, 0.0500001)], stop=True) == [True, False]


class TestBuild:
    def test_built_only_where_declared(self):
        plateau = _Plateau()
        assert build_hop_watch_callback(plateau, {"curriculum_kwargs": {}}, stage_dir=None) is None
        raptor = load_stage_config("velociraptor", 1)
        assert build_hop_watch_callback(plateau, raptor, stage_dir=None) is None
        assert build_hop_watch_callback(None, load_stage_config("trex", "stance"), stage_dir=None) is None

    def test_trex_stance_builds_a_stopping_watch(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        watch = build_hop_watch_callback(_Plateau(), load_stage_config("trex", "stance"), stage_dir=tmp_path)
        assert isinstance(watch, HopRegimeWatchCallback)
        assert (watch.max_unsupported_duty, watch.after_timesteps, watch.stop) == (0.05, 7_000_000, True)

    def test_recovery_inherits_no_watch(self):
        # [curriculum] is never inherited (recovery.toml's extends), so the warm-started stage has none.
        assert build_hop_watch_callback(_Plateau(), load_stage_config("trex", "recovery"), stage_dir=None) is None


class TestTrainerWiring:
    """``train_base._build_core_callbacks`` is what makes the watch run during training.

    The tests above call the builder and ``_on_step`` directly, so they would
    all still pass with the watch dropped from the trainer's callback list.
    """

    @staticmethod
    def _callbacks(tmp_path, monkeypatch, stage_config) -> list:
        pytest.importorskip("stable_baselines3")
        import tempfile

        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv

        from environments.shared.train_base import _build_core_callbacks

        from .tiny_env_helpers import tiny_env_class

        # The staging and evaluation dirs ``_build_core_callbacks`` makes with mkdtemp stay in tmp_path.
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))
        eval_env = DummyVecEnv([tiny_env_class()])
        try:
            callbacks, _, _ = _build_core_callbacks(
                {"CheckpointCallback": CheckpointCallback},
                eval_env,
                tmp_path / "models",
                tmp_path / "logs",
                1,
                1,
                100,
                100,
                0,
                stage_config,
            )
        finally:
            eval_env.close()
        return callbacks

    def test_the_trex_stance_trains_with_the_watch_after_the_plateau_callback(self, tmp_path, monkeypatch):
        from environments.shared.eval_diagnostics import StageGatePlateauCallback

        callbacks = self._callbacks(tmp_path, monkeypatch, load_stage_config("trex", "stance"))
        kinds = [type(callback) for callback in callbacks]
        assert kinds.count(HopRegimeWatchCallback) == 1
        # After the plateau callback, so each evaluation is recorded before the watch reads the series.
        assert kinds.index(StageGatePlateauCallback) < kinds.index(HopRegimeWatchCallback)
        watch = callbacks[kinds.index(HopRegimeWatchCallback)]
        assert watch.plateau_callback is callbacks[kinds.index(StageGatePlateauCallback)]
        assert (watch.max_unsupported_duty, watch.after_timesteps, watch.stop) == (0.05, 7_000_000, True)
        assert watch.stage_dir == tmp_path / "logs"

    def test_a_stage_without_the_keys_trains_without_one(self, tmp_path, monkeypatch):
        callbacks = self._callbacks(tmp_path, monkeypatch, load_stage_config("velociraptor", 1))
        assert not any(isinstance(callback, HopRegimeWatchCallback) for callback in callbacks)


class TestPlateauAccessor:
    def test_the_series_is_aligned_by_evaluation(self):
        from environments.shared.eval_diagnostics import StageGatePlateauCallback

        callback = object.__new__(StageGatePlateauCallback)
        callback._gate_progress_timesteps = [100, 200]
        callback._gate_progress = {"unsupported_duty": [0.1, 0.0], "short": [1.0]}
        assert callback.gate_progress_series("unsupported_duty") == ([100, 200], [0.1, 0.0])
        timesteps, values = callback.gate_progress_series("short")
        assert timesteps == [100, 200] and all(math.isnan(value) for value in values)
        fresh = object.__new__(StageGatePlateauCallback)
        assert fresh.gate_progress_series("unsupported_duty") == ([], [])
