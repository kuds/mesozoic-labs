"""Tests for environments.shared.curriculum.schedules.

The pickle cases matter because SB3 accepts callables for learning rate and
clip range, and cloudpickle drags a notebook lambda's __globals__ along with
it; ``_ConstantSchedule`` exists to be picklable where a lambda is not."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.curriculum import (
    StageWarmupCallback,
    _ConstantSchedule,
)


class TestPickleSafety:
    """Ensure objects assigned to model attributes survive cloudpickle round-trips.

    In Colab/Jupyter, cloudpickle serialises the __globals__ of lambdas
    defined in notebook cells, which pulls in zmq.Context and fails.
    These tests verify that our replacements are safely picklable.
    """

    def test_constant_schedule_roundtrips(self):
        """_ConstantSchedule survives pickle and returns the same value."""
        import pickle

        import cloudpickle

        sched = _ConstantSchedule(0.02)
        restored = pickle.loads(cloudpickle.dumps(sched))
        assert restored(0.5) == pytest.approx(0.02)
        assert restored(1.0) == pytest.approx(0.02)

    def test_warmup_clip_range_is_picklable(self):
        """After StageWarmupCallback sets clip_range, the model can be pickled."""
        import pickle

        import cloudpickle

        cb = object.__new__(StageWarmupCallback)
        cb.warmup_clip_range = 0.02
        cb.warmup_ent_coef = 0.02
        cb.warmup_lr_scale = 0.1
        cb.warmup_timesteps = 100_000
        cb._warmup_done = False
        cb._original_clip_range = None
        cb._original_ent_coef = None
        cb._original_lr_schedule = None
        cb._original_log_ent_coef = None
        cb._is_sac = False

        mock_model = MagicMock()
        mock_model.clip_range = lambda _: 0.2  # original PPO schedule
        mock_model.ent_coef = 0.01
        # Ensure it's detected as PPO (no log_ent_coef)
        del mock_model.log_ent_coef
        cb.model = mock_model

        # Simulate _on_training_start
        cb._on_training_start()

        # The clip_range assigned by the callback must be picklable
        restored = pickle.loads(cloudpickle.dumps(mock_model.clip_range))
        assert restored(0.5) == pytest.approx(0.02)


class TestEntCoefDecayCallback:
    """EntCoefDecayCallback linearly decays model.ent_coef."""

    @staticmethod
    def _make_cb(end_value=0.0005, decay_timesteps=1000, initial=0.005):
        from environments.shared.curriculum import EntCoefDecayCallback

        cb = object.__new__(EntCoefDecayCallback)
        cb.end_value = end_value
        cb.decay_timesteps = decay_timesteps
        cb._initial = None
        cb._started = False
        cb.model = MagicMock()
        cb.model.ent_coef = initial
        # A fresh model has no warm-up marker; a MagicMock auto-creates a
        # truthy attribute for it, which would read as "warm-up active".
        cb.model._ent_coef_warmup_active = False
        cb.num_timesteps = 0
        return cb

    def test_decays_linearly(self):
        cb = self._make_cb()
        cb._on_training_start()

        cb.num_timesteps = 500
        cb._on_step()
        assert cb.model.ent_coef == pytest.approx(0.005 + 0.5 * (0.0005 - 0.005))

    def test_holds_at_end_value(self):
        cb = self._make_cb()
        cb._on_training_start()

        cb.num_timesteps = 5000  # past decay_timesteps
        cb._on_step()
        assert cb.model.ent_coef == pytest.approx(0.0005)

    def test_noop_before_training_start(self):
        cb = self._make_cb()
        cb.num_timesteps = 500
        assert cb._on_step() is True
        assert cb.model.ent_coef == 0.005  # untouched

    def test_raises_without_sb3(self):
        from environments.shared.curriculum import EntCoefDecayCallback

        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                EntCoefDecayCallback()


class _FakePPOModel:
    """A plain object, because the warm-up detects PPO via hasattr checks.

    No ``log_ent_coef`` (so it reads as PPO, not SAC) and no auto-created
    attributes (a MagicMock would answer the warm-up marker probe with a
    truthy child mock)."""

    def __init__(self, ent_coef=0.005):
        self.ent_coef = ent_coef
        self.clip_range = _ConstantSchedule(0.2)


class TestEntCoefDecayDefersToStageWarmup:
    """The decay must not clobber StageWarmupCallback's stage-entry boost.

    Every PPO stage-2/3 config sets both ``ent_coef_end`` and
    ``warmup_ent_coef``, and before the marker existed the decay's per-step
    assignment overwrote the boost on the first step after the warm-up set
    it: the log claimed "warm-up active … ent_coef=0.020" while training
    ran at the decaying base value. These tests drive the two REAL callbacks
    together, in both construction orders."""

    @staticmethod
    def _callbacks(warmup_timesteps=100):
        pytest.importorskip("stable_baselines3")
        from environments.shared.curriculum import EntCoefDecayCallback, StageWarmupCallback

        decay = EntCoefDecayCallback(end_value=0.0, decay_timesteps=1000)
        warmup = StageWarmupCallback(warmup_timesteps=warmup_timesteps, warmup_clip_range=0.02, warmup_ent_coef=0.02)
        model = _FakePPOModel(ent_coef=0.005)
        decay.model = model
        warmup.model = model
        return decay, warmup, model

    @staticmethod
    def _tick(model, *callbacks, step):
        """One CallbackList tick: every callback sees the same num_timesteps."""
        for cb in callbacks:
            cb.num_timesteps = step
            cb._on_step()
        return model.ent_coef

    def test_the_boost_holds_for_the_whole_warmup_window(self):
        decay, warmup, model = self._callbacks(warmup_timesteps=100)
        # CallbackList fires _on_training_start in list order; the launch
        # paths append the decay callback first.
        decay._on_training_start()
        warmup._on_training_start()
        assert model.ent_coef == pytest.approx(0.02)

        # Before the marker, the very first _on_step overwrote the boost.
        for step in (1, 50, 99):
            assert self._tick(model, decay, warmup, step=step) == pytest.approx(0.02)

    def test_the_decay_resumes_from_the_base_after_the_warmup_ends(self):
        decay, warmup, model = self._callbacks(warmup_timesteps=100)
        decay._on_training_start()
        warmup._on_training_start()

        self._tick(model, decay, warmup, step=100)  # warm-up restores here
        assert warmup._warmup_done is True
        # Next tick: the decay owns ent_coef again and continues the
        # configured schedule from the BASE value, not the boost.
        value = self._tick(model, decay, warmup, step=101)
        assert value == pytest.approx(0.005 + (101 / 1000) * (0.0 - 0.005))

    def test_the_base_is_captured_after_a_warmup_that_started_first(self):
        """Robust to the reverse order: the warm-up boosts ent_coef before the
        decay ever sees it, so the decay must defer its base capture."""
        decay, warmup, model = self._callbacks(warmup_timesteps=100)
        warmup._on_training_start()
        decay._on_training_start()
        assert decay._initial is None  # 0.02 must NOT be captured as the base

        self._tick(model, warmup, decay, step=100)  # restore, then decay's step
        assert model.ent_coef == pytest.approx(0.005 + (100 / 1000) * (0.0 - 0.005))
        assert decay._initial == pytest.approx(0.005)
