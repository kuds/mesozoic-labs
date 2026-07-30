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
        cb.model = MagicMock()
        cb.model.ent_coef = initial
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
