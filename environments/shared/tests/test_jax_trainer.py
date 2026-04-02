"""Tests for jax_trainer and jax_hooks modules."""

from __future__ import annotations

import pytest

from environments.shared.jax_hooks import LoggingHook, StabilityHook
from environments.shared.jax_trainer import JaxTrainer, StopTraining, TrainerState


class TestTrainerState:
    def test_defaults(self):
        state = TrainerState(
            params=None, opt_state=None, obs_stats=None,
            env_states=None, rng=None,
        )
        assert state.update == 0
        assert state.total_steps == 0
        assert state.history == []

    def test_history_is_independent(self):
        s1 = TrainerState(
            params=None, opt_state=None, obs_stats=None,
            env_states=None, rng=None,
        )
        s2 = TrainerState(
            params=None, opt_state=None, obs_stats=None,
            env_states=None, rng=None,
        )
        s1.history.append({"x": 1})
        assert len(s2.history) == 0


class TestStopTraining:
    def test_with_reason(self):
        exc = StopTraining("KL exploded")
        assert exc.reason == "KL exploded"
        assert str(exc) == "KL exploded"

    def test_empty_reason(self):
        exc = StopTraining()
        assert exc.reason == ""


class TestLoggingHook:
    def test_interval(self):
        hook = LoggingHook(interval=5, num_updates=100)
        assert hook.interval == 5
        assert hook.num_updates == 100

    def test_on_update_end_no_crash(self):
        hook = LoggingHook(interval=1, num_updates=10)
        state = TrainerState(
            params=None, opt_state=None, obs_stats=None,
            env_states=None, rng=None, update=0, total_steps=100,
        )
        # Should not raise
        hook.on_update_end(state, {"update": 0, "mean_reward": 1.5, "fps": 1000.0})

    def test_skips_non_interval(self, caplog):
        hook = LoggingHook(interval=10, num_updates=100)
        state = TrainerState(
            params=None, opt_state=None, obs_stats=None,
            env_states=None, rng=None, update=3, total_steps=100,
        )
        import logging
        with caplog.at_level(logging.INFO):
            hook.on_update_end(state, {"update": 3, "mean_reward": 1.0, "fps": 500.0})
        # Should NOT log for update 3 with interval 10
        assert "Update" not in caplog.text


class TestStabilityHook:
    def test_stable_no_halt(self):
        hook = StabilityHook()
        state = TrainerState(
            params=None, opt_state=None, obs_stats=None,
            env_states=None, rng=None, update=0,
        )
        # Normal metrics — should not raise
        hook.on_update_end(state, {
            "approx_kl": 0.01,
            "grad_norm": 5.0,
            "total_loss": 1.0,
        })

    def test_kl_halt_raises(self):
        hook = StabilityHook(kl_halt=1e6)
        state = TrainerState(
            params=None, opt_state=None, obs_stats=None,
            env_states=None, rng=None, update=5,
        )
        with pytest.raises(StopTraining, match="HALTING"):
            hook.on_update_end(state, {
                "approx_kl": 1e7,
                "grad_norm": 5.0,
                "total_loss": 1.0,
            })

    def test_consecutive_warnings_halt(self):
        hook = StabilityHook(kl_warn=0.5, max_warnings=3)
        state = TrainerState(
            params=None, opt_state=None, obs_stats=None,
            env_states=None, rng=None, update=0,
        )
        metrics = {"approx_kl": 1.0, "grad_norm": 0.0, "total_loss": 0.0}
        # First two warnings — no halt
        hook.on_update_end(state, metrics)
        hook.on_update_end(state, metrics)
        # Third consecutive warning — should halt
        with pytest.raises(StopTraining, match="HALTING"):
            hook.on_update_end(state, metrics)

    def test_monitor_accessible(self):
        hook = StabilityHook(kl_warn=10.0)
        assert hook.monitor.kl_warn == 10.0


class TestTrainingHookProtocol:
    """Verify that arbitrary objects implementing hook methods are accepted."""

    def test_custom_hook_class(self):
        class MyHook:
            def __init__(self):
                self.updates = []

            def on_update_end(self, state, metrics):
                self.updates.append(metrics)

        hook = MyHook()
        # Ensure JaxTrainer accepts it (structural typing check)
        trainer = JaxTrainer.__new__(JaxTrainer)
        trainer.hooks = [hook]
        trainer._dispatch("on_update_end", None, {"reward": 1.0})
        assert len(hook.updates) == 1
        assert hook.updates[0] == {"reward": 1.0}

    def test_partial_hook(self):
        """Hook that only implements on_train_end still works."""
        class EndOnlyHook:
            def __init__(self):
                self.called = False

            def on_train_end(self, state):
                self.called = True

        hook = EndOnlyHook()
        trainer = JaxTrainer.__new__(JaxTrainer)
        trainer.hooks = [hook]
        # Dispatching a method the hook doesn't have should not crash
        trainer._dispatch("on_update_end", None, {})
        assert not hook.called
        trainer._dispatch("on_train_end", None)
        assert hook.called
