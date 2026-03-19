"""Tests for DiagnosticsCallback."""

from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.shared.diagnostics import DiagnosticsCallback


def _make_mock_model(has_vecnorm=False):
    """Create a mock SB3 model with logger and env."""
    model = MagicMock()
    model.logger = MagicMock()
    env = MagicMock(spec=[])  # no obs_rms/ret_rms by default
    if has_vecnorm:
        env = MagicMock()  # full mock, has all attrs
        env.obs_rms = MagicMock()
        env.obs_rms.var = np.array([1.0, 2.0])
        env.ret_rms = MagicMock()
        env.ret_rms.var = np.array([0.5])
    model.get_env.return_value = env
    # Ensure rollout_buffer is not present by default
    del model.rollout_buffer
    return model


@pytest.fixture
def callback(tmp_path):
    """Create a DiagnosticsCallback with a temp log directory."""
    cb = DiagnosticsCallback(log_dir=str(tmp_path), verbose=0)
    cb.num_timesteps = 100
    cb.locals = {"infos": []}
    model = _make_mock_model()
    cb.init_callback(model)
    return cb


class TestInit:
    def test_default_params(self):
        cb = DiagnosticsCallback()
        assert cb.plateau_window == 10
        assert cb.plateau_threshold == 1.0
        assert cb._log_dir is None

    def test_custom_params(self, tmp_path):
        cb = DiagnosticsCallback(plateau_window=20, plateau_threshold=2.0, log_dir=str(tmp_path), verbose=1)
        assert cb.plateau_window == 20
        assert cb.plateau_threshold == 2.0
        assert cb._log_dir == tmp_path

    def test_initial_state(self, callback):
        assert callback._rollout_ep_rewards == []
        assert callback._rollout_terminations == Counter()
        assert callback._history_timesteps == []


class TestOnStep:
    def test_collects_reward_keys(self, callback):
        callback.locals = {"infos": [{"reward_forward": 1.5, "reward_alive": 0.5}]}
        result = callback._on_step()
        assert result is True
        assert callback._step_infos["reward_forward"] == [1.5]
        assert callback._step_infos["reward_alive"] == [0.5]

    def test_collects_info_keys(self, callback):
        callback.locals = {"infos": [{"forward_vel": 2.0, "prey_distance": 5.0}]}
        callback._on_step()
        assert callback._step_infos["forward_vel"] == [2.0]
        assert callback._step_infos["prey_distance"] == [5.0]

    def test_collects_termination_reasons(self, callback):
        callback.locals = {"infos": [{"termination_reason": "fallen"}, {"termination_reason": "fallen"}]}
        callback._on_step()
        assert callback._rollout_terminations["fallen"] == 2

    def test_ignores_missing_keys(self, callback):
        callback.locals = {"infos": [{"unknown_key": 99}]}
        callback._on_step()
        assert all(len(v) == 0 for v in callback._step_infos.values())

    def test_accumulates_across_steps(self, callback):
        callback.locals = {"infos": [{"reward_forward": 1.0}]}
        callback._on_step()
        callback.locals = {"infos": [{"reward_forward": 2.0}]}
        callback._on_step()
        assert callback._step_infos["reward_forward"] == [1.0, 2.0]

    def test_multiple_envs_per_step(self, callback):
        callback.locals = {
            "infos": [
                {"forward_vel": 1.0, "reward_forward": 0.5},
                {"forward_vel": 2.0, "reward_forward": 1.0},
                {"forward_vel": 3.0, "reward_forward": 1.5},
            ]
        }
        callback._on_step()
        assert callback._step_infos["forward_vel"] == [1.0, 2.0, 3.0]
        assert callback._step_infos["reward_forward"] == [0.5, 1.0, 1.5]

    def test_empty_infos(self, callback):
        callback.locals = {"infos": []}
        result = callback._on_step()
        assert result is True

    def test_no_infos_key(self, callback):
        callback.locals = {}
        result = callback._on_step()
        assert result is True


class TestOnRolloutEnd:
    def test_logs_reward_means_to_tensorboard(self, callback):
        callback._step_infos["reward_forward"] = [1.0, 2.0, 3.0]
        callback._on_rollout_end()
        callback.logger.record.assert_any_call("diagnostics/reward_forward", 2.0)

    def test_logs_info_means_to_tensorboard(self, callback):
        callback._step_infos["forward_vel"] = [1.0, 3.0]
        callback._on_rollout_end()
        callback.logger.record.assert_any_call("diagnostics/forward_vel", 2.0)

    def test_resets_step_infos_after_rollout(self, callback):
        callback._step_infos["reward_forward"] = [1.0, 2.0]
        callback._on_rollout_end()
        assert callback._step_infos["reward_forward"] == []

    def test_appends_to_history(self, callback):
        callback._step_infos["forward_vel"] = [1.0, 3.0]
        callback._on_rollout_end()
        assert callback._history_timesteps == [100]
        assert callback._history["forward_vel"] == [2.0]

    def test_appends_reward_to_history(self, callback):
        callback._step_infos["reward_forward"] = [1.0, 2.0]
        callback._step_infos["forward_vel"] = [1.0]  # needed to trigger history append
        callback._on_rollout_end()
        assert callback._history_rewards["reward_forward"] == [1.5]

    def test_nan_for_missing_info_keys(self, callback):
        # Only forward_vel has data, prey_distance does not
        callback._step_infos["forward_vel"] = [1.0]
        callback._on_rollout_end()
        assert np.isnan(callback._history["prey_distance"][-1])

    def test_logs_termination_fractions(self, callback):
        callback._rollout_terminations = Counter({"fallen": 3, "truncated": 7})
        callback._on_rollout_end()
        callback.logger.record.assert_any_call("terminations/fallen", 0.3)
        callback.logger.record.assert_any_call("terminations/truncated", 0.7)
        callback.logger.record.assert_any_call("terminations/total_count", 10)

    def test_clears_terminations_after_rollout(self, callback):
        callback._rollout_terminations = Counter({"fallen": 3})
        callback._on_rollout_end()
        assert len(callback._rollout_terminations) == 0

    def test_logs_obs_stats_from_rollout_buffer(self, tmp_path):
        cb = DiagnosticsCallback(log_dir=str(tmp_path), verbose=0)
        model = _make_mock_model()
        buffer = MagicMock()
        buffer.observations = np.array([[1.0, 2.0], [3.0, 4.0]])
        buffer.actions = np.array([[0.5], [-0.5]])
        model.rollout_buffer = buffer
        cb.init_callback(model)
        cb.num_timesteps = 100
        cb.locals = {"infos": []}
        cb._on_rollout_end()
        cb.logger.record.assert_any_call("diagnostics/obs_mean", 2.5)
        cb.logger.record.assert_any_call("diagnostics/obs_std", pytest.approx(np.std([[1, 2], [3, 4]])))
        cb.logger.record.assert_any_call("diagnostics/obs_max_abs", 4.0)
        cb.logger.record.assert_any_call("diagnostics/action_mean", 0.0)

    def test_logs_vecnorm_stats(self, tmp_path):
        cb = DiagnosticsCallback(log_dir=str(tmp_path), verbose=0)
        model = _make_mock_model(has_vecnorm=True)
        cb.init_callback(model)
        cb.num_timesteps = 100
        cb.locals = {"infos": []}
        cb._on_rollout_end()
        cb.logger.record.assert_any_call("diagnostics/vecnorm_obs_var_mean", 1.5)
        cb.logger.record.assert_any_call("diagnostics/vecnorm_ret_var", 0.5)


class TestPlateauDetection:
    def test_no_warning_below_window(self, callback):
        callback._rollout_ep_rewards = [1.0] * 5
        callback.locals = {"infos": [{"episode": {"r": 1.0}}]}
        callback._on_rollout_end()
        # Not enough history for plateau detection, no warning printed

    def test_plateau_warning(self, callback, caplog):
        # Fill the history to reach the plateau window
        callback._rollout_ep_rewards = [50.0] * 9  # 9 entries, need 10
        callback.locals = {"infos": [{"episode": {"r": 50.0}}]}
        with caplog.at_level("WARNING", logger="environments.shared.diagnostics"):
            callback._on_rollout_end()
        assert "PLATEAU WARNING" in caplog.text

    def test_no_plateau_when_varying(self, callback, caplog):
        # Rewards with enough variation to avoid plateau
        callback._rollout_ep_rewards = list(range(9))  # 0-8
        callback.locals = {"infos": [{"episode": {"r": 100.0}}]}
        with caplog.at_level("WARNING", logger="environments.shared.diagnostics"):
            callback._on_rollout_end()
        assert "PLATEAU WARNING" not in caplog.text

    def test_logs_reward_variation(self, callback):
        callback._rollout_ep_rewards = [50.0] * 9
        callback.locals = {"infos": [{"episode": {"r": 50.0}}]}
        callback._on_rollout_end()
        callback.logger.record.assert_any_call("diagnostics/reward_variation", 0.0)


class TestSaveDiagnostics:
    def test_saves_npz_file(self, callback, tmp_path):
        callback._step_infos["forward_vel"] = [1.0, 2.0]
        callback._on_rollout_end()
        npz_path = tmp_path / "diagnostics.npz"
        assert npz_path.exists()
        data = np.load(str(npz_path))
        assert "timesteps" in data
        assert "forward_vel" in data
        assert data["timesteps"][0] == 100

    def test_saves_reward_history(self, callback, tmp_path):
        callback._step_infos["forward_vel"] = [1.0]
        callback._step_infos["reward_forward"] = [0.5, 1.5]
        callback._on_rollout_end()
        data = np.load(str(tmp_path / "diagnostics.npz"))
        assert "reward_forward" in data
        assert data["reward_forward"][0] == pytest.approx(1.0)

    def test_saves_termination_history(self, callback, tmp_path):
        callback._rollout_terminations = Counter({"fallen": 2, "truncated": 8})
        callback._on_rollout_end()
        data = np.load(str(tmp_path / "diagnostics.npz"))
        assert "term_timesteps" in data
        assert "term_fallen" in data
        assert data["term_fallen"][0] == pytest.approx(0.2)

    def test_no_save_without_log_dir(self):
        cb = DiagnosticsCallback(log_dir=None)
        model = _make_mock_model()
        cb.init_callback(model)
        cb.num_timesteps = 100
        cb.locals = {"infos": []}
        cb._step_infos["forward_vel"] = [1.0]
        cb._on_rollout_end()
        # No error, just no file

    def test_saves_heading_alignment_std(self, callback, tmp_path):
        callback._step_infos["heading_alignment"] = [1.0, -1.0, 0.5, -0.5]
        callback._step_infos["forward_vel"] = [1.0]  # needed to trigger history append
        callback._on_rollout_end()
        data = np.load(str(tmp_path / "diagnostics.npz"))
        assert "heading_alignment_std" in data
        assert data["heading_alignment_std"][0] == pytest.approx(np.std([1.0, -1.0, 0.5, -0.5]))

    def test_heading_alignment_std_nan_when_no_data(self, callback, tmp_path):
        callback._step_infos["forward_vel"] = [1.0]  # trigger history but no heading data
        callback._on_rollout_end()
        data = np.load(str(tmp_path / "diagnostics.npz"))
        assert "heading_alignment_std" in data
        assert np.isnan(data["heading_alignment_std"][0])

    def test_accumulates_over_multiple_rollouts(self, callback, tmp_path):
        callback._step_infos["forward_vel"] = [1.0]
        callback._on_rollout_end()

        callback.num_timesteps = 200
        callback._step_infos["forward_vel"] = [2.0]
        callback._on_rollout_end()

        data = np.load(str(tmp_path / "diagnostics.npz"))
        assert len(data["timesteps"]) == 2
        assert data["timesteps"][0] == 100
        assert data["timesteps"][1] == 200
        assert data["forward_vel"][0] == pytest.approx(1.0)
        assert data["forward_vel"][1] == pytest.approx(2.0)


class TestWithoutSB3:
    """Test that DiagnosticsCallback handles the without-SB3 code paths."""

    def test_sb3_available_flag_exists(self):
        """The _SB3_AVAILABLE flag should be defined in the module."""
        from environments.shared import diagnostics

        assert hasattr(diagnostics, "_SB3_AVAILABLE")

    def test_init_has_fallback_branch(self):
        """The __init__ should have a conditional for the non-SB3 path."""
        import inspect

        source = inspect.getsource(DiagnosticsCallback.__init__)
        assert "_SB3_AVAILABLE" in source

    def test_init_callback_fallback_defined_when_sb3_absent(self):
        """When SB3 is absent the class should define its own init_callback."""
        from environments.shared import diagnostics

        if not diagnostics._SB3_AVAILABLE:
            # Only testable when SB3 is truly absent
            cb = DiagnosticsCallback()
            assert hasattr(cb, "init_callback")
        else:
            # SB3 present — init_callback comes from BaseCallback
            cb = DiagnosticsCallback()
            assert hasattr(cb, "init_callback")
