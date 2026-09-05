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
        assert cb._log_dir is None

    def test_custom_params(self, tmp_path):
        cb = DiagnosticsCallback(
            log_dir=str(tmp_path),
            verbose=1,
            action_saturation_threshold=0.95,
            save_interval_seconds=30.0,
        )
        assert cb._log_dir == tmp_path
        assert cb.action_saturation_threshold == 0.95
        assert cb.save_interval_seconds == 30.0

    def test_legacy_plateau_params_are_rejected(self):
        # The 2026-07-24 deprecation shim is retired: the kwargs are gone,
        # not silently ignored.
        with pytest.raises(TypeError):
            DiagnosticsCallback(plateau_window=20, plateau_threshold=2.0)

    def test_initial_state(self, callback):
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

    def test_derives_trex_contact_duties_and_load_share(self, callback):
        callback.locals = {
            "infos": [
                {
                    "bite_success": 0.0,
                    "r_foot_contact": 75.0,
                    "l_foot_contact": 25.0,
                }
            ]
        }
        callback._on_step()

        assert callback._step_infos["bilateral_support_duty"] == [1.0]
        assert callback._step_infos["r_foot_load_share"] == [0.75]
        assert callback._step_infos["foot_load_imbalance"] == [0.5]

    def test_environment_stance_value_wins_over_derivative(self, callback):
        callback.locals = {
            "infos": [
                {
                    "bite_success": 0.0,
                    "r_foot_contact": 75.0,
                    "l_foot_contact": 25.0,
                    "foot_load_imbalance": 0.125,
                }
            ]
        }
        callback._on_step()

        assert callback._step_infos["foot_load_imbalance"] == [0.125]

    def test_does_not_derive_biped_stance_metrics_for_quadruped_contacts(self, callback):
        callback.locals = {
            "infos": [
                {
                    "r_foot_contact": 75.0,
                    "l_foot_contact": 25.0,
                    "rr_foot_contact": 50.0,
                    "rl_foot_contact": 50.0,
                }
            ]
        }
        callback._on_step()

        assert callback._step_infos["r_foot_contact"] == [75.0]
        assert callback._step_infos["rr_foot_contact"] == [50.0]
        assert callback._step_infos["bilateral_support_duty"] == []
        assert callback._step_infos["r_foot_load_share"] == []
        assert callback._step_infos["foot_load_imbalance"] == []

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

    def test_collects_stance_reward_components(self, callback):
        callback.locals = {
            "infos": [
                {
                    "reward_bilateral_support": 0.5,
                    "reward_foot_load_balance": -0.1,
                    "reward_leg_home_pose": 0.4,
                    "reward_head_clearance": 0.3,
                    "reward_neck_posture": 0.2,
                }
            ]
        }
        callback._on_step()

        assert callback._step_infos["reward_bilateral_support"] == [0.5]
        assert callback._step_infos["reward_foot_load_balance"] == [-0.1]
        assert callback._step_infos["reward_leg_home_pose"] == [0.4]
        assert callback._step_infos["reward_head_clearance"] == [0.3]
        assert callback._step_infos["reward_neck_posture"] == [0.2]

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
        # Action stats now come from the per-step accumulator (see
        # TestActionStats), not the rollout buffer — works for SAC too.

    def test_logs_vecnorm_stats(self, tmp_path):
        cb = DiagnosticsCallback(log_dir=str(tmp_path), verbose=0)
        model = _make_mock_model(has_vecnorm=True)
        cb.init_callback(model)
        cb.num_timesteps = 100
        cb.locals = {"infos": []}
        cb._on_rollout_end()
        cb.logger.record.assert_any_call("diagnostics/vecnorm_obs_var_mean", 1.5)
        cb.logger.record.assert_any_call("diagnostics/vecnorm_ret_var", 0.5)

    def test_constant_training_episode_returns_do_not_emit_plateau_warning(self, callback, caplog):
        callback.locals = {"infos": [{"episode": {"r": 50.0}}]}
        callback._on_step()

        with caplog.at_level("WARNING", logger="environments.shared.diagnostics"):
            callback._on_rollout_end()

        assert "PLATEAU" not in caplog.text


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
        # Disable the save throttle: this test checks accumulation, and the
        # two rollout-ends happen within one save interval.
        callback.save_interval_seconds = 0.0
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


class TestActionStats:
    """Action stats + saturation come from the actions actually taken each step
    (self.locals["actions"]), so they work for PPO and SAC alike."""

    def test_logs_action_stats_and_saturation(self, callback):
        # 4 components; 2 saturated at |a| >= 0.99 (1.0 and -0.995).
        callback.locals = {"infos": [], "actions": np.array([[1.0, 0.5], [-0.995, 0.0]])}
        callback._on_step()
        callback._on_rollout_end()
        callback.logger.record.assert_any_call("diagnostics/raw_action_saturation", pytest.approx(0.5))
        callback.logger.record.assert_any_call("diagnostics/action_abs_max", pytest.approx(1.0))
        callback.logger.record.assert_any_call(
            "diagnostics/action_mean", pytest.approx(float(np.mean([1.0, 0.5, -0.995, 0.0])))
        )

    def test_accumulates_actions_across_steps_then_clears(self, callback):
        callback.locals = {"infos": [], "actions": np.array([[0.1, 0.2]])}
        callback._on_step()
        callback.locals = {"infos": [], "actions": np.array([[0.3, 0.4]])}
        callback._on_step()
        assert len(callback._step_actions) == 2
        callback._on_rollout_end()
        assert callback._step_actions == []  # flushed after rollout

    def test_no_actions_no_record(self, callback):
        callback.locals = {"infos": []}  # no "actions" key
        callback._on_step()
        callback._on_rollout_end()
        recorded = {c.args[0] for c in callback.logger.record.call_args_list}
        assert "diagnostics/raw_action_saturation" not in recorded

    def test_custom_saturation_threshold(self):
        cb = DiagnosticsCallback(action_saturation_threshold=0.5)
        assert cb.action_saturation_threshold == 0.5


class TestGradNorm:
    def test_global_grad_norm_none_without_grads(self):
        from environments.shared.diagnostics import _global_grad_norm

        class _P:
            grad = None

        assert _global_grad_norm([]) is None
        assert _global_grad_norm([_P(), _P()]) is None

    def test_global_grad_norm_value(self):
        torch = pytest.importorskip("torch")
        from environments.shared.diagnostics import _global_grad_norm

        p1 = torch.nn.Parameter(torch.zeros(2))
        p1.grad = torch.tensor([3.0, 0.0])
        p2 = torch.nn.Parameter(torch.zeros(1))
        p2.grad = torch.tensor([4.0])
        assert _global_grad_norm([p1, p2]) == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_logs_grad_norm_for_ppo(self, tmp_path):
        torch = pytest.importorskip("torch")
        cb = DiagnosticsCallback(log_dir=str(tmp_path), verbose=0)
        model = _make_mock_model()
        buffer = MagicMock()
        buffer.observations = np.array([[1.0]])
        buffer.actions = np.array([[0.5]])
        model.rollout_buffer = buffer  # PPO-like
        p1 = torch.nn.Parameter(torch.zeros(1))
        p1.grad = torch.tensor([3.0])
        p2 = torch.nn.Parameter(torch.zeros(1))
        p2.grad = torch.tensor([4.0])
        model.policy.parameters.return_value = [p1, p2]
        cb.init_callback(model)
        cb.num_timesteps = 100
        cb.locals = {"infos": []}
        cb._on_rollout_end()
        cb.logger.record.assert_any_call("diagnostics/grad_norm", pytest.approx(5.0))

    def test_no_grad_norm_for_sac(self, callback):
        # Default fixture model has no rollout_buffer (SAC-like) → grad norm skipped.
        callback.locals = {"infos": []}
        callback._on_rollout_end()
        recorded = {c.args[0] for c in callback.logger.record.call_args_list}
        assert "diagnostics/grad_norm" not in recorded


class TestAlgoMetrics:
    """SB3's algorithm-specific train/* metrics (PPO clip_fraction/
    explained_variance, SAC critic_loss/ent_coef) are captured into the
    durable diagnostics.npz, not just TensorBoard."""

    def test_collect_filters_train_keys(self, callback):
        callback.model.logger.name_to_value = {
            "train/clip_fraction": 0.1,
            "train/explained_variance": 0.8,
            "train/n_updates": 12,
            "train/some_flag": True,  # bool → skipped
            "rollout/ep_rew_mean": 5.0,  # not train/ → skipped
        }
        m = callback._collect_algo_metrics()
        assert m == {"clip_fraction": 0.1, "explained_variance": 0.8, "n_updates": 12.0}

    def test_collect_handles_non_dict_logger(self, callback):
        # The default MagicMock name_to_value is not a real dict → empty.
        assert callback._collect_algo_metrics() == {}

    def test_saves_algo_metrics_to_npz(self, tmp_path):
        cb = DiagnosticsCallback(log_dir=str(tmp_path), verbose=0)
        model = _make_mock_model()
        model.logger.name_to_value = {
            "train/clip_fraction": 0.1,
            "train/explained_variance": 0.8,
            "rollout/x": 1.0,
        }
        cb.init_callback(model)
        cb.num_timesteps = 100
        cb.locals = {"infos": []}
        cb._on_rollout_end()
        data = np.load(str(tmp_path / "diagnostics.npz"))
        assert "algo_timesteps" in data and data["algo_timesteps"][0] == 100
        assert data["algo_clip_fraction"][0] == pytest.approx(0.1)
        assert data["algo_explained_variance"][0] == pytest.approx(0.8)
        assert "algo_x" not in data  # non-train/ key excluded

    def test_algo_history_backfills_new_keys(self):
        cb = DiagnosticsCallback(log_dir=None, verbose=0)
        model = _make_mock_model()
        cb.init_callback(model)
        cb.num_timesteps = 100
        model.logger.name_to_value = {"train/clip_fraction": 0.1}
        cb._record_algo_metrics()
        cb.num_timesteps = 200
        model.logger.name_to_value = {"train/clip_fraction": 0.2, "train/approx_kl": 0.05}
        cb._record_algo_metrics()
        assert cb._history_algo["clip_fraction"] == [0.1, 0.2]
        assert np.isnan(cb._history_algo["approx_kl"][0])  # back-filled for prior rollout
        assert cb._history_algo["approx_kl"][1] == pytest.approx(0.05)
        assert all(len(v) == len(cb._history_algo_timesteps) for v in cb._history_algo.values())


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


class TestSaveThrottle:
    """diagnostics.npz rewrites are throttled; _on_training_end always flushes."""

    def test_throttled_save_then_final_flush(self, tmp_path):
        from environments.shared.diagnostics import DiagnosticsCallback

        cb = DiagnosticsCallback(log_dir=str(tmp_path), save_interval_seconds=3600.0)
        model = _make_mock_model()
        model.rollout_buffer = MagicMock(observations=None)
        cb.init_callback(model)
        cb.num_timesteps = 100
        cb.locals = {"infos": []}

        cb._step_infos["forward_vel"] = [1.0]
        cb._on_rollout_end()  # first save always happens (last_save_time=0)
        cb.num_timesteps = 200
        cb._step_infos["forward_vel"] = [2.0]
        cb._on_rollout_end()  # throttled — no rewrite

        data = np.load(str(tmp_path / "diagnostics.npz"))
        assert len(data["timesteps"]) == 1

        cb._on_training_end()  # final flush writes the full history
        data = np.load(str(tmp_path / "diagnostics.npz"))
        assert len(data["timesteps"]) == 2


class TestActionJerkIsNotDiscarded:
    """The environment emitted `action_jerk` every step and nothing kept it.

    `INFO_KEYS` carried `action_delta` but not its second difference, so the
    signal `reward_action_jerk` charges was computed and thrown away. Their
    ratio is a frequency — `jerk/delta = (2 sin(pi f dt))^2`, blind to any
    constant offset — and recovering the 22.6 Hz tremor in issue #489 had to
    invert it back out of per-episode reward totals instead of reading it.
    """

    def test_both_action_differences_are_tracked(self):
        from environments.shared.diagnostics import DiagnosticsCallback

        assert "action_delta" in DiagnosticsCallback.INFO_KEYS
        assert "action_jerk" in DiagnosticsCallback.INFO_KEYS

    def test_the_trex_env_actually_emits_it(self):
        """A tracked key the environment never sets is a silently empty column."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[3] / "environments/trex/envs/trex_env.py").read_text()
        assert 'info["action_jerk"]' in source
