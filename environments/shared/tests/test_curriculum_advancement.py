"""Tests for environments.shared.curriculum.advancement."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.curriculum import (
    CurriculumCallback,
    RewardRampCallback,
    StageThreshold,
    StageWarmupCallback,
)


class TestCallbackMethodsMocked:
    """Test callback methods by constructing instances via __new__ and mocking."""

    def test_on_step_returns_true_before_eval_freq(self):
        """_on_step returns True when eval_freq hasn't been reached."""
        cb = object.__new__(CurriculumCallback)
        cb.eval_freq = 10000
        cb._last_eval_step = 0
        cb.num_timesteps = 5000
        assert cb._on_step() is True

    def test_on_step_delegates_to_standalone(self):
        """_on_step calls _on_step_standalone when eval_callback is None."""
        cb = object.__new__(CurriculumCallback)
        cb.eval_freq = 10000
        cb._last_eval_step = 0
        cb.num_timesteps = 15000
        cb.eval_callback = None
        with patch.object(CurriculumCallback, "_on_step_standalone", return_value=True) as mock_standalone:
            result = cb._on_step()
        assert cb._last_eval_step == 15000
        mock_standalone.assert_called_once()
        assert result is True

    def test_on_step_delegates_to_eval_callback_path(self):
        """_on_step calls _on_step_with_eval_callback when eval_callback is set."""
        cb = object.__new__(CurriculumCallback)
        cb.eval_freq = 10000
        cb._last_eval_step = 0
        cb.num_timesteps = 15000
        cb.eval_callback = MagicMock()
        with patch.object(CurriculumCallback, "_on_step_with_eval_callback", return_value=True) as mock_ecb:
            result = cb._on_step()
        mock_ecb.assert_called_once()
        assert result is True

    def test_log_locomotion_metrics_empty(self):
        """_log_locomotion_metrics returns early on empty list."""
        cb = object.__new__(CurriculumCallback)
        cb._log_locomotion_metrics([])  # should not raise

    def test_log_locomotion_metrics_with_data(self):
        """_log_locomotion_metrics aggregates and logs metrics."""
        cb = object.__new__(CurriculumCallback)
        cb.curriculum_manager = MagicMock()
        cb.curriculum_manager.current_stage = 1
        cb.num_timesteps = 1000

        mock_agg = {
            "mean_forward_velocity": 1.5,
            "mean_total_distance": 10.0,
            "mean_cost_of_transport": 0.3,
            "termination_counts": {"timeout": 3, "fall": 1},
        }

        with (
            patch("environments.shared.curriculum.advancement.LocomotionMetrics") as MockMetrics,
            patch("environments.shared.curriculum.advancement.log_eval_metrics") as mock_log,
        ):
            MockMetrics.aggregate_episodes.return_value = mock_agg
            cb._log_locomotion_metrics([{"some": "report"}])

        mock_log.assert_called_once_with(mock_agg, 1, step=1000)

    def test_success_samples_are_na_when_stage_has_no_success_gate(self):
        cb = object.__new__(CurriculumCallback)
        cb.curriculum_manager = MagicMock()
        cb.curriculum_manager.current_threshold = StageThreshold(min_success_rate=0.0)

        result = cb._success_rates_for_stage([1.0, 0.0], [1.0])

        assert result is None

    def test_success_gate_prefers_full_eval_samples(self):
        cb = object.__new__(CurriculumCallback)
        cb.curriculum_manager = MagicMock()
        cb.curriculum_manager.current_threshold = StageThreshold(min_success_rate=0.5)

        result = cb._success_rates_for_stage([0.0, 0.0], [1.0])

        assert result == [0.0, 0.0]

    def test_success_gate_falls_back_to_supplementary_samples(self):
        cb = object.__new__(CurriculumCallback)
        cb.curriculum_manager = MagicMock()
        cb.curriculum_manager.current_threshold = StageThreshold(min_success_rate=0.5)

        result = cb._success_rates_for_stage(None, [1.0, 0.0])

        assert result == [1.0, 0.0]

    def test_forward_velocity_gate_prefers_main_eval_sample(self):
        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_forward_velocities = [[1.0, 1.5], [2.0, 2.5]]

        result = cb._forward_velocities_for_eval(2, [9.0])

        assert result == [2.0, 2.5]

    def test_forward_velocity_gate_supports_plain_eval_callback(self):
        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock(spec=[])

        result = cb._forward_velocities_for_eval(1, [0.5, 0.75])

        assert result == [0.5, 0.75]

    def test_eval_callback_path_gates_on_main_eval_velocity_sample(self):
        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_forward_velocities = [[1.0, 1.5, 2.0]]
        cb.curriculum_manager = MagicMock()
        cb.curriculum_manager.current_threshold = StageThreshold(min_avg_forward_vel=1.0)
        cb.curriculum_manager.should_advance.return_value = False
        cb._read_latest_eval = MagicMock(return_value=([100.0] * 3, [1000.0] * 3, None, 1))
        cb._run_supplementary_eval = MagicMock(return_value=([9.0], [1.0], []))
        cb._log_locomotion_metrics = MagicMock()

        assert cb._on_step_with_eval_callback() is True

        cb.curriculum_manager.should_advance.assert_called_once_with(
            [100.0] * 3,
            [1000.0] * 3,
            [1.0, 1.5, 2.0],
            None,
            # No stance panel: this stage gates on reward_and_length/v1, so
            # the stance capture is not consulted at all.
            None,
        )


class TestStageWarmupCallbackMocked:
    """Test StageWarmupCallback lifecycle without SB3 training."""

    def test_warmup_applies_reduced_clip_range(self):
        """Warmup should set clip_range to the configured small value (PPO)."""
        pytest.importorskip("stable_baselines3")
        cb = StageWarmupCallback(warmup_timesteps=100_000, warmup_clip_range=0.02, warmup_ent_coef=0.02)

        mock_model = MagicMock()
        mock_model.clip_range = lambda _: 0.2
        mock_model.ent_coef = 0.01
        # Ensure detected as PPO (no log_ent_coef)
        del mock_model.log_ent_coef
        cb.model = mock_model

        cb._on_training_start()

        # clip_range should be replaced with _ConstantSchedule(0.02)
        assert mock_model.clip_range(0.5) == pytest.approx(0.02)
        assert mock_model.ent_coef == 0.02

    def test_warmup_restores_original_values(self):
        """After warmup_timesteps, original clip_range and ent_coef should be restored (PPO)."""
        pytest.importorskip("stable_baselines3")
        cb = StageWarmupCallback(warmup_timesteps=100, warmup_clip_range=0.02, warmup_ent_coef=0.02)

        original_clip = MagicMock()
        original_ent = 0.005
        mock_model = MagicMock()
        mock_model.clip_range = original_clip
        mock_model.ent_coef = original_ent
        # Ensure detected as PPO (no log_ent_coef)
        del mock_model.log_ent_coef
        cb.model = mock_model

        cb._on_training_start()

        # Simulate reaching warmup_timesteps
        cb.num_timesteps = 100
        assert cb._on_step() is True
        assert cb._warmup_done is True
        assert mock_model.clip_range == original_clip
        assert mock_model.ent_coef == original_ent

    def test_warmup_applies_reduced_lr_for_sac(self):
        """Warmup should reduce LR and seed log_ent_coef for SAC models.

        SAC's train() reads log_ent_coef (auto mode), not the ent_coef
        attribute, so the warm-up value must be written into the tensor.
        """
        import math

        pytest.importorskip("stable_baselines3")
        torch = pytest.importorskip("torch")

        cb = StageWarmupCallback(
            warmup_timesteps=100_000,
            warmup_ent_coef=0.02,
            warmup_lr_scale=0.1,
        )

        mock_model = MagicMock()
        mock_model.lr_schedule = lambda _: 3e-4
        mock_model.ent_coef = "auto"
        mock_model.log_ent_coef = torch.tensor(0.0)
        cb.model = mock_model

        cb._on_training_start()

        assert cb._is_sac is True
        # LR should be reduced by warmup_lr_scale
        assert mock_model.lr_schedule(1.0) == pytest.approx(3e-5)
        # The ent_coef attribute is left alone (SAC ignores it in auto mode);
        # the learned tensor is seeded at log(warmup_ent_coef) instead.
        assert mock_model.ent_coef == "auto"
        assert mock_model.log_ent_coef.item() == pytest.approx(math.log(0.02))

    def test_warmup_restores_sac_values(self):
        """After warmup, SAC LR schedule and auto-entropy should be restored."""
        pytest.importorskip("stable_baselines3")
        torch = pytest.importorskip("torch")

        cb = StageWarmupCallback(
            warmup_timesteps=100,
            warmup_ent_coef=0.02,
            warmup_lr_scale=0.1,
        )

        def original_lr_schedule(_):
            return 3e-4

        mock_model = MagicMock()
        mock_model.lr_schedule = original_lr_schedule
        mock_model.ent_coef = "auto"
        mock_model.log_ent_coef = torch.tensor(0.5)
        cb.model = mock_model

        cb._on_training_start()

        # Simulate auto-tuning progress during the warm-up window
        mock_model.log_ent_coef.data.fill_(-1.25)

        # Simulate reaching warmup_timesteps
        cb.num_timesteps = 100
        assert cb._on_step() is True
        assert cb._warmup_done is True
        assert mock_model.lr_schedule is original_lr_schedule
        assert mock_model.ent_coef == "auto"
        # Tuning progress made during warm-up is kept, not rolled back
        assert mock_model.log_ent_coef.item() == pytest.approx(-1.25)

    def test_on_step_noop_after_warmup(self):
        """_on_step returns True immediately once warmup is done."""
        cb = object.__new__(StageWarmupCallback)
        cb._warmup_done = True
        assert cb._on_step() is True


class TestRewardRampCallbackMocked:
    """Test RewardRampCallback logic without SB3 training."""

    def test_sets_start_value_on_training_start(self):
        """_on_training_start should set the attribute to start_value."""
        cb = object.__new__(RewardRampCallback)
        cb.attr_name = "forward_vel_weight"
        cb.start_value = 0.1
        cb.end_value = 1.0
        cb.ramp_timesteps = 500_000
        cb._last_set_value = None

        mock_venv = MagicMock()
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_venv
        cb.model = mock_model

        cb._on_training_start()

        # Should have called env_method to set 0.1
        inner = mock_venv.venv
        inner.env_method.assert_called_once_with("set_reward_weight", "forward_vel_weight", 0.1)
        assert cb._last_set_value == 0.1

    def test_ramp_complete_sets_end_value(self):
        """After ramp_timesteps, the attribute should be set to end_value."""
        cb = object.__new__(RewardRampCallback)
        cb.attr_name = "forward_vel_weight"
        cb.start_value = 0.1
        cb.end_value = 1.0
        cb.ramp_timesteps = 1000
        cb._last_set_value = 0.5

        mock_venv = MagicMock()
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_venv
        cb.model = mock_model
        cb.num_timesteps = 1000

        result = cb._on_step()

        assert result is True
        inner = mock_venv.venv
        inner.env_method.assert_called_with("set_reward_weight", "forward_vel_weight", 1.0)
        assert cb._last_set_value == 1.0

    def test_ramp_midpoint_value(self):
        """At 50% through ramp, value should be midpoint between start and end."""
        cb = object.__new__(RewardRampCallback)
        cb.attr_name = "forward_vel_weight"
        cb.start_value = 0.0
        cb.end_value = 1.0
        cb.ramp_timesteps = 1000
        cb._last_set_value = None
        cb._last_update_bucket = -1

        mock_venv = MagicMock()
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_venv
        cb.model = mock_model
        cb.num_timesteps = 500

        cb._on_step()

        inner = mock_venv.venv
        call_args = inner.env_method.call_args
        set_value = call_args[0][2]
        assert set_value == pytest.approx(0.5, abs=0.01)

    def test_no_update_when_value_unchanged(self):
        """Quantised value that hasn't changed should not trigger env_method."""
        cb = object.__new__(RewardRampCallback)
        cb.attr_name = "forward_vel_weight"
        cb.start_value = 0.0
        cb.end_value = 1.0
        cb.ramp_timesteps = 1_000_000
        cb._last_set_value = 0.0  # Already set to quantised value
        cb._last_update_bucket = -1

        mock_venv = MagicMock()
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_venv
        cb.model = mock_model
        # With 1M ramp steps and timestep=1, progress=0.000001, quantised to 0.0
        cb.num_timesteps = 1

        cb._on_step()

        inner = mock_venv.venv
        inner.env_method.assert_not_called()


class TestReadLatestEval:
    """Test CurriculumCallback._read_latest_eval npz reading logic."""

    def test_returns_none_when_no_log_path(self):
        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock(spec=[])  # no log_path attr
        cb._last_seen_n_evals = 0
        rewards, lengths, successes, n = cb._read_latest_eval()
        assert rewards is None
        assert lengths is None
        assert successes is None

    def test_returns_none_when_npz_missing(self, tmp_path):
        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path / "evaluations")
        cb._last_seen_n_evals = 0
        rewards, lengths, successes, n = cb._read_latest_eval()
        assert rewards is None

    def test_reads_latest_eval_from_npz(self, tmp_path):
        import numpy as np

        # Create a fake evaluations.npz
        eval_rewards = np.array([[10.0, 20.0], [30.0, 40.0]])
        eval_lengths = np.array([[100.0, 200.0], [300.0, 400.0]])
        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=eval_rewards,
            ep_lengths=eval_lengths,
        )

        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path / "evaluations")
        cb._last_seen_n_evals = 0

        rewards, lengths, successes, n = cb._read_latest_eval()

        assert rewards == [30.0, 40.0]  # last row
        assert lengths == [300.0, 400.0]
        assert successes is None  # npz has no successes array
        assert n == 2
        assert cb._last_seen_n_evals == 2

    def test_no_new_eval_returns_none(self, tmp_path):
        import numpy as np

        eval_rewards = np.array([[10.0, 20.0]])
        eval_lengths = np.array([[100.0, 200.0]])
        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=eval_rewards,
            ep_lengths=eval_lengths,
        )

        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path / "evaluations")
        cb._last_seen_n_evals = 1  # Already seen

        rewards, lengths, successes, n = cb._read_latest_eval()
        assert rewards is None


class TestReadLatestEvalSuccesses:
    """_read_latest_eval returns per-episode successes when SB3 saved them."""

    def test_reads_successes_array(self, tmp_path):
        import numpy as np

        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=np.array([[10.0, 20.0], [30.0, 40.0]]),
            ep_lengths=np.array([[100.0, 200.0], [300.0, 400.0]]),
            successes=np.array([[0.0, 1.0], [1.0, 1.0]]),
        )

        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock()
        # SB3's EvalCallback stores log_path as the "<dir>/evaluations"
        # file prefix (np.savez appends ".npz").
        cb.eval_callback.log_path = str(tmp_path / "evaluations")
        cb._last_seen_n_evals = 0

        rewards, lengths, successes, n = cb._read_latest_eval()
        assert successes == [1.0, 1.0]  # latest eval row
        assert n == 2

    def test_reads_from_sb3_prefix_convention(self, tmp_path):
        """The npz path is derived from SB3's file-prefix log_path, not a dir."""
        import numpy as np

        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=np.array([[10.0, 20.0]]),
            ep_lengths=np.array([[100.0, 200.0]]),
        )

        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path / "evaluations")
        cb._last_seen_n_evals = 0

        rewards, lengths, successes, n = cb._read_latest_eval()
        assert rewards == [10.0, 20.0]
        assert lengths == [100.0, 200.0]
        assert successes is None
        assert n == 1
