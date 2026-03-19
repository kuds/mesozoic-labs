"""Tests for CurriculumManager."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.curriculum import (
    CurriculumCallback,
    CurriculumManager,
    EvalCollapseEarlyStopCallback,
    RewardRampCallback,
    SaveVecNormalizeCallback,
    StageThreshold,
    StageWarmupCallback,
    _ConstantSchedule,
    load_vecnorm_stats,
    thresholds_from_configs,
)


class TestStageThreshold:
    """Test StageThreshold defaults."""

    def test_default_values(self):
        t = StageThreshold()
        assert t.min_avg_reward == float("-inf")
        assert t.min_avg_episode_length == 0.0
        assert t.min_avg_forward_vel == 0.0
        assert t.min_eval_episodes == 10
        assert t.required_consecutive == 3

    def test_custom_values(self):
        t = StageThreshold(min_avg_reward=50.0, required_consecutive=5)
        assert t.min_avg_reward == 50.0
        assert t.required_consecutive == 5

    def test_forward_vel_threshold(self):
        t = StageThreshold(min_avg_forward_vel=0.5)
        assert t.min_avg_forward_vel == 0.5


class TestCurriculumManager:
    """Test CurriculumManager lifecycle."""

    @pytest.fixture
    def manager(self):
        return CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_eval_episodes": 3,
                    "required_consecutive": 2,
                },
                2: {
                    "min_avg_reward": 50.0,
                    "min_avg_episode_length": 200,
                    "min_eval_episodes": 3,
                    "required_consecutive": 2,
                },
            },
            start_stage=1,
        )

    def test_initial_stage(self, manager):
        assert manager.current_stage == 1
        assert not manager.is_final_stage

    def test_current_config_returns_dict(self, manager):
        config = manager.current_config()
        assert "name" in config
        assert "env_kwargs" in config
        assert "ppo_kwargs" in config

    def test_should_not_advance_without_data(self, manager):
        assert not manager.should_advance()

    def test_should_not_advance_below_threshold(self, manager):
        # Reward below threshold
        rewards = [5.0, 5.0, 5.0]
        lengths = [100.0, 100.0, 100.0]
        assert not manager.should_advance(rewards, lengths)

    def test_should_advance_after_consecutive_passes(self, manager):
        rewards = [15.0, 15.0, 15.0]
        lengths = [100.0, 100.0, 100.0]

        # First pass
        assert not manager.should_advance(rewards, lengths)
        # Second consecutive pass -> should advance
        assert manager.should_advance(rewards, lengths)

    def test_consecutive_resets_on_failure(self, manager):
        good_rewards = [15.0, 15.0, 15.0]
        bad_rewards = [5.0, 5.0, 5.0]
        lengths = [100.0, 100.0, 100.0]

        # First pass
        manager.should_advance(good_rewards, lengths)
        # Failure resets counter
        manager.should_advance(bad_rewards, lengths)
        # First pass again (not enough consecutive)
        assert not manager.should_advance(good_rewards, lengths)
        # Second consecutive -> now passes
        assert manager.should_advance(good_rewards, lengths)

    def test_advance_increments_stage(self, manager):
        new_stage = manager.advance()
        assert new_stage == 2
        assert manager.current_stage == 2

    def test_advance_to_final_stage(self, manager):
        manager.advance()
        manager.advance()
        assert manager.current_stage == 3
        assert manager.is_final_stage

    def test_advance_past_final_raises(self, manager):
        manager.advance()
        manager.advance()
        with pytest.raises(RuntimeError, match="Cannot advance past final stage"):
            manager.advance()

    def test_should_not_advance_on_final_stage(self, manager):
        manager.advance()
        manager.advance()
        # Even with good data, can't advance past final
        rewards = [100.0] * 10
        lengths = [500.0] * 10
        assert not manager.should_advance(rewards, lengths)

    def test_record_eval_returns_summary(self, manager):
        summary = manager.record_eval([10.0, 20.0], [100.0, 200.0])
        assert summary["mean_reward"] == 15.0
        assert summary["mean_length"] == 150.0
        assert summary["n_episodes"] == 2

    def test_summary_contains_history(self, manager):
        manager.record_eval([10.0, 20.0], [100.0, 200.0])
        s = manager.summary()
        assert s["species"] == "velociraptor"
        assert s["current_stage"] == 1
        assert len(s["eval_history"][1]) == 1

    def test_min_eval_episodes_enforced(self):
        """Threshold requires 5 episodes but we only provide 3."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {"min_avg_reward": 0.0, "min_eval_episodes": 5, "required_consecutive": 1},
            },
        )
        # Only 3 episodes provided
        assert not mgr.should_advance([100.0, 100.0, 100.0], [500.0, 500.0, 500.0])

    def test_forward_vel_gate_blocks_without_velocity(self):
        """Stage with forward velocity threshold should block if velocity is too low."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_avg_forward_vel": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        # Good reward/length but no forward velocity data -> defaults to 0.0
        assert not mgr.should_advance(rewards, lengths)

    def test_forward_vel_gate_blocks_low_velocity(self):
        """Stage with forward velocity threshold should block if velocity is below threshold."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_avg_forward_vel": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        low_vels = [0.1, 0.2, 0.1]
        assert not mgr.should_advance(rewards, lengths, low_vels)

    def test_forward_vel_gate_passes_with_good_velocity(self):
        """Stage with forward velocity threshold should pass when all metrics met."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_avg_forward_vel": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        good_vels = [1.0, 1.2, 0.8]
        assert mgr.should_advance(rewards, lengths, good_vels)

    def test_success_rate_gate_blocks_low_rate(self):
        """Success rate below threshold should block advancement."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_success_rate": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        low_success = [0.0, 0.0, 1.0]  # mean = 0.33, below 0.5
        assert not mgr.should_advance(rewards, lengths, success_rates=low_success)

    def test_success_rate_gate_passes_high_rate(self):
        """Success rate above threshold should allow advancement."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_success_rate": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        high_success = [1.0, 1.0, 0.0]  # mean = 0.67, above 0.5
        assert mgr.should_advance(rewards, lengths, success_rates=high_success)

    def test_record_eval_with_forward_vel(self):
        """record_eval should include forward velocity in summary."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={1: {"min_avg_reward": 0.0, "required_consecutive": 1}},
        )
        summary = mgr.record_eval([10.0, 20.0], [100.0, 200.0], forward_velocities=[1.0, 2.0])
        assert summary["mean_forward_vel"] == 1.5

    def test_record_eval_with_success_rate(self):
        """record_eval should include success rate in summary."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={1: {"min_avg_reward": 0.0, "required_consecutive": 1}},
        )
        summary = mgr.record_eval([10.0, 20.0], [100.0, 200.0], success_rates=[1.0, 0.0])
        assert summary["mean_success_rate"] == 0.5


class TestThresholdsFromConfigs:
    """Test extracting thresholds from loaded TOML configs."""

    def test_extracts_reward_threshold(self):
        configs = {
            1: {"curriculum_kwargs": {"min_avg_reward": 10.0, "required_consecutive": 2}},
            2: {"curriculum_kwargs": {"min_avg_reward": 50.0}},
            3: {"curriculum_kwargs": {}},
        }
        thresholds = thresholds_from_configs(configs)
        assert thresholds[1]["min_avg_reward"] == 10.0
        assert thresholds[1]["required_consecutive"] == 2
        assert thresholds[2]["min_avg_reward"] == 50.0
        assert 3 not in thresholds  # empty curriculum_kwargs -> no entry

    def test_extracts_all_threshold_fields(self):
        configs = {
            1: {
                "curriculum_kwargs": {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 100,
                    "min_avg_forward_vel": 0.5,
                    "min_success_rate": 0.3,
                    "required_consecutive": 3,
                },
            },
        }
        thresholds = thresholds_from_configs(configs)
        assert thresholds[1]["min_avg_forward_vel"] == 0.5
        assert thresholds[1]["min_success_rate"] == 0.3

    def test_empty_configs(self):
        thresholds = thresholds_from_configs({})
        assert thresholds == {}

    def test_with_real_configs(self):
        """Integration test: extract thresholds from actual TOML configs."""
        from environments.shared.config import load_all_stages

        configs = load_all_stages("velociraptor")
        thresholds = thresholds_from_configs(configs)
        assert isinstance(thresholds, dict)


class TestLoadVecnormStats:
    """Test that VecNormalize stats are correctly carried across stages."""

    @pytest.fixture
    def vec_envs(self):
        """Create a pair of VecNormalize-wrapped dummy envs for stage 1."""
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from environments.velociraptor.envs.raptor_env import RaptorEnv

        def _make():
            env = RaptorEnv()
            return Monitor(env)

        train = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True)
        eval_ = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True)
        yield train, eval_
        train.close()
        eval_.close()

    def test_stats_carried_forward(self, vec_envs, tmp_path):
        """obs_rms/ret_rms are copied from saved file into fresh envs."""
        import numpy as np
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from environments.shared.curriculum import load_vecnorm_stats
        from environments.velociraptor.envs.raptor_env import RaptorEnv

        train_env, _ = vec_envs

        # Run a few steps so the running mean/var diverge from defaults
        obs = train_env.reset()
        for _ in range(50):
            action = [train_env.action_space.sample()]
            obs, _, dones, _ = train_env.step(action)
            if dones[0]:
                train_env.reset()

        # Snapshot the trained stats
        saved_obs_mean = train_env.obs_rms.mean.copy()
        saved_obs_var = train_env.obs_rms.var.copy()

        # Save to disk
        save_path = str(tmp_path / "stage1_final_vecnorm.pkl")
        train_env.save(save_path)

        # Create fresh envs (simulating a new stage)
        def _make():
            return Monitor(RaptorEnv())

        new_train = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True)
        new_eval = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True)

        # Before loading, stats should be at defaults (mean≈0, var≈1)
        assert not np.allclose(new_train.obs_rms.mean, saved_obs_mean)

        # Load stats from the "previous stage"
        loaded = load_vecnorm_stats(save_path, new_train, new_eval)
        assert loaded is True

        # Stats should now match the saved values
        np.testing.assert_array_equal(new_train.obs_rms.mean, saved_obs_mean)
        np.testing.assert_array_equal(new_train.obs_rms.var, saved_obs_var)
        np.testing.assert_array_equal(new_eval.obs_rms.mean, saved_obs_mean)
        np.testing.assert_array_equal(new_eval.obs_rms.var, saved_obs_var)

        # Train env should still be in training mode
        assert new_train.training is True
        assert new_train.norm_reward is True

        # Eval env should NOT be training or normalizing reward
        assert new_eval.training is False
        assert new_eval.norm_reward is False

        new_train.close()
        new_eval.close()

    def test_missing_file_returns_false(self, vec_envs):
        """load_vecnorm_stats returns False when file doesn't exist."""
        from environments.shared.curriculum import load_vecnorm_stats

        train_env, eval_env = vec_envs
        result = load_vecnorm_stats("/nonexistent/path_vecnorm.pkl", train_env, eval_env)
        assert result is False


class TestSaveVecNormalizeCallback:
    """Test SaveVecNormalizeCallback saves VecNormalize on new best model."""

    def test_saves_vecnormalize_on_step(self, tmp_path):
        """_on_step saves VecNormalize to the configured path."""
        save_path = str(tmp_path / "best_model_vecnorm.pkl")

        mock_vec_env = MagicMock()
        mock_model = MagicMock()
        mock_model.get_vec_normalize_env.return_value = mock_vec_env

        cb = object.__new__(SaveVecNormalizeCallback)
        cb.save_path = save_path
        cb.verbose = 0
        cb.model = mock_model

        result = cb._on_step()

        assert result is True
        mock_model.get_vec_normalize_env.assert_called_once()
        mock_vec_env.save.assert_called_once_with(save_path)

    def test_no_op_without_vecnormalize(self, tmp_path):
        """_on_step is a no-op when there is no VecNormalize wrapper."""
        save_path = str(tmp_path / "best_model_vecnorm.pkl")

        mock_model = MagicMock()
        mock_model.get_vec_normalize_env.return_value = None

        cb = object.__new__(SaveVecNormalizeCallback)
        cb.save_path = save_path
        cb.verbose = 0
        cb.model = mock_model

        result = cb._on_step()

        assert result is True
        assert not (tmp_path / "best_model_vecnorm.pkl").exists()

    def test_raises_without_sb3(self):
        """Constructor raises ImportError when SB3 is unavailable."""
        with patch("environments.shared.curriculum._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                SaveVecNormalizeCallback(save_path="/tmp/test.pkl")


class TestCallbacksWithoutSB3:
    """Test that SB3-dependent classes fail gracefully when SB3 is unavailable."""

    def test_curriculum_callback_raises_without_sb3(self):
        with patch("environments.shared.curriculum._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                CurriculumCallback(
                    curriculum_manager=MagicMock(),
                    eval_env=MagicMock(),
                )

    def test_stage_warmup_callback_raises_without_sb3(self):
        with patch("environments.shared.curriculum._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                StageWarmupCallback()

    def test_reward_ramp_callback_raises_without_sb3(self):
        with patch("environments.shared.curriculum._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                RewardRampCallback()

    def test_load_vecnorm_stats_returns_false_without_sb3(self):
        with patch("environments.shared.curriculum._SB3_AVAILABLE", False):
            result = load_vecnorm_stats("/any/path.pkl", MagicMock())
            assert result is False


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


class TestLoadVecnormStatsMocked:
    """Test load_vecnorm_stats logic with mocked SB3 dependencies."""

    def _sb3_mock_modules(self):
        mock_vec_env_mod = MagicMock()
        return {
            "stable_baselines3": MagicMock(),
            "stable_baselines3.common": MagicMock(),
            "stable_baselines3.common.vec_env": mock_vec_env_mod,
        }, mock_vec_env_mod

    def test_missing_file_returns_false_with_sb3(self):
        mods, _ = self._sb3_mock_modules()
        with patch("environments.shared.curriculum._SB3_AVAILABLE", True), patch.dict(sys.modules, mods):
            result = load_vecnorm_stats("/nonexistent/path.pkl", MagicMock())
        assert result is False

    def test_loads_and_applies_stats(self, tmp_path):
        fake_pkl = tmp_path / "vecnorm.pkl"
        fake_pkl.write_bytes(b"fake")

        mods, mock_vec_env_mod = self._sb3_mock_modules()
        mock_prev = MagicMock()
        mock_vec_env_mod.VecNormalize.load.return_value = mock_prev

        mock_train = MagicMock()
        mock_eval = MagicMock()

        with patch("environments.shared.curriculum._SB3_AVAILABLE", True), patch.dict(sys.modules, mods):
            result = load_vecnorm_stats(str(fake_pkl), mock_train, mock_eval)

        assert result is True
        assert mock_train.obs_rms == mock_prev.obs_rms
        assert mock_train.training is True
        assert mock_train.norm_reward is True
        assert mock_eval.training is False
        assert mock_eval.norm_reward is False

    def test_loads_without_eval_env(self, tmp_path):
        fake_pkl = tmp_path / "vecnorm.pkl"
        fake_pkl.write_bytes(b"fake")

        mods, mock_vec_env_mod = self._sb3_mock_modules()
        mock_prev = MagicMock()
        mock_vec_env_mod.VecNormalize.load.return_value = mock_prev

        mock_train = MagicMock()

        with patch("environments.shared.curriculum._SB3_AVAILABLE", True), patch.dict(sys.modules, mods):
            result = load_vecnorm_stats(str(fake_pkl), mock_train, eval_env=None)

        assert result is True
        assert mock_train.obs_rms == mock_prev.obs_rms


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
            patch("environments.shared.curriculum.LocomotionMetrics") as MockMetrics,
            patch("environments.shared.curriculum.log_eval_metrics") as mock_log,
        ):
            MockMetrics.aggregate_episodes.return_value = mock_agg
            cb._log_locomotion_metrics([{"some": "report"}])

        mock_log.assert_called_once_with(mock_agg, 1, step=1000)


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
        """Warmup should reduce LR and fix ent_coef for SAC models."""
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
        assert mock_model.ent_coef == 0.02

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

        # Simulate reaching warmup_timesteps
        cb.num_timesteps = 100
        assert cb._on_step() is True
        assert cb._warmup_done is True
        assert mock_model.lr_schedule is original_lr_schedule
        assert mock_model.ent_coef == "auto"

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
        rewards, lengths, n = cb._read_latest_eval()
        assert rewards is None
        assert lengths is None

    def test_returns_none_when_npz_missing(self, tmp_path):
        cb = object.__new__(CurriculumCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path)
        cb._last_seen_n_evals = 0
        rewards, lengths, n = cb._read_latest_eval()
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
        cb.eval_callback.log_path = str(tmp_path)
        cb._last_seen_n_evals = 0

        rewards, lengths, n = cb._read_latest_eval()

        assert rewards == [30.0, 40.0]  # last row
        assert lengths == [300.0, 400.0]
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
        cb.eval_callback.log_path = str(tmp_path)
        cb._last_seen_n_evals = 1  # Already seen

        rewards, lengths, n = cb._read_latest_eval()
        assert rewards is None


class TestEvalCollapseEarlyStopCallback:
    """Test EvalCollapseEarlyStopCallback early stopping logic."""

    def test_raises_without_sb3(self):
        with patch("environments.shared.curriculum._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                EvalCollapseEarlyStopCallback(eval_callback=MagicMock())

    def test_returns_true_when_no_log_path(self):
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock(spec=[])  # no log_path
        cb._last_seen_n_evals = 0
        cb._peak_reward = float("-inf")
        cb._consecutive_drops = 0
        assert cb._on_step() is True

    def test_returns_true_when_npz_missing(self, tmp_path):
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path)
        cb._last_seen_n_evals = 0
        cb._peak_reward = float("-inf")
        cb._consecutive_drops = 0
        assert cb._on_step() is True

    def test_returns_true_before_min_evals(self, tmp_path):
        import numpy as np

        eval_rewards = np.array([[10.0, 20.0], [15.0, 25.0]])
        np.savez(str(tmp_path / "evaluations.npz"), results=eval_rewards)

        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path)
        cb._last_seen_n_evals = 0
        cb._peak_reward = float("-inf")
        cb._consecutive_drops = 0
        cb.min_evals = 5  # Need 5 evals, only have 2
        cb.drop_fraction = 0.3
        cb.patience = 3
        assert cb._on_step() is True

    def test_returns_true_no_new_eval(self, tmp_path):
        import numpy as np

        eval_rewards = np.array([[10.0]])
        np.savez(str(tmp_path / "evaluations.npz"), results=eval_rewards)

        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path)
        cb._last_seen_n_evals = 1  # Already seen this eval
        cb._peak_reward = float("-inf")
        cb._consecutive_drops = 0
        assert cb._on_step() is True

    def test_stops_after_patience_drops(self, tmp_path):
        import numpy as np

        # 6 evals: peak at eval 3 (50.0), then drops to 20.0
        eval_rewards = np.array(
            [
                [10.0, 10.0],
                [30.0, 30.0],
                [50.0, 50.0],  # peak
                [20.0, 20.0],  # drop 1
                [15.0, 15.0],  # drop 2
                [10.0, 10.0],  # drop 3 -> stop
            ]
        )
        np.savez(str(tmp_path / "evaluations.npz"), results=eval_rewards)

        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path)
        cb._last_seen_n_evals = 0
        cb._peak_reward = float("-inf")
        cb._consecutive_drops = 0
        cb.min_evals = 3
        cb.drop_fraction = 0.3
        cb.patience = 1  # Stop after 1 drop
        cb.num_timesteps = 1000

        result = cb._on_step()
        # latest_mean = 10.0, peak = 50.0, threshold = 35.0
        # 10.0 < 35.0, so consecutive_drops=1 >= patience=1 -> stop
        assert result is False

    def test_resets_drops_on_recovery(self, tmp_path):
        import numpy as np

        # Evals: peak, drop, recovery
        eval_rewards = np.array(
            [
                [50.0, 50.0],
                [40.0, 40.0],
                [30.0, 30.0],
                [20.0, 20.0],
                [10.0, 10.0],
                [45.0, 45.0],  # recovery
            ]
        )
        np.savez(str(tmp_path / "evaluations.npz"), results=eval_rewards)

        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(tmp_path)
        cb._last_seen_n_evals = 0
        cb._peak_reward = float("-inf")
        cb._consecutive_drops = 0
        cb.min_evals = 5
        cb.drop_fraction = 0.3
        cb.patience = 5  # High patience
        cb.num_timesteps = 1000

        result = cb._on_step()
        # latest_mean = 45.0, peak = 50.0, threshold = 35.0
        # 45.0 >= 35.0 -> consecutive_drops reset to 0
        assert result is True
        assert cb._consecutive_drops == 0
