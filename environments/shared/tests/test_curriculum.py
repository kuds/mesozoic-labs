"""Tests for CurriculumManager."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared.config import load_all_stages
from environments.shared.curriculum import (
    CurriculumCallback,
    CurriculumManager,
    StageThreshold,
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

    def test_record_eval_with_forward_velocities(self, manager):
        summary = manager.record_eval(
            [10.0, 20.0], [100.0, 200.0], forward_velocities=[1.0, 2.0]
        )
        assert summary["mean_forward_vel"] == pytest.approx(1.5)


class TestThresholdsFromConfigs:
    """Test thresholds_from_configs helper."""

    def test_extracts_from_real_configs(self):
        configs = load_all_stages("velociraptor")
        thresholds = thresholds_from_configs(configs)
        assert isinstance(thresholds, dict)
        # Any stage that has curriculum_kwargs should appear
        for stage, fields in thresholds.items():
            assert isinstance(stage, int)
            assert isinstance(fields, dict)

    def test_handles_empty_curriculum_kwargs(self):
        configs = {
            1: {"curriculum_kwargs": {}},
            2: {"curriculum_kwargs": {"min_avg_reward": 50.0}},
        }
        thresholds = thresholds_from_configs(configs)
        assert 1 not in thresholds  # empty kwargs -> no entry
        assert thresholds[2] == {"min_avg_reward": 50.0}

    def test_handles_all_threshold_fields(self):
        configs = {
            1: {
                "curriculum_kwargs": {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 100,
                    "min_avg_forward_vel": 0.5,
                    "required_consecutive": 3,
                }
            }
        }
        thresholds = thresholds_from_configs(configs)
        assert thresholds[1]["min_avg_reward"] == 10.0
        assert thresholds[1]["min_avg_episode_length"] == 100
        assert thresholds[1]["min_avg_forward_vel"] == 0.5
        assert thresholds[1]["required_consecutive"] == 3

    def test_missing_curriculum_key(self):
        configs = {1: {"env_kwargs": {}}}  # no curriculum_kwargs
        thresholds = thresholds_from_configs(configs)
        assert thresholds == {}


class TestCurriculumCallback:
    """Test CurriculumCallback with mocked model and eval_env."""

    @pytest.fixture
    def manager(self):
        return CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_eval_episodes": 2,
                    "required_consecutive": 1,
                },
            },
            start_stage=1,
        )

    def _make_eval_env(self, episode_rewards, episode_lengths, forward_vels=None):
        """Create a mock VecEnv that runs predetermined episodes."""
        mock_env = MagicMock()
        mock_env.reset.return_value = np.zeros(10)

        # Build step returns for each episode
        step_sequences = []
        for ep_idx, length in enumerate(episode_lengths):
            per_step_reward = episode_rewards[ep_idx] / length
            fwd_vel = forward_vels[ep_idx] if forward_vels else 1.0
            for step in range(int(length)):
                done = step == int(length) - 1
                info: Dict[str, Any] = {
                    "forward_vel": fwd_vel,
                    "reward_energy": -0.01,
                    "pelvis_height": 0.5,
                    "l_foot_contact": 1.0 if step % 2 == 0 else 0.0,
                    "r_foot_contact": 0.0 if step % 2 == 0 else 1.0,
                }
                if done:
                    info["termination_reason"] = "truncated"
                step_sequences.append(
                    (
                        np.zeros(10),
                        np.array([per_step_reward]),
                        np.array([done]),
                        [info],
                    )
                )

        mock_env.step.side_effect = step_sequences
        return mock_env

    def test_init(self, manager):
        mock_env = MagicMock()
        cb = CurriculumCallback(manager, mock_env, eval_freq=5000, n_eval_episodes=5)
        assert cb.eval_freq == 5000
        assert cb.n_eval_episodes == 5
        assert cb.ready_to_advance is False

    def test_on_step_returns_true_before_eval_freq(self, manager):
        mock_env = MagicMock()
        cb = CurriculumCallback(manager, mock_env, eval_freq=10000)
        cb.num_timesteps = 5000
        cb._last_eval_step = 0
        cb.model = MagicMock()
        assert cb._on_step() is True

    def test_on_step_returns_true_on_final_stage(self, manager):
        manager.advance()  # -> stage 2
        manager.advance()  # -> stage 3 (final)
        mock_env = MagicMock()
        cb = CurriculumCallback(manager, mock_env, eval_freq=100)
        cb.num_timesteps = 200
        cb._last_eval_step = 0
        assert cb._on_step() is True

    @patch("environments.shared.curriculum.log_eval_metrics")
    def test_on_step_runs_eval_and_advances(self, mock_log, manager):
        eval_env = self._make_eval_env(
            episode_rewards=[50.0, 50.0],
            episode_lengths=[100, 100],
            forward_vels=[1.5, 1.5],
        )
        cb = CurriculumCallback(
            manager, eval_env, eval_freq=1000, n_eval_episodes=2
        )
        cb.num_timesteps = 1000
        cb._last_eval_step = 0
        cb.model = MagicMock()
        cb.model.predict.return_value = (np.zeros(10), None)

        result = cb._on_step()
        # Thresholds should be met -> ready_to_advance
        assert cb.ready_to_advance is True
        assert result is False  # Stops training
        mock_log.assert_called_once()

    @patch("environments.shared.curriculum.log_eval_metrics")
    def test_on_step_continues_when_thresholds_not_met(self, mock_log, manager):
        eval_env = self._make_eval_env(
            episode_rewards=[1.0, 1.0],  # below threshold of 10.0
            episode_lengths=[100, 100],
        )
        cb = CurriculumCallback(
            manager, eval_env, eval_freq=1000, n_eval_episodes=2
        )
        cb.num_timesteps = 1000
        cb._last_eval_step = 0
        cb.model = MagicMock()
        cb.model.predict.return_value = (np.zeros(10), None)

        result = cb._on_step()
        assert cb.ready_to_advance is False
        assert result is True
