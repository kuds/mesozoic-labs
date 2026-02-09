"""Tests for CurriculumManager."""

import pytest

from environments.shared.curriculum import CurriculumManager, StageThreshold


class TestStageThreshold:
    """Test StageThreshold defaults."""

    def test_default_values(self):
        t = StageThreshold()
        assert t.min_avg_reward == float("-inf")
        assert t.min_avg_episode_length == 0.0
        assert t.min_eval_episodes == 10
        assert t.required_consecutive == 3

    def test_custom_values(self):
        t = StageThreshold(min_avg_reward=50.0, required_consecutive=5)
        assert t.min_avg_reward == 50.0
        assert t.required_consecutive == 5


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
