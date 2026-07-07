"""Tests for the JAX curriculum gate logic (jax_curriculum.py).

These tests run without JAX installed — check_stage_gate is pure Python.
"""

from __future__ import annotations

from environments.shared.jax_curriculum import check_stage_gate


class TestCheckStageGate:
    def test_passes_when_reward_meets_threshold(self):
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}
        assert check_stage_gate({"mean_episode_return": 150.0}, stage_config) is True

    def test_fails_when_reward_below_threshold(self):
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}
        assert check_stage_gate({"mean_episode_return": 50.0}, stage_config) is False

    def test_reads_curriculum_kwargs_key_from_real_config(self):
        """Regression: the gate must read the loader's 'curriculum_kwargs'
        key.  It previously read 'curriculum' (the raw TOML table name),
        which load_stage_config never emits — min_reward defaulted to inf
        and every curriculum run stopped after stage 1.
        """
        from environments.shared.config import load_stage_config

        stage_config = load_stage_config("velociraptor", 1)
        assert "curriculum_kwargs" in stage_config
        min_reward = stage_config["curriculum_kwargs"]["min_avg_reward"]
        assert check_stage_gate({"mean_episode_return": min_reward + 1.0}, stage_config) is True
        assert check_stage_gate({"mean_episode_return": min_reward - 1.0}, stage_config) is False

    def test_prefers_episode_return_over_per_step_reward(self):
        """Regression: JaxTrainer eval_metrics carry BOTH a per-step
        'mean_reward' (~0.5-2 for a good policy) and an episode-level
        'mean_episode_return'.  The TOML min_avg_reward thresholds are
        episode-level, so the gate must use the episode return — gating on
        the per-step value failed every well-trained policy.
        """
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}
        realistic_trainer_metrics = {"mean_reward": 1.2, "mean_episode_return": 240.0}
        assert check_stage_gate(realistic_trainer_metrics, stage_config) is True

    def test_falls_back_to_mean_reward_when_no_episode_return(self):
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}
        assert check_stage_gate({"mean_reward": 150.0}, stage_config) is True
        assert check_stage_gate({"mean_reward": 50.0}, stage_config) is False

    def test_missing_threshold_passes_with_warning(self):
        # No threshold configured -> gate passes (matches the SB3
        # CurriculumManager default of -inf), rather than blocking forever.
        assert check_stage_gate({"mean_reward": 0.0}, {}) is True
        assert check_stage_gate({"mean_reward": 0.0}, {"curriculum_kwargs": {}}) is True
