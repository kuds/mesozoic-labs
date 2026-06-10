"""Tests for the JAX curriculum gate logic (jax_curriculum.py).

These tests run without JAX installed — check_stage_gate is pure Python.
"""

from __future__ import annotations

from environments.shared.jax_curriculum import check_stage_gate


class TestCheckStageGate:
    def test_passes_when_reward_meets_threshold(self):
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}
        assert check_stage_gate({"mean_reward": 150.0}, stage_config) is True

    def test_fails_when_reward_below_threshold(self):
        stage_config = {"curriculum_kwargs": {"min_avg_reward": 100.0}}
        assert check_stage_gate({"mean_reward": 50.0}, stage_config) is False

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
        assert check_stage_gate({"mean_reward": min_reward + 1.0}, stage_config) is True
        assert check_stage_gate({"mean_reward": min_reward - 1.0}, stage_config) is False

    def test_missing_threshold_passes_with_warning(self):
        # No threshold configured -> gate passes (matches the SB3
        # CurriculumManager default of -inf), rather than blocking forever.
        assert check_stage_gate({"mean_reward": 0.0}, {}) is True
        assert check_stage_gate({"mean_reward": 0.0}, {"curriculum_kwargs": {}}) is True
