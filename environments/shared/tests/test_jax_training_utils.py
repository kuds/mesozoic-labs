"""Tests for jax_training_utils — focuses on fixes from the JAX review."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from environments.shared.jax_training_utils import (
    DEFAULT_CSV_FIELDS,
    EpisodeStatsAccumulator,
    StabilityMonitor,
    TrainingCSVLogger,
    compute_episode_stats,
)


# ---------------------------------------------------------------------------
# Fix #8: learning_rate in CSV fields
# ---------------------------------------------------------------------------


class TestCSVFields:
    def test_learning_rate_in_default_fields(self):
        """DEFAULT_CSV_FIELDS should include learning_rate."""
        assert "learning_rate" in DEFAULT_CSV_FIELDS

    def test_csv_logger_writes_learning_rate(self, tmp_path):
        """CSV logger should accept and write a learning_rate column."""
        path = tmp_path / "test.csv"
        logger = TrainingCSVLogger(path)
        logger.log({
            "update": 0,
            "reward_per_step": "0.1",
            "learning_rate": "3.00e-04",
        })
        logger.close()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["learning_rate"] == "3.00e-04"


# ---------------------------------------------------------------------------
# StabilityMonitor
# ---------------------------------------------------------------------------


class TestStabilityMonitor:
    def test_stable_returns_no_message(self):
        mon = StabilityMonitor()
        halt, unstable, msg = mon.check(0.01, 5.0, 1.0, update=0)
        assert not halt
        assert not unstable
        assert msg == ""

    def test_kl_halt(self):
        mon = StabilityMonitor(kl_halt=1e6)
        halt, unstable, msg = mon.check(2e6, 0.0, 0.0, update=5)
        assert halt
        assert unstable
        assert "HALTING" in msg

    def test_consecutive_warnings_reset_on_stable(self):
        mon = StabilityMonitor(kl_warn=0.5, max_warnings=3)
        # Two warnings
        mon.check(1.0, 0.0, 0.0, update=0)
        mon.check(1.0, 0.0, 0.0, update=1)
        assert mon._consecutive_warnings == 2
        # One stable step resets
        mon.check(0.01, 0.0, 0.0, update=2)
        assert mon._consecutive_warnings == 0
        assert mon.total_warnings == 2

    def test_total_warnings_accumulate(self):
        mon = StabilityMonitor(kl_warn=0.5, max_warnings=10)
        mon.check(1.0, 0.0, 0.0, update=0)
        mon.check(0.01, 0.0, 0.0, update=1)  # reset consecutive
        mon.check(1.0, 0.0, 0.0, update=2)
        assert mon.total_warnings == 2


# ---------------------------------------------------------------------------
# EpisodeStatsAccumulator
# ---------------------------------------------------------------------------


class TestEpisodeStatsAccumulator:
    def test_multi_rollout_tracking(self):
        """Episodes spanning multiple rollouts should be tracked correctly."""
        acc = EpisodeStatsAccumulator()
        # Rollout 1: no episodes complete (2 envs, 3 steps)
        rewards1 = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        dones1 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        rets1, lens1 = compute_episode_stats(rewards1, dones1, acc)
        assert len(rets1) == 0

        # Rollout 2: env 0 finishes at step 1
        rewards2 = np.array([[1.0, 2.0], [1.0, 2.0]])
        dones2 = np.array([[0.0, 0.0], [1.0, 0.0]])
        rets2, lens2 = compute_episode_stats(rewards2, dones2, acc)
        assert len(rets2) == 1
        # Total: 3 steps from rollout1 + 2 from rollout2 = 5 steps
        assert lens2[0] == 5
        # Total return: 5 * 1.0 = 5.0
        assert rets2[0] == pytest.approx(5.0)

    def test_without_accumulator(self):
        """Without accumulator, only within-rollout episodes are captured."""
        rewards = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        dones = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        rets, lens = compute_episode_stats(rewards, dones)
        assert len(rets) == 2
        assert lens[0] == 2  # env 0: steps 0-1
        assert lens[1] == 3  # env 1: steps 0-2


# ---------------------------------------------------------------------------
# Fix #13: TOML kwargs override in curriculum
# ---------------------------------------------------------------------------


class TestCurriculumTomlOverride:
    def test_toml_kwargs_override_train_kwargs(self):
        """TOML jax_kwargs should override existing train_kwargs."""
        from environments.shared.jax_curriculum import run_curriculum

        calls = []

        def mock_train(species, stage, **kwargs):
            calls.append(kwargs.copy())
            return {"w": 1.0}, {"mean_reward": 999.0}

        # Patch load_stage_config to return a config with jax_kwargs
        import environments.shared.jax_curriculum as jc
        original_load = jc.load_stage_config

        def mock_load(species, stage):
            return {
                "jax_kwargs": {"learning_rate": 1e-5, "num_updates": 200},
                "env_kwargs": {},
                "curriculum": {"min_avg_reward": -999.0},
            }

        jc.load_stage_config = mock_load
        try:
            run_curriculum(
                "trex",
                mock_train,
                stages=(1,),
                learning_rate=3e-4,  # This should be overridden by TOML
            )
        finally:
            jc.load_stage_config = original_load

        assert len(calls) == 1
        # TOML value (1e-5) should override the kwarg (3e-4)
        assert calls[0]["learning_rate"] == 1e-5
        assert calls[0]["num_updates"] == 200
