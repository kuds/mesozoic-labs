"""Tests for jax_eval module (CPU-based evaluation for JAX-trained policies)."""

from __future__ import annotations

import numpy as np
import pytest

from environments.shared.jax_eval import EvalConfig, EvalResults, check_stage_gate


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.n_episodes == 25
        assert cfg.max_episode_steps == 1000
        assert cfg.frame_skip == 5
        assert cfg.healthy_z_range == (0.3, 2.0)
        assert cfg.max_tilt_angle == pytest.approx(1.047)
        assert cfg.root_body_id == 1
        assert cfg.sensor_quat_start == 6
        assert cfg.reset_noise_scale == pytest.approx(0.01)
        assert cfg.forward_vel_max == pytest.approx(8.0)

    def test_custom_values(self):
        cfg = EvalConfig(
            n_episodes=10,
            max_episode_steps=500,
            frame_skip=3,
            healthy_z_range=(0.5, 1.5),
            max_tilt_angle=0.8,
            root_body_id=2,
        )
        assert cfg.n_episodes == 10
        assert cfg.max_episode_steps == 500
        assert cfg.frame_skip == 3
        assert cfg.healthy_z_range == (0.5, 1.5)
        assert cfg.max_tilt_angle == pytest.approx(0.8)
        assert cfg.root_body_id == 2


class TestEvalResults:
    def test_empty_results(self):
        results = EvalResults()
        assert results.mean_reward == 0.0
        assert results.std_reward == 0.0
        assert results.mean_length == 0.0
        assert results.std_length == 0.0
        assert results.mean_forward_vel == 0.0
        assert results.mean_distance == 0.0
        assert results.mean_tilt == 0.0
        assert results.mean_height == 0.0

    def test_with_data(self):
        results = EvalResults()
        results.rewards = [10.0, 20.0, 30.0]
        results.lengths = [100, 200, 300]
        results.forward_vels = [1.0, 2.0, 3.0]
        results.distances = [5.0, 10.0, 15.0]
        results.tilt_angles = [0.1, 0.2, 0.3]
        results.pelvis_heights = [0.5, 0.6, 0.7]

        assert results.mean_reward == pytest.approx(20.0)
        assert results.std_reward == pytest.approx(np.std([10.0, 20.0, 30.0]))
        assert results.mean_length == pytest.approx(200.0)
        assert results.std_length == pytest.approx(np.std([100, 200, 300]))
        assert results.mean_forward_vel == pytest.approx(2.0)
        assert results.mean_distance == pytest.approx(10.0)
        assert results.mean_tilt == pytest.approx(0.2)
        assert results.mean_height == pytest.approx(0.6)

    def test_default_diag_reward_components(self):
        results = EvalResults()
        assert "forward" in results.diag_reward_components
        assert "alive" in results.diag_reward_components
        assert "energy" in results.diag_reward_components
        assert "posture" in results.diag_reward_components

    def test_independent_instances(self):
        """Ensure default_factory creates independent lists per instance."""
        r1 = EvalResults()
        r2 = EvalResults()
        r1.rewards.append(1.0)
        assert len(r2.rewards) == 0


class TestCheckStageGate:
    def test_passes_when_above_thresholds(self):
        results = EvalResults()
        results.rewards = [100.0, 200.0, 300.0]
        results.lengths = [500, 600, 700]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=400)
        assert passed is True
        assert failures == []

    def test_fails_on_low_reward(self):
        results = EvalResults()
        results.rewards = [10.0, 20.0, 30.0]
        results.lengths = [500, 600, 700]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=100)
        assert passed is False
        assert len(failures) == 1
        assert "reward" in failures[0]

    def test_fails_on_short_episodes(self):
        results = EvalResults()
        results.rewards = [100.0, 200.0, 300.0]
        results.lengths = [10, 20, 30]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=100)
        assert passed is False
        assert len(failures) == 1
        assert "episode length" in failures[0]

    def test_fails_on_both(self):
        results = EvalResults()
        results.rewards = [1.0, 2.0, 3.0]
        results.lengths = [10, 20, 30]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=100)
        assert passed is False
        assert len(failures) == 2

    def test_passes_with_default_thresholds(self):
        results = EvalResults()
        results.rewards = [-1000.0]
        results.lengths = [1]
        passed, failures = check_stage_gate(results)
        assert passed is True
        assert failures == []

    def test_exact_threshold(self):
        results = EvalResults()
        results.rewards = [50.0]
        results.lengths = [100]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=100)
        assert passed is True
        assert failures == []


class TestSuccessAndVelGates:
    """New gate criteria: forward velocity (stage 2) and success rate (stage 3)."""

    def _results(self):
        results = EvalResults()
        results.rewards = [100.0] * 4
        results.lengths = [500] * 4
        results.forward_vels = [1.0, 2.0, 3.0, 2.0]
        results.successes = [True, True, False, False]
        return results

    def test_mean_success_rate_property(self):
        assert self._results().mean_success_rate == 0.5
        assert EvalResults().mean_success_rate == 0.0

    def test_forward_vel_gate(self):
        passed, failures = check_stage_gate(self._results(), gate_min_forward_vel=1.5)
        assert passed
        passed, failures = check_stage_gate(self._results(), gate_min_forward_vel=5.0)
        assert not passed
        assert any("forward vel" in f for f in failures)

    def test_success_rate_gate(self):
        passed, failures = check_stage_gate(self._results(), gate_min_success_rate=0.5)
        assert passed
        passed, failures = check_stage_gate(self._results(), gate_min_success_rate=0.75)
        assert not passed
        assert any("success rate" in f for f in failures)

    def test_zero_thresholds_disable_new_gates(self):
        results = EvalResults()
        results.rewards = [1.0]
        results.lengths = [10]
        # No forward_vels/successes recorded at all -- gates must not fire.
        passed, failures = check_stage_gate(results)
        assert passed
