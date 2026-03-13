"""Tests for LocomotionMetrics."""

import numpy as np
import pytest

from environments.shared.metrics import LocomotionMetrics


class TestLocomotionMetrics:
    """Test locomotion metrics computation."""

    @pytest.fixture
    def metrics(self):
        return LocomotionMetrics()

    def test_empty_compute_returns_nan_defaults(self, metrics):
        result = metrics.compute()
        assert result["episode_length"] == 0
        assert np.isnan(result["mean_forward_velocity"])
        assert np.isnan(result["total_distance"])
        assert "mean_forward_velocity" in result
        assert "cost_of_transport" in result

    def test_forward_velocity_stats(self, metrics):
        for _ in range(10):
            metrics.record_step({"forward_vel": 2.0}, reward=1.0)
        result = metrics.compute()
        assert result["mean_forward_velocity"] == pytest.approx(2.0)
        assert result["std_forward_velocity"] == pytest.approx(0.0, abs=1e-6)

    def test_total_distance(self, metrics):
        metrics._dt = 0.1
        for _ in range(10):
            metrics.record_step({"forward_vel": 1.0}, reward=1.0)
        result = metrics.compute()
        assert result["total_distance"] == pytest.approx(1.0)  # 10 * 1.0 * 0.1

    def test_cost_of_transport(self, metrics):
        metrics._dt = 0.1
        for _ in range(10):
            metrics.record_step(
                {"forward_vel": 1.0, "reward_energy": -0.5},
                reward=1.0,
            )
        result = metrics.compute(body_mass=2.0)
        # total_energy = 10 * 0.5 = 5.0
        # distance = 10 * 1.0 * 0.1 = 1.0
        # CoT = 5.0 / (2.0 * 9.81 * 1.0) ≈ 0.255
        expected_cot = 5.0 / (2.0 * 9.81 * 1.0)
        assert result["cost_of_transport"] == pytest.approx(expected_cot, rel=0.01)

    def test_cost_of_transport_zero_distance(self, metrics):
        for _ in range(5):
            metrics.record_step({"forward_vel": 0.0, "reward_energy": -1.0}, reward=0.0)
        result = metrics.compute()
        assert result["cost_of_transport"] == float("inf")

    def test_pelvis_height_stats(self, metrics):
        for h in [0.5, 0.6, 0.5, 0.6]:
            metrics.record_step({"forward_vel": 0.0, "pelvis_height": h})
        result = metrics.compute()
        assert result["mean_pelvis_height"] == pytest.approx(0.55)
        assert result["pelvis_height_variance"] > 0

    def test_prey_distance_tracking(self, metrics):
        distances = [5.0, 4.0, 3.0, 2.0, 1.0, 0.3]
        for d in distances:
            metrics.record_step({"forward_vel": 1.0, "prey_distance": d})
        result = metrics.compute()
        assert result["initial_prey_distance"] == 5.0
        assert result["final_prey_distance"] == 0.3
        assert result["min_prey_distance"] == 0.3

    def test_time_to_target(self, metrics):
        metrics._dt = 0.5
        distances = [5.0, 3.0, 1.0, 0.4]  # reaches <0.5 at step 3
        for d in distances:
            metrics.record_step({"forward_vel": 1.0, "prey_distance": d})
        result = metrics.compute()
        assert result["time_to_target"] == pytest.approx(3 * 0.5)  # 1.5 seconds

    def test_time_to_target_never_reached(self, metrics):
        for d in [5.0, 4.0, 3.0]:
            metrics.record_step({"forward_vel": 1.0, "prey_distance": d})
        result = metrics.compute()
        assert result["time_to_target"] == -1.0

    def test_reward_statistics(self, metrics):
        rewards = [1.0, 2.0, 3.0, 4.0]
        for r in rewards:
            metrics.record_step({"forward_vel": 1.0}, reward=r)
        result = metrics.compute()
        assert result["total_reward"] == pytest.approx(10.0)
        assert result["mean_step_reward"] == pytest.approx(2.5)
        assert result["episode_length"] == 4

    def test_reset_clears_data(self, metrics):
        metrics.record_step({"forward_vel": 1.0})
        metrics.reset()
        result = metrics.compute()
        assert result["episode_length"] == 0
        assert np.isnan(result["mean_forward_velocity"])

    def test_velocity_consistency_perfect(self, metrics):
        for _ in range(10):
            metrics.record_step({"forward_vel": 2.0})
        result = metrics.compute()
        assert result["velocity_consistency"] == pytest.approx(1.0, abs=0.01)

    def test_velocity_consistency_variable(self, metrics):
        for v in [0.0, 4.0, 0.0, 4.0, 0.0, 4.0]:
            metrics.record_step({"forward_vel": v})
        result = metrics.compute()
        assert result["velocity_consistency"] < 1.0

    def test_distance_traveled(self, metrics):
        """distance_traveled uses the last recorded value (cumulative from env)."""
        distances = [0.0, 0.5, 1.2, 2.0, 3.5]
        for d in distances:
            metrics.record_step({"forward_vel": 1.0, "distance_traveled": d})
        result = metrics.compute()
        assert result["distance_traveled"] == pytest.approx(3.5)

    def test_distance_traveled_missing(self, metrics):
        """When distance_traveled is not in info, it should not appear in result."""
        metrics.record_step({"forward_vel": 1.0})
        result = metrics.compute()
        assert "distance_traveled" not in result

    def test_distance_traveled_reset(self, metrics):
        metrics.record_step({"forward_vel": 1.0, "distance_traveled": 5.0})
        metrics.reset()
        result = metrics.compute()
        assert "distance_traveled" not in result


class TestGaitSymmetry:
    """Test gait symmetry computation."""

    def test_perfect_symmetry(self):
        # Perfectly alternating contacts
        left = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=float)
        right = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=float)
        sym = LocomotionMetrics._compute_gait_symmetry(left, right)
        assert sym == pytest.approx(1.0, abs=0.1)

    def test_no_contacts(self):
        left = np.zeros(10)
        right = np.zeros(10)
        sym = LocomotionMetrics._compute_gait_symmetry(left, right)
        assert sym == 0.0

    def test_single_foot_only(self):
        left = np.array([1, 0, 1, 0, 1, 0], dtype=float)
        right = np.zeros(6)
        sym = LocomotionMetrics._compute_gait_symmetry(left, right)
        assert sym == 0.0


class TestAggregateEpisodes:
    """Test multi-episode aggregation."""

    def test_aggregate_two_episodes(self):
        reports = [
            {"mean_forward_velocity": 1.0, "total_reward": 50.0},
            {"mean_forward_velocity": 3.0, "total_reward": 150.0},
        ]
        agg = LocomotionMetrics.aggregate_episodes(reports)
        assert agg["mean_mean_forward_velocity"] == pytest.approx(2.0)
        assert agg["mean_total_reward"] == pytest.approx(100.0)
        assert agg["n_episodes"] == 2.0

    def test_aggregate_empty(self):
        assert LocomotionMetrics.aggregate_episodes([]) == {}
