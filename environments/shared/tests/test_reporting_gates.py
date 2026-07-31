"""Tests for environments.shared.reporting.gates."""

from environments.shared.reporting import evaluate_recorded_gate


class TestStrictRecordedGate:
    def test_requires_every_enabled_metric_and_consecutive_windows(self):
        curriculum = {
            "min_avg_reward": 100.0,
            "min_avg_forward_vel": 2.0,
            "min_eval_episodes": 10,
            "required_consecutive": 2,
        }
        incomplete = [
            {"mean_reward": 120.0, "n_episodes": 10},
            {"mean_reward": 130.0, "n_episodes": 10},
        ]
        assert evaluate_recorded_gate(curriculum, incomplete) is None

        complete = [
            {"mean_reward": 120.0, "mean_forward_vel": 2.1, "n_episodes": 10},
            {"mean_reward": 130.0, "mean_forward_vel": 2.2, "n_episodes": 10},
        ]
        assert evaluate_recorded_gate(curriculum, complete) is True

    def test_returns_false_when_complete_history_never_passes(self):
        curriculum = {"min_avg_reward": 100.0, "required_consecutive": 2}
        evaluations = [
            {"mean_reward": 120.0, "n_episodes": 10},
            {"mean_reward": 90.0, "n_episodes": 10},
        ]
        assert evaluate_recorded_gate(curriculum, evaluations) is False
