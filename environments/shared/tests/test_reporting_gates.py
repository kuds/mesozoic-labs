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


class TestStanceQualityRecordedGate:
    """The stance criteria are CEILINGS, so the direction has to be right.

    Reporting them as floors, or omitting them, would describe a stance-gated
    stage as gated on its reward rail alone -- a threshold the statue clears
    by 68%, which is the claim the gate exists to refute.
    """

    CURRICULUM = {
        "min_avg_reward": 1950.0,
        "min_full_horizon_fraction": 0.95,
        "max_unsupported_duty": 0.02,
        "max_unsupported_duty_ucb": 0.02,
        "min_eval_episodes": 40,
        "required_consecutive": 2,
    }

    def _eval(self, duty, ucb=None, horizon=1.0, reward=3271.8):
        return {
            "mean_reward": reward,
            "full_horizon_fraction": horizon,
            "mean_unsupported_duty": duty,
            "unsupported_duty_ucb": duty if ucb is None else ucb,
            "n_episodes": 40,
        }

    def test_statue_panel_passes(self):
        assert evaluate_recorded_gate(self.CURRICULUM, [self._eval(0.0)] * 2) is True

    def test_chatterer_panel_fails_despite_clearing_the_reward_rail(self):
        # Measured: duty 0.319, reward 2133.4 -- above the 1950 rail.
        history = [self._eval(0.319, ucb=0.322, reward=2133.4)] * 2
        assert evaluate_recorded_gate(self.CURRICULUM, history) is False

    def test_duty_is_a_ceiling_not_a_floor(self):
        """A duty far ABOVE the threshold must fail, not pass."""
        assert evaluate_recorded_gate(self.CURRICULUM, [self._eval(0.9)] * 2) is False

    def test_bound_binds_when_the_raw_mean_would_pass(self):
        history = [self._eval(0.018, ucb=0.021)] * 2
        assert evaluate_recorded_gate(self.CURRICULUM, history) is False

    def test_missing_stance_metrics_cannot_prove_a_pass(self):
        """An old history lacking the stance keys is unprovable, not a pass."""
        legacy = [{"mean_reward": 3271.8, "n_episodes": 40}] * 2
        assert evaluate_recorded_gate(self.CURRICULUM, legacy) is None
