"""Tests for environments.shared.reporting.gates."""

import pytest

from environments.shared.reporting import evaluate_recorded_gate, evaluate_stage_gate


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


class TestEvaluateStageGate:
    """The advancement decision, which used to live in each trainer.

    Run 20260802_203215 is the regression these cover: a stance-gated stage 1
    whose reward cleared its rail advanced to stage 2 with
    ``publication_gate_passed = True`` recorded beside a stance report reading
    ``GATE: FAIL`` at 10.6x the duty ceiling, because the notebook's private
    checklist knew only about reward and length.
    """

    # As shipped in configs/trex/stance.toml: note there is no
    # min_avg_episode_length, so a reward-only reading of this block has
    # exactly one criterion, which the statue clears by 68%.
    STANCE_CURRICULUM = {
        "gate_kind": "stance_quality/v1",
        "gate_schema_version": 1,
        "min_full_horizon_fraction": 0.95,
        "max_unsupported_duty": 0.02,
        "max_unsupported_duty_ucb": 0.02,
        "settle_steps": 200,
        "min_eval_episodes": 40,
        "min_avg_reward": 1950.0,
        "required_consecutive": 3,
    }

    # The measured stage-1 result: clears the 1950 rail at 2295.5.
    STANCE_RESULTS = {
        "best_model_reward": 2297.0,
        "best_model_length": 1000.0,
        "mean_reward": 2314.7,
        "mean_episode_length": 1000.0,
    }

    def _report(self, passed, failures=(), gate_kind="stance_quality/v1"):
        return {"gate_kind": gate_kind, "passed": passed, "failures": list(failures)}

    def test_stance_stage_without_a_panel_cannot_pass_on_its_reward_rail(self):
        """The exact 20260802_203215 defect: reward clears, nothing measured stance."""
        passed, failures = evaluate_stage_gate(
            self.STANCE_CURRICULUM,
            self.STANCE_RESULTS,
            stage=1,
        )
        assert passed is False
        assert any("no stance panel was measured" in failure for failure in failures)

    def test_stance_stage_fails_with_the_panel_s_own_reasons(self):
        report = self._report(
            False,
            ["mean_unsupported_duty 0.2120 > 0.0200", "unsupported_duty_ucb 0.2153 > 0.0200"],
        )
        passed, failures = evaluate_stage_gate(
            self.STANCE_CURRICULUM,
            self.STANCE_RESULTS,
            stage=1,
            stance_report=report,
        )
        assert passed is False
        assert failures == report["failures"]

    def test_stance_stage_passes_when_the_panel_passes(self):
        passed, failures = evaluate_stage_gate(
            self.STANCE_CURRICULUM,
            self.STANCE_RESULTS,
            stage=1,
            stance_report=self._report(True),
        )
        assert passed is True
        assert failures == []

    def test_stance_stage_rejects_a_verdict_for_a_different_gate(self):
        passed, failures = evaluate_stage_gate(
            self.STANCE_CURRICULUM,
            self.STANCE_RESULTS,
            stage=1,
            stance_report=self._report(True, gate_kind="reward_and_length/v1"),
        )
        assert passed is False
        assert any("different gate" in failure for failure in failures)

    @pytest.mark.parametrize("verdict", [None, "true", 1])
    def test_stance_stage_rejects_a_non_boolean_verdict(self, verdict):
        passed, _ = evaluate_stage_gate(
            self.STANCE_CURRICULUM,
            self.STANCE_RESULTS,
            stage=1,
            stance_report=self._report(verdict),
        )
        assert passed is False

    def test_stance_stage_that_failed_without_naming_a_reason_still_fails(self):
        passed, failures = evaluate_stage_gate(
            self.STANCE_CURRICULUM,
            self.STANCE_RESULTS,
            stage=1,
            stance_report=self._report(False, []),
        )
        assert passed is False
        assert failures  # never silently empty, or the raised message says nothing


class TestEvaluateStageGateRewardAndLength:
    """Stage 2/3 semantics, preserved exactly from the notebook checklist."""

    CURRICULUM = {
        "gate_kind": "reward_and_length/v1",
        "gate_schema_version": 1,
        "min_avg_reward": 100.0,
        "min_avg_episode_length": 750.0,
        "min_avg_forward_vel": 2.0,
        "required_consecutive": 3,
    }

    def test_passes_when_every_declared_criterion_holds(self):
        passed, failures = evaluate_stage_gate(
            self.CURRICULUM,
            {"best_model_reward": 150.0, "best_model_length": 800.0, "best_model_fwd_vel": 2.5},
            stage=2,
        )
        assert (passed, failures) == (True, [])

    def test_names_every_criterion_that_failed(self):
        passed, failures = evaluate_stage_gate(
            self.CURRICULUM,
            {"best_model_reward": 50.0, "best_model_length": 700.0, "best_model_fwd_vel": 1.0},
            stage=2,
        )
        assert passed is False
        assert len(failures) == 3

    def test_undeclared_criteria_do_not_apply(self):
        """stage3 declares no length or velocity floor; absence must not gate."""
        curriculum = {"gate_kind": "reward_and_length/v1", "gate_schema_version": 1, "min_avg_reward": 100.0}
        passed, _ = evaluate_stage_gate(curriculum, {"best_model_reward": 150.0}, stage=3)
        assert passed is True

    def test_falls_back_through_best_eval_to_the_live_mean(self):
        """The trainers write "" for a metric they could not measure."""
        passed, _ = evaluate_stage_gate(
            {"gate_kind": "reward_and_length/v1", "gate_schema_version": 1, "min_avg_reward": 100.0},
            {"best_model_reward": "", "best_eval_reward": "", "mean_reward": 150.0},
            stage=2,
        )
        assert passed is True

    def test_an_unmeasurable_criterion_fails_rather_than_passing(self):
        passed, failures = evaluate_stage_gate(
            {"gate_kind": "reward_and_length/v1", "gate_schema_version": 1, "min_avg_reward": 100.0},
            {"best_model_reward": "", "best_eval_reward": "", "mean_reward": ""},
            stage=2,
        )
        assert passed is False
        assert any("no reward measurement" in failure for failure in failures)


class TestEvaluateStageGateFailsClosed:
    def test_an_undeclared_gate_kind_cannot_pass(self):
        passed, failures = evaluate_stage_gate({"min_avg_reward": 100.0}, {"best_model_reward": 9999.0}, stage=1)
        assert passed is False
        assert any("no gate_kind" in failure for failure in failures)

    def test_an_unknown_gate_kind_cannot_pass(self):
        passed, failures = evaluate_stage_gate(
            {"gate_kind": "stance_quality/v2", "gate_schema_version": 1},
            {"best_model_reward": 9999.0},
            stage=1,
        )
        assert passed is False
        assert any("unknown gate_kind" in failure for failure in failures)

    def test_none_v1_refuses_to_advance(self):
        passed, failures = evaluate_stage_gate(
            {"gate_kind": "none/v1", "gate_schema_version": 1},
            {"best_model_reward": 9999.0},
            stage=1,
        )
        assert passed is False
        assert failures


class TestEvaluateStageGateAdversarial:
    """Probes that returned a spurious PASS against the first implementation.

    Each was measured, not imagined: the assertions below are the observed
    wrong answers, inverted.
    """

    def test_a_nan_metric_does_not_clear_a_floor(self):
        """``nan < threshold`` is False, so an unfiltered NaN passed everything."""
        passed, failures = evaluate_stage_gate(
            {"gate_kind": "reward_and_length/v1", "gate_schema_version": 1, "min_avg_reward": 100.0},
            {"best_model_reward": float("nan")},
            stage=2,
        )
        assert passed is False
        assert any("no reward measurement" in failure for failure in failures)

    def test_an_infinite_metric_does_not_clear_a_floor(self):
        passed, _ = evaluate_stage_gate(
            {"gate_kind": "reward_and_length/v1", "gate_schema_version": 1, "min_avg_reward": 100.0},
            {"best_model_reward": float("-inf")},
            stage=2,
        )
        assert passed is False

    def test_a_nan_still_falls_through_to_a_finite_fallback(self):
        """Rejecting NaN must not break the best_model -> best_eval -> mean chain."""
        passed, _ = evaluate_stage_gate(
            {"gate_kind": "reward_and_length/v1", "gate_schema_version": 1, "min_avg_reward": 100.0},
            {"best_model_reward": float("nan"), "best_eval_reward": 150.0},
            stage=2,
        )
        assert passed is True

    def test_a_declared_gate_with_no_threshold_checks_nothing_and_fails(self):
        """An empty conjunction is vacuously true, so it passed any policy."""
        passed, failures = evaluate_stage_gate(
            {"gate_kind": "reward_and_length/v1", "gate_schema_version": 1},
            {"best_model_reward": -9999.0},
            stage=2,
        )
        assert passed is False
        assert any("checks nothing" in failure for failure in failures)

    def test_a_string_failure_list_is_not_shredded_into_characters(self):
        """A bare string is iterable; comprehending it produced 13 one-char reasons."""
        _, failures = evaluate_stage_gate(
            {"gate_kind": "stance_quality/v1", "gate_schema_version": 1},
            {},
            stage=1,
            stance_report={"gate_kind": "stance_quality/v1", "passed": False, "failures": "duty too high"},
        )
        assert failures == ["duty too high"]


class TestFiniteGateMetricCannotRaise:
    """A gate helper that raises takes a finished stage's artifacts with it.

    `value == ""` is an ELEMENTWISE comparison against a numpy array, so the
    following `if` raised "truth value of an array ... is ambiguous". Reachable
    from any caller that hands through an array-valued metric, and on the JAX
    path there is no try/except above it.
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("abc", None),
            ("3.5", 3.5),
            (1.5, 1.5),
            (float("nan"), None),
            (float("inf"), None),
            (float("-inf"), None),
        ],
    )
    def test_scalar_sentinels(self, value, expected):
        from environments.shared.curriculum.gate_schema import finite_gate_metric

        assert finite_gate_metric(value) == expected or finite_gate_metric(value) is expected

    def test_numpy_inputs_never_raise(self):
        import numpy as np

        from environments.shared.curriculum.gate_schema import finite_gate_metric

        assert finite_gate_metric(np.float64(2.5)) == 2.5
        assert finite_gate_metric(np.float64("nan")) is None
        assert finite_gate_metric(np.array(2.0)) == 2.0
        # The case that used to raise. Unmeasurable is the right answer; what
        # matters is that it is an answer rather than an exception.
        assert finite_gate_metric(np.array([1.0, 2.0])) is None
        assert finite_gate_metric(np.array([])) is None


class TestUnmeasuredPanelMetricsFailByName:
    """Review ER4: an absent or null panel metric is unmeasured, never zero.

    ``build_stage_results_from_eval_data`` now leaves the velocity/success
    keys out when no post-training panel ran; the gate must name the missing
    measurement rather than compare a fabricated 0.0 against the floor.
    """

    CURRICULUM = {
        "gate_kind": "reward_and_length/v1",
        "gate_schema_version": 1,
        "min_avg_reward": -1.0,
        "min_avg_forward_vel": 1.0,
        "min_success_rate": 0.5,
    }

    @pytest.mark.parametrize(
        "results",
        [
            {"mean_reward": 10.0},
            {"mean_reward": 10.0, "mean_forward_vel": None, "mean_success_rate": None},
        ],
        ids=["absent", "null"],
    )
    def test_names_each_unmeasured_criterion(self, results):
        passed, failures = evaluate_stage_gate(self.CURRICULUM, results, stage=2)
        assert passed is False
        assert any("no forward-velocity measurement" in failure for failure in failures)
        assert any("no success-rate measurement" in failure for failure in failures)
        assert not any("0.00" in failure for failure in failures)

    def test_measured_metrics_still_judge_normally(self):
        results = {"mean_reward": 10.0, "mean_forward_vel": 1.5, "mean_success_rate": 0.9}
        assert evaluate_stage_gate(self.CURRICULUM, results, stage=2) == (True, [])
