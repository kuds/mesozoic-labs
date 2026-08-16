"""Tests for environments.shared.curriculum.recovery_gate.

The exact-bound pins come from STAGE1_SPLIT_PLAN §3.4's own recomputed
figures — the module must reproduce the design doc's arithmetic to the
digit or the gate means something other than what was designed.
"""

from __future__ import annotations

import math

import pytest

from environments.shared.curriculum.recovery_gate import (
    RecoveryGateThresholds,
    RecoveryPanel,
    binomial_lcb,
    binomial_ucb,
    episode_recovery_success,
    evaluate_recovery_gate,
    paired_difference_lcb,
    per_push_recovery,
)


class TestExactBinomialBounds:
    def test_reproduces_the_split_plans_pinned_figures(self):
        # "the old trained checkpoint was 34/40 = 85%, whose exact LCB
        # 0.72526 clears 0.70 only narrowly" — §3.4.
        assert binomial_lcb(34, 40) == pytest.approx(0.72526, abs=5e-5)
        # "zero-action under push: 0/40 survival, exact 95% upper bound
        # 7.216%" — §2 / §3.4.
        assert binomial_ucb(0, 40) == pytest.approx(0.07216, abs=5e-5)

    def test_edge_cases(self):
        assert binomial_lcb(0, 40) == 0.0
        assert binomial_ucb(40, 40) == 1.0
        # All-success LCB: alpha**(1/n) in closed form.
        assert binomial_lcb(40, 40) == pytest.approx(0.05 ** (1 / 40), abs=1e-6)

    def test_monotonic_in_successes(self):
        lcbs = [binomial_lcb(k, 40) for k in range(41)]
        assert lcbs == sorted(lcbs)

    def test_invalid_inputs_are_fatal(self):
        with pytest.raises(ValueError):
            binomial_lcb(5, 0)
        with pytest.raises(ValueError):
            binomial_ucb(41, 40)


class TestPairedDifference:
    def test_positive_uniform_difference_bounds_above_zero(self):
        assert paired_difference_lcb([1.0] * 10) == pytest.approx(1.0)
        assert paired_difference_lcb([0.6, 0.7, 0.5, 0.8, 0.6, 0.7]) > 0.0

    def test_fails_closed_on_insufficient_or_nonfinite_pairs(self):
        assert math.isnan(paired_difference_lcb([1.0]))
        assert math.isnan(paired_difference_lcb([0.5, math.nan, 0.5]))


class TestPerPushRecovery:
    def _mask(self, spec: str) -> list[bool]:
        return [c == "1" for c in spec]

    def test_reentry_plus_dwell_recovers(self):
        # Push ends at 4; steps 4-6 unsafe; safe from 7 — dwell 5 fits at 7.
        mask = self._mask("1111000111111111")
        recovered, step = per_push_recovery(mask, push_end_step=4, t_recover_steps=5, dwell_steps=5)
        assert recovered and step == 7

    def test_touching_the_safe_set_without_dwelling_is_not_recovery(self):
        # Safe for 2 steps at 6, then falls again: dwell 5 never fits.
        mask = self._mask("1111000110000000")
        recovered, step = per_push_recovery(mask, push_end_step=4, t_recover_steps=5, dwell_steps=5)
        assert not recovered and step == -1

    def test_reentry_after_the_window_is_too_late(self):
        # First safe step is 11, but t_recover only reaches 8.
        mask = self._mask("1111000000011111")
        recovered, _ = per_push_recovery(mask, push_end_step=4, t_recover_steps=4, dwell_steps=3)
        assert not recovered

    def test_truncated_episode_fails_closed(self):
        # Episode ended before any dwell window could complete.
        mask = self._mask("111100011")
        recovered, _ = per_push_recovery(mask, push_end_step=4, t_recover_steps=5, dwell_steps=5)
        assert not recovered

    def test_episode_success_is_the_full_conjunction(self):
        assert episode_recovery_success(True, [True, True, True])
        assert not episode_recovery_success(True, [True, False, True])
        assert not episode_recovery_success(False, [True, True, True])


_THRESHOLDS = RecoveryGateThresholds(
    min_recovery_success_lcb=0.70,
    t_recover_steps=100,
    dwell_steps=50,
    min_eval_episodes=40,
)


class TestEvaluateRecoveryGate:
    def test_a_strong_panel_passes(self):
        result = evaluate_recovery_gate(RecoveryPanel(tuple([True] * 38 + [False] * 2)), _THRESHOLDS)
        # 38/40: exact LCB 0.845 clears 0.70.
        assert result.passed and result.success_lcb > 0.70

    def test_the_split_plans_marginal_case_fails_at_a_higher_bar(self):
        panel = RecoveryPanel(tuple([True] * 34 + [False] * 6))
        result = evaluate_recovery_gate(
            panel, RecoveryGateThresholds(min_recovery_success_lcb=0.73, t_recover_steps=100, dwell_steps=50)
        )
        # 34/40 LCB 0.72526: narrowly clears 0.70, fails 0.73 — the
        # marginality §3.4 warns about is visible in the statistic.
        assert not result.passed
        assert any("recovery_success_lcb" in f for f in result.failures)

    def test_small_panels_fail_the_floor(self):
        result = evaluate_recovery_gate(RecoveryPanel(tuple([True] * 10)), _THRESHOLDS)
        assert not result.passed
        assert any("min_eval_episodes" in f for f in result.failures)

    def test_declared_paired_criterion_with_missing_nulls_fails_closed(self):
        thresholds = RecoveryGateThresholds(
            min_recovery_success_lcb=0.70,
            t_recover_steps=100,
            dwell_steps=50,
            min_paired_success_delta_lcb=0.2,
        )
        result = evaluate_recovery_gate(RecoveryPanel(tuple([True] * 40)), thresholds)
        assert not result.passed
        assert any("paired null panel missing" in f for f in result.failures)

    def test_paired_criterion_judges_the_difference(self):
        thresholds = RecoveryGateThresholds(
            min_recovery_success_lcb=0.70,
            t_recover_steps=100,
            dwell_steps=50,
            min_paired_success_delta_lcb=0.2,
        )
        good = RecoveryPanel(tuple([True] * 40), paired_null_differences=tuple([0.8] * 40))
        assert evaluate_recovery_gate(good, thresholds).passed
        flat = RecoveryPanel(tuple([True] * 40), paired_null_differences=tuple([0.0] * 40))
        assert not evaluate_recovery_gate(flat, thresholds).passed


class TestGateSchemaRegistration:
    def test_a_complete_recovery_gate_block_validates(self):
        from environments.shared.curriculum.gate_schema import validate_gate_config

        curriculum = {
            "gate_schema_version": 1,
            "gate_kind": "recovery_quality/v1",
            "min_recovery_success_lcb": 0.70,
            "recovery_t_recover_steps": 100,
            "recovery_dwell_steps": 50,
            "min_eval_episodes": 40,
            "timesteps": 3_000_000,
        }
        assert validate_gate_config("recovery", curriculum) == "recovery_quality/v1"

    def test_missing_required_keys_are_fatal(self):
        from environments.shared.curriculum.gate_schema import GateSchemaError, validate_gate_config

        with pytest.raises(GateSchemaError, match="missing required"):
            validate_gate_config(
                "recovery",
                {
                    "gate_schema_version": 1,
                    "gate_kind": "recovery_quality/v1",
                    "min_recovery_success_lcb": 0.70,
                },
            )

    def test_recovery_keys_on_a_stance_gate_are_fatal(self):
        from environments.shared.curriculum.gate_schema import GateSchemaError, validate_gate_config

        with pytest.raises(GateSchemaError):
            validate_gate_config(
                1,
                {
                    "gate_schema_version": 1,
                    "gate_kind": "stance_quality/v1",
                    "min_full_horizon_fraction": 0.95,
                    "max_unsupported_duty": 0.02,
                    "max_unsupported_duty_ucb": 0.02,
                    "recovery_t_recover_steps": 100,
                },
            )
