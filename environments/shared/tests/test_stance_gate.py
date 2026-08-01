"""Tests for the ``stance_quality/v1`` advancement gate.

The measured anchors these assert against, all at reset noise 0.05 and all
using ``derive_stance_info``'s 0.1 N contact threshold:

* the zero-action statue -- 119/120 = 99.17% full-horizon pooled over three
  independent 40-seed blocks, unsupported duty 0.000, reward 3271.8 +/- 12.0;
* run ``20260801_021545``'s best checkpoint -- full-horizon on every rolled
  out episode, reward 2347.67, unsupported duty **0.284**.

The gate exists to pass the first and reject the second.  The old
``reward_and_length/v1`` configuration passed both.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from environments.shared.curriculum.stance_gate import (
    STANCE_GATE_KIND,
    StanceGateThresholds,
    episode_unsupported_duty,
    evaluate_stance_gate,
    mean_upper_confidence_bound,
    one_sided_t95,
    stance_panel_from_episode_duties,
    summarize_stance_panel,
)

HORIZON = 1000
SETTLE = 200

#: The adopted T-Rex 1a thresholds.
TREX_1A = StanceGateThresholds(
    min_full_horizon_fraction=0.95,
    max_unsupported_duty=0.02,
    max_unsupported_duty_ucb=0.02,
    settle_steps=SETTLE,
    min_eval_episodes=40,
    min_avg_reward=1950.0,
)


def _episode(duty: float, length: int = HORIZON) -> np.ndarray:
    """A per-step unsupported trace whose post-settle duty is ``duty``."""
    flags = np.zeros(length)
    measurable = max(0, length - SETTLE)
    flags[SETTLE : SETTLE + int(round(duty * measurable))] = 1.0
    return flags


def _panel(duty, reward, *, n=40, full_fraction=1.0, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_full = int(round(n * full_fraction))
    lengths = [HORIZON] * n_full + [400] * (n - n_full)
    traces = [_episode(max(0.0, duty + (rng.normal(0, jitter) if jitter else 0.0)), L) for L in lengths]
    return summarize_stance_panel(
        episode_lengths=lengths,
        episode_unsupported=traces,
        episode_rewards=[reward] * n,
        horizon=HORIZON,
        settle_steps=SETTLE,
    )


class TestDiscrimination:
    """The gate must separate the two policies the reward gate could not."""

    def test_statue_passes(self):
        passed, failures = evaluate_stance_gate(_panel(0.000, 3271.8), TREX_1A)
        assert passed, failures

    def test_statue_passes_with_its_measured_one_in_120_nosedive(self):
        # 39/40 = 0.975 full-horizon still clears the 0.95 floor, which is the
        # point of gating on a fraction rather than demanding a clean sweep.
        passed, failures = evaluate_stance_gate(_panel(0.000, 3250.0, full_fraction=39 / 40), TREX_1A)
        assert passed, failures

    def test_chatterer_fails_on_duty(self):
        """Run 20260801_021545's best model: passed the old gate, must fail this one."""
        panel = _panel(0.284, 2347.7)
        passed, failures = evaluate_stance_gate(panel, TREX_1A)
        assert not passed
        assert any("mean_unsupported_duty" in f for f in failures)
        assert any("unsupported_duty_ucb" in f for f in failures)
        # It cleared the retired reward criterion, which is exactly why the
        # gate had to stop being about reward.
        assert panel.mean_reward > TREX_1A.min_avg_reward
        assert panel.full_horizon_fraction >= TREX_1A.min_full_horizon_fraction

    def test_quiet_active_policy_passes(self):
        passed, failures = evaluate_stance_gate(_panel(0.005, 3100.0, jitter=0.004, seed=3), TREX_1A)
        assert passed, failures

    def test_reward_rail_rejects_a_policy_that_threw_away_the_return(self):
        passed, failures = evaluate_stance_gate(_panel(0.001, 1200.0), TREX_1A)
        assert not passed
        assert any("rail" in f for f in failures)

    def test_falling_policy_fails_on_horizon(self):
        passed, failures = evaluate_stance_gate(_panel(0.000, 3000.0, full_fraction=0.5), TREX_1A)
        assert not passed
        assert any("full_horizon_fraction" in f for f in failures)


class TestFailedEpisodesAreExcludedFromDuty:
    """Decided 2026-08-01: failures do not contribute duty."""

    def test_short_episode_duty_is_not_counted(self):
        # One clean full-horizon episode and one that fell at step 400 while
        # airborne throughout its measurable tail. Scoring the failure would
        # drag the mean to ~0.5; excluding it leaves the survivor's 0.0.
        lengths = [HORIZON, 400]
        traces = [_episode(0.0, HORIZON), _episode(1.0, 400)]
        panel = stance_panel_from_episode_duties(
            episode_lengths=lengths,
            episode_duties=[episode_unsupported_duty(t, settle_steps=SETTLE) for t in traces],
            episode_rewards=[3271.8, 900.0],
            horizon=HORIZON,
        )
        assert panel.n_duty_episodes == 1
        assert panel.mean_unsupported_duty == pytest.approx(0.0)
        assert panel.full_horizon_fraction == pytest.approx(0.5)

    def test_panel_with_no_survivors_fails_closed(self):
        panel = _panel(0.0, 3000.0, full_fraction=0.0)
        assert panel.n_duty_episodes == 0
        assert panel.mean_unsupported_duty == math.inf
        passed, failures = evaluate_stance_gate(panel, TREX_1A)
        assert not passed
        assert any("no full-horizon episode" in f for f in failures)

    def test_unmeasurable_duty_is_dropped_not_zeroed(self):
        """A NaN duty must not be read as a perfect 0.0."""
        panel = stance_panel_from_episode_duties(
            episode_lengths=[HORIZON, HORIZON],
            episode_duties=[float("nan"), 0.5],
            episode_rewards=[3000.0, 3000.0],
            horizon=HORIZON,
        )
        assert panel.n_duty_episodes == 1
        assert panel.mean_unsupported_duty == pytest.approx(0.5)


class TestSettlingWindow:
    def test_duty_before_the_window_is_ignored(self):
        # Airborne for the whole settling prefix, planted afterwards.
        flags = np.zeros(HORIZON)
        flags[:SETTLE] = 1.0
        assert episode_unsupported_duty(flags, settle_steps=SETTLE) == pytest.approx(0.0)

    def test_duty_after_the_window_is_counted(self):
        flags = np.zeros(HORIZON)
        flags[SETTLE:] = 1.0
        assert episode_unsupported_duty(flags, settle_steps=SETTLE) == pytest.approx(1.0)

    def test_episode_shorter_than_the_window_has_no_measurable_tail(self):
        assert episode_unsupported_duty(np.zeros(SETTLE), settle_steps=SETTLE) is None


class TestConfidenceBound:
    def test_bound_is_an_upper_bound(self):
        """Direction matters: STAGE1_SPLIT_PLAN 2.3 says LCB, which is a slip.

        Duty must stay BELOW a ceiling, so the conservative bound is the upper
        one. A lower bound would pass a policy whose mean already exceeded the
        ceiling whenever the panel was noisy, inverting the gate.
        """
        values = [0.01, 0.02, 0.03, 0.04, 0.05]
        assert mean_upper_confidence_bound(values) > float(np.mean(values))

    def test_bound_subsumes_the_screening_ceiling(self):
        panel = _panel(0.019, 3000.0, jitter=0.01, seed=11)
        assert panel.unsupported_duty_ucb >= panel.mean_unsupported_duty

    def test_noise_can_fail_a_panel_whose_raw_mean_passes(self):
        """The bound is load-bearing, not decorative.

        Duties alternating 0.008/0.028 have mean 0.018 -- inside the 0.02
        screening ceiling -- but a spread wide enough that the 95% bound lands
        above it. The screening check alone would advance this panel.
        """
        duties = [0.008, 0.028] * 20
        panel = stance_panel_from_episode_duties(
            episode_lengths=[HORIZON] * 40,
            episode_duties=duties,
            episode_rewards=[3000.0] * 40,
            horizon=HORIZON,
        )
        assert panel.mean_unsupported_duty == pytest.approx(0.018)
        assert panel.mean_unsupported_duty <= TREX_1A.max_unsupported_duty
        assert panel.unsupported_duty_ucb > TREX_1A.max_unsupported_duty_ucb
        passed, failures = evaluate_stance_gate(panel, TREX_1A)
        assert not passed
        assert any("unsupported_duty_ucb" in f for f in failures)
        assert not any("mean_unsupported_duty" in f for f in failures)

    def test_zero_variance_panel_returns_the_sample_mean(self):
        assert mean_upper_confidence_bound([0.0] * 40) == pytest.approx(0.0)

    def test_single_observation_fails_closed(self):
        assert mean_upper_confidence_bound([0.0]) == math.inf

    def test_t_quantile_matches_scipy(self):
        stats = pytest.importorskip("scipy.stats")
        for dof in (1, 2, 5, 7, 11, 20, 29, 39, 49, 99, 250, 1000):
            assert one_sided_t95(dof) == pytest.approx(float(stats.t.ppf(0.95, dof)), abs=2e-4)

    def test_bound_matches_scipy_at_the_operating_point(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        sample = np.abs(rng.normal(0.01, 0.008, 40))
        expected = sample.mean() + stats.t.ppf(0.95, 39) * sample.std(ddof=1) / math.sqrt(40)
        # abs, not rel: the table carries the quantile to 5 decimals, which is
        # three orders of magnitude below the 0.02 ceiling it is compared to.
        assert mean_upper_confidence_bound(sample) == pytest.approx(float(expected), abs=1e-6)


class TestUnderPoweredPanel:
    def test_thirty_episodes_cannot_certify_a_forty_episode_bound(self):
        passed, failures = evaluate_stance_gate(_panel(0.000, 3271.8, n=30), TREX_1A)
        assert not passed
        assert any("n_episodes" in f for f in failures)


class TestBackendParity:
    """SB3 and JAX must reach the same verdict from the same panel."""

    @pytest.mark.parametrize(
        "duty,reward,expected",
        [(0.000, 3271.8, True), (0.284, 2347.7, False), (0.005, 3100.0, True)],
    )
    def test_sb3_and_jax_agree(self, duty, reward, expected):
        from environments.shared.curriculum.manager import CurriculumManager
        from environments.shared.jax_curriculum import check_stage_gate

        panel = _panel(duty, reward)

        thresholds = {
            "gate_kind": STANCE_GATE_KIND,
            "min_full_horizon_fraction": TREX_1A.min_full_horizon_fraction,
            "max_unsupported_duty": TREX_1A.max_unsupported_duty,
            "max_unsupported_duty_ucb": TREX_1A.max_unsupported_duty_ucb,
            "settle_steps": TREX_1A.settle_steps,
            "min_eval_episodes": TREX_1A.min_eval_episodes,
            "min_avg_reward": TREX_1A.min_avg_reward,
            "required_consecutive": 1,
        }
        jax_config = {
            "stage": 1,
            "curriculum_kwargs": {
                "gate_schema_version": 1,
                **thresholds,
            },
        }
        jax_verdict = check_stage_gate(
            {
                "n_eval_episodes": panel.n_episodes,
                "full_horizon_fraction": panel.full_horizon_fraction,
                "n_duty_episodes": panel.n_duty_episodes,
                "mean_unsupported_duty": panel.mean_unsupported_duty,
                "unsupported_duty_ucb": panel.unsupported_duty_ucb,
                "mean_episode_return": panel.mean_reward,
            },
            jax_config,
        )

        manager = CurriculumManager(species="trex", stage_thresholds={1: thresholds})
        sb3_verdict = manager.should_advance(
            [panel.mean_reward] * panel.n_episodes,
            [float(HORIZON)] * panel.n_episodes,
            stance_panel=panel,
        )

        assert jax_verdict is expected
        assert sb3_verdict is expected

    def test_jax_raises_rather_than_half_enforcing(self):
        from environments.shared.curriculum.gate_schema import GateSchemaError
        from environments.shared.jax_curriculum import check_stage_gate

        config = {
            "stage": 1,
            "curriculum_kwargs": {
                "gate_schema_version": 1,
                "gate_kind": STANCE_GATE_KIND,
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
            },
        }
        with pytest.raises(GateSchemaError, match="stance quality cannot be checked"):
            check_stage_gate({"mean_episode_return": 3271.8, "mean_episode_length": 1000.0}, config)


class TestManagerFailsClosedWithoutAPanel:
    def test_declaring_the_kind_without_a_panel_refuses_to_advance(self):
        from environments.shared.curriculum.manager import CurriculumManager

        manager = CurriculumManager(
            species="trex",
            stage_thresholds={
                1: {
                    "gate_kind": STANCE_GATE_KIND,
                    "min_full_horizon_fraction": 0.95,
                    "max_unsupported_duty": 0.02,
                    "max_unsupported_duty_ucb": 0.02,
                    "min_eval_episodes": 40,
                    "required_consecutive": 1,
                }
            },
        )
        assert manager.should_advance([3271.8] * 40, [1000.0] * 40) is False
