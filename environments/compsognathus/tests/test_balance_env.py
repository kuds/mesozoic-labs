"""Protect the opt-in study's comparison, reward ordering and filter semantics."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from environments.compsognathus.envs.compsognathus_env import CompsognathusEnv
from environments.compsognathus.experiments.balance_calibration import home_reference_report, reward_ordering_report
from environments.compsognathus.experiments.balance_env import (
    ARMS,
    DEFAULT_SHAPING,
    BalanceShaping,
    CompsognathusBalanceEnv,
    balance_shaping_terms,
    make_balance_env,
)
from environments.shared.action_filter import apply_low_pass, low_pass_alpha


def test_same_total_load_prefers_balanced_stance_without_erasing_single_support_recovery():
    loads = [(0.5, 0.5), (0.75, 0.25), (0.9, 0.1), (1.0, 0.0), (0.0, 0.0)]
    scores = [balance_shaping_terms(right, left, 1.0)["balance_shaping_reward"] for right, left in loads]
    assert scores == pytest.approx([0.75, 0.675, 0.18, -0.15, -0.25])
    assert all(a > b for a, b in zip(scores, scores[1:]))
    # The canonical upright, supported terms are approximately 3 points.
    assert 3.0 + scores[3] > 0
    assert balance_shaping_terms(1.0, 0.0, 1.0) == {
        **balance_shaping_terms(0.0, 1.0, 1.0),
        "balance_right_load_bw": 1.0,
        "balance_left_load_bw": 0.0,
    }


def test_shape_is_mass_invariant_bounded_for_impacts_and_rejects_tiny_balanced_contacts():
    expected = balance_shaping_terms(0.75, 0.25, 1.0)
    assert balance_shaping_terms(7.5, 2.5, 10.0) == pytest.approx(expected)
    high = balance_shaping_terms(1e300, 1e300, 9.81)
    assert high["balance_shaping_reward"] == DEFAULT_SHAPING.bilateral_weight
    assert all(np.isfinite(list(high.values())))
    assert balance_shaping_terms(1e300, 0, 9.81)["balance_shaping_reward"] == -0.15
    assert balance_shaping_terms(0.001, 0.001, 1.0)["balance_shaping_reward"] < 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("field", ["bilateral_weight", "imbalance_weight", "unsupported_weight"])
def test_shaping_weights_must_be_finite_and_nonnegative(field, invalid):
    with pytest.raises(ValueError, match=field):
        replace(DEFAULT_SHAPING, **{field: invalid})


@pytest.mark.parametrize("kwargs", [{"min_foot_load_bw": 0}, {"min_foot_load_bw": 0.51}, {"load_clip_bw": 0.9}])
def test_shaping_load_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        BalanceShaping(**kwargs)


@pytest.mark.parametrize("loads", [(float("nan"), 1, 1), (0, float("inf"), 1), (-1, 0, 1), (0, 1, 0)])
def test_load_inputs_must_be_finite_physical_values(loads):
    with pytest.raises(ValueError):
        balance_shaping_terms(*loads)


@pytest.mark.parametrize("seed", [7, 3011])
def test_arm_a_is_exact_canonical_trajectory_reward_and_reset(seed):
    random = np.random.default_rng(11)
    with CompsognathusEnv(max_episode_steps=80) as canonical, make_balance_env("A", max_episode_steps=80) as study:
        assert study.action_space.shape == (14,)
        assert study.observation_space.shape == (53,)
        expected, reset_info = canonical.reset(seed=seed)
        actual, study_reset_info = study.reset(seed=seed)
        np.testing.assert_array_equal(actual, expected)
        assert study_reset_info == reset_info
        for _ in range(80):
            action = random.uniform(-0.02, 0.02, 14)
            actual_obs, reward, terminated, truncated, info = study.step(action)
            expected_obs, expected_reward, expected_term, expected_trunc, expected_info = canonical.step(action)
            np.testing.assert_array_equal(actual_obs, expected_obs)
            np.testing.assert_array_equal(study.data.qpos, canonical.data.qpos)
            np.testing.assert_array_equal(study.data.qvel, canonical.data.qvel)
            np.testing.assert_array_equal(study.data.ctrl, canonical.data.ctrl)
            assert reward == expected_reward == info["balance_base_reward"] == info["balance_study_reward"]
            assert (terminated, truncated) == (expected_term, expected_trunc)
            assert info["balance_shaping_reward"] == 0.0
            assert all(info[key] == value for key, value in expected_info.items())
            if terminated or truncated:
                break


@pytest.mark.parametrize("arm", ["B", "D"])
def test_filter_uses_existing_rc_law_for_both_physics_and_reward_and_reseeds_on_reset(arm):
    class FilteredCanonical(CompsognathusEnv):
        action_filter_cutoff_hz = 10.0

    with make_balance_env(arm) as study, FilteredCanonical() as canonical:
        study.reset(seed=4)
        canonical.reset(seed=4)
        alpha = low_pass_alpha(10.0, study.dt)
        carried = None
        for action in (np.full(14, 0.1), np.full(14, -0.2), np.full(14, 1.5)):
            clipped = np.clip(action, -1, 1)
            carried = clipped if carried is None else apply_low_pass(carried, clipped, alpha)
            _, expected_reward, _, _, expected_info = canonical.step(action)
            _, reward, _, _, info = study.step(action)
            np.testing.assert_array_equal(study._action_filter_state, carried)
            np.testing.assert_array_equal(study.data.ctrl, study._scale_action(carried))
            np.testing.assert_array_equal(study.data.qpos, canonical.data.qpos)
            assert info["reward_energy"] == expected_info["reward_energy"]
            assert info["balance_base_reward"] == expected_reward
            assert reward == expected_reward + info["balance_shaping_reward"]
        study.reset(seed=4)
        fresh = np.full(14, -0.05)
        study.step(fresh)
        np.testing.assert_array_equal(study._action_filter_state, fresh)


def test_alternating_substep_loads_cannot_fake_bilateral_support():
    with make_balance_env("C", reset_noise_scale=0) as study:
        study.reset(seed=4)
        weight = study.body_mass * abs(study.model.opt.gravity[2])
        count = 0

        def alternate_sensor_loads():
            nonlocal count
            count += 1
            study.data.sensordata[study._foot_sensor_groups[0][0]] = weight if count % 2 else 0.0
            study.data.sensordata[study._foot_sensor_groups[1][0]] = 0.0 if count % 2 else weight

        study._substep_probe_hook = alternate_sensor_loads
        _, _, _, _, info = study.step(np.zeros(14))
        assert count == study.frame_skip
        assert info["balance_right_load_bw"] == info["balance_left_load_bw"] == 0.0
        assert info["reward_balance_bilateral"] == 0.0
        assert info["balance_shaping_reward"] == -DEFAULT_SHAPING.unsupported_weight


@pytest.mark.parametrize("arm", ["A", "C"])
def test_terminal_penalty_is_in_base_and_study_totals_exactly_once(arm):
    with make_balance_env(arm, reset_noise_scale=0) as study, CompsognathusEnv(reset_noise_scale=0) as canonical:
        for env in (study, canonical):
            env.reset(seed=5)
            env.data.qpos[2] = 0.1
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
        _, base_reward, base_terminated, _, base_info = canonical.step(np.zeros(14))
        _, reward, terminated, _, info = study.step(np.zeros(14))
        assert base_terminated and terminated
        assert not info["is_success"]
        component_total = sum(
            value for key, value in base_info.items() if key.startswith("reward_") and key != "reward_total"
        )
        assert base_reward == pytest.approx(component_total + study.fall_penalty)
        assert info["balance_base_reward"] == base_reward
        assert reward == info["balance_study_reward"] == info["reward_total"]
        assert reward == base_reward + info["balance_shaping_reward"]


def test_study_arms_are_explicit_and_filter_cannot_be_silently_misdeclared():
    assert ARMS["A"].shaping == ARMS["B"].shaping
    assert ARMS["C"].shaping == ARMS["D"].shaping
    with pytest.raises(ValueError, match="Unknown"):
        make_balance_env("unknown")
    with pytest.raises(ValueError, match="class declaration"):
        CompsognathusBalanceEnv(balance_arm=ARMS["B"])


def test_calibration_report_separates_returns_from_post_settle_contact_metrics():
    report = home_reference_report(seeds=[4], env_kwargs={"max_episode_steps": 12}, settle_steps=2)[0]
    assert report["steps"] == 12 and report["full_horizon"]
    assert report["post_settle_samples"] == 10
    assert report["base_return"] < report["study_return"]
    assert report["post_settle_means"] is not None
    scenarios = reward_ordering_report()
    assert scenarios[0]["balance_shaping_reward"] > scenarios[3]["balance_shaping_reward"]
    assert scenarios[3]["balance_shaping_reward"] > scenarios[4]["balance_shaping_reward"]


def test_calibration_rejects_a_settling_prefix_that_consumes_the_horizon():
    with pytest.raises(ValueError, match="shorter than"):
        home_reference_report(seeds=[4], env_kwargs={"max_episode_steps": 12}, settle_steps=12)
