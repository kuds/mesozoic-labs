"""Behavior selection must not reward censoring, duplicated panels or chatter."""

import csv
import json
import tomllib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environments.compsognathus.envs.compsognathus_env import CompsognathusEnv
from environments.compsognathus.experiments.balance_metrics import (
    REFERENCE_STANCE_THRESHOLDS,
    BalanceProbe,
    BalanceTargets,
    evaluate_balance_episode,
    summarize_balance_panel,
)
from environments.shared.curriculum.stance_gate import StanceGateThresholds


def _good_rows(n=40, **overrides):
    return [
        {
            "seed": seed,
            "episode_steps": 1000,
            "horizon_steps": 1000,
            "settle_steps": 200,
            "full_horizon": True,
            "base_reward": 2800.0,
            "reward": 3400.0,
            "unsupported_windows": 0.0,
            "physics_bilateral_20pct_weight": 0.99,
            "pelvis_angular_velocity_rms": 0.02,
            "com_vertical_std_over_home_height": 0.001,
            "sole_pitch_motion_rms_rad": 0.002,
            **overrides,
        }
        for seed in range(3042, 3042 + n)
    ]


def test_reference_thresholds_match_shipped_stance_config():
    config = Path(__file__).resolve().parents[3] / "configs/compsognathus/stance.toml"
    expected = StanceGateThresholds.from_curriculum(tomllib.loads(config.read_text())["curriculum"])
    assert REFERENCE_STANCE_THRESHOLDS == expected


def test_contacts_are_sampled_at_physics_rate_and_existing_hook_is_preserved(tmp_path):
    calls = []
    with CompsognathusEnv(max_episode_steps=5, reset_noise_scale=0) as env:

        def hook():
            calls.append(float(env.data.time))

        env._substep_probe_hook = hook
        row = evaluate_balance_episode(env, lambda obs: (np.zeros(14), None), 4, tmp_path / "trace.csv", settle_steps=2)
        assert env._substep_probe_hook is hook
        assert len(calls) == 50
        assert len(set(calls)) == 50
        assert row["physics_samples"] == 30
        assert row["max_sensor_error_N"] < 1e-8
        assert row["full_horizon"]
        assert env._renderer is None
    records = list(csv.DictReader((tmp_path / "trace.csv").open()))
    assert len(records) == 5
    assert "raw_action.r_ankle_act" in records[0]
    assert "applied_action.r_ankle_act" in records[0]
    assert "force.r_ankle_act" in records[0]
    assert "qpos.r_ankle" in records[0]
    json.dumps(row, allow_nan=False)


def test_hook_is_restored_when_prediction_raises():
    with CompsognathusEnv() as env:

        def original():
            return None

        env._substep_probe_hook = original

        def fail(obs):
            raise RuntimeError("inference failed")

        with pytest.raises(RuntimeError, match="inference failed"):
            evaluate_balance_episode(env, fail, 1)
        assert env._substep_probe_hook is original


def test_simultaneous_support_differs_from_conservative_control_window(monkeypatch):
    """Alternating feet never leave ground together; their separate minima do."""
    loads = iter(([9.81, 0.0], [0.0, 9.81], [4.905, 4.905], [0.0, 0.0]))
    with CompsognathusEnv(reset_noise_scale=0) as env:
        env.reset(seed=1)
        probe = BalanceProbe(env.model, env.data, env.target_standing_z)

        # Feed synthetic floor/foot contacts through the actual parent sampler.
        # Kinematics and model properties still come from the real MuJoCo model.
        class SyntheticContact:
            geom1 = env.model.geom("floor").id

            def __init__(self, geom):
                self.geom2 = geom
                self.dist = 0.0

        class SyntheticData:
            ncon = 2
            contact = [SyntheticContact(next(iter(ids))) for ids in probe.groups]
            sensordata = env.data.sensordata.copy()

            def __getattr__(self, name):
                return getattr(env.data, name)

        synthetic = SyntheticData()
        probe.data = synthetic
        current = [0.0, 0.0]

        def contact_force(m, d, i, out):
            out[0] = current[i]

        monkeypatch.setattr(mujoco, "mj_contactForce", contact_force)
        for _ in range(4):
            current[:] = next(loads)
            for side, sensors in enumerate(probe.sensors):
                synthetic.sensordata[sensors] = current[side] / len(sensors)
            probe.sample()
        result = probe.results()
    assert result["physics_samples"] == 4
    assert result["physics_unsupported"] == 0.25
    assert result["physics_bilateral_20pct_weight"] == 0.25
    assert result["weaker_foot_load_bw_mean"] == pytest.approx(0.125)
    assert result["physics_load_imbalance_mean"] == pytest.approx(0.75)
    assert result["max_sensor_error_N"] == 0
    # For the first two samples, per-foot minima would falsely imply a fully
    # unsupported window, while the instantaneous unsupported fraction is zero.
    assert np.all(np.min([[9.81, 0], [0, 9.81]], axis=0) <= 0.1)


def test_no_settling_tail_reports_missing_measurements():
    with CompsognathusEnv(max_episode_steps=2) as env:
        row = evaluate_balance_episode(env, lambda obs: np.zeros(14), 1)
    assert row["physics_samples"] == 0
    assert row["unsupported_windows"] is None
    assert row["physics_bilateral_20pct_weight"] is None
    assert row["sole_pitch_motion_rms_rad"] is None
    assert row["joint_hard_stop_duty"] is None
    assert not summarize_balance_panel([row])["behavior_qualified"]


def test_failed_episodes_are_censored_from_duty_but_cannot_win_selection():
    good = summarize_balance_panel(_good_rows())
    failures = summarize_balance_panel(
        _good_rows(
            episode_steps=10,
            full_horizon=False,
            unsupported_windows=0,
            physics_bilateral_20pct_weight=1,
            pelvis_angular_velocity_rms=0,
            reward=1e6,
        )
    )
    assert failures["projected_stance_panel"]["n_duty_episodes"] == 0
    assert failures["n_behavior_episodes"] == 0
    assert failures["behavior_means"]["physics_bilateral_20pct_weight"] is None
    assert not failures["behavior_qualified"]
    assert good["selection_key"] > failures["selection_key"]
    json.dumps(failures, allow_nan=False)


def test_last_step_terminal_failure_is_not_full_horizon():
    rows = _good_rows(full_horizon=False)
    summary = summarize_balance_panel(rows)
    assert summary["projected_stance_panel"]["full_horizon_fraction"] == 0
    assert summary["projected_stance_panel"]["n_duty_episodes"] == 0


def test_study_shaping_cannot_pay_the_base_reward_rail():
    summary = summarize_balance_panel(_good_rows(base_reward=1700, reward=10000))
    assert not summary["projected_stance_gate_passed"]
    assert any("1800" in failure for failure in summary["projected_stance_gate_failures"])


def test_behavior_selection_prefers_quiet_bilateral_policy_to_high_reward_chatter():
    quiet = summarize_balance_panel(_good_rows())
    chatter = summarize_balance_panel(_good_rows(physics_bilateral_20pct_weight=0.08, reward=1e6))
    assert quiet["behavior_qualified"]
    assert chatter["projected_stance_gate_passed"]
    assert not chatter["behavior_qualified"]
    assert quiet["selection_key"] > chatter["selection_key"]
    assert not quiet["advances_curriculum"]
    assert quiet["research_only"]


def test_missing_measurements_and_smoke_or_duplicate_panels_never_qualify():
    for rows in (_good_rows(8), _good_rows(40, seed=1), _good_rows(sole_pitch_motion_rms_rad=None)):
        assert not summarize_balance_panel(rows)["behavior_qualified"]
    with pytest.raises(ValueError, match="at least 40"):
        BalanceTargets(min_eval_episodes=8)


def test_mismatched_settling_window_cannot_project_canonical_gate():
    assert not summarize_balance_panel(_good_rows(settle_steps=0))["projected_stance_gate_passed"]
