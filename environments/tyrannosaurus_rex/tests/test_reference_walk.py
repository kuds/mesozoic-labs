"""Mechanical walk regressions use measured floor contacts and the free-root Gym plant."""

import pytest

from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv
from environments.tyrannosaurus_rex.scripts.audit_motion import run


@pytest.mark.parametrize(("seed", "noise"), [(0, 0.0), (4, 0.05)])
def test_reference_walk_alternates_load_and_clears_both_feet(seed, noise):
    report = run(seed=seed, noise=noise, seconds=12.0, warmup=4.0)
    assert not report["terminated"]
    assert 0.55 < report["forward_speed_mps"] < 0.72
    assert abs(report["pelvis_y_drift_m"]) < 0.25
    assert report["alternating_touchdown_fraction"] == 1.0
    assert report["airborne_substep_fraction"] == 0.0
    assert 0.08 < report["double_support_fraction"] < 0.35
    assert report["unregistered_floor_load_fraction"] < 1e-6
    assert 0.15 < report["opposite_foot_width_y_m_median"] < 0.35
    for foot in report["feet"].values():
        assert foot["touchdowns"] >= 6
        assert 0.50 < foot["complete_cycle_duty_median"] < 0.75
        assert 0.02 < foot["swing_peak_ground_clearance_m_median"] < 0.12
        assert foot["loaded_patch_slip_mps_mean"] < 0.07
        # The passive toes carry a measurable impulse through lift-off.
        assert foot["mtp_impulse_share_last_stance_fifth"] < 0.90


def test_audit_restores_settings_and_preserves_existing_probe():
    samples = []

    def probe():
        samples.append(None)

    with TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.02) as env:
        horizon = env.max_episode_steps
        env._substep_probe_hook = probe
        result = run(seed=0, noise=0.0, seconds=0.10, warmup=0.04, mode="stand", env=env, actuators=True)
        motors = result["actuators"]
        assert motors["samples"] == 6 * env.frame_skip
        assert len(motors["minimum_angular_margin_deg"]) == env.model.nu
        assert max(motors["fraction_at_action_limit"].values()) == 0.0
        assert 0 < max(motors["abs_force_share_peak"].values()) <= 1.0
        assert len(samples) == 10 * env.frame_skip
        assert env._substep_probe_hook is probe
        assert env.max_episode_steps == horizon
        assert env.reset_noise_scale == 0.02
