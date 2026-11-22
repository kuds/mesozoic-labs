"""The registered Allosaurus must balance and walk through its real motors."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import environments  # noqa: F401
from environments.allosaurus_fragilis.scripts.audit_motion import run
from environments.allosaurus_fragilis.scripts.reference_walk import ReferenceWalk
from environments.shared.model_id import ModelId


@pytest.mark.parametrize("walking", [False, True])
def test_reference_action_drives_registered_plant(walking):
    model_id = ModelId("allosaurus_fragilis", "anatomical")
    with gym.make(model_id.gym_id, reset_noise_scale=0.0) as wrapped:
        env = wrapped.unwrapped
        wrapped.reset(seed=0)
        reference = ReferenceWalk(env.model, env.data, walking=walking)
        # Construct a second driver so that the stateful foot placement correction
        # advances exactly once per control interval in each representation.
        native = ReferenceWalk(env.model, env.data, walking=walking)
        for _ in range(300):
            action = reference.action(env.data)
            np.testing.assert_allclose(env._scale_action(action), native.control(env.data), atol=1e-9)
            _, _, terminated, truncated, _ = wrapped.step(action)
            assert not terminated and not truncated
        assert env.data.qpos[2] > 1.8
        if walking:
            assert env.data.qpos[0] > 0.20
        else:
            assert abs(env.data.qpos[0]) < 0.03


def test_reference_walk_uses_alternating_digitigrade_support():
    metrics = run("walk", seconds=10, warmup=3)
    assert metrics["mean_forward_speed_mps"] > 0.25
    assert abs(metrics["lateral_drift_m"]) < 0.12
    assert metrics["no_foot_support_fraction"] == 0
    loading = metrics["floor_impulse_fraction"]
    assert loading["other"] == 0
    for side in ("l", "r"):
        for digit in ("d2", "d3", "d4"):
            assert loading[f"{side}_pad_{digit}"] > 0.035
    assert metrics["total_touchdown_transitions"] == metrics["alternating_transitions"] >= 9
    assert metrics["hallux_site_height_m"][0] > 0.03
    assert metrics["loaded_patch_slip_speed_mps_mean"] < 0.025
    for foot in metrics["feet"].values():
        assert foot["swing_clearance_m_median"] > 0.02
        assert 0.48 < foot["cycle_duty_median"] < 0.90


def test_reference_walk_follows_initial_heading():
    model_id = ModelId("allosaurus_fragilis", "anatomical")
    with gym.make(model_id.gym_id, reset_noise_scale=0.0) as wrapped:
        env = wrapped.unwrapped
        wrapped.reset(seed=0)
        quarter_turn = np.sqrt(0.5)
        env.data.qpos[3:7] = [quarter_turn, 0, 0, quarter_turn]
        mujoco.mj_forward(env.model, env.data)
        start = env.data.qpos[:2].copy()
        reference = ReferenceWalk(env.model, env.data)
        for _ in range(400):
            _, _, terminated, truncated, _ = wrapped.step(reference.action(env.data))
            assert not terminated and not truncated
        sideways, forward = env.data.qpos[:2] - start
        assert abs(sideways) < 0.10
        assert forward > 0.60


def test_reference_stance_loads_both_feet_inside_support_polygon():
    metrics = run("stand", seconds=4, warmup=2)
    assert metrics["min_static_support_margin_m"] > 0.025
    assert metrics["double_support_fraction"] == 1
    assert abs(metrics["travel_x_m"]) < 0.01
    assert all(0.45 < foot["normal_impulse_share"] < 0.55 for foot in metrics["feet"].values())
