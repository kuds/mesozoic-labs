import math

import numpy as np
import pytest

from environments.velociraptor.envs.raptor_env import RaptorEnv


def test_base_env_lifecycle():
    env = RaptorEnv(render_mode="rgb_array")

    # Test reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    # Test step
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)

    # Test render
    frame = env.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)

    # Test close
    env.close()


def test_base_env_scale_action():
    env = RaptorEnv()

    # Test action scaling
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    scaled = env._scale_action(action)
    assert scaled.shape == action.shape

    env.close()


def test_rolling_metrics_absent_before_warmup():
    """Rolling metrics should not appear in info until 10 steps have been taken."""
    env = RaptorEnv()
    env.reset(seed=0)
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    # Only 1 step: rolling metrics not yet emitted
    assert "cost_of_transport" not in info
    assert "gait_symmetry" not in info
    assert "stride_frequency" not in info
    env.close()


def test_rolling_metrics_present_after_warmup():
    """After 10+ steps the rolling metrics should appear in every info dict."""
    env = RaptorEnv()
    env.reset(seed=0)
    # Use a zero action so the raptor stays upright long enough for 10+ steps
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    info = {}
    steps_since_reset = 0
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(action)
        steps_since_reset += 1
        if terminated or truncated:
            env.reset(seed=0)
            steps_since_reset = 0
        # Stop as soon as we've hit the warmup threshold in a single episode
        if steps_since_reset >= 10 and "cost_of_transport" in info:
            break
    assert "cost_of_transport" in info
    assert "gait_symmetry" in info
    assert "stride_frequency" in info
    env.close()


def test_rolling_metrics_valid_types():
    """Rolling metrics should be finite floats (or NaN for CoT when stationary)."""
    env = RaptorEnv()
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    info = {}
    steps_since_reset = 0
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(action)
        steps_since_reset += 1
        if terminated or truncated:
            env.reset(seed=0)
            steps_since_reset = 0
        if steps_since_reset >= 10 and "cost_of_transport" in info:
            break
    # gait_symmetry is always in [0, 1]
    sym = info["gait_symmetry"]
    assert isinstance(sym, float)
    assert 0.0 <= sym <= 1.0
    # stride_frequency is non-negative
    sf = info["stride_frequency"]
    assert isinstance(sf, float)
    assert sf >= 0.0
    # cost_of_transport is either NaN or non-negative
    cot = info["cost_of_transport"]
    assert math.isnan(cot) or cot >= 0.0
    env.close()


def test_rolling_buffers_cleared_on_reset():
    """Rolling buffers should be empty immediately after reset."""
    env = RaptorEnv()
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(15):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert len(env._rolling_fwd_vel) > 0
    env.reset(seed=1)
    assert len(env._rolling_fwd_vel) == 0
    assert len(env._rolling_r_contact) == 0
    assert len(env._rolling_l_contact) == 0
    assert len(env._rolling_energy) == 0
    env.close()
