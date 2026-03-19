"""Shared reward assertion helpers for species test files.

These helpers encode common reward invariants that must hold across all species.
Species-specific reward tests import the ones relevant to their reward signal.
"""

from __future__ import annotations

import numpy as np


def assert_alive_bonus_positive(env, seed: int = 42) -> None:
    """Alive bonus must be positive on a zero-action step."""
    env.reset(seed=seed)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["reward_alive"] > 0, f"Expected positive alive bonus, got {info['reward_alive']}"


def assert_energy_penalty_structure(env, seed: int = 42) -> None:
    """Energy penalty must be zero for zero action, negative for full action."""
    env.reset(seed=seed)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info_zero = env.step(zero_action)
    assert abs(info_zero["reward_energy"]) < 1e-8, "Energy should be ~0 for zero action"

    env.reset(seed=seed)
    full_action = np.ones(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info_full = env.step(full_action)
    assert info_full["reward_energy"] < 0, "Energy should be negative for full action"


def assert_approach_reward_zero_on_first_step(env, seed: int = 42) -> None:
    """Approach reward must be zero on the first step (no prior distance)."""
    env.reset(seed=seed)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["reward_approach"] == 0.0
    assert info["approach_delta"] == 0.0


def assert_posture_reward_non_positive(env, seed: int = 42) -> None:
    """Posture reward must be non-positive (it's a penalty for tilt)."""
    env.reset(seed=seed)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["reward_posture"] <= 0.0
    assert info["tilt_angle"] >= 0.0


def assert_gait_reward_non_negative(env, seed: int = 42) -> None:
    """Gait reward must be non-negative."""
    env.reset(seed=seed)
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    assert info["reward_gait"] >= 0.0


def assert_smoothness_zero_on_first_step(env, seed: int = 42) -> None:
    """Smoothness penalty must be zero on the first step (no prior action)."""
    env.reset(seed=seed)
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    assert info["reward_smoothness"] == 0.0
    assert info["action_delta"] == 0.0


def assert_smoothness_penalty_for_action_change(env, seed: int = 42) -> None:
    """Smoothness penalty must be negative when action changes between steps."""
    env.reset(seed=seed)
    action1 = np.ones(env.action_space.shape, dtype=np.float32)
    env.step(action1)
    action2 = -np.ones(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action2)
    assert info["reward_smoothness"] < 0.0
    assert info["action_delta"] > 0.0


def assert_zero_weight_zeroes_reward(env, weight_name: str, reward_key: str, seed: int = 42) -> None:
    """Setting a weight to zero must produce zero reward for that component."""
    env.reset(seed=seed)
    original = getattr(env, weight_name)
    env.set_reward_weight(weight_name, 0.0)
    try:
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info[reward_key] == 0.0, f"Expected {reward_key}=0 when {weight_name}=0, got {info[reward_key]}"
    finally:
        env.set_reward_weight(weight_name, original)


def assert_nosedive_penalty_non_positive(env, seed: int = 42) -> None:
    """Nosedive penalty must be non-positive."""
    env.reset(seed=seed)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["reward_nosedive"] <= 0.0


def assert_spin_penalty_non_positive(env, weight: float = 0.5, seed: int = 42) -> None:
    """Spin penalty must be non-positive when weight > 0."""
    env.reset(seed=seed)
    original = env.spin_penalty_weight
    env.set_reward_weight("spin_penalty_weight", weight)
    try:
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_spin"] <= 0.0
        assert info["spin_instability"] >= 0.0
    finally:
        env.set_reward_weight("spin_penalty_weight", original)


def assert_heading_reward_bounded(env, seed: int = 42) -> None:
    """Heading alignment reward must be in [-weight, +weight]."""
    env.reset(seed=seed)
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    assert -1.0 <= info["heading_alignment"] <= 1.0


def assert_backward_vel_penalty_non_positive(env, seed: int = 42) -> None:
    """Backward velocity penalty must be non-positive."""
    env.reset(seed=seed)
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    assert info["reward_backward"] <= 0.0
    assert info["backward_vel"] >= 0.0


def assert_drift_penalty_non_positive(env, seed: int = 42) -> None:
    """Drift penalty must be non-positive."""
    env.reset(seed=seed)
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    assert info["reward_drift"] <= 0.0
    assert info["drift_distance"] >= 0.0
