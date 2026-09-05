"""Shared reward assertion helpers for species test files.

These helpers encode common reward invariants that must hold across all species.
Species-specific reward tests import the ones relevant to their reward signal.
"""

from __future__ import annotations

import numpy as np


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


def assert_nosedive_penalty_non_positive(env, seed: int = 42) -> None:
    """Nosedive penalty must be non-positive."""
    env.reset(seed=seed)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["reward_nosedive"] <= 0.0


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
