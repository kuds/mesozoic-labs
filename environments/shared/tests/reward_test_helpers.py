"""Reusable reward assertion helpers for species test files.

These functions encode patterns common to all dinosaur environments so
species-specific test modules can call them without duplicating logic.
"""

import numpy as np


def assert_alive_bonus_positive(env, n_steps: int = 5):
    """The alive bonus should always be positive when the agent is alive."""
    env.reset(seed=42)
    for _ in range(n_steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        assert info["reward_alive"] > 0, "alive bonus should be positive"
        if terminated or truncated:
            env.reset()


def assert_energy_penalty_zero_for_zero_action(env):
    """Zero action should produce zero (or near-zero) energy penalty."""
    env.reset(seed=42)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert abs(info["reward_energy"]) < 1e-8, (
        f"energy penalty should be ~0 for zero action, got {info['reward_energy']}"
    )


def assert_energy_penalty_negative_for_nonzero_action(env):
    """Non-zero actions should produce a negative energy penalty."""
    env.reset(seed=42)
    action = np.ones(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["reward_energy"] < 0, f"energy penalty should be negative, got {info['reward_energy']}"


def assert_reward_components_sum_to_total(env, n_steps: int = 10):
    """reward_total should equal the sum of all individual reward_* components."""
    env.reset(seed=42)
    for _ in range(n_steps):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)

        # Collect all reward_* keys except reward_total
        component_sum = sum(v for k, v in info.items() if k.startswith("reward_") and k != "reward_total")
        assert abs(info["reward_total"] - component_sum) < 1e-6, (
            f"reward_total ({info['reward_total']:.6f}) != sum of components ({component_sum:.6f})"
        )
        # Also check that the returned scalar reward matches
        assert abs(reward - info["reward_total"]) < 1e-6

        if terminated or truncated:
            env.reset()


def assert_approach_reward_zero_on_first_step(env):
    """On the very first step the approach delta should be zero."""
    env.reset(seed=42)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert abs(info["reward_approach"]) < 1e-8, (
        f"approach reward should be ~0 on first step, got {info['reward_approach']}"
    )
