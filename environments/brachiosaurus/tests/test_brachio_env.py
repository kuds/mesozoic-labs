"""Species-specific tests for the Brachiosaurus Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv


@pytest.fixture
def env():
    e = BrachioEnv()
    yield e
    e.close()


class TestBrachioSpecific:
    """Brachiosaurus-specific observation and info tests."""

    def test_pelvis_height_tracked(self, env):
        """pelvis_height (aliased from torso height) should be present."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "pelvis_height" in info
        assert info["pelvis_height"] > 0

    def test_foot_contacts_tracked(self, env):
        """Foot contact sensors should be reported in info."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "r_foot_contact" in info
        assert "l_foot_contact" in info

    def test_head_food_distance_reported(self, env):
        """Head-food distance should be tracked in info."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "head_food_distance" in info
        assert info["head_food_distance"] > 0  # Food starts far away

    def test_food_not_reached_initially(self, env):
        """Food is far away on first step."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["food_reached"] == 0.0
        assert info["reward_food"] == 0.0

    def test_gait_reward_non_positive(self, env):
        """Gait stability reward is a penalty (non-positive) for angular velocity."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] <= 0.0
        assert info["gait_instability"] >= 0.0
