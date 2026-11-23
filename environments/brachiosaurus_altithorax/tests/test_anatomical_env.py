"""Species-specific tests for the Brachiosaurus Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import numpy as np
import pytest

from environments.brachiosaurus_altithorax.anatomical.env import BrachiosaurusAltithoraxAnatomicalEnv


@pytest.fixture
def env():
    e = BrachiosaurusAltithoraxAnatomicalEnv()
    yield e
    e.close()


class TestBrachiosaurusAltithoraxSpecific:
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

    def test_gait_reward_non_positive(self, env):
        """Gait stability reward is a penalty (non-positive) for angular velocity."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] <= 0.0
        assert info["gait_instability"] >= 0.0


class TestNoFoodReach:
    """The food reach was removed with the hunt (decision D-D59).

    The food sphere stays in the scene and the observation until the next
    policy-interface revision removes it; nothing pays for, records or ends an
    episode on reaching it, the species declares no success key, and the replay
    camera keeps the locomotion framing it always used on the walk and stance
    configs (the food-reach widening went with its kwargs).
    """

    def test_food_on_the_head_tip_pays_nothing_and_ends_nothing(self, env):
        env.reset(seed=42)
        env.data.mocap_pos[0] = env.data.site_xpos[env.head_tip_site_id].copy()
        _, info = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
        hunt = {
            "food_reached",
            "reward_food",
            "head_food_distance",
            "prey_distance",
            "approach_delta",
            "reward_approach",
            "head_proximity",
            "head_food_proximity_distance",
            "reward_head_proximity",
        }
        assert not hunt & set(info)
        terminated, info = env._is_terminated()
        assert info.get("termination_reason") != "food_reached"
        assert not info.get("success", False)

    def test_no_success_key_no_food_kwarg_and_the_locomotion_camera(self, env):
        assert BrachiosaurusAltithoraxAnatomicalEnv.SUCCESS_KEYS == ()
        with pytest.raises(TypeError, match="food_reach_bonus"):
            BrachiosaurusAltithoraxAnatomicalEnv(food_reach_bonus=10.0)
        assert (env._camera_distance, env._camera_azimuth, env._camera_elevation) == (5.0, 135, -20)


def test_neck_support_reach_and_recovery(env):
    """Hold the head up, lower it to browse, then recover without a large rebound."""
    env.reset_noise_scale = 0.0
    env.reset(seed=0)
    head_tip = env.data.site("head_tip")
    neck_root = env.data.body("neck_1")
    home_height = head_tip.xpos[2] - neck_root.xpos[2]
    pitch_actuators = [env.model.actuator(f"neck_{i}_pitch_act").id for i in range(1, 5)]

    def sample_reach(command, seconds):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[pitch_actuators] = command
        heights = []
        for _ in range(round(seconds / env.dt)):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
            heights.append(head_tip.xpos[2] - neck_root.xpos[2])
        return np.asarray(heights)

    standing = sample_reach(0.0, 2.0)
    assert np.max(np.abs(standing - home_height)) < 0.05
    browsing = sample_reach(0.65, 3.0)
    assert home_height - browsing[-1] > 0.70
    assert browsing.min() > browsing[-1] - 0.15
    returned = sample_reach(0.0, 3.0)
    assert np.max(np.abs(returned[round(2.0 / env.dt) :] - home_height)) < 0.08
