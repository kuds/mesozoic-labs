"""Species-specific reward tests for Dibothrosuchus.

Common reward invariants (alive bonus, energy penalty, approach zero on first
step, zero forward weight) are tested in
environments/shared/tests/test_species_integration.py::TestRewardConsistency.
"""

import numpy as np
import pytest

from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv


@pytest.fixture
def env():
    e = DibothrosuchusEnv()
    yield e
    e.close()


class TestRewardComponents:
    def test_total_reward_is_sum_of_components(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, terminated, _, info = env.step(action)
        expected = (
            info["reward_forward"]
            + info["reward_backward"]
            + info["reward_drift"]
            + info["reward_alive"]
            + info["reward_energy"]
            + info["reward_gait"]
            + info["reward_tail"]
            + info["reward_snap"]
            + info["reward_approach"]
            + info["reward_snout_proximity"]
            + info["reward_posture"]
            + info["reward_nosedive"]
            + info["reward_height"]
            + info["reward_gait_symmetry"]
            + info["reward_smoothness"]
            + info["reward_heading"]
            + info["reward_lateral"]
            + info["reward_spin"]
            + info["reward_speed"]
            + info["reward_idle"]
        )
        if terminated:
            expected += env.fall_penalty
        assert abs(info["reward_total"] - expected) < 1e-6

    def test_speed_penalty_non_positive(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_speed"] <= 0.0

    def test_tail_instability_non_negative(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["tail_instability"] >= 0.0

    def test_snout_proximity_is_bounded(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert 0.0 <= info["snout_proximity"] <= 1.0


class TestRewardWeightEffects:
    def test_zero_snap_bonus_gives_no_snap_reward(self):
        env = DibothrosuchusEnv(snap_bonus=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_snap"] == 0.0
        env.close()

    def test_zero_gait_stability_weight_zeroes_gait_reward(self):
        env = DibothrosuchusEnv(gait_stability_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] == 0.0
        env.close()

    def test_nonzero_gait_stability_weight_gives_nonpositive_reward(self):
        env = DibothrosuchusEnv(gait_stability_weight=0.5)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] <= 0.0
        env.close()

    def test_high_alive_bonus_dominates(self):
        env = DibothrosuchusEnv(
            alive_bonus=100.0,
            forward_vel_weight=0.0,
            snap_approach_weight=0.0,
            snap_bonus=0.0,
        )
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        if not terminated:
            assert info["reward_alive"] == 100.0
        env.close()

    def test_snout_proximity_weight_scales_its_reward(self):
        weak = DibothrosuchusEnv(snap_snout_proximity_weight=1.0)
        strong = DibothrosuchusEnv(snap_snout_proximity_weight=4.0)
        try:
            action = np.zeros(weak.action_space.shape, dtype=np.float32)
            weak.reset(seed=11)
            strong.reset(seed=11)
            _, _, _, _, weak_info = weak.step(action)
            _, _, _, _, strong_info = strong.step(action)
            assert strong_info["reward_snout_proximity"] == pytest.approx(
                4.0 * weak_info["reward_snout_proximity"], rel=1e-6
            )
        finally:
            weak.close()
            strong.close()

    def test_actuator_count(self):
        """27 actuators: 3 neck/skull + 20 legs (5 per leg) + 4 tail."""
        env = DibothrosuchusEnv()
        assert env.model.nu == 27, f"Expected 27 actuators, got {env.model.nu}"
        assert env.action_space.shape == (27,)
        env.close()


class TestApproachIsMeasuredFromTheSnout:
    """The 0.16 m skull rides a mobile neck, so trunk-referenced approach
    shaping would keep paying out after the snout had already overshot."""

    def test_approach_delta_tracks_snout_distance(self):
        env = DibothrosuchusEnv(snap_approach_weight=1.0)
        try:
            env.reset(seed=5)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            env.step(action)
            previous = env._prev_snout_prey_distance
            _, _, _, _, info = env.step(action)
            assert info["snout_prey_distance"] == pytest.approx(env._prev_snout_prey_distance)
            expected_sign = np.sign(previous - info["snout_prey_distance"])
            assert np.sign(info["approach_delta"]) == expected_sign or info["approach_delta"] == 0.0
        finally:
            env.close()

    def test_snout_distance_differs_from_trunk_distance(self):
        env = DibothrosuchusEnv()
        try:
            env.reset(seed=5)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, _, _, info = env.step(action)
            trunk_distance = float(
                np.linalg.norm(env.data.mocap_pos[0] - env.data.xpos[env.torso_id]),
            )
            assert abs(info["snout_prey_distance"] - trunk_distance) > 0.1
        finally:
            env.close()


class TestCurriculumStageRewards:
    def test_stage1_balance_zeroes_task_rewards(self):
        env = DibothrosuchusEnv(
            forward_vel_weight=0.0,
            snap_bonus=0.0,
            snap_approach_weight=0.0,
            snap_snout_proximity_weight=0.0,
        )
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
        assert info["reward_snap"] == 0.0
        assert info["reward_approach"] == 0.0
        assert info["reward_snout_proximity"] == 0.0
        env.close()

    def test_stage3_snap_has_approach_shaping(self):
        env = DibothrosuchusEnv(snap_approach_weight=10.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "approach_delta" in info
        assert "reward_approach" in info
        assert info["approach_delta"] != 0.0
        env.close()
