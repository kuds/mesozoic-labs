"""Species-specific tests for the Brachiosaurus Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import mujoco
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


class TestNoFood:
    """The food reach was removed with the hunt (decision D-D59), and the food with it in physics r7.

    The scene holds the animal and the floor alone: no mocap body and no food
    geom, so nothing pays for, records or ends an episode on reaching food,
    the species declares no success key, and the replay camera keeps the
    locomotion framing it always used on the walk and stance configs.  The
    head_tip site and its framepos sensor stay (the posture telemetry reads
    the head height).  The forward reference is the spawn's own heading.
    """

    def test_no_reward_term_or_termination_of_the_hunt(self, env):
        env.reset(seed=42)
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
        for kwarg in ("food_reach_bonus", "food_distance_range", "food_lateral_range", "food_height_range"):
            with pytest.raises(TypeError, match=kwarg):
                BrachiosaurusAltithoraxAnatomicalEnv(**{kwarg: (1.0, 2.0)})
        assert (env._camera_distance, env._camera_azimuth, env._camera_elevation) == (5.0, 135, -20)

    def test_the_scene_has_no_food_and_no_mocap_body(self, env):
        model = env.model
        assert model.nmocap == 0
        for kind, name in ((mujoco.mjtObj.mjOBJ_BODY, "food"), (mujoco.mjtObj.mjOBJ_GEOM, "food_geom")):
            assert mujoco.mj_name2id(model, kind, name) == -1
        for attribute in ("food_geom_id", "_initial_food_dir_2d", "_prev_head_food_distance", "food_distance_range"):
            assert not hasattr(env, attribute), attribute
        # Kept: the head tip and its position sensor.
        assert env.head_tip_site_id == model.site("head_tip").id
        sensor = model.sensor("head_tip_pos")
        assert sensor.type == mujoco.mjtSensor.mjSENS_FRAMEPOS and sensor.objid == env.head_tip_site_id

    def test_the_forward_reference_is_the_spawn_heading(self, env):
        """The progress term reads _reference_heading_2d: velocity along it pays, across it does not."""
        env.reset(seed=3042)
        np.testing.assert_array_equal(env._reference_heading_2d, env._quat_to_forward_2d(env.data.qpos[3:7]))
        env.forward_vel_weight = 1.0
        env._reference_heading_2d = np.array([0.0, 1.0])
        env.data.qvel[0:2] = (0.0, 0.5)
        _, along = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
        env.data.qvel[0:2] = (0.5, 0.0)
        _, across = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
        assert along["reward_forward"] > 0.0
        assert across["reward_forward"] == pytest.approx(0.0, abs=1e-12)


class TestHeadingFreeObservation:
    """quadrupedal-heading-free/v1 since policy interface r12: 81 values, no world heading in any channel.

    2 x 31 joints, the torso's projected gravity, gyro, heading-frame linear
    velocity and accelerometer (12), the four per-leg foot channels and the
    command (3), last.  r11 observed the world quaternion (4), the world
    linear velocity and the food direction and distance (86 values).
    """

    def test_the_layout(self, env):
        assert env.observation_space.shape == (81,)
        obs, _ = env.reset(seed=3042)
        for _ in range(20):
            obs, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert not terminated
        joints = env.model.nq - 7
        assert joints == 31 and env.model.nv - 6 == joints
        np.testing.assert_array_equal(obs[:joints], env.data.qpos[7:].astype(np.float32))
        np.testing.assert_array_equal(obs[joints : 2 * joints], env.data.qvel[6:].astype(np.float32))
        root = 2 * joints
        # Projected gravity: the unit down direction in torso coordinates, near (0, 0, -1) at stance.
        np.testing.assert_allclose(np.linalg.norm(obs[root : root + 3]), 1.0, rtol=1e-6)
        assert obs[root + 2] < -0.99
        gyro = env.data.sensordata[env._sensor_gyro_start : env._sensor_gyro_start + 3]
        accel = env.data.sensordata[env._sensor_accel_start : env._sensor_accel_start + 3]
        np.testing.assert_array_equal(obs[root + 3 : root + 6], gyro.astype(np.float32))
        np.testing.assert_array_equal(obs[root + 9 : root + 12], accel.astype(np.float32))
        np.testing.assert_array_equal(obs[-7:-3], np.asarray(env._foot_contact_forces(), dtype=np.float32))
        assert np.all(obs[-7:-3] > 0.0)
        np.testing.assert_array_equal(obs[-3:], np.zeros(3, dtype=np.float32))

    def test_a_turn_about_vertical_leaves_the_observation_unchanged(self, env):
        """Turning the root (position and orientation) about world vertical changes no observed value."""
        env.reset(seed=3042)
        for _ in range(20):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        env.data.qvel[0:3] = (0.3, -0.2, 0.05)
        mujoco.mj_forward(env.model, env.data)
        before = env._get_obs()
        angle = 0.7
        turn = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]])
        quaternion = np.empty(4)
        mujoco.mju_mulQuat(quaternion, turn, env.data.qpos[3:7].copy())
        env.data.qpos[0:3] = rotation @ env.data.qpos[0:3]
        env.data.qpos[3:7] = quaternion
        env.data.qvel[0:3] = rotation @ env.data.qvel[0:3]
        mujoco.mj_forward(env.model, env.data)
        after = env._get_obs()
        # The forward pass recomputes the site-frame sensors and the contacts on the turned scene;
        # the floor is an infinite plane and the friction cone elliptic, so they agree to solver
        # round-off (measured bit-equal in float32 on MuJoCo 3.10.0, at 0.7, 2.0, -1.3 and pi rad).
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)


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
