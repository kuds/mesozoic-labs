"""Species-specific tests for the T-Rex Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import mujoco
import numpy as np
import pytest

from environments.trex.envs.trex_env import TRexEnv


@pytest.fixture
def env():
    e = TRexEnv()
    yield e
    e.close()


class TestHeadFloorTermination:
    def test_head_geom_ids_cached(self, env):
        """Head geom IDs should be resolved and present in termination sets."""
        env.reset(seed=42)
        for attr in ("skull_upper_geom_id", "snout_geom_id"):
            gid = getattr(env, attr)
            assert gid >= 0, f"{attr} was not resolved"
            assert gid in env._body_ground_geoms
            assert gid in env._head_ground_geoms

    def test_head_floor_contact_terminates(self, env):
        """Forcing the skull into the ground should terminate the episode."""
        env.reset(seed=42)

        # Pitch the T-Rex forward so the skull hits the floor
        env.data.qpos[2] = 0.3  # lower pelvis
        # Pitch pelvis forward aggressively via quaternion (w, x, y, z)
        env.data.qpos[3] = 0.7071  # w
        env.data.qpos[4] = 0.0  # x
        env.data.qpos[5] = 0.7071  # y (90-deg pitch forward)
        env.data.qpos[6] = 0.0  # z
        mujoco.mj_forward(env.model, env.data)

        terminated, info = env._is_terminated()
        assert terminated, f"Expected termination but got info={info}"

    def test_head_contact_reason_is_reported(self, env):
        """When the skull contacts the floor the reason should be 'head_contact'."""
        env.reset(seed=42)

        # Pitch forward so skull slams into the ground while pelvis stays in healthy range
        neck_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch")
        head_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch")
        neck_qadr = env.model.jnt_qposadr[neck_jid]
        head_qadr = env.model.jnt_qposadr[head_jid]
        env.data.qpos[neck_qadr] = -0.52  # pitch neck downward (radians)
        env.data.qpos[head_qadr] = -0.44  # pitch head downward

        mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)

        terminated, info = env._is_terminated()
        if terminated and "termination_reason" in info:
            assert info["termination_reason"] in (
                "head_contact",
                "torso_contact",
                "tail_contact",
                "fallen",
                "excessive_tilt",
                "nosedive",
            )


class TestTRexSpecific:
    """T-Rex-specific reward component tests."""

    def test_bite_not_triggered_initially(self, env):
        """Prey is far away on first step."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["bite_success"] == 0.0
        assert info["reward_bite"] == 0.0

    def test_height_reward_non_negative(self, env):
        """Height reward should be non-negative (bonus for staying upright)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_height"] >= 0.0

    def test_nosedive_penalty_non_positive(self, env):
        """Nosedive penalty should be non-positive."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_nosedive"] <= 0.0


class TestNominalPoseActionScaling:
    """T-Rex actions are residuals around the complete XML home controls."""

    def test_reset_and_action_origin_use_same_named_home_keyframe(self, env):
        assert env._reset_keyframe_id == env.home_keyframe_id

    def test_zero_action_maps_exactly_to_home_controls(self, env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        np.testing.assert_array_equal(
            env._scale_action(action),
            env.model.key_ctrl[env.home_keyframe_id],
        )

    def test_action_endpoints_retain_full_control_range(self, env):
        ctrl_range = env.model.actuator_ctrlrange
        np.testing.assert_allclose(
            env._scale_action(-np.ones(env.action_space.shape, dtype=np.float32)),
            ctrl_range[:, 0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            env._scale_action(np.ones(env.action_space.shape, dtype=np.float32)),
            ctrl_range[:, 1],
            atol=1e-12,
        )

    def test_piecewise_mapping_interpolates_on_each_side_of_home(self, env):
        ctrl_range = env.model.actuator_ctrlrange
        home_ctrl = env.model.key_ctrl[env.home_keyframe_id]

        below = env._scale_action(np.full(env.action_space.shape, -0.5, dtype=np.float32))
        above = env._scale_action(np.full(env.action_space.shape, 0.5, dtype=np.float32))

        np.testing.assert_allclose(below, (ctrl_range[:, 0] + home_ctrl) / 2.0, atol=1e-12)
        np.testing.assert_allclose(above, (home_ctrl + ctrl_range[:, 1]) / 2.0, atol=1e-12)

    def test_zero_residual_keeps_energy_and_smoothness_penalties_zero(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        _, _, terminated, truncated, first_info = env.step(action)
        assert not terminated
        assert not truncated
        _, _, _, _, second_info = env.step(action)

        assert first_info["reward_energy"] == pytest.approx(0.0, abs=1e-12)
        assert second_info["reward_energy"] == pytest.approx(0.0, abs=1e-12)
        assert second_info["reward_smoothness"] == pytest.approx(0.0, abs=1e-12)
        assert second_info["action_delta"] == pytest.approx(0.0, abs=1e-12)


class TestFootContactSensors:
    """Touch observations must measure plantar-floor load, not self-contact."""

    def test_sites_share_the_load_bearing_plantar_bodies(self, env):
        for side in ("r", "l"):
            site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
            geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_plantar_geom")
            assert env.model.site_bodyid[site_id] == env.model.geom_bodyid[geom_id]

    def test_sensors_read_ground_force_at_settled_stance(self):
        env = TRexEnv(reset_noise_scale=0.0, nosedive_termination_threshold=0.35)
        try:
            env.reset(seed=0)
            info = {}
            for _ in range(200):
                _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                assert not terminated
            assert info["r_foot_contact"] > 1.0, "right foot touch sensor dead at stance"
            assert info["l_foot_contact"] > 1.0, "left foot touch sensor dead at stance"
        finally:
            env.close()

    def test_sensors_read_zero_airborne(self):
        env = TRexEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            mujoco.mj_resetDataKeyframe(env.model, env.data, env.home_keyframe_id)
            env.data.qpos[2] += 1.0
            mujoco.mj_forward(env.model, env.data)
            for _ in range(10):
                mujoco.mj_step(env.model, env.data)
            for name in ("r_foot_touch", "l_foot_touch"):
                sensor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                sensor_address = env.model.sensor_adr[sensor_id]
                assert env.data.sensordata[sensor_address] == pytest.approx(0.0, abs=1e-9)
        finally:
            env.close()

    def test_observation_foot_contact_dims_are_live(self):
        env = TRexEnv(reset_noise_scale=0.0, nosedive_termination_threshold=0.35)
        try:
            env.reset(seed=0)
            obs = None
            for _ in range(200):
                obs, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                assert not terminated
            foot_dims = obs[-6:-4]
            assert np.all(foot_dims > 0.0), f"foot-contact obs dims dead at stance: {foot_dims}"
        finally:
            env.close()

    def test_adjacent_digits_do_not_self_contact_at_home(self, env):
        env.reset(seed=0)
        adjacent_pairs = {
            frozenset(
                (
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_toe_d2_geom"),
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_toe_d3_geom"),
                )
            )
            for side in ("r", "l")
        } | {
            frozenset(
                (
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_toe_d3_geom"),
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_toe_d4_geom"),
                )
            )
            for side in ("r", "l")
        }
        actual_pairs = {
            frozenset((env.data.contact[index].geom1, env.data.contact[index].geom2)) for index in range(env.data.ncon)
        }
        assert adjacent_pairs.isdisjoint(actual_pairs)
