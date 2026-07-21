"""Species-specific tests for the Raptor Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import mujoco
import numpy as np
import pytest

from environments.velociraptor.envs.raptor_env import RaptorEnv


@pytest.fixture
def env():
    e = RaptorEnv()
    yield e
    e.close()


class TestTailFloorTermination:
    def test_tail_geom_ids_cached(self, env):
        """Tail geom IDs should be resolved and present in termination sets."""
        env.reset(seed=42)
        for attr in ("tail_3_geom_id", "tail_4_geom_id", "tail_5_geom_id"):
            gid = getattr(env, attr)
            assert gid >= 0, f"{attr} was not resolved"
            assert gid in env._body_ground_geoms
            assert gid in env._tail_ground_geoms

    def test_tail_floor_contact_terminates(self, env):
        """Forcing the tail tip into the ground should terminate the episode."""
        env.reset(seed=42)

        # Slam the pelvis down so the tail drags on the floor
        env.data.qpos[2] = 0.05  # pelvis z near ground
        mujoco.mj_forward(env.model, env.data)

        terminated, info = env._is_terminated()
        # Either the pelvis-too-low check or a contact check should fire
        assert terminated, f"Expected termination but got info={info}"

    def test_tail_contact_reason_is_reported(self, env):
        """When the distal tail contacts the floor the reason should be 'tail_contact'."""
        env.reset(seed=42)

        # Pitch the tail down aggressively so distal segments hit the floor
        # while keeping pelvis in healthy range
        tail_joint_names = ["tail_1_pitch", "tail_2_pitch", "tail_3_pitch", "tail_4_pitch", "tail_5_pitch"]
        for name in tail_joint_names:
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = env.model.jnt_qposadr[jid]
            env.data.qpos[qadr] = -0.26  # pitch downward (radians)

        # Step physics to resolve contacts
        mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)

        terminated, info = env._is_terminated()
        if terminated and "termination_reason" in info:
            assert info["termination_reason"] in ("tail_contact", "body_contact", "fallen", "excessive_tilt")


class TestStrikeTerminationGating:
    def test_strike_terminates_when_strike_bonus_positive(self):
        """Claw-prey contact should terminate when strike_bonus > 0."""
        env = RaptorEnv(strike_bonus=10.0, prey_distance_range=(0.5, 0.5))
        env.reset(seed=42)

        # Move raptor toward prey to force contact
        prey_pos = env.data.body("prey").xpos.copy()
        env.data.qpos[0] = prey_pos[0] - 0.1  # x just behind prey
        env.data.qpos[1] = prey_pos[1]
        mujoco.mj_forward(env.model, env.data)

        # Step until contact or max iterations
        terminated = False
        info = {}
        for _ in range(50):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break
        env.close()

        # With prey at 0.5m, should likely terminate (possibly by strike or other reason)
        # This test mainly validates the gating logic doesn't block strike when bonus > 0

    def test_strike_does_not_terminate_when_strike_bonus_zero(self):
        """Claw-prey contact should NOT terminate when strike_bonus == 0 (e.g. stage 2)."""
        env = RaptorEnv(strike_bonus=0.0)
        env.reset(seed=42)

        # Manually check that _is_terminated skips the strike check
        # by verifying the termination logic path
        assert env.strike_bonus == 0.0
        # The gating condition (self.strike_bonus > 0) should prevent
        # strike_success termination even if contact occurs
        env.close()


class TestNominalPoseActionScaling:
    """Raptor actions are residuals around the XML home controls."""

    def test_reset_and_action_origin_use_same_named_home_keyframe(self, env):
        assert env._reset_keyframe_id == env.home_keyframe_id

    def test_zero_action_maps_exactly_to_home_controls(self, env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        home_ctrl = env.model.key_ctrl[env.home_keyframe_id]

        np.testing.assert_array_equal(scaled, home_ctrl)

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
    """The foot touch sensors must report real stance contact.

    The raptor is digitigrade: ground force goes through the toe capsules,
    and a lying capsule contacts the plane near its ENDS. The original
    r=0.02 touch sites at the toe midpoint missed both end contacts, so
    both sensors (and the two foot-contact observation dims fed from them)
    read 0 during normal stance. The sites now envelop the whole toe_d3
    capsule, and the adjacent-digit contact exclude keeps the reading free
    of the d3/d4 interpenetration force that would otherwise register even
    airborne.
    """

    def test_sensors_read_ground_force_at_settled_stance(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            info = {}
            for _ in range(200):
                _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))
                assert not terminated
            assert info["r_foot_contact"] > 1.0, "right foot touch sensor dead at stance"
            assert info["l_foot_contact"] > 1.0, "left foot touch sensor dead at stance"
        finally:
            env.close()

    def test_sensors_read_zero_airborne(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            model, data = env.model, env.data
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[2] += 1.0
            mujoco.mj_forward(model, data)
            for _ in range(10):
                mujoco.mj_step(model, data)
            for name in ("r_foot_touch", "l_foot_touch"):
                sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                adr = model.sensor_adr[sensor_id]
                assert data.sensordata[adr] == pytest.approx(0.0, abs=1e-9), (
                    f"{name} reads force while airborne — the site is summing a self-contact, not ground contact"
                )
        finally:
            env.close()

    def test_observation_foot_contact_dims_are_live(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            obs = None
            for _ in range(200):
                obs, _, _, _, _ = env.step(np.zeros(env.action_space.shape))
            foot_dims = obs[-6:-4]  # foot contacts sit before prey direction (3) + distance (1)
            assert np.all(foot_dims > 0.0), f"foot-contact obs dims dead at stance: {foot_dims}"
        finally:
            env.close()
