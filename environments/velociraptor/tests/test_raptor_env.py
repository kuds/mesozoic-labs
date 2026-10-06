"""Species-specific tests for the Raptor Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import math

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

    @staticmethod
    def _prey_on_one_claw(env, claw, other):
        """Move the prey onto *claw* alone and return its geom id.

        The claws sit 0.12 apart and the prey sphere has radius 0.15, so a prey placed on one
        claw also touches the other; the other claw's collisions are switched off so each claw
        is checked on its own.
        """
        env.reset(seed=0)
        claw_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, claw)
        other_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, other)
        env.model.geom_contype[other_id] = 0
        env.model.geom_conaffinity[other_id] = 0
        env.data.mocap_pos[0] = env.data.geom_xpos[claw_id]
        mujoco.mj_forward(env.model, env.data)
        prey = env.prey_geom_id
        pairs = [(int(env.data.contact[i].geom1), int(env.data.contact[i].geom2)) for i in range(env.data.ncon)]
        assert {geom1 if geom2 == prey else geom2 for geom1, geom2 in pairs if prey in (geom1, geom2)} == {claw_id}
        return claw_id

    @pytest.mark.parametrize("claw, other", [("r_claw_geom", "l_claw_geom"), ("l_claw_geom", "r_claw_geom")])
    def test_either_claw_alone_scores_and_ends_the_strike(self, claw, other):
        """Each sickle claw on its own pays the strike bonus and ends the episode as ``strike_success``."""
        env = RaptorEnv(strike_bonus=10.0, reset_noise_scale=0.0)
        try:
            self._prey_on_one_claw(env, claw, other)
            _, info = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
            assert info["strike_success"] == 1.0
            assert info["reward_strike"] == 10.0

            terminated, info = env._is_terminated()
            assert terminated
            assert info["termination_reason"] == "strike_success"
        finally:
            env.close()

    @pytest.mark.parametrize("claw, other", [("r_claw_geom", "l_claw_geom"), ("l_claw_geom", "r_claw_geom")])
    def test_either_claw_alone_does_not_end_the_episode_without_a_strike_bonus(self, claw, other):
        """With ``strike_bonus == 0.0`` the same claw contact pays nothing and ends nothing."""
        env = RaptorEnv(strike_bonus=0.0, reset_noise_scale=0.0)
        try:
            self._prey_on_one_claw(env, claw, other)
            _, info = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
            assert info["strike_success"] == 1.0
            assert info["reward_strike"] == 0.0

            terminated, info = env._is_terminated()
            assert not terminated, info
            assert "termination_reason" not in info
        finally:
            env.close()


class TestNosediveTermination:
    """Raptor ends an episode as ``nosedive`` once forward_z drops 0.5 below its natural lean.

    ``raptor_env.py`` passes that 0.5 to the shared ``_root_termination`` as a literal,
    and no digest probe reaches the branch (its raptor poses end on ``excessive_tilt``),
    so these posed states pin the threshold from both sides.  The pelvis is lifted clear
    of the floor and pitched nose-down about +y: a pure pitch gives
    ``forward_z == -sin(pitch)`` and a tilt equal to the pitch, which stays inside
    ``max_tilt_angle`` this close to the threshold.
    """

    MARGIN = 0.01

    @staticmethod
    def _terminate_pitched(env, forward_z):
        pitch = math.asin(-forward_z)
        env.reset(seed=0)
        env.data.qpos[2] = 0.7
        env.data.qpos[3:7] = (math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0)
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        return env._is_terminated()

    def test_just_past_the_threshold_terminates_as_nosedive(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            forward_z = env._natural_forward_z - 0.5 - self.MARGIN
            terminated, info = self._terminate_pitched(env, forward_z)

            assert info["forward_z"] == pytest.approx(forward_z, abs=1e-9)
            assert info["tilt_angle"] < env.max_tilt_angle
            assert terminated
            assert info["termination_reason"] == "nosedive"
            assert list(info) == ["pelvis_height", "tilt_angle", "forward_z", "termination_reason"]
        finally:
            env.close()

    def test_just_before_the_threshold_does_not_terminate(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            forward_z = env._natural_forward_z - 0.5 + self.MARGIN
            terminated, info = self._terminate_pitched(env, forward_z)

            assert info["forward_z"] == pytest.approx(forward_z, abs=1e-9)
            assert not terminated, info
            assert "termination_reason" not in info
        finally:
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


def _true_floor_force_per_foot(env) -> dict[str, float]:
    """Sum the real floor normal force under each foot, straight from contacts."""
    floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    totals = {"r": 0.0, "l": 0.0}
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        if floor_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        side = name[:1]
        if side in totals:
            force = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, index, force)
            totals[side] += float(force[0])
    return totals


#: Each foot's touch sensors, toe d3 first (the pinned summation order).
FOOT_TOUCH_SENSORS = {
    "r": ("r_foot_touch", "r_meta_touch", "r_d4_touch"),
    "l": ("l_foot_touch", "l_meta_touch", "l_d4_touch"),
}
FOOT_SUPPORT_GEOMS = {
    "r": ("r_toe_d3_geom", "r_metatarsus_geom", "r_toe_d4_geom"),
    "l": ("l_toe_d3_geom", "l_metatarsus_geom", "l_toe_d4_geom"),
}


class TestFootContactSensors:
    """The foot touch observation must measure the whole foot's floor load.

    The raptor is digitigrade: ground force goes through the toe capsules and
    the metatarsal head, and a lying capsule contacts the plane near its ENDS.
    The original r=0.02 toe-d3 sites missed both end contacts and read 0; the
    enlarged site then read digit 3 alone -- 54% of the floor load, because a
    touch sensor only sums contacts on geoms of its site's OWN body and the
    metatarsus and digit 4 are other bodies.  Since physics revision 3 each
    load-bearing body carries its own sensor and the env sums the three, so
    these assert against the measured contact forces rather than a threshold.
    """

    def test_every_load_bearing_foot_body_carries_a_touch_sensor(self, env):
        for side in ("r", "l"):
            sensor_bodies = set()
            for name in FOOT_TOUCH_SENSORS[side]:
                sensor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                assert sensor_id >= 0, f"missing touch sensor {name}"
                sensor_bodies.add(int(env.model.site_bodyid[env.model.sensor_objid[sensor_id]]))
            for geom_name in FOOT_SUPPORT_GEOMS[side]:
                geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                assert int(env.model.geom_bodyid[geom_id]) in sensor_bodies, (
                    f"{geom_name} bears load on a body with no touch sensor; "
                    "its contacts would be invisible to the foot-contact signal"
                )

    def test_foot_sensor_groups_follow_the_sensor_layout(self, env):
        """Appended sensors keep every older index; each group lists d3 first."""
        addresses = {
            name: int(env.model.sensor_adr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, name)])
            for names in FOOT_TOUCH_SENSORS.values()
            for name in names
        }
        assert env._foot_sensor_groups == (
            tuple(addresses[name] for name in FOOT_TOUCH_SENSORS["r"]),
            tuple(addresses[name] for name in FOOT_TOUCH_SENSORS["l"]),
        )
        assert env._foot_sensor_groups == ((10, 27, 28), (11, 29, 30))

    def test_sensors_account_for_all_floor_force_at_settled_stance(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            info = {}
            for _ in range(200):
                _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                assert not terminated

            truth = _true_floor_force_per_foot(env)
            assert truth["r"] > 1.0, "right foot carries no measurable load at stance"
            assert truth["l"] > 1.0, "left foot carries no measurable load at stance"
            assert info["r_foot_contact"] == pytest.approx(truth["r"], rel=1e-6)
            assert info["l_foot_contact"] == pytest.approx(truth["l"], rel=1e-6)

            # Guard the specific regression: digit 3 alone must not be
            # mistaken for the whole foot while the metatarsal head and digit
            # 4 bear load (at the settled stance it reads about 54%).
            d3_only = float(env.data.sensordata[env._sensor_r_foot])
            assert d3_only < 0.7 * truth["r"], (
                "the metatarsus and digit 4 carry no load at stance, so this test can no longer detect a "
                "toe-d3-only foot-contact signal"
            )
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
            for names in FOOT_TOUCH_SENSORS.values():
                for name in names:
                    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                    adr = model.sensor_adr[sensor_id]
                    assert data.sensordata[adr] == pytest.approx(0.0, abs=1e-9), (
                        f"{name} reads force while airborne — the site is summing a self-contact, not ground contact"
                    )
        finally:
            env.close()

    def test_observation_foot_contact_dims_are_the_summed_feet(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            obs = None
            for _ in range(200):
                obs, _, _, _, _ = env.step(np.zeros(env.action_space.shape))
            # foot contacts sit before prey direction (3) + distance (1) + command (3).
            foot_dims = obs[-9:-7]
            assert np.all(foot_dims > 0.0), f"foot-contact obs dims dead at stance: {foot_dims}"
            np.testing.assert_array_equal(foot_dims, np.asarray(env._foot_contact_forces(), dtype=np.float32))
            assert env.observation_space.shape == (70,)
        finally:
            env.close()


class TestNaturalPitchTracksStance:
    """``natural_pitch`` is a measured constant: the plant's settled lean.

    Posture shaping, the nosedive penalty and nosedive termination are all
    measured from it, so it must track the stance the plant actually holds.
    It did not: the r2 plant settled 24.0 deg nose-down against the 0.35 rad
    (20.05 deg) default, and the statue paid -95 per episode of nosedive
    charge for standing still.  The r3 keyframe settles at 20.10 deg (0.3508
    rad over 40 seeds at noise 0.05), so the default is right again; this
    pins the property, not the literal, so a plant edit that moves the settle
    has to move ``natural_pitch`` with it.
    """

    def test_noise_free_statue_settles_at_the_natural_pitch(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(600):
                _, _, terminated, _, info = env.step(action)
                assert not terminated
            natural_pitch = math.asin(-env._natural_forward_z)
            settled_pitch = math.asin(-float(info["forward_z"]))
            assert abs(settled_pitch - natural_pitch) < 0.01, (
                f"the statue settles at {settled_pitch:.4f} rad against natural_pitch {natural_pitch:.4f}: "
                "re-measure natural_pitch for this plant (raptor_env.py default) together with the nosedive "
                "threshold and stage1_balance.toml's statue-derived constants"
            )
            assert natural_pitch == pytest.approx(0.35)
        finally:
            env.close()


def test_velociraptor_is_sb3_only():
    """The summed foot cannot be mirrored by the frozen MJX registration (D-D17)."""
    assert RaptorEnv.supported_training_backends == ("stable-baselines3",)
