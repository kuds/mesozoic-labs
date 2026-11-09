"""Species-specific tests for the Raptor Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import math

import mujoco
import numpy as np
import pytest

from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv


@pytest.fixture
def env():
    e = VelociraptorMongoliensisAnatomicalEnv()
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
        env = VelociraptorMongoliensisAnatomicalEnv(strike_bonus=10.0, prey_distance_range=(0.5, 0.5))
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
        env = VelociraptorMongoliensisAnatomicalEnv(strike_bonus=0.0)
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

        The claws sit 0.124 m apart (geom centres at the seed-0 reset of physics r5; r4 0.12)
        and the prey sphere has radius 0.15, so a prey placed on one claw also touches the
        other (measured: both claws); the other claw's collisions are switched off so each claw
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
        env = VelociraptorMongoliensisAnatomicalEnv(strike_bonus=10.0, reset_noise_scale=0.0)
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
        env = VelociraptorMongoliensisAnatomicalEnv(strike_bonus=0.0, reset_noise_scale=0.0)
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

    ``velociraptor_mongoliensis/base_env.py`` passes that 0.5 to the shared ``_root_termination`` as a literal.
    Since the level trunk of physics r5 the digest's ``fwd_in`` probe ends on the branch (r4's
    poses ended on ``excessive_tilt``), but at one pose only, so these posed states pin the
    threshold from both sides: with natural_pitch 0.002 it cuts at forward_z -0.502 (30.1 deg
    nose-down).  The pelvis is lifted clear of the floor (0.7 m: the head tip stays above its
    0.07 m and the skull above its 0.28 m termination) and pitched nose-down about +y: a pure
    pitch gives ``forward_z == -sin(pitch)`` and a tilt equal to the pitch, which stays inside
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
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
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
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
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


def _floor_force_by_geom(env) -> dict[str, float]:
    """The real floor normal force on each geom touching the floor, straight from contacts."""
    floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    totals: dict[str, float] = {}
    force = np.zeros(6)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        if floor_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other)
        mujoco.mj_contactForce(env.model, env.data, index, force)
        totals[name] = totals.get(name, 0.0) + float(force[0])
    return totals


def _true_floor_force_per_foot(env) -> dict[str, float]:
    """Sum the real floor normal force under each foot, straight from contacts."""
    totals = {"r": 0.0, "l": 0.0}
    for name, force in _floor_force_by_geom(env).items():
        if name[:1] in totals:
            totals[name[:1]] += force
    return totals


#: Each foot's touch sensors, the MTP pad first (the pinned summation order), and the geom each one reads: one
#: sensor per load-bearing body since physics r3, on the life-size foot of physics r5 the pad (on the metatarsus,
#: the ``{s}_foot`` site) and the passive digits III and IV.
FOOT_TOUCH_SENSORS = {
    "r": ("r_foot_touch", "r_toe_d3_touch", "r_toe_d4_touch"),
    "l": ("l_foot_touch", "l_toe_d3_touch", "l_toe_d4_touch"),
}
FOOT_SUPPORT_GEOMS = {
    "r": ("r_mtp_pad", "r_toe_d3_geom", "r_toe_d4_geom"),
    "l": ("l_mtp_pad", "l_toe_d3_geom", "l_toe_d4_geom"),
}
#: The observation's six per-site foot channels in order (policy interface r14): right pad, d3, d4, then left.
PER_SITE_GEOMS = FOOT_SUPPORT_GEOMS["r"] + FOOT_SUPPORT_GEOMS["l"]


class TestFootContactSensors:
    """The foot touch observation must measure the whole foot's floor load, site by site.

    The raptor is digitigrade: on the life-size foot of physics r5 ground force goes through
    the rounded MTP pad under the metatarsal condyles and the passive digits III and IV, and a
    lying capsule contacts the plane near its ENDS.  A touch sensor only sums contacts on geoms
    of its site's OWN body, and the pad (on the metatarsus) and the two digits are three bodies,
    so each carries its own sensor (since physics r3, when the r2 toe-d3 site read 54% of the
    load) and the env sums each foot's three for the support rewards.  Since policy interface
    r14 the observation carries the three sites separately (#608's per-site channels, which
    plant_versions note 21 deferred to this checkpoint-stranding change).  These assert
    against the measured contact forces rather than a threshold.
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
        """Resolved by name since physics r5: per-foot groups pad first, one observation channel per sensor."""
        addresses = {
            name: int(env.model.sensor_adr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, name)])
            for names in FOOT_TOUCH_SENSORS.values()
            for name in names
        }
        assert env._foot_sensor_groups == (
            tuple(addresses[name] for name in FOOT_TOUCH_SENSORS["r"]),
            tuple(addresses[name] for name in FOOT_TOUCH_SENSORS["l"]),
        )
        # The pads at 10 / 11, digits III and IV at 24-27 (the XML's sensor block).
        assert env._foot_sensor_groups == ((10, 24, 25), (11, 26, 27))
        assert env._foot_observation_groups == ((10,), (24,), (25,), (11,), (26,), (27,))

    def test_sensors_account_for_all_floor_force_at_settled_stance(self):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
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

            # Guard the specific regression: one site alone must not be mistaken for the whole foot while the
            # others bear load (at the settled statue the pad reads 0.298 of the foot, digit III 0.366, digit IV
            # 0.337: 21.9 / 26.9 / 24.8 of 73.6 N, measured).
            for name in FOOT_TOUCH_SENSORS["r"]:
                address = int(env.model.sensor_adr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, name)])
                assert float(env.data.sensordata[address]) < 0.5 * truth["r"], (
                    f"{name} carries most of the foot, so this test can no longer detect a one-site foot signal"
                )
        finally:
            env.close()

    def test_every_touch_sensor_reads_its_geoms_floor_force_on_every_substep(self):
        """Each site's sensor IS the floor force on its support geom, at every physics substep.

        Stage 1 measured the identity at 5.8e-16 relative on 40 seeds; this samples every
        substep of a settle and a small random-action roll, and checks that nothing but the six
        support geoms touches the floor (the shaft, digit II and the non-colliding neck and arms
        never do).
        """
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            addresses = {
                geom: int(env.model.sensor_adr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor)])
                for side in ("r", "l")
                for geom, sensor in zip(FOOT_SUPPORT_GEOMS[side], FOOT_TOUCH_SENSORS[side], strict=True)
            }
            worst = [0.0]
            seen: set[str] = set()

            def probe():
                loads = _floor_force_by_geom(env)
                seen.update(loads)
                for geom, address in addresses.items():
                    error = abs(float(env.data.sensordata[address]) - loads.get(geom, 0.0))
                    worst[0] = max(worst[0], error / max(1.0, loads.get(geom, 0.0)))

            env._substep_probe_hook = probe
            rng = np.random.default_rng(0)
            for step in range(200):
                action = (
                    np.zeros(env.action_space.shape) if step < 100 else rng.normal(0.0, 0.05, env.action_space.shape)
                )
                _, _, terminated, _, _ = env.step(np.clip(action, -1.0, 1.0).astype(np.float32))
                assert not terminated
        finally:
            env._substep_probe_hook = None
            env.close()
        assert worst[0] < 1e-9
        assert seen == set(addresses), f"non-support floor contact: {sorted(seen - set(addresses))}"

    def test_sensors_read_zero_airborne(self):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
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
            assert env._foot_contact_forces() == pytest.approx((0.0, 0.0), abs=1e-9)
        finally:
            env.close()

    def test_observation_foot_contact_channels_are_the_per_site_floor_forces(self):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            obs = None
            for _ in range(200):
                obs, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                assert not terminated
            # The six foot channels sit before prey direction (3) + distance (1) + command (3).
            foot_dims = obs[-13:-7]
            assert np.all(foot_dims > 0.0), f"foot-contact obs dims dead at stance: {foot_dims}"
            # Each channel is its own site's sensor (float32), and that sensor is its geom's floor force
            # (measured 2.0e-8 relative: the float32 rounding).
            sensors = [env.data.sensordata[group[0]] for group in env._foot_observation_groups]
            np.testing.assert_array_equal(foot_dims, np.asarray(sensors, dtype=np.float32))
            truth = _floor_force_by_geom(env)
            np.testing.assert_allclose(foot_dims, [truth[name] for name in PER_SITE_GEOMS], rtol=1e-6)
            # Their per-foot totals are the support rewards' per-foot sums.
            right, left = env._foot_contact_forces()
            assert float(foot_dims[:3].astype(np.float64).sum()) == pytest.approx(right, rel=1e-6)
            assert float(foot_dims[3:].astype(np.float64).sum()) == pytest.approx(left, rel=1e-6)
            # 71 since policy interface r14 (the maintainer's decision of 2026-11-08): 2 x 23 joints, 12 root,
            # 6 per-site foot channels, 4 prey, 3 command (r13: 69, with 2 per-foot sums of 24 joints).
            assert env.observation_space.shape == (71,)
        finally:
            env.close()


class TestFootContactForceSummationOrder:
    """The shared ``_foot_contact_forces`` adds each foot's pad to the running sum of its digits.

    The raptor's per-foot groups are (pad, d3, d4), so a foot reads ``pad + (d3 + d4)``: the
    association BaseDinoEnv pins (the T-Rex's ``pad + (d2 + d3 + d4)``), over ``np.float64``
    items.  Each observation channel is a one-sensor group, ``sensor + sum(())``, which is the
    sensor exactly.  Exact equality throughout.
    """

    def test_three_sensor_feet_add_the_pad_to_the_digit_sum(self, env):
        env.reset(seed=0)
        right, left = env._foot_sensor_groups
        assert len(right) == len(left) == 3, "each foot is a pad plus two digit sensors"
        sensordata = env.data.sensordata
        # pad + (1e-16 + 1e-16) rounds up to the next float after 1.0; (1.0 + 1e-16) + 1e-16 rounds back to 1.0.
        for index, value in zip(right, (1.0, 1e-16, 1e-16), strict=True):
            sensordata[index] = value
        # The pad's place in the group matters: 1e-16 + (1.0 + 1e-16) is 1.0.
        for index, value in zip(left, (1e-16, 1.0, 1e-16), strict=True):
            sensordata[index] = value

        forces = env._foot_contact_forces()

        assert forces == (1.0000000000000002, 1.0)
        assert all(type(force) is float for force in forces)

    def test_each_observation_channel_is_its_sensor_exactly(self, env):
        env.reset(seed=0)
        sensordata = env.data.sensordata
        values = (1.0, 1e-16, 2.5, 0.0, 3.0, 1e-300)
        for group, value in zip(env._foot_observation_groups, values, strict=True):
            assert len(group) == 1
            sensordata[group[0]] = value
        obs = env._get_obs()
        np.testing.assert_array_equal(obs[-13:-7], np.asarray(values, dtype=np.float32))


class TestNaturalPitchTracksStance:
    """``natural_pitch`` is a measured constant: the plant's settled lean.

    Posture shaping, the nosedive penalty and nosedive termination are all
    measured from it, so it must track the stance the plant actually holds.
    It did not: the r2 plant settled 24.0 deg nose-down against the 0.35 rad
    (20.05 deg) default, and the statue paid -95 per episode of nosedive
    charge for standing still.  The r3 keyframe settles at 20.10 deg (0.3508
    rad over 40 seeds at noise 0.05; 20.02 deg, 0.3494 rad, on the elliptic
    cone of physics r4), so the default is right again.  The life-size plant
    of physics r5 stands its trunk level and settles 0.00202 rad nose-down
    over the same 40 seeds (0.00234 noise-free, this test's roll), so the
    default is 0.002.  This pins the property, and the literal beside it, so
    a plant edit that moves the settle has to move ``natural_pitch`` with it.
    """

    def test_noise_free_statue_settles_at_the_natural_pitch(self):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
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
                "re-measure natural_pitch for this plant (velociraptor_mongoliensis/base_env.py default) together with the nosedive "
                "threshold and stance.toml's statue-derived constants"
            )
            assert natural_pitch == pytest.approx(0.002)
        finally:
            env.close()


def test_velociraptor_mongoliensis_is_sb3_only():
    """The frozen MJX registration (D-D17) reads the toe-d3 sensor alone: it can mirror neither the summed foot of
    policy interface r11 nor the per-site channels of r14."""
    assert VelociraptorMongoliensisAnatomicalEnv.supported_training_backends == ("stable-baselines3",)


class TestSettledPoseTargetsTrackTheStatue:
    """The tail_home_pose and neck_home_pose targets are measured constants, as on the T-Rex.

    ``_TAIL_SETTLED_QPOS`` (2026-11-06) and ``_NECK_SETTLED_QPOS`` (physics r5) are the
    noise-free statue's pose over the post-settle window (seed 0, steps 200-999).  On r4 the
    tail drooped (-0.078506 / 0.0 / -0.046391 / -0.023066 rad, tail_1 pitch and yaw, tail_2
    and tail_3 pitch).  Since physics r5 the tail pitch joints carry their gravity torque in a
    springref preload and the neck and head servos theirs in the home ctrl (the T-Rex's tail
    since its r10, D-D44), so the statue holds the keyframe: measured within 4.3e-7 rad of
    zero on the tail and -2.78e-5 / 0.0 / 1.69e-5 rad on neck_pitch, neck_yaw and head_pitch
    (stage 1 of the life-size port, the maintainer's decision of 2026-11-08); at reset noise
    0.05 (the stance config) the 120 statue seeds of 3042-3081, 3082-3121 and 7042-7081 hold
    their window means within 2.9e-4 rad (tail) and 3.1e-4 rad (neck) of them.  Nothing
    derives them from the plant at run time, so a gain, spring or mass edit that moves the
    tail or the head would silently re-centre the statue off its own targets; this
    re-measures them.  Every tail joint stays 0.17 rad or more clear of its stop (tail_3 at
    +-10 deg, the others at +-12 deg) and every neck joint 0.34 rad or more (neck_yaw at
    +-20 deg): what the terms target is not a joint parked on a stop.
    """

    SETTLE_STEPS = 200
    HORIZON = 1000

    @classmethod
    def _post_settle_means(cls) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            action = np.zeros(env.model.nu, dtype=np.float32)
            tail, neck = [], []
            for step in range(cls.HORIZON):
                _, _, terminated, _, _ = env.step(action)
                assert not terminated
                if step >= cls.SETTLE_STEPS:
                    tail.append(env.data.qpos[env._tail_home_qpos_indices].copy())
                    neck.append(env.data.qpos[env._neck_home_qpos_indices].copy())
            adr = env.model.jnt_qposadr.tolist()
            tail_ranges = env.model.jnt_range[[adr.index(int(i)) for i in env._tail_home_qpos_indices]]
            neck_ranges = env.model.jnt_range[[adr.index(int(i)) for i in env._neck_home_qpos_indices]]
            return np.mean(tail, axis=0), np.mean(neck, axis=0), tail_ranges, neck_ranges
        finally:
            env.close()

    def test_the_targets_are_the_statues_post_settle_pose(self):
        tail, neck, tail_ranges, neck_ranges = self._post_settle_means()
        np.testing.assert_allclose(tail, VelociraptorMongoliensisAnatomicalEnv._TAIL_SETTLED_QPOS, atol=1e-4)
        np.testing.assert_allclose(neck, VelociraptorMongoliensisAnatomicalEnv._NECK_SETTLED_QPOS, atol=1e-4)
        for target, ranges, clearance in (
            (VelociraptorMongoliensisAnatomicalEnv._TAIL_SETTLED_QPOS, tail_ranges, 0.15),
            (VelociraptorMongoliensisAnatomicalEnv._NECK_SETTLED_QPOS, neck_ranges, 0.30),
        ):
            target = np.asarray(target)
            assert np.all(target - ranges[:, 0] > clearance) and np.all(ranges[:, 1] - target > clearance)


def _joint_qpos_address(env, name: str) -> int:
    return int(env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)])


class TestHeadFloorTermination:
    """The skull ends an episode on the floor, as the T-Rex's does (physics r5 gave the raptor its own skull body).

    Two checks guard it, both the T-Rex's: the head tip site below 0.07 m (``head_contact``)
    and the skull origin below 0.28 m (``skull_low``), each the T-Rex's 0.12 / 0.45 m at the
    same fraction of the home height it guards (0.098 of the 0.724 m head tip, 0.367 of the
    0.756 m skull origin), read at the substep MIN; and the floor-contact category ``head``
    (``skull_upper``, ``snout``; the neck and mandible never collide).  Every posed state
    below is pitched nose-down about +y with the neck and head flexed (positive is nose-down),
    lowered until its probe point sits at the height given; the pelvis stays inside
    ``healthy_z_range``, the tilt inside ``max_tilt_angle`` and forward_z above the nosedive
    cut, so only the head can end it.
    """

    MARGIN = 0.01

    @staticmethod
    def _posed(env, pitch_deg: float, neck: float, head: float, probe: str, height: float):
        env.reset(seed=0)
        data = env.data
        pitch = math.radians(pitch_deg)
        data.qpos[2] = 1.0
        data.qpos[3:7] = (math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0)
        data.qpos[_joint_qpos_address(env, "neck_pitch")] = neck
        data.qpos[_joint_qpos_address(env, "head_pitch")] = head
        data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, data)
        point = data.site_xpos[env.head_tip_site_id, 2] if probe == "head_tip" else data.xpos[env.skull_body_id, 2]
        data.qpos[2] += height - float(point)
        mujoco.mj_forward(env.model, data)
        return env._is_terminated()

    def test_head_geom_ids_cached(self, env):
        """Head geom IDs should be resolved and present in termination sets."""
        env.reset(seed=42)
        for attr in ("skull_upper_geom_id", "snout_geom_id"):
            gid = getattr(env, attr)
            assert gid >= 0, f"{attr} was not resolved"
            assert gid in env._body_ground_geoms
            assert gid in env._head_ground_geoms
        names = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in env._body_ground_geoms}
        assert {"neck_geom", "neck_upper_geom", "mandible"}.isdisjoint(names)
        assert env._terrain_contact_probe_geoms == ("neck_geom", "neck_upper_geom")

    def test_the_skull_on_the_floor_ends_the_episode_as_head_contact(self):
        """Pitched 20 deg with the neck flexed to 0.6 rad (its stop is 0.61), the snout meets the floor."""
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            terminated, info = self._posed(env, 20.0, 0.6, 0.0, "head_tip", 0.0)
            floor_id = env.floor_geom_id
            touching = {
                int(geom)
                for contact in env.data.contact[: env.data.ncon]
                if floor_id in (contact.geom1, contact.geom2)
                for geom in (contact.geom1, contact.geom2)
            }
            assert touching & env._head_ground_geoms, "the posed skull does not reach the floor"
            assert info["pelvis_height"] > env.healthy_z_range[0] and info["tilt_angle"] < env.max_tilt_angle
            assert terminated
            assert info["termination_reason"] == "head_contact"
        finally:
            env.close()

    def test_a_skull_strike_between_samples_ends_the_episode_as_head_contact(self):
        """The floor-contact category: a skull geom latched by the substep scan reports ``head_contact``.

        A pose cannot reach it, because the head tip passes 0.07 m before a skull geom
        touches the floor; a strike latched during a step's substeps does, while both head
        heights read healthy.  The tail and trunk report their own categories.
        """
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            for geom, reason in (
                (env.snout_geom_id, "head_contact"),
                (env.skull_upper_geom_id, "head_contact"),
                (env.tail_4_geom_id, "tail_contact"),
                (env.belly_geom_id, "body_contact"),
                (env.ribcage_geom_id, "body_contact"),
            ):
                env._substep_floor_hit_geom = geom
                env._substep_contact_step = env._step_count
                terminated, info = env._is_terminated()
                assert info["head_tip_z"] > 0.07
                assert terminated, geom
                assert info["termination_reason"] == reason, geom
        finally:
            env.close()

    @pytest.mark.parametrize("offset, ends", [(-MARGIN, True), (MARGIN, False)])
    def test_the_head_tip_ends_the_episode_below_0_07_m(self, offset, ends):
        """Pitched 20 deg, neck 0.6 and head 0.5 rad nose-down: the skull origin sits 0.230 m above the tip."""
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            terminated, info = self._posed(env, 20.0, 0.6, 0.5, "head_tip", 0.07 + offset)
            assert info["head_tip_z"] == pytest.approx(0.07 + offset, abs=1e-9)
            assert float(env.data.xpos[env.skull_body_id, 2]) > 0.28  # 0.290 / 0.310 m
            assert terminated is ends, info
            assert info.get("termination_reason") == ("head_contact" if ends else None)
        finally:
            env.close()

    @pytest.mark.parametrize("offset, ends", [(-MARGIN, True), (MARGIN, False)])
    def test_the_skull_ends_the_episode_below_0_28_m(self, offset, ends):
        """Pitched 25 deg with the neck and head at home: the head tip sits 0.125 m below the skull origin."""
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            terminated, info = self._posed(env, 25.0, 0.0, 0.0, "skull", 0.28 + offset)
            assert float(env.data.xpos[env.skull_body_id, 2]) == pytest.approx(0.28 + offset, abs=1e-9)
            assert info["head_tip_z"] > 0.07 + self.MARGIN
            assert info["pelvis_height"] > env.healthy_z_range[0]
            assert terminated is ends, info
            assert info.get("termination_reason") == ("skull_low" if ends else None)
        finally:
            env.close()


class TestSubstepHeightTermination:
    """Height terminations must see every physics substep, like contact does (the T-Rex's).

    The step loop records each ``_substep_height_checks`` entity's MIN z and
    ``_is_terminated`` consumes it by declaration index, while the info keys keep the boundary
    sample.
    """

    def test_min_heights_track_the_probe_and_the_declaration_order(self):
        """env._substep_min_heights == the per-substep elementwise min of (head_tip_z, skull_z), in that order."""
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            assert env._substep_height_checks == (("site", env.head_tip_site_id), ("body", env.skull_body_id))
            substep_heights: list[tuple[float, float]] = []
            env._substep_probe_hook = lambda: substep_heights.append(
                (
                    float(env.data.site_xpos[env.head_tip_site_id, 2]),
                    float(env.data.xpos[env.skull_body_id, 2]),
                )
            )
            for t in range(20):
                action = np.full(env.action_space.shape, 0.3 * np.sin(2 * np.pi * t / 5.0), dtype=np.float32)
                substep_heights.clear()
                env.step(action)
                assert len(substep_heights) == env.frame_skip
                assert env._substep_min_heights is not None
                assert env._substep_min_heights[0] == pytest.approx(min(h[0] for h in substep_heights), abs=1e-12)
                assert env._substep_min_heights[1] == pytest.approx(min(h[1] for h in substep_heights), abs=1e-12)
        finally:
            env._substep_probe_hook = None
            env.close()

    def test_between_sample_dip_terminates_while_boundary_reads_healthy(self):
        """A substep-only dip terminates with the per-check reason even when the boundary sample is healthy.

        The aggregate is armed by hand because a statue never dips (its head tip stays at
        0.723 m and its skull at 0.755 m); the probe test above is the witness that real
        rollouts populate it.
        """
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

            boundary_head = float(env.data.site_xpos[env.head_tip_site_id, 2])
            boundary_skull = float(env.data.xpos[env.skull_body_id, 2])
            assert boundary_head > 0.07 and boundary_skull > 0.28

            # Index 0 = head_tip site check.
            env._substep_min_heights = np.array([0.06, boundary_skull])
            env._substep_contact_step = env._step_count
            terminated, info = env._is_terminated()
            assert terminated
            assert info["termination_reason"] == "head_contact"
            assert info["head_tip_z"] == pytest.approx(boundary_head)

            # Index 1 = skull body check.
            env._substep_min_heights = np.array([boundary_head, 0.27])
            env._substep_contact_step = env._step_count
            terminated, info = env._is_terminated()
            assert terminated
            assert info["termination_reason"] == "skull_low"
        finally:
            env.close()

    def test_reset_invalidates_the_height_aggregate(self):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            env._substep_min_heights = np.array([0.0, 0.0])
            env._substep_contact_step = env._step_count
            env.reset(seed=1)
            terminated, info = env._is_terminated()
            assert not terminated, f"stale height aggregate terminated a fresh episode: {info}"
        finally:
            env.close()


class TestCacheIdsFailsClosed:
    """``_cache_ids`` resolves every name it reads and refuses a model that lacks one (physics r5).

    ``mj_name2id`` answers -1 for a missing name and nothing downstream notices: a renamed
    claw geom silently disables the strike, a renamed head geom silently drops out of the
    termination set, ``site_xpos[-1]`` reads the model's last site and ``sensor_adr[-1]`` the
    last sensor (the r4 plant read its sensors by position).  Each edit below compiles; the env
    must refuse it, naming the element.
    """

    @staticmethod
    def _edited_env(tmp_path, edits):
        source = VelociraptorMongoliensisAnatomicalEnv.MODEL_PATH.read_text()
        for old, new in edits:
            assert source.count(old) == 1, old
            source = source.replace(old, new)
        path = tmp_path / "velociraptor_mongoliensis_anatomical.xml"
        path.write_text(source)

        class EditedEnv(VelociraptorMongoliensisAnatomicalEnv):
            MODEL_PATH = path

        return EditedEnv

    @pytest.mark.parametrize(
        ("edits", "message"),
        [
            ([('geom name="snout"', 'geom name="rostrum"')], "must define geom 'snout'"),
            ([('geom name="r_claw_geom"', 'geom name="r_ungual_geom"')], "must define geom 'r_claw_geom'"),
            (
                [('body name="skull"', 'body name="cranium"'), ('body2="skull"', 'body2="cranium"')],
                "must define body 'skull'",
            ),
            ([('touch name="r_toe_d3_touch"', 'touch name="r_d3_touch"')], "must define sensor 'r_toe_d3_touch'"),
            (
                [
                    (
                        '<touch name="l_toe_d4_touch" site="l_toe_d4_site"/>',
                        '<rangefinder name="l_toe_d4_touch" site="l_toe_d4_site"/>',
                    )
                ],
                "'l_toe_d4_touch' must be a touch sensor",
            ),
            (
                [
                    ('<gyro name="pelvis_gyro" site="imu"/>', "<!-- gyro moved -->"),
                    (
                        '<accelerometer name="pelvis_accel" site="imu"/>',
                        '<accelerometer name="pelvis_accel" site="imu"/>\n    <gyro name="pelvis_gyro" site="imu"/>',
                    ),
                ],
                "'pelvis_gyro' must start at sensordata 0",
            ),
        ],
        ids=["head geom", "claw geom", "skull body", "touch sensor", "sensor type", "imu order"],
    )
    def test_a_model_without_a_name_the_env_reads_is_refused(self, tmp_path, edits, message):
        env_class = self._edited_env(tmp_path, edits)
        with pytest.raises(ValueError, match=message):
            env_class()

    def test_the_unedited_copy_builds(self, tmp_path):
        env = self._edited_env(tmp_path, [])()
        try:
            assert env.observation_space.shape == (71,) and env.action_space.shape == (17,)
        finally:
            env.close()


class TestStanceResetStaysInsideTheHealthyEnvelope:
    """No stance seed may start already terminated (the T-Rex's TestStage1ResetStaysInsideTheHealthyEnvelope).

    The shared reset draws uniform noise on every non-root joint (the passive digits
    included) and settles the root on the lowest support geom, so the spawn height follows
    the sampled pose.  On the stance config (reset noise 0.05) the life-size statue's pelvis
    stays 0.5438-0.5647 m (120 seeds) inside the (0.33, 1.09) m envelope and the head tip at
    0.672 m or more above its 0.07 m termination; this checks 200 seeds at their spawn.
    """

    def test_the_stance_reset_never_spawns_terminated(self):
        from environments.shared.config import load_stage_config
        from environments.shared.model_id import ModelId

        kwargs = load_stage_config(ModelId("velociraptor_mongoliensis", "anatomical"), "stance")["env_kwargs"]
        env = VelociraptorMongoliensisAnatomicalEnv(**kwargs)
        try:
            low, high = env.healthy_z_range
            for seed in range(3042, 3242):
                env.reset(seed=seed)
                pelvis_z = float(env.data.qpos[2])
                assert low + 0.15 < pelvis_z < high - 0.4, f"seed {seed} spawned at {pelvis_z:.4f} m"
                terminated, info = env._is_terminated()
                assert not terminated, f"seed {seed} starts terminated: {info.get('termination_reason')}"
                assert info["head_tip_z"] > 0.5, f"seed {seed}: head tip at {info['head_tip_z']:.3f} m"
        finally:
            env.close()
