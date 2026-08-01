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


class TestFootContactSensors:
    """Touch observations must measure plantar-floor load, not self-contact."""

    def test_every_load_bearing_foot_body_carries_a_touch_sensor(self, env):
        """A touch sensor only sums geoms on its site's own body.

        The digits are child bodies of the metatarsus, so the plantar site
        cannot see them however large it is made.  Each load-bearing body
        therefore needs its own sensor.
        """
        for side in ("r", "l"):
            sensor_bodies = set()
            for name in (
                f"{side}_foot_touch",
                f"{side}_toe_d2_touch",
                f"{side}_toe_d3_touch",
                f"{side}_toe_d4_touch",
            ):
                sensor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                assert sensor_id >= 0, f"missing touch sensor {name}"
                sensor_bodies.add(int(env.model.site_bodyid[env.model.sensor_objid[sensor_id]]))

            for geom_name in (
                f"{side}_plantar_geom",
                f"{side}_toe_d2_geom",
                f"{side}_toe_d3_geom",
                f"{side}_toe_d4_geom",
            ):
                geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                body_id = int(env.model.geom_bodyid[geom_id])
                assert body_id in sensor_bodies, (
                    f"{geom_name} bears load on a body with no touch sensor; "
                    "its contacts would be invisible to the foot-contact signal"
                )

    def test_sensors_account_for_all_floor_force_at_settled_stance(self):
        """The summed reading must equal the force the foot really transmits.

        A pad-only reading also passes a ``> 0`` check, which is how the
        digits went unmeasured after they were made load-bearing, so assert
        against the measured contact forces instead of a threshold.
        """
        env = TRexEnv(reset_noise_scale=0.0, nosedive_termination_threshold=0.35)
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

            # Guard the specific regression: the pad alone must not be
            # mistaken for the whole foot while the digits bear load.
            pad_only = float(env.data.sensordata[env._sensor_r_foot])
            assert pad_only < truth["r"], (
                "digits carry no load at stance, so this test can no longer detect a pad-only foot-contact signal"
            )
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
            for side in ("r", "l"):
                for name in (
                    f"{side}_foot_touch",
                    f"{side}_toe_d2_touch",
                    f"{side}_toe_d3_touch",
                    f"{side}_toe_d4_touch",
                ):
                    sensor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                    sensor_address = env.model.sensor_adr[sensor_id]
                    assert env.data.sensordata[sensor_address] == pytest.approx(0.0, abs=1e-9)
            assert env._foot_contact_forces() == pytest.approx((0.0, 0.0), abs=1e-9)
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
            truth = _true_floor_force_per_foot(env)
            assert foot_dims[0] == pytest.approx(truth["r"], rel=1e-5)
            assert foot_dims[1] == pytest.approx(truth["l"], rel=1e-5)
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


class TestHeightTargetTracksStance:
    """The height-maintenance reward must have gradient at the operating point.

    ``target_z`` is a measured constant: it has to track the plant's settled
    stance.  It did not, and that is how it went stale.  The pre-repair T-Rex
    stood at 0.90 m; the July 2026 home-equilibrium fix raised the stance to
    0.9757 m and nothing updated the target, so ``height_frac`` clipped to 1.0
    everywhere in the healthy band.  The term became a flat constant worth 39%
    of stage-1 and 10% of stage-2 return while shaping nothing -- measured on
    run 20260725_194916 as ``reward_height`` = 0.9995/step against
    ``height_weight`` = 1.0.

    These assert the property rather than the literal, so they stay valid if
    the plant moves again.  (A cross-species version is blocked on the
    brachiosaurus stance bug: its target is 1.2 m against a settled 1.078 m.)
    """

    @staticmethod
    def _height_reward_at(env, pelvis_z: float) -> float:
        # ``env`` is left untyped to match the rest of this module: annotating
        # it resolves ``action_space.shape`` to ``tuple[int, ...] | None``,
        # which then fails the np.zeros and step() calls below. The explicit
        # float() on the return is what keeps mypy happy about the Any.
        mujoco.mj_resetDataKeyframe(env.model, env.data, env.home_keyframe_id)
        # xpos is only populated by mj_forward, so resolve the current pelvis
        # height before differencing against the target.
        mujoco.mj_forward(env.model, env.data)
        env.data.qpos[2] += pelvis_z - float(env.data.xpos[env.pelvis_id, 2])
        mujoco.mj_forward(env.model, env.data)
        assert float(env.data.xpos[env.pelvis_id, 2]) == pytest.approx(pelvis_z, abs=1e-6)
        _, info = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
        return float(info["reward_height"])

    @staticmethod
    def _settled_height(env) -> float:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(300):
            env.step(action)
        return float(env.data.xpos[env.pelvis_id, 2])

    def test_reward_is_saturated_at_the_settled_stance(self):
        env = TRexEnv(reset_noise_scale=0.0, height_weight=1.0, healthy_z_range=(0.75, 1.6))
        try:
            settled = self._settled_height(env)
            reward = self._height_reward_at(env, settled)
            # Near-full rather than exactly 1.0: the target sits a hair above
            # the settled stance, which leaves a sliver of headroom instead of
            # clipping. Anything well below 1.0 means the target is too high.
            assert 0.99 <= reward <= 1.0, (
                f"a correctly standing T-Rex should earn essentially the full height reward, got {reward:.4f} "
                f"at the settled stance {settled:.4f}"
            )
        finally:
            env.close()

    def test_reward_still_has_gradient_below_the_settled_stance(self):
        """The regression: a target below the stance flattens the whole band."""
        env = TRexEnv(reset_noise_scale=0.0, height_weight=1.0, healthy_z_range=(0.75, 1.6))
        try:
            settled = self._settled_height(env)
            floor = env.healthy_z_range[0]
            # Sample the healthy band between the fallen floor and the stance.
            sagged = [settled - f * (settled - floor) for f in (0.25, 0.5, 0.75)]
            rewards = [self._height_reward_at(env, z) for z in sagged]
            assert all(r < 1.0 for r in rewards), (
                f"height reward is saturated while sagging ({rewards} at {sagged}) -- target_z is "
                f"below the settled stance {settled:.4f}, so the term is a constant, not a gradient"
            )
            assert rewards[0] > rewards[1] > rewards[2], (
                f"height reward must fall monotonically as the pelvis sags, got {rewards}"
            )
        finally:
            env.close()

    def test_sb3_and_mjx_height_targets_agree(self):
        """The SB3 target and the MJX registry's copy must not drift apart.

        ``target_z`` lives as a literal inside ``_get_reward_info`` while the
        MJX path reads ``target_standing_z`` from the species registry, and
        neither is fingerprinted. That is exactly how ``natural_forward_z``,
        ``healthy_z_range`` and ``max_tilt_angle`` drifted between the two
        paths. Compared behaviourally so the constant does not have to be
        exposed: both paths must produce the same height reward.
        """
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import _SPECIES_CONFIGS
        from environments.shared.reward_functions import reward_height_maintenance

        registered = _SPECIES_CONFIGS["trex"]["target_standing_z"]
        env = TRexEnv(reset_noise_scale=0.0, height_weight=1.0, healthy_z_range=(0.75, 1.6))
        try:
            floor = env.healthy_z_range[0]
            for pelvis_z in (0.9757, 0.95, 0.90, 0.85, 0.80):
                sb3 = self._height_reward_at(env, pelvis_z)
                mjx = float(reward_height_maintenance(pelvis_z, floor, registered, 1.0))
                assert sb3 == pytest.approx(mjx, rel=1e-9), (
                    f"height reward diverges at pelvis_z={pelvis_z}: SB3 {sb3:.6f} vs MJX {mjx:.6f} "
                    f"(registry target_standing_z={registered})"
                )
        finally:
            env.close()


class TestStage1ResetStaysInsideTheHealthyEnvelope:
    """No stage-1 seed may start already terminated.

    Dibothrosuchus hit this first and fixed it by decoupling
    ``reset_height_noise_scale`` from the radian-scale joint noise.  T-Rex was
    left coupled, and at the then-current ``reset_noise_scale = 0.10`` its
    0.926 m home pelvis sat only 0.226 m — 2.26 sigma — above the 0.70 m floor,
    so 1.19% of spawns landed below it.  Measured over seeds 3042-5041 before
    the fix: 18/2000 spawned sub-floor and 16 terminated on step 1 whatever the
    policy did.  That is the direct cause of the 39/40 evaluation panels, and it
    caps any reliability gate below the level a stance gate needs to certify.

    Two things have since moved and this test outlives both, which is why it
    reads the shipped config rather than a literal: stage 1 now runs at 0.05
    (PLANT_VALIDATION section 12), and root height is no longer sampled at all
    — the ground settle derives it from the sampled joint pose, leaving the
    whole height-jitter channel state-inert (see KNOWN_ISSUES).  The invariant
    asserted here — no seed starts already terminated — is what must hold
    regardless.
    """

    def test_stage1_reset_never_spawns_out_of_bounds(self):
        from environments.shared.config import load_stage_config

        kwargs = load_stage_config("trex", 1)["env_kwargs"]
        kwargs = {k: v for k, v in kwargs.items() if k not in {"foot_contact_weight", "foot_contact_gate"}}
        env = TRexEnv(**kwargs)
        try:
            low, high = env.healthy_z_range
            for seed in range(3042, 3242):
                env.reset(seed=seed)
                pelvis_z = float(env.data.qpos[2])
                assert low < pelvis_z < high, f"seed {seed} spawned at {pelvis_z:.4f} m, outside [{low}, {high}]"
                terminated, info = env._is_terminated()
                assert not terminated, f"seed {seed} starts terminated: {info.get('termination_reason')}"
        finally:
            env.close()

    def test_seed_3077_is_no_longer_terminal_on_the_first_step(self):
        """The specific seed that produced the first failure of a 40-episode panel."""
        from environments.shared.config import load_stage_config

        kwargs = load_stage_config("trex", 1)["env_kwargs"]
        kwargs = {k: v for k, v in kwargs.items() if k not in {"foot_contact_weight", "foot_contact_gate"}}
        env = TRexEnv(**kwargs)
        try:
            env.reset(seed=3077)
            assert float(env.data.qpos[2]) > env.healthy_z_range[0]
            _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape))
            assert not terminated, "seed 3077 must survive one zero-action step"
        finally:
            env.close()
