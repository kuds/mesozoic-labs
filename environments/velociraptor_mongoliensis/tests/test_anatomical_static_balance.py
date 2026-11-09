"""Tests that the raptor model is physically set up for static balance.

Validates the home keyframe: COM projection, support polygon, joint limits,
neutral-action stability, and actuator-disabled passive behavior. These catch
model regressions (e.g. mass changes that shift COM behind the feet, or
keyframe edits that violate joint limits) before any RL training is attempted.

Since physics r5 (the life-size plant of IGM 100/986, the maintainer's decision
of 2026-11-08) the home keyframe stands the trunk level, the sacrum horizontal,
and the statue settles 0.002 rad nose-down from it (natural_pitch); the r3 / r4
plant leaned 20 deg nose-down.  The tilt tests measure from the settled pose.

``TestHomeKeyframeStance`` pins the stance the r3 plant introduced (plant_versions
note 14) on the r5 foot (note 22): the MTP pad and the passive digits III and IV
flat at the authored 0.5 mm contact, the CoM over the foot, the servoed leg
springs anchored at the pose, the passive digits and the tail carried by their
springref preloads and the servos by a gravity preload in the home ctrl.  The r2
plant passed every other test in this file while standing on toes 44.6 mm inside
the floor, propped up by springs pulling toward qpos 0.
"""

import math

import mujoco
import numpy as np
import pytest

from environments.shared.tests.static_balance_helpers import (
    ActuatorDisabledPassiveBase,
    HomePoseCOMBase,
    JointLimitsAtHomeBase,
    MassDistributionBase,
    NeutralActionStabilityBase,
)
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

#: The geoms the floor carries the animal on since physics r5: each foot's MTP pad and passive digits III and IV
#: (FOOT_SUPPORT in anatomical/env.py).  The metatarsus shaft and digit II never touch.
SUPPORT_GEOM_NAMES = {side: (f"{side}_mtp_pad", f"{side}_toe_d3_geom", f"{side}_toe_d4_geom") for side in ("r", "l")}
FOOT_GEOM_NAMES = [name for names in SUPPORT_GEOM_NAMES.values() for name in names]
ROOT_BODY = "pelvis"


@pytest.fixture
def env():
    e = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
    e.reset(seed=0)
    yield e
    e.close()


class TestHomePoseCOM(HomePoseCOMBase):
    foot_geom_names = FOOT_GEOM_NAMES
    root_body = ROOT_BODY
    ankle_body_names = ("r_metatarsus", "l_metatarsus")
    max_ankle_offset = 0.10
    max_support_distance = 0.15
    species_label = "Velociraptor"


class TestNeutralActionStability(NeutralActionStabilityBase):
    species_name = "Velociraptor"
    root_body_id_attr = "pelvis_id"
    max_height_drop = 0.10
    max_tilt_increase = 0.53  # 30 degrees

    def test_survives_full_noise_free_episode(self, env):
        """The home-centered zero residual must remain viable for 1,000 Gym steps."""
        env.reset(seed=0)
        neutral_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for step in range(1, env.max_episode_steps + 1):
            _, _, terminated, truncated, info = env.step(neutral_action)
            assert not terminated, (
                f"Velociraptor terminated at step {step}/{env.max_episode_steps} "
                f"under the XML home command: {info.get('termination_reason', 'unknown')}"
            )
            assert truncated is (step == env.max_episode_steps)


class TestLateralSwayStiffness:
    """The hip-roll servos must out-stiffen the body's lateral gravity moment (plant_versions note 13, the T-Rex's).

    With both feet planted the two hip-roll servos and their joint springs resist lateral
    sway in parallel, so the stance is stable only while ``2 x (kp + k)`` exceeds ``m.g.h``,
    the animal's weight times its CoM height, measured on the settled statue: 76.07 N.m/rad
    on physics r5 (15.0 kg, subtree CoM 0.5169 m above the floor after 600 noise-free steps).
    kp 120 and k 24 give 2 x 144 = 288 (3.79x).  The T-Rex's r7 kept kp 150 against its 704
    and stood rolled onto its pad edges (0.54x); its bar, 1.5x, keeps a later re-sizing to real
    margin rather than just clear of the instability.
    """

    def test_hip_roll_servos_out_stiffen_the_lateral_gravity_moment(self, env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(600):
            env.step(action)
        model = env.model
        mass = float(model.body_subtreemass[env.pelvis_id])
        com_height = env._clearance(env.data.subtree_com[env.pelvis_id])
        gravity_stiffness = mass * -float(model.opt.gravity[2]) * com_height
        assert gravity_stiffness == pytest.approx(76.07, rel=0.01)

        servo_stiffness = []
        for side in ("r", "l"):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_hip_roll_act")
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_roll")
            servo_stiffness.append(float(model.actuator_gainprm[actuator_id, 0] + model.jnt_stiffness[joint_id]))
        assert servo_stiffness[0] == servo_stiffness[1], "hip-roll servos must be symmetric"
        assert sum(servo_stiffness) > 1.5 * gravity_stiffness, (
            f"2 x (kp + k) = {sum(servo_stiffness):.0f} N.m/rad against m.g.h = {gravity_stiffness:.1f}: "
            f"{sum(servo_stiffness) / gravity_stiffness:.2f}x, under the 1.5x lateral-sway margin"
        )


class TestActuatorDisabledPassive(ActuatorDisabledPassiveBase):
    species_name = "Velociraptor"
    root_body_id_attr = "pelvis_id"
    max_height_drop = 0.08
    max_tilt_increase = 0.30


class TestJointLimitsAtHome(JointLimitsAtHomeBase):
    knee_names = ["r_knee", "l_knee"]
    knee_margin_deg = 20.0


#: The eight servoed leg joints and their actuators (since physics r5 the digits are passive, as on the T-Rex).
LEG_JOINTS = [f"{side}_{suffix}" for side in ("r", "l") for suffix in ("hip_pitch", "hip_roll", "knee", "ankle")]
LEG_ACTUATORS = [f"{name}_act" for name in LEG_JOINTS]
#: The neck and head servos, gravity-preloaded in the home ctrl like the legs.
NECK_ACTUATORS = ["neck_pitch_act", "neck_yaw_act", "head_pitch_act"]
TAIL_ACTUATORS = ["tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"]
#: The passive joints whose springref carries a static load as a preload: the tail pitch joints (the T-Rex's SR3,
#: D-D44) and each foot's digits II (held raised), III and IV (pressed onto the floor).
PRELOADED_JOINTS = [
    "tail_1_pitch",
    "tail_2_pitch",
    "tail_3_pitch",
    *(f"{side}_toe_{digit}_joint" for side in ("r", "l") for digit in ("d2", "d3", "d4")),
]


def _floor_force_by_geom(model, data) -> dict[str, float]:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    totals: dict[str, float] = {}
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if floor_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        mujoco.mj_contactForce(model, data, index, force)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other)
        totals[name] = totals.get(name, 0.0) + float(force[0])
    return totals


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2-D points (monotone chain)."""
    ordered = sorted(map(tuple, points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def _geom_floor_distance(env, name: str) -> float:
    geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return float(mujoco.mj_geomDistance(env.model, env.data, geom_id, env.floor_geom_id, 1.0, None))


def _settle(env, steps: int) -> None:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(steps):
        _, _, terminated, _, info = env.step(action)
        assert not terminated, info.get("termination_reason")


class TestHomeKeyframeStance:
    """The home keyframe stands on its flat feet, on its servos and its passive preloads."""

    @staticmethod
    def _at_home(env):
        mujoco.mj_resetDataKeyframe(env.model, env.data, env.home_keyframe_id)
        mujoco.mj_forward(env.model, env.data)

    def test_leg_springs_reference_home(self, env):
        """The servoed leg joints' springs must not fight the intended stance.

        MuJoCo's default springref is 0, not ref: before r3 the springs pulled
        the settled hip, knee and ankle toward qpos 0 with 11.9 / 15.8 / 28.0
        N.m, which the servos then spent most of their torque cancelling.  The
        tail's yaw joint is neutral at home too; its pitch joints and the
        digits carry a preload instead (the next test).
        """
        home = env.model.key_qpos[env.home_keyframe_id]
        spring_torque = 0.0
        for name in [*LEG_JOINTS, "tail_1_yaw"]:
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            address = env.model.jnt_qposadr[joint_id]
            assert env.model.qpos_spring[address] == pytest.approx(home[address], abs=1e-6), name
            spring_torque += abs(env.model.jnt_stiffness[joint_id] * (home[address] - env.model.qpos_spring[address]))
        assert spring_torque < 1e-4

    def test_passive_preloads_carry_the_static_load(self, env):
        """Each preloaded passive joint's spring torque at the keyframe is the load it carries at the statue.

        ``k (springref - key)`` against the joint-space static load at the settled noise-free
        statue (``qfrc_bias - qfrc_constraint - qfrc_actuator``: gravity and contact on the
        joint's subtree, 600 steps).  Measured on physics r5: the tail pitch joints 4.743 /
        2.984 / 1.875 N.m to 1e-6 (springref 45.3 / 28.5 / 17.9 deg above a level tail; the
        tail servos then hold about 0 N.m), digits III and IV 2.271 / 1.763 N.m against
        2.266 / 1.765 (0.23 % / 0.11 %; a 1.08 / 0.84 deg preload pressing each digit onto the
        floor with the share of the load the statue gives it) and digit II -0.0081 N.m to
        0.06 % (its own weight).  The preload, not a penetration or the servos, carries the load,
        so the statue settles on the keyframe.
        """
        home = env.model.key_qpos[env.home_keyframe_id]
        _settle(env, 600)
        for name in PRELOADED_JOINTS:
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            address, dof = env.model.jnt_qposadr[joint_id], env.model.jnt_dofadr[joint_id]
            preload = float(env.model.jnt_stiffness[joint_id] * (env.model.qpos_spring[address] - home[address]))
            load = float(env.data.qfrc_bias[dof] - env.data.qfrc_constraint[dof] - env.data.qfrc_actuator[dof])
            assert preload == pytest.approx(load, rel=0.01), f"{name}: preload {preload:.4f} N.m, load {load:.4f}"
            assert abs(math.degrees(env.data.qpos[address] - home[address])) < 0.01, name
            if "toe_d3" in name or "toe_d4" in name:
                assert 0.0 < math.degrees(env.model.qpos_spring[address] - home[address]) < 2.0, name
        for name in TAIL_ACTUATORS:
            actuator_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert (
                abs(float(env.data.actuator_force[actuator_id])) < 1e-4 * env.model.actuator_forcerange[actuator_id, 1]
            )

    def test_keyframe_feet_lie_flat_at_the_authored_contact(self, env):
        """Every support geom touches at 0.5 mm, the digits level; nothing else reaches the floor.

        The r2 keyframe tipped the toes 20 deg toe-down, 44.6 mm into the floor, and every
        reset settles to the keyframe's own clearance.  Measured at the r5 home: the pads and
        digits III / IV at -0.500 mm, the digits' axes level to 2e-5 deg, the metatarsus shaft
        14.4 mm clear (it ends 8 mm short of the MTP condyle), digit II's phalanges 5.9 mm
        and its sickle claw 19.6 mm.
        """
        assert env.home_ground_clearance() == pytest.approx(-0.0005, abs=1e-5)
        self._at_home(env)
        for side in ("r", "l"):
            for digit in ("d3", "d4"):
                geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_toe_{digit}_geom")
                axis = env.data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
                assert abs(math.degrees(math.asin(abs(axis[2])))) < 0.05, f"{side} digit {digit} is not level"
            for geom_name in SUPPORT_GEOM_NAMES[side]:
                distance = _geom_floor_distance(env, geom_name)
                assert -0.0006 <= distance <= 0.0, f"{geom_name} at {1000 * distance:.3f} mm"
            assert _geom_floor_distance(env, f"{side}_metatarsus_geom") > 0.010
            for geom_name in (f"{side}_toe_d2_p1", f"{side}_toe_d2_p2", f"{side}_claw_geom"):
                assert _geom_floor_distance(env, geom_name) > 0.003, geom_name
        assert set(_floor_force_by_geom(env.model, env.data)) == set(FOOT_GEOM_NAMES)

    def test_com_sits_well_inside_the_support_at_home(self, env):
        """The CoM projects 5.9 cm inside the floor-contact hull (12 contact points, measured on r5).

        The r3 / r4 keyframe kept it about 5 cm inside by holding the hip over the flattened
        foot; the life-size foot keeps it there with the level trunk (the CoM 0.113 m ahead of
        the hip, over the MTP pads and the digits).  The loose HomePoseCOMBase bounds cannot
        see a foot that slides out from under the CoM; this 4 cm bar can.
        """
        self._at_home(env)
        floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        points = np.asarray(
            [
                env.data.contact[index].pos[:2]
                for index in range(env.data.ncon)
                if floor_id in (env.data.contact[index].geom1, env.data.contact[index].geom2)
            ]
        )
        hull = _convex_hull(points)
        com = env.data.subtree_com[env.pelvis_id][:2]
        edges = np.roll(hull, -1, axis=0) - hull
        offsets = com - hull
        # Signed distance to each counter-clockwise edge: positive inside.
        signed = (edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]) / np.linalg.norm(edges, axis=1)
        margin = float(np.min(signed))
        assert margin > 0.04, f"CoM only {margin:.4f} m inside the support polygon"

    def test_servos_hold_the_stance_from_a_gravity_preload(self, env):
        """Zero action holds the keyframe on the servos alone.

        The home ctrl sits past the pose by tau_g / kp, so each servo's steady-state error is
        the gravity torque and the joints settle on the keyframe (measured on r5 after 600
        noise-free steps: every joint within 0.0026 deg of it).  The hold costs at most 3.4 %
        of a leg forcerange (hip roll; r4 5.8 % on its servoed toe d3) and 10.2 % of the neck
        and head servos' (neck pitch 10.17 %, head pitch 10.10 %: they hold the head's weight
        out ahead of the trunk), where the r2 ankle spent 23 % of its range on cancelling a
        spring anchored at qpos 0.
        """
        _settle(env, 600)
        home = env.model.key_qpos[env.home_keyframe_id]
        for joint_id in range(1, env.model.njnt):
            address = env.model.jnt_qposadr[joint_id]
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            assert abs(math.degrees(env.data.qpos[address] - home[address])) < 0.25, name
        for names, bound in ((LEG_ACTUATORS, 0.10), (NECK_ACTUATORS, 0.15)):
            for name in names:
                actuator_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                force = abs(float(env.data.actuator_force[actuator_id]))
                assert force < bound * env.model.actuator_forcerange[actuator_id, 1], name

    def test_statue_stands_a_full_noise_free_episode_without_leg_springs(self, env):
        """The servoed joints' springs are a declared compliance, not the support.

        With the hip, knee and ankle springs deleted the servos still hold the full horizon
        (noise-free here; 40/40 seeds 3042-3081 at the stance's reset noise, stage 1 of the
        life-size port); on the r2 plant the same edit fell onto its tail at step 111.  The
        passive digits keep theirs: a passive toe's spring is its only support, as on the
        T-Rex (r4 deleted its servoed toes' springs too).
        """
        for name in LEG_JOINTS:
            env.model.jnt_stiffness[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)] = 0.0
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for step in range(1, env.max_episode_steps + 1):
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated, f"fell at step {step} without leg springs: {info.get('termination_reason')}"
        assert truncated

    def test_settled_stance_loads_every_support_geom_without_a_reset_pop(self, env):
        """The MTP pad, digit III and digit IV all bear load, gently, and nothing else touches the floor.

        The noise-free reset lands the 0.5 mm contact at about one body weight (1.003 BW
        measured on r5), where the r2 plant's 44.6 mm spawn depth popped 2.26 BW on every
        reset.  At step 100 each foot's 73.6 N sits 0.300 / 0.365 / 0.336 on the pad, digit III
        and digit IV (22.1 / 26.8 / 24.7 N; r4 stood on d3 / d4 / metatarsus at 0.540 / 0.176
        / 0.284); every substep of the settle loads the six support geoms alone.
        """
        weight = float(env.model.body_subtreemass[env.pelvis_id]) * abs(float(env.model.opt.gravity[2]))
        peak = 0.0
        touching: set[str] = set()

        def probe():
            nonlocal peak
            loads = _floor_force_by_geom(env.model, env.data)
            touching.update(loads)
            peak = max(peak, sum(loads.values()))

        env.reset(seed=0)
        env._substep_probe_hook = probe
        try:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(100):
                env.step(action)
        finally:
            env._substep_probe_hook = None
        assert peak < 1.1 * weight, f"reset peak floor force {peak / weight:.3f} BW"
        assert touching == set(FOOT_GEOM_NAMES), f"non-support floor contact: {sorted(touching - set(FOOT_GEOM_NAMES))}"
        loads = _floor_force_by_geom(env.model, env.data)
        for side in ("r", "l"):
            foot = sum(loads.get(geom_name, 0.0) for geom_name in SUPPORT_GEOM_NAMES[side])
            for geom_name in SUPPORT_GEOM_NAMES[side]:
                assert loads.get(geom_name, 0.0) > 1.0, f"{geom_name} carries {loads.get(geom_name, 0.0):.3f} N"
                assert 0.2 < loads[geom_name] / foot < 0.45, f"{geom_name} carries {loads[geom_name] / foot:.3f}"


class TestSickleClawClearsTheFloor:
    """The sickle claw never meets the floor, nor stands in for a foot at reset (physics r5).

    The claw is contype / conaffinity 2/2 against the floor's 1/1, so it meets only the prey,
    but the shared reset settles the root on the lowest geom with a non-zero contype
    (``_root_subtree_geoms``), and a claw lower than the feet would settle them off the floor.
    Digit II hinges passively on its spring and the claw is a bounded 4 N.m servo, so their
    whole joint ranges are swept here (measured: the claw 3.30 mm clear at worst), and the
    stance's noisy reset (0.05, uniform on every non-root joint, the passive digits included)
    keeps it 16.9 mm clear on seeds 3042-3241 and never the lowest geom.
    """

    def test_the_claw_clears_the_floor_over_the_whole_digit_ii_and_claw_range(self, env):
        model, data = env.model, env.data
        home = model.key_qpos[env.home_keyframe_id]
        worst = math.inf
        for side in ("r", "l"):
            digit = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_toe_d2_joint")
            claw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_claw")
            for digit_angle in np.linspace(*model.jnt_range[digit], 11):
                for claw_angle in np.linspace(*model.jnt_range[claw], 21):
                    data.qpos[:] = home
                    data.qpos[model.jnt_qposadr[digit]] = digit_angle
                    data.qpos[model.jnt_qposadr[claw]] = claw_angle
                    mujoco.mj_kinematics(model, data)
                    worst = min(worst, _geom_floor_distance(env, f"{side}_claw_geom"))
        assert worst > 0.003, f"the claw comes {1000 * worst:.2f} mm from the floor"

    def test_the_stance_reset_keeps_the_claw_clear_and_never_lowest(self):
        from environments.shared.config import load_stage_config
        from environments.shared.model_id import ModelId

        kwargs = load_stage_config(ModelId("velociraptor_mongoliensis", "anatomical"), "stance")["env_kwargs"]
        env = VelociraptorMongoliensisAnatomicalEnv(**kwargs)
        try:
            claws = {env.r_claw_geom_id, env.l_claw_geom_id}
            colliding = [
                geom
                for geom in range(env.model.ngeom)
                if env.model.geom_contype[geom]
                and geom != env.floor_geom_id
                and env.model.geom_bodyid[geom] != env.prey_mocap_id
            ]
            for seed in range(3042, 3082):
                env.reset(seed=seed)
                distance = {
                    geom: float(mujoco.mj_geomDistance(env.model, env.data, geom, env.floor_geom_id, 1.0, None))
                    for geom in colliding
                }
                assert min(distance, key=distance.get) not in claws, f"seed {seed}: the claw is the lowest geom"
                assert min(distance[geom] for geom in claws) > 0.010, f"seed {seed}"
        finally:
            env.close()


class TestMassDistribution(MassDistributionBase):
    root_body = ROOT_BODY
    mass_range = (10.0, 25.0)
    leg_body_names = [
        "r_thigh",
        "r_tibia",
        "r_metatarsus",
        "r_toe_d2",
        "r_toe_claw",
        "r_toe_d3",
        "r_toe_d4",
        "l_thigh",
        "l_tibia",
        "l_metatarsus",
        "l_toe_d2",
        "l_toe_claw",
        "l_toe_d3",
        "l_toe_d4",
    ]
    min_leg_fraction = 0.15
    tail_body_names = ["tail_1", "tail_2", "tail_3", "tail_4", "tail_5"]
    max_tail_fraction = 0.30
    symmetry_pairs = [
        ("r_thigh", "l_thigh"),
        ("r_tibia", "l_tibia"),
        ("r_metatarsus", "l_metatarsus"),
        ("r_toe_d2", "l_toe_d2"),
        ("r_toe_claw", "l_toe_claw"),
        ("r_toe_d3", "l_toe_d3"),
        ("r_toe_d4", "l_toe_d4"),
        ("r_arm", "l_arm"),
        ("r_forearm", "l_forearm"),
        ("r_hand", "l_hand"),
    ]

    def test_the_life_size_mass_budget(self, env):
        """15.0 kg (Dececchi et al. 2020, IGM 100/986): trunk 7.1, tail 2.05, neck and head 1.65, legs 1.8 and arms
        0.3 kg each (the XML's budget; the tail's gravity carried by its springref preloads)."""
        model = env.model

        def subtree(name):
            return float(model.body_subtreemass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)])

        assert subtree("pelvis") == pytest.approx(15.0, abs=1e-9)
        assert subtree("tail_1") == pytest.approx(2.05, abs=1e-9)
        assert subtree("neck") == pytest.approx(1.65, abs=1e-9)
        assert subtree("r_thigh") == subtree("l_thigh") == pytest.approx(1.8, abs=1e-9)
        assert subtree("r_arm") == subtree("l_arm") == pytest.approx(0.3, abs=1e-9)
        trunk = subtree("pelvis") - subtree("tail_1") - subtree("neck") - 2 * subtree("r_thigh") - 2 * subtree("r_arm")
        assert trunk == pytest.approx(7.1, abs=1e-9)
