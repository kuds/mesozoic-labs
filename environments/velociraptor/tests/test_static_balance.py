"""Tests that the raptor model is physically set up for static balance.

Validates the home keyframe: COM projection, support polygon, joint limits,
neutral-action stability, and actuator-disabled passive behavior. These catch
model regressions (e.g. mass changes that shift COM behind the feet, or
keyframe edits that violate joint limits) before any RL training is attempted.

The raptor model uses a ~20 deg forward-leaning pelvis to place the COM over the
digitigrade feet, matching dromaeosaurid biomechanics. The tilt tests account
for this natural lean.

``TestHomeKeyframeStance`` pins the physics-r3 stance (plant_versions note 14):
flat toes at the authored 0.5 mm contact, the CoM over the foot, springs
anchored at the pose and the servos carrying it from a gravity preload.  The
r2 plant passed every other test in this file while standing on toes 44.6 mm
inside the floor, propped up by springs pulling toward qpos 0.
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
from environments.velociraptor.envs.raptor_env import RaptorEnv

FOOT_GEOM_NAMES = [
    "r_toe_d3_geom",
    "l_toe_d3_geom",
    "r_toe_d4_geom",
    "l_toe_d4_geom",
    "r_metatarsus_geom",
    "l_metatarsus_geom",
]
ROOT_BODY = "pelvis"


@pytest.fixture
def env():
    e = RaptorEnv(reset_noise_scale=0.0)
    e.reset(seed=0)
    yield e
    e.close()


class TestHomePoseCOM(HomePoseCOMBase):
    foot_geom_names = FOOT_GEOM_NAMES
    root_body = ROOT_BODY
    ankle_body_names = ("r_metatarsus", "l_metatarsus")
    max_ankle_offset = 0.10
    max_support_distance = 0.15
    species_label = "raptor"


class TestNeutralActionStability(NeutralActionStabilityBase):
    species_name = "Raptor"
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
                f"Raptor terminated at step {step}/{env.max_episode_steps} "
                f"under the XML home command: {info.get('termination_reason', 'unknown')}"
            )
            assert truncated is (step == env.max_episode_steps)


class TestActuatorDisabledPassive(ActuatorDisabledPassiveBase):
    species_name = "Raptor"
    root_body_id_attr = "pelvis_id"
    max_height_drop = 0.08
    max_tilt_increase = 0.30


class TestJointLimitsAtHome(JointLimitsAtHomeBase):
    knee_names = ["r_knee", "l_knee"]
    knee_margin_deg = 20.0


LEG_JOINTS = [
    f"{side}_{suffix}"
    for side in ("r", "l")
    for suffix in ("hip_pitch", "hip_roll", "knee", "ankle", "toe_d3_joint", "toe_d4_joint")
]
LEG_ACTUATORS = [
    f"{side}_{suffix}_act"
    for side in ("r", "l")
    for suffix in ("hip_pitch", "hip_roll", "knee", "ankle", "toe_d3", "toe_d4")
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


class TestHomeKeyframeStance:
    """The physics-r3 home keyframe stands on flat toes, on its servos."""

    @staticmethod
    def _at_home(env):
        mujoco.mj_resetDataKeyframe(env.model, env.data, env.home_keyframe_id)
        mujoco.mj_forward(env.model, env.data)

    def test_leg_springs_reference_home(self, env):
        """Passive leg springs must not fight the intended stance.

        MuJoCo's default springref is 0, not ref: before r3 the springs pulled
        the settled hip, knee and ankle toward qpos 0 with 11.9 / 15.8 / 28.0
        N.m, which the servos then spent most of their torque cancelling.
        """
        home = env.model.key_qpos[env.home_keyframe_id]
        spring_torque = 0.0
        for name in LEG_JOINTS:
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            address = env.model.jnt_qposadr[joint_id]
            assert env.model.qpos_spring[address] == pytest.approx(home[address], abs=1e-6), name
            spring_torque += abs(env.model.jnt_stiffness[joint_id] * (home[address] - env.model.qpos_spring[address]))
        assert spring_torque < 1e-4

    def test_keyframe_toes_lie_flat_at_the_authored_contact(self, env):
        """Digit 3 is level and every support geom touches at 0.5 mm or less.

        The r2 keyframe tipped the toes 20 deg toe-down, 44.6 mm into the
        floor, and every reset settles to the keyframe's own clearance.
        """
        assert env.home_ground_clearance() == pytest.approx(-0.0005, abs=1e-5)
        self._at_home(env)
        floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for side in ("r", "l"):
            toe = env.data.xmat[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_toe_d3")]
            assert abs(math.degrees(math.asin(toe.reshape(3, 3)[2, 0]))) < 0.05
            for geom_name in (f"{side}_toe_d3_geom", f"{side}_toe_d4_geom", f"{side}_metatarsus_geom"):
                geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                distance = mujoco.mj_geomDistance(env.model, env.data, geom_id, floor_id, 1.0, None)
                assert -0.0006 <= distance <= 0.0, f"{geom_name} at {1000 * distance:.3f} mm"

    def test_com_sits_well_inside_the_support_at_home(self, env):
        """The CoM projects ~5 cm inside the floor-contact hull.

        Keeping hip 38 / knee -50 and flattening the foot at the ankle and toe
        is what keeps it there: a hip-18 keyframe puts the support 7.6 cm
        ahead of the CoM, which the loose HomePoseCOMBase bounds cannot see.
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

        The home ctrl sits past the pose by tau_g / kp, so each servo's
        steady-state error is the gravity torque and the joints settle on
        the keyframe.  The hold costs at most 5.8% of a leg forcerange (toe
        d3; the ankle 3.7%), where the r2 ankle spent 23% of its range on
        cancelling a spring anchored at qpos 0.
        """
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(600):
            _, _, terminated, _, _ = env.step(action)
            assert not terminated
        home = env.model.key_qpos[env.home_keyframe_id]
        for name in LEG_JOINTS:
            address = env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            assert abs(math.degrees(env.data.qpos[address] - home[address])) < 0.25, name
        for name in LEG_ACTUATORS:
            actuator_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            force = abs(float(env.data.actuator_force[actuator_id]))
            assert force < 0.10 * env.model.actuator_forcerange[actuator_id, 1], name

    def test_statue_stands_a_full_noise_free_episode_without_leg_springs(self, env):
        """The springs are a declared compliance, not the support.

        With every leg spring deleted the servos still hold the full horizon;
        on the r2 plant the same edit fell onto its tail at step 111.
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
        """Digit 3, digit 4 and the metatarsal head all bear load, gently.

        The noise-free reset lands the 0.5 mm contact at about one body weight
        (1.015 BW measured), where the r2 plant's 44.6 mm spawn depth popped
        2.26 BW on every reset.
        """
        weight = float(env.model.body_subtreemass[env.pelvis_id]) * abs(float(env.model.opt.gravity[2]))
        peak = 0.0

        def probe():
            nonlocal peak
            peak = max(peak, sum(_floor_force_by_geom(env.model, env.data).values()))

        env.reset(seed=0)
        env._substep_probe_hook = probe
        try:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(100):
                env.step(action)
        finally:
            env._substep_probe_hook = None
        assert peak < 1.1 * weight, f"reset peak floor force {peak / weight:.3f} BW"
        loads = _floor_force_by_geom(env.model, env.data)
        for side in ("r", "l"):
            for geom_name in (f"{side}_toe_d3_geom", f"{side}_toe_d4_geom", f"{side}_metatarsus_geom"):
                assert loads.get(geom_name, 0.0) > 1.0, f"{geom_name} carries {loads.get(geom_name, 0.0):.3f} N"


class TestMassDistribution(MassDistributionBase):
    root_body = ROOT_BODY
    mass_range = (10.0, 25.0)
    leg_body_names = [
        "r_thigh",
        "r_tibia",
        "r_metatarsus",
        "r_toe_d3",
        "r_toe_d4",
        "r_toe_claw",
        "l_thigh",
        "l_tibia",
        "l_metatarsus",
        "l_toe_d3",
        "l_toe_d4",
        "l_toe_claw",
    ]
    min_leg_fraction = 0.15
    tail_body_names = ["tail_1", "tail_2", "tail_3", "tail_4", "tail_5"]
    max_tail_fraction = 0.30
    symmetry_pairs = [
        ("r_thigh", "l_thigh"),
        ("r_tibia", "l_tibia"),
        ("r_metatarsus", "l_metatarsus"),
        ("r_toe_d3", "l_toe_d3"),
        ("r_toe_d4", "l_toe_d4"),
        ("r_toe_claw", "l_toe_claw"),
    ]
