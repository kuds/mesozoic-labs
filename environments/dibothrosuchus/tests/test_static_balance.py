"""Tests that the Dibothrosuchus model is physically set up correctly.

Validates the home keyframe: joint limits, mass distribution, and foot
contacts. These catch model regressions (e.g. mass changes, keyframe edits
that violate joint limits) before any RL training is attempted.

Dibothrosuchus is a small erect-limbed quadruped whose hindlimbs are longer
than its forelimbs.  Unlike the other species the stance geometry is baked
into the body offsets rather than into joint references, so every hinge joint
sits at 0 in the home keyframe and all four pads contact the floor at reset
with a 0.5 mm preload.
"""

import mujoco
import pytest

from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.shared.tests.static_balance_helpers import (
    ActuatorDisabledPassiveBase,
    JointLimitsAtHomeBase,
    MassDistributionBase,
    NeutralActionStabilityBase,
    body_group_mass,
    com_xy,
    get_foot_contacts_xy,
)

FOOT_GEOM_NAMES = ["fr_foot_geom", "fl_foot_geom", "rr_foot_geom", "rl_foot_geom"]
ROOT_BODY = "torso"


@pytest.fixture
def env():
    e = DibothrosuchusEnv(reset_noise_scale=0.0)
    e.reset(seed=0)
    yield e
    e.close()


class TestHomePoseCOM:
    """Verify the home keyframe COM and foot contact properties."""

    def test_all_four_feet_contact(self, env):
        contacts = get_foot_contacts_xy(env.model, env.data, FOOT_GEOM_NAMES)
        assert len(contacts) >= 4, (
            f"Expected at least 4 pad-floor contacts at home pose, got {len(contacts)}. "
            "The keyframe height may no longer place all four pads on the floor."
        )

    def test_com_centered_laterally(self, env):
        com = com_xy(env.model, env.data, ROOT_BODY)
        assert abs(com[1]) < 0.02, (
            f"COM Y ({com[1]:.4f}) is off-center by more than 2 cm on a 1.24 m animal. "
            "The model may have asymmetric mass distribution."
        )

    def test_com_between_the_girdles(self, env):
        """A hindlimb-dominant quadruped carries its COM behind the midpoint of
        the support polygon, but it must still sit between the two girdles."""
        com = com_xy(env.model, env.data, ROOT_BODY)
        shoulder_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "fr_thigh")
        hip_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rr_thigh")
        front_x = env.data.xpos[shoulder_id, 0]
        rear_x = env.data.xpos[hip_id, 0]
        assert rear_x < com[0] < front_x, (
            f"COM X ({com[0]:.4f}) is outside the girdle span [{rear_x:.4f}, {front_x:.4f}]."
        )

    def test_rear_pads_carry_more_load_than_front(self, env):
        """The hindlimbs are the longer, heavier pair and take the larger share."""
        for _ in range(600):
            mujoco.mj_step(env.model, env.data)
        front = env.data.sensordata[env._sensor_fr_foot] + env.data.sensordata[env._sensor_fl_foot]
        rear = env.data.sensordata[env._sensor_rr_foot] + env.data.sensordata[env._sensor_rl_foot]
        assert rear > front, f"Rear pad load {rear:.2f} N should exceed front {front:.2f} N."

    def test_tail_is_carried_clear_of_the_ground(self, env):
        """The croc high walk holds the whole tail off the floor."""
        for _ in range(600):
            mujoco.mj_step(env.model, env.data)
        tail_tip_z = env.data.site_xpos[env.tail_tip_site_id, 2]
        assert tail_tip_z > 0.15, (
            f"Tail tip settled at {tail_tip_z:.3f} m. The tail joint stiffness may have "
            "regressed, which makes incidental tail-ground contact the dominant failure."
        )


class TestNeutralActionStability(NeutralActionStabilityBase):
    species_name = "Dibothrosuchus"
    root_body_id_attr = "torso_id"
    max_height_drop = 0.02  # 6% of the 0.313 m stance
    max_tilt_increase = 0.10  # ~6 degrees

    def test_trunk_starts_in_healthy_range(self, env):
        torso_z = env.data.xpos[env.torso_id, 2]
        assert 0.18 < torso_z < 0.55, f"Trunk z ({torso_z:.3f}) is outside plausible range at reset."


class TestActuatorDisabledPassive(ActuatorDisabledPassiveBase):
    species_name = "Dibothrosuchus"
    root_body_id_attr = "torso_id"
    max_height_drop = 0.03
    max_tilt_increase = 0.10


class TestJointLimitsAtHome(JointLimitsAtHomeBase):
    knee_names = ["fr_knee", "fl_knee", "rr_knee", "rl_knee"]
    knee_margin_deg = 15.0


class TestMassDistribution(MassDistributionBase):
    root_body = ROOT_BODY
    mass_range = (7.0, 11.0)
    leg_body_names = [
        "fr_thigh",
        "fr_shin",
        "fr_meta",
        "fr_foot",
        "fl_thigh",
        "fl_shin",
        "fl_meta",
        "fl_foot",
        "rr_thigh",
        "rr_shin",
        "rr_meta",
        "rr_foot",
        "rl_thigh",
        "rl_shin",
        "rl_meta",
        "rl_foot",
    ]
    min_leg_fraction = 0.20
    tail_body_names = ["tail_1", "tail_2", "tail_3", "tail_4"]
    max_tail_fraction = 0.20
    symmetry_pairs = [
        ("fr_thigh", "fl_thigh"),
        ("fr_shin", "fl_shin"),
        ("fr_meta", "fl_meta"),
        ("fr_foot", "fl_foot"),
        ("rr_thigh", "rl_thigh"),
        ("rr_shin", "rl_shin"),
        ("rr_meta", "rl_meta"),
        ("rr_foot", "rl_foot"),
    ]

    def test_rear_legs_heavier_than_front(self, env):
        """Hindlimbs are the longer, more robust pair — the reverse of Brachiosaurus."""
        front_names = ["fr_thigh", "fr_shin", "fr_meta", "fr_foot", "fl_thigh", "fl_shin", "fl_meta", "fl_foot"]
        rear_names = ["rr_thigh", "rr_shin", "rr_meta", "rr_foot", "rl_thigh", "rl_shin", "rl_meta", "rl_foot"]
        front_mass = body_group_mass(env.model, front_names)
        rear_mass = body_group_mass(env.model, rear_names)
        assert rear_mass > front_mass, (
            f"Rear leg mass ({rear_mass:.2f} kg) should exceed front leg mass ({front_mass:.2f} kg) — "
            "Dibothrosuchus has characteristically longer hindlimbs."
        )

    def test_paramedian_osteoderms_are_present_and_paired(self, env):
        """The two dorsal scute rows are what the genus is named for."""
        right = [
            name
            for i in range(env.model.ngeom)
            if (name := mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)) and name.startswith("scute_r_")
        ]
        left = [
            name
            for i in range(env.model.ngeom)
            if (name := mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)) and name.startswith("scute_l_")
        ]
        assert len(right) == len(left) >= 5, f"Expected paired paramedian scute rows, got {right} / {left}"
        for name in right + left:
            gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert env.model.geom_contype[gid] == 0 and env.model.geom_conaffinity[gid] == 0, (
                f"{name} must stay collision-disabled so it cannot inject self-contact into the trunk"
            )


class TestHomeGeometry:
    """Pin the authored stance so a keyframe edit cannot silently change it."""

    def test_every_hinge_joint_is_zero_at_home(self, env):
        home = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        qpos = env.model.key_qpos[home]
        assert qpos[7:].tolist() == [0.0] * (env.model.nq - 7), (
            "The stance is authored into the body offsets: every hinge joint must be 0 at home."
        )

    def test_home_control_vector_is_zero(self, env):
        home = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert env.model.key_ctrl[home].tolist() == [0.0] * env.model.nu

    def test_snout_tip_and_tail_tip_span_the_animal(self, env):
        """Nose-to-tail length is the headline dimension in the MJCF comment."""
        snout_x = env.data.site_xpos[env.snout_tip_site_id, 0]
        tail_x = env.data.site_xpos[env.tail_tip_site_id, 0]
        length = float(snout_x - tail_x)
        assert 1.15 < length < 1.35, f"Nose-to-tail length {length:.3f} m drifted from the documented 1.24 m."
