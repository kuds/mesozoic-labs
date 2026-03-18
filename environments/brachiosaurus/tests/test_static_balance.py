"""Tests that the Brachiosaurus model is physically set up correctly.

Validates the home keyframe: joint limits, mass distribution, and foot
contacts. These catch model regressions (e.g. mass changes, keyframe edits
that violate joint limits) before any RL training is attempted.

The Brachiosaurus is quadrupedal with front legs longer than rear legs
(giraffe-like posture). Note: unlike the bipedal species, the Brachiosaurus
does not passively balance at the home pose — it requires active control
from the RL policy. The front feet contact the ground at reset, but the
rear feet may settle over the first few simulation steps.
"""

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.shared.tests.static_balance_helpers import (
    JointLimitsAtHomeBase,
    MassDistributionBase,
    body_group_mass,
    com_xy,
    get_foot_contacts_xy,
)

FOOT_GEOM_NAMES = ["fr_foot_geom", "fl_foot_geom", "rr_foot_geom", "rl_foot_geom"]
ROOT_BODY = "torso"


@pytest.fixture
def env():
    e = BrachioEnv()
    e.reset(seed=0)
    yield e
    e.close()


class TestHomePoseCOM:
    """Verify the home keyframe COM and foot contact properties."""

    def test_foot_contacts_exist(self, env):
        contacts = get_foot_contacts_xy(env.model, env.data, FOOT_GEOM_NAMES)
        assert len(contacts) >= 2, (
            f"Expected at least 2 foot-floor contacts at home pose, got {len(contacts)}. "
            "The Brachiosaurus may be floating or the keyframe places feet above the floor."
        )

    def test_com_centered_laterally(self, env):
        com = com_xy(env.model, env.data, ROOT_BODY)
        assert abs(com[1]) < 0.15, (
            f"COM Y ({com[1]:.3f}) is off-center by more than 15 cm. The model may have asymmetric mass distribution."
        )

    def test_com_x_between_front_and_rear_hips(self, env):
        com = com_xy(env.model, env.data, ROOT_BODY)
        fr_thigh_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "fr_thigh")
        rr_thigh_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rr_thigh")
        front_x = env.data.xpos[fr_thigh_id, 0]
        rear_x = env.data.xpos[rr_thigh_id, 0]
        min_x = min(front_x, rear_x) - 0.30
        max_x = max(front_x, rear_x) + 0.30
        assert min_x <= com[0] <= max_x, (
            f"COM X ({com[0]:.3f}) is outside hip range [{min_x:.3f}, {max_x:.3f}]. "
            "The mass distribution may be too front- or rear-heavy."
        )


class TestInitialSettling:
    """The Brachiosaurus does not passively balance (requires active control).

    These tests verify the model's settling behavior rather than static stability.
    The torso starts at z=2.0 and settles quickly under gravity.
    """

    def test_torso_starts_in_healthy_range(self, env):
        torso_z = env.data.xpos[env.torso_id, 2]
        assert 1.0 < torso_z < 3.5, f"Torso z ({torso_z:.3f}) is outside plausible range at reset."

    def test_initial_tilt_is_small(self, env):
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        tilt = info.get("tilt_angle", 0.0)
        assert tilt < np.radians(15), f"Initial tilt is {np.degrees(tilt):.1f} deg — model should start near upright."

    def test_settling_drops_less_than_1m(self, env):
        env.reset(seed=0)
        initial_z = env.data.xpos[env.torso_id, 2]
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(10):
            _, _, terminated, _, _ = env.step(zero_action)
            if terminated:
                break
        final_z = env.data.xpos[env.torso_id, 2]
        drop = initial_z - final_z
        assert drop < 1.0, (
            f"Torso dropped {drop:.3f} m in 10 steps — model may be free-falling "
            f"(from {initial_z:.3f} to {final_z:.3f})."
        )


class TestJointLimitsAtHome(JointLimitsAtHomeBase):
    knee_names = ["fr_knee", "fl_knee", "rr_knee", "rl_knee"]
    knee_margin_deg = 5.0


class TestMassDistribution(MassDistributionBase):
    root_body = ROOT_BODY
    mass_range = (150.0, 230.0)
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
    max_tail_fraction = 0.15
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

    def test_front_legs_heavier_than_rear(self, env):
        """Front legs should be heavier than rear legs (longer in Brachiosaurus)."""
        front_names = ["fr_thigh", "fr_shin", "fr_meta", "fr_foot", "fl_thigh", "fl_shin", "fl_meta", "fl_foot"]
        rear_names = ["rr_thigh", "rr_shin", "rr_meta", "rr_foot", "rl_thigh", "rl_shin", "rl_meta", "rl_foot"]
        front_mass = body_group_mass(env.model, front_names)
        rear_mass = body_group_mass(env.model, rear_names)
        assert front_mass > rear_mass, (
            f"Front leg mass ({front_mass:.2f} kg) should exceed rear leg mass "
            f"({rear_mass:.2f} kg) — Brachiosaurus has characteristically longer front legs."
        )
