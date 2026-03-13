"""Tests that the raptor model is physically set up for static balance.

Validates the home keyframe: COM projection, support polygon, joint limits,
and zero-torque stability. These catch model regressions (e.g. mass changes
that shift COM behind the feet, or keyframe edits that violate joint limits)
before any RL training is attempted.

The raptor model uses a ~20° forward-leaning pelvis to place the COM over the
digitigrade feet, matching dromaeosaurid biomechanics. The tilt tests account
for this natural lean.
"""

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from environments.shared.tests.static_balance_helpers import (
    JointLimitsAtHomeBase,
    MassDistributionBase,
    ZeroTorqueStabilityBase,
    com_xy,
    get_foot_contacts_xy,
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
    e = RaptorEnv()
    e.reset(seed=0)
    yield e
    e.close()


class TestHomePoseCOM:
    """Verify the home keyframe places COM over the support polygon."""

    def test_foot_contacts_exist(self, env):
        contacts = get_foot_contacts_xy(env.model, env.data, FOOT_GEOM_NAMES)
        assert len(contacts) >= 2, (
            f"Expected at least 2 foot-floor contacts at home pose, got {len(contacts)}. "
            "The raptor may be floating or the keyframe places feet above the floor."
        )

    def test_com_inside_support_polygon(self, env):
        contacts = get_foot_contacts_xy(env.model, env.data, FOOT_GEOM_NAMES)
        if len(contacts) < 3:
            if len(contacts) == 0:
                pytest.skip("No foot contacts detected (model may need ground settling)")
            com = com_xy(env.model, env.data, ROOT_BODY)
            bbox_min = contacts.min(axis=0) - 0.05
            bbox_max = contacts.max(axis=0) + 0.05
            assert np.all(com >= bbox_min) and np.all(com <= bbox_max), (
                f"COM XY {com} is outside bounding box of foot contacts [{bbox_min}, {bbox_max}]"
            )
            return

        com = com_xy(env.model, env.data, ROOT_BODY)
        hull = ConvexHull(contacts)
        inside = np.all(hull.equations[:, :2] @ com + hull.equations[:, 2] <= 0)
        assert inside, (
            f"COM XY {com} is outside the convex hull of foot contacts. "
            f"Contact points: {contacts.tolist()}. "
            "The model's mass distribution may not balance over its feet."
        )

    def test_com_not_too_far_back(self, env):
        com = com_xy(env.model, env.data, ROOT_BODY)
        r_meta_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "r_metatarsus")
        l_meta_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "l_metatarsus")
        avg_ankle_x = (env.data.xpos[r_meta_id, 0] + env.data.xpos[l_meta_id, 0]) / 2.0
        assert com[0] >= avg_ankle_x - 0.10, (
            f"COM X ({com[0]:.3f}) is more than 10 cm behind the ankle midpoint "
            f"({avg_ankle_x:.3f}). The tail mass may be pulling the COM too far "
            "rearward for stable bipedal balance."
        )

    def test_com_distance_from_support(self, env):
        contacts = get_foot_contacts_xy(env.model, env.data, FOOT_GEOM_NAMES)
        com = com_xy(env.model, env.data, ROOT_BODY)
        if len(contacts) == 0:
            pytest.skip("No foot contacts")
        dists = np.linalg.norm(contacts - com, axis=1)
        min_dist = dists.min()
        assert min_dist < 0.15, (
            f"COM is {min_dist:.3f} m from nearest foot contact. "
            f"COM: {com}, nearest contact: {contacts[dists.argmin()]}"
        )


class TestZeroTorqueStability(ZeroTorqueStabilityBase):
    species_name = "Raptor"
    root_body_id_attr = "pelvis_id"
    max_height_drop = 0.10


class TestJointLimitsAtHome(JointLimitsAtHomeBase):
    knee_names = ["r_knee", "l_knee"]
    knee_margin_deg = 20.0


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
