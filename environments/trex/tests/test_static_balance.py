"""Tests that the T-Rex model is physically set up for static balance.

Validates the home keyframe: COM projection, support polygon, joint limits,
neutral-action stability, and actuator-disabled passive behavior. These catch
model regressions (e.g. mass changes that shift COM behind the feet, or
keyframe edits that violate joint limits) before any RL training is attempted.

Mirrors the velociraptor static balance tests, adapted for T-Rex proportions
(heavier mass, higher pelvis, same digitigrade foot structure).
"""

import pytest

from environments.shared.tests.static_balance_helpers import (
    ActuatorDisabledPassiveBase,
    HomePoseCOMBase,
    JointLimitsAtHomeBase,
    MassDistributionBase,
    NeutralActionStabilityBase,
)
from environments.trex.envs.trex_env import TRexEnv

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
    e = TRexEnv(reset_noise_scale=0.0)
    e.reset(seed=0)
    yield e
    e.close()


class TestHomePoseCOM(HomePoseCOMBase):
    foot_geom_names = FOOT_GEOM_NAMES
    root_body = ROOT_BODY
    ankle_body_names = ("r_metatarsus", "l_metatarsus")
    max_ankle_offset = 0.15
    max_support_distance = 0.20
    species_label = "T-Rex"


class TestNeutralActionStability(NeutralActionStabilityBase):
    species_name = "T-Rex"
    root_body_id_attr = "pelvis_id"
    max_height_drop = 0.15
    max_tilt_increase = 0.27  # 15 degrees


class TestActuatorDisabledPassive(ActuatorDisabledPassiveBase):
    species_name = "T-Rex"
    root_body_id_attr = "pelvis_id"
    max_height_drop = 0.05
    max_tilt_increase = 0.15


class TestJointLimitsAtHome(JointLimitsAtHomeBase):
    knee_names = ["r_knee", "l_knee"]
    knee_margin_deg = 20.0


class TestMassDistribution(MassDistributionBase):
    root_body = ROOT_BODY
    mass_range = (60.0, 100.0)
    leg_body_names = [
        "r_thigh",
        "r_tibia",
        "r_metatarsus",
        "r_toe_d2",
        "r_toe_d3",
        "r_toe_d4",
        "l_thigh",
        "l_tibia",
        "l_metatarsus",
        "l_toe_d2",
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
        ("r_toe_d3", "l_toe_d3"),
        ("r_toe_d4", "l_toe_d4"),
    ]
