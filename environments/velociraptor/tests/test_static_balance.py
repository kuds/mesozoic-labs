"""Tests that the raptor model is physically set up for static balance.

Validates the home keyframe: COM projection, support polygon, joint limits,
neutral-action stability, and actuator-disabled passive behavior. These catch
model regressions (e.g. mass changes that shift COM behind the feet, or
keyframe edits that violate joint limits) before any RL training is attempted.

The raptor model uses a ~20 deg forward-leaning pelvis to place the COM over the
digitigrade feet, matching dromaeosaurid biomechanics. The tilt tests account
for this natural lean.
"""

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
