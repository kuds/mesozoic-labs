"""Tests that the T-Rex model is physically set up for static balance.

Validates the home keyframe: COM projection, support polygon, joint limits,
neutral-action stability, and actuator-disabled passive behavior. These catch
model regressions (e.g. mass changes that shift COM behind the feet, or
keyframe edits that violate joint limits) before any RL training is attempted.

Mirrors the velociraptor static balance tests, adapted for T-Rex proportions
(heavier mass, higher pelvis, same digitigrade foot structure).
"""

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
from environments.trex.envs.trex_env import TRexEnv

FOOT_GEOM_NAMES = [
    "r_plantar_geom",
    "l_plantar_geom",
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
    e = TRexEnv(reset_noise_scale=0.0, nosedive_termination_threshold=0.35)
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

    def test_plantar_contacts_are_shallow(self, env):
        """The home stance loads the pads without a reset impact."""
        floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        plantar_ids = {
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("r_plantar_geom", "l_plantar_geom")
        }
        contacts = {
            plantar_id: [
                env.data.contact[index]
                for index in range(env.data.ncon)
                if plantar_id in (env.data.contact[index].geom1, env.data.contact[index].geom2)
                and floor_id in (env.data.contact[index].geom1, env.data.contact[index].geom2)
            ]
            for plantar_id in plantar_ids
        }

        assert all(contacts.values()), "both plantar pads must load the floor"
        assert min(contact.dist for pad_contacts in contacts.values() for contact in pad_contacts) >= -0.001


class TestNeutralActionStability(NeutralActionStabilityBase):
    species_name = "T-Rex"
    root_body_id_attr = "pelvis_id"
    max_height_drop = 0.15
    max_tilt_increase = 0.27  # 15 degrees

    def test_neutral_controls_match_complete_home(self, env):
        """Residual action zero must command every home-keyframe control."""
        neutral_ctrl = env._scale_action(np.zeros(env.action_space.shape, dtype=np.float32))
        np.testing.assert_allclose(
            neutral_ctrl,
            env.model.key_ctrl[env.home_keyframe_id],
            atol=1e-12,
        )

    def test_survives_full_noise_free_episode(self, env):
        """The neutral stance must remain viable for the full Stage-1 horizon."""
        env.reset(seed=0)
        neutral_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for step in range(1, env.max_episode_steps + 1):
            _, _, terminated, truncated, info = env.step(neutral_action)
            assert not terminated, (
                f"T-Rex terminated at step {step}/{env.max_episode_steps} "
                f"under its neutral stance command: {info.get('termination_reason', 'unknown')}"
            )
            assert truncated is (step == env.max_episode_steps)

    @pytest.mark.parametrize("seed", (0, 27, 38, 44))
    def test_survives_representative_jax_joint_noise(self, env, seed):
        """The stance tolerates Stage-1 joint noise without SB3-only root noise."""
        env.reset_noise_scale = 0.05
        env.reset(seed=seed)

        home_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        env.data.qpos[2] = env.model.key_qpos[home_id, 2]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        neutral_action = np.zeros(env.action_space.shape, dtype=np.float32)
        for step in range(1, env.max_episode_steps + 1):
            _, _, terminated, truncated, info = env.step(neutral_action)
            assert not terminated, (
                f"T-Rex terminated at step {step}/{env.max_episode_steps} "
                f"from seed {seed} with JAX-style joint noise: "
                f"{info.get('termination_reason', 'unknown')}"
            )
            assert truncated is (step == env.max_episode_steps)


class TestActuatorDisabledPassive(ActuatorDisabledPassiveBase):
    species_name = "T-Rex"
    root_body_id_attr = "pelvis_id"
    max_height_drop = 0.05
    max_tilt_increase = 0.15


class TestJointLimitsAtHome(JointLimitsAtHomeBase):
    knee_names = ["r_knee", "l_knee"]
    knee_margin_deg = 20.0

    def test_leg_springs_reference_home(self, env):
        """Passive leg springs must not fight the intended stance."""
        home_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        joint_names = [
            f"{side}_{suffix}"
            for side in ("r", "l")
            for suffix in (
                "hip_pitch",
                "hip_roll",
                "knee",
                "ankle",
                "toe_d2_joint",
                "toe_d3_joint",
                "toe_d4_joint",
            )
        ]
        for name in joint_names:
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_address = env.model.jnt_qposadr[joint_id]
            assert env.model.qpos_spring[qpos_address] == pytest.approx(
                env.model.key_qpos[home_id, qpos_address],
                abs=1e-6,
            )


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
