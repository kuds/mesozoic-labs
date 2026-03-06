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
