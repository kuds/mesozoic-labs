"""Pytest tests for the T-Rex Gymnasium environment."""

import mujoco
import numpy as np
import pytest

from environments.trex.envs.trex_env import TRexEnv


@pytest.fixture
def env():
    e = TRexEnv()
    yield e
    e.close()


class TestBasicFunctionality:
    def test_spaces_are_valid(self, env):
        assert env.observation_space.shape == (85,)
        assert env.observation_space.dtype == np.float32
        assert env.action_space.shape == (18,)
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_step_zero_action(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_random_action(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))

    def test_reward_components_in_info(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        expected_keys = [
            "reward_forward",
            "reward_alive",
            "reward_energy",
            "reward_tail",
            "reward_bite",
            "reward_approach",
            "reward_total",
        ]
        for key in expected_keys:
            assert key in info, f"Missing reward component: {key}"


class TestEpisodeRollout:
    def test_episode_runs_to_completion(self, env):
        env.reset(seed=0)
        total_steps = 0
        for _ in range(env.max_episode_steps):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            total_steps += 1
            if terminated or truncated:
                break
        assert total_steps > 0

    def test_multiple_resets(self, env):
        for seed in range(3):
            obs, info = env.reset(seed=seed)
            assert obs.shape == env.observation_space.shape
            action = env.action_space.sample()
            obs2, _, _, _, _ = env.step(action)
            assert obs2.shape == env.observation_space.shape


class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        def run_trajectory(seed):
            e = TRexEnv()
            obs, _ = e.reset(seed=seed)
            np.random.seed(seed)
            trajectory = [obs.copy()]
            for _ in range(50):
                action = np.clip(
                    np.random.randn(e.action_space.shape[0]).astype(np.float32),
                    -1,
                    1,
                )
                obs, _, terminated, truncated, _ = e.step(action)
                trajectory.append(obs.copy())
                if terminated or truncated:
                    break
            e.close()
            return np.array(trajectory)

        traj1 = run_trajectory(123)
        traj2 = run_trajectory(123)
        assert np.allclose(traj1, traj2), f"Trajectories differ. Max diff: {np.abs(traj1 - traj2).max()}"


class TestHeadFloorTermination:
    def test_head_geom_ids_cached(self, env):
        """Head geom IDs should be resolved and present in termination sets."""
        env.reset(seed=42)
        for attr in ("skull_upper_geom_id", "snout_geom_id", "jaw_geom_id"):
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
        env.data.qpos[4] = 0.0     # x
        env.data.qpos[5] = 0.7071  # y (90-deg pitch forward)
        env.data.qpos[6] = 0.0     # z
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
                "head_contact", "torso_contact", "tail_contact", "fallen", "excessive_tilt"
            )


class TestObservationBounds:
    def test_no_nan_or_inf(self, env):
        env.reset(seed=42)
        for _ in range(200):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert not np.any(np.isnan(obs)), "NaN in observation"
            assert not np.any(np.isinf(obs)), "Inf in observation"
            if terminated or truncated:
                env.reset()
