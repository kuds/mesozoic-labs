"""Gym control, foot sensing and the standing and walking task boundaries."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import environments  # noqa: F401
from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId

MODEL = ModelId("allosaurus_fragilis", "anatomical")


def test_allosaurus_gym_home_and_torque_endpoints():
    with gym.make(MODEL.gym_id, reset_noise_scale=0) as wrapped:
        env = wrapped.unwrapped
        observation, _ = wrapped.reset(seed=42)
        assert wrapped.observation_space.contains(observation)
        for level, torque in (
            (-1.0, env.model.actuator_ctrlrange[:, 0]),
            (0.0, env.model.key("home").ctrl),
            (1.0, env.model.actuator_ctrlrange[:, 1]),
        ):
            np.testing.assert_allclose(env._scale_action(np.full(env.model.nu, level)), torque, atol=1e-8)
        observation, reward, terminated, truncated, info = wrapped.step(np.zeros(env.model.nu, dtype=np.float32))
        assert wrapped.observation_space.contains(observation)
        assert not terminated and not truncated
        assert min(info["r_foot_contact"], info["l_foot_contact"]) > 0
        components = [value for key, value in info.items() if key.startswith("reward_") and key != "reward_total"]
        assert reward == pytest.approx(sum(components))


@pytest.mark.parametrize(("stage", "should_finish"), [("stance", False), ("locomotion", True)])
def test_course_target_only_ends_supported_walking(stage, should_finish):
    kwargs = {**load_stage_config(MODEL, stage)["env_kwargs"], "reset_noise_scale": 0}
    with gym.make(MODEL.gym_id, **kwargs) as wrapped:
        env = wrapped.unwrapped
        wrapped.reset(seed=0)
        # Bring the invisible goal to a settled pelvis, preserving all foot contacts.
        for _ in range(4):
            wrapped.step(np.zeros(env.model.nu, dtype=np.float32))
        env.data.mocap_pos[0, :2] = env.data.xpos[env.pelvis_id, :2]
        _, _, terminated, _, info = wrapped.step(np.zeros(env.model.nu, dtype=np.float32))
        assert terminated == should_finish
        assert bool(info["course_complete"]) == should_finish
        # An airborne arrival cannot complete the walk, even at the same XY.
        env.data.qpos[2] += 0.15
        mujoco.mj_forward(env.model, env.data)
        env._invalidate_substep_aggregates()
        terminated, _ = env._is_terminated()
        assert not terminated
