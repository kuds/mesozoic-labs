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


def test_observation_is_heading_free_and_targetless():
    """12 + 12 hinge channels, gravity, gyro, heading-frame velocity, accelerometer, 2 feet, command: 41.

    bipedal-heading-free/v1 since policy interface r2 (decision D-D59; r1:
    45, with the target direction and distance before the command).  The
    scene has no mocap body, and the command (last 3) is zero under
    ``command_mode = "none"``.
    """
    with gym.make(MODEL.gym_id) as wrapped:
        env = wrapped.unwrapped
        assert env.model.nmocap == 0
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target") == -1
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "target_site") == -1
        assert not hasattr(env, "_target_mocap_id") and not hasattr(env, "_initial_prey_dir_2d")
        wrapped.reset(seed=3042)
        for _ in range(5):
            observation, _, terminated, _, _ = wrapped.step(np.zeros(env.model.nu, dtype=np.float32))
        assert not terminated
        assert observation.shape == (12 + 12 + 3 + 3 + 3 + 3 + 2 + 3,) == wrapped.observation_space.shape
        np.testing.assert_array_equal(observation[-3:], np.zeros(3, dtype=np.float32))
        # The two foot channels sit just before the command.
        touch = [env.data.sensordata[group[0]] for group in env._foot_sensor_groups]
        np.testing.assert_array_equal(observation[-5:-3], np.asarray(touch, dtype=np.float32))

        # A 45 deg turn of the root's pose and velocity about vertical leaves every channel unchanged.
        # After a step the sensors still hold the state before its last integration: recompute them first.
        mujoco.mj_forward(env.model, env.data)
        before = env._get_obs()
        yaw = np.radians(45.0)
        turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, turn, env.data.qpos[3:7].copy())
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        env.data.qpos[0:3] = rotation @ env.data.qpos[0:3]
        env.data.qpos[3:7] = quat
        env.data.qvel[0:3] = rotation @ env.data.qvel[0:3]
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env._get_obs(), before, rtol=0.0, atol=1e-6)


def _forward_walk_kwargs():
    """The forward walk before decision D-D62, under command_mode "none": the stance's [env] with the forward weight
    and the 30 s horizon on, and no drift hold.  Its course ends the episode along the spawn heading; the committed
    walk's live course (an event along the commanded heading) is tested in test_walk_command_mechanism.py."""
    return {
        **load_stage_config(MODEL, "stance")["env_kwargs"],
        "forward_vel_weight": 1.5,
        "drift_penalty_weight": 0.0,
        "max_episode_steps": 3000,
        "reset_noise_scale": 0,
    }


def _carry_along_course(env, along, lateral=0.0):
    """Translate the root to *along* / *lateral* metres from the drift anchor in the spawn-heading frame.

    The floor is a plane, so a horizontal translation keeps every foot contact.
    """
    heading = env._reference_heading_2d
    left = np.array([-heading[1], heading[0]])
    env.data.qpos[0:2] = env._initial_pos_2d + along * heading + lateral * left
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize(("stage", "should_finish"), [("stance", False), ("forward_walk", True)])
def test_course_end_only_ends_supported_walking(stage, should_finish):
    if stage == "forward_walk":
        kwargs = _forward_walk_kwargs()
    else:
        kwargs = {**load_stage_config(MODEL, stage)["env_kwargs"], "reset_noise_scale": 0}
    with gym.make(MODEL.gym_id, **kwargs) as wrapped:
        env = wrapped.unwrapped
        zero = np.zeros(env.model.nu, dtype=np.float32)
        wrapped.reset(seed=0)
        np.testing.assert_array_equal(env._reference_heading_2d, env._quat_to_forward_2d(env.data.qpos[3:7]))
        course_end = env.COURSE_LENGTH_M - env.COURSE_TOLERANCE_M
        for _ in range(4):
            _, _, terminated, _, info = wrapped.step(zero)
            assert not terminated and info["course_complete"] == 0.0
        # Short of the line nothing ends.
        _carry_along_course(env, course_end - 0.05)
        _, _, terminated, _, info = wrapped.step(zero)
        assert not terminated and info["course_complete"] == 0.0
        # Past the line the course ends on the walking stage, here a metre to the side of the old target point.
        _carry_along_course(env, env.COURSE_LENGTH_M, lateral=1.0)
        _, _, terminated, _, info = wrapped.step(zero)
        assert terminated == should_finish
        assert bool(info["course_complete"]) == should_finish
        assert (info.get("termination_reason") == "course_complete") == should_finish
        # An airborne arrival cannot complete the walk, even past the line.
        env.data.qpos[2] += 0.15
        mujoco.mj_forward(env.model, env.data)
        env._invalidate_substep_aggregates()
        terminated, _ = env._is_terminated()
        assert not terminated


def test_the_course_runs_along_the_spawn_heading():
    """A turned spawn turns the course: the same displacement completes it along, never against or across."""
    kwargs = _forward_walk_kwargs()
    with gym.make(MODEL.gym_id, **kwargs) as wrapped:
        env = wrapped.unwrapped
        wrapped.reset(seed=0)
        yaw = np.radians(60.0)
        env.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        env._reference_heading_2d = env._quat_to_forward_2d(env.data.qpos[3:7])
        np.testing.assert_allclose(env._reference_heading_2d, [np.cos(yaw), np.sin(yaw)], atol=1e-12)
        mujoco.mj_forward(env.model, env.data)
        for along, lateral, finished in ((-env.COURSE_LENGTH_M, 0.0, False), (0.0, env.COURSE_LENGTH_M, False)):
            _carry_along_course(env, along, lateral)
            env._invalidate_substep_aggregates()
            assert env._is_terminated()[0] == finished
        _carry_along_course(env, env.COURSE_LENGTH_M)
        env._invalidate_substep_aggregates()
        terminated, info = env._is_terminated()
        assert terminated and info["termination_reason"] == "course_complete" and info["success"]


def test_a_turned_spawn_moves_the_course_line_with_it():
    """The course line lies COURSE_LENGTH_M less COURSE_TOLERANCE_M (7.65 m) along the heading reset() took from
    the spawn, measured from the spawn's own position (policy interface r2): a home keyframe turned by 120 degrees
    and moved off the origin turns and moves the line.  A millimetre short of it nothing ends at any lateral
    offset, a millimetre past it the supported walk completes, and the point 8 m along world +x, where the r1
    target sat for an unturned spawn at the origin, completes nothing."""
    kwargs = _forward_walk_kwargs()
    with gym.make(MODEL.gym_id, **kwargs) as wrapped:
        env = wrapped.unwrapped
        key = env._reset_keyframe_id
        yaw = np.radians(120.0)
        turned = np.empty(4)
        mujoco.mju_mulQuat(turned, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), env.model.key_qpos[key, 3:7])
        env.model.key_qpos[key, 3:7] = turned
        env.model.key_qpos[key, 0:2] = (2.0, -1.0)
        wrapped.reset(seed=0)
        np.testing.assert_allclose(env._reference_heading_2d, [np.cos(yaw), np.sin(yaw)], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(env._initial_pos_2d, [2.0, -1.0])
        line = env.COURSE_LENGTH_M - env.COURSE_TOLERANCE_M
        assert line == pytest.approx(7.65)
        for along, lateral, finished in (
            (line - 1e-3, 0.0, False),
            (line - 1e-3, 3.0, False),
            (line + 1e-3, -3.0, True),
            (line + 1e-3, 0.0, True),
        ):
            _carry_along_course(env, along, lateral)
            env._invalidate_substep_aggregates()
            terminated, info = env._is_terminated()
            assert terminated == finished, (along, lateral)
            assert (info.get("termination_reason") == "course_complete") == finished
        env.data.qpos[0:2] = (8.0, 0.0)
        mujoco.mj_forward(env.model, env.data)
        env._invalidate_substep_aggregates()
        assert not env._is_terminated()[0]
        # Through step(): half a metre past the line the episode ends as a success.
        _carry_along_course(env, line + 0.5)
        _, _, terminated, _, info = wrapped.step(np.zeros(env.model.nu, dtype=np.float32))
        assert terminated and info["course_complete"] == 1.0 and info["is_success"]
        assert info["termination_reason"] == "course_complete"
