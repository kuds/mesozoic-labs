"""Physical and task regressions for both Compsognathus Gymnasium variants."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environments.compsognathus.envs import CompsognathusEnv, CompsognathusRobotEnv
from environments.shared.config import load_all_stages
from environments.shared.metrics import LocomotionMetrics
from environments.shared.scripts.sweep import build_search_space, resolve_config_path
from environments.shared.species_registry import get_species_config


@pytest.fixture(params=[CompsognathusEnv, CompsognathusRobotEnv], ids=["anatomical", "robot"])
def env(request):
    instance = request.param(reset_noise_scale=0)
    yield instance
    instance.close()


def test_gymnasium_contract(env):
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("gym_id", ["MesozoicLabs/Compsognathus-v0", "MesozoicLabs/CompsognathusRobot-v0"])
def test_registered_horizon_is_configurable(gym_id):
    with gym.make(gym_id, max_episode_steps=7, reset_noise_scale=0) as instance:
        instance.reset(seed=4)
        for step in range(7):
            _, _, terminated, truncated, info = instance.step(np.zeros(instance.action_space.shape))
            assert not terminated
            assert truncated == (step == 6)
        assert not info["is_success"]


@pytest.mark.parametrize("species", ["compsognathus", "compsognathus_robot"])
def test_every_stage_resets_and_steps_across_seeds(species):
    config = get_species_config(species)
    for stage in load_all_stages(species).values():
        with config.env_class(**stage["env_kwargs"]) as instance:
            for seed in range(10):
                first, _ = instance.reset(seed=seed)
                repeat, _ = instance.reset(seed=seed)
                np.testing.assert_array_equal(first, repeat)
                obs, reward, terminated, truncated, info = instance.step(np.zeros(instance.action_space.shape))
                assert instance.observation_space.contains(obs)
                assert np.isfinite(reward)
                assert not terminated and not truncated, (species, seed, info)
                assert reward == info["reward_total"]


def test_quaternion_sensor_tracks_body_axes_instead_of_principal_inertia(env):
    env.reset(seed=3)
    for angle in (0.0, 0.3, -0.5):
        env.data.qpos[3:7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
        mujoco.mj_forward(env.model, env.data)
        measured = env.data.sensor("diagnostic_pelvis_quat").data
        np.testing.assert_allclose(measured, env.data.xquat[env.pelvis_id], atol=1e-12)
        assert env._quat_to_tilt(measured) == pytest.approx(abs(angle), abs=1e-10)


def test_home_residual_holds_full_twenty_second_episode(env):
    env.reset(seed=3)
    zero = np.zeros(env.action_space.shape)
    np.testing.assert_array_equal(env._scale_action(zero), env.model.key("home").ctrl)
    np.testing.assert_allclose(env._scale_action(zero - 1), env.model.actuator_ctrlrange[:, 0])
    np.testing.assert_allclose(env._scale_action(zero + 1), env.model.actuator_ctrlrange[:, 1])
    for step in range(env.max_episode_steps):
        obs, reward, terminated, truncated, info = env.step(zero)
        assert not terminated, (step, info)
        assert np.isfinite(reward) and env.observation_space.contains(obs)
        assert info["tilt_angle"] < 0.03
        assert truncated == (step == env.max_episode_steps - 1)
    assert not info["is_success"]


def test_invalid_actions_are_rejected_before_physics(env):
    env.reset(seed=2)
    for invalid in (np.zeros(env.action_space.shape[0] + 1), np.full(env.action_space.shape, np.nan)):
        before = env.data.time
        with pytest.raises(ValueError, match="finite action"):
            env.step(invalid)
        assert env.data.time == before


def test_target_success_requires_slow_upright_arrival(env):
    env.target_reach_bonus = 25
    env.reset(seed=5)
    env.data.mocap_pos[env._target_mocap_id, :2] = env.data.qpos[:2]
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
    assert terminated and not truncated
    assert info["is_success"] and info["target_success"] == 1
    assert info["reward_target"] == 25

    env.reset(seed=5)
    env.data.mocap_pos[env._target_mocap_id, :2] = env.data.qpos[:2]
    env.data.qvel[0] = 2 * env.target_max_speed
    mujoco.mj_forward(env.model, env.data)
    assert not env._is_terminated()[0]

    # Falling onto the goal cannot turn a failure into a successful episode.
    env.data.qpos[2] = 0.05
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    terminated, info = env._is_terminated()
    assert terminated and not info["success"]
    _, reward_info = env._get_reward_info(np.zeros(env.action_space.shape))
    assert reward_info["target_success"] == 0
    assert reward_info["reward_target"] == 0


def test_target_event_reaches_locomotion_metrics():
    metrics = LocomotionMetrics()
    metrics.record_step({"target_success": 1, "forward_vel": 0.0})
    assert metrics.compute()["success_rate"] == 1


def test_robot_training_does_not_add_head_or_tail_motors():
    with CompsognathusRobotEnv() as instance:
        assert instance.model.nu == 12
        for name in ("fixed_head", "passive_tail"):
            assert instance.model.body(name).jntnum[0] == 0


@pytest.mark.parametrize("species", ["compsognathus", "compsognathus_robot"])
@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_ray_notebook_can_resolve_each_stage_search_space(species, algorithm):
    assert resolve_config_path(algorithm, species).is_file()
    for stage in (1, 2, 3):
        parameters = build_search_space(species, stage, algorithm)
        assert parameters
        assert all(name.startswith(f"{algorithm}_") for name in parameters)
