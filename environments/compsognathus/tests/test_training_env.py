"""Physical and task regressions for both Compsognathus Gymnasium variants."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environments.compsognathus.envs import CompsognathusBiologicalEnv, CompsognathusEnv, CompsognathusRobotEnv
from environments.shared.config import load_all_stages
from environments.shared.metrics import LocomotionMetrics
from environments.shared.plant_contract.manifest import _load_environment
from environments.shared.plant_contract.versions import _species_entries
from environments.shared.species_registry import get_species_config

LEG_ACTUATORS = [f"{side}_{joint}_act" for side in "rl" for joint in ("hip_pitch", "hip_roll", "knee", "ankle", "toe")]


@pytest.fixture(params=[CompsognathusBiologicalEnv, CompsognathusRobotEnv], ids=["anatomical", "robot"])
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


def test_the_anatomical_species_trains_on_the_soft_cubic_subclass():
    """Every path that builds the anatomical species gets CompsognathusBiologicalEnv (D-D26); the robot keeps the
    linear residual of CompsognathusEnv, so its policy interface does not move."""
    assert get_species_config("compsognathus").env_class is CompsognathusBiologicalEnv
    assert _load_environment(_species_entries()["compsognathus"]["env_entrypoint"]) is CompsognathusBiologicalEnv
    with gym.make("MesozoicLabs/Compsognathus-v0") as instance:
        assert type(instance.unwrapped) is CompsognathusBiologicalEnv
    assert get_species_config("compsognathus_robot").env_class is CompsognathusRobotEnv
    assert not issubclass(CompsognathusRobotEnv, CompsognathusBiologicalEnv)
    assert CompsognathusRobotEnv._scale_action is CompsognathusEnv._scale_action
    assert CompsognathusRobotEnv.action_mapping == "home-keyframe-residual/v1"
    assert CompsognathusBiologicalEnv.action_mapping == "home-keyframe-residual-softcubic/v1"


def test_soft_cubic_leg_residual_keeps_home_and_both_endpoints():
    """b = 0.1 a + 0.9 a**3 on the ten leg servos, then the unchanged home-keyframe span: b(0) = 0 commands the
    gravity-preloaded home ctrl, b(+-1) = +-1 the ctrlrange ends, the slope at home is a tenth of the linear
    map's and the map is monotone.  Neck, jaw and tail keep the linear residual."""
    with CompsognathusBiologicalEnv(reset_noise_scale=0) as soft, CompsognathusEnv(reset_noise_scale=0) as linear:
        model = soft.model
        names = [model.actuator(i).name for i in range(model.nu)]
        assert [name for name, shaped in zip(names, soft._shaped_residual_mask) if shaped] == LEG_ACTUATORS
        legs = soft._shaped_residual_mask
        home = model.key("home").ctrl
        low, high = model.actuator_ctrlrange.T
        zero = np.zeros(model.nu)
        np.testing.assert_array_equal(soft._scale_action(zero), home)
        for end, bound in ((-1.0, low), (1.0, high), (7.0, high)):  # +7: clipped first
            # b(+-1) is exactly +-1, so the ends are the linear map's, which reach the ctrlrange to an ulp.
            np.testing.assert_array_equal(soft._scale_action(zero + end), linear._scale_action(zero + end))
            np.testing.assert_allclose(soft._scale_action(zero + end), bound, rtol=0, atol=1e-15)
        for a in (1e-6, -1e-6):
            ratio = (soft._scale_action(zero + a) - home) / (linear._scale_action(zero + a) - home)
            np.testing.assert_allclose(ratio[legs], 0.1, rtol=1e-9)
            np.testing.assert_allclose(ratio[~legs], 1.0, rtol=1e-12)
        grid = np.linspace(-1.0, 1.0, 401)
        ctrl = np.array([soft._scale_action(zero + a) for a in grid])
        assert np.all(np.diff(ctrl, axis=0) > 0)
        b = np.where(grid[:, None] >= 0, (ctrl - home) / (high - home), (ctrl - home) / (home - low)).T
        np.testing.assert_allclose(b[legs], np.broadcast_to(0.1 * grid + 0.9 * grid**3, b[legs].shape), atol=1e-12)
        np.testing.assert_allclose(b[~legs], np.broadcast_to(grid, b[~legs].shape), atol=1e-12)
        rng = np.random.default_rng(7)
        for action in rng.uniform(-1.0, 1.0, (20, model.nu)):
            np.testing.assert_array_equal(soft._scale_action(action)[~legs], linear._scale_action(action)[~legs])


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
