"""Physical and task regressions for both Compsognathus Gymnasium variants."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environments.compsognathus_longipes.anatomical.env import CompsognathusLongipesAnatomicalEnv
from environments.compsognathus_longipes.base_env import CompsognathusLongipesEnv
from environments.compsognathus_longipes.robot.env import CompsognathusLongipesRobotEnv
from environments.shared.config import load_all_stages, load_stage_config
from environments.shared.metrics import LocomotionMetrics
from environments.shared.model_id import ModelId
from environments.shared.plant_contract.manifest import _load_environment
from environments.shared.plant_contract.versions import _species_entries
from environments.shared.species_registry import get_species_config

COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")


class _LinearAnatomicalEnv(CompsognathusLongipesEnv):
    """The anatomical plant on the species base's linear residual: ``CompsognathusLongipesEnv`` is the abstract
    species base (decision D-D33), so the embodiment, MJCF and nominal pelvis height come from the anatomical
    class."""

    embodiment = CompsognathusLongipesAnatomicalEnv.embodiment
    MODEL_PATH = CompsognathusLongipesAnatomicalEnv.MODEL_PATH
    nominal_pelvis_height = CompsognathusLongipesAnatomicalEnv.nominal_pelvis_height


LEG_ACTUATORS = [f"{side}_{joint}_act" for side in "rl" for joint in ("hip_pitch", "hip_roll", "knee", "ankle", "toe")]


@pytest.fixture(params=[CompsognathusLongipesAnatomicalEnv, CompsognathusLongipesRobotEnv], ids=["anatomical", "robot"])
def env(request):
    instance = request.param(reset_noise_scale=0)
    yield instance
    instance.close()


def test_gymnasium_contract(env):
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize(
    "gym_id", ["MesozoicLabs/CompsognathusLongipes-Anatomical-v0", "MesozoicLabs/CompsognathusLongipes-Robot-v0"]
)
def test_registered_horizon_is_configurable(gym_id):
    with gym.make(gym_id, max_episode_steps=7, reset_noise_scale=0) as instance:
        instance.reset(seed=4)
        for step in range(7):
            _, _, terminated, truncated, info = instance.step(np.zeros(instance.action_space.shape))
            assert not terminated
            assert truncated == (step == 6)
        # No success event (SUCCESS_KEYS is empty); the wrapper's horizon, not the env's, ended this episode, and
        # the env reports is_success at its own episode end only (BaseDinoEnv.step).
        assert not info.get("is_success", False)


@pytest.mark.parametrize(
    "model_id", [COMPSOGNATHUS_LONGIPES, COMPSOGNATHUS_LONGIPES_ROBOT], ids=lambda m: f"{m.species}-{m.embodiment}"
)
def test_every_stage_resets_and_steps_across_seeds(model_id):
    config = get_species_config(model_id)
    for stage in load_all_stages(model_id).values():
        with config.env_class(**stage["env_kwargs"]) as instance:
            for seed in range(10):
                first, _ = instance.reset(seed=seed)
                repeat, _ = instance.reset(seed=seed)
                np.testing.assert_array_equal(first, repeat)
                obs, reward, terminated, truncated, info = instance.step(np.zeros(instance.action_space.shape))
                assert instance.observation_space.contains(obs)
                assert np.isfinite(reward)
                assert not terminated and not truncated, (model_id, seed, info)
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
    """Every path that builds the anatomical species gets CompsognathusLongipesAnatomicalEnv (D-D26); the robot keeps the
    linear residual of CompsognathusLongipesEnv, so its policy interface does not move."""
    assert get_species_config(COMPSOGNATHUS_LONGIPES).env_class is CompsognathusLongipesAnatomicalEnv
    assert (
        _load_environment(_species_entries()[COMPSOGNATHUS_LONGIPES]["env_entrypoint"])
        is CompsognathusLongipesAnatomicalEnv
    )
    with gym.make("MesozoicLabs/CompsognathusLongipes-Anatomical-v0") as instance:
        assert type(instance.unwrapped) is CompsognathusLongipesAnatomicalEnv
    assert get_species_config(COMPSOGNATHUS_LONGIPES_ROBOT).env_class is CompsognathusLongipesRobotEnv
    assert not issubclass(CompsognathusLongipesRobotEnv, CompsognathusLongipesAnatomicalEnv)
    assert CompsognathusLongipesRobotEnv._scale_action is CompsognathusLongipesEnv._scale_action
    assert CompsognathusLongipesRobotEnv.action_mapping == "home-keyframe-residual/v1"
    assert CompsognathusLongipesAnatomicalEnv.action_mapping == "home-keyframe-residual-softcubic/v1"


def test_soft_cubic_leg_residual_keeps_home_and_both_endpoints():
    """b = 0.1 a + 0.9 a**3 on the ten leg servos, then the unchanged home-keyframe span: b(0) = 0 commands the
    gravity-preloaded home ctrl, b(+-1) = +-1 the ctrlrange ends, the slope at home is a tenth of the linear
    map's and the map is monotone.  Neck, jaw and tail keep the linear residual."""
    with (
        CompsognathusLongipesAnatomicalEnv(reset_noise_scale=0) as soft,
        _LinearAnatomicalEnv(reset_noise_scale=0) as linear,
    ):
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


REMOVED_TARGET_KWARGS = (
    "target_approach_weight",
    "target_reach_bonus",
    "target_radius",
    "target_max_speed",
    "prey_distance_range",
    "prey_lateral_range",
    "reset_height_noise_scale",
)


@pytest.mark.parametrize("env_class", [CompsognathusLongipesAnatomicalEnv, CompsognathusLongipesRobotEnv])
def test_the_species_declares_no_success_key_and_refuses_the_target_kwargs(env_class):
    """Target reaching, the species' only success event, left with the prey (decision D-D59): no success key, and
    the target's kwargs and ranges (and the height-jitter knob, whose draw left the reset in the same revision) are
    refused rather than silently ignored."""
    assert env_class.SUCCESS_KEYS == ()
    for kwarg in REMOVED_TARGET_KWARGS:
        with pytest.raises(TypeError, match=kwarg):
            env_class(**{kwarg: 0.0})


def test_the_scene_has_no_target_and_the_task_reports_none(env):
    model = env.model
    assert model.nmocap == 0
    for kind, name in ((mujoco.mjtObj.mjOBJ_BODY, "prey"), (mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")):
        assert mujoco.mj_name2id(model, kind, name) == -1
    for attribute in ("prey_id", "_target_mocap_id", "_initial_prey_dir_2d", "_prev_prey_distance"):
        assert not hasattr(env, attribute), attribute
    env.reset(seed=5)
    _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))
    assert not terminated
    for key in ("reward_approach", "reward_target", "prey_distance", "target_success", "is_success"):
        assert key not in info, key
    # A posed state at the spawn spot, at rest and upright, no longer ends the episode as a success.
    env.reset(seed=5)
    assert env._is_terminated() == (False, {"success": False})


def test_the_forward_reference_is_the_spawn_heading(env):
    """Forward, lateral and heading read _reference_heading_2d, the spawn's own forward (BaseDinoEnv.reset)."""
    env.reset(seed=3042)
    np.testing.assert_array_equal(env._reference_heading_2d, env._quat_to_forward_2d(env.data.qpos[3:7]))
    np.testing.assert_allclose(env._reference_heading_2d, [1.0, 0.0], atol=1e-12)
    env.forward_vel_weight = 1.0
    env.lateral_penalty_weight = 1.0
    env.heading_weight = 1.0
    env._reference_heading_2d = np.array([0.0, 1.0])
    zero = np.zeros(env.action_space.shape)
    env.data.qvel[0:2] = (0.0, 0.1)
    _, along = env._get_reward_info(zero)
    env.data.qvel[0:2] = (0.1, 0.0)
    _, across = env._get_reward_info(zero)
    assert along["forward_vel"] == pytest.approx(0.1) and along["reward_lateral"] == pytest.approx(0.0, abs=1e-12)
    assert across["forward_vel"] == pytest.approx(0.0, abs=1e-12) and across["reward_lateral"] == pytest.approx(-0.01)
    # The spawn faces +x, a quarter turn from the reference: no heading credit.
    assert along["heading_alignment"] == pytest.approx(0.0, abs=1e-3)


def test_the_robot_foot_channels_read_the_floor_not_a_sole_resting_on_the_other_foot():
    """The robot's two foot channels are each sole's normal force against the floor (FOOT_OBSERVATION_FLOOR_GEOMS,
    policy interface r4).  Posed with its right sole on the left sole, the right channel reads nothing and the left
    only the left sole's floor load, where the touch sensors count the sole-on-sole force on both feet.  The touch
    sensors still feed the reward's support flag (_aggregated_foot_contact_forces)."""
    with CompsognathusLongipesRobotEnv(reset_noise_scale=0.0) as env:
        env.reset(seed=0)
        model, data = env.model, env.data
        assert env._foot_observation_floor_geoms == ((model.geom("right_sole").id,), (model.geom("left_sole").id,))
        # The right leg rolled inward, hip and ankle roll together so the sole stays level, and lifted onto the left
        # sole; every joint stays inside its range.
        for joint, delta in (
            ("right_hip_roll", -0.25),
            ("right_ankle_roll", -0.25),
            ("right_knee", -0.275),
            ("right_hip_pitch", 0.1375),
            ("right_ankle", 0.1375),
        ):
            data.qpos[model.jnt_qposadr[model.joint(joint).id]] += delta
        mujoco.mj_forward(model, data)
        pairs = {
            tuple(sorted((model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name)))
            for contact in data.contact[: data.ncon]
        }
        assert pairs == {("floor", "left_sole"), ("left_sole", "right_sole")}
        touch = np.array(env._foot_contact_forces())
        channels = env._get_obs()[-5:-3]
        np.testing.assert_array_equal(channels, np.array(env._foot_floor_contact_forces(), dtype=np.float32))
        assert channels[0] == 0.0 and touch[0] > 1.0
        assert 1.0 < channels[1] < touch[1]
        assert touch[1] == pytest.approx(channels[1] + touch[0], rel=1e-5)


def test_on_the_statue_the_robot_floor_channels_are_its_touch_readings():
    """Where neither sole rests on the other foot the floor reading is the touch reading, so the r4 switch moved
    the standing robot's observation nowhere: over the whole 1000-step zero-action stance episode of seeds 3042 and
    3043 every step's foot channels are the soles' floor force and equal the touch sums (measured 0.0 apart,
    float64)."""
    with CompsognathusLongipesRobotEnv(
        **load_stage_config(COMPSOGNATHUS_LONGIPES_ROBOT, "stance")["env_kwargs"]
    ) as env:
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        for seed in (3042, 3043):
            env.reset(seed=seed)
            truncated = False
            steps = 0
            while not truncated:
                obs, _, terminated, truncated, _ = env.step(zero)
                steps += 1
                assert not terminated, (seed, steps)
                floor = np.array(env._foot_floor_contact_forces())
                np.testing.assert_array_equal(obs[-5:-3], floor.astype(np.float32))
                np.testing.assert_allclose(floor, env._foot_contact_forces(), rtol=1e-12, atol=0.0)
                assert np.all(floor > 1.0)
            assert steps == env.max_episode_steps == 1000


def test_the_anatomical_foot_channels_are_its_touch_sensors():
    with CompsognathusLongipesAnatomicalEnv() as env:
        assert env._foot_observation_floor_geoms is None
        env.reset(seed=3042)
        for _ in range(20):
            obs, *_ = env.step(np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(obs[-5:-3], np.array(env._foot_contact_forces(), dtype=np.float32))


def test_no_success_event_reaches_locomotion_metrics():
    """The model's success keys drive LocomotionMetrics (decision D-D59's amendment): none, so no success rate."""
    metrics = LocomotionMetrics(success_keys=tuple(CompsognathusLongipesRobotEnv.SUCCESS_KEYS))
    metrics.record_step({"target_success": 1, "forward_vel": 0.0})
    assert "success_rate" not in metrics.compute()


def test_robot_training_does_not_add_head_or_tail_motors():
    with CompsognathusLongipesRobotEnv() as instance:
        assert instance.model.nu == 12
        for name in ("fixed_head", "passive_tail"):
            assert instance.model.body(name).jntnum[0] == 0
