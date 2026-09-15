"""Physical and temporal integration checks for the opt-in behavior environment."""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from environments.shared.direction_commands import DirectionCommandConfig, gaussian_tracking_reward
from environments.shared.plant_contract import current_plant_identity, validate_compiled_plant
from environments.shared.terrain import TerrainConfig, TerrainRealization, apply_terrain
from environments.trex.envs.behavior_env import TRexBehaviorEnv
from environments.trex.envs.trex_env import TRexEnv


@pytest.fixture
def make_env():
    instances = []

    def factory(**kwargs):
        env = TRexBehaviorEnv(**kwargs)
        instances.append(env)
        return env

    yield factory
    for env in instances:
        env.close()


def small_terrain(**kwargs):
    # 0.3 m samples retain >= four samples at the default 1.5 m wavelength.
    return TerrainConfig(extent=12.0, nrow=81, ncol=81, **kwargs)


def test_seeded_replay_repeats_pose_surface_commands_and_rollout(make_env):
    commands = DirectionCommandConfig(speed_range=(0.7, 1.1), switch_interval_s=0.02, straight_probability=0.0)
    env = make_env(terrain=small_terrain(), commands=commands, run_seed=17, reset_noise_scale=0.05)
    actions = np.random.default_rng(912).uniform(-0.01, 0.01, (8, env.model.nu))

    def replay():
        obs, reset_info = env.reset(seed=1042)
        initial = (obs.copy(), env.data.qpos.copy(), env.data.qvel.copy(), env.model.hfield_data.copy(), reset_info)
        rows = []
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append((obs.copy(), reward, terminated, truncated, info["command_event_id"], info["desired_heading"]))
        return initial, rows

    first, second = replay(), replay()
    for a, b in zip(first[0][:-1], second[0][:-1], strict=True):
        np.testing.assert_array_equal(a, b)
    assert first[0][-1] == second[0][-1]
    for a, b in zip(first[1], second[1], strict=True):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
    env.reset(seed=1043)
    assert not np.array_equal(env.data.qpos, first[0][1])
    assert not np.array_equal(env.model.hfield_data, first[0][3])


def test_different_run_seeds_change_surface_without_perturbing_pose_stream(make_env):
    left = make_env(terrain=small_terrain(), run_seed=42, reset_noise_scale=0.05)
    right = make_env(terrain=small_terrain(), run_seed=44, reset_noise_scale=0.05)
    left.reset(seed=2042)
    right.reset(seed=2042)
    np.testing.assert_array_equal(left.data.qpos, right.data.qpos)
    np.testing.assert_array_equal(left.data.qvel, right.data.qvel)
    assert not np.array_equal(left.model.hfield_data, right.model.hfield_data)
    first_surface = left.model.hfield_data.copy()
    _, info = left.reset()
    assert info["episode_index"] == 1
    assert not np.array_equal(first_surface, left.model.hfield_data)


def test_hfield_spawn_uses_canonical_pose_and_does_not_inject_motion(make_env):
    env = make_env(terrain=small_terrain(), reset_noise_scale=0.05)
    canonical = TRexEnv(reset_noise_scale=0.05)
    try:
        for seed in (42, 43, 44, 1042):
            env.reset(seed=seed)
            canonical.reset(seed=seed)
            np.testing.assert_allclose(env.data.qpos, canonical.data.qpos, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(env.data.qvel, canonical.data.qvel)
            assert env.model.nhfield == 1
            assert not np.any(env.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            geoms = env._root_subtree_geoms()
            floor_heights = env.terrain.height_at(env.data.geom_xpos[geoms, 0], env.data.geom_xpos[geoms, 1])
            np.testing.assert_array_equal(floor_heights, 0.0)
            contacts = env.data.contact
            floor_hits = np.any(contacts.geom == env.floor_geom_id, axis=1)
            assert np.all(contacts.dist[floor_hits] >= env.home_ground_clearance() - 0.002)
            assert not env._is_terminated()[0]
            for _ in range(10):
                _, reward, terminated, _, _ = env.step(np.zeros(env.model.nu))
                assert np.isfinite(reward)
                assert np.all(np.isfinite(env.data.qpos))
                assert not terminated
        for field in (
            "timestep",
            "gravity",
            "solver",
            "integrator",
            "iterations",
            "ls_iterations",
            "tolerance",
            "disableflags",
            "enableflags",
            "cone",
            "impratio",
        ):
            np.testing.assert_array_equal(getattr(env.model.opt, field), getattr(canonical.model.opt, field))
    finally:
        canonical.close()


def _translate_surface_and_animal(env, height):
    original_z = env.data.qpos[2]
    realization = TerrainRealization(
        env.terrain_config,
        999,
        0,
        np.full((env.terrain_config.nrow, env.terrain_config.ncol), height),
    )
    env.terrain = realization
    apply_terrain(env.model, realization, env.data)
    apply_terrain(env._probe_model, realization, env._probe_data)
    shift = realization.height_at(0.0, 0.0)
    env.data.qpos[2] = original_z + shift
    mujoco.mj_forward(env.model, env.data)
    return shift


def test_clearance_rewards_and_safety_follow_translated_surface(make_env):
    env = make_env(
        terrain=small_terrain(),
        reset_noise_scale=0.0,
        height_weight=0.3,
        height_target_tolerance=0.1,
        head_clearance_weight=0.35,
        head_clearance_target=1.0,
    )
    env.reset(seed=1)
    _, original = env._get_reward_info(np.zeros(env.model.nu))
    original_clearances = env._current_clearances().copy()
    shift = _translate_surface_and_animal(env, 0.8)
    assert env.data.qpos[2] > env.healthy_z_range[1]  # Absolute height would incorrectly terminate.
    _, translated = env._get_reward_info(np.zeros(env.model.nu))
    np.testing.assert_allclose(env._current_clearances(), original_clearances, atol=1e-12, rtol=0)
    for key in ("reward_height", "reward_head_clearance", "height_quality", "height_error"):
        assert translated[key] == pytest.approx(original[key], abs=1e-12)
    assert not env._is_terminated()[0]
    env.data.qpos[2] = shift + env.healthy_z_range[0] - 0.01
    mujoco.mj_forward(env.model, env.data)
    assert env.data.qpos[2] > env.healthy_z_range[0]
    terminated, info = env._is_terminated()
    assert terminated
    assert info["pelvis_clearance"] < env.healthy_z_range[0]
    assert info["termination_reason"] == "fallen"


def test_substep_clearance_probe_latches_a_transient_low_body(make_env):
    env = make_env(terrain=small_terrain(), reset_noise_scale=0.0)
    env.reset(seed=1)
    standing_z = env.data.qpos[2]
    env.data.qpos[2] = env.healthy_z_range[0] - 0.01
    mujoco.mj_forward(env.model, env.data)
    env._probe_ground_clearance()
    env.data.qpos[2] = standing_z
    mujoco.mj_forward(env.model, env.data)
    env._probe_ground_clearance()
    assert env._current_clearances()[0] > env.healthy_z_range[0]
    terminated, info = env._is_terminated()
    assert terminated and info["termination_reason"] == "fallen"


@pytest.mark.parametrize("on_terrain", [False, True])
def test_neck_probe_detects_penetration_without_adding_physical_support(make_env, on_terrain):
    env = make_env(terrain=small_terrain() if on_terrain else None, reset_noise_scale=0.0)
    env.reset(seed=1)
    neck = env._neck_geom_id
    assert env.model.geom_contype[neck] == 0
    assert env.model.geom_conaffinity[neck] == 0
    env._probe_ground_clearance()
    assert not env._neck_hit
    rotation = env.data.geom_xmat[neck].reshape(3, 3)
    lowest_z = env.data.geom_xpos[neck, 2] - abs(rotation[2, 2]) * env.model.geom_size[neck, 1]
    lowest_z -= env.model.geom_size[neck, 0]
    env.data.qpos[2] -= lowest_z + 0.03
    mujoco.mj_forward(env.model, env.data)
    env._probe_ground_clearance()
    assert env._neck_hit
    assert not np.any(env.data.contact.geom == neck)
    assert env.model.geom_contype[neck] == 0
    assert env.model.geom_conaffinity[neck] == 0
    assert env._probe_model.geom_contype[neck] == 1


def test_behavior_scene_leaves_canonical_model_and_reset_contract_unchanged(make_env):
    canonical = TRexEnv(reset_noise_scale=0.05)
    identity = current_plant_identity("trex")
    try:
        first, _ = canonical.reset(seed=1042)
        parameters = {
            name: getattr(canonical.model, name).copy()
            for name in (
                "body_mass",
                "body_inertia",
                "geom_type",
                "geom_contype",
                "geom_conaffinity",
                "actuator_gainprm",
            )
        }
        env = make_env(terrain=small_terrain(), reset_noise_scale=0.05)
        env.reset(seed=1042)
        env.step(np.zeros(env.model.nu))
        validate_compiled_plant(canonical.model, identity)
        for name, value in parameters.items():
            np.testing.assert_array_equal(getattr(canonical.model, name), value)
        repeated, _ = canonical.reset(seed=1042)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(repeated[-3:], 0.0)
        assert canonical.model.nhfield == 0
        assert canonical._substep_probe_hook is None
    finally:
        canonical.close()


def test_switch_rewards_executed_command_and_exposes_next_command(make_env):
    commands = DirectionCommandConfig(
        switch_interval_s=0.02,
        straight_probability=0.0,
        turn_increment_max=0.6,
    )
    env = make_env(commands=commands, reset_noise_scale=0.0)
    observation, _ = env.reset(seed=123)
    executed = env._command_state
    assert executed.event_id == 0
    for _ in range(5):
        np.testing.assert_array_equal(observation[-3:], executed.normalized)
        observation, reward, terminated, truncated, info = env.step(np.zeros(env.model.nu))
        assert np.isfinite(reward) and not terminated and not truncated
        assert info["command_event_id"] == executed.event_id
        assert info["desired_heading"] == executed.desired_heading
        assert info["command_v_x"] == float(executed.physical[0])
        assert info["command_yaw_rate"] == float(executed.physical[2])
        assert info["command_time_s"] == executed.time_s
        assert info["actual_heading"] == pytest.approx(env._heading())
        expected_reward = env.tracking_weight * gaussian_tracking_reward(
            info["tracking_error_v"], info["tracking_error_yaw"]
        )
        assert info["reward_tracking"] == pytest.approx(expected_reward)
        executed = env._command_state
        np.testing.assert_array_equal(observation[-3:], executed.normalized)
    assert executed.event_id == 2
    assert len(env.direction_controller.events) == 3


def test_external_direction_takes_effect_on_next_action_and_masks_stops(make_env):
    env = make_env(commands=DirectionCommandConfig(switch_interval_s=0.02), reset_noise_scale=0.0)
    env.reset(seed=17)
    target_heading = env._heading() + np.pi / 4
    observation = env.set_direction(target_heading, 0.9)
    np.testing.assert_array_equal(observation[-3:], env._command_state.normalized)
    assert observation[-1] > 0.0
    external_event = env._command_state.event_id
    for _ in range(5):
        _, _, _, _, info = env.step(np.zeros(env.model.nu))
        assert info["command_event_id"] == external_event
        assert info["command_target_source"] == "external"
        assert info["desired_heading"] == pytest.approx(target_heading)
    observation = env.set_direction(target_heading, 0.0)
    np.testing.assert_array_equal(observation[-3:], 0.0)
    _, _, _, _, info = env.step(np.zeros(env.model.nu))
    assert not info["tracking_heading_active"]
    assert info["tracking_error_heading"] == 0.0
    assert info["desired_speed"] == 0.0


def test_identity_binds_task_options_but_not_episode_draws(make_env):
    env = make_env(terrain=small_terrain(), run_seed=42)
    first = copy.deepcopy(env.behavior_identity)
    env.reset(seed=1)
    env.step(np.zeros(env.model.nu))
    env.reset(seed=2)
    assert env.behavior_identity == first
    changed = make_env(terrain=small_terrain(max_slope_degrees=1.0), run_seed=44)
    assert changed.behavior_identity != first


@pytest.mark.parametrize("flat_probability", [0.0, 1.0])
def test_flat_retention_uses_plane_and_flat_terrain_keeps_hfield(make_env, flat_probability):
    env = make_env(terrain=small_terrain(mode="flat"), flat_probability=flat_probability, reset_noise_scale=0.0)
    observation, info = env.reset(seed=1042)
    is_plane = flat_probability == 1.0
    expected_type = mujoco.mjtGeom.mjGEOM_PLANE if is_plane else mujoco.mjtGeom.mjGEOM_HFIELD
    expected_model = env._plane_model if is_plane else env._terrain_model
    expected_data = env._plane_data if is_plane else env._terrain_data
    expected_probe_model = env._plane_probe_model if is_plane else env._terrain_probe_model
    expected_probe_data = env._plane_probe_data if is_plane else env._terrain_probe_data
    assert env.model is expected_model and env.data is expected_data
    assert env._probe_model is expected_probe_model and env._probe_data is expected_probe_data
    assert env.model.geom_type[env.floor_geom_id] == expected_type
    assert env._probe_model.geom_type[env.floor_geom_id] == expected_type
    assert env.model.nhfield == (0 if is_plane else 1)
    assert env._probe_model.nhfield == env.model.nhfield
    assert (env.terrain is None) == is_plane
    if is_plane:
        assert info["terrain"] == {"family": "flat_plane"}
        assert env.lowest_ground_clearance() == pytest.approx(env.home_ground_clearance(), abs=1e-12)
    else:
        assert env.terrain.config.mode == "flat"
        np.testing.assert_array_equal(env.terrain.heights, 0.0)
        with pytest.raises(NotImplementedError, match="heightfields"):
            env.lowest_ground_clearance()
    assert observation.shape == (64,)
    for _ in range(5):
        observation, reward, terminated, truncated, _ = env.step(np.zeros(15))
        assert np.all(np.isfinite(observation)) and np.isfinite(reward)
        assert not terminated and not truncated


def _mixed_surface_seeds(env):
    seeds = {}
    for seed in range(32):
        env.reset(seed=seed)
        seeds.setdefault(env.terrain is None, seed)
        if len(seeds) == 2:
            return seeds
    pytest.fail("Mixed retention did not produce both plane and heightfield episodes")


def test_mixed_resets_reuse_compiled_pairs_and_replay_both_surfaces(make_env):
    env = make_env(terrain=small_terrain(), flat_probability=0.5, run_seed=17, reset_noise_scale=0.05)
    pairs = {
        True: (env._plane_model, env._plane_data, env._plane_probe_model, env._plane_probe_data),
        False: (env._terrain_model, env._terrain_data, env._terrain_probe_model, env._terrain_probe_data),
    }
    assert pairs[True][0] is not pairs[False][0]
    assert pairs[True][1] is not pairs[False][1]
    assert pairs[True][2] is not pairs[False][2]
    assert pairs[True][3] is not pairs[False][3]
    geom_types = {key: (pair[0].geom_type.copy(), pair[2].geom_type.copy()) for key, pair in pairs.items()}
    identity = copy.deepcopy(env.behavior_identity)
    actions = np.random.default_rng(921).uniform(-0.01, 0.01, (3, 15))

    def replay():
        rows = []
        for episode_index in range(12):
            observation, info = env.reset(seed=1042 if episode_index == 0 else None)
            is_plane = env.terrain is None
            pair = pairs[is_plane]
            assert env.model is pair[0] and env.data is pair[1]
            assert env._probe_model is pair[2] and env._probe_data is pair[3]
            assert info["episode_index"] == episode_index
            initial = (
                observation.copy(),
                env.data.qpos.copy(),
                env.data.qvel.copy(),
                env.model.hfield_data.copy(),
            )
            rollout = []
            for action in actions:
                obs, reward, terminated, truncated, step_info = env.step(action)
                rollout.append((obs.copy(), reward, terminated, truncated, step_info["desired_heading"]))
            rows.append((is_plane, info, initial, rollout))
            for key, models in pairs.items():
                np.testing.assert_array_equal(models[0].geom_type, geom_types[key][0])
                np.testing.assert_array_equal(models[2].geom_type, geom_types[key][1])
        return rows

    first, second = replay(), replay()
    assert {row[0] for row in first} == {False, True}
    assert sum(a[0] != b[0] for a, b in zip(first, first[1:])) >= 2
    for a, b in zip(first, second, strict=True):
        assert a[:2] == b[:2]
        for first_array, second_array in zip(a[2], b[2], strict=True):
            np.testing.assert_array_equal(first_array, second_array)
        for first_step, second_step in zip(a[3], b[3], strict=True):
            np.testing.assert_array_equal(first_step[0], second_step[0])
            assert first_step[1:] == second_step[1:]
    assert env.behavior_identity == identity


def test_switching_surfaces_clears_reused_physics_and_contact_caches(make_env):
    kwargs = dict(terrain=small_terrain(), flat_probability=0.5, run_seed=17, reset_noise_scale=0.05)
    env, fresh = make_env(**kwargs), make_env(**kwargs)
    seeds = _mixed_surface_seeds(env)
    for is_plane in (False, True, False):
        env.reset(seed=seeds[not is_plane])
        target_data = env._plane_data if is_plane else env._terrain_data
        target_probe = env._plane_probe_data if is_plane else env._terrain_probe_data
        for data in (target_data, target_probe):
            data.qpos[2] = -10.0
            data.qvel[:] = 37.0
            data.ctrl[:] = 19.0
            data.qacc_warmstart[:] = 23.0
            data.xfrc_applied[:] = 7.0
            data.qfrc_applied[:] = 9.0
            data.time = 123.0
        env._clearance_min = np.full(3, -99.0)
        env._neck_hit = True
        env._substep_min_foot_forces = np.full(2, -99.0)
        env._substep_floor_hit_geom = env.torso_geom_id
        env._substep_min_heights = np.full(2, -99.0)
        env._substep_contact_step = 0
        env._root_subtree_geom_ids = np.empty(0, dtype=np.int32)
        env._static_floor_geom_ids = np.empty(0, dtype=np.int32)
        env._ground_geom_array = np.empty(0, dtype=np.int32)
        observation, info = env.reset(seed=seeds[is_plane])
        fresh_observation, fresh_info = fresh.reset(seed=seeds[is_plane])
        np.testing.assert_array_equal(observation, fresh_observation)
        assert info == fresh_info
        assert env._clearance_min is None and not env._neck_hit
        assert env._substep_min_foot_forces is None
        assert env._substep_floor_hit_geom is None
        assert env._substep_min_heights is None
        assert env._substep_contact_step == -1
        assert env._ground_geom_array is None
        np.testing.assert_array_equal(env._root_subtree_geoms(), fresh._root_subtree_geoms())
        np.testing.assert_array_equal(env._static_floor_geoms(), fresh._static_floor_geoms())
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "xfrc_applied", "qfrc_applied", "sensordata"):
            np.testing.assert_array_equal(getattr(env.data, name), getattr(fresh.data, name))
        assert env.data.time == 0.0 and env._probe_data.time == 0.0
        for name in ("qvel", "xfrc_applied", "qfrc_applied"):
            np.testing.assert_array_equal(getattr(env._probe_data, name), 0.0)
        env._probe_ground_clearance()
        fresh._probe_ground_clearance()
        np.testing.assert_array_equal(env._current_clearances(), fresh._current_clearances())
        np.testing.assert_array_equal(env._probe_data.contact.geom, fresh._probe_data.contact.geom)
        np.testing.assert_array_equal(env._probe_data.contact.dist, fresh._probe_data.contact.dist)
        assert not env._is_terminated()[0]
        for _ in range(3):
            actual = env.step(np.zeros(15))
            expected = fresh.step(np.zeros(15))
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:4] == expected[1:4]


def test_model_switch_closes_rendering_resources_bound_to_old_model(make_env):
    from unittest.mock import Mock

    env = make_env(terrain=small_terrain(), flat_probability=0.5)
    seeds = _mixed_surface_seeds(env)
    env.reset(seed=seeds[True])
    for is_plane in (False, True):
        renderer, viewer = Mock(), Mock()
        env._renderer, env._viewer = renderer, viewer
        env.reset(seed=seeds[is_plane])
        renderer.close.assert_called_once_with()
        viewer.close.assert_called_once_with()
        viewer.update_hfield.assert_not_called()
        assert env._renderer is None and env._viewer is None


def test_plane_retention_pool_does_not_modify_independent_canonical_environment(make_env):
    canonical = TRexEnv(reset_noise_scale=0.05)
    try:
        canonical_observation, _ = canonical.reset(seed=1042)
        canonical_qpos = canonical.data.qpos.copy()
        canonical_qvel = canonical.data.qvel.copy()
        arrays = {
            name: getattr(canonical.model, name).copy()
            for name in ("geom_type", "geom_contype", "geom_conaffinity", "body_mass", "body_inertia")
        }
        env = make_env(terrain=small_terrain(), flat_probability=0.5, reset_noise_scale=0.05)
        seeds = _mixed_surface_seeds(env)
        for is_plane in (True, False, True):
            env.reset(seed=seeds[is_plane])
            assert env.model is not canonical.model and env.data is not canonical.data
            assert env.model.geom_contype[env.prey_geom_id] == 0
            assert env.model.geom_conaffinity[env.prey_geom_id] == 0
            for _ in range(3):
                env.step(np.zeros(15))
            validate_compiled_plant(canonical.model, current_plant_identity("trex"))
            np.testing.assert_array_equal(canonical.data.qpos, canonical_qpos)
            np.testing.assert_array_equal(canonical.data.qvel, canonical_qvel)
            for name, value in arrays.items():
                np.testing.assert_array_equal(getattr(canonical.model, name), value)
        repeated, _ = canonical.reset(seed=1042)
        np.testing.assert_array_equal(repeated, canonical_observation)
        assert canonical.model.nhfield == 0
        assert canonical._substep_probe_hook is None
    finally:
        canonical.close()
