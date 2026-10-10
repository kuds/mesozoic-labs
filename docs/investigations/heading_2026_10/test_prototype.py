"""Independent symmetry checks for the design prototype, not production gates."""

import math

import mujoco
import numpy as np
import pytest
from prototype import rotation, transform_observation

from environments.compsognathus.envs.compsognathus_env import CompsognathusBiologicalEnv
from environments.shared.config import load_stage_config
from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.direction_commands import DirectionCommandConfig, DirectionCommandController
from environments.shared.reporting.stance_report import SpawnYaw, _apply_spawn_yaw
from environments.trex.envs.trex_env import TRexEnv


@pytest.fixture(params=[("trex", TRexEnv), ("compsognathus", CompsognathusBiologicalEnv)])
def animal(request):
    species, cls = request.param
    env = cls(**load_stage_config(species, "stance")["env_kwargs"])
    yield env
    env.close()


@pytest.mark.parametrize("candidate", ["gravity", "quat"])
@pytest.mark.parametrize("yaw", [-179.9, -90.0, -45.0, 17.0, 90.0, 180.0])
def test_complete_scene_rotation_preserves_candidate(animal, candidate, yaw):
    obs, _ = animal.reset(seed=18042)
    j = animal.model.nq + animal.model.nv - 13
    before = transform_observation(obs, j, candidate=candidate)
    after = transform_observation(_apply_spawn_yaw(animal, SpawnYaw(yaw, True)), j, candidate=candidate)
    np.testing.assert_allclose(before, after, atol=2e-6, rtol=2e-6)


def test_animal_only_turn_retains_changed_target_bearing(animal):
    obs, _ = animal.reset(seed=18042)
    j = animal.model.nq + animal.model.nv - 13
    before = transform_observation(obs, j)
    after = transform_observation(_apply_spawn_yaw(animal, SpawnYaw(90, False)), j)
    assert np.linalg.norm(before[-7:-4] - after[-7:-4]) > 0.5
    np.testing.assert_array_equal(before[-3:], after[-3:])


def test_quaternion_sign_has_no_effect(animal):
    obs, _ = animal.reset(seed=18042)
    j = animal.model.nq + animal.model.nv - 13
    opposite = obs.copy()
    opposite[j : j + 4] *= -1
    for candidate in ["gravity", "quat"]:
        np.testing.assert_array_equal(
            transform_observation(obs, j, candidate=candidate),
            transform_observation(opposite, j, candidate=candidate),
        )


def test_rotation_formula_against_mujoco():
    rng = np.random.default_rng(18042)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        reference = np.empty(9)
        mujoco.mju_quat2Mat(reference, q)
        np.testing.assert_allclose(rotation(q), reference.reshape(3, 3), atol=2e-15)


def test_candidate_preserves_tilt_information():
    q = np.array([math.cos(0.2), math.sin(0.2), 0, 0])
    obs = np.zeros(22)
    obs[:4] = q
    g = transform_observation(obs, 0)[:3]
    assert abs(g[1]) > 0.3
    assert g[2] < -0.9


def test_invalid_quaternion_refuses_explicitly():
    with pytest.raises(ValueError):
        rotation(np.zeros(4))
    with pytest.raises(ValueError):
        transform_observation(np.zeros(22), 0)


@pytest.mark.parametrize("pitch", [-math.pi / 2, math.pi / 2, math.pi / 2 - 1e-10])
@pytest.mark.parametrize("yaw", [-math.pi, -math.pi / 2, 0.73])
def test_vertical_forward_axis_is_finite_and_scene_covariant(pitch, yaw):
    q = np.empty(4)
    mujoco.mju_euler2Quat(q, np.array([0.2, pitch, 0.0]), "zyx")
    obs = np.zeros(22)
    obs[:4] = q
    obs[7:10] = [0.2, -0.4, 0.3]
    obs[15:18] = [0.6, 0.8, 0.0]
    rotated = obs.copy()
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    mujoco.mju_mulQuat(rotated[:4], qz, q)
    world_rotation = rotation(qz)
    rotated[7:10] = world_rotation @ obs[7:10]
    rotated[15:18] = world_rotation @ obs[15:18]
    before = transform_observation(obs, 0)
    after = transform_observation(rotated, 0)
    assert np.isfinite(before).all() and np.isfinite(after).all()
    np.testing.assert_allclose(before, after, atol=2e-6, rtol=2e-6)


def test_root_and_orientation_sensor_agree_after_forward(animal):
    animal.reset(seed=18042)
    j = animal.model.nq + animal.model.nv - 13
    rng = np.random.default_rng(812)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        animal.data.qpos[3:7] = q
        mujoco.mj_forward(animal.model, animal.data)
        obs = animal._get_obs()
        np.testing.assert_allclose(rotation(obs[j : j + 4]), rotation(q), atol=2e-7)


@pytest.mark.parametrize("yaw", [-math.pi, -math.pi / 2, 0.73])
def test_live_command_schedule_requires_rotating_the_requested_heading(yaw):
    config = DirectionCommandConfig(switch_interval_s=0.5, switch_jitter_s=0.1, straight_probability=0.0)
    original = DirectionCommandController(config)
    rotated = DirectionCommandController(config)
    original.reset(np.random.default_rng(991), 0.1)
    rotated.reset(np.random.default_rng(991), 0.1 + yaw)
    for t in np.linspace(0, 5, 101):
        current = 0.1 + 0.2 * math.sin(t)
        a = original.update(t, current)
        b = rotated.update(t, current + yaw)
        np.testing.assert_allclose(a.normalized, b.normalized, atol=2e-7)
    original.set_target(0.7, 0.6)
    rotated.set_target(0.7 + yaw, 0.6)
    np.testing.assert_allclose(original.update(6, 0.1).normalized, rotated.update(6, 0.1 + yaw).normalized)
    # Rotating only the animal changes the task; a probe cannot leave this
    # controller's world request behind and call the states equivalent.
    rotated.set_target(0.7, 0.6)
    assert not np.allclose(original.update(7, 0.1).normalized, rotated.update(7, 0.1 + yaw).normalized)


def test_pooled_rotations_can_hide_failure_of_the_seed_group_bar():
    # Six headings, each with a different failing seed. Every heading looks
    # strong, but six of the forty reset states fail somewhere on the grid.
    clean = np.ones((40, 6), dtype=bool)
    clean[np.arange(6), np.arange(6)] = False
    naive_lcb = binomial_lcb(int(clean.sum()), clean.size)
    group_lcb = binomial_lcb(int(clean.all(axis=1).sum()), len(clean))
    assert naive_lcb > 0.95
    assert group_lcb < 0.80
