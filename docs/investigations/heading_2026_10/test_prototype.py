"""Independent symmetry checks for the design prototype, not production gates."""

import math

import mujoco
import numpy as np
import pytest

from environments.compsognathus.envs.compsognathus_env import CompsognathusBiologicalEnv
from environments.shared.config import load_stage_config
from environments.shared.reporting.stance_report import SpawnYaw, _apply_spawn_yaw
from environments.trex.envs.trex_env import TRexEnv
from prototype import rotation, transform_observation


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


def test_invalid_and_vertical_frames_refuse_explicitly():
    with pytest.raises(ValueError):
        rotation(np.zeros(4))
    obs = np.zeros(22)
    obs[:4] = [math.sqrt(0.5), 0, math.sqrt(0.5), 0]
    with pytest.raises(ValueError, match="vertical"):
        transform_observation(obs, 0)
