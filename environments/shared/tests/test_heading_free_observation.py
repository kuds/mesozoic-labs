"""The heading-free observation on its two species (``bipedal-target-heading-free/v1``; plant_versions note 18).

The T-Rex (policy interface r14) and the anatomical Compsognathus (r4) build
their observation through ``environments/shared/heading_observation.py``.
These tests run the real environments (no SB3; the shared legs of the test
matrix) and pin the design's acceptance list
(docs/HEADING_INVARIANCE_DESIGN_2026_10.md, "Migration and acceptance"):

* the vector is laid out exactly as the plant contract's segments say, with
  the command trailing;
* it is finite and correctly sized at construction (before any forward pass,
  where the orientation sensor still reads zero), at reset, at every step and
  at a terminal state;
* the gyro and accelerometer are the sensors' own site-frame readings, not
  rotated a second time, and the root rotation agrees with MuJoCo's after a
  forward pass;
* a whole-scene turn of a lifted, tilted, moving state leaves the float32
  observation unchanged (budget 2e-6), across the yaw wrap; an animal-only
  turn moves the target bearing and nothing else; ``q`` and ``-q`` read the
  same;
* six-heading paired rollouts repeat to solver tolerance: the same seed,
  turned with its whole scene to each of -90, -45, 0, 45, 90 and 180 degrees
  in a fresh environment, steps the same actions, joint trajectory, rewards
  and observations.  Here over 250 steps of seed 3042 under the zero action
  and a fixed deterministic linear policy (about 5 s per species); the
  full-horizon version, with a frozen normalizer, an untrained PPO policy and
  the floor-truth metrics, is ``test_heading_free_rollouts.py`` in the SB3
  job's full depth.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from environments.compsognathus.envs.compsognathus_env import CompsognathusBiologicalEnv
from environments.shared.command_frame import COMMAND_SEGMENT_NAME, COMMAND_WIDTH
from environments.shared.config import load_stage_config
from environments.shared.heading_observation import HEADING_FREE_OBSERVATION_SCHEMA, body_to_world, root_quaternion
from environments.shared.plant_contract import load_plant_versions
from environments.shared.plant_contract.policy_layer import _policy_interface_payload
from environments.shared.reporting.stance_report import SpawnYaw, _apply_spawn_yaw
from environments.trex.envs.trex_env import TRexEnv

from .heading_free_helpers import FLOAT32_BUDGET, assert_paired, linear_policy, paired_traces, zero_policy

SPECIES = (
    pytest.param("trex", TRexEnv, id="trex"),
    pytest.param("compsognathus", CompsognathusBiologicalEnv, id="compsognathus"),
)
#: Whole-scene turns for the pointwise test, across the yaw wrap.
TURNS_DEG = (-179.9, -90.0, -45.0, 17.0, 90.0, 180.0, 359.0)
#: The shared legs' paired-rollout window (about 5 s per species for both policies).
PAIRED_STEPS = 250


def _stance_env(env_class, species: str):
    return env_class(**load_stage_config(species, "stance")["env_kwargs"])


@pytest.fixture(params=[pytest.param(tuple(param.values), id=param.id) for param in SPECIES])
def animal(request):
    species, env_class = request.param
    env = _stance_env(env_class, species)
    yield species, env
    env.close()


def _layout(species: str, env) -> dict[str, slice]:
    """Each observation segment's slice, read off the plant contract's ``observation_segments``."""
    version = load_plant_versions()[1][species]
    assert version.observation_schema == HEADING_FREE_OBSERVATION_SCHEMA
    payload = _policy_interface_payload(env.model, env, version)
    layout: dict[str, slice] = {}
    start = 0
    for segment in payload["observation_segments"]:
        width = len(segment["components"]) if "components" in segment else int(segment["width"])
        layout[segment["name"]] = slice(start, start + width)
        start += width
    assert start == env.observation_space.shape[0]
    return layout


def _target(env) -> np.ndarray:
    return np.asarray(env.data.mocap_pos[int(env.model.body_mocapid[_target_body(env)])])


def _target_body(env) -> int:
    (body,) = np.flatnonzero(np.asarray(env.model.body_mocapid) >= 0)
    return int(body)


def test_layout_matches_segments_and_command_is_last(animal):
    species, env = animal
    obs, _ = env.reset(seed=3042)
    for _ in range(5):
        obs, *_ = env.step(np.full(env.action_space.shape, 0.05, dtype=np.float32))
    layout = _layout(species, env)
    names = list(layout)
    assert names == [
        "joint_position",
        "joint_velocity",
        "pelvis_projected_gravity",
        "pelvis_angular_velocity",
        "pelvis_heading_linear_velocity",
        "pelvis_linear_acceleration",
        "foot_contact",
        "prey_direction_body",
        "prey_distance",
        COMMAND_SEGMENT_NAME,
    ]
    assert obs.shape == env.observation_space.shape == (layout[COMMAND_SEGMENT_NAME].stop,)
    assert layout[COMMAND_SEGMENT_NAME] == slice(obs.shape[0] - COMMAND_WIDTH, obs.shape[0])
    data = env.data
    as32 = np.float32
    np.testing.assert_array_equal(obs[layout["joint_position"]], data.qpos[7:].astype(as32))
    np.testing.assert_array_equal(obs[layout["joint_velocity"]], data.qvel[6:].astype(as32))
    gyro, accel = env._sensor_gyro_start, env._sensor_accel_start
    np.testing.assert_array_equal(obs[layout["pelvis_angular_velocity"]], data.sensordata[gyro : gyro + 3].astype(as32))
    np.testing.assert_array_equal(
        obs[layout["pelvis_linear_acceleration"]], data.sensordata[accel : accel + 3].astype(as32)
    )
    np.testing.assert_array_equal(obs[layout["foot_contact"]], np.asarray(env._foot_contact_forces()).astype(as32))
    distance = np.linalg.norm(_target(env) - data.xpos[env.pelvis_id])
    np.testing.assert_array_equal(obs[layout["prey_distance"]], [as32(distance)])
    np.testing.assert_array_equal(obs[layout[COMMAND_SEGMENT_NAME]], np.zeros(COMMAND_WIDTH))
    # The three heading-free blocks, from MuJoCo's own quaternion-to-matrix map of the free root's qpos[3:7]:
    # gravity is minus the third row of R, the velocity is qvel[0:3] in the gravity-aligned heading frame, the
    # target is in pelvis coordinates.  (After a step qpos is post-integration while xmat and the sensors come
    # from the last substep's forward pass, so xmat lags R by one physics substep, as it lagged qvel[0:3].)
    quaternion = data.qpos[3:7] / np.linalg.norm(data.qpos[3:7])
    flat = np.empty(9)
    mujoco.mju_quat2Mat(flat, quaternion)
    rotation = flat.reshape(3, 3)
    np.testing.assert_allclose(obs[layout["pelvis_projected_gravity"]], -rotation[2], atol=1e-6)
    heading = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(heading), math.sin(heading)
    v = data.qvel[0:3]
    np.testing.assert_allclose(
        obs[layout["pelvis_heading_linear_velocity"]], [c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]], atol=1e-6
    )
    direction = (_target(env) - data.xpos[env.pelvis_id]) / (distance + 1e-8)
    np.testing.assert_allclose(obs[layout["prey_direction_body"]], rotation.T @ direction, atol=1e-6)


@pytest.mark.parametrize(("species", "env_class"), SPECIES)
def test_construction_observation_is_finite_and_upright(species, env_class):
    """The constructor calls _get_obs before any forward pass: the orientation sensor still reads zero there."""
    env = _stance_env(env_class, species)
    try:
        quat = env._sensor_quat_start
        np.testing.assert_array_equal(env.data.sensordata[quat : quat + 4], np.zeros(4))
        obs = env._get_obs()
        assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        gravity = _layout(species, env)["pelvis_projected_gravity"]
        np.testing.assert_array_equal(obs[gravity], [0.0, 0.0, -1.0])
    finally:
        env.close()


def test_reset_step_terminal_observations_are_finite(animal):
    species, env = animal
    obs, _ = env.reset(seed=3042)
    assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
    rng = np.random.default_rng(1)
    terminated = truncated = False
    steps = 0
    info: dict = {}
    while not (terminated or truncated):
        action = np.clip(rng.normal(scale=0.8, size=env.action_space.shape), -1.0, 1.0).astype(np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
        assert np.all(np.isfinite(obs)), f"{species} step {steps}"
    assert terminated, f"{species}: wild actions should end the episode early, so the terminal state is exercised"
    assert steps < env.max_episode_steps and info.get("termination_reason")
    np.testing.assert_array_equal(obs[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH))


def test_local_sensors_are_not_rotated_twice(animal):
    """The gyro and accelerometer report in their own site frame already; a second rotation would re-leak yaw."""
    species, env = animal
    layout = _layout(species, env)
    env.reset(seed=3043)
    rng = np.random.default_rng(5)
    gyro, accel = env._sensor_gyro_start, env._sensor_accel_start
    for _ in range(10):
        obs, *_ = env.step(np.clip(rng.normal(scale=0.2, size=env.action_space.shape), -1, 1).astype(np.float32))
        np.testing.assert_array_equal(
            obs[layout["pelvis_angular_velocity"]], env.data.sensordata[gyro : gyro + 3].astype(np.float32)
        )
        np.testing.assert_array_equal(
            obs[layout["pelvis_linear_acceleration"]], env.data.sensordata[accel : accel + 3].astype(np.float32)
        )


def test_root_and_sensor_rotation_agree_after_forward(animal):
    """After mj_forward the qpos-derived R equals xmat and the framequat sensor's rotation (the site is unrotated)."""
    _, env = animal
    env.reset(seed=3042)
    rng = np.random.default_rng(812)
    quat = env._sensor_quat_start
    for _ in range(20):
        quaternion = rng.normal(size=4)
        env.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        mujoco.mj_forward(env.model, env.data)
        rotation = body_to_world(root_quaternion(env.data.qpos))
        np.testing.assert_allclose(rotation, env.data.xmat[env.pelvis_id].reshape(3, 3), atol=1e-12)
        sensed = np.empty(9)
        mujoco.mju_quat2Mat(sensed, env.data.sensordata[quat : quat + 4])
        np.testing.assert_allclose(rotation, sensed.reshape(3, 3), atol=1e-12)


def _lifted_moving_states(env, count: int):
    """*count* tilted, moving spawn states lifted 1 m clear of the floor (the turn then meets no contact)."""
    rng = np.random.default_rng(18042)
    for index in range(count):
        env.reset(seed=18042 + index)
        quaternion = np.empty(4)
        euler = np.array([rng.uniform(-math.pi, math.pi), rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)])
        mujoco.mju_euler2Quat(quaternion, euler, "zyx")
        env.data.qpos[3:7] = quaternion
        env.data.qpos[2] += 1.0
        env.data.qvel[:] = rng.normal(scale=0.5, size=env.data.qvel.shape)
        mujoco.mj_forward(env.model, env.data)
        env._invalidate_substep_aggregates()
        yield (
            env._get_obs().copy(),
            (env.data.qpos.copy(), env.data.qvel.copy(), env.data.mocap_pos.copy(), env.data.mocap_quat.copy()),
        )


def _restore(env, state) -> None:
    qpos, qvel, mocap_pos, mocap_quat = state
    env.data.qpos[:], env.data.qvel[:] = qpos, qvel
    env.data.mocap_pos[:], env.data.mocap_quat[:] = mocap_pos, mocap_quat


def test_whole_scene_rotation_preserves_observation(animal):
    _, env = animal
    worst = 0.0
    for before, state in _lifted_moving_states(env, 100):
        for turn in TURNS_DEG:
            _restore(env, state)
            turned = _apply_spawn_yaw(env, SpawnYaw(turn, whole_scene=True))
            worst = max(worst, float(np.max(np.abs(turned - before))))
    assert worst <= FLOAT32_BUDGET


def test_animal_only_turn_changes_only_target_bearing(animal):
    species, env = animal
    bearing = _layout(species, env)["prey_direction_body"]
    others = np.ones(env.observation_space.shape, dtype=bool)
    others[bearing] = False
    for before, state in _lifted_moving_states(env, 10):
        _restore(env, state)
        turned = _apply_spawn_yaw(env, SpawnYaw(90.0, whole_scene=False))
        assert np.max(np.abs(turned[others] - before[others])) <= FLOAT32_BUDGET
        assert np.linalg.norm(turned[bearing] - before[bearing]) > 0.5


def test_quaternion_sign_has_no_effect(animal):
    _, env = animal
    for before, state in _lifted_moving_states(env, 10):
        _restore(env, state)
        env.data.qpos[3:7] *= -1.0
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_array_equal(env._get_obs(), before)


@pytest.mark.parametrize("policy", ["zero", "linear"])
@pytest.mark.parametrize(("species", "env_class"), SPECIES)
def test_six_heading_paired_rollouts_repeat_to_solver_tolerance(species, env_class, policy):
    traces = paired_traces(species, 3042, {"zero": zero_policy, "linear": linear_policy}[policy], PAIRED_STEPS)
    assert len(traces[0.0]["actions"]) == PAIRED_STEPS, "the comparison must cover the whole window"
    assert_paired(traces)
