"""The heading-free observation on every model (``bipedal-heading-free/v1`` and ``quadrupedal-heading-free/v1``).

The T-Rex (policy interface r14), the anatomical Compsognathus (r4) and the
velociraptor (r13) were the first models to build their observation through
``environments/shared/heading_observation.py`` (plant_versions notes 18 and
21).  Since the target-free revision (plant_versions note 24, decision D-D59)
every model does, with no target channel (``bipedal-target-heading-free/v1``
carried the target direction and distance before the command): the
Allosaurus on the same biped schema, the Compsognathus robot and the two
quadrupeds switched from their world-frame layouts (the robot to the biped
schema with floor-contact feet, the quadrupeds to the quadruped schema, whose
root segments describe the torso).
These tests run the real environments (no SB3; the shared legs of the test
matrix) and pin the design's acceptance list
(docs/HEADING_INVARIANCE_DESIGN_2026_10.md, "Migration and acceptance"):

* the vector is laid out exactly as the plant contract's segments say, with
  the root segments named for the schema's root body and the command
  trailing;
* it is finite and correctly sized at construction (before any forward pass,
  where the orientation sensor still reads zero), at reset, at every step and
  at a terminal state;
* the gyro and accelerometer are the sensors' own site-frame readings, not
  rotated a second time, and the root rotation agrees with MuJoCo's after a
  forward pass;
* a whole-scene turn of a lifted, tilted, moving state leaves the float32
  observation unchanged (budget 2e-6), across the yaw wrap; so does an
  animal-only turn, which leaves the episode's reference heading behind (no
  channel reads it); ``q`` and ``-q`` read the same;
* six-heading paired rollouts repeat to solver tolerance: the same seed,
  turned with its whole scene to each of -90, -45, 0, 45, 90 and 180 degrees
  in a fresh environment, steps the same actions, joint trajectory, rewards
  and observations.  Here over 250 steps of seed 3042 under the zero action
  and a fixed deterministic linear policy (about 5 s per species); the
  full-horizon version, with a frozen normalizer, an untrained PPO policy and
  the floor-truth metrics, is ``test_heading_free_rollouts.py`` in the SB3
  job's full depth (on the three first bipeds).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus_altithorax.anatomical.env import BrachiosaurusAltithoraxAnatomicalEnv
from environments.shared.command_frame import COMMAND_SEGMENT_NAME, COMMAND_WIDTH
from environments.shared.config import load_stage_config
from environments.shared.heading_observation import (
    BIPEDAL_HEADING_FREE_SCHEMA,
    HEADING_FREE_ROOT_LABELS,
    QUADRUPEDAL_HEADING_FREE_SCHEMA,
    body_to_world,
    root_quaternion,
)
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import load_plant_versions
from environments.shared.plant_contract.policy_layer import _policy_interface_payload
from environments.shared.reporting.stance_report import SpawnYaw, _apply_spawn_yaw
from environments.shared.species_names import model_ids

from .heading_free_helpers import (
    FLOAT32_BUDGET,
    HEADING_FREE_ENVS,
    assert_paired,
    linear_policy,
    paired_traces,
    zero_policy,
)

MODELS = tuple(
    pytest.param(model_id, env_class, id=f"{model_id.species}-{model_id.embodiment}")
    for model_id, env_class in HEADING_FREE_ENVS.items()
)
#: Each model's schema and observation width since the target-free revision (plant_versions note 24): the
#: quadrupeds' root segments describe the torso, every biped's the pelvis.
EXPECTED_LAYOUTS = {
    ModelId("tyrannosaurus_rex", "anatomical"): (BIPEDAL_HEADING_FREE_SCHEMA, 59),
    ModelId("compsognathus_longipes", "anatomical"): (BIPEDAL_HEADING_FREE_SCHEMA, 51),
    ModelId("velociraptor_mongoliensis", "anatomical"): (BIPEDAL_HEADING_FREE_SCHEMA, 67),
    ModelId("allosaurus_fragilis", "anatomical"): (BIPEDAL_HEADING_FREE_SCHEMA, 41),
    ModelId("compsognathus_longipes", "robot"): (BIPEDAL_HEADING_FREE_SCHEMA, 41),
    ModelId("brachiosaurus_altithorax", "anatomical"): (QUADRUPEDAL_HEADING_FREE_SCHEMA, 81),
    ModelId("dibothrosuchus_elaphros", "anatomical"): (QUADRUPEDAL_HEADING_FREE_SCHEMA, 75),
}
#: The wild-action scale that ends an episode early, so the terminal state is exercised: N(0, 0.8) clipped
#: unless listed.  The sauropod's four-legged base stands N(0, 0.8) for the whole 1000-step horizon (seeds 3042
#: and 3043); N(0, 5) clipped, nearly bang-bang, fells it at step 723 of seed 3042.
WILD_ACTION_SCALES: dict[type, float] = {BrachiosaurusAltithoraxAnatomicalEnv: 5.0}
#: Whole-scene turns for the pointwise test, across the yaw wrap.
TURNS_DEG = (-179.9, -90.0, -45.0, 17.0, 90.0, 180.0, 359.0)
#: The shared legs' paired-rollout window (about 5 s per species for both policies).
PAIRED_STEPS = 250
#: Episodes that end inside the paired window, by (model, policy): the steps they last and how they end.  The
#: Allosaurus's statue does not stand (its actions are offsets from the home torques, and only a controller holds
#: the pose): seed 3042 falls at step 91 under the zero action and at step 154 under the linear policy, the same at
#: every heading, so its comparison covers the fall and the terminal state (``assert_paired`` compares the ends;
#: joint state agreed to 3.4e-12).  Every other model stands the whole window under both policies.
PAIRED_ENDS = {
    (ModelId("allosaurus_fragilis", "anatomical"), "zero"): (91, (True, False, "fallen")),
    (ModelId("allosaurus_fragilis", "anatomical"), "linear"): (154, (True, False, "fallen")),
}


def _stance_env(env_class, model_id: ModelId):
    return env_class(**load_stage_config(model_id, "stance")["env_kwargs"])


@pytest.fixture(params=[pytest.param(tuple(param.values), id=param.id) for param in MODELS])
def animal(request):
    model_id, env_class = request.param
    env = _stance_env(env_class, model_id)
    yield model_id, env
    env.close()


def test_every_model_is_heading_free():
    """All seven models are covered here, each on its schema (the plant's ``observation_schema``)."""
    assert set(HEADING_FREE_ENVS) == set(EXPECTED_LAYOUTS) == set(model_ids())
    versions = load_plant_versions()[1]
    assert {model_id: versions[model_id].observation_schema for model_id in EXPECTED_LAYOUTS} == {
        model_id: schema for model_id, (schema, _) in EXPECTED_LAYOUTS.items()
    }


def _root(model_id: ModelId) -> str:
    """The schema's root label: the free body the root segments describe and name."""
    return HEADING_FREE_ROOT_LABELS[EXPECTED_LAYOUTS[model_id][0]]


def _layout(model_id: ModelId, env) -> dict[str, slice]:
    """Each observation segment's slice, read off the plant contract's ``observation_segments``."""
    version = load_plant_versions()[1][model_id]
    assert version.observation_schema == EXPECTED_LAYOUTS[model_id][0]
    payload = _policy_interface_payload(env.model, env, version)
    layout: dict[str, slice] = {}
    start = 0
    for segment in payload["observation_segments"]:
        width = len(segment["components"]) if "components" in segment else int(segment["width"])
        layout[segment["name"]] = slice(start, start + width)
        start += width
    assert start == env.observation_space.shape[0]
    return layout


def test_layout_matches_segments_and_command_is_last(animal):
    model_id, env = animal
    obs, _ = env.reset(seed=3042)
    for _ in range(5):
        obs, *_ = env.step(np.full(env.action_space.shape, 0.05, dtype=np.float32))
    layout = _layout(model_id, env)
    root = _root(model_id)
    names = list(layout)
    assert names == [
        "joint_position",
        "joint_velocity",
        f"{root}_projected_gravity",
        f"{root}_angular_velocity",
        f"{root}_heading_linear_velocity",
        f"{root}_linear_acceleration",
        "foot_contact",
        COMMAND_SEGMENT_NAME,
    ]
    # The root segments describe the free body itself: the one body with the free joint whose qpos[3:7] they read.
    assert env.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and env.model.jnt_bodyid[0] == env.model.body(root).id
    assert obs.shape == env.observation_space.shape == (layout[COMMAND_SEGMENT_NAME].stop,)
    assert obs.shape == (EXPECTED_LAYOUTS[model_id][1],)
    assert layout[COMMAND_SEGMENT_NAME] == slice(obs.shape[0] - COMMAND_WIDTH, obs.shape[0])
    data = env.data
    as32 = np.float32
    np.testing.assert_array_equal(obs[layout["joint_position"]], data.qpos[7:].astype(as32))
    np.testing.assert_array_equal(obs[layout["joint_velocity"]], data.qvel[6:].astype(as32))
    gyro, accel = env._sensor_gyro_start, env._sensor_accel_start
    np.testing.assert_array_equal(
        obs[layout[f"{root}_angular_velocity"]], data.sensordata[gyro : gyro + 3].astype(as32)
    )
    np.testing.assert_array_equal(
        obs[layout[f"{root}_linear_acceleration"]], data.sensordata[accel : accel + 3].astype(as32)
    )
    # The foot segment: one channel per foot (its summed touch), or per group of the env's _foot_observation_groups
    # where it declares them (the velociraptor's six per-site channels since policy interface r14), each summed in
    # the pinned group[0] + sum(rest) order; the robot's two channels are its soles' floor-contact normal force
    # (_foot_observation_floor_geoms, policy interface r4), which on these standing steps equals their touch.
    channels = getattr(env, "_foot_observation_groups", None)
    floor_groups = env._foot_observation_floor_geoms
    if floor_groups is not None:
        assert channels is None
        expected_foot = env._foot_floor_contact_forces()
        assert len(expected_foot) == len(env._foot_sensor_groups)
        np.testing.assert_allclose(expected_foot, env._foot_contact_forces(), rtol=1e-9)
    else:
        expected_foot = (
            env._foot_contact_forces()
            if channels is None
            else tuple(
                float(data.sensordata[group[0]] + sum(data.sensordata[index] for index in group[1:]))
                for group in channels
            )
        )
    np.testing.assert_array_equal(obs[layout["foot_contact"]], np.asarray(expected_foot).astype(as32))
    if channels is not None:  # per-site channels still total the per-foot sums
        totals = [
            sum(expected_foot[i] for i, group in enumerate(channels) if set(group) <= set(foot))
            for foot in env._foot_sensor_groups
        ]
        np.testing.assert_allclose(totals, env._foot_contact_forces(), rtol=1e-12)
    np.testing.assert_array_equal(obs[layout[COMMAND_SEGMENT_NAME]], np.zeros(COMMAND_WIDTH))
    # The two heading-free blocks, from MuJoCo's own quaternion-to-matrix map of the free root's qpos[3:7]:
    # gravity is minus the third row of R, the velocity is qvel[0:3] in the gravity-aligned heading frame.  (After
    # a step qpos is post-integration while xmat and the sensors come from the last substep's forward pass, so xmat
    # lags R by one physics substep, as it lagged qvel[0:3].)
    quaternion = data.qpos[3:7] / np.linalg.norm(data.qpos[3:7])
    flat = np.empty(9)
    mujoco.mju_quat2Mat(flat, quaternion)
    rotation = flat.reshape(3, 3)
    np.testing.assert_allclose(obs[layout[f"{root}_projected_gravity"]], -rotation[2], atol=1e-6)
    heading = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(heading), math.sin(heading)
    v = data.qvel[0:3]
    np.testing.assert_allclose(
        obs[layout[f"{root}_heading_linear_velocity"]], [c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]], atol=1e-6
    )


@pytest.mark.parametrize(("model_id", "env_class"), MODELS)
def test_construction_observation_is_finite_and_upright(model_id, env_class):
    """The constructor calls _get_obs before any forward pass: the orientation sensor still reads zero there."""
    env = _stance_env(env_class, model_id)
    try:
        quat = env._sensor_quat_start
        np.testing.assert_array_equal(env.data.sensordata[quat : quat + 4], np.zeros(4))
        obs = env._get_obs()
        assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        gravity = _layout(model_id, env)[f"{_root(model_id)}_projected_gravity"]
        np.testing.assert_array_equal(obs[gravity], [0.0, 0.0, -1.0])
    finally:
        env.close()


def test_reset_step_terminal_observations_are_finite(animal):
    model_id, env = animal
    obs, _ = env.reset(seed=3042)
    assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
    rng = np.random.default_rng(1)
    scale = WILD_ACTION_SCALES.get(type(env), 0.8)
    terminated = truncated = False
    steps = 0
    info: dict = {}
    while not (terminated or truncated):
        action = np.clip(rng.normal(scale=scale, size=env.action_space.shape), -1.0, 1.0).astype(np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
        assert np.all(np.isfinite(obs)), f"{model_id} step {steps}"
    assert terminated, f"{model_id}: wild actions should end the episode early, so the terminal state is exercised"
    assert steps < env.max_episode_steps and info.get("termination_reason")
    np.testing.assert_array_equal(obs[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH))


def test_local_sensors_are_not_rotated_twice(animal):
    """The gyro and accelerometer report in their own site frame already; a second rotation would re-leak yaw."""
    model_id, env = animal
    layout = _layout(model_id, env)
    root = _root(model_id)
    env.reset(seed=3043)
    rng = np.random.default_rng(5)
    gyro, accel = env._sensor_gyro_start, env._sensor_accel_start
    for _ in range(10):
        obs, *_ = env.step(np.clip(rng.normal(scale=0.2, size=env.action_space.shape), -1, 1).astype(np.float32))
        np.testing.assert_array_equal(
            obs[layout[f"{root}_angular_velocity"]], env.data.sensordata[gyro : gyro + 3].astype(np.float32)
        )
        np.testing.assert_array_equal(
            obs[layout[f"{root}_linear_acceleration"]], env.data.sensordata[accel : accel + 3].astype(np.float32)
        )


def test_root_and_sensor_rotation_agree_after_forward(animal):
    """After mj_forward the qpos-derived R equals the root's xmat and the framequat sensor's rotation (its site is
    unrotated on the root, or the sensor reads the root body's frame)."""
    model_id, env = animal
    root_id = env.model.body(_root(model_id)).id
    env.reset(seed=3042)
    rng = np.random.default_rng(812)
    quat = env._sensor_quat_start
    for _ in range(20):
        quaternion = rng.normal(size=4)
        env.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        mujoco.mj_forward(env.model, env.data)
        rotation = body_to_world(root_quaternion(env.data.qpos))
        np.testing.assert_allclose(rotation, env.data.xmat[root_id].reshape(3, 3), atol=1e-12)
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
            (env.data.qpos.copy(), env.data.qvel.copy(), env._reference_heading_2d.copy()),
        )


def _restore(env, state) -> None:
    qpos, qvel, reference = state
    env.data.qpos[:], env.data.qvel[:] = qpos, qvel
    env._reference_heading_2d = reference.copy()


def test_whole_scene_rotation_preserves_observation(animal):
    _, env = animal
    worst = 0.0
    for before, state in _lifted_moving_states(env, 100):
        for turn in TURNS_DEG:
            _restore(env, state)
            turned = _apply_spawn_yaw(env, SpawnYaw(turn, whole_scene=True))
            worst = max(worst, float(np.max(np.abs(turned - before))))
    assert worst <= FLOAT32_BUDGET


def test_animal_only_turn_changes_nothing_either(animal):
    """The reference heading stays behind, and no channel reads it: the observation is unchanged."""
    _, env = animal
    for before, state in _lifted_moving_states(env, 10):
        _restore(env, state)
        turned = _apply_spawn_yaw(env, SpawnYaw(90.0, whole_scene=False))
        np.testing.assert_array_equal(env._reference_heading_2d, state[2])
        assert np.max(np.abs(turned - before)) <= FLOAT32_BUDGET


def test_quaternion_sign_has_no_effect(animal):
    _, env = animal
    for before, state in _lifted_moving_states(env, 10):
        _restore(env, state)
        env.data.qpos[3:7] *= -1.0
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_array_equal(env._get_obs(), before)


@pytest.mark.parametrize("policy", ["zero", "linear"])
@pytest.mark.parametrize(("model_id", "env_class"), MODELS)
def test_six_heading_paired_rollouts_repeat_to_solver_tolerance(model_id, env_class, policy):
    traces = paired_traces(model_id, 3042, {"zero": zero_policy, "linear": linear_policy}[policy], PAIRED_STEPS)
    steps, end = PAIRED_ENDS.get((model_id, policy), (PAIRED_STEPS, None))
    assert len(traces[0.0]["actions"]) == steps, "the comparison must cover the whole window or the listed fall"
    assert traces[0.0]["end"] == end
    assert_paired(traces)
