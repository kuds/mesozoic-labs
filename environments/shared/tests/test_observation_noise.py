"""The opt-in observation noise: ``environments/shared/observation_noise.py`` and its hook in ``BaseDinoEnv.step``.

Design record ``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §7.  No committed
model lists the noise keys, so the noise itself runs on test-only subclasses
of every manifest model that opt in (``observation_noise_helpers.py``).  Numpy
and MuJoCo only: no SB3 (the behaviour-transition check of the
``observation_noise`` section is in ``test_behavior_checkpoint.py``).
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import pickle
import re
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from environments.shared import species_registry, task_fingerprint
from environments.shared.base_env import BaseDinoEnv
from environments.shared.command_frame import COMMAND_PROBE_VECTOR
from environments.shared.config import load_stage_config
from environments.shared.episode_physics import EpisodePhysicsRandomizer
from environments.shared.episode_start import (
    FEATURE_STREAM_KEYS,
    OBSERVATION_NOISE_STREAM_KEY,
    feature_stream,
    mark_training_rollouts,
)
from environments.shared.model_id import ModelId
from environments.shared.observation_noise import (
    OBSERVATION_NOISE_ENV_KEYS,
    OBSERVATION_NOISE_IMPLEMENTATION,
    UNIT_NORM_FLOOR,
    UNIT_SEGMENT_SUFFIXES,
    NoiseSegment,
    ObservationNoise,
    observation_layout,
    observation_schema,
    resolve_observation_noise,
)
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.plant_contract import (
    PlantContractError,
    current_plant_identity,
    fingerprint_model_layers,
    load_plant_versions,
    policy_layer,
    validate_environment_plant,
)
from environments.shared.species_names import model_ids
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import (
    TaskFingerprintError,
    compute_task_fingerprint,
    stage_task_fingerprint,
    validate_recorded_task,
)

from . import observation_noise_helpers
from .observation_noise_helpers import NOISY_ENV_CLASSES, OBSERVATION_NOISE_CARVE_OUT

ROBOT = ModelId("compsognathus_longipes", "robot")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
ALLOSAURUS = ModelId("allosaurus_fragilis", "anatomical")
NOISY_ROBOT = NOISY_ENV_CLASSES[ROBOT]
MODEL_IDS = model_ids()
MODEL_STAGES = [
    pytest.param(model_id, entry.reference, id=f"{model_id.species}-{model_id.embodiment}-{entry.id}")
    for model_id in MODEL_IDS
    for entry in load_stage_manifest(model_id).stages
]

#: The manifest models that opt into the noise: each lists both keys (the table defaulting to ``None``, the flag to
#: ``False``), forwards them to ``BaseDinoEnv`` and declares the carve-out at off value ``None``.  Empty until the
#: Triceratops robot joins it (design record §8.9); every other model refuses the keys.
OPTED_IN_OBSERVATION_NOISE: frozenset[ModelId] = frozenset()

#: Each segment kind of the heading-free schemas, every model's since main's target-free revision (D-D59) removed the
#: legacy schemas with their orientation, world-velocity and target segments: the segment's name less its root label,
#: longest first, so a suffix match is the kind.  A segment of any other kind fails :func:`_kind`, so a new schema
#: (commit 5's planned target schema among them) has to name its kinds here.
KINDS = (
    "heading_linear_velocity",
    "linear_acceleration",
    "projected_gravity",
    "angular_velocity",
    "joint_position",
    "joint_velocity",
    "foot_contact",
    "command",
)

#: The test table by segment kind: the robot's values of design record §7, the root's projected gravity (the one unit
#: segment of a heading-free schema, so its renormalisation runs) among them, and nothing on the command.
STD_BY_KIND = {
    "joint_position": 0.005,
    "joint_velocity": 0.05,
    "projected_gravity": 0.02,
    "angular_velocity": 0.02,
    "heading_linear_velocity": 0.02,
    "linear_acceleration": 0.2,
    "foot_contact": 2.5,
}


def _kind(name: str) -> str:
    for kind in KINDS:
        if name == kind or name.endswith(f"_{kind}"):
            return kind
    raise AssertionError(f"no segment kind for {name!r}")


def _first_stage_kwargs(model_id: ModelId) -> dict[str, Any]:
    return dict(load_stage_config(model_id, load_stage_manifest(model_id).stages[0].reference)["env_kwargs"])


@pytest.fixture(scope="module")
def stance_kwargs():
    return dict(load_stage_config(ROBOT, "stance")["env_kwargs"])


def _table(env) -> dict[str, float]:
    """The test table for *env*'s observation (STD_BY_KIND)."""
    return {name: STD_BY_KIND[_kind(name)] for name, _ in observation_layout(env) if _kind(name) in STD_BY_KIND}


@pytest.fixture(scope="module")
def robot_table(stance_kwargs):
    env = NOISY_ROBOT(**stance_kwargs)
    try:
        return _table(env)
    finally:
        env.close()


def _noisy(stance_kwargs, table, *, eval_noise=False, training=False, **overrides):
    env = NOISY_ROBOT(**{**stance_kwargs, **overrides}, observation_noise_std=table, observation_noise_eval=eval_noise)
    if training:
        mark_training_rollouts(env, True)
    return env


def _zeros(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _segment_slice(env, name: str) -> slice:
    """The slice of segment *name* in *env*'s observation."""
    start = 0
    for segment, width in observation_layout(env):
        if segment == name:
            return slice(start, start + width)
        start += width
    raise AssertionError(f"no segment {name!r}")


# ---------------------------------------------------------------------------------------------------- the module
def test_the_keys_the_tag_and_the_stream_key():
    assert OBSERVATION_NOISE_ENV_KEYS == ("observation_noise_std", "observation_noise_eval")
    assert OBSERVATION_NOISE_IMPLEMENTATION == (
        "obs_noise/v1+gaussian-per-segment+post-step+episode-stream+unit-segment-renormalize"
    )
    assert UNIT_SEGMENT_SUFFIXES == ("_orientation_quaternion", "_projected_gravity", "_direction", "_direction_body")
    assert OBSERVATION_NOISE_STREAM_KEY == 0x0B5E4015 and FEATURE_STREAM_KEYS.count(OBSERVATION_NOISE_STREAM_KEY) == 1


def _segments(*specs: tuple[str, int, float]) -> tuple[NoiseSegment, ...]:
    segments, start = [], 0
    for name, width, std in specs:
        segments.append(NoiseSegment(name, start, width, std, std > 0.0 and name.endswith(UNIT_SEGMENT_SUFFIXES)))
        start += width
    return tuple(segments)


def test_one_draw_covers_the_whole_width_so_a_segment_on_or_off_never_moves_another():
    """Each step draws width standard normals in observation order whatever the table holds; the unit segments
    with a positive std are renormalised (floor 1e-9) and the result is float32."""
    obs = np.array([0.1, -0.2, 0.0, 0.0, 1.0, 0.5, 3.0], dtype=np.float32)
    both = ObservationNoise(
        _segments(("joint_position", 2, 0.5), ("pelvis_projected_gravity", 3, 0.25), ("x", 2, 2.0)), False
    )
    first = ObservationNoise(
        _segments(("joint_position", 2, 0.5), ("pelvis_projected_gravity", 3, 0.0), ("x", 2, 0.0)), False
    )
    last = ObservationNoise(
        _segments(("joint_position", 2, 0.0), ("pelvis_projected_gravity", 3, 0.0), ("x", 2, 2.0)), False
    )
    assert both.width == 7 and both.unit_slices == (slice(2, 5),) and first.unit_slices == ()
    normals = np.random.default_rng(3).standard_normal(7)
    out_both = both.apply(obs, np.random.default_rng(3))
    out_first = first.apply(obs, np.random.default_rng(3))
    out_last = last.apply(obs, np.random.default_rng(3))
    assert out_both.dtype == np.float32 and out_both.shape == obs.shape
    expected = obs.astype(np.float64) + both.std * normals
    expected[2:5] /= np.linalg.norm(expected[2:5])
    np.testing.assert_array_equal(out_both, expected.astype(np.float32))
    np.testing.assert_array_equal(out_both[:2], out_first[:2])
    np.testing.assert_array_equal(out_both[5:], out_last[5:])
    np.testing.assert_array_equal(out_first[2:], obs[2:])
    assert abs(float(np.linalg.norm(out_both[2:5].astype(np.float64))) - 1.0) < 1e-6
    # The floor: a noisy unit segment of zero length is divided by 1e-9, never by zero.
    degenerate = ObservationNoise(
        _segments(
            ("torso_projected_gravity", 3, 1.0),
        ),
        False,
    )
    degenerate.std[:] = 0.0  # so the noisy vector is the zero vector
    with np.errstate(all="raise"):
        out = degenerate.apply(np.zeros(3, dtype=np.float32), np.random.default_rng(1))
    np.testing.assert_array_equal(out, np.zeros(3))
    assert UNIT_NORM_FLOOR == 1e-9
    assert both.std_by_segment == {"joint_position": 0.5, "pelvis_projected_gravity": 0.25, "x": 2.0}


INVALID = [
    pytest.param({"joint_position": 0.01}, "yes", "observation_noise_eval must be true or false", id="eval-string"),
    pytest.param({"joint_position": 0.01}, 1, "observation_noise_eval must be true or false", id="eval-int"),
    pytest.param(None, "no", "observation_noise_eval must be true or false", id="eval-string-while-off"),
    pytest.param(None, True, "observation_noise_eval needs observation_noise_std", id="eval-without-a-table"),
    pytest.param([0.01], False, "must be a table of segment = std", id="a-list"),
    pytest.param("joint_position=0.01", False, "must be a table of segment = std", id="a-string"),
    pytest.param(
        {"joint_angle": 0.01}, False, r"names \['joint_angle'\], not segments.*'joint_position'", id="unknown"
    ),
    pytest.param({"joint_position": True}, False, "joint_position must be a number", id="a-bool"),
    pytest.param({"joint_position": "0.01"}, False, "joint_position must be a number", id="a-string-std"),
    pytest.param({"joint_position": float("nan")}, False, "finite and nonnegative", id="nan"),
    pytest.param({"joint_position": float("inf")}, False, "finite and nonnegative", id="inf"),
    pytest.param({"joint_position": 1e39}, False, r"finite and nonnegative, in float32", id="overflows-float32"),
    pytest.param({"joint_position": -0.01}, False, "finite and nonnegative", id="negative"),
    pytest.param({"joint_position": 0.01, "command": 0.1}, False, "the command is the task's input", id="command"),
    pytest.param({"joint_position": 0.0, "command": 0.0}, False, "sets no positive std", id="all-zero"),
    pytest.param({}, False, "sets no positive std", id="empty"),
]


@pytest.mark.parametrize(("table", "eval_noise", "message"), INVALID)
def test_an_invalid_config_is_refused(stance_kwargs, table, eval_noise, message):
    env = NOISY_ROBOT(**stance_kwargs)
    try:
        with pytest.raises(ValueError, match=message):
            resolve_observation_noise(env, table, eval_noise)
    finally:
        env.close()


@pytest.mark.parametrize(("table", "eval_noise", "message"), INVALID)
def test_an_invalid_config_is_refused_at_construction(stance_kwargs, table, eval_noise, message):
    with pytest.raises(ValueError, match=message):
        NOISY_ROBOT(**stance_kwargs, observation_noise_std=table, observation_noise_eval=eval_noise)


def test_a_layout_that_does_not_sum_to_the_observation_width_is_refused(stance_kwargs, robot_table):
    env = NOISY_ROBOT(**stance_kwargs)
    try:
        width = env.observation_space.shape[0]
        env.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(width + 1,), dtype=np.float32)
        with pytest.raises(ValueError, match=f"sum to {width}, not the observation width {width + 1}"):
            resolve_observation_noise(env, robot_table, False)
    finally:
        env.close()


@pytest.mark.parametrize("schema", ["bipedal-target/v1", "quadrupedal-target/v1", "bipedal-heading-free/v2"])
def test_a_schema_that_is_not_heading_free_is_refused_by_the_segment_table_and_by_the_noise(stance_kwargs, schema):
    """observation_segments refuses every schema but the heading-free ones with PlantContractError, so neither the
    layout nor a noise resolved for an explicit schema (commit 5's planned target schema among them, design record
    §3) names its segments by a heading-free table; the robot's own schema, named, is its default layout."""
    env = NOISY_ROBOT(**stance_kwargs)
    try:
        refused = f"observation schema {re.escape(repr(schema))} is not one of the observation schemas"
        with pytest.raises(PlantContractError, match=refused):
            policy_layer.observation_segments(env.model, env, schema)
        with pytest.raises(PlantContractError, match=refused):
            observation_layout(env, schema)
        with pytest.raises(PlantContractError, match=refused):
            resolve_observation_noise(env, {"joint_position": 0.01}, False, schema=schema)
        assert observation_layout(env, "bipedal-heading-free/v1") == observation_layout(env)
    finally:
        env.close()


def test_an_env_of_no_manifest_model_has_no_schema_to_name_its_segments():
    with pytest.raises(ValueError, match="not the env class of a manifest model"):
        observation_schema(object())


def test_the_validated_noise_keeps_every_segment_in_order_with_its_std(stance_kwargs, robot_table):
    """A plain-dict table resolves to every segment of the layout in order: the table's std where it names one, 0.0
    elsewhere; a unit segment is renormalised exactly when its std is positive."""
    env = _noisy(stance_kwargs, robot_table, eval_noise=True)
    try:
        noise = env._observation_noise
        assert isinstance(noise, ObservationNoise) and noise.eval_noise is True
        layout = observation_layout(env)
        assert [(s.name, s.width) for s in noise.segments] == list(layout)
        assert noise.std_by_segment == {name: robot_table.get(name, 0.0) for name, _ in layout}
        assert [s.start for s in noise.segments] == list(np.cumsum([0, *[w for _, w in layout]])[:-1])
        assert {s.name for s in noise.segments if s.renormalize} == {"pelvis_projected_gravity"}
        assert noise.width == env.observation_space.shape[0]
        # A table that leaves every unit segment out renormalises none (std 0 is never renormalised).
        partial = resolve_observation_noise(env, {"joint_position": 0.01}, False)
        assert not any(s.renormalize for s in partial.segments) and partial.unit_slices == ()
        half = resolve_observation_noise(env, {"joint_position": 0.01, "pelvis_projected_gravity": 0.01}, False)
        assert [s.name for s in half.segments if s.renormalize] == ["pelvis_projected_gravity"]
        assert resolve_observation_noise(env, None, False) is None
        assert (
            resolve_observation_noise(env, {"joint_position": 1}, np.bool_(False)).std_by_segment["joint_position"]
            == 1.0
        )
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- off by default
def test_the_base_env_takes_both_keys_after_the_randomisation_keys_off_by_default():
    """The two keys follow the delay's two and the randomisation's six, which follow command_config."""
    parameters = list(inspect.signature(BaseDinoEnv.__init__).parameters.values())
    names = [parameter.name for parameter in parameters]
    start = names.index("command_config") + 1 + 2 + 6
    assert names[start:] == list(OBSERVATION_NOISE_ENV_KEYS)
    assert [parameter.default for parameter in parameters[start:]] == [None, False]
    assert BaseDinoEnv._observation_noise is None and BaseDinoEnv._observation_noise_active is None


def test_the_models_listing_a_key_are_exactly_the_opted_in_table():
    assert OPTED_IN_OBSERVATION_NOISE <= set(MODEL_IDS)
    listing = {
        model_id
        for model_id in MODEL_IDS
        if set(OBSERVATION_NOISE_ENV_KEYS) & set(inspect.signature(species_registry.env_class(model_id)).parameters)
    }
    assert listing == OPTED_IN_OBSERVATION_NOISE


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_an_opted_in_model_takes_both_keys_off_by_default_and_every_other_refuses_them(model_id):
    """An opted-in class lists both keys (None and False), forwards them to BaseDinoEnv and declares the carve-out;
    a TOML setting either key on any other model raises TypeError."""
    env_class = species_registry.env_class(model_id)
    parameters = inspect.signature(env_class).parameters
    if model_id in OPTED_IN_OBSERVATION_NOISE:
        assert parameters["observation_noise_std"].default is None
        assert parameters["observation_noise_eval"].default is False
        assert OBSERVATION_NOISE_CARVE_OUT in tuple(getattr(env_class, "OPT_IN_TASK_KWARGS", ()))
        env = env_class(observation_noise_std={"joint_position": 0.01})
        try:
            assert env._observation_noise is not None
        finally:
            env.close()
        return
    assert not set(OBSERVATION_NOISE_ENV_KEYS) & set(parameters)
    for key, value in zip(OBSERVATION_NOISE_ENV_KEYS, ({"joint_position": 0.01}, True)):
        with pytest.raises(TypeError, match=key):
            env_class(**{key: value})


def test_the_test_subclasses_opt_in_as_a_model_would():
    assert NOISY_ENV_CLASSES.keys() == set(MODEL_IDS)
    for model_id, env_class in NOISY_ENV_CLASSES.items():
        canonical = species_registry.env_class(model_id)
        assert env_class.__mro__[1] is canonical and env_class.__name__ == f"Noisy{canonical.__name__}"
        parameters = inspect.signature(env_class).parameters
        assert list(parameters)[-2:] == list(OBSERVATION_NOISE_ENV_KEYS)
        assert [parameters[key].default for key in OBSERVATION_NOISE_ENV_KEYS] == [None, False]
        assert env_class.OPT_IN_TASK_KWARGS[-1] == OBSERVATION_NOISE_CARVE_OUT
        assert getattr(observation_noise_helpers, env_class.__qualname__) is env_class
        assert pickle.loads(pickle.dumps(env_class)) is env_class
        assert observation_noise_helpers.with_observation_noise(canonical) is env_class


@pytest.mark.parametrize(("model_id", "stage"), MODEL_STAGES)
def test_the_noise_is_off_exactly_when_the_stage_omits_the_table(model_id, stage):
    env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
    env = species_registry.env_class(model_id)(**env_kwargs)
    try:
        assert (env._observation_noise is None) == ("observation_noise_std" not in env_kwargs)
        if env._observation_noise is None:
            env.reset(seed=1)
            assert not env.observation_noise_active and env._observation_noise_rng is None
    finally:
        env.close()


def _record_get_obs(env) -> list[np.ndarray]:
    """Shadow *env*'s ``_get_obs`` with one that records every array it returns."""
    built: list[np.ndarray] = []
    original = env._get_obs

    def recording():
        obs = original()
        built.append(obs)
        return obs

    env._get_obs = recording
    return built


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_while_no_episode_is_noised_step_returns_the_very_object_get_obs_built(model_id):
    """Off, and on but in an episode that is not noised (an evaluation env without the eval flag), step() returns
    the array _get_obs built itself, not a copy (P5)."""
    env_kwargs = _first_stage_kwargs(model_id)
    off = species_registry.env_class(model_id)(**env_kwargs)
    probe = NOISY_ENV_CLASSES[model_id](**env_kwargs)
    inactive = NOISY_ENV_CLASSES[model_id](**env_kwargs, observation_noise_std=_table(probe))
    probe.close()
    try:
        assert off._observation_noise is None and inactive._observation_noise is not None
        for env in (off, inactive):
            built = _record_get_obs(env)
            env.reset(seed=5)
            assert not env.observation_noise_active
            rng = np.random.default_rng(5)
            for _ in range(4):
                obs, *_ = env.step(rng.uniform(-0.3, 0.3, size=env.action_space.shape).astype(np.float32))
                assert obs is built[-1]
    finally:
        off.close()
        inactive.close()


def _state(env) -> bytes:
    data = env.data
    return b"".join(
        np.asarray(value).tobytes() for value in (data.qpos, data.qvel, data.ctrl, data.act, data.time)
    ) + pickle.dumps(env.np_random.bit_generator.state)


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_marking_an_env_is_inert_while_the_noise_is_off(model_id):
    """A trainer's mark noises nothing while the table is off: a seeded zero-action then random-action rollout is
    byte-identical in its observations, rewards, infos, simulator state and generator."""
    env_kwargs = _first_stage_kwargs(model_id)
    assert "observation_noise_std" not in env_kwargs
    env_class = species_registry.env_class(model_id)
    plain, marked = env_class(**env_kwargs), env_class(**env_kwargs)
    try:
        mark_training_rollouts(marked, True)
        rows = []
        for env in (plain, marked):
            obs, info = env.reset(seed=2026)
            assert not env.observation_noise_active
            record = [obs.tobytes(), pickle.dumps(info), _state(env)]
            rng = np.random.default_rng(7)
            shape = env.action_space.shape
            actions = [np.zeros(shape, dtype=np.float32)] * 10 + [
                rng.uniform(-0.5, 0.5, size=shape).astype(np.float32) for _ in range(10)
            ]
            for action in actions:
                obs, reward, terminated, truncated, info = env.step(action)
                record += [obs.tobytes(), pickle.dumps((reward, terminated, truncated, info)), _state(env)]
            rows.append(record)
        assert rows[0] == rows[1]
    finally:
        plain.close()
        marked.close()


# ---------------------------------------------------------------------------------------------------- the layout
@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_the_layout_names_the_plant_contracts_segments_and_sums_to_the_observation_width(model_id):
    """Names and widths are observation_segments' and the policy payload's (the refactor records the very table);
    foot_contact's width is the number of foot channels; they sum to the observation width."""
    env = species_registry.env_class(model_id)()
    try:
        version = load_plant_versions()[1][model_id]
        assert observation_schema(env) == version.observation_schema
        segments = policy_layer.observation_segments(env.model, env, version.observation_schema)
        payload = policy_layer._policy_interface_payload(env.model, env, version)
        assert payload["observation_segments"] == segments
        layout = observation_layout(env)
        assert [name for name, _ in layout] == [segment["name"] for segment in segments]
        assert [w for _, w in layout] == [policy_layer._segment_width(segment) for segment in segments]
        floor = env._foot_observation_floor_geoms
        declared = env._foot_observation_groups
        channels = len(floor if floor is not None else declared if declared is not None else env._foot_sensor_groups)
        assert dict(layout)["foot_contact"] == channels
        assert sum(width for _, width in layout) == env.observation_space.shape[0]
        # A subclass of the model's class (a test subclass, a behaviour env) reads the model's schema.
        assert observation_schema(NOISY_ENV_CLASSES[model_id].__new__(NOISY_ENV_CLASSES[model_id])) == (
            version.observation_schema
        )
    finally:
        env.close()


def _shift(array: np.ndarray, index: Any, delta: Any) -> None:
    array[index] = np.asarray(array[index]) + delta


#: Each source of the observation, perturbed on the plant contract's probe state, and the segment kinds it may feed.
#: The orientation sensor and the root position feed no segment of a heading-free schema: perturbing them changes
#: nothing.
SOURCES = {
    "joint positions": (lambda env, data: _shift(data.qpos, slice(7, None), 0.0123), {"joint_position"}),
    "joint velocities": (lambda env, data: _shift(data.qvel, slice(6, None), 0.0123), {"joint_velocity"}),
    "root linear velocity": (
        lambda env, data: _shift(data.qvel, slice(0, 3), np.array([0.11, -0.07, 0.05])),
        {"heading_linear_velocity"},
    ),
    "root orientation": (
        lambda env, data: _shift(data.qpos, slice(3, 7), np.array([0.05, -0.04, 0.03, 0.02])),
        {"projected_gravity", "heading_linear_velocity"},
    ),
    "orientation sensor": (
        lambda env, data: _shift(data.sensordata, slice(env._sensor_quat_start, env._sensor_quat_start + 4), 0.1),
        set(),
    ),
    "gyro": (
        lambda env, data: _shift(data.sensordata, slice(env._sensor_gyro_start, env._sensor_gyro_start + 3), 0.1),
        {"angular_velocity"},
    ),
    "accelerometer": (
        lambda env, data: _shift(data.sensordata, slice(env._sensor_accel_start, env._sensor_accel_start + 3), 0.1),
        {"linear_acceleration"},
    ),
    "touch sensors": (
        lambda env, data: _shift(data.sensordata, sorted({a for group in env._foot_sensor_groups for a in group}), 0.5),
        {"foot_contact"},
    ),
    # The probe state has no contacts, so a floor-contact reading (the Compsognathus robot's foot channels) is fed
    # through the env's own reader, shadowed on the instance for this one observation.
    "floor contacts": (
        lambda env, data: setattr(
            env, "_foot_floor_contact_forces", lambda: tuple(0.5 for _ in env._foot_observation_floor_geoms or ())
        ),
        {"foot_contact"},
    ),
    "root position": (
        lambda env, data: _shift(data.xpos, env._root_body_id, np.array([0.1, -0.2, 0.05])),
        set(),
    ),
    "command": (lambda env, data: setattr(env, "_command", env._command + np.float32(0.125)), {"command"}),
}


def _fed_kinds(env, source: str, kinds: set[str]) -> set[str]:
    """The kinds *source* feeds on *env*: its foot channels read the touch sensors or the floor contacts, not both."""
    floor = getattr(env, "_foot_observation_floor_geoms", None) is not None
    if (source == "touch sensors" and floor) or (source == "floor contacts" and not floor):
        return set()
    return kinds


def _probe_observation(env, edit=None) -> np.ndarray:
    """*env*'s observation on the plant contract's deterministic probe state, after *edit* (as its probe runs it)."""
    data = policy_layer._deterministic_probe_data(env.model)
    saved = env.data, env._command
    try:
        env.data = data
        env._command = np.asarray(COMMAND_PROBE_VECTOR, dtype=np.float32)
        if edit is not None:
            edit(env, data)
        return np.asarray(env._get_obs(), dtype=np.float64)
    finally:
        env.data, env._command = saved
        vars(env).pop("_foot_floor_contact_forces", None)


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_perturbing_each_source_on_the_probe_state_changes_only_its_segments_and_every_segment_is_fed(model_id):
    env = species_registry.env_class(model_id)()
    try:
        names = [name for name, width in observation_layout(env) for _ in range(width)]
        present = {_kind(name) for name in names}
        base = _probe_observation(env)
        fed: set[str] = set()
        for source, (edit, kinds) in SOURCES.items():
            changed = {names[index] for index in np.flatnonzero(_probe_observation(env, edit) != base)}
            assert {_kind(name) for name in changed} == _fed_kinds(env, source, kinds) & present, source
            fed |= changed
        assert fed == set(names)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- the noise
@pytest.mark.parametrize("model_id", [BRACHIOSAURUS_ALTITHORAX, ALLOSAURUS], ids=str)
def test_each_segment_carries_its_std_and_the_unit_segments_stay_unit(model_id):
    """The quadrupedal (Brachiosaurus) and the bipedal (Allosaurus) heading-free schemas, the test table, reset until at least
    2000 noised steps: every component of a segment with a std carries it, within 5 / sqrt(2 n) of the setting (P6),
    and a mean within 5 std / sqrt(n); a segment without one is the clean observation exactly; each noised unit
    segment has norm 1 within 1e-6."""
    env_kwargs = _first_stage_kwargs(model_id)
    probe = NOISY_ENV_CLASSES[model_id](**env_kwargs)
    table = _table(probe)
    probe.close()
    env = NOISY_ENV_CLASSES[model_id](**env_kwargs, observation_noise_std=table, observation_noise_eval=True)
    try:
        noisy_rows, clean_rows = [], []
        obs, _ = env.reset(seed=11)
        assert env.observation_noise_active
        while len(noisy_rows) < 2000:
            obs, _, terminated, truncated, _ = env.step(_zeros(env))
            noisy_rows.append(obs)
            clean_rows.append(env._get_obs())
            if terminated or truncated:
                env.reset()
        noisy = np.asarray(noisy_rows, dtype=np.float64)
        clean = np.asarray(clean_rows, dtype=np.float64)
        n = len(noisy)
        for segment in env._observation_noise.segments:
            block = slice(segment.start, segment.start + segment.width)
            if segment.std == 0.0:
                np.testing.assert_array_equal(noisy[:, block], clean[:, block], err_msg=segment.name)
            elif segment.renormalize:
                norms = np.linalg.norm(noisy[:, block], axis=1)
                assert np.max(np.abs(norms - 1.0)) < 1e-6, segment.name
                assert np.any(noisy[:, block] != clean[:, block]), segment.name
            else:
                realised = noisy[:, block] - clean[:, block]
                rms = np.sqrt(np.mean(realised**2, axis=0))
                assert np.max(np.abs(rms / segment.std - 1.0)) <= 5.0 / np.sqrt(2 * n), (segment.name, rms)
                assert np.max(np.abs(realised.mean(axis=0))) <= 5.0 * segment.std / np.sqrt(n), segment.name
        assert {s.name for s in env._observation_noise.segments if s.std == 0.0} >= {"command"}
    finally:
        env.close()


#: Per schema, its one unit segment (main's target-free revision removed the target directions): the root's
#: projected gravity.
UNIT_SEGMENTS = {
    BRACHIOSAURUS_ALTITHORAX: ["torso_projected_gravity"],
    ALLOSAURUS: ["pelvis_projected_gravity"],
}


@pytest.mark.parametrize("model_id", [BRACHIOSAURUS_ALTITHORAX, ALLOSAURUS], ids=str)
def test_a_unit_segment_without_a_std_is_never_renormalised(model_id):
    """Design record §7: a unit segment is renormalised only when its std is above 0, so one the table leaves out is
    the clean observation exactly.  On the quadrupedal and the bipedal heading-free schema, whose one unit segment
    is the root's projected gravity: left out of a noised table, it is not renormalised; apply() passes it through
    untouched even where it is not of unit length; and over 300 noised zero-action steps it equals _get_obs() bit
    for bit.  Noised, it has norm 1 within 1e-6 at every step."""
    env_kwargs = _first_stage_kwargs(model_id)
    probe = NOISY_ENV_CLASSES[model_id](**env_kwargs)
    units = [name for name, _ in observation_layout(probe) if name.endswith(UNIT_SEGMENT_SUFFIXES)]
    probe.close()
    assert units == UNIT_SEGMENTS[model_id]
    (unit_name,) = units
    for noised in (False, True):
        table = {"joint_position": 0.005, **({unit_name: 0.02} if noised else {})}
        env = NOISY_ENV_CLASSES[model_id](**env_kwargs, observation_noise_std=table, observation_noise_eval=True)
        try:
            segment = {segment.name: segment for segment in env._observation_noise.segments}[unit_name]
            assert segment.renormalize is noised and (segment.std > 0.0) is noised
            block = _segment_slice(env, unit_name)
            if not noised:
                # apply() on an observation whose left-out unit segment has norm 0.5: a renormalisation would
                # scale it to 1.
                width = block.stop - block.start
                synthetic = np.zeros(env._observation_noise.width, dtype=np.float32)
                synthetic[block] = np.float32(0.5 / np.sqrt(width))
                out = env._observation_noise.apply(synthetic, np.random.default_rng(0))
                np.testing.assert_array_equal(out[block], synthetic[block])
            env.reset(seed=11)
            for _ in range(300):
                obs, _, terminated, truncated, _ = env.step(_zeros(env))
                if noised:
                    assert abs(float(np.linalg.norm(obs[block].astype(np.float64))) - 1.0) < 1e-6, unit_name
                else:
                    np.testing.assert_array_equal(obs[block], env._get_obs()[block], err_msg=unit_name)
                if terminated or truncated:
                    env.reset()
        finally:
            env.close()


def _rollout(env, seeds, actions) -> list[list[bytes]]:
    """Per episode: the reset observation, then per step the observation, reward, info, qpos, qvel and np_random."""
    episodes = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        record = [obs.tobytes(), pickle.dumps(info), env.np_random.bit_generator.state.__repr__().encode()]
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            record += [
                obs.tobytes(),
                pickle.dumps((reward, terminated, truncated, info)),
                env.data.qpos.tobytes(),
                env.data.qvel.tobytes(),
                repr(env.np_random.bit_generator.state).encode(),
            ]
        episodes.append(record)
    return episodes


def test_the_noise_is_reproducible_per_seed_and_moves_nothing_but_the_returned_observation(stance_kwargs, robot_table):
    """Two noised envs give the same episodes for the same seeds; against the clean env only the step observations
    differ: the reset observations, the rewards, the infos, the physics and np_random are byte-identical, so the
    noise draws nothing from np_random.  Unseeded episodes continue the stream (each its own noise); a seeded reset
    rebuilds it."""
    seeds = (7, None, None, 7)
    rng = np.random.default_rng(7)
    actions = [rng.uniform(-0.3, 0.3, size=12).astype(np.float32) for _ in range(30)]
    first, second = _noisy(stance_kwargs, robot_table, training=True), _noisy(stance_kwargs, robot_table, training=True)
    clean = species_registry.env_class(ROBOT)(**stance_kwargs)
    try:
        noisy_a, noisy_b, plain = (_rollout(env, seeds, actions) for env in (first, second, clean))
        assert noisy_a == noisy_b
        assert noisy_a[3] == noisy_a[0]  # reset(seed=7) again: the same episode, noise included
        for noised, unnoised in zip(noisy_a, plain):
            assert noised[:3] == unnoised[:3]  # the reset observation and info, and np_random after the reset
            for step in range(len(actions)):
                row = slice(3 + 5 * step, 8 + 5 * step)
                (obs_n, *rest_n), (obs_c, *rest_c) = noised[row], unnoised[row]
                assert rest_n == rest_c and obs_n != obs_c
        # The joint velocities' realised noise at each episode's first step (std 0.05, not a unit segment, so the
        # realisation is the draw itself up to the float32 rounding of the observation): neither unseeded episode
        # repeats an earlier episode's draw.  A stream rebuilt at every reset repeats it to within 1e-7 here.
        block = _segment_slice(first, "joint_velocity")
        realised = [
            (
                np.frombuffer(noised[3], dtype=np.float32).astype(np.float64)
                - np.frombuffer(unnoised[3], dtype=np.float32).astype(np.float64)
            )[block]
            for noised, unnoised in zip(noisy_a[:3], plain[:3])
        ]
        for a, b in ((0, 1), (1, 2), (0, 2)):
            assert np.max(np.abs(realised[a] - realised[b])) > 1e-3, (a, b)
    finally:
        for env in (first, second, clean):
            env.close()


def test_an_unseeded_reset_continues_the_stream_and_a_seeded_one_rebuilds_it(stance_kwargs, robot_table):
    """Design record §4.2 and §7: the noise's stream is rebuilt on a seeded reset and on the first, and continued
    across unseeded ones (SB3's auto-reset), as np_random is: each unseeded episode starts from the state the previous
    one ended in, never from the stream a reset of the env's seed would build (which an unseeded reset leaves
    unchanged), so no auto-reset episode replays another's noise.  An episode that is not noised draws nothing."""
    env = _noisy(stance_kwargs, robot_table, training=True)
    try:
        env.reset(seed=7)
        stream = env._observation_noise_rng
        assert stream.bit_generator.state == feature_stream(env, 7, OBSERVATION_NOISE_STREAM_KEY).bit_generator.state
        for _ in range(2):
            for _ in range(5):
                env.step(_zeros(env))
            ended = copy.deepcopy(stream.bit_generator.state)
            env.reset()
            assert env.np_random_seed == 7 and env._observation_noise_rng is stream
            assert stream.bit_generator.state == ended
            assert ended != feature_stream(env, None, OBSERVATION_NOISE_STREAM_KEY).bit_generator.state
        # Unmarked, the next unseeded episode is not noised: it continues the stream and draws nothing from it.
        mark_training_rollouts(env, False)
        env.step(_zeros(env))
        ended = copy.deepcopy(stream.bit_generator.state)
        env.reset()
        assert not env.observation_noise_active and env._observation_noise_rng is stream
        for _ in range(5):
            env.step(_zeros(env))
        assert stream.bit_generator.state == ended
        env.reset(seed=7)
        assert env._observation_noise_rng is not stream
        assert (
            env._observation_noise_rng.bit_generator.state
            == feature_stream(env, 7, OBSERVATION_NOISE_STREAM_KEY).bit_generator.state
        )
    finally:
        env.close()


def test_only_a_trainers_rollout_env_or_the_eval_flag_noises_an_episode_latched_at_reset(stance_kwargs, robot_table):
    """An evaluation env is noised only with observation_noise_eval; a trainer's rollout env always; a change of
    role takes effect at the next reset, for the whole episode."""
    plain = _noisy(stance_kwargs, robot_table)
    flagged = _noisy(stance_kwargs, robot_table, eval_noise=True)
    marked = _noisy(stance_kwargs, robot_table, training=True)
    try:
        assert [env.observation_noise_active for env in (plain, flagged, marked)] == [False, False, False]
        for env, active in ((plain, False), (flagged, True), (marked, True)):
            env.reset(seed=4)
            assert env.observation_noise_active is active
            obs, *_ = env.step(_zeros(env))
            assert np.array_equal(obs, env._get_obs()) is not active
        # Marked mid-episode: this episode stays clean, the next is noised; unmarked mid-episode, the reverse.
        mark_training_rollouts(plain, True)
        mark_training_rollouts(marked, False)
        for env, active in ((plain, False), (marked, True)):
            obs, *_ = env.step(_zeros(env))
            assert env.observation_noise_active is active
            assert np.array_equal(obs, env._get_obs()) is not active
            env.reset()
            assert env.observation_noise_active is not active
        mark_training_rollouts(flagged, False)
        flagged.reset()
        assert flagged.observation_noise_active  # the eval flag noises an evaluation env too
    finally:
        for env in (plain, flagged, marked):
            env.close()


def test_stepping_a_noisy_env_before_its_first_reset_is_refused(stance_kwargs, robot_table):
    env = _noisy(stance_kwargs, robot_table, training=True)
    try:
        with pytest.raises(RuntimeError, match=r"steps only after reset\(\)"):
            env.step(_zeros(env))
        env.reset(seed=1)
        env.step(_zeros(env))
        # Reconfiguring ends the episode's noise: the next episode begins it again.
        env._configure_observation_noise(robot_table, False)
        with pytest.raises(RuntimeError, match=r"steps only after reset\(\)"):
            env.step(_zeros(env))
    finally:
        env.close()


def test_the_noise_begins_third_in_the_hook_and_draws_from_its_own_stream(stance_kwargs, robot_table, monkeypatch):
    """Design record §4.1: _begin_episode begins the physics randomisation, then the actuation delay, then the noise;
    each draws from its own stream, so the noise stream's position is the same with the other two on or off."""
    from .actuation_delay_helpers import DelayedCompsognathusLongipesRobotEnv
    from .episode_physics_helpers import with_episode_physics

    every_class = observation_noise_helpers.with_observation_noise(
        with_episode_physics(DelayedCompsognathusLongipesRobotEnv)
    )
    order: list[Any] = []
    begin_physics = EpisodePhysicsRandomizer.begin_episode
    begin_delay, begin_noise = BaseDinoEnv._begin_actuation_delay_episode, BaseDinoEnv._begin_observation_noise_episode

    def physics(self, env, seed):
        order.append("physics")
        begin_physics(self, env, seed)

    def delay(self, seed):
        order.append("delay")
        begin_delay(self, seed)

    def noise(self, seed):
        order.append(("noise", self.episode_physics_factors is not None, self.actuation_delay_steps is not None))
        begin_noise(self, seed)

    monkeypatch.setattr(EpisodePhysicsRandomizer, "begin_episode", physics)
    monkeypatch.setattr(BaseDinoEnv, "_begin_actuation_delay_episode", delay)
    monkeypatch.setattr(BaseDinoEnv, "_begin_observation_noise_episode", noise)
    every = every_class(
        **stance_kwargs,
        actuation_delay_range_s=(0.0, 0.040),
        randomize_friction=(0.6, 1.0),
        randomize_body_mass_scale=(0.9, 1.1),
        observation_noise_std=robot_table,
    )
    alone = _noisy(stance_kwargs, robot_table)
    try:
        for env in (every, alone):
            mark_training_rollouts(env, True)
        for seed in (3, None, None):
            order.clear()
            every.reset(seed=seed)
            assert order == ["physics", "delay", ("noise", True, True)]
            alone.reset(seed=seed)
            for _ in range(5):
                every.step(_zeros(every))
                alone.step(_zeros(alone))
            assert every._observation_noise_rng.bit_generator.state == alone._observation_noise_rng.bit_generator.state
        expected = feature_stream(alone, 3, OBSERVATION_NOISE_STREAM_KEY)
        alone.reset(seed=3)
        assert alone._observation_noise_rng.bit_generator.state == expected.bit_generator.state
    finally:
        every.close()
        alone.close()


def test_a_stage_tomls_table_stays_a_plain_dict_in_the_kwargs_and_they_pickle(stance_kwargs, robot_table, tmp_path):
    """Design record §7: the table is an inline TOML table, kept a plain dict (SubprocVecEnv pickles the stage's
    kwargs).  Loaded from a stage TOML, set by ``--override`` into the declared table and handed through the policy
    probe bank's keywords, it is a dict each time, and the kwargs pickle to an env with the same noise."""
    from environments.shared.cli import _apply_overrides
    from environments.shared.policy_probe import bank_env_kwargs

    text = (REPOSITORY_ROOT / "configs" / "compsognathus_longipes" / "robot" / "stance.toml").read_text("utf-8")
    assert text.count("\n[env]\n") == 1
    inline = ", ".join(f"{name} = {std!r}" for name, std in robot_table.items())
    path = tmp_path / "stance.toml"
    path.write_text(text.replace("\n[env]\n", f"\n[env]\nobservation_noise_std = {{ {inline} }}\n"), "utf-8")
    config = load_stage_config(ROBOT, "stance", config_path=str(path))
    kwargs = config["env_kwargs"]
    assert type(kwargs["observation_noise_std"]) is dict and kwargs["observation_noise_std"] == robot_table
    assert {name: value for name, value in kwargs.items() if name != "observation_noise_std"} == stance_kwargs
    _apply_overrides({"stance": config}, ["env.observation_noise_std.joint_position=0.01"], ROBOT)
    table = kwargs["observation_noise_std"]
    assert type(table) is dict and table == {**robot_table, "joint_position": 0.01}
    bank = bank_env_kwargs({**kwargs, "observation_noise_eval": True}, 700)
    assert type(bank["observation_noise_std"]) is dict and bank["observation_noise_eval"] is False
    for each in (kwargs, bank):
        copied = pickle.loads(pickle.dumps(each))
        assert type(copied["observation_noise_std"]) is dict and copied == each
        built, unpickled = NOISY_ROBOT(**each), NOISY_ROBOT(**copied)
        try:
            assert built._observation_noise is not None
            assert built._observation_noise == unpickled._observation_noise
            assert built._observation_noise.std_by_segment == {
                name: table.get(name, 0.0) for name, _ in observation_layout(built)
            }
        finally:
            built.close()
            unpickled.close()


def test_a_noisy_env_pickles_and_its_copies_continue_bit_identically(stance_kwargs, robot_table):
    """The noise pickles with the env; a pickled and a deep-copied env continue mid-episode, the noise's stream and
    latch included, bit for bit."""
    env = _noisy(stance_kwargs, robot_table, training=True)
    twins = []
    try:
        env.reset(seed=21)
        rng = np.random.default_rng(21)
        for _ in range(5):
            env.step(rng.uniform(-0.3, 0.3, size=12))
        twins = [copy.deepcopy(env), pickle.loads(pickle.dumps(env))]
        actions = [rng.uniform(-0.3, 0.3, size=12) for _ in range(10)]
        rows = []
        for each in (env, *twins):
            record = [each._observation_noise == env._observation_noise, each.observation_noise_active]
            for action in actions:
                obs, reward, *_ = each.step(action)
                record += [obs.tobytes(), each.data.qpos.tobytes(), np.float64(reward).tobytes()]
            each.reset()
            obs, *_ = each.step(actions[0])
            record.append(obs.tobytes())
            rows.append(record)
        assert rows[0] == rows[1] == rows[2]
    finally:
        for each in (env, *twins):
            each.close()


def test_a_behavior_env_noises_its_step_observations_and_never_its_reset(opted_in_robot, robot_table):
    """The behaviour env of an opted-in class takes the keys through to it: its reset observation is clean, its
    step observations noised, its rewards and infos those of the clean behaviour env."""
    from environments.shared.behavior_env import get_behavior_env_class

    behavior_class = get_behavior_env_class(ROBOT)
    assert NOISY_ROBOT in behavior_class.__mro__
    noisy = behavior_class(observation_noise_std=robot_table, observation_noise_eval=True)
    clean = behavior_class()
    try:
        assert observation_schema(noisy) == "bipedal-heading-free/v1"
        rows = []
        for env in (noisy, clean):
            obs, info = env.reset(seed=8)
            record = [obs.tobytes(), pickle.dumps(info)]
            steps = []
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(_zeros(env))
                record.append(pickle.dumps((reward, terminated, truncated, info)))
                steps.append(obs.tobytes())
            rows.append((record, steps))
        assert rows[0][0] == rows[1][0]
        assert all(a != b for a, b in zip(rows[0][1], rows[1][1]))
    finally:
        noisy.close()
        clean.close()


def _record_stand_block(env) -> list[dict[str, float]]:
    """Shadow *env*'s ``_stand_block`` with one that records the weights of every stand block ``step()`` holds."""
    held: list[dict[str, float]] = []
    original = env._stand_block

    @contextlib.contextmanager
    def recording():
        with original() as weights:
            held.append(dict(weights))
            yield weights

    env._stand_block = recording
    return held


@pytest.mark.parametrize(
    ("kind", "speed"),
    [
        pytest.param("behavior", 0.04, id="behavior"),
        pytest.param("walk", 0.04, id="walk"),
        pytest.param("walk", 0.0, id="walk-stand"),
    ],
)
def test_set_direction_noises_the_observation_it_returns_exactly_when_the_episode_is_noised(
    opted_in_robot, robot_table, kind, speed
):
    """BaseDinoEnv.set_direction, which a behaviour env and a live walk share, returns the observation the policy
    acts on next: in a noised episode it is _get_obs() plus one draw of the noise's stream, as step()'s is (its
    command exact: never noised); otherwise it is the very array _get_obs built, and the stream is not drawn.  The
    step after it is noised the same way, a stand-block step included: on the walk, whose stage declares stand
    weights, a zero speed is a live zero command, so step() holds the stand block (main's decision D-D62) around
    the step body that noises."""
    from environments.shared.behavior_env import get_behavior_env_class
    from environments.shared.direction_commands import DirectionCommandConfig

    if kind == "behavior":
        behavior_class = get_behavior_env_class(ROBOT)
        commands = DirectionCommandConfig(cruise_speed=0.04, speed_scale=0.06)
        noisy = behavior_class(commands=commands, observation_noise_std=robot_table, observation_noise_eval=True)
        quiet = behavior_class(commands=commands, observation_noise_std=robot_table)
    else:
        # The walk follows a live command on every model (main's decision D-D62).
        walk = dict(load_stage_config(ROBOT, "locomotion")["env_kwargs"])
        assert walk["command_mode"] != "none"
        noisy = NOISY_ROBOT(**walk, observation_noise_std=robot_table, observation_noise_eval=True)
        quiet = NOISY_ROBOT(**walk, observation_noise_std=robot_table)

    def check(env, active, obs, clean, state) -> None:
        """*obs* is *clean* plus exactly one draw of the stream from *state* when *active*, else *clean* itself."""
        if active:
            replay = np.random.Generator(np.random.PCG64())
            replay.bit_generator.state = state
            np.testing.assert_array_equal(obs, env._observation_noise.apply(clean, replay))
            assert not np.array_equal(obs, clean)
            assert env._observation_noise_rng.bit_generator.state == replay.bit_generator.state
        else:
            assert obs is clean and env._observation_noise_rng.bit_generator.state == state

    try:
        for env, active in ((noisy, True), (quiet, False)):
            built = _record_get_obs(env)
            env.reset(seed=9)
            assert env.observation_noise_active is active
            env.step(_zeros(env))
            state = copy.deepcopy(env._observation_noise_rng.bit_generator.state)
            obs = env.set_direction(env._heading() + np.pi / 4, speed)
            np.testing.assert_array_equal(obs[-3:], env._command_state.normalized)
            check(env, active, obs, built[-1], state)
            stand = kind == "walk" and speed == 0.0
            assert env._live_zero_command() == stand and bool(env._stand_block_weights()) == stand
            held = _record_stand_block(env)
            state = copy.deepcopy(env._observation_noise_rng.bit_generator.state)
            obs, *_ = env.step(_zeros(env))
            check(env, active, obs, built[-1], state)
            assert held == ([env._stand_weights] if stand else [])
    finally:
        noisy.close()
        quiet.close()


def test_the_policy_probe_bank_holds_the_noise_off_only_where_the_stage_declares_it(robot_table):
    """bank_env_kwargs sets observation_noise_eval False exactly where the stage declares it, as it sets the push
    multiple to 0; the bank env is never marked, so its observations are the clean ones."""
    from environments.shared.policy_probe import OBSERVATION_NOISE_EVAL_KWARG, PUSH_MULTIPLE_KWARG, bank_env_kwargs

    recovery = load_stage_config(ROBOT, "recovery")["env_kwargs"]
    assert OBSERVATION_NOISE_EVAL_KWARG not in recovery and recovery[PUSH_MULTIPLE_KWARG] > 0.0
    assert bank_env_kwargs(recovery, 700) == {**recovery, "max_episode_steps": 700, PUSH_MULTIPLE_KWARG: 0.0}
    declared = {**recovery, "observation_noise_std": robot_table, OBSERVATION_NOISE_EVAL_KWARG: True}
    kwargs = bank_env_kwargs(declared, 700)
    assert kwargs == {
        **declared,
        "max_episode_steps": 700,
        PUSH_MULTIPLE_KWARG: 0.0,
        OBSERVATION_NOISE_EVAL_KWARG: False,
    }
    assert declared[OBSERVATION_NOISE_EVAL_KWARG] is True  # the stage's own mapping is untouched
    bank = NOISY_ROBOT(**kwargs)
    try:
        built = _record_get_obs(bank)
        bank.reset(seed=1)
        assert bank._observation_noise is not None and not bank.observation_noise_active
        obs, *_ = bank.step(_zeros(bank))
        assert obs is built[-1]
    finally:
        bank.close()


def test_plant_validation_and_the_plant_probe_stay_clean_on_a_noised_env(stance_kwargs, robot_table):
    """The noise never touches the model or the hashed methods: a noised env mid-episode validates, and its plant
    layers (the observation probe among them) are the noise-free env's."""
    env = _noisy(stance_kwargs, robot_table, training=True)
    off = species_registry.env_class(ROBOT)(**stance_kwargs)
    try:
        env.reset(seed=6)
        rng = np.random.default_rng(6)
        for _ in range(20):
            env.step(rng.uniform(-0.3, 0.3, size=12))
        assert env.observation_noise_active
        validate_environment_plant(env, current_plant_identity(ROBOT), artifact="noised robot mid-episode")
        version = load_plant_versions()[1][ROBOT]
        assert fingerprint_model_layers(env.model, env, version) == fingerprint_model_layers(off.model, off, version)
        assert np.array_equal(
            policy_layer._observation_probe(env.model, env)["values"],
            policy_layer._observation_probe(off.model, off)["values"],
        )
    finally:
        env.close()
        off.close()


# ---------------------------------------------------------------------------------------------------- the fingerprint
@pytest.mark.parametrize(("model_id", "stage"), MODEL_STAGES)
def test_a_committed_stage_carries_the_keys_and_the_section_only_where_it_opts_in(model_id, stage):
    env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
    effective = task_fingerprint._effective_env_kwargs(model_id, env_kwargs)
    fingerprint = stage_task_fingerprint(model_id, stage)
    on = effective.get("observation_noise_std") is not None
    if model_id not in OPTED_IN_OBSERVATION_NOISE or not on:
        assert not set(OBSERVATION_NOISE_ENV_KEYS) & set(effective)
    assert ("observation_noise" in fingerprint) is on
    if on:
        assert fingerprint["observation_noise"] == {"implementation": OBSERVATION_NOISE_IMPLEMENTATION}


def test_the_section_is_absent_while_the_committed_classes_take_no_key():
    fingerprint = compute_task_fingerprint(
        model_id=ROBOT,
        stage="stance",
        backend="stable-baselines3",
        env_kwargs={"alive_bonus": 1.0},
        plant_identity=None,
        perturbation_manifest=None,
    )
    assert "observation_noise" not in fingerprint and not set(OBSERVATION_NOISE_ENV_KEYS) & set(fingerprint["env"])


@pytest.fixture
def opted_in_robot(monkeypatch):
    """The registry resolving the Compsognathus robot to its opted-in test subclass (carve-out at off value None).

    The behavior env classes are cached per model, so the cache is emptied on both sides of the test.
    """
    from environments.shared.behavior_env import _behavior_env_class

    identity = current_plant_identity(ROBOT)
    real = species_registry.env_class
    monkeypatch.setattr(
        species_registry, "env_class", lambda model_id: NOISY_ROBOT if model_id == ROBOT else real(model_id)
    )
    _behavior_env_class.cache_clear()
    yield identity
    _behavior_env_class.cache_clear()


def test_the_section_carries_the_tag_while_the_table_is_set_and_names_what_moved(
    opted_in_robot, robot_table, monkeypatch
):
    """The table writes the section and hashes both keys; off, the carve-out drops both, so an opted-in class keeps
    the committed task; a changed std or flag names env, switching off names env and observation_noise, a tag bump
    names observation_noise alone."""
    stance = load_stage_config(ROBOT, "stance")["env_kwargs"]

    def fingerprint(**noise):
        return stage_task_fingerprint(ROBOT, "stance", env_kwargs={**stance, **noise}, plant_identity=opted_in_robot)

    on = fingerprint(observation_noise_std=robot_table, observation_noise_eval=True)
    assert on["observation_noise"] == {"implementation": OBSERVATION_NOISE_IMPLEMENTATION}
    assert on["env"]["observation_noise_std"] == robot_table and on["env"]["observation_noise_eval"] is True
    off = fingerprint()
    assert "observation_noise" not in off and not set(OBSERVATION_NOISE_ENV_KEYS) & set(off["env"])
    assert off["task_sha256"] == stage_task_fingerprint(ROBOT, "stance", plant_identity=opted_in_robot)["task_sha256"]
    changed_std = fingerprint(
        observation_noise_std={**robot_table, "joint_position": 0.01}, observation_noise_eval=True
    )
    changed_flag = fingerprint(observation_noise_std=robot_table, observation_noise_eval=False)
    monkeypatch.setattr(
        task_fingerprint, "OBSERVATION_NOISE_IMPLEMENTATION", OBSERVATION_NOISE_IMPLEMENTATION + "+next"
    )
    bumped = fingerprint(observation_noise_std=robot_table, observation_noise_eval=True)
    for current, sections in (
        (changed_std, ["env"]),
        (changed_flag, ["env"]),
        (off, ["env", "observation_noise"]),
        (bumped, ["observation_noise"]),
    ):
        with pytest.raises(TaskFingerprintError, match=re.escape(f"differing sections: {sections!r}")):
            validate_recorded_task(on, current, mode="resume_same_stage")
    assert validate_recorded_task(on, on, mode="resume_same_stage") is None


def test_an_existing_species_opting_in_keeps_every_committed_robot_stage_digest(opted_in_robot, monkeypatch):
    """An opted-in class whose stages omit the keys moves no stage's task_sha256 or stage_config.json view."""
    from environments.shared.harnesses import digest_snapshot

    assert set(OBSERVATION_NOISE_ENV_KEYS) <= set(inspect.signature(species_registry.env_class(ROBOT)).parameters)
    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.setattr(digest_snapshot, "_models", lambda: [ROBOT])
    out = digest_snapshot._Snapshot(REPOSITORY_ROOT, keep=True)
    digest_snapshot.stage_section(out, {ROBOT: opted_in_robot})
    golden = (REPOSITORY_ROOT / "configs" / "digest_snapshot.generated.txt").read_text(encoding="utf-8").splitlines()
    prefix = f"stage\t{ROBOT.species}\t{ROBOT.embodiment}\t"
    committed = [line for line in golden if line.startswith(prefix)]
    assert out.errors == 0 and len(committed) == 21  # stance, recovery and locomotion, 7 lines each
    assert out.kept == committed


def test_an_existing_species_opting_in_keeps_every_committed_robot_behavior_digest(opted_in_robot, monkeypatch):
    from environments.shared.behavior_env import get_behavior_env_class
    from environments.shared.harnesses import digest_snapshot

    monkeypatch.chdir(REPOSITORY_ROOT)
    assert NOISY_ROBOT in get_behavior_env_class(ROBOT).__mro__
    golden = (REPOSITORY_ROOT / "configs" / "digest_snapshot.generated.txt").read_text(encoding="utf-8").splitlines()
    prefix = f"behavior\t{ROBOT.species}\t{ROBOT.embodiment}\t"
    committed = {
        line.split("\t")[3]: line.split("\t")[5]
        for line in golden
        if line.startswith(prefix) and "\ttask_sha256\t" in line
    }
    recipes = digest_snapshot._recipes(ROBOT)
    assert len(committed) == len(recipes) == 11
    for path in recipes:
        assert digest_snapshot._behavior_task(path, ROBOT)["task_sha256"] == committed[path.stem], path.stem
