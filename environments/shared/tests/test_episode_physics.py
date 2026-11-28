"""The opt-in per-episode physics randomisation: ``environments/shared/episode_physics.py`` and its hook in ``BaseDinoEnv``.

Design record ``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §6.  No committed
model lists the ``randomize_*`` keys, so the randomisation itself runs on
test-only subclasses of every manifest model that opt in
(``episode_physics_helpers.py``).  Numpy and MuJoCo only: no SB3 (the
behaviour-transition check of the ``episode_physics`` section is in
``test_behavior_checkpoint.py``).
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
import pickle
import re
from typing import Any

import mujoco
import numpy as np
import pytest

from environments.shared import species_registry, task_fingerprint
from environments.shared.actuation_delay import ACTUATION_DELAY_ENV_KEYS
from environments.shared.base_env import BaseDinoEnv
from environments.shared.config import load_stage_config
from environments.shared.episode_physics import (
    EPISODE_PHYSICS_ENV_KEYS,
    EPISODE_PHYSICS_IMPLEMENTATION,
    GEOMETRIC_STAT_FIELDS,
    RANDOMIZED_FIELDS,
    SETCONST_FIELDS,
    STAT_FIELDS,
    TRAINING_ONLY_ENV_KEYS,
    EpisodePhysicsConfig,
    EpisodePhysicsRandomizer,
    animal_bodies,
    colliding_animal_geoms,
    force_limited_actuators,
    payload_body,
    position_servos,
    validate_episode_physics,
)
from environments.shared.episode_start import EPISODE_PHYSICS_STREAM_KEY, feature_stream, mark_training_rollouts
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.plant_contract import (
    PlantCompatibilityError,
    current_plant_identity,
    validate_compiled_plant,
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

from . import episode_physics_helpers, reset_golden
from .episode_physics_helpers import EPISODE_PHYSICS_CARVE_OUTS, RANDOMIZED_ENV_CLASSES

ROBOT = ModelId("compsognathus_longipes", "robot")
ALLOSAURUS = ModelId("allosaurus_fragilis", "anatomical")
RANDOMIZED_ROBOT = RANDOMIZED_ENV_CLASSES[ROBOT]

#: The manifest models that opt into the randomisation: each lists the six keys, defaulting to ``None``, forwards
#: them to ``BaseDinoEnv`` and declares one carve-out per key at off value ``None``.  Empty until the Triceratops robot
#: joins it (design record §8.9); every other model refuses the keys.
OPTED_IN_EPISODE_PHYSICS: frozenset[ModelId] = frozenset()

#: The build's per-body ranges and D38's whole-robot ones (design record §6), the payload scaled per model below.
ROBOT_RANGES = {
    "randomize_friction": (0.6, 1.0),
    "randomize_body_mass_scale": (0.9, 1.1),
    "randomize_actuator_kp_scale": (0.9, 1.1),
    "randomize_actuator_force_limit_scale": (0.85, 1.0),
    "randomize_total_mass_scale": (0.9, 1.1),
    "randomize_payload_kg": (0.0, 1.0),
}

MODEL_IDS = model_ids()
MODEL_STAGES = [
    pytest.param(model_id, entry.reference, id=f"{model_id.species}-{model_id.embodiment}-{entry.id}")
    for model_id in MODEL_IDS
    for entry in load_stage_manifest(model_id).stages
]


def _first_stage_kwargs(model_id: ModelId) -> dict[str, Any]:
    return dict(load_stage_config(model_id, load_stage_manifest(model_id).stages[0].reference)["env_kwargs"])


def _animal_mass(model: mujoco.MjModel) -> float:
    return float(model.body_mass[animal_bodies(model)].sum())


@functools.cache
def _model_ranges(model_id: ModelId) -> tuple[tuple[str, tuple[float, float]], ...]:
    env = species_registry.env_class(model_id)()
    try:
        model = env.model
        ranges = dict(ROBOT_RANGES, randomize_payload_kg=(0.0, 0.1 * _animal_mass(model)))
        if not position_servos(model).size:
            ranges.pop("randomize_actuator_kp_scale")
        if not force_limited_actuators(model).size:
            ranges.pop("randomize_actuator_force_limit_scale")
    finally:
        env.close()
    return tuple(ranges.items())


def _all_ranges(model_id: ModelId) -> dict[str, tuple[float, float]]:
    """Every key the model can take: kp and force only where it has servos and force limits (not Allosaurus); a
    payload of up to a tenth of the animal."""
    return dict(_model_ranges(model_id))


def _randomized(model_id: ModelId, env_kwargs: dict[str, Any] | None = None, *, training: bool = True, **ranges: Any):
    env_kwargs = _first_stage_kwargs(model_id) if env_kwargs is None else env_kwargs
    env = RANDOMIZED_ENV_CLASSES[model_id](**{**env_kwargs, **ranges})
    mark_training_rollouts(env, training)
    return env


def _model_arrays(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Every ndarray field of *model*, copied, and its five statistics."""
    arrays = {
        name: np.array(getattr(model, name))
        for name in dir(model)
        if not name.startswith("_") and isinstance(getattr(model, name), np.ndarray)
    }
    arrays.update({f"stat.{name}": np.array(getattr(model.stat, name), dtype=np.float64) for name in STAT_FIELDS})
    return arrays


def _changed(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> set[str]:
    assert a.keys() == b.keys()
    return {name for name in a if not np.array_equal(a[name], b[name], equal_nan=a[name].dtype.kind in "fc")}


MANAGED = set(RANDOMIZED_FIELDS) | set(SETCONST_FIELDS) | {f"stat.{name}" for name in STAT_FIELDS}


# ---------------------------------------------------------------------------------------------------- the module
def test_the_keys_the_fields_and_the_tag():
    assert EPISODE_PHYSICS_ENV_KEYS == tuple(ROBOT_RANGES)
    assert TRAINING_ONLY_ENV_KEYS == ("randomize_total_mass_scale", "randomize_payload_kg")
    assert EPISODE_PHYSICS_IMPLEMENTATION == "mesozoic.episode-physics-randomization/v1"
    assert RANDOMIZED_FIELDS == (
        "geom_friction",
        "body_mass",
        "body_inertia",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
    )
    assert set(GEOMETRIC_STAT_FIELDS) < set(STAT_FIELDS)


def test_every_key_off_is_none_and_a_live_range_is_a_float_pair():
    assert validate_episode_physics() is None
    assert validate_episode_physics(**dict.fromkeys(EPISODE_PHYSICS_ENV_KEYS)) is None
    config = validate_episode_physics(randomize_friction=[0.6, 1], randomize_payload_kg=(0, 0))
    assert config == EpisodePhysicsConfig(randomize_friction=(0.6, 1.0), randomize_payload_kg=(0.0, 0.0))
    assert all(isinstance(end, float) for end in config.randomize_friction + config.randomize_payload_kg)
    # A degenerate range is one value; numpy scalars are numbers.
    assert validate_episode_physics(randomize_body_mass_scale=(np.float64(1.1), 1.1)).randomize_body_mass_scale == (
        1.1,
        1.1,
    )
    with pytest.raises(TypeError, match="unknown episode physics keys: randomize_gravity"):
        validate_episode_physics(randomize_gravity=(0.9, 1.1))


#: (key, range, the refusal): every way a range is refused.
INVALID_RANGES = [
    pytest.param("randomize_friction", (0.0, 1.0), r"0 < low <= high", id="friction low zero"),
    pytest.param("randomize_friction", (1.0, 0.5), r"0 < low <= high", id="friction low above high"),
    pytest.param("randomize_friction", (float("nan"), 1), "finite", id="friction not finite"),
    pytest.param("randomize_friction", (0.6, 0.8, 1.0), "a pair", id="friction three values"),
    pytest.param("randomize_friction", "0.6, 1.0", "a pair", id="friction a string"),
    pytest.param("randomize_friction", 0.8, "a pair", id="friction one value"),
    pytest.param("randomize_friction", (True, 1.0), "a pair of numbers", id="friction a bool"),
    pytest.param("randomize_body_mass_scale", (-0.9, 1.1), r"0 < low <= high", id="mass negative"),
    pytest.param("randomize_actuator_kp_scale", (0.9, float("inf")), "finite", id="kp infinite"),
    pytest.param("randomize_actuator_force_limit_scale", (0.0, 1.0), r"0 < low <= high", id="force low zero"),
    pytest.param("randomize_total_mass_scale", (1.1, 0.9), r"0 < low <= high", id="total low above high"),
    pytest.param("randomize_payload_kg", (-0.1, 0.0), r"0 <= low <= high", id="payload negative"),
    pytest.param("randomize_payload_kg", (0.5, "1"), "a pair of numbers", id="payload a string end"),
]


@pytest.mark.parametrize(("key", "value", "message"), INVALID_RANGES)
def test_an_invalid_range_is_refused(key, value, message):
    with pytest.raises(ValueError, match=f"{key} must be.*{message}"):
        validate_episode_physics(**{key: value})


@pytest.mark.parametrize(("key", "value", "message"), INVALID_RANGES[:7] + INVALID_RANGES[-2:])
def test_an_invalid_range_is_refused_at_construction(key, value, message):
    with pytest.raises(ValueError, match=message):
        RANDOMIZED_ROBOT(**{key: value})


def test_the_payload_body_is_the_root_or_the_heaviest_welded_body_as_the_push_target():
    for model_id in MODEL_IDS:
        env = species_registry.env_class(model_id)()
        try:
            model, root = env.model, int(animal_bodies(env.model)[0])
            assert root == env._root_body_id
            body = payload_body(model)
            if model.body_mass[root] > 0:
                assert body == root
            else:
                welded = np.flatnonzero(model.body_weldid == root)
                assert body in welded and model.body_mass[body] == model.body_mass[welded].max() > 0
        finally:
            env.close()


# ---------------------------------------------------------------------------------------------------- off by default
def test_the_base_env_takes_the_six_keys_after_the_delay_keys_off_by_default():
    """Right after the delay's two keys, which follow command_config; the observation noise's keys follow them
    (test_observation_noise.py)."""
    parameters = list(inspect.signature(BaseDinoEnv.__init__).parameters.values())
    names = [parameter.name for parameter in parameters]
    start = names.index("command_config") + 1
    assert names[start : start + 8] == [*ACTUATION_DELAY_ENV_KEYS, *EPISODE_PHYSICS_ENV_KEYS]
    assert all(parameter.default is None for parameter in parameters[start + 2 : start + 8])
    assert BaseDinoEnv._episode_physics is None


def test_the_models_listing_a_key_are_exactly_the_opted_in_table():
    assert OPTED_IN_EPISODE_PHYSICS <= set(MODEL_IDS)
    listing = {
        model_id
        for model_id in MODEL_IDS
        if set(EPISODE_PHYSICS_ENV_KEYS) & set(inspect.signature(species_registry.env_class(model_id)).parameters)
    }
    assert listing == OPTED_IN_EPISODE_PHYSICS


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_an_opted_in_model_takes_every_key_off_by_default_and_every_other_refuses_them(model_id):
    """An opted-in class lists the six keys at default None, forwards them to BaseDinoEnv and declares one carve-out
    per key; a TOML setting any key on any other model raises TypeError."""
    env_class = species_registry.env_class(model_id)
    parameters = inspect.signature(env_class).parameters
    if model_id in OPTED_IN_EPISODE_PHYSICS:
        assert all(parameters[key].default is None for key in EPISODE_PHYSICS_ENV_KEYS)
        assert set(EPISODE_PHYSICS_CARVE_OUTS) <= set(getattr(env_class, "OPT_IN_TASK_KWARGS", ()))
        env = env_class(randomize_friction=(0.8, 0.8))
        try:
            assert env._episode_physics is not None and env._episode_physics.config.randomize_friction == (0.8, 0.8)
        finally:
            env.close()
        return
    assert not set(EPISODE_PHYSICS_ENV_KEYS) & set(parameters)
    for key, value in ROBOT_RANGES.items():
        with pytest.raises(TypeError, match=key):
            env_class(**{key: value})


def test_the_test_subclasses_opt_in_as_a_model_would():
    assert RANDOMIZED_ENV_CLASSES.keys() == set(MODEL_IDS)
    for model_id, env_class in RANDOMIZED_ENV_CLASSES.items():
        canonical = species_registry.env_class(model_id)
        assert env_class.__mro__[1] is canonical and env_class.__name__ == f"Randomized{canonical.__name__}"
        parameters = inspect.signature(env_class).parameters
        assert list(parameters)[-6:] == list(EPISODE_PHYSICS_ENV_KEYS)
        assert all(parameters[key].default is None for key in EPISODE_PHYSICS_ENV_KEYS)
        assert env_class.OPT_IN_TASK_KWARGS[-6:] == EPISODE_PHYSICS_CARVE_OUTS
        assert getattr(episode_physics_helpers, env_class.__qualname__) is env_class
        assert pickle.loads(pickle.dumps(env_class)) is env_class
        assert episode_physics_helpers.with_episode_physics(canonical) is env_class


@pytest.mark.parametrize(("model_id", "stage"), MODEL_STAGES)
def test_randomisation_is_off_exactly_when_the_stage_omits_every_key(model_id, stage):
    env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
    env = species_registry.env_class(model_id)(**env_kwargs)
    try:
        listed = bool(set(EPISODE_PHYSICS_ENV_KEYS) & set(env_kwargs))
        assert (env._episode_physics is None) == (not listed)
        if env._episode_physics is None:
            assert env.episode_physics_factors is None
            before = _model_arrays(env.model)
            env.reset(seed=1)
            with env.nominal_physics():
                pass
            env._restore_episode_physics()
            assert env.episode_physics_factors is None
            assert _changed(before, _model_arrays(env.model)) == set()
    finally:
        env.close()


GOLDEN = reset_golden.load_fixture()
GOLDEN_KEYS = [
    pytest.param(ModelId(species, embodiment), stage, id=f"{species}-{embodiment}-{stage}")
    for species, embodiments in GOLDEN["captures"].items()
    for embodiment, stages in embodiments.items()
    for stage in stages
]


@pytest.mark.parametrize(("model_id", "stage"), GOLDEN_KEYS)
def test_an_off_reset_never_reaches_the_randomiser_and_keeps_the_golden_reset(model_id, stage, monkeypatch):
    """With every key off neither a reset nor a behaviour env's model selection calls the randomiser, and the
    seeded and unseeded reset records equal the Phase C golden's (tests/fixtures/phase_c_reset_golden.json)."""

    def refuse(*args, **kwargs):
        raise AssertionError("a randomisation that is off was reached")

    for name in ("begin_episode", "restore_all", "nominal", "refuse_inside_nominal", "__init__"):
        monkeypatch.setattr(EpisodePhysicsRandomizer, name, refuse)
    golden = GOLDEN["captures"][model_id.species][model_id.embodiment][stage]
    current = reset_golden.capture(model_id, stage, [int(seed) for seed in golden["seeds"]], int(golden["steps"]))
    assert current["seeds"].keys() == golden["seeds"].keys()
    for seed, record in golden["seeds"].items():
        assert current["seeds"][seed]["reset"] == record["reset"], seed
        assert current["seeds"][seed]["second_reset"] == record["second_reset"], seed
    from environments.shared.behavior_env import get_behavior_env_class

    behavior = get_behavior_env_class(model_id)()
    try:
        behavior.reset(seed=3)
        behavior.reset()
        assert behavior._episode_physics is None
    finally:
        behavior.close()


def test_the_physics_begins_first_in_the_hook_and_moves_no_other_features_draw(monkeypatch):
    """Design record §4.1: _begin_episode begins the randomisation before the actuation delay, so the delay's episode
    starts on this episode's plant; each draws from its own stream, so the delay drawn is the same with the
    randomisation on or off."""
    from .actuation_delay_helpers import DelayedCompsognathusLongipesRobotEnv

    both_class = episode_physics_helpers.with_episode_physics(DelayedCompsognathusLongipesRobotEnv)
    order = []
    begin_physics, begin_delay = EpisodePhysicsRandomizer.begin_episode, BaseDinoEnv._begin_actuation_delay_episode

    def physics(self, env, seed):
        order.append("physics")
        begin_physics(self, env, seed)

    def delay(self, seed):
        order.append(("delay", self.episode_physics_factors is not None))
        begin_delay(self, seed)

    monkeypatch.setattr(EpisodePhysicsRandomizer, "begin_episode", physics)
    monkeypatch.setattr(BaseDinoEnv, "_begin_actuation_delay_episode", delay)
    both = both_class(actuation_delay_range_s=(0.0, 0.040), **_all_ranges(ROBOT))
    delayed = DelayedCompsognathusLongipesRobotEnv(actuation_delay_range_s=(0.0, 0.040))
    try:
        for env in (both, delayed):
            mark_training_rollouts(env, True)
        ks = []
        for seed in (3, None, None):
            order.clear()
            both.reset(seed=seed)
            assert order == ["physics", ("delay", True)]
            delayed.reset(seed=seed)
            ks.append((both.actuation_delay_steps, delayed.actuation_delay_steps))
        assert all(a == b for a, b in ks) and len({a for a, _ in ks}) > 1
    finally:
        both.close()
        delayed.close()


# ---------------------------------------------------------------------------------------------------- on every model
def _reset_view(env) -> list[bytes]:
    data = env.data
    view = [data.qpos.tobytes(), data.qvel.tobytes(), data.mocap_pos.tobytes(), data.mocap_quat.tobytes()]
    for name in ("_push_schedule_starts", "_push_schedule_directions"):
        value = getattr(env, name, None)
        view.append(b"" if value is None else np.asarray(value).tobytes())
    return view + [pickle.dumps(env.np_random.bit_generator.state)]


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_a_randomised_reset_keeps_every_reset_draw_and_changes_the_plant(model_id):
    """Every applicable key on, in a training env: each reset's qpos, qvel, push schedule and np_random state (and its
    mocap arrays, empty on every current plant) equal the off env's, seeded and unseeded, while the managed fields
    differ."""
    env_kwargs = _first_stage_kwargs(model_id)
    on = _randomized(model_id, env_kwargs, **_all_ranges(model_id))
    off = species_registry.env_class(model_id)(**env_kwargs)
    try:
        rng = np.random.default_rng(0)
        for seed in (11, None, None, 12, None):
            on.reset(seed=seed)
            off.reset(seed=seed)
            assert _reset_view(on) == _reset_view(off)
            changed = _changed(_model_arrays(off.model), _model_arrays(on.model))
            assert {"geom_friction", "body_mass", "body_inertia", "body_subtreemass"} <= changed <= MANAGED, changed
            for _ in range(3):
                action = rng.uniform(-0.2, 0.2, size=on.action_space.shape)
                on.step(action)
                off.step(action)
    finally:
        on.close()
        off.close()


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_after_a_restore_every_model_array_equals_an_off_envs_exactly(model_id):
    """After randomised episodes, a restore (and a nominal_physics() block) leaves every ndarray of env.model and its
    five statistics exactly those of a randomisation-off env of the same kwargs driven the same way; the block's exit
    writes the episode's values back exactly."""
    env_kwargs = _first_stage_kwargs(model_id)
    on = _randomized(model_id, env_kwargs, **_all_ranges(model_id))
    off = species_registry.env_class(model_id)(**env_kwargs)
    try:
        rng = np.random.default_rng(1)
        for seed in (21, None, 22):
            on.reset(seed=seed)
            off.reset(seed=seed)
            for _ in range(4):
                action = rng.uniform(-0.2, 0.2, size=on.action_space.shape)
                on.step(action)
                off.step(action)
        nominal = _model_arrays(off.model)
        live = _model_arrays(on.model)
        assert _changed(nominal, live)
        with on.nominal_physics():
            assert _changed(nominal, _model_arrays(on.model)) == set()
        assert _changed(live, _model_arrays(on.model)) == set()
        on._restore_episode_physics()
        assert _changed(nominal, _model_arrays(on.model)) == set()
        with on.nominal_physics():  # nothing randomised now: the block changes nothing
            pass
        assert _changed(nominal, _model_arrays(on.model)) == set()
    finally:
        on.close()
        off.close()


def test_a_species_edit_after_its_constructor_is_part_of_the_nominal_plant():
    """The record is taken at a model's first episode, so a managed field a species writes after super().__init__()
    is what every restore returns to (design record R16)."""

    class EditedRobot(RANDOMIZED_ROBOT):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model.body_mass[self._root_body_id + 1] *= 2.0
            self.model.geom_friction[1, 1] = 0.123

    env = EditedRobot(**_all_ranges(ROBOT))
    try:
        mark_training_rollouts(env, True)
        edited = _model_arrays(env.model)
        for seed in (1, None, 2):
            env.reset(seed=seed)
            assert _changed(edited, _model_arrays(env.model))
        env._restore_episode_physics()
        assert _changed(edited, _model_arrays(env.model)) == set()
        assert env.model.geom_friction[1, 1] == 0.123
    finally:
        env.close()


def test_pushes_stay_in_newtons_of_the_nominal_plant():
    """On the Compsognathus robot's recovery stage the derived push force and the perturbation manifest are the off
    env's, and every pushed step writes the same force."""
    recovery = dict(load_stage_config(ROBOT, "recovery")["env_kwargs"])
    on = _randomized(ROBOT, recovery, **_all_ranges(ROBOT))
    off = species_registry.env_class(ROBOT)(**recovery)
    try:
        assert on._push_force_n == off._push_force_n > 0
        assert on.perturbation_manifest() == off.perturbation_manifest()
        pushed = 0
        for env in (on, off):
            env.reset(seed=3042)
        assert on.episode_physics_factors["total_mass_scale"] != 1.0
        for _ in range(250):
            for env in (on, off):
                env.step(np.zeros(env.action_space.shape))
            assert np.array_equal(on.data.xfrc_applied, off.data.xfrc_applied)
            pushed += bool(np.any(on.data.xfrc_applied))
        assert pushed > 0
        assert on.perturbation_manifest() == off.perturbation_manifest()
    finally:
        on.close()
        off.close()


def test_a_randomised_env_copies_and_pickles_with_its_own_records():
    """A deep copy and a pickle keep each record bound to the copy's own model, and continue bit-identically."""
    env = _randomized(ROBOT, **_all_ranges(ROBOT))
    twins = []
    try:
        env.reset(seed=5)
        for _ in range(3):
            env.step(np.zeros(12))
        twins = [copy.deepcopy(env), pickle.loads(pickle.dumps(env))]
        rows = []
        for each in (env, *twins):
            (record,) = each._episode_physics._records
            assert record.model is each.model and record.randomized
            row = []
            for seed in (None, 6, None):
                each.reset(seed=seed)
                row += [each.model.body_mass.tobytes(), each.model.geom_friction.tobytes(), each.data.qpos.tobytes()]
            rows.append(row)
        assert rows[0] == rows[1] == rows[2]
    finally:
        for each in (env, *twins):
            each.close()


# ---------------------------------------------------------------------------------------------------- each quantity
def _single(model_id: ModelId, key: str, *, training: bool = True, seed: int = 7, **extra: Any):
    """An env with *key* on alone (and *extra*), reset once; its fields and an off env's."""
    ranges = {key: _all_ranges(model_id)[key], **extra}
    env = _randomized(model_id, training=training, **ranges)
    env.reset(seed=seed)
    off = species_registry.env_class(model_id)(**_first_stage_kwargs(model_id))
    try:
        nominal = _model_arrays(off.model)
    finally:
        off.close()
    return env, nominal, _model_arrays(env.model)


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_friction_writes_the_sliding_coefficient_of_the_floor_and_the_hooks_geoms_only(model_id):
    env, nominal, live = _single(model_id, "randomize_friction")
    try:
        mu = env.episode_physics_factors["friction"]
        assert 0.6 <= mu <= 1.0 and env.episode_physics_factors.keys() == {"friction"}
        assert _changed(nominal, live) == {"geom_friction"}
        written = np.union1d(env._static_floor_geoms(), env._randomized_friction_geoms())
        assert np.array_equal(written, np.union1d(env._static_floor_geoms(), env._root_subtree_geoms()))
        assert np.all(live["geom_friction"][written, 0] == mu)
        others = np.setdiff1d(np.arange(env.model.ngeom), written)
        assert np.array_equal(live["geom_friction"][others], nominal["geom_friction"][others])
        assert np.array_equal(live["geom_friction"][:, 1:], nominal["geom_friction"][:, 1:])
    finally:
        env.close()


def test_a_species_narrows_the_friction_geoms_with_its_hook():
    """The hook decides the animal's geoms (the robot of commit 5 returns its soles); the floor always takes the
    draw, and every other geom keeps its authored friction."""
    probe = RANDOMIZED_ROBOT()
    try:
        feet = np.asarray(probe._root_subtree_geoms()[:2])
    finally:
        probe.close()
    narrowed = type(
        "NarrowFrictionRobot", (RANDOMIZED_ROBOT,), {"_randomized_friction_geoms": lambda self: feet.copy()}
    )
    env = narrowed(randomize_friction=(0.6, 1.0))
    off = species_registry.env_class(ROBOT)()
    try:
        env.reset(seed=1)
        written = np.union1d(env._static_floor_geoms(), feet)
        mu = env.episode_physics_factors["friction"]
        assert np.all(env.model.geom_friction[written, 0] == mu)
        others = np.setdiff1d(np.arange(env.model.ngeom), written)
        assert np.array_equal(env.model.geom_friction[others], off.model.geom_friction[others])
    finally:
        env.close()
        off.close()


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_per_body_mass_scales_each_animal_body_and_its_inertia_together(model_id):
    env, nominal, live = _single(model_id, "randomize_body_mass_scale")
    try:
        bodies = animal_bodies(env.model)
        scale = env.episode_physics_factors["body_mass_scale"]
        assert scale.shape == bodies.shape and np.all((0.9 <= scale) & (scale <= 1.1)) and len(set(scale)) > 1
        assert np.array_equal(live["body_mass"][bodies], nominal["body_mass"][bodies] * scale)
        massive = bodies[nominal["body_mass"][bodies] > 0]
        np.testing.assert_allclose(
            live["body_inertia"][massive] / nominal["body_inertia"][massive],
            np.repeat((live["body_mass"][massive] / nominal["body_mass"][massive])[:, None], 3, axis=1),
            rtol=1e-12,
        )
        # The world and every other body outside the animal keep their mass and inertia (no plant carries a mocap
        # target since main's target-free revision).
        outside = np.setdiff1d(np.arange(env.model.nbody), bodies)
        assert 0 in outside and env.model.nmocap == 0
        for name in ("body_mass", "body_inertia"):
            assert np.array_equal(live[name][outside], nominal[name][outside])
        # setConst ran on the new masses; the geometric statistics are nominal.
        root = int(bodies[0])
        assert live["body_subtreemass"][root] == pytest.approx(live["body_mass"][bodies].sum(), rel=1e-12)
        assert {"body_mass", "body_inertia", "body_subtreemass"} <= _changed(nominal, live) <= MANAGED
        for name in GEOMETRIC_STAT_FIELDS:
            assert np.array_equal(live[f"stat.{name}"], nominal[f"stat.{name}"]), name
        for name in RANDOMIZED_FIELDS:
            if name not in ("body_mass", "body_inertia"):
                assert np.array_equal(live[name], nominal[name]), name
    finally:
        env.close()


@pytest.mark.parametrize("model_id", [m for m in MODEL_IDS if m != ALLOSAURUS], ids=str)
def test_the_kp_scale_moves_gain_and_bias_together_and_keeps_the_damping(model_id):
    env, nominal, live = _single(model_id, "randomize_actuator_kp_scale")
    try:
        servos = position_servos(env.model)
        kp = env.episode_physics_factors["actuator_kp_scale"]
        assert servos.size and kp.shape == servos.shape and np.all((0.9 <= kp) & (kp <= 1.1))
        assert _changed(nominal, live) == {"actuator_gainprm", "actuator_biasprm"}
        assert np.array_equal(live["actuator_gainprm"][servos, 0], nominal["actuator_gainprm"][servos, 0] * kp)
        assert np.array_equal(live["actuator_gainprm"][:, 0], -live["actuator_biasprm"][:, 1])
        assert np.array_equal(live["actuator_gainprm"][:, 1:], nominal["actuator_gainprm"][:, 1:])
        assert np.array_equal(
            live["actuator_biasprm"][:, [0, *range(2, 10)]], nominal["actuator_biasprm"][:, [0, *range(2, 10)]]
        )
    finally:
        env.close()


@pytest.mark.parametrize("model_id", [m for m in MODEL_IDS if m != ALLOSAURUS], ids=str)
def test_the_force_limit_scale_moves_both_ends_by_one_factor(model_id):
    env, nominal, live = _single(model_id, "randomize_actuator_force_limit_scale")
    try:
        limited = force_limited_actuators(env.model)
        factor = env.episode_physics_factors["actuator_force_limit_scale"]
        assert limited.size and np.all((0.85 <= factor) & (factor <= 1.0))
        assert _changed(nominal, live) == {"actuator_forcerange"}
        assert np.array_equal(
            live["actuator_forcerange"][limited], nominal["actuator_forcerange"][limited] * factor[:, None]
        )
    finally:
        env.close()


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_the_whole_robot_mass_and_payload_apply_in_training_only(model_id):
    """For one seed: in training the animal's mass is (per-body masses) x factor + payload, the payload body's inertia
    scaled by (m + p) / m; in evaluation the per-body factors are the same and the masses are the per-body ones, the
    factors reported as 1 and 0 kg."""
    ranges = _all_ranges(model_id)
    keys = ("randomize_body_mass_scale", "randomize_total_mass_scale", "randomize_payload_kg")
    envs = {
        training: _randomized(model_id, training=training, **{k: ranges[k] for k in keys}) for training in (True, False)
    }
    alone = _randomized(model_id, training=True, randomize_body_mass_scale=ranges["randomize_body_mass_scale"])
    try:
        for env in (*envs.values(), alone):
            env.reset(seed=99)
        train, evaluate = envs[True], envs[False]
        bodies, body = animal_bodies(train.model), payload_body(train.model)
        factors = train.episode_physics_factors
        assert (
            0.9 <= factors["total_mass_scale"] <= 1.1
            and 0.0 < factors["payload_kg"] <= ranges["randomize_payload_kg"][1]
        )
        np.testing.assert_array_equal(factors["body_mass_scale"], evaluate.episode_physics_factors["body_mass_scale"])
        np.testing.assert_array_equal(factors["body_mass_scale"], alone.episode_physics_factors["body_mass_scale"])
        assert (
            evaluate.episode_physics_factors["total_mass_scale"],
            evaluate.episode_physics_factors["payload_kg"],
        ) == (
            1.0,
            0.0,
        )
        per_body_mass, per_body_inertia = alone.model.body_mass.copy(), alone.model.body_inertia.copy()
        assert np.array_equal(evaluate.model.body_mass, per_body_mass)
        assert np.array_equal(evaluate.model.body_inertia, per_body_inertia)
        whole, payload = factors["total_mass_scale"], factors["payload_kg"]
        assert _animal_mass(train.model) == pytest.approx(per_body_mass[bodies].sum() * whole + payload, rel=1e-12)
        others = bodies[bodies != body]
        np.testing.assert_allclose(train.model.body_mass[others], per_body_mass[others] * whole, rtol=1e-15)
        m = per_body_mass[body] * whole
        assert train.model.body_mass[body] == pytest.approx(m + payload, rel=1e-15)
        np.testing.assert_allclose(
            train.model.body_inertia[body], per_body_inertia[body] * whole * (m + payload) / m, rtol=1e-12
        )
        assert train.model.body_subtreemass[bodies[0]] == pytest.approx(_animal_mass(train.model), rel=1e-12)
        for name in GEOMETRIC_STAT_FIELDS:
            assert np.array_equal(
                np.asarray(getattr(train.model.stat, name)), np.asarray(getattr(alone.model.stat, name))
            )
    finally:
        for env in (*envs.values(), alone):
            env.close()


def test_the_training_only_keys_leave_an_evaluation_env_on_the_nominal_plant():
    identity = current_plant_identity(ROBOT)
    env = _randomized(ROBOT, training=False, randomize_total_mass_scale=(0.9, 1.1), randomize_payload_kg=(0.0, 0.1))
    try:
        nominal = _model_arrays(env.model)
        for seed in (0, None, 1):
            env.reset(seed=seed)
            assert env.episode_physics_factors == {"total_mass_scale": 1.0, "payload_kg": 0.0}
            assert _changed(nominal, _model_arrays(env.model)) == set()
        validate_compiled_plant(env.model, identity)
        mark_training_rollouts(env, True)
        env.reset(seed=0)
        assert env.episode_physics_factors["total_mass_scale"] != 1.0
        with pytest.raises(PlantCompatibilityError, match="physics_sha256"):
            validate_compiled_plant(env.model, identity)
    finally:
        env.close()


@pytest.mark.parametrize(
    "keys",
    [TRAINING_ONLY_ENV_KEYS[:1], TRAINING_ONLY_ENV_KEYS[1:], TRAINING_ONLY_ENV_KEYS],
    ids=["total", "payload", "total-and-payload"],
)
def test_every_training_episode_starts_from_the_nominal_plant(keys):
    """The training-only keys without the per-body key, whose write from the nominal copies would hide a missing
    restore: the whole-robot factor multiplies and the payload adds onto the live masses, so each episode's masses
    are nominal x factor + payload, never compounded from the episode before (design record §6, "restore every model
    this env changed")."""
    ranges = {key: _all_ranges(ROBOT)[key] for key in keys}
    env = _randomized(ROBOT, **ranges)
    try:
        nominal_mass, nominal_inertia = env.model.body_mass.copy(), env.model.body_inertia.copy()
        bodies, body = animal_bodies(env.model), payload_body(env.model)
        wholes = []
        for seed in (3, *(None,) * 7):
            env.reset(seed=seed)
            factors = env.episode_physics_factors
            whole, payload = factors.get("total_mass_scale", 1.0), factors.get("payload_kg", 0.0)
            assert factors.keys() == {FACTOR_OF[key] for key in keys}
            if "randomize_total_mass_scale" in keys:
                assert 0.9 <= whole <= 1.1
            if "randomize_payload_kg" in keys:
                assert 0.0 <= payload <= ranges["randomize_payload_kg"][1]
            mass, inertia = nominal_mass.copy(), nominal_inertia.copy()
            mass[bodies] *= whole
            inertia[bodies] *= whole
            m = float(mass[body])
            mass[body] = m + payload
            inertia[body] *= (m + payload) / m
            np.testing.assert_array_equal(env.model.body_mass, mass)
            np.testing.assert_array_equal(env.model.body_inertia, inertia)
            assert _animal_mass(env.model) == pytest.approx(nominal_mass[bodies].sum() * whole + payload, rel=1e-12)
            wholes.append(whole)
        assert len(set(wholes)) == (8 if "randomize_total_mass_scale" in keys else 1)
    finally:
        env.close()


def test_a_role_switch_between_episodes_restores_the_plant_and_keeps_the_seeds_draws():
    """One env, its role switched between episodes (as train_behaviors unmarks its env before scoring).  With the
    training-only keys alone: training episodes are nominal x factor + payload; unmarked, the next episode's model is
    exactly an off env's, every record nominal and the factors 1 and 0 kg; marked again, the seed's first training
    plant comes back bit for bit.  With every key on: a seed's per-body, kp and force draws are the same in both
    roles, and the evaluation masses are the per-body ones."""
    off = species_registry.env_class(ROBOT)(**_first_stage_kwargs(ROBOT))
    env = _randomized(ROBOT, **{key: _all_ranges(ROBOT)[key] for key in TRAINING_ONLY_ENV_KEYS})
    every = _randomized(ROBOT, **_all_ranges(ROBOT))
    try:
        nominal = _model_arrays(off.model)
        bodies, body = animal_bodies(env.model), payload_body(env.model)
        first = None
        for seed in (1, None, None):
            env.reset(seed=seed)
            whole, payload = env.episode_physics_factors["total_mass_scale"], env.episode_physics_factors["payload_kg"]
            expected = nominal["body_mass"].copy()
            expected[bodies] *= whole
            expected[body] += payload
            np.testing.assert_array_equal(env.model.body_mass, expected)
            first = first or _model_arrays(env.model)
        mark_training_rollouts(env, False)
        env.reset(seed=1)
        assert _changed(nominal, _model_arrays(env.model)) == set()
        assert not any(record.randomized for record in env._episode_physics._records)
        assert env.episode_physics_factors == {"total_mass_scale": 1.0, "payload_kg": 0.0}
        mark_training_rollouts(env, True)
        env.reset(seed=1)
        assert _changed(first, _model_arrays(env.model)) == set()

        draws = {}
        for training in (True, False, True):
            mark_training_rollouts(every, training)
            every.reset(seed=4)
            draws.setdefault(training, []).append((_factors(every), every.model.body_mass.copy()))
        (train, train_mass), (train_again, train_again_mass) = draws[True]
        (evaluate, evaluate_mass) = draws[False][0]
        assert train == train_again and np.array_equal(train_mass, train_again_mass)
        train_factors, evaluate_factors = dict(pickle.loads(train)), dict(pickle.loads(evaluate))
        for name in ("friction", "body_mass_scale", "actuator_kp_scale", "actuator_force_limit_scale"):
            assert train_factors[name] == evaluate_factors[name], name
        assert (evaluate_factors["total_mass_scale"], evaluate_factors["payload_kg"]) == (1.0, 0.0)
        per_body = nominal["body_mass"].copy()
        per_body[bodies] = per_body[bodies] * np.asarray(evaluate_factors["body_mass_scale"])
        np.testing.assert_array_equal(evaluate_mass, per_body)
    finally:
        for each in (off, env, every):
            each.close()


def test_set_const_runs_on_scratch_data_never_the_live_data(monkeypatch):
    """Every mass key on and reset noise above 0: mj_setConst only ever sees the record's scratch MjData, never the
    live one.  The identity of the data it saw is the check.  The scratch data is defensive: begin_episode runs before
    the keyframe load, which overwrites the live data, so the reset pose equal to the off env's holds either way and is
    a consequence check only; the closing counterfactual, outside reset, shows what setConst does to the data it is
    given (it moves qpos to qpos_spring)."""
    seen = []
    set_const = mujoco.mj_setConst

    def record(model, data):
        seen.append(data)
        set_const(model, data)

    monkeypatch.setattr(mujoco, "mj_setConst", record)
    kwargs = {**_first_stage_kwargs(ROBOT), "reset_noise_scale": 0.05}
    keys = ("randomize_body_mass_scale", "randomize_total_mass_scale", "randomize_payload_kg")
    on = _randomized(ROBOT, kwargs, **{k: _all_ranges(ROBOT)[k] for k in keys})
    off = species_registry.env_class(ROBOT)(**kwargs)
    try:
        for seed in (1, None, 2):
            on.reset(seed=seed)
            off.reset(seed=seed)
            assert np.array_equal(on.data.qpos, off.data.qpos) and np.array_equal(on.data.qvel, off.data.qvel)
        assert len(seen) == 3 and all(data is on._episode_physics._records[0].scratch for data in seen)
        assert all(data is not on.data for data in seen)
        # Outside reset (mid-episode, where no keyframe load follows), setConst on the live data moves its pose.
        before = on.data.qpos.copy()
        set_const(on.model, on.data)
        assert not np.array_equal(on.data.qpos, before)
    finally:
        on.close()
        off.close()


# ---------------------------------------------------------------------------------------------------- the draws
def _factors(env) -> bytes:
    return pickle.dumps(sorted((key, np.asarray(value).tolist()) for key, value in env.episode_physics_factors.items()))


def test_draws_are_repeatable_per_seed_new_on_unseeded_resets_and_equal_across_envs():
    envs = [_randomized(ROBOT, **_all_ranges(ROBOT)) for _ in range(2)]
    try:
        rows = []
        for env in envs:
            row = []
            for seed in (4, None, None, 4, None, 9):
                env.reset(seed=seed)
                row.append(_factors(env))
            rows.append(row)
        assert rows[0] == rows[1]
        row = rows[0]
        assert row[0] == row[3] and row[1] == row[4]
        assert len({row[0], row[1], row[2], row[5]}) == 4
    finally:
        for env in envs:
            env.close()


#: Each key's entry in ``episode_physics_factors``.
FACTOR_OF = {
    "randomize_friction": "friction",
    "randomize_body_mass_scale": "body_mass_scale",
    "randomize_actuator_kp_scale": "actuator_kp_scale",
    "randomize_actuator_force_limit_scale": "actuator_force_limit_scale",
    "randomize_total_mass_scale": "total_mass_scale",
    "randomize_payload_kg": "payload_kg",
}


def test_one_key_never_moves_another_keys_draw():
    """The block of uniforms is the same size whatever is enabled: each of the six keys draws the same value alone as
    with every other key on, in a training env (where the whole-robot factor and payload apply)."""
    assert FACTOR_OF.keys() == set(EPISODE_PHYSICS_ENV_KEYS)
    ranges = _all_ranges(ROBOT)
    assert ranges.keys() == set(EPISODE_PHYSICS_ENV_KEYS)
    alone = {key: _randomized(ROBOT, **{key: ranges[key]}) for key in EPISODE_PHYSICS_ENV_KEYS}
    every = _randomized(ROBOT, **ranges)
    try:
        for seed in (0, None, 13):
            for env in (*alone.values(), every):
                env.reset(seed=seed)
            for key, env in alone.items():
                assert env.episode_physics_factors.keys() == {FACTOR_OF[key]}
                np.testing.assert_array_equal(
                    env.episode_physics_factors[FACTOR_OF[key]], every.episode_physics_factors[FACTOR_OF[key]]
                )
    finally:
        for env in (*alone.values(), every):
            env.close()


@pytest.mark.parametrize("training", [True, False], ids=["training", "evaluation"])
@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_the_draw_block_has_the_records_layout_on_every_model(model_id, training):
    """Design record §6: u[0] friction, u[1 + body] each body's factor, u[1 + nbody + actuator] each servo's kp,
    u[1 + nbody + nu + actuator] each force limit, u[-2] the whole-robot factor, u[-1] the payload, each
    low + (high - low) x u.  The block is rebuilt from its own stream and every reported factor and every randomised
    model field is recomputed from it and the off env's nominal values, bit for bit (the training-only two at 1 and
    0 kg in evaluation)."""
    ranges = _all_ranges(model_id)
    env = _randomized(model_id, training=training, **ranges)
    off = species_registry.env_class(model_id)(**_first_stage_kwargs(model_id))
    try:
        nominal = _model_arrays(off.model)
        model = env.model
        nbody, nu = int(model.nbody), int(model.nu)
        bodies, body = animal_bodies(model), payload_body(model)
        servos, limited = position_servos(model), force_limited_actuators(model)
        geoms = np.union1d(env._static_floor_geoms(), env._randomized_friction_geoms())
        for seed in (5, 123456789):
            env.reset(seed=seed)
            count = env._episode_physics.draw_count
            assert count == 3 + nbody + 2 * nu
            u = feature_stream(env, seed, EPISODE_PHYSICS_STREAM_KEY).random(count)

            def span(key, x):
                low, high = ranges[key]
                return low + (high - low) * x

            expected = {
                "friction": span("randomize_friction", u[0]),
                "body_mass_scale": span("randomize_body_mass_scale", u[1 + bodies]),
                "total_mass_scale": span("randomize_total_mass_scale", u[-2]) if training else 1.0,
                "payload_kg": span("randomize_payload_kg", u[-1]) if training else 0.0,
            }
            if "randomize_actuator_kp_scale" in ranges:
                expected["actuator_kp_scale"] = span("randomize_actuator_kp_scale", u[1 + nbody + servos])
            if "randomize_actuator_force_limit_scale" in ranges:
                expected["actuator_force_limit_scale"] = span(
                    "randomize_actuator_force_limit_scale", u[1 + nbody + nu + limited]
                )
            factors = env.episode_physics_factors
            assert factors.keys() == expected.keys()
            for name, value in expected.items():
                np.testing.assert_array_equal(factors[name], value, err_msg=name)
            # The model's randomised fields, from the nominal values in the record's write order.
            friction = nominal["geom_friction"].copy()
            friction[geoms, 0] = expected["friction"]
            mass, inertia = nominal["body_mass"].copy(), nominal["body_inertia"].copy()
            mass[bodies] = mass[bodies] * expected["body_mass_scale"]
            inertia[bodies] = inertia[bodies] * expected["body_mass_scale"][:, None]
            if training:
                mass[bodies] *= expected["total_mass_scale"]
                inertia[bodies] *= expected["total_mass_scale"]
                m = float(mass[body])
                mass[body] = m + expected["payload_kg"]
                inertia[body] *= (m + expected["payload_kg"]) / m
            gain, bias = nominal["actuator_gainprm"].copy(), nominal["actuator_biasprm"].copy()
            forcerange = nominal["actuator_forcerange"].copy()
            if "actuator_kp_scale" in expected:
                gain[servos, 0] = gain[servos, 0] * expected["actuator_kp_scale"]
                bias[servos, 1] = bias[servos, 1] * expected["actuator_kp_scale"]
            if "actuator_force_limit_scale" in expected:
                forcerange[limited] = forcerange[limited] * expected["actuator_force_limit_scale"][:, None]
            for name, value in (
                ("geom_friction", friction),
                ("body_mass", mass),
                ("body_inertia", inertia),
                ("actuator_gainprm", gain),
                ("actuator_biasprm", bias),
                ("actuator_forcerange", forcerange),
            ):
                np.testing.assert_array_equal(getattr(model, name), value, err_msg=name)
    finally:
        env.close()
        off.close()


def test_each_episode_draws_3_plus_nbody_plus_2_nu_uniforms_from_its_own_stream():
    for key in ("randomize_friction", None):
        ranges = {key: ROBOT_RANGES[key]} if key else _all_ranges(ROBOT)
        env = _randomized(ROBOT, **ranges)
        try:
            randomizer = env._episode_physics
            count = randomizer.draw_count
            assert count == 3 + env.model.nbody + 2 * env.model.nu == 3 + 19 + 24
            env.reset(seed=31)
            for _ in range(4):
                env.reset()
            reference = feature_stream(env, 31, EPISODE_PHYSICS_STREAM_KEY)
            reference.random(5 * count)
            assert randomizer.rng.bit_generator.state == reference.bit_generator.state
            # The friction of the episode is u[0] of the block.
            env.reset(seed=8)
            u = feature_stream(env, 8, EPISODE_PHYSICS_STREAM_KEY).random(count)
            assert env.episode_physics_factors["friction"] == 0.6 + (1.0 - 0.6) * u[0]
        finally:
            env.close()


def test_the_randomisation_draws_nothing_from_np_random():
    on, off = _randomized(ROBOT, **_all_ranges(ROBOT)), species_registry.env_class(ROBOT)(**_first_stage_kwargs(ROBOT))
    try:
        for seed in (4, None, None, 9, None):
            on.reset(seed=seed)
            off.reset(seed=seed)
            assert on.np_random.bit_generator.state == off.np_random.bit_generator.state
            on.step(np.zeros(12))
            off.step(np.zeros(12))
    finally:
        on.close()
        off.close()


def test_both_validators_refuse_after_a_randomised_reset_and_pass_inside_nominal_physics():
    """Every key alone moves the compiled plant's physics digest; the live env, every key on, fails both validators
    after a reset and passes them before it and inside nominal_physics(), and fails again after the block."""
    identity = current_plant_identity(ROBOT)
    for key in EPISODE_PHYSICS_ENV_KEYS:
        env = _randomized(ROBOT, **{key: _all_ranges(ROBOT)[key]})
        try:
            env.reset(seed=2)
            with pytest.raises(PlantCompatibilityError, match="physics_sha256"):
                validate_compiled_plant(env.model, identity)
        finally:
            env.close()
    env = _randomized(ROBOT, **_all_ranges(ROBOT))
    try:
        validate_environment_plant(env, identity, artifact="before the first reset")
        env.reset(seed=2)
        with pytest.raises(PlantCompatibilityError, match="physics_sha256"):
            validate_environment_plant(env, identity, artifact="randomised")
        with env.nominal_physics():
            validate_environment_plant(env, identity, artifact="inside nominal_physics")
            validate_compiled_plant(env.model, identity)
        with pytest.raises(PlantCompatibilityError, match="physics_sha256"):
            validate_compiled_plant(env.model, identity)
    finally:
        env.close()


def test_nominal_physics_writes_the_episode_back_when_its_block_raises_plain_or_nested():
    """Its main use is plant validation, which raises on a failure: the episode's values come back bit for bit after a
    raising block and after a raising inner block of a nested pair, the validators refuse again after the block, and
    the next restore is to nominal."""
    identity = current_plant_identity(ROBOT)
    env = _randomized(ROBOT, **_all_ranges(ROBOT))
    off = species_registry.env_class(ROBOT)(**_first_stage_kwargs(ROBOT))
    try:
        nominal = _model_arrays(off.model)
        env.reset(seed=6)
        live = _model_arrays(env.model)
        with pytest.raises(KeyError, match="plain"):
            with env.nominal_physics():
                assert _changed(nominal, _model_arrays(env.model)) == set()
                raise KeyError("plain")
        assert _changed(live, _model_arrays(env.model)) == set()
        assert env._episode_physics.holds_episode_values(env.model)
        with env.nominal_physics():
            with pytest.raises(KeyError, match="nested"):
                with env.nominal_physics():
                    assert _changed(nominal, _model_arrays(env.model)) == set()
                    raise KeyError("nested")
            assert _changed(nominal, _model_arrays(env.model)) == set()  # still nominal in the outer block
            validate_environment_plant(env, identity, artifact="the outer block")
        assert _changed(live, _model_arrays(env.model)) == set()
        with pytest.raises(PlantCompatibilityError, match="physics_sha256"):
            validate_environment_plant(env, identity, artifact="after the blocks")
        assert env._episode_physics._nominal_depth == 0
        env._restore_episode_physics()
        assert _changed(nominal, _model_arrays(env.model)) == set()
    finally:
        env.close()
        off.close()


def _snapshot(env) -> dict[str, tuple[int, bytes]]:
    """Every instance attribute's identity and pickled value (``env.model`` and ``env.data`` with every array, the
    generators with their states, the randomiser with its records and factors), so any write or rebinding shows."""
    return {name: (id(value), pickle.dumps(value)) for name, value in vars(env).items()}


def _refuse_resets_inside_a_block(env, seeds) -> None:
    """Inside a plain and a nested ``nominal_physics()`` block, each reset of *seeds* is refused and changes nothing:
    every instance attribute is as the block left it, and the block's exit writes the episode back."""
    live, factors = _model_arrays(env.model), _factors(env)
    for nested in (False, True):
        with env.nominal_physics():
            with contextlib.ExitStack() as stack:
                if nested:
                    stack.enter_context(env.nominal_physics())
                before = _snapshot(env)
                for seed in seeds:
                    with pytest.raises(RuntimeError, match="a reset inside a nominal_physics"):
                        env.reset(seed=seed)
                after = _snapshot(env)
                assert after.keys() == before.keys()
                assert [name for name in before if after[name] != before[name]] == []
        assert _changed(live, _model_arrays(env.model)) == set() and _factors(env) == factors
        assert env._episode_physics.holds_episode_values(env.model)


def test_a_reset_inside_a_nominal_physics_block_is_refused_before_anything_moves():
    """Its exit would write the previous episode's values back over the new one's, the factors reporting a plant the
    env does not run: the reset is refused before any of it runs (before gymnasium reseeds np_random), so a refused
    reset changes no attribute of the env, the block's exit puts the episode back, and the resets after the block,
    unseeded and seeded, equal those of a twin that never tried."""
    env = _randomized(ROBOT, **_all_ranges(ROBOT))
    twin = _randomized(ROBOT, **_all_ranges(ROBOT))
    try:
        for each in (env, twin):
            each.reset(seed=7)
        _refuse_resets_inside_a_block(env, (99, None))
        before = RANDOMIZED_ROBOT(**_all_ranges(ROBOT))
        try:  # before the first episode, too: the block has nothing to restore, and still refuses an episode
            with before.nominal_physics(), pytest.raises(RuntimeError, match="a reset inside a nominal_physics"):
                before.reset(seed=1)
            assert before.episode_physics_factors is None and before._episode_physics._records == []
        finally:
            before.close()
        for seed in (None, 99, None):
            observations = env.reset(seed=seed)[0], twin.reset(seed=seed)[0]
            assert np.array_equal(*observations) and _factors(env) == _factors(twin)
            assert _changed(_model_arrays(twin.model), _model_arrays(env.model)) == set()
            assert np.array_equal(env.data.qpos, twin.data.qpos) and np.array_equal(env.data.qvel, twin.data.qvel)
    finally:
        env.close()
        twin.close()


def test_the_episode_is_refused_inside_a_block_even_when_called_directly():
    """The randomiser refuses too (a direct _begin_episode call never reaches the refusal of reset), before anything is
    restored or drawn."""
    env = _randomized(ROBOT, **_all_ranges(ROBOT))
    try:
        env.reset(seed=7)
        with env.nominal_physics():
            before = _snapshot(env)
            with pytest.raises(RuntimeError, match="a reset inside a nominal_physics"):
                env._begin_episode(3)
            assert _snapshot(env) == before
    finally:
        env.close()


@pytest.fixture
def opted_in_robot(monkeypatch):
    """The registry resolving the Compsognathus robot to its opted-in test subclass (carve-outs at off value None).

    The behavior env classes are cached per model, so the cache is emptied on both sides of the test.
    """
    from environments.shared.behavior_env import _behavior_env_class

    identity = current_plant_identity(ROBOT)
    real = species_registry.env_class
    monkeypatch.setattr(
        species_registry, "env_class", lambda model_id: RANDOMIZED_ROBOT if model_id == ROBOT else real(model_id)
    )
    _behavior_env_class.cache_clear()
    yield identity
    _behavior_env_class.cache_clear()


#: Seed 0 then 15 unseeded resets of the mixed_terrain recipe select P H H H H H H P H H P H H H H P (P the plane, H the
#: heightfield): the plane returns at 7, 10 and 15 while the heightfield's record is the newest, so a restore of only
#: the live or the newest model would leave _cache_ids reading randomised masses there.
POOL_SEEDS = (0, *(None,) * 15)


def test_a_behavior_pool_restores_every_model_before_cache_ids(opted_in_robot, monkeypatch):
    """The Compsognathus robot's mixed_terrain recipe (one plane episode in four, three heightfield ones): over 16
    training resets, every pool model is nominal whenever _cache_ids runs (it reads masses: the contact threshold
    follows them), both pool models take the draws, the heightfield included, and the selection returns to each model
    while the other's record is the newer; without the restore, the threshold follows the previous episode's mass."""
    from environments.shared.behavior_env import get_behavior_env_class
    from environments.shared.train_behaviors import create_behavior_env, read_recipe

    assert RANDOMIZED_ROBOT in get_behavior_env_class(ROBOT).__mro__
    recipe = REPOSITORY_ROOT / "configs" / ROBOT.species / ROBOT.embodiment / "behaviors" / "mixed_terrain.toml"
    _, commands, terrain, env_kwargs = read_recipe(recipe, ROBOT)

    def build():
        env = create_behavior_env(
            ROBOT, commands=commands, terrain=terrain, run_seed=0, **env_kwargs, **_all_ranges(ROBOT)
        )
        mark_training_rollouts(env, True)
        return env

    env = build()
    plane = species_registry.env_class(ROBOT)()
    try:
        nominal_mass = plane.model.body_mass.copy()
        nominal_threshold = plane._contact_threshold
        cache_ids = env._cache_ids
        seen = []

        def record():
            records = env._episode_physics._records
            seen.append((env.model.body_mass.copy(), [each.randomized for each in records]))
            cache_ids()
            seen[-1] += (env._contact_threshold,)

        env._cache_ids = record
        floors, frictions = [], []
        for seed in POOL_SEEDS:
            env.reset(seed=seed)
            floors.append("P" if env.model.geom_type[0] == mujoco.mjtGeom.mjGEOM_PLANE else "H")
            frictions.append(float(env.model.geom_friction[0, 0]))
            assert env.model.geom_friction[0, 0] == env.episode_physics_factors["friction"]
            assert not np.array_equal(env.model.body_mass, nominal_mass)
        assert "".join(floors) == "PHHHHHHPHHPHHHHP"
        assert len(seen) == 16
        for masses, randomized, threshold in seen:
            assert np.array_equal(masses, nominal_mass) and threshold == nominal_threshold
            assert not any(randomized)
        assert len(env._episode_physics._records) == 2 and len(set(frictions)) == 16
    finally:
        env.close()
        plane.close()
    # Live: with the selection's restore skipped, _cache_ids reads the last episode's masses.
    broken = build()
    try:
        monkeypatch.setattr(broken, "_restore_episode_physics", lambda: None)
        thresholds = []
        cache_ids = broken._cache_ids

        def record_threshold():
            cache_ids()
            thresholds.append(broken._contact_threshold)

        broken._cache_ids = record_threshold
        for seed in POOL_SEEDS:
            broken.reset(seed=seed)
        assert any(threshold != nominal_threshold for threshold in thresholds)
    finally:
        broken.close()


def test_a_behavior_reset_inside_a_nominal_physics_block_selects_no_pool_model(opted_in_robot):
    """The mixed_terrain behaviour env (seed 0 selects the plane, the next unseeded reset the heightfield, POOL_SEEDS):
    a reset inside the block is refused before the mixin moves its episode counters or selects a pool model, so the
    env stays on the plane, which the block's exit puts back to the episode's values, every attribute is unchanged,
    and the resets after the block equal those of a twin that never tried, the swap to the heightfield included."""
    from environments.shared.train_behaviors import create_behavior_env, read_recipe

    recipe = REPOSITORY_ROOT / "configs" / ROBOT.species / ROBOT.embodiment / "behaviors" / "mixed_terrain.toml"
    _, commands, terrain, env_kwargs = read_recipe(recipe, ROBOT)

    def build():
        env = create_behavior_env(
            ROBOT, commands=commands, terrain=terrain, run_seed=0, **env_kwargs, **_all_ranges(ROBOT)
        )
        mark_training_rollouts(env, True)
        return env

    def floor(env) -> str:
        return "P" if env.model.geom_type[0] == mujoco.mjtGeom.mjGEOM_PLANE else "H"

    env, twin = build(), build()
    try:
        for each in (env, twin):
            each.reset(seed=0)
        plane, data = env.model, env.data
        assert floor(env) == "P" and env.model is env._plane_model
        _refuse_resets_inside_a_block(env, (None, 3))
        assert env.model is plane and env.data is data
        floors = []
        for seed in (None, None, 3):
            observations = env.reset(seed=seed)[0], twin.reset(seed=seed)[0]
            assert floor(env) == floor(twin) and np.array_equal(*observations) and _factors(env) == _factors(twin)
            assert _changed(_model_arrays(twin.model), _model_arrays(env.model)) == set()
            floors.append(floor(env))
        assert floors[0] == "H"  # the selection the refused resets would have made, made after the block
    finally:
        env.close()
        twin.close()


# ---------------------------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("randomize_actuator_kp_scale", "has no position servo"),
        ("randomize_actuator_force_limit_scale", "has no force-limited actuator"),
    ],
)
def test_the_actuator_keys_are_refused_on_a_model_without_the_actuators(key, message):
    with pytest.raises(ValueError, match=f"{key}: the model {message}"):
        RANDOMIZED_ENV_CLASSES[ALLOSAURUS](**{key: (0.9, 1.0)})


def _swap_model(env, model: mujoco.MjModel) -> None:
    env.model, env.data = model, mujoco.MjData(model)
    env._root_subtree_geom_ids = env._static_floor_geom_ids = None


def test_friction_is_refused_on_a_model_with_explicit_contact_pairs():
    spec = mujoco.MjSpec.from_file(str(REPOSITORY_ROOT / current_plant_identity(ROBOT).model_path))
    env = RANDOMIZED_ROBOT()
    try:
        foot = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(env._root_subtree_geoms()[0]))
        spec.add_pair(geomname1="floor", geomname2=foot)
        paired = spec.compile()
        assert paired.npair == 1
        _swap_model(env, paired)
        with pytest.raises(ValueError, match=r"randomize_friction: the model has 1 explicit contact pair"):
            env._configure_episode_physics(randomize_friction=(0.6, 1.0))
        env._configure_episode_physics(randomize_body_mass_scale=(0.9, 1.1))  # pairs only matter to friction
    finally:
        env.close()


def test_a_payload_on_a_body_without_mass_is_refused():
    env = RANDOMIZED_ROBOT()
    try:
        model = copy.deepcopy(env.model)
        root = int(animal_bodies(model)[0])
        assert model.body_mass[root] == 0  # the Compsognathus robot's root is a massless frame
        model.body_mass[model.body_weldid == root] = 0.0
        _swap_model(env, model)
        with pytest.raises(ValueError, match="randomize_payload_kg: the payload body .* has no mass"):
            env._configure_episode_physics(randomize_payload_kg=(0.0, 0.1))
        env._configure_episode_physics(randomize_total_mass_scale=(0.9, 1.1))
    finally:
        env.close()


def test_friction_is_refused_on_a_model_with_no_static_floor_geom():
    spec = mujoco.MjSpec.from_file(str(REPOSITORY_ROOT / current_plant_identity(ROBOT).model_path))
    spec.delete(spec.geom("floor"))
    floorless = spec.compile()
    env = RANDOMIZED_ROBOT()
    try:
        _swap_model(env, floorless)
        assert not env._static_floor_geoms().size
        with pytest.raises(ValueError, match=r"randomize_friction needs a static floor geom"):
            env._configure_episode_physics(randomize_friction=(0.6, 1.0))
        env._configure_episode_physics(randomize_body_mass_scale=(0.9, 1.1))  # the floor only matters to friction
    finally:
        env.close()


def _hook(geoms):
    """A test-only Compsognathus robot whose friction hook returns *geoms(env)*."""
    return type("HookedRobot", (RANDOMIZED_ROBOT,), {"_randomized_friction_geoms": lambda self: geoms(self)})


def _floor_geom(env) -> int:
    return int(env._static_floor_geoms()[0])


def _unused_animal_geom(env) -> int:
    """A non-colliding geom of the animal (cosmetic), whose friction no contact reads."""
    model = env.model
    animal = np.isin(model.geom_bodyid, animal_bodies(model))
    return int(np.flatnonzero(animal & (model.geom_contype == 0) & (model.geom_conaffinity == 0))[0])


@pytest.mark.parametrize(
    ("geoms", "message"),
    [
        pytest.param(lambda env: np.empty(0, dtype=np.int64), "returned no geom", id="no geom"),
        pytest.param(lambda env: [], "returned no geom", id="an empty list"),
        pytest.param(
            lambda env: np.isin(np.arange(env.model.ngeom), env._root_subtree_geoms()[:2]),
            r"must return a 1-D array of integer geom ids, got dtype bool",
            id="a boolean mask",
        ),
        pytest.param(lambda env: np.array([[1, 2]]), "a 1-D array of integer geom ids", id="two-dimensional"),
        pytest.param(lambda env: np.array([1.0, 2.0]), "integer geom ids, got dtype float64", id="floats"),
        pytest.param(lambda env: [env.model.ngeom], r"outside \[0, ngeom = \d+\)", id="past the last geom"),
        pytest.param(lambda env: [-1], r"outside \[0, ngeom", id="negative"),
        pytest.param(lambda env: [_floor_geom(env)], "not colliding animal geoms: \\['floor'\\]", id="the floor"),
        pytest.param(lambda env: [_unused_animal_geom(env)], "not colliding animal geoms", id="non-colliding"),
    ],
)
def test_the_friction_hook_must_return_colliding_animal_geom_ids(geoms, message):
    """Refused at construction, each naming the hook: a boolean mask would otherwise index as geoms 0 and 1, an id
    past the model raise only at the write, a negative one wrap; a geom outside the animal is not the hook's to name
    (the floor always takes the draw)."""
    with pytest.raises(ValueError, match=r"randomize_friction: _randomized_friction_geoms\(\) .*" + message):
        _hook(geoms)(randomize_friction=(0.6, 1.0))
    _hook(geoms)(randomize_body_mass_scale=(0.9, 1.1)).close()  # the hook only matters to friction


def test_the_colliding_animal_geoms_are_the_root_subtree_geoms_on_every_model():
    for model_id in MODEL_IDS:
        env = species_registry.env_class(model_id)()
        try:
            np.testing.assert_array_equal(colliding_animal_geoms(env.model), env._root_subtree_geoms())
        finally:
            env.close()


def test_the_hook_is_read_again_at_the_first_episode_on_a_model():
    """A hook that turns invalid after construction is refused when the model's record is taken, before any write."""
    env = _hook(lambda self: getattr(self, "hook_geoms", self._root_subtree_geoms()))(randomize_friction=(0.6, 1.0))
    try:
        env.hook_geoms = np.zeros(env.model.ngeom, dtype=bool)
        before = _model_arrays(env.model)
        with pytest.raises(ValueError, match="got dtype bool"):
            env.reset(seed=1)
        assert _changed(before, _model_arrays(env.model)) == set() and env.episode_physics_factors is None
    finally:
        env.close()


def test_the_free_joint_root_is_required():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="hinge"/><geom size="0.1"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="needs a model with a free-joint root body"):
        animal_bodies(model)


def test_a_pool_model_of_another_nbody_or_nu_is_refused_at_its_first_episode():
    """The draw block's layout is the constructor model's (3 + nbody + 2 x nu), so a behaviour pool model of another
    shape is refused when an episode first starts on it, before anything is written."""
    spec = mujoco.MjSpec.from_file(str(REPOSITORY_ROOT / current_plant_identity(ROBOT).model_path))
    spec.worldbody.add_body(name="extra_prop", pos=[5.0, 5.0, 0.1]).add_geom(size=[0.05, 0.0, 0.0])
    larger = spec.compile()
    env = RANDOMIZED_ROBOT(randomize_body_mass_scale=(0.9, 1.1))
    try:
        assert larger.nbody == env.model.nbody + 1 and larger.nu == env.model.nu
        _swap_model(env, larger)
        before = _model_arrays(env.model)
        with pytest.raises(ValueError, match=r"a pool model changed nbody or nu \(20, 12 against the constructor"):
            env.reset(seed=1)
        assert _changed(before, _model_arrays(env.model)) == set() and not env._episode_physics._records
    finally:
        env.close()


def test_a_species_edit_after_its_constructor_decides_what_the_episodes_write():
    """Design record R16: what each model's episodes write is derived when its record is taken (its first episode),
    so an edit made after super().__init__() decides it: a payload goes to the edited model's payload body; an edit
    that leaves the payload body massless is refused by name at the first reset; and a collision-mask edit reaches
    the friction draw and the settle, since the construction check leaves the env's geom caches as it found them."""
    robot = RANDOMIZED_ROBOT()
    try:
        root, before_body = int(animal_bodies(robot.model)[0]), payload_body(robot.model)
        welded = np.flatnonzero((robot.model.body_weldid == root) & (robot.model.body_mass > 0))
        other = int(next(body for body in welded if body != before_body))
        colliding = int(robot._root_subtree_geoms()[0])
    finally:
        robot.close()

    class Heavier(RANDOMIZED_ROBOT):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model.body_mass[other] = 2.0 * self.model.body_mass[before_body]

    env = Heavier(randomize_payload_kg=(0.05, 0.1))
    try:
        mark_training_rollouts(env, True)
        nominal = env.model.body_mass.copy()
        assert payload_body(env.model) == other != before_body
        env.reset(seed=1)
        added = env.model.body_mass - nominal
        assert np.flatnonzero(added).tolist() == [other]
        assert env.model.body_mass[other] == nominal[other] + env.episode_physics_factors["payload_kg"]
    finally:
        env.close()

    class Massless(RANDOMIZED_ROBOT):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model.body_mass[self.model.body_weldid == root] = 0.0

    env = Massless(randomize_payload_kg=(0.0, 0.1))
    try:
        mark_training_rollouts(env, True)
        before = _model_arrays(env.model)
        with pytest.raises(ValueError, match="randomize_payload_kg: the payload body .* has no mass"):
            env.reset(seed=1)
        assert _changed(before, _model_arrays(env.model)) == set()
    finally:
        env.close()

    class Masked(RANDOMIZED_ROBOT):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model.geom_contype[colliding] = self.model.geom_conaffinity[colliding] = 0

    env, off = Masked(randomize_friction=(0.6, 0.6)), Masked()
    try:
        assert env._root_subtree_geom_ids is None and env._static_floor_geom_ids is None
        nominal = env.model.geom_friction[colliding].copy()
        env.reset(seed=1)
        off.reset(seed=1)
        assert (
            colliding not in env._root_subtree_geoms()
            and colliding not in env._episode_physics._records[0].targets.friction_geoms
        )
        np.testing.assert_array_equal(env.model.geom_friction[colliding], nominal)
        np.testing.assert_array_equal(env._root_subtree_geoms(), off._root_subtree_geoms())
        assert np.array_equal(env.data.qpos, off.data.qpos)
    finally:
        env.close()
        off.close()


def test_a_raise_while_an_episode_is_written_leaves_the_nominal_plant(monkeypatch):
    """A failure after some quantities are written (here mj_setConst, the last step) restores the model before it
    propagates, so no model holds randomised values while flagged nominal, which every later restore and
    nominal_physics() block would skip; the next reset randomises as usual."""
    env = _randomized(ROBOT, **_all_ranges(ROBOT))
    off = species_registry.env_class(ROBOT)(**_first_stage_kwargs(ROBOT))
    try:
        nominal = _model_arrays(off.model)
        env.reset(seed=1)
        set_const = mujoco.mj_setConst
        written = []

        def fail(model, data):
            written.append(_changed(nominal, _model_arrays(model)))
            raise FloatingPointError("setConst failed")

        monkeypatch.setattr(mujoco, "mj_setConst", fail)
        with pytest.raises(FloatingPointError, match="setConst failed"):
            env.reset(seed=2)
        (changed,) = written  # the failure came after every quantity was written
        assert set(RANDOMIZED_FIELDS) <= changed
        assert _changed(nominal, _model_arrays(env.model)) == set()
        assert not any(record.randomized for record in env._episode_physics._records)
        assert env.episode_physics_factors is None
        monkeypatch.setattr(mujoco, "mj_setConst", set_const)
        env.reset(seed=2)
        assert _changed(nominal, _model_arrays(env.model)) and env._episode_physics.holds_episode_values(env.model)
    finally:
        env.close()
        off.close()


def test_reconfiguring_inside_a_nominal_physics_block_is_refused():
    """The block's exit would write the episode's values back over the plant the new randomisation records as nominal
    at its first episode, so every later restore would return to them: refused before anything moves, off or on."""
    env = _randomized(ROBOT, **_all_ranges(ROBOT))
    off = species_registry.env_class(ROBOT)(**_first_stage_kwargs(ROBOT))
    try:
        nominal = _model_arrays(off.model)
        env.reset(seed=1)
        live, randomizer = _model_arrays(env.model), env._episode_physics
        with env.nominal_physics():
            for ranges in ({}, _all_ranges(ROBOT)):
                with pytest.raises(RuntimeError, match="reconfiguring the randomisation inside a nominal_physics"):
                    env._configure_episode_physics(**ranges)
            assert env._episode_physics is randomizer and _changed(nominal, _model_arrays(env.model)) == set()
        assert _changed(live, _model_arrays(env.model)) == set()
        env._configure_episode_physics(**_all_ranges(ROBOT))  # after the block: the plant restored first
        assert _changed(nominal, _model_arrays(env.model)) == set()
        env.reset(seed=2)
        env._restore_episode_physics()
        assert _changed(nominal, _model_arrays(env.model)) == set()
    finally:
        env.close()
        off.close()


def test_reconfiguring_restores_the_plant_first():
    env = _randomized(ROBOT, **_all_ranges(ROBOT))
    off = species_registry.env_class(ROBOT)(**_first_stage_kwargs(ROBOT))
    try:
        env.reset(seed=1)
        assert _changed(_model_arrays(off.model), _model_arrays(env.model))
        env._configure_episode_physics()
        assert env._episode_physics is None and env.episode_physics_factors is None
        assert _changed(_model_arrays(off.model), _model_arrays(env.model)) == set()
    finally:
        env.close()
        off.close()


# ---------------------------------------------------------------------------------------------------- the fingerprint
@pytest.mark.parametrize(("model_id", "stage"), MODEL_STAGES)
def test_a_committed_stage_carries_the_keys_and_the_section_only_where_it_opts_in(model_id, stage):
    env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
    effective = task_fingerprint._effective_env_kwargs(model_id, env_kwargs)
    fingerprint = stage_task_fingerprint(model_id, stage)
    on = any(effective.get(key) is not None for key in EPISODE_PHYSICS_ENV_KEYS)
    if model_id not in OPTED_IN_EPISODE_PHYSICS:
        assert not set(EPISODE_PHYSICS_ENV_KEYS) & set(effective) and not on
    else:
        assert {key for key in EPISODE_PHYSICS_ENV_KEYS if effective.get(key) is None}.isdisjoint(effective)
    assert ("episode_physics" in fingerprint) is on
    if on:
        assert fingerprint["episode_physics"] == {"implementation": EPISODE_PHYSICS_IMPLEMENTATION}


def test_the_section_is_absent_while_the_committed_classes_take_no_key():
    fingerprint = compute_task_fingerprint(
        model_id=ROBOT,
        stage="stance",
        backend="stable-baselines3",
        env_kwargs={"alive_bonus": 1.0},
        plant_identity=None,
        perturbation_manifest=None,
    )
    assert "episode_physics" not in fingerprint and not set(EPISODE_PHYSICS_ENV_KEYS) & set(fingerprint["env"])


def test_the_section_carries_the_tag_while_a_key_is_live_and_names_what_moved(opted_in_robot, monkeypatch):
    """Any live key writes the section and hashes every live key; the off keys are carved out, so an opted-in class
    keeps the committed task; a changed range names env, switching on or off names env and episode_physics, a tag
    bump names episode_physics alone."""
    stance = load_stage_config(ROBOT, "stance")["env_kwargs"]

    def fingerprint(**ranges):
        return stage_task_fingerprint(ROBOT, "stance", env_kwargs={**stance, **ranges}, plant_identity=opted_in_robot)

    off = fingerprint()
    assert "episode_physics" not in off and not set(EPISODE_PHYSICS_ENV_KEYS) & set(off["env"])
    assert off["task_sha256"] == stage_task_fingerprint(ROBOT, "stance", plant_identity=opted_in_robot)["task_sha256"]
    for key in EPISODE_PHYSICS_ENV_KEYS:
        one = fingerprint(**{key: ROBOT_RANGES[key]})
        assert one["episode_physics"] == {"implementation": EPISODE_PHYSICS_IMPLEMENTATION}
        assert one["env"][key] == list(ROBOT_RANGES[key]) and not (set(EPISODE_PHYSICS_ENV_KEYS) - {key}) & set(
            one["env"]
        )
    on = fingerprint(**ROBOT_RANGES)
    assert {key: on["env"][key] for key in EPISODE_PHYSICS_ENV_KEYS} == {k: list(v) for k, v in ROBOT_RANGES.items()}
    changed = fingerprint(**{**ROBOT_RANGES, "randomize_friction": (0.5, 1.0)})
    monkeypatch.setattr(task_fingerprint, "EPISODE_PHYSICS_IMPLEMENTATION", EPISODE_PHYSICS_IMPLEMENTATION + "+next")
    bumped = fingerprint(**ROBOT_RANGES)
    for current, sections in ((changed, ["env"]), (off, ["env", "episode_physics"]), (bumped, ["episode_physics"])):
        with pytest.raises(TaskFingerprintError, match=re.escape(f"differing sections: {sections!r}")):
            validate_recorded_task(on, current, mode="resume_same_stage")
    assert validate_recorded_task(on, on, mode="resume_same_stage") is None


def test_an_existing_species_opting_in_keeps_every_committed_robot_stage_digest(opted_in_robot, monkeypatch):
    """An opted-in class whose stages omit the keys moves no stage's task_sha256 or stage_config.json view."""
    from environments.shared.harnesses import digest_snapshot

    assert set(EPISODE_PHYSICS_ENV_KEYS) <= set(inspect.signature(species_registry.env_class(ROBOT)).parameters)
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
    assert RANDOMIZED_ROBOT in get_behavior_env_class(ROBOT).__mro__
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
