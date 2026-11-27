"""The opt-in actuation delay: ``environments/shared/actuation_delay.py`` and its FIFO in ``BaseDinoEnv.step``.

Design record ``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §5.  No committed model
lists the delay keys, so the delay itself runs on a test-only subclass of the
Compsognathus robot (2 ms physics, ``frame_skip`` 10) that opts in
(``actuation_delay_helpers.py``).  Numpy and MuJoCo only: no SB3 (the trainers'
marking is tested in ``test_train_base.py`` and ``test_behavior_checkpoint.py``).
"""

from __future__ import annotations

import copy
import inspect
import pickle
import re
from collections import deque

import mujoco
import numpy as np
import pytest

from environments.compsognathus_longipes.robot.env import CompsognathusLongipesRobotEnv
from environments.shared import base_env, species_registry, task_fingerprint
from environments.shared.actuation_delay import (
    ACTUATION_DELAY_ENV_KEYS,
    ACTUATION_DELAY_IMPLEMENTATION,
    ActuationDelay,
    delay_steps,
    primed_fifo,
    validate_actuation_delay,
)
from environments.shared.base_env import BaseDinoEnv
from environments.shared.config import load_stage_config
from environments.shared.episode_start import ACTUATION_DELAY_STREAM_KEY, feature_stream, mark_training_rollouts
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.species_names import model_ids
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import (
    TaskFingerprintError,
    compute_task_fingerprint,
    stage_task_fingerprint,
    validate_recorded_task,
)

from . import actuation_delay_helpers
from .actuation_delay_helpers import DELAYED_ENV_CLASSES, DelayedCompsognathusLongipesRobotEnv

ROBOT = ModelId("compsognathus_longipes", "robot")
DT = 0.002
FRAME_SKIP = 10
NU = 12

#: The manifest models that opt into the delay: each lists both keys, defaulting to ``None``, forwards them to
#: ``BaseDinoEnv`` and declares the carve-out at off value ``None``.  Empty until the Triceratops robot joins it (design
#: record §8.9); every other model refuses the keys.
OPTED_IN_ACTUATION_DELAY: frozenset[ModelId] = frozenset()

#: The carve-out an opted-in class declares in ``OPT_IN_TASK_KWARGS``.
DELAY_CARVE_OUT = ("actuation_delay_range_s", None, ACTUATION_DELAY_ENV_KEYS)

MODEL_STAGES = [
    pytest.param(model_id, entry.reference, id=f"{model_id.species}-{model_id.embodiment}-{entry.id}")
    for model_id in model_ids()
    for entry in load_stage_manifest(model_id).stages
]


@pytest.fixture(scope="module")
def stance_kwargs():
    return dict(load_stage_config(ROBOT, "stance")["env_kwargs"])


def _delayed(stance_kwargs, range_s, eval_s=None, **overrides):
    return DelayedCompsognathusLongipesRobotEnv(
        **{**stance_kwargs, **overrides}, actuation_delay_range_s=range_s, actuation_delay_eval_s=eval_s
    )


def _steps(k: int) -> float:
    return k * DT


# ---------------------------------------------------------------------------------------------------- the module
def test_delay_steps_counts_whole_physics_steps():
    assert delay_steps(0.010, 0.001, name="d") == 10
    assert delay_steps(0.010, 0.002, name="d") == 5
    assert delay_steps(0.0, 0.002, name="d") == 0
    assert delay_steps(0.009, 0.001, name="d") == 9
    assert delay_steps(np.float64(0.020), 0.001, name="d") == 20
    assert delay_steps(0, 0.002, name="d") == 0


def test_delay_steps_refuses_a_value_off_the_step_grid_naming_the_timestep_and_the_nearest_value():
    with pytest.raises(ValueError, match=r"0\.003 s is not a whole number of physics steps of 0\.002 s") as error:
        delay_steps(0.003, 0.002, name="actuation_delay_eval_s")
    assert "actuation_delay_eval_s" in str(error.value) and "the nearest whole value is 2 steps" in str(error.value)


@pytest.mark.parametrize("value", [True, False, "0.01", None, [0.01], float("nan"), float("inf"), -0.002])
def test_delay_steps_refuses_a_bool_a_non_number_a_non_finite_or_a_negative_value(value):
    with pytest.raises(ValueError, match="delay"):
        delay_steps(value, 0.002, name="delay")


def test_the_validated_delay_is_none_while_off_and_whole_steps_while_on():
    assert validate_actuation_delay(None, None, 0.001) is None
    assert validate_actuation_delay((0.0, 0.020), 0.010, 0.001) == ActuationDelay(
        range_s=(0.0, 0.02), eval_s=0.01, timestep=0.001, range_steps=(0, 20), eval_steps=10
    )
    # The eval value need not lie in the range; TOML arrays arrive as lists or tuples.
    config = validate_actuation_delay([0.002, 0.004], 0.010, 0.002)
    assert config is not None and (config.range_steps, config.eval_steps) == ((1, 2), 5)
    assert validate_actuation_delay((0.004, 0.004), None, 0.002).eval_steps is None


#: The Compsognathus robot's stance episode in physics steps (1000 control steps x frame_skip 10).
EPISODE_STEPS = 1000 * FRAME_SKIP

#: (range, eval, the refusal) at the robot's 2 ms step: every way a config is refused.
INVALID_CONFIGS = [
    pytest.param(None, 0.010, "needs actuation_delay_range_s", id="eval without a range"),
    pytest.param("0.0, 0.02", None, "must be a pair", id="a string range"),
    pytest.param((0.0, 0.01, 0.02), None, "must be a pair", id="three values"),
    pytest.param(0.01, None, "must be a pair", id="one value"),
    pytest.param((0.02, 0.01), None, "lo <= hi", id="lo above hi"),
    pytest.param((-0.002, 0.01), None, "nonnegative", id="negative"),
    pytest.param((0.0, float("nan")), None, "finite", id="not finite"),
    pytest.param((0.0, 0.003), None, "not a whole number of physics steps", id="off the step grid"),
    pytest.param((0.0, 0.01), 0.003, "not a whole number of physics steps", id="eval off the step grid"),
    pytest.param((True, 0.01), None, "number of seconds", id="a bool"),
    pytest.param((0.0, 0.01), "0.01", "number of seconds", id="a string eval"),
    # Milliseconds typed into the seconds keys: every target, or the eval value's, lands after the episode ends.
    pytest.param(
        (0.0, 20.0),
        None,
        r"actuation_delay_range_s\[1\] = 20\.0 s is 10000 physics steps of 0\.002 s, at least the episode's 10000",
        id="a range as long as the episode",
    ),
    pytest.param(
        (0.0, 0.02),
        25.0,
        r"actuation_delay_eval_s = 25\.0 s is 12500 physics steps of 0\.002 s, at least the episode's 10000",
        id="an eval value longer than the episode",
    ),
]


@pytest.mark.parametrize(("range_s", "eval_s", "message"), INVALID_CONFIGS)
def test_an_invalid_config_is_refused(range_s, eval_s, message):
    with pytest.raises(ValueError, match=message):
        validate_actuation_delay(range_s, eval_s, DT, episode_steps=EPISODE_STEPS)


@pytest.mark.parametrize(("range_s", "eval_s", "message"), INVALID_CONFIGS[:7] + INVALID_CONFIGS[-2:])
def test_an_invalid_config_is_refused_at_construction(stance_kwargs, range_s, eval_s, message):
    with pytest.raises(ValueError, match=message):
        _delayed(stance_kwargs, range_s, eval_s)


def test_a_delay_one_physics_step_short_of_the_episode_is_accepted_and_the_ceiling_follows_the_episode(stance_kwargs):
    """The ceiling is the episode's physics steps as constructed, so a shorter episode lowers it; without
    episode_steps (a bare validation) there is none."""
    last = _steps(EPISODE_STEPS - 1)
    assert validate_actuation_delay((0.0, last), last, DT, episode_steps=EPISODE_STEPS).range_steps == (
        0,
        EPISODE_STEPS - 1,
    )
    assert validate_actuation_delay((0.0, 20.0), 25.0, DT).eval_steps == 12500
    env = _delayed(stance_kwargs, (0.0, last), last)
    try:
        assert env._actuation_delay.eval_steps == EPISODE_STEPS - 1
    finally:
        env.close()
    with pytest.raises(ValueError, match=r"at least the episode's 50 \(max_episode_steps x frame_skip\)"):
        _delayed(stance_kwargs, (0.0, _steps(50)), None, max_episode_steps=5)


def test_the_primed_fifo_holds_float64_copies_with_room_for_one_more():
    hold = np.arange(3, dtype=np.float32)
    fifo = primed_fifo(hold, 4)
    assert len(fifo) == 4 and fifo.maxlen == 5
    assert all(entry.dtype == np.float64 and np.array_equal(entry, hold) for entry in fifo)
    assert len({id(entry) for entry in fifo}) == 4 and all(entry is not hold for entry in fifo)
    empty = primed_fifo(hold, 0)
    assert len(empty) == 0 and empty.maxlen == 1


# ---------------------------------------------------------------------------------------------------- off by default
def test_the_base_env_takes_both_keys_after_the_command_kwargs_off_by_default():
    """Right after command_config; the later opt-in features' keys follow them (test_episode_physics.py)."""
    parameters = list(inspect.signature(BaseDinoEnv.__init__).parameters.values())
    names = [parameter.name for parameter in parameters]
    start = names.index("command_config") + 1
    assert names[start : start + 2] == list(ACTUATION_DELAY_ENV_KEYS)
    assert all(parameter.default is None for parameter in parameters[start : start + 2])


def test_the_models_listing_a_key_are_exactly_the_opted_in_table():
    assert OPTED_IN_ACTUATION_DELAY <= set(model_ids())
    listing = {
        model_id
        for model_id in model_ids()
        if set(ACTUATION_DELAY_ENV_KEYS) & set(inspect.signature(species_registry.env_class(model_id)).parameters)
    }
    assert listing == OPTED_IN_ACTUATION_DELAY


@pytest.mark.parametrize("model_id", model_ids(), ids=str)
def test_an_opted_in_model_takes_both_keys_off_by_default_and_every_other_refuses_them(model_id):
    """An opted-in class lists both keys at default None, forwards them to BaseDinoEnv and declares the carve-out; a
    TOML setting either key on any other model raises TypeError."""
    env_class = species_registry.env_class(model_id)
    parameters = inspect.signature(env_class).parameters
    if model_id in OPTED_IN_ACTUATION_DELAY:
        assert all(parameters[key].default is None for key in ACTUATION_DELAY_ENV_KEYS)
        assert DELAY_CARVE_OUT in tuple(getattr(env_class, "OPT_IN_TASK_KWARGS", ()))
        env = env_class(actuation_delay_range_s=(0.0, 0.0), actuation_delay_eval_s=0.0)
        try:
            assert env._actuation_delay is not None and env._actuation_delay.range_steps == (0, 0)
        finally:
            env.close()
        return
    assert not set(ACTUATION_DELAY_ENV_KEYS) & set(parameters)
    for key, value in zip(ACTUATION_DELAY_ENV_KEYS, ((0.0, 0.01), 0.01)):
        with pytest.raises(TypeError, match=key):
            env_class(**{key: value})


def test_the_test_subclasses_opt_in_as_a_model_would():
    for env_class in DELAYED_ENV_CLASSES:
        parameters = inspect.signature(env_class).parameters
        assert all(parameters[key].default is None for key in ACTUATION_DELAY_ENV_KEYS)
        assert DELAY_CARVE_OUT in env_class.OPT_IN_TASK_KWARGS


@pytest.mark.parametrize(("model_id", "stage"), MODEL_STAGES)
def test_the_delay_is_off_exactly_when_the_stage_omits_the_keys(model_id, stage):
    env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
    env = species_registry.env_class(model_id)(**env_kwargs)
    try:
        assert (env._actuation_delay is None) == ("actuation_delay_range_s" not in env_kwargs)
        assert env.training_rollouts is False
        if env._actuation_delay is None:
            env.reset(seed=1)
            assert env.actuation_delay_steps is None
            *_, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert not {"actuation_delay_steps", "actuation_delay_s"} & set(info)
    finally:
        env.close()


def _state(env) -> bytes:
    data = env.data
    return b"".join(
        np.asarray(value).tobytes()
        for value in (data.qpos, data.qvel, data.ctrl, data.act, data.time, data.xfrc_applied)
    ) + pickle.dumps(env.np_random.bit_generator.state)


@pytest.mark.parametrize("model_id", model_ids(), ids=str)
def test_marking_an_env_without_the_delay_changes_no_byte(model_id):
    """A trainer's mark is inert while every feature is off: a seeded zero-action then random-action rollout is
    byte-identical in its observations, rewards, infos, simulator state and generator.  The env is built from the
    model's first stage, or from its constructor defaults (every feature off) where that stage sets the delay."""
    stage = load_stage_manifest(model_id).stages[0].reference
    env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
    if set(ACTUATION_DELAY_ENV_KEYS) & set(env_kwargs):
        assert model_id in OPTED_IN_ACTUATION_DELAY
        env_kwargs = {}
    env_class = species_registry.env_class(model_id)
    plain, marked = env_class(**env_kwargs), env_class(**env_kwargs)
    try:
        mark_training_rollouts(marked, True)
        assert (plain.training_rollouts, marked.training_rollouts) == (False, True)
        rows = []
        for env in (plain, marked):
            obs, info = env.reset(seed=2026)
            record = [obs.tobytes(), pickle.dumps(info), _state(env)]
            rng = np.random.default_rng(7)
            shape = env.action_space.shape
            actions = [np.zeros(shape, dtype=np.float32)] * 15 + [
                rng.uniform(-0.5, 0.5, size=shape).astype(np.float32) for _ in range(15)
            ]
            for action in actions:
                obs, reward, terminated, truncated, info = env.step(action)
                assert not {"actuation_delay_steps", "actuation_delay_s"} & set(info)
                record += [obs.tobytes(), pickle.dumps((reward, terminated, truncated, info)), _state(env)]
            rows.append(record)
        assert rows[0] == rows[1]
    finally:
        plain.close()
        marked.close()


# ---------------------------------------------------------------------------------------------------- the exact delay
@pytest.mark.parametrize("k", [0, 1, 3, 10, 25])
def test_each_target_reaches_the_plant_exactly_k_physics_steps_later(stance_kwargs, k):
    """Per substep, the applied ctrl is the reset ctrl for the episode's first k substeps, then the target of control
    step (s - k) // frame_skip; the trajectory is bit-equal to an independent FIFO on a copied MjData."""
    env = _delayed(stance_kwargs, (_steps(k), _steps(k)))
    try:
        assert (env.frame_skip, env.model.opt.timestep, env.action_filter_cutoff_hz) == (FRAME_SKIP, DT, 0.0)
        env.reset(seed=11)
        assert env.actuation_delay_steps == k
        prime = env.data.ctrl.copy()
        applied: list[np.ndarray] = []
        env._substep_probe_hook = lambda: applied.append(env.data.ctrl.copy())
        shadow = copy.deepcopy(env.data)
        shadow_fifo = deque(prime.copy() for _ in range(k))
        rng = np.random.default_rng(k)
        targets = []
        for _ in range(40):
            action = rng.uniform(-0.2, 0.2, size=NU)
            target = np.asarray(env._scale_action(action), dtype=np.float64)
            targets.append(target)
            *_, info = env.step(action)
            assert (info["actuation_delay_steps"], info["actuation_delay_s"]) == (k, round(k * DT, 12))
            for _ in range(FRAME_SKIP):
                shadow_fifo.append(target)
                shadow.ctrl[:] = shadow_fifo.popleft()
                mujoco.mj_step(env.model, shadow)
            for name in ("qpos", "qvel", "ctrl", "act"):
                assert np.array_equal(getattr(env.data, name), getattr(shadow, name)), name
            assert env.data.time == shadow.time
        assert len(applied) == 40 * FRAME_SKIP
        for substep, ctrl in enumerate(applied):
            expected = prime if substep < k else targets[(substep - k) // FRAME_SKIP]
            assert np.array_equal(ctrl, expected), substep
        # The FIFO holds the last k + 1 targets, the oldest first.
        assert len(env._actuation_delay_fifo) == k + 1
        assert np.array_equal(env._actuation_delay_fifo[-1], targets[-1])
    finally:
        env.close()


def test_a_zero_step_delay_is_bit_identical_to_the_delay_off(stance_kwargs):
    delayed, plain = _delayed(stance_kwargs, (0.0, 0.0)), CompsognathusLongipesRobotEnv(**stance_kwargs)
    try:
        rows = []
        for env in (delayed, plain):
            obs, _ = env.reset(seed=5)
            record = [obs.tobytes(), _state(env)]
            rng = np.random.default_rng(3)
            for _ in range(40):
                obs, reward, terminated, truncated, info = env.step(rng.uniform(-0.3, 0.3, size=NU))
                info = {key: value for key, value in info.items() if not key.startswith("actuation_delay")}
                record += [obs.tobytes(), pickle.dumps((reward, terminated, truncated, info)), _state(env)]
            rows.append(record)
        assert delayed.actuation_delay_steps == 0
        assert rows[0] == rows[1]
    finally:
        delayed.close()
        plain.close()


def test_the_seconds_in_info_are_rounded_and_the_steps_exact(stance_kwargs):
    env = _delayed(stance_kwargs, (_steps(9), _steps(9)))
    try:
        env.reset(seed=0)
        *_, info = env.step(np.zeros(NU))
        assert info["actuation_delay_steps"] == 9 and isinstance(info["actuation_delay_steps"], int)
        assert 9 * DT != 0.018 and repr(info["actuation_delay_s"]) == "0.018"
    finally:
        env.close()


def test_stepping_a_delayed_env_before_its_first_reset_is_refused(stance_kwargs):
    env = _delayed(stance_kwargs, (0.0, 0.010))
    try:
        assert env.actuation_delay_steps is None
        with pytest.raises(RuntimeError, match="only after reset"):
            env.step(np.zeros(NU))
    finally:
        env.close()


def test_every_episode_primes_the_fifo_again_from_its_own_reset(stance_kwargs):
    """An evaluation env holds one k and is reset many times (EvalCallback, CLI evaluation, the reports): each
    episode clears the FIFO, so its first k substeps hold that reset's ctrl, never the previous episode's last k
    in-flight targets, and substep k applies the new episode's first target."""
    env = _delayed(stance_kwargs, (0.0, 0.040), 0.014)
    try:
        rng = np.random.default_rng(12)
        env.reset(seed=1)
        for _ in range(10):
            env.step(rng.uniform(-0.5, 0.5, size=NU))
        leaked = env._actuation_delay_fifo[-1].copy()
        env.reset(seed=2)
        k = env.actuation_delay_steps
        prime = env.data.ctrl.copy()
        assert k == 7 < FRAME_SKIP and not np.array_equal(leaked, prime)
        applied = []
        env._substep_probe_hook = lambda: applied.append(env.data.ctrl.copy())
        action = rng.uniform(-0.5, 0.5, size=NU)
        target = np.asarray(env._scale_action(action), dtype=np.float64)
        *_, info = env.step(action)
        assert len(applied) == FRAME_SKIP and info["actuation_delay_steps"] == k
        assert all(np.array_equal(ctrl, prime) for ctrl in applied[:k])
        assert all(np.array_equal(ctrl, target) for ctrl in applied[k:])
        assert len(env._actuation_delay_fifo) == k + 1
    finally:
        env.close()


def test_a_delayed_env_pickles_and_its_copies_continue_bit_identically(stance_kwargs):
    """Each test subclass is bound in its module under its own name, so pickle finds it by reference (as a
    SubprocVecEnv or a pickled env attribute needs); a pickled and a deep-copied env continue mid-episode, the FIFO
    and the delay's stream included, bit for bit."""
    for env_class in DELAYED_ENV_CLASSES:
        assert getattr(actuation_delay_helpers, env_class.__qualname__) is env_class
        assert pickle.loads(pickle.dumps(env_class)) is env_class
    assert actuation_delay_helpers.with_actuation_delay(CompsognathusLongipesRobotEnv) is (
        DelayedCompsognathusLongipesRobotEnv
    )
    env = _delayed(stance_kwargs, (0.0, 0.040))
    twins = []
    try:
        mark_training_rollouts(env, True)
        env.reset(seed=21)
        rng = np.random.default_rng(21)
        for _ in range(5):
            env.step(rng.uniform(-0.3, 0.3, size=NU))
        twins = [copy.deepcopy(env), pickle.loads(pickle.dumps(env))]
        actions = [rng.uniform(-0.3, 0.3, size=NU) for _ in range(10)]
        rows = []
        for each in (env, *twins):
            record = [len(each._actuation_delay_fifo)]
            for action in actions:
                obs, reward, *_ = each.step(action)
                record += [obs.tobytes(), each.data.qpos.tobytes(), np.float64(reward).tobytes()]
            for _ in range(3):
                each.reset()
                record.append(each.actuation_delay_steps)
            rows.append(record)
        assert rows[0] == rows[1] == rows[2]
    finally:
        for each in (env, *twins):
            each.close()


# ---------------------------------------------------------------------------------------------------- who draws what
def test_an_evaluation_env_holds_the_eval_value(stance_kwargs):
    env = _delayed(stance_kwargs, (0.0, 0.040), 0.010)
    try:
        assert env.training_rollouts is False
        for seed in range(20):
            env.reset(seed=seed)
            assert env.actuation_delay_steps == 5
            *_, info = env.step(np.zeros(NU))
            assert (info["actuation_delay_steps"], info["actuation_delay_s"]) == (5, 0.01)
    finally:
        env.close()


def test_a_training_env_draws_its_delay_from_its_own_stream_each_episode(stance_kwargs):
    env = _delayed(stance_kwargs, (0.0, 0.040), 0.010)
    try:
        mark_training_rollouts(env, True)
        drawn = []
        for seed in range(60):
            env.reset(seed=seed)
            k = env.actuation_delay_steps
            # Chosen at reset, so readable before the first step, which then applies it.
            assert k == int(feature_stream(env, seed, ACTUATION_DELAY_STREAM_KEY).integers(0, 21))
            *_, info = env.step(np.zeros(NU))
            assert info["actuation_delay_steps"] == k and 0 <= k <= 20
            drawn.append(k)
        assert len(set(drawn)) >= 15, sorted(set(drawn))
        # Unmarked again, the env holds the eval value from its next reset on.
        mark_training_rollouts(env, False)
        env.reset(seed=3)
        assert env.actuation_delay_steps == 5
    finally:
        env.close()


def test_unseeded_episodes_continue_the_stream_whatever_the_actions(stance_kwargs):
    envs = [_delayed(stance_kwargs, (0.0, 0.040)) for _ in range(2)]
    try:
        sequences = []
        for index, env in enumerate(envs):
            mark_training_rollouts(env, True)
            rng = np.random.default_rng(index)
            env.reset(seed=8)
            ks = [env.actuation_delay_steps]
            for _ in range(12):
                for _ in range(3):
                    env.step(np.zeros(NU) if index == 0 else rng.uniform(-0.5, 0.5, size=NU))
                env.reset()
                ks.append(env.actuation_delay_steps)
            sequences.append(ks)
        assert sequences[0] == sequences[1]
        assert len(set(sequences[0])) > 3, sequences[0]
        stream = feature_stream(envs[0], 8, ACTUATION_DELAY_STREAM_KEY)
        assert sequences[0] == [int(stream.integers(0, 21)) for _ in range(13)]
    finally:
        for env in envs:
            env.close()


def test_a_training_env_whose_delay_changes_between_episodes_holds_k_plus_one_entries(stance_kwargs):
    env = _delayed(stance_kwargs, (0.0, 0.040), 0.010)
    try:
        mark_training_rollouts(env, True)
        rng = np.random.default_rng(4)
        ks = []
        for seed in (30, None, None, 31, None, None, None, 32):
            env.reset(seed=seed)
            k = env.actuation_delay_steps
            for _ in range(3):
                *_, info = env.step(rng.uniform(-0.3, 0.3, size=NU))
                assert len(env._actuation_delay_fifo) == k + 1 == info["actuation_delay_steps"] + 1
            ks.append(k)
        assert sum(a != b for a, b in zip(ks, ks[1:])) >= 4, ks
    finally:
        env.close()


def test_a_first_reset_without_a_seed_builds_the_stream_from_np_random_seed(stance_kwargs):
    """The stream is built on an env's first reset even when it is unseeded (from np_random_seed, which gymnasium
    takes from OS entropy) and continued across the unseeded resets after it."""
    env = _delayed(stance_kwargs, (0.0, 0.040))
    try:
        mark_training_rollouts(env, True)
        env.reset()
        assert env._actuation_delay_rng is not None and env.np_random_seed >= 0
        stream = feature_stream(env, None, ACTUATION_DELAY_STREAM_KEY)
        ks = [env.actuation_delay_steps]
        for _ in range(9):
            env.step(np.zeros(NU))
            env.reset()
            ks.append(env.actuation_delay_steps)
        assert ks == [int(stream.integers(0, 21)) for _ in range(10)]
        assert len(set(ks)) > 3, ks
    finally:
        env.close()


def test_an_evaluation_env_without_an_eval_value_draws_from_the_range(stance_kwargs):
    env = _delayed(stance_kwargs, (0.0, 0.040))
    try:
        assert env.training_rollouts is False
        drawn = set()
        for seed in range(30):
            env.reset(seed=seed)
            drawn.add(env.actuation_delay_steps)
        assert len(drawn) >= 10 and drawn <= set(range(21))
    finally:
        env.close()


@pytest.mark.parametrize(
    ("training", "eval_k", "held"),
    [(False, None, 3), (True, None, 3), (False, 5, 5), (True, 5, 3), (False, 0, 0), (True, 0, 3)],
)
def test_a_degenerate_range_never_draws(stance_kwargs, monkeypatch, training, eval_k, held):
    """k is lo itself, with no call of the stream, except in an evaluation env with an eval value, which holds that
    value instead (record §5 takes the eval value first, so a run trained at one fixed delay is evaluated at another
    or at none; an eval value of 0 is a value, not an absent one).  numpy's integers(lo, lo + 1) leaves PCG64's state
    as it was, so the call is observed, not the state.  Every stream the env builds refuses to draw; the same patch
    makes a drawing range fail, so the refusal is live."""

    class NoDraws:
        def integers(self, *args, **kwargs):
            raise AssertionError("the delay drew from its stream")

    built = []

    def stream(env, seed, key):
        assert key == ACTUATION_DELAY_STREAM_KEY
        built.append(seed)
        return NoDraws()

    monkeypatch.setattr(base_env, "_feature_stream", stream)
    env = _delayed(stance_kwargs, (_steps(3), _steps(3)), None if eval_k is None else _steps(eval_k))
    drawing = _delayed(stance_kwargs, (0.0, _steps(3)))
    try:
        mark_training_rollouts(env, training)
        for seed in (0, None, 1, None, None):
            env.reset(seed=seed)
            assert env.actuation_delay_steps == held
            *_, info = env.step(np.zeros(NU))
            assert info["actuation_delay_steps"] == held
        assert built == [0, 1]
        mark_training_rollouts(drawing, training)
        with pytest.raises(AssertionError, match="drew from its stream"):
            drawing.reset(seed=0)
    finally:
        env.close()
        drawing.close()


def test_the_delay_draws_nothing_from_np_random(stance_kwargs):
    """A drawing training env's reset observation and np_random state equal the delay-free env's, every episode."""
    delayed, plain = _delayed(stance_kwargs, (0.0, 0.040)), CompsognathusLongipesRobotEnv(**stance_kwargs)
    try:
        mark_training_rollouts(delayed, True)
        for seed in (4, None, None, 9, None):
            obs_delayed, _ = delayed.reset(seed=seed)
            obs_plain, _ = plain.reset(seed=seed)
            assert obs_delayed.tobytes() == obs_plain.tobytes()
            assert delayed.np_random.bit_generator.state == plain.np_random.bit_generator.state
            for _ in range(3):
                delayed.step(np.zeros(NU))
                plain.step(np.zeros(NU))
    finally:
        delayed.close()
        plain.close()


def test_the_fifo_holds_the_filtered_scaled_target(stance_kwargs):
    """With the shared low-pass on, the FIFO carries the filtered command, as the plant would have received it."""
    filtered_class = type(
        "FilteredDelayedRobot", (DelayedCompsognathusLongipesRobotEnv,), {"action_filter_cutoff_hz": 5.0}
    )
    env = filtered_class(**stance_kwargs, actuation_delay_range_s=(_steps(4), _steps(4)))
    try:
        env.reset(seed=1)
        rng = np.random.default_rng(1)
        differed = 0
        for _ in range(10):
            action = rng.uniform(-0.5, 0.5, size=NU)
            env.step(action)
            expected = np.asarray(env._scale_action(env._action_filter_state), dtype=np.float64)
            assert np.array_equal(env._actuation_delay_fifo[-1], expected)
            differed += not np.array_equal(expected, env._scale_action(action))
        assert differed >= 9
    finally:
        env.close()


def test_a_pushed_zero_action_rollout_is_bit_identical_with_the_delay():
    """Zero action commands the reset ctrl, the FIFO's prime, so a delay changes nothing; pushes included."""
    recovery = load_stage_config(ROBOT, "recovery")["env_kwargs"]
    delayed = DelayedCompsognathusLongipesRobotEnv(
        **recovery, actuation_delay_range_s=(0.0, 0.040), actuation_delay_eval_s=0.014
    )
    plain = CompsognathusLongipesRobotEnv(**recovery)
    try:
        rows, pushed = [], 0
        for env in (delayed, plain):
            obs, _ = env.reset(seed=3042)
            record = [obs.tobytes()]
            for _ in range(250):
                obs, reward, terminated, truncated, _ = env.step(np.zeros(NU))
                pushed += bool(np.any(env.data.xfrc_applied))
                record += [
                    obs.tobytes(),
                    env.data.qpos.tobytes(),
                    env.data.qvel.tobytes(),
                    np.float64(reward).tobytes(),
                ]
                if terminated or truncated:
                    break
            rows.append(record)
        assert delayed.actuation_delay_steps == 7
        assert np.array_equal(delayed._scale_action(np.zeros(NU)), delayed.model.key_ctrl[delayed._reset_keyframe_id])
        assert pushed > 0 and len(rows[0]) > 1
        assert rows[0] == rows[1]
    finally:
        delayed.close()
        plain.close()


def test_plant_validation_holds_mid_episode(stance_kwargs):
    from environments.shared.plant_contract import current_plant_identity, validate_environment_plant

    env = _delayed(stance_kwargs, (0.0, 0.040), 0.010)
    try:
        mark_training_rollouts(env, True)
        env.reset(seed=6)
        rng = np.random.default_rng(6)
        for _ in range(20):
            env.step(rng.uniform(-0.3, 0.3, size=NU))
        validate_environment_plant(env, current_plant_identity(ROBOT), artifact="delayed robot mid-episode")
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- the fingerprint
@pytest.mark.parametrize(("model_id", "stage"), MODEL_STAGES)
def test_a_committed_stage_carries_the_delay_and_its_section_only_where_it_opts_in(model_id, stage):
    """No stage of a model outside the opted-in table carries either key or the section; an opted-in model's stage
    carries the section exactly when it sets the range, and a stage that leaves the delay off drops both keys (the
    carve-out)."""
    env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
    effective = task_fingerprint._effective_env_kwargs(model_id, env_kwargs)
    fingerprint = stage_task_fingerprint(model_id, stage)
    on = effective.get("actuation_delay_range_s") is not None
    if model_id not in OPTED_IN_ACTUATION_DELAY:
        assert not set(ACTUATION_DELAY_ENV_KEYS) & set(effective) and not on
    elif not on:
        assert not set(ACTUATION_DELAY_ENV_KEYS) & set(effective)
    assert ("actuation_delay" in fingerprint) is on
    if on:
        assert fingerprint["actuation_delay"] == {"implementation": ACTUATION_DELAY_IMPLEMENTATION}


def test_the_section_is_written_exactly_while_the_delay_is_on():
    def fingerprint(**env_kwargs):
        return compute_task_fingerprint(
            model_id=ROBOT,
            stage="stance",
            backend="stable-baselines3",
            env_kwargs=env_kwargs,
            plant_identity=None,
            perturbation_manifest=None,
        )

    assert "actuation_delay" not in fingerprint()
    # The committed robot does not list the keys; the registry's class decides what the env section carries.
    assert "actuation_delay" not in fingerprint(alive_bonus=1.0)


@pytest.fixture
def opted_in_robot(monkeypatch):
    """The registry resolving the Compsognathus robot to its opted-in test subclass (carve-out at off value None).

    The behavior env classes are cached per model, so the cache is emptied on both sides of the test.
    """
    from environments.shared.behavior_env import _behavior_env_class
    from environments.shared.plant_contract import current_plant_identity

    identity = current_plant_identity(ROBOT)
    real = species_registry.env_class
    monkeypatch.setattr(
        species_registry,
        "env_class",
        lambda model_id: DelayedCompsognathusLongipesRobotEnv if model_id == ROBOT else real(model_id),
    )
    _behavior_env_class.cache_clear()
    yield identity
    _behavior_env_class.cache_clear()


def test_an_existing_species_opting_in_keeps_every_committed_robot_stage_digest(opted_in_robot, monkeypatch):
    """An opted-in class whose stages omit the keys moves no stage's task_sha256 or stage_config.json view."""
    from environments.shared.harnesses import digest_snapshot

    assert set(ACTUATION_DELAY_ENV_KEYS) <= set(inspect.signature(species_registry.env_class(ROBOT)).parameters)
    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.setattr(digest_snapshot, "_models", lambda: [ROBOT])
    out = digest_snapshot._Snapshot(REPOSITORY_ROOT, keep=True)
    digest_snapshot.stage_section(out, {ROBOT: opted_in_robot})
    golden = (REPOSITORY_ROOT / "configs" / "digest_snapshot.generated.txt").read_text(encoding="utf-8").splitlines()
    prefix = f"stage\t{ROBOT.species}\t{ROBOT.embodiment}\t"
    committed = [line for line in golden if line.startswith(prefix)]
    assert out.errors == 0 and len(committed) == 28
    assert out.kept == committed


def test_an_existing_species_opting_in_keeps_every_committed_robot_behavior_digest(opted_in_robot, monkeypatch):
    """The same for every behavior recipe's task, built through the behavior env of the opted-in class."""
    from environments.shared.behavior_env import get_behavior_env_class
    from environments.shared.harnesses import digest_snapshot

    monkeypatch.chdir(REPOSITORY_ROOT)
    assert DelayedCompsognathusLongipesRobotEnv in get_behavior_env_class(ROBOT).__mro__
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


def test_the_delay_section_names_what_moved(opted_in_robot, monkeypatch):
    """A changed value is refused naming env; switching the delay off names env and actuation_delay; a tag bump
    names actuation_delay alone."""
    stance = load_stage_config(ROBOT, "stance")["env_kwargs"]

    def fingerprint(**delay):
        return stage_task_fingerprint(ROBOT, "stance", env_kwargs={**stance, **delay}, plant_identity=opted_in_robot)

    on = fingerprint(actuation_delay_range_s=(0.0, 0.020), actuation_delay_eval_s=0.010)
    assert on["actuation_delay"] == {"implementation": ACTUATION_DELAY_IMPLEMENTATION}
    assert on["env"]["actuation_delay_range_s"] == [0.0, 0.02] and on["env"]["actuation_delay_eval_s"] == 0.01
    off = fingerprint()
    assert "actuation_delay" not in off and "actuation_delay_range_s" not in off["env"]
    # Off, the opted-in class keeps the committed task.
    assert off["task_sha256"] == stage_task_fingerprint(ROBOT, "stance", plant_identity=opted_in_robot)["task_sha256"]
    changed = fingerprint(actuation_delay_range_s=(0.0, 0.020), actuation_delay_eval_s=0.012)
    monkeypatch.setattr(task_fingerprint, "ACTUATION_DELAY_IMPLEMENTATION", ACTUATION_DELAY_IMPLEMENTATION + "+next")
    bumped = fingerprint(actuation_delay_range_s=(0.0, 0.020), actuation_delay_eval_s=0.010)
    for current, sections in ((changed, ["env"]), (off, ["env", "actuation_delay"]), (bumped, ["actuation_delay"])):
        with pytest.raises(TaskFingerprintError, match=re.escape(f"differing sections: {sections!r}")):
            validate_recorded_task(on, current, mode="resume_same_stage")
    assert validate_recorded_task(on, on, mode="resume_same_stage") is None
