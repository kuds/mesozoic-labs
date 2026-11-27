"""The episode-start hook, the per-feature streams and the role flag (``environments/shared/episode_start.py``).

Design record ``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §4.  Numpy, gymnasium
and MuJoCo only: no SB3.  The trainers' marking itself (``make_env``, which
needs SB3, and ``train_behaviors.main``) is exercised in ``test_train_base.py``
and ``test_behavior_checkpoint.py``; the AST scan here pins where it happens.
"""

from __future__ import annotations

import ast
import pickle
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils import seeding

from environments.shared import species_registry
from environments.shared.base_env import BaseDinoEnv
from environments.shared.behavior_env import SpeciesBehaviorMixin, get_behavior_env_class
from environments.shared.episode_physics import EpisodePhysicsRandomizer
from environments.shared.episode_start import (
    ACTUATION_DELAY_STREAM_KEY,
    EPISODE_PHYSICS_STREAM_KEY,
    FEATURE_STREAM_KEYS,
    OBSERVATION_NOISE_STREAM_KEY,
    EpisodeStartHook,
    feature_stream,
    mark_training_rollouts,
)
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.species_names import model_ids

from .actuation_delay_helpers import DelayedCompsognathusLongipesRobotEnv

ROBOT = ModelId("compsognathus_longipes", "robot")
ENVIRONMENTS = REPOSITORY_ROOT / "environments"


def _stream(entropy: int, key: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(key,))))


def _same(a: np.random.Generator, b: np.random.Generator) -> bool:
    return a.bit_generator.state == b.bit_generator.state


# ---------------------------------------------------------------------------------------------------- the hook
@pytest.mark.parametrize("model_id", model_ids(), ids=str)
def test_every_model_resets_through_the_hook(model_id):
    """The hook sits right after BaseDinoEnv in every model's MRO, canonical and behavior env alike, and no species
    overrides reset: the policy layer's hashed home_reset is still BaseDinoEnv.reset (or the behavior mixin's)."""
    canonical = species_registry.env_class(model_id)
    behavior = get_behavior_env_class(model_id)
    for cls, reset in ((canonical, BaseDinoEnv.reset), (behavior, SpeciesBehaviorMixin.reset)):
        mro = cls.__mro__
        assert mro[mro.index(BaseDinoEnv) + 1] is EpisodeStartHook, mro
        assert issubclass(cls, gym.Env)
        assert cls.reset is reset, cls


def test_the_hook_begins_the_episode_after_gymnasium_seeds_np_random():
    calls = []

    class Probe(EpisodeStartHook):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

        def _begin_episode(self, seed):
            calls.append((seed, self.np_random_seed, self.np_random.bit_generator.state))

    probe = Probe()
    assert probe.reset(seed=17) is None
    probe.reset()
    expected = seeding.np_random(17)[0].bit_generator.state
    assert calls == [(17, 17, expected), (None, 17, expected)]


def test_the_hook_may_refuse_a_reset_before_gymnasium_seeds_anything():
    """_check_episode_may_begin runs first: a reset it refuses reseeds nothing and begins no episode."""
    calls = []

    class Probe(EpisodeStartHook):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        refuse = False

        def _check_episode_may_begin(self):
            calls.append("check")
            if self.refuse:
                raise RuntimeError("no episode now")

        def _begin_episode(self, seed):
            calls.append(("begin", seed))

    probe = Probe()
    probe.reset(seed=17)
    state = probe.np_random.bit_generator.state
    probe.refuse = True
    for seed in (5, None):
        with pytest.raises(RuntimeError, match="no episode now"):
            probe.reset(seed=seed)
    assert probe.np_random_seed == 17 and probe.np_random.bit_generator.state == state
    assert calls == ["check", ("begin", 17), "check", "check"]
    assert EpisodeStartHook._check_episode_may_begin(probe) is None  # nothing by default


def test_the_hook_runs_before_the_keyframe_is_loaded():
    env = DelayedCompsognathusLongipesRobotEnv(actuation_delay_range_s=(0.0, 0.010))
    seen = []
    begin = env._begin_actuation_delay_episode

    def record(seed):
        seen.append(env.data.qpos.copy())
        begin(seed)

    env._begin_actuation_delay_episode = record
    try:
        env.reset(seed=0)
        for _ in range(20):
            env.step(np.full(12, 0.3))
        moved = env.data.qpos.copy()
        env.reset(seed=1)
        assert np.array_equal(seen[-1], moved)
        assert not np.array_equal(env.data.qpos, moved)
    finally:
        env.close()


def _snapshot(env) -> dict[str, object]:
    """Every instance attribute's value (each pickled whole: ``env.model`` and ``env.data`` with every array) and
    the identity of each, so a write to an attribute's value, to the model or data, or a rebinding all show."""
    return {name: (id(value), pickle.dumps(value)) for name, value in vars(env).items()}


@pytest.mark.parametrize("model_id", model_ids(), ids=str)
def test_with_every_feature_off_the_hook_reads_and_writes_nothing(model_id, monkeypatch):
    def refuse(self, seed):
        raise AssertionError("a feature that is off began an episode")

    monkeypatch.setattr(BaseDinoEnv, "_begin_actuation_delay_episode", refuse)
    monkeypatch.setattr(EpisodePhysicsRandomizer, "begin_episode", refuse)
    monkeypatch.setattr(EpisodePhysicsRandomizer, "refuse_inside_nominal", refuse)
    env = species_registry.env_class(model_id)()
    try:
        assert env._episode_physics is None
        env.reset(seed=3)
        before = _snapshot(env)
        assert {"model", "data", "_np_random", "_distance_traveled", "training_rollouts"} <= set(before)
        env._begin_episode(4)
        env._begin_episode(None)
        after = _snapshot(env)
        assert after.keys() == before.keys()
        assert [name for name in before if after[name] != before[name]] == []
        assert env._actuation_delay_rng is None and env._actuation_delay_k is None
    finally:
        env.close()


def test_the_off_hook_snapshot_sees_a_write_to_a_value_the_model_or_the_data():
    """The snapshot above is not blind: a value written in place, a model or data array written, and a rebinding to
    an equal value each change it."""
    env = species_registry.env_class(ROBOT)()
    try:
        env.reset(seed=3)
        writes = (
            lambda: setattr(env, "_distance_traveled", env._distance_traveled + 123.0),
            lambda: env.model.geom_friction.__setitem__((0, 0), env.model.geom_friction[0, 0] + 0.5),
            lambda: env.data.ctrl.__setitem__(0, env.data.ctrl[0] + 0.25),
            lambda: setattr(env, "_prev_pos_2d", env._prev_pos_2d.copy()),
        )
        for write in writes:
            before = _snapshot(env)
            write()
            assert _snapshot(env) != before
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- the streams
def test_the_stream_keys_are_distinct_named_constants():
    assert FEATURE_STREAM_KEYS == (EPISODE_PHYSICS_STREAM_KEY, ACTUATION_DELAY_STREAM_KEY, OBSERVATION_NOISE_STREAM_KEY)
    assert (EPISODE_PHYSICS_STREAM_KEY, ACTUATION_DELAY_STREAM_KEY, OBSERVATION_NOISE_STREAM_KEY) == (
        0xD0A1,
        0xDE1A,
        0x0B5E4015,
    )
    assert len(set(FEATURE_STREAM_KEYS)) == 3
    for seed in (0, 1, 3042):
        first = {int(feature_stream(None, seed, key).integers(2**63)) for key in FEATURE_STREAM_KEYS}
        gymnasium = int(seeding.np_random(seed)[0].integers(2**63))
        assert len(first) == 3 and gymnasium not in first


def test_a_seed_is_the_entropy_whatever_the_env():
    for env in (None, SimpleNamespace(np_random_seed=99)):
        assert _same(feature_stream(env, 5, ACTUATION_DELAY_STREAM_KEY), _stream(5, ACTUATION_DELAY_STREAM_KEY))
    assert _same(feature_stream(None, np.int64(7), 0xD0A1), _stream(7, 0xD0A1))


def test_without_a_seed_the_entropy_is_np_random_seed():
    env = gym.Env()
    env.reset(seed=21)
    env.np_random.random(5)  # a continued generator keeps its seed
    assert _same(feature_stream(env, None, OBSERVATION_NOISE_STREAM_KEY), _stream(21, OBSERVATION_NOISE_STREAM_KEY))


def test_a_directly_assigned_pcg64_generator_is_read_never_drawn():
    env = gym.Env()
    env.np_random = np.random.Generator(np.random.PCG64(1234))
    env.np_random.random(3)
    assert env.np_random_seed == -1
    before = env.np_random.bit_generator.state
    state = before["state"]
    entropy = (int(state["state"]) << 128) | int(state["inc"])
    stream = feature_stream(env, None, ACTUATION_DELAY_STREAM_KEY)
    assert env.np_random.bit_generator.state == before
    assert _same(stream, _stream(entropy, ACTUATION_DELAY_STREAM_KEY))
    # gymnasium before 1.0 has no np_random_seed: the same state rule.
    legacy = SimpleNamespace(np_random=env.np_random)
    assert _same(feature_stream(legacy, None, ACTUATION_DELAY_STREAM_KEY), _stream(entropy, ACTUATION_DELAY_STREAM_KEY))


def test_another_bit_generator_is_refused():
    env = gym.Env()
    env.np_random = np.random.Generator(np.random.MT19937(5))
    with pytest.raises(ValueError, match="needs a seeded reset or a PCG64 np_random; got MT19937"):
        feature_stream(env, None, ACTUATION_DELAY_STREAM_KEY)
    # A seeded reset never reads the generator.
    assert _same(feature_stream(env, 5, ACTUATION_DELAY_STREAM_KEY), _stream(5, ACTUATION_DELAY_STREAM_KEY))


# ---------------------------------------------------------------------------------------------------- the role
def test_every_env_is_an_evaluation_env_until_a_trainer_marks_it():
    env = species_registry.env_class(ROBOT)()
    try:
        assert env.training_rollouts is False
        mark_training_rollouts(gym.wrappers.TimeLimit(env, 10), True)  # marked at its base
        assert env.training_rollouts is True
        mark_training_rollouts(env, False)
        assert env.training_rollouts is False
        with pytest.raises(TypeError, match="must be a bool"):
            env.set_training_rollouts(1)
        # The role is the env's, not its class's.
        assert BaseDinoEnv.training_rollouts is False
    finally:
        env.close()


def test_marking_a_plain_gymnasium_env_does_nothing():
    class Plain(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    env = Plain()
    before = dict(vars(env))
    for enabled in (True, False):
        assert mark_training_rollouts(env, enabled) is None
        assert vars(env) == before and not hasattr(env, "training_rollouts")


def test_marking_a_vectorised_env_is_refused():
    """A vector env's unwrapped is the vector env itself, which has no set_training_rollouts, so marking one would
    otherwise do nothing and say nothing; each env is marked before it is vectorised (make_env).  The SB3 VecEnv and
    VecNormalize case is in test_train_base.py."""
    env = species_registry.env_class(ROBOT)()
    try:

        class VectorDouble:
            def __init__(self, num_envs=None, env_method=None):
                if num_envs is not None:
                    self.num_envs = num_envs
                if env_method is not None:
                    self.env_method = env_method
                self.unwrapped = self

        for vector in (VectorDouble(num_envs=1), VectorDouble(env_method=lambda *args: None)):
            with pytest.raises(TypeError, match=r"not the vectorised VectorDouble: mark each env before it is"):
                mark_training_rollouts(vector, True)
        mark_training_rollouts(gym.wrappers.TimeLimit(gym.wrappers.OrderEnforcing(env), 10), True)
        assert env.training_rollouts is True
    finally:
        env.close()


def _calls(tree: ast.AST, name: str) -> list[tuple[str, ast.Call]]:
    """(enclosing function qualname, call) for every call of *name*, a bare name or an attribute."""
    found: list[tuple[str, ast.Call]] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = (*scope, node.name)
        if isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if called == name:
                found.append((".".join(scope), node))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return found


def _sources() -> dict[str, ast.AST]:
    """Every non-test module under environments/, parsed."""
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(ENVIRONMENTS.rglob("*.py"))
        if "tests" not in path.relative_to(ENVIRONMENTS).parts
    }


def test_the_trainers_mark_their_rollout_envs_in_exactly_two_places():
    """mark_training_rollouts is called only in make_env (training rollouts) and train_behaviors.main (True before
    training, False before scoring); set_training_rollouts only through it; training=True only on train_env."""
    sources = _sources()
    marks = {
        (module, scope, ast.unparse(call))
        for module, tree in sources.items()
        for scope, call in _calls(tree, "mark_training_rollouts")
    }
    assert marks == {
        ("environments/shared/train_base.py", "make_env._init", "mark_training_rollouts(env, True)"),
        ("environments/shared/train_behaviors.py", "main", "mark_training_rollouts(env, True)"),
        ("environments/shared/train_behaviors.py", "main", "mark_training_rollouts(env, False)"),
    }
    setters = [
        (module, scope) for module, tree in sources.items() for scope, _ in _calls(tree, "set_training_rollouts")
    ]
    assert setters == []  # reached only through mark_training_rollouts' getattr
    writes = [
        module
        for module, tree in sources.items()
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute) and target.attr == "training_rollouts"
    ]
    assert writes == ["environments/shared/base_env.py", "environments/shared/base_env.py"]
    training = [
        (module, scope, ast.unparse(keyword.value))
        for module, tree in sources.items()
        for name in ("create_vec_env", "make_env")
        for scope, call in _calls(tree, name)
        for keyword in call.keywords
        if keyword.arg == "training"
    ]
    assert sorted(training) == [
        ("environments/shared/train_base.py", "_train_stage_body", "True"),
        ("environments/shared/train_base.py", "create_vec_env", "training"),
    ]
    (body,) = [
        node
        for node in ast.walk(sources["environments/shared/train_base.py"])
        if isinstance(node, ast.FunctionDef) and node.name == "_train_stage_body"
    ]
    assigned = {
        ast.unparse(node.targets[0]): node.value
        for node in ast.walk(body)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
    }
    keywords = {
        name: {kw.arg: ast.unparse(kw.value) for kw in assigned[name].keywords} for name in ("train_env", "eval_env")
    }
    assert keywords["train_env"]["training"] == "True" and "training" not in keywords["eval_env"]


def _neighbours(function: ast.FunctionDef, statement: str) -> tuple[str, str]:
    """The statements just before and just after *statement* (its first line), in its own block of *function*."""
    for node in ast.walk(function):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for index, child in enumerate(block):
                if isinstance(child, ast.stmt) and ast.unparse(child).splitlines()[0] == statement:
                    before = ast.unparse(block[index - 1]).splitlines()[0] if index else ""
                    after = ast.unparse(block[index + 1]).splitlines()[0] if index + 1 < len(block) else ""
                    return before, after
    raise AssertionError(f"{statement!r} not found")


def test_the_behavior_trainer_marks_its_env_on_creation_and_unmarks_it_right_before_scoring():
    tree = ast.parse((ENVIRONMENTS / "shared" / "train_behaviors.py").read_text(encoding="utf-8"))
    (main,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    _, after_creation = _neighbours(
        main, "env = create_behavior_env(model_id, commands=commands, terrain=terrain, run_seed=run_seed, **env_kwargs)"
    )
    assert after_creation == "mark_training_rollouts(env, True)"
    _, after_unmark = _neighbours(main, "mark_training_rollouts(env, False)")
    assert after_unmark.startswith("manifest['evaluation'] = evaluate_behavior(model, normalizer, ")
