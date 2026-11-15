"""``designlab.stage2.vecpool``: the worker count does not matter (round C design D13; section 4's row of that name).

What is held here:

* a rollout of a seeded stub env on ``PoolVecEnv`` equals the same rollout on stable-baselines3's own ``DummyVecEnv``
  bit for bit: observations, rewards, dones, the infos with ``terminal_observation`` and ``TimeLimit.truncated``, and
  ``reset_infos``; 400 steps of fixed random actions after ``seed(123)`` and ``reset()``, 8 envs, at W = 0, 1, 2, 3
  (uneven blocks) and 4, and at W = 8 with ``STAGE2_SLOW=1``; also with ``Dict`` and ``Tuple`` observations;
* ``seed`` returns ``DummyVecEnv``'s list; ``env_method``, ``get_attr`` and ``set_attr`` keep env order;
  ``set_options`` reaches the next reset and no later one; every argument reaches each env as its own copy;
* a worker's ``sys.modules`` holds neither ``torch`` nor ``stable_baselines3``; its command line is the
  ``worker_cmd`` given; it runs with ``MUJOCO_GL=disable`` and one BLAS thread and refuses anything else;
* a killed worker makes the next ``step_wait`` raise with the worker and its exit code; an env's own exception is
  raised in the parent and the pool goes on; a worker that cannot start fails the constructor;
* ``close()`` leaves no child process and no socket directory; so do an unclosed pool that is collected, a parent
  that exits without closing, and a parent that is killed;
* a 2-update PPO run under ``VecNormalize`` on the pool equals the same run on ``DummyVecEnv`` (policy tensors,
  normalisation statistics).

The stub envs are defined here and named to the workers as ``tests.test_stage2_vecpool:<factory>``: a worker's working
directory is the study directory.  This module therefore imports neither torch nor stable-baselines3 at its top (a
worker imports it): the tests that need them import them inside and skip without them.  No test touches a reference
set or runs a plant: the envs are stubs.
"""

from __future__ import annotations

import copy
import functools
import gc
import glob
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces

from designlab.stage2 import vecpool

STUDY = Path(__file__).resolve().parents[1]
HERE = "tests.test_stage2_vecpool"
STUB = f"{HERE}:make_stub"
N_ENVS = 8
STEPS = 400

slow = pytest.mark.skipif(not os.environ.get("STAGE2_SLOW"), reason="8 worker processes; set STAGE2_SLOW=1")
linux = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="reads /proc")


# ---------------------------------------------------------------------------------------------------------------
# The stub envs (imported by the workers under the name tests.test_stage2_vecpool).
# ---------------------------------------------------------------------------------------------------------------


class StubEnv(gymnasium.Env):
    """8 observations, 3 actions.  Its own generator, seeded at a seeded reset and carried on through the resets
    without a seed; episodes of 5 to 40 steps that end terminated, truncated or both.  Odd tags return float64
    observations (the vec env casts them into its float32 buffer; ``terminal_observation`` keeps the env's own)."""

    metadata = {"render_modes": [], "stub": "vecpool"}

    def __init__(self, tag: int = 0, gain: float = 1.0, raw64: bool = False, shared: list | None = None,
                 closed: str | None = None):
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.tag, self.gain, self.raw64 = int(tag), float(gain), bool(raw64)
        self.calls = 0
        self.shared, self.closed = shared, closed
        if shared is not None:
            shared.append(self.tag)
        self._mix = np.random.default_rng(77).normal(size=(8, 3))
        self._rng = np.random.default_rng(10_000 + self.tag)
        self._episodes = 0
        self._t, self._length, self._end, self._x = 0, 1, "terminated", np.zeros(8)

    def _obs(self):
        return self._x.copy() if self.raw64 else self._x.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episodes += 1
        self._t = 0
        self._length = int(self._rng.integers(5, 41))
        self._end = ("terminated", "truncated", "both")[int(self._rng.choice(3, p=[0.45, 0.45, 0.10]))]
        self._x = self._rng.normal(size=8)
        if options:
            self._x = self._x + float(options.get("offset", 0.0))
        return self._obs(), {"seed": seed, "episode": self._episodes, "length": self._length, "options": options}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        self._t += 1
        self._x = 0.9 * self._x + self.gain * (self._mix @ a) + 0.05 * self._rng.normal(size=8)
        reward = float(np.tanh(self._x[0]) - 0.01 * (a @ a))
        done = self._t >= self._length
        terminated = bool(done and self._end in ("terminated", "both"))
        truncated = bool(done and self._end in ("truncated", "both"))
        return self._obs(), reward, terminated, truncated, {"t": self._t, "tag": self.tag, "x": self._x.copy()}

    # what the tests call through env_method
    def bump(self, k: int = 1, *, scale: int = 1):
        self.calls += k * scale
        return self.tag, self.calls

    def keep(self, box: list):
        box.append(self.tag)
        self.box = box
        return list(box)

    def mark(self):
        self.shared["list"].append(self.tag)
        return self.shared

    def whoami(self) -> int:
        return os.getpid()

    def modules(self) -> list[str]:
        return sorted(sys.modules)

    def environ(self, name: str):
        return os.environ.get(name)

    def fail(self):
        raise KeyError(f"stub env {self.tag} was asked to fail")

    def deaf(self, seconds: float):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(seconds)

    def close(self):
        if self.closed is not None:
            Path(f"{self.closed}.{self.tag}").write_text(str(os.getpid()))


def make_stub(**kwargs) -> StubEnv:
    return StubEnv(**kwargs)


def make_broken(**kwargs):
    raise ValueError("the stub factory refuses")


def make_not_an_env(**kwargs):
    return object()


class KeyedStub(gymnasium.Env):
    """A ``Dict`` or ``Tuple`` observation of two boxes (float32 and int64); episodes of 3 to 8 steps."""

    def __init__(self, tag: int = 0, kind: str = "dict"):
        parts = (spaces.Box(-np.inf, np.inf, (4,), np.float32), spaces.Box(0, 99, (2,), np.int64))
        self.observation_space = spaces.Dict({"pos": parts[0], "count": parts[1]}) if kind == "dict" \
            else spaces.Tuple(parts)
        self.action_space = spaces.Discrete(3)
        self.tag, self.kind = int(tag), kind
        self._rng = np.random.default_rng(500 + self.tag)
        self._t, self._length = 0, 1

    def _obs(self):
        pos = self._rng.normal(size=4).astype(np.float32)
        count = np.array([self._t, self.tag], dtype=np.int64)
        return {"pos": pos, "count": count} if self.kind == "dict" else (pos, count)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t, self._length = 0, int(self._rng.integers(3, 9))
        return self._obs(), {"length": self._length}

    def step(self, action):
        self._t += 1
        done = self._t >= self._length
        return self._obs(), float(int(action) - 1), bool(done and self.tag % 2 == 0), \
            bool(done and self.tag % 2 == 1), {"t": self._t}


def make_keyed(**kwargs) -> KeyedStub:
    return KeyedStub(**kwargs)


# ---------------------------------------------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------------------------------------------


def _kwargs(i: int) -> dict:
    return {"tag": i, "gain": 1.0 + 0.1 * i, "raw64": bool(i % 2)}


def _specs(n: int = N_ENVS) -> list:
    return [(STUB, _kwargs(i)) for i in range(n)]


def _pool(specs, workers, worker_cmd=None):
    pytest.importorskip("stable_baselines3")
    return vecpool.PoolVecEnv(specs, workers, worker_cmd)


def _dummy(n: int = N_ENVS):
    """The reference: stable-baselines3's own ``DummyVecEnv`` on this module's stub."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([functools.partial(make_stub, **_kwargs(i)) for i in range(n)])


def _same(a, b, where: str = "") -> None:
    """``a`` and ``b`` are the same value bit for bit: the same types, dtypes, shapes and bytes, all the way down."""
    assert type(a) is type(b), f"{where}: {type(a).__name__} against {type(b).__name__}"
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, f"{where}: {a.dtype}{a.shape} against {b.dtype}{b.shape}"
        assert a.tobytes() == b.tobytes(), f"{where}: arrays differ"
    elif isinstance(a, np.generic):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), f"{where}: {a!r} against {b!r}"
    elif isinstance(a, dict):
        assert list(a) == list(b), f"{where}: keys {list(a)} against {list(b)}"
        for k in a:
            _same(a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{where}: lengths {len(a)} and {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    elif isinstance(a, float):
        assert struct.pack("<d", a) == struct.pack("<d", b), f"{where}: {a!r} against {b!r}"
    else:
        assert a == b, f"{where}: {a!r} against {b!r}"


def _rollout(venv, steps: int = STEPS, seed: int = 123, action=None) -> dict:
    """Seed, reset, then ``steps`` steps of fixed random actions; everything the vec env returned and held."""
    rng = np.random.default_rng(7)
    if action is None:
        actions = rng.uniform(-1.0, 1.0, (steps, venv.num_envs, 3)).astype(np.float32)
    else:
        actions = action(rng, steps, venv.num_envs)
    out = {"seeds": list(venv.seed(seed)), "obs0": venv.reset(), "reset_infos0": copy.deepcopy(venv.reset_infos),
           "steps": []}
    for a in actions:
        obs, rews, dones, infos = venv.step(a)
        out["steps"].append({"obs": obs, "rews": rews, "dones": dones, "infos": infos,
                             "reset_infos": copy.deepcopy(venv.reset_infos)})
    return out


def _alive(pid: int) -> bool:
    """Whether the process exists and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _gone(pids, within: float = 15.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.05)
    return not any(_alive(p) for p in pids)


def _cmdline(pid: int) -> list[str]:
    with open(f"/proc/{pid}/cmdline", "rb") as f:
        return [p.decode() for p in f.read().split(b"\0")[:-1]]


def _socket_dirs() -> set:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "vecpool-*")))


def _child_script_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in (vecpool.AUTHKEY_VAR, vecpool.INDEX_VAR)}
    env.update(PYTHONPATH=".", MUJOCO_GL="disable", PYTHONHASHSEED="0", **{v: "1" for v in vecpool.THREAD_VARS})
    return env


@pytest.fixture(scope="module")
def reference() -> dict:
    """The rollout on ``DummyVecEnv`` that every worker count must reproduce."""
    venv = _dummy()
    try:
        ref = _rollout(venv)
    finally:
        venv.close()
    return ref


# ---------------------------------------------------------------------------------------------------------------
# The worker count does not matter.
# ---------------------------------------------------------------------------------------------------------------


def test_the_reference_rollout_ends_episodes_in_every_way(reference):
    """The stub does what the comparison needs: terminations, truncations, both at once, and float64 raw
    observations in ``terminal_observation`` beside float32 buffers."""
    assert reference["seeds"] == [123 + i for i in range(N_ENVS)]
    assert reference["obs0"].dtype == np.float32 and reference["obs0"].shape == (N_ENVS, 8)
    kinds = {"terminated": 0, "truncated": 0}
    dtypes = set()
    for step in reference["steps"]:
        assert step["rews"].dtype == np.float32 and step["dones"].dtype == np.bool_
        for done, info in zip(step["dones"], step["infos"]):
            assert ("terminal_observation" in info) == bool(done)
            if done:
                dtypes.add(info["terminal_observation"].dtype)
                kinds["truncated" if info["TimeLimit.truncated"] else "terminated"] += 1
    assert kinds["terminated"] > 20 and kinds["truncated"] > 20
    assert dtypes == {np.dtype(np.float32), np.dtype(np.float64)}
    lengths = [i["length"] for step in reference["steps"] for i in step["reset_infos"]]
    assert min(lengths) == 5 and max(lengths) == 40


@pytest.mark.parametrize("workers", [0, 1, 2, 3, 4, pytest.param(8, marks=slow)])
def test_a_rollout_equals_dummy_vec_envs_bit_for_bit(reference, workers):
    """400 steps of 8 envs after ``seed(123)``, ``reset()``: everything ``DummyVecEnv`` returns and holds, at every
    worker count (3: blocks of 3, 3 and 2 envs)."""
    pool = _pool(_specs(), workers)
    try:
        assert pool.num_envs == N_ENVS and pool.workers == workers and len(pool.worker_pids) == workers
        got = _rollout(pool)
    finally:
        pool.close()
    _same(got, reference, f"W={workers}")


@pytest.mark.parametrize("kind", ["dict", "tuple"])
def test_keyed_observations_equal_dummy_vec_envs(kind):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    def actions(rng, steps, n):
        return rng.integers(0, 3, (steps, n))

    ref_env = DummyVecEnv([functools.partial(make_keyed, tag=i, kind=kind) for i in range(4)])
    pool = _pool([(f"{HERE}:make_keyed", {"tag": i, "kind": kind}) for i in range(4)], 2)
    try:
        assert pool.observation_space == ref_env.observation_space and pool.action_space == ref_env.action_space
        ref, got = _rollout(ref_env, 60, 5, actions), _rollout(pool, 60, 5, actions)
    finally:
        pool.close()
        ref_env.close()
    _same(got, ref, kind)
    assert isinstance(got["obs0"], dict if kind == "dict" else tuple)


def test_blocks_are_numpy_array_splits():
    assert vecpool.blocks(8, 3) == [(0, 3), (3, 6), (6, 8)]
    assert vecpool.blocks(8, 1) == [(0, 8)] and vecpool.blocks(8, 8) == [(i, i + 1) for i in range(8)]
    for n in range(1, 10):
        for w in range(1, n + 1):
            parts = np.array_split(range(n), w)
            assert vecpool.blocks(n, w) == [(int(p[0]), int(p[-1]) + 1) for p in parts]
            assert [i for lo, hi in vecpool.blocks(n, w) for i in range(lo, hi)] == list(range(n))


@pytest.fixture(scope="class")
def pair():
    """One pool of 8 envs in 3 workers (blocks 0-2, 3-5, 6-7) and ``DummyVecEnv`` on the same stub; closed when the
    class that uses it is done, so that no worker idles beside the later tests."""
    pool, ref = _pool(_specs(), 3), _dummy()
    yield pool, ref
    pool.close()
    ref.close()


class TestOnePool:
    """The calls of a ``VecEnv`` on one pool of 3 workers, each against ``DummyVecEnv``'s answer."""

    def test_spaces_metadata_and_blocks(self, pair):
        pool, ref = pair
        assert pool.observation_space == ref.observation_space and pool.action_space == ref.action_space
        assert pool.metadata == ref.metadata == StubEnv.metadata
        assert pool.render_mode is None and pool.num_envs == ref.num_envs == N_ENVS
        assert pool.blocks == [(0, 3), (3, 6), (6, 8)] and len(set(pool.worker_pids)) == 3
        assert os.getpid() not in pool.worker_pids

    def test_seed_returns_dummy_vec_envs_list_and_seeds_by_the_pools_index(self, pair):
        pool, ref = pair
        assert pool.seed(40) == ref.seed(40) == [40 + i for i in range(N_ENVS)]
        _same(pool.reset(), ref.reset(), "reset after seed(40)")
        # env i got 40 + i whichever worker hosts it, once: the next reset is not seeded
        assert [i["seed"] for i in pool.reset_infos] == [40 + i for i in range(N_ENVS)]
        _same(pool.reset_infos, ref.reset_infos, "reset_infos")
        _same(pool.reset(), ref.reset(), "the next reset")
        assert [i["seed"] for i in pool.reset_infos] == [None] * N_ENVS
        _same(pool.reset_infos, ref.reset_infos, "reset_infos of the next reset")

    def test_env_method_get_attr_and_set_attr_keep_env_order(self, pair):
        pool, ref = pair
        assert pool.get_attr("tag") == ref.get_attr("tag") == list(range(N_ENVS))
        for indices in ([5, 0, 7, 2], 3, [6], [7, 6, 5, 4, 3, 2, 1, 0], -1, [-8, 4], range(2, 7), np.arange(3)):
            assert pool.get_attr("tag", indices) == ref.get_attr("tag", indices), indices
        assert pool.env_method("bump") == ref.env_method("bump") == [(i, 1) for i in range(N_ENVS)]
        for venv in pair:
            assert venv.env_method("bump", 2, indices=[6, 1], scale=5) == [(6, 11), (1, 11)]
            assert venv.env_method("bump", indices=[1, 6, 3]) == [(1, 12), (6, 12), (3, 2)]
        assert pool.get_attr("calls") == ref.get_attr("calls") == [1, 12, 1, 2, 1, 1, 12, 1]
        assert pool.set_attr("gain", 2.5, indices=[1, 6]) is None
        ref.set_attr("gain", 2.5, indices=[1, 6])
        assert pool.get_attr("gain") == ref.get_attr("gain")
        assert pool.get_attr("gain", [1, 6, 0]) == [2.5, 2.5, 1.0]
        pool.set_attr("gain", 1.0)
        ref.set_attr("gain", 1.0)
        assert pool.get_attr("gain") == ref.get_attr("gain") == [1.0] * N_ENVS
        with pytest.raises(IndexError):
            pool.get_attr("tag", [8])

    def test_each_env_answers_from_its_own_worker(self, pair):
        pool, _ = pair
        pids = pool.env_method("whoami")
        assert pids == [pool.worker_pids[k] for k in (0, 0, 0, 1, 1, 1, 2, 2)]
        assert pool.env_method("whoami", indices=[7, 0, 4]) == [pool.worker_pids[k] for k in (2, 0, 1)]

    def test_set_options_reaches_the_next_reset_and_no_later_one(self, pair):
        pool, ref = pair
        for options in ({"offset": 3.0}, [{"offset": float(i)} for i in range(N_ENVS)]):
            for venv in pair:
                venv.set_options(options)
                venv.seed(9)
            a, b = pool.reset(), ref.reset()
            _same(a, b, "the reset with options")
            _same(pool.reset_infos, ref.reset_infos, "reset_infos")
            want = [options] * N_ENVS if isinstance(options, dict) else options
            assert [i["options"] for i in pool.reset_infos] == want
            for venv in pair:
                venv.seed(9)
            plain = pool.reset()
            _same(plain, ref.reset(), "the reset after it")
            assert [i["options"] for i in pool.reset_infos] == [None] * N_ENVS
            offsets = np.array([o["offset"] for o in want])[:, None]
            assert np.abs((a - plain) - offsets).max() < 1e-5

    def test_a_worker_imports_neither_torch_nor_stable_baselines3(self, pair):
        pool, _ = pair
        assert "stable_baselines3" in sys.modules  # the parent does
        for names in pool.env_method("modules"):
            heavy = [m for m in names if m.split(".")[0] in ("torch", "stable_baselines3")]
            assert heavy == []
            assert {"numpy", "gymnasium", "designlab.stage2.vecpool", HERE} <= set(names)

    def test_a_worker_runs_without_gl_on_one_thread_and_forgets_its_key(self, pair):
        pool, _ = pair
        assert pool.env_method("environ", "MUJOCO_GL") == ["disable"] * N_ENVS
        for name in vecpool.THREAD_VARS:
            assert pool.env_method("environ", name) == ["1"] * N_ENVS
        assert pool.env_method("environ", "PYTHONHASHSEED") == ["0"] * N_ENVS
        for name in (vecpool.AUTHKEY_VAR, vecpool.INDEX_VAR):
            assert pool.env_method("environ", name) == [None] * N_ENVS
            assert name not in os.environ

    @linux
    def test_the_default_command_line(self, pair):
        pool, _ = pair
        for pid in pool.worker_pids:
            parts = _cmdline(pid)
            assert parts[:-1] == [sys.executable, "-m", "designlab.stage2.vecpool", "--worker"]
            assert os.path.dirname(parts[-1]) == pool.socket_dir
        assert not os.path.exists(pool.socket_dir)  # gone as soon as the workers were connected

    def test_arguments_reach_each_env_as_its_own_copy(self, pair):
        pool, _ = pair
        box = []
        assert pool.env_method("keep", box) == [[i] for i in range(N_ENVS)]
        assert pool.get_attr("box") == [[i] for i in range(N_ENVS)] and box == []
        value = {"list": []}
        pool.set_attr("shared", value)
        assert pool.env_method("mark") == [{"list": [i]} for i in range(N_ENVS)] and value == {"list": []}
        pool.set_attr("shared", None)

    def test_an_envs_exception_is_raised_here_and_the_pool_goes_on(self, pair):
        pool, ref = pair
        with pytest.raises(KeyError, match="stub env 3 was asked to fail") as err:
            pool.env_method("fail", indices=[7, 3, 4])  # both workers fail; the first in worker order is raised
        notes = "\n".join(getattr(err.value, "__notes__", []))
        assert "raised in vecpool worker 1" in notes and "in fail" in notes
        with pytest.raises(AttributeError):
            pool.get_attr("no_such_attribute")
        assert pool.has_attr("tag") and not pool.has_attr("no_such_attribute")
        with pytest.raises(TypeError):
            pool.env_method("bump", "one", "two", "three")
        with pytest.raises(Exception, match="pickle|Can't"):
            pool.env_method("bump", lambda: 1)  # refused here: no worker saw it
        assert pool.get_attr("tag") == ref.get_attr("tag")

    def test_env_is_wrapped_is_false(self, pair):
        pool, ref = pair
        assert pool.env_is_wrapped(gymnasium.Wrapper) == ref.env_is_wrapped(gymnasium.Wrapper) == [False] * N_ENVS
        assert pool.env_is_wrapped(gymnasium.Wrapper, indices=[2, 5]) == [False, False]

    def test_step_wait_steps_with_the_actions_kept(self, pair):
        """``step_async`` keeps the actions and ``step_wait`` steps, as ``DummyVecEnv``: a second ``step_wait``
        steps again with the same actions."""
        pool, ref = pair
        for venv in pair:
            venv.seed(77)
        _same(pool.reset(), ref.reset(), "reset")
        actions = np.random.default_rng(3).uniform(-1, 1, (N_ENVS, 3)).astype(np.float32)
        for venv in pair:
            venv.step_async(actions)
        _same(pool.step_wait(), ref.step_wait(), "first step_wait")
        _same(pool.step_wait(), ref.step_wait(), "second step_wait")


def test_no_two_envs_share_a_kwarg():
    """One list object named in every spec: each env gets its own copy, in this process and in workers."""
    for workers in (0, 2):
        shared = []
        pool = _pool([(STUB, {"tag": i, "shared": shared}) for i in range(4)], workers)
        try:
            assert pool.get_attr("shared") == [[i] for i in range(4)] and shared == []
            assert pool.env_method("keep", []) == [[i] for i in range(4)]
        finally:
            pool.close()


def test_the_module_imports_without_torch_and_makes_the_class_once():
    code = ("import sys\nimport designlab.stage2.vecpool as v\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'stable_baselines3')]\n"
            "sys.exit(3 if bad else 0)\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=STUDY, env=_child_script_env(), capture_output=True)
    assert done.returncode == 0, done.stderr.decode()
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import VecEnv

    cls = vecpool.PoolVecEnv
    assert cls is vecpool.PoolVecEnv and issubclass(cls, VecEnv)
    assert (cls.__module__, cls.__qualname__) == ("designlab.stage2.vecpool", "PoolVecEnv")
    with pytest.raises(AttributeError):
        vecpool.no_such_name


def test_child_env_is_the_track_b_child_environment(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("VECPOOL_TEST_KEPT", "yes")
    env = vecpool.child_env()
    assert env["MUJOCO_GL"] == "disable" and "PYOPENGL_PLATFORM" not in env
    assert [env[v] for v in vecpool.THREAD_VARS] == ["1", "1", "1"]
    assert env["PYTHONHASHSEED"] == "0" and env["PYTHONPATH"] == "." and env["VECPOOL_TEST_KEPT"] == "yes"
    assert os.environ["MUJOCO_GL"] == "osmesa"  # the parent's own environment is not touched
    assert vecpool.environment_problem(env) is None
    try:
        from designlab.stage2 import tbenv
    except ImportError:
        return
    assert env == tbenv.child_env("disable") and vecpool.study_dir() == tbenv.study_dir()


def test_the_study_directory_is_the_workers_working_directory():
    assert Path(vecpool.study_dir()) == STUDY and (STUDY / "designlab" / "stage2" / "vecpool.py").is_file()


# ---------------------------------------------------------------------------------------------------------------
# Command line, failure and clean-up.
# ---------------------------------------------------------------------------------------------------------------

_WORKER_C = "import sys; from designlab.stage2 import vecpool; sys.exit(vecpool.worker_main(sys.argv[-1]))"


@linux
def test_a_workers_command_line_is_the_worker_cmd_given():
    """The caller's command line, then the address: what a guard that protects by a substring of it reads."""
    cmd = [sys.executable, "-c", _WORKER_C, "designlab.stage2.train2 worker (a stand-in)"]
    pool = _pool(_specs(2), 2, worker_cmd=cmd)
    try:
        assert pool.worker_cmd == cmd
        for pid in pool.worker_pids:
            parts = _cmdline(pid)
            assert parts[:-1] == cmd and os.path.dirname(parts[-1]) == pool.socket_dir
            assert "designlab.stage2.train2 worker" in " ".join(parts)
        assert pool.env_method("whoami") == pool.worker_pids
    finally:
        pool.close()


@linux
def test_killing_a_worker_makes_the_next_step_wait_raise():
    pool = _pool(_specs(4), 2)
    try:
        pool.seed(1)
        pool.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        pool.step(actions)
        pids = pool.worker_pids
        os.kill(pids[1], signal.SIGKILL)
        pool.step_async(actions)
        with pytest.raises(RuntimeError, match=rf"vecpool worker 1 \(pid {pids[1]}, envs 2 to 3\) is gone: "
                                               r"exit code -9 \(SIGKILL\)"):
            pool.step_wait()
        for call in (pool.reset, pool.step_wait, lambda: pool.get_attr("tag"), lambda: pool.env_method("bump")):
            with pytest.raises(RuntimeError, match="vecpool worker 1 .* is gone: exit code -9"):
                call()
    finally:
        pool.close()
    assert _gone(pids, 0.0)


@linux
def test_close_leaves_no_child_process_and_no_socket_directory():
    before = _socket_dirs()
    pool = _pool(_specs(4), 2)
    pids = pool.worker_pids
    assert len(pids) == 2 and all(_alive(p) for p in pids)
    assert os.path.dirname(pool.socket_dir) == tempfile.gettempdir()
    pool.reset()
    pool.close()
    assert not any(_alive(p) for p in pids)
    assert not os.path.exists(pool.socket_dir) and _socket_dirs() <= before
    pool.close()  # a second close does nothing
    for call in (pool.reset, lambda: pool.get_attr("tag"), lambda: pool.step(np.zeros((4, 3), np.float32))):
        with pytest.raises(RuntimeError, match="closed"):
            call()


@linux
def test_close_closes_every_env_in_its_own_process(tmp_path):
    """``DummyVecEnv.close`` closes each env; so does the pool, in the env's worker, before the worker leaves."""
    for workers in (0, 2):
        mark = str(tmp_path / f"closed{workers}")
        pool = _pool([(STUB, {"tag": i, "closed": mark}) for i in range(4)], workers)
        pids = pool.worker_pids
        t0 = time.monotonic()
        pool.close()
        assert time.monotonic() - t0 < vecpool.CLOSE_WAIT_S  # the workers left by themselves
        said = [int(Path(f"{mark}.{i}").read_text()) for i in range(4)]
        assert said == ([os.getpid()] * 4 if workers == 0 else [pids[0], pids[0], pids[1], pids[1]])


@linux
def test_a_worker_that_does_not_go_is_killed(monkeypatch):
    """A pool closed while a worker is busy and deaf to SIGTERM: terminated, then killed after ``TERM_WAIT_S``."""
    monkeypatch.setattr(vecpool, "TERM_WAIT_S", 0.5)
    pool = _pool(_specs(2), 2)
    pids = pool.worker_pids

    def cut(worker):
        time.sleep(0.5)  # worker 1 is in its method by now
        raise KeyboardInterrupt

    monkeypatch.setattr(pool, "_read", cut)
    with pytest.raises(KeyboardInterrupt):
        pool.env_method("deaf", 600.0, indices=[1])
    assert all(_alive(p) for p in pids)
    t0 = time.monotonic()
    pool.close()
    assert not any(_alive(p) for p in pids) and 0.4 < time.monotonic() - t0 < 10.0
    worker = pool._workers[1]
    assert worker.proc.returncode == -signal.SIGKILL and pool._workers[0].proc.returncode is not None


@linux
def test_an_unclosed_pool_that_is_collected_takes_its_workers_with_it():
    pool = _pool(_specs(2), 1)
    pids = pool.worker_pids
    assert all(_alive(p) for p in pids)
    del pool
    gc.collect()
    assert _gone(pids, 0.0)


_PARENT = f"""
import sys, time
from designlab.stage2 import vecpool
pool = vecpool.PoolVecEnv([("{STUB}", {{"tag": i}}) for i in range(2)], 2)
pool.reset()
print(*pool.worker_pids, pool.socket_dir, flush=True)
if sys.argv[1] == "exit":
    sys.exit(0)
time.sleep(600)
"""


@linux
@pytest.mark.parametrize("how", ["exit", "kill"])
def test_workers_leave_with_their_parent(how):
    """A parent that exits without closing its pool cleans up at exit; a parent that is killed cleans nothing up,
    and its workers leave because their connection closed."""
    pytest.importorskip("stable_baselines3")
    proc = subprocess.Popen([sys.executable, "-c", _PARENT, how], cwd=STUDY, env=_child_script_env(),
                            stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline().split()
        assert len(line) == 3, f"the parent printed {line!r}"
        pids, sockdir = [int(line[0]), int(line[1])], line[2]
        if how == "kill":
            assert all(_alive(p) for p in pids)
            proc.kill()
            assert proc.wait(timeout=30) == -signal.SIGKILL
        else:
            assert proc.wait(timeout=60) == 0
        assert _gone(pids)
        assert not os.path.exists(sockdir)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def test_a_worker_that_cannot_start_fails_the_constructor(tmp_path):
    from multiprocessing.connection import Listener

    with Listener(str(tmp_path / "s"), family="AF_UNIX") as listener:  # what makes the wait below a bounded one
        assert hasattr(vecpool.listening_socket(listener), "settimeout")
    before = _socket_dirs()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"vecpool worker 0 \(pid \d+, envs 0 to 1\) left before it connected: "
                                           r"exit code 7"):
        _pool(_specs(2), 1, worker_cmd=[sys.executable, "-c", "import sys; sys.exit(7)"])
    assert time.monotonic() - t0 < 30.0 and _socket_dirs() <= before


def test_a_worker_refuses_another_gl_backend_or_thread_count():
    assert vecpool.environment_problem({"MUJOCO_GL": "disable", **{v: "1" for v in vecpool.THREAD_VARS}}) is None
    said = vecpool.environment_problem({"MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2"})
    assert "MUJOCO_GL='egl'" in said and "MKL_NUM_THREADS='2'" in said and "OPENBLAS_NUM_THREADS=None" in said
    assert "OMP_NUM_THREADS" not in said
    code = ("import os, sys; os.environ['MUJOCO_GL'] = 'osmesa'; os.environ['OMP_NUM_THREADS'] = '4'; "
            "from designlab.stage2 import vecpool; sys.exit(vecpool.worker_main(sys.argv[-1]))")
    before = _socket_dirs()
    with pytest.raises(RuntimeError, match="MUJOCO_GL='osmesa', OMP_NUM_THREADS='4'"):
        _pool(_specs(2), 1, worker_cmd=[sys.executable, "-c", code])
    assert _socket_dirs() <= before


def test_a_worker_is_not_started_by_hand(tmp_path):
    done = subprocess.run([sys.executable, "-m", "designlab.stage2.vecpool", "--worker", str(tmp_path / "s")],
                          cwd=STUDY, env=_child_script_env(), capture_output=True, text=True)
    assert done.returncode == vecpool.EXIT_NOT_A_POOL and "started by PoolVecEnv" in done.stderr


def test_a_factory_that_fails_fails_the_constructor_with_its_exception():
    before = _socket_dirs()
    for workers in (0, 1):
        with pytest.raises(ValueError, match="the stub factory refuses") as err:
            _pool([(STUB, {"tag": 0}), (f"{HERE}:make_broken", {})], workers)
        if workers:
            assert "raised in vecpool worker 0" in "\n".join(err.value.__notes__)
        with pytest.raises(ValueError, match="not a gymnasium.Env"):
            _pool([(f"{HERE}:make_not_an_env", {})], workers)
        with pytest.raises(ModuleNotFoundError):
            _pool([("tests.no_such_module_for_vecpool:make", {})], workers)
        with pytest.raises(TypeError, match="unexpected keyword"):
            _pool([(STUB, {"no_such_kwarg": 1})], workers)
    assert _socket_dirs() <= before


def test_specs_are_refused_before_anything_starts():
    pytest.importorskip("stable_baselines3")
    before = _socket_dirs()
    for workers in (0, 2):
        with pytest.raises(TypeError, match="do not travel by pickle"):
            vecpool.PoolVecEnv([(STUB, {"tag": 0}), (STUB, {"tag": lambda: 1})], workers)
    for specs, workers, exc in (([], 0, ValueError), (_specs(2), 3, ValueError), (_specs(2), -1, ValueError),
                                ([("no_colon", {})], 1, ValueError), ([(":f", {})], 1, ValueError),
                                ([(STUB, [1, 2])], 1, ValueError), ([(STUB, {1: 2})], 1, ValueError),
                                ([(STUB,)], 1, ValueError), (_specs(2), 1.5, TypeError)):
        with pytest.raises(exc):
            vecpool.PoolVecEnv(specs, workers)
    with pytest.raises(ValueError, match="worker_cmd is empty"):
        vecpool.PoolVecEnv(_specs(2), 1, worker_cmd=[])
    assert _socket_dirs() <= before
    assert vecpool.resolve(STUB)(tag=4).tag == 4
    with pytest.raises(ValueError):
        vecpool.resolve("designlab.stage2.vecpool")


def test_an_interrupted_exchange_breaks_the_pool(monkeypatch):
    """A call cut between its commands and their replies (Ctrl-C in the trainer) must not let the next call read the
    replies of the last: the pool refuses every later call."""
    pool = _pool(_specs(2), 1)
    try:
        pool.reset()
        real = pool._read

        def cut(worker):
            raise KeyboardInterrupt

        monkeypatch.setattr(pool, "_read", cut)
        with pytest.raises(KeyboardInterrupt):
            pool.get_attr("tag")
        monkeypatch.setattr(pool, "_read", real)
        with pytest.raises(RuntimeError, match="interrupted"):
            pool.get_attr("tag")
        pids = pool.worker_pids
    finally:
        pool.close()
    assert _gone(pids, 0.0) or not os.path.isdir("/proc/self")


# ---------------------------------------------------------------------------------------------------------------
# Under stable-baselines3's own training loop.
# ---------------------------------------------------------------------------------------------------------------


def _ppo_run(venv) -> dict:
    """Two PPO updates under ``VecNormalize`` (16 steps of 8 envs each); the policy, the optimizer's moments and the
    normalisation statistics."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    torch.set_num_threads(1)
    vecnorm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0, gamma=0.99)
    try:
        model = PPO("MlpPolicy", vecnorm, n_steps=16, batch_size=32, n_epochs=2, gamma=0.99, seed=11, device="cpu",
                    policy_kwargs={"net_arch": [16, 16]}, verbose=0)
        model.learn(total_timesteps=2 * 16 * N_ENVS)
        moments = [state[k].clone() for state in model.policy.optimizer.state_dict()["state"].values()
                   for k in ("exp_avg", "exp_avg_sq")]
        return {"steps": int(model.num_timesteps),
                "policy": {k: v.detach().clone() for k, v in model.policy.state_dict().items()},
                "moments": moments,
                "obs_rms": (vecnorm.obs_rms.mean.copy(), vecnorm.obs_rms.var.copy(), float(vecnorm.obs_rms.count)),
                "ret_rms": (np.array(vecnorm.ret_rms.mean), np.array(vecnorm.ret_rms.var),
                            float(vecnorm.ret_rms.count)),
                "returns": vecnorm.returns.copy(), "last_obs": np.array(model._last_obs),
                "last_original_obs": np.array(vecnorm.get_original_obs())}
    finally:
        vecnorm.close()


def test_ppo_under_vecnormalize_is_the_same_run_on_the_pool():
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    ref = _ppo_run(_dummy())
    got = _ppo_run(_pool(_specs(), 2))
    assert got["steps"] == ref["steps"] == 2 * 16 * N_ENVS
    assert list(got["policy"]) == list(ref["policy"]) and len(ref["policy"]) >= 9
    for name, tensor in ref["policy"].items():
        assert torch.equal(got["policy"][name], tensor), name
    assert len(got["moments"]) == len(ref["moments"]) > 0
    assert all(torch.equal(a, b) for a, b in zip(got["moments"], ref["moments"]))
    for key in ("obs_rms", "ret_rms", "returns", "last_obs", "last_original_obs"):
        _same(got[key], ref[key], key)
    assert ref["obs_rms"][2] > 2 * 16 * N_ENVS  # the statistics did move
    assert any(v.abs().sum() > 0 for v in ref["moments"])  # the optimizer did step
