"""Track B's vec env: N envs in W worker processes, equal to stable-baselines3's ``DummyVecEnv`` at every W.

``PoolVecEnv(specs, workers, worker_cmd=None)`` is a stable-baselines3 ``VecEnv``.  Env i is ``factory(**kwargs)`` of
``specs[i] = ("pkg.module:factory", kwargs)``.  With ``workers == 0`` the envs are built in this process and SB3's own
``DummyVecEnv`` does everything (the reference).  With ``workers >= 1`` the envs live in W child processes, env block
k being ``numpy.array_split(range(N), W)[k]`` (contiguous), and each child runs ``DummyVecEnv``'s loop on its block.
Every env has its own generator and its own plant and nothing is shared, so a rollout does not depend on W: the batch
PPO learns from is the same whatever the machine's core count.  ``tests/test_stage2_vecpool.py`` holds that equality
bit for bit against ``DummyVecEnv`` itself.

Why not SB3's ``SubprocVecEnv``: it starts one process per env through ``multiprocessing``, each importing torch and
stable-baselines3, with a command line the caller cannot choose.  Here

* a worker imports numpy and gymnasium and whatever the env factory imports, never torch or stable-baselines3 (this
  module does not import them either: ``PoolVecEnv`` is built on first access, see ``__getattr__`` at the end);
* a worker is a subprocess started as ``worker_cmd + [address]``, so its command line is the caller's (the trainer
  passes ``[sys.executable, "-m", "designlab.stage2.train2", "worker"]`` and ``tools/memguard.sh`` protects by a
  substring of the command line); ``multiprocessing.Process`` is not used;
* W is chosen freely between 1 and N, so a core can be left to another job.

The worker loop (``_Block``) is ``DummyVecEnv.step_wait`` and ``reset`` statement for statement
(``stable_baselines3/common/vec_env/dummy_vec_env.py``, version 2.9.2): the reward goes into a float32 buffer, ``done =
terminated or truncated``, ``info["TimeLimit.truncated"] = truncated and not terminated``, a finished env's last
observation is ``info["terminal_observation"]`` and the env is reset at once without a seed; ``reset`` passes each env
the seed and the options set for it, once.  The seeds stay where SB3 keeps them, in the pool: ``seed(s)`` is
``VecEnv.seed`` itself, so env i of the whole pool gets ``s + i`` at its next reset whichever worker hosts it.

Start-up.  The pool opens a ``multiprocessing.connection.Listener`` on a Unix socket in a private temporary directory
with a random authkey, starts the children (the environment of ``tbenv.child_env("disable")``, the study directory as
working directory, the authkey and the worker's index in two environment variables that the worker removes from its
own environment), accepts the connections, sends each child its specs and receives the spaces.  A child that exits
before it connects, or does not connect within ``CONNECT_TIMEOUT_S``, fails the constructor.  The listener and its
directory are removed as soon as every child is connected (and by ``close`` after a failed start), so a trainer that
is killed later leaves nothing in the temporary directory.

What crosses the process boundary travels by pickle.  So that no result can depend on which envs share a process,
every argument reaches each env as its own copy at every W, 0 included: the kwargs of a spec, the options of
``set_options``, the value of ``set_attr`` and the arguments of ``env_method`` are pickled per env, and what
``get_attr`` and ``env_method`` return is a copy.  Kwargs or arguments that do not pickle (a closure, a lambda) are
refused at every W, 0 included, before any env sees them; a class they name must be importable in the worker.

Failure.  A worker that has exited, or whose connection broke, raises ``RuntimeError`` naming the worker and its exit
code from the next call, and from every call after it.  An exception raised by an env in a worker (also in a
factory) is raised here as the same exception, the worker's traceback attached as a note; the worker goes on.
``close`` asks the workers to close their envs, terminates what is left, kills what does not go, and waits for all of
them; the same clean-up runs at interpreter exit and when an unclosed pool is collected.  A worker leaves by itself
when its parent's connection closes, so a killed trainer leaves no worker behind.

Each worker checks at its start that ``MUJOCO_GL`` is ``disable`` and that each BLAS thread variable is ``1``, and
refuses to serve otherwise (the pool's constructor raises with the values found).

``step_async`` keeps the actions and ``step_wait`` sends them and reads the replies, as ``DummyVecEnv`` keeps them
and steps in ``step_wait``; the workers step side by side between the two.  Not mirrored: ``env_is_wrapped`` answers
``False`` for every env at W >= 1 (a worker cannot import a wrapper class of stable-baselines3 to compare with; the
envs of this study are not wrapped).

    python -m designlab.stage2.vecpool --worker ADDRESS      # a worker; started by PoolVecEnv, never by hand
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # before any numpy import: one core per process
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import operator  # noqa: E402
import pickle  # noqa: E402
import secrets  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
import weakref  # noqa: E402
from collections import OrderedDict  # noqa: E402
from multiprocessing.connection import Client, Listener  # noqa: E402

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

#: Each of these is "1" in a worker (``tbenv.THREAD_VARS``, ``s2io.THREAD_VARS``).
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
#: The child's environment variables that carry the connection's authkey (hex) and the worker's index.  The worker
#: removes both from its environment before it builds an env.
AUTHKEY_VAR = "STAGE2_VECPOOL_AUTHKEY"
INDEX_VAR = "STAGE2_VECPOOL_WORKER"
#: A child must have connected this long after the pool started it (interpreter start and imports on a loaded
#: machine).  Building the envs afterwards has no limit: the pool only watches that the child is alive.
CONNECT_TIMEOUT_S = 120.0
#: How often a waiting pool looks whether the worker it waits for is still alive.
POLL_S = 0.5
#: ``close``: how long the workers get to close their envs and leave before they are terminated, and how long a
#: terminated worker gets before it is killed.
CLOSE_WAIT_S = 5.0
TERM_WAIT_S = 5.0
#: How long the pool waits for the exit code of a worker whose connection broke.
EXIT_WAIT_S = 2.0
#: A Unix socket's path must fit ``sun_path`` (108 bytes on Linux).
SOCKET_PATH_MAX = 100
#: A worker's exit codes: served until told to close or until its parent went away; was not started by a pool; its
#: environment was refused; interrupted from the keyboard.
EXIT_OK, EXIT_NOT_A_POOL, EXIT_ENVIRONMENT, EXIT_INTERRUPTED = 0, 2, 3, 130

_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _dump(obj) -> bytes:
    return pickle.dumps(obj, protocol=_PROTOCOL)


def _load(raw):
    return pickle.loads(raw)


def _private(obj):
    """A copy of ``obj`` that shares nothing with it, made the way a value reaches a worker."""
    return _load(_dump(obj))


# ---------------------------------------------------------------------------------------------------------------
# Specs and factories (both sides).
# ---------------------------------------------------------------------------------------------------------------


def resolve(name: str):
    """The callable named ``"pkg.module:factory"`` (the module is imported; the attribute may be dotted)."""
    module, sep, attr = str(name).partition(":")
    if not sep or not module or not attr:
        raise ValueError(f"an env factory is named 'pkg.module:factory', not {name!r}")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{name} is not callable")
    return obj


def make_env(name: str, kwargs: dict):
    """``factory(**kwargs)`` of a spec; a ``gymnasium.Env`` or ``ValueError`` (``DummyVecEnv``'s ``_patch_env``)."""
    env = resolve(name)(**kwargs)
    if not isinstance(env, gymnasium.Env):
        raise ValueError(f"{name} returned a {type(env).__name__}, not a gymnasium.Env")
    return env


def pack_specs(specs) -> list[tuple[str, bytes]]:
    """``[(factory name, pickled kwargs)]``: each spec's kwargs pickled on their own, so that no two envs share an
    object.  ``ValueError`` for a malformed spec, ``TypeError`` for kwargs that do not pickle."""
    specs = list(specs)
    if not specs:
        raise ValueError("a pool needs at least one env spec")
    packed = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, (tuple, list)) or len(spec) != 2:
            raise ValueError(f"env {i}: a spec is ('pkg.module:factory', kwargs), not {spec!r}")
        name, kwargs = spec
        if not isinstance(name, str) or name.count(":") != 1 or not all(name.split(":")):
            raise ValueError(f"env {i}: a factory is named 'pkg.module:factory', not {name!r}")
        if not isinstance(kwargs, dict) or not all(isinstance(k, str) for k in kwargs):
            raise ValueError(f"env {i}: the kwargs of {name} must be a dict with string keys")
        try:
            blob = _dump(kwargs)
            _load(blob)
        except Exception as exc:
            raise TypeError(f"env {i}: the kwargs of {name} do not travel by pickle (plain values and importable "
                            f"dataclasses only, never a closure or a lambda): {type(exc).__name__}: {exc}") from exc
        packed.append((name, blob))
    return packed


def blocks(n_envs: int, workers: int) -> list[tuple[int, int]]:
    """``[(lo, hi)]``: worker k hosts envs ``lo .. hi - 1``, the blocks of ``numpy.array_split(range(n), W)``."""
    out = []
    for part in np.array_split(np.arange(int(n_envs)), int(workers)):
        out.append((int(part[0]), int(part[-1]) + 1))
    return out


def obs_space_info(obs_space):
    """``(keys, shapes, dtypes)`` of an observation space, as ``stable_baselines3.common.vec_env.util
    .obs_space_info``: a ``Dict`` by its keys, a ``Tuple`` by position, anything else under the key ``None``."""
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        inner = obs_space.spaces.values() if isinstance(obs_space, spaces.Dict) else obs_space.spaces
        for sub in inner:
            if isinstance(sub, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space).")
    if isinstance(obs_space, spaces.Dict):
        subspaces = obs_space.spaces
    elif isinstance(obs_space, spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}
    else:
        assert not hasattr(obs_space, "spaces"), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}
    keys, shapes, dtypes = [], {}, {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


def dict_to_obs(obs_space, obs_dict):
    """The observation in the space's own form, as ``stable_baselines3.common.vec_env.util.dict_to_obs``."""
    if isinstance(obs_space, spaces.Dict):
        return obs_dict
    if isinstance(obs_space, spaces.Tuple):
        assert len(obs_dict) == len(obs_space.spaces), "size of observation does not match size of observation space"
        return tuple(obs_dict[i] for i in range(len(obs_space.spaces)))
    assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
    return obs_dict[None]


# ---------------------------------------------------------------------------------------------------------------
# The worker.
# ---------------------------------------------------------------------------------------------------------------


class _Block:
    """One worker's envs under ``DummyVecEnv``'s loop.  The statements of ``reset`` and ``step`` are those of
    ``DummyVecEnv.reset`` and ``DummyVecEnv.step_wait``, in their order; what differs is only where the seeds and
    options come from (the pool sends them with ``reset``) and that the buffers are returned as they are (sending
    them is the copy)."""

    def __init__(self, envs: list):
        self.envs = envs
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError("a factory returned the same env instance for two envs of one pool")
        env = self.envs[0]
        self.num_envs = len(envs)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata
        self.keys, shapes, dtypes = obs_space_info(env.observation_space)
        self.buf_obs = OrderedDict(
            [(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.reset_infos = [{} for _ in range(self.num_envs)]

    def step(self, actions):
        """``(obs buffers, rewards, dones, infos, {env index in the block: the info of its reset})``."""
        fresh = {}
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                actions[env_idx])
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                fresh[env_idx] = self.reset_infos[env_idx]
            self._save_obs(env_idx, obs)
        return self.buf_obs, self.buf_rews, self.buf_dones, self.buf_infos, fresh

    def reset(self, seeds, options):
        """``(obs buffers, reset infos)``; env j of the block is reset with ``seeds[j]`` and ``options[j]``."""
        for env_idx in range(self.num_envs):
            maybe_options = {"options": options[env_idx]} if options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        return self.buf_obs, self.reset_infos

    def _save_obs(self, env_idx, obs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def get_attr(self, local, attr_name):
        return [self.envs[j].get_wrapper_attr(attr_name) for j in local]

    def set_attr(self, local, attr_name, value_blob) -> None:
        for j in local:
            setattr(self.envs[j], attr_name, _load(value_blob))

    def env_method(self, local, method_name, call_blob):
        out = []
        for j in local:
            method_args, method_kwargs = _load(call_blob)
            out.append(self.envs[j].get_wrapper_attr(method_name)(*method_args, **method_kwargs))
        return out

    def close(self) -> None:
        for env in self.envs:
            env.close()


def environment_problem(environ=None) -> str | None:
    """None if this process may serve envs: ``MUJOCO_GL`` is ``disable`` and each of ``THREAD_VARS`` is ``1``;
    otherwise the sentence the pool raises with."""
    environ = os.environ if environ is None else environ
    wanted = {"MUJOCO_GL": "disable", **{v: "1" for v in THREAD_VARS}}
    bad = {k: environ.get(k) for k, v in wanted.items() if environ.get(k) != v}
    if not bad:
        return None
    return ("a vecpool worker runs with MUJOCO_GL=disable and one thread per BLAS variable; this one has "
            + ", ".join(f"{k}={v!r}" for k, v in bad.items()))


def _pack_error(exc: BaseException) -> tuple:
    """``(the pickled exception or None, the traceback's text, "Type: message")`` of an exception raised here."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        blob = _dump(exc)
        _load(blob)
    except Exception:
        blob = None
    return blob, text, f"{type(exc).__name__}: {exc}"


def _serve(conn, index: int, problem: str | None) -> int:
    """A worker's life on its connection: hello, then one reply per command until ``close`` or until the parent's
    end closes.  A command's exception is the reply; the loop goes on."""
    conn.send_bytes(_dump(("hello", index, os.getpid(), problem)))
    if problem is not None:
        print(f"vecpool worker {index}: {problem}", file=sys.stderr, flush=True)
        return EXIT_ENVIRONMENT
    block = None
    try:
        while True:
            try:
                raw = conn.recv_bytes()
            except (EOFError, OSError):
                break  # the parent's connection closed: the parent closed the pool or is gone
            try:
                cmd, data = _load(raw)
                if cmd == "close":
                    break
                if cmd == "build":
                    if block is not None:
                        raise RuntimeError("this worker has its envs already")
                    block = _Block([make_env(name, _load(blob)) for name, blob in data])
                    result = (block.observation_space, block.action_space, block.metadata)
                elif block is None:
                    raise RuntimeError(f"{cmd!r} before the envs were built")
                elif cmd == "step":
                    result = block.step(data)
                elif cmd == "reset":
                    result = block.reset(*data)
                elif cmd == "get_attr":
                    result = block.get_attr(*data)
                elif cmd == "set_attr":
                    result = block.set_attr(*data)
                elif cmd == "env_method":
                    result = block.env_method(*data)
                else:
                    raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
                out = _dump((True, result))
            except Exception as exc:  # the env's, the factory's or the pickle's: the parent raises it
                out = _dump((False, _pack_error(exc)))
            try:
                conn.send_bytes(out)
            except OSError:
                break
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    finally:
        if block is not None:
            try:
                block.close()
            except Exception:
                traceback.print_exc()
    return EXIT_OK


def worker_main(address: str) -> int:
    """A worker: connect to the pool at ``address`` (a Unix socket), receive this worker's specs, build the envs and
    serve ``reset``, ``step``, ``get_attr``, ``set_attr``, ``env_method`` and ``close``.  Returns the process's exit
    code.  The caller is a process the pool started (``python -m designlab.stage2.vecpool --worker ADDRESS``, or the
    trainer's ``worker`` entry): the authkey and the worker's index are in its environment."""
    problem = environment_problem()
    key, index = os.environ.pop(AUTHKEY_VAR, None), os.environ.pop(INDEX_VAR, None)
    if key is None or index is None:
        print(f"vecpool worker: {AUTHKEY_VAR} and {INDEX_VAR} are not set: a worker is started by PoolVecEnv, "
              "not by hand", file=sys.stderr, flush=True)
        return EXIT_NOT_A_POOL
    conn = Client(address, family="AF_UNIX", authkey=bytes.fromhex(key))
    try:
        return _serve(conn, int(index), problem)
    except OSError:
        return EXIT_OK  # the parent went away while this worker was writing to it
    finally:
        conn.close()


# ---------------------------------------------------------------------------------------------------------------
# The pool (the parent's side).
# ---------------------------------------------------------------------------------------------------------------


def study_dir() -> str:
    """The study directory (``hardware/trex_form_study``), from this file's location (``tbenv.study_dir``)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def child_env() -> dict:
    """A worker's environment: ``tbenv.child_env("disable")``.  Until ``tbenv`` is on the branch the same
    environment is built here: a copy of ``os.environ`` with ``MUJOCO_GL=disable``, no ``PYOPENGL_PLATFORM``, the
    thread variables at ``1``, ``PYTHONHASHSEED=0`` and ``PYTHONPATH=.`` (the child's working directory is the study
    directory)."""
    try:
        from . import tbenv
    except ImportError:
        env = {str(k): str(v) for k, v in os.environ.items()}
        env["MUJOCO_GL"] = "disable"
        env.pop("PYOPENGL_PLATFORM", None)
        for name in THREAD_VARS:
            env[name] = "1"
        env.update(PYTHONHASHSEED="0", PYTHONPATH=".")
        return env
    return tbenv.child_env("disable")


def default_worker_cmd() -> list[str]:
    return [sys.executable, "-m", "designlab.stage2.vecpool", "--worker"]


class _Worker:
    """One child: its index, its envs ``lo .. hi - 1``, its process, its connection and the pid it reported."""

    __slots__ = ("index", "lo", "hi", "proc", "conn", "pid")

    def __init__(self, index, lo, hi, proc):
        self.index, self.lo, self.hi, self.proc, self.conn, self.pid = index, lo, hi, proc, None, proc.pid

    def __str__(self) -> str:
        envs = f"env {self.lo}" if self.hi - self.lo == 1 else f"envs {self.lo} to {self.hi - 1}"
        return f"vecpool worker {self.index} (pid {self.pid}, {envs})"


def _exit_text(code) -> str:
    if code is None:
        return "still running, its connection closed"
    if code < 0:
        try:
            return f"exit code {code} ({signal.Signals(-code).name})"
        except ValueError:
            pass
    return f"exit code {code}"


def _shutdown(owner_pid: int, workers: list, sockdir: str) -> None:
    """Close the connections, terminate the children, kill what does not go within ``TERM_WAIT_S``, wait for all of
    them and remove the socket directory.  Runs once per pool: from ``close``, when an unclosed pool is collected, or
    at interpreter exit; never in a forked copy of the owner."""
    if os.getpid() != owner_pid:
        return
    for w in workers:
        if w.conn is not None:
            try:
                w.conn.close()
            except OSError:
                pass
    for w in workers:
        if w.proc.poll() is None:
            try:
                w.proc.terminate()
            except OSError:
                pass
    deadline = time.monotonic() + TERM_WAIT_S
    for w in workers:
        try:
            w.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            w.proc.kill()
            w.proc.wait()
    shutil.rmtree(sockdir, ignore_errors=True)


def listening_socket(listener):
    """The socket a ``multiprocessing.connection.Listener`` accepts on, or None if this Python keeps it elsewhere.
    ``Listener`` has no timeout of its own and no public way to its socket; the two attribute names are CPython's
    since 3.3 (a test holds them for the running version).  Without the socket the pool still starts, but a child
    that never connects would then block the constructor."""
    return getattr(getattr(listener, "_listener", None), "_socket", None)


def _socket_dir() -> str:
    """A new private directory (mode 0700) whose socket path fits a Unix socket's address."""
    roots = [tempfile.gettempdir(), "/tmp"]
    for root in roots:
        if len(os.fsencode(os.path.join(root, "vecpool-12345678", "s"))) <= SOCKET_PATH_MAX and os.path.isdir(root):
            return tempfile.mkdtemp(prefix="vecpool-", dir=root)
    raise RuntimeError(f"no temporary directory short enough for a Unix socket among {roots}")


def _remote_error(worker: _Worker, payload) -> BaseException:
    """The exception a worker replied with: the same exception where it unpickles, a ``RuntimeError`` with its text
    otherwise; the worker's traceback is attached as a note."""
    blob, text, summary = payload
    exc = None
    if blob is not None:
        try:
            exc = _load(blob)
        except Exception:
            exc = None
    if not isinstance(exc, Exception):
        exc = RuntimeError(summary)
    note = f"raised in {worker}:\n{text}"
    if hasattr(exc, "add_note"):
        exc.add_note(note)
    return exc


class _Pool:
    """The body of ``PoolVecEnv`` (module docstring).  ``PoolVecEnv`` is this class on stable-baselines3's
    ``VecEnv``; the two are joined on first access so that importing this module imports no stable-baselines3."""

    actions: np.ndarray

    def __init__(self, specs: list[tuple[str, dict]], workers: int, worker_cmd: list[str] | None = None):
        packed = pack_specs(specs)
        n = len(packed)
        workers = operator.index(workers)
        if not 0 <= workers <= n:
            raise ValueError(f"workers must be 0 (this process) or 1 to the number of envs ({n}), not {workers}")
        self.workers = workers
        self.blocks = blocks(n, workers) if workers else []
        self.worker_cmd = default_worker_cmd() if worker_cmd is None else [str(c) for c in worker_cmd]
        if not self.worker_cmd:
            raise ValueError("worker_cmd is empty")
        self.socket_dir = None
        self._dummy = None
        self._workers: list[_Worker] = []
        self._closed = False
        self._broken: str | None = None
        self._in_flight = False
        if workers == 0:
            from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

            self._dummy = DummyVecEnv([functools.partial(make_env, name, _load(blob)) for name, blob in packed])
            super().__init__(n, self._dummy.observation_space, self._dummy.action_space)
            self.reset_infos = self._dummy.reset_infos  # DummyVecEnv writes into this list, never replaces it
            self.metadata = self._dummy.metadata
            return
        self.socket_dir = _socket_dir()
        self._finalizer = weakref.finalize(self, _shutdown, os.getpid(), self._workers, self.socket_dir)
        try:
            observation_space, action_space, metadata = self._start(packed)
            self._keys = obs_space_info(observation_space)[0]
            super().__init__(n, observation_space, action_space)
            self.metadata = metadata
        except BaseException:
            self._closed = True
            self._finalizer()
            raise

    # -- start-up ------------------------------------------------------------------------------------------------

    def _start(self, packed):
        """Start the children, take their connections, send each its specs; ``(observation space, action space,
        metadata)`` of env 0.  The listener and its directory are gone when this returns or raises."""
        address = os.path.join(self.socket_dir, "s")
        authkey = secrets.token_bytes(32)
        listener = Listener(address, family="AF_UNIX", backlog=len(self.blocks), authkey=authkey)
        try:
            base = child_env()
            for k, (lo, hi) in enumerate(self.blocks):
                env = dict(base)
                env[AUTHKEY_VAR], env[INDEX_VAR] = authkey.hex(), str(k)
                proc = subprocess.Popen(self.worker_cmd + [address], env=env, cwd=study_dir(),
                                        stdin=subprocess.DEVNULL, close_fds=True)
                self._workers.append(_Worker(k, lo, hi, proc))
            self._accept(listener)
        finally:
            listener.close()
            shutil.rmtree(self.socket_dir, ignore_errors=True)
        replies = self._exchange([(w, ("build", packed[w.lo:w.hi])) for w in self._workers])
        observation_space, action_space, metadata = replies[0]
        for w, (obs_k, act_k, _) in zip(self._workers, replies):
            if obs_k != observation_space or act_k != action_space:
                raise ValueError(f"the envs of {w} have other spaces than env 0: {obs_k}, {act_k} against "
                                 f"{observation_space}, {action_space}")
        return observation_space, action_space, metadata

    def _accept(self, listener) -> None:
        """Take one connection per child and its hello.  The wait is cut into ``POLL_S`` slices by a timeout on the
        listening socket, so that a child that exits before it connects, or never connects, raises here instead of
        blocking."""
        sock = listening_socket(listener)
        if sock is not None:
            sock.settimeout(POLL_S)
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        pending = {w.index: w for w in self._workers}
        while pending:
            for w in pending.values():
                code = w.proc.poll()
                if code is not None:
                    raise RuntimeError(f"{w} left before it connected: {_exit_text(code)}; its command line: "
                                       f"{' '.join(self.worker_cmd)} <address>")
            if time.monotonic() > deadline:
                raise RuntimeError(f"vecpool workers {sorted(pending)} did not connect within "
                                   f"{CONNECT_TIMEOUT_S:g} s; their command line: {' '.join(self.worker_cmd)} "
                                   "<address>")
            try:
                conn = listener.accept()
            except TimeoutError:
                continue
            try:
                if not conn.poll(CONNECT_TIMEOUT_S):
                    raise EOFError
                _, index, pid, problem = _load(conn.recv_bytes())
            except (EOFError, OSError):
                conn.close()  # a child that died between its connection and its hello: found dead above
                continue
            w = pending.pop(int(index))
            w.conn, w.pid = conn, int(pid)
            if problem is not None:
                raise RuntimeError(f"{w}: {problem}")

    # -- the exchange --------------------------------------------------------------------------------------------

    def _usable(self) -> None:
        if self._closed:
            raise RuntimeError("this PoolVecEnv is closed")
        if self._broken is None and self._in_flight:
            self._broken = ("an earlier call to this PoolVecEnv was interrupted between its commands and their "
                            "replies; the pool cannot continue (close it and build a new one)")
        if self._broken is not None:
            raise RuntimeError(self._broken)

    def _lost(self, w: _Worker) -> RuntimeError:
        try:
            code = w.proc.wait(timeout=EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            code = None
        self._broken = (f"{w} is gone: {_exit_text(code)}; the pool cannot continue (close it and build a new one)")
        return RuntimeError(self._broken)

    def _read(self, w: _Worker) -> bytes:
        try:
            while not w.conn.poll(POLL_S):
                if w.proc.poll() is not None and not w.conn.poll(0):
                    raise EOFError
            return w.conn.recv_bytes()
        except (EOFError, OSError):
            raise self._lost(w) from None

    def _exchange(self, calls: list) -> list:
        """Send each ``(worker, command)`` of ``calls``, then read every reply: the workers run side by side.  The
        replies' payloads in the order of ``calls``; a worker's exception is raised after all replies are read (the
        first in worker order), a lost worker at once."""
        self._usable()
        blobs = [_dump(command) for _, command in calls]  # a command that does not pickle touches no worker
        self._in_flight = True
        for (w, _), blob in zip(calls, blobs):
            try:
                w.conn.send_bytes(blob)
            except OSError:
                raise self._lost(w) from None
        raws = [self._read(w) for w, _ in calls]
        self._in_flight = False
        out, error = [], None
        for (w, _), raw in zip(calls, raws):
            ok, payload = _load(raw)
            if ok:
                out.append(payload)
            elif error is None:
                error = _remote_error(w, payload)
        if error is not None:
            raise error
        return out

    def _indices(self, indices) -> list[int]:
        """``VecEnv._get_indices`` as a list of env indices in ``0 .. N - 1`` (negative ones count from the end, as
        ``DummyVecEnv``'s list of envs reads them)."""
        out = []
        for i in self._get_indices(indices):
            i = operator.index(i)
            if not -self.num_envs <= i < self.num_envs:
                raise IndexError(f"env index {i} out of range for {self.num_envs} envs")
            out.append(i % self.num_envs)
        return out

    def _per_env(self, cmd: str, indices, *rest) -> list:
        """One command for the envs ``indices``, each worker getting its own; the results in the order of
        ``indices``."""
        idx = self._indices(indices)
        by_worker: dict[int, list[tuple[int, int]]] = {}
        for pos, i in enumerate(idx):
            k = next(w.index for w in self._workers if w.lo <= i < w.hi)
            by_worker.setdefault(k, []).append((pos, i - self._workers[k].lo))
        order = sorted(by_worker)
        replies = self._exchange([(self._workers[k], (cmd, ([j for _, j in by_worker[k]], *rest))) for k in order])
        out = [None] * len(idx)
        for k, reply in zip(order, replies):
            if reply is not None:
                for (pos, _), value in zip(by_worker[k], reply):
                    out[pos] = value
        return out

    def _join_obs(self, parts: list):
        joined = OrderedDict([(key, np.concatenate([p[key] for p in parts], axis=0)) for key in self._keys])
        return dict_to_obs(self.observation_space, joined)

    # -- VecEnv --------------------------------------------------------------------------------------------------

    def reset(self):
        if self._dummy is not None:
            return self._dummy.reset()
        parts = self._exchange([(w, ("reset", (self._seeds[w.lo:w.hi], self._options[w.lo:w.hi])))
                                for w in self._workers])
        for w, (_, infos) in zip(self._workers, parts):
            for j, info in enumerate(infos):
                self.reset_infos[w.lo + j] = info
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._join_obs([obs for obs, _ in parts])

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        if self._dummy is not None:
            self._dummy.step_async(actions)

    def step_wait(self):
        if self._dummy is not None:
            return self._dummy.step_wait()
        parts = self._exchange([(w, ("step", self.actions[w.lo:w.hi])) for w in self._workers])
        for w, part in zip(self._workers, parts):
            for j, info in part[4].items():
                self.reset_infos[w.lo + j] = info
        return (self._join_obs([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]), [info for p in parts for info in p[3]])

    def seed(self, seed: int | None = None):
        if self._dummy is not None:
            return self._dummy.seed(seed)
        return super().seed(seed)

    def set_options(self, options: list[dict] | dict | None = None) -> None:
        if isinstance(options, dict):
            options = [_private(options) for _ in range(self.num_envs)]
        elif options is not None:
            options = [_private(o) for o in options]
        if self._dummy is not None:
            self._dummy.set_options(options)
        else:
            super().set_options(options)

    def get_attr(self, attr_name: str, indices=None) -> list:
        if self._dummy is not None:
            return [_private(v) for v in self._dummy.get_attr(attr_name, indices)]
        return self._per_env("get_attr", indices, attr_name)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        blob = _dump(value)
        if self._dummy is not None:
            for i in self._indices(indices):
                self._dummy.set_attr(attr_name, _load(blob), indices=[i])
            return
        self._per_env("set_attr", indices, attr_name, blob)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        blob = _dump((method_args, method_kwargs))
        if self._dummy is not None:
            out = []
            for i in self._indices(indices):
                args, kwargs = _load(blob)
                out.extend(self._dummy.env_method(method_name, *args, indices=[i], **kwargs))
            return [_private(v) for v in out]
        return self._per_env("env_method", indices, method_name, blob)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        if self._dummy is not None:
            return self._dummy.env_is_wrapped(wrapper_class, indices)
        return [False for _ in self._indices(indices)]

    def get_images(self):
        if self._dummy is not None:
            return self._dummy.get_images()
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images.")
            return [None for _ in range(self.num_envs)]
        return self.env_method("render")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._dummy is not None:
            self._dummy.close()
            return
        if self._broken is None and not self._in_flight:
            blob = _dump(("close", None))
            for w in self._workers:
                try:
                    w.conn.send_bytes(blob)
                except OSError:
                    pass
            deadline = time.monotonic() + CLOSE_WAIT_S
            for w in self._workers:
                try:
                    w.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
        self._finalizer()

    @property
    def worker_pids(self) -> list[int]:
        """The workers' process ids, worker 0 first (empty with ``workers=0``)."""
        return [w.pid for w in self._workers]


def __getattr__(name: str):
    """``PoolVecEnv`` is made on first access: ``_Pool`` on stable-baselines3's ``VecEnv``.  A worker imports this
    module and never asks for it, so it never imports stable-baselines3 or torch."""
    if name == "PoolVecEnv":
        from stable_baselines3.common.vec_env.base_vec_env import VecEnv

        class PoolVecEnv(_Pool, VecEnv):
            __doc__ = ("``PoolVecEnv(specs, workers, worker_cmd=None)``: a stable-baselines3 ``VecEnv`` of "
                       "``len(specs)`` envs in ``workers`` child processes (0: ``DummyVecEnv`` in this process), "
                       "equal to ``DummyVecEnv`` at every worker count.  See the module's docstring.")

        PoolVecEnv.__module__, PoolVecEnv.__qualname__ = __name__, "PoolVecEnv"
        globals()["PoolVecEnv"] = PoolVecEnv
        return PoolVecEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.vecpool",
                                 description="A worker process of PoolVecEnv (started by the pool, never by hand).")
    ap.add_argument("--worker", metavar="ADDRESS", required=True,
                    help="serve this pool's envs: the Unix socket the pool listens on")
    args = ap.parse_args(argv)
    return worker_main(args.worker)


__all__ = ["AUTHKEY_VAR", "INDEX_VAR", "PoolVecEnv", "THREAD_VARS", "blocks", "child_env", "default_worker_cmd",
           "environment_problem", "listening_socket", "main", "make_env", "pack_specs", "resolve", "study_dir",
           "worker_main"]

if __name__ == "__main__":
    from designlab.stage2 import vecpool as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
