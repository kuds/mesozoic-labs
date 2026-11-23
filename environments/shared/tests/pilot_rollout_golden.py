"""Golden rollouts of the pilot behavior recipes (Phase D, BEHAVIOR_RECIPES_PLAN §4.6).

Phase D moved the command-tracking reward, the tracking metrics and info
keys and the tracking dwell out of ``behavior_env.SpeciesBehaviorMixin`` into
``BaseDinoEnv``, where a manifest stage with a live ``[commands]`` table runs
the same code.  The move must change nothing a pilot recipe does.  This
module makes that claim executable: it rolls each pilot recipe out the way
``train_behaviors.main`` builds its env, and the fixture it writes
(``fixtures/phase_d_pilot_rollout_golden.json``), captured on the tree
before the move, is replayed by ``test_phase_d_command_plumbing.py``.

Per pilot, two episodes: ``reset(seed=RESET_SEED)``, then an unseeded
``reset()`` on the same env, each stepped with a fixed uniform action
sequence until it terminates, truncates or reaches the spec's step count.
The fixture records, per episode:

* ``reset_sha256``: the observation, the reset ``info``, the command, the
  exact ``qpos`` and ``qvel``, the generator state and the push schedule;
* ``rewards``: every step's reward, exactly (JSON floats round-trip);
* ``step_sha256``: per step, the first 16 hex digits of a digest of the
  reward, both flags, the whole step ``info``, the observation and the
  command;
* ``episode_sha256``: one digest of every full step digest;
* ``portable``: the first :data:`PORTABLE_STEPS` steps, one row each in the
  order of :data:`PORTABLE_COLUMNS`: the reward, both flags, a digest of the
  info's key order, and the plain values of the info keys the move produced
  (:data:`PORTABLE_INFO_KEYS`: the tracking reward and total, the dwell, the
  command state and the tracking metrics, and the course keys around them).

Every digest hashes each value with its type (a ``numpy.float64`` and a
Python ``float`` of the same value differ, and arrays carry dtype and shape),
so a move that changes a type without changing a value shows.  The pilot's
``task_sha256`` is recorded beside the rollout.

The rollout crosses the MuJoCo solver and numpy's CPU-dispatched kernels, so
the digests are exact only in the numerics environment they were taken in
(:func:`numerics_environment`: Python, the C library, the machine, MuJoCo,
numpy and numpy's enabled CPU features).  The replay compares exactly where
that environment matches the fixture's and skips, naming the difference,
elsewhere, as ``digest_snapshot --exact`` is never a golden across machines.
The ``portable`` rows are the companion that runs on other machines too.
The replay compares their floats within a relative 1e-9 and the key order,
the flags, integers and strings exactly, wherever the MuJoCo release and
numpy's major version are the fixture's (:func:`portable_environment`),
which the repository's MuJoCo pin makes every CI job.  Python, the C library,
the CPU and numpy 2.x change only the last bits of a step: the digest
snapshot's 6-decimal reward captures, hundreds of steps long, held across
all four, and numpy 1.x, which promotes float32 differently, moved them.

Capture (the tree must be the one the fixture describes)::

    PYTHONPATH=. python environments/shared/tests/pilot_rollout_golden.py \\
        --out environments/shared/tests/fixtures/phase_d_pilot_rollout_golden.json

The fixture describes the source commit, before the move.  Schema v2 (the
``portable`` rows) was captured on an export of that commit with this module
copied in (``git archive 769a657``, then the command above with
``--source-commit 769a657`` and ``PYTHONPATH`` the export); every v1 field
came back unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_SCHEMA = "mesozoic.phase-d-pilot-rollout-golden/v2"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "phase_d_pilot_rollout_golden.json"

RESET_SEED = 3042
ACTION_SEED = 20261123
EPISODES = 2
STEP_DIGEST_HEX = 16
#: Steps per episode in the ``portable`` rows.
PORTABLE_STEPS = 5
#: The step info keys whose values the ``portable`` rows keep: ``reward_total``, the tracking term and the dwell,
#: the command state's ``as_info()`` keys, the tracking metrics, and the behavior env's course keys.
PORTABLE_INFO_KEYS: tuple[str, ...] = (
    "reward_total",
    "reward_tracking",
    "tracking_dwell_s",
    "desired_heading",
    "desired_speed",
    "actual_heading",
    "heading_error_rad",
    "heading_active",
    "command_v_x",
    "command_v_y",
    "command_yaw_rate",
    "command_v_x_normalized",
    "command_v_y_normalized",
    "command_yaw_rate_normalized",
    "command_time_s",
    "command_event_id",
    "command_event_start_s",
    "command_target_source",
    "actual_v_x",
    "actual_v_y",
    "actual_speed",
    "actual_yaw_rate",
    "tracking_error_v",
    "tracking_error_yaw",
    "tracking_error_heading",
    "tracking_heading_active",
    "tracking_error_requested_speed",
    "tracking_command_in_tolerance",
    "tracking_in_tolerance",
    "course_progress_m",
    "course_reached",
)
#: The columns of one ``portable`` row.
PORTABLE_COLUMNS: tuple[str, ...] = ("reward", "terminated", "truncated", "info_keys_sha256", *PORTABLE_INFO_KEYS)
#: The tolerance a ``portable`` float is compared with (``math.isclose``).  A one-ulp nudge of the reset ``qpos``
#: or ``qvel`` moved these rows by at most 7e-12 relative and 5e-14 absolute; a 1% change of the tracking velocity
#: sigma moved eight of the nine pilots out of it.
PORTABLE_REL_TOL, PORTABLE_ABS_TOL = 1e-9, 1e-12

#: (species, embodiment, recipe, run_seed, action amplitude, steps per episode).  The seven
#: ``follow_direction`` pilots (three of them terminate early, so the fall path is covered), one terrain
#: recipe whose two episodes visit two terrain families, and a long ``follow_direction_speed`` rollout:
#: at run_seed 4250 its first episode switches at step 691 into a stop that holds to the end, and its
#: second starts on a stop at reset and switches to a moving event at step 587, so the tracking dwell's
#: event restart and the stop branch of the tracking metrics are both exercised.
PILOT_SPECS: tuple[tuple[str, str, str, int, float, int], ...] = (
    ("velociraptor_mongoliensis", "anatomical", "follow_direction", 4242, 0.3, 200),
    ("tyrannosaurus_rex", "anatomical", "follow_direction", 4242, 0.3, 200),
    ("brachiosaurus_altithorax", "anatomical", "follow_direction", 4242, 0.3, 200),
    ("dibothrosuchus_elaphros", "anatomical", "follow_direction", 4242, 0.3, 200),
    ("compsognathus_longipes", "anatomical", "follow_direction", 4242, 0.3, 200),
    ("compsognathus_longipes", "robot", "follow_direction", 4242, 0.3, 200),
    ("allosaurus_fragilis", "anatomical", "follow_direction", 4242, 0.3, 200),
    ("velociraptor_mongoliensis", "anatomical", "follow_direction_difficult_terrain", 4242, 0.3, 200),
    ("velociraptor_mongoliensis", "anatomical", "follow_direction_speed", 4250, 0.1, 1000),
)


def pilot_key(species: str, embodiment: str, recipe: str) -> str:
    """The fixture's name for one pilot capture."""
    return f"{species}/{embodiment}/{recipe}"


def numerics_environment() -> dict[str, Any]:
    """What the exactness of a rollout digest depends on, besides the repository tree."""
    import importlib

    import mujoco

    try:
        umath = importlib.import_module("numpy._core._multiarray_umath")
    except ImportError:  # pragma: no cover - numpy < 2 keeps the table under numpy.core
        umath = importlib.import_module("numpy.core._multiarray_umath")
    features: dict[str, bool] = getattr(umath, "__cpu_features__", {})
    return {
        "python": f"{platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}",
        "libc": " ".join(platform.libc_ver()),
        "machine": platform.machine(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "numpy_cpu_features": sorted(name for name, enabled in features.items() if enabled),
    }


def portable_environment(environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the ``portable`` rows depend on: the MuJoCo release and numpy's major version.

    *environment* is a :func:`numerics_environment` record (a fixture's); by default the running one.
    """
    recorded = numerics_environment() if environment is None else environment
    return {"mujoco": recorded["mujoco"], "numpy_major": str(recorded["numpy"]).split(".")[0]}


def plain(value: Any) -> Any:
    """*value* as the JSON scalar a ``portable`` row stores: a bool, an int, a float or a str."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"no portable form for {type(value).__name__}")


def portable_row(reward: Any, terminated: Any, truncated: Any, info: dict[str, Any]) -> list[Any]:
    """One ``portable`` row, in the order of :data:`PORTABLE_COLUMNS`."""
    keys = hashlib.sha256(json.dumps(list(info)).encode("utf-8")).hexdigest()[:STEP_DIGEST_HEX]
    return [plain(reward), plain(terminated), plain(truncated), keys, *(plain(info[key]) for key in PORTABLE_INFO_KEYS)]


def portable_differences(expected: list[list[Any]], actual: list[list[Any]]) -> list[str]:
    """Each (step, column) where *actual* leaves the tolerance, or the exact value, of *expected*."""
    if len(actual) != len(expected):
        return [f"{len(actual)} portable rows, expected {len(expected)}"]
    differences = []
    for step, (want_row, got_row) in enumerate(zip(expected, actual)):
        for column, want, got in zip(PORTABLE_COLUMNS, want_row, got_row, strict=True):
            if type(got) is not type(want):
                same = False
            elif isinstance(want, float):
                same = math.isclose(got, want, rel_tol=PORTABLE_REL_TOL, abs_tol=PORTABLE_ABS_TOL)
            else:
                same = got == want
            if not same:
                differences.append(f"step {step} {column}: {got!r} != {want!r}")
    return differences


def typed(value: Any) -> Any:
    """A JSON form of *value* that keeps its type: numpy scalars, Python scalars and arrays all differ."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ["ndarray", array.dtype.str, list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]
    if isinstance(value, np.generic):
        name = f"numpy.{type(value).__name__}"
        if isinstance(value, np.floating):
            return [name, float(value).hex()]
        if isinstance(value, np.bool_):
            return [name, bool(value)]
        if isinstance(value, np.integer):
            return [name, str(int(value))]
        raise TypeError(f"no typed form for {name}")
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if value is None:
        return ["none"]
    if isinstance(value, dict):
        return ["dict", [[str(key), typed(value[key])] for key in sorted(value, key=str)]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [typed(item) for item in value]]
    raise TypeError(f"no typed form for {type(value).__name__}")


def digest(value: Any) -> str:
    """The sha256 of :func:`typed` *value*."""
    payload = json.dumps(typed(value), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pilot_env(species: str, embodiment: str, recipe: str, run_seed: int) -> Any:
    """The env ``train_behaviors.main`` builds, unwrapped: ``read_recipe`` then ``create_behavior_env``.

    Training wraps it in ``Monitor(EpisodeManifestRecorder(...))``; neither wrapper changes a reward, a
    flag or an observation, and ``Monitor`` adds a wall-clock ``episode`` entry to the last step's info.
    """
    from environments.shared.model_id import ModelId
    from environments.shared.paths import REPOSITORY_ROOT
    from environments.shared.train_behaviors import create_behavior_env, read_recipe

    model_id = ModelId(species, embodiment)
    path = model_id.path(REPOSITORY_ROOT / "configs") / "behaviors" / f"{recipe}.toml"
    _, commands, terrain, env_kwargs = read_recipe(path, model_id)
    return create_behavior_env(model_id, commands=commands, terrain=terrain, run_seed=run_seed, **env_kwargs)


def _reset_state(env: Any, observation: Any, info: dict[str, Any]) -> dict[str, Any]:
    starts = getattr(env, "_push_schedule_starts", None)
    directions = getattr(env, "_push_schedule_directions", None)
    return {
        "observation": np.array(observation),
        "info": info,
        "command": np.array(env._command),
        "qpos": np.array(env.data.qpos, dtype=np.float64),
        "qvel": np.array(env.data.qvel, dtype=np.float64),
        "rng_state": env.np_random.bit_generator.state,
        "push_schedule_starts": None if starts is None else np.array(starts),
        "push_schedule_directions": None if directions is None else np.array(directions),
    }


def capture_pilot(species: str, embodiment: str, recipe: str, run_seed: int, amplitude: float, steps: int) -> dict:
    """One pilot's record: its task digest and both episodes."""
    env = build_pilot_env(species, embodiment, recipe, run_seed)
    try:
        nu = int(np.prod(env.action_space.shape))
        actions = np.random.default_rng(ACTION_SEED).uniform(-amplitude, amplitude, size=(EPISODES, steps, nu))
        actions = actions.astype(np.float32)
        episodes = []
        for episode in range(EPISODES):
            observation, info = env.reset(seed=RESET_SEED if episode == 0 else None)
            reset_sha256 = digest(_reset_state(env, observation, info))
            rewards: list[float] = []
            step_digests: list[str] = []
            portable: list[list[Any]] = []
            ended_by = "steps"
            for t in range(steps):
                observation, reward, terminated, truncated, info = env.step(actions[episode, t])
                rewards.append(float(reward))
                step_digests.append(
                    digest(
                        {
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "info": info,
                            "observation": np.array(observation),
                            "command": np.array(env._command),
                        }
                    )
                )
                if t < PORTABLE_STEPS:
                    portable.append(portable_row(reward, terminated, truncated, info))
                if terminated or truncated:
                    ended_by = "terminated" if terminated else "truncated"
                    break
            episodes.append(
                {
                    "seed": RESET_SEED if episode == 0 else None,
                    "length": len(rewards),
                    "ended_by": ended_by,
                    "reset_sha256": reset_sha256,
                    "rewards": rewards,
                    "step_sha256": [value[:STEP_DIGEST_HEX] for value in step_digests],
                    "episode_sha256": hashlib.sha256("".join(step_digests).encode("ascii")).hexdigest(),
                    "portable": portable,
                }
            )
        return {
            "species": species,
            "embodiment": embodiment,
            "recipe": recipe,
            "run_seed": run_seed,
            "action_amplitude": amplitude,
            "steps": steps,
            "task_sha256": env.task_fingerprint["task_sha256"],
            "episodes": episodes,
        }
    finally:
        env.close()


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_fixture(*, source_commit: str | None = None) -> dict[str, Any]:
    """The complete fixture document."""
    return {
        "schema": FIXTURE_SCHEMA,
        "source_commit": _source_commit() if source_commit is None else source_commit,
        "numerics_environment": numerics_environment(),
        "reset_seed": RESET_SEED,
        "action_seed": ACTION_SEED,
        "portable_columns": list(PORTABLE_COLUMNS),
        "captures": {pilot_key(*spec[:3]): capture_pilot(*spec) for spec in PILOT_SPECS},
    }


def dumps(value: Any, indent: int = 0) -> str:
    """JSON with one line per list of scalars, so a capture's rewards and step digests stay one line each."""
    pad = " " * (indent + 1)
    if isinstance(value, dict) and value:
        items = [f"{pad}{json.dumps(key)}: {dumps(item, indent + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        items = [f"{pad}{dumps(item, indent + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(value, allow_nan=False)


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    with path.open() as stream:
        fixture: dict[str, Any] = json.load(stream)
    return fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)
    fixture = build_fixture(source_commit=args.source_commit)
    args.out.write_text(dumps(fixture) + "\n")
    print(f"wrote {args.out} ({len(fixture['captures'])} pilots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
