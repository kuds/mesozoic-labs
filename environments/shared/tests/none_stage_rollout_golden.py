"""Golden rollouts of every command-free stage: each stance and recovery (BEHAVIOR_RECIPES_PLAN §4.6).

The live walk (decision D-D62) turns every model's locomotion stage into a
live-command stage and adds code that runs while the command is exactly zero
(a stand block of stance-like reward terms, idle penalties gated off), a
forward reference along the commanded heading, an Allosaurus course along it
and new reward kwargs.  Under ``command_mode = "none"`` the command is exactly
zero too, so each of those changes has to stay out of a command-free stage by
construction, and none of them may move a stance or a recovery.  This module
makes that claim executable: it rolls each model's stance, and its recovery
where the stage manifest declares one, out of the committed stage config, and
the fixture it writes (``fixtures/none_stage_rollout_golden.json``), captured
on the tree of decision D-D61 before the live walk changed anything, is
replayed by ``test_none_stage_rollout_golden.py``.

Per stage (:data:`CAPTURE_SPECS`), per seed in :data:`SEEDS` and per action
kind in :data:`ACTION_KINDS` (``zero``: the zero action; ``random``: a fixed
uniform sequence in +-:data:`ACTION_AMPLITUDE`, drawn as float32 from
``default_rng([ACTION_SEED, seed])``), one rollout of :data:`STEPS` steps on
an env built fresh by ``config.build_env`` (the committed ``[env]`` kwargs and
``[commands]`` table, as ``train_base.make_env`` passes them): a seeded
``reset(seed=seed)``, then every step, with an unseeded ``reset()`` after each
episode that terminates or truncates before the last step, as a training
worker's vector env resets.  Each step's record is the reward, both flags, the
step ``info`` (its key order and every value with its type) and the sha256 of
the observation's float64 bytes with its dtype and shape; each reset's record
is the observation the same way, the reset ``info``, the exact ``qpos`` and
``qvel``, the generator state and the push schedule.  The fixture stores, per
rollout:

* ``rollout_sha256``: one digest of every reset and step record, in order;
* ``component_sha256``: one digest per component (:data:`COMPONENTS`) of every
  record in order, so a mismatch names what moved (a reward term, a flag, an
  info value, the observation or a reset);
* ``block_sha256``: per :data:`BLOCK_STEPS` steps, the first
  :data:`~environments.shared.tests.pilot_rollout_golden.STEP_DIGEST_HEX` hex
  digits of a digest of that block's records (a reset belongs to the block of
  the step after it), so a mismatch names where;
* ``episodes`` and ``reward_sum``: each episode's first step, length and how
  it ended (``terminated``, ``truncated``, or ``running`` at the last step),
  and the ``math.fsum`` of the rewards, readable;
* ``portable``: the first :data:`PORTABLE_STEPS` steps, one row each: the
  reward, both flags, the first 16 hex digits of a digest of the info's key
  order, and every info value in that order (the capture's ``info_keys``, the
  first step's, name the columns).

The per-step records are not stored: 60 rollouts of 300 steps with 25 to 99
info keys each would be megabytes, and the digests pin every value exactly.
Every digest hashes each value with its type (``pilot_rollout_golden.typed``),
so a change of type without a change of value shows.  The stage's
``task_sha256`` and ``command_mode`` are recorded beside its rollouts.

Exactness is the pilot golden's (``pilot_rollout_golden.py``): the digests
hold only in the numerics environment they were taken in
(``numerics_environment``), and the replay skips the exact comparison,
naming the difference, elsewhere; the ``portable`` rows are compared within
the pilot golden's tolerance wherever the MuJoCo release and numpy's major
version are the fixture's (``portable_environment``).

Check the current tree (the default; exits 1 and names each difference) and
capture::

    PYTHONPATH=. python environments/shared/tests/none_stage_rollout_golden.py
    PYTHONPATH=. python environments/shared/tests/none_stage_rollout_golden.py --write

A capture belongs only to a change that moves a command-free stage on
purpose.  The fixture records the commit it was taken on and every tracked
file that differed from that commit when it was taken (none, for a capture of
a committed tree).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from environments.shared.model_id import ModelId
from environments.shared.tests.pilot_rollout_golden import (
    PORTABLE_ABS_TOL,
    PORTABLE_REL_TOL,
    STEP_DIGEST_HEX,
    dumps,
    numerics_environment,
    plain,
    portable_environment,
    typed,
)

FIXTURE_SCHEMA = "mesozoic.none-stage-rollout-golden/v1"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "none_stage_rollout_golden.json"

#: A training seed and the first two certification-panel seeds.
SEEDS: tuple[int, ...] = (42, 3042, 3043)
STEPS = 300
ACTION_KINDS: tuple[str, ...] = ("zero", "random")
ACTION_SEED = 20261124
#: Half the action range: the small bipeds fall several times in 300 steps and the quadrupeds stay up, so both the
#: termination and reset path and the long-episode reward terms are covered.
ACTION_AMPLITUDE = 0.5
BLOCK_STEPS = 25
#: Steps per rollout in the ``portable`` rows.
PORTABLE_STEPS = 2
#: The columns every ``portable`` row starts with; the info values follow, in the row's info key order.
PORTABLE_PREFIX: tuple[str, ...] = ("reward", "terminated", "truncated", "info_keys_sha256")
#: The record components ``component_sha256`` digests separately.
COMPONENTS: tuple[str, ...] = ("reward", "flags", "info", "observation", "reset")

#: Every command-free stage of the manifest: each model's stance, and its recovery where one is declared.
CAPTURE_SPECS: tuple[tuple[ModelId, str], ...] = (
    (ModelId("velociraptor_mongoliensis", "anatomical"), "stance"),
    (ModelId("tyrannosaurus_rex", "anatomical"), "stance"),
    (ModelId("tyrannosaurus_rex", "anatomical"), "recovery"),
    (ModelId("brachiosaurus_altithorax", "anatomical"), "stance"),
    (ModelId("dibothrosuchus_elaphros", "anatomical"), "stance"),
    (ModelId("compsognathus_longipes", "anatomical"), "stance"),
    (ModelId("compsognathus_longipes", "anatomical"), "recovery"),
    (ModelId("compsognathus_longipes", "robot"), "stance"),
    (ModelId("compsognathus_longipes", "robot"), "recovery"),
    (ModelId("allosaurus_fragilis", "anatomical"), "stance"),
)


def capture_id(model_id: ModelId, stage: str) -> str:
    """A test id for one capture (the fixture itself nests ``[species][embodiment][stage]``)."""
    return f"{model_id.species}-{model_id.embodiment}-{stage}"


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def observation_record(observation: Any) -> list[Any]:
    """The observation's dtype, shape and the sha256 of its float64 bytes."""
    array = np.asarray(observation)
    as_float64 = np.ascontiguousarray(array, dtype=np.float64)
    return [array.dtype.str, list(array.shape), hashlib.sha256(as_float64.tobytes()).hexdigest()]


def _reset_record(env: Any, observation: Any, info: dict[str, Any]) -> list[Any]:
    starts = getattr(env, "_push_schedule_starts", None)
    directions = getattr(env, "_push_schedule_directions", None)
    state = {
        "info": [list(info), typed(info)],
        "qpos": typed(np.array(env.data.qpos, dtype=np.float64)),
        "qvel": typed(np.array(env.data.qvel, dtype=np.float64)),
        "rng_state": typed(env.np_random.bit_generator.state),
        "push_schedule_starts": typed(None if starts is None else np.array(starts)),
        "push_schedule_directions": typed(None if directions is None else np.array(directions)),
    }
    return [observation_record(observation), state]


def _step_parts(reward: Any, terminated: Any, truncated: Any, info: dict[str, Any], observation: Any) -> dict:
    return {
        "reward": typed(reward),
        "flags": [typed(terminated), typed(truncated)],
        "info": [list(info), typed(info)],
        "observation": observation_record(observation),
    }


def portable_row(reward: Any, terminated: Any, truncated: Any, info: dict[str, Any]) -> list[Any]:
    """One ``portable`` row: :data:`PORTABLE_PREFIX`, then every info value in the info's key order."""
    keys = _sha256(json.dumps(list(info)))[:STEP_DIGEST_HEX]
    return [plain(reward), plain(terminated), plain(truncated), keys, *(plain(value) for value in info.values())]


def _actions(kind: str, seed: int, shape: tuple[int, ...]) -> np.ndarray:
    if kind == "zero":
        return np.zeros((STEPS, *shape), dtype=np.float32)
    if kind == "random":
        generator = np.random.default_rng([ACTION_SEED, seed])
        return generator.uniform(-ACTION_AMPLITUDE, ACTION_AMPLITUDE, size=(STEPS, *shape)).astype(np.float32)
    raise ValueError(f"unknown action kind {kind!r}; expected one of {ACTION_KINDS}")


def rollout(model_id: ModelId, stage: str, seed: int, kind: str) -> dict[str, Any]:
    """One rollout's record (see the module docstring), and the first step's info keys under ``info_keys``."""
    from environments.shared.config import build_env

    env = build_env(model_id, stage)
    try:
        actions = _actions(kind, seed, tuple(env.action_space.shape))
        components = {name: hashlib.sha256() for name in COMPONENTS}
        whole = hashlib.sha256()
        blocks = [hashlib.sha256() for _ in range(math.ceil(STEPS / BLOCK_STEPS))]

        def record(name: str, part: Any, step: int) -> None:
            text = _json([step, part])
            components[name].update(text.encode("utf-8") + b"\n")
            whole.update(_json([name, step, part]).encode("utf-8") + b"\n")
            blocks[step // BLOCK_STEPS].update(_json([name, step, part]).encode("utf-8") + b"\n")

        observation, info = env.reset(seed=int(seed))
        record("reset", _reset_record(env, observation, info), 0)
        rewards: list[float] = []
        portable: list[list[Any]] = []
        episodes: list[dict[str, Any]] = []
        info_keys: list[str] = []
        start = 0
        for step in range(STEPS):
            observation, reward, terminated, truncated, info = env.step(actions[step])
            if step == 0:
                info_keys = list(info)
            parts = _step_parts(reward, terminated, truncated, info, observation)
            for name, part in parts.items():
                record(name, part, step)
            rewards.append(float(reward))
            if step < PORTABLE_STEPS:
                portable.append(portable_row(reward, terminated, truncated, info))
            if terminated or truncated:
                ended_by = "terminated" if terminated else "truncated"
                episodes.append({"start_step": start, "length": step + 1 - start, "ended_by": ended_by})
                start = step + 1
                if step + 1 < STEPS:
                    observation, info = env.reset()
                    record("reset", _reset_record(env, observation, info), step + 1)
        if start < STEPS:
            episodes.append({"start_step": start, "length": STEPS - start, "ended_by": "running"})
        return {
            "seed": int(seed),
            "action": kind,
            "rollout_sha256": whole.hexdigest(),
            "component_sha256": {name: components[name].hexdigest() for name in COMPONENTS},
            "block_sha256": [block.hexdigest()[:STEP_DIGEST_HEX] for block in blocks],
            "episodes": episodes,
            "reward_sum": math.fsum(rewards),
            "portable": portable,
            "info_keys": info_keys,
            "command_mode": str(env.command_mode),
        }
    finally:
        env.close()


def capture(model_id: ModelId, stage: str) -> dict[str, Any]:
    """One stage's record: its task digest, command mode, first-step info keys and every rollout.

    Returned in its JSON form (lists, never tuples), so it compares equal to the fixture's.
    """
    from environments.shared.task_fingerprint import stage_task_fingerprint

    rollouts: dict[str, dict[str, Any]] = {}
    info_keys: list[str] | None = None
    command_modes = set()
    for kind in ACTION_KINDS:
        for seed in SEEDS:
            record = rollout(model_id, stage, seed, kind)
            keys = record.pop("info_keys")
            command_modes.add(record.pop("command_mode"))
            if info_keys is None:
                info_keys = keys
            rollouts.setdefault(kind, {})[str(seed)] = record
    (command_mode,) = command_modes
    document = {
        **model_id.record(),
        "stage": stage,
        "command_mode": command_mode,
        "task_sha256": stage_task_fingerprint(model_id, stage)["task_sha256"],
        "info_keys": info_keys,
        "rollouts": rollouts,
    }
    loaded: dict[str, Any] = json.loads(_json(document))
    return loaded


def stored_capture(fixture: dict[str, Any], model_id: ModelId, stage: str) -> dict[str, Any] | None:
    """The fixture's capture of (*model_id*, *stage*), or ``None``."""
    captures = fixture.get("captures", {})
    found = captures.get(model_id.species, {}).get(model_id.embodiment, {}).get(stage)
    return found if isinstance(found, dict) else None


def exact_differences(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Every field of *actual* that is not byte-for-byte *expected*'s, naming the component and block of a rollout."""
    differences = [
        f"{key}: {actual.get(key)!r} != {expected.get(key)!r}"
        for key in ("species", "embodiment", "stage", "command_mode", "task_sha256", "info_keys")
        if actual.get(key) != expected.get(key)
    ]
    for kind in sorted(set(expected.get("rollouts", {})) | set(actual.get("rollouts", {}))):
        want_kind, got_kind = expected.get("rollouts", {}).get(kind, {}), actual.get("rollouts", {}).get(kind, {})
        for seed in sorted(set(want_kind) | set(got_kind)):
            want, got = want_kind.get(seed), got_kind.get(seed)
            where = f"{kind} seed {seed}"
            if want is None or got is None:
                differences.append(f"{where}: {'not in the fixture' if want is None else 'not captured'}")
                continue
            if got == want:
                continue
            moved = [name for name in COMPONENTS if got["component_sha256"].get(name) != want["component_sha256"][name]]
            blocks = [
                index for index, pair in enumerate(zip(got["block_sha256"], want["block_sha256"])) if len(set(pair)) > 1
            ]
            if blocks:
                first = blocks[0] * BLOCK_STEPS
                at = f"first in block {blocks[0]} (steps {first}-{min(first + BLOCK_STEPS, STEPS) - 1})"
            else:
                at = "in no block"
            differences.append(f"{where}: {', '.join(moved) or 'no digested component'} moved, {at}")
            if got["episodes"] != want["episodes"]:
                differences.append(f"{where}: episodes {got['episodes']} != {want['episodes']}")
            if got["reward_sum"] != want["reward_sum"]:
                differences.append(f"{where}: reward_sum {got['reward_sum']!r} != {want['reward_sum']!r}")
            for step, (want_row, got_row) in enumerate(zip(want["portable"], got["portable"])):
                if got_row != want_row:
                    differences.append(f"{where} step {step}: the portable row moved")
    return differences


def portable_differences(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Each portable row value of *actual* outside the tolerance (floats) or unequal (the rest) to *expected*'s.

    The task digest, command mode and info keys are compared exactly; a row whose info key order moved is reported
    once, without comparing its values.
    """
    differences = [
        f"{key}: {actual.get(key)!r} != {expected.get(key)!r}"
        for key in ("command_mode", "task_sha256", "info_keys")
        if actual.get(key) != expected.get(key)
    ]
    for kind, by_seed in expected.get("rollouts", {}).items():
        for seed, want in by_seed.items():
            got = actual.get("rollouts", {}).get(kind, {}).get(seed)
            where = f"{kind} seed {seed}"
            if got is None:
                differences.append(f"{where}: not captured")
                continue
            if len(got["portable"]) != len(want["portable"]):
                differences.append(f"{where}: {len(got['portable'])} portable rows, expected {len(want['portable'])}")
                continue
            for step, (want_row, got_row) in enumerate(zip(want["portable"], got["portable"])):
                found = row_differences(want_row, got_row, expected.get("info_keys"))
                differences.extend(f"{where} step {step}: {text}" for text in found)
    return differences


def row_differences(expected: list[Any], actual: list[Any], info_keys: list[str] | None = None) -> list[str]:
    """Each column of one ``portable`` row where *actual* leaves *expected*'s tolerance or exact value.

    *info_keys* names the info columns when the row has exactly that many (the capture's first-step keys).
    """
    keys_column = PORTABLE_PREFIX.index("info_keys_sha256")
    if len(actual) != len(expected) or actual[keys_column] != expected[keys_column]:
        return [f"the info key order moved ({len(actual)} columns, expected {len(expected)})"]
    names = list(PORTABLE_PREFIX)
    if info_keys is not None and len(info_keys) == len(expected) - len(names):
        names.extend(info_keys)
    differences = []
    for column, (want, got) in enumerate(zip(expected, actual)):
        name = names[column] if column < len(names) else f"info value {column - len(PORTABLE_PREFIX)}"
        if type(got) is not type(want):
            same = False
        elif isinstance(want, float):
            same = math.isclose(got, want, rel_tol=PORTABLE_REL_TOL, abs_tol=PORTABLE_ABS_TOL)
        else:
            same = got == want
        if not same:
            differences.append(f"{name}: {got!r} != {want!r}")
    return differences


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _source_commit() -> str:
    try:
        return _git("rev-parse", "--short=7", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _modified_tracked_files() -> list[str] | None:
    """Every tracked file that differs from ``HEAD`` (untracked files are not listed); ``None`` without git."""
    try:
        return sorted(line for line in _git("diff", "--name-only", "HEAD").splitlines() if line)
    except (OSError, subprocess.CalledProcessError):
        return None


def build_fixture(*, source_commit: str | None = None) -> dict[str, Any]:
    """The complete fixture document."""
    captures: dict[str, Any] = {}
    for model_id, stage in CAPTURE_SPECS:
        captures.setdefault(model_id.species, {}).setdefault(model_id.embodiment, {})[stage] = capture(model_id, stage)
    return {
        "schema": FIXTURE_SCHEMA,
        "source_commit": _source_commit() if source_commit is None else source_commit,
        "source_modified_files": _modified_tracked_files(),
        "numerics_environment": numerics_environment(),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "action_kinds": list(ACTION_KINDS),
        "action_seed": ACTION_SEED,
        "action_amplitude": ACTION_AMPLITUDE,
        "block_steps": BLOCK_STEPS,
        "portable_steps": PORTABLE_STEPS,
        "portable_prefix": list(PORTABLE_PREFIX),
        "components": list(COMPONENTS),
        "spec": [{**model_id.record(), "stage": stage} for model_id, stage in CAPTURE_SPECS],
        "captures": captures,
    }


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        fixture: dict[str, Any] = json.load(stream)
    return fixture


def numerics_mismatch(fixture: dict[str, Any]) -> list[str]:
    """The numerics-environment keys in which this process differs from *fixture*'s (empty: the digests are exact)."""
    recorded, running = fixture["numerics_environment"], numerics_environment()
    return [key for key in sorted(set(recorded) | set(running)) if recorded.get(key) != running.get(key)]


def portable_mismatch(fixture: dict[str, Any]) -> list[str]:
    """The keys in which this process's portable environment differs from *fixture*'s (empty: the rows compare)."""
    recorded = portable_environment(fixture["numerics_environment"])
    running = portable_environment()
    return sorted(key for key in recorded if recorded[key] != running[key])


def check(fixture: dict[str, Any]) -> list[str]:
    """Every difference between the current tree's captures and *fixture*'s, exact where the numerics match."""
    exact = not numerics_mismatch(fixture)
    if not exact and portable_mismatch(fixture):
        return [f"cannot compare here: the fixture's {portable_mismatch(fixture)} differ from this machine's"]
    differences = []
    for model_id, stage in CAPTURE_SPECS:
        expected = stored_capture(fixture, model_id, stage)
        if expected is None:
            differences.append(f"{model_id} {stage}: not in the fixture")
            continue
        actual = capture(model_id, stage)
        found = exact_differences(expected, actual) if exact else portable_differences(expected, actual)
        differences.extend(f"{model_id} {stage}: {text}" for text in found)
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="re-capture the fixture instead of checking it")
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH, help="the fixture to check or write")
    parser.add_argument("--source-commit", default=None, help="with --write: the commit to record instead of HEAD")
    args = parser.parse_args(argv)
    if args.write:
        fixture = build_fixture(source_commit=args.source_commit)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(dumps(fixture) + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(CAPTURE_SPECS)} stages, {args.out.stat().st_size} bytes)")
        return 0
    fixture = load_fixture(args.out)
    mode = "exact" if not numerics_mismatch(fixture) else f"portable (numerics differ in {numerics_mismatch(fixture)})"
    differences = check(fixture)
    for line in differences:
        print(line)
    print(f"{args.out}: {len(differences)} difference(s), {mode} comparison of {len(CAPTURE_SPECS)} stages")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
