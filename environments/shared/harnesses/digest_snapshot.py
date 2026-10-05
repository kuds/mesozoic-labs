"""Print every identity and digest a certified run depends on; CI checks it.

The acceptance check for a change that claims to move no digest (retiring a
backend, a refactor, a dependency bump).  Its output for the committed tree
is committed as ``configs/digest_snapshot.generated.txt``, and CI compares a
full run with that golden on every pull request (see "The committed golden"
below): any line that differs names the digest that moved.  One
tab-separated line per value, in a fixed order, with no timestamps and no
checkout paths, so two runs on the same tree are byte-identical:

    plant.check_plant_manifest  OK
    plant     <species>  <identity key>  <value>
    policy    <species>  <payload key>   <digest>
    stage     <species>  <stage id>      <name>  <digest>
    recovery  <species>  <name>          <digest or OK>
    behavior  <species>  <recipe>        <name>  <digest>
    reward    <species>  <stage id>      <name>  <summary, poses or digest>
    <label>   ERROR      <exception>     <first line of its message>

The sections are the plant identities and their policy-interface payload
(whole and per key), each stage's task fingerprint, gate, hyperparameter and
``stage_config.json`` digests for PPO and SAC, the recovery calibrations,
every behavior recipe's file and task fingerprint digests, and (CU-11) what
each stage's reward, info and termination code computes: a fixed capture per
stage (a noisy roll that ends in a held kick, an unseeded second reset, zero
action through the pushes of a stage that has them, one-step probes: the
target moved onto the effector, the root lifted, rolled, both, and a
three-step horizon; and state probes, scored without a physics step: twelve
root and neck poses sized by the stage's own thresholds and a non-finite
velocity), printed as a summary (steps, end and reward sum per stepped
part), the end of each state probe, a digest of the discrete
records (info keys and their order, flags, reasons, the reset's generator
state) and a digest of every value rounded to 6 decimals (7 significant
digits from 10 up), which held across Python 3.11 to 3.13, numpy 2.0 to
2.5, x86 numpy and OpenBLAS dispatch and a libm without FMA (numpy 1.x,
which promotes float32 differently, moves a line).  A value
that cannot be computed prints an ERROR line instead of stopping the run, so
the diff shows the failure.

The committed golden (D-D22).  The plant-contract CI job runs the full
harness, never ``--skip-behaviors``, from the repository root with the
canonical MuJoCo of ``configs/plant_versions.toml``:

    python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check

A pull request that moves a digest on purpose regenerates the golden the
same way and commits it in its own diff, where the move is reviewed; every
other pull request leaves it unchanged:

    python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write
    git diff configs/digest_snapshot.generated.txt    # only the moves you meant

``--check [PATH]`` compares the run with PATH (default: the golden; a
relative PATH is under ``--repo``) and, when they differ, names every moved
line and prints the diff and the ``--write`` command.  ``--write [PATH]``
writes the run to PATH, and refuses a run with an ERROR line.  Both print
no snapshot lines, imply ``--block-optional-backends`` and refuse
``--skip-behaviors``.

Comparing two checkouts.  The digests are computed by whichever
``environments`` package Python imports, and ``configs/`` is read from
``--repo``; both must be the same checkout.  A base older than this file does
not have it, so run the head's copy BY FILE PATH, with ``PYTHONPATH`` set to
the checkout being measured (repo root, head checked out):

    git worktree add --detach /tmp/base <base commit>
    PYTHONPATH=/tmp/base python environments/shared/harnesses/digest_snapshot.py \\
        --repo /tmp/base --block-optional-backends > base.txt
    PYTHONPATH=. python environments/shared/harnesses/digest_snapshot.py \\
        --block-optional-backends > head.txt
    diff base.txt head.txt              # must print nothing
    git worktree remove /tmp/base

Do not run ``python -m environments.shared.harnesses.digest_snapshot --repo
<other checkout>``: ``-m`` imports this checkout's ``environments`` before the
arguments are read, so the plant and stage digests would come from here and
the recipe files from there.  The harness refuses any run in which
``environments.__file__`` does not resolve under ``--repo`` (default: the
working directory).  ``-m`` from the root of the checkout being measured,
without ``--repo``, is fine.

``--exact`` prints only the reward captures, one bit-exact digest per part
and stream (the reset record with the raw state after the reset, the
reward, the flags, each info key, the observation, qpos and qvel), with
every behavior recipe's env after the stages (a seeded episode of up to 650
zero-action steps, past the first command switch unless it ends first, as
the trex terrain recipes' episodes do; unseeded resets; and on terrain the
root moved off the flat spawn apron) unless
``--skip-behaviors``.  The rounded golden cannot see a change of a few ulp
(a reordered sum, a dtype); ``--exact`` can, but only on one machine and in
one environment, so it is never a golden: run it on both checkouts as above,
each with its own copy when both have it, and ``diff`` (CU-12, PR-8, PR-9).

``--block-optional-backends`` makes every import of jax, jaxlib, flax, optax,
``mujoco.mjx``, ray, mjlab, hypertune and ``google.cloud`` raise
ImportError, so the run also proves the digests are computable on an
SB3-only install; it refuses to start if one of them is already imported.
``--skip-behaviors`` leaves out the behavior section, which builds one
environment per recipe and is most of the run time.

Exit status: 0 when every value was computed (and, with ``--check``, the
run equals PATH), 1 when any ERROR line was printed (the rest of the
snapshot is still complete) or ``--check`` found a difference, 2 when the
invocation is refused (also when ``--check`` finds no file at PATH).  A
one-line summary (line and error counts, elapsed time) goes to stderr, so it
never reaches the diff.
"""

from __future__ import annotations

import argparse
import difflib
import functools
import hashlib
import importlib.abc
import itertools
import json
import logging
import math
import os
import shlex
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

#: The optional backends an SB3-only install does not have.
BLOCKED_BACKENDS = ("jax", "jaxlib", "flax", "optax", "mujoco.mjx", "ray", "mjlab", "hypertune", "google.cloud")

#: ``stage_config.json`` keys left out of ``stage_config_view_sha256``: the
#: line already names the species and stage, and the rest change per run.
UNPINNED_STAGE_CONFIG_KEYS = ("species", "stage", "library_version", "git_commit", "run", "gpu")

#: The committed golden (D-D22): the full output for the committed tree,
#: relative to the repository root.  The default PATH of --check and --write.
GOLDEN = "configs/digest_snapshot.generated.txt"

#: The command that regenerates GOLDEN, which a failed --check prints.
WRITE_COMMAND = "python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write"

#: The reward captures (CU-11).  Every part starts from ``reset(seed=REWARD_SEED)`` unless it says otherwise.
REWARD_SEED, REWARD_ACTION_SEED = 1042, 7
#: The roll: uniform action noise at the species' amplitude, then a held-sign kick until the episode ends.
REWARD_NOISE_STEPS, REWARD_KICK_STEPS = 300, 200
#: Per species: the roll's noise amplitude (half the largest that kept every stage of the species up for 300
#: steps and past the first push), and the MJCF element the success probe moves the target (mocap body 0) onto.
REWARD_SPECIES = {
    "brachiosaurus": (0.2, "site", "head_tip"),
    "compsognathus": (0.01, "body", "pelvis"),
    "compsognathus_robot": (0.01, "body", "pelvis"),
    "dibothrosuchus": (0.2, "geom", "snout_snap"),
    "trex": (0.1, "geom", "head_bite"),
    "velociraptor": (0.1, "geom", "r_claw_geom"),
}
#: State probes: (name, root dz as a fraction of the healthy_z_range width, root pitch about world y as
#: (sign, kind): zero, mid (half max_tilt_angle), in / out (max_tilt_angle -/+ REWARD_TILT_MARGIN), roll about
#: world x beyond max_tilt_angle or None, every limited neck/head hinge at its upper limit).  Each is scored
#: without a physics step.  The angles are sized by the stage's own threshold but applied on top of the reset
#: orientation (world pre-multiplication), as dz is added to the reset height, so a stage whose reset pose leans
#: is offset by that lean: velociraptor's home keyframe leans about 20 degrees forward, so its fwd_in ends past
#: max_tilt_angle and its back_out inside it.  The margin test checks the actual distances from the thresholds.
REWARD_TILT_MARGIN = 0.15
REWARD_POSES: tuple[tuple[str, float, tuple[int, str], float | None, bool], ...] = (
    ("low", -0.5, (1, "zero"), None, False),
    ("high", 1.05, (1, "zero"), None, False),
    ("low_roll", -1.2, (1, "zero"), 0.2, False),
    ("fwd_mid", 0.0, (1, "mid"), None, False),
    ("fwd_in", 0.0, (1, "in"), None, False),
    ("fwd_out", 0.0, (1, "out"), None, False),
    ("back_mid", 0.0, (-1, "mid"), None, False),
    ("back_in", 0.0, (-1, "in"), None, False),
    ("back_out", 0.0, (-1, "out"), None, False),
    ("back_in_low", -0.25, (-1, "in"), None, False),
    ("neck_mid", 0.0, (1, "mid"), None, True),
    ("neck_mid_low", -0.3, (1, "mid"), None, True),
)
#: --exact's behavior recipes: up to 650 zero-action steps (past the first command switch, except where the episode
#: ends first, as the trex terrain recipes' episodes do), then unseeded resets (episodes 1-9).
REWARD_BEHAVIOR_STEPS, REWARD_BEHAVIOR_RESETS = 650, 9
#: A capture part: its name, its set-up (reset seed, pose, horizon) and its actions (or a count of zero actions).
_Part = tuple[str, dict[str, Any], Any]


def _is_blocked(name: str) -> bool:
    return any(name == blocked or name.startswith(blocked + ".") for blocked in BLOCKED_BACKENDS)


class _BackendBlocker(importlib.abc.MetaPathFinder):
    """Make every import of an optional backend raise ImportError."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        if _is_blocked(fullname):
            raise ImportError(f"blocked optional backend: {fullname}")
        return None


class _Snapshot:
    """Prints the snapshot lines (or keeps them, for --check and --write) and counts them and the ERROR lines."""

    def __init__(self, repo: Path, *, debug: bool = False, keep: bool = False) -> None:
        self.repo = repo
        self.debug = debug
        self.keep = keep
        self.kept: list[str] = []
        self.lines = 0
        self.errors = 0

    def emit(self, *parts: object) -> None:
        line = "\t".join(str(part) for part in parts)
        if self.keep:
            self.kept.append(line)
        else:
            print(line, flush=True)
        self.lines += 1

    def guard(self, label: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Return ``func(*args, **kwargs)``, or print an ERROR line and return None."""
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            # The repo path is replaced so that the same failure on two
            # checkouts prints the same line.
            message = str(exc).replace(str(self.repo), "<repo>")
            self.emit(label, "ERROR", type(exc).__name__, message.splitlines()[0][:300] if message else "")
            self.errors += 1
            if self.debug:
                traceback.print_exc()
            return None


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plant_section(out: _Snapshot) -> dict[str, Any]:
    """Plant identities, and the policy-interface payload digest by key."""
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.plant_contract.manifest import check_plant_manifest
    from environments.shared.plant_contract.versions import _species_entries, load_plant_versions

    if out.guard("plant.check_plant_manifest", check_plant_manifest) is not None:
        out.emit("plant.check_plant_manifest", "OK")
    registry = out.guard("plant.registry", lambda: (load_plant_versions()[1], _species_entries()))
    if registry is None:
        return {}
    versions, entries = registry
    identities: dict[str, Any] = {}
    for species in sorted(entries):
        # The committed manifest must still describe the live plant ...
        if out.guard(f"plant.{species}.verify_generated", current_plant_identity, species) is not None:
            out.emit("plant", species, "verify_generated", "OK")
        # ... and the live identity prints either way, so a diff names the moved digest.
        identity = out.guard(f"plant.{species}.identity", current_plant_identity, species, verify_generated=False)
        if identity is None:
            continue
        identities[species] = identity
        for key, value in sorted(identity.to_dict().items()):
            out.emit("plant", species, key, value)
        out.guard(
            f"policy.{species}", _policy_lines, out, species, str(entries[species]["env_entrypoint"]), versions[species]
        )
    return identities


def _policy_lines(out: _Snapshot, species: str, entrypoint: str, version: Any) -> None:
    # Private helpers on purpose: the per-key digests name the part of the
    # payload that moved, which the identity's one digest cannot.
    from environments.shared.plant_contract.constants import POLICY_INTERFACE_SCHEMA
    from environments.shared.plant_contract.digests import _canonical_value, _semantic_digest
    from environments.shared.plant_contract.manifest import _load_environment
    from environments.shared.plant_contract.policy_layer import _policy_interface_payload

    env = _load_environment(entrypoint)(reset_noise_scale=0.0)
    try:
        payload = _policy_interface_payload(env.model, env, version, require_backend_parity=True)
    finally:
        env.close()
    out.emit("policy", species, "WHOLE", _semantic_digest(POLICY_INTERFACE_SCHEMA, payload))
    for key in sorted(payload):
        out.emit("policy", species, key, _semantic_digest("breakdown", _canonical_value(payload[key])))
    for group in ("interface_implementations", "jax_interface"):
        value = payload.get(group)
        if isinstance(value, dict):
            for sub in sorted(value):
                out.emit(
                    "policy", species, f"{group}.{sub}", _semantic_digest("breakdown", _canonical_value(value[sub]))
                )


def stage_section(out: _Snapshot, identities: dict[str, Any]) -> None:
    """Per stage: task fingerprint, gate, hyperparameters, stage_config.json body."""
    from environments.shared.config import SPECIES_NAMES, load_stage_config
    from environments.shared.stage_manifest import load_stage_manifest

    with tempfile.TemporaryDirectory(prefix="digest_snapshot_") as scratch:
        for species in SPECIES_NAMES:
            manifest = out.guard(f"stage.{species}.manifest", load_stage_manifest, species)
            if manifest is None:
                continue
            for entry in manifest.stages:
                config = out.guard(f"stage.{species}.{entry.id}.config", load_stage_config, species, entry.reference)
                if config is not None:
                    _stage_lines(out, species, entry, config, identities.get(species), Path(scratch))


def _stage_lines(
    out: _Snapshot, species: str, entry: Any, config: dict[str, Any], identity: Any, scratch: Path
) -> None:
    from environments.shared.config import hyperparameters_sha256
    from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
    from environments.shared.task_fingerprint import derive_stage_task_fingerprint

    ref = entry.reference
    label = f"{species}\t{entry.id}"
    prefix = f"stage.{species}.{entry.id}"
    out.emit("stage", label, "config_file", entry.config_file)
    if identity is not None:
        # The call train_base makes before it writes stage_config.json.
        fingerprint = out.guard(
            f"{prefix}.task",
            derive_stage_task_fingerprint,
            species=species,
            stage=ref,
            backend="stable-baselines3",
            env_kwargs=config.get("env_kwargs", {}),
            plant_identity=identity.to_dict(),
        )
        if fingerprint is not None:
            out.emit("stage", label, "task_sha256", fingerprint["task_sha256"])
    curriculum = config.get("curriculum_kwargs", {})
    gate = out.guard(f"{prefix}.gate", lambda: gate_config_sha256(gate_config_view(curriculum)))
    if gate is not None:
        out.emit("stage", label, "gate_sha256", gate)
    for algorithm in ("PPO", "SAC"):
        digest = out.guard(f"{prefix}.hyperparameters", hyperparameters_sha256, config, algorithm)
        if digest is not None:
            out.emit("stage", label, f"hyperparameters_sha256.{algorithm}", digest)
    for algorithm in ("PPO", "SAC"):
        stage_dir = scratch / species / str(entry.id) / algorithm
        digest = out.guard(f"{prefix}.view", _stage_config_view_sha256, species, ref, config, algorithm, stage_dir)
        if digest is not None:
            out.emit("stage", label, f"stage_config_view_sha256.{algorithm}", digest)


def _stage_config_view_sha256(species: str, stage: Any, config: dict[str, Any], algorithm: str, stage_dir: Path) -> str:
    """Digest of the stage_config.json a new stage records, minus UNPINNED_STAGE_CONFIG_KEYS.

    Written by the real writer, with the env class train_base passes, so a
    change to what a stage records shows here.
    """
    from environments.shared.config import save_stage_config
    from environments.shared.result_bundle.hashing import canonical_json_sha256
    from environments.shared.species_registry import get_species_config

    env_class = get_species_config(species).env_class
    path = save_stage_config(stage_dir, stage, config, algorithm, env_class=env_class, species=species)
    record = json.loads(path.read_text())
    return canonical_json_sha256({key: value for key, value in record.items() if key not in UNPINNED_STAGE_CONFIG_KEYS})


def recovery_section(out: _Snapshot) -> None:
    """Every committed recovery calibration: loads, and its file digest."""
    from environments.shared.recovery_calibration import load_recovery_calibration

    for path in sorted(Path("configs").glob("*/recovery_calibration.json")):
        species = path.parent.name
        if out.guard(f"recovery.{species}", load_recovery_calibration, species) is not None:
            out.emit("recovery", species, "load_recovery_calibration", "OK")
        out.emit("recovery", species, "file_sha256", "sha256:" + _file_sha(path))


def behavior_section(out: _Snapshot) -> None:
    """Every behavior recipe: its file digest and its task fingerprint's digest."""
    for recipe_path in sorted(Path("configs").glob("*/behaviors/*.toml")):
        species = recipe_path.parent.parent.name
        label = f"{species}\t{recipe_path.stem}"
        out.emit("behavior", label, "recipe_sha256", _file_sha(recipe_path))
        value = out.guard(f"behavior.{species}.{recipe_path.stem}", _behavior_task, recipe_path, species)
        if value is not None:
            out.emit("behavior", label, "task_sha256", value["task_sha256"])


def _behavior_env(recipe_path: Path, species: str) -> Any:
    """The recipe's environment, built as train_behaviors builds it."""
    from environments.shared.train_behaviors import create_behavior_env, read_recipe

    _, commands, terrain, kwargs = read_recipe(recipe_path, species)
    return create_behavior_env(species, commands=commands, terrain=terrain, run_seed=0, **kwargs)


def _behavior_task(recipe_path: Path, species: str) -> Any:
    """The task fingerprint the recipe's environment records."""
    env = _behavior_env(recipe_path, species)
    try:
        return env.task_fingerprint
    finally:
        env.close()


def reward_section(out: _Snapshot, exact: bool = False, behaviors: bool = False) -> None:
    """Per stage: the capture's summary, its state probes' ends, a digest of its discrete records and one of its
    rounded values.

    With *exact* (--exact), one bit-exact digest per part and stream instead, also for every behavior recipe
    with *behaviors*; exact digests move across machines and environments, so they are never a golden.
    """
    for species, name, env in _reward_envs(out, behaviors):
        parts = _behavior_parts if hasattr(env, "terrain_config") else _stage_parts
        # Encoded inside the guard, so a value no encoder takes is an ERROR line, not a stopped run.
        lines = out.guard(f"reward.{species}.{name}", lambda: _reward_lines(_capture(env, parts(env, species)), exact))
        for kind, *fields in lines or ():
            out.emit(kind, species, name, *fields)


def _reward_lines(capture: list[tuple[str, Any, list[Any]]], exact: bool) -> list[tuple[str, ...]]:
    """(section, *fields) per line: one per part and stream when *exact*, else the capture's four golden lines."""
    if exact:
        return [("reward-exact", part, stream, digest) for (part, stream), digest in _streams(capture, _exact).items()]
    shape = _streams(capture, lambda value: "float" if _is_float(value) else value)
    return [
        ("reward", "summary", _summary(capture)),
        ("reward", "poses", _poses(capture)),
        ("reward", "shape_sha256", _sha(sorted(shape.items()))),
        ("reward", "rounded_values_sha256", _sha(sorted(_streams(capture, _rounded).items()))),
    ]


def _reward_envs(out: _Snapshot, behaviors: bool) -> Iterator[tuple[str, str, Any]]:
    """(species, stage id or recipe, env), each closed after use: the stages as train_base.make_env builds them."""
    from environments.shared.config import SPECIES_NAMES, load_stage_config
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import load_stage_manifest

    def stage_env(species: str, stage: Any) -> Any:
        return get_species_config(species).env_class(**load_stage_config(species, stage)["env_kwargs"])

    builds: list[tuple[str, str, Callable[[], Any]]] = []
    for species in SPECIES_NAMES:
        for entry in getattr(out.guard(f"reward.{species}.manifest", load_stage_manifest, species), "stages", ()):
            builds.append((species, str(entry.id), functools.partial(stage_env, species, entry.reference)))
    for path in sorted(Path("configs").glob("*/behaviors/*.toml")) if behaviors else []:
        builds.append(
            (path.parent.parent.name, path.stem, functools.partial(_behavior_env, path, path.parent.parent.name))
        )
    for species, name, build in builds:
        env = out.guard(f"reward.{species}.{name}.env", build)
        if env is not None:
            try:
                yield species, name, env
            finally:
                env.close()


def _stage_parts(env: Any, species: str) -> list[_Part]:
    """(part, set-up, actions or zero-action steps): the roll, zero action if the stage pushes, the probes."""
    import numpy as np

    shape, (amplitude, *effector) = env.action_space.shape, REWARD_SPECIES[species]

    def roll() -> Iterator[Any]:
        rng = np.random.default_rng(REWARD_ACTION_SEED)
        for _ in range(REWARD_NOISE_STEPS):
            yield rng.uniform(-amplitude, amplitude, size=shape).astype(np.float32)
        sign = rng.choice(np.array([-1.0, 1.0]), size=shape)
        for _ in range(REWARD_KICK_STEPS):
            yield np.clip(0.9 * sign + rng.uniform(-0.1, 0.1, size=shape), -1.0, 1.0).astype(np.float32)

    env.reset(seed=REWARD_SEED)
    pushes: list[_Part] = [("zero", {}, env.max_episode_steps)] if env._push_schedule_starts is not None else []
    return [
        ("roll", {}, roll()),
        ("second_reset", {"seed": None}, 0),
        *pushes,
        ("success", {"target": effector}, 1),
        ("too_high", {"lift": True}, 1),
        ("tilt", {"roll": True}, 1),
        ("high_tilt", {"lift": True, "roll": True}, 1),
        ("truncation", {"horizon": 3}, 3),
        *_state_parts(),
    ]


def _state_parts() -> list[_Part]:
    """The state probes: every pose of REWARD_POSES, then a non-finite velocity; scored without a physics step."""
    poses: list[_Part] = [
        (name, {"state": (dz, pitch, roll, neck)}, None) for name, dz, pitch, roll, neck in REWARD_POSES
    ]
    return [*poses, ("nonfinite", {"nonfinite": True}, None)]


def _behavior_parts(env: Any, species: str) -> list[_Part]:
    """A seeded zero-action episode, unseeded resets and, on terrain, the root halfway from the apron to the edge."""
    parts: list[_Part] = [("episode", {}, REWARD_BEHAVIOR_STEPS)]
    parts += [(f"reset{index}", {"seed": None}, 0) for index in range(1, REWARD_BEHAVIOR_RESETS + 1)]
    terrain = env.terrain_config
    if terrain is not None:
        parts.append(("off_apron", {"x": (terrain.apron_radius + terrain.extent) / 2}, 1))
    return parts


def _capture(env: Any, parts: list[_Part]) -> list[tuple[str, Any, list[Any]]]:
    """Per part: (part, reset record, rows of reward, info, terminated, truncated, obs, qpos, qvel)."""
    import mujoco
    import numpy as np

    from environments.shared.tests.reset_golden import _reset_record

    zero, captured = np.zeros(env.action_space.shape, dtype=np.float32), []
    for part, setup, actions in parts:
        setup, horizon = dict(setup), env.max_episode_steps
        obs, info = env.reset(seed=setup.pop("seed", REWARD_SEED))
        env.max_episode_steps = setup.pop("horizon", horizon)
        if setup:
            _pose(env, **setup)
            mujoco.mj_forward(env.model, env.data)
        # The reset record rounds the state; "state" keeps it raw for --exact, and the golden leaves it out.
        state = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.mocap_pos.copy()]
        record, rows = {**_reset_record(env), "info": info, "obs": obs, "state": state}, []
        if actions is None:  # a state probe: score the posed state as step() would, without a physics step
            env._invalidate_substep_aggregates()
            if setup.get("nonfinite"):
                terminated, info = env._is_terminated()
                reward, obs = 0.0, np.zeros(0, dtype=np.float32)
            else:
                obs = env._get_obs()
                reward, info = env._get_reward_info(zero)
                terminated, term_info = env._is_terminated()
                info = {**info, **term_info}
            rows.append((reward, info, terminated, False, obs, env.data.qpos.copy(), env.data.qvel.copy()))
            actions = 0
        try:
            for action in itertools.repeat(zero, actions) if isinstance(actions, int) else actions:
                obs, reward, terminated, truncated, info = env.step(action)
                rows.append((reward, info, terminated, truncated, obs, env.data.qpos.copy(), env.data.qvel.copy()))
                if terminated or truncated:
                    break
        finally:
            env.max_episode_steps = horizon
        captured.append((part, record, rows))
    return captured


def _pose(
    env: Any,
    *,
    lift: bool = False,
    roll: bool = False,
    target: Any = None,
    x: float = 0.0,
    state: Any = None,
    nonfinite: bool = False,
) -> None:
    """Move the root (up by the healthy height range, rolled past max_tilt_angle, along x, or to a REWARD_POSES
    *state*), the target onto an MJCF element, or make the velocity non-finite; the caller runs mj_forward."""
    import mujoco
    import numpy as np

    qpos = env.data.qpos
    qpos[0] += x
    if nonfinite:
        env.data.qvel[0] = np.nan
    if state is not None:
        dz, (sign, kind), extra_roll, neck = state
        model, limit = env.model, float(env.max_tilt_angle)
        if neck:
            for joint in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
                hinge = model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
                if hinge and model.jnt_limited[joint] and ("neck" in name or "head" in name):
                    qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint][1]
        qpos[2] += dz * (env.healthy_z_range[1] - env.healthy_z_range[0])
        pitch = (
            sign
            * {"zero": 0.0, "mid": 0.5 * limit, "in": limit - REWARD_TILT_MARGIN, "out": limit + REWARD_TILT_MARGIN}[
                kind
            ]
        )
        rotation = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
        if extra_roll is not None:
            half = (limit + extra_roll) / 2
            rolled = np.zeros(4)
            mujoco.mju_mulQuat(rolled, np.array([np.cos(half), np.sin(half), 0.0, 0.0]), rotation)
            rotation = rolled
        mujoco.mju_mulQuat(qpos[3:7], rotation, qpos[3:7].copy())
    if lift:
        qpos[2] += env.healthy_z_range[1] - env.healthy_z_range[0]
    if roll:  # about world x
        half = (env.max_tilt_angle + 0.2) / 2
        mujoco.mju_mulQuat(qpos[3:7], np.array([np.cos(half), np.sin(half), 0.0, 0.0]), qpos[3:7].copy())
    if target is not None:
        kind, name = target
        positions = getattr(env.data, {"site": "site_xpos", "geom": "geom_xpos", "body": "xpos"}[kind])
        point = positions[getattr(env.model, kind)(name).id].copy()
        if kind == "body":  # the compsognathus target stays on the ground, under the pelvis
            point[2] = env.data.mocap_pos[0][2]
        env.data.mocap_pos[0] = point


def _is_float(value: Any) -> bool:
    import numpy as np

    return isinstance(value, (float, np.floating))


def _rounded(value: Any) -> Any:
    """A float to 6 decimals, or 7 significant digits from 10 up; -0.0 reads 0.0; anything else unchanged."""
    if not _is_float(value) or not math.isfinite(value) or value == 0:
        return float(value) + 0.0 if _is_float(value) else value
    return round(float(value), 6 - max(0, int(math.log10(abs(value))))) + 0.0


def _exact(value: Any) -> Any:
    return [type(value).__name__, float(value).hex() if _is_float(value) else value]


def _encoded(value: Any, encode: Callable[[Any], Any]) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _encoded(item, encode) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_encoded(item, encode) for item in value]
    return encode(value)


def _streams(capture: list[tuple[str, Any, list[Any]]], encode: Callable[[Any], Any]) -> dict[tuple[str, str], str]:
    """One digest per (part, stream): reset record, reward, flags, each info key; obs and state when exact."""
    hashes: dict[tuple[str, str], Any] = {}

    def feed(part: str, stream: str, value: Any) -> None:
        hashes.setdefault((part, stream), hashlib.sha256()).update(_sha(_encoded(value, encode)).encode())

    for part, record, rows in capture:
        feed(
            part,
            "reset",
            record if encode is _exact else {k: v for k, v in record.items() if k not in ("obs", "state")},
        )
        for step, (reward, info, terminated, truncated, *state) in enumerate(rows, 1):
            feed(part, "reward", reward)
            feed(part, "flags", [list(info), bool(terminated), bool(truncated)])
            for key, value in info.items():
                feed(part, f"info:{key}", [step, value])
            for stream, array in zip(("obs", "qpos", "qvel"), state if encode is _exact else ()):
                feed(part, stream, array)
    return {key: "sha256:" + digest.hexdigest() for key, digest in hashes.items()}


_STATE_PARTS = {name for name, *_ in REWARD_POSES} | {"nonfinite"}


def _poses(capture: list[tuple[str, Any, list[Any]]]) -> str:
    """Per state probe: the termination reason, or - when the posed state is alive."""
    ends = []
    for part, _, rows in capture:
        if part in _STATE_PARTS:
            _, info, terminated, *_ = rows[0]
            ends.append(f"{part}={info.get('termination_reason', '?') if terminated else '-'}")
    return " ".join(ends)


def _summary(capture: list[tuple[str, Any, list[Any]]]) -> str:
    """Per part: steps, how it ended (the reason, truncated, or - when it did not end) and the reward sum."""
    parts = []
    for part, _, rows in capture:
        if part in _STATE_PARTS:
            continue
        end = "-"
        if rows:
            _, info, terminated, truncated, *_ = rows[-1]
            end = str(info.get("termination_reason", "?")) if terminated else "truncated" if truncated else "-"
        parts.append(f"{part}={len(rows)}:{end}:{round(math.fsum(float(row[0]) for row in rows), 4) + 0.0:.4f}")
    return " ".join(parts)


def render(lines: Sequence[str]) -> str:
    """The snapshot as a file: every line newline-terminated (what --write writes)."""
    return "".join(f"{line}\n" for line in lines)


def _is_error(line: str) -> bool:
    return line.split("\t")[1:2] == ["ERROR"]


def _digest_name(line: str) -> str:
    """Every field of *line* but its value, space-separated: the name of the digest it prints."""
    return (line.rpartition("\t")[0] or line).replace("\t", " ")


def _write_command(label: str) -> str:
    """The command that regenerates *label*: the committed golden, or the PATH given to --check."""
    return WRITE_COMMAND if label == GOLDEN else f"{WRITE_COMMAND} {shlex.quote(label)}"


def golden_mismatch(golden: str, lines: Sequence[str], errors: int, label: str) -> str | None:
    """None when *lines* reproduce the text *golden* with no ERROR line; otherwise the failure message.

    The message names every moved line (changed, added or removed), then
    prints the unified diff and what to do: fix the ERROR lines, or, for a
    deliberate move, regenerate the golden in the same pull request.
    """
    if render(lines) == golden and not errors:
        return None
    # A byte-order mark or extra final newlines move no value: they are reported as the file's form below.
    expected, actual = golden.removeprefix("\ufeff").rstrip("\n").splitlines(), list(lines)
    report = [f"digest_snapshot: {label} does not match this checkout."]
    if expected != actual:
        old, new = set(expected), set(actual)
        gone = Counter(_digest_name(line) for line in expected if line not in new)
        came = Counter(_digest_name(line) for line in actual if line not in old)
        changed = gone & came
        # One entry per moved line, so two identical ERROR lines count twice.
        moved = [
            (kind, name)
            for kind, names in (("changed", changed), ("added", came - changed), ("removed", gone - changed))
            for name in names.elements()
        ]
        if moved:
            counts = ", ".join(f"{sum(k == kind for k, _ in moved)} {kind}" for kind in ("changed", "added", "removed"))
            report.append(f"{len(moved)} digest line(s) moved ({counts}):")
            report += [f"  {kind:<8} {name}" for kind, name in moved]
        else:
            report.append("No value moved, but the lines differ in order or number:")
        report += difflib.unified_diff(expected, actual, f"{label} (committed)", "this checkout", n=0, lineterm="")
    elif render(lines) != golden:
        report.append(
            f"Every line matches, but {label} is not the text --write writes"
            " (a byte-order mark, or not exactly one final newline)."
        )
    if errors:
        report.append(f"{errors} value(s) could not be computed (--debug prints each traceback):")
        report += [f"  {line}" for line in lines if _is_error(line)]
        report.append("Fix these first: the golden never holds an ERROR line, and --write refuses a run with one.")
    else:
        command = _write_command(label)
        report += [
            "If every move above is deliberate, regenerate the golden (from the repository root, with the canonical",
            "MuJoCo) and commit it in this pull request, where the move is reviewed (D-D22):",
            f"    {command}",
            f"A change that claims to move no digest leaves {label} unchanged: find what moved it instead.",
        ]
    return "\n".join(report)


def check_golden(path: Path, label: str, lines: Sequence[str], errors: int) -> int:
    """--check: 0 when *lines* reproduce *path* with no ERROR line, else print why and return 1."""
    mismatch = golden_mismatch(path.read_text(encoding="utf-8"), lines, errors, label)
    if mismatch is not None:
        print(mismatch, file=sys.stderr)
        return 1
    print(f"Digest snapshot is current: {label} ({len(lines)} lines)")
    return 0


def write_golden(path: Path, label: str, lines: Sequence[str], errors: int) -> int:
    """--write: replace *path* with *lines*, unless the run printed an ERROR line (then 1, nothing written)."""
    if errors:
        print(f"digest_snapshot: not writing {label}: {errors} value(s) could not be computed:", file=sys.stderr)
        for line in lines:
            if _is_error(line):
                print(f"  {line}", file=sys.stderr)
        return 1
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_text(render(lines), encoding="utf-8", newline="\n")
    temp.replace(path)
    print(f"Wrote {label} ({len(lines)} lines)")
    return 0


def _refuse(message: str) -> int:
    print(f"digest_snapshot: refused: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print every identity and digest a certified run depends on, one per line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--repo", default=".", help="the checkout to measure (default: the working directory)")
    parser.add_argument(
        "--block-optional-backends",
        action="store_true",
        help="make jax, flax, optax, mujoco.mjx, ray, mjlab, hypertune and google.cloud imports fail",
    )
    parser.add_argument("--skip-behaviors", action="store_true", help="leave out the behavior section (the slow part)")
    parser.add_argument("--debug", action="store_true", help="print the traceback of every ERROR line to stderr")
    golden_mode = parser.add_mutually_exclusive_group()
    golden_mode.add_argument(
        "--exact",
        action="store_true",
        help="print only the reward captures, bit-exact, to diff two checkouts on one machine (never a golden)",
    )
    golden_mode.add_argument(
        "--check",
        nargs="?",
        const=GOLDEN,
        metavar="PATH",
        help=f"compare the full run with PATH (default: {GOLDEN}) and name every moved line",
    )
    golden_mode.add_argument(
        "--write",
        nargs="?",
        const=GOLDEN,
        metavar="PATH",
        help=f"write the full run to PATH (default: {GOLDEN}), after a deliberate digest move",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()

    target: str | None = args.check if args.check is not None else args.write
    if target is not None:
        if args.skip_behaviors:
            return _refuse("--check and --write need the full run (D-D22); drop --skip-behaviors")
        args.block_optional_backends = True
    if args.block_optional_backends:
        loaded = [name for name in BLOCKED_BACKENDS if any(m == name or m.startswith(name + ".") for m in sys.modules)]
        if loaded:
            return _refuse(f"--block-optional-backends, but already imported: {', '.join(loaded)}")
        sys.meta_path.insert(0, _BackendBlocker())

    repo = Path(args.repo).resolve()
    if not (repo / "environments" / "__init__.py").is_file():
        return _refuse(f"--repo {repo} is not a checkout (no environments/__init__.py)")
    golden = None if target is None else repo / target
    label = ""
    if golden is not None:
        label = golden.relative_to(repo).as_posix() if golden.is_relative_to(repo) else str(golden)
        if args.check is not None and not golden.is_file():
            return _refuse(f"--check: no file at {label}; generate it with `{_write_command(label)}`")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import environments

    # Under ``python -m``, or when another checkout is installed, the package
    # may already come from elsewhere: then the plant and stage digests would
    # not be --repo's.
    origin = environments.__file__
    if origin is None or Path(origin).resolve() != repo / "environments" / "__init__.py":
        return _refuse(
            f"environments is imported from {origin}, not from --repo {repo}; "
            f"run this file by path with PYTHONPATH={repo} (see --help)"
        )

    logging.disable(logging.CRITICAL)
    out = _Snapshot(repo, debug=args.debug, keep=golden is not None)
    if args.exact:
        reward_section(out, exact=True, behaviors=not args.skip_behaviors)
    else:
        identities = plant_section(out)
        stage_section(out, identities)
        recovery_section(out)
        if not args.skip_behaviors:
            behavior_section(out)
        reward_section(out)
    print(
        f"digest_snapshot: {out.lines} lines, {out.errors} errors, {time.monotonic() - started:.1f} s ({repo})",
        file=sys.stderr,
    )
    if golden is None:
        return 1 if out.errors else 0
    finish = check_golden if args.check is not None else write_golden
    return finish(golden, label, out.kept, out.errors)


if __name__ == "__main__":
    raise SystemExit(main())
