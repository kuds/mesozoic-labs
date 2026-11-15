"""Track B's one loader: checkpoint files, their integrity, the policy fingerprint, version sets, and the tar that
carries a checkpoint from Colab to this machine (``STAGE2_PLAN.md`` S2-12, S2-13, S2-14; Track B rules K2, K4, K9).

A checkpoint of a job is four files under ``<job>/checkpoints/``, written in this order, each through a temporary
name and ``os.replace``:

* ``model_<n>.zip``: the stable-baselines3 archive (``model.save``).  It is refused, and nothing of the checkpoint is
  left, if the repository's ``inspect_sb3_archive`` finds a member with bytecode in it (S2-14: schedules are instances
  of ``designlab.stage2.schedules``' classes, pickled by reference);
* ``vecnormalize_<n>.pkl``: the ``VecNormalize`` wrapper (``vecnorm.save``); only a training resume unpickles it to
  use it;
* ``fingerprint_<n>.npz``: ``obs`` (256 seeded observations, ``fingerprint_obs``), ``actions`` (the deterministic
  float32 actions the policy and the statistics give them at save time) and the observation statistics as plain
  arrays (``obs_mean``, ``obs_var``, ``obs_count``, ``clip_obs``, ``epsilon``);
* ``checkpoint_<n>.json`` (schema ``s2-ckpt-2``): the three files' names, sizes and sha256, the library versions,
  the layout, the run's ``run_uuid`` and settings hash, and the run state a resume starts from.  It is written last:
  its presence with matching hashes is what "an intact checkpoint" means, on local disk and on Drive.

Every load, anywhere in Track B (the trainer's resume, the monitor, the policy rows, the smoke), goes through
``load_checkpoint``: ``pair_problem`` (this module's hashes and sizes, then the repository's
``checkpoint_pair_problem``), the load through the repository's ``load_sb3_model`` with the schedule objects passed
explicitly and ``force_reset=False``, then the fingerprint reproduced from the stored observations and the plain
statistics.  ``max |difference| > 1e-5`` raises ``CheckpointError``: the checkpoint is not used and not judged
(S2-12).  **An inference load unpickles as little as it can**: it passes the two spaces (rebuilt from the record's
layout) and ``None`` for the archive's last-observation members as ``custom_objects`` and normalises from the plain
arrays, so that only torch tensors and class references cross library versions.

``pack`` writes ``ckpt_<n>.tar`` (the three binaries, byte-identical whenever the files are: USTAR, sorted names,
mtime 0, uid and gid 0, empty user and group names, mode 0644, no compression); ``import_tar`` unpacks one here,
checks it against its record and returns the validation record (``stage2/tb/validation/colab_archive.json`` when the
CLI's ``--write`` names it).

torch, stable-baselines3, gymnasium and the repository's ``environments`` package are imported inside the functions
that need them: importing this module imports none of them (a test asserts it).  This file is in the sources of
Track B's policy rows.

    python -m designlab.stage2.s2load check JOBDIR N
    python -m designlab.stage2.s2load pack JOBDIR N --out DIR
    python -m designlab.stage2.s2load import --record checkpoint_<n>.json --tar PATH --into JOBDIR [--write PATH]
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":                       # before any numpy import (round A's module skeleton)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse
import hashlib
import re
import tarfile
from pathlib import Path

import numpy as np

from designlab.stage2 import s2io, schedules, tbenv

SCHEMA = "s2-ckpt-2"
IMPORT_SCHEMA = "s2-ckpt-import-1"
#: The libraries whose versions a checkpoint records and a resume or a judged row compares (K4).
VERSION_KEYS = ("python", "numpy", "mujoco", "scipy", "torch", "stable_baselines3", "gymnasium", "cloudpickle")
FINGERPRINT_ROWS, FINGERPRINT_TOL = 256, 1e-5
#: First entry of the fingerprint generator's seed ``[20261115, observation width, actor width]`` (K9).
FINGERPRINT_SEED = 20261115
#: The three binaries of a checkpoint, in the order they are written; the record is the fourth file.
FILE_KINDS = ("model", "vecnormalize", "fingerprint")
FILE_NAMES = {"model": "model_%d.zip", "vecnormalize": "vecnormalize_%d.pkl", "fingerprint": "fingerprint_%d.npz",
              "record": "checkpoint_%d.json"}
#: The plain normalisation arrays of ``fingerprint_<n>.npz`` (``stats_of``).
STATS_KEYS = ("obs_mean", "obs_var", "obs_count", "clip_obs", "epsilon")
#: What ``save_checkpoint``'s ``run`` holds; the first and the last are what ``expect_run`` compares.
RUN_KEYS = ("run_uuid", "run_id", "job", "settings_sha256")
#: PPO's clip range in a training load (design 2's value; the trainer builds the model with the float 0.2).
CLIP_RANGE = 0.2
#: The archive members an inference load replaces by ``None`` instead of unpickling them.
INFERENCE_NONE = ("_last_obs", "_last_original_obs", "_last_episode_starts", "ep_info_buffer", "ep_success_buffer")

#: How each actor block of the layout (design section 3.1) is drawn for the fingerprint's observations: per row, in
#: the order of the columns.  ``normal``: scale x z.  ``gravity``: (0, 0, -1) + scale x z, renormalised.  ``uniform``:
#: on [lo, hi).  ``circle``: (sin, cos) of an angle uniform on [0, 2 pi).  ``bits``: 0 or 1.  ``columns``: one uniform
#: range per column; the command block is ``[v / 0.5 m/s, w / 1.0 rad/s]`` at the commanded clock (5 % fast), so its
#: ranges are 0 to 1.05 (0 to 0.50 m/s) and -1.05 to 1.05 (turns of up to 1.0 rad/s either way).
FINGERPRINT_BLOCKS = {
    "q": ("normal", 0.3),
    "qd": ("normal", 2.0),
    "gravity": ("gravity", 0.05),
    "gyro": ("normal", 0.5),
    "foot_load": ("uniform", (0.0, 1.0)),
    "phase": ("circle", None),
    "support": ("bits", None),
    "command": ("columns", ((0.0, 1.05), (-1.05, 1.05))),
    "heading": ("normal", 0.3),
    "prev_actions": ("normal", 0.1),
    "head_command": ("uniform", (-1.0, 1.0)),
}

FINGERPRINT_RULE = (
    "every checkpoint is saved with 256 observations drawn from numpy.random.default_rng([20261115, observation "
    "width, actor width]) (each actor block from a fixed distribution of its kind: joint angles 0.3 z, rates 2 z, "
    "gravity (0, 0, -1) + 0.05 z renormalised, gyro 0.5 z, foot loads uniform on 0-1, phase a point of the unit "
    "circle, support bits, commands uniform on their ranges, the heading 0.3 z, previous actions 0.1 z, the head "
    "command uniform on +-1; the critic's columns zero), with the deterministic float32 actions that its policy and "
    "its normalisation statistics give them at save time, and with the statistics as plain arrays; a load must "
    "reproduce every action within 1e-5 (absolute), and a checkpoint that does not is not used and not judged "
    "(STAGE2_PLAN.md S2-12; Track B rule K9)")
LOADER_RULE = (
    "every load goes through s2load.load_checkpoint: the three files' sizes and sha256 against checkpoint_<n>.json, "
    "the repository's checkpoint_pair_problem, then environments.shared.policy_loading.load_sb3_model with the "
    "schedule objects passed explicitly (instances of designlab.stage2.schedules' classes; an archive holds no "
    "bytecode) and force_reset=False; a load for judging also passes the two spaces and None for the "
    "last-observation members and normalises from the plain arrays, so that only tensors and class references cross "
    "library versions; torch's build label is dropped before versions are compared (STAGE2_PLAN.md S2-12, S2-14; "
    "Track B rule K9)")


class CheckpointError(RuntimeError):
    """A checkpoint cannot be written, is not intact, is another run's, or does not reproduce its fingerprint: it is
    not used, and if it was to be judged it is not judged (S2-12).  ``fingerprint_max_abs_diff`` holds the measured
    difference when that is the reason, else None."""

    fingerprint_max_abs_diff = None


# ------------------------------------------------------------------------------------------------
# Versions
# ------------------------------------------------------------------------------------------------
def versions() -> dict:
    """The imported modules' own ``__version__`` for ``VERSION_KEYS`` (as ``train._versions`` of Stage 1 reads them),
    torch's build label ("+cpu", "+cu...") kept: this is what a record stores."""
    import platform

    import cloudpickle
    import gymnasium
    import mujoco
    import scipy
    import stable_baselines3
    import torch

    return {"python": platform.python_version(), "numpy": str(np.__version__), "mujoco": str(mujoco.__version__),
            "scipy": str(scipy.__version__), "torch": str(torch.__version__),
            "stable_baselines3": str(stable_baselines3.__version__), "gymnasium": str(gymnasium.__version__),
            "cloudpickle": str(cloudpickle.__version__)}


def comparable(v: dict) -> dict:
    """What is compared of a version set (a record's, this process's, or a table of pins): its entries for
    ``VERSION_KEYS``, in that order, with torch's build label dropped (S2-12: "2.15.0+cpu" and "2.15.0+cu126" are the
    version 2.15.0).  A key the set does not hold is left out, so a set that lacks one differs from a set that has
    it."""
    out = {}
    for key in VERSION_KEYS:
        if key not in v:
            continue
        x = v[key]
        out[key] = x.split("+", 1)[0] if key == "torch" and isinstance(x, str) else x
    return out


# ------------------------------------------------------------------------------------------------
# Names
# ------------------------------------------------------------------------------------------------
def _steps(steps) -> int:
    n = int(steps)
    if n != steps or n < 0:
        raise ValueError(f"a step count is a whole number >= 0, not {steps!r}")
    return n


def checkpoint_files(job_dir, steps) -> dict:
    """``{"model", "vecnormalize", "fingerprint", "record"}``: the four paths of checkpoint ``steps`` of a job."""
    n, ck = _steps(steps), Path(job_dir) / "checkpoints"
    return {kind: ck / (name % n) for kind, name in FILE_NAMES.items()}


def list_checkpoints(job_dir) -> list:
    """The step counts that have a ``checkpoint_<n>.json``, sorted by the number in the name."""
    ck = Path(job_dir) / "checkpoints"
    if not ck.is_dir():
        return []
    found = (re.fullmatch(r"checkpoint_(\d+)\.json", p.name) for p in ck.iterdir())
    return sorted(int(m.group(1)) for m in found if m)


def _orphans(job_dir) -> list:
    """Step counts with a binary under ``checkpoints/`` and no record: what a save killed before its last file left."""
    ck = Path(job_dir) / "checkpoints"
    if not ck.is_dir():
        return []
    have, out = set(list_checkpoints(job_dir)), set()
    for p in ck.iterdir():
        m = re.fullmatch(r"(?:model_(\d+)\.zip|vecnormalize_(\d+)\.pkl|fingerprint_(\d+)\.npz)", p.name)
        if m:
            n = int(next(g for g in m.groups() if g is not None))
            if n not in have:
                out.add(n)
    return sorted(out)


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def remove_checkpoint(job_dir, steps) -> list:
    """Delete checkpoint ``steps`` (K2), the record first: at no moment does a record stand beside a missing file.
    The names removed."""
    files, gone = checkpoint_files(job_dir, steps), []
    for kind in ("record",) + FILE_KINDS:
        for p in (files[kind], _tmp(files[kind])):
            if p.exists():
                p.unlink()
                gone.append(p.name)
    return gone


# ------------------------------------------------------------------------------------------------
# The layout and the fingerprint
# ------------------------------------------------------------------------------------------------
def _dims(layout: dict) -> tuple:
    try:
        obs_dim, actor_dim, action_dim = (int(layout[k]) for k in ("obs_dim", "actor_dim", "action_dim"))
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"a layout holds obs_dim, actor_dim and action_dim ({exc!r})") from exc
    if not (0 < actor_dim <= obs_dim and action_dim > 0):
        raise CheckpointError(f"layout widths actor {actor_dim}, observation {obs_dim}, action {action_dim}")
    return obs_dim, actor_dim, action_dim


def _actor_blocks(layout: dict) -> list:
    """``[(name, i0, i1)]`` of the actor's blocks in column order; they must cover columns 0 to ``actor_dim``
    exactly, and no block may lie across ``actor_dim``."""
    obs_dim, actor_dim, _ = _dims(layout)
    blocks = layout.get("blocks")
    if not isinstance(blocks, dict) or not blocks:
        raise CheckpointError("the layout holds no blocks (name -> [i0, i1]): the fingerprint is drawn block by block")
    spans = sorted((int(i0), int(i1), str(name)) for name, (i0, i1) in blocks.items())
    out, at = [], 0
    for i0, i1, name in spans:
        if not 0 <= i0 < i1 <= obs_dim:
            raise CheckpointError(f"layout block {name!r} [{i0}, {i1}) outside 0..{obs_dim}")
        if i0 >= actor_dim:
            continue
        if i1 > actor_dim:
            raise CheckpointError(f"layout block {name!r} [{i0}, {i1}) lies across actor_dim {actor_dim}")
        if i0 != at:
            raise CheckpointError(f"the actor's blocks do not tile 0..{actor_dim}: block {name!r} starts at {i0}, "
                                  f"the one before ends at {at}")
        out.append((name, i0, i1))
        at = i1
    if at != actor_dim:
        raise CheckpointError(f"the actor's blocks end at {at}, actor_dim is {actor_dim}")
    return out


def fingerprint_obs(layout: dict) -> np.ndarray:
    """The fingerprint's observations: ``[256, obs_dim] float32`` from ``numpy.random.default_rng([20261115, obs_dim,
    actor_dim])``, the actor's blocks drawn one after the other in column order, each as one ``[256, width]`` draw
    of its kind (``FINGERPRINT_BLOCKS``); the critic's columns are zero (the actor does not read them).  ``layout``:
    ``{"blocks": {name: [i0, i1]}, "actor_dim", "obs_dim", "action_dim"}``.  A block name without a distribution is
    refused."""
    obs_dim, actor_dim, _ = _dims(layout)
    rng = np.random.default_rng([FINGERPRINT_SEED, obs_dim, actor_dim])
    rows = FINGERPRINT_ROWS
    out = np.zeros((rows, obs_dim), dtype=np.float64)
    for name, i0, i1 in _actor_blocks(layout):
        if name not in FINGERPRINT_BLOCKS:
            raise CheckpointError(f"no fingerprint distribution for the actor block {name!r} "
                                  f"(known: {', '.join(FINGERPRINT_BLOCKS)})")
        kind, arg = FINGERPRINT_BLOCKS[name]
        w = i1 - i0
        if kind == "normal":
            x = arg * rng.standard_normal((rows, w))
        elif kind == "gravity":
            if w != 3:
                raise CheckpointError(f"block {name!r} is {w} wide, a gravity vector has 3 entries")
            x = np.array([0.0, 0.0, -1.0]) + arg * rng.standard_normal((rows, 3))
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        elif kind == "uniform":
            x = rng.uniform(arg[0], arg[1], (rows, w))
        elif kind == "circle":
            if w != 2:
                raise CheckpointError(f"block {name!r} is {w} wide, a phase is (sin, cos)")
            angle = rng.uniform(0.0, 2.0 * np.pi, rows)
            x = np.stack([np.sin(angle), np.cos(angle)], axis=1)
        elif kind == "bits":
            x = rng.integers(0, 2, (rows, w)).astype(np.float64)
        else:  # "columns"
            if w != len(arg):
                raise CheckpointError(f"block {name!r} is {w} wide, its ranges are {len(arg)}")
            x = np.stack([rng.uniform(lo, hi, rows) for lo, hi in arg], axis=1)
        out[:, i0:i1] = x
    return out.astype(np.float32)


def stats_of(vecnorm) -> dict:
    """A ``VecNormalize``'s observation statistics as plain float64 arrays: ``{"obs_mean", "obs_var", "obs_count",
    "clip_obs", "epsilon"}`` (copies)."""
    rms = vecnorm.obs_rms
    if isinstance(rms, dict) or not getattr(vecnorm, "norm_obs", True):
        raise CheckpointError("the plain statistics need a VecNormalize that normalises one flat Box observation")
    values = {"obs_mean": rms.mean, "obs_var": rms.var, "obs_count": rms.count, "clip_obs": vecnorm.clip_obs,
              "epsilon": vecnorm.epsilon}
    return {key: np.array(values[key], dtype=np.float64) for key in STATS_KEYS}


def normalize(stats: dict, obs) -> np.ndarray:
    """``clip((obs - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)`` as float32: ``VecNormalize.normalize_obs``'s
    formula on plain arrays.  ``obs`` may hold the actor's columns alone (they are the first of the observation): the
    leading entries of the statistics are then used, and ``PrivilegedCriticPolicy`` pads the rest."""
    x = np.asarray(obs)
    mean, var = np.asarray(stats["obs_mean"]), np.asarray(stats["obs_var"])
    w = x.shape[-1] if x.ndim else 0
    if w > mean.shape[0] or w == 0:
        raise CheckpointError(f"an observation {w} wide against statistics {mean.shape[0]} wide")
    if w < mean.shape[0]:
        mean, var = mean[:w], var[:w]
    clip, eps = float(stats["clip_obs"]), float(stats["epsilon"])
    return np.clip((x - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)


def fingerprint(model, stats: dict, obs) -> np.ndarray:
    """The deterministic actions of ``model`` on ``normalize(stats, obs)``: ``[rows, action_dim] float32``."""
    x = np.asarray(obs)
    actions, _ = model.predict(normalize(stats, x), deterministic=True)
    return np.asarray(actions, dtype=np.float32).reshape(x.shape[0], -1)


def _obs_sha256(obs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(obs, dtype="<f4").tobytes()).hexdigest()


def _clipped_rows(stats: dict, obs: np.ndarray, actor_dim: int) -> int:
    """How many rows the normalisation clips in at least one of the actor's columns."""
    x = np.asarray(obs)[:, :actor_dim]
    z = (x - stats["obs_mean"][:actor_dim]) / np.sqrt(stats["obs_var"][:actor_dim] + float(stats["epsilon"]))
    return int(np.count_nonzero(np.any(np.abs(z) > float(stats["clip_obs"]), axis=1)))


def _layout_record(layout: dict) -> dict:
    """The layout as the record holds it: what was given (blocks included, so that ``fingerprint_obs`` can be drawn
    again from the record alone), with ``actor_dim``, ``obs_dim``, ``action_dim``, ``nj`` and ``heading`` present."""
    obs_dim, actor_dim, action_dim = _dims(layout)
    blocks = {name: [i0, i1] for name, i0, i1 in _actor_blocks(layout)}
    lay = s2io.plain(dict(layout))
    lay.update(obs_dim=obs_dim, actor_dim=actor_dim, action_dim=action_dim)
    if lay.get("nj") is None:
        lay["nj"] = (blocks["q"][1] - blocks["q"][0]) if "q" in blocks else None
    lay["heading"] = "heading" in blocks
    return lay


def _check_model(model, vecnorm_stats: dict, layout: dict) -> None:
    obs_dim, actor_dim, action_dim = _dims(layout)
    have = (tuple(model.observation_space.shape), tuple(model.action_space.shape))
    if have != ((obs_dim,), (action_dim,)):
        raise CheckpointError(f"the model's spaces {have} are not the layout's ({obs_dim},), ({action_dim},)")
    actor = getattr(model.policy, "actor_dim", None)
    if actor is not None and int(actor) != actor_dim:
        raise CheckpointError(f"the policy's actor_dim {actor} is not the layout's {actor_dim}")
    if vecnorm_stats["obs_mean"].shape != (obs_dim,) or vecnorm_stats["obs_var"].shape != (obs_dim,):
        raise CheckpointError(f"the statistics are {vecnorm_stats['obs_mean'].shape[0]} wide, the layout {obs_dim}")


def _run_part(run: dict) -> dict:
    unknown = sorted(set(run) - set(RUN_KEYS))
    if unknown:
        raise CheckpointError(f"run holds {unknown}; a checkpoint stores {list(RUN_KEYS)}")
    for key in ("run_uuid", "settings_sha256"):
        if not isinstance(run.get(key), str) or not run[key]:
            raise CheckpointError(f"run[{key!r}] must be a non-empty string (a resume and a judged row compare it)")
    return {key: run.get(key) for key in RUN_KEYS}


# ------------------------------------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------------------------------------
def save_checkpoint(job_dir, steps, model, vecnorm, layout, run: dict, state: dict) -> dict:
    """Write checkpoint ``steps`` of a job: the four files in order, each atomic; the record (also returned).

    ``layout``: ``r5core.layout``'s dict (``blocks``, ``actor_dim``, ``obs_dim``, ``action_dim``, ...).  ``run``:
    ``{run_uuid, run_id, job, settings_sha256}``.  ``state``: the run state a resume starts from, stored as given
    (JSON values).  An older record of the same step is removed first, so a save that is cut leaves no record.
    ``CheckpointError`` (and none of the three binaries left) if the archive holds a member with bytecode, or if
    the model or the statistics are not the layout's.  Nothing here draws a random number or resets an env."""
    tbenv.repo_on_path()
    from environments.shared.policy_loading import inspect_sb3_archive

    n = _steps(steps)
    files = checkpoint_files(job_dir, n)
    lay = _layout_record(layout)
    run_part = _run_part(run)
    run_state = s2io.plain(state)                 # before any file is touched: a state that is no JSON stops here
    stats = stats_of(vecnorm)
    _check_model(model, stats, lay)
    obs = fingerprint_obs(lay)
    files["record"].parent.mkdir(parents=True, exist_ok=True)
    _unlink(files["record"])
    try:
        tmp = _tmp(files["model"])
        with open(tmp, "wb") as fh:
            model.save(fh)
            fh.flush()
            os.fsync(fh.fileno())
        bytecode = sorted(inspect_sb3_archive(tmp).bytecode_members)
        if bytecode:
            raise CheckpointError(
                f"checkpoint {n}: the archive's member(s) {', '.join(bytecode)} hold bytecode (a schedule that is a "
                "function or a closure is pickled by value); schedules are instances of designlab.stage2.schedules' "
                "classes (S2-14): nothing is saved")
        os.replace(tmp, files["model"])
        tmp = _tmp(files["vecnormalize"])
        vecnorm.save(str(tmp))
        _fsync(tmp)
        os.replace(tmp, files["vecnormalize"])
        actions = fingerprint(model, stats, obs)
        if actions.shape != (FINGERPRINT_ROWS, lay["action_dim"]) or not np.all(np.isfinite(actions)):
            raise CheckpointError(f"checkpoint {n}: the fingerprint actions are not {FINGERPRINT_ROWS} finite rows of "
                                  f"{lay['action_dim']}")
        s2io.write_npz(files["fingerprint"], {"obs": obs, "actions": actions, **stats})
    except BaseException:
        for kind in FILE_KINDS:
            _unlink(files[kind])
            _unlink(_tmp(files[kind]))
        raise
    record = {"schema": SCHEMA, **run_part}
    record.update(
        steps=n, num_timesteps=int(model.num_timesteps),
        files={kind: {"name": files[kind].name, "bytes": files[kind].stat().st_size,
                      "sha256": s2io.sha256_file(files[kind])} for kind in FILE_KINDS},
        versions=versions(), layout=lay,
        fingerprint={"rows": FINGERPRINT_ROWS, "seed": [FINGERPRINT_SEED, lay["obs_dim"], lay["actor_dim"]],
                     "obs_sha256": _obs_sha256(obs), "tol": FINGERPRINT_TOL,
                     "clipped_rows": _clipped_rows(stats, obs, lay["actor_dim"]), "rule": FINGERPRINT_RULE},
        run_state=run_state, saved_at=s2io.now_utc())
    record = s2io.plain(record)
    record["sha256"] = s2io.sha256_json(record)   # of the record without this key
    s2io.write_json(files["record"], record)
    return record


# ------------------------------------------------------------------------------------------------
# Integrity
# ------------------------------------------------------------------------------------------------
def _read_record(path: Path) -> tuple:
    """``(record, None)`` or ``(None, why)``: the JSON, its schema and its own sha256."""
    if not path.is_file():
        return None, f"missing record {path.name}"
    try:
        doc = s2io.read_json(path)
    except (OSError, ValueError) as exc:
        return None, f"record {path.name} is not JSON ({exc})"
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        schema = doc.get("schema") if isinstance(doc, dict) else None
        return None, f"record {path.name} has schema {schema!r}, not {SCHEMA!r}"
    body = {k: v for k, v in doc.items() if k != "sha256"}
    if doc.get("sha256") != s2io.sha256_json(body):
        return None, f"record {path.name} does not hash to its own sha256 (it was changed or cut)"
    return doc, None


def _hash_problem(job_dir, steps) -> tuple:
    """``(record, None)`` when the record reads and the three files have its names, sizes and sha256; else ``(None,
    why)``.  Imports no library."""
    n = _steps(steps)
    files = checkpoint_files(job_dir, n)
    doc, why = _read_record(files["record"])
    if why:
        return None, why
    if doc.get("steps") != n:
        return None, f"record {files['record'].name} is of step {doc.get('steps')!r}"
    entries = doc.get("files")
    if not isinstance(entries, dict) or sorted(entries) != sorted(FILE_KINDS):
        return None, f"record {files['record'].name} does not list {', '.join(FILE_KINDS)}"
    for kind in FILE_KINDS:
        if entries[kind].get("name") != files[kind].name:
            return None, f"record names {entries[kind].get('name')!r} as its {kind}, not {files[kind].name}"
    for kind in FILE_KINDS:
        if not files[kind].is_file():
            return None, f"missing file {files[kind].name}"
    for kind in FILE_KINDS:
        size = files[kind].stat().st_size
        if size != entries[kind].get("bytes"):
            return None, f"{files[kind].name} has {size} bytes, the record says {entries[kind].get('bytes')}"
    for kind in FILE_KINDS:
        sha = s2io.sha256_file(files[kind])
        if sha != entries[kind].get("sha256"):
            said = str(entries[kind].get("sha256"))
            return None, f"{files[kind].name} has sha256 {sha[:16]}..., the record says {said[:16]}..."
    return doc, None


def pair_problem(job_dir, steps) -> str | None:
    """None when checkpoint ``steps`` is intact, else why not: a missing or unreadable record, a missing file, a
    size or sha256 other than the record's, then the reason of the repository's ``checkpoint_pair_problem`` (the zip
    opens, tests clean and holds SB3's members; the statistics unpickle)."""
    _, why = _hash_problem(job_dir, steps)
    if why:
        return why
    tbenv.repo_on_path()
    from environments.shared.curriculum.checkpoints import checkpoint_pair_problem

    files = checkpoint_files(job_dir, steps)
    return checkpoint_pair_problem(files["model"], files["vecnormalize"])


def newest_intact(job_dir) -> tuple:
    """``(n, skipped)``: the newest step count without a ``pair_problem`` (None if there is none) and ``[{steps,
    reason}]`` of every newer one that was passed over, newest first; a step whose save was cut before its record is
    listed too."""
    have, skipped = set(list_checkpoints(job_dir)), []
    for n in sorted(have | set(_orphans(job_dir)), reverse=True):
        if n not in have:
            skipped.append({"steps": n, "reason": f"no record {FILE_NAMES['record'] % n} (a save that was cut)"})
            continue
        why = pair_problem(job_dir, n)
        if why is None:
            return n, skipped
        skipped.append({"steps": n, "reason": why})
    return None, skipped


def keep(steps_all: list, monitor_every: int) -> list:
    """The checkpoints K2 keeps of ``steps_all``: each check's and the newest two, ascending.  A check runs at the
    first rollout start at or past each multiple of ``monitor_every`` above zero (M2), so check k's checkpoint is the
    first one in ``[k x monitor_every, (k + 1) x monitor_every)``; the checkpoint of step 0 is no check's."""
    every = int(monitor_every)
    if every <= 0:
        raise ValueError(f"monitor_every {monitor_every!r} must be positive")
    steps = sorted({_steps(s) for s in steps_all})
    first = {}
    for n in steps:
        k = n // every
        if k >= 1 and k not in first:
            first[k] = n
    return sorted(set(first.values()) | set(steps[-2:]))


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------
def load_model(path, layout: dict, env=None, lr: tuple | None = None):
    """The PPO model of an archive, through the repository's ``load_sb3_model(path, algorithm="ppo", device="cpu",
    env=env, force_reset=False, custom_objects=...)``; no schedule of the archive is unpickled.

    A training load (``env`` given, the ``VecNormalize`` of the run; ``lr = (start, end)``): ``learning_rate`` and
    ``lr_schedule`` are ``LinearSchedule(*lr)`` and ``clip_range`` is ``ConstantSchedule(0.2)``; the archive's last
    observation is kept and nothing resets the env (``force_reset=False``).  An inference load (``env`` None): the
    three are ``ConstantSchedule(0.0)``, the two spaces are built from ``layout`` (observation +-inf, action +-1,
    float32) and the last-observation and episode-buffer members are ``None``, so none of them is unpickled.  SB3's
    load seeds python, numpy and torch with the model's seed, and the env it is given, as it always does."""
    obs_dim, _, action_dim = _dims(layout)
    if env is not None:
        if lr is None or len(lr) != 2:
            raise ValueError("a training load passes its schedule: lr=(start, end) (S2-14)")
        sched = schedules.LinearSchedule(lr[0], lr[1])
        custom = {"learning_rate": sched, "lr_schedule": sched, "clip_range": schedules.ConstantSchedule(CLIP_RANGE)}
    else:
        if lr is not None:
            raise ValueError("lr belongs to a training load (env given); an inference load has no schedule")
        from gymnasium import spaces

        zero = schedules.ConstantSchedule(0.0)
        custom = {"learning_rate": zero, "lr_schedule": zero, "clip_range": zero,
                  "observation_space": spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
                  "action_space": spaces.Box(-1.0, 1.0, (action_dim,), np.float32)}
        custom.update({name: None for name in INFERENCE_NONE})
    tbenv.repo_on_path()
    from environments.shared.policy_loading import load_sb3_model

    return load_sb3_model(str(path), algorithm="ppo", device="cpu", env=env, force_reset=False, custom_objects=custom)


def load_vecnormalize(path, venv):
    """``VecNormalize.load`` on a vec env: a training resume only (every other load reads the plain arrays).  It
    binds the wrapper to ``venv`` and neither resets nor steps it."""
    from stable_baselines3.common.vec_env import VecNormalize

    return VecNormalize.load(str(path), venv)


def _read_fingerprint(path: Path, record: dict) -> dict:
    obs_dim, _, action_dim = _dims(record["layout"])
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError) as exc:
        raise CheckpointError(f"{path.name} does not read ({exc})") from exc
    missing = [k for k in ("obs", "actions") + STATS_KEYS if k not in arrays]
    if missing:
        raise CheckpointError(f"{path.name} lacks {', '.join(missing)}")
    if arrays["obs"].shape != (FINGERPRINT_ROWS, obs_dim) or arrays["actions"].shape != (FINGERPRINT_ROWS, action_dim):
        raise CheckpointError(f"{path.name}: obs {arrays['obs'].shape}, actions {arrays['actions'].shape}; the layout "
                              f"asks ({FINGERPRINT_ROWS}, {obs_dim}) and ({FINGERPRINT_ROWS}, {action_dim})")
    if _obs_sha256(arrays["obs"]) != (record.get("fingerprint") or {}).get("obs_sha256"):
        raise CheckpointError(f"{path.name}: the observations do not hash to the record's obs_sha256")
    return arrays


def load_checkpoint(job_dir, steps, env=None, lr=None, tol=FINGERPRINT_TOL, expect_run: dict | None = None) -> dict:
    """Checkpoint ``steps`` of a job, loaded and checked: ``{"model", "stats", "vecnorm" (training only), "record",
    "versions_saved", "versions_here", "cross_version", "fingerprint_max_abs_diff", "tol"}``.

    Inference (``env`` None): the model through ``load_model`` with the record's layout; ``stats`` are the plain
    arrays of the fingerprint file (use ``normalize(stats, obs)``).  Training (``env``: the run's vec env, NOT yet
    wrapped; ``lr = (start, end)``): ``vecnorm`` is the ``.pkl`` bound to ``env``, the model is loaded on it with
    ``force_reset=False``, and the ``.pkl``'s statistics must equal the plain arrays.

    ``CheckpointError`` on a ``pair_problem``; on a ``run_uuid`` or settings hash other than ``expect_run``'s
    (``{"run_uuid", "settings_sha256"}``); on a load that fails; on fingerprint actions further than ``tol`` from
    the stored ones ("not judged": S2-12)."""
    n = _steps(steps)
    if expect_run is not None and not all(k in expect_run for k in ("run_uuid", "settings_sha256")):
        raise ValueError("expect_run holds run_uuid and settings_sha256")
    if env is not None:
        if lr is None or len(lr) != 2:
            raise ValueError("a training load passes its schedule: lr=(start, end) (S2-14)")
        from stable_baselines3.common.vec_env import VecNormalize

        if isinstance(env, VecNormalize):
            raise ValueError("pass the vec env itself: its VecNormalize is rebuilt from the checkpoint's .pkl")
    elif lr is not None:
        raise ValueError("lr belongs to a training load (env given)")
    where = f"checkpoint {n} of {job_dir}"
    why = pair_problem(job_dir, n)
    if why:
        raise CheckpointError(f"{where}: {why}")
    files = checkpoint_files(job_dir, n)
    record = s2io.read_json(files["record"])
    if expect_run is not None:
        for key in ("run_uuid", "settings_sha256"):
            if record.get(key) != expect_run[key]:
                raise CheckpointError(f"{where}: its {key} {record.get(key)!r} is not the run's {expect_run[key]!r}: "
                                      "another run's checkpoint is not used and not judged")
    arrays = _read_fingerprint(files["fingerprint"], record)
    stats = {k: arrays[k] for k in STATS_KEYS}
    out = {}
    try:
        if env is not None:
            vecnorm = load_vecnormalize(files["vecnormalize"], env)
            model = load_model(files["model"], record["layout"], env=vecnorm, lr=lr)
            out["vecnorm"] = vecnorm
        else:
            model = load_model(files["model"], record["layout"])
    except Exception as exc:  # noqa: BLE001 - whatever a library raises on an archive it cannot use: not used
        raise CheckpointError(f"{where}: does not load ({type(exc).__name__}: {exc})") from exc
    if env is not None:
        live = stats_of(out["vecnorm"])
        differ = [k for k in STATS_KEYS if not np.array_equal(live[k], stats[k])]
        if differ:
            raise CheckpointError(f"{where}: the .pkl's statistics are not the fingerprint file's "
                                  f"({', '.join(differ)})")
    actions = fingerprint(model, stats, arrays["obs"])
    if actions.shape != arrays["actions"].shape:
        raise CheckpointError(f"{where}: fingerprint not reproduced: actions {actions.shape}, stored "
                              f"{arrays['actions'].shape}: not judged (S2-12)")
    diff = float(np.max(np.abs(actions.astype(np.float64) - arrays["actions"].astype(np.float64))))
    if not diff <= float(tol):
        err = CheckpointError(f"{where}: fingerprint not reproduced: max |difference| {diff!r} > {float(tol)!r} over "
                              f"{FINGERPRINT_ROWS} observations: not judged (S2-12)")
        err.fingerprint_max_abs_diff = diff
        raise err
    here = versions()
    saved = record.get("versions") or {}
    out.update(model=model, stats=stats, record=record, versions_saved=saved, versions_here=here,
               cross_version=comparable(saved) != comparable(here), fingerprint_max_abs_diff=diff, tol=float(tol))
    return out


# ------------------------------------------------------------------------------------------------
# The tar of a checkpoint and its import
# ------------------------------------------------------------------------------------------------
def tar_name(steps) -> str:
    return "ckpt_%d.tar" % _steps(steps)


def _tar_members(path) -> list:
    """``[{name, bytes, sha256}]`` of a plain tar's members, in archive order; a member that is no regular file, a
    name with a directory in it or a name twice is refused."""
    out, seen = [], set()
    try:
        with tarfile.open(path, mode="r:") as tar:
            for info in tar:
                if not info.isreg() or "/" in info.name or "\\" in info.name or info.name in ("", ".", ".."):
                    raise CheckpointError(f"{Path(path).name}: member {info.name!r} is not a plain file name")
                if info.name in seen:
                    raise CheckpointError(f"{Path(path).name}: member {info.name!r} twice")
                seen.add(info.name)
                h, fh = hashlib.sha256(), tar.extractfile(info)
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
                out.append({"name": info.name, "bytes": int(info.size), "sha256": h.hexdigest()})
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise CheckpointError(f"{Path(path).name} is not a readable plain tar ({exc})") from exc
    return out


def pack(job_dir, steps, out) -> dict:
    """Write ``<out>/ckpt_<n>.tar``: the three binaries of an intact checkpoint (sizes and sha256 checked against
    the record first and again in the written tar), with the bundle's tar rules: USTAR, names sorted, mtime 0, uid
    and gid 0, empty user and group names, mode 0644, no compression.  The same files give the same bytes.  Returns
    ``{tar, name, bytes, sha256, steps, members [{name, bytes, sha256}], record {name, sha256}}``."""
    n = _steps(steps)
    record, why = _hash_problem(job_dir, n)
    if why:
        raise CheckpointError(f"checkpoint {n} of {job_dir}: {why}: not packed")
    files = checkpoint_files(job_dir, n)
    want = sorted(({"name": record["files"][k]["name"], "bytes": record["files"][k]["bytes"],
                    "sha256": record["files"][k]["sha256"]} for k in FILE_KINDS), key=lambda m: m["name"])
    by_name = {files[k].name: files[k] for k in FILE_KINDS}
    path = Path(out) / tar_name(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    try:
        with open(tmp, "wb") as fh:
            with tarfile.open(fileobj=fh, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for member in want:
                    info = tarfile.TarInfo(member["name"])
                    info.size, info.mtime, info.mode, info.type = member["bytes"], 0, 0o644, tarfile.REGTYPE
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with open(by_name[member["name"]], "rb") as src:
                        tar.addfile(info, src)
            fh.flush()
            os.fsync(fh.fileno())
        if _tar_members(tmp) != want:
            raise CheckpointError(f"checkpoint {n} of {job_dir}: a file changed while it was packed")
        os.replace(tmp, path)
    except BaseException:
        _unlink(tmp)
        raise
    return {"tar": str(path), "name": path.name, "bytes": path.stat().st_size, "sha256": s2io.sha256_file(path),
            "steps": n, "members": want,
            "record": {"name": files["record"].name, "sha256": s2io.sha256_file(files["record"])}}


def import_tar(record, tar, into) -> dict:
    """Bring a checkpoint that was packed elsewhere (Colab) into job directory ``into`` and validate it here.

    ``record``: its ``checkpoint_<n>.json``; ``tar``: its ``ckpt_<n>.tar``.  The tar must hold exactly the record's
    three files; each is unpacked to ``<into>/checkpoints/`` through a temporary name and must have the record's size
    and sha256; then the record is copied (last) and ``load_checkpoint`` runs (an inference load: the fingerprint
    reproduced under this machine's libraries).  Returns the validation record: ``{schema, steps, run_uuid, job,
    settings_sha256, record_sha256, tar {name, bytes, sha256}, files, versions_saved, versions_here, cross_version,
    fingerprint_max_abs_diff, tol, ok, reason, rules, at}``; ``ok`` is False with its ``reason`` when the tar, a
    file or the fingerprint fails (nothing of a failed unpack is left with a record).  ``CheckpointError`` only when
    the record itself cannot be read."""
    record_path, tar_path = Path(record), Path(tar)
    doc, why = _read_record(record_path)
    if why:
        raise CheckpointError(f"{record_path}: {why}")
    raw = record_path.read_bytes()                # now: the record may be the very file this import replaces
    n = _steps(doc["steps"])
    out = {"schema": IMPORT_SCHEMA, "steps": n, "run_uuid": doc.get("run_uuid"), "job": doc.get("job"),
           "settings_sha256": doc.get("settings_sha256"), "record_sha256": hashlib.sha256(raw).hexdigest(),
           "tar": {"name": tar_path.name, "bytes": None, "sha256": None}, "files": doc.get("files"),
           "versions_saved": doc.get("versions"), "versions_here": versions(), "cross_version": None,
           "fingerprint_max_abs_diff": None, "tol": FINGERPRINT_TOL, "ok": False, "reason": None,
           "rules": {"fingerprint": FINGERPRINT_RULE, "loader": LOADER_RULE}, "at": s2io.now_utc()}
    out["cross_version"] = comparable(out["versions_saved"] or {}) != comparable(out["versions_here"])
    files = checkpoint_files(into, n)
    placed = []
    try:
        if not tar_path.is_file():
            raise CheckpointError(f"missing tar {tar_path}")
        out["tar"].update(bytes=tar_path.stat().st_size, sha256=s2io.sha256_file(tar_path))
        entries = doc.get("files")
        if not isinstance(entries, dict) or sorted(entries) != sorted(FILE_KINDS):
            raise CheckpointError(f"the record does not list {', '.join(FILE_KINDS)}")
        want = {}
        for kind in FILE_KINDS:
            if entries[kind].get("name") != files[kind].name:
                raise CheckpointError(f"the record names {entries[kind].get('name')!r} as its {kind}, not "
                                      f"{files[kind].name}")
            want[files[kind].name] = (kind, entries[kind].get("bytes"), entries[kind].get("sha256"))
        members = _tar_members(tar_path)
        names = [m["name"] for m in members]
        if sorted(names) != sorted(want):
            raise CheckpointError(f"{tar_path.name} holds {sorted(names)}, the record lists {sorted(want)}")
        for m in members:
            kind, size, sha = want[m["name"]]
            if m["bytes"] != size or m["sha256"] != sha:
                raise CheckpointError(f"{tar_path.name}: {m['name']} has {m['bytes']} bytes and sha256 "
                                      f"{m['sha256'][:16]}..., the record says {size} and {str(sha)[:16]}...")
        files["record"].parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, mode="r:") as archive:
            for info in archive:
                kind, size, sha = want[info.name]
                tmp, h, src = _tmp(files[kind]), hashlib.sha256(), archive.extractfile(info)
                placed.append(tmp)
                with open(tmp, "wb") as fh:
                    for block in iter(lambda: src.read(1 << 20), b""):
                        h.update(block)
                        fh.write(block)
                    fh.flush()
                    os.fsync(fh.fileno())
                if h.hexdigest() != sha or tmp.stat().st_size != size:
                    raise CheckpointError(f"{info.name} as unpacked is not the record's file")
        _unlink(files["record"])                  # no record stands beside files that are being replaced
        for kind in FILE_KINDS:
            os.replace(_tmp(files[kind]), files[kind])
            placed.append(files[kind])
        tmp = _tmp(files["record"])
        placed.append(tmp)
        with open(tmp, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, files["record"])
        placed = []                               # the checkpoint stands; a failed load below leaves it for inspection
        ck = load_checkpoint(into, n)
        out.update(fingerprint_max_abs_diff=ck["fingerprint_max_abs_diff"], tol=ck["tol"],
                   versions_here=ck["versions_here"], cross_version=ck["cross_version"], ok=True)
    except CheckpointError as exc:
        for p in placed:
            _unlink(p)
        out.update(ok=False, reason=str(exc), fingerprint_max_abs_diff=exc.fingerprint_max_abs_diff)
    except BaseException:
        for p in placed:
            _unlink(p)
        raise
    return out


def write_validation(path, result: dict) -> dict:
    """Store ``import_tar``'s result as a record (``stage2/tb/validation/colab_archive.json``): the result, the
    checkout's provenance (commit, dirty paths less this file) and the record's own sha256; atomic.  Returned."""
    doc = dict(result)
    doc["provenance"] = s2io.git_state(own_outputs=[str(Path(path).resolve())])
    doc = s2io.plain(doc)
    doc["sha256"] = s2io.sha256_json(doc)         # of the record without this key
    s2io.write_json(path, doc)
    return doc


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    tbenv.assign_gl("disable")                    # this process builds a policy: no GL library may be loaded in it
    s2io.require_one_thread()
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.s2load", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("check", help="integrity and fingerprint of one checkpoint (an inference load)")
    a.add_argument("job_dir")
    a.add_argument("steps", type=int)
    a = sub.add_parser("pack", help="write ckpt_<n>.tar of one intact checkpoint")
    a.add_argument("job_dir")
    a.add_argument("steps", type=int)
    a.add_argument("--out", required=True)
    a = sub.add_parser("import", help="unpack a ckpt_<n>.tar beside its record and validate it here")
    a.add_argument("--record", required=True)
    a.add_argument("--tar", required=True)
    a.add_argument("--into", required=True)
    a.add_argument("--write", default=None, help="store the validation record at this path")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        try:
            ck = load_checkpoint(args.job_dir, args.steps)
        except CheckpointError as exc:
            print(f"S2 CHECKPOINT {args.steps}: NOT USABLE: {exc}", flush=True)
            return 1
        print(f"S2 CHECKPOINT {args.steps}: intact; fingerprint {ck['fingerprint_max_abs_diff']!r} "
              f"(max |difference| over {FINGERPRINT_ROWS} observations, tolerance {ck['tol']!r}); "
              f"cross_version {ck['cross_version']}", flush=True)
        return 0
    if args.cmd == "pack":
        try:
            got = pack(args.job_dir, args.steps, args.out)
        except CheckpointError as exc:
            print(f"S2 CHECKPOINT {args.steps}: NOT PACKED: {exc}", flush=True)
            return 1
        print(f"S2 CHECKPOINT {args.steps}: packed {got['tar']} ({got['bytes']} bytes, sha256 {got['sha256']})",
              flush=True)
        return 0
    try:
        result = import_tar(args.record, args.tar, args.into)
    except CheckpointError as exc:
        print(f"S2 CHECKPOINT IMPORT: NOT USABLE: {exc}", flush=True)
        return 1
    if args.write:
        write_validation(args.write, result)
    if result["ok"]:
        print(f"S2 CHECKPOINT {result['steps']}: imported into {args.into}; intact; fingerprint "
              f"{result['fingerprint_max_abs_diff']!r} (tolerance {result['tol']!r}); cross_version "
              f"{result['cross_version']}" + (f"; record {args.write}" if args.write else ""), flush=True)
        return 0
    print(f"S2 CHECKPOINT {result['steps']}: IMPORT FAILED: {result['reason']}"
          + (f"; record {args.write}" if args.write else ""), flush=True)
    return 1


__all__ = ["CLIP_RANGE", "CheckpointError", "FILE_KINDS", "FILE_NAMES", "FINGERPRINT_BLOCKS", "FINGERPRINT_ROWS",
           "FINGERPRINT_RULE", "FINGERPRINT_SEED", "FINGERPRINT_TOL", "IMPORT_SCHEMA", "INFERENCE_NONE", "LOADER_RULE",
           "RUN_KEYS", "SCHEMA", "STATS_KEYS", "VERSION_KEYS", "checkpoint_files", "comparable", "fingerprint",
           "fingerprint_obs", "import_tar", "keep", "list_checkpoints", "load_checkpoint", "load_model",
           "load_vecnormalize", "main", "newest_intact", "normalize", "pack", "pair_problem", "remove_checkpoint",
           "save_checkpoint", "stats_of", "tar_name", "versions", "write_validation"]

if __name__ == "__main__":
    from designlab.stage2 import s2load as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
