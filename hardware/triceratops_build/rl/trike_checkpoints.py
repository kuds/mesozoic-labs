"""Checkpoint pairs of the 60 cm Triceratops runs: find, check, save, set aside. Simulation only.

A pair is ``trike60_<steps>_steps.zip`` (the SB3 model) and ``trike60_vecnormalize_<steps>_steps.pkl`` (its
VecNormalize statistics). The check here separates two kinds of failure that the shared
``environments.shared.curriculum.checkpoints.checkpoint_pair_problem`` does not (it reports both as a problem):

* a STRUCTURAL problem of the files (missing sidecar, a zip that is not a zip, missing SB3 members, a corrupt member,
  a pickle that does not load): the pair is broken and can be skipped, or set aside;
* an I/O ERROR while reading them (``OSError``, e.g. EIO from a Drive mount): says nothing about the files. The scan
  retries after a pause, and if the error persists the caller refuses to act rather than treat good saves as broken.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import stat
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

NAME_PREFIX = "trike60"
CANDIDATE_RE = re.compile(rf"{NAME_PREFIX}_(\d+)_steps\.zip")
NONFINITE_DIR = "nonfinite"  # under checkpoints/: saves whose parameters are not finite (never resume candidates)


def pair_names(steps: int) -> tuple[str, str]:
    return f"{NAME_PREFIX}_{steps}_steps.zip", f"{NAME_PREFIX}_vecnormalize_{steps}_steps.pkl"


def pair_problem(zip_path: Path, pkl_path: Path) -> str | None:
    """``None`` when the pair loads; a structural problem as text; an ``OSError`` (I/O) is RAISED, not returned."""
    zip_path, pkl_path = Path(zip_path), Path(pkl_path)
    try:
        os.lstat(pkl_path)  # not os.path.exists: that answers False on any OSError, an EIO included
    except FileNotFoundError:
        return f"missing matched VecNormalize sidecar {pkl_path.name}"
    try:
        with zipfile.ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            names = archive.namelist()
    except OSError:
        raise
    except Exception as exc:  # BadZipFile, zlib.error, EOFError, ...: the file itself is damaged
        return f"bad/truncated checkpoint zip {zip_path.name} ({exc!r})"
    if bad_member is not None:
        return f"corrupt member {bad_member!r} in {zip_path.name}"
    if "data" not in names or not any(n.endswith("policy.pth") for n in names):
        return f"{zip_path.name} lacks SB3 members (truncated checkpoint)"
    try:
        with open(pkl_path, "rb") as f:
            pickle.load(f)
    except OSError:
        raise
    except Exception as exc:
        return f"VecNormalize sidecar {pkl_path.name} does not unpickle ({exc!r})"
    return None


@dataclass
class Scan:
    pair: tuple[int, Path, Path] | None = None  # (steps, zip, pkl) of the newest intact pair
    skipped: list[str] = field(default_factory=list)  # "<zip name>: <problem>", newest first
    candidates: list[str] = field(default_factory=list)  # every candidate zip name found
    io_error: str | None = None  # set when reading kept failing: nothing may be concluded


def _scan_once(ckpt_dir: Path) -> Scan:
    scan = Scan()
    try:
        st = os.stat(ckpt_dir)  # not os.path.isdir: that answers False on any OSError, an EIO included
    except FileNotFoundError:
        return scan  # really no folder (a new run)
    if not stat.S_ISDIR(st.st_mode):
        raise NotADirectoryError(f"{ckpt_dir} is not a folder")
    found = []
    for name in os.listdir(ckpt_dir):  # listdir, not glob: glob drops listing errors silently
        m = CANDIDATE_RE.fullmatch(name)
        if m:
            found.append((int(m.group(1)), name))
    for steps, name in sorted(found, reverse=True):
        scan.candidates.append(name)
        zip_name, pkl_name = pair_names(steps)
        problem = pair_problem(ckpt_dir / zip_name, ckpt_dir / pkl_name)
        if problem is None:
            scan.pair = (steps, ckpt_dir / zip_name, ckpt_dir / pkl_name)
            return scan
        scan.skipped.append(f"{zip_name}: {problem}")
    return scan


def scan_pairs(ckpt_dir: Path, *, retries: int = 3, wait_s: float = 20.0, log: Callable[[str], None] = print,
               sleep: Callable[[float], None] = time.sleep) -> Scan:
    """The newest intact pair, walking newest first. An I/O error while listing or reading makes the whole scan
    start again after ``wait_s`` seconds, up to ``retries`` more times; if it persists, ``io_error`` is set and
    ``pair``/``skipped`` must not be used."""
    ckpt_dir = Path(ckpt_dir)
    last = None
    for attempt in range(retries + 1):
        try:
            return _scan_once(ckpt_dir)
        except OSError as exc:
            last = exc
            if attempt < retries:
                log(f"WARNING: reading the checkpoints failed ({exc!r}); trying again in {wait_s:.0f} s")
                sleep(wait_s)
    return Scan(io_error=repr(last))


def quarantine_broken(ckpt_dir: Path) -> tuple[Path, list[str]]:
    """Move every trike60_* file of ``ckpt_dir`` into ``ckpt_dir/broken_<time>/`` (kept, never deleted)."""
    qdir = ckpt_dir / f"broken_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    qdir.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in sorted(os.listdir(ckpt_dir)):
        p = ckpt_dir / name
        if name.startswith(f"{NAME_PREFIX}_") and p.is_file():
            shutil.move(str(p), str(qdir / name))
            moved.append(name)
    return qdir, moved


def files_present(ckpt_dir: Path) -> set[str]:
    """Names of every file in ``ckpt_dir`` and its subfolders (broken_*, nonfinite). A folder that does not exist
    holds nothing; any other error while listing is RAISED (os.walk would skip it silently)."""
    out: set[str] = set()

    def fail(exc: OSError) -> None:
        if not (isinstance(exc, FileNotFoundError) and Path(exc.filename or "") == Path(ckpt_dir)):
            raise exc

    for root, _, files in os.walk(ckpt_dir, onerror=fail):
        out.update(files)
    return out


def params_finite(model, vecnorm) -> bool:
    """Every policy parameter and the observation statistics are finite."""
    import numpy as np
    import torch

    with torch.no_grad():
        if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
            return False
    rms = getattr(vecnorm, "obs_rms", None)
    return rms is None or bool(np.isfinite(rms.mean).all() and np.isfinite(rms.var).all())


def _write_atomic(path: Path, write: Callable) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_pair(model, vecnorm, ckpt_dir: Path) -> tuple[str, bool]:
    """Save the model and its VecNormalize statistics, each to ``<name>.tmp``, flushed, then renamed (the
    statistics first, so a zip under its final name always has its sidecar). Returns (path relative to
    ``ckpt_dir``, is a resume candidate). Non-finite parameters go to ``nonfinite/`` instead: never resumed from."""
    steps = int(model.num_timesteps)
    zip_name, pkl_name = pair_names(steps)
    finite = params_finite(model, vecnorm)
    folder = ckpt_dir if finite else ckpt_dir / NONFINITE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    _write_atomic(folder / pkl_name, lambda f: pickle.dump(vecnorm, f))  # what VecNormalize.save does
    _write_atomic(folder / zip_name, lambda f: model.save(f))
    return (zip_name if finite else f"{NONFINITE_DIR}/{zip_name}"), finite


def write_json_atomic(path: Path, obj) -> None:
    """JSON to a temporary name in the same folder, flushed to disk, then renamed over ``path``."""
    path = Path(path)
    _write_atomic(path, lambda f: f.write(json.dumps(obj, indent=1, default=str).encode("utf-8")))


def remove_tmp_files(ckpt_dir: Path) -> list[str]:
    gone = []
    try:
        names = sorted(os.listdir(ckpt_dir))
    except FileNotFoundError:
        return gone
    for name in names:
        if name.endswith(".tmp"):
            (ckpt_dir / name).unlink(missing_ok=True)
            gone.append(name)
    return gone
