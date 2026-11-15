"""Stage 2 shared I/O: what every round-A module needs to write a record that can be trusted later.

* ``require_one_thread``: the three BLAS thread variables must each be exactly ``"1"`` (an inherited
  ``OMP_NUM_THREADS=4`` is refused, not silently kept; arrays depend on the thread count through the BLAS).
* ``r2``: the accuracy level of ``designlab.gait`` is process state; an array-producing item runs inside ``with
  r2():``, which sets level r2, checks it on entry and on exit and restores the level it found.
* ``canonical`` / ``sha256_json``: one byte string per JSON value (sorted keys, no spaces), so that a hash of a
  record's part does not depend on key order, on numpy scalar types or on a round trip through a file.  Numbers
  that are not finite (the ``nan`` period of a stand in Stage 1's ``transitions.json``) are hashed as the strings
  ``"NaN"``, ``"Infinity"`` and ``"-Infinity"`` (``NONFINITE``): strict JSON has no such number.
* ``write_json`` / ``write_npz``: atomic (a temporary ``<name>.tmp`` beside the target, flush, fsync,
  ``os.replace``).  Arrays go through an open file object: ``np.savez(path)`` would append ``.npz`` to the
  temporary name.  The bytes are those ``np.savez(path, **arrays)`` writes.
* ``sweep_tmp``: remove ``*.tmp`` files a killed writer left (call it holding the directory's lock).
* ``Lock``: one ``.lock`` per set or run directory (``O_CREAT | O_EXCL``).
* ``portable``: absolute paths under the study directory made relative to it (``panels._portable``'s rule).
* ``built_by``: provenance of an item (commit, dirty paths less the caller's own outputs, the panels' source hash,
  the sha256 of the named Stage 2 modules, library versions, thread count, accuracy level, a UTC time).

This module is in no record key.  It imports ``designlab.gait`` and ``designlab.stage1.panels`` only inside the
functions that need them (``r2``, ``built_by``), so importing it loads neither MuJoCo nor Stage 1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: The study directory (``hardware/trex_form_study``): every default path is anchored here, never at the working
#: directory.
STUDY = Path(__file__).resolve().parents[2]
#: The variables ``require_one_thread`` checks.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
#: How ``canonical`` writes a float that is not finite.
NONFINITE = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}
ACCURACY = "r2"


class S2Error(RuntimeError):
    """A Stage 2 precondition does not hold (threads, accuracy level, a held lock, an array that cannot be saved)."""


# ------------------------------------------------------------------------------------------------
# Process state
# ------------------------------------------------------------------------------------------------
def require_one_thread() -> int:
    """1 if each of ``THREAD_VARS`` is exactly ``"1"``; ``S2Error`` otherwise (unset counts as not 1)."""
    bad = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != "1"}
    if bad:
        raise S2Error("one BLAS thread is required: " + ", ".join(f"{v}={x!r}" for v, x in bad.items())
                      + " (set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 before Python starts)")
    return 1


def blas_threads() -> int | None:
    """The thread count the environment asks for: 1 when ``require_one_thread`` would pass, else the largest of the
    three variables, or None when one is unset or not a number (the library's default then applies)."""
    vals = [os.environ.get(v) for v in THREAD_VARS]
    if all(x == "1" for x in vals):
        return 1
    try:
        return max(int(x) for x in vals)
    except (TypeError, ValueError):
        return None


@contextmanager
def r2():
    """``gait.set_accuracy("r2")`` for the block.  The level is checked on entry and again on exit (``S2Error`` if
    something inside left another level behind: its arrays cannot be trusted), and the level found before the block
    is restored in every case."""
    from designlab import gait

    prev = gait.ACCURACY
    gait.set_accuracy(ACCURACY)
    try:
        if gait.ACCURACY != ACCURACY:
            raise S2Error(f"gait.set_accuracy({ACCURACY!r}) left the level at {gait.ACCURACY!r}")
        yield
        if gait.ACCURACY != ACCURACY:
            raise S2Error(f"the accuracy level changed inside an r2() block: it is {gait.ACCURACY!r} at the exit, "
                          f"so what the block computed after the change is not at {ACCURACY}")
    finally:
        gait.set_accuracy(prev)


# ------------------------------------------------------------------------------------------------
# Canonical JSON and hashes
# ------------------------------------------------------------------------------------------------
def plain(obj, nonfinite: bool = False):
    """``obj`` as plain JSON types: dicts with string keys (an int key becomes its decimal string, as ``json``
    writes it), lists (tuples and arrays too), str, bool, int, float, None; numpy scalars as Python's; a ``Path`` as
    its string.  With ``nonfinite`` a float that is not finite becomes its ``NONFINITE`` string.  Anything else (a
    set, an object) raises ``TypeError``."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if nonfinite and not math.isfinite(x):
            return NONFINITE["nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")]
        return x
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, (bool, np.bool_)) or not isinstance(k, (str, int, np.integer)):
                raise TypeError(f"a JSON key must be a string or an int, not {type(k).__name__} ({k!r})")
            key = k if isinstance(k, str) else str(int(k))
            if key in out:
                raise TypeError(f"two keys become {key!r}")
            out[key] = plain(v, nonfinite)
        return out
    if isinstance(obj, (list, tuple)):
        return [plain(v, nonfinite) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist(), nonfinite)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not a JSON value ({obj!r})")


def canonical(obj) -> bytes:
    """The canonical bytes of a JSON value: ``plain`` (non-finite floats as their ``NONFINITE`` strings), keys
    sorted, no spaces, ASCII.  Equal values give equal bytes whatever their key order or number types; ``1`` and
    ``1.0`` differ, as they do in a JSON file."""
    return json.dumps(plain(obj, nonfinite=True), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_json(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# Atomic writes
# ------------------------------------------------------------------------------------------------
def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _replace(tmp: Path, path: Path) -> None:
    """``os.replace``; the temporary file is removed if it fails (the target is then as it was)."""
    try:
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj) -> None:
    """``obj`` (through ``plain``) as JSON with ``indent=1``, Stage 1's layout, written to ``<path>.tmp``, flushed,
    fsynced and moved into place: a reader sees the old file or the new one, never a part.  A non-finite float is
    written as Python's ``json`` writes it (``NaN``), as Stage 1's records hold it."""
    path = Path(path)
    text = json.dumps(plain(obj), indent=1)  # before any file is touched: a value that cannot be written changes nothing
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)


def write_npz(path, arrays: dict, compressed: bool = False) -> str:
    """``np.savez(fh, **arrays)`` on ``<path>.tmp`` opened for writing, flushed, fsynced and moved into place; the
    file's sha256.  The bytes equal ``np.savez(path, **arrays)``'s (the order of ``arrays`` is the archive's
    order).  An object array is refused: it would be pickled."""
    path = Path(path)
    for name, a in arrays.items():
        if np.asanyarray(a).dtype.hasobject:
            raise S2Error(f"{path.name}: array {name!r} has an object dtype (it would be pickled)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    try:
        with open(tmp, "wb") as fh:
            (np.savez_compressed if compressed else np.savez)(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        sha = sha256_file(tmp)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _replace(tmp, path)
    return sha


def sweep_tmp(root) -> list[str]:
    """Remove every ``*.tmp`` file below ``root`` (what a killed writer left); their paths relative to ``root``.
    Call it holding the directory's ``Lock``: a live writer's temporary file looks the same."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob("*.tmp")):
        if p.is_file() and not p.is_symlink():
            p.unlink()
            out.append(p.relative_to(root).as_posix())
    return out


# ------------------------------------------------------------------------------------------------
# Locks
# ------------------------------------------------------------------------------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _proc_start(pid: int) -> int | None:
    """The process's start time in clock ticks since boot (``/proc/<pid>/stat`` field 22), None where there is no
    such file: it tells a holder from another process that later got the same pid."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return int(stat[stat.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lock:
    """``with Lock(directory / ".lock"): ...``: one holder per directory.

    The file is created with ``O_CREAT | O_EXCL | O_WRONLY`` and holds JSON ``{pid, host, started_at, boot_id,
    proc_start}``.  If it exists and its holder is gone, the lock is taken over and a line is logged; otherwise
    ``S2Error`` names the holder.  The holder is gone when the file names this host and its pid is not alive here,
    or the pid is alive but belongs to another process (another start time, or the machine has booted since: after
    a restart a new process can carry the old pid).  A lock of another host, or a file that cannot be read, is
    never taken over.  ``release`` removes the file only while it still names this holder."""

    def __init__(self, path, log=None):
        self.path = Path(path)
        self.log = log if log is not None else (lambda line: print(line, file=sys.stderr))
        self.info: dict | None = None
        self.took_over: dict | None = None

    # -- the holder a lock file names
    def holder(self) -> dict | None:
        try:
            doc = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) and "pid" in doc and "host" in doc else None

    @staticmethod
    def stale(holder: dict) -> str | None:
        """Why the holder is gone (a sentence), or None if it may be alive."""
        if holder.get("host") != socket.gethostname():
            return None
        pid = holder.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            return None
        boot = _boot_id()
        if holder.get("boot_id") and boot and holder["boot_id"] != boot:
            return "this machine has booted since"
        if not _alive(pid):
            return f"pid {pid} is not alive"
        start = _proc_start(pid)
        if holder.get("proc_start") is not None and start is not None and holder["proc_start"] != start:
            return f"pid {pid} now belongs to another process"
        return None

    def acquire(self) -> "Lock":
        if self.info is not None:
            raise S2Error(f"{self.path} is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        info = dict(pid=os.getpid(), host=socket.gethostname(), started_at=now_utc(), boot_id=_boot_id(),
                    proc_start=_proc_start(os.getpid()))
        for attempt in (0, 1):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                holder = self.holder()
                if holder is None:
                    raise S2Error(f"{self.path} exists and names no holder (unreadable): check that nothing runs "
                                  "there, then remove it") from None
                why = self.stale(holder) if attempt == 0 else None
                if why is None:
                    raise S2Error(f"{self.path} is held by pid {holder.get('pid')} on {holder.get('host')} since "
                                  f"{holder.get('started_at')}") from None
                if self.holder() == holder:  # still that holder's file: remove it and try once more
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                self.took_over = dict(holder, reason=why)
                self.log(f"lock {self.path}: taken over from pid {holder.get('pid')} on {holder.get('host')} "
                         f"(started {holder.get('started_at')}): {why}")
                continue
            with os.fdopen(fd, "w") as fh:
                json.dump(info, fh)
                fh.flush()
                os.fsync(fh.fileno())
            self.info = info
            return self
        raise S2Error(f"{self.path}: could not be taken")  # (not reached: the second attempt returns or raises)

    def release(self) -> None:
        info, self.info = self.info, None
        if info is not None and self.holder() == info:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "Lock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


# ------------------------------------------------------------------------------------------------
# Paths and provenance
# ------------------------------------------------------------------------------------------------
def portable(x):
    """``x`` with every absolute path under the study directory made relative to it, in dicts (keys too), lists and
    tuples (returned as lists), as ``panels._portable`` does; a ``Path`` is treated as its string.  A key or a
    description then does not depend on where the checkout is."""
    if isinstance(x, dict):
        return {(portable(k) if isinstance(k, (str, Path)) else k): portable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [portable(v) for v in x]
    if isinstance(x, Path):
        x = str(x)
    if isinstance(x, str) and x.startswith(str(STUDY) + os.sep):
        return Path(x).relative_to(STUDY).as_posix()
    return x


def module_path(name) -> Path:
    """The file of a Stage 2 module given as ``"moves.py"`` (in ``designlab/stage2``), as a path relative to the
    study directory, or as an absolute path."""
    p = Path(name)
    if p.is_absolute():
        return p
    return STUDY / "designlab" / "stage2" / p if len(p.parts) == 1 else STUDY / p


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(STUDY), *args], capture_output=True, text=True, check=True).stdout


def _status_paths(line: str) -> list[str]:
    """The path or paths (a rename has two) of a ``git status --porcelain`` line, from the repository root."""
    out = []
    for part in line[3:].split(" -> "):
        part = part.strip()
        if len(part) >= 2 and part[0] == '"' and part[-1] == '"':
            part = part[1:-1]
        out.append(part)
    return out


def git_state(own_outputs=()) -> dict:
    """``{commit, dirty, dirty_paths}`` of the checkout, limited to the study directory: ``git rev-parse HEAD`` and
    the lines of ``git status --porcelain --untracked-files=all`` (status code and path from the repository root),
    less the lines whose every path is one of ``own_outputs`` or lies below one (paths relative to the study
    directory, or absolute: what the caller is itself writing).  All None where git cannot tell."""
    out = dict(commit=None, dirty=None, dirty_paths=None)
    try:
        out["commit"] = _git("rev-parse", "HEAD").strip()
        top = Path(_git("rev-parse", "--show-toplevel").strip())
        lines = [x for x in _git("status", "--porcelain", "--untracked-files=all", "--", ".").splitlines() if x.strip()]
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None, dirty_paths=None)
    own = [(Path(o) if Path(o).is_absolute() else STUDY / o) for o in own_outputs]
    own = [Path(os.path.normpath(o)) for o in own]

    def is_own(rel: str) -> bool:
        p = Path(os.path.normpath(top / rel))
        return any(p == o or o in p.parents for o in own)

    out["dirty_paths"] = [x for x in lines if not all(is_own(p) for p in _status_paths(x))]
    out["dirty"] = bool(out["dirty_paths"])
    return out


def built_by(stage2_modules: list[str], own_outputs: list[str] = ()) -> dict:
    """Provenance of what the caller is building: ``{commit, dirty, dirty_paths, panel_sources_sha256,
    stage2_modules {path: sha256}, versions, blas_threads, accuracy, at}``.  ``stage2_modules`` names the Stage 2
    files that made it (``module_path``; recorded by their path relative to the study directory); ``own_outputs``
    are left out of ``dirty_paths`` and of ``dirty`` (``git_state``).  ``accuracy`` is ``gait.ACCURACY`` at the call:
    call it inside the item's ``r2()`` block.  Provenance is recorded once and never recomputed; it is in no key."""
    from designlab import gait
    from designlab.stage1 import panels

    mods = {}
    for name in stage2_modules:
        p = module_path(name)
        mods[portable(str(p))] = sha256_file(p)
    return dict(git_state(own_outputs), panel_sources_sha256=panels.panel_sources()["sha256"], stage2_modules=mods,
                versions=panels.library_versions(), blas_threads=blas_threads(), accuracy=gait.ACCURACY, at=now_utc())
