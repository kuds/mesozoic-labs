"""Track B's notebook steps, the queue launcher, the collection of finished records and the tally of runtime-hours
(round C design D19, D22, D24, D25; sections 3.9, 3.10, 3.12; Track B rules K2, K4-K7, N1-N5, Y1-Y11, X1).
Simulation tooling only: nothing here is a result.

**Standard library only at import.**  This module runs before the pinned packages are trusted (its ``setup`` step
installs them) and is imported by the notebook's kernel for ``release_runtime``.  It reaches the trainer, the
bundle tool, the loader, the clip tool and everything that needs numpy, torch or MuJoCo by subprocess only
(``[sys.executable, "-m", "designlab.stage2.<module>", ...]`` with ``tbenv.child_env(...)`` and the study directory
as working directory).  ``g8a``, ``r5env`` and ``setio`` have no command line of their own: they run in this
module's ``child`` entry, a subprocess whose functions import them (never this module's top).

    python -m designlab.stage2.nb {setup|session|inputs|preflight|smoke|launch|finish} --params FILE
                                  [--set-param KEY=JSON ...]
    python -m designlab.stage2.nb collect --src DIR --dst DIR --kind run|bar|r2l
    python -m designlab.stage2.nb hours --runs-root DIR [--out stage2/tb/hours.json]
    python -m designlab.stage2.nb local-run --drive DIR [--notebook stage2/stage2_training.ipynb] [--params FILE]
                                  [--record PATH]
    python -m designlab.stage2.nb acceptance --ref <40 hex> --parts DIR --scratch DIR --record PATH [--notebook PATH]
    python -m designlab.stage2.nb child {read-inputs|archive-preflight|envs|g8a} ...     # this module's own children

**The owner's path** (``stage2/stage2_training.ipynb``): each code cell runs one step in a subprocess and prints its
lines.  ``setup`` (cell 2, after the cell's own Git block): the pins of ``stage2/tb/requirements-colab.txt`` and
line 1.  ``session`` (cell 3): rule K6's and K7's refusals, ``QUEUE.json`` (a new run), ``SESSION_<k>.json``,
``LIVE.json``, the run-directory line.  ``inputs`` (cell 4): the bundle's parts verified and unpacked, line 2.
``preflight`` (cell 5): the archive-load preflight, a policy built, every env of the queue built, G8a's short form
per body, line 3.  ``smoke`` (cell 6, only with ``SMOKE``): rules Y1-Y8.  ``launch`` (cell 7): the queue, one job at
a time, line 4 per job.  ``finish`` (cell 8), then ``release_runtime`` in the kernel.

**Parameters** (``PARAMS``): the fifteen of the notebook's cell 1 and, for local runs and tests only, the keys of
``EXTRA`` (the set, ``TINY``, the two directories, the monitor options, the launcher's period).  The notebook writes
them to one JSON file that every step reads.

**The four lines** (rule N1) are formatted by ``line1`` .. ``line4`` and matched by ``LINE_RE``; the session line's
three forms by ``session_line``; the smoke's lines by ``smoke_line``.  Nothing else prints them.

**Files of a run** (design 3.9; ``<drive>/runs/<run_id>/``): ``QUEUE.json`` (the queue, set, budget, bundle,
REPO_REF and commit as first given, the trainer's options), ``SESSION_<k>.json``, ``LIVE.json``, ``STATUS.json``
(rewritten every period), ``DONE.json`` or ``HALT.json``, ``logs/``, ``jobs/<job>/`` (the trainer's directory with
the copied checkpoints), ``smoke/``.  The trainer writes under ``<local>/<run_id>/``; with ``drive == local`` it
writes into the run directory itself and nothing is copied.  The launcher's own state is ``work/`` under the local
run directory (never collected, never committed).

**One writer** (rule K7, ``FENCE_RULE``).  A session owns the run while ``LIVE.json`` names its uuid.  The launcher
re-reads it before every copy or delete on Drive; when it names another session the launcher ends its job's
process group and stops, and from then on writes nothing into the run directory but its own log and one file
named after its own session (``SUPERSEDED_<k>.json``): the run's ``HALT.json`` on Drive is left to the session
that owns the run (``HALT_RULE``).

**A fault of Drive is no halt** (``COPY_RULE``'s end, ``BACK_RULE``, ``UNFINISHED_RULE``).  A copy to Drive that
fails is counted and made again in the next period.  A job that has ended is finished on Drive only when its last
checkpoint is there: until then its ``run.json`` is held back, the copy is tried again, and the run gets no
``DONE.json``.  A job's directory is copied back from Drive all or nothing, and a job is never started while Drive
holds a newer checkpoint of it than this machine.  Where Drive cannot be had, the launch ends ``"unfinished"``
(exit code 6): no ``DONE.json``, no ``HALT.json`` (that file ends a run), ``LIVE.json`` closed, the run open to a
resume.  ``finish`` says whether every finished job's last checkpoint is on Drive (exit code 6 when one is not).

**Seams** (``_trainer_cmd``, ``_clip_cmd``, ``_pack_cmd``, ``_bundle_cmd``, ``_child_cmd``, ``_rebuild_cmd``,
``ROOT``): the places where this module meets the other tools; the tests replace them with stubs.  ``S2_NB_SEAMS``
is refused on Colab.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # round A's module skeleton: before anything of this process could import numpy
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import errno  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from designlab.stage2 import tbenv  # noqa: E402  (standard library only)

SESSION_SCHEMA = "s2-session-2"
STATUS_SCHEMA = "s2-status-2"
QUEUE_SCHEMA = "s2-queue-1"
LIVE_SCHEMA = "s2-live-1"
END_SCHEMA = "s2-end-1"
SETUP_SCHEMA = "s2-setup-1"
COLLECTED_SCHEMA = "s2-collected-1"
HOURS_SCHEMA = "s2-hours-1"
SMOKE_SCHEMA = "s2-smoke-1"
LOCAL_RUN_SCHEMA = "s2-local-run-1"
ACCEPTANCE_SCHEMA = "s2-acceptance-1"

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_HALT, EXIT_INTERRUPTED = 0, 1, 4, 5, 130
#: ``launch``: the session ended with no DONE.json and no HALT.json because something could not be read from or
#: written to Drive (``UNFINISHED_RULE``); ``finish``: a finished job's last checkpoint is not on Drive.
EXIT_UNFINISHED = 6
#: The trainer's exit codes (``train2``): done or stopped by a rule; G8a failed; refused; halted for the lead.
TRAIN_DONE, TRAIN_G8A, TRAIN_REFUSED, TRAIN_HALT = 0, 3, 4, 5

#: The launcher's period (design 3.9: 60 s); ``PERIOD_S`` of the parameters replaces it in local runs and tests.
PERIOD_S = 60.0
#: Rule K7: a ``LIVE.json`` younger than this refuses a RESUME.
LIVE_FRESH_S = 600.0
#: Design 3.9: a clip child's limit.
CLIP_LIMIT_S = 300.0
PACK_LIMIT_S = 120.0
#: How often ``s2load pack`` is tried for one check's tar (a failed copy of a packed tar to Drive is not a try).
PACK_TRIES = 3
#: ``COPY_RULE``'s end: how many further periods a finished job's missing copies are tried again at its end.
END_COPY_TRIES = 10
#: How often a job's directory is copied back from Drive before the session ends unfinished.
COPY_BACK_TRIES = 3
#: Rule K2: the checkpoints copied to Drive are those at multiples of this many steps, and each check's.
COPY_EVERY = 1_000_000
N_ENVS = 8
MAX_WORKERS = 8
#: Rule N1: line 4's floor is the rate at which the budget takes this long (22 h).
FLOOR_SECONDS = 79_200
#: Rule Y11: the budget from the smoke's rate r: 40M if r x 72,000 >= 40M, else r x 72,000 rounded down to a
#: multiple of 5M; under 15M rule Y4 fails.
BUDGET_SECONDS, BUDGET_FULL, BUDGET_STEP, BUDGET_MIN = 72_000, 40_000_000, 5_000_000, 15_000_000
#: Rule Y8: the rebuild's cap.
REBUILD_CAP_S = 2400.0
#: Rule Y3: the trainer's and the workers' resident memory must stay under this share of the RAM.
RAM_SHARE = 0.8
HOURS_CAP = 220.0
PYTHON_MINOR = (3, 13)
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PINS_FILE = "stage2/tb/requirements-colab.txt"
BUNDLE_DIR = "stage2/tb/bundle"
NOTEBOOK = "stage2/stage2_training.ipynb"
S1_SET = "S1"
S1_BUNDLE = "S1-view"
SMOKE_BODY = "75-M-H-10"
#: Rules Y1-Y8: the item names of the smoke's lines (the steps sheet quotes them).
SMOKE_ITEMS = {1: "setup", 2: "machine", 3: "memory", 4: "rate at 8 envs", 5: "test clip",
               6: "kill and resume on Drive", 7: "setup time", 8: "inputs rebuilt once"}
#: The cells ``local_run`` and ``acceptance`` execute (design 3.10 step 3: 1-5 and 7-8; cell 6 is the smoke's).
RUN_CELLS = (1, 2, 3, 4, 5, 7, 8)

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

#: The notebook's cell 1 (design 3.9): name -> default.
PARAMS = dict(REPO_REF="", RUN_ID="", QUEUE=[], BUNDLE_ID="", BUNDLE_SOURCE="drive", RESUME=False, TAKEOVER=False,
              SMOKE=False, BUDGET_STEPS=0, WORKERS=0, CLIPS=True, ALLOW_VERSION_CHANGE=False,
              ALLOW_SOURCE_CHANGE=False, USE_GOOGLE_DRIVE=True, AUTO_DISCONNECT=True)
#: What an ``S2_PARAMS`` file may hold besides (local runs, tests, the lead's own launches), and what the notebook
#: adds on its way (``IN_COLAB``, ``SETUP_STARTED``, ``SETUP_RECORD``, the two directories).
EXTRA = dict(SET="V6", TINY=False, DRIVE_DIR=None, LOCAL_DIR=None, MONITOR_EVERY=None, MONITOR_ALSO_K=[],
             PERIOD_S=None, BUDGET_BY=None, BUDGET_SET_ON=None, PREFLIGHT_BODIES=None, RETRY_CHECK=False, LR=None,
             CONFIG=None, IN_COLAB=False, SETUP_STARTED=None, SETUP_RECORD=None, STUDY_DIR=None, COPY_PAUSE_S=None)

#: Where the inputs are verified and unpacked: None is the study directory (a test's value is a scratch tree).
ROOT = None

K6_RULE = (
    "rule K6: a run keeps one RUN_ID, one bundle, one queue, one budget and one commit for its whole life.  A "
    "session is refused when its REPO_REF is not the 40 hex digits of the checkout's commit (a smoke may name a "
    "branch or a tag); when its RUN_ID has a run directory with QUEUE.json and RESUME is not set; when RESUME is "
    "set and there is no such run, or the run is a smoke; when RESUME comes with another queue, set, bundle or "
    "budget (or other trainer options) than QUEUE.json holds; when the checkout's commit is not QUEUE.json's, "
    "unless ALLOW_SOURCE_CHANGE is set (recorded and noted).  On a resume REPO_REF is QUEUE.json's")
K7_RULE = (
    f"rule K7: a session with RESUME is refused while LIVE.json is younger than {LIVE_FRESH_S:g} s and was not "
    "closed by the session that wrote it, unless TAKEOVER is set (recorded); a run that has DONE.json or HALT.json "
    "is finished and is not resumed")
FENCE_RULE = (
    "rule K7: the launcher writes LIVE.json (its session's uuid, the host, the time) with STATUS.json every period "
    "and re-reads it before every copy or delete on Drive; when it names another session the launcher ends its "
    "job's process group and stops (\"superseded by session <k>\")")
HALT_RULE = (
    "a launcher that was superseded writes HALT.json in its local run directory and SUPERSEDED_<k>.json, named "
    "after its own session, in the run directory; it never writes the run's HALT.json on Drive: that file ends a "
    "run (rule K7: a run with HALT.json is not resumed), and the run now belongs to the session that took it over")
COPY_RULE = (
    f"rule K2: to Drive the launcher copies a job's checkpoints at multiples of {COPY_EVERY:,} steps (the first "
    "rollout start at or past each) and each check's (one with a monitor record, or at a multiple of "
    "monitor_every, or the run's last); in a tiny run every checkpoint; never the checkpoint of step 0 (a job cut "
    "before its first later checkpoint restarts from step 0 in place and is the same run, rule K5).  Each "
    "checkpoint goes as a whole: the three binaries each as <name>.part and renamed, then checkpoint_<n>.json the "
    "same way, last.  It is called copied when all four files read back through the mount with the record's "
    "sha256 (copied, not verified: the mount reads its own cache).  Drive keeps each check's and the newest two "
    "of the copied ones; an older one is deleted (the record first) only after a newer one was copied in an "
    "earlier cycle and reads back again in this one.  For each check's checkpoint ckpt_<n>.tar is written by "
    "s2load pack and copied (it is there when the copy has the packed tar's sha256).  Text records (*.json, "
    "*.jsonl, *.txt outside checkpoints/ and traces/; the logs; a clip with its record) are copied whenever they "
    "changed, each as <name>.part and renamed.  A job that has ended is finished on Drive only when its last "
    "checkpoint and each check's read back from there and each check's tar that could be packed is there: until "
    "then its run.json and progress.json are not copied (on Drive a job never reads finished before its last "
    f"checkpoint is there), the missing copies are tried again every period, for {END_COPY_TRIES} further periods "
    "at the job's end and then while later jobs run, and the run gets no DONE.json (UNFINISHED_RULE)")
UNFINISHED_RULE = (
    "a launch ends unfinished, with no DONE.json and no HALT.json, LIVE.json closed and the run open to a resume, "
    "when what it needs of Drive cannot be had: a finished job's last checkpoint (or a check's, or a check's tar) "
    "is not on Drive after the tries of the copy rule; a job's directory cannot be copied back from Drive in "
    f"{COPY_BACK_TRIES} tries; Drive holds a checkpoint of a job that is newer than every one in this machine's "
    "directory of it (a resume there would restart the job from step 0 or from an older step, silently); the "
    "run's last files cannot be written.  HALT.json is not written for a fault of Drive: that file ends a run")
BACK_RULE = (
    "the copy back (design 3.9): before a job is started, its directory is copied from Drive when this machine "
    "holds no run.json of it or when Drive holds a checkpoint that reads back whole and is newer than every whole "
    "one here.  The copy is all or nothing: it is made beside the job's place, every checkpoint that reads back on "
    "Drive must read back in the copy, and only then it is moved into place (a directory that was there is kept "
    "aside under work/aside/, never deleted).  A job is never started while Drive holds a checkpoint newer than "
    "every one here (UNFINISHED_RULE)")
JOB_RULE = (
    "job rules (design 3.9; the trainer's exit codes): a job that ends with exit code 0 (done, or stopped by rule "
    "M4 or M5) or 3 (G8a failed: no run) lets the next job start; exit code 4 (refused: nothing was run) or 5 "
    "(halted for the lead) halts the run without a retry; any other code or a signal is a crash: the job's "
    "process group is ended and the job is started once more with --resume (the trainer chooses its newest "
    "intact checkpoint, or restarts from step 0 in place), and a second crash halts the run.  The count of "
    "retries is the job's, across sessions.  Finished jobs are skipped on a resume; on a new runtime the job's "
    "directory is copied back from Drive first")
CLIP_RULE = (
    f"clips (design 3.9): one clip child at a time, started when a check has a record and traces and no clip yet; "
    f"the newest such check; its walk/v0.4 episode of seed 6004 when there is one; limit {CLIP_LIMIT_S:g} s; a "
    "failure is recorded and ignored; the trainer never waits for a clip")
LINE4_RULE = (
    f"rule N1, line 4: the session's first_three rate of gates/rate.json (env steps per wall second over its first "
    f"three rollouts; the session is the one of the trainer that this launcher started, by the pid in "
    f"progress.json), the budget in millions of steps and the hours the budget takes at that rate; BELOW FLOOR "
    f"<f> when the rate is under budget / {FLOOR_SECONDS:,} (the rate at which the budget takes 22 h)")
BUDGET_RULE = (
    f"rule Y11: the budget is {BUDGET_FULL:,} env steps if r x {BUDGET_SECONDS:,} >= {BUDGET_FULL:,}, else r x "
    f"{BUDGET_SECONDS:,} rounded down to a multiple of {BUDGET_STEP:,}; under {BUDGET_MIN:,} rule Y4 fails and no "
    "job is queued until the owner has set the budget")
COLLECT_RULE = (
    "design D22: collect copies a finished job's text records (*.json, *.jsonl, *.txt; nothing under partial/, "
    "traces/, logs/ or work/; of checkpoints/ only checkpoint_<n>.json; never LIVE.json) to their committed "
    "place, verifies each copy by sha256 and writes COLLECTED.json; a source that is still running is refused "
    "(a LIVE.json that is fresh and not closed without DONE.json or HALT.json; a lock whose holder is alive; a "
    "run.json that reads running whose process is alive); a file already there with other bytes is refused")
HOURS_RULE = (
    f"runtime-hours (design 3.9): per run, each session from started_at to last_status_at, summed, against the cap "
    f"of {HOURS_CAP:g} h; a session that never wrote a status counts nothing and is listed")


class NbError(RuntimeError):
    """A step cannot go on.  ``code`` is the exit code of the command line."""

    code = EXIT_FAILED


class Refused(NbError):
    """A step the rules refuse: nothing was changed.  ``line`` is True when the message is itself the line to
    print (the session line's third form)."""

    code = EXIT_REFUSED

    def __init__(self, message: str, line: bool = False):
        super().__init__(message)
        self.line = line


class _Superseded(Exception):
    """``LIVE.json`` names another session (rule K7)."""

    def __init__(self, by):
        super().__init__(f"superseded by session {by}")
        self.by = by


# ------------------------------------------------------------------------------------------------
# Small helpers (standard library only; the JSON layout and the hashes are s2io's)
# ------------------------------------------------------------------------------------------------
def _say(*text) -> None:
    """Print a line, and never raise: when nobody reads this process's output any more (the kernel that started
    the step died alone and left a closed pipe) the step goes on, and a launcher's lines are in its log."""
    try:
        print(*text, flush=True)
    except (OSError, ValueError):
        try:  # (the lines that follow, and the interpreter's own flush at its end, go nowhere instead of failing)
            sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
        except OSError:
            pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(text) -> float | None:
    """The epoch seconds of a UTC time written by ``_now`` (None: not one)."""
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def study() -> Path:
    """The study directory (the working directory of every child)."""
    return Path(tbenv.study_dir())


def _root() -> Path:
    return study() if ROOT is None else Path(ROOT)


def _anchored(path) -> Path:
    """A relative path is taken from the study directory, never from the working directory (``setio``'s rule)."""
    p = Path(path)
    return p if p.is_absolute() else study() / p


def _plain(obj):
    """``obj`` as plain JSON types (a ``Path`` as its string, a tuple as a list); a float that is not finite as
    None: no record of this module holds one."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not a JSON value ({obj!r})")


def sha256_json(obj) -> str:
    """``s2io.sha256_json`` for plain finite values: sorted keys, no spaces, ASCII (a test holds the two equal)."""
    text = json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stamped(doc: dict) -> dict:
    """``doc`` with its own ``sha256`` (of everything else in it)."""
    doc = _plain({k: v for k, v in doc.items() if k != "sha256"})
    doc["sha256"] = sha256_json(doc)
    return doc


def _read_json(path):
    """The JSON document of a file; None when it is not there or cannot be read as JSON (a copy that was cut)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _part(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def _fsync(fh) -> None:
    try:
        fh.flush()
        os.fsync(fh.fileno())
    except OSError:  # (a mount that cannot sync: the rename below is still atomic for a reader)
        pass


def _write_json(path, obj) -> None:
    """``obj`` as JSON with ``indent=1`` (``s2io.write_json``'s layout), written beside its place and moved into
    it: a reader sees the old file or the new one, never a part.  The temporary name is this writer's own
    (``<name>.<pid>-<random>.part``): during a takeover two sessions write ``LIVE.json`` in the same seconds."""
    path = Path(path)
    text = json.dumps(_plain(obj), indent=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.part")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            _fsync(fh)
        os.replace(tmp, path)
    except BaseException:
        _unlink(tmp)
        raise


def _unlink(path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _listing(path) -> list | None:
    """The names in a directory, sorted; None when it is not there.  Any other error is raised: a mount that
    cannot be read is not an empty directory (``Path.is_dir`` answers False for both)."""
    try:
        return sorted(os.listdir(path))
    except (FileNotFoundError, NotADirectoryError):
        return None


def _walk_error(exc: OSError):
    """``os.walk``'s ``onerror``: a directory that cannot be listed is an error, not an empty one."""
    raise exc


def sweep_parts(root) -> list:
    """Remove every ``*.part`` file below ``root`` (what a killed copy left); their relative paths, sorted."""
    root, out = Path(root), []
    if not root.is_dir():
        return out
    for p in root.rglob("*.part"):
        if p.is_file() and not p.is_symlink():
            _unlink(p)
            out.append(p.relative_to(root).as_posix())
    return sorted(out)


def _copy_file(src: Path, dst: Path, pause: float = 0.0) -> None:
    """``src`` to ``dst`` through ``<dst>.part`` and a rename.  ``pause``: seconds the part is left standing before
    the rename (``COPY_PAUSE_S``: the acceptance run ends a launcher while a copy is in progress)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _part(dst)
    try:
        with open(src, "rb") as a, open(tmp, "wb") as b:
            shutil.copyfileobj(a, b, 1 << 20)
            _fsync(b)
        if pause:
            time.sleep(pause)
        os.replace(tmp, dst)
    except BaseException:
        _unlink(tmp)
        raise


def _vcpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _proc_stat(pid) -> list | None:
    """The fields of ``/proc/<pid>/stat`` after the command name (state first); None where there is none."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return stat[stat.rindex(")") + 2:].split()
    except (OSError, ValueError, TypeError):
        return None


def _proc_start(pid) -> int | None:
    rest = _proc_stat(pid)
    try:
        return None if rest is None else int(rest[19])
    except (IndexError, ValueError):
        return None


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _alive(pid) -> bool:
    """Whether ``pid`` is a process that has not ended (a zombie has ended)."""
    rest = _proc_stat(pid)
    if rest is not None:
        return rest[0] != "Z"
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def _group_pids(pgid: int) -> list:
    out = []
    for p in Path("/proc").glob("[0-9]*"):
        rest = _proc_stat(p.name)
        if rest is not None and len(rest) > 3 and rest[2] == str(pgid) and rest[0] != "Z":
            out.append(int(p.name))
    return out


def session_pids(sid: int) -> list:
    """The live processes of the session ``sid`` (``/proc``): a launcher's whole tree, its job's group included."""
    out = []
    for p in Path("/proc").glob("[0-9]*"):
        rest = _proc_stat(p.name)
        if rest is not None and len(rest) > 3 and rest[3] == str(sid) and rest[0] != "Z":
            out.append(int(p.name))
    return out


def kill_session(sid: int, wait_s: float = 20.0) -> int:
    """SIGKILL to every process of the session ``sid`` until none is left (what a dead runtime does to a
    launcher, its trainer, the workers and the monitor child at once); how many were signalled."""
    n, t0 = 0, time.time()
    while True:
        pids = [p for p in session_pids(sid) if p != os.getpid()]
        if not pids:
            return n
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                n += 1
            except (ProcessLookupError, PermissionError):
                pass
        if time.time() - t0 > wait_s:
            raise NbError(f"processes of session {sid} are still alive after {wait_s:g} s: {pids}")
        time.sleep(0.1)


def _kill_group(pgid, wait_s: float = 20.0) -> None:
    """SIGKILL to the process group ``pgid`` until it is empty (a job's trainer, workers and monitor child)."""
    if not pgid:
        return
    t0 = time.time()
    while True:
        try:
            os.killpg(int(pgid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
        if not _group_pids(int(pgid)) or time.time() - t0 > wait_s:
            return
        time.sleep(0.05)


def _git(*args, cwd=None) -> str | None:
    """A read-only git command's output in the study's repository; None where git cannot tell.  It takes no lock
    on the index (``GIT_OPTIONAL_LOCKS=0``: a killed ``git status`` leaves no ``index.lock``)."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        done = subprocess.run(["git", "-C", str(cwd or study()), *args], capture_output=True, text=True, env=env,
                              timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def repo_state() -> dict:
    """``{commit, dirty, dirty_paths}`` of the repository the study is in (the whole checkout, untracked files
    included; ignored files are not dirt).  All None where git cannot tell."""
    head, status = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    if head is None or status is None:
        return dict(commit=None, dirty=None, dirty_paths=None)
    lines = [x for x in status.splitlines() if x.strip()]
    return dict(commit=head.strip(), dirty=bool(lines), dirty_paths=lines[:50])


def _ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(float(line.split()[1]) / 1024.0 / 1024.0, 2)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return None


def _gpu() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip().splitlines()[0] if done.returncode == 0 and done.stdout.strip() else None


def _disk_free_gb(where) -> float | None:
    p = Path(where)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return round(shutil.disk_usage(p).free / 1e9, 2)
    except OSError:
        return None


def machine(where=None, in_colab: bool = False) -> dict:
    """``SESSION_<k>.json``'s ``runtime``: what this process runs on."""
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return dict(in_colab=bool(in_colab), vcpus=_vcpus(), affinity=affinity, cpu_model=_cpu_model(),
                ram_gb=_ram_gb(), disk_free_gb=_disk_free_gb(where or study()), gpu=_gpu(),
                host=socket.gethostname())


# ------------------------------------------------------------------------------------------------
# The lines (rule N1; design 3.9): one function formats each, LINE_RE matches each
# ------------------------------------------------------------------------------------------------
def line1(commit, ref_is_commit: bool, clean: bool) -> str:
    return (f"S2 LINE 1/4 commit: {commit} (REPO_REF is a commit: {'yes' if ref_is_commit else 'no'}; "
            f"checkout clean: {'yes' if clean else 'no'})")


def line2(bundle_id, content_sha256, files_ok: int, files: int, problem=None) -> str:
    """Line 2; ``problem`` (the first path or reason) gives the failing form."""
    head = (f"S2 LINE 2/4 inputs: bundle {bundle_id}, manifest {str(content_sha256)[:12] if content_sha256 else 'none'}"
            f", {int(files_ok)} of {int(files)} files match their sha256")
    return head + (": OK" if problem is None else f": FAILED ({_one_line(problem)})")


def line3(body, set_name, identical: int, of: int, steps: int, failure=None) -> str:
    """Line 3; ``failure`` gives the failing form: ``{condition, seed, ms, step}`` (the first episode that is not
    identical) or a reason."""
    head = (f"S2 LINE 3/4 G8a {body} set {set_name}: zero action = R0 at 0 ms and 10 ms, {int(identical)} of "
            f"{int(of)} episodes identical ({int(steps)} physics steps)")
    if failure is None:
        return head + ": PASS"
    if isinstance(failure, dict):
        return head + (f": FAIL ({failure['condition']} seed {int(failure['seed'])} at {int(failure['ms'])} ms, "
                       f"first difference at step {int(failure['step'])})")
    return head + f": FAIL ({_one_line(failure)})"


def floor_rate(budget) -> float:
    """Rule N1: the rate at which ``budget`` steps take 22 h."""
    return float(budget) / FLOOR_SECONDS


def line4(job, rate, n_envs, workers, vcpus, budget) -> str:
    """Line 4 (``LINE4_RULE``), with `` BELOW FLOOR <f>`` when the rate is under the launch's floor."""
    rate, budget = float(rate), int(budget)
    hours = budget / rate / 3600.0 if rate > 0 else float("inf")
    text = (f"S2 LINE 4/4 first rate {job}: {rate:.0f} env steps/s ({int(n_envs)} envs, {int(workers)} workers, "
            f"{int(vcpus)} vCPUs); {budget / 1e6:g}M steps in about {hours:.1f} h")
    return text + (f" BELOW FLOOR {floor_rate(budget):.0f}" if rate < floor_rate(budget) else "")


def session_line(path, kind: str, **kw) -> str:
    """The run-directory line (design 3.9): ``kind`` ``"new"``; ``"resume"`` (``session``, ``job``, ``steps``,
    ``done``); ``"live"`` (``age_s``: the refused resume of a run that is still running)."""
    if kind == "new":
        return f"Run directory: {path} (new run)"
    if kind == "resume":
        return (f"Run directory: {path} (resuming session {int(kw['session'])}: {kw['job']} from {int(kw['steps'])} "
                f"steps; {int(kw['done'])} jobs done)")
    if kind == "live":
        return f"Run directory: {path}: still running (LIVE.json written {int(kw['age_s'])} s ago); nothing to do"
    raise ValueError(f"session line {kind!r}")


def smoke_line(i: int, item: str, value, verdict: str) -> str:
    """``S2 SMOKE Y<i> <item>: <value>: OK | FAILED | REPORTED`` (design 3.9, cell 6)."""
    if verdict not in ("OK", "FAILED", "REPORTED"):
        raise ValueError(f"smoke verdict {verdict!r}")
    return f"S2 SMOKE Y{int(i)} {item}: {_one_line(value)}: {verdict}"


def _one_line(text) -> str:
    return " ".join(str(text).split())


_BODY = r"[A-Za-z0-9][A-Za-z0-9._-]*"
#: The lines' regular expressions (the tests, ``local_run`` and ``acceptance`` read the printed lines with them).
LINE_RE = {
    1: re.compile(r"S2 LINE 1/4 commit: (?P<commit>[0-9a-f]{40}) \(REPO_REF is a commit: (?P<ref>yes|no); "
                  r"checkout clean: (?P<clean>yes|no)\)\Z"),
    2: re.compile(r"S2 LINE 2/4 inputs: bundle (?P<bundle>" + _BODY + r"), manifest (?P<manifest>[0-9a-f]{12}|none)"
                  r", (?P<ok>\d+) of (?P<n>\d+) files match their sha256: (?P<end>OK|FAILED \(.+\))\Z"),
    3: re.compile(r"S2 LINE 3/4 G8a (?P<body>" + _BODY + r") set (?P<set>" + _BODY + r"): zero action = R0 at 0 ms "
                  r"and 10 ms, (?P<identical>\d+) of (?P<of>\d+) episodes identical \((?P<steps>\d+) physics steps\)"
                  r": (?P<end>PASS|FAIL \(.+\))\Z"),
    4: re.compile(r"S2 LINE 4/4 first rate (?P<job>" + _BODY + r"): (?P<rate>\d+) env steps/s \((?P<envs>\d+) envs, "
                  r"(?P<workers>\d+) workers, (?P<vcpus>\d+) vCPUs\); (?P<budget>[0-9.e+-]+)M steps in about "
                  r"(?P<hours>[0-9.]+|inf) h(?: BELOW FLOOR (?P<floor>\d+))?\Z"),
    "new": re.compile(r"Run directory: (?P<path>.+) \(new run\)\Z"),
    "resume": re.compile(r"Run directory: (?P<path>.+) \(resuming session (?P<session>\d+): (?P<job>\S+) from "
                         r"(?P<steps>\d+) steps; (?P<done>\d+) jobs done\)\Z"),
    "live": re.compile(r"Run directory: (?P<path>.+): still running \(LIVE\.json written (?P<age>\d+) s ago\); "
                       r"nothing to do\Z"),
    "smoke": re.compile(r"S2 SMOKE Y(?P<i>[1-8]) (?P<item>[^:]+): (?P<value>.+): (?P<verdict>OK|FAILED|REPORTED)\Z"),
}
FAIL3_RE = re.compile(r"FAIL \((?P<condition>\S+) seed (?P<seed>\d+) at (?P<ms>\d+) ms, first difference at step "
                      r"(?P<step>\d+)\)\Z")


def lines_of(text: str) -> dict:
    """The S2 lines of a step's output by kind: ``{1: [...], 2: [...], 3: [...], 4: [...], "new": [...], ...}``."""
    out: dict = {k: [] for k in LINE_RE}
    for raw in str(text).splitlines():
        row = raw.rstrip("\n")
        for kind, rx in LINE_RE.items():
            if rx.match(row):
                out[kind].append(row)
    return out


# ------------------------------------------------------------------------------------------------
# Parameters, the queue, the pins
# ------------------------------------------------------------------------------------------------
def load_params(source) -> dict:
    """The parameters with their defaults: ``source`` is a dict or the path of the JSON file the notebook writes.
    A key that is neither of ``PARAMS`` nor of ``EXTRA`` is refused (a misspelt parameter must not be a default)."""
    doc = dict(source) if isinstance(source, dict) else _read_json(source)
    if not isinstance(doc, dict):
        raise Refused(f"{source}: the parameters cannot be read (a JSON object of the notebook's cell 1)")
    unknown = sorted(k for k in doc if k not in PARAMS and k not in EXTRA)
    if unknown:
        raise Refused(f"unknown parameters: {', '.join(unknown)}")
    return {**PARAMS, **EXTRA, **doc}


def job_name(mode: str, body: str, seed: int) -> str:
    """``train2.job_name`` (a test holds the two equal)."""
    return "%s-%s-s%d" % (mode, body, int(seed))


def parse_queue(queue) -> list:
    """``["R5:75-M-H-10:42", ...]`` as ``[{entry, job, mode, body, seed}]``; a queue that is no list of
    ``MODE:BODY:SEED`` with MODE R5 or R6, or that names a job twice, is refused."""
    if isinstance(queue, str):
        queue = [x for x in queue.split(",") if x.strip()]
    if not isinstance(queue, (list, tuple)):
        raise Refused(f"QUEUE is a list of MODE:BODY:SEED, not {queue!r}")
    out = []
    for entry in queue:
        parts = str(entry).strip().split(":")
        if len(parts) != 3 or parts[0] not in ("R5", "R6") or not _NAME.match(parts[1]) or not parts[2].isdigit():
            raise Refused(f"QUEUE entry {entry!r} is not MODE:BODY:SEED (MODE R5 or R6, SEED a whole number)")
        row = dict(entry=":".join(parts), job=job_name(parts[0], parts[1], int(parts[2])), mode=parts[0],
                   body=parts[1], seed=int(parts[2]))
        if any(r["job"] == row["job"] for r in out):
            raise Refused(f"QUEUE names {row['job']} twice")
        out.append(row)
    return out


def resolve_workers(workers) -> int:
    """``WORKERS``: 0 is one per vCPU, at most 8 (``train2`` reads 0 as "no worker process")."""
    w = int(workers or 0)
    return min(MAX_WORKERS, _vcpus()) if w <= 0 else min(MAX_WORKERS, w)


def run_dirs(run_id, drive, local=None) -> tuple:
    """``(the run directory on Drive, the local run directory, whether they are the same)``: ``<drive>/runs/
    <run_id>`` and ``<local>/<run_id>``; with ``local`` None or equal to ``drive`` the trainer writes into the run
    directory itself and nothing is copied."""
    if not isinstance(run_id, str) or not _NAME.match(run_id):
        raise Refused(f"RUN_ID {run_id!r} is not a run id (letters, digits, '.', '_' and '-', no path)")
    if not drive:
        raise Refused("no Drive directory (DRIVE_DIR)")
    d = _anchored(drive).resolve()
    same = local is None or _anchored(local).resolve() == d
    run = d / "runs" / run_id
    return run, (run if same else _anchored(local).resolve() / run_id), same


def pins(path=None) -> dict:
    """``{distribution: version}`` of ``stage2/tb/requirements-colab.txt``, in the file's order (lines
    ``name==version``; ``#`` starts a comment)."""
    p = _root() / PINS_FILE if path is None else Path(path)
    out = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        row = raw.split("#", 1)[0].strip()
        if not row:
            continue
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.]+)", row)
        if not m:
            raise NbError(f"{p}: {raw!r} is not name==version")
        out[m.group(1)] = m.group(2)
    if "torch" not in out:
        raise NbError(f"{p}: no pin for torch")
    return out


def installed(dist: str) -> str | None:
    """The installed version of a distribution as the files on disk say it now (None: not installed)."""
    import importlib
    import importlib.metadata

    importlib.invalidate_caches()
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions(pinned: dict) -> dict:
    """``{"python": ..., <each pinned distribution>: version | None}``; torch with its build label."""
    return dict(python="%d.%d.%d" % sys.version_info[:3], **{name: installed(name) for name in pinned})


def comparable(version) -> str | None:
    """A version with its build label dropped (``2.15.0+cpu`` -> ``2.15.0``): what a pin is compared with."""
    return None if version is None else str(version).split("+", 1)[0]


def torch_build(version) -> str | None:
    """torch's build label (``cpu``, ``cu126``; None: no label, PyPI's default build)."""
    return str(version).split("+", 1)[1] if version and "+" in str(version) else None


def pin_differences(pinned: dict, vers: dict) -> list:
    """Every pin whose installed version (label dropped) is another, as sentences."""
    return [f"{name}: installed {vers.get(name)}, pinned {want}" for name, want in pinned.items()
            if comparable(vers.get(name)) != want]


def pip_commands(pinned: dict, vers: dict) -> list:
    """What ``setup`` runs on Colab (rule N2), as argument lists: torch's CPU build from the CPU index whenever the
    installed version is another or its build label is not ``cpu`` (pip takes an installed ``2.15.0+cu...`` as
    satisfying ``==2.15.0``, so the label is named); then every other pin in one command when one of them
    differs.  No unpinned fallback."""
    cmds = []
    if comparable(vers.get("torch")) != pinned["torch"] or torch_build(vers.get("torch")) != "cpu":
        cmds.append([sys.executable, "-m", "pip", "install", "-q", f"torch=={pinned['torch']}+cpu", "--index-url",
                     TORCH_CPU_INDEX])
    rest = [n for n in pinned if n != "torch"]
    if any(comparable(vers.get(n)) != pinned[n] for n in rest):
        cmds.append([sys.executable, "-m", "pip", "install", "-q", *[f"{n}=={pinned[n]}" for n in rest]])
    return cmds


def _call(cmd, timeout=None) -> int:
    """Run a setup command with its output passed through; its exit code (127: it could not be started)."""
    try:
        return subprocess.run(cmd, timeout=timeout).returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        _say(f"{cmd[0]}: {exc}")
        return 127


def setup(params: dict, record=None) -> dict:
    """Cell 2's step after the Git block (design 3.9; rule N2): on Colab the legacy gym packages are removed, the
    pins applied where the installed versions differ (``pip_commands``) and ``libosmesa6`` and ``ffmpeg``
    installed after ``apt-get update`` (a failure of either is reported, not fatal); anywhere, the installed
    versions are compared with the pins (the build label dropped) and **line 1** is printed.  A pin that does not
    hold afterwards raises, and on Colab so does a torch whose build label is not ``cpu`` after the installs (rule
    N2: the CPU build, forced when another build is installed): there is no unpinned fallback.  ``record``: where
    the step's record is written (``SETUP_SCHEMA``; ``session`` reads it)."""
    p = load_params(params)
    t0 = float(p["SETUP_STARTED"] or time.time())
    in_colab = bool(p["IN_COLAB"])
    pinned = pins()
    before = versions(pinned)
    ran, apt, apt_update = [], None, None
    if in_colab:
        _call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "gym", "gym-notices", "shimmy"])
        for cmd in pip_commands(pinned, before):
            _say("S2 setup: " + " ".join(cmd[2:]))
            code = _call(cmd)
            ran.append(dict(cmd=cmd[2:], exit_code=code))
            if code != 0:
                raise NbError(f"a pin did not install (exit code {code}): {' '.join(cmd[2:])}.  There is no "
                              "unpinned fallback: stop here and send this cell's output")
        # (a runtime's package lists can be older than its mirrors: without the update an install may not be found)
        apt_update = _call(["apt-get", "update", "-qq"])
        if apt_update != 0:
            _say(f"S2 setup: apt-get update failed (exit code {apt_update}): the install below may fail too")
        apt = _call(["apt-get", "install", "-y", "-qq", "libosmesa6", "ffmpeg"])
        if apt != 0:
            _say(f"S2 setup: apt-get install libosmesa6 ffmpeg failed (exit code {apt}): clips may not be made; "
                 "training does not need them")
    after = versions(pinned)
    differ = pin_differences(pinned, after)
    if in_colab and torch_build(after.get("torch")) != "cpu":
        differ.append(f"torch: the installed build is {after.get('torch')}, not the CPU build "
                      f"{pinned['torch']}+cpu (rule N2)")
    minor_ok = sys.version_info[:2] == PYTHON_MINOR
    ffmpeg = shutil.which("ffmpeg")
    repo = repo_state()
    ref = str(p["REPO_REF"] or "")
    ref_is_commit = bool(_HEX40.match(ref)) and ref == repo["commit"]
    doc = dict(schema=SETUP_SCHEMA, started_epoch=t0, in_colab=in_colab, pins=pinned, versions_before=before,
               versions=after, torch_build=torch_build(after.get("torch")), pins_match=not differ,
               pin_differences=differ, python_minor_matches=minor_ok, pip=ran, apt_exit_code=apt,
               apt_update_exit_code=apt_update, ffmpeg=ffmpeg,
               repo=dict(ref=ref, commit=repo["commit"], dirty=repo["dirty"], dirty_paths=repo["dirty_paths"],
                         ref_is_commit=ref_is_commit),
               line=None, seconds=None, at=_now())
    _say(f"S2 setup: Python {after['python']}"
         + ("" if minor_ok else f" (not {PYTHON_MINOR[0]}.{PYTHON_MINOR[1]}: reported; rule K9 covers it)")
         + f"; torch {after.get('torch')}; ffmpeg: {ffmpeg or 'not found'}")
    if differ:
        if record:
            _write_json(record, doc)
        raise NbError("the installed versions are not the pins (" + "; ".join(differ) + ").  There is no unpinned "
                      "fallback: stop here and send this cell's output")
    if repo["commit"] is None:
        raise NbError(f"{study()}: git cannot tell this checkout's commit")
    doc["line"] = line1(repo["commit"], ref_is_commit, not repo["dirty"])
    doc["seconds"] = time.time() - t0
    _say(doc["line"])
    if record:
        _write_json(record, doc)
    return doc


# ------------------------------------------------------------------------------------------------
# The seams: where this module meets the other tools (the tests replace them)
# ------------------------------------------------------------------------------------------------
def seams() -> dict | None:
    """The stub commands of ``S2_NB_SEAMS`` (a JSON file ``{"trainer": [...], "clip": [...], ...}``), None when
    the variable is not set.  It exists for the tests, whose launchers run in processes of their own: a session
    opened with it says so in its record and on its first line, and no run made with it is a run."""
    path = os.environ.get("S2_NB_SEAMS")
    doc = _read_json(path) if path else None
    return doc if isinstance(doc, dict) else None


def _seam(name: str, default: list) -> list:
    doc = seams()
    return [str(x) for x in doc[name]] if doc and doc.get(name) else default


def _trainer_cmd() -> list:
    return _seam("trainer", [sys.executable, "-m", "designlab.stage2.train2"])


def _clip_cmd() -> list:
    return _seam("clip", [sys.executable, "-m", "designlab.stage2.clip2"])


def _pack_cmd() -> list:
    return _seam("pack", [sys.executable, "-m", "designlab.stage2.s2load"])


def _bundle_cmd() -> list:
    return _seam("bundle", [sys.executable, "-m", "designlab.stage2.bundle"])


def _child_cmd(what: str, *args) -> list:
    """This module's own child (``_child_main``): the steps that import ``setio``, ``r5env`` or ``g8a``."""
    return _seam("child", [sys.executable, "-m", "designlab.stage2.nb", "child"]) + [what, *[str(a) for a in args]]


def _rebuild_cmd(set_name: str, body: str, root) -> list:
    """Rule N4's second fallback and rule Y8: a set's arrays built again from the committed records (on a frozen
    set ``sets build`` puts an array in place only when its sha256 is the freeze manifest's)."""
    return _seam("rebuild", [sys.executable, "-m", "designlab.stage2.sets", "build"]) + [
        "--set", set_name, "--body", body, "--root", str(root), "--procs", "2"]


def _child_env(gl: str = "disable") -> dict:
    """``tbenv.child_env`` and, with it, no lock on the checkout's index, unbuffered output and no bytecode files
    in the checkout."""
    env = tbenv.child_env(gl)
    env.setdefault("GIT_OPTIONAL_LOCKS", "0")
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("MPLBACKEND", None)  # (a kernel's inline backend is not importable in a child)
    return env


def _new_group() -> dict:
    """``Popen`` arguments that put a child in a process group of its own (the job's trainer, workers and monitor
    child are then ended as one, and never the launcher with them)."""
    return {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp}


def _json_child(cmd, gl: str = "disable", timeout=None, echo: bool = True) -> tuple:
    """Run a child whose last output line is a JSON object: ``(exit code, that object or None, its other
    lines)``.  The other lines are printed as they are when ``echo``; the child's stderr passes through."""
    try:
        done = subprocess.run(cmd, cwd=str(study()), env=_child_env(gl), stdout=subprocess.PIPE, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
        code, out = done.returncode, done.stdout or ""
    except subprocess.TimeoutExpired as exc:
        code, out = 124, (exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout) or ""
    except OSError as exc:
        return 127, None, [str(exc)]
    rows = [x for x in out.splitlines() if x.strip()]
    doc = None
    if rows:
        try:
            doc = json.loads(rows[-1])
        except ValueError:
            doc = None
        if isinstance(doc, dict):
            rows = rows[:-1]
        else:
            doc = None
    if echo:
        for row in rows:
            _say(row)
    return code, doc, rows


# ------------------------------------------------------------------------------------------------
# The session's state and the run's live files
# ------------------------------------------------------------------------------------------------
def _state_path(lrun: Path) -> Path:
    return Path(lrun) / "work" / "session.json"


def _open_state(run_id, drive, local) -> dict:
    """The session the ``session`` step opened here (the launcher's own state under ``work/``)."""
    _, lrun, _ = run_dirs(run_id, drive, local)
    st = _read_json(_state_path(lrun))
    if not isinstance(st, dict) or (st.get("session") or {}).get("run_id") != run_id:
        raise Refused(f"no open session for run {run_id} here ({_state_path(lrun)}): run the session step first")
    return st


def _save_state(st: dict, new: bool = False) -> None:
    """Write the session's state.  A state file that another session opened here since (the same local directory:
    a takeover on one machine) is not this session's to replace, unless ``new`` (``session`` itself)."""
    path = _state_path(Path(st["local_run"]))
    if not new:
        there = _read_json(path)
        if isinstance(there, dict) and (there.get("session") or {}).get("session_uuid") not in (
                None, st["session"]["session_uuid"]):
            return
    _write_json(path, st)


def _live_age(live) -> float | None:
    """Seconds since a ``LIVE.json`` was written; None when there is none, it cannot be read, or the session that
    wrote it closed it (it ended and said so)."""
    if not isinstance(live, dict) or live.get("closed_at"):
        return None
    t = _epoch(live.get("written_at"))
    return None if t is None else max(0.0, time.time() - t)


def _owner(d: Path, uid: str):
    """None when ``LIVE.json`` names ``uid`` (or names nobody that can be read); else the session it names."""
    live = _read_json(Path(d) / "LIVE.json")
    if isinstance(live, dict) and live.get("session_uuid") not in (None, uid):
        return live.get("session", "?")
    return None


def _write_live(st: dict, closed: str | None = None) -> None:
    s = st["session"]
    doc = dict(schema=LIVE_SCHEMA, run_id=s["run_id"], session=s["session"], session_uuid=s["session_uuid"],
               host=socket.gethostname(), pid=os.getpid(), written_at=_now(), rule=FENCE_RULE)
    if closed is not None:
        doc.update(closed_at=_now(), closed_reason=closed)
    _write_json(Path(st["drive_run"]) / "LIVE.json", doc)


def _finished(status) -> bool:
    return status == "done" or str(status or "").startswith("stopped")


def _drive_checkpoints(jobdir: Path) -> list:
    ck = Path(jobdir) / "checkpoints"
    if not ck.is_dir():
        return []
    found = (re.fullmatch(r"checkpoint_(\d+)\.json", p.name) for p in ck.iterdir())
    return sorted(int(m.group(1)) for m in found if m)


def _job_entry(st: dict, row: dict, mine: dict | None = None) -> dict:
    """One job's entry of ``STATUS.json`` ``queue`` (design 3.9): the trainer's ``progress.json`` (the local one,
    else the copy on Drive) with the launcher's own fields (``mine``)."""
    mine = mine or {}
    d, lrun = Path(st["drive_run"]), Path(st["local_run"])
    prog = run = None
    for base in (lrun, d):
        jd = base / "jobs" / row["job"]
        prog = prog or _read_json(jd / "progress.json")
        run = run or _read_json(jd / "run.json")
    prog = prog if isinstance(prog, dict) else {}
    run = run if isinstance(run, dict) else {}
    status = run.get("status") or prog.get("status")
    state = mine.get("state")
    if state is None:
        state = "done" if status == "done" else "stopped" if _finished(status) else "queued"
    mon = prog.get("last_monitor") if isinstance(prog.get("last_monitor"), dict) else None
    flags = prog.get("flags") if isinstance(prog.get("flags"), dict) else {}
    return dict(
        job=row["job"], state=state, steps=prog.get("steps", run.get("num_timesteps")),
        budget=prog.get("budget", run.get("budget")), env_steps_per_s=prog.get("env_steps_per_s"),
        eta_h=prog.get("eta_h"), last_checkpoint=prog.get("last_checkpoint"),
        last_checkpoint_copied=mine.get("last_checkpoint_copied"),
        last_monitor=None if mon is None else {k: mon.get(k) for k in (
            "at_steps", "N", "S", "mean_clock_factor", "gain", "counted_without_record")},
        g8a=prog.get("g8a"), g8b=prog.get("g8b"), fix=prog.get("fix"), flags=dict(kl_cut=flags.get("kl_cut")),
        minibatch_steps_last=prog.get("minibatch_steps_last"), retries=int(mine.get("retries") or 0),
        last_error=mine.get("last_error"),
        # (the trainer's three later keys: a check counted without a record and a halt go to the lead)
        checks_without_record=prog.get("checks_without_record") or [], halt=prog.get("halt"),
        run_status=status, run_uuid=run.get("run_uuid") or prog.get("run_uuid"),
        stopped=(run.get("stopped") or prog.get("stopped")), exit_codes=list(mine.get("exit_codes") or []),
        clip=mine.get("clip"), drive_errors=int(mine.get("drive_errors") or 0),
        not_on_drive=list(mine.get("not_on_drive") or []))


def _write_status(st: dict, state: str, entries: list | None = None) -> dict:
    """``STATUS.json`` (``STATUS_SCHEMA``; design 3.9) into the run directory; the document."""
    s = st["session"]
    rows = entries if entries is not None else [_job_entry(st, r) for r in parse_queue(s["queue"])]
    rt = dict(vcpus=_vcpus(), ram_gb=_ram_gb(), disk_free_gb=_disk_free_gb(st["local_run"]))
    doc = dict(schema=STATUS_SCHEMA, run_id=s["run_id"], run_uuids={r["job"]: r["run_uuid"] for r in rows},
               session=s["session"], session_uuid=s["session_uuid"], written_at=_now(), started_at=s["started_at"],
               commit=s["repo"]["commit"], bundle_id=s["bundle"]["id"], state=state,
               runtime=dict(vcpus=rt["vcpus"], ram_gb=rt["ram_gb"], disk_free_gb=rt["disk_free_gb"]),
               lines=list(st["lines"]), queue=rows)
    _write_json(Path(st["drive_run"]) / "STATUS.json", doc)
    return doc


def _note(st: dict, line: str | None = None, state: str = "running", **fields) -> None:
    """A step's bookkeeping while the session owns the run: the printed line kept, the session's record updated
    and ``last_status_at`` moved, ``SESSION_<k>.json``, ``STATUS.json`` and ``LIVE.json`` rewritten.  A session
    that no longer owns the run writes only its own state."""
    s = st["session"]
    if line is not None:
        st["lines"].append(line)
    s.update(fields)
    s["last_status_at"] = _now()
    s["lines"] = list(st["lines"])
    _save_state(st)
    d = Path(st["drive_run"])
    if _owner(d, s["session_uuid"]) is None:
        _write_json(d / f"SESSION_{s['session']}.json", s)
        _write_status(st, state)
        _write_live(st)


def _close(st: dict, reason: str) -> None:
    """The session ends before any job could run on: ``LIVE.json`` is closed, so that a later RESUME is not held
    back for ten minutes by a session that is known to be over.  The session is marked closed in its own state and
    record: it launches nothing afterwards (``_refuse_closed``)."""
    s = st["session"]
    s["closed"] = dict(at=_now(), reason=reason)
    _save_state(st)
    d = Path(st["drive_run"])
    if _owner(d, s["session_uuid"]) is None:
        _write_json(d / f"SESSION_{s['session']}.json", s)
        _write_live(st, closed=reason)


def _refuse_closed(st: dict, what: str) -> None:
    """Rule N1: line 2 ending FAILED or line 3 ending FAIL stops the session before any training.  The cells'
    order does that for "Run all"; this does it for a cell run by hand afterwards."""
    closed = st["session"].get("closed")
    if isinstance(closed, dict):
        raise Refused(f"{st['drive_run']}: this session was closed ({closed.get('reason')}): it {what} (rule N1: a "
                      "failed line 2 or line 3 stops the session before any training); only a new session "
                      "re-enters the run")


# ------------------------------------------------------------------------------------------------
# session (cell 3; rules K6, K7)
# ------------------------------------------------------------------------------------------------
#: What a RESUME must repeat (rule K6: one queue, one set, one bundle, one budget; the trainer's own options are
#: its settings, rule K4).  The workers, the clips and the launcher's period may differ between sessions.
RESUME_KEYS = ("queue", "set", "budget_steps", "bundle_id", "tiny", "smoke", "monitor_every", "monitor_also_k", "lr",
               "config")


def _queue_doc(run_id, jobs: list, repo_ref, commit, p: dict) -> dict:
    also = [int(k) for k in (p["MONITOR_ALSO_K"] or [])]
    return dict(schema=QUEUE_SCHEMA, run_id=run_id, queue=[r["entry"] for r in jobs],
                jobs=[dict(job=r["job"], mode=r["mode"], body=r["body"], seed=r["seed"]) for r in jobs],
                set=str(p["SET"]), budget_steps=int(p["BUDGET_STEPS"] or 0), budget_by=p["BUDGET_BY"],
                budget_set_on=p["BUDGET_SET_ON"], bundle_id=str(p["BUNDLE_ID"] or "") or None,
                bundle=dict(id=str(p["BUNDLE_ID"] or "") or None, content_sha256=None,
                            source=str(p["BUNDLE_SOURCE"] or "drive")),
                repo_ref=repo_ref, commit=commit, tiny=bool(p["TINY"]), smoke=bool(p["SMOKE"]),
                monitor_every=None if p["MONITOR_EVERY"] is None else int(p["MONITOR_EVERY"]),
                monitor_also_k=also, lr=None if p["LR"] is None else float(p["LR"]),
                config=p["CONFIG"] if isinstance(p["CONFIG"], dict) else None,
                rules=dict(k6=K6_RULE, k7=K7_RULE, fence=FENCE_RULE, halt=HALT_RULE, copy=COPY_RULE, jobs=JOB_RULE,
                           clips=CLIP_RULE, line4=LINE4_RULE, budget=BUDGET_RULE, back=BACK_RULE,
                           unfinished=UNFINISHED_RULE))


def _resume_point(st_like: dict, jobs: list) -> dict:
    """What a resume starts from, read from the run directory: the first job that is not finished, the newest
    checkpoint of it that Drive holds a record of (the trainer's own checks choose the one it loads), and how many
    jobs are finished."""
    d = Path(st_like["drive_run"])
    done, first = 0, None
    for r in jobs:
        e = _job_entry(st_like, r)
        if e["state"] in ("done", "stopped"):
            done += 1
        elif first is None:
            first = r
    if first is None:
        return dict(job="none", steps=0, done=done)
    have = [n for n in _drive_checkpoints(d / "jobs" / first["job"])]
    return dict(job=first["job"], steps=max(have, default=0), done=done)


def session(run_id, drive, local, queue, repo_ref, commit, resume, params) -> dict:
    """Open a session on a run (cell 3; ``K6_RULE``, ``K7_RULE``): the refusals, the stale ``*.part`` sweep,
    ``QUEUE.json`` (a new run), ``SESSION_<k>.json`` (``SESSION_SCHEMA``), ``LIVE.json``, ``STATUS.json`` and the
    run-directory line.  ``commit`` is the checkout's commit, ``repo_ref`` the cell's ``REPO_REF``, ``params`` the
    whole parameter set (recorded).  Raises ``Refused`` before anything is written; returns the session's record
    with ``line``."""
    p = load_params(params)
    jobs = parse_queue(queue)
    d, lrun, same = run_dirs(run_id, drive, local)
    smoke, tiny, resume = bool(p["SMOKE"]), bool(p["TINY"]), bool(resume)
    repo_ref, commit = str(repo_ref or ""), str(commit or "")
    if not jobs and not smoke:
        raise Refused("QUEUE is empty: a launch names its jobs")
    if not _HEX40.match(commit):
        raise Refused(f"the checkout's commit is not known ({commit!r})")
    if not smoke and not (_HEX40.match(repo_ref) and repo_ref == commit):
        raise Refused(f"REPO_REF {repo_ref!r} is not the 40 hex digits of this checkout's commit ({commit}): a "
                      "launch names its commit (rule K6)")
    if not smoke and not tiny and int(p["BUDGET_STEPS"] or 0) <= 0:
        raise Refused("BUDGET_STEPS is not set: a launch names its budget (rule Y11 fixes it after the smoke)")
    if str(p["BUNDLE_ID"] or "") == S1_BUNDLE:
        raise Refused(f"BUNDLE_ID {S1_BUNDLE!r} is the name line 2 gives the Stage 1 view; a bundle has another")
    if p["IN_COLAB"] and seams() is not None:
        raise Refused("S2_NB_SEAMS names stub commands in place of the trainer: it is a hook of the tests and is "
                      "refused on Colab; nothing was changed")
    want = _queue_doc(run_id, jobs, repo_ref, commit, p)
    qpath = d / "QUEUE.json"
    qdoc, notes, took_over, source_change = _read_json(qpath), [], None, False
    known = qpath.exists() or (d.is_dir() and any(d.glob("SESSION_*.json")))
    if not resume:
        if known:
            raise Refused(f"{d}: RUN_ID {run_id} exists, and only a resume re-enters a run (RESUME); nothing was "
                          "changed (rule K6)")
    else:
        if smoke:
            raise Refused("a smoke is not resumed (rule K6): nothing was changed")
        if not isinstance(qdoc, dict):
            raise Refused(f"{d}: nothing to resume (no QUEUE.json that can be read); nothing was changed")
        for name in ("DONE.json", "HALT.json"):
            if (d / name).exists():
                why = (_read_json(d / name) or {}).get("reason")
                raise Refused(f"{d}: the run is finished ({name}: {why}); it is not resumed (rule K7)")
        diff = [f"{k}: {qdoc.get(k)!r} -> {want[k]!r}" for k in RESUME_KEYS if qdoc.get(k) != want[k]]
        if diff:
            raise Refused(f"{d}: RESUME with other settings than the run's ({'; '.join(diff)}): refused, nothing "
                          "was changed (rule K6)")
        if qdoc.get("commit") != commit:
            if not p["ALLOW_SOURCE_CHANGE"]:
                raise Refused(f"{d}: this checkout is commit {commit}, the run's is {qdoc.get('commit')}: another "
                              "commit is accepted only with ALLOW_SOURCE_CHANGE, on the lead's word (rule K6)")
            source_change = True
            notes.append(f"ALLOW_SOURCE_CHANGE: the run's commit is {qdoc.get('commit')}, this session's {commit}")
        # (with the same commit the cell's REPO_REF is QUEUE.json's: both are that commit's 40 hex digits)
        live = _read_json(d / "LIVE.json")
        age = _live_age(live)
        if age is not None and age < LIVE_FRESH_S:
            if not p["TAKEOVER"]:
                raise Refused(session_line(d, "live", age_s=age), line=True)
            took_over = dict(session=live.get("session"), session_uuid=live.get("session_uuid"), age_s=age,
                             host=live.get("host"))
            notes.append(f"TAKEOVER: LIVE.json of session {live.get('session')} was written {age:.0f} s ago")

    # ---- from here on the run directory is written
    swept = sweep_parts(d) + ([] if same else sweep_parts(lrun))
    ks = [int(m.group(1)) for f in (d.glob("SESSION_*.json") if d.is_dir() else [])
          if (m := re.fullmatch(r"SESSION_(\d+)\.json", f.name))]
    k = 1 + max(ks, default=-1)
    if not resume:
        qdoc = dict(want, created_at=_now())
        _write_json(qpath, qdoc)
    setup_doc = _read_json(p["SETUP_RECORD"]) if p["SETUP_RECORD"] else None
    setup_doc = setup_doc if isinstance(setup_doc, dict) else {}
    pinned = pins()
    vers = versions(pinned)
    repo = repo_state()
    prev = _read_json(_state_path(lrun))
    sess = dict(
        schema=SESSION_SCHEMA, run_id=run_id, session=k, session_uuid=uuid.uuid4().hex, started_at=_now(),
        last_status_at=None, resume=resume, takeover=bool(p["TAKEOVER"]), took_over=took_over, smoke=smoke,
        repo=dict(ref=qdoc.get("repo_ref") if resume else repo_ref, commit=commit, dirty=repo["dirty"],
                  cell_ref=repo_ref, run_commit=qdoc.get("commit")),
        runtime=machine(lrun, p["IN_COLAB"]), versions=vers, torch_build=torch_build(vers.get("torch")),
        pins=pinned, pins_match=not pin_differences(pinned, vers),
        python_minor_matches=sys.version_info[:2] == PYTHON_MINOR, ffmpeg=shutil.which("ffmpeg"),
        bundle=dict(id=qdoc.get("bundle_id"), content_sha256=(qdoc.get("bundle") or {}).get("content_sha256"),
                    files_ok=None, source=str(p["BUNDLE_SOURCE"] or "drive")),
        queue=list(qdoc["queue"]), budget_steps=qdoc["budget_steps"], workers=resolve_workers(p["WORKERS"]),
        allow_version_change=bool(p["ALLOW_VERSION_CHANGE"]), allow_source_change=bool(p["ALLOW_SOURCE_CHANGE"]),
        source_change=source_change, setup_seconds=None, params=p, swept_parts=swept, notes=notes,
        setup=dict(started_epoch=setup_doc.get("started_epoch"), pip=setup_doc.get("pip"),
                   apt_exit_code=setup_doc.get("apt_exit_code"), versions_before=setup_doc.get("versions_before")),
        lines=[], seams=seams(), rules=dict(k6=K6_RULE, k7=K7_RULE))
    st = dict(session=sess, lines=[setup_doc["line"]] if setup_doc.get("line") else [], drive_run=str(d),
              local_run=str(lrun), same=same, child=None,
              stale_child=prev.get("child") if isinstance(prev, dict) else None)
    if resume:
        at = _resume_point(st, jobs)
        line = session_line(d, "resume", session=k, job=at["job"], steps=at["steps"], done=at["done"])
    else:
        line = session_line(d, "new")
    _save_state(st, new=True)
    _write_live(st)  # (first: from here a second session is refused, and a superseded launcher sees it)
    if sess["seams"]:
        _say("S2 NOTE: stub commands are in force (S2_NB_SEAMS): this session is a test, not a run")
    for text in notes:
        _say(f"S2 NOTE: {text}")
    if swept:
        _say(f"S2 session: removed {len(swept)} stale .part file{'' if len(swept) == 1 else 's'} (a copy that was cut)")
    _say(line)
    _note(st, line)
    return dict(sess, line=line)


# ------------------------------------------------------------------------------------------------
# inputs (cell 4; rules N3, N4)
# ------------------------------------------------------------------------------------------------
def _inputs_s1(bodies) -> dict:
    """The Stage 1 view has no bundle (development; design 3.10): every file of each body's Stage 1 run directory
    against its entry's sha256 (``setio.stage1_view``, in a child).  Line 2 names the bundle ``S1-view`` and, as
    its manifest, the hash of the bodies' file tables."""
    tables, ok, n, problems = {}, 0, 0, []
    for body in bodies:
        code, doc, _ = _json_child(_child_cmd("read-inputs", "--set", S1_SET, "--body", body))
        if code != 0 or not isinstance(doc, dict) or not doc.get("ok"):
            problems.append(f"{body}: {(doc or {}).get('problem') or f'the check ended with exit code {code}'}")
            continue
        tables[body] = doc["table_sha256"]
        ok += int(doc["files"])
        n += int(doc["files"])
    if not bodies:
        problems.append("no body to verify")
    digest = sha256_json(tables) if tables else None
    return dict(ok=not problems, bundle_id=S1_BUNDLE, content_sha256=digest, files_ok=ok, files=n, source="s1-view",
                problems=problems, sets={S1_SET: dict.fromkeys(bodies, True)})


def _fetch_parts(src: Path, names, dst: Path, parts: dict, problems: list) -> None:
    """Copy the named parts from ``src`` to ``dst`` (a part already there with the manifest's size and sha256 is
    left); a part that is missing is a problem."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        want, there = parts[name], dst / name
        if there.is_file() and there.stat().st_size == want.get("bytes") and sha256_file(there) == want.get("sha256"):
            continue
        if not (src / name).is_file():
            problems.append(f"{name}: missing in {src}")
            continue
        _copy_file(src / name, there)


def _parts_from_git(ref: str, bundle_id: str, names, dst: Path, problems: list) -> None:
    """Rule N4's third fallback: the parts read from the data branch ``ref`` of the clone's remote (``git fetch
    origin <ref>``, then each part's blob)."""
    top = (_git("rev-parse", "--show-toplevel") or "").strip()
    prefix = (_git("rev-parse", "--show-prefix") or "").strip()
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    if not top or not ref or ref.startswith("-"):
        problems.append(f"git:{ref}: not a ref of a git checkout")
        return
    got = subprocess.run(["git", "-C", top, "fetch", "--quiet", "origin", ref], env=env)
    sha = (_git("rev-parse", "FETCH_HEAD^{commit}", cwd=top) or "").strip() if got.returncode == 0 else ""
    if not sha:
        problems.append(f"git:{ref}: the ref could not be fetched from the remote")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        tmp = _part(dst / name)
        with open(tmp, "wb") as fh:
            done = subprocess.run(["git", "-C", top, "cat-file", "blob",
                                   f"{sha}:{prefix}{BUNDLE_DIR}/{bundle_id}/parts/{name}"], stdout=fh,
                                  stderr=subprocess.DEVNULL, env=env)
        if done.returncode != 0:
            _unlink(tmp)
            problems.append(f"{name}: not on the data branch {ref} ({sha[:12]})")
        else:
            os.replace(tmp, dst / name)


def rebuilt_dir(drive, bundle_id) -> Path:
    """Where a verified rebuild's arrays are kept on Drive (design D20), with the smoke's permission to use them
    (``REBUILD_OK.json``, written by rule Y8 when every file was compared and matched)."""
    return _anchored(drive) / "inputs" / str(bundle_id) / "rebuilt"


def _other_bundle(expect, got) -> str | None:
    """Rule K6 (one bundle for a run's whole life): the problem when a resume's inputs have another content hash
    than the run's first session recorded in QUEUE.json (``expect``; None: a new run); else None."""
    if expect and expect != got:
        return (f"RESUME with another bundle: the run's content hash is {str(expect)[:12]}, this checkout's inputs "
                f"give {str(got)[:12]} (rule K6)")
    return None


def inputs(bundle_id, drive, source="drive", *, set_name=None, bodies=(), local=None, expect=None) -> dict:
    """Cell 4 (rules N3, N4): the bundle's parts brought to local disk, each against the committed manifest's
    sha256 (``bundle unpack``), unpacked at their study paths, every file's sha256 (``bundle verify --json``: the
    bundle id and the 64-hex content hash it reports must be the quoted ones), then ``setio.read_inputs(
    require_frozen=True)`` per set and body in a child.  Prints **line 2** and returns ``{ok, bundle_id,
    content_sha256, files_ok, files, source, problems, line, sets}``; never raises on a difference.

    ``source``: ``"drive"`` (the parts under ``<drive>/inputs/<bundle_id>/``); ``"git:<ref>"`` (the parts on a data
    branch); ``"rebuild"`` (the model parts from Drive and the arrays from ``rebuilt/`` on Drive or built again by
    ``sets build``, allowed only when the smoke's rule Y8 left ``REBUILD_OK.json`` for this content hash; verified
    arrays are kept on Drive).  Without a ``bundle_id`` only the Stage 1 view (``set_name`` ``S1``) is verified,
    for ``bodies``.  ``local``: the local root the parts are copied under (default: a temporary directory).
    ``expect``: the content hash the run's first session recorded (a resume); inputs with another are a failed
    line 2 (rule K6), so the line that is printed and kept never reads OK for them."""
    source = str(source or "drive")
    if not bundle_id:
        if set_name != S1_SET:
            res = dict(ok=False, bundle_id="none", content_sha256=None, files_ok=0, files=0, source=source, sets={},
                       problems=[f"no BUNDLE_ID: set {set_name} needs its bundle (only the Stage 1 view has none)"])
        else:
            res = _inputs_s1(list(bodies))
        other = None if res["problems"] else _other_bundle(expect, res["content_sha256"])
        if other:
            res["problems"].append(other)
            res["ok"] = False
        res["line"] = line2(res["bundle_id"], res["content_sha256"], res["files_ok"], res["files"],
                            res["problems"][0] if res["problems"] else None)
        _say(res["line"])
        return res
    res = dict(ok=False, bundle_id=str(bundle_id), content_sha256=None, files_ok=0, files=0, source=source,
               problems=[], sets={}, unpacked=None)
    problems = res["problems"]

    def end() -> dict:
        res["ok"] = not problems
        res["line"] = line2(res["bundle_id"], res["content_sha256"], res["files_ok"], res["files"],
                            problems[0] if problems else None)
        _say(res["line"])
        return res

    root = _root()
    mpath = root / BUNDLE_DIR / str(bundle_id) / "MANIFEST.json"
    doc = _read_json(mpath) if _NAME.match(str(bundle_id)) else None
    content = doc.get("content") if isinstance(doc, dict) else None
    if not isinstance(content, dict) or not isinstance(content.get("parts"), dict):
        problems.append(f"{BUNDLE_DIR}/{bundle_id}/MANIFEST.json is not in this checkout or cannot be read")
        return end()
    quoted, parts, files = doc.get("content_sha256"), content["parts"], content.get("files") or {}
    res.update(content_sha256=quoted if isinstance(quoted, str) else None, files=len(files))
    base = Path(tempfile.gettempdir()) / "s2inputs" if local is None else _anchored(local) / "inputs"
    pdir = base / str(bundle_id) / "parts"
    src = _anchored(drive) / "inputs" / str(bundle_id)
    names = sorted(parts)
    rebuild = source in ("rebuild", "rebuilt")
    if rebuild:
        ok_doc = _read_json(rebuilt_dir(drive, bundle_id) / "REBUILD_OK.json")
        if not isinstance(ok_doc, dict) or ok_doc.get("content_sha256") != quoted or not ok_doc.get("allowed"):
            problems.append("the rebuild is not an allowed fallback: no smoke's rule Y8 found every file of this "
                            "bundle equal (rule N4)")
            return end()
        names = [n for n in names if n.startswith("models-")]
        _fetch_parts(src, names, pdir, parts, problems)
    elif source == "drive":
        _fetch_parts(src, names, pdir, parts, problems)
    elif source.startswith("git:"):
        _parts_from_git(source[4:], str(bundle_id), names, pdir, problems)
    else:
        problems.append(f"BUNDLE_SOURCE {source!r}: drive, rebuild or git:<ref>")
    if problems:
        return end()
    cmd = _bundle_cmd() + ["unpack", "--manifest", str(mpath), "--parts", str(pdir), "--root", str(root), "--json"]
    if rebuild:
        cmd += ["--only", ",".join(names)]
    code, got, _ = _json_child(cmd, echo=False)
    if code != 0 or not isinstance(got, dict) or not got.get("ok"):
        problems.append((got or {}).get("first_problem") or f"bundle unpack ended with exit code {code}")
        return end()
    res["unpacked"] = dict(parts=got.get("parts"), files=got.get("files"), written=len(got.get("written") or []),
                           replaced=len(got.get("replaced") or []), left=len(got.get("left") or []))

    def verify(with_parts: bool) -> dict:
        cmd = _bundle_cmd() + ["verify", "--manifest", str(mpath), "--root", str(root), "--json"]
        if with_parts:
            cmd += ["--parts", str(pdir)]
        code, got, _ = _json_child(cmd, echo=False)
        return got if isinstance(got, dict) else dict(ok=False, first_problem=f"bundle verify ended with exit code "
                                                      f"{code} and no result", files_ok=0, files=len(files))

    got = verify(not rebuild)
    if rebuild and not got.get("ok"):
        got = _rebuild(drive, str(bundle_id), content, root, verify)
    res.update(files_ok=int(got.get("files_ok") or 0), files=int(got.get("files") or len(files)))
    if not got.get("ok"):
        problems.append(got.get("first_problem") or "bundle verify found a difference")
        return end()
    # (the lead's ruling of 16 Nov: the bundle id and the 64-hex content hash verify reports are the quoted ones)
    if got.get("bundle_id") != str(bundle_id):
        problems.append(f"the manifest is bundle {got.get('bundle_id')!r}, not {bundle_id!r}")
    sha = got.get("content_sha256")
    if not (isinstance(sha, str) and _HEX64.match(sha)) or sha != quoted:
        problems.append(f"the content hash verify reports ({str(sha)[:12]}) is not the manifest's own "
                        f"({str(quoted)[:12]})")
    if problems:
        return end()
    res["content_sha256"] = sha
    other = _other_bundle(expect, sha)
    if other:
        problems.append(other)
        return end()
    sets_root = root / "stage2" / "sets"
    for name in sorted(content.get("sets") or {}):
        res["sets"][name] = {}
        for body in sorted(content["sets"][name]):
            head = f"stage2/sets/{name}/{body}/"
            rels = sorted(f[len(head):] for f in files if f.startswith(head))
            code, doc, _ = _json_child(_child_cmd("read-inputs", "--set", name, "--body", body, "--root",
                                                  str(sets_root), "--files", json.dumps(rels)))
            good = code == 0 and isinstance(doc, dict) and bool(doc.get("ok")) and doc.get("files") == len(rels)
            res["sets"][name][body] = good
            if not good:
                problems.append(f"set {name} {body}: " + str((doc or {}).get("problem") or
                                                             f"read_inputs ended with exit code {code}"))
    return end()


def _rebuild(drive, bundle_id: str, content: dict, root: Path, verify) -> dict:
    """Rule N4's second fallback after the model parts are unpacked: the arrays kept on Drive by an earlier
    rebuild are put in place; what still differs is built again per set and body (``_rebuild_cmd``); a rebuild is
    accepted only if every sha256 then equals the manifest's, and its arrays are kept on Drive for the next
    launch.  ``verify``'s last result."""
    cache, files = rebuilt_dir(drive, bundle_id), content.get("files") or {}
    arrays = sorted(f for f, e in files.items() if not str(e.get("part", "")).startswith("models-"))
    for rel in arrays:
        there = root / rel
        if (cache / rel).is_file() and not (there.is_file() and sha256_file(there) == files[rel]["sha256"]):
            _copy_file(cache / rel, there)
    got = verify(False)
    if not got.get("ok"):
        for name in sorted(content.get("sets") or {}):
            for body in sorted(content["sets"][name]):
                head = f"stage2/sets/{name}/{body}/"
                if all((root / f).is_file() and sha256_file(root / f) == files[f]["sha256"]
                       for f in arrays if f.startswith(head)):
                    continue
                _say(f"S2 inputs: building the arrays of set {name} {body} again (rule N4; this takes a while)")
                subprocess.run(_rebuild_cmd(name, body, root / "stage2" / "sets"), cwd=str(study()),
                               env=_child_env("disable"), stdin=subprocess.DEVNULL)
        got = verify(False)
    if got.get("ok"):
        for rel in arrays:
            kept = cache / rel
            if not (kept.is_file() and sha256_file(kept) == files[rel]["sha256"]):
                _copy_file(root / rel, kept)
    return got


# ------------------------------------------------------------------------------------------------
# preflight (cell 5)
# ------------------------------------------------------------------------------------------------
def preflight(queue, set_name, *, bodies=None) -> dict:
    """Cell 5: the repository's archive-load preflight (its two lines), a policy built with ``MUJOCO_GL``
    assigned (``train2 preflight``), every env of the queue's modes and bodies built on both floors, and G8a's
    short form for each body (``g8a.short``; on the Stage 1 view ``g8a.stock`` on ``walk/brisk``): **line 3** per
    body.  ``bodies``: further bodies to gate (the acceptance run's: every body of the bundle).  Returns ``{ok,
    archive, policy, envs, g8a {body: ...}, lines, problems}``; a failure stops at its step ("G8a fails: no
    run")."""
    jobs = parse_queue(queue)
    gate = list(dict.fromkeys([r["body"] for r in jobs] + [str(b) for b in (bodies or [])]))
    res = dict(ok=False, archive=None, policy=None, envs=None, g8a={}, lines=[], problems=[])
    code, doc, _ = _json_child(_child_cmd("archive-preflight"))
    res["archive"] = dict(exit_code=code, ok=code == 0 and bool((doc or {}).get("ok")))
    if not res["archive"]["ok"]:
        res["problems"].append(f"the archive-load preflight failed (exit code {code})")
        return res
    code, doc, _ = _json_child(_trainer_cmd() + ["preflight"], echo=False)
    res["policy"] = dict(exit_code=code, ok=code == 0 and bool((doc or {}).get("ok")),
                         mujoco_gl=(doc or {}).get("mujoco_gl"), bytecode_members=(doc or {}).get("bytecode_members"),
                         fingerprint_max_abs_diff=(doc or {}).get("fingerprint_max_abs_diff"))
    if not res["policy"]["ok"]:
        res["problems"].append(f"train2 preflight failed (exit code {code}): {doc}")
        return res
    _say(f"S2 preflight: a policy built and saved with MUJOCO_GL={res['policy']['mujoco_gl']}; bytecode members: "
         f"{', '.join(res['policy']['bytecode_members'] or []) or 'none'}")
    pairs = list(dict.fromkeys((r["mode"], r["body"]) for r in jobs))
    pairs += [("R5", b) for b in gate if not any(b == x[1] for x in pairs)]
    code, doc, _ = _json_child(_child_cmd("envs", "--set", set_name, "--pairs", json.dumps(pairs)))
    res["envs"] = dict(exit_code=code, ok=code == 0 and bool((doc or {}).get("ok")), built=(doc or {}).get("built"),
                       problem=(doc or {}).get("problem"))
    if not res["envs"]["ok"]:
        res["problems"].append(f"an env could not be built: {res['envs']['problem'] or f'exit code {code}'}")
        return res
    _say(f"S2 preflight: {len(res['envs']['built'])} envs built on set {set_name} ("
         + ", ".join(f"{b['mode']} {b['body']} {b['floor']}" for b in res["envs"]["built"]) + ")")
    for body in gate:
        code, doc, _ = _json_child(_child_cmd("g8a", "--body", body, "--set", set_name))
        doc = doc if isinstance(doc, dict) else {}
        eps = doc.get("episodes") or []
        failure = None
        if code != 0 or not doc.get("ok"):
            failure = doc.get("problem") or f"the gate ended with exit code {code}"
        elif not doc.get("passes"):
            bad = next((e for e in eps if not e.get("identical")), None)
            if bad is None:
                failure = "no episode was run: the set holds no reference for the gate's condition"
            else:
                step = bad.get("first_difference")
                failure = dict(condition=bad.get("condition"), seed=bad.get("seed"), ms=bad.get("ms"),
                               step=bad.get("compared_steps") if step is None else step)
        line = line3(body, set_name, doc.get("identical") or 0, doc.get("of") or 0, doc.get("steps") or 0, failure)
        _say(line)
        res["lines"].append(line)
        res["g8a"][body] = dict(passes=failure is None, identical=doc.get("identical"), of=doc.get("of"),
                                steps=doc.get("steps"), form=doc.get("form"), episodes=eps, line=line)
        if failure is not None:
            res["problems"].append(f"G8a fails on {body}: no run")
            return res
    res["ok"] = True
    return res


# ------------------------------------------------------------------------------------------------
# This module's children: the steps that import setio, r5env, g8a (never at this module's top)
# ------------------------------------------------------------------------------------------------
def _child_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.nb child")
    sub = ap.add_subparsers(dest="what", required=True)
    a = sub.add_parser("read-inputs")
    a.add_argument("--set", required=True)
    a.add_argument("--body", required=True)
    a.add_argument("--root", default=None)
    a.add_argument("--files", default=None, help="a JSON list of paths relative to the set directory")
    sub.add_parser("archive-preflight")
    a = sub.add_parser("envs")
    a.add_argument("--set", required=True)
    a.add_argument("--pairs", required=True, help="a JSON list of [mode, body]")
    a = sub.add_parser("g8a")
    a.add_argument("--body", required=True)
    a.add_argument("--set", required=True)
    args = ap.parse_args(argv)
    tbenv.assign_gl("disable")  # (before any import below: nothing here may load a GL library)
    try:
        doc = {"read-inputs": _child_read_inputs, "archive-preflight": _child_archive, "envs": _child_envs,
               "g8a": _child_g8a}[args.what](args)
    except Exception as exc:  # noqa: BLE001 - the parent prints the reason in its line
        print(json.dumps(dict(ok=False, problem=_one_line(f"{type(exc).__name__}: {exc}")[:600])), flush=True)
        return 1
    print(json.dumps(_plain(doc)), flush=True)
    return 0 if doc.get("ok") else 1


def _child_read_inputs(args) -> dict:
    from designlab.stage2 import r5lib, setio

    if args.set == S1_SET:
        info, _ = setio.stage1_view(r5lib.s1_root(args.body), args.body)
    else:
        if args.files is None:
            return dict(ok=False, problem="read-inputs on a frozen set needs --files (the bundle's files of the set)")
        d = setio.set_dir(args.set, args.body, **({} if args.root is None else dict(root=args.root)))
        info, _ = setio.read_inputs(d, list(json.loads(args.files)), require_frozen=True)
    table = {f: v["sha256"] for f, v in info["files"].items()}
    return dict(ok=True, files=len(table), table_sha256=sha256_json(table), freeze=info.get("freeze"))


def _child_archive(args) -> dict:
    tbenv.repo_on_path()
    from environments.shared.policy_loading import sb3_archive_load_preflight

    seen = sb3_archive_load_preflight(None, None, trunk_dir=None)  # (no trunk: a throwaway archive of this runtime)
    return dict(ok=True, bytecode_members=sorted(seen.bytecode_members))


def _child_envs(args) -> dict:
    from dataclasses import replace

    import numpy as np

    from designlab.stage2 import r5env

    built = []
    for mode, body in json.loads(args.pairs):
        cfg = r5env.env_configs(mode, body, args.set, 1)[0]
        for floor in r5env.FLOORS:
            t0 = time.perf_counter()
            e = r5env.make(mode, replace(cfg, floor=floor))
            obs, _ = e.reset(seed=0)
            e.step(np.zeros(e.action_space.shape, np.float32))
            built.append(dict(mode=mode, body=body, floor=floor, build_s=round(time.perf_counter() - t0, 2),
                              obs_dim=int(np.asarray(obs).shape[0]), action_dim=int(e.action_space.shape[0])))
            e.close()
    return dict(ok=True, built=built)


def _child_g8a(args) -> dict:
    from designlab.stage2 import g8a

    if args.set == S1_SET:  # the development form: the env against the panels' R0 row on the stock walk/brisk
        eps = [g8a.stock(args.body, "walk/brisk", g8a.SEEDS[0], k) for k in g8a.DELAYS]
        g8a.release()
        doc = dict(form="stock", episodes=eps, identical=sum(1 for e in eps if e["identical"]), of=len(eps),
                   passes=bool(eps) and all(e["identical"] and e["compared_steps"] == e["reference_steps"]
                                            for e in eps), not_run=[])
    else:
        doc = dict(form="bar", **g8a.short(args.body, args.set))
    eps = [dict({k: e.get(k) for k in ("condition", "seed", "k", "identical", "first_difference", "compared_steps",
                                       "reference_steps", "ends_alike")},
                ms=2 * int(e["k"]))  # (a delay of k physics steps of 2 ms: 0 and 10 ms)
           for e in doc["episodes"]]
    return dict(ok=True, form=doc["form"], passes=bool(doc["passes"]), identical=int(doc["identical"]),
                of=int(doc["of"]), steps=int(sum(e["compared_steps"] or 0 for e in eps)), episodes=eps,
                not_run=doc.get("not_run") or [])


# ------------------------------------------------------------------------------------------------
# launch (cell 7): the queue, the Drive copies, the fence, the job rules
# ------------------------------------------------------------------------------------------------
class _Halt(Exception):
    """The run halts (``HALT.json``) with this reason."""


class _Interrupted(Exception):
    """The launcher was told to stop by a signal: the run is left as it is and can be resumed."""


class _Unfinished(Exception):
    """The session cannot go on, or cannot call the run done, for a fault of Drive (``UNFINISHED_RULE``): it ends
    with no DONE.json and no HALT.json, and the run can be resumed."""


def at_or_past(n: int, every, rollout) -> bool:
    """Whether checkpoint ``n`` is the first rollout start at or past a positive multiple of ``every`` (rule K1
    saves there; a rollout is ``rollout`` steps)."""
    every = int(every or 0)
    return every > 0 and n >= every and n % every < max(int(rollout or 0), 1)


def keep_on_drive(copied, checks) -> list:
    """Rule K2 on Drive: of the copied checkpoints, each check's and the newest two; ascending."""
    copied = sorted(set(int(n) for n in copied))
    return sorted({n for n in copied if n in set(checks)} | set(copied[-2:]))


def clip_trace(traces: Path) -> Path | None:
    """The trace a check's clip is drawn from (design 3.9): its ``walk/v0.4`` episode of seed 6004 when there is
    one, else seed 6004's (on the Stage 1 view: ``walk/brisk``), else the first trace in sorted order."""
    files = sorted(Path(traces).rglob("*.npz")) if Path(traces).is_dir() else []
    for want in (lambda p: p.name == "6004.npz" and p.parent.name == "v0.4", lambda p: p.name == "6004.npz"):
        hit = [p for p in files if want(p)]
        if hit:
            return hit[0]
    return files[0] if files else None


def _walk_text(jobdir: Path):
    """The text records of a job directory that go to Drive whenever they changed (``COPY_RULE``): paths relative
    to the job directory.  Checkpoints go by their own rule; traces and temporary files stay."""
    for base, dirs, names in os.walk(jobdir):
        rel = Path(base).relative_to(jobdir)
        if rel == Path("."):
            dirs[:] = [x for x in dirs if x not in ("checkpoints", "work", "partial")]
        dirs[:] = sorted(x for x in dirs if x not in ("traces",))
        for name in sorted(names):
            p = Path(base) / name
            if p.suffix in (".json", ".jsonl", ".txt", ".log"):
                yield rel / name
            elif p.suffix == ".mp4" and p.with_suffix(".json").is_file():
                yield rel / name


class _Launcher:
    """One session's run of the queue (``launch``)."""

    def __init__(self, st: dict, q: dict, workers: int, clips: bool, period: float):
        self.st, self.q, self.sess = st, q, st["session"]
        self.k, self.uuid = int(self.sess["session"]), self.sess["session_uuid"]
        self.d, self.l, self.same = Path(st["drive_run"]), Path(st["local_run"]), bool(st["same"])
        self.jobs = parse_queue(q["queue"])
        self.workers, self.clips, self.period = int(workers), bool(clips), float(period)
        self.params = self.sess.get("params") or {}
        self.pause = float(self.params.get("COPY_PAUSE_S") or 0.0)
        self.mine = {r["job"]: dict(state=None, retries=0, last_error=None, exit_codes=[], clip=None,
                                    last_checkpoint_copied=None, drive_errors=0, not_on_drive=[])
                     for r in self.jobs}
        self.copied: dict = {r["job"]: {} for r in self.jobs}   # job -> {steps: the cycle it was copied in}
        self.sent: dict = {}                                   # a copied text file -> (size, mtime_ns)
        self.cycle = 0
        self.child = self.child_log = None
        self.clip = None
        self.clip_tried: set = set()
        self.pack_tries: dict = {}
        self.packed: set = set()
        self.line4_done: set = set()
        self.pids: dict = {}                                   # job -> the pid of the trainer this launcher started
        self.echo_at: dict = {}
        self.stop = None
        (self.l / "logs").mkdir(parents=True, exist_ok=True)
        self.log = open(self.l / "logs" / f"launcher.s{self.k}.log", "a", encoding="utf-8")

    # ---------------------------------------------------------------- output and the live files
    def say(self, text: str) -> None:
        _say(text)
        try:
            self.log.write(f"{_now()} {text}\n")
            self.log.flush()
        except (OSError, ValueError):
            pass

    def fence(self) -> None:
        """``FENCE_RULE``: before every copy or delete on Drive."""
        by = _owner(self.d, self.uuid)
        if by is not None:
            raise _Superseded(by)

    def entries(self) -> list:
        return [_job_entry(self.st, r, self.mine[r["job"]]) for r in self.jobs]

    def beat(self, state: str = "running") -> None:
        self.fence()
        self.sess["last_status_at"] = _now()
        self.sess["lines"] = list(self.st["lines"])
        _save_state(self.st)
        _write_status(self.st, state, self.entries())
        _write_json(self.d / f"SESSION_{self.k}.json", self.sess)
        _write_live(self.st)

    def line(self, text: str) -> None:
        self.say(text)
        self.st["lines"].append(text)

    # ---------------------------------------------------------------- Drive copies (COPY_RULE)
    def _drive(self, job: str, what, *args):
        """Run one batch of Drive work; an ``OSError`` (a full Drive, a mount that went away) is counted and
        reported, and the job goes on: the next cycle tries again."""
        try:
            return what(*args)
        except OSError as exc:
            m = self.mine[job]
            m["drive_errors"] += 1
            m["last_error"] = f"Drive: {type(exc).__name__}: {exc}"
            self.say(f"S2 WARNING: a copy to Drive failed ({exc}); it is tried again in the next cycle")
            return None

    def _final(self, job: str) -> int | None:
        """A finished job's last checkpoint as this machine holds it (None: the job is not finished, or it saved
        none after step 0)."""
        jd = self.l / "jobs" / job
        if not _finished((_read_json(jd / "run.json") or {}).get("status")):
            return None
        return max((n for n in _drive_checkpoints(jd) if n > 0), default=None)

    def copy_text(self, job: str) -> None:
        src, dst = self.l / "jobs" / job, self.d / "jobs" / job
        if not src.is_dir():
            return
        final = self._final(job)
        hold = final is not None and final not in self.copied[job]
        for rel in _walk_text(src):
            if hold and rel.as_posix() in ("run.json", "progress.json"):
                continue  # (COPY_RULE: on Drive a job never reads finished before its last checkpoint is there)
            self._send(src / rel, dst / rel)

    def copy_logs(self) -> None:
        src = self.l / "logs"
        for p in sorted(src.glob("*.log")) if src.is_dir() else []:
            self._send(p, self.d / "logs" / p.name)

    def _send(self, a: Path, b: Path) -> None:
        try:
            stat = a.stat()
        except OSError:
            return
        key, sig = str(a), (stat.st_size, stat.st_mtime_ns)
        if self.sent.get(key) == sig:
            return
        self.fence()
        try:
            _copy_file(a, b)
        except FileNotFoundError:  # (the trainer replaced or removed it meanwhile: the next cycle sees the new one)
            if not a.exists():
                return
            raise
        self.sent[key] = sig

    def _record(self, base: Path, job: str, n: int) -> Path:
        return base / "jobs" / job / "checkpoints" / ("checkpoint_%d.json" % n)

    def _reads_back(self, job: str, n: int) -> bool:
        """Whether checkpoint ``n`` reads back from Drive whole ("copied", ``COPY_RULE``)."""
        return self._whole(self._record(self.d, job, n), n)

    @staticmethod
    def _whole(path: Path, n: int) -> bool:
        """Whether the checkpoint whose record is ``path`` reads whole: the record parses, is of step ``n``, and
        each of its three files beside it has the record's size and sha256."""
        rec = _read_json(path)
        if not isinstance(rec, dict) or not isinstance(rec.get("files"), dict) or rec.get("steps") != n:
            return False
        for e in rec["files"].values():
            f = path.parent / str(e.get("name"))
            try:
                if f.stat().st_size != e.get("bytes") or sha256_file(f) != e.get("sha256"):
                    return False
            except OSError:
                return False
        return True

    def _remove_on_drive(self, job: str, n: int) -> None:
        """Delete checkpoint ``n`` on Drive, the record first: at no moment a record stands beside a missing file."""
        path = self._record(self.d, job, n)
        rec = _read_json(path) or {}
        names = [str(e.get("name")) for e in (rec.get("files") or {}).values() if isinstance(e, dict)]
        names = names or ["model_%d.zip" % n, "vecnormalize_%d.pkl" % n, "fingerprint_%d.npz" % n]
        for f in [path] + [path.parent / x for x in names]:
            self.fence()
            _unlink(f)
            _unlink(_part(f))

    def _copy_checkpoint(self, job: str, n: int) -> bool:
        src = self._record(self.l, job, n)
        rec = _read_json(src)
        if not isinstance(rec, dict) or not isinstance(rec.get("files"), dict):
            return False
        dst = self._record(self.d, job, n)
        try:
            for e in rec["files"].values():  # the three binaries, each as <name>.part and renamed
                self.fence()
                _copy_file(src.parent / e["name"], dst.parent / e["name"], self.pause)
            self.fence()
            _copy_file(src, dst)             # the record, last
        except FileNotFoundError:            # (rule K2's local keep deleted it while it was being copied)
            self._remove_on_drive(job, n)
            return False
        if src.is_file() and sha256_file(dst) != sha256_file(src):
            return False
        return self._reads_back(job, n)

    def _kinds(self, job: str) -> dict:
        """What the job's own settings say about its checkpoints: ``{tiny, monitor_every, rollout, last, final}``
        (``last``: a finished job's step count; ``final``: its newest checkpoint here, ``_final``)."""
        run = _read_json(self.l / "jobs" / job / "run.json") or {}
        s = run.get("settings") if isinstance(run.get("settings"), dict) else {}
        rollout = int(s.get("n_envs") or N_ENVS) * int(s.get("n_steps") or 0)
        return dict(tiny=bool(s.get("tiny")), monitor_every=s.get("monitor_every"), rollout=rollout,
                    last=run.get("num_timesteps") if _finished(run.get("status")) else None, final=self._final(job))

    def _is_check(self, job: str, n: int, kinds: dict) -> bool:
        mon = self.l / "jobs" / job / "monitor" / ("monitor_%d.json" % n)
        return n > 0 and (mon.is_file() or at_or_past(n, kinds["monitor_every"], kinds["rollout"])
                          or n == kinds["last"])

    def copy_checkpoints(self, job: str) -> None:
        kinds, done = self._kinds(job), self.copied[job]
        for n in _drive_checkpoints(self.l / "jobs" / job):
            if n <= 0 or n in done:
                continue
            check = self._is_check(job, n, kinds)
            if not (kinds["tiny"] or at_or_past(n, COPY_EVERY, kinds["rollout"]) or check or n == kinds["final"]):
                continue
            if done and n < max(done) and not check:
                continue  # (older than what Drive holds and no check's: it would only be deleted again)
            if self._copy_checkpoint(job, n):
                done[n] = self.cycle
                self.mine[job]["last_checkpoint_copied"] = max(done)
        for n in sorted(done):
            if self._is_check(job, n, kinds) and (job, n) not in self.packed:
                self._pack(job, n)
        if not done:
            return
        newest = max(done)
        keep = keep_on_drive(done, [n for n in done if self._is_check(job, n, kinds)])
        old = [n for n in sorted(done) if n not in keep]
        # an older one goes only after a newer one was copied in an earlier cycle and reads back again in this one
        if old and done[newest] < self.cycle and self._reads_back(job, newest):
            for n in old:
                self._remove_on_drive(job, n)
                del done[n]

    def _packed(self, job: str, n: int) -> Path:
        return self.l / "work" / "packs" / job / ("ckpt_%d.tar" % n)

    def _can_pack(self, job: str, n: int) -> bool:
        """Whether a check's tar can still reach Drive: it is packed and waits for its copy, or the pack tool has
        tries left (``PACK_TRIES``; a tool that fails every time does not hold the run's end back)."""
        return self._packed(job, n).is_file() or self.pack_tries.get((job, n), 0) < PACK_TRIES

    def _pack(self, job: str, n: int) -> None:
        """``ckpt_<n>.tar`` of a check's checkpoint (``s2load pack``), copied to the job's directory on Drive.  A
        tar that is packed stays here until its copy has its sha256: a copy to Drive that fails (``OSError``, for
        ``_drive``) or reads back otherwise is made again in the next cycle and is no try of the pack tool."""
        tar = self._packed(job, n)
        if not tar.is_file():
            tries = self.pack_tries.get((job, n), 0)
            if tries >= PACK_TRIES or not self._record(self.l, job, n).is_file():
                return
            self.pack_tries[(job, n)] = tries + 1
            try:
                done = subprocess.run(_pack_cmd() + ["pack", str(self.l / "jobs" / job), str(n), "--out",
                                                     str(tar.parent)],
                                      cwd=str(study()), env=_child_env("disable"), capture_output=True, text=True,
                                      timeout=PACK_LIMIT_S, stdin=subprocess.DEVNULL)
                code, err = done.returncode, (done.stderr or done.stdout or "").strip().splitlines()[-1:]
            except (OSError, subprocess.TimeoutExpired) as exc:
                code, err = 124, [str(exc)]
            if code != 0 or not tar.is_file():
                self.mine[job]["last_error"] = f"ckpt_{n}.tar was not packed (exit code {code}): {' '.join(err)}"
                return
        self.fence()
        dst = self.d / "jobs" / job / tar.name
        _copy_file(tar, dst)
        if sha256_file(dst) != sha256_file(tar):
            return
        _unlink(tar)
        self.packed.add((job, n))

    def _adopt(self, job: str) -> None:
        """What Drive holds of a job's checkpoints is what was copied (an earlier session's work): each one that
        reads back whole, and its tar when it is there.  Raises ``OSError`` when Drive cannot be listed."""
        src = self.d / "jobs" / job
        for name in _listing(src / "checkpoints") or []:
            m = re.fullmatch(r"checkpoint_(\d+)\.json", name)
            if m and self._reads_back(job, int(m.group(1))):
                n = int(m.group(1))
                self.copied[job].setdefault(n, -1)
                if (src / ("ckpt_%d.tar" % n)).is_file():
                    self.packed.add((job, n))
        if self.copied[job]:
            self.mine[job]["last_checkpoint_copied"] = max(self.copied[job])

    def _from_drive(self, job: str, what, *args) -> None:
        """Run a read from Drive that the session cannot go on without.  An ``OSError`` is counted and said, and
        the read is made again, ``COPY_BACK_TRIES`` times in all; then the session ends unfinished
        (``UNFINISHED_RULE``): a fault of Drive is no halt of the run."""
        m, last = self.mine[job], None
        for attempt in range(COPY_BACK_TRIES):
            try:
                what(*args)
                return
            except OSError as exc:
                last = exc
                m["drive_errors"] += 1
                m["last_error"] = f"Drive: {type(exc).__name__}: {exc}"
                self.say(f"S2 WARNING: {job}: reading from Drive failed ({exc}); attempt {attempt + 1} of "
                         f"{COPY_BACK_TRIES}")
                if attempt + 1 < COPY_BACK_TRIES:
                    self._nap(min(self.period, 5.0))
        raise _Unfinished(f"Drive could not be read for job {job} ({last}): the job was not started, and nothing "
                          "was trained for it in this session")

    def copy_back(self, job: str) -> None:
        """``BACK_RULE``: on a resume in a new runtime the job's directory is copied back from Drive before the
        job is started (design 3.9), all or nothing; what Drive holds of its checkpoints is what was copied."""
        if not self.same:
            self._from_drive(job, self._copy_back, job)

    def _copy_back(self, job: str) -> None:
        src, dst = self.d / "jobs" / job, self.l / "jobs" / job
        self._adopt(job)
        if _listing(src) is None:
            return  # (Drive holds nothing of this job: it is new, or it was cut before its first copy)
        on_drive = sorted(n for n in self.copied[job] if n > 0)
        here = [n for n in _drive_checkpoints(dst) if n > 0 and self._whole(self._record(self.l, job, n), n)]
        if (dst / "run.json").is_file() and (not on_drive or (here and max(here) >= on_drive[-1])):
            return  # (this machine's own files are at least as new as their copies)
        tmp = self.l / "work" / "copy_back" / job
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        count = 0
        for base, dirs, names in os.walk(src, onerror=_walk_error):
            dirs.sort()
            for name in sorted(names):
                p = Path(base) / name
                if p.suffix in (".part", ".tar", ".mp4"):
                    continue
                _copy_file(p, tmp / p.relative_to(src))
                count += 1
        cut = [n for n in on_drive if not self._whole(tmp / "checkpoints" / ("checkpoint_%d.json" % n), n)]
        if cut:
            raise OSError(errno.EIO, f"checkpoint {cut[0]} did not read back whole after its copy from Drive")
        if dst.exists():  # (older than Drive's copy, or what a cut copy left: kept aside, never deleted)
            aside, i = self.l / "work" / "aside" / f"{job}.s{self.k}", 0
            while aside.exists():
                aside, i = aside.with_name(f"{job}.s{self.k}.{i + 1}"), i + 1
            aside.parent.mkdir(parents=True, exist_ok=True)
            os.rename(dst, aside)
            self.say(f"S2 NOTE: {job}: this machine's directory of the job was older than Drive's copy, or cut; it "
                     f"is kept aside as {aside.relative_to(self.l)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, dst)
        self.say(f"S2 launch: {job}: {count} files copied back from Drive")

    def _guard(self, job: str, resume: bool) -> None:
        """``BACK_RULE``'s last sentence: a job is never started while Drive holds a checkpoint of it that is
        newer than every one in this machine's directory of it.  The trainer would restart the job from step 0 in
        place, or go on from an older step, and say nothing that stops anyone."""
        if self.same:
            return
        on_drive = max((n for n in self.copied[job] if n > 0), default=None)
        if on_drive is None:
            return
        here = max((n for n in _drive_checkpoints(self.l / "jobs" / job) if n > 0), default=None)
        if not resume or here is None or here < on_drive:
            has = "no checkpoint" if here is None else f"none newer than {here}"
            raise _Unfinished(f"job {job} was not started: Drive holds its checkpoint {on_drive} and this machine's "
                              f"directory of the job holds {has}: a start here would restart the job from step 0 "
                              "or go on from an older step")

    def not_on_drive(self, job: str, tars: bool = True) -> list:
        """What a finished job still lacks on Drive (``COPY_RULE``'s end), by name: its last checkpoint and each
        check's that this machine holds, where it was not copied; with ``tars``, each check's tar, where it is not
        there and can still get there.  Empty for a job that is not finished, or of which this machine holds
        nothing."""
        if self.same:
            return []
        kinds = self._kinds(job)
        if kinds["final"] is None:
            return []
        have = self.copied[job]
        want = [n for n in _drive_checkpoints(self.l / "jobs" / job)
                if n > 0 and (n == kinds["final"] or self._is_check(job, n, kinds))]
        out = ["checkpoint %d" % n for n in want if n not in have]
        if not tars:
            return out
        return out + ["ckpt_%d.tar" % n for n in want if n in have and self._is_check(job, n, kinds)
                      and (job, n) not in self.packed and self._can_pack(job, n)]

    def _retry_pending(self, skip: str) -> None:
        """The copies that another job of this queue still lacks on Drive are tried again (``COPY_RULE``'s end)."""
        for r in self.jobs:
            job, m = r["job"], self.mine[r["job"]]
            if job == skip or not m.get("not_on_drive"):
                continue
            self._drive(job, self.copy_checkpoints, job)
            m["not_on_drive"] = self.not_on_drive(job)
            if not m["not_on_drive"]:
                self._drive(job, self.copy_text, job)
                self.say(f"S2 launch: {job}: its last copies are on Drive now")

    def settle(self, row: dict) -> None:
        """``COPY_RULE``'s end for a job that has ended: while its last checkpoint, a check's or a check's tar is
        not on Drive the copies are tried again, one period apart, ``END_COPY_TRIES`` times.  What is still
        missing then is said, kept in the job's entry (``not_on_drive``) and tried again in every later cycle of
        this session; the run is not written DONE without it."""
        job, m = row["job"], self.mine[row["job"]]
        if self.same:
            return
        m["not_on_drive"] = self.not_on_drive(job)
        waited = False
        for _ in range(END_COPY_TRIES):
            if not m["not_on_drive"]:
                break
            waited = True
            self._nap(self.period)
            self._copy_cycle(job)
            m["not_on_drive"] = self.not_on_drive(job)
        if m["not_on_drive"]:
            self.say(f"S2 WARNING: {job}: not on Drive after {END_COPY_TRIES} further tries: "
                     f"{', '.join(m['not_on_drive'])}.  It is tried again while this session lives; the run is not "
                     "written DONE without it")
            return
        self._drive(job, self.copy_text, job)  # (now with the job's run.json and progress.json)
        if waited:
            self.say(f"S2 launch: {job}: its last copies are on Drive now")

    def _nap(self, seconds: float) -> None:
        """Sleep, unless the launcher was told to stop."""
        end = time.time() + max(0.0, float(seconds))
        while True:
            if self.stop is not None:
                raise _Interrupted(self.stop)
            left = end - time.time()
            if left <= 0:
                return
            time.sleep(min(0.5, left))

    # ---------------------------------------------------------------- clips (CLIP_RULE)
    def clip_tick(self, row: dict) -> None:
        job = row["job"]
        c = self.clip
        if c is not None:
            code = c["proc"].poll()
            if code is None and time.time() - c["t0"] > CLIP_LIMIT_S:
                _kill_group(c["proc"].pid)
                c["proc"].wait()
                code = f"no clip within {CLIP_LIMIT_S:g} s"
            if code is not None:
                rec_path = c["out"].with_suffix(".json")
                rec = _read_json(rec_path)
                ok = isinstance(rec, dict) and bool(rec.get("ok")) and c["out"].is_file()
                why = None if ok else ((rec or {}).get("reason") or f"the clip child ended with {code!r}")
                if not isinstance(rec, dict):  # (the child left no record of its own: the launcher's)
                    _write_json(rec_path, dict(schema="s2-clip-launcher-1", ok=False, reason=why, at=_now(),
                                               trace=str(c["trace"]), rule=CLIP_RULE))
                self.mine[c["job"]]["clip"] = dict(at_steps=c["n"], ok=ok, reason=why,
                                                   seconds=round(time.time() - c["t0"], 1))
                self.say(f"S2 launch: clip of {c['job']} at {c['n']} steps: " + ("made" if ok else f"not made ({why})"))
                c["log"].close()
                self.clip = None
        if self.clip is not None or not self.clips:
            return
        mon = self.l / "jobs" / job / "monitor"
        checks = sorted((int(m.group(1)) for p in (mon.glob("monitor_*.json") if mon.is_dir() else [])
                         if (m := re.fullmatch(r"monitor_(\d+)\.json", p.name))), reverse=True)
        for n in checks[:1]:  # (the newest check's; an older check that got no clip is passed over)
            if (job, n) in self.clip_tried:
                continue
            self.clip_tried.add((job, n))
            trace, out = clip_trace(mon / "traces" / str(n)), mon / ("clip_%d.mp4" % n)
            if trace is None or out.with_suffix(".json").is_file():
                continue
            log = open(self.l / "logs" / f"clip.s{self.k}.log", "ab")
            cmd = _clip_cmd() + ["--trace", str(trace), "--body", row["body"], "--set", str(self.q["set"]), "--out",
                                 str(out)]
            try:
                proc = subprocess.Popen(cmd, cwd=str(study()), env=_child_env("osmesa"), stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, **_new_group())
            except OSError as exc:
                log.close()
                self.mine[job]["clip"] = dict(at_steps=n, ok=False, reason=str(exc), seconds=0.0)
                return
            self.clip = dict(proc=proc, job=job, n=n, t0=time.time(), out=out, trace=trace, log=log)
            return

    def finish_clip(self, row: dict) -> None:
        """At a job's end the clip child that is still drawing is waited for, within its limit."""
        while self.clip is not None:
            self.clip_tick(row)
            if self.clip is not None:
                time.sleep(0.2)

    # ---------------------------------------------------------------- one job
    def _cmd(self, row: dict, resume: bool) -> list:
        q, p = self.q, self.params
        cmd = _trainer_cmd() + ["run", "--mode", row["mode"], "--body", row["body"], "--seed", str(row["seed"]),
                                "--set", str(q["set"]), "--out", str(self.l), "--run-id", str(q["run_id"]),
                                "--workers", str(self.workers)]
        if q.get("tiny"):
            cmd.append("--tiny")
        if int(q.get("budget_steps") or 0) > 0:
            cmd += ["--steps", str(int(q["budget_steps"]))]
        if q.get("monitor_every") is not None:
            cmd += ["--monitor-every", str(int(q["monitor_every"]))]
        if q.get("monitor_also_k"):
            cmd += ["--monitor-also-k", ",".join(str(int(k)) for k in q["monitor_also_k"])]
        if q.get("lr") is not None:
            cmd += ["--lr", repr(float(q["lr"]))]
        if q.get("config"):
            cmd += ["--config", json.dumps(q["config"], sort_keys=True)]
        if resume:
            cmd.append("--resume")
            if p.get("RETRY_CHECK"):
                cmd.append("--retry-check")
        if self.sess.get("allow_version_change"):
            cmd.append("--allow-version-change")
        if self.sess.get("allow_source_change"):
            cmd.append("--allow-source-change")
        return cmd

    def _start(self, row: dict, resume: bool) -> None:
        job = row["job"]
        path = self.l / "logs" / f"{job}.s{self.k}.log"
        self.child_log = open(path, "ab")
        self.echo_at[job] = path.stat().st_size
        self.child = subprocess.Popen(self._cmd(row, resume), cwd=str(study()), env=_child_env("disable"),
                                      stdin=subprocess.DEVNULL, stdout=self.child_log, stderr=subprocess.STDOUT,
                                      **_new_group())
        self.pids[job] = self.child.pid
        self.st["child"] = dict(job=job, pid=self.child.pid, proc_start=_proc_start(self.child.pid),
                                boot_id=_boot_id(), host=socket.gethostname())
        _save_state(self.st)
        self.say(f"S2 launch: {job} started{' with --resume' if resume else ''} (pid {self.child.pid}, "
                 f"{self.workers} workers); its lines follow, indented")

    def _echo(self, job: str) -> None:
        """The trainer's new output lines, indented (its log is ``logs/<job>.s<k>.log``)."""
        path = self.l / "logs" / f"{job}.s{self.k}.log"
        try:
            with open(path, "rb") as fh:
                fh.seek(self.echo_at.get(job, 0))
                data = fh.read()
        except OSError:
            return
        cut = data.rfind(b"\n") + 1
        if cut:
            self.echo_at[job] = self.echo_at.get(job, 0) + cut
            for row in data[:cut].decode("utf-8", "replace").splitlines():
                _say("    " + row)

    def _line4(self, job: str) -> None:
        jd = self.l / "jobs" / job
        prog, rate = _read_json(jd / "progress.json") or {}, _read_json(jd / "gates" / "rate.json") or {}
        k = prog.get("session")
        # (rule N1: line 4 is this session's first rate.  A progress.json that the copy back brought, or that an
        # earlier session left on this machine, is not written by this launcher's trainer and names an older
        # session, whose rate gates/rate.json still holds)
        if k is None or prog.get("pid") != self.pids.get(job) or (job, k) in self.line4_done:
            return
        row = next((s.get("first_three") for s in rate.get("sessions") or [] if s.get("session") == k), None)
        budget = prog.get("budget") or (_read_json(jd / "run.json") or {}).get("budget")
        if not isinstance(row, dict) or not budget:
            return
        self.line4_done.add((job, k))
        self.line(line4(job, row["env_steps_per_wall_s"], row.get("n_envs", N_ENVS), row.get("workers", 0),
                        row.get("vcpus", _vcpus()), budget))

    def _told(self, job: str) -> None:
        """Rule M1's "the lead is told": a check counted without a record, and a halt, are said once."""
        prog = _read_json(self.l / "jobs" / job / "progress.json") or {}
        m = self.mine[job]
        for n in prog.get("checks_without_record") or []:
            if n not in m.setdefault("told", []):
                m["told"].append(n)
                self.say(f"S2 NOTE: {job}: the check at {n} steps was counted without a record (rule M1): tell the "
                         "lead")

    def cycle_once(self, row: dict) -> None:
        """One period's work (design 3.9): the fence, STATUS.json and LIVE.json, the text records, the
        checkpoints, one clip child, line 4."""
        job = row["job"]
        self.cycle += 1
        if self.sess.get("setup_seconds") is None and self.child is not None:
            prog = _read_json(self.l / "jobs" / job / "progress.json") or {}
            t0 = (self.sess.get("setup") or {}).get("started_epoch")
            if t0 and prog.get("pid") == self.child.pid:  # (rule Y7: from the setup cell to the first training
                self.sess["setup_seconds"] = round(time.time() - float(t0), 1)  # step of this session's trainer)
        self._line4(job)
        self._told(job)
        self.clip_tick(row)
        self._copy_cycle(job, count=False)

    def _copy_cycle(self, job: str, count: bool = True) -> None:
        """The Drive half of a period: the fence, the live files, the job's copies, and the copies another job
        still lacks (``COPY_RULE``'s end)."""
        if count:
            self.cycle += 1
        self.fence()
        self._drive(job, self.beat)
        if not self.same:
            self._drive(job, self.copy_text, job)
            self._drive(job, self.copy_checkpoints, job)
            self._retry_pending(job)
            self._drive(job, self.copy_logs)

    def _watch(self, row: dict) -> int:
        job, due = row["job"], time.time() + min(self.period, 5.0)
        while True:
            code = self.child.poll()
            self._echo(job)
            if self.stop is not None:
                raise _Interrupted(self.stop)
            if code is not None:
                self._echo(job)
                return code
            if time.time() >= due:
                self.cycle_once(row)
                due = time.time() + self.period
            time.sleep(min(0.5, max(self.period / 4.0, 0.02)))

    def _end_child(self) -> None:
        if self.child is not None:
            _kill_group(self.child.pid)
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            self.child = None
        if self.child_log is not None:
            self.child_log.close()
            self.child_log = None
        self.st["child"] = None

    def run_job(self, row: dict) -> None:
        """``JOB_RULE``."""
        job, m = row["job"], self.mine[row["job"]]
        jd = self.l / "jobs" / job
        self.copy_back(job)
        while True:
            resume = (jd / "run.json").is_file()
            self._guard(job, resume)
            self._start(row, resume)
            m["state"] = "running"
            code = self._watch(row)
            self._end_child()  # (whatever the trainer left behind: its workers, its monitor child)
            m["exit_codes"].append(code)
            status = (_read_json(jd / "run.json") or {}).get("status")
            if code == TRAIN_DONE:
                m["state"] = "done" if status == "done" else "stopped"
            elif code == TRAIN_G8A:
                m["state"] = "stopped"
                self.say(f"S2 launch: {job}: G8a fails: no run (the job was stopped by the gate; the next job starts)")
            else:
                m["state"] = "failed"
            self.cycle_once(row)
            self.finish_clip(row)
            if not self.same:
                self._drive(job, self.copy_text, job)
            if m["state"] != "failed":
                self.settle(row)
                self.say(f"S2 launch: {job} ended: {m['state']} ({status}) at "
                         f"{(_read_json(jd / 'progress.json') or {}).get('steps')} steps")
                return
            tail = self._tail(job)
            if code in (TRAIN_REFUSED, TRAIN_HALT):
                what = "was refused by the trainer" if code == TRAIN_REFUSED else "halted and goes to the lead"
                m["last_error"] = f"exit code {code}: {tail}"
                raise _Halt(f"job {job} {what} (exit code {code}): {tail}")
            m["last_error"] = f"crashed (exit code {code}): {tail}"
            if m["retries"] >= 1:
                raise _Halt(f"job {job} crashed twice (exit codes {', '.join(map(str, m['exit_codes'][-2:]))}): {tail}")
            m["retries"] += 1
            self.say(f"S2 NOTE: {job} crashed (exit code {code}); it is retried once from its newest intact checkpoint")

    def _tail(self, job: str) -> str:
        try:
            rows = (self.l / "logs" / f"{job}.s{self.k}.log").read_text("utf-8", "replace").strip().splitlines()
        except OSError:
            rows = []
        return _one_line(rows[-1])[:400] if rows else "no output"

    # ---------------------------------------------------------------- the queue
    def _prepare(self) -> None:
        stale = self.st.get("stale_child")
        if (isinstance(stale, dict) and stale.get("host") == socket.gethostname()
                and stale.get("boot_id") == _boot_id() and stale.get("proc_start") is not None
                and _proc_start(stale.get("pid")) == stale["proc_start"]):
            # an earlier launcher of this run on this machine died alone and its job lived on: two trainers never
            # write one job directory
            _kill_group(stale["pid"])
            self.say(f"S2 NOTE: the job process {stale['pid']} of an earlier session was still alive and was ended")
        self.st["stale_child"] = None
        was = _read_json(self.d / "STATUS.json") or {}
        for e in was.get("queue") or []:  # (the count of retries is the job's, across sessions)
            if isinstance(e, dict) and e.get("job") in self.mine:
                self.mine[e["job"]]["retries"] = int(e.get("retries") or 0)
        for r in self.jobs:
            e = _job_entry(self.st, r)
            if e["state"] in ("done", "stopped"):
                self.mine[r["job"]]["state"] = e["state"]
                if not self.same and (self.l / "jobs" / r["job"] / "run.json").is_file():
                    # (a job that ended here: what Drive holds of it, for COPY_RULE's end)
                    self._from_drive(r["job"], self._adopt, r["job"])
        if self.st["lines"]:
            _say("S2 launch: the lines so far:")
            for text in self.st["lines"]:
                _say(text)

    def _jobs_doc(self) -> dict:
        return {e["job"]: dict(state=e["state"], steps=e["steps"], last_checkpoint=e["last_checkpoint"],
                               last_checkpoint_copied=e["last_checkpoint_copied"], stopped=e["stopped"],
                               retries=e["retries"], exit_codes=e["exit_codes"], last_error=e["last_error"],
                               run_uuid=e["run_uuid"], not_on_drive=e["not_on_drive"]) for e in self.entries()}

    def _end(self, state: str, reason: str) -> dict:
        """The session's end: ``DONE.json`` or ``HALT.json`` (``END_SCHEMA``) or, for ``"unfinished"``, neither
        (``UNFINISHED_RULE``); the last STATUS.json; ``LIVE.json`` closed.  A DONE.json that cannot be written is
        no end of the run: the result is then ``"unfinished"`` too."""
        name = dict(done="DONE.json", halted="HALT.json").get(state)
        doc = _stamped(dict(schema=END_SCHEMA, run_id=self.sess["run_id"], session=self.k, session_uuid=self.uuid,
                            at=_now(), state=state, reason=reason, jobs=self._jobs_doc(), lines=list(self.st["lines"]),
                            commit=self.sess["repo"]["commit"],
                            rules=dict(jobs=JOB_RULE, halt=HALT_RULE, copy=COPY_RULE, unfinished=UNFINISHED_RULE)))
        written = False
        for attempt in range(3):
            try:
                self.fence()
                self.sess["ended"] = dict(at=doc["at"], state=state, reason=reason)
                self.beat("running" if name is None else state)  # (design 3.9: STATUS.json has no fourth state)
                if not self.same:
                    for r in self.jobs:
                        self.copy_text(r["job"])
                    self.copy_logs()
                if name is not None:
                    _write_json(self.d / name, doc)
                _write_live(self.st, closed=f"{state}: {reason}")
                written = True
                break
            except OSError as exc:
                self.say(f"S2 WARNING: the run's last files could not be written ({exc}); attempt {attempt + 1} of 3")
                time.sleep(min(self.period, 5.0))
        if state == "done" and not written:
            state, name = "unfinished", None
            reason = f"{reason}, but DONE.json could not be written to the run directory"
            self.sess["ended"] = dict(at=doc["at"], state=state, reason=reason)
        if name is None:
            self.say(f"S2 launch: UNFINISHED: {reason}.  The run has no DONE.json and no HALT.json and can be "
                     "resumed: tell the lead" + ("" if written else " (the run's last files could NOT be written "
                                                 "to the run directory)"))
        else:
            self.say(f"S2 launch: {name[:-5]}: {reason}" + ("" if written else f" ({name} could NOT be written to "
                                                             "the run directory: tell the lead)"))
        return dict(state=state, reason=reason, session=self.k, jobs=doc["jobs"], lines=doc["lines"],
                    record=name if written else None)

    def _superseded(self, by) -> dict:
        reason = f"superseded by session {by}"
        doc = _stamped(dict(schema=END_SCHEMA, run_id=self.sess["run_id"], session=self.k, session_uuid=self.uuid,
                            at=_now(), state="halted", reason=reason, jobs=self._jobs_doc(),
                            lines=list(self.st["lines"]), commit=self.sess["repo"]["commit"],
                            rules=dict(fence=FENCE_RULE, halt=HALT_RULE)))
        self.sess["ended"] = dict(at=doc["at"], state="halted", reason=reason)
        _save_state(self.st)
        if not self.same:
            _write_json(self.l / "HALT.json", doc)
        try:
            _write_json(self.d / f"SUPERSEDED_{self.k}.json", doc)
        except OSError:
            pass
        self.say(f"S2 launch: HALT: {reason}: this session stopped its job and writes nothing more to the run")
        return dict(state="halted", reason=reason, session=self.k, jobs=doc["jobs"], lines=doc["lines"],
                    record=f"SUPERSEDED_{self.k}.json")

    def run(self) -> dict:
        """The queue; never raises (a failure becomes ``HALT.json``; a fault of Drive ends the session unfinished
        and writes none, ``UNFINISHED_RULE``)."""
        try:
            try:
                self._prepare()
                self._drive(self.jobs[0]["job"], self.beat)
                for row in self.jobs:
                    if self.mine[row["job"]]["state"] in ("done", "stopped"):
                        self.say(f"S2 launch: {row['job']} is finished ({self.mine[row['job']]['state']}): skipped")
                        self.settle(row)  # (what an earlier session of this machine could not copy of it)
                        continue
                    self.run_job(row)
                lacks = [f"{r['job']}: {', '.join(self.mine[r['job']]['not_on_drive'])}" for r in self.jobs
                         if self.mine[r["job"]]["not_on_drive"]]
                if lacks:
                    raise _Unfinished("the queue is finished, but not all of it is on Drive (" + "; ".join(lacks)
                                      + "): what is missing is on this machine's disk only")
                return self._end("done", "the queue is finished")
            except _Superseded as exc:
                self._end_child()
                return self._superseded(exc.by)
            except _Interrupted as exc:
                self._end_child()
                self.say(f"S2 launch: interrupted by signal {exc}: the job was ended; the run is not finished and "
                         "can be resumed")
                if _owner(self.d, self.uuid) is None:
                    if not self.same:
                        for r in self.jobs:
                            self._drive(r["job"], self.copy_text, r["job"])
                    self._drive(self.jobs[0]["job"], _write_live, self.st, f"interrupted by signal {exc}")
                return dict(state="interrupted", reason=f"signal {exc}", session=self.k, jobs=self._jobs_doc(),
                            lines=list(self.st["lines"]), record=None)
            except _Unfinished as exc:
                self._end_child()
                return self._end("unfinished", str(exc))
            except _Halt as exc:
                self._end_child()
                return self._end("halted", str(exc))
            except BaseException as exc:  # noqa: BLE001 - never raises: a failure becomes HALT.json
                self._end_child()
                return self._end("halted", f"launcher error: {type(exc).__name__}: {_one_line(exc)[:400]}")
        except _Superseded as exc:  # (found while the end itself was written)
            return self._superseded(exc.by)
        except BaseException as exc:  # noqa: BLE001 - the end could not be written either
            self.say(f"S2 launch: HALT (not written to the run directory): {type(exc).__name__}: {exc}")
            return dict(state="halted", reason=f"launcher error: {type(exc).__name__}: {_one_line(exc)[:400]}",
                        session=self.k, jobs={}, lines=list(self.st["lines"]), record=None)
        finally:
            self._end_child()
            if self.clip is not None:
                _kill_group(self.clip["proc"].pid)
                self.clip["log"].close()
            _save_state(self.st)
            self.log.close()


def launch(run_id, drive, local, queue, set_name, budget, workers, tiny=False, clips=True, monitor_every=None,
           monitor_also_k=()) -> dict:
    """Cell 7: the queue of the session that ``session`` opened, one job at a time (``train2 run ... --out <local
    run directory> [--resume]`` as a child in its own process group, on ``workers`` workers; 0: one per vCPU, at
    most 8).  Every period: ``LIVE.json`` re-read (``FENCE_RULE``), ``STATUS.json`` and ``LIVE.json`` written,
    the text records and the checkpoints copied (``COPY_RULE``; nothing with ``drive == local``), one clip child
    (``CLIP_RULE``), **line 4** from ``gates/rate.json`` (``LINE4_RULE``), the job rules (``JOB_RULE``).  The
    queue, the set, the budget, ``tiny`` and the two monitor options must be ``QUEUE.json``'s (they go to every
    ``train2 run``).  Never raises: returns ``{state: "done" | "halted" | "unfinished" | "interrupted" |
    "refused", reason, session, jobs, lines, record}`` (``DONE.json`` or ``HALT.json`` is written for the first
    two; ``"unfinished"``: neither, for a fault of Drive, ``UNFINISHED_RULE``; a session that its own failed line
    2 or line 3 closed is refused)."""
    try:
        st = _open_state(run_id, drive, local)
        d, s = Path(st["drive_run"]), st["session"]
        q = _read_json(d / "QUEUE.json")
        if not isinstance(q, dict):
            raise Refused(f"{d}: no QUEUE.json that can be read")
        by = _owner(d, s["session_uuid"])
        if by is not None:
            raise Refused(f"{d}: the run belongs to session {by} now (LIVE.json): this session does not launch")
        for name in ("DONE.json", "HALT.json"):
            if (d / name).exists():
                raise Refused(f"{d}: the run is finished ({name}); nothing to launch")
        if q.get("smoke"):
            raise Refused(f"{d}: this run is a smoke: it queues no training job (rule Y10)")
        _refuse_closed(st, "launches nothing")
        mine = dict(queue=[r["entry"] for r in parse_queue(queue)], set=str(set_name), budget_steps=int(budget or 0),
                    tiny=bool(tiny), monitor_every=None if monitor_every is None else int(monitor_every),
                    monitor_also_k=[int(k) for k in (monitor_also_k or ())])
        diff = [f"{k}: {q.get(k)!r} -> {v!r}" for k, v in mine.items() if q.get(k) != v]
        if diff:
            raise Refused(f"{d}: the launch's arguments are not QUEUE.json's ({'; '.join(diff)})")
        period = (s.get("params") or {}).get("PERIOD_S")
        launcher = _Launcher(st, q, resolve_workers(workers), bool(clips),
                             PERIOD_S if period is None else float(period))
    except NbError as exc:
        _say(f"S2 REFUSED launch: {exc}")
        return dict(state="refused", reason=str(exc), session=None, jobs={}, lines=[], record=None)

    def stop(signum, _frame):
        launcher.stop = signum

    old = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old[sig] = signal.signal(sig, stop)
    except ValueError:  # (not the main thread: no handler; a test calls launch in a thread)
        pass
    try:
        return launcher.run()
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


# ------------------------------------------------------------------------------------------------
# finish (cell 8) and the runtime's release
# ------------------------------------------------------------------------------------------------
def finish(run_id, drive, local=None) -> dict:
    """Cell 8: the run's end as its files say it (``DONE.json`` or ``HALT.json``; a smoke's session ends with
    ``DONE.json`` here), a last copy of the text records while this session still owns the run, and the summary.
    ``local``: the local root (default: the one the run's newest session recorded).  Returns ``{state, reason,
    jobs, files, on_drive, not_on_drive}``; ``state`` is ``"done"``, ``"halted"``, ``"superseded"`` or
    ``"unfinished"``.  ``on_drive``: True when every finished job this machine holds has its last checkpoint and
    each check's on Drive (``COPY_RULE``'s end), False when one has not or Drive could not be read (``not_on_drive``
    names what is missing: it is on this machine's disk only), None when this session cannot tell (it does not own
    the run).  A write to Drive that fails here is said, never raised: the cell's ``finally`` runs either way."""
    d = run_dirs(run_id, drive, None)[0]
    if local is None:
        recs = sorted(d.glob("SESSION_*.json"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
        newest = _read_json(recs[-1]) if recs else None
        local = ((newest or {}).get("params") or {}).get("LOCAL_DIR") or drive
    try:
        st = _open_state(run_id, drive, local)
    except Refused:
        st = None
    owns = st is not None and _owner(d, st["session"]["session_uuid"]) is None
    lrun = Path(st["local_run"]) if st else None
    if st is not None and not owns:
        end = _read_json(lrun / "HALT.json") or _read_json(d / f"SUPERSEDED_{st['session']['session']}.json") or {}
        _say(f"S2 FINISH {run_id}: this session no longer owns the run ({end.get('reason') or 'superseded'}); "
             "nothing was copied")
        return dict(state="superseded", reason=end.get("reason"), jobs=end.get("jobs") or {}, files=str(d),
                    on_drive=None, not_on_drive={})
    if owns and st["session"].get("smoke") and not (d / "DONE.json").exists() and not (d / "HALT.json").exists():
        ok = (d / "smoke" / "SMOKE.json").is_file()
        doc = _stamped(dict(schema=END_SCHEMA, run_id=run_id, session=st["session"]["session"],
                            session_uuid=st["session"]["session_uuid"], at=_now(), state="done" if ok else "halted",
                            reason="the smoke is written (smoke/SMOKE.json)" if ok else "the smoke left no SMOKE.json",
                            jobs={}, lines=list(st["lines"]), commit=st["session"]["repo"]["commit"]))
        try:
            _write_json(d / ("DONE.json" if ok else "HALT.json"), doc)
            _note(st, state="done" if ok else "halted")
            _write_live(st, closed=doc["reason"])
        except OSError as exc:
            _say(f"S2 FINISH {run_id}: the smoke's end could not be written to the run directory ({exc})")
    on_drive, lacks = (True if owns else None), {}
    if owns and not st["same"]:
        launcher = _Launcher(st, _read_json(d / "QUEUE.json") or dict(queue=st["session"]["queue"]), 0, False, 1.0)
        try:
            for r in launcher.jobs:
                launcher._adopt(r["job"])
            for r in launcher.jobs:
                launcher.copy_text(r["job"])
            launcher.copy_logs()
            lacks = {r["job"]: launcher.not_on_drive(r["job"], tars=False) for r in launcher.jobs}
            lacks = {job: names for job, names in lacks.items() if names}
            on_drive = not lacks
        except (OSError, _Superseded) as exc:
            on_drive = False
            _say(f"S2 FINISH {run_id}: the last copy failed ({exc})")
        finally:
            launcher.log.close()
    done, halt = _read_json(d / "DONE.json"), _read_json(d / "HALT.json")
    end = done or halt
    if isinstance(end, dict):
        state, reason, jobs = ("done" if done else "halted"), end.get("reason"), end.get("jobs") or {}
    else:
        state, reason, jobs = "unfinished", "no DONE.json and no HALT.json: the run can be resumed", {}
    if st is not None and owns:
        st["session"]["last_status_at"] = _now()
        _save_state(st)
        try:
            _write_json(d / f"SESSION_{st['session']['session']}.json", st["session"])
        except OSError as exc:
            on_drive = False
            _say(f"S2 FINISH {run_id}: the session's record could not be written to the run directory ({exc})")
    _say(f"S2 FINISH {run_id}: {state.upper()}: {reason}")
    for job, e in jobs.items():
        _say(f"S2 FINISH   {job}: {e.get('state')} at {e.get('steps')} steps; last checkpoint "
             f"{e.get('last_checkpoint')}, copied {e.get('last_checkpoint_copied')}")
    if on_drive is False:
        what = ("; ".join(f"{job}: {', '.join(names)}" for job, names in lacks.items())
                or "Drive could not be read or written")
        _say(f"S2 FINISH {run_id}: NOT ON DRIVE: {what}.  It is on this machine's disk only: tell the lead")
    _say(f"S2 FINISH the run's files are under {d}")
    return dict(state=state, reason=reason, jobs=jobs, files=str(d), on_drive=on_drive, not_on_drive=lacks)


def release_runtime(reason: str, in_colab: bool, auto: bool, flush_drive: bool) -> None:
    """The final cell's flush and disconnect (design D19): outside Colab, or with ``auto`` False, only the reason
    is printed; on Colab the repository's ``disconnect_runtime`` is tried first and, whatever it raises, Drive is
    still flushed and the runtime unassigned; the runtime is unassigned whatever the flush itself raises."""
    if not (in_colab and auto):
        print(f"\n{reason}\nAuto-disconnect skipped (not in Colab or AUTO_DISCONNECT is False).", flush=True)
        return
    try:
        tbenv.repo_on_path()
        from environments.shared.notebook_runtime import disconnect_runtime

        disconnect_runtime(reason, in_colab=in_colab, auto=auto, flush_drive=flush_drive)
    except Exception as exc:  # noqa: BLE001 - importing the repository in the kernel may fail after the pins moved
        print(f"\n{reason}\n(the repository's disconnect_runtime was not used: {type(exc).__name__}: {exc})",
              flush=True)
        from google.colab import drive, runtime

        try:
            if flush_drive:
                print("Flushing Google Drive writes...", flush=True)
                drive.flush_and_unmount()
        except Exception as flush_exc:  # noqa: BLE001 - a Drive that cannot be flushed must not keep the runtime
            print(f"(Google Drive could not be flushed: {type(flush_exc).__name__}: {flush_exc})", flush=True)
        finally:
            print("Disconnecting runtime in 5 seconds...", flush=True)
            time.sleep(5)
            runtime.unassign()


# ------------------------------------------------------------------------------------------------
# collect (design D22) and hours
# ------------------------------------------------------------------------------------------------
_SKIP_DIRS = ("partial", "traces", "logs", "work")
KINDS = ("run", "bar", "r2l")


def _lock_alive(path: Path) -> str | None:
    """Who holds a round-A lock file (``s2io.Lock``) and is alive on this host; None otherwise."""
    doc = _read_json(path)
    if not isinstance(doc, dict) or doc.get("host") != socket.gethostname():
        return None
    if doc.get("boot_id") and _boot_id() and doc["boot_id"] != _boot_id():
        return None
    pid = doc.get("pid")
    if not isinstance(pid, int) or not _alive(pid):
        return None
    if doc.get("proc_start") is not None and _proc_start(pid) not in (None, doc["proc_start"]):
        return None
    return f"pid {pid} since {doc.get('started_at')}"


def running_problem(src) -> str | None:
    """Why ``src`` is still running (``COLLECT_RULE``), or None."""
    src = Path(src)
    ended = (src / "DONE.json").exists() or (src / "HALT.json").exists()
    age = _live_age(_read_json(src / "LIVE.json"))
    if not ended and age is not None and age < LIVE_FRESH_S:
        return f"LIVE.json was written {age:.0f} s ago and the run has no DONE.json or HALT.json"
    for lock in src.rglob(".lock"):
        who = _lock_alive(lock)
        if who:
            return f"{lock.relative_to(src)} is held by {who}"
    for run in src.rglob("run.json"):
        doc = _read_json(run) or {}
        prog = _read_json(run.with_name("progress.json")) or {}
        if doc.get("status") == "running" and prog.get("pid") and _alive(prog["pid"]):
            sess = (doc.get("sessions") or [{}])[-1]
            if sess.get("host") in (None, socket.gethostname()):
                return f"{run.relative_to(src)} reads running and its process {prog['pid']} is alive"
    return None


def collect_files(src) -> list:
    """The finished text records under ``src`` that ``collect`` copies, as sorted relative paths."""
    src, out = Path(src), []
    for base, dirs, names in os.walk(src):
        dirs[:] = sorted(x for x in dirs if x not in _SKIP_DIRS)
        rel = Path(base).relative_to(src)
        for name in sorted(names):
            if name in ("LIVE.json", "COLLECTED.json") or Path(name).suffix not in (".json", ".jsonl", ".txt"):
                continue
            if "checkpoints" in rel.parts and not re.fullmatch(r"checkpoint_\d+\.json", name):
                continue
            out.append((rel / name).as_posix())
    return out


def collect(src, dst, kind) -> dict:
    """Copy a finished job's text records from ``src`` to their committed place ``dst`` (``COLLECT_RULE``;
    ``kind``: ``"run"`` for a launcher's run directory, ``"bar"``, ``"r2l"``), verify each copy by sha256 and
    write ``<dst>/COLLECTED.json`` (the source, the files with their hashes, the run ids, the time).  ``Refused``
    when the source is still running, holds nothing, or ``dst`` holds another file under a name; a file already
    there with the same bytes is left.  Returns the record."""
    if kind not in KINDS:
        raise Refused(f"kind {kind!r}: choose from {', '.join(KINDS)}")
    src, dst = _anchored(src), _anchored(dst)
    if not src.is_dir():
        raise Refused(f"{src}: no such directory")
    why = running_problem(src)
    if why:
        raise Refused(f"{src} is still running ({why}): nothing was collected")
    rels = collect_files(src)
    if not rels:
        raise Refused(f"{src}: no text record to collect")
    other = [r for r in rels if (dst / r).is_file() and sha256_file(dst / r) != sha256_file(src / r)]
    if other:
        raise Refused(f"{dst} holds other files under {len(other)} of the names ({', '.join(other[:5])}"
                      f"{', ...' if len(other) > 5 else ''}): a collected record is never replaced; nothing was "
                      "collected")
    files, copied, left = {}, [], []
    for rel in rels:
        sha = sha256_file(src / rel)
        if (dst / rel).is_file():
            left.append(rel)
        else:
            _copy_file(src / rel, dst / rel)
            copied.append(rel)
        got = sha256_file(dst / rel)
        if got != sha:
            raise NbError(f"{dst / rel}: the copy has sha256 {got[:12]}..., the source {sha[:12]}...")
        files[rel] = dict(sha256=sha, bytes=(dst / rel).stat().st_size)
    uuids, unfinished = {}, []
    for rel in rels:
        if Path(rel).name == "run.json":
            doc = _read_json(src / rel) or {}
            uuids[str(doc.get("job") or Path(rel).parent.name)] = doc.get("run_uuid")
            if not _finished(doc.get("status")):
                unfinished.append(dict(job=doc.get("job"), status=doc.get("status")))
    end = _read_json(src / "DONE.json") or _read_json(src / "HALT.json") or {}
    rec = _stamped(dict(schema=COLLECTED_SCHEMA, kind=kind, source=str(src), run_id=end.get("run_id"),
                        end=dict(state=end.get("state"), reason=end.get("reason")) if end else None,
                        run_uuids=uuids, unfinished=unfinished, files=files, copied=copied, left=left,
                        collected_at=_now(), rule=COLLECT_RULE, **{k: v for k, v in repo_state().items()
                                                                  if k != "dirty_paths"}))
    _write_json(dst / "COLLECTED.json", rec)
    _say(f"S2 COLLECT {kind} {src} -> {dst}: {len(files)} files ({len(copied)} copied, {len(left)} already there), "
         f"each verified by sha256")
    return rec


def hours(runs_root, out="stage2/tb/hours.json") -> dict:
    """Runtime-hours per run and session (``HOURS_RULE``): every ``<runs_root>/<run_id>/SESSION_<k>.json`` from
    ``started_at`` to ``last_status_at``, summed into ``out`` (None: not written) against the cap.  The record."""
    root, runs, total, never = _anchored(runs_root), {}, 0.0, []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        rows = []
        for f in sorted(d.glob("SESSION_*.json"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
            s = _read_json(f) or {}
            a, b = _epoch(s.get("started_at")), _epoch(s.get("last_status_at"))
            h = 0.0 if a is None or b is None else max(0.0, b - a) / 3600.0
            if a is None or b is None:
                never.append(f"{d.name}/{f.name}")
            rows.append(dict(session=s.get("session"), session_uuid=s.get("session_uuid"),
                             started_at=s.get("started_at"), last_status_at=s.get("last_status_at"), hours=h,
                             in_colab=(s.get("runtime") or {}).get("in_colab"), smoke=s.get("smoke")))
        if rows:
            runs[d.name] = dict(sessions=rows, hours=sum(r["hours"] for r in rows))
            total += runs[d.name]["hours"]
    rec = _stamped(dict(schema=HOURS_SCHEMA, runs_root=str(root), runs=runs, total_hours=total, cap_hours=HOURS_CAP,
                        remaining_hours=HOURS_CAP - total, over_cap=total > HOURS_CAP, without_status=never,
                        rule=HOURS_RULE, at=_now()))
    if out is not None:
        _write_json(_anchored(out), rec)
    _say(f"S2 HOURS {root}: {len(runs)} runs, {total:.2f} of {HOURS_CAP:g} runtime-hours"
         + (" OVER THE CAP" if rec["over_cap"] else ""))
    return rec


# ------------------------------------------------------------------------------------------------
# smoke (cell 6; rules Y1-Y8)
# ------------------------------------------------------------------------------------------------
def budget_for(rate) -> int | None:
    """``BUDGET_RULE``: the budget rule Y11 gives the rate ``rate``; None under 15,000,000 (rule Y4 fails)."""
    steps = float(rate) * BUDGET_SECONDS
    if steps >= BUDGET_FULL:
        return BUDGET_FULL
    return None if steps < BUDGET_MIN else int(steps // BUDGET_STEP) * BUDGET_STEP


def _nb_step(step: str, params: dict, pfile: Path, **popen):
    """Run one step of this module in a child with ``params`` written to ``pfile``."""
    _write_json(pfile, params)
    cmd = [sys.executable, "-m", "designlab.stage2.nb", step, "--params", str(pfile)]
    return subprocess.Popen(cmd, cwd=str(study()), env=_child_env("disable"), stdin=subprocess.DEVNULL, **popen)


def _wait_until(what, limit_s: float, proc=None, every: float = 0.2):
    """Poll ``what()`` until it is truthy (returned), ``proc`` has ended (None) or ``limit_s`` passed (None)."""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        got = what()
        if got:
            return got
        if proc is not None and proc.poll() is not None:
            return what() or None
        time.sleep(every)
    return None


def smoke(run_id, drive, local) -> dict:
    """Cell 6 (rules Y1-Y8, in that order), on the body of the queue's first job (``75-M-H-10`` when the queue is
    empty) and the session's set; one line per item (``smoke_line``).  ``smoke/SMOKE.json`` is written after Y7
    and again after Y8.  Y3, Y4 and Y7 read one R5 job of four rollouts without gates; Y6 runs a tiny job through
    this module's own ``session`` and ``launch`` in a run directory under ``smoke/kill``, ends the launcher's whole
    session with SIGKILL when its first checkpoint is on Drive, deletes the local job directory and resumes in a
    new process; Y5 draws its clip from a trace of that job, taken after the resumed launch has ended (nothing
    writes a trace then); Y8 builds one body's arrays again in a scratch directory (capped at 40 minutes; "not
    finished" is an outcome) and, when every file was compared and equal, leaves ``REBUILD_OK.json`` with the
    arrays on Drive (rule N4).  Y9 is the lead's download, Y10 the pass rule: neither is decided here.  A session
    that its own failed line 2 or line 3 closed runs no smoke.  Returns the record."""
    st = _open_state(run_id, drive, local)
    _refuse_closed(st, "runs no smoke")
    s, d, lrun = st["session"], Path(st["drive_run"]), Path(st["local_run"])
    p = load_params(s["params"])
    jobs = parse_queue(p["QUEUE"])
    body, set_name, dev = (jobs[0]["body"] if jobs else SMOKE_BODY), str(p["SET"]), str(p["SET"]) == S1_SET
    work, out = lrun / "work" / "smoke", d / "smoke"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    items, lines = {}, []
    doc = dict(schema=SMOKE_SCHEMA, run_id=run_id, session=s["session"], session_uuid=s["session_uuid"], body=body,
               set=set_name, tiny=dev, items=items, lines=lines, rate=None, budget=None, written_after=None,
               rules=dict(budget=BUDGET_RULE, copy=COPY_RULE),
               note="Y9 (an archive that loads in the study's venv) is the lead's download; Y10 is the pass rule: "
                    "the smoke passes when Y1, Y3, Y6 and Y9 pass and Y4 passes or was answered by the owner's "
                    "budget.  With tiny true the items were read on the Stage 1 view (development): no smoke",
               commit=s["repo"]["commit"], versions=s["versions"], runtime=s["runtime"], seams=s.get("seams"))

    def item(i: int, name: str, value: str, verdict: str, **more) -> None:
        assert name == SMOKE_ITEMS[i], (i, name)
        text = smoke_line(i, name, value, verdict)
        items[f"Y{i}"] = dict(item=name, value=value, verdict=verdict, line=text, **more)
        lines.append(text)
        _say(text)
        _note(st, text)

    def write(after: str) -> None:
        doc.update(written_after=after, at=_now())
        if _owner(d, s["session_uuid"]) is None:
            _write_json(out / "SMOKE.json", _stamped(doc))

    # Y1, Y2: the setup and the machine
    rt, vers = s["runtime"], s["versions"]
    item(1, "setup", f"pins {'match' if s['pins_match'] else 'DO NOT match'}; torch {vers.get('torch')}; Python "
         f"{vers.get('python')}" + ("" if s["python_minor_matches"] else " (not 3.13: reported)")
         + f"; ffmpeg {s['ffmpeg'] or 'not found'}", "OK" if s["pins_match"] else "FAILED",
         pins_match=s["pins_match"], torch_build=s["torch_build"], python_minor_matches=s["python_minor_matches"])
    item(2, "machine", f"{rt['vcpus']} vCPUs, {rt['ram_gb']} GB RAM, {rt['cpu_model']}", "REPORTED", runtime=rt)

    # Y3, Y4, Y7: one R5 job of four rollouts without gates (its third update ends at its fourth rollout start)
    workers = resolve_workers(p["WORKERS"])
    n_steps = 64 if dev else 1024
    rate_root, job = work / "rate", "smoke-rate"
    cmd = _trainer_cmd() + ["run", "--mode", "R5", "--body", body, "--seed", "42", "--set", set_name, "--out",
                            str(rate_root), "--job", job, "--steps", str(4 * N_ENVS * n_steps), "--no-gates",
                            "--workers", str(workers), "--run-id", run_id] + (["--tiny"] if dev else [])
    (work / "logs").mkdir(exist_ok=True)
    jd = rate_root / "jobs" / job
    with open(work / "logs" / "rate.log", "ab") as log:
        proc = subprocess.Popen(cmd, cwd=str(study()), env=_child_env("disable"), stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, **_new_group())
        first = _wait_until(lambda: (jd / "progress.json").is_file() and time.time(), 3600.0, proc)
        code = proc.wait()
        _kill_group(proc.pid)
    t_setup = (s.get("setup") or {}).get("started_epoch")
    rate_doc = _read_json(jd / "gates" / "rate.json") or {}
    roll = ((rate_doc.get("sessions") or [{}])[0]).get("first_rollout") or {}
    rss = roll.get("rss_mb") or {}
    ups = []
    try:
        ups = [json.loads(x) for x in (jd / "updates.jsonl").read_text().splitlines() if x.strip()][:3]
    except (OSError, ValueError):
        pass
    tail = _one_line((work / "logs" / "rate.log").read_text("utf-8", "replace").strip().splitlines()[-1:] or "")[:300]
    if code != 0 or rss.get("trainer") is None:
        item(3, "memory", f"not measured: the rate job ended with exit code {code} ({tail})", "FAILED", exit_code=code)
    else:
        used = float(rss["trainer"]) + sum(float(x or 0) for x in rss.get("workers") or [])
        ram = float(rt["ram_gb"] or 0) * 1024.0
        ok = ram > 0 and used < RAM_SHARE * ram
        item(3, "memory", f"trainer {rss['trainer']:.0f} MB + {len(rss.get('workers') or [])} workers "
             f"{used - float(rss['trainer']):.0f} MB = {used:.0f} MB of {ram:.0f} MB RAM "
             f"({100.0 * used / ram if ram else 0:.0f} %, limit {100 * RAM_SHARE:.0f} %)", "OK" if ok else "FAILED",
             rss_mb=rss, ram_mb=ram, total_mb=used)
    if code != 0 or len(ups) < 3:
        item(4, "rate at 8 envs", f"not measured: the rate job ended with exit code {code} after {len(ups)} updates "
             f"({tail})", "FAILED", exit_code=code)
    else:
        steps, wall = int(ups[2]["steps"]), sum(float(u["wall_s"]) for u in ups)
        rate = steps / max(wall, 1e-9)
        budget = budget_for(rate)
        doc.update(rate=rate, budget=budget)
        item(4, "rate at 8 envs", f"{rate:.0f} env steps/s over {steps} steps (three rollouts and three updates, "
             f"{workers} workers); " + (f"budget {budget} steps" if budget is not None else
                                        f"under {BUDGET_MIN} steps in 20 h: no job is queued until the owner has set "
                                        "the budget"),
             "OK" if budget is not None else "FAILED", rate=rate, steps=steps, wall_s=wall, budget=budget,
             workers=workers, n_steps=n_steps, updates=ups)
    if _owner(d, s["session_uuid"]) is None:
        for rel in _walk_text(jd) if jd.is_dir() else []:
            _copy_file(jd / rel, out / "rate" / rel)

    # Y6 (and Y5's trace): a tiny job killed after its first checkpoint is on Drive, and resumed in a new process
    kdrive, klocal, krun = out / "kill", work / "kill-local", "y6"
    kp = dict(p, RUN_ID=krun, QUEUE=[f"R5:{body}:42"], DRIVE_DIR=str(kdrive), LOCAL_DIR=str(klocal), TINY=True,
              REPO_REF=s["repo"]["commit"],
              SMOKE=False, RESUME=False, TAKEOVER=False, BUDGET_STEPS=0, MONITOR_EVERY=None, MONITOR_ALSO_K=[],
              PERIOD_S=p["PERIOD_S"] if p["PERIOD_S"] is not None else 5.0, SETUP_RECORD=None)
    pfile, kjob = work / "y6.params.json", job_name("R5", body, 42)
    ck = kdrive / "runs" / krun / "jobs" / kjob
    trace_dir, kept_trace = klocal / krun / "jobs" / kjob / "monitor" / "traces", work / "clip" / "trace.npz"
    y6 = dict(ok=False, why=None, killed_at=None, resumed_from=None, fingerprint_max_abs_diff=None, steps=None)
    with open(work / "logs" / "y6.log", "ab") as log:
        both = dict(stdout=log, stderr=subprocess.STDOUT)
        if _nb_step("session", kp, pfile, **both).wait() != 0:
            y6["why"] = "the tiny run's session could not be opened"
        else:
            proc = _nb_step("launch", kp, pfile, start_new_session=True, **both)

            def on_drive():
                have = [n for n in _drive_checkpoints(ck) if n > 0]
                return min(have) if have else None

            n = _wait_until(on_drive, 1800.0, proc)
            kill_session(proc.pid)
            proc.wait()
            if n is None:
                y6["why"] = "no checkpoint of the tiny job reached Drive"
            else:
                y6["killed_at"] = dict(first_checkpoint_on_drive=n, on_drive=_drive_checkpoints(ck),
                                       progress=(_read_json(klocal / krun / "jobs" / kjob / "progress.json") or {}
                                                 ).get("steps"))
                shutil.rmtree(klocal / krun, ignore_errors=True)  # (the local job directory is deleted)
                kp2 = dict(kp, RESUME=True, TAKEOVER=True)  # (this process ended the launcher itself: no wait)
                if _nb_step("session", kp2, pfile, **both).wait() != 0:
                    y6["why"] = "the resume's session was refused"
                else:
                    _nb_step("launch", kp2, pfile, **both).wait()
                    # (Y5: a trace of the job, taken now: the resumed launch has ended, so nothing writes a trace
                    # any more, and the checks after the kill have left theirs)
                    t = clip_trace(trace_dir)
                    if t is not None:
                        kept_trace.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(t, kept_trace)
                    run = _read_json(klocal / krun / "jobs" / kjob / "run.json") or {}
                    last = (run.get("sessions") or [{}])[-1]
                    diff = last.get("fingerprint_max_abs_diff")
                    y6.update(resumed_from=last.get("resumed_from"), fingerprint_max_abs_diff=diff,
                              steps=run.get("num_timesteps"), status=run.get("status"),
                              skipped_checkpoints=last.get("skipped_checkpoints"),
                              done=(kdrive / "runs" / krun / "DONE.json").is_file())
                    y6["ok"] = bool(y6["done"] and run.get("status") == "done" and (last.get("resumed_from") or 0) > 0
                                    and diff is not None and float(diff) <= 1e-5
                                    and (run.get("num_timesteps") or 0) >= (run.get("budget") or 1))
                    if not y6["ok"]:
                        y6["why"] = "the resumed job did not reach its budget from a checkpoint on Drive"
    # Y5: a test clip of 3 s from that trace (reported: a clip is no stop condition)
    clip = dict(ok=False, reason="the tiny job left no trace" + (" (the Stage 1 view's stand-in writes none)"
                                                                 if dev else ""))
    if kept_trace.is_file():
        mp4 = work / "clip" / "clip_test.mp4"
        try:
            with open(work / "logs" / "clip.log", "ab") as log:
                done = subprocess.run(_clip_cmd() + ["--trace", str(kept_trace), "--body", body, "--set", set_name,
                                                     "--out", str(mp4), "--seconds", "3"], cwd=str(study()),
                                      env=_child_env("osmesa"), stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=CLIP_LIMIT_S)
            rec = _read_json(mp4.with_suffix(".json")) or {}
            clip = dict(ok=bool(rec.get("ok")) and mp4.is_file(), reason=rec.get("reason"), exit_code=done.returncode,
                        bytes=mp4.stat().st_size if mp4.is_file() else None)
        except (OSError, subprocess.TimeoutExpired) as exc:
            clip = dict(ok=False, reason=f"{type(exc).__name__}: {exc}")
        if _owner(d, s["session_uuid"]) is None:
            for f in (mp4, mp4.with_suffix(".json")):
                if f.is_file():
                    _copy_file(f, out / "clip" / f.name)
    item(5, "test clip", f"an MP4 of 3 s, {clip['bytes']} bytes" if clip["ok"] else f"not made: {clip['reason']}",
         "REPORTED", **clip)
    k = y6["killed_at"] or {}
    item(6, "kill and resume on Drive",
         (f"killed with checkpoint {k.get('first_checkpoint_on_drive')} on Drive; resumed from {y6['resumed_from']} "
          f"steps; fingerprint difference {y6['fingerprint_max_abs_diff']}; reached {y6['steps']} steps")
         if y6["ok"] else f"{y6['why']}", "OK" if y6["ok"] else "FAILED", **y6)
    secs = None if not (t_setup and first) else round(float(first) - float(t_setup), 1)
    item(7, "setup time", f"{secs} s from the setup cell to the first training step" if secs is not None
         else "not measured (no setup record, or no training step)", "REPORTED", seconds=secs)
    write("Y7")

    # Y8: the inputs rebuilt once (last; capped)
    y8 = _smoke_rebuild(p, body, work, drive, run_id)
    item(8, "inputs rebuilt once", y8.pop("value"), "REPORTED", **y8)
    write("Y8")
    return doc


def _smoke_rebuild(p: dict, body: str, work: Path, drive, run_id) -> dict:
    """Rule Y8: one body's arrays built again in a scratch directory from the committed records, each sha256
    against the bundle's manifest; capped at ``REBUILD_CAP_S`` ("not finished" is an outcome).  When every file
    was compared and equal, the arrays and ``REBUILD_OK.json`` go to Drive (rule N4's permission)."""
    bundle_id = str(p["BUNDLE_ID"] or "")
    doc = _read_json(_root() / BUNDLE_DIR / bundle_id / "MANIFEST.json") if bundle_id else None
    content = doc.get("content") if isinstance(doc, dict) else None
    if not isinstance(content, dict):
        return dict(value="not run: no bundle (the Stage 1 view has none)" if not bundle_id else
                    f"not run: the manifest of bundle {bundle_id} cannot be read", finished=False, allowed=False)
    files, t0 = content.get("files") or {}, time.time()
    scratch = work / "rebuild" / "stage2" / "sets"
    total = compared = equal = 0
    finished, codes = True, {}
    for name in sorted(content.get("sets") or {}):
        if body not in content["sets"][name]:
            continue
        head = f"stage2/sets/{name}/{body}/"
        src = _root() / head
        for rec in sorted(src.glob("*.json")) if src.is_dir() else []:  # (the committed records, the freeze included)
            _copy_file(rec, scratch / name / body / rec.name)
        left = REBUILD_CAP_S - (time.time() - t0)
        code = None
        if left > 0:
            with open(work / "logs" / "rebuild.log", "ab") as log:
                proc = subprocess.Popen(_rebuild_cmd(name, body, scratch), cwd=str(study()), env=_child_env("disable"),
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **_new_group())
                try:
                    code = proc.wait(timeout=left)
                except subprocess.TimeoutExpired:
                    code = None
                _kill_group(proc.pid)
        codes[name] = code
        finished = finished and code == 0
        for rel in sorted(f for f in files if f.startswith(head)):
            total += 1
            got = scratch / name / body / rel[len(head):]
            if got.is_file():
                compared += 1
                equal += sha256_file(got) == files[rel]["sha256"]
    models = [f for f, e in files.items() if e.get("part") == f"models-{body}.tar"]
    models_ok = sum((_root() / f).is_file() and sha256_file(_root() / f) == files[f]["sha256"] for f in models)
    allowed = bool(total) and finished and compared == total and equal == total and models_ok == len(models)
    secs = round(time.time() - t0, 1)
    if allowed:
        cache = rebuilt_dir(drive, bundle_id)
        for name in sorted(content.get("sets") or {}):
            head = f"stage2/sets/{name}/{body}/"
            for rel in sorted(f for f in files if f.startswith(head)):
                _copy_file(scratch / name / body / rel[len(head):], cache / rel)
        _write_json(cache / "REBUILD_OK.json", _stamped(dict(
            schema="s2-rebuild-ok-1", allowed=True, bundle_id=bundle_id, content_sha256=doc.get("content_sha256"),
            body=body, files=total, smoke_run_id=run_id, seconds=secs, at=_now(),
            rule="rule N4: the arrays may be rebuilt on Colab only after rule Y8 found every hash equal")))
    value = (f"{compared} of {total} arrays of {body} compared, {equal} equal; {models_ok} of {len(models)} model "
             f"files equal; {secs:.0f} s" + ("" if finished else f"; not finished (cap {REBUILD_CAP_S:g} s)")
             + ("; the rebuild is an allowed fallback (rule N4)" if allowed else ""))
    return dict(value=value, finished=finished, allowed=allowed, total=total, compared=compared, equal=equal,
                models=len(models), models_equal=models_ok, seconds=secs, exit_codes=codes)


# ------------------------------------------------------------------------------------------------
# The notebook's cells run as Python (local_run, acceptance)
# ------------------------------------------------------------------------------------------------
#: The stand-in for the notebook's kernel: the named cells executed in order in one namespace; a cell that raises
#: ends it, as it ends "Run all".
_KERNEL = (
    "import json, sys\n"
    "book = json.load(open(sys.argv[1], encoding='utf-8'))\n"
    "space = {'__name__': '__main__'}\n"
    "for i in [int(x) for x in sys.argv[2].split(',')]:\n"
    "    cell = book['cells'][i]\n"
    "    if cell['cell_type'] == 'code':\n"
    "        print(f'--- cell {i}', flush=True)\n"
    "        exec(compile(''.join(cell['source']), f'<cell {i}>', 'exec'), space)\n")


def run_cells(notebook, params: dict, log, cwd=None, cells=RUN_CELLS, until=None, python=None,
              limit_s: float = 7200.0) -> dict:
    """Execute the notebook's ``cells`` as Python in a child (a session of its own; ``S2_PARAMS`` names
    ``params``; working directory ``cwd``, default the study).  ``until``: a function polled while the child runs;
    when it returns something true the child's whole session is ended with SIGKILL (a dead runtime) and that
    value is returned as ``killed``.  A relative ``notebook`` is the working directory's own.  Returns
    ``{exit_code, killed, seconds, text, lines, log}``."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    pfile = log.with_suffix(".params.json")
    _write_json(pfile, params)
    env = dict(os.environ, S2_PARAMS=str(pfile), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    t0, killed = time.time(), None
    with open(log, "wb") as fh:
        book = Path(notebook) if Path(notebook).is_absolute() else Path(cwd or study()) / notebook
        proc = subprocess.Popen([python or sys.executable, "-c", _KERNEL, str(book),
                                 ",".join(str(int(c)) for c in cells)], cwd=str(cwd or study()), env=env,
                                stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                if until is not None:
                    killed = until()
                    if killed:
                        break
                if time.time() - t0 > limit_s:
                    killed = f"no end within {limit_s:g} s"
                    break
                time.sleep(0.05)
        finally:
            kill_session(proc.pid)  # (also after a clean end: nothing of the session may live on)
            code = proc.wait()
    text = log.read_text("utf-8", "replace")
    return dict(exit_code=code, killed=killed, seconds=round(time.time() - t0, 1), text=text, lines=lines_of(text),
                log=str(log))


def _drill(notebook, params: dict, logs: Path, cwd_of, kill_when, wipe=None) -> dict:
    """The kill and the resume both ``local_run`` and ``acceptance`` make: session 0 run until ``kill_when()``
    is true and then ended with SIGKILL (its whole session); the local run directory deleted (and ``wipe()``
    called: a dead runtime loses its disk); a RESUME at once, which must be refused because ``LIVE.json`` is
    fresh; a RESUME with TAKEOVER, which must reach DONE; a further RESUME, which must be refused because the
    run is finished.  ``cwd_of(i)`` gives the working directory of session attempt ``i``.  Returns the attempts
    and what the run directory holds."""
    d, lrun, _ = run_dirs(params["RUN_ID"], params["DRIVE_DIR"], params["LOCAL_DIR"])
    first = parse_queue(params["QUEUE"])[0]["job"]
    a = run_cells(notebook, params, logs / "session0.log", cwd_of(0), until=kill_when)
    kill = dict(at=_now(), after_s=a["seconds"], when=a["killed"],
                progress=(_read_json(lrun / "jobs" / first / "progress.json") or {}).get("steps"),
                on_drive=_drive_checkpoints(d / "jobs" / first),
                parts_on_drive=sorted(p.relative_to(d).as_posix() for p in d.rglob("*.part")) if d.is_dir() else [])
    shutil.rmtree(lrun, ignore_errors=True)
    if wipe is not None:
        wipe()
    b = run_cells(notebook, dict(params, RESUME=True), logs / "session1-refused.log", cwd_of(1))
    c = run_cells(notebook, dict(params, RESUME=True, TAKEOVER=True), logs / "session1.log", cwd_of(2))
    e = run_cells(notebook, dict(params, RESUME=True), logs / "session2-refused.log", cwd_of(3))
    return dict(drive_run=str(d), local_run=str(lrun), first_job=first, kill=kill,
                attempts=dict(session0=a, refused_live=b, session1=c, refused_finished=e))


def _drill_checks(drill: dict, params: dict) -> dict:
    """What a drill must show, as ``{name: bool}`` (design 3.10 step 7, without what needs a real set)."""
    d = Path(drill["drive_run"])
    at = drill["attempts"]
    a, b, c, e = at["session0"], at["refused_live"], at["session1"], at["refused_finished"]
    jobs = parse_queue(params["QUEUE"])
    bodies = list(dict.fromkeys([r["body"] for r in jobs] + list(params.get("PREFLIGHT_BODIES") or [])))
    status, done = _read_json(d / "STATUS.json") or {}, _read_json(d / "DONE.json") or {}
    s0, s1 = _read_json(d / "SESSION_0.json") or {}, _read_json(d / "SESSION_1.json") or {}
    runs = {r["job"]: _read_json(d / "jobs" / r["job"] / "run.json") or {} for r in jobs}
    first = runs[drill["first_job"]]
    resumed = [x.get("resumed_from") for x in first.get("sessions") or [] if x.get("resumed_from")]

    def once(x, kind) -> list:  # (lines 1-3 are printed by their step and again above the launch)
        return list(dict.fromkeys(x["lines"][kind]))

    out = dict(
        killed=bool(a["killed"]) and a["exit_code"] != 0,
        line1=len(once(a, 1)) == 1 and len(once(c, 1)) == 1,
        line2_ok=all(len(once(x, 2)) == 1 and once(x, 2)[0].endswith(": OK") for x in (a, c)),
        line3_pass=all(len(once(x, 3)) == len(bodies) and all(t.endswith(": PASS") for t in once(x, 3))
                       for x in (a, c)),
        line4_after_resume=len(c["lines"][4]) >= 1,
        new_run_line=len(once(a, "new")) == 1,
        refused_live=b["exit_code"] != 0 and len(b["lines"]["live"]) == 1 and not b["lines"][2],
        resume_line=len(once(c, "resume")) == 1,
        session1_done=c["exit_code"] == 0 and done.get("state") == "done",
        refused_finished=e["exit_code"] != 0 and "is finished" in e["text"] and not e["lines"]["resume"],
        session_files=s0.get("schema") == SESSION_SCHEMA and s1.get("schema") == SESSION_SCHEMA
        and s1.get("resume") is True and s1.get("takeover") is True and s0.get("resume") is False
        and not (d / "SESSION_2.json").exists(),
        status_schema=status.get("schema") == STATUS_SCHEMA and status.get("state") == "done"
        and [x.get("job") for x in status.get("queue") or []] == [r["job"] for r in jobs],
        jobs_done=all(r.get("status") == "done" for r in runs.values()),
        resumed_from_drive=bool(resumed) and all(n > 0 for n in resumed)
        and first.get("run_uuid") is not None and len(first.get("sessions") or []) >= 2,
        checkpoints_copied=all(bool(_drive_checkpoints(d / "jobs" / r["job"])) for r in jobs),
        no_part_left=not list(d.rglob("*.part")))
    return out


def local_run(notebook, drive_dir, params, record=None) -> dict:
    """Development (design 3.10's last sentence): the notebook's cells 1-5 and 7-8 executed as Python in this
    checkout with ``S2_PARAMS``, on the Stage 1 view with ``--tiny`` and a directory standing in for Drive
    (``drive_dir``); the launcher's session ended with SIGKILL once the first job's first checkpoint is on the
    stand-in Drive; the local run directory deleted; then ``_drill``'s resumes.  ``params`` replaces the defaults
    (a queue of a tiny R5 and a tiny R6 job on ``75-M-H-10``, 2 workers, a 2 s period); a run that is not tiny is
    refused.  Returns the record (``LOCAL_RUN_SCHEMA``; written to ``record`` when given): the attempts' lines,
    the kill point, the checks, the seconds and whether ``git status --porcelain`` of this checkout changed."""
    drive = _anchored(drive_dir).resolve()
    head = repo_state()
    p = dict(RUN_ID="local-" + time.strftime("%Y%m%d-%H%M%S"), QUEUE=["R5:75-M-H-10:42", "R6:75-M-H-10:42"],
             SET=S1_SET, TINY=True, WORKERS=2, CLIPS=True, PERIOD_S=2.0, REPO_REF=head["commit"], BUNDLE_ID="",
             USE_GOOGLE_DRIVE=False, AUTO_DISCONNECT=False, DRIVE_DIR=str(drive),
             LOCAL_DIR=str(drive.parent / (drive.name + "-local")), STUDY_DIR=str(study()))
    p.update(params or {})
    p = load_params(p)
    if not p["TINY"]:
        raise Refused("local_run is development: its jobs are tiny (TINY)")
    logs = drive.parent / (drive.name + "-logs")
    before = _git("status", "--porcelain")
    d, _, _ = run_dirs(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"])
    first = parse_queue(p["QUEUE"])[0]["job"]

    def copied():
        have = [n for n in _drive_checkpoints(d / "jobs" / first) if n > 0]
        return f"checkpoint {min(have)} of {first} is on the stand-in Drive" if have else None

    t0 = time.time()
    drill = _drill(notebook, p, logs, lambda i: study(), copied)
    checks = _drill_checks(drill, p)
    after = _git("status", "--porcelain")
    checks["git_status_unchanged"] = before is not None and before == after
    rec = _stamped(dict(
        schema=LOCAL_RUN_SCHEMA, ok=all(checks.values()), checks=checks, params=p, kill=drill["kill"],
        drive_run=drill["drive_run"], seconds=round(time.time() - t0, 1),
        attempts={k: dict(exit_code=v["exit_code"], killed=v["killed"], seconds=v["seconds"], log=v["log"],
                          lines={str(n): rows for n, rows in v["lines"].items() if rows})
                  for k, v in drill["attempts"].items()},
        git_status_before=before, git_status_after=after, commit=head["commit"], dirty=head["dirty"], at=_now(),
        note="development on the Stage 1 view; nothing here is a record of a run"))
    if record is not None:
        _write_json(_anchored(record), rec)
    _say(f"S2 LOCAL RUN {p['RUN_ID']}: {'OK' if rec['ok'] else 'FAILED'} in {rec['seconds']:.0f} s"
         + ("" if rec["ok"] else ": " + ", ".join(k for k, v in checks.items() if not v)))
    return rec


#: The file that marks a directory as an acceptance run's scratch: only such a directory is ever deleted.
ACCEPTANCE_MARK = "S2_ACCEPTANCE_SCRATCH"


def g8a_counts(lines) -> dict:
    """Line 3's counts per body as the printed lines give them: ``{body: {set, identical, of, steps, passes}}``
    (the acceptance run's record: design 3.10, "the G8a counts per body")."""
    out = {}
    for text in lines:
        m = LINE_RE[3].match(text)
        if m:
            out[m.group("body")] = dict(set=m.group("set"), identical=int(m.group("identical")),
                                        of=int(m.group("of")), steps=int(m.group("steps")),
                                        passes=m.group("end") == "PASS")
    return out


def acceptance(notebook, ref, parts_dir, scratch, record) -> dict:
    """The acceptance run (design 3.10; rule N5): a fresh clone of the launch commit ``ref`` that holds nothing
    git ignores, the bundle's parts (``parts_dir``: ``stage2/tb/bundle/<bundle_id>/parts``) in a directory standing
    in for Drive under ``scratch`` (outside the checkout), the notebook's cells 1-5 and 7-8 with a tiny R5 and a
    tiny R6 job on ``75-M-H-10`` and set V6, preflight over every body of the bundle; the launcher's session ended
    with SIGKILL at the moment a ``.part`` file exists on the stand-in Drive; the clone and the whole local
    directory deleted (a dead VM loses both: the run directory and the inputs' local copies); a second clone;
    ``_drill``'s resumes; then the record ``ACCEPTANCE.json`` (``ACCEPTANCE_SCHEMA``).  ``scratch`` is deleted
    first, so it must be new, empty or an earlier acceptance run's own (``ACCEPTANCE_MARK``); any other directory
    is refused.

    It needs the pushed commit and the built bundle and is the lead's step: **this function was written with the
    package and has not been run** (its drill is ``local_run``'s, which has)."""
    ref, scratch = str(ref), _anchored(scratch).resolve()
    if not _HEX40.match(ref):
        raise Refused(f"--ref {ref!r}: the 40 hex digits of the launch commit (rule K6)")
    top = Path((_git("rev-parse", "--show-toplevel") or "").strip() or study())
    if scratch == top or top in scratch.parents:
        raise Refused(f"--scratch {scratch} is inside the checkout {top}: the clones are made outside it")
    if scratch.exists() and (not scratch.is_dir() or (any(scratch.iterdir())
                                                      and not (scratch / ACCEPTANCE_MARK).is_file())):
        raise Refused(f"--scratch {scratch} exists, is not empty and is not an earlier acceptance run's (it holds "
                      f"no {ACCEPTANCE_MARK}): the scratch directory is deleted before the run, so name a new one")
    parts_dir = _anchored(parts_dir).resolve()
    bundle_id = parts_dir.parent.name
    manifest = _read_json(parts_dir.parent / "MANIFEST.json")
    content = (manifest or {}).get("content")
    if not parts_dir.is_dir() or not isinstance(content, dict):
        raise Refused(f"{parts_dir}: not the parts directory of a bundle with its MANIFEST.json beside it")
    prefix = (_git("rev-parse", "--show-prefix") or "").strip()
    url = (_git("remote", "get-url", "origin") or "").strip()
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    listed = subprocess.run(["git", "-C", str(top), "ls-remote", "origin"], capture_output=True, text=True, env=env)
    on_remote = [x.split("\t")[1] for x in listed.stdout.splitlines() if x.startswith(ref + "\t")] \
        if listed.returncode == 0 else []
    source = url
    if not on_remote:
        tracking = [x.strip() for x in (_git("branch", "-r", "--contains", ref, cwd=top) or "").splitlines()]
        if listed.returncode == 0 or not tracking:
            raise Refused(f"{ref} is not the head of a branch or tag of the remote {url}: the acceptance run needs "
                          "the pushed launch commit")
        source = str(top)  # (the remote cannot be reached from here: the local repository's remote-tracking ref)
    before = _git("status", "--porcelain")
    shutil.rmtree(scratch, ignore_errors=True)  # (new, empty or marked as this function's own: checked above)
    scratch.mkdir(parents=True)
    (scratch / ACCEPTANCE_MARK).write_text(f"the scratch directory of nb acceptance --ref {ref}; made {_now()}\n",
                                           encoding="utf-8")
    drive = scratch / "drive"
    for name in sorted(content["parts"]):
        _copy_file(parts_dir / name, drive / "inputs" / bundle_id / name)

    fresh, heads = {}, {}

    def clone(i: int) -> Path:
        dst = scratch / f"clone{i}"
        shutil.rmtree(dst, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", source, str(dst)], check=True, env=env)
        subprocess.run(["git", "-C", str(dst), "checkout", "--quiet", "--detach", ref], check=True, env=env)
        # a fresh clone holds nothing git ignores, and nothing else that is not the commit's
        status = _git("status", "--porcelain", "--ignored", cwd=dst)
        fresh[i] = status is not None and not status.strip()
        heads[i] = (_git("rev-parse", "HEAD", cwd=dst) or "").strip() or None
        return dst / prefix

    def dead_vm() -> None:
        """Design 3.10 step 4: a dead VM loses its clone and its whole local disk (``_drill`` has deleted the
        local run directory; the parts ``inputs`` copied under ``local/inputs`` go too, so that the resume takes
        them from the stand-in Drive again)."""
        shutil.rmtree(scratch / "clone1", ignore_errors=True)
        shutil.rmtree(scratch / "local", ignore_errors=True)

    clones = {0: clone(1)}
    p = load_params(dict(
        RUN_ID="acceptance", QUEUE=["R5:75-M-H-10:42", "R6:75-M-H-10:42"], SET="V6", TINY=True, WORKERS=2, CLIPS=True,
        PERIOD_S=2.0, COPY_PAUSE_S=0.5, REPO_REF=ref, BUNDLE_ID=bundle_id, BUNDLE_SOURCE="drive",
        USE_GOOGLE_DRIVE=False, AUTO_DISCONNECT=False, DRIVE_DIR=str(drive), LOCAL_DIR=str(scratch / "local"),
        PREFLIGHT_BODIES=list(content.get("bodies") or [])))
    d, _, _ = run_dirs(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"])
    first = parse_queue(p["QUEUE"])[0]["job"]

    def part_exists():
        ck = d / "jobs" / first / "checkpoints"
        hit = sorted(x.name for x in ck.glob("*.part")) if ck.is_dir() else []
        return f"{hit[0]} exists on the stand-in Drive (a copy in progress)" if hit else None

    def cwd_of(i: int) -> Path:
        """Session 0 runs in the first clone; every later attempt in the second, which is made when it is first
        asked for (the first is deleted by then)."""
        if i == 0:
            return clones[0]
        if 1 not in clones:
            clones[1] = clone(2)
        return clones[1]

    t0 = time.time()
    # (STUDY_DIR stays unset: the cells then take their working directory, which is each attempt's clone)
    drill = _drill(notebook, dict(p, STUDY_DIR=None), scratch / "logs", cwd_of, part_exists, wipe=dead_vm)
    checks = _drill_checks(drill, p)
    at = drill["attempts"]
    jobs = parse_queue(p["QUEUE"])
    r5 = d / "jobs" / jobs[0]["job"]
    mons = sorted(r5.glob("monitor/monitor_*.json"))
    mon = _read_json(mons[-1]) if mons else {}
    run5, run6 = _read_json(r5 / "run.json") or {}, _read_json(d / "jobs" / jobs[1]["job"] / "run.json") or {}
    clips = sorted(r5.glob("monitor/clip_*.mp4"))
    probe = None
    if clips and shutil.which("ffprobe"):
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(clips[-1])], capture_output=True, text=True).returncode
    line2 = (at["session1"]["lines"][2] or [""])[0]
    m2 = LINE_RE[2].match(line2)
    checks.update(
        head_is_ref=len(heads) == 2 and all(h == ref for h in heads.values()),
        fresh_clones_hold_nothing_ignored=len(fresh) == 2 and all(fresh.values()),
        killed_during_a_copy=bool(drill["kill"]["when"]) and "copy in progress" in str(drill["kill"]["when"]),
        part_swept=bool((_read_json(d / "SESSION_1.json") or {}).get("swept_parts")),
        content_hash=bool(m2) and m2.group("manifest") == str(manifest.get("content_sha256"))[:12]
        and m2.group("bundle") == bundle_id and m2.group("ok") == m2.group("n") == str(len(content.get("files") or {})),
        g8a_every_body={LINE_RE[3].match(x).group("body") for x in at["session1"]["lines"][3]} >= set(
            p["PREFLIGHT_BODIES"]),
        monitor_six_judged=len((mon or {}).get("episodes") or []) == 6,
        gain_recorded=any(isinstance(e.get("gain"), dict) for e in run5.get("monitor") or []),
        g8b_read=(run5.get("gates") or {}).get("g8b") is not None,
        clip_made=bool(clips) and clips[-1].stat().st_size > 0 and probe == 0,
        r6_done=run6.get("status") == "done" and run6.get("mode") == "R6",
        main_checkout_unchanged=before is not None and before == _git("status", "--porcelain"))
    rec = _stamped(dict(
        schema=ACCEPTANCE_SCHEMA, ok=all(checks.values()), checks=checks, remote=url, ls_remote=on_remote,
        ls_remote_sha=ref if on_remote else None, clone_source=source, ref=ref, clone_head=heads.get(2),
        clone_heads={str(i): h for i, h in sorted(heads.items())}, bundle_id=bundle_id,
        content_sha256=manifest.get("content_sha256"), files=len(content.get("files") or {}),
        bodies=p["PREFLIGHT_BODIES"], g8a=g8a_counts(at["session1"]["lines"][3]), params=p, kill=drill["kill"],
        seconds=round(time.time() - t0, 1),
        attempts={k: dict(exit_code=v["exit_code"], killed=v["killed"], seconds=v["seconds"],
                          lines={str(n): rows for n, rows in v["lines"].items() if rows})
                  for k, v in at.items()},
        sessions=[_read_json(f) for f in sorted(d.glob("SESSION_*.json"))], versions=versions(pins()),
        takeover_note="the resume ran with TAKEOVER: LIVE.json is fresh right after the kill, and the session "
                      "before it, without TAKEOVER, was refused as rule K7 asks",
        not_done=["ALLOW_VERSION_CHANGE and ALLOW_SOURCE_CHANGE with a faked session record (design 3.10 step 6) "
                  "are not exercised by this function"],
        at=_now(), rule="rule N5; design 3.10"))
    _write_json(_anchored(record), rec)
    _say(f"S2 ACCEPTANCE {ref[:12]} bundle {bundle_id}: {'OK' if rec['ok'] else 'FAILED'} in {rec['seconds']:.0f} s"
         + ("" if rec["ok"] else ": " + ", ".join(k for k, v in checks.items() if not v)))
    return rec


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _bodies(p: dict) -> list:
    jobs = parse_queue(p["QUEUE"])
    return list(dict.fromkeys(r["body"] for r in jobs)) or [SMOKE_BODY]


def _step_setup(p: dict, args) -> int:
    setup(p, record=args.record or p["SETUP_RECORD"])
    return EXIT_OK


def _step_session(p: dict, args) -> int:
    session(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"], p["QUEUE"], p["REPO_REF"], repo_state()["commit"],
            p["RESUME"], p)
    return EXIT_OK


def _step_inputs(p: dict, args) -> int:
    st = _open_state(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"])
    d = Path(st["drive_run"])
    if _owner(d, st["session"]["session_uuid"]) is not None:
        raise Refused(f"{d}: this session no longer owns the run (LIVE.json)")
    q = _read_json(d / "QUEUE.json") or {}
    was = (q.get("bundle") or {}).get("content_sha256")
    res = inputs(p["BUNDLE_ID"], p["DRIVE_DIR"], p["BUNDLE_SOURCE"], set_name=p["SET"], bodies=_bodies(p),
                 local=p["LOCAL_DIR"], expect=was)
    if res["ok"] and not was and isinstance(q.get("bundle"), dict):
        q["bundle"]["content_sha256"] = res["content_sha256"]
        _write_json(d / "QUEUE.json", q)
    _note(st, res["line"], bundle=dict(id=res["bundle_id"], content_sha256=res["content_sha256"],
                                       files_ok=res["files_ok"], files=res["files"], source=res["source"]))
    if not res["ok"]:
        for extra in res["problems"][1:10]:
            _say(f"  also: {extra}")
        _close(st, "line 2 failed: nothing was trained")
        return EXIT_FAILED
    return EXIT_OK


def _step_preflight(p: dict, args) -> int:
    st = _open_state(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"])
    queue = p["QUEUE"] or [f"R5:{SMOKE_BODY}:42"]
    res = preflight(queue, p["SET"], bodies=p["PREFLIGHT_BODIES"])
    st["lines"].extend(res["lines"])
    _note(st, preflight=dict(ok=res["ok"], archive=res["archive"], policy=res["policy"],
                             envs=(res["envs"] or {}).get("built"), problems=res["problems"],
                             g8a={b: {k: v for k, v in g.items() if k != "episodes"} for b, g in res["g8a"].items()}))
    if not res["ok"]:
        for text in res["problems"]:
            _say(f"S2 PREFLIGHT FAILED: {text}")
        _close(st, "the preflight failed: nothing was trained")
        return EXIT_FAILED
    return EXIT_OK


def _step_smoke(p: dict, args) -> int:
    smoke(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"])
    return EXIT_OK


def _step_launch(p: dict, args) -> int:
    res = launch(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"], p["QUEUE"], p["SET"], p["BUDGET_STEPS"], p["WORKERS"],
                 tiny=p["TINY"], clips=p["CLIPS"], monitor_every=p["MONITOR_EVERY"],
                 monitor_also_k=p["MONITOR_ALSO_K"] or ())
    return {"done": EXIT_OK, "halted": EXIT_HALT, "refused": EXIT_REFUSED, "unfinished": EXIT_UNFINISHED,
            "interrupted": EXIT_INTERRUPTED}.get(res["state"], EXIT_FAILED)


def _step_finish(p: dict, args) -> int:
    res = finish(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"])
    if res["state"] == "halted":
        return EXIT_HALT
    return EXIT_UNFINISHED if res["on_drive"] is False else EXIT_OK


STEPS = dict(setup=_step_setup, session=_step_session, inputs=_step_inputs, preflight=_step_preflight,
             smoke=_step_smoke, launch=_step_launch, finish=_step_finish)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["child"]:
        return _child_main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.nb", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in STEPS:
        a = sub.add_parser(name, help=f"the notebook's {name} step")
        a.add_argument("--params", required=True, help="the JSON file of the notebook's parameters")
        a.add_argument("--set-param", action="append", default=[], metavar="KEY=JSON",
                       help="replace one parameter (e.g. RESUME=true)")
        if name == "setup":
            a.add_argument("--record", default=None, help="where the setup's record is written")
    a = sub.add_parser("collect", help="copy a finished job's text records to their committed place")
    a.add_argument("--src", required=True)
    a.add_argument("--dst", required=True)
    a.add_argument("--kind", required=True, choices=KINDS)
    a = sub.add_parser("hours", help="runtime-hours per run and session against the cap")
    a.add_argument("--runs-root", required=True)
    a.add_argument("--out", default="stage2/tb/hours.json")
    a = sub.add_parser("local-run", help="development: the notebook's cells here, on the Stage 1 view, with a kill")
    a.add_argument("--notebook", default=NOTEBOOK)
    a.add_argument("--drive", required=True, help="a directory standing in for Drive (scratch, or stage2/tb/work/...)")
    a.add_argument("--params", default=None, help="a JSON file of parameters over the defaults")
    a.add_argument("--record", default=None)
    a = sub.add_parser("acceptance", help="design 3.10: a fresh clone and the bundle; kill; resume (the lead's step)")
    a.add_argument("--notebook", default=NOTEBOOK)
    a.add_argument("--ref", required=True)
    a.add_argument("--parts", required=True)
    a.add_argument("--scratch", required=True)
    a.add_argument("--record", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd in STEPS:
            raw = _read_json(args.params)
            if not isinstance(raw, dict):
                raise Refused(f"{args.params}: the parameters cannot be read")
            for item in args.set_param:
                key, _, value = item.partition("=")
                raw[key] = json.loads(value)
            return STEPS[args.cmd](load_params(raw), args)
        if args.cmd == "collect":
            collect(args.src, args.dst, args.kind)
            return EXIT_OK
        if args.cmd == "hours":
            hours(args.runs_root, args.out)
            return EXIT_OK
        if args.cmd == "local-run":
            extra = _read_json(args.params) if args.params else {}
            return EXIT_OK if local_run(args.notebook, args.drive, extra, args.record)["ok"] else EXIT_FAILED
        return EXIT_OK if acceptance(args.notebook, args.ref, args.parts, args.scratch, args.record)["ok"] \
            else EXIT_FAILED
    except Refused as exc:
        _say(str(exc) if exc.line else f"S2 REFUSED {args.cmd}: {exc}")
        return exc.code
    except NbError as exc:
        _say(f"S2 FAILED {args.cmd}: {exc}")
        return exc.code


__all__ = ["BACK_RULE", "BUDGET_RULE", "COPY_RULE", "FENCE_RULE", "HALT_RULE", "JOB_RULE", "K6_RULE", "K7_RULE",
           "LINE_RE", "NbError", "PARAMS", "Refused", "UNFINISHED_RULE", "acceptance", "at_or_past", "budget_for",
           "collect", "finish", "floor_rate", "g8a_counts", "hours", "inputs", "job_name", "keep_on_drive", "launch",
           "line1", "line2", "line3", "line4", "lines_of",
           "load_params", "local_run", "main", "parse_queue", "pin_differences", "pins", "pip_commands", "preflight",
           "release_runtime", "run_cells", "session", "session_line", "setup", "sha256_json", "smoke", "smoke_line"]

if __name__ == "__main__":
    from designlab.stage2 import nb as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
