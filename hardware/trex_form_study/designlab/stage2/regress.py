"""Closed-loop bit-identity regression: committed Stage 1 and Stage 1b episodes run again through the stock ``eval``
command and compared with their committed records (round A design, D18 and section 4.2).

Stage 1's records are cited, never rerun.  Every committed episode record holds the sha256 of its 50 Hz trace
(``panels.TRACE_RULE``), and unchanged code regenerates those traces byte for byte.  That is the only closed-loop
proof that an edit to a Stage 1 module changes nothing by default, so this module makes it a command, a committed
record (``stage2/regress/regress.json``) and a test (``tests/test_stage2_regress.py``).

What one run does (``RULE``):

* each case of ``CASES`` is one subprocess, the stock command of ``tools/README.md`` ("Eval traces"), writing into a
  scratch directory, never into a committed run directory (``eval`` overwrites a record whose key differs);
* per episode, ``trace_equal``: the sha256 of the regenerated trace file equals the one in the committed episode
  record; ``record_equal``: the regenerated episode record equals the committed one on every key but ``wall_s``
  (``EXCLUDED_KEYS``), value for value, floats exactly (``s2io.canonical``: a NaN equals a NaN, 1 differs from 1.0).
  The second reading matters: an episode's ``strides`` is in no trace;
* ``run`` refuses to start without the arrays and the round-two model files its cases read, without the committed
  records, or in a checkout with uncommitted changes, and checks the checkout again at the end (its own record aside);
* the record holds every episode with both flags, ``matched`` and ``of``, the four source hashes read at run time
  (the panels', the Stage 1 package's, the tuner's and the speed fit's), the library versions, the commit and a time.

``check`` compares a record with today's code: tier ``full``, every episode matched, and the four hashes equal to
today's.  No hash is written anywhere as a constant: after any edit to a Stage 1 module the committed record is stale
until ``run --tier full --record stage2/regress/regress.json`` has been run on the new code and its record committed.

A run is resumable: each finished case is kept in ``<scratch>/cases/<case id>.json`` and used again only by a run of
the same case on the same four source hashes, library versions and ``regress.py`` (``RESUME_RULE``); the record is
written only when every case is there.

Run from hardware/trex_form_study:
    python -m designlab.stage2.regress run --tier fast
    bash tools/lowprio.sh python -m designlab.stage2.regress run --tier full --record stage2/regress/regress.json
    python -m designlab.stage2.regress check

Exit codes: 0 everything matches; 1 a difference; 2 the run was refused (inputs, checkout, threads).

This module is in no record key.  It imports Stage 1 only inside the functions that need it.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # before any numpy import, as look.py does
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from . import s2io  # noqa: E402
from .s2io import STUDY  # noqa: E402

SCHEMA = 1
TIERS = ("fast", "full")
#: The committed record of the full tier.
RECORD = STUDY / "stage2" / "regress" / "regress.json"
#: Where a run keeps its cases, the evals' output and their logs unless ``--scratch`` says otherwise (git-ignored).
SCRATCH = STUDY / "logs" / "regress"
#: The committed run directories the cases read their records from: Stage 1's panels and Stage 1b's speed round.
STAGE1_RUN = "stage1/runs/2026-11-08"
STAGE1B_RUN = "stage1/runs/2026-11-10-speed"
#: The module that registers Stage 1b's rows (``eval --row-module``).
SPEED_MODULE = "designlab.stage1.speedfit"
#: ``panels.SEEDS``' first seed and count (checked against the panels when a run starts).
SEED0, SEEDS_PER_CONDITION = 3042, 20
#: The episode keys left out of ``record_equal``.
EXCLUDED_KEYS = ("wall_s",)
#: The environment variable that would point Stage 1b's rows at another speed run directory; removed for every eval.
SPEED_RUN_ENV = "STAGE1B_SPEED_RUN"
#: The keys of a record's ``sources``.
SOURCE_KEYS = ("panel_sources_sha256", "stage1_sources_sha256", "tuner_sources_sha256", "speed_sources_sha256")

RULE = (
    "each case is one subprocess `python -m designlab.stage1 eval --body B [--row-module M] --rows ROWS --conditions "
    "CONDITIONS --episodes N --workers 1 --inputs I --out <scratch>` (tools/README.md, 'Eval traces') with "
    "OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS at 1, MUJOCO_GL=disable and STAGE1B_SPEED_RUN removed; "
    "per episode, trace_equal: the sha256 of the regenerated <out>/<body>/eval/traces/<row>/<behavior>/<condition>/"
    "<seed>.npz equals episodes[i].trace.sha256 of the committed record <run>/<body>/eval/<row>/<behavior>/"
    "<condition>.json; record_equal: the regenerated episode record equals the committed one on every key but "
    + ", ".join(EXCLUDED_KEYS) + ", compared as canonical JSON after a round trip through a file (floats exactly; a "
    "NaN equals a NaN; 1 differs from 1.0); an episode matches when both hold; an eval that fails, or an episode it "
    "does not write, matches nothing"
)
CLEAN_RULE = (
    "a run starts only in a checkout whose `git status --porcelain` is empty (the record it is writing aside) and "
    "with the arrays, the round-two model files and the committed records its cases read; the status is read again "
    "after the last case and the run is refused if anything changed; the four source hashes are read before the first "
    "case and after the last and must be equal"
)
RESUME_RULE = (
    "a finished case is kept in <scratch>/cases/<case id>.json and used again only if its case definition, its four "
    "source hashes, its library versions and the sha256 of regress.py are the run's; otherwise the case runs again "
    "from an empty directory; the record is written only when every case of the tier is there"
)
CHECK_RULE = (
    "a record is current when its tier is full, every episode of today's full tier is in it with both flags true "
    "(matched == of), and its four source hashes equal the ones read from today's code"
)

#: Every hard-floor condition of Stage 1's panels (``panels.conditions`` less the pile floors): the stand, the six
#: walks, the six turns, the start and stop, the eight pushes.  R0 runs hard floor only, so both rows run all 22.
HARD_FLOOR = (
    ("stand/hard",)
    + tuple(f"walk/{g}{mu}" for mu in ("", "_mu0.54") for g in ("nominal", "brisk", "fr05"))
    + tuple(f"turn/{c}_{d}" for c in ("walking_0.35", "in_place_0.50", "walking_0.50") for d in ("left", "right"))
    + ("start_stop/twice",)
    + tuple(f"push/{m}_{d}" for m in ("standing", "walking") for d in ("forward", "backward", "left", "right"))
)
#: Stage 1b's turn conditions (``speedfit.ROW_CONDITIONS``' turns: what ``--behaviors turn`` runs for its rows).
STAGE1B_TURNS = tuple(f"turn/{c}_{d}" for c in ("walking_0.35", "in_place_0.50") for d in ("left", "right"))
STAGE1B_ROWS = ("R2s/42@10ms", "R2@10ms")


def _case(id: str, tier: str, run: str, body: str, rows: tuple, conditions: tuple, episodes: int = 1,
          row_module: str | None = None) -> dict:
    return dict(id=id, tier=tier, run=run, inputs=STAGE1_RUN, body=body, row_module=row_module, rows=tuple(rows),
                conditions=tuple(conditions), seeds=tuple(range(SEED0, SEED0 + int(episodes))))


#: The cases (design section 4.2).  ``run``: the committed run directory the records are in; ``inputs``: the run
#: directory holding the body's references and transitions; ``seeds``: the first N of ``panels.SEEDS`` (``eval
#: --episodes N``).  The fast tier is 8 episodes: a hold, a stop event, a speed-change plan and mode R1 without terms
#: at 0 ms on 75-M-H-10, and the 10 ms delay queue on two turns of 60-N-T-10.  The full tier adds seed 3042 of every
#: hard-floor condition of both bodies under R0 and R2/42 (88 episodes) and all 20 seeds of Stage 1b's four turns
#: under its two 10 ms rows on 60-N-T-10 (160 episodes, one case per turn so that a killed run loses one turn).
CASES = (
    _case("fast-75-M-H-10", "fast", STAGE1_RUN, "75-M-H-10", ("R0", "R2/42"), ("start_stop/twice", "walk/brisk")),
    _case("fast-60-N-T-10-10ms", "fast", STAGE1B_RUN, "60-N-T-10", STAGE1B_ROWS,
          ("turn/walking_0.35_left", "turn/in_place_0.50_left"), row_module=SPEED_MODULE),
    _case("full-75-M-H-10-hard-floor", "full", STAGE1_RUN, "75-M-H-10", ("R0", "R2/42"), HARD_FLOOR),
    _case("full-60-N-T-10-hard-floor", "full", STAGE1_RUN, "60-N-T-10", ("R0", "R2/42"), HARD_FLOOR),
) + tuple(
    _case(f"full-60-N-T-10-10ms-{c.split('/', 1)[1]}", "full", STAGE1B_RUN, "60-N-T-10", STAGE1B_ROWS, (c,),
          episodes=SEEDS_PER_CONDITION, row_module=SPEED_MODULE)
    for c in STAGE1B_TURNS
)


class RegressError(s2io.S2Error):
    """A run was refused: inputs missing, a checkout with uncommitted changes, sources that changed under it."""


# ------------------------------------------------------------------------------------------------
# Cases, their committed records and their commands
# ------------------------------------------------------------------------------------------------
def cases_of(tier: str) -> list[dict]:
    """The cases a tier runs: the fast tier's own; the full tier is the fast tier plus its own."""
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r}: choose from {', '.join(TIERS)}")
    return [c for c in CASES if c["tier"] == "fast" or tier == "full"]


def episodes_of(case: dict) -> list[tuple[str, str, int]]:
    """(row, condition, seed) of every episode of a case, in the order the record lists them."""
    return [(row, cond, seed) for row in case["rows"] for cond in case["conditions"] for seed in case["seeds"]]


def committed_record(case: dict, row: str, condition: str) -> Path:
    """``<run>/<body>/eval/<row>/<behavior>/<condition>.json`` of a case's row and condition ("turn/x"), committed."""
    behavior, name = condition.split("/", 1)
    return STUDY / case["run"] / case["body"] / "eval" / row / behavior / f"{name}.json"


def command(case: dict, out) -> list[str]:
    """The stock ``eval`` command of a case writing into ``out`` (after ``sys.executable``)."""
    cmd = ["-m", "designlab.stage1", "eval", "--body", case["body"]]
    if case["row_module"]:
        cmd += ["--row-module", case["row_module"]]
    return cmd + ["--rows", ",".join(case["rows"]), "--conditions", ",".join(case["conditions"]), "--episodes",
                  str(len(case["seeds"])), "--workers", "1", "--inputs", case["inputs"], "--out", str(out)]


def eval_env() -> dict:
    """The environment of every eval: the caller's with one BLAS thread, no GL backend and no ``STAGE1B_SPEED_RUN``
    (Stage 1b's rows then read their gains from the committed speed round)."""
    env = dict(os.environ)
    for v in s2io.THREAD_VARS:
        env[v] = "1"
    env["MUJOCO_GL"] = "disable"
    env.pop(SPEED_RUN_ENV, None)
    return env


def sources() -> dict:
    """The four source hashes of today's code (``SOURCE_KEYS``), each read through its own module."""
    from designlab.stage1 import panels, plant, speedfit, tune

    return dict(panel_sources_sha256=panels.panel_sources()["sha256"], stage1_sources_sha256=plant.sources_sha256(),
                tuner_sources_sha256=tune.tuner_sources()["sha256"],
                speed_sources_sha256=speedfit.speed_sources()["sha256"])


def missing_inputs(tier: str = "full") -> list[str]:
    """What a run of ``tier`` needs and this checkout does not hold: the round-two model files of each body (without
    them ``plant.model_files`` would rebuild the models into ``stage1/models/<body>/``), the reference and transition
    arrays each condition loads, and the committed records.  [] is good."""
    from designlab.stage1 import panels
    from designlab.stage1.__main__ import round_two_models

    conds = {c.key: c for c in panels.conditions()}
    out = []
    for body in dict.fromkeys(c["body"] for c in cases_of(tier)):
        if not round_two_models(body):
            out.append(f"{body}: no round-two model files (python tools/restore_inputs.py restore --group picks-models)")
    arrays, records = {}, {}
    for case in cases_of(tier):
        for key in case["conditions"]:
            if key not in conds:
                out.append(f"case {case['id']}: {key} is not a panel condition")
                continue
            for f in panels.files_of(conds[key]):
                arrays[STUDY / case["inputs"] / case["body"] / f] = None
            for row in case["rows"]:
                records[committed_record(case, row, key)] = None
    gone = [p for p in arrays if not p.is_file()]
    if gone:
        out.append(f"{len(gone)} of {len(arrays)} arrays are missing, the first {s2io.portable(str(gone[0]))} "
                   "(python tools/restore_inputs.py restore --group picks-arrays)")
    out += [f"{s2io.portable(str(p))}: no committed record" for p in records if not p.is_file()]
    return out


# ------------------------------------------------------------------------------------------------
# The checkout
# ------------------------------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(STUDY), *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegressError(f"git {' '.join(args)} failed: {getattr(exc, 'stderr', '') or exc}") from None


def git_status(own=()) -> list[str]:
    """The lines of ``git status --porcelain`` of the whole repository, less the ones that name only a path of
    ``own`` (the record a run is writing)."""
    top = Path(_git("rev-parse", "--show-toplevel").strip())
    own = {Path(os.path.realpath(p)) for p in own if p is not None}
    out = []
    for line in _git("status", "--porcelain", "--untracked-files=all").splitlines():
        if not line.strip():
            continue
        paths = [x.strip().strip('"') for x in line[3:].split(" -> ")]
        if not all(Path(os.path.realpath(top / x)) in own for x in paths):
            out.append(line)
    return out


def _tracked_below(path: Path) -> bool:
    """Whether git tracks a file at or below ``path`` (a run's scratch root must hold none)."""
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(_git("rev-parse", "--show-toplevel").strip()))
    except ValueError:
        return False  # outside the repository
    return bool(_git("ls-files", "--", str(Path(os.path.realpath(path)))).strip())


# ------------------------------------------------------------------------------------------------
# Comparing one case
# ------------------------------------------------------------------------------------------------
def differing(a, b, path: str = "", limit: int = 12) -> list[str]:
    """The paths at which two JSON values differ (canonical JSON, ``RULE``), at most ``limit`` of them."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in list(a) + [k for k in b if k not in a]:
            here = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                out.append(f"{here} ({'missing' if k not in a else 'added'})")
            else:
                out += differing(a[k], b[k], here, limit)
            if len(out) >= limit:
                break
        return out[:limit]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differing(x, y, f"{path}[{i}]", limit)
            if len(out) >= limit:
                break
        return out[:limit]
    return [] if s2io.canonical(a) == s2io.canonical(b) else [path or "(the whole value)"]


def _less_excluded(episode: dict) -> dict:
    return {k: v for k, v in episode.items() if k not in EXCLUDED_KEYS}


def _episodes_by_seed(path: Path) -> tuple[dict, str | None]:
    """A condition record's episodes by seed, and why there are none (None if the record could be read)."""
    try:
        doc = s2io.read_json(path)
    except OSError:
        return {}, "no record"
    except ValueError as exc:
        return {}, f"unreadable record ({exc})"
    eps = doc.get("episodes") if isinstance(doc, dict) else None
    if not isinstance(eps, list):
        return {}, f"a record without episodes (status {doc.get('status') if isinstance(doc, dict) else None!r})"
    return {e.get("seed"): e for e in eps if isinstance(e, dict)}, None


def compare_case(case: dict, out) -> list[dict]:
    """One entry per episode of ``case`` (``episodes_of``): the regenerated files under ``out`` against the committed
    record (``RULE``).  ``note`` says why an episode could not be compared; ``differs`` lists where the two episode
    records differ."""
    out = Path(out)
    entries, cache = [], {}
    for row, condition, seed in episodes_of(case):
        behavior, name = condition.split("/", 1)
        old_path = committed_record(case, row, condition)
        new_path = out / case["body"] / "eval" / row / behavior / f"{name}.json"
        if (row, condition) not in cache:
            cache[row, condition] = (_episodes_by_seed(old_path), _episodes_by_seed(new_path))
        (old, old_why), (new, new_why) = cache[row, condition]
        e = dict(case=case["id"], run=case["run"], body=case["body"], row=row, condition=condition, seed=int(seed),
                 record=s2io.portable(str(old_path)), trace_sha256=None, recorded=None, trace_equal=False,
                 record_equal=False)
        notes = []
        eo, en = old.get(seed), new.get(seed)
        if eo is None:
            notes.append(f"committed: {old_why or 'no episode of this seed'}")
        else:
            e["recorded"] = (eo.get("trace") or {}).get("sha256")
            if e["recorded"] is None:
                notes.append("committed: the episode record holds no trace sha256")
        trace = out / case["body"] / "eval" / "traces" / row / behavior / name / f"{int(seed)}.npz"
        if trace.is_file():
            e["trace_sha256"] = s2io.sha256_file(trace)
        else:
            notes.append("regenerated: no trace file")
        e["trace_equal"] = e["trace_sha256"] is not None and e["trace_sha256"] == e["recorded"]
        if en is None:
            notes.append(f"regenerated: {new_why or 'no episode of this seed'}")
        elif eo is not None:
            diff = differing(_less_excluded(eo), _less_excluded(en))
            e["record_equal"] = not diff
            if diff:
                e["differs"] = diff
            if en.get("error"):
                notes.append("regenerated: the episode failed to run: " + str(en["error"]).strip().splitlines()[-1])
        if notes:
            e["note"] = "; ".join(notes)
        entries.append(e)
    return entries


def matched(entry: dict) -> bool:
    return bool(entry.get("trace_equal")) and bool(entry.get("record_equal"))


# ------------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------------
def _own_sha256() -> str:
    return s2io.sha256_file(Path(__file__).resolve())


def _case_key(case: dict, src: dict, versions: dict, own: str) -> dict:
    """What a kept case result must share with the run that uses it again (``RESUME_RULE``)."""
    return s2io.plain(dict(case=case, sources=src, versions=versions, regress_sha256=own))


def run_case(case: dict, scratch, key: dict, log=print) -> dict:
    """Run one case into ``<scratch>/work/<case id>`` (emptied first), compare it and keep the result in
    ``<scratch>/cases/<case id>.json``.  The eval's output goes to ``<scratch>/logs/<case id>.log``.  The work
    directory is removed when every episode matched and kept for inspection otherwise."""
    scratch = Path(scratch)
    work, logf = scratch / "work" / case["id"], scratch / "logs" / f"{case['id']}.log"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    logf.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(case, work)
    started, t0 = s2io.now_utc(), time.perf_counter()
    with open(logf, "w") as fh:
        fh.write("$ python " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run([sys.executable, *cmd], cwd=STUDY, env=eval_env(), stdout=fh, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    entries = compare_case(case, work)
    n = sum(matched(e) for e in entries)
    doc = dict(key=key, command=["python", *s2io.portable(cmd[:-1]), "<scratch>"], returncode=proc.returncode,
               started_at=started, ended_at=s2io.now_utc(), wall_s=wall, episodes=entries, matched=n, of=len(entries),
               commit=_git("rev-parse", "HEAD").strip(), log=str(logf))
    s2io.write_json(scratch / "cases" / f"{case['id']}.json", doc)
    if n == len(entries) and proc.returncode == 0:
        shutil.rmtree(work)
    log(f"{case['id']}: {n} of {len(entries)} episodes match ({wall:.0f} s"
        + ("" if proc.returncode == 0 else f"; eval exit code {proc.returncode}, see {logf}") + ")")
    for e in entries:
        if not matched(e):
            log(f"  DIFFERENT {e['body']} {e['row']} {e['condition']} seed {e['seed']}: trace "
                f"{'equal' if e['trace_equal'] else 'differs'}, record {'equal' if e['record_equal'] else 'differs'}"
                + (f" at {', '.join(e['differs'])}" if e.get("differs") else "")
                + (f" [{e['note']}]" if e.get("note") else ""))
    return doc


def _kept(case: dict, scratch: Path, key: dict) -> dict | None:
    """A case's kept result if it may be used again (``RESUME_RULE``), else None."""
    try:
        doc = s2io.read_json(Path(scratch) / "cases" / f"{case['id']}.json")
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("key") != key:
        return None
    if [(e.get("row"), e.get("condition"), e.get("seed")) for e in doc.get("episodes") or []] != [
            (r, c, int(s)) for r, c, s in episodes_of(case)]:
        return None
    return doc


def run(tier: str = "fast", record=None, procs: int = 1, log=print, scratch=None) -> dict:
    """Run a tier (module docstring) and return its record; with ``record`` (a path) also write it there through
    ``s2io.write_json``.  ``procs`` cases run at once (1: one eval process at a time).  ``scratch``: where the cases,
    the evals' output and their logs go (default ``SCRATCH``).  Raises ``RegressError`` when the run is refused
    (``CLEAN_RULE``); a difference is not an error here: read ``matched`` and ``of``."""
    from designlab.stage1 import panels

    cases = cases_of(tier)
    if panels.SEEDS != tuple(range(SEED0, SEED0 + SEEDS_PER_CONDITION)):
        raise RegressError(f"panels.SEEDS is no longer {SEED0}..{SEED0 + SEEDS_PER_CONDITION - 1}: `eval --episodes N` "
                           "would run other seeds than the cases name")
    record = None if record is None else (Path(record) if Path(record).is_absolute() else Path.cwd() / record)
    scratch = SCRATCH if scratch is None else (Path(scratch) if Path(scratch).is_absolute() else Path.cwd() / scratch)
    missing = missing_inputs(tier)
    if missing:
        raise RegressError("this checkout cannot run the regression: " + "; ".join(missing))
    dirty = git_status(own=(record,))
    if dirty:
        raise RegressError("the checkout has uncommitted changes (commit or remove them first; the record names a "
                           "commit): " + "; ".join(dirty[:8]) + (f"; and {len(dirty) - 8} more" if len(dirty) > 8 else ""))
    if _tracked_below(scratch):
        raise RegressError(f"{scratch}: git tracks files below the scratch root; an eval would overwrite them")
    commit = _git("rev-parse", "HEAD").strip()
    src, versions, own = sources(), panels.library_versions(), _own_sha256()
    t0 = time.perf_counter()
    scratch.mkdir(parents=True, exist_ok=True)
    with s2io.Lock(scratch / ".lock", log=log):
        docs, todo = {}, []
        for case in cases:
            key = _case_key(case, src, versions, own)
            kept = _kept(case, scratch, key)
            if kept is None:
                todo.append((case, key))
            else:
                docs[case["id"]] = kept
                log(f"{case['id']}: kept from {kept.get('ended_at')} ({kept['matched']} of {kept['of']} episodes match)")
        if todo:
            with ThreadPoolExecutor(max_workers=max(1, int(procs))) as pool:
                for (case, _), doc in zip(todo, pool.map(lambda x: run_case(x[0], scratch, x[1], log), todo)):
                    docs[case["id"]] = doc
    after = sources()
    if after != src or _own_sha256() != own:
        raise RegressError("the sources changed during the run: "
                           + (", ".join(k for k in SOURCE_KEYS if after[k] != src[k]) or "regress.py"))
    dirty = git_status(own=(record,))
    if dirty or _git("rev-parse", "HEAD").strip() != commit:
        raise RegressError("the checkout changed during the run (an eval must write nothing outside its scratch "
                           "directory): " + ("; ".join(dirty[:8]) or "another commit is checked out"))
    entries = [e for case in cases for e in docs[case["id"]]["episodes"]]
    doc = dict(
        schema=SCHEMA, tier=tier, rules=dict(regress=RULE, clean=CLEAN_RULE, resume=RESUME_RULE, check=CHECK_RULE),
        cases=entries, matched=sum(matched(e) for e in entries), of=len(entries),
        commands=[dict(case, command=docs[case["id"]]["command"], returncode=docs[case["id"]]["returncode"],
                       matched=docs[case["id"]]["matched"], of=docs[case["id"]]["of"],
                       wall_s=docs[case["id"]]["wall_s"], started_at=docs[case["id"]]["started_at"],
                       ended_at=docs[case["id"]]["ended_at"], commit=docs[case["id"]]["commit"]) for case in cases],
        sources=src, regress_sha256=own, versions=versions, blas_threads=int(eval_env()[s2io.THREAD_VARS[0]]),
        environment={k: eval_env().get(k) for k in (*s2io.THREAD_VARS, "MUJOCO_GL", SPEED_RUN_ENV)},
        commit=commit, dirty=bool(dirty), eval_wall_s=sum(docs[case["id"]]["wall_s"] for case in cases),
        session_wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    if record is not None:
        s2io.write_json(record, doc)
        log(f"record: {record}")
    log(f"{tier} tier: {doc['matched']} of {doc['of']} episodes match their committed trace and episode record "
        f"({doc['eval_wall_s']:.0f} s of evals)")
    return doc


# ------------------------------------------------------------------------------------------------
# Checking a record against today's code
# ------------------------------------------------------------------------------------------------
def check(record=RECORD) -> list[str]:
    """The differences between a record and today's code (``CHECK_RULE``); [] is good."""
    try:
        doc = s2io.read_json(record)
    except (OSError, ValueError) as exc:
        return [f"{record}: no readable record ({exc})"]
    if not isinstance(doc, dict):
        return [f"{record}: not a record"]
    out = []
    if doc.get("tier") != "full":
        out.append(f"tier is {doc.get('tier')!r}, not 'full'")
    entries = doc.get("cases") if isinstance(doc.get("cases"), list) else []
    n = sum(matched(e) for e in entries)
    if not entries or doc.get("of") != len(entries) or doc.get("matched") != n or n != len(entries):
        out.append(f"matched {doc.get('matched')} of {doc.get('of')} ({n} of {len(entries)} listed episodes hold both "
                   "flags)")
    want = [(c["id"], r, k, int(s)) for c in cases_of("full") for r, k, s in episodes_of(c)]
    got = [(e.get("case"), e.get("row"), e.get("condition"), e.get("seed")) for e in entries]
    if got != want:
        out.append(f"the record lists {len(got)} episodes, today's full tier names {len(want)}"
                   + ("" if len(got) != len(want) else " others"))
    today, held = sources(), doc.get("sources") or {}
    for k in SOURCE_KEYS:
        if held.get(k) != today[k]:
            out.append(f"{k}: the record holds {str(held.get(k))[:12]}..., today's code is {today[k][:12]}...")
    if out:
        out.append("run `python -m designlab.stage2.regress run --tier full --record stage2/regress/regress.json` on "
                   "this code and commit its record")
    return out


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.regress", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="run a tier's episodes again and compare them with the committed records")
    rp.add_argument("--tier", choices=TIERS, default="fast")
    rp.add_argument("--record", default=None, help="write the record here (the committed one: "
                                                   "stage2/regress/regress.json)")
    rp.add_argument("--procs", type=int, default=1, help="cases run at once (default 1)")
    rp.add_argument("--scratch", default=None, help="where the cases, the evals' output and their logs go (default "
                                                    "logs/regress; a killed run continues from it)")
    cp = sub.add_parser("check", help="compare a record with today's code")
    cp.add_argument("--record", default=str(RECORD))
    a = ap.parse_args(argv)
    try:
        s2io.require_one_thread()
        if a.cmd == "check":
            diff = check(a.record)
            for line in diff:
                print(line, flush=True)
            if not diff:
                print(f"{s2io.portable(str(Path(a.record).resolve()))}: current (full tier, every episode matched, "
                      "the four source hashes are today's)", flush=True)
            return 1 if diff else 0
        doc = run(a.tier, record=a.record, procs=a.procs, scratch=a.scratch,
                  log=lambda line: print(line, flush=True))
    except s2io.S2Error as exc:
        print(f"refused: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0 if doc["matched"] == doc["of"] else 1


if __name__ == "__main__":
    from designlab.stage2 import regress as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
