"""Tests for Stage 2's shared I/O (designlab/stage2/s2io.py): the thread check, the accuracy context, canonical JSON,
atomic writes, locks, portable paths and provenance.

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage2_s2io.py -q
All fast.  Nothing is written into the checkout: files go to pytest's temporary directory, and the git part of
``built_by`` is exercised on a throwaway repository there.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage2 import s2io  # noqa: E402

RUN = ROOT / "stage1" / "runs" / "2026-11-08" / "75-M-H-10"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


# ------------------------------------------------------------------------------------------------
# Threads and accuracy
# ------------------------------------------------------------------------------------------------
def _thread_check(**threads):
    env = {k: v for k, v in os.environ.items() if k not in s2io.THREAD_VARS}
    env.update({k: v for k, v in threads.items() if v is not None}, PYTHONPATH=str(ROOT))
    code = "from designlab.stage2 import s2io; print(s2io.require_one_thread(), s2io.blas_threads())"
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)


def test_require_one_thread_refuses_four_and_unset():
    """A process started with ``OMP_NUM_THREADS=4`` is refused, and so is one with a variable unset; only "1" in all
    three passes."""
    one = dict.fromkeys(s2io.THREAD_VARS, "1")
    ok = _thread_check(**one)
    assert ok.returncode == 0 and ok.stdout.split() == ["1", "1"], ok.stderr
    four = _thread_check(**dict(one, OMP_NUM_THREADS="4"))
    assert four.returncode != 0 and "S2Error" in four.stderr and "OMP_NUM_THREADS='4'" in four.stderr
    unset = _thread_check(**dict(one, MKL_NUM_THREADS=None))
    assert unset.returncode != 0 and "S2Error" in unset.stderr and "MKL_NUM_THREADS=None" in unset.stderr
    none = _thread_check()
    assert none.returncode != 0 and all(v in none.stderr for v in s2io.THREAD_VARS)


def test_blas_threads(monkeypatch):
    for v in s2io.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    assert s2io.require_one_thread() == 1 and s2io.blas_threads() == 1
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    assert s2io.blas_threads() == 4
    with pytest.raises(s2io.S2Error, match="OPENBLAS_NUM_THREADS='4'"):
        s2io.require_one_thread()
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", " 1")  # exactly "1", nothing that merely parses as one
    with pytest.raises(s2io.S2Error):
        s2io.require_one_thread()
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    assert s2io.blas_threads() is None


def test_r2_restores_the_level_and_catches_a_change():
    from designlab import gait

    before = gait.ACCURACY
    try:
        for prev in ("r1", "r2"):
            gait.set_accuracy(prev)
            with s2io.r2():
                assert gait.ACCURACY == "r2" and gait.EXACT_UNWRAP is True
            assert gait.ACCURACY == prev
            # something inside leaves another level behind: refused at the exit, and the level is restored
            with pytest.raises(s2io.S2Error, match="changed inside"):
                with s2io.r2():
                    gait.set_accuracy("r1")
            assert gait.ACCURACY == prev
            # the block's own error is not hidden, and the level is restored
            with pytest.raises(KeyError):
                with s2io.r2():
                    gait.set_accuracy("r1")
                    raise KeyError("the block's own")
            assert gait.ACCURACY == prev
        gait.set_accuracy("r1")
        with s2io.r2():  # nested blocks
            with s2io.r2():
                pass
            assert gait.ACCURACY == "r2"
        assert gait.ACCURACY == "r1"
    finally:
        gait.set_accuracy(before)


# ------------------------------------------------------------------------------------------------
# Canonical JSON
# ------------------------------------------------------------------------------------------------
def test_canonical_is_stable_under_key_order():
    a = {"b": 1, "a": {"y": [1.5, 2, "x"], "x": None}, "c": True}
    b = {"c": True, "a": {"x": None, "y": (1.5, 2, "x")}, "b": 1}
    assert s2io.canonical(a) == s2io.canonical(b) == b'{"a":{"x":null,"y":[1.5,2,"x"]},"b":1,"c":true}'
    assert s2io.sha256_json(a) == hashlib.sha256(s2io.canonical(b)).hexdigest()
    # numpy types are their Python values; an int and a float stay apart, as in a JSON file
    n = {"b": np.int64(1), "a": {"y": np.array([1.5, 2, 3]), "x": None}, "c": np.bool_(True), "d": np.float32(0.5)}
    p = {"b": 1, "a": {"y": [1.5, 2.0, 3.0], "x": None}, "c": True, "d": 0.5}
    assert s2io.canonical(n) == s2io.canonical(p)
    assert s2io.canonical({"a": 1}) != s2io.canonical({"a": 1.0})
    assert s2io.canonical({3: "x"}) == s2io.canonical({"3": "x"}) == b'{"3":"x"}'
    # a value read back from a file hashes as the value that was written
    doc = dict(period_s=0.8606217748321, speed=0.4, n=160, gaits=dict(b=dict(period=math.nan, duty=math.nan)))
    assert s2io.sha256_json(json.loads(json.dumps(s2io.plain(doc)))) == s2io.sha256_json(doc)
    # Stage 1's records hold NaN (a stand's period): hashed as a string, never written as a bare token
    assert s2io.canonical([math.nan, math.inf, -math.inf, np.float64("nan")]) == b'["NaN","Infinity","-Infinity","NaN"]'
    for bad in ({1.5: "x"}, {True: "x"}, {"a": {1, 2}}, {"a": object()}, {1: "x", "1": "y"}):
        with pytest.raises(TypeError):
            s2io.canonical(bad)
    assert s2io.plain(Path("a") / "b") == str(Path("a") / "b")


def test_portable_makes_study_paths_relative():
    from designlab.stage1 import panels

    inside = str(s2io.STUDY / "stage1" / "runs" / "2026-11-08" / "75-M-H-10" / "tune.json")
    outside = "/somewhere/else/tune.json"
    x = dict(file=inside, other=outside, n=3, nested=[inside, (outside, inside)], study=str(s2io.STUDY))
    x[inside] = "as a key"
    want = dict(file="stage1/runs/2026-11-08/75-M-H-10/tune.json", other=outside, n=3,
                nested=["stage1/runs/2026-11-08/75-M-H-10/tune.json", [outside, "stage1/runs/2026-11-08/75-M-H-10/tune.json"]],
                study=str(s2io.STUDY))
    want["stage1/runs/2026-11-08/75-M-H-10/tune.json"] = "as a key"
    assert s2io.portable(x) == want == panels._portable(x)
    assert s2io.portable(Path(inside)) == "stage1/runs/2026-11-08/75-M-H-10/tune.json"
    assert s2io.STUDY == panels.STUDY == ROOT
    # a key built from a portable description does not depend on the working directory or on where the checkout is
    assert s2io.sha256_json(s2io.portable(x)) == s2io.sha256_json(want)


# ------------------------------------------------------------------------------------------------
# Atomic writes
# ------------------------------------------------------------------------------------------------
def _arrays():
    rng = np.random.default_rng(3)
    return dict(t=np.linspace(0, 1, 7), q=rng.normal(size=(7, 5)), support=rng.random((7, 2)) > 0.5,
                swing=np.arange(7) % 3 - 1, joints=np.array(["left_hip_yaw", "right_knee"]), dt=np.array(0.0071),
                small=rng.normal(size=4).astype(np.float32))


def test_write_npz_bytes_equal_savez_and_no_tmp(tmp_path):
    arrays = _arrays()
    np.savez(tmp_path / "numpy.npz", **arrays)
    sha = s2io.write_npz(tmp_path / "refs" / "ours.npz", arrays)
    assert (tmp_path / "refs" / "ours.npz").read_bytes() == (tmp_path / "numpy.npz").read_bytes()
    assert sha == hashlib.sha256((tmp_path / "numpy.npz").read_bytes()).hexdigest() == s2io.sha256_file(
        tmp_path / "refs" / "ours.npz")
    assert sorted(p.name for p in (tmp_path / "refs").iterdir()) == ["ours.npz"]  # no temporary file, no ours.npz.npz
    back = np.load(tmp_path / "refs" / "ours.npz")
    assert back.files == list(arrays) and all(np.array_equal(back[k], v) for k, v in arrays.items())
    # written again, the same bytes (no time stamp inside)
    assert s2io.write_npz(tmp_path / "refs" / "ours.npz", arrays) == sha
    np.savez_compressed(tmp_path / "numpy_c.npz", **arrays)
    s2io.write_npz(tmp_path / "ours_c.npz", arrays, compressed=True)
    assert (tmp_path / "ours_c.npz").read_bytes() == (tmp_path / "numpy_c.npz").read_bytes()
    # an object array would be pickled: refused, nothing written
    with pytest.raises(s2io.S2Error, match="object"):
        s2io.write_npz(tmp_path / "bad.npz", dict(arrays, bad=np.array([{"a": 1}], dtype=object)))
    assert not (tmp_path / "bad.npz").exists() and not list(tmp_path.rglob("*.tmp"))


def test_write_npz_keeps_the_old_file_when_the_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "a.npz"
    sha = s2io.write_npz(path, dict(x=np.arange(3)))

    def boom(*a, **k):
        raise OSError("the disk is full")

    monkeypatch.setattr(s2io.np, "savez", boom)
    with pytest.raises(OSError):
        s2io.write_npz(path, dict(x=np.arange(4)))
    monkeypatch.undo()
    assert s2io.sha256_file(path) == sha and not list(tmp_path.rglob("*.tmp"))


@pytest.mark.skipif(not (RUN / "refs" / "nominal.npz").exists(), reason="the Stage 1 arrays are not restored")
def test_write_npz_reproduces_a_stage1_reference(tmp_path):
    """A Stage 1 reference's arrays, written through the file object, hash to the sha256 its record holds."""
    listed = json.loads((RUN / "refs.json").read_text())["references"]["nominal"]
    with np.load(RUN / "refs" / "nominal.npz") as z:
        arrays = {k: z[k] for k in z.files}
    assert s2io.write_npz(tmp_path / "nominal.npz", arrays) == listed["sha256"]
    tr = json.loads((RUN / "transitions.json").read_text())["transitions"]["start"]
    with np.load(RUN / "transitions" / "start.npz") as z:
        arrays = {k: z[k] for k in z.files}
    assert s2io.write_npz(tmp_path / "start.npz", arrays) == tr["sha256"]


def test_write_json_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "set" / "refs.json"
    doc = dict(b=[1, 2.5, None], a=dict(x=np.float64(0.25), y=(1, 2)), nan=math.nan)
    s2io.write_json(path, doc)
    text = path.read_text()
    assert text == json.dumps(dict(b=[1, 2.5, None], a=dict(x=0.25, y=[1, 2]), nan=math.nan), indent=1)  # Stage 1's layout
    assert list(json.loads(text)) == ["b", "a", "nan"] and math.isnan(s2io.read_json(path)["nan"])
    assert sorted(p.name for p in path.parent.iterdir()) == ["refs.json"]

    # a failure before the replace: the file that was there survives, whole, and no temporary file is left
    def boom(src, dst):
        assert Path(src).name == "refs.json.tmp" and Path(src).read_text() != text  # the new text was written beside it
        raise OSError("killed before the replace")

    monkeypatch.setattr(s2io.os, "replace", boom)
    with pytest.raises(OSError, match="killed"):
        s2io.write_json(path, dict(doc, b="changed"))
    monkeypatch.undo()
    assert path.read_text() == text and sorted(p.name for p in path.parent.iterdir()) == ["refs.json"]
    # a value that cannot be written changes nothing either
    with pytest.raises(TypeError):
        s2io.write_json(path, dict(doc, bad={1, 2}))
    assert path.read_text() == text and sorted(p.name for p in path.parent.iterdir()) == ["refs.json"]
    s2io.write_json(path, dict(doc, b="changed"))
    assert s2io.read_json(path)["b"] == "changed"


def test_sweep_tmp(tmp_path):
    for rel in ("refs/a.npz.tmp", "refs.json.tmp", "partial/x.json", "refs/a.npz", "deep/er/set.json.tmp"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    assert s2io.sweep_tmp(tmp_path) == ["deep/er/set.json.tmp", "refs.json.tmp", "refs/a.npz.tmp"]
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "partial/x.json", "refs/a.npz"]
    assert s2io.sweep_tmp(tmp_path) == []


# ------------------------------------------------------------------------------------------------
# Locks
# ------------------------------------------------------------------------------------------------
def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_lock_refuses_a_live_holder_and_takes_over_a_dead_pid(tmp_path):
    path = tmp_path / "set" / ".lock"
    logged = []
    with s2io.Lock(path) as lock:
        held = json.loads(path.read_text())
        assert held == lock.info and held["pid"] == os.getpid() and held["host"] == socket.gethostname()
        assert set(held) == {"pid", "host", "started_at", "boot_id", "proc_start"}
        with pytest.raises(s2io.S2Error, match=f"held by pid {os.getpid()} on {socket.gethostname()}"):
            s2io.Lock(path, log=logged.append).acquire()
        assert path.exists() and not logged
    assert not path.exists()  # released

    # a lock left by a process that is gone, on this host: taken over, and said so
    dead = dict(pid=_dead_pid(), host=socket.gethostname(), started_at="2026-11-14T00:00:00Z")
    path.write_text(json.dumps(dead))
    with s2io.Lock(path, log=logged.append) as lock:
        assert lock.took_over["pid"] == dead["pid"] and "not alive" in lock.took_over["reason"]
        assert json.loads(path.read_text())["pid"] == os.getpid()
        assert len(logged) == 1 and f"taken over from pid {dead['pid']}" in logged[0]
    assert not path.exists()

    # the same dead pid on another host is none of our business
    path.write_text(json.dumps(dict(dead, host="another-host")))
    with pytest.raises(s2io.S2Error, match="another-host"):
        s2io.Lock(path, log=logged.append).acquire()
    assert json.loads(path.read_text())["host"] == "another-host" and len(logged) == 1

    # a file that names no holder is not taken over
    path.write_text("")
    with pytest.raises(s2io.S2Error, match="names no holder"):
        s2io.Lock(path).acquire()
    path.unlink()

    # a lock that is no longer ours is not removed on release
    lock = s2io.Lock(path).acquire()
    other = dict(pid=1, host="another-host", started_at="x")
    path.write_text(json.dumps(other))
    lock.release()
    assert json.loads(path.read_text()) == other


@pytest.mark.skipif(s2io._proc_start(os.getpid()) is None, reason="no /proc on this system")
def test_lock_takes_over_a_reused_pid(tmp_path):
    """After a restart a new process can carry the holder's pid (this very one, even): the start time or the boot
    tells them apart."""
    path = tmp_path / ".lock"
    logged = []
    me = dict(pid=os.getpid(), host=socket.gethostname(), started_at="2026-11-14T00:00:00Z", boot_id=s2io._boot_id(),
              proc_start=s2io._proc_start(os.getpid()))
    path.write_text(json.dumps(me))  # really this process: refused
    with pytest.raises(s2io.S2Error, match="held by pid"):
        s2io.Lock(path, log=logged.append).acquire()
    path.write_text(json.dumps(dict(me, proc_start=me["proc_start"] - 1)))  # this pid, another process
    with s2io.Lock(path, log=logged.append) as lock:
        assert "another process" in lock.took_over["reason"]
    if me["boot_id"]:
        path.write_text(json.dumps(dict(me, boot_id="00000000-0000-0000-0000-000000000000")))  # an earlier boot
        with s2io.Lock(path, log=logged.append) as lock:
            assert "booted" in lock.took_over["reason"]
    assert len(logged) == (2 if me["boot_id"] else 1)


# ------------------------------------------------------------------------------------------------
# Provenance
# ------------------------------------------------------------------------------------------------
def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
                    *args], cwd=cwd, check=True, capture_output=True)


@needs_git
def test_built_by_leaves_own_outputs_out_of_dirty_paths(tmp_path, monkeypatch):
    """On a throwaway repository whose study directory is two levels down, as here: the builder's own output
    directory is left out of ``dirty_paths`` and of ``dirty``; anything else that differs from the commit stays."""
    repo = tmp_path / "repo"
    study = repo / "hardware" / "study"
    (study / "designlab" / "stage2").mkdir(parents=True)
    (study / "designlab" / "stage2" / "moves.py").write_text("x = 1\n")
    (repo / "outside.txt").write_text("not the study's\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    monkeypatch.setattr(s2io, "STUDY", study)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert s2io.git_state() == dict(commit=head, dirty=False, dirty_paths=[])

    out = study / "stage2" / "sets" / "V0" / "75-M-H-10"
    out.mkdir(parents=True)
    (out / "refs.json").write_text("{}")
    (out / "set.json").write_text("{}")
    (repo / "outside.txt").write_text("changed, outside the study\n")
    own = ["stage2/sets/V0/75-M-H-10"]
    assert s2io.git_state()["dirty_paths"] == ["?? hardware/study/stage2/sets/V0/75-M-H-10/refs.json",
                                               "?? hardware/study/stage2/sets/V0/75-M-H-10/set.json"]
    assert s2io.git_state(own) == dict(commit=head, dirty=False, dirty_paths=[])
    assert s2io.git_state([str(out)]) == s2io.git_state(own) == s2io.git_state(["stage2/sets/V0/75-M-H-10/"])
    assert s2io.git_state(["stage2/sets/V0/75-M-H-10/refs.json"])["dirty_paths"] == [
        "?? hardware/study/stage2/sets/V0/75-M-H-10/set.json"]
    assert s2io.git_state(["stage2/sets/V0/75-M-H"])["dirty"] is True  # a name's beginning is not its directory

    (study / "designlab" / "stage2" / "moves.py").write_text("x = 2\n")  # the code itself differs from the commit
    state = s2io.git_state(own)
    assert state["dirty"] is True and state["dirty_paths"] == [" M hardware/study/designlab/stage2/moves.py"]

    # built_by: the same state plus the module hashes (the Stage 1 parts are stubbed: this repository has none)
    import designlab.stage1.panels as panels

    monkeypatch.setattr(panels, "panel_sources", lambda: dict(sha256="p" * 64))
    b = s2io.built_by(["moves.py"], own_outputs=own)
    assert (b["commit"], b["dirty"], b["dirty_paths"]) == (head, True, state["dirty_paths"])
    assert b["stage2_modules"] == {"designlab/stage2/moves.py": hashlib.sha256(b"x = 2\n").hexdigest()}
    assert b["panel_sources_sha256"] == "p" * 64

    monkeypatch.setattr(s2io, "STUDY", tmp_path / "not-a-repository")
    assert s2io.git_state() == dict(commit=None, dirty=None, dirty_paths=None)


@needs_git
def test_built_by_here():
    """On this checkout: the commit, the panels' source hash, the named modules by their study-relative path, the
    versions, the thread count, the accuracy level at the call and a UTC time."""
    from designlab import gait
    from designlab.stage1 import panels

    before = gait.ACCURACY
    with s2io.r2():
        b = s2io.built_by(["s2io.py", "designlab/stage2/setio.py", str(s2io.STUDY / "designlab" / "stage2" / "readings.py")],
                          own_outputs=["stage2/sets/V0/75-M-H-10"])
    assert gait.ACCURACY == before
    assert list(b) == ["commit", "dirty", "dirty_paths", "panel_sources_sha256", "stage2_modules", "versions",
                       "blas_threads", "accuracy", "at"]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert b["commit"] == (head or None)
    assert b["panel_sources_sha256"] == panels.panel_sources()["sha256"] and b["versions"] == panels.library_versions()
    assert b["stage2_modules"] == {f"designlab/stage2/{n}": s2io.sha256_file(ROOT / "designlab" / "stage2" / n)
                                   for n in ("s2io.py", "setio.py", "readings.py")}
    assert b["accuracy"] == "r2"  # the level inside the item's block, where it was called
    try:  # ... and outside a block the record says what the level was, not "r2"
        gait.set_accuracy("r1")
        assert s2io.built_by(["s2io.py"])["accuracy"] == "r1"
    finally:
        gait.set_accuracy(before)
    assert b["blas_threads"] == s2io.blas_threads()
    assert len(b["at"]) == 20 and b["at"].endswith("Z") and b["at"][10] == "T"
    if b["dirty_paths"] is not None:
        assert b["dirty"] == bool(b["dirty_paths"])
        assert all("stage2/sets/V0/75-M-H-10/" not in line for line in b["dirty_paths"])
    json.dumps(b, allow_nan=False)  # plain JSON
