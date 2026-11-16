"""Tests for Track B's Drive bundle (designlab/stage2/bundle.py) and Track B's ignore file (stage2/tb/.gitignore).

Run from hardware/trex_form_study (the study's venv):  python -m pytest -p no:cacheprovider tests/test_stage2_bundle.py
All fast.  Nothing is written into the checkout: every bundle is built, verified and unpacked under pytest's temporary
directory, most of them from a synthetic study tree made there (two synthetic bodies registered for the test, their
model files and records, a frozen reference set per body).  The tests on the Stage 1 picks' real model files and on a
real frozen set skip where those git-ignored files are not here.  No test opens a network connection.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import importlib.util
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab import stage1  # noqa: E402
from designlab.stage2 import bundle, s2io  # noqa: E402

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
IGNORE = ROOT / "stage2" / "tb" / ".gitignore"
IGNORE_LINES = ["work/", "*.tar", "*.zip", "*.npz", "*.pkl", "*.mp4", "*.tmp", "*.part", ".lock", "logs/", "partial/",
                "traces/", "checkpoints/", "parts/"]
#: The synthetic bodies: name -> refine case directory (their candidate id is their name).
CASES = {"T-A": "round2/refine/t_case_a", "T-B": "round2/refine/t_case_b"}
BODIES = sorted(CASES)
#: A synthetic set's frozen entries: (kind, name, bytes).  Sorted by path the sizes split into four parts at 60 kB.
SET_FILES = (("walk", "nominal", 9000), ("walk", "v0.4", 9000), ("change", "nominal_to_v0.4", 5000),
             ("start", "start", 14000), ("stop_two_step", "stop2_v0.4", 21000), ("turn", "turn90_pi4_left", 30000),
             ("turn", "turn90_pi4_right", 30000))
FIRST_FREEZE = ("nominal", "start")
PICKS = ("75-M-H-10", "60-N-T-10")


# ------------------------------------------------------------------------------------------------
# A synthetic study tree
# ------------------------------------------------------------------------------------------------
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob(tag: str, n: int) -> bytes:
    return random.Random(tag).randbytes(n)  # the same bytes in every process


def _put(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data, indent=1).encode())


def make_set(root: Path, set_name: str, body: str, freezes: int = 2) -> dict:
    """A reference set in round A's layout with ``freezes`` freezes (the first lists two entries, the second all of
    ``SET_FILES``), and what a bundle must leave out: a look-format twin per walk, a walk that is in ``refs.json``
    and in no freeze, and an array no record lists.  The latest manifest is returned."""
    d = root / bundle.SETS_DIR / set_name / body
    refs, moves, rows = {}, {}, {}
    for kind, name, n in SET_FILES:
        sub = "refs" if kind == "walk" else "transitions"
        data = _blob(f"{set_name}/{body}/{sub}/{name}", n)
        _put(d / sub / f"{name}.npz", data)
        e = dict(name=name, file=f"{sub}/{name}.npz", sha256=_sha(data), frames=10, available=True)
        row = dict(kind=kind, sha256=_sha(data), entry_sha256=_sha(b"entry" + data))
        if kind == "walk":
            look = _blob(f"{set_name}/{body}/look/{name}", 3000)
            _put(d / "look" / f"{name}.npz", look)
            e.update(look_file=f"look/{name}.npz", look_sha256=_sha(look))
            row["look_sha256"] = _sha(look)
            refs[name] = e
        else:
            moves[name] = e
        rows[name] = row
    late = _blob(f"{set_name}/{body}/late", 4000)
    _put(d / "refs" / "v0.5.npz", late)
    refs["v0.5"] = dict(name="v0.5", file="refs/v0.5.npz", sha256=_sha(late), available=True)
    _put(d / "transitions" / "stray.npz", _blob("stray", 2000))
    first = {n: rows[n] for n in FIRST_FREEZE}
    fz = [dict(index=0, manifest=first, manifest_sha256=s2io.sha256_json(first)),
          dict(index=1, manifest=rows, manifest_sha256=s2io.sha256_json(rows))][:freezes]
    _put(d / "set.json", dict(set=set_name, body=body, freezes=fz))
    _put(d / "refs.json", dict(references=refs))
    _put(d / "transitions.json", dict(transitions=moves))
    return rows


def make_tree(root: Path, sets=("SX",)) -> dict:
    """Two synthetic bodies' model files, candidate records and inputs lists, and a frozen set per (set, body);
    returns what a process must register to build from it: ``{bodies: {name: case}, prefix: {name: 8 hex}}``."""
    reg = dict(bodies={}, prefix={})
    lists = {bundle.INPUT_LISTS[0]: {}, bundle.INPUT_LISTS[2]: {}}
    for (body, case), rel in zip(sorted(CASES.items()), lists, strict=True):
        xml, ledger = _blob(f"{body}.xml", 30_000), _blob(f"{body}.json", 9_000)
        _put(root / case / "models" / f"{body}.xml", xml)
        _put(root / case / "models" / f"{body}.json", ledger)
        _put(root / case / "candidates" / f"{body}.json", dict(id=body, xml_sha256=_sha(xml)))
        lists[rel].update({f"{case}/models/{body}.xml": _sha(xml), f"{case}/models/{body}.json": _sha(ledger)})
        reg["bodies"][body], reg["prefix"][body] = case, _sha(xml)[:8]
        for set_name in sets:
            make_set(root, set_name, body)
    for rel, files in lists.items():
        _put(root / rel, dict(rule="test", files=files))
    return reg


def _register(monkeypatch, reg: dict) -> None:
    for body, case in reg["bodies"].items():
        monkeypatch.setitem(stage1.BODIES, body, case)
        monkeypatch.setitem(stage1.SHA256_PREFIX, body, reg["prefix"][body])


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "study"
    _register(monkeypatch, make_tree(root))
    return root


@pytest.fixture
def built(tree, tmp_path):
    """(tree, out directory, manifest) of a bundle of both bodies and the set SX, the set split at 60 kB a part."""
    out = tmp_path / "out"
    return tree, out, bundle.build("t1", BODIES, sets=("SX",), out=out, part_mb=0.06, root=tree)


def _expected_files(tree: Path, sets=("SX",)) -> dict:
    """What a bundle of the synthetic tree must hold: path -> bytes on disk."""
    out = {}
    for body, case in CASES.items():
        for suffix in ("xml", "json"):
            out[f"{case}/models/{body}.{suffix}"] = None
        for set_name in sets:
            for kind, name, _ in SET_FILES:
                sub = "refs" if kind == "walk" else "transitions"
                out[f"{bundle.SETS_DIR}/{set_name}/{body}/{sub}/{name}.npz"] = None
    return {rel: (tree / rel).read_bytes() for rel in out}


def _files_under(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if not p.is_dir()) if root.exists() else []


def _run(args, **kw):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.update(kw.pop("env", {}))
    return subprocess.run([sys.executable, *args], cwd=kw.pop("cwd", ROOT), env=env, capture_output=True, text=True,
                          **kw)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                          capture_output=True, text=True, check=True).stdout


# ------------------------------------------------------------------------------------------------
# The ignore file
# ------------------------------------------------------------------------------------------------
IGNORED = ["work/x", "bundle/b/parts/a.tar", "runs/r/jobs/j/checkpoints/model_1.zip", "x.npz", "x.pkl", "clip_1.mp4",
           "logs/a.log", "a.part", "partial/x.json", "traces/t.npz",
           # live files and the rest of what Track B writes and never commits
           "work/runs/r/LIVE.json", "work/runs/r/STATUS.json", "work/r2l/fit/gen_3.json",
           "runs/r/jobs/j/checkpoints/vecnormalize_1.pkl", "runs/r/jobs/j/checkpoints/fingerprint_1.npz",
           "runs/r/jobs/j/checkpoints/checkpoint_1.json", "runs/r/jobs/j/ckpt_1.tar",
           "runs/r/jobs/j/monitor/clip_1.mp4", "runs/r/logs/launcher.s1.log", "runs/r/jobs/j/model_1.zip.part",
           "bundle/b/.lock", "bundle/b/MANIFEST.json.tmp", "bundle/b/parts/a.tar.tmp"]
NOT_IGNORED = ["bundle/b/MANIFEST.json", "runs/r/jobs/j/run.json", "gates/V6/b/g8a.json",
               "bundle/b/UPLOADS.json", "bundle/b/ACCEPTANCE.json", "runs/r/STATUS.json", "colab_steps.txt",
               "validation/rates.json", ".gitignore"]


def test_the_ignore_file_holds_exactly_its_lines():
    assert IGNORE.read_text() == "".join(line + "\n" for line in IGNORE_LINES)


def _scratch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "stage2" / "tb").mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(repo))
    shutil.copyfile(IGNORE, repo / "stage2" / "tb" / ".gitignore")
    return repo


@needs_git
def test_git_check_ignore_on_every_output_path(tmp_path):
    """In a scratch repository that holds only this ignore file, at ``stage2/tb/``: every live or binary output path
    is ignored, by a line of this file; every record that is committed is not."""
    repo = _scratch_repo(tmp_path)
    paths = [f"stage2/tb/{p}" for p in IGNORED + NOT_IGNORED]
    got = subprocess.run(["git", "-C", str(repo), "check-ignore", "-v", "-n", "--stdin"], input="\n".join(paths),
                         capture_output=True, text=True)
    by_path = {}
    for line in got.stdout.splitlines():
        source, path = line.split("\t")
        by_path[path] = source
    assert sorted(by_path) == sorted(paths), got.stderr
    for p in IGNORED:
        src, _, pattern = by_path[f"stage2/tb/{p}"].split(":", 2)
        assert src == "stage2/tb/.gitignore" and pattern in IGNORE_LINES, (p, by_path[f"stage2/tb/{p}"])
    for p in NOT_IGNORED:
        assert by_path[f"stage2/tb/{p}"] == "::", (p, by_path[f"stage2/tb/{p}"])
    # and with the files really there: git status lists the committed kind only
    for p in paths:
        _put(repo / p, b"x") if not p.endswith(".gitignore") else None
    status = _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    assert sorted(x[3:] for x in status) == sorted(f"stage2/tb/{p}" for p in NOT_IGNORED)


@needs_git
def test_a_build_under_stage2_tb_leaves_only_the_manifest_for_git(tree, tmp_path):
    """``build`` into ``stage2/tb/bundle/<id>`` of a repository: the parts are ignored, ``MANIFEST.json`` is not, and
    nothing else is left (no lock, no temporary file)."""
    repo = _scratch_repo(tmp_path)
    out = repo / "stage2" / "tb" / "bundle" / "t1"
    doc = bundle.build("t1", BODIES, sets=("SX",), out=out, part_mb=0.06, root=tree)
    status = _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    assert sorted(x[3:] for x in status) == ["stage2/tb/.gitignore", "stage2/tb/bundle/t1/MANIFEST.json"]
    assert _files_under(out) == ["MANIFEST.json"] + sorted(f"parts/{n}" for n in doc["content"]["parts"])
    ignored = _git(repo, "status", "--porcelain", "--ignored", "--untracked-files=all").splitlines()
    assert sorted(x[3:] for x in ignored if x.startswith("!!")) == sorted(
        f"stage2/tb/bundle/t1/parts/{n}" for n in doc["content"]["parts"])


# ------------------------------------------------------------------------------------------------
# Determinism and the tar rules
# ------------------------------------------------------------------------------------------------
BUILD_SCRIPT = """
import json, os, sys
job = json.loads(sys.argv[1])
if job["umask"] is not None:
    os.umask(job["umask"])
from designlab import stage1
stage1.BODIES.update(job["bodies"])
stage1.SHA256_PREFIX.update(job["prefix"])
from designlab.stage2 import bundle
doc = bundle.build("t1", sorted(job["bodies"]), sets=job["sets"], out=job["out"], part_mb=job["part_mb"],
                   root=job["root"])
print(bundle.content_sha256(doc))
"""


def test_two_processes_and_another_umask_give_the_same_bytes(tmp_path):
    """Two builds of the same files in two processes, the second under umask 077, with another hash seed, time zone
    and working directory: byte-identical parts and one content hash."""
    root = tmp_path / "study"
    reg = make_tree(root)
    outs, hashes = [], []
    for i, (umask, env, cwd) in enumerate(((None, dict(PYTHONHASHSEED="0", TZ="UTC"), ROOT),
                                           (0o077, dict(PYTHONHASHSEED="4321", TZ="Asia/Tokyo"), tmp_path))):
        out = tmp_path / f"out{i}"
        job = dict(reg, umask=umask, sets=["SX"], out=str(out), part_mb=0.06, root=str(root))
        got = _run(["-c", BUILD_SCRIPT, json.dumps(job)], env=env, cwd=cwd)
        assert got.returncode == 0, got.stderr
        outs.append(out)
        hashes.append(got.stdout.strip())
    a, b = (sorted(p.name for p in (o / "parts").iterdir()) for o in outs)
    assert a == b and len(a) > 4
    for name in a:
        assert (outs[0] / "parts" / name).read_bytes() == (outs[1] / "parts" / name).read_bytes(), name
        assert (outs[1] / "parts" / name).stat().st_mode & 0o777 == 0o600  # the umask was really another one
    docs = [json.loads((o / "MANIFEST.json").read_text()) for o in outs]
    assert hashes[0] == hashes[1] == docs[0]["content_sha256"] == docs[1]["content_sha256"]
    assert docs[0]["content"] == docs[1]["content"]


def test_tar_rules(built):
    """USTAR, sorted members named by their study path, mtime 0, uid and gid 0, no user or group name, mode 0644,
    regular files, no compression (a member's bytes stand in the part as they are), and the predicted size."""
    tree, out, doc = built
    expected = _expected_files(tree)
    for name, part in doc["content"]["parts"].items():
        path = out / "parts" / name
        raw = path.read_bytes()
        with tarfile.open(path, "r:") as tar:
            infos = tar.getmembers()
        assert [i.name for i in infos] == sorted(i.name for i in infos)
        at = 0
        for i in infos:
            assert (i.mtime, i.uid, i.gid, i.uname, i.gname, i.mode, i.type) == (0, 0, 0, "", "", 0o644,
                                                                               tarfile.REGTYPE)
            assert (i.offset, i.offset_data) == (at, at + 512) and not i.pax_headers  # one plain header per member
            assert raw[at + 257:at + 265] == b"ustar\x0000" and raw[at + 156:at + 157] == b"0"
            assert raw[i.offset_data:i.offset_data + i.size] == expected[i.name]
            assert doc["content"]["files"][i.name]["part"] == name
            at = i.offset_data + -(-i.size // 512) * 512
        assert not any(raw[at:])  # the end blocks and the padding, nothing else
        assert len(raw) == part["bytes"] == bundle.tar_bytes([i.size for i in infos]) and part["files"] == len(infos)
        assert s2io.sha256_file(path) == part["sha256"]


def test_tar_bytes():
    assert bundle.tar_bytes([]) == 10240 and bundle.tar_bytes([1]) == 10240
    assert bundle.tar_bytes([512 * 17]) == 10240 and bundle.tar_bytes([512 * 17 + 1]) == 20480
    assert bundle.tar_bytes([9000, 9000, 5000, 14000]) == 40960


# ------------------------------------------------------------------------------------------------
# What is packed
# ------------------------------------------------------------------------------------------------
def test_a_bundle_holds_the_models_and_the_latest_freeze_and_nothing_else(built):
    """Per body the two model files, per set and body the files of the latest freeze with its hashes: never a
    look-format twin, never a file the freeze does not list (though ``refs.json`` lists it), never a stray array."""
    tree, out, doc = built
    c, expected = doc["content"], _expected_files(tree)
    assert doc["schema"] == "s2-bundle-2" and sorted(doc) == ["content", "content_sha256", "provenance", "schema"]
    assert sorted(c) == ["bodies", "bundle_id", "files", "models", "parts", "sets"]
    assert c["bundle_id"] == "t1" and c["bodies"] == BODIES
    assert sorted(c["files"]) == sorted(expected)
    assert not [f for f in c["files"] if "/look/" in f or f.endswith(("v0.5.npz", "stray.npz"))]
    for rel, e in c["files"].items():
        assert e == dict(sha256=_sha(expected[rel]), bytes=len(expected[rel]), part=e["part"])
    for body, case in CASES.items():
        m = c["models"][body]
        assert m["case"] == Path(case).name
        assert m["xml"] == dict(path=f"{case}/models/{body}.xml", sha256=c["files"][m["xml"]["path"]]["sha256"])
        assert m["record"]["path"] == f"{case}/models/{body}.json"
        assert c["files"][m["xml"]["path"]]["part"] == c["files"][m["record"]["path"]]["part"] == f"models-{body}.tar"
        latest = json.loads((tree / bundle.SETS_DIR / "SX" / body / "set.json").read_text())["freezes"][-1]
        assert c["sets"]["SX"][body] == dict(freeze_index=1, manifest_sha256=latest["manifest_sha256"],
                                             files=len(SET_FILES))
    assert doc["content_sha256"] == bundle.content_sha256(doc) == s2io.sha256_json(c)
    p = doc["provenance"]
    assert {"made_at", "commit", "dirty", "rule", "tool_sha256", "versions"} <= set(p)
    assert p["rule"] == bundle.RULE and p["tool_sha256"] == s2io.sha256_file(bundle.__file__)
    assert p["records_committed"] is None  # the synthetic tree is no git checkout
    assert sorted(p["records"]) == sorted(
        [f"{case}/candidates/{body}.json" for body, case in CASES.items()]
        + [f"{bundle.SETS_DIR}/SX/{body}/{n}" for body in CASES for n in ("set.json", "refs.json", "transitions.json")]
        + [bundle.INPUT_LISTS[0], bundle.INPUT_LISTS[2]])
    assert all(p["hash_sources"][rel] for rel in c["files"])


def test_sets_empty_packs_the_models_only(tree, tmp_path):
    doc = bundle.build("m", BODIES, sets=(), out=tmp_path / "m", root=tree)
    assert doc["content"]["sets"] == {} and sorted(doc["content"]["parts"]) == [f"models-{b}.tar" for b in BODIES]
    assert len(doc["content"]["files"]) == 4


def test_a_set_over_the_part_size_splits_and_every_file_is_in_exactly_one_part(built, tree, tmp_path):
    tree, out, doc = built
    c, limit = doc["content"], 60_000
    seen = []
    for body in BODIES:
        names = sorted(n for n in c["parts"] if n.startswith(f"SX-{body}-"))
        assert names == [f"SX-{body}-{i}.tar" for i in range(1, len(names) + 1)] and len(names) == 4
        order = []
        for name in names:
            with tarfile.open(out / "parts" / name, "r:") as tar:
                order += tar.getnames()
        assert order == sorted(order)  # the set's files in sorted order across its parts
        seen += order
    for body in BODIES:
        with tarfile.open(out / "parts" / f"models-{body}.tar", "r:") as tar:
            seen += tar.getnames()
    assert sorted(seen) == sorted(c["files"]) and len(seen) == len(set(seen))
    assert all(p["bytes"] <= limit for p in c["parts"].values())
    assert sum(p["files"] for p in c["parts"].values()) == len(c["files"])
    # a larger limit: one part per set and body, and another content hash
    one = bundle.build("t2", BODIES, sets=("SX",), out=tmp_path / "one", part_mb=20, root=tree)
    assert sorted(one["content"]["parts"]) == sorted([f"models-{b}.tar" for b in BODIES]
                                                     + [f"SX-{b}-1.tar" for b in BODIES])
    assert {f: (e["sha256"], e["bytes"]) for f, e in one["content"]["files"].items()} == {
        f: (e["sha256"], e["bytes"]) for f, e in c["files"].items()}
    # a file that alone is over the limit is refused with its path, and nothing is written
    with pytest.raises(bundle.BundleError, match="turn90_pi4_left.npz.*over the limit"):
        bundle.build("t3", BODIES, sets=("SX",), out=tmp_path / "small", part_mb=0.03, root=tree)
    assert _files_under(tmp_path / "small") == []


def test_the_ledger_hash_where_no_record_holds_it(tree, tmp_path):
    """A model ledger JSON whose sha256 no committed list holds is packed with its own, and the provenance says so."""
    (tree / bundle.INPUT_LISTS[2]).unlink()
    doc = bundle.build("m", BODIES, sets=(), out=tmp_path / "m", root=tree)
    xml, ledger = (f"{CASES['T-B']}/models/T-B.{s}" for s in ("xml", "json"))
    assert doc["content"]["files"][ledger]["sha256"] == s2io.sha256_file(tree / ledger)
    assert doc["provenance"]["hash_sources"][ledger] == []
    assert doc["provenance"]["hash_sources"][xml] == [f"{CASES['T-B']}/candidates/T-B.json: xml_sha256"]


def _edit(path: Path, change) -> None:
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc, indent=1))


def _latest(doc):
    return doc["freezes"][-1]


def _refreeze(doc, change) -> None:
    change(_latest(doc)["manifest"])
    _latest(doc)["manifest_sha256"] = s2io.sha256_json(_latest(doc)["manifest"])


SETD = f"{bundle.SETS_DIR}/SX/T-A"
MODEL_A, MODEL_B = f"{CASES['T-A']}/models/T-A", f"{CASES['T-B']}/models/T-B"
CANDIDATE_A = f"{CASES['T-A']}/candidates/T-A.json"


def _list_says_another_xml(t):
    _edit(t / bundle.INPUT_LISTS[0], lambda d: d["files"].update({f"{MODEL_A}.xml": "0" * 64}))


def _record_says_another_array(t):
    _edit(t / SETD / "refs.json", lambda d: d["references"]["v0.4"].update(sha256="1" * 64))


def _freeze_changed_without_its_hash(t):
    _edit(t / SETD / "set.json", lambda d: _latest(d)["manifest"]["start"].update(sha256="2" * 64))


def _freeze_lists(name, kind):
    row = dict(kind=kind, sha256="3" * 64, entry_sha256="4" * 64)
    return lambda t: _edit(t / SETD / "set.json", lambda d: _refreeze(d, lambda m: m.update({name: row})))


def _list_holds_no_sha256(t):
    _edit(t / bundle.INPUT_LISTS[0], lambda d: d["files"].update({f"{MODEL_A}.json": ["0" * 64]}))


#: what is spoiled in the tree -> (how, the start of the refusal's sentence about it)
REFUSALS = {
    "a missing model file": (lambda t: (t / f"{MODEL_A}.json").unlink(), f"{MODEL_A}.json: missing"),
    "a changed model file": (lambda t: (t / f"{MODEL_B}.xml").write_bytes(b"<mujoco/>"), f"{MODEL_B}.xml: sha256"),
    "a changed ledger": (lambda t: (t / f"{MODEL_A}.json").write_bytes(b"{}"), f"{MODEL_A}.json: sha256"),
    "no candidate record": (lambda t: (t / CANDIDATE_A).unlink(), f"{CANDIDATE_A}: the record cannot be read"),
    "records that disagree": (_list_says_another_xml, f"{MODEL_A}.xml: the committed records disagree"),
    "a set that is not there": (lambda t: shutil.rmtree(t / SETD), f"{SETD}: set.json is missing"),
    "a set that was never frozen": (lambda t: _edit(t / SETD / "set.json", lambda d: d.update(freezes=[])),
                                    f"{SETD}/set.json: the set is not frozen"),
    "a missing array": (lambda t: (t / SETD / "transitions" / "start.npz").unlink(),
                        f"{SETD}/transitions/start.npz: missing"),
    "a changed array": (lambda t: (t / SETD / "refs" / "nominal.npz").write_bytes(b"other"),
                        f"{SETD}/refs/nominal.npz: sha256"),
    "a freeze its record contradicts": (_record_says_another_array, f"{SETD}/refs/v0.4.npz: freeze 1"),
    "a freeze whose hash is not its manifest's": (_freeze_changed_without_its_hash,
                                                  f"{SETD}/set.json: freeze 1's manifest_sha256"),
    "a frozen entry with no record": (_freeze_lists("ghost", "turn"), f"{SETD}/transitions/ghost.npz: in freeze 1"),
    "a frozen entry of no kind": (_freeze_lists("odd", "dance"), f"{SETD}/set.json: freeze 1 lists 'odd'"),
    "a frozen name that is a path": (_freeze_lists("../../x", "walk"), "the frozen name '../../x' holds '..'"),
    # records that are not what they must be: each a refusal with the record's path, never another exception
    "a set record that is not JSON": (lambda t: (t / SETD / "set.json").write_text("{not json"),
                                      f"{SETD}/set.json: the record is not JSON"),
    "a move record that is not JSON": (lambda t: (t / SETD / "transitions.json").write_text("{not json"),
                                       f"{SETD}/transitions.json: the record is not JSON"),
    "a set record nested too deep": (lambda t: (t / SETD / "set.json").write_text("[" * 100_000 + "]" * 100_000),
                                     f"{SETD}/set.json: the record is not JSON"),
    "a set record that is a list": (lambda t: (t / SETD / "set.json").write_text("[1]"),
                                    f"{SETD}/set.json: the record is not a JSON object"),
    "a walk record that is a list": (lambda t: (t / SETD / "refs.json").write_text("[1]"),
                                     f"{SETD}/refs.json: the record is not a JSON object"),
    "a freeze without its index": (lambda t: _edit(t / SETD / "set.json", lambda d: _latest(d).pop("index")),
                                   f"{SETD}/set.json: its latest freeze is not"),
    "a freeze that is a number": (lambda t: _edit(t / SETD / "set.json", lambda d: d.update(freezes=[3])),
                                  f"{SETD}/set.json: its latest freeze is not"),
    "freezes that are a table": (lambda t: _edit(t / SETD / "set.json", lambda d: d.update(freezes={"a": 1})),
                                 f"{SETD}/set.json: its latest freeze is not"),
    "a freeze whose manifest is a list": (
        lambda t: _edit(t / SETD / "set.json",
                        lambda d: _latest(d).update(manifest=[1], manifest_sha256=s2io.sha256_json([1]))),
        f"{SETD}/set.json: freeze 1's manifest is not a table"),
    "a record whose table is a list": (lambda t: _edit(t / SETD / "refs.json", lambda d: d.update(references=[1])),
                                       f"{SETD}/refs/nominal.npz: in freeze 1, and refs.json has no entry"),
    "a list that holds no sha256": (_list_holds_no_sha256,
                                    f"{bundle.INPUT_LISTS[0]}: its entry for {MODEL_A}.json is no sha256"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_build_refuses_and_writes_nothing(case, tree, tmp_path):
    """Each difference between the records and the files is refused with its path, nothing is written, and no model
    is rebuilt (``plant.model_files`` would rebuild a missing one into ``stage1/models``)."""
    change, text = REFUSALS[case]
    change(tree)
    before = _files_under(tree)
    with pytest.raises(bundle.BundleError) as exc:
        bundle.build("t1", BODIES, sets=("SX",), out=tmp_path / "out", root=tree)
    assert text in str(exc.value), str(exc.value)
    assert _files_under(tmp_path / "out") == [] and _files_under(tree) == before
    assert not (tree / "stage1" / "models").exists()


def test_build_refuses_bad_arguments(tree, tmp_path, monkeypatch):
    out = tmp_path / "out"
    for kw, text in ((dict(bundle_id="a/b"), "not a name"), (dict(bundle_id=""), "not a name"),
                     (dict(bodies=["T-A", "T-A"]), "given twice"), (dict(bodies=[]), "no body"),
                     (dict(bodies=["T-Z"]), "not a Stage 1 body"), (dict(bodies="T-A"), "a list of names"),
                     (dict(sets="SX"), "a list of names"), (dict(sets=["../x"]), "not a name"),
                     (dict(part_mb=0), "positive number"), (dict(part_mb=True), "positive number")):
        args = dict(dict(bundle_id="t1", bodies=BODIES, sets=("SX",), out=out, root=tree), **kw)
        with pytest.raises(bundle.BundleError, match=text):
            bundle.build(**args)
    assert _files_under(out) == []
    # the prefix the plan lists for a body is a second check on its record
    monkeypatch.setitem(stage1.SHA256_PREFIX, "T-A", "ffffffff")
    with pytest.raises(bundle.BundleError, match="lacks the plan's prefix ffffffff for T-A"):
        bundle.build("t1", BODIES, sets=(), out=out, root=tree)


def test_a_name_ustar_cannot_hold_is_refused(tree, tmp_path):
    """The parts are USTAR and nothing else: a file name over USTAR's 100 characters is refused, never written with
    a PAX or GNU extension header (which another tar reader may take differently), and nothing is left."""
    name = "w" * 120
    data = _blob("a long name", 1000)
    _put(tree / SETD / "refs" / f"{name}.npz", data)
    _edit(tree / SETD / "refs.json", lambda d: d["references"].update(
        {name: dict(name=name, file=f"refs/{name}.npz", sha256=_sha(data), available=True)}))
    row = dict(kind="walk", sha256=_sha(data), entry_sha256=_sha(b"entry" + data))
    _edit(tree / SETD / "set.json", lambda d: _refreeze(d, lambda m: m.update({name: row})))
    with pytest.raises(bundle.BundleError, match=r"SX-T-A-1\.tar\.tmp: cannot be written as USTAR"):
        bundle.build("t1", BODIES, sets=("SX",), out=tmp_path / "out", root=tree)
    assert _files_under(tmp_path / "out") == []


def test_the_order_of_the_names_given_does_not_matter(tmp_path, monkeypatch):
    """Bodies and sets are sorted before anything is built: given in another order they are one content, with the
    same parts byte for byte."""
    root = tmp_path / "study"
    _register(monkeypatch, make_tree(root, sets=("SX", "SY")))
    a = bundle.build("o", BODIES, sets=("SX", "SY"), out=tmp_path / "a", part_mb=0.06, root=root)
    b = bundle.build("o", BODIES[::-1], sets=("SY", "SX"), out=tmp_path / "b", part_mb=0.06, root=root)
    assert a["content"]["bodies"] == BODIES and list(a["content"]["sets"]) == ["SX", "SY"]
    assert b["content"] == a["content"] and b["content_sha256"] == a["content_sha256"]
    assert json.dumps(b["content"]) == json.dumps(a["content"])  # the tables' own order too
    for name in a["content"]["parts"]:
        assert (tmp_path / "a" / "parts" / name).read_bytes() == (tmp_path / "b" / "parts" / name).read_bytes(), name
    # and the same bundle directory takes the other order as the same build
    again = bundle.build("o", BODIES[::-1], sets=("SY", "SX"), out=tmp_path / "a", part_mb=0.06, root=root)
    assert again == a


def test_a_file_that_changes_while_it_is_packed_is_refused(tree, tmp_path, monkeypatch):
    """Between the hash that goes to the manifest and the bytes that go to the part a file can change: every part is
    read back against the manifest before any part is moved into place."""
    changed, real = f"{MODEL_B}.xml", bundle._write_part

    def write(path, root, members):
        if any(rel == changed for rel, _, _ in members):  # hashed already; now one bit of it changes
            data = bytearray((root / changed).read_bytes())
            data[10] ^= 1
            (root / changed).write_bytes(data)
        real(path, root, members)

    monkeypatch.setattr(bundle, "_write_part", write)
    with pytest.raises(bundle.BundleError, match=f"models-T-B.tar.tmp: member '{changed}' has 30000 bytes and sha256"):
        bundle.build("t1", BODIES, sets=("SX",), out=tmp_path / "out", root=tree)
    assert _files_under(tmp_path / "out") == []  # not the parts written before it either


def test_a_build_removes_the_temporary_parts_a_killed_build_left(tree, tmp_path):
    """A killed build leaves ``<part>.tar.tmp`` files; the next build removes them all (it holds the lock), also the
    ones of a build with other part names, and never writes through a link that stands at a temporary name."""
    out = tmp_path / "out"
    _put(out / "parts" / "SX-T-A-9.tar.tmp", b"half a part of a build with other part names")
    _put(out / "parts" / "models-T-A.tar.tmp", b"half a part of this build")
    victim = tmp_path / "victim"
    victim.write_bytes(b"not to be written through a link")
    (out / "parts" / "models-T-B.tar.tmp").symlink_to(victim)
    doc = bundle.build("t1", BODIES, sets=("SX",), out=out, part_mb=0.06, root=tree)
    assert _files_under(out) == ["MANIFEST.json"] + sorted(f"parts/{n}" for n in doc["content"]["parts"])
    assert victim.read_bytes() == b"not to be written through a link"
    assert not any((out / "parts" / n).is_symlink() for n in doc["content"]["parts"])
    assert bundle.verify(doc, tree, out / "parts")["ok"]
    # the writer itself makes a new file: a name at which something stands is refused, a link there is not followed
    link = tmp_path / "x.tar.tmp"
    link.symlink_to(victim)
    with pytest.raises(bundle.BundleError, match="x.tar.tmp: not written"):
        bundle._write_part(link, tree, [])
    assert victim.read_bytes() == b"not to be written through a link" and link.is_symlink()


def test_build_refuses_a_file_that_is_no_manifest_at_its_place(tree, tmp_path):
    """An existing ``MANIFEST.json`` that cannot be read as a manifest is refused with a sentence (never another
    exception) and left as it is; no part is written beside it."""
    texts = ('{"content": [1]}', "[1]", "{not json", '"x"', '{"content": {"bundle_id": 3}}', '{"content": "abc"}',
             "[" * 100_000 + "]" * 100_000)
    for i, text in enumerate(texts):
        out = tmp_path / f"out{i}"
        _put(out / "MANIFEST.json", text.encode())
        with pytest.raises(bundle.BundleError, match="cannot be read as a manifest|a bundle id names one content"):
            bundle.build("t1", BODIES, sets=(), out=out, root=tree)
        assert _files_under(out) == ["MANIFEST.json"] and (out / "MANIFEST.json").read_text() == text


def test_a_second_build_leaves_the_manifest_and_another_content_is_refused(built, tree, tmp_path):
    """A bundle id names one content: the same build again changes no byte (the manifest keeps the first build's
    provenance); another content under the same directory, or a part that is not the bundle's, replaces nothing."""
    tree, out, doc = built
    before = {p: (out / p).read_bytes() for p in _files_under(out)}
    stamp = (out / "MANIFEST.json").stat().st_mtime_ns
    again = bundle.build("t1", BODIES, sets=("SX",), out=out, part_mb=0.06, root=tree)
    assert again == doc and (out / "MANIFEST.json").stat().st_mtime_ns == stamp
    assert {p: (out / p).read_bytes() for p in _files_under(out)} == before
    with pytest.raises(bundle.BundleError, match="a bundle id names one content"):
        bundle.build("t1", BODIES, sets=("SX",), out=out, part_mb=20, root=tree)
    with pytest.raises(bundle.BundleError, match="a bundle id names one content"):
        bundle.build("t1", ["T-A"], sets=("SX",), out=out, part_mb=0.06, root=tree)
    assert {p: (out / p).read_bytes() for p in _files_under(out)} == before
    # a build that died before its manifest was written, then one with other parts
    (out / "MANIFEST.json").unlink()
    with pytest.raises(bundle.BundleError, match=r"holds parts that are not this bundle's \(SX-T-A-2.tar"):
        bundle.build("t1", BODIES, sets=("SX",), out=out, part_mb=20, root=tree)
    assert sorted(_files_under(out)) == sorted(p for p in before if p != "MANIFEST.json")
    back = bundle.build("t1", BODIES, sets=("SX",), out=out, part_mb=0.06, root=tree)
    assert back["content"] == doc["content"]


@needs_git
def test_records_must_be_committed(tree, tmp_path):
    """Where the tree is a git checkout, every record a build reads must be tracked and as committed."""
    records = [p for p in _files_under(tree) if p.endswith(".json") and "/models/" not in p]
    _git(tmp_path, "init", "-q", str(tree))
    _git(tree, "add", "--", *records)
    _git(tree, "commit", "-q", "-m", "records")
    doc = bundle.build("t1", BODIES, sets=("SX",), out=tmp_path / "a", root=tree)
    assert doc["provenance"]["records_committed"] is True
    assert sorted(doc["provenance"]["records"]) == sorted(records)
    (tree / SETD / "set.json").write_text(json.dumps(json.loads((tree / SETD / "set.json").read_text()), indent=3))
    with pytest.raises(bundle.BundleError, match=f"{SETD}/set.json: differs from the commit"):
        bundle.build("t1", BODIES, sets=("SX",), out=tmp_path / "b", root=tree)
    _git(tree, "checkout", "--", f"{SETD}/set.json")
    _git(tree, "rm", "-q", "--cached", "--", bundle.INPUT_LISTS[0])
    _git(tree, "commit", "-q", "-m", "one list dropped")
    with pytest.raises(bundle.BundleError, match=f"{bundle.INPUT_LISTS[0]}: not tracked by git"):
        bundle.build("t1", BODIES, sets=("SX",), out=tmp_path / "c", root=tree)
    assert _files_under(tmp_path / "b") == [] and _files_under(tmp_path / "c") == []


# ------------------------------------------------------------------------------------------------
# The content hash
# ------------------------------------------------------------------------------------------------
def _reversed(obj):
    """``obj`` with every table's keys in the opposite order (lists keep theirs)."""
    if isinstance(obj, dict):
        return {k: _reversed(obj[k]) for k in reversed(list(obj))}
    return [_reversed(v) for v in obj] if isinstance(obj, list) else obj


def test_the_content_hash_ignores_layout_and_provenance(built, tmp_path):
    """Rewritten with CRLF line endings, another indentation, another key order or a changed provenance, the file
    gives the same content hash and still verifies; one changed character of the content does not."""
    tree, out, doc = built
    want = doc["content_sha256"]
    text = (out / "MANIFEST.json").read_text()
    variants = {
        "crlf": text.replace("\n", "\r\n").encode(),
        "indent": json.dumps(doc, indent=4).encode(),
        "compact": json.dumps(doc, separators=(",", ":")).encode(),
        "reordered": json.dumps(_reversed(doc), indent=2).encode(),
        "provenance": json.dumps(dict(doc, provenance=dict(doc["provenance"], made_at="2030-01-01T00:00:00Z",
                                                           commit="0" * 40, uploaded_by="someone")), indent=1).encode(),
        "no provenance": json.dumps({k: v for k, v in doc.items() if k != "provenance"}).encode(),
    }
    assert b"\r\n" in variants["crlf"] and list(json.loads(variants["reordered"])) == list(reversed(list(doc)))
    for name, raw in variants.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(raw)
        assert bundle.content_sha256(bundle.load(path)) == want, name
        got = bundle.verify(path, tree, out / "parts")
        assert got["ok"] and got["content_sha256"] == want, (name, got)
    changed = copy.deepcopy(doc)
    changed["content"]["bundle_id"] = "t9"
    assert bundle.content_sha256(changed) != want
    got = bundle.verify(changed, tree)
    assert not got["ok"] and got["first_problem"].startswith("MANIFEST: content_sha256")


# ------------------------------------------------------------------------------------------------
# verify
# ------------------------------------------------------------------------------------------------
def test_verify_counts_and_reports_each_difference_with_its_path(built, tmp_path):
    tree, out, doc = built
    n, nparts = len(doc["content"]["files"]), len(doc["content"]["parts"])
    good = bundle.verify(doc, tree, out / "parts")
    assert good == dict(ok=True, bundle_id="t1", content_sha256=doc["content_sha256"], files_ok=n, files=n,
                        parts_ok=nparts, parts=nparts, first_problem=None, problems=[])
    assert bundle.verify(out / "MANIFEST.json", tree)["parts_ok"] is None
    root = tmp_path / "copy"
    bundle.unpack(doc, out / "parts", root)
    changed, missing = f"{CASES['T-A']}/models/T-A.xml", f"{SETD}/transitions/start.npz"
    data = bytearray((root / changed).read_bytes())
    data[100] ^= 1
    (root / changed).write_bytes(data)  # the same size, one bit
    (root / missing).unlink()
    got = bundle.verify(doc, root)
    assert not got["ok"] and (got["files_ok"], got["files"]) == (n - 2, n)
    assert got["first_problem"] == got["problems"][0] and got["first_problem"].startswith(f"{changed}: ")
    assert got["problems"][1] == f"{missing}: missing" and len(got["problems"]) == 2
    # parts: a truncated one, a missing one, one changed in one bit
    pdir = tmp_path / "parts"
    shutil.copytree(out / "parts", pdir)
    cut, gone, flipped = "SX-T-A-2.tar", "models-T-B.tar", "SX-T-B-1.tar"
    (pdir / cut).write_bytes((pdir / cut).read_bytes()[:-10240])
    (pdir / gone).unlink()
    raw = bytearray((pdir / flipped).read_bytes())
    raw[700] ^= 1
    (pdir / flipped).write_bytes(raw)
    got = bundle.verify(doc, tree, pdir)
    assert not got["ok"] and (got["files_ok"], got["parts_ok"], got["parts"]) == (n, nparts - 3, nparts)
    assert [p.split(":")[0] for p in got["problems"]] == sorted([cut, gone, flipped])
    assert got["first_problem"].startswith(f"{cut}: ") and f"{gone}: missing" in got["problems"]


def test_verify_never_raises_on_a_bad_manifest(built, tmp_path):
    tree, out, doc = built
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"x")
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]")
    deep = tmp_path / "deep.json"  # nested too deep to be parsed
    deep.write_text("[" * 100_000 + "]" * 100_000)
    nested = tmp_path / "nested.json"  # the manifest itself with a value in its content nested too deep to be hashed
    nested.write_text(json.dumps(dict(doc, content=dict(doc["content"], extra=0)))
                      .replace('"extra": 0', '"extra": ' + "[" * 3000 + "]" * 3000))

    def edited(change):
        d = copy.deepcopy(doc)
        change(d)
        if isinstance(d.get("content"), dict):
            d["content_sha256"] = s2io.sha256_json(d["content"])
        return d

    cases = [
        (tmp_path / "nothing.json", "MANIFEST: "), (junk, "MANIFEST: "), (listy, "MANIFEST: "), (3.5, "MANIFEST: "),
        (deep, "MANIFEST: "), (nested, "MANIFEST: "),
        ({}, "MANIFEST: schema"), (dict(doc, schema="s2-bundle-1"), "MANIFEST: schema 's2-bundle-1'"),
        (dict(doc, content=None), "MANIFEST: no content block"),
        (dict(doc, content_sha256="0" * 64), "MANIFEST: content_sha256"),
        (edited(lambda d: d["content"].pop("files")), "MANIFEST: content holds no files"),
        (edited(lambda d: d["content"]["files"].update({"../outside.npz": dict(sha256=_sha(b"x"), bytes=1,
                                                                                part="models-T-A.tar")})),
         "'../outside.npz': the path holds '..'"),
        (edited(lambda d: d["content"]["files"].update({str(outside): dict(sha256=_sha(b"x"), bytes=1,
                                                                            part="models-T-A.tar")})),
         f"{str(outside)!r}: the path is absolute"),
        (edited(lambda d: d["content"]["files"][f"{SETD}/refs/nominal.npz"].update(part="nope.tar")),
         f"{SETD}/refs/nominal.npz: its part 'nope.tar'"),
        (edited(lambda d: d["content"]["files"][f"{SETD}/refs/nominal.npz"].update(bytes="9000")),
         f"{SETD}/refs/nominal.npz: its entry"),
        (edited(lambda d: d["content"]["parts"].update({"../x.tar": dict(sha256="0" * 64, bytes=1, files=0)})),
         "'../x.tar': not a part's file name"),
        (edited(lambda d: d["content"]["parts"]["models-T-A.tar"].update(files=3)), "models-T-A.tar: lists 3 files"),
    ]
    for manifest, text in cases:
        got = bundle.verify(manifest, tree, out / "parts")
        assert got["ok"] is False and text in got["first_problem"], (text, got["first_problem"])
        assert set(got) == {"ok", "bundle_id", "content_sha256", "files_ok", "files", "parts_ok", "parts",
                            "first_problem", "problems"}
        with pytest.raises(bundle.BundleError):  # and unpack refuses the same manifests, before writing
            bundle.unpack(manifest, out / "parts", tmp_path / "never")
    assert _files_under(tmp_path / "never") == []


def _gone(path: Path):
    """What a file on a mount that went away gives: an error that is not "no such file"."""
    raise OSError(errno.ENOTCONN, "Transport endpoint is not connected", str(path))


def test_a_part_that_cannot_be_read_is_reported_and_refused(built, tmp_path, monkeypatch):
    """A part whose stat or read fails with another error than "no such file" (a Drive mount that went away):
    ``verify`` reports it with the part's name and counts on, ``unpack`` refuses with a sentence and writes nothing;
    neither ends in another exception."""
    tree, out, doc = built
    part = out / "parts" / PART
    is_file, hash_ = Path.is_file, bundle._hash
    patches = ((Path, "is_file", lambda self, **kw: _gone(self) if self == part else is_file(self, **kw)),
               (bundle, "_hash", lambda path: _gone(path) if Path(path) == part else hash_(path)))
    for target, name, fake in patches:
        with monkeypatch.context() as m:
            m.setattr(target, name, fake)
            got = bundle.verify(doc, tree, out / "parts")
            with pytest.raises(bundle.BundleError) as exc:
                bundle.unpack(doc, out / "parts", tmp_path / "never")
        assert got["problems"] == [f"{PART}: cannot be read (Transport endpoint is not connected)"], name
        assert not got["ok"] and got["parts_ok"] == got["parts"] - 1 and got["files_ok"] == got["files"]
        assert str(exc.value).startswith(f"{PART}: cannot be read ") and "nothing is unpacked" in str(exc.value)
    assert _files_under(tmp_path / "never") == []
    # a file under the root likewise
    lost = tree / FIRST
    with monkeypatch.context() as m:
        m.setattr(Path, "is_file", lambda self, **kw: _gone(self) if self == lost else is_file(self, **kw))
        got = bundle.verify(doc, tree)
    assert got["problems"] == [f"{FIRST}: cannot be read (Transport endpoint is not connected)"]
    assert got["files_ok"] == got["files"] - 1


def test_unpack_refuses_a_place_it_cannot_look_at(built, tmp_path, monkeypatch):
    """A file's place under the root that cannot be looked at (a directory on the way that may not be searched) is a
    refusal with the path, before anything is written."""
    tree, out, doc = built
    root, is_dir = tmp_path / "unpacked", Path.is_dir
    place = Path(os.path.realpath(root)) / FIRST

    def fake(self, **kw):
        if self == place:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return is_dir(self, **kw)

    monkeypatch.setattr(Path, "is_dir", fake)
    with pytest.raises(bundle.BundleError, match=f"{FIRST}: its place .* cannot be looked at .Permission denied"):
        bundle.unpack(doc, out / "parts", root)
    assert _files_under(root) == []


def test_a_part_that_changes_while_it_is_unpacked_is_refused(built, tmp_path, monkeypatch):
    """Between the check of the parts and the writing a part can change: what is written is compared with the
    manifest once more before it is moved into place, and the wrong bytes never stand at a file's place."""
    tree, out, doc = built
    pdir, root = tmp_path / "parts", tmp_path / "unpacked"
    shutil.copytree(out / "parts", pdir)
    real = bundle.scan_part

    def scan(path, expect):
        real(path, expect)
        if Path(path).name == PART:  # checked; now it becomes a tar with the same names and sizes and other bytes
            members = _members(Path(path))
            _write_tar(Path(path), [(members[0][0], bytes(len(members[0][1])))] + members[1:])

    monkeypatch.setattr(bundle, "scan_part", scan)
    assert sorted(doc["content"]["parts"])[0] == PART  # the first part written, and FIRST its first member
    with pytest.raises(bundle.BundleError, match=f"{PART}: changed while it was unpacked"):
        bundle.unpack(doc, pdir, root)
    assert _files_under(root) == []  # no wrong file, no temporary file


# ------------------------------------------------------------------------------------------------
# unpack
# ------------------------------------------------------------------------------------------------
def test_unpack_gives_the_files_and_twice_is_a_no_op(built, tmp_path):
    tree, out, doc = built
    expected, root = _expected_files(tree), tmp_path / "unpacked"
    first = bundle.unpack(doc, out / "parts", root)
    assert first["ok"] and first["written"] == sorted(expected) and first["left"] == first["replaced"] == []
    assert first["parts"] == sorted(doc["content"]["parts"]) and first["files"] == len(expected)
    assert first["content_sha256"] == doc["content_sha256"] and first["root"] == os.path.realpath(root)
    assert _files_under(root) == sorted(expected)  # nothing else: no temporary file, no look twin
    assert all((root / rel).read_bytes() == data for rel, data in expected.items())
    assert bundle.verify(doc, root)["ok"]
    stamps = {rel: ((root / rel).stat().st_mtime_ns, (root / rel).stat().st_ino) for rel in expected}
    second = bundle.unpack(out / "MANIFEST.json", out / "parts", root)
    assert second["written"] == second["replaced"] == [] and second["left"] == sorted(expected)
    assert stamps == {rel: ((root / rel).stat().st_mtime_ns, (root / rel).stat().st_ino) for rel in expected}
    # a wrong file in the way is replaced by the bundle's (its temporary name too, had a killed unpack left one); a
    # link at a file's place is replaced, not followed
    wrong, linked, flipped = f"{SETD}/refs/nominal.npz", f"{SETD}/refs/v0.4.npz", f"{SETD}/transitions/start.npz"
    (root / wrong).write_bytes(b"wrong")
    data = bytearray(expected[flipped])
    data[-1] ^= 1
    (root / flipped).write_bytes(data)  # the right size, one bit wrong: its sha256 decides, not its size
    (root / (wrong + ".tmp")).write_bytes(b"what a killed unpack left")
    victim = tmp_path / "victim"
    victim.write_bytes(b"not to be written through a link")
    (root / (linked + ".tmp")).symlink_to(victim)
    target = tmp_path / "elsewhere.npz"
    target.write_bytes(expected[linked])
    (root / linked).unlink()
    (root / linked).symlink_to(target)
    third = bundle.unpack(doc, out / "parts", root)
    assert third["replaced"] == sorted([wrong, linked, flipped]) and third["written"] == []
    assert len(third["left"]) == len(expected) - 3
    assert not (root / linked).is_symlink() and target.read_bytes() == expected[linked]
    assert victim.read_bytes() == b"not to be written through a link"
    assert bundle.verify(doc, root)["ok"] and _files_under(root) == sorted(expected)


def test_unpack_only_the_named_parts(built, tmp_path):
    """Rule N4's second fallback: the model parts alone (the other parts need not even be there)."""
    tree, out, doc = built
    pdir = tmp_path / "parts"
    pdir.mkdir()
    models = [f"models-{b}.tar" for b in BODIES]
    for name in models:
        shutil.copyfile(out / "parts" / name, pdir / name)
    with pytest.raises(bundle.BundleError, match="SX-T-A-1.tar: missing"):
        bundle.unpack(doc, pdir, tmp_path / "all")
    assert _files_under(tmp_path / "all") == []
    got = bundle.unpack(doc, pdir, tmp_path / "some", parts=models)
    assert got["parts"] == models and _files_under(tmp_path / "some") == sorted(
        f"{case}/models/{body}.{s}" for body, case in CASES.items() for s in ("xml", "json"))
    with pytest.raises(bundle.BundleError, match="not parts of bundle t1: nope.tar"):
        bundle.unpack(doc, pdir, tmp_path / "some", parts=["nope.tar"])
    # an empty selection is no selection (it would report "OK" with nothing unpacked); nor is a name alone a list
    for nothing, text in (([], "no part is named"), ((), "no part is named"),
                          ("models-T-A.tar", "a list of part names"), ([3, "models-T-A.tar"], "a list of part names")):
        with pytest.raises(bundle.BundleError, match=text):
            bundle.unpack(doc, pdir, tmp_path / "none", parts=nothing)
    assert _files_under(tmp_path / "none") == []


def _members(path: Path) -> list:
    with tarfile.open(path, "r:") as tar:
        return [(info, tar.extractfile(info).read()) for info in tar]


def _info(name: str, data: bytes = b"", kind: bytes = tarfile.REGTYPE, link: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.type, info.linkname, info.mode = (len(data) if kind == tarfile.REGTYPE else 0), kind, link, 0o644
    return info


def _write_tar(path: Path, members: list) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if info.type == tarfile.REGTYPE else None)


def _refit(doc: dict, pdir: Path) -> dict:
    """The manifest made to agree with the parts as they now are (their sizes and sha256, the content hash): what
    someone who can change a part and the manifest would write.  ``content.files`` is left alone."""
    doc = copy.deepcopy(doc)
    for name, e in doc["content"]["parts"].items():
        if (pdir / name).is_file():
            e.update(sha256=s2io.sha256_file(pdir / name), bytes=(pdir / name).stat().st_size)
    doc["content_sha256"] = s2io.sha256_json(doc["content"])
    return doc


PART, OTHER = "SX-T-A-1.tar", "SX-T-A-2.tar"
FIRST = f"{SETD}/refs/nominal.npz"


def _tamper(case: str, pdir: Path, doc: dict, tmp_path: Path) -> tuple[dict, str]:
    """Spoil one part as ``case`` says; (the manifest to unpack with, what the refusal must name)."""
    path = pdir / PART
    members = _members(path)
    if case == "a truncated part":
        path.write_bytes(path.read_bytes()[:-10240])
        return doc, f"{PART}: "
    if case == "a part changed in one bit":
        raw = bytearray(path.read_bytes())
        raw[600] ^= 1
        path.write_bytes(raw)
        return doc, f"{PART}: "
    if case == "a missing part":
        path.unlink()
        return doc, f"{PART}: missing"
    if case == "an extra member":
        extra = f"{SETD}/refs/extra.npz"
        _write_tar(path, members + [(_info(extra, b"more"), b"more")])
        return _refit(doc, pdir), f"member {extra!r} is not one of this part's files"
    if case == "a member with ..":
        _write_tar(path, members + [(_info("../evil.npz", b"evil"), b"evil")])
        return _refit(doc, pdir), "member '../evil.npz' holds '..': it would leave the root"
    if case == "a member with .. inside":
        name = f"{SETD}/refs/../../../../../../evil.npz"
        _write_tar(path, members + [(_info(name, b"evil"), b"evil")])
        return _refit(doc, pdir), f"member {name!r} holds '..'"
    if case == "an absolute member":
        name = str(tmp_path / "abs.npz")
        _write_tar(path, members + [(_info(name, b"evil"), b"evil")])
        return _refit(doc, pdir), f"member {name!r} is absolute: it would leave the root"
    if case == "a symbolic link":
        _write_tar(path, [(_info(FIRST, kind=tarfile.SYMTYPE, link="/etc/hostname"), b"")] + members[1:])
        return _refit(doc, pdir), f"member {FIRST!r} is a symbolic link"
    if case == "a hard link":
        _write_tar(path, members + [(_info(f"{SETD}/refs/hard.npz", kind=tarfile.LNKTYPE, link=FIRST), b"")])
        return _refit(doc, pdir), f"member '{SETD}/refs/hard.npz' is a hard link"
    if case == "a directory":
        _write_tar(path, [(_info(f"{SETD}/refs", kind=tarfile.DIRTYPE), b"")] + members)
        return _refit(doc, pdir), f"member '{SETD}/refs' is a directory"
    if case == "a member with other bytes":
        _write_tar(path, [(members[0][0], bytes(len(members[0][1])))] + members[1:])
        return _refit(doc, pdir), f"member {FIRST!r} has 9000 bytes and sha256"
    if case == "a member of another part":
        moved = _members(pdir / OTHER)[0]
        _write_tar(path, members + [moved])
        return _refit(doc, pdir), f"member {moved[0].name!r} is not one of this part's files"
    if case == "a member twice":
        _write_tar(path, members + [members[0]])
        return _refit(doc, pdir), f"member {FIRST!r} appears twice"
    if case == "a member short":
        _write_tar(path, members[:-1])
        return _refit(doc, pdir), f"{PART}: 1 of its {len(members)} files are not in it (first: {members[-1][0].name})"
    if case == "no tar at all":
        path.write_bytes(b"PK\x03\x04" + bytes(20000))
        return _refit(doc, pdir), f"{PART}: not a readable plain tar"
    if case == "a compressed part":
        with tarfile.open(path, "w:gz") as tar:
            for info, data in members:
                tar.addfile(info, io.BytesIO(data))
        return _refit(doc, pdir), f"{PART}: not a readable plain tar"
    raise KeyError(case)


TAMPERED = ["a truncated part", "a part changed in one bit", "a missing part", "an extra member", "a member with ..",
            "a member with .. inside", "an absolute member", "a symbolic link", "a hard link", "a directory",
            "a member with other bytes", "a member of another part", "a member twice", "a member short",
            "no tar at all", "a compressed part"]


@pytest.mark.parametrize("case", TAMPERED)
def test_unpack_refuses_before_it_writes_anything(case, built, tmp_path):
    """A part whose sha256 differs is refused; so is, in a part that carries the manifest's sha256 (someone changed
    both), every member that is not one of the part's files, that would leave the root or that is no regular file.
    The refusal names the part or the member, and not one file is written, of this part or of any other."""
    tree, out, doc = built
    pdir, root = tmp_path / "parts", tmp_path / "unpacked"
    shutil.copytree(out / "parts", pdir)
    manifest, text = _tamper(case, pdir, doc, tmp_path)
    with pytest.raises(bundle.BundleError) as exc:
        bundle.unpack(manifest, pdir, root)
    assert text in str(exc.value), str(exc.value)
    assert _files_under(root) == []
    assert not (tmp_path / "abs.npz").exists() and not (tmp_path / "evil.npz").exists()
    if manifest is not doc:  # the part alone, against the files the manifest gives it
        want = {rel: e for rel, e in doc["content"]["files"].items() if e["part"] == PART}
        with pytest.raises(bundle.BundleError):
            bundle.scan_part(pdir / PART, want)
        bundle.scan_part(out / "parts" / PART, want)  # the part as built passes


def test_unpack_refuses_a_place_that_resolves_outside_the_root(built, tmp_path):
    """A symbolic link on a file's way (here ``<root>/stage2`` pointing elsewhere) is a path that leaves the root."""
    tree, out, doc = built
    root, outside = tmp_path / "unpacked", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "stage2").symlink_to(outside, target_is_directory=True)
    with pytest.raises(bundle.BundleError, match=f"{SETD}/refs/nominal.npz: its directory resolves to .*outside"):
        bundle.unpack(doc, out / "parts", root)
    assert _files_under(outside) == [] and _files_under(root) == []
    # a directory where a file belongs
    (root / "stage2").unlink()
    (root / FIRST).mkdir(parents=True)
    with pytest.raises(bundle.BundleError, match=f"{FIRST}: a directory stands at"):
        bundle.unpack(doc, out / "parts", root)
    assert _files_under(root) == []
    # a file where a directory belongs
    shutil.rmtree(root / "stage2")
    (root / "stage2").write_text("in the way")
    with pytest.raises(bundle.BundleError, match=f"{FIRST}: .*stage2 is in its way and is no directory"):
        bundle.unpack(doc, out / "parts", root)
    assert _files_under(root) == ["stage2"]


def test_path_problem():
    for good in ("a", "a/b.npz", "round2/refine/feet_75/models/75-M-H-10.xml", "a/..b/c", ".hidden.npz"):
        assert bundle.path_problem(good) is None, good
    for bad in ("", "/a", "../a", "a/../b", "a/..", "a//b", "./a", "a/./b", "a/", "a\\b", "a\x00b", None, 3, b"a"):
        assert bundle.path_problem(bad), bad


# ------------------------------------------------------------------------------------------------
# The CLI and what the module imports
# ------------------------------------------------------------------------------------------------
def test_cli_verify_and_unpack(built, tmp_path):
    tree, out, doc = built
    manifest, parts, root = str(out / "MANIFEST.json"), str(out / "parts"), tmp_path / "unpacked"
    short, n = doc["content_sha256"][:12], len(doc["content"]["files"])
    cli = ["-m", "designlab.stage2.bundle"]
    got = _run(cli + ["unpack", "--manifest", manifest, "--parts", parts, "--root", str(root)])
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout.strip() == (f"S2 BUNDLE t1: manifest {short}, {n} files of 10 parts under "
                                  f"{os.path.realpath(root)}: {n} written, 0 replaced, 0 already there: OK")
    got = _run(cli + ["verify", "--manifest", manifest, "--root", str(root), "--parts", parts])
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout.strip() == f"S2 BUNDLE t1: manifest {short}, {n} of {n} files match their sha256, 10 of 10 parts: OK"
    (root / FIRST).write_bytes(b"other")
    (root / CASES["T-A"] / "models" / "T-A.xml").unlink()
    got = _run(cli + ["verify", "--manifest", manifest, "--root", str(root)])
    assert got.returncode == 1
    lines = got.stdout.strip().splitlines()
    assert lines[0].startswith(f"S2 BUNDLE t1: manifest {short}, {n - 2} of {n} files match their sha256: FAILED "
                               f"({CASES['T-A']}/models/T-A.xml: missing)")
    assert lines[1].startswith(f"  also: {FIRST}: 5 bytes")
    # a part alone, and a spoiled part
    got = _run(cli + ["unpack", "--manifest", manifest, "--parts", parts, "--root", str(tmp_path / "one"), "--only",
                      "models-T-B.tar"])
    assert got.returncode == 0 and ", 2 files of 1 part under " in got.stdout
    spoiled = tmp_path / "parts"
    shutil.copytree(out / "parts", spoiled)
    (spoiled / PART).write_bytes((spoiled / PART).read_bytes()[:-512])
    got = _run(cli + ["unpack", "--manifest", manifest, "--parts", str(spoiled), "--root", str(tmp_path / "two")])
    assert got.returncode == 1 and got.stdout.startswith(f"S2 BUNDLE: NOT UNPACKED: {PART}: ")
    assert _files_under(tmp_path / "two") == []
    # an empty selection is refused, not reported as "0 files of 0 parts: OK"
    got = _run(cli + ["unpack", "--manifest", manifest, "--parts", parts, "--root", str(tmp_path / "two"), "--only",
                      ""])
    assert got.returncode == 1 and got.stdout.startswith("S2 BUNDLE: NOT UNPACKED: no part is named")
    assert _files_under(tmp_path / "two") == []


def test_cli_json(built, tmp_path):
    """``--json``: the result as one line of JSON and nothing else (what a caller in another process reads: the
    whole content hash, the counts, every problem), with the same exit codes."""
    tree, out, doc = built
    manifest, parts, root = str(out / "MANIFEST.json"), str(out / "parts"), tmp_path / "unpacked"
    cli = ["-m", "designlab.stage2.bundle"]

    def one_line(got):
        assert got.stdout.endswith("\n") and got.stdout.count("\n") == 1, got.stdout + got.stderr
        return json.loads(got.stdout)

    got = _run(cli + ["unpack", "--manifest", manifest, "--parts", parts, "--root", str(root), "--json"])
    first = one_line(got)
    assert got.returncode == 0 and first == dict(
        ok=True, bundle_id="t1", content_sha256=doc["content_sha256"], root=os.path.realpath(root),
        parts=sorted(doc["content"]["parts"]), files=len(doc["content"]["files"]),
        written=sorted(doc["content"]["files"]), replaced=[], left=[])
    got = _run(cli + ["verify", "--manifest", manifest, "--root", str(root), "--parts", parts, "--json"])
    assert got.returncode == 0 and one_line(got) == bundle.verify(manifest, root, parts)
    assert one_line(got)["ok"] is True and one_line(got)["content_sha256"] == doc["content_sha256"]
    (root / FIRST).write_bytes(b"other")
    got = _run(cli + ["verify", "--manifest", manifest, "--root", str(root), "--json"])
    bad = one_line(got)
    assert got.returncode == 1 and bad == bundle.verify(manifest, root)
    assert bad["ok"] is False and bad["first_problem"].startswith(f"{FIRST}: 5 bytes") and bad["parts_ok"] is None
    assert (bad["files_ok"], bad["files"]) == (len(doc["content"]["files"]) - 1, len(doc["content"]["files"]))
    got = _run(cli + ["verify", "--manifest", str(tmp_path / "nothing.json"), "--root", str(root), "--json"])
    assert got.returncode == 1 and one_line(got)["first_problem"].startswith("MANIFEST: ")
    spoiled = tmp_path / "parts"
    shutil.copytree(out / "parts", spoiled)
    (spoiled / PART).write_bytes((spoiled / PART).read_bytes()[:-512])
    got = _run(cli + ["unpack", "--manifest", manifest, "--parts", str(spoiled), "--root", str(tmp_path / "two"),
                      "--json"])
    refused = one_line(got)
    assert got.returncode == 1 and sorted(refused) == ["first_problem", "ok"] and refused["ok"] is False
    assert refused["first_problem"].startswith(f"{PART}: ") and _files_under(tmp_path / "two") == []


def test_verify_and_unpack_import_neither_mujoco_nor_setio(built, tmp_path):
    """What a Colab session calls needs the standard library, numpy and ``s2io`` only; a models-only build imports
    no plant either (so it cannot rebuild a model)."""
    tree, out, doc = built
    code = ("import json, sys\n"
            "from designlab import stage1\n"
            "from designlab.stage2 import bundle\n"
            "job = json.loads(sys.argv[1])\n"
            "stage1.BODIES.update(job['bodies']); stage1.SHA256_PREFIX.update(job['prefix'])\n"
            "got = bundle.unpack(job['manifest'], job['parts'], job['root'])\n"
            "assert bundle.verify(job['manifest'], job['root'], job['parts'])['ok']\n"
            "bundle.build('m', sorted(job['bodies']), sets=(), out=job['out'], root=job['tree'])\n"
            "bad = [m for m in ('mujoco', 'torch', 'designlab.stage2.setio', 'designlab.stage1.plant', "
            "'designlab.stage1.control') if m in sys.modules]\n"
            "print(len(got['written']), bad)\n")
    reg = dict(bodies=CASES, prefix={b: stage1.SHA256_PREFIX[b] for b in CASES})
    job = dict(reg, manifest=str(out / "MANIFEST.json"), parts=str(out / "parts"), root=str(tmp_path / "unpacked"),
               out=str(tmp_path / "m"), tree=str(tree))
    got = _run(["-c", code, json.dumps(job)])
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == f"{len(doc['content']['files'])} []"


# ------------------------------------------------------------------------------------------------
# The study's own files
# ------------------------------------------------------------------------------------------------
def test_the_lists_and_the_sets_directory_are_the_studys():
    """``INPUT_LISTS`` are the lists ``tools/restore_inputs.py`` restores model files from, and each is committed
    here; ``SETS_DIR`` is ``setio.ROOT``."""
    spec = importlib.util.spec_from_file_location("restore_inputs_for_bundle", ROOT / "tools" / "restore_inputs.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    theirs = [tool.MANIFEST] + [p.with_suffix(".json") for p in tool.HELPERS.values()]
    assert sorted(bundle.INPUT_LISTS) == sorted(p.relative_to(tool.STUDY).as_posix() for p in theirs)
    for rel in bundle.INPUT_LISTS:
        assert isinstance(json.loads((ROOT / rel).read_text())["files"], dict), rel
    for body in stage1.BODIES:  # every registered body's two files are in one of the lists, with a sha256 each
        lists = [json.loads((ROOT / rel).read_text())["files"] for rel in bundle.INPUT_LISTS]
        for path in bundle.model_paths(body):
            assert sum(path in files for files in lists) == 1, path
    from designlab.stage2 import setio

    assert (s2io.STUDY / bundle.SETS_DIR) == setio.ROOT


def _have_models(body: str) -> bool:
    return all((ROOT / p).is_file() for p in bundle.model_paths(body))


def test_model_paths_are_plant_model_files():
    """The two paths a bundle packs per body are the ones the plant reads (asked only where both files are here:
    ``plant.model_files`` rebuilds missing ones)."""
    bodies = [b for b in stage1.BODIES if _have_models(b)]
    if not bodies:
        pytest.skip("no body's round-two model files are here (tools/restore_inputs.py)")
    from designlab.stage1 import plant

    for body in bodies:
        xml, ledger = plant.model_files(body)
        assert (xml.relative_to(ROOT).as_posix(), ledger.relative_to(ROOT).as_posix()) == bundle.model_paths(body)


@pytest.mark.skipif(not all(_have_models(b) for b in PICKS), reason="the picks' model files are not here")
def test_a_stand_in_bundle_of_the_picks_models(tmp_path):
    """The brief's acceptance in small: the two picks' real model files, no set: built from this checkout's committed
    records (through the CLI), verified against the checkout and against the parts, unpacked elsewhere."""
    out, root = tmp_path / "standin", tmp_path / "unpacked"
    rebuilt = _files_under(ROOT / "stage1" / "models")
    cli = ["-m", "designlab.stage2.bundle"]
    got = _run(cli + ["build", "--bundle-id", "standin", "--bodies", ",".join(PICKS), "--sets", "", "--out", str(out)])
    if got.returncode != 0 and "committed records only" in got.stdout:
        pytest.skip("a record of the picks is not committed in this checkout: " + got.stdout.strip()[:300])
    assert got.returncode == 0, got.stdout + got.stderr
    doc = json.loads((out / "MANIFEST.json").read_text())
    c = doc["content"]
    assert f"content sha256 {doc['content_sha256']} (manifest {doc['content_sha256'][:12]})" in got.stdout
    assert c["bodies"] == sorted(PICKS) and c["sets"] == {} and sorted(c["parts"]) == sorted(
        f"models-{b}.tar" for b in PICKS)
    listed = json.loads((ROOT / "stage1" / "runs" / "2026-11-08" / "stage1_inputs.json").read_text())["files"]
    assert len(c["files"]) == 4
    for body in PICKS:
        xml, ledger = bundle.model_paths(body)
        assert c["models"][body]["xml"] == dict(path=xml, sha256=listed[xml])
        assert c["models"][body]["record"] == dict(path=ledger, sha256=listed[ledger])
        assert c["files"][xml]["sha256"].startswith(stage1.SHA256_PREFIX[body])
        assert doc["provenance"]["hash_sources"][ledger] == ["stage1/runs/2026-11-08/stage1_inputs.json: files"]
    assert all(p["bytes"] <= 20 * bundle.MB for p in c["parts"].values())
    assert doc["provenance"]["root"] == "." and doc["provenance"]["records_committed"] in (True, None)
    got = _run(cli + ["verify", "--manifest", str(out / "MANIFEST.json"), "--root", ".", "--parts", str(out / "parts")])
    assert got.returncode == 0 and got.stdout.strip().endswith("4 of 4 files match their sha256, 2 of 2 parts: OK")
    got = _run(cli + ["unpack", "--manifest", str(out / "MANIFEST.json"), "--parts", str(out / "parts"), "--root",
                      str(root)])
    assert got.returncode == 0 and got.stdout.strip().endswith("4 written, 0 replaced, 0 already there: OK")
    assert _files_under(root) == sorted(c["files"])
    assert all((root / rel).read_bytes() == (ROOT / rel).read_bytes() for rel in c["files"])
    assert _files_under(ROOT / "stage1" / "models") == rebuilt  # nothing was rebuilt


def _frozen_set_here():
    """(set, body, manifest) of the first frozen reference set whose arrays and model files are all here."""
    from designlab.stage2 import setio

    for path in sorted(setio.ROOT.glob("*/*/set.json")):
        fz = setio.frozen(json.loads(path.read_text()))
        body = path.parent.name
        if fz is None or body not in stage1.BODIES or not _have_models(body):
            continue
        files = [path.parent / ("refs" if row["kind"] in setio.WALK_KINDS else "transitions") / f"{name}.npz"
                 for name, row in fz["entries"].items()]
        if all(f.is_file() for f in files):
            return path.parent.parent.name, body, fz
    return None


def test_a_real_frozen_set(tmp_path):
    """A set as round A froze it: the bundle holds exactly the freeze's Stage 1-format files with its hashes (files
    are read and hashed only: no episode runs on the set)."""
    found = _frozen_set_here()
    if found is None:
        pytest.skip("no frozen reference set has its arrays here (tools/restore_inputs.py restore --group sets)")
    set_name, body, fz = found
    out = tmp_path / "real"
    try:
        doc = bundle.build("real", [body], sets=(set_name,), out=out)
    except bundle.BundleError as exc:
        if "committed records only" in str(exc):
            pytest.skip(f"a record of {set_name} {body} is not committed in this checkout")
        raise
    c = doc["content"]
    assert c["sets"] == {set_name: {body: dict(freeze_index=fz["index"], manifest_sha256=fz["manifest_sha256"],
                                               files=len(fz["entries"]))}}
    set_files = {f: e for f, e in c["files"].items() if f.startswith(f"{bundle.SETS_DIR}/")}
    assert sorted(e["sha256"] for e in set_files.values()) == sorted(r["sha256"] for r in fz["entries"].values())
    assert len(set_files) == len(fz["entries"]) and not [f for f in set_files if "/look/" in f]
    assert all(p["bytes"] <= 20 * bundle.MB for p in c["parts"].values())
    assert bundle.verify(doc, ROOT, out / "parts")["ok"]
    got = bundle.unpack(doc, out / "parts", tmp_path / "unpacked")
    assert got["written"] == sorted(c["files"]) and bundle.verify(doc, tmp_path / "unpacked")["ok"]
