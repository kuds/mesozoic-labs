"""Tests for tools/restore_inputs.py: the picks' list (``stage1/runs/2026-11-08/stage1_inputs.json``) against the
committed records, every group's list against its records, and the check, copy, not-restored and restore logic on
temporary directories, each regeneration replaced by a stub that writes chosen bytes where the child process would.
Fast: nothing is regenerated (``restore`` itself was proven in a fresh worktree; tools/README.md).

Run from hardware/trex_form_study:  python -m pytest tests/test_restore_inputs.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

STUDY = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("restore_inputs", STUDY / "tools" / "restore_inputs.py")
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

from designlab import stage1  # noqa: E402

SHA = re.compile(r"[0-9a-f]{64}")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------
# The picks' list
# ------------------------------------------------------------------------------------------------
def test_the_picks_list_has_the_helpers_format_and_every_pick_file():
    doc = json.loads(ri.MANIFEST.read_text())
    k = json.loads(ri.HELPERS["k"].with_suffix(".json").read_text())
    assert set(doc) == {"rule", "bodies", "cases", "files"} and {"rule", "bodies", "case", "files"} == set(k)
    assert doc["rule"] == ri.RULE
    assert tuple(doc["bodies"]) == ri.PICKS == stage1.PROBE_BODIES
    assert doc["cases"] == {b: Path(stage1.BODIES[b]).name for b in ri.PICKS}
    assert doc["cases"] == {"60-N-T-10": "narrow_carpet_60", "60-N-H-10": "narrow_carpet_60", "75-M-H-10": "feet_75"}
    files = doc["files"]
    assert list(files) == sorted(files) and all(SHA.fullmatch(s) for s in files.values())
    models = {p for b in ri.PICKS for p in ri.model_paths(b)}
    assert models == {"round2/refine/narrow_carpet_60/models/60-N-T-10.xml",
                      "round2/refine/narrow_carpet_60/models/60-N-T-10.json",
                      "round2/refine/narrow_carpet_60/models/60-N-H-10.xml",
                      "round2/refine/narrow_carpet_60/models/60-N-H-10.json",
                      "round2/refine/feet_75/models/75-M-H-10.xml", "round2/refine/feet_75/models/75-M-H-10.json"}
    arrays = {p for b in ri.PICKS for kind in ("refs", "transitions") for p, _ in ri.array_paths(b, kind).values()}
    assert set(files) == models | arrays and len(arrays) == 3 * (10 + 18)
    assert all(p.startswith("stage1/runs/2026-11-08/") and p.endswith(".npz") for p in arrays)
    assert set(ri.listed("picks-models")) == models and set(ri.listed("picks-arrays")) == arrays


def test_the_picks_list_matches_the_records():
    """The XML's sha256 is the candidate record's (with the plan's prefix), the ledger JSON's the one the committed
    eval records' keys hold, each array's the one refs.json / transitions.json records."""
    files = json.loads(ri.MANIFEST.read_text())["files"]
    for body in ri.PICKS:
        xml, ledger = ri.model_paths(body)
        rec = json.loads((STUDY / stage1.BODIES[body] / "candidates" / f"{body}.json").read_text())
        assert files[xml] == rec["xml_sha256"] and files[xml].startswith(stage1.SHA256_PREFIX[body])
        keyed = ri.keyed_inputs(body)
        assert list(keyed["mjcf"]) == [files[xml]] and list(keyed["ledger"]) == [files[ledger]]
        for kind in ("refs", "transitions"):
            for path, sha in ri.array_paths(body, kind).values():
                assert files[path] == sha
        recorded, problems = ri.recorded(body)
        assert not problems and recorded == {p: files[p] for p in recorded}


def test_every_listed_input_is_git_ignored_and_untracked():
    paths = [p for g in ri.GROUPS for p in ri.listed(g)]
    # the sets group's list is made from the frozen sets' manifests: two files per frozen walk, one per frozen move
    frozen = sum(2 if row.get("kind") in ("walk", "ref") else 1 for _, _, manifest in ri.frozen_sets()
                 for row in manifest.values())
    assert len(ri.listed("sets")) == frozen
    assert len(paths) == len(set(paths)) == 6 + 84 + 60 + 60 + 30 + 73 + frozen
    ignored = subprocess.run(["git", "-C", str(STUDY), "check-ignore", "--no-index", "--stdin"], input="\n".join(paths),
                             capture_output=True, text=True).stdout.split()
    assert sorted(ignored) == sorted(paths)
    tracked = subprocess.run(["git", "-C", str(STUDY), "ls-files", "--", *paths], capture_output=True, text=True,
                             check=True).stdout.split()
    assert tracked == []


@pytest.mark.parametrize("group", ri.GROUPS)
def test_every_list_agrees_with_the_committed_records(group):
    assert ri.agreement(group) == []


def test_the_restore_order():
    """Models before arrays; the look ladder (whose price reads the picks' models) after them; the frozen reference
    sets (built on the picks' models, checked against the picks' and the ladder's arrays) last."""
    assert ri.GROUPS == ("picks-models", "picks-arrays", "k", "x", "j", "look", "sets")
    assert set(ri.GROUPS) == set(ri.TITLES) == set(ri.RESTORE)
    assert ri.NEEDS == {"look": ("picks-models",), "sets": ("picks-models", "picks-arrays")}
    for g, needs in ri.NEEDS.items():
        assert all(ri.GROUPS.index(n) < ri.GROUPS.index(g) for n in needs)
    assert ri.helper_bodies("k") == ("60-N-T-10-K", "60-N-H-10-K") and ri.helper_bodies("x") == ("60-N-T-10-X",
                                                                                                  "75-M-H-10-X")
    assert ri.helper_bodies("j") == ("75-M-H-10-J",) == stage1.JETSON_BODIES and set(ri.HELPERS) == {"k", "x", "j"}
    j = ri.listed("j")  # 2 model files, 10 references, 18 transitions
    assert len(j) == 30 and sum(p.startswith("round2/refine/jetson_75/models/") for p in j) == 2
    assert sum("/refs/" in p for p in j) == 10 and sum("/transitions/" in p for p in j) == 18
    assert ri.SIZE_BLAS_THREADS >= 2  # with one BLAS thread the picks' ledger JSON is not the recorded one


# ------------------------------------------------------------------------------------------------
# The check logic on temporary directories
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def tree(tmp_path):
    """A study with one file right, one wrong and one missing, and a source checkout holding all three right."""
    study, src = tmp_path / "study", tmp_path / "src" / "hardware" / "trex_form_study"
    (src / "designlab").mkdir(parents=True)
    data = {"a/right.npz": b"right", "a/wrong.npz": b"wrong", "b/missing.json": b"missing"}
    for rel, b in data.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(b)
    (study / "a").mkdir(parents=True)
    (study / "a" / "right.npz").write_bytes(b"right")
    (study / "a" / "wrong.npz").write_bytes(b"not it")
    return study, src, {rel: _sha(b) for rel, b in data.items()}


def test_tally_counts_matching_missing_and_wrong(tree):
    study, src, files = tree
    assert ri.tally(files, study) == dict(ok=["a/right.npz"], missing=["b/missing.json"], wrong=["a/wrong.npz"])
    assert ri.tally(files, src) == dict(ok=sorted(files), missing=[], wrong=[])


def test_put_matching_puts_only_files_with_their_sha256(tree):
    study, src, files = tree
    (src / "b" / "missing.json").write_bytes(b"changed")  # the source's copy is not the listed file
    put, refused = ri.put_matching(files, src, study)
    assert put == ["a/wrong.npz (replaced a file with another sha256)"]  # the right one is left alone
    assert len(refused) == 1 and refused[0].startswith("b/missing.json: sha256")
    assert (study / "a" / "wrong.npz").read_bytes() == b"wrong" and not (study / "b" / "missing.json").exists()
    assert not list(study.rglob(".part-*"))
    (src / "b" / "missing.json").unlink()
    assert ri.put_matching({"b/missing.json": files["b/missing.json"]}, src, study)[1] == ["b/missing.json: not made"]


def test_put_refuses_a_copy_with_another_sha256(tmp_path):
    (tmp_path / "f.npz").write_bytes(b"x")
    with pytest.raises(SystemExit, match="differs"):
        ri._put(tmp_path / "f.npz", tmp_path / "out" / "f.npz", _sha(b"y"))
    assert not (tmp_path / "out" / "f.npz").exists() and not list(tmp_path.rglob(".part-*"))


def test_check_exits_zero_only_when_every_file_matches_and_the_records_agree(tree, monkeypatch, capsys):
    study, src, files = tree
    disagree = {"look": []}
    monkeypatch.setattr(ri, "STUDY", study)
    monkeypatch.setattr(ri, "listed", lambda g: files)
    monkeypatch.setattr(ri, "agreement", lambda g: disagree[g])
    assert ri.check(("look",)) == 1
    out = capsys.readouterr().out
    assert re.search(r"look\s+3\s+1\s+1\s+1\s+agree", out) and "b/missing.json" in out and "a/wrong.npz" in out
    ri.put_matching(files, src, study)
    assert ri.check(("look",)) == 0
    assert "all here with their listed sha256" in capsys.readouterr().out
    disagree["look"] = ["a/right.npz: the list has 11111111, the record 22222222"]
    assert ri.check(("look",)) == 1
    assert "1 disagreements" in capsys.readouterr().out


def test_copy_from_finds_the_study_and_refuses_this_checkout(tree, tmp_path, monkeypatch, capsys):
    """Another checkout (its root or its study), or a plain directory holding the files at their study paths (an
    unpacked archive: the route where no other checkout is at hand); never this checkout."""
    study, src, files = tree
    assert ri._study_of(src.parents[1]) == src.resolve() and ri._study_of(src) == src.resolve()
    assert ri._study_of(study) == study.resolve()
    with pytest.raises(SystemExit, match="no such directory"):
        ri._study_of(tmp_path / "nope")
    with pytest.raises(SystemExit, match="this checkout"):
        ri.copy_from(str(STUDY))
    unpacked = tmp_path / "unpacked"  # what `tar xf` of the files gives: no designlab, no hardware/trex_form_study
    for rel in files:
        (unpacked / rel).parent.mkdir(parents=True, exist_ok=True)
        (unpacked / rel).write_bytes((src / rel).read_bytes())
    monkeypatch.setattr(ri, "STUDY", study)
    monkeypatch.setattr(ri, "listed", lambda g: files)
    monkeypatch.setattr(ri, "agreement", lambda g: [])
    assert ri.copy_from(str(unpacked), ("look",)) == 0
    assert ri.tally(files, study) == dict(ok=sorted(files), missing=[], wrong=[])
    assert "copied 2, already here 1" in capsys.readouterr().out


def test_not_restored_lists_the_four_items_with_their_reasons():
    rows = ri.not_restored_here()
    assert [w for w, _, _ in rows] == [w for w, _, _ in ri.NOT_RESTORED] and len(rows) == 4
    assert "checkpoints" in rows[0][0] and "traces" in rows[1][0] and "media" in rows[2][0] and "MJCF" in rows[3][0]
    assert rows[2][1] is None and all(isinstance(n, int) for _, n, _ in (rows[0], rows[1], rows[3]))
    assert "retraining" in rows[0][2] and "designlab.stage2.clips" in rows[2][2] and "designlab.round2" in rows[3][2]
    # the traces: pinned by their records, and regenerated into a scratch run directory (G9 needs Stage 1b's turns)
    assert "TRACE_RULE" in rows[1][2] and "--out <scratch>" in rows[1][2] and "G9" in rows[1][2]
    # the clips' command keeps the committed stage2/look/clips.json (render writes it unless --no-record)
    assert "render --media DIR --no-record" in rows[2][2]
    readme = " ".join((STUDY / "tools" / "README.md").read_text().split())
    assert "clips render --media DIR --no-record" in readme
    # the README names every review and sizing-model path the item counts
    for glob in ri.NOT_RESTORED[3][1]:
        assert f"`{glob.removesuffix('/*.xml')}`" in readme, glob


def test_not_restored_counts_files_and_names_the_other_ignored_ones(tmp_path, monkeypatch, capsys):
    """Each item counts its files here (a matched directory's included; tracked and listed files not), and the
    git-ignored files that no group lists and no item names are listed (caches, logs and the venv aside): the
    round-two outputs the first globs missed are counted now, and the plant's rebuild in stage1/models shows."""
    ignored = [
        "round2/refine/carpet_lift_models/60-N-T-10.xml", "round2/refine/carpet_lift_models/feet_75/75-M-H-10.xml",
        "round2/refine/fixed_models/mat60_cf/60-M-T-10.xml", "round2/size_study/80_headroom/review/mjcf/80-M-T-10.xml",
        "round2/whatif_fr05/review/mjcf/75-M-H-10.xml", "round2/models/narrow_search/75-M-T-00.xml",
        "round2/refine/feet_75/models/75-M-T-10.json",  # a pick's case's other candidate: counted
        "round2/refine/feet_75/models/75-M-H-10.json",  # a group's file: not counted
        "round1/review/mjcf/100-M-H-00.xml",
        "stage1/runs/2026-11-08/60-N-T-10/eval/traces/R0/walk/nominal/3042.npz",
        "stage1/runs/2026-11-08/60-N-T-10/ppo/42/checkpoints/model_1507328.zip",
        "stage1/models/60-N-T-10/60-N-T-10.xml",  # the plant's rebuild: no item names it
        "logs/pytest.log", "MUJOCO_LOG.TXT", "designlab/__pycache__/plant.cpython-313.pyc", ".venv/bin/python",
    ]
    tracked = ["round1/review/mjcf/60-M-T-10.xml"]  # a committed reference MJCF beside the ignored ones
    for rel in ignored + tracked:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rel)
    monkeypatch.setattr(ri, "STUDY", tmp_path)
    monkeypatch.setattr(ri, "listed", lambda g: {"round2/refine/feet_75/models/75-M-H-10.json": "0" * 64}
                        if g == "picks-models" else {})
    monkeypatch.setattr(ri, "_git", lambda *a: sorted(ignored) if "--ignored" in a else sorted(tracked))
    assert [n for _, n, _ in ri.not_restored_here()] == [1, 1, None, 8]
    assert ri.unaccounted() == ["stage1/models/60-N-T-10/60-N-T-10.xml"]
    ri.not_restored()
    out = capsys.readouterr().out
    assert "(8 files here)" in out and "Other git-ignored files here" in out and "stage1/models/60-N-T-10" in out
    monkeypatch.setattr(ri, "_git", lambda *a: None)  # outside git: the items still counted, the rest not listed
    assert [n for _, n, _ in ri.not_restored_here()] == [1, 1, None, 9] and ri.unaccounted() is None
    ri.not_restored()
    assert "Not a git checkout" in capsys.readouterr().out


def test_the_command_line_takes_groups_and_refuses_unknown_ones():
    r = subprocess.run([sys.executable, str(STUDY / "tools" / "restore_inputs.py"), "check", "--group", "nope"],
                       cwd=STUDY, capture_output=True, text=True)
    assert r.returncode == 2 and "invalid choice" in r.stderr
    r = subprocess.run([sys.executable, str(STUDY / "tools" / "restore_inputs.py"), "not-restored"], cwd=STUDY,
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("Not restored by this tool:")


@pytest.fixture
def groups_stub(tmp_path, monkeypatch):
    """``restore``'s groups as stubs: each group's one listed file is here once its stub ran and succeeded (``here``);
    ``fail`` names the groups whose stub fails, ``rebuild`` the bodies whose models a group's stub makes in
    stage1/models (as the plant would); ``ran`` records (group, procs)."""
    s = dict(ran=[], here=set(), fail=set(), rebuild={})

    def stub(g):
        def run(procs, scratch):
            s["ran"].append((g, procs))
            for body in s["rebuild"].get(g, ()):
                (tmp_path / "stage1" / "models" / body).mkdir(parents=True, exist_ok=True)
                (tmp_path / "stage1" / "models" / body / f"{body}.xml").write_text("rebuilt")
            if g in s["fail"]:
                return 1
            s["here"].add(g)
            return 0
        return run

    monkeypatch.setattr(ri, "STUDY", tmp_path)
    monkeypatch.setattr(ri, "RESTORE", {g: stub(g) for g in ri.GROUPS})
    monkeypatch.setattr(ri, "listed", lambda g: {f"{g}.npz": "0" * 64})
    monkeypatch.setattr(ri, "tally", lambda files, root=None: dict(
        ok=[f for f in files if f[:-4] in s["here"]], missing=[f for f in files if f[:-4] not in s["here"]], wrong=[]))
    monkeypatch.setattr(ri, "check", lambda groups: 0)
    monkeypatch.setattr(ri, "git_changes", lambda: ([], []))
    return s


def test_restore_runs_the_groups_in_order_and_adds_what_one_reads(groups_stub, tmp_path, monkeypatch, capsys):
    """``restore --group look`` restores the picks' models first when they are not all here (the look ladder's price
    reads them); the groups run in ``GROUPS`` order whatever order they are asked in, then check and git status."""
    s = groups_stub
    assert ri.restore(("look",), 2, str(tmp_path)) == 0
    assert s["ran"] == [("picks-models", 2), ("look", 2)]
    assert "look reads picks-models" in capsys.readouterr().out
    s["ran"].clear()
    assert ri.restore(("look", "k", "picks-arrays"), 1, str(tmp_path)) == 0  # picks-models is here now
    assert s["ran"] == [("picks-arrays", 1), ("k", 1), ("look", 1)]
    monkeypatch.setattr(ri, "git_changes", lambda: (["M  stage1/runs/2026-11-08/60-N-T-10/refs.json"], []))
    assert ri.restore(("k",), 1, str(tmp_path)) == 0  # a change made before the run is not the run's
    calls = iter([([], []), ([" M stage1/runs/2026-11-08/60-N-T-10/refs.json"], [])])
    monkeypatch.setattr(ri, "git_changes", lambda: next(calls))
    assert ri.restore(("k",), 1, str(tmp_path)) == 1  # the run changed a tracked file
    assert "tracked files changed by this run" in capsys.readouterr().out


def test_restore_leaves_out_a_group_whose_needs_failed(groups_stub, tmp_path, capsys):
    """When the picks' models cannot be restored (one CPU: SIZE_BLAS_THREADS), the look ladder is not run (its
    clearance gate would rebuild them into stage1/models through plant.model_files), and the run fails; the groups
    that read nothing missing still run."""
    s = groups_stub
    s["fail"].add("picks-models")
    assert ri.restore(("look", "k"), 1, str(tmp_path)) == 1
    assert s["ran"] == [("picks-models", 1), ("k", 1)]
    out = capsys.readouterr().out
    assert "look reads picks-models" in out and "== look: not run: it reads picks-models" in out
    assert "copy-from DIR" in out and not (tmp_path / "stage1" / "models").exists()
    s["fail"].clear()  # brought in: look runs
    s["ran"].clear()
    assert ri.restore(("look",), 1, str(tmp_path)) == 0 and s["ran"] == [("picks-models", 1), ("look", 1)]


def test_restore_fails_on_model_files_the_run_makes_in_stage1_models(groups_stub, tmp_path, capsys):
    """A file that appears under stage1/models during the run (the plant's rebuild, which this tool exists to prevent)
    fails the run and is named with how to remove it; one that was there before is not the run's."""
    s = groups_stub
    s["here"].add("picks-models")
    (tmp_path / "stage1" / "models" / "75-M-H-10").mkdir(parents=True)
    (tmp_path / "stage1" / "models" / "75-M-H-10" / "75-M-H-10.xml").write_text("from before")
    assert ri.restore(("look",), 1, str(tmp_path)) == 0  # nothing new there
    s["rebuild"]["look"] = ("60-N-T-10",)
    assert ri.restore(("look",), 1, str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "stage1/models: this run made 1 files there" in out
    assert "stage1/models/60-N-T-10/60-N-T-10.xml" in out and "rm -r stage1/models/60-N-T-10)" in out
    assert "75-M-H-10" not in out.split("stage1/models: this run made")[1]


# ------------------------------------------------------------------------------------------------
# The restore steps on a temporary study, each regeneration a stub writing chosen bytes
# ------------------------------------------------------------------------------------------------
LOOK_RUNGS = {"60-N-T-10": {"nominal": ("V0", "V1")}, "75-M-H-10": {"fr05": ("V0",)}}


@pytest.fixture
def fake_study(tmp_path, monkeypatch):
    """A study holding two picks' lists (their two model files, references nominal and brisk, transition start) and a
    look ladder of three rungs, and none of those files; ``right[path]`` is each listed file's bytes.  ``regen`` stands
    for the child processes: it writes, where each job would, ``made[path]`` (default: the listed bytes), and records
    each job's (label, argv) in ``jobs``."""
    study = tmp_path / "study"
    run, look = study / "stage1" / "runs" / "2026-11-08", study / "stage2" / "look"
    picks = ("60-N-T-10", "75-M-H-10")
    right = {p: f"model {p}".encode() for b in picks for p in ri.model_paths(b)}
    for b in picks:
        for kind, key, names in (("refs", "references", ("nominal", "brisk")), ("transitions", "transitions",
                                                                                 ("start",))):
            entries = {}
            for n in names:
                rel = f"stage1/runs/2026-11-08/{b}/{kind}/{n}.npz"
                right[rel] = f"array {rel}".encode()
                entries[n] = dict(file=f"{kind}/{n}.npz", sha256=_sha(right[rel]))
            (run / b).mkdir(parents=True, exist_ok=True)
            (run / b / f"{kind}.json").write_text(json.dumps({key: entries}))
    ladder: dict = {}
    for b, speeds in LOOK_RUNGS.items():
        for sp, variants in speeds.items():
            for v in variants:
                rel = f"stage2/look/refs/{b}/{sp}/{v}.npz"
                right[rel] = f"rung {rel}".encode()
                ladder.setdefault(b, {}).setdefault(sp, {})[v] = dict(file=f"refs/{b}/{sp}/{v}.npz",
                                                                      sha256=_sha(right[rel]))
    look.mkdir(parents=True)
    (look / "look_refs.json").write_text(json.dumps(dict(bodies=ladder)))
    (run / "stage1_inputs.json").write_text(json.dumps(dict(files={p: _sha(b) for p, b in right.items()
                                                                   if not p.startswith("stage2/")})))
    for name, value in (("STUDY", study), ("RUN", run), ("MANIFEST", run / "stage1_inputs.json"), ("LOOK", look),
                        ("LOOK_REFS", look / "look_refs.json"), ("PICKS", picks), ("SETS", study / "stage2" / "sets")):
        monkeypatch.setattr(ri, name, value)
    made, jobs = {}, []

    def regen(job_list, procs):
        for label, argv, _ in job_list:
            jobs.append((label, argv))
            if "designlab.stage2.sets" in argv:  # sets build --set S --body B --procs N: in place, on a frozen set
                d = f"stage2/sets/{argv[argv.index('--set') + 1]}/{argv[argv.index('--body') + 1]}/"
                for rel, sha in ri.listed("sets").items():  # ... only an array whose sha256 is the manifest's
                    if rel.startswith(d) and _sha(made.get(rel, right[rel])) == sha and ri._sha(study / rel) != sha:
                        (study / rel).parent.mkdir(parents=True, exist_ok=True)
                        (study / rel).write_bytes(made.get(rel, right[rel]))
                continue
            if "designlab.stage2.look" in argv:  # look prepare --out DIR --body B --speed S --variant V ...
                dest = Path(argv[argv.index("--out") + 1])
                opt = {k: [argv[i + 1] for i, a in enumerate(argv) if a == k] for k in ("--body", "--speed",
                                                                                       "--variant")}
                writes = {f"stage2/look/refs/{b}/{sp}/{v}.npz": dest / "refs" / b / sp / f"{v}.npz"
                          for b in opt["--body"] for sp in opt["--speed"] for v in opt["--variant"]}
            else:  # this tool's own children: _size BODY WORK, _refs BODY OUT NAMES, _transitions BODY OUT PROCS NAMES
                cmd, body, dest = argv[2], argv[3], Path(argv[4])
                if cmd == "_size":
                    writes = {p: dest / p for p in ri.model_paths(body)}
                else:
                    kind, names = ("refs", argv[5:]) if cmd == "_refs" else ("transitions", argv[6:])
                    writes = {f"stage1/runs/2026-11-08/{body}/{kind}/{n}.npz": dest / kind / f"{n}.npz" for n in names}
            for rel, path in writes.items():
                if rel in right:  # a rung the ladder does not list: prepare would make it; not put in place anyway
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(made.get(rel, right[rel]))
        return {label: 0 for label, _, _ in job_list}

    monkeypatch.setattr(ri, "run_all", regen)
    return SimpleNamespace(study=study, right=right, made=made, jobs=jobs)


def _place(fs, rel: str, data: bytes | None = None) -> None:
    (fs.study / rel).parent.mkdir(parents=True, exist_ok=True)
    (fs.study / rel).write_bytes(fs.right[rel] if data is None else data)


def _here_bytes(fs, rel: str) -> bytes | None:
    p = fs.study / rel
    return p.read_bytes() if p.exists() else None


def test_restore_picks_models_puts_a_bodys_two_files_together_or_not_at_all(fake_study, tmp_path, capsys):
    fs = fake_study
    t_xml, t_json = ri.model_paths("60-N-T-10")
    h_xml, h_json = ri.model_paths("75-M-H-10")
    fs.made[h_json] = b"one BLAS thread"  # 75-M-H-10's XML regenerates right, its ledger JSON does not
    assert ri.restore_picks_models(2, tmp_path) == 1
    assert [label for label, _ in fs.jobs] == ["60-N-T-10", "75-M-H-10"]
    assert _here_bytes(fs, t_xml) == fs.right[t_xml] and _here_bytes(fs, t_json) == fs.right[t_json]
    assert _here_bytes(fs, h_xml) is None and _here_bytes(fs, h_json) is None  # neither of the two
    out = capsys.readouterr().out
    assert f"75-M-H-10 regenerated with other sha256 ({Path(h_xml).name} " in out and "not put in place" in out
    assert "60-N-T-10 regenerated, matching" in out
    fs.jobs.clear()
    fs.made.clear()  # with the right threads: only the body still missing is sized, and both its files go in
    assert ri.restore_picks_models(2, tmp_path) == 0
    assert [label for label, _ in fs.jobs] == ["75-M-H-10"]
    assert _here_bytes(fs, h_xml) == fs.right[h_xml] and _here_bytes(fs, h_json) == fs.right[h_json]
    assert not list(fs.study.rglob(".part-*")) and not (fs.study / "stage1" / "models").exists()


def test_restore_picks_arrays_maps_each_scratch_file_to_its_run_path(fake_study, tmp_path, capsys):
    fs = fake_study
    run = "stage1/runs/2026-11-08"
    _place(fs, f"{run}/60-N-T-10/refs/nominal.npz")  # right already: left alone, not regenerated
    fs.made[f"{run}/75-M-H-10/refs/nominal.npz"] = b"another nominal"
    assert ri.restore_picks_arrays(2, tmp_path) == 1
    refs_jobs = [argv for label, argv in fs.jobs if argv[2] == "_refs"]
    assert [a[3:4] + a[5:] for a in refs_jobs] == [["60-N-T-10", "brisk"], ["75-M-H-10", "brisk", "nominal"]]
    for rel in (f"{run}/60-N-T-10/refs/nominal.npz", f"{run}/60-N-T-10/refs/brisk.npz",
                f"{run}/60-N-T-10/transitions/start.npz", f"{run}/75-M-H-10/refs/brisk.npz"):
        assert _here_bytes(fs, rel) == fs.right[rel], rel
    assert _here_bytes(fs, f"{run}/75-M-H-10/refs/nominal.npz") is None  # regenerated with another sha256
    # 75-M-H-10's transitions are made from its references, which are not all here: not regenerated
    assert [argv[3] for _, argv in fs.jobs if argv[2] == "_transitions"] == ["60-N-T-10"]
    assert _here_bytes(fs, f"{run}/75-M-H-10/transitions/start.npz") is None
    out = capsys.readouterr().out
    assert "picks-arrays 75-M-H-10 refs: 1 put in place" in out and "refs/nominal.npz: sha256" in out
    assert "picks-arrays 75-M-H-10 transitions: not regenerated, its references are not all here" in out
    assert "picks-arrays 60-N-T-10 transitions: 1 put in place" in out


def test_restore_look_maps_each_rung_and_refuses_one_with_another_sha256(fake_study, tmp_path, capsys):
    fs = fake_study
    _place(fs, "stage2/look/refs/60-N-T-10/nominal/V0.npz")  # right already
    _place(fs, "stage2/look/refs/75-M-H-10/fr05/V0.npz", b"an older rung")  # wrong here, regenerated wrong again
    fs.made["stage2/look/refs/75-M-H-10/fr05/V0.npz"] = b"another rung"
    assert ri.restore_look(2, tmp_path) == 1
    [(label, argv)] = fs.jobs
    assert label == "look" and argv[1:4] == ["-m", "designlab.stage2.look", "prepare"]
    opt = {k: [argv[i + 1] for i, a in enumerate(argv) if a == k]
           for k in ("--procs", "--body", "--speed", "--variant")}
    assert opt == {"--procs": ["2"], "--body": ["60-N-T-10", "75-M-H-10"], "--speed": ["fr05", "nominal"],
                   "--variant": ["V0", "V1"]}  # the bad rungs' bodies, speeds and variants
    assert _here_bytes(fs, "stage2/look/refs/60-N-T-10/nominal/V1.npz") == fs.right[
        "stage2/look/refs/60-N-T-10/nominal/V1.npz"]
    assert _here_bytes(fs, "stage2/look/refs/75-M-H-10/fr05/V0.npz") == b"an older rung"  # not replaced
    out = capsys.readouterr().out
    assert "look: 1 put in place\n  refs/60-N-T-10/nominal/V1.npz" in out
    assert "look: 1 regenerated with another sha256 or not made, not put in place:\n  refs/75-M-H-10/fr05/V0.npz" in out
    fs.jobs.clear()
    fs.made.clear()
    assert ri.restore_look(2, tmp_path) == 0 and _here_bytes(fs, "stage2/look/refs/75-M-H-10/fr05/V0.npz") == fs.right[
        "stage2/look/refs/75-M-H-10/fr05/V0.npz"]
    assert ri.restore_look(2, tmp_path) == 0 and "look: all here" in capsys.readouterr().out


def test_restore_helper_runs_only_the_steps_whose_files_are_bad(monkeypatch, tmp_path):
    """Arm K's and the -X helpers: models only when a model file is missing or wrong, then the references of the
    bodies whose references are (the bodies at once), then the transitions of the bodies whose transitions are."""
    k = ri.listed("k")
    bad = {p: s for p, s in k.items() if "/60-N-H-10-K/refs/" in p or "/60-N-T-10-K/transitions/" in p}
    jobs = []
    monkeypatch.setattr(ri, "_bad", lambda files: bad)
    monkeypatch.setattr(ri, "run_all", lambda js, procs: jobs.append(([argv[2:] for _, argv, _ in js], procs))
                        or {label: 0 for label, _, _ in js})
    assert ri.restore_helper("k", 2, tmp_path) == 0
    assert jobs == [([["arrays", "60-N-H-10-K", "refs"]], 2), ([["arrays", "60-N-T-10-K", "transitions"]], 1)]
    jobs.clear()
    bad = {ri.helper("k").model_paths("60-N-T-10-K")[1]: "0" * 64}
    assert ri.restore_helper("k", 2, tmp_path) == 0 and jobs == [([["models"]], 1)]


# ------------------------------------------------------------------------------------------------
# The sets group: its list is the frozen sets' manifests
# ------------------------------------------------------------------------------------------------
def _frozen_set(fs, set_name="V0", body="75-M-H-10", walks=("nominal", "v0.4"), moves=("start", "turn90_pi4_left"),
                freezes=1) -> dict:
    """A reference set under the fake study with ``freezes`` freezes (the last one holds every entry; an earlier one
    only the first walk) and none of its arrays; its files' bytes go to ``fs.right``.  Returns the latest manifest."""
    d = fs.study / "stage2" / "sets" / set_name / body
    rel = f"stage2/sets/{set_name}/{body}"
    refs, trans, manifest = {}, {}, {}
    for w in walks:
        fs.right[f"{rel}/refs/{w}.npz"] = f"walk {set_name} {body} {w}".encode()
        fs.right[f"{rel}/look/{w}.npz"] = f"look {set_name} {body} {w}".encode()
        refs[w] = dict(file=f"refs/{w}.npz", sha256=_sha(fs.right[f"{rel}/refs/{w}.npz"]), look_file=f"look/{w}.npz",
                       look_sha256=_sha(fs.right[f"{rel}/look/{w}.npz"]))
        manifest[w] = dict(kind="walk", sha256=refs[w]["sha256"], entry_sha256="e" * 64, look_sha256=refs[w]["look_sha256"])
    for m in moves:
        fs.right[f"{rel}/transitions/{m}.npz"] = f"move {set_name} {body} {m}".encode()
        trans[m] = dict(file=f"transitions/{m}.npz", sha256=_sha(fs.right[f"{rel}/transitions/{m}.npz"]))
        manifest[m] = dict(kind="turn" if m.startswith("turn") else m, sha256=trans[m]["sha256"], entry_sha256="e" * 64)
    trans["turn90_1_left"] = dict(name="turn90_1_left", available=False, reason="no reference within the heading rule")
    d.mkdir(parents=True)
    (d / "refs.json").write_text(json.dumps(dict(references=refs)))
    (d / "transitions.json").write_text(json.dumps(dict(transitions=trans)))
    first = {walks[0]: manifest[walks[0]]}
    (d / "set.json").write_text(json.dumps(dict(set=set_name, body=body, freezes=[
        dict(index=i, manifest=manifest if i == freezes - 1 else first) for i in range(freezes)])))
    return manifest


def test_the_sets_group_lists_nothing_while_no_set_is_frozen(fake_study, tmp_path, capsys):
    fs = fake_study
    assert ri.frozen_sets() == [] and ri.listed("sets") == {} and ri.agreement("sets") == []
    _frozen_set(fs, freezes=0)  # built, not frozen
    assert ri.frozen_sets() == [] and ri.listed("sets") == {}
    assert ri.restore_sets(2, tmp_path) == 0 and fs.jobs == []
    assert "sets: no reference set is frozen" in capsys.readouterr().out
    assert ri.tally(ri.listed("sets")) == dict(ok=[], missing=[], wrong=[])


def test_the_sets_group_lists_the_latest_freeze_of_every_set(fake_study):
    """Per frozen walk its Stage 1-format file and its look-format twin, per frozen move its file, each with the
    manifest's sha256; an entry that is not in the manifest (an item with no reference) is not listed; the latest
    freeze counts."""
    fs = fake_study
    m = _frozen_set(fs, freezes=2)
    _frozen_set(fs, "F25F9cl", walks=("v0.4",), moves=())
    _frozen_set(fs, "F25F9", freezes=0)
    assert [(s, b) for s, b, _ in ri.frozen_sets()] == [("F25F9cl", "75-M-H-10"), ("V0", "75-M-H-10")]
    d = "stage2/sets/V0/75-M-H-10"
    files = ri.listed("sets")
    assert files == {
        f"{d}/refs/nominal.npz": m["nominal"]["sha256"], f"{d}/look/nominal.npz": m["nominal"]["look_sha256"],
        f"{d}/refs/v0.4.npz": m["v0.4"]["sha256"], f"{d}/look/v0.4.npz": m["v0.4"]["look_sha256"],
        f"{d}/transitions/start.npz": m["start"]["sha256"],
        f"{d}/transitions/turn90_pi4_left.npz": m["turn90_pi4_left"]["sha256"],
        "stage2/sets/F25F9cl/75-M-H-10/refs/v0.4.npz": _sha(fs.right["stage2/sets/F25F9cl/75-M-H-10/refs/v0.4.npz"]),
        "stage2/sets/F25F9cl/75-M-H-10/look/v0.4.npz": _sha(fs.right["stage2/sets/F25F9cl/75-M-H-10/look/v0.4.npz"])}
    assert all(files[p] == _sha(fs.right[p]) for p in files)
    assert ri.agreement("sets") == []
    # the set's own records must hold the frozen arrays
    path = fs.study / d / "transitions.json"
    doc = json.loads(path.read_text())
    doc["transitions"]["start"]["sha256"] = "0" * 64
    del doc["transitions"]["turn90_pi4_left"]
    path.write_text(json.dumps(doc))
    assert ri.agreement("sets") == [
        f"{d}/transitions/start.npz: the list has {m['start']['sha256'][:8]}, the record 00000000",
        f"{d}/transitions/turn90_pi4_left.npz: in the freeze, and its entry in the set's records holds no sha256"]


def test_restore_sets_runs_sets_build_on_each_frozen_set_that_lacks_an_array(fake_study, tmp_path, capsys):
    fs = fake_study
    _frozen_set(fs)
    _frozen_set(fs, "F25F9cl", walks=("v0.4",), moves=("start",))
    d = "stage2/sets/V0/75-M-H-10"
    for rel in ri.listed("sets"):
        if rel.startswith("stage2/sets/F25F9cl/"):
            _place(fs, rel)  # that set is all here
    _place(fs, f"{d}/refs/nominal.npz")
    _place(fs, f"{d}/look/v0.4.npz", b"another array")
    records = {p: p.read_bytes() for p in (fs.study / d).glob("*.json")}
    assert ri.restore_sets(2, tmp_path) == 0
    assert fs.jobs == [("V0/75-M-H-10", [sys.executable, "-m", "designlab.stage2.sets", "build", "--set", "V0",
                                         "--body", "75-M-H-10", "--procs", "2"])]
    assert ri.tally(ri.listed("sets")) == dict(ok=sorted(ri.listed("sets")), missing=[], wrong=[])
    assert {p: p.read_bytes() for p in (fs.study / d).glob("*.json")} == records  # no record is written
    assert "sets: V0 75-M-H-10: 5 arrays: sets build" in capsys.readouterr().out
    assert ri.restore_sets(2, tmp_path) == 0 and len(fs.jobs) == 1 and "sets: all here" in capsys.readouterr().out
    # an array today's code does not reproduce is not put in place, and the group fails
    (fs.study / d / "transitions" / "start.npz").unlink()
    fs.made[f"{d}/transitions/start.npz"] = b"other code's array"
    assert ri.restore_sets(1, tmp_path) == 1 and not (fs.study / d / "transitions" / "start.npz").exists()
    out = capsys.readouterr().out
    assert "1 arrays still missing or wrong" in out and f"{d}/transitions/start.npz" in out
    assert fs.jobs[-1][1][-1] == "1"


def test_copy_from_brings_a_frozen_sets_arrays(fake_study, tmp_path, capsys):
    fs = fake_study
    _frozen_set(fs)
    src = tmp_path / "other"
    for rel in ri.listed("sets"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(fs.right[rel])
    (src / "stage2/sets/V0/75-M-H-10/look/v0.4.npz").write_bytes(b"not the frozen file")
    assert ri.copy_from(str(src), ("sets",)) == 1
    out = capsys.readouterr().out
    assert "== sets: copied 5, already here 0" in out and "not copied (1)" in out
    assert ri.tally(ri.listed("sets"))["missing"] == ["stage2/sets/V0/75-M-H-10/look/v0.4.npz"]

