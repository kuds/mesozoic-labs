"""Tests for tools/restore_inputs.py: the picks' list (``stage1/runs/2026-11-08/stage1_inputs.json``) against the
committed records, every group's list against its records, and the check, copy and not-restored logic on temporary
directories.  Fast: nothing is regenerated (``restore`` itself was proven in a fresh worktree; tools/README.md).

Run from hardware/trex_form_study:  python -m pytest tests/test_restore_inputs.py -q
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

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
    assert len(paths) == len(set(paths)) == 6 + 84 + 60 + 60 + 73
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
    """Models before arrays; the look ladder (whose price reads the picks' models) after them."""
    assert ri.GROUPS == ("picks-models", "picks-arrays", "k", "x", "look")
    for g, needs in ri.NEEDS.items():
        assert all(ri.GROUPS.index(n) < ri.GROUPS.index(g) for n in needs)
    assert ri.helper_bodies("k") == ("60-N-T-10-K", "60-N-H-10-K") and ri.helper_bodies("x") == ("60-N-T-10-X",
                                                                                                  "75-M-H-10-X")
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


def test_copy_from_finds_the_study_and_refuses_this_checkout(tree):
    study, src, _ = tree
    assert ri._study_of(src.parents[1]) == src.resolve() and ri._study_of(src) == src.resolve()
    with pytest.raises(SystemExit, match="not a checkout"):
        ri._study_of(study)
    with pytest.raises(SystemExit, match="this checkout"):
        ri.copy_from(str(STUDY))


def test_not_restored_lists_the_four_items_with_their_reasons():
    rows = ri.not_restored_here()
    assert [w for w, _, _ in rows] == [w for w, _, _ in ri.NOT_RESTORED] and len(rows) == 4
    assert "checkpoints" in rows[0][0] and "traces" in rows[1][0] and "media" in rows[2][0] and "MJCF" in rows[3][0]
    assert rows[2][1] is None and all(isinstance(n, int) for _, n, _ in (rows[0], rows[1], rows[3]))
    assert "retraining" in rows[0][2] and "designlab.stage2.clips" in rows[2][2] and "designlab.round2" in rows[3][2]


def test_the_command_line_takes_groups_and_refuses_unknown_ones():
    r = subprocess.run([sys.executable, str(STUDY / "tools" / "restore_inputs.py"), "check", "--group", "nope"],
                       cwd=STUDY, capture_output=True, text=True)
    assert r.returncode == 2 and "invalid choice" in r.stderr
    r = subprocess.run([sys.executable, str(STUDY / "tools" / "restore_inputs.py"), "not-restored"], cwd=STUDY,
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("Not restored by this tool:")
