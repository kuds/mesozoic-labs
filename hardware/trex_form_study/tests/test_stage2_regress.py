"""Tests for the closed-loop bit-identity regression (designlab/stage2/regress.py; round A design, D18 and section
4.2) and for the one Stage 1 edit it guards (``control.Track.add_periodic``'s optional ``window_strides`` key).

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage2_regress.py -q
Fast by default.  ``test_regress_record_is_current`` needs no arrays: it fails, by design, after any edit to a Stage 1
module until ``python -m designlab.stage2.regress run --tier full --record stage2/regress/regress.json`` has been run
on the new code and its record committed.  ``test_regress_fast`` (8 closed-loop episodes, about a minute) runs with
STAGE2_SLOW=1, in a checkout without uncommitted changes.  The tests that build a plant or run an eval need the
restored arrays and the round-two model files (git-ignored; ``python tools/restore_inputs.py``) and skip without
them, so that none makes ``plant.model_files`` write ``stage1/models/``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab.stage1 import control, panels, plant, speedfit  # noqa: E402
from designlab.stage1.__main__ import round_two_models  # noqa: E402
from designlab.stage1.refs import Reference  # noqa: E402
from designlab.stage2 import regress, s2io  # noqa: E402

BODY = "75-M-H-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
slow = pytest.mark.skipif(not os.environ.get("STAGE2_SLOW"), reason="8 closed-loop episodes; set STAGE2_SLOW=1")
needs_plant = pytest.mark.skipif(not (all((RUN / BODY / "refs" / f"{g}.npz").exists() for g in ("nominal", "brisk"))
                                      and round_two_models(BODY)),
                                 reason="the Stage 1 arrays or the round-two model files are not restored "
                                        "(tools/restore_inputs.py)")


def _in_git() -> bool:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True).returncode == 0
    except OSError:
        return False


needs_git = pytest.mark.skipif(not _in_git(), reason="git is missing, or this is not a checkout: a run names a commit")


# ------------------------------------------------------------------------------------------------
# The committed record and the cases
# ------------------------------------------------------------------------------------------------
def test_regress_record_is_current():
    """The committed record is of the full tier, every episode matched, and its four source hashes are today's (no
    hash is a constant here: a rerun on the new code is the only way to pass after a Stage 1 edit)."""
    assert regress.check() == []


def test_cases_only_name_committed_records():
    """Every (row, condition) of every case has its committed record, and the record holds an episode with a trace
    sha256 for each seed the case names (the JSON records are committed; the traces are not)."""
    for case in regress.CASES:
        assert case["seeds"] == panels.SEEDS[: len(case["seeds"])], case["id"]
        assert (ROOT / case["inputs"] / case["body"] / "refs.json").is_file(), case["id"]
        for row in case["rows"]:
            for cond in case["conditions"]:
                path = regress.committed_record(case, row, cond)
                assert path.is_file(), path
                doc = json.loads(path.read_text())
                assert (doc["body"], doc["row"], doc["key_name"]) == (case["body"], row, cond)
                by_seed = {e["seed"]: e for e in doc["episodes"]}
                for seed in case["seeds"]:
                    assert len(by_seed[seed]["trace"]["sha256"]) == 64, (path, seed)
                    assert by_seed[seed]["trace"]["path"] == f"{row}/{cond}/{seed}.npz"


def test_cases_are_the_design_table():
    """Design 4.2: the fast tier is 8 episodes, the full tier adds seed 3042 of the 22 hard-floor conditions of two
    bodies under two rows and the 160 Stage 1b turn episodes; ids are unique; no eval writes into a run directory.
    The full tier's own cases hold the fast tier's 8 episodes again, so its 256 comparisons are of 248 distinct
    committed episodes (``regress.distinct``, ``COUNT_RULE``): 248 is what a full record proves."""
    n = {t: sum(len(regress.episodes_of(c)) for c in regress.cases_of(t)) for t in regress.TIERS}
    assert n == {"fast": 8, "full": 8 + 2 * 2 * 22 + 2 * 4 * 20} and n["full"] == 256
    named = {t: [dict(run=c["run"], body=c["body"], row=r, condition=k, seed=s) for c in regress.cases_of(t)
                 for r, k, s in regress.episodes_of(c)] for t in regress.TIERS}
    assert {t: regress.distinct(named[t]) for t in regress.TIERS} == {"fast": 8, "full": 2 * 2 * 22 + 2 * 4 * 20}
    assert regress.distinct(named["full"]) == 248 == regress.distinct(named["full"][8:])  # the fast 8 are in the rest
    assert regress.distinct(named["fast"] + named["fast"]) == 8 and regress.distinct([]) == 0
    assert regress.distinct([dict(named["fast"][0], case="a"), dict(named["fast"][0], case="b")]) == 1
    assert regress.distinct([named["fast"][0], dict(named["fast"][0], run="another/run")]) == 2
    assert len({c["id"] for c in regress.CASES}) == len(regress.CASES)
    assert regress.cases_of("fast") == [c for c in regress.CASES if c["tier"] == "fast"]
    assert regress.cases_of("full") == list(regress.CASES)
    with pytest.raises(ValueError):
        regress.cases_of("slow")
    assert set(regress.HARD_FLOOR) == {c.key for c in panels.conditions() if c.floor == "hard"}
    assert len(regress.HARD_FLOOR) == 22
    assert set(regress.STAGE1B_TURNS) == {k for k in speedfit.ROW_CONDITIONS if k.startswith("turn/")}
    assert regress.SPEED_RUN_ENV == speedfit.SPEED_RUN_ENV
    assert (regress.SEED0, regress.SEEDS_PER_CONDITION) == (panels.SEEDS[0], len(panels.SEEDS))
    assert Path(regress.STAGE1B_RUN) == speedfit.RUN.relative_to(ROOT)
    stage1b = [c for c in regress.CASES if c["run"] == regress.STAGE1B_RUN]
    assert all(c["row_module"] == regress.SPEED_MODULE and c["inputs"] == regress.STAGE1_RUN for c in stage1b)
    assert sum(len(regress.episodes_of(c)) for c in stage1b if c["tier"] == "full") == 160
    for case in regress.CASES:
        cmd = regress.command(case, "OUT")
        assert cmd[:3] == ["-m", "designlab.stage1", "eval"] and cmd[-2:] == ["--out", "OUT"]
        assert cmd[cmd.index("--workers") + 1] == "1" and cmd[cmd.index("--episodes") + 1] == str(len(case["seeds"]))
        assert cmd[cmd.index("--inputs") + 1] == case["inputs"] and "--behaviors" not in cmd
        assert ("--row-module" in cmd) == (case["row_module"] is not None)


def test_eval_environment(monkeypatch):
    """Every eval gets one BLAS thread, no GL backend and no STAGE1B_SPEED_RUN, whatever the caller's environment."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv(regress.SPEED_RUN_ENV, "/elsewhere")
    env = regress.eval_env()
    assert [env[v] for v in s2io.THREAD_VARS] == ["1", "1", "1"] and env["MUJOCO_GL"] == "disable"
    assert regress.SPEED_RUN_ENV not in env and os.environ[regress.SPEED_RUN_ENV] == "/elsewhere"


# ------------------------------------------------------------------------------------------------
# The comparison, on synthetic records
# ------------------------------------------------------------------------------------------------
def _case(tmp_path, monkeypatch, seeds=(3042, 3043)):
    """A one-row, one-condition case under a study directory of its own, with its committed record and a regenerated
    run that equals it (but for wall_s and the key order)."""
    monkeypatch.setattr(regress, "STUDY", tmp_path / "study")
    case = dict(id="synthetic", tier="fast", run="runs/a", inputs="runs/a", body="B", row_module=None,
                rows=("R2/42",), conditions=("turn/x_left",), seeds=tuple(seeds))
    out = tmp_path / "out"
    old, new = [], []
    for seed in seeds:
        rel = f"R2/42/turn/x_left/{seed}.npz"
        trace = out / "B" / "eval" / "traces" / rel
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_bytes(b"trace of %d" % seed)
        sha = s2io.sha256_file(trace)
        e = dict(seed=seed, fell=False, strides=19.043471455636954, drift=float("nan"), count=3,
                 measures=dict(window=dict(duty=[0.5630873825234953, 0.5462907418516296], flight_s=0.0)),
                 trace=dict(path=rel, sha256=sha, rows=7, every_s=0.02), relaxed=[], wall_s=4.0)
        old.append(e)
        new.append(dict(reversed(list(dict(e, wall_s=9.5).items()))))
    committed = regress.committed_record(case, "R2/42", "turn/x_left")
    assert committed == tmp_path / "study" / "runs" / "a" / "B" / "eval" / "R2" / "42" / "turn" / "x_left.json"
    s2io.write_json(committed, dict(status="done", episodes=old))
    regenerated = out / "B" / "eval" / "R2" / "42" / "turn" / "x_left.json"
    s2io.write_json(regenerated, dict(status="done", episodes=new))
    return case, out, committed, regenerated


def _rewrite(path, change):
    doc = json.loads(path.read_text())
    change(doc["episodes"])
    s2io.write_json(path, doc)


def test_compare_equal_and_each_kind_of_difference(tmp_path, monkeypatch):
    case, out, committed, regenerated = _case(tmp_path, monkeypatch)
    got = regress.compare_case(case, out)
    assert [(e["row"], e["condition"], e["seed"]) for e in got] == regress.episodes_of(case)
    assert all(e["trace_equal"] and e["record_equal"] and regress.matched(e) for e in got)  # wall_s and a NaN aside
    assert all("differs" not in e and "note" not in e and e["trace_sha256"] == e["recorded"] for e in got)
    assert got[0]["record"] == "runs/a/B/eval/R2/42/turn/x_left.json" and got[0]["case"] == "synthetic"

    # a value no trace holds (the reason the episode record is compared as well): the last bit of ``strides``
    _rewrite(regenerated, lambda eps: eps[0].update(strides=float(np.nextafter(eps[0]["strides"], 1e9))))
    a, b = regress.compare_case(case, out)
    assert a["trace_equal"] and not a["record_equal"] and a["differs"] == ["strides"] and not regress.matched(a)
    assert regress.matched(b)

    # an int where the committed record holds a float, a nested float, a missing and an added key
    def change(eps):
        eps[0].update(strides=19.043471455636954, count=3.0, extra=1)
        eps[0]["measures"]["window"]["duty"][1] = 0.5462907418516297
        del eps[0]["fell"]

    _rewrite(regenerated, change)
    a = regress.compare_case(case, out)[0]
    assert not a["record_equal"] and set(a["differs"]) == {"count", "measures.window.duty[1]", "fell (missing)",
                                                           "extra (added)"}

    # the trace's bytes (the regenerated record still naming the committed sha256: the file itself is read)
    case, out, committed, regenerated = _case(tmp_path / "second", monkeypatch)
    (out / "B" / "eval" / "traces" / "R2" / "42" / "turn" / "x_left" / "3043.npz").write_bytes(b"other bytes")
    a, b = regress.compare_case(case, out)
    assert regress.matched(a) and b["record_equal"] and not b["trace_equal"] and b["trace_sha256"] != b["recorded"]

    # an episode the eval did not write, a trace it did not write, an episode that failed to run
    def lose(eps):
        eps[1].update(error="Traceback ...\nValueError: boom")
        del eps[0]

    _rewrite(regenerated, lose)
    (out / "B" / "eval" / "traces" / "R2" / "42" / "turn" / "x_left" / "3042.npz").unlink()
    a, b = regress.compare_case(case, out)
    assert not a["trace_equal"] and not a["record_equal"] and a["trace_sha256"] is None
    assert "no trace file" in a["note"] and "no episode of this seed" in a["note"]
    assert not b["record_equal"] and "ValueError: boom" in b["note"]

    # no regenerated record at all (an eval that failed), and a committed record without the seed
    regenerated.unlink()
    assert not any(e["record_equal"] for e in regress.compare_case(case, out))
    assert "regenerated: no record" in regress.compare_case(case, out)[1]["note"]
    _rewrite(committed, lambda eps: eps.pop())
    b = regress.compare_case(case, out)[1]
    assert b["recorded"] is None and not b["trace_equal"] and "committed: no episode of this seed" in b["note"]


def test_run_case_keeps_its_result_and_the_evidence(tmp_path, monkeypatch):
    """One case through ``run_case`` with the eval replaced by a copy of prepared files: the command, the log, the
    kept result; the work directory is removed only when everything matched; a failed eval matches nothing."""
    case, out, committed, regenerated = _case(tmp_path, monkeypatch)
    scratch, state, seen = tmp_path / "scratch", dict(code=0), []

    def fake_eval(cmd, cwd=None, env=None, stdout=None, stderr=None):
        seen.append((cmd, cwd, env))
        stdout.write("the eval's own output\n")
        shutil.copytree(out, cmd[-1], dirs_exist_ok=True)
        return SimpleNamespace(returncode=state["code"])

    monkeypatch.setattr(regress.subprocess, "run", fake_eval)
    monkeypatch.setattr(regress, "_git", lambda *args: "c" * 40 + "\n")
    monkeypatch.setenv(regress.SPEED_RUN_ENV, "/elsewhere")
    lines = []
    doc = regress.run_case(case, scratch, dict(k=1), log=lines.append)
    cmd, cwd, env = seen[0]
    assert cmd[0] == sys.executable and cmd[1:-1] == regress.command(case, "x")[:-1]
    assert Path(cmd[-1]) == scratch / "work" / "synthetic" and cwd == regress.STUDY
    assert regress.SPEED_RUN_ENV not in env and env["OMP_NUM_THREADS"] == "1" and env["MUJOCO_GL"] == "disable"
    assert (doc["matched"], doc["of"], doc["returncode"], doc["key"], doc["commit"]) == (2, 2, 0, dict(k=1), "c" * 40)
    assert doc["command"][0] == "python" and doc["command"][-2:] == ["--out", "<scratch>"]
    assert json.loads((scratch / "cases" / "synthetic.json").read_text()) == json.loads(json.dumps(s2io.plain(doc)))
    assert not (scratch / "work" / "synthetic").exists()  # everything matched: the regenerated files are not kept
    log_text = (scratch / "logs" / "synthetic.log").read_text()
    assert log_text.startswith("$ python -m designlab.stage1 eval --body B ") and "the eval's own output" in log_text
    assert lines == ["synthetic: 2 of 2 episodes match (0 s)"]
    assert regress._kept(case, scratch, dict(k=1)) == json.loads(json.dumps(s2io.plain(doc)))
    assert regress._kept(case, scratch, dict(k=2)) is None  # another key: other sources, versions or regress.py

    # a difference: reported line by line, the regenerated files kept for inspection
    _rewrite(regenerated, lambda eps: eps[1].update(strides=1.0))
    (scratch / "work" / "synthetic").mkdir(parents=True)
    (scratch / "work" / "synthetic" / "stale.txt").write_text("of an earlier attempt")
    doc = regress.run_case(case, scratch, dict(k=1), log=lines.append)
    assert (doc["matched"], doc["of"]) == (1, 2) and lines[-2].startswith("synthetic: 1 of 2 episodes match")
    assert lines[-1].startswith("  DIFFERENT B R2/42 turn/x_left seed 3043: trace equal, record differs at strides")
    assert (scratch / "work" / "synthetic" / "B" / "eval").is_dir()
    assert not (scratch / "work" / "synthetic" / "stale.txt").exists()  # each attempt starts from an empty directory
    assert regress._kept(case, scratch, dict(k=1))["matched"] == 1  # the eval ran to its end: its difference is kept

    # an eval that fails matches nothing, whatever it wrote
    _rewrite(regenerated, lambda eps: eps[1].update(strides=eps[0]["strides"]))
    state["code"] = 3
    doc = regress.run_case(case, scratch, dict(k=1), log=lines.append)
    assert (doc["matched"], doc["of"], doc["returncode"]) == (0, 2, 3)
    assert all(e["note"] == "the eval exited with code 3" for e in doc["episodes"])
    assert all(e["trace_sha256"] == e["recorded"] for e in doc["episodes"])  # read, and still not a match
    assert "eval exit code 3" in lines[-3] and (scratch / "work" / "synthetic").is_dir()
    # ... and is not kept (RESUME_RULE): the next run on this scratch directory starts that eval again
    assert json.loads((scratch / "cases" / "synthetic.json").read_text())["returncode"] == 3
    assert regress._kept(case, scratch, dict(k=1)) is None
    state["code"] = -9  # killed by a signal
    assert regress.run_case(case, scratch, dict(k=1), log=lines.append)["returncode"] == -9
    assert regress._kept(case, scratch, dict(k=1)) is None
    state["code"] = 0
    assert regress.run_case(case, scratch, dict(k=1), log=lines.append)["matched"] == 2
    assert regress._kept(case, scratch, dict(k=1))["returncode"] == 0


def test_differing_is_exact():
    nan = float("nan")
    assert regress.differing(dict(a=nan, b=[1, 2.0, None], c="x"), dict(c="x", b=[1, 2.0, None], a=nan)) == []
    assert regress.differing(dict(a=0.1 + 0.2), dict(a=0.3)) == ["a"]
    assert regress.differing(dict(a=0.0), dict(a=-0.0)) == ["a"]
    assert regress.differing(dict(a=[1, 2]), dict(a=[1, 2, 3])) == ["a"]
    assert regress.differing(dict(a=True), dict(a=1)) == ["a"]
    assert regress.differing(1, 2) == ["(the whole value)"]
    assert len(regress.differing({str(i): i for i in range(40)}, {str(i): -1 - i for i in range(40)})) == 12


# ------------------------------------------------------------------------------------------------
# check, on records made here
# ------------------------------------------------------------------------------------------------
def _record(tier="full") -> dict:
    entries = [dict(case=c["id"], run=c["run"], body=c["body"], row=r, condition=k, seed=s, trace_sha256="0" * 64,
                    recorded="0" * 64, trace_equal=True, record_equal=True)
               for c in regress.cases_of(tier) for r, k, s in regress.episodes_of(c)]
    return dict(schema=regress.SCHEMA, tier=tier, cases=entries, matched=len(entries), of=len(entries),
                distinct=regress.distinct(entries), sources=regress.sources())


def test_check_reads_the_record_against_todays_code(tmp_path, capsys):
    path = tmp_path / "regress.json"
    s2io.write_json(path, _record())
    assert (_record()["of"], _record()["distinct"]) == (256, 248)
    assert regress.check(path) == []
    assert regress.main(["check", "--record", str(path)]) == 0
    assert "256 of 256 episode runs matched, 248 distinct committed episodes" in capsys.readouterr().out

    for key in regress.SOURCE_KEYS:  # each of the four hashes is read
        doc = _record()
        doc["sources"][key] = "f" * 64
        s2io.write_json(path, doc)
        diff = regress.check(path)
        assert len(diff) == 2 and diff[0].startswith(f"{key}: ") and "regress run --tier full" in diff[1]
    assert regress.main(["check", "--record", str(path)]) == 1

    s2io.write_json(path, _record("fast"))  # a fast record is not the full tier's
    assert any("tier is 'fast'" in line for line in regress.check(path))

    doc = _record()  # an episode that did not match, counted or not
    doc["cases"][5]["record_equal"] = False
    s2io.write_json(path, doc)
    assert any(line.startswith("matched 256 of 256 (255 of 256") for line in regress.check(path))
    doc["matched"] = 255
    s2io.write_json(path, doc)
    assert any(line.startswith("matched 255 of 256") for line in regress.check(path))

    for wrong in (256, None):  # the count of distinct committed episodes is the listed ones', not the comparisons'
        doc = _record()
        doc["distinct"] = wrong
        if wrong is None:
            del doc["distinct"]
        s2io.write_json(path, doc)
        diff = regress.check(path)
        assert len(diff) == 2 and diff[0] == f"distinct is {wrong!r}; the record lists 248 different committed episodes"

    doc = _record()  # a record of other cases than today's
    del doc["cases"][-1]
    doc["matched"] = doc["of"] = 255
    doc["distinct"] = 247
    s2io.write_json(path, doc)
    assert any("lists 255 episodes, today's full tier names 256" in line for line in regress.check(path))

    assert "no readable record" in regress.check(tmp_path / "none.json")[0]
    path.write_text("[1, 2]")
    assert regress.check(path) == [f"{path}: not a record"]


# ------------------------------------------------------------------------------------------------
# run: what it refuses, what it keeps, what it writes (the evals replaced by a stub)
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """``run`` with a clean checkout, nothing missing and ``run_case`` replaced by a stub that matches everything;
    ``calls`` lists the cases it ran."""
    calls, state = [], dict(status=[], missing=[], equal=True)

    def run_case(case, scratch, key, log=print):
        calls.append(case["id"])
        eps = [dict(case=case["id"], run=case["run"], body=case["body"], row=r, condition=c, seed=int(s),
                    trace_sha256="a" * 64, recorded="a" * 64, trace_equal=True, record_equal=state["equal"])
               for r, c, s in regress.episodes_of(case)]
        doc = dict(key=key, command=["python", *regress.command(case, "<scratch>")], returncode=0, started_at="t0",
                   ended_at="t1", wall_s=1.5, episodes=eps, matched=sum(regress.matched(e) for e in eps), of=len(eps),
                   commit="c" * 40, log="x.log")
        s2io.write_json(Path(scratch) / "cases" / f"{case['id']}.json", doc)
        return doc

    monkeypatch.setattr(regress, "run_case", run_case)
    monkeypatch.setattr(regress, "git_status", lambda own=(): list(state["status"]))
    monkeypatch.setattr(regress, "missing_inputs", lambda tier="full": list(state["missing"]))
    return calls, state, tmp_path


@needs_git
def test_run_writes_the_record_and_keeps_finished_cases(stubbed, monkeypatch):
    calls, state, tmp = stubbed
    lines = []
    doc = regress.run("fast", record=tmp / "r.json", scratch=tmp / "scratch", log=lines.append)
    assert calls == [c["id"] for c in regress.cases_of("fast")]
    assert (doc["matched"], doc["of"], doc["distinct"]) == (8, 8, 8)
    assert lines[-1].startswith("fast tier: 8 of 8 episode runs match their committed trace and episode record "
                                "(8 episode runs, 8 distinct; ")
    saved = json.loads((tmp / "r.json").read_text())
    assert saved == json.loads(json.dumps(s2io.plain(doc)))
    assert set(saved) >= {"tier", "cases", "matched", "of", "distinct", "sources", "versions", "blas_threads", "commit",
                          "dirty", "at", "rules", "commands", "regress_sha256"}
    assert saved["rules"] == dict(regress=regress.RULE, clean=regress.CLEAN_RULE, resume=regress.RESUME_RULE,
                                  count=regress.COUNT_RULE, check=regress.CHECK_RULE)
    assert saved["tier"] == "fast" and saved["blas_threads"] == 1 and saved["dirty"] is False
    assert saved["sources"] == regress.sources() and set(saved["sources"]) == set(regress.SOURCE_KEYS)
    assert saved["versions"] == panels.library_versions() and len(saved["commit"]) == 40
    assert set(saved["cases"][0]) >= {"run", "body", "row", "condition", "seed", "trace_sha256", "recorded",
                                      "trace_equal", "record_equal"}
    assert saved["environment"][regress.SPEED_RUN_ENV] is None and saved["environment"]["MUJOCO_GL"] == "disable"
    assert regress.main(["check", "--record", str(tmp / "r.json")]) == 1  # a fast record never passes the check
    assert not (tmp / "scratch" / ".lock").exists()

    # the same run again: every case is kept; the full tier then runs only its own cases
    del calls[:]
    again = regress.run("fast", scratch=tmp / "scratch", log=lines.append)
    assert calls == [] and again["cases"] == doc["cases"] and any("kept from" in line for line in lines)
    full = regress.run("full", scratch=tmp / "scratch", log=lines.append)
    assert calls == [c["id"] for c in regress.CASES if c["tier"] == "full"]
    assert (full["matched"], full["of"], full["distinct"]) == (256, 256, 248)
    assert lines[-1].startswith("full tier: 256 of 256 episode runs match their committed trace and episode record "
                                "(256 episode runs, 248 distinct; ")
    s2io.write_json(tmp / "full.json", full)
    assert regress.check(tmp / "full.json") == []  # the stub's full record is current: the sources are today's

    # a kept case whose eval exited non-zero (killed, or failed) runs again; a kept result without an exit code too
    kept = tmp / "scratch" / "cases" / "fast-75-M-H-10.json"
    for change in (dict(returncode=3), dict(returncode=-9), dict(returncode=None), None):
        was = json.loads(kept.read_text())
        assert was["returncode"] == 0
        s2io.write_json(kept, dict(was, **change) if change else {k: v for k, v in was.items() if k != "returncode"})
        del calls[:]
        again = regress.run("fast", scratch=tmp / "scratch", log=lines.append)
        assert calls == ["fast-75-M-H-10"] and (again["matched"], again["of"]) == (8, 8), change
        assert "fast-60-N-T-10-10ms: kept from" in lines[-2] and json.loads(kept.read_text())["returncode"] == 0

    # a kept case of other sources, or of another regress.py, runs again
    del calls[:]
    monkeypatch.setattr(regress, "_own_sha256", lambda: "e" * 64)
    regress.run("fast", scratch=tmp / "scratch", log=lines.append)
    assert calls == [c["id"] for c in regress.cases_of("fast")]
    del calls[:]
    src = dict(regress.sources(), stage1_sources_sha256="d" * 64)
    monkeypatch.setattr(regress, "sources", lambda: dict(src))
    assert regress.run("fast", scratch=tmp / "scratch", log=lines.append)["sources"] == src
    assert calls == [c["id"] for c in regress.cases_of("fast")]

    # a difference is reported, not raised; the CLI then exits 1
    state["equal"] = False
    monkeypatch.setattr(regress, "_own_sha256", lambda: "b" * 64)
    assert regress.run("fast", scratch=tmp / "scratch", log=lines.append)["matched"] == 0
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    assert regress.main(["run", "--tier", "fast", "--scratch", str(tmp / "scratch")]) == 1
    state["equal"] = True
    monkeypatch.setattr(regress, "_own_sha256", lambda: "9" * 64)
    assert regress.main(["run", "--tier", "fast", "--scratch", str(tmp / "scratch")]) == 0


@needs_git
def test_run_refuses(stubbed, monkeypatch, capsys):
    calls, state, tmp = stubbed
    state["status"] = [" M hardware/trex_form_study/designlab/stage1/control.py"]
    with pytest.raises(regress.RegressError, match="uncommitted changes.*control.py"):
        regress.run("fast", scratch=tmp / "scratch", log=lambda line: None)
    state["status"], state["missing"] = [], ["60-N-T-10: no round-two model files"]
    with pytest.raises(regress.RegressError, match="no round-two model files"):
        regress.run("fast", scratch=tmp / "scratch", log=lambda line: None)
    for v in s2io.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    assert regress.main(["run", "--scratch", str(tmp / "scratch")]) == 2 and "refused" in capsys.readouterr().err
    state["missing"] = []
    with pytest.raises(regress.RegressError, match="git tracks files below"):  # a committed run directory as scratch
        regress.run("fast", scratch=ROOT / regress.STAGE1_RUN, log=lambda line: None)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")  # the CLI refuses an inherited thread count
    assert regress.main(["run", "--scratch", str(tmp / "scratch")]) == 2
    assert calls == [] and not (tmp / "scratch" / "cases").exists()

    # the checkout changes under the run: refused at the end, no record written
    for v in s2io.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    seen = []

    def status(own=()):
        seen.append(1)
        return [] if len(seen) == 1 else ["?? hardware/trex_form_study/stage1/runs/2026-11-08/x.json"]

    monkeypatch.setattr(regress, "git_status", status)
    with pytest.raises(regress.RegressError, match="changed during the run"):
        regress.run("fast", record=tmp / "r.json", scratch=tmp / "scratch", log=lambda line: None)
    assert not (tmp / "r.json").exists()


def test_git_status_leaves_out_the_record_only(tmp_path, monkeypatch):
    top = tmp_path / "repo"
    lines = ["?? hardware/trex_form_study/stage2/regress/regress.json", " M hardware/trex_form_study/a.py",
             "R  old.json -> hardware/trex_form_study/stage2/regress/regress.json"]

    def git(*args):
        return str(top) + "\n" if args[0] == "rev-parse" else "\n".join(lines) + "\n"

    monkeypatch.setattr(regress, "_git", git)
    own = top / "hardware" / "trex_form_study" / "stage2" / "regress" / "regress.json"
    assert regress.git_status() == lines
    assert regress.git_status(own=(own, None)) == lines[1:]


# ------------------------------------------------------------------------------------------------
# The real thing
# ------------------------------------------------------------------------------------------------
@slow
@needs_git
def test_regress_fast(tmp_path):
    """``regress run --tier fast`` into a temporary record: 8 of 8 (a hold, a stop event, a speed-change plan and mode
    R1 without terms on 75-M-H-10; the 10 ms delay queue on two turns of 60-N-T-10)."""
    missing = regress.missing_inputs("fast")
    if missing:
        pytest.skip("; ".join(missing))
    doc = regress.run("fast", record=tmp_path / "regress.json", scratch=tmp_path / "scratch")
    assert (doc["matched"], doc["of"]) == (8, 8)
    assert all(e["trace_equal"] and e["record_equal"] and e["trace_sha256"] == e["recorded"] for e in doc["cases"])
    assert all(c["returncode"] == 0 for c in doc["commands"])
    assert json.loads((tmp_path / "regress.json").read_text())["matched"] == 8
    assert not (tmp_path / "scratch" / "work" / "fast-75-M-H-10").exists()  # removed once everything matched


# ------------------------------------------------------------------------------------------------
# The Stage 1 edit the regression guards: control.Track.add_periodic's optional window_strides key (D18)
# ------------------------------------------------------------------------------------------------
def _tiled(ref: Reference, k: int, key: bool) -> Reference:
    """``ref``'s stride repeated ``k`` times as one window (every per-frame array tiled, the time axis continued),
    with or without the ``window_strides`` key."""
    n = ref.n
    assert not any(len(ref.arrays[x]) == n for x in ("q_home", "joints", "stride_motion"))
    arrays = {name: (np.concatenate([a] * k) if a.ndim and len(a) == n else a) for name, a in ref.arrays.items()}
    arrays["t"] = np.concatenate([ref.arrays["t"] + i * ref.period for i in range(k)])
    if key:
        arrays["window_strides"] = np.array(k)
    return Reference(f"{ref.name}_x{k}", arrays)


def _same(a, b) -> None:
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    else:
        assert a == b


@needs_plant
def test_window_strides_key(monkeypatch):
    """A reference whose file holds a window of K strides and says so (``window_strides``) gives a segment whose
    feedback quantities are the one-stride reference's frame for frame (``info["period"]`` is one stride), whose
    ``period`` is still the window (it drives the wrap) and whose joint targets are the one-stride track's at the
    same times; an episode on it counts strides, not windows.  The stride is ``(n // K) * step``, the one-stride
    reference's period bit for bit for every K (K = 2 and K = 3 here; the window's period over K is not, see the brisk
    walk below); a K that is under 1 or does not divide the window's frames is refused.  Without the key nothing
    changes: a Stage 1 reference's segment is what ``add_periodic`` built before the edit, and a window is read as
    one long stride, as it was."""
    p = plant.build(BODY, "hard", settle=False)
    ref = Reference.load(RUN / BODY / "refs" / "nominal.npz")
    assert "window_strides" not in ref.arrays
    om = control.omega0(p, ref.q_home)
    asked, frame_info = [], control.frame_info

    def spy(*args, **kw):  # what add_periodic asks frame_info for
        asked.append((kw["period"], kw.get("frames_per_stride")))
        return frame_info(*args, **kw)

    monkeypatch.setattr(control, "frame_info", spy)
    t1 = control.Track(p, om).add_periodic(ref, 4.0)
    one = t1.segments[0]
    assert asked == [(ref.period, ref.n)] and type(asked[0][1]) is int

    # without the key: the segment of before the edit (frame_info called with the reference's own period and frames)
    want = frame_info(p, ref.q, ref.qd, ref.support, ref.zmp_mb, ref.normal, om, periodic=True, period=ref.period,
                      frames_per_stride=ref.n)
    assert set(one.info) == set(want)
    for k, v in want.items():
        _same(one.info[k], v)
    assert (one.kind, one.period, one.n, one.step, one.duration) == ("periodic", ref.period, ref.n, ref.step,
                                                                    4.0 * ref.period)
    assert one.stride_s == ref.period and set(one.info["period"]) == {ref.period}
    assert control.Segment("hold", "stand", 0.0, 1.0, one.S[:1], {}).stride_s == 0.0  # other segments: no stride
    held = control.Track(p, om).add_hold(ref.q[0], ref.tau[0], ref.zmp_mb[0], 1.0)
    assert [s.stride_s for s in held.segments] == [0.0]

    # with the key: two strides in one window
    win = _tiled(ref, 2, key=True)
    t2 = control.Track(p, om).add_periodic(win, 2.0)
    two = t2.segments[0]
    assert asked[-1] == (ref.period, ref.n) and type(asked[-1][1]) is int  # one stride's period and frames
    assert two.period == win.period == 2 * ref.period and two.n == 2 * ref.n and two.step == ref.step
    assert two.duration == one.duration and t2.t_end == t1.t_end
    assert two.stride_s == ref.period
    assert set(two.info["period"]) == {ref.period} and len(two.info["period"]) == 2 * ref.n
    assert set(two.info) == set(one.info)
    for k, v in one.info.items():  # every feedback quantity is the one-stride reference's, frame for frame
        _same(two.info[k], np.concatenate([v, v]) if isinstance(v, np.ndarray) else v + v)
    assert np.array_equal(two.S, np.concatenate([one.S, one.S]))
    times = np.r_[np.linspace(0.0, t1.t_end, 1201, endpoint=False), np.arange(4 * ref.n) * ref.step]
    assert times.max() < t1.t_end
    for t in times:
        assert np.array_equal(t2.joint(float(t)), t1.joint(float(t))), t

    # the same window without the key is one long stride, as before the edit
    t3 = control.Track(p, om).add_periodic(_tiled(ref, 2, key=False), 2.0)
    long = t3.segments[0]
    assert asked[-1] == (2 * ref.period, 2 * ref.n)
    assert long.period == long.stride_s == 2 * ref.period and set(long.info["period"]) == {2 * ref.period}
    assert np.array_equal(long.S, two.S)

    # an episode counts the strides of the commanded gait: with the key, a window of two strides counts two
    a = control.run(control.Tracker(p, t1, "R0"), 0.6)
    qa = p.d.qpos.copy()
    b = control.run(control.Tracker(p, t2, "R0"), 0.6)
    qb = p.d.qpos.copy()
    c = control.run(control.Tracker(p, t3, "R0"), 0.6)
    assert np.array_equal(qa, qb) and np.array_equal(qa, p.d.qpos)
    assert b["strides"] == a["strides"] and b["reference_strides"] == a["reference_strides"] and a["strides"] > 0
    assert c["strides"] == a["strides"] / 2 and c["reference_strides"] == a["reference_strides"] / 2

    # refused: a key under 1, or one that does not divide the window's frames (a stride is a whole number of frames)
    assert ref.n % 3 and (2 * ref.n) % 7
    n_asked = len(asked)
    for arrays, k in ((ref.arrays, 3), (_tiled(ref, 2, key=False).arrays, 7), (ref.arrays, 0), (ref.arrays, -2)):
        bad = Reference("bad", dict(arrays, window_strides=np.array(k)))
        track = control.Track(p, om)
        with pytest.raises(ValueError, match=f"window_strides {k} does not divide its {bad.n} frames"):
            track.add_periodic(bad, 1.0)
        assert track.segments == [] and track.t_end == 0.0
    assert len(asked) == n_asked  # refused before anything is built
    whole = Reference("whole", dict(ref.arrays, window_strides=np.array(1)))  # the key at 1 is the default
    assert control.Track(p, om).add_periodic(whole, 1.0).segments[0].stride_s == ref.period
    assert asked[-1] == (ref.period, ref.n)

    # K = 3, on the brisk walk: there (3 * n * step) / 3 is one ulp off n * step, so the stride has to come from the
    # frames, (n // K) * step, to be the one-stride period bit for bit, and an episode's strides with it
    brisk = Reference.load(RUN / BODY / "refs" / "brisk.npz")
    win3 = _tiled(brisk, 3, key=True)
    assert win3.period / 3 != brisk.period  # what makes this reference tell the two expressions apart
    om_b = control.omega0(p, brisk.q_home)
    b1 = control.Track(p, om_b).add_periodic(brisk, 3.0)
    b3 = control.Track(p, om_b).add_periodic(win3, 1.0)
    single, three = b1.segments[0], b3.segments[0]
    assert asked[-1] == (brisk.period, brisk.n) and type(asked[-1][0]) is float and type(asked[-1][1]) is int
    assert three.stride_s == single.stride_s == brisk.period and set(three.info["period"]) == {brisk.period}
    assert three.period == win3.period == (3 * brisk.n) * brisk.step and three.n == 3 * brisk.n
    assert three.step == brisk.step and three.duration == win3.period
    assert set(three.info) == set(single.info)
    for k, v in single.info.items():
        _same(three.info[k], np.concatenate([v, v, v]) if isinstance(v, np.ndarray) else v + v + v)
    assert np.array_equal(three.S, np.concatenate([single.S, single.S, single.S]))
    end = min(b1.t_end, b3.t_end)  # 3 * (n * step) and (3 * n) * step may differ in the last bit
    for t in np.r_[np.linspace(0.0, end, 901, endpoint=False), np.arange(3 * brisk.n) * brisk.step]:
        assert np.array_equal(b3.joint(float(t)), b1.joint(float(t))), t
    a = control.run(control.Tracker(p, b1, "R0"), 0.6)
    qa = p.d.qpos.copy()
    b = control.run(control.Tracker(p, b3, "R0"), 0.6)
    assert np.array_equal(qa, p.d.qpos)
    assert b["strides"] == a["strides"] and b["reference_strides"] == a["reference_strides"] and a["strides"] > 0
