"""PS build: the edits of tests/test_waves_bs.py (each old string asserted unique), run once.  The RR-1 / I1 tests and
the stub scorer are LB's (S2/r10round/long_blind/tools/tests/test_waves_bs.py l125-147, l273-285, l471-505)."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''decision (PR-M2, lesson P4); the REFUSED note naming --preflight-only, not the attempt log (PR-M3).
''', '''decision (PR-M2, lesson P4); the REFUSED note naming --preflight-only, not the attempt log (PR-M3).
PS (LB's tests, ported): RR-1 (a relaunch after ALL_DECIDED keeps the first snapshot and ALL_DECIDED) and I1 (the
atimes at each attempt's exit, before CHECK and the re-arm); the scorer the APPROVED pin names is a stub here
(BS_SCORER; the real one is the criteria lane's); the queue's intervention is critic_phase; the real root is PS.
''')
rep('''QUEUE = [("control", 2101), ("critic_blind_tail3", 2101), ("control", 2111), ("critic_blind_tail3", 2111),
         ("control", 2121), ("critic_blind_tail3", 2121)]''', '''QUEUE = [("control", 2101), ("critic_phase", 2101), ("control", 2111), ("critic_phase", 2111),
         ("control", 2121), ("critic_phase", 2121)]''')
rep('''    root = tmp_path / "bs"
    root.mkdir()
    (root / "APPROVED").write_text(json.dumps({"tool_sha256": {"score_bs.py": _sha(TOOLS / "score_bs.py")}}))
    (root / "criteria.txt").write_text("text\\n")
    (root / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "critic_blind_tail3",''',
    '''    root = tmp_path / "bs"
    root.mkdir()
    scorer = tmp_path / "stub_score_bs.py"                 # the criteria lane's scorer is not in this build (LB's)
    scorer.write_text("# stub scorer\\n")
    (root / "APPROVED").write_text(json.dumps({"tool_sha256": {"score_bs.py": _sha(scorer)}}))
    (root / "criteria.txt").write_text("text\\n")
    (root / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "critic_phase",''')
rep('''           "BS_POLL_SECONDS": "0.2", "BS_LAUNCH_DISK_BYTES": "0", "BS_STUB_S": str(C.S)}   # BS2: S from bs_common''',
    '''           "BS_POLL_SECONDS": "0.2", "BS_LAUNCH_DISK_BYTES": "0", "BS_STUB_S": str(C.S),   # BS2: S from bs_common
           "BS_SCORER": str(scorer)}''')
rep('''        assert r.returncode == 2 and "score_bs.py" in r.stderr, r.stderr
        assert _calls(bs) == [] and not (bs["root"] / "runs").exists()
''', '''        assert r.returncode == 2 and "score_bs.py" in r.stderr, r.stderr
        assert _calls(bs) == [] and not (bs["root"] / "runs").exists()
    # the real scorer path when no test override is given (absent in this build): refused, nothing started (LB's)
    (bs["root"] / "APPROVED").write_text(json.dumps({"tool_sha256": {"score_bs.py": _sha(bs["scorer"])}}))
    env = {k: v for k, v in bs["env"].items() if k != "BS_SCORER"}
    r = subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=60, cwd="/tmp")
    if not (TOOLS / "score_bs.py").exists():
        assert r.returncode == 2 and "score_bs.py" in r.stderr and _calls(bs) == []
''')
rep('''    bad = {"schema": C.SCREEN_SCHEMA, "intervention": "critic_blind_tail3",
           "queue": [["control", 3000 + i] for i in range(5)] + [["critic_blind_tail3", 4000 + i] for i in range(5)]}''',
    '''    bad = {"schema": C.SCREEN_SCHEMA, "intervention": "critic_phase",
           "queue": [["control", 3000 + i] for i in range(5)] + [["critic_phase", 4000 + i] for i in range(4)]}''')
rep('''    (root / "SCREEN.json").write_text(json.dumps({**bad, "queue": [["control", 1], ["gae_lambda_1", 2]]}))
    assert _run(bs).returncode == 2''', '''    (root / "SCREEN.json").write_text(json.dumps({**bad, "queue": [["control", 1], ["gae_lambda_1", 2]]}))
    assert _run(bs).returncode == 2
    (root / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "critic_blind_tail3",
                                                  "queue": [["control", 1], ["critic_blind_tail3", 2]]}))
    assert _run(bs).returncode == 2                     # a BS2 arm is not a PS intervention''')
rep('''def test_real_root_is_bs2():
    text = RUNNER.read_text()
    assert "BS_REAL=$S2/r10round/bias_screen2\\n" in text and C.BS.endswith("/r10round/bias_screen2")''',
    '''def test_real_root_is_ps():
    text = RUNNER.read_text()
    assert "BS_REAL=/home/user/work/phase_screen\\n" in text and C.BS == "/home/user/work/phase_screen"


def test_rr1_a_relaunch_after_all_decided_keeps_the_first_snapshot(bs):
    """RR-1 (LB's test): the second run of the runner (everything decided) writes its snapshot to .relaunch1.tsv; the
    first snapshot and ALL_DECIDED are unchanged."""
    _plan(bs, {})
    assert _run(bs).returncode == 0
    logs = bs["root"] / "logs"
    first_tsv, first_marker = (logs / "atimes_at_ALL_DECIDED.tsv").read_bytes(), (logs / "ALL_DECIDED").read_bytes()
    time.sleep(1.1)
    r = _run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (logs / "atimes_at_ALL_DECIDED.tsv").read_bytes() == first_tsv
    assert (logs / "ALL_DECIDED").read_bytes() == first_marker
    assert (logs / "atimes_at_ALL_DECIDED.relaunch1.tsv").exists()
    assert "atimes_at_ALL_DECIDED" not in (logs / "atimes_at_ALL_DECIDED.relaunch1.tsv").read_text()
    assert "ALL_DECIDED kept" in (logs / "run_waves_bs.log").read_text()


def test_i1_atimes_are_taken_at_exit_before_check(bs):
    """I1 (LB's test): logs/atimes_at_exit/<name>.attempt<k>.tsv lists the attempt's files right after it exits
    (lstat), before CHECK wrote check.json and before the re-arm; a crash's snapshot is taken before the rename."""
    first, second = _names(2)
    _plan(bs, {first: ["crash"]})
    r = _run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    snap = bs["root"] / "logs" / "atimes_at_exit"
    a1 = (snap / f"{first}.attempt1.tsv").read_text().splitlines()
    assert a1[0].split("\\t") == ["path", "atime_ns", "mtime_ns", "ctime_ns", "atime_after_mtime"]
    assert any(f"/runs/{first}/run_meta.json" in ln for ln in a1) and not any("check.json" in ln for ln in a1)
    assert any(ln.split("\\t")[0].endswith(f"logs/{first}.attempt1.log") for ln in a1[1:])
    a2 = (snap / f"{first}.attempt2.tsv").read_text()
    assert f"/runs/{first}/run_meta.json" in a2 and "check.json" not in a2
    s2 = (snap / f"{second}.attempt1.tsv").read_text().splitlines()
    rows = {ln.split("\\t")[0].rsplit("/", 1)[-1]: ln.split("\\t") for ln in s2[1:]}
    assert int(rows["run_meta.json"][1]) != int(rows["run_meta.json"][2]) - 1     # not yet re-armed at exit''')
p.write_text(s)
print("ok")
