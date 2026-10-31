"""Tests of run_waves_bs.sh with a stub harness (ported from the ECS runner's tests): all COUNTED then read only after
all are decided; a crash renamed and rerun; an early crash halts and is not counted; a post-run check failure is VOID;
three crashes to lost.json; a CHECK failure VOID; a refusal halts; a half-way folder at relaunch; the disk pause; the
launch floor; the lock; SCREEN.json's queue (and its refusal when invalid, more than 9 or missing); 4 slots.
BS2 review fixes: the refusal without APPROVED's score_bs.py pin (PR-M4); the atime re-arm of a decided continuation's
files and the snapshot logs/atimes_at_ALL_DECIDED.tsv before the full reads, which shows a read made after the
decision (PR-M2, lesson P4); the REFUSED note naming --preflight-only, not the attempt log (PR-M3).
PS (LB's tests, ported): RR-1 (a relaunch after ALL_DECIDED keeps the first snapshot and ALL_DECIDED) and I1 (the
atimes at each attempt's exit, before CHECK and the re-arm); the scorer the APPROVED pin names is a stub here
(BS_SCORER; the real one is the criteria lane's); the queue's intervention is critic_phase; the real root is PS.

The runner runs in its test mode (BS_WAVES_TEST=1 with a temporary BS_ROOT, never the real <BS>); every command it
starts is a stub written here.  Nothing trains, nothing simulates.
    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <BS>/tools/tests/test_waves_bs.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]
RUNNER = TOOLS / "run_waves_bs.sh"
_spec = importlib.util.spec_from_file_location("bs_common", str(TOOLS / "bs_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

STUB_RUN = r'''
import json, os, sys, time
a = sys.argv[1:]
arm, seed = a[a.index("--arm") + 1], a[a.index("--seed") + 1]
root = os.environ["BS_ROOT"]
name = f"{arm}_{seed}"
S = int(os.environ["BS_STUB_S"])
os.makedirs(f"{root}/attempts", exist_ok=True)
with open(f"{root}/attempts/{name}", "a") as fh:
    fh.write("x\n")
n = sum(1 for _ in open(f"{root}/attempts/{name}"))
plan = json.load(open(f"{root}/plan.json")).get(name, [])
what = plan[n - 1] if n <= len(plan) else "end"
with open(f"{root}/calls.log", "a") as fh:
    fh.write(f"run {name} {what}\n")
print(f"stub attempt {n} of {name}: {what}", flush=True)
if what.startswith("vslow"):
    time.sleep(4.0)
    what = what[5:] or "end"
if what.startswith("slow"):
    time.sleep(1.0)
    what = what[4:] or "end"
if what == "refuse":
    sys.exit(2)
run = f"{root}/runs/{name}"
os.makedirs(f"{run}/01_stance/models")
meta = {"status": "running", "exit": None, "stop_reason": None}
if what in ("end", "collapse", "checkfail"):
    meta = {"status": "done", "exit": 0, "stop_reason": "collapse" if what == "collapse" else "end"}
    if what == "checkfail":
        meta["stub_check"] = "fail"
elif what == "postfail":
    meta = {"status": "failed", "exit": 5, "stop_reason": "end"}
elif what in ("crash", "earlycrash"):
    meta.update(status="crashed", exit=1, last_step=S + (8192 if what == "earlycrash" else 3 * 131072))
elif what == "kill":
    meta["heartbeat"] = {"rollout_end": S + 2 * 131072}
json.dump(meta, open(f"{run}/run_meta.json", "w"))
with open(f"{run}/01_stance/policy_probe.npz", "w") as fh:
    fh.write("probe")
os.makedirs(f"{root}/tmp/{name}/eval_stage1_x", exist_ok=True)
with open(f"{root}/tmp/{name}/eval_stage1_x/evaluations.npz", "w") as fh:
    fh.write("returns")
if what in ("crash", "earlycrash"):
    sys.exit(1)
if what == "kill":
    os._exit(137)
if what == "postfail":
    sys.exit(5)
sys.exit(0)
'''

STUB_TOOL = r'''
import json, os, sys
root = os.environ["BS_ROOT"]
kind, args = sys.argv[1], sys.argv[2:]
with open(f"{root}/calls.log", "a") as fh:
    fh.write(kind + " " + " ".join(args) + "\n")
if kind == "check":
    meta = json.load(open(os.path.join(args[0], "run_meta.json")))
    if meta.get("stub_check") == "fail" or meta.get("status") != "done" or meta.get("exit") != 0:
        sys.exit(4)
    json.dump({"pass": True}, open(os.path.join(args[0], "check.json"), "w"))
sys.exit(0)
'''

QUEUE = [("control", 2101), ("critic_phase", 2101), ("control", 2111), ("critic_phase", 2111),
         ("control", 2121), ("critic_phase", 2121)]


def _names(n):
    return [C.run_name(a, s) for a, s in QUEUE[:n]]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atimes(bs):
    """logs/atimes_at_ALL_DECIDED.tsv as {path: row dict}."""
    lines = (bs["root"] / "logs" / "atimes_at_ALL_DECIDED.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    return {r[0]: dict(zip(head, r, strict=True)) for r in (ln.split("\t") for ln in lines[1:])}


@pytest.fixture()
def bs(tmp_path):
    root = tmp_path / "bs"
    root.mkdir()
    scorer = tmp_path / "stub_score_bs.py"                 # the criteria lane's scorer is not in this build (LB's)
    scorer.write_text("# stub scorer\n")
    (root / "APPROVED").write_text(json.dumps({"tool_sha256": {"score_bs.py": _sha(scorer)}}))
    (root / "criteria.txt").write_text("text\n")
    (root / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "critic_phase",
                                                  "queue": [list(q) for q in QUEUE[:4]]}))
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "stub_run_bs.py").write_text(STUB_RUN)
    (stubs / "stub_tool.py").write_text(STUB_TOOL)
    py = sys.executable
    tool = f"{py} -I -B {stubs / 'stub_tool.py'}"
    env = {**os.environ, "BS_WAVES_TEST": "1", "BS_ROOT": str(root),
           "BS_RUN_CMD": f"{py} -I -B {stubs / 'stub_run_bs.py'}", "BS_CHECK_CMD": f"{tool} check",
           "BS_READ_CMD": f"{tool} read", "BS_ORPHAN_PATTERN": f"{stubs / 'stub_run_bs.py'} --arm",
           "BS_POLL_SECONDS": "0.2", "BS_LAUNCH_DISK_BYTES": "0", "BS_STUB_S": str(C.S),   # BS2: S from bs_common
           "BS_SCORER": str(scorer)}
    return {"root": root, "env": env, "stubs": stubs, "scorer": scorer}


def _plan(bs, plan):
    (bs["root"] / "plan.json").write_text(json.dumps(plan))


def _run(bs, *args, timeout=120, **env_over):
    env = {**bs["env"], **env_over}
    return subprocess.run(["bash", str(RUNNER), *args], env=env, capture_output=True, text=True, timeout=timeout,
                          cwd="/tmp")


def _decided(bs):
    p = bs["root"] / "logs" / "decided.tsv"
    return {ln.split("\t")[1]: ln.split("\t")[2] for ln in p.read_text().splitlines()} if p.exists() else {}


def _calls(bs):
    p = bs["root"] / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


def _crash_rows(bs):
    p = bs["root"] / "logs" / "crash_ledger.tsv"
    return [ln.split("\t") for ln in p.read_text().splitlines()] if p.exists() else []


def test_plan_reads_screen_queue(bs):
    r = _run(bs, "--plan")
    assert r.returncode == 0, r.stderr
    assert "slots 4 at nice 10" in r.stdout
    for name in _names(4):
        arm, seed = name.rsplit("_", 1)
        assert f"--arm {arm} --seed {seed}" in r.stdout
    assert f"S {C.S}, E {C.E}" in r.stdout and "S 2097152, E 3538944" in r.stdout


def test_all_counted_four_slots_then_read_after_all_decided(bs):
    _plan(bs, {})
    r = _run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    names = _names(4)
    assert _decided(bs) == {n: "COUNTED" for n in names}
    log = (bs["root"] / "logs" / "run_waves_bs.log").read_text().splitlines()
    assert [ln.split("] start ")[1].split()[0] for ln in log if "] start " in ln] == names
    calls = _calls(bs)
    last_check = max(i for i, c in enumerate(calls) if c.startswith("check "))
    reads = [i for i, c in enumerate(calls) if c.startswith("read ")]
    assert len(reads) == 4 and min(reads) > last_check
    assert (bs["root"] / "logs" / "ALL_DECIDED").exists() and not (bs["root"] / "logs" / "HALTED").exists()
    assert (bs["root"] / "logs" / "run_waves_bs.exit").read_text().split()[0] == "0"


def test_four_run_at_once(bs):
    _plan(bs, {n: ["slowend"] for n in _names(4)})
    r = _run(bs)
    assert r.returncode == 0
    log = (bs["root"] / "logs" / "run_waves_bs.log").read_text().splitlines()
    first_exit = next(i for i, ln in enumerate(log) if " exited " in ln)
    assert sum("] start " in ln for ln in log[:first_exit]) == 4


def test_crash_renamed_and_rerun(bs):
    first, second = _names(2)
    _plan(bs, {first: ["crash"], second: ["kill"]})
    r = _run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (bs["root"] / "runs" / f"{first}_crashed1").is_dir() and (bs["root"] / "runs" / first).is_dir()
    assert sorted((row[1], row[4]) for row in _crash_rows(bs)) == sorted([(first, "1"), (second, "1")])
    assert set(_decided(bs).values()) == {"COUNTED"}
    assert (bs["root"] / "logs" / f"{first}.attempt2.log").read_text().startswith(f"stub attempt 2 of {first}: end")


def test_early_crash_halts_not_counted(bs):
    first = _names(1)[0]
    _plan(bs, {first: ["earlycrash"]})
    r = _run(bs, BS_SLOTS="1")
    assert r.returncode == 2
    row = _crash_rows(bs)[0]
    assert row[4] == "0" and "EARLY CRASH" in row[3]
    halted = bs["root"] / "logs" / "HALTED"
    assert halted.exists()
    assert _run(bs).returncode == 2
    halted.unlink()
    assert _run(bs).returncode == 0 and set(_decided(bs).values()) == {"COUNTED"}


def test_post_run_failure_void_and_check_failure_void(bs):
    a, b = _names(2)
    _plan(bs, {a: ["postfail"], b: ["checkfail"]})
    r = _run(bs, BS_SLOTS="1")
    assert r.returncode == 4
    assert _decided(bs) == {a: "VOID"} and (bs["root"] / "runs" / f"{a}_void1").is_dir() and _crash_rows(bs) == []
    assert not any(c.startswith("read") for c in _calls(bs))
    (bs["root"] / "logs" / "HALTED").unlink()
    assert _run(bs, BS_SLOTS="1").returncode == 4          # the VOID line still refuses the relaunch


def test_three_crashes_lost(bs):
    first = _names(1)[0]
    _plan(bs, {first: ["crash", "crash", "crash"]})
    r = _run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    lost = json.loads((bs["root"] / "lost.json").read_text())
    assert lost["schema"] == C.LOST_SCHEMA and lost["lost"][first]["cause"] == "crashes"
    assert _decided(bs)[first] == "LOST"
    assert f"read {bs['root']}/runs/{first}" not in _calls(bs)


def test_refusal_halts(bs):
    first = _names(1)[0]
    _plan(bs, {first: ["refuse"]})
    r = _run(bs, BS_SLOTS="1")
    assert r.returncode == 2 and _decided(bs) == {}
    assert "refused " + first in (bs["root"] / "logs" / "HALTED").read_text()
    # PR-M3: the note names the read-only preflight re-run for that entry, never the attempt log
    note = next(ln for ln in (bs["root"] / "logs" / "run_waves_bs.log").read_text().splitlines() if "REFUSED" in ln)
    arm, seed = first.rsplit("_", 1)
    assert f"--arm {arm} --seed {seed} --preflight-only" in note and "never the attempt log" in note
    assert ".attempt" not in note


def test_needs_the_scorer_pin_in_approved(bs):
    """PR-M4: APPROVED must pin tools/score_bs.py (run_bs.py's preflight pins the five TOOL_FILES only)."""
    _plan(bs, {})
    for approved in ("{}", json.dumps({"tool_sha256": {"score_bs.py": "0" * 64}}), "not json {"):
        (bs["root"] / "APPROVED").write_text(approved)
        r = _run(bs)
        assert r.returncode == 2 and "score_bs.py" in r.stderr, r.stderr
        assert _calls(bs) == [] and not (bs["root"] / "runs").exists()
    # the real scorer path when no test override is given (absent in this build): refused, nothing started (LB's)
    (bs["root"] / "APPROVED").write_text(json.dumps({"tool_sha256": {"score_bs.py": _sha(bs["scorer"])}}))
    env = {k: v for k, v in bs["env"].items() if k != "BS_SCORER"}
    r = subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=60, cwd="/tmp")
    if not (TOOLS / "score_bs.py").exists():
        assert r.returncode == 2 and "score_bs.py" in r.stderr and _calls(bs) == []


def test_atimes_rearmed_at_the_decision_and_recorded_before_the_reads(bs):
    """PR-M2 (lesson P4): every file of a COUNTED continuation (runs/<name>, tmp/<name>, its attempt log) is re-armed
    right after the decision; the snapshot is written before logs/ALL_DECIDED and before any full read."""
    _plan(bs, {})
    r = _run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = _atimes(bs)
    for name in _names(4):
        mine = [p for p in rows if p.startswith((f"runs/{name}/", f"tmp/{name}/")) or p == f"logs/{name}.attempt1.log"]
        assert {f"runs/{name}/run_meta.json", f"runs/{name}/check.json", f"runs/{name}/01_stance/policy_probe.npz",
                f"tmp/{name}/eval_stage1_x/evaluations.npz", f"logs/{name}.attempt1.log"} <= set(mine)
        assert all(rows[p]["rearmed_unread"] == "1" for p in mine), {p: rows[p] for p in mine}
        assert all(int(rows[p]["atime_ns"]) == int(rows[p]["mtime_ns"]) - 1 for p in mine)
    assert rows["logs/decided.tsv"]["rearmed_unread"] == "0"               # the runner's own records: not re-armed
    log = (bs["root"] / "logs" / "run_waves_bs.log").read_text()
    assert log.index("atimes at ALL_DECIDED") < log.index("decided, none VOID")
    assert "logs/ALL_DECIDED" not in rows                                  # the snapshot precedes the marker
    calls = _calls(bs)
    assert sum(c.startswith("read ") for c in calls) == 4


def test_a_read_after_the_decision_shows_in_the_snapshot(bs):
    """PR-M2: a file of a decided continuation opened before ALL_DECIDED loses its re-armed atime (relatime)."""
    first, second = _names(2)
    _plan(bs, {second: ["vslowend"]})
    proc = subprocess.Popen(["bash", str(RUNNER)], env={**bs["env"], "BS_SLOTS": "1"}, cwd="/tmp",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 30
    while time.time() < deadline and _decided(bs).get(first) != "COUNTED":
        time.sleep(0.1)
    peeked = bs["root"] / "runs" / first / "01_stance" / "policy_probe.npz"
    log = bs["root"] / "logs" / "run_waves_bs.log"
    while time.time() < deadline and f"{first}: atimes re-armed" not in log.read_text():   # past the re-arm
        time.sleep(0.05)
    assert _decided(bs).get(second) is None
    assert peeked.read_text() == "probe"                                   # the peek (a forbidden read)
    out, err = proc.communicate(timeout=90)
    assert proc.returncode == 0, out + err
    rows = _atimes(bs)
    assert rows[f"runs/{first}/01_stance/policy_probe.npz"]["rearmed_unread"] == "0"
    assert rows[f"runs/{first}/run_meta.json"]["rearmed_unread"] == "1"
    assert rows[f"runs/{second}/01_stance/policy_probe.npz"]["rearmed_unread"] == "1"


def test_half_way_folders_at_relaunch(bs):
    first, second, third = _names(3)
    for name, meta in ((first, {"status": "running", "exit": None}),
                       (second, {"status": "crashed", "exit": 1, "last_step": C.S + 3 * C.EVAL_FREQ}),
                       (third, {"status": "done", "exit": 0, "stop_reason": "end"})):
        run = bs["root"] / "runs" / name
        run.mkdir(parents=True)
        (run / "run_meta.json").write_text(json.dumps(meta))
    _plan(bs, {})
    r = _run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = {row[1]: row[4] for row in _crash_rows(bs)}
    assert rows == {first: "0", second: "1"}
    assert set(_decided(bs).values()) == {"COUNTED"}
    assert not any(c.startswith(f"run {third} ") for c in _calls(bs))


def test_disk_pause_and_launch_floor(bs, tmp_path):
    free = tmp_path / "free"
    free.write_text("100\n")
    _plan(bs, {})
    r = _run(bs, BS_FREE_BYTES_FILE=str(free), BS_LAUNCH_DISK_BYTES="1000")
    assert r.returncode == 2 and "launch floor" in r.stderr and _calls(bs) == []
    # past the launch (a decision exists): below the floor it pauses; the stub frees space after a moment
    (bs["root"] / "logs").mkdir(exist_ok=True)
    (bs["root"] / "logs" / "decided.tsv").write_text("")
    (bs["root"] / "logs" / "crash_ledger.tsv").write_text("t\tx\tnone\tc\t0\n")
    proc = subprocess.Popen(["bash", str(RUNNER)], env={**bs["env"], "BS_FREE_BYTES_FILE": str(free),
                                                          "BS_DISK_FLOOR_BYTES": "1000"}, cwd="/tmp",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import time
    deadline = time.time() + 20
    while time.time() < deadline and not (bs["root"] / "logs" / "PAUSED_disk").exists():
        time.sleep(0.2)
    assert (bs["root"] / "logs" / "PAUSED_disk").exists() and _calls(bs) == []
    free.write_text("5000\n")
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, out + err
    assert set(_decided(bs).values()) == {"COUNTED"} and not (bs["root"] / "logs" / "PAUSED_disk").exists()


def test_screen_refusals(bs):
    root = bs["root"]
    bad = {"schema": C.SCREEN_SCHEMA, "intervention": "critic_phase",
           "queue": [["control", 3000 + i] for i in range(5)] + [["critic_phase", 4000 + i] for i in range(4)]}
    (root / "SCREEN.json").write_text(json.dumps(bad))
    r = _run(bs)
    assert r.returncode == 2 and "SCREEN" in r.stderr and _calls(bs) == []
    (root / "SCREEN.json").write_text(json.dumps({**bad, "queue": [["control", 1], ["gae_lambda_1", 2]]}))
    assert _run(bs).returncode == 2
    (root / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "critic_blind_tail3",
                                                  "queue": [["control", 1], ["critic_blind_tail3", 2]]}))
    assert _run(bs).returncode == 2                     # a BS2 arm is not a PS intervention
    (root / "SCREEN.json").unlink()
    r = _run(bs)
    assert r.returncode == 2 and "SCREEN" in r.stderr


def test_needs_approved_and_criteria(bs):
    (bs["root"] / "APPROVED").unlink()
    r = _run(bs)
    assert r.returncode == 2 and "APPROVED" in r.stderr


def test_lock_held(bs):
    (bs["root"] / "logs").mkdir()
    lock = bs["root"] / "logs" / "run_waves_bs.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "5"])
    import time
    time.sleep(0.5)
    try:
        r = _run(bs)
        assert r.returncode == 3
    finally:
        holder.kill()
        holder.wait()


def test_refuses_the_real_root_in_test_mode(bs):
    r = _run(bs, BS_ROOT=C.BS)
    assert r.returncode == 2 and "may not use the real" in r.stderr


def test_real_root_is_ps():
    text = RUNNER.read_text()
    assert "BS_REAL=/home/user/work/phase_screen\n" in text and C.BS == "/home/user/work/phase_screen"


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
    assert a1[0].split("\t") == ["path", "atime_ns", "mtime_ns", "ctime_ns", "atime_after_mtime"]
    assert any(f"/runs/{first}/run_meta.json" in ln for ln in a1) and not any("check.json" in ln for ln in a1)
    assert any(ln.split("\t")[0].endswith(f"logs/{first}.attempt1.log") for ln in a1[1:])
    a2 = (snap / f"{first}.attempt2.tsv").read_text()
    assert f"/runs/{first}/run_meta.json" in a2 and "check.json" not in a2
    s2 = (snap / f"{second}.attempt1.tsv").read_text().splitlines()
    rows = {ln.split("\t")[0].rsplit("/", 1)[-1]: ln.split("\t") for ln in s2[1:]}
    assert int(rows["run_meta.json"][1]) != int(rows["run_meta.json"][2]) - 1     # not yet re-armed at exit
