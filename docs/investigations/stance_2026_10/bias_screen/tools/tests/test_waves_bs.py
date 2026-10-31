"""Tests of run_waves_bs.sh with a stub harness (ported from the ECS runner's tests): all COUNTED then read only after
all are decided; a crash renamed and rerun; an early crash halts and is not counted; a post-run check failure is VOID;
three crashes to lost.json; a CHECK failure VOID; a refusal halts; a half-way folder at relaunch; the disk pause; the
launch floor; the lock; SCREEN.json's queue (and its refusal when invalid, more than 9 or missing); 4 slots.

The runner runs in its test mode (BS_WAVES_TEST=1 with a temporary BS_ROOT, never the real <BS>); every command it
starts is a stub written here.  Nothing trains, nothing simulates.
    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <BS>/tools/tests/test_waves_bs.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
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
S = 1048576
os.makedirs(f"{root}/attempts", exist_ok=True)
with open(f"{root}/attempts/{name}", "a") as fh:
    fh.write("x\n")
n = sum(1 for _ in open(f"{root}/attempts/{name}"))
plan = json.load(open(f"{root}/plan.json")).get(name, [])
what = plan[n - 1] if n <= len(plan) else "end"
with open(f"{root}/calls.log", "a") as fh:
    fh.write(f"run {name} {what}\n")
print(f"stub attempt {n} of {name}: {what}", flush=True)
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

QUEUE = [("control", 2101), ("critic_blind_tail3", 2101), ("control", 2111), ("critic_blind_tail3", 2111),
         ("control", 2121), ("critic_blind_tail3", 2121)]


def _names(n):
    return [C.run_name(a, s) for a, s in QUEUE[:n]]


@pytest.fixture()
def bs(tmp_path):
    root = tmp_path / "bs"
    root.mkdir()
    (root / "APPROVED").write_text("{}")
    (root / "criteria.txt").write_text("text\n")
    (root / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "critic_blind_tail3",
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
           "BS_POLL_SECONDS": "0.2", "BS_LAUNCH_DISK_BYTES": "0"}
    return {"root": root, "env": env, "stubs": stubs}


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
    assert "S 1048576, E 2490368" in r.stdout


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


def test_half_way_folders_at_relaunch(bs):
    first, second, third = _names(3)
    for name, meta in ((first, {"status": "running", "exit": None}),
                       (second, {"status": "crashed", "exit": 1, "last_step": 1048576 + 3 * 131072}),
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
    bad = {"schema": C.SCREEN_SCHEMA, "intervention": "critic_blind_tail3",
           "queue": [["control", 3000 + i] for i in range(5)] + [["critic_blind_tail3", 4000 + i] for i in range(5)]}
    (root / "SCREEN.json").write_text(json.dumps(bad))
    r = _run(bs)
    assert r.returncode == 2 and "SCREEN" in r.stderr and _calls(bs) == []
    (root / "SCREEN.json").write_text(json.dumps({**bad, "queue": [["control", 1], ["gae_lambda_1", 2]]}))
    assert _run(bs).returncode == 2
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
