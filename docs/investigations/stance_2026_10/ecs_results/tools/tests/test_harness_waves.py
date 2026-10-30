"""Tests of run_waves_ecs.sh with a stub harness (build list v2 items 9 and V4): a non-zero exit renamed and rerun,
a half-way folder at relaunch treated as a crash, a CHECK failure stopping the queue (VOID), a refusal, the disk
pause, three crashes to lost.json, the reading after all are decided; and the build review's fixes: an early crash
halts and is not counted (H4), a post-run check failure is VOID (H3), logs/HALTED (H8), a relaunch waits for a live
orphan and no child holds the lock (H1), a large process table and a closed stdout (H2), the launch floor (H7), the
orphan pattern (I5), one log per attempt (I2, RR-6).

The runner runs in its test mode (ECS_WAVES_TEST=1 with a temporary ECS_ROOT, never the real <ECS>); every command it
starts is a stub written here.  Nothing trains, nothing simulates.
Run: OMP_NUM_THREADS=1 nice -n 15 PY -I -B -m pytest -q -p no:cacheprovider tests/test_harness_waves.py
"""

from __future__ import annotations

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
RUNNER = TOOLS / "run_waves_ecs.sh"
_spec = importlib.util.spec_from_file_location("ecs_common", str(TOOLS / "ecs_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

STUB_RUN = r'''
import json, os, sys, time
a = sys.argv[1:]
start, arm, seed = a[a.index("--start") + 1], a[a.index("--arm") + 1], a[a.index("--seed") + 1]
root = os.environ["ECS_ROOT"]
name = f"{start}_{arm}_{seed}"
S = {"s42": 3145728, "s80": 4194304}[start]
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
if what.startswith("hang"):  # an orphan: still running when its runner is killed
    open(f"{root}/hanging_{name}", "w").close()
    time.sleep(6.0)
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
elif what == "postfail":  # run_ecs.py's EXIT_POST_CHECK: the run reached its end, a post-run check failed
    meta = {"status": "failed", "exit": 5, "stop_reason": "end"}
elif what in ("crash", "earlycrash"):  # run_ecs.py's crash record: last_step, the last step StopAtEnd saw
    meta.update(status="crashed", exit=1, last_step=S + (8192 if what == "earlycrash" else 3 * 131072))
elif what == "kill":  # killed after the window's first evaluation: only the heartbeat says how far it got
    meta["heartbeat"] = {"rollout_end": S + 2 * 131072}
elif what == "earlykill":
    meta["heartbeat"] = {"rollout_end": S + 131072 - 8192}
json.dump(meta, open(f"{run}/run_meta.json", "w"))
if what in ("crash", "earlycrash"):
    sys.exit(1)
if what in ("kill", "earlykill"):
    os._exit(137)
if what == "postfail":
    sys.exit(5)
sys.exit(0)
'''

STUB_TOOL = r'''
import json, os, sys
root = os.environ["ECS_ROOT"]
kind, args = sys.argv[1], sys.argv[2:]
with open(f"{root}/calls.log", "a") as fh:
    fh.write(kind + " " + " ".join(args) + "\n")
if kind == "check":
    meta = json.load(open(os.path.join(args[0], "run_meta.json")))
    if meta.get("stub_check") == "fail" or meta.get("status") != "done" or meta.get("exit") != 0:
        sys.exit(4)
    json.dump({"pass": True}, open(os.path.join(args[0], "check.json"), "w"))
elif kind == "trace":
    out = args[args.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    json.dump({}, open(os.path.join(out, "m1_summary.json"), "w"))
sys.exit(0)
'''

BIG_PS = r'''
import sys
for i in range(4000):
    sys.stdout.write(f"{i:6d} " + "x" * 120 + "\n")
'''


def _queue(n: int) -> str:
    rows = []
    for start, arm, seed in C.QUEUE[:n]:
        S, E = C.window(start)
        rows.append(f"{start} {arm} {seed} {C.run_name(start, arm, seed)} {S} {E}")
    return ";".join(rows)


def _names(n: int) -> list[str]:
    return [C.run_name(*q) for q in C.QUEUE[:n]]


@pytest.fixture()
def ecs(tmp_path):
    root = tmp_path / "ecs"
    root.mkdir()
    (root / "APPROVED").write_text("{}")
    (root / "criteria.txt").write_text("text\n")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "stub_run_ecs.py").write_text(STUB_RUN)
    (stubs / "stub_tool.py").write_text(STUB_TOOL)
    (stubs / "big_ps.py").write_text(BIG_PS)
    py = sys.executable
    tool = f"{py} -I -B {stubs / 'stub_tool.py'}"
    env = {**os.environ, "ECS_WAVES_TEST": "1", "ECS_ROOT": str(root),
           "ECS_RUN_CMD": f"{py} -I -B {stubs / 'stub_run_ecs.py'}", "ECS_CHECK_CMD": f"{tool} check",
           "ECS_READ_CMD": f"{tool} read", "ECS_TRACE_CMD": f"{tool} trace", "ECS_SCORE_CMD": f"{tool} score",
           "ECS_ORPHAN_PATTERN": f"{stubs / 'stub_run_ecs.py'} --start", "ECS_POLL_SECONDS": "0.2",
           "ECS_SLOTS": "2", "ECS_QUEUE": _queue(4), "ECS_LAUNCH_DISK_BYTES": "0"}
    return {"root": root, "env": env, "stubs": stubs}


def _plan(ecs, plan: dict) -> None:
    (ecs["root"] / "plan.json").write_text(json.dumps(plan))


def _run(ecs, *args, timeout=120, **env_over):
    env = {**ecs["env"], **env_over}
    return subprocess.run(["bash", str(RUNNER), *args], env=env, capture_output=True, text=True, timeout=timeout,
                          cwd="/tmp")


def _decided(ecs) -> dict:
    out = {}
    p = ecs["root"] / "logs" / "decided.tsv"
    for line in p.read_text().splitlines() if p.exists() else []:
        _, name, d, _ = line.split("\t")
        out[name] = d
    return out


def _calls(ecs) -> list[str]:
    p = ecs["root"] / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


def _crash_rows(ecs) -> list[list[str]]:
    p = ecs["root"] / "logs" / "crash_ledger.tsv"
    return [line.split("\t") for line in p.read_text().splitlines()] if p.exists() else []


def test_all_end_counted_then_read_traced_scored(ecs):
    _plan(ecs, {})
    r = _run(ecs)
    assert r.returncode == 0, r.stdout + r.stderr
    names = _names(4)
    assert _decided(ecs) == {n: "COUNTED" for n in names}
    calls = _calls(ecs)
    assert sorted(c.split()[1] for c in calls if c.startswith("run ")) == sorted(names)
    # the fixed order, from the runner's own "start" lines: two stubs started in the same poll write calls.log in
    # whatever order their interpreters come up (measured at 00:32 UTC: s80_B_1401's line before s80_A_1301's)
    log = (ecs["root"] / "logs" / "run_waves_ecs.log").read_text().splitlines()
    assert [ln.split("] start ")[1].split()[0] for ln in log if "] start " in ln] == names
    assert sum(c.startswith("check ") for c in calls) == 4 and sum(c.startswith("read ") for c in calls) == 4
    traces = [c for c in calls if c.startswith("trace ")]
    assert len(traces) == 4 and all("stage1_4718592_steps.zip" in t or "stage1_5767168_steps.zip" in t for t in traces)
    assert calls[-1].startswith("score")
    last_check = max(i for i, c in enumerate(calls) if c.startswith("check "))
    assert all(calls.index(f"read {ecs['root']}/runs/{n}") > last_check
               for n in names)  # nothing is read until every continuation is decided
    last_trace = max(i for i, c in enumerate(calls) if c.startswith("trace "))
    assert last_trace < min(i for i, c in enumerate(calls) if c.startswith("read "))  # R3 before the full read
    assert (ecs["root"] / "logs" / "proctable.log").read_text().count("== ") == 4
    assert (ecs["root"] / "logs" / "run_waves_ecs.exit").read_text().split()[0] == "0"
    # one log per attempt (review I2, RR-6); no single per-name process log
    logs = ecs["root"] / "logs"
    assert all((logs / f"{n}.attempt1.log").read_text().startswith(f"stub attempt 1 of {n}") for n in names)
    assert not any((logs / f"{n}.log").exists() for n in names) and not (logs / "HALTED").exists()


def test_a_nonzero_exit_is_renamed_and_rerun_with_the_same_arguments(ecs):
    first = _names(4)[0]
    _plan(ecs, {first: ["crash"], _names(4)[1]: ["kill"]})
    r = _run(ecs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ecs["root"] / "runs" / f"{first}_crashed1").is_dir() and (ecs["root"] / "runs" / first).is_dir()
    assert (ecs["root"] / "runs" / f"{_names(4)[1]}_crashed1").is_dir()
    rows = _crash_rows(ecs)
    assert sorted((row[1], row[4]) for row in rows) == sorted([(first, "1"), (_names(4)[1], "1")])
    assert set(_decided(ecs).values()) == {"COUNTED"}
    assert sum(c == f"run {first} crash" for c in _calls(ecs)) == 1
    assert sum(c.startswith(f"run {first} ") for c in _calls(ecs)) == 2
    logs = ecs["root"] / "logs"
    assert (logs / f"{first}.attempt1.log").read_text().startswith(f"stub attempt 1 of {first}: crash")
    assert (logs / f"{first}.attempt2.log").read_text().startswith(f"stub attempt 2 of {first}: end")


def test_an_early_crash_halts_is_not_counted_and_starts_again_after_the_fix(ecs):
    """Review H4: a crash before the window's first evaluation (last step < S + 131,072, or none) is not counted toward
    LOST and halts new starts; once the coordinator removes logs/HALTED the continuation starts again under its name."""
    first, second = _names(4)[:2]
    _plan(ecs, {first: ["earlycrash"], second: ["earlykill"]})
    r = _run(ecs, ECS_SLOTS="2")
    assert r.returncode == 2, r.stdout + r.stderr
    rows = {row[1]: row for row in _crash_rows(ecs)}
    assert rows[first][4] == "0" and "EARLY CRASH" in rows[first][3] and str(3145728 + 8192) in rows[first][3]
    assert rows[second][4] == "0" and "EARLY CRASH" in rows[second][3]
    # both start in the runner's first poll, so their calls.log lines may come in either order
    assert sorted(c for c in _calls(ecs) if c.startswith("run ")) == [f"run {first} earlycrash",
                                                                      f"run {second} earlykill"]
    halted = ecs["root"] / "logs" / "HALTED"
    assert "early crash of" in halted.read_text() and _decided(ecs) == {}
    assert (ecs["root"] / "runs" / f"{first}_crashed1").is_dir() and not (ecs["root"] / "runs" / first).exists()
    r = _run(ecs)
    assert r.returncode == 2 and "halted" in r.stderr  # refused while logs/HALTED exists; nothing started
    assert len([c for c in _calls(ecs) if c.startswith("run ")]) == 2
    halted.unlink()
    r = _run(ecs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(_decided(ecs).values()) == {"COUNTED"} and sum(c.startswith(f"run {first} ") for c in _calls(ecs)) == 2


def test_a_post_run_check_failure_is_void_not_a_crash(ecs):
    """Review H3: exit 5 (a post-run check failed) is checked; CHECK fails on it: VOID, never renamed _crashed nor
    rerun."""
    first = _names(4)[0]
    _plan(ecs, {first: ["postfail"]})
    r = _run(ecs, ECS_SLOTS="1")
    assert r.returncode == 4, r.stdout + r.stderr
    assert _decided(ecs) == {first: "VOID"} and _crash_rows(ecs) == []
    assert (ecs["root"] / "runs" / f"{first}_void1").is_dir()
    assert [c for c in _calls(ecs) if c.startswith("run ")] == [f"run {first} postfail"]


def test_three_crashes_write_lost_json(ecs):
    first = _names(4)[0]
    _plan(ecs, {first: ["crash", "crash", "crash"]})
    r = _run(ecs)
    assert r.returncode == 0, r.stdout + r.stderr
    lost = json.loads((ecs["root"] / "lost.json").read_text())
    assert lost["schema"] == C.LOST_SCHEMA and lost["lost"][first]["cause"] == "crashes"
    assert _decided(ecs)[first] == "LOST"
    assert sorted(p.name for p in (ecs["root"] / "runs").iterdir() if p.name.startswith(first)) == [
        f"{first}_crashed1", f"{first}_crashed2", f"{first}_crashed3"]
    run = f"{ecs['root']}/runs/{first}"
    assert f"read {run}" not in _calls(ecs) and not any(f"{run}/01_stance" in c for c in _calls(ecs))
    assert _calls(ecs)[-1].startswith("score")


def test_a_check_failure_is_void_and_starts_nothing_new(ecs):
    first = _names(4)[0]
    _plan(ecs, {first: ["checkfail"]})
    r = _run(ecs, ECS_SLOTS="1")
    assert r.returncode == 4, r.stdout + r.stderr
    assert (ecs["root"] / "runs" / f"{first}_void1").is_dir()
    assert _decided(ecs) == {first: "VOID"}
    assert [c for c in _calls(ecs) if c.startswith("run ")] == [f"run {first} checkfail"]
    assert not any(c.startswith(("read", "trace", "score")) for c in _calls(ecs))
    # VOID is decided before the folder is renamed (review I4), and logs/HALTED holds every later start (H8)
    assert "folder " + f"{first}_void1" in (ecs["root"] / "logs" / "decided.tsv").read_text()
    halted = ecs["root"] / "logs" / "HALTED"
    assert "VOID" in halted.read_text()
    r2 = _run(ecs, ECS_SLOTS="1")
    assert r2.returncode == 2 and "halted" in r2.stderr
    # with logs/HALTED removed but the VOID line still in the ledger, the relaunch still refuses to start anything
    halted.unlink()
    r2 = _run(ecs, ECS_SLOTS="1")
    assert r2.returncode == 4 and [c for c in _calls(ecs) if c.startswith("run ")] == [f"run {first} checkfail"]
    assert halted.exists()


def test_running_ones_finish_and_are_checked_after_a_void(ecs):
    a, b = _names(4)[:2]
    _plan(ecs, {a: ["checkfail"], b: ["slowend"]})
    r = _run(ecs, ECS_SLOTS="2")
    assert r.returncode == 4
    assert _decided(ecs) == {a: "VOID", b: "COUNTED"}
    assert len([c for c in _calls(ecs) if c.startswith("run ")]) == 2


def test_a_refusal_stops_new_starts(ecs):
    first = _names(4)[0]
    _plan(ecs, {first: ["refuse"]})
    r = _run(ecs, ECS_SLOTS="1")
    assert r.returncode == 2, r.stdout + r.stderr
    assert [c for c in _calls(ecs) if c.startswith("run ")] == [f"run {first} refuse"]
    assert not (ecs["root"] / "runs" / first).exists() and _decided(ecs) == {}
    # review H8: a relaunch before the coordinator's fix refuses (logs/HALTED) and starts nothing
    halted = ecs["root"] / "logs" / "HALTED"
    assert "refused " + first in halted.read_text()
    r = _run(ecs, ECS_SLOTS="1")
    assert r.returncode == 2 and "halted" in r.stderr
    assert [c for c in _calls(ecs) if c.startswith("run ")] == [f"run {first} refuse"]
    # fixed (and logs/HALTED removed by the coordinator), it starts again under the same name
    _plan(ecs, {first: ["refuse", "end"]})
    halted.unlink()
    r = _run(ecs, ECS_SLOTS="1")
    assert r.returncode == 0 and _decided(ecs)[first] == "COUNTED"
    log = (ecs["root"] / "logs" / "run_waves_ecs.log").read_text()
    assert f"{first}.attempt1.log" in log  # the refusal note names the refused attempt's own log


def test_a_half_way_folder_at_relaunch_is_a_crash_not_counted_for_a_restart(ecs):
    first, second, third = _names(4)[:3]
    for name, meta in ((first, {"status": "running", "exit": None}),
                       (second, {"status": "crashed", "exit": 1, "last_step": 3145728 + 3 * 131072}),
                       (third, {"status": "done", "exit": 0, "stop_reason": "end"})):
        run = ecs["root"] / "runs" / name
        run.mkdir(parents=True)
        (run / "run_meta.json").write_text(json.dumps(meta))
    _plan(ecs, {})
    r = _run(ecs)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = {row[1]: row for row in _crash_rows(ecs)}
    assert rows[first][4] == "0" and "half-way" in rows[first][3]  # a container restart: not counted
    assert rows[second][4] == "1"  # its own recorded crash after the window's first evaluation: counted
    assert (ecs["root"] / "runs" / f"{first}_crashed1").is_dir() and _decided(ecs)[first] == "COUNTED"


def test_an_early_crash_found_at_relaunch_halts(ecs):
    first = _names(4)[0]
    run = ecs["root"] / "runs" / first
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": "crashed", "exit": 1, "last_step": None}))
    _plan(ecs, {})
    r = _run(ecs)
    assert r.returncode == 2, r.stdout + r.stderr
    assert _crash_rows(ecs)[0][4] == "0" and "EARLY CRASH" in _crash_rows(ecs)[0][3]
    assert not any(c.startswith("run ") for c in _calls(ecs)) and (ecs["root"] / "logs" / "HALTED").exists()


def test_a_failed_folder_at_relaunch_is_checked_not_rerun(ecs):
    first = _names(4)[0]
    run = ecs["root"] / "runs" / first
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": "failed", "exit": 5, "stop_reason": "end"}))
    _plan(ecs, {})
    r = _run(ecs)
    assert r.returncode == 4 and _decided(ecs) == {first: "VOID"} and _crash_rows(ecs) == []


def test_a_done_folder_at_relaunch_is_checked_not_rerun(ecs):
    first = _names(4)[0]
    run = ecs["root"] / "runs" / first
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": "done", "exit": 0, "stop_reason": "end"}))
    _plan(ecs, {})
    r = _run(ecs)
    assert r.returncode == 0
    assert _decided(ecs)[first] == "COUNTED" and not any(c.startswith(f"run {first} ") for c in _calls(ecs))


def test_a_collapse_stop_is_stopped_and_traced_on_its_final_pair(ecs):
    first = _names(4)[0]
    _plan(ecs, {first: ["collapse"]})
    r = _run(ecs)
    assert r.returncode == 0
    assert _decided(ecs)[first] == "STOPPED"
    trace = [c for c in _calls(ecs) if c.startswith("trace ") and f"/runs/{first}/" in c]
    assert len(trace) == 1 and "stage1_final.zip" in trace[0] and "stage1_final_vecnorm.pkl" in trace[0]


def test_the_disk_pause(ecs, tmp_path):
    free = tmp_path / "free"
    free.write_text("50\n")
    _plan(ecs, {})
    env = {**ecs["env"], "ECS_FREE_BYTES_FILE": str(free), "ECS_DISK_FLOOR_BYTES": "100", "ECS_QUEUE": _queue(2),
           "ECS_LAUNCH_DISK_BYTES": "10"}
    p = subprocess.Popen(["bash", str(RUNNER)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         cwd="/tmp")
    paused = ecs["root"] / "logs" / "PAUSED_disk"
    t0 = time.monotonic()
    while not paused.exists() and time.monotonic() - t0 < 30:
        time.sleep(0.1)
    assert paused.exists()
    time.sleep(0.6)
    assert not (ecs["root"] / "runs").exists() or not any((ecs["root"] / "runs").iterdir())  # nothing started
    free.write_text("1000\n")
    out, err = p.communicate(timeout=60)
    assert p.returncode == 0, out + err
    log = (ecs["root"] / "logs" / "run_waves_ecs.log").read_text()
    assert "PAUSED" in log and "resumed" in log and not paused.exists()
    assert set(_decided(ecs).values()) == {"COUNTED"}


def test_refusals_of_the_runner_itself(ecs, tmp_path):
    (ecs["root"] / "APPROVED").unlink()
    r = _run(ecs)
    assert r.returncode == 2 and "no <ECS>/APPROVED" in r.stderr
    r = _run(ecs, ECS_ROOT=C.ECS)
    assert r.returncode == 2 and "may not use the real <ECS>" in r.stderr


def test_plan_lists_the_24_in_the_fixed_order_and_writes_nothing():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECS_")}
    logs_before = os.path.exists(os.path.join(C.ECS, "logs"))
    r = subprocess.run(["bash", str(RUNNER), "--plan"], env=env, capture_output=True, text=True, timeout=60, cwd="/tmp")
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("  ") and "--start" in ln]
    assert len(lines) == 24
    got = [tuple(ln.split("--start ")[1].split()[0:5:2]) for ln in lines]
    assert got == [(s, a, str(seed)) for s, a, seed in C.QUEUE]
    assert "nice -n 10" in lines[0] and "slots 3 at nice 10" in r.stdout
    assert os.path.exists(os.path.join(C.ECS, "logs")) == logs_before


def test_the_launch_floor(ecs, tmp_path):
    """Review H7: a fresh queue starts only at or above ecs_common.LAUNCH_DISK_BYTES (T4's 6 GiB); a resumed one
    checks only the per-start floor."""
    assert C.LAUNCH_DISK_BYTES == 6 * 1024**3
    free = tmp_path / "free"
    free.write_text("50\n")
    _plan(ecs, {})
    over = {"ECS_FREE_BYTES_FILE": str(free), "ECS_DISK_FLOOR_BYTES": "10", "ECS_LAUNCH_DISK_BYTES": "100"}
    r = _run(ecs, **over)
    assert r.returncode == 2 and "launch floor 100" in r.stderr, r.stdout + r.stderr
    assert not _calls(ecs) and not (ecs["root"] / "logs" / "HALTED").exists()
    (ecs["root"] / "logs" / "decided.tsv").write_text(f"2026-10-28T00:00:00Z\t{_names(4)[0]}\tCOUNTED\tx\n")
    r = _run(ecs, **over)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len([c for c in _calls(ecs) if c.startswith("run ")]) == 3


def test_a_relaunch_waits_for_a_live_orphan_and_no_child_holds_the_lock(ecs):
    """Review H1: the runner is killed while a continuation runs; the orphan does not hold the runner's lock, so a
    relaunch waits for it (instead of exiting at the lock) and then checks it.  A second live runner exits 3."""
    import signal

    first = _names(4)[0]
    _plan(ecs, {first: ["hangend"]})
    env = {**ecs["env"], "ECS_SLOTS": "1"}
    p = subprocess.Popen(["bash", str(RUNNER)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                         cwd="/tmp", start_new_session=True)
    t0 = time.monotonic()
    while not (ecs["root"] / f"hanging_{first}").exists() and time.monotonic() - t0 < 30:
        time.sleep(0.05)
    assert (ecs["root"] / f"hanging_{first}").exists()
    second = subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=60, cwd="/tmp")
    assert second.returncode == 3 and "another run_waves_ecs.sh holds" in second.stderr
    p.send_signal(signal.SIGKILL)  # the runner only; its continuation lives on as an orphan
    p.wait(timeout=30)
    r = _run(ecs, ECS_SLOTS="1")
    assert r.returncode == 0, r.stdout + r.stderr
    log = (ecs["root"] / "logs" / "run_waves_ecs.log").read_text()
    assert "still alive" in log and f"{first}: found done at relaunch; checking" in log
    assert _decided(ecs)[first] == "COUNTED" and sum(c.startswith(f"run {first} ") for c in _calls(ecs)) == 1


def test_a_large_process_table_and_a_closed_stdout_do_not_end_the_runner(ecs):
    """Review H2: the process-table pipe (ps | head | cut) and the log lines survive SIGPIPE under pipefail."""
    big = f"{sys.executable} -I -B {ecs['stubs'] / 'big_ps.py'}"
    _plan(ecs, {})
    env = {**ecs["env"], "ECS_PS_CMD": big}
    p = subprocess.Popen(["bash", str(RUNNER)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         cwd="/tmp")
    p.stdout.close()  # nobody reads the runner's stdout any more: every write to it gets SIGPIPE / EPIPE
    p.wait(timeout=120)
    err = p.stderr.read()
    assert p.returncode == 0, err
    assert set(_decided(ecs).values()) == {"COUNTED"} and len(_decided(ecs)) == 4
    table = (ecs["root"] / "logs" / "proctable.log").read_text()
    assert table.count("== ") == 4 and sum("xxxxxxxxxx" in ln for ln in table.splitlines()) == 4 * 12


def test_the_orphan_pattern_matches_only_this_ecs_run_ecs_processes():
    """Review I5: the real pattern is anchored on this ECS's venv python and run_ecs.py, and its own text (as a watcher
    or shell would quote it) does not match it."""
    import re

    env = {k: v for k, v in os.environ.items() if not k.startswith("ECS_")}
    r = subprocess.run(["bash", str(RUNNER), "--plan"], env=env, capture_output=True, text=True, timeout=60, cwd="/tmp")
    pat = next(ln for ln in r.stdout.splitlines() if ln.startswith("orphan pattern: "))[len("orphan pattern: "):]
    py = f"{C.S2}/venv/bin/python"
    assert re.search(pat, f"{py} -I -B {C.TOOLS}/run_ecs.py --start s42 --arm A --seed 1101")
    assert re.search(pat, f"{py} -I -B {C.TOOLS}/run_ecs.py --start s80 --arm B --seed 1201 --no-update --until 1")
    for other in (f"bash -c pgrep -f '{pat}'", f"{py} -I -B /elsewhere/tools/run_ecs.py --start s42",
                  f"/usr/bin/python3 {C.TOOLS}/run_ecs.py --start s42", f"{py} -I -B {C.TOOLS}/read_ecs.py --check x",
                  f"watch.sh {py} -I -B {C.TOOLS}/run_ecs.py --start s42",
                  f"{py} -I -B {C.TOOLS}/[r]un_ecs[.]py --start s42"):
        assert not re.search(pat, other), other
