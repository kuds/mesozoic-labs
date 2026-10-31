"""The fixer's edit of tools/tests/test_waves_bs.py (tests for PR-M2, PR-M3, PR-M4).  Exact replacements, each old text
present once.  Run: PY -I -B edit_test_waves.py <BS2>/tools/tests/test_waves_bs.py"""
import sys

p = sys.argv[1]
src = open(p, encoding="utf-8").read()
E = []

E.append(('''launch floor; the lock; SCREEN.json's queue (and its refusal when invalid, more than 9 or missing); 4 slots.
''', '''launch floor; the lock; SCREEN.json's queue (and its refusal when invalid, more than 9 or missing); 4 slots.
BS2 review fixes: the refusal without APPROVED's score_bs.py pin (PR-M4); the atime re-arm of a decided continuation's
files and the snapshot logs/atimes_at_ALL_DECIDED.tsv before the full reads, which shows a read made after the
decision (PR-M2, lesson P4); the REFUSED note naming --preflight-only, not the attempt log (PR-M3).
'''))

E.append(('''import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
''', '''import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
'''))

# the stub run: a very slow variant, and files like a real run's (a probe under the run, an evaluations.npz in TMPDIR)
E.append(('''print(f"stub attempt {n} of {name}: {what}", flush=True)
if what.startswith("slow"):''', '''print(f"stub attempt {n} of {name}: {what}", flush=True)
if what.startswith("vslow"):
    time.sleep(4.0)
    what = what[5:] or "end"
if what.startswith("slow"):'''))

E.append(('''json.dump(meta, open(f"{run}/run_meta.json", "w"))
if what in ("crash", "earlycrash"):''', '''json.dump(meta, open(f"{run}/run_meta.json", "w"))
with open(f"{run}/01_stance/policy_probe.npz", "w") as fh:
    fh.write("probe")
os.makedirs(f"{root}/tmp/{name}/eval_stage1_x", exist_ok=True)
with open(f"{root}/tmp/{name}/eval_stage1_x/evaluations.npz", "w") as fh:
    fh.write("returns")
if what in ("crash", "earlycrash"):'''))

E.append(('''    (root / "APPROVED").write_text("{}")
''', '''    (root / "APPROVED").write_text(json.dumps({"tool_sha256": {"score_bs.py": _sha(TOOLS / "score_bs.py")}}))
'''))

E.append(('''def _names(n):
    return [C.run_name(a, s) for a, s in QUEUE[:n]]
''', '''def _names(n):
    return [C.run_name(a, s) for a, s in QUEUE[:n]]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atimes(bs):
    """logs/atimes_at_ALL_DECIDED.tsv as {path: row dict}."""
    lines = (bs["root"] / "logs" / "atimes_at_ALL_DECIDED.tsv").read_text().splitlines()
    head = lines[0].split("\\t")
    return {r[0]: dict(zip(head, r, strict=True)) for r in (ln.split("\\t") for ln in lines[1:])}
'''))

E.append(('''def test_refusal_halts(bs):
    first = _names(1)[0]
    _plan(bs, {first: ["refuse"]})
    r = _run(bs, BS_SLOTS="1")
    assert r.returncode == 2 and _decided(bs) == {}
    assert "refused " + first in (bs["root"] / "logs" / "HALTED").read_text()
''', '''def test_refusal_halts(bs):
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
    """PR-M2: a file of a decided continuation opened before ALL_DECIDED loses its re-armed atime (relatime moves it)."""
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
'''))

for old, new in E:
    assert src.count(old) == 1, old[:100]
    src = src.replace(old, new)
open(p, "w", encoding="utf-8").write(src)
print("ok", len(E))
