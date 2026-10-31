"""Re-reviewer's own runner probes (stub harness, test mode, temp roots only; nothing trains).
Reuses the fixed test module's stubs and fixture by import (read only)."""
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import pytest

BS2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen2")
spec = importlib.util.spec_from_file_location("tw", str(BS2 / "tools/tests/test_waves_bs.py"))
tw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tw)
bs = tw.bs  # the fixture


def test_relaunch_after_all_decided_overwrites_the_snapshot(bs):
    """Q: if the runner is relaunched after logs/ALL_DECIDED (e.g. a restart during the full reads), is the
    first snapshot kept?"""
    tw._plan(bs, {})
    r = tw._run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    tsv = bs["root"] / "logs" / "atimes_at_ALL_DECIDED.tsv"
    first = tsv.read_text()
    rows1 = tw._atimes(bs)
    name = tw._names(1)[0]
    probe = f"runs/{name}/01_stance/policy_probe.npz"
    assert rows1[probe]["rearmed_unread"] == "1"
    # the full read (simulated: the stub read does not open files) reads the probe after ALL_DECIDED
    (bs["root"] / probe).read_text()
    r2 = tw._run(bs)
    second = tsv.read_text()
    rows2 = tw._atimes(bs)
    print("relaunch rc", r2.returncode)
    print("snapshot changed:", first != second)
    print("probe flag before / after relaunch:", rows1[probe]["rearmed_unread"], rows2[probe]["rearmed_unread"])
    print("reads issued:", sum(c.startswith("read ") for c in tw._calls(bs)))
    log = (bs["root"] / "logs" / "run_waves_bs.log").read_text()
    print("snapshot notes:", log.count("atimes at ALL_DECIDED"))
    # record the finding as a test outcome: the first snapshot is overwritten
    assert r2.returncode == 0
    assert first != second and rows2[probe]["rearmed_unread"] == "0"


def test_counted_at_relaunch_is_rearmed(bs):
    """A run found done at relaunch is checked, decided and re-armed."""
    name = tw._names(1)[0]
    run = bs["root"] / "runs" / name
    (run / "01_stance").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": "done", "exit": 0, "stop_reason": "end"}))
    (run / "01_stance" / "policy_probe.npz").write_text("probe")
    tw._plan(bs, {})
    r = tw._run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = tw._atimes(bs)
    assert rows[f"runs/{name}/01_stance/policy_probe.npz"]["rearmed_unread"] == "1"
    assert rows[f"runs/{name}/check.json"]["rearmed_unread"] == "1"


def test_collapse_is_stopped_and_rearmed(bs):
    name = tw._names(1)[0]
    tw._plan(bs, {name: ["collapse"]})
    r = tw._run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert tw._decided(bs)[name] == "STOPPED"
    rows = tw._atimes(bs)
    mine = [p for p in rows if p.startswith((f"runs/{name}/", f"tmp/{name}/"))]
    assert mine and all(rows[p]["rearmed_unread"] == "1" for p in mine)


def test_crashed_then_counted_attempt_logs(bs):
    """Both attempt logs of a continuation that crashed once are re-armed; its crashed folder is not."""
    name = tw._names(1)[0]
    tw._plan(bs, {name: ["crash"]})
    r = tw._run(bs)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = tw._atimes(bs)
    print({p: rows[p]["rearmed_unread"] for p in rows if name in p})
    assert rows[f"logs/{name}.attempt1.log"]["rearmed_unread"] == "1"
    assert rows[f"logs/{name}.attempt2.log"]["rearmed_unread"] == "1"
    assert rows[f"runs/{name}_crashed1/run_meta.json"]["rearmed_unread"] == "0"


def test_scorer_pin_ok_passes_and_wrong_key_form_refuses(bs):
    """APPROVED with the pin under 'tools/score_bs.py' (not the bare name) is refused."""
    tw._plan(bs, {})
    sha = tw._sha(tw.TOOLS / "score_bs.py")
    (bs["root"] / "APPROVED").write_text(json.dumps({"tool_sha256": {"tools/score_bs.py": sha}}))
    r = tw._run(bs)
    assert r.returncode == 2 and "score_bs.py" in r.stderr
    (bs["root"] / "APPROVED").write_text(json.dumps({"tool_sha256": {"score_bs.py": sha.upper()}}))
    r = tw._run(bs)
    assert r.returncode == 2
    (bs["root"] / "APPROVED").write_text(json.dumps({"tool_sha256": {"score_bs.py": sha}}))
    r = tw._run(bs)
    assert r.returncode == 0, r.stderr
