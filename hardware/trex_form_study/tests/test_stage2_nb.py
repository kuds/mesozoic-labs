"""``designlab.stage2.nb`` and ``stage2/stage2_training.ipynb``: the notebook's steps, the queue launcher, the
collection of finished records and the tally of runtime-hours (round C design 3.9, 3.10 and section 4's rows "The
notebook" and "No checkout dirt"; Track B rules K2, K6, K7, N1, N2, Y1-Y8).

Fast, on a **stub trainer** (a script that writes ``progress.json``, ``run.json``, ``gates/rate.json``,
``updates.jsonl``, checkpoint files and monitor records on a timer, and ends as its plan says), a stub clip tool and
a stub pack tool, named to the launcher through ``S2_NB_SEAMS``; every file under ``tmp_path``:

* the notebook: nine cells, no accelerator, every code cell parses, the kernel imports the standard library,
  ``google.colab`` and this module only; this module imports with numpy blocked; the pins file is the study venv;
* the four lines, their failing forms, the session line's three forms, line 4's floor;
* ``session``'s refusals (an existing RUN_ID, RESUME with another queue, set, bundle or budget, a REPO_REF that is
  no commit, a live ``LIVE.json``, a finished run, another commit without its flag) and what it accepts (TAKEOVER,
  ``ALLOW_SOURCE_CHANGE``, a ``LIVE.json`` older than ten minutes); the ``.part`` sweep; the SESSION and STATUS
  schemas;
* the launcher: a queue to DONE with its copies and line 4 per job; the copy cadence and the
  delete-one-cycle-later rule; a second launcher on a live stand-in Drive is refused; a superseded launcher halts
  with "superseded" and leaves the run's files alone; a crashed job is retried once and then HALT; a job stopped by
  a rule (exit 0 or 3) lets the next start; exit 4 ends the session with no HALT.json and exit 5 halts the run,
  resumable on the lead's LEAD.json, neither retried; a launcher killed with its job and
  resumed from the stand-in Drive in a new local directory;
* ``inputs`` on a real ``bundle`` round trip (a hand-made bundle of model files), on a data branch of a scratch
  repository and on the rebuild fallback; ``preflight``'s line 3; ``collect``; ``hours``; ``smoke`` on the stubs;
  ``release_runtime``; **a local job leaves ``git status --porcelain`` unchanged**.

Slow (``STAGE2_SLOW=1``; the Stage 1 view of 75-M-H-10, 2 workers, minutes): ``local_run``: the notebook's own
cells here with a tiny R5 and a tiny R6 job, a kill after the first copied checkpoint and a resume.
"""

from __future__ import annotations

import ast
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest
from designlab.stage2 import nb

STUDY = Path(__file__).resolve().parents[1]
NOTEBOOK = STUDY / "stage2" / "stage2_training.ipynb"
COMMIT = "c" * 40
slow = pytest.mark.skipif(not os.environ.get("STAGE2_SLOW"), reason="the real trainer on the S1 view; set STAGE2_SLOW=1")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

# ------------------------------------------------------------------------------------------------
# The stubs: a trainer, a clip tool and a pack tool with train2's, clip2's and s2load's command lines
# ------------------------------------------------------------------------------------------------
STUB_TRAINER = r'''
import argparse, hashlib, json, os, socket, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("cmd")
for name in ("--mode", "--body", "--set", "--out", "--run-id", "--job", "--monitor-also-k", "--lr", "--config"):
    ap.add_argument(name, default=None)
for name in ("--seed", "--steps", "--workers", "--monitor-every"):
    ap.add_argument(name, type=int, default=None)
for name in ("--resume", "--tiny", "--no-gates", "--allow-version-change", "--allow-source-change", "--retry-check"):
    ap.add_argument(name, action="store_true")
a = ap.parse_args()
if a.cmd == "preflight":
    print(json.dumps(dict(ok=True, mujoco_gl="disable", bytecode_members=[], fingerprint_max_abs_diff=0.0)))
    sys.exit(0)
job = a.job or "%s-%s-s%d" % (a.mode, a.body, a.seed)
jd = os.path.join(a.out, "jobs", job)
for sub in ("checkpoints", "gates", "monitor"):
    os.makedirs(os.path.join(jd, sub), exist_ok=True)
plans = json.loads(os.environ.get("STUB_PLAN") or "{}")
p = dict(plans.get("*", {}), **plans.get(job, {}))
time.sleep(float(p.get("start_delay", 0)))
rollout, every = int(p.get("rollout", 512)), int(p.get("checkpoint_every", 1024))
mon = int(a.monitor_every or p.get("monitor_every", 2048))
budget, tick = int(a.steps or p.get("budget", 4096)), float(p.get("tick", 0.02))

def write(path, doc):
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f)
    os.replace(path + ".tmp", path)

def read(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None

with open(os.path.join(a.out, "stub_calls.jsonl"), "a") as f:
    f.write(json.dumps(dict(argv=sys.argv[1:], job=job, pid=os.getpid(), mujoco_gl=os.environ.get("MUJOCO_GL"),
                            hashseed=os.environ.get("PYTHONHASHSEED"), cwd=os.getcwd(), pgid=os.getpgid(0))) + "\n")
run_path = os.path.join(jd, "run.json")
run = read(run_path)
if run is not None and not a.resume:
    print("stub: run.json exists: pass --resume", file=sys.stderr)
    sys.exit(4)
if run is None and a.resume:
    print("stub: nothing to resume", file=sys.stderr)
    sys.exit(4)
if p.get("refuse"):
    print("stub: Refused: other settings than the run's", file=sys.stderr)
    sys.exit(4)
VERSIONS, version_change = dict(numpy="stub-1"), False      # (train2's rule K4 on the library versions)
if a.resume and run is not None and run["sessions"] and run["sessions"][-1].get("versions") not in (None, VERSIONS):
    if not a.allow_version_change:
        print("stub: Refused: other library versions than the run's last session (pass --allow-version-change to "
              "resume on them, flagged)", file=sys.stderr)
        sys.exit(4)
    version_change = True
if run is None:
    run = dict(schema="s2-run-1", run_uuid="uuid-" + job, run_id=a.run_id, job=job, mode=a.mode, body=a.body,
               status="running", num_timesteps=0, budget=budget, sessions=[], stopped=None, halt=None,
               settings=dict(tiny=bool(a.tiny), n_envs=8, n_steps=rollout // 8, checkpoint_every=every,
                             monitor_every=mon))

def ck(n, kind):
    return os.path.join(jd, "checkpoints", {"model": "model_%d.zip", "vecnormalize": "vecnormalize_%d.pkl",
                                            "fingerprint": "fingerprint_%d.npz", "record": "checkpoint_%d.json"}[kind] % n)

def have():
    out = []
    for name in os.listdir(os.path.join(jd, "checkpoints")):
        if name.startswith("checkpoint_") and name.endswith(".json"):
            out.append(int(name[len("checkpoint_"):-5]))
    return sorted(out)

def save(n):
    files = {}
    for kind in ("model", "vecnormalize", "fingerprint"):
        data = ("%s %s %d " % (job, kind, n)).encode() * int(p.get("blob", 50))
        with open(ck(n, kind), "wb") as f:
            f.write(data)
        files[kind] = dict(name=os.path.basename(ck(n, kind)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    write(ck(n, "record"), dict(schema="s2-ckpt-2", steps=n, run_uuid=run["run_uuid"], files=files))
    if not p.get("keep_all"):
        all_n = have()
        checks = [m for m in all_n if m > 0 and m % mon < rollout]
        for m in all_n:
            if m not in checks and m not in all_n[-2:]:
                for kind in ("record", "model", "vecnormalize", "fingerprint"):
                    os.remove(ck(m, kind))

def check(n):
    write(os.path.join(jd, "monitor", "monitor_%d.json" % n), dict(schema="s2-monitor-1", at_steps=n, episodes=[]))
    if p.get("traces"):
        t = os.path.join(jd, "monitor", "traces", str(n), "R5", "42@10ms", "walk", "v0.4")
        os.makedirs(t, exist_ok=True)
        open(os.path.join(t, "6004.npz"), "wb").write(b"trace %d" % n)
        open(os.path.join(t, "6005.npz"), "wb").write(b"other")

n = max(have(), default=0) if a.resume else 0
k = len(run["sessions"])
run["sessions"].append(dict(index=k, resumed_from=n if a.resume and have() else None, pid=os.getpid(),
                            host=socket.gethostname(), fingerprint_max_abs_diff=0.0 if a.resume else None,
                            skipped_checkpoints=[], workers=a.workers, argv=sys.argv[1:], versions=VERSIONS,
                            version_change=version_change))
run["status"] = "running"
write(run_path, run)
if p.get("halt"):
    why = p["halt"] if isinstance(p["halt"], str) else (
        "stub: two attempts at one check were never closed; with the fault repaired, a resume with --retry-check "
        "lets the check run once more")
    run["status"], run["halt"] = "partial", dict(session=k, reason=why)
    write(run_path, run)
    print("stub: Halt: " + why, file=sys.stderr)
    sys.exit(5)
if p.get("g8a_fail"):
    run["status"], run["stopped"] = "stopped_g8a", dict(at_steps=0, reason="G8a fails: no run")
    write(run_path, run)
    print(job + ": G8a fails: no run")
    sys.exit(3)
count_path = os.path.join(jd, "crashes")
done_rollouts, last_ck, last_mon = 0, None, None
print("%s: stub session %d from %d steps" % (job, k, n), flush=True)
while True:
    if n % every < rollout and (n == 0 or n >= every) and n not in have():
        save(n)
    if n in have():
        last_ck = n
    if n > 0 and n % mon < rollout and not os.path.exists(os.path.join(jd, "monitor", "monitor_%d.json" % n)):
        check(n)
        last_mon = dict(at_steps=n, N=1, S=2.0, mean_clock_factor=1.0, gain=True, counted_without_record=False)
    write(os.path.join(jd, "progress.json"), dict(
        schema="s2-progress-1", job=job, run_uuid=run["run_uuid"], run_id=a.run_id, session=k, pid=os.getpid(),
        status="running", steps=n, budget=budget, env_steps_per_s=float(p.get("rate", 500.0)), eta_h=0.1,
        workers=a.workers, last_checkpoint=last_ck, last_monitor=last_mon,
        checks_without_record=list(p.get("checks_without_record", [])), halt=None, g8a=True, g8b=None, fix=None,
        flags=dict(kl_cut=False), stopped=None, minibatch_steps_last=[160, 160]))
    crashes = int(open(count_path).read()) if os.path.exists(count_path) else 0
    if p.get("crash_at") is not None and n >= int(p["crash_at"]) and crashes < int(p.get("crash_times", 1)):
        open(count_path, "w").write(str(crashes + 1))
        print("stub: boom at %d steps" % n, flush=True)
        os._exit(int(p.get("crash_code", 1)))
    if p.get("stop_at") is not None and n >= int(p["stop_at"]):
        run["status"], run["stopped"] = "stopped_no_gain", dict(at_steps=n, reason="stub: three checks without a gain")
        run["num_timesteps"] = n
        write(run_path, run)
        print("%s: stopped at %d steps" % (job, n), flush=True)
        sys.exit(0)
    if n >= budget:
        break
    time.sleep(tick)
    n += rollout
    done_rollouts += 1
    with open(os.path.join(jd, "updates.jsonl"), "a") as f:
        f.write(json.dumps(dict(session=k, update=n // rollout, steps=n, wall_s=rollout / float(p.get("rate", 500.0)))) + "\n")
    if done_rollouts in (1, 3):
        rate = read(os.path.join(jd, "gates", "rate.json")) or dict(schema="s2-rate-1", job=job, sessions=[])
        mine = next((s for s in rate["sessions"] if s["session"] == k), None)
        if mine is None:
            mine = dict(session=k)
            rate["sessions"].append(mine)
        mine["first_rollout" if done_rollouts == 1 else "first_three"] = dict(
            env_steps_per_wall_s=float(p.get("rate", 500.0)), workers=a.workers, vcpus=4, n_envs=8,
            rss_mb=dict(trainer=100.0, workers=[50.0] * int(a.workers or 0)))
        write(os.path.join(jd, "gates", "rate.json"), rate)
    run["num_timesteps"] = n
    write(run_path, run)
run["status"], run["num_timesteps"] = "done", n
write(run_path, run)
print("%s: done at %d steps" % (job, n), flush=True)
sys.exit(int(p.get("exit", 0)))
'''

STUB_CLIP = r'''
import json, os, sys, time
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
plan = json.loads(os.environ.get("STUB_CLIP") or "{}")
time.sleep(float(plan.get("sleep", 0)))
if plan.get("crash"):
    sys.exit(7)
out = a["--out"]
ok = not plan.get("fail")
if ok:
    open(out, "wb").write(b"mp4 of " + open(a["--trace"], "rb").read())
json.dump(dict(schema="s2-clip-1", ok=ok, reason=None if ok else "no renderer", trace=a["--trace"],
               mujoco_gl=os.environ.get("MUJOCO_GL"), platform=os.environ.get("PYOPENGL_PLATFORM")),
          open(os.path.splitext(out)[0] + ".json", "w"))
'''

STUB_PACK = r'''
import os, sys
cmd, job_dir, n, flag, out = sys.argv[1:6]
assert cmd == "pack" and flag == "--out"
os.makedirs(out, exist_ok=True)
data = b"".join(open(os.path.join(job_dir, "checkpoints", name % int(n)), "rb").read()
                for name in ("fingerprint_%d.npz", "model_%d.zip", "vecnormalize_%d.pkl"))
open(os.path.join(out, "ckpt_%d.tar" % int(n)), "wb").write(data)
'''


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    """The stub tools, named through ``S2_NB_SEAMS`` (so that a launcher in a process of its own finds them)."""
    d = tmp_path / "stubs"
    d.mkdir()
    for name, text in (("trainer.py", STUB_TRAINER), ("clip.py", STUB_CLIP), ("pack.py", STUB_PACK)):
        (d / name).write_text(text)
    seams = dict(trainer=[sys.executable, str(d / "trainer.py")], clip=[sys.executable, str(d / "clip.py")],
                 pack=[sys.executable, str(d / "pack.py")])
    (d / "seams.json").write_text(json.dumps(seams))
    monkeypatch.setenv("S2_NB_SEAMS", str(d / "seams.json"))
    monkeypatch.delenv("STUB_PLAN", raising=False)
    monkeypatch.delenv("STUB_CLIP", raising=False)
    return d


@pytest.fixture
def inproc_stubs(stubs, monkeypatch):
    """The stub tools for a launcher that runs in this process, named by monkeypatching the seams instead of
    ``S2_NB_SEAMS``: that variable is refused in a run that is not tiny (the lead's ruling 4 of 17 Nov), and these
    tests read what a run that is not tiny does."""
    seams = json.loads((stubs / "seams.json").read_text())
    for name, attr in (("trainer", "_trainer_cmd"), ("clip", "_clip_cmd"), ("pack", "_pack_cmd")):
        monkeypatch.setattr(nb, attr, lambda cmd=seams[name]: list(cmd))
    monkeypatch.delenv("S2_NB_SEAMS")
    return stubs


def _params(tmp_path, **over) -> dict:
    p = dict(RUN_ID="r1", QUEUE=["R5:B1:42", "R6:B1:42"], SET="S1", TINY=True, REPO_REF=COMMIT, BUNDLE_ID="",
             WORKERS=2, CLIPS=True, PERIOD_S=0.05, DRIVE_DIR=str(tmp_path / "drive"), LOCAL_DIR=str(tmp_path / "local"),
             USE_GOOGLE_DRIVE=False, AUTO_DISCONNECT=False)
    p.update(over)
    return nb.load_params(p)


def _session(p, **over):
    commit = over.pop("commit", COMMIT)
    p = dict(p, **over)
    return nb.session(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"], p["QUEUE"], p["REPO_REF"], commit, p["RESUME"], p)


def _launch(p):
    return nb.launch(p["RUN_ID"], p["DRIVE_DIR"], p["LOCAL_DIR"], p["QUEUE"], p["SET"], p["BUDGET_STEPS"], p["WORKERS"],
                     tiny=p["TINY"], clips=p["CLIPS"], monitor_every=p["MONITOR_EVERY"],
                     monitor_also_k=p["MONITOR_ALSO_K"] or ())


def _run(tmp_path, plan=None, monkeypatch=None, **over):
    """A new session and its launch on the stub; ``(params, launch's result, the Drive run directory)``."""
    if plan is not None:
        monkeypatch.setenv("STUB_PLAN", json.dumps(plan))
    p = _params(tmp_path, **over)
    _session(p)
    return p, _launch(p), Path(p["DRIVE_DIR"]) / "runs" / p["RUN_ID"]


def _calls(p) -> list:
    path = Path(p["LOCAL_DIR"]) / p["RUN_ID"] / "stub_calls.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


def _reads_back(jobdir: Path, n: int) -> bool:
    rec = json.loads((jobdir / "checkpoints" / f"checkpoint_{n}.json").read_text())
    return all(hashlib.sha256((jobdir / "checkpoints" / e["name"]).read_bytes()).hexdigest() == e["sha256"]
               for e in rec["files"].values())


# ------------------------------------------------------------------------------------------------
# The notebook, the import, the pins
# ------------------------------------------------------------------------------------------------
def _cells() -> list:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def test_notebook_cells_parse():
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = book["cells"]
    assert [c["cell_type"] for c in cells] == ["markdown"] + ["code"] * 8        # design 3.9: nine cells
    assert "accelerator" not in book["metadata"] and "gpuType" not in book["metadata"].get("colab", {})
    for i, c in enumerate(cells[1:], 1):
        src = "".join(c["source"])
        ast.parse(src, filename=f"<cell {i}>")                                   # plain Python: no magic
        assert "get_ipython" not in src and not c["outputs"] and c["execution_count"] is None
    text = "".join(cells[0]["source"])
    assert "stage2/tb/colab_steps.txt" in text and "Nothing here trains in the notebook's own process" in text
    # cell 1: the parameters exactly as the design lists them, each a form field
    tree = ast.parse("".join(cells[1]["source"]))
    names = [t.id for node in tree.body if isinstance(node, ast.Assign) for t in node.targets
             if isinstance(t, ast.Name) and t.id.isupper()]
    assert names[:len(nb.PARAMS)] == list(nb.PARAMS) and "S2_PARAMS" in "".join(cells[1]["source"])
    # the lead's ruling 2 of 17 Nov: the owner is never asked to set a remedy (they come from LEAD.json)
    assert not any(n.startswith("ALLOW_") or n == "RETRY_CHECK" for n in names) and "LEAD.json" in "".join(cells[1]["source"])
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and node.targets[0].id in nb.PARAMS}
    assert values == nb.PARAMS                                                   # the cell's defaults are the module's
    for name in nb.PARAMS:
        assert f"{name} = " in "".join(cells[1]["source"]) and "# @param" in "".join(cells[1]["source"])
    # each later cell runs one step (cells 6 and 7 behind SMOKE); the last one releases the runtime in a finally
    steps = ["setup", "session", "inputs", "preflight", "smoke", "launch", "finish"]
    for c, step in zip(cells[2:], steps, strict=True):
        assert f's2_step("{step}"' in "".join(c["source"])
    last = ast.parse("".join(cells[8]["source"])).body
    assert isinstance(last[-1], ast.Try) and "release_or_keep(S2_FINISH_CODE" in ast.unparse(last[-1].finalbody)
    two = "".join(cells[2]["source"])
    assert 'fullmatch(r"[0-9a-f]{40}", REPO_REF)' in two and '"designlab"' in two and "--no-checkout" in two
    assert 'MUJOCO_GL="disable"' in two and 'PYTHONHASHSEED="0"' in two and 'PYTHONPATH="."' in two
    assert "pip" not in two                                                      # the pins are nb.setup's, in a child


def test_kernel_cells_import_only_stdlib():
    allowed = set(sys.stdlib_module_names)
    seen = set()
    for c in _cells()[1:]:
        for node in ast.walk(ast.parse("".join(c["source"]))):
            if isinstance(node, ast.Import):
                seen |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0
                seen.add(node.module)
    other = {m for m in seen if m.split(".")[0] not in allowed}
    assert other == {"google.colab", "designlab.stage2"}, other
    assert "numpy" not in seen and "environments" not in " ".join(seen)


def test_nb_imports_with_numpy_blocked():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('numpy', 'torch', 'mujoco', 'scipy', 'gymnasium', 'stable_baselines3'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from designlab.stage2 import nb\n"
        "nb.release_runtime('over', in_colab=False, auto=True, flush_drive=True)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] not in sys.stdlib_module_names\n"
        "             and not m.startswith('designlab') and m != '__main__'))\n")
    env = dict(os.environ, PYTHONPATH=".")
    done = subprocess.run([sys.executable, "-c", code], cwd=STUDY, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "Auto-disconnect skipped" in done.stdout
    left = [m for m in json.loads(done.stdout.strip().splitlines()[-1].replace("'", '"'))
            if m.split(".")[0] not in ("_distutils_hack", "sitecustomize", "usercustomize")
            and not m.startswith("__editable__")]
    assert left == [], left                                                      # nothing outside the standard library


def test_pins_equal_the_study_venv():
    import importlib.metadata as md

    pinned = nb.pins(STUDY / nb.PINS_FILE)
    assert list(pinned) == ["mujoco", "numpy", "scipy", "gymnasium", "stable-baselines3", "torch", "cloudpickle",
                            "pillow", "PyOpenGL"]                                 # rule N2's list
    for name, want in pinned.items():
        assert nb.comparable(md.version(name)) == want, name                     # every pin, PyOpenGL and pillow too
    vers = nb.versions(pinned)
    assert nb.pin_differences(pinned, vers) == []
    # torch's CPU build is forced whenever the installed build label is not "cpu"
    assert nb.pip_commands(pinned, dict(vers, torch=pinned["torch"] + "+cpu")) == []
    for other in (pinned["torch"] + "+cu126", pinned["torch"], "2.9.0+cpu", None):
        cmds = nb.pip_commands(pinned, dict(vers, torch=other))
        assert len(cmds) == 1 and f"torch=={pinned['torch']}+cpu" in cmds[0] and nb.TORCH_CPU_INDEX in cmds[0]
    cmds = nb.pip_commands(pinned, dict(vers, torch=pinned["torch"] + "+cpu", numpy="2.0.2"))
    assert len(cmds) == 1 and "numpy==" + pinned["numpy"] in cmds[0] and not any("torch" in x for x in cmds[0])
    assert all("==" in x for c in cmds for x in c if x[0].isalpha() and x not in ("pip", "install"))  # no unpinned name
    assert nb.pin_differences(pinned, dict(vers, pillow="12.0.0")) == [f"pillow: installed 12.0.0, pinned "
                                                                       f"{pinned['pillow']}"]


def test_setup_prints_line_1_and_has_no_unpinned_fallback(tmp_path, monkeypatch, capsys):
    head = nb.repo_state()["commit"]
    rec = nb.setup(dict(REPO_REF=head), record=tmp_path / "setup.json")
    m = nb.LINE_RE[1].match(rec["line"])
    assert m and m.group("commit") == head and m.group("ref") == "yes" and rec["pins_match"] and rec["pip"] == []
    assert json.loads((tmp_path / "setup.json").read_text())["line"] == rec["line"] in capsys.readouterr().out
    assert nb.LINE_RE[1].match(nb.setup(dict(REPO_REF="main"))["line"]).group("ref") == "no"
    real = nb.versions
    monkeypatch.setattr(nb, "versions", lambda pinned: dict(real(pinned), numpy="1.26.4"))
    with pytest.raises(nb.NbError, match="no unpinned fallback"):
        nb.setup(dict(REPO_REF=head))
    # on Colab a pin that does not install stops the step, and nothing else is tried
    ran = []
    monkeypatch.setattr(nb, "_call", lambda cmd, timeout=None: ran.append(cmd) or (1 if "install" in cmd and "pip" in cmd else 0))
    with pytest.raises(nb.NbError, match="a pin did not install"):
        nb.setup(dict(REPO_REF=head, IN_COLAB=True))
    assert [c[3] for c in ran if "pip" in c] == ["uninstall", "install"]
    # rule N2: on Colab the installed torch is the CPU build after the installs, whatever pip reported; a torch that
    # still carries another build label stops the step.  apt's package lists are updated before the install
    ran.clear()
    monkeypatch.setattr(nb, "_call", lambda cmd, timeout=None: ran.append(cmd) or 0)
    monkeypatch.setattr(nb, "versions", lambda pinned: dict(real(pinned), torch=pinned["torch"] + "+cu126"))
    with pytest.raises(nb.NbError, match="not the CPU build"):
        nb.setup(dict(REPO_REF=head, IN_COLAB=True), record=tmp_path / "setup2.json")
    assert [c[:2] for c in ran if c[0] == "apt-get"] == [["apt-get", "update"], ["apt-get", "install"]]
    assert any(f"torch=={nb.pins()['torch']}+cpu" in c for c in ran)             # the CPU build was asked for
    rec = json.loads((tmp_path / "setup2.json").read_text())
    assert rec["pins_match"] is False and rec["torch_build"] == "cu126" and rec["line"] is None
    monkeypatch.setattr(nb, "versions", lambda pinned: dict(real(pinned), torch=pinned["torch"] + "+cpu"))
    assert nb.setup(dict(REPO_REF=head, IN_COLAB=True))["torch_build"] == "cpu"
    # outside Colab nothing is installed: the build label is recorded, not judged
    monkeypatch.setattr(nb, "versions", lambda pinned: dict(real(pinned), torch=pinned["torch"] + "+cu126"))
    assert nb.setup(dict(REPO_REF=head))["torch_build"] == "cu126"


# ------------------------------------------------------------------------------------------------
# The lines
# ------------------------------------------------------------------------------------------------
def test_the_four_lines_and_their_failing_forms():
    a = "S2 LINE 1/4 commit: " + COMMIT + " (REPO_REF is a commit: yes; checkout clean: yes)"
    assert nb.line1(COMMIT, True, True) == a and nb.LINE_RE[1].match(a)
    assert nb.line1(COMMIT, False, False).endswith("(REPO_REF is a commit: no; checkout clean: no)")
    b = nb.line2("tb-1", "ab" * 32, 137, 137)
    assert b == "S2 LINE 2/4 inputs: bundle tb-1, manifest abababababab, 137 of 137 files match their sha256: OK"
    bad = nb.line2("tb-1", "ab" * 32, 136, 137, "V6-75-M-H-10-1.tar: missing in /drive")
    assert bad.endswith(": FAILED (V6-75-M-H-10-1.tar: missing in /drive)") and nb.LINE_RE[2].match(bad)
    assert nb.LINE_RE[2].match(b).group("manifest") == "abababababab"
    c = nb.line3("75-M-H-10-X", "V6", 2, 2, 6000)
    assert c == ("S2 LINE 3/4 G8a 75-M-H-10-X set V6: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes identical "
                 "(6000 physics steps): PASS")
    f = nb.line3("75-M-H-10", "V6", 1, 2, 6000, dict(condition="walk/v0.4", seed=3042, ms=10, step=17))
    assert f.endswith(": FAIL (walk/v0.4 seed 3042 at 10 ms, first difference at step 17)")
    got = nb.FAIL3_RE.search(nb.LINE_RE[3].match(f).group("end"))
    assert got and got.group("condition") == "walk/v0.4" and got.group("ms") == "10" and got.group("step") == "17"
    assert nb.LINE_RE[3].match(nb.line3("75-M-H-10", "V6", 0, 0, 0, "the gate ended\nwith exit code 1"))
    d = nb.line4("R5-75-M-H-10-s42", 560.4, 8, 8, 8, 40_000_000)
    assert d == "S2 LINE 4/4 first rate R5-75-M-H-10-s42: 560 env steps/s (8 envs, 8 workers, 8 vCPUs); 40M steps in about 19.8 h"
    low = nb.line4("R5-75-M-H-10-s42", 504.9, 8, 8, 8, 40_000_000)
    assert low.endswith(" h BELOW FLOOR 505") and nb.LINE_RE[4].match(low).group("floor") == "505"
    assert "BELOW FLOOR" not in nb.line4("R5-75-M-H-10-s42", 505.1, 8, 8, 8, 40_000_000)   # 40M / 79,200 = 505.05
    assert nb.floor_rate(40_000_000) == pytest.approx(505.0505, abs=1e-3)
    assert nb.LINE_RE[4].match(nb.line4("R6-B-s1", 431.2, 8, 2, 4, 4096)).group("budget") == "0.004096"
    for text in (a, b, bad, c, f, d, low):
        assert "\n" not in text and text.startswith("S2 LINE ")
    # the session line's three forms, and the smoke's
    assert nb.session_line("/d/runs/r", "new") == "Run directory: /d/runs/r (new run)"
    r = nb.session_line("/d/runs/r", "resume", session=1, job="R5-75-M-H-10-s42", steps=3000000, done=0)
    assert r == "Run directory: /d/runs/r (resuming session 1: R5-75-M-H-10-s42 from 3000000 steps; 0 jobs done)"
    live = nb.session_line("/d/runs/r", "live", age_s=42.7)
    assert live == "Run directory: /d/runs/r: still running (LIVE.json written 42 s ago); nothing to do"
    assert nb.LINE_RE["new"].match(nb.session_line("/d", "new")) and nb.LINE_RE["resume"].match(r)
    assert nb.LINE_RE["live"].match(live).group("age") == "42"
    s = nb.smoke_line(4, "rate at 8 envs", "560 env steps/s", "OK")
    assert s == "S2 SMOKE Y4 rate at 8 envs: 560 env steps/s: OK" and nb.LINE_RE["smoke"].match(s)
    with pytest.raises(ValueError):
        nb.smoke_line(4, "x", "y", "PASSED")
    got = nb.lines_of("\n".join(["noise", a, "  " + a, b, c, c, d, r, s]))
    assert got[1] == [a] and got[2] == [b] and got[3] == [c, c] and got[4] == [d] and got["resume"] == [r]
    # rule Y11's budget
    assert nb.budget_for(555.56) == 40_000_000 and nb.budget_for(555.5) == 35_000_000
    assert nb.budget_for(208.34) == 15_000_000 and nb.budget_for(208.3) is None and nb.budget_for(300) == 20_000_000


def test_queue_is_a_text_field():
    """The lead's ruling 5 of 17 Nov: QUEUE is a text field of cell 1.  The jobs separated by commas; brackets and
    straight or curly quotation marks around them are dropped (a phone keyboard makes curly ones, and the raw field
    stopped with a SyntaxError); the entries recorded are the same whatever the spelling."""
    cell = "".join(_cells()[1]["source"])
    assert 'QUEUE = ""  # @param {"type":"string"}' in cell and nb.PARAMS["QUEUE"] == ""
    want = ["R5:75-M-H-10:42", "R6:75-M-H-10:42"]
    for text in ("R5:75-M-H-10:42, R6:75-M-H-10:42", "R5:75-M-H-10:42,R6:75-M-H-10:42 ",
                 '["R5:75-M-H-10:42", "R6:75-M-H-10:42"]', "\u201cR5:75-M-H-10:42\u201d, \u2018R6:75-M-H-10:42\u2019"):
        assert [r["entry"] for r in nb.parse_queue(text)] == want, text
    assert nb.parse_queue("") == [] and nb.parse_queue(want) == nb.parse_queue(", ".join(want))
    for bad in ("R5:75-M-H-10:42; R6:75-M-H-10:42", "R7:75-M-H-10:42", "R5:75-M-H-10"):
        with pytest.raises(nb.Refused, match="MODE:BODY:SEED"):
            nb.parse_queue(bad)


def test_the_steps_sheet_quotes_the_programs_own_lines():
    """``stage2/tb/colab_steps.txt`` (design 3.12): plain text of 115 columns; every line it quotes is the
    formatter's once its fields are filled; every message it names is one this module prints.  It is a draft
    until the acceptance run: the values only that run can supply are marked ``<<...>>``."""
    sheet = (STUDY / "stage2" / "tb" / "colab_steps.txt").read_text(encoding="ascii")          # plain ASCII
    rows = sheet.splitlines()
    assert all(len(r) <= 115 or r.strip().startswith("https://colab.research.google.com/") for r in rows)
    assert "\t" not in sheet and "DRAFT" in sheet.split("\n\n")[1] and "STILL OPEN IN THIS DRAFT" in sheet
    for field in ("<<C1>>", "<<BUNDLE_ID>>", "<<HASH12>>", "<<N_FILES>>", "<<G8A_STEPS>>", "<<BUDGET_STEPS", "<<RATE>>",
                  "<<FLOOR>>", "<<MIN_Y1>>", "<<MIN_Y3>>", "<<MIN_Y5>>", "<<MIN_Y6>>", "<<MIN_Y8>>", "<<MIN_LINE4>>",
                  "<<RULES_FILE", "<<RULES_SHA256>>", "<<BACKGROUND"):
        assert field in sheet, field                           # the commit, the bundle, the minutes: placeholders
    # every field of the text is listed again at the end, for the one who fills them in
    listed = sheet.split("STILL OPEN IN THIS DRAFT")[1]
    named = set(re.findall(r"<<([A-Z0-9_]+)", sheet.split("STILL OPEN IN THIS DRAFT")[0]))
    for name in sorted(named - {"LIKE", "TO"}):               # ("<<LIKE THIS>>"; section 8's "<<TO BE WRITTEN ...>>")
        assert f"<<{name}" in listed, name
    assert "section 8's first paragraph" in listed
    assert "PREREG_TRACKB.md" not in sheet                      # the rules file's name is a field: the text is a draft
    assert "<<RULES_FILE: THE COMMITTED RULES TEXT; TODAY STAGE2_TRACKB_PLAN.md>>" in sheet     # (ruling 5 of 17 Nov)
    # ruling 1 of 17 Nov decided what the sheet left open: the machine is kept, and the owner runs cell 8 again
    assert "RUNTIME_KEPT" not in sheet and "THIS MACHINE IS KEPT ON PURPOSE" in nb.KEEP_RUNTIME_TEXT
    fill = {"<<C1>>": COMMIT, "<<BUNDLE_ID>>": "tb-1", "<<HASH12>>": "0123456789ab", "<<N_FILES>>": "137",
            "<<G8A_STEPS>>": "6000", "<<RATE>>": "560", "<<W>>": "8", "<<VCPUS>>": "8", "<<BUDGET_M>>": "40",
            "<<HOURS>>": "19.8", "<RUN_ID>": "run-1", "<job>": "R5-75-M-H-10-s42", "<k>": "1", "<n>": "3000000",
            "<m>": "0", "<s>": "42"}

    def filled(text: str) -> str:
        for a, b in fill.items():
            text = text.replace(a, b)
        return text

    quoted, i = [], 0
    while i < len(rows):
        row = rows[i].strip()
        row = row[4:] if row[:1] == "(" and row[2:4] == ") " else row              # (a step's letter: "(d) ")
        if row.startswith(("S2 LINE ", "Run directory: ")):
            if not any(rx.match(filled(row)) for rx in nb.LINE_RE.values()):     # a line the sheet shows on two
                row, i = row + " " + rows[i + 1].strip(), i + 1
            quoted.append(filled(row))
        i += 1
    kinds = [next((k for k, rx in nb.LINE_RE.items() if rx.match(q)), None) for q in quoted]
    assert None not in kinds, [q for q, k in zip(quoted, kinds, strict=True) if k is None]
    assert kinds == [1, "new", 2, 3, "new", 4, "resume", "live"]
    assert quoted[0] == nb.line1(COMMIT, True, True)
    assert quoted[1] == nb.session_line("/content/drive/MyDrive/mesozoic-labs/stage2/runs/smoke-1", "new")
    assert quoted[2] == nb.line2("tb-1", "0123456789ab" + "0" * 52, 137, 137)
    assert quoted[3] == nb.line3("75-M-H-10", "V6", 2, 2, 6000)
    assert quoted[5] == nb.line4("R5-75-M-H-10-s42", 560, 8, 8, 8, 40_000_000)
    assert quoted[6] == nb.session_line("/content/drive/MyDrive/mesozoic-labs/stage2/runs/run-1", "resume", session=1,
                                        job="R5-75-M-H-10-s42", steps=3000000, done=0)
    assert quoted[7] == nb.session_line("/content/drive/MyDrive/mesozoic-labs/stage2/runs/run-1", "live", age_s=42)
    smoke = [r.strip() for r in rows if r.strip().startswith("S2 SMOKE Y")]
    assert [x.split(": ...: ")[0] for x in smoke] == [f"S2 SMOKE Y{i} {name}" for i, name in nb.SMOKE_ITEMS.items()]
    assert [x.split(": ...: ")[1].split()[0] for x in smoke] == ["OK", "REPORTED", "OK", "OK", "REPORTED", "OK",
                                                               "REPORTED", "REPORTED"]
    # every message the sheet names is printed by this module or by the notebook (the same words in both)
    source = (STUDY / "designlab" / "stage2" / "nb.py").read_text() + NOTEBOOK.read_text(encoding="utf-8")
    for said in ("S2 REFUSED", "S2 launch: the lines so far:", "the queue is finished", "S2 FINISH ", "S2 launch: ",
                 "the smoke is written (smoke/SMOKE.json)", "crashed twice", "counted without a record",
                 "S2 WARNING: a copy to Drive failed", "S2 FAILED", "G8a fails: no run", "BELOW FLOOR",
                 "no job is queued until the owner has set", "the rebuild is an allowed fallback",
                 "not made", "not finished", "Google Drive mounted. Stage 2 files are under:",
                 "Disconnecting runtime in 5 seconds...", "Repository ref: ", "S2 launch: UNFINISHED: ",
                 "NOT ON DRIVE: ", "was not started", "reading from Drive failed", "not on Drive after",
                 "S2 launch: interrupted by signal ", "stopped with exit code", "S2 launch: REFUSED: ",
                 "THIS MACHINE IS KEPT ON PURPOSE", "is resumed only on the lead's word"):
        assert said in source and said in sheet, said
    assert 'notes.append("LEAD.json: "' in source and '"S2 NOTE: LEAD.json: ..."' in sheet     # (a session's note)
    assert "RuntimeError: S2 step ... stopped with exit code" in sheet
    assert "S2 step {step!r} stopped with exit code {code}" in source          # (the cell's own RuntimeError)
    # the review of 16 Nov and ruling 5 of 17 Nov: what only Colab does is marked "expected, not yet seen" (no
    # "inferred" is left); the resume's two traps; QUEUE is a text field without quotation marks
    assert "inferred" not in sheet.lower() and sheet.count("(expected, not yet seen)") >= 10
    for row in ("You can close the tab; the run goes on.", "A lost browser does not end a run (expected, not yet seen"):
        assert any(row in r and "(expected, not yet seen" in r for r in rows), row
    example = sheet.split("for example\n", 1)[1].splitlines()[0].strip()
    assert [r["entry"] for r in nb.parse_queue(example)] == ["R5:75-M-H-10:42", "R6:75-M-H-10:42"] and '"' not in example
    assert "QUEUE                 (leave it empty)" in sheet and "no quotation marks" in sheet
    for word in ("Wait ten\n        minutes, then do 5.4 again", "stay with the page until you have seen line (d)",
                 'It is fine only if the line that begins "Run directory" said "from 0 steps"',
                 "Its two numbers are the trainer's own", "leave the old one alone"):
        assert (word in sheet) == (word != "leave the old one alone"), word
    for word in ("Never change RUN_ID or REPO_REF", "never set TAKEOVER", "(expected, not yet seen)", "stop and tell me",
                 "s2-launch-1/hardware/trex_form_study/stage2/stage2_training.ipynb", "High-RAM", "220 hours"):
        assert word in sheet, word
    # the two parameter tables (3.4, the smoke; 4.3, a launch) name exactly the boxes of the notebook's cell 1: the
    # owner is never asked to set a remedy (the lead's ruling 2 of 17 Nov: they are the lead's, in LEAD.json)
    for start, end in (("   3.4  ", "   3.5  "), ("   4.3  ", "   4.4  ")):
        table = sheet.split(start, 1)[1].split(end, 1)[0]
        assert re.findall(r"^ {10}([A-Z][A-Z_]+) {2,}", table, re.M) == list(nb.PARAMS), start
    # no instruction asks the owner to change RUN_ID, REPO_REF, TAKEOVER or an ALLOW parameter on an error's advice:
    # outside the two parameter tables each of them appears only with "not", "never", "same" or "do not"
    for n, r in enumerate(rows):
        if any(w in r for w in ("TAKEOVER", "ALLOW_", "ALLOW box")) and "not ticked" not in r:
            near = " ".join(rows[max(0, n - 1): n + 2]).lower()
            assert any(w in near for w in ("never", "do not", "not for you")), r


def test_helpers_equal_the_modules_they_stand_in_for():
    s2io = pytest.importorskip("designlab.stage2.s2io")
    doc = dict(b=[1, 2.5, None, True, "x"], a=dict(z="é", y=[]), c=1.0, d=-0.0)
    assert nb.sha256_json(doc) == s2io.sha256_json(doc)
    train2 = pytest.importorskip("designlab.stage2.train2")
    assert nb.job_name("R5", "75-M-H-10-X", 42) == train2.job_name("R5", "75-M-H-10-X", 42) == "R5-75-M-H-10-X-s42"
    assert (nb.TRAIN_DONE, nb.TRAIN_G8A, nb.TRAIN_REFUSED, nb.TRAIN_HALT) == (
        train2.EXIT_DONE, train2.EXIT_G8A, train2.EXIT_REFUSED, train2.EXIT_HALT)
    from designlab.stage2 import s2load

    have = [0, 1024, 2048, 3072, 4096, 5120]
    assert nb.keep_on_drive(have, [2048, 4096]) == s2load.keep(have, 2048)
    q = nb.parse_queue(["R5:75-M-H-10:42", "R6:75-M-H-10-X:7"])
    assert [r["job"] for r in q] == ["R5-75-M-H-10-s42", "R6-75-M-H-10-X-s7"]
    for bad in (["R7:b:1"], ["R5:b"], ["R5:b:x"], ["R5:b:1", "R5:b:1"], "R5", [["R5", "b", 1]]):
        with pytest.raises(nb.Refused):
            nb.parse_queue(bad)
    with pytest.raises(nb.Refused, match="unknown parameters"):
        nb.load_params(dict(RUNID="x"))


# ------------------------------------------------------------------------------------------------
# session: rules K6 and K7
# ------------------------------------------------------------------------------------------------
def test_a_session_opens_a_new_run(tmp_path, capsys):
    p = _params(tmp_path)
    s = _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    assert s["line"] == f"Run directory: {d} (new run)" in capsys.readouterr().out
    q = json.loads((d / "QUEUE.json").read_text())
    assert q["queue"] == p["QUEUE"] and q["commit"] == q["repo_ref"] == COMMIT and q["set"] == "S1" and q["tiny"]
    assert q["budget_steps"] == 0 and q["bundle"]["id"] is None and "rule K6" in q["rules"]["k6"]
    doc = json.loads((d / "SESSION_0.json").read_text())
    for key in ("run_id", "session", "session_uuid", "started_at", "last_status_at", "resume", "takeover", "repo",
                "runtime", "versions", "torch_build", "pins", "pins_match", "python_minor_matches", "ffmpeg", "bundle",
                "queue", "budget_steps", "workers", "allow_version_change", "allow_source_change", "setup_seconds",
                "params"):
        assert key in doc, key                                                    # design 3.9's schema
    assert doc["schema"] == "s2-session-2" and doc["session"] == 0 and doc["resume"] is False and doc["workers"] == 2
    assert set(doc["repo"]) >= {"ref", "commit", "dirty"} and doc["pins_match"] is True and doc["last_status_at"]
    assert set(doc["runtime"]) >= {"in_colab", "vcpus", "affinity", "cpu_model", "ram_gb", "disk_free_gb", "gpu"}
    assert set(doc["bundle"]) >= {"id", "content_sha256", "files_ok", "source"} and doc["params"]["RUN_ID"] == "r1"
    live = json.loads((d / "LIVE.json").read_text())
    assert live["session_uuid"] == doc["session_uuid"] and live["host"] and "closed_at" not in live
    status = json.loads((d / "STATUS.json").read_text())
    assert status["schema"] == "s2-status-2" and status["state"] == "running" and status["lines"] == [s["line"]]
    assert [e["state"] for e in status["queue"]] == ["queued", "queued"]
    # the session's own state is under work/ of the local run directory, which collect never copies
    assert (tmp_path / "local" / "r1" / "work" / "session.json").is_file()


def test_session_refusals(tmp_path, capsys):
    p = _params(tmp_path)
    d = tmp_path / "drive" / "runs" / "r1"
    # a REPO_REF that is no commit: a branch, a short hash, another commit than the checkout's
    for ref in ("main", COMMIT[:12], "d" * 40, ""):
        with pytest.raises(nb.Refused, match="REPO_REF"):
            _session(p, REPO_REF=ref)
    with pytest.raises(nb.Refused, match="BUDGET_STEPS"):
        _session(p, TINY=False)
    with pytest.raises(nb.Refused, match="nothing to resume"):
        _session(p, RESUME=True)
    assert not d.exists()                                                         # a refusal writes nothing
    _session(p)
    before = sorted(x.name for x in d.iterdir())
    with pytest.raises(nb.Refused, match="exists, and only a resume re-enters a run"):
        _session(p)                                                               # an existing RUN_ID without RESUME
    # RESUME while LIVE.json is fresh: the third form of the session line, nothing written
    with pytest.raises(nb.Refused) as exc:
        _session(p, RESUME=True)
    assert exc.value.line and nb.LINE_RE["live"].match(str(exc.value)) and exc.value.code == nb.EXIT_REFUSED
    # RESUME with another queue, set, bundle or budget (rule K6): refused whatever LIVE.json says
    for over, word in ((dict(QUEUE=["R5:B1:42"]), "queue"), (dict(QUEUE=["R6:B1:42", "R5:B1:42"]), "queue"),
                       (dict(BUNDLE_ID="tb-2"), "bundle_id"), (dict(BUDGET_STEPS=8192), "budget_steps"),
                       (dict(SET="V6"), "set"), (dict(MONITOR_EVERY=500000), "monitor_every")):
        with pytest.raises(nb.Refused, match=f"RESUME with other settings.*{word}"):
            _session(p, RESUME=True, TAKEOVER=True, **over)
    # another commit only with allow_source_change (LEAD.json, or a parameter file of the lead's own launches)
    with pytest.raises(nb.Refused, match="only with allow_source_change, which the lead writes in LEAD.json"):
        _session(p, RESUME=True, TAKEOVER=True, REPO_REF="d" * 40, commit="d" * 40)
    assert sorted(x.name for x in d.iterdir()) == before                          # none of them wrote anything
    # TAKEOVER on the lead's word: accepted and recorded
    s1 = _session(p, RESUME=True, TAKEOVER=True)
    assert s1["session"] == 1 and s1["takeover"] and s1["took_over"]["session"] == 0
    assert nb.LINE_RE["resume"].match(s1["line"]).group("job") == "R5-B1-s42"
    assert json.loads((d / "LIVE.json").read_text())["session_uuid"] == s1["session_uuid"]
    # a LIVE.json older than ten minutes lets a RESUME through without TAKEOVER
    live = json.loads((d / "LIVE.json").read_text())
    live["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - nb.LIVE_FRESH_S - 5))
    (d / "LIVE.json").write_text(json.dumps(live))
    s2 = _session(p, RESUME=True)
    assert s2["session"] == 2 and not s2["takeover"] and s2["took_over"] is None
    # ALLOW_SOURCE_CHANGE: another commit accepted, flagged, and the run's REPO_REF kept
    s3 = _session(p, RESUME=True, TAKEOVER=True, ALLOW_SOURCE_CHANGE=True, REPO_REF="d" * 40, commit="d" * 40)
    assert s3["source_change"] and s3["repo"]["commit"] == "d" * 40 and s3["repo"]["run_commit"] == COMMIT
    assert s3["repo"]["ref"] == COMMIT and "S2 NOTE: allow_source_change: " in capsys.readouterr().out
    # a finished run is not resumed, with DONE.json and with a HALT.json that is not marked resumable
    for name in ("DONE.json", "HALT.json"):
        (d / name).write_text(json.dumps(dict(reason="over")))
        with pytest.raises(nb.Refused, match=f"the run is finished \\({name}: over\\)"):
            _session(p, RESUME=True, TAKEOVER=True)
        (d / name).unlink()
    # a smoke is not resumed; a session closed by its own step does not hold a resume back
    with pytest.raises(nb.Refused, match="smoke"):
        _session(p, RESUME=True, SMOKE=True)
    st = nb._open_state("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    nb._close(st, "line 2 failed")
    assert _session(p, RESUME=True)["session"] == 4


def test_the_part_sweep(tmp_path, capsys):
    p = _params(tmp_path)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    ck = d / "jobs" / "R5-B1-s42" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "model_1024.zip.part").write_bytes(b"cut")
    (d / "STATUS.json.part").write_text("{")
    (ck / "model_1024.zip").write_bytes(b"whole")
    with pytest.raises(nb.Refused):                                               # a refused session sweeps nothing:
        _session(p, RESUME=True)                                                  # the parts may be a live copy's
    assert (ck / "model_1024.zip.part").is_file()
    s = _session(p, RESUME=True, TAKEOVER=True)
    assert sorted(s["swept_parts"]) == ["STATUS.json.part", "jobs/R5-B1-s42/checkpoints/model_1024.zip.part"]
    assert not list(d.rglob("*.part")) and (ck / "model_1024.zip").read_bytes() == b"whole"
    assert "removed 2 stale .part files" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------
# The launcher on the stub trainer
# ------------------------------------------------------------------------------------------------
def test_a_queue_runs_to_done(tmp_path, stubs, monkeypatch, capsys):
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(traces=True)}))
    p = _params(tmp_path)
    _session(p)
    capsys.readouterr()
    res = _launch(p)
    out = capsys.readouterr().out
    d, lrun = tmp_path / "drive" / "runs" / "r1", tmp_path / "local" / "r1"
    assert res["state"] == "done" and res["record"] == "DONE.json", res
    done = json.loads((d / "DONE.json").read_text())
    assert done["state"] == "done" and list(done["jobs"]) == ["R5-B1-s42", "R6-B1-s42"] and done["session"] == 0
    assert all(j["state"] == "done" and j["steps"] == 4096 for j in done["jobs"].values())
    assert not (d / "HALT.json").exists() and "closed_at" in json.loads((d / "LIVE.json").read_text())
    # the trainer's command line (design 3.6) and its environment
    calls = _calls(p)
    assert [c["job"] for c in calls] == ["R5-B1-s42", "R6-B1-s42"]
    argv = calls[0]["argv"]
    assert argv[:3] == ["run", "--mode", "R5"] and "--tiny" in argv and "--resume" not in argv and "--steps" not in argv
    assert argv[argv.index("--out") + 1] == str(lrun) and argv[argv.index("--workers") + 1] == "2"
    assert argv[argv.index("--run-id") + 1] == "r1" and argv[argv.index("--set") + 1] == "S1"
    assert calls[0]["mujoco_gl"] == "disable" and calls[0]["hashseed"] == "0" and calls[0]["cwd"] == str(STUDY)
    assert calls[0]["pgid"] == calls[0]["pid"] != os.getpgid(0)                   # a process group of its own
    # line 4 once per job, from gates/rate.json; lines 1-3 are not the launch's
    lines = nb.lines_of(out)
    assert [nb.LINE_RE[4].match(x).group("job") for x in lines[4]] == ["R5-B1-s42", "R6-B1-s42"]
    assert nb.LINE_RE[4].match(lines[4][0]).group("rate") == "500" and "BELOW FLOOR" not in lines[4][0]
    status = json.loads((d / "STATUS.json").read_text())
    assert status["state"] == "done" and status["lines"][-2:] == lines[4] and status["run_uuids"] == {
        "R5-B1-s42": "uuid-R5-B1-s42", "R6-B1-s42": "uuid-R6-B1-s42"}
    entry = status["queue"][0]
    for key in ("job", "state", "steps", "budget", "env_steps_per_s", "eta_h", "last_checkpoint",
                "last_checkpoint_copied", "last_monitor", "g8a", "g8b", "fix", "flags", "minibatch_steps_last", "retries",
                "last_error"):
        assert key in entry, key                                                  # design 3.9's schema
    assert entry["state"] == "done" and entry["last_checkpoint_copied"] == 4096 and entry["retries"] == 0
    assert set(entry["last_monitor"]) >= {"at_steps", "N", "S", "mean_clock_factor", "gain", "counted_without_record"}
    assert entry["checks_without_record"] == [] and entry["halt"] is None        # the trainer's three later keys
    assert set(status["runtime"]) == {"vcpus", "ram_gb", "disk_free_gb"} and entry["minibatch_steps_last"] == [160, 160]
    # the copies: text records, a tiny run's checkpoints (never step 0), each check's tar, the clip, the logs
    for job in ("R5-B1-s42", "R6-B1-s42"):
        jd = d / "jobs" / job
        for rel in ("run.json", "progress.json", "updates.jsonl", "gates/rate.json", "monitor/monitor_2048.json",
                    "monitor/monitor_4096.json"):
            assert (jd / rel).read_bytes() == (lrun / "jobs" / job / rel).read_bytes(), rel
        have = sorted(int(x.name[11:-5]) for x in (jd / "checkpoints").glob("checkpoint_*.json"))
        assert 0 not in have and {2048, 4096} <= set(have) and all(_reads_back(jd, n) for n in have)
        assert (jd / "ckpt_2048.tar").is_file() and (jd / "ckpt_4096.tar").is_file()
        assert not (jd / "monitor" / "traces").exists()                           # traces stay local
        clip = json.loads((jd / "monitor" / "clip_4096.json").read_text())
        assert clip["ok"] and clip["mujoco_gl"] == "osmesa" and clip["platform"] == "osmesa"
        assert clip["trace"].endswith("walk/v0.4/6004.npz")                       # the check's walk/v0.4 episode, 6004
        assert (jd / "monitor" / "clip_4096.mp4").read_bytes() == b"mp4 of trace 4096"
    assert (d / "logs" / "launcher.s0.log").is_file() and "done at 4096 steps" in (d / "logs" / "R5-B1-s42.s0.log").read_text()
    assert "    R5-B1-s42: done at 4096 steps" in out                             # the trainer's lines, indented
    assert not list(d.rglob("*.part")) and not list(d.rglob("*.tmp"))
    sess = json.loads((d / "SESSION_0.json").read_text())
    assert sess["last_status_at"] and sess["ended"]["state"] == "done" and sess["seams"]["trainer"]
    # a further launch, and a further session, find the run finished
    assert _launch(p)["state"] == "refused"
    with pytest.raises(nb.Refused, match="finished"):
        _session(p, RESUME=True)
    fin = nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    assert fin["state"] == "done" and "S2 FINISH r1: DONE" in capsys.readouterr().out
    # runtime-hours: the one session, from started_at to last_status_at
    rec = nb.hours(tmp_path / "drive" / "runs", out=tmp_path / "hours.json")
    assert list(rec["runs"]) == ["r1"] and rec["cap_hours"] == 220.0 and 0 <= rec["total_hours"] < 0.01


def test_with_drive_equal_to_local_nothing_is_copied(tmp_path, stubs):
    p = _params(tmp_path, LOCAL_DIR=str(tmp_path / "drive"), QUEUE=["R5:B1:42"])
    _session(p)
    res = _launch(p)
    d = tmp_path / "drive" / "runs" / "r1"
    assert res["state"] == "done" and (d / "jobs" / "R5-B1-s42" / "run.json").is_file() and (d / "DONE.json").is_file()
    assert _calls(dict(p, LOCAL_DIR=str(tmp_path / "drive" / "runs")))[0]["argv"][10] == str(d)   # --out: the run directory
    assert not list(d.rglob("ckpt_*.tar")) and (d / "work" / "session.json").is_file()


def test_the_copy_cadence_and_the_delete_one_cycle_later_rule(tmp_path, inproc_stubs):
    p = _params(tmp_path, TINY=False, BUDGET_STEPS=40_000_000, QUEUE=["R5:B1:42"])
    _session(p)
    st = nb._open_state("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    d, lrun = Path(st["drive_run"]), Path(st["local_run"])
    job = "R5-B1-s42"
    jd, dd = lrun / "jobs" / job, d / "jobs" / job
    (jd / "checkpoints").mkdir(parents=True)
    (jd / "monitor").mkdir()
    (jd / "run.json").write_text(json.dumps(dict(status="running", settings=dict(
        tiny=False, n_envs=8, n_steps=1024, checkpoint_every=250_000, monitor_every=2_500_000))))

    def save(n):
        files = {}
        for kind, name in (("model", "model_%d.zip"), ("vecnormalize", "vecnormalize_%d.pkl"),
                           ("fingerprint", "fingerprint_%d.npz")):
            data = f"{kind} {n}".encode()
            (jd / "checkpoints" / (name % n)).write_bytes(data)
            files[kind] = dict(name=name % n, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        (jd / "checkpoints" / f"checkpoint_{n}.json").write_text(json.dumps(dict(steps=n, files=files)))

    def on_drive():
        return sorted(int(x.name[11:-5]) for x in (dd / "checkpoints").glob("checkpoint_*.json")) if dd.is_dir() else []

    def cycle():
        launcher.cycle += 1
        launcher.copy_checkpoints(job)
        return on_drive()

    launcher = nb._Launcher(st, json.loads((d / "QUEUE.json").read_text()), 2, False, 0.05)
    try:
        # a rollout is 8,192 steps: the checkpoint of the multiple 1,000,000 is at 1,007,616 (rule K1)
        assert nb.at_or_past(1_007_616, 1_000_000, 8192) and not nb.at_or_past(1_015_808, 1_000_000, 8192)
        assert not nb.at_or_past(0, 1_000_000, 8192) and nb.at_or_past(2_506_752, 2_500_000, 8192)
        for n in (0, 253_952, 507_904, 753_664):
            save(n)
        assert cycle() == []                                    # neither step 0 nor a checkpoint between the millions
        save(1_007_616)
        assert cycle() == [1_007_616]
        save(2_007_040)
        assert cycle() == [1_007_616, 2_007_040]
        save(2_506_752)                                         # a check's (the multiple of monitor_every)
        assert cycle() == [1_007_616, 2_007_040, 2_506_752]     # copied in this cycle: nothing is deleted yet
        assert (dd / "ckpt_2506752.tar").is_file() and not (dd / "ckpt_2007040.tar").exists()
        assert cycle() == [2_007_040, 2_506_752]                # one cycle later the older one goes
        save(3_006_464)
        assert cycle() == [2_007_040, 2_506_752, 3_006_464]
        save(4_005_888)
        # the newest copy does not read back (a cut upload): nothing is deleted on its word
        assert cycle() == [2_007_040, 2_506_752, 3_006_464, 4_005_888]
        whole = (dd / "checkpoints" / "model_4005888.zip").read_bytes()
        (dd / "checkpoints" / "model_4005888.zip").write_bytes(b"cut")
        assert cycle() == [2_007_040, 2_506_752, 3_006_464, 4_005_888]
        (dd / "checkpoints" / "model_4005888.zip").write_bytes(whole)
        assert cycle() == [2_506_752, 3_006_464, 4_005_888]     # the check's stays beside the newest two
        assert not (dd / "checkpoints" / "model_2007040.zip").exists()           # all four files went
        assert all(_reads_back(dd, n) for n in on_drive())
        # a checkpoint the trainer deletes while it is copied leaves nothing half on Drive
        save(5_005_312)
        (jd / "checkpoints" / "fingerprint_5005312.npz").unlink()
        assert cycle() == [2_506_752, 3_006_464, 4_005_888] and not list(dd.rglob("*5005312*"))
        # a monitor record makes any checkpoint a check's
        save(6_004_736)
        (jd / "monitor" / "monitor_6004736.json").write_text("{}")
        assert 6_004_736 in cycle() and (dd / "ckpt_6004736.tar").is_file()
        assert launcher.mine[job]["last_checkpoint_copied"] == 6_004_736
        # the fence: once LIVE.json names another session nothing more is copied or deleted
        live = json.loads((d / "LIVE.json").read_text())
        (d / "LIVE.json").write_text(json.dumps(dict(live, session=1, session_uuid="other")))
        save(7_004_160)
        before = on_drive()
        with pytest.raises(nb._Superseded):
            cycle()
        assert on_drive() == before and not list(dd.rglob("*7004160*"))
    finally:
        launcher.log.close()


def test_a_second_launcher_is_refused_and_a_superseded_one_halts(tmp_path, stubs, monkeypatch, capsys):
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.05, budget=512 * 400)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"])
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    box = {}
    t = threading.Thread(target=lambda: box.update(res=_launch(p)))
    t.start()
    try:
        deadline = time.time() + 30
        while not list((d / "jobs" / "R5-B1-s42").glob("checkpoints/checkpoint_*.json")) and time.time() < deadline:
            time.sleep(0.02)                                                      # (a checkpoint is on the stand-in Drive)
        pid = json.loads((d / "jobs" / "R5-B1-s42" / "progress.json").read_text())["pid"]
        assert nb._alive(pid)
        # a second session on the live stand-in Drive is refused: "still running", nothing to do
        other = dict(p, LOCAL_DIR=str(tmp_path / "local2"))
        with pytest.raises(nb.Refused) as exc:
            _session(other, RESUME=True)
        assert nb.LINE_RE["live"].match(str(exc.value)) and t.is_alive() and nb._alive(pid)
        # a takeover (another runtime: its own local directory): the first launcher ends its job and stops
        s1 = _session(other, RESUME=True, TAKEOVER=True)
        t.join(30)
        assert not t.is_alive()
    finally:
        if t.is_alive():
            nb._kill_group(json.loads((d / "jobs" / "R5-B1-s42" / "progress.json").read_text())["pid"])
            t.join(30)
    res = box["res"]
    assert res["state"] == "halted" and res["reason"] == "superseded by session 1" and not nb._alive(pid)
    # the run's HALT.json is not the superseded launcher's to write: the run goes on under session 1
    assert not (d / "HALT.json").exists() and not (d / "DONE.json").exists()
    mark = json.loads((d / "SUPERSEDED_0.json").read_text())
    assert mark["reason"] == "superseded by session 1" and mark["session"] == 0
    assert json.loads((tmp_path / "local" / "r1" / "HALT.json").read_text())["reason"] == "superseded by session 1"
    assert json.loads((d / "LIVE.json").read_text())["session_uuid"] == s1["session_uuid"]
    assert "superseded by session 1" in capsys.readouterr().out
    assert nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])["state"] == "superseded"
    assert _launch(p)["state"] == "refused"                                       # the old session launches no more
    # ... and session 1 resumes the job from the stand-in Drive and ends the run
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.0, budget=512 * 400)}))
    assert _launch(other)["state"] == "done"
    run = json.loads((d / "jobs" / "R5-B1-s42" / "run.json").read_text())
    assert run["status"] == "done" and run["sessions"][1]["resumed_from"] > 0


def test_job_rules(tmp_path, stubs, monkeypatch, capsys):
    def go(name, plan, queue=("R5:B1:42", "R6:B1:42")):
        p, res, d = _run(tmp_path / name, plan, monkeypatch, QUEUE=list(queue))
        return p, res, d, capsys.readouterr().out

    # a crashed job is retried once from its newest intact checkpoint, and the queue goes on
    p, res, d, out = go("once", {"R5-B1-s42": dict(crash_at=2048, crash_times=1, crash_code=9)})
    assert res["state"] == "done" and res["jobs"]["R5-B1-s42"]["retries"] == 1
    assert res["jobs"]["R5-B1-s42"]["exit_codes"] == [9, 0] and "retried once" in out
    calls = [c for c in _calls(p) if c["job"] == "R5-B1-s42"]
    assert len(calls) == 2 and "--resume" not in calls[0]["argv"] and "--resume" in calls[1]["argv"]
    assert json.loads((d / "jobs" / "R5-B1-s42" / "run.json").read_text())["sessions"][1]["resumed_from"] == 2048
    assert [c["job"] for c in _calls(p)][-1] == "R6-B1-s42"
    # a second crash halts the run: the next job never starts
    p, res, d, out = go("twice", {"R5-B1-s42": dict(crash_at=1024, crash_times=2, crash_code=1)})
    halt = json.loads((d / "HALT.json").read_text())
    assert res["state"] == "halted" and "crashed twice (exit codes 1, 1)" in halt["reason"] and "boom" in halt["reason"]
    assert halt["resumable"] is False and halt["halt_id"] is None and "The run is over" in out   # (the one final halt)
    assert [c["job"] for c in _calls(p)] == ["R5-B1-s42"] * 2 and not (d / "DONE.json").exists()
    assert halt["jobs"]["R5-B1-s42"]["state"] == "failed" and halt["jobs"]["R6-B1-s42"]["state"] == "queued"
    assert json.loads((d / "STATUS.json").read_text())["state"] == "halted"
    (d / "LEAD.json").write_text(json.dumps(dict(resume_halt="anything", retry_check=True)))
    with pytest.raises(nb.Refused, match="finished"):                             # whatever the lead writes
        _session(p, RESUME=True)
    # a job stopped by a rule (exit code 0) lets the next start
    p, res, d, out = go("stopped", {"R5-B1-s42": dict(stop_at=2048)})
    assert res["state"] == "done" and res["jobs"]["R5-B1-s42"]["state"] == "stopped"
    assert res["jobs"]["R5-B1-s42"]["stopped"]["reason"].startswith("stub: three checks")
    assert res["jobs"]["R6-B1-s42"]["state"] == "done"
    # exit code 3 (G8a failed: no run) lets the next start too
    p, res, d, out = go("g8a", {"R5-B1-s42": dict(g8a_fail=True)})
    assert res["state"] == "done" and res["jobs"]["R5-B1-s42"]["state"] == "stopped" and "G8a fails: no run" in out
    assert res["jobs"]["R5-B1-s42"]["exit_codes"] == [3] and res["jobs"]["R6-B1-s42"]["state"] == "done"
    # exit code 4 (refused) ends the session with no HALT.json, exit code 5 (halted for the lead) halts the run,
    # resumable: neither is retried (test_a_refusal_of_the_trainer_writes_no_halt, test_a_halt_is_resumed_only_...)
    for name, plan, word, state in (("refused", dict(refuse=True), "was refused by the trainer (exit code 4)",
                                     "refused"),
                                    ("halted", dict(halt=True), "halted and goes to the lead (exit code 5)", "halted")):
        p, res, d, out = go(name, {"R5-B1-s42": plan})
        assert res["state"] == state and word in res["reason"] and len(_calls(p)) == 1, res
        assert res["jobs"]["R5-B1-s42"]["retries"] == 0 and res["jobs"]["R6-B1-s42"]["state"] == "queued"
        assert (d / "HALT.json").exists() is (state == "halted")
    # a check counted without a record is said once (the trainer's later progress key)
    p, res, d, out = go("told", {"*": dict(checks_without_record=[2048])}, queue=("R5:B1:42",))
    assert out.count("the check at 2048 steps was counted without a record") == 1
    assert json.loads((d / "STATUS.json").read_text())["queue"][0]["checks_without_record"] == [2048]


def _lead(d: Path, **doc) -> None:
    (d / "LEAD.json").write_text(json.dumps(dict(schema="s2-lead-1", **doc)))


def test_a_refusal_of_the_trainer_writes_no_halt(tmp_path, stubs, monkeypatch, capsys):
    """The lead's ruling 2 of 17 Nov (the review's major 2): the trainer's exit code 4 (here: other library versions
    than the run's, rule K4) writes no HALT.json; the session ends with LIVE.json closed, and the run is resumed
    with the lead's remedy from LEAD.json, which reaches the trainer.  As built, HALT.json ended the run and no
    resume was accepted, with any flag."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"R5-B1-s42": dict(refuse=True)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    res = _launch(p)
    out = capsys.readouterr().out
    assert res["state"] == "refused" and res["session"] == 0 and res["record"] is None, res
    assert "was refused by the trainer (exit code 4)" in res["reason"] and "S2 launch: REFUSED: job R5-B1-s42" in out
    assert not (d / "HALT.json").exists() and not (d / "DONE.json").exists() and len(_calls(p)) == 1
    live = json.loads((d / "LIVE.json").read_text())
    assert live["closed_reason"].startswith("refused: job R5-B1-s42 was refused by the trainer")
    assert json.loads((d / "SESSION_0.json").read_text())["ended"]["state"] == "refused"
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(dict(p, RESUME=True)))
    # the lead's remedy: LEAD.json beside the run; the owner sets RESUME only (LIVE.json is closed: no TAKEOVER)
    _lead(d, allow_version_change=True, note="other versions accepted on the lead's word")
    monkeypatch.setenv("STUB_PLAN", "{}")
    s1 = _session(p, RESUME=True)
    assert s1["allow_version_change"] is True and s1["allow_source_change"] is False and s1["takeover"] is False
    assert s1["lead"]["doc"]["allow_version_change"] is True and len(s1["lead"]["sha256"]) == 64
    assert json.loads((d / "SESSION_1.json").read_text())["lead"]["path"] == str(d / "LEAD.json")
    assert "S2 NOTE: LEAD.json: allow_version_change true" in capsys.readouterr().out
    assert nb.main(["launch", "--params", str(pfile)]) == 0
    argv = _calls(p)[-1]["argv"]          # (the stub refused before it wrote run.json: the job starts afresh)
    assert len(_calls(p)) == 2 and "--allow-version-change" in argv and "--allow-source-change" not in argv
    assert json.loads((d / "DONE.json").read_text())["state"] == "done"


def test_a_halt_is_resumed_only_with_the_leads_remedy(tmp_path, stubs, monkeypatch, capsys):
    """Ruling 2 of 17 Nov: the trainer's exit code 5 writes a HALT.json marked resumable, with its halt_id and the
    remedy it names (retry_check when the trainer names --retry-check); a resume is refused until LEAD.json names
    the halt and holds the remedy, and then the old halt is kept as HALT_<k>.json and the remedy reaches the trainer.
    As built every resume was refused ("the run is finished")."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02), "R5-B1-s42": dict(halt=True)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    res = _launch(p)
    out = capsys.readouterr().out
    halt = json.loads((d / "HALT.json").read_text())
    assert res["state"] == "halted" and res["record"] == "HALT.json" and res["resumable"] is True, res
    assert halt["resumable"] is True and halt["remedy"] == ["retry_check"] and halt["halt_id"].startswith("r1-s0-")
    assert halt["session"] == 0 and f"resumed only on the lead's word (LEAD.json; halt {halt['halt_id']})" in out
    assert "closed_at" in json.loads((d / "LIVE.json").read_text())
    # without the lead's word, or with half of it: refused, nothing changed
    for doc, want in ((None, f'"resume_halt": "{halt["halt_id"]}", "retry_check": true'),
                      (dict(resume_halt="r1-s0-other", retry_check=True), f'"resume_halt": "{halt["halt_id"]}"'),
                      (dict(resume_halt=halt["halt_id"]), '"retry_check": true'),
                      (dict(resume_halt=halt["halt_id"], retry_check=False), '"retry_check": true')):
        if doc is not None:
            _lead(d, **doc)
        with pytest.raises(nb.Refused, match="the run halted") as exc:
            _session(p, RESUME=True)
        assert f"LEAD.json beside the run with {want}" in str(exc.value)
        assert (d / "HALT.json").is_file() and not list(d.glob("HALT_*.json")) and not (d / "SESSION_1.json").exists()
    # RETRY_CHECK in a parameter file is no answer to the halt either: the halt names itself
    with pytest.raises(nb.Refused, match="the run halted"):
        _session(p, RESUME=True, RETRY_CHECK=True)
    _lead(d, resume_halt=halt["halt_id"], retry_check=True)
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02)}))
    s1 = _session(p, RESUME=True)
    assert s1["retry_check"] is True and s1["halt_resumed"]["halt_id"] == halt["halt_id"]
    assert s1["halt_resumed"]["kept_as"] == "HALT_0.json" and s1["halt_resumed"]["remedy"] == ["retry_check"]
    assert not (d / "HALT.json").exists() and json.loads((d / "HALT_0.json").read_text()) == halt
    assert "is answered by LEAD.json; it is kept as HALT_0.json" in capsys.readouterr().out
    res = _launch(p)
    assert res["state"] == "done" and "--retry-check" in _calls(p)[-1]["argv"] and "--resume" in _calls(p)[-1]["argv"]
    assert json.loads((d / "DONE.json").read_text())["state"] == "done"


def test_a_launcher_error_is_a_resumable_halt(tmp_path, stubs, monkeypatch, capsys):
    """Ruling 2 of 17 Nov: an error of the launcher itself writes a resumable halt whose remedy is the lead's word
    alone (LEAD.json's resume_halt)."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"

    def broken(self, job):
        raise KeyError("a fault of the launcher's own")

    real = nb._Launcher._line4
    monkeypatch.setattr(nb._Launcher, "_line4", broken)
    res = _launch(p)
    halt = json.loads((d / "HALT.json").read_text())
    assert res["state"] == "halted" and halt["reason"].startswith("launcher error: KeyError")
    assert halt["resumable"] is True and halt["remedy"] == [] and halt["halt_id"]
    monkeypatch.setattr(nb._Launcher, "_line4", real)
    with pytest.raises(nb.Refused, match=f'with "resume_halt": "{halt["halt_id"]}" \\(RESUMABLE_RULE\\)'):
        _session(p, RESUME=True)
    _lead(d, resume_halt=halt["halt_id"])
    assert _session(p, RESUME=True)["halt_resumed"]["kept_as"] == "HALT_0.json"
    assert _launch(p)["state"] == "done" and "--retry-check" not in _calls(p)[-1]["argv"]


def test_lead_json(tmp_path, capsys):
    """LEAD_RULE: what LEAD.json may hold; budget_by and budget_set_on go to QUEUE.json when the run is made (rule
    Y10's Check, which the notebook could not set on Colab before); every value is recorded in the session."""
    p = _params(tmp_path)
    d = tmp_path / "drive" / "runs" / "r1"
    d.mkdir(parents=True)
    for doc, word in ((dict(allow_sorce_change=True), "'allow_sorce_change' is not a key of LEAD.json"),
                      (dict(retry_check="yes"), "retry_check must be true or false"),
                      (dict(budget_set_on="20 Nov"), "budget_set_on must be a UTC time"),
                      (dict(budget_by=1), "budget_by must be a text")):
        _lead(d, **doc)
        with pytest.raises(nb.Refused, match=re.escape(word)):
            _session(p)
        assert not (d / "QUEUE.json").exists()
    (d / "LEAD.json").write_text("{")
    with pytest.raises(nb.Refused, match="not a JSON object"):
        _session(p)
    _lead(d, budget_by="owner", budget_set_on="2026-11-20T09:00:00Z", note="Y4 failed; the owner set the budget")
    s = _session(p)
    q = json.loads((d / "QUEUE.json").read_text())
    assert q["budget_by"] == "owner" and q["budget_set_on"] == "2026-11-20T09:00:00Z"
    assert s["budget_by"] == "owner" and s["lead"]["doc"]["note"].startswith("Y4 failed")
    assert s["allow_version_change"] is False and s["allow_source_change"] is False and s["retry_check"] is False
    assert "lead" in s["rules"] and "resumable" in s["rules"]


def test_launch_arguments_must_be_the_queues(tmp_path, inproc_stubs, capsys):
    p = _params(tmp_path, MONITOR_EVERY=1024, MONITOR_ALSO_K=[0], TINY=False, BUDGET_STEPS=2048, LR=3e-5,
                CONFIG=dict(filter_hz=10.0), QUEUE=["R5:B1:42"])
    _session(p)
    assert _launch(dict(p, MONITOR_EVERY=2048))["state"] == "refused"
    assert _launch(dict(p, BUDGET_STEPS=4096))["state"] == "refused" and "REFUSED launch" in capsys.readouterr().out
    assert not _calls(p)
    assert _launch(p)["state"] == "done"
    argv = _calls(p)[0]["argv"]
    # the two monitor options, the budget and the trainer's other settings go to every train2 run, from QUEUE.json
    assert argv[argv.index("--monitor-every") + 1] == "1024" and argv[argv.index("--monitor-also-k") + 1] == "0"
    assert argv[argv.index("--steps") + 1] == "2048" and "--tiny" not in argv
    assert argv[argv.index("--lr") + 1] == "3e-05" and json.loads(argv[argv.index("--config") + 1]) == dict(filter_hz=10.0)
    q = json.loads((tmp_path / "drive" / "runs" / "r1" / "QUEUE.json").read_text())
    assert q["monitor_every"] == 1024 and q["monitor_also_k"] == [0] and q["budget_steps"] == 2048


def test_a_clip_that_fails_or_hangs_is_recorded_and_ignored(tmp_path, stubs, monkeypatch):
    plan = {"*": dict(traces=True)}
    monkeypatch.setenv("STUB_CLIP", json.dumps(dict(fail=True)))
    p, res, d = _run(tmp_path / "fail", plan, monkeypatch, QUEUE=["R5:B1:42"])
    assert res["state"] == "done"
    assert json.loads((d / "STATUS.json").read_text())["queue"][0]["clip"]["reason"] == "no renderer"
    assert not list(d.rglob("*.mp4")) and json.loads((d / "jobs/R5-B1-s42/monitor/clip_4096.json").read_text())["ok"] is False
    monkeypatch.setenv("STUB_CLIP", json.dumps(dict(sleep=30)))
    monkeypatch.setattr(nb, "CLIP_LIMIT_S", 0.3)
    t0 = time.time()
    p, res, d = _run(tmp_path / "hang", plan, monkeypatch, QUEUE=["R5:B1:42"])
    clip = json.loads((d / "STATUS.json").read_text())["queue"][0]["clip"]
    assert res["state"] == "done" and clip["ok"] is False and "no clip within 0.3 s" in clip["reason"]
    assert time.time() - t0 < 20 and json.loads((d / "jobs/R5-B1-s42/monitor/clip_4096.json").read_text())["schema"] == "s2-clip-launcher-1"
    monkeypatch.delenv("STUB_CLIP")
    p, res, d = _run(tmp_path / "off", plan, monkeypatch, QUEUE=["R5:B1:42"], CLIPS=False)
    assert res["state"] == "done" and not list(d.rglob("clip_*"))


def test_a_killed_launcher_is_resumed_from_the_stand_in_drive(tmp_path, stubs, monkeypatch, capsys):
    """The launcher and its job die together (a dead runtime); the local directory is lost; a new session resumes:
    the finished job is skipped, the cut job is copied back from the stand-in Drive and resumed."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"R6-B1-s42": dict(tick=0.05, budget=512 * 200)}))
    p = _params(tmp_path)
    _session(p)
    d, lrun = tmp_path / "drive" / "runs" / "r1", tmp_path / "local" / "r1"
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(p))
    env = dict(os.environ, PYTHONPATH=".")
    proc = subprocess.Popen([sys.executable, "-m", "designlab.stage2.nb", "launch", "--params", str(pfile)], cwd=STUDY,
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        ck = d / "jobs" / "R6-B1-s42" / "checkpoints"
        deadline = time.time() + 60
        while time.time() < deadline and not (ck.is_dir() and list(ck.glob("checkpoint_*.json"))):
            assert proc.poll() is None, "the launcher ended before its second job's first copy"
            time.sleep(0.02)
        pids = nb.session_pids(proc.pid)
        assert len(pids) >= 2                                                     # the launcher and its trainer
    finally:
        nb.kill_session(proc.pid)
        proc.wait()
    assert not nb.session_pids(proc.pid) and not (d / "DONE.json").exists()
    on_drive = sorted(int(x.name[11:-5]) for x in ck.glob("checkpoint_*.json"))
    shutil.rmtree(lrun)                                                           # a dead runtime loses its local disk
    (ck / "model_999.zip.part").write_bytes(b"a copy that was cut")
    with pytest.raises(nb.Refused):                                               # LIVE.json is fresh
        _session(p, RESUME=True)
    capsys.readouterr()
    s1 = _session(p, RESUME=True, TAKEOVER=True)
    m = nb.LINE_RE["resume"].match(s1["line"])
    assert m.group("job") == "R6-B1-s42" and int(m.group("steps")) == max(on_drive) and m.group("done") == "1"
    assert s1["swept_parts"] == ["jobs/R6-B1-s42/checkpoints/model_999.zip.part"]
    monkeypatch.setenv("STUB_PLAN", json.dumps({"R6-B1-s42": dict(tick=0.0, budget=512 * 200)}))
    res = _launch(p)
    out = capsys.readouterr().out
    assert res["state"] == "done" and "R5-B1-s42 is finished (done): skipped" in out and "copied back from Drive" in out
    calls = _calls(p)                                                             # (the new local directory's)
    assert [c["job"] for c in calls] == ["R6-B1-s42"] and "--resume" in calls[0]["argv"]
    run = json.loads((d / "jobs" / "R6-B1-s42" / "run.json").read_text())
    assert run["status"] == "done" and run["sessions"][-1]["resumed_from"] == max(on_drive) > 0
    assert len(nb.lines_of(out)[4]) == 1                                          # line 4 again, for the session
    assert json.loads((d / "SESSION_1.json").read_text())["resume"] is True and (d / "DONE.json").is_file()
    rec = nb.hours(tmp_path / "drive" / "runs", out=None)
    assert [s["session"] for s in rec["runs"]["r1"]["sessions"]] == [0, 1]


def test_a_launcher_killed_while_a_copy_is_in_progress(tmp_path, stubs, monkeypatch):
    """The acceptance run's kill point (design 3.10 step 4): a ``.part`` file exists on the stand-in Drive.  The
    cut checkpoint has no record there; the resume sweeps the part and goes on from what Drive holds whole."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.05, budget=512 * 400)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"], COPY_PAUSE_S=0.4)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    ck = d / "jobs" / "R5-B1-s42" / "checkpoints"
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(p))
    proc = subprocess.Popen([sys.executable, "-m", "designlab.stage2.nb", "launch", "--params", str(pfile)], cwd=STUDY,
                            env=dict(os.environ, PYTHONPATH="."), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not (ck.is_dir() and list(ck.glob("*.part"))):
            assert proc.poll() is None
            time.sleep(0.005)
    finally:
        nb.kill_session(proc.pid)
        proc.wait()
    parts = sorted(x.name for x in ck.glob("*.part"))
    assert len(parts) == 1, parts
    cut = int("".join(c for c in parts[0] if c.isdigit()))
    assert not (ck / f"checkpoint_{cut}.json").exists()                           # the record is written last
    whole = sorted(int(x.name[11:-5]) for x in ck.glob("checkpoint_*.json"))
    shutil.rmtree(tmp_path / "local" / "r1")
    s1 = _session(dict(p, COPY_PAUSE_S=None), RESUME=True, TAKEOVER=True)
    assert s1["swept_parts"] == [f"jobs/R5-B1-s42/checkpoints/{parts[0]}"] and not list(d.rglob("*.part"))
    assert int(nb.LINE_RE["resume"].match(s1["line"]).group("steps")) == max(whole, default=0)
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.0, budget=512 * 400)}))
    assert _launch(dict(p, COPY_PAUSE_S=None))["state"] == "done"
    run = json.loads((d / "jobs" / "R5-B1-s42" / "run.json").read_text())
    assert run["status"] == "done" and run["sessions"][-1]["resumed_from"] == (max(whole) if whole else None)
    assert all(_reads_back(d / "jobs" / "R5-B1-s42", n) for n in sorted(int(x.name[11:-5]) for x in ck.glob("checkpoint_*.json")))


def test_an_interrupted_launcher_leaves_the_run_resumable(tmp_path, stubs, monkeypatch, capsys):
    """SIGTERM to the launcher alone (an interrupted cell): it ends its job, writes no HALT.json and closes
    LIVE.json, so the run is resumed at once and without TAKEOVER."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.05, budget=512 * 400)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"])
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(p))
    proc = subprocess.Popen([sys.executable, "-m", "designlab.stage2.nb", "launch", "--params", str(pfile)], cwd=STUDY,
                            env=dict(os.environ, PYTHONPATH="."), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not list((d / "jobs" / "R5-B1-s42").glob("checkpoints/checkpoint_*.json")):
            assert proc.poll() is None
            time.sleep(0.02)
        pid = json.loads((d / "jobs" / "R5-B1-s42" / "progress.json").read_text())["pid"]
        proc.terminate()
        out = proc.communicate(timeout=60)[0]
    finally:
        nb.kill_session(proc.pid)
    assert proc.returncode == nb.EXIT_INTERRUPTED and "interrupted by signal 15" in out and not nb._alive(pid)
    assert not (d / "HALT.json").exists() and not (d / "DONE.json").exists()
    live = json.loads((d / "LIVE.json").read_text())
    assert live["closed_reason"] == "interrupted by signal 15" and json.loads((d / "STATUS.json").read_text())["state"] == "running"
    assert nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])["state"] == "unfinished"
    s1 = _session(p, RESUME=True)                                                 # at once, and without TAKEOVER
    assert s1["session"] == 1 and s1["took_over"] is None
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.0, budget=512 * 400)}))
    capsys.readouterr()
    assert _launch(p)["state"] == "done" and "started with --resume" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------
# Faults of Drive: a job's last copy, the copy back, the launcher's own output (the review of 16 Nov)
# ------------------------------------------------------------------------------------------------
def _failing_copies(monkeypatch, fails):
    """``nb._copy_file`` that raises ``OSError`` (EIO, as a mount that fails) whenever ``fails(src, dst)`` is true."""
    real = nb._copy_file

    def copy(src, dst, pause=0.0):
        if fails(Path(src), Path(dst)):
            raise OSError(errno.EIO, "Input/output error (the stand-in Drive fails)", str(dst))
        return real(src, dst, pause)

    monkeypatch.setattr(nb, "_copy_file", copy)


def _on_drive(jobdir: Path) -> list:
    ck = jobdir / "checkpoints"
    return sorted(int(x.name[11:-5]) for x in ck.glob("checkpoint_*.json")) if ck.is_dir() else []


def test_a_jobs_last_copy_is_tried_again(tmp_path, stubs, monkeypatch, capsys):
    """One failed copy of a job's final checkpoint at the job's end is no end of the matter: the copy is made again
    before the run is written DONE (as built, the end-of-job cycle was the only try and DONE.json named a last
    checkpoint that Drive did not hold)."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02)}))
    monkeypatch.setattr(nb, "END_COPY_PERIODS", 4.0)
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d, job = tmp_path / "drive" / "runs" / "r1", "R5-B1-s42"
    seen = dict(ended=False, after_end=0)
    real_end = nb._Launcher._end_child

    def end_child(self):  # (the launcher has seen its trainer end: what follows is the job's end-of-job cycle)
        seen["ended"] = True
        return real_end(self)

    def fails(src, dst):  # every copy of the final checkpoint fails while the job runs, and once more at its end
        if "_4096." not in src.name or "checkpoints" not in dst.parts or d not in dst.parents:
            return False
        if seen["ended"] and seen["after_end"] >= 1:
            return False
        seen["after_end"] += int(seen["ended"])
        return True

    monkeypatch.setattr(nb._Launcher, "_end_child", end_child)
    _failing_copies(monkeypatch, fails)
    res = _launch(p)
    out = capsys.readouterr().out
    assert seen["after_end"] == 1 and "S2 WARNING: a copy to Drive failed" in out
    assert res["state"] == "done" and res["record"] == "DONE.json", res
    done = json.loads((d / "DONE.json").read_text())["jobs"][job]
    assert done["last_checkpoint"] == done["last_checkpoint_copied"] == 4096 and done["not_on_drive"] == []
    assert 4096 in _on_drive(d / "jobs" / job) and _reads_back(d / "jobs" / job, 4096)
    assert (d / "jobs" / job / "ckpt_4096.tar").is_file() and "its last copies are on Drive now" in out
    assert json.loads((d / "jobs" / job / "run.json").read_text())["status"] == "done"
    fin = nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    assert fin["state"] == "done" and fin["on_drive"] is True and fin["not_on_drive"] == {}


def test_an_earlier_jobs_last_copy_is_tried_again_while_the_next_job_runs(tmp_path, stubs, monkeypatch, capsys):
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02), "R6-B1-s42": dict(tick=0.05)}))
    monkeypatch.setattr(nb, "END_COPY_PERIODS", 2.0)
    p = _params(tmp_path, CLIPS=False)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    second = tmp_path / "local" / "r1" / "jobs" / "R6-B1-s42" / "progress.json"
    # the first job's final checkpoint cannot be copied until the second job trains
    _failing_copies(monkeypatch, lambda src, dst: "R5-B1-s42" in dst.parts and "_4096." in src.name
                    and "checkpoints" in dst.parts and d in dst.parents and not second.is_file())
    res = _launch(p)
    out = capsys.readouterr().out
    assert re.search(r"S2 WARNING: R5-B1-s42: not on Drive after 3 further tries in \d+ s: checkpoint 4096", out)
    assert "S2 launch: R5-B1-s42: its last copies are on Drive now" in out
    assert out.index("not on Drive after 3 further tries") < out.index("R6-B1-s42 started") < out.index(
        "R5-B1-s42: its last copies are on Drive now")
    assert res["state"] == "done" and (d / "DONE.json").is_file(), res
    for job in ("R5-B1-s42", "R6-B1-s42"):
        assert _reads_back(d / "jobs" / job, 4096) and (d / "jobs" / job / "ckpt_4096.tar").is_file()
        assert json.loads((d / "jobs" / job / "run.json").read_text())["status"] == "done"
        assert res["jobs"][job]["last_checkpoint_copied"] == 4096 and res["jobs"][job]["not_on_drive"] == []


def test_no_done_without_a_jobs_last_checkpoint_on_drive(tmp_path, stubs, monkeypatch, capsys):
    """A final checkpoint that never reaches Drive: the launch ends unfinished (no DONE.json, no HALT.json), on
    Drive the job does not read finished, ``finish`` names what is on this machine's disk only, and a resume on
    the same machine makes the copy and ends the run without training again."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.05)}))
    monkeypatch.setattr(nb, "END_COPY_PERIODS", 2.0)
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d, job = tmp_path / "drive" / "runs" / "r1", "R5-B1-s42"
    broken = dict(on=True)
    _failing_copies(monkeypatch, lambda src, dst: broken["on"] and "_4096." in src.name
                    and "checkpoints" in dst.parts and d in dst.parents)
    res = _launch(p)
    out = capsys.readouterr().out
    assert res["state"] == "unfinished" and res["record"] is None, res
    assert "checkpoint 4096" in res["reason"] and "S2 launch: UNFINISHED: " in out and "S2 launch: DONE" not in out
    assert not (d / "DONE.json").exists() and not (d / "HALT.json").exists()
    assert _on_drive(d / "jobs" / job) and 4096 not in _on_drive(d / "jobs" / job)
    # on Drive the job never reads finished before its last checkpoint is there: a new runtime would resume it
    assert json.loads((d / "jobs" / job / "run.json").read_text())["status"] == "running"
    assert json.loads((tmp_path / "local" / "r1" / "jobs" / job / "run.json").read_text())["status"] == "done"
    status = json.loads((d / "STATUS.json").read_text())
    assert status["queue"][0]["not_on_drive"] == ["checkpoint 4096"] and status["state"] == "running"
    assert status["queue"][0]["last_checkpoint"] == 4096 and status["queue"][0]["last_checkpoint_copied"] == 3072
    live = json.loads((d / "LIVE.json").read_text())
    assert live["closed_reason"].startswith("unfinished: ")
    assert json.loads((d / "SESSION_0.json").read_text())["ended"]["state"] == "unfinished"
    # the steps' exit codes: 6 for a launch that ended unfinished and for a finish that finds something off Drive
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(p))
    assert nb.main(["launch", "--params", str(pfile)]) == 6          # (the same session again: still not on Drive)
    fin = nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    assert fin["state"] == "unfinished" and fin["on_drive"] is False and fin["not_on_drive"] == {job: ["checkpoint 4096"]}
    assert nb.main(["finish", "--params", str(pfile)]) == 6
    assert "NOT ON DRIVE: R5-B1-s42: checkpoint 4096" in capsys.readouterr().out
    assert len(_calls(p)) == 1 and not (d / "HALT.json").exists()
    # Drive takes the file again: a resume on the same machine copies it and ends the run; nothing is trained again
    broken["on"] = False
    s1 = _session(p, RESUME=True)                                    # (LIVE.json was closed: no TAKEOVER)
    m = nb.LINE_RE["resume"].match(s1["line"])                       # (this machine's files say the job is done)
    assert s1["session"] == 1 and m.group("job") == "none" and m.group("done") == "1"
    res = _launch(p)
    out = capsys.readouterr().out
    assert res["state"] == "done" and "R5-B1-s42 is finished (done): skipped" in out and len(_calls(p)) == 1
    assert _reads_back(d / "jobs" / job, 4096) and (d / "jobs" / job / "ckpt_4096.tar").is_file()
    assert json.loads((d / "jobs" / job / "run.json").read_text())["status"] == "done"
    assert json.loads((d / "DONE.json").read_text())["jobs"][job]["last_checkpoint_copied"] == 4096
    assert nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])["on_drive"] is True


def test_a_drive_that_is_gone_for_good_ends_the_launch_unfinished(tmp_path, stubs, monkeypatch, capsys):
    """Every read and write under the stand-in Drive fails after the job's first copied checkpoint (a dead mount):
    the launch does not call the run done, and ``finish`` reports instead of raising."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02)}))
    monkeypatch.setattr(nb, "END_COPY_PERIODS", 2.0)
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    drive = (tmp_path / "drive").resolve()
    d, job = drive / "runs" / "r1", "R5-B1-s42"
    gone = dict(on=False)
    real_copy, real_write, real_read = nb._copy_file, nb._write_json, nb._read_json

    def dead(path):
        if gone["on"] and drive in Path(path).parents:
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected", str(path))

    def copy(src, dst, pause=0.0):
        dead(dst)
        real_copy(src, dst, pause)
        if Path(dst).name.startswith("checkpoint_") and drive in Path(dst).parents:
            gone["on"] = True                                        # the first checkpoint is on Drive: now it dies

    def write(path, obj):
        dead(path)
        return real_write(path, obj)

    def read(path):
        try:
            dead(path)
        except OSError:
            return None
        return real_read(path)

    monkeypatch.setattr(nb, "_copy_file", copy)
    monkeypatch.setattr(nb, "_write_json", write)
    monkeypatch.setattr(nb, "_read_json", read)
    res = _launch(p)
    out = capsys.readouterr().out
    assert gone["on"] and res["state"] == "unfinished" and res["record"] is None, res
    assert "S2 launch: UNFINISHED: " in out and "could NOT be written" in out and "S2 launch: DONE" not in out
    fin = nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])            # as built this raised OSError
    assert fin["state"] == "unfinished" and fin["on_drive"] is False
    assert "S2 FINISH r1: NOT ON DRIVE" in capsys.readouterr().out
    monkeypatch.setattr(nb, "_read_json", real_read)
    assert not (d / "DONE.json").exists() and not (d / "HALT.json").exists()
    assert json.loads((tmp_path / "local" / "r1" / "jobs" / job / "run.json").read_text())["status"] == "done"


def test_the_end_of_job_copies_wait_longer_each_time_for_at_most_20_minutes(tmp_path, stubs, monkeypatch, capsys):
    """The lead's ruling 1 of 17 Nov: at a job's end its missing copies are tried again, fenced and with a growing
    wait, for at most 20 minutes (as built: ten tries one period apart, ten minutes)."""
    assert nb.end_copy_waits(60.0) == [30.0, 60.0, 120.0, 240.0, 300.0, 300.0, 150.0]
    assert sum(nb.end_copy_waits(60.0)) == 1200.0 and sum(nb.end_copy_waits(0.05)) == pytest.approx(1.0)
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False, PERIOD_S=0.02)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    _failing_copies(monkeypatch, lambda src, dst: "_4096." in src.name and "checkpoints" in dst.parts
                    and d in dst.parents)
    waits, fences, real_nap, real_fence = [], [], nb._Launcher._nap, nb._Launcher.fence

    def nap(self, seconds):
        waits.append(round(seconds, 6))
        return real_nap(self, seconds)

    def fence(self):
        fences.append(len(waits))
        return real_fence(self)

    monkeypatch.setattr(nb._Launcher, "_nap", nap)
    monkeypatch.setattr(nb._Launcher, "fence", fence)
    res = _launch(p)
    out = capsys.readouterr().out
    assert waits == [round(w, 6) for w in nb.end_copy_waits(0.02)] and len(waits) == 7, waits
    assert all(b > a for a, b in zip(waits[:4], waits[1:5], strict=True))          # each wait twice the one before
    assert all(i in fences for i in range(1, 8))                                    # every try is fenced
    assert res["state"] == "unfinished" and "not on Drive after 7 further tries in " in out


def test_finish_makes_the_last_copy_and_ends_the_run(tmp_path, stubs, monkeypatch, capsys):
    """Ruling 1 of 17 Nov: the finish cell, run again later, tries the copies a finished job still lacks (its last
    checkpoint and the check's tar); once they are on Drive the run is written DONE there, and the finish answers
    that everything is on Drive (as built it only looked: the run stayed unfinished and the save stayed on this
    machine's disk)."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02)}))
    monkeypatch.setattr(nb, "END_COPY_PERIODS", 1.0)
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d, job = tmp_path / "drive" / "runs" / "r1", "R5-B1-s42"
    broken = dict(on=True)
    _failing_copies(monkeypatch, lambda src, dst: broken["on"] and src.name in ("model_4096.zip", "ckpt_4096.tar")
                    and d in dst.parents)
    assert _launch(p)["state"] == "unfinished"
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(p))
    assert nb.main(["finish", "--params", str(pfile)]) == nb.EXIT_UNFINISHED        # Drive still fails: kept
    capsys.readouterr()
    broken["on"] = False
    fin = nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    out = capsys.readouterr().out
    assert fin["state"] == "done" and fin["on_drive"] is True and fin["not_on_drive"] == {}, fin
    assert _reads_back(d / "jobs" / job, 4096) and (d / "jobs" / job / "ckpt_4096.tar").is_file()
    assert json.loads((d / "jobs" / job / "run.json").read_text())["status"] == "done"
    done = json.loads((d / "DONE.json").read_text())
    assert done["reason"].startswith("the queue is finished") and done["jobs"][job]["last_checkpoint_copied"] == 4096
    assert "S2 FINISH r1: DONE: the queue is finished" in out and "closed_at" in json.loads((d / "LIVE.json").read_text())
    rec = json.loads((d / "SESSION_0.json").read_text())
    assert [f["on_drive"] for f in rec["finishes"]] == [False, True] and rec["last_status_at"] == rec["finishes"][-1]["at"]
    assert nb.main(["finish", "--params", str(pfile)]) == 0 and len(_calls(p)) == 1  # (nothing was trained again)


def test_the_finish_keeps_the_runtime_while_a_save_is_not_on_drive(tmp_path, stubs, monkeypatch, capsys):
    """Ruling 1 of 17 Nov: when a job's last save is not on Drive the finish step says so by its exit code (also
    when the run halted afterwards: as built a halt's exit code 5 came first and the runtime was let go), and the
    notebook's cell 8 then keeps the runtime and prints one plain paragraph; in every other case it releases it.
    The finish never raises."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.02), "R6-B1-s42": dict(halt=True)}))
    monkeypatch.setattr(nb, "END_COPY_PERIODS", 1.0)
    p = _params(tmp_path, CLIPS=False)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    _failing_copies(monkeypatch, lambda src, dst: "R5-B1-s42" in dst.parts and src.name == "model_4096.zip"
                    and d in dst.parents)
    res = _launch(p)
    assert res["state"] == "halted" and (d / "HALT.json").is_file(), res      # (job 2 halted; job 1's save is here)
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(p))
    assert nb.main(["finish", "--params", str(pfile)]) == nb.EXIT_UNFINISHED
    assert "NOT ON DRIVE: R5-B1-s42: checkpoint 4096" in capsys.readouterr().out
    # a fault in the finish itself (here: this machine's disk is full) is no traceback: it cannot tell, so it keeps
    real_init = nb._Launcher.__init__

    def full(self, *a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(nb._Launcher, "__init__", full)
    fin = nb.finish("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    assert fin["on_drive"] is False and "the finish could not be completed (OSError" in capsys.readouterr().out
    monkeypatch.setattr(nb._Launcher, "__init__", real_init)
    # cell 8 as the notebook runs it: s2_step's exit code decides
    released = []
    monkeypatch.setattr(nb, "release_runtime", lambda reason, in_colab, auto, flush_drive: released.append(reason))
    cell = "".join(_cells()[8]["source"])
    for code, want in ((nb.EXIT_UNFINISHED, []), (0, ["Stage 2 session of run 'r1' is over."]),
                       (nb.EXIT_HALT, ["Stage 2 session of run 'r1' is over."]), ("raise", ["Stage 2 session of run 'r1' is over."])):
        released.clear()

        def s2_step(step, check=True, code=code):
            assert step == "finish" and check is False
            if code == "raise":
                raise KeyboardInterrupt
            return code

        space = dict(s2_step=s2_step, STUDY=str(STUDY), sys=sys, RUN_ID="r1", IN_COLAB=True, AUTO_DISCONNECT=True,
                     USE_GOOGLE_DRIVE=True)
        if code == "raise":
            with pytest.raises(KeyboardInterrupt):
                exec(compile(cell, "<cell 8>", "exec"), space)
        else:
            exec(compile(cell, "<cell 8>", "exec"), space)
        out = capsys.readouterr().out
        assert released == want, code
        assert ("THIS MACHINE IS KEPT ON PURPOSE" in out) is (code == nb.EXIT_UNFINISHED), code
    assert "run the last grey box (8. FINISH) once more" in nb.KEEP_RUNTIME_TEXT and "\n" not in nb.KEEP_RUNTIME_TEXT


def test_a_failed_first_write_to_drive_is_no_halt(tmp_path, stubs, monkeypatch, capsys):
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    real, hit = nb._write_json, dict(n=0)

    def write(path, obj):
        if Path(path).name == "STATUS.json" and hit["n"] == 0:      # the launcher's first beat
            hit["n"] += 1
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real(path, obj)

    monkeypatch.setattr(nb, "_write_json", write)
    res = _launch(p)
    assert hit["n"] == 1 and res["state"] == "done" and not (d / "HALT.json").exists(), res
    assert res["jobs"]["R5-B1-s42"]["state"] == "done" and "a copy to Drive failed" in capsys.readouterr().out


def test_the_launcher_goes_on_when_nobody_reads_its_output(tmp_path, stubs, monkeypatch):
    """The reader of the launcher's output goes away (a kernel that died alone): the run goes on to its end.  As
    built, the next printed line raised BrokenPipeError and the run was halted for good."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.05, budget=512 * 40)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(p))
    proc = subprocess.Popen([sys.executable, "-m", "designlab.stage2.nb", "launch", "--params", str(pfile)], cwd=STUDY,
                            env=dict(os.environ, PYTHONPATH="."), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not _on_drive(d / "jobs" / "R5-B1-s42"):
            assert proc.poll() is None
            time.sleep(0.02)
        proc.stdout.close()                                          # the reader goes; the launcher is not signalled
        code = proc.wait(timeout=120)
    finally:
        nb.kill_session(proc.pid)
    assert not (d / "HALT.json").exists(), json.loads((d / "HALT.json").read_text())["reason"]
    assert code == 0 and json.loads((d / "DONE.json").read_text())["state"] == "done"
    run = json.loads((d / "jobs" / "R5-B1-s42" / "run.json").read_text())
    assert run["status"] == "done" and run["num_timesteps"] == 512 * 40
    assert "done at 20480 steps" in (d / "logs" / "R5-B1-s42.s0.log").read_text()


def _killed_run(tmp_path, monkeypatch, checkpoints=2):
    """A launcher and its job killed together once the stand-in Drive holds ``checkpoints`` checkpoints of the
    job, and the local run directory lost (a dead runtime): ``(params, the run directory, the job, what Drive
    holds)``.  The stub's plan is then set so that a resume finishes at once."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.05, budget=512 * 200)}))
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d, job = tmp_path / "drive" / "runs" / "r1", "R5-B1-s42"
    pfile = tmp_path / "killed.json"
    pfile.write_text(json.dumps(p))
    proc = subprocess.Popen([sys.executable, "-m", "designlab.stage2.nb", "launch", "--params", str(pfile)], cwd=STUDY,
                            env=dict(os.environ, PYTHONPATH="."), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        deadline = time.time() + 60
        while time.time() < deadline and len(_on_drive(d / "jobs" / job)) < checkpoints:
            assert proc.poll() is None
            time.sleep(0.02)
    finally:
        nb.kill_session(proc.pid)
        proc.wait()
    shutil.rmtree(tmp_path / "local" / "r1")
    nb.sweep_parts(d)
    have = [n for n in _on_drive(d / "jobs" / job) if _reads_back(d / "jobs" / job, n)]
    assert have and have == _on_drive(d / "jobs" / job)
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.0, budget=512 * 200)}))
    return p, d, job, have


def test_a_failed_read_during_the_copy_back_is_no_halt(tmp_path, stubs, monkeypatch, capsys):
    """The copy back from Drive on a resume: a read that fails is made again; a Drive that cannot be read ends the
    session unfinished, with nothing half copied, no HALT.json and the run open to the next resume.  As built, one
    failed read wrote HALT.json ("launcher error: OSError") and the run was never resumed."""
    p, d, job, have = _killed_run(tmp_path, monkeypatch)
    lj = tmp_path / "local" / "r1" / "jobs" / job
    broken = dict(mode="always", n=0)

    def fails(src, dst):
        if d not in src.parents or broken["mode"] is None:
            return False
        broken["n"] += 1
        return broken["mode"] == "always" or (src.suffix == ".zip" and broken.pop("once", False))

    _failing_copies(monkeypatch, fails)
    _session(p, RESUME=True, TAKEOVER=True)
    capsys.readouterr()
    res = _launch(p)
    out = capsys.readouterr().out
    assert res["state"] == "unfinished" and "Drive could not be read" in res["reason"], res
    assert out.count("reading from Drive failed") == nb.COPY_BACK_TRIES and "S2 launch: UNFINISHED: " in out
    assert not (d / "HALT.json").exists() and not (d / "DONE.json").exists() and not _calls(p)
    assert not lj.exists()                                           # all or nothing: nothing half copied
    assert json.loads((d / "LIVE.json").read_text())["closed_reason"].startswith("unfinished: ")
    # one failed read: the copy is made again, whole, and the job goes on from Drive's newest checkpoint
    broken.update(mode="once", once=True)
    s2 = _session(p, RESUME=True)                                    # (no HALT.json, LIVE.json closed: accepted)
    assert s2["session"] == 2
    res = _launch(p)
    out = capsys.readouterr().out
    assert res["state"] == "done" and out.count("reading from Drive failed") == 1 and "copied back from Drive" in out
    run = json.loads((d / "jobs" / job / "run.json").read_text())
    assert run["status"] == "done" and run["sessions"][-1]["resumed_from"] == max(have) > 0
    assert len(_calls(p)) == 1 and "--resume" in _calls(p)[0]["argv"] and not (d / "HALT.json").exists()


def test_a_resume_never_restarts_a_job_that_drive_holds(tmp_path, stubs, monkeypatch, capsys):
    """As built, a local job directory that held a ``run.json`` was never copied back, so a directory that a cut
    copy had left (``run.json`` and no checkpoint) made the trainer restart the job from step 0, silently, although
    Drive held its checkpoints."""
    p, d, job, have = _killed_run(tmp_path, monkeypatch)
    lrun = tmp_path / "local" / "r1"
    lj = lrun / "jobs" / job
    lj.mkdir(parents=True)
    shutil.copy(d / "jobs" / job / "run.json", lj / "run.json")     # what a cut copy back leaves: run.json alone
    s1 = _session(p, RESUME=True, TAKEOVER=True)
    assert int(nb.LINE_RE["resume"].match(s1["line"]).group("steps")) == max(have)
    capsys.readouterr()
    res = _launch(p)
    out = capsys.readouterr().out
    run = json.loads((d / "jobs" / job / "run.json").read_text())
    resumed = [x["resumed_from"] for x in run["sessions"]]
    assert resumed[-1] == max(have) > 0 and run["status"] == "done", resumed   # as built: None (from step 0)
    assert res["state"] == "done" and f"stub session {len(resumed) - 1} from {max(have)} steps" in out
    # the directory that was there is kept aside, never deleted, and the whole directory is copied again
    assert "copied back from Drive" in out and "kept aside" in out
    assert sorted(x.name for x in (lrun / "work" / "aside" / f"{job}.s1").iterdir()) == ["run.json"]
    # this machine's own files are used as they are when they are at least as new as Drive's copies
    launcher = nb._Launcher(nb._open_state("r1", p["DRIVE_DIR"], p["LOCAL_DIR"]),
                            json.loads((d / "QUEUE.json").read_text()), 2, False, 0.05)
    try:
        before = sorted(x.name for x in (lj / "checkpoints").iterdir())
        launcher.copy_back(job)
        assert sorted(x.name for x in (lj / "checkpoints").iterdir()) == before
        assert not (lrun / "work" / "aside" / f"{job}.s1.1").exists() and "copied back" not in capsys.readouterr().out
        launcher._guard(job, True)                                   # (and nothing stops a start here)
    finally:
        launcher.log.close()


def test_the_copy_back_is_skipped_only_on_its_marker(tmp_path, stubs, monkeypatch, capsys):
    """The lead's ruling 3 of 17 Nov: the copy back is skipped only on its marker (work/whole/<job>.json, written by
    a whole copy back or when this machine's launcher starts the trainer).  A directory here that holds run.json and
    a checkpoint as new as Drive's newest, but that no copy back and no trainer of this machine made whole (here: it
    lacks updates.jsonl and the gates), is copied back and kept aside.  As built it was used as it was."""
    p, d, job, have = _killed_run(tmp_path, monkeypatch)
    lrun = tmp_path / "local" / "r1"
    lj = lrun / "jobs" / job
    (lj / "checkpoints").mkdir(parents=True)
    shutil.copy(d / "jobs" / job / "run.json", lj / "run.json")
    for f in (d / "jobs" / job / "checkpoints").iterdir():             # a whole checkpoint as new as Drive's newest
        if f.name.endswith((f"_{max(have)}.json", f"_{max(have)}.zip", f"_{max(have)}.pkl", f"_{max(have)}.npz")):
            shutil.copy(f, lj / "checkpoints" / f.name)
    assert (d / "jobs" / job / "updates.jsonl").is_file() and not (lj / "updates.jsonl").exists()
    _session(p, RESUME=True, TAKEOVER=True)
    capsys.readouterr()
    res = _launch(p)
    out = capsys.readouterr().out
    assert res["state"] == "done" and "copied back from Drive" in out and "not known whole" in out, res
    assert (lrun / "work" / "aside" / f"{job}.s1" / "run.json").is_file()
    mark = json.loads((lrun / "work" / "whole" / f"{job}.json").read_text())
    assert mark["how"] == "started here" and mark["session"] == 1             # (rewritten when the trainer started)
    run = json.loads((d / "jobs" / job / "run.json").read_text())
    assert run["sessions"][-1]["resumed_from"] == max(have) and run["status"] == "done"
    rows = [json.loads(x) for x in (d / "jobs" / job / "updates.jsonl").read_text().splitlines()]
    assert rows[0]["session"] == 0                                             # (the first session's updates came back)


def test_a_job_is_not_started_while_drive_holds_a_newer_checkpoint(tmp_path, stubs, monkeypatch, capsys):
    """The guard behind the copy back: whatever the copy back did or left, a job is not started while Drive holds
    a checkpoint of it that is newer than every one in this machine's directory.  The session ends unfinished."""
    p, d, job, have = _killed_run(tmp_path, monkeypatch)
    real = getattr(nb._Launcher, "_copy_back", None)
    # (a copy back that learns what Drive holds and copies nothing)
    monkeypatch.setattr(nb._Launcher, "_copy_back", lambda self, job: self._adopt(job), raising=False)
    _session(p, RESUME=True, TAKEOVER=True)
    res = _launch(p)
    assert res["state"] == "unfinished" and f"Drive holds its checkpoint {max(have)}" in res["reason"], res
    assert "was not started" in res["reason"] and not _calls(p)
    assert not (d / "HALT.json").exists() and not (d / "DONE.json").exists()
    assert "S2 launch: UNFINISHED: job R5-B1-s42 was not started" in capsys.readouterr().out
    monkeypatch.setattr(nb._Launcher, "_copy_back", real)           # with the real copy back the run goes on
    assert _session(p, RESUME=True)["session"] == 2
    assert _launch(p)["state"] == "done"
    run = json.loads((d / "jobs" / job / "run.json").read_text())
    assert run["sessions"][-1]["resumed_from"] == max(have) and len(_calls(p)) == 1


def test_line_4_is_this_sessions_first_rate(tmp_path, stubs, monkeypatch, capsys):
    """Rule N1.  After a resume the copy back brings the dead session's ``progress.json`` and ``gates/rate.json``;
    until the resumed trainer has written its own, they name the older session.  As built, a cycle in that time
    printed the older session's rate as this session's line 4 (seen in the slow local run of 16 Nov, when the
    kill came after the job's third rollout)."""
    p, d, job, have = _killed_run(tmp_path, monkeypatch)
    rate = json.loads((d / "jobs" / job / "gates" / "rate.json").read_text())
    assert rate["sessions"][0]["first_three"]["env_steps_per_wall_s"] == 500.0   # the dead session's, on Drive
    assert json.loads((d / "jobs" / job / "progress.json").read_text())["session"] == 0
    # the resumed trainer needs a while before its first progress.json, and runs at another rate
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(start_delay=0.6, tick=0.0, rate=700.0, budget=512 * 200)}))
    _session(p, RESUME=True, TAKEOVER=True)
    capsys.readouterr()
    res = _launch(p)
    lines = nb.lines_of(capsys.readouterr().out)[4]
    assert res["state"] == "done" and len(lines) == 1, lines
    assert nb.LINE_RE[4].match(lines[0]).group("rate") == "700"
    assert [x for x in json.loads((d / "STATUS.json").read_text())["lines"] if x.startswith("S2 LINE 4/4")] == lines


def test_a_session_closed_by_a_failed_line_launches_nothing(tmp_path, stubs, capsys):
    """Rule N1: line 2 ending FAILED or line 3 ending FAIL stops the session before any training.  As built only
    the cells' order did: the same session's launch, run by hand afterwards, trained."""
    p = _params(tmp_path, QUEUE=["R5:B1:42"], CLIPS=False)
    _session(p)
    d = tmp_path / "drive" / "runs" / "r1"
    st = nb._open_state("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    nb._note(st, nb.line2("tb-1", None, 0, 4, "models-B1.tar: missing"))
    nb._close(st, "line 2 failed: nothing was trained")              # what the inputs step does on a failure
    res = _launch(p)
    assert res["state"] == "refused" and "this session was closed (line 2 failed" in res["reason"], res
    assert not _calls(p) and not (d / "jobs").exists() and "S2 REFUSED launch" in capsys.readouterr().out
    assert json.loads((d / "SESSION_0.json").read_text())["closed"]["reason"].startswith("line 2 failed")
    with pytest.raises(nb.Refused, match="this session was closed .* runs no smoke"):
        nb.smoke("r1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    assert _session(p, RESUME=True)["session"] == 1                  # a new session re-enters the run
    assert _launch(p)["state"] == "done"


def test_the_stub_hook_is_refused_on_colab_and_in_a_run_that_is_not_tiny(tmp_path, stubs):
    """The lead's ruling 4 of 17 Nov: S2_NB_SEAMS is refused on Colab and in any run that is not tiny (as built: on
    Colab only), by the session and by the launch."""
    p = _params(tmp_path)
    with pytest.raises(nb.Refused, match="S2_NB_SEAMS .* refused on Colab and in any run that is not tiny"):
        _session(p, IN_COLAB=True)
    with pytest.raises(nb.Refused, match="S2_NB_SEAMS .* refused on Colab and in any run that is not tiny"):
        _session(p, TINY=False, BUDGET_STEPS=40_000_000)
    assert not (tmp_path / "drive").exists()
    # a launch of a session opened without the hook, with the hook named afterwards
    seams = os.environ.pop("S2_NB_SEAMS")
    try:
        _session(dict(p, TINY=False, BUDGET_STEPS=8192))
    finally:
        os.environ["S2_NB_SEAMS"] = seams
    res = _launch(dict(p, TINY=False, BUDGET_STEPS=8192))
    assert res["state"] == "refused" and "S2_NB_SEAMS" in res["reason"] and not _calls(p)
    assert "differs in spelling" not in nb.K6_RULE                   # (a note that no launch could print is gone)


def test_a_colab_launch_on_a_checkout_with_modified_tracked_files_is_refused(tmp_path, monkeypatch, capsys):
    """The lead's ruling 4 of 17 Nov: a Colab launch is refused when tracked files of the checkout are modified;
    untracked files are only reported (as built: line 1 said "checkout clean: no" and nothing stopped)."""
    p = _params(tmp_path)
    clean = dict(commit=COMMIT, dirty=False, dirty_paths=[], modified=[], untracked=[])
    monkeypatch.setattr(nb, "repo_state", lambda: dict(clean, dirty=True, dirty_paths=[" M designlab/stage2/nb.py"],
                                                       modified=[" M designlab/stage2/nb.py"]))
    with pytest.raises(nb.Refused, match="tracked files of the checkout are modified .* M designlab/stage2/nb.py"):
        _session(p, IN_COLAB=True)
    assert not (tmp_path / "drive").exists()
    assert _session(p, RUN_ID="here")["repo"]["modified"] == [" M designlab/stage2/nb.py"]   # (not on Colab: kept)
    smoke = _session(p, RUN_ID="smoke1", IN_COLAB=True, SMOKE=True, QUEUE=[], REPO_REF="s2-launch-1")
    assert smoke["repo"]["dirty"] is True                                       # (a smoke: reported, as line 1 does)
    monkeypatch.setattr(nb, "repo_state", lambda: dict(clean, dirty=True, dirty_paths=["?? notes.txt"],
                                                       untracked=["?? notes.txt"]))
    capsys.readouterr()
    s = _session(p, RUN_ID="r2", IN_COLAB=True)
    assert s["repo"]["untracked"] == ["?? notes.txt"] and s["repo"]["modified"] == []
    assert "S2 NOTE: untracked files in the checkout (reported, not refused): notes.txt" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------
# inputs and preflight
# ------------------------------------------------------------------------------------------------
def _tar(path: Path, root: Path, members: list) -> None:
    """``bundle``'s tar rules (design D20) for a hand-made part."""
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for rel in sorted(members):
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode, info.type = (root / rel).stat().st_size, 0, 0o644, tarfile.REGTYPE
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(root / rel, "rb") as fh:
                tar.addfile(info, fh)


def _bundle(tmp_path, bundle_id="tb-test", with_set=False) -> dict:
    """A bundle of one body's two model files (and, ``with_set``, of two arrays of a set ``SX``), made by hand
    under a scratch root: ``{root, drive, manifest, parts, src}``."""
    s2io = pytest.importorskip("designlab.stage2.s2io")
    src, root, drive = tmp_path / "src", tmp_path / "root", tmp_path / "drive"
    files, sets, part_docs = {}, {}, {}
    rows = [("round2/refine/case1/models/m1.xml", b"<mujoco/>" * 40, "models-B1.tar"),
            ("round2/refine/case1/models/m1.json", b"{}", "models-B1.tar")]
    if with_set:
        rows += [("stage2/sets/SX/B1/refs/a.npz", b"walk " * 300, "SX-B1-1.tar"),
                 ("stage2/sets/SX/B1/transitions/b.npz", b"move " * 200, "SX-B1-1.tar")]
        sets = dict(SX=dict(B1=dict(freeze_index=0, manifest_sha256="5" * 64, files=2)))
    for name, data, part in rows:
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
        files[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), part=part)
    parts = drive / "inputs" / bundle_id
    parts.mkdir(parents=True)
    for part in sorted({r[2] for r in rows}):
        _tar(parts / part, src, [r[0] for r in rows if r[2] == part])
        blob = (parts / part).read_bytes()
        part_docs[part] = dict(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), files=2)
    content = dict(bundle_id=bundle_id, bodies=["B1"], sets=sets, files=files,
                   models=dict(B1=dict(case="case1", xml=dict(path="round2/refine/case1/models/m1.xml",
                                                              sha256=files["round2/refine/case1/models/m1.xml"]["sha256"]),
                                       record=dict(path="round2/refine/case1/models/m1.json",
                                                   sha256=files["round2/refine/case1/models/m1.json"]["sha256"]))),
                   parts=part_docs)
    doc = dict(schema="s2-bundle-2", content=content, content_sha256=s2io.sha256_json(content), provenance={})
    mpath = root / nb.BUNDLE_DIR / bundle_id / "MANIFEST.json"
    mpath.parent.mkdir(parents=True)
    mpath.write_text(json.dumps(doc, indent=1))
    return dict(root=root, drive=drive, manifest=doc, parts=parts, src=src)


def test_inputs_verify_and_unpack_the_bundle(tmp_path, monkeypatch, capsys):
    b = _bundle(tmp_path)
    monkeypatch.setattr(nb, "ROOT", b["root"])
    res = nb.inputs("tb-test", b["drive"], local=tmp_path / "local")
    sha = b["manifest"]["content_sha256"]
    assert res["ok"] and res["line"] == (f"S2 LINE 2/4 inputs: bundle tb-test, manifest {sha[:12]}, 2 of 2 files match "
                                         "their sha256: OK") in capsys.readouterr().out
    assert res["content_sha256"] == sha and res["unpacked"]["written"] == 2
    assert (b["root"] / "round2/refine/case1/models/m1.xml").read_bytes() == b"<mujoco/>" * 40
    assert nb.inputs("tb-test", b["drive"], local=tmp_path / "local")["unpacked"]["left"] == 2   # a second time: nothing new
    # a file changed after the unpack: FAILED with its path (and the part is unpacked over it only by a new unpack)
    (b["root"] / "round2/refine/case1/models/m1.json").write_bytes(b"{ }")
    assert nb.inputs("tb-test", b["drive"], local=tmp_path / "local")["ok"]       # unpack replaces a wrong file
    # a part cut short on Drive, with no local copy: FAILED names the part; nothing was unpacked
    shutil.rmtree(tmp_path / "local")
    whole = (b["parts"] / "models-B1.tar").read_bytes()
    (b["parts"] / "models-B1.tar").write_bytes(whole[:-512])
    bad = nb.inputs("tb-test", b["drive"], local=tmp_path / "local")
    assert not bad["ok"] and nb.LINE_RE[2].match(bad["line"]) and "FAILED (models-B1.tar:" in bad["line"]
    (b["parts"] / "models-B1.tar").unlink()
    gone = nb.inputs("tb-test", b["drive"], local=tmp_path / "local")
    assert "FAILED (models-B1.tar: missing in" in gone["line"] and gone["files_ok"] == 0
    (b["parts"] / "models-B1.tar").write_bytes(whole)
    # a manifest whose own content hash was changed, another bundle id, an unknown source
    mpath = b["root"] / nb.BUNDLE_DIR / "tb-test" / "MANIFEST.json"
    mpath.write_text(json.dumps(dict(b["manifest"], content_sha256="0" * 64)))
    assert "FAILED (" in nb.inputs("tb-test", b["drive"], local=tmp_path / "local")["line"]
    mpath.write_text(json.dumps(b["manifest"]))
    assert "MANIFEST.json is not in this checkout" in nb.inputs("tb-other", b["drive"])["line"]
    assert "BUNDLE_SOURCE" in nb.inputs("tb-test", b["drive"], "ftp")["line"]
    assert "needs its bundle" in nb.inputs("", b["drive"], set_name="V6")["line"]
    # the rebuild fallback is refused until a smoke's rule Y8 allowed it for this content hash (rule N4)
    assert "not an allowed fallback" in nb.inputs("tb-test", b["drive"], "rebuild", local=tmp_path / "l2")["line"]
    ok = nb.rebuilt_dir(b["drive"], "tb-test") / "REBUILD_OK.json"
    ok.parent.mkdir(parents=True)
    ok.write_text(json.dumps(dict(allowed=True, content_sha256=sha)))
    res = nb.inputs("tb-test", b["drive"], "rebuild", local=tmp_path / "l2")
    assert res["ok"] and res["source"] == "rebuild", res                          # (a bundle of models: nothing to build)


STUB_BUNDLE = r"""
import json, os, sys
plan = json.loads(os.environ["STUB_BUNDLE"])
print(json.dumps(dict(plan[sys.argv[1]], ok=True)))
"""


def test_inputs_compare_what_verify_reports_with_the_manifest(tmp_path, stubs, monkeypatch):
    """The lead's ruling of 16 Nov on the bundle tool, item 3: the notebook reads ``bundle_id`` and the 64-hex
    ``content_sha256`` from ``verify --json`` and compares them with the quoted ones.  The bundle tool refuses the
    same cases itself; a tool that reports them and says ok shows the notebook's own comparison."""
    b = _bundle(tmp_path)
    (stubs / "bundle.py").write_text(STUB_BUNDLE)
    seams = json.loads((stubs / "seams.json").read_text())
    seams["bundle"] = [sys.executable, str(stubs / "bundle.py")]
    (stubs / "seams.json").write_text(json.dumps(seams))
    monkeypatch.setattr(nb, "ROOT", b["root"])
    sha = b["manifest"]["content_sha256"]

    def run(**verify):
        plan = dict(unpack=dict(parts=1, files=2, written=[], replaced=[], left=[]),
                    verify=dict(dict(files_ok=2, files=2, bundle_id="tb-test", content_sha256=sha), **verify))
        monkeypatch.setenv("STUB_BUNDLE", json.dumps(plan))
        return nb.inputs("tb-test", b["drive"], local=tmp_path / "local")

    good = run()
    assert good["ok"] and good["content_sha256"] == sha and good["line"].endswith("2 of 2 files match their sha256: OK")
    bad = run(bundle_id="tb-other")
    assert not bad["ok"] and "FAILED (the manifest is bundle 'tb-other', not 'tb-test')" in bad["line"]
    for other in ("0" * 64, sha[:12], None):
        bad = run(content_sha256=other)
        assert not bad["ok"] and "FAILED (the content hash verify reports" in bad["line"], other


def test_a_resume_with_another_bundle_fails_line_2(tmp_path, monkeypatch, capsys):
    """Rule K6: one bundle for a run's whole life.  As built, line 2 was printed and kept as OK and the refusal
    came after it on a line of its own."""
    b = _bundle(tmp_path)
    shutil.copy(STUDY / nb.PINS_FILE, b["root"] / nb.PINS_FILE)
    monkeypatch.setattr(nb, "ROOT", b["root"])
    p = _params(tmp_path, BUNDLE_ID="tb-test", QUEUE=["R5:B1:42"])
    _session(p)
    d, sha = tmp_path / "drive" / "runs" / "r1", b["manifest"]["content_sha256"]
    assert nb._step_inputs(p, None) == nb.EXIT_OK
    q = json.loads((d / "QUEUE.json").read_text())
    assert q["bundle"]["content_sha256"] == sha                      # the first session's hash is the run's
    q["bundle"]["content_sha256"] = "1" * 64                         # (as if the run had begun on another bundle)
    (d / "QUEUE.json").write_text(json.dumps(q))
    p2 = dict(p, RESUME=True, TAKEOVER=True)
    _session(p2)
    capsys.readouterr()
    assert nb._step_inputs(p2, None) == nb.EXIT_FAILED
    lines = nb.lines_of(capsys.readouterr().out)[2]
    assert len(lines) == 1 and lines[0].endswith(
        f": FAILED (RESUME with another bundle: the run's content hash is 111111111111, this checkout's inputs give "
        f"{sha[:12]} (rule K6))"), lines
    assert json.loads((d / "STATUS.json").read_text())["lines"][-1] == lines[0]
    assert json.loads((d / "SESSION_1.json").read_text())["closed"]["reason"].startswith("line 2 failed")
    assert json.loads((d / "QUEUE.json").read_text())["bundle"]["content_sha256"] == "1" * 64


STUB_REBUILD = r'''
import json, os, shutil, sys
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
plan = json.loads(os.environ["STUB_REBUILD"])
open(plan["log"], "a").write(a["--set"] + " " + a["--body"] + "\n")
src = os.path.join(plan["src"], "stage2", "sets", a["--set"], a["--body"])
for base, _, names in os.walk(src):
    for name in names:
        dst = os.path.join(a["--root"], a["--set"], a["--body"], os.path.relpath(os.path.join(base, name), src))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(base, name), dst)
        if plan.get("wrong"):
            open(dst, "ab").write(b"x")
'''
STUB_CHILD = r'''
import json, sys
what, args = sys.argv[1], dict(zip(sys.argv[2::2], sys.argv[3::2]))
assert what == "read-inputs"
print(json.dumps(dict(ok=True, files=len(json.loads(args["--files"])), table_sha256="0" * 64)))
'''


def test_the_rebuild_fallback_and_rule_y8(tmp_path, stubs, monkeypatch, capsys):
    """Rule N4's second fallback: allowed only after rule Y8 found every hash equal; the arrays a verified rebuild
    made are kept on the stand-in Drive and reused; a rebuild whose bytes differ is refused."""
    b = _bundle(tmp_path, with_set=True)
    (stubs / "rebuild.py").write_text(STUB_REBUILD)
    (stubs / "child.py").write_text(STUB_CHILD)
    seams = json.loads((stubs / "seams.json").read_text())
    seams.update(rebuild=[sys.executable, str(stubs / "rebuild.py")], child=[sys.executable, str(stubs / "child.py")])
    (stubs / "seams.json").write_text(json.dumps(seams))
    log = tmp_path / "rebuild.log"
    plan = dict(src=str(b["src"]), log=str(log))
    monkeypatch.setenv("STUB_REBUILD", json.dumps(plan))
    monkeypatch.setattr(nb, "ROOT", b["root"])
    full = nb.inputs("tb-test", b["drive"], local=tmp_path / "local")            # the smoke's own inputs: every part
    assert full["ok"] and full["files"] == 4 and full["sets"] == dict(SX=dict(B1=True)), full
    p = nb.load_params(dict(BUNDLE_ID="tb-test"))
    work = tmp_path / "work"
    (work / "logs").mkdir(parents=True)
    cache = nb.rebuilt_dir(b["drive"], "tb-test")
    # a rebuild with other bytes: every file compared, none equal: no permission, nothing kept
    monkeypatch.setenv("STUB_REBUILD", json.dumps(dict(plan, wrong=True)))
    y8 = nb._smoke_rebuild(p, "B1", work, b["drive"], "smoke-1")
    assert (y8["compared"], y8["equal"], y8["total"], y8["allowed"]) == (2, 0, 2, False) and not cache.exists()
    # the same bytes: allowed; the arrays and REBUILD_OK.json go to the stand-in Drive
    monkeypatch.setenv("STUB_REBUILD", json.dumps(plan))
    shutil.rmtree(work / "rebuild")
    y8 = nb._smoke_rebuild(p, "B1", work, b["drive"], "smoke-1")
    assert (y8["compared"], y8["equal"], y8["total"], y8["models_equal"]) == (2, 2, 2, 2) and y8["allowed"] and y8["finished"]
    assert "the rebuild is an allowed fallback" in y8["value"] and "2 of 2 arrays of B1 compared, 2 equal" in y8["value"]
    ok = json.loads((cache / "REBUILD_OK.json").read_text())
    assert ok["allowed"] and ok["content_sha256"] == b["manifest"]["content_sha256"] and ok["smoke_run_id"] == "smoke-1"
    assert (cache / "stage2/sets/SX/B1/refs/a.npz").read_bytes() == b"walk " * 300
    # a launch on a new runtime with only the model part on Drive: the arrays come from the kept rebuild, nothing is built
    (b["parts"] / "SX-B1-1.tar").unlink()
    calls = len(log.read_text().splitlines())

    def fresh(name):
        root = tmp_path / name
        shutil.copytree(b["root"] / nb.BUNDLE_DIR, root / nb.BUNDLE_DIR)
        monkeypatch.setattr(nb, "ROOT", root)
        return root

    root = fresh("root2")
    assert "FAILED (SX-B1-1.tar: missing" in nb.inputs("tb-test", b["drive"], local=tmp_path / "l2")["line"]
    res = nb.inputs("tb-test", b["drive"], "rebuild", local=tmp_path / "l2")
    assert res["ok"] and res["files_ok"] == 4 and len(log.read_text().splitlines()) == calls, res
    assert (root / "stage2/sets/SX/B1/transitions/b.npz").read_bytes() == b"move " * 200
    # without the kept arrays they are built again, verified, and kept for the next launch
    shutil.rmtree(cache / "stage2")
    root = fresh("root3")
    res = nb.inputs("tb-test", b["drive"], "rebuild", local=tmp_path / "l3")
    assert res["ok"] and log.read_text().splitlines()[calls:] == ["SX B1"] and (cache / "stage2/sets/SX/B1/refs/a.npz").is_file()
    # a rebuild that gives other bytes is refused, and is not kept
    shutil.rmtree(cache / "stage2")
    monkeypatch.setenv("STUB_REBUILD", json.dumps(dict(plan, wrong=True)))
    root = fresh("root4")
    bad = nb.inputs("tb-test", b["drive"], "rebuild", local=tmp_path / "l4")
    assert not bad["ok"] and "FAILED (stage2/sets/SX/B1/refs/a.npz:" in bad["line"] and bad["files_ok"] == 2
    assert not (cache / "stage2").exists()


@needs_git
def test_inputs_from_a_data_branch(tmp_path, monkeypatch):
    """Rule N4's third fallback, on a scratch repository whose remote is another scratch repository."""
    b = _bundle(tmp_path)
    remote, clone = tmp_path / "remote", tmp_path / "clone"
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_AUTHOR_NAME="t",
               GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")

    def git(cwd, *args):
        return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True, env=env).stdout

    remote.mkdir()
    git(remote, "init", "-q", "-b", "main")
    (remote / "README").write_text("x")
    git(remote, "add", "README")
    git(remote, "commit", "-q", "-m", "main")
    git(remote, "checkout", "-q", "-b", "s2-data")
    dst = remote / nb.BUNDLE_DIR / "tb-test" / "parts"
    dst.mkdir(parents=True)
    shutil.copy(b["parts"] / "models-B1.tar", dst / "models-B1.tar")
    git(remote, "add", "-f", ".")
    git(remote, "commit", "-q", "-m", "data")
    git(remote, "checkout", "-q", "main")
    git(tmp_path, "clone", "-q", str(remote), str(clone))
    shutil.copytree(b["root"] / nb.BUNDLE_DIR, clone / nb.BUNDLE_DIR)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(nb, "ROOT", clone)
    monkeypatch.setattr(nb, "_git", lambda *a, cwd=None: git(clone if cwd is None else cwd, *a))
    res = nb.inputs("tb-test", tmp_path / "no-drive", "git:s2-data", local=tmp_path / "local")
    assert res["ok"] and res["source"] == "git:s2-data" and (clone / "round2/refine/case1/models/m1.xml").is_file()
    bad = nb.inputs("tb-test", tmp_path / "no-drive", "git:no-such-branch", local=tmp_path / "local3")
    assert not bad["ok"] and "could not be fetched" in bad["line"]


def test_the_s1_view_has_no_bundle(capsys):
    if not (STUDY / "stage1/runs/2026-11-08/75-M-H-10/refs").is_dir():
        pytest.skip("the Stage 1 arrays are not in this checkout")
    pytest.importorskip("mujoco")
    res = nb.inputs("", "unused", set_name="S1", bodies=["75-M-H-10"])
    m = nb.LINE_RE[2].match(res["line"])
    assert res["ok"] and m.group("bundle") == "S1-view" and m.group("ok") == m.group("n") and int(m.group("n")) > 10
    assert m.group("end") == "OK" and len(res["content_sha256"]) == 64
    bad = nb.inputs("", "unused", set_name="S1", bodies=["no-such-body"])
    assert not bad["ok"] and bad["line"].startswith("S2 LINE 2/4 inputs: bundle S1-view, manifest none, 0 of 0")


def test_preflight_prints_line_3_per_body_and_stops_at_a_failure(monkeypatch, capsys):
    def child(fail=None):
        def run(cmd, gl="disable", timeout=None, echo=True):
            assert gl == "disable"
            if "archive-preflight" in cmd or "preflight" in cmd:
                return 0, dict(ok=True, mujoco_gl="disable", bytecode_members=[], fingerprint_max_abs_diff=0.0), []
            if "envs" in cmd:
                pairs = json.loads(cmd[cmd.index("--pairs") + 1])
                return 0, dict(ok=True, built=[dict(mode=m, body=b, floor=f) for m, b in pairs
                                               for f in ("hard", "robot_floor_carpet")]), []
            body = cmd[cmd.index("--body") + 1]
            eps = [dict(condition="walk/v0.4", seed=3042, k=k, ms=2 * k, identical=True, first_difference=None,
                        compared_steps=3000, reference_steps=3000) for k in (0, 5)]
            if body == fail:
                eps[1].update(identical=False, first_difference=17)
            return 0, dict(ok=True, form="bar", passes=body != fail, identical=sum(e["identical"] for e in eps), of=2,
                           steps=6000, episodes=eps, not_run=[]), []
        return run

    monkeypatch.setattr(nb, "_json_child", child())
    res = nb.preflight(["R5:75-M-H-10:42", "R6:75-M-H-10:42", "R5:75-M-H-10-X:42"], "V6", bodies=["60-N-T-10"])
    assert res["ok"] and [nb.LINE_RE[3].match(x).group("body") for x in res["lines"]] == [
        "75-M-H-10", "75-M-H-10-X", "60-N-T-10"]                                  # once per body, the extra ones too
    assert res["lines"][0] == ("S2 LINE 3/4 G8a 75-M-H-10 set V6: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes "
                               "identical (6000 physics steps): PASS")
    assert len(res["envs"]["built"]) == 8                                         # both floors, each job's mode
    capsys.readouterr()
    monkeypatch.setattr(nb, "_json_child", child(fail="75-M-H-10-X"))
    res = nb.preflight(["R5:75-M-H-10:42", "R5:75-M-H-10-X:42", "R5:60-N-T-10:42"], "V6")
    assert not res["ok"] and len(res["lines"]) == 2 and res["problems"] == ["G8a fails on 75-M-H-10-X: no run"]
    assert res["lines"][1].endswith("1 of 2 episodes identical (6000 physics steps): FAIL (walk/v0.4 seed 3042 at 10 ms, "
                                    "first difference at step 17)")


# ------------------------------------------------------------------------------------------------
# collect, hours, the runtime's release
# ------------------------------------------------------------------------------------------------
def test_collect(tmp_path, stubs, monkeypatch, capsys):
    p, res, d = _run(tmp_path, {"*": dict(traces=True)}, monkeypatch, LOCAL_DIR=str(tmp_path / "drive"), QUEUE=["R5:B1:42"])
    assert res["state"] == "done"
    (d / "jobs" / "R5-B1-s42" / "partial").mkdir()
    (d / "jobs" / "R5-B1-s42" / "partial" / "x.json").write_text("{}")
    (d / "notes.txt").write_text("a note")
    (d / "logs" / "notes.txt").write_text("a text file under logs/ is no record")
    (d / "jobs" / "R5-B1-s42" / "checkpoints" / "notes.json").write_text("{}")   # of checkpoints/ only the records
    dst = tmp_path / "committed" / "r1"
    rec = nb.collect(d, dst, "run")
    got = sorted(x.relative_to(dst).as_posix() for x in dst.rglob("*") if x.is_file())
    assert not [x for x in got if x.startswith("logs/")] and "jobs/R5-B1-s42/checkpoints/notes.json" not in got
    assert got == sorted(list(rec["files"]) + ["COLLECTED.json"])
    assert "jobs/R5-B1-s42/run.json" in got and "jobs/R5-B1-s42/updates.jsonl" in got and "notes.txt" in got
    assert "DONE.json" in got and "QUEUE.json" in got and "SESSION_0.json" in got and "STATUS.json" in got
    # never the binaries, the traces, the logs, the launcher's work, a partial, or LIVE.json; of a checkpoint its record
    assert not [x for x in got if x.endswith((".zip", ".pkl", ".npz", ".mp4", ".tar", ".log"))]
    assert not [x for x in got if "/traces/" in x or "/logs/" in x or "work/" in x or "partial/" in x or "LIVE" in x]
    assert "jobs/R5-B1-s42/checkpoints/checkpoint_4096.json" in got and "jobs/R5-B1-s42/monitor/clip_4096.json" in got
    for rel, e in rec["files"].items():
        assert hashlib.sha256((dst / rel).read_bytes()).hexdigest() == e["sha256"] == \
            hashlib.sha256((d / rel).read_bytes()).hexdigest()
    assert rec["run_uuids"] == {"R5-B1-s42": "uuid-R5-B1-s42"} and rec["kind"] == "run" and rec["run_id"] == "r1"
    assert rec["schema"] == "s2-collected-1" and rec["sha256"] == nb.sha256_json({k: v for k, v in rec.items() if k != "sha256"})
    assert json.loads((dst / "COLLECTED.json").read_text()) == rec
    again = nb.collect(d, dst, "run")
    assert again["copied"] == [] and len(again["left"]) == len(rec["files"])     # a second time: nothing new
    # a collected record is never replaced
    (d / "notes.txt").write_text("another note")
    with pytest.raises(nb.Refused, match="never replaced"):
        nb.collect(d, dst, "run")
    assert (dst / "notes.txt").read_text() == "a note"
    with pytest.raises(nb.Refused, match="kind"):
        nb.collect(d, tmp_path / "x", "runs")
    # a job that is still running is refused: a fresh LIVE.json without an end; a live process under a running run.json
    live = tmp_path / "live"
    shutil.copytree(d, live)
    (live / "DONE.json").unlink()
    doc = json.loads((live / "LIVE.json").read_text())
    doc.pop("closed_at")
    (live / "LIVE.json").write_text(json.dumps(doc))
    with pytest.raises(nb.Refused, match="still running .LIVE.json was written"):
        nb.collect(live, tmp_path / "x", "run")
    (live / "LIVE.json").unlink()
    jd = live / "jobs" / "R5-B1-s42"
    run = json.loads((jd / "run.json").read_text())
    (jd / "run.json").write_text(json.dumps(dict(run, status="running")))
    prog = json.loads((jd / "progress.json").read_text())
    (jd / "progress.json").write_text(json.dumps(dict(prog, pid=os.getpid())))
    with pytest.raises(nb.Refused, match="reads running and its process"):
        nb.collect(live, tmp_path / "x", "run")
    (jd / "progress.json").write_text(json.dumps(dict(prog, pid=2 ** 22 + 12345)))
    rec = nb.collect(live, tmp_path / "x", "bar")                                 # a dead one is collected, and named
    assert rec["unfinished"] == [dict(job="R5-B1-s42", status="running")] and not (tmp_path / "x" / "LIVE.json").exists()
    (live / ".lock").write_text(json.dumps(dict(pid=os.getpid(), host=__import__("socket").gethostname())))
    with pytest.raises(nb.Refused, match=".lock is held by pid"):
        nb.collect(live, tmp_path / "y", "r2l")


def test_hours(tmp_path, capsys):
    root = tmp_path / "runs"
    for run, rows in (("a", [("2026-11-20T00:00:00Z", "2026-11-20T10:30:00Z"), ("2026-11-21T00:00:00Z", "2026-11-21T09:00:00Z")]),
                      ("b", [("2026-11-22T00:00:00Z", "2026-11-22T00:45:00Z"), ("2026-11-23T00:00:00Z", None)])):
        (root / run).mkdir(parents=True)
        for k, (a, b) in enumerate(rows):
            (root / run / f"SESSION_{k}.json").write_text(json.dumps(dict(session=k, started_at=a, last_status_at=b)))
    (root / "empty").mkdir()
    rec = nb.hours(root, out=tmp_path / "hours.json")
    assert rec["runs"]["a"]["hours"] == pytest.approx(19.5) and rec["runs"]["b"]["hours"] == pytest.approx(0.75)
    assert rec["total_hours"] == pytest.approx(20.25) and rec["remaining_hours"] == pytest.approx(199.75)
    assert not rec["over_cap"] and rec["without_status"] == ["b/SESSION_1.json"] and "empty" not in rec["runs"]
    assert json.loads((tmp_path / "hours.json").read_text())["sha256"] == rec["sha256"]
    assert "20.25 of 220 runtime-hours" in capsys.readouterr().out
    (root / "a" / "SESSION_2.json").write_text(json.dumps(dict(session=2, started_at="2026-11-01T00:00:00Z",
                                                               last_status_at="2026-11-09T08:00:00Z")))
    assert nb.hours(root, out=None)["over_cap"] is True
    # ruling 1 of 17 Nov: a runtime kept for a save on it only counts to the last finish, and is listed
    (root / "c").mkdir()
    (root / "c" / "SESSION_0.json").write_text(json.dumps(dict(
        session=0, started_at="2026-11-24T00:00:00Z", last_status_at="2026-11-24T21:00:00Z",
        finishes=[dict(at="2026-11-24T20:00:00Z", on_drive=False), dict(at="2026-11-24T21:00:00Z", on_drive=False)])))
    rec = nb.hours(root, out=None)
    assert rec["runs"]["c"]["hours"] == pytest.approx(21.0) and rec["kept_runtime"] == ["c/SESSION_0.json"]
    assert rec["runs"]["c"]["sessions"][0]["kept_runtime"] is True and "kept the runtime" in capsys.readouterr().out


def test_release_runtime(monkeypatch, capsys):
    before = set(sys.modules)
    nb.release_runtime("over", in_colab=False, auto=True, flush_drive=True)
    nb.release_runtime("over", in_colab=True, auto=False, flush_drive=True)
    assert capsys.readouterr().out.count("Auto-disconnect skipped") == 2
    assert not {m for m in set(sys.modules) - before if m.split(".")[0] in ("environments", "google")}
    # on Colab: the repository's disconnect_runtime first; whatever it raises, Drive is flushed and the runtime unassigned
    import types

    done = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(flush_and_unmount=lambda: done.append("flush"))
    colab.runtime = types.SimpleNamespace(unassign=lambda: done.append("unassign"))
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    broken = types.ModuleType("environments.shared.notebook_runtime")   # (no disconnect_runtime: the import fails)
    monkeypatch.setitem(sys.modules, "environments.shared.notebook_runtime", broken)
    monkeypatch.setattr(nb.time, "sleep", lambda s: done.append(f"sleep {s}"))
    nb.release_runtime("the run is over", in_colab=True, auto=True, flush_drive=True)
    assert done == ["flush", "sleep 5", "unassign"] and "the run is over" in capsys.readouterr().out
    done.clear()
    broken.disconnect_runtime = lambda reason, in_colab, auto, flush_drive: done.append(("repo", reason, flush_drive))
    nb.release_runtime("the run is over", in_colab=True, auto=True, flush_drive=False)
    assert done == [("repo", "the run is over", False)]
    # whatever the flush itself raises, the runtime is unassigned (design D19); as built it stayed allocated
    done.clear()
    del broken.disconnect_runtime

    def flush():
        done.append("flush")
        raise OSError(107, "Transport endpoint is not connected")

    colab.drive = types.SimpleNamespace(flush_and_unmount=flush)
    nb.release_runtime("the run is over", in_colab=True, auto=True, flush_drive=True)
    assert done == ["flush", "sleep 5", "unassign"] and "could not be flushed" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------
# The smoke on the stubs; no checkout dirt; the cells as Python
# ------------------------------------------------------------------------------------------------
def test_smoke_on_the_stubs(tmp_path, stubs, monkeypatch, capsys):
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(traces=True, tick=0.03, rate=560.0)}))
    p = _params(tmp_path, SMOKE=True, QUEUE=[], RUN_ID="smoke1", REPO_REF="s2-launch-1", PERIOD_S=0.05,
                SETUP_RECORD=str(tmp_path / "setup.json"))
    (tmp_path / "setup.json").write_text(json.dumps(dict(started_epoch=time.time() - 30, line=nb.line1(COMMIT, False, True))))
    head = nb.repo_state()["commit"]          # (the smoke's own tiny run opens its session in a child, on this checkout)
    _session(p, commit=head)                                                      # a smoke may name a tag
    capsys.readouterr()
    doc = nb.smoke("smoke1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    out = capsys.readouterr().out
    d = tmp_path / "drive" / "runs" / "smoke1"
    lines = nb.lines_of(out)["smoke"]
    assert [nb.LINE_RE["smoke"].match(x).group("i") for x in lines] == list("12345678")   # rules Y1-Y8, in that order
    verdicts = {k: v["verdict"] for k, v in doc["items"].items()}
    assert verdicts == dict(Y1="OK", Y2="REPORTED", Y3="OK", Y4="OK", Y5="REPORTED", Y6="OK", Y7="REPORTED",
                            Y8="REPORTED"), doc["items"]
    assert doc["body"] == nb.SMOKE_BODY and doc["rate"] == pytest.approx(560.0) and doc["budget"] == 40_000_000
    y6 = doc["items"]["Y6"]
    assert y6["resumed_from"] > 0 and y6["fingerprint_max_abs_diff"] == 0.0 and y6["steps"] == 4096 and y6["done"]
    assert doc["items"]["Y5"]["ok"] and (d / "smoke" / "clip" / "clip_test.mp4").is_file()
    assert doc["items"]["Y7"]["seconds"] >= 30 and "not run: no bundle" in doc["items"]["Y8"]["value"]
    written = json.loads((d / "smoke" / "SMOKE.json").read_text())
    assert written["written_after"] == "Y8" and written["schema"] == "s2-smoke-1" and written["lines"] == lines
    assert written["seams"] and written["tiny"] is True
    assert (d / "smoke" / "rate" / "updates.jsonl").is_file() and (d / "smoke" / "kill" / "runs" / "y6" / "DONE.json").is_file()
    assert json.loads((d / "STATUS.json").read_text())["lines"][-8:] == lines
    assert _launch(p)["state"] == "refused"                                       # a smoke queues no training job
    fin = nb.finish("smoke1", p["DRIVE_DIR"], p["LOCAL_DIR"])
    assert fin["state"] == "done" and "smoke" in json.loads((d / "DONE.json").read_text())["reason"]
    # a slow machine: under 15,000,000 steps in 20 h rule Y4 fails and no budget is given
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.0, rate=150.0)}))
    p2 = _params(tmp_path / "slow", SMOKE=True, QUEUE=[], RUN_ID="smoke2", REPO_REF="main")
    _session(p2, commit=head)
    doc = nb.smoke("smoke2", p2["DRIVE_DIR"], p2["LOCAL_DIR"])
    assert doc["items"]["Y4"]["verdict"] == "FAILED" and doc["budget"] is None and "owner" in doc["items"]["Y4"]["value"]


def test_the_smokes_clip_is_drawn_from_a_trace_taken_after_the_resume(tmp_path, stubs, monkeypatch):
    """Rule Y5.  As built the trace was looked for only until the kill (and copied while its writer lived), so a
    tiny job whose first check came after its first copied checkpoint left no clip."""
    # the tiny job's only check is its last one: no trace exists yet when its first checkpoint is on Drive
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(traces=True, tick=0.15, rate=560.0, monitor_every=4096)}))
    p = _params(tmp_path, SMOKE=True, QUEUE=[], RUN_ID="smoke3", REPO_REF="main", PERIOD_S=0.05)
    _session(p, commit=nb.repo_state()["commit"])
    doc = nb.smoke("smoke3", p["DRIVE_DIR"], p["LOCAL_DIR"])
    y5, y6 = doc["items"]["Y5"], doc["items"]["Y6"]
    assert y6["verdict"] == "OK" and y6["killed_at"]["first_checkpoint_on_drive"] < 4096, y6
    assert (y6["killed_at"]["progress"] or 0) < 4096                 # killed before the job's only check
    assert y5["ok"] and "an MP4 of 3 s" in y5["value"], y5
    clip = tmp_path / "drive" / "runs" / "smoke3" / "smoke" / "clip" / "clip_test.mp4"
    assert clip.read_bytes() == b"mp4 of trace 4096"                 # the check the resumed job made


def test_acceptance_refuses_a_scratch_directory_that_is_not_its_own(tmp_path):
    """The acceptance run deletes its scratch directory first.  As built it deleted whatever directory outside
    the checkout it was given."""
    scratch = tmp_path / "s"
    scratch.mkdir()
    (scratch / "keep.txt").write_text("mine")
    with pytest.raises(nb.Refused, match="is not empty and is not an earlier acceptance run's"):
        nb.acceptance(NOTEBOOK, nb.repo_state()["commit"], tmp_path / "x" / "parts", scratch, tmp_path / "a.json")
    assert (scratch / "keep.txt").read_text() == "mine"
    (tmp_path / "file").write_text("x")
    with pytest.raises(nb.Refused, match="is not empty and is not an earlier acceptance run's"):
        nb.acceptance(NOTEBOOK, nb.repo_state()["commit"], tmp_path / "x" / "parts", tmp_path / "file", tmp_path / "a.json")


@needs_git
def test_the_acceptance_runs_plumbing_on_stubs(tmp_path, monkeypatch):
    """``acceptance`` needs the pushed launch commit and the built bundle and **has never run**.  Here its own
    plumbing runs on a scratch repository with the drill replaced by a stand-in: the remote's SHA, the two fresh
    clones and their HEADs, what a dead VM loses (the clone and the whole local directory, the inputs' local copies
    included), the record's fields of design 3.10, and the scratch directory's mark."""
    b = _bundle(tmp_path)
    parts = b["root"] / nb.BUNDLE_DIR / "tb-test" / "parts"
    parts.mkdir()
    shutil.copy(b["parts"] / "models-B1.tar", parts / "models-B1.tar")
    sha = b["manifest"]["content_sha256"]
    env = dict(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_AUTHOR_NAME="t",
               GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    def git(cwd, *args):
        return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout

    remote, main = tmp_path / "remote", tmp_path / "main"
    (remote / "study").mkdir(parents=True)
    git(remote, "init", "-q", "-b", "main")
    (remote / "study" / "README").write_text("the study")
    git(remote, "add", ".")
    git(remote, "commit", "-q", "-m", "the launch commit")
    git(tmp_path, "clone", "-q", str(remote), str(main))
    ref = git(main, "rev-parse", "HEAD").strip()
    monkeypatch.setattr(nb, "study", lambda: main / "study")         # (the checkout the acceptance run is started in)
    monkeypatch.setattr(nb, "ROOT", STUDY)                           # (the pins file, for the record's versions)
    scratch = tmp_path / "acc"
    drive, seen = scratch / "drive", {}

    def drill(notebook, params, logs, cwd_of, kill_when, wipe=None):
        local, first = Path(params["LOCAL_DIR"]), cwd_of(0)
        assert first == scratch / "clone1" / "study" and (first / "README").is_file() and kill_when() is None
        (local / "inputs" / "tb-test" / "parts").mkdir(parents=True)  # what the inputs step leaves on the local disk
        (local / "inputs" / "tb-test" / "parts" / "models-B1.tar").write_bytes(b"a local copy")
        (local / "acceptance" / "jobs").mkdir(parents=True)
        wipe()
        seen.update(clone_gone=not (scratch / "clone1").exists(), local_gone=not local.exists(),
                    on_drive=sorted(x.name for x in (drive / "inputs" / "tb-test").iterdir()))
        second = cwd_of(1)                                           # (the drill's order: attempts 1, 2, 3)
        assert second == scratch / "clone2" / "study" and (second / "README").is_file()
        assert cwd_of(2) == second and cwd_of(3) == second and cwd_of(0) == first
        lines = {k: [] for k in nb.LINE_RE}
        lines[2], lines[3] = [nb.line2("tb-test", sha, 2, 2)], [nb.line3("B1", "V6", 2, 2, 6000)]
        one = dict(exit_code=0, killed=None, seconds=1.0, text="", lines=lines, log="x")
        return dict(drive_run=str(drive / "runs" / "acceptance"), local_run=str(local / "acceptance"),
                    first_job="R5-75-M-H-10-s42", kill=dict(when="model_1024.zip.part exists (a copy in progress)"),
                    attempts=dict(session0=one, refused_live=one, session1=one, refused_finished=one))

    def remedies(notebook, params, logs, cwd):
        seen["remedies_cwd"] = cwd
        return dict(run_id=params["RUN_ID"] + "-remedies", checks=dict(remedies_stand_in=True))

    monkeypatch.setattr(nb, "_drill", drill)
    monkeypatch.setattr(nb, "_drill_checks", lambda drill, params: dict(stand_in=True))
    monkeypatch.setattr(nb, "_remedies", remedies)
    rec = nb.acceptance("stage2/stage2_training.ipynb", ref, parts, scratch, tmp_path / "ACCEPTANCE.json")
    assert json.loads((tmp_path / "ACCEPTANCE.json").read_text())["sha256"] == rec["sha256"]
    # design 3.10: the remote URL, ls-remote's SHA, the clone's HEAD, the content hash, the G8a counts per body
    assert rec["remote"] == rec["clone_source"] == str(remote) and rec["ls_remote_sha"] == ref == rec["ref"]
    assert rec["clone_head"] == ref and rec["clone_heads"] == {"1": ref, "2": ref} and rec["content_sha256"] == sha
    assert rec["g8a"] == {"B1": dict(set="V6", identical=2, of=2, steps=6000, passes=True)} and rec["files"] == 2
    assert rec["checks"]["head_is_ref"] and rec["checks"]["fresh_clones_hold_nothing_ignored"]
    assert rec["checks"]["content_hash"] and rec["checks"]["g8a_every_body"] and rec["bodies"] == ["B1"]
    # step 4: a dead VM loses the clone and its whole local disk; the stand-in Drive keeps the parts
    assert seen.pop("remedies_cwd") == scratch / "clone2" / "study"            # step 6 runs in the second clone
    assert seen == dict(clone_gone=True, local_gone=True, on_drive=["models-B1.tar"])
    assert rec["checks"]["remedies_stand_in"] and rec["remedies"]["run_id"] == "acceptance-remedies"
    assert "not_done" not in rec and "step 6" in rec["step6_note"]
    # the scratch directory is marked as this function's own, so a second run may delete and remake it
    assert (scratch / nb.ACCEPTANCE_MARK).is_file()
    (scratch / "left-over.txt").write_text("of the first run")
    again = nb.acceptance("stage2/stage2_training.ipynb", ref, parts, scratch, tmp_path / "ACCEPTANCE2.json")
    assert again["clone_head"] == ref and not (scratch / "left-over.txt").exists()
    assert nb.g8a_counts([nb.line3("B2", "V6", 1, 2, 5000, "the gate ended with exit code 1"), "noise"]) == {
        "B2": dict(set="V6", identical=1, of=2, steps=5000, passes=False)}


def test_design_3_10_step_6_the_leads_remedies_after_a_faked_record(tmp_path, stubs, monkeypatch):
    """Design 3.10 step 6 as the lead's ruling 2 of 17 Nov has it (``_remedies``, which ``acceptance`` and
    ``local_run`` run), on the stub trainer through the notebook's own cells: a session killed once its job's first
    checkpoint is on the stand-in Drive; a faked record (QUEUE.json's commit, the job's last session's library
    versions); a resume refused for the commit; LEAD.json's allow_source_change, and a resume that the trainer
    refuses for the versions (exit code 4: no HALT.json); LEAD.json's allow_version_change too, and a resume to
    DONE.  As built, step 6 was not done by acceptance at all ("not_done")."""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(tick=0.05, budget=512 * 200)}))
    real, logs = nb.run_cells, []

    def run_cells(*a, **k):
        res = real(*a, **k)
        logs.append(Path(a[2]).name)
        os.environ["STUB_PLAN"] = json.dumps({"*": dict(tick=0.0, budget=512 * 200)})   # (later sessions: at once)
        return res

    monkeypatch.setattr(nb, "run_cells", run_cells)
    p = _params(tmp_path, RUN_ID="acc", REPO_REF=nb.repo_state()["commit"], PERIOD_S=0.2, CLIPS=False,
                STUDY_DIR=str(STUDY))
    rem = nb._remedies(NOTEBOOK, p, tmp_path / "logs", STUDY)
    assert rem["checks"] and all(rem["checks"].values()), json.dumps(rem, indent=1)[:4000]
    assert logs == ["remedies-0.log", "remedies-1-refused-source.log", "remedies-2-refused-by-the-trainer.log",
                    "remedies-3-done.log"]
    d = tmp_path / "drive" / "runs" / "acc-remedies"
    assert rem["run_id"] == "acc-remedies" and rem["faked"]["queue_commit"] == "0" * 40
    assert rem["faked"]["version"] == dict(key="numpy", was="stub-1", now=nb.FAKED_VERSION[1])
    assert sorted(x.name for x in d.glob("SESSION_*.json")) == ["SESSION_0.json", "SESSION_1.json", "SESSION_2.json"]
    s1, s2 = (json.loads((d / f"SESSION_{k}.json").read_text()) for k in (1, 2))
    assert s1["ended"]["state"] == "refused" and s1["allow_source_change"] and not s1["allow_version_change"]
    assert s2["lead"]["doc"]["allow_version_change"] is True and s2["source_change"] is True
    assert not (d / "HALT.json").exists() and json.loads((d / "DONE.json").read_text())["state"] == "done"
    argv = [c["argv"] for c in _calls(dict(p, RUN_ID="acc-remedies"))]
    assert "--allow-source-change" in argv[-2] and "--allow-version-change" not in argv[-2]
    assert "--allow-version-change" in argv[-1] and "--resume" in argv[-1]
    run = json.loads((d / "jobs" / "R5-B1-s42" / "run.json").read_text())
    assert run["sessions"][-1]["version_change"] is True and run["sessions"][-1]["resumed_from"] >= 1024


@needs_git
def test_a_local_job_leaves_git_status_unchanged(tmp_path, stubs, monkeypatch):
    """Design D22: a job under ``stage2/tb/work/`` of a checkout changes nothing git sees (a scratch repository
    that holds Track B's own ignore file)."""
    repo = tmp_path / "repo"
    (repo / "stage2" / "tb").mkdir(parents=True)
    shutil.copy(STUDY / "stage2" / "tb" / ".gitignore", repo / "stage2" / "tb" / ".gitignore")
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_AUTHOR_NAME="t",
               GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "the ignore file")
    assert git("status", "--porcelain", "--untracked-files=all") == ""
    monkeypatch.setenv("STUB_PLAN", json.dumps({"*": dict(traces=True)}))
    work = repo / "stage2" / "tb" / "work"
    for name, local in (("same", work / "s24"), ("apart", work / "s24-local")):   # with and without a stand-in Drive
        p = _params(tmp_path, RUN_ID=name, DRIVE_DIR=str(work / "s24"), LOCAL_DIR=str(local))
        _session(p)
        seen = []
        t = threading.Thread(target=lambda: seen.append(_launch(p)))
        t.start()
        while t.is_alive():
            assert git("status", "--porcelain", "--untracked-files=all") == ""    # while it runs
            time.sleep(0.05)
        t.join()
        assert seen[0]["state"] == "done" and git("status", "--porcelain", "--untracked-files=all") == ""
    written = sorted(x.relative_to(repo).as_posix() for x in work.rglob("*") if x.is_file())
    assert len(written) > 60 and any("checkpoints/model_" in x for x in written)
    got = subprocess.run(["git", "-C", str(repo), "check-ignore", "-v", "--stdin"], input="\n".join(written),
                         capture_output=True, text=True, env=env).stdout.splitlines()
    assert len(got) == len(written) and all(x.startswith("stage2/tb/.gitignore:1:work/") for x in got)
    # the collected records, in their committed place, are what git then sees
    nb.collect(work / "s24" / "runs" / "same", repo / "stage2" / "tb" / "runs" / "same", "run")
    seen = [x[3:] for x in git("status", "--porcelain", "--untracked-files=all").splitlines()]
    assert seen and all(x.startswith("stage2/tb/runs/same/") and x.endswith((".json", ".jsonl", ".txt")) for x in seen)
    assert "stage2/tb/runs/same/jobs/R5-B1-s42/checkpoints/checkpoint_4096.json" in seen


def test_run_cells_ends_the_whole_session(tmp_path):
    """The stand-in for the kernel: cells in order in one namespace; a kill takes every process of its session."""
    book = dict(cells=[
        dict(cell_type="markdown", source=["text"]),
        dict(cell_type="code", source=["import json, os, subprocess, sys, time\n", "x = json.load(open(os.environ['S2_PARAMS']))['RUN_ID']\n"]),
        dict(cell_type="code", source=[
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'], process_group=0)\n",
            "print('S2 SMOKE Y2 machine: ' + x + ' ' + str(child.pid) + ': REPORTED', flush=True)\n",
            "time.sleep(600)\n"]),
        dict(cell_type="code", source=["print('never')\n"])])
    (tmp_path / "book.ipynb").write_text(json.dumps(book))
    log = tmp_path / "logs" / "k.log"
    got = nb.run_cells(tmp_path / "book.ipynb", dict(RUN_ID="abc"), log, cwd=tmp_path, cells=(1, 2, 3),
                       until=lambda: "seen" if log.is_file() and "REPORTED" in log.read_text() else None)
    assert got["killed"] == "seen" and got["exit_code"] != 0 and "never" not in got["text"]
    line = got["lines"]["smoke"][0]
    assert " abc " in line and not nb._alive(int(line.split()[5][:-1]))           # the cell's own child died with it
    done = nb.run_cells(tmp_path / "book.ipynb", dict(RUN_ID="abc"), tmp_path / "logs" / "k2.log", cwd=tmp_path, cells=(1, 3))
    assert done["exit_code"] == 0 and done["killed"] is None and "never" in done["text"]
    boom = dict(cells=[dict(cell_type="code", source=["raise RuntimeError('S2 step stopped')\n"]),
                       dict(cell_type="code", source=["print('never')\n"])])
    (tmp_path / "boom.ipynb").write_text(json.dumps(boom))
    bad = nb.run_cells(tmp_path / "boom.ipynb", {}, tmp_path / "logs" / "k3.log", cwd=tmp_path, cells=(0, 1))
    assert bad["exit_code"] == 1 and "never" not in bad["text"]                   # a cell that raises ends "Run all"


def test_the_command_line(tmp_path, stubs):
    p = _params(tmp_path, QUEUE=["R5:B1:42"])
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(p))
    env = dict(os.environ, PYTHONPATH=".")

    def run(*args):
        return subprocess.run([sys.executable, "-m", "designlab.stage2.nb", *args], cwd=STUDY, env=env,
                              capture_output=True, text=True, timeout=120)

    head = nb.repo_state()["commit"]
    done = run("session", "--params", str(pfile))                                 # REPO_REF is not the checkout's commit
    assert done.returncode == nb.EXIT_REFUSED and done.stdout.startswith("S2 REFUSED session: REPO_REF")
    ok = run("session", "--params", str(pfile), "--set-param", f'REPO_REF="{head}"')
    assert ok.returncode == 0 and nb.LINE_RE["new"].match(ok.stdout.strip().splitlines()[-1]), ok.stdout + ok.stderr
    again = run("session", "--params", str(pfile), "--set-param", f'REPO_REF="{head}"', "--set-param", "RESUME=true")
    assert again.returncode == nb.EXIT_REFUSED and nb.LINE_RE["live"].match(again.stdout.strip())   # the line itself
    # a step that fails before any job could start closes the session: the next resume is not held back ten minutes
    bad = run("inputs", "--params", str(pfile), "--set-param", f'REPO_REF="{head}"', "--set-param", 'SET="V6"')
    assert bad.returncode == nb.EXIT_FAILED and "S2 LINE 2/4 inputs: bundle none" in bad.stdout and "FAILED (" in bad.stdout
    d = tmp_path / "drive" / "runs" / "r1"
    assert json.loads((d / "LIVE.json").read_text())["closed_reason"].startswith("line 2 failed")
    assert json.loads((d / "STATUS.json").read_text())["lines"][-1].startswith("S2 LINE 2/4 inputs: bundle none")
    more = run("session", "--params", str(pfile), "--set-param", f'REPO_REF="{head}"', "--set-param", "RESUME=true")
    assert more.returncode == 0 and nb.LINE_RE["resume"].match(more.stdout.strip().splitlines()[-1]).group("session") == "1"
    go = run("launch", "--params", str(pfile), "--set-param", f'REPO_REF="{head}"')
    assert go.returncode == 0 and "S2 launch: DONE: the queue is finished" in go.stdout, go.stdout + go.stderr
    fin = run("finish", "--params", str(pfile))
    assert fin.returncode == 0 and "S2 FINISH r1: DONE" in fin.stdout
    col = run("collect", "--src", str(tmp_path / "drive" / "runs" / "r1"), "--dst", str(tmp_path / "c"), "--kind", "run")
    assert col.returncode == 0 and (tmp_path / "c" / "COLLECTED.json").is_file()
    hrs = run("hours", "--runs-root", str(tmp_path / "drive" / "runs"), "--out", str(tmp_path / "h.json"))
    assert hrs.returncode == 0 and json.loads((tmp_path / "h.json").read_text())["schema"] == "s2-hours-1"
    bad = run("acceptance", "--ref", "main", "--parts", "x", "--scratch", str(tmp_path / "s"), "--record", str(tmp_path / "a.json"))
    assert bad.returncode == nb.EXIT_REFUSED and "40 hex digits" in bad.stdout
    inside = run("acceptance", "--ref", head, "--parts", "x", "--scratch", "stage2/tb/work/acc", "--record", str(tmp_path / "a.json"))
    assert inside.returncode == nb.EXIT_REFUSED and "inside the checkout" in inside.stdout


@slow
def test_local_run_with_a_kill_and_a_resume(tmp_path):
    """The notebook's own cells in this checkout on the Stage 1 view with ``--tiny``: session 0 killed after the
    first job's first copied checkpoint, a resume refused while LIVE.json is fresh, a resume to DONE."""
    if not (STUDY / "stage1/runs/2026-11-08/75-M-H-10/refs").is_dir():
        pytest.skip("the Stage 1 arrays are not in this checkout")
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    rec = nb.local_run(NOTEBOOK, tmp_path / "drive", dict(RUN_ID="local"), record=tmp_path / "LOCAL_RUN.json")
    assert rec["ok"], json.dumps(dict(checks=rec["checks"], kill=rec["kill"],
                                      attempts={k: (v["exit_code"], v["lines"]) for k, v in rec["attempts"].items()}),
                                 indent=1)
    d = tmp_path / "drive" / "runs" / "local"
    assert (d / "DONE.json").is_file() and (d / "SESSION_0.json").is_file() and (d / "SESSION_1.json").is_file()
    first = json.loads((d / "jobs" / "R5-75-M-H-10-s42" / "run.json").read_text())
    assert first["status"] == "done" and first["sessions"][-1]["resumed_from"] >= 1024
    assert json.loads((d / "jobs" / "R6-75-M-H-10-s42" / "run.json").read_text())["status"] == "done"
    lines = rec["attempts"]["session1"]["lines"]
    assert len(lines["4"]) == 2 and all(nb.LINE_RE[4].match(x) for x in lines["4"])
    assert rec["checks"]["git_status_unchanged"] and json.loads((tmp_path / "LOCAL_RUN.json").read_text())["ok"]
