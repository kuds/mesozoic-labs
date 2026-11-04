"""Run walking pilot 3's Colab steps locally on a clone of the pilot branch: the notebook smoke (not a judging tool).

Scratch, never merged; frozen by sha256 before the smoke. Pilot 1's tools/nb_pilot_runner.py (cc87f44f...) and pilot
2's extension of it (scripts/nb_pilot_runner_guard.py) are lost; this is a rebuild to the contract pilot 2 states
(main docs/investigations/stance_2026_10/pilot2/criteria_2026-10-21.txt lines 395-405): the Colab steps' cells, with
the five MANUAL lines for the parent and /content/mesozoic-labs replaced by the clone root (--params), the guard line
typed as the chain-loop cell's first line, "Run all" over the code cells, which the guard stops at the chain loop, then
the manual cell alone, then section 9's cell; one namespace (one kernel); stdout and stderr to logs; record.json.
Walking pilot 3 adds (criteria_pilot3.txt "Pre-launch checks"):

  --cadence-override N  In memory only, the locomotion stage's [curriculum] eval_freq and checkpoint_freq are set to N
                        in STAGE_CONFIGS, the dict the notebook's train_stage hands to train_base.train, right after
                        the cell that builds it (the resolve cell) has run. With N = 32,768 a QUICK_TEST node ends at
                        65,536 with pairs and evaluations at 32,768 and 65,536. The runner refuses unless the stage's
                        hyperparameters_sha256 is the same before and after (the override enters no digest); the
                        override shows in stage_config.json's curriculum block.
  --phase resume        The notebook's RESUME cell: RESUME_STAGE = <params resume_stage> (section 5), then Run all,
                        which the guard stops after the RESUME cell trained; then section 9's cell. The manual cell
                        is not run.
  --stop-after-pair N   Simulate a runtime loss: once the periodic pair stage2_<N>_steps.zip with
                        stage2_vecnormalize_<N>_steps.pkl is on disk (SB3's CheckpointCallback, which the trainer
                        uses), record.json is written and the process exits at once (os._exit(75)), with no final save,
                        as a reclaimed runtime would.
  --plan-only           Write the edited cells (cells_edited.json) and stop: nothing runs.

    cd CLONE && OMP_NUM_THREADS=1 nice -n 10 python3 -I TOOLS/nb_pilot_runner3.py --params PARAMS.json --out LOGDIR
        [--phase launch|resume] [--cadence-override 32768] [--stop-after-pair 32768] [--plan-only]

PARAMS (JSON): {"repo_ref": "claude/pilot-walk-r10-p3", "config": {"SPECIES": "tyrannosaurus_rex", ..., "RUN_ID":
"pilot3_walk_s42"}, "manual": ["MANUAL_NODE = \\"locomotion\\"", ... the five lines as the Colab steps type them],
"guard_line": "raise RuntimeError(...)", "resume_stage": "locomotion"}. Every config name must be assigned on exactly
one line of the configuration cell, and every MANUAL_ name on one line of the manual cell, else the runner refuses.
Cells are found by their banners, never by index (the indices found are recorded). The Colab branch of the setup cell
is never taken: the runner refuses where /content exists or a Colab variable is set. MPLBACKEND=Agg; without IPython
a stand-in IPython.display module prints what a notebook would display. Exit status: 0 when every cell ran as the
phase expects (the guard's RuntimeError at the chain loop is expected), 1 otherwise, 2 for a refusal before any cell.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

STOP_EXIT = 75
CONTENT_ROOT = "/content/mesozoic-labs"
MARKERS = {  # cell role -> a line its source holds (the notebook's banners)
    "setup": "# Install dependencies (Colab auto-detected",
    "config": "# ===== SPECIES SELECTION =====",
    "resume": "# ===== RESUME AN INTERRUPTED STAGE",
    "chain": "# ===== BEHAVIOR CHAIN LOOP =====",
    "manual": "# ===== MANUAL SINGLE NODE",
    "disconnect": "disconnect_runtime(",
}
MANUAL_NAMES = ("MANUAL_NODE", "MANUAL_LOAD_PATH", "MANUAL_VECNORM_PATH", "MANUAL_LOAD_MODE", "MANUAL_TIMESTEPS")
STAGE_FOR_OVERRIDE = "locomotion"
RECORD: dict = {}
RECORD_PATH: Path | None = None


class RunnerRefusal(Exception):
    pass


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def find_cells(nb: dict) -> dict:
    """{role: index} for every role in MARKERS: the one code cell whose source holds the marker (the disconnect cell:
    the last code cell that starts with it)."""
    found = {}
    code = [(i, cell_source(c)) for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"]
    for role, marker in MARKERS.items():
        if role == "disconnect":
            hits = [i for i, s in code if s.lstrip().startswith(marker)]
            hits = hits[-1:]
        else:
            hits = [i for i, s in code if marker in s]
        if len(hits) != 1:
            raise RunnerRefusal(f"the {role} cell (marker {marker!r}) is found {len(hits)} times, not once")
        found[role] = hits[0]
    return found


def replace_assignment(src: str, name: str, new_line: str) -> str:
    """Replace the one line that assigns *name* at the start of a line (``NAME = ...``) by *new_line*."""
    pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", re.M)
    hits = pattern.findall(src)
    if len(hits) != 1:
        raise RunnerRefusal(f"{name} is assigned on {len(hits)} lines of its cell, not one")
    return pattern.sub(lambda _m: new_line, src, count=1)


def edit_cells(nb: dict, params: dict, clone: Path, phase: str) -> tuple[dict, dict, list]:
    """The Colab steps' edits, on a copy of the notebook's cells: (sources by index, roles, edits applied)."""
    roles = find_cells(nb)
    sources = {i: cell_source(c) for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"}
    edits = []
    i = roles["setup"]
    if params.get("repo_ref"):
        sources[i] = replace_assignment(sources[i], "REPO_REF", f"REPO_REF = {params['repo_ref']!r}  # nb_pilot_runner3")
        edits.append(("setup", "REPO_REF", params["repo_ref"]))
    i = roles["config"]
    for name, value in params["config"].items():
        sources[i] = replace_assignment(sources[i], name, f"{name} = {value!r}  # nb_pilot_runner3 (Colab steps)")
        edits.append(("config", name, value))
    i = roles["manual"]
    manual = {}
    for line in params["manual"]:
        name = line.split("=", 1)[0].strip()
        if name not in MANUAL_NAMES:
            raise RunnerRefusal(f"manual line {line!r} sets {name!r}, not one of {MANUAL_NAMES}")
        manual[name] = line.replace(CONTENT_ROOT, str(clone))
    if sorted(manual) != sorted(MANUAL_NAMES):
        raise RunnerRefusal(f"params.manual sets {sorted(manual)}, not the five {sorted(MANUAL_NAMES)}")
    for name in MANUAL_NAMES:
        sources[i] = replace_assignment(sources[i], name, manual[name])
        edits.append(("manual", name, manual[name]))
    i = roles["chain"]
    guard = params["guard_line"]
    if "\n" in guard or not guard.startswith("raise RuntimeError("):
        raise RunnerRefusal("the guard line must be one line that raises RuntimeError")
    sources[i] = guard + "\n" + sources[i]
    edits.append(("chain", "guard line", guard))
    if phase == "resume":
        i = roles["resume"]
        sources[i] = replace_assignment(sources[i], "RESUME_STAGE",
                                        f"RESUME_STAGE = {params['resume_stage']!r}  # nb_pilot_runner3 (resume)")
        edits.append(("resume", "RESUME_STAGE", params["resume_stage"]))
    for idx, src in sources.items():
        compile(src, f"<cell {idx}>", "exec")
    return sources, roles, edits


def guard_message(guard_line: str) -> str:
    """The message the guard line raises (the string literal inside RuntimeError(...))."""
    import ast

    node = ast.parse(guard_line).body[0]
    assert isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
    return str(ast.literal_eval(node.exc.args[0]))


def apply_cadence_override(ns: dict, value: int, digest=None) -> dict:
    """Set the locomotion stage's curriculum eval_freq and checkpoint_freq to *value* in ns["STAGE_CONFIGS"]."""
    ref = ns["MANIFEST"].resolve(STAGE_FOR_OVERRIDE).reference
    cfg = ns["STAGE_CONFIGS"][ref]
    cur = cfg.setdefault("curriculum_kwargs", {})
    if digest is None:
        from environments.shared.config import hyperparameters_sha256 as digest
    before_digest = digest(cfg, "ppo")
    before = {k: cur.get(k) for k in ("eval_freq", "checkpoint_freq", "eval_episodes", "max_checkpoints")}
    cur["eval_freq"] = int(value)
    cur["checkpoint_freq"] = int(value)
    after_digest = digest(cfg, "ppo")
    rec = {"stage_reference": ref, "before": before, "after": {k: cur.get(k) for k in before},
           "hyperparameters_sha256_before": before_digest, "hyperparameters_sha256_after": after_digest}
    if before_digest != after_digest:
        raise RunnerRefusal(f"the cadence override moved hyperparameters_sha256: {rec}")
    return rec


def write_record() -> None:
    if RECORD_PATH is not None:
        RECORD_PATH.write_text(json.dumps(RECORD, indent=1, default=str))


def install_stop_after_pair(callback_cls, steps: int, prefix: str = "stage2") -> None:
    """Patch *callback_cls*._on_step (SB3's CheckpointCallback) to end the process once the pair at *steps* exists."""
    original = callback_cls._on_step

    def _on_step(self):
        result = original(self)
        zip_path = Path(self.save_path) / f"{prefix}_{steps}_steps.zip"
        pkl_path = Path(self.save_path) / f"{prefix}_vecnormalize_{steps}_steps.pkl"
        if getattr(self, "name_prefix", None) == prefix and zip_path.is_file() and pkl_path.is_file():
            RECORD["stopped_after_pair"] = {"steps": steps, "zip": str(zip_path), "vecnormalize": str(pkl_path),
                                            "num_timesteps": int(getattr(self, "num_timesteps", -1)),
                                            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            RECORD["exit"] = STOP_EXIT
            print(f"nb_pilot_runner3: simulated runtime loss after the {steps:,} pair ({zip_path.name}); exiting "
                  f"{STOP_EXIT} with no final save", flush=True)
            write_record()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(STOP_EXIT)
        return result

    callback_cls._on_step = _on_step


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def install_ipython_stand_in() -> bool:
    """A minimal IPython.display (Video, display) when IPython is not installed; True when installed here."""
    try:
        import IPython.display  # noqa: F401
        return False
    except ImportError:
        pass
    ipy = types.ModuleType("IPython")
    disp = types.ModuleType("IPython.display")

    class Video:
        def __init__(self, data=None, **kw):
            self.data, self.kw = data, kw

        def __repr__(self):
            return f"Video({self.data!r})"

    def display(*objs, **kw):
        for o in objs:
            print(f"[nb_pilot_runner3 display] {o!r}", flush=True)

    disp.Video, disp.display = Video, display
    ipy.display = disp
    sys.modules["IPython"], sys.modules["IPython.display"] = ipy, disp
    return True


def run_cell(idx: int, src: str, ns: dict) -> dict:
    t0 = time.time()
    print(f"\n##### nb_pilot_runner3: cell {idx} start {time.strftime('%H:%M:%S', time.gmtime())} UTC", flush=True)
    try:
        exec(compile(src, f"<cell {idx}>", "exec"), ns)
        res = {"cell": idx, "ok": True, "seconds": round(time.time() - t0, 1)}
    except BaseException as exc:  # noqa: BLE001 - a cell's exception is the notebook's result, recorded
        traceback.print_exc()
        res = {"cell": idx, "ok": False, "seconds": round(time.time() - t0, 1), "exception": type(exc).__name__,
               "message": str(exc)[:2000]}
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            res["fatal"] = True
    print(f"##### nb_pilot_runner3: cell {idx} {'ok' if res['ok'] else 'RAISED ' + res['exception']} "
          f"({res['seconds']} s)", flush=True)
    return res


def main(argv=None) -> int:
    global RECORD_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clone", default=".", help="the clone's root (default: the working directory)")
    ap.add_argument("--notebook", default="notebooks/sb3_training.ipynb")
    ap.add_argument("--phase", choices=("launch", "resume"), default="launch")
    ap.add_argument("--cadence-override", type=int, default=None)
    ap.add_argument("--stop-after-pair", type=int, default=None)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args(argv)
    clone = Path(a.clone).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    RECORD_PATH = out / "record.json"
    params = json.loads(Path(a.params).read_text())
    nb_path = clone / a.notebook
    RECORD.clear()
    RECORD.update({"tool": "nb_pilot_runner3", "tool_sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv[1:] if argv is None else argv,
                   "clone": str(clone), "notebook": str(nb_path), "phase": a.phase, "params": params,
                   "python": platform.python_version(), "cells": []})
    try:
        if os.path.exists("/content") or "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
            raise RunnerRefusal("this host looks like Colab (/content or a COLAB_ variable): the setup cell would take "
                                "its Colab branch")
        if a.stop_after_pair is not None and a.cadence_override is not None and a.stop_after_pair % a.cadence_override:
            raise RunnerRefusal(f"--stop-after-pair {a.stop_after_pair} is not a multiple of the cadence "
                                f"{a.cadence_override}: no pair would be written there")
        nb = json.loads(nb_path.read_text())
        RECORD["notebook_sha256"] = "sha256:" + hashlib.sha256(nb_path.read_bytes()).hexdigest()
        try:
            RECORD["clone_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True,
                                                           env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()
            RECORD["clone_status_before"] = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=clone, text=True,
                env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).splitlines()
        except (OSError, subprocess.CalledProcessError) as exc:
            RECORD["clone_head"] = f"unreadable: {exc}"
        sources, roles, edits = edit_cells(nb, params, clone, a.phase)
        RECORD.update(roles=roles, edits=edits)
        (out / "cells_edited.json").write_text(json.dumps({str(k): v for k, v in sorted(sources.items())}, indent=1))
    except RunnerRefusal as exc:
        RECORD.update(refused=str(exc), exit=2)
        write_record()
        print(f"nb_pilot_runner3: refusing: {exc}", flush=True)
        return 2
    if a.plan_only:
        RECORD["exit"] = 0
        write_record()
        print(f"nb_pilot_runner3: plan only; cells {roles}; {len(edits)} edits; wrote {out / 'cells_edited.json'}")
        return 0

    stdout_log = open(out / "runner.stdout.log", "a", encoding="utf-8")
    stderr_log = open(out / "runner.stderr.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, stdout_log), Tee(sys.__stderr__, stderr_log)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(clone)
    RECORD["ipython_stand_in"] = install_ipython_stand_in()
    if a.stop_after_pair is not None:
        from stable_baselines3.common.callbacks import CheckpointCallback

        install_stop_after_pair(CheckpointCallback, a.stop_after_pair)
    RECORD["stop_after_pair_requested"] = a.stop_after_pair
    RECORD["cadence_override_requested"] = a.cadence_override

    def _no_ipython():
        raise RuntimeError("nb_pilot_runner3: get_ipython() is a Colab-only call here")

    ns: dict = {"__name__": "__main__", "get_ipython": _no_ipython}
    expected_guard = guard_message(params["guard_line"])
    run_all = sorted(i for i in sources if i <= roles["chain"])
    status = 0
    stopped_at = None
    for i in run_all:
        res = run_cell(i, sources[i], ns)
        RECORD["cells"].append(res)
        write_record()
        if res["ok"] and a.cadence_override is not None and "override" not in RECORD and "STAGE_CONFIGS" in ns \
                and "MANIFEST" in ns:
            try:
                RECORD["override"] = apply_cadence_override(ns, a.cadence_override)
                print(f"nb_pilot_runner3: cadence override applied in memory: {RECORD['override']}", flush=True)
            except RunnerRefusal as exc:
                RECORD["override_refused"] = str(exc)
                print(f"nb_pilot_runner3: {exc}", flush=True)
                status = 1
                break
        if not res["ok"]:
            stopped_at = i
            break
    guard_ok = (stopped_at == roles["chain"] and RECORD["cells"][-1].get("exception") == "RuntimeError"
                and RECORD["cells"][-1].get("message") == expected_guard)
    RECORD["run_all"] = {"stopped_at": stopped_at, "stopped_by_the_guard": guard_ok}
    if not guard_ok:
        status = 1
        print(f"nb_pilot_runner3: Run all did not stop at the guard (stopped at {stopped_at})", flush=True)
    if a.cadence_override is not None and "override" not in RECORD:
        status = 1
        RECORD.setdefault("override_refused", "STAGE_CONFIGS and MANIFEST never appeared in the namespace")
    if status == 0 and a.phase == "launch":
        res = run_cell(roles["manual"], sources[roles["manual"]], ns)
        RECORD["cells"].append(res)
        write_record()
        if not res["ok"]:
            status = 1
    if status == 0:
        res = run_cell(roles["disconnect"], sources[roles["disconnect"]], ns)
        RECORD["cells"].append(res)
        if not res["ok"]:
            status = 1
    try:
        cur = ns["STAGE_CONFIGS"][ns["MANIFEST"].resolve(STAGE_FOR_OVERRIDE).reference]["curriculum_kwargs"]
        RECORD["locomotion_curriculum_at_end"] = {k: cur.get(k) for k in ("eval_freq", "checkpoint_freq",
                                                                          "eval_episodes", "max_checkpoints",
                                                                          "warm_start_log_std", "timesteps")}
    except Exception as exc:  # noqa: BLE001 - report only
        RECORD["locomotion_curriculum_at_end"] = f"unreadable: {type(exc).__name__}: {exc}"
    RECORD["run_dir"] = str(ns.get("RUN_DIR"))
    RECORD["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    RECORD["exit"] = status
    write_record()
    print(f"nb_pilot_runner3: done, exit {status}; record {RECORD_PATH}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
