"""Tests of nb_pilot_runner3.py (pytest; walking pilot 3 scratch, never merged).

The edits on the branch head's own notebook (--plan-only: nothing runs), refusals, and the executor on synthetic
notebooks that carry the real notebook's banners (Run all stopped by the guard, the manual cell alone, section 9's
cell, the in-memory cadence override and its digest check, the resume phase, the simulated runtime loss). No test
here imports the repository or trains: the smoke stage runs the runner on a clone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import nb_pilot_runner3 as nr  # noqa: E402

TREE = Path(os.environ.get("PILOT3_TREE", str(TOOLS / "tree"))).resolve()
GUARD = ('raise RuntimeError("Walking pilot 3: chain loop skipped on purpose. Now run the manual cell (section 7) by '
         'itself -- unless you are resuming: then stop here.")')
PARAMS = {
    "repo_ref": "claude/pilot-walk-r10-p3",
    "config": {"SPECIES": "tyrannosaurus_rex", "EMBODIMENT": "anatomical", "ALGORITHM": "ppo", "N_ENVS": 4, "SEED": 42,
               "QUICK_TEST": True, "USE_GOOGLE_DRIVE": False, "AUTO_DISCONNECT": False, "BEHAVIOR": "walk",
               "TRUNK_FROM": "", "RETRAIN_FROM": "", "RUN_LABEL": "walking pilot 3 smoke s42",
               "RUN_ID": "pilot3_walk_s42", "K3_PANEL": False, "CRITIC_PHASE": False},
    "manual": ['MANUAL_NODE = "locomotion"',
               'MANUAL_LOAD_PATH = "/content/mesozoic-labs/pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final"',
               'MANUAL_VECNORM_PATH = MANUAL_LOAD_PATH + "_vecnorm.pkl"',
               'MANUAL_LOAD_MODE = "initialize_next_stage"',
               "MANUAL_TIMESTEPS = None"],
    "guard_line": GUARD,
    "resume_stage": "locomotion",
}


def run_runner(*args, cwd=None, timeout=300):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-I", "-B", str(TOOLS / "nb_pilot_runner3.py"), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- the branch head's notebook (plan only)

needs_tree = pytest.mark.skipif(not (TREE / "notebooks" / "sb3_training.ipynb").is_file(), reason=f"no tree at {TREE}")


@needs_tree
@pytest.mark.parametrize("phase", ["launch", "resume"])
def test_plan_only_on_the_head_notebook(tmp_path, phase):
    params = tmp_path / "params.json"
    params.write_text(json.dumps(PARAMS))
    r = run_runner("--params", str(params), "--out", str(tmp_path / "out"), "--clone", str(TREE), "--phase", phase,
                   "--plan-only")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((tmp_path / "out" / "record.json").read_text())
    assert rec["roles"] == {"setup": 3, "config": 6, "resume": 18, "chain": 20, "manual": 22, "disconnect": 32}
    cells = {int(k): v for k, v in json.loads((tmp_path / "out" / "cells_edited.json").read_text()).items()}
    assert cells[20].splitlines()[0] == GUARD
    assert cells[20].splitlines()[1] == "# ===== BEHAVIOR CHAIN LOOP ====="
    manual = cells[22].splitlines()
    assert manual[1:6] == ['MANUAL_NODE = "locomotion"',
                           f'MANUAL_LOAD_PATH = "{TREE}/pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final"',
                           'MANUAL_VECNORM_PATH = MANUAL_LOAD_PATH + "_vecnorm.pkl"',
                           'MANUAL_LOAD_MODE = "initialize_next_stage"', "MANUAL_TIMESTEPS = None"]
    for name, value in PARAMS["config"].items():
        assert f"\n{name} = {value!r}  # nb_pilot_runner3 (Colab steps)" in cells[6]
    assert "REPO_REF = 'claude/pilot-walk-r10-p3'" in cells[3]
    resume_line = [ln for ln in cells[18].splitlines() if ln.startswith("RESUME_STAGE =")][0]
    assert resume_line.startswith("RESUME_STAGE = 'locomotion'" if phase == "resume" else "RESUME_STAGE = None")
    assert not (tmp_path / "out" / "runner.stdout.log").exists()  # nothing ran
    head_cells = json.loads((TREE / "notebooks" / "sb3_training.ipynb").read_text())["cells"]
    unedited = [i for i in cells if i not in (3, 6, 18, 20, 22)]
    assert all(cells[i] == nr.cell_source(head_cells[i]) for i in unedited)


def test_refusals_before_any_cell(tmp_path):
    nb = synthetic_notebook()
    bad = dict(PARAMS, config={"NOT_A_SETTING": 1})
    with pytest.raises(nr.RunnerRefusal, match="NOT_A_SETTING is assigned on 0 lines"):
        nr.edit_cells(nb, bad, tmp_path, "launch")
    bad = dict(PARAMS, config={}, manual=PARAMS["manual"][:4])
    with pytest.raises(nr.RunnerRefusal, match="not the five"):
        nr.edit_cells(nb, bad, tmp_path, "launch")
    bad = dict(PARAMS, config={}, guard_line="print('no')")
    with pytest.raises(nr.RunnerRefusal, match="raises RuntimeError"):
        nr.edit_cells(nb, bad, tmp_path, "launch")
    nb2 = synthetic_notebook()
    nb2["cells"].append(code_cell("# ===== BEHAVIOR CHAIN LOOP =====\nprint(2)"))
    with pytest.raises(nr.RunnerRefusal, match="chain cell"):
        nr.find_cells(nb2)


def test_guard_message():
    assert nr.guard_message(GUARD).startswith("Walking pilot 3: chain loop skipped on purpose.")


# ---------------------------------------------------------------- the executor on synthetic notebooks


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": src.splitlines(keepends=True)}


def synthetic_notebook(config_extra: str = "", manual_extra: str = "") -> dict:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# synthetic"]},
        code_cell(textwrap.dedent('''\
            # Install dependencies (Colab auto-detected; no-op locally)
            import os, sys
            REPO_REF = "main"
            sys.path.insert(0, os.getcwd())
            def disconnect_runtime(reason, **kw):
                print("DISCONNECT", reason, kw)
            print("setup ran, REPO_REF", REPO_REF)''')),
        code_cell(textwrap.dedent('''\
            # ===== SPECIES SELECTION =====
            SPECIES = "velociraptor_mongoliensis"  # @param
            SEED = 1
            RUN_ID = ""
            ''') + config_extra),
        code_cell(textwrap.dedent('''\
            class _Entry:
                reference = 2
            class _Manifest:
                def resolve(self, name):
                    assert name == "locomotion"
                    return _Entry()
            MANIFEST = _Manifest()
            STAGE_CONFIGS = {2: {"ppo_kwargs": {"n_steps": 4096},
                                 "curriculum_kwargs": {"eval_freq": 131072, "checkpoint_freq": 500000,
                                                       "eval_episodes": 10, "max_checkpoints": 0,
                                                       "warm_start_log_std": -1.5}}}''')),
        code_cell(textwrap.dedent('''\
            # ===== RESUME AN INTERRUPTED STAGE — opt-in escape hatch =====
            RESUME_STAGE = None  # None = skip
            if RESUME_STAGE is not None:
                print("RESUMING", RESUME_STAGE, STAGE_CONFIGS[2]["curriculum_kwargs"]["checkpoint_freq"])''')),
        code_cell("# ===== BEHAVIOR CHAIN LOOP =====\nprint('THE CHAIN LOOP TRAINED')\n"),
        code_cell(textwrap.dedent('''\
            # ===== MANUAL SINGLE NODE — debugging escape hatch =====
            MANUAL_NODE = None
            MANUAL_LOAD_PATH = None
            MANUAL_VECNORM_PATH = None
            MANUAL_LOAD_MODE = "initialize_next_stage"
            MANUAL_TIMESTEPS = None
            print("MANUAL", MANUAL_NODE, MANUAL_LOAD_PATH, MANUAL_VECNORM_PATH, SPECIES, SEED, RUN_ID,
                  STAGE_CONFIGS[2]["curriculum_kwargs"])
            ''') + manual_extra),
        {"cell_type": "markdown", "metadata": {}, "source": ["## 9"]},
        code_cell('disconnect_runtime("finished", in_colab=False, auto=False)\n'),
    ]
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


FAKE_CONFIG = '''\
import hashlib, json
SHAPING = ("warmup_", "ramp_", "warm_start_")
def hyperparameters_sha256(cfg, algorithm):
    view = {"a": algorithm, "h": cfg.get("ppo_kwargs", {}),
            "s": {k: v for k, v in cfg.get("curriculum_kwargs", {}).items() if k.startswith(SHAPING)}}
    return "sha256:" + hashlib.sha256(json.dumps(view, sort_keys=True).encode()).hexdigest()
'''


def synthetic_clone(tmp_path: Path, nb: dict, digest_covers_cadence: bool = False) -> Path:
    clone = tmp_path / "clone"
    (clone / "notebooks").mkdir(parents=True)
    (clone / "notebooks" / "sb3_training.ipynb").write_text(json.dumps(nb))
    pkg = clone / "environments" / "shared"
    pkg.mkdir(parents=True)
    (clone / "environments" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    cfg = FAKE_CONFIG.replace('("warmup_", "ramp_", "warm_start_")', '("warmup_", "ramp_", "warm_start_", "eval_")') \
        if digest_covers_cadence else FAKE_CONFIG
    (pkg / "config.py").write_text(cfg)
    return clone


def params_file(tmp_path: Path, **over) -> Path:
    p = tmp_path / "params.json"
    p.write_text(json.dumps(dict(PARAMS, config={"SPECIES": "tyrannosaurus_rex", "SEED": 42, "RUN_ID": "pilot3_walk_s42"},
                                 **over)))
    return p


def test_launch_stops_at_the_guard_then_runs_the_manual_cell_with_the_override(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone),
                   "--cadence-override", "32768")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((out / "record.json").read_text())
    assert rec["run_all"] == {"stopped_at": 5, "stopped_by_the_guard": True}
    assert [c["cell"] for c in rec["cells"]] == [1, 2, 3, 4, 5, 6, 8]
    assert [c["ok"] for c in rec["cells"]] == [True, True, True, True, False, True, True]
    ov = rec["override"]
    assert ov["before"]["eval_freq"] == 131072 and ov["after"] == {"eval_freq": 32768, "checkpoint_freq": 32768,
                                                                  "eval_episodes": 10, "max_checkpoints": 0}
    assert ov["hyperparameters_sha256_before"] == ov["hyperparameters_sha256_after"]
    log = (out / "runner.stdout.log").read_text()
    assert "THE CHAIN LOOP TRAINED" not in log
    assert f"MANUAL locomotion {clone}/pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final" in log
    assert "'eval_freq': 32768, 'checkpoint_freq': 32768" in log and "tyrannosaurus_rex 42 pilot3_walk_s42" in log
    assert "DISCONNECT finished" in log
    assert "RuntimeError: Walking pilot 3: chain loop skipped on purpose." in (out / "runner.stderr.log").read_text()
    assert rec["locomotion_curriculum_at_end"]["checkpoint_freq"] == 32768 and rec["exit"] == 0


def test_without_the_override_the_cadence_stays_the_config_s(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone))
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((out / "record.json").read_text())
    assert "override" not in rec and rec["locomotion_curriculum_at_end"]["checkpoint_freq"] == 500000


def test_an_override_that_moves_the_digest_is_refused(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook(), digest_covers_cadence=True)
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone),
                   "--cadence-override", "32768")
    rec = json.loads((out / "record.json").read_text())
    assert r.returncode == 1 and "moved hyperparameters_sha256" in rec["override_refused"]
    assert "MANUAL locomotion" not in (out / "runner.stdout.log").read_text()


def test_an_unexpected_error_stops_run_all_and_fails(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook(config_extra="raise ValueError('a broken config cell')\n"))
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone))
    rec = json.loads((out / "record.json").read_text())
    assert r.returncode == 1 and rec["run_all"] == {"stopped_at": 2, "stopped_by_the_guard": False}
    assert rec["cells"][-1]["exception"] == "ValueError" and len(rec["cells"]) == 2


def test_a_failing_manual_cell_fails_the_run(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook(manual_extra="raise RuntimeError('training failed')\n"))
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone))
    rec = json.loads((out / "record.json").read_text())
    assert r.returncode == 1 and rec["cells"][-1]["cell"] == 6 and rec["cells"][-1]["message"] == "training failed"


def test_the_resume_phase_runs_the_resume_cell_inside_run_all_and_not_the_manual_cell(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone), "--phase",
                   "resume", "--cadence-override", "32768")
    assert r.returncode == 0, r.stdout + r.stderr
    log = (out / "runner.stdout.log").read_text()
    assert "RESUMING locomotion 32768" in log and "MANUAL " not in log and "DISCONNECT finished" in log
    rec = json.loads((out / "record.json").read_text())
    assert [c["cell"] for c in rec["cells"]] == [1, 2, 3, 4, 5, 8]


def test_a_colab_host_is_refused(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    env = dict(os.environ, COLAB_RELEASE_TAG="x")
    r = subprocess.run([sys.executable, "-I", "-B", str(TOOLS / "nb_pilot_runner3.py"), "--params",
                        str(params_file(tmp_path)), "--out", str(tmp_path / "out"), "--clone", str(clone)],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "looks like Colab" in r.stdout


def test_stop_after_pair_must_land_on_the_cadence(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(tmp_path / "out"), "--clone", str(clone),
                   "--cadence-override", "32768", "--stop-after-pair", "50000")
    assert r.returncode == 2 and "not a multiple of the cadence" in r.stdout


def test_the_simulated_runtime_loss_exits_75_after_the_pair_with_a_record(tmp_path):
    script = tmp_path / "loss.py"
    script.write_text(textwrap.dedent(f'''\
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(TOOLS)!r})
        import nb_pilot_runner3 as nr
        class FakeCheckpointCallback:
            def __init__(self, path):
                self.save_path, self.name_prefix, self.num_timesteps = path, "stage2", 0
            def _on_step(self):
                self.num_timesteps += 16384
                for name in (f"stage2_{{self.num_timesteps}}_steps.zip",
                             f"stage2_vecnormalize_{{self.num_timesteps}}_steps.pkl"):
                    (Path(self.save_path) / name).write_text("x")
                return True
        nr.RECORD_PATH = Path({str(tmp_path / "record.json")!r})
        nr.install_stop_after_pair(FakeCheckpointCallback, 32768)
        cb = FakeCheckpointCallback({str(tmp_path)!r})
        print("first", cb._on_step(), flush=True)
        cb._on_step()
        print("NOT REACHED", flush=True)
        '''))
    r = subprocess.run([sys.executable, "-I", "-B", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == nr.STOP_EXIT == 75
    assert "first True" in r.stdout and "NOT REACHED" not in r.stdout
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["stopped_after_pair"]["steps"] == 32768 and rec["stopped_after_pair"]["num_timesteps"] == 32768
