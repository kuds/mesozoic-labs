"""Tests of nb_pilot_runner4.py (pytest; walking pilot 4 scratch, never merged), adapted from pilot 3's
test_nb_pilot_runner3.py.

The edits on the branch head's own notebook (--plan-only: nothing runs), the refusals (the parent pair, the explicit
sidecar, the cadence, the head, a Colab host), the executor on synthetic notebooks that carry the real notebook's
banners and print the entry lines the setup checks measured (Run all stopped by the guard, the manual cell alone,
section 9's cell, the in-memory cadence override 65,536 / 250,000 -> 32,768 and its digest checks, the entry-line
check, --stop-at-learn, the resume phase, the simulated runtime loss). Unit tests only: no test imports the
repository or runs a notebook's training; the parent pair's sha256s are read from the verified run copies (read only),
and the tests that need them are skipped when the copies are absent.
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
import nb_pilot_runner4 as nr  # noqa: E402

TREE = Path(os.environ.get("PILOT4_TREE", str(TOOLS / "tree"))).resolve()
HEAD = "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8"
RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/"
            "runs/final")
DRIVE = "/content/drive/MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo"
MODELS = "pilot3_walk_s{seed}/03_locomotion/models"
GUARD = ('raise RuntimeError("Walking pilot 4: chain loop skipped on purpose. Now run the manual cell (section 7) by '
         'itself -- unless you are resuming: then stop here.")')


def manual_lines(seed=42, vecnorm=None, mode="initialize_next_stage", timesteps="None", stem="stage2_8000000_steps"):
    base = f"{DRIVE}/{MODELS.format(seed=seed)}"
    vec = vecnorm if vecnorm is not None else f'"{base}/stage2_vecnormalize_8000000_steps.pkl"'
    return ['MANUAL_NODE = "locomotion"', f'MANUAL_LOAD_PATH = "{base}/{stem}"', f"MANUAL_VECNORM_PATH = {vec}",
            f'MANUAL_LOAD_MODE = "{mode}"', f"MANUAL_TIMESTEPS = {timesteps}"]


PARAMS = {
    "repo_ref": "claude/pilot-walk-r10-p4",
    "config": {"SPECIES": "tyrannosaurus_rex", "EMBODIMENT": "anatomical", "ALGORITHM": "ppo", "N_ENVS": 4, "SEED": 42,
               "QUICK_TEST": True, "USE_GOOGLE_DRIVE": False, "AUTO_DISCONNECT": False, "BEHAVIOR": "walk",
               "TRUNK_FROM": "", "RETRAIN_FROM": "", "RUN_LABEL": "walking pilot 4 smoke s42",
               "RUN_ID": "pilot4_walk_s42", "K3_PANEL": False, "CRITIC_PHASE": False},
    "manual": manual_lines(),
    "path_map": [[DRIVE, str(RUNS)]],
    "guard_line": GUARD,
    "resume_stage": "locomotion",
}
needs_runs = pytest.mark.skipif(not (RUNS / MODELS.format(seed=42) / "stage2_8000000_steps.zip").is_file(),
                                reason=f"no verified run copies at {RUNS}")
needs_tree = pytest.mark.skipif(not (TREE / "notebooks" / "sb3_training.ipynb").is_file(), reason=f"no tree at {TREE}")


def run_runner(*args, cwd=None, timeout=300, env_extra=None):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-I", "-B", str(TOOLS / "nb_pilot_runner4.py"), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- the branch head's notebook (plan only)


@needs_tree
@needs_runs
@pytest.mark.parametrize("phase", ["launch", "resume"])
def test_plan_only_on_the_head_notebook(tmp_path, phase):
    params = tmp_path / "params.json"
    params.write_text(json.dumps(dict(PARAMS, expect_head=HEAD)))
    r = run_runner("--params", str(params), "--out", str(tmp_path / "out"), "--clone", str(TREE), "--phase", phase,
                   "--plan-only")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((tmp_path / "out" / "record.json").read_text())
    assert rec["roles"] == {"setup": 3, "config": 6, "resume": 18, "chain": 20, "manual": 22, "disconnect": 32}
    assert rec["clone_head"] == HEAD and rec["environments_resolves_to"].startswith(str(TREE) + os.sep)
    cells = {int(k): v for k, v in json.loads((tmp_path / "out" / "cells_edited.json").read_text()).items()}
    assert cells[20].splitlines()[0] == GUARD
    assert cells[20].splitlines()[1] == "# ===== BEHAVIOR CHAIN LOOP ====="
    base = f"{RUNS}/{MODELS.format(seed=42)}"
    manual = cells[22].splitlines()
    assert manual[1:6] == ['MANUAL_NODE = "locomotion"', f'MANUAL_LOAD_PATH = "{base}/stage2_8000000_steps"',
                           f'MANUAL_VECNORM_PATH = "{base}/stage2_vecnormalize_8000000_steps.pkl"',
                           'MANUAL_LOAD_MODE = "initialize_next_stage"', "MANUAL_TIMESTEPS = None"]
    for name, value in PARAMS["config"].items():
        assert f"\n{name} = {value!r}  # nb_pilot_runner4 (Colab steps)" in cells[6]
    assert "REPO_REF = 'claude/pilot-walk-r10-p4'" in cells[3]
    resume_line = [ln for ln in cells[18].splitlines() if ln.startswith("RESUME_STAGE =")][0]
    assert resume_line.startswith("RESUME_STAGE = 'locomotion'" if phase == "resume" else "RESUME_STAGE = None")
    if phase == "launch":
        assert rec["parent_pair"]["zip_sha256"] == nr.PARENTS[42][0]
        assert rec["parent_pair"]["vecnorm_sha256"] == nr.PARENTS[42][1]
    assert not (tmp_path / "out" / "runner.stdout.log").exists()  # nothing ran
    head_cells = json.loads((TREE / "notebooks" / "sb3_training.ipynb").read_text())["cells"]
    unedited = [i for i in cells if i not in (3, 6, 18, 20, 22)]
    assert all(cells[i] == nr.cell_source(head_cells[i]) for i in unedited)


@needs_tree
def test_another_head_is_refused(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps(dict(PARAMS, expect_head="31d02e42d148511e50f9cba9a598d9a9d36c23af")))
    r = run_runner("--params", str(params), "--out", str(tmp_path / "out"), "--clone", str(TREE), "--plan-only")
    assert r.returncode == 2 and "not the expected head" in r.stdout


# ---------------------------------------------------------------- refusals before any cell


def test_cell_edits_refusals(tmp_path):
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
    bad = dict(PARAMS, config={}, path_map=[["only one"]])
    with pytest.raises(nr.RunnerRefusal, match="path_map entry"):
        nr.edit_cells(nb, bad, tmp_path, "launch")
    nb2 = synthetic_notebook()
    nb2["cells"].append(code_cell("# ===== BEHAVIOR CHAIN LOOP =====\nprint(2)"))
    with pytest.raises(nr.RunnerRefusal, match="chain cell"):
        nr.find_cells(nb2)


def test_path_map_and_manual_values():
    maps = nr.path_maps(PARAMS, Path("/clone"))
    assert maps[-1] == (nr.CONTENT_ROOT, "/clone")
    line = nr.map_line(f'MANUAL_LOAD_PATH = "{DRIVE}/x/stage2_8000000_steps"', maps)
    assert line == f'MANUAL_LOAD_PATH = "{RUNS}/x/stage2_8000000_steps"'
    edits = [("manual", "MANUAL_NODE", 'MANUAL_NODE = "locomotion"  # a note'),
             ("manual", "MANUAL_VECNORM_PATH", 'MANUAL_VECNORM_PATH = MANUAL_LOAD_PATH + "_vecnorm.pkl"'),
             ("manual", "MANUAL_TIMESTEPS", "MANUAL_TIMESTEPS = None"), ("config", "SEED", 42),
             ("manual", "MANUAL_LOAD_PATH", 'MANUAL_LOAD_PATH = "/a#b/stage2_8000000_steps"  # c')]
    v = nr.manual_values(edits)
    assert v["MANUAL_NODE"] == "locomotion" and v["MANUAL_TIMESTEPS"] is None
    assert v["MANUAL_VECNORM_PATH"] == ("<expression>", "MANUAL_LOAD_PATH + '_vecnorm.pkl'")
    assert v["MANUAL_LOAD_PATH"] == "/a#b/stage2_8000000_steps" and "SEED" not in v


def values(seed=42, **over):
    base = f"{RUNS}/{MODELS.format(seed=seed)}"
    v = {"MANUAL_NODE": "locomotion", "MANUAL_LOAD_PATH": f"{base}/stage2_8000000_steps",
         "MANUAL_VECNORM_PATH": f"{base}/stage2_vecnormalize_8000000_steps.pkl",
         "MANUAL_LOAD_MODE": "initialize_next_stage", "MANUAL_TIMESTEPS": None}
    v.update(over)
    return v


@needs_runs
def test_the_parent_pair_checks():
    problems, info = nr.parent_pair_problems(values(), 42)
    assert problems == [] and info["zip_sha256"] == nr.PARENTS[42][0]
    # the default sidecar expression is not explicit
    problems, _ = nr.parent_pair_problems(values(MANUAL_VECNORM_PATH=("<expression>", "MANUAL_LOAD_PATH + 'x'")), 42)
    assert any("not an explicit string literal" in p for p in problems)
    # another seed's 8M pair under SEED 42
    problems, _ = nr.parent_pair_problems(values(seed=52), 42)
    assert any("is not s42's 8M pair" in p for p in problems)
    # another checkpoint of the same run, a mismatched sidecar, a resume edge, a step budget, an unscored seed
    problems, _ = nr.parent_pair_problems(values(MANUAL_LOAD_PATH=f"{RUNS}/{MODELS.format(seed=42)}/"
                                                 "stage2_7500000_steps"), 42)
    assert any("does not name stage2_8000000_steps" in p for p in problems)
    problems, _ = nr.parent_pair_problems(values(MANUAL_VECNORM_PATH=f"{RUNS}/{MODELS.format(seed=42)}/"
                                                 "stage2_vecnormalize_7500000_steps.pkl"), 42)
    assert any("is not" in p and "pair's sidecar" in p for p in problems)
    assert any("resume" in p for p in nr.parent_pair_problems(values(MANUAL_LOAD_MODE="resume_same_stage"), 42)[0])
    assert any("MANUAL_TIMESTEPS" in p for p in nr.parent_pair_problems(values(MANUAL_TIMESTEPS=65536), 42)[0])
    assert any("scored seeds" in p for p in nr.parent_pair_problems(values(), 43)[0])


def test_a_missing_parent_pair_is_refused(tmp_path):
    v = values()
    v["MANUAL_LOAD_PATH"] = str(tmp_path / "stage2_8000000_steps")
    v["MANUAL_VECNORM_PATH"] = str(tmp_path / "stage2_vecnormalize_8000000_steps.pkl")
    problems, _ = nr.parent_pair_problems(v, 42)
    assert len([p for p in problems if "not on disk" in p]) == 2


def test_guard_message_and_constants():
    assert nr.guard_message(GUARD).startswith("Walking pilot 4: chain loop skipped on purpose.")
    assert nr.CADENCE_OVERRIDE == 32768 and nr.FIRST_EVAL_DEFAULT == 65536
    assert nr.CADENCE_BEFORE == {"eval_freq": 65536, "checkpoint_freq": 250000, "eval_episodes": 10,
                                 "max_checkpoints": 0}
    assert nr.PILOT4_PPO_SHA256.startswith("sha256:fbc94bef")
    assert nr.TIMESTEPS_LINE == {False: "Timesteps: 2,000,000", True: "Timesteps: 50,000"}


def test_the_entry_line_check_reads_required_forbidden_and_the_first_eval():
    exp = nr.expected_entry_lines(PARAMS, "launch", "/v/stage2_vecnormalize_8000000_steps.pkl")
    good = "\n".join(["Running locally; using this checkout. REPO_REF selects only the Colab checkout.",
                      "Quick test mode: True", "Behavior recipe: walk", "In-run K3 panel: off", "Critic phase: off",
                      "Run directory: /c/logs/x/ppo_quick_test/pilot4_walk_s42 (new run: RUN_ID names no existing run)",
                      "Node locomotion (stage 2): Locomotion (ppo)", "Timesteps: 50,000",
                      "VecNormalize loaded from: /v/stage2_vecnormalize_8000000_steps.pkl",
                      "Eval num_timesteps=32768, episode_reward=1.0 +/- 0.0"])
    assert nr.check_entry_lines(good, exp, learn_ran=True, first_eval=32768)["ok"]
    assert not nr.check_entry_lines(good, exp, learn_ran=True, first_eval=65536)["ok"]
    r = nr.check_entry_lines(good + "\nWarm start: policy log_std set to -2", exp, learn_ran=True, first_eval=32768)
    assert not r["ok"] and r["forbidden_present"]
    r = nr.check_entry_lines(good + "\nWarm start: something else", exp, learn_ran=True, first_eval=32768)
    assert r["ok"] and r["other_warm_start_lines"] == ["Warm start: something else"]
    r = nr.check_entry_lines(good.replace("Critic phase: off", "Critic phase: on"), exp, learn_ran=False,
                             first_eval=32768)
    assert not r["ok"] and r["missing"] == ["critic phase"] and r["first_eval_ok"] is None
    wrong_vec = good.replace("/v/stage2_vecnormalize_8000000_steps.pkl", "/v/stage2_final_vecnorm.pkl")
    assert nr.check_entry_lines(wrong_vec, exp, learn_ran=False, first_eval=32768)["missing"] == ["vecnormalize"]
    full = nr.expected_entry_lines(dict(PARAMS, config=dict(PARAMS["config"], QUICK_TEST=False)), "launch", None)
    assert ("timesteps", r"^Timesteps:\ 2,000,000$") in full["required"]


# ---------------------------------------------------------------- the executor on synthetic notebooks


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": src.splitlines(keepends=True)}


CADENCE = '{"eval_freq": 65536, "checkpoint_freq": 250000, "eval_episodes": 10, "max_checkpoints": 0}'


def synthetic_notebook(config_extra: str = "", manual_extra: str = "", cadence: str = CADENCE,
                       manual_body: str = "") -> dict:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# synthetic"]},
        code_cell(textwrap.dedent('''\
            # Install dependencies (Colab auto-detected; no-op locally)
            import os, sys
            REPO_REF = "main"
            def disconnect_runtime(reason, **kw):
                print("DISCONNECT", reason, kw)
            print("Running locally; using this checkout. REPO_REF selects only the Colab checkout.")''')),
        code_cell(textwrap.dedent('''\
            # ===== SPECIES SELECTION =====
            SPECIES = "velociraptor_mongoliensis"  # @param
            SEED = 1
            QUICK_TEST = False
            RUN_ID = ""
            print(f"Quick test mode: {QUICK_TEST}")
            print("Behavior recipe: walk")
            print("In-run K3 panel: off")
            print("Critic phase: off")
            ''') + config_extra),
        code_cell(textwrap.dedent(f'''\
            class _Entry:
                reference = 2
            class _Manifest:
                def resolve(self, name):
                    assert name == "locomotion"
                    return _Entry()
            MANIFEST = _Manifest()
            RUN_DIR = "/c/logs/t/ppo_quick_test/" + RUN_ID
            print(f"Run directory: {{RUN_DIR}} (new run: RUN_ID names no existing run)")
            STAGE_CONFIGS = {{2: {{"ppo_kwargs": {{"n_steps": 4096}}, "curriculum_kwargs": {cadence}}}}}''')),
        code_cell(textwrap.dedent('''\
            # ===== RESUME AN INTERRUPTED STAGE — opt-in escape hatch =====
            RESUME_STAGE = None  # None = skip
            if RESUME_STAGE is not None:
                print("Resuming 'locomotion' (stage 2) from /c/models")
                print("RESUMING", RESUME_STAGE, STAGE_CONFIGS[2]["curriculum_kwargs"]["checkpoint_freq"])''')),
        code_cell("# ===== BEHAVIOR CHAIN LOOP =====\nprint('THE CHAIN LOOP TRAINED')\n"),
        code_cell(textwrap.dedent('''\
            # ===== MANUAL SINGLE NODE — debugging escape hatch =====
            MANUAL_NODE = None
            MANUAL_LOAD_PATH = None
            MANUAL_VECNORM_PATH = None
            MANUAL_LOAD_MODE = "initialize_next_stage"
            MANUAL_TIMESTEPS = None
            print("Node locomotion (stage 2): Locomotion (ppo)")
            print("Timesteps: 50,000")
            print(f"VecNormalize loaded from: {MANUAL_VECNORM_PATH}")
            ''') + (manual_body or textwrap.dedent('''\
            print("MANUAL", MANUAL_NODE, MANUAL_LOAD_PATH, SPECIES, SEED, RUN_ID, STAGE_CONFIGS[2]["curriculum_kwargs"])
            print(f"Eval num_timesteps={STAGE_CONFIGS[2]['curriculum_kwargs']['eval_freq']}, episode_reward=1.0 +/- 0.0")
            ''')) + manual_extra),
        {"cell_type": "markdown", "metadata": {}, "source": ["## 9"]},
        code_cell('disconnect_runtime("finished", in_colab=False, auto=False)\n'),
    ]
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


FAKE_DIGEST = "sha256:" + "ab" * 32
FAKE_CONFIG = f'''\
import hashlib, json
SHAPING = ("warmup_", "ramp_", "warm_start_")
def hyperparameters_sha256(cfg, algorithm):
    view = {{"s": {{k: v for k, v in cfg.get("curriculum_kwargs", {{}}).items() if k.startswith(SHAPING)}}}}
    if not view["s"]:
        return "{FAKE_DIGEST}"
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
    d = dict(PARAMS, config={"SPECIES": "tyrannosaurus_rex", "SEED": 42, "QUICK_TEST": True,
                             "RUN_ID": "pilot4_walk_s42"}, expect_hyperparameters_sha256=FAKE_DIGEST)
    d.update(over)
    p.write_text(json.dumps(d))
    return p


@needs_runs
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
    assert ov["before"] == nr.CADENCE_BEFORE and ov["after"] == {"eval_freq": 32768, "checkpoint_freq": 32768,
                                                                 "eval_episodes": 10, "max_checkpoints": 0}
    assert ov["hyperparameters_sha256_before"] == ov["hyperparameters_sha256_after"] == FAKE_DIGEST
    log = (out / "runner.stdout.log").read_text()
    assert "THE CHAIN LOOP TRAINED" not in log
    base = f"{RUNS}/{MODELS.format(seed=42)}"
    assert f"MANUAL locomotion {base}/stage2_8000000_steps tyrannosaurus_rex 42 pilot4_walk_s42" in log
    assert f"VecNormalize loaded from: {base}/stage2_vecnormalize_8000000_steps.pkl" in log
    assert "DISCONNECT finished" in log
    assert "RuntimeError: Walking pilot 4: chain loop skipped on purpose." in (out / "runner.stderr.log").read_text()
    assert rec["entry_lines"]["ok"] and rec["entry_lines"]["first_eval_line"].startswith("Eval num_timesteps=32768,")
    assert rec["locomotion_curriculum_at_end"]["checkpoint_freq"] == 32768 and rec["exit"] == 0
    assert rec["environments_resolves_to"].startswith(str(clone) + os.sep)


@needs_runs
def test_without_the_override_the_first_eval_must_be_65536(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone))
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((out / "record.json").read_text())
    assert "override" not in rec and rec["locomotion_curriculum_at_end"]["checkpoint_freq"] == 250000
    assert rec["entry_lines"]["first_eval_line"].startswith("Eval num_timesteps=65536,")


@needs_runs
def test_a_missing_or_forbidden_entry_line_fails_the_run(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook(manual_extra='print("Warm start: policy log_std set to -2")\n'))
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone),
                   "--cadence-override", "32768")
    rec = json.loads((out / "record.json").read_text())
    assert r.returncode == 1 and rec["entry_lines"]["forbidden_present"] == ["Warm start: policy log_std set to -2"]
    nb = synthetic_notebook()
    nb["cells"][2]["source"] = [ln.replace("Critic phase: off", "Critic phase: on") for ln in nb["cells"][2]["source"]]
    clone2 = synthetic_clone(tmp_path / "b", nb)
    out2 = tmp_path / "out2"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out2), "--clone", str(clone2))
    rec = json.loads((out2 / "record.json").read_text())
    assert r.returncode == 1 and rec["entry_lines"]["missing"] == ["critic phase"]


@needs_runs
def test_an_override_that_moves_the_digest_or_meets_another_cadence_or_digest_is_refused(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook(), digest_covers_cadence=True)
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone),
                   "--cadence-override", "32768")
    rec = json.loads((out / "record.json").read_text())
    assert r.returncode == 1 and "moved hyperparameters_sha256" in rec["override_refused"]
    assert "MANUAL locomotion" not in (out / "runner.stdout.log").read_text()
    pilot3_cadence = '{"eval_freq": 131072, "checkpoint_freq": 500000, "eval_episodes": 10, "max_checkpoints": 0}'
    clone2 = synthetic_clone(tmp_path / "b", synthetic_notebook(cadence=pilot3_cadence))
    out2 = tmp_path / "out2"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out2), "--clone", str(clone2),
                   "--cadence-override", "32768")
    rec = json.loads((out2 / "record.json").read_text())
    assert r.returncode == 1 and "not the committed" in rec["override_refused"]
    out3 = tmp_path / "out3"
    (tmp_path / "c").mkdir()
    p = params_file(tmp_path / "c", expect_hyperparameters_sha256=nr.PILOT4_PPO_SHA256)
    clone3 = synthetic_clone(tmp_path / "d", synthetic_notebook())
    r = run_runner("--params", str(p), "--out", str(out3), "--clone", str(clone3), "--cadence-override", "32768")
    rec = json.loads((out3 / "record.json").read_text())
    assert r.returncode == 1 and "not sha256:fbc94bef" in rec["override_refused"]


def test_another_override_value_is_refused_before_any_cell(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(tmp_path / "out"), "--clone", str(clone),
                   "--cadence-override", "65536")
    assert r.returncode == 2 and "32,768 only" in r.stdout


@pytest.mark.parametrize("phase", ["launch", "resume"])
@pytest.mark.parametrize("quick", [False, None, "true", 1])
def test_a_run_that_is_not_the_quick_test_smoke_is_refused_before_any_cell(tmp_path, phase, quick):
    """Fix stage (review finding: the runner could reach a full 2,000,000-step training run): without --stop-at-learn
    or --plan-only, only a QUICK_TEST that is the JSON literal true runs; anything else is refused, no cell runs."""
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    cfg = {"SPECIES": "tyrannosaurus_rex", "SEED": 42, "RUN_ID": "pilot4_walk_s42"}
    if quick is not None:
        cfg["QUICK_TEST"] = quick
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path, config=cfg)), "--out", str(out), "--clone", str(clone),
                   "--phase", phase)
    assert r.returncode == 2 and f"config QUICK_TEST is {quick!r}, not true" in r.stdout, r.stdout + r.stderr
    assert not (out / "runner.stdout.log").exists() and not (out / "cells_edited.json").exists()
    rec = json.loads((out / "record.json").read_text())
    assert rec["exit"] == 2 and rec["cells"] == [] and "QUICK_TEST" in rec["refused"]


@needs_runs
def test_the_quick_test_rule_leaves_stop_at_learn_and_plan_only_open(tmp_path):
    """A non-QUICK_TEST launch may still be checked to learn() (it trains nothing) or planned (nothing runs)."""
    body = textwrap.dedent('''\
        from stable_baselines3 import PPO
        PPO.learn(None, total_timesteps=2000000, reset_num_timesteps=True)
        ''')
    clone = synthetic_clone(tmp_path, synthetic_notebook(manual_body=body))
    cfg = {"SPECIES": "tyrannosaurus_rex", "SEED": 42, "QUICK_TEST": False, "RUN_ID": "pilot4_walk_s42"}
    r = run_runner("--params", str(params_file(tmp_path, config=cfg)), "--out", str(tmp_path / "o1"), "--clone",
                   str(clone), "--plan-only")
    assert r.returncode == 0, r.stdout + r.stderr
    r = run_runner("--params", str(params_file(tmp_path, config=cfg)), "--out", str(tmp_path / "o2"), "--clone",
                   str(clone), "--stop-at-learn")
    rec = json.loads((tmp_path / "o2" / "record.json").read_text())
    assert rec["manual_cell_stopped_at_learn"] and rec["guard_hits"] == [] and "refused" not in rec, r.stdout


@needs_runs
def test_an_implicit_sidecar_or_another_seeds_pair_is_refused_before_any_cell(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    p = params_file(tmp_path, manual=manual_lines(vecnorm='MANUAL_LOAD_PATH + "_vecnorm.pkl"'))
    r = run_runner("--params", str(p), "--out", str(tmp_path / "o1"), "--clone", str(clone))
    assert r.returncode == 2 and "explicit string literal" in r.stdout
    assert not (tmp_path / "o1" / "runner.stdout.log").exists()
    p = params_file(tmp_path, manual=manual_lines(seed=60))
    r = run_runner("--params", str(p), "--out", str(tmp_path / "o2"), "--clone", str(clone))
    assert r.returncode == 2 and "is not s42's 8M pair" in r.stdout


@needs_runs
def test_an_unexpected_error_stops_run_all_and_fails(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook(config_extra="raise ValueError('a broken config cell')\n"))
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone))
    rec = json.loads((out / "record.json").read_text())
    assert r.returncode == 1 and rec["run_all"] == {"stopped_at": 2, "stopped_by_the_guard": False}
    assert rec["cells"][-1]["exception"] == "ValueError" and len(rec["cells"]) == 2


@needs_runs
def test_a_failing_manual_cell_fails_the_run(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook(manual_extra="raise RuntimeError('training failed')\n"))
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone))
    rec = json.loads((out / "record.json").read_text())
    assert r.returncode == 1 and rec["cells"][-1]["cell"] == 6 and rec["cells"][-1]["message"] == "training failed"


@needs_runs
def test_stop_at_learn_ends_the_manual_cell_at_learn_and_checks_the_entry_lines(tmp_path):
    body = textwrap.dedent('''\
        from stable_baselines3 import PPO
        PPO.learn(None, total_timesteps=2000000, reset_num_timesteps=True)
        print("NOT REACHED: learn returned")
        ''')
    clone = synthetic_clone(tmp_path, synthetic_notebook(manual_body=body))
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone),
                   "--cadence-override", "32768", "--stop-at-learn")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((out / "record.json").read_text())
    assert rec["manual_cell_stopped_at_learn"] and rec["guard_hits"] == []
    assert rec["learn_called"][0]["total_timesteps"] == 2000000 and rec["learn_called"][0]["reset_num_timesteps"]
    assert any(s.endswith("PPO.learn") for s in rec["learn_stops"]["learn_stops"])
    assert "torch.autograd.backward" in rec["learn_stops"]["guards"]
    log = (out / "runner.stdout.log").read_text()
    assert "NOT REACHED" not in log and "DISCONNECT finished" not in log
    assert rec["entry_lines"]["ok"] and rec["entry_lines"]["first_eval_ok"] is None


@needs_runs
def test_stop_at_learn_counts_a_guard_hit_as_a_failure(tmp_path):
    body = textwrap.dedent('''\
        from stable_baselines3 import PPO
        try:
            PPO.train(None)
        except BaseException as exc:
            print("blocked:", type(exc).__name__)
        PPO.learn(None, total_timesteps=1)
        ''')
    clone = synthetic_clone(tmp_path, synthetic_notebook(manual_body=body))
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone),
                   "--stop-at-learn")
    rec = json.loads((out / "record.json").read_text())
    assert r.returncode == 1 and rec["guard_hits"] == ["PPO.train"]
    assert "blocked: TrainingBlocked" in (out / "runner.stdout.log").read_text()


def test_the_resume_phase_runs_the_resume_cell_inside_run_all_and_not_the_manual_cell(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    out = tmp_path / "out"
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(out), "--clone", str(clone), "--phase",
                   "resume", "--cadence-override", "32768")
    assert r.returncode == 0, r.stdout + r.stderr
    log = (out / "runner.stdout.log").read_text()
    assert "RESUMING locomotion 32768" in log and "MANUAL " not in log and "DISCONNECT finished" in log
    rec = json.loads((out / "record.json").read_text())
    assert [c["cell"] for c in rec["cells"]] == [1, 2, 3, 4, 5, 8] and rec["entry_lines"]["ok"]


def test_a_colab_host_is_refused(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(tmp_path / "out"), "--clone", str(clone),
                   env_extra={"COLAB_RELEASE_TAG": "x"})
    assert r.returncode == 2 and "looks like Colab" in r.stdout


def test_stop_after_pair_must_land_on_the_cadence_and_not_with_stop_at_learn(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(tmp_path / "out"), "--clone", str(clone),
                   "--cadence-override", "32768", "--stop-after-pair", "50000")
    assert r.returncode == 2 and "not a multiple of the cadence" in r.stdout
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(tmp_path / "out2"), "--clone", str(clone),
                   "--stop-after-pair", "32768", "--stop-at-learn")
    assert r.returncode == 2 and "cannot apply" in r.stdout
    r = run_runner("--params", str(params_file(tmp_path)), "--out", str(tmp_path / "out3"), "--clone", str(clone),
                   "--phase", "resume", "--stop-at-learn")
    assert r.returncode == 2 and "launch's entry" in r.stdout


def test_environments_from_outside_the_clone_is_refused(tmp_path):
    clone = synthetic_clone(tmp_path, synthetic_notebook())
    (clone / "environments" / "__init__.py").unlink()
    (clone / "environments" / "shared" / "__init__.py").unlink()
    (clone / "environments" / "shared" / "config.py").unlink()
    (clone / "environments" / "shared").rmdir()
    (clone / "environments").rmdir()
    r = run_runner("--params", str(params_file(tmp_path, manual=manual_lines(vecnorm='"/x/y.pkl"'))), "--out",
                   str(tmp_path / "out"), "--clone", str(clone), "--phase", "resume")
    assert r.returncode == 2 and "not from the clone" in r.stdout


def test_the_simulated_runtime_loss_exits_75_after_the_pair_with_a_record(tmp_path):
    script = tmp_path / "loss.py"
    script.write_text(textwrap.dedent(f'''\
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(TOOLS)!r})
        import nb_pilot_runner4 as nr
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
