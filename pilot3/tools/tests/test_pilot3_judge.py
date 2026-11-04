"""Tests of pilot3_judge.py and iso.py (pytest; walking pilot 3 scratch, never merged).

Pure tests call the judge's functions on synthetic stage_config.json dicts and model folders. Integration tests run
the judge through iso.py on a clean clone of the pilot branch (PILOT3_TREE, default ../tree) against synthetic run
copies: --check-only acceptances and every refusal, each refusal writing nothing. The synthetic checkpoints are tiny
zips holding SB3's member names and a JSON data member, with a pickled dict as the sidecar: they pass
checkpoint_pair_problem but no probe could roll them, so no test here reaches the probe (that is
test_pilot3_judge_probe.py's).
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot3_judge as pj  # noqa: E402

TREE = Path(os.environ.get("PILOT3_TREE", str(TOOLS / "tree"))).resolve()
PARENT = "sha256:" + pj.PILOT_PARENT_SHA256
PILOT1_PARENT = "sha256:70f3a4447794a01036c27bca5e0377a17db96d0e3d7bfb2d179e9ce94e6f353e"
RECORD = {"schema": "mesozoic.critic-phase/v1", "features": ["ramp", "settled", "discount"], "settle_steps": 200,
          "gamma": 0.98, "width": 3, "step_counter": "the env's _step_count when the observation was made (0 at reset)",
          "input": "the critic's only: appended after the normalized observation, never normalized"}
HP = "sha256:" + "ab" * 32


def good_sc(**run_over) -> dict:
    run = {"seed": 42, "n_envs": 4, "timesteps": 8_000_000, "critic_phase_dropped": copy.deepcopy(RECORD),
           "warm_start_log_std": -2.0, "hyperparameters_sha256": HP, "load_path": "/x/stage1_final.zip",
           "load_mode": "initialize_next_stage", "parent_checkpoint_sha256": PARENT}
    run.update(run_over)
    return {"species": "tyrannosaurus_rex", "embodiment": "anatomical", "stage": 2, "git_commit": "abc",
            "hyperparameters": {"n_steps": 4096, "batch_size": 256},
            "curriculum": {"warm_start_log_std": -2.0, "eval_freq": 131072, "eval_episodes": 10,
                           "checkpoint_freq": 500000, "max_checkpoints": 0, "ramp_timesteps": 2_000_000},
            "run": run}


def problems_of(sc: dict, *, want=pj.PILOT_PARENT_SHA256, budget=8_000_000, report=False, record=RECORD):
    return pj.binding_problems(sc, seed=42, want_parent=want, expect_budget=budget, ramp=2_000_000, warmup=300_000,
                               hp_here=HP, parent_record=record, report_off_schedule=report)


# ---------------------------------------------------------------- pure functions

@pytest.mark.parametrize("budget, end", [(8_000_000, 8_011_776), (4_194_304, 4_194_304), (50_000, 65_536),
                                         (16_384, 16_384), (16_385, 32_768)])
def test_uninterrupted_end_is_whole_rollouts_at_or_past_the_budget(budget, end):
    assert pj.uninterrupted_end(budget, 4 * 4096) == end


def test_constants_are_the_criteria_s():
    assert pj.PILOT_BUDGET == 8_000_000
    assert pj.PILOT_PARENT_SHA256 == "1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6"
    assert pj.WARM_START_LOG_STD == -2.0 and pj.N_ENVS == 4
    assert pj.PROBE_ARGS == ["--species", "tyrannosaurus_rex", "--stage", "locomotion", "--episodes", "40",
                             "--seed-scheme", "panel", "--seed-base", "3042", "--settle-steps", "100"]
    assert pj.BRANCH_PREFIX == "claude/pilot-walk-"


def test_a_run_trained_as_fixed_has_no_problem():
    problems, info = problems_of(good_sc())
    assert problems == [] and info == {"budget": 8_000_000, "resume_step": None, "off_schedule": None,
                                       "resumed": False}


@pytest.mark.parametrize("edit, needle", [
    (lambda sc: sc["run"].pop("critic_phase_dropped"), "run.critic_phase_dropped is absent"),
    (lambda sc: sc["run"]["critic_phase_dropped"].update(settle_steps=140), "is not the parent's critic-phase record"),
    (lambda sc: sc["run"].pop("warm_start_log_std"), "run.warm_start_log_std is absent"),
    (lambda sc: sc["run"].update(warm_start_log_std=-1.0), "run.warm_start_log_std is -1.0"),
    (lambda sc: sc["run"].update(warm_start_log_std=-1.5), "run.warm_start_log_std is -1.5, not -2.0"),
    (lambda sc: sc["curriculum"].update(warm_start_log_std=-1.5), "curriculum.warm_start_log_std is -1.5, not -2.0"),
    (lambda sc: sc["run"].update(warm_start_log_std=True), "run.warm_start_log_std is True"),
    (lambda sc: sc["curriculum"].pop("warm_start_log_std"), "curriculum.warm_start_log_std is absent"),
    (lambda sc: sc["curriculum"].update(warm_start_log_std="-2.0"), "curriculum.warm_start_log_std is '-2.0'"),
    (lambda sc: sc["run"].update(n_envs=2), "run.n_envs is 2, not 4"),
    (lambda sc: sc["run"].pop("n_envs"), "run.n_envs is None, not 4"),
    (lambda sc: sc["run"].update(n_envs=4.0000001), "run.n_envs is 4.0000001"),
    (lambda sc: sc["hyperparameters"].update(critic_phase=False), "hold a critic_phase key (False)"),
    (lambda sc: sc["hyperparameters"].update(critic_phase=True), "hold a critic_phase key (True)"),
    (lambda sc: sc["run"].update(seed=52), "seed 52 (not --seed 42)"),
    (lambda sc: sc["run"].update(load_mode="resume_same_stage"), "load_mode 'resume_same_stage'"),
    (lambda sc: sc["run"].update(timesteps=4_194_304), "budget 4194304 (not 8000000)"),
    (lambda sc: sc["run"].update(hyperparameters_sha256="sha256:" + "cd" * 32), "!= this tree's"),
    (lambda sc: sc.update(stage=1), "stage 1"),
    (lambda sc: sc["run"].update(parent_checkpoint_sha256=PILOT1_PARENT), "is not the chosen parent"),
])
def test_each_binding_record_is_refused_when_absent_or_different(edit, needle):
    sc = good_sc()
    edit(sc)
    problems, _ = problems_of(sc)
    assert len(problems) == 1 and needle in problems[0], problems


def test_pilot_1_s_parent_is_refused_both_as_the_argument_and_against_the_run():
    problems, _ = problems_of(good_sc(), want=pj.norm_sha(PILOT1_PARENT))
    assert len(problems) == 2
    assert "is not walking pilot 3's parent" in problems[0] and "is not the chosen parent" in problems[1]


def test_an_unreadable_parent_record_refuses():
    problems, _ = problems_of(good_sc(), record=None)
    assert problems == ["the parent's critic-phase record could not be read, so run.critic_phase_dropped cannot be bound"]


def test_every_new_problem_is_listed_at_once():
    sc = good_sc(n_envs=1)
    sc["run"].pop("critic_phase_dropped")
    sc["run"].pop("warm_start_log_std")
    sc["curriculum"].pop("warm_start_log_std")
    sc["hyperparameters"]["critic_phase"] = False
    assert len(pj.warm_start_problems(sc, RECORD)) == 5


# the resume branch

def resumed(step: int, segment: int, **over) -> dict:
    return good_sc(timesteps=segment, resume_load_path=f"/d/03_locomotion/models/stage2_{step}_steps.zip",
                   resume_checkpoint_sha256="sha256:" + "ef" * 32, **over)


@pytest.mark.parametrize("step", [2_000_000, 4_000_000, 6_000_000, 7_500_000])
def test_a_resume_on_schedule_reads_the_budget_as_the_segment_plus_the_resume_step(step):
    problems, info = problems_of(resumed(step, 8_000_000 - step))
    assert problems == [] and info["budget"] == 8_000_000 and info["resume_step"] == step and info["resumed"]


def test_a_resume_whose_sum_is_not_the_budget_is_refused():
    problems, info = problems_of(resumed(6_000_000, 2_500_000))
    assert info["budget"] == 8_500_000 and len(problems) == 1 and "budget 8500000 (not 8000000)" in problems[0]


@pytest.mark.parametrize("step", [32_768, 500_000, 1_500_000, 1_999_999])
def test_a_resume_before_ramp_timesteps_is_off_schedule(step):
    problems, info = problems_of(resumed(step, 8_000_000 - step))
    assert info["off_schedule"] and len(problems) == 1 and problems[0].startswith("OFF-SCHEDULE: resumed at step")
    problems, info = problems_of(resumed(step, 8_000_000 - step), report=True)
    assert problems == [] and info["off_schedule"].startswith("OFF-SCHEDULE")


def test_a_resume_from_a_non_periodic_checkpoint_is_refused():
    sc = good_sc(timesteps=1_000_000, resume_load_path="/d/models/stage2_final.zip")
    problems, info = problems_of(sc)
    assert info["budget"] is None and any("not a periodic stage2_<steps>_steps checkpoint" in p for p in problems)


@pytest.mark.parametrize("key", ["critic_phase_dropped", "warm_start_log_std", "n_envs"])
def test_a_resumed_copy_is_held_to_the_warm_start_records(key):
    sc = resumed(6_000_000, 2_000_000)
    sc["run"].pop(key)
    problems, _ = problems_of(sc)
    assert len(problems) == 1 and key in problems[0]


def test_a_resumed_copy_with_a_critic_phase_key_is_refused():
    sc = resumed(6_000_000, 2_000_000)
    sc["hyperparameters"]["critic_phase"] = True
    assert len(problems_of(sc)[0]) == 1


def test_resume_check_and_cell_path_are_pilot_1_s(tmp_path):
    assert pj.resume_check({}, 2_000_000, 300_000) == (None, None)
    step, why = pj.resume_check({"resume_load_path": "stage2_2000000_steps"}, 2_000_000, 300_000)
    assert step == 2_000_000 and why is None
    run = tmp_path / "pilot3_walk_s42"
    assert pj.cell_path(tmp_path, 42, 8_000_000, None, run) == tmp_path / "s42" / "8000000"
    assert pj.cell_path(tmp_path, 42, 8_000_000, "x", run) == tmp_path / "off_schedule" / "pilot3_walk_s42" / "s42" / "8000000"


# the final-model fallback

def write_ckpt(path: Path, num_timesteps: int | None = None, *, members=("data", "policy.pth")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name in members:
            body = json.dumps({"num_timesteps": num_timesteps}) if name == "data" else "synthetic"
            z.writestr(name, body)
    return path


def write_vec(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"synthetic": True}))
    return path


def fallback(models: Path, *, steps=8_000_000, budget=8_000_000, resumed=False, intact=None):
    def pair_problem(z, v):  # the tree's function needs the tree; this one reports what the test says
        return intact
    return pj.final_fallback_problems(models, steps=steps, expect_budget=budget, rollout=16_384, resumed=resumed,
                                      pair_problem=pair_problem)


def test_fallback_accepts_a_final_pair_at_the_uninterrupted_end_when_the_deciding_pair_is_missing(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 8_011_776)
    problems, rec = fallback(tmp_path)
    assert problems == [] and rec["uninterrupted_end"] == 8_011_776 and rec["final_num_timesteps"] == 8_011_776
    assert rec["deciding_pair_problem"].startswith("missing")


def test_fallback_accepts_when_the_deciding_pair_is_broken(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 8_011_776)
    write_ckpt(tmp_path / "stage2_8000000_steps.zip", 8_000_000)
    problems, rec = fallback(tmp_path, intact="missing matched VecNormalize sidecar")
    assert problems == [] and rec["deciding_pair_problem"] == "missing matched VecNormalize sidecar"


@pytest.mark.parametrize("end", [8_000_000, 8_015_232, 7_995_392, 8_028_160])
def test_fallback_refuses_any_other_num_timesteps(tmp_path, end):
    write_ckpt(tmp_path / "stage2_final.zip", end)
    problems, _ = fallback(tmp_path)
    assert len(problems) == 1 and "not the uninterrupted end 8,011,776" in problems[0]


def test_fallback_refuses_when_the_deciding_pair_is_intact(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 8_011_776)
    write_ckpt(tmp_path / "stage2_8000000_steps.zip", 8_000_000)
    problems, _ = fallback(tmp_path, intact=None)
    assert len(problems) == 1 and "is intact: judge it" in problems[0]


def test_fallback_refuses_another_step_a_resumed_run_and_an_unreadable_final(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 8_011_776)
    assert "stands in only for the deciding checkpoint" in fallback(tmp_path, steps=6_000_000)[0][0]
    assert "a resumed run's final pair" in fallback(tmp_path, resumed=True)[0][0]
    write_ckpt(tmp_path / "bad" / "stage2_final.zip", None)
    problems, _ = fallback(tmp_path / "bad")
    assert len(problems) == 1 and "has no readable num_timesteps" in problems[0]


def test_fallback_for_the_smoke_budget_is_65_536(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 65_536)
    problems, rec = fallback(tmp_path, steps=50_000, budget=50_000)
    assert problems == [] and rec["uninterrupted_end"] == 65_536


def test_run_end_reads_the_final_num_timesteps(tmp_path):
    for s in (500_000, 1_000_000, 8_000_000):
        write_ckpt(tmp_path / "models" / f"stage2_{s}_steps.zip", s)
    write_ckpt(tmp_path / "models" / "stage2_final.zip", 8_011_776)
    end = pj.run_end(tmp_path / "models", 8_000_000, tmp_path)
    assert end["newest_periodic_steps"] == 8_000_000 and end["final_num_timesteps"] == 8_011_776
    assert end["stopped_early"] is False


def test_run_end_names_the_newest_intact_pair(tmp_path):
    """FIX2: a collapse-stopped seed is scored at its newest INTACT pair, not its newest zip by file name."""
    models = tmp_path / "models"
    for s in (5_500_000, 6_000_000, 6_500_000):
        write_ckpt(models / f"stage2_{s}_steps.zip", s)
    write_vec(models / "stage2_vecnormalize_5500000_steps.pkl")
    write_vec(models / "stage2_vecnormalize_6000000_steps.pkl")  # 6,500,000 has no sidecar
    write_ckpt(models / "stage2_final.zip", 6_553_600)

    def pair_problem(z, v):
        return "does not unpickle" if "6000000" in Path(v).name else None
    end = pj.run_end(models, 8_000_000, tmp_path, pair_problem)
    assert end["newest_periodic_steps"] == 6_500_000 and end["newest_intact_periodic_steps"] == 5_500_000
    assert end["intact_periodic_steps"] == [5_500_000]
    assert end["broken_periodic"] == {"6000000": "does not unpickle",
                                      "6500000": "missing sidecar stage2_vecnormalize_6500000_steps.pkl"}
    assert end["stopped_early"] is True
    unchecked = pj.run_end(models, 8_000_000, tmp_path)
    assert unchecked["intact_periodic_steps"] is None and unchecked["newest_intact_periodic_steps"] is None
    none = pj.run_end(tmp_path / "empty", 8_000_000, tmp_path, pair_problem)
    assert none["intact_periodic_steps"] == [] and none["newest_intact_periodic_steps"] is None


@pytest.mark.parametrize("recorded, ok", [
    ("3ae1c95967547ada569ad71aebaa88999afa02cd", True), ("3AE1C95967547ADA569AD71AEBAA88999AFA02CD", True),
    ("3ae1c959", True), ("3ae1c95", True), ("3ae1c9", False), ("3ae1c959x", False), (None, False), ("", False),
    ("3ae1c95967547ada" + "0" * 24, False), ("76ceae9d1d0985b2585b2266a4d26d9a34df820b", False),
    ("3ae1c95967547ada569ad71aebaa88999afa02cd0", False)])
def test_commit_problem_needs_7_to_40_hex_digits_that_begin_head(recorded, ok):
    got = pj.commit_problem(recorded, "3ae1c95967547ada569ad71aebaa88999afa02cd")
    assert (got is None) == ok, got
    if recorded in (None, ""):
        assert got == "stage_config.json records no git_commit"


def test_other_run_cells_lists_cells_of_another_run_copy(tmp_path):
    out = tmp_path / "OUT"
    for steps, run in ((2_000_000, "/a/pilot3_walk_s42"), (4_000_000, "/b/pilot3_walk_s42_b"),
                       (6_000_000, "/c/pilot3_walk_s42")):
        (out / "s42" / str(steps)).mkdir(parents=True)
        (out / "s42" / str(steps) / "binding.json").write_text(json.dumps({"run": run}))
    (out / "s42" / "7000000.partial").mkdir()
    (out / "s42" / "7000000.partial" / "binding.json").write_text(json.dumps({"run": "/z/other"}))
    (out / "s42" / "8000000").mkdir()
    (out / "s42" / "8000000" / "binding.json").write_text("not json")
    got = pj.other_run_cells(out, 42, "pilot3_walk_s42_b")
    assert got == ["2000000 (pilot3_walk_s42)", "6000000 (pilot3_walk_s42)",
                   "8000000 (unreadable binding.json: JSONDecodeError)"]
    assert pj.other_run_cells(out, 52, "pilot3_walk_s52") == []


def test_isolated_path_puts_the_tree_first_and_drops_the_tools_folder_and_the_empty_entry():
    import iso
    got = iso.isolated_path("/t", ["/x", "/t"], ["", "/tools", "/lib", "/x"], "/tools")
    assert got == ["/t", "/x", "/lib"]


# ---------------------------------------------------------------- integration: the judge on a clean clone

needs_tree = pytest.mark.skipif(not (TREE / "environments" / "__init__.py").is_file(), reason=f"no tree at {TREE}")


def iso_run(script: Path, *args: str, cwd: Path = TREE, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, ISO_WT=str(TREE), MUJOCO_GL="disable", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-I", str(TOOLS / "iso.py"), str(script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def facts():
    if not (TREE / "environments" / "__init__.py").is_file():
        pytest.skip(f"no tree at {TREE}")
    r = iso_run(TOOLS / "tests" / "_tree_facts.py")
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def make_run(root: Path, facts: dict, *, name="pilot3_walk_s42", run_over: dict | None = None,
             periodic=(500_000, 8_000_000), final: int | None = 8_011_776, edit=None) -> Path:
    facts = copy.deepcopy(facts)  # the module fixture is shared: edits stay in this run copy
    run = root / name
    stage = run / "03_locomotion"
    models = stage / "models"
    for s in periodic:
        write_ckpt(models / f"stage2_{s}_steps.zip", s)
        write_vec(models / f"stage2_vecnormalize_{s}_steps.pkl")
    if final is not None:
        write_ckpt(models / "stage2_final.zip", final)
        write_vec(models / "stage2_final_vecnorm.pkl")
    block = {"seed": 42, "n_envs": 4, "timesteps": 8_000_000, "critic_phase_dropped": facts["parent_record"],
             "warm_start_log_std": -2.0, "hyperparameters_sha256": facts["hp"],
             "load_path": "/content/mesozoic-labs/pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final.zip",
             "load_mode": "initialize_next_stage", "parent_checkpoint_sha256": PARENT,
             "parent_task_sha256": "sha256:" + "11" * 32}
    block.update(run_over or {})
    sc = {"species": "tyrannosaurus_rex", "embodiment": "anatomical", "stage": 2, "name": "Locomotion",
          "algorithm": "PPO", "git_commit": facts["head"], "hyperparameters": facts["hyperparameters"],
          "curriculum": facts["curriculum"], "run": block, "plant_identity": facts["plant"]}
    if edit:
        edit(sc)
    (stage / "stage_config.json").write_text(json.dumps(sc, indent=1))
    (stage / "plant_identity.json").write_text(json.dumps(facts["plant"], indent=1))
    return run


def judge(run: Path, out: Path, *args: str, seed=42, steps=8_000_000, parent=PARENT) -> subprocess.CompletedProcess:
    return iso_run(TOOLS / "pilot3_judge.py", str(run), "--seed", str(seed), "--steps", str(steps),
                   "--parent-sha256", parent, "--out", str(out), *args)


def assert_refused(r: subprocess.CompletedProcess, out: Path, needle: str) -> None:
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "refusing (nothing written)" in r.stdout and needle in r.stdout, r.stdout
    assert not out.exists() or not any(out.rglob("*")), list(out.rglob("*"))


@needs_tree
def test_the_tree_is_a_clean_clone_of_the_pilot_branch(facts):
    assert pj.git(TREE, "branch", "--show-current").startswith("claude/pilot-walk-")
    assert pj.git(TREE, "status", "--porcelain", "--untracked-files=no") == ""
    assert facts["curriculum"]["warm_start_log_std"] == -2.0 and facts["n_steps"] == 4096


@needs_tree
def test_check_only_accepts_a_run_trained_as_fixed_and_writes_nothing(tmp_path, facts):
    out = tmp_path / "OUT"
    r = judge(make_run(tmp_path, facts), out, "--check-only", "--expect-head", facts["head"])
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "all checks passed (--check-only" in r.stdout and str(out / "s42" / "8000000") in r.stdout
    assert not out.exists()


@needs_tree
@pytest.mark.parametrize("label, kwargs, needle", [
    ("pilot 1's parent", {"parent": PILOT1_PARENT}, "is not walking pilot 3's parent"),
    ("no drop record", {"run_over": {"critic_phase_dropped": None}}, "run.critic_phase_dropped is absent"),
    ("no reset record", {"run_over": {"warm_start_log_std": None}}, "run.warm_start_log_std is absent"),
    ("n_envs 2", {"run_over": {"n_envs": 2}}, "run.n_envs is 2, not 4"),
    ("another head", {"run_over": {}, "commit": "4d5f4e76232d6e19a4526e2945c28228d561f30d"}, "!= tree HEAD"),
    ("a critic_phase key", {"edit": lambda sc: sc["hyperparameters"].update(critic_phase=False)},
     "hold a critic_phase key"),
    ("no curriculum reset key", {"edit": lambda sc: sc["curriculum"].pop("warm_start_log_std")},
     "curriculum.warm_start_log_std is absent"),
    ("seed 52 judged as 42", {"run_over": {"seed": 52}}, "seed 52 (not --seed 42)"),
    ("no git_commit (FIX2)", {"edit": lambda sc: sc.pop("git_commit")}, "stage_config.json records no git_commit"),
    ("a git_commit that only extends HEAD's prefix (FIX2)",
     {"edit": lambda sc: sc.update(git_commit=sc["git_commit"][:12] + "0" * 30)}, "!= tree HEAD"),
])
def test_refusals_on_the_tree_write_nothing(tmp_path, facts, label, kwargs, needle):
    kwargs = dict(kwargs)
    parent = kwargs.pop("parent", PARENT)
    commit = kwargs.pop("commit", None)
    run_over = {k: v for k, v in kwargs.pop("run_over", {}).items()}
    drop = [k for k, v in run_over.items() if v is None]
    run_over = {k: v for k, v in run_over.items() if v is not None}

    _edit = kwargs.pop("edit", None)

    def edit(sc):
        for k in drop:
            sc["run"].pop(k)
        if commit:
            sc["git_commit"] = commit
        if _edit:
            _edit(sc)
    out = tmp_path / "OUT"
    r = judge(make_run(tmp_path, facts, run_over=run_over, edit=edit), out, parent=parent)
    assert_refused(r, out, needle)


@needs_tree
def test_refusals_for_the_tree_state(tmp_path, facts):
    out = tmp_path / "OUT"
    run = make_run(tmp_path, facts)
    assert_refused(judge(run, out, "--expect-head", "4d5f4e7"), out, "is not 4d5f4e7")
    r = judge(run, out, "--check-only", "--expect-head", facts["head"][:12])
    assert r.returncode == 0, r.stdout


@needs_tree
def test_a_missing_or_broken_pair_is_refused(tmp_path, facts):
    out = tmp_path / "OUT"
    run = make_run(tmp_path, facts)
    (run / "03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl").unlink()
    assert_refused(judge(run, out), out, "checkpoint missing or incomplete run copy")
    write_vec(run / "03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl")
    write_ckpt(run / "03_locomotion/models/stage2_8000000_steps.zip", 8_000_000, members=("data",))
    assert_refused(judge(run, out), out, "not an intact pair: bad/truncated checkpoint zip")
    write_ckpt(run / "03_locomotion/models/stage2_8000000_steps.zip", 8_000_000)
    (run / "03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl").write_bytes(b"not a pickle")
    assert_refused(judge(run, out), out, "does not unpickle")


@needs_tree
def test_the_resume_branch_on_the_tree(tmp_path, facts):
    out = tmp_path / "OUT"
    on = make_run(tmp_path / "a", facts, run_over={
        "timesteps": 2_000_000, "resume_load_path": "/d/stage2_6000000_steps.zip"})
    r = judge(on, out, "--check-only")
    assert r.returncode == 0, r.stdout
    off = make_run(tmp_path / "b", facts, run_over={
        "timesteps": 8_000_000 - 32_768, "resume_load_path": "/d/stage2_32768_steps.zip"})
    assert_refused(judge(off, out), out, "OFF-SCHEDULE: resumed at step 32,768")
    r = judge(off, out, "--report-off-schedule", "--check-only")
    assert r.returncode == 0 and str(out / "off_schedule" / "pilot3_walk_s42" / "s42" / "8000000") in r.stdout
    assert "OFF-SCHEDULE" in r.stdout
    bad_sum = make_run(tmp_path / "c", facts, run_over={
        "timesteps": 2_000_000, "resume_load_path": "/d/stage2_4000000_steps.zip"})
    assert_refused(judge(bad_sum, out), out, "budget 6000000 (not 8000000)")


@needs_tree
def test_final_fallback_on_the_tree(tmp_path, facts):
    out = tmp_path / "OUT"
    run = make_run(tmp_path / "a", facts, periodic=(500_000, 7_500_000))
    r = judge(run, out, "--final-fallback", "--check-only")
    assert r.returncode == 0 and "[final-model fallback]" in r.stdout, r.stdout
    assert_refused(judge(run, out, "--check-only"), out, "checkpoint missing or incomplete run copy")
    other = make_run(tmp_path / "b", facts, periodic=(500_000,), final=8_015_232)
    assert_refused(judge(other, out, "--final-fallback"), out, "not the uninterrupted end 8,011,776")
    intact = make_run(tmp_path / "c", facts)
    assert_refused(judge(intact, out, "--final-fallback"), out, "is intact: judge it")
    smoke = make_run(tmp_path / "d", facts, periodic=(32_768, 65_536), final=65_536,
                     run_over={"timesteps": 50_000})
    r = judge(smoke, out, "--final-fallback", "--check-only", "--expect-budget", "50000", steps=50_000)
    assert r.returncode == 0, r.stdout
    r = judge(smoke, out, "--check-only", "--expect-budget", "50000", steps=65_536)
    assert r.returncode == 0, r.stdout


@needs_tree
def test_a_restart_is_not_judged_into_a_seed_folder_holding_the_retired_run_s_cells(tmp_path, facts):
    """FIX2: the retired run's cells must be moved to OUT/retired/<run>/ first; then the _b run is judged."""
    out = tmp_path / "OUT"
    old = out / "s42" / "2000000"
    old.mkdir(parents=True)
    (old / "binding.json").write_text(json.dumps({"run": str(tmp_path / "x" / "pilot3_walk_s42")}))
    b_run = make_run(tmp_path / "b", facts, name="pilot3_walk_s42_b")
    r = judge(b_run, out, "--check-only")
    assert r.returncode == 2 and "holds cells judged from another run than pilot3_walk_s42_b" in r.stdout, r.stdout
    assert "2000000 (pilot3_walk_s42)" in r.stdout and "OUT/retired/" in r.stdout
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == ["s42", "s42/2000000",
                                                                             "s42/2000000/binding.json"]
    retired = out / "retired" / "pilot3_walk_s42" / "s42"
    retired.mkdir(parents=True)
    old.rename(retired / "2000000")
    r = judge(b_run, out, "--check-only")
    assert r.returncode == 0 and str(out / "s42" / "8000000") in r.stdout, r.stdout
    same = out / "s42" / "4000000"
    same.mkdir(parents=True)
    (same / "binding.json").write_text(json.dumps({"run": str(b_run)}))
    assert judge(b_run, out, "--check-only").returncode == 0


@needs_tree
def test_iso_refuses_without_a_tree_and_rewrites_pythonpath(tmp_path):
    probe = tmp_path / "show.py"
    probe.write_text("import os, sys, environments\nprint(os.environ['PYTHONPATH'].split(os.pathsep)[0])\n"
                     "print(sys.path[0])\nprint(environments.__file__)\nprint(sys.dont_write_bytecode)\n")
    r = iso_run(probe, extra_env={"PYTHONPATH": str(tmp_path)})
    lines = r.stdout.split()
    assert r.returncode == 0 and lines[0] == str(TREE) and lines[1] == str(TREE)
    assert lines[2].startswith(str(TREE) + "/") and lines[3] == "True"
    r = iso_run(probe, extra_env={"ISO_WT": str(tmp_path)})
    assert r.returncode == 2 and "is not a tree" in r.stderr
    exits = tmp_path / "exits.py"
    exits.write_text("import sys\nsys.exit(7)\n")
    assert iso_run(exits).returncode == 7
