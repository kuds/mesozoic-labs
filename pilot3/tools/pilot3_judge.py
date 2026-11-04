"""Judge ONE checkpoint of a walking-pilot-3 run: the gait probe's 40-episode floor-truth panel.

Scratch, never merged: pilot 1's pilot_judge.py (sha256 2dc44b6a...) adapted for walking pilot 3 (criteria_pilot3.txt,
"How it is judged"). Run with the pilot branch's clean tree as the working directory (a clean clone of
pilot-walk-r10-p3 at the head the runs recorded), isolated, e.g.
    cd TREE && ISO_WT=TREE python3 -I TOOLS/iso.py TOOLS/pilot3_judge.py RUN --seed 42 --steps 8000000
        --parent-sha256 sha256:1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6 --out OUT
        [--expect-head HEAD]
RUN is a local copy of the pilot run directory (paths relative to the run, as on Drive):
    03_locomotion/models/stage2_<steps>_steps.zip   03_locomotion/models/stage2_vecnormalize_<steps>_steps.pkl
    03_locomotion/stage_config.json                 03_locomotion/plant_identity.json
    03_locomotion/models/stage2_final.zip and stage2_final_vecnorm.pkl, or 03_locomotion/stage_summary.txt
        (optional: the run-end record below; required under --final-fallback)
It writes OUT/s<seed>/<steps>/probe.json (the probe's JSON), the probe's trace and plots beside it, and binding.json;
pilot3_score.py reads probe.json and binding.json. Nothing is written when a check refuses.
An off-schedule run judged with --report-off-schedule is written to OUT/off_schedule/<RUN's directory name>/s<seed>/
<steps>/ instead, never to OUT/s<seed>/<steps>: the seed's restart (a fresh "_b" RUN_ID, judged into OUT/s<seed>/) can
then never collide with it, and pilot3_score.py lists it as reported, never scored.

Checks before rolling anything (each refusal names its reason; pilot 1's, then pilot 3's):
  * the tree is clean and on a pilot branch head (--expect-head pins one); its git reads take no optional locks;
  * stage_config.json records the locomotion stage (2) of tyrannosaurus_rex/anatomical, an initialize_next_stage load,
    the seed --seed, the pilot's budget 8,000,000, and a git_commit equal to the tree's HEAD: 7 to 40 hex digits
    that begin HEAD; a stage_config.json with no git_commit is refused (FIX2; pilot 1's judge skipped the check
    then) (--allow-commit-mismatch relaxes only this one, and binding.json says so);
  * its run.parent_checkpoint_sha256 equals --parent-sha256, which must be walking pilot 3's parent, the PC round's
    stance7_r10LTSBpc_s42 final (PILOT_PARENT_SHA256);
  * a resumed run (run.resume_load_path set: the notebook's RESUME cell) resumed at or after the stage's
    ramp_timesteps (read from this tree's config, 2,000,000). A RESUME attaches no stage-entry shaping, so a resume
    before the ramp's end is OFF-SCHEDULE and refused; --report-off-schedule judges it anyway for the report and marks
    binding.json off_schedule, which pilot3_score.py never scores. The budget of a resumed run is its recorded
    run.timesteps (the segment the RESUME cell trained) plus the step it resumed from. stage_config.json keeps only
    the LATEST resume;
  * its hyperparameters_sha256 equals the one this tree derives for the stage (it covers warm_start_log_std);
  * plant_identity.json's physics and policy-interface digests equal the tree's;
  * NEW (pilot 3, C4/C5): run.critic_phase_dropped equals the parent's critic-phase record (read, never unpickled, from
    the JSON data member of the tree's committed parent, pilot_checkpoints/stance7_r10LTSBpc_s42_final/
    stage1_final.zip, whose sha256 must be PILOT_PARENT_SHA256); run.warm_start_log_std and
    curriculum.warm_start_log_std are both -2.0; run.n_envs is 4; the recorded hyperparameters hold no critic_phase
    key. A resumed copy is held to the same records (the RESUME cell keeps both warm-start records);
  * NEW: the judged checkpoint is an intact pair: both files exist and pass the tree's checkpoint_pair_problem (the
    zip opens with SB3's members, the sidecar unpickles: the notebook's RESUME-cell test); the probe's own loader
    then enforces the plant contract (exit 3 = refused).
--final-fallback (NEW; criteria "Final-model fallback"): judge the run's final pair (stage2_final.zip with
stage2_final_vecnorm.pkl) in place of the deciding checkpoint, only when --steps is the budget (--expect-budget: the
deciding checkpoint at 8M), that periodic pair is missing or broken, the final pair is intact, and the final zip's
num_timesteps (read from its JSON data member) equals the uninterrupted end the judge computes from the budget: whole
rollouts of N_ENVS x n_steps (16,384) at or past it (8,000,000 -> 8,011,776; the smoke's 50,000 -> 65,536). The
cell is still OUT/s<seed>/<steps>/, and binding.json records final_fallback with the reason. A resumed run's final
ends elsewhere (it continues from its checkpoint's step in whole rollouts), so it is refused.
--check-only (NEW): run every check above and stop before the probe; nothing is written.
Run-end record (binding.json run_end, read by pilot3_score.py for a seed without an 8,000,000 checkpoint): the newest
periodic checkpoint in the run copy, the newest INTACT periodic pair (FIX2: each stage2_<steps>_steps.zip with its
stage2_vecnormalize_<steps>_steps.pkl passing the tree's checkpoint_pair_problem; the criteria score a
collapse-stopped seed there), and the steps the run ended at, from stage2_final.zip's num_timesteps (else
stage_summary.txt's "Timesteps:"). A final model short of the budget means the collapse backstop stopped the run.
One run per seed in OUT (FIX2): the judge refuses to write OUT/s<seed>/<steps> while OUT/s<seed>/ holds a cell judged
from a run copy with another directory name (the RUN_ID). When a seed is restarted under _b, the retired run's cells
are moved, never deleted, to OUT/retired/<retired RUN_ID>/s<seed>/<steps>/ before the _b run is judged; the scorer
lists them and never scores them. This refusal concerns OUT, not the run, so it restarts nothing. A run copy keeps its
RUN_ID as its directory name.
Probe arguments are fixed: --stage locomotion --episodes 40 --seed-scheme panel --seed-base 3042 --settle-steps 100.
MUJOCO_GL is passed to the probe as set, else "disable" (the probe renders nothing; pilot 1 used "egl").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT_BUDGET = 8_000_000
PILOT_PARENT_SHA256 = "1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6"
PARENT_IN_TREE = Path("pilot_checkpoints") / "stance7_r10LTSBpc_s42_final" / "stage1_final.zip"
CRITIC_PHASE_ATTRIBUTE = "mesozoic_critic_phase"
WARM_START_LOG_STD = -2.0
N_ENVS = 4
BRANCH_PREFIX = "claude/pilot-walk-"
PROBE_ARGS = ["--species", "tyrannosaurus_rex", "--stage", "locomotion", "--episodes", "40", "--seed-scheme", "panel",
              "--seed-base", "3042", "--settle-steps", "100"]
PERIODIC_RE = re.compile(r"stage2_(\d+)_steps(\.zip)?$")
FINAL_ZIP, FINAL_VEC = "stage2_final.zip", "stage2_final_vecnorm.pkl"
CADENCE_KEYS = ("eval_freq", "eval_episodes", "checkpoint_freq", "max_checkpoints")


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def norm_sha(value: object) -> str | None:
    if not value:
        return None
    s = str(value).strip().lower()
    s = s[len("sha256:"):] if s.startswith("sha256:") else s
    return s if re.fullmatch(r"[0-9a-f]{64}", s) else None


def refuse(msg: str) -> int:
    print(f"refusing (nothing written): {msg}", flush=True)
    return 2


def git(tree: Path, *args: str) -> str:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(["git", *args], cwd=tree, text=True, env=env).strip()


def read_zip_data(path: Path) -> dict:
    """An SB3 archive's JSON 'data' member (json and zipfile only: nothing is unpickled)."""
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("data"))


def resume_check(run_block: dict, ramp_timesteps: int, warmup_timesteps: int) -> tuple[int | None, str | None]:
    """(resume step or None, off-schedule reason or None) for a stage_config run block (pilot 1's, unchanged)."""
    resumed = run_block.get("resume_load_path")
    if not resumed:
        return None, None
    m = PERIODIC_RE.match(Path(str(resumed)).name)
    if not m:
        return None, f"resumed from {resumed}, which is not a periodic stage2_<steps>_steps checkpoint"
    step = int(m.group(1))
    shaping_end = max(int(ramp_timesteps), int(warmup_timesteps))
    if step < shaping_end:
        return step, (f"OFF-SCHEDULE: resumed at step {step:,} < ramp_timesteps {int(ramp_timesteps):,} (warm-up "
                      f"{int(warmup_timesteps):,}): the RESUME cell attaches no warm-up or forward-weight ramp, so steps "
                      f"{step:,}-{shaping_end:,} trained at the full forward weight; restart the seed from the parent "
                      f"in a fresh RUN_ID")
    return step, None


def cell_path(out: Path, seed: int, steps: int, off_schedule: str | None, run: Path) -> Path:
    """Where one judgement goes: OUT/s<seed>/<steps>, or OUT/off_schedule/<run dir name>/s<seed>/<steps> for an
    off-schedule run judged with --report-off-schedule (pilot 1's, unchanged)."""
    if off_schedule:
        return out / "off_schedule" / run.name / f"s{seed}" / str(steps)
    return out / f"s{seed}" / str(steps)


def run_end(models: Path, budget: int | None, stage_dir: Path, pair_problem=None) -> dict:
    """Pilot 1's run-end record, plus (FIX2) the intact periodic pairs: pair_problem is the tree's
    checkpoint_pair_problem (None: not checked, the intact fields are None)."""
    steps = sorted(int(m.group(1)) for p in models.glob("stage2_*_steps.zip") if (m := PERIODIC_RE.match(p.name)))
    intact, broken = None, None
    if pair_problem is not None:
        intact, broken = [], {}
        for s in steps:
            vec = models / f"stage2_vecnormalize_{s}_steps.pkl"
            why = f"missing sidecar {vec.name}" if not vec.is_file() else pair_problem(
                models / f"stage2_{s}_steps.zip", vec)
            if why is None:
                intact.append(s)
            else:
                broken[str(s)] = str(why)
    final = models / FINAL_ZIP
    end, source = None, None
    if final.is_file():
        try:
            with zipfile.ZipFile(final) as z:
                end, source = int(json.loads(z.read("data"))["num_timesteps"]), "stage2_final.zip num_timesteps"
        except (KeyError, ValueError, zipfile.BadZipFile) as exc:
            source = f"stage2_final.zip unreadable: {type(exc).__name__}"
    if end is None and (stage_dir / "stage_summary.txt").is_file():
        m = re.search(r"^Timesteps:\s*([\d,]+)", (stage_dir / "stage_summary.txt").read_text(), re.M)
        if m:
            end, source = int(m.group(1).replace(",", "")), "stage_summary.txt Timesteps"
    return {"periodic_steps": steps, "newest_periodic_steps": steps[-1] if steps else None,
            "intact_periodic_steps": intact,
            "newest_intact_periodic_steps": (intact[-1] if intact else None) if intact is not None else None,
            "broken_periodic": broken,
            "final_num_timesteps": end, "final_source": source, "budget": budget,
            "stopped_early": None if end is None or budget is None else end < budget}


def uninterrupted_end(budget: int, rollout: int) -> int:
    """The num_timesteps an unbroken run ends at: whole rollouts at or past the budget (SB3 trains while
    num_timesteps < total_timesteps)."""
    if budget <= 0 or rollout <= 0:
        raise ValueError(f"budget {budget} and rollout {rollout} must be positive")
    return math.ceil(budget / rollout) * rollout


def warm_start_problems(sc: dict, parent_record: dict | None) -> list[str]:
    """Pilot 3's new binding checks on a stage_config.json (C4, C5): the drop record, both reset records, n_envs 4 and
    no critic_phase key in the recorded hyperparameters. Every problem is listed."""
    run_block = sc.get("run") or {}
    cur = sc.get("curriculum") or {}
    hyper = sc.get("hyperparameters") or {}
    problems = []
    dropped = run_block.get("critic_phase_dropped")
    if parent_record is None:
        problems.append("the parent's critic-phase record could not be read, so run.critic_phase_dropped cannot be bound")
    elif dropped is None:
        problems.append("run.critic_phase_dropped is absent (the warm start dropped no critic phase input)")
    elif dropped != parent_record:
        problems.append(f"run.critic_phase_dropped {dropped!r} is not the parent's critic-phase record {parent_record!r}")
    for where, value in (("run.warm_start_log_std", run_block.get("warm_start_log_std")),
                         ("curriculum.warm_start_log_std", cur.get("warm_start_log_std"))):
        if value is None:
            problems.append(f"{where} is absent (not {WARM_START_LOG_STD})")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != WARM_START_LOG_STD:
            problems.append(f"{where} is {value!r}, not {WARM_START_LOG_STD}")
    n_envs = run_block.get("n_envs")
    if isinstance(n_envs, bool) or n_envs != N_ENVS:
        problems.append(f"run.n_envs is {n_envs!r}, not {N_ENVS}")
    if not isinstance(hyper, dict):
        problems.append(f"the recorded hyperparameters are not a table: {hyper!r}")
    elif "critic_phase" in hyper:
        problems.append(f"the recorded hyperparameters hold a critic_phase key ({hyper['critic_phase']!r})")
    return problems


def binding_problems(sc: dict, *, seed: int, want_parent: str, expect_budget: int, ramp: int, warmup: int,
                     hp_here: str, parent_record: dict | None, report_off_schedule: bool) -> tuple[list[str], dict]:
    """Every stage_config.json check that needs no tree (pilot 1's and pilot 3's), and what they read.

    Returns (problems, info) with info = {"budget", "resume_step", "off_schedule", "resumed"}."""
    run_block = sc.get("run") or {}
    problems = []
    if sc.get("species") != "tyrannosaurus_rex" or sc.get("embodiment") != "anatomical":
        problems.append(f"model {sc.get('species')}/{sc.get('embodiment')}")
    if sc.get("stage") not in (2, "2", "locomotion"):
        problems.append(f"stage {sc.get('stage')!r}")
    if run_block.get("load_mode") != "initialize_next_stage":
        problems.append(f"load_mode {run_block.get('load_mode')!r}")
    if run_block.get("seed") != seed:
        problems.append(f"seed {run_block.get('seed')!r} (not --seed {seed})")
    if want_parent != PILOT_PARENT_SHA256:
        problems.append(f"--parent-sha256 sha256:{want_parent} is not walking pilot 3's parent "
                        f"sha256:{PILOT_PARENT_SHA256}")
    if norm_sha(run_block.get("parent_checkpoint_sha256")) != want_parent:
        problems.append(f"parent_checkpoint_sha256 {run_block.get('parent_checkpoint_sha256')} is not the chosen parent "
                        f"sha256:{want_parent}")
    budget = run_block.get("timesteps")
    resumed = run_block.get("resume_load_path")
    resume_step, off_schedule = resume_check(run_block, ramp, warmup)
    if resumed:  # the RESUME cell records the segment it trained; the budget is that plus the checkpoint it resumed
        budget = (budget or 0) + resume_step if resume_step is not None else None
    if budget != expect_budget:
        problems.append(f"budget {budget!r} (not {expect_budget})" + (f"; resumed from {resumed}" if resumed else ""))
    if off_schedule and not report_off_schedule:
        problems.append(off_schedule)
    if run_block.get("hyperparameters_sha256") != hp_here:
        problems.append(f"hyperparameters_sha256 {run_block.get('hyperparameters_sha256')} != this tree's {hp_here}")
    problems += warm_start_problems(sc, parent_record)
    return problems, {"budget": budget, "resume_step": resume_step, "off_schedule": off_schedule,
                      "resumed": bool(resumed)}


def final_fallback_problems(models: Path, *, steps: int, expect_budget: int, rollout: int, resumed: bool,
                            pair_problem) -> tuple[list[str], dict]:
    """Whether the final pair may stand in for the deciding checkpoint (--final-fallback). pair_problem is the tree's
    checkpoint_pair_problem. Returns (problems, record)."""
    problems = []
    end = uninterrupted_end(expect_budget, rollout)
    deciding = (models / f"stage2_{steps}_steps.zip", models / f"stage2_vecnormalize_{steps}_steps.pkl")
    if not deciding[0].exists():
        deciding_state = f"missing ({deciding[0].name})"
    else:
        deciding_state = pair_problem(*deciding)
    record = {"deciding_pair": [str(p) for p in deciding], "deciding_pair_problem": deciding_state,
              "uninterrupted_end": end, "rollout": rollout, "final_num_timesteps": None}
    if steps != expect_budget:
        problems.append(f"--final-fallback stands in only for the deciding checkpoint at the budget {expect_budget:,}, "
                        f"not --steps {steps:,}")
    if deciding_state is None:
        problems.append(f"the deciding pair {deciding[0].name} is intact: judge it, not the final pair")
    if resumed:
        problems.append("a resumed run's final pair does not end at the uninterrupted end; no fallback for it")
    final = models / FINAL_ZIP
    if final.is_file():
        try:
            record["final_num_timesteps"] = int(read_zip_data(final)["num_timesteps"])
        except (KeyError, ValueError, TypeError, zipfile.BadZipFile) as exc:
            problems.append(f"{FINAL_ZIP} has no readable num_timesteps ({type(exc).__name__}: {exc})")
    got = record["final_num_timesteps"]
    if got is not None and got != end:
        problems.append(f"{FINAL_ZIP} records num_timesteps {got:,}, not the uninterrupted end {end:,} of the "
                        f"{expect_budget:,}-step budget (whole rollouts of {rollout:,})")
    return problems, record


def other_run_cells(out: Path, seed: int, run_name: str) -> list[str]:
    """FIX2: the cells under OUT/s<seed>/ judged from a run copy whose directory name is not run_name ("<steps>
    (<run>)"); a cell whose binding.json cannot be read is listed too. Unfinished .partial folders are skipped."""
    found = []
    for b in sorted((out / f"s{seed}").glob("*/binding.json")):
        if b.parent.name.endswith(".partial"):
            continue
        try:
            name = Path(str(json.loads(b.read_text()).get("run") or "")).name
        except (OSError, ValueError, AttributeError) as exc:
            found.append(f"{b.parent.name} (unreadable binding.json: {type(exc).__name__})")
            continue
        if name != run_name:
            found.append(f"{b.parent.name} ({name or 'no run recorded'})")
    return found


def commit_problem(recorded: object, head: str) -> str | None:
    """FIX2: None when stage_config.json's git_commit is 7 to 40 hex digits that begin the tree's HEAD; else why."""
    if recorded is None or str(recorded).strip() == "":
        return "stage_config.json records no git_commit"
    rec = str(recorded).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", rec) or not head.lower().startswith(rec):
        return f"git_commit {recorded} != tree HEAD {head}"
    return None


def parent_phase_record(tree: Path) -> tuple[dict | None, str | None]:
    """(the critic-phase record of the tree's committed parent, or None; the reason when None)."""
    path = tree / PARENT_IN_TREE
    if not path.is_file():
        return None, f"the tree holds no parent at {PARENT_IN_TREE}"
    got = norm_sha(sha(path))
    if got != PILOT_PARENT_SHA256:
        return None, f"the tree's {PARENT_IN_TREE} is sha256:{got}, not the pilot's parent sha256:{PILOT_PARENT_SHA256}"
    try:
        record = read_zip_data(path).get(CRITIC_PHASE_ATTRIBUTE)
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        return None, f"the tree's parent has no readable data member ({type(exc).__name__}: {exc})"
    if not isinstance(record, dict):
        return None, f"the tree's parent records no {CRITIC_PHASE_ATTRIBUTE}"
    return record, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent-sha256", required=True,
                    help="sha256 of the pilot's parent checkpoint: the PC round's stance7_r10LTSBpc_s42 final, "
                         f"sha256:{PILOT_PARENT_SHA256}")
    ap.add_argument("--expect-head", default=None)
    ap.add_argument("--allow-commit-mismatch", action="store_true")
    ap.add_argument("--report-off-schedule", action="store_true",
                    help="judge a run resumed before ramp_timesteps anyway; binding.json marks it and the scorer skips it")
    ap.add_argument("--expect-budget", type=int, default=PILOT_BUDGET,
                    help="validation only: accept a run trained to another budget (pilot3_score.py then refuses the cell)")
    ap.add_argument("--final-fallback", action="store_true",
                    help="judge the final pair in place of a missing or broken deciding pair (criteria: final-model "
                         "fallback)")
    ap.add_argument("--check-only", action="store_true", help="run every check, then stop before the probe; write nothing")
    a = ap.parse_args()
    want_parent = norm_sha(a.parent_sha256)
    if want_parent is None:
        return refuse(f"--parent-sha256 {a.parent_sha256!r} is not a sha256")

    tree = Path.cwd().resolve()
    head = git(tree, "rev-parse", "HEAD")
    branch = git(tree, "branch", "--show-current")
    dirty = git(tree, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        return refuse(f"the tree {tree} has local edits")
    if a.expect_head and not head.startswith(a.expect_head):
        return refuse(f"tree HEAD {head} is not {a.expect_head}")
    if not branch.startswith(BRANCH_PREFIX):
        return refuse(f"tree branch {branch!r} is not a walking-pilot branch")

    import environments
    from environments.shared.config import hyperparameters_sha256, load_stage_config
    from environments.shared.curriculum import checkpoint_pair_problem
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.species_names import require_model_id

    if not str(Path(environments.__file__).resolve()).startswith(str(tree) + "/"):
        return refuse(f"environments imported from {environments.__file__}, not from {tree}")
    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    run = Path(a.run).resolve()
    stage_dir = run / "03_locomotion"
    models = stage_dir / "models"
    if a.final_fallback:
        ckpt, vec = models / FINAL_ZIP, models / FINAL_VEC
    else:
        ckpt = models / f"stage2_{a.steps}_steps.zip"
        vec = models / f"stage2_vecnormalize_{a.steps}_steps.pkl"
    for p in (ckpt, vec, stage_dir / "stage_config.json", stage_dir / "plant_identity.json"):
        if not p.is_file():
            return refuse(f"checkpoint missing or incomplete run copy: {p}")
    pair_problem = checkpoint_pair_problem(ckpt, vec)
    if pair_problem is not None:
        return refuse(f"not an intact pair: {pair_problem}")
    sc = json.loads((stage_dir / "stage_config.json").read_text())
    run_block = sc.get("run", {})
    config = load_stage_config(mid, 2)
    cur = config.get("curriculum_kwargs", {})
    ramp, warmup = int(cur.get("ramp_timesteps", 0)), int(cur.get("warmup_timesteps", 0))
    rollout = N_ENVS * int(config.get("ppo_kwargs", {}).get("n_steps", 0))
    hp_here = hyperparameters_sha256(config, "ppo")
    parent_record, parent_note = parent_phase_record(tree)
    problems, info = binding_problems(sc, seed=a.seed, want_parent=want_parent, expect_budget=a.expect_budget,
                                      ramp=ramp, warmup=warmup, hp_here=hp_here, parent_record=parent_record,
                                      report_off_schedule=a.report_off_schedule)
    if parent_note:
        problems.append(parent_note)
    off_schedule = info["off_schedule"]
    fallback = None
    if a.final_fallback:
        fb_problems, fallback = final_fallback_problems(models, steps=a.steps, expect_budget=a.expect_budget,
                                                        rollout=rollout, resumed=info["resumed"],
                                                        pair_problem=checkpoint_pair_problem)
        problems += fb_problems
    plant = current_plant_identity(mid).to_dict()
    rec_plant = json.loads((stage_dir / "plant_identity.json").read_text())
    for k in ("physics_sha256", "policy_interface_sha256"):
        if rec_plant.get(k) != plant.get(k):
            problems.append(f"plant {k} {rec_plant.get(k)} != this tree's {plant.get(k)}")
    commit_note = None
    why_commit = commit_problem(sc.get("git_commit"), head)
    if why_commit:
        if a.allow_commit_mismatch:
            commit_note = f"{why_commit} (allowed)"
        else:
            problems.append(why_commit)
    if problems:
        return refuse("; ".join(problems))

    cell = cell_path(Path(a.out).resolve(), a.seed, a.steps, off_schedule, run)
    others = [] if off_schedule else other_run_cells(Path(a.out).resolve(), a.seed, run.name)
    if others:
        return refuse(f"OUT/s{a.seed} holds cells judged from another run than {run.name}: {', '.join(others)}. A "
                      f"restarted seed's retired-run cells are moved, never deleted, to OUT/retired/<retired RUN_ID>/"
                      f"s{a.seed}/<steps>/ before its _b run is judged (criteria, Run handling); this refusal concerns "
                      f"OUT, not the run")
    if cell.exists():
        return refuse(f"{cell} exists (one judgement per checkpoint, never overwritten)")
    if a.check_only:
        print(f"all checks passed (--check-only: nothing rolled or written); would judge {ckpt} into {cell}"
              + (f"  [{off_schedule}]" if off_schedule else "")
              + ("  [final-model fallback]" if a.final_fallback else ""), flush=True)
        return 0
    tmp = cell.with_name(cell.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    run_id = f"s{a.seed}_{a.steps}" + ("_final" if a.final_fallback else "")
    cmd = [sys.executable, "-I", str(HERE / "iso.py"), str(HERE / "gait_probe_pilot.py"), "--embodiment", "anatomical",
           "--", *PROBE_ARGS, "--checkpoint", str(ckpt), "--vecnorm", str(vec), "--run-id", run_id,
           "--out", str(tmp), "--stage-config-json", str(stage_dir / "stage_config.json")]
    env = dict(os.environ, ISO_WT=str(tree), MUJOCO_GL=os.environ.get("MUJOCO_GL", "disable"))
    print("running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=tree, env=env)
    if rc != 0:
        shutil.rmtree(tmp)
        return refuse(f"the gait probe exited {rc} (3 = checkpoint refused by the plant contract or loader)")
    probe_json = tmp / f"tyrannosaurus_rex_{run_id}_locomotion.json"
    shutil.copyfile(probe_json, tmp / "probe.json")
    binding = {"schema": "walking-pilot3-judge/v1", "tree": str(tree), "branch": branch, "head": head, "run": str(run),
               "seed": a.seed, "steps": a.steps, "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt),
               "vecnorm": str(vec), "vecnorm_sha256": sha(vec), "stage_config_run": run_block,
               "stage_config_git_commit": sc.get("git_commit"), "commit_note": commit_note,
               "parent_checkpoint_sha256": f"sha256:{want_parent}",
               "parent_critic_phase_record": parent_record,
               "warm_start": {"critic_phase_dropped": run_block.get("critic_phase_dropped"),
                              "run_warm_start_log_std": run_block.get("warm_start_log_std"),
                              "curriculum_warm_start_log_std": (sc.get("curriculum") or {}).get("warm_start_log_std")},
               "n_envs": run_block.get("n_envs"),
               "curriculum_cadence": {k: (sc.get("curriculum") or {}).get(k) for k in CADENCE_KEYS},
               "hyperparameters_sha256": hp_here, "plant": {k: plant.get(k) for k in ("physics_revision", "physics_sha256",
                                                                                     "policy_interface_sha256")},
               "ramp_timesteps": ramp, "warmup_timesteps": warmup, "resume_step": info["resume_step"],
               "off_schedule": bool(off_schedule), "off_schedule_reason": off_schedule,
               "final_fallback": bool(a.final_fallback), "final_fallback_record": fallback,
               "run_end": run_end(models, a.expect_budget, stage_dir, checkpoint_pair_problem),
               "probe_args": PROBE_ARGS, "expected_budget": a.expect_budget,
               "validation_only": a.expect_budget != PILOT_BUDGET}
    (tmp / "binding.json").write_text(json.dumps(binding, indent=1, default=str))
    tmp.rename(cell)
    print(f"wrote {cell}" + (f"  [{off_schedule}]" if off_schedule else "")
          + ("  [final-model fallback]" if a.final_fallback else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
