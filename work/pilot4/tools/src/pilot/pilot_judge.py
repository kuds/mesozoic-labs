"""Judge ONE periodic checkpoint of a walking-pilot run: the gait probe's 40-episode floor-truth panel.

Scratch, never merged. Run with the pilot branch's clean tree as the working directory (the tree the run trained on:
claude/pilot-walk-r9, path A, the maintainer's decision (2) of 2026-10-17), isolated, e.g.
    cd TREE && ISO_WT=TREE PYTHONPATH=TBPKGS python3 iso.py pilot_judge.py RUN --seed 42 --steps 4194304
        --parent-sha256 sha256:70f3a4447794a01036c27bca5e0377a17db96d0e3d7bfb2d179e9ce94e6f353e --out OUT
RUN is a local copy of the pilot run directory (paths relative to the run, as on Drive):
    03_locomotion/models/stage2_<steps>_steps.zip   03_locomotion/models/stage2_vecnormalize_<steps>_steps.pkl
    03_locomotion/stage_config.json                 03_locomotion/plant_identity.json
    03_locomotion/models/stage2_final.zip or 03_locomotion/stage_summary.txt (optional: the run-end record below)
It writes OUT/s<seed>/<steps>/probe.json (the probe's JSON), the probe's trace and plots beside it, and binding.json;
pilot_score.py reads probe.json and binding.json. Nothing is written when a check refuses.
An off-schedule run judged with --report-off-schedule is written to OUT/off_schedule/<RUN's directory name>/s<seed>/
<steps>/ instead, never to OUT/s<seed>/<steps>: the seed's restart (a fresh "_b" RUN_ID, judged into OUT/s<seed>/) can
then never collide with it, and pilot_score.py lists it as reported, never scored.

Checks before rolling anything (each refusal names its reason):
  * the tree is clean and on a pilot branch head (--expect-head pins one);
  * stage_config.json records the locomotion stage (2) of tyrannosaurus_rex/anatomical, an initialize_next_stage load,
    the seed --seed, the pilot's budget 4,194,304, and a git_commit equal to the tree's HEAD (--allow-commit-mismatch
    relaxes only this one, and binding.json says so);
  * its run.parent_checkpoint_sha256 equals --parent-sha256: the committed r9 s60 checkpoint above, fixed by decision
    (2) (not pilot_parent.py's output, which on the failed screen names the r10 s42 checkpoint);
  * a resumed run (run.resume_load_path set: the notebook's RESUME cell) resumed at or after the stage's
    ramp_timesteps (read from this tree's config, 2,000,000; the first periodic checkpoint past it is 2,097,152).
    A RESUME attaches no stage-entry shaping (train_base._stage_entry_shaping_callbacks: never on resume_same_stage),
    so a resume before the ramp's end trains the rest of the ramp at the full forward weight 2.0 and skips any
    remaining warm-up: that run is OFF-SCHEDULE and refused. --report-off-schedule judges it anyway for the report and
    marks binding.json off_schedule, which pilot_score.py never scores. stage_config.json keeps only the LATEST
    resume, so an earlier resume is invisible here (the Colab steps forbid resuming below 2,097,152);
  * its hyperparameters_sha256 equals the one this tree derives for the stage, so the run trained the pilot config;
  * plant_identity.json's physics and policy-interface digests equal the tree's;
  * the checkpoint and sidecar exist; the probe's own loader then enforces the plant contract (exit 3 = refused).
Run-end record (binding.json run_end, read by pilot_score.py for a seed without a 4,194,304 checkpoint): the newest
periodic checkpoint in the run copy, and the steps the run ended at, from stage2_final.zip's num_timesteps (else
stage_summary.txt's "Timesteps:"). A final model short of the budget means the collapse backstop stopped the run
(EvalCollapseEarlyStop is the only callback that ends the manual node early; a disconnect writes no final model).
Probe arguments are fixed: --stage locomotion --episodes 40 --seed-scheme panel --seed-base 3042 --settle-steps 100.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT_BUDGET = 4_194_304
PROBE_ARGS = ["--species", "tyrannosaurus_rex", "--stage", "locomotion", "--episodes", "40", "--seed-scheme", "panel",
              "--seed-base", "3042", "--settle-steps", "100"]
PERIODIC_RE = re.compile(r"stage2_(\d+)_steps(\.zip)?$")


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


def resume_check(run_block: dict, ramp_timesteps: int, warmup_timesteps: int) -> tuple[int | None, str | None]:
    """(resume step or None, off-schedule reason or None) for a stage_config run block."""
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
    off-schedule run judged with --report-off-schedule (so a restart's judgement of the same seed and step never
    collides with it, and the scorer never reads it as the seed's)."""
    if off_schedule:
        return out / "off_schedule" / run.name / f"s{seed}" / str(steps)
    return out / f"s{seed}" / str(steps)


def run_end(models: Path, budget: int | None, stage_dir: Path) -> dict:
    steps = sorted(int(m.group(1)) for p in models.glob("stage2_*_steps.zip") if (m := PERIODIC_RE.match(p.name)))
    final = models / "stage2_final.zip"
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
            "final_num_timesteps": end, "final_source": source, "budget": budget,
            "stopped_early": None if end is None or budget is None else end < budget}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent-sha256", required=True,
                    help="sha256 of the pilot's parent checkpoint: the committed r9 s60 one (criteria.txt, decision 2)")
    ap.add_argument("--expect-head", default=None)
    ap.add_argument("--allow-commit-mismatch", action="store_true")
    ap.add_argument("--report-off-schedule", action="store_true",
                    help="judge a run resumed before ramp_timesteps anyway; binding.json marks it and the scorer skips it")
    ap.add_argument("--expect-budget", type=int, default=PILOT_BUDGET,
                    help="validation only: accept a run trained to another budget (pilot_score.py then refuses the cell)")
    a = ap.parse_args()
    want_parent = norm_sha(a.parent_sha256)
    if want_parent is None:
        return refuse(f"--parent-sha256 {a.parent_sha256!r} is not a sha256")

    tree = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=tree, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=tree, text=True).strip()
    if dirty:
        return refuse(f"the tree {tree} has local edits")
    if a.expect_head and not head.startswith(a.expect_head):
        return refuse(f"tree HEAD {head} is not {a.expect_head}")
    if not branch.startswith("claude/pilot-walk-"):
        return refuse(f"tree branch {branch!r} is not a walking-pilot branch")

    import environments
    from environments.shared.config import hyperparameters_sha256, load_stage_config
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.species_names import require_model_id

    if not str(Path(environments.__file__).resolve()).startswith(str(tree) + "/"):
        return refuse(f"environments imported from {environments.__file__}, not from {tree}")
    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    run = Path(a.run).resolve()
    stage_dir = run / "03_locomotion"
    ckpt = stage_dir / "models" / f"stage2_{a.steps}_steps.zip"
    vec = stage_dir / "models" / f"stage2_vecnormalize_{a.steps}_steps.pkl"
    for p in (ckpt, vec, stage_dir / "stage_config.json", stage_dir / "plant_identity.json"):
        if not p.is_file():
            return refuse(f"checkpoint missing or incomplete run copy: {p}")
    sc = json.loads((stage_dir / "stage_config.json").read_text())
    run_block = sc.get("run", {})
    config = load_stage_config(mid, 2)
    cur = config.get("curriculum_kwargs", {})
    ramp, warmup = int(cur.get("ramp_timesteps", 0)), int(cur.get("warmup_timesteps", 0))
    problems = []
    if sc.get("species") != "tyrannosaurus_rex" or sc.get("embodiment") != "anatomical":
        problems.append(f"model {sc.get('species')}/{sc.get('embodiment')}")
    if sc.get("stage") not in (2, "2", "locomotion"):
        problems.append(f"stage {sc.get('stage')!r}")
    if run_block.get("load_mode") != "initialize_next_stage":
        problems.append(f"load_mode {run_block.get('load_mode')!r}")
    if run_block.get("seed") != a.seed:
        problems.append(f"seed {run_block.get('seed')!r} (not --seed {a.seed})")
    if norm_sha(run_block.get("parent_checkpoint_sha256")) != want_parent:
        problems.append(f"parent_checkpoint_sha256 {run_block.get('parent_checkpoint_sha256')} is not the chosen parent "
                        f"sha256:{want_parent}")
    budget = run_block.get("timesteps")
    resumed = run_block.get("resume_load_path")
    resume_step, off_schedule = resume_check(run_block, ramp, warmup)
    if resumed:  # the RESUME cell records the segment it trained; the budget is that plus the checkpoint it resumed
        budget = (budget or 0) + resume_step if resume_step is not None else None
    if budget != a.expect_budget:
        problems.append(f"budget {budget!r} (not {a.expect_budget})" + (f"; resumed from {resumed}" if resumed else ""))
    if off_schedule and not a.report_off_schedule:
        problems.append(off_schedule)
    hp_here = hyperparameters_sha256(config, "ppo")
    if run_block.get("hyperparameters_sha256") != hp_here:
        problems.append(f"hyperparameters_sha256 {run_block.get('hyperparameters_sha256')} != this tree's {hp_here}")
    plant = current_plant_identity(mid).to_dict()
    rec_plant = json.loads((stage_dir / "plant_identity.json").read_text())
    for k in ("physics_sha256", "policy_interface_sha256"):
        if rec_plant.get(k) != plant.get(k):
            problems.append(f"plant {k} {rec_plant.get(k)} != this tree's {plant.get(k)}")
    commit_note = None
    if sc.get("git_commit") and not head.startswith(str(sc.get("git_commit"))) and not str(sc.get("git_commit")).startswith(head[:12]):
        if a.allow_commit_mismatch:
            commit_note = f"run recorded git_commit {sc.get('git_commit')}, tree HEAD {head} (allowed)"
        else:
            problems.append(f"git_commit {sc.get('git_commit')} != tree HEAD {head}")
    if problems:
        return refuse("; ".join(problems))

    cell = cell_path(Path(a.out).resolve(), a.seed, a.steps, off_schedule, run)
    if cell.exists():
        return refuse(f"{cell} exists (one judgement per checkpoint, never overwritten)")
    tmp = cell.with_name(cell.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cmd = [sys.executable, "-I", str(HERE / "iso.py"), str(HERE / "gait_probe_pilot.py"), "--embodiment", "anatomical",
           "--", *PROBE_ARGS, "--checkpoint", str(ckpt), "--vecnorm", str(vec), "--run-id", f"s{a.seed}_{a.steps}",
           "--out", str(tmp), "--stage-config-json", str(stage_dir / "stage_config.json")]
    env = dict(os.environ, ISO_WT=str(tree), MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    print("running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=tree, env=env)
    if rc != 0:
        shutil.rmtree(tmp)
        return refuse(f"the gait probe exited {rc} (3 = checkpoint refused by the plant contract or loader)")
    probe_json = tmp / f"tyrannosaurus_rex_s{a.seed}_{a.steps}_locomotion.json"
    shutil.copyfile(probe_json, tmp / "probe.json")
    binding = {"schema": "walking-pilot-judge/v2", "tree": str(tree), "branch": branch, "head": head, "run": str(run),
               "seed": a.seed, "steps": a.steps, "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt),
               "vecnorm": str(vec), "vecnorm_sha256": sha(vec), "stage_config_run": run_block,
               "stage_config_git_commit": sc.get("git_commit"), "commit_note": commit_note,
               "parent_checkpoint_sha256": f"sha256:{want_parent}",
               "hyperparameters_sha256": hp_here, "plant": {k: plant.get(k) for k in ("physics_revision", "physics_sha256",
                                                                                     "policy_interface_sha256")},
               "ramp_timesteps": ramp, "warmup_timesteps": warmup, "resume_step": resume_step,
               "off_schedule": bool(off_schedule), "off_schedule_reason": off_schedule,
               "run_end": run_end(stage_dir / "models", a.expect_budget, stage_dir),
               "probe_args": PROBE_ARGS, "expected_budget": a.expect_budget,
               "validation_only": a.expect_budget != PILOT_BUDGET}
    (tmp / "binding.json").write_text(json.dumps(binding, indent=1, default=str))
    tmp.rename(cell)
    print(f"wrote {cell}" + (f"  [{off_schedule}]" if off_schedule else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
