"""Validation fixture (never a run's records): an r10 run folder for judge_r10ltsb.py's full binding checks, written by
the measuring copy's own writer as train_base writes a stage's records at its start (config.save_stage_config with
the run block {seed, n_envs 4, timesteps 12,000,000}, the env class, the plant identity and the task fingerprint;
it also writes 01_stance/plant_identity.json and task_fingerprint.json), and a trained checkpoint pair of the branch
HEAD placed under 01_stance/models/ (copied, never moved).

    PY -I -B write_r10_records.py COPY RUN_DIR ZIP PKL SEED

git_commit: the copy is a git archive (no .git), so config.get_git_commit() falls back to GITHUB_SHA, which is set
to the copy's sealed commit (the commit the archive is).  Nothing is trained; no optimizer step.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
VERSION = "write_r10_records 1.0 (2026-10-26)"


def main() -> int:
    copy, run, z, p, seed = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
    print(VERSION, flush=True)
    seal = json.loads((Path(copy) / ".r10m_seal.json").read_text())
    if run.exists():
        raise SystemExit(f"{run} exists")
    if subprocess.run(["git", "-C", copy, "rev-parse", "HEAD"], capture_output=True).returncode == 0:
        raise SystemExit("the copy resolves to a git repository: get_git_commit would not read GITHUB_SHA")
    os.environ["GITHUB_SHA"] = seal["commit"]
    sys.path.insert(0, os.path.realpath(copy))
    from environments.shared.config import load_stage_config, save_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.species_registry import get_species_config
    from environments.shared.task_fingerprint import stage_task_fingerprint

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    cfg = load_stage_config(mid, 1)
    ident = current_plant_identity(mid)
    fp = stage_task_fingerprint(mid, 1, stage_config=cfg, plant_identity=ident)
    stage_dir = run / "01_stance"
    (stage_dir / "models").mkdir(parents=True)
    save_stage_config(stage_dir, 1, cfg, "PPO", extra={"seed": seed, "n_envs": 4, "timesteps": 12000000},
                      env_class=get_species_config(mid).env_class, model_id=mid, plant_identity=ident,
                      task_fingerprint=fp)
    shutil.copy2(z, stage_dir / "models" / Path(z).name)
    shutil.copy2(p, stage_dir / "models" / Path(p).name)
    sc = json.loads((stage_dir / "stage_config.json").read_text())
    print(json.dumps({"run": str(run), "git_commit": sc.get("git_commit"), "run_block": sc.get("run"),
                      "task_sha256": (sc.get("task_fingerprint") or {}).get("task_sha256"),
                      "physics_revision": (sc.get("plant_identity") or {}).get("physics_revision"),
                      "files": sorted(str(q.relative_to(run)) for q in run.rglob("*") if q.is_file())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
