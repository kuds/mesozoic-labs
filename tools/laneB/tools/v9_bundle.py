"""T. rex r10 round: V9 (b)'s run folder and its bundle (criteria lean v5 VALIDATIONS, V9 (b)).  No training.

    PY -I -B v9_bundle.py --copy MEASURING_COPY --source STAGE_DIR --out RUN_DIR [--run-id stance6_r10LTSB_s60]
                          [--seed 60] [--label "r10 LTSB s60"]

One run folder written on the branch HEAD at test scale, from a run already made on the run branch (--source: a stage
folder its harness wrote, e.g. S2/r10round/runbranch/build/equiv/k3on/scr_B_head_60_k3default: the r10-column ledger,
progress.json, evaluations.npz, gate_progress.npz, policy_probe.npz, train_episodes.csv, diagnostics.npz,
k3_panel.csv/json and its periodic pairs, copied byte for byte) and the branch's own writers, imported from the
measuring copy (a git archive of the branch HEAD; nothing is run inside the clone):
  provenance.json        result_bundle.initialize_result_bundle, called as notebook cell 8 calls it;
  stage_config.json      config.save_stage_config, called as train_base.train calls it (lines 1855-1871): the run
                         block's seed, n_envs 4 and timesteps, the plant identity and the task fingerprint embedded;
  the final pair         stage1_final.zip / stage1_final_vecnorm.pkl: byte copies of the folder's last periodic pair
                         (no training; v5 V9 (b): "its final pair a copy of its last periodic pair; R1 its last periodic pair");
  gate_verdict.json      result_bundle.write_gate_verdict as reporting.stage_artifacts calls it (a failed gate: no
                         certificate is measured on a test folder);
  the bundle             reporting.bundles.save_result_bundle (what cell 15's save_run_bundle forwards to), with the
                         notebook's arguments: plant_identity.json, collected_results.csv and artifact_manifest.json.
Prints what it wrote; refuses an --out that exists.  Exit codes: 0 written; 2 refused.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r10round_common", TOOLS / "r10round_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

RUN_FILES = ("eval_episodes.csv", "progress.json", "evaluations.npz", "gate_progress.npz", "policy_probe.npz",
             "train_episodes.csv", "diagnostics.npz", "k3_panel.csv", "k3_panel.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="stance6_r10LTSB_s60")
    ap.add_argument("--seed", type=int, default=60)
    ap.add_argument("--label", default="r10 LTSB s60")
    a = ap.parse_args(argv)
    print(C.VERSION)
    copy, src, run_dir = Path(a.copy).resolve(), Path(a.source).resolve(), Path(a.out).resolve()
    if run_dir.exists():
        print(f"REFUSED: {run_dir} exists")
        return 2
    if str(run_dir).startswith(str(Path(C.BR).resolve())) or str(run_dir).startswith(str(copy)):
        print("REFUSED: the output lies in the clone or the measuring copy")
        return 2
    sys.path.insert(0, str(copy))
    from environments.shared.config import load_all_stages, save_stage_config
    from environments.shared.constants import PUBLICATION_SEED_START
    from environments.shared.curriculum.gate_schema import gate_config_view
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.reporting import save_result_bundle
    from environments.shared.result_bundle import initialize_result_bundle, write_gate_verdict
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import load_stage_manifest, stage_dirname
    from environments.shared.task_fingerprint import stage_task_fingerprint

    model_id = require_model_id("tyrannosaurus_rex", "anatomical")
    species_cfg = get_species_config(model_id)
    manifest = load_stage_manifest(model_id)
    target = manifest.resolve_behavior("stance")
    stage = target.reference
    stage_configs = load_all_stages(model_id)
    config = stage_configs[stage]
    seed = a.seed
    plant = current_plant_identity(model_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    # provenance.json: notebook cell 8
    roles = {"training": seed, "checkpoint_selection_evaluation": seed + 1000, "publication_evaluation": seed + 3000,
             "certification_panel": PUBLICATION_SEED_START}
    initialize_result_bundle(run_dir, model_id=model_id, algorithm="ppo", backend="stable-baselines3", seed=seed,
                             evaluation_seeds=[seed + 1000, seed + 3000], seed_roles=roles, evaluation_episodes=30,
                             parallel_envs=4, hardware="local", plant_identity=plant.to_dict(), run_id=a.run_id,
                             repository_root=copy)
    stage_dir = run_dir / stage_dirname(model_id, stage)
    (stage_dir / "models").mkdir(parents=True)
    # stage_config.json: train_base.train's call
    task_fp = stage_task_fingerprint(model_id, stage, stage_config=config, plant_identity=plant)
    save_stage_config(stage_dir, stage, config, "PPO", extra={"seed": seed, "n_envs": 4, "timesteps": 12_000_000},
                      env_class=species_cfg.env_class, model_id=model_id, plant_identity=plant,
                      task_fingerprint=task_fp, load_path=None, load_mode=None, parent_run_id=None, label=a.label)
    # the run's own files, byte for byte
    copied = []
    for name in RUN_FILES:
        if (src / name).is_file():
            shutil.copyfile(src / name, stage_dir / name)
            copied.append(name)
    steps = []
    for f in sorted((src / "models").iterdir()):
        m = re.fullmatch(r"stage1_(\d+)_steps\.zip", f.name)
        if m and (src / "models" / f"stage1_vecnormalize_{m.group(1)}_steps.pkl").is_file():
            steps.append(int(m.group(1)))
    for s in steps:
        for n in (f"stage1_{s}_steps.zip", f"stage1_vecnormalize_{s}_steps.pkl"):
            shutil.copyfile(src / "models" / n, stage_dir / "models" / n)
            copied.append("models/" + n)
    last = max(steps)
    shutil.copyfile(src / "models" / f"stage1_{last}_steps.zip", stage_dir / "models" / "stage1_final.zip")
    shutil.copyfile(src / "models" / f"stage1_vecnormalize_{last}_steps.pkl",
                    stage_dir / "models" / "stage1_final_vecnorm.pkl")
    # gate_verdict.json: a failed gate (no certificate is measured on a test folder)
    cur = config["curriculum_kwargs"]
    write_gate_verdict(stage_dir, model_id=model_id, stage=stage, stage_id=target.id, gate_kind=cur.get("gate_kind"),
                       gate_schema_version=cur.get("gate_schema_version"), passed=False,
                       failures=["V9 (b) test folder: no certificate measured"], task_sha256=task_fp.get("task_sha256"),
                       judged_by="v9_bundle.py (V9 (b) test folder)", checkpoint=stage_dir / "models" / "stage1_final.zip",
                       normalization=stage_dir / "models" / "stage1_final_vecnorm.pkl", gate_config=gate_config_view(cur))
    prog = json.loads((stage_dir / "progress.json").read_text())
    result = {"stage": stage, "name": config.get("name", ""), "description": config.get("description", ""),
              "timesteps": int(prog.get("timesteps") or last), "duration_seconds": float(prog.get("wall_clock_s") or 0.0),
              "mean_reward": 0.0, "std_reward": 0.0, "mean_episode_length": 0.0, "std_episode_length": 0.0,
              "mean_forward_vel": 0.0, "std_forward_vel": 0.0, "mean_distance_traveled": 0.0, "mean_success_rate": 0.0,
              "best_eval_reward": 0.0, "best_eval_std": 0.0, "best_eval_length": 0.0, "best_eval_timestep": 0,
              "best_model_reward": 0.0, "best_model_std_reward": 0.0, "best_model_length": 0.0,
              "best_model_std_length": 0.0, "best_model_fwd_vel": 0.0, "best_model_std_fwd_vel": 0.0,
              "best_model_distance": 0.0, "best_model_success_rate": 0.0, "gate_passed": False,
              "publication_gate_passed": False, "gate_failures": ["V9 (b) test folder: no certificate measured"],
              "plant_identity": plant.to_dict(), "model_path": str(stage_dir / "models" / "stage1_final"),
              "vecnorm_path": str(stage_dir / "models" / "stage1_final_vecnorm.pkl")}
    paths = save_result_bundle([result], stage_configs, model_id, "PPO", seed, run_dir, backend="stable-baselines3",
                               hardware="local", parallel_envs=4, evaluation_episodes=30,
                               evaluation_seeds=[seed + 1000, seed + 3000], seed_roles=roles,
                               plant_identity=plant.to_dict(), run_id=a.run_id, repository_root=copy,
                               target_deliverable=target.key, replicates=None)
    man = json.loads(Path(paths["artifact_manifest"]).read_text())
    print(json.dumps({"run_dir": str(run_dir), "copied": copied, "final_pair_from": last,
                      "bundle": {k: str(v) for k, v in paths.items()}, "manifest_status": man["status"],
                      "manifest_entries": len(man["files"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
