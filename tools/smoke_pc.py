"""PC dry runs (ST/tools/dry): (A) the switch-ON configuration pins from stage_config.json files written by the
measuring copy's own config.save_stage_config, called as train_base.train calls it (no training); (B) a tiny smoke
run of the measuring copy's train_base.train with [ppo] critic_phase = true (at most 2 PPO updates: n_envs 2,
n_steps 256, 1024 steps); (C) a run folder bundled from (B) as v9_bundle.py does (provenance.json by
result_bundle.initialize_result_bundle as notebook cell 8 calls it, stage_config.json with the launch run block, the
final pair a copy of the last periodic pair, gate_verdict.json, the notebook's save_result_bundle), for the launch
check and C1-C8 dry runs.  Nothing here is study training; outputs stay in --out and are deleted after recording.
PC AMENDMENT (2026-11-01): --switch off (train, bundle) does the same with the switch OFF, as the switch-off arm's
notebook leaves it (CRITIC_PHASE False: no critic_phase key is set): (B) a stock-critic smoke for the switch-off arm's
C1-C8 dry run (f2dry.py --arm off), (C) a run folder whose stage_config.json records the switch OFF for VP3's POS-off
(the launch check reads configuration files only: a bundle of the switch-ON smoke's files with --switch off is
enough for VP3; f2dry's C8 needs the switch-OFF smoke).  Default --switch on: unchanged.

    PY -I -B smoke_pc.py pins  --copy COPY --out DIR
    PY -I -B smoke_pc.py train --copy COPY --out DIR [--switch on|off]
    PY -I -B smoke_pc.py bundle --copy COPY --source STAGE_DIR --out RUN_DIR --run-id RUN_ID --seed 42 --label L
                                [--switch on|off]
"""
from __future__ import annotations

import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy as _copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent / "pc"
spec = importlib.util.spec_from_file_location("r10round_common", TOOLS / "r10round_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def _import_copy(copy: Path):
    sys.path.insert(0, str(copy))
    import environments  # noqa: F401

    assert Path(environments.__file__).resolve().is_relative_to(copy.resolve()), environments.__file__


def cmd_pins(a) -> int:
    copy, out = Path(a.copy).resolve(), Path(a.out).resolve()
    _import_copy(copy)
    from environments.shared.config import hyperparameters_sha256, load_all_stages, save_stage_config
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import load_stage_manifest
    from environments.shared.task_fingerprint import stage_task_fingerprint

    model_id = require_model_id("tyrannosaurus_rex", "anatomical")
    species_cfg = get_species_config(model_id)
    stage = load_stage_manifest(model_id).resolve_behavior("stance").reference
    plant = current_plant_identity(model_id)
    pins_round = C.load_pins(Path(__file__).resolve().parent / "orig" / "laneB" / "pins_r10round.json")
    res = {}
    for name, on in (("off", False), ("on", True)):
        configs = _copy.deepcopy(load_all_stages(model_id))
        config = configs[stage]
        if on:   # notebook cell 9: STAGE_CONFIGS[node.reference].setdefault("ppo_kwargs", {})["critic_phase"] = True
            config.setdefault("ppo_kwargs", {})["critic_phase"] = True
        task_fp = stage_task_fingerprint(model_id, stage, stage_config=config, plant_identity=plant)
        d = out / name
        d.mkdir(parents=True, exist_ok=False)
        save_stage_config(d, stage, config, "PPO", extra={"seed": 42, "n_envs": 4, "timesteps": 12_000_000},
                          env_class=species_cfg.env_class, model_id=model_id, plant_identity=plant,
                          task_fingerprint=task_fp, load_path=None, load_mode=None, parent_run_id=None,
                          label="r10 LTSB phase-critic s42" if on else "r10 LTSB s42")
        sc = json.loads((d / "stage_config.json").read_text())
        res[name] = dict(
            file=str(d / "stage_config.json"), file_sha256=C.sha256_file(d / "stage_config.json"),
            hyperparameters_sha256=C.dig(sc, "run.hyperparameters_sha256"),
            hyperparameters_sha256_recomputed=hyperparameters_sha256(config, "PPO"),
            task_sha256=C.dig(sc, "task_fingerprint.task_sha256"),
            stage_config_view_sha256=C.view_sha256(sc),
            plant_identity_equals_round_pin=sc.get("plant_identity") == C.pin(pins_round, "plant_identity"),
            switch=dict(path=C.SWITCH_PATH_DEFAULT, value=C.dig(sc, C.SWITCH_PATH_DEFAULT)),
            ppo_table_keys=sorted((sc.get("hyperparameters") or {}).keys()))
    res["round_pins"] = {k: C.pin(pins_round, k) for k in ("hyperparameters_sha256", "task_sha256",
                                                            "stage_config_view_sha256")}
    res["off_equals_round"] = {k: res["off"][k] == res["round_pins"][k] for k in res["round_pins"]}
    res["on_vs_round"] = {k: res["on"][k] == res["round_pins"][k] for k in res["round_pins"]}
    print(json.dumps(res, indent=1))
    return 0


def cmd_train(a) -> int:
    copy, out = Path(a.copy).resolve(), Path(a.out).resolve()
    _import_copy(copy)
    from environments.shared import train_base
    from environments.shared.config import load_all_stages
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import get_species_config

    model_id = ModelId("tyrannosaurus_rex", "anatomical")
    configs = _copy.deepcopy(load_all_stages(model_id))
    configs[1]["ppo_kwargs"].update(n_steps=256, batch_size=128)
    if a.switch == "on":                                     # PC AMENDMENT: --switch off leaves the key unset (OFF)
        configs[1]["ppo_kwargs"]["critic_phase"] = True
    # the stage's own cadence keys override train()'s eval_freq / save_freq (train_base._stage_cadence): a ledger
    # and periodic pairs at 512 and 1024 steps, 2 episodes per evaluation (fast; the ledger's columns are the point)
    configs[1]["curriculum_kwargs"].update(eval_freq=512, checkpoint_freq=512, eval_episodes=2)
    out.mkdir(parents=True, exist_ok=False)
    model = train_base.train(get_species_config(model_id), configs, 1, 1024, n_envs=2, seed=42, eval_freq=1024,
                             save_freq=512, verbose=0, algorithm="ppo", output_dir=str(out / "run"),
                             use_tensorboard=False, task_load_mode="resume_same_stage", report_metrics=False,
                             save_on_interrupt=False)
    print(json.dumps(dict(num_timesteps=int(model.num_timesteps), n_updates=int(model._n_updates),
                          policy=type(model.policy).__name__, model=type(model).__name__,
                          buffer=type(model.rollout_buffer).__name__, buffer_obs_shape=list(model.rollout_buffer.obs_shape),
                          critic_phase=getattr(model, "mesozoic_critic_phase", None)), indent=1))
    return 0


RUN_FILES = ("eval_episodes.csv", "progress.json", "evaluations.npz", "gate_progress.npz", "policy_probe.npz",
             "train_episodes.csv", "diagnostics.npz")


def cmd_bundle(a) -> int:
    """v9_bundle.py's writers (the round's V9 (b)), with the critic-phase switch ON in the stage config."""
    copy, src, run_dir = Path(a.copy).resolve(), Path(a.source).resolve(), Path(a.out).resolve()
    if run_dir.exists():
        print(f"REFUSED: {run_dir} exists")
        return 2
    # The copy is a git archive (no .git): the writers read the commit from GITHUB_SHA (config.get_git_commit,
    # result_bundle.provenance._repository_state) = the copy's sealed commit; provenance's repository state is read
    # read-only from --repo-root (WT; GIT_OPTIONAL_LOCKS=0 so git status writes nothing), as Colab's clone would be.
    seal = json.loads((copy / ".r10m_seal.json").read_text())
    os.environ["GITHUB_SHA"] = seal["commit"]
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    repo_root = Path(a.repo_root).resolve()
    _import_copy(copy)
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
    stage_configs = _copy.deepcopy(load_all_stages(model_id))
    config = stage_configs[stage]
    if a.switch == "on":           # notebook cell 9 with CRITIC_PHASE = True; PC AMENDMENT: --switch off sets nothing
        config.setdefault("ppo_kwargs", {})["critic_phase"] = True
    seed = a.seed
    plant = current_plant_identity(model_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    roles = {"training": seed, "checkpoint_selection_evaluation": seed + 1000, "publication_evaluation": seed + 3000,
             "certification_panel": PUBLICATION_SEED_START}
    initialize_result_bundle(run_dir, model_id=model_id, algorithm="ppo", backend="stable-baselines3", seed=seed,
                             evaluation_seeds=[seed + 1000, seed + 3000], seed_roles=roles, evaluation_episodes=30,
                             parallel_envs=4, hardware="local", plant_identity=plant.to_dict(), run_id=a.run_id,
                             repository_root=repo_root)
    stage_dir = run_dir / stage_dirname(model_id, stage)
    (stage_dir / "models").mkdir(parents=True)
    task_fp = stage_task_fingerprint(model_id, stage, stage_config=config, plant_identity=plant)
    save_stage_config(stage_dir, stage, config, "PPO", extra={"seed": seed, "n_envs": 4, "timesteps": 12_000_000},
                      env_class=species_cfg.env_class, model_id=model_id, plant_identity=plant,
                      task_fingerprint=task_fp, load_path=None, load_mode=None, parent_run_id=None, label=a.label)
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
    cur = config["curriculum_kwargs"]
    write_gate_verdict(stage_dir, model_id=model_id, stage=stage, stage_id=target.id, gate_kind=cur.get("gate_kind"),
                       gate_schema_version=cur.get("gate_schema_version"), passed=False,
                       failures=["PC dry-run folder: no certificate measured"], task_sha256=task_fp.get("task_sha256"),
                       judged_by="smoke_pc.py (PC dry-run folder)", checkpoint=stage_dir / "models" / "stage1_final.zip",
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
              "publication_gate_passed": False, "gate_failures": ["PC dry-run folder: no certificate measured"],
              "plant_identity": plant.to_dict(), "model_path": str(stage_dir / "models" / "stage1_final"),
              "vecnorm_path": str(stage_dir / "models" / "stage1_final_vecnorm.pkl")}
    paths = save_result_bundle([result], stage_configs, model_id, "PPO", seed, run_dir, backend="stable-baselines3",
                               hardware="local", parallel_envs=4, evaluation_episodes=30,
                               evaluation_seeds=[seed + 1000, seed + 3000], seed_roles=roles,
                               plant_identity=plant.to_dict(), run_id=a.run_id, repository_root=repo_root,
                               target_deliverable=target.key, replicates=None)
    man = json.loads(Path(paths["artifact_manifest"]).read_text())
    print(json.dumps({"run_dir": str(run_dir), "copied": copied, "final_pair_from": last,
                      "manifest_status": man["status"], "manifest_entries": len(man["files"])}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for k in ("pins", "train", "bundle"):
        p = sub.add_parser(k)
        p.add_argument("--copy", required=True)
        p.add_argument("--out", required=True)
        if k in ("train", "bundle"):
            p.add_argument("--switch", choices=("on", "off"), default="on")          # PC AMENDMENT
        if k == "bundle":
            p.add_argument("--source", required=True)
            p.add_argument("--run-id", required=True)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--label", default="r10 LTSB phase-critic s42")
            p.add_argument("--repo-root", default="/home/user/work/pc_branch")
    a = ap.parse_args(argv)
    return {"pins": cmd_pins, "train": cmd_train, "bundle": cmd_bundle}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
