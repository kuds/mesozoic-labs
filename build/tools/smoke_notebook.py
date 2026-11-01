"""Smoke of the critic phase through the notebook's own entry point (notebooks/sb3_training.ipynb), switch ON.

usage (from the code tree): python -I -B smoke_notebook.py TREE OUTDIR

Executes the notebook's own cells, as test_compsognathus_longipes_training's notebook test does: the setup cell
("# Add repo root to path"), the configuration cell (SPECIES tyrannosaurus_rex, BEHAVIOR stance, SEED 42 and
CRITIC_PHASE True substituted into its text; N_ENVS 2, no Drive, no disconnect), the resolve cell ("EnvClass =
SPECIES_CFG.env_class", which writes [ppo] critic_phase = true into the chain's nodes) and the helpers cell
("def train_stage(").  The storage cell is not run (no Drive, no result bundle); PLANT_IDENTITY and EVALUATION_SEED
are set as it sets them.  Then, on the real stance stage with n_steps 512 (batch 128) and the cadences shortened so
the run writes a periodic pair, evaluates and keeps a best model:
  session 1: train_stage(stance, 1024 steps)  -- 1 PPO update, evaluate=True (the chain loop's call);
  session 2: train_stage(stance, 1024 more, load_path=<the periodic pair>, task_load_mode="resume_same_stage",
             evaluate=False)  -- the RESUME cell's call; 1 PPO update.
Two PPO updates in all.  Prints what it checks; the outputs are deleted by the caller after recording.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import zipfile
from pathlib import Path


def first_code_cell(notebook: dict, marker: str) -> str:
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and marker in source:
            return source
    raise LookupError(marker)


def substitute(source: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^{name} = .*$", re.MULTILINE)
    assert len(pattern.findall(source)) == 1, name
    return pattern.sub(f"{name} = {value}", source)


def main() -> int:
    tree, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    os.chdir(tree)
    sys.path.insert(0, str(tree))
    notebook = json.loads((tree / "notebooks/sb3_training.ipynb").read_text(encoding="utf-8"))
    ns: dict = {"IN_COLAB": False, "__name__": "smoke"}
    exec(compile(first_code_cell(notebook, "# Add repo root to path"), "cell_setup", "exec"), ns)
    config = first_code_cell(notebook, "# ===== SPECIES SELECTION =====")
    for name, value in (
        ("SPECIES", '"tyrannosaurus_rex"  # @param'),
        ("BEHAVIOR", '"stance"'),
        ("SEED", "42"),
        ("N_ENVS", "2"),
        ("USE_GOOGLE_DRIVE", "False"),
        ("AUTO_DISCONNECT", "False"),
        ("TRUNK_FROM", '""'),
        ("CRITIC_PHASE", "True"),
    ):
        config = substitute(config, name, value)
    exec(compile(config, "cell_configuration", "exec"), ns)
    assert ns["CRITIC_PHASE"] is True and ns["K3_PANEL"] is False
    exec(compile(first_code_cell(notebook, "EnvClass = SPECIES_CFG.env_class"), "cell_resolve", "exec"), ns)
    configs = ns["STAGE_CONFIGS"]
    (node,) = ns["CHAIN"]
    stage = node.reference
    assert configs[stage]["ppo_kwargs"]["critic_phase"] is True
    # The smoke's sizes (the rules: at most 2 PPO updates, small n_steps): everything else is the stage's.
    configs[stage]["ppo_kwargs"].update(n_steps=512, batch_size=128)
    configs[stage]["curriculum_kwargs"].update(eval_freq=1024, checkpoint_freq=1024)
    from environments.shared.plant_contract import current_plant_identity

    ns.update(PLANT_IDENTITY=current_plant_identity(ns["MODEL_ID"]), EVALUATION_SEED=ns["SEED"] + 3000)
    exec(compile(first_code_cell(notebook, "def train_stage("), "cell_helpers", "exec"), ns)
    run_dir = out / "run"
    resume_only = "--resume-only" in sys.argv[3:]
    from stable_baselines3.common.vec_env import VecNormalize

    from environments.shared.critic_phase import CHECKPOINT_ATTRIBUTE, CriticPhaseSpec
    from environments.shared.critic_phase_ppo import CriticPhaseActorCriticPolicy, CriticPhaseRolloutBuffer
    from environments.shared.curriculum import newest_intact_periodic_pair
    from environments.shared.stage_manifest import stage_dirname, stage_label

    spec = CriticPhaseSpec(200, 0.98)
    if resume_only:
        return session_two(ns, stage, run_dir, run_dir / stage_dirname(ns["MODEL_ID"], stage) / "models", spec,
                           newest_intact_periodic_pair, stage_label, CHECKPOINT_ATTRIBUTE)
    model, best, final, stage_dir, vecnorm, results = ns["train_stage"](
        stage, 1024, run_dir=run_dir, task_load_mode="resume_same_stage", evaluate=True
    )

    print(f"session 1: model {type(model).__name__} policy {type(model.policy).__name__} "
          f"buffer {type(model.rollout_buffer).__name__}{model.rollout_buffer.obs_shape} "
          f"num_timesteps {model.num_timesteps} n_updates {model._n_updates}")
    assert type(model.policy) is CriticPhaseActorCriticPolicy and model.policy.critic_phase == spec
    assert type(model.rollout_buffer) is CriticPhaseRolloutBuffer and model.rollout_buffer.obs_shape == (66,)
    models = stage_dir / "models"
    print("session 1 files:", sorted(p.name for p in models.iterdir()))
    recorded = json.loads((stage_dir / "stage_config.json").read_text())
    print("stage_config hyperparameters.critic_phase:", recorded["hyperparameters"].get("critic_phase"),
          "hyperparameters_sha256:", recorded["run"]["hyperparameters_sha256"])
    assert recorded["hyperparameters"]["critic_phase"] is True
    for pkl in sorted(models.glob("*vecnorm*.pkl")):
        payload = pkl.read_bytes()
        loaded = pickle.loads(payload)
        assert type(loaded) is VecNormalize and loaded.observation_space.shape == (63,), pkl
        assert loaded.obs_rms.mean.shape == (63,) and b"CriticPhase" not in payload, pkl
        print(f"sidecar {pkl.name}: stock VecNormalize, observation (63,), obs_rms {loaded.obs_rms.mean.shape}")
    for archive in sorted(models.glob("*.zip")):
        data = json.loads(zipfile.ZipFile(archive).read("data"))
        assert data[CHECKPOINT_ATTRIBUTE] == spec.record(), archive
        assert data["policy_kwargs"]["critic_phase_settle_steps"] == 200, archive
        print(f"archive {archive.name}: {CHECKPOINT_ATTRIBUTE} ok, observation {data['observation_space']['_shape']}")
    print("evaluation (stage_results keys):", sorted(results)[:12] if results else results)
    if results:
        for key in ("eval_mean_reward", "mean_reward", "final_mean_reward", "eval_episodes"):
            if key in results:
                print(f"  {key}: {results[key]}")

    return session_two(ns, stage, run_dir, models, spec, newest_intact_periodic_pair, stage_label, CHECKPOINT_ATTRIBUTE)


def session_two(ns, stage, run_dir, models, spec, newest_intact_periodic_pair, stage_label, CHECKPOINT_ATTRIBUTE):
    # The RESUME cell's resolution: the newest intact periodic pair, then its train_stage call.
    pair, skipped = newest_intact_periodic_pair(models, stage_label(stage))
    assert pair is not None and not skipped, (pair, skipped)
    steps, ckpt, vecnorm_ckpt = pair
    stem = str(ckpt)[: -len(".zip")] if str(ckpt).endswith(".zip") else str(ckpt)
    model2, _best2, final2, _dir2, _vecnorm2, _results2 = ns["train_stage"](
        stage,
        1024,
        load_path=stem,
        run_dir=run_dir,
        vecnorm_path=str(vecnorm_ckpt),
        task_load_mode="resume_same_stage",
        evaluate=False,
    )
    print(f"session 2 (resumed from {Path(ckpt).name} at {steps}): num_timesteps {model2.num_timesteps} "
          f"n_updates {model2._n_updates} policy {type(model2.policy).__name__} spec {model2.policy.critic_phase}")
    assert model2.num_timesteps == 2048 and model2.policy.critic_phase == spec
    assert getattr(model2, CHECKPOINT_ATTRIBUTE) == spec.record()
    print("session 2 files:", sorted(p.name for p in models.iterdir()))
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
