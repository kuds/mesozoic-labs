"""Validation input for V4 and V5 (never a run's checkpoint): a zero-action checkpoint pair built on the measuring copy
the way train_base builds a stage's model, with no learn() and no optimizer step.

    PY -I -B statue_ckpt.py COPY OUTDIR --identity {r10,r9,none} [--name stage1_final] [--seed 60]

The stage's model (MirrorKLPPO, the stance stage's PPO kwargs and symmetry, plant identity and task fingerprint
attached as train_base._create_or_load_model attaches them) on the training VecEnv (4 envs, the seed), then its
action_net weight and bias set to zero: the Gaussian mean is 0 for every observation, so the deterministic action is
0 on every actuator, the zero-action statue.  Saved as OUTDIR/<name>.zip with its VecNormalize statistics as
OUTDIR/<pkl name> (the notebook's pair names: stage1_final -> stage1_final_vecnorm.pkl, stage1_<N>_steps ->
stage1_vecnormalize_<N>_steps.pkl).
  --identity r10   the copy's own plant identity (V4: M1 on the statue);
  --identity r9    the r9 plant identity (REPL's judge record of stance5_LTSB_s70 final) on both artifacts: the
                   synthetic r9 record of V5;
  --identity none  no plant identity on either artifact (no recorded physics_revision).
Writes OUTDIR/statue_ckpt.json (what was built, the sha256s, the zero check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
VERSION = "statue_ckpt 1.0 (2026-10-26)"
R9_IDENTITY_SOURCE = ("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/"
                      "stance5/replication/judged/stance5_LTSB_s70/judge_final.json")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("copy")
    ap.add_argument("outdir")
    ap.add_argument("--identity", choices=("r10", "r9", "none"), required=True)
    ap.add_argument("--name", default="stage1_final")
    ap.add_argument("--seed", type=int, default=60)
    a = ap.parse_args()
    print(VERSION, flush=True)
    copy = os.path.realpath(a.copy)
    out = Path(a.outdir).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    if str(out).startswith(copy + "/"):
        raise SystemExit("the output may not lie inside the measuring copy")
    out.mkdir(parents=True)
    scratch = out / "_scratch"
    scratch.mkdir()
    sys.path.insert(0, copy)
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from environments.shared import train_base as tb
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.plant_contract.constants import MODEL_IDENTITY_ATTRIBUTE
    from environments.shared.policy_loading import _ensure_sb3
    from environments.shared.species_registry import get_species_config
    from environments.shared.symmetry import pop_symmetry_config
    from environments.shared.task_fingerprint import stage_task_fingerprint

    import environments
    assert os.path.realpath(environments.__file__).startswith(copy + "/"), environments.__file__
    mid = ModelId("tyrannosaurus_rex", "anatomical")
    cfg = load_stage_config(mid, 1)
    species = get_species_config(mid)
    ident = current_plant_identity(mid)
    fp = stage_task_fingerprint(mid, 1, stage_config=cfg, plant_identity=ident)
    sb3 = _ensure_sb3()
    gamma = cfg["ppo_kwargs"].get("gamma")
    env = tb.create_vec_env(species, {1: cfg}, 1, 4, a.seed, False, algorithm="ppo", gamma=gamma, plant_identity=ident)
    alg_kwargs, _, _ = tb._prepare_alg_kwargs(cfg, "ppo", 0, scratch, False)
    alg_kwargs.setdefault("seed", a.seed)
    symmetry = pop_symmetry_config(alg_kwargs)
    model = tb._create_or_load_model(sb3, "ppo", alg_kwargs, env, None, plant_identity=ident, task_fingerprint=fp,
                                     task_load_mode="resume_same_stage", symmetry=symmetry)
    before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    changed = sorted(k for k, v in model.policy.state_dict().items() if not torch.equal(v, before[k]))
    assert set(changed) <= {"action_net.bias", "action_net.weight"} and "action_net.weight" in changed, changed
    assert not model.policy.action_net.weight.any() and not model.policy.action_net.bias.any()
    if a.identity == "r9":
        r9 = json.loads(Path(R9_IDENTITY_SOURCE).read_text())["plant_identity"]
        assert r9["physics_revision"] == 9, r9
        setattr(model, MODEL_IDENTITY_ATTRIBUTE, dict(r9))
        setattr(env, MODEL_IDENTITY_ATTRIBUTE, dict(r9))
    elif a.identity == "none":
        for obj in (model, env):
            if hasattr(obj, MODEL_IDENTITY_ATTRIBUTE):
                delattr(obj, MODEL_IDENTITY_ATTRIBUTE)
    obs = env.reset()
    act, _ = model.predict(obs, deterministic=True)
    assert np.all(np.asarray(act) == 0.0), act
    if a.name == "stage1_final":
        pkl = "stage1_final_vecnorm.pkl"
    elif a.name.startswith("stage1_") and a.name.endswith("_steps"):
        pkl = f"stage1_vecnormalize_{a.name[len('stage1_'):-len('_steps')]}_steps.pkl"
    else:
        raise SystemExit(f"--name {a.name} is not a notebook pair name")
    model.save(out / f"{a.name}.zip")
    env.save(str(out / pkl))
    env.close()
    import shutil

    scratch_listing = sorted(str(p.relative_to(scratch)) for p in scratch.rglob("*"))
    shutil.rmtree(scratch)
    rec = {"tool": VERSION, "copy": copy, "identity": a.identity, "seed": a.seed, "zip": str(out / f"{a.name}.zip"),
           "scratch_log_dir_listing_removed": scratch_listing,
           "zip_sha256": sha(out / f"{a.name}.zip"), "pkl": str(out / pkl), "pkl_sha256": sha(out / pkl),
           "num_timesteps": int(model.num_timesteps), "model_class": type(model).__name__,
           "zeroed": changed, "deterministic_action_on_reset_obs": np.asarray(act).tolist(),
           "copy_plant_physics_revision": ident.physics_revision,
           "recorded_physics_revision": (getattr(model, MODEL_IDENTITY_ATTRIBUTE, None) or {}).get("physics_revision"),
           "no_learn_no_optimizer_step": True}
    (out / "statue_ckpt.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
