"""EARLY-BIAS PROBE (role early_bias): shared loaders.  Read only on every input; nothing here trains.

Builds, for one r10 stance6 F2 checkpoint, the objects one PPO update uses, the way train_base builds them:
  - the stage config from the sealed run-branch code copy (r10round/ecs/code), checked against the run's own
    stage_config.json (reward weights and [ppo] hyperparameters);
  - a DummyVecEnv of n_envs = 4 Monitor-wrapped stance envs, under VecNormalize with the checkpoint's own pickle,
    training = False (statistics FROZEN at the checkpoint), norm_reward = True (the training env's reward scaling);
  - the checkpoint model loaded by the code's own load_sb3_model with the training env and the stage's [ppo]
    kwargs (so the schedules are the run's), then enable_mirror_kl with the stage's SymmetryConfig and mirror map;
  - ent_coef and learning rate set to the values the policy probe recorded for the coming train() at this step.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
from pathlib import Path

import numpy as np

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
OUT = SCR / "com_revisit/notes/early_bias"
F2 = S1 / "foot/stance6/fetch/F2"
TAIL = 14          # tail_3_pitch_act
OBS_TAIL = 6       # qpos.tail_3_pitch in the observation


def copies(run: str) -> Path:
    return F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance"


def ckpt_paths(run: str, steps: int) -> tuple[Path, Path]:
    m = copies(run) / "models"
    return m / f"stage1_{steps}_steps.zip.copy1", m / f"stage1_vecnormalize_{steps}_steps.pkl.copy1"


def stage_config_json(run: str) -> dict:
    with open(copies(run) / "stage_config.json.copy1") as fh:
        return json.load(fh)


def probe(run: str) -> dict:
    with np.load(copies(run) / "policy_probe.npz.copy1", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def load_code_config():
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    cfg = copy.deepcopy(load_stage_config(mid, "stance"))
    return mid, cfg


def check_config(run: str, cfg: dict) -> list[str]:
    """Differences between the code copy's stage config and the run's recorded one (empty list = same)."""
    from environments.shared.model_id import ModelId
    from environments.shared.task_fingerprint import stage_task_fingerprint

    rec = stage_config_json(run)
    problems = []
    fp = stage_task_fingerprint(ModelId("tyrannosaurus_rex", "anatomical"), "stance", stage_config=cfg,
                                plant_identity=rec["plant_identity"])
    if fp.get("task_sha256") != rec["task_fingerprint"]["task_sha256"]:
        problems.append(f"task_sha256: code {fp.get('task_sha256')} run {rec['task_fingerprint']['task_sha256']}")
    env_kwargs = cfg["env_kwargs"]
    for k, v in rec["reward_weights"].items():
        if k not in env_kwargs:
            continue  # constructor defaults: covered by the task fingerprint above
        a, b = env_kwargs[k], v
        if isinstance(a, tuple):
            a = list(a)
        if a != b:
            problems.append(f"env kwarg {k}: code {a!r} run {b!r}")
    hp = rec["hyperparameters"]
    ppo = cfg["ppo_kwargs"]
    for k, v in hp.items():
        if k == "policy_kwargs":
            pk = ppo.get("policy_kwargs", {})
            if list(pk.get("net_arch", [])) != list(v["net_arch"]) or pk.get("log_std_init") != v["log_std_init"]:
                problems.append(f"policy_kwargs: code {pk!r} run {v!r}")
            continue
        if ppo.get(k) != v:
            problems.append(f"ppo {k}: code {ppo.get(k)!r} run {v!r}")
    return problems


class Built:
    pass


def build(run: str, steps: int, env_seed: int, n_envs: int = 4) -> Built:
    """The training objects at checkpoint (run, steps), envs seeded env_seed + rank (as make_env does)."""
    import torch
    from stable_baselines3.common.vec_env import VecNormalize

    from environments.shared.mirror_kl_ppo import MirrorKLPPO, enable_mirror_kl
    from environments.shared.plant_contract import PlantIdentity  # noqa: F401  (import check only)
    from environments.shared.policy_loading import load_sb3_model
    from environments.shared.species_registry import env_class
    from environments.shared.symmetry import MirrorMap, pop_symmetry_config
    from environments.shared.train_base import create_vec_env

    torch.set_num_threads(1)
    mid, cfg = load_code_config()
    problems = check_config(run, cfg)
    rec = stage_config_json(run)

    class _Species:
        model_id = mid
        env_class = None

    species = _Species()
    species.env_class = env_class(mid)
    stage_configs = {1: {"env_kwargs": dict(cfg["env_kwargs"])}}
    ppo = copy.deepcopy(cfg["ppo_kwargs"])
    sym = pop_symmetry_config(ppo)
    venv = create_vec_env(species, stage_configs, 1, n_envs, seed=env_seed, use_subproc=False, algorithm="ppo",
                          gamma=float(ppo["gamma"]))
    # the checkpoint's statistics, FROZEN: VecNormalize.load over the inner DummyVecEnv
    zpath, ppath = ckpt_paths(run, steps)
    inner = venv.venv
    vn = VecNormalize.load(str(ppath), inner)
    vn.training = False
    vn.norm_reward = True
    # alg kwargs exactly as _prepare_alg_kwargs builds them (schedules), minus logging
    from environments.shared.train_base import _prepare_alg_kwargs

    conf = {"ppo_kwargs": copy.deepcopy(cfg["ppo_kwargs"])}
    alg_kwargs, _tb, _gcs = _prepare_alg_kwargs(conf, "ppo", 0, OUT / "_tb_unused", use_tensorboard=False)
    pop_symmetry_config(alg_kwargs)
    alg_kwargs.pop("policy_kwargs", None)
    model = load_sb3_model(str(zpath), algorithm="ppo", env=vn, device="cpu", **alg_kwargs)
    plant = rec["plant_identity"]
    mirror_map = MirrorMap.load(mid, plant)
    model = enable_mirror_kl(model, sym, mirror_map)
    assert isinstance(model, MirrorKLPPO)
    b = Built()
    b.run, b.steps, b.env_seed = run, steps, env_seed
    b.model, b.vn, b.cfg, b.sym, b.mirror_map = model, vn, cfg, sym, mirror_map
    b.config_problems = problems
    b.rec = rec
    return b


def set_probe_schedule(b: Built, pr: dict) -> dict:
    """ent_coef and lr the probe recorded for the coming train() at this step; returns what it set."""
    idx = np.flatnonzero(pr["update_timesteps"] == b.steps)
    assert len(idx) == 1, (b.steps, idx)
    i = int(idx[0])
    ent = float(pr["ent_coef"][i])
    lr = float(pr["learning_rate"][i])
    b.model.ent_coef = ent
    b.probe_row = i
    return {"probe_row": i, "ent_coef": ent, "learning_rate": lr}


def raw_obs_rms(ppath: Path):
    with open(ppath, "rb") as fh:
        v = pickle.load(fh)  # noqa: S301 - the run's own pinned pickle, read only
    return v
