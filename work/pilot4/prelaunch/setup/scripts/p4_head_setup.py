"""Walking pilot 4's setup/loading check for ONE seed, CPU only, NO LEARNING.

PRELAUNCH/SETUP ADAPTATION (this file; the postfix/code harness p4_setup_check.py, sha256 fe06521a..., copied):
  * the tree is this lane's clone at the head 0e6a57b (C/clones/setup), the pairs this lane's byte copies;
  * step 1-2 run through the PINNED pilot-4 smoke runner (TOOLS/nb_pilot_runner4.py, sha256 33fd52cc... checked
    before import) with its own --stop-at-learn: its learn() stops and training guards are the ones installed; the
    only addition (as the fix lane's edge_pairs.py) wraps the runner's install_learn_stops so that each stopped
    learn() first records the model, its arguments and the at-learn snapshot, then the runner's own stop raises;
  * the params are the Colab steps' (the Drive paths of each seed's 8M pair, mapped by the runner's path_map onto
    this lane's byte copies; QUICK_TEST False; expect_head 0e6a57b; the PPO digest fbc94bef...);
  * a new read-only section "judge_binding": the stage folder's records copied, the edge-prepared model written once
    with SB3's own save() (no update) to read its lineage the way the judge reads it (the zip's JSON data member),
    and the PINNED judge's own check functions (TOOLS/pilot4_judge.py, sha256 648dd557..., imported from its file)
    applied to the recorded stage_config.json, plant_identity.json and that lineage at the pilot budget.
Everything else (the analyses and their order) is the postfix/code harness unchanged.

Usage (cwd = the clone):
    cd TREE && ISO_WT=TREE PYTHONPATH=C/scripts python -I TOOLS/iso.py C/scripts/p4_head_setup.py SEED

1. Installs, in this process only, a stop on every learn() of SB3's algorithm classes (and the repository's
   MirrorKLPPO): the first call records the model, its arguments and its state at that instant, then raises
   LearnBlocked (a BaseException, so no `except Exception` on the way can swallow it) before any rollout or update.
   Guards also refuse PPO/SAC train(), collect_rollouts(), the torch optimizers' step() and backward(); each refusal
   is counted.
2. Runs the notebook's own cells through a copy of walking pilot 3's runner (nb_pilot_runner3.py, sha256-pinned):
   setup, configuration (SPECIES tyrannosaurus_rex, BEHAVIOR walk, SEED = the seed, N_ENVS 4, QUICK_TEST False,
   TRUNK_FROM "", CRITIC_PHASE False), storage, resolve, preflights, helpers, RESUME (skipped), the chain loop stopped
   by its guard line, then the MANUAL cell with MANUAL_NODE "locomotion", MANUAL_LOAD_MODE "initialize_next_stage" and
   the seed's 8M pair (byte copies in CHECK/pairs) as MANUAL_LOAD_PATH / MANUAL_VECNORM_PATH.  The manual cell calls
   the notebook's train_stage -> train_base.train -> _train_stage_body, which reaches model.learn(): the stop.
3. Read-only analyses of what the setup built (weights, Adam moments, VecNormalize, callbacks, stage_config.json),
   the start-of-training actions against pilot 3's walker, a 1,000-step deterministic rollout in the built env with a
   command-free twin stepped on the same actions, and the inputs for the 31d02e42 code-identity replay.
Writes CHECK/runs/s<SEED>/ only (the run directory itself is under TREE/logs/, as the notebook puts it).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

from p4c_common import (
    C,
    COMMAND_KEYS,
    EPISODE_SEED,
    EXTRA_EPISODE_SEEDS,
    MAX_STEPS,
    PREY_KEY,
    PRICE_KEYS,
    heading_metrics,
    jsonable,
    my_pair,
    replay,
    run_pair,
    same,
    sha256,
)

SEED = int(sys.argv[1])
TREE = Path(C) / "clones" / "setup"
TOOLS = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools")
RUNNER_SHA256 = "33fd52ccef7692b56e77849f3bc8b885c21df5790b08d66b5e00ce36fc4952bd"
JUDGE_SHA256 = "648dd5570d0a6dc72f0792f392c95e93c0d024b8918be0e054ee36e05f3e08c0"
EXPECT_HEAD = "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8"
DRIVE = "/content/drive/MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo"
OUT = Path(C) / "runs" / f"s{SEED}"
OUT.mkdir(parents=True, exist_ok=True)
RESULT: dict = {"seed": SEED, "tree": str(TREE), "errors": {}}


def save() -> None:
    with open(OUT / "setup_check.json", "w") as f:
        json.dump(RESULT, f, indent=1, default=str)


import torch  # noqa: E402

torch.set_num_threads(1)


class LearnBlocked(BaseException):
    """learn() was called: the setup path is complete; nothing has rolled out or updated."""


class TrainingBlocked(BaseException):
    """A training primitive was called (must never happen in this check)."""


CAPTURE: dict = {}
GUARD_HITS: list = []


def _snapshot_at_learn(model, kwargs) -> dict:
    from environments.shared.curriculum.schedules import ENT_COEF_WARMUP_MARKER

    env = model.get_env()
    inner = getattr(env, "venv", env)

    def attr(name):
        return [jsonable(v) for v in inner.get_attr(name)]

    items = {
        "model_class": lambda: f"{type(model).__module__}.{type(model).__qualname__}",
        "learn_kwargs": lambda: {k: (repr(v) if k == "callback" else v) for k, v in kwargs.items()},
        "num_timesteps": lambda: int(model.num_timesteps),
        "ent_coef": lambda: float(model.ent_coef),
        "clip_range_type": lambda: type(model.clip_range).__name__,
        "clip_range(1.0)": lambda: float(model.clip_range(1.0)) if callable(model.clip_range) else float(model.clip_range),
        "lr_schedule(1.0)": lambda: float(model.lr_schedule(1.0)),
        "lr_schedule(0.0)": lambda: float(model.lr_schedule(0.0)),
        "optimizer_lr": lambda: [g["lr"] for g in model.policy.optimizer.param_groups],
        "ent_coef_warmup_marker": lambda: bool(getattr(model, ENT_COEF_WARMUP_MARKER, False)),
        "train_env_forward_vel_weight": lambda: attr("forward_vel_weight"),
        "train_env_walk_heading_price_weight": lambda: attr("walk_heading_price_weight"),
        "train_env_walk_heading_price_scale_rad": lambda: attr("walk_heading_price_scale_rad"),
        "train_env_command_mode": lambda: attr("command_mode"),
        "train_env_prey_lateral_range": lambda: attr("prey_lateral_range"),
        "vecnorm_training": lambda: bool(env.training),
        "vecnorm_obs_rms_count": lambda: float(env.obs_rms.count),
        "device": lambda: str(model.device),
        "n_envs": lambda: int(model.n_envs),
        "n_steps": lambda: int(model.n_steps),
        "batch_size": lambda: int(model.batch_size),
        "n_epochs": lambda: int(model.n_epochs),
        "gamma": lambda: float(model.gamma),
        "gae_lambda": lambda: float(model.gae_lambda),
        "target_kl": lambda: model.target_kl,
        "seed": lambda: model.seed,
    }
    snap = {}
    for key, fn in items.items():
        try:
            snap[key] = fn()
        except BaseException as exc:  # noqa: BLE001 - recorded per item
            snap[key] = f"unreadable: {type(exc).__name__}: {exc}"
    return snap


def _capture_wrapper(stop, cls_name):
    """prelaunch/setup: record the model and its at-learn state, then call the pinned runner's own stop (it raises)."""

    def learn(self, *args, **kwargs):
        if "model" not in CAPTURE:
            CAPTURE["model"] = self
            CAPTURE["args"] = args
            CAPTURE["kwargs"] = kwargs
            CAPTURE["patched_class"] = cls_name
            CAPTURE["stack"] = traceback.format_stack(limit=12)
            try:
                CAPTURE["at_learn"] = _snapshot_at_learn(self, kwargs)
            except BaseException as exc:  # noqa: BLE001 - recorded, then the stop
                CAPTURE["at_learn_error"] = f"{type(exc).__name__}: {exc}"
        return stop(self, *args, **kwargs)  # the pinned runner's stop: raises its LearnBlocked

    return learn


def params_for(seed: int) -> dict:
    """The Colab steps' params for one seed (the fix lane's make_params.py, this lane's path map)."""
    models = f"{DRIVE}/pilot3_walk_s{seed}/03_locomotion/models"
    return {
        "repo_ref": "claude/pilot-walk-r10-p4",
        "config": {
            "SPECIES": "tyrannosaurus_rex", "EMBODIMENT": "anatomical", "ALGORITHM": "ppo", "N_ENVS": 4,
            "SEED": seed, "QUICK_TEST": False, "USE_GOOGLE_DRIVE": False, "AUTO_DISCONNECT": False,
            "BEHAVIOR": "walk", "TRUNK_FROM": "", "RETRAIN_FROM": "", "RUN_LABEL": f"walking pilot 4 s{seed}",
            "RUN_ID": f"pilot4_walk_s{seed}", "K3_PANEL": False, "CRITIC_PHASE": False,
        },
        "manual": [
            'MANUAL_NODE = "locomotion"',
            f'MANUAL_LOAD_PATH = "{models}/stage2_8000000_steps"',
            f'MANUAL_VECNORM_PATH = "{models}/stage2_vecnormalize_8000000_steps.pkl"',
            'MANUAL_LOAD_MODE = "initialize_next_stage"',
            "MANUAL_TIMESTEPS = None",
        ],
        "path_map": [[DRIVE, str(Path(C) / "pairs")]],
        "guard_line": 'raise RuntimeError("Walking pilot 4: chain loop skipped on purpose. Now run the manual cell '
                      '(section 7) by itself -- unless you are resuming: then stop here.")',
        "resume_stage": "locomotion",
        "expect_head": EXPECT_HEAD,
        "expect_hyperparameters_sha256": "sha256:fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62",
    }


# ---------------------------------------------------------------------------------------------------------------
# 1-2. the pinned runner's stops (with the capture), then the notebook
# ---------------------------------------------------------------------------------------------------------------
import hashlib  # noqa: E402
import importlib.util  # noqa: E402

RESULT["pair_sha256_before"] = {"run_copy": [sha256(p) for p in run_pair(SEED)], "my_copy": [sha256(p) for p in my_pair(SEED)]}
_runner_path = TOOLS / "nb_pilot_runner4.py"
RESULT["runner_file"] = str(_runner_path)
RESULT["runner_sha256"] = hashlib.sha256(_runner_path.read_bytes()).hexdigest()
if RESULT["runner_sha256"] != RUNNER_SHA256:
    save()
    print(f"refusing: {_runner_path} is not the pinned runner ({RESULT['runner_sha256']})", flush=True)
    sys.exit(2)
_spec = importlib.util.spec_from_file_location("nb_pilot_runner4", _runner_path)
runner = importlib.util.module_from_spec(_spec)
sys.modules["nb_pilot_runner4"] = runner
_spec.loader.exec_module(runner)
GUARD_HITS = runner.GUARD_HITS  # the pinned runner's own hit list (every later read sees the runner's hits)
_install = runner.install_learn_stops


def install_with_capture() -> dict:
    rec = _install()  # the pinned runner's own learn() stops and training guards
    import stable_baselines3 as sb3
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

    from environments.shared.mirror_kl_ppo import MirrorKLPPO

    wrapped = []
    for cls in (BaseAlgorithm, OnPolicyAlgorithm, OffPolicyAlgorithm, sb3.PPO, sb3.SAC, sb3.A2C, MirrorKLPPO):
        if "learn" in cls.__dict__:
            setattr(cls, "learn", _capture_wrapper(cls.__dict__["learn"], f"{cls.__module__}.{cls.__qualname__}"))
            wrapped.append(f"{cls.__module__}.{cls.__qualname__}.learn")
    rec["prelaunch_capture_wraps"] = wrapped
    RESULT["stops"] = rec
    return rec


runner.install_learn_stops = install_with_capture
params = params_for(SEED)
(OUT / "params.json").write_text(json.dumps(params, indent=1))
rc = runner.main(["--params", str(OUT / "params.json"), "--out", str(OUT / "nb"), "--clone", str(TREE),
                  "--stop-at-learn"])
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
cells = runner.RECORD.get("cells", [])
manual_idx = (runner.RECORD.get("roles") or {}).get("manual")
manual = [c for c in cells if c.get("cell") == manual_idx]
RESULT["runner"] = {
    "exit": rc, "roles": runner.RECORD.get("roles"), "run_all": runner.RECORD.get("run_all"),
    "cells": cells, "clone_head": runner.RECORD.get("clone_head"), "run_dir": runner.RECORD.get("run_dir"),
    "manual_cell": manual[-1] if manual else None,
    "record": {k: runner.RECORD.get(k) for k in (
        "clone_head", "manual_cell_stopped_at_learn", "learn_called", "guard_hits", "learn_stops",
        "environments_resolves_to", "environments_imported_from", "entry_lines", "parent_pair",
        "locomotion_curriculum_at_end", "notebook_sha256", "exit", "refused", "expected_entry_lines",
        "manual_values", "clone_status_before", "tool_sha256", "run_dir")},
}
RESULT["learn_reached"] = "model" in CAPTURE
RESULT["manual_cell_stopped_by_learn_stop"] = bool(manual and manual[-1].get("exception") == "LearnBlocked")
RESULT["guard_hits"] = list(GUARD_HITS)
RESULT["capture"] = {k: CAPTURE.get(k) for k in ("patched_class", "at_learn", "at_learn_error")}
RESULT["capture"]["stack_tail"] = "".join(CAPTURE.get("stack", [])[-6:])
save()
if "model" not in CAPTURE or rc != 0:
    print(f"the set-up path did not end as expected (runner exit {rc}, learn reached {'model' in CAPTURE}); "
          f"see setup_check.json", flush=True)
    sys.exit(1)

import environments  # noqa: E402

RESULT["environments_imported_from"] = environments.__file__

from stable_baselines3 import PPO  # noqa: E402

from environments.shared.config import hyperparameters_sha256, load_all_stages, load_stage_config  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.policy_loading import load_sb3_model  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.task_fingerprint import (  # noqa: E402
    MODEL_TASK_ATTRIBUTE,
    MODEL_TASK_LINEAGE_ATTRIBUTE,
    stage_task_fingerprint,
)

MID = ModelId("tyrannosaurus_rex", "anatomical")
EnvClass = get_species_config(MID).env_class
model = CAPTURE["model"]
callback_list = CAPTURE["kwargs"].get("callback")
train_env = model.get_env()
zip_path, pkl_path = my_pair(SEED)
orig = load_sb3_model(zip_path, algorithm=PPO, device="cpu")
with open(pkl_path, "rb") as f:
    orig_vn = pickle.load(f)
config = load_stage_config(MID, 2)
RESULT["config_equals_load_all_stages"] = jsonable(config) == jsonable(load_all_stages(MID)[2])
P4_KW = dict(config["env_kwargs"])
TW_KW = {k: v for k, v in P4_KW.items() if k not in COMMAND_KEYS + PRICE_KEYS}
P3X_KW = {k: v for k, v in TW_KW.items() if k != PREY_KEY}
(OUT / "kwargs.json").write_text(json.dumps({"p4": jsonable(P4_KW), "twin": jsonable(TW_KW), "p3x": jsonable(P3X_KW)},
                                            indent=1))

stage_dir = Path(RESULT["runner"]["run_dir"]) / "03_locomotion"
RESULT["stage_dir"] = str(stage_dir)
RESULT["stage_dir_listing"] = sorted(str(p.relative_to(stage_dir)) for p in stage_dir.rglob("*") if p.is_file())


def section(name):
    def wrap(fn):
        try:
            RESULT[name] = fn()
        except BaseException as exc:  # noqa: BLE001 - recorded per section
            RESULT["errors"][name] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        save()
        return fn

    return wrap


# ---------------------------------------------------------------------------------------------------------------
# 3a. the recorded stage_config.json
# ---------------------------------------------------------------------------------------------------------------
@section("stage_config")
def _stage_config():
    sc = json.loads((stage_dir / "stage_config.json").read_text())
    p3 = json.loads(Path(run_pair(SEED)[0]).parent.parent.joinpath("stage_config.json").read_text())
    rw, rw3 = sc.get("reward_weights", {}), p3.get("reward_weights", {})
    cur, cur3 = sc.get("curriculum", {}), p3.get("curriculum", {})
    fp = sc.get("task_fingerprint", {})
    tree_fp = stage_task_fingerprint(MID, 2, stage_config=config)
    out = {
        "top_level_keys": sorted(sc),
        "git_commit": sc.get("git_commit"),
        "run": sc.get("run"),
        "task_fingerprint_task_sha256": fp.get("task_sha256") if isinstance(fp, dict) else fp,
        "task_fingerprint_has_command_section": isinstance(fp, dict) and "command" in fp,
        "task_fingerprint_command_config": (fp.get("command") or {}).get("config") if isinstance(fp, dict) else None,
        "tree_task_sha256": tree_fp.get("task_sha256"),
        "hyperparameters_sha256_PPO_tree": hyperparameters_sha256(config, "PPO"),
        "hyperparameters_sha256_SAC_tree": hyperparameters_sha256(config, "SAC"),
        "curriculum": cur,
        "curriculum_changed_vs_pilot3": {k: [cur3.get(k), cur.get(k)] for k in sorted(set(cur) | set(cur3))
                                         if not same(jsonable(cur3.get(k)), jsonable(cur.get(k)))},
        "reward_weights_changed_vs_pilot3": {k: [rw3.get(k), rw.get(k)] for k in sorted(set(rw) | set(rw3))
                                             if not same(jsonable(rw3.get(k)), jsonable(rw.get(k)))},
        "reward_weights_price_and_command": {k: rw.get(k) for k in COMMAND_KEYS + PRICE_KEYS + (PREY_KEY,)},
        "hyperparameters_changed_vs_pilot3": {
            k: [p3.get("hyperparameters", {}).get(k), sc.get("hyperparameters", {}).get(k)]
            for k in sorted(set(sc.get("hyperparameters", {})) | set(p3.get("hyperparameters", {})))
            if not same(jsonable(p3.get("hyperparameters", {}).get(k)), jsonable(sc.get("hyperparameters", {}).get(k)))},
        "run_changed_vs_pilot3": {k: [p3.get("run", {}).get(k), sc.get("run", {}).get(k)]
                                  for k in sorted(set(sc.get("run", {})) | set(p3.get("run", {})))
                                  if not same(jsonable(p3.get("run", {}).get(k)), jsonable(sc.get("run", {}).get(k)))},
        "pilot3_task_sha256": (p3.get("task_fingerprint") or {}).get("task_sha256"),
    }
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3b. weights, log_std, Adam moments, model attributes
# ---------------------------------------------------------------------------------------------------------------
@section("weights")
def _weights():
    o_sd, n_sd = orig.policy.state_dict(), model.policy.state_dict()
    changed = {}
    for name, old in o_sd.items():
        new = n_sd[name]
        diff = (new.double() - old.double()).abs()
        n = int((diff > 0).sum())
        if n:
            idx = torch.nonzero(diff > 0)
            cols = sorted(set(idx[:, 1].tolist())) if idx.shape[1] > 1 else None
            changed[name] = {"n_changed": n, "shape": list(old.shape), "changed_columns": cols,
                             "loaded_values_at_changed_max_abs": float(new.double()[diff > 0].abs().max())}
    first = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
    return {
        "n_tensors": len(o_sd), "tensors_changed": changed,
        "parent_command_column_l2": {n: [float(o_sd[n][:, c].norm()) for c in (60, 61, 62)] for n in first},
        "loaded_command_columns_max_abs": {n: float(n_sd[n][:, 60:63].abs().max()) for n in first},
        "columns_0_60_bit_equal": {n: bool(torch.equal(n_sd[n][:, :60], o_sd[n][:, :60])) for n in first},
        "log_std_bit_equal_parent": bool(torch.equal(model.policy.log_std.detach(), orig.policy.log_std.detach())),
        "log_std": {"mean": float(model.policy.log_std.mean()), "min": float(model.policy.log_std.min()),
                    "max": float(model.policy.log_std.max()), "values": model.policy.log_std.detach().tolist()},
    }


@section("optimizer")
def _optimizer():
    o_sd, n_sd = orig.policy.optimizer.state_dict(), model.policy.optimizer.state_dict()
    names = [n for n, _ in model.policy.named_parameters()]
    per, steps = {}, set()
    for idx, name in enumerate(names):
        so, sn = o_sd["state"].get(idx, {}), n_sd["state"].get(idx, {})
        e = {}
        for key in sorted(set(so) | set(sn)):
            a, b = so.get(key), sn.get(key)
            if a is None or b is None:
                e[key] = f"present parent={a is not None} loaded={b is not None}"
                continue
            if key == "step":
                e["step_parent"], e["step_loaded"] = float(a), float(b)
                steps.add((float(a), float(b)))
                continue
            d = (b.double() - a.double()).abs()
            e[f"{key}_n_changed"] = int((d > 0).sum())
            if d.ndim == 2 and int((d > 0).sum()):
                e[f"{key}_changed_columns"] = sorted(set(torch.nonzero(d > 0)[:, 1].tolist()))
            if name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
                e[f"{key}_parent_cmd_cols_max_abs"] = float(a[:, 60:63].abs().max())
                e[f"{key}_loaded_cmd_cols_max_abs"] = float(b[:, 60:63].abs().max())
        if any(v for k, v in e.items() if k.endswith("_n_changed")):
            per[name] = e
        elif name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight", "log_std"):
            per[name] = e
    return {"param_groups_lr_parent": [g["lr"] for g in o_sd["param_groups"]],
            "param_groups_lr_loaded": [g["lr"] for g in n_sd["param_groups"]],
            "step_pairs(parent,loaded)": sorted(steps), "params_with_changed_moments_or_listed": per,
            "n_params": len(names)}


@section("model_attributes")
def _model_attributes():
    task = getattr(model, MODEL_TASK_ATTRIBUTE, None)
    lineage = getattr(model, MODEL_TASK_LINEAGE_ATTRIBUTE, None)
    return {
        "num_timesteps_loaded": int(model.num_timesteps), "num_timesteps_parent": int(orig.num_timesteps),
        "attached_task_sha256": task.get("task_sha256") if isinstance(task, dict) else None,
        "parent_task_sha256": (getattr(orig, MODEL_TASK_ATTRIBUTE, None) or {}).get("task_sha256"),
        "lineage": jsonable(lineage),
        "parent_ent_coef_in_archive": float(orig.ent_coef),
        "critic_phase_attribute": getattr(model, "mesozoic_critic_phase", None),
        "symmetry_attribute": getattr(model, "mesozoic_symmetry", None),
    }



# ---------------------------------------------------------------------------------------------------------------
# prelaunch/setup: 3b'. what the judge binds, read from the records at learn() and checked by the pinned judge
# ---------------------------------------------------------------------------------------------------------------
ABSENT = "<absent>"


def _get(block, key):
    return block.get(key, ABSENT) if isinstance(block, dict) else ABSENT


@section("judge_binding")
def _judge_binding():
    import inspect
    import shutil
    import subprocess

    from environments.shared.plant_contract import current_plant_identity

    jpath = TOOLS / "pilot4_judge.py"
    jsha = hashlib.sha256(jpath.read_bytes()).hexdigest()
    if jsha != JUDGE_SHA256:
        raise RuntimeError(f"{jpath} is not the pinned judge ({jsha})")
    jspec = importlib.util.spec_from_file_location("pilot4_judge", jpath)
    judge = importlib.util.module_from_spec(jspec)
    jspec.loader.exec_module(judge)

    # the stage folder's records at learn(), copied byte for byte
    dst = OUT / "stage"
    dst.mkdir(exist_ok=True)
    copied = {}
    for p in sorted(stage_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(stage_dir)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)
            copied[str(rel)] = "sha256:" + sha256(str(p))
    sc = json.loads((stage_dir / "stage_config.json").read_text())
    rec_plant = json.loads((stage_dir / "plant_identity.json").read_text())

    # the lineage the way the judge reads it: the edge-prepared model written once by SB3's save() (no update),
    # then the zip's JSON data member read by the judge's own zip_lineage (json and zipfile only)
    zpath = OUT / "edge_model_at_learn.zip"
    model.save(str(zpath))
    lineage, lineage_note = judge.zip_lineage(zpath)
    data = judge.read_zip_data(zpath)
    in_memory = jsonable(getattr(model, MODEL_TASK_LINEAGE_ATTRIBUTE, None))

    # what judge.main derives from the tree, the same calls
    cur = config.get("curriculum_kwargs", {})
    env_kwargs = config.get("env_kwargs", {})
    ramp, warmup = int(cur.get("ramp_timesteps", 0)), int(cur.get("warmup_timesteps", 0))
    ramp_start, forward_weight = cur.get("ramp_start_value"), env_kwargs.get("forward_vel_weight")
    committed_table = env_kwargs.get("command_config")
    rollout = judge.N_ENVS * int(config.get("ppo_kwargs", {}).get("n_steps", 0))
    hp_here = hyperparameters_sha256(config, "ppo")
    task_here = stage_task_fingerprint(MID, 2, stage_config=config)["task_sha256"]
    defaults = {name: prm.default for name, prm in inspect.signature(EnvClass.__init__).parameters.items()
                if prm.default is not inspect.Parameter.empty}
    kwargs_drift = judge.env_kwargs_drift(sc.get("reward_weights"), dict(env_kwargs), defaults)
    want_parent = judge.PARENT_SHA256[SEED]
    problems, info = judge.binding_problems(
        sc, seed=SEED, want_parent=want_parent, expect_budget=judge.PILOT_BUDGET, ramp=ramp, warmup=warmup,
        ramp_start=ramp_start, forward_weight=forward_weight, hp_here=hp_here, task_here=task_here,
        committed_table=committed_table, lineage=lineage, lineage_note=lineage_note, report_off_schedule=False,
        kwargs_drift=kwargs_drift)
    run_name = Path(RESULT["runner"]["run_dir"]).name
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TREE, text=True,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()
    plant = current_plant_identity(MID).to_dict()
    plant_problems = [f"plant {k} {rec_plant.get(k)} != this tree's {plant.get(k)}"
                      for k in ("physics_sha256", "policy_interface_sha256") if rec_plant.get(k) != plant.get(k)]

    run_block = sc.get("run") or {}
    cur_rec = sc.get("curriculum") or {}
    rw = sc.get("reward_weights") or {}
    hyper = sc.get("hyperparameters") or {}
    fp = sc.get("task_fingerprint") or {}
    bound = {
        "stage_config.json": {
            "species": _get(sc, "species"), "embodiment": _get(sc, "embodiment"), "stage": _get(sc, "stage"),
            "git_commit": _get(sc, "git_commit"),
            **{f"run.{k}": _get(run_block, k) for k in (
                "load_mode", "seed", "parent_checkpoint_sha256", "timesteps", "resume_load_path",
                "hyperparameters_sha256", "n_envs", "parent_task_sha256", "critic_phase_dropped",
                "warm_start_log_std")},
            "task_fingerprint.task_sha256": _get(fp, "task_sha256"),
            **{f"curriculum.{k}": _get(cur_rec, k) for k in (
                "warmup_timesteps", "ramp_start_value", "initialize_from_own_checkpoint", "warm_start_log_std",
                *judge.CADENCE_KEYS)},
            "hyperparameters.critic_phase": _get(hyper, "critic_phase"),
            **{f"reward_weights.{k}": _get(rw, k) for k in (
                "command_mode", "command_config", "prey_lateral_range", *judge.PRICE_KEYS, *judge.R6_KEYS)},
        },
        "plant_identity.json": {k: _get(rec_plant, k) for k in ("physics_revision", "physics_sha256",
                                                               "policy_interface_sha256")},
        "zip data member mesozoic_task_lineage": {
            "mode": _get(lineage, "mode"), "parent_stage": _get(lineage, "parent_stage"),
            "zeroed_command_parameters": _get(lineage, "zeroed_command_parameters"), "record": lineage},
    }
    anchors = {
        "HYPERPARAMETERS_SHA256": [judge.HYPERPARAMETERS_SHA256, hp_here, run_block.get("hyperparameters_sha256")],
        "TASK_SHA256": [judge.TASK_SHA256, task_here, fp.get("task_sha256")],
        "COMMAND_TABLE_vs_committed_same_json": judge.same_json(committed_table, judge.COMMAND_TABLE),
        "COMMAND_TABLE_vs_recorded_same_json": judge.same_json(rw.get("command_config"), judge.COMMAND_TABLE),
        "PILOT_HEAD_vs_tree_head": [judge.PILOT_HEAD, head],
        "PARENT_TASK_SHA256_vs_recorded": [judge.PARENT_TASK_SHA256, run_block.get("parent_task_sha256")],
        "PARENT_SHA256_seed_vs_recorded": [want_parent, run_block.get("parent_checkpoint_sha256")],
        "CADENCE_vs_recorded": {k: [v, cur_rec.get(k)] for k, v in judge.CADENCE.items()},
        "PRICE_and_R6_vs_recorded": {k: [v, rw.get(k)] for k, v in {**judge.PRICE_KEYS, **judge.R6_KEYS}.items()},
        "PREY_LATERAL_RANGE_vs_recorded": [list(judge.PREY_LATERAL_RANGE), rw.get("prey_lateral_range")],
        "COMMAND_MODE_vs_recorded": [judge.COMMAND_MODE, rw.get("command_mode")],
        "entry_constants_vs_recorded": {
            "WARMUP_TIMESTEPS": [judge.WARMUP_TIMESTEPS, cur_rec.get("warmup_timesteps")],
            "RAMP_START_VALUE": [judge.RAMP_START_VALUE, cur_rec.get("ramp_start_value")],
            "N_ENVS": [judge.N_ENVS, run_block.get("n_envs")]},
        "LINEAGE_vs_zip": {"mode": [judge.LINEAGE_MODE, _get(lineage, "mode")],
                           "parent_stage": [judge.LINEAGE_PARENT_STAGE, _get(lineage, "parent_stage")]},
        "PILOT_UNINTERRUPTED_END_vs_tree": [judge.PILOT_UNINTERRUPTED_END,
                                            judge.uninterrupted_end(judge.PILOT_BUDGET, rollout), rollout],
        "tree_shaping_inert": judge.shaping_inert(warmup, ramp_start, forward_weight),
    }
    out = {
        "judge_file": str(jpath), "judge_sha256": jsha, "stage_records_copied": copied,
        "edge_zip": str(zpath), "edge_zip_sha256": "sha256:" + sha256(str(zpath)),
        "edge_zip_data_num_timesteps": data.get("num_timesteps"),
        "lineage_zip_equals_in_memory": judge.same_json(lineage, in_memory), "lineage_note": lineage_note,
        "bound": bound, "anchors": anchors,
        "binding_problems": problems, "binding_info": info,
        "run_id_problem": judge.run_id_problem(run_name, SEED), "run_name": run_name,
        "commit_problem": judge.commit_problem(sc.get("git_commit"), head), "tree_head": head,
        "plant_problems": plant_problems, "plant_tree": {k: plant.get(k) for k in (
            "physics_revision", "physics_sha256", "policy_interface_sha256")},
        "env_kwargs_drift": kwargs_drift,
        "checks_separately": {
            "entry_problems": judge.entry_problems(sc, lineage, lineage_note),
            "env_problems": judge.env_problems(sc, committed_table),
            "cadence_problems": judge.cadence_problems(sc),
            "anchor_problems": judge.anchor_problems(hp_here, task_here, committed_table),
            "parent_problems": judge.parent_problems(SEED, want_parent, run_block.get("parent_checkpoint_sha256")),
            "lineage_problems": judge.lineage_problems(lineage),
            "drift_problem": judge.drift_problem(kwargs_drift)},
        "hyperparameters_sha256_ppo_lower_equals_PPO": hp_here == hyperparameters_sha256(config, "PPO"),
    }
    out["accepted"] = (not problems and out["run_id_problem"] is None and out["commit_problem"] is None
                       and not plant_problems and not kwargs_drift and out["lineage_zip_equals_in_memory"])
    (OUT / "judge_bound_keys.json").write_text(json.dumps(bound, indent=1, default=str) + "\n")
    return out

# ---------------------------------------------------------------------------------------------------------------
# 3c. VecNormalize
# ---------------------------------------------------------------------------------------------------------------
def _eval_env():
    for cb in getattr(callback_list, "callbacks", []):
        env = getattr(cb, "eval_env", None)
        if env is not None and hasattr(env, "obs_rms"):
            return env
    return None


@section("vecnormalize")
def _vecnormalize():
    out = {"parent": {"mean_60_63": orig_vn.obs_rms.mean[60:].tolist(), "var_60_63": orig_vn.obs_rms.var[60:].tolist(),
                      "count": float(orig_vn.obs_rms.count), "ret_rms_mean": float(np.ravel(orig_vn.ret_rms.mean)[0]),
                      "ret_rms_var": float(np.ravel(orig_vn.ret_rms.var)[0]), "ret_rms_count": float(orig_vn.ret_rms.count),
                      "clip_obs": float(orig_vn.clip_obs), "epsilon": float(orig_vn.epsilon), "gamma": float(orig_vn.gamma)}}
    for label, env in (("train_env", train_env), ("eval_env", _eval_env())):
        if env is None:
            out[label] = None
            continue
        m, v = env.obs_rms.mean, env.obs_rms.var
        out[label] = {
            "class": type(env).__name__,
            "mean_0_60_bit_equal_parent": bool(np.array_equal(m[:60], orig_vn.obs_rms.mean[:60])),
            "var_0_60_bit_equal_parent": bool(np.array_equal(v[:60], orig_vn.obs_rms.var[:60])),
            "mean_60_63": m[60:].tolist(), "var_60_63": v[60:].tolist(),
            "count": float(env.obs_rms.count),
            "ret_rms_mean": float(np.ravel(env.ret_rms.mean)[0]), "ret_rms_var": float(np.ravel(env.ret_rms.var)[0]),
            "ret_rms_count": float(env.ret_rms.count),
            "returns": np.asarray(env.returns).tolist() if getattr(env, "returns", None) is not None else None,
            "training": bool(env.training), "norm_obs": bool(env.norm_obs), "norm_reward": bool(env.norm_reward),
            "clip_obs": float(env.clip_obs), "clip_reward": float(env.clip_reward), "epsilon": float(env.epsilon),
            "gamma": float(env.gamma), "num_envs": int(env.num_envs),
        }
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3d. callbacks as attached to learn()
# ---------------------------------------------------------------------------------------------------------------
def _scalar_attrs(obj) -> dict:
    out = {}
    for k, v in vars(obj).items():
        if k in ("model", "locals", "globals", "logger", "parent", "training_env"):
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 8 and all(isinstance(x, (bool, int, float, str)) for x in v):
            out[k] = list(v)
    return out


@section("callbacks")
def _callbacks():
    cbs = list(getattr(callback_list, "callbacks", []))
    return {"callback_list_class": type(callback_list).__name__,
            "order": [type(cb).__name__ for cb in cbs],
            "attributes": [{"class": type(cb).__name__, **_scalar_attrs(cb)} for cb in cbs]}


# ---------------------------------------------------------------------------------------------------------------
# 3e. policy helpers
# ---------------------------------------------------------------------------------------------------------------
def norm_new(o):
    return train_env.normalize_obs(np.asarray(o, dtype=np.float32))


def norm_orig(o):
    return orig_vn.normalize_obs(np.asarray(o, dtype=np.float32))


def act(m, x):
    return m.predict(np.asarray(x)[None], deterministic=True)[0][0]


def mean_and_value(m, x):
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(x)[None])
        mu = m.policy.get_distribution(t).distribution.mean[0].numpy()
        v = float(m.policy.predict_values(t)[0, 0])
    return mu, v


def zero_cmd(o):
    z = np.array(o, dtype=np.float32, copy=True)
    z[60:] = 0.0
    return z


# ---------------------------------------------------------------------------------------------------------------
# 4. the walker's actions at the start against pilot 3's, on pilot 3's own observations (its exact [env])
# ---------------------------------------------------------------------------------------------------------------
@section("start_actions_on_pilot3_observations")
def _start_actions():
    env = EnvClass(**P3X_KW)
    obs, _ = env.reset(seed=EPISODE_SEED)
    da = dm = dv = 0.0
    nonzero_cmd = 0
    actions, headings, xy = [], [env._heading()], [env.data.qpos[0:2].copy()]
    n = 0
    for _ in range(MAX_STEPS):
        nonzero_cmd += int(np.any(obs[60:] != 0.0))
        a_o = act(orig, norm_orig(obs))
        a_n = act(model, norm_new(obs))
        mu_o, v_o = mean_and_value(orig, norm_orig(obs))
        mu_n, v_n = mean_and_value(model, norm_new(obs))
        da = max(da, float(np.max(np.abs(a_o - a_n))))
        dm = max(dm, float(np.max(np.abs(mu_o - mu_n))))
        dv = max(dv, abs(v_o - v_n))
        actions.append(a_o)
        obs, r, te, tr, info = env.step(a_o)
        headings.append(env._heading())
        xy.append(env.data.qpos[0:2].copy())
        n += 1
        if te or tr:
            break
    env.close()
    np.save(OUT / "actions_p3x_closed_loop.npy", np.asarray(actions, dtype=np.float32))
    return {"env": "built tree, pilot 3's exact [env] (no command, no price, prey_lateral_range default)",
            "episode_seed": EPISODE_SEED, "steps": n, "terminated": bool(te), "truncated": bool(tr),
            "steps_with_nonzero_command_inputs": nonzero_cmd,
            "max_abs_action_delta_new_vs_pilot3": da, "max_abs_mean_delta_new_vs_pilot3": dm,
            "max_abs_value_delta_new_vs_pilot3": dv,
            "pilot3_walker_metrics": heading_metrics(headings, xy),
            "distance_traveled": float(info.get("distance_traveled", float("nan")))}


# ---------------------------------------------------------------------------------------------------------------
# 5. the 1,000-step deterministic rollout in the built env, with the command-free twin on the same actions
# ---------------------------------------------------------------------------------------------------------------
@section("rollout")
def _rollout():
    from environments.shared.direction_commands import wrap_angle

    env4, envt = EnvClass(**P4_KW), EnvClass(**TW_KW)
    # postfix/code: the same env with only the price weight set to 0.0 (command live, prey on the line), stepped on
    # the same actions: everything but the price term must be bit-identical.
    P0_KW = {**P4_KW, "walk_heading_price_weight": 0.0}
    env0 = EnvClass(**P0_KW)
    o0, _ = env0.reset(seed=EPISODE_SEED)
    z = {"obs_mismatch_steps": 0, "state_mismatch_steps": 0, "flag_mismatch_steps": 0,
         "reward_identity_fail_steps": 0, "reward_total_identity_fail_steps": 0, "keys_only_on": set(),
         "keys_only_off": set(), "common_mismatch": {}, "reset_obs_equal": None}
    cfg = env4.command_config
    o4, _ = env4.reset(seed=EPISODE_SEED)
    ot, _ = envt.reset(seed=EPISODE_SEED)
    z["reset_obs_equal"] = bool(np.array_equal(o4, o0))
    st0 = env4._command_state
    rec = {
        "episode_seed": EPISODE_SEED, "command_config_class": type(cfg).__name__, "command_config": jsonable(vars(cfg)),
        "reset_qpos_qvel_equal": bool(np.array_equal(env4.data.qpos, envt.data.qpos)
                                      and np.array_equal(env4.data.qvel, envt.data.qvel)),
        "reset_obs_0_60_equal": bool(np.array_equal(o4[:60], ot[:60])),
        "reset_obs_60_63": o4[60:].tolist(), "spawn_heading_rad": env4._heading(),
        "desired_heading_at_reset": st0.desired_heading, "heading_active": bool(st0.heading_active),
        "root_body_is_pelvis": bool(int(env4._root_body_id) == int(getattr(env4, "pelvis_id", -1))),
        "prey_line_heading_deg": (math.degrees(math.atan2(env4._initial_prey_dir_2d[1], env4._initial_prey_dir_2d[0]))
                                  if hasattr(env4, "_initial_prey_dir_2d") else None),
    }
    # per-step tallies
    common_mismatch: dict = {}
    extra_keys, missing_keys = set(), set()
    m = {"obs_0_60_mismatch_steps": 0, "state_mismatch_steps": 0, "flag_mismatch_steps": 0,
         "reward_identity_fail_steps": 0, "reward_total_identity_fail_steps": 0,
         "max_action_delta_vs_pilot3_on_zeroed": 0.0, "max_mean_delta_vs_pilot3_on_zeroed": 0.0,
         "max_value_delta_vs_pilot3_on_zeroed": 0.0, "max_action_delta_pilot3_unprepared_on_live": 0.0,
         "max_abs_obs62_minus_state_normalized": 0.0, "max_abs_obs60_minus_state_normalized": 0.0,
         "max_abs_obs61": 0.0, "max_abs_obs62_minus_formula": 0.0, "max_abs_obs60_minus_formula": 0.0,
         "max_abs_state_error_plus_price_theta": 0.0, "max_abs_desired_heading_drift": 0.0,
         "obs62_saturated_steps": 0, "price_saturated_steps": 0}
    actions, price, theta, heading, xy, r4s, rts, cmd = [], [], [], [env4._heading()], [env4.data.qpos[0:2].copy()], [], [], []
    n = 0
    for _ in range(MAX_STEPS):
        x = norm_new(o4)
        a = act(model, x)
        mu_n, v_n = mean_and_value(model, x)
        mu_o, v_o = mean_and_value(orig, norm_orig(zero_cmd(o4)))
        a_oz = act(orig, norm_orig(zero_cmd(o4)))
        a_ol = act(orig, norm_orig(o4))
        m["max_action_delta_vs_pilot3_on_zeroed"] = max(m["max_action_delta_vs_pilot3_on_zeroed"], float(np.max(np.abs(a - a_oz))))
        m["max_mean_delta_vs_pilot3_on_zeroed"] = max(m["max_mean_delta_vs_pilot3_on_zeroed"], float(np.max(np.abs(mu_n - mu_o))))
        m["max_value_delta_vs_pilot3_on_zeroed"] = max(m["max_value_delta_vs_pilot3_on_zeroed"], abs(v_n - v_o))
        m["max_action_delta_pilot3_unprepared_on_live"] = max(m["max_action_delta_pilot3_unprepared_on_live"],
                                                              float(np.max(np.abs(a_ol - a_oz))))
        actions.append(a)
        o4, r4, te4, tr4, i4 = env4.step(a)
        ot, rt, tet, trt, it = envt.step(a)
        oz, rz, tez, trz, iz = env0.step(a)
        n += 1
        jz4, jz0 = jsonable(i4), jsonable(iz)
        z["keys_only_on"] |= set(jz4) - set(jz0)
        z["keys_only_off"] |= set(jz0) - set(jz4)
        for k in jz0:
            if k in jz4 and not same(jz4[k], jz0[k]):
                z["common_mismatch"][k] = z["common_mismatch"].get(k, 0) + 1
        z["obs_mismatch_steps"] += int(not np.array_equal(o4, oz))
        z["state_mismatch_steps"] += int(not (np.array_equal(env4.data.qpos, env0.data.qpos)
                                              and np.array_equal(env4.data.qvel, env0.data.qvel)))
        z["flag_mismatch_steps"] += int((te4, tr4) != (tez, trz))
        pz = float(i4["reward_walk_heading_price"])
        if not (te4 or tr4):
            z["reward_identity_fail_steps"] += int(not (float(r4) == float(rz) + pz))
        z["reward_total_identity_fail_steps"] += int(not (float(i4["reward_total"]) == float(iz["reward_total"]) + pz))
        j4, jt = jsonable(i4), jsonable(it)
        extra_keys |= set(j4) - set(jt)
        missing_keys |= set(jt) - set(j4)
        for k in jt:
            if k in j4 and not same(j4[k], jt[k]):
                common_mismatch[k] = common_mismatch.get(k, 0) + 1
        if not np.array_equal(o4[:60], ot[:60]):
            m["obs_0_60_mismatch_steps"] += 1
        if not (np.array_equal(env4.data.qpos, envt.data.qpos) and np.array_equal(env4.data.qvel, envt.data.qvel)):
            m["state_mismatch_steps"] += 1
        if (te4, tr4) != (tet, trt):
            m["flag_mismatch_steps"] += 1
        p = float(i4["reward_walk_heading_price"])
        th = float(i4["walk_heading_error_rad"])
        if not (te4 or tr4):
            m["reward_identity_fail_steps"] += int(not (float(r4) == float(rt) + p))
        m["reward_total_identity_fail_steps"] += int(not (float(i4["reward_total"]) == float(it["reward_total"]) + p))
        st = env4._command_state
        m["max_abs_obs62_minus_state_normalized"] = max(m["max_abs_obs62_minus_state_normalized"], abs(float(o4[62]) - float(st.normalized[2])))
        m["max_abs_obs60_minus_state_normalized"] = max(m["max_abs_obs60_minus_state_normalized"], abs(float(o4[60]) - float(st.normalized[0])))
        m["max_abs_obs61"] = max(m["max_abs_obs61"], abs(float(o4[61])))
        err = wrap_angle(st.desired_heading - env4._heading())
        f62 = float(np.clip(cfg.yaw_gain * err, -cfg.yaw_rate_max, cfg.yaw_rate_max)) / cfg.yaw_rate_scale
        f60 = cfg.cruise_speed * max(cfg.minimum_turn_speed_fraction, math.cos(err)) / cfg.speed_scale
        m["max_abs_obs62_minus_formula"] = max(m["max_abs_obs62_minus_formula"], abs(float(o4[62]) - f62))
        m["max_abs_obs60_minus_formula"] = max(m["max_abs_obs60_minus_formula"], abs(float(o4[60]) - f60))
        m["max_abs_state_error_plus_price_theta"] = max(m["max_abs_state_error_plus_price_theta"], abs(float(st.heading_error) + th))
        m["max_abs_desired_heading_drift"] = max(m["max_abs_desired_heading_drift"], abs(st.desired_heading - st0.desired_heading))
        m["obs62_saturated_steps"] += int(abs(float(o4[62])) >= 0.5 - 1e-6)
        m["price_saturated_steps"] += int(abs(th) >= float(P4_KW["walk_heading_price_scale_rad"]))
        price.append(p)
        theta.append(th)
        heading.append(env4._heading())
        xy.append(env4.data.qpos[0:2].copy())
        r4s.append(float(r4))
        rts.append(float(rt))
        cmd.append(o4[60:].astype(np.float64))
        if te4 or tr4:
            break
    env4.close()
    envt.close()
    env0.close()
    rec["price_weight_zero_twin"] = {
        **{k: (sorted(v) if isinstance(v, set) else v) for k, v in z.items()},
        "n_common_info_keys": len(set(jsonable(i4)) & set(jsonable(iz))),
        "r6_info_keys_common": sorted(k for k in set(jsonable(i4)) & set(jsonable(iz)) if "walk" in k),
        "kwargs_differ_only_in": sorted(k for k in set(P4_KW) | set(P0_KW) if P4_KW.get(k) != P0_KW.get(k)),
    }
    th = np.asarray(theta)
    rec.update(m)
    rec.update({
        "steps": n, "terminated": bool(te4), "truncated": bool(tr4),
        "info_keys_only_in_built_env": sorted(extra_keys), "info_keys_only_in_twin": sorted(missing_keys),
        "common_info_keys_with_any_mismatch": common_mismatch,
        "n_common_info_keys": len(set(jt) & set(j4)),
        "price_sum": float(np.sum(price)), "price_mean_per_step": float(np.mean(price)),
        "price_min_step": float(np.min(price)),
        "return_built_env": float(np.sum(r4s)), "return_twin": float(np.sum(rts)),
        "theta_deg": {"end": float(np.degrees(th[-1])), "max_abs": float(np.degrees(np.max(np.abs(th)))),
                      "rms": float(np.degrees(np.sqrt(np.mean(th ** 2)))), "mean": float(np.degrees(np.mean(th)))},
        "obs62": {"min": float(np.min([c[2] for c in cmd])), "max": float(np.max([c[2] for c in cmd])),
                  "end": float(cmd[-1][2])},
        "obs60": {"min": float(np.min([c[0] for c in cmd])), "max": float(np.max([c[0] for c in cmd])),
                  "end": float(cmd[-1][0])},
        "metrics": heading_metrics(heading, xy),
        "distance_traveled": float(i4.get("distance_traveled", float("nan"))),
    })
    np.save(OUT / "actions_built_rollout.npy", np.asarray(actions, dtype=np.float32))
    np.savez(OUT / "rollout_series.npz", price=np.asarray(price), theta=th, heading=np.asarray(heading),
             xy=np.asarray(xy), reward_built=np.asarray(r4s), reward_twin=np.asarray(rts), obs_60_63=np.asarray(cmd))
    return rec


@section("extra_episodes")
def _extra():
    out = {}
    for es in EXTRA_EPISODE_SEEDS:
        row = {}
        for label, kw, mdl, norm in (("built_env_new_walker", P4_KW, model, norm_new),
                                     ("pilot3_env_pilot3_walker", P3X_KW, orig, norm_orig)):
            env = EnvClass(**kw)
            obs, _ = env.reset(seed=es)
            hs, xy, ps, n = [env._heading()], [env.data.qpos[0:2].copy()], [], 0
            for _ in range(MAX_STEPS):
                obs, r, te, tr, info = env.step(act(mdl, norm(obs)))
                hs.append(env._heading())
                xy.append(env.data.qpos[0:2].copy())
                ps.append(float(info.get("reward_walk_heading_price", 0.0)))
                n += 1
                if te or tr:
                    break
            env.close()
            row[label] = {**heading_metrics(hs, xy), "terminated": bool(te), "price_sum": float(np.sum(ps))}
        out[str(es)] = row
    return out


# ---------------------------------------------------------------------------------------------------------------
# 6. built-tree replays for the 31d02e42 code-identity check (the p3 side replays the same actions)
# ---------------------------------------------------------------------------------------------------------------
@section("replays_written")
def _replays():
    a_built = np.load(OUT / "actions_built_rollout.npy")
    a_p3x = np.load(OUT / "actions_p3x_closed_loop.npy")
    rec_a = replay(EnvClass(**TW_KW), a_built, EPISODE_SEED)
    rec_b = replay(EnvClass(**P3X_KW), a_p3x, EPISODE_SEED)
    with open(OUT / "replay_twin_built_tree.json", "w") as f:
        json.dump(rec_a, f)
    with open(OUT / "replay_p3x_built_tree.json", "w") as f:
        json.dump(rec_b, f)
    # postfix/code: a second action sequence in pilot 3's exact [env] (the new walker's live-command actions).
    rec_c = replay(EnvClass(**P3X_KW), a_built, EPISODE_SEED)
    with open(OUT / "replay_p3x_builtactions_built_tree.json", "w") as f:
        json.dump(rec_c, f)
    return {"twin_steps": len(rec_a["reward"]), "p3x_steps": len(rec_b["reward"]), "p3x_builtactions_steps": len(rec_c["reward"])}


# ---------------------------------------------------------------------------------------------------------------
# 7. the shaping callbacks driven by hand on the captured model (no learn(), no update): weights at step 0
# ---------------------------------------------------------------------------------------------------------------
@section("shaping_simulation")
def _shaping():
    from stable_baselines3.common.logger import Logger

    from environments.shared.curriculum.schedules import ENT_COEF_WARMUP_MARKER

    names = ("EntCoefDecayCallback", "StageWarmupCallback", "RewardRampCallback")
    cbs = [cb for cb in callback_list.callbacks if type(cb).__name__ in names]
    inner = train_env.venv

    def snap():
        return {"clip_range(1.0)": float(model.clip_range(1.0)), "ent_coef": float(model.ent_coef),
                "warmup_marker": bool(getattr(model, ENT_COEF_WARMUP_MARKER, False)),
                "forward_vel_weight": [float(v) for v in inner.get_attr("forward_vel_weight")]}

    rows = {"order": [type(cb).__name__ for cb in cbs], "before_training_start": snap()}
    model._logger = Logger(folder=None, output_formats=[])
    model.num_timesteps = 0  # what learn()'s _setup_learn sets under reset_num_timesteps=True
    for cb in cbs:
        cb.init_callback(model)
    for cb in cbs:
        cb.on_training_start({}, {})
    rows["after_training_start"] = snap()
    for t in (4, 8, 10_000, 1_000_000, 2_000_000, 2_015_232):
        model.num_timesteps = t
        for cb in cbs:
            cb.on_step()
        rows[f"after_step_at_{t}"] = snap()
    return rows


RESULT["pair_sha256_after"] = {"run_copy": [sha256(p) for p in run_pair(SEED)], "my_copy": [sha256(p) for p in my_pair(SEED)]}
RESULT["guard_hits_final"] = list(GUARD_HITS)
save()
print(json.dumps({k: RESULT.get(k) for k in ("learn_reached", "manual_cell_stopped_by_learn_stop", "guard_hits_final",
                                            "errors")}, indent=1, default=str), flush=True)
