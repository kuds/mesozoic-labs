"""Tail_3 bound screen: ONE continuation replicate (training entry point; BUILD_SPEC section 4).

    PY -I -B SCREEN/tools/run_tailscreen.py --state {s42,s80} --arm {control,b90,b60} --rep-seed {101,202,303} \
        --steps 1048576 [--preflight-only]

(-I ignores PYTHONDONTWRITEBYTECODE; this module also sets sys.dont_write_bytecode before it loads anything, so
no __pycache__ is written into the scratch clone or tools/ even without -B.)

Refuses (exit 2, nothing created) unless every pre-training check passes (preflight): SCREEN/APPROVED exists and pins
criteria.txt's and screen_common.py's sha256; calib/coef.json follows the rule and equals APPROVED's pins; the
arguments are in their sets; runs/<name> does not exist; the start pair's sha256; the scratch clone (clean with
ignored files counted, on its branch, HEAD = the approved scratch commit, a linear descendant of cf80ecb1 with no
merge, push URL disabled, the two changed files); the stack; tests_out/gate_pass.json names that HEAD, records
failed 0 and every screen_common.GATE_REQUIRED test as passed; 5 GB free; every pinned tool's sha256.  There is
no override: no --coef, --bound, --out-dir, --save-every or --allow-dirty.
Coef and b come only from the arm and coef.json: control (0.0, 0.90), b90 (COEF_90, 0.90), b60 (COEF_60, 0.60).

Derived from P1's run_continuation.py (TAILEXP): the run's training setup at 4,194,304 through train_base's own
same-stage resume path (create_vec_env with n_envs 4 and the replicate's seed, the eval env built and seeded in the
same order, the run's VecNormalize pickle with training on, _prepare_alg_kwargs, _create_or_load_model with
enable_mirror_kl), device cpu, one torch thread, EntCoefDecayCallback, SB3's CheckpointCallback every 131,072 steps,
learn(reset_num_timesteps=False) with the run's 12,000,000 budget, stopped after exactly --steps.  The tail_3 upper
bound is enabled for every arm (control at coef 0: logged only, training bit for bit as without it).

Outputs, only in SCREEN/runs/<name>/: stage1_<N>_steps.zip / stage1_vecnormalize_<N>_steps.pkl (N = 4,325,376 ...
5,242,880), stage1_final.zip / stage1_final_vecnorm.pkl, progress.csv, telemetry.npz, run_meta.json.
Exit codes: 0 done; 2 refused (nothing created); 1 crash (the folder stays for the crash rule).

Importing this module needs neither torch nor environments (preflight and arm_config are pure; T6).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any import from tools/ or the clone: -I ignores PYTHONDONTWRITEBYTECODE

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

TOOLS = Path(os.path.realpath(__file__)).parent


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(TOOLS / f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load("screen_common")

PUSH_DISABLED = "DISABLED-never-push"
MIN_FREE_BYTES = 5 * 1024**3
TOTAL_TIMESTEPS = 12_000_000
COEF_SCHEMA = "tailscreen-coef/1"
REFERENCE_EXCESS = 0.11
UNIT_PULL = 2 * REFERENCE_EXCESS  # d L_B / d bias[14] per unit coef with every mu14 = b + 0.11
RUN_SCHEMA = "tailscreen-run/1"


# ---------------------------------------------------------------------------------------------------- pure helpers
def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_name(state: str, arm: str, rep_seed: int) -> str:
    return f"{state}_{arm}_r{rep_seed}"


def coef_by_rule(g_pg_rms: float) -> float:
    """COEF_b = RMS(G_pg) / (2 x 0.11), two significant figures (format '.2g'); the same for both b."""
    return float(format(g_pg_rms / UNIT_PULL, ".2g"))


def coef_rule_problems(coef_json: Any) -> list[str]:
    """Why a coef.json does not follow the rule (empty when it does)."""
    if not isinstance(coef_json, dict):
        return ["not a JSON object"]
    problems = []
    if coef_json.get("schema") != COEF_SCHEMA:
        problems.append(f"schema {coef_json.get('schema')!r} != {COEF_SCHEMA!r}")
    if coef_json.get("reference_excess") != REFERENCE_EXCESS or coef_json.get("unit_pull_per_coef") != UNIT_PULL:
        problems.append("reference_excess / unit_pull_per_coef are not 0.11 / 0.22")
    pooled = coef_json.get("pooled") if isinstance(coef_json.get("pooled"), dict) else {}
    rms = pooled.get("G_pg_rms")
    if not isinstance(rms, (int, float)) or isinstance(rms, bool) or not math.isfinite(rms) or rms <= 0:
        problems.append(f"pooled.G_pg_rms {rms!r} is not a finite positive number")
        expected = None
    else:
        expected = coef_by_rule(float(rms))
    if pooled.get("minibatches") != 512:
        problems.append(f"pooled.minibatches {pooled.get('minibatches')!r} != 512")
    for key in ("COEF_90", "COEF_60"):
        value = coef_json.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            problems.append(f"{key} {value!r} is not a finite positive number")
        elif expected is not None and float(value) != expected:
            problems.append(f"{key} {value!r} != the rule's {expected!r} from pooled.G_pg_rms")
    states = coef_json.get("states")
    if not isinstance(states, dict) or set(states) != set(C.STATES):
        problems.append(f"states {sorted(states) if isinstance(states, dict) else states!r} != {sorted(C.STATES)}")
    else:
        for state, record in states.items():
            if not isinstance(record, dict) or record.get("minibatches") != 256:
                problems.append(f"states.{state}.minibatches != 256")
                continue
            if record.get("zip_sha256") != C.STATES[state]["zip_sha256"]:
                problems.append(f"states.{state}.zip_sha256 is not the start zip's")
            if record.get("pkl_sha256") != C.STATES[state]["pkl_sha256"]:
                problems.append(f"states.{state}.pkl_sha256 is not the start pkl's")
            if record.get("digest_before") != record.get("digest_after"):
                problems.append(f"states.{state}: the parameter/Adam digest changed during calibration")
    if coef_json.get("stack") != C.STACK:
        problems.append(f"stack {coef_json.get('stack')!r} != the pinned {C.STACK!r}")
    return problems


def arm_config(arm: str, coef_json: dict) -> tuple[float, float]:
    """(coef, b) of an arm: control (0.0, 0.90), b90 (COEF_90, 0.90), b60 (COEF_60, 0.60)."""
    if arm not in C.ARMS:
        raise ValueError(f"unknown arm {arm!r}; arms are {sorted(C.ARMS)}")
    key, bound = C.ARMS[arm]
    if key == "COEF0":
        return 0.0, float(bound)
    return float(coef_json[key]), float(bound)


def _git(clone: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--no-optional-locks", "-C", clone, *args], capture_output=True, text=True, check=False)


def clone_state(clone: str) -> dict[str, Any]:
    """HEAD, status (ignored files included: a stray __pycache__ makes the clone dirty), branch, parent, push URLs,
    the commit chain PARENT..HEAD and the changed-file set against PARENT (read only)."""

    def out(*args: str) -> str | None:
        done = _git(clone, *args)
        return done.stdout.strip() if done.returncode == 0 else None

    remotes = out("remote", "-v") or ""
    push_urls = sorted({line.split()[1] for line in remotes.splitlines() if line.endswith("(push)")})
    diff = out("diff", "--name-only", f"{C.PARENT}..HEAD")
    chain_text = out("rev-list", "--reverse", "--parents", f"{C.PARENT}..HEAD")
    chain = [line.split() for line in chain_text.splitlines()] if chain_text else []
    # Linear: every commit after PARENT has exactly one parent, the first one's is PARENT, each the previous one's.
    chain_ok = (
        chain_text is not None
        and _git(clone, "merge-base", "--is-ancestor", C.PARENT, "HEAD").returncode == 0
        and len(chain) >= 1
        and all(len(c) == 2 for c in chain)
        and chain[0][1] == C.PARENT
        and all(chain[i][1] == chain[i - 1][0] for i in range(1, len(chain)))
    )
    return {
        "head": out("rev-parse", "HEAD"),
        "parent": out("rev-parse", "HEAD^"),
        "status": out("status", "--porcelain", "--ignored"),
        "branch": out("rev-parse", "--abbrev-ref", "HEAD"),
        "push_url": push_urls,
        "commits": [c[0] for c in chain],
        "chain_ok": bool(chain_ok),
        "changed_files": sorted(diff.splitlines()) if diff is not None else None,
    }


def clone_problems(git: dict, approved: dict | None) -> list[str]:
    """Why the scratch clone may not train (empty when it may).  Shared by preflight and gate.sh."""
    problems: list[str] = []
    if git["head"] is None:
        return [f"clone: {C.CLONE} is not a git checkout"]
    if git["status"] != "":
        problems.append(f"clone: not clean, ignored files included ({git['status']!r})")
    if git["branch"] != C.BRANCH:
        problems.append(f"clone: on {git['branch']!r}, not {C.BRANCH!r}")
    if approved is None or git["head"] != approved.get("scratch_commit"):
        problems.append(
            f"clone: HEAD {git['head']} != the APPROVED scratch_commit "
            f"{None if approved is None else approved.get('scratch_commit')!r}"
        )
    if C.SCRATCH_COMMIT is None or git["head"] != C.SCRATCH_COMMIT:
        problems.append(f"clone: HEAD {git['head']} != the pin SCRATCH_COMMIT {C.SCRATCH_COMMIT!r}")
    if not git.get("chain_ok"):
        problems.append(f"clone: {C.PARENT[:8]}..HEAD is not a linear chain of commits on {C.PARENT}")
    if not git["push_url"] or any(url != PUSH_DISABLED for url in git["push_url"]):
        problems.append(f"clone: push URL(s) {git['push_url']} not all {PUSH_DISABLED!r}")
    if git["changed_files"] != sorted(C.CHANGED_FILES):
        problems.append(f"clone: changed files {git['changed_files']} != {sorted(C.CHANGED_FILES)}")
    return problems


def tool_pin_problems() -> list[str]:
    """Every pinned tool (screen_common.TOOL_FILES) must be on disk with its TOOL_SHA256 pin."""
    pins = C.TOOL_SHA256 if isinstance(C.TOOL_SHA256, dict) else {}
    problems = []
    for name in C.TOOL_FILES:
        path = TOOLS / name
        got = sha256_file(path) if path.is_file() else None
        if not pins.get(name) or pins.get(name) != got:
            problems.append(f"pins: TOOL_SHA256[{name!r}] {pins.get(name)!r} is not the file's sha256 {got!r}")
    return problems


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        with open(path) as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as exc:
        return None, f"does not parse ({exc.__class__.__name__})"


def preflight(ns: Any, root: Path) -> list[str]:
    """Every reason this replicate may not train; empty means pass.  Reads only; creates nothing."""
    root = Path(root)
    problems: list[str] = []
    # 1. APPROVED and criteria.txt
    approved, err = _read_json(root / "APPROVED")
    if err:
        problems.append(f"APPROVED: {err}")
        approved = None
    elif not isinstance(approved, dict):
        problems.append("APPROVED: not a JSON object")
        approved = None
    else:
        missing = [k for k in ("criteria_sha256", "scratch_commit", "COEF_90", "COEF_60", "coef_json_sha256",
                               "screen_common_sha256") if k not in approved]
        if missing:
            problems.append(f"APPROVED: missing keys {missing}")
        criteria = root / "criteria.txt"
        if not criteria.is_file():
            problems.append("APPROVED: criteria.txt missing")
        elif sha256_file(criteria) != approved.get("criteria_sha256"):
            problems.append(
                f"APPROVED: criteria_sha256 {approved.get('criteria_sha256')!r} != sha256(criteria.txt) "
                f"{sha256_file(criteria)}"
            )
        common = TOOLS / "screen_common.py"
        if "screen_common_sha256" in approved and sha256_file(common) != approved["screen_common_sha256"]:
            problems.append(
                f"APPROVED: screen_common_sha256 {approved['screen_common_sha256']!r} != sha256(screen_common.py) "
                f"{sha256_file(common)}"
            )
    # 2. coef.json
    coef_path = root / "calib" / "coef.json"
    coef_json, err = _read_json(coef_path)
    if err:
        problems.append(f"coef.json: {err}")
        coef_json = None
    else:
        problems.extend(f"coef.json: {p}" for p in coef_rule_problems(coef_json))
        if approved is not None and isinstance(coef_json, dict):
            for key in ("COEF_90", "COEF_60"):
                if coef_json.get(key) != approved.get(key):
                    problems.append(f"coef.json: {key} {coef_json.get(key)!r} != the APPROVED pin {approved.get(key)!r}")
            if "coef_json_sha256" in approved and sha256_file(coef_path) != approved["coef_json_sha256"]:
                problems.append("coef.json: sha256 != the APPROVED pin coef_json_sha256")
    # 3. arguments
    state, arm = getattr(ns, "state", None), getattr(ns, "arm", None)
    rep_seed, steps = getattr(ns, "rep_seed", None), getattr(ns, "steps", None)
    args_ok = True
    if arm not in C.ARMS:
        problems.append(f"args: arm {arm!r} not in {sorted(C.ARMS)}")
        args_ok = False
    if state not in C.STATES:
        problems.append(f"args: state {state!r} not in {sorted(C.STATES)}")
        args_ok = False
    if rep_seed not in C.REP_SEEDS or isinstance(rep_seed, bool):
        problems.append(f"args: rep seed {rep_seed!r} not in {list(C.REP_SEEDS)}")
        args_ok = False
    if steps != C.STEPS or isinstance(steps, bool):
        problems.append(f"args: steps {steps!r} != {C.STEPS}")
    # 4. the run folder
    if args_ok:
        name = run_name(state, arm, rep_seed)
        if os.path.lexists(root / "runs" / name):
            problems.append(f"runs: {root / 'runs' / name} exists")
    # 5. the start pair
    if state in C.STATES:
        record = C.STATES[state]
        for kind in ("zip", "pkl"):
            path = record[kind]
            if not os.path.isfile(path):
                problems.append(f"start: {path} missing")
            elif sha256_file(path) != record[f"{kind}_sha256"]:
                problems.append(f"start: sha256 of {path} != {record[f'{kind}_sha256']}")
    # 6. the scratch clone
    git = clone_state(C.CLONE)
    problems.extend(clone_problems(git, approved))
    # 7. the stack
    problems.extend(f"stack: {p}" for p in C.check_stack())
    # 8. the gate
    gate, err = _read_json(root / "tests_out" / "gate_pass.json")
    if err:
        problems.append(f"gate: gate_pass.json {err}")
    else:
        problems.extend(f"gate: {p}" for p in C.gate_problems(gate, git["head"]))
    # 9. disk (and this tool's own pin)
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    if free < MIN_FREE_BYTES:
        problems.append(f"disk: {free / 1024**3:.1f} GB free < {MIN_FREE_BYTES / 1024**3:.0f} GB")
    problems.extend(tool_pin_problems())
    return problems


# ---------------------------------------------------------------------------------------------------- training side
def bootstrap_clone() -> str:
    """Put the scratch clone at sys.path[0] and prove environments imports from it; returns its file."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    clone = os.path.realpath(C.CLONE)
    if sys.path[:1] != [clone]:
        sys.path.insert(0, clone)
    import environments

    origin = os.path.realpath(environments.__file__)
    if not origin.startswith(clone + os.sep):
        raise SystemExit(f"environments imported from {origin}, not the scratch clone {clone}")
    return origin


def modules_under_clone() -> int:
    clone = os.path.realpath(C.CLONE)
    n = 0
    for name, module in sorted(sys.modules.items()):
        if name == "environments" or name.startswith("environments."):
            path = getattr(module, "__file__", None)
            if path is None:
                continue
            if not os.path.realpath(path).startswith(clone + os.sep):
                raise SystemExit(f"{name} resolves to {os.path.realpath(path)}, outside {clone}")
            n += 1
    return n


def norm(x: Any) -> Any:
    """JSON round trip (tuples -> lists, numpy scalars -> numbers) for comparisons and records."""
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def recipe_checks(state: str, stage: str, config: dict, task_fingerprint: dict) -> dict:
    """The clone's stance stage against the start run's own stage_config.json (refused on a mismatch).  The run's
    seed is compared with the state's own seed (42 / 80), not with the replicate's seed."""
    from environments.shared.config import hyperparameters_sha256

    record = C.STATES[state]
    with open(record["stage_config"]) as fh:
        sc = json.load(fh)
    problems = []
    env_rec, env_now = norm(sc["reward_weights"]), norm(config["env_kwargs"])
    diff_env = sorted(k for k in env_now if env_rec.get(k) != env_now[k])
    if diff_env:
        problems.append(f"[env] keys differ from the run's stage_config.json: {diff_env}")
    if norm(sc["hyperparameters"]) != norm(config.get("ppo_kwargs", {})):
        problems.append("[ppo] differs from the run's stage_config.json hyperparameters")
    hp = hyperparameters_sha256(config, "ppo")
    if hp != sc["run"]["hyperparameters_sha256"]:
        problems.append(f"hyperparameters_sha256 {hp} != the run's {sc['run']['hyperparameters_sha256']}")
    if (sc["run"]["seed"], sc["run"]["n_envs"]) != (record["own_seed"], C.N_ENVS):
        problems.append(
            f"the run's seed / n_envs {sc['run']['seed']} / {sc['run']['n_envs']} != {record['own_seed']} / {C.N_ENVS}"
        )
    if task_fingerprint["task_sha256"] != sc["task_fingerprint"]["task_sha256"]:
        problems.append(f"task_sha256 {task_fingerprint['task_sha256']} != the run's {sc['task_fingerprint']['task_sha256']}")
    if sc["git_commit"] != C.PARENT:
        problems.append(f"the run's git_commit {sc['git_commit']} != {C.PARENT}")
    if norm(config.get("curriculum_kwargs", {})) != norm(sc["curriculum"]):
        problems.append("[curriculum] differs from the run's stage_config.json")
    if int(sc["run"]["timesteps"]) != TOTAL_TIMESTEPS:
        problems.append(f"the run's timesteps {sc['run']['timesteps']} != {TOTAL_TIMESTEPS}")
    if problems:
        raise SystemExit("recipe check refused: " + " | ".join(problems))
    return {"stage": stage, "stage_config": record["stage_config"], "hyperparameters_sha256": hp,
            "task_sha256": task_fingerprint["task_sha256"], "run_timesteps": int(sc["run"]["timesteps"]),
            "run_seed": int(sc["run"]["seed"]), "run_git_commit": sc["git_commit"], "env_keys_checked": len(env_now),
            "ppo_equal": True, "curriculum_equal": True}


def build(state: str, rep_seed: int, out_dir: str | None, verbose: int = 0) -> dict:
    """The start run's training setup at 4,194,304 through train_base's own resume path, with the replicate's seed.
    Returns the model (MirrorKLPPO), the training VecNormalize, the stage config and the checks."""
    if state not in C.STATES:
        raise ValueError(f"unknown state {state!r}")
    if rep_seed not in C.REP_SEEDS:
        raise ValueError(f"rep seed {rep_seed!r} not in {C.REP_SEEDS}")
    origin = bootstrap_clone()
    import torch

    torch.set_num_threads(1)
    from environments.shared.cli import _resolve_stage_ref
    from environments.shared.config import load_all_stages
    from environments.shared.mirror_kl_ppo import MirrorKLPPO
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.policy_loading import _ensure_sb3
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import get_species_config
    from environments.shared.symmetry import pop_symmetry_config
    from environments.shared.task_fingerprint import stage_task_fingerprint
    from environments.shared.train_base import (
        _create_or_load_model,
        _load_vecnorm_into_envs,
        _prepare_alg_kwargs,
        create_vec_env,
        eval_env_seed,
    )

    record = C.STATES[state]
    start_zip, start_pkl = record["zip"], record["pkl"]
    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    configs = load_all_stages(mid)
    stage = _resolve_stage_ref("stance", configs, mid)
    config = configs[stage]
    species_cfg = get_species_config(mid)
    plant = current_plant_identity(mid)
    task_fp = stage_task_fingerprint(mid, stage, stage_config=config, plant_identity=plant)
    checks = recipe_checks(state, stage, config, task_fp)
    sb3 = _ensure_sb3()
    gamma = config["ppo_kwargs"].get("gamma")
    # _train_stage_body's order: training env, evaluation env (seeded eval_env_seed), statistics, kwargs, model.
    train_env = create_vec_env(species_cfg, configs, stage, C.N_ENVS, rep_seed, False, algorithm="ppo", gamma=gamma,
                               plant_identity=plant)
    eval_env = create_vec_env(species_cfg, configs, stage, 1, eval_env_seed(rep_seed), use_subproc=False,
                              algorithm="ppo", gamma=gamma, plant_identity=plant)
    _load_vecnorm_into_envs(start_zip, train_env, eval_env, plant_identity=plant, task_load_mode="resume_same_stage",
                            allow_fresh_vecnorm=False, command_mode=str(config["env_kwargs"].get("command_mode", "none")),
                            vecnorm_path=start_pkl)
    log_path = Path(out_dir) if out_dir else Path(C.SCREEN) / "tests_out"
    alg_kwargs, _, _ = _prepare_alg_kwargs(config, "ppo", verbose, log_path, use_tensorboard=False)
    alg_kwargs["seed"] = int(rep_seed)
    symmetry = pop_symmetry_config(alg_kwargs)
    alg_kwargs["device"] = "cpu"
    model = _create_or_load_model(sb3, "ppo", alg_kwargs, train_env, start_zip, plant_identity=plant,
                                  task_fingerprint=task_fp, task_load_mode="resume_same_stage", symmetry=symmetry)
    eval_env.close()  # built only for the order and the statistics load; it never runs
    if type(model) is not MirrorKLPPO:
        raise SystemExit(f"the run trains MirrorKLPPO; the load gave {type(model).__name__}")
    if int(model.num_timesteps) != C.START_STEPS:
        raise SystemExit(f"the checkpoint's num_timesteps {model.num_timesteps} != {C.START_STEPS}")
    if int(model._total_timesteps) != TOTAL_TIMESTEPS:
        raise SystemExit(f"the checkpoint's _total_timesteps {model._total_timesteps} != {TOTAL_TIMESTEPS}")
    if str(model.device) != "cpu":
        raise SystemExit(f"device {model.device} != cpu")
    vecnorm = model.get_vec_normalize_env()
    assert vecnorm is train_env and vecnorm.training and vecnorm.norm_reward, (vecnorm.training, vecnorm.norm_reward)
    return {"model": model, "vecnorm": vecnorm, "config": config, "stage": stage, "checks": checks, "sb3": sb3,
            "plant": plant, "task_fingerprint": task_fp, "symmetry": symmetry, "modules_root": origin,
            "start_zip": start_zip, "start_pkl": start_pkl, "rep_seed": int(rep_seed), "state": state}


def _hash_arrays(arrays: list) -> str:
    import numpy as np

    d = hashlib.sha256()
    for name, a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        d.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return "sha256:" + d.hexdigest()


def state_digest(model: Any, vecnorm: Any) -> dict:
    """P1's digest: sha256 of the policy parameters, the Adam state and the VecNormalize statistics."""
    import numpy as np

    sd = model.policy.state_dict()
    params = [(k, sd[k].detach().cpu().numpy()) for k in sorted(sd)]
    opt = model.policy.optimizer.state_dict()
    adam = [(f"{i}.{n}", np.asarray(v[n].cpu() if hasattr(v[n], "cpu") else v[n]))
            for i, v in sorted(opt["state"].items()) for n in sorted(v)]
    vn = [("obs.mean", vecnorm.obs_rms.mean), ("obs.var", vecnorm.obs_rms.var),
          ("obs.count", np.float64(vecnorm.obs_rms.count)), ("ret.mean", vecnorm.ret_rms.mean),
          ("ret.var", vecnorm.ret_rms.var), ("ret.count", np.float64(vecnorm.ret_rms.count))]
    return {"params": _hash_arrays(params), "adam": _hash_arrays(adam), "vecnormalize": _hash_arrays(vn),
            "all": _hash_arrays(params + adam + vn), "n_updates": int(model._n_updates),
            "num_timesteps": int(model.num_timesteps)}


TRAIN_KEYS = (
    "train/tail3_bound_penalty", "train/tail3_bound_loss", "train/tail3_bound_beyond_fraction",
    "train/tail3_bound_mu_max", "train/tail3_bound_coef", "train/tail3_bound_b", "train/approx_kl",
    "train/clip_fraction", "train/policy_gradient_loss", "train/value_loss", "train/entropy_loss", "train/std",
    "train/mirror_kl_loss", "train/mirror_asymmetry", "train/learning_rate", "train/n_updates",
    "train/explained_variance",
)


def make_callbacks(model: Any, *, target: int, out_dir: str, sb3: dict, config: dict, vecnorm: Any, bound: float):
    import numpy as np
    import torch
    from environments.shared.train_base import _maybe_ent_coef_decay_callback
    from stable_baselines3.common.callbacks import BaseCallback

    tail = C.TAIL3

    class Telemetry(BaseCallback):
        """Per rollout, from the buffer before its update; the previous update's train/* values."""

        def __init__(self) -> None:
            super().__init__(0)
            self.rows: dict[str, list] = {}
            self.t0 = time.monotonic()

        def _add(self, key: str, value: Any) -> None:
            self.rows.setdefault(key, []).append(value)

        def _on_step(self) -> bool:
            return True

        def previous_update(self) -> dict:
            values = self.model.logger.name_to_value
            return {k: float(values[k]) if k in values else float("nan") for k in TRAIN_KEYS}

        def _on_rollout_end(self) -> None:
            buf = self.model.rollout_buffer
            obs = torch.as_tensor(buf.observations.reshape(-1, buf.observations.shape[-1]))
            act = buf.actions.reshape(-1, buf.actions.shape[-1]).astype(np.float64)
            with torch.no_grad():
                mu = self.model.policy.get_distribution(obs).mode().cpu().numpy().astype(np.float64)
            cmd = np.clip(act, -1.0, 1.0)
            ep = self.model.ep_info_buffer
            self._add("num_timesteps", int(self.model.num_timesteps))
            self._add("seconds", time.monotonic() - self.t0)
            self._add("ent_coef", float(self.model.ent_coef))
            self._add("learning_rate_now", float(self.model.lr_schedule(self.model._current_progress_remaining)))
            self._add("progress_remaining", float(self.model._current_progress_remaining))
            self._add("total_timesteps", int(self.model._total_timesteps))
            self._add("sample_at_clip_hi", (act >= 1.0).mean(0))
            self._add("sample_at_clip_lo", (act <= -1.0).mean(0))
            self._add("cmd_ge_0p9", (np.abs(cmd) >= 0.9).mean(0))
            self._add("cmd_ge_0p99", (np.abs(cmd) >= 0.99).mean(0))
            self._add("cmd_mean", cmd.mean(0))
            self._add("mu_mean", mu.mean(0))
            self._add("mu_beyond", (np.abs(mu) > 1.0).mean(0))
            self._add("mu_gt_0p60", (mu > 0.60).mean(0))
            self._add("mu_gt_0p90", (mu > 0.90).mean(0))
            self._add("mu_tail3_q", np.quantile(mu[:, tail], [0.05, 0.25, 0.5, 0.75, 0.95]))
            self._add("tail3_mu_gt_0p60", float((mu[:, tail] > 0.60).mean()))
            self._add("tail3_mu_gt_0p90", float((mu[:, tail] > 0.90).mean()))
            self._add("tail3_mu_gt_1", float((mu[:, tail] > 1.0).mean()))
            self._add("tail3_bound_penalty_buffer", float(np.mean(np.maximum(mu[:, tail] - bound, 0.0) ** 2)))
            self._add("log_std", self.model.policy.log_std.detach().cpu().numpy().astype(np.float64).copy())
            self._add("ep_rew_mean", float(np.mean([e["r"] for e in ep])) if ep else float("nan"))
            self._add("ep_len_mean", float(np.mean([e["l"] for e in ep])) if ep else float("nan"))
            prev = self.previous_update()
            for k, v in prev.items():
                self._add("prev_" + k.replace("/", "_"), v)
            last = {"num_timesteps": int(self.model.num_timesteps), "seconds": round(time.monotonic() - self.t0, 1),
                    "tail3_mu_mean": round(float(mu[:, tail].mean()), 4),
                    "tail3_mu_gt_b": round(float((mu[:, tail] > bound).mean()), 4),
                    "tail3_cmd_ge_0p99": round(float((np.abs(cmd[:, tail]) >= 0.99).mean()), 4),
                    "prev_tail3_bound_loss": prev["train/tail3_bound_loss"],
                    "prev_tail3_bound_penalty": prev["train/tail3_bound_penalty"],
                    "prev_approx_kl": prev["train/approx_kl"]}
            print("[rollout] " + json.dumps(last), flush=True)

        def save(self, path: str) -> None:
            arrays = {k: np.asarray(v) for k, v in self.rows.items()}
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)

    class StopAndSave(BaseCallback):
        """After exactly `target` steps' updates: save stage1_final.* (the state after the last update) and stop."""

        def __init__(self, telemetry: Telemetry) -> None:
            super().__init__(0)
            self.stop = False
            self.telemetry = telemetry
            self.final_update: dict | None = None
            self.final_digest: dict | None = None

        def _on_rollout_start(self) -> None:
            if self.model.num_timesteps >= target and not self.stop:
                self.final_update = self.telemetry.previous_update()
                self.model.logger.dump(self.model.num_timesteps)  # progress.csv gets the last update's train/* row
                self.final_digest = state_digest(self.model, vecnorm)
                self.model.save(os.path.join(out_dir, "stage1_final.zip"))
                vecnorm.save(os.path.join(out_dir, "stage1_final_vecnorm.pkl"))
                self.stop = True

        def _on_step(self) -> bool:
            if self.n_calls % (C.SAVE_EVERY // C.N_ENVS) == 0:
                self.telemetry.save(os.path.join(out_dir, "telemetry.npz"))
            return not self.stop

    telemetry = Telemetry()
    stopper = StopAndSave(telemetry)
    checkpoint = sb3["CheckpointCallback"](save_freq=C.SAVE_EVERY // C.N_ENVS, save_path=out_dir,
                                           name_prefix="stage1", save_vecnormalize=True)
    ent = _maybe_ent_coef_decay_callback(config, "ppo", int(config["curriculum_kwargs"]["timesteps"]))
    if ent is None:
        raise SystemExit("the stance [ppo] declares an entropy decay; none was built")
    return [checkpoint, ent, telemetry, stopper], telemetry, stopper, ent


def _write_meta(path: str, meta: dict, mode: str) -> None:
    with open(path, mode) as fh:
        json.dump(norm(meta), fh, indent=1)


def train(ns: argparse.Namespace, root: Path) -> int:
    """The replicate (after preflight passed).  Never called by the build.

    The model is built (recipe check, start load) before runs/<name> is created, so a refusal on the replicate's own
    records leaves nothing behind (exit 2); once the folder exists, any failure is a crash (exit 1)."""
    t_start = time.monotonic()
    name = run_name(ns.state, ns.arm, ns.rep_seed)
    out_dir = str(root / "runs" / name)
    with open(root / "calib" / "coef.json") as fh:
        coef_json = json.load(fh)
    coef, bound = arm_config(ns.arm, coef_json)
    git = clone_state(C.CLONE)
    bootstrap_clone()
    import numpy as np
    import torch
    from environments.shared import mirror_kl_ppo as MK
    from stable_baselines3.common.logger import configure

    try:
        built = build(ns.state, ns.rep_seed, out_dir)
    except SystemExit as exc:
        print(json.dumps({"preflight": "refused", "problems": [f"build: {exc}"]}), flush=True)
        return 2
    model, vecnorm, config = built["model"], built["vecnorm"], built["config"]
    os.makedirs(out_dir)  # refused if it exists (preflight checked it too)
    meta_path = os.path.join(out_dir, "run_meta.json")
    meta: dict[str, Any] = {"schema": RUN_SCHEMA, "name": name, "status": "starting", "utc_start": utc_now()}
    try:
        config_bound = MK.Tail3UpperBoundConfig(coef, bound)
        MK.enable_tail3_upper_bound(model, config_bound)
        model.set_logger(configure(out_dir, ["csv"]))
        target = C.START_STEPS + ns.steps
        callbacks, telemetry, stopper, ent = make_callbacks(model, target=target, out_dir=out_dir, sb3=built["sb3"],
                                                             config=config, vecnorm=vecnorm, bound=bound)
        start_digest = state_digest(model, vecnorm)
        learn_budget = TOTAL_TIMESTEPS - C.START_STEPS
        meta.update({
            "state": ns.state, "arm": ns.arm, "rep_seed": ns.rep_seed, "coef": coef, "b": bound,
            "bound_record": config_bound.checkpoint_record(), "steps": ns.steps, "target_num_timesteps": target,
            "save_every": C.SAVE_EVERY, "argv": sys.argv,
            "clone": {"path": C.CLONE, "head": git["head"], "status": git["status"], "branch": git["branch"],
                      "push_url": git["push_url"], "parent": git["parent"], "commits": git["commits"],
                      "chain_ok": git["chain_ok"], "changed_files": git["changed_files"],
                      "status_flags": "--porcelain --ignored"},
            "screen_common_sha256": sha256_file(TOOLS / "screen_common.py"),
            "modules_root": built["modules_root"], "stack": C.stack_versions(),
            "torch_threads": torch.get_num_threads(), "device": str(model.device),
            "criteria_sha256": sha256_file(root / "criteria.txt"),
            "coef_json_sha256": sha256_file(root / "calib" / "coef.json"),
            "approved_sha256": sha256_file(root / "APPROVED"),
            "start": {"zip": built["start_zip"], "zip_sha256": sha256_file(built["start_zip"]),
                      "pkl": built["start_pkl"], "pkl_sha256": sha256_file(built["start_pkl"]),
                      "num_timesteps": int(model.num_timesteps), "total_timesteps_recorded": int(model._total_timesteps),
                      "n_updates": int(model._n_updates),
                      "current_progress_remaining": float(model._current_progress_remaining),
                      "ent_coef_after_load": float(model.ent_coef), "digest": start_digest},
            "recipe_checks": built["checks"], "class": type(model).__name__,
            "symmetry": getattr(model, "mesozoic_symmetry", None),
            "ent_coef_decay": {"end": ent.end_value, "decay_timesteps": ent.decay_timesteps},
            "learn": {"total_timesteps_arg": learn_budget, "reset_num_timesteps": False},
            "status": "running", "numpy": np.__version__,
        })
        _write_meta(meta_path, meta, "x")
        print(json.dumps({"name": name, "coef": coef, "b": bound, "target": target, "out_dir": out_dir}), flush=True)
        model.learn(total_timesteps=learn_budget, callback=built["sb3"]["CallbackList"](callbacks),
                    reset_num_timesteps=False, progress_bar=False)
        if not stopper.stop:
            raise RuntimeError(f"training ended at {model.num_timesteps} without reaching {target}")
        telemetry.save(os.path.join(out_dir, "telemetry.npz"))
        meta.update({
            "status": "done", "utc_end": utc_now(), "final_num_timesteps_env": int(model.num_timesteps),
            "final_checkpoint_num_timesteps": target, "final_update_train": stopper.final_update,
            "final_digest": stopper.final_digest,
            "checkpoints": sorted(f for f in os.listdir(out_dir) if f.endswith((".zip", ".pkl"))),
            "final_sha256": {f: sha256_file(os.path.join(out_dir, f))
                             for f in ("stage1_final.zip", "stage1_final_vecnorm.pkl")},
            "modules_all_under_clone": modules_under_clone(), "wall_seconds": round(time.monotonic() - t_start, 1),
        })
        _write_meta(meta_path, meta, "w")
        print(json.dumps({"done": True, "name": name, "digest": stopper.final_digest["all"],
                          "wall_seconds": meta["wall_seconds"]}), flush=True)
        return 0
    except BaseException as exc:  # a crash: the folder stays for the crash rule
        meta.update({"status": "crashed", "utc_end": utc_now(), "error": repr(exc),
                     "traceback": traceback.format_exc(), "wall_seconds": round(time.monotonic() - t_start, 1)})
        try:
            _write_meta(meta_path, meta, "w")
        except Exception:
            pass
        traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rep-seed", dest="rep_seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--preflight-only", action="store_true", help="check only; never builds a model or a folder")
    ns = ap.parse_args(argv)
    root = Path(C.SCREEN)
    problems = preflight(ns, root)
    verdict = {"preflight": "pass" if not problems else "refused", "problems": problems}
    print(json.dumps(verdict, indent=1), flush=True)
    if problems:
        return 2
    if ns.preflight_only:
        return 0
    return train(ns, root)


if __name__ == "__main__":
    sys.exit(main())
