"""ECS: one continuation from a stance6 F2-held checkpoint pair, or a validation run that makes no optimizer step.

    PY -I -B run_ecs.py --start s42|s80 --arm A|B --seed N [--preflight-only]
    PY -I -B run_ecs.py --start s42|s80 --arm A|B --seed N --save-loaded-only [--out DIR]      (V1; no learn())
    PY -I -B run_ecs.py --start s42|s80 --arm A|B --seed N --no-update --until STEPS [--out DIR]  (V2; no train())

Build list v2 items 3-5 (BL; criteria_ecs_v2.txt = CRIT; survey_harness.txt = HS; TB = <code>/environments/shared/
train_base.py).  Ported from D-D43's tools/run_tailscreen.py where it fits (preflight, bootstrap, state digest,
run_meta); the body is the run branch's own same-stage resume, train_base._train_stage_body (T3), not a
re-implementation.

Preflight (exit 2, nothing created; every reason listed): the arguments in their sets (start, arm, the arm's seed
list, the mode's options); for a continuation <ECS>/APPROVED (JSON, ecs_common.APPROVED_KEYS) present, its
criteria_sha256 equal to sha256(<ECS>/criteria.txt), its code_tree_digest the pinned one and its tool_sha256 equal to
every ECS tool's sha256; the run folder absent; the F2 start pair, the five parent records and the parent's
stage_config.json at their ecs_common sha256; the code copy's seal (M1's r10m_common.check_copy: commit 8737cd3d, tree
digest 331fe8fc..., 1,406 files); the stack; the M1/M5 pins; disk at or above 3 GiB free.  The no-optimizer-step modes
write only under <ECS>/validate/ (default validate/v1 or validate/v2) and need no APPROVED (D2 (a)).

The run (folder <root>/<start>_<arm>_<seed>/01_stance/): the start pair byte-copied under the resume path's periodic
names (models/stage1_<S>_steps.zip, stage1_vecnormalize_<S>_steps.pkl) and the five parent records under their real
names, each copy's sha256 checked; the environment pinned before any import of the code (OMP/MKL/OPENBLAS threads 1,
TMPDIR and MPLCONFIGDIR under <ECS>/tmp/<name>, GITHUB_SHA the copy's commit, the copy first on sys.path and refused if
`environments` imports from elsewhere); the copy's own config loader for the T. rex stance stage, deep-copied, with
arm B's in-memory edit of [ppo] ent_coef and ent_coef_end to 0.0 (D1); stage_config.json written by the copy's own
save_stage_config as train() writes it (TB l1838-l1855) and compared with the parent's stage_config.json.copy1 (V5,
recipe_problems) before learn(); then _train_stage_body with every keyword argument named (TB l1249-l1282).  A wrapper
around train_base._create_or_load_model (looked up at call time, TB l1395) checks the loaded model (MirrorKLPPO,
num_timesteps S, _total_timesteps 12,000,000, device cpu, ent_coef 0.005 or 0.0, the training env's obs_rms and
ret_rms equal to the start pkl's, torch's intra-op threads 1) and refuses otherwise (exit 2: no optimizer step has
been made).  The stage_callback factory refuses (exit 2, before learn()) unless the wrapper ran (and, in --no-update,
installed the no-op train).  StopAtEnd (the stage_callback) returns False at num_timesteps == E (or --until), after
every other callback ran on that step, so the evaluation and the checkpoint pair at E exist and no update is made on
the rollout ending at E (HS 4.3).  At each rollout's end (before its update) it rewrites run_meta's heartbeat
{rollout_end}; a crash records last_step, the last num_timesteps it saw (the runner's early-crash rule).

--save-loaded-only (V1): after the wrapper's checks, the loaded policy and statistics are saved by the run's own
checkpoint callback class (SB3 CheckpointCallback, as TB l846 builds it) to <run>/saved/, digests compared with the
start pair's, and the process exits before learn().  --no-update (V2): the model's train() is replaced by a function
that counts its calls and returns; at the end the parameter and Adam digests must equal the start's (else exit 1).

run_meta.json (ecs_common RUN_META keys) is written with "x" when the folder is made and rewritten (atomically) at
each rollout's end and at the end.
Exit codes: 0 done; 2 refused (preflight, including an unexpected error inside it, or a check before learn(): no
optimizer step); 5 finished but a post-run check failed (status "failed": the code seal or the modules' origin after
the run, or in the no-optimizer-step modes the digests; the runner checks it and CHECK fails); 1 crash (the folder
stays).
Importing this module needs neither torch nor the code copy (preflight and the checks are pure).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any import from tools/ or the copy (-I ignores PYTHONDONTWRITEBYTECODE)

import argparse
import copy
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

TOOLS = Path(os.path.realpath(__file__)).parent


def _load(name: str, path: str | os.PathLike[str] | None = None) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path or TOOLS / f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load("ecs_common")

VERSION = "run_ecs 1.0 (2026-10-28; ECS build list v2 items 3-5)"
LINEAGE_KEYS = ("load_path", "load_mode", "parent_checkpoint_sha256", "parent_task_sha256", "parent_run_id")
RUN_KEYS_EXPECTED_TO_DIFFER = frozenset({"seed", "timesteps", "label", "duration_seconds", "hyperparameters_sha256",
                                         *LINEAGE_KEYS})
TOP_KEYS_EXPECTED_TO_DIFFER = frozenset({"gpu"})  # compared separately: run, hyperparameters
MISSING = object()


class Refused(Exception):
    """A check before learn() failed: nothing trained; exit 2."""


class _LoadedOnly(Exception):
    """--save-loaded-only: the loaded pair is saved; leave the body before learn()."""


# ---------------------------------------------------------------------------------------------------- pure helpers
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | os.PathLike[str]) -> str:
    return C.sha256_file(path)


def norm(x: Any) -> Any:
    """JSON round trip (tuples -> lists, numpy scalars -> numbers) for records."""
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def ecs_root() -> Path:
    return Path(C.ECS)


def mode_of(ns: Any) -> str:
    if getattr(ns, "save_loaded_only", False):
        return "save_loaded_only"
    if getattr(ns, "no_update", False):
        return "no_update"
    return "continuation"


def run_root(ns: Any) -> Path:
    """<ECS>/runs for a continuation; --out, else <ECS>/validate/v1 or v2, for the no-optimizer-step modes."""
    mode = mode_of(ns)
    if mode == "continuation":
        return ecs_root() / "runs"
    if getattr(ns, "out", None):
        return Path(os.path.abspath(ns.out))
    return ecs_root() / "validate" / ("v1" if mode == "save_loaded_only" else "v2")


def target_of(ns: Any) -> int:
    """The stop step: E for a continuation, --until for --no-update, S for --save-loaded-only."""
    S, E = C.window(ns.start)
    mode = mode_of(ns)
    if mode == "no_update":
        return int(ns.until)
    if mode == "save_loaded_only":
        return S
    return E


def arm_config(config: dict, arm: str) -> dict:
    """A deep copy of the stage config with the arm's [ppo] edit (D1): A none; B ent_coef = ent_coef_end = 0.0."""
    if arm not in C.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    out = copy.deepcopy(config)
    ppo = out.setdefault("ppo_kwargs", {})
    for key, value in C.ARM_PPO_EDIT[arm].items():
        if key not in ppo:
            raise ValueError(f"the stage's [ppo] has no {key!r} to edit")
        ppo[key] = value
    return out


def recipe_problems(written: dict, parent: dict, *, arm: str, S: int, seed: int, label: str, load_path: str,
                    zip_sha256: str, hp_sha256_expected: str) -> list[str]:
    """V5: the stage_config.json the harness wrote against the parent's stage_config.json.copy1.  Every key equal
    except: hyperparameters (arm B: exactly ent_coef and ent_coef_end, both 0.0); run.seed (= seed), run.timesteps
    (= 12,000,000 - S), run.label (= label), run.duration_seconds, run.hyperparameters_sha256 (= the expected
    digest: 054ab483... for A, the recomputed one for B), the load-lineage keys (load_path = the periodic zip,
    load_mode resume_same_stage, parent_checkpoint_sha256 = "sha256:" + the start zip's, parent_task_sha256 = the
    parent's
    task_sha256, no parent_run_id); gpu.  Empty means pass."""
    problems: list[str] = []
    if not isinstance(written, dict) or not isinstance(parent, dict):
        return ["stage_config: not JSON objects"]
    for key in sorted((set(written) | set(parent)) - {"run", "hyperparameters"} - TOP_KEYS_EXPECTED_TO_DIFFER):
        if written.get(key, MISSING) != parent.get(key, MISSING):
            problems.append(f"stage_config.{key} differs from the parent's")
    if written.get("git_commit") != C.COMMIT:
        problems.append(f"stage_config.git_commit {written.get('git_commit')!r} != {C.COMMIT}")
    hw, hp = written.get("hyperparameters"), parent.get("hyperparameters")
    if not isinstance(hw, dict) or not isinstance(hp, dict):
        problems.append("stage_config.hyperparameters missing")
    else:
        expected = dict(hp)
        for key, value in C.ARM_PPO_EDIT[arm].items():
            if key not in expected:
                problems.append(f"the parent's hyperparameters have no {key!r}")
            expected[key] = value
        for key in sorted(set(hw) | set(expected)):
            if hw.get(key, MISSING) != expected.get(key, MISSING):
                problems.append(f"stage_config.hyperparameters.{key} {hw.get(key, 'absent')!r} != "
                                f"{expected.get(key, 'absent')!r}")
    rw, rp = written.get("run"), parent.get("run")
    if not isinstance(rw, dict) or not isinstance(rp, dict):
        return problems + ["stage_config.run missing"]
    for key in sorted((set(rw) | set(rp)) - RUN_KEYS_EXPECTED_TO_DIFFER):
        if rw.get(key, MISSING) != rp.get(key, MISSING):
            problems.append(f"stage_config.run.{key} {rw.get(key, 'absent')!r} != the parent's "
                            f"{rp.get(key, 'absent')!r}")
    want = {"seed": int(seed), "timesteps": C.TOTAL_TIMESTEPS - int(S), "label": label,
            "hyperparameters_sha256": hp_sha256_expected, "load_path": load_path, "load_mode": "resume_same_stage",
            "parent_checkpoint_sha256": zip_sha256 if zip_sha256.startswith("sha256:") else f"sha256:{zip_sha256}"}
    for key, value in want.items():
        if rw.get(key, MISSING) != value:
            problems.append(f"stage_config.run.{key} {rw.get(key, 'absent')!r} != {value!r}")
    if arm == "A" and hp_sha256_expected != C.HYPERPARAMETERS_SHA256_A:
        problems.append(f"arm A's expected digest {hp_sha256_expected} != {C.HYPERPARAMETERS_SHA256_A}")
    if arm == "B" and hp_sha256_expected == C.HYPERPARAMETERS_SHA256_A:
        problems.append("arm B's expected digest equals arm A's")
    task_p = (parent.get("task_fingerprint") or {}).get("task_sha256")
    if "parent_task_sha256" in rw and rw["parent_task_sha256"] != task_p:
        problems.append(f"stage_config.run.parent_task_sha256 {rw['parent_task_sha256']!r} != {task_p!r}")
    if rw.get("parent_run_id", None) is not None:
        problems.append(f"stage_config.run.parent_run_id {rw['parent_run_id']!r} recorded (a plain load records none)")
    dur = rw.get("duration_seconds", 0.0)
    if isinstance(dur, bool) or not isinstance(dur, (int, float)) or dur < 0:
        problems.append(f"stage_config.run.duration_seconds {dur!r} is not a non-negative number")
    return problems


def stats_problems(env_stats: Any, pkl_stats: Any) -> list[str]:
    """The training env's obs_rms and ret_rms (mean, var, count) against the start pkl's, exactly."""
    import numpy as np

    problems = []
    for rms in ("obs_rms", "ret_rms"):
        a, b = getattr(env_stats, rms, None), getattr(pkl_stats, rms, None)
        if a is None or b is None:
            problems.append(f"{rms} missing")
            continue
        for field in ("mean", "var", "count"):
            x, y = np.asarray(getattr(a, field)), np.asarray(getattr(b, field))
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y):
                problems.append(f"training env {rms}.{field} differs from the start pkl's")
    return problems


def loaded_problems(model: Any, *, mirror_cls: type, S: int, arm: str, train_env: Any,
                    threads: int | None = None) -> list[str]:
    """The post-load checks of BL 3 (the statistics are compared by stats_problems); threads: torch's intra-op
    thread count at the load (None: not checked)."""
    problems = []
    if threads is not None and threads != C.TORCH_THREADS:
        problems.append(f"torch intra-op threads {threads} != {C.TORCH_THREADS}")
    if type(model) is not mirror_cls:
        problems.append(f"the load gave {type(model).__name__}, not {mirror_cls.__name__}")
    if int(getattr(model, "num_timesteps", -1)) != int(S):
        problems.append(f"num_timesteps {getattr(model, 'num_timesteps', None)} != S {S}")
    if int(getattr(model, "_total_timesteps", -1)) != C.TOTAL_TIMESTEPS:
        problems.append(f"_total_timesteps {getattr(model, '_total_timesteps', None)} != {C.TOTAL_TIMESTEPS}")
    if str(getattr(model, "device", None)) != "cpu":
        problems.append(f"device {getattr(model, 'device', None)} != cpu")
    want = C.ARM_ENT_COEF_AFTER_LOAD[arm]
    got = getattr(model, "ent_coef", None)
    if isinstance(got, bool) or not isinstance(got, (int, float)) or float(got) != want:
        problems.append(f"ent_coef after load {got!r} != arm {arm}'s {want!r}")
    vecnorm = model.get_vec_normalize_env() if hasattr(model, "get_vec_normalize_env") else None
    if vecnorm is None or vecnorm is not train_env:
        problems.append("the model's VecNormalize is not the training env")
    elif not (getattr(vecnorm, "training", False) and getattr(vecnorm, "norm_reward", False)):
        problems.append("the training VecNormalize does not update (training / norm_reward off)")
    return problems


def no_update_problems(start: dict, end: dict, *, train_calls: int, expected_calls: int) -> list[str]:
    """--no-update: parameters and Adam state unchanged; train() reached once per completed rollout."""
    problems = []
    for key in ("params", "adam"):
        if start.get(key) is None or start.get(key) != end.get(key):
            problems.append(f"{key} digest changed: {start.get(key)} -> {end.get(key)}")
    if train_calls != expected_calls:
        problems.append(f"train() reached {train_calls} times, expected {expected_calls}")
    return problems


def post_run_status(no_update: list[str], outside: list[str], seal_after: list[str]) -> tuple[str, int]:
    """(status, exit) after a run reached its end.  A failed post-run check is not a crash (review H3): "failed" with
    EXIT_POST_CHECK, which the runner checks (CHECK then fails: VOID) instead of rerunning it."""
    if no_update or outside or seal_after:
        return "failed", C.EXIT_POST_CHECK
    return "done", C.EXIT_OK


def stop_reason(*, flag: bool, T: int, target: int, collapse_tripped: bool) -> str:
    """"end" (StopAtEnd's flag, at the target), "collapse" (the collapse callback's own state, before the target),
    or "other"."""
    if flag and T == target:
        return "end"
    if collapse_tripped and T < target:
        return "collapse"
    return "other"


def install_no_update(model: Any, holder: dict) -> None:
    """--no-update (BL 5): the model's train() becomes a method that counts its calls and returns; learn() calls
    self.train() (OPA l337), so no optimizer step can follow.  Done by a subclass swapped in as the model's class (the
    idiom enable_mirror_kl uses, mirror_kl_ppo.py l471), not by an instance attribute: SB3's save serialises every
    instance attribute into the zip's data (save_util.data_to_json), and a function there was refused at the final
    save in lane H's first no-update smoke (TypeError: cannot pickle '_thread.lock' object)."""
    holder["train_calls"] = 0
    base = type(model)

    def train(self: Any) -> None:
        holder["train_calls"] += 1

    model.__class__ = type(f"NoUpdate{base.__name__}", (base,), {"train": train, "__module__": base.__module__})
    holder["no_update_class"] = model.__class__


def torch_threads() -> tuple[int, int]:
    """(intra-op, inter-op) thread counts of the running torch (imported lazily: the module needs no torch)."""
    import torch

    return int(torch.get_num_threads()), int(torch.get_num_interop_threads())


def make_checked_load(original: Any, *, mirror_cls: type, S: int, arm: str, start_stats: Any, mode: str,
                      holder: dict, save_loaded: Any = None) -> Any:
    """The thin wrapper around train_base._create_or_load_model (BL 3): the load as the body makes it, then the
    checks (Refused on any failure, before learn()), the start digests into `holder`, and the mode's step."""

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        train_env = args[3] if len(args) > 3 else kwargs.get("train_env")
        intra, interop = torch_threads()
        holder["torch_threads"], holder["torch_interop_threads"] = intra, interop
        problems = loaded_problems(model, mirror_cls=mirror_cls, S=S, arm=arm, train_env=train_env, threads=intra)
        problems += stats_problems(train_env, start_stats)
        got = getattr(model, "ent_coef", None)
        holder["ent_coef_after_load"] = float(got) if isinstance(got, (int, float)) else repr(got)
        holder["digest_start"] = state_digest(model, train_env)
        if problems:
            raise Refused("load check: " + " | ".join(problems))
        if mode == "save_loaded_only":
            holder["saved"] = save_loaded(model)
            raise _LoadedOnly()
        if mode == "no_update":
            install_no_update(model, holder)
        holder["model_class"] = type(model)
        return model

    return checked_load


def wrapper_problems(holder: dict, *, mode: str, update_classes: tuple = ()) -> list[str]:
    """Before learn() (review I1): the load wrapper ran and, in --no-update, the model's class is the no-op train's.
    update_classes: the classes whose train() makes an optimizer step (MirrorKLPPO, PPO in a run)."""
    problems = []
    if "digest_start" not in holder or "model_class" not in holder:
        problems.append("the load wrapper did not run before the callbacks were built (TB._create_or_load_model "
                        "bypassed)")
    elif mode == "no_update":
        cls = holder["model_class"]
        if cls is not holder.get("no_update_class") or any(cls.train is u.train for u in update_classes):
            problems.append(f"--no-update: the model's class {cls.__name__} does not carry the no-op train()")
    return problems


def make_stop_at_end_class(base_callback: type) -> type:
    """StopAtEnd (G2) on a given BaseCallback class (SB3's in a run; the same class in the tests)."""

    class StopAtEnd(base_callback):  # type: ignore[misc, valid-type]
        """_on_step returns False when num_timesteps == target and sets `flag`; refuses a step past the target.
        At training start it records the callback list it runs in (read only), so the run can read the collapse
        callback's state at the end.  last_step is the last num_timesteps it saw (every other callback ran on that
        step before it); heartbeat(n), if given, is called at each rollout's end (before that rollout's update) and
        never changes what training does."""

        def __init__(self, target: int, heartbeat: Any = None) -> None:
            super().__init__(0)
            self.target = int(target)
            self.flag = False
            self.step: int | None = None
            self.last_step: int | None = None
            self.siblings: list[Any] = []
            self.calls = 0
            self.heartbeat = heartbeat

        def _on_training_start(self) -> None:
            self.siblings = list(flatten_callbacks(self.locals.get("callback")))

        def _on_rollout_end(self) -> None:
            if self.heartbeat is not None:
                self.heartbeat(int(self.model.num_timesteps))

        def _on_step(self) -> bool:
            self.calls += 1
            n = int(self.model.num_timesteps)
            self.last_step = n
            if n > self.target:
                raise RuntimeError(f"StopAtEnd: num_timesteps {n} passed the end step {self.target}")
            if n == self.target:
                self.flag = True
                self.step = n
                return False
            return True

    return StopAtEnd


def flatten_callbacks(cb: Any) -> list[Any]:
    out: list[Any] = []
    if cb is None:
        return out
    inner = getattr(cb, "callbacks", None)
    if isinstance(inner, list):
        for c in inner:
            out.extend(flatten_callbacks(c))
    else:
        out.append(cb)
    return out


def collapse_state(callbacks: list[Any]) -> dict:
    """The collapse early stop's own counters (early_stopping.py l254-l261), read only."""
    found = [c for c in callbacks if type(c).__name__ == "EvalCollapseEarlyStopCallback"]
    if len(found) != 1:
        return {"found": len(found), "consecutive_drops": None, "patience": None, "tripped": False}
    c = found[0]
    drops, patience = int(getattr(c, "_consecutive_drops", 0)), int(getattr(c, "patience", 0))
    return {"found": 1, "consecutive_drops": drops, "patience": patience, "tripped": patience > 0 and drops >= patience}


def _hash_arrays(arrays: list) -> str:
    import numpy as np

    d = hashlib.sha256()
    for name, a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        d.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return "sha256:" + d.hexdigest()


def params_arrays(state_dict: dict) -> list:
    return [(k, state_dict[k].detach().cpu().numpy()) for k in sorted(state_dict)]


def adam_arrays(opt_state: dict) -> list:
    import numpy as np

    out = []
    for i, v in sorted(opt_state.get("state", {}).items()):
        for n in sorted(v):
            x = v[n]
            out.append((f"{i}.{n}", np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)))
    return out


def vecnorm_arrays(vecnorm: Any) -> list:
    import numpy as np

    return [("obs.mean", vecnorm.obs_rms.mean), ("obs.var", vecnorm.obs_rms.var),
            ("obs.count", np.float64(vecnorm.obs_rms.count)), ("ret.mean", vecnorm.ret_rms.mean),
            ("ret.var", vecnorm.ret_rms.var), ("ret.count", np.float64(vecnorm.ret_rms.count))]


def state_digest(model: Any, vecnorm: Any) -> dict:
    """D-D43's digest: sha256 of the policy parameters, the Adam state and the VecNormalize statistics."""
    params = params_arrays(model.policy.state_dict())
    adam = adam_arrays(model.policy.optimizer.state_dict())
    vn = vecnorm_arrays(vecnorm) if vecnorm is not None else []
    return {"params": _hash_arrays(params), "adam": _hash_arrays(adam), "vecnormalize": _hash_arrays(vn),
            "all": _hash_arrays(params + adam + vn), "n_updates": int(getattr(model, "_n_updates", 0)),
            "num_timesteps": int(model.num_timesteps)}


def zip_digests(zip_path: str | os.PathLike[str]) -> dict:
    """Parameter and Adam digests of an SB3 zip's policy.pth / policy.optimizer.pth (torch.load, weights only)."""
    import io
    import zipfile

    import torch

    with zipfile.ZipFile(zip_path) as z:
        pol = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
    return {"params": _hash_arrays(params_arrays(pol)), "adam": _hash_arrays(adam_arrays(opt))}


def pkl_stats(pkl_path: str | os.PathLike[str]) -> Any:
    """The start pkl's VecNormalize object (the run's own file, F2- and sha256-checked; the code unpickles it too)."""
    import pickle

    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - a pinned file of the run, read by the code's own loader as well


# ---------------------------------------------------------------------------------------------------- preflight
def free_bytes(path: str | os.PathLike[str]) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        with open(path) as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as exc:
        return None, f"does not parse ({exc.__class__.__name__})"


def approval_problems() -> list[str]:
    """<ECS>/APPROVED against <ECS>/criteria.txt, the code digest and the ECS tools' sha256."""
    root = ecs_root()
    approved, err = _read_json(root / "APPROVED")
    if err:
        return [f"APPROVED: {err}"]
    if not isinstance(approved, dict):
        return ["APPROVED: not a JSON object"]
    problems = []
    missing = [k for k in C.APPROVED_KEYS if k not in approved]
    if missing:
        problems.append(f"APPROVED: missing keys {missing}")
    crit = root / "criteria.txt"
    if not crit.is_file():
        problems.append("APPROVED: <ECS>/criteria.txt missing")
    elif sha256_file(crit) != approved.get("criteria_sha256"):
        problems.append(f"APPROVED: criteria_sha256 {approved.get('criteria_sha256')!r} != sha256(criteria.txt)")
    if approved.get("code_tree_digest") != C.TREE_DIGEST:
        problems.append(f"APPROVED: code_tree_digest {approved.get('code_tree_digest')!r} != {C.TREE_DIGEST}")
    pins = approved.get("tool_sha256") if isinstance(approved.get("tool_sha256"), dict) else {}
    for name in C.TOOL_FILES:
        path = TOOLS / name
        got = sha256_file(path) if path.is_file() else None
        if not pins.get(name) or pins.get(name) != got:
            problems.append(f"APPROVED: tool_sha256[{name!r}] {pins.get(name)!r} != the file's {got!r}")
    return problems


def seal_problems() -> list[str]:
    """The code copy's seal by M1's own r10m_common.check_copy (loaded only at its pinned sha256)."""
    common = C.M1_COMMON
    try:
        if sha256_file(common) != C.PIN_SHA256[common]:
            return [f"code: {common} is not at its pin; the seal was not checked"]
    except OSError as exc:
        return [f"code: {common}: {exc}"]
    rc = _load("ecs_r10m_common", common)
    try:
        seal = rc.check_copy(C.CODE, expect_commit=C.COMMIT)
    except rc.Refusal as exc:  # a SystemExit subclass carrying .message
        return [f"code: {getattr(exc, 'message', exc)}"]
    problems = []
    if seal.get("tree_digest") != C.TREE_DIGEST or seal.get("files") != C.TREE_FILES:
        problems.append(f"code: seal {seal.get('tree_digest')} ({seal.get('files')} files) != pinned "
                        f"{C.TREE_DIGEST} ({C.TREE_FILES} files)")
    return problems


def input_problems(start: str) -> list[str]:
    """The F2 start pair, the five parent records and the parent's stage_config.json at their pins."""
    st = C.STARTS[start]
    items = [(C.f2_file(start, st["zip_rel"]), st["zip_sha256"]), (C.f2_file(start, st["pkl_rel"]), st["pkl_sha256"])]
    items += [(C.f2_record(start, n), st["records"][n]) for n in (*C.PARENT_RECORDS, "stage_config.json")]
    problems = []
    for path, want in items:
        if not os.path.isfile(path):
            problems.append(f"input: {path} missing")
        elif sha256_file(path) != want:
            problems.append(f"input: sha256 of {path} != {want}")
    return problems


def preflight(ns: Any) -> list[str]:
    """Every reason this run may not start; empty means pass.  Reads only; creates nothing."""
    problems: list[str] = []
    start, arm, seed = getattr(ns, "start", None), getattr(ns, "arm", None), getattr(ns, "seed", None)
    mode = mode_of(ns)
    args_ok = True
    if start not in C.STARTS:
        problems.append(f"args: start {start!r} not in {sorted(C.STARTS)}")
        args_ok = False
    if arm not in C.ARMS:
        problems.append(f"args: arm {arm!r} not in {list(C.ARMS)}")
        args_ok = False
    if isinstance(seed, bool) or not isinstance(seed, int) or (args_ok and not C.seed_ok(start, arm, seed)):
        problems.append(f"args: seed {seed!r} not in arm {arm}'s list at {start}")
        args_ok = False
    if getattr(ns, "save_loaded_only", False) and getattr(ns, "no_update", False):
        problems.append("args: --save-loaded-only and --no-update exclude each other")
    until = getattr(ns, "until", None)
    if mode == "no_update":
        if until is None:
            problems.append("args: --no-update needs --until")
        elif start in C.STARTS:
            S, E = C.window(start)
            if isinstance(until, bool) or not isinstance(until, int) or not S < until <= E or (until - S) % C.ROLLOUT:
                problems.append(f"args: --until {until!r} not in (S, E] = ({S}, {E}] on the {C.ROLLOUT}-step grid")
    elif until is not None:
        problems.append("args: --until is only for --no-update")
    if mode == "continuation":
        if getattr(ns, "out", None):
            problems.append("args: --out is only for the no-optimizer-step modes")
        problems.extend(approval_problems())
    else:
        root = os.path.realpath(run_root(ns))
        validate = os.path.realpath(ecs_root() / "validate")
        if root != validate and not root.startswith(validate + os.sep):
            problems.append(f"args: {root} is not under {validate} (the no-optimizer-step modes write only there)")
    if args_ok:
        path = run_root(ns) / C.run_name(start, arm, seed)
        if os.path.lexists(path):
            problems.append(f"runs: {path} exists")
        problems.extend(input_problems(start))
    problems.extend(seal_problems())
    problems.extend(f"stack: {p}" for p in C.check_stack())
    problems.extend(f"pins: {p}" for p in C.check_pins())
    free = free_bytes(ecs_root())
    if free < C.DISK_FLOOR_BYTES:
        problems.append(f"disk: {free} bytes free < the floor {C.DISK_FLOOR_BYTES}")
    return problems


# ---------------------------------------------------------------------------------------------------- the run
def setup_environment(name: str) -> dict:
    """Before any import of numpy, torch or the code (BL 3)."""
    tmp = ecs_root() / "tmp" / name
    (tmp / "mpl").mkdir(parents=True, exist_ok=True)
    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TMPDIR": str(tmp),
           "MPLCONFIGDIR": str(tmp / "mpl"), "GITHUB_SHA": C.COMMIT, "GIT_CEILING_DIRECTORIES": str(ecs_root())}
    os.environ.update(env)
    tempfile.tempdir = None
    return env


def bootstrap_code() -> str:
    """The copy at sys.path[0]; refused unless `environments` imports from it."""
    code = os.path.realpath(C.CODE)
    if sys.path[:1] != [code]:
        sys.path.insert(0, code)
    import environments

    origin = os.path.realpath(environments.__file__)
    if not origin.startswith(code + os.sep):
        raise Refused(f"environments imported from {origin}, not the code copy {code}")
    return origin


def modules_outside_code() -> list[str]:
    code = os.path.realpath(C.CODE)
    bad = []
    for name, module in sorted(sys.modules.items()):
        if name == "environments" or name.startswith("environments."):
            path = getattr(module, "__file__", None)
            if path is not None and not os.path.realpath(path).startswith(code + os.sep):
                bad.append(f"{name}: {path}")
    return bad


def copy_inputs(start: str, stage_dir: Path) -> dict:
    """Byte copies of the start pair (periodic names) and the five parent records, each checked."""
    st = C.STARTS[start]
    S = st["S"]
    zip_name, pkl_name = C.periodic_names(S)
    models = stage_dir / "models"
    models.mkdir(parents=True)
    pairs = [(C.f2_file(start, st["zip_rel"]), models / zip_name, st["zip_sha256"]),
             (C.f2_file(start, st["pkl_rel"]), models / pkl_name, st["pkl_sha256"])]
    pairs += [(C.f2_record(start, n), stage_dir / n, st["records"][n]) for n in C.PARENT_RECORDS]
    out = {}
    for src, dst, want in pairs:
        shutil.copyfile(src, dst)
        got = sha256_file(dst)
        if got != want:
            raise RuntimeError(f"copy {dst}: sha256 {got} != {want}")
        out[dst.name] = {"from": src, "path": str(dst), "sha256": got}
    return out


def _write_meta(path: Path, meta: dict, mode: str) -> None:
    """mode "x": the first write (refused if the file exists); "w": an atomic rewrite (a reader, or a container
    restart, sees the old record or the new one, never a part)."""
    if mode == "x":
        with open(path, "x") as fh:
            json.dump(norm(meta), fh, indent=1)
            fh.write("\n")
        return
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(norm(meta), fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def run(ns: argparse.Namespace) -> int:
    """After preflight passed.  The folder is made first; from then on a failure is a crash (exit 1) unless it is a
    check before learn() (Refused, exit 2)."""
    t0 = time.monotonic()
    mode = mode_of(ns)
    start, arm, seed = ns.start, ns.arm, int(ns.seed)
    S, E = C.window(start)
    target = target_of(ns)
    name = C.run_name(start, arm, seed)
    root = run_root(ns)
    run_dir = root / name
    stage_dir = run_dir / C.STAGE_DIR
    label = f"ECS {start} {arm} {seed}"
    env = setup_environment(name)
    root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()  # refused if it exists (preflight checked it too)
    meta_path = run_dir / C.RUN_META
    crit = ecs_root() / "criteria.txt"
    meta: dict[str, Any] = {
        "schema": C.RUN_SCHEMA, "version": VERSION, "name": name, "mode": mode, "status": "starting", "exit": None,
        "utc_start": utc_now(), "argv": sys.argv, "start": start, "arm": arm, "seed": seed, "S": S, "E": E,
        "target": target, "T": None, "stop_reason": None, "run_id_parent": C.STARTS[start]["run_id"],
        "code": {"dir": C.CODE, "commit": C.COMMIT, "tree_digest": C.TREE_DIGEST, "files": C.TREE_FILES},
        "stack": C.stack_versions(), "environment": env,
        "criteria_sha256": sha256_file(crit) if crit.is_file() else None,
        "approved_sha256": sha256_file(ecs_root() / "APPROVED") if (ecs_root() / "APPROVED").is_file() else None,
        "tool_sha256": {n: sha256_file(TOOLS / n) for n in C.TOOL_FILES if (TOOLS / n).is_file()},
        "arm_ppo": dict(C.ARM_PPO_EDIT[arm]) or "unchanged", "ppo_applied": None,
    }
    _write_meta(meta_path, meta, "x")
    holder: dict[str, Any] = {}
    try:
        copies = copy_inputs(start, stage_dir)
        zip_copy = stage_dir / "models" / C.periodic_names(S)[0]
        pkl_copy = stage_dir / "models" / C.periodic_names(S)[1]
        meta.update({"status": "running", "copies": copies,
                     "start_pair": {"zip": str(zip_copy), "zip_sha256": copies[zip_copy.name]["sha256"],
                                    "pkl": str(pkl_copy), "pkl_sha256": copies[pkl_copy.name]["sha256"]},
                     "parent_records": {n: copies[n]["sha256"] for n in C.PARENT_RECORDS}})
        _write_meta(meta_path, meta, "w")
        meta["modules_root"] = bootstrap_code()
        import torch

        torch.set_num_threads(1)
        from environments.shared import train_base as TB
        from environments.shared.cli import _resolve_stage_ref
        from environments.shared.config import (
            hyperparameters_sha256,
            load_all_stages,
            read_stage_duration,
            record_stage_duration,
            refuse_occupied_stage_dir,
            save_stage_config,
        )
        from environments.shared.mirror_kl_ppo import MirrorKLPPO
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.policy_loading import _ensure_sb3
        from environments.shared.species_names import require_model_id
        from environments.shared.species_registry import get_species_config
        from environments.shared.stage_manifest import load_stage_manifest, stage_label
        from environments.shared.task_fingerprint import stage_task_fingerprint
        from stable_baselines3.common.callbacks import BaseCallback

        sb3 = _ensure_sb3()
        mid = require_model_id("tyrannosaurus_rex", "anatomical")
        configs = copy.deepcopy(load_all_stages(mid))
        stage = _resolve_stage_ref(C.STAGE_KEY, configs, mid)
        if stage is None:
            raise Refused("the code copy has no T. rex stance stage")
        configs[stage] = arm_config(configs[stage], arm)  # D1: in memory, before the load
        config = configs[stage]
        meta["ppo_applied"] = {k: config["ppo_kwargs"][k] for k in ("ent_coef", "ent_coef_end",
                                                                    "ent_coef_decay_timesteps")}
        cur = config.get("curriculum_kwargs", {})
        if (cur.get("eval_freq"), cur.get("checkpoint_freq")) != (C.EVAL_FREQ, C.SAVE_FREQ):
            raise Refused(f"stage cadence {cur.get('eval_freq')}/{cur.get('checkpoint_freq')} != "
                          f"{C.EVAL_FREQ}/{C.SAVE_FREQ}")
        species_cfg = get_species_config(mid)
        plant = current_plant_identity(mid)
        task_fp = stage_task_fingerprint(mid, stage, stage_config=config, plant_identity=plant)
        entry = load_stage_manifest(mid).resolve(stage)
        total = C.TOTAL_TIMESTEPS - S
        # train()'s steps before the body (TB l1790-l1855), on this run's stage folder.
        refuse_occupied_stage_dir(stage_dir, task_load_mode="resume_same_stage")
        prior_duration = read_stage_duration(stage_dir)
        stage_start = time.monotonic()
        save_stage_config(stage_dir, stage, config, "PPO", extra={"seed": seed, "n_envs": C.N_ENVS, "timesteps": total},
                          env_class=species_cfg.env_class, model_id=mid, plant_identity=plant,
                          task_fingerprint=task_fp, load_path=str(zip_copy), load_mode="resume_same_stage",
                          parent_run_id=None, label=label)
        # V5 before learn(): the written recipe against the parent's.
        with open(C.f2_record(start, "stage_config.json")) as fh:
            parent_sc = json.load(fh)
        with open(stage_dir / "stage_config.json") as fh:
            written_sc = json.load(fh)
        recomputed = {}
        for a in C.ARMS:
            hp = dict(parent_sc["hyperparameters"])
            hp.update(C.ARM_PPO_EDIT[a])
            recomputed[a] = hyperparameters_sha256({"ppo_kwargs": hp, "curriculum_kwargs": parent_sc["curriculum"]},
                                                   "ppo")
        v5 = recipe_problems(written_sc, parent_sc, arm=arm, S=S, seed=seed, label=label, load_path=str(zip_copy),
                             zip_sha256=C.STARTS[start]["zip_sha256"], hp_sha256_expected=recomputed[arm])
        if recomputed["A"] != C.HYPERPARAMETERS_SHA256_A:
            v5.append(f"recomputed arm A digest {recomputed['A']} != {C.HYPERPARAMETERS_SHA256_A}")
        meta["recipe_v5"] = {"problems": v5, "recomputed_hyperparameters_sha256": recomputed,
                             "stage_config_sha256": sha256_file(stage_dir / "stage_config.json")}
        _write_meta(meta_path, meta, "w")
        if v5:
            raise Refused("V5 recipe comparison: " + " | ".join(v5))

        original = TB._create_or_load_model

        def save_loaded(model: Any) -> list[str]:
            saver = sb3["CheckpointCallback"](save_freq=1, save_path=str(run_dir / "saved"),
                                              name_prefix=stage_label(stage), save_vecnormalize=True)
            saver.init_callback(model)
            saver.on_step()
            return sorted(str(p) for p in (run_dir / "saved").iterdir())

        checked_load = make_checked_load(original, mirror_cls=MirrorKLPPO, S=S, arm=arm,
                                         start_stats=pkl_stats(pkl_copy), mode=mode, holder=holder,
                                         save_loaded=save_loaded)
        stop_cls = make_stop_at_end_class(BaseCallback)

        def heartbeat(n: int) -> None:
            meta["heartbeat"] = {"rollout_end": int(n), "utc": utc_now()}
            _write_meta(meta_path, meta, "w")

        def stage_callback(*, eval_env: Any, eval_callback: Any) -> Any:
            # Built by the body after the load (TB l1395, l1529-l1533): the wrapper must have run (review I1).
            wp = wrapper_problems(holder, mode=mode, update_classes=(MirrorKLPPO, sb3["PPO"]))
            if wp:
                raise Refused(" | ".join(wp))
            holder["stop"] = stop_cls(target, heartbeat=heartbeat)
            return holder["stop"]

        TB._create_or_load_model = checked_load
        try:
            trained = TB._train_stage_body(
                sb3, species_cfg, configs, stage,
                log_path=stage_dir, model_dir=stage_dir / "models", total_timesteps=total, n_envs=C.N_ENVS,
                seed=seed, eval_freq=C.EVAL_FREQ, save_freq=C.SAVE_FREQ, use_subproc=False, verbose=0,
                algorithm=C.ALGORITHM, use_wandb=False, use_tensorboard=False, label=label, plant_identity=plant,
                task_fingerprint=task_fp, parent_id=entry.warm_start_from, load_path=str(zip_copy),
                task_load_mode="resume_same_stage", vecnorm_load_path=str(zip_copy), vecnorm_path=str(pkl_copy),
                allow_fresh_vecnorm=False, allow_legacy_plant=False, stage_callback=stage_callback,
                save_on_interrupt=False, announce=False, k3_panel=None)
        finally:
            TB._create_or_load_model = original
        record_stage_duration(stage_dir, (prior_duration or 0.0) + (time.monotonic() - stage_start))
        model, train_env = trained.model, trained.train_env
        stop = holder["stop"]
        digest_end = state_digest(model, train_env)
        coll = collapse_state(stop.siblings)
        T = int(model.num_timesteps)
        reason = stop_reason(flag=stop.flag, T=T, target=target, collapse_tripped=bool(coll["tripped"]))
        trained.train_env.close()
        trained.eval_env.close()
        meta.update({
            "T": T, "num_timesteps_end": T, "stop_reason": reason,
            "stop_at_end": {"target": target, "flag": stop.flag, "step": stop.step, "calls": stop.calls,
                            "last_step": stop.last_step},
            "collapse": coll, "ent_coef_after_load": holder.get("ent_coef_after_load"),
            "torch_threads": holder.get("torch_threads"), "torch_interop_threads": holder.get("torch_interop_threads"),
            "digest_start": holder.get("digest_start"), "digest_end": digest_end,
            "final_path": str(trained.final_path), "learn_seconds": round(trained.learn_seconds, 1),
            "checkpoints": sorted(p.name for p in (stage_dir / "models").iterdir()),
        })
        nu: list[str] = []
        if mode == "no_update":
            meta["train_calls"] = holder.get("train_calls")
            nu = no_update_problems(holder["digest_start"], digest_end, train_calls=holder.get("train_calls", -1),
                                    expected_calls=C.expected_update_rows(S, T))
            meta["no_update_check"] = {"problems": nu, "pass": not nu}
        outside = modules_outside_code()
        seal_after = seal_problems()
        meta.update({"modules_outside_code": outside, "code_seal_after": seal_after or "unchanged"})
        status, rc = post_run_status(nu, outside, seal_after)
        meta.update({"status": status, "exit": rc, "utc_end": utc_now(),
                     "wall_seconds": round(time.monotonic() - t0, 1)})
        _write_meta(meta_path, meta, "w")
        print(json.dumps({"done": status == "done", "name": name, "T": T, "stop_reason": reason, "exit": rc}),
              flush=True)
        return rc
    except _LoadedOnly:
        st = C.STARTS[start]
        saved_zip = run_dir / "saved" / C.periodic_names(S)[0]
        saved_pkl = run_dir / "saved" / C.periodic_names(S)[1]
        start_zip = zip_digests(stage_dir / "models" / C.periodic_names(S)[0])
        saved = zip_digests(saved_zip)
        vn_start = _hash_arrays(vecnorm_arrays(pkl_stats(stage_dir / "models" / C.periodic_names(S)[1])))
        vn_saved = _hash_arrays(vecnorm_arrays(pkl_stats(saved_pkl)))
        loaded = holder.get("digest_start") or {}
        v1 = {"start_zip": start_zip, "saved_zip": saved, "loaded": {k: loaded.get(k) for k in ("params", "adam")},
              "start_pkl_vecnormalize": vn_start, "saved_pkl_vecnormalize": vn_saved,
              "loaded_vecnormalize": loaded.get("vecnormalize"), "saved": holder.get("saved"),
              "saved_sha256": {p.name: sha256_file(p) for p in (saved_zip, saved_pkl)},
              "start_zip_sha256": st["zip_sha256"]}
        v1["params_equal"] = start_zip["params"] == saved["params"] == loaded.get("params")
        v1["vecnormalize_equal"] = vn_start == vn_saved == loaded.get("vecnormalize")
        ok = v1["params_equal"] and v1["vecnormalize_equal"]
        rc = C.EXIT_OK if ok else C.EXIT_POST_CHECK
        meta.update({"status": "done" if ok else "failed", "exit": rc, "T": S, "stop_reason": None,
                     "v1": v1, "ent_coef_after_load": holder.get("ent_coef_after_load"),
                     "torch_threads": holder.get("torch_threads"),
                     "torch_interop_threads": holder.get("torch_interop_threads"),
                     "digest_start": holder.get("digest_start"), "utc_end": utc_now(),
                     "wall_seconds": round(time.monotonic() - t0, 1),
                     "code_seal_after": seal_problems() or "unchanged"})
        _write_meta(meta_path, meta, "w")
        print(json.dumps({"saved_loaded_only": ok, "name": name}), flush=True)
        return rc
    except Refused as exc:
        meta.update({"status": "refused", "exit": 2, "error": str(exc), "utc_end": utc_now(),
                     "wall_seconds": round(time.monotonic() - t0, 1)})
        _write_meta(meta_path, meta, "w")
        print(json.dumps({"refused": str(exc), "name": name}), flush=True)
        return 2
    except BaseException as exc:  # a crash (an interrupt included, save_on_interrupt False): the folder stays
        stop = holder.get("stop")
        meta.update({"status": "crashed", "exit": 1, "error": repr(exc), "traceback": traceback.format_exc(),
                     "last_step": getattr(stop, "last_step", None), "utc_end": utc_now(),
                     "wall_seconds": round(time.monotonic() - t0, 1)})
        try:
            _write_meta(meta_path, meta, "w")
        except Exception:  # noqa: BLE001 - the exit code still says crash
            pass
        traceback.print_exc()
        return 1


def parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--save-loaded-only", dest="save_loaded_only", action="store_true")
    ap.add_argument("--no-update", dest="no_update", action="store_true")
    ap.add_argument("--until", type=int, default=None)
    ap.add_argument("--out", default=None, help="the no-optimizer-step modes only: a folder under <ECS>/validate/")
    ap.add_argument("--preflight-only", dest="preflight_only", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse(argv)
    print(VERSION, flush=True)
    try:
        problems = preflight(ns)
    except Exception as exc:  # noqa: BLE001 - an unexpected error in a check is a refusal (nothing made), not a crash
        problems = [f"preflight error: {exc!r}"]
    print(json.dumps({"preflight": "pass" if not problems else "refused", "mode": mode_of(ns), "problems": problems},
                     indent=1), flush=True)
    if problems:
        return 2
    if ns.preflight_only:
        return 0
    return run(ns)


if __name__ == "__main__":
    sys.exit(main())
