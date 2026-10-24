"""TRACK TAIL (scratch): continue stance5_LTSB_s42 from its 4,194,304 checkpoint on CPU, with or without the mean bound.

    S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
    T=$S/foot/stance5/tailexp
    env ISO_WT=$S/fix/scratch_tailfix nice -n 19 python3 $S/foot/stance3/tools/iso.py $T/run_continuation.py \
        --arm control --steps 1048576
    env ISO_WT=$S/fix/scratch_tailfix nice -n 19 python3 $S/foot/stance3/tools/iso.py $T/run_continuation.py \
        --arm p1 --coef LAMBDA --steps 1048576

Arms (identical except the bound; maintainer decision 2026-10-22, a scratch experiment that changes no screen):
  control  the recipe as is.  The bound is configured at coef 0, so L_B = mean over the batch and the 15 actuators
           of relu(|mu| - 1)^2 is measured and logged (train/mean_bound_*) but never added; MirrorKLPPO then trains bit
           for bit as without it (unit test; --no-telemetry drops even the measurement, for the bitwise check here).
  p1       the same + coef * L_B in the loss (MeanBoundConfig(coef), branch scratch/tail-mean-bound of fix/scratch_tailfix).

The load is the repository's own same-stage resume (environments.shared.train_base, as train() -> _train_stage_body
does it): create_vec_env (n_envs 4, DummyVecEnv as the notebook trains, seed 42), the eval env built and seeded in the
same order, _load_vecnorm_into_envs (the run's own stage1_vecnormalize_4194304_steps.pkl, ret_rms carried, training
on: the statistics keep updating), _prepare_alg_kwargs (the LinearSchedule 3e-5 -> 1e-5), seed 42 (D-D11),
pop_symmetry_config, _create_or_load_model (load_sb3_model, plant and task checks, enable_mirror_kl: MirrorKLPPO, kappa
0.3, the run's class), EntCoefDecayCallback (absolute anchor 7,000,000) and SB3's CheckpointCallback (the stage1_*
naming, VecNormalize saved beside each).  Device cpu.  learn(reset_num_timesteps=False) keeps the step counter, and
the budget passed is the run's 12,000,000 minus the loaded steps, so SB3's _total_timesteps is 12,000,000 as the
checkpoint records and the learning rate and clip schedules sit where the run had them; a callback ends training after
exactly --steps (whole rollouts of 8,192), saving stage1_final.zip / stage1_final_vecnorm.pkl after the last update.
Left out (report only in the repository, inferred not to touch training): the eval panels, best-model trackers,
ledgers, policy probe, progress.json, collapse / hop watches (the hop watch only acts after 7M) and TensorBoard.
SB3 resume semantics kept: the first action uses the checkpoint's stored _last_obs (the envs themselves are freshly
reset by make_env, seeds 42..45).

Writes only into --out-dir (default tailexp/<arm>; refused if it exists): periodic stage1_<N>_steps.zip and
stage1_vecnormalize_<N>_steps.pkl every --save-every steps (cumulative N, as the run names them), stage1_final.*,
progress.csv (SB3's logger: every update's train/*, mean_bound_* included), telemetry.npz (per rollout, from the
rollout buffer before its update: per actuator the window of sampled actions at the clip, the clipped-command mean, the
policy mean on the visited states and its share beyond +-1; log_std; the previous update's train/* values) and
run_meta.json (provenance, the recipe checks, the final parameter / Adam / VecNormalize digests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
TAILEXP = SCRATCH + "/foot/stance5/tailexp"
CLONE = SCRATCH + "/fix/scratch_tailfix"
BRANCH = "scratch/tail-mean-bound"
PARENT_HEAD = "cf80ecb14d1630ce3b4dcdea28a05835d143470b"  # fix/scratch_copband, the run's git_commit
RUN_DIR = SCRATCH + "/foot/stance5/runs/stance5_LTSB_s42"
STAGE_DIR = RUN_DIR + "/01_stance"
START_STEPS = 4194304
START_ZIP = STAGE_DIR + f"/models/stage1_{START_STEPS}_steps.zip"
START_PKL = STAGE_DIR + f"/models/stage1_vecnormalize_{START_STEPS}_steps.pkl"
SEED, N_ENVS = 42, 4
ROLLOUT = 2048 * N_ENVS  # n_steps x n_envs
TAIL3 = 14  # tail_3_pitch_act


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def norm(x):
    """JSON round trip (tuples -> lists, numpy scalars -> numbers) for comparisons."""
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def git_state(clone: str) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", "--no-optional-locks", "-C", clone, *args], capture_output=True, text=True,
                              check=True).stdout.strip()

    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--porcelain"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "parent_is_ancestor": subprocess.run(
                ["git", "--no-optional-locks", "-C", clone, "merge-base", "--is-ancestor", PARENT_HEAD, "HEAD"]).returncode == 0}


def check_isolation(allow_dirty: bool) -> tuple[dict, str]:
    import environments

    origin = os.path.realpath(environments.__file__)
    if not origin.startswith(os.path.realpath(CLONE) + "/"):
        raise SystemExit(f"environments imported from {origin}, not the scratch clone {CLONE} (run through iso.py)")
    git = git_state(CLONE)
    if git["branch"] != BRANCH or not git["parent_is_ancestor"]:
        raise SystemExit(f"{CLONE} is on {git['branch']} (want {BRANCH}, a descendant of {PARENT_HEAD})")
    if git["status"] and not allow_dirty:
        raise SystemExit(f"{CLONE} has uncommitted changes:\n{git['status']}")
    return git, origin


def modules_under_clone() -> int:
    wt = os.path.realpath(CLONE)
    n = 0
    for name, m in sorted(sys.modules.items()):
        if name == "environments" or name.startswith("environments."):
            f = getattr(m, "__file__", None)
            if f is None:
                continue
            if not os.path.realpath(f).startswith(wt + "/"):
                raise SystemExit(f"{name} resolves to {os.path.realpath(f)}, outside {wt}")
            n += 1
    return n


def recipe_checks(configs: dict, stage, config: dict, task_fingerprint: dict) -> dict:
    """The clone's stance stage against the run's own records (refused on a mismatch)."""
    from environments.shared.config import hyperparameters_sha256

    with open(STAGE_DIR + "/stage_config.json") as fh:
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
    if (sc["run"]["seed"], sc["run"]["n_envs"]) != (SEED, N_ENVS):
        problems.append(f"the run's seed / n_envs {sc['run']['seed']} / {sc['run']['n_envs']} != {SEED} / {N_ENVS}")
    if task_fingerprint["task_sha256"] != sc["task_fingerprint"]["task_sha256"]:
        problems.append(f"task_sha256 {task_fingerprint['task_sha256']} != the run's {sc['task_fingerprint']['task_sha256']}")
    if sc["git_commit"] != PARENT_HEAD:
        problems.append(f"the run's git_commit {sc['git_commit']} != {PARENT_HEAD}")
    if norm(config.get("curriculum_kwargs", {})) != norm(sc["curriculum"]):
        problems.append("[curriculum] differs from the run's stage_config.json")
    if problems:
        raise SystemExit("recipe check refused: " + " | ".join(problems))
    return {"stage": stage, "hyperparameters_sha256": hp, "task_sha256": task_fingerprint["task_sha256"],
            "run_timesteps": int(sc["run"]["timesteps"]), "run_git_commit": sc["git_commit"],
            "env_keys_checked": len(env_now), "ppo_equal": True, "curriculum_equal": True}


def build(*, out_dir: str | None, verbose: int = 0):
    """The run's training setup at 4,194,304 through train_base's own resume path.  Returns a dict with the model
    (MirrorKLPPO), the training VecNormalize, the stage config and the checks."""
    import torch

    torch.set_num_threads(1)
    from environments.shared.cli import _resolve_stage_ref
    from environments.shared.config import load_all_stages
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

    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    configs = load_all_stages(mid)
    stage = _resolve_stage_ref("stance", configs, mid)
    config = configs[stage]
    species_cfg = get_species_config(mid)
    plant = current_plant_identity(mid)
    task_fp = stage_task_fingerprint(mid, stage, stage_config=config, plant_identity=plant)
    checks = recipe_checks(configs, stage, config, task_fp)
    sb3 = _ensure_sb3()
    gamma = config["ppo_kwargs"].get("gamma")
    # _train_stage_body's order: training env, evaluation env (seeded eval_env_seed), statistics, kwargs, model.
    train_env = create_vec_env(species_cfg, configs, stage, N_ENVS, SEED, False, algorithm="ppo", gamma=gamma,
                               plant_identity=plant)
    eval_env = create_vec_env(species_cfg, configs, stage, 1, eval_env_seed(SEED), use_subproc=False, algorithm="ppo",
                              gamma=gamma, plant_identity=plant)
    _load_vecnorm_into_envs(START_ZIP, train_env, eval_env, plant_identity=plant, task_load_mode="resume_same_stage",
                            allow_fresh_vecnorm=False, command_mode=str(config["env_kwargs"].get("command_mode", "none")),
                            vecnorm_path=START_PKL)
    alg_kwargs, _, _ = _prepare_alg_kwargs(config, "ppo", verbose, __import__("pathlib").Path(out_dir or TAILEXP),
                                           use_tensorboard=False)
    alg_kwargs.setdefault("seed", SEED)
    symmetry = pop_symmetry_config(alg_kwargs)
    alg_kwargs["device"] = "cpu"
    model = _create_or_load_model(sb3, "ppo", alg_kwargs, train_env, START_ZIP, plant_identity=plant,
                                  task_fingerprint=task_fp, task_load_mode="resume_same_stage", symmetry=symmetry)
    eval_env.close()  # built only for the order and the statistics load; it never runs
    from environments.shared.mirror_kl_ppo import MirrorKLPPO

    if type(model) is not MirrorKLPPO:
        raise SystemExit(f"the run trains MirrorKLPPO; the load gave {type(model).__name__}")
    if int(model.num_timesteps) != START_STEPS:
        raise SystemExit(f"the checkpoint's num_timesteps {model.num_timesteps} != {START_STEPS}")
    if int(model._total_timesteps) != checks["run_timesteps"]:
        raise SystemExit(f"the checkpoint's _total_timesteps {model._total_timesteps} != the run's {checks['run_timesteps']}")
    vecnorm = model.get_vec_normalize_env()
    assert vecnorm is train_env and vecnorm.training and vecnorm.norm_reward, (vecnorm.training, vecnorm.norm_reward)
    return {"model": model, "vecnorm": vecnorm, "config": config, "stage": stage, "checks": checks, "sb3": sb3,
            "plant": plant, "task_fingerprint": task_fp, "symmetry": symmetry}


def state_digest(model, vecnorm) -> dict:
    """sha256 of the policy parameters, the Adam state and the VecNormalize statistics (the determinism check)."""
    import numpy as np

    def h(arrays) -> str:
        d = hashlib.sha256()
        for name, a in arrays:
            a = np.ascontiguousarray(np.asarray(a))
            d.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        return "sha256:" + d.hexdigest()

    sd = model.policy.state_dict()
    params = [(k, sd[k].detach().cpu().numpy()) for k in sorted(sd)]
    opt = model.policy.optimizer.state_dict()
    adam = [(f"{i}.{n}", np.asarray(v[n].cpu() if hasattr(v[n], "cpu") else v[n]))
            for i, v in sorted(opt["state"].items()) for n in sorted(v)]
    vn = [("obs.mean", vecnorm.obs_rms.mean), ("obs.var", vecnorm.obs_rms.var), ("obs.count", np.float64(vecnorm.obs_rms.count)),
          ("ret.mean", vecnorm.ret_rms.mean), ("ret.var", vecnorm.ret_rms.var), ("ret.count", np.float64(vecnorm.ret_rms.count))]
    return {"params": h(params), "adam": h(adam), "vecnormalize": h(vn), "all": h(params + adam + vn),
            "n_updates": int(model._n_updates), "num_timesteps": int(model.num_timesteps)}


def write_state(model, vecnorm, path: str) -> None:
    import numpy as np

    sd = model.policy.state_dict()
    opt = model.policy.optimizer.state_dict()["state"]
    arrays = {f"param.{k}": sd[k].detach().cpu().numpy() for k in sd}
    arrays.update({f"adam.{i}.{n}": np.asarray(v[n]) for i, v in opt.items() for n in v})
    arrays.update({"obs_rms.mean": vecnorm.obs_rms.mean, "obs_rms.var": vecnorm.obs_rms.var,
                   "obs_rms.count": np.float64(vecnorm.obs_rms.count), "ret_rms.mean": vecnorm.ret_rms.mean,
                   "ret_rms.var": vecnorm.ret_rms.var, "ret_rms.count": np.float64(vecnorm.ret_rms.count)})
    np.savez_compressed(path, **arrays)


def make_callbacks(model, *, target: int, out_dir: str, save_every: int, sb3: dict, config: dict, vecnorm,
                   save_params: bool = False):
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from environments.shared.train_base import _maybe_ent_coef_decay_callback

    class Telemetry(BaseCallback):
        """Per rollout, from the buffer before its update; the previous update's train/* values."""

        KEYS = ("train/mean_bound_penalty", "train/mean_bound_loss", "train/mean_bound_coef",
                "train/mean_bound_beyond_fraction", "train/mean_bound_beyond_fraction_max", "train/approx_kl",
                "train/clip_fraction", "train/policy_gradient_loss", "train/value_loss", "train/entropy_loss",
                "train/std", "train/mirror_kl_loss", "train/mirror_asymmetry", "train/learning_rate", "train/n_updates",
                "train/explained_variance")

        def __init__(self):
            super().__init__(0)
            self.rows: dict[str, list] = {}
            self.t0 = time.monotonic()

        def _add(self, key, value):
            self.rows.setdefault(key, []).append(value)

        def _on_step(self) -> bool:
            return True

        def previous_update(self) -> dict:
            values = self.model.logger.name_to_value
            return {k: float(values[k]) if k in values else float("nan") for k in self.KEYS}

        def _on_rollout_end(self) -> None:
            buf = self.model.rollout_buffer
            obs = torch.as_tensor(buf.observations.reshape(-1, buf.observations.shape[-1]))
            act = buf.actions.reshape(-1, buf.actions.shape[-1]).astype(np.float64)
            with torch.no_grad():
                mu = self.model.policy.get_distribution(obs).mode().cpu().numpy().astype(np.float64)
            cmd = np.clip(act, -1.0, 1.0)
            self._add("num_timesteps", int(self.model.num_timesteps))
            self._add("seconds", time.monotonic() - self.t0)
            self._add("ent_coef", float(self.model.ent_coef))
            self._add("learning_rate_now", float(self.model.lr_schedule(self.model._current_progress_remaining)))
            self._add("progress_remaining", float(self.model._current_progress_remaining))
            self._add("total_timesteps", int(self.model._total_timesteps))
            self._add("sample_at_clip_hi", (act >= 1.0).mean(0))       # sampled a >= +1: env sees +1
            self._add("sample_at_clip_lo", (act <= -1.0).mean(0))
            self._add("cmd_ge_0p9", (np.abs(cmd) >= 0.9).mean(0))
            self._add("cmd_ge_0p99", (np.abs(cmd) >= 0.99).mean(0))
            self._add("cmd_mean", cmd.mean(0))
            self._add("mu_mean", mu.mean(0))
            self._add("mu_beyond", (np.abs(mu) > 1.0).mean(0))
            self._add("mu_tail3_q", np.quantile(mu[:, TAIL3], [0.05, 0.25, 0.5, 0.75, 0.95]))
            self._add("mean_bound_penalty_buffer", float(np.mean(np.maximum(np.abs(mu) - 1.0, 0.0) ** 2)))
            self._add("log_std", self.model.policy.log_std.detach().cpu().numpy().astype(np.float64).copy())
            self._add("ep_rew_mean", float(np.mean([e["r"] for e in self.model.ep_info_buffer])) if self.model.ep_info_buffer else float("nan"))
            self._add("ep_len_mean", float(np.mean([e["l"] for e in self.model.ep_info_buffer])) if self.model.ep_info_buffer else float("nan"))
            prev = self.previous_update()
            for k, v in prev.items():
                self._add("prev_" + k.replace("/", "_"), v)
            last = {k: (v[-1].tolist() if hasattr(v[-1], "tolist") else v[-1]) for k, v in self.rows.items()
                    if k in ("num_timesteps", "seconds", "ent_coef", "learning_rate_now")}
            last.update({"tail3_mu_mean": round(float(mu[:, TAIL3].mean()), 4),
                         "tail3_sample_at_clip_hi": round(float((act[:, TAIL3] >= 1.0).mean()), 4),
                         "tail3_cmd_ge_0p99": round(float((np.abs(cmd[:, TAIL3]) >= 0.99).mean()), 4),
                         "prev_mean_bound_loss": prev["train/mean_bound_loss"], "prev_mean_bound_penalty": prev["train/mean_bound_penalty"],
                         "prev_approx_kl": prev["train/approx_kl"]})
            print("[rollout] " + json.dumps(last), flush=True)

        def save(self, path: str) -> None:
            arrays = {k: np.asarray(v) for k, v in self.rows.items()}
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)

    class StopAndSave(BaseCallback):
        """After exactly `target` steps' updates: save stage1_final.* (state after the last update) and stop."""

        def __init__(self, telemetry: Telemetry):
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
                if save_params:
                    write_state(self.model, vecnorm, os.path.join(out_dir, "final_state.npz"))
                self.stop = True

        def _on_step(self) -> bool:
            if self.n_calls % (save_every // N_ENVS) == 0:
                self.telemetry.save(os.path.join(out_dir, "telemetry.npz"))
            return not self.stop

    telemetry = Telemetry()
    stopper = StopAndSave(telemetry)
    checkpoint = sb3["CheckpointCallback"](save_freq=save_every // N_ENVS, save_path=out_dir, name_prefix="stage1",
                                           save_vecnormalize=True)
    ent = _maybe_ent_coef_decay_callback(config, "ppo", int(config["curriculum_kwargs"]["timesteps"]))
    if ent is None:
        raise SystemExit("the stance [ppo] declares an entropy decay; none was built")
    callbacks = [checkpoint, ent, telemetry, stopper]
    return callbacks, telemetry, stopper, ent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=("control", "p1"))
    ap.add_argument("--coef", type=float, default=None, help="p1 only: lambda (see lambda.json); control is coef 0")
    ap.add_argument("--steps", type=int, required=True, help="steps to train past 4,194,304 (a multiple of 8,192)")
    ap.add_argument("--save-every", type=int, default=131072, help="periodic checkpoint cadence (default 131,072)")
    ap.add_argument("--out-dir", default=None, help="default tailexp/<arm>; refused if it exists")
    ap.add_argument("--no-telemetry", action="store_true", help="control only: do not configure the bound at all")
    ap.add_argument("--save-params", action="store_true", help="also write final_state.npz (parameters, Adam, stats)")
    ap.add_argument("--allow-dirty", action="store_true", help="development only: a dirty scratch clone")
    a = ap.parse_args()
    if a.steps <= 0 or a.steps % ROLLOUT:
        raise SystemExit(f"--steps must be a positive multiple of {ROLLOUT}")
    if a.save_every <= 0 or a.save_every % N_ENVS:
        raise SystemExit(f"--save-every must be a positive multiple of {N_ENVS}")
    if a.arm == "control":
        if a.coef not in (None, 0.0):
            raise SystemExit("the control arm is coef 0")
        coef = 0.0
    else:
        if a.coef is None or not (math.isfinite(a.coef) and a.coef > 0):
            raise SystemExit("--arm p1 needs --coef > 0 (lambda.json)")
        if a.no_telemetry:
            raise SystemExit("--no-telemetry is a control-arm option")
        coef = float(a.coef)
    out_dir = os.path.abspath(a.out_dir or os.path.join(TAILEXP, a.arm))
    if not out_dir.startswith(TAILEXP + "/"):
        raise SystemExit(f"outputs go under {TAILEXP}")
    if os.path.exists(out_dir):
        raise SystemExit(f"refusing to reuse {out_dir}")
    git, origin = check_isolation(a.allow_dirty)
    os.makedirs(out_dir)
    t_start = time.monotonic()
    import numpy as np
    import torch
    from stable_baselines3.common.logger import configure

    from environments.shared import mirror_kl_ppo as MK

    built = build(out_dir=out_dir)
    model, vecnorm, config = built["model"], built["vecnorm"], built["config"]
    bound = None if a.no_telemetry else MK.MeanBoundConfig(coef=coef)
    MK.enable_mean_bound(model, bound)
    model.set_logger(configure(out_dir, ["csv"]))
    target = START_STEPS + a.steps
    callbacks, telemetry, stopper, ent = make_callbacks(model, target=target, out_dir=out_dir, save_every=a.save_every,
                                                         sb3=built["sb3"], config=config, vecnorm=vecnorm,
                                                         save_params=a.save_params)
    start_digest = state_digest(model, vecnorm)
    meta = {
        "experiment": "TRACK TAIL (maintainer decision 2026-10-22): scratch, never pushed, changes nothing in any screen",
        "arm": a.arm, "coef": coef, "mean_bound": None if bound is None else bound.checkpoint_record(),
        "telemetry_only": a.arm == "control" and bound is not None, "steps": a.steps, "target_num_timesteps": target,
        "save_every": a.save_every, "out_dir": out_dir, "argv": sys.argv,
        "clone": CLONE, "git": git, "modules_environments": origin, "torch_threads": torch.get_num_threads(),
        "device": str(model.device), "versions": {"torch": torch.__version__, "numpy": np.__version__,
                                                  "stable_baselines3": __import__("stable_baselines3").__version__,
                                                  "mujoco": __import__("mujoco").__version__, "python": sys.version.split()[0]},
        "start": {"model": START_ZIP, "model_sha256": sha256_file(START_ZIP), "vecnorm": START_PKL,
                  "vecnorm_sha256": sha256_file(START_PKL), "num_timesteps": int(model.num_timesteps),
                  "total_timesteps_recorded": int(model._total_timesteps), "n_updates": int(model._n_updates),
                  "current_progress_remaining": float(model._current_progress_remaining),
                  "ent_coef_after_load": float(model.ent_coef), "digest": start_digest},
        "recipe_checks": built["checks"], "class": type(model).__name__,
        "symmetry": getattr(model, "mesozoic_symmetry", None),
        "ent_coef_decay": {"end": ent.end_value, "decay_timesteps": ent.decay_timesteps},
        "learn": {"total_timesteps_arg": built["checks"]["run_timesteps"] - START_STEPS, "reset_num_timesteps": False},
        "status": "running",
    }
    with open(os.path.join(out_dir, "run_meta.json"), "x") as fh:
        json.dump(norm(meta), fh, indent=1)
    print(json.dumps({"arm": a.arm, "coef": coef, "start": START_STEPS, "target": target, "out_dir": out_dir}), flush=True)
    model.learn(total_timesteps=built["checks"]["run_timesteps"] - START_STEPS,
                callback=built["sb3"]["CallbackList"](callbacks), reset_num_timesteps=False, progress_bar=False)
    if not stopper.stop:
        raise SystemExit(f"training ended at {model.num_timesteps} without reaching {target}")
    telemetry.save(os.path.join(out_dir, "telemetry.npz"))
    meta.update({"status": "done", "final_num_timesteps_env": int(model.num_timesteps),
                 "final_checkpoint_num_timesteps": target, "final_update_train": stopper.final_update,
                 "final_digest": stopper.final_digest,
                 "note_final": "stage1_final.* and final_digest are the state after the last update (exactly `target` "
                               "steps), taken at the start of the next rollout; the run then takes one env step (4 "
                               "transitions, saved nowhere) and stops",
                 "checkpoints": sorted(f for f in os.listdir(out_dir) if f.endswith((".zip", ".pkl"))),
                 "final_sha256": {f: sha256_file(os.path.join(out_dir, f)) for f in ("stage1_final.zip", "stage1_final_vecnorm.pkl")},
                 "modules_all_under_clone": modules_under_clone(), "wall_seconds": round(time.monotonic() - t_start, 1)})
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fh:
        json.dump(norm(meta), fh, indent=1)
    print(json.dumps({"done": True, "arm": a.arm, "num_timesteps": target, "digest": stopper.final_digest["all"],
                      "wall_seconds": meta["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
