"""Stage 1 PPO training of R3 (the residual on R2) and R4 (policy only) (``STAGE1_PLAN.md`` section 2, section 3 step
7, section 6; decision 21).

Run from hardware/trex_form_study, in the PPO venv (SB3 2.9.0, gymnasium 1.3.0, torch CPU):

    python -m designlab.stage1 train --body 60-N-T-10 --seed 42 [--mode R3|R4] [--steps 5000000] [--resume]
                                     [--n-envs 4] [--vec dummy|subproc] [--out stage1/runs/2026-11-08]

* Settings: ``policy.DESIGN2`` (design 2's): the privileged critic (``policy.PrivilegedCriticPolicy``, [256, 256]),
  ``n_envs`` x 2,048 steps per rollout, batch 256, 5 epochs, gamma 0.99, lambda 0.95, clip 0.2, learning rate 1e-4 to
  3e-5 (linear), target_kl 0.03, VecNormalize (clip_obs 10, clip_reward 50, the stack's ``create_vec_env``); the
  envs are ``env.EnvConfig``'s defaults (all tasks, the section 2 randomisation, pushes), env ``i`` seeded
  ``1000 seed + i``.  One job runs its envs in one process (``--vec dummy``, the plan's per-job rate) unless asked.
* Records: ``<out>/<body>/ppo/<seed>/`` for R3 and ``<out>/<body>/policy_only/<seed>/`` for R4: ``run.json`` (the
  settings, library versions, the provenance of the code and the tracker's gains -- tune.json's selection or the
  tuning start, flagged --, every session, the checkpoints, the throughput and the G6 results), and
  ``checkpoints/model_<steps>.zip`` with ``vecnormalize_<steps>.pkl`` every ``CHECKPOINT_EVERY`` steps and at the end
  (git-ignored).
* ``--resume`` continues from the newest checkpoint: SB3's own continuation (the model with its optimiser, the
  normalisation statistics, the step counter, the learning-rate schedule's progress); the envs start new episodes, so
  a resumed run is not the uninterrupted one bit for bit.  It is refused if the settings differ.
* Final-checkpoint rule (``FINAL_RULE``): the policy judged is the checkpoint at the budget, with no best-model
  selection; if the stage's cap stops training first, every body is judged at the last checkpoint all of them reached
  (``final_checkpoint``).
* G6 (plan section 3 step 7, ``G6_RULE``): G6a before training -- a zero residual reproduces R2 exactly
  (``env.zero_residual_check``) and the job's env steps per second (the training loop's, measured over the first
  rollouts) at least ``G6A_ENV_STEPS_PER_S``; G6b at ``G6B_STEPS`` -- on the monitor episodes (``MONITOR``) R3
  survives at least as long as R2; G6c at ``G6C_STEPS`` -- R3 beats R2 on at least one monitor condition.  The results
  and the plan's consequence of a failure (cut the steps equally for every body; make one fix, then stop; skip seed 43
  and judge with R2) are written to run.json; acting on them is the operator's.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import env as env_mod
from . import plant

CHECKPOINT_EVERY = 250_000
BUDGET_STEPS = 5_000_000  # plan section 1/6 and decision 21 (if the faster tracker lands)
SEEDS = (42, 43)
G6A_ENV_STEPS_PER_S = 600
G6A_MEASURE_STEPS = 16_384  # the first rollouts of 4 x 2,048 steps
G6B_STEPS = 1_000_000
G6C_STEPS = 2_500_000
#: The monitor of G6b and G6c: these tasks, each on these episode seeds, on the nominal plant (no randomisation, no
#: push); the deterministic policy against the zero residual (R2) on the same episodes.
MONITOR = dict(tasks=("walk", "walking_turn_left", "walking_turn_right", "in_place_turn_left", "in_place_turn_right",
                      "brisk"), episode_seeds=(5001, 5002, 5003))
FINAL_RULE = ("the policy judged is the checkpoint at the budget (no best-model selection); if the stage cap stops "
              "training first, every body is judged at the last checkpoint all of them reached")
G6_RULE = (f"G6a: before training, a zero residual reproduces R2 exactly (env.zero_residual_check: every physics "
           f"step's qpos, qvel, ctrl and sensordata) and the job runs at least {G6A_ENV_STEPS_PER_S} env steps/s (the "
           f"training loop over its first {G6A_MEASURE_STEPS} steps, per wall second and per CPU second of the job's "
           f"process); otherwise the steps are cut equally for every body.  G6b at {G6B_STEPS} steps: on the monitor "
           f"(the deterministic policy and the zero residual on the same episodes: {', '.join(MONITOR['tasks'])}, "
           f"episode seeds {MONITOR['episode_seeds']}, nominal plant, no push) R3's mean survival time is at least "
           f"R2's; otherwise make one fix, then stop.  G6c at {G6C_STEPS} steps: R3 beats R2 on at least one monitor "
           "condition (a task: fewer falls, or as many and a higher mean reward per step); otherwise skip seed 43 and "
           "judge with R2")


def run_dir_of(out: Path, body: str, mode: str, seed: int) -> Path:
    return Path(out) / body / ("ppo" if mode == "R3" else "policy_only") / str(int(seed))


def _versions() -> dict:
    from importlib import metadata

    out = {}
    for pkg in ("stable-baselines3", "gymnasium", "torch", "numpy", "mujoco", "scipy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=float))
    os.replace(tmp, path)


def checkpoints(run_dir: Path) -> list[int]:
    """The step counts of a run's complete checkpoints (model and normalisation both saved)."""
    ck = Path(run_dir) / "checkpoints"
    steps = []
    for f in ck.glob("model_*.zip") if ck.exists() else ():
        s = f.stem.split("_", 1)[1]
        if s.isdigit() and (ck / f"vecnormalize_{s}.pkl").exists():
            steps.append(int(s))
    return sorted(steps)


def final_checkpoint(run_dirs, budget: int = BUDGET_STEPS) -> int | None:
    """``FINAL_RULE``: the budget's checkpoint if every run has it, else the last checkpoint all runs reached."""
    sets = [set(checkpoints(d)) for d in run_dirs]
    if not sets or not all(sets):
        return None
    common = set.intersection(*sets)
    if budget in common:
        return budget
    return max(common) if common else None


def monitor(model, vecnorm, body: str, mode: str, out: Path, config: dict | None = None) -> dict:
    """G6b/G6c's monitor (``MONITOR``): per task and episode seed, the deterministic policy and the zero residual."""
    cfg = dict(config or {}, randomize=False, push_prob=0.0, tasks=MONITOR["tasks"])
    if not cfg.get("run_dir"):
        cfg["run_dir"] = str(Path(out) / body)
    e = env_mod.make(mode, env_mod.EnvConfig(body=body, **cfg))
    rows = {}
    for controller in ("policy", "zero"):
        for task in MONITOR["tasks"]:
            for ep in MONITOR["episode_seeds"]:
                obs, _ = e.reset(seed=ep, options=dict(task=task, episode_seed=ep, push=None))
                total, steps, fell = 0.0, 0, False
                while True:
                    if controller == "policy":
                        a, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
                        a = a[0]
                    else:
                        a = np.zeros(env_mod.ACTION_DIM, np.float32)
                    obs, r, term, trunc, info = e.step(a)
                    total += r
                    steps += 1
                    if term or trunc:
                        fell = bool(term)
                        break
                rows.setdefault(controller, {}).setdefault(task, []).append(
                    dict(episode_seed=ep, fell=fell, survival_s=steps * e.dt_env, mean_reward=total / max(1, steps),
                         fall_cause=info.get("fall_cause")))
    summary = {}
    for controller, tasks in rows.items():
        summary[controller] = {t: dict(falls=sum(r["fell"] for r in v), survival_s=float(np.mean([r["survival_s"]
                                                                                                  for r in v])),
                                       mean_reward=float(np.mean([r["mean_reward"] for r in v]))) for t, v in tasks.items()}
    surv = {c: float(np.mean([s["survival_s"] for s in summary[c].values()])) for c in summary}
    beats = [t for t in MONITOR["tasks"] if (summary["policy"][t]["falls"] < summary["zero"][t]["falls"]) or (
        summary["policy"][t]["falls"] == summary["zero"][t]["falls"]
        and summary["policy"][t]["mean_reward"] > summary["zero"][t]["mean_reward"])]
    return dict(rows=rows, summary=summary, mean_survival_s=surv, policy_beats_zero_on=beats)


def train(body: str, seed: int, mode: str = "R3", steps: int = BUDGET_STEPS, out: Path | None = None,
          resume: bool = False, n_envs: int | None = None, vec: str = "dummy", checkpoint_every: int = CHECKPOINT_EVERY,
          g6: bool = True, config: dict | None = None, log_std_init: float = -1.0, ppo: dict | None = None,
          log=print) -> dict:
    """One PPO run (module docstring); returns its run.json.  ``config`` overrides ``env.EnvConfig`` fields and
    ``ppo`` design 2's PPO arguments (for smoke checks; both are recorded in the settings)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    from . import tune
    from .policy import DESIGN2, model_kwargs

    out = Path(out) if out is not None else tune.RUN
    n_envs = int(n_envs or DESIGN2["n_envs"])
    rd = run_dir_of(out, body, mode, seed)
    ck = rd / "checkpoints"
    path = rd / "run.json"
    settings = dict(body=body, seed=int(seed), mode=mode, n_envs=n_envs, design2=DESIGN2, log_std_init=log_std_init,
                    ppo_overrides=dict(ppo or {}), env=dict(env_mod.EnvConfig(body=body, **(config or {})).describe()),
                    checkpoint_every=checkpoint_every)
    doc = json.loads(path.read_text()) if path.exists() else None
    if doc and not resume:
        raise SystemExit(f"{path} exists: pass --resume to continue it (or remove it to start over)")
    if resume:
        if not doc:
            raise SystemExit(f"{path}: nothing to resume")
        if doc["settings"] != json.loads(json.dumps(settings, default=float)):
            raise SystemExit(f"{path}: its settings differ from this run's; refusing to resume")
    torch.set_num_threads(1)
    cfg = dict(config or {})
    cfg.setdefault("run_dir", str(Path(out) / body))
    cfg.setdefault("gains_seed", int(seed))

    def factory(i: int):
        def _make():
            return env_mod.make(mode, env_mod.EnvConfig(body=body, **cfg), seed=1000 * int(seed) + i)
        return _make

    fns = [factory(i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if vec == "subproc" else DummyVecEnv(fns)
    first_env = venv.envs[0] if hasattr(venv, "envs") else None
    describe = first_env.describe() if first_env is not None else venv.env_method("describe", indices=[0])[0]
    session = dict(started_at=_now(), provenance=plant.provenance(), versions=_versions(), vec=vec)
    if resume:
        done = checkpoints(rd)
        if not done:
            raise SystemExit(f"{ck}: no complete checkpoint to resume from")
        last = done[-1]
        vecnorm = VecNormalize.load(str(ck / f"vecnormalize_{last}.pkl"), venv)
        model = PPO.load(str(ck / f"model_{last}.zip"), env=vecnorm, device="cpu")
        session["resumed_from"] = last
        doc["sessions"].append(session)
    else:
        vecnorm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=DESIGN2["clip_obs"],
                               clip_reward=DESIGN2["clip_reward"], gamma=DESIGN2["gamma"])
        kwargs = model_kwargs(env_mod.ACTOR_DIM, log_std_init=log_std_init)
        kwargs.update(ppo or {})
        model = PPO(env=vecnorm, seed=int(seed), device="cpu", verbose=0, **kwargs)
        doc = dict(schema=1, body=body, seed=int(seed), mode=mode, settings=json.loads(json.dumps(settings,
                                                                                                default=float)),
                   env=describe, final_rule=FINAL_RULE, g6_rule=G6_RULE, sessions=[session], checkpoints=[],
                   throughput=[], g6={}, status="running", budget=int(steps))
        rd.mkdir(parents=True, exist_ok=True)
        if g6 and mode == "R3":
            log(f"{body} seed {seed}: G6a zero-residual check")
            doc["g6"]["G6a_zero_residual"] = env_mod.zero_residual_check(body, run_dir=cfg["run_dir"],
                                                                         gains_seed=int(seed))
        _write_json(path, doc)
    doc["budget"] = int(steps)

    class _Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0, self.c0, self.n0 = time.perf_counter(), time.process_time(), None
            self.next_ck = (model.num_timesteps // checkpoint_every + 1) * checkpoint_every
            self.measured = any(t.get("first_rollouts") for t in doc["throughput"])

        def _save(self, n: int) -> None:
            ck.mkdir(parents=True, exist_ok=True)
            model.save(str(ck / f"model_{n}.zip"))
            vecnorm.save(str(ck / f"vecnormalize_{n}.pkl"))
            doc["checkpoints"] = checkpoints(rd)
            doc["num_timesteps"] = int(n)
            _write_json(path, doc)

        def _on_step(self) -> bool:
            n = int(self.num_timesteps)
            if self.n0 is None:
                self.n0 = n - self.training_env.num_envs
            if not self.measured and n - self.n0 >= G6A_MEASURE_STEPS:
                wall, cpu = time.perf_counter() - self.t0, time.process_time() - self.c0
                rate = dict(first_rollouts=True, env_steps=n - self.n0, wall_s=wall, cpu_s=cpu,
                            env_steps_per_wall_s=(n - self.n0) / wall, env_steps_per_cpu_s=(n - self.n0) / cpu)
                doc["throughput"].append(rate)
                if g6:
                    doc["g6"]["G6a_rate"] = dict(rate, threshold=G6A_ENV_STEPS_PER_S,
                                                 pass_=bool(rate["env_steps_per_cpu_s"] >= G6A_ENV_STEPS_PER_S),
                                                 consequence_if_failed="cut the steps equally for every body")
                self.measured = True
                _write_json(path, doc)
            if n >= self.next_ck:
                self._save(n)
                self.next_ck += checkpoint_every
            for key, at in (("G6b", G6B_STEPS), ("G6c", G6C_STEPS)):
                if g6 and mode == "R3" and key not in doc["g6"] and n >= at:
                    self._save(n)
                    log(f"{body} seed {seed}: {key} monitor at {n} steps")
                    mon = monitor(model, vecnorm, body, mode, out, cfg)
                    if key == "G6b":
                        ok = mon["mean_survival_s"]["policy"] >= mon["mean_survival_s"]["zero"]
                        res = dict(at_steps=n, pass_=bool(ok), consequence_if_failed="make one fix, then stop")
                    else:
                        ok = bool(mon["policy_beats_zero_on"])
                        res = dict(at_steps=n, pass_=ok, consequence_if_failed="skip seed 43 and judge with R2")
                    doc["g6"][key] = dict(res, monitor=mon)
                    _write_json(path, doc)
            return True

    cb = _Callback()
    remaining = int(steps) - int(model.num_timesteps)
    t0, c0, n0 = time.perf_counter(), time.process_time(), int(model.num_timesteps)
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=False, progress_bar=False)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    n = int(model.num_timesteps)
    if n not in checkpoints(rd):
        cb._save(n)
    doc["throughput"].append(dict(session=len(doc["sessions"]), env_steps=n - n0, wall_s=wall, cpu_s=cpu,
                                  env_steps_per_wall_s=(n - n0) / max(wall, 1e-9),
                                  env_steps_per_cpu_s=(n - n0) / max(cpu, 1e-9)))
    doc["status"] = "done" if n >= int(steps) else "partial"
    doc["final_checkpoint"] = n if n >= int(steps) else None
    doc["sessions"][-1]["ended_at"] = _now()
    _write_json(path, doc)
    venv.close()
    return doc


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
