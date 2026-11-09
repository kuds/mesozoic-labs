"""Stage 1 PPO training of R3 (the residual on R2) and R4 (policy only) (``STAGE1_PLAN.md`` section 2, section 3 step
7, section 6; decision 21).

Run from hardware/trex_form_study, in the PPO venv (``PPO_VENV``: torch 2.9.0+cpu, Stable-Baselines3 2.9.0, gymnasium
1.3.0):

    python -m designlab.stage1 train --body 60-N-T-10 --seed 42 [--mode R3|R4] [--steps 5000000] [--resume]
                                     [--n-envs 4] [--vec dummy|subproc] [--allow-untuned] [--tune-json PATH]
                                     [--out stage1/runs/2026-11-08]
    python -m designlab.stage1 judge [--mode R3|R4] [--final] [--out stage1/runs/2026-11-08]
    python -m designlab.stage1 noise [--body 75-M-H-10] [--tune-json PATH] [--out stage1/runs/2026-11-08]

* Settings: ``policy.DESIGN2`` (design 2's): the privileged critic (``policy.PrivilegedCriticPolicy``, [256, 256]),
  ``n_envs`` x 2,048 steps per rollout, batch 256, 5 epochs, gamma 0.99, lambda 0.95, clip 0.2, target_kl 0.03,
  VecNormalize (clip_obs 10, clip_reward 50, the stack's ``create_vec_env``); the initial action noise
  ``policy.LOG_STD_INIT`` (``policy.LOG_STD_RULE``, measured by ``action_noise_survival``) and, for it, the learning
  rate 1e-5 to 3e-6 (linear; design 2's x 0.1, ``policy.LR_RULE``).  The budget: ``BUDGET_RULE`` (4M steps per run,
  decision 23) in whole rollouts.
* Envs: ``env.EnvConfig``'s defaults (every task of ``env.TASKS``, the carpet walk at mu 0.54 among them; section 2's
  randomisation, kp x 0.8-1.2 (``env.KP_RULE``); pushes), env i on ``env.FLOOR_SPREAD[i % 4]`` (``env.FLOOR_RULE``),
  seeded by ``SEED_RULE``.  A job runs its envs in its own process by default (``--vec dummy``: one env process per
  job, the plan's per-job rate); ``--vec subproc`` gives each env a process.
* R3's tracker gains (``env.GAINS_RULE``): tune.json's body-level G5 selection.  Training R3 is refused if tune.json is
  missing or no CMA-ES seed passes G5, unless ``--allow-untuned``.
* Records: ``<out>/<body>/ppo/<seed>/`` for R3 and ``<out>/<body>/policy_only/<seed>/`` for R4: ``run.json`` (the
  settings; every env's config as used; the tracker's gains with their selection; per session the provenance, the
  imported libraries' versions, every env's ``describe()``, the env seeds and the episodes each drew, the rate; the
  checkpoints; the G6 results; the judged checkpoint, ``JUDGED_RULE``), and ``checkpoints/model_<steps>.zip`` with
  ``vecnormalize_<steps>.pkl`` at rollout starts every ``CHECKPOINT_EVERY`` steps and at the end
  (``CHECKPOINT_RULE``; git-ignored).
* ``--resume`` continues from the newest checkpoint: SB3's own continuation (the model with its optimiser, the
  normalisation statistics, the step counter, the learning-rate schedule's progress) on fresh env streams (``SEED_RULE``:
  the session's own seeds), so a resumed run is not the uninterrupted one bit for bit.  It is refused if the settings,
  the env configs, the tracker's gains or their selection, the libraries or the Stage 1 sources differ from the run's
  (the sources only with ``--allow-source-change``, flagged in the session).
* Final-checkpoint rule (``FINAL_RULE``, ``JUDGED_RULE``): the policy judged is the checkpoint at the budget, with no
  best-model selection; if the stage's cap stops training first, every body is judged at the last checkpoint all of
  them reached (``final_checkpoint``, ``judged_checkpoint``; ``judge --final`` writes it into every run.json).
* G6 (plan section 3 step 7, ``G6_RULE``): G6a before training -- a zero residual reproduces R2 exactly
  (``env.zero_residual_check``) and the job's env steps per wall second (the training loop's, over its first rollouts,
  with the number of env processes) at least ``G6A_ENV_STEPS_PER_S``; G6b at ``G6B_STEPS`` -- on the monitor episodes
  (``MONITOR``, hard floor and the carpet condition) R3 survives at least as long as R2; G6c at ``G6C_STEPS`` -- R3
  beats R2 on at least one monitor condition by ``G6_RULE``'s margin.  The results and the plan's consequence of a failure (cut the steps equally for every body; make one fix,
  then stop; skip seed 43 and judge with R2) are written to run.json; acting on them is the operator's.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import BODIES, control, plant
from . import env as env_mod

CHECKPOINT_EVERY = 250_000
#: One PPO update per rollout of n_envs x n_steps env steps (design 2: 4 x 2,048).
ROLLOUT_STEPS = 4 * 2048


def whole_rollouts(steps: int, rollout: int = ROLLOUT_STEPS) -> int:
    """``steps`` rounded up to whole rollouts: where SB3's learn() stops (BUDGET_RULE)."""
    return -(-int(steps) // int(rollout)) * int(rollout)


#: Decision 23 (G6a's consequence: the steps cut equally for every body): 4M steps per run, R3 and R4, against the
#: plan's 5M (section 6, decision 21), in whole rollouts.
BUDGET_ASKED = 4_000_000
BUDGET_STEPS = whole_rollouts(BUDGET_ASKED)  # 4,005,888: 489 rollouts of 8,192
BUDGET_RULE = (f"every run of R3 and R4, every body, trains {BUDGET_ASKED:,} env steps rounded up to whole rollouts "
               f"({BUDGET_STEPS:,} with 4 envs x 2,048 steps), STAGE1_PLAN.md decision 23 (G6a's consequence: the plan's "
               "5M cut equally for every body to fit decision 21's cap on 4 cores)")
CHECKPOINT_RULE = ("a checkpoint is saved at the first rollout start at or past each multiple of the checkpoint "
                   "interval, after the previous update, so model_<n> is the policy trained on the run's first n steps "
                   "(every run with the same rollout size saves at the same n), and once more when learn() returns, "
                   "after the last update; the G6b and G6c monitors run on the policy at a rollout start too")
SEEDS = (42, 43)
R4_SEEDS = (42,)  # plan section 1: one policy-only seed per body
G6A_ENV_STEPS_PER_S = 600
G6A_MEASURE_STEPS = 16_384  # the first rollouts of 4 x 2,048 steps
G6A_CHECK = dict(task="walk", seconds=30.0, episode_seed=7)  # the zero-residual check: the whole walk task
G6B_STEPS = 1_000_000
G6C_STEPS = 2_500_000
#: G6c's margin on a condition with as many falls (G6_RULE): the mean per-step reward gain, paired by episode seed.
G6C_MIN_GAIN = 0.01
#: The PPO venv's library versions (decision 2's "validated versions where they install"; recorded per session).
PPO_VENV = {"torch": "2.9.0+cpu", "stable_baselines3": "2.9.0", "gymnasium": "1.3.0"}
#: The monitor of G6b and G6c: these conditions (task, floor), each on these episode seeds, on the nominal plant (no
#: randomisation, no push); the deterministic policy against the zero residual (R2) on the same episodes.  The carpet
#: walk on robot_floor_carpet and low pile is plan section 4's carpet condition, where R2 falls (decision 23).
MONITOR = dict(conditions=(("walk", "hard"), ("walking_turn_left", "hard"), ("walking_turn_right", "hard"),
                           ("in_place_turn_left", "hard"), ("in_place_turn_right", "hard"), ("brisk", "hard"),
                           ("carpet_walk", "robot_floor_carpet"), ("carpet_walk", "low")),
               episode_seeds=(5001, 5002, 5003))
FINAL_RULE = ("the policy judged is the checkpoint at the budget (no best-model selection); if the stage cap stops "
              "training first, every body is judged at the last checkpoint all of them reached")
JUDGED_RULE = (FINAL_RULE + "; the runs: every body x PPO seeds 42 and 43 for R3, every body x seed 42 for R4, each mode "
               "judged over its own runs; provisional (final false) until every run holds its budget's checkpoint or "
               "`judge --final` records that the stage's cap stopped training")
#: Env seeds (``env_seeds``): env i of session k of PPO seed s is reset with SEED_BASE s + 100 k + i.
SEED_BASE = 10_000
SEED_RULE = (f"right before learn() in every session (k = 0 for a new run, one more per resume), "
             f"model.set_random_seed({SEED_BASE:,} s + 100 k) for PPO seed s: torch, numpy, random and the action space "
             f"take that seed and the vec env's env i is reset with {SEED_BASE:,} s + 100 k + i (sessions below 100, at "
             "most 100 envs); SB3's own seeding (PPO's constructor and PPO.load both reseed the vec env with s + i) "
             "would give seeds 42 and 43 three of their four env streams in common and replay a run's streams on "
             "resume.  Each env then draws its episode seeds by env.EPISODE_SEED_RULE (1,000,000 x its reset seed + the "
             "episode's index), so no two runs, sessions or envs share an episode seed")
G6_RULE = (f"G6a: before training, a zero residual reproduces R2 exactly (env.zero_residual_check, "
           f"env.ZERO_RESIDUAL_RULE: every physics step's qpos, qvel, ctrl and sensordata to R2's end or fall, and the "
           f"same end) and the job runs at least {G6A_ENV_STEPS_PER_S} env steps per wall second (the training loop, "
           f"PPO updates included, over its first {G6A_MEASURE_STEPS} steps, with the number of env processes stated; "
           f"the CPU seconds of the job's processes, env worker processes included, beside it); otherwise the steps are "
           f"cut equally for every body (decision 23: G6a is judged per wall second of the job; BUDGET_RULE).  G6b at "
           f"the first rollout start at or past {G6B_STEPS} steps: on the monitor (the deterministic policy and the "
           f"zero residual on the same episodes: "
           f"{', '.join(f'{t} on {f}' for t, f in MONITOR['conditions'])}, episode seeds {MONITOR['episode_seeds']}, "
           f"nominal plant, no push) R3's mean survival time over every condition is at least R2's; otherwise make one "
           f"fix, then stop.  G6c at the first rollout start at or past {G6C_STEPS} steps: R3 beats R2 on at least one "
           f"monitor condition: fewer falls, or as many falls and, paired by episode seed, a higher mean reward per step "
           f"on every seed with a mean gain of at least {G6C_MIN_GAIN:g} per step (about 1 % of the reward's 0.9 "
           "ceiling) and at least the spread (largest minus smallest) of R2's per-seed mean rewards; otherwise skip "
           "seed 43 and judge with R2")


def run_dir_of(out: Path, body: str, mode: str, seed: int) -> Path:
    return Path(out) / body / ("ppo" if mode == "R3" else "policy_only") / str(int(seed))


def _versions() -> dict:
    """The versions of the libraries this process imported (their own ``__version__``)."""
    import gymnasium
    import mujoco
    import scipy
    import stable_baselines3
    import torch

    return dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                scipy=scipy.__version__, torch=torch.__version__, stable_baselines3=stable_baselines3.__version__,
                gymnasium=gymnasium.__version__)


def _jsonable(x):
    return json.loads(json.dumps(x, default=float))


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


def expected_runs(out: Path, mode: str) -> dict:
    """The runs a mode is judged over (``JUDGED_RULE``): "<body>/<seed>" -> run directory."""
    seeds = SEEDS if mode == "R3" else R4_SEEDS
    return {f"{b}/{s}": run_dir_of(out, b, mode, s) for b in BODIES for s in seeds}


def judged_checkpoint(out: Path, mode: str, budget: int = BUDGET_STEPS, final: bool = False) -> dict:
    """``JUDGED_RULE``: the checkpoint every run of ``mode`` is judged at, as of now (``final`` False: provisional
    unless every run holds its budget's checkpoint)."""
    runs = expected_runs(out, mode)
    have = {k: checkpoints(d) for k, d in runs.items()}
    return dict(rule=JUDGED_RULE, steps=final_checkpoint(list(runs.values()), budget), budget=int(budget),
                final=bool(final or all(budget in v for v in have.values())),
                runs={k: dict(last_checkpoint=v[-1] if v else None, has_budget=budget in v) for k, v in have.items()})


def judge(out: Path, mode: str = "R3", budget: int = BUDGET_STEPS, final: bool = False) -> dict:
    """``judged_checkpoint`` into the run.json of every run of ``mode`` that has one (``judge`` on the CLI)."""
    rec = judged_checkpoint(out, mode, budget, final)
    for d in expected_runs(out, mode).values():
        path = Path(d) / "run.json"
        if path.exists():
            doc = json.loads(path.read_text())
            doc["judged_checkpoint"] = rec
            _write_json(path, doc)
    return rec


def env_seeds(seed: int, session: int, n_envs: int) -> tuple[int, list[int]]:
    """``SEED_RULE``: the session's seed base and its envs' reset seeds."""
    if not 0 <= int(session) < 100 or not 0 < int(n_envs) <= 100:
        raise SystemExit(f"session {session} with {n_envs} envs: SEED_RULE gives disjoint seeds to at most 100 "
                         "sessions of at most 100 envs")
    base = SEED_BASE * int(seed) + 100 * int(session)
    return base, [base + i for i in range(int(n_envs))]


def env_configs(body: str, n_envs: int, config: dict | None = None, out: Path | None = None) -> list:
    """Each env's ``env.EnvConfig`` as a job uses it: ``config``'s fields, the references of ``<out>/<body>``, and env
    i on ``env.FLOOR_SPREAD[i % 4]`` unless ``config`` names a floor (``env.FLOOR_RULE``)."""
    from . import tune

    base = dict(config or {})
    base.setdefault("run_dir", str(Path(out if out is not None else tune.RUN) / body))
    out_cfgs = []
    for i in range(int(n_envs)):
        c = dict(base)
        if "floor" not in (config or {}):
            c["floor"] = env_mod.FLOOR_SPREAD[i % len(env_mod.FLOOR_SPREAD)]
        out_cfgs.append(env_mod.EnvConfig(body=body, **c))
    return out_cfgs


def _gains_of(cfg) -> tuple[np.ndarray, dict]:
    """The tracker gains an R3 env with ``cfg`` takes (``env.GAINS_RULE``), without building it."""
    if cfg.gains == "tuned":
        return env_mod.tuned_gains(Path(cfg.run_dir), cfg.body, cfg.tune_json)
    return (np.asarray(control.GAIN_SETS[cfg.gains], float).copy(),
            dict(source=control.GAINS_SOURCE[cfg.gains], status="asked", flag="untuned (asked for)",
                 candidate=cfg.gains, cma_seed=None, sha256=None))


def _diff(base: dict, d: dict) -> dict:
    """``d``'s entries that differ from ``base``'s (the other envs' ``describe()`` beside env 0's)."""
    return {k: v for k, v in d.items() if base.get(k) != v}


def condition_name(task: str, floor: str) -> str:
    return task if floor == "hard" else f"{task} on {floor}"


def beats_r2(policy_rows: list[dict], zero_rows: list[dict], min_gain: float = G6C_MIN_GAIN) -> dict:
    """G6c on one condition (``G6_RULE``): fewer falls, or as many and a paired per-seed reward gain on every seed of at
    least ``min_gain`` and R2's per-seed spread."""
    zr = {r["episode_seed"]: r for r in zero_rows}
    pairs = [(r["mean_reward"], zr[r["episode_seed"]]["mean_reward"]) for r in policy_rows]
    gains = [p - z for p, z in pairs]
    falls_p, falls_z = sum(r["fell"] for r in policy_rows), sum(r["fell"] for r in zero_rows)
    spread = float(max(z for _, z in pairs) - min(z for _, z in pairs)) if pairs else 0.0
    mean_gain = float(np.mean(gains)) if gains else 0.0
    need = max(float(min_gain), spread)
    by_reward = falls_p == falls_z and bool(gains) and min(gains) > 0.0 and mean_gain >= need
    return dict(beats=bool(falls_p < falls_z or by_reward), fewer_falls=bool(falls_p < falls_z), falls=falls_p,
                falls_r2=falls_z, gains=gains, mean_gain=mean_gain, r2_spread=spread, gain_needed=need)


def monitor(model, vecnorm, body: str, mode: str, config) -> dict:
    """G6b/G6c's monitor (``MONITOR``): per condition (task, floor) and episode seed, the deterministic policy and the
    zero residual, on env 0's config on that floor with no randomisation and no push."""
    rows = {}
    for task, floor in MONITOR["conditions"]:
        cfg = dataclasses.replace(config, randomize=False, push_prob=0.0, tasks=(task,), floor=floor)
        e = env_mod.make(mode, cfg)
        name = condition_name(task, floor)
        for controller in ("policy", "zero"):
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
                rows.setdefault(controller, {}).setdefault(name, []).append(
                    dict(episode_seed=ep, fell=fell, survival_s=steps * e.dt_env, mean_reward=total / max(1, steps),
                         fall_cause=info.get("fall_cause")))
        e.close()
    summary = {}
    for controller, conds in rows.items():
        summary[controller] = {c: dict(falls=sum(r["fell"] for r in v), survival_s=float(np.mean([r["survival_s"]
                                                                                                  for r in v])),
                                       mean_reward=float(np.mean([r["mean_reward"] for r in v]))) for c, v in conds.items()}
    surv = {c: float(np.mean([s["survival_s"] for s in summary[c].values()])) for c in summary}
    g6c = {c: beats_r2(rows["policy"][c], rows["zero"][c]) for c in rows["policy"]}
    beats = [c for c, v in g6c.items() if v["beats"]]
    return dict(rows=rows, summary=summary, mean_survival_s=surv, g6c_by_condition=g6c, policy_beats_zero_on=beats)


def train(body: str, seed: int, mode: str = "R3", steps: int = BUDGET_STEPS, out: Path | None = None,
          resume: bool = False, n_envs: int | None = None, vec: str = "dummy", checkpoint_every: int = CHECKPOINT_EVERY,
          g6: bool = True, config: dict | None = None, log_std_init: float | None = None, ppo: dict | None = None,
          allow_untuned: bool = False, allow_source_change: bool = False, log=print) -> dict:
    """One PPO run (module docstring); returns its run.json.  ``config`` overrides ``env.EnvConfig`` fields for every
    env and ``ppo`` design 2's PPO arguments (for smoke checks; both are recorded in the settings);
    ``allow_untuned`` trains R3 on gains without a G5 pass (flagged); ``allow_source_change`` resumes although the
    Stage 1 sources differ from the run's first session (flagged in the session)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    from . import tune
    from .policy import DESIGN2, LOG_STD_INIT, LR_END, LR_RULE, LR_START, model_kwargs

    if mode not in ("R3", "R4"):
        raise ValueError(f"mode {mode!r}: R3 or R4")
    if vec not in ("dummy", "subproc"):
        raise ValueError(f"vec {vec!r}: dummy or subproc")
    out = Path(out) if out is not None else tune.RUN
    n_envs = int(n_envs or DESIGN2["n_envs"])
    log_std_init = LOG_STD_INIT if log_std_init is None else float(log_std_init)
    steps_asked = int(steps)
    steps = whole_rollouts(steps_asked, n_envs * int((ppo or {}).get("n_steps", DESIGN2["n_steps"])))  # BUDGET_RULE
    rd = run_dir_of(out, body, mode, seed)
    ck = rd / "checkpoints"
    path = rd / "run.json"
    cfg_objs = env_configs(body, n_envs, config, out)
    cfgs = _jsonable([c.describe() for c in cfg_objs])
    settings = _jsonable(dict(body=body, seed=int(seed), mode=mode, n_envs=n_envs, design2=DESIGN2,
                              learning_rate=dict(start=LR_START, end=LR_END, schedule="linear", rule=LR_RULE),
                              log_std_init=log_std_init, ppo_overrides=dict(ppo or {}), config=dict(config or {}),
                              checkpoint_every=checkpoint_every, checkpoint_rule=CHECKPOINT_RULE,
                              floor_rule=env_mod.FLOOR_RULE, seed_rule=SEED_RULE))
    doc = json.loads(path.read_text()) if path.exists() else None
    if doc and not resume:
        raise SystemExit(f"{path} exists: pass --resume to continue it (or remove it to start over)")
    if resume:
        if not doc:
            raise SystemExit(f"{path}: nothing to resume")
        if doc["settings"] != settings:
            raise SystemExit(f"{path}: its settings differ from this run's; refusing to resume")
        if doc.get("env_configs") != cfgs:
            raise SystemExit(f"{path}: its env configs differ from this run's; refusing to resume")
    # R3's tracker gains (env.GAINS_RULE), before any env is built
    gains_rec = None
    if mode == "R3":
        gains, info = _gains_of(cfg_objs[0])
        if info["status"] != "passed" and not allow_untuned:
            raise SystemExit(f"{body}: R3's tracker gains have no G5 pass ({info['status']}: "
                             f"{info.get('file') or 'no tune.json'}); run G5 first, or pass --allow-untuned to train on "
                             f"{info['source']} (flagged {info['flag']!r})")
        gains_rec = _jsonable(dict(values=control.gains_dict(gains), candidate=info.get("candidate"),
                                   cma_seed=info.get("cma_seed"), status=info["status"], flag=info.get("flag"),
                                   source=info.get("source"), rule=env_mod.GAINS_RULE, tune_json=info.get("file"),
                                   tune_json_sha256=info.get("sha256"), allow_untuned=bool(allow_untuned)))
    versions = _versions()
    if resume:
        if gains_rec is not None:
            was = doc.get("gains") or {}
            changed = [k for k in ("values", "candidate", "cma_seed", "status") if was.get(k) != gains_rec[k]]
            if changed:
                raise SystemExit(f"{path}: the tracker's gains changed since the run started ({', '.join(changed)}: "
                                 f"tune.json now gives {gains_rec['source']}); refusing to resume")
        first = doc["sessions"][0].get("versions") or {}
        if first != versions:
            raise SystemExit(f"{path}: the libraries differ from the run's first session ({first} -> {versions}); "
                             "refusing to resume")
        was_src = ((doc["sessions"][0].get("provenance") or {}).get("stage1_sources") or {}).get("sha256")
        now_src = plant.provenance()["stage1_sources"]["sha256"]
        if was_src != now_src and not allow_source_change:
            raise SystemExit(f"{path}: the Stage 1 sources differ from the run's first session ({was_src} -> "
                             f"{now_src}); refusing to resume (pass --allow-source-change to resume on them, flagged)")
    session_k = len(doc["sessions"]) if resume else 0
    seed_base, seeds = env_seeds(seed, session_k, n_envs)
    torch.set_num_threads(1)

    def factory(i: int):
        def _make():
            return env_mod.make(mode, cfg_objs[i])
        return _make

    fns = [factory(i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if vec == "subproc" else DummyVecEnv(fns)
    describes = _jsonable(venv.env_method("describe"))
    if gains_rec is not None and any(d["gains"] != gains_rec["values"] for d in describes):
        venv.close()
        raise RuntimeError("an env's tracker gains differ from the run's")
    session = dict(index=session_k, started_at=_now(), provenance=plant.provenance(), versions=versions,
                   ppo_venv=PPO_VENV, matches_ppo_venv=all(versions[k] == v for k, v in PPO_VENV.items()), vec=vec,
                   env_processes=1 if vec == "dummy" else n_envs, n_envs=n_envs, torch_threads=torch.get_num_threads(),
                   solve2=control.SOLVE2_PATH, gains=gains_rec, env=describes[0],
                   other_envs=[_diff(describes[0], d) for d in describes[1:]], floors=[d["floor"] for d in describes])
    if resume:
        done = checkpoints(rd)
        if not done:
            venv.close()
            raise SystemExit(f"{ck}: no complete checkpoint to resume from")
        first_src = ((doc["sessions"][0].get("provenance") or {}).get("stage1_sources") or {}).get("sha256")
        if session["provenance"]["stage1_sources"]["sha256"] != first_src:
            session["source_change"] = dict(allowed=True, first_session=first_src,
                                            this_session=session["provenance"]["stage1_sources"]["sha256"])
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
        doc = dict(schema=3, body=body, seed=int(seed), mode=mode, settings=settings, env_configs=cfgs, gains=gains_rec,
                   final_rule=FINAL_RULE, judged_rule=JUDGED_RULE, g6_rule=G6_RULE, seed_rule=SEED_RULE,
                   budget_rule=BUDGET_RULE, sessions=[session], checkpoints=[], throughput=[], g6={}, status="running",
                   budget=int(steps), budget_asked=steps_asked)
        rd.mkdir(parents=True, exist_ok=True)
        if g6 and mode == "R3":
            log(f"{body} seed {seed}: G6a zero-residual check")
            c0 = cfg_objs[0]
            doc["g6"]["G6a_zero_residual"] = env_mod.zero_residual_check(
                body, run_dir=c0.run_dir, gains=c0.gains, tune_json=c0.tune_json, **G6A_CHECK)
        _write_json(path, doc)
    doc["budget"] = int(steps)
    doc["budget_asked"] = steps_asked
    # SEED_RULE: this session's own streams, after SB3's reseeding (the constructor's and PPO.load's)
    model.set_random_seed(seed_base)
    got = list(getattr(venv, "_seeds", seeds))
    if got != seeds:
        venv.close()
        raise RuntimeError(f"the vec env's seeds {got} are not SEED_RULE's {seeds}")
    session.update(seed_base=seed_base, env_seeds=seeds)
    me = os.getpid()

    def cpu_now() -> tuple[float, float]:
        """CPU seconds of this process and of the env worker processes (subproc; none with dummy)."""
        if vec == "dummy":
            return time.process_time(), 0.0
        rows = dict(venv.env_method("process_cpu"))
        return time.process_time(), float(sum(c for pid, c in rows.items() if pid != me))

    class _Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0, self.c0, self.n0 = time.perf_counter(), cpu_now(), None
            self.next_ck = (model.num_timesteps // checkpoint_every + 1) * checkpoint_every
            self.measured = any(t.get("first_rollouts") for t in doc["throughput"])

        def _save(self, n: int) -> None:
            ck.mkdir(parents=True, exist_ok=True)
            model.save(str(ck / f"model_{n}.zip"))
            vecnorm.save(str(ck / f"vecnormalize_{n}.pkl"))
            doc["checkpoints"] = checkpoints(rd)
            doc["num_timesteps"] = int(n)
            session["episode_streams"] = _jsonable(venv.env_method("episode_stream"))  # a killed session keeps them
            _write_json(path, doc)

        def _on_rollout_start(self) -> None:
            # CHECKPOINT_RULE: after the previous update, before this rollout's first step
            n = int(model.num_timesteps)
            if n >= self.next_ck:
                self._save(n)
                while self.next_ck <= n:
                    self.next_ck += checkpoint_every
            for key, at in (("G6b", G6B_STEPS), ("G6c", G6C_STEPS)):
                if g6 and mode == "R3" and key not in doc["g6"] and n >= at:
                    if n not in doc["checkpoints"]:
                        self._save(n)
                    log(f"{body} seed {seed}: {key} monitor at {n} steps")
                    mon = monitor(model, vecnorm, body, mode, cfg_objs[0])
                    if key == "G6b":
                        ok = mon["mean_survival_s"]["policy"] >= mon["mean_survival_s"]["zero"]
                        res = dict(at_steps=n, pass_=bool(ok), consequence_if_failed="make one fix, then stop")
                    else:
                        ok = bool(mon["policy_beats_zero_on"])
                        res = dict(at_steps=n, pass_=ok, consequence_if_failed="skip seed 43 and judge with R2")
                    doc["g6"][key] = dict(res, monitor=mon)
                    _write_json(path, doc)

        def _on_step(self) -> bool:
            n = int(self.num_timesteps)
            if self.n0 is None:
                self.n0 = n - self.training_env.num_envs
            if not self.measured and n - self.n0 >= G6A_MEASURE_STEPS:
                wall = time.perf_counter() - self.t0
                c1 = cpu_now()
                cpu_main, cpu_workers = c1[0] - self.c0[0], c1[1] - self.c0[1]
                k = n - self.n0
                rate = dict(first_rollouts=True, session=session_k, env_steps=k, n_envs=n_envs, vec=vec,
                            env_processes=session["env_processes"], wall_s=wall, cpu_s_main=cpu_main,
                            cpu_s_env_workers=cpu_workers, cpu_s=cpu_main + cpu_workers,
                            env_steps_per_wall_s=k / wall, env_steps_per_cpu_s=k / max(cpu_main + cpu_workers, 1e-9),
                            load_average=list(os.getloadavg()), cores=os.cpu_count())
                doc["throughput"].append(rate)
                if g6:
                    doc["g6"]["G6a_rate"] = dict(rate, threshold=G6A_ENV_STEPS_PER_S, judged_on="env steps per wall "
                                                 "second of the job", pass_=bool(rate["env_steps_per_wall_s"]
                                                                                 >= G6A_ENV_STEPS_PER_S),
                                                 consequence_if_failed="cut the steps equally for every body")
                self.measured = True
                _write_json(path, doc)
            return True

    cb = _Callback()
    remaining = int(steps) - int(model.num_timesteps)
    t0, c0, n0 = time.perf_counter(), cpu_now(), int(model.num_timesteps)
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=False, progress_bar=False)
    wall, c1 = time.perf_counter() - t0, cpu_now()
    n = int(model.num_timesteps)
    if n not in checkpoints(rd):  # CHECKPOINT_RULE: the last update's policy (no rollout start follows it)
        cb._save(n)
    cpu_main, cpu_workers = c1[0] - c0[0], c1[1] - c0[1]
    doc["throughput"].append(dict(session=session_k, env_steps=n - n0, n_envs=n_envs, vec=vec,
                                  env_processes=session["env_processes"], wall_s=wall, cpu_s_main=cpu_main,
                                  cpu_s_env_workers=cpu_workers, cpu_s=cpu_main + cpu_workers,
                                  env_steps_per_wall_s=(n - n0) / max(wall, 1e-9),
                                  env_steps_per_cpu_s=(n - n0) / max(cpu_main + cpu_workers, 1e-9),
                                  load_average=list(os.getloadavg())))
    session["episode_streams"] = _jsonable(venv.env_method("episode_stream"))
    doc["status"] = "done" if n >= int(steps) else "partial"
    doc["judged_checkpoint"] = judged_checkpoint(out, mode, int(steps))
    session["ended_at"] = _now()
    _write_json(path, doc)
    venv.close()
    return doc


# ------------------------------------------------------------------------------------------------
# The initial action noise (policy.LOG_STD_RULE)
# ------------------------------------------------------------------------------------------------
NOISE_BODY = "75-M-H-10"
NOISE_SIGMAS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04)
NOISE_EPISODES = tuple(range(7000, 7010))
NOISE_SURVIVE = 18  # of 20
NOISE_RULE = (
    "R3 on the nominal walk (env.TASKS['walk']: start, 7 strides, stop, at the commanded rate 1.05; hard floor, the "
    "nominal plant, no push) under the initial policy's action noise: every env step a = clip(sigma z, -1, 1), z a "
    "standard normal per action (the same z for every sigma, common random numbers, from default_rng(episode seed)), "
    "through the env's clip and 10 Hz low-pass; zero mean (SB3 initialises the action net's last layer at gain 0.01); "
    "10 episodes (seeds 7000-7009) at each of two gain sets on 75-M-H-10, the one body with a G5 record: the tuning "
    "start (decision 16) and tune.json's body-level G5 selection (env.GAINS_RULE), 20 episodes per sigma; an episode "
    "survives if it reaches the task's end")


def action_noise_survival(body: str = NOISE_BODY, sigmas=NOISE_SIGMAS, episodes=NOISE_EPISODES,
                          run_dir: str | None = None, tune_json: str | None = None, log=print) -> dict:
    """``NOISE_RULE``'s measurement and the ``policy.LOG_STD_RULE`` choice from it."""
    t_start = time.perf_counter()
    rows, gains = {}, {}
    for gname in ("start", "tuned"):
        cfg = env_mod.EnvConfig(body=body, tasks=("walk",), randomize=False, push_prob=0.0, run_dir=run_dir,
                                gains=gname, tune_json=tune_json)
        e = env_mod.ResidualEnv(cfg)
        if gname == "tuned" and e.gains_info["status"] != "passed":
            raise SystemExit(f"{body}: no G5-passed selection ({e.gains_info['status']}): the measurement needs one")
        gains[gname] = dict(values=control.gains_dict(e.gains), info=e.gains_info)
        rows[gname] = {}
        for sigma in sigmas:
            runs = []
            for ep in episodes:
                e.reset(seed=0, options=dict(task="walk", episode_seed=ep, push=None))
                rng = np.random.default_rng(ep)
                while True:
                    a = np.clip(sigma * rng.standard_normal(env_mod.ACTION_DIM), -1.0, 1.0).astype(np.float32)
                    _, _, term, trunc, info = e.step(a)
                    if term or trunc:
                        break
                runs.append(dict(episode_seed=ep, survived=not term, env_steps=e.k_env, of=e.n_steps,
                                 survival_s=e.k_env * e.dt_env, fall_cause=info.get("fall_cause")))
            n_ok = sum(r["survived"] for r in runs)
            rows[gname][repr(float(sigma))] = dict(sigma=float(sigma), survived=n_ok, episodes=len(runs), runs=runs)
            log(f"{body} {gname} sigma {sigma:g}: {n_ok}/{len(runs)} survive")
    total = {repr(float(s)): sum(rows[g][repr(float(s))]["survived"] for g in rows) for s in sigmas}
    n_eps = 2 * len(episodes)
    need = math.ceil(NOISE_SURVIVE * n_eps / 20)
    ok = [float(s) for s in sorted(sigmas) if total[repr(float(s))] >= need]
    # the largest sigma at and below which every measured sigma keeps the survival
    chosen = None
    for s in sorted(float(x) for x in sigmas):
        if total[repr(s)] < need:
            break
        chosen = s
    return dict(rule=NOISE_RULE, choice=f"the largest sigma at and below which every measured sigma keeps at least "
                                        f"{need}/{n_eps} episodes surviving; log_std_init = ln(sigma)",
                body=body, sigmas=[float(s) for s in sigmas], episodes=list(episodes), gains=gains, results=rows,
                survived=total, of=n_eps, passing=ok, sigma=chosen, log_std_init=None if chosen is None else
                math.log(chosen), filter_alpha=e.alpha, residual_leg_rad=env_mod.RESIDUAL_LEG_RAD,
                residual_rate=env_mod.RESIDUAL_RATE, seconds=time.perf_counter() - t_start,
                provenance=plant.provenance())


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
