"""Stage 1 policy rows R3 and R4 for the behaviour panels (``STAGE1_PLAN.md`` section 2's R3 and R4, section 3 step 8,
section 4; the ``panels.Row`` interface): the trained PPO policies, run as training ran them, on every panel condition.

Run from hardware/trex_form_study in the PPO venv (torch, Stable-Baselines3 and gymnasium, ``train.PPO_VENV``)::

    python -m designlab.stage1 eval --body 60-N-T-10 --rows R3/42,R3/43,R4/42 --row-module designlab.stage1.policy_rows
                                    [--attribute] [--workers 4] --out stage1/runs/2026-11-08
    python -m designlab.stage1.policy_rows verdicts --body 60-N-T-10 --out stage1/runs/2026-11-08
    python -m designlab.stage1.policy_rows check --body 75-M-H-10 --row R3/42 [--condition walk/nominal]
    python -m designlab.stage1.policy_rows costs --body 75-M-H-10 [--rows R3/42,R4/42] [--floors hard,low]

``--row-module`` imports this module, which registers the controllers R3 and R4 (``panels.register_rows``).  The
panels are unchanged, and so are the R0-R2 records' keys: this module is not among the panels' sources
(``panels.PANEL_SOURCES_RULE``) and enters only the policy rows' keys, through ``Row.sources`` and ``Row.describe``.
Importing it needs no torch; building a row needs the PPO venv.

* Rows (``POLICY_ROWS_RULE``): ``R3/<s>`` is a ``panels.TrackerRow`` in mode R2 with the body-level G5 gains and
  ``TrainedPolicy``, PPO seed s's residual policy; ``R4/<s>`` is ``PolicyOnlyRow``, which runs PPO seed s's
  policy-only policy on ``PolicyOnlyTracker`` (``R4_RULE``) through ``control.run``, so both have control.run's
  outcome (falls, onsets, candidates, stops, drive ratios), both take ``relax`` (decision 20's attribution, which
  the panels apply to R3, ``panels.ATTRIBUTED``) and both are judged as every row is.
* Checkpoint (``CHECKPOINT_SELECTION_RULE``): the mode's judged checkpoint by default (``train.judged_checkpoint``);
  ``STAGE1_POLICY_CHECKPOINT`` selects another for tests and interim runs; ``STAGE1_POLICY_RUNS`` names the
  training run directory (the checkpoints are git-ignored and live where training ran).
* Acting (``POLICY_RULE``, ``PERIOD_RULE``): env.py's own observation, action and physics-step functions, bound to a
  view of the panel episode (``ResidualView`` for R3; for R4 ``PolicyOnlyTracker`` is its own view, as
  ``env.PolicyOnlyEnv`` is), at env.py's physics steps, on the nominal model, latency 0, the deterministic action.
* Equivalence (``EQUIVALENCE_RULE``, ``equivalence_check``): a row's episode against training's env, every physics
  step's state hashed (``tests/test_stage1_policy_rows.py``).
* Verdicts (``R4_VERDICT_RULE``, ``policy_verdicts``): R3's are the panels' (R3/42 and R3/43, ``panels.VERDICT_RULE``);
  R4's, from its one seed, are at most provisional.
* Costs (``POLICY_COST_RULE``, ``measure_costs``, ``project``): CPU and wall seconds per simulated second, and the
  panels' core-hours at those costs (``panels.dry_run``).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import types
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import numpy as np

from . import STUDY, control, env, panels, plant, train, tune

#: The environment variables of ``CHECKPOINT_SELECTION_RULE``.
CHECKPOINT_ENV = "STAGE1_POLICY_CHECKPOINT"
RUNS_ENV = "STAGE1_POLICY_RUNS"
MODES = ("R3", "R4")
#: The module that defines the checkpoints' policy class (``policy.PrivilegedCriticPolicy``), hashed without importing
#: torch.
POLICY_MODULE = Path(env.__file__).resolve().with_name("policy.py")
LIBRARIES = ("torch", "stable_baselines3", "gymnasium")
#: The versions a policy row must share with its run (``CHECKPOINT_SELECTION_RULE``; train._versions' names).
RUN_VERSIONS = ("python", "numpy", "mujoco", "scipy") + LIBRARIES
#: The panels' cost condition's start for the steady part (``POLICY_COST_RULE``).
COST_SKIP_S = 2.0
COST_SAMPLE_S = 4.0

# ------------------------------------------------------------------------------------------------
# Rules (each copied into every record a policy row writes, through its description)
# ------------------------------------------------------------------------------------------------
POLICY_ROWS_RULE = (
    "R3/<s>: panels.TrackerRow in mode R2 (control.TERMS, the decided ankle cap and inward bound, clock rate "
    f"{panels.RATE:g}, the plant's own model for the tracker and its estimator, as R2's rows) with the body-level G5 "
    "selection (env.tuned_gains, env.GAINS_RULE: the gains both PPO seeds trained on; refused unless they equal the "
    "run's run.json gains value for value, with the same candidate, CMA-ES seed and status) and TrainedPolicy, PPO "
    "seed s's residual policy (<runs>/<body>/ppo/<s>), acting as env.ResidualEnv acts (POLICY_RULE).  R4/<s>: "
    "PolicyOnlyRow, no tracker: PPO seed s's policy-only policy (<runs>/<body>/policy_only/<s>) on PolicyOnlyTracker "
    "through control.run, as env.PolicyOnlyEnv runs it (R4_RULE).  Both run every panel condition "
    "(panels.CONDITIONS_RULE; the carpet walk is gated for them, panels.GATING_RULE, decision 23) from the panels' "
    "episode seeds and start noise (panels.EPISODE_RULE), with any relaxation (control.relaxed; the panels attribute "
    "R3's falls by decision 20, not R4's: panels.ATTRIBUTED), and are judged by the panels' judge unchanged; each "
    "names its checkpoint, run settings and sources (env.py, policy_rows.py, policy.py) in its description and key.  "
    "The panels' recorder reads each feedback update's reference time as the clock less one physics step at the "
    "commanded rate and the phase-rate term (metrics.Recorder.update), not counting a policy's rate factor: for these "
    "rows that reading is off by at most 2 ms x 1.05 x 1.05 x 0.25 = 0.55 ms of reference time (a quarter of a "
    "physics step; R4 at the track's end, where its clock stops, by one step), which the judgement takes as it is")
CHECKPOINT_SELECTION_RULE = (
    "a policy row runs, by default, its mode's judged checkpoint: train.judged_checkpoint over every run of the mode "
    "(train.JUDGED_RULE: the budget's checkpoint, or the last checkpoint every run reached), final if every run holds "
    "its budget's checkpoint or the run's run.json holds a 'judge --final' record at the same step count, else "
    f"provisional.  The environment variable {CHECKPOINT_ENV} selects another for tests and interim runs: a step "
    "count N (model_N, every policy row), 'newest' (each run's newest complete checkpoint) or per mode, e.g. "
    "'R3=newest,R4=1007616' (a mode it does not name keeps the judged one); such a checkpoint is never final.  The "
    f"runs are read from {RUNS_ENV} (the training run directory holding <body>/ppo/<s> and <body>/policy_only/<s>; "
    "default: the panels' run directory, eval's --inputs or --out).  A row names its model zip and VecNormalize pkl "
    "(paths relative to that directory) with their sha256, the step count, the selection and judgement, and the "
    "run's settings (log_std_init, the learning-rate rule, design 2's settings, the env config, R3's gains); a worker "
    "refuses a file whose sha256 differs from the row's.  A row is refused unless this process's libraries are the "
    "run's (its run.json's first session: python, numpy, mujoco, scipy, torch, stable_baselines3, gymnasium; "
    "train.PPO_VENV), so the policy computes as it did in training (the fit's venv has other torch, SB3 and gymnasium "
    "versions)")
POLICY_RULE = (
    "a policy row acts as training's env acts (env.Stage1Env.step) on the nominal model (no randomisation: the "
    "condition's friction, trunk mass and servo stiffness x 1, sensor-noise scale 1, latency 0): before the first "
    f"physics step and after every {env.ENV_SUBSTEPS}th (50 Hz), the observation is env.Stage1Env._observation over "
    "the panel episode (the estimator's encoder angles and rates, IMU gravity and gyro and foot loads over the nominal "
    "weight; the reference clock's phase over PERIOD_RULE's period, the reference's support flags and its command at "
    "the commanded rate; the previous action; then the privileged inputs, filled where the panel episode has them, "
    "which the actor never reads, policy.PrivilegedMlpExtractor), normalised by the checkpoint's VecNormalize "
    "(normalize_obs: its saved statistics, unchanged; clip 10); the action is the checkpoint's "
    "PPO.predict(normalized, deterministic=True) (the Gaussian's mean, clipped to the action space as SB3 clips), as "
    "train.monitor asks it, on one torch thread; then env.Stage1Env.step's own processing: clipped to [-1, 1], "
    f"low-passed at env.FILTER_HZ = {env.FILTER_HZ:g} Hz (a_f += alpha (a - a_f), alpha = 1 - exp(-2 pi x "
    f"{env.FILTER_HZ:g} Hz x {env.ENV_SUBSTEPS} physics steps)), and the mode's own action function on a_f (R3 "
    f"env.ResidualEnv._apply_action: joint offsets a_f x {env.RESIDUAL_LEG_RAD:g} rad on the legs and "
    f"{env.RESIDUAL_NECK_TAIL_RAD:g} rad on neck and tail added to every servo target, none when all are zero, and "
    f"the reference clock's rate x (1 + {env.RESIDUAL_RATE:g} a_f[14]); R4 env.PolicyOnlyEnv._apply_action, R4_RULE), "
    "held for the next physics steps; the previous action is the clipped a")
PERIOD_RULE = (
    "the phase input runs over the period of the stride reference env.TASKS names for the env task whose track plan "
    "the condition's plan is (the nominal walk, the 0.35 rad/s walking turns and the 0.5 rad/s in-place turns: the "
    "nominal walk's; the carpet walks on every floor: the carpet reference's), else over the nominal walk's (the "
    "stands, the brisk and Fr 0.05 walks, the 0.5 rad/s walking turns, the start and stop, the pushes), the reference "
    "env.TASKS names for every task but the carpet walk, the stand, brisk and Fr 0.05 tasks included")
R4_RULE = (
    "R4's controller (PolicyOnlyTracker) is a control.Tracker that control.run runs as R2 with no feedback (no terms, "
    "no ankle cap; its estimator, sensor noise and rest speed R2's) and env.PolicyOnlyEnv's own functions: an episode "
    "starts as env.PolicyOnlyEnv._begin starts it (the settled stance with the panels' start noise and start seed, "
    "the sensors' noise from the noise seed, the estimator at rest, the clock at 0, the targets at the settled "
    "stance), every physics step is env.PolicyOnlyEnv._physics_step (the joint targets set by Plant.command alone and "
    "held by the servos' PD, one physics step, the clock advanced by dt x rate x the rate factor and capped at the "
    "track's end, the estimator stepped on the sensors and updated every 5 steps), and its mapping is "
    "env.PolicyOnlyEnv._setup's on the condition's plant: home = that plant's settled stance, span = "
    f"{env.POLICY_ONLY_SPAN_FACTOR:g} x each joint's largest excursion from home over the run's env tasks' tracks, "
    f"built on that plant as the env builds them, at least {env.POLICY_ONLY_MIN_SPAN_RAD:g} rad "
    "(env.POLICY_ONLY_SPAN_RULE; the evaluation piles' settled stances, not the 8 mm training pile's); the targets are "
    "env.PolicyOnlyEnv._apply_action's (home + a_f x span, clipped to the usable range).  control.run reads its falls, "
    "onsets, candidates (the CoP-at-pad-edge and leg-to-leg candidates are the tracker's and never active), stops and "
    "drive statistics as for any row; the estimator's model is the plant's own, as the panels' R2 and R3 rows have it "
    "(env.py's copy of it holds the same values on the nominal plant)")
EQUIVALENCE_RULE = (
    "a policy row's episode of a panel condition against training's env (env.ResidualEnv for R3, env.PolicyOnlyEnv "
    "for R4) on the env task whose plan the condition's plan is, on the condition's floor (a pile: the panels' 4 mm "
    "evaluation pile, EnvConfig pile_grid None) and friction, randomize False, push None, latency 0, the same "
    "checkpoint asked the same way (train.monitor's call); the env's episode starts from the panel episode's start "
    "and sensor-noise seeds (its _begin given them: the harness's alignment) with the same start noise and, for R4, "
    "the run's env tasks (the span's tracks); sha256 of qpos, qvel, ctrl and sensordata after every physics step "
    "(env.ZERO_RESIDUAL_RULE's digest): equal at every physics step both ran, over the shorter horizon (control.run "
    "runs round(seconds / 2 ms) physics steps, the env round(seconds / 20 ms) env steps of 10, so the two may differ "
    "by up to 5 physics steps) or to the panel's fall, and the same end: a panel fall at physics step k within the "
    "shorter horizon is an env termination in the env step that holds k (the env reads falls at its steps' ends); "
    "without one, both run through the shorter horizon and the env does not terminate within it")
R4_VERDICT_RULE = (
    "R4 trains one PPO seed per body (train.R4_SEEDS, decision 28), so its verdict on a condition is panels.verdict on "
    "R4/42's record alone with 'shown' (both seeds pass, panels.VERDICT_RULE) read as 'provisional' (one seed passes): "
    "at most provisional; the other verdicts as panels.verdict gives them (R4's falls are not attributed, "
    "panels.ATTRIBUTED, so a condition it fails by falls reads 'not attributed').  R3's verdicts are the panels' own "
    "(panels.VERDICT_ROWS: R3/42 and R3/43)")
POLICY_COST_RULE = (
    "per row and floor: the nominal walk on that floor (the panels' cost condition, panels.COST_RULE) from the "
    f"settled stance on seed {panels.SEEDS[0]}, in this process, at the time of measuring (the machine shared with "
    "the training jobs and the R0-R2 panels: the load average is recorded beside), timed from the first action at or "
    "after skip_s simulated (the plant build, R4's span tracks and the model's load come before it) to the episode's "
    "end or fall: CPU seconds (process time) and wall seconds per simulated second; R2/42 (a zero residual on R2, "
    "panels.ZeroResidual, bit for bit R2) beside for comparison.  Core-hours: panels.dry_run's projection (every "
    "episode's planned seconds x its floor's CPU cost; the attribution's bound lifts every relaxable limit on every "
    "condition of an attributed row, R3) with a row's costs per floor, a floor not measured at the cost of the nearest "
    "measured one (medium and high pile: low pile's; robot_floor_carpet: the mean of hard floor's and low pile's)")


class PolicyRowError(ValueError):
    """A policy row cannot be built or run as asked."""


# ------------------------------------------------------------------------------------------------
# Checkpoints (CHECKPOINT_SELECTION_RULE)
# ------------------------------------------------------------------------------------------------
def _parse_choice(value: str):
    value = value.strip()
    if value in ("judged", "newest"):
        return value
    if value.isdigit():
        return int(value)
    raise PolicyRowError(f"{CHECKPOINT_ENV}: {value!r} is not 'judged', 'newest' or a step count")


def checkpoint_choice(mode: str, value: str | None = None):
    """``CHECKPOINT_SELECTION_RULE``'s selection for ``mode`` from ``value`` (default: ``STAGE1_POLICY_CHECKPOINT``):
    "judged", "newest" or a step count.  Items are separated by commas; "MODE=VALUE" applies to that mode and wins
    over a bare value, which applies to every mode."""
    raw = os.environ.get(CHECKPOINT_ENV, "") if value is None else str(value)
    bare = per_mode = None
    for item in (x.strip() for x in raw.split(",")):
        if not item:
            continue
        if "=" in item:
            k, _, v = item.partition("=")
            if k.strip() not in MODES:
                raise PolicyRowError(f"{CHECKPOINT_ENV}: {item!r} names no mode of {', '.join(MODES)}")
            if k.strip() == mode:
                per_mode = _parse_choice(v)
        else:
            bare = _parse_choice(item)
    if per_mode is not None:
        return per_mode
    return "judged" if bare is None else bare


def runs_dir(run_dir: Path) -> Path:
    """The training run directory a policy row reads its run from (``STAGE1_POLICY_RUNS``, default the parent of the
    panels' ``<run>/<body>``)."""
    v = os.environ.get(RUNS_ENV)
    return Path(v) if v else Path(run_dir).parent


def _rel(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def judged_record(root: Path, mode: str, doc: dict) -> dict:
    """The mode's judged checkpoint now (``train.judged_checkpoint``) and whether it is final: every run at its
    budget, or a 'judge --final' record in this run's run.json at the same step count."""
    now = train.judged_checkpoint(Path(root), mode)
    rec = doc.get("judged_checkpoint") or {}
    recorded = bool(rec.get("final")) and now["steps"] is not None and rec.get("steps") == now["steps"]
    return dict(rule=train.JUDGED_RULE, steps=now["steps"], budget=int(now["budget"]), runs=sorted(now["runs"]),
                every_run_at_budget=bool(now["final"]), judge_final_recorded=recorded,
                final=bool(now["final"] or recorded))


ENV_CONFIG_KEYS = ("rate", "tasks", "task_weights", "max_seconds", "randomize", "friction", "mass", "kp",
                   "latency_steps", "noise_scale", "push_prob", "push_frac", "push_s", "start_joint_noise",
                   "start_root_velocity_noise", "gains", "pile_grid")


def run_settings(doc: dict, path: Path, root: Path) -> dict:
    """What a row names of its run (run.json): the settings it trained with, its env config (the same in every env
    but the floor) and R3's gains."""
    s = doc["settings"]
    cfgs = doc["env_configs"]
    c0 = cfgs[0]
    for c in cfgs[1:]:
        diff = [k for k in ENV_CONFIG_KEYS if c.get(k) != c0.get(k)]
        if diff:
            raise PolicyRowError(f"{path}: the envs' configs differ in {diff} (a policy row runs one config)")
    return dict(run_json=_rel(path, root), log_std_init=s["log_std_init"], learning_rate=s["learning_rate"],
                design2=s["design2"], n_envs=s["n_envs"], ppo_overrides=s.get("ppo_overrides"),
                config=s.get("config"), checkpoint_every=s.get("checkpoint_every"), budget=doc.get("budget"),
                env=dict({k: c0.get(k) for k in ENV_CONFIG_KEYS}, floors=[c["floor"] for c in cfgs],
                         floor_rule=s.get("floor_rule")),
                gains=doc.get("gains"), versions=dict((doc.get("sessions") or [{}])[0].get("versions") or {}))


def resolve(body: str, mode: str, seed: int, run_dir: Path, choice=None) -> dict:
    """A policy row's checkpoint (``CHECKPOINT_SELECTION_RULE``): the run's files, step count, selection and
    settings, checked against what the panels run (the commanded rate, the start noise, the env tasks)."""
    if mode not in MODES:
        raise PolicyRowError(f"mode {mode!r}: {', '.join(MODES)}")
    root = runs_dir(run_dir).resolve()
    rd = train.run_dir_of(root, body, mode, seed)
    path = rd / "run.json"
    if not path.exists():
        raise PolicyRowError(f"{path}: no {mode} run of PPO seed {seed} for {body} ({RUNS_ENV} names the training run "
                             "directory)")
    doc = json.loads(path.read_text())
    if (doc.get("mode"), doc.get("body"), doc.get("seed")) != (mode, body, int(seed)):
        raise PolicyRowError(f"{path} is {doc.get('mode')} {doc.get('body')} seed {doc.get('seed')}")
    have = train.checkpoints(rd)
    choice = checkpoint_choice(mode) if choice is None else _parse_choice(str(choice))
    judged = None
    if choice == "judged":
        judged = judged_record(root, mode, doc)
        steps, final = judged["steps"], judged["final"]
        selection = "the judged checkpoint (train.judged_checkpoint)"
        if steps is None:
            raise PolicyRowError(f"{root}: no checkpoint every {mode} run reached (train.JUDGED_RULE); set "
                                 f"{CHECKPOINT_ENV} for an interim run")
    elif choice == "newest":
        if not have:
            raise PolicyRowError(f"{rd}: no complete checkpoint (the checkpoints are git-ignored: {RUNS_ENV} names "
                                 "the directory training wrote them to)")
        steps, final = have[-1], False
        selection = f"the run's newest complete checkpoint ({CHECKPOINT_ENV}; interim, not the judgement)"
    else:
        steps, final = int(choice), False
        selection = f"a step count given by {CHECKPOINT_ENV} (interim, not the judgement)"
    if steps not in have:
        raise PolicyRowError(f"{rd}: no complete checkpoint at {steps} steps (complete: {have or 'none'}; the "
                             f"checkpoints are git-ignored: {RUNS_ENV} names the directory training wrote them to)")
    run = run_settings(doc, path, root)
    e = run["env"]
    bad = []
    if e["rate"] != panels.RATE:
        bad.append(f"commanded rate {e['rate']} (the panels' {panels.RATE})")
    if (e["start_joint_noise"], e["start_root_velocity_noise"]) != (tune.START_JOINT_NOISE_RAD,
                                                                    tune.START_ROOT_VELOCITY_NOISE_M_S):
        bad.append("start noise other than the panels' (tune.START_JOINT_NOISE_RAD, START_ROOT_VELOCITY_NOISE_M_S)")
    if set(e["tasks"] or ()) - set(env.TASKS):
        bad.append(f"tasks {sorted(set(e['tasks']) - set(env.TASKS))} unknown to env.TASKS")
    if mode == "R3" and not run["gains"]:
        bad.append("no tracker gains")
    if not run["versions"]:
        bad.append("no recorded library versions")
    if bad:
        raise PolicyRowError(f"{path}: trained with {'; '.join(bad)}: the panels cannot run it as training ran it")
    check_versions(run, str(path))
    ck = rd / "checkpoints"
    files = {"model": ck / f"model_{steps}.zip", "vecnormalize": ck / f"vecnormalize_{steps}.pkl"}
    return dict(mode=mode, body=body, seed=int(seed), steps=int(steps), selection=selection, final=bool(final),
                judged=judged, root=str(root),
                files={k: dict(file=_rel(f, root), sha256=plant.sha256(f)) for k, f in files.items()}, run=run)


def _file_sha256(path: Path) -> dict:
    """{path relative to the study directory (else absolute): sha256}, as ``panels.module_sha256`` names files."""
    path = Path(path).resolve()
    try:
        name = path.relative_to(STUDY).as_posix()
    except ValueError:
        name = str(path)
    return {name: plant.sha256(path)}


def library_versions() -> dict:
    """This process's versions of ``RUN_VERSIONS`` (installed packages; None if missing), as ``train._versions``
    names them."""
    import platform

    out = {"python": platform.python_version()}
    for name in RUN_VERSIONS[1:]:
        try:
            out[name] = metadata.version(name.replace("_", "-"))
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def check_versions(run: dict, where: str) -> None:
    """``CHECKPOINT_SELECTION_RULE``: this process's libraries must be the run's."""
    have, want = library_versions(), run.get("versions") or {}
    diff = {k: (have.get(k), want.get(k)) for k in RUN_VERSIONS if have.get(k) != want.get(k)}
    if diff:
        said = ", ".join(f"{k} {a} here, {b} in training" for k, (a, b) in diff.items())
        raise PolicyRowError(f"{where}: the libraries differ from the run's ({said}): run the policy rows in the PPO "
                             "venv (train.PPO_VENV)")


def _require_ppo() -> None:
    missing = [m for m in ("torch", "stable_baselines3") if importlib.util.find_spec(m) is None]
    if missing:
        raise PolicyRowError(f"the policy rows need {', '.join(missing)}: run them in the PPO venv (train.PPO_VENV)")


# ------------------------------------------------------------------------------------------------
# The env's functions over a panel episode (POLICY_RULE)
# ------------------------------------------------------------------------------------------------
def env_task(plan) -> tuple[str | None, str]:
    """The env task whose track plan ``plan`` is (None: none) and the stride reference whose period the phase input
    runs over (``PERIOD_RULE``)."""
    items = [tuple(x) for x in plan]
    for task, (tplan, ref) in env.TASKS.items():
        if [tuple(x) for x in tplan] == items:
            return task, ref
    return None, env.TASKS["walk"][1]


def filter_alpha(dt: float) -> float:
    """The 10 Hz low-pass's coefficient per env step, as ``env.Stage1Env`` computes it."""
    dt_env = env.ENV_SUBSTEPS * dt
    return 1.0 - math.exp(-2.0 * math.pi * env.FILTER_HZ * dt_env)


def residual_scale(p: plant.Plant) -> np.ndarray:
    """R3's joint-offset scale per joint, as ``env.ResidualEnv._setup`` sets it (legs, neck and tail)."""
    legs = np.array([not jn.startswith(("neck", "tail")) for jn in p.joints])
    return np.where(legs, env.RESIDUAL_LEG_RAD, env.RESIDUAL_NECK_TAIL_RAD)


def episode_attrs(view, built: panels.Built, seed: int, task: str, period: dict, cfg: env.EnvConfig) -> None:
    """The attributes env.py's observation functions read of an env, set on ``view`` for one panel episode: the
    nominal weight and model factors (no randomisation), the condition's floor, friction and push, the phase's
    period (``PERIOD_RULE``)."""
    p = built.p
    view.nj = len(p.joints)
    view.dt = float(p.m.opt.timestep)
    view.dt_env = env.ENV_SUBSTEPS * view.dt
    view.H = float(p.meta["H_m"])
    view.weight = float(p.m.body_subtreemass[p.trunk]) * env.GRAVITY
    view.omega = built.track.omega
    view._R_home = p._R_home
    view.cfg = cfg
    view.task, view.period = task, dict(period)
    view.factors = dict(friction=built.cond.friction, mass=1.0, kp=1.0, latency_steps=0, noise_scale=1.0)
    view.push = built.push(seed)


class ResidualView:
    """R3's env view of a panel episode: ``env.ResidualEnv``'s observation and action functions over the panel's
    tracker ``tk`` (its estimator, reference clock and residual) and plant."""

    _observation = env.Stage1Env._observation
    _command = env.Stage1Env._command
    _true_dcm_error = env.Stage1Env._true_dcm_error
    _estimate = env.ResidualEnv._estimate
    _clock = env.ResidualEnv._clock
    _apply_action = env.ResidualEnv._apply_action

    def __init__(self, tk: control.Tracker, built: panels.Built, seed: int, task: str, period: dict,
                 cfg: env.EnvConfig, **_):
        self.tk, self.p, self.track = tk, tk.p, tk.track
        episode_attrs(self, built, seed, task, period, cfg)
        self.residual_scale = residual_scale(tk.p)


class PolicyOnlyTracker(control.Tracker):
    """R4's controller in ``control.run`` (``R4_RULE``): a ``control.Tracker`` with no feedback whose reset and physics
    step are ``env.PolicyOnlyEnv``'s own, and which is its own env view (``env.PolicyOnlyEnv``'s observation and
    action functions read and set its estimator, clock and joint targets)."""

    _observation = env.Stage1Env._observation
    _command = env.Stage1Env._command
    _true_dcm_error = env.Stage1Env._true_dcm_error
    _estimate = env.PolicyOnlyEnv._estimate
    _clock = env.PolicyOnlyEnv._clock
    _apply_action = env.PolicyOnlyEnv._apply_action
    _begin = env.PolicyOnlyEnv._begin
    _physics_step = env.PolicyOnlyEnv._physics_step

    def __init__(self, built: panels.Built, seed: int, q_home, span, task: str, period: dict, cfg: env.EnvConfig,
                 **_):
        super().__init__(built.p, built.track, "R2", terms=(), ankle_cap=(), rate=cfg.rate)
        episode_attrs(self, built, seed, task, period, cfg)
        self.nominal_model = built.p.m  # the estimator's model (R4_RULE)
        self.q_home, self.span = np.asarray(q_home, float), np.asarray(span, float)
        self.q_target = self.q_home.copy()
        self.tau = 0.0

    @property
    def tau_ref(self) -> float:
        """The reference clock (``env.PolicyOnlyEnv``'s ``tau``), which ``control.run`` reads as the tracker's."""
        return self.tau

    @tau_ref.setter
    def tau_ref(self, value: float) -> None:
        self.tau = value

    def reset(self, start: str = "stance", velocity_offset=None, joint_noise: float = 0.0, seed: int | None = None,
              root_velocity_noise: float = 0.0, noise_seed: int | None = None) -> None:
        """The tracker's episode state (what ``control.run`` reads), then ``env.PolicyOnlyEnv._begin`` with
        ``control.run``'s start noise and seeds."""
        if start != "stance" or velocity_offset is not None:
            raise ValueError("R4 starts from the settled stance at rest, as env.PolicyOnlyEnv does")
        super().reset(start, None, joint_noise, seed, root_velocity_noise, noise_seed)
        seed = self.seed if seed is None else int(seed)
        noise_seed = seed if noise_seed is None else int(noise_seed)
        self.cfg = dataclasses.replace(self.cfg, start_joint_noise=float(joint_noise),
                                       start_root_velocity_noise=float(root_velocity_noise))
        self._begin(dict(start=seed, noise=noise_seed))
        self.state = control.EstimatedState(self.est)

    def step(self) -> None:
        self._physics_step()


# ------------------------------------------------------------------------------------------------
# The trained policy (a panels.Policy)
# ------------------------------------------------------------------------------------------------
class TrainedPolicy(panels.Policy):
    """A checkpoint's deterministic policy acting as training's env acts (``POLICY_RULE``): ``prepare(ctx, cond)``
    before an episode (the condition's env task, period and config), ``begin`` and ``act`` at the panels' hooks
    (``panels._Acting``).  The model loads on first use, in the process that runs the episode, after its files'
    sha256 is checked.  ``marks`` (a list, None: off) receives (physics steps done, CPU s, wall s) at every action
    (``measure_costs``)."""

    every = env.ENV_SUBSTEPS

    def __init__(self, ckpt: dict):
        self.ckpt = ckpt
        self.mode = ckpt["mode"]
        self._model = self._vecnorm = None
        self._episode = None
        self.marks = None

    def describe(self) -> dict:
        c = self.ckpt
        return dict(policy=type(self).__name__, rule=POLICY_RULE, rows_rule=POLICY_ROWS_RULE, period_rule=PERIOD_RULE,
                    r4_rule=R4_RULE if self.mode == "R4" else None, mode=c["mode"], body=c["body"], ppo_seed=c["seed"],
                    checkpoint=dict(steps=c["steps"], selection=c["selection"], final=c["final"], judged=c["judged"],
                                    model=c["files"]["model"], vecnormalize=c["files"]["vecnormalize"],
                                    rule=CHECKPOINT_SELECTION_RULE, checkpoint_rule=train.CHECKPOINT_RULE),
                    run=c["run"], observation=dict(actor_dim=env.ACTOR_DIM, obs_dim=env.OBS_DIM,
                                                   actor=[list(x) for x in env.ACTOR_OBS],
                                                   critic=[list(x) for x in env.CRITIC_OBS]),
                    action=dict(dim=env.ACTION_DIM, substeps=env.ENV_SUBSTEPS, filter_hz=env.FILTER_HZ,
                                residual_leg_rad=env.RESIDUAL_LEG_RAD,
                                residual_neck_tail_rad=env.RESIDUAL_NECK_TAIL_RAD, residual_rate=env.RESIDUAL_RATE,
                                policy_only_span_rule=env.POLICY_ONLY_SPAN_RULE if self.mode == "R4" else None),
                    deterministic=True, latency_steps=0, noise_scale=1.0, randomize=False, torch_threads=1,
                    libraries={k: c["run"]["versions"].get(k) for k in LIBRARIES})

    # ------------------------------------------------------------------ the checkpoint
    def load(self):
        """The model and its VecNormalize (``CHECKPOINT_SELECTION_RULE``: refused if a file's sha256 differs)."""
        if self._model is None:
            check_versions(self.ckpt["run"], f"{self.ckpt['mode']} {self.ckpt['body']} seed {self.ckpt['seed']}")
            root = Path(self.ckpt["root"])
            paths = {k: root / v["file"] for k, v in self.ckpt["files"].items()}
            for k, path in paths.items():
                got = plant.sha256(path)
                if got != self.ckpt["files"][k]["sha256"]:
                    want = self.ckpt["files"][k]["sha256"]
                    raise PolicyRowError(f"{path}: sha256 {got[:12]}..., the row names {want[:12]}...: the checkpoint "
                                         "changed since the row was built")
            import torch
            from stable_baselines3 import PPO

            from .policy import PrivilegedCriticPolicy

            torch.set_num_threads(1)
            model = PPO.load(str(paths["model"]), device="cpu")
            with open(paths["vecnormalize"], "rb") as f:
                vecnorm = pickle.load(f)  # (VecNormalize without its venv: normalize_obs needs none)
            vecnorm.training = False
            pol = model.policy
            bad = []
            if not isinstance(pol, PrivilegedCriticPolicy) or pol.actor_dim != env.ACTOR_DIM:
                bad.append(f"policy {type(pol).__name__} (actor_dim {getattr(pol, 'actor_dim', None)})")
            if tuple(model.observation_space.shape) != (env.OBS_DIM,):
                bad.append(f"observation shape {model.observation_space.shape}")
            if tuple(model.action_space.shape) != (env.ACTION_DIM,):
                bad.append(f"action shape {model.action_space.shape}")
            if not vecnorm.norm_obs or tuple(vecnorm.obs_rms.mean.shape) != (env.OBS_DIM,):
                bad.append("observation normalisation")
            if bad:
                raise PolicyRowError(f"{paths['model']}: {'; '.join(bad)}: not a Stage 1 policy of env.py's layout")
            self._model, self._vecnorm = model, vecnorm
        return self._model, self._vecnorm

    def action(self, obs: np.ndarray) -> np.ndarray:
        """The deterministic action of an observation, as ``train.monitor`` asks it."""
        model, vecnorm = self.load()
        a, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
        return a[0]

    # ------------------------------------------------------------------ an episode
    def prepare(self, ctx: panels.Context, cond: panels.Condition) -> dict:
        """The condition's env task, the phase's period (``PERIOD_RULE``) and the env config the view reads."""
        b = ctx.built(cond)
        task, ref = env_task(b.plan)
        e = self.ckpt["run"]["env"]
        cfg = env.EnvConfig(body=self.ckpt["body"], floor=cond.floor, pile_grid=None, run_dir=str(ctx.run_dir),
                            tasks=tuple(e["tasks"]), rate=float(e["rate"]), randomize=False, push_prob=0.0,
                            start_joint_noise=float(e["start_joint_noise"]),
                            start_root_velocity_noise=float(e["start_root_velocity_noise"]))
        name = task or "panel"
        self._episode = dict(task=name, period={name: ctx.planner.period(ref)}, cfg=cfg, stride_reference=ref)
        return self._episode

    def begin(self, tk, built, seed):
        if self._episode is None:
            raise PolicyRowError("TrainedPolicy.prepare(ctx, cond) must run before an episode")
        v = tk if isinstance(tk, PolicyOnlyTracker) else ResidualView(tk, built, seed, **self._episode)
        v.a_prev = np.zeros(env.ACTION_DIM)
        self.view, self.a_filt, self.alpha = v, np.zeros(env.ACTION_DIM), filter_alpha(v.dt)
        self._act(tk)

    def act(self, tk, k):
        self._act(tk)

    def _act(self, tk) -> None:
        """One env step's action (``POLICY_RULE``): env.Stage1Env.step's clip and filter and the mode's action
        function, on the observation of the state now."""
        v = self.view
        a = np.clip(np.asarray(self.action(v._observation()), float), -1.0, 1.0)
        self.a_filt = self.a_filt + self.alpha * (a - self.a_filt)
        v._apply_action(self.a_filt)
        v.a_prev = a
        if self.marks is not None:
            self.marks.append((int(tk.k), time.process_time(), time.perf_counter()))


# ------------------------------------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------------------------------------
def _policy_sources() -> dict:
    return {**panels.module_sha256(env), **_file_sha256(Path(__file__)), **_file_sha256(POLICY_MODULE)}


class PolicyTrackerRow(panels.TrackerRow):
    """R3/<s> (``POLICY_ROWS_RULE``): ``panels.TrackerRow`` in mode R2 with a ``TrainedPolicy``; its episodes are the
    tracker row's (``TrackerRow.episode``) after the policy's ``prepare``."""

    def __init__(self, name: str, gains, gains_info: dict, policy: TrainedPolicy, controller: str):
        super().__init__(name, "R2", gains, gains_info, policy=policy, controller=controller)

    def sources(self) -> dict:
        return _policy_sources()

    def spec(self) -> dict:
        return dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name,
                    gains=self.gains.tolist(), gains_info=self.gains_info, controller=self.controller,
                    ckpt=self.policy.ckpt)

    @classmethod
    def from_spec(cls, s: dict) -> "PolicyTrackerRow":
        return cls(s["name"], s["gains"], s["gains_info"], TrainedPolicy(s["ckpt"]), s["controller"])

    def episode(self, ctx, cond, seed, relax=()):
        self.policy.prepare(ctx, cond)
        return super().episode(ctx, cond, seed, relax)


def task_files(tasks) -> list[str]:
    """The reference and transition files the env tasks' tracks load (R4's span), with refs/nominal.npz."""
    out = {"refs/nominal.npz"}
    for t in tasks:
        for item in env.TASKS[t][0]:
            if item[0] == "ref":
                out.add(f"refs/{item[1]}.npz")
            elif item[0] == "transition":
                out.add(f"transitions/{item[1]}.npz")
    return sorted(out)


class PolicyOnlyRow(panels.Row):
    """R4/<s> (``POLICY_ROWS_RULE``, ``R4_RULE``): PPO seed s's policy-only policy on ``PolicyOnlyTracker`` through
    ``control.run``.  ``span_inputs`` are the sha256 of the env tasks' references and transitions the span's tracks
    load (checked again where the tracks are built)."""

    hard_only = False

    def __init__(self, name: str, policy: TrainedPolicy, span_inputs: dict):
        self.name, self.policy, self.span_inputs = name, policy, dict(span_inputs)
        self._maps: dict = {}

    def describe(self) -> dict:
        c = self.policy.ckpt
        return dict(row=self.name, controller=f"R4: PPO seed {c['seed']}'s policy-only policy at {c['steps']:,} steps, "
                                              "the joint targets held by the servos alone (no tracker)",
                    tracker_mode=None, terms=[], ankle_cap=[], inward_bound=None, rate=float(c["run"]["env"]["rate"]),
                    gains=None, gains_info=dict(source="R4 runs no tracker: no gains"), hard_only=self.hard_only,
                    mapping=dict(rule=R4_RULE, span_rule=env.POLICY_ONLY_SPAN_RULE,
                                 span_factor=env.POLICY_ONLY_SPAN_FACTOR, min_span_rad=env.POLICY_ONLY_MIN_SPAN_RAD,
                                 tasks=list(c["run"]["env"]["tasks"]), inputs=self.span_inputs),
                    policy=self.policy.describe())

    def sources(self) -> dict:
        return _policy_sources()

    def spec(self) -> dict:
        return dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name, ckpt=self.policy.ckpt,
                    span_inputs=self.span_inputs)

    @classmethod
    def from_spec(cls, s: dict) -> "PolicyOnlyRow":
        return cls(s["name"], TrainedPolicy(s["ckpt"]), s["span_inputs"])

    def mapping(self, ctx: panels.Context, built: panels.Built) -> tuple[np.ndarray, np.ndarray]:
        """(home, span) on the condition's plant: ``env.PolicyOnlyEnv._setup`` over the run's env tasks' tracks built
        on that plant (one per settled stance, kept for the worker's later episodes)."""
        p = built.p
        key = (np.asarray(p.q_settled).tobytes(), np.asarray(p.qadr).tobytes(), len(p.joints))
        if key not in self._maps:
            inp, _ = tune.read_inputs(ctx.run_dir, sorted(self.span_inputs))
            if inp["files"] != self.span_inputs:
                raise PolicyRowError(f"{ctx.run_dir}: the env tasks' references or transitions differ from the row's")
            tasks = self.policy.ckpt["run"]["env"]["tasks"]
            view = types.SimpleNamespace(p=p, nj=len(p.joints), tracks={
                t: control.build_track(p, ctx.run_dir, env.TASKS[t][0], built.track.omega) for t in tasks})
            env.PolicyOnlyEnv._setup(view)
            self._maps[key] = (view.q_home.copy(), view.span.copy())
        return self._maps[key]

    def episode(self, ctx, cond, seed, relax=()):
        b = ctx.built(cond)
        q_home, span = self.mapping(ctx, b)
        e = self.policy.prepare(ctx, cond)
        tk = PolicyOnlyTracker(b, seed, q_home, span, **e)
        rec = b.recorder()
        hook = panels._Acting(rec, self.policy, b, seed)
        s = panels.episode_seeds(cond, seed)
        out = control.run(tk, b.seconds, start="stance", seed=s["start"], noise_seed=s["noise"], recorder=hook,
                          joint_noise=tune.START_JOINT_NOISE_RAD,
                          root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S, push=b.push(seed), relax=tuple(relax))
        out["mode"] = "R4"
        return b, tune.warned_fall(out, tune.sim_warnings(b.p.d)), rec


def _seed_of(name: str) -> int:
    ctrl, _, s = name.partition("/")
    if not s.isdigit():
        raise PolicyRowError(f"row {name!r}: {ctrl}/<PPO seed>, e.g. {ctrl}/42")
    return int(s)


def check_gains(gains, info: dict, ckpt: dict) -> None:
    """R3's gains must be the run's (``POLICY_ROWS_RULE``): every value equal, and the same candidate, CMA-ES seed and
    status."""
    want = ckpt["run"].get("gains") or {}
    values = want.get("values") or {}
    have = control.gains_dict(gains)
    diff = sorted(n for n in set(have) | set(values) if n not in values or n not in have
                  or float(values[n]) != have[n])
    other = [k for k in ("candidate", "cma_seed", "status") if want.get(k) != info.get(k)]
    if diff or other:
        raise PolicyRowError(f"{ckpt['body']} R3 seed {ckpt['seed']}: env.tuned_gains gives {info.get('source')}, the "
                             f"run trained on {want.get('source')} (differences: {diff + other}); refusing")


def r3_row(body: str, run_dir: Path, name: str, choice=None) -> PolicyTrackerRow:
    """The row R3/<s> (``POLICY_ROWS_RULE``)."""
    _require_ppo()
    seed = _seed_of(name)
    ckpt = resolve(body, "R3", seed, run_dir, choice)
    gains, info = env.tuned_gains(Path(run_dir), body)
    check_gains(gains, info, ckpt)
    controller = (f"R3: R2 (the tracker on the estimate) with the body-level G5 selection {info.get('candidate')} "
                  f"plus PPO seed {seed}'s residual policy at {ckpt['steps']:,} steps")
    return PolicyTrackerRow(name, gains, info, TrainedPolicy(ckpt), controller)


def r4_row(body: str, run_dir: Path, name: str, choice=None) -> PolicyOnlyRow:
    """The row R4/<s> (``POLICY_ROWS_RULE``)."""
    _require_ppo()
    ckpt = resolve(body, "R4", _seed_of(name), run_dir, choice)
    inp, _ = tune.read_inputs(Path(run_dir), task_files(ckpt["run"]["env"]["tasks"]))
    return PolicyOnlyRow(name, TrainedPolicy(ckpt), inp["files"])


ROW_FACTORIES = {"R3": r3_row, "R4": r4_row}
for _mode, _factory in ROW_FACTORIES.items():
    panels.register_rows(_mode, _factory)


def policy_row(body: str, run_dir: Path, name: str, choice=None) -> panels.Row:
    """R3/<s> or R4/<s> with an explicit checkpoint selection (``CHECKPOINT_SELECTION_RULE``; None: the variable's)."""
    mode = panels.controller_of(name)
    if mode not in ROW_FACTORIES:
        raise PolicyRowError(f"row {name!r}: R3/<s> or R4/<s>")
    return ROW_FACTORIES[mode](body, Path(run_dir), name, choice)


# ------------------------------------------------------------------------------------------------
# Equivalence with training's env (EQUIVALENCE_RULE)
# ------------------------------------------------------------------------------------------------
@contextmanager
def digesting(p: plant.Plant, rows: list):
    """``env._state_digest`` of ``p.d`` after every physics step ``p`` takes (its ``step`` wrapped for the block)."""
    orig = p.step

    def step(n: int = 1) -> None:
        for _ in range(n):
            orig(1)
            rows.append(env._state_digest(p.d))

    p.step = step
    try:
        yield rows
    finally:
        p.__dict__.pop("step", None)


def _aligned(cls, seeds: dict):
    """``cls`` (an env class) whose episodes start from ``seeds`` (start, noise): the harness's alignment."""

    class Aligned(cls):
        def _begin(self, _seeds):
            super()._begin(dict(seeds))

    Aligned.__name__ = Aligned.__qualname__ = f"Aligned{cls.__name__}"
    return Aligned


def equivalence_check(body: str, row: str = "R3/42", condition: str = "walk/nominal", seed: int = panels.SEEDS[0],
                      run_dir: Path | None = None, seconds: float | None = None, choice=None) -> dict:
    """``EQUIVALENCE_RULE`` for one row, condition and episode seed; ``seconds`` cuts the condition (and the env's
    horizon) there."""
    t0 = time.perf_counter()
    run_dir = Path(run_dir) if run_dir is not None else tune.RUN / body
    r = policy_row(body, run_dir, row, choice)
    mode = panels.controller_of(row)
    cond = {c.key: c for c in panels.conditions()}[condition]
    if seconds is not None:
        cond = dataclasses.replace(cond, seconds=float(seconds))
    _, docs = tune.read_inputs(run_dir, panels.panel_files())
    ctx = panels.Context(body, run_dir, docs)
    b = ctx.built(cond)
    task, _ = env_task(b.plan)
    if task is None:
        raise PolicyRowError(f"{condition}: no env task runs its plan (EQUIVALENCE_RULE needs one)")
    panel = []
    with digesting(b.p, panel):
        _, out, _ = r.episode(ctx, cond, int(seed))
    # training's env on the same floor, friction, seeds and checkpoint
    s = panels.episode_seeds(cond, int(seed))
    e_cfg = r.policy.ckpt["run"]["env"]
    tasks = (task,) if mode == "R3" else tuple(e_cfg["tasks"])
    cfg = env.EnvConfig(body=body, floor=cond.floor, pile_grid=None, run_dir=str(run_dir), tasks=tasks,
                        rate=float(e_cfg["rate"]), max_seconds=float(b.seconds), randomize=False, push_prob=0.0,
                        start_joint_noise=tune.START_JOINT_NOISE_RAD,
                        start_root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S)
    E = _aligned(env.ResidualEnv if mode == "R3" else env.PolicyOnlyEnv, dict(start=s["start"], noise=s["noise"]))
    e = E(cfg)
    factors = {}
    if cond.friction is not None and task not in env.TASK_FRICTION:
        factors["friction"] = float(cond.friction)
    trained = []
    e._step_hook = lambda p_: trained.append(env._state_digest(p_.d))
    term = trunc = False
    info = {}
    try:
        obs, _ = e.reset(seed=0, options=dict(task=task, push=None, factors=factors, episode_seed=0))
        while True:
            obs, _, term, trunc, info = e.step(r.policy.action(obs))
            if term or trunc:
                break
    finally:
        e._step_hook = None
    n_p, n_e = len(panel), len(trained)
    horizon = e.n_steps * env.ENV_SUBSTEPS
    planned = int(round(b.seconds / float(b.p.m.opt.timestep)))  # control.run's physics steps
    common = min(planned, horizon)
    first = next((i for i in range(min(n_p, n_e)) if panel[i] != trained[i]), None)
    fell = bool(out["fell"])
    if fell and n_p <= common:
        fall_env_step = -(-n_p // env.ENV_SUBSTEPS)
        ends_alike = bool(term) and e.k_env == fall_env_step
        if first is None and n_e < n_p:
            first = n_e
    else:
        fall_env_step = None
        ends_alike = n_p >= common and n_e >= common and not (term and e.k_env * env.ENV_SUBSTEPS <= common)
        if first is None and min(n_p, n_e) < common:
            first = min(n_p, n_e)
    c = r.policy.ckpt
    return dict(rule=EQUIVALENCE_RULE, body=body, row=row, mode=mode, checkpoint=dict(
                    steps=c["steps"], selection=c["selection"], model=c["files"]["model"],
                    vecnormalize=c["files"]["vecnormalize"]),
                condition=condition, cut_s=seconds, task=task, floor=cond.floor, friction=cond.friction,
                episode_seed=int(seed), seeds=s, panel_seconds=float(b.seconds), env_seconds=float(e.seconds),
                panel_physics_steps=n_p, panel_planned_physics_steps=planned, env_physics_steps=n_e,
                env_horizon_physics_steps=horizon, common_horizon_physics_steps=common, env_steps=e.k_env,
                compared_steps=min(n_p, n_e), first_difference=first, panel_fell=fell,
                panel_fall_cause=out["fall_cause"], panel_fall_time_s=out["fall_time_s"], fall_env_step=fall_env_step,
                env_terminated=bool(term), env_truncated=bool(trunc), env_fall_cause=info.get("fall_cause"),
                ends_alike=bool(ends_alike), identical=bool(first is None and ends_alike),
                wall_s=time.perf_counter() - t0)


# ------------------------------------------------------------------------------------------------
# Verdicts (R4_VERDICT_RULE)
# ------------------------------------------------------------------------------------------------
R4_ROWS = tuple(f"R4/{s}" for s in train.R4_SEEDS)


def r4_verdict(records: list) -> dict:
    """``R4_VERDICT_RULE`` on R4's records (its one seed's)."""
    v = panels.verdict(records)
    out = dict(v, rule=R4_VERDICT_RULE, panels_rule=v["rule"])
    if v["verdict"] == "shown":
        out.update(verdict="provisional", read_from="shown")
    return out


def policy_verdicts(body: str, out: Path) -> dict:
    """Per condition, R3's verdict (``panels.verdict`` on R3/42 and R3/43, as the panels' summary has it) and R4's
    (``r4_verdict``), with the checkpoints the records ran, into ``<out>/<body>/eval/policy_rows.json``."""
    grid = {}
    ckpts = {}
    for cond in panels.conditions():
        rows = {}
        for ctrl, names in (("R3", panels.VERDICT_ROWS["R3"]), ("R4", R4_ROWS)):
            recs = [panels._read(panels.record_path(out, body, r, cond)) for r in names]
            v = panels.verdict(recs) if ctrl == "R3" else r4_verdict(recs)
            rows[ctrl] = dict(v, gated=panels.gated_for(names[0], cond))
            for r, rec in zip(names, recs, strict=True):
                pol = ((rec or {}).get("controller") or {}).get("policy") or {}
                if pol.get("checkpoint"):
                    ckpts.setdefault(r, set()).add((pol["checkpoint"]["steps"], pol["checkpoint"]["selection"],
                                                    bool(pol["checkpoint"]["final"])))
        grid.setdefault(cond.behavior, {})[cond.name] = rows
    counts = {ctrl: {} for ctrl in ("R3", "R4")}
    for conds in grid.values():
        for rows in conds.values():
            for ctrl, v in rows.items():
                if v["gated"]:
                    counts[ctrl][v["verdict"]] = counts[ctrl].get(v["verdict"], 0) + 1
    doc = dict(body=body, rule=R4_VERDICT_RULE, verdict_rule=panels.VERDICT_RULE, gating_rule=panels.GATING_RULE,
               r3_rows=list(panels.VERDICT_ROWS["R3"]), r4_rows=list(R4_ROWS), gated_verdicts=counts,
               checkpoints={r: [dict(steps=s, selection=sel, final=f) for s, sel, f in sorted(v)]
                            for r, v in sorted(ckpts.items())},
               grid=grid, written_at=tune._now())
    panels._write_json(Path(out) / body / "eval" / "policy_rows.json", doc)
    return doc


# ------------------------------------------------------------------------------------------------
# Costs (POLICY_COST_RULE)
# ------------------------------------------------------------------------------------------------
class _Marked(panels.ZeroResidual):
    """R2 bit for bit (``panels.ZeroResidual``), with ``TrainedPolicy``'s timing marks."""

    def __init__(self):
        self.marks = None

    def begin(self, tk, built, seed):
        super().begin(tk, built, seed)
        if self.marks is not None:
            self.marks.append((int(tk.k), time.process_time(), time.perf_counter()))

    def act(self, tk, k):
        super().act(tk, k)
        if self.marks is not None:
            self.marks.append((int(tk.k), time.process_time(), time.perf_counter()))


def measure_costs(body: str, run_dir: Path, rows=("R3/42", "R4/42"), floors=("hard", "low"),
                  skip_s: float = COST_SKIP_S, seconds: float = COST_SAMPLE_S, r2: bool = True, choice=None,
                  log=print) -> dict:
    """``POLICY_COST_RULE``: CPU and wall seconds per simulated second of each row (and R2/42) on each floor."""
    run_dir = Path(run_dir)
    _, docs = tune.read_inputs(run_dir, panels.panel_files())
    ctx = panels.Context(body, run_dir, docs)
    objs = {name: policy_row(body, run_dir, name, choice) for name in rows}
    if r2:
        g, info = panels.r2_gains(run_dir, 42)
        objs["R2/42"] = panels.TrackerRow("R2/42", "R2", g, info, policy=_Marked())
    out = {}
    for floor in floors:
        t0 = time.perf_counter()
        ctx.plant(floor, None)
        build = time.perf_counter() - t0
        cond = panels.Condition("walk", f"cost_{floor}", False, "walk", floor=floor, seconds=skip_s + seconds)
        for name, row in objs.items():
            row.policy.marks = marks = []
            c0, w0 = time.process_time(), time.perf_counter()
            b, res, _ = row.episode(ctx, cond, panels.SEEDS[0])
            c1, w1 = time.process_time(), time.perf_counter()
            row.policy.marks = None
            dt = float(b.p.m.opt.timestep)
            mark = next((m for m in marks if m[0] * dt >= skip_s - 1e-9), None)
            steady = mark is not None and res["seconds"] - mark[0] * dt >= 1.0
            if not steady:
                mark = marks[0] if marks else (0, c0, w0)
            sim = max(float(res["seconds"]) - mark[0] * dt, 1e-9)
            r = dict(cpu_per_sim_s=(c1 - mark[1]) / sim, wall_per_sim_s=(w1 - mark[2]) / sim, simulated_s=sim,
                     cpu_s=c1 - mark[1], wall_s=w1 - mark[2], fell=bool(res["fell"]), fall_time_s=res["fall_time_s"],
                     part=(f"simulated {mark[0] * dt:g}-{res['seconds']:.2f} s" if steady else
                           f"simulated {mark[0] * dt:g}-{res['seconds']:.2f} s (from the first action: the walk fell "
                           f"at {res['fall_time_s']})"),
                     episode_cpu_s=c1 - c0, episode_wall_s=w1 - w0, load_average=list(os.getloadavg()))
            out.setdefault(name, {})[floor] = dict(r, plant_build_s=build)
            log(f"{body} {name} {floor}: {r['cpu_per_sim_s']:.3f} CPU s and {r['wall_per_sim_s']:.3f} wall s per "
                f"simulated s over {r['part']} (load {r['load_average'][0]:.1f})")
    ck = {n: dict(steps=o.policy.ckpt["steps"], selection=o.policy.ckpt["selection"]) for n, o in objs.items()
          if isinstance(o.policy, TrainedPolicy)}
    return dict(rule=POLICY_COST_RULE, body=body, rows=out, checkpoints=ck, skip_s=skip_s, seconds=seconds,
                cpu_count=os.cpu_count(), measured_at=tune._now())


def floor_costs(costs: dict) -> dict:
    """A row's costs per floor for ``panels.dry_run`` (``POLICY_COST_RULE``: unmeasured floors from measured ones)."""
    fl = {f: dict(cpu_per_sim_s=v["cpu_per_sim_s"], wall_per_sim_s=v["wall_per_sim_s"]) for f, v in costs.items()}
    if "low" in fl:
        for f in ("medium", "high"):
            fl.setdefault(f, dict(fl["low"], assumed="low pile's cost"))
        if "hard" in fl and "robot_floor_carpet" not in fl:
            fl["robot_floor_carpet"] = dict(
                cpu_per_sim_s=0.5 * (fl["hard"]["cpu_per_sim_s"] + fl["low"]["cpu_per_sim_s"]),
                wall_per_sim_s=0.5 * (fl["hard"]["wall_per_sim_s"] + fl["low"]["wall_per_sim_s"]),
                assumed="the mean of hard floor's and low pile's")
    return fl


def project(body: str, out: Path, costs: dict, rows=("R3/42", "R3/43", "R4/42"), inputs: Path | None = None,
            workers: int = 1) -> dict:
    """The panels' core-hours of ``rows`` at measured costs (``costs``: row or mode -> floor -> cost;
    ``panels.dry_run``), with the attribution's bound for the attributed rows."""
    res = {}
    for name in rows:
        c = costs.get(name) or costs.get(panels.controller_of(name))
        if c is None:
            raise PolicyRowError(f"no measured cost for {name}")
        doc = panels.dry_run(body, Path(out), rows=(name,), inputs=inputs, measure=False,
                             costs=dict(floors=floor_costs(c)), workers=workers, log=lambda m: None)
        t = doc["totals"]
        res[name] = dict(episodes=int(t["episodes"]), simulated_h=t["simulated_s"] / 3600, core_h=t["core_h"],
                         attribution_bound_core_h=t["attribution_bound_core_h"],
                         core_h_with_attribution_bound=t["core_h"] + t["attribution_bound_core_h"],
                         wall_h_1_worker=t["wall_h_to_run_1_worker"], floors=floor_costs(c),
                         conditions={x["condition"]: dict(core_h=x["core_h"], episodes=x["episodes"],
                                                          seconds=x["seconds"], floor=x["floor"])
                                     for x in doc["episodes"]})
    return dict(rule=POLICY_COST_RULE, cost_rule=panels.COST_RULE, body=body, rows=res,
                core_h=sum(v["core_h"] for v in res.values()),
                core_h_with_attribution_bound=sum(v["core_h_with_attribution_bound"] for v in res.values()))


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage1.policy_rows", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    vp = sub.add_parser("verdicts", help="R3's and R4's verdicts per condition (R4_VERDICT_RULE)")
    vp.add_argument("--body", action="append", choices=sorted(panels.BODIES), required=True)
    vp.add_argument("--out", required=True, help="run directory holding <body>/eval")
    cp = sub.add_parser("check", help="EQUIVALENCE_RULE: a row's episode against training's env")
    cp.add_argument("--body", choices=sorted(panels.BODIES), required=True)
    cp.add_argument("--row", default="R3/42")
    cp.add_argument("--condition", default="walk/nominal")
    cp.add_argument("--episode-seed", type=int, default=panels.SEEDS[0])
    cp.add_argument("--seconds", type=float, default=None, help="cut the condition here (default: its whole length)")
    cp.add_argument("--checkpoint", default=None, help=f"judged, newest or a step count (default: {CHECKPOINT_ENV})")
    cp.add_argument("--inputs", default=str(tune.RUN), help="run directory holding <body>/refs and transitions")
    kp = sub.add_parser("costs", help="POLICY_COST_RULE: CPU and wall seconds per simulated second, and core-hours")
    kp.add_argument("--body", action="append", choices=sorted(panels.BODIES), required=True)
    kp.add_argument("--rows", default="R3/42,R4/42")
    kp.add_argument("--floors", default="hard,low")
    kp.add_argument("--skip", type=float, default=COST_SKIP_S)
    kp.add_argument("--seconds", type=float, default=COST_SAMPLE_S)
    kp.add_argument("--checkpoint", default=None, help=f"judged, newest or a step count (default: {CHECKPOINT_ENV})")
    kp.add_argument("--inputs", default=str(tune.RUN), help="run directory holding <body>/refs and transitions")
    kp.add_argument("--project", default="R3/42,R3/43,R4/42", help="rows whose panels' core-hours to project")
    kp.add_argument("--json", default=None, help="write the measurements and projections here")
    a = ap.parse_args(argv)
    if getattr(a, "checkpoint", None):
        os.environ[CHECKPOINT_ENV] = a.checkpoint  # (panels.dry_run builds its rows through the factories)
    if a.cmd == "verdicts":
        for body in a.body:
            doc = policy_verdicts(body, Path(a.out))
            print(f"{body}: gated verdicts {json.dumps(doc['gated_verdicts'])}; checkpoints "
                  f"{json.dumps(doc['checkpoints'])}", flush=True)
        return 0
    if a.cmd == "check":
        res = equivalence_check(a.body, a.row, a.condition, a.episode_seed, Path(a.inputs) / a.body, a.seconds,
                                a.checkpoint)
        print(json.dumps(res, indent=1, default=str), flush=True)
        return 0 if res["identical"] else 1
    docs = {}
    for body in a.body:
        rows = tuple(x for x in a.rows.split(",") if x)
        m = measure_costs(body, Path(a.inputs) / body, rows, tuple(a.floors.split(",")), a.skip, a.seconds,
                          choice=a.checkpoint, log=lambda msg: print(msg, flush=True))
        by_mode = {panels.controller_of(n): v for n, v in m["rows"].items()}
        rows_p = tuple(x for x in a.project.split(",") if x)
        proj = project(body, Path(a.inputs), by_mode, rows_p) if rows_p else None
        docs[body] = dict(measured=m, projection=proj)
        if proj:
            for n, v in proj["rows"].items():
                print(f"{body} {n}: {v['episodes']} episodes, {v['simulated_h']:.2f} simulated h, {v['core_h']:.2f} "
                      f"core-h; with the attribution's bound {v['core_h_with_attribution_bound']:.2f}", flush=True)
    if a.json:
        panels._write_json(Path(a.json), docs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
