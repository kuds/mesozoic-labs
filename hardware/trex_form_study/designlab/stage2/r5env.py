"""R5's and R6's training env for 14-joint and 16-joint plants (round C design D1, D6, D8, D10, D11, D16; section
3.2; rules E1-E5, T1-T3, W1-W5, J1-J5).

A new ``gymnasium.Env`` that composes Stage 1's parts and edits none of them: the plant (``plant.build``), the
tracker (``control.Tracker`` in R2 mode with no feedback term and no ankle cap, at the commanded clock, driven
through its own ``residual`` and ``rate_factor`` hooks), the tracker's estimator, the bar's servo delay
(``speedfit.servo_delay``) and ``env.py``'s helpers by name.  The references are an ``r5lib.Library``'s scripts;
the layout, the observation, the action mapping and the reward are ``r5core.Core``'s, which the policy rows share.

**Reset** (``RESET_ORDER``) and **step** (``STEP_ORDER``) follow ``env.Stage1Env``'s statement for statement where
gate G8a needs it: with a zero action, head neutral, the tracker keeps ``residual None`` and ``rate_factor 1.0`` and
every physics step is the bar's R0 row's.

**Reset is history-free** (design D16): every tracker is built at construction, on the nominal model; each reset
restores the model's nominal arrays before anything is chosen or built and asserts that the chosen tracker's damping
and friction-loss copies are the nominal ones; everything an episode draws comes from the env's own generator, in
a fixed order (``DRAW_ORDER``), or from the episode seed.  An env after any number of randomised episodes equals a
fresh env after ``reset(seed)``.

**Mid-task starts** (design D6, rule T3): ``R5Tracker.reset_at`` lends the track the start state for one call of
``control.Tracker.reset("frame")``; the delay queue is refilled with the commands the zero action would have given
over the k physics steps before the start; the servos are set to the oldest of them (k = 0: to the command of one
physics step before the start) and the estimator's first reading is taken from the forward pass on that command.

R6 (``R6Env``; design D8, rule E2) sets joint targets alone through ``r5core.R6Tracker``; the same observation,
commands, reward, starts and randomisation.

**A later freeze of the set** (rule K4): settings that name a freeze are checked against the set as it is read
today (``FREEZE_RULE``), whether the named freeze is the latest or an earlier one.  ``R5Env.describe`` says which
keys of the env description a trainer compares on a resume (``RESUME_KEYS``) and which an accepted later freeze
changes (``LATER_FREEZE_KEYS``); ``env_configs`` names a job's tasks in its configs, so that a later freeze which
makes a further task usable leaves a resumed run its own tasks.

**The servo option** (``R5Config.servo``; ``stage2/design/servo_model/SERVO_MODEL_DESIGN.md`` 5.1, package S2;
``SERVO_RULE``).  Off by default: the env then runs today's code path and prints today's description.  On, every
episode runs under ``servo.served`` at a named setting or a training draw (simulation only: the servo model's
values are a model, none is measured), and the env prints the servo variants of its reset order, draw order and
factor rule (``SERVO_RESET_ORDER``, ``SERVO_DRAW_ORDER``, ``SERVO_FACTOR_RULE``) with the table's sha256.
"""

from __future__ import annotations

import copy
import math
import os
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..stage1 import control, env, estimator, legs, plant, speedfit, tune
from . import r5core, r5lib, s2io, servo, setio

SUBSTEPS = r5core.SUBSTEPS
MODES = ("R5", "R6")
FLOORS = ("hard", "robot_floor_carpet")
#: Design D13 / rule J5: env i of a job runs on ``FLOOR_SPREAD[i % 4]``; no pile.
FLOOR_SPREAD = ("hard", "hard", "hard", "robot_floor_carpet")
INTERFERENCE = ("terminate", "reward", "off")
#: Consecutive env-step ends with a pair below zero that end an episode (rule W4: 40 ms).
INTERFERENCE_STEPS = 2
FACTORS = ("friction", "mass", "kp", "latency_steps", "noise_scale", "joint_damping", "joint_friction")
NOMINAL_FACTORS = dict(friction=None, mass=1.0, kp=1.0, latency_steps=0, noise_scale=1.0, joint_damping=1.0,
                       joint_friction=1.0)

RESET_ORDER = (
    "1 the generator (a seeded reset restarts the episode stream); 2 the model's nominal arrays restored (friction, "
    "servo stiffness, joint damping and friction-loss, the masses and the constants MuJoCo derives from them); 3 the "
    "script (options.plan with options.seconds, else options.task, else a draw by the task weights) and its "
    "tracker, whose damping and friction-loss copies must be the nominal ones; 4 the start (options.start, else "
    "\"mid\" with probability mid_start where the script has mid spans, the reference time uniform on their union "
    "and snapped to a file frame); 5 the length; 6 the factors, drawn in the order friction, mass, kp, "
    "latency_steps, noise_scale, joint_damping, joint_friction, then options.factors, then applied; 7 the push and, "
    "on a body with a neck yaw and a jaw, the head script; 8 the episode seed and its start and noise seeds; 9 "
    "counters; 10 the previous episode's delay left; 11 the start; 12 the delay entered (a mid start refills its "
    "queue, sets the servos to the zero action's command before the start, computes the forward pass and gives "
    "the estimator its first reading from it); 13 the core's baselines, histories and accumulators, the "
    "observation")
STEP_ORDER = (
    "1 the action clipped and low-passed; 2 R5: joint offsets and the clock factor on the tracker (None and exactly "
    "1.0 for a zero action with the head neutral); R6: joint targets; 3 ten times: the push, the tracker's physics "
    "step, the step hook, the core's per-step readings; 4 the fall and any MuJoCo warning, the interference "
    "reading; 5 the counter, the reward; 6 terminated on a fall or on interference at two env-step ends in a row, "
    "truncated at the episode's length; 7 info and the observation")
DRAW_ORDER = (
    "from the env's own generator, in this order: the task; the start (one uniform for stance or mid, then the mid "
    "start's reference time); the factors (friction, mass, kp, latency_steps, noise_scale, joint_damping, "
    "joint_friction); the push (one uniform, then its time, its size and its direction); the head script (one "
    "uniform each for a gaze train and a jaw event, then their numbers).  The start's noise and the sensors' noise "
    "come from the episode seed (env.EPISODE_SEED_RULE), also at a mid start")
MID_START_RULE = (
    "half of the episodes of every task but the stand start mid-task (rule T3): the reference time is uniform over "
    "the task's non-hold segments between 0.2 s after the first non-hold frame and 2.0 s x the clock rate before "
    "the plan's end, snapped to a file frame; the plant is set to that frame (rates x the clock rate) with Stage "
    "1's start noise (uniform on the joints, uniform added to the root's velocity), the estimator is reset from "
    "that state, and the delay queue holds the k commands the zero action would have given before it.  The "
    "servos are then set to the oldest of those (without a delay, k = 0: to the command the zero action would have "
    "given one physics step before the start), the forward pass is computed once on that, and the estimator is "
    "reset again from the same state with a sensor reading of that pass (its foot loads, accelerometer and contact "
    "flags; the plant's first pass ran with the servos at their home targets)")
PUSH_RULE = (
    "with probability push_prob in an episode longer than 4 s: one push_s pulse at the trunk's CoM at a time uniform "
    "in [2 s, end - 2 s], direction uniform, dv = u x the body's Stage 0 forward bound (env.PUSH_BOUND_M_S[body]: "
    "no default), u uniform on push_walking if the reference walks at that time at the commanded clock and on "
    "push_standing if it stands; force = true mass x dv / push_s (rule J3)")
FREEZE_RULE = (
    "rule K4: settings that name a freeze of the set (set_freeze: its manifest's sha256, the freeze the run started "
    "on) are accepted when, as the set is read today under its latest freeze, (1) every file this env's tasks use "
    "(and the nominal walk's) has the sha256 and the entry hash that the named freeze's manifest holds, and (2) "
    "the resolved task table is the same: a task is in the table under a freeze exactly when that freeze's "
    "manifest holds every file of its plan (a manifest lists every available entry), so with (1) the names, plans, "
    "window counts and lengths are the same; settings that name their tasks are read for those tasks alone.  Both "
    "are checked whether the named freeze is the set's latest or an earlier one (a set's records can be edited "
    "after its latest freeze without a new one).  Anything else is refused with what differs: a hash that is none "
    "of the set's freezes, a manifest that is not the one its hash names, a file missing from the named freeze, a "
    "changed file or entry, a task that entered or left the table (a named task that left it included).  Both "
    "freezes are in the env description, each with its index and manifest sha256: set_freeze_started is the named "
    "freeze, by the index of the first freeze of the set that holds that manifest (the settings hold the hash "
    "alone; sets.freeze appends no freeze whose manifest an earlier one holds), and never changes with a later "
    "freeze; set_freeze_now is the latest")
#: What a trainer compares on a resume (``R5Env.describe``; rule K4, the lead's ruling of 16 Nov 2026): the keys of
#: the env description that hold the config with its named freeze, the task names and the input files this env's
#: tasks use.  They are equal in every session of a run, also across an accepted later freeze of the set.
RESUME_KEYS = ("config", "task_names", "tasks", "task_files", "set_freeze_started")
#: The keys of the env description (dotted: a key inside a key) that an accepted later freeze of the set changes:
#: the first three always, the last two where the later freeze made further tasks usable.  Every other key is
#: unchanged by it.
LATER_FREEZE_KEYS = ("set_freeze", "set_freeze_now", "inputs.freeze", "inputs.files", "absent")
FACTOR_RULE = (
    "per episode (rules J1, J2): servo delay k uniform on the integers latency_steps (inclusive), through "
    "speedfit.servo_delay; servo stiffness x kp; sliding friction; trunk mass and inertia x mass (env.MASS_RULE); "
    "joint damping and joint friction-loss, one factor each for all hinges; IMU noise scale.  The controller's "
    "model and feedforward stay nominal: every tracker is built on the nominal model before any episode")

#: ``R5Config.servo``'s value for a training draw per episode (``SERVO_RULE``).
SERVO_DRAW = "draw"
SERVO_RULE = (
    "the servo option (SERVO_MODEL_DESIGN.md 5.1, package S2): R5Config.servo None (the default) is off, and the env "
    "runs today's code path with today's description.  On, every episode runs inside servo.served(plant, the "
    "episode's parameters), the one context manager the judging side's served rows use.  The episode's parameters: "
    "options.servo if given, else the config's; \"draw\" is the training draw servo.training_draw(episode seed), "
    "SeedSequence([episode seed, 2]) (refused with options.seeds and no episode_seed: the env has no episode seed "
    "then, design section 4); a name of servo.NAMED but D is that named setting, which draws nothing; \"D/<s>\" is "
    "the drawn panel's set of bar seed s (servo.panel_draw); options.servo may also be a servo.ServoParams, the one "
    "parameter object the judging side is given.  options.servo with the option off is refused.  The layer is left "
    "first thing in reset, before the env restores its arrays (trap 9); it is entered after the delay is entered and "
    "before a mid-task start fills the delay queue, so that the queue's commands are folded with the plant's gain "
    "copies the layer sets at entry (design 2d, test T14); it acts from the first physics step, so the reset's "
    "forward pass (a mid start's included), the start's first reading and every tracker's copy of the friction are "
    "today's model's (traps 1 and 9).  The layer owns the drives' damping and friction: joint_damping and "
    "joint_friction are 1 (with randomize their ranges are (1, 1), and options.factors may not move them; the draws "
    "stay, so the env's stream does not shift).  A setting with a trunk mass factor (M-, M+, W1, W2) gives the "
    "episode's mass factor, in the place of the drawn one and before the start, as the judging side applies it "
    "before the inner episode (design 5.1); options.factors may not name another.  Every other factor (servo "
    "stiffness x kp, the floor's friction, the delay, the IMU noise) stays the env's, and the layer acts on the "
    "plant as they leave it.  Body 75-M-H-10 only (the table's)")
SERVO_RESET_ORDER = (
    "0 the servo layer left (every model array it wrote, the envelope, the step and the plant's gain copies back, bit "
    "for bit); 1 the generator (a seeded reset restarts the episode stream); 2 the model's nominal arrays restored "
    "(friction, servo stiffness, joint damping and friction-loss, the masses and the constants MuJoCo derives from "
    "them); 3 the script (options.plan with options.seconds, else options.task, else a draw by the task weights) and "
    "its tracker, whose damping and friction-loss copies must be the nominal ones; 4 the start (options.start, else "
    "\"mid\" with probability mid_start where the script has mid spans, the reference time uniform on their union "
    "and snapped to a file frame); 5 the length; 6 the factors, drawn in the order friction, mass, kp, "
    "latency_steps, noise_scale, joint_damping, joint_friction, then options.factors, then a named servo setting's "
    "trunk mass factor, then applied (joint damping and joint friction-loss 1: the layer's); 7 the push and, on a "
    "body with a neck yaw and a jaw, the head script; 8 the episode seed and its start and noise seeds, then the "
    "episode's servo parameters (a named setting, or the training draw from the episode seed); 9 counters; 10 the "
    "previous episode's delay left; 11 the start; 12 the delay entered, then the servo layer entered at the "
    "episode's parameters (the plant's gain copies the run's from here, its model arrays from the first physics "
    "step); a mid start then refills the delay queue (its commands folded with the run's gain copies), sets the "
    "servos to the zero action's command before the start, computes the forward pass on today's model arrays and "
    "gives the estimator its first reading from it; 13 the core's baselines, histories and accumulators, the "
    "observation")
SERVO_DRAW_ORDER = (
    DRAW_ORDER + ".  The servo layer's parameters come from neither: a named setting draws nothing, and the option "
    "\"draw\" draws the episode's set from its own generator, SeedSequence([episode seed, 2]), in servo.DRAW_RULE's "
    f"order (table sha256 {servo.TABLE_SHA256}); the env's own generator is not touched, so the task, the factors, "
    "the push and the head script of an episode are the same with the option on or off")
SERVO_FACTOR_RULE = (
    FACTOR_RULE + ".  With the servo layer on (SERVO_RULE): joint damping and joint friction-loss are 1, since the "
    "layer writes the drives' damping and friction itself (servo.LAYER_RULE); a named setting's trunk mass factor "
    "(M-, M+, W1, W2) is the episode's mass factor; servo stiffness x kp, the floor's friction, the delay and the "
    "IMU noise stay, and the layer's gains (the gain cap, kv by the plant's rule under the run's inertia, the box "
    "servos' kp and kv scales) act on the plant as they leave it; the feedforward keeps today's friction (A18)")
#: The mid-task start's forward pass with the servo option on, in a sentence (the lead's fourth list of 17 Nov 2026,
#: on the builder's (c) of package S2; added in package S3 beside ``SERVO_RULE`` and ``SERVO_RESET_ORDER``, which are
#: not edited).
SERVO_MID_START_RULE = (
    "with the servo option on, once per episode at a mid-task start the first reading's forward pass runs on today's "
    "model arrays with the servos set to the oldest queued command (without a delay, the zero action's command of the "
    "physics step before the start), which was folded with the run's gain copies (the plant's kp and kv where the "
    "run's inertia or the gain cap moves them: the tail's kv in every setting but today, every drive's kv under J+, "
    "W1 and every training draw, the leg drives' kp and kv under G5); no command is served in that pass, and the "
    "run's model gains, damping, friction-loss, inertia and force ranges act from the first physics step.  "
    "SERVO_RULE's words \"today's model's\" are read as the model arrays only")


@dataclass(frozen=True)
class R5Config:
    """One env's settings (design 3.2).  Ranges are uniform (lo, hi); ``randomize`` False draws no factor (the
    nominal plant, no delay, the default IMU noise).  ``tasks`` ``()`` takes every task the set has on the day the
    env is built; a run's stored configs name their tasks (``env_configs`` resolves them), so that a later freeze
    of the set that makes a further task usable changes nothing the run plays.  No field is a path: the set is
    named and pinned by ``set_freeze`` (the manifest sha256 of the freeze the run started on; a later freeze of
    the set is accepted by ``FREEZE_RULE``, rule K4), and ``setio.set_dir`` resolves it under the study
    directory.  ``servo`` None is the servo option off (``SERVO_RULE``: "draw", a named setting or "D/<s>" turn it
    on); ``describe`` then leaves the key out, so that today's dictionary is unchanged."""

    body: str
    set: str = "V6"
    mode: str = "R5"
    floor: str = "hard"
    rate: float = 1.05
    max_seconds: float = 30.0
    tasks: tuple = ()
    task_weights: tuple | None = None
    mid_start: float = 0.5
    randomize: bool = True
    friction: tuple = (0.4, 1.2)
    mass: tuple = (0.9, 1.1)
    kp: tuple = (0.6, 1.2)
    latency_steps: tuple = (0, 12)
    noise_scale: tuple = (0.5, 1.5)
    joint_damping: tuple = (0.5, 1.5)
    joint_friction: tuple = (0.5, 1.5)
    push_prob: float = 0.5
    push_walking: tuple = (0.0, 0.5)
    push_standing: tuple = (0.0, 1.0)
    push_s: float = 0.1
    head_events: bool = True
    gaze_prob: float = 0.5
    jaw_prob: float = 0.25
    start_joint_noise: float = 0.01
    start_root_velocity_noise: float = 0.02
    action_leg_rad: float = 0.35
    action_other_rad: float = 0.15
    action_rate: float = 0.25
    filter_hz: float = 10.0
    heading_input: bool = True
    reward: tuple = r5core.REWARD_WEIGHTS_ITEMS
    interference: str = "terminate"
    set_freeze: str | None = None
    servo: str | None = None

    def describe(self) -> dict:
        doc = {k: _plain(v) for k, v in asdict(self).items()}
        if doc["servo"] is None:  # the option off: today's dictionary (design 5.2, trap 12; test T2)
            del doc["servo"]
        return doc


def _plain(x):
    return [_plain(v) for v in x] if isinstance(x, (tuple, list)) else x


class R5Tracker(control.Tracker):
    """The tracker of an R5 episode: ``control.Tracker`` with a mid-task start (design D6)."""

    def reset_at(self, t_ref: float, qpos, qvel, noise_seed: int) -> None:
        r5core.reset_at(self, t_ref, qpos, qvel, noise_seed)

    def zero_command(self, t_ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, qd, tau) that ``step`` gives ``p.command`` at reference time ``t_ref`` with a zero action: the
        tracker's own lines without a feedback term (``control.Tracker.step``)."""
        x = self.track.joint(t_ref)
        nj = self.nj
        q, qd, tau = x[:nj], x[nj : 2 * nj], x[2 * nj : 3 * nj]
        s = self.rate
        scaled = s != 1.0
        qd_c = s * qd if scaled else qd
        ff = tau + self.damp * qd_c + self.fric * np.tanh(qd_c / control.FRICTION_SMOOTH)
        if self.extra is not None:
            ff = ff + self.extra @ (s * s * x[3 * nj :] if scaled else x[3 * nj :])
        return q, qd_c, ff


class R5Env(gym.Env):
    """Mode "R5" (module docstring).  ``root`` and ``require_frozen`` are the library's (a test's set in a
    temporary directory); they are no part of the settings."""

    metadata = {"render_modes": []}
    mode = "R5"

    def __init__(self, config: R5Config, seed: int | None = None, root=None, require_frozen: bool = True):
        super().__init__()
        self.cfg = cfg = config
        if cfg.mode != self.mode:
            raise ValueError(f"config mode {cfg.mode!r}: this env is {self.mode}")
        if cfg.floor not in FLOORS:
            raise ValueError(f"floor {cfg.floor!r}: choose from {', '.join(FLOORS)} (rule J5: no pile)")
        if cfg.interference not in INTERFERENCE:
            raise ValueError(f"interference {cfg.interference!r}: choose from {', '.join(INTERFERENCE)}")
        if (cfg.start_joint_noise, cfg.start_root_velocity_noise) != (tune.START_JOINT_NOISE_RAD,
                                                                      tune.START_ROOT_VELOCITY_NOISE_M_S):
            raise ValueError("the start noise is Stage 1's (tune.START_JOINT_NOISE_RAD, "
                             "START_ROOT_VELOCITY_NOISE_M_S): the bar's rows start with it")
        if cfg.servo is not None:
            _check_servo_config(cfg)
        self.p = p = plant.build(cfg.body, cfg.floor)
        m = p.m
        #: The model as built: the plant's nominal model and the controller's model throughout (``env.MASS_RULE``).
        self.nominal_model = copy.deepcopy(m)
        changes = env.setconst_changes(m)
        if changes:
            raise plant.PlantError(f"mj_setConst changes the {cfg.body} plant as built ({', '.join(changes)}): the "
                                   "mass randomisation could not restore the nominal model")
        self._scratch = mujoco.MjData(m)  # mj_setConst's workspace
        self.nj = len(p.joints)
        self.dt = float(m.opt.timestep)
        self.dt_env = SUBSTEPS * self.dt
        self.H = float(p.meta["H_m"])
        self.weight = float(m.body_subtreemass[p.trunk]) * r5core.G
        self.push_bound = float(env.PUSH_BOUND_M_S[cfg.body])  # (a KeyError stops the build: no default bound)
        # nominal model values the randomisation scales (restored at every reset; the masses from nominal_model)
        self._mu0 = m.geom_friction.copy()
        self._gain0, self._bias0 = m.actuator_gainprm[:, 0].copy(), m.actuator_biasprm[:, 1].copy()
        self._damp0, self._fric0 = m.dof_damping.copy(), m.dof_frictionloss.copy()
        self.lib = r5lib.Library(p, cfg.body, cfg.set, cfg.rate, **({} if root is None else dict(root=root)),
                                 require_frozen=require_frozen)
        table = self.lib.tasks
        held = ", ".join(t for t in table if table[t]["available"])
        self.tasks = [str(t) for t in cfg.tasks] if cfg.tasks else [t for t in table if table[t]["available"]]
        unknown = [t for t in self.tasks if t not in table]  # (no row of the task table: the settings' own error)
        if unknown or not self.tasks:
            raise ValueError(f"{cfg.set}/{cfg.body}: tasks {unknown or '(none)'} are not available ({held})")
        #: Rule K4: the freeze the settings name (the one the run started on) and the one the files were read
        #: under (the set's latest), each ``{index, manifest_sha256}``; None without a freeze.
        self.freeze_started, self.freeze_now = self._check_freeze()
        gone = [t for t in self.tasks if not table[t]["available"]]
        if gone:  # (a named task the set does not hold: an input that is not the one expected, rule T2's reason)
            raise tune.InputError(f"{cfg.set}/{cfg.body}: the settings name tasks that are not available: "
                                  + "; ".join(f"{t} ({table[t]['reason']})" for t in gone) + f".  Available: {held}")
        self.scripts = {t: self.lib.script(t) for t in self.tasks}
        #: Every task's tracker, built here, while the model is nominal (design D16).
        self.trackers = {t: self._tracker(self.scripts[t]) for t in self.tasks}
        self._plans: dict = {}
        self.set_task_weights(cfg.task_weights if cfg.task_weights is not None else
                              [table[t]["weight"] for t in self.tasks])
        self.lg = None
        self.fixed_pairs = []
        if cfg.interference != "off":
            self.lg = lg = legs.LegGap(cfg.body)
            for k in np.flatnonzero(~lg.moving):
                if lg.home[k] < 0.0:
                    raise plant.PlantError(f"{cfg.body}: the leg pair {lg.pair(int(k))} interferes at the home pose "
                                           f"({1e3 * float(lg.home[k]):.2f} mm)")
                self.fixed_pairs.append(dict(pair=lg.pair(int(k)), home_gap_mm=1e3 * float(lg.home[k])))
        self.core = r5core.Core(p, dict(cfg.describe(), **self._core_extra()), self.lg)
        self.layout = self.core.layout
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.layout["obs_dim"],), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.layout["action_dim"],), np.float32)
        self._step_hook = None  # called after every physics step (the gates' digests)
        self._stream, self._stream_k = None, 0  # the last seeded reset's seed, episodes since (env.EPISODE_SEED_RULE)
        self.episode_seed = None
        self._delay = ExitStack()
        self._servo = ExitStack()
        #: The episode's servo parameters and the layer in (None: the option off, or between episodes).
        self.servo_params: servo.ServoParams | None = None
        self.servo_layer: servo.Layer | None = None
        self.tk = self.trackers[self.tasks[0]]
        self.script = self.scripts[self.tasks[0]]
        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ the set's freeze (rule K4)
    def _used_files(self) -> list[str]:
        """The files this env's tasks use, with the nominal walk's (the home pose of the track's LIPM frequency),
        sorted: what rule K4 compares.  A named task the set does not hold has none."""
        used = [f"refs/{setio.NOMINAL}.npz"]
        for t in self.tasks:
            if self.lib.tasks[t]["available"]:
                for f in r5lib.plan_files(self.lib.tasks[t]["plan"]):
                    if f not in used:
                        used.append(f)
        return sorted(used)

    def _check_freeze(self) -> tuple[dict | None, dict | None]:
        """``FREEZE_RULE``: (the freeze the settings name, the freeze the files were read under), or
        ``tune.InputError`` naming what differs.  Without ``set_freeze`` in the settings both are the one read and
        nothing is compared."""
        cfg, lib = self.cfg, self.lib
        now = None if lib.inputs["freeze"] is None else dict(lib.inputs["freeze"])
        if cfg.set_freeze is None:
            return now, now
        where = f"{cfg.set}/{cfg.body}"
        freezes = list(lib.set_doc.get("freezes") or [])
        if now is None:
            raise tune.InputError(f"{where}: the settings name the freeze {cfg.set_freeze[:12]}... and "
                                  + ("the set's latest freeze does not hold every file read" if freezes else
                                     "the set has no freeze"))
        # the named freeze: the first of the set's freezes that holds the named manifest (the settings hold the
        # hash alone; ``sets.freeze`` appends no freeze whose manifest an earlier one holds)
        old = next((f for f in freezes if f.get("manifest_sha256") == cfg.set_freeze), None)
        if old is None:
            raise tune.InputError(f"{where}: the settings name the freeze {cfg.set_freeze[:12]}..., which is none of "
                                  f"the set's {len(freezes)} (the latest: index {now['index']}, "
                                  f"{now['manifest_sha256'][:12]}...)")
        i0, held = old["index"], dict(old.get("manifest") or {})
        diff = []
        if s2io.sha256_json(old.get("manifest")) != cfg.set_freeze:
            diff.append(f"the manifest that set.json holds for freeze {i0} is not the one its hash names")

        def name_of(f: str) -> str:
            return f.split("/", 1)[1][: -len(".npz")]

        # (1) every used file's and entry's hash: the files of this env's tasks and the nominal walk's
        read = lib.inputs["files"]
        for f in self._used_files():
            row = held.get(name_of(f))
            if row is None:
                diff.append(f"{f} is not in freeze {i0}")
            elif row.get("sha256") != read[f]["sha256"]:
                diff.append(f"{f} has sha256 {read[f]['sha256'][:12]}..., freeze {i0} held "
                            f"{str(row.get('sha256'))[:12]}...")
            elif row.get("entry_sha256") != read[f]["entry_sha256"]:
                diff.append(f"{f}: its entry's hashed subset is {read[f]['entry_sha256'][:12]}..., freeze {i0} held "
                            f"{str(row.get('entry_sha256'))[:12]}...")
        # (2) the resolved task table: under freeze i0 a task was available exactly when the manifest held every
        # file of its plan (a manifest lists every available entry, and a frozen set is read through it).  Also
        # when freeze i0 is the latest: the records can be edited after it without a new freeze.
        for td in lib.table:
            if cfg.tasks and td.name not in self.tasks:
                continue  # (settings that name their tasks: the table is those tasks')
            missing = [f for f in r5lib.plan_files(td.plan) if name_of(f) not in held]
            was, is_ = not missing, bool(lib.tasks[td.name]["available"])
            if is_ and not was:
                diff.append(f"task {td.name} is in the table now and was absent under freeze {i0} "
                            f"({', '.join(missing)} not in it)")
            elif was and not is_:
                diff.append(f"task {td.name} was in the table under freeze {i0} and is absent now "
                            f"({lib.tasks[td.name]['reason']})")
        if diff and i0 == now["index"]:
            raise tune.InputError(f"{where}: the settings name the set's latest freeze, index {i0} "
                                  f"({cfg.set_freeze[:12]}...), and what the run uses, as the set's records are "
                                  f"read today, is not what that freeze holds (rule K4): {'; '.join(diff)}")
        if diff:
            raise tune.InputError(f"{where}: the set's latest freeze is index {now['index']} "
                                  f"({now['manifest_sha256'][:12]}...), the settings name index {i0} "
                                  f"({cfg.set_freeze[:12]}...), and what the run uses is not the same under both "
                                  f"(rule K4): {'; '.join(diff)}")
        return dict(index=i0, manifest_sha256=cfg.set_freeze), now

    # ------------------------------------------------------------------ hooks R6 replaces
    def _tracker(self, script: r5lib.Script) -> control.Tracker:
        return R5Tracker(self.p, script.track, "R2", gains=None, terms=(), ankle_cap=(), rate=self.cfg.rate,
                         model=self.nominal_model)

    def _core_extra(self) -> dict:
        return {}

    def _stance(self, tk, seeds: dict) -> None:
        tk.noise = estimator.SensorNoise(scale=self.factors["noise_scale"])
        tk.residual, tk.rate_factor = None, 1.0
        tk.reset("stance", joint_noise=self.cfg.start_joint_noise, seed=seeds["start"],
                 root_velocity_noise=self.cfg.start_root_velocity_noise, noise_seed=seeds["noise"])

    def _mid(self, tk, t_ref: float, qpos, qvel, seeds: dict) -> None:
        tk.noise = estimator.SensorNoise(scale=self.factors["noise_scale"])
        tk.residual, tk.rate_factor = None, 1.0
        tk.reset_at(t_ref, qpos, qvel, seeds["noise"])

    def _zero_ctrl(self, tk, t_ref: float) -> np.ndarray:
        """The servo targets the zero action gives at reference time ``t_ref`` (undelayed; ``d.ctrl`` is used as
        the workspace)."""
        p = self.p
        type(p).command(p, *tk.zero_command(t_ref))
        return p.d.ctrl.copy()

    def _after_begin(self, start: str, t_ref: float) -> None:
        pass

    def _apply(self, a_f: np.ndarray) -> None:
        self.tk.residual, self.tk.rate_factor = self.core.offsets(a_f)

    # ------------------------------------------------------------------ the model under randomisation
    def _draw_factors(self, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        f = dict(NOMINAL_FACTORS)
        if cfg.randomize:
            f.update(friction=float(rng.uniform(*cfg.friction)), mass=float(rng.uniform(*cfg.mass)),
                     kp=float(rng.uniform(*cfg.kp)),
                     latency_steps=int(rng.integers(cfg.latency_steps[0], cfg.latency_steps[1] + 1)),
                     noise_scale=float(rng.uniform(*cfg.noise_scale)),
                     joint_damping=float(rng.uniform(*cfg.joint_damping)),
                     joint_friction=float(rng.uniform(*cfg.joint_friction)))
        return f

    def _apply_factors(self, f: dict) -> None:
        """The model back to nominal, then the episode's factors (``FACTOR_RULE``; friction None: the model's own):
        Stage 1's scaling (``env.Stage1Env._apply_factors``), then the hinges' damping and friction-loss, which
        ``mj_setConst`` does not read."""
        p, m = self.p, self.p.m
        m.geom_friction[:] = self._mu0
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = self._gain0, self._bias0
        m.dof_damping[:], m.dof_frictionloss[:] = self._damp0, self._fric0
        env.set_trunk_mass(m, p.trunk, float(f.get("mass", 1.0)), self.nominal_model, self._scratch)
        if f.get("friction") is not None:
            control.set_friction(p, f["friction"])
        if f.get("kp", 1.0) != 1.0:
            m.actuator_gainprm[:, 0] = self._gain0 * f["kp"]
            m.actuator_biasprm[:, 1] = self._bias0 * f["kp"]
        if f.get("joint_damping", 1.0) != 1.0:
            m.dof_damping[p.dofs] = self._damp0[p.dofs] * f["joint_damping"]
        if f.get("joint_friction", 1.0) != 1.0:
            m.dof_frictionloss[p.dofs] = self._fric0[p.dofs] * f["joint_friction"]
        self.weight_true = float(m.body_subtreemass[p.trunk]) * r5core.G

    def _draw_push(self, rng: np.random.Generator, script: r5lib.Script, t_start: float, seconds: float):
        cfg = self.cfg
        if not (cfg.push_prob > 0 and rng.random() < cfg.push_prob and seconds > 4.0):
            return None
        t0 = float(rng.uniform(2.0, seconds - 2.0))
        walking = script.walking(t_start + t0 * cfg.rate)
        dv = float(rng.uniform(*(cfg.push_walking if walking else cfg.push_standing))) * self.push_bound
        ang = float(rng.uniform(-math.pi, math.pi))
        F = float(self.p.m.body_subtreemass[self.p.trunk]) * dv / cfg.push_s  # the true mass: the push's dv
        return (t0, cfg.push_s, F * math.cos(ang), F * math.sin(ang))

    def _plan(self, plan) -> tuple:
        """The script and tracker of a plan given by options (a bar condition's): built on first use, after the
        nominal arrays were restored, and kept by the plan."""
        key = tuple(tuple(x) for x in plan)
        if key not in self._plans:
            script = self.lib.from_plan(list(key))
            self._plans[key] = (script, self._tracker(script))
        return self._plans[key]

    def _leave_delay(self) -> None:
        self._delay.close()
        self._delay = ExitStack()

    # ------------------------------------------------------------------ the servo option (SERVO_RULE)
    def _leave_servo(self) -> None:
        """The servo layer out (``servo.Layer.exit``: the plant back bit for bit); nothing while the option is
        off."""
        self.servo_params = self.servo_layer = None
        self._servo.close()
        self._servo = ExitStack()

    def _servo_spec(self, opts: dict):
        """The episode's servo setting before anything is drawn: None (the option off), ``SERVO_DRAW`` (drawn at step
        8 from the episode seed) or a ``servo.ServoParams``; ``SERVO_RULE``'s refusals."""
        given = opts.get("servo")
        if self.cfg.servo is None:
            if given is not None:
                raise ValueError("options['servo'] with the servo option off (R5Config.servo None): the config turns "
                                 "it on (SERVO_RULE)")
            return None
        spec = self.cfg.servo if given is None else given
        if isinstance(spec, str) and spec == SERVO_DRAW:
            if opts.get("seeds") is not None and opts.get("episode_seed") is None:
                raise ValueError("a servo draw needs the episode seed, and with options['seeds'] and no episode_seed "
                                 "the env has none: give a named setting (design section 4; SERVO_RULE)")
            return spec
        return servo_params(spec)

    def _servo_factors(self, spec, given: dict) -> None:
        """Step 6 with the option on: the factors the layer owns are 1, and a named setting's trunk mass factor is the
        episode's (``SERVO_RULE``); ``given`` is options.factors."""
        f = self.factors
        for k in ("joint_damping", "joint_friction"):
            if f[k] != 1.0:
                raise ValueError(f"{k} {f[k]!r}: with the servo option on the layer owns the drives' damping and "
                                 "friction, and this factor is 1 (design section 4; SERVO_RULE)")
        mass = 1.0 if isinstance(spec, str) else float(spec.mass)  # (a training draw carries no mass)
        if mass != 1.0:
            if "mass" in given and float(given["mass"]) != mass:
                raise ValueError(f"options['factors'] mass {given['mass']!r} and the servo setting {spec.name}'s "
                                 f"trunk mass x {mass:g}: the setting's is the episode's (SERVO_RULE)")
            f["mass"] = mass

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """``RESET_ORDER``.  ``options``: ``task`` | ``plan`` (+ ``seconds``), ``start`` ("stance", "mid" or a
        reference time), ``factors``, ``push``, ``head`` (a ``r5core.HeadScript`` or None), ``episode_seed``,
        ``seeds`` ({"start", "noise"}), ``terminate_on_interference``; with the servo option on, ``servo``
        (``SERVO_RULE``; the order is then ``SERVO_RESET_ORDER``)."""
        self._leave_servo()  # (0 of SERVO_RESET_ORDER: before the arrays are restored; nothing while it is off)
        super().reset(seed=seed)  # 1
        if seed is not None:
            self._stream, self._stream_k = int(seed), 0
        rng = self.np_random
        opts = options or {}
        cfg, p = self.cfg, self.p
        spec = self._servo_spec(opts)  # (None while the option is off)
        self._apply_factors(NOMINAL_FACTORS)  # 2
        if opts.get("plan") is not None:  # 3
            script, tk = self._plan(opts["plan"])
            self.task = str(opts.get("task") or "plan")
        else:
            self.task = str(opts.get("task") or rng.choice(self.tasks, p=self.task_p))
            script, tk = self.scripts[self.task], self.trackers[self.task]
        if not (np.array_equal(tk.damp, self._damp0[p.dofs]) and np.array_equal(tk.fric, self._fric0[p.dofs])):
            raise RuntimeError(f"task {self.task!r}: the tracker's damping or friction-loss copy is not the nominal "
                               "model's (it was built on a randomised model)")
        self.script, self.tk = script, tk
        start = opts.get("start")  # 4
        t_start, q0, qd0 = 0.0, None, None
        if start is None:
            spans = script.mid_spans() if cfg.mid_start > 0 else []
            start = "mid" if spans and rng.random() < cfg.mid_start else "stance"
        if isinstance(start, str) and start == "mid":
            spans = script.mid_spans()
            if not spans:
                raise ValueError(f"task {self.task!r} has no mid-task start (rule T3)")
            t_start, q0, qd0 = script.start_state(mid_start_time(
                spans, float(rng.uniform(0.0, sum(b - a for a, b in spans)))))
        elif not isinstance(start, str):
            t_start, q0, qd0 = script.start_state(float(start))
            start = "mid"
        elif start != "stance":
            raise ValueError(f"start {start!r}: 'stance', 'mid' or a reference time")
        self.start, self.t_start = start, float(t_start)
        if opts.get("plan") is not None and opts.get("seconds") is not None:  # 5
            self.seconds = float(opts["seconds"])
            self.n_steps = int(math.ceil(int(round(self.seconds / self.dt)) / SUBSTEPS))
        else:
            self.seconds = min((script.t_stop - t_start) / cfg.rate, cfg.max_seconds)
            self.n_steps = int(round(self.seconds / self.dt_env))
        drawn = self._draw_factors(rng)  # 6
        self.factors = dict(drawn, **(opts.get("factors") or {}))
        if spec is not None:
            self._servo_factors(spec, dict(opts.get("factors") or {}))
        self._apply_factors(self.factors)
        self.push = opts["push"] if "push" in opts else self._draw_push(rng, script, t_start, self.seconds)  # 7
        self.head = None
        if self.layout["head_x"]:
            if "head" in opts:
                self.head = opts["head"]
            elif cfg.head_events:
                self.head = r5core.HeadScript.drawn(rng, self.seconds, cfg.gaze_prob, cfg.jaw_prob)
        if opts.get("seeds") is not None:  # 8
            ep = None if opts.get("episode_seed") is None else int(opts["episode_seed"])
            seeds = dict(start=int(opts["seeds"]["start"]), noise=int(opts["seeds"]["noise"]))
        else:
            if opts.get("episode_seed") is not None:
                ep = int(opts["episode_seed"])
            elif self._stream is not None:  # env.EPISODE_SEED_RULE
                ep = env.EPISODE_STRIDE * self._stream + self._stream_k
                self._stream_k += 1
            else:
                ep = int(rng.integers(2**31 - 1))
            seeds = dict(start=int(np.random.SeedSequence([ep, 0]).generate_state(1)[0]),
                         noise=int(np.random.SeedSequence([ep, 1]).generate_state(1)[0]))
        self.episode_seed, self.episode_seeds = ep, seeds
        if spec is not None:
            self.servo_params = servo_params(spec, ep)
        self.k_env = 0  # 9
        self._below, self.interference_step = 0, None
        self.terminate_on_interference = (cfg.interference == "terminate"
                                          and bool(opts.get("terminate_on_interference", True)))
        self._leave_delay()  # 10
        if start == "stance":  # 11
            self._stance(tk, seeds)
        else:
            r = np.random.default_rng(seeds["start"])
            qpos, qvel = q0.copy(), qd0 * cfg.rate
            if cfg.start_joint_noise:
                qpos[p.qadr] += r.uniform(-cfg.start_joint_noise, cfg.start_joint_noise, self.nj)
            if cfg.start_root_velocity_noise:
                qvel[:3] += r.uniform(-cfg.start_root_velocity_noise, cfg.start_root_velocity_noise, 3)
            self._mid(tk, t_start, qpos, qvel, seeds)
        k = int(self.factors["latency_steps"])  # 12
        queue = self._delay.enter_context(speedfit.servo_delay(p, k)) if k else None
        if self.servo_params is not None:  # after the delay and before a mid start's queue is filled (SERVO_RULE)
            self.servo_layer = self._servo.enter_context(servo.served(p, self.servo_params))
        if start == "mid":
            # The servos hold what the zero action commanded before the start, not their home targets
            # (``Plant.set_state``'s): the queue's oldest command, or without a delay the one of the physics step
            # before.  One forward pass on that, and the estimator's first reading (foot loads, accelerometer,
            # contact flags) is taken from it, as is the core's (``MID_START_RULE``).
            step = self.dt * cfg.rate
            if k:
                queue.clear()
                queue.extend(self._zero_ctrl(tk, max(t_start - j * step, 0.0)) for j in range(k, 0, -1))
                p.d.ctrl[:] = queue[0]
            else:
                p.d.ctrl[:] = self._zero_ctrl(tk, max(t_start - step, 0.0))
            mujoco.mj_forward(p.m, p.d)
            tk.est.reset(p.d.qpos.copy(), p.d.qvel.copy(), tk.sensors.read())
        self.core.begin(lambda: self.tk.tau_ref, lambda: r5core.tracker_estimate(self.tk), script, self.head)  # 13
        self._after_begin(start, t_start)
        obs = self._observation()
        info = dict(self._info(), seconds=self.seconds, n_steps=self.n_steps, seeds=dict(seeds),
                    head=None if self.head is None else self.head.describe())
        if self.servo_params is not None:
            info["servo"] = self.servo_params.name
        return obs, info

    def step(self, action):
        """``STEP_ORDER``."""
        core, tk, p = self.core, self.tk, self.p
        a_f = core.filter(action)  # 1
        self._apply(a_f)  # 2
        clock0 = tk.tau_ref
        push, hook = self.push, self._step_hook
        for _ in range(SUBSTEPS):  # 3
            if push is not None:
                control._apply_push(p, push)
            tk.step()
            if hook is not None:
                hook(p)
            core.substep()
        fall = p.fall_cause()  # 4
        if fall is None and p.d.warning.number.any():  # (any(d.number for d in p.d.warning), read at once)
            fall = "MuJoCo warning"
        pen, below = core.interference()
        self.k_env += 1  # 5
        r, terms = core.reward(clock0)
        self._below = self._below + 1 if below else 0
        if self._below >= INTERFERENCE_STEPS and self.interference_step is None:
            self.interference_step = self.k_env
        terminated = fall is not None  # 6
        if not terminated and self._below >= INTERFERENCE_STEPS and self.terminate_on_interference:
            terminated, fall = True, "leg interference"
        truncated = (not terminated) and self.k_env >= self.n_steps
        info = self._info()  # 7
        info.update(fall_cause=fall, reward_terms=terms,
                    interference=dict(penalty=pen, below_zero=bool(below), consecutive=self._below,
                                      min_gap_m=core._gap[2], first_env_step=self.interference_step))
        return self._observation(), float(r), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        return np.concatenate([self.core.actor_obs(), self.core.critic_obs(self.factors, self.push, self.cfg.floor)])

    def _info(self) -> dict:
        return dict(step=self.k_env, task=self.task, mode=self.mode, floor=self.cfg.floor, factors=self.factors,
                    push=self.push, episode_seed=self.episode_seed, time_s=float(self.p.d.time), start=self.start,
                    t_start=self.t_start, t_ref=float(self.tk.tau_ref), clock_factor=float(self.tk.rate_factor))

    def close(self) -> None:
        self._leave_servo()
        self._leave_delay()

    # ------------------------------------------------------------------ records
    def set_task_weights(self, w) -> None:
        """The tasks' relative weights: a mapping by name (tasks it does not name keep theirs) or one number per
        task in ``tasks``' order (rule M4's fix calls this through the vec env)."""
        if isinstance(w, dict):
            unknown = [t for t in w if t not in self.tasks]
            if unknown:
                raise ValueError(f"tasks {unknown} are not this env's ({', '.join(self.tasks)})")
            cur = dict(zip(self.tasks, getattr(self, "task_w", np.ones(len(self.tasks))), strict=True))
            w = [float(w.get(t, cur[t])) for t in self.tasks]
        w = np.asarray(w, float)
        if w.shape != (len(self.tasks),) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError(f"task weights {w.tolist()}: one non-negative number per task, not all zero")
        self.task_w = w
        self.task_p = w / w.sum()

    def describe(self) -> dict:
        """The env's settings as an episode executes them, without any path: the config, the resolved task table
        with the files' hashes and the absent tasks, the layout, the action mapping and the reward's weights,
        scales and rule texts.

        **What a resumed run compares** (rule K4; the lead's ruling of 16 Nov 2026).  A resumed run builds its
        envs again from the configs it stored, and the env itself refuses what rule K4 refuses (``FREEZE_RULE``).
        Of this description a trainer compares, with what the run's first session stored, ``RESUME_KEYS``:

        * ``config``: the settings as given.  It holds the named freeze (``set_freeze``: the manifest sha256 of
          the freeze the run started on) and, in a config that names its tasks, the task names; ``env_configs``
          makes such configs.  A config with ``tasks`` ``()`` takes whatever tasks the set holds on the day the
          env is built, so a later freeze that makes a further task usable is then refused (rule K4's "the
          resolved task table is unchanged"); with the names stored it is accepted and the run plays on with its
          own tasks.
        * ``task_names``: this env's tasks, in the order the task draw uses; ``tasks``: their resolved table
          (plans with their window counts, lengths, weights, headings), with the weights in force when the
          description is made (``set_task_weights`` changes them: compare the description of the env as built).
        * ``task_files``: the input files this env's tasks use and the nominal walk's, each with its sha256 and
          its entry's hashed subset (what ``FREEZE_RULE`` compares with the named freeze's manifest).
        * ``set_freeze_started``: the named freeze with its index (``FREEZE_RULE`` says which).  A config that
          names no freeze has none to keep: it and ``set_freeze_now`` are then both the latest.

        **An accepted later freeze of the set changes** ``LATER_FREEZE_KEYS`` and no other key: ``set_freeze``
        (the manifest sha256 of the set's latest freeze, not the settings' ``config.set_freeze``),
        ``set_freeze_now`` and ``inputs.freeze`` always; ``inputs.files`` (every file the library read: it reads
        the files of every task the set holds, and of every plan given by options since) and ``absent`` where the
        later freeze made further tasks usable.  A trainer that compared the description whole would refuse what
        rule K4 accepts.

        **The servo option** (``SERVO_RULE``).  Off, the description is today's.  On, ``config`` holds ``servo``,
        ``factor_rule``, ``reset_order`` and ``draw_order`` are the servo variants (``SERVO_FACTOR_RULE``,
        ``SERVO_RESET_ORDER``, ``SERVO_DRAW_ORDER``) and ``servo`` holds the option, the table's sha256, the rules
        and the kinds of the table's rows."""
        on = self.cfg.servo is not None
        settle = self.p.meta.get("settle") or {}
        lib = self.lib.describe()
        files = lib["inputs"]["files"]
        extra = {} if not on else dict(servo=dict(
            option=self.cfg.servo, body=servo.BODY, table_sha256=servo.TABLE_SHA256, rule=SERVO_RULE,
            mid_start_rule=SERVO_MID_START_RULE, draw_rule=servo.DRAW_RULE, layer_rule=servo.LAYER_RULE,
            hooks_rule=servo.HOOKS_RULE,
            kind_counts=servo.kind_counts(), named=list(servo.NAMED),
            simulation_only="a prospective design built in code: no value of the servo model is measured on this "
                            "build's hardware, and nothing the env runs with it is a result"))
        return dict(mode=self.mode, config=self.cfg.describe(), floor=self.cfg.floor,
                    floor_params=self.p.meta.get("floor_params"), settled=settle.get("status"),
                    task_names=list(self.tasks),
                    tasks={t: dict(lib["tasks"][t], weight=float(w))
                           for t, w in zip(self.tasks, self.task_w, strict=True)},
                    task_files={f: dict(files[f]) for f in self._used_files()},
                    absent=lib["absent"], inputs=lib["inputs"], set_freeze=(lib["inputs"]["freeze"] or {}).get(
                        "manifest_sha256"), set_freeze_started=self.freeze_started, set_freeze_now=self.freeze_now,
                    freeze_rule=FREEZE_RULE, omega=lib["omega"], crouch_frac=lib["crouch_frac"],
                    crouch_rule=lib["crouch_rule"],
                    command_rule=lib["command_rule"], core=self.core.describe(), layout=self.layout,
                    reward=dict(self.core.weights), reward_scales=self.core.describe()["reward_scales"],
                    reward_rule=r5core.REWARD_RULE, head_rule=r5core.HEAD_RULE if self.layout["head_x"] else None,
                    interference=self.cfg.interference, interference_rule=r5core.INTERFERENCE_RULE,
                    fixed_pairs=self.fixed_pairs, push_bound_m_s=self.push_bound, push_rule=PUSH_RULE,
                    factor_rule=SERVO_FACTOR_RULE if on else FACTOR_RULE, mid_start_rule=MID_START_RULE,
                    reset_order=SERVO_RESET_ORDER if on else RESET_ORDER, step_order=STEP_ORDER,
                    draw_order=SERVO_DRAW_ORDER if on else DRAW_ORDER, mass_rule=env.MASS_RULE,
                    episode_seed_rule=env.EPISODE_SEED_RULE, nominal_weight_N=self.weight, substeps=SUBSTEPS, **extra)

    def process_cpu(self) -> tuple[int, float]:
        """This env's process and its CPU seconds so far (the trainer sums its env processes' CPU time)."""
        return os.getpid(), time.process_time()

    def episode_stream(self) -> dict:
        """The last seeded reset's seed and the episodes drawn since (``env.EPISODE_SEED_RULE``)."""
        return dict(reset_seed=self._stream, episodes=self._stream_k,
                    first_episode_seed=None if self._stream is None else env.EPISODE_STRIDE * self._stream)


class R6Env(R5Env):
    """Mode "R6": the policy sets the joint targets (module docstring)."""

    mode = "R6"

    def _tracker(self, script: r5lib.Script) -> control.Tracker:
        return r5core.R6Tracker(self.p, script.track, self.cfg.rate, model=self.nominal_model)

    def _core_extra(self) -> dict:
        #: Stage 1's span rule over the env's task scripts (``env.POLICY_ONLY_SPAN_RULE``).
        self.span = r5core.policy_span(self.p, [self.scripts[t].track for t in self.tasks])
        return dict(span=self.span)

    def _stance(self, tk, seeds: dict) -> None:
        tk.noise = estimator.SensorNoise(scale=self.factors["noise_scale"])
        tk.reset("stance", joint_noise=self.cfg.start_joint_noise, seed=seeds["start"],
                 root_velocity_noise=self.cfg.start_root_velocity_noise, noise_seed=seeds["noise"])

    def _mid(self, tk, t_ref: float, qpos, qvel, seeds: dict) -> None:
        tk.noise = estimator.SensorNoise(scale=self.factors["noise_scale"])
        tk.reset_at(t_ref, qpos, qvel, seeds["noise"])

    def _zero_ctrl(self, tk, t_ref: float) -> np.ndarray:
        """The servo targets of the reference's joint angles at ``t_ref`` (R6 commands angles alone)."""
        p = self.p
        type(p).command(p, np.clip(tk.track.joint(t_ref)[: self.nj], tk.lo, tk.hi))
        return p.d.ctrl.copy()

    def _after_begin(self, start: str, t_ref: float) -> None:
        if start == "mid":  # design D8: the history and the filter hold the actions that give the reference's targets
            acts = self.core.reference_actions(t_ref)
            self.core.set_history(acts, acts[0])

    def _apply(self, a_f: np.ndarray) -> None:
        self.tk.q_target, self.tk.rate_factor = self.core.targets(a_f)


def mid_start_time(spans, u: float) -> float:
    """The reference time of a mid-task start (``MID_START_RULE``): ``u``, drawn uniform on [0, the spans' total
    length], laid along ``spans`` (``Script.mid_spans``: (start, end) in order), and kept inside the span it falls
    in.  The subtractions along the spans can leave ``u`` a rounding step past the last span's length, and a span
    may end at a hold's first instant, where no start can land: the result is always before its span's end."""
    spans = [(float(a), float(b)) for a, b in spans]
    if not spans:
        raise ValueError("no span: this task has no mid-task start (rule T3)")
    u = float(u)
    for a, b in spans:
        if u <= b - a:
            break
        u -= b - a
    else:
        u = b - a  # (past the last span's length, by rounding: its end)
    t = a + min(max(u, 0.0), b - a)
    return float(np.nextafter(b, a)) if t >= b else t


def servo_params(spec, episode_seed: int | None = None) -> servo.ServoParams:
    """The servo parameters of a setting (``SERVO_RULE``): a ``servo.ServoParams`` as given; ``SERVO_DRAW`` the
    training draw of ``episode_seed`` (refused without one); ``"D/<s>"`` the drawn panel's set of bar seed s; any
    other name a named setting of ``servo.NAMED``."""
    if isinstance(spec, servo.ServoParams):
        return spec
    if not isinstance(spec, str):
        raise ValueError(f"servo setting {spec!r}: a name, \"{SERVO_DRAW}\", \"D/<bar seed>\" or a servo.ServoParams")
    if spec == SERVO_DRAW:
        return servo.training_draw(episode_seed)
    if spec.startswith("D/"):
        s = spec[2:]
        if not s.isdigit():
            raise ValueError(f"servo setting {spec!r}: the drawn panel is \"D/<bar seed>\" (a whole number)")
        return servo.panel_draw(int(s))
    if spec == "D":
        raise ValueError("servo setting 'D': the drawn panel draws from a bar seed, \"D/<seed>\"")
    return servo.setting(spec)


def _check_servo_config(cfg: R5Config) -> None:
    """``SERVO_RULE``'s refusals of a config with the option on, before anything is built."""
    if not isinstance(cfg.servo, str) or not cfg.servo:
        raise ValueError(f"servo {cfg.servo!r}: None (off), \"{SERVO_DRAW}\", a named setting or \"D/<bar seed>\"")
    if cfg.body != servo.BODY:
        raise ValueError(f"servo {cfg.servo!r} on {cfg.body}: the servo table is for {servo.BODY} only (design, status "
                         "line; its own table first, choice 14)")
    if cfg.servo != SERVO_DRAW:
        servo_params(cfg.servo)
    if cfg.randomize and (tuple(cfg.joint_damping) != (1.0, 1.0) or tuple(cfg.joint_friction) != (1.0, 1.0)):
        raise ValueError(f"servo {cfg.servo!r} with joint_damping {cfg.joint_damping} and joint_friction "
                         f"{cfg.joint_friction}: the layer owns the drives' damping and friction, so both ranges are "
                         "(1, 1) (the draws stay and the stream does not shift; design section 4)")


def make(mode: str, cfg: R5Config, seed: int | None = None) -> R5Env:
    """The factory the worker pool names (``"designlab.stage2.r5env:make"``).  ``cfg`` may be a config's
    ``describe()`` (a plain dict crosses a process boundary as JSON)."""
    if isinstance(cfg, dict):
        cfg = config_from(cfg)
    if mode != cfg.mode:
        cfg = replace(cfg, mode=mode)
    if mode == "R5":
        return R5Env(cfg, seed)
    if mode == "R6":
        return R6Env(cfg, seed)
    raise ValueError(f"mode {mode!r}: R5 or R6")


def config_from(doc: dict) -> R5Config:
    """A config from its ``describe()`` (lists back to tuples; no ``servo`` key: the option off)."""
    def tup(x):
        return tuple(tup(v) for v in x) if isinstance(x, (list, tuple)) else x

    return R5Config(**{k: tup(v) for k, v in doc.items()})


def set_freeze(body: str, set_name: str, root=setio.ROOT) -> str | None:
    """The manifest sha256 of a set's latest freeze (None: the S1 view, or a set that was never frozen)."""
    if set_name == r5lib.STOCK_SET:
        return None
    fz = setio.frozen(setio.load(setio.set_dir(set_name, body, root))["set.json"])
    return None if fz is None else fz["manifest_sha256"]


def env_configs(mode: str, body: str, set_name: str, n_envs: int = 8, overrides: dict | None = None,
                root=setio.ROOT) -> list[R5Config]:
    """A job's env configs: env i on ``FLOOR_SPREAD[i % 4]`` (design D13), the set pinned by its freeze today and
    the tasks named: ``tasks`` holds the names of the tasks the set holds today, in the table's order
    (``r5lib.task_names``: the tasks an env resolves from a config that names none).  A run stores these configs
    and builds its envs from them in every session, so a later freeze of the set that makes a further task usable
    is accepted (rule K4, ``FREEZE_RULE``) and the run plays on with the tasks it started with.  ``overrides``
    replaces fields in every config (a field named ``floor`` puts every env on that floor; ``tasks`` names other
    tasks).  ``root`` is where the sets are read from (a test's; it is no part of the settings)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: choose from {', '.join(MODES)}")
    base = R5Config(body=body, set=set_name, mode=mode, tasks=tuple(r5lib.task_names(body, set_name, root)),
                    set_freeze=set_freeze(body, set_name, root))
    over = dict(overrides or {})
    return [replace(base, **{"floor": FLOOR_SPREAD[i % len(FLOOR_SPREAD)], **over}) for i in range(int(n_envs))]


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def throughput(mode: str, cfg: R5Config, env_steps: int = 2000, sigma: float = 0.1, seed: int = 0) -> dict:
    """Env steps per wall second of one env in this process: actions drawn normal with standard deviation
    ``sigma`` (0: the zero action), each episode reset as it ends (resets included), with the build time, the CPU
    seconds, the resident memory and the machine's load beside it."""
    t0 = time.perf_counter()
    e = make(mode, cfg)
    build_s = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    n = e.layout["action_dim"]
    e.reset(seed=seed)
    t0, c0 = time.perf_counter(), time.process_time()
    resets = falls = 0
    for _ in range(int(env_steps)):
        a = (sigma * rng.standard_normal(n)).astype(np.float32) if sigma else np.zeros(n, np.float32)
        _, _, term, trunc, _ = e.step(a)
        if term or trunc:
            falls += bool(term)
            e.reset()
            resets += 1
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    out = dict(mode=mode, body=cfg.body, set=cfg.set, floor=cfg.floor, env_steps=int(env_steps), sigma=float(sigma),
               resets=resets, terminations=falls, env_processes=1, build_s=build_s, wall_s=wall, cpu_s=cpu,
               env_steps_per_wall_s=env_steps / wall, env_steps_per_cpu_s=env_steps / cpu, rss_mb=_rss_mb(),
               load_average=list(os.getloadavg()),
               rule="actions normal(0, sigma) per entry (clipped by the env), each episode reset as it ends (resets "
                    "included), one env in one process: env steps per wall second, with the CPU seconds of the "
                    "process, its resident memory after the run and the machine's load average beside it")
    e.close()
    return out


__all__ = ["DRAW_ORDER", "FLOOR_SPREAD", "FREEZE_RULE", "LATER_FREEZE_KEYS", "MODES", "R5Config", "R5Env", "R5Tracker",
           "R6Env", "RESET_ORDER", "RESUME_KEYS", "SERVO_DRAW", "SERVO_DRAW_ORDER", "SERVO_FACTOR_RULE",
           "SERVO_RESET_ORDER", "SERVO_RULE", "STEP_ORDER", "config_from", "env_configs", "make", "mid_start_time",
           "servo_params", "set_freeze", "throughput"]
