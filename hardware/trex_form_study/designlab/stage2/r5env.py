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
over the k physics steps before the start.

R6 (``R6Env``; design D8, rule E2) sets joint targets alone through ``r5core.R6Tracker``; the same observation,
commands, reward, starts and randomisation.
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
from . import r5core, r5lib, setio

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
    "queue); 13 the core's baselines, histories and accumulators, the observation")
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
    "that state, and the delay queue holds the k commands the zero action would have given before it (the servos "
    "are set to the oldest and the forward pass is computed once)")
PUSH_RULE = (
    "with probability push_prob in an episode longer than 4 s: one push_s pulse at the trunk's CoM at a time uniform "
    "in [2 s, end - 2 s], direction uniform, dv = u x the body's Stage 0 forward bound (env.PUSH_BOUND_M_S[body]: "
    "no default), u uniform on push_walking if the reference walks at that time at the commanded clock and on "
    "push_standing if it stands; force = true mass x dv / push_s (rule J3)")
FACTOR_RULE = (
    "per episode (rules J1, J2): servo delay k uniform on the integers latency_steps (inclusive), through "
    "speedfit.servo_delay; servo stiffness x kp; sliding friction; trunk mass and inertia x mass (env.MASS_RULE); "
    "joint damping and joint friction-loss, one factor each for all hinges; IMU noise scale.  The controller's "
    "model and feedforward stay nominal: every tracker is built on the nominal model before any episode")


@dataclass(frozen=True)
class R5Config:
    """One env's settings (design 3.2).  Ranges are uniform (lo, hi); ``randomize`` False draws no factor (the
    nominal plant, no delay, the default IMU noise).  ``tasks`` ``()`` takes every task the set has.  No field is a
    path: the set is named and pinned by ``set_freeze`` (its freeze's manifest sha256), and ``setio.set_dir``
    resolves it under the study directory."""

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

    def describe(self) -> dict:
        return {k: _plain(v) for k, v in asdict(self).items()}


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
        freeze = (self.lib.inputs["freeze"] or {}).get("manifest_sha256")
        if cfg.set_freeze is not None and cfg.set_freeze != freeze:
            raise tune.InputError(f"{cfg.set}/{cfg.body}: the set's freeze is {str(freeze)[:12]}..., the settings "
                                  f"name {cfg.set_freeze[:12]}...")
        table = self.lib.tasks
        self.tasks = [str(t) for t in cfg.tasks] if cfg.tasks else [t for t in table if table[t]["available"]]
        bad = [t for t in self.tasks if t not in table or not table[t]["available"]]
        if bad or not self.tasks:
            raise ValueError(f"{cfg.set}/{cfg.body}: tasks {bad or '(none)'} are not available "
                             f"({', '.join(t for t in table if table[t]['available'])})")
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
        self.tk = self.trackers[self.tasks[0]]
        self.script = self.scripts[self.tasks[0]]
        if seed is not None:
            self.reset(seed=seed)

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

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """``RESET_ORDER``.  ``options``: ``task`` | ``plan`` (+ ``seconds``), ``start`` ("stance", "mid" or a
        reference time), ``factors``, ``push``, ``head`` (a ``r5core.HeadScript`` or None), ``episode_seed``,
        ``seeds`` ({"start", "noise"}), ``terminate_on_interference``."""
        super().reset(seed=seed)  # 1
        if seed is not None:
            self._stream, self._stream_k = int(seed), 0
        rng = self.np_random
        opts = options or {}
        cfg, p = self.cfg, self.p
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
            u = float(rng.uniform(0.0, sum(b - a for a, b in spans)))
            for a, b in spans:
                if u <= b - a:
                    break
                u -= b - a
            t_start, q0, qd0 = script.start_state(min(a + u, b))
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
        if k:
            queue = self._delay.enter_context(speedfit.servo_delay(p, k))
            if start == "mid":
                step = self.dt * cfg.rate
                queue.clear()
                queue.extend(self._zero_ctrl(tk, max(t_start - j * step, 0.0)) for j in range(k, 0, -1))
                p.d.ctrl[:] = queue[0]
                mujoco.mj_forward(p.m, p.d)
        self.core.begin(lambda: self.tk.tau_ref, lambda: r5core.tracker_estimate(self.tk), script, self.head)  # 13
        self._after_begin(start, t_start)
        return self._observation(), dict(self._info(), seconds=self.seconds, n_steps=self.n_steps, seeds=dict(seeds),
                                         head=None if self.head is None else self.head.describe())

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
        scales and rule texts."""
        settle = self.p.meta.get("settle") or {}
        lib = self.lib.describe()
        return dict(mode=self.mode, config=self.cfg.describe(), floor=self.cfg.floor,
                    floor_params=self.p.meta.get("floor_params"), settled=settle.get("status"),
                    tasks={t: dict(lib["tasks"][t], weight=float(w))
                           for t, w in zip(self.tasks, self.task_w, strict=True)},
                    absent=lib["absent"], inputs=lib["inputs"], set_freeze=(lib["inputs"]["freeze"] or {}).get(
                        "manifest_sha256"), omega=lib["omega"], crouch_frac=lib["crouch_frac"],
                    command_rule=lib["command_rule"], core=self.core.describe(), layout=self.layout,
                    reward=dict(self.core.weights), reward_scales=self.core.describe()["reward_scales"],
                    reward_rule=r5core.REWARD_RULE, head_rule=r5core.HEAD_RULE if self.layout["head_x"] else None,
                    interference=self.cfg.interference, interference_rule=r5core.INTERFERENCE_RULE,
                    fixed_pairs=self.fixed_pairs, push_bound_m_s=self.push_bound, push_rule=PUSH_RULE,
                    factor_rule=FACTOR_RULE, mid_start_rule=MID_START_RULE, reset_order=RESET_ORDER,
                    step_order=STEP_ORDER, draw_order=DRAW_ORDER, mass_rule=env.MASS_RULE,
                    episode_seed_rule=env.EPISODE_SEED_RULE, nominal_weight_N=self.weight, substeps=SUBSTEPS)

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
    """A config from its ``describe()`` (lists back to tuples)."""
    def tup(x):
        return tuple(tup(v) for v in x) if isinstance(x, (list, tuple)) else x

    return R5Config(**{k: tup(v) for k, v in doc.items()})


def set_freeze(body: str, set_name: str, root=setio.ROOT) -> str | None:
    """The manifest sha256 of a set's latest freeze (None: the S1 view, or a set that was never frozen)."""
    if set_name == r5lib.STOCK_SET:
        return None
    fz = setio.frozen(setio.load(setio.set_dir(set_name, body, root))["set.json"])
    return None if fz is None else fz["manifest_sha256"]


def env_configs(mode: str, body: str, set_name: str, n_envs: int = 8, overrides: dict | None = None) -> list[R5Config]:
    """A job's env configs: env i on ``FLOOR_SPREAD[i % 4]`` (design D13), the set pinned by its freeze today.
    ``overrides`` replaces fields in every config (a field named ``floor`` puts every env on that floor)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: choose from {', '.join(MODES)}")
    base = R5Config(body=body, set=set_name, mode=mode, set_freeze=set_freeze(body, set_name))
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


__all__ = ["DRAW_ORDER", "FLOOR_SPREAD", "MODES", "R5Config", "R5Env", "R5Tracker", "R6Env", "RESET_ORDER",
           "STEP_ORDER", "config_from", "env_configs", "make", "set_freeze", "throughput"]
