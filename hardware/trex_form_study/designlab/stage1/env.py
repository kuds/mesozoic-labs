"""Stage 1 PPO environments (``STAGE1_PLAN.md`` section 2, the env.py / policy.py / train.py row; section 3 step 7).

A standalone Gymnasium env, because the repository's ``BaseDinoEnv`` holds one ctrl per step while the tracker sets the
servo targets every 2 ms physics step.

* **R3** (``ResidualEnv``): the tracker R2 (``control.Tracker`` on the estimate, with the tuned gains: ``tune.json``'s
  selection for the run's seed, else the tuning start, flagged in ``gains_info``) plus a learned residual at 50 Hz
  (``ENV_SUBSTEPS`` physics steps per env step): 14 joint offsets added to every servo target (``RESIDUAL_LEG_RAD``
  on the legs, ``RESIDUAL_NECK_TAIL_RAD`` on neck and tail) and a factor ``1 + RESIDUAL_RATE`` x a on the reference
  clock's rate, both through a first-order low-pass of ``FILTER_HZ`` (10 Hz).  A zero residual is R2 exactly: the
  tracker then adds no offset and does not touch its rate (``Tracker.residual`` None, ``rate_factor`` 1), so every
  physics step's state is R2's (gate G6a, ``zero_residual_check``).
* **R4** (``PolicyOnlyEnv``): no reference controller.  The policy's 14 values set the joint targets directly, the
  home pose plus the action times each joint's span (``POLICY_ONLY_SPAN_RULE``) within the plant's usable ranges, held
  by the servos' PD alone; its 15th sets the rate of the clock its phase input runs on.  The same actor inputs, rates,
  filter, reward, tasks and randomisation as R3.

Observations (``obs_layout``): the actor's deployable inputs first (``ACTOR_OBS``): the encoders' angles and rates,
the IMU's gravity direction in the trunk frame (the estimator's attitude) and its gyro, the two foot loads (over the
weight), the phase (the reference clock's sin and cos and the reference's support flags), the command (forward speed
and yaw rate, the reference's) and the previous action; then the privileged inputs only the critic reads
(``CRITIC_OBS``): the true root velocity in the heading frame, the root height, roll and pitch, the true DCM error and
the joint tracking errors against the reference, the heading error, the friction, the floor, the push and the
randomised plant factors (``policy.PrivilegedCriticPolicy`` slices the actor's part off).

Reward (``REWARD_RULE``): 0.55 imitation (the joints against the reference) plus 0.35
``direction_commands.gaussian_tracking_reward`` (forward speed and yaw rate against the command), minus the duty,
slip, flight, impact, clearance and tilt penalties of plan section 4's walk limits (``PENALTIES``).  Episodes end on
``plant.Plant.fall_cause`` (or a MuJoCo warning) and are truncated at the task's end or ``EnvConfig.max_seconds``.

Randomisation per episode (``EnvConfig``; plan section 2): friction, the trunk's mass (and inertia), the servos'
stiffness (the controller keeps the nominal values: a model error), actuation latency (0-20 ms) and the IMU noise
scale; and pushes (``EnvConfig.push_prob``).  Tasks (``TASKS``): the tune mix's start-walk-stop, walking turns and
in-place turns, the brisk and Fr 0.05 walks through their speed changes, and a stand.
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import STUDY, control, estimator, metrics, plant
from .refs import Reference

ENV_SUBSTEPS = 10  # 2 ms physics steps per 50 Hz env step
FILTER_HZ = 10.0
RESIDUAL_LEG_RAD = 0.2
RESIDUAL_NECK_TAIL_RAD = 0.1
RESIDUAL_RATE = 0.25
#: R4: each joint's action span is this times the largest excursion from home over the env's reference tracks.
POLICY_ONLY_SPAN_FACTOR = 1.25
POLICY_ONLY_MIN_SPAN_RAD = 0.10
POLICY_ONLY_SPAN_RULE = (f"joint target = home + action x span, clipped to the plant's usable range; span = "
                         f"{POLICY_ONLY_SPAN_FACTOR:g} x the largest excursion of that joint from home over the env's "
                         f"reference tracks, at least {POLICY_ONLY_MIN_SPAN_RAD:g} rad")
COMMAND_SCALE = 0.5  # command inputs: m/s and rad/s over this
GRAVITY = 9.81

#: The episode tasks: name -> (track plan as tune.episode_plan writes them, the reference whose period is a stride).
TASKS = {
    "walk": ([("transition", "start"), ("ref", "nominal", 7), ("transition", "stop")], "nominal"),
    "walking_turn_left": ([("transition", "start"), ("transition", "enter_walk_0.35_left"),
                           ("ref", "walk_0.35_left", 5), ("transition", "exit_walk_0.35_left"),
                           ("transition", "stop")], "nominal"),
    "walking_turn_right": ([("transition", "start"), ("transition", "enter_walk_0.35_right"),
                            ("ref", "walk_0.35_right", 5), ("transition", "exit_walk_0.35_right"),
                            ("transition", "stop")], "nominal"),
    "in_place_turn_left": ([("transition", "start_in_place_0.50_left"), ("ref", "in_place_0.50_left", 3),
                            ("transition", "stop_in_place_0.50_left")], "nominal"),
    "in_place_turn_right": ([("transition", "start_in_place_0.50_right"), ("ref", "in_place_0.50_right", 3),
                             ("transition", "stop_in_place_0.50_right")], "nominal"),
    "brisk": ([("transition", "start"), ("ref", "nominal", 1), ("transition", "nominal_to_brisk"),
               ("ref", "brisk", 6), ("transition", "brisk_to_nominal"), ("ref", "nominal", 1),
               ("transition", "stop")], "nominal"),
    "fr05": ([("transition", "start"), ("ref", "nominal", 1), ("transition", "nominal_to_brisk"),
              ("ref", "brisk", 1), ("transition", "brisk_to_fr05"), ("ref", "fr05", 6),
              ("transition", "fr05_to_brisk"), ("ref", "brisk", 1), ("transition", "brisk_to_nominal"),
              ("ref", "nominal", 1), ("transition", "stop")], "nominal"),
    "stand": ([("hold", 12.0), ("transition", "start")], "nominal"),
}
STAND_SECONDS = 10.0  # the stand task's length (inside its hold)

#: Actor inputs (name, width): deployable only.
ACTOR_OBS = (("q", 14), ("qd", 14), ("gravity", 3), ("gyro", 3), ("foot_load", 2), ("phase", 2), ("support", 2),
             ("command", 2), ("prev_action", 15))
#: The critic's privileged inputs (name, width), after the actor's.
CRITIC_OBS = (("root_vel", 3), ("height", 1), ("roll_pitch", 2), ("dcm_error", 2), ("joint_error", 14),
              ("heading_error", 1), ("friction", 1), ("floor", 3), ("push", 3), ("plant_factors", 3))
FLOOR_CODES = {"hard": 0, "robot_floor_carpet": 1}  # any pile: 2

#: The reward's penalty terms (name -> weight); each term lies in [0, 1] (``REWARD_RULE``).
PENALTIES = dict(duty=0.02, slip=0.02, flight=0.05, impact=0.02, clearance=0.02, tilt=0.02)
IMITATION_W, TRACKING_W = 0.55, 0.35
IMITATION_SCALE_RAD = 0.1
REWARD_RULE = (
    f"{IMITATION_W:g} exp(-mean_j ((q_j - q_ref_j) / {IMITATION_SCALE_RAD:g} rad)^2) (true joints, the reference at the "
    f"env step's end) + {TRACKING_W:g} gaussian_tracking_reward(|v - v_cmd|, |w - w_cmd|) (the CoM's forward speed "
    "and the trunk's yaw rate over the env step against the reference's, at the commanded clock rate) - the penalties, "
    "each in [0, 1]: duty (each foot's contact share over the last stride outside 0.55-0.65, over 0.05, while the "
    "reference walks), slip (a foot carrying 20 % of the weight sliding faster than 0.05 m/s at a loaded contact, the "
    "excess over 0.05 m/s), flight (the share of the env step's physics steps with neither foot carrying 2 % of the "
    "weight), impact (the peak floor load over 1.6 BW, the excess over 1.6 BW), clearance (the reference's swing foot "
    "in mid-swing below 0.04 H, the shortfall over 0.04 H), tilt (roll or pitch about home beyond 4 deg, the excess "
    "over 4 deg); weights " + ", ".join(f"{k} {v:g}" for k, v in PENALTIES.items()))


def _gaussian_tracking_reward():
    """``environments.shared.direction_commands.gaussian_tracking_reward`` (plan section 2), from the repository."""
    try:
        from environments.shared.direction_commands import gaussian_tracking_reward
    except ImportError:
        root = str(STUDY.parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from environments.shared.direction_commands import gaussian_tracking_reward
    return gaussian_tracking_reward


def tuned_gains(run_dir: Path, seed: int) -> tuple[np.ndarray, dict]:
    """The tracker's gains for a PPO run: ``<run_dir>/tune.json``'s selection (G5) for ``seed``, else for the other
    seed, else the tuning start (decision 16), flagged ``untuned``."""
    path = Path(run_dir) / "tune.json"
    if path.exists():
        doc = json.loads(path.read_text())
        seeds = doc.get("seeds") or {}
        for s in [str(seed)] + [k for k in seeds if k != str(seed)]:
            v = seeds.get(s) or {}
            if "selected_gains" in v:
                g = np.array([float(v["selected_gains"][n]) for n in control.GAIN_NAMES])
                flag = None if v.get("pass_") else ("G5 not judged" if v.get("pass_") is None else "G5 failed")
                if s != str(seed):
                    flag = f"{flag + '; ' if flag else ''}the selection of CMA-ES seed {s}, not {seed}"
                return g, dict(source=f"tune.json, CMA-ES seed {s}: {v.get('selected')}", flag=flag,
                               file=str(path), sha256=plant.sha256(path))
    return control.START_GAINS.copy(), dict(source="the tuning start (decision 16): no tune.json selection",
                                            flag="untuned", file=str(path) if path.exists() else None)


@dataclass
class EnvConfig:
    """One env's settings (module docstring).  Ranges are uniform (lo, hi); ``randomize`` False draws none (the
    nominal plant, no latency, the default IMU noise) and ``push_prob`` 0 pushes never."""

    body: str
    floor: str = "hard"
    pile_grid: float | None = 0.008  # piles: plan section 2's training pile (carpet.pile_floor grid)
    run_dir: str | None = None  # references, transitions and tune.json (default tune.RUN / body)
    gains: str = "tuned"  # "tuned" (tune.json), "start" or "earlier_defaults"
    gains_seed: int = 42
    tasks: tuple = tuple(TASKS)
    task_weights: tuple | None = None
    rate: float = 1.05  # the commanded clock rate (tune.COMMAND_RATE: 5 % above the targets)
    max_seconds: float = 30.0
    randomize: bool = True
    friction: tuple = (0.54, 1.25)
    mass: tuple = (0.9, 1.1)  # trunk mass factor
    kp: tuple = (0.8, 1.2)  # servo stiffness factor
    latency_steps: tuple = (0, 10)  # physics steps of actuation delay (0-20 ms)
    noise_scale: tuple = (0.5, 1.5)  # estimator.SensorNoise.scale
    push_prob: float = 0.5
    push_frac: tuple = (0.0, 1.0)  # x the Stage 0 forward push bound (plan section 5's ledger)
    push_s: float = 0.1
    start_joint_noise: float = 0.01  # tune.START_JOINT_NOISE_RAD
    start_root_velocity_noise: float = 0.02  # tune.START_ROOT_VELOCITY_NOISE_M_S

    def describe(self) -> dict:
        return asdict(self)


#: The Stage 0 forward push bound per body (plan section 5's margin ledger), m/s.
PUSH_BOUND_M_S = {"60-N-T-10": 0.487, "60-N-H-10": 0.478, "75-M-H-10": 0.589}


def obs_layout() -> dict:
    """name -> (start, stop) of every observation block, and the actor's width."""
    out, i = {}, 0
    for name, w in ACTOR_OBS + CRITIC_OBS:
        out[name] = (i, i + w)
        i += w
    return dict(blocks=out, actor_dim=sum(w for _, w in ACTOR_OBS), critic_dim=i)


ACTOR_DIM = sum(w for _, w in ACTOR_OBS)
OBS_DIM = ACTOR_DIM + sum(w for _, w in CRITIC_OBS)
ACTION_DIM = 15


class Stage1Env(gym.Env):
    """The common part of R3 and R4 (module docstring); ``mode`` "R3" or "R4"."""

    metadata = {"render_modes": []}
    mode = "R3"

    def __init__(self, config: EnvConfig, seed: int | None = None):
        super().__init__()
        from . import tune

        self.cfg = cfg = config
        self.run_dir = Path(cfg.run_dir) if cfg.run_dir else tune.RUN / cfg.body
        self.p = p = plant.build(cfg.body, cfg.floor, pile_grid=cfg.pile_grid if cfg.floor in plant.FLOORS[2:] else None)
        m = p.m
        self.nj = len(p.joints)
        self.dt = float(m.opt.timestep)
        self.dt_env = ENV_SUBSTEPS * self.dt
        self.H = float(p.meta["H_m"])
        self.weight = float(m.body_subtreemass[p.trunk]) * GRAVITY
        self.omega = control.omega0(p, Reference.load(self.run_dir / "refs" / "nominal.npz").q_home)
        self.tasks = list(cfg.tasks)
        w = np.ones(len(self.tasks)) if cfg.task_weights is None else np.asarray(cfg.task_weights, float)
        self.task_p = w / w.sum()
        self.tracks = {t: control.build_track(p, self.run_dir, TASKS[t][0], self.omega) for t in self.tasks}
        self.period = {t: Reference.load(self.run_dir / "refs" / f"{TASKS[t][1]}.npz").period for t in self.tasks}
        if cfg.gains == "tuned":
            self.gains, self.gains_info = tuned_gains(self.run_dir, cfg.gains_seed)
        else:
            self.gains = np.asarray(control.GAIN_SETS[cfg.gains], float).copy()
            self.gains_info = dict(source=control.GAINS_SOURCE[cfg.gains], flag="untuned (asked for)")
        self._foot_of = control.foot_geoms(p)
        # nominal model values the randomisation scales (restored every episode)
        self._mu0 = m.geom_friction.copy()
        self._mass0, self._inertia0 = float(m.body_mass[p.trunk]), m.body_inertia[p.trunk].copy()
        self._gain0, self._bias0 = m.actuator_gainprm[:, 0].copy(), m.actuator_biasprm[:, 1].copy()
        self._legs = np.array([not jn.startswith(("neck", "tail")) for jn in p.joints])
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in p.joints]
        self.lo, self.hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        self._foot_tables = metrics._lowest_point_tables(m, p.feet)
        self._R_home = p._R_home
        self.gaussian_tracking_reward = _gaussian_tracking_reward()
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * FILTER_HZ * self.dt_env)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.layout = obs_layout()
        self._step_hook = None  # called after every physics step (zero_residual_check)
        self._setup()
        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ hooks for R3 / R4
    def _setup(self) -> None:
        raise NotImplementedError

    def _begin(self, seeds: dict) -> None:
        raise NotImplementedError

    def _apply_action(self, a: np.ndarray) -> None:
        raise NotImplementedError

    def _physics_step(self) -> None:
        raise NotImplementedError

    def _clock(self) -> float:
        """The reference time the phase, the command and the imitation read."""
        raise NotImplementedError

    def _estimate(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ episode
    def _draw_factors(self, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        f = dict(friction=None, mass=1.0, kp=1.0, latency_steps=0, noise_scale=1.0)
        if cfg.randomize:
            f.update(friction=float(rng.uniform(*cfg.friction)), mass=float(rng.uniform(*cfg.mass)),
                     kp=float(rng.uniform(*cfg.kp)), latency_steps=int(rng.integers(cfg.latency_steps[0],
                                                                                   cfg.latency_steps[1] + 1)),
                     noise_scale=float(rng.uniform(*cfg.noise_scale)))
        return f

    def _apply_factors(self, f: dict) -> None:
        """The model back to nominal, then the episode's factors (friction None: the model's own)."""
        p, m = self.p, self.p.m
        m.geom_friction[:] = self._mu0
        m.body_mass[p.trunk], m.body_inertia[p.trunk] = self._mass0, self._inertia0
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = self._gain0, self._bias0
        if f.get("friction") is not None:
            control.set_friction(p, f["friction"])
        if f.get("mass", 1.0) != 1.0:
            m.body_mass[p.trunk] = self._mass0 * f["mass"]
            m.body_inertia[p.trunk] = self._inertia0 * f["mass"]
        if f.get("kp", 1.0) != 1.0:
            m.actuator_gainprm[:, 0] = self._gain0 * f["kp"]
            m.actuator_biasprm[:, 1] = self._bias0 * f["kp"]

    def _draw_push(self, rng: np.random.Generator, seconds: float):
        cfg = self.cfg
        if not (cfg.push_prob > 0 and rng.random() < cfg.push_prob and seconds > 4.0):
            return None
        t0 = float(rng.uniform(2.0, seconds - 2.0))
        dv = float(rng.uniform(*cfg.push_frac)) * PUSH_BOUND_M_S.get(cfg.body, 0.5)
        ang = float(rng.uniform(-math.pi, math.pi))
        F = float(self.p.m.body_subtreemass[self.p.trunk]) * dv / cfg.push_s
        return (t0, cfg.push_s, F * math.cos(ang), F * math.sin(ang))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        opts = options or {}
        self.task = opts.get("task") or str(rng.choice(self.tasks, p=self.task_p))
        self.track = self.tracks[self.task]
        self.stride_s = self.period[self.task] / self.cfg.rate
        seconds = STAND_SECONDS if self.task == "stand" else self.track.t_end / self.cfg.rate
        self.seconds = min(seconds, self.cfg.max_seconds)
        self.n_steps = int(round(self.seconds / self.dt_env))
        drawn = self._draw_factors(rng)
        self.factors = dict(drawn, **(opts.get("factors") or {}))
        self._apply_factors(self.factors)
        self.push = opts["push"] if "push" in opts else self._draw_push(rng, self.seconds)
        ep = int(opts.get("episode_seed", rng.integers(2**31 - 1)))
        self.episode_seed = ep
        seeds = dict(start=int(np.random.SeedSequence([ep, 0]).generate_state(1)[0]),
                     noise=int(np.random.SeedSequence([ep, 1]).generate_state(1)[0]))
        self.k_env = 0
        self.a_prev = np.zeros(ACTION_DIM)
        self.a_filt = np.zeros(ACTION_DIM)
        self.contact_hist = deque(maxlen=max(1, int(round(self.stride_s / self.dt_env))))
        self._latency_off()
        self._begin(seeds)
        if self.factors["latency_steps"]:
            self._latency_on(self.factors["latency_steps"])
        self._com0 = self.p._v.subtree_com[self.p.trunk][:2].copy()
        self._yaw0 = control._yaw(self.p._v.xmat[self.p.trunk])
        self._yaw_unwrapped = 0.0
        self._ref_yaw0 = float(self.track.info(self._clock())[1]["yaw"])
        return self._observation(), self._info()

    # ------------------------------------------------------------------ latency (a delayed ctrl)
    def _latency_on(self, steps: int) -> None:
        p = self.p
        orig = type(p).command.__get__(p)
        queue = deque([p.d.ctrl.copy() for _ in range(steps)], maxlen=steps + 1)

        def command(*args, **kw):
            orig(*args, **kw)
            ctrl = p._v.ctrl
            queue.append(ctrl.copy())
            ctrl[:] = queue[0]

        p.command = command

    def _latency_off(self) -> None:
        self.p.__dict__.pop("command", None)

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.a_filt = self.a_filt + self.alpha * (a - self.a_filt)
        self._apply_action(self.a_filt)
        p = self.p
        weight = self.weight
        loaded = 0.02 * weight
        clock0 = self._clock()
        grf_peak, flight, contacts = 0.0, 0, np.zeros(2)
        push = self.push
        hook = self._step_hook
        fall = None
        for _ in range(ENV_SUBSTEPS):
            if push is not None:
                control._apply_push(p, push)
            self._physics_step()
            if hook is not None:
                hook(p)
            l0, l1 = p.foot_loads().tolist()
            grf_peak = max(grf_peak, l0 + l1)
            c0, c1 = l0 > loaded, l1 > loaded
            flight += not (c0 or c1)
            contacts[0] += c0
            contacts[1] += c1
        fall = p.fall_cause()
        if fall is None and any(d.number for d in p.d.warning):
            fall = "MuJoCo warning"
        self.k_env += 1
        self.contact_hist.append(contacts / ENV_SUBSTEPS)
        r, terms = self._reward(clock0, grf_peak, flight)
        self.a_prev = a
        terminated = fall is not None
        truncated = (not terminated) and self.k_env >= self.n_steps
        info = self._info()
        info.update(fall_cause=fall, reward_terms=terms)
        return self._observation(), float(r), terminated, truncated, info

    def _info(self) -> dict:
        return dict(step=self.k_env, task=self.task, mode=self.mode, factors=self.factors, push=self.push,
                    episode_seed=self.episode_seed, time_s=float(self.p.d.time))

    # ------------------------------------------------------------------ reward
    def _reward(self, clock0: float, grf_peak: float, flight: int) -> tuple[float, dict]:
        p, v = self.p, self.p._v
        trunk = p.trunk
        clock = self._clock()
        x = self.track.joint(clock)
        nj = self.nj
        q = v.qpos[p.qadr]
        imit = math.exp(-float(np.mean(np.square((q - x[:nj]) / IMITATION_SCALE_RAD))))
        com = v.subtree_com[trunk][:2]
        yaw = control._yaw(v.xmat[trunk])
        dyaw = control._wrap(yaw - self._yaw0)
        self._yaw_unwrapped += dyaw
        fwd = np.array([math.cos(self._yaw0), math.sin(self._yaw0)])
        speed = float((com - self._com0) @ fwd) / self.dt_env
        yaw_rate = dyaw / self.dt_env
        self._com0, self._yaw0 = com.copy(), yaw
        v_cmd, w_cmd = self._command(clock0)
        track = self.gaussian_tracking_reward(abs(speed - v_cmd), abs(yaw_rate - w_cmd))
        _, ref = self.track.info(clock)
        pen = dict.fromkeys(PENALTIES, 0.0)
        walking = not bool(ref["stand"])
        if walking and len(self.contact_hist) == self.contact_hist.maxlen:
            duty = np.mean(self.contact_hist, axis=0)
            pen["duty"] = min(1.0, float(np.sum(np.maximum(0.0, np.maximum(0.55 - duty, duty - 0.65)))) / 0.05)
        loads = p.foot_loads()
        heavy = [i for i in (0, 1) if loads[i] > 0.2 * self.weight]
        if heavy:
            s = control.foot_slip_speed(p, heavy, self._foot_of)
            pen["slip"] = min(1.0, max(0.0, float(s.max()) - 0.05) / 0.05)
        pen["flight"] = flight / ENV_SUBSTEPS
        pen["impact"] = min(1.0, max(0.0, grf_peak / self.weight - 1.6) / 1.6)
        sw, u = int(ref["swing"]), float(ref["u"])
        if sw >= 0 and 0.25 <= u <= 0.75:
            low = metrics.foot_lowest(p.m, p.d, *self._foot_tables)[sw] - p.z_floor
            pen["clearance"] = min(1.0, max(0.0, 0.04 * self.H - low) / (0.04 * self.H))
        Rr = v.xmat[trunk].reshape(3, 3) @ self._R_home.T
        roll, pitch = control._roll_pitch(Rr)
        pen["tilt"] = min(1.0, max(0.0, math.degrees(max(abs(roll), abs(pitch))) - 4.0) / 4.0)
        r = IMITATION_W * imit + TRACKING_W * track - sum(PENALTIES[k] * pen[k] for k in PENALTIES)
        return r, dict(imitation=imit, tracking=track, **pen)

    def _command(self, clock: float) -> tuple[float, float]:
        """The reference's forward speed and yaw rate at the reference time ``clock``, at the commanded rate."""
        _, ref = self.track.info(clock)
        h = 0.02
        y0 = float(ref["yaw"])
        y1 = float(self.track.info(min(clock + h, self.track.t_end))[1]["yaw"])
        rate = self.cfg.rate
        return rate * float(ref["comv_rel"][0]), rate * (y1 - y0) / h

    # ------------------------------------------------------------------ observation
    def _observation(self) -> np.ndarray:
        p, v = self.p, self.p._v
        trunk = p.trunk
        est = self._estimate()
        clock = self._clock()
        _, ref = self.track.info(clock)
        phase = 2.0 * math.pi * (clock / self.period[self.task])
        sw = int(ref["swing"])
        v_cmd, w_cmd = self._command(clock)
        R = v.xmat[trunk].reshape(3, 3)
        yaw = control._yaw(R)
        c, s = math.cos(yaw), math.sin(yaw)
        vel = v.qvel[:3]
        Rr = R @ self._R_home.T
        roll, pitch = control._roll_pitch(Rr)
        x = self.track.joint(clock)
        nj = self.nj
        f = self.factors
        push = self.push
        pushing = push is not None and push[0] <= p.d.time < push[0] + push[1]
        floor = np.zeros(3)
        floor[FLOOR_CODES.get(self.cfg.floor, 2)] = 1.0
        e_dcm = self._true_dcm_error(ref)
        heading_err = control._wrap(self._yaw_unwrapped - (float(ref["yaw"]) - self._ref_yaw0)) if hasattr(
            self, "_yaw_unwrapped") else 0.0
        actor = np.concatenate([
            est["q"], est["qd"], est["gravity"], est["gyro"], est["foot_load"] / self.weight,
            [math.sin(phase), math.cos(phase)], [1.0 if sw != 0 else 0.0, 1.0 if sw != 1 else 0.0],
            [v_cmd / COMMAND_SCALE, w_cmd / COMMAND_SCALE], self.a_prev])
        critic = np.concatenate([
            [c * vel[0] + s * vel[1], -s * vel[0] + c * vel[1], vel[2]], [(v.qpos[2] - p.z_floor) / self.H],
            [roll, pitch], e_dcm, v.qpos[p.qadr] - x[:nj], [heading_err],
            [f["friction"] if f["friction"] is not None else 0.9], floor,
            [push[2] / (self.weight / GRAVITY) if pushing else 0.0, push[3] / (self.weight / GRAVITY) if pushing
             else 0.0, 1.0 if pushing else 0.0],
            [f["mass"], f["kp"], f["latency_steps"] * self.dt / 0.02]])
        return np.concatenate([actor, critic]).astype(np.float32)

    def _true_dcm_error(self, ref: dict) -> np.ndarray:
        """The true DCM against the reference's, relative to the anchor foot in its frame (``metrics.Recorder``'s
        reading, at the commanded rate)."""
        p, v = self.p, self.p._v
        mujoco.mj_subtreeVel(p.m, p.d)  # (outputs only: the state is untouched)
        com, cv = v.subtree_com[p.trunk], v.subtree_linvel[p.trunk]
        a = int(ref["anchor"])
        if a < 2:
            pa, ya = v.xpos[p.feet[a]][:2], control._yaw(v.xmat[p.feet[a]])
        else:
            fy = [control._yaw(v.xmat[f]) for f in p.feet]
            pa = 0.5 * (v.xpos[p.feet[0]][:2] + v.xpos[p.feet[1]][:2])
            ya = math.atan2(math.sin(fy[0]) + math.sin(fy[1]), math.cos(fy[0]) + math.cos(fy[1]))
        xi = com[:2] + cv[:2] / self.omega
        rate = self.cfg.rate
        ref_dcm = ref["dcm_rel"] if rate == 1.0 else ref["dcm_rel"] + (rate - 1.0) * ref["comv_rel"] / self.omega
        return control._rot2(ya).T @ (xi - pa) - ref_dcm

    def describe(self) -> dict:
        return dict(mode=self.mode, config=self.cfg.describe(), gains=control.gains_dict(self.gains),
                    gains_info=self.gains_info, obs=self.layout, actor_obs=list(ACTOR_OBS), critic_obs=list(CRITIC_OBS),
                    action_dim=ACTION_DIM, substeps=ENV_SUBSTEPS, filter_hz=FILTER_HZ, filter_alpha=self.alpha,
                    reward=REWARD_RULE, penalties=PENALTIES, tasks={t: TASKS[t] for t in self.tasks})


class ResidualEnv(Stage1Env):
    """R3: R2 plus the learned residual (module docstring)."""

    mode = "R3"

    def _setup(self) -> None:
        self.trackers = {t: control.Tracker(self.p, self.tracks[t], "R2", gains=self.gains, rate=self.cfg.rate)
                         for t in self.tasks}
        scale = np.where(self._legs, RESIDUAL_LEG_RAD, RESIDUAL_NECK_TAIL_RAD)
        self.residual_scale = scale
        self.tk = self.trackers[self.tasks[0]]

    def _begin(self, seeds: dict) -> None:
        tk = self.tk = self.trackers[self.task]
        tk.noise = estimator.SensorNoise(scale=self.factors["noise_scale"])
        tk.residual, tk.rate_factor = None, 1.0
        tk.reset("stance", joint_noise=self.cfg.start_joint_noise, seed=seeds["start"],
                 root_velocity_noise=self.cfg.start_root_velocity_noise, noise_seed=seeds["noise"])

    def _apply_action(self, a: np.ndarray) -> None:
        tk = self.tk
        off = a[: self.nj] * self.residual_scale
        tk.residual = off if off.any() else None  # (no offset at all: R2's targets, bit for bit)
        tk.rate_factor = 1.0 + RESIDUAL_RATE * float(a[self.nj])

    def _physics_step(self) -> None:
        self.tk.step()

    def _clock(self) -> float:
        return self.tk.tau_ref

    def _estimate(self) -> dict:
        e = self.tk.est
        return dict(q=e.q, qd=e.qd, gravity=-e.R[2], gyro=e.gyro, foot_load=e.load)


class PolicyOnlyEnv(Stage1Env):
    """R4: the policy sets the joint targets directly (module docstring)."""

    mode = "R4"

    def _setup(self) -> None:
        p = self.p
        q_home = p.q_settled[p.qadr]
        ex = np.zeros(self.nj)
        for tr in self.tracks.values():
            for seg in tr.segments:
                ex = np.maximum(ex, np.max(np.abs(seg.S[:, : self.nj] - q_home), axis=0))
        self.span = np.maximum(POLICY_ONLY_SPAN_FACTOR * ex, POLICY_ONLY_MIN_SPAN_RAD)
        self.q_home = q_home
        self.q_target = q_home.copy()

    def _begin(self, seeds: dict) -> None:
        p = self.p
        p.reset(joint_noise=self.cfg.start_joint_noise, root_velocity_noise=self.cfg.start_root_velocity_noise,
                seed=seeds["start"])
        noise = estimator.SensorNoise(scale=self.factors["noise_scale"])
        self.sensors = estimator.NoisySensors(p, noise, seeds["noise"])
        self.est = estimator.Estimator(p)
        self.est.reset_at_rest(self.sensors.read(), yaw=control._yaw(p.d.xmat[p.trunk]), base_xy=p.d.qpos[:2].copy())
        self.tau = 0.0
        self.k = 0
        self.rate_factor = 1.0
        self.q_target = p.q_settled[p.qadr].copy()

    def _apply_action(self, a: np.ndarray) -> None:
        self.q_target = np.clip(self.q_home + a[: self.nj] * self.span, self.lo, self.hi)
        self.rate_factor = 1.0 + RESIDUAL_RATE * float(a[self.nj])

    def _physics_step(self) -> None:
        p = self.p
        p.command(self.q_target)
        p.step()
        self.k += 1
        self.tau = min(self.tau + self.dt * self.cfg.rate * self.rate_factor, self.track.t_end)
        self.est.step(self.sensors.read())
        if self.k % control.FEEDBACK_EVERY == 0:
            self.est.update(control.FEEDBACK_EVERY * self.dt)

    def _clock(self) -> float:
        return self.tau

    def _estimate(self) -> dict:
        e = self.est
        return dict(q=e.q, qd=e.qd, gravity=-e.R[2], gyro=e.gyro, foot_load=e.load)


def make(mode: str, config: EnvConfig, seed: int | None = None) -> Stage1Env:
    if mode == "R3":
        return ResidualEnv(config, seed)
    if mode == "R4":
        return PolicyOnlyEnv(config, seed)
    raise ValueError(f"mode {mode!r}: R3 or R4")


# ------------------------------------------------------------------------------------------------
# G6a: a zero residual reproduces R2 exactly; env throughput
# ------------------------------------------------------------------------------------------------
def _state_digest(d) -> bytes:
    import hashlib

    return b"".join(hashlib.sha256(a.tobytes()).digest() for a in (d.qpos, d.qvel, d.ctrl, d.sensordata))


def zero_residual_check(body: str, task: str = "walk", seconds: float = 4.0, episode_seed: int = 7,
                        run_dir: str | None = None, gains: str = "tuned", gains_seed: int = 42) -> dict:
    """G6a's first half: R3 with a zero action, every physics step's qpos, qvel, ctrl and sensordata against R2 run
    by ``control.run`` from the same stance, seeds and gains (sha256 of each, per step)."""
    cfg = EnvConfig(body=body, tasks=(task,), randomize=False, push_prob=0.0, run_dir=run_dir, gains=gains,
                    gains_seed=gains_seed, max_seconds=seconds)
    env = ResidualEnv(cfg)
    digests = []
    env._step_hook = lambda p: digests.append(_state_digest(p.d))
    env.reset(seed=0, options=dict(task=task, episode_seed=episode_seed, push=None))
    a = np.zeros(ACTION_DIM, np.float32)
    steps = 0
    for _ in range(int(round(seconds / env.dt_env))):
        _, _, term, trunc, _ = env.step(a)
        steps += 1
        if term:
            break
    env._step_hook = None
    # R2 itself on the same plant and track
    tk = control.Tracker(env.p, env.tracks[task], "R2", gains=env.gains, rate=cfg.rate)
    ref_digests = []

    class _Hash:
        def begin(self, tk_, n):
            pass

        def step(self, k):
            ref_digests.append(_state_digest(env.p.d))

        def update(self, *a_):
            pass

        def end(self, *a_):
            pass

    seeds = dict(start=int(np.random.SeedSequence([episode_seed, 0]).generate_state(1)[0]),
                 noise=int(np.random.SeedSequence([episode_seed, 1]).generate_state(1)[0]))
    control.run(tk, len(digests) * env.dt, start="stance", seed=seeds["start"], noise_seed=seeds["noise"],
                recorder=_Hash(), joint_noise=cfg.start_joint_noise, root_velocity_noise=cfg.start_root_velocity_noise)
    n = min(len(digests), len(ref_digests))
    first = next((i for i in range(n) if digests[i] != ref_digests[i]), None)
    return dict(rule="R3 with a zero action against R2 (control.run): sha256 of qpos, qvel, ctrl and sensordata after "
                     "every physics step, from the same settled stance, seeds and gains",
                body=body, task=task, seconds=seconds, episode_seed=episode_seed, physics_steps=len(digests),
                reference_steps=len(ref_digests), identical=bool(first is None and len(digests) == len(ref_digests)),
                first_difference=first, gains_info=env.gains_info)


def throughput(mode: str, body: str, env_steps: int = 500, seed: int = 0, run_dir: str | None = None,
               config: EnvConfig | None = None, policy=None) -> dict:
    """Env steps per second of one env in this process (random actions unless ``policy``; the env reset as its
    episodes end): per CPU second of the process (what one core gives, whatever else runs) and per wall second."""
    import time

    cfg = config or EnvConfig(body=body, run_dir=run_dir)
    env = make(mode, cfg, seed=seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    t0, c0 = time.perf_counter(), time.process_time()
    resets = 0
    for _ in range(env_steps):
        a = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32) if policy is None else policy(obs)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
            resets += 1
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return dict(mode=mode, body=body, env_steps=env_steps, resets=resets, cpu_s=cpu, wall_s=wall,
                env_steps_per_cpu_s=env_steps / cpu, env_steps_per_wall_s=env_steps / wall,
                rule="random actions in [-1, 1] (each episode reset as it ends, resets included); per CPU second of "
                     "this process and per wall second")


__all__ = ["ACTION_DIM", "ACTOR_DIM", "ACTOR_OBS", "CRITIC_OBS", "EnvConfig", "OBS_DIM", "PolicyOnlyEnv",
           "ResidualEnv", "Stage1Env", "TASKS", "make", "obs_layout", "throughput", "tuned_gains",
           "zero_residual_check"]
