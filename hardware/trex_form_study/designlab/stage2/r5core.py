"""What R5's env and the policy rows share: layout, observation, action mapping, head overlay, reward, the R6 tracker
(round C design D7-D10; sections 3.1, 3.4; rules E1-E5, W1-W5, J4).

One ``Core`` sits on one plant, one clock, one estimator and one ``r5lib.Script``.  The env (``r5env``) and a
policy row (``s2rows``) drive the same methods in the same order, so a row turns the tracker's state into
observations, and actions into joint offsets, with the code the policy was trained with:

    core.begin(clock, est, script, head)          after the episode's reset
    obs = core.actor_obs()                        the deployable inputs (``layout``)
    a_f = core.filter(action)                     clip, first-order low-pass, the three-action history
    tk.residual, tk.rate_factor = core.offsets(a_f)        R5;  ``targets`` for R6

and, in the env only, ``substep()`` after every physics step, ``interference()`` and ``reward(clock0)`` at the
env step's end and ``critic_obs`` (the privileged inputs).

**Zero is zero.**  A zero action with the head neutral leaves ``offsets`` at ``(None, 1.0)``: no array of zeros
and no ``1.0 + 0.0`` that only happens to be one, so the tracker is R0L bit for bit (gate G8a).

**The plant's data is only read.**  Nothing here calls ``mj_forward`` or ``mj_kinematics`` on the plant's data (it
would change the solver's warm start): positions are the plant's own, as of the last step's start, like the
sensors; joint angles are ``qpos``, as Stage 1's reward reads them.

**Reward** (``REWARD_RULE``, ``REWARD_WEIGHTS``, ``REWARD_SCALES``): rule W2's table.  Every term lies in [0, 1].

``Core`` must be built while the plant's model is nominal (it reads the nominal weight).  This module imports
``r5lib``; it never imports ``r5env``.
"""

from __future__ import annotations

import math
from collections import deque

import mujoco
import numpy as np

from .. import clearance
from ..stage1 import control, env, legs, metrics, plant
from . import r5lib

SUBSTEPS = env.ENV_SUBSTEPS  # 2 ms physics steps per 50 Hz env step
G = control.G

# ------------------------------------------------------------------------------------------------
# Layout (section 3.1; rule E3)
# ------------------------------------------------------------------------------------------------
HEAD_X_JOINTS = ("neck_yaw", "jaw")
COMMAND_SCALE = (0.5, 1.0)  # m/s, rad/s
HEADING_SCALE_RAD = 0.5
HEADING_CLIP = 2.0
HEAD_COMMAND_SCALE_RAD = 0.6
DELAY_SCALE_S = 0.024  # the largest randomised delay: the plant-factor input and the contact terms' lag
LAYOUT_RULE = (
    "actor (deployable): q, qd (the estimator's encoder angles and rates, plant joint order), gravity (-R[2] of "
    "the estimator's attitude), gyro, foot_load (over the nominal weight), phase (sin, cos of 2 pi the script's "
    "phase), support (the reference's flags), command ([v / 0.5 m/s, w / 1.0 rad/s] x the commanded clock rate), "
    "heading (the estimator's yaw about home less the commanded heading, over 0.5 rad, clipped to +-2; one input "
    "beyond plan B.1's list), prev_actions (the last three clipped, unfiltered actions, newest first), and on a "
    "body with a neck yaw and a jaw head_command ([yaw now, jaw now, yaw goal, jaw goal] / 0.6 rad).  Critic "
    "(privileged): root_vel (heading frame), height (over H), roll_pitch about home, dcm_error, joint_error (true "
    "less reference with the head overlay), heading_error (true yaw less the reference's), friction, floor (hard, "
    "robot_floor_carpet), push, plant_factors (mass, stiffness, delay / 24 ms, joint damping, joint friction), "
    "foot_error (the feet relative to the root in its heading frame, less the reference's), ref_contact (the "
    "reference's support 24 ms x the clock rate earlier).  Action: nj joint entries in plant order, then the clock")


def layout(joints: list[str], heading: bool = True) -> dict:
    """name -> (start, stop) of every observation block for a plant's joint list, the actor's and the whole
    observation's width and the action's.  ``heading`` False drops the heading block (every later block one
    lower)."""
    joints = list(joints)
    nj = len(joints)
    head_x = all(j in joints for j in HEAD_X_JOINTS)
    actor = [("q", nj), ("qd", nj), ("gravity", 3), ("gyro", 3), ("foot_load", 2), ("phase", 2), ("support", 2),
             ("command", 2)]
    if heading:
        actor.append(("heading", 1))
    actor.append(("prev_actions", 3 * (nj + 1)))
    if head_x:
        actor.append(("head_command", 4))
    critic = [("root_vel", 3), ("height", 1), ("roll_pitch", 2), ("dcm_error", 2), ("joint_error", nj),
              ("heading_error", 1), ("friction", 1), ("floor", 2), ("push", 3), ("plant_factors", 5),
              ("foot_error", 6), ("ref_contact", 2)]
    blocks, i = {}, 0
    for name, w in actor + critic:
        blocks[name] = (i, i + w)
        i += w
    return dict(blocks=blocks, actor_dim=sum(w for _, w in actor), obs_dim=i, action_dim=nj + 1, head_x=head_x,
                nj=nj, heading=bool(heading))


# ------------------------------------------------------------------------------------------------
# Reward constants (rule W2's table; a test compares them with a literal copy of it)
# ------------------------------------------------------------------------------------------------
POSITIVE = ("joint", "tracking", "foot", "contact", "head")
REWARD_WEIGHTS = dict(joint=0.55, tracking=0.35, foot=0.15, contact=0.10, head=0.10,
                      duty=0.02, slip=0.02, flight=0.05, impact=0.02, clearance=0.02, tilt=0.02,
                      action_rate=0.03, touchdown=0.02, servo=0.05, interference=0.10,
                      progress=0.10, heading=0.05, range=0.05)
REWARD_WEIGHTS_ITEMS = tuple(REWARD_WEIGHTS.items())
PENALTIES = tuple(k for k in REWARD_WEIGHTS if k not in POSITIVE)
REWARD_SCALES = dict(
    joint_rad=0.1, foot_frac_H=0.10, head_deg=3.0, contact_load_frac=0.02, contact_lag_s=DELAY_SCALE_S,
    duty_band=(0.55, 0.65), duty_band_slow=(0.55, 0.72), duty_slow_below_m_s=0.30, duty_over=0.05,
    slip_load_frac=0.2, slip_free_m_s=0.05, slip_scale_m_s=0.05, impact_free_bw=1.6, impact_scale_bw=1.6,
    clearance_frac_H=0.04, mid_swing=r5lib.MID_SWING, tilt_free_deg=4.0, tilt_scale_deg=4.0,
    action_rate=0.25, touchdown_free_m_s=0.15, touchdown_scale_m_s=0.30, servo_tau_s=1.0, servo_over=0.5,
    interference_m=clearance.TIGHT, progress_short=0.10, heading_free_deg=3.0, heading_scale_deg=17.0,
    range_rad=0.02)
REWARD_RULE = (
    "r = 0.55 joint + 0.35 tracking + 0.15 foot + 0.10 contact + 0.10 head - sum of the penalties x their weights; "
    "every term in [0, 1], read on the true state at the env step's end (c the reference clock then, c0 at its "
    "start).  joint: exp(-mean_j ((q_j - q_ref_j(c)) / 0.1 rad)^2) over all hinges, the reference with the head "
    "overlay.  tracking: Stage 1's gaussian_tracking_reward of the CoM's speed along the previous heading and the "
    "trunk's yaw rate over the step against the reference's own at c0 times the commanded clock rate.  foot: "
    "exp(-mean_f |r_f - r_ref_f(c)|^2 / (0.10 H)^2), r the foot body's position less the root's in the root's "
    "heading frame.  contact: the share, over both feet and the step's physics steps, of steps in which the foot's "
    "contact (load above 2 % of the true weight) equals the reference's support at the clock or 24 ms x the clock "
    "rate earlier.  head: exp(-((roll - roll_ref)^2 + (pitch - pitch_ref)^2) / (3 deg)^2) of the head body in the "
    "world about home.  Penalties: duty (Stage 1's: each foot's contact share over the last stride outside "
    "0.55-0.65, or 0.55-0.72 while the commanded speed is below 0.30 m/s, over 0.05, while the reference walks; a "
    "foot is read only where the reference's own share for it over the same stride lies inside the band, so that "
    "a start, a stop or a change, where the reference itself is outside, is not charged); slip, flight, impact, "
    "tilt (Stage 1's); clearance, the larger of (a) for a foot in the air between 25 % and 75 % of the planned "
    "length of the reference's swing in progress (else of its next) since its own lift-off, on the reference "
    "clock, and lower than 0.04 H, the shortfall over 0.04 H, and (b) 1 if in any physics step of the env step a "
    "foot is loaded while the reference "
    "has that foot in mid-swing both at the clock and 24 ms x the clock rate earlier; action_rate: min(1, mean_i "
    "(a_i - a_prev_i)^2 / 0.25) on the clipped actions; touchdown: in a step in which a foot's contact begins, "
    "min(1, max(0, v_down - 0.15 m/s) / 0.30 m/s), v_down the foot body's downward speed one physics step before; "
    "servo: per drive that is no leg drive, a bias-corrected exponential mean of the squared actuator force (time "
    "constant 1.0 s), min(1, max_d max(0, RMS_d / sustained_d - 1) / 0.5); interference: clip((3 mm - g) / 3 mm, "
    "0, 1), g the smallest distance over the moving left-to-right leg pairs; progress: only while the script is "
    "one stride or more into a periodic walking segment with a positive command and the mean has run one stride, "
    "min(1, max(0, (v_ref_bar - v_bar) / (rate x v_cmd)) / 0.10), v_bar the bias-corrected exponential mean (time "
    "constant: the stride period over the clock rate; restarted at the segment's first frame) of the CoM's speed "
    "along the commanded heading and v_ref_bar the same mean of the reference's own CoM speed along its stride "
    "direction at the commanded clock rate over the same steps (its stride mean is rate x v_cmd; a playback's "
    "speed ripple then charges nothing); heading: clip((|yaw - yaw_home - commanded heading(c)| - 3 deg) / 17 deg, "
    "0, 1) on the "
    "true yaw, zero in a hold; range: min(1, max_j max(0, required_j - margin_j) / 0.02 rad), margin the true "
    "joint's distance to its nearer limit (an end a joint rests on by design is none), required the script's.  "
    "progress, heading and range are beyond plan B.1's list (rule W5)")
INTERFERENCE_RULE = (
    "legs.LegGap at the end of every env step: the penalty reads the pairs that move (home distance at least the "
    "report cap); pairs that stay at one distance in every pose carry none and are listed; termination when the "
    "smallest distance over all pairs is below zero at two env-step ends in a row.  A one-pose contact is "
    "penalised, not terminated.  A leg against the trunk or the tail is not read")
#: How far the contact history reaches back (env steps): longer than any stride at the commanded clock.
HISTORY_STEPS = 256

# ------------------------------------------------------------------------------------------------
# The head command (rules E5, J4)
# ------------------------------------------------------------------------------------------------
GAZE_YAW_RAD = 0.5
GAZE_MOVE_S = (0.8, 1.2)
GAZE_HOLD_S = (2.0, 6.0)
JAW_OPEN_RAD = (0.3, 0.6)
JAW_MOVE_S = (0.4, 0.6)
JAW_HOLD_S = (1.0, 1.5)
JAW_AT = (0.25, 0.75)  # the jaw event starts at a time uniform in this share of the episode (its middle half)
H1_STREAM = 77
HEAD_RULE = (
    "the head command is an overlay added to the reference's neck-yaw and jaw targets before the policy's offsets; "
    "the actor sees [yaw now, jaw now, yaw goal, jaw goal] / 0.6 rad; head neutral is all zero.  A gaze train: "
    "from the episode's first instant, shifts to a yaw target uniform on +-0.5 rad, minimum-jerk over 0.8-1.2 s, "
    "each held 2-6 s, to the episode's end.  A jaw event: at a time uniform in the episode's middle half, open to "
    "0.3-0.6 rad over 0.4-0.6 s, held 1.0-1.5 s, closed over the same time.  Training draws a train with "
    "probability gaze_prob and a jaw event with probability jaw_prob from the env's generator (rule J4); the "
    "bar's head script H1 has both, from numpy SeedSequence([episode seed, 77]) (rule E5)")


class HeadScript:
    """The ``-X`` head command of one episode: ``events`` of (kind, t0, duration, hold, target) on the episode's
    simulated clock; ``value(t)`` -> (yaw, jaw, yaw goal, jaw goal) in rad."""

    def __init__(self, events):
        self.events = [(str(k), float(t0), float(d), float(h), float(x)) for k, t0, d, h, x in events]
        self._gaze = [e for e in self.events if e[0] == "gaze"]
        self._jaw = [e for e in self.events if e[0] == "jaw"]

    @staticmethod
    def _draw(rng: np.random.Generator, seconds: float, gaze: bool, jaw: bool) -> "HeadScript":
        events, t = [], 0.0
        while gaze and t < seconds:
            target = float(rng.uniform(-GAZE_YAW_RAD, GAZE_YAW_RAD))
            move, hold = float(rng.uniform(*GAZE_MOVE_S)), float(rng.uniform(*GAZE_HOLD_S))
            events.append(("gaze", t, move, hold, target))
            t += move + hold
        if jaw:
            t0 = float(rng.uniform(JAW_AT[0] * seconds, JAW_AT[1] * seconds))
            target = float(rng.uniform(*JAW_OPEN_RAD))
            events.append(("jaw", t0, float(rng.uniform(*JAW_MOVE_S)), float(rng.uniform(*JAW_HOLD_S)), target))
        return HeadScript(events)

    @classmethod
    def drawn(cls, rng: np.random.Generator, seconds: float, gaze_prob: float, jaw_prob: float) -> "HeadScript | None":
        """Training's draw (rule J4): None when neither a gaze train nor a jaw event is drawn (head neutral)."""
        gaze = bool(gaze_prob > 0 and rng.random() < gaze_prob)
        jaw = bool(jaw_prob > 0 and rng.random() < jaw_prob)
        return cls._draw(rng, float(seconds), gaze, jaw) if gaze or jaw else None

    @classmethod
    def fixed(cls, seed: int, seconds: float) -> "HeadScript":
        """The bar's head script H1 (rule E5) of an episode seed: a gaze train and one jaw event."""
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), H1_STREAM]))
        return cls._draw(rng, float(seconds), True, True)

    def value(self, t: float) -> tuple[float, float, float, float]:
        yaw = yaw_goal = jaw = jaw_goal = 0.0
        prev = 0.0
        for _, t0, d, _, x in self._gaze:
            if t < t0:
                break
            yaw, yaw_goal = prev + (x - prev) * control._min_jerk((t - t0) / d), x
            prev = x
        for _, t0, d, h, x in self._jaw:
            if t0 <= t < t0 + d + h:
                jaw, jaw_goal = x * control._min_jerk((t - t0) / d), x
            elif t >= t0 + d + h:
                jaw = x * (1.0 - control._min_jerk((t - t0 - d - h) / d))
        return yaw, jaw, yaw_goal, jaw_goal

    def describe(self) -> dict:
        return dict(events=[list(e) for e in self.events])


# ------------------------------------------------------------------------------------------------
# R6: the policy sets the joint targets (rule E2)
# ------------------------------------------------------------------------------------------------
def policy_span(p: plant.Plant, tracks) -> np.ndarray:
    """Stage 1's span rule (``env.POLICY_ONLY_SPAN_RULE``) over ``tracks``: per joint, 1.25 x its largest excursion
    from the settled home pose, at least 0.10 rad."""
    nj = len(p.joints)
    q_home = p.q_settled[p.qadr]
    ex = np.zeros(nj)
    for tr in tracks:
        for seg in tr.segments:
            ex = np.maximum(ex, np.max(np.abs(seg.S[:, :nj] - q_home), axis=0))
    return np.maximum(env.POLICY_ONLY_SPAN_FACTOR * ex, env.POLICY_ONLY_MIN_SPAN_RAD)


def reset_at(tk: control.Tracker, t_ref: float, qpos, qvel, noise_seed: int) -> None:
    """A mid-task start (design D6): the track is lent ``qpos`` and ``qvel`` as its first frame for one call of
    ``control.Tracker.reset("frame")`` (so ``Plant.set_state`` and the estimator's reset from a known state run as
    Stage 1 wrote them), then gets its own back, and the reference clock is set to ``t_ref``."""
    tr = tk.track
    q0, qd0 = tr.q0, tr.qd0
    tr.q0, tr.qd0 = np.array(qpos, float), np.array(qvel, float)
    try:
        control.Tracker.reset(tk, "frame", noise_seed=int(noise_seed))
    finally:
        tr.q0, tr.qd0 = q0, qd0
    tk.tau_ref = float(t_ref)


class R6Tracker(control.Tracker):
    """R6's controller under ``control.run`` and in the env: a ``control.Tracker`` with no feedback whose physics
    step is ``env.PolicyOnlyEnv``'s (``p.command(q_target)`` alone, its own sensors and estimator on ``model``, its
    own clock), as ``policy_rows.PolicyOnlyTracker``.  The clock is not held at the track's end: a script's tables
    continue past it.  ``q_target`` and ``rate_factor`` are what ``Core.targets`` returns."""

    def __init__(self, p: plant.Plant, track: control.Track, rate: float, model=None):
        super().__init__(p, track, "R2", gains=None, terms=(), ankle_cap=(), rate=rate, model=model)
        self.q_target = p.q_settled[p.qadr].copy()

    def reset(self, start: str = "stance", velocity_offset=None, joint_noise: float = 0.0, seed: int | None = None,
              root_velocity_noise: float = 0.0, noise_seed: int | None = None) -> None:
        if start != "stance" or velocity_offset is not None:
            raise ValueError("R6 starts from the settled stance at rest (reset_at for a mid-task start)")
        super().reset(start, None, joint_noise, seed, root_velocity_noise, noise_seed)
        self.rate_factor = 1.0
        self.q_target = self.p.q_settled[self.p.qadr].copy()

    def reset_at(self, t_ref: float, qpos, qvel, noise_seed: int) -> None:
        reset_at(self, t_ref, qpos, qvel, noise_seed)
        self.rate_factor = 1.0
        self.q_target = np.clip(self.track.joint(self.tau_ref)[: self.nj], self.lo, self.hi)

    def step(self) -> None:
        p = self.p
        p.command(self.q_target)
        p.step()
        self.k += 1
        self.tau_ref += self.dt * self.rate * self.rate_factor
        self.est.step(self.sensors.read())
        if self.k % control.FEEDBACK_EVERY == 0:
            self.est.update(control.FEEDBACK_EVERY * self.dt)


def tracker_estimate(tk: control.Tracker) -> dict:
    """What ``Core`` reads of a tracker's estimator (R5's and R6's): ``est`` of ``Core.begin`` is
    ``lambda: tracker_estimate(tk)``."""
    e = tk.est
    return dict(q=e.q, qd=e.qd, gravity=-e.R[2], gyro=e.gyro, foot_load=e.load, yaw=control._yaw(e.R))


# ------------------------------------------------------------------------------------------------
# The core
# ------------------------------------------------------------------------------------------------
class Core:
    """Everything the env and a policy row share, on one plant (module docstring).  ``cfg_core`` holds the keys of
    ``r5env.R5Config`` that an episode executes: ``action_leg_rad``, ``action_other_rad``, ``action_rate``,
    ``rate``, ``filter_hz``, ``heading_input``, ``reward`` (a mapping or (name, weight) pairs) and, for R6, ``span``
    (``policy_span``); a config's ``describe()`` can be passed as it is.  ``lg`` is the leg-gap oracle (None: no
    interference reading)."""

    def __init__(self, p: plant.Plant, cfg_core: dict, lg: legs.LegGap | None):
        cfg = dict(cfg_core)
        self.p, self.lg = p, lg
        m = p.m
        self.nj = len(p.joints)
        self.heading_input = bool(cfg.get("heading_input", True))
        self.layout = layout(p.joints, self.heading_input)
        self.rate = float(cfg.get("rate", 1.05))
        self.dt = float(m.opt.timestep)
        self.dt_env = SUBSTEPS * self.dt
        self.filter_hz = float(cfg.get("filter_hz", 10.0))
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * self.filter_hz * self.dt_env)
        self.leg_rad, self.other_rad = float(cfg.get("action_leg_rad", 0.35)), float(cfg.get("action_other_rad", 0.15))
        self.scale = np.where(plant.leg_joints(p.joints), self.leg_rad, self.other_rad)
        self.action_rate = float(cfg.get("action_rate", 0.25))
        self.weights = dict(REWARD_WEIGHTS)
        self.weights.update({str(k): float(v) for k, v in dict(cfg.get("reward") or ()).items()})
        unknown = [k for k in self.weights if k not in REWARD_WEIGHTS]
        if unknown:
            raise ValueError(f"reward terms {unknown}: choose from {', '.join(REWARD_WEIGHTS)}")
        if not self.heading_input:  # the heading term is struck with its input (rule E3)
            self.weights["heading"] = 0.0
        self.span = None if cfg.get("span") is None else np.asarray(cfg["span"], float)
        self.head_x = bool(self.layout["head_x"])
        self.i_yaw, self.i_jaw = ((p.joints.index(j) for j in HEAD_X_JOINTS) if self.head_x else (None, None))
        self.H = float(p.meta["H_m"])
        #: The nominal weight (the observations' load scale) and the episode's true one (the reward's; ``begin``).
        self.weight = float(m.body_subtreemass[p.trunk]) * G
        self.weight_true = self.weight
        self.lo, self.hi = (m.jnt_range[[_jid(m, jn) for jn in p.joints], i].copy() for i in (0, 1))
        self.rest = plant.rest_ends(p.joints)
        self.q_home = p.q_settled[p.qadr].copy()
        self.R_home = p._R_home
        self.head = r5lib.head_body(p)
        self.R_head_home = r5lib.head_home(p)
        self._foot_of = control.foot_geoms(p)
        self._foot_tables = env.foot_tables(m, p.feet)
        self._load_adr = [int(a) for a in p._load_adr]
        self._servo = np.flatnonzero(~plant.leg_drives(p.drives))
        self._sustained = np.array([p.drives[i].sustained for i in self._servo])
        self._servo_a = 1.0 - math.exp(-self.dt / REWARD_SCALES["servo_tau_s"])
        self._lag = REWARD_SCALES["contact_lag_s"] * self.rate
        self.gaussian_tracking_reward = env._gaussian_tracking_reward()
        self.clock = self.est = self.script = self.head_script = None

    # ---------------------------------------------------------------- episode
    def begin(self, clock, est, script: r5lib.Script, head: HeadScript | None = None) -> None:
        """After the episode's reset: ``clock()`` gives the reference time, ``est()`` the estimate (``q``, ``qd``,
        ``gravity``, ``gyro``, ``foot_load``, ``yaw``).  The histories, the filter and the reward's accumulators
        start at zero; the headings are taken about the start, less what the reference itself had turned by the
        start's reference time (a mid-task start)."""
        if head is not None and not self.head_x:
            raise ValueError("a head script needs a body with a neck yaw and a jaw")
        self.clock, self.est, self.script, self.head_script = clock, est, script, head
        p, v = self.p, self.p._v
        n = self.nj + 1
        self.hist = np.zeros((3, n))
        self.a_f = np.zeros(n)
        self.weight_true = float(p.m.body_subtreemass[p.trunk]) * G
        self._loaded = REWARD_SCALES["contact_load_frac"] * self.weight_true
        c = float(clock())
        self.c_start = c
        turned = script.ref_yaw(c) if c > 0.0 else 0.0
        yaw = control._yaw(v.xmat[p.trunk])
        self.yaw_home = yaw - turned
        self.yaw_home_est = float(est()["yaw"]) - turned
        self._com0 = v.subtree_com[p.trunk][:2].copy()
        self._yaw0 = yaw
        l0, l1 = (float(v.sensordata[a]) for a in self._load_adr)
        self._contact = [l0 > self._loaded, l1 > self._loaded]
        sw, u = script.swing_progress(c)
        self._c_lift = [c - (u * script.swing_s(c) if sw == f else 0.0) for f in (0, 1)]
        self._foot_z = [float(v.xpos[f, 2]) for f in p.feet]
        self._v_down = [0.0, 0.0]
        self._servo_m = np.zeros(len(self._servo))
        self._servo_t = 0.0
        self._vb = self._vb_ref = self._vb_t = 0.0
        self._vb_key = self._vb_c0 = None
        self._hist_contact = deque(maxlen=HISTORY_STEPS)
        self._gap = (0.0, False, None)
        self._rf_sum, self._rf_min, self._rf_n = 0.0, math.inf, 0
        self._new_step()

    def _new_step(self) -> None:
        self._grf, self._flight, self._match, self._drag, self._td = 0.0, 0, 0, False, 0.0
        self._cont, self._rcont = [0, 0], [0, 0]

    def set_history(self, actions3: np.ndarray, filter_state: np.ndarray) -> None:
        """An R6 mid-task start (design D8): the three-action history (newest first) and the filter's state."""
        n = self.nj + 1
        self.hist = np.array(actions3, float).reshape(3, n)
        self.a_f = np.array(filter_state, float).reshape(n)

    def reference_actions(self, t_ref: float) -> np.ndarray:
        """[3, nj + 1]: the R6 actions that give the reference's joint targets at the three env steps before
        ``t_ref`` at the commanded clock, newest first: ``clip((q_ref - q_home) / span, -1, 1)``, clock action 0."""
        out = np.zeros((3, self.nj + 1))
        for i in range(3):
            t = max(t_ref - (i + 1) * self.dt_env * self.rate, 0.0)
            out[i, : self.nj] = np.clip((self.script.track.joint(t)[: self.nj] - self.q_home) / self.span, -1.0, 1.0)
        return out

    # ---------------------------------------------------------------- observation
    def _overlay(self):
        """(neck-yaw offset, jaw offset, the head command's four values) at the plant's time; zeros when neutral."""
        if self.head_script is None:
            return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        val = self.head_script.value(float(self.p.d.time))
        return val[0], val[1], val

    def actor_obs(self) -> np.ndarray:
        e = self.est()
        c = float(self.clock())
        s = self.script
        ph = 2.0 * math.pi * s.phase(c)
        s0, s1, _, _ = s.contact_ref(c)
        v_cmd, w_cmd = s.command(c)
        parts = [e["q"], e["qd"], e["gravity"], e["gyro"], np.asarray(e["foot_load"]) / self.weight,
                 [math.sin(ph), math.cos(ph)], [1.0 if s0 else 0.0, 1.0 if s1 else 0.0],
                 [self.rate * v_cmd / COMMAND_SCALE[0], self.rate * w_cmd / COMMAND_SCALE[1]]]
        if self.heading_input:
            err = control._wrap(float(e["yaw"]) - self.yaw_home_est - s.heading(c)) / HEADING_SCALE_RAD
            parts.append([min(max(err, -HEADING_CLIP), HEADING_CLIP)])
        parts.append(self.hist.ravel())
        if self.head_x:
            parts.append(np.array(self._overlay()[2]) / HEAD_COMMAND_SCALE_RAD)
        return np.concatenate(parts).astype(np.float32)

    def critic_obs(self, factors: dict, push, floor: str) -> np.ndarray:
        """The privileged inputs (the env only): Stage 1's (``env.py``'s observation) with the layout's additions."""
        p, v, s = self.p, self.p._v, self.script
        trunk = p.trunk
        c = float(self.clock())
        _, ref = s.track.info(c)
        R = v.xmat[trunk].reshape(3, 3)
        yaw = control._yaw(R)
        cy, sy = math.cos(yaw), math.sin(yaw)
        vel = v.qvel[:3]
        roll, pitch = control._roll_pitch(R @ self.R_home.T)
        x = s.track.joint(c)[: self.nj]
        oy, oj, _ = self._overlay()
        if oy or oj:
            x = x.copy()
            x[self.i_yaw] += oy
            x[self.i_jaw] += oj
        pushing = push is not None and push[0] <= p.d.time < push[0] + push[1]
        mass = self.weight / G
        fl = np.zeros(2)
        if floor in env.FLOOR_CODES:
            fl[env.FLOOR_CODES[floor]] = 1.0
        heading_err = control._wrap(yaw - self.yaw_home - (float(ref["yaw"]) - s._ref_yaw0))
        f = factors
        foot_err = r5lib.foot_rel_of(v.xpos, v.xmat, trunk, p.feet) - s.foot_rel(c)
        t0, t1, _, _ = s.contact_ref(max(c - self._lag, 0.0))
        return np.concatenate([
            [cy * vel[0] + sy * vel[1], -sy * vel[0] + cy * vel[1], vel[2]], [(v.qpos[2] - p.z_floor) / self.H],
            [roll, pitch], self._dcm_error(ref), v.qpos[p.qadr] - x, [heading_err],
            [f["friction"] if f.get("friction") is not None else 0.9], fl,
            [push[2] / mass if pushing else 0.0, push[3] / mass if pushing else 0.0, 1.0 if pushing else 0.0],
            [f.get("mass", 1.0), f.get("kp", 1.0), f.get("latency_steps", 0) * self.dt / DELAY_SCALE_S,
             f.get("joint_damping", 1.0), f.get("joint_friction", 1.0)],
            foot_err.ravel(), [1.0 if t0 else 0.0, 1.0 if t1 else 0.0]]).astype(np.float32)

    def _dcm_error(self, ref: dict) -> np.ndarray:
        """Stage 1's ``Stage1Env._true_dcm_error``: the true DCM against the reference's, relative to the anchor
        foot in its frame, at the commanded rate."""
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
        omega = self.script.track.omega
        xi = com[:2] + cv[:2] / omega
        rate = self.rate
        ref_dcm = ref["dcm_rel"] if rate == 1.0 else ref["dcm_rel"] + (rate - 1.0) * ref["comv_rel"] / omega
        return control._rot2(ya).T @ (xi - pa) - ref_dcm

    # ---------------------------------------------------------------- action
    def filter(self, action) -> np.ndarray:
        """The action clipped to [-1, 1] and low-passed (``a_f += alpha (a - a_f)``); the clipped action enters the
        three-action history (newest first).  Returns the filtered action."""
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.a_f = self.a_f + self.alpha * (a - self.a_f)
        self.hist = np.vstack([a[None], self.hist[:2]])
        return self.a_f

    def _clock_factor(self, a_f) -> float:
        rf = 1.0 + self.action_rate * float(a_f[self.nj])
        self._rf_sum += rf
        self._rf_min = min(self._rf_min, rf)
        self._rf_n += 1
        return rf

    def offsets(self, a_f) -> tuple[np.ndarray | None, float]:
        """R5 (rule E1): (``tk.residual``, ``tk.rate_factor``): the joint offsets ``a_f[:nj] x scale`` plus the
        head overlay, None when every entry is zero; the clock factor ``1 + 0.25 a_f[nj]``."""
        off = a_f[: self.nj] * self.scale
        if self.head_script is not None:
            oy, oj, _ = self._overlay()
            off[self.i_yaw] += oy
            off[self.i_jaw] += oj
        return (off if off.any() else None), self._clock_factor(a_f)

    def targets(self, a_f) -> tuple[np.ndarray, float]:
        """R6 (rule E2): (``q_target``, ``rate_factor``): ``clip(home + head overlay + a_f[:nj] x span, range)``."""
        q = self.q_home + a_f[: self.nj] * self.span
        if self.head_script is not None:
            oy, oj, _ = self._overlay()
            q[self.i_yaw] += oy
            q[self.i_jaw] += oj
        return np.clip(q, self.lo, self.hi), self._clock_factor(a_f)

    def clock_stats(self) -> dict:
        """The clock factor over the env steps so far and the reference time covered since the start."""
        n = self._rf_n
        return dict(steps=n, mean=self._rf_sum / n if n else 1.0, min=self._rf_min if n else 1.0,
                    reference_s=float(self.clock()) - self.c_start)

    # ---------------------------------------------------------------- per physics step
    def substep(self) -> None:
        """After every physics step: the foot loads (peak, flight, contact shares), the contact match with the
        reference, a loaded foot through the reference's mid-swing, each foot's own lift-off and touchdown speed,
        and the servo mean squares."""
        p, v = self.p, self.p._v
        sd = v.sensordata
        l0, l1 = float(sd[self._load_adr[0]]), float(sd[self._load_adr[1]])
        th = self._loaded
        now = (l0 > th, l1 > th)
        if l0 + l1 > self._grf:
            self._grf = l0 + l1
        if not (now[0] or now[1]):
            self._flight += 1
        self._cont[0] += now[0]
        self._cont[1] += now[1]
        c = float(self.clock())
        s = self.script
        r = s.contact_ref(c)
        b = s.contact_ref(c - self._lag if c > self._lag else 0.0)
        self._rcont[0] += r[0]
        self._rcont[1] += r[1]
        self._match += (now[0] == r[0] or now[0] == b[0]) + (now[1] == r[1] or now[1] == b[1])
        if (now[0] and r[2] and b[2]) or (now[1] and r[3] and b[3]):
            self._drag = True
        xpos, feet, was = v.xpos, p.feet, self._contact
        for f in (0, 1):
            z = float(xpos[feet[f], 2])
            down = (self._foot_z[f] - z) / self.dt
            if now[f] and not was[f]:
                td = min(1.0, max(0.0, self._v_down[f] - REWARD_SCALES["touchdown_free_m_s"])
                         / REWARD_SCALES["touchdown_scale_m_s"])
                if td > self._td:
                    self._td = td
            elif was[f] and not now[f]:
                self._c_lift[f] = c
            self._foot_z[f], self._v_down[f] = z, down
        self._contact = [now[0], now[1]]
        if len(self._servo):
            force = v.actuator_force[self._servo]
            self._servo_m += self._servo_a * (force * force - self._servo_m)
            self._servo_t += self.dt

    # ---------------------------------------------------------------- per env step
    def interference(self) -> tuple[float, bool]:
        """(the penalty in [0, 1] on the moving pairs, whether any pair is below zero) at the plant's pose
        (``INTERFERENCE_RULE``); (0, False) without an oracle."""
        lg = self.lg
        if lg is None:
            self._gap = (0.0, False, None)
            return 0.0, False
        d = lg.at(self.p.d.qpos)
        mov = d[lg.moving]
        g = float(mov.min()) if len(mov) else float(lg.cap)
        tight = REWARD_SCALES["interference_m"]
        pen = min(1.0, max(0.0, (tight - g) / tight))
        low = float(d.min()) if len(d) else float(lg.cap)
        self._gap = (pen, low < 0.0, low)
        return pen, low < 0.0

    def _reference_motion(self, clock: float) -> tuple[float, float]:
        """Stage 1's tracking target (``Stage1Env._command``): the reference's own forward speed and yaw rate at
        ``clock``, at the commanded rate."""
        tr = self.script.track
        ref = tr.info(clock)[1]
        h = 0.02
        y1 = float(tr.info(clock + h)[1]["yaw"])
        return self.rate * float(ref["comv_rel"][0]), self.rate * (y1 - float(ref["yaw"])) / h

    def reward(self, clock0: float) -> tuple[float, dict]:
        """The env step's reward and its terms (``REWARD_RULE``); ``clock0`` is the reference clock at the env
        step's start.  Call it once per env step, after ``interference``."""
        p, v, s, sc = self.p, self.p._v, self.script, REWARD_SCALES
        trunk, nj = p.trunk, self.nj
        c = float(self.clock())
        tr = s.track
        q = v.qpos[p.qadr]
        q_ref = tr.joint(c)[:nj]
        oy, oj, _ = self._overlay()
        if oy or oj:
            q_ref = q_ref.copy()
            q_ref[self.i_yaw] += oy
            q_ref[self.i_jaw] += oj
        t = {}
        t["joint"] = math.exp(-float(np.mean(np.square((q - q_ref) / sc["joint_rad"]))))
        com = v.subtree_com[trunk][:2]
        R = v.xmat[trunk].reshape(3, 3)
        yaw = control._yaw(R)
        dyaw = control._wrap(yaw - self._yaw0)
        step_xy = com - self._com0
        speed = float(step_xy[0] * math.cos(self._yaw0) + step_xy[1] * math.sin(self._yaw0)) / self.dt_env
        v_ref, w_ref = self._reference_motion(clock0)
        t["tracking"] = float(self.gaussian_tracking_reward(abs(speed - v_ref), abs(dyaw / self.dt_env - w_ref)))
        d_foot = r5lib.foot_rel_of(v.xpos, v.xmat, trunk, p.feet) - s.foot_rel(c)
        t["foot"] = math.exp(-float(np.mean(np.sum(d_foot * d_foot, axis=1))) / (sc["foot_frac_H"] * self.H) ** 2)
        t["contact"] = self._match / (2.0 * SUBSTEPS)
        d_head = r5lib.head_rp_of(v.xmat, self.head, self.R_head_home) - s.head_rp(c)
        t["head"] = math.exp(-float(d_head @ d_head) / math.radians(sc["head_deg"]) ** 2)
        # --- penalties
        v_cmd, _ = s.command(c)
        self._hist_contact.append((self._cont[0] / SUBSTEPS, self._cont[1] / SUBSTEPS, self._rcont[0] / SUBSTEPS,
                                   self._rcont[1] / SUBSTEPS))
        t["duty"] = 0.0
        n = max(1, int(round(s.stride_s(c) / self.rate / self.dt_env)))
        if s.walking(c) and len(self._hist_contact) >= n:
            lo, hi = sc["duty_band_slow"] if v_cmd < sc["duty_slow_below_m_s"] else sc["duty_band"]
            share = np.mean(np.array(self._hist_contact)[-n:], axis=0)
            duty, ref_duty = share[:2], share[2:]
            over = np.where((ref_duty >= lo) & (ref_duty <= hi), np.maximum(0.0, np.maximum(lo - duty, duty - hi)), 0.0)
            t["duty"] = min(1.0, float(np.sum(over)) / sc["duty_over"])
        weight = self.weight_true
        loads = p.foot_loads()
        heavy = [i for i in (0, 1) if loads[i] > sc["slip_load_frac"] * weight]
        t["slip"] = 0.0
        if heavy:
            slip = control.foot_slip_speed(p, heavy, self._foot_of)
            t["slip"] = min(1.0, max(0.0, float(slip.max()) - sc["slip_free_m_s"]) / sc["slip_scale_m_s"])
        t["flight"] = self._flight / SUBSTEPS
        t["impact"] = min(1.0, max(0.0, self._grf / weight - sc["impact_free_bw"]) / sc["impact_scale_bw"])
        clear = 1.0 if self._drag else 0.0
        if not (self._contact[0] and self._contact[1]):
            need = sc["clearance_frac_H"] * self.H
            swing = s.swing_s(c)
            low = None
            for f in (0, 1):
                if self._contact[f] or swing <= 0.0:
                    continue
                u = (c - self._c_lift[f]) / swing
                if sc["mid_swing"][0] <= u <= sc["mid_swing"][1]:
                    if low is None:
                        low = metrics.foot_lowest(p.m, p.d, *self._foot_tables) - p.z_floor
                    clear = max(clear, min(1.0, max(0.0, need - float(low[f])) / need))
        t["clearance"] = clear
        roll, pitch = control._roll_pitch(R @ self.R_home.T)
        t["tilt"] = min(1.0, max(0.0, math.degrees(max(abs(roll), abs(pitch))) - sc["tilt_free_deg"])
                        / sc["tilt_scale_deg"])
        da = self.hist[0] - self.hist[1]
        t["action_rate"] = min(1.0, float(np.mean(da * da)) / sc["action_rate"])
        t["touchdown"] = self._td
        t["servo"] = 0.0
        if self._servo_t > 0.0:
            rms = np.sqrt(self._servo_m / (1.0 - math.exp(-self._servo_t / sc["servo_tau_s"])))
            t["servo"] = min(1.0, max(0.0, float(np.max(rms / self._sustained)) - 1.0) / sc["servo_over"])
        t["interference"] = self._gap[0]
        t["progress"] = self._progress(c, clock0, step_xy, v_cmd)
        t["heading"] = 0.0
        if s.kind(c) != "hold":
            err = abs(control._wrap(yaw - self.yaw_home - s.heading(c)))
            t["heading"] = min(1.0, max(0.0, math.degrees(err) - sc["heading_free_deg"]) / sc["heading_scale_deg"])
        margin = metrics.joint_margins(q, self.lo, self.hi, self.rest)
        t["range"] = min(1.0, max(0.0, float(np.max(s.required_margin - margin))) / sc["range_rad"])
        w = self.weights
        r = sum(w[k] * t[k] for k in POSITIVE) - sum(w[k] * t[k] for k in PENALTIES)
        self._com0, self._yaw0 = com.copy(), yaw
        self._new_step()
        return float(r), t

    def _progress(self, c: float, clock0: float, step_xy, v_cmd: float) -> float:
        """Rule W5's ``progress``: the two means are restarted at the first frame of each periodic walking segment
        (and start with the episode inside one) and are read once they have run one stride of reference time.  The
        reference's sample is its own CoM speed over the reference time the step covered, at the commanded clock
        rate: a clock held slow covers less of it per step and is charged in full."""
        s = self.script
        key = s.walk_start(c)
        if key is None:
            self._vb_key = None
            return 0.0
        if key != self._vb_key:
            self._vb_key, self._vb, self._vb_ref, self._vb_t, self._vb_c0 = key, 0.0, 0.0, 0.0, clock0
        stride = s.stride_s(c)
        tau = stride / self.rate
        h = self.yaw_home + s.heading(c)
        along = float(step_xy[0] * math.cos(h) + step_xy[1] * math.sin(h)) / self.dt_env
        c_a = max(clock0, key)
        ref = self.rate * s.walk_progress(c_a, c) / (c - c_a) if c > c_a else self.rate * v_cmd
        a = 1.0 - math.exp(-self.dt_env / tau)
        self._vb += a * (along - self._vb)
        self._vb_ref += a * (ref - self._vb_ref)
        self._vb_t += self.dt_env
        if not (v_cmd > 0.0 and s.steady(c) and c - self._vb_c0 >= stride):
            return 0.0
        short = (self._vb_ref - self._vb) / (1.0 - math.exp(-self._vb_t / tau)) / (self.rate * v_cmd)
        return min(1.0, max(0.0, short) / REWARD_SCALES["progress_short"])

    def describe(self) -> dict:
        """What an episode executes of this core, for the env's description and a row's key (no measured value)."""
        return dict(layout=self.layout, layout_rule=LAYOUT_RULE, substeps=SUBSTEPS, rate=self.rate,
                    filter_hz=self.filter_hz, filter_alpha=self.alpha,
                    action=dict(leg_rad=self.leg_rad, other_rad=self.other_rad, clock=self.action_rate,
                                scale=[float(x) for x in self.scale]),
                    span=None if self.span is None else [float(x) for x in self.span],
                    span_rule=None if self.span is None else env.POLICY_ONLY_SPAN_RULE,
                    reward=dict(self.weights), reward_scales=_plain(REWARD_SCALES), reward_rule=REWARD_RULE,
                    interference_rule=INTERFERENCE_RULE, head_rule=HEAD_RULE if self.head_x else None,
                    nominal_weight_N=self.weight)


def _jid(m, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def _plain(d: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


__all__ = ["Core", "HEAD_RULE", "HeadScript", "INTERFERENCE_RULE", "LAYOUT_RULE", "PENALTIES", "POSITIVE", "R6Tracker",
           "REWARD_RULE", "REWARD_SCALES", "REWARD_WEIGHTS", "REWARD_WEIGHTS_ITEMS", "SUBSTEPS", "layout",
           "policy_span", "reset_at", "tracker_estimate"]
