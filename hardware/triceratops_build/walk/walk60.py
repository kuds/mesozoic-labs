"""Model-based trot for the 60 cm Triceratops (task 5, decision D10). Simulation only.

The plant is the training environment (``../rl/trike_env.py``), so this controller and the learned policies are
measured on the same simulated robot: the same model, 1 ms physics, 0-10 ms servo-delay FIFO, sensor noise, domain
randomisation, termination rules and gait measures (``../rl/gait_metrics.py``). The controller writes joint targets
through the env's action (``target = neutral + action x scale``), with the env's action low-pass filter off: the
filter is part of the learned policy's interface, not of the robot.

What the controller uses (all from the observation, with its noise): leg joint angles and speeds, the IMU's gravity
direction and gyro, the gyro-integrated heading estimate (with its drift), and the foot loads. It does NOT use the
simulator's body velocity (``obs_privileged_lin_vel``), which the robot cannot measure: it estimates the body's velocity
from the stance legs (leg odometry).

How it walks:
  - a trot with fixed timing: diagonal pairs (left fore + right hind, right fore + left hind) half a period apart;
  - each foot's target is planned in a level frame that turns with the body's heading: stance feet slide back at the
    commanded speed, turn against the heading error and slide sideways against the estimated sideways drift; swing
    feet lift on a smooth arc and land where the Raibert rule puts them (half the stance sweep ahead of the hip,
    plus a gain times the speed error);
  - the level-frame targets are turned into the body frame with the measured roll and pitch (so the stiff position
    servos level the body), stance legs get a further roll and pitch correction, and joint angles come from damped
    least-squares inverse kinematics on the model itself.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
RL_DIR = HERE.parent / "rl"
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

import gait_metrics  # noqa: E402
from trike_env import ANKLE_ROLL_MODEL, CORNERS, LEG_ACTUATORS, TrikeEnv, TrikeEnvConfig  # noqa: E402

# Pairs: left fore + right hind at phase 0, right fore + left hind at phase 0.5.
PHASE_OFFSET = {"left_fore": 0.0, "right_hind": 0.0, "right_fore": 0.5, "left_hind": 0.5}


@dataclass
class WalkParams:
    """Every setting of the controller (untuned starting points until a development log says otherwise)."""

    period_s: float = 0.70  # one stride: both pairs swing once (0.60 first: more foot bounce and heat)
    duty: float = 0.55  # share of the period each foot is down (> 0.5: both pairs down briefly at each switch)
    leg_drop_m: float = 0.278  # stance ankle height below the chassis origin, level frame (standing: 0.283)
    lift_m: float = 0.035  # swing clearance at mid-swing
    press_m: float = 0.003  # stance feet aimed this far below their touchdown height (contact preload)
    k_speed: float = 0.08  # s: touchdown moved forward by this x (estimated - commanded speed)
    k_side: float = 0.08  # s: touchdown moved sideways by this x estimated sideways speed
    k_roll: float = 0.6  # stance foot height change per rad of roll, x its sideways offset
    k_pitch: float = 0.6  # ... per rad of pitch, x its fore-aft offset
    k_heading: float = 1.0  # 1/s: turn rate commanded per rad of heading error
    k_drift: float = 0.4  # 1/s: sideways stance speed commanded per m of estimated sideways drift
    velocity_tau_s: float = 0.15  # low-pass of the leg-odometry velocity
    max_reach_m: float = 0.08  # touchdown limited to this far from the neutral foot position (fore-aft, sideways)
    stand_s: float = 0.5  # standing before the first step
    ramp_s: float = 1.0  # commanded speed ramps from 0 over this time after the first step
    stance_load: float = 0.08  # a foot counts as loaded above this share of body weight (odometry)
    # Frame of the foot targets: "body" keeps them fixed to the chassis, so the stiff servos hold the chassis parallel
    # to the ground under the stance feet; "level" turns them by the measured roll and pitch first (tried first: it
    # leaves roll free, since a target below the chassis in a level frame is met at any roll; the trot rocked over).
    frame: str = "body"
    # Roll and pitch for the feedback: a complementary filter (gyro integrated, pulled towards the IMU's gravity
    # reading with this time constant). The raw reading carries about 1 deg of noise per sample, which at 100 Hz
    # through the stance-height correction shook the stiff legs by 1-3 mm and made the feet chatter and slide.
    attitude_tau_s: float = 0.20
    # Landing. The swing's arc and its fore-aft travel finish at this share of the swing; the rest of the swing holds
    # the foot just at the estimated ground, so it arrives slowly. The ground under a swing foot is estimated from the
    # filtered roll and pitch (a nose-down pitch brings the ground under the front feet closer). A swing foot loaded
    # above early_load (share of body weight) in the second half of its swing has landed early and becomes a stance
    # foot where it is (tried without: front feet struck the floor late in the swing at 2.5-3 body weights, and the
    # robot left the ground for a moment).
    swing_land_frac: float = 0.90
    swing_ground_from_attitude: bool = False  # tried True: a positive feedback (nose down -> shorter front landing)
    early_touchdown: bool = False  # tried True: it cut the strides short (slower, more slip)
    early_load: float = 0.15
    land_ramp_s: float = 0.05  # a new stance foot moves from its landing height to the stance height over this time

    def replace(self, **kw) -> "WalkParams":
        return dataclasses.replace(self, **kw)


def min_jerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def plant_config(**overrides) -> TrikeEnvConfig:
    """The env as the controller's plant: the roll-ankle model, 100 Hz control, action filter off, action scales wide
    enough that no reachable target is clipped by the scale (the joint ranges still clip). Physics, delay, noise,
    randomisation and termination are the env's own."""
    base = dict(model_path=str(ANKLE_ROLL_MODEL), frame_skip=10, action_filter_hz=0.0, action_scale_roll=0.45,
                action_scale_pitch=0.70, action_scale_flexion=0.85)
    base.update(overrides)
    return TrikeEnvConfig(**base)


class Kinematics:
    """Leg kinematics on the model itself, chassis fixed at the origin, level: ankle (foot body) positions in the
    chassis frame, their Jacobians, and damped least-squares inverse kinematics."""

    def __init__(self, model_path: str):
        m = mujoco.MjModel.from_xml_path(model_path)
        self.m, self.d = m, mujoco.MjData(m)
        self.qadr = {c: np.array([m.jnt_qposadr[m.joint(f"{c}_{a}").id] for a in ("roll", "pitch", "flexion")])
                     for c in CORNERS}
        self.dof = {c: np.array([m.jnt_dofadr[m.joint(f"{c}_{a}").id] for a in ("roll", "pitch", "flexion")])
                    for c in CORNERS}
        self.lo = {c: m.jnt_range[[m.joint(f"{c}_{a}").id for a in ("roll", "pitch", "flexion")], 0] for c in CORNERS}
        self.hi = {c: m.jnt_range[[m.joint(f"{c}_{a}").id for a in ("roll", "pitch", "flexion")], 1] for c in CORNERS}
        self.foot = {c: m.body(f"{c}_foot").id for c in CORNERS}
        self.jac = np.zeros((3, m.nv))
        key = m.key("home").id
        self.d.qpos[:] = m.key_qpos[key]
        self.d.qpos[0:3], self.d.qpos[3:7] = 0.0, (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_kinematics(m, self.d)
        self.q_home = {c: self.d.qpos[self.qadr[c]].copy() for c in CORNERS}
        self.p_home = {c: self.d.xpos[self.foot[c]].copy() for c in CORNERS}

    def set_q(self, q: dict) -> None:
        for c in CORNERS:
            self.d.qpos[self.qadr[c]] = q[c]
        mujoco.mj_kinematics(self.m, self.d)

    def foot_and_jacobian(self, q: dict) -> tuple[dict, dict]:
        self.set_q(q)
        mujoco.mj_comPos(self.m, self.d)
        p, J = {}, {}
        for c in CORNERS:
            p[c] = self.d.xpos[self.foot[c]].copy()
            mujoco.mj_jacBody(self.m, self.d, self.jac, None, self.foot[c])
            J[c] = self.jac[:, self.dof[c]].copy()
        return p, J

    def ik(self, c: str, target, q0, iters: int = 8, tol: float = 2e-5) -> tuple[np.ndarray, float]:
        """Joint angles (roll, pitch, flexion) putting corner c's ankle at ``target`` (chassis frame); also the
        remaining error, m."""
        q = np.array(q0, dtype=np.float64)
        adr, dof = self.qadr[c], self.dof[c]
        err = np.inf
        for _ in range(iters):
            self.d.qpos[adr] = q
            mujoco.mj_kinematics(self.m, self.d)
            e = np.asarray(target) - self.d.xpos[self.foot[c]]
            err = float(np.linalg.norm(e))
            if err < tol:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacBody(self.m, self.d, self.jac, None, self.foot[c])
            J = self.jac[:, dof]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(3), e), self.lo[c], self.hi[c])
        return q, err


def level_rotation(g_body) -> tuple[np.ndarray, float, float]:
    """Rotation from the chassis frame to the level frame (roll and pitch only) from the measured gravity direction in
    the chassis frame; also roll and pitch, rad (roll + = left side up, pitch + = nose down)."""
    g = np.asarray(g_body, dtype=np.float64)
    g = g / max(np.linalg.norm(g), 1e-9)
    pitch = math.asin(max(-1.0, min(1.0, g[0])))
    roll = math.atan2(-g[1], -g[2])
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx, roll, pitch


class TrotController:
    """``act(obs) -> action`` for a ``TrikeEnv`` built with :func:`plant_config`. Resets itself at each episode's
    first step (the env's step count is 0)."""

    def __init__(self, env: TrikeEnv, P: WalkParams | None = None, *, phase_offsets=None, support_shift=None):
        self.env, self.P = env, P or WalkParams()
        self.phase_offsets = PHASE_OFFSET if phase_offsets is None else phase_offsets
        self.support_shift = support_shift
        self.kin = Kinematics(env.cfg.model_path)
        self.dt = env.dt
        self.scale = env._action_scale.copy()
        self.ctrl_neutral = env._ctrl_neutral.copy()
        self.q_neutral = env._q_neutral.copy()
        self.idx = {c: np.array([LEG_ACTUATORS.index(f"{c}_{a}") for a in ("roll", "pitch", "flexion")])
                    for c in CORNERS}
        self.p0 = {c: self.kin.p_home[c].copy() for c in CORNERS}
        self.log: list[dict] = []
        self.reset()

    def reset(self) -> None:
        P = self.P
        self.t = 0.0
        self.v_est = np.zeros(2)
        self.drift = 0.0
        self.q_guess = {c: self.kin.q_home[c].copy() for c in CORNERS}
        z = -P.leg_drop_m
        self.p = {c: np.array([self.p0[c][0], self.p0[c][1], z]) for c in CORNERS}  # level frame, per foot
        self.p_lift = {c: self.p[c].copy() for c in CORNERS}
        self.in_swing = {c: False for c in CORNERS}
        self.landed = {c: False for c in CORNERS}  # landed early, before the clock's stance
        self.z_down = {c: z for c in CORNERS}  # height at touchdown
        self.t_down = {c: 1e9 for c in CORNERS}  # time since touchdown
        self.early_touchdowns = 0
        self.ik_err_max = 0.0
        self.att = None  # filtered (roll, pitch), rad
        self.log = []

    # -------------------------------------------------------------------------------------------- one control step
    def act(self, obs) -> np.ndarray:
        env, P = self.env, self.P
        if env._step_count == 0:
            self.reset()
        s = env.obs_slots(obs)
        q_meas = np.asarray(s["leg_joint_pos"], dtype=np.float64) + self.q_neutral
        qd_meas = np.asarray(s["leg_joint_vel"], dtype=np.float64)
        R_lb, roll_m, pitch_m = level_rotation(s["gravity_body"])
        sc = s["heading_error_sin_cos_ESTIMATE"]
        heading_err = math.atan2(float(sc[0]), float(sc[1]))
        gyro = np.asarray(s["gyro"], dtype=np.float64)
        loads = np.asarray(s["foot_load_over_weight"], dtype=np.float64)
        command = float(np.asarray(s["command_speed_mps"])[0])
        if self.att is None:
            self.att = np.array([roll_m, pitch_m])
        else:
            a_att = self.dt / (P.attitude_tau_s + self.dt) if P.attitude_tau_s > 0 else 1.0
            pred = self.att + gyro[:2] * self.dt  # roll rate ~ gyro x, pitch rate ~ gyro y (small angles)
            self.att = pred + a_att * (np.array([roll_m, pitch_m]) - pred)
        roll, pitch = float(self.att[0]), float(self.att[1])

        # --- leg odometry: body velocity from the loaded stance legs (chassis frame), then level frame
        qd = {c: q_meas[self.idx[c]] for c in CORNERS}
        feet_b, J = self.kin.foot_and_jacobian(qd)
        vs = []
        for k, c in enumerate(CORNERS):
            if not self.in_swing[c] and loads[k] > P.stance_load:
                v_foot_rel = J[c] @ qd_meas[self.idx[c]]
                vs.append(-(v_foot_rel + np.cross(gyro, feet_b[c])))
        if vs:
            v_level = R_lb @ np.mean(vs, axis=0)
            a = 1.0 - math.exp(-self.dt / P.velocity_tau_s)
            self.v_est += a * (v_level[:2] - self.v_est)
        self.drift += self.v_est[1] * self.dt

        # --- commands for this step
        t_walk = self.t - P.stand_s
        # A steering environment supplies its bounded body-frame yaw rate in the shared command slot.
        # A zero-forward turn still needs the alternating support clock and ordinary actuator contacts.
        yaw_request = (float(s["command"][2]) * env.cfg.cmd_yaw_rate_scale) if "command" in s else None
        walking = t_walk >= 0.0 and (command > 0.0 or (yaw_request is not None and yaw_request != 0.0))
        v_cmd = command * min(1.0, max(0.0, t_walk / P.ramp_s)) if walking else 0.0
        v_side = -P.k_drift * self.drift if walking else 0.0
        turn = (-P.k_heading * heading_err if yaw_request is None else yaw_request) if walking else 0.0
        phase = (t_walk / P.period_s) % 1.0 if walking else None
        T_st = P.duty * P.period_s

        # --- foot targets, level frame
        z_stance = -P.leg_drop_m - P.press_m
        for k, c in enumerate(CORNERS):
            p = self.p[c]
            clock_stance, w = True, 0.0
            if phase is not None:
                u = (phase - self.phase_offsets[c]) % 1.0
                clock_stance = u < P.duty
                w = (u - P.duty) / (1.0 - P.duty)
            if clock_stance:
                self.landed[c] = False  # the clock has caught up with an early landing
            if clock_stance or self.landed[c]:
                if self.in_swing[c]:  # touchdown by the clock: the stance starts where the swing aimed
                    self.in_swing[c] = False
                    self.z_down[c], self.t_down[c] = p[2], 0.0
                # slide back at the commanded speed, sideways against the drift, and turn against the heading error
                p[0] -= v_cmd * self.dt
                p[1] -= v_side * self.dt
                c_t, s_t = math.cos(-turn * self.dt), math.sin(-turn * self.dt)
                p[0], p[1] = c_t * p[0] - s_t * p[1], s_t * p[0] + c_t * p[1]
                ramp = min(1.0, self.t_down[c] / P.land_ramp_s) if P.land_ramp_s > 0 else 1.0
                p[2] = self.z_down[c] + (z_stance - self.z_down[c]) * ramp
                self.t_down[c] += self.dt
            else:
                if not self.in_swing[c]:  # lift-off
                    self.in_swing[c] = True
                    self.p_lift[c] = p.copy()
                x0, y0 = self.p0[c][0], self.p0[c][1]
                tx = x0 + 0.5 * self.v_est[0] * T_st + P.k_speed * (self.v_est[0] - v_cmd)
                ty = y0 + 0.5 * self.v_est[1] * T_st + P.k_side * self.v_est[1]
                tx = min(max(tx, x0 - P.max_reach_m), x0 + P.max_reach_m)
                ty = min(max(ty, y0 - P.max_reach_m), y0 + P.max_reach_m)
                w2 = min(1.0, w / P.swing_land_frac)
                f = min_jerk(w2)
                ground = -P.leg_drop_m  # the floor under the foot, chassis frame
                if P.swing_ground_from_attitude:
                    ground += pitch * p[0] - roll * p[1]
                p[0] = self.p_lift[c][0] + (tx - self.p_lift[c][0]) * f
                p[1] = self.p_lift[c][1] + (ty - self.p_lift[c][1]) * f
                p[2] = ground + P.lift_m * 0.5 * (1.0 - math.cos(2.0 * math.pi * w2))
                if P.early_touchdown and w > 0.5 and loads[k] > P.early_load:  # landed early: stance from here
                    self.landed[c], self.in_swing[c] = True, False
                    self.z_down[c], self.t_down[c] = p[2], 0.0
                    self.early_touchdowns += 1

        # --- level frame to chassis frame, attitude correction on stance legs, inverse kinematics
        action = np.zeros(12)
        R_bl = R_lb.T if P.frame == "level" else np.eye(3)
        shift = self.support_shift(phase, t_walk) if self.support_shift is not None and phase is not None else None
        for c in CORNERS:
            p = self.p[c].copy()
            if shift is not None:
                p[:2] -= shift
            if not self.in_swing[c]:
                p[2] += P.k_roll * roll * p[1] - P.k_pitch * pitch * p[0]
            target = R_bl @ p
            q, err = self.kin.ik(c, target, self.q_guess[c])
            self.q_guess[c] = q
            self.ik_err_max = max(self.ik_err_max, err)
            i = self.idx[c]
            action[i] = (q - self.ctrl_neutral[i]) / self.scale[i]
        self.t += self.dt
        return np.clip(action, -1.0, 1.0)


# ------------------------------------------------------------------------------------------------ episodes
def run_episode(P: WalkParams, *, seed: int, command: float = 0.25, delay_steps: int = 10,
                impratio: float | None = None, **plant) -> dict:
    """One episode on the plant; returns gait_metrics' measures plus the episode's facts. ``impratio``: the stiffer
    contact setting (the model's own is 1), set on the model before the episode."""
    env = TrikeEnv(plant_config(**plant))
    if impratio is not None:
        env.model.opt.impratio = float(impratio)
    ctrl = TrotController(env, P)
    log = gait_metrics.record_episode(env, ctrl.act, seed=seed, command=command, delay_steps=delay_steps)
    ms = gait_metrics.measures(log, command)
    return {"seed": seed, "command_mps": command, "delay_steps": delay_steps, "fell": bool(log["fell"]),
            "termination": log["termination"], "length_s": float(log["time_s"]),
            "speed_mps": float(log["episode_forward_speed"]), "ik_err_max_m": ctrl.ik_err_max,
            "friction": env.factors.get("friction") if env.factors else None, "measures": ms}


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--command", type=float, default=0.25)
    ap.add_argument("--delay", type=int, default=10)
    ap.add_argument("--friction", type=float, default=None, help="fixed floor friction (randomisation otherwise on)")
    ap.add_argument("--nominal", action="store_true", help="no randomisation, no observation noise")
    ap.add_argument("--impratio", type=float, default=None, help="stiffer contact setting, e.g. 10")
    ap.add_argument("--set", action="append", default=[], help="WalkParams field=value")
    args = ap.parse_args(argv)
    P = WalkParams()
    for kv in args.set:
        k, v = kv.split("=", 1)
        P = P.replace(**{k: type(getattr(P, k))(v)})
    plant = {}
    if args.nominal:
        plant.update(domain_randomization=False, observation_noise=False)
    if args.friction is not None:
        plant.update(friction_range=(args.friction, args.friction))
    for seed in (int(x) for x in args.seeds.split(",")):
        r = run_episode(P, seed=seed, command=args.command, delay_steps=args.delay, impratio=args.impratio, **plant)
        print(json.dumps({k: v for k, v in r.items() if k != "measures"}))
        print("   ", gait_metrics.one_line(gait_metrics.summarise([r["measures"]])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
