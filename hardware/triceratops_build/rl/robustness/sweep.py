"""Robustness sweep of a trained 60 cm Triceratops walking policy. Simulation only; one policy is a report, not a verdict.

One factor at a time, every other factor at its nominal value (the model as built, no domain randomisation, servo
delay 10 ms), observation noise and heading drift ON, deterministic actions, 20 s episodes, the same N seeds at every
point (seeds 200000, 200001, ...). Sweeps (README.md has the definitions; [..] = beyond the requested set):

  delay     servo delay 0, 2, 5, 10, 15, 20 [30, 40, 60] ms, at the 0.25 and the 0.15 m/s command
  friction  floor and sole sliding friction [0.15, 0.2] 0.3 ... 1.2 (and the model's own, 0.85 in contact)
  mass      every body's mass and inertia x 0.8 ... 1.2 [1.4, 1.6]; a payload of +0.5 / +1.0 [+2, +3] kg on the chassis
  drive     the 12 leg drives' force limits x [0.4, 0.55] 0.7 / 0.85 / 1.0; their kp x [0.4, 0.6] 0.8 / 1.0 / 1.2 [1.5]
  push      a sideways push on the chassis at t = 8 s for 0.1 s, impulse = robot mass x dv, dv 0.1 ... 0.5
            [0.75 ... 1.5] m/s, both ways
  terrain   a 1 / 2 [3] cm step up and a 3 / 5 [8] deg ramp up, 0.6 m ahead of the chassis (added in memory)

Usage:
  python sweep.py --ckpt-dir DIR --run pipe1 [--seeds 8] [--workers 2] [--out results/pipe1]
  python sweep.py --from-json results/pipe1/sweep_pipe1.json.gz   (summaries, Markdown, charts again; no simulation)
DIR holds the checkpoint zip, its VecNormalize pickle and run_record.json. The environment is imported through
``analysis/gait_pipe1.py``: rl/trike_env.py, or the copy of rl/ that TRIKE_ENV_DIR names (an export of the run's commit).
Before the sweep, gait_pipe1's target case (randomisation on, delay 10 ms, seeds 100000-100004, 0.25 m/s) is re-run and
compared with the analysis's speeds when its JSON is present (``--no-repro`` skips it).

Nothing stored in the checkpoint is executed: the weights load with torch.load(weights_only=True) and the
normalisation statistics through gait_pipe1's restricted unpickler (``build_policy``, ``load_obs_stats``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

HERE = Path(__file__).resolve().parent
RL = HERE.parent
for _p in (RL / "analysis", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import gait_pipe1 as G  # noqa: E402  (puts the env directory on sys.path and imports trike_env from it)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

trike_env = G.trike_env
TrikeEnv, TrikeEnvConfig = G.TrikeEnv, G.TrikeEnvConfig

# ---------------------------------------------------------------- the sweep's settings
SEED0 = 200_000
NOMINAL_DELAY_MS = 10
COMMAND = 0.25
PUSH_T, PUSH_S = 8.0, 0.1  # onset (s) and duration (s) of the push
PUSH_LOOK_S = 3.0  # lateral displacement: the largest within this time after the onset
RECOVER_VLAT, RECOVER_WIN_S, RECOVER_HOLD_S = 0.05, 0.5, 1.0  # recovery: |0.5 s-mean lateral speed| <= 0.05 for 1 s
TERRAIN_EDGE_M = 0.6  # the step's edge / the ramp's foot, ahead of the chassis start (front soles at 0.26-0.30 m)
DROP = 0.20  # a speed drop of more than this fraction of the nominal point's speed counts as a break
REPRO_SEEDS = G.SEEDS  # gait_pipe1's target case
TRAINING = {"delay_ms": (0, 10), "friction": (0.6, 1.0), "mass_scale": (0.9, 1.1), "kp_scale": (0.9, 1.1),
            "force_scale": (0.85, 1.0)}

DEFAULT_POINT = {"command": COMMAND, "delay_ms": NOMINAL_DELAY_MS, "friction": None, "mass_scale": 1.0,
                 "payload_kg": 0.0, "force_scale": 1.0, "kp_scale": 1.0, "push_dv": 0.0, "terrain": None,
                 "randomise": False}
POINT_KEYS = tuple(DEFAULT_POINT)

# (sweep, family, x key, x values, nominal x, fixed settings, x values beyond the requested set). x None in friction
# = the model's own friction; in terrain = the flat floor. The extra points were added after a one-seed trial found no
# fall or 20 % speed drop anywhere in the requested ranges, to find where the policy does break.
_DELAYS = (0, 2, 5, 10, 15, 20, 30, 40, 60)
_PUSHES = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5)
FAMILIES = (
    ("delay", "delay, 0.25 m/s", "delay_ms", _DELAYS, 10, {"command": 0.25}, (30, 40, 60)),
    ("delay", "delay, 0.15 m/s", "delay_ms", _DELAYS, 10, {"command": 0.15}, (30, 40, 60)),
    ("friction", "friction", "friction", (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, None, 1.0, 1.2), None, {}, (0.15, 0.2)),
    ("mass", "mass, every body", "mass_scale", (0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6), 1.0, {}, (1.4, 1.6)),
    ("mass", "payload on the chassis", "payload_kg", (0.0, 0.5, 1.0, 2.0, 3.0), 0.0, {}, (2.0, 3.0)),
    ("drive", "force limit", "force_scale", (0.4, 0.55, 0.7, 0.85, 1.0), 1.0, {}, (0.4, 0.55)),
    ("drive", "position gain kp", "kp_scale", (0.4, 0.6, 0.8, 1.0, 1.2, 1.5), 1.0, {}, (0.4, 0.6, 1.5)),
    ("push", "sideways push", "push_dv", _PUSHES, 0.0, {}, (-1.5, -1.25, -1.0, -0.75, 0.75, 1.0, 1.25, 1.5)),
    ("terrain", "step up", "terrain", (None, ("step", 0.01), ("step", 0.02), ("step", 0.03)), None, {},
     (("step", 0.03),)),
    ("terrain", "ramp up", "terrain", (None, ("ramp", 3.0), ("ramp", 5.0), ("ramp", 8.0)), None, {},
     (("ramp", 8.0),)),
)


def make_point(**kw) -> dict:
    bad = set(kw) - set(POINT_KEYS)
    if bad:
        raise KeyError(f"unknown point settings {bad}")
    p = dict(DEFAULT_POINT)
    p.update(kw)
    if p["terrain"] is not None:
        p["terrain"] = tuple(p["terrain"])
    return p


def point_key(p: dict) -> str:
    return json.dumps([p[k] for k in POINT_KEYS])


# ---------------------------------------------------------------- applying a point to the model
def apply_point(env, p: dict) -> None:
    """Apply a point's physical factors to ``env.model``, which must hold the nominal values (``_apply_factors({})``
    has just restored them). Same arrays and rules as the env's domain randomisation, with one value for all:
    friction on the floor and the four soles (the env's ``geom_friction[:, 0]``), every body's mass and inertia x
    ``mass_scale`` (the world body excluded), and, on the 12 leg drives only, kp (``gainprm[0]``, ``biasprm[1]``)
    and the force limits (``forcerange``). The payload adds ``payload_kg`` to the chassis at its centre of mass with
    the chassis inertia scaled by the same mass ratio (a payload spread like the chassis). Then ``mj_setConst``."""
    m = env.model
    if p["friction"] is not None:
        m.geom_friction[env._floor, 0] = p["friction"]
        m.geom_friction[env._soles, 0] = p["friction"]
        if getattr(env, "_terrain_geom", None) is not None:
            m.geom_friction[env._terrain_geom, 0] = p["friction"]
    s = float(p["mass_scale"])
    if s != 1.0:
        m.body_mass[1:] *= s
        m.body_inertia[1:] *= s
    if p["payload_kg"]:
        c = env._chassis
        ratio = (m.body_mass[c] + p["payload_kg"]) / m.body_mass[c]
        m.body_mass[c] *= ratio
        m.body_inertia[c] *= ratio
    leg = np.asarray(env._leg_act, dtype=np.int64)
    if p["kp_scale"] != 1.0:
        k = np.where(env._pos_servo[leg], float(p["kp_scale"]), 1.0)
        m.actuator_gainprm[leg, 0] *= k
        m.actuator_biasprm[leg, 1] *= k
    if p["force_scale"] != 1.0:
        m.actuator_forcerange[leg] *= float(p["force_scale"])
    mujoco.mj_setConst(m, env.data)


class PointEnv(TrikeEnv):
    """The env with a sweep point applied at every reset, after the env restores the nominal model (its
    ``_apply_factors`` always starts from the saved nominal copies, so nothing carries over between points)."""

    point: dict | None = None
    _terrain_geom: int | None = None

    def _apply_factors(self, f):
        super()._apply_factors(f)
        if self.point is not None:
            apply_point(self, self.point)


# ---------------------------------------------------------------- terrain, added in memory
def terrain_xml(model_path: str, terrain) -> str:
    """The model's MJCF with one static box added to the world: ("step", h) a step up of h metres whose edge is
    TERRAIN_EDGE_M ahead of the chassis start, ("ramp", deg) a slope up starting there. The box copies the floor's
    contact settings. Built with MjSpec; nothing on disk is changed."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    ref = mujoco.MjModel.from_xml_path(str(model_path))
    fl = ref.geom("floor").id
    kind, val = terrain
    g = spec.worldbody.add_geom()
    g.name = "terrain"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    x0, half_w = TERRAIN_EDGE_M, 3.0
    if kind == "step":
        h = float(val)
        g.size = [10.0, half_w, h / 2]
        g.pos = [x0 + 10.0, 0.0, h / 2]
    elif kind == "ramp":
        th, L, T = math.radians(float(val)), 10.0, 0.05
        g.size = [L, half_w, T]
        # rotation -th about y: local +x points up the slope; the top face's near edge sits at (x0, 0)
        g.quat = [math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0]
        g.pos = [x0 + L * math.cos(th) + T * math.sin(th), 0.0, L * math.sin(th) - T * math.cos(th)]
    else:
        raise ValueError(f"unknown terrain {terrain}")
    g.contype, g.conaffinity, g.group = int(ref.geom_contype[fl]), int(ref.geom_conaffinity[fl]), 0
    g.condim, g.priority = int(ref.geom_condim[fl]), int(ref.geom_priority[fl])
    g.friction = ref.geom_friction[fl].tolist()
    g.solref, g.solimp = ref.geom_solref[fl].tolist(), ref.geom_solimp[fl].tolist()
    g.solmix, g.margin, g.gap = float(ref.geom_solmix[fl]), float(ref.geom_margin[fl]), float(ref.geom_gap[fl])
    g.material = "floor_mat"
    return spec.to_xml()


def build_env(env_config: dict, terrain=None, randomise=False) -> PointEnv:
    """The run's env (its run record's config) on the nominal model, randomisation off unless ``randomise``,
    observation noise as configured (on for pipe1)."""
    cfg = G.flat_floor(TrikeEnvConfig.from_dict({**env_config,
                                                "model_path": str(G.GEN / Path(env_config["model_path"]).name)}))
    if not randomise:
        cfg = cfg.replace(domain_randomization=False)
    if terrain is None:
        return PointEnv(cfg)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "terrain.xml"
        path.write_text(terrain_xml(cfg.model_path, terrain))
        env = PointEnv(cfg.replace(model_path=str(path)))
    env._terrain_geom = env.model.geom("terrain").id
    return env


# ---------------------------------------------------------------- the push
def push_force(total_mass: float, dv: float, duration: float, direction) -> np.ndarray:
    """Constant force (N, world frame) whose impulse over ``duration`` is total_mass x dv along ``direction``."""
    u = np.asarray(direction, dtype=np.float64)
    return (total_mass * dv / duration) * u / np.linalg.norm(u)


def lateral_axis(xmat) -> np.ndarray:
    """The body's left (+y) axis projected on the floor, unit length."""
    y = np.asarray(xmat, dtype=np.float64).reshape(3, 3)[:, 1].copy()
    y[2] = 0.0
    return y / np.linalg.norm(y)


class Pusher:
    """Sets ``data.xfrc_applied`` on one body for the physics steps that START in [t0, t0 + duration): call
    ``after_step`` after every mj_step (it prepares the next step). Impulse = total_mass x dv along the body's left
    axis at the onset (dv < 0: to the right). With dv = 0 it applies nothing but still records the onset and axis,
    so the no-push episodes give the baseline of the push measures."""

    def __init__(self, model, data, body: int, dv: float, t0: float = PUSH_T, duration: float = PUSH_S,
                 total_mass: float | None = None):
        self.m, self.d, self.body, self.dv, self.t0, self.dur = model, data, body, float(dv), t0, duration
        self.mass = float(model.body_mass.sum()) if total_mass is None else float(total_mass)
        self.half = 0.5 * float(model.opt.timestep)
        self.direction = None
        self.onset = None
        self.steps = 0

    def after_step(self) -> None:
        t = self.d.time
        if self.t0 - self.half <= t < self.t0 + self.dur - self.half:
            if self.direction is None:
                self.direction = lateral_axis(self.d.xmat[self.body])
                self.onset = t
                self.force = push_force(self.mass, self.dv, self.dur, self.direction)
            if self.dv != 0.0:
                self.d.xfrc_applied[self.body, :3] = self.force
                self.steps += 1
        elif self.steps:
            self.d.xfrc_applied[self.body, :3] = 0.0


# ---------------------------------------------------------------- one episode
def run_episode(env: PointEnv, policy, stats, seed: int, p: dict) -> dict:
    """One deterministic episode at point ``p``; logs like gait_pipe1.run_episode, then gait_pipe1.measures."""
    import torch

    mean, var, clip, eps = stats
    env.point = p
    obs, _ = env.reset(seed=seed, options={"command_speed": p["command"], "delay_steps": p["delay_ms"]})
    m, d = env.model, env.data
    d.xfrc_applied[:] = 0.0
    idx = np.asarray(env._leg_act, dtype=np.int64)
    names = [m.actuator(int(i)).name for i in idx]
    log = {k: [] for k in ("t", "tau", "omega", "load", "sole_v", "pos", "rp", "yaw", "com")}
    vel6 = np.zeros(6)
    lim_ep = m.actuator_forcerange[idx, 1].copy()  # this point's limits
    pusher = Pusher(m, d, env._chassis, p["push_dv"])
    tg = env._terrain_geom
    hit = {"t": None, "geom": ""}

    def hook(e):
        log["t"].append(d.time)
        log["tau"].append(d.actuator_force[idx].copy())
        log["omega"].append(d.actuator_velocity[idx].copy())
        log["load"].append(d.sensordata[e._touch_adr].copy())
        sv = []
        for g in e._soles:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, int(g), vel6, 0)
            sv.append(math.hypot(vel6[3], vel6[4]))
        log["sole_v"].append(sv)
        log["pos"].append(d.xpos[e._chassis].copy())
        R = d.xmat[e._chassis].reshape(3, 3)
        log["rp"].append((math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(math.asin(-max(-1, min(1, R[2, 0]))))))
        log["yaw"].append(math.atan2(R[1, 0], R[0, 0]))
        log["com"].append(d.subtree_com[e._chassis].copy())  # the whole robot's centre of mass (chassis = root)
        pusher.after_step()
        if tg is not None and hit["t"] is None and d.ncon:
            # the env's fall rule applied to the added box: any geom but a sole touching it
            gg = d.contact.geom[:d.ncon]
            bad = ((gg[:, 0] == tg) & ~e._is_sole[gg[:, 1]]) | ((gg[:, 1] == tg) & ~e._is_sole[gg[:, 0]])
            if bad.any():
                g1, g2 = (int(x) for x in gg[int(np.argmax(bad))])
                hit["t"], hit["geom"] = d.time, m.geom(g2 if g1 == tg else g1).name

    env.substep_hook = hook
    fell, term = False, ""
    try:
        while True:
            o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
            with torch.no_grad():
                act, _, _ = policy(torch.as_tensor(o[None]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(act.numpy()[0])
            if hit["t"] is not None and not terminated:
                terminated, info = True, {**info, "termination": f"terrain_contact:{hit['geom']}"}
            if terminated or truncated:
                fell, term = bool(terminated), info["termination"]
                break
    finally:
        env.substep_hook = None
        d.xfrc_applied[:] = 0.0
    arr = {k: np.asarray(v) for k, v in log.items()}
    arr.update(names=names, limit_ep=lim_ep, heading0=env._heading, mass=float(m.body_mass.sum()),
               limit_nominal=np.array([G.LIMIT[G.group_of(nm)] for nm in names]), fell=fell, termination=term,
               contact_n=env.cfg.contact_threshold_n)
    return episode_record(arr, p, seed, pusher)


def _r(x, n=5):
    return None if x is None else float(f"{float(x):.{n}g}")


def episode_record(arr: dict, p: dict, seed: int, pusher: Pusher | None) -> dict:
    t = arr["t"]
    rec = {"seed": seed, "fell": bool(arr["fell"]), "termination": arr["termination"], "t_end_s": _r(t[-1]),
           "mass_kg": _r(arr["mass"])}
    if (t >= G.WINDOW_S).sum() >= 100:  # gait_pipe1's window: t >= 2 s
        ms = G.measures(arr, p["command"])
        rec.update(forward_speed_mps=float(ms["forward_speed_mps"]), sideways_drift_m=_r(ms["sideways_drift_m"]),
                   heading_change_deg=_r(ms["heading_change_deg"]), power_w=_r(ms["power_w"]),
                   heat=[_r(ms["per_joint"][nm]["rms_over_thermal"], 4) for nm in arr["names"]],
                   at_nominal_limit=[_r(ms["per_joint"][nm]["sat_nominal"], 4) for nm in arr["names"]],
                   at_point_limit=[_r(ms["per_joint"][nm]["sat_episode_limit"], 4) for nm in arr["names"]],
                   slip=[_r(ms["stance_slip"][c]["slip_ratio"], 4) for c in G.CORNERS])
    rec["push"] = push_measures(arr, pusher)
    return rec


def push_measures(arr: dict, pusher: Pusher) -> dict:
    """After the push: the largest sideways displacement of the chassis along the push (within PUSH_LOOK_S of the
    onset, from its position at the onset; ``lateral_extra_*``, added in main, subtract the same seed's unpushed
    episode), the recovery time (first time from which the 0.5 s-centred mean lateral speed of the chassis stays within
    RECOVER_VLAT for RECOVER_HOLD_S; None if it does not before the episode ends or it fell), the largest body roll,
    and (paired in main) the velocity change the push actually gave the robot's centre of mass along the push."""
    out = {"applied_steps": pusher.steps, "impulse_ns": _r(pusher.mass * abs(pusher.dv)),
           "force_n": _r(pusher.mass * abs(pusher.dv) / pusher.dur)}
    if pusher.onset is None:
        return out
    t, pos = arr["t"], arr["pos"]
    sgn = math.copysign(1.0, pusher.dv)
    u = pusher.direction * sgn  # along the push
    i0 = int(np.searchsorted(t, pusher.onset - pusher.half))
    lat = (pos[:, :2] - pos[i0, :2]) @ u[:2]
    look = (t >= t[i0]) & (t <= t[i0] + PUSH_LOOK_S)
    out["lateral_peak_m"] = _r(lat[look].max())
    out["lateral_at_3s_m"] = _r(lat[look][-1])
    out["roll_peak_deg"] = _r(np.abs(arr["rp"][t >= t[i0], 0]).max(), 4)
    dt = float(np.median(np.diff(t)))
    n10, i1 = int(round(0.01 / dt)), i0 + int(round(pusher.dur / dt))
    c = (arr["com"][:, :2] - arr["com"][i0, :2]) @ pusher.direction[:2]
    if i1 + 1 + n10 < len(t):  # centre-of-mass speed along the onset's left axis over the 10 ms after the push
        out["_com_v_left_after"] = float((c[i1 + 1 + n10] - c[i1 + 1]) / (n10 * dt))
    left = (pos[:, :2] - pos[i0, :2]) @ pusher.direction[:2]  # along the onset's left axis, for the pairing in main
    out["_lat_left"] = left[i0:i0 + int(round(PUSH_LOOK_S / dt)) + 1:10].tolist()
    k = int(round(RECOVER_WIN_S / 2 / dt))
    # 0.5 s-centred mean lateral speed at sample i: (lat[i + k] - lat[i - k]) / (2 k dt)
    v = np.full(len(t), np.nan)
    v[k:len(t) - k] = (lat[2 * k:] - lat[:-2 * k]) / (2 * k * dt)
    bad = ~(np.abs(v) <= RECOVER_VLAT)  # nan (the last k samples) counts as not recovered
    hold = int(round(RECOVER_HOLD_S / dt))
    rec = None
    if not arr["fell"]:
        cs = np.concatenate([[0], np.cumsum(bad)])
        cand = np.arange(i0 + int(round(pusher.dur / dt)), len(t) - hold)
        good = (cs[cand + hold] - cs[cand]) == 0
        if good.any():
            rec = float(t[cand[int(np.argmax(good))]] - t[i0])
    out["recovery_s"] = _r(rec, 3)
    return out


# ---------------------------------------------------------------- workers
_W: dict = {}


def _init_worker(ckpt: dict, env_config: dict, net, log_std_init):
    import torch

    torch.set_num_threads(1)
    _W["env_config"] = env_config
    _W["envs"] = {}
    base = env_for(None, False)
    _W["stats"] = G.load_obs_stats(Path(ckpt["pkl"]))
    _W["policy"] = G.build_policy(base, Path(ckpt["zip"]), net, log_std_init)


def env_for(terrain, randomise):
    key = (json.dumps(terrain), randomise)
    if key not in _W["envs"]:
        _W["envs"][key] = build_env(_W["env_config"], terrain, randomise)
    return _W["envs"][key]


def _task(args):
    key, p, seed = args
    env = env_for(p["terrain"], p["randomise"])
    t0 = time.time()
    rec = run_episode(env, _W["policy"], _W["stats"], seed, p)
    rec["wall_s"] = round(time.time() - t0, 2)
    return key, rec


# ---------------------------------------------------------------- summaries and breaks
def summarise_point(p: dict, eps: list, names: list) -> dict:
    ok = [e for e in eps if not e["fell"] and "forward_speed_mps" in e]
    s = {"episodes": len(eps), "falls": sum(e["fell"] for e in eps),
         "fall_terminations": sorted({e["termination"] for e in eps if e["fell"]}),
         "fall_times_s": [e["t_end_s"] for e in eps if e["fell"]], "completed": len(ok),
         "mass_kg": eps[0]["mass_kg"]}
    s["fall_rate"] = s["falls"] / len(eps)
    if ok:
        v = np.array([e["forward_speed_mps"] for e in ok])
        dr = np.abs([e["sideways_drift_m"] for e in ok])
        heat = np.array([e["heat"] for e in ok])
        satn = np.array([e["at_nominal_limit"] for e in ok])
        satp = np.array([e["at_point_limit"] for e in ok])
        hm, nm_, pm = heat.mean(0), satn.mean(0), satp.mean(0)
        s.update(speed_mps=float(v.mean()), speed_min_mps=float(v.min()), speed_max_mps=float(v.max()),
                 speed_error_mps=float(v.mean() - p["command"]),
                 drift_abs_mean_m=float(dr.mean()), drift_abs_worst_m=float(dr.max()),
                 heading_change_abs_worst_deg=float(np.abs([e["heading_change_deg"] for e in ok]).max()),
                 power_w=float(np.mean([e["power_w"] for e in ok])),
                 heat_worst_joint=names[int(hm.argmax())], heat_worst=float(hm.max()), heat_worst_episode=float(heat.max()),
                 heat_mean_by_joint=dict(zip(names, map(float, hm))),
                 at_nominal_limit_worst_joint=names[int(nm_.argmax())], at_nominal_limit_worst=float(nm_.max()),
                 at_nominal_limit_all=float(satn.mean()),
                 at_point_limit_worst_joint=names[int(pm.argmax())], at_point_limit_worst=float(pm.max()),
                 at_point_limit_all=float(satp.mean()))
        slip = np.array([[np.nan if x is None else x for x in e.get("slip", [None] * 4)] for e in ok], dtype=float)
        if np.isfinite(slip).any():
            fm = np.array([np.nanmean(c) if np.isfinite(c).any() else -1.0 for c in slip.T])
            s.update(slip_worst_foot=G.CORNERS[int(fm.argmax())], slip_worst=float(fm.max()),
                     slip_mean_by_foot=dict(zip(G.CORNERS, map(float, fm))))
    pe = [e["push"] for e in eps if "lateral_peak_m" in e.get("push", {})]  # episodes that reached the push time
    if pe:
        okp = [e["push"] for e in eps if "lateral_peak_m" in e.get("push", {}) and not e["fell"]]
        rec = [q["recovery_s"] for q in okp if q.get("recovery_s") is not None]
        ex = [q["lateral_extra_peak_m"] for q in pe if q.get("lateral_extra_peak_m") is not None]
        cv = [q["com_dv_achieved_mps"] for q in pe if q.get("com_dv_achieved_mps") is not None]
        s["push"] = {"force_n": pe[0]["force_n"], "impulse_ns": pe[0]["impulse_ns"],
                     "applied_steps": sorted({q["applied_steps"] for q in pe}),
                     "com_dv_achieved_mps_mean": float(np.mean(cv)) if cv else None,
                     "lateral_extra_peak_m_mean": float(np.mean(ex)) if ex else None,
                     "lateral_extra_peak_m_max": float(np.max(ex)) if ex else None,
                     "lateral_peak_m_mean": float(np.mean([q["lateral_peak_m"] for q in pe])),
                     "lateral_peak_m_max": float(np.max([q["lateral_peak_m"] for q in pe])),
                     "roll_peak_deg_max": float(np.max([q["roll_peak_deg"] for q in pe])),
                     "recovered": len(rec), "not_recovered_upright": len(okp) - len(rec),
                     "recovery_s_median": float(np.median(rec)) if rec else None,
                     "recovery_s_max": float(np.max(rec)) if rec else None}
    return s


def pair_pushes(points: dict, results_eps: dict) -> None:
    """Against the same seed's unpushed episode, which starts from the same state at the onset (same seed, same model;
    the runs are identical until then), so the differences are the push's effect. Per pushed episode:
    ``lateral_extra_peak_m``, the largest of (its sideways travel along the push - the unpushed episode's) within
    PUSH_LOOK_S of the onset; ``com_dv_achieved_mps``, the centre-of-mass speed along the push over the 10 ms after the
    push minus the unpushed episode's at the same time (the feet's friction resists, so it is below the free-body dv).
    Drops the trajectories afterwards."""
    base_key = point_key(make_point())
    base = {e["seed"]: e["push"] for e in results_eps.get(base_key, [])}
    for k, p in points.items():
        for e in results_eps[k]:
            q = e["push"]
            b = base.get(e["seed"], {})
            sgn = math.copysign(1.0, p["push_dv"])
            traj, bt = np.asarray(q.get("_lat_left", []), float), np.asarray(b.get("_lat_left", []), float)
            if p["push_dv"] and len(traj) and len(bt):
                n = min(len(traj), len(bt))
                diff = sgn * (traj[:n] - bt[:n])
                q["lateral_extra_peak_m"] = _r(diff.max())
                q["lateral_extra_at_3s_m"] = _r(diff[-1]) if n == len(bt) else None
            if p["push_dv"] and "_com_v_left_after" in q and "_com_v_left_after" in b:
                q["com_dv_achieved_mps"] = _r(sgn * (q["_com_v_left_after"] - b["_com_v_left_after"]))
    for k in points:
        for e in results_eps[k]:
            e["push"].pop("_lat_left", None)
    for k in points:
        for e in results_eps[k]:
            e["push"].pop("_com_v_left_after", None)
    for k, p in points.items():  # the push measures matter only for the push sweep and its unpushed baseline
        if not p["push_dv"] and k != base_key:
            for e in results_eps[k]:
                e.pop("push", None)


def find_breaks(fam_rows: list, nominal_x, nominal_speed: float | None) -> dict:
    """Walking outward from the nominal point on each side, the first point with a fall or a speed more than DROP
    below the nominal point's mean speed (same command). ``fam_rows``: [(x, summary)] in sweep order."""
    xs = [x for x, _ in fam_rows]
    i0 = xs.index(nominal_x)

    def broken(s):
        reasons = []
        if s["falls"]:
            reasons.append(f"{s['falls']}/{s['episodes']} fell")
        if nominal_speed and s.get("speed_mps") is not None and s["speed_mps"] < (1 - DROP) * nominal_speed:
            reasons.append(f"speed {s['speed_mps']:.3f} m/s, {100 * (1 - s['speed_mps'] / nominal_speed):.0f} % "
                           "below nominal")
        return reasons

    out = {}
    for side, rng in (("low", range(i0 - 1, -1, -1)), ("high", range(i0 + 1, len(xs)))):
        if not len(rng):
            continue
        out[side] = None
        for i in rng:
            r = broken(fam_rows[i][1])
            if r:
                out[side] = {"x": xs[i], "reasons": r}
                break
    return out


# ---------------------------------------------------------------- reports
def fmt_x(fam_key, x):
    if fam_key == "friction":
        return "0.85 (model)" if x is None else f"{x:g}"
    if fam_key == "terrain":
        return "flat" if x is None else (f"step {100 * x[1]:g} cm" if x[0] == "step" else f"ramp {x[1]:g} deg")
    if fam_key == "delay_ms":
        return f"{x:g} ms"
    if fam_key == "payload_kg":
        return f"+{x:g} kg"
    if fam_key == "push_dv":
        return f"{x:+g} m/s" if x else "none"
    return f"x{x:g}"


def push_desc(res: dict, dv: float) -> str:
    """A push as the free-body velocity change, the impulse (N s) and the centre of mass's measured velocity change
    (against the same seed's unpushed episode; the feet absorb the rest)."""
    s = res["points"][point_key(make_point(push_dv=dv))]["summary"]
    q = s.get("push") or {}
    imp = q.get("impulse_ns") or abs(dv) * s["mass_kg"]
    got = q.get("com_dv_achieved_mps_mean")
    side = "left" if dv > 0 else "right"
    return (f"{abs(dv):g} m/s free-body to the {side} (impulse {imp:.1f} N s"
            + (f"; the centre of mass gained {got:.2f} m/s" if got is not None else "") + ")")


def push_absorption(res: dict) -> tuple | None:
    """Range over the push points of (centre-of-mass velocity change achieved) / (free-body dv), |dv| >= 0.3 m/s
    (smaller pushes are within the gait's own sway)."""
    r = []
    for v in res["points"].values():
        dv = v["point"]["push_dv"]
        got = (v["summary"].get("push") or {}).get("com_dv_achieved_mps_mean")
        if abs(dv) >= 0.3 and got is not None:
            r.append(got / abs(dv))
    return (min(r), max(r)) if r else None


def headline(res: dict) -> list:
    """Bullets for the top of the summary, from the breaks (no hand-written findings)."""
    L, quiet = [], []
    for fam in res["families"]:
        b = fam["breaks"]
        found = {side: v for side, v in b.items() if v}
        if fam["x_key"] == "push_dv":
            continue
        if not found:
            xs = [r["x"] for r in fam["rows"]]
            quiet.append(f"{fam['family']} ({fmt_x(fam['x_key'], xs[0])} to {fmt_x(fam['x_key'], xs[-1])})")
            continue
        L.append(f"- **{fam['family']}**: " + "; ".join(
            f"first break at {fmt_x(fam['x_key'], v['x'])} ({', '.join(v['reasons'])})" for v in found.values()) + ".")
    if quiet:
        L.insert(0, "- **No fall and no speed drop of more than 20 %** in: " + "; ".join(quiet) + ".")
    pf = next((f for f in res["families"] if f["x_key"] == "push_dv"), None)
    if pf:
        def largest_ok(sign, brk):  # the largest push on one side with no fall, inside that side's first break
            xs = [r["x"] for r in pf["rows"] if r["x"] * sign > 0 and not res["points"][r["key"]]["summary"]["falls"]
                  and (brk is None or abs(r["x"]) < abs(brk["x"]))]
            return max(xs, key=abs) if xs else None

        def group(pushes):  # {dv: note}; pushes of the same size on both sides share one description
            out = []
            for size in sorted({abs(x) for x in pushes}):
                xs = sorted((x for x in pushes if abs(x) == size), reverse=True)
                q = [res["points"][point_key(make_point(push_dv=x))]["summary"].get("push") or {} for x in xs]
                imp = q[0].get("impulse_ns") or size * res["points"][point_key(make_point())]["summary"]["mass_kg"]
                got = [f"{v['com_dv_achieved_mps_mean']:.2f}" for v in q if v.get("com_dv_achieved_mps_mean") is not None]
                got_s = "" if not got else (f"; the centre of mass gained {got[0]} m/s" if len(set(got)) == 1 else
                                            f"; the centre of mass gained {' / '.join(got)} m/s")
                sides = " and ".join(("left" if x > 0 else "right") + (f" ({pushes[x]})" if pushes[x] else "")
                                     for x in xs)
                out.append(f"{size:g} m/s free-body (impulse {imp:.1f} N s{got_s}), pushed to the {sides}")
            return "; ".join(out)

        brk = {v["x"]: ", ".join(v["reasons"]) for v in pf["breaks"].values() if v}
        msg = "- **Sideways push**: " + (f"falls start at {group(brk)}" if brk else "no fall in the tested range")
        ok = [x for x in (largest_ok(1, pf["breaks"].get("high")), largest_ok(-1, pf["breaks"].get("low")))
              if x is not None]
        if ok:
            msg += f". Largest push with no fall: {group({x: '' for x in ok})}"
        ab = push_absorption(res)
        if ab:
            msg += (f". The feet absorb part of every push: the centre of mass gains {100 * ab[0]:.0f}-"
                    f"{100 * ab[1]:.0f} % of the free-body velocity change (pushes of 0.3 m/s and more)")
        L.append(msg + ".")
    return L


def markdown(res: dict) -> str:
    L = [f"# Robustness sweep: {res['run_label']} (SIMULATION)", "",
         f"Written by `rl/robustness/sweep.py` on {res['date']}. One policy, so this is a report, not a verdict: "
         "a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.", "",
         f"- Policy `{res['checkpoint']['zip_name']}` (sha256 {res['checkpoint']['zip_sha256'][:16]}...), run commit "
         f"{res['run_commit'][:8]}; environment `{res['env']['trike_env_path']}` (sha256 "
         f"{res['env']['trike_env_sha256'][:16]}...).",
         f"- Every point: {res['seeds_per_point']} episodes (seeds {res['seeds'][0]}-{res['seeds'][-1]}), 20 s, "
         "deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point "
         f"is the **nominal model with observation noise on** (the model as built, delay {NOMINAL_DELAY_MS} ms, "
         "0.25 m/s command); the gait report's \"nominal model\" case has the noise off, so the two are not directly "
         "comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.",
         "- \"At ... limit\" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the "
         "limit; the gait report quotes per-joint and per-group figures.",
         f"- Wall time {res['wall_s'] / 60:.1f} min on {res['workers']} worker processes ({res['episodes_run']} "
         "episodes)."]
    rp = res.get("reproduction")
    if rp:
        L.append(f"- Reproduction check (gait_pipe1's target case, randomisation on, seeds {rp['seeds'][0]}-"
                 f"{rp['seeds'][-1]}): mean speed {rp['speed_mps']:.4f} m/s against the analysis's "
                 f"{rp['reference_speed_mps']:.4f} m/s; per-episode speeds identical: {rp['identical']}.")
    L += ["", "## Headline", ""] + headline(res)
    L += ["", "## Where it breaks", "",
          f"First point outward from nominal with a fall or a speed more than {100 * DROP:.0f} % below the nominal "
          "point's (same command). \"none\": no such point in the tested range.", "",
          "| Sweep | Nominal | Low side | High side |", "|---|---|---|---|"]
    for fam in res["families"]:
        b = fam["breaks"]

        def cell(side):
            if side not in b:
                return "(not tested)"
            if b[side] is None:
                return "none"
            if fam["x_key"] == "push_dv":
                return f"**{push_desc(res, b[side]['x'])}**: " + "; ".join(b[side]["reasons"])
            return f"**{fmt_x(fam['x_key'], b[side]['x'])}**: " + "; ".join(b[side]["reasons"])
        lo = cell("low")
        hi = cell("high")
        L.append(f"| {fam['family']} | {fmt_x(fam['x_key'], fam['nominal_x'])} | {lo} | {hi} |")
    L += ["", "Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so "
          "the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.", ""]
    for fam in res["families"]:
        L += [f"## {fam['family']}", ""]
        if fam["x_key"] == "push_dv":
            ab = push_absorption(res)
            L += ["Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would "
                  "gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), "
                  "and the velocity change its centre of mass actually got, measured against the same seed's unpushed "
                  "episode (identical until the push). The feet's friction absorbs the rest"
                  + (f": the centre of mass gains {100 * ab[0]:.0f}-{100 * ab[1]:.0f} % of dv for pushes of 0.3 m/s "
                     "and more." if ab else "."), "",
                  "| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | "
                  "Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        else:
            L += ["| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean "
                  "(worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | "
                  "At this point's limit, worst joint |", "|---|---|---|---|---|---|---|---|---|---|"]
        for row in fam["rows"]:
            s = res["points"][row["key"]]["summary"]
            x = fmt_x(fam["x_key"], row["x"])
            falls = f"{s['falls']}/{s['episodes']}"
            if "speed_mps" not in s:
                if fam["x_key"] == "push_dv":
                    q = s.get("push") or {}
                    cdv = "" if q.get("com_dv_achieved_mps_mean") is None else f"{q['com_dv_achieved_mps_mean']:.2f}"
                    L.append(f"| {x} | {q.get('impulse_ns', 0):.1f} | {q.get('force_n', 0):.0f} | {cdv} | {falls} | "
                             "all fell | | | | |")
                else:
                    L.append(f"| {x} | {falls} | all fell | | | | | | | |")
                continue
            rel = (f"{100 * (s['speed_mps'] / fam['nominal_speed'] - 1):+.1f} %" if fam.get("nominal_speed")
                   else "")
            if fam["x_key"] == "push_dv":
                q = s.get("push")
                if q is None:
                    L.append(f"| {x} | | | | {falls} | {s['speed_mps']:.3f} | (fell before the push) | | | |")
                    continue
                med = "" if q["recovery_s_median"] is None else f"{q['recovery_s_median']:.2f} ({q['recovery_s_max']:.2f})"
                ext = ("(no push)" if q["lateral_extra_peak_m_mean"] is None else
                       f"{q['lateral_extra_peak_m_mean']:.3f} ({q['lateral_extra_peak_m_max']:.3f})")
                cdv = "" if q["com_dv_achieved_mps_mean"] is None else f"{q['com_dv_achieved_mps_mean']:.2f}"
                L.append(f"| {x} | {q['impulse_ns']:.1f} | {q['force_n']:.0f} | {cdv} | {falls} | {s['speed_mps']:.3f} | "
                         f"{ext} | {med} | {q['not_recovered_upright']} | {q['roll_peak_deg_max']:.1f} |")
            else:
                L.append(f"| {x} | {falls} | {s['speed_mps']:.3f} ({s['speed_min_mps']:.3f}) | {rel} | "
                         f"{s['speed_error_mps']:+.3f} | {s['drift_abs_mean_m']:.3f} ({s['drift_abs_worst_m']:.3f}) | "
                         + (f"{s['slip_worst']:.2f} ({s['slip_worst_foot']}) | " if "slip_worst" in s else "| ")
                         + f"{s['heat_worst']:.2f} ({s['heat_worst_joint']}) | {100 * s['at_nominal_limit_worst']:.1f} % "
                         f"({s['at_nominal_limit_worst_joint']}) | {100 * s['at_point_limit_worst']:.1f} % "
                         f"({s['at_point_limit_worst_joint']}) |")
        if fam["x_key"] == "push_dv":
            q0 = res["points"][point_key(make_point())]["summary"].get("push") or {}
            if q0.get("recovery_s_max") is not None:
                L += ["", f"Noise floor: the unpushed episodes, measured the same way from t = {PUSH_T:g} s, score a "
                      f"recovery time of {q0['recovery_s_median']:.2f} s median and {q0['recovery_s_max']:.2f} s at "
                      "most (the gait's own sideways sway can exceed the threshold), so recovery times below that are "
                      "not distinguishable from walking unpushed."]
        terms = sorted({tt for row in fam["rows"] for tt in res["points"][row["key"]]["summary"]["fall_terminations"]})
        if terms:
            L += ["", "Fall terminations: " + ", ".join(f"`{tt}`" for tt in terms) + "."]
        if fam.get("beyond_requested"):
            L += ["", "Beyond the requested set (added to find the edge): "
                  + ", ".join(fmt_x(fam["x_key"], tuple(x) if isinstance(x, list) else x)
                              for x in fam["beyond_requested"]) + "."]
        if fam.get("chart"):
            L += ["", f"Chart: `{fam['chart']}`."]
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def find_checkpoint(cdir: Path):
    zips = sorted(cdir.glob("*_steps.zip"), key=lambda q: int(q.stem.split("_")[-2]))
    if not zips:
        raise FileNotFoundError(f"no *_steps.zip in {cdir}")
    z = zips[-1]
    steps = z.stem.split("_")[-2]
    pk = [q for q in cdir.glob(f"*vecnormalize_{steps}_steps.pkl")]
    if len(pk) != 1:
        raise FileNotFoundError(f"no unique VecNormalize pickle for {z.name} in {cdir}")
    return z, pk[0]


def add_breaks(res: dict) -> None:
    for fam in res["families"]:
        nom_key = next(r["key"] for r in fam["rows"] if r["x"] == fam["nominal_x"])
        fam["nominal_speed"] = res["points"][nom_key]["summary"].get("speed_mps")
        fam["breaks"] = find_breaks([(r["x"], res["points"][r["key"]]["summary"]) for r in fam["rows"]],
                                    fam["nominal_x"], fam["nominal_speed"])


def _tuples(x):
    return tuple(x) if isinstance(x, list) else x


def rebuild(path: Path, charts_too: bool = True) -> dict:
    """Summaries, breaks, Markdown and charts again from the episodes in a finished run's JSON."""
    if path.suffix == ".gz":
        import gzip
        res = json.loads(gzip.decompress(path.read_bytes()).decode())
    else:
        res = json.loads(path.read_text())
    for v in res["points"].values():
        v["point"]["terrain"] = _tuples(v["point"]["terrain"])
        v["summary"] = summarise_point(v["point"], v["episodes"], res["joint_names"])
    for fam in res["families"]:
        fam["nominal_x"] = _tuples(fam["nominal_x"])
        for r in fam["rows"]:
            r["x"] = _tuples(r["x"])
    add_breaks(res)
    out = path.parent
    write_outputs(res, out)
    if charts_too:
        import charts

        for fam, p in charts.make_charts(res, out).items():
            for f in res["families"]:
                if f["sweep"] == fam:
                    f["chart"] = p.name
        write_outputs(res, out)
    print(markdown(res))
    return res


_NUM = r"(?:-?[0-9][0-9.eE+-]*|null|true|false)"
_NUM_LIST = re.compile(rf"\[(\s*{_NUM}(?:\s*,\s*{_NUM})*\s*)\]")


def _round(o, n=6):
    if isinstance(o, float):
        return float(f"{o:.{n}g}")
    if isinstance(o, dict):
        return {k: _round(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, n) for v in o]
    return o


def compact_json(res: dict) -> str:
    """Indented JSON with each episode on one line, every list of numbers on one line, and the per-point summaries
    rounded to 6 significant figures (the episodes keep their own precision)."""
    out, eps = dict(res), []
    out["points"] = {}
    for k, v in res["points"].items():
        out["points"][k] = {**v, "summary": _round(v["summary"]),
                            "episodes": [f"@@EP{len(eps) + i}@@" for i in range(len(v["episodes"]))]}
        eps += v["episodes"]
    txt = _NUM_LIST.sub(lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
                        json.dumps(out, indent=1))
    return re.sub(r'"@@EP(\d+)@@"', lambda m: json.dumps(eps[int(m.group(1))]), txt)


def write_outputs(res: dict, out: Path) -> None:
    # gzip (reproducible: mtime 0) keeps the per-episode file under the repository's 500 KB limit
    import gzip
    with open(out / f"sweep_{res['run']}.json.gz", "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write((compact_json(res) + "\n").encode())
    (out / f"summary_{res['run']}.md").write_text(markdown(res))


def plan(families=FAMILIES, sweeps=None):
    fams, points = [], {}
    for sweep, fam, xk, xs, x_nom, fixed, extra in families:
        if sweeps and sweep not in sweeps:
            continue
        rows = []
        for x in xs:
            p = make_point(**fixed, **{xk: x})
            k = point_key(p)
            points[k] = p
            rows.append({"x": x, "key": k})
        fams.append({"sweep": sweep, "family": fam, "x_key": xk, "nominal_x": x_nom, "fixed": fixed, "rows": rows,
                     "beyond_requested": list(extra)})
    return fams, points


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", help="checkpoint folder (zip, VecNormalize pickle, run_record.json)")
    ap.add_argument("--run", help="short run name: results go to results/<run>/")
    ap.add_argument("--from-json", default=None,
                    help="rebuild the summaries, Markdown and charts from a finished run's JSON (no simulation)")
    ap.add_argument("--run-label", default=None, help="name on the charts (default: the run record's folder name)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=8, help="episodes per point (seeds SEED0, SEED0 + 1, ...)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sweeps", nargs="*", default=None, help="subset: delay friction mass drive push terrain")
    ap.add_argument("--no-repro", action="store_true")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args(argv)
    if a.from_json:
        return rebuild(Path(a.from_json), charts_too=not a.no_charts)
    if not (a.ckpt_dir and a.run):
        ap.error("--ckpt-dir and --run are required (or --from-json)")
    t_start = time.time()
    cdir = Path(a.ckpt_dir)
    out = Path(a.out) if a.out else HERE / "results" / a.run
    out.mkdir(parents=True, exist_ok=True)
    zp, pk = find_checkpoint(cdir)
    record = json.loads((cdir / "run_record.json").read_text())
    st = record["settings"]
    ec = dict(st["env_config"])
    label = a.run_label or Path(record["sessions"][0]["args"].get("out", a.run)).name

    # consistency checks, as gait_pipe1 does
    env = build_env(ec, randomise=True)
    mine = json.loads(json.dumps(env.cfg.to_dict()))
    diffs = {k: (mine.get(k), v) for k, v in ec.items() if k != "model_path" and mine.get(k) != v}
    assert not diffs, f"environment config differs from the run record: {diffs}"
    stats = G.load_obs_stats(pk)
    assert stats[0].shape == env.observation_space.shape, (stats[0].shape, env.observation_space.shape)
    names = [env.model.actuator(int(i)).name for i in env._leg_act]
    del env

    fams, points = plan(sweeps=a.sweeps)
    seeds = [SEED0 + i for i in range(a.seeds)]
    tasks = [(k, p, s) for k, p in points.items() for s in seeds]
    repro_key = None
    if not a.no_repro:
        rp = make_point(randomise=True)
        repro_key = point_key(rp)
        tasks += [(repro_key, rp, s) for s in REPRO_SEEDS]
    # longest first (terrain envs are built on first use), then interleave so both workers share every kind
    print(f"{len(points)} points x {len(seeds)} seeds (+ {len(REPRO_SEEDS) if repro_key else 0} reproduction "
          f"episodes) = {len(tasks)} episodes on {a.workers} workers", flush=True)
    results_eps: dict = {k: [] for k in list(points) + ([repro_key] if repro_key else [])}
    ckpt = {"zip": str(zp), "pkl": str(pk)}
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(ckpt, ec, st["net"], st["log_std_init"])) as pool:
        for k, rec in pool.imap_unordered(_task, tasks, chunksize=1):
            results_eps[k].append(rec)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                el = time.time() - t_start
                print(f"  {done}/{len(tasks)} episodes, {el / 60:.1f} min, about "
                      f"{el / done * (len(tasks) - done) / 60:.1f} min left", flush=True)

    res = {"run": a.run, "run_label": label, "date": a.date, "simulation_only": True,
           "checkpoint": {"zip_name": zp.name, "zip_sha256": sha256(zp), "vecnormalize_name": pk.name,
                          "vecnormalize_sha256": sha256(pk)},
           "run_commit": record["sessions"][0]["git"]["commit"],
           "env": {"trike_env_path": os.path.relpath(trike_env.__file__, RL.parents[2]),
                   "trike_env_sha256": sha256(Path(trike_env.__file__)),
                   "model": Path(ec["model_path"]).name, "model_sha256": sha256(G.GEN / Path(ec["model_path"]).name),
                   "TRIKE_ENV_DIR": os.environ.get("TRIKE_ENV_DIR")},
           "seeds": seeds, "seeds_per_point": len(seeds), "workers": a.workers,
           "settings": {"nominal_delay_ms": NOMINAL_DELAY_MS, "command_mps": COMMAND, "push_t_s": PUSH_T,
                        "push_duration_s": PUSH_S, "push_look_s": PUSH_LOOK_S, "recover_vlat_mps": RECOVER_VLAT,
                        "recover_window_s": RECOVER_WIN_S, "recover_hold_s": RECOVER_HOLD_S,
                        "terrain_edge_m": TERRAIN_EDGE_M, "speed_drop_break": DROP, "window_from_s": G.WINDOW_S,
                        "continuous_rating_nm": G.THERMAL, "nominal_limit_nm": G.LIMIT, "training_ranges": TRAINING},
           "joint_names": names, "points": {}, "families": []}
    pair_pushes(points, results_eps)
    for k, p in points.items():
        eps = sorted(results_eps[k], key=lambda e: e["seed"])
        res["points"][k] = {"point": p, "summary": summarise_point(p, eps, names), "episodes": eps}
    if repro_key:
        eps = sorted(results_eps[repro_key], key=lambda e: e["seed"])
        ref_path = RL / "analysis" / "results" / "gait_pipe1_2026-11-19.json"
        rp_out = {"seeds": list(REPRO_SEEDS), "speeds_mps": [e.get("forward_speed_mps") for e in eps],
                  "speed_mps": float(np.mean([e["forward_speed_mps"] for e in eps])), "falls": sum(e["fell"] for e in eps)}
        if ref_path.exists():
            ref = json.loads(ref_path.read_text())["cases"]["target 0.25 m/s"]["episodes"]
            rv = [float(e["forward_speed_mps"]) for e in ref]
            rp_out.update(reference=os.path.relpath(ref_path, RL.parents[2]), reference_speeds_mps=rv,
                          reference_speed_mps=float(np.mean([e["forward_speed_mps"] for e in ref])),
                          identical=rv == rp_out["speeds_mps"])
        res["reproduction"] = rp_out
    res["families"] = fams
    add_breaks(res)
    res["episodes_run"] = len(tasks)
    res["wall_s"] = time.time() - t_start
    write_outputs(res, out)  # before the charts, so a chart error cannot lose the results
    if not a.no_charts:
        import charts

        for fam, path in charts.make_charts(res, out).items():
            for f in res["families"]:
                if f["sweep"] == fam:
                    f["chart"] = path.name
        write_outputs(res, out)
    print(markdown(res))
    print(f"wrote {out} in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
