"""Gait and drive measures of an evaluation episode, for train.py's periodic evaluation and evaluate.py. Simulation.

The definitions are those of ``analysis/gait_pipe1.py`` (the gait analysis of the trike60-pipe1 policy), so a training
run reports them directly; a test applies that script's ``measures`` to the same log and compares. Logged every
physics step (1 ms) through the env's ``substep_hook``; measured over the window t >= 2 s.

Per episode (``measures``):
  forward_speed_mps          chassis travel along the heading at reset over the window / window length
  sideways_drift_m           chassis travel across that heading over the window (signed, +left)
  heading_change_deg         chassis yaw at the window's end minus at its start (wrapped)
  drives[joint]              per leg drive: RMS torque, RMS / continuous rating, and the fraction of physics steps
                             at >= 99 % of its NOMINAL force limit (21 / 21 / 37.8 / 18.9 N m) and, separately, of the
                             EPISODE'S randomised limit (0.85-1.0 x nominal)
  drive_groups[group]        roll, pitch, front elbow, hind knee: the worst joint's RMS / rating (and its name), and
                             the group's at-limit fractions (nominal and episode limit), over its joints and steps
  slip_ratio[foot]           sole horizontal travel while loaded (> contact_threshold_n) / chassis forward travel
                             (None when the chassis travelled 0.05 m or less)
  touchdowns_per_s[foot]     DEBOUNCED: a foot counts as lifted after contact_debounce_s unloaded (the env's rule);
  touchdowns_per_s_raw[foot] every unloaded-to-loaded change at 2 N, as the analysis counted them (includes the
                             ~2 ms contact-model rebounds)
  power_w, cost_of_transport mean of sum |torque x joint speed| over the leg drives; power / (m g v)
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from trike_env import CORNERS, DRIVE_GROUPS, LEG_ACTUATORS, TrikeEnv, drive_group

WINDOW_S = 2.0
AT_LIMIT = 0.99


def record_episode(env: TrikeEnv, act, *, seed: int, command: float | None, delay_steps: int = 10) -> dict:
    """Run one episode with ``act(obs) -> action`` and log every physics step; returns the log (``measures`` input).
    The log's keys and shapes follow ``analysis/gait_pipe1.run_episode``. ``command`` None: drawn as in training."""
    options = {"delay_steps": delay_steps}
    if command is not None:
        options["command_speed"] = command
    obs, reset_info = env.reset(seed=seed, options=options)
    step_v = []
    m, d = env.model, env.data
    idx = np.asarray(env._leg_act, dtype=np.int64)
    log: dict = {k: [] for k in ("t", "tau", "omega", "load", "sole_v", "pos", "rp", "yaw")}
    vel6 = np.zeros(6)

    def hook(e: TrikeEnv) -> None:
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
        log["rp"].append((math.degrees(math.atan2(R[2, 1], R[2, 2])),
                          math.degrees(math.asin(-max(-1.0, min(1.0, R[2, 0]))))))
        log["yaw"].append(math.atan2(R[1, 0], R[0, 0]))

    env.substep_hook = hook
    try:
        while True:
            obs, _, terminated, truncated, info = env.step(act(obs))
            step_v.append(info["forward_speed"])
            if terminated or truncated:
                break
    finally:
        env.substep_hook = None
    out = {k: np.asarray(v) for k, v in log.items()}
    out.update(names=[m.actuator(int(i)).name for i in idx], limit_ep=m.actuator_forcerange[idx, 1].copy(),
               limit_nominal=env._nominal["actuator_forcerange"][idx, 1].copy(), heading0=env._heading,
               mass=float(m.body_mass.sum()), fell=bool(terminated), termination=info["termination"],
               contact_n=env.cfg.contact_threshold_n, ratings=env.continuous_torque.copy(),
               debounce_steps=env._debounce_steps, env_touchdowns=env.touchdowns, command=env.command_speed,
               step_forward_speed=np.asarray(step_v), episode_forward_speed=info["episode_forward_speed"],
               time_s=info["time_s"], control_dt=env.dt, reset_info=reset_info)
    return out


def debounced_touchdowns(down: np.ndarray, debounce_steps: int, start: int = 0) -> np.ndarray:
    """Touchdowns per foot counted at samples >= ``start``, with the env's rule: a foot is lifted once unloaded for
    ``debounce_steps`` consecutive samples, but only after it has been loaded at least once since reset (the soles
    start 2 mm up, so the first landing is not a step); a touchdown is the first loaded sample after a lift.
    ``down``: (samples, feet) loaded flags, from the first physics step after reset."""
    counts = np.zeros(down.shape[1], dtype=np.int64)
    for i in range(down.shape[1]):
        x = down[:, i]
        edges = np.flatnonzero(np.diff(x.astype(np.int8))) + 1  # starts of runs after the first
        starts = np.r_[0, edges]
        lengths = np.diff(np.r_[starts, len(x)])
        for s, n in zip(starts, lengths):
            ends_at = s + n  # the sample after the run: a touchdown if this was a long unloaded run
            if s > 0 and not x[s] and n >= debounce_steps and ends_at < len(x) and ends_at >= start:
                counts[i] += 1
    return counts


def measures(a: dict, command: float) -> dict:
    t = a["t"]
    start = WINDOW_S if (t >= WINDOW_S).sum() >= 100 else 0.0  # short episodes (tests): the whole episode
    w = t >= start
    dt = float(np.median(np.diff(t)))
    T = float(t[w][-1] - t[w][0] + dt)
    h = np.array([math.cos(a["heading0"]), math.sin(a["heading0"])])
    p = a["pos"][w]
    disp = p[-1, :2] - p[0, :2]
    fwd = float(disp @ h)
    yaw = a["yaw"][w]
    out = {"command_mps": command, "fell": bool(a["fell"]), "termination": a["termination"], "window_from_s": start,
           "window_s": T,
           "forward_speed_mps": fwd / T, "sideways_drift_m": float(disp @ np.array([-h[1], h[0]])),
           "heading_change_deg": math.degrees((yaw[-1] - yaw[0] + math.pi) % (2 * math.pi) - math.pi)}
    down_all = a["load"] > a["contact_n"]
    down = down_all[w]
    first = int(np.argmax(w))
    out["touchdowns_per_s"] = dict(zip(CORNERS, (debounced_touchdowns(down_all, int(a["debounce_steps"]), first)
                                                 / T).tolist()))
    td_raw = (down[1:] & ~down[:-1]).sum(0)
    out["touchdowns_per_s_raw"] = {c: float(td_raw[i] / T) for i, c in enumerate(CORNERS)}
    sv = a["sole_v"][w]
    out["slip_ratio"] = {}
    out["stance_slip_mean_mps"] = {}
    for i, c in enumerate(CORNERS):
        s = sv[down[:, i], i]
        out["slip_ratio"][c] = float(s.sum() * dt / fwd) if len(s) and fwd > 0.05 else None
        out["stance_slip_mean_mps"][c] = float(s.mean()) if len(s) else None
    tau, om = a["tau"][w], a["omega"][w]
    rms = np.sqrt((tau ** 2).mean(0))
    at_nom = np.abs(tau) >= AT_LIMIT * a["limit_nominal"]
    at_ep = np.abs(tau) >= AT_LIMIT * a["limit_ep"]
    out["drives"] = {nm: {"rms_nm": float(rms[j]), "rms_over_continuous": float(rms[j] / a["ratings"][j]),
                          "at_limit_nominal": float(at_nom[:, j].mean()), "at_limit_episode": float(at_ep[:, j].mean())}
                     for j, nm in enumerate(a["names"])}
    out["drive_groups"] = {}
    for g in DRIVE_GROUPS:
        cols = [j for j, nm in enumerate(a["names"]) if drive_group(nm) == g]
        ratio = rms[cols] / a["ratings"][cols]
        k = int(np.argmax(ratio))
        out["drive_groups"][g] = {"rms_over_continuous_worst": float(ratio[k]), "worst_joint": a["names"][cols[k]],
                                  "at_limit_nominal": float(at_nom[:, cols].mean()),
                                  "at_limit_episode": float(at_ep[:, cols].mean())}
    power = float(np.abs(tau * om).sum(1).mean())
    out["power_w"] = power
    out["cost_of_transport"] = power / (a["mass"] * 9.81 * (fwd / T)) if fwd / T > 0.02 else None
    return out


def summarise(eps: list[dict]) -> dict:
    """Across episodes: means for speeds, rates and power; the worst episode for drive loads, at-limit fractions
    and slip (as the analysis does); the largest drift and heading change in magnitude."""

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def worst(vals):
        vals = [v for v in vals if v is not None]
        return float(max(vals)) if vals else None

    s = {"episodes": len(eps), "falls": sum(e["fell"] for e in eps),
         "forward_speed_mps": mean([e["forward_speed_mps"] for e in eps]),
         "sideways_drift_m_worst": max((e["sideways_drift_m"] for e in eps), key=abs),
         "heading_change_deg_worst": max((e["heading_change_deg"] for e in eps), key=abs),
         "power_w": mean([e["power_w"] for e in eps]),
         "cost_of_transport": mean([e["cost_of_transport"] for e in eps]),
         "touchdowns_per_s": {c: mean([e["touchdowns_per_s"][c] for e in eps]) for c in CORNERS},
         "touchdowns_per_s_raw": {c: mean([e["touchdowns_per_s_raw"][c] for e in eps]) for c in CORNERS},
         "slip_ratio_mean": {c: mean([e["slip_ratio"][c] for e in eps]) for c in CORNERS},
         "slip_ratio_worst": {c: worst([e["slip_ratio"][c] for e in eps]) for c in CORNERS},
         "drives_worst": {nm: {k: worst([e["drives"][nm][k] for e in eps]) for k in eps[0]["drives"][nm]}
                          for nm in LEG_ACTUATORS},
         "drive_groups_worst": {}}
    for g in DRIVE_GROUPS:
        ratios = [(e["drive_groups"][g]["rms_over_continuous_worst"], e["drive_groups"][g]["worst_joint"]) for e in eps]
        r, joint = max(ratios)
        s["drive_groups_worst"][g] = {"rms_over_continuous_worst": r, "worst_joint": joint,
                                      "at_limit_nominal": worst([e["drive_groups"][g]["at_limit_nominal"] for e in eps]),
                                      "at_limit_episode": worst([e["drive_groups"][g]["at_limit_episode"] for e in eps])}
    return s


def one_line(s: dict) -> str:
    """The summary's key figures on one line (train.py and evaluate.py print it)."""
    g = s["drive_groups_worst"]

    def f(x, fmt="{:.2f}"):
        return "-" if x is None else fmt.format(x)

    short = {"roll": "roll", "pitch": "pitch", "front elbow": "elbow", "hind knee": "knee"}
    rms = " ".join(f"{short[k]} {v['rms_over_continuous_worst']:.2f}" for k, v in g.items())
    worst = max(g.values(), key=lambda v: v["rms_over_continuous_worst"])["worst_joint"]
    lim = "/".join(f(v["at_limit_nominal"] * 100, "{:.1f}") for v in g.values())
    slip = "/".join(f(s["slip_ratio_mean"][c]) for c in CORNERS)
    steps = "/".join(f(s["touchdowns_per_s"][c], "{:.1f}") for c in CORNERS)
    return (f"RMS torque/rating {rms} (worst {worst}); at >=99% nominal limit {lim} %; slip ratio LF/RF/LH/RH {slip}; "
            f"touchdowns/s {steps}; power {f(s['power_w'], '{:.0f}')} W; drift {s['sideways_drift_m_worst']:+.2f} m, "
            f"heading {s['heading_change_deg_worst']:+.1f} deg")
