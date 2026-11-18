"""Scripted, statically stable crawl for the Triceratops build models (task 4). Rules: PREREG_CRAWL.md.

One foot moves at a time (left hind, left fore, right hind, right fore). Before each swing, with four feet down, the
body moves forward by a quarter stride and sideways as needed so that the planned centre of mass (computed from the
full model at the start and the end of the swing) lies at least `margin_plan` inside the triangle of the three feet
that stay down, on the ankle-pin basis. Joint targets come from damped-least-squares inverse kinematics on the model
itself; the drives are the model's position servos. Open loop: no feedback from the simulated body.

Usage:
  python crawl.py --model generated/trike_T20_closed.xml --seed 100 [--out DIR] [--video PATH]
Simulation only.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
ORDER = ("left_hind", "left_fore", "right_hind", "right_fore")
AXES = ("roll", "pitch", "flexion")

# Gait parameters. Frozen for the judged runs in the commit that runs them (PREREG_CRAWL.md).
PARAMS = {
    "stride_m": 0.15,      # each foot advances one stride per cycle; the body a quarter stride per step
    "lift_m": 0.04,        # swing foot clearance
    "shift_s": 0.50,       # four-foot body shift before each swing
    "swing_s": 0.50,       # swing duration
    "margin_plan_m": 0.030,
    "body_y_max_m": 0.070,
    "cycles": 8,
    "stand_s": 1.0,
    "control_dt": 0.010,
}


def cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]


def tri_margin(p, tri):
    """Signed distance from p to the boundary of triangle tri (3 x 2); positive inside."""
    a, b, c = tri
    if cross2(b - a, c - a) < 0:
        b, c = c, b
    m = 1e9
    for u, v in ((a, b), (b, c), (c, a)):
        t = v - u
        m = min(m, cross2(t, p - u) / np.linalg.norm(t))
    return m


def min_jerk(u):
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


class Kin:
    """Kinematics of the model with the chassis held level at a given position."""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.q_adr = {c: [model.jnt_qposadr[model.joint(f"{c}_{a}").id] for a in AXES] for c in CORNERS}
        self.dof = {c: [model.jnt_dofadr[model.joint(f"{c}_{a}").id] for a in AXES] for c in CORNERS}
        self.rng = {c: np.array([model.jnt_range[model.joint(f"{c}_{a}").id] for a in AXES]) for c in CORNERS}
        self.foot = {c: model.body(f"{c}_foot").id for c in CORNERS}
        self.chassis = model.body("chassis").id
        self.jacp = np.zeros((3, model.nv))
        self.d.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, self.d)
        self.z0 = float(self.d.xpos[self.chassis][2])
        self.neutral_ankle = {c: self.d.xpos[self.foot[c]].copy() for c in CORNERS}

    def set_body(self, bx, by):
        self.d.qpos[0:3] = (bx, by, self.z0)
        self.d.qpos[3:7] = (1, 0, 0, 0)

    def ik(self, c, target, q0, tol=1e-6, iters=60):
        q = np.array(q0, dtype=float)
        adr, dof, lo, hi = self.q_adr[c], self.dof[c], self.rng[c][:, 0], self.rng[c][:, 1]
        for _ in range(iters):
            self.d.qpos[adr] = q
            mujoco.mj_kinematics(self.m, self.d)
            e = target - self.d.xpos[self.foot[c]]
            if e @ e < tol * tol:
                return q, True
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacBody(self.m, self.d, self.jacp, None, self.foot[c])
            J = self.jacp[:, dof]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
            q = np.clip(q + dq, lo, hi)
        self.d.qpos[adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        e = target - self.d.xpos[self.foot[c]]
        return q, bool(e @ e < 1e-4 ** 2)

    def pose(self, bx, by, feet, qguess):
        """Joint angles for body at (bx, by) and ankle targets `feet` (world); returns q dict, ok, CoM xy."""
        self.set_body(bx, by)
        q, ok = {}, True
        for c in CORNERS:
            q[c], good = self.ik(c, feet[c], qguess[c])
            ok = ok and good
        for c in CORNERS:
            self.d.qpos[self.q_adr[c]] = q[c]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return q, ok, self.d.subtree_com[self.chassis][:2].copy()


def plan(model, P):
    """Body positions and foot placements per step, with the planned margins."""
    kin = Kin(model)
    feet = {c: kin.neutral_ankle[c].copy() for c in CORNERS}
    hip_x = {c: feet[c][0] for c in CORNERS}  # body at x = 0 in the neutral pose
    qg = {c: np.zeros(3) for c in CORNERS}
    bx, by = 0.0, 0.0
    S = P["stride_m"]
    steps = []
    for k in range(4 * P["cycles"]):
        L = ORDER[k % 4]
        bx_nom = bx + S / 4
        best = None
        for dx in (0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03):
            for y in np.arange(-P["body_y_max_m"], P["body_y_max_m"] + 1e-9, 0.002):
                cx = bx_nom + dx
                end = {c: feet[c].copy() for c in CORNERS}
                end[L][0] = cx + hip_x[L] + S / 2
                q1, ok1, com1 = kin.pose(cx, y, feet, qg)
                q2, ok2, com2 = kin.pose(cx, y, end, q1)
                if not (ok1 and ok2):
                    continue
                tri = np.array([feet[c][:2] for c in CORNERS if c != L])
                mg = min(tri_margin(com1, tri), tri_margin(com2, tri))
                cost = abs(y - by) + 2 * abs(dx)
                cand = (mg >= P["margin_plan_m"], -cost if mg >= P["margin_plan_m"] else mg, cx, y, end, q1, mg)
                if best is None or cand[:2] > best[:2]:
                    best = cand
            if best is not None and best[0]:
                break
        if best is None:
            raise RuntimeError(f"step {k}: no reachable body position")
        ok, _, bx, by, end, q1, mg = best
        steps.append({"k": k, "leg": L, "body": [bx, by], "from": feet[L].tolist(), "to": end[L].tolist(),
                      "margin_plan_m": float(mg), "meets_plan": bool(ok)})
        feet = end
        qg = {c: q1[c] for c in CORNERS}
    return kin, steps


def trajectory(kin, steps, P):
    """Joint targets every control step, plus the phase table (swing windows)."""
    feet = {c: kin.neutral_ankle[c].copy() for c in CORNERS}
    b = np.array([0.0, 0.0])
    dt = P["control_dt"]
    t = P["stand_s"]
    times, targets, phases = [0.0, t], [], []
    qg = {c: np.zeros(3) for c in CORNERS}
    q0 = np.zeros(12)
    targets += [q0.copy(), q0.copy()]
    leg_index = {c: i for i, c in enumerate(CORNERS)}
    for s in steps:
        b0, b1 = b.copy(), np.array(s["body"])
        L = s["leg"]
        p0, p1 = np.array(s["from"]), np.array(s["to"])
        n_shift, n_swing = round(P["shift_s"] / dt), round(P["swing_s"] / dt)
        for i in range(1, n_shift + n_swing + 1):
            if i <= n_shift:
                u = min_jerk(i / n_shift)
                bb = b0 + u * (b1 - b0)
                ft = feet
            else:
                v = (i - n_shift) / n_swing
                bb = b1
                ft = dict(feet)
                p = p0 + min_jerk(v) * (p1 - p0)
                p[2] = p0[2] + P["lift_m"] * math.sin(math.pi * v) ** 2
                ft[L] = p
            q, ok, _ = kin.pose(bb[0], bb[1], ft, qg)
            qg = q
            times.append(t + i * dt)
            targets.append(np.concatenate([q[c] for c in CORNERS]))
        phases.append({"leg": L, "swing_start": t + P["shift_s"], "swing_end": t + P["shift_s"] + P["swing_s"],
                       "leg_index": leg_index[L]})
        t += P["shift_s"] + P["swing_s"]
        b = b1
        feet = dict(feet)
        feet[L] = p1
    times.append(t + P["stand_s"])
    targets.append(targets[-1].copy())
    return np.array(times), np.array(targets), phases


def apply_seed(model, data, seed):
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-0.02, 0.02, size=model.njnt)
    mu = float(rng.uniform(0.6, 1.0))
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            data.qpos[model.jnt_qposadr[j]] += offsets[j]
    floor = model.geom("floor").id
    model.geom_friction[floor, 0] = mu
    for c in CORNERS:
        model.geom_friction[model.geom(f"{c}_sole").id, 0] = mu
    return mu


def run(model_path, seed, P=PARAMS, video=None):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    kin, steps = plan(model, P)
    times, targets, phases = trajectory(kin, steps, P)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mu = apply_seed(model, data, seed)
    mujoco.mj_forward(model, data)
    act = [model.actuator(f"{c}_{a}").id for c in CORNERS for a in AXES]
    lim = model.actuator_forcerange[:, 1].copy()
    W = -model.opt.gravity[2] * model.body_mass.sum()
    ch = model.body("chassis").id
    foot = {c: model.body(f"{c}_foot").id for c in CORNERS}
    touch = {model.body(model.site_bodyid[model.sensor_objid[s]]).name.replace("_foot", ""): model.sensor_adr[s]
             for s in np.flatnonzero(model.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)}
    jq = [model.jnt_qposadr[model.joint(f"{c}_{a}").id] for c in CORNERS for a in AXES]
    jv = [model.jnt_dofadr[model.joint(f"{c}_{a}").id] for c in CORNERS for a in AXES]
    T = times[-1]
    n = round(T / model.opt.timestep)
    every = round(P["control_dt"] / model.opt.timestep)
    log = {k: [] for k in ("t", "com", "base", "tilt", "yaw", "load", "ankle", "frc", "qvel", "sat")}
    frames = []
    renderer = None
    if video:
        model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = ch
        cam.distance, cam.azimuth, cam.elevation = 2.2, 140, -14
    warn0 = 0
    for k in range(n):
        t = k * model.opt.timestep
        data.ctrl[act] = [np.interp(t, times, targets[:, i]) for i in range(12)]
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            log["nan"] = True
            break
        if k % every == 0:
            R = data.xmat[ch].reshape(3, 3)
            log["t"].append(t)
            log["com"].append(data.subtree_com[ch][:2].tolist())
            log["base"].append(data.xpos[ch].tolist())
            log["tilt"].append(math.degrees(math.acos(max(-1, min(1, R[2, 2])))))
            log["yaw"].append(math.degrees(math.atan2(R[1, 0], R[0, 0])))
            log["load"].append([float(data.sensordata[touch[c]]) for c in CORNERS])
            log["ankle"].append([data.xpos[foot[c]][:2].tolist() for c in CORNERS])
            log["frc"].append(data.actuator_force[act].tolist())
            log["qvel"].append(data.qvel[jv].tolist())
            log["sat"].append(bool(np.any(np.abs(data.actuator_force) >= 0.99 * lim)))
        if renderer is not None and k % 33 == 0:  # about 30 frames per second
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    warn = int(np.sum(data.warning.number))
    if renderer is not None:
        import imageio
        imageio.mimsave(video, frames, fps=30, codec="libx264", quality=6, macro_block_size=8)
    return {"model": str(model_path), "seed": seed, "mu": mu, "params": P, "plan": steps, "phases": phases,
            "warnings": warn, "weight_n": W, "log": log, "limits": lim[act].tolist()}


def judge(r):
    P, log = r["params"], r["log"]
    t = np.array(log["t"])
    W = r["weight_n"]
    out, info = {}, {}
    nan = r["log"].get("nan", False)
    after = t >= P["stand_s"]
    base = np.array(log["base"])
    h0 = base[np.searchsorted(t, P["stand_s"]), 2]
    out["C1"] = (not nan and r["warnings"] == 0 and max(log["tilt"]) <= 10.0
                 and float(np.min(base[after, 2])) >= 0.85 * h0)
    info["max_tilt_deg"] = float(max(log["tilt"]))
    info["min_height_ratio"] = float(np.min(base[after, 2]) / h0)
    i0 = np.searchsorted(t, P["stand_s"])
    travel = float(base[-1, 0] - base[i0, 0])
    planned = P["cycles"] * P["stride_m"]
    drift = float(abs(base[-1, 1] - base[i0, 1]))
    yaw = float(abs(log["yaw"][-1] - log["yaw"][i0]))
    out["C2"] = travel >= 0.8 * planned and drift <= 0.10 and yaw <= 10.0
    info.update(travel_m=travel, planned_m=planned, drift_m=drift, yaw_change_deg=yaw,
                speed_mps=travel / (t[-1] - 2 * P["stand_s"]))
    com = np.array(log["com"])
    ankle = np.array(log["ankle"])
    load = np.array(log["load"])
    margins, c4_ok = [], True
    for ph in r["phases"]:
        w = (t >= ph["swing_start"]) & (t <= ph["swing_end"])
        L = ph["leg_index"]
        others = [i for i in range(4) if i != L]
        for i in np.flatnonzero(w):
            margins.append(tri_margin(com[i], ankle[i, others]))
        unloaded = np.mean(load[w, L] < 0.05 * W)
        others_ok = np.all(load[w][:, others] >= 0.05 * W)
        if unloaded < 0.80 or not others_ok:
            c4_ok = False
    out["C3"] = bool(min(margins) >= 0.010)
    info["min_margin_measured_m"] = float(min(margins))
    out["C4"] = c4_ok
    # C5: planted feet hold, from 0.10 s after touchdown to the next lift-off
    slips = []
    events = {i: [(0.0, None)] for i in range(4)}
    for ph in r["phases"]:
        L = ph["leg_index"]
        events[L][-1] = (events[L][-1][0], ph["swing_start"])
        events[L].append((ph["swing_end"], None))
    for i in range(4):
        events[i][-1] = (events[i][-1][0], t[-1])
        for td, lo in events[i]:
            w = np.flatnonzero((t >= td + 0.10) & (t <= lo))
            if len(w) >= 2:
                p = ankle[w, i]
                slips.append(float(np.max(np.linalg.norm(p - p[0], axis=1))))
    out["C5"] = max(slips) <= 0.010
    info["max_planted_slip_m"] = max(slips)
    sat = np.mean(log["sat"])
    out["C6"] = sat <= 0.01
    info["saturated_fraction"] = float(sat)
    frc = np.abs(np.array(log["frc"]))
    qv = np.abs(np.array(log["qvel"]))
    names = [f"{c}_{a}" for c in CORNERS for a in AXES]
    info["joint_peak_nm"] = dict(zip(names, frc.max(0).round(2).tolist()))
    info["joint_rms_nm"] = dict(zip(names, np.sqrt((np.array(log["frc"]) ** 2).mean(0)).round(2).tolist()))
    info["joint_peak_rpm"] = dict(zip(names, (qv.max(0) * 60 / (2 * math.pi)).round(1).tolist()))
    info["plan_min_margin_m"] = min(s["margin_plan_m"] for s in r["plan"])
    info["plan_all_meet"] = all(s["meets_plan"] for s in r["plan"])
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--video")
    a = ap.parse_args()
    r = run(a.model, a.seed, video=a.video)
    rules, info = judge(r)
    print(json.dumps({"model": Path(a.model).name, "seed": a.seed, "mu": round(r["mu"], 3), "rules": rules,
                      **{k: v for k, v in info.items() if not k.startswith("joint_")}}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in r.items() if k != "log"}
        Path(a.out).write_text(json.dumps({**slim, "rules": rules, "info": info}, indent=1) + "\n")


if __name__ == "__main__":
    main()
