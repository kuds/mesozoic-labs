"""Independent posture / load re-measurement: the statue settle and one r9 arm L checkpoint (report-only).

    python -P vx_roll.py --out OUT_JSON --target statue|s60 [--seeds ...] [--noise X]

Builds the env from main's unmodified stance stage config (options off), runs deterministic episodes
(statue = zero action = home control; s60 = the 6,291,456-step checkpoint, policy mean, frozen
VecNormalize), and after every env.step reads the boundary state on env.data (read-only) with its OWN
contact loop (mj_contactForce on every floor contact): pad-only CoP and whole-foot CoP in pad half-lengths
along the pad's forward axis, digit share, COM vs hip, COM projection, femur / knee / ankle / MTP angles
(the same sagittal sweep conventions as vx_static.py), pelvis pitch, the knee's fore-aft offset from the
vertical through the whole-foot CoP, tail/neck/head joint angles.  Window means over steps [200, T).

Optionally (statue only, --tail-test) copies the settled state into a private MjData of the env's model,
commands tail pitch to the top of its ctrlrange, steps 3 s of raw physics with the other controls held,
and reports tail_1 pitch and the holding budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

import environments  # noqa: E402,F401
from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.reporting import stance_report as SR  # noqa: E402
from environments.shared.species_names import require_model_id  # noqa: E402

S = "scratchpad"
RUNS = f"{S}/foot/symmetry/screen/runs"
CKPT = {
    "s60": (f"{RUNS}/sym_lean_settled_s60/resume6m/models/stage1_6291456_steps.zip",
            f"{RUNS}/sym_lean_settled_s60/resume6m/models/stage1_vecnormalize_6291456_steps.pkl"),
}
G = 9.81
WIN0 = 200


def sweep(a, b):
    return math.degrees((math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])) % (2 * math.pi))


class Probe:
    def __init__(self, env):
        m = env.unwrapped.model if hasattr(env, "unwrapped") else env.model
        self.m = m
        n2 = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
        J, B, Gm = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
        self.pel = n2(B, "pelvis")
        self.floor = n2(Gm, "floor")
        self.j = {s: {k: n2(J, f"{s}_{k}") for k in ("hip_pitch", "knee", "ankle", "toe_d3_joint")} for s in "rl"}
        self.pad = {s: n2(Gm, f"{s}_plantar_geom") for s in "rl"}
        self.toe = {s: {n2(Gm, f"{s}_toe_d{k}_geom"): k for k in (2, 3, 4)} for s in "rl"}
        self.d3 = {s: n2(Gm, f"{s}_toe_d3_geom") for s in "rl"}
        self.qadr = {n: m.jnt_qposadr[n2(J, n)] for n in ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch", "neck_pitch",
                                                         "head_pitch")}
        self.M = float(m.body_subtreemass[self.pel])
        self.rows = []

    def sample(self, d):
        m = self.m
        row = {}
        R = d.xmat[self.pel].reshape(3, 3)
        px = R[:, 0]
        fwd = np.array([px[0], px[1], 0.0])
        fwd /= np.linalg.norm(fwd)
        row["pelvis_pitch_nose_down_deg"] = math.degrees(math.asin(-px[2]))
        row["pelvis_z"] = float(d.xpos[self.pel][2])
        com = d.subtree_com[self.pel].copy()
        hips = {s: d.xanchor[self.j[s]["hip_pitch"]].copy() for s in "rl"}
        hip_mid = 0.5 * (hips["r"] + hips["l"])
        # contacts
        f6 = np.zeros(6)
        acc = {s: {"pad": [0.0, np.zeros(3)], "dig": [0.0, np.zeros(3)], "dk": {2: 0.0, 3: 0.0, 4: 0.0}} for s in "rl"}
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.floor else c.geom1
            mujoco.mj_contactForce(m, d, i, f6)
            fz = float(f6[0])  # normal force (floor normal is +z)
            for s in "rl":
                if other == self.pad[s]:
                    acc[s]["pad"][0] += fz
                    acc[s]["pad"][1] += fz * c.pos
                elif other in self.toe[s]:
                    acc[s]["dig"][0] += fz
                    acc[s]["dig"][1] += fz * c.pos
                    acc[s]["dk"][self.toe[s][other]] += fz
        fem = None
        for s in "rl":
            hip = hips[s]
            knee = d.xanchor[self.j[s]["knee"]].copy()
            ankle = d.xanchor[self.j[s]["ankle"]].copy()
            mtp = d.xanchor[self.j[s]["toe_d3_joint"]].copy()
            g = self.d3[s]
            zc = d.geom_xmat[g].reshape(3, 3)[:, 2]
            e0 = d.geom_xpos[g] - zc * m.geom_size[g][1]
            e1 = d.geom_xpos[g] + zc * m.geom_size[g][1]
            end3 = e1 if np.linalg.norm(e1 - mtp) > np.linalg.norm(e0 - mtp) else e0
            pc = d.geom_xpos[self.pad[s]].copy()
            pf = d.geom_xmat[self.pad[s]].reshape(3, 3)[:, 0].copy()
            hp = float(m.geom_size[self.pad[s]][0])
            # horizontal pad heading: a COM ~0.84 m up must be projected on the FLOOR, not along a pad tilted ~0.08 deg
            pfh = np.array([pf[0], pf[1], 0.0]) / np.linalg.norm(pf[:2])

            def xz(v):
                return np.array([float(np.dot(v, fwd)), float(v[2])])

            fem = float(np.linalg.norm(knee - hip))
            fv = xz(knee - hip)
            row[f"{s}_femur_from_vertical_deg"] = math.degrees(math.atan2(fv[0], -fv[1]))
            row[f"{s}_knee_incl_deg"] = sweep(xz(hip - knee), xz(ankle - knee))
            row[f"{s}_ankle_incl_deg"] = sweep(xz(mtp - ankle), xz(knee - ankle))
            row[f"{s}_mtp_incl_deg"] = sweep(xz(end3 - mtp), xz(ankle - mtp))
            mv = xz(mtp - ankle)
            row[f"{s}_metatarsus_from_vertical_deg"] = math.degrees(math.atan2(mv[0], -mv[1]))
            row[f"{s}_com_padstat"] = float(np.dot(com - pc, pfh)) / hp
            row[f"{s}_com_padstat_along_tilted_pad_axis"] = float(np.dot(com - pc, pf)) / hp
            row[f"{s}_pad_pitch_deg"] = math.degrees(math.asin(pf[2]))
            fp = float(np.dot(end3 + (end3 - mtp) / np.linalg.norm(end3 - mtp) * m.geom_size[g][0] - (pc - pf * hp), pf))
            row[f"{s}_com_footfrac"] = float(np.dot(com - (pc - pf * hp), pfh)) / fp
            Fp, Fd = acc[s]["pad"][0], acc[s]["dig"][0]
            row[f"{s}_Fz_pad"] = Fp
            row[f"{s}_Fz_dig"] = Fd
            row[f"{s}_digit_share"] = Fd / (Fp + Fd) if Fp + Fd > 1e-6 else float("nan")
            row[f"{s}_pad_cop_stat"] = (float(np.dot(acc[s]["pad"][1] / Fp - pc, pf)) / hp) if Fp > 1e-6 else float("nan")
            if Fp + Fd > 1e-6:
                cop = (acc[s]["pad"][1] + acc[s]["dig"][1]) / (Fp + Fd)
                row[f"{s}_wholefoot_cop_stat"] = float(np.dot(cop - pc, pf)) / hp
                row[f"{s}_knee_ahead_of_vcop_m"] = float(np.dot(knee - cop, fwd))
                row[f"{s}_hip_ahead_of_vcop_m"] = float(np.dot(hip - cop, fwd))
            else:
                row[f"{s}_wholefoot_cop_stat"] = float("nan")
            if Fd > 1e-6:
                dc = acc[s]["dig"][1] / Fd
                row[f"{s}_digit_load_ahead_of_mtp3_m"] = float(np.dot(dc - mtp, pf))
            row[f"{s}_knee_ahead_of_pad_centre_m"] = float(np.dot(knee - pc, fwd))
        row["femur_m"] = fem
        row["com_ahead_hip_pct_femur"] = 100 * float(np.dot(com - hip_mid, fwd)) / fem
        row["com_above_hip_pct_femur"] = 100 * float(com[2] - hip_mid[2]) / fem
        row["com_z"] = float(com[2])
        row["mgh_floor"] = self.M * G * float(com[2])
        for n, a in self.qadr.items():
            row[f"q_{n}_deg"] = math.degrees(float(d.qpos[a]))
        self.rows.append(row)


def window_means(rows, w0=WIN0):
    keys = list(dict.fromkeys(k for r in rows for k in r))  # union: a key can be absent on an unloaded step
    out = {}
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in rows[w0:]], dtype=float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        out[k + "__min"] = float(np.nanmin(v)) if np.isfinite(v).any() else float("nan")
        out[k + "__max"] = float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")
    # the gate's own per-foot form: window mean of per-step |pad CoP| (NaN -> 1.0 as the gate does)
    for s in "rl":
        v = np.array([r[f"{s}_pad_cop_stat"] for r in rows[w0:]], dtype=float)
        out[f"{s}_gate_mean_abs_pad_cop"] = float(np.where(np.isfinite(v), np.abs(v), 1.0).mean())
    out["gate_stat_worse_foot"] = max(out["r_gate_mean_abs_pad_cop"], out["l_gate_mean_abs_pad_cop"])
    return out


def tail_test(env, d_settled):
    m = env.unwrapped.model
    d = mujoco.MjData(m)
    d.qpos[:] = d_settled.qpos
    d.qvel[:] = d_settled.qvel
    d.ctrl[:] = d_settled.ctrl
    d.act[:] = d_settled.act if m.na else d.act
    mujoco.mj_forward(m, d)
    a = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "tail_1_pitch")
    q1 = m.jnt_qposadr[j1]
    res = {"tail_1_settled_deg": math.degrees(float(d.qpos[q1])),
           "tail_1_range_deg": [math.degrees(x) for x in m.jnt_range[j1]],
           "ctrl_before": {n: float(d.ctrl[i]) for n, i in a.items() if n.startswith("tail")}}
    for n in ("tail_1_pitch_act", "tail_2_pitch_act", "tail_3_pitch_act"):
        d.ctrl[a[n]] = m.actuator_ctrlrange[a[n]][1]
    for _ in range(1500):  # 3 s
        mujoco.mj_step(m, d)
    q = float(d.qpos[q1])
    i1 = a["tail_1_pitch_act"]
    res["tail_1_after_full_up_3s_deg"] = math.degrees(q)
    res["tail_1_servo_torque_Nm"] = float(d.actuator_force[i1])
    res["tail_1_spring_torque_Nm"] = float(-m.jnt_stiffness[j1] * (q - m.qpos_spring[q1]))
    # gravity torque of the tail subtree about the tail_1 axis
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tail_1")
    tau = np.cross(d.subtree_com[b] - d.xanchor[j1], np.array([0, 0, -m.body_subtreemass[b] * G]))
    res["tail_1_gravity_torque_Nm"] = float(np.dot(tau, d.xaxis[j1]))
    res["tail_1_limit_constraint_active"] = bool(any(
        d.efc_type[k] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT and d.efc_id[k] == j1 for k in range(d.nefc)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="statue")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--noise", type=float, default=None, help="override reset_noise_scale (statue: 0)")
    ap.add_argument("--tail-test", action="store_true")
    args = ap.parse_args()
    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    cfg = load_stage_config(mid, "stance")
    env_kwargs = dict(cfg["env_kwargs"])
    assert float(env_kwargs.get("lateral_lean_weight", 0.0)) == 0.0
    assert env_kwargs.get("leg_home_pose_reference", "keyframe") == "keyframe"
    if args.noise is not None:
        env_kwargs["reset_noise_scale"] = args.noise
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    env_class = SR.model_env_class(mid)
    if args.target == "statue":
        predict = SR._zero_action_predict(env_class, env_kwargs)
        note = "zero action"
    else:
        zp, vp = CKPT[args.target]
        predict, note = SR._load_policy(zp, vp, lambda: env_class(**env_kwargs),
                                        plant_identity=current_plant_identity(mid), allow_legacy_plant=False)
    env = env_class(**env_kwargs)
    result = {"target": args.target, "note": str(note), "reset_noise_scale": env_kwargs.get("reset_noise_scale"),
              "horizon": horizon, "window": [WIN0, horizon], "episodes": {}}
    for seed in args.seeds:
        obs, _ = env.reset(seed=seed)
        probe = Probe(env)
        steps, term, trunc = 0, False, False
        while True:
            a = np.asarray(predict(obs), dtype=np.float64).ravel()
            obs, r, term, trunc, info = env.step(a)
            probe.sample(env.unwrapped.data)
            steps += 1
            if term or trunc or steps >= horizon:
                break
        ep = {"length": steps, "terminated": bool(term), "window_means": window_means(probe.rows),
              "step0": probe.rows[0]}
        if args.tail_test and args.target == "statue":
            ep["tail_test"] = tail_test(env, env.unwrapped.data)
        result["episodes"][str(seed)] = ep
        wm = ep["window_means"]
        print(json.dumps({"seed": seed, "len": steps, "term": bool(term),
                          "femur_r": round(wm["r_femur_from_vertical_deg"], 2),
                          "com_pad_r": round(wm["r_com_padstat"], 3), "padcop_r": round(wm["r_pad_cop_stat"], 3),
                          "gate": round(wm["gate_stat_worse_foot"], 3), "dshare_r": round(wm["r_digit_share"], 3)}),
              flush=True)
    env.close()
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
