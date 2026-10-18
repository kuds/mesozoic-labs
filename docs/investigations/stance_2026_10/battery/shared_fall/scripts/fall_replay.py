"""Shared-fall replay of one panel seed at the six certificate headings, with per-step balance telemetry.

OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen). Report only;
certifies nothing; no training.

Rolls exactly the certificate's episode (stance_report.run_panel/_roll_episodes conventions): main's stance stage
config (options off), a fresh env per heading, reset(seed), the whole-scene SpawnYaw turn, the deterministic
policy through stance_report._load_policy (frozen VecNormalize) or the zero-action statue, and main's floor-truth
SubstepContactRecorder attached for the whole episode. A second read-only substep probe is chained under the
recorder (it only reads model/data; mj_contactForce / mj_jacSubtreeCom write only their own output buffers), and
the boundary kinematics are computed on a scratch MjData copy, so env.data is never written.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True, help="label, e.g. s60_6M, s52_6M, statue")
ap.add_argument("--model", default=None)
ap.add_argument("--vecnorm", default=None)
ap.add_argument("--seed", type=int, default=7047)
ap.add_argument("--seeds", default=None, help="comma list; overrides --seed (context runs)")
ap.add_argument("--yaws", default="-90,-45,0,45,90,180")
ap.add_argument("--full-yaws", default="0", help="headings whose full traces are kept")
ap.add_argument("--out", required=True)
args = ap.parse_args()

import torch  # noqa: E402

torch.set_num_threads(1)
import mujoco  # noqa: E402

import environments  # noqa: E402

assert "main_ro" in environments.__file__, environments.__file__
from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.gait.constants import CONTACT_THRESHOLD_N  # noqa: E402
from environments.shared.gait.morphology import Morphology  # noqa: E402
from environments.shared.gait.recorder import SubstepContactRecorder, _digit_axis_ends  # noqa: E402
from environments.shared.gait.stance_metrics import episode_stance_metrics  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity, validate_environment_plant  # noqa: E402
from environments.shared.reporting.stance_report import (  # noqa: E402
    SpawnYaw,
    _apply_spawn_yaw,
    _load_policy,
    _root_yaw,
    _zero_action_predict,
)
from environments.shared.species_registry import env_class as model_env_class  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


MID = ModelId("tyrannosaurus_rex", "anatomical")
CFG = load_stage_config(MID, 1)
assert CFG["curriculum_kwargs"]["gate_kind"] == "stance_quality_heading/v1"
ENV_KW = dict(CFG["env_kwargs"])
SETTLE = int(CFG["curriculum_kwargs"]["settle_steps"])
ENV_CLASS = model_env_class(MID)
PLANT = current_plant_identity(MID)
G = 9.81

if args.model is None:
    predict = _zero_action_predict(ENV_CLASS, ENV_KW)
    policy_note = "zero-action statue"
else:
    predict, policy_note = _load_policy(args.model, args.vecnorm, lambda: ENV_CLASS(**ENV_KW), plant_identity=PLANT)


# ----------------------------------------------------------------------------- 2-D convex geometry
def hull(points):
    pts = sorted(set(map(tuple, np.round(np.asarray(points, dtype=np.float64), 12))))
    if len(pts) < 3:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)  # counter-clockwise


def seg_dist(p, a, b):
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def margins(poly, q):
    """(signed distance to polygon, + inside; front extent - q_fwd; q_fwd - rear extent; left - q_lat; q_lat - right).

    Coordinates are (fwd, lat). Extents are the polygon's chord through q along fwd (at q's lateral coordinate) and
    along lat (at q's fwd coordinate); NaN where that chord misses the polygon.
    """
    out = [np.nan] * 5
    if poly.shape[0] < 3 or not np.all(np.isfinite(q)):
        return out
    n = poly.shape[0]
    inside = True
    dmin = np.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) < 0:
            inside = False
        dmin = min(dmin, seg_dist(q, a, b))
    out[0] = dmin if inside else -dmin
    for axis, (i_hi, i_lo) in ((1, (1, 2)), (0, (3, 4))):
        # chord along the OTHER axis through q: intersect the line coord[axis] == q[axis]
        other = 1 - axis
        hits = []
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            if (a[axis] - q[axis]) * (b[axis] - q[axis]) <= 0 and a[axis] != b[axis]:
                t = (q[axis] - a[axis]) / (b[axis] - a[axis])
                hits.append(a[other] + t * (b[other] - a[other]))
            elif a[axis] == b[axis] == q[axis]:
                hits += [a[other], b[other]]
        if hits:
            out[i_hi] = max(hits) - q[other]
            out[i_lo] = q[other] - min(hits)
    return out


# ----------------------------------------------------------------------------- the read-only substep probe
class Probe:
    def __init__(self, env, morph):
        u = env.unwrapped
        self.u, self.m = u, u.model
        m = self.m
        self.morph = morph
        self.nf = morph.n_feet
        self.soles = [f.sole_geom for f in morph.feet]
        self.lo = m.actuator_forcerange[:, 0].copy()
        self.hi = m.actuator_forcerange[:, 1].copy()
        self.f6 = np.zeros(6)
        self.reset()

    def reset(self):
        self.sub = []  # per substep dicts

    def __call__(self):
        m, d = self.m, self.u.data
        nf = self.nf
        fn = np.zeros(nf)  # normal force sum (contact normal)
        fw = np.zeros((nf, 3))  # world force on the foot
        ratio_max = np.zeros(nf)  # max per-contact |ft| / (mu fn) over loaded contacts
        ratio_wsum = np.zeros(nf)
        lat_num = np.zeros(nf)  # whole-limb lateral CoP numerator (sole frame, left +, m from pad centre)
        fwd_num = np.zeros(nf)  # whole-limb fore-aft numerator (sole frame, m from pad centre)
        net = np.zeros(3)  # sum fz, sum fz*x, sum fz*y over every animal floor contact
        nonleg = 0.0
        pts = []
        ncon = int(d.ncon)
        if ncon:
            cg = np.asarray(d.contact.geom).copy()
            cpos = np.asarray(d.contact.pos).copy()
            cframe = np.asarray(d.contact.frame).copy()
            cfric = np.asarray(d.contact.friction).copy()
        for i in range(ncon):
            g1, g2 = int(cg[i, 0]), int(cg[i, 1])
            f1, f2 = bool(self.morph.is_floor[g1]), bool(self.morph.is_floor[g2])
            if f1 == f2:
                continue
            other = g2 if f1 else g1
            if not self.morph.is_animal[other]:
                continue
            mujoco.mj_contactForce(m, d, i, self.f6)
            normal = float(self.f6[0])
            frame = cframe[i].reshape(3, 3)
            world = frame.T @ self.f6[:3]
            if not f1:  # animal is geom1: force on the animal is the reaction
                world = -world
            pos = cpos[i]
            net += (world[2], world[2] * pos[0], world[2] * pos[1])
            foot = int(self.morph.geom_foot[other])
            if foot < 0:
                nonleg += normal
                continue
            fn[foot] += normal
            fw[foot] += world
            if normal > CONTACT_THRESHOLD_N:
                mu = float(cfric[i, 0])
                r = math.hypot(self.f6[1], self.f6[2]) / (mu * normal)
                ratio_max[foot] = max(ratio_max[foot], r)
                ratio_wsum[foot] += r * normal
                pts.append((pos[0], pos[1]))
            sole = self.soles[foot]
            R = d.geom_xmat[sole].reshape(3, 3)
            local = R.T @ (pos - d.geom_xpos[sole])
            fwd_num[foot] += normal * local[0]
            lat_num[foot] += normal * local[1]
        af = d.actuator_force.copy()
        lim = np.where(af >= 0.0, self.hi, -self.lo)
        self.sub.append(
            dict(fn=fn, fw=fw, rmax=ratio_max, rws=ratio_wsum, lat=lat_num, fwd=fwd_num, net=net, nonleg=nonleg,
                 af=af, afrac=np.abs(af) / lim, pts=pts)
        )

    def close_step(self, frame_skip):
        rows = self.sub[-frame_skip:]
        self.sub = []
        fn = np.mean([r["fn"] for r in rows], axis=0)
        fw = np.mean([r["fw"] for r in rows], axis=0)
        rws = np.sum([r["rws"] for r in rows], axis=0)
        fn_sum = np.sum([r["fn"] for r in rows], axis=0)
        lat = np.sum([r["lat"] for r in rows], axis=0)
        fwd = np.sum([r["fwd"] for r in rows], axis=0)
        net = np.sum([r["net"] for r in rows], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cop_lat = np.where(fn_sum > 1e-9, lat / fn_sum, np.nan)
            cop_fwd = np.where(fn_sum > 1e-9, fwd / fn_sum, np.nan)
            ratio_mean = np.where(fn_sum > 1e-9, rws / fn_sum, np.nan)
        afrac = np.array([r["afrac"] for r in rows])
        return dict(
            fn=fn,
            fw=fw,
            ratio_max=np.max([r["rmax"] for r in rows], axis=0),
            ratio_mean=ratio_mean,
            cop_lat_m=cop_lat,
            cop_fwd_pad_m=cop_fwd,
            net_cop_xy=(net[1:] / net[0]) if net[0] > 1e-9 else np.array([np.nan, np.nan]),
            net_fz=net[0] / len(rows),
            nonleg=float(np.mean([r["nonleg"] for r in rows])),
            af_last=rows[-1]["af"],
            afrac_max=afrac.max(axis=0),
            at_limit_frac=(afrac >= 0.999).mean(axis=0),
            pts_last=rows[-1]["pts"],
        )


# ----------------------------------------------------------------------------- one episode
def roll(yaw, seed):
    env = ENV_CLASS(**ENV_KW)
    validate_environment_plant(env, PLANT, artifact=f"{MID} shared-fall replay environment")
    u = env.unwrapped
    m = u.model
    morph = Morphology.from_env(env, MID)
    probe = Probe(env, morph)
    u._substep_probe_hook = probe  # chained under the recorder (it calls the previous hook first)
    recorder = SubstepContactRecorder(env, morph).attach()
    scratch = mujoco.MjData(m)
    root = morph.root_body
    jac = np.zeros((3, m.nv))
    key = int(getattr(u, "_reset_keyframe_id", 0))
    home_qpos = m.key_qpos[key].copy()
    act_jnt = m.actuator_trnid[:, 0].copy()
    act_qadr = m.jnt_qposadr[act_jnt]
    kp = m.actuator_gainprm[:, 0].copy()
    digit_ids = np.array([g for g, j in zip(morph.support_geom_ids, morph.support_geom_mtp) if j >= 0])
    digit_foot = np.array([morph.geom_foot[g] for g in digit_ids])
    _, digit_tip = _digit_axis_ends(m, digit_ids, [j for j in morph.support_geom_mtp if j >= 0])
    half = [m.geom_size[f.sole_geom].copy() for f in morph.feet]
    head_site = int(u.head_tip_site_id)
    groups = {"trunk_arms": ["pelvis", "r_arm", "r_forearm", "l_arm", "l_forearm"], "neck_head": ["neck", "skull"],
              "tail": ["tail_1", "tail_2", "tail_3", "tail_4", "tail_5"],
              "r_leg": ["r_thigh", "r_tibia", "r_metatarsus", "r_toe_d2", "r_toe_d3", "r_toe_d4"],
              "l_leg": ["l_thigh", "l_tibia", "l_metatarsus", "l_toe_d2", "l_toe_d3", "l_toe_d4"]}
    group_ids = {g: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bs] for g, bs in groups.items()}
    assert all(i >= 0 for ids in group_ids.values() for i in ids)
    group_mass = {g: float(m.body_mass[ids].sum()) for g, ids in group_ids.items()}
    skull = int(u.skull_body_id)
    rec = {k: [] for k in (
        "com", "vcom", "xcom", "com_hf", "vcom_hf", "xcom_hf", "omega0", "pelvis_pos", "pitch_down_deg", "roll_deg",
        "yaw_change_deg", "forward_z", "head_tip_z", "skull_z", "qpos_j", "qvel_j", "action", "action_applied", "ctrl",
        "af_last", "afrac_max", "at_limit_frac", "servo_demand", "fn", "fw_hf", "ratio_max", "ratio_mean",
        "cop_lat_m", "cop_fwd_pad_m", "net_cop_hf", "nonleg_probe", "pad_heel_hf", "pad_toe_hf", "pad_corners_hf",
        "digit_tips_hf", "m_xcom_pad", "m_xcom_full", "m_xcom_contact", "m_com_pad", "m_cop_pad", "m_cop_full",
        "reward", "root_qpos", "com_groups_hf", "pad_center_hf",
    )}
    try:
        obs, _ = env.reset(seed=seed)
        obs = _apply_spawn_yaw(env, SpawnYaw(yaw, whole_scene=True))
        psi0 = _root_yaw(env)
        fwd_axis = np.array([math.cos(psi0), math.sin(psi0)])
        left_axis = np.array([-math.sin(psi0), math.cos(psi0)])
        origin = 0.5 * (u.data.geom_xpos[morph.feet[0].sole_geom, :2] + u.data.geom_xpos[morph.feet[1].sole_geom, :2])
        spawn_qpos = u.data.qpos.copy()
        spawn_qvel = u.data.qvel.copy()

        def hf(xy):
            xy = np.asarray(xy, dtype=np.float64) - origin
            return np.stack([xy @ fwd_axis, xy @ left_axis], axis=-1)

        def hfv(v):
            v = np.asarray(v, dtype=np.float64)
            return np.stack([v @ fwd_axis, v @ left_axis], axis=-1)

        total, steps, reason = 0.0, 0, ""
        yaw_prev, yaw_turned = psi0, 0.0
        terminated = truncated = False
        while True:
            action = np.asarray(predict(obs), dtype=np.float64).ravel()
            obs, reward, terminated, truncated, info = env.step(action)
            recorder.end_step(action, float(reward))
            step_probe = probe.close_step(u.frame_skip)
            total += float(reward)
            steps += 1
            yaw_now = _root_yaw(env)
            yaw_turned += math.remainder(yaw_now - yaw_prev, 2 * math.pi)
            yaw_prev = yaw_now
            # boundary kinematics on scratch
            mujoco.mj_copyData(scratch, m, u.data)
            mujoco.mj_forward(m, scratch)
            com = scratch.subtree_com[root].copy()
            mujoco.mj_jacSubtreeCom(m, scratch, jac, root)
            vcom = jac @ scratch.qvel
            l = max(float(com[2]), 1e-6)
            w0 = math.sqrt(G / l)
            xcom = com[:2] + vcom[:2] / w0
            R = scratch.xmat[root].reshape(3, 3)
            q = scratch.qpos
            # feet geometry (boundary)
            corners, heel, toe = [], [], []
            for fi, f in enumerate(morph.feet):
                Rs = scratch.geom_xmat[f.sole_geom].reshape(3, 3)
                c0 = scratch.geom_xpos[f.sole_geom]
                hx, hy, hz = half[fi]
                cs = [c0 + Rs @ np.array([sx * hx, sy * hy, -hz]) for sx in (-1, 1) for sy in (-1, 1)]
                corners.append(hf(np.array(cs)[:, :2]))
                heel.append(hf((c0 + Rs @ np.array([-hx, 0, -hz]))[:2]))
                toe.append(hf((c0 + Rs @ np.array([hx, 0, -hz]))[:2]))
            tips = []
            for gi, g in enumerate(digit_ids):
                Rg = scratch.geom_xmat[g].reshape(3, 3)
                tips.append(hf((scratch.geom_xpos[g] + Rg[:, 2] * digit_tip[gi])[:2]))
            loaded = step_probe["fn"] > CONTACT_THRESHOLD_N
            pad_pts = [corners[fi] for fi in range(morph.n_feet) if loaded[fi]]
            pad_poly = hull(np.concatenate(pad_pts)) if pad_pts else np.zeros((0, 2))
            full_pts = pad_pts + [tips[i][None] for i in range(len(tips)) if loaded[digit_foot[i]]]
            full_poly = hull(np.concatenate(full_pts)) if full_pts else np.zeros((0, 2))
            cpts = step_probe["pts_last"]
            contact_poly = hull(hf(np.array(cpts))) if len(cpts) >= 3 else np.zeros((0, 2))
            xcom_hf = hf(xcom)
            com_hf = hf(com[:2])
            net_cop_hf = hf(step_probe["net_cop_xy"])
            rec["com"].append(com)
            rec["vcom"].append(vcom)
            rec["xcom"].append(xcom)
            rec["com_hf"].append(np.r_[com_hf, com[2]])
            rec["vcom_hf"].append(np.r_[hfv(vcom[:2]), vcom[2]])
            rec["xcom_hf"].append(xcom_hf)
            rec["omega0"].append(w0)
            rec["pelvis_pos"].append(scratch.xpos[root].copy())
            rec["pitch_down_deg"].append(-math.degrees(math.asin(float(np.clip(R[2, 0], -1, 1)))))
            rec["roll_deg"].append(math.degrees(math.atan2(R[2, 1], R[2, 2])))  # + = left side up
            rec["yaw_change_deg"].append(math.degrees(yaw_turned))
            rec["forward_z"].append(float(R[2, 0]))
            rec["head_tip_z"].append(float(u._clearance(scratch.site_xpos[head_site])))
            rec["skull_z"].append(float(u._clearance(scratch.xpos[skull])))
            rec["qpos_j"].append(q[7:].copy())
            rec["qvel_j"].append(scratch.qvel[6:].copy())
            rec["action"].append(np.clip(action, -1, 1))
            st = getattr(u, "_action_filter_state", None)
            rec["action_applied"].append(np.full(m.nu, np.nan) if st is None else np.asarray(st).copy())
            rec["ctrl"].append(scratch.ctrl.copy())
            rec["af_last"].append(step_probe["af_last"])
            rec["afrac_max"].append(step_probe["afrac_max"])
            rec["at_limit_frac"].append(step_probe["at_limit_frac"])
            rec["servo_demand"].append(kp * (scratch.ctrl - q[act_qadr]))
            rec["fn"].append(step_probe["fn"])
            rec["fw_hf"].append(np.array([np.r_[hfv(v[:2]), v[2]] for v in step_probe["fw"]]))
            rec["ratio_max"].append(step_probe["ratio_max"])
            rec["ratio_mean"].append(step_probe["ratio_mean"])
            rec["cop_lat_m"].append(step_probe["cop_lat_m"])
            rec["cop_fwd_pad_m"].append(step_probe["cop_fwd_pad_m"])
            rec["net_cop_hf"].append(net_cop_hf)
            rec["nonleg_probe"].append(step_probe["nonleg"])
            rec["pad_heel_hf"].append(np.array(heel))
            rec["pad_toe_hf"].append(np.array(toe))
            rec["pad_corners_hf"].append(np.array(corners))
            rec["digit_tips_hf"].append(np.array(tips))
            rec["m_xcom_pad"].append(margins(pad_poly, xcom_hf))
            rec["m_xcom_full"].append(margins(full_poly, xcom_hf))
            rec["m_xcom_contact"].append(margins(contact_poly, xcom_hf))
            rec["m_com_pad"].append(margins(pad_poly, com_hf))
            rec["m_cop_pad"].append(margins(pad_poly, net_cop_hf))
            rec["m_cop_full"].append(margins(full_poly, net_cop_hf))
            rec["reward"].append(float(reward))
            rec["root_qpos"].append(q[:7].copy())
            rec["com_groups_hf"].append(np.array([
                hf((m.body_mass[ids][:, None] * scratch.xipos[ids][:, :2]).sum(0) / m.body_mass[ids].sum())
                for ids in group_ids.values()]))
            rec["pad_center_hf"].append(hf(np.array([scratch.geom_xpos[f.sole_geom, :2] for f in morph.feet])))
            if terminated or truncated:
                reason = info.get("termination_reason", "terminated" if terminated else "truncated")
                trace = recorder.end_episode(terminated=bool(terminated), truncated=bool(truncated),
                                             termination_reason=str(reason))
                break
    finally:
        try:
            recorder.detach()
        finally:
            env.close()
    metrics = episode_stance_metrics(trace, settle_steps=SETTLE)
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    for name in ("leg_floor_mean", "support_floor_mean", "offsupport_floor_mean", "nonleg_floor_mean", "sole_cop",
                 "foot_cop", "sole_tilt_deg", "sole_roll_deg", "sole_pitch_deg", "sole_corner_lift", "sole_min_height",
                 "sole_contacts_mean", "support_load_pos", "touch", "foot_pos", "joint_qpos", "ctrl", "action"):
        arrays["rec_" + name] = np.asarray(getattr(trace, name), dtype=np.float64)
    meta = dict(
        seed=seed, yaw_deg=yaw, length=steps, reward=total, terminated=bool(terminated), truncated=bool(truncated),
        termination_reason=str(reason), psi0_deg=math.degrees(psi0), episode_yaw_change_deg=math.degrees(yaw_turned),
        metrics={k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                 for k, v in dataclasses.asdict(metrics).items() if isinstance(v, (int, float, bool, str))},
        spawn_joint_offset_from_home=(spawn_qpos[7:] - home_qpos[7:]).tolist(),
        spawn_qvel=spawn_qvel.tolist(),
    )
    names = dict(
        joint_names=[m.joint(int(j)).name for j in range(m.njnt) if int(m.jnt_type[j]) != 0],
        actuator_names=[m.actuator(i).name for i in range(m.nu)],
        support_geom_names=list(trace.support_geom_names),
        digit_geom_names=[m.geom(int(g)).name for g in digit_ids],
        home_qpos_j=home_qpos[7:].tolist(),
        home_ctrl=np.asarray(u._home_ctrl).tolist(),
        actuator_qposadr_minus7=(act_qadr - 7).tolist(),
        forcerange=m.actuator_forcerange.tolist(),
        ctrlrange=m.actuator_ctrlrange.tolist(),
        kp=kp.tolist(),
        foot_labels=list(morph.labels),
        pad_half_size=[h.tolist() for h in half],
        heel_toe_line_from_ankle_m=[[f.heel_line_m, f.toe_line_m] for f in morph.feet],
        ankle_forward_in_sole_m=[f.ankle_forward_m for f in morph.feet],
        body_weight_n=morph.body_weight_n,
        com_groups=list(groups), com_group_mass=group_mass, total_mass=float(m.body_subtreemass[root]),
        control_dt=float(u.dt),
        frame_skip=int(u.frame_skip),
        settle_steps=SETTLE,
        natural_forward_z=float(u._natural_forward_z),
        nosedive_threshold=float(u.nosedive_termination_threshold),
        head_contact_threshold_m=0.12,
        skull_low_threshold_m=0.45,
    )
    return meta, arrays, names


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
yaws = [float(y) for y in args.yaws.split(",")]
full = {float(y) for y in args.full_yaws.split(",")} if args.full_yaws else set()
summary = dict(
    label="OFFLINE EVALUATION of an uncertified scratch-recipe checkpoint; report only, certifies nothing",
    role=args.role, seed=args.seed, policy=policy_note, environments_file=environments.__file__,
    model=args.model, model_sha256=None if args.model is None else sha(args.model),
    vecnorm=args.vecnorm, vecnorm_sha256=None if args.vecnorm is None else sha(args.vecnorm),
    env_overrides={}, settle_steps=SETTLE, episodes=[],
)
compact = {}
t0 = time.time()
seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]
for seed in seeds:
    for yaw in yaws:
        meta, arrays, names = roll(yaw, seed)
        summary["episodes"].append(meta)
        summary["names"] = names
        tag = f"y{int(yaw):+d}" if len(seeds) == 1 else f"s{seed}_y{int(yaw):+d}"
        if yaw in full:
            np.savez_compressed(out / f"trace_{args.role}_{tag}.npz",
                                **{k: v.astype(np.float32) for k, v in arrays.items()}, reward_f64=arrays["reward"])
        keys = ("com_hf", "xcom_hf", "m_xcom_pad", "pitch_down_deg", "head_tip_z", "forward_z", "qpos_j", "action")
        if len(seeds) > 1:
            keys = ("com_hf", "xcom_hf", "m_xcom_pad", "m_xcom_full", "pitch_down_deg", "pad_center_hf", "rec_sole_cop",
                    "rec_sole_pitch_deg", "rec_foot_cop")
        for k in keys:
            compact[f"{tag}_{k}"] = arrays[k]
        print(json.dumps({k: meta[k] for k in ("seed", "yaw_deg", "length", "reward", "termination_reason")}), flush=True)
np.savez_compressed(out / f"headings_{args.role}.npz", **{k: v.astype(np.float32) for k, v in compact.items()})
summary["seconds"] = time.time() - t0
(out / f"replay_{args.role}.json").write_text(json.dumps(summary, indent=1, default=str))
print("done", args.role, round(summary["seconds"], 1))
