"""Independent re-measurement of the r9 T. rex model at the authored home keyframe (kinematics only).

    python -P vx_static.py OUT_JSON

Read-only: loads the MJCF from main_ro, runs mj_forward / mj_kinematics on a private MjData,
never writes anywhere but OUT_JSON.  Conventions are defined here, independently of the sibling audits:

* fore-aft = pelvis heading projected on the floor (+ forward); at the keyframe the pelvis quat is identity.
* included angles are SIGNED sweeps in the sagittal (x-z) plane, so values past 180 deg are reachable:
    knee  : posterior sweep from (hip - knee) to (ankle - knee)
    ankle : anterior sweep from (mtp3 - ankle) to (knee - ankle)
    mtp   : dorsal sweep from (tip3 - mtp3) to (ankle - mtp3)
    hip   : sweep from the femur (knee - hip) up to the pelvis cranial axis (+x)
  Ranges are found by setting ONE joint to each limit (others at home) and re-running kinematics.
* pad statistic = (x - pad_centre) . fwd / pad half-length (signed; the project's statistic is |.|).
"""

from __future__ import annotations

import json
import math
import sys

import mujoco
import numpy as np

MJCF = ("scratchpad/fix/main_ro/"
        "environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml")
G = 9.81


def jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def bid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def gid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)


def sweep(a, b):
    """CCW angle (deg, 0..360) from 2D vector a to 2D vector b in the x-z plane."""
    ta = math.atan2(a[1], a[0])
    tb = math.atan2(b[1], b[0])
    return math.degrees((tb - ta) % (2 * math.pi))


def xz(v, fwd):
    return np.array([float(np.dot(v, fwd)), float(v[2])])


def capsule_ends(m, d, g):
    c = d.geom_xpos[g]
    z = d.geom_xmat[g].reshape(3, 3)[:, 2]
    h = m.geom_size[g][1]
    return c - z * h, c + z * h, m.geom_size[g][0]


def leg(m, d, s, fwd):
    hip = d.xanchor[jid(m, f"{s}_hip_pitch")].copy()
    knee = d.xanchor[jid(m, f"{s}_knee")].copy()
    ankle = d.xanchor[jid(m, f"{s}_ankle")].copy()
    mtp = {k: d.xanchor[jid(m, f"{s}_toe_d{k}_joint")].copy() for k in (2, 3, 4)}
    tips = {}
    for k in (2, 3, 4):
        e0, e1, r = capsule_ends(m, d, gid(m, f"{s}_toe_d{k}_geom"))
        far = e1 if np.linalg.norm(e1 - mtp[k]) > np.linalg.norm(e0 - mtp[k]) else e0
        near = e0 if far is e1 else e1
        u = (far - near) / np.linalg.norm(far - near)
        tips[k] = {"end": far, "tip": far + u * r, "dir": u, "near": near}
    pg = gid(m, f"{s}_plantar_geom")
    pc = d.geom_xpos[pg].copy()
    pR = d.geom_xmat[pg].reshape(3, 3)
    pfwd = pR[:, 0].copy()
    half = m.geom_size[pg].copy()
    rear = pc - pfwd * half[0]
    front = pc + pfwd * half[0]
    out = {
        "hip": hip, "knee": knee, "ankle": ankle, "mtp": mtp, "tips": tips,
        "pad_centre": pc, "pad_fwd": pfwd, "pad_half": half, "pad_rear": rear, "pad_front": front,
    }
    # lengths
    out["femur"] = float(np.linalg.norm(knee - hip))
    out["tibia"] = float(np.linalg.norm(ankle - knee))
    out["metatarsus"] = float(np.linalg.norm(mtp[3] - ankle))
    # included angles (sagittal)
    out["knee_incl"] = sweep(xz(hip - knee, fwd), xz(ankle - knee, fwd))
    out["ankle_incl"] = sweep(xz(mtp[3] - ankle, fwd), xz(knee - ankle, fwd))
    out["mtp_incl"] = sweep(xz(tips[3]["end"] - mtp[3], fwd), xz(ankle - mtp[3], fwd))
    out["hip_incl"] = sweep(xz(knee - hip, fwd), np.array([1.0, 0.0]))
    fv = xz(knee - hip, fwd)
    out["femur_from_vertical"] = math.degrees(math.atan2(fv[0], -fv[1]))
    mv = xz(mtp[3] - ankle, fwd)
    out["metatarsus_from_vertical"] = math.degrees(math.atan2(mv[0], -mv[1]))
    return out


def fa(v, fwd):
    return float(np.dot(v, fwd))


def main(out_path):
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, k)
    mujoco.mj_forward(m, d)
    res = {"mjcf": MJCF, "mujoco": mujoco.__version__}

    pel = bid(m, "pelvis")
    M = float(m.body_subtreemass[pel])
    res["mass_total_kg"] = M
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    bm = {n: float(m.body_mass[i]) for i, n in enumerate(names)}
    res["body_mass_kg"] = bm
    grp = {
        "head(skull)": ["skull"], "neck": ["neck"], "trunk(pelvis body)": ["pelvis"],
        "tail": [n for n in names if n.startswith("tail_")],
        "forelimbs(both)": [n for n in names if "arm" in n],
        "thigh(per side)": ["r_thigh"], "shank(per side)": ["r_tibia"],
        "foot(per side: metatarsus+pad+digits)": ["r_metatarsus", "r_toe_d2", "r_toe_d3", "r_toe_d4"],
        "both hindlimbs": [n for n in names if n[:2] in ("r_", "l_") and "arm" not in n],
    }
    res["mass_share_pct"] = {g: 100 * sum(bm[n] for n in ns) / M for g, ns in grp.items()}
    res["mass_share_members"] = grp

    # heading from pelvis x axis projected on floor
    px = d.xmat[pel].reshape(3, 3)[:, 0]
    fwd = np.array([px[0], px[1], 0.0])
    fwd /= np.linalg.norm(fwd)
    res["pelvis_quat"] = d.xquat[pel].tolist()

    L = {s: leg(m, d, s, fwd) for s in ("r", "l")}
    r = L["r"]
    com = d.subtree_com[pel].copy()
    hip_mid = 0.5 * (L["r"]["hip"] + L["l"]["hip"])
    fem = r["femur"]
    res["lengths_m"] = {"femur": r["femur"], "tibia": r["tibia"], "metatarsus": r["metatarsus"],
                        "femur_l": L["l"]["femur"], "tibia_l": L["l"]["tibia"], "metatarsus_l": L["l"]["metatarsus"]}
    res["ratios"] = {"metatarsus_over_femur": r["metatarsus"] / fem, "tibia_over_femur": r["tibia"] / fem,
                     "limb_over_femur": (r["femur"] + r["tibia"] + r["metatarsus"]) / fem}
    res["com_m"] = com.tolist()
    res["com_vs_hip"] = {
        "ahead_m": fa(com - hip_mid, fwd), "ahead_pct_femur": 100 * fa(com - hip_mid, fwd) / fem,
        "above_m": float(com[2] - hip_mid[2]), "above_pct_femur": 100 * float(com[2] - hip_mid[2]) / fem,
    }
    res["hip_height_m"] = float(hip_mid[2])

    # pad / footprint frame, per foot and mean
    frames = {}
    for s, Lg in L.items():
        pc, pf, half = Lg["pad_centre"], Lg["pad_fwd"], Lg["pad_half"]
        rear, tip3, mtp3 = Lg["pad_rear"], Lg["tips"][3]["tip"], Lg["mtp"][3]
        end3 = Lg["tips"][3]["end"]
        fp = fa(tip3 - rear, pf)
        fr = {
            "pad_half_m": half.tolist(), "pad_centre_ahead_of_ankle_m": fa(pc - Lg["ankle"], pf),
            "pad_tilt_deg": math.degrees(math.acos(min(1.0, abs(float(d.geom_xmat[gid(m, f'{s}_plantar_geom')][8]))))),
            "pad_bottom_z_m": float(pc[2] - half[2]),
            "pad_rear_rel_mtp3_m": fa(rear - mtp3, pf), "pad_front_rel_mtp3_m": fa(Lg["pad_front"] - mtp3, pf),
            "pad_centre_rel_mtp3_m": fa(pc - mtp3, pf),
            "mtp_d2_rel_mtp3_m": fa(Lg["mtp"][2] - mtp3, pf), "mtp_d4_rel_mtp3_m": fa(Lg["mtp"][4] - mtp3, pf),
            "mtp3_height_above_pad_bottom_m": float(mtp3[2] - (pc[2] - half[2])),
            "footprint_rear_to_tip3_m": fp, "footprint_rear_to_end3_m": fa(end3 - rear, pf),
            "digit3_tip_len_from_mtp_m": fa(tip3 - mtp3, pf),
            "digit_len_mtp_to_end_m": {k: float(np.linalg.norm(Lg["tips"][k]["end"] - Lg["mtp"][k])) for k in (2, 3, 4)},
            "digit_len_mtp_to_tip_m": {k: float(np.linalg.norm(Lg["tips"][k]["tip"] - Lg["mtp"][k])) for k in (2, 3, 4)},
            "com_padstat": fa(com - pc, pf) / half[0],
            "com_footfrac": fa(com - rear, pf) / fp,
            "pad_centre_footfrac": fa(pc - rear, pf) / fp,
            "mtp3_padstat": fa(mtp3 - pc, pf) / half[0], "mtp3_footfrac": fa(mtp3 - rear, pf) / fp,
            "pad_centre_frac_of_digit3_tip": fa(pc - mtp3, pf) / fa(tip3 - mtp3, pf),
            "pad_front_frac_of_digit3_tip": fa(Lg["pad_front"] - mtp3, pf) / fa(tip3 - mtp3, pf),
            "knee_ahead_of_pad_centre_m": fa(Lg["knee"] - pc, pf),
            "hip_ahead_of_pad_centre_m": fa(Lg["hip"] - pc, pf),
            "ankle_ahead_of_pad_centre_m": fa(Lg["ankle"] - pc, pf),
        }
        # digit spread in the ground plane: horizontal projection of each digit's near->far axis
        ang = {}
        for kk in (2, 3, 4):
            u = Lg["tips"][kk]["dir"]
            ang[kk] = math.degrees(math.atan2(float(u[1]), float(np.dot(u, pf))))
        fr["digit_heading_deg_left_positive"] = ang
        fr["spread_II_III_deg"] = abs(ang[2] - ang[3])
        fr["spread_III_IV_deg"] = abs(ang[4] - ang[3])
        fr["spread_total_deg"] = abs(ang[2] - ang[4])
        # pose angles
        for key in ("knee_incl", "ankle_incl", "mtp_incl", "hip_incl", "femur_from_vertical", "metatarsus_from_vertical"):
            fr[key] = Lg[key]
        frames[s] = fr
    res["feet"] = frames
    res["com_padstat_mean"] = 0.5 * (frames["r"]["com_padstat"] + frames["l"]["com_padstat"])
    res["com_footfrac_mean"] = 0.5 * (frames["r"]["com_footfrac"] + frames["l"]["com_footfrac"])
    res["footprint_over_hip_height"] = frames["r"]["footprint_rear_to_tip3_m"] / res["hip_height_m"]

    # frame conversions on the right foot (pad frame)
    fr = frames["r"]
    fp, hp = fr["footprint_rear_to_tip3_m"], fr["pad_half_m"][0]
    rear_rel_mtp = fr["pad_rear_rel_mtp3_m"]

    def stat_to(x):  # pad statistic -> (footprint frac, m ahead of MTP III)
        dist_rear = hp + x * hp
        return {"footfrac": dist_rear / fp, "ahead_mtp3_m": rear_rel_mtp + dist_rear}

    def frac_to_stat(f):
        return (f * fp - hp) / hp

    res["frame_conversions"] = {
        "mtp3": {"stat": fr["mtp3_padstat"], "footfrac": fr["mtp3_footfrac"]},
        "pad_centre": stat_to(0.0),
        "human_45.7pct_footfrac": {"stat": frac_to_stat(0.457), "ahead_mtp3_m": rear_rel_mtp + 0.457 * fp},
        "gate_0.80": stat_to(0.80), "r9_diag_0.76": stat_to(0.76), "r9_diag_0.79": stat_to(0.79),
        "drift_1.0": stat_to(1.0),
        "drift_1.0_frac_of_digit3_tip": (rear_rel_mtp + 2 * hp) / fr["digit3_tip_len_from_mtp_m"],
        "fraction_of_pad_length_from_rear": {str(x): (x + 1) / 2 for x in (-0.86, 0.0, 0.5, 0.8, 1.0)},
    }

    # mgh and stiffness ratios
    h_floor = float(com[2])
    ankle_z = 0.5 * (L["r"]["ankle"][2] + L["l"]["ankle"][2])
    mtp_z = 0.5 * (L["r"]["mtp"][3][2] + L["l"]["mtp"][3][2])
    mgh = {"floor": M * G * h_floor, "ankle_axis": M * G * (h_floor - ankle_z), "mtp3_axis": M * G * (h_floor - mtp_z)}
    res["com_height_m"] = h_floor
    res["mgh_Nm_per_rad"] = mgh
    toe_k = [float(m.jnt_stiffness[jid(m, f"{s}_toe_d{k}_joint")]) for s in "rl" for k in (2, 3, 4)]
    toe_c = [float(m.dof_damping[m.jnt_dofadr[jid(m, f"{s}_toe_d{k}_joint")]]) for s in "rl" for k in (2, 3, 4)]
    ank_k = [float(m.jnt_stiffness[jid(m, f"{s}_ankle")]) for s in "rl"]
    acts = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    kp = {n: float(m.actuator_gainprm[i][0]) for n, i in acts.items()}
    fr_rng = {n: m.actuator_forcerange[i].tolist() for n, i in acts.items()}
    res["stiffness"] = {
        "toe_k_each": toe_k, "toe_k_sum": sum(toe_k), "toe_c_each": toe_c,
        "ankle_passive_k_each": ank_k, "ankle_kp": [kp["r_ankle_act"], kp["l_ankle_act"]],
        "toe_sum_over_mgh": {k2: sum(toe_k) / v for k2, v in mgh.items()},
        "ankle_kp_sum_over_mgh": {k2: (kp["r_ankle_act"] + kp["l_ankle_act"]) / v for k2, v in mgh.items()},
        "ankle_kp_plus_passive_over_mgh": {k2: (kp["r_ankle_act"] + kp["l_ankle_act"] + sum(ank_k)) / v for k2, v in mgh.items()},
        "kp": kp, "forcerange": fr_rng,
        "toe_actuators_present": [n for n in kp if "toe" in n],
    }

    # joint table and ranges in included angles via kinematic sweeps (one joint at a time, at each limit)
    qhome = d.qpos.copy()
    table = {}
    for jn, key in (("r_knee", "knee_incl"), ("r_ankle", "ankle_incl"), ("r_toe_d3_joint", "mtp_incl"),
                    ("r_hip_pitch", "hip_incl")):
        j = jid(m, jn)
        lo, hi = (math.degrees(x) for x in m.jnt_range[j])
        row = {"qpos_range_deg": [lo, hi], "qpos_home_deg": math.degrees(qhome[m.jnt_qposadr[j]]),
               "ref_deg": math.degrees(m.qpos0[m.jnt_qposadr[j]]),
               "springref_deg": math.degrees(m.qpos_spring[m.jnt_qposadr[j]]),
               "stiffness": float(m.jnt_stiffness[j]), "damping": float(m.dof_damping[m.jnt_dofadr[j]])}
        vals = {}
        for tag, qd in (("lo", lo), ("hi", hi), ("home", row["qpos_home_deg"])):
            d.qpos[:] = qhome
            d.qpos[m.jnt_qposadr[j]] = math.radians(qd)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            Lg = leg(m, d, "r", fwd)
            vals[tag] = Lg[key]
            if jn == "r_hip_pitch":
                vals[tag + "_femur_from_vertical"] = Lg["femur_from_vertical"]
            if jn == "r_toe_d3_joint":
                v = xz(Lg["tips"][3]["end"] - Lg["mtp"][3], fwd)
                vals[tag + "_digit3_above_horizontal"] = math.degrees(math.atan2(v[1], v[0]))
        row["included_deg"] = vals
        row["included_minus_qpos_at_lo_hi"] = [vals["lo"] - lo, vals["hi"] - hi]
        row["included_plus_qpos_at_lo_hi"] = [vals["lo"] + lo, vals["hi"] + hi]
        table[jn] = row
    d.qpos[:] = qhome
    mujoco.mj_forward(m, d)
    # ankle home control in included angle: ctrl 82.4 deg vs qpos 76.9
    j = jid(m, "r_ankle")
    off = table["r_ankle"]["included_deg"]["home"] - table["r_ankle"]["qpos_home_deg"]
    res["ankle_home_ctrl_deg"] = math.degrees(float(d.ctrl[acts["r_ankle_act"]]))
    res["ankle_home_ctrl_included_deg"] = res["ankle_home_ctrl_deg"] + off
    res["joint_ranges"] = table

    # tail / neck actuator ranges and gravity torque about tail_1 pitch at home
    def subtree_gravity_torque(body, joint):
        b = bid(m, body)
        j2 = jid(m, joint)
        anc = d.xanchor[j2]
        axis = d.xaxis[j2]
        mass = float(m.body_subtreemass[b])
        c = d.subtree_com[b]
        tau = np.cross(c - anc, np.array([0, 0, -mass * G]))
        return {"mass_kg": mass, "torque_about_axis_Nm": float(np.dot(tau, axis)),
                "lever_m": float(np.linalg.norm((c - anc)[:2]))}

    res["gravity_torque_home"] = {
        "tail_1_pitch": subtree_gravity_torque("tail_1", "tail_1_pitch"),
        "neck_pitch": subtree_gravity_torque("neck", "neck_pitch"),
        "head_pitch": subtree_gravity_torque("skull", "head_pitch"),
    }
    res["tail_neck_ctrlrange_deg"] = {n: [math.degrees(x) for x in m.actuator_ctrlrange[i]]
                                     for n, i in acts.items() if n.startswith(("tail", "neck", "head"))}
    res["tail_joint_range_deg"] = {n: [math.degrees(x) for x in m.jnt_range[jid(m, n)]]
                                   for n in ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch")}

    # contact forces at the keyframe instant (single mj_forward; not equilibrium)
    floor = gid(m, "floor")
    f6 = np.zeros(6)
    load = {s: {"pad": 0.0, "d2": 0.0, "d3": 0.0, "d4": 0.0, "other": 0.0} for s in "rl"}
    for i in range(d.ncon):
        c = d.contact[i]
        if floor not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == floor else c.geom1
        mujoco.mj_contactForce(m, d, i, f6)
        fz = float(f6[0])
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other)
        s = nm[0]
        if nm.endswith("plantar_geom"):
            load[s]["pad"] += fz
        elif "_toe_d" in nm:
            load[s]["d" + nm.split("_toe_d")[1][0]] += fz
        elif s in load:
            load[s]["other"] += fz
    res["keyframe_instant_contacts_N"] = load
    res["keyframe_instant_digit_share"] = {
        s: (v["d2"] + v["d3"] + v["d4"]) / max(1e-9, v["pad"] + v["d2"] + v["d3"] + v["d4"]) for s, v in load.items()}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    with open(out_path, "w") as fh:
        json.dump(clean(res), fh, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
