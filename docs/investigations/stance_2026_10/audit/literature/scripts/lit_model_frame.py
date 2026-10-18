"""Kinematics-only measurement of the main-branch anatomical T. rex foot frame.

No time stepping: the home keyframe is loaded and only mj_kinematics and
mj_comPos are called (pure forward kinematics and centre-of-mass sums), so
no contact solve, no actuation and no simulation take place.  Joint-range
ends are evaluated the same way, one joint at a time from the home pose.

Usage: python -P lit_model_frame.py <main_ro root> <output json>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
XML = ROOT / "environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml"

m = mujoco.MjModel.from_xml_path(str(XML))
KEY = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")


def oid(kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    assert i >= 0, name
    return i


def bid(n):
    return oid(mujoco.mjtObj.mjOBJ_BODY, n)


def gid(n):
    return oid(mujoco.mjtObj.mjOBJ_GEOM, n)


def jid(n):
    return oid(mujoco.mjtObj.mjOBJ_JOINT, n)


def kin(qpos):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    return d


def home_qpos():
    return m.key_qpos[KEY].copy()


def capsule_end(d, g, with_cap=True):
    """Distal end of a capsule geom (local +z half-length), world frame."""
    c = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    hl = m.geom_size[g][1]
    rad = m.geom_size[g][0] if with_cap else 0.0
    axis = R[:, 2]
    # choose the end farther from the parent body origin (distal)
    b = m.geom_bodyid[g]
    o = d.xpos[b]
    e1 = c + axis * (hl + rad)
    e2 = c - axis * (hl + rad)
    return e1 if np.linalg.norm(e1 - o) > np.linalg.norm(e2 - o) else e2


def angle(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v)))))


def leg_points(d, s):
    p = {
        "hip": d.xpos[bid(f"{s}_thigh")].copy(),
        "knee": d.xpos[bid(f"{s}_tibia")].copy(),
        "ankle": d.xpos[bid(f"{s}_metatarsus")].copy(),
    }
    for dig in ("d2", "d3", "d4"):
        p[f"mtp_{dig}"] = d.xpos[bid(f"{s}_toe_{dig}")].copy()
        p[f"tip_{dig}"] = capsule_end(d, gid(f"{s}_toe_{dig}_geom"), True)
        p[f"tip_{dig}_nocap"] = capsule_end(d, gid(f"{s}_toe_{dig}_geom"), False)
    # plantar pad box corners in world
    g = gid(f"{s}_plantar_geom")
    c = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    hx, hy, hz = m.geom_size[g]
    p["pad_centre"] = c.copy()
    p["pad_rear"] = c - R[:, 0] * hx
    p["pad_front"] = c + R[:, 0] * hx
    p["pad_axis"] = R[:, 0].copy()
    p["pad_half"] = np.array([hx, hy, hz])
    p["pad_bottom_z"] = float(c[2] - abs(R[2, 2]) * hz - abs(R[2, 0]) * hx - abs(R[2, 1]) * hy)
    return p


HOME_SIGN = {}


def sag_angles(p):
    """Sagittal included angles (deg, 180 = straight) plus segment inclinations."""
    out = {}
    # signed sagittal deviation from straight (rotation of the distal segment
    # direction relative to the proximal one about world +y); included angle
    # = 180 - SIGN * deviation, SIGN fixed so the home pose reads < 180.
    def dev(prox, dist):
        u = prox[[0, 2]]; v = dist[[0, 2]]
        return math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v))
    for key, prox, dist in (("knee_included", p["knee"] - p["hip"], p["ankle"] - p["knee"]),
                            ("ankle_included", p["ankle"] - p["knee"], p["mtp_d3"] - p["ankle"]),
                            ("mtp_d3_included", p["mtp_d3"] - p["ankle"], p["tip_d3"] - p["mtp_d3"])):
        dv = dev(prox, dist)
        sgn = HOME_SIGN.setdefault(key, 1.0 if dv >= 0 else -1.0)
        out[key] = 180.0 - sgn * dv
        out[key + "_unsigned_check"] = angle(-prox, dist)
    fem = p["knee"] - p["hip"]
    out["femur_from_vertical"] = math.degrees(math.atan2(fem[0], -fem[2]))  # + = knee ahead of hip
    tib = p["ankle"] - p["knee"]
    out["tibia_from_vertical"] = math.degrees(math.atan2(tib[0], -tib[2]))
    mt = p["mtp_d3"] - p["ankle"]
    out["metatarsus_from_vertical"] = math.degrees(math.atan2(mt[0], -mt[2]))
    out["metatarsus_from_horizontal"] = 90.0 - out["metatarsus_from_vertical"]
    dg = p["tip_d3"] - p["mtp_d3"]
    out["digit3_below_horizontal"] = math.degrees(math.atan2(-dg[2], math.hypot(dg[0], dg[1])))
    return out


qh = home_qpos()
dh = kin(qh)
res = {"xml": str(XML), "mujoco": mujoco.__version__, "method": "mj_kinematics + mj_comPos at keyframe 'home'; no mj_step/mj_forward"}

legs = {}
for s in ("r", "l"):
    p = leg_points(dh, s)
    L = {}
    L["femur_len"] = float(np.linalg.norm(p["knee"] - p["hip"]))
    L["tibia_len"] = float(np.linalg.norm(p["ankle"] - p["knee"]))
    L["metatarsus_len_ankle_to_mtp3"] = float(np.linalg.norm(p["mtp_d3"] - p["ankle"]))
    for dig in ("d2", "d3", "d4"):
        L[f"digit_{dig}_len_mtp_to_tip_with_cap"] = float(np.linalg.norm(p[f"tip_{dig}"] - p[f"mtp_{dig}"]))
        L[f"digit_{dig}_len_mtp_to_tip_nocap"] = float(np.linalg.norm(p[f"tip_{dig}_nocap"] - p[f"mtp_{dig}"]))
    # fore-aft frame: foot heading = horizontal projection of pad x axis
    ax = p["pad_axis"].copy()
    ax[2] = 0.0
    ax /= np.linalg.norm(ax)
    L["pad_axis_world"] = p["pad_axis"].tolist()
    L["pad_axis_tilt_deg"] = math.degrees(math.asin(max(-1, min(1, float(p["pad_axis"][2])))))
    x0 = float(p["mtp_d3"] @ ax)  # MTP III fore-aft coordinate
    fa = lambda v: float(v @ ax) - x0  # noqa: E731  fore-aft distance ahead of MTP III
    L["pad_half_length"] = float(p["pad_half"][0])
    L["pad_half_width"] = float(p["pad_half"][1])
    L["pad_rear_ahead_of_mtp3"] = fa(p["pad_rear"])
    L["pad_front_ahead_of_mtp3"] = fa(p["pad_front"])
    L["pad_centre_ahead_of_mtp3"] = fa(p["pad_centre"])
    for dig in ("d2", "d3", "d4"):
        L[f"mtp_{dig}_ahead_of_mtp3"] = fa(p[f"mtp_{dig}"])
        L[f"tip_{dig}_ahead_of_mtp3"] = fa(p[f"tip_{dig}"])
        L[f"tip_{dig}_nocap_ahead_of_mtp3"] = fa(p[f"tip_{dig}_nocap"])
        lat = p[f"mtp_{dig}"] - p["mtp_d3"]
        L[f"mtp_{dig}_lateral_offset_m"] = float(lat[1]) if s == "l" else float(-lat[1])  # + = medial? recorded raw below
        L[f"mtp_{dig}_y_minus_mtp3_y"] = float(lat[1])
        L[f"mtp_{dig}_height_above_floor"] = float(p[f"mtp_{dig}"][2])
    L["ankle_height_above_floor"] = float(p["ankle"][2])
    L["ankle_ahead_of_mtp3"] = fa(p["ankle"])
    L["hip_height_above_floor"] = float(p["hip"][2])
    L["pad_bottom_z"] = p["pad_bottom_z"]
    L["footprint_pad_rear_to_d3_tip"] = L["tip_d3_ahead_of_mtp3"] - L["pad_rear_ahead_of_mtp3"]
    L["footprint_pad_rear_to_d3_tip_nocap"] = L["tip_d3_nocap_ahead_of_mtp3"] - L["pad_rear_ahead_of_mtp3"]
    L["mtp3_to_d3_tip"] = L["tip_d3_ahead_of_mtp3"]
    # digit spread in the ground plane
    def heading(v):
        return math.degrees(math.atan2(v[1], v[0]))
    hd = {dig: heading(p[f"tip_{dig}"] - p[f"mtp_{dig}"]) for dig in ("d2", "d3", "d4")}
    L["digit_heading_deg"] = hd
    L["spread_d2_d3_deg"] = abs(hd["d2"] - hd["d3"])
    L["spread_d3_d4_deg"] = abs(hd["d3"] - hd["d4"])
    L["spread_total_d2_d4_deg"] = abs(hd["d2"] - hd["d4"])
    L["angles_home"] = sag_angles(p)
    legs[s] = L
res["legs_home"] = legs

# COM vs hip (hip = midpoint of the two hip joints)
pel = bid("pelvis")
com = dh.subtree_com[pel].copy()
hipmid = 0.5 * (dh.xpos[bid("r_thigh")] + dh.xpos[bid("l_thigh")])
fem = legs["r"]["femur_len"]
# heading of the animal: pelvis x axis projected
px = dh.xmat[pel].reshape(3, 3)[:, 0].copy()
px[2] = 0
px /= np.linalg.norm(px)
res["com"] = {
    "com_world": com.tolist(),
    "hip_mid_world": hipmid.tolist(),
    "com_ahead_of_hip_m": float((com - hipmid) @ px),
    "com_ahead_of_hip_pct_femur": 100.0 * float((com - hipmid) @ px) / fem,
    "com_below_hip_m": float(hipmid[2] - com[2]),
    "com_below_hip_pct_femur": 100.0 * float(hipmid[2] - com[2]) / fem,
    "com_height_m": float(com[2]),
}
# COM projection in pad statistic (right foot frame), signed
Lr = legs["r"]
pr = leg_points(dh, "r")
ax = pr["pad_axis"].copy(); ax[2] = 0; ax /= np.linalg.norm(ax)
com_fa = float(com @ ax) - float(pr["mtp_d3"] @ ax)
res["com"]["com_ahead_of_mtp3_m"] = com_fa
res["com"]["com_projection_signed_pad_stat"] = (com_fa - Lr["pad_centre_ahead_of_mtp3"]) / Lr["pad_half_length"]

# segment masses (% total)
tot = float(m.body_subtreemass[pel])
def bm(names):
    return float(sum(m.body_mass[bid(n)] for n in names))
seg = {
    "total_kg": tot,
    "head": bm(["skull"]),
    "neck": bm(["neck"]),
    "tail": bm(["tail_1", "tail_2", "tail_3", "tail_4", "tail_5"]),
    "trunk_pelvis_body": bm(["pelvis"]),
    "forelimbs_both": bm(["r_arm", "r_forearm", "l_arm", "l_forearm"]),
    "thigh_one": bm(["r_thigh"]),
    "shank_one": bm(["r_tibia"]),
    "metatarsus_one": bm(["r_metatarsus"]),
    "digits_one": bm(["r_toe_d2", "r_toe_d3", "r_toe_d4"]),
}
seg["foot_one_incl_digits"] = seg["metatarsus_one"] + seg["digits_one"]
res["segment_mass_kg"] = seg
res["segment_mass_pct"] = {k: 100.0 * v / tot for k, v in seg.items() if k != "total_kg"}

# joint ranges, included-angle convention, one joint at a time from home
rng = {}
for jn, key in (("r_hip_pitch", "femur_from_vertical"), ("r_knee", "knee_included"),
                ("r_ankle", "ankle_included"), ("r_toe_d3_joint", "mtp_d3_included")):
    j = jid(jn)
    adr = m.jnt_qposadr[j]
    lo, hi = m.jnt_range[j]
    out = {"qpos_range_deg": [math.degrees(lo), math.degrees(hi)], "qpos_home_deg": math.degrees(qh[adr])}
    for tag, val in (("lo", lo), ("hi", hi)):
        q = qh.copy(); q[adr] = val
        a = sag_angles(leg_points(kin(q), "r"))
        out[f"{key}_at_{tag}"] = a[key]
        if jn == "r_toe_d3_joint":
            out[f"digit3_below_horizontal_at_{tag}"] = a["digit3_below_horizontal"]
        if jn == "r_ankle":
            out[f"metatarsus_from_horizontal_at_{tag}"] = a["metatarsus_from_horizontal"]
    out[f"{key}_at_home"] = legs["r"]["angles_home"][key]
    out["range_width_deg"] = math.degrees(hi - lo)
    rng[jn] = out
res["ranges"] = rng

# passive and servo stiffness from the model
def jstiff(n):
    return float(m.jnt_stiffness[jid(n)])
res["stiffness"] = {
    "digit_joint_stiffness_each_Nm_per_rad": jstiff("r_toe_d3_joint"),
    "digit_joint_damping_each": float(m.dof_damping[m.jnt_dofadr[jid("r_toe_d3_joint")]]),
    "ankle_kp_each": float(m.actuator_gainprm[oid(mujoco.mjtObj.mjOBJ_ACTUATOR, "r_ankle_act")][0]),
    "mgh_floor_home_Nm_per_rad": tot * 9.81 * float(com[2]),
}
res["stiffness"]["six_digit_springs_over_mgh"] = 6 * res["stiffness"]["digit_joint_stiffness_each_Nm_per_rad"] / res["stiffness"]["mgh_floor_home_Nm_per_rad"]
res["stiffness"]["two_ankle_kp_over_mgh"] = 2 * res["stiffness"]["ankle_kp_each"] / res["stiffness"]["mgh_floor_home_Nm_per_rad"]

# derived ratios
r_ = legs["r"]
res["ratios"] = {
    "metatarsus_over_femur": r_["metatarsus_len_ankle_to_mtp3"] / r_["femur_len"],
    "tibia_over_femur": r_["tibia_len"] / r_["femur_len"],
    "digit3_with_cap_over_metatarsus": r_["digit_d3_len_mtp_to_tip_with_cap"] / r_["metatarsus_len_ankle_to_mtp3"],
    "digit3_with_cap_over_femur": r_["digit_d3_len_mtp_to_tip_with_cap"] / r_["femur_len"],
    "digit2_over_digit3_with_cap": r_["digit_d2_len_mtp_to_tip_with_cap"] / r_["digit_d3_len_mtp_to_tip_with_cap"],
    "digit4_over_digit3_with_cap": r_["digit_d4_len_mtp_to_tip_with_cap"] / r_["digit_d3_len_mtp_to_tip_with_cap"],
    "footprint_over_hip_height": r_["footprint_pad_rear_to_d3_tip"] / r_["hip_height_above_floor"],
    "hip_height_over_footprint": r_["hip_height_above_floor"] / r_["footprint_pad_rear_to_d3_tip"],
    "pad_length_over_footprint": 2 * r_["pad_half_length"] / r_["footprint_pad_rear_to_d3_tip"],
    "pad_width_over_footprint": 2 * r_["pad_half_width"] / r_["footprint_pad_rear_to_d3_tip"],
    "hip_height_over_femur": r_["hip_height_above_floor"] / r_["femur_len"],
    "limb_len_sum_over_femur": (r_["femur_len"] + r_["tibia_len"] + r_["metatarsus_len_ankle_to_mtp3"]) / r_["femur_len"],
}


def tojson(o):
    if isinstance(o, dict):
        return {k: tojson(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [tojson(v) for v in o]
    if isinstance(o, np.ndarray):
        return [tojson(v) for v in o.tolist()]
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(tojson(res), indent=2))
print(json.dumps(tojson(res), indent=1))
