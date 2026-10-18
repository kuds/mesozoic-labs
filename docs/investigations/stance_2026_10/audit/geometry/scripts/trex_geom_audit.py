"""Read-only geometry audit of the main-branch (physics r9) anatomical T. rex.

Measures, at the authored home keyframe and at the env's noise-free zero-action
settle (the "statue"), every model number the foot/balance audit needs.  It
never writes outside the output directory given on the command line and never
edits the model.

Usage: python -P trex_geom_audit.py <main_ro root> <output dir>
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
OUT.mkdir(parents=True, exist_ok=True)
XML = ROOT / "environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml"
G = 9.81
SIDES = ("r", "l")
DIGITS = ("d2", "d3", "d4")

model = mujoco.MjModel.from_xml_path(str(XML))
G_MODEL = -float(model.opt.gravity[2])


def bid(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert i >= 0, name
    return i


def jid(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert i >= 0, name
    return i


def gid(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert i >= 0, name
    return i


def aid(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert i >= 0, name
    return i


KEY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
PELVIS = bid("pelvis")
FLOOR = gid("floor")


def r(x, n=4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, n) for v in x]
    if x is None:
        return None
    return round(float(x), n)


# ----------------------------------------------------------------------------
# A. Masses (compiled model)
# ----------------------------------------------------------------------------
def body_mass(names):
    return float(sum(model.body_mass[bid(n)] for n in names))


total_mass = float(model.body_subtreemass[PELVIS])
import xml.etree.ElementTree as _ET  # noqa: E402

_XML_TREE = _ET.parse(str(XML))
segments = {
    "head (skull body)": ["skull"],
    "neck": ["neck"],
    "trunk (pelvis body: torso+ribcage+belly)": ["pelvis"],
    "forelimbs (both)": ["r_arm", "r_forearm", "l_arm", "l_forearm"],
    "tail (tail_1..tail_5)": [f"tail_{i}" for i in range(1, 6)],
}
per_side = {}
for s in SIDES:
    per_side[s] = {
        "thigh": body_mass([f"{s}_thigh"]),
        "shank (tibia)": body_mass([f"{s}_tibia"]),
        "metatarsus incl. plantar pad": body_mass([f"{s}_metatarsus"]),
        "plantar pad geom alone (declared mass attr)": float(
            next(e for e in _XML_TREE.iter("geom") if e.get("name") == f"{s}_plantar_geom").get("mass")
        ),
        "digits (d2+d3+d4)": body_mass([f"{s}_toe_{d}" for d in DIGITS]),
        "digit d2": body_mass([f"{s}_toe_d2"]),
        "digit d3": body_mass([f"{s}_toe_d3"]),
        "digit d4": body_mass([f"{s}_toe_d4"]),
    }
    per_side[s]["foot (metatarsus+pad+digits)"] = (
        per_side[s]["metatarsus incl. plantar pad"] + per_side[s]["digits (d2+d3+d4)"]
    )
    per_side[s]["whole limb"] = per_side[s]["thigh"] + per_side[s]["shank (tibia)"] + per_side[s]["foot (metatarsus+pad+digits)"]
mass_rows = {k: {"kg": body_mass(v), "pct": 100 * body_mass(v) / total_mass} for k, v in segments.items()}
mass_side = {s: {k: {"kg": v, "pct": 100 * v / total_mass} for k, v in per_side[s].items()} for s in SIDES}
accounted = sum(body_mass(v) for v in segments.values()) + sum(per_side[s]["whole limb"] for s in SIDES)


# ----------------------------------------------------------------------------
# Geometry helpers on a forwarded MjData
# ----------------------------------------------------------------------------
def heading_axes(d):
    """Horizontal forward unit vector (pelvis x projected on the ground), lateral-left, up."""
    px = d.xmat[PELVIS].reshape(3, 3)[:, 0].copy()
    fwd = np.array([px[0], px[1], 0.0])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    return fwd, left, up


def sag(vec, fwd, up):
    """Signed CCW angle (deg) of a vector in the sagittal (forward, up) plane, 0 = forward, 90 = up."""
    return math.degrees(math.atan2(float(vec @ up), float(vec @ fwd)))


def capsule_ends(d, g):
    c = d.geom_xpos[g]
    ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
    h = model.geom_size[g][1]
    return c - h * ax, c + h * ax, ax, float(model.geom_size[g][0])


def digit_geometry(d, s, dig):
    j = jid(f"{s}_toe_{dig}_joint")
    mtp = d.xanchor[j].copy()
    a, b, ax, rad = capsule_ends(d, gid(f"{s}_toe_{dig}_geom"))
    distal = a if np.linalg.norm(a - mtp) > np.linalg.norm(b - mtp) else b
    unit = (distal - mtp) / np.linalg.norm(distal - mtp)
    tip = distal + rad * unit
    return mtp, distal, tip, rad


def included_angles(d, s):
    """Sagittal included angles (deg) for one leg, 180 = straight.

    hip   : angle between the pelvis's cranial (+x) axis and the femur (hip->knee), measured
            from the femur round through the front; 90 = femur perpendicular to the pelvis axis,
            smaller = femur swung forward (flexed), larger = swung back (extended).
    knee  : angle between knee->hip and knee->ankle measured on the POSTERIOR side.
    ankle : (intertarsal) angle between ankle->knee and ankle->MTP(III) on the ANTERIOR side,
            the bird intertarsal convention (ostrich engaged ~168 deg, Schaller et al. 2009).
    mtp   : angle between MTP->ankle and MTP->digit III tip on the DORSAL (upper/front) side.
    """
    fwd, _left, up = heading_axes(d)
    hip = d.xanchor[jid(f"{s}_hip_pitch")]
    knee = d.xanchor[jid(f"{s}_knee")]
    ankle = d.xanchor[jid(f"{s}_ankle")]
    mtp, _distal, tip, _ = digit_geometry(d, s, "d3")
    pel_x = d.xmat[PELVIS].reshape(3, 3)[:, 0]
    out = {}
    out["hip"] = (sag(pel_x, fwd, up) - sag(knee - hip, fwd, up)) % 360
    out["knee"] = (sag(ankle - knee, fwd, up) - sag(hip - knee, fwd, up)) % 360
    out["ankle"] = (sag(knee - ankle, fwd, up) - sag(mtp - ankle, fwd, up)) % 360
    out["mtp_dorsal"] = (sag(ankle - mtp, fwd, up) - sag(tip - mtp, fwd, up)) % 360
    out["femur_from_vertical_knee_fwd_positive"] = 90.0 + sag(knee - hip, fwd, up)
    out["tibia_from_vertical_ankle_back_positive"] = -(90.0 + sag(ankle - knee, fwd, up))
    out["metatarsus_from_vertical_mtp_fwd_positive"] = 90.0 + sag(mtp - ankle, fwd, up)
    out["digit3_below_horizontal"] = -sag(tip - mtp, fwd, up)
    out["pelvis_pitch_nose_down"] = -sag(pel_x, fwd, up)
    return out


def floor_contacts(d):
    """Per-contact rows (geom, normal force, world position) for floor contacts."""
    rows = []
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if FLOOR not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == FLOOR else c.geom1
        mujoco.mj_contactForce(model, d, i, f6)
        rows.append((int(other), float(f6[0]), c.pos.copy()))
    return rows


def geom_side_part(g):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    for s in SIDES:
        if name == f"{s}_plantar_geom":
            return s, "pad"
        for dig in DIGITS:
            if name == f"{s}_toe_{dig}_geom":
                return s, dig
        if name.startswith(f"{s}_"):
            return s, "other_leg"
    return None, name


def measure(d):
    """All geometry at a forwarded state."""
    fwd, left, up = heading_axes(d)
    res = {}
    hips = np.array([d.xanchor[jid(f"{s}_hip_pitch")] for s in SIDES])
    hip_mid = hips.mean(0)
    com = d.subtree_com[PELVIS].copy()
    legs = {}
    for s in SIDES:
        hip = d.xanchor[jid(f"{s}_hip_pitch")]
        knee = d.xanchor[jid(f"{s}_knee")]
        ankle = d.xanchor[jid(f"{s}_ankle")]
        leg = {
            "femur_hip_to_knee_m": np.linalg.norm(knee - hip),
            "tibia_knee_to_ankle_m": np.linalg.norm(ankle - knee),
        }
        dg = {}
        for dig in DIGITS:
            mtp, distal, tip, rad = digit_geometry(d, s, dig)
            dg[dig] = {
                "mtp_world": mtp,
                "metatarsus_ankle_to_mtp_m": np.linalg.norm(mtp - ankle),
                "digit_mtp_to_capsule_end_m": np.linalg.norm(distal - mtp),
                "digit_mtp_to_tip_incl_radius_m": np.linalg.norm(tip - mtp),
                "capsule_radius_m": rad,
                "tip_world": tip,
                "axis_horizontal": (tip - mtp) - ((tip - mtp) @ up) * up,
            }
        leg["metatarsus_ankle_to_mtp3_m"] = dg["d3"]["metatarsus_ankle_to_mtp_m"]
        leg["metatarsus_over_femur"] = leg["metatarsus_ankle_to_mtp3_m"] / leg["femur_hip_to_knee_m"]
        leg["tibia_over_femur"] = leg["tibia_knee_to_ankle_m"] / leg["femur_hip_to_knee_m"]
        for dig in DIGITS:
            leg[f"digit_{dig}_mtp_to_capsule_end_m"] = dg[dig]["digit_mtp_to_capsule_end_m"]
            leg[f"digit_{dig}_mtp_to_tip_incl_radius_m"] = dg[dig]["digit_mtp_to_tip_incl_radius_m"]
            leg[f"digit_{dig}_over_femur_tip"] = dg[dig]["digit_mtp_to_tip_incl_radius_m"] / leg["femur_hip_to_knee_m"]
        # Ground-plane digit spread.  Medial for the right foot is +left; outer = lateral.
        lat_sign = -1.0 if s == "r" else 1.0  # +1 along this = lateral (outer)

        def heading_of(v):
            return math.degrees(math.atan2(float(v @ left) * lat_sign, float(v @ fwd)))  # + = lateral

        ang = {dig: heading_of(dg[dig]["axis_horizontal"]) for dig in DIGITS}
        leg["digit_heading_deg_lateral_positive"] = ang
        leg["spread_II_III_deg"] = ang["d3"] - ang["d2"]
        leg["spread_III_IV_deg"] = ang["d4"] - ang["d3"]
        leg["spread_total_II_IV_deg"] = ang["d4"] - ang["d2"]
        # Pad box.
        pg = gid(f"{s}_plantar_geom")
        pc = d.geom_xpos[pg].copy()
        pm = d.geom_xmat[pg].reshape(3, 3)
        px, py, pz = pm[:, 0], pm[:, 1], pm[:, 2]
        half = model.geom_size[pg].copy()
        leg["pad_half_size_m"] = half
        leg["pad_full_size_m"] = 2 * half
        leg["pad_centre_world"] = pc
        leg["pad_tilt_deg"] = math.degrees(math.acos(min(1.0, abs(float(pz @ up)))))
        leg["pad_sole_height_above_floor_m"] = float(pc @ up) - half[2] * abs(float(pz @ up))
        rel = {}
        for dig in DIGITS:
            mtp = dg[dig]["mtp_world"]
            v = pc - mtp
            fore = float(v @ px)
            lat = float(v @ py) * lat_sign  # + = pad centre lateral of this MTP
            vert = float(v @ pz)
            rel[dig] = {
                "pad_centre_ahead_of_mtp_m": fore,
                "pad_rear_edge_rel_mtp_m": fore - half[0],
                "pad_front_edge_rel_mtp_m": fore + half[0],
                "pad_centre_lateral_of_mtp_m": lat,
                "pad_lateral_edge_rel_mtp_m": lat + half[1],
                "pad_medial_edge_rel_mtp_m": lat - half[1],
                "mtp_axis_above_pad_sole_m": -(vert - half[2]),
            }
        leg["pad_rel_mtp"] = rel
        # Footprint along the pad forward axis (horizontalised).
        pfwd = px - (px @ up) * up
        pfwd /= np.linalg.norm(pfwd)
        rear = pc - half[0] * px
        leg["footprint_pad_rear_to_d3_tip_incl_radius_m"] = float((dg["d3"]["tip_world"] - rear) @ pfwd)
        d3_end = dg["d3"]["tip_world"] - dg["d3"]["capsule_radius_m"] * (
            (dg["d3"]["tip_world"] - dg["d3"]["mtp_world"]) / dg["d3"]["digit_mtp_to_tip_incl_radius_m"]
        )
        leg["footprint_pad_rear_to_d3_capsule_end_m"] = float((d3_end - rear) @ pfwd)
        leg["pad_centre_frac_of_footprint_from_rear"] = half[0] / leg["footprint_pad_rear_to_d3_tip_incl_radius_m"]
        leg["pad_centre_ahead_of_mtp3_frac_of_d3_tip"] = rel["d3"]["pad_centre_ahead_of_mtp_m"] / dg["d3"]["digit_mtp_to_tip_incl_radius_m"]
        leg["pad_front_ahead_of_mtp3_frac_of_d3_tip"] = rel["d3"]["pad_front_edge_rel_mtp_m"] / dg["d3"]["digit_mtp_to_tip_incl_radius_m"]
        # Ankle-axis frame (the env's whole-foot CoP frame): pad centre ahead of ankle along pad x.
        leg["pad_centre_ahead_of_ankle_along_pad_x_m"] = float((pc - ankle) @ px)
        leg["ankle_height_m"] = float(ankle @ up)
        leg["mtp3_height_m"] = float(dg["d3"]["mtp_world"] @ up)
        leg["hip_height_m"] = float(hip @ up)
        leg["knee_ahead_of_pad_centre_m"] = float((knee - pc) @ pfwd)
        leg["hip_ahead_of_pad_centre_m"] = float((hip - pc) @ pfwd)
        leg["ankle_ahead_of_pad_centre_m"] = float((ankle - pc) @ pfwd)
        leg["com_ahead_of_pad_centre_m"] = float((com - pc) @ pfwd)
        leg["com_proj_pad_stat_signed"] = leg["com_ahead_of_pad_centre_m"] / half[0]
        leg["com_proj_frac_of_footprint_from_rear"] = float((com - rear) @ pfwd) / leg["footprint_pad_rear_to_d3_tip_incl_radius_m"]
        leg["included_angles_deg"] = included_angles(d, s)
        leg["pfwd"] = pfwd
        leg["px"] = px
        leg["pc"] = pc
        leg["rear"] = rear
        legs[s] = leg
    femur = np.mean([legs[s]["femur_hip_to_knee_m"] for s in SIDES])
    hip_h = float(hip_mid @ up)
    res["total_mass_kg"] = total_mass
    res["hip_mid_height_m"] = hip_h
    res["com_world"] = com
    res["com_height_above_floor_m"] = float(com @ up)
    ank_h = np.mean([legs[s]["ankle_height_m"] for s in SIDES])
    mtp_h = np.mean([legs[s]["mtp3_height_m"] for s in SIDES])
    res["com_height_above_ankle_axis_m"] = float(com @ up) - ank_h
    res["com_height_above_mtp3_axis_m"] = float(com @ up) - mtp_h
    res["com_ahead_of_hip_m"] = float((com - hip_mid) @ fwd)
    res["com_ahead_of_hip_pct_femur"] = 100 * res["com_ahead_of_hip_m"] / femur
    res["com_above_hip_m"] = float((com - hip_mid) @ up)
    res["com_above_hip_pct_femur"] = 100 * res["com_above_hip_m"] / femur
    res["com_lateral_left_of_hip_mid_m"] = float((com - hip_mid) @ left)
    pc_mid = np.mean([legs[s]["pc"] for s in SIDES], axis=0)
    res["com_ahead_of_mean_pad_centre_m"] = float((com - pc_mid) @ fwd)
    res["com_proj_pad_stat_signed_mean_feet"] = res["com_ahead_of_mean_pad_centre_m"] / legs["r"]["pad_half_size_m"][0]
    res["foot_site_width_m"] = None
    res["pad_centre_lateral_separation_m"] = float(abs((legs["r"]["pc"] - legs["l"]["pc"]) @ left))
    res["footprint_over_hip_height"] = np.mean([legs[s]["footprint_pad_rear_to_d3_tip_incl_radius_m"] for s in SIDES]) / hip_h
    res["mgh_floor_Nm_per_rad"] = total_mass * G_MODEL * res["com_height_above_floor_m"]
    res["mgh_ankle_Nm_per_rad"] = total_mass * G_MODEL * res["com_height_above_ankle_axis_m"]
    res["mgh_mtp3_Nm_per_rad"] = total_mass * G_MODEL * res["com_height_above_mtp3_axis_m"]
    res["omega0_floor_per_s"] = math.sqrt(G_MODEL / res["com_height_above_floor_m"])
    # Contacts
    rows = floor_contacts(d)
    per = {s: {"pad": 0.0, "d2": 0.0, "d3": 0.0, "d4": 0.0, "other_leg": 0.0} for s in SIDES}
    nonleg = 0.0
    padcop = {s: [0.0, 0.0] for s in SIDES}
    dig_pos = {s: {dg: [0.0, 0.0] for dg in DIGITS} for s in SIDES}
    dig_fore = {s: [0.0, 0.0] for s in SIDES}
    dgeo = {s: {dg: digit_geometry(d, s, dg) for dg in DIGITS} for s in SIDES}
    footcop = {s: [0.0, 0.0] for s in SIDES}
    xs = {s: [] for s in SIDES}
    for g, fn, pos in rows:
        s, part = geom_side_part(g)
        if s is None:
            nonleg += fn
            continue
        per[s][part] += fn
        L = legs[s]
        if part == "pad":
            padcop[s][0] += fn * float((pos - L["pc"]) @ L["px"]) / L["pad_half_size_m"][0]
            padcop[s][1] += fn
        if part in DIGITS:
            mtp_w, _dist, tip_w, _rad = dgeo[s][part]
            tip_len = float(np.linalg.norm(tip_w - mtp_w))
            unit = (tip_w - mtp_w) / tip_len
            dig_pos[s][part][0] += fn * float((pos - mtp_w) @ unit) / tip_len
            dig_pos[s][part][1] += fn
            dig_fore[s][0] += fn * float((pos - dgeo[s]["d3"][0]) @ L["px"])
            dig_fore[s][1] += fn
        footcop[s][0] += fn * float((pos - L["pc"]) @ L["px"]) / L["pad_half_size_m"][0]
        footcop[s][1] += fn
        xs[s].append(float((pos - L["rear"]) @ L["pfwd"]))
    contact = {}
    for s in SIDES:
        tot = sum(per[s].values())
        digits = per[s]["d2"] + per[s]["d3"] + per[s]["d4"]
        contact[s] = {
            "normal_N": dict(per[s]),
            "total_N": tot,
            "digit_share": digits / tot if tot > 0 else None,
            "pad_cop_stat_signed": padcop[s][0] / padcop[s][1] if padcop[s][1] > 0 else None,
            "wholefoot_cop_signed_pad_halflengths": footcop[s][0] / footcop[s][1] if footcop[s][1] > 0 else None,
            "contact_x_from_pad_rear_min_max_m": [min(xs[s]), max(xs[s])] if xs[s] else None,
            "n_contacts": len(xs[s]),
            "digit_load_pos_frac_of_tip_length": {dg: (dig_pos[s][dg][0] / dig_pos[s][dg][1] if dig_pos[s][dg][1] > 0 else None) for dg in DIGITS},
            "digit_load_centre_ahead_of_mtp3_m": dig_fore[s][0] / dig_fore[s][1] if dig_fore[s][1] > 0 else None,
        }
    res["contacts"] = contact
    res["nonleg_floor_N"] = nonleg
    res["weight_N"] = total_mass * G_MODEL
    sd = {}
    for name in ("r_foot_touch", "l_foot_touch", "r_toe_d2_touch", "r_toe_d3_touch", "r_toe_d4_touch",
                 "l_toe_d2_touch", "l_toe_d3_touch", "l_toe_d4_touch"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        sd[name] = float(d.sensordata[model.sensor_adr[sid]])
    res["touch_sensors_N"] = sd
    for s in SIDES:
        res[f"touch_digit_share_{s}"] = (sd[f"{s}_toe_d2_touch"] + sd[f"{s}_toe_d3_touch"] + sd[f"{s}_toe_d4_touch"]) / (
            sd[f"{s}_foot_touch"] + sd[f"{s}_toe_d2_touch"] + sd[f"{s}_toe_d3_touch"] + sd[f"{s}_toe_d4_touch"])
    for s in SIDES:
        for k in ("pfwd", "px", "pc", "rear"):
            legs[s].pop(k)
    res["legs"] = legs
    res["foot_site_width_m"] = float(abs((d.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_foot")]
                                           - d.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_foot")]) @ left))
    res["joint_qpos_deg"] = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): math.degrees(d.qpos[model.jnt_qposadr[j]])
                             for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    res["pelvis_height_m"] = float(d.qpos[2])
    return res


def tojson(o):
    if isinstance(o, dict):
        return {str(k): tojson(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [tojson(v) for v in o]
    if isinstance(o, np.ndarray):
        return [tojson(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


# ----------------------------------------------------------------------------
# B. Home keyframe
# ----------------------------------------------------------------------------
d_home = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, d_home, KEY)
mujoco.mj_forward(model, d_home)
home = measure(d_home)

# Metatarsus frame orientation check at home (is it world-aligned?).
mt_rot_home = {s: d_home.xmat[bid(f"{s}_metatarsus")].reshape(3, 3).tolist() for s in SIDES}


# ----------------------------------------------------------------------------
# C. The env's settle: noise-free zero-action statue, 1000 control steps.
# ----------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv  # noqa: E402

env = TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0)
env.reset(seed=0)
assert env.model.nq == model.nq
env_model_check = {
    name: float(np.max(np.abs(np.asarray(getattr(env.model, name), dtype=float) - np.asarray(getattr(model, name), dtype=float))))
    for name in ("body_mass", "body_ipos", "body_pos", "jnt_stiffness", "dof_damping", "jnt_range", "qpos_spring",
                 "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange", "geom_size", "geom_pos",
                 "geom_friction", "key_qpos", "key_ctrl")
}
env_model_check["opt_timestep"] = float(env.model.opt.timestep - model.opt.timestep)
env_model_check["opt_cone"] = int(env.model.opt.cone) - int(model.opt.cone)
zero = np.zeros(env.action_space.shape)
trace = {"pelvis_z": [], "com_x_rel_padmid": [], "pad_stat_r": [], "pad_stat_l": [], "digit_share_r": [],
         "digit_share_l": [], "wholefoot_r": [], "wholefoot_l": [], "com_stat": [], "tail_1_pitch": [],
         "neck_pitch": [], "head_pitch": [], "terminated": []}
window_states = []
measured_window = []
ctrl_used = None
for step in range(1000):
    _obs, _rew, term, trunc, _info = env.step(zero)
    if step == 0:
        ctrl_used = env.data.ctrl.copy()
    trace["terminated"].append(bool(term))
    if step >= 200:
        mm = measure(env.data)
        measured_window.append(mm)
    if step in (199, 999):
        window_states.append((step, env.data.qpos.copy(), env.data.qvel.copy()))
env_final_qpos = env.data.qpos.copy()
env_final_qvel = env.data.qvel.copy()
settled_final = measure(env.data)


def wmean(key_fn):
    vals = [key_fn(mm) for mm in measured_window]
    vals = [v for v in vals if v is not None]
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None


window = {
    "pelvis_height_m": wmean(lambda m: m["pelvis_height_m"]),
    "com_height_above_floor_m": wmean(lambda m: m["com_height_above_floor_m"]),
    "com_ahead_of_hip_m": wmean(lambda m: m["com_ahead_of_hip_m"]),
    "com_ahead_of_hip_pct_femur": wmean(lambda m: m["com_ahead_of_hip_pct_femur"]),
    "com_above_hip_m": wmean(lambda m: m["com_above_hip_m"]),
    "com_proj_pad_stat_signed_mean_feet": wmean(lambda m: m["com_proj_pad_stat_signed_mean_feet"]),
    "mgh_floor_Nm_per_rad": wmean(lambda m: m["mgh_floor_Nm_per_rad"]),
    "mgh_ankle_Nm_per_rad": wmean(lambda m: m["mgh_ankle_Nm_per_rad"]),
}
for s in SIDES:
    window[f"pad_cop_stat_signed_{s}"] = wmean(lambda m, s=s: m["contacts"][s]["pad_cop_stat_signed"])
    window[f"wholefoot_cop_signed_{s}"] = wmean(lambda m, s=s: m["contacts"][s]["wholefoot_cop_signed_pad_halflengths"])
    window[f"digit_share_contacts_{s}"] = wmean(lambda m, s=s: m["contacts"][s]["digit_share"])
    window[f"digit_share_touch_{s}"] = wmean(lambda m, s=s: m[f"touch_digit_share_{s}"])
    window[f"foot_total_N_{s}"] = wmean(lambda m, s=s: m["contacts"][s]["total_N"])
    for part in ("pad", "d2", "d3", "d4", "other_leg"):
        window[f"normal_N_{s}_{part}"] = wmean(lambda m, s=s, part=part: m["contacts"][s]["normal_N"][part])
window["nonleg_floor_N"] = wmean(lambda m: m["nonleg_floor_N"])
for s in SIDES:
    window[f"digit_load_centre_ahead_of_mtp3_m_{s}"] = wmean(lambda m, s=s: m["contacts"][s]["digit_load_centre_ahead_of_mtp3_m"])
    for dg in DIGITS:
        window[f"digit_load_pos_frac_{s}_{dg}"] = wmean(lambda m, s=s, dg=dg: m["contacts"][s]["digit_load_pos_frac_of_tip_length"][dg])
window["terminated_any"] = any(trace["terminated"])
window_joint_means_deg = {k: float(np.mean([m["joint_qpos_deg"][k] for m in measured_window])) for k in measured_window[0]["joint_qpos_deg"]}

# Raw MuJoCo cross-check: key ctrl held, same 5000 substeps.
d_raw = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, d_raw, KEY)
d_raw.ctrl[:] = model.key_ctrl[KEY]
mujoco.mj_forward(model, d_raw)
for _ in range(5000):
    mujoco.mj_step(model, d_raw)
mujoco.mj_forward(model, d_raw)
raw_vs_env_max_abs_qpos = float(np.max(np.abs(d_raw.qpos - env_final_qpos)))
settled_raw = measure(d_raw)


# ----------------------------------------------------------------------------
# D. Joint ranges in the included-angle convention.
# ----------------------------------------------------------------------------
def angles_with(qbase, s, jname, value):
    dd = mujoco.MjData(model)
    dd.qpos[:] = qbase
    dd.qpos[model.jnt_qposadr[jid(jname)]] = value
    mujoco.mj_kinematics(model, dd)
    mujoco.mj_comPos(model, dd)
    return included_angles(dd, s)


ranges = {}
qhome = model.key_qpos[KEY].copy()
for label, jn, key_angle in (
    ("hip", "r_hip_pitch", "hip"),
    ("knee", "r_knee", "knee"),
    ("ankle (intertarsal)", "r_ankle", "ankle"),
    ("MTP / digit III", "r_toe_d3_joint", "mtp_dorsal"),
):
    j = jid(jn)
    lo, hi = model.jnt_range[j]
    a_lo = angles_with(qhome, "r", jn, lo)
    a_hi = angles_with(qhome, "r", jn, hi)
    a_home = angles_with(qhome, "r", jn, qhome[model.jnt_qposadr[j]])
    rng = {
        "joint": jn,
        "mjcf_range_deg": [math.degrees(lo), math.degrees(hi)],
        "ref_deg": math.degrees(model.qpos0[model.jnt_qposadr[j]]),
        "springref_deg": math.degrees(model.qpos_spring[model.jnt_qposadr[j]]),
        "home_qpos_deg": math.degrees(qhome[model.jnt_qposadr[j]]),
        "included_at_range_lo_deg": a_lo[key_angle],
        "included_at_range_hi_deg": a_hi[key_angle],
        "included_at_home_deg": a_home[key_angle],
        "included_at_settled_deg": settled_final["legs"]["r"]["included_angles_deg"][key_angle],
        "width_deg": math.degrees(hi - lo),
        "d_included_d_qpos": (a_hi[key_angle] - a_lo[key_angle]) / math.degrees(hi - lo),
    }
    if label == "MTP / digit III":
        rng["digit3_below_horizontal_at_lo_deg"] = a_lo["digit3_below_horizontal"]
        rng["digit3_below_horizontal_at_hi_deg"] = a_hi["digit3_below_horizontal"]
        rng["digit3_below_horizontal_at_home_deg"] = a_home["digit3_below_horizontal"]
    if label == "hip":
        rng["femur_from_vertical_at_lo_deg"] = a_lo["femur_from_vertical_knee_fwd_positive"]
        rng["femur_from_vertical_at_hi_deg"] = a_hi["femur_from_vertical_knee_fwd_positive"]
        rng["femur_from_vertical_at_home_deg"] = a_home["femur_from_vertical_knee_fwd_positive"]
    ranges[label] = rng
# Other digits share the d3 hinge parameters; record their MTP dorsal angle at home for completeness.
for jn in ("r_toe_d2_joint", "r_toe_d4_joint", "r_hip_roll", "l_hip_roll"):
    j = jid(jn)
    ranges[jn] = {
        "mjcf_range_deg": [math.degrees(v) for v in model.jnt_range[j]],
        "ref_deg": math.degrees(model.qpos0[model.jnt_qposadr[j]]),
        "springref_deg": math.degrees(model.qpos_spring[model.jnt_qposadr[j]]),
        "home_qpos_deg": math.degrees(qhome[model.jnt_qposadr[j]]),
    }


# ----------------------------------------------------------------------------
# E. Passive joint properties and actuators, as ratios to m*g*h.
# ----------------------------------------------------------------------------
act_of_joint = {}
for a in range(model.nu):
    act_of_joint[int(model.actuator_trnid[a][0])] = a
joint_table = []
for j in range(model.njnt):
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    dof = model.jnt_dofadr[j]
    row = {
        "joint": name,
        "range_deg": [math.degrees(v) for v in model.jnt_range[j]],
        "stiffness_Nm_per_rad": float(model.jnt_stiffness[j]),
        "springref_deg": math.degrees(model.qpos_spring[model.jnt_qposadr[j]]),
        "damping_Nms_per_rad": float(model.dof_damping[dof]),
        "armature": float(model.dof_armature[dof]),
        "actuated": j in act_of_joint,
    }
    if j in act_of_joint:
        a = act_of_joint[j]
        row["actuator"] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        row["kp"] = float(model.actuator_gainprm[a][0])
        row["kv_from_biasprm"] = float(-model.actuator_biasprm[a][2])
        row["forcerange_Nm"] = [float(v) for v in model.actuator_forcerange[a]]
        row["ctrlrange_rad"] = [float(v) for v in model.actuator_ctrlrange[a]]
        row["ctrlrange_deg"] = [math.degrees(v) for v in model.actuator_ctrlrange[a]]
        row["home_ctrl_rad"] = float(model.key_ctrl[KEY][a])
    for tag, mgh in (("home_floor", home["mgh_floor_Nm_per_rad"]), ("home_ankle", home["mgh_ankle_Nm_per_rad"]),
                     ("settled_floor", settled_final["mgh_floor_Nm_per_rad"]), ("settled_ankle", settled_final["mgh_ankle_Nm_per_rad"])):
        row[f"stiffness_over_mgh_{tag}"] = row["stiffness_Nm_per_rad"] / mgh
        if "kp" in row:
            row[f"kp_over_mgh_{tag}"] = row["kp"] / mgh
    joint_table.append(row)


def jrow(n):
    return next(x for x in joint_table if x["joint"] == n)


groups = {}
for tag, mgh_key, src in (("home_floor", "mgh_floor_Nm_per_rad", home), ("home_ankle", "mgh_ankle_Nm_per_rad", home),
                          ("home_mtp3", "mgh_mtp3_Nm_per_rad", home),
                          ("settled_floor", "mgh_floor_Nm_per_rad", settled_final), ("settled_ankle", "mgh_ankle_Nm_per_rad", settled_final),
                          ("settled_mtp3", "mgh_mtp3_Nm_per_rad", settled_final)):
    mgh = src[mgh_key]
    digit_k = sum(jrow(f"{s}_toe_{dg}_joint")["stiffness_Nm_per_rad"] for s in SIDES for dg in DIGITS)
    ankle_k = sum(jrow(f"{s}_ankle")["stiffness_Nm_per_rad"] for s in SIDES)
    ankle_kp = sum(jrow(f"{s}_ankle")["kp"] for s in SIDES)
    ankle_fr = sum(jrow(f"{s}_ankle")["forcerange_Nm"][1] for s in SIDES)
    digit_c = sum(jrow(f"{s}_toe_{dg}_joint")["damping_Nms_per_rad"] for s in SIDES for dg in DIGITS)
    ankle_c = sum(jrow(f"{s}_ankle")["damping_Nms_per_rad"] for s in SIDES)
    groups[tag] = {
        "mgh_Nm_per_rad": mgh,
        "six_digit_springs_sum": digit_k, "six_digit_springs_over_mgh": digit_k / mgh,
        "two_ankle_passive_springs_sum": ankle_k, "two_ankle_passive_springs_over_mgh": ankle_k / mgh,
        "two_ankle_servo_kp_sum": ankle_kp, "two_ankle_servo_kp_over_mgh": ankle_kp / mgh,
        "ankle_servo_plus_spring_over_mgh": (ankle_kp + ankle_k) / mgh,
        "two_ankle_forcerange_sum_Nm": ankle_fr,
        "ankle_forcerange_over_mg_m": ankle_fr / (total_mass * G_MODEL),
        "ankle_forcerange_over_mgh_rad": ankle_fr / mgh,
        "six_digit_damping_sum_Nms": digit_c, "two_ankle_damping_sum_Nms": ankle_c,
        "digit_damping_over_mgh_s": digit_c / mgh, "ankle_damping_over_mgh_s": ankle_c / mgh,
    }


# ----------------------------------------------------------------------------
# F. Balance authority of tail and neck pitch.
# ----------------------------------------------------------------------------
def com_after(qbase, changes):
    dd = mujoco.MjData(model)
    dd.qpos[:] = qbase
    for jn, v in changes.items():
        dd.qpos[model.jnt_qposadr[jid(jn)]] = v
    mujoco.mj_kinematics(model, dd)
    mujoco.mj_comPos(model, dd)
    return dd.subtree_com[PELVIS].copy()


def gravity_torque_about(d, jname):
    """Static gravity torque (N m) about a pitch joint from its child subtree, sign = rotation that gravity drives."""
    j = jid(jname)
    b = model.jnt_bodyid[j]
    anchor = d.xanchor[j]
    axis = d.xaxis[j]
    m_sub = model.body_subtreemass[b]
    c = d.subtree_com[b]
    f = np.array([0.0, 0.0, -m_sub * G_MODEL])
    return float(np.cross(c - anchor, f) @ axis), float(m_sub)


auth = {}
pitch_sets = {
    "tail_pitch_all (tail_1,2,3)": ["tail_1_pitch", "tail_2_pitch", "tail_3_pitch"],
    "tail_1_pitch only": ["tail_1_pitch"],
    "neck_pitch + head_pitch": ["neck_pitch", "head_pitch"],
    "neck_pitch only": ["neck_pitch"],
}
for basis_name, qbase, ref in (("home", model.key_qpos[KEY].copy(), home), ("settled", env_final_qpos.copy(), settled_final)):
    com0 = com_after(qbase, {})
    fwd = np.array([1.0, 0.0, 0.0])
    out = {}
    for label, js in pitch_sets.items():
        lo = {jn: model.jnt_range[jid(jn)][0] for jn in js}
        hi = {jn: model.jnt_range[jid(jn)][1] for jn in js}
        c_lo = com_after(qbase, lo)
        c_hi = com_after(qbase, hi)
        out[label] = {
            "joint_ranges_deg": {jn: [math.degrees(v) for v in model.jnt_range[jid(jn)]] for jn in js},
            "current_deg": {jn: math.degrees(qbase[model.jnt_qposadr[jid(jn)]]) for jn in js},
            "com_dx_at_range_lo_m": float((c_lo - com0)[0]), "com_dz_at_range_lo_m": float((c_lo - com0)[2]),
            "com_dx_at_range_hi_m": float((c_hi - com0)[0]), "com_dz_at_range_hi_m": float((c_hi - com0)[2]),
            "com_dx_span_m": float(abs((c_hi - c_lo)[0])),
            "com_dx_span_pad_halflengths": float(abs((c_hi - c_lo)[0])) / 0.07,
        }
    auth[basis_name] = out
dd_h = mujoco.MjData(model)
dd_h.qpos[:] = model.key_qpos[KEY]
mujoco.mj_kinematics(model, dd_h)
mujoco.mj_comPos(model, dd_h)
grav = {}
for jn in ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch", "neck_pitch", "head_pitch"):
    tq, ms = gravity_torque_about(dd_h, jn)
    row = jrow(jn)
    grav[jn] = {
        "subtree_mass_kg": ms,
        "gravity_torque_at_home_Nm": tq,
        "forcerange_Nm": row["forcerange_Nm"],
        "kp": row["kp"],
        "passive_stiffness": row["stiffness_Nm_per_rad"],
        "ctrlrange_deg": row["ctrlrange_deg"],
        "joint_range_deg": row["range_deg"],
        "max_servo_torque_at_q0_from_ctrlrange_Nm": [row["kp"] * row["ctrlrange_rad"][0], row["kp"] * row["ctrlrange_rad"][1]],
        "forcerange_over_mg_pad_half": row["forcerange_Nm"][1] / (total_mass * G_MODEL * 0.07),
        "forcerange_over_mg_m": row["forcerange_Nm"][1] / (total_mass * G_MODEL),
    }

# Dynamic settles with the tail / neck commanded to their extremes (residual +-1), raw MuJoCo, key start.
def residual_ctrl(res_map):
    ctrl = model.key_ctrl[KEY].copy()
    for an, rv in res_map.items():
        a = aid(an)
        lo, hi = model.actuator_ctrlrange[a]
        home_c = model.key_ctrl[KEY][a]
        ctrl[a] = home_c + (rv * (home_c - lo) if rv < 0 else rv * (hi - home_c))
    return ctrl


def settle_with(ctrl, substeps=5000, ramp_substeps=0, window_start=1000):
    dd = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, dd, KEY)
    home_c = model.key_ctrl[KEY].copy()
    dd.ctrl[:] = ctrl if ramp_substeps == 0 else home_c
    mujoco.mj_forward(model, dd)
    hist = []
    for k in range(substeps):
        if ramp_substeps:
            frac = min(1.0, (k + 1) / ramp_substeps)
            dd.ctrl[:] = home_c + frac * (ctrl - home_c)
        mujoco.mj_step(model, dd)
        if k >= window_start and k % 5 == 4:
            mujoco.mj_forward(model, dd)
            hist.append(measure(dd))
    mujoco.mj_forward(model, dd)
    fin = measure(dd)
    return dd, hist, fin


dyn = {}
tail_acts = ["tail_1_pitch_act", "tail_2_pitch_act", "tail_3_pitch_act"]
neck_acts = ["neck_pitch_act", "head_pitch_act"]
cases = {
    "baseline (home ctrl)": {},
    "tail pitch residual +1 (up)": {a: +1.0 for a in tail_acts},
    "tail pitch residual -1 (down)": {a: -1.0 for a in tail_acts},
    "neck+head pitch residual -1 (up)": {a: -1.0 for a in neck_acts},
    "neck+head pitch residual +1 (down)": {a: +1.0 for a in neck_acts},
}
case_runs = []
for label, rm in cases.items():
    case_runs.append((label + " [step at t=0, raw MuJoCo]", rm, dict(substeps=5000)))
for label, rm in list(cases.items())[1:]:
    case_runs.append((label + " [ramped over 3 s, hold 5 s, raw MuJoCo]", rm, dict(substeps=4000, ramp_substeps=1500, window_start=2500)))
for half_label, half in (("tail pitch residual +0.5 (up)", {a: 0.5 for a in tail_acts}),
                         ("neck+head pitch residual -0.5 (up)", {a: -0.5 for a in neck_acts})):
    case_runs.append((half_label + " [step at t=0, raw MuJoCo]", half, dict(substeps=5000)))
for label, rm, kw in case_runs:
    ctrl = residual_ctrl(rm)
    dd, hist, fin = settle_with(ctrl, **kw)
    dyn[label] = {
        "ctrl_rad": {an: float(ctrl[aid(an)]) for an in tail_acts + neck_acts},
        "final_joint_deg": {jn: fin["joint_qpos_deg"][jn] for jn in ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch", "neck_pitch", "head_pitch")},
        "window_com_ahead_of_hip_m": float(np.mean([h["com_ahead_of_hip_m"] for h in hist])),
        "window_com_height_m": float(np.mean([h["com_height_above_floor_m"] for h in hist])),
        "window_com_proj_pad_stat": float(np.mean([h["com_proj_pad_stat_signed_mean_feet"] for h in hist])),
        "window_pad_cop_stat_r": float(np.nanmean([h["contacts"]["r"]["pad_cop_stat_signed"] if h["contacts"]["r"]["pad_cop_stat_signed"] is not None else np.nan for h in hist])),
        "window_pad_cop_stat_l": float(np.nanmean([h["contacts"]["l"]["pad_cop_stat_signed"] if h["contacts"]["l"]["pad_cop_stat_signed"] is not None else np.nan for h in hist])),
        "window_wholefoot_cop_r": float(np.nanmean([np.nan if h["contacts"]["r"]["wholefoot_cop_signed_pad_halflengths"] is None else h["contacts"]["r"]["wholefoot_cop_signed_pad_halflengths"] for h in hist])),
        "window_min_pelvis_z": float(np.min([h["pelvis_height_m"] for h in hist])),
        "window_digit_share_r": float(np.nanmean([np.nan if h["contacts"]["r"]["digit_share"] is None else h["contacts"]["r"]["digit_share"] for h in hist])),
        "window_pelvis_pitch_nose_down_deg": float(np.mean([h["legs"]["r"]["included_angles_deg"]["pelvis_pitch_nose_down"] for h in hist])),
        "final_pelvis_z": fin["pelvis_height_m"],
        "fell": bool(fin["pelvis_height_m"] < 0.7),
    }
base = dyn["baseline (home ctrl) [step at t=0, raw MuJoCo]"]
for label, v in dyn.items():
    v["delta_pad_cop_stat_r_vs_baseline"] = v["window_pad_cop_stat_r"] - base["window_pad_cop_stat_r"]
    v["delta_com_proj_stat_vs_baseline"] = v["window_com_proj_pad_stat"] - base["window_com_proj_pad_stat"]
    v["delta_com_ahead_of_hip_m_vs_baseline"] = v["window_com_ahead_of_hip_m"] - base["window_com_ahead_of_hip_m"]


env_auth = {}
for label, rm in (("neck+head residual -1 (up) via env.step (10 Hz filter)", {a: -1.0 for a in neck_acts}),
                  ("neck+head residual -0.5 (up) via env.step", {a: -0.5 for a in neck_acts}),
                  ("tail pitch residual +1 (up) via env.step", {a: 1.0 for a in tail_acts}),
                  ("tail pitch residual -1 (down) via env.step", {a: -1.0 for a in tail_acts})):
    e2 = TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0)
    e2.reset(seed=0)
    act = np.zeros(e2.action_space.shape)
    for an, rv in rm.items():
        act[aid(an)] = rv
    terminated_at = None
    zs, stats, wf = [], [], []
    for step in range(1000):
        _o, _r, term, _tr, _i = e2.step(act)
        if term and terminated_at is None:
            terminated_at = step
        zs.append(float(e2.data.qpos[2]))
        if step >= 200:
            mm = measure(e2.data)
            stats.append(mm["contacts"]["r"]["pad_cop_stat_signed"])
            wf.append(mm["contacts"]["r"]["wholefoot_cop_signed_pad_halflengths"])
    env_auth[label] = {
        "first_terminated_step": terminated_at,
        "min_pelvis_z": float(np.min(zs)),
        "final_pelvis_z": zs[-1],
        "final_joint_deg": {jn: math.degrees(e2.data.qpos[model.jnt_qposadr[jid(jn)]]) for jn in ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch", "neck_pitch", "head_pitch")},
        "window_pad_cop_stat_r_mean": float(np.nanmean([np.nan if v is None else v for v in stats])),
        "window_wholefoot_cop_r_mean": float(np.nanmean([np.nan if v is None else v for v in wf])),
    }

# Tail_1 holding budget at its ventral stop (static): servo at full up command + spring vs gravity.
q_stop = model.jnt_range[jid("tail_1_pitch")][0]
dd_t = mujoco.MjData(model)
dd_t.qpos[:] = model.key_qpos[KEY]
dd_t.qpos[model.jnt_qposadr[jid("tail_1_pitch")]] = q_stop
mujoco.mj_kinematics(model, dd_t)
mujoco.mj_comPos(model, dd_t)
g_stop, _ = gravity_torque_about(dd_t, "tail_1_pitch")
row_t1 = jrow("tail_1_pitch")
tail1_budget = {
    "q_stop_deg": math.degrees(q_stop),
    "gravity_torque_at_stop_Nm": g_stop,
    "servo_torque_full_up_at_stop_Nm": min(row_t1["kp"] * (row_t1["ctrlrange_rad"][1] - q_stop), row_t1["forcerange_Nm"][1]),
    "spring_torque_at_stop_Nm": row_t1["stiffness_Nm_per_rad"] * (0.0 - q_stop),
    "note": "distal tail joints at 0; positive torque lifts the tail",
}
tail1_budget["net_lift_margin_Nm"] = tail1_budget["servo_torque_full_up_at_stop_Nm"] + tail1_budget["spring_torque_at_stop_Nm"] + g_stop

# Frame conversions (report's one-frame table), from the measured home geometry.
Lh = home["legs"]["r"]
fp = Lh["footprint_pad_rear_to_d3_tip_incl_radius_m"]
half = Lh["pad_half_size_m"][0]
pad_c_from_rear = half
mtp_from_rear = -Lh["pad_rel_mtp"]["d3"]["pad_rear_edge_rel_mtp_m"]
d3tip = Lh["digit_d3_mtp_to_tip_incl_radius_m"]


def conv_from_stat(st):
    x = pad_c_from_rear + st * half
    return {"pad_stat_signed": st, "frac_pad_length_from_rear": x / (2 * half), "frac_footprint_from_rear": x / fp,
            "ahead_of_mtp3_m": x - mtp_from_rear, "frac_of_d3_tip_length": (x - mtp_from_rear) / d3tip}


def conv_from_fp(fr):
    x = fr * fp
    return conv_from_stat((x - pad_c_from_rear) / half)


frames = {
    "bird analogue (load under MTP III)": conv_from_stat((mtp_from_rear - pad_c_from_rear) / half),
    "pad centre": conv_from_stat(0.0),
    "human 45.7% foot length mapped onto footprint": conv_from_fp(0.457),
    "gate bar +0.80": conv_from_stat(0.80),
    "drifted stance +1.0": conv_from_stat(1.0),
    "statue whole-foot CoP (settle window mean)": conv_from_stat(window["wholefoot_cop_signed_r"]["mean"]),
    "statue pad-only CoP (settle window mean)": conv_from_stat(window["pad_cop_stat_signed_r"]["mean"]),
    "statue COM projection (settle window mean)": conv_from_stat(window["com_proj_pad_stat_signed_mean_feet"]["mean"]),
    "home COM projection": conv_from_stat(home["com_proj_pad_stat_signed_mean_feet"]),
}

# ----------------------------------------------------------------------------
# Write outputs.
# ----------------------------------------------------------------------------
result = {
    "model_xml": str(XML),
    "mujoco_version": mujoco.__version__,
    "gravity": G_MODEL,
    "masses": {"total_kg": total_mass, "accounted_kg": accounted, "segments": mass_rows, "per_side": mass_side},
    "home_keyframe": home,
    "metatarsus_rotation_matrix_home": mt_rot_home,
    "settle": {
        "procedure": "TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0), reset(seed=0), 1000 zero-action env steps (frame_skip 5, dt 0.002 -> 10 s); window = steps 200-999",
        "ctrl_applied_step0": ctrl_used,
        "home_ctrl": model.key_ctrl[KEY],
        "window_means": window,
        "window_joint_means_deg": window_joint_means_deg,
        "final_state_measure": settled_final,
        "raw_mujoco_key_ctrl_vs_env_final_max_abs_qpos": raw_vs_env_max_abs_qpos,
        "env_model_vs_raw_model_max_abs_diff": env_model_check,
        "final_qvel_max_abs": float(np.max(np.abs(env_final_qvel))),
    },
    "settled_raw_mujoco_measure": settled_raw,
    "ranges_included_angle": ranges,
    "joint_table": joint_table,
    "stiffness_groups": groups,
    "balance_authority_static_sweeps": auth,
    "gravity_torques_tail_neck_home": grav,
    "balance_authority_dynamic_settles": dyn,
    "balance_authority_env_pathway": env_auth,
    "tail_1_holding_budget_at_ventral_stop": tail1_budget,
    "frame_conversions": frames,
}
(OUT / "geometry_audit.json").write_text(json.dumps(tojson(result), indent=1))
print("wrote", OUT / "geometry_audit.json")
