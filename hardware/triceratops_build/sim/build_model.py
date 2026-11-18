"""Build the Triceratops robot model for the T20 (20 kg) and T15 (15 kg) builds.

Starts from PR #669's robot.xml (copied to base/pr669_robot.xml) and rebuilds its mass model category by category
to the checked parts list, hardware/triceratops_kickoff/PARTS_PREMIUM.md section 6.2 (18 Nov 2026):

- removes the PR's 1.84 kg reserve block, so no block stands in for parts;
- swaps the leg drives (T20: X6 roll, X4 pitch, X8 front elbow at 1:1, X6 hind knee at 1.3:1; T15: X4 roll and
  pitch, X6 flexion at 1.3:1), with joint-side torque limits 0.70 x maker peak x ratio x 0.90 efficiency and
  reflected inertia (catalog estimate) x ratio^2;
- replaces the 1:1 belt with a rigid four-bar pushrod (cranks, rod, output bearing pair), modelled as a constant
  ratio;
- adds foot load-cell cartridges, the neck yaw stage and beak servo (mass), eyelids and the heavier frill (T20),
  the XM430 tail servo(s) and a tail pitch joint (T20);
- re-masses shells, compute, battery (two 6S 3300 mAh packs in series at x = -200 mm) and the power and safety parts.

Variants: "listed" is the sum of listed parts (T20 18.83 kg, T15 14.61 kg). "closed" adds the parts list's reserve
(T20 1.17 kg, T15 0.39 kg) as growth of the shells, harness and frame structure, pro rata, not as a block.

Every mass and position added here is a planning estimate (parts list or this file), not a weighed part.

Usage: python build_model.py [--out DIR] [--battery-x -0.200]
Writes trike_<BUILD>_<VARIANT>.xml and mass_ledger_<BUILD>_<VARIANT>.json for both builds and both variants.
"""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BASE = HERE / "base" / "pr669_robot.xml"
DRIVES = json.loads((HERE / "data" / "drives.json").read_text())
ETA = DRIVES["transmission_efficiency"]
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
LEG_AXES = ("roll", "pitch", "flexion")

# Parts list section 6.2 (kg). "reserve" is closed into the "closed" variant.
BUDGET = {
    "T20": {"drives": 9.02, "frame": 3.38, "head": 1.51, "tail": 0.49, "shells": 1.25, "compute": 0.62,
            "battery": 1.01, "power": 1.55, "listed": 18.83, "reserve": 1.17, "total": 20.00},
    "T15": {"drives": 6.16, "frame": 3.03, "head": 1.33, "tail": 0.41, "shells": 0.85, "compute": 0.62,
            "battery": 1.01, "power": 1.20, "listed": 14.61, "reserve": 0.39, "total": 15.00},
}

# Drive per axis and flexion ratio (PARTS_PREMIUM.md sections 2.6 and 2.7).
BUILD_DRIVES = {
    "T20": {"roll": "X6-60", "pitch": "X4-36-L",
            "flexion": {"fore": ("X8-150", 1.0), "hind": ("X6-60", 1.3)}},
    "T15": {"roll": "X4-36-L", "pitch": "X4-36-L",
            "flexion": {"fore": ("X6-60", 1.3), "hind": ("X6-60", 1.3)}},
}

# Joint-side position-servo gains. PR #669's gains at 15 kg, scaled with mass for T20 (assumption A2).
PR_GAINS = {"roll": (150.0, 4.0), "pitch": (200.0, 5.0), "flexion": (330.0, 7.2)}

# Per-leg added transmission and mount mass over the PR's belt parts (parts list 6.2: +0.50 kg T20, +0.15 kg T15).
TRANS_DELTA_PER_LEG = {"T20": 0.125, "T15": 0.0375}
PR_BELT_PARTS_PER_LEG = 0.045  # input pulley 0.014 + two belt spans 0.012 + output pulley 0.019
FOOT_CELL_KG = 0.045  # 0.18 kg for four cells and cartridges (parts list 6.2, estimate)


def category_of(name):
    """Parts-list 6.2 category for a geom name (base or added)."""
    n = name
    if any(k in n for k in ("_roll_stator", "_pitch_stator", "_flexion_stator")):
        return "drives"
    if any(k in n for k in ("6061_", "crossmember", "service_tray", "tray_stay", "neck_cradle", "shoulder_yoke",
                            "upper_strut", "lower_strut", "elbow_axle", "ankle_hub", "_sole", "drive_crank",
                            "pushrod", "output_crank", "load_cell_cartridge")):
        return "frame"
    if n.startswith("PA11_rib") or "shell_seam" in n or "limb_cover" in n:
        return "shells"
    if n in ("xm540_w270_r_pitch_case", "neck_hub_and_skull_carrier", "frill_pa11_panel", "hollow_skull_equivalent",
             "left_brow_horn_tpu", "right_brow_horn_tpu", "nasal_horn_tpu", "left_eye", "right_eye",
             "xm540_w150_r_yaw_case", "neck_yaw_bearing", "xm430_w350_r_beak_case", "beak_linkage_and_lights",
             "left_xl330_eyelid", "right_xl330_eyelid"):
        return "head"
    if "tail" in n and ("case" in n or "shell" in n):
        return "tail"
    if n in ("jetson_orin_nano_super_kit", "nvme_and_vibration_tray", "teensy41_4can_carrier", "oak_d_pro_w",
             "oak_d_usb_c_lock_cable", "left_vl53l8cx", "right_vl53l8cx", "trunk_imu", "io_and_sensor_carriers"):
        return "compute"
    if n.startswith("gtech_6s_3300"):
        return "battery"
    if n in ("power_board_diode_or_precharge_clamp", "safety_relay_contactor_fuse_stop_rx", "dc_dc_rails",
             "left_can_and_power_loom", "right_can_and_power_loom", "drive_bay_blowers_and_ducts",
             "orion_bms_jr2_and_current_sensor"):
        return "power"
    raise KeyError(f"no category for geom {name!r}")


GROWTH_CATEGORIES = ("shells",)  # plus the harness and frame structure, matched by name below
GROWTH_FRAME = ("6061_", "crossmember", "service_tray", "tray_stay", "neck_cradle", "shoulder_yoke", "upper_strut",
                "lower_strut")


def is_growth(name):
    if category_of(name) in GROWTH_CATEGORIES:
        return True
    if name in ("left_can_and_power_loom", "right_can_and_power_loom"):
        return True
    return category_of(name) == "frame" and any(k in name for k in GROWTH_FRAME)


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def parse_vec(s):
    return [float(x) for x in s.split()]


class Model:
    def __init__(self, path):
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        self.tree = ET.parse(path, parser)
        self.root = self.tree.getroot()
        self.wb = self.root.find("worldbody")

    def body(self, name):
        for b in self.root.iter("body"):
            if b.get("name") == name:
                return b
        raise KeyError(name)

    def geom(self, name):
        for b in self.root.iter("body"):
            for g in b.findall("geom"):
                if g.get("name") == name:
                    return g, b
        raise KeyError(name)

    def remove_geom(self, name):
        g, b = self.geom(name)
        b.remove(g)
        return g

    def add_geom(self, body, **attrs):
        g = ET.SubElement(body, "geom", {k: (fmt(v) if isinstance(v, (list, tuple)) else str(v))
                                         for k, v in attrs.items()})
        # keep geoms before child bodies, so the XML stays readable
        body.remove(g)
        idx = next((i for i, c in enumerate(list(body)) if c.tag == "body"), len(list(body)))
        body.insert(idx, g)
        return g


def leg_drive(build, corner, axis):
    spec = BUILD_DRIVES[build][axis]
    if axis == "flexion":
        return spec["fore" if "fore" in corner else "hind"]
    return spec, 1.0


def build(build_name, variant, battery_x=-0.200):
    m = Model(BASE)
    r = m.root
    budget = BUDGET[build_name]
    r.set("model", f"triceratops_75cm_{build_name}_{variant}")
    r.insert(0, ET.Comment(
        f" Triceratops build model {build_name} ({variant}), generated by hardware/triceratops_build/sim/build_model.py "
        f"from PR #669's robot.xml. Masses and positions are planning estimates from PARTS_PREMIUM.md section 6.2 "
        f"(18 Nov 2026), not weighed parts. Simulation model only. "))
    chassis = m.body("chassis")

    # 1. No reserve block.
    m.remove_geom("design_mass_reserve")

    # 2. Leg drives, transmissions, feet.
    for c in CORNERS:
        side = 1 if c.startswith("left") else -1
        end = "fore" if "fore" in c else "hind"
        # roll stator on the chassis
        g, _ = m.geom(f"{c}_x4_roll_stator")
        rd = BUILD_DRIVES[build_name]["roll"]
        d = DRIVES["leg_drives"][rd]
        ft = parse_vec(g.get("fromto"))
        cx = 0.5 * (ft[0] + ft[3])
        half = d["length_m"] / 2
        g.set("name", f"{c}_{rd.split('-')[0].lower()}_roll_stator")
        g.set("fromto", fmt([cx - half, ft[1], ft[2], cx + half, ft[4], ft[5]]))
        g.set("size", f"{d['diameter_m'] / 2:.4f}")
        g.set("mass", f"{d['mass_kg']}")
        # pitch stator (X4 in both builds): rename only
        g, _ = m.geom(f"{c}_x4_pitch_stator")
        g.set("name", f"{c}_x4_pitch_stator")
        # flexion stator
        fd, ratio = BUILD_DRIVES[build_name]["flexion"][end]
        d = DRIVES["leg_drives"][fd]
        g, upper = m.geom(f"{c}_x6_flexion_stator")
        ft = parse_vec(g.get("fromto"))
        sx, sz = ft[0], ft[2]
        half = d["length_m"] / 2
        g.set("name", f"{c}_{fd.split('-')[0].lower()}_flexion_stator")
        g.set("fromto", fmt([sx, -half, sz, sx, half, sz]))
        g.set("size", f"{d['diameter_m'] / 2:.4f}")
        g.set("mass", f"{d['mass_kg']}")
        # belt -> pushrod
        ip = m.remove_geom(f"{c}_input_pulley")
        yb = 0.5 * (parse_vec(ip.get("fromto"))[1] + parse_vec(ip.get("fromto"))[4])
        m.remove_geom(f"{c}_belt_span_1")
        m.remove_geom(f"{c}_belt_span_2")
        m.remove_geom(f"{c}_output_pulley")
        strut, _ = m.geom(f"{c}_upper_strut")
        sft = parse_vec(strut.get("fromto"))
        ex, ez = sft[3], sft[5]
        L = math.hypot(ex, ez)
        ux, uz = ex / L, ez / L
        px, pz = -uz, ux  # perpendicular in the leg plane
        r_out = 0.035 if ratio == 1.0 else 0.036
        r_in = r_out / ratio
        mtot = PR_BELT_PARTS_PER_LEG + TRANS_DELTA_PER_LEG[build_name]
        m_in, m_rod, m_out = 0.4 * mtot, 0.3 * mtot, 0.3 * mtot
        m.add_geom(upper, name=f"{c}_drive_crank", type="cylinder",
                   fromto=[sx, yb - 0.006, sz, sx, yb + 0.006, sz], size=f"{r_in + 0.008:.4f}",
                   material="bracket", mass=f"{m_in:.5f}")
        m.add_geom(upper, name=f"{c}_pushrod", type="capsule",
                   fromto=[sx + r_in * px, yb, sz + r_in * pz, ex + r_out * px, yb, ez + r_out * pz],
                   size="0.005", material="rail", mass=f"{m_rod:.5f}")
        lower = m.body(f"{c}_lower_link")
        m.add_geom(lower, name=f"{c}_output_crank", type="cylinder",
                   fromto=[0, yb - 0.006, 0, 0, yb + 0.006, 0], size=f"{r_out + 0.008:.4f}",
                   material="bracket", mass=f"{m_out:.5f}")
        # foot load-cell cartridge
        foot = m.body(f"{c}_foot")
        m.add_geom(foot, name=f"{c}_load_cell_cartridge", type="box", pos=[0, 0, -0.022],
                   size="0.02 0.02 0.006", material="compute", mass=f"{FOOT_CELL_KG}")
        # limb covers (T20 only: "full limb covers at T20")
        if build_name == "T20":
            m.add_geom(upper, name=f"{c}_upper_limb_cover", type="capsule", fromto=[0, 0, 0, ex, 0, ez],
                       size="0.03", material="skin", mass="0.06", rgba=".55 .50 .40 .35")
            lft = parse_vec(m.geom(f"{c}_lower_strut")[0].get("fromto"))
            m.add_geom(lower, name=f"{c}_lower_limb_cover", type="capsule", fromto=lft, size="0.024",
                       material="skin", mass="0.04", rgba=".55 .50 .40 .35")

    # 3. Shells: body shell to 0.85 kg in both builds (parts list 6.2; T20's extra 0.40 kg is the limb covers).
    shell_geoms = [g for b in r.iter("body") for g in b.findall("geom")
                   if (g.get("name") or "").startswith("PA11_rib") or "shell_seam" in (g.get("name") or "")]
    base_shell = sum(float(g.get("mass")) for g in shell_geoms)
    for g in shell_geoms:
        g.set("mass", f"{float(g.get('mass')) * 0.85 / base_shell:.6f}")

    # 4. Head: pitch stage keeps the pitch servo and hub, gains the yaw servo and bearing; the skull, frill, horns,
    #    eyes and beak move to a yaw body (yaw axis through the pitch pivot).
    head = m.body("head")
    g, _ = m.geom("xm540_w270_t_case")
    g.set("name", "xm540_w270_r_pitch_case")
    yaw = ET.Element("body", {"name": "head_yaw", "pos": "0 0 0"})
    ET.SubElement(yaw, "joint", {"name": "neck_yaw", "axis": "0 0 1", "range": "-35 35", "damping": ".08",
                                 "armature": ".00035"})
    keep = {"xm540_w270_r_pitch_case", "neck_hub_and_skull_carrier"}
    for child in list(head):
        if child.tag in ("geom", "site") and child.get("name") not in keep:
            head.remove(child)
            yaw.append(child)
    head.append(yaw)
    m.add_geom(head, name="xm540_w150_r_yaw_case", type="box", pos=[0.035, 0, 0.032],
               size="0.01675 0.022 0.0295", material="rotor", mass="0.165")
    m.add_geom(head, name="neck_yaw_bearing", type="cylinder", fromto=[0.06, 0, 0.0, 0.06, 0, 0.008],
               size="0.03", material="bracket", mass="0.04")
    m.add_geom(yaw, name="xm430_w350_r_beak_case", type="box", pos=[0.13, 0, -0.012],
               size="0.0143 0.0172 0.0232", material="rotor", mass="0.082")
    m.add_geom(yaw, name="beak_linkage_and_lights", type="box", pos=[0.02, 0, 0.09],
               size="0.02 0.03 0.01", material="compute", mass="0.043")
    if build_name == "T20":
        for s, y in (("left", 0.07), ("right", -0.07)):
            m.add_geom(yaw, name=f"{s}_xl330_eyelid", type="box", pos=[0.092, y, 0.022],
                       size="0.010 0.0068 0.0101", material="rotor", mass="0.015")
        g, _ = m.geom("frill_pa11_panel")
        g.set("mass", f"{float(g.get('mass')) + 0.15:.3f}")
    nj = head.find("joint")
    nj.set("range", "-20 20")  # parts list: neck pitch +/-20 deg, spring-assisted

    # 5. Tail: XM430 yaw case replaces the STS3215; T20 adds a pitch joint and a second XM430.
    g, _ = m.geom("sts3215_tail_case")
    g.set("name", "xm430_w350_r_tail_yaw_case")
    g.set("size", "0.0143 0.0172 0.0232")
    g.set("mass", "0.082")
    tail = m.body("tail")
    if build_name == "T20":
        tj = tail.find("joint")
        idx = list(tail).index(tj)
        tail.insert(idx + 1, ET.Element("joint", {"name": "tail_pitch", "axis": "0 1 0", "range": "-15 15",
                                                   "damping": ".09", "armature": ".0003"}))
        m.add_geom(tail, name="xm430_w350_r_tail_pitch_case", type="box", pos=[-0.03, 0, 0.0],
                   size="0.0143 0.0172 0.0232", material="rotor", mass="0.082")

    # 6. Compute and sensors (0.62 kg).
    pi = m.remove_geom("pi5_ai_hat_enclosure")
    m.add_geom(chassis, name="jetson_orin_nano_super_kit", type="box", pos=[-0.018, 0, -0.046],
               size="0.05 0.04 0.0175", material="compute", mass="0.174")
    m.add_geom(chassis, name="nvme_and_vibration_tray", type="box", pos=[-0.018, 0, -0.067],
               size="0.06 0.05 0.003", material="bracket", mass="0.10")
    g, _ = m.geom("teensy41_dual_can")
    g.set("name", "teensy41_4can_carrier")
    g, _ = m.geom("oak_d_lite")
    g.set("name", "oak_d_pro_w")
    g.set("mass", "0.091")
    m.add_geom(chassis, name="oak_d_usb_c_lock_cable", type="capsule", fromto=[0.25, 0, 0.055, 0.15, 0, 0.0],
               size="0.003", material="harness", mass="0.03")
    for s in ("left", "right"):
        g, _ = m.geom(f"{s}_vl53l5cx")
        g.set("name", f"{s}_vl53l8cx")
    g, _ = m.geom("io_and_sensor_carriers")
    g.set("mass", "0.105")
    cam = next(c for c in r.iter("camera") if c.get("name") == "oak_d_lite_rgb")
    cam.set("name", "oak_d_pro_w_rgb")

    # 7. Battery: two 6S 3300 mAh packs (138 x 42 x 43 mm, 0.506 kg each) side by side, 12S in series.
    m.remove_geom("6s_5ah_pack")
    for k, y in ((1, 0.0215), (2, -0.0215)):
        m.add_geom(chassis, name=f"gtech_6s_3300_{k}", type="box", pos=[battery_x, y, -0.0485],
                   size="0.069 0.021 0.0215", material="power", mass="0.506")

    # 8. Power, safety, wiring, cooling (T15 1.20 kg, T20 1.55 kg).
    g, _ = m.geom("fused_pdu_disconnect")
    g.set("name", "power_board_diode_or_precharge_clamp")
    g.set("mass", "0.25")
    m.add_geom(chassis, name="safety_relay_contactor_fuse_stop_rx", type="box", pos=[0.07, -0.05, -0.05],
               size="0.035 0.025 0.02", material="power", mass="0.33")
    g, _ = m.geom("regulated_logic_rails")
    g.set("name", "dc_dc_rails")
    g, _ = m.geom("cooling_and_fan")
    g.set("name", "drive_bay_blowers_and_ducts")
    if build_name == "T20":
        m.add_geom(chassis, name="orion_bms_jr2_and_current_sensor", type="box", pos=[-0.105, 0, -0.05],
                   size="0.024 0.05 0.02", material="compute", mass="0.35")

    # 9. Closed variant: reserve as pro-rata growth of shells, harness and frame structure.
    all_geoms = [g for b in r.iter("body") for g in b.findall("geom") if g.get("mass") and float(g.get("mass")) > 0]
    growth = [g for g in all_geoms if is_growth(g.get("name"))]
    if variant == "closed":
        gsum = sum(float(g.get("mass")) for g in growth)
        f = (gsum + budget["reserve"]) / gsum
        for g in growth:
            g.set("mass", f"{float(g.get('mass')) * f:.6f}")

    # 10. Actuators and sensors: rewrite the leg actuators with joint-side limits; add neck yaw and tail pitch.
    scale = BUDGET[build_name]["total"] / 15.0
    act = r.find("actuator")
    for a in list(act):
        act.remove(a)
    drive_table = {}
    for c in CORNERS:
        for axis in LEG_AXES:
            dname, ratio = leg_drive(build_name, c, axis)
            d = DRIVES["leg_drives"][dname]
            eta = ETA if axis == "flexion" else 1.0
            limit = 0.70 * d["peak_nm"] * ratio * eta
            kp, kv = PR_GAINS[axis]
            j = next(jj for jj in r.iter("joint") if jj.get("name") == f"{c}_{axis}")
            lo, hi = parse_vec(j.get("range"))
            j.set("armature", f"{d['armature_kgm2'] * ratio ** 2:.6f}")
            ET.SubElement(act, "position", {
                "name": f"{c}_{axis}", "joint": f"{c}_{axis}", "kp": f"{kp * scale:.4g}", "kv": f"{kv * scale:.4g}",
                "ctrlrange": f"{math.radians(lo):.6f} {math.radians(hi):.6f}", "forcerange": f"{-limit:.4f} {limit:.4f}"})
            drive_table[f"{c}_{axis}"] = {"drive": dname, "ratio": ratio, "eta": eta,
                                          "joint_peak_limit_nm": round(limit, 4),
                                          "joint_thermal_eq_nm": round(d["thermal_eq_nm"] * ratio * eta, 4),
                                          "armature_joint_kgm2": round(d["armature_kgm2"] * ratio ** 2, 6)}
    servos = DRIVES["servos"]
    for name, joint, sv, kp, kv in (("neck_pitch", "neck_pitch", "XM540-W270-R", 14, 0.6),
                                    ("neck_yaw", "neck_yaw", "XM540-W150-R", 10, 0.4),
                                    ("tail_yaw", "tail_yaw", "XM430-W350-R", 8, 0.25)) + (
            (("tail_pitch", "tail_pitch", "XM430-W350-R", 15, 0.4),) if build_name == "T20" else ()):
        lim = 0.70 * servos[sv]["peak_nm"]
        j = next(jj for jj in r.iter("joint") if jj.get("name") == joint)
        lo, hi = parse_vec(j.get("range"))
        ET.SubElement(act, "position", {"name": name, "joint": joint, "kp": str(kp), "kv": str(kv),
                                        "ctrlrange": f"{math.radians(lo):.6f} {math.radians(hi):.6f}",
                                        "forcerange": f"{-lim:.4f} {lim:.4f}"})
        drive_table[name] = {"drive": sv, "ratio": 1.0, "eta": 1.0, "joint_peak_limit_nm": round(lim, 4),
                             "joint_thermal_eq_nm": servos[sv]["rated_nm"]}
    sens = r.find("sensor")
    for s in list(sens):
        if s.tag == "actuatorfrc":
            sens.remove(s)
    for a in act:
        ET.SubElement(sens, "actuatorfrc", {"name": f"{a.get('name')}_effort", "actuator": a.get("name")})
    ET.SubElement(sens, "jointpos", {"name": "neck_yaw_angle", "joint": "neck_yaw"})
    if build_name == "T20":
        ET.SubElement(sens, "jointpos", {"name": "tail_pitch_angle", "joint": "tail_pitch"})
    return m, drive_table


def set_neck_spring(xml_text, frac=0.70):
    """Set the neck spring to cancel `frac` of the head's gravity moment at neutral (parts list: 70 %)."""
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jid = model.joint("neck_pitch").id
    dof = model.jnt_dofadr[jid]
    # gravity torque about the pitch axis = qfrc_bias at zero velocity minus the spring's own contribution
    spring = -model.jnt_stiffness[jid] * (data.qpos[model.jnt_qposadr[jid]] - model.qpos_spring[model.jnt_qposadr[jid]])
    grav = data.qfrc_bias[dof]
    k = 0.70
    ref_rad = frac * grav / k  # spring torque k*(ref - q) at q = 0 supplies frac of the holding torque qfrc_bias
    head = model.body("head").id
    com = data.subtree_com[head] - data.xanchor[jid]
    info = {"head_pitch_subtree_kg": float(model.body_subtreemass[head]),
            "head_com_from_pitch_pivot_m": [round(float(x), 4) for x in com],
            "head_com_distance_m": round(float(np.linalg.norm(com)), 4),
            "gravity_moment_nm": round(float(grav), 4), "spring_stiffness_nm_per_rad": k,
            "springref_deg": round(math.degrees(ref_rad), 3), "spring_fraction": frac,
            "spring_before_nm": round(float(spring), 4)}
    return ref_rad, info


def ledger(xml_text, build_name, variant, drive_table, neck):
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = ET.fromstring(xml_text)
    rows = []
    for b in root.iter("body"):
        for g in b.findall("geom"):
            if g.get("mass") and float(g.get("mass")) > 0:
                gid = model.geom(g.get("name")).id
                rows.append({"geom": g.get("name"), "body": b.get("name"), "category": category_of(g.get("name")),
                             "mass_kg": float(g.get("mass")),
                             "world_pos_m": [round(float(x), 4) for x in data.geom_xpos[gid]]})
    cats = {}
    for row in rows:
        cats[row["category"]] = cats.get(row["category"], 0.0) + row["mass_kg"]
    total = float(model.body_mass.sum())
    com = data.subtree_com[1]
    hips = 0.325
    budget = BUDGET[build_name]
    return {
        "build": build_name, "variant": variant,
        "source": "PARTS_PREMIUM.md section 6.2 (18 Nov 2026) budgets; positions are this model's estimates",
        "total_mass_kg": round(total, 4),
        "budget_total_kg": budget["listed"] if variant == "listed" else budget["total"],
        "category_kg": {k: round(v, 4) for k, v in sorted(cats.items())},
        "budget_category_kg": {k: budget[k] for k in ("drives", "frame", "head", "tail", "shells", "compute",
                                                       "battery", "power")},
        "com_world_m": [round(float(x), 4) for x in com],
        "front_share_static_estimate": round((com[0] + hips) / (2 * hips), 4),
        "front_share_note": "lever estimate (x_com + 325 mm) / 650 mm at the nominal hip stations; the standing "
                            "check measures the actual foot loads",
        "neck": neck, "drives": drive_table, "parts": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "generated"))
    ap.add_argument("--battery-x", type=float, default=-0.200)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for b in ("T20", "T15"):
        for v in ("listed", "closed"):
            m, dt = build(b, v, args.battery_x)
            ET.indent(m.tree, space="  ")
            text = ET.tostring(m.root, encoding="unicode")
            ref, neck = set_neck_spring(text)
            nj = next(j for j in m.root.iter("joint") if j.get("name") == "neck_pitch")
            nj.set("stiffness", "0.70")
            nj.set("springref", f"{math.degrees(ref):.3f}")
            text = ET.tostring(m.root, encoding="unicode") + "\n"
            path = out / f"trike_{b}_{v}.xml"
            path.write_text(text)
            led = ledger(text, b, v, dt, neck)
            (out / f"mass_ledger_{b}_{v}.json").write_text(json.dumps(led, indent=1) + "\n")
            print(f"{path.name}: {led['total_mass_kg']:.3f} kg (budget {led['budget_total_kg']}), "
                  f"CoM x {1000 * led['com_world_m'][0]:+.1f} mm, front share (lever estimate) "
                  f"{100 * led['front_share_static_estimate']:.1f} %, head CoM {1000 * neck['head_com_distance_m']:.0f} mm "
                  f"from the pitch pivot, neck spring ref {neck['springref_deg']:.1f} deg")
            print("   categories:", {k: (led['category_kg'][k], led['budget_category_kg'][k])
                                     for k in led['budget_category_kg']})


if __name__ == "__main__":
    main()
