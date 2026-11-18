"""Build the 60 cm Triceratops model (owner's decision D2, 18 Nov 2026: option C).

Starts from PR #669's 75 cm robot.xml (base/pr669_robot.xml) and scales its geometry by s = 0.8, which reproduces PR
#669's own 60 cm companion figures (600 mm crown, 1,200 mm long, 320 mm hips, 188 / 184 mm links, 520 x 280 mm
footprint). Drives, servos and bought electronics do not shrink: they keep their catalog sizes and masses. The flexion
drive keeps PR #669's 90 mm offset along the upper link and the crank lane its unscaled position (PR #669 holds both
fixed between sizes), so the catalog-size housings clear their neighbours.

Option C: 10 x MyActuator X4-36-L (roll, pitch, hind knees) + 2 x X6 (front elbows), all at 1:1 through parallelogram
pushrods (a constant ratio, unlike the 75 cm 1.3:1 four-bar); 48 V (12S) bus; full expression package (moving: neck
pitch and yaw, tail yaw and pitch, on ROBOTIS XC430 / XL430; the beak servo and the eyelids as mass only, their joints
not modelled yet); fail-safe radio stop; no foot load cells in the core build (the model keeps ideal touch sensors).

Masses are planning estimates, each with its basis in the ledger, not weighed parts. The "closed" variant adds a 10 %
reserve: 0.15 kg of small parts on the chassis, the rest spread pro rata over every part whose mass is an estimate
(EST), not over catalog parts.

Revision 2 (18 Nov 2026, after the independent check; "changed after a result": the first standing and crawl runs used
revision 1, commit 6739087d): flexion drive at 90 mm, crank lane unscaled, bought electronics at catalog size, body shell
on the budget's 0.85 kg x s^2 basis, linkage mass on T15's four-bar basis, reserve spread over EST parts, ankle springs
scaled for similarity, servo joint parameters made consistent, leftover belt parts removed, audio / bus adapter / second
Teensy added. Details: REPORT_60CM.md.

Usage: python build_model60.py [--out generated] [--ankle-roll]
Writes trike60_C_listed.xml, trike60_C_closed.xml (+ _ankleroll variants with --ankle-roll) and their mass ledgers.
"""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import build_model as bm   # Model, parse_vec, fmt, set_neck_spring, DRIVES (the 75 cm generator; unchanged)

HERE = Path(__file__).resolve().parent
S = 0.8
CORNERS = bm.CORNERS
DRIVES = bm.DRIVES["leg_drives"]
ETA = bm.ETA

# Per-axis drives for option C: (drive, ratio)
AXIS_DRIVE = {"roll": ("X4-36-L", 1.0), "pitch": ("X4-36-L", 1.0),
              ("flexion", "fore"): ("X6-60", 1.0), ("flexion", "hind"): ("X4-36-L", 1.0)}

# Head and tail servos (ROBOTIS DYNAMIXEL, -T TTL versions): mass kg, position-servo force limit = 0.70 x stall at 12 V
# (stall from web-search summaries 18 Nov 2026: XC430-W240 1.9 N m, XL430-W250 1.5 N m), kp, kv (estimates).
SERVO60 = {"XC430-W240-T": dict(kg=0.065, limit=0.70 * 1.9), "XL430-W250-T": dict(kg=0.057, limit=0.70 * 1.5),
           "XL330-M288-T": dict(kg=0.018)}

STRUCT_MASS_EXP = 2.5   # structural parts' mass scales as s^2.5 (between area and volume; EST; the budget review's range)
LIMB_COVER_KG = (0.06 * S ** 2, 0.04 * S ** 2)   # upper / lower limb covers per leg (EST, T20's covers scaled by area)
BODY_SHELL_KG = 0.85 * S ** 2   # the checked parts list's body shell (6.2), scaled by area, as size_budget.py does
LINKAGE_KG = 0.0825   # per leg: T15's 75 cm four-bar (build_model.py), not scaled down: the cranks carry the same torque
FLEX_OFFSET_M = 0.090  # flexion drive axis along the upper link from the pitch axis (PR #669's OpenSCAD, both sizes)
# Ankle springs scaled for similarity (torque per radian proportional to weight x length): 75 cm T20 (20 kg) values
# stiffness 3 N m/rad, damping 0.15 N m s/rad; damping also x sqrt(s) for the time scale.
ANKLE_K = 3.0 * (12.44 / 20.0) * S
ANKLE_C = 0.15 * (12.44 / 20.0) * S * math.sqrt(S)
SERVO_JOINT = dict(damping="0.04", armature="0.0002")   # all four XC430 / XL430 joints (EST; one set for consistency)
# Bought electronics: catalog-size boxes (half extents, m) and their chassis positions (m, chassis frame), laid out so
# no two boxes intersect (revision 2; check_layout60.py). All sizes EST: the Jetson dev kit with its fan about 100 x 80 x
# 35 mm; the Pilz PNOZ s5 (a 22.5 mm DIN-rail relay, about 100 x 120 mm in its other two dimensions) lying flat with the
# contactor, watchdog board and radio-stop receiver in one 121 x 100 x 45 mm block.
ELECTRONICS = {
    "jetson_orin_nano_super_kit": ((0.050, 0.040, 0.0175), (-0.030, 0.0, -0.030)),
    "nvme_and_vibration_tray": ((0.052, 0.042, 0.003), (-0.030, 0.0, -0.0505)),
    "pilz_relay_contactor_watchdog_radio_rx": ((0.0605, 0.050, 0.0225), (0.082, 0.0, -0.0335)),
    "teensy41_can_carriers_x2": ((0.045, 0.030, 0.010), (0.080, 0.0, 0.0)),
    "power_board_diode_or_precharge_clamp": ((0.040, 0.035, 0.012), (0.185, 0.0, -0.030)),
    "dc_dc_rails_and_pack_monitor": ((0.030, 0.035, 0.012), (0.185, 0.0, 0.0)),
    "io_and_sensor_carriers": ((0.025, 0.030, 0.008), (-0.190, 0.0, -0.001)),
    "audio_and_servo_bus_adapter": ((0.025, 0.030, 0.010), (-0.100, 0.0, 0.0)),
}


def scale_attr(el, name, f=S):
    v = el.get(name)
    if v is not None:
        el.set(name, bm.fmt([x * f for x in bm.parse_vec(v)]))


def scale_geometry(m):
    """Scale every position and size by S, except the floor plane and the default classes."""
    r = m.root
    for mesh in r.iter("mesh"):
        mesh.set("scale", bm.fmt([S, S, S]))
    wb = r.find("worldbody")
    for b in wb.iter("body"):
        scale_attr(b, "pos")
    for g in wb.iter("geom"):
        if g.get("type") == "plane":
            continue
        for a in ("pos", "fromto", "size"):
            scale_attr(g, a)
    for el in list(wb.iter("site")) + list(wb.iter("camera")) + list(wb.iter("joint")):
        for a in ("pos", "fromto", "size"):
            scale_attr(el, a)
    st = r.find("statistic")
    if st is not None:
        scale_attr(st, "center")
        if st.get("extent"):
            st.set("extent", f"{float(st.get('extent')) * S:.4g}")
            st.set("meansize", f"{float(st.get('meansize')) * S:.4g}")


def build(variant, ankle_roll=False, battery_x=-0.200 * S):
    m = bm.Model(bm.BASE)
    r = m.root
    r.set("model", f"triceratops_60cm_C_{variant}" + ("_ankleroll" if ankle_roll else ""))
    r.insert(0, ET.Comment(
        " Triceratops 60 cm build model (option C), generated by hardware/triceratops_build/sim/build_model60.py from PR "
        "#669's robot.xml scaled by 0.8. Drives and servos keep their catalog sizes. Masses are planning estimates. "
        "Simulation model only. "))
    scale_geometry(m)
    chassis = m.body("chassis")
    ledger_notes = {}

    def note(name, basis):
        ledger_notes[name] = basis

    m.remove_geom("design_mass_reserve")

    # ---- structure: every structural geom's mass scales as s^2.5 (EST) ----
    struct_keys = ("6061_", "crossmember", "service_tray", "tray_stay", "neck_cradle", "shoulder_yoke", "upper_strut",
                   "lower_strut", "elbow_axle", "ankle_hub", "_sole")
    for b in r.iter("body"):
        for g in b.findall("geom"):
            n = g.get("name") or ""
            if g.get("mass") and any(k in n for k in struct_keys):
                g.set("mass", f"{float(g.get('mass')) * S ** STRUCT_MASS_EXP:.5f}")
                note(n, f"PR #669 part x 0.8^{STRUCT_MASS_EXP} (EST)")

    # ---- passive ankles: springs scaled for similarity (ANKLE_K, ANKLE_C) ----
    for c in CORNERS:
        aj = m.body(f"{c}_foot").find("joint")
        aj.set("stiffness", f"{ANKLE_K:.4f}")
        aj.set("damping", f"{ANKLE_C:.4f}")

    # ---- leg drives (catalog size and mass) and 1:1 parallelogram pushrods ----
    for c in CORNERS:
        end = "fore" if "fore" in c else "hind"
        d = DRIVES["X4-36-L"]
        g, _ = m.geom(f"{c}_x4_roll_stator")
        ft = bm.parse_vec(g.get("fromto"))
        cx, half = 0.5 * (ft[0] + ft[3]), d["length_m"] / 2
        g.set("fromto", bm.fmt([cx - half, ft[1], ft[2], cx + half, ft[4], ft[5]]))
        g.set("size", f"{d['diameter_m'] / 2:.4f}")
        g.set("mass", f"{d['mass_kg']}")
        note(g.get("name"), "MyActuator RMD-X4-P36-36-E-L, catalog 0.36 kg, 55 x 61 mm")
        g, _ = m.geom(f"{c}_x4_pitch_stator")
        ft = bm.parse_vec(g.get("fromto"))
        cy = 0.5 * (ft[1] + ft[4])
        sgn = 1 if ft[4] > ft[1] else -1
        g.set("fromto", bm.fmt([ft[0], cy - sgn * half, ft[2], ft[3], cy + sgn * half, ft[5]]))
        g.set("size", f"{d['diameter_m'] / 2:.4f}")
        g.set("mass", f"{d['mass_kg']}")
        note(g.get("name"), "MyActuator RMD-X4-P36-36-E-L, catalog")
        fd, ratio = AXIS_DRIVE[("flexion", end)]
        D = DRIVES[fd]
        g, upper = m.geom(f"{c}_x6_flexion_stator")
        ft = bm.parse_vec(g.get("fromto"))
        sx, sz = ft[0] / S, ft[2] / S   # PR #669's installed offset, 90 mm along the upper link at both sizes
        assert abs(math.hypot(sx, sz) - FLEX_OFFSET_M) < 1e-4, math.hypot(sx, sz)
        hl = D["length_m"] / 2
        g.set("name", f"{c}_{fd.split('-')[0].lower()}_flexion_stator")
        g.set("fromto", bm.fmt([sx, -hl, sz, sx, hl, sz]))
        g.set("size", f"{D['diameter_m'] / 2:.4f}")
        g.set("mass", f"{D['mass_kg']}")
        note(g.get("name"), f"MyActuator {fd}, catalog {D['mass_kg']} kg" +
             (" (the store's X6 listing shows 0.78 kg, the X6-50-L; F9)" if fd == "X6-60" else ""))
        ip = m.remove_geom(f"{c}_input_pulley")
        yb = 0.5 * (bm.parse_vec(ip.get("fromto"))[1] + bm.parse_vec(ip.get("fromto"))[4]) / S   # unscaled lane
        for gone in ("belt_span_1", "belt_span_2", "output_pulley", "belt_guard_spine", "adjustable_idler",
                     "flexion_motor_cap", "pitch_bearing_cap"):
            m.remove_geom(f"{c}_{gone}")
        strut, _ = m.geom(f"{c}_upper_strut")
        sft = bm.parse_vec(strut.get("fromto"))
        ex, ez = sft[3], sft[5]
        L = math.hypot(ex, ez)
        px, pz = -ez / L, ex / L
        rc = 0.028   # crank radius at both ends, 1:1 parallelogram (EST: about 0.8 x the 75 cm 35 mm)
        mt = LINKAGE_KG   # cranks, rod, rod ends and output bearing pair per leg
        m.add_geom(upper, name=f"{c}_drive_crank", type="cylinder", fromto=[sx, yb - 0.005, sz, sx, yb + 0.005, sz],
                   size=f"{rc + 0.007:.4f}", material="bracket", mass=f"{0.4 * mt:.4f}")
        m.add_geom(upper, name=f"{c}_pushrod", type="capsule",
                   fromto=[sx + rc * px, yb, sz + rc * pz, ex + rc * px, yb, ez + rc * pz], size="0.0045",
                   material="rail", mass=f"{0.3 * mt:.4f}")
        lower = m.body(f"{c}_lower_link")
        m.add_geom(lower, name=f"{c}_output_crank", type="cylinder", fromto=[0, yb - 0.005, 0, 0, yb + 0.005, 0],
                   size=f"{rc + 0.007:.4f}", material="bracket", mass=f"{0.3 * mt:.4f}")
        for n in (f"{c}_drive_crank", f"{c}_pushrod", f"{c}_output_crank"):
            note(n, f"1:1 parallelogram pushrod, {LINKAGE_KG} kg per leg: T15's 75 cm four-bar, not scaled (EST)")
        lft = bm.parse_vec(m.geom(f"{c}_lower_strut")[0].get("fromto"))
        m.add_geom(upper, name=f"{c}_upper_limb_cover", type="capsule", fromto=[0, 0, 0, ex, 0, ez], size="0.024",
                   material="skin", mass=f"{LIMB_COVER_KG[0]:.4f}", rgba=".55 .50 .40 .35")
        m.add_geom(lower, name=f"{c}_lower_limb_cover", type="capsule", fromto=lft, size="0.019", material="skin",
                   mass=f"{LIMB_COVER_KG[1]:.4f}", rgba=".55 .50 .40 .35")
        for n in (f"{c}_upper_limb_cover", f"{c}_lower_limb_cover"):
            note(n, "MJF limb cover, T20's scaled by area (EST)")
        if ankle_roll:
            foot = m.body(f"{c}_foot")
            aj = foot.find("joint")
            rj = ET.Element("joint", {k: v for k, v in aj.attrib.items() if k not in ("name", "axis", "range")})
            rj.attrib.update({"name": f"{c}_ankle_roll", "axis": "1 0 0", "range": "-20 20"})
            foot.insert(list(foot).index(aj) + 1, rj)

    # massless collision twins for the drive housings
    for b in r.iter("body"):
        for g in list(b.findall("geom")):
            n = g.get("name") or ""
            if n.endswith("_stator"):
                twin = {k: v for k, v in g.attrib.items() if k in ("type", "fromto", "size", "pos")}
                cls = "chassis_collision" if b.get("name") == "chassis" else "upper_collision"
                m.add_geom(b, name=n + "_collision", mass="0", rgba="0 0 0 0", group="3", **{"class": cls}, **twin)

    # ---- body shell: the checked parts list's 0.85 kg x s^2 (as size_budget.py), spread over PR #669's shell parts ----
    shell = [g for b in r.iter("body") for g in b.findall("geom")
             if (g.get("name") or "").startswith("PA11_rib") or "shell_seam" in (g.get("name") or "")]
    base_shell = sum(float(g.get("mass")) for g in shell)
    for g in shell:
        g.set("mass", f"{float(g.get('mass')) * BODY_SHELL_KG / base_shell:.6f}")
        note(g.get("name"), "body shell 0.85 kg (parts list 6.2) x 0.8^2, pro rata over PR #669's shell parts (EST)")

    # ---- head: pitch stage (XC430 pitch + yaw servos, bearing), yaw stage (skull, frill, horns, beak XC430, eyelids) ----
    head = m.body("head")
    for n in ("frill_pa11_panel", "hollow_skull_equivalent", "left_brow_horn_tpu", "right_brow_horn_tpu",
              "nasal_horn_tpu", "left_eye", "right_eye", "neck_hub_and_skull_carrier"):
        g, _ = m.geom(n)
        g.set("mass", f"{float(g.get('mass')) * S ** 3:.5f}")
        note(n, "PR #669 head part x 0.8^3 (EST)")
    g, _ = m.geom("xm540_w270_t_case")
    g.set("name", "xc430_w240_neck_pitch_case")
    g.set("size", "0.0143 0.017 0.0233")
    g.set("mass", f"{SERVO60['XC430-W240-T']['kg']}")
    note(g.get("name"), "ROBOTIS XC430-W240-T, 65 g (EST from the XC430 family)")
    yaw = ET.Element("body", {"name": "head_yaw", "pos": "0 0 0"})
    ET.SubElement(yaw, "joint", {"name": "neck_yaw", "axis": "0 0 1", "range": "-35 35", **SERVO_JOINT})
    keep = {"xc430_w240_neck_pitch_case", "neck_hub_and_skull_carrier"}
    for child in list(head):
        if child.tag in ("geom", "site") and child.get("name") not in keep:
            head.remove(child)
            yaw.append(child)
    head.append(yaw)
    for name, pos, kg in (("xc430_w240_neck_yaw_case", [0.028, 0, 0.026], SERVO60["XC430-W240-T"]["kg"]),
                          ("neck_yaw_bearing", None, 0.03)):
        if pos:
            m.add_geom(head, name=name, type="box", pos=pos, size="0.0143 0.017 0.0233", material="rotor", mass=f"{kg}")
        else:
            m.add_geom(head, name=name, type="cylinder", fromto=[0.048, 0, 0.0, 0.048, 0, 0.006], size="0.024",
                       material="bracket", mass=f"{kg}")
    note("xc430_w240_neck_yaw_case", "ROBOTIS XC430-W240-T (EST mass)")
    note("neck_yaw_bearing", "thin-section bearing, scaled (EST)")
    m.add_geom(yaw, name="xc430_w240_beak_case", type="box", pos=[0.104, 0, -0.010], size="0.0143 0.017 0.0233",
               material="rotor", mass=f"{SERVO60['XC430-W240-T']['kg']}")
    note("xc430_w240_beak_case", "ROBOTIS XC430-W240-T, beak (mass only; no beak joint yet)")
    for s_, y in (("left", 0.056), ("right", -0.056)):
        m.add_geom(yaw, name=f"{s_}_xl330_eyelid", type="box", pos=[0.074, y, 0.018], size="0.010 0.017 0.013",
                   material="rotor", mass=f"{SERVO60['XL330-M288-T']['kg']}")
        note(f"{s_}_xl330_eyelid", "ROBOTIS XL330-M288-T, 18 g (PL)")
    m.add_geom(yaw, name="frill_lights_and_linkage", type="box", pos=[0.016, 0, 0.072], size="0.016 0.024 0.008",
               material="compute", mass="0.035")
    note("frill_lights_and_linkage", "LEDs, diffusers, beak linkage (EST)")
    nj = head.find("joint")
    nj.set("range", "-20 20")
    nj.attrib.update(SERVO_JOINT)

    # ---- tail: XL430 yaw on the chassis, XL430 pitch in the tail ----
    g, _ = m.geom("sts3215_tail_case")
    g.set("name", "xl430_w250_tail_yaw_case")
    g.set("size", "0.0143 0.017 0.0233")
    g.set("mass", f"{SERVO60['XL430-W250-T']['kg']}")
    note(g.get("name"), "ROBOTIS XL430-W250-T, 57 g (WS)")
    p = bm.parse_vec(g.get("pos"))
    g.set("pos", bm.fmt([p[0], p[1], 0.038]))   # on top of the pelvic crossmember, which the catalog-size case crossed
    tail = m.body("tail")
    g, _ = m.geom("tail_shell_and_flex_core")
    g.set("mass", f"{float(g.get('mass')) * S ** 3:.5f}")
    note("tail_shell_and_flex_core", "PR #669 tail x 0.8^3 (EST)")
    tj = tail.find("joint")
    tj.attrib.update(SERVO_JOINT)
    tail.insert(list(tail).index(tj) + 1, ET.Element("joint", {"name": "tail_pitch", "axis": "0 1 0", "range": "-15 15",
                                                                **SERVO_JOINT}))
    m.add_geom(tail, name="xl430_w250_tail_pitch_case", type="box", pos=[-0.024, 0, 0.0], size="0.0143 0.017 0.0233",
               material="rotor", mass=f"{SERVO60['XL430-W250-T']['kg']}")
    note("xl430_w250_tail_pitch_case", "ROBOTIS XL430-W250-T (WS)")

    # ---- compute and sensors ----
    m.remove_geom("pi5_ai_hat_enclosure")
    m.add_geom(chassis, name="jetson_orin_nano_super_kit", type="box", pos=[0, 0, 0], size="0.05 0.04 0.0175",
               material="compute", mass="0.174")
    m.add_geom(chassis, name="nvme_and_vibration_tray", type="box", pos=[0, 0, 0], size="0.05 0.042 0.003",
               material="bracket", mass="0.08")
    note("jetson_orin_nano_super_kit", "NVIDIA Jetson Orin Nano Super dev kit, 0.174 kg (RC)")
    note("nvme_and_vibration_tray", "NVMe + tray (EST)")
    for old, new, kg, basis in (("teensy41_dual_can", "teensy41_can_carriers_x2", 0.10,
                                 "2 x Teensy 4.1 + CAN carrier, as budgeted (EST)"),
                                ("oak_d_lite", "oak_d_lite", 0.061, "Luxonis OAK-D Lite, 61 g (RC)"),
                                ("io_and_sensor_carriers", "io_and_sensor_carriers", 0.08, "I/O carriers, BNO085 (EST)")):
        g, _ = m.geom(old)
        g.set("name", new)
        g.set("mass", f"{kg}")
        note(new, basis)
    for s_ in ("left", "right"):
        g, _ = m.geom(f"{s_}_vl53l5cx")
        g.set("name", f"{s_}_vl53l8cx")
        p = bm.parse_vec(g.get("pos"))
        g.set("pos", bm.fmt([p[0], p[1], -0.040]))   # 20 mm lower than the scaled position: clears the X4 roll drive
        site = next(x for x in chassis.iter("site") if x.get("name") == f"{s_}_tof_ray")
        sp = bm.parse_vec(site.get("pos"))
        site.set("pos", bm.fmt([sp[0], sp[1], sp[2] - 0.020]))
        note(f"{s_}_vl53l8cx", "Pololu VL53L8CX carrier (RC)")
    g, _ = m.geom("trunk_imu")
    g.set("mass", "0.005")
    note("trunk_imu", "Adafruit BNO085 (MD)")
    m.add_geom(chassis, name="oak_d_usb_c_lock_cable", type="capsule", fromto=[0.19, 0, 0.05, 0.12, 0, 0.03], size="0.003",
               material="harness", mass="0.025")
    note("oak_d_usb_c_lock_cable", "locking USB-C cable (EST)")
    cam = next(c for c in r.iter("camera") if c.get("name") == "oak_d_lite_rgb")
    cam.set("name", "oak_d_lite_rgb")

    # ---- battery: two 6S 3300 packs in series (12S), side by side ----
    m.remove_geom("6s_5ah_pack")
    tray_top = -0.056   # the scaled service tray's top surface (-0.075 x 0.8 + 0.004)
    for k, y in ((1, 0.0215), (2, -0.0215)):
        m.add_geom(chassis, name=f"gtech_6s_3300_{k}", type="box", pos=[battery_x, y, tray_top + 0.0215],
                   size="0.069 0.021 0.0215", material="power", mass="0.506")
        note(f"gtech_6s_3300_{k}", "Gens ace G-Tech 6S 3300, 506 g, 138 x 42 x 43 mm (RC/PL)")

    # ---- power, safety, wiring, cooling ----
    g, _ = m.geom("fused_pdu_disconnect")
    g.set("name", "power_board_diode_or_precharge_clamp")
    g.set("mass", "0.22")
    note(g.get("name"), "power board, ideal-diode OR, precharge, 52.5 V clamp + resistors, fuses (EST)")
    m.add_geom(chassis, name="pilz_relay_contactor_watchdog_radio_rx", type="box", pos=[0, 0, 0],
               size="0.03 0.022 0.018", material="power", mass="0.42")
    note("pilz_relay_contactor_watchdog_radio_rx", "Pilz PNOZ s5, Altran contactor, watchdog board, radio-stop receiver (EST)")
    g, _ = m.geom("regulated_logic_rails")
    g.set("name", "dc_dc_rails_and_pack_monitor")
    g.set("mass", "0.13")
    note(g.get("name"), "2 x 12 V rails, 24 V safety rail, pack monitor (EST)")
    for s_ in ("left", "right"):
        g, _ = m.geom(f"{s_}_can_and_power_loom")
        g.set("mass", f"{0.15 * S ** 2:.4f}")
        note(g.get("name"), "harness, PR #669 x 0.8^2 (EST)")
    g, _ = m.geom("cooling_and_fan")
    g.set("name", "drive_bay_blowers_and_ducts")
    g.set("mass", "0.10")
    note(g.get("name"), "blowers, ducts, filters (EST)")
    m.add_geom(chassis, name="audio_and_servo_bus_adapter", type="box", pos=[0, 0, 0], size="0.01 0.01 0.01",
               material="compute", mass="0.10")
    note("audio_and_servo_bus_adapter", "speaker, amplifier, microphone, U2D2 bus adapter, eyelid 5 V rail (EST)")
    # bought parts back to catalog size (PR #669 keeps installed sizes of bought parts between sizes), laid out so no
    # two boxes intersect (checked by check_layout60.py)
    for n, (half, pos) in ELECTRONICS.items():
        g, _ = m.geom(n)
        g.set("size", bm.fmt(half))
        g.set("pos", bm.fmt(pos))
    for n in ("oak_d_lite", "left_vl53l8cx", "right_vl53l8cx", "trunk_imu", "drive_bay_blowers_and_ducts"):
        scale_attr(m.geom(n)[0], "size", 1 / S)
    g, _ = m.geom("drive_bay_blowers_and_ducts")
    p = bm.parse_vec(g.get("pos"))
    g.set("pos", bm.fmt([p[0], p[1], 0.0072]))   # above the IMU at its catalog size

    # ---- closed variant: 10 % reserve ----
    def all_mass():
        return [(g, b) for b in r.iter("body") for g in b.findall("geom") if g.get("mass") and float(g.get("mass")) > 0]
    listed = sum(float(g.get("mass")) for g, _ in all_mass())
    if variant == "closed":
        reserve = 0.10 * listed
        m.add_geom(chassis, name="reserve_small_parts_est", type="box", pos=[0.0, -0.055, -0.016],
                   size="0.032 0.01 0.008", material="harness", mass=f"{0.15:.3f}")
        note("reserve_small_parts_est", "reserve: connectors, brackets, fasteners added at build (EST)")
        rest = reserve - 0.15
        growth = [g for g, _ in all_mass() if "(EST)" in ledger_notes.get(g.get("name"), "")
                  and g.get("name") != "reserve_small_parts_est"]
        gsum = sum(float(g.get("mass")) for g in growth)
        factor = (gsum + rest) / gsum
        for g in growth:
            g.set("mass", f"{float(g.get('mass')) * factor:.6f}")
            note(g.get("name"), ledger_notes[g.get("name")] + f"; x {factor:.4f} reserve growth (closed variant)")

    # ---- actuators: joint-side limits from the drives; T. rex nominal gains x ratio^2 ----
    act = r.find("actuator")
    for a in list(act):
        act.remove(a)
    drive_table = {}
    for c in CORNERS:
        end = "fore" if "fore" in c else "hind"
        for axis in ("roll", "pitch", "flexion"):
            dname, ratio = AXIS_DRIVE[(axis, end)] if axis == "flexion" else AXIS_DRIVE[axis]
            D = DRIVES[dname]
            eta = ETA if axis == "flexion" else 1.0
            limit = 0.70 * D["peak_nm"] * ratio * eta
            kp, kv = bm.DRIVE_GAINS[axis]
            j = next(jj for jj in r.iter("joint") if jj.get("name") == f"{c}_{axis}")
            lo, hi = bm.parse_vec(j.get("range"))
            j.set("armature", f"{D['armature_kgm2'] * ratio ** 2:.6f}")
            ET.SubElement(act, "position", {"name": f"{c}_{axis}", "joint": f"{c}_{axis}", "kp": f"{kp * ratio ** 2:.4g}",
                                            "kv": f"{kv * ratio ** 2:.4g}",
                                            "ctrlrange": f"{math.radians(lo):.6f} {math.radians(hi):.6f}",
                                            "forcerange": f"{-limit:.4f} {limit:.4f}"})
            drive_table[f"{c}_{axis}"] = {"drive": dname, "ratio": ratio, "eta": eta, "joint_peak_limit_nm": round(limit, 4),
                                          "joint_thermal_eq_nm": round(D["thermal_eq_nm"] * ratio * eta, 4)}
    for name, joint, sv, kp, kv in (("neck_pitch", "neck_pitch", "XC430-W240-T", 8, 0.3),
                                    ("neck_yaw", "neck_yaw", "XC430-W240-T", 6, 0.2),
                                    ("tail_yaw", "tail_yaw", "XL430-W250-T", 5, 0.15),
                                    ("tail_pitch", "tail_pitch", "XL430-W250-T", 8, 0.2)):
        lim = SERVO60[sv]["limit"]
        j = next(jj for jj in r.iter("joint") if jj.get("name") == joint)
        lo, hi = bm.parse_vec(j.get("range"))
        ET.SubElement(act, "position", {"name": name, "joint": joint, "kp": str(kp), "kv": str(kv),
                                        "ctrlrange": f"{math.radians(lo):.6f} {math.radians(hi):.6f}",
                                        "forcerange": f"{-lim:.4f} {lim:.4f}"})
        drive_table[name] = {"drive": sv, "joint_peak_limit_nm": round(lim, 4)}
    sens = r.find("sensor")
    for s_ in list(sens):
        if s_.tag == "actuatorfrc":
            sens.remove(s_)
    for a in act:
        ET.SubElement(sens, "actuatorfrc", {"name": f"{a.get('name')}_effort", "actuator": a.get("name")})
    ET.SubElement(sens, "jointpos", {"name": "neck_yaw_angle", "joint": "neck_yaw"})
    ET.SubElement(sens, "jointpos", {"name": "tail_pitch_angle", "joint": "tail_pitch"})
    # a second, identical keyframe named "home" for the training environment (BaseDinoEnv looks it up by name);
    # "neutral" stays first, so the standing and crawl scripts (keyframe 0) are unchanged
    ET.SubElement(r.find("keyframe"), "key", {"name": "home"})
    return m, drive_table, ledger_notes


CATEGORY60 = [("drives", ("_stator",)), ("leg transmissions", ("drive_crank", "pushrod", "output_crank")),
              ("frame and feet", ("6061_", "crossmember", "service_tray", "tray_stay", "neck_cradle", "shoulder_yoke",
                                  "upper_strut", "lower_strut", "elbow_axle", "ankle_hub", "_sole")),
              ("shells", ("PA11_rib", "shell_seam", "limb_cover")),
              ("head", ("neck_pitch_case", "neck_hub", "frill", "skull", "horn", "_eye", "neck_yaw", "beak", "eyelid",
                        "frill_lights")),
              ("tail", ("tail_",)), ("compute and sensors", ("jetson", "nvme", "teensy", "oak_d", "vl53", "trunk_imu", "audio_",
                                                             "io_and_sensor")),
              ("battery", ("gtech",)), ("power, safety, wiring, cooling", ("power_board", "pilz", "dc_dc", "loom",
                                                                            "blowers")),
              ("reserve", ("reserve_",))]


def category_of(name):
    for cat, keys in CATEGORY60:
        if any(k in name for k in keys):
            return cat
    raise KeyError(name)


def ledger(text, variant, drive_table, neck, notes):
    import mujoco
    model = mujoco.MjModel.from_xml_string(text)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = ET.fromstring(text)
    rows, cats = [], {}
    for b in root.iter("body"):
        for g in b.findall("geom"):
            if g.get("mass") and float(g.get("mass")) > 0:
                n = g.get("name")
                cat = category_of(n)
                kg = float(g.get("mass"))
                cats[cat] = cats.get(cat, 0.0) + kg
                rows.append({"geom": n, "body": b.get("name"), "category": cat, "mass_kg": round(kg, 5),
                             "basis": notes.get(n, "PR #669 part, unchanged mass"),
                             "world_pos_m": [round(float(x), 4) for x in data.geom_xpos[model.geom(n).id]]})
    com = data.subtree_com[1]
    hip = 0.325 * S
    return {"build": "60 cm option C", "variant": variant, "total_mass_kg": round(float(model.body_mass.sum()), 4),
            "category_kg": {k: round(v, 4) for k, v in cats.items()}, "com_world_m": [round(float(x), 4) for x in com],
            "front_share_static_estimate": round((com[0] + hip) / (2 * hip), 4), "neck": neck, "drives": drive_table,
            "parts": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "generated"))
    ap.add_argument("--ankle-roll", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("listed", False), ("closed", False)] + ([("closed", True)] if a.ankle_roll else [])
    for variant, ar in jobs:
        m, dt, notes = build(variant, ar)
        ET.indent(m.tree, space="  ")
        text = ET.tostring(m.root, encoding="unicode")
        ref, neck = bm.set_neck_spring(text)
        nj = next(j for j in m.root.iter("joint") if j.get("name") == "neck_pitch")
        nj.set("stiffness", "0.40")
        nj.set("springref", f"{math.degrees(ref * 0.70 / 0.40):.3f}")
        neck["spring_stiffness_nm_per_rad"] = 0.40
        neck["springref_deg"] = round(math.degrees(ref * 0.70 / 0.40), 3)
        neck.pop("spring_before_nm", None)   # it described PR #669's spring, not this one
        neck["spring_at_neutral_nm"] = round(0.40 * ref * 0.70 / 0.40, 4)
        neck["spring_basis"] = ("0.40 N m/rad (EST): soft enough that over the +-20 deg neck range the spring's torque "
                                "changes by 0.14 N m, under a quarter of the XC430's 0.63 N m continuous estimate")
        text = ET.tostring(m.root, encoding="unicode") + "\n"
        name = f"trike60_C_{variant}" + ("_ankleroll" if ar else "")
        (out / f"{name}.xml").write_text(text)
        led = ledger(text, variant, dt, neck, notes)
        (out / f"mass_ledger_{name}.json").write_text(json.dumps(led, indent=1) + "\n")
        print(f"{name}.xml: {led['total_mass_kg']:.3f} kg; CoM x {1000 * led['com_world_m'][0]:+.1f} mm, "
              f"y {1000 * led['com_world_m'][1]:+.1f} mm; front share (lever estimate) "
              f"{100 * led['front_share_static_estimate']:.1f} %")
        print("   categories:", led["category_kg"])


if __name__ == "__main__":
    main()
