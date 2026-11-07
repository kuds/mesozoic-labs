"""Generate three installed T. rex plants from the M1 dimensions and reservations.

The zero pose has both soles tangent to the floor. All links and component
centers are in their parent frames. Device dimensions never follow body scale.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from environments.shared.robot_building import element, values
from examples.tyrannosaurus_rex_60cm.build import DESIGN as M1_DESIGN
from examples.tyrannosaurus_rex_60cm.build import Skeleton

HERE = Path(__file__).resolve().parent
DESIGN = HERE / "design.json"


def named(parent: ET.Element, tag: str, name: str) -> ET.Element:
    node = parent.find(f".//{tag}[@name='{name}']")
    if node is None:
        raise ValueError(f"missing {tag} {name}")
    return node


def translated(node: ET.Element, delta: np.ndarray) -> None:
    node.set("pos", values(np.fromstring(node.attrib.get("pos", "0 0 0"), sep=" ") + delta))


def cover(builder: Skeleton, body: ET.Element, name: str, stations: list[tuple], material: str) -> None:
    """Closed eight-sided sections with explicit vertices in the body frame."""
    vertices = []
    faces = []
    for i, (x, width, zlow, zhigh) in enumerate(stations):
        z = (zhigh + zlow) / 2
        h = zhigh - z
        for yunit, zunit in (
            (0.68, 1),
            (-0.68, 1),
            (-1, 0.4),
            (-1, -0.65),
            (-0.65, -1),
            (0.65, -1),
            (1, -0.65),
            (1, 0.4),
        ):
            vertices.append((x, width * yunit, z + zunit * h))
        if i:
            for j in range(8):
                a, b = 8 * (i - 1) + j, 8 * (i - 1) + (j + 1) % 8
                faces.extend(((a, b, a + 8), (b, b + 8, a + 8)))
    last = 8 * (len(stations) - 1)
    for i in range(1, 7):
        faces.extend(((0, i + 1, i), (last, last + i, last + i + 1)))
    element(builder.asset, "mesh", name=name, vertex=np.ravel(vertices), face=np.ravel(faces))
    element(body, "geom", name=name, type="mesh", mesh=name, material=material, group=1, contype=1, conaffinity=1)


def install_legs(b: Skeleton, p: dict, roll_limits: dict) -> None:
    for side, sign in (("left", 1), ("right", -1)):
        b.leg(side, sign)
        for kind, limit in roll_limits.items():
            name = side + "_" + kind
            named(b.root, "joint", name).set("range", values((-limit, limit)))
            next(j for j in b.joints if j["name"] == name)["range"] = [-limit, limit]
        offset = np.array([0, sign * (p["hip_yaw_offset"] - 0.045), 0])
        for node in b.root:
            if node.get("name", "").startswith(side + "_hip_yaw") or node.get("name") == side + "_yaw":
                translated(node, offset)
        width, length = p["sole_width"] / 2, p["sole_length"] / 2
        for name, kind, size in (
            ("sole", "geom", (length, width, 0.006)),
            ("plate", "geom", (length - 0.006, width - 0.006, 0.002)),
            ("touch", "site", (length + 0.001, width + 0.001, 0.007)),
        ):
            named(b.root, kind, side + "_" + name).set("size", values(size))
        for c in b.components:
            if c["name"] == side + "_hip_yaw_motor":
                c["pos_m"] = (np.array(c["pos_m"]) + offset).tolist()
            if c["body"] in (side + "_thigh", side + "_shin") and c["category"] == "frames_bearings_feet":
                vec = b.design["thigh_vector_m"] if c["body"].endswith("thigh") else b.design["shin_vector_m"]
                c["mass_kg"] *= np.linalg.norm(vec) / math.hypot(0.06, 0.14)
                c["inertia_box_halfsize_m"][2] *= abs(vec[2]) / 0.14
            if c["name"] == side + "_foot_structure":
                c["mass_kg"] *= width * length / (0.044 * 0.080)
                c["inertia_box_halfsize_m"][:2] = [length, width]


def install_body(b: Skeleton, p: dict) -> None:
    b.torso()
    # Replace the old fixed-payload proxies and wire outlines. The M1 frame,
    # board trays, load paths through hip yokes, and their mass survive.
    b.components[:] = [c for c in b.components if c["category"] != "future_fixed_payloads"]
    for node in list(b.root):
        name = node.get("name", "")
        if name.startswith("outline_") or name in ("fixed_tail_spar", "future_head_crown"):
            b.root.remove(node)
    width = p["rib_half_width"]
    back = p["back_rise"]
    aft = p["battery_x"] - 0.090
    for sign in (-1, 1):
        # Removable side covers on raised ribs; leave underside and fan outlet open.
        b.rod(b.root, f"rib_mount_{sign}", (-0.08, sign * 0.075, 0.13), (-0.08, sign * width, back), 0.004)
        b.rod(
            b.root,
            f"side_panel_{sign}",
            (aft + 0.045, sign * width, back - 0.032),
            (0.134, sign * width, back - 0.05),
            0.015,
            material="skin",
        )
        b.rod(
            b.root,
            f"fixed_arm_upper_{sign}",
            (0.140, sign * width, 0.070),
            (0.185, sign * (width + 0.027), 0.027),
            0.012,
            material="skin",
        )
        b.rod(
            b.root,
            f"fixed_arm_fore_{sign}",
            (0.185, sign * (width + 0.027), 0.027),
            (0.221, sign * (width + 0.020), 0.043),
            0.008,
            material="skin",
        )
        b.component(
            b.root, f"arm_{sign}", "fixed_arms", 25, (0.178, sign * (width + 0.014), 0.048), (0.045, 0.018, 0.025)
        )
        b.component(
            b.root,
            f"torso_side_panel_{sign}",
            "covers_brackets",
            90,
            (aft / 2 + 0.052, sign * width, back - 0.04),
            ((0.134 - aft) / 2, 0.012, 0.035),
        )
    # Spine ends and battery tray follow actual cassette location.
    for n in b.root.findall("geom"):
        name = n.get("name", "")
        if name.startswith(("rear_battery", "spine_", "tail_root")) and "fromto" in n.attrib:
            ends = np.fromstring(n.get("fromto"), sep=" ").reshape(2, 3)
            ends[:, 0] += (ends[:, 0] < -0.15) * (p["battery_x"] + 0.17)
            n.set("fromto", values(ends.ravel()))
    # Catalogue bodies within the retained installed power reservations.
    named(b.root, "geom", "compute_converter").set("size", values(np.array([0.0254, 0.0254, 0.00953]) / 2))
    named(b.root, "geom", "logic_converter").set("size", values(np.array([0.013, 0.010, 0.006]) / 2))
    for label, pos, size, mass in [
        ("regulated_12v_ttl", (-0.155, 0.002, 0.105), (0.065, 0.051, 0.032), 0.085),
        ("drive_disconnect", (-0.225, -0.060, 0.065), (0.055, 0.026, 0.036), 0.055),
    ]:
        b.geom(b.root, label + "_reserve", "box", np.array(size) / 2, pos, "reserve", group=5)
        b.reserves.append(label + "_reserve")
        halfsize = np.array([0.0432, 0.0318, 0.007]) / 2 if label == "regulated_12v_ttl" else np.array(size) * 0.39
        b.geom(b.root, label + "_case", "box", halfsize, pos, "pcb", group=4)
        b.component(b.root, label, "power_auxiliaries", 1000 * mass, pos, np.array(size) / 2)


def install_appendages(b: Skeleton, p: dict) -> None:
    for label, body_name, joint_name, pivot, axis, rng in [
        ("neck", "head", "neck_pitch", np.array([0.224, 0, 0.174]), (0, -1, 0), (-0.25, 0.28)),
        ("tail", "tail", "tail_yaw", np.array([p["battery_x"] - 0.113, 0, 0.10]), (0, 0, 1), (-0.38, 0.38)),
    ]:
        body = b.body(b.root, body_name, pivot)
        b.add_joint(body, joint_name, axis, rng, 0.88, 11, 0.38, damping=0.018, frictionloss=0.009, armature=0.00014)
        # Two bearings on the fixed fork carry side load. The motor coupling
        # transmits torque only; servo case has its full supplier dimensions.
        along = np.array([0, 1, 0]) if label == "neck" else np.array([0, 0, 1])
        separation = 0.043 if label == "neck" else 0.030
        center = pivot - (separation + 0.025) * along + np.array([-0.012, 0, 0])
        half = np.array([0.0465, 0.034, 0.0285]) / 2 if label == "neck" else np.array([0.0465, 0.0285, 0.034]) / 2
        b.geom(b.root, label + "_motor_case", "box", half, center, "motor", group=2, contype=1, conaffinity=1)
        b.component(b.root, label + "_xm430", "appendage_motors", 82, center, half)
        plug = label + "_motor_cable_reserve"
        b.geom(b.root, plug, "box", (0.012, 0.010, 0.009), center + (-0.034, 0, 0), "reserve", group=5)
        b.reserves.append(plug)
        b.plugs[plug] = label + "_motor_case"
        b.motor_names.append(label + "_motor_case")
        for sign in (-1, 1):
            pos = pivot + sign * separation * along
            euler = (math.pi / 2, 0, 0) if label == "neck" else (0, 0, 0)
            b.geom(b.root, f"{label}_bearing_{sign}", "cylinder", (0.012, 0.004), pos, "metal", group=2, euler=euler)
            anchor = (0.18, sign * 0.067, 0.13) if label == "neck" else (pivot[0] + 0.05, sign * 0.055, 0.125)
            b.rod(b.root, f"{label}_fork_{sign}", anchor, pos, 0.006)
        b.rod(body, label + "_output_shaft", -separation * along, separation * along, 0.004, material="metal")
        b.component(b.root, label + "_fork_bearings_encoder", "appendage_mounts", 80, pivot, (0.035, 0.045, 0.034))
        b.component(body, label + "_shaft_hub", "appendage_mounts", 25, (0, 0, 0), (0.02, 0.025, 0.014))
        if label == "neck":
            length = p["snout"] - pivot[0]
            crown = 0.600 - p["hip_height"] - pivot[2]
            cover(
                b,
                body,
                "cranium",
                [
                    (-0.021, 0.038, -0.019, 0.024),
                    (0.025, 0.057, -0.013, crown),
                    (0.48 * length, 0.060, -0.004, crown - 0.004),
                    (0.85 * length, 0.049, -0.005, 0.029),
                    (length, 0.036, 0.000, 0.017),
                ],
                "skin",
            )
            cover(
                b,
                body,
                "fixed_jaw",
                [
                    (0.01, 0.039, -0.038, -0.020),
                    (0.40 * length, 0.05, -0.044, -0.017),
                    (0.96 * length, 0.033, -0.019, -0.012),
                ],
                "jaw",
            )
            for sign in (-1, 1):
                b.geom(
                    body,
                    f"eye_{sign}",
                    "ellipsoid",
                    (0.012, 0.004, 0.009),
                    (0.046, sign * 0.057, 0.029),
                    "motor",
                    group=2,
                )
            b.component(
                body,
                "skull_and_jaw_cover",
                "moving_head",
                270,
                (0.39 * length, 0, 0.012),
                (length * 0.47, 0.057, 0.045),
            )
            b.component(body, "skull_mount", "moving_head", 120, (0.015, 0, 0.0), (0.031, 0.04, 0.029))
            element(body, "site", name="head_crown", pos=(0.025, 0, crown), size=0.001, rgba=(0, 0, 0, 0))
            element(body, "site", name="snout", pos=(length, 0, 0.010), size=0.001, rgba=(0, 0, 0, 0))
        else:
            length = pivot[0] - p["tail_tip"]
            cover(
                b,
                body,
                "tail_shroud",
                [
                    (-length, 0.005, -0.065, -0.055),
                    (-0.62 * length, 0.018, -0.053, -0.024),
                    (-0.27 * length, 0.031, -0.035, 0.019),
                    (-0.014, 0.038, -0.032, 0.036),
                ],
                "skin",
            )
            b.rod(body, "tail_spar", (-0.010, 0, 0), (-length, 0, -0.061), 0.004)
            b.component(
                body,
                "tail_spar_shell",
                "moving_tail",
                210 * length / 0.292,
                (-0.36 * length, 0, -0.015),
                (0.42 * length, 0.032, 0.025),
            )
            element(body, "site", name="tail_tip", pos=(-length, 0, -0.060), size=0.001, rgba=(0, 0, 0, 0))


def sensors(b: Skeleton, design: dict) -> None:
    cam = design["camera"]
    pitch = cam["pitch_down_rad"]
    c, s = math.cos(pitch), math.sin(pitch)
    lens = np.array(cam["position_m"])
    case = lens - np.array([c, 0, -s]) * (cam["device_size_m"][0] / 2 + 0.001)
    element(b.root, "camera", name="front_rgbd", pos=lens, xyaxes=(0, -1, 0, s, 0, c), fovy=cam["vertical_fov_deg"])
    b.geom(b.root, "oak_d_lite", "box", np.array(cam["device_size_m"]) / 2, case, "motor", euler=(0, pitch, 0), group=2)
    b.geom(b.root, "rgbd_reserve", "box", np.array(cam["reserve_size_m"]) / 2, case, "reserve", group=5)
    b.reserves.append("rgbd_reserve")
    b.component(
        b.root,
        "oak_d_lite_bracket_cable",
        "sensors",
        cam["installed_kg"] * 1000,
        case,
        np.array(cam["reserve_size_m"]) / 2,
    )
    for side, sign in (("left", 1), ("right", -1)):
        b.rod(b.root, "camera_brace_" + side, (0.160, sign * 0.046, 0.078), case + (-0.010, sign * 0.043, 0), 0.003)
        tof = design["tof"]
        mount = np.array(tof["position_m"]) * [1, sign, 1]
        tilt, yaw = np.deg2rad([tof["pitch_down_deg"], sign * tof["yaw_out_deg"]])
        direction = np.array([math.cos(tilt) * math.cos(yaw), math.cos(tilt) * math.sin(yaw), -math.sin(tilt)])
        b.rod(b.root, "sensor_strut_" + side, (0.155, sign * 0.062, 0.085), mount, 0.004)
        b.geom(
            b.root, "tof_" + side + "_reserve", "box", np.array(tof["reserve_size_m"]) / 2, mount, "reserve", group=5
        )
        b.reserves.append("tof_" + side + "_reserve")
        b.geom(b.root, "tof_" + side + "_carrier", "box", (0.004, 0.009, 0.007), mount, "pcb", group=2)
        b.component(
            b.root,
            "tof_" + side + "_strut_cable",
            "sensors",
            tof["installed_kg_each"] * 1000,
            mount,
            np.array(tof["reserve_size_m"]) / 2,
        )
        element(
            b.root,
            "site",
            name="tof_" + side,
            pos=mount + direction * 0.006,
            zaxis=direction,
            size=0.001,
            rgba=(0, 0, 0, 0),
        )


def build(variant: str = "nominal", destination: Path | None = None) -> Path:
    d = json.loads(DESIGN.read_text())
    p = d["variants"][variant]
    m1 = json.loads(M1_DESIGN.read_text())
    m1["hip_height_m"] = p["hip_height"]
    leg_drop = (p["hip_height"] - 0.08) / 2
    m1["thigh_vector_m"] = [p["knee_advance"], 0, -leg_drop]
    m1["shin_vector_m"] = [-p["knee_advance"], 0, -leg_drop]
    next(a for a in m1["assemblies"] if a["name"] == "battery")["position_m"][0] = p["battery_x"]
    b = Skeleton(m1)
    b.xml.set(
        "model",
        "tyrannosaurus_rex_intermediate" if variant == "nominal" else f"tyrannosaurus_rex_intermediate_{variant}",
    )
    element(b.asset, "material", name="skin", rgba=(0.30, 0.43, 0.24, 1))
    element(b.asset, "material", name="jaw", rgba=(0.54, 0.52, 0.34, 1))
    limits = {}
    if variant == "nominal":
        limits = {kind: d["drives"][kind + "_limit_nominal_rad"] for kind in ("hip_roll", "ankle_roll")}
    install_legs(b, p, limits)
    install_body(b, p)
    install_appendages(b, p)
    sensors(b, d)
    element(b.world, "body", name="target", mocap="true", pos=(0, 0, 0))
    drives = d["drives"]
    custom = element(b.xml, "custom")
    params = [drives.get(j["name"], drives["leg"]) for j in b.joints]
    for key in ("peak_nm", "continuous_nm", "no_load_rad_s", "deadband_rad"):
        element(custom, "numeric", name="drive_" + key, data=[item[key] for item in params])
    element(
        custom,
        "numeric",
        name="drive_traction",
        data=[int(j["name"] not in ("neck_pitch", "tail_yaw")) for j in b.joints],
    )
    element(
        custom,
        "numeric",
        name="drive_timing",
        data=[drives[k] for k in ("reference_bus_v", "delay_s", "effort_window_s")],
    )
    for j, item in zip(b.joints, params, strict=True):
        j["torque_limit_nm"] = item["peak_nm"]
    out = destination or HERE / variant
    out.mkdir(parents=True, exist_ok=True)
    b.write(
        out,
        imu_site="pelvis_imu",
        contact_sites=[(s + "_contact", s + "_touch") for s in ("left", "right")],
        limitations=[
            "Unweighed installation and box-inertia estimates; tolerances and strength unqualified.",
            "Use InstalledDrive on every substep to apply torque-speed, bus voltage, command play and drive effort limits.",
            "Only cases, soles and rigid covers have physics contact. Installation reservations are sampled by the independent clearance screen.",
        ],
        metadata={
            "hardware_revision": d["revision"],
            "proportion_variant": variant,
            "motor_cases": b.motor_names,
            "installation_reserves": b.reserves,
            "connector_owners": b.plugs,
        },
    )
    b.xml.remove(b.xml.find("equality"))
    sens = b.xml.find("sensor")
    for side in ("left", "right"):
        element(sens, "rangefinder", name="tof_" + side + "_range", site="tof_" + side)
    ET.indent(b.xml, space="  ")
    (out / "model.xml").write_text(ET.tostring(b.xml, encoding="unicode") + "\n")
    if destination is None and variant == "nominal":
        assets = HERE.parents[2] / "environments/tyrannosaurus_rex/intermediate/assets"
        assets.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "model.xml", assets / "tyrannosaurus_rex_intermediate.xml")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("compact", "nominal", "wide", "all"), default="all")
    args = parser.parse_args()
    for variant in ("compact", "nominal", "wide") if args.variant == "all" else (args.variant,):
        print(build(variant))
