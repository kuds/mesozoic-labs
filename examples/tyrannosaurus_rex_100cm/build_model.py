"""Generate a one-metre character with explicit, unweighed hardware allocations.

This is packaging and rigid-body simulation geometry, not fabrication CAD.
Only the shared mesh/ledger builder is reused; hardware is independently sized.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from examples.tyrannosaurus_rex_50cm.build_model import Builder as BaseBuilder
from examples.tyrannosaurus_rex_50cm.build_model import element, make_meshes, mesh_file, values

HERE = Path(__file__).resolve().parent
ROOT_HEIGHT = 0.460
LEG_TORQUE_NM = 20.0
MOTOR_MASS_G = 900.0
MOTOR_RADIUS_M = 0.053
MOTOR_HALF_DEPTH_M = 0.028


def make_head(assets: Path) -> None:
    """Keep silhouette C: a single rounded, tapered skull with side eyes."""
    vertices, faces = [], []
    for line in (assets / "rounded.obj").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "v":
            x, y, z = map(float, parts[1:])
            vertices.append((x, y * (0.96 - 0.10 * x), z - 0.12 * (x + 1)))
        elif parts and parts[0] == "f":
            faces.append(tuple(int(i) - 1 for i in parts[1:]))
    points = np.asarray(vertices)
    # Preserve exactly 1.000 m skull top after tapering the generated mesh.
    points[:, 2] -= (points[:, 2].max() + points[:, 2].min()) / 2
    points[:, 2] /= np.abs(points[:, 2]).max()
    mesh_file(assets / "skull_c.obj", points.tolist(), faces)


class Builder(BaseBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.xml.set("model", "tyrannosaurus_rex_robot_100cm_concept")
        self.torso.set("pos", values((0, 0, ROOT_HEIGHT)))
        statistic = self.xml.find("statistic")
        assert statistic is not None
        statistic.set("center", "-0.10 0 0.50")
        statistic.set("extent", "1.50")
        for camera in self.world.findall("camera"):
            position = np.fromstring(camera.get("pos", ""), sep=" ") * 2
            camera.set("pos", values(position))
        self.motor_specs: dict[str, Any] = {
            "candidate": "RobStride 03",
            "count": 12,
            "mass_kg_each": 0.900,
            "body_envelope_m": [0.106, 0.106, 0.056],
            "vendor_rated_torque_nm": 21,
            "vendor_peak_torque_nm": 60,
            "simulation_cap_nm": LEG_TORQUE_NM,
            "source": "https://github.com/RobStride/Product_Information",
            "caveat": "Cylindrical packaging surrogate, not supplier mounting CAD; torque rating is conditional on cooling.",
        }

    def joint(self, body: ET.Element, name: str, axis: Any, limits: Any, cap: float, kp: float, kv: float) -> None:
        super().joint(body, name, axis, limits, cap, kp, kv)
        joint = body.find("joint")
        assert joint is not None
        joint.set("damping", "0.15" if cap >= 10 else "0.015")
        joint.set("frictionloss", "0.025" if cap >= 10 else "0.002")
        joint.set("armature", "0.002" if cap >= 10 else "0.0001")

    def motor(self, parent: ET.Element, name: str, pos: Any, axis: str) -> None:
        rotation = {"x": (0, math.pi / 2, 0), "y": (math.pi / 2, 0, 0), "z": (0, 0, 0)}[axis]
        size = (MOTOR_RADIUS_M, MOTOR_HALF_DEPTH_M)
        self.geom(parent, name + "_case", "cylinder", size, pos, "graphite", euler=rotation, group=2)
        self.collision(parent, name + "_case", "cylinder", size, pos, euler=rotation)
        dims = [MOTOR_RADIUS_M] * 3
        dims["xyz".index(axis)] = MOTOR_HALF_DEPTH_M
        self.component(parent, name + "_motor", "leg_actuators", MOTOR_MASS_G, pos, dims)
        face = np.array(pos, dtype=float)
        sign = -1 if axis == "y" and pos[1] < 0 else 1
        face["xyz".index(axis)] += sign * (MOTOR_HALF_DEPTH_M + 0.003)
        self.geom(parent, name + "_hub", "cylinder", (0.032, 0.003), face, "metal", euler=rotation, group=2)

    def expression_motor(self, parent: ET.Element, name: str, pos: Any, heavy: bool = False) -> None:
        grams = 310 if heavy else 191
        size = (0.0285, 0.0285, 0.0255) if heavy else (0.023, 0.023, 0.022)
        self.component(parent, name + "_motor", "expression_actuators", grams, pos, size)
        self.geom(parent, name + "_case", "box", size, pos, "graphite", group=2)
        self.collision(parent, name + "_case", "box", size, pos)

    def link(self, body: ET.Element, name: str, end: Any, radius: float) -> None:
        for collision in (False, True):
            attrs: dict[str, Any] = dict(
                name=name + ("_collision" if collision else ""), type="capsule", fromto=(0, 0, 0, *end), size=radius
            )
            if collision:
                attrs["class_"] = "collision"
            else:
                attrs.update(material="metal", group=2)
            element(body, "geom", **attrs)

    def leg(self, side: str, sign: int) -> None:
        self.motor(self.torso, side + "_hip_yaw", (0, sign * 0.115, 0.090), "z")
        yaw = self.body(self.torso, side + "_hip_yaw_body", (0, sign * 0.115, 0.090))
        self.joint(yaw, side + "_hip_yaw", (0, 0, 1), (-0.30, 0.30), LEG_TORQUE_NM, 350, 15)
        self.motor(yaw, side + "_hip_roll", (-0.045, 0, -0.090), "x")
        roll = self.body(yaw, side + "_hip_roll_body", (0, 0, -0.090))
        self.joint(roll, side + "_hip_roll", (1, 0, 0), (-0.30, 0.30), LEG_TORQUE_NM, 500, 20)
        self.motor(roll, side + "_hip_pitch", (0.055, sign * 0.050, -0.015), "y")
        thigh = self.body(roll, side + "_thigh", (0.055, sign * 0.020, -0.015))
        self.joint(thigh, side + "_hip_pitch", (0, 1, 0), (-0.65, 0.65), LEG_TORQUE_NM, 650, 25)
        knee_pos = (0.030, 0, -0.180)
        self.motor(thigh, side + "_knee", (0.030, sign * 0.030, -0.180), "y")
        shin = self.body(thigh, side + "_shin", knee_pos)
        self.joint(shin, side + "_knee", (0, 1, 0), (-0.15, 1.10), LEG_TORQUE_NM, 650, 25)
        self.motor(shin, side + "_ankle_pitch", (-0.045, sign * 0.030, -0.180), "y")
        ankle = self.body(shin, side + "_ankle_pitch_body", (-0.045, 0, -0.180))
        self.joint(ankle, side + "_ankle_pitch", (0, 1, 0), (-0.55, 0.55), LEG_TORQUE_NM, 650, 25)
        self.motor(ankle, side + "_ankle_roll", (-0.090, 0, -0.020), "x")
        foot = self.body(ankle, side + "_foot", (-0.060, 0, -0.020))
        self.joint(foot, side + "_ankle_roll", (1, 0, 0), (-0.25, 0.25), LEG_TORQUE_NM, 550, 20)
        for body, grams, pos, half in (
            (yaw, 140, (0, 0, -0.04), (0.035, 0.030, 0.04)),
            (roll, 160, (0.025, sign * 0.010, -0.008), (0.045, 0.035, 0.020)),
            (thigh, 240, (0.015, 0, -0.09), (0.035, 0.025, 0.095)),
            (shin, 200, (-0.0225, 0, -0.09), (0.028, 0.023, 0.095)),
            (ankle, 100, (-0.045, 0, -0.020), (0.060, 0.030, 0.025)),
            (foot, 300, (0.060, 0, -0.040), (0.140, 0.070, 0.020)),
        ):
            self.component(body, body.get("name", "") + "_structure", "structure_bearings_feet", grams, pos, half)
        self.component(
            thigh, side + "_thigh_cover", "shells_guards", 100, (0.015, -sign * 0.015, -0.080), (0.065, 0.045, 0.085)
        )
        self.component(
            shin, side + "_shin_cover", "shells_guards", 70, (-0.020, -sign * 0.015, -0.075), (0.050, 0.040, 0.07)
        )
        self.component(shin, side + "_leg_wiring", "wiring_power", 100, (-0.020, 0, -0.08), (0.008, 0.008, 0.09))
        self.component(foot, side + "_foot_sensors", "sensors", 80, (0.060, 0, -0.045), (0.120, 0.055, 0.008))
        self.link(thigh, side + "_thigh_link", knee_pos, 0.025)
        self.link(shin, side + "_shin_link", (-0.045, 0, -0.180), 0.020)
        self.mesh(thigh, side + "_thigh_fairing", (0.065, 0.045, 0.080), (0.015, -sign * 0.012, -0.08))
        self.mesh(shin, side + "_shin_fairing", (0.047, 0.035, 0.068), (-0.020, -sign * 0.012, -0.075))
        self.mesh(
            foot, side + "_sole_visual", (0.140, 0.070, 0.006), (0.060, 0, -0.059), "rubber", source="rounded", group=2
        )
        self.collision(foot, side + "_sole", "box", (0.140, 0.070, 0.006), (0.060, 0, -0.059))
        self.geom(foot, side + "_foot_plate", "box", (0.090, 0.063, 0.004), (0.100, 0, -0.051), "metal", group=2)
        self.mesh(foot, side + "_heel_cover", (0.080, 0.064, 0.020), (0.020, 0, -0.025))
        for i, y in enumerate((-0.047, 0, 0.047)):
            self.mesh(foot, f"{side}_toe_{i}", (0.065, 0.025, 0.028), (0.130, y, -0.030))
            self.geom(foot, f"{side}_claw_{i}", "ellipsoid", (0.020, 0.020, 0.016), (0.182, y, -0.038), "cream")
        for i, (x, y) in enumerate(((-0.090, -0.035), (-0.090, 0.035), (0.090, -0.035), (0.090, 0.035))):
            element(
                foot,
                "site",
                name=f"{side}_contact_{i}",
                type="box",
                pos=(0.060 + x, y, -0.059),
                size=(0.050, 0.035, 0.008),
                rgba=(0.2, 0.7, 0.2, 0),
                group=3,
            )
        element(foot, "site", name=side + "_sole_center", pos=(0.060, 0, -0.065), size=0.002, rgba=(0, 0, 0, 0))

    def upper_body(self) -> None:
        t = self.torso
        self.mesh(t, "torso_shell", (0.185, 0.200, 0.145), (-0.035, 0, 0.140))
        self.mesh(t, "belly_panel", (0.035, 0.155, 0.112), (0.130, 0, 0.145), "cream")
        self.mesh(t, "back_access_panel", (0.025, 0.140, 0.110), (-0.205, 0, 0.145), "dark_sage")
        self.collision(t, "upper_torso", "ellipsoid", (0.170, 0.190, 0.095), (-0.035, 0, 0.190))
        self.collision(t, "central_pelvis", "box", (0.090, 0.060, 0.060), (-0.025, 0, 0.070))
        for name, category, grams, pos, size, mat in (
            ("battery", "battery_restraint", 2800, (-0.055, 0, 0.120), (0.090, 0.060, 0.045), "graphite"),
            ("compute_tray", "compute_cooling_storage", 650, (-0.055, 0, 0.250), (0.065, 0.060, 0.025), "pcb"),
            ("control_can_imu", "control", 220, (0.060, 0, 0.225), (0.035, 0.055, 0.020), "pcb"),
            ("power_tray", "wiring_power", 650, (-0.165, 0, 0.160), (0.025, 0.090, 0.055), "metal"),
            ("torso_structure", "structure_bearings_feet", 1200, (-0.035, 0, 0.135), (0.145, 0.145, 0.120), "metal"),
            ("integration_reserve", "unallocated_reserve", 1800, (-0.025, 0, 0.110), (0.100, 0.090, 0.060), "orange"),
        ):
            self.component(t, name, category, grams, pos, size)
            if name not in ("torso_structure", "integration_reserve"):
                self.geom(t, name + "_envelope", "box", size, pos, mat, group=4)
        self.component(t, "torso_shell_mass", "shells_guards", 550, (-0.035, 0, 0.140), (0.185, 0.200, 0.145))
        element(t, "site", name="pelvis_imu", pos=(-0.025, 0, 0.040), size=0.005, rgba=(1, 0.2, 0, 0), group=3)
        for i in range(6):
            self.geom(t, f"back_vent_{i}", "box", (0.002, 0.060, 0.002), (-0.231, 0, 0.115 + i * 0.014), "graphite")
        # Fixed forward camera stays stable as the character turns its head.
        self.component(t, "chest_depth_camera", "sensors", 120, (0.157, 0, 0.205), (0.016, 0.052, 0.015))
        self.geom(t, "chest_camera_bar", "box", (0.017, 0.055, 0.017), (0.168, 0, 0.205), "graphite", group=2)
        for side, y in (("left", 0.0375), ("right", -0.0375)):
            self.geom(
                t,
                "chest_" + side + "_lens",
                "cylinder",
                (0.007, 0.002),
                (0.187, y, 0.205),
                "glass",
                euler=(0, math.pi / 2, 0),
                group=2,
            )
            element(t, "camera", name="chest_" + side, pos=(0.190, y, 0.205), xyaxes=(0, -1, 0, 0, 0, 1), fovy=60)
        for side, sign in (("left", 1), ("right", -1)):
            base = (0.040, sign * 0.215, 0.205)
            self.expression_motor(t, side + "_shoulder", base)
            arm = self.body(t, side + "_arm", (0.040, sign * 0.245, 0.205))
            self.joint(arm, side + "_shoulder_pitch", (0, 1, 0), (-0.70, 0.35), 0.8, 8, 0.4)
            self.component(arm, side + "_arm_shell", "moving_arms", 120, (0.055, 0, -0.040), (0.065, 0.022, 0.045))
            self.mesh(arm, side + "_arm_upper", (0.035, 0.024, 0.055), (0.015, 0, -0.045))
            self.mesh(arm, side + "_hand", (0.045, 0.026, 0.022), (0.065, 0, -0.075), "cream")
            self.collision(arm, side + "_arm", "capsule", (0.022,), fromto=(0, 0, 0, 0.080, 0, -0.075))
        self.expression_motor(t, "head_yaw", (-0.020, 0, 0.305), heavy=True)
        yaw = self.body(t, "head_yaw_body", (-0.020, 0, 0.305))
        self.joint(yaw, "head_yaw", (0, 0, 1), (-0.45, 0.45), 2.0, 25, 1)
        self.component(yaw, "neck_frame", "structure_bearings_feet", 220, (0, 0, 0.030), (0.040, 0.035, 0.035))
        self.expression_motor(yaw, "head_pitch", (0, -0.045, 0.065), heavy=True)
        self.geom(yaw, "neck_column", "cylinder", (0.038, 0.035), (0, 0, 0.030), "graphite", group=2)
        head = self.body(yaw, "head", (0, 0, 0.065))
        self.joint(head, "head_pitch", (0, 1, 0), (-0.16, 0.16), 2.0, 30, 1.2)
        self.mesh(head, "head_shell", (0.220, 0.175, 0.095), (0.080, 0, 0.075), source="skull_c")
        self.collision(head, "head", "box", (0.190, 0.145, 0.070), (0.080, 0, 0.075))
        self.component(head, "upper_head_shell", "head_shell_frame", 650, (0.055, 0, 0.070), (0.220, 0.175, 0.095))
        self.component(head, "head_frame", "head_shell_frame", 180, (0, 0, 0.025), (0.060, 0.055, 0.045))
        self.expression_motor(head, "jaw", (-0.090, 0, -0.020))
        self.component(head, "eyes_audio", "sensors", 90, (0.025, 0, 0.060), (0.070, 0.130, 0.040))
        for side, sign in (("left", 1), ("right", -1)):
            for suffix, eye_size, eye_pos, eye_material in (
                ("socket", (0.059, 0.014, 0.058), (-0.020, sign * 0.163, 0.085), "dark_sage"),
                ("white", (0.052, 0.012, 0.051), (-0.016, sign * 0.173, 0.085), "white"),
                ("pupil", (0.036, 0.009, 0.038), (-0.005, sign * 0.183, 0.080), "black"),
                ("glint", (0.010, 0.004, 0.011), (0.006, sign * 0.191, 0.100), "white"),
                ("nostril", (0.014, 0.005, 0.008), (0.240, sign * 0.145, 0.055), "black"),
            ):
                self.geom(head, side + "_eye_" + suffix, "ellipsoid", eye_size, eye_pos, eye_material)
            self.geom(head, side + "_tooth", "ellipsoid", (0.010, 0.008, 0.018), (0.185, sign * 0.140, -0.010), "cream")
        jaw = self.body(head, "jaw", (-0.090, 0, -0.020))
        self.joint(jaw, "jaw_pitch", (0, 1, 0), (0, 0.24), 0.8, 8, 0.35)
        self.component(jaw, "jaw_shell", "head_shell_frame", 260, (0.175, 0, -0.012), (0.210, 0.160, 0.025))
        self.mesh(jaw, "jaw_chin", (0.085, 0.153, 0.025), (0.300, 0, -0.012), "cream", source="rounded")
        self.collision(jaw, "jaw_chin", "box", (0.075, 0.135, 0.020), (0.300, 0, -0.012))
        for side, sign in (("left", 1), ("right", -1)):
            self.mesh(
                jaw, side + "_jaw_rail", (0.125, 0.020, 0.025), (0.100, sign * 0.145, -0.012), "cream", source="rounded"
            )
            self.collision(jaw, side + "_jaw_rail", "box", (0.115, 0.016, 0.020), (0.100, sign * 0.145, -0.012))
        tail_pos = (-0.225, 0, 0.105)
        self.expression_motor(t, "tail_yaw", tail_pos)
        tail = self.body(t, "tail", tail_pos)
        self.joint(tail, "tail_yaw", (0, 0, 1), (-0.26, 0.26), 0.8, 8, 0.4)
        self.component(tail, "tail_shell_spar", "moving_tail", 350, (-0.145, 0, 0.020), (0.200, 0.045, 0.060))
        self.mesh(tail, "tail_shell", (1.65, 1.8, 1.8), (0, 0, 0), "sage", source="tail")
        element(
            tail,
            "geom",
            name="tail_collision",
            class_="collision",
            type="capsule",
            fromto=(-0.030, 0, 0, -0.330, 0, 0.080),
            size=0.026,
        )

    def finish(self, directory: Path) -> None:
        super().finish(directory)
        key = self.xml.find("keyframe/key")
        assert key is not None
        key.set("qpos", values((0, 0, ROOT_HEIGHT, 1, 0, 0, 0, *([0] * len(self.joints)))))
        ET.indent(self.xml, space="  ")
        (directory / "model.xml").write_text(ET.tostring(self.xml, encoding="unicode") + "\n")
        path = directory / "components.json"
        report = json.loads(path.read_text())
        report["nominal_height_m"] = 1.0
        report["leg_motor_specification"] = self.motor_specs
        report["expression_motor_specification"] = {
            "head": {
                "candidate": "RobStride 00",
                "count": 2,
                "mass_kg_each": 0.310,
                "body_envelope_m": [0.057, 0.057, 0.051],
                "simulation_cap_nm": 2.0,
            },
            "arms_jaw_tail": {
                "candidate": "RobStride 05",
                "count": 4,
                "mass_kg_each": 0.191,
                "body_envelope_m": [0.046, 0.046, 0.044],
                "simulation_cap_nm": 0.8,
            },
            "source": "https://github.com/RobStride/Product_Information",
        }
        report["limitations"] = [
            "All component masses, locations and box inertias are design allocations, not weighed CAD properties.",
            "Motor body envelopes omit mounting bolts, shafts and cable bends; fabrication requires supplier CAD.",
            "Drives use conservative planning torque caps; no torque-speed, thermal, backlash or bus dynamics.",
            "The 1.8 kg integration reserve is explicitly carried near the pelvis.",
            "Camera frusta and contact patches are ideal simulation sensors, not calibrated hardware.",
        ]
        path.write_text(json.dumps(report, indent=2) + "\n")


def build(directory: Path = HERE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    make_meshes(directory / "assets")
    make_head(directory / "assets")
    b = Builder()
    b.leg("left", 1)
    b.leg("right", -1)
    b.upper_body()
    b.finish(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output)
