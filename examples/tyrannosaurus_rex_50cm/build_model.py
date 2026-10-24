"""Generate the 50 cm concept's MJCF, visual meshes and assigned-mass ledger.

Motor cases are provisional 60 mm diameter x 55 mm packaging envelopes, not
manufacturer CAD. Inertias come from explicit component surrogates, never the
decorative solid shells. No component mass or actuator response is measured.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT_HEIGHT = 0.253
RANGES = {
    "hip_yaw": (-0.35, 0.35),
    "hip_roll": (-0.30, 0.30),
    "hip_pitch": (-0.65, 0.65),
    "knee": (-0.30, 1.10),
    "ankle_pitch": (-0.60, 0.60),
    "ankle_roll": (-0.30, 0.30),
}


def values(items: Any) -> str:
    if isinstance(items, (str, int, float)):
        return str(items)
    return " ".join(f"{float(x):.10g}" for x in items)


def element(parent: ET.Element, tag: str, **attributes: Any) -> ET.Element:
    return ET.SubElement(parent, tag, {k.rstrip("_"): values(v) for k, v in attributes.items()})


def mesh_file(path: Path, vertices: list[Any], faces: list[Any]) -> None:
    lines = ["# Generated visual surface; not fabrication geometry"]
    lines += ["v " + values(v) for v in vertices]
    lines += ["f " + " ".join(str(i + 1) for i in f) for f in faces]
    path.write_text("\n".join(lines) + "\n")


def make_meshes(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cols, rings = 32, 18
    for name, exponent in (("rounded", 0.55), ("soft", 0.85)):
        vertices: list[Any] = [(0, 0, -1)]
        for j in range(1, rings):
            latitude = -math.pi / 2 + j * math.pi / rings
            for i in range(cols):
                longitude = i * 2 * math.pi / cols
                raw = (
                    math.cos(latitude) * math.cos(longitude),
                    math.cos(latitude) * math.sin(longitude),
                    math.sin(latitude),
                )
                vertices.append(tuple(math.copysign(abs(x) ** exponent, x) for x in raw))
        vertices.append((0, 0, 1))
        faces = [(0, 1 + (i + 1) % cols, 1 + i) for i in range(cols)]
        for row in range(rings - 2):
            for i in range(cols):
                a, b = 1 + row * cols + i, 1 + row * cols + (i + 1) % cols
                faces.extend(((a, b, a + cols), (b, b + cols, a + cols)))
        last = 1 + (rings - 2) * cols
        faces += [(last + i, last + (i + 1) % cols, len(vertices) - 1) for i in range(cols)]
        mesh_file(directory / f"{name}.obj", vertices, faces)
    vertices = []
    for j in range(17):
        t = j / 16
        radius = 0.034 * (1 - t) ** 0.9 + 0.002
        for i in range(cols):
            angle = 2 * math.pi * i / cols
            vertices.append((-0.24 * t, radius * math.cos(angle), 0.060 * t * t + radius * math.sin(angle)))
    faces = []
    for row in range(16):
        for i in range(cols):
            a, b = row * cols + i, row * cols + (i + 1) % cols
            faces.extend(((a, a + cols, b), (b, a + cols, b + cols)))
    vertices.extend(((0, 0, 0), (-0.24, 0, 0.060)))
    for i in range(cols):
        faces.extend(
            ((len(vertices) - 2, i, (i + 1) % cols), (len(vertices) - 1, 16 * cols + (i + 1) % cols, 16 * cols + i))
        )
    mesh_file(directory / "tail.obj", vertices, faces)


class Builder:
    def __init__(self) -> None:
        self.xml = ET.Element("mujoco", model="tyrannosaurus_rex_robot_50cm_concept")
        self.xml.append(
            ET.Comment(
                "Generated concept: unweighed component inertias, ideal torque-bounded drives; not hardware CAD."
            )
        )
        element(self.xml, "compiler", angle="radian", autolimits="true", inertiafromgeom="false", meshdir="assets")
        element(
            self.xml,
            "option",
            timestep="0.002",
            integrator="implicitfast",
            gravity=(0, 0, -9.81),
            cone="elliptic",
            iterations=100,
            tolerance="1e-9",
        )
        element(self.xml, "statistic", center=(-0.04, 0, 0.25), extent=0.75)
        visual = element(self.xml, "visual")
        element(visual, "global", offwidth=1600, offheight=1200)
        element(visual, "quality", shadowsize=4096)
        element(visual, "headlight", ambient=(0.4, 0.4, 0.4), diffuse=(0.6, 0.6, 0.6))
        defaults = element(self.xml, "default")
        element(defaults, "geom", contype=0, conaffinity=0, group=1)
        element(defaults, "joint", limited="true", damping=0.08, frictionloss=0.01, armature=0.0003)
        collision = element(defaults, "default", class_="collision")
        element(
            collision,
            "geom",
            contype=1,
            conaffinity=1,
            group=3,
            rgba=(0.9, 0.25, 0.1, 0.2),
            condim=4,
            friction=(0.9, 0.005, 0.0001),
            solref=(0.008, 1),
        )
        self.asset = element(self.xml, "asset")
        for name, rgba in {
            "sage": (0.29, 0.38, 0.22, 1),
            "cream": (0.78, 0.71, 0.55, 1),
            "dark_sage": (0.16, 0.22, 0.12, 1),
            "graphite": (0.065, 0.075, 0.075, 1),
            "metal": (0.25, 0.29, 0.30, 1),
            "rubber": (0.035, 0.04, 0.038, 1),
            "amber": (0.78, 0.32, 0.025, 1),
            "black": (0.008, 0.012, 0.014, 1),
            "glass": (0.025, 0.10, 0.15, 1),
            "white": (0.96, 0.95, 0.87, 1),
            "pcb": (0.07, 0.23, 0.16, 1),
            "orange": (0.9, 0.32, 0.04, 1),
        }.items():
            element(self.asset, "material", name=name, rgba=rgba, specular=0.35, shininess=0.45)
        self.world = element(self.xml, "worldbody")
        element(self.world, "light", pos=(1, -1, 2), dir=(-1, 1, -2), diffuse=(0.8, 0.8, 0.75))
        element(self.world, "light", pos=(-1, 1, 1.2), dir=(1, -1, -1.2), diffuse=(0.4, 0.4, 0.4), castshadow="false")
        element(
            self.world,
            "geom",
            name="floor",
            type="plane",
            size=(3, 3, 0.02),
            rgba=(0.88, 0.89, 0.84, 1),
            group=0,
            contype=1,
            conaffinity=1,
            friction=(0.9, 0.005, 0.0001),
            condim=4,
        )
        for name, pos in {"isometric": (1.0, -1.25, 0.8), "front": (1.5, 0, 0.43), "side": (0, -1.6, 0.38)}.items():
            direction = np.array(pos) - [-0.035, 0, 0.25]
            z = direction / np.linalg.norm(direction)
            x = np.cross([0, 0, 1], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            element(self.world, "camera", name=name, pos=pos, xyaxes=(*x, *y), fovy=34)
        self.bodies: dict[str, ET.Element] = {}
        self.components: list[dict[str, Any]] = []
        self.joints: list[dict[str, Any]] = []
        self.torso = self.body(self.world, "torso", (0, 0, ROOT_HEIGHT))
        element(self.torso, "freejoint", name="root")

    def body(self, parent: ET.Element, name: str, pos: Any) -> ET.Element:
        body = element(parent, "body", name=name, pos=pos)
        self.bodies[name] = body
        return body

    def geom(
        self, body: ET.Element, name: str, kind: str, size: Any, pos: Any = (0, 0, 0), material: str = "sage", **kw: Any
    ) -> ET.Element:
        return element(body, "geom", name=name, type=kind, size=size, pos=pos, material=material, **kw)

    def mesh(
        self, body: ET.Element, name: str, scale: Any, pos: Any, material: str = "sage", source: str = "soft", **kw: Any
    ) -> None:
        element(self.asset, "mesh", name=name + "_mesh", file=source + ".obj", scale=scale)
        element(body, "geom", name=name, type="mesh", mesh=name + "_mesh", pos=pos, material=material, **kw)

    def collision(self, body: ET.Element, name: str, kind: str, size: Any, pos: Any = (0, 0, 0), **kw: Any) -> None:
        element(body, "geom", name=name + "_collision", class_="collision", type=kind, size=size, pos=pos, **kw)

    def component(self, body: ET.Element, name: str, category: str, grams: float, pos: Any, halfsize: Any) -> None:
        self.components.append(
            dict(
                name=name,
                body=body.get("name"),
                category=category,
                mass_kg=grams / 1000,
                pos_m=list(pos),
                inertia_box_halfsize_m=list(halfsize),
            )
        )

    def joint(self, body: ET.Element, name: str, axis: Any, limits: Any, cap: float, kp: float, kv: float) -> None:
        element(
            body,
            "joint",
            name=name,
            axis=axis,
            range=limits,
            damping=0.08 if cap == 4 else 0.008,
            frictionloss=0.01 if cap == 4 else 0.0005,
            armature=0.0003 if cap == 4 else 0.000005,
        )
        self.joints.append(dict(name=name, range=list(limits), torque_limit_nm=cap, kp=kp, kv=kv))

    def motor(self, parent: ET.Element, name: str, pos: Any, axis: str) -> None:
        rotation = {"x": (0, math.pi / 2, 0), "y": (math.pi / 2, 0, 0), "z": (0, 0, 0)}[axis]
        self.geom(parent, name + "_case", "cylinder", (0.030, 0.0275), pos, "graphite", euler=rotation, group=2)
        self.collision(parent, name + "_case", "cylinder", (0.030, 0.0275), pos, euler=rotation)
        dims = [0.030] * 3
        dims["xyz".index(axis)] = 0.0275
        self.component(parent, name + "_motor", "leg_motors_12_x_330g", 330, pos, dims)
        cap = np.array(pos, dtype=float)
        face_sign = -1 if axis == "y" and pos[1] < 0 else 1
        cap["xyz".index(axis)] += face_sign * 0.028
        self.geom(parent, name + "_rim", "cylinder", (0.031, 0.0018), cap, "dark_sage", euler=rotation, group=2)
        cap["xyz".index(axis)] += face_sign * 0.002
        self.geom(parent, name + "_cap", "cylinder", (0.025, 0.0015), cap, "metal", euler=rotation, group=2)
        self.geom(parent, name + "_hub", "cylinder", (0.010, 0.0025), cap, "graphite", euler=rotation, group=2)
        plane = [i for i in range(3) if i != "xyz".index(axis)]
        for i in range(6):
            bolt = cap.copy()
            bolt["xyz".index(axis)] += face_sign * 0.002
            bolt[plane[0]] += 0.020 * math.cos(i * math.pi / 3)
            bolt[plane[1]] += 0.020 * math.sin(i * math.pi / 3)
            self.geom(
                parent, f"{name}_bolt_{i}", "cylinder", (0.0015, 0.0007), bolt, "graphite", euler=rotation, group=2
            )

    def leg(self, side: str, sign: int) -> None:
        self.motor(self.torso, side + "_hip_yaw", (0, sign * 0.045, 0.075), "z")
        yaw = self.body(self.torso, side + "_hip_yaw_body", (0, sign * 0.045, 0.075))
        self.joint(yaw, side + "_hip_yaw", (0, 0, 1), RANGES["hip_yaw"], 4, 80, 3)
        self.motor(yaw, side + "_hip_roll", (-0.032, sign * 0.035, -0.075), "x")
        roll = self.body(yaw, side + "_hip_roll_body", (0, sign * 0.035, -0.075))
        self.joint(roll, side + "_hip_roll", (1, 0, 0), RANGES["hip_roll"], 4, 100, 4)
        self.motor(roll, side + "_hip_pitch", (0.030, sign * 0.018, 0), "y")
        thigh = self.body(roll, side + "_thigh", (0.030, sign * 0.032, 0))
        self.joint(thigh, side + "_hip_pitch", (0, 1, 0), RANGES["hip_pitch"], 4, 110, 4)
        knee = (0.027, 0, -0.097)
        self.motor(thigh, side + "_knee", (0.027, sign * 0.026, -0.097), "y")
        shin = self.body(thigh, side + "_shin", knee)
        self.joint(shin, side + "_knee", (0, 1, 0), RANGES["knee"], 4, 110, 4)
        self.motor(shin, side + "_ankle_pitch", (-0.027, sign * 0.026, -0.096), "y")
        ankle = self.body(shin, side + "_ankle_pitch_body", (-0.027, 0, -0.096))
        self.joint(ankle, side + "_ankle_pitch", (0, 1, 0), RANGES["ankle_pitch"], 4, 100, 3)
        self.motor(ankle, side + "_ankle_roll", (-0.060, 0, -0.015), "x")
        foot = self.body(ankle, side + "_foot", (-0.032, 0, -0.015))
        self.joint(foot, side + "_ankle_roll", (1, 0, 0), RANGES["ankle_roll"], 4, 100, 3)
        for body, grams, p, dims in (
            (yaw, 35, (0, sign * 0.018, -0.04), (0.012, 0.025, 0.04)),
            (roll, 40, (0.018, sign * 0.016, 0), (0.025, 0.02, 0.025)),
            (thigh, 45, (0.014, 0, -0.049), (0.018, 0.014, 0.05)),
            (shin, 40, (-0.014, 0, -0.048), (0.015, 0.014, 0.05)),
            (ankle, 15, (-0.03, 0, -0.012), (0.035, 0.02, 0.012)),
        ):
            self.component(body, body.get("name", "") + "_structure", "load_bearing_structure", grams, p, dims)
        for body, hardware_grams in ((yaw, 15), (roll, 15), (thigh, 10), (shin, 10), (ankle, 10), (foot, 75.5)):
            self.component(
                body,
                body.get("name", "") + "_hardware",
                "bearings_feet_fasteners",
                hardware_grams,
                (0, 0, -0.015),
                (0.02, 0.02, 0.015),
            )
        self.component(
            shin, side + "_leg_wiring", "power_boards_distribution_wiring", 25, (-0.01, 0, -0.045), (0.006, 0.006, 0.04)
        )
        self.component(
            foot,
            side + "_foot_sensors",
            "mcu_interfaces_imu_contacts_audio",
            10,
            (0.015, 0, -0.037),
            (0.05, 0.03, 0.003),
        )
        self.component(
            thigh, side + "_leg_cover", "shells_and_guards", 20, (0.014, -sign * 0.015, -0.048), (0.02, 0.008, 0.05)
        )
        self.component(foot, side + "_foot_cover", "shells_and_guards", 5, (0.025, 0, -0.024), (0.05, 0.035, 0.018))
        for body, tip, radius in ((thigh, knee, 0.014), (shin, (-0.027, 0, -0.096), 0.010)):
            n = body.get("name", "")
            element(
                body,
                "geom",
                name=n + "_link",
                type="capsule",
                fromto=(0, 0, 0, *tip),
                size=radius,
                material="metal",
                group=2,
            )
            element(
                body,
                "geom",
                name=n + "_link_collision",
                class_="collision",
                type="capsule",
                fromto=(0, 0, 0, *tip),
                size=radius,
            )
        self.mesh(
            thigh, side + "_thigh_fairing", (0.027, 0.020, 0.047), (0.014, -sign * 0.012, -0.047), euler=(0, -0.27, 0)
        )
        self.mesh(
            shin, side + "_shin_fairing", (0.022, 0.017, 0.037), (-0.013, -sign * 0.012, -0.043), euler=(0, 0.27, 0)
        )
        self.mesh(
            foot, side + "_sole_visual", (0.070, 0.039, 0.004), (0.015, 0, -0.041), "rubber", source="rounded", group=2
        )
        self.collision(foot, side + "_sole", "box", (0.070, 0.039, 0.004), (0.015, 0, -0.041))
        self.geom(foot, side + "_foot_plate", "box", (0.062, 0.033, 0.002), (0.015, 0, -0.035), "metal", group=2)
        self.mesh(foot, side + "_heel_cover", (0.050, 0.035, 0.016), (0.012, 0, -0.022))
        for i, y in enumerate((-0.026, 0, 0.026)):
            self.mesh(foot, f"{side}_toe_{i}", (0.030, 0.012, 0.020), (0.052, y, -0.020))
            self.geom(foot, f"{side}_toe_pad_{i}", "ellipsoid", (0.012, 0.011, 0.013), (0.075, y, -0.027), "rubber")
        for i, (x, y) in enumerate(((-0.05, -0.023), (-0.05, 0.023), (0.05, -0.023), (0.05, 0.023))):
            element(
                foot,
                "site",
                name=f"{side}_contact_{i}",
                type="box",
                pos=(0.015 + x, y, -0.041),
                size=(0.023, 0.018, 0.006),
                rgba=(0.2, 0.7, 0.2, 0),
                group=3,
            )
        element(foot, "site", name=side + "_sole_center", pos=(0.015, 0, -0.045), size=0.002, rgba=(0, 0, 0, 0))

    def upper_body(self) -> None:
        t = self.torso
        self.mesh(t, "torso_shell", (0.089, 0.084, 0.085), (-0.012, 0, 0.050))
        self.mesh(t, "belly_panel", (0.028, 0.065, 0.074), (0.060, 0, 0.046), "cream")
        self.mesh(t, "back_access_panel", (0.033, 0.060, 0.066), (-0.072, 0, 0.050), "dark_sage")
        self.collision(t, "torso", "ellipsoid", (0.078, 0.072, 0.078), (-0.012, 0, 0.050))
        for i, z in enumerate((0.030, 0.040, 0.050, 0.060, 0.070)):
            self.geom(t, f"back_vent_{i}", "box", (0.001, 0.025, 0.0015), (-0.105, 0, z), "graphite")
        for i, (y, z) in enumerate(((-0.035, 0.018), (0.035, 0.018), (-0.035, 0.084), (0.035, 0.084))):
            self.geom(
                t, f"back_screw_{i}", "cylinder", (0.0025, 0.001), (-0.098, y, z), "metal", euler=(0, math.pi / 2, 0)
            )
        for name, category, grams, pos, size, material in (
            ("battery", "battery_506g_plus_restraint_44g", 550, (-0.026, 0, 0.018), (0.069, 0.021, 0.0215), "graphite"),
            ("compute_tray", "jetson_storage_cooling_mount", 300, (-0.040, 0, 0.090), (0.030, 0.050, 0.021), "pcb"),
            (
                "control_board",
                "mcu_interfaces_imu_contacts_audio",
                180,
                (0.018, 0, 0.065),
                (0.020, 0.040, 0.012),
                "pcb",
            ),
            (
                "power_board",
                "power_boards_distribution_wiring",
                300,
                (-0.026, 0, 0.049),
                (0.030, 0.040, 0.009),
                "metal",
            ),
            ("pelvis_structure", "load_bearing_structure", 250, (-0.010, 0, 0.020), (0.060, 0.065, 0.025), "metal"),
            (
                "integration_reserve",
                "unallocated_integration_reserve",
                600,
                (-0.018, 0, 0.045),
                (0.060, 0.060, 0.045),
                "orange",
            ),
        ):
            self.component(t, name, category, grams, pos, size)
            if name != "integration_reserve":
                self.geom(t, name + "_envelope", "box", size, pos, material, group=4)
        self.component(t, "torso_shell_mass", "shells_and_guards", 100, (-0.012, 0, 0.05), (0.089, 0.084, 0.085))
        element(t, "site", name="pelvis_imu", pos=(-0.010, 0, 0.025), size=0.003, rgba=(0.9, 0.3, 0.1, 0))
        for side, sign in (("left", 1), ("right", -1)):
            base = (0.025, sign * 0.080, 0.094)
            self.geom(
                t,
                side + "_shoulder_case",
                "cylinder",
                (0.020, 0.012),
                base,
                "graphite",
                euler=(math.pi / 2, 0, 0),
                group=2,
            )
            self.component(t, side + "_shoulder_motor", "character_motors_4_x_18g", 18, base, (0.010, 0.013, 0.017))
            self.component(t, side + "_shoulder_mount", "load_bearing_structure", 10, base, (0.015, 0.015, 0.015))
            arm = self.body(t, side + "_arm", (0.025, sign * 0.097, 0.094))
            self.joint(arm, side + "_shoulder_pitch", (0, 1, 0), (-1.1, 0.35), 0.1, 1.2, 0.035)
            self.component(arm, side + "_arm_shell", "moving_arms", 25, (0.027, 0, -0.020), (0.035, 0.009, 0.02))
            element(
                arm,
                "geom",
                name=side + "_upper_arm",
                type="capsule",
                fromto=(0, 0, 0, 0.025, sign * 0.004, -0.026),
                size=0.009,
                material="sage",
            )
            element(
                arm,
                "geom",
                name=side + "_forearm",
                type="capsule",
                fromto=(0.025, sign * 0.004, -0.026, 0.058, sign * 0.004, -0.020),
                size=0.007,
                material="sage",
            )
            element(
                arm,
                "geom",
                name=side + "_arm_collision",
                class_="collision",
                type="capsule",
                fromto=(0, 0, 0, 0.058, sign * 0.004, -0.020),
                size=0.007,
            )
            for i, y in enumerate((-0.006, 0.006)):
                self.geom(arm, f"{side}_finger_{i}", "ellipsoid", (0.010, 0.004, 0.007), (0.063, y, -0.023), "cream")
        self.component(t, "tail_motor", "character_motors_4_x_18g", 18, (-0.086, 0, 0.016), (0.013, 0.010, 0.017))
        self.geom(
            t,
            "tail_bearing",
            "cylinder",
            (0.039, 0.009),
            (-0.093, 0, 0.018),
            "graphite",
            euler=(0, math.pi / 2, 0),
            group=2,
        )
        tail = self.body(t, "tail", (-0.102, 0, 0.018))
        self.joint(tail, "tail_yaw", (0, 0, 1), (-0.35, 0.35), 0.1, 1.2, 0.04)
        self.component(tail, "tail_shell_mass", "moving_tail", 80, (-0.095, 0, 0.015), (0.10, 0.025, 0.025))
        self.mesh(tail, "tail_shell", (1, 1, 1), (0, 0, 0), source="tail")
        for i, (a, b, r) in enumerate(
            (
                ((0, 0, 0), (-0.075, 0, 0.006), 0.027),
                ((-0.075, 0, 0.006), (-0.16, 0, 0.026), 0.017),
                ((-0.16, 0, 0.026), (-0.235, 0, 0.058), 0.006),
            )
        ):
            element(
                tail,
                "geom",
                name=f"tail_segment_{i}_collision",
                class_="collision",
                type="capsule",
                fromto=(*a, *b),
                size=r,
            )
        self.head()

    def head(self) -> None:
        t = self.torso
        self.component(t, "head_yaw_motor", "head_motors_2_x_82g", 82, (-0.015, 0, 0.150), (0.01425, 0.017, 0.02325))
        self.geom(t, "neck_yaw_case", "cylinder", (0.025, 0.018), (-0.015, 0, 0.150), "graphite", group=2)
        yaw = self.body(t, "head_yaw_body", (-0.015, 0, 0.150))
        self.joint(yaw, "head_yaw", (0, 0, 1), (-0.65, 0.65), 0.6, 6, 0.25)
        self.component(yaw, "head_nod_motor", "head_motors_2_x_82g", 82, (0.040, 0, 0.030), (0.01425, 0.02325, 0.017))
        self.geom(yaw, "neck_nod_case", "box", (0.01425, 0.02325, 0.017), (0.040, 0, 0.030), "graphite", group=2)
        head = self.body(yaw, "head", (0.040, 0, 0.030))
        self.joint(head, "head_nod", (0, 1, 0), (-0.25, 0.20), 0.6, 6, 0.25)
        self.component(head, "head_upper_shell", "shells_and_guards", 100, (0.025, 0, 0.010), (0.11, 0.095, 0.057))
        self.component(head, "head_structure", "load_bearing_structure", 30, (0.010, 0, 0), (0.03, 0.035, 0.02))
        self.component(head, "head_bearings", "bearings_feet_fasteners", 29, (0, 0, 0), (0.02, 0.03, 0.02))
        self.component(head, "snout_camera", "camera", 61, (0.105, 0, 0.005), (0.00875, 0.0455, 0.014))
        self.component(head, "jaw_motor", "character_motors_4_x_18g", 18, (0.015, 0, -0.020), (0.013, 0.010, 0.017))
        self.mesh(head, "head_shell", (0.078, 0.089, 0.052), (-0.020, 0, 0.030))
        self.mesh(head, "upper_snout_shell", (0.068, 0.076, 0.026), (0.070, 0, 0.006), source="rounded")
        self.collision(head, "head", "ellipsoid", (0.075, 0.085, 0.050), (-0.020, 0, 0.030))
        self.collision(head, "upper_snout", "ellipsoid", (0.064, 0.073, 0.024), (0.070, 0, 0.006))
        self.mesh(head, "snout_camera_panel", (0.004, 0.048, 0.019), (0.135, 0, 0.006), "graphite", source="rounded")
        self.geom(head, "camera_package_envelope", "box", (0.00875, 0.0455, 0.014), (0.120, 0, 0.006), "metal", group=4)
        for label, y in (("left", 0.035), ("rgb", 0), ("right", -0.035)):
            self.geom(
                head,
                "snout_" + label + "_rim",
                "cylinder",
                (0.010, 0.002),
                (0.140, y, 0.006),
                "metal",
                euler=(0, math.pi / 2, 0),
            )
            self.geom(
                head,
                "snout_" + label + "_lens",
                "cylinder",
                (0.0075, 0.0025),
                (0.142, y, 0.006),
                "glass",
                euler=(0, math.pi / 2, 0),
            )
            element(head, "camera", name="snout_" + label, pos=(0.145, y, 0.006), xyaxes=(0, -1, 0, 0, 0, 1), fovy=69)
        for side, sign in (("left", 1), ("right", -1)):
            rotation = (0, 0, -sign * 0.44)
            self.geom(
                head,
                side + "_eye_socket",
                "ellipsoid",
                (0.034, 0.014, 0.037),
                (0.015, sign * 0.077, 0.042),
                "dark_sage",
                euler=rotation,
            )
            self.geom(
                head,
                side + "_eye_white",
                "ellipsoid",
                (0.029, 0.009, 0.033),
                (0.020, sign * 0.088, 0.042),
                "cream",
                euler=rotation,
            )
            self.geom(
                head,
                side + "_iris",
                "ellipsoid",
                (0.025, 0.006, 0.028),
                (0.026, sign * 0.097, 0.042),
                "amber",
                euler=rotation,
            )
            self.geom(
                head,
                side + "_pupil",
                "ellipsoid",
                (0.017, 0.004, 0.023),
                (0.029, sign * 0.102, 0.042),
                "black",
                euler=rotation,
            )
            self.geom(
                head,
                side + "_eye_glint",
                "ellipsoid",
                (0.006, 0.002, 0.007),
                (0.035, sign * 0.107, 0.055),
                "white",
                euler=rotation,
            )
            self.mesh(head, side + "_brow", (0.040, 0.014, 0.010), (0.008, sign * 0.081, 0.071), euler=rotation)
        jaw = self.body(head, "jaw", (0.010, 0, -0.032))
        self.joint(jaw, "jaw_pitch", (0, 1, 0), (0, 0.28), 0.1, 1.5, 0.04)
        self.component(jaw, "jaw_shell_mass", "shells_and_guards", 30, (0.052, 0, 0), (0.080, 0.077, 0.022))
        self.mesh(jaw, "jaw_shell", (0.082, 0.079, 0.024), (0.052, 0, 0), "cream", source="rounded")
        self.collision(jaw, "jaw", "ellipsoid", (0.073, 0.070, 0.020), (0.052, 0, 0))
        for i, y in enumerate((-0.055, 0.055)):
            self.geom(head, f"tooth_{i}", "ellipsoid", (0.008, 0.006, 0.013), (0.131, y, -0.020), "white")

    def finish(self, directory: Path) -> None:
        for name, body in self.bodies.items():
            components = [c for c in self.components if c["body"] == name]
            mass = sum(c["mass_kg"] for c in components)
            com = sum((c["mass_kg"] * np.array(c["pos_m"]) for c in components), np.zeros(3)) / mass
            inertia = np.zeros((3, 3))
            for c in components:
                half = np.array(c["inertia_box_halfsize_m"])
                local = np.diag(c["mass_kg"] * (np.sum(half**2) - half**2) / 3)
                r = np.array(c["pos_m"]) - com
                inertia += local + c["mass_kg"] * ((r @ r) * np.eye(3) - np.outer(r, r))
            attrs = dict(
                pos=values(com),
                mass=values(mass),
                fullinertia=values(
                    (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
                ),
            )
            body.insert(0, ET.Element("inertial", attrs))
        equality = element(self.xml, "equality")
        element(equality, "weld", name="display_support", body1="torso", active="false")
        actuators = element(self.xml, "actuator")
        sensors = element(self.xml, "sensor")
        for joint in self.joints:
            name = joint["name"]
            element(
                actuators,
                "position",
                name=name + "_servo",
                joint=name,
                kp=joint["kp"],
                kv=joint["kv"],
                ctrlrange=joint["range"],
                forcerange=(-joint["torque_limit_nm"], joint["torque_limit_nm"]),
                forcelimited="true",
            )
            element(sensors, "jointpos", name=name + "_position", joint=name)
            element(sensors, "jointvel", name=name + "_velocity", joint=name)
        element(sensors, "gyro", name="pelvis_gyro", site="pelvis_imu")
        element(sensors, "accelerometer", name="pelvis_accelerometer", site="pelvis_imu")
        element(sensors, "framequat", name="pelvis_orientation", objtype="site", objname="pelvis_imu")
        element(sensors, "subtreecom", name="robot_center_of_mass", body="torso")
        for side in ("left", "right"):
            for i in range(4):
                element(sensors, "touch", name=f"{side}_contact_force_{i}", site=f"{side}_contact_{i}")
        key = element(self.xml, "keyframe")
        element(
            key,
            "key",
            name="home",
            qpos=(0, 0, ROOT_HEIGHT, 1, 0, 0, 0, *([0] * len(self.joints))),
            ctrl=[0] * len(self.joints),
        )
        ET.indent(self.xml, space="  ")
        (directory / "model.xml").write_text(ET.tostring(self.xml, encoding="unicode") + "\n")
        totals: defaultdict[str, float] = defaultdict(float)
        for c in self.components:
            totals[c["category"]] += c["mass_kg"]
        report = {
            "status": "unweighed_concept_inertia_and_component_surrogates",
            "mass_kg": sum(totals.values()),
            "categories_kg": dict(totals),
            "components": self.components,
            "joints": self.joints,
            "limitations": [
                "Motor envelopes, brackets and mass locations are provisional, not supplier CAD.",
                "Inertias use component box surrogates and the parallel-axis theorem.",
                "Ideal position drives are torque-limited; electrical, thermal, gearbox and bus dynamics are not modeled.",
                "600 g integration reserve is assigned near the pelvis; no mass is removed to make the model balance.",
            ],
        }
        (directory / "components.json").write_text(json.dumps(report, indent=2) + "\n")


def build(directory: Path = HERE) -> None:
    make_meshes(directory / "assets")
    b = Builder()
    b.leg("left", 1)
    b.leg("right", -1)
    b.upper_body()
    b.finish(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output)
