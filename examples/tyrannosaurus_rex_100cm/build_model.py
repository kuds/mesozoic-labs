"""Generate the long-bodied metre T-Rex, with explicit hardware allocations.

Original lofted surfaces follow a continuous dinosaur silhouette. Components,
collision proxies and load-bearing joints remain separate from the visual skin.
This is a packaging/simulation concept, not fabrication CAD or a walking policy.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from environments.shared.robot_building import RobotBuilder, element, loft, mesh_file, superellipsoid_mesh, values

HERE = Path(__file__).resolve().parent
ROOT_HEIGHT = 0.575
SPECIFICATION = "https://github.com/RobStride/Product_Information/blob/main/灵足时代RS系列产品规格介绍(2026.010.09).pdf"
# Rounded-up October 2026 masses include the stated production tolerances.
MOTORS = {
    "RS03": dict(mass_g=950, radius=0.053, depth=0.056, cap=20.0, rated=21.0),
    "RS04": dict(mass_g=1530, radius=0.060, depth=0.056, cap=30.0, rated=35.0),
    "RS00": dict(mass_g=320, radius=0.0285, depth=0.052, cap=4.0, rated=5.0),
    "RS05": dict(mass_g=250, radius=0.023, depth=0.045, cap=0.8, rated=1.8),
}


def make_surfaces(assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    for name, exponent in (("rounded", 0.55), ("soft", 0.85)):
        superellipsoid_mesh(assets / f"{name}.obj", exponent=exponent)
    loft(
        assets / "body.obj",
        [
            (-0.43, 0.130, 0.105, 0.100),
            (-0.32, 0.140, 0.135, 0.120),
            (-0.16, 0.165, 0.205, 0.160),
            (0.03, 0.145, 0.205, 0.170),
            (0.18, 0.140, 0.160, 0.135),
            (0.29, 0.175, 0.100, 0.095),
            (0.34, 0.205, 0.055, 0.065),
        ],
    )
    loft(
        assets / "neck.obj",
        [
            (-0.07, -0.015, 0.095, 0.100),
            (0.0, 0.025, 0.092, 0.092),
            (0.065, 0.080, 0.083, 0.080),
            (0.15, 0.120, 0.073, 0.063),
            (0.20, 0.125, 0.055, 0.055),
        ],
    )
    loft(
        assets / "skull.obj",
        [
            (-0.08, 0.020, 0.035, 0.050),
            (-0.02, 0.035, 0.100, 0.080),
            (0.075, 0.030, 0.125, 0.080),
            (0.22, 0.005, 0.112, 0.060),
            (0.38, -0.005, 0.090, 0.042),
            (0.43, -0.004, 0.055, 0.035),
            (0.445, -0.004, 0.008, 0.020),
        ],
        exponent=0.85,
    )
    loft(
        assets / "jaw.obj",
        [
            (0.015, 0.0, 0.065, 0.012),
            (0.10, -0.016, 0.105, 0.020),
            (0.25, -0.015, 0.100, 0.030),
            (0.40, -0.010, 0.075, 0.027),
            (0.445, -0.005, 0.008, 0.012),
        ],
    )
    # Rounded cone teeth, drawn from the same MJCF as every other surface.
    vertices = [(0.008 * math.cos(a), 0.007 * math.sin(a), 0.009) for a in np.arange(12) * math.pi / 6]
    vertices += [(0, 0, -0.020), (0, 0, 0.009)]
    faces = [(i, (i + 1) % 12, 12) for i in range(12)] + [(13, (i + 1) % 12, i) for i in range(12)]
    mesh_file(assets / "tooth.obj", vertices, faces)
    loft(
        assets / "tail.obj",
        [
            (-0.86, 0.100, 0.003, 0.004),
            (-0.70, 0.080, 0.022, 0.026),
            (-0.48, 0.040, 0.045, 0.050),
            (-0.25, 0.010, 0.077, 0.076),
            (-0.06, 0.000, 0.110, 0.100),
            (0.045, 0.0, 0.125, 0.110),
        ],
    )


class Builder(RobotBuilder):
    def __init__(self) -> None:
        super().__init__(
            "tyrannosaurus_rex_robot_100cm_long_body_concept",
            root_name="torso",
            root_position=(0, 0, ROOT_HEIGHT),
            statistic_center="-0.15 0 0.50",
            statistic_extent="2.40",
            materials={
                "sage": (0.40, 0.49, 0.43, 1),
                "cream": (0.72, 0.77, 0.73, 1),
                "dark_sage": (0.11, 0.19, 0.16, 1),
                "graphite": (0.065, 0.075, 0.075, 1),
                "metal": (0.25, 0.29, 0.30, 1),
                "rubber": (0.035, 0.04, 0.038, 1),
                "amber": (0.78, 0.32, 0.025, 1),
                "black": (0.008, 0.012, 0.014, 1),
                "glass": (0.025, 0.10, 0.15, 1),
                "white": (0.96, 0.95, 0.87, 1),
                "pcb": (0.07, 0.23, 0.16, 1),
                "orange": (0.9, 0.32, 0.04, 1),
            },
            cameras={"isometric": (3, -3, 1.9), "front": (3.3, 0, 1.0), "side": (0, -3.8, 1.1)},
            camera_target=(-0.15, 0, 0.50),
            joint_defaults=dict(damping=0.08, frictionloss=0.01, armature=0.0003),
            default_material="sage",
            default_mesh="soft",
        )
        self.torso = self.root
        self.motor_assignments: list[dict[str, Any]] = []

    def joint(self, body: ET.Element, name: str, axis: Any, limits: Any, cap: float, kp: float, kv: float) -> None:
        self.add_joint(
            body,
            name,
            axis,
            limits,
            cap,
            kp,
            kv,
            damping=0.15 if cap >= 10 else 0.015,
            frictionloss=0.025 if cap >= 10 else 0.002,
            armature=0.002 if cap >= 10 else 0.0001,
        )

    def drive(self, parent: ET.Element, name: str, pos: Any, axis: str, part: str) -> None:
        spec = MOTORS[part]
        rotation = {"x": (0, math.pi / 2, 0), "y": (math.pi / 2, 0, 0), "z": (0, 0, 0)}[axis]
        radius, halfdepth = spec["radius"], spec["depth"] / 2
        self.geom(parent, name + "_case", "cylinder", (radius, halfdepth), pos, "graphite", euler=rotation, group=2)
        self.collision(parent, name + "_case", "cylinder", (radius, halfdepth), pos, euler=rotation)
        dims = [radius] * 3
        dims["xyz".index(axis)] = halfdepth
        self.component(
            parent,
            name + "_motor",
            "leg_actuators" if part in ("RS03", "RS04") else "expression_actuators",
            spec["mass_g"],
            pos,
            dims,
        )
        self.motor_assignments.append(
            dict(
                name=name,
                part=part,
                mass_kg=spec["mass_g"] / 1000,
                envelope_m=[2 * radius, 2 * radius, 2 * halfdepth],
                axis=axis,
            )
        )
        face = np.array(pos, dtype=float)
        sign = -1 if axis == "y" and pos[1] < 0 else 1
        face["xyz".index(axis)] += sign * (halfdepth + 0.003)
        self.geom(parent, name + "_hub", "cylinder", (radius * 0.65, 0.003), face, "metal", euler=rotation, group=2)

    def link(self, body: ET.Element, name: str, end: Any, radius: float) -> None:
        for collision in (False, True):
            attrs: dict[str, Any] = dict(
                name=name + ("_collision" if collision else ""), type="capsule", fromto=(0, 0, 0, *end), size=radius
            )
            attrs.update(class_="collision") if collision else attrs.update(material="metal", group=2)
            element(body, "geom", **attrs)

    def leg(self, side: str, sign: int) -> None:
        self.drive(self.torso, side + "_hip_yaw", (0, sign * 0.125, 0.090), "z", "RS03")
        yaw = self.body(self.torso, side + "_hip_yaw_body", (0, sign * 0.125, 0.090))
        self.joint(yaw, side + "_hip_yaw", (0, 0, 1), (-0.30, 0.30), 20, 450, 20)
        self.drive(yaw, side + "_hip_roll", (-0.045, 0, -0.090), "x", "RS03")
        roll = self.body(yaw, side + "_hip_roll_body", (0, 0, -0.090))
        self.joint(roll, side + "_hip_roll", (1, 0, 0), (-0.30, 0.30), 20, 650, 25)
        self.drive(roll, side + "_hip_pitch", (0.065, sign * 0.060, -0.015), "y", "RS04")
        thigh = self.body(roll, side + "_thigh", (0.065, sign * 0.020, -0.015))
        self.joint(thigh, side + "_hip_pitch", (0, 1, 0), (-0.65, 0.65), 30, 900, 35)
        knee_pos = (0.100, 0, -0.230)
        self.drive(thigh, side + "_knee", (0.100, sign * 0.040, -0.230), "y", "RS04")
        shin = self.body(thigh, side + "_shin", knee_pos)
        self.joint(shin, side + "_knee", (0, 1, 0), (-0.15, 1.10), 30, 900, 35)
        ankle_pos = (-0.140, 0, -0.190)
        self.drive(shin, side + "_ankle_pitch", (-0.140, sign * 0.030, -0.190), "y", "RS03")
        ankle = self.body(shin, side + "_ankle_pitch_body", ankle_pos)
        self.joint(ankle, side + "_ankle_pitch", (0, 1, 0), (-0.55, 0.55), 20, 800, 30)
        self.drive(ankle, side + "_ankle_roll", (-0.090, 0, -0.020), "x", "RS03")
        foot = self.body(ankle, side + "_foot", (-0.060, 0, -0.020))
        self.joint(foot, side + "_ankle_roll", (1, 0, 0), (-0.25, 0.25), 20, 700, 25)
        for body, grams, pos, half in (
            (yaw, 180, (0, 0, -0.04), (0.035, 0.030, 0.04)),
            (roll, 200, (0.030, sign * 0.010, -0.008), (0.045, 0.035, 0.020)),
            (thigh, 350, (0.050, 0, -0.115), (0.045, 0.030, 0.120)),
            (shin, 270, (-0.070, 0, -0.095), (0.045, 0.025, 0.105)),
            (ankle, 160, (-0.045, 0, -0.020), (0.060, 0.030, 0.025)),
            (foot, 350, (0.060, 0, -0.040), (0.150, 0.075, 0.020)),
        ):
            self.component(body, body.get("name", "") + "_structure", "structure_bearings_feet", grams, pos, half)
        for body, pos in (
            (yaw, (0, 0, -0.02)),
            (roll, (0.02, 0, 0)),
            (thigh, (0.03, 0, -0.08)),
            (shin, (-0.05, 0, -0.08)),
            (ankle, (-0.04, 0, 0)),
            (foot, (0.02, 0, -0.025)),
        ):
            self.component(
                body, body.get("name", "") + "_cooling", "cooling_allocation", 100, pos, (0.040, 0.030, 0.010)
            )
        self.component(thigh, side + "_thigh_cover", "shells_guards", 180, (0.040, 0, -0.075), (0.100, 0.085, 0.150))
        self.component(shin, side + "_shin_cover", "shells_guards", 100, (-0.070, 0, -0.085), (0.070, 0.045, 0.095))
        self.component(shin, side + "_leg_wiring", "wiring_power", 130, (-0.070, 0, -0.095), (0.010, 0.010, 0.11))
        self.component(foot, side + "_foot_sensors", "sensors", 100, (0.060, 0, -0.045), (0.125, 0.060, 0.008))
        self.link(thigh, side + "_thigh_link", knee_pos, 0.028)
        self.link(shin, side + "_shin_link", ankle_pos, 0.023)
        self.geom(
            thigh,
            side + "_thigh_fairing",
            "ellipsoid",
            (0.110, 0.088, 0.168),
            (0.040, -sign * 0.008, -0.065),
            "cream",
            euler=(0, -0.40, 0),
        )
        self.geom(
            shin,
            side + "_shin_fairing",
            "ellipsoid",
            (0.043, 0.040, 0.109),
            (-0.070, 0, -0.095),
            "sage",
            euler=(0, 0.63, 0),
        )
        self.mesh(
            foot, side + "_sole_visual", (0.150, 0.075, 0.006), (0.060, 0, -0.059), "rubber", source="rounded", group=2
        )
        self.collision(foot, side + "_sole", "box", (0.150, 0.075, 0.006), (0.060, 0, -0.059))
        self.geom(foot, side + "_foot_plate", "box", (0.100, 0.067, 0.004), (0.100, 0, -0.051), "metal", group=2)
        self.geom(foot, side + "_heel_cover", "ellipsoid", (0.080, 0.063, 0.024), (0.015, 0, -0.025), "cream")
        for i, y in enumerate((-0.050, 0, 0.050)):
            self.geom(foot, f"{side}_toe_{i}", "ellipsoid", (0.090, 0.022, 0.022), (0.122, y, -0.037), "cream")
            self.geom(foot, f"{side}_claw_{i}", "ellipsoid", (0.028, 0.018, 0.012), (0.195, y, -0.044), "graphite")
        for i, (x, y) in enumerate(((-0.100, -0.040), (-0.100, 0.040), (0.100, -0.040), (0.100, 0.040))):
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
        # A raised ankle with a rigid sloping metatarsal; no hidden extra joint.
        for child in foot:
            if child.tag in ("geom", "site") and child.get("pos"):
                position = np.fromstring(child.get("pos", ""), sep=" ")
                position[2] -= 0.055
                child.set("pos", values(position))
        for component in self.components:
            if component["body"] == side + "_foot":
                component["pos_m"][2] -= 0.055
        self.component(
            foot,
            side + "_metatarsal",
            "structure_bearings_feet",
            180,
            (0.040, -sign * 0.030, -0.050),
            (0.030, 0.025, 0.050),
        )
        for collision in (False, True):
            attrs = dict(
                name=side + "_metatarsal_link" + ("_collision" if collision else ""),
                type="capsule",
                fromto=(0.025, -sign * 0.040, 0, 0.060, -sign * 0.040, -0.085),
                size=0.018,
            )
            attrs.update(class_="collision") if collision else attrs.update(material="metal", group=2)
            element(foot, "geom", **attrs)
        self.geom(
            foot,
            side + "_metatarsal_cover",
            "ellipsoid",
            (0.029, 0.032, 0.050),
            (0.042, -sign * 0.030, -0.050),
            "cream",
            euler=(0, -0.38, 0),
        )

    def upper_body(self) -> None:
        t = self.torso
        self.mesh(t, "torso_shell", (1, 1, 1), (0, 0, 0), "cream", source="body")

        self.collision(t, "upper_torso", "ellipsoid", (0.275, 0.155, 0.095), (-0.070, 0, 0.190))
        self.collision(t, "central_pelvis", "box", (0.080, 0.065, 0.060), (-0.025, 0, 0.070))
        for name, category, grams, pos, half, mat in (
            ("battery", "battery_restraint", 2800, (-0.150, 0, 0.105), (0.090, 0.060, 0.045), "graphite"),
            ("compute_tray", "compute_cooling_storage", 650, (0.065, 0, 0.150), (0.065, 0.060, 0.025), "pcb"),
            ("control_can_imu", "control", 220, (0.205, 0, 0.150), (0.035, 0.055, 0.020), "pcb"),
            ("power_tray", "wiring_power", 750, (-0.300, 0, 0.160), (0.025, 0.080, 0.045), "metal"),
            ("torso_structure", "structure_bearings_feet", 1700, (-0.040, 0, 0.150), (0.280, 0.140, 0.120), "metal"),
            ("integration_reserve", "unallocated_reserve", 2200, (-0.025, 0, 0.110), (0.100, 0.090, 0.060), "orange"),
        ):
            self.component(t, name, category, grams, pos, half)
            if name not in ("torso_structure", "integration_reserve"):
                self.geom(t, name + "_envelope", "box", half, pos, mat, group=4)
        self.component(t, "torso_shell_mass", "shells_guards", 900, (-0.040, 0, 0.150), (0.360, 0.205, 0.165))
        element(t, "site", name="pelvis_imu", pos=(-0.025, 0, 0.040), size=0.005, rgba=(1, 0.2, 0, 0), group=3)
        for side, sign in (("left", 1), ("right", -1)):
            self.geom(
                t, side + "_flank_vent", "ellipsoid", (0.165, 0.010, 0.060), (-0.065, sign * 0.201, 0.150), "graphite"
            )
            for i in range(4):
                self.geom(
                    t,
                    f"{side}_flank_rib_{i}",
                    "ellipsoid",
                    (0.008, 0.012, 0.060),
                    (-0.165 + i * 0.065, sign * 0.210, 0.150),
                    "sage",
                    euler=(0, -0.18, 0),
                )
        for side, sign in (("left", 1), ("right", -1)):
            self.drive(t, side + "_shoulder", (0.190, sign * 0.160, 0.095), "y", "RS05")
            arm = self.body(t, side + "_arm", (0.190, sign * 0.190, 0.095))
            self.joint(arm, side + "_shoulder_pitch", (0, 1, 0), (-0.70, 0.35), 0.8, 12, 0.5)
            self.component(arm, side + "_arm_shell", "moving_arms", 160, (0.055, 0, -0.050), (0.070, 0.022, 0.060))
            self.geom(
                arm,
                side + "_upper_arm",
                "ellipsoid",
                (0.025, 0.023, 0.065),
                (0.018, 0, -0.050),
                "cream",
                euler=(0, -0.3, 0),
            )
            self.geom(arm, side + "_forearm", "ellipsoid", (0.052, 0.021, 0.022), (0.066, 0, -0.095), "sage")
            self.collision(arm, side + "_arm", "capsule", (0.020,), fromto=(0, 0, 0, 0.105, 0, -0.095))
            for i, y in enumerate((-0.012, 0.012)):
                self.geom(
                    arm,
                    f"{side}_finger_{i}",
                    "ellipsoid",
                    (0.025, 0.008, 0.016),
                    (0.110, y, -0.098),
                    "graphite",
                    euler=(0, -0.4, 0),
                )
        self.head()
        self.tail()

    def head(self) -> None:
        self.drive(self.torso, "head_yaw", (0.290, 0, 0.190), "z", "RS00")
        yaw = self.body(self.torso, "head_yaw_body", (0.290, 0, 0.190))
        self.joint(yaw, "head_yaw", (0, 0, 1), (-0.40, 0.40), 3.0, 55, 2.0)
        self.component(yaw, "neck_frame", "structure_bearings_feet", 300, (0.060, 0, 0.065), (0.110, 0.055, 0.075))
        self.component(yaw, "neck_skin", "shells_guards", 160, (0.055, 0, 0.060), (0.135, 0.090, 0.110))
        self.mesh(yaw, "neck_shell", (1, 1, 1), (0, 0, 0), "cream", source="neck")
        self.drive(yaw, "head_pitch", (0.140, -0.055, 0.120), "y", "RS00")
        head = self.body(yaw, "head", (0.140, 0, 0.120))
        self.joint(head, "head_pitch", (0, 1, 0), (-0.16, 0.16), 4.0, 80, 3)
        self.mesh(head, "head_shell", (1, 1, 1), (0, 0, 0), "cream", source="skull")
        self.collision(head, "skull", "ellipsoid", (0.120, 0.100, 0.065), (0.060, 0, 0.030))
        self.collision(head, "snout", "ellipsoid", (0.150, 0.080, 0.035), (0.290, 0, 0.010))
        self.component(head, "upper_head_shell", "head_shell_frame", 650, (0.150, 0, 0.030), (0.260, 0.125, 0.085))
        self.component(head, "head_frame", "head_shell_frame", 200, (0.060, 0, 0), (0.100, 0.080, 0.040))
        self.component(head, "eyes_audio", "sensors", 80, (0.050, 0, 0.040), (0.055, 0.115, 0.025))
        # The long head occluded the upper chest camera view. Move the same
        # camera assembly to the snout and carry its mass on the head joint.
        self.component(head, "snout_depth_camera", "sensors", 120, (0.405, 0, 0.005), (0.016, 0.052, 0.015))
        self.geom(head, "snout_camera_bar", "box", (0.017, 0.053, 0.016), (0.428, 0, 0.005), "graphite", group=2)
        for side, y in (("left", 0.0375), ("right", -0.0375)):
            self.geom(
                head,
                "snout_" + side + "_lens",
                "cylinder",
                (0.006, 0.002),
                (0.447, y, 0.005),
                "glass",
                euler=(0, math.pi / 2, 0),
                group=2,
            )
            element(head, "camera", name="snout_" + side, pos=(0.450, y, 0.005), xyaxes=(0, -1, 0, 0, 0, 1), fovy=60)
        self.drive(head, "jaw", (-0.015, 0.055, -0.078), "y", "RS05")
        for side, sign in (("left", 1), ("right", -1)):
            self.geom(
                head, side + "_eye_socket", "ellipsoid", (0.040, 0.012, 0.025), (0.040, sign * 0.119, 0.047), "graphite"
            )
            self.geom(head, side + "_eye", "ellipsoid", (0.021, 0.008, 0.016), (0.045, sign * 0.130, 0.048), "amber")
            self.geom(head, side + "_pupil", "ellipsoid", (0.009, 0.004, 0.014), (0.050, sign * 0.137, 0.047), "black")
            self.geom(
                head,
                side + "_brow",
                "ellipsoid",
                (0.070, 0.015, 0.015),
                (0.040, sign * 0.116, 0.074),
                "sage",
                euler=(0, 0.15, 0),
            )
            self.geom(
                head, side + "_nostril", "ellipsoid", (0.015, 0.004, 0.007), (0.365, sign * 0.089, 0.011), "graphite"
            )
            for i in range(5):
                x = 0.13 + i * 0.054
                y = sign * (0.110 - 0.11 * (x - 0.13))
                self.mesh(head, f"{side}_tooth_{i}", (1, 1, 1), (x, y, -0.050), "white", source="tooth")
        jaw = self.body(head, "jaw", (-0.015, 0, -0.078))
        self.joint(jaw, "jaw_pitch", (0, 1, 0), (0, 0.22), 1.2, 14, 0.6)
        self.component(jaw, "jaw_shell", "head_shell_frame", 280, (0.210, 0, -0.015), (0.220, 0.105, 0.025))
        self.mesh(jaw, "jaw_shell", (1, 1, 1), (0, 0, 0), "sage", source="jaw")
        self.collision(jaw, "jaw_chin", "ellipsoid", (0.110, 0.065, 0.012), (0.330, 0, -0.015))

    def tail(self) -> None:
        pos = (-0.365, 0, 0.130)
        self.drive(self.torso, "tail_yaw", pos, "z", "RS00")
        tail = self.body(self.torso, "tail", pos)
        self.joint(tail, "tail_yaw", (0, 0, 1), (-0.22, 0.22), 2.0, 22, 1.2)
        self.component(
            tail, "tail_root_bearings", "structure_bearings_feet", 200, (-0.030, 0, 0), (0.045, 0.050, 0.060)
        )
        self.component(tail, "tail_shell_spar", "moving_tail", 650, (-0.310, 0, 0.030), (0.430, 0.080, 0.080))
        self.mesh(tail, "tail_shell", (1, 1, 1), (0, 0, 0), "cream", source="tail")
        for i, (a, b, r) in enumerate(
            (
                ((-0.09, 0, 0), (-0.30, 0, 0.02), 0.065),
                ((-0.30, 0, 0.02), (-0.57, 0, 0.06), 0.035),
                ((-0.57, 0, 0.06), (-0.85, 0, 0.10), 0.010),
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

    def finish(self, directory: Path) -> None:
        self.write(
            directory,
            imu_site="pelvis_imu",
            contact_sites=[
                (f"{side}_contact_force_{i}", f"{side}_contact_{i}") for side in ("left", "right") for i in range(4)
            ],
            limitations=[
                "All masses, locations and box inertias are unweighed design allocations.",
                "October 2026 motor dimensions/masses checked; cases omit bolts, shafts, connectors and brackets.",
                "Planning torque caps are not measured continuous limits in this chassis; thermal and electrical models are absent.",
                "A 2.2 kg integration reserve and 1.2 kg cooling allocation are included, without claiming the vendor heat-sink condition is met.",
                "Visual skins are not collision CAD; full-range clearance, load paths and walking remain unqualified.",
            ],
            metadata=dict(
                nominal_height_m=1.0,
                design="long_body_revision_2",
                motor_assignments=self.motor_assignments,
                motor_specifications=MOTORS,
                specification_source=SPECIFICATION,
            ),
        )


def build(directory: Path = HERE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    make_surfaces(directory / "assets")
    b = Builder()
    b.leg("left", 1)
    b.leg("right", -1)
    b.upper_body()
    b.finish(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output)
