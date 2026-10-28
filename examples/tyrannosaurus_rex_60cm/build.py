"""Build a free-base 60 cm T. rex skeleton study with full-size part reserves.

600 mm is the eventual head crown, not the bare torso height. All inertia is
explicitly allocated; decorative skeleton/outline geoms do not add hidden mass.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from environments.shared.robot_building import RobotBuilder, element

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARDWARE = ROOT / "hardware/tyrannosaurus_rex_60cm/m01_skeleton"
DESIGN = HARDWARE / "design.json"
RANGES = {
    "hip_yaw": (-0.25, 0.25),
    "hip_roll": (-0.25, 0.25),
    "hip_pitch": (-0.6, 0.45),
    "knee": (-0.25, 0.9),
    "ankle_pitch": (-0.6, 0.5),
    "ankle_roll": (-0.25, 0.25),
}


class Skeleton(RobotBuilder):
    def __init__(self, design: dict[str, Any]) -> None:
        self.design = design
        self.motor_names: list[str] = []
        self.reserves: list[str] = []
        self.plugs: dict[str, str] = {}
        super().__init__(
            "tyrannosaurus_rex_60cm_skeleton_concept_r01",
            root_name="pelvis",
            root_position=(0, 0, design["hip_height_m"]),
            statistic_center=(-0.07, 0, 0.30),
            statistic_extent=1.25,
            materials={
                "frame": (0.23, 0.29, 0.27, 1),
                "metal": (0.52, 0.58, 0.57, 1),
                "motor": (0.055, 0.075, 0.075, 1),
                "rim": (0.32, 0.39, 0.36, 1),
                "pcb": (0.055, 0.30, 0.18, 1),
                "compute": (0.14, 0.27, 0.32, 1),
                "battery": (0.57, 0.28, 0.07, 1),
                "rubber": (0.035, 0.043, 0.039, 1),
                "outline": (0.50, 0.65, 0.45, 1),
                "reserve": (0.25, 0.53, 0.72, 0.20),
            },
            cameras={"side": (0, -2, 0.45), "isometric": (1.1, -1.8, 0.9)},
            camera_target=(-0.06, 0, 0.30),
            joint_defaults={"damping": 0.10, "frictionloss": 0.20, "armature": 0.002},
            default_material="frame",
            default_mesh="unused",
        )

    def rod(self, parent: Any, name: str, a: Any, b: Any, radius: float = 0.006, **kw: Any) -> None:
        element(
            parent,
            "geom",
            name=name,
            type="capsule",
            fromto=(*a, *b),
            size=radius,
            material=kw.pop("material", "frame"),
            group=kw.pop("group", 2),
            **kw,
        )

    def motor(self, parent: Any, name: str, pos: Any, axis: str, plug_offset: Any) -> None:
        spec = self.design["motor"]
        radius, half = spec["planning_diameter_m"] / 2, spec["planning_length_m"] / 2
        euler = {"x": (0, math.pi / 2, 0), "y": (math.pi / 2, 0, 0), "z": (0, 0, 0)}[axis]
        geom = name + "_case"
        self.geom(
            parent,
            geom,
            "cylinder",
            (radius, half),
            pos,
            "motor",
            euler=euler,
            group=2,
            contype=1,
            conaffinity=1,
            friction=(0.7, 0.005, 0.0001),
        )
        self.motor_names.append(geom)
        dims = np.full(3, radius)
        dims["xyz".index(axis)] = half
        self.component(parent, name + "_motor", "leg_motors", spec["mass_kg"] * 1000, pos, dims)
        # Connector/bend reservation is deliberately separate from the motor case.
        plug = name + "_cable_reserve"
        self.geom(parent, plug, "box", (0.009, 0.012, 0.009), np.array(pos) + plug_offset, "reserve", group=5)
        self.reserves.append(plug)
        self.plugs[plug] = geom
        for face in (-1, 1):
            cap = np.array(pos, dtype=float)
            cap["xyz".index(axis)] += face * (half + 0.0005)
            self.geom(parent, name + f"_face_{face}", "cylinder", (0.022, 0.0005), cap, "rim", euler=euler, group=2)

    def joint(self, body: Any, name: str, axis: Any, kind: str) -> None:
        self.add_joint(
            body,
            name,
            axis,
            RANGES[kind],
            self.design["motor"]["simulation_torque_cap_nm"],
            160 if kind in ("hip_pitch", "knee", "ankle_pitch") else 120,
            6,
            damping=0.10,
            frictionloss=0.20,
            armature=0.002,
        )

    def leg(self, side: str, sign: int) -> None:
        self.motor(self.root, side + "_hip_yaw", (0, sign * 0.045, 0.09), "z", (-0.044, 0, 0))
        yaw = self.body(self.root, side + "_yaw", (0, sign * 0.045, 0.09))
        self.joint(yaw, side + "_hip_yaw", (0, 0, 1), "hip_yaw")
        self.motor(yaw, side + "_hip_roll", (-0.035, sign * 0.035, -0.09), "x", (0, 0, -0.044))
        roll = self.body(yaw, side + "_roll", (0, sign * 0.035, -0.09))
        self.joint(roll, side + "_hip_roll", (1, 0, 0), "hip_roll")
        self.motor(roll, side + "_hip_pitch", (0.035, 0, 0), "y", (0.044, 0, 0))
        thigh = self.body(roll, side + "_thigh", (0.035, sign * 0.032, 0))
        self.joint(thigh, side + "_hip_pitch", (0, 1, 0), "hip_pitch")
        knee = np.array(self.design["thigh_vector_m"])
        self.motor(thigh, side + "_knee", knee + (0, sign * 0.032, 0), "y", (-0.044, 0, 0))
        shin = self.body(thigh, side + "_shin", knee)
        self.joint(shin, side + "_knee", (0, 1, 0), "knee")
        shin_vector = np.array(self.design["shin_vector_m"])
        self.motor(shin, side + "_ankle_pitch", shin_vector + (0, sign * 0.032, 0), "y", (0, 0, 0.044))
        ankle = self.body(shin, side + "_ankle", shin_vector)
        self.joint(ankle, side + "_ankle_pitch", (0, 1, 0), "ankle_pitch")
        self.motor(ankle, side + "_ankle_roll", (-0.068, 0, -0.02), "x", (0, 0, 0.044))
        foot = self.body(ankle, side + "_foot", (-0.036, 0, -0.02))
        self.joint(foot, side + "_ankle_roll", (1, 0, 0), "ankle_roll")
        for body, grams, center, halfsize in (
            (yaw, 100, (0, sign * 0.017, -0.06), (0.03, 0.025, 0.05)),
            (roll, 70, (0.018, sign * 0.02, 0), (0.03, 0.025, 0.025)),
            (thigh, 160, knee / 2, (0.04, 0.02, 0.08)),
            (shin, 140, shin_vector / 2, (0.04, 0.02, 0.08)),
            (ankle, 70, (-0.035, 0, -0.02), (0.04, 0.025, 0.025)),
            (foot, 180, (0.025, 0, -0.045), (0.080, 0.044, 0.015)),
        ):
            self.component(body, body.get("name", "") + "_structure", "frames_bearings_feet", grams, center, halfsize)
        self.component(shin, side + "_harness", "harness", 50, shin_vector / 2, (0.01, 0.01, 0.07))
        self.component(foot, side + "_sensing", "foot_sensors", 20, (0.025, 0, -0.05), (0.06, 0.03, 0.004))
        self.rod(yaw, side + "_hip_yoke", (0, 0, -0.032), (0, sign * 0.035, -0.09), 0.008)
        self.rod(roll, side + "_pitch_yoke", (0, 0, 0), (0.035, sign * 0.032, 0), 0.008)
        for parent, name, end in ((thigh, "thigh", knee), (shin, "shin", shin_vector)):
            for rail, offset in enumerate((-0.006, -0.020)):
                start = np.array([0, sign * offset, 0])
                self.rod(parent, f"{side}_{name}_rail_{rail}", start, end + start, 0.005)
        self.rod(ankle, side + "_ankle_yoke", (0, 0, 0), (-0.036, 0, -0.02), 0.007)
        for y in (-0.025, 0.025):
            self.rod(foot, f"{side}_metatarsal_{y}", (0, y, 0), (0.025, y, -0.044), 0.006)
        self.geom(
            foot,
            side + "_sole",
            "box",
            (0.080, 0.044, 0.006),
            (0.025, 0, -0.054),
            "rubber",
            group=2,
            contype=1,
            conaffinity=1,
            condim=4,
            friction=(0.8, 0.005, 0.0001),
            solref=(0.008, 1),
        )
        self.geom(foot, side + "_plate", "box", (0.073, 0.038, 0.002), (0.025, 0, -0.046), "metal", group=2)
        # Three fixed toe ribs over the flat test sole; no extra toe joints.
        for i, y in enumerate((-0.027, 0, 0.027)):
            self.rod(foot, f"{side}_toe_{i}", (0.025, y, -0.039), (0.096, y, -0.039), 0.005, material="metal")
        element(foot, "site", name=side + "_sole_center", pos=(0.025, 0, -0.060), size=0.002, rgba=(0, 0, 0, 0))
        element(
            foot,
            "site",
            name=side + "_touch",
            type="box",
            pos=(0.025, 0, -0.054),
            size=(0.081, 0.045, 0.007),
            rgba=(0, 0, 0, 0),
            group=3,
        )

    def torso(self) -> None:
        for item in self.design["assemblies"]:
            name, pos = item["name"], np.array(item["position_m"])
            self.component(
                self.root,
                name,
                "electronics_and_battery",
                item["mass_kg"] * 1000,
                pos,
                np.array(item["installed_size_m"]) / 2,
            )
            reserve = name + "_reserve"
            self.reserves.append(reserve)
            self.geom(self.root, reserve, "box", np.array(item["installed_size_m"]) / 2, pos, "reserve", group=5)
            for part in item["parts"]:
                self.geom(
                    self.root,
                    part["name"],
                    "box",
                    np.array(part["size_m"]) / 2,
                    pos + part["offset_m"],
                    item["material"],
                    group=4,
                )
        self.component(
            self.root, "pelvis_spine_trays", "frames_bearings_feet", 400, (-0.025, 0, 0.08), (0.18, 0.08, 0.06)
        )
        self.component(self.root, "torso_harness", "harness", 80, (-0.04, 0, 0.055), (0.12, 0.04, 0.05))
        self.component(self.root, "integration_allowance", "integration_reserve", 600, (0, 0, 0.06), (0.10, 0.07, 0.08))
        for sign in (-1, 1):
            y = sign * 0.075
            self.rod(self.root, f"spine_{sign}", (-0.255, y, 0.13), (0.17, y, 0.13), 0.006)
            self.rod(self.root, f"rear_battery_rail_{sign}", (-0.25, sign * 0.050, 0.003), (-0.09, sign * 0.050, 0.003))
            self.rod(self.root, f"rear_battery_hanger_{sign}", (-0.20, y, 0.13), (-0.20, sign * 0.050, 0.003), 0.005)
            self.rod(self.root, f"pelvis_web_{sign}", (0, y, 0.13), (0, sign * 0.045, 0.105), 0.006)
            self.rod(self.root, f"neck_rail_{sign}", (0.17, sign * 0.050, 0.13), (0.23, sign * 0.040, 0.17), 0.005)
            self.rod(self.root, f"tail_root_{sign}", (-0.20, y, 0.13), (-0.30, sign * 0.025, 0.10), 0.005)
        for i, x in enumerate((-0.25, -0.06, 0.17)):
            self.rod(self.root, f"spine_cross_{i}", (x, -0.075, 0.13), (x, 0.075, 0.13), 0.005)
        self.rod(self.root, "fixed_tail_spar", (-0.30, 0, 0.10), (-0.61, 0, 0.045), 0.004, material="metal")
        for item in self.design["future_payloads"]:
            self.component(
                self.root,
                "future_" + item["name"],
                "future_fixed_payloads",
                item["mass_kg"] * 1000,
                item["position_m"],
                item["inertia_halfsize_m"],
            )
        # Thin outline is a size/character guide, not fabricated head or shell CAD.
        skull = [
            (0.18, 0.17),
            (0.22, 0.24),
            (0.32, 0.24),
            (0.43, 0.20),
            (0.43, 0.15),
            (0.29, 0.12),
            (0.21, 0.13),
            (0.18, 0.17),
        ]
        body = [
            (-0.26, 0.13),
            (-0.07, 0.165),
            (0.12, 0.165),
            (0.19, 0.12),
            (0.17, 0.01),
            (0.03, -0.06),
            (-0.19, -0.015),
            (-0.26, 0.13),
        ]
        tail = [(-0.22, 0.13), (-0.40, 0.115), (-0.61, 0.045), (-0.40, 0.040), (-0.22, 0.015)]
        for name, outline, halfwidth in (("skull", skull, 0.065), ("torso", body, 0.085), ("tail", tail, 0.015)):
            for sign in (-1, 1):
                for i, (a, b) in enumerate(zip(outline[:-1], outline[1:], strict=True)):
                    self.rod(
                        self.root,
                        f"outline_{name}_{sign}_{i}",
                        (a[0], sign * halfwidth, a[1]),
                        (b[0], sign * halfwidth, b[1]),
                        0.0015,
                        material="outline",
                        group=1,
                    )
        element(self.root, "site", name="pelvis_imu", pos=(0, 0, 0.04), size=0.003, rgba=(0, 0, 0, 0))
        element(self.root, "site", name="future_head_crown", pos=(0.27, 0, 0.24), size=0.002, rgba=(0, 0, 0, 0))


def build() -> Skeleton:
    design = json.loads(DESIGN.read_text())
    builder = Skeleton(design)
    builder.leg("left", 1)
    builder.leg("right", -1)
    builder.torso()
    builder.write(
        HERE,
        imu_site="pelvis_imu",
        contact_sites=[(s + "_contact", s + "_touch") for s in ("left", "right")],
        limitations=[
            "Design allocations, not measured mass or manufacturing CAD.",
            "Only motor cases and flat soles have simulation contact. Frames are noncontact visual routing proxies.",
            "Future head, arms and fixed tail contribute mass/inertia but outlines are not collision-qualified modules.",
            "Ideal direct position drives: no measured torque-speed, backlash, thermal or electrical model.",
            "M1 engineering study only; not registered in SB3 or approved for physical construction.",
        ],
        metadata={
            "hardware_revision": design["revision"],
            "motor_cases": builder.motor_names,
            "installation_reserves": builder.reserves,
            "connector_owners": builder.plugs,
        },
    )
    # Remove the reusable builder's inactive display weld: this study has no fixture at all.
    equality = builder.xml.find("equality")
    if equality is not None:
        builder.xml.remove(equality)
    import xml.etree.ElementTree as ET

    ET.indent(builder.xml, space="  ")
    (HERE / "model.xml").write_text(ET.tostring(builder.xml, encoding="unicode") + "\n")
    with (HARDWARE / "mass_budget.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("component", "body", "category", "allocated_mass_kg", "basis"))
        for row in builder.components:
            writer.writerow(
                (
                    row["name"],
                    row["body"],
                    row["category"],
                    row["mass_kg"],
                    "catalog nominal" if row["category"] == "leg_motors" else "unweighed allocation",
                )
            )
    return builder


if __name__ == "__main__":
    build()
