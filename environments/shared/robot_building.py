"""Reusable MJCF construction and visual-mesh helpers for robot concepts.

Components carry explicit box-surrogate mass properties independently of visual
geometry. Callers own model dimensions, component allocations and drive settings.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


def values(items: Any) -> str:
    """Canonical numeric text, including scalars and numerical zero residue."""
    if isinstance(items, (str, int)):
        return str(items)
    if isinstance(items, (float, np.floating)):
        number = float(items)
        return "0" if abs(number) < 1e-14 else f"{number:.10g}"
    return " ".join(values(float(x)) for x in items)


def element(parent: ET.Element, tag: str, **attributes: Any) -> ET.Element:
    return ET.SubElement(parent, tag, {k.rstrip("_"): values(v) for k, v in attributes.items()})


def mesh_file(path: Path, vertices: list[Any], faces: list[Any]) -> None:
    lines = ["# Generated visual surface; not fabrication geometry"]
    lines += ["v " + values(v) for v in vertices]
    lines += ["f " + " ".join(str(i + 1) for i in f) for f in faces]
    path.write_text("\n".join(lines) + "\n")


def superellipsoid_mesh(path: Path, *, exponent: float, columns: int = 32, rings: int = 18) -> None:
    """Write a unit rounded-box surface with the selected signed-power profile."""
    cols = columns
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
    mesh_file(path, vertices, faces)


def tapered_tail_mesh(
    path: Path,
    *,
    length: float,
    rise: float,
    base_radius: float,
    tip_radius: float,
    taper: float,
    columns: int = 32,
    rings: int = 16,
) -> None:
    """Write a closed circular-section taper with a quadratic centreline rise."""
    cols = columns
    vertices: list[Any] = []
    for j in range(rings + 1):
        t = j / rings
        radius = base_radius * (1 - t) ** taper + tip_radius
        for i in range(cols):
            angle = 2 * math.pi * i / cols
            vertices.append((-length * t, radius * math.cos(angle), rise * t * t + radius * math.sin(angle)))
    faces: list[tuple[int, int, int]] = []
    for row in range(rings):
        for i in range(cols):
            a, b = row * cols + i, row * cols + (i + 1) % cols
            faces.extend(((a, a + cols, b), (b, a + cols, b + cols)))
    vertices.extend(((0, 0, 0), (-length, 0, rise)))
    for i in range(cols):
        faces.extend(
            (
                (len(vertices) - 2, i, (i + 1) % cols),
                (len(vertices) - 1, rings * cols + (i + 1) % cols, rings * cols + i),
            )
        )
    mesh_file(path, vertices, faces)


def loft(path: Path, profile: list[tuple[float, float, float, float]], *, exponent: float = 1.0) -> None:
    """Closed elliptical sections: x, centre-z, half-width, half-height.

    Smoothstep between stations gives a continuous first derivative without spline
    overshoot; the shape stays inside the declared section envelopes.
    """
    vertices: list[Any] = []
    faces: list[Any] = []
    columns, subdivisions = 32, 6
    sections: list[tuple[float, ...]] = []
    for left, right in zip(profile[:-1], profile[1:], strict=True):
        for t in np.linspace(0, 1, subdivisions, endpoint=False):
            w = t * t * (3 - 2 * t)
            sections.append(
                (
                    float(left[0] + t * (right[0] - left[0])),
                    *[float(left[i] + w * (right[i] - left[i])) for i in (1, 2, 3)],
                )
            )
    sections.append(profile[-1])
    for x, z, width, height in sections:
        for angle in np.arange(columns) * (2 * math.pi / columns):
            c, s = math.cos(angle), math.sin(angle)
            vertices.append(
                (x, width * math.copysign(abs(c) ** exponent, c), z + height * math.copysign(abs(s) ** exponent, s))
            )
    for row in range(len(sections) - 1):
        for i in range(columns):
            a, b = row * columns + i, row * columns + (i + 1) % columns
            faces.extend(((a, b, a + columns), (b, b + columns, a + columns)))
    vertices.extend(((profile[0][0], 0, profile[0][1]), (profile[-1][0], 0, profile[-1][1])))
    last = (len(sections) - 1) * columns
    for i in range(columns):
        faces.extend(
            ((len(vertices) - 2, (i + 1) % columns, i), (len(vertices) - 1, last + i, last + (i + 1) % columns))
        )
    mesh_file(path, vertices, faces)


def component_inertia(components: list[dict[str, Any]]) -> tuple[float, np.ndarray, np.ndarray]:
    """Return mass, centre of mass and central inertia via the parallel-axis theorem."""
    # Python 3.12 changed float sum accumulation; fsum is stable across
    # the supported versions and cancels symmetric contributions cleanly.
    mass = math.fsum(c["mass_kg"] for c in components)
    com = np.array([math.fsum(c["mass_kg"] * c["pos_m"][i] for c in components) / mass for i in range(3)])
    inertia = np.zeros((3, 3))
    for c in components:
        half = np.array(c["inertia_box_halfsize_m"])
        local = np.diag(c["mass_kg"] * (np.sum(half**2) - half**2) / 3)
        r = np.array(c["pos_m"]) - com
        inertia += local + c["mass_kg"] * ((r @ r) * np.eye(3) - np.outer(r, r))
    return mass, com, inertia


class RobotBuilder:
    """Free-root, joint-actuated MJCF assembly with explicit component inertias."""

    def __init__(
        self,
        model_name: str,
        *,
        root_name: str,
        root_position: tuple[float, float, float],
        statistic_center: Any,
        statistic_extent: Any,
        materials: dict[str, tuple[float, float, float, float]],
        cameras: dict[str, tuple[float, float, float]],
        camera_target: tuple[float, float, float],
        joint_defaults: dict[str, float],
        default_material: str,
        default_mesh: str,
    ) -> None:
        self.xml = ET.Element("mujoco", model=model_name)
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
        element(self.xml, "statistic", center=statistic_center, extent=statistic_extent)
        visual = element(self.xml, "visual")
        element(visual, "global", offwidth=1600, offheight=1200)
        element(visual, "quality", shadowsize=4096)
        element(visual, "headlight", ambient=(0.4, 0.4, 0.4), diffuse=(0.6, 0.6, 0.6))
        defaults = element(self.xml, "default")
        element(defaults, "geom", contype=0, conaffinity=0, group=1)
        element(defaults, "joint", limited="true", **joint_defaults)
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
        for name, rgba in materials.items():
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
        for name, pos in cameras.items():
            direction = np.array(pos) - camera_target
            z = direction / np.linalg.norm(direction)
            x = np.cross([0, 0, 1], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            element(self.world, "camera", name=name, pos=pos, xyaxes=(*x, *y), fovy=34)
        self.bodies: dict[str, ET.Element] = {}
        self.components: list[dict[str, Any]] = []
        self.joints: list[dict[str, Any]] = []
        self.root_position = root_position
        self.root_name = root_name
        self.default_material = default_material
        self.default_mesh = default_mesh
        self.root = self.body(self.world, root_name, root_position)
        element(self.root, "freejoint", name="root")

    def body(self, parent: ET.Element, name: str, pos: Any) -> ET.Element:
        body = element(parent, "body", name=name, pos=pos)
        self.bodies[name] = body
        return body

    def geom(
        self,
        body: ET.Element,
        name: str,
        kind: str,
        size: Any,
        pos: Any = (0, 0, 0),
        material: str | None = None,
        **kw: Any,
    ) -> ET.Element:
        return element(
            body,
            "geom",
            name=name,
            type=kind,
            size=size,
            pos=pos,
            material=self.default_material if material is None else material,
            **kw,
        )

    def mesh(
        self,
        body: ET.Element,
        name: str,
        scale: Any,
        pos: Any,
        material: str | None = None,
        source: str | None = None,
        **kw: Any,
    ) -> None:
        element(
            self.asset,
            "mesh",
            name=name + "_mesh",
            file=(self.default_mesh if source is None else source) + ".obj",
            scale=scale,
        )
        element(
            body,
            "geom",
            name=name,
            type="mesh",
            mesh=name + "_mesh",
            pos=pos,
            material=self.default_material if material is None else material,
            **kw,
        )

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

    def add_joint(
        self,
        body: ET.Element,
        name: str,
        axis: Any,
        limits: Any,
        cap: float,
        kp: float,
        kv: float,
        *,
        damping: float,
        frictionloss: float,
        armature: float,
    ) -> None:
        element(
            body,
            "joint",
            name=name,
            axis=axis,
            range=limits,
            damping=damping,
            frictionloss=frictionloss,
            armature=armature,
        )
        self.joints.append(dict(name=name, range=list(limits), torque_limit_nm=cap, kp=kp, kv=kv))

    def write(
        self,
        directory: Path,
        *,
        imu_site: str,
        contact_sites: list[tuple[str, str]],
        limitations: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for name, body in self.bodies.items():
            components = [c for c in self.components if c["body"] == name]
            mass, com, inertia = component_inertia(components)
            attrs = dict(
                pos=values(com),
                mass=values(mass),
                fullinertia=values(
                    (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
                ),
            )
            body.insert(0, ET.Element("inertial", attrs))
        equality = element(self.xml, "equality")
        element(equality, "weld", name="display_support", body1=self.root_name, active="false")
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
        element(sensors, "gyro", name="pelvis_gyro", site=imu_site)
        element(sensors, "accelerometer", name="pelvis_accelerometer", site=imu_site)
        element(sensors, "framequat", name="pelvis_orientation", objtype="site", objname=imu_site)
        element(sensors, "subtreecom", name="robot_center_of_mass", body=self.root_name)
        for name, site in contact_sites:
            element(sensors, "touch", name=name, site=site)
        key = element(self.xml, "keyframe")
        element(
            key,
            "key",
            name="home",
            qpos=(*self.root_position, 1, 0, 0, 0, *([0] * len(self.joints))),
            ctrl=[0] * len(self.joints),
        )
        ET.indent(self.xml, space="  ")
        (directory / "model.xml").write_text(ET.tostring(self.xml, encoding="unicode") + "\n")
        totals = {
            category: math.fsum(c["mass_kg"] for c in self.components if c["category"] == category)
            for category in dict.fromkeys(c["category"] for c in self.components)
        }
        report = {
            "status": "unweighed_concept_inertia_and_component_surrogates",
            "mass_kg": math.fsum(totals.values()),
            "categories_kg": dict(totals),
            "components": self.components,
            "joints": self.joints,
            "limitations": limitations,
        }
        report.update(metadata or {})
        (directory / "components.json").write_text(json.dumps(report, indent=2) + "\n")
