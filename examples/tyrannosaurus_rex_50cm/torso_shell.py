"""Continuous outer torso with separate convex sections around the hip bay."""

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from environments.shared.robot_building import element, mesh_file

from .build_model import Builder

# x, center-z, half-width, half-height. The raised middle underside preserves
# the hip-motor bay; no whole-body convex hull may fill that opening.
PROFILE = (
    (-0.285, 0.080, 0.003, 0.005),
    (-0.265, 0.080, 0.060, 0.070),
    (-0.240, 0.081, 0.098, 0.085),
    (-0.185, 0.094, 0.112, 0.100),
    (-0.090, 0.1215, 0.112, 0.1065),
    (-0.075, 0.134, 0.112, 0.094),
    (-0.040, 0.1715, 0.110, 0.0565),
    (0.040, 0.1715, 0.110, 0.0565),
    (0.056, 0.129, 0.106, 0.079),
    (0.075, 0.130, 0.100, 0.070),
    (0.110, 0.120, 0.100, 0.070),
    (0.173, 0.120, 0.095, 0.054),
    (0.197, 0.120, 0.002, 0.002),
)
SECTIONS = {"rear": PROFILE[:7], "bridge": PROFILE[6:8], "shoulder": PROFILE[7:10], "front": PROFILE[9:]}
COLLISIONS = tuple("torso_" + name + "_shell_collision" for name in SECTIONS)


def build_meshes(directory: Path) -> None:
    profile = np.array(PROFILE)
    # Monotone top/underside interpolation avoids station ridges and spline
    # overshoot into the hip bay. Every collision section shares these samples.
    dimensions = np.column_stack((profile[:, 1] - profile[:, 3], profile[:, 1] + profile[:, 3], profile[:, 2]))
    interpolate = PchipInterpolator(profile[:, 0], dimensions)
    samples = np.unique(
        np.concatenate(
            [
                np.linspace(a, b, max(2, int(np.ceil((b - a) / 0.007)) + 1))
                for a, b in zip(profile[:-1, 0], profile[1:, 0], strict=True)
            ]
        )
    )
    sections = [(float(x), *map(float, interpolate(x))) for x in samples]
    _surface(directory / "torso_continuous.obj", sections)
    for name, bounds in SECTIONS.items():
        _surface(directory / f"torso_{name}.obj", [row for row in sections if bounds[0][0] <= row[0] <= bounds[-1][0]])


def _surface(path: Path, sections: list[tuple[float, ...]]) -> None:
    columns = 32
    vertices = [
        (x, width * np.cos(angle), (low + high) / 2 + (high - low) / 2 * np.sin(angle))
        for x, low, high, width in sections
        for angle in np.arange(columns) * (2 * np.pi / columns)
    ]
    faces: list[tuple[int, int, int]] = []
    for row in range(len(sections) - 1):
        for i in range(columns):
            a, b = row * columns + i, row * columns + (i + 1) % columns
            faces.extend(((a, b, a + columns), (b, b + columns, a + columns)))
    vertices.extend((x, 0, (low + high) / 2) for x, low, high, _ in (sections[0], sections[-1]))
    last = (len(sections) - 1) * columns
    for i in range(columns):
        faces.extend(
            ((len(vertices) - 2, (i + 1) % columns, i), (len(vertices) - 1, last + i, last + (i + 1) % columns))
        )
    mesh_file(path, vertices, faces)


def add_shell(builder: Builder) -> None:
    old = builder.asset.find("mesh[@name='torso_shell_mesh']")
    if old is not None:
        builder.asset.remove(old)
    builder.mesh(builder.torso, "torso_shell", (1, 1, 1), (0, 0, 0), source="torso_continuous")
    for name in SECTIONS:
        mesh = f"torso_{name}_collision_mesh"
        element(builder.asset, "mesh", name=mesh, file=f"torso_{name}.obj")
        element(
            builder.torso,
            "geom",
            name=f"torso_{name}_shell_collision",
            class_="collision",
            type="mesh",
            mesh=mesh,
        )
