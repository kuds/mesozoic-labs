"""Author MuJoCo render skins in millimetres on an unrotated reference skeleton."""

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator


def format_vector(values):
    """Store coordinates to one micrometre in the metre-based MJCF."""
    return " ".join("0" if abs(x) < 5e-7 else f"{x:.6f}".rstrip("0").rstrip(".") for x in values)


def angles(n):
    return np.arange(n) * 2 * np.pi / n


def format_rows(items, per_row, formatter):
    rows = []
    for i in range(0, len(items), per_row):
        rows.append("  ".join(formatter(item) for item in items[i : i + per_row]))
    return "\n                ".join(rows)


@dataclass
class Surface:
    vertices: list[np.ndarray] = field(default_factory=list)
    texcoords: list[tuple[float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    weights: dict[str, list[tuple[int, float]]] = field(default_factory=lambda: defaultdict(list))


class SkinBuilder:
    """Triangulated surfaces with bone weights in the posed skeleton's world frame."""

    def __init__(self, xml: str):
        self.bind_positions: dict[str, np.ndarray] = {}
        self.layers: dict[tuple[str, str], Surface] = {}
        world = ET.fromstring(xml).find("worldbody")
        if world is None:
            raise ValueError("Missing worldbody in anatomical MJCF.")
        self._bind_bodies(world, np.zeros(3))

    def _bind_bodies(self, parent: ET.Element, origin: np.ndarray) -> None:
        for body in parent.findall("body"):
            position = origin + np.fromstring(body.get("pos", "0 0 0"), sep=" ") * 1000
            self.bind_positions[body.attrib["name"]] = position
            self._bind_bodies(body, position)

    def surface(self, region, material):
        key = (region, material)
        if key not in self.layers:
            self.layers[key] = Surface()
        return self.layers[key]

    def add_point(self, surface, position, weights, shade=0.5):
        index = len(surface.vertices)
        surface.vertices.append(np.asarray(position) / 1000)
        surface.texcoords.append((0.35, np.clip(shade, 0.01, 0.99)))
        for bone, weight in weights.items():
            if weight > 1e-7:
                surface.weights[bone].append((index, weight))

    def tube(
        self,
        region,
        mat,
        stations,
        anchor,
        *,
        axial=False,
        weights=None,
        color=None,
        spokes=16,
        subdivisions=2,
        exponent=1,
    ):
        # Cross sections are (x,y,z,half-width,half-depth). The end caps stay inside a neighbouring surface.
        rows = np.array(stations, dtype=float)
        distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1))]
        sections = [
            start + (end - start) * np.arange(subdivisions) / subdivisions
            for start, end in zip(distances[:-1], distances[1:])
        ]
        samples = np.concatenate([*sections, distances[-1:]])
        interpolator = PchipInterpolator(distances, rows, axis=0)
        curve = interpolator(samples)
        tangents = interpolator.derivative()(samples)[:, :3]
        s = self.surface(region, mat)
        first = len(s.vertices)
        for row, tan in zip(curve, tangents):
            centre = row[:3] + self.bind_positions[anchor]
            w = weights(row[:3]) if weights else {anchor: 1}
            forward = tan / np.linalg.norm(tan)
            if axial:
                lateral = np.array([0, 1, 0])
                vertical = np.array([0, 0, 1])
            else:
                guide = np.array([0, 1, 0]) if abs(forward[1]) < 0.92 else np.array([0, 0, 1])
                vertical = np.cross(forward, guide)
                vertical /= np.linalg.norm(vertical)
                lateral = np.cross(vertical, forward)
            power = exponent(row[:3]) if callable(exponent) else exponent
            for t in angles(spokes):
                c, st = np.cos(t), np.sin(t)
                p = (
                    centre
                    + lateral * row[3] * np.sign(c) * abs(c) ** power
                    + vertical * row[4] * np.sign(st) * abs(st) ** power
                )
                self.add_point(s, p, w, color(row[:3], t) if color else 0.5)
        for j in range(len(samples) - 1):
            for k in range(spokes):
                a = first + j * spokes + k
                b = first + j * spokes + (k + 1) % spokes
                s.faces.extend(((a, b, b + spokes), (a, b + spokes, a + spokes)))
        for j in range(1, spokes - 1):
            a = first + (len(samples) - 1) * spokes
            s.faces.extend(((first, first + j + 1, first + j), (a, a + j, a + j + 1)))

    def ellipsoid(self, region, mat, bone, center, radii, shade=0.5):
        x, y, z = center
        rx, ry, rz = radii
        stations = [
            (x + rx * t, y, z, ry * np.sqrt(1 - t * t), rz * np.sqrt(1 - t * t))
            for t in [-0.999, -0.84, -0.5, 0, 0.5, 0.84, 0.999]
        ]
        self.tube(region, mat, stations, bone, axial=True, spokes=14, subdivisions=1, color=lambda p, t: shade)

    def xml(self, palette, comment="Articulated render surfaces; UVs encode the local hide tone.") -> str:
        assets = [f"    <!-- {comment} -->"]
        for mat, (low, high) in palette.items():
            assets.append(
                f'    <texture name="tone_{mat}" type="2d" builtin="gradient" rgb1="{format_vector(low)}" rgb2="{format_vector(high)}" width="8" height="128"/>'
            )
            spec = ".48" if mat in ("eye", "iris") else ".12" if mat in ("horn", "scale") else ".045"
            assets.append(f'    <material name="skin_{mat}" texture="tone_{mat}" specular="{spec}" shininess=".24"/>')
        for (region, mat), s in self.layers.items():
            vertex = format_rows(s.vertices, 8, format_vector)
            uv = format_rows(s.texcoords, 12, format_vector)
            face = format_rows(s.faces, 10, lambda f: " ".join(map(str, f)))
            assets.append(
                f'    <skin name="{region}_{mat}" group="1" material="skin_{mat}"\n          vertex="{vertex}"\n          texcoord="{uv}"\n          face="{face}">'
            )
            for bone, weights in s.weights.items():
                ids = format_rows([x[0] for x in weights], 24, str)
                vals = format_rows([x[1] for x in weights], 24, lambda x: f"{x:.5f}".rstrip("0").rstrip("."))
                assets.append(
                    f'      <bone body="{bone}" bindpos="{format_vector(self.bind_positions[bone] / 1000)}" bindquat="1 0 0 0"\n            vertid="{ids}"\n            vertweight="{vals}"/>'
                )
            assets.append("    </skin>")
        return "\n".join(assets)
