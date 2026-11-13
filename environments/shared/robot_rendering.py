"""CPU rendering and review-sheet mechanics for compiled MuJoCo geometry.

Callers supply view directions, visible groups, removable-panel patterns and
captions. No robot example, controller or model-specific dimensions live here.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _normalize(values: np.ndarray) -> np.ndarray:
    return np.asarray(values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12))


def _primitive(kind: int, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle positions and matching corner normals, in geom space."""
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.array(list(itertools.product((-1, 1), repeat=3))) * size
        faces = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
        indices = np.concatenate((faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]))
        triangles = vertices[indices]
        normals = _normalize(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]))
        return triangles, np.repeat(normals[:, None], 3, axis=1)
    angles = np.linspace(0, 2 * math.pi, 33)[:-1]
    ring = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        vertices = np.concatenate((ring * size[0] + [0, 0, -size[1]], ring * size[0] + [0, 0, size[1]]))
        faces = np.array(
            [[i, (i + 1) % 32, i + 32] for i in range(32)]
            + [[(i + 1) % 32, (i + 1) % 32 + 32, i + 32] for i in range(32)]
        )
        triangles, normals = vertices[faces], np.tile(ring, (2, 1))[faces]
        for sign in (-1, 1):
            cap = ring * size[0] + [0, 0, sign * size[1]]
            cap_triangles = np.array([[np.array([0, 0, sign * size[1]]), cap[i], cap[(i + 1) % 32]] for i in range(32)])
            triangles = np.concatenate((triangles, cap_triangles))
            normals = np.concatenate((normals, np.tile([0, 0, sign], (32, 3, 1))))
        return triangles, normals
    if kind not in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CAPSULE):
        raise ValueError(f"Unsupported visible geom type: {kind}")
    latitudes = np.linspace(-math.pi / 2, math.pi / 2, 21)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        latitudes = np.concatenate((latitudes[:11], latitudes[10:]))
    unit = np.array(
        [[math.cos(t) * math.cos(a), math.cos(t) * math.sin(a), math.sin(t)] for t in latitudes for a in angles]
    )
    radii = size if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else np.full(3, size[0])
    vertices, normals = unit * radii, _normalize(unit / radii)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        vertices[: 11 * 32, 2] -= size[1]
        vertices[11 * 32 :, 2] += size[1]
    surface_indices: list[tuple[int, int, int]] = []
    for row in range(len(latitudes) - 1):
        for col in range(32):
            a, b = row * 32 + col, row * 32 + (col + 1) % 32
            surface_indices.extend(((a, b, a + 32), (b, b + 32, a + 32)))
    return vertices[surface_indices], normals[surface_indices]


def geometry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    groups: tuple[int, ...] = (1, 2),
    hidden_pattern: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read visible geometry only; no geometry, dimensions or pose are invented."""
    triangles, normals, colors = [], [], []
    for geom in range(model.ngeom):
        if model.geom_group[geom] not in groups or (
            hidden_pattern and re.fullmatch(hidden_pattern, model.geom(geom).name)
        ):
            continue
        material = model.geom_matid[geom]
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
        if rgba[3] < 0.5:
            continue
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            na, nn = model.mesh_normaladr[mesh], model.mesh_normalnum[mesh]
            local = model.mesh_vert[va : va + vn][model.mesh_face[fa : fa + fn]]
            local_normals = model.mesh_normal[na : na + nn][model.mesh_facenormal[fa : fa + fn]]
        else:
            local, local_normals = _primitive(int(model.geom_type[geom]), model.geom_size[geom])
        rotation = data.geom_xmat[geom].reshape(3, 3)
        triangles.append(local @ rotation.T + data.geom_xpos[geom])
        normals.append(local_normals @ rotation.T)
        colors.append(np.tile(rgba[:3], (len(local), 1)))
    return np.concatenate(triangles), np.concatenate(normals), np.concatenate(colors)


def camera_basis(direction: tuple[float, float, float]) -> np.ndarray:
    eye = _normalize(np.array(direction))
    right = _normalize(np.cross([0, 0, 1], eye))
    return np.array([right, np.cross(eye, right), eye])


def render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    direction: tuple[float, float, float],
    groups: tuple[int, ...] = (1, 2),
    hidden_pattern: str | None = None,
    width: int = 1000,
    height: int = 1000,
    projection_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> Image.Image:
    """Orthographic triangle rasterization with interpolated normals and depth."""
    triangles, normals, colors = geometry(model, data, groups=groups, hidden_pattern=hidden_pattern)
    basis = camera_basis(direction)
    eye = basis[2]
    projected = triangles @ basis.T
    lo, hi = projection_bounds or (projected[:, :, :2].min(axis=(0, 1)), projected[:, :, :2].max(axis=(0, 1)))
    scale = min((width * 0.84) / (hi[0] - lo[0]), (height * 0.82) / (hi[1] - lo[1]))
    projected[:, :, :2] = (projected[:, :, :2] - (lo + hi) / 2) * [scale, -scale] + [width / 2, height / 2]
    shade = np.linspace(0, 1, height)[:, None, None]
    pixels = np.broadcast_to(np.array([245, 247, 244]) + shade * np.array([-9, -8, -7]), (height, width, 3)).copy()
    depth = np.full((height, width), -np.inf)
    key = _normalize(np.array([0.4, -0.5, 1.0]))
    fill = _normalize(np.array([-0.7, 0.3, 0.4]))
    for tri, normal, color in zip(projected, normals, colors, strict=True):
        lower = np.maximum(np.floor(tri[:, :2].min(axis=0)).astype(int), 0)
        upper = np.minimum(np.ceil(tri[:, :2].max(axis=0)).astype(int), [width - 1, height - 1])
        x0, y0 = lower
        x1, y1 = upper
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = tri
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-8:
            continue
        x, y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        w2 = 1 - w0 - w1
        z = w0 * a[2] + w1 * b[2] + w2 * c[2]
        region = depth[y0 : y1 + 1, x0 : x1 + 1]
        visible = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (z > region)
        if not visible.any():
            continue
        n = _normalize(w0[visible, None] * normal[0] + w1[visible, None] * normal[1] + w2[visible, None] * normal[2])
        illumination = 0.27 + 0.65 * np.maximum(n @ key, 0) + 0.08 * np.maximum(n @ fill, 0)
        specular = np.maximum(n @ _normalize(key + eye), 0) ** 45 * 0.10
        linear = np.clip(color * illumination[:, None] + specular[:, None], 0, 1)
        srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
        pixels[y0 : y1 + 1, x0 : x1 + 1][visible] = np.clip(srgb * 255, 0, 255)
        region[visible] = z[visible]
    return Image.fromarray(pixels.astype(np.uint8))


@lru_cache
def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    fonts = ["DejaVuSans.ttf"]
    matplotlib = find_spec("matplotlib")
    if matplotlib and matplotlib.origin:
        fonts.append(str(Path(matplotlib.origin).parent / "mpl-data/fonts/ttf/DejaVuSans.ttf"))
    for font in fonts:
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


@dataclass(frozen=True)
class GeometryRenderer:
    """A renderer with caller-owned named views and service-panel selection."""

    views: Mapping[str, tuple[float, float, float]]
    service_panels: str

    def camera_basis(self, view: str) -> np.ndarray:
        return camera_basis(self.views[view])

    def geometry(
        self, model: mujoco.MjModel, data: mujoco.MjData, *, service: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return geometry(
            model,
            data,
            groups=(1, 2, 4) if service else (1, 2),
            hidden_pattern=self.service_panels if service else None,
        )

    def render(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        view: str = "isometric",
        service: bool = False,
        width: int = 1000,
        height: int = 1000,
        projection_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Image.Image:
        return render(
            model,
            data,
            direction=self.views[view],
            groups=(1, 2, 4) if service else (1, 2),
            hidden_pattern=self.service_panels if service else None,
            width=width,
            height=height,
            projection_bounds=projection_bounds,
        )


def four_view_sheet(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    renderer: Callable[..., Image.Image],
    title: str,
    subtitle: str,
    panels: Sequence[tuple[str, bool, str]],
    footnotes: tuple[str, str],
) -> Image.Image:
    """Compose four geometry views with caller-provided review labels."""
    sheet = Image.new("RGB", (1600, 1580), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((55, 34), title, font=font(35), fill=(38, 55, 43))
    draw.text(
        (55, 88),
        subtitle,
        font=font(20),
        fill=(83, 97, 88),
    )
    for i, (view, service, label) in enumerate(panels):
        x, y = i % 2 * 800, 145 + i // 2 * 660
        sheet.paste(renderer(model, data, view=view, service=service, width=790, height=605), (x, y + 34))
        draw.text((x + 45, y), label, font=font(18), fill=(60, 76, 66))
    draw.text(
        (55, 1510),
        footnotes[0],
        font=font(18),
        fill=(83, 97, 88),
    )
    draw.text(
        (55, 1540),
        footnotes[1],
        font=font(18),
        fill=(83, 97, 88),
    )
    return sheet


def gesture_gif(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    path: Path,
    *,
    controls: Callable[[mujoco.MjModel, float], np.ndarray],
    renderer: Callable[..., Image.Image],
    direction: tuple[float, float, float],
    title: str,
    footer: Callable[[float], str],
    frames: int = 24,
    duration: float = 4.8,
) -> None:
    """Render one rollout at a fixed scale with caller-provided captions."""
    if frames < 2 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("At least two frames and positive finite duration are required")
    poses = []
    for timestamp in np.linspace(0, duration, frames, endpoint=False):
        while data.time < timestamp:
            data.ctrl[:] = controls(model, float(data.time))
            mujoco.mj_step(model, data)
        poses.append(data.qpos.copy())
    basis = camera_basis(direction)
    lows, highs = [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        points = geometry(model, data)[0] @ basis.T
        lows.append(points[:, :, :2].min(axis=(0, 1)))
        highs.append(points[:, :, :2].max(axis=(0, 1)))
    bounds = (np.min(lows, axis=0), np.max(highs, axis=0))
    images = []
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        frame = renderer(model, data, width=560, height=570, projection_bounds=bounds)
        draw = ImageDraw.Draw(frame)
        draw.text((18, 12), title, font=font(17), fill=(38, 55, 43))
        draw.text(
            (18, 540),
            footer(i * duration / frames),
            font=font(14),
            fill=(60, 76, 66),
        )
        # A small nondithered palette keeps the actual-geometry animation below
        # the repository's 500 KiB file limit without changing its poses/timing.
        images.append(frame.quantize(colors=48, dither=Image.Dither.NONE))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], loop=0, duration=round(1000 * duration / frames), optimize=True
    )
