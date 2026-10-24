"""Render the exact MJCF geometry on a CPU, without an OpenGL connection.

Uses MuJoCo's compiled mesh vertices, normals and geom transforms. Primitive
surfaces are tessellated; collision proxies are hidden. The service view removes
named torso panels for inspection, not to claim a fabricated removable assembly.
Pillow (included by the ``viz`` extra's matplotlib) supplies PNG and typography.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).with_name("model.xml")
SERVICE_PANELS = r"torso_shell|belly_panel|back_access_panel|.*torso_cover.*"
VIEWS = {"isometric": (1.0, -1.35, 0.65), "side": (0.0, -1.0, 0.0), "front": (1.0, 0.0, 0.0)}


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
    model: mujoco.MjModel, data: mujoco.MjData, *, service: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read visible geometry only; no geometry, dimensions or pose are invented."""
    triangles, normals, colors = [], [], []
    for geom in range(model.ngeom):
        groups = (1, 2, 4) if service else (1, 2)
        if model.geom_group[geom] not in groups or (service and re.fullmatch(SERVICE_PANELS, model.geom(geom).name)):
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


def _camera_basis(view: str) -> np.ndarray:
    eye = _normalize(np.array(VIEWS[view]))
    right = _normalize(np.cross([0, 0, 1], eye))
    return np.array([right, np.cross(eye, right), eye])


def render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    view: str = "isometric",
    service: bool = False,
    width: int = 1000,
    height: int = 1000,
    projection_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> Image.Image:
    """Orthographic triangle rasterization with interpolated normals and depth."""
    triangles, normals, colors = geometry(model, data, service=service)
    basis = _camera_basis(view)
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
    return Image.fromarray(np.uint8(pixels))


@lru_cache
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def design_sheet(model: mujoco.MjModel, data: mujoco.MjData) -> Image.Image:
    sheet = Image.new("RGB", (1600, 1800), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    tail = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tail_yaw_servo") >= 0
    selected_c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "mini_c_skull") >= 0
    title = "MINI T-REX / SILHOUETTE C" if selected_c else "MINI TYRANNOSAURUS REX"
    title += " / TAIL OPTION" if tail else ""
    draw.text((62, 42), title, font=_font(38), fill=(35, 46, 42))
    draw.text(
        (62, 98),
        "MJCF design study  /  exact model geometry  /  not fabrication CAD",
        font=_font(21),
        fill=(83, 97, 88),
    )
    views = [
        ("isometric", False, "CHARACTER"),
        ("side", False, "SIDE"),
        ("front", False, "FRONT"),
        ("isometric", True, "COMPONENT STUDY — PANELS HIDDEN"),
    ]
    for index, (view, service, label) in enumerate(views):
        x, y = (index % 2) * 800, 160 + (index // 2) * 780
        sheet.paste(render(model, data, view=view, service=service, width=800, height=710), (x, y + 40))
        draw.text((x + 62, y + 5), label, font=_font(22), fill=(60, 76, 66))
    draw.text(
        (62, 1745),
        "Hidden panels expose schematic components; mounting, wiring and thermal clearances require CAD verification.",
        font=_font(17),
        fill=(83, 97, 88),
    )
    return sheet


def family_sheet(model: mujoco.MjModel, data: mujoco.MjData) -> Image.Image:
    """Show both current defaults at the same orthographic scale and orientation."""
    medium_path = MODEL_PATH.parent.parent / "tyrannosaurus_rex_50cm" / "model.xml"
    medium = mujoco.MjModel.from_xml_path(str(medium_path))
    medium_data = mujoco.MjData(medium)
    mujoco.mj_resetDataKeyframe(medium, medium_data, 0)
    mujoco.mj_forward(medium, medium_data)
    models = [(model, data), (medium, medium_data)]
    basis = _camera_basis("isometric")
    clouds = [geometry(m, d)[0] for m, d in models]
    projected = np.concatenate(clouds) @ basis.T
    bounds = (projected[:, :, :2].min(axis=(0, 1)), projected[:, :, :2].max(axis=(0, 1)))
    sheet = Image.new("RGB", (1600, 1180), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((60, 32), "ONE SPECIES / TWO SILHOUETTE-C ROBOTS", font=_font(35), fill=(35, 46, 42))
    draw.text(
        (60, 87),
        "Actual compiled models at the same scale • unweighed design concepts",
        font=_font(22),
        fill=(83, 97, 88),
    )
    for index, ((m, d), points) in enumerate(zip(models, clouds, strict=True)):
        sheet.paste(render(m, d, width=800, height=900, projection_bounds=bounds), (index * 800, 200))
        height = float(np.ptp(points[:, :, 2]))
        label = "MINI" if index == 0 else "MEDIUM"
        draw.text(
            (60 + index * 800, 145),
            f"{label}  /  {height * 100:.1f} cm  /  {m.body_mass.sum():.3f} kg",
            font=_font(24),
            fill=(60, 76, 66),
        )
    draw.text(
        (60, 1120),
        "Shared character; independently sized hardware. Standing/gesture checks do not establish walking.",
        font=_font(20),
        fill=(83, 97, 88),
    )
    return sheet


def gesture_gif(model: mujoco.MjModel, path: Path, *, frames: int = 20, duration: float = 4.8) -> None:
    """Render snapshots from one supported physics rollout at a fixed scale."""
    demo = import_module(f"{__package__}.demo" if __package__ else "demo")

    if frames < 2 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("A gesture needs at least two frames and a positive finite duration.")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    demo.set_support(model, data, True)
    poses = []
    for target in np.linspace(0, duration, frames, endpoint=False):
        while data.time + model.opt.timestep / 2 < target:
            data.ctrl[:] = demo.gesture_controls(model, data.time)
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
            raise RuntimeError("Supported rollout produced a non-finite state or simulator warning.")
        poses.append(data.qpos.copy())
    basis = _camera_basis("isometric")
    bounds = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        projected = geometry(model, data)[0] @ basis.T
        bounds.append((projected[:, :, :2].min(axis=(0, 1)), projected[:, :, :2].max(axis=(0, 1))))
    fixed_bounds = (np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0))
    images = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        frame = render(model, data, width=640, height=640, projection_bounds=fixed_bounds)
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 602, 640, 640), fill=(245, 247, 244))
        draw.text((24, 610), "Supported gesture study • not walking", font=_font(18), fill=(50, 66, 56))
        images.append(frame)
    # Bound the animation below the repository's 500 KiB artifact limit.
    palette_images = [image.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for image in images]
    palette_images[0].save(
        path,
        save_all=True,
        append_images=palette_images[1:],
        duration=round(1000 * duration / frames),
        loop=0,
        optimize=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet", action="store_true", help="Also render a labelled four-view design sheet.")
    parser.add_argument("--family", action="store_true", help="Also compare the mini and medium defaults at one scale.")
    parser.add_argument(
        "--tail", action="store_true", help="Render the optional tail-yaw study with _tail output suffixes."
    )
    parser.add_argument(
        "--gesture", action="store_true", help="Also render a supported 4.8-second gesture GIF; not walking."
    )
    args = parser.parse_args()
    if args.tail:
        tail_option = import_module(f"{__package__}.tail_option" if __package__ else "tail_option")
        model = tail_option.load_model(args.model.resolve())
    else:
        model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_tail" if args.tail else ""
    render(model, data).save(args.output_dir / f"preview{suffix}.png")
    if args.sheet:
        design_sheet(model, data).save(args.output_dir / f"design_views{suffix}.png")
    if args.gesture:
        gesture_gif(model, args.output_dir / f"supported_gesture{suffix}.gif")
    if args.family:
        family_sheet(model, data).save(args.output_dir / f"family_comparison{suffix}.png")
    print(f"Rendered model geometry to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
