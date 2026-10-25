"""Render the exact MJCF geometry on a CPU, without an OpenGL connection.

Uses MuJoCo's compiled mesh vertices, normals and geom transforms. Primitive
surfaces are tessellated; collision proxies are hidden. The service view removes
named torso panels for inspection, not to claim a fabricated removable assembly.
Pillow (included by the ``viz`` extra's matplotlib) supplies PNG and typography.
"""

from __future__ import annotations

import argparse
import math
from importlib import import_module
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from environments.shared import robot_rendering
from environments.shared.robot_rendering import font

MODEL_PATH = Path(__file__).with_name("model.xml")
SERVICE_PANELS = r"torso_shell|belly_panel|back_access_panel|.*torso_cover.*"
VIEWS = {"isometric": (1.0, -1.35, 0.65), "side": (0.0, -1.0, 0.0), "front": (1.0, 0.0, 0.0)}


_renderer = robot_rendering.GeometryRenderer(views=VIEWS, service_panels=SERVICE_PANELS)
camera_basis = _renderer.camera_basis
geometry = _renderer.geometry
render = _renderer.render


def design_sheet(model: mujoco.MjModel, data: mujoco.MjData) -> Image.Image:
    sheet = Image.new("RGB", (1600, 1800), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    tail = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tail_yaw_servo") >= 0
    selected_c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "mini_c_skull") >= 0
    title = "MINI T-REX / SILHOUETTE C" if selected_c else "MINI TYRANNOSAURUS REX"
    title += " / TAIL OPTION" if tail else ""
    draw.text((62, 42), title, font=font(38), fill=(35, 46, 42))
    draw.text(
        (62, 98),
        "MJCF design study  /  exact model geometry  /  not fabrication CAD",
        font=font(21),
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
        draw.text((x + 62, y + 5), label, font=font(22), fill=(60, 76, 66))
    draw.text(
        (62, 1745),
        "Hidden panels expose schematic components; mounting, wiring and thermal clearances require CAD verification.",
        font=font(17),
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
    basis = camera_basis("isometric")
    clouds = [geometry(m, d)[0] for m, d in models]
    projected = np.concatenate(clouds) @ basis.T
    bounds = (projected[:, :, :2].min(axis=(0, 1)), projected[:, :, :2].max(axis=(0, 1)))
    sheet = Image.new("RGB", (1600, 1180), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((60, 32), "ONE SPECIES / TWO SILHOUETTE-C ROBOTS", font=font(35), fill=(35, 46, 42))
    draw.text(
        (60, 87),
        "Actual compiled models at the same scale • unweighed design concepts",
        font=font(22),
        fill=(83, 97, 88),
    )
    for index, ((m, d), points) in enumerate(zip(models, clouds, strict=True)):
        sheet.paste(render(m, d, width=800, height=900, projection_bounds=bounds), (index * 800, 200))
        height = float(np.ptp(points[:, :, 2]))
        label = "MINI" if index == 0 else "MEDIUM"
        draw.text(
            (60 + index * 800, 145),
            f"{label}  /  {height * 100:.1f} cm  /  {m.body_mass.sum():.3f} kg",
            font=font(24),
            fill=(60, 76, 66),
        )
    draw.text(
        (60, 1120),
        "Shared character; independently sized hardware. Standing/gesture checks do not establish walking.",
        font=font(20),
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
    basis = camera_basis("isometric")
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
        draw.text((24, 610), "Supported gesture study • not walking", font=font(18), fill=(50, 66, 56))
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
