"""Render the actual compiled 50 cm MJCF and a free-base standing gesture."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from examples.mini_tyrannosaurus_rex.render_design import _camera_basis, _font, geometry
from examples.mini_tyrannosaurus_rex.render_design import render as render_geometry

from .demo import MODEL_PATH, gesture_controls, load_model
from .validate import visual_bounds


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
    """Hide panel-mounted details with the removed service panels."""
    groups = model.geom_group.copy()
    try:
        if service:
            for g in range(model.ngeom):
                if model.geom(g).name.startswith(("back_vent_", "back_screw_")):
                    model.geom_group[g] = 5
        return render_geometry(
            model, data, view=view, service=service, width=width, height=height, projection_bounds=projection_bounds
        )
    finally:
        model.geom_group[:] = groups


def design_sheet(path: Path) -> None:
    model, data = load_model()
    low, high = visual_bounds(model, data)
    sheet = Image.new("RGB", (1600, 1580), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((55, 34), "T-REX / SILHOUETTE C", font=_font(35), fill=(38, 55, 43))
    draw.text(
        (55, 88),
        f"Compiled MuJoCo geometry  |  {(high[2] - low[2]) * 100:.1f} cm  |  {model.body_mass.sum():.3f} kg assigned mass",
        font=_font(20),
        fill=(83, 97, 88),
    )
    for i, (view, service, label) in enumerate(
        (
            ("isometric", False, "CHARACTER"),
            ("side", False, "SIDE / MECHANICAL LAYOUT"),
            ("front", False, "FRONT / CAMERAS IN SNOUT"),
            ("isometric", True, "SERVICE VIEW / PROVISIONAL ENVELOPES"),
        )
    ):
        x, y = i % 2 * 800, 145 + i // 2 * 660
        sheet.paste(render(model, data, view=view, service=service, width=790, height=605), (x, y + 34))
        draw.text((x + 45, y), label, font=_font(18), fill=(60, 76, 66))
    draw.text(
        (55, 1510),
        "Visual shells + separate collision proxies. Motor envelopes and inertias are provisional.",
        font=_font(18),
        fill=(83, 97, 88),
    )
    draw.text(
        (55, 1540),
        "Standing checks are not walking validation or fabrication approval.",
        font=_font(18),
        fill=(83, 97, 88),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def gesture_gif(path: Path, *, frames: int = 24, duration: float = 4.8) -> None:
    if frames < 2 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("At least two frames and positive finite duration are required")
    model, data = load_model()
    poses = []
    for timestamp in np.linspace(0, duration, frames, endpoint=False):
        while data.time < timestamp:
            data.ctrl[:] = gesture_controls(model, float(data.time))
            mujoco.mj_step(model, data)
        poses.append(data.qpos.copy())
    basis = _camera_basis("isometric")
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
        frame = render(model, data, width=560, height=570, projection_bounds=bounds)
        draw = ImageDraw.Draw(frame)
        draw.text((18, 12), "FREE-BASE STANDING GESTURE", font=_font(17), fill=(38, 55, 43))
        draw.text(
            (18, 540),
            f"No torso support  /  not walking  /  {i * duration / frames:.1f} s",
            font=_font(14),
            fill=(60, 76, 66),
        )
        images.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], loop=0, duration=round(1000 * duration / frames), optimize=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=MODEL_PATH.with_name("design_views.png"))
    parser.add_argument("--gif", type=Path)
    args = parser.parse_args()
    design_sheet(args.sheet)
    if args.gif:
        gesture_gif(args.gif)
