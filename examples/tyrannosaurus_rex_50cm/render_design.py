"""Render the actual compiled 50 cm MJCF and a free-base standing gesture."""

from __future__ import annotations

import argparse
from pathlib import Path

from environments.shared import robot_rendering
from environments.shared.robot_geometry import visual_bounds

from .demo import MODEL_PATH, gesture_controls, load_model

SERVICE_PANELS = r"torso_shell|belly_panel|back_access_panel|.*torso_cover.*|back_vent_.*|back_screw_.*"
VIEWS = {"isometric": (1.0, -1.35, 0.65), "side": (0.0, -1.0, 0.0), "front": (1.0, 0.0, 0.0)}


_renderer = robot_rendering.GeometryRenderer(views=VIEWS, service_panels=SERVICE_PANELS)
camera_basis = _renderer.camera_basis
geometry = _renderer.geometry
render = _renderer.render


def design_sheet(path: Path) -> None:
    model, data = load_model()
    low, high = visual_bounds(model, data)
    sheet = robot_rendering.four_view_sheet(
        model,
        data,
        renderer=render,
        title="T-REX / SILHOUETTE C",
        subtitle=f"Compiled MuJoCo geometry  |  {(high[2] - low[2]) * 100:.1f} cm  |  {model.body_mass.sum():.3f} kg assigned mass",
        panels=(
            ("isometric", False, "CHARACTER"),
            ("side", False, "SIDE / MECHANICAL LAYOUT"),
            ("front", False, "FRONT / CAMERAS IN SNOUT"),
            ("isometric", True, "SERVICE VIEW / PROVISIONAL ENVELOPES"),
        ),
        footnotes=(
            "Visual shells + separate collision proxies. Motor envelopes and inertias are provisional.",
            "Standing checks are not walking validation or fabrication approval.",
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def gesture_gif(path: Path, *, frames: int = 24, duration: float = 4.8) -> None:
    model, data = load_model()
    robot_rendering.gesture_gif(
        model,
        data,
        path,
        controls=gesture_controls,
        renderer=render,
        direction=VIEWS["isometric"],
        title="FREE-BASE STANDING GESTURE",
        footer=lambda elapsed: f"No torso support  /  not walking  /  {elapsed:.1f} s",
        frames=frames,
        duration=duration,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=MODEL_PATH.with_name("design_views.png"))
    parser.add_argument("--gif", type=Path)
    args = parser.parse_args()
    design_sheet(args.sheet)
    if args.gif:
        gesture_gif(args.gif)
