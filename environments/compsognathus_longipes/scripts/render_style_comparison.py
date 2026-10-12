"""Render the robot before and after a styling revision at equal camera scale.

Pass a prior robot MJCF with --before; its mesh files are resolved against the
current robot's unchanged pinned mechanical assets. Needs Pillow and OSMesa/EGL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from environments.compsognathus.model import load_model
from environments.compsognathus.scripts.validate_shells import read_reference


def render(before, output):
    canvas = Image.new("RGB", (1440, 1050), "#16212b")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype("DejaVuSans.ttf", 29)
    label = ImageFont.truetype("DejaVuSans.ttf", 25)
    small = ImageFont.truetype("DejaVuSans.ttf", 18)
    draw.text((28, 20), "COMPSOGNATHUS  /  STYLING REVISION", font=title, fill="#f2efe8")
    draw.text((28, 65), "Actual MuJoCo renders • identical standing pose and camera scale", font=small, fill="#bac9cf")
    cases = [
        ("BEFORE", "12 cm torso • original grey leg covers", read_reference(before)),
        ("UPDATED", "15.5 cm tapered torso • coordinated silver / olive / graphite", load_model("robot")),
    ]
    for column, (name, caption, (model, data)) in enumerate(cases):
        draw.text((column * 720 + 24, 110), name, font=label, fill="#dfbc77")
        draw.text((column * 720 + 24, 150), caption, font=small, fill="#d4e0e3")
        with mujoco.Renderer(model, height=380, width=704) as renderer:
            for row, (azimuth, elevation) in enumerate(((90, 0), (125, -18))):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [-0.060, 0, 0.175]
                camera.distance = 0.67
                camera.azimuth, camera.elevation = azimuth, elevation
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (column * 720 + 8, 188 + row * 386))
    draw.text(
        (28, 971),
        "Original leg geometry and soles retained • 12 leg actuators • fixed head and unpowered tail",
        font=small,
        fill="#d4e0e3",
    )
    draw.text(
        (28, 1008),
        "1.586 kg modeled allocation. Shell fabrication, internal packaging and walking remain to be validated.",
        font=small,
        fill="#bac9cf",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.before, args.output)


if __name__ == "__main__":
    main()
