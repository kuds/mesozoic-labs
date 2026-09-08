"""Desktop viewer or actual MuJoCo renders for the two experimental plants.

Examples:
  python -m environments.compsognathus.scripts.view_model --model robot
  MUJOCO_GL=osmesa python -m environments.compsognathus.scripts.view_model --compare /tmp/compso.png
  MUJOCO_GL=osmesa python -m environments.compsognathus.scripts.view_model --model robot --video /tmp/robot.mp4

Render outputs need Pillow; videos additionally need imageio[ffmpeg]. A headless
host needs an EGL or OSMesa runtime. Interactive viewing uses the platform GL
window system (on macOS use mjpython instead of python).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.model import load_model, model_bounds, robot_body_ids, set_motors_enabled


def camera_for(model, data, *, side=False, shared=False):
    low, high = model_bounds(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (low + high) / 2
    camera.distance = 0.9 if shared else max(0.65, float(np.max(high - low)) * 1.45)
    camera.azimuth = 90 if side else 125
    camera.elevation = 0 if side else -18
    return camera


def comparison(path):
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1440, 1040), "#131d26")
    draw = ImageDraw.Draw(canvas)
    try:
        title = ImageFont.truetype("DejaVuSans.ttf", 30)
        body = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        title = body = ImageFont.load_default()
    draw.text((32, 18), "MESOZOIC LABS  /  COMPSOGNATHUS", font=title, fill="#f3f0e8")
    draw.text((32, 62), "Initial MuJoCo models • authored home poses • equal camera scale", font=body, fill="#bac9cf")
    for column, variant in enumerate(MODEL_PATHS):
        model, data = load_model(variant)
        low, high = model_bounds(model, data)
        mass = model.body_mass[robot_body_ids(model)].sum()
        with mujoco.Renderer(model, height=380, width=704) as renderer:
            for row, side in enumerate((True, False)):
                renderer.update_scene(data, camera=camera_for(model, data, side=side, shared=True))
                canvas.paste(Image.fromarray(renderer.render()), (8 + column * 720, 170 + row * 390))
        x = 24 + column * 720
        draw.text(
            (x, 106), "ANATOMICAL PROXY" if variant == "biological" else "REV B ROBOT", font=title, fill="#efc06b"
        )
        draw.text(
            (x, 145),
            f"{100 * (high - low)[0]:.1f} cm long  ·  {100 * high[2]:.1f} cm tall  ·  {mass:.3f} kg  ·  {model.nu} actuators",
            font=body,
            fill="#d4e0e3",
        )
    draw.text(
        (32, 964),
        "Biological: longer tail, digitigrade legs.  Robot: original leg geometry, broad soles, fixed unmotorized tail.",
        font=body,
        fill="#d4e0e3",
    )
    draw.text(
        (32, 999),
        "Standing smoke tests passed. Walking, manufactured fit, motor identification and hardware transfer remain open.",
        font=body,
        fill="#bac9cf",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_PATHS, default="robot")
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--motors-off", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be finite and positive")
    if args.compare:
        comparison(args.compare)
        return
    model, data = load_model(args.model, args.mass_scale)
    if args.motors_off:
        set_motors_enabled(model, False)
    camera = camera_for(model, data)
    if args.snapshot or args.video:
        with mujoco.Renderer(model, height=720, width=1080) as renderer:
            if args.snapshot:
                from PIL import Image

                renderer.update_scene(data, camera=camera)
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(renderer.render()).save(args.snapshot)
            if args.video:
                import imageio.v2 as imageio

                args.video.parent.mkdir(parents=True, exist_ok=True)
                with imageio.get_writer(args.video, fps=30, codec="libx264", macro_block_size=1) as writer:
                    for frame in range(round(args.seconds * 30)):
                        target = (frame + 1) / 30
                        while data.time < target - 1e-10:
                            mujoco.mj_step(model, data)
                        renderer.update_scene(data, camera=camera)
                        writer.append_data(renderer.render())
        return
    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        while viewer.is_running():
            started = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
