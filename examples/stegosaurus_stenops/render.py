"""Render the actual reference model and a time-stamped, contact-labelled walk."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from .build import MODEL_PATH
from .validate import FEET, LOAD_FRACTION, FloorProbe
from .walk import WalkController


def camera(target, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _font(size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _view(model, data, width, height, cam, options):
    from PIL import Image

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=options)
        return Image.fromarray(renderer.render())


def render_study(path, model, data, options):
    from PIL import Image, ImageDraw

    width = 1200
    paper = "#f6f5ee"
    ink = "#25352d"
    sheet = Image.new("RGB", (width, 886), paper)
    draw = ImageDraw.Draw(sheet)
    draw.text((34, 16), "Stegosaurus stenops", font=_font(36), fill=ink)
    draw.text((35, 65), "Anatomical reference    /    NHMUK PV R36730 scale", font=_font(20), fill=ink)
    sheet.paste(_view(model, data, width, 490, camera((-1.16, 0, 1.22), 4.12, 77, -9), options), (0, 110))
    views = (
        ("Skull and beak", data.site("snout").xpos + [-0.17, 0, 0.07], 0.71, 108, -10),
        ("Forefoot   /   manus", data.body("rf_foot").xpos + [0.03, 0, -0.12], 0.77, 137, -18),
        ("Hindfoot   /   pes", data.body("rh_foot").xpos + [0.04, 0, -0.13], 0.77, 137, -18),
    )
    for column, (label, target, distance, azimuth, elevation) in enumerate(views):
        x = column * 401
        sheet.paste(_view(model, data, 398, 223, camera(target, distance, azimuth, elevation), options), (x, 604))
        draw.text((x + 18, 840), label, font=_font(22), fill=ink)
    sheet.save(path)


def render_walk(path, model, options, seconds=10):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    fps, width, height = 25, 1080, 510
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measured = mujoco.MjData(model)
    ctrl = WalkController(model)
    probe = FloorProbe(model)
    font = _font(19)
    history = []
    with (
        mujoco.Renderer(model, height=height, width=width) as renderer,
        imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=2, quality=8) as writer,
    ):
        for frame in range(int(fps * seconds)):
            while data.time < frame / fps - 0.5 * model.opt.timestep:
                ctrl.step(data, walking=True)
                mujoco.mj_step(model, data)
            mujoco.mj_copyData(measured, model, data)
            mujoco.mj_forward(model, measured)
            sample = probe.sample(measured)
            cam = camera((0.84 * sample.root[0] - 1.03, 0, 1.19), 4.25, 77, -8.5)
            renderer.update_scene(measured, camera=cam, scene_option=options)
            sheet = Image.new("RGB", (width, height + 164), "#f6f5ee")
            sheet.paste(Image.fromarray(renderer.render()), (0, 0))
            draw = ImageDraw.Draw(sheet)
            draw.text((24, height + 9), "Loaded feet    /    previous 3 seconds", font=font, fill="#25352d")
            draw.text((width - 136, height + 9), f"{data.time:5.2f} s", font=font, fill="#25352d")
            history.append(sample.load > LOAD_FRACTION)
            window = fps * 3
            age = np.array(history[-window:])
            for row, (code, name) in enumerate(
                (("lh", "Left hind"), ("lf", "Left fore"), ("rh", "Right hind"), ("rf", "Right fore"))
            ):
                y = height + 46 + row * 28
                draw.text((24, y - 5), name, font=font, fill="#34443a")
                left, right = 167, width - 35
                draw.rectangle((left, y, right, y + 12), fill="#d9ddd3")
                flags = age[:, FEET.index(code)]
                transitions = np.flatnonzero(np.diff(np.r_[False, flags, False]))
                for start, end in transitions.reshape(-1, 2):
                    a = right - (len(age) - start) * (right - left) / window
                    b = right - (len(age) - end) * (right - left) / window
                    draw.rectangle((a, y, b, y + 12), fill="#456954" if code.endswith("h") else "#976842")
            writer.append_data(np.asarray(sheet))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="save the neutral-pose anatomical study")
    parser.add_argument("--video", type=Path, help="save an MP4 of the physics rollout")
    args = parser.parse_args()
    if not (args.image or args.video):
        parser.error("provide --image or --video")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    if args.image:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        render_study(args.image, model, data, options)
    if args.video:
        render_walk(args.video, model, options)


if __name__ == "__main__":
    main()
