"""Museum-contact stills (task 9): the standing robot and the anatomical adult, each labelled as simulation.

Renders MuJoCo scenes offscreen (MUJOCO_GL=osmesa on a headless machine), adds a caption band that says
"SIMULATION", and writes JPEGs under the repository's 500 KB limit.

Usage:
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py \
      --robot ../sim/generated/trike_T20_closed.xml --anatomical PATH/TO/anatomical.xml --out stills_2026-11-18
The anatomical model is PR #669's models/anatomical.xml (on main once PR #669 merges).
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H = 1600, 900
BAND = 92
MAX_BYTES = 500 * 1024


def settle(model, data, seconds):
    mujoco.mj_resetDataKeyframe(model, data, 0) if model.nkey else mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data, nstep=round(seconds / model.opt.timestep))
    mujoco.mj_forward(model, data)


def shot(model, data, lookat, distance, azimuth, elevation):
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    r = mujoco.Renderer(model, H - BAND, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    opt = mujoco.MjvOption()
    r.update_scene(data, camera=cam, scene_option=opt)
    img = r.render()
    r.close()
    return Image.fromarray(img)


def label(img, title, caption):
    out = Image.new("RGB", (W, H), (24, 28, 32))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    fb, f, fs = ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT_B, 22)
    # badge on the image itself, so a crop of the picture keeps the label
    badge = "SIMULATION"
    bw = d.textlength(badge, font=fs) + 28
    d.rounded_rectangle([20, 20, 20 + bw, 58], radius=8, fill=(200, 40, 40))
    d.text((34, 26), badge, font=fs, fill=(255, 255, 255))
    y0 = H - BAND
    d.text((24, y0 + 12), title, font=fb, fill=(240, 240, 240))
    d.text((24, y0 + 54), caption, font=f, fill=(190, 196, 204))
    return out


def save(img, path):
    for q in (90, 85, 80, 75, 70):
        img.save(path, "JPEG", quality=q, optimize=True, progressive=True)
        if path.stat().st_size <= MAX_BYTES:
            return q, path.stat().st_size
    raise RuntimeError(f"{path} stays over 500 KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--anatomical", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    m = mujoco.MjModel.from_xml_path(a.robot)
    d = mujoco.MjData(m)
    settle(m, d, 2.0)
    mass = m.body_mass.sum()
    note = (f"Simulation: MuJoCo model of the proposed 75 cm Triceratops robot, {mass:.0f} kg design, standing. "
            "Not hardware; nothing is built yet.")
    for name, lookat, dist, az, el in (
            ("robot_three_quarter", (0.05, 0, 0.36), 2.3, 140, -14),
            ("robot_side", (0.05, 0, 0.36), 2.4, 90, -4),
            ("robot_head", (0.38, 0, 0.48), 1.05, 150, -10)):
        img = label(shot(m, d, lookat, dist, az, el), "Triceratops robot, 75 cm to the frill: simulation", note)
        q, size = save(img, out / f"{name}.jpg")
        rows.append((name, q, size))

    m = mujoco.MjModel.from_xml_path(a.anatomical)
    d = mujoco.MjData(m)
    settle(m, d, 1.0)
    note = ("Simulation: MuJoCo reconstruction of an adult Triceratops horridus, 7.3 m long, about 3.2 m at the frill. "
            "Estimated proportions and masses.")
    for name, lookat, dist, az, el in (
            ("adult_three_quarter", (0.2, 0, 1.7), 11.0, 140, -12),
            ("adult_side", (0.2, 0, 1.7), 11.5, 90, -3)):
        img = label(shot(m, d, lookat, dist, az, el), "Adult Triceratops, the animal the robot is modelled on: simulation",
                    note)
        q, size = save(img, out / f"{name}.jpg")
        rows.append((name, q, size))
    for name, q, size in rows:
        print(f"{name}.jpg  JPEG quality {q}  {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
