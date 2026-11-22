"""Museum-contact stills (task 9): the standing robot and the anatomical adult, each labelled as simulation.

Renders MuJoCo scenes offscreen (MUJOCO_GL=osmesa on a headless machine), adds a caption band that says
"SIMULATION", and writes JPEGs under the repository's 500 KB limit.

Usage (75 cm, 18 Nov 2026):
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py \
      --robot ../sim/generated/trike_T20_closed.xml --anatomical PATH/TO/anatomical.xml --out stills_2026-11-18
Usage (60 cm; robot only, the adult stills are unchanged):
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py --robot ../sim/generated/trike60_C_closed.xml \
      --size 60 --scale 0.8 --out stills_60cm_<date>
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


def top_height(model, data):
    """Highest point of the visible geoms (the frill's top on these models), m, from mesh vertices or box corners."""
    best = -1.0
    for g in range(model.ngeom):
        if model.geom(g).name == "floor" or model.geom_rgba[g, 3] <= 0:
            continue
        R, p = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[a:a + n] @ R.T + p
        else:
            c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            v = (np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h + c) @ R.T + p
        best = max(best, float(v[:, 2].max()))
    return best


def settle(model, data, seconds):
    mujoco.mj_resetDataKeyframe(model, data, 0) if model.nkey else mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data, nstep=round(seconds / model.opt.timestep))
    mujoco.mj_forward(model, data)


def shot(model, data, lookat, distance, azimuth, elevation, reflection=True):
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    r = mujoco.Renderer(model, H - BAND, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    opt = mujoco.MjvOption()
    r.update_scene(data, camera=cam, scene_option=opt)
    if not reflection:
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    img = r.render()
    r.close()
    return Image.fromarray(img)


def label(img, title, caption, second_badge=False):
    out = Image.new("RGB", (W, H), (24, 28, 32))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    fb, f, fs = ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT_B, 22)
    # badge on the image itself, so a crop of the picture keeps the label
    badge = "SIMULATION"
    bw = d.textlength(badge, font=fs) + 28
    d.rounded_rectangle([20, 20, 20 + bw, 58], radius=8, fill=(200, 40, 40))
    d.text((34, 26), badge, font=fs, fill=(255, 255, 255))
    if second_badge:  # a second badge, bottom right of the picture, so a crop around the robot keeps one
        x1, y1 = W - 20 - bw, H - BAND - 58
        d.rounded_rectangle([x1, y1, x1 + bw, y1 + 38], radius=8, fill=(200, 40, 40))
        d.text((x1 + 14, y1 + 6), badge, font=fs, fill=(255, 255, 255))
    y0 = H - BAND
    for text, font in ((title, fb), (caption, f)):  # one line each: refuse rather than clip at the right edge
        if d.textlength(text, font=font) > W - 48:
            raise SystemExit(f"caption line too wide for the {W} px picture; nothing written: {text!r}")
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
    ap.add_argument("--anatomical", help="also render the adult reconstruction (PR #669's models/anatomical.xml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=75, help="the robot's size to the frill, cm, for the captions")
    ap.add_argument("--scale", type=float, default=1.0, help="camera targets and distances x this (0.8 for 60 cm)")
    ap.add_argument("--no-reflection", action="store_true", help="turn off the floor's reflection")
    ap.add_argument("--second-badge", action="store_true", help="a second SIMULATION badge, bottom right")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    m = mujoco.MjModel.from_xml_path(a.robot)
    d = mujoco.MjData(m)
    settle(m, d, 2.0)
    mass = m.body_mass.sum()
    top = top_height(m, d)
    if abs(100 * top - a.size) > 2.0:
        raise SystemExit(f"--size {a.size} cm, but the model's frill top stands at {100 * top:.1f} cm; nothing written")
    kg = f"{mass:.0f}" if abs(mass - round(mass)) < 0.05 else f"{mass:.1f}"
    feet = " with passive roll ankles (a design variant)" if "ankleroll" in Path(a.robot).name else ""
    note = (f"Simulation: MuJoCo model of the proposed {a.size} cm Triceratops robot, {kg} kg design{feet}, standing. "
            "Not hardware; nothing is built yet.")
    if "capsule" in Path(a.robot).name:  # the foot chosen on 22 Nov 2026 (D19); worded to fit the caption band
        note = (f"Simulation: MuJoCo model of the proposed {a.size} cm Triceratops robot ({kg} kg, capsule feet), "
                "standing. Not hardware; nothing is built yet.")
    s = a.scale
    for name, lookat, dist, az, el in (
            ("robot_three_quarter", (0.05, 0, 0.36), 1.75, 140, -13),
            ("robot_side", (0.0, 0, 0.36), 1.95, 90, -4),
            ("robot_head", (0.40, 0, 0.47), 0.85, 150, -8)):
        img = label(shot(m, d, tuple(s * v for v in lookat), s * dist, az, el, reflection=not a.no_reflection),
                    f"Triceratops robot, {a.size} cm to the frill: simulation", note, second_badge=a.second_badge)
        q, size = save(img, out / f"{name}.jpg")
        rows.append((name, q, size))

    if not a.anatomical:
        for name, q, size in rows:
            print(f"{name}.jpg  JPEG quality {q}  {size / 1024:.0f} KB")
        return
    m = mujoco.MjModel.from_xml_path(a.anatomical)
    d = mujoco.MjData(m)
    settle(m, d, 1.0)
    note = ("Simulation: MuJoCo reconstruction of an adult Triceratops horridus, 7.3 m long, about 3.2 m at the frill. "
            "Estimated proportions and masses.")
    for name, lookat, dist, az, el in (
            ("adult_three_quarter", (0.2, 0, 1.6), 8.6, 140, -11),
            ("adult_side", (0.0, 0, 1.6), 9.2, 90, -3)):
        img = label(shot(m, d, lookat, dist, az, el), "Adult Triceratops, the animal the robot is modelled on: simulation",
                    note)
        q, size = save(img, out / f"{name}.jpg")
        rows.append((name, q, size))
    for name, q, size in rows:
        print(f"{name}.jpg  JPEG quality {q}  {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
