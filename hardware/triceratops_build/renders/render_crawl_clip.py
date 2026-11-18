"""Short museum-contact clip (task 9): the scripted crawl in simulation, labelled on every frame.

Runs the frozen crawl (../sim/crawl.py) on a model and renders frames with a "SIMULATION" badge and a caption,
then writes an H.264 MP4. Simulation only; a design variant is named as such in the caption.

Usage:
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_crawl_clip.py \
      --model ../sim/generated/trike_T20_closed_ankleroll.xml --out clip.mp4 [--speed 2] [--seconds 20]
"""

import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
import crawl  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H, BAND, FPS = 1280, 720, 70, 30


def overlay(frame, title, caption):
    img = Image.new("RGB", (W, H), (24, 28, 32))
    img.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(img)
    fs, fb, f = ImageFont.truetype(FONT_B, 20), ImageFont.truetype(FONT_B, 24), ImageFont.truetype(FONT, 17)
    badge = "SIMULATION"
    bw = d.textlength(badge, font=fs) + 24
    d.rounded_rectangle([16, 16, 16 + bw, 50], radius=7, fill=(200, 40, 40))
    d.text((28, 21), badge, font=fs, fill=(255, 255, 255))
    d.text((20, H - BAND + 8), title, font=fb, fill=(240, 240, 240))
    d.text((20, H - BAND + 40), caption, font=f, fill=(190, 196, 204))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed-up, stated on every frame")
    ap.add_argument("--seconds", type=float, default=20.0, help="clip length in playback seconds")
    ap.add_argument("--still-at", type=float, help="instead of a clip, save one JPEG still at this simulated time (s)")
    a = ap.parse_args()
    P = crawl.PARAMS
    model = mujoco.MjModel.from_xml_path(a.model)
    kin, steps = crawl.plan(model, P)
    times, targets, phases = crawl.trajectory(kin, steps, P)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    crawl.apply_seed(model, data, a.seed)
    mujoco.mj_forward(model, data)
    act = [model.actuator(f"{c}_{x}").id for c in crawl.CORNERS for x in crawl.AXES]
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    r = mujoco.Renderer(model, H - BAND, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("chassis").id
    cam.distance, cam.azimuth, cam.elevation = 1.9, 135, -12
    variant = "design variant with passive roll ankles" if "ankleroll" in a.model else "as designed"
    mass = model.body_mass.sum()
    title = "Triceratops robot, 75 cm: statically stable crawl (simulation)"
    caption = (f"MuJoCo model, {mass:.0f} kg, {variant}. One foot at a time, 0.02 m/s. "
               f"Shown at {a.speed:g}x speed. Not hardware.")
    if a.still_at is not None:
        swing = next((p["leg"] for p in phases if p["swing_start"] <= a.still_at <= p["swing_end"]), None)
        n = round(a.still_at / model.opt.timestep)
        for k in range(n):
            data.ctrl[act] = [np.interp(k * model.opt.timestep, times, targets[:, i]) for i in range(12)]
            mujoco.mj_step(model, data)
        r.update_scene(data, camera=cam)
        what = f"the {swing.replace('_', ' ')} foot in swing" if swing else "four feet down, body shifting"
        cap = (f"MuJoCo model, {mass:.0f} kg, {variant}; one foot at a time, 0.02 m/s; {what} "
               f"(t = {a.still_at:g} s). Not hardware.")
        Image.fromarray(overlay(r.render(), title, cap)).save(a.out, "JPEG", quality=88, optimize=True)
        print(f"{a.out}: still at {a.still_at} s, {what}, {Path(a.out).stat().st_size / 1024:.0f} KB")
        return
    sim_per_frame = a.speed / FPS
    n_frames = int(a.seconds * FPS)
    every = max(1, round(sim_per_frame / model.opt.timestep))
    frames = []
    k = 0
    while len(frames) < n_frames and k * model.opt.timestep < times[-1]:
        t = k * model.opt.timestep
        data.ctrl[act] = [np.interp(t, times, targets[:, i]) for i in range(12)]
        mujoco.mj_step(model, data)
        if k % every == 0:
            r.update_scene(data, camera=cam)
            frames.append(overlay(r.render(), title, caption))
        k += 1
    imageio.mimsave(a.out, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=8,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"{a.out}: {len(frames)} frames, {len(frames) / FPS:.1f} s at {FPS} fps, "
          f"{Path(a.out).stat().st_size / 1024:.0f} KB, simulated {k * model.opt.timestep:.1f} s")


if __name__ == "__main__":
    main()
