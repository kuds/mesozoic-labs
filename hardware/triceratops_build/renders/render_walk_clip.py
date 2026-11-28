"""Render clip (task 9) at 60 cm: the model-based trot in simulation, labelled on every frame.

Runs the frozen model-based trot (../walk/walk60.py, the controller judged in ../walk/REPORT_WALK60.md) on its
plant: the training env's roll-ankle model by default, or the model ``--model`` names (the capsule foot chosen on
22 Nov 2026 is ../sim/generated/trike60_C_closed_capsule.xml). 10 ms servo delay and sensor noise on, no
randomisation, the 0.25 m/s demo command. ``--shadow-clip 6`` keeps the ground shadow over the whole 20 s walk:
the model's directional light casts shadows only within its shadowclip x extent of the origin (about 0.9 m).
It renders frames at 30 per second of simulated time (so the clip plays in real time unless --speed says otherwise)
with a "SIMULATION" badge and a caption, and writes an H.264 MP4. Rendering only reads the simulation; the episode
is the same as without it.

Usage:
  IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
      --out walk60_trot_1x.mp4 [--model ../sim/generated/trike60_C_closed_capsule.xml] [--seed 101] [--seconds 20]
      [--speed 1]
  ... --out still.jpg --still-at 8.0       (one JPEG still at that simulated time instead of a clip)
"""

import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "walk"))
import walk60  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H, BAND, FPS = 1280, 720, 96, 30
MAX_STILL_BYTES = 500 * 1024


def foot_label(model_name: str) -> str:
    """The caption's words for the robot's feet, from the model file's name."""
    if "ankleroll" in model_name or "roll_k" in model_name:
        return "design variant with passive roll ankles"
    if "capsule" in model_name:
        return "capsule feet (the foot chosen on 22 Nov 2026)"
    if "ball" in model_name:
        return "ball feet (a foot-study variant, not chosen)"
    return "as designed"


def overlay(frame, title, lines):
    """The frame with a SIMULATION badge top left and bottom right of the picture, and the caption band."""
    img = Image.new("RGB", (W, H), (24, 28, 32))
    img.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(img)
    fs, fb, f = ImageFont.truetype(FONT_B, 20), ImageFont.truetype(FONT_B, 24), ImageFont.truetype(FONT, 17)
    badge = "SIMULATION"
    bw = d.textlength(badge, font=fs) + 24
    d.rounded_rectangle([16, 16, 16 + bw, 50], radius=7, fill=(200, 40, 40))
    d.text((28, 21), badge, font=fs, fill=(255, 255, 255))
    x1, y1 = W - 16 - bw, H - BAND - 50
    d.rounded_rectangle([x1, y1, x1 + bw, y1 + 34], radius=7, fill=(200, 40, 40))
    d.text((x1 + 12, y1 + 5), badge, font=fs, fill=(255, 255, 255))
    for text, font in [(title, fb)] + [(line, f) for line in lines]:  # refuse rather than clip at the right edge
        if d.textlength(text, font=font) > W - 40:
            raise SystemExit(f"caption line too wide for the {W} px frame; nothing written: {text!r}")
    d.text((20, H - BAND + 8), title, font=fb, fill=(240, 240, 240))
    for i, line in enumerate(lines):
        d.text((20, H - BAND + 40 + 24 * i), line, font=f, fill=(190, 196, 204))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101, help="episode seed (the judged runs used 101-110)")
    ap.add_argument("--command", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed-up, stated on every frame")
    ap.add_argument("--seconds", type=float, default=20.0, help="clip length in playback seconds")
    ap.add_argument("--still-at", type=float, help="instead of a clip, save one JPEG still at this simulated time (s)")
    ap.add_argument("--azimuth", type=float, default=125.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=1.5)
    ap.add_argument("--model", default=None, help="the robot model (default: the plant's roll-ankle model)")
    ap.add_argument("--shadow-clip", type=float, default=None,
                    help="the model's vis.map.shadowclip for the render (default: the model's own, 1)")
    a = ap.parse_args()

    plant = dict(domain_randomization=False, observation_noise=True)
    if a.model:
        plant["model_path"] = str(Path(a.model).resolve())
    env = walk60.TrikeEnv(walk60.plant_config(**plant))
    model_name = Path(env.cfg.model_path).name
    feet = foot_label(model_name)
    ctrl = walk60.TrotController(env, walk60.WalkParams())
    m, d = env.model, env.data
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    if a.shadow_clip is not None:  # rendering only: the physics does not read it
        m.vis.map.shadowclip = a.shadow_clip
    r = mujoco.Renderer(m, H - BAND, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env._chassis
    cam.distance, cam.azimuth, cam.elevation = a.distance, a.azimuth, a.elevation
    mass = env.nominal_mass
    pace_title = "at its planned walking speed, 0.25 m/s" if a.command == 0.25 else f"at {a.command:g} m/s"
    title = f"Triceratops robot, 60 cm: trotting {pace_title} (simulation)"
    pace = "shown at actual speed (1x)" if a.speed == 1 else f"shown at {a.speed:g}x speed"
    caption = [f"MuJoCo model, {mass:.1f} kg, {feet}. Model-based trot, {pace}.",
               "10 ms servo delay and sensor noise simulated. Not hardware; nothing is built yet."]

    obs, _ = env.reset(seed=a.seed, options={"command_speed": a.command, "delay_steps": 10})
    n_frames = int(a.seconds * FPS)
    t_end = a.still_at if a.still_at is not None else a.seconds * a.speed
    t_end = min(t_end, env.cfg.episode_seconds)
    frames, fell = [], ""

    def grab():
        r.update_scene(d, camera=cam)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0  # no mirror image under the floor
        return r.render()

    def hook(e):  # after every 1 ms physics step: a frame at each 1/30 s of playback (reads the state only)
        if a.still_at is None and len(frames) < n_frames and d.time + 1e-9 >= len(frames) * a.speed / FPS:
            frames.append(overlay(grab(), title, caption))

    if a.still_at is None:
        frames.append(overlay(grab(), title, caption))  # t = 0
        env.substep_hook = hook
    while True:
        if d.time >= t_end - 1e-9:
            break
        obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
        if terminated:
            fell = info["termination"]
            break
        if truncated:
            break
    if fell:
        raise SystemExit(f"the episode ended early ({fell}) at {d.time:.2f} s; nothing written")
    env.substep_hook = None
    if a.still_at is not None:
        cap = [f"MuJoCo model, {mass:.1f} kg, {feet}. Model-based trot, "
               f"{d.time:.1f} s into the episode.", caption[1]]
        img = Image.fromarray(overlay(grab(), title, cap))
        out = Path(a.out)
        for q in (90, 85, 80, 75):
            img.save(out, "JPEG", quality=q, optimize=True, progressive=True)
            if out.stat().st_size <= MAX_STILL_BYTES:
                break
        print(f"{out}: still at {d.time:.2f} s, JPEG quality {q}, {out.stat().st_size / 1024:.0f} KB")
        return
    imageio.mimsave(a.out, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=8,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"{a.out}: {len(frames)} frames, {len(frames) / FPS:.1f} s at {FPS} fps, "
          f"{Path(a.out).stat().st_size / 1024:.0f} KB, simulated {d.time:.1f} s, seed {a.seed}")


if __name__ == "__main__":
    main()
