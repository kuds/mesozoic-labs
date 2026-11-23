"""Render the complete exterior from an untouched capsule-foot controller replay.

The articulated jaw and lids are optional cosmetic previews. The original model
supplies the locomotion states. A decorative envelope is never a source of
rangefinder or contact input to the controller.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from build_exterior import SOURCE
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
FPS = 24


def fresh_replay():
    """Record 8 seconds of the frozen controller on the canonical robot."""
    sys.path.insert(0, str(HERE.parents[1] / "walk"))
    import walk60

    env = walk60.TrikeEnv(
        walk60.plant_config(model_path=str(SOURCE), domain_randomization=False, observation_noise=True)
    )
    ctrl = walk60.TrotController(env, walk60.WalkParams())
    obs, _ = env.reset(seed=101, options={"command_speed": 0.25, "delay_steps": 10})
    times = []
    states = []

    def record(e):
        times.append(e.data.time)
        states.append(e.data.qpos.copy())

    env.substep_hook = record
    while env.data.time < 8:
        obs, _, done, truncated, info = env.step(ctrl.act(obs))
        if done or truncated:
            raise RuntimeError(f"Replay ended at {env.data.time}: {info}")
    return np.asarray(times), np.asarray(states)


def font(size, bold=False):
    file = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(file, size)
    except OSError:
        for p in (
            Path("/usr/share/fonts/truetype/dejavu") / file,
            Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
        ):
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def annotation(image, title, detail):
    w, h = image.size
    band = max(round(h * 0.085), round(w * 0.048))
    plate = Image.new("RGB", (w, h + band), (35, 43, 38))
    plate.paste(image, (0, 0))
    draw = ImageDraw.Draw(plate)
    s = w / 1450
    pad = int(26 * s)
    ft = font(max(15, int(23 * s)), True)
    fs = font(max(11, int(15 * s)))
    draw.text((pad, h + int(10 * s)), title, font=ft, fill=(248, 239, 214))
    draw.text((pad, h + int(39 * s)), detail, font=fs, fill=(211, 211, 192))
    text = "SIMULATION"
    badge = font(max(12, int(15 * s)), True)
    width = draw.textlength(text, font=badge)
    draw.rounded_rectangle((w - pad - width - 24 * s, 18 * s, w - pad, 46 * s), radius=5 * s, fill=(48, 59, 49))
    draw.text((w - pad - width - 12 * s, 23 * s), text, font=badge, fill=(245, 236, 207))
    return plate


def expression(model, jaw=0.0, blink=0.0):
    """Independent render-layer jaw pivot and two eyelid sliders."""
    q = model.body_quat[model.body("exterior_jaw").id]
    a = math.radians(18 * jaw) / 2
    q[:] = [math.cos(a), 0, math.sin(a), 0]
    for side in ("left", "right"):
        p = model.body_pos[model.body("exterior_" + side + "_lid").id]
        p[2] = 0.0295 - 0.0165 * blink
        p[1] = (1 if side == "left" else -1) * (0.0778 + 0.0025 * blink)
        lid = model.geom("exterior_" + side + "_lid_scale").id
        model.geom_size[lid, 2] = 0.0073 + 0.0032 * blink


def photo(model, data, renderer, q, view, face=(0, 0), time=0):
    expression(model, *face)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    # Camera tracks translation while keeping the frame's orientation fixed.
    camera.lookat[:] = q[:3] + np.asarray(view[:3])
    camera.distance, camera.azimuth, camera.elevation = view[3:]
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    renderer.update_scene(data, camera=camera, scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    color = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera, scene_option=opt)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    # An ID matte preserves the actual MuJoCo silhouette. Use the floor shadow
    # from the same scene, blurred for a soft studio background without a horizon.
    is_geom = seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
    floor = is_geom & (seg[:, :, 0] == model.geom("floor").id)
    animal = is_geom & ~floor
    if not animal.any():
        raise RuntimeError("No presentation mesh in the camera frame")
    h, w = color.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    light = np.exp(-(((xx / w - 0.49) / 0.68) ** 2) - ((yy / h - 0.17) / 0.84) ** 2)
    base = np.array([181, 186, 175]) + light[..., None] * np.array([44, 42, 42])
    gray = color.mean(axis=2)
    valid = floor & (yy > h * 0.48)
    if valid.any():
        threshold = np.percentile(gray[valid], 88)
        shadow = np.clip((threshold - gray) / max(30, threshold * 0.46), 0, 1) * valid
        sm = Image.fromarray(np.uint8(shadow * 255)).filter(ImageFilter.GaussianBlur(max(2.2, w * 0.006)))
        shadow = np.asarray(sm) / 255
        base -= shadow[..., None] * np.array([51, 49, 44])
    mask = Image.fromarray(np.uint8(animal) * 255).filter(ImageFilter.GaussianBlur(0.34))
    return Image.composite(Image.fromarray(color), Image.fromarray(np.uint8(np.clip(base, 0, 255))), mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replay", type=Path, help="optional npz with t and qpos arrays from the canonical simulation")
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.replay:
        with np.load(a.replay) as replay:
            ts, q = replay["t"], replay["qpos"]
    else:
        ts, q = fresh_replay()
    model = mujoco.MjModel.from_xml_path(str(a.model))
    data = mujoco.MjData(model)

    def state(t):
        return q[np.argmin(abs(ts - t))]

    views = {
        "three_quarter": (-0.020, 0, -0.038, 1.43, 118, -11),
        "side": (-0.035, 0, -0.045, 1.48, 92, -6),
        "face": (0.320, 0, 0.093, 0.73, 135, -10),
    }
    with mujoco.Renderer(model, height=848, width=1472) as renderer:
        for key, p in views.items():
            shot = photo(model, data, renderer, state(0.25), p)
            shot = annotation(
                shot,
                {
                    "three_quarter": "Triceratops  /  Complete exterior",
                    "side": "Triceratops  /  Side profile",
                    "face": "Triceratops  /  Head and frill",
                }[key],
                "60 cm Option C     •     Capsule-foot robot     •     Proposed cosmetic shell",
            )
            shot.save(a.out / (key + ".png"))
    panels = []
    with mujoco.Renderer(model, height=440, width=600) as renderer:
        for label, face in (("Rest", (0, 0)), ("Blink", (0, 1)), ("Beak open", (1, 0))):
            shot = photo(model, data, renderer, state(0.25), (0.358, 0, 0.077, 0.635, 136, -8), face)
            d = ImageDraw.Draw(shot)
            d.text((18, 16), label, font=font(22, True), fill=(36, 45, 37), stroke_width=2, stroke_fill=(206, 206, 186))
            panels.append(shot)
    strip = Image.new("RGB", (1800, 440))
    for i, panel in enumerate(panels):
        strip.paste(panel, (i * 600, 0))
    strip = annotation(
        strip,
        "Triceratops  /  Facial movement",
        "Selectable visual poses     •     Jaw pivot and independent eyelid sliders     •     Actuator performance untested",
    )
    strip.save(a.out / "expressions.png")
    if a.video:
        import imageio_ffmpeg

        width, height = 1120, 624
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            writer = None
            output = a.out / "exterior_trot.mp4"
            for t in np.arange(0, 7.5, 1 / FPS):
                walktime = 0.15 + t
                jaw = max(0, 1 - abs(t - 3.75) / 0.62) * 0.65
                blink = max(max(0, 1 - abs(t - 2.00) / 0.13), max(0, 1 - abs(t - 5.75) / 0.13))
                shot = photo(
                    model,
                    data,
                    renderer,
                    state(walktime),
                    (-0.022, 0, -0.044, 1.43, 116 + 6 * np.cos(t * 0.48), -10),
                    (jaw, blink),
                )
                frame = annotation(
                    shot,
                    "Triceratops  /  Model-based trot",
                    "0.25 m/s command   •   Seed 101   •   Simulated 10 ms delay   •   Jaw and blink are cosmetic cues",
                )
                if writer is None:
                    w, h = frame.size
                    h -= h % 2
                    writer = imageio_ffmpeg.write_frames(
                        str(output),
                        (w, h),
                        fps=FPS,
                        codec="libx264",
                        quality=8,
                        macro_block_size=1,
                        output_params=["-pix_fmt", "yuv420p"],
                    )
                    writer.send(None)
                writer.send(np.asarray(frame.crop((0, 0, w, h))).tobytes())
            writer.close()


if __name__ == "__main__":
    main()
