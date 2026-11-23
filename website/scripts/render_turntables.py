"""Render the rotating specimen plates used on the Animals pages.

Run with the repository's MuJoCo dependencies and FFmpeg with libx264. The adult
Triceratops is loaded from examples/triceratops_horridus.
"""

import argparse
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "website" / "static" / "videos" / "turntables"
WIDTH, HEIGHT = 800, 450
FPS, SECONDS = 24, 30


def subjects(triceratops_xml):
    for path in sorted((ROOT / "environments").glob("*/*/assets/*.xml")):
        species = path.parents[2].name
        version = path.parents[1].name
        if path.stem == f"{species}_{version}":
            yield f"{species}-{version}", path
    yield "stegosaurus_stenops-anatomical", ROOT / "examples/stegosaurus_stenops/model.xml"
    yield "triceratops-anatomical", triceratops_xml or ROOT / "examples/triceratops_horridus/anatomical.xml"


def studio(model):
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.tendongroup[:] = 0
    # Keep the animal's visible surfaces. Exclude floor, targets and collision proxies.
    for i, body in enumerate(model.geom_bodyid):
        root = model.body(model.body_rootid[body]).name
        if body == 0 or root in {"prey", "food", "target"}:
            model.geom_group[i] = 5

    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    model.vis.global_.fovy = 19
    model.vis.quality.offsamples = 4
    model.vis.headlight.ambient[:] = 0.50
    model.vis.headlight.diffuse[:] = 0.52
    model.vis.headlight.specular[:] = 0.08
    return options


def set_background(model, color):
    for tex in np.flatnonzero(model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX):
        count = model.tex_width[tex] * model.tex_height[tex]
        address = model.tex_adr[tex]
        channels = model.tex_nchannel[tex]
        model.tex_data[address : address + count * channels].reshape(count, channels)[:, :3] = color


def orbit_camera(model, data, scene, options):
    # Use the world-space bounds of both rigid geometry and deformed skins.
    corners = []
    for i, group in enumerate(model.geom_group):
        if not options.geomgroup[group] or model.geom_rgba[i, 3] == 0:
            continue
        rotation = data.geom_xmat[i].reshape(3, 3)
        center = data.geom_xpos[i] + rotation @ model.geom_aabb[i, :3]
        extent = abs(rotation) @ model.geom_aabb[i, 3:]
        corners.extend([center - extent, center + extent])
    points = np.asarray(corners).reshape(-1, 3)
    if scene.nskin:
        points = np.vstack([points, scene.skinvert.reshape(-1, 3)])
    low, high = points.min(axis=0), points.max(axis=0)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (low + high) / 2
    camera.elevation = -11
    # Fit the broadside view and the longest diagonal at this camera elevation.
    centered = points - camera.lookat
    radius = np.linalg.norm(centered[:, :2], axis=1)
    slope = np.tan(np.deg2rad(model.vis.global_.fovy / 2))
    elevation = np.deg2rad(abs(camera.elevation))
    half_height = np.max(abs(centered[:, 2]) * np.cos(elevation) + radius * np.sin(elevation))
    camera.distance = max(radius.max() * HEIGHT / WIDTH, half_height) * 1.09 / slope + 0.32 * radius.max()
    return camera


def render_orbit(model, data, options, destination):
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-g",
        str(FPS // 2),  # Keep drag-to-rotate seeks within half a second of a keyframe.
        "-preset",
        "slow",
        "-crf",
        "24",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-color_range",
        "tv",
        "-pix_fmt",
        "yuv420p",
        "-bsf:v",
        "h264_metadata=colour_primaries=1:transfer_characteristics=13:matrix_coefficients=1",
        "-movflags",
        "+faststart+write_colr",
        str(destination),
    ]
    with mujoco.Renderer(model, width=WIDTH, height=HEIGHT) as renderer:
        renderer.update_scene(data, scene_option=options)
        scene = renderer.scene
        for flag in ("mjRND_FOG", "mjRND_HAZE", "mjRND_REFLECTION", "mjRND_SHADOW"):
            scene.flags[getattr(mujoco.mjtRndFlag, flag)] = False
        camera = orbit_camera(model, data, scene, options)
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            with encoder.stdin:
                for frame in range(FPS * SECONDS):
                    camera.azimuth = 102 + 360 * frame / (FPS * SECONDS)
                    mujoco.mjv_updateCamera(model, data, camera, scene)
                    mujoco.mjv_makeLights(model, data, scene)
                    rgb = renderer.render()
                    if frame == 0:
                        first = rgb.copy()
                    encoder.stdin.write(rgb.tobytes())
            if encoder.wait():
                raise RuntimeError(f"FFmpeg failed to encode {destination}")

    return first


def render(path, name):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    options = studio(model)
    if name in {"triceratops-anatomical", "compsognathus_longipes-robot"}:
        model.vis.headlight.ambient[:] = 0.59
        model.vis.headlight.diffuse[:] = 0.55
    if name == "compsognathus_longipes-robot":
        # Keep the silhouette of the charcoal chassis visible on a dark panel.
        np.maximum(model.mat_rgba[:, :3], 0.17, out=model.mat_rgba[:, :3])
        np.maximum(model.geom_rgba[:, :3], 0.17, out=model.geom_rgba[:, :3])
    # Encode into temporary files so an interrupted render preserves the published assets.
    with TemporaryDirectory(dir=OUTPUT) as temp:
        stage = Path(temp)
        # Match the sRGB light and dark specimen panels in the site stylesheet.
        set_background(model, (243, 243, 235))
        light = render_orbit(model, data, options, stage / f"{name}.mp4").astype(float)
        set_background(model, (34, 46, 39))
        dark = render_orbit(model, data, options, stage / f"{name}-dark.mp4").astype(float)

        # The difference between the two backgrounds gives edge coverage at each pixel.
        alpha = np.clip(1 - np.mean((light - dark) / (light[0, 0] - dark[0, 0]), axis=2), 0, 1)[..., None]
        rgb = np.divide(dark - dark[0, 0] * (1 - alpha), alpha, out=np.zeros_like(dark), where=alpha > 0)
        rgba = np.concatenate((rgb, alpha * 255), axis=2).clip(0, 255).round().astype("uint8")
        Image.fromarray(rgba).save(stage / f"{name}.webp", quality=92, method=6)
        for asset in stage.iterdir():
            asset.replace(OUTPUT / asset.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triceratops-xml", type=Path, help="override the checked-in adult reference XML")
    parser.add_argument("names", nargs="*", help="render only the named specimen plates")
    args = parser.parse_args()
    models = dict(subjects(args.triceratops_xml))
    if unknown := set(args.names) - models.keys():
        parser.error(f"missing source for: {', '.join(sorted(unknown))}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in args.names or models:
        print(f"Rendering {name}", flush=True)
        render(models[name], name)


if __name__ == "__main__":
    main()
