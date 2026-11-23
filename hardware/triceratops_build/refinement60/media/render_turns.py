"""Replay two recorded controller trials on the same dressed model at real time.

From refinement60/, run the following for each of the original and gentle profiles:

    python motion/run.py six-headings --seed 103 --turn-profile original --record --out trials/original
    python motion/run.py six-headings --seed 103 --turn-profile gentle --record --out trials/gentle
    python media/render_turns.py --mode six-headings --original trials/original.npz \
        --gentle trials/gentle.npz --out six-headings.mp4

Use course/103 or pivot-right/102 for the other films. The JSON next to each NPZ
supplies the mobility score from the full-rate controller trace. Requires Pillow,
an ffmpeg executable, NumPy and MuJoCo.
"""

import argparse
import functools
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1440, 812
PW, PH = 680, 352
COLUMNS = (30, 730)
FPS = 24
INK, MUTED, PAPER, RULE = "#20352e", "#50635a", "#faf9f3", "#c7cec1"
COLORS = ("#9a5235", "#28645b")
TRIALS = {
    "pivot-right": ("Right pivot", 12, 102),
    "course": ("Walking a heading course", 23, 103),
    "six-headings": ("Walking six commanded headings", 40, 103),
}


@functools.lru_cache
def font(size, bold, serif):
    name = "Georgia.ttf" if serif else ("Arial Bold.ttf" if bold else "Arial.ttf")
    fallback = "DejaVuSerif.ttf" if serif else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    for candidate in (Path("/System/Library/Fonts/Supplemental") / name, fallback):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def label(draw, x, y, s, size, *, color=INK, bold=False, serif=False, right=False):
    draw.text((x, y), s, font=font(size, bold, serif), fill=color, anchor="rt" if right else "lt")


def staged_model(path, extended_floor):
    root = ET.parse(path).getroot()
    assets = {}
    for element in root.findall("./asset/*[@file]"):
        source = path.parent / element.get("file")
        assets[source.name] = source.read_bytes()
        element.set("file", source.name)
    world = root.find("worldbody")
    for axis in range(2):
        for tick in range(-16, 57 if extended_floor else 33):
            pos = [0.0, 0.0, 0.00022]
            pos[axis] = tick * 0.25
            size = [10.0, 10.0, 0.00015]
            size[axis] = 0.0016 if tick % 4 == 0 else 0.0008
            ET.SubElement(
                world,
                "geom",
                type="box",
                pos=" ".join(map(str, pos)),
                size=" ".join(map(str, size)),
                rgba=".29 .32 .27 1",
                group="0",
                mass="0",
                contype="0",
                conaffinity="0",
            )
    root.find("./visual/quality").set("shadowsize", "2048")
    root.find("./asset/material[@name='floor_mat']").set("rgba", ".45 .48 .42 1")
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)


class Trial:
    def __init__(self, recording, model, total):
        with np.load(recording) as a:
            self.t = a["t"]
            self.pose = a["qpos"]
            self.command = a["command_deg"]
            # motion/heading.observe already expresses yaw relative to the reset heading.
            self.yaw = np.rad2deg(a["yaw_rad"])
            self.physics = a["physics_t_s"]
            self.rf = a["physics_foot_load_N"][:, 1]
        if self.pose.shape != (len(self.t), model.nq) or not np.allclose(np.diff(self.t), 0.02):
            raise ValueError("Expected 50 Hz poses matching the dressed model")
        if (
            not np.allclose(np.diff(self.physics), 0.001)
            or not np.isclose(self.physics[0], 0.001)
            or not np.allclose([self.t[0], self.t[-1], self.physics[-1]], [0, total, total])
        ):
            raise ValueError("Expected a complete trial with 1 kHz force")
        self.data = mj.MjData(model)
        self.scored = self.physics > 2.0  # Same strict mask as motion/run.py.
        self.peak = np.maximum.accumulate(np.where(self.scored, self.rf, 0.0))

    def draw_pose(self, idx, renderer, model, opt):
        self.data.qpos[:] = self.pose[idx]
        mj.mj_forward(model, self.data)
        camera = mj.MjvCamera()
        camera.lookat[:] = (self.pose[idx, 0] - 0.04, self.pose[idx, 1], 0.25)
        camera.distance, camera.azimuth, camera.elevation = 1.55, 114, -36
        renderer.update_scene(self.data, camera=camera, scene_option=opt)
        renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
        return Image.fromarray(renderer.render())


def coord(value, lo, hi, pixels):
    return np.rint((value - lo) * pixels / (hi - lo)).astype(int)


def graph_layers(trial, color, total, xmax_force):
    """Return full plots for lossless crops at the advancing playback cursor."""
    w, h = PW - 78, 61
    x = coord(trial.t, 0, total, w - 1)
    yaw = coord(50 - trial.yaw, 0, 100, h - 1)
    command = coord(50 - trial.command, 0, 100, h - 1)
    background = Image.new("RGB", (w, h), "#f0f1e8")
    d = ImageDraw.Draw(background)
    d.line((0, h // 2, w - 1, h // 2), fill=RULE)
    for k in np.r_[0, np.nonzero(np.diff(trial.command))[0] + 1]:
        nxt = np.flatnonzero(trial.command[k:] != trial.command[k])
        end = x[k + nxt[0]] - 1 if len(nxt) else w - 1
        d.rectangle((x[k], command[k] - 3, end, command[k] + 3), fill="#d5dcd1")
        for p in range(x[k], end + 1, 10):
            d.line((p, command[k], min(end, p + 5), command[k]), fill=MUTED)
    heading = background.copy()
    ImageDraw.Draw(heading).line(list(zip(x.tolist(), yaw.tolist())), fill=color, width=3)
    force = Image.new("RGB", (w, h), "#f0f1e8")
    blank = force.copy()
    fp = coord(trial.physics[trial.scored], 0, total, w - 1)
    peaks = np.zeros(w)
    np.maximum.at(peaks, fp, trial.rf[trial.scored])
    d = ImageDraw.Draw(force)
    for p, height in enumerate(coord(peaks, 0, xmax_force, h - 1)):
        if height:
            d.line((p, h - 1 - height, p, h - 1), fill=color)
    return (background, heading), (blank, force)


def decorate(base, title, seed, total, scale):
    d = ImageDraw.Draw(base)
    label(d, 30, 17, title, 40, serif=True)
    label(
        d, 31, 73, f"60 cm open-rib Triceratops   /   128 mm upper forelegs   /   nominal trial {seed}", 19, color=MUTED
    )
    label(d, WIDTH - 30, 25, "MuJoCo simulation   ·   real time", 21, color=MUTED, right=True)
    for j, x in enumerate(COLUMNS):
        d.line((x, 108, x + PW, 108), fill=COLORS[j], width=3)
        label(d, x, 122, ["Original controller", "Gentler controller"][j], 26, bold=True)
        label(d, x, 535, "Heading", 19, bold=True)
        label(d, x + PW - 50, 566, "+50°", 14, color=MUTED)
        label(d, x + PW - 50, 612, "−50°", 14, color=MUTED)
        label(d, x, 644, "Right forefoot", 19, bold=True)
        label(d, x + PW - 50, 675, f"{scale} N", 14, color=MUTED)
        label(d, x + PW - 50, 719, "0 N", 14, color=MUTED)
        label(d, x, 741, "0 s", 14, color=MUTED)
        label(d, x + PW - 78, 741, f"{total} s", 14, color=MUTED, right=True)
    d.line((30, 769, WIDTH - 30, 769), fill=RULE, width=1)
    label(
        d,
        30,
        784,
        "Dashed heading target and ±5° band. Foot load shows 1 kHz step peaks after 2 s. Floor grid is 25 cm.",
        18,
        color=MUTED,
    )


def write(args):
    title, total, seed = TRIALS[args.mode]
    model = staged_model(args.model, args.mode == "six-headings")
    files = (args.original, args.gentle)
    trials = [Trial(path, model, total) for path in files]
    metadata = [json.loads(f.with_suffix(".json").read_text()) for f in files]
    for index, run in enumerate(metadata):
        settings = run["settings"]
        if (
            settings["program"] != args.mode
            or settings["seed"] != seed
            or settings["random"]
            or (settings.get("turn_profile") == "gentle") != bool(index)
        ):
            raise ValueError("Expected original and gentler trials on the same nominal seed")
    if not np.array_equal(trials[0].command, trials[1].command):
        raise ValueError("Command schedules differ")
    scale = max(100, int(np.ceil(max(t.peak[-1] for t in trials) / 100) * 100))
    base = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    decorate(base, title, seed, total, scale)
    layers = [graph_layers(t, c, total, scale) for t, c in zip(trials, COLORS)]
    opt = mj.MjvOption()
    opt.sitegroup[:] = 0
    opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    count = total * FPS
    encode = [
        args.ffmpeg,
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
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.out),
    ]
    with mj.Renderer(model, height=PH, width=PW) as renderer:

        def frame(time, final=False):
            im = base.copy()
            draw = ImageDraw.Draw(im)
            label(draw, WIDTH - 30, 73, f"{time:05.2f} / {total:.0f} s", 20, bold=True, right=True)
            for j, (trial, x) in enumerate(zip(trials, COLUMNS)):
                idx = int(np.argmin(np.abs(trial.t - time)))
                fi = int(np.searchsorted(trial.physics, time + 1e-8, side="right") - 1)
                im.paste(trial.draw_pose(idx, renderer, model, opt), (x, 166))
                segments = metadata[j]["segments"]
                command_text = f"Command {trial.command[idx]:+.0f}°"
                if args.mode == "six-headings":
                    step = sum(leg["start_s"] <= time for leg in segments[1:])
                    command_text = f"{step}/6   ·   {command_text}" if step else "Starting straight   ·   0°"
                label(draw, x + PW, 122, command_text, 20, color=MUTED, right=True)
                label(
                    draw, x + PW, 535, f"Measured  {trial.yaw[idx]:+.1f}°", 20, color=COLORS[j], bold=True, right=True
                )
                peak = f"{trial.peak[fi]:.1f} N" if fi >= 0 and trial.scored[fi] else "—"
                label(draw, x + PW, 644, f"Peak so far  {peak}", 20, color=COLORS[j], bold=True, right=True)
                if final:
                    acquired = metadata[j]["segments"][-1]["first_1s_in_5deg_band_s"]
                    note = (
                        "No 1 s hold on the final target"
                        if acquired is None
                        else f"Entered final ±5° band after {acquired:.2f} s and held for 1 s"
                    )
                    if args.mode == "six-headings":
                        passed = sum(leg["first_1s_in_5deg_band_s"] is not None for leg in segments[1:])
                        note = f"Held {passed} of 6 commanded headings within ±5° for 1 s"
                    draw.rectangle((x, 166, x + PW, 196), fill=PAPER)
                    label(draw, x, 169, note, 18, bold=True, color=COLORS[j])
                for y, (background, history) in zip((564, 673), layers[j]):
                    w, h = background.size
                    cursor = int(coord(time, 0, total, w - 1))
                    im.paste(background, (x, y))
                    if cursor > 0:
                        im.paste(history.crop((0, 0, cursor, h)), (x, y))
                    draw.line((x + cursor, y, x + cursor, y + h - 1), fill=INK)
            return im

        if args.still is not None:
            frame(args.still, final=np.isclose(args.still, total)).save(args.out)
            return
        proc = subprocess.Popen(encode, stdin=subprocess.PIPE)
        try:
            for index in range(count):
                proc.stdin.write(frame(index / FPS).tobytes())
            endcard = frame(total, final=True).tobytes()
            for _ in range(2 * FPS):
                proc.stdin.write(endcard)
        finally:
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("ffmpeg failed")
    print(
        json.dumps(
            {
                "file": args.out.name,
                "frames": count + 2 * FPS,
                "fps": FPS,
                "sim_seconds": total,
                "end_hold_seconds": 2,
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "RF_peak_N": [float(t.peak[-1]) for t in trials],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=TRIALS, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/refined/triceratops.xml")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--gentle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to the ffmpeg executable.")
    parser.add_argument("--still", type=float, help="Save one inspection frame at this simulated time.")
    write(parser.parse_args())
