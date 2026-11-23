"""Render matched foreleg studies from the nominal physics recordings made by trials.py.

Example: python mid160/render.py pivot-right --records /tmp/records --out pivot.mp4
For the neutral motor detail: python mid160/render.py detail --out comparison.png
The optional graphics dependencies are Pillow and ffmpeg with H.264 encoding.
"""

import argparse
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "media"))
from render_turns import COLORS, FPS, MUTED, PAPER, RULE, label, staged_model  # noqa: E402

WIDTH, HEIGHT = 1344, 696
PW, PH = 640, 350
X = (24, 680)
SIZES = (128, 160)
FOLDERS = ("refined", "mid160")
# (title, seed, replay start and end in simulated seconds)
ROUTINES = {
    "walk": ("Four-beat exhibit walk", 11, 5, 17),
    "pivot-right": ("Right pivot", 102, 0, 12),
    "six-headings": ("Six changes of heading", 103, 0, 40),
}


class Exhibit:
    def __init__(self, folder, extended, stack):
        self.folder = folder
        self.model = staged_model(ROOT / "models" / folder / "triceratops.xml", extended)
        self.data = mj.MjData(self.model)
        self.renderer = stack.enter_context(mj.Renderer(self.model, height=PH, width=PW))
        self.option = mj.MjvOption()
        self.option.geomgroup[:] = [1, 0, 1, 0, 0, 0]
        self.option.sitegroup[:] = 0

    def pose(self, qpos, target, view):
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)
        camera = mj.MjvCamera()
        camera.lookat[:] = target
        camera.distance, camera.azimuth, camera.elevation = view
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)
        self.renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
        return Image.fromarray(self.renderer.render())


class Recording:
    def __init__(self, path, exhibit, mode, seed, start, end):
        self.score = json.loads(path.with_suffix(".json").read_text())
        settings = self.score["settings"]
        if (
            settings["seed"] != seed
            or settings["program"] != mode
            or settings["random"]
            or Path(settings["model"]).parent.name != exhibit.folder
            or (mode != "walk" and settings["turn_profile"] != "gentle")
        ):
            raise ValueError(f"Wrong model or trial for replay: {path}")
        with np.load(path) as a:
            self.t, self.pose = a["t"], a["qpos"]
            self.physics, self.load = a["physics_t_s"], a["physics_foot_load_N"]
            if mode != "walk":
                self.command, self.yaw = a["command_deg"], np.degrees(a["yaw_rad"])
        if self.pose.shape != (len(self.t), exhibit.model.nq) or not np.allclose(np.diff(self.physics), 0.001):
            raise ValueError(f"Missing pose or 1 kHz forces: {path}")
        if self.t[0] > start + 0.001 or self.t[-1] < end - 0.001:
            raise ValueError(f"Incomplete recording: {path}")
        if mode == "walk":
            lo, hi = self.score["steady_seconds"]
            scored = (self.physics >= lo) & (self.physics <= hi)
        else:
            scored = self.physics > 2.0
        show = scored & (self.physics >= start - 1e-8) & (self.physics <= end + 1e-8)
        self.when, self.force = self.physics[show], self.load[show]
        self.all_peaks = np.maximum.accumulate(self.force.max(axis=1))
        self.rf_peaks = np.maximum.accumulate(self.force[:, 1])


def header(title, subtitle, height=HEIGHT):
    out = Image.new("RGB", (WIDTH, height), PAPER)
    draw = ImageDraw.Draw(out)
    label(draw, 24, 18, title, 38, serif=True)
    label(draw, WIDTH - 24, 25, "60 cm open-rib Triceratops", 20, color=MUTED, right=True)
    label(draw, 25, 67, subtitle, 18, color=MUTED)
    for i, x in enumerate(X):
        draw.line((x, 99, x + PW, 99), fill=COLORS[i], width=3)
        label(draw, x, 112, f"{SIZES[i]} mm upper foreleg", 25, bold=True)
    return out


def motor_detail(exhibits):
    out = header("Foreleg and motor proportions", "Neutral pose    /    Same scale within each row", 816)
    draw = ImageDraw.Draw(out)
    for i, (exhibit, x) in enumerate(zip(exhibits, X)):
        m = exhibit.model
        neutral = m.key("neutral").qpos
        # Both cameras observe fixed coordinates relative to the common pelvis and floor.
        full = exhibit.pose(neutral, (-0.045, 0, 0.30), (1.48, 94, -7))
        out.paste(full.crop((0, 12, PW, 311)), (x, 151))
        motor_id = m.geom("study_actual_right_fore_x6_flexion_stator").id
        diameter = m.geom_size[motor_id, 0] * 2000
        m.geom_matid[motor_id] = -1
        m.geom_rgba[motor_id] = [0.95, 0.51, 0.13, 1.0]
        detail = exhibit.pose(neutral, (0.215, -0.14, 0.241), (0.47, 98, -7))
        out.paste(detail.crop((0, 25, PW, 303)), (x, 486))
        label(
            draw,
            x,
            772,
            f"X6 diameter  {diameter:.0f} mm   /   {diameter / SIZES[i]:.1%} of upper span",
            19,
            bold=True,
            color=COLORS[i],
        )
    label(draw, 24, 462, "Foreleg detail   ·   Right X6 housing highlighted in amber", 17, color=MUTED)
    return out


def chart(record, start, end, cap, color):
    w, h = PW, 53
    xs = np.clip(np.floor((record.when - start) / (end - start) * (w - 1)).astype(int), 0, w - 1)
    tallest = np.zeros(w)
    np.maximum.at(tallest, xs, record.force.max(axis=1))
    im = Image.new("RGB", (w, h), "#e9ede4")
    draw = ImageDraw.Draw(im)
    for x, n in enumerate(tallest):
        if n > 0:
            draw.line((x, h - 1, x, h - 1 - round((h - 1) * n / cap)), fill=color)
    return im


def movie(exhibits, args):
    title, seed, start, end = ROUTINES[args.program]
    walk = args.program == "walk"
    subtitle = "Level ground" if walk else "Gentle steering"
    base = header(title, f"{subtitle}    /    Nominal trial {seed}    /    Real time")
    records = [
        Recording(
            args.records / str(size) / f"{args.program}-{seed}-nominal.npz", exhibit, args.program, seed, start, end
        )
        for size, exhibit in zip(SIZES, exhibits)
    ]
    if not walk and not np.array_equal(records[0].command, records[1].command):
        raise ValueError("Command schedules differ")
    limit = 50 * np.ceil(max(r.all_peaks[-1] for r in records) / 50)
    charts = [chart(r, start, end, limit, c) for r, c in zip(records, COLORS)]
    d = ImageDraw.Draw(base)
    for x in X:
        d.rectangle((x, 560, x + PW - 1, 612), fill="#e9ede4")
        label(d, x, 530, "Highest load on any foot", 18, bold=True)
        label(d, x, 617, f"{start} s", 14, color=MUTED)
        label(d, x + PW, 617, f"{end} s", 14, right=True, color=MUTED)
    d.line((24, 647, WIDTH - 24, 647), fill=RULE)
    threshold = "during the excerpt" if walk else "after 2 s"
    label(
        d,
        24,
        660,
        f"Peak force from 1 kHz physics {threshold}   ·   Shared vertical scale 0–{limit:.0f} N   ·   Floor grid 25 cm",
        17,
        color=MUTED,
    )

    def frame(t, final=False):
        im = base.copy()
        draw = ImageDraw.Draw(im)
        label(draw, WIDTH - 24, 65, f"{t:05.2f} s", 19, bold=True, right=True)
        for j, (x, record, exhibit) in enumerate(zip(X, records, exhibits)):
            idx = int(np.argmin(abs(record.t - t)))
            q = record.pose[idx]
            if walk:
                view = (1.48, 96, -7)
                target = (q[0] - 0.05, 0, 0.30)
            else:
                view = (1.51, 108, -31)
                target = (q[0] - 0.05, q[1], 0.27)
            im.paste(exhibit.pose(q, target, view), (x, 151))
            fi = int(np.searchsorted(record.when, t + 1e-8, side="right") - 1)
            rf = f"{record.rf_peaks[fi]:.1f} N" if fi >= 0 else "—"
            peak = f"{record.all_peaks[fi]:.1f} N" if fi >= 0 else "—"
            label(draw, x + PW, 532, f"Peak  {peak}", 18, color=COLORS[j], bold=True, right=True)
            progress = max(0, min(PW, round((t - start) / (end - start) * PW)))
            if progress:
                im.paste(charts[j].crop((0, 0, progress, 53)), (x, 560))
            label(draw, x, 506, f"Right forefoot peak   {rf}", 16, color=MUTED)
            if walk:
                start_x = record.pose[np.argmin(abs(record.t - start)), 0]
                label(
                    draw, x + PW, 119, f"Forward travel  {(q[0] - start_x) * 1000:.0f} mm", 18, right=True, color=MUTED
                )
            else:
                label(
                    draw,
                    x + PW,
                    117,
                    f"Target  {record.command[idx]:+.0f}°    Actual  {record.yaw[idx]:+.1f}°",
                    18,
                    right=True,
                    color=MUTED,
                )
                if final and args.program == "six-headings":
                    moves = record.score["segments"][1:]
                    passed = sum(s["first_1s_in_5deg_band_s"] is not None and s["completed"] for s in moves)
                    label(
                        draw, x + PW, 506, f"Headings held   {passed} / 6", 16, color=COLORS[j], right=True, bold=True
                    )
        return im

    if args.still is not None:
        frame(args.still, np.isclose(args.still, end)).save(args.out)
        return
    cmd = [
        args.ffmpeg,
        "-v",
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
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for k in range((end - start) * FPS):
            proc.stdin.write(frame(start + k / FPS).tobytes())
        final = frame(end, True).tobytes()
        for _ in range(2 * FPS):
            proc.stdin.write(final)
    finally:
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Video encoder failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", choices=[*ROUTINES, "detail"])
    parser.add_argument("--records", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--still", type=float, help="Render one frame at the given simulation time")
    args = parser.parse_args()
    if args.program != "detail" and args.records is None:
        parser.error("--records is required for motion")
    with ExitStack() as stack:
        exhibits = [Exhibit(f, args.program == "six-headings", stack) for f in FOLDERS]
        if args.program == "detail":
            motor_detail(exhibits).save(args.out)
        else:
            movie(exhibits, args)


if __name__ == "__main__":
    main()
