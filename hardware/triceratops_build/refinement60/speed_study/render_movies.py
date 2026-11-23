"""Replay the speed survey's matched pose and 1 kHz foot-force recordings.

python speed_study/render_movies.py walk --records /tmp/records --out walk.mp4
Pass --record-missing to simulate any missing inputs. Paths in media/stages.json are relative to --records.
Optional graphics dependencies are Pillow, MuJoCo offscreen rendering and H.264 ffmpeg.
"""

import argparse
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mid160"))
from render import COLORS, FOLDERS, FPS, MUTED, PW, SIZES, WIDTH, Exhibit, X, header, label  # noqa: E402

HEIGHT = 718
CONTACT_N = 5.0


class Run:
    def __init__(self, stem, model, routine, drive, seed, footfall):
        self.score = json.loads(stem.with_suffix(".json").read_text())
        settings = self.score["settings"]
        walk = routine == "walk"
        rehearsed = routine == "rehearsed-course"
        param = settings.get("walk_cadence", 1.0) if walk or rehearsed else settings.get("speed_mps", 0.25)
        if (
            settings["program"] != ("six-headings" if rehearsed else routine)
            or settings["seed"] != seed
            or (not rehearsed and Path(settings["model"]).parent.name != model.folder)
            or (rehearsed and self.score["upper_mm"] != dict(zip(FOLDERS, SIZES))[model.folder])
            or not np.isclose(drive, param)
            or (
                not (walk or rehearsed)
                and (settings["turn_profile"] != "gentle" or settings.get("steering_footfall", "diagonal") != footfall)
            )
        ):
            raise ValueError(f"Recording and stage settings differ: {stem}")
        with np.load(stem.with_suffix(".npz")) as trace:
            self.t, self.poses = trace["t"], trace["qpos"]
            self.force_t, self.forces = trace["physics_t_s"], trace["physics_foot_load_N"]
            if not walk:
                self.command, self.yaw = trace["command_deg"], np.degrees(trace["yaw_rad"])
        if self.poses.shape != (len(self.t), model.model.nq) or not np.allclose(np.diff(self.force_t), 0.001):
            raise ValueError(f"Invalid recording time base or model: {stem}")
        if walk:
            self.mean_mps = self.score["speed_mps"]
        else:
            segments = self.score["segments"]
            durations = np.array([s["observed_until_s"] - s["start_s"] for s in segments])
            self.mean_mps = np.average([s["projected_mean_speed_mps"] for s in segments], weights=durations)
            self.held = sum(s["completed"] and s["first_1s_in_5deg_band_s"] is not None for s in segments[1:])
        self.peak = max(self.score["peak_by_foot_N"] if rehearsed else self.score["peak_foot_N"])
        if rehearsed:
            self.mean_mps = self.score["measured_forward_mm_s"] / 1000
        self.contact_threshold = (
            0.05 * model.model.body_mass.sum() * 9.81 if footfall in ("four-beat", "rehearsed") else CONTACT_N
        )
        if footfall == "four-beat":
            counts = (self.forces[self.force_t > 3] > self.contact_threshold).sum(1)
            self.three_contact = 100 * np.mean(counts >= 3)

    def ribbon(self, draw, x0, t0, t1, color):
        x = x0 + 40
        width = PW - 41
        for foot, abbr in enumerate(("LF", "RF", "LH", "RH")):
            top = 579 + foot * 17
            label(draw, x0, top - 2, abbr, 13, color=MUTED, bold=True)
            draw.rectangle((x, top, x + width, top + 10), fill="#e7eadf")
            # Intervals preserve subframe contact transitions while the movie runs at 24 fps.
            times = self.force_t
            mask = (times >= t0) & (times <= t1) & (self.forces[:, foot] > self.contact_threshold)
            edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
            for start, stop in edges.reshape((-1, 2)):
                left = x + round((times[start] - t0) / (t1 - t0) * width)
                right = x + round((times[stop - 1] - t0) / (t1 - t0) * width)
                draw.rectangle((left, top, max(left, right), top + 10), fill=color)


def record_inputs(program, footfall, stages, destination):
    for stage in stages:
        for mm, folder in zip(SIZES, FOLDERS):
            stem = destination / stage["record"].format(mm=mm)
            if stem.with_suffix(".npz").exists() and stem.with_suffix(".json").exists():
                continue
            if program == "rehearsed-course":
                from choreography import trial

                trial((mm, stage["seed"], False, stem.parents[1], stage["setting"]))
                continue
            root = HERE.parent
            options = (
                ["--walk-cadence", str(stage["setting"])]
                if program == "walk"
                else ["--turn-profile", "gentle", "--steering-footfall", footfall, "--speed-mps", str(stage["setting"])]
            )
            subprocess.run(
                [
                    sys.executable,
                    str(root / "motion/run.py"),
                    program,
                    "--seed",
                    str(stage["seed"]),
                    "--record",
                    "--model",
                    str(root / "models" / folder / "physics.xml"),
                    "--out",
                    str(stem),
                    *options,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )


def rehearsed_passes(paths, setting, upper_mm):
    """Score each six-run set independently of the nominal recording on screen."""
    rows = [
        row
        for path in paths
        for row in json.loads((HERE / path).read_text())["runs"]
        if row["upper_mm"] == upper_mm and row["cadence"] == setting
    ]
    if not rows:
        return None
    passed = sum(
        row["commands_complete"]
        and len(row["holds_s"]) == 6
        and all(t is not None for t in row["holds_s"])
        and row["upright"]
        and row["min_support_margin_mm"] >= 10
        and row["strict_four_beat"]
        for row in rows
    )
    return f"{passed}/{len(rows)}"


def movie(args):
    specification = json.loads((HERE / "media/stages.json").read_text())[args.film]
    routine = specification["program"]
    if args.record_missing:
        record_inputs(routine, specification.get("footfall"), specification["stages"], args.records)
    walk = routine == "walk"
    title = "Four-beat walking speed" if walk else "Steering at increasing speeds"
    if args.film == "fourbeat-turns":
        title = "Four-beat steering through six headings"
    elif args.film == "rehearsed-speeds":
        title = "Six-heading exhibit speed trials"
    elif args.film == "rehearsed-course":
        title = "Rehearsed six-heading exhibit walk"
    elif not walk:
        title = "Diagonal steering at increasing speeds"
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
    with ExitStack() as stack:
        exhibits = [Exhibit(folder, not walk, stack) for folder in FOLDERS]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for stage in specification["stages"]:
                t0, t1 = stage["window_s"]
                drive, seed = stage["setting"], stage["seed"]
                runs = [
                    Run(
                        args.records / stage["record"].format(mm=mm),
                        model,
                        routine,
                        drive,
                        seed,
                        specification.get("footfall"),
                    )
                    for mm, model in zip(SIZES, exhibits)
                ]
                if any(run.t[-1] < t0 - 0.001 for run in runs):
                    raise ValueError("Stage starts after a trial has ended")
                setting = (
                    f"Cadence  {drive:g}×" if walk or routine == "rehearsed-course" else f"Command  {drive:.2f} m/s"
                )
                subtitle = f"{setting}    /    {stage['name']}    /    Real-time simulation"
                base = header(title, subtitle, HEIGHT)
                d = ImageDraw.Draw(base)
                for x, run, color in zip(X, runs, COLORS):
                    speed = (
                        f"{1000 * run.mean_mps:.1f} mm/s"
                        if walk or routine == "rehearsed-course"
                        else f"{run.mean_mps:.3f} m/s"
                    )
                    label(d, x + PW, 118, f"Mean forward speed   {speed}", 17, color=color, right=True, bold=True)
                    label(d, x, 540, f"Scored peak on any foot  {run.peak:.0f} N", 17, bold=True)
                    if walk:
                        status = f"Support margin  {run.score['min_support_margin_mm']:.1f} mm"
                    else:
                        status = f"Headings held across run  {run.held}/6"
                        if routine == "rehearsed-course":
                            count = run.score["swing_count"]
                            status = f"Headings  {run.held}/6    Clean swings  {count - len(run.score['failed_swings'])}/{count}"
                    label(d, x + PW, 540, status, 17, right=True, color=color, bold=True)
                    if args.film == "rehearsed-speeds":
                        mm = run.score["upper_mm"]
                        main = rehearsed_passes(specification["main_results"], drive, mm)
                        fresh = rehearsed_passes(specification["holdout_results"], drive, mm)
                        note = f"Complete course passes     Main {main}"
                        if fresh is not None:
                            note += f"     Fresh seeds {fresh}"
                        label(d, x, 561, note, 14, color=color, bold=True)
                    run.ribbon(d, x, t0, t1, color)
                    label(d, x + 40, 651, f"{t0:g} s", 14, color=MUTED)
                    label(d, x + PW, 651, f"{t1:g} s", 14, color=MUTED, right=True)
                threshold = "5% body weight" if specification.get("footfall") in ("four-beat", "rehearsed") else "5 N"
                label(
                    d,
                    X[0],
                    686,
                    f"Shaded intervals show foot contact above {threshold}. Left/right forefeet (LF/RF), then hindfeet (LH/RH).",
                    16,
                    color=MUTED,
                )
                for k in range(round((t1 - t0) * FPS)):
                    t = t0 + k / FPS
                    canvas = base.copy()
                    d = ImageDraw.Draw(canvas)
                    label(d, WIDTH - 24, 67, f"Trial  {t:05.2f} s", 18, color=MUTED, right=True)
                    for x, model, run in zip(X, exhibits, runs):
                        idx = int(np.argmin(abs(run.t - t)))
                        q = run.poses[idx]
                        target = (q[0] - 0.05, 0 if walk else q[1], 0.30)
                        view = (1.49, 96 if walk else 102, -7 if walk else -19)
                        if routine == "rehearsed-course":
                            window = np.abs(run.t - t) <= 0.7
                            xy = run.poses[window, :2].mean(axis=0)
                            target = (xy[0] - 0.05, xy[1], 0.30)
                        if stage.get("reframe_fall"):
                            target = (q[0] - 0.05, q[1], 0.24)
                            view = (1.73, 96, -16)
                        canvas.paste(model.pose(q, target, view), (x, 151))
                        if t > run.t[-1] + 0.02:
                            sign = f"Trial ended at {run.t[-1]:.2f} s"
                        elif walk:
                            sign = (
                                f"{'Walk checks passed' if run.score['pass_all'] else 'Walk checks failed'}     "
                                f"Worst slide  {run.score['max_slide_mm']:.1f} mm"
                            )
                        else:
                            sign = f"Requested bearing  {run.command[idx]:+.0f}°     Measured  {run.yaw[idx]:+.1f}°"
                        label(d, x, 507, sign, 17, color=MUTED)
                        if routine == "rehearsed-course":
                            label(
                                d,
                                x + PW,
                                507,
                                f"3+ feet loaded after 3 s   {run.score['three_loaded_pct']:.0f}%",
                                16,
                                color=MUTED,
                                right=True,
                            )
                        if specification.get("footfall") == "four-beat":
                            label(
                                d,
                                x + PW,
                                507,
                                f"3+ feet loaded after 3 s   {run.three_contact:.0f}%",
                                16,
                                color=MUTED,
                                right=True,
                            )
                        cursor = x + 40 + round((t - t0) / (t1 - t0) * (PW - 41))
                        d.line((cursor, 577, cursor, 642), fill="#192d24", width=2)
                    process.stdin.write(canvas.tobytes())
        finally:
            process.stdin.close()
            if process.wait():
                raise RuntimeError("ffmpeg encoding failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("film", choices=("walk", "steering", "fourbeat-turns", "rehearsed-course", "rehearsed-speeds"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--record-missing", action="store_true", help="Simulate missing input recordings before rendering."
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    movie(parser.parse_args())
