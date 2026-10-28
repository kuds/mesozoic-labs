"""Same-scale compiled-model views and a replay of measured simulation poses."""

from __future__ import annotations

import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from environments.shared.robot_rendering import font, render
from examples.mini_tyrannosaurus_rex.leg_study.build import HERE, VARIANTS
from examples.mini_tyrannosaurus_rex.leg_study.validate import alpha_for_drop, load, set_fold

BG = (245, 247, 244)
INK = (34, 52, 43)
MUTED = (86, 99, 91)
COLORS = [(72, 116, 88), (65, 100, 135)]
BOUNDS = (np.array([-0.135, -0.012]), np.array([0.135, 0.250]))


def side_tile(model: mujoco.MjModel, data: mujoco.MjData, width: int, height: int) -> Image.Image:
    tile = render(model, data, direction=(0, -1, 0), width=width, height=height, projection_bounds=BOUNDS)
    draw = ImageDraw.Draw(tile)
    lo, hi = BOUNDS
    scale = min(width * 0.84 / (hi[0] - lo[0]), height * 0.82 / (hi[1] - lo[1]))

    def project(point: np.ndarray) -> tuple[float, float]:
        p = (point[[0, 2]] - (lo + hi) / 2) * [scale, -scale] + [width / 2, height / 2]
        return float(p[0]), float(p[1])

    ground = project(np.array([0.0, 0.0, 0.0]))[1]
    draw.line((30, ground, width - 30, ground), fill=(159, 169, 160), width=2)
    # Camera looks from -Y; show the near/right leg's actual pivot positions.
    chain = [project(data.xpos[model.body(f"right_{b}").id]) for b in ("thigh", "shin", "foot")]
    draw.line(chain, fill=(225, 133, 45), width=4)
    for x, y in chain:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(250, 213, 161), outline=INK, width=1)
    draw.line((40, height - 24, 40 + 0.05 * scale, height - 24), fill=MUTED, width=3)
    draw.text((42, height - 50), "50 mm", font=font(15), fill=MUTED)
    draw.text((width - 138, height - 43), "FRONT →", font=font(17), fill=MUTED)
    return tile


def comparison(report: dict) -> None:
    sheet = Image.new("RGB", (1760, 1550), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((42, 30), "TORSO A / TWO LEG DIRECTIONS", font=font(34), fill=INK)
    draw.text(
        (42, 80),
        "Identical torso, full-size motors, 43.7 mm links and forward-facing feet. Same scale in each row.",
        font=font(20),
        fill=MUTED,
    )
    for col, (name, sign) in enumerate(VARIANTS.items()):
        x = 880 * col
        draw.rounded_rectangle((x + 28, 125, x + 852, 188), radius=10, fill=COLORS[col])
        draw.text((x + 50, 140), name.upper() + "-FOLDING", font=font(29), fill="white")
        model, data = load(name)
        for row, alpha in enumerate((0, alpha_for_drop(30))):
            set_fold(model, data, sign, alpha)
            top = 215 + row * 430
            label = "NEUTRAL / DELIVERED MODEL" if row == 0 else "30 mm LOWER / KINEMATIC TARGET"
            draw.text((x + 40, top), label, font=font(18), fill=MUTED)
            sheet.paste(side_tile(model, data, 840, 400), (x + 20, top + 25))
        set_fold(model, data, sign, alpha_for_drop(30))
        tile = render(
            model,
            data,
            direction=(1, -1.9, 0.8),
            groups=(2, 4),
            width=520,
            height=330,
            projection_bounds=(np.array([-0.14, -0.06]), np.array([0.14, 0.22])),
        )
        sheet.paste(tile, (x + 5, 1090))
        draw.text((x + 40, 1085), "HARDWARE / COVER HIDDEN", font=font(16), fill=MUTED)
        r = report["variants"][col]
        status = r["crouching"][3]["status"].upper()
        lines = ["Loaded simulation", "30 s stand: PASS", "12 mm crouch: PASS", f"30 mm cycle: {status}"]
        for i, label in enumerate(lines):
            draw.text(
                (x + 510, 1130 + i * 37), label, font=font(20), fill=INK if status == "PASS" or i < 3 else (164, 61, 49)
            )
    draw.text(
        (42, 1440),
        "Orange lines mark actual hip, knee and ankle pivots. Rounded torso is a silhouette guide, not finished shell CAD.",
        font=font(18),
        fill=MUTED,
    )
    draw.text(
        (42, 1472),
        "The 30 mm poses require wider joint limits. They show geometry only; the rearward loaded cycle fails on the rise.",
        font=font(18),
        fill=MUTED,
    )
    draw.text(
        (42, 1504),
        "754.1 g loaded allocation · 10 full 20 × 34 × 26 mm motor cases · 115 g fixed future payload · tail stays fixed",
        font=font(18),
        fill=MUTED,
    )
    sheet.save(HERE / "comparison.png")


def charts(report: dict, samples: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#f5f7f4")
    colors = ["#487458", "#416487"]
    for i, name in enumerate(VARIANTS):
        r = report["variants"][i]
        pts = np.array(samples[name]["workspace_mm"])
        axes[i].scatter(pts[:, 0], pts[:, 1], s=3, color=colors[i], alpha=0.30)
        axes[i].axhline(25, color="#b55d3b", ls="--", lw=1, label="25 mm lift reference")
        axes[i].set(
            xlim=(-45, 45),
            ylim=(-5, 30),
            title=name.title() + " · flat-foot ankle reach",
            xlabel="rear ← ankle offset / mm → front",
            ylabel="ankle lift from home / mm",
        )
        axes[i].text(-42, 27, "25 mm target", color="#b55d3b", fontsize=9)
        axes[i].grid(alpha=0.15)
        trial = r["crouching"][3]
        trajectory = samples[name]["trajectories"][3]
        times = [row["t_s"] for row in trajectory]
        drop = [(0.113 - row["qpos"][2]) * 1000 for row in trajectory]
        axes[2].plot(times, drop, color=colors[i], label=f"{name.title()} · {trial['status']}")
        axes[2].plot(times[-1], drop[-1], "o", color=colors[i])
    t = np.linspace(0, 14, 281)
    phase = np.where(t < 8, np.clip((t - 2) / 4, 0, 1), 1 - np.clip((t - 8) / 4, 0, 1))
    alpha = alpha_for_drop(30) * phase**3 * (10 - 15 * phase + 6 * phase**2)
    axes[2].plot(
        t, 85 - 2 * (42.5 * np.cos(alpha) - 10 * np.sin(alpha)), "--", color="#899188", label="Kinematic target"
    )
    axes[2].set(
        title="30 mm loaded cycle · actual simulation", xlabel="time / seconds", ylabel="pelvis drop / mm", xlim=(0, 14)
    )
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.15)
    fig.suptitle("Changing fold direction does not enlarge the joint-limited reach envelope", fontsize=15)
    fig.text(
        0.04,
        0.025,
        "Reach plots are kinematic only; not validated footsteps. Dynamic trial uses identical ideal PD / 0.10 Nm caps; rearward run stops at 60° tilt.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(HERE / "reach_and_motion.png", dpi=140)
    plt.close(fig)


def replay(report: dict, samples: dict) -> None:
    frames = []
    models = {name: load(name) for name in VARIANTS}
    for time in np.arange(0, 14.001, 0.5):
        frame = Image.new("RGB", (1120, 710), BG)
        draw = ImageDraw.Draw(frame)
        draw.text((27, 16), "30 mm CROUCH / LOADED SIMULATION REPLAY", font=font(24), fill=INK)
        draw.text(
            (27, 51), f"t = {time:4.1f} s   ·   Wider limits · Same controller · 0.10 Nm cap", font=font(18), fill=MUTED
        )
        for col, name in enumerate(VARIANTS):
            model, data = models[name]
            trajectory = samples[name]["trajectories"][3]
            row = min(trajectory, key=lambda p: abs(p["t_s"] - time))
            data.qpos[:] = row["qpos"]
            mujoco.mj_forward(model, data)
            tile = side_tile(model, data, 550, 490)
            frame.paste(tile, (col * 560 + 5, 132))
            draw.text((col * 560 + 28, 101), name.upper(), font=font(22), fill=COLORS[col])
            stopped = time > report["variants"][col]["crouching"][3]["completed_duration_s"]
            if stopped:
                draw.text((col * 560 + 28, 625), "STOPPED: 60° tilt on rise", font=font(21), fill=(164, 61, 49))
            elif time > 12:
                draw.text((col * 560 + 28, 625), "Returned to standing", font=font(21), fill=INK)
        draw.text(
            (27, 676),
            "Recorded free-root poses; no support fixtures. Head/tail load is included but its mass markers are hidden.",
            font=font(16),
            fill=MUTED,
        )
        frames.append(frame)
    frames[0].save(
        HERE / "loaded_30mm.gif", save_all=True, append_images=frames[1:], duration=500, loop=0, optimize=True
    )


def main() -> None:
    report = json.loads((HERE / "comparison.json").read_text())
    samples = json.loads((HERE / "samples.json").read_text())
    comparison(report)
    charts(report, samples)
    print("Static figures rendered", flush=True)
    replay(report, samples)
    print("Recorded simulation replay rendered", flush=True)


if __name__ == "__main__":
    main()
