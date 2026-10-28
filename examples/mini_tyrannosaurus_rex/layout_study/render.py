"""Same-scale views from compiled geometry; no generated concept imagery."""

from __future__ import annotations

import itertools
import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from environments.shared.robot_rendering import camera_basis, font, geometry, render
from examples.mini_tyrannosaurus_rex.layout_study.build import HERE, LAYOUTS
from examples.mini_tyrannosaurus_rex.layout_study.validate import load

BG = (245, 247, 244)
INK = (34, 52, 43)
MUTED = (86, 99, 91)
COLORS = [(80, 122, 96), (80, 109, 136), (156, 114, 60)]


def main() -> None:
    models = [load(name) for name in LAYOUTS]
    report = json.loads((HERE / "comparison.json").read_text()) if (HERE / "comparison.json").exists() else None
    views = [
        ("SILHOUETTE ENVELOPE", (1.0, -1.9, 0.8), (1, 2)),
        ("HARDWARE · ENVELOPE HIDDEN", (1.0, -1.9, 0.8), (2, 4)),
        ("SIDE · FRONT TO THE RIGHT", (0.0, -1.0, 0.0), (1, 2)),
    ]
    sheet = Image.new("RGB", (1800, 1640), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((48, 32), "MINI / THREE COMPACT CHASSIS STUDIES", font=font(32), fill=INK)
    draw.text(
        (48, 79),
        "Full-size parts. Identical leg travel, feet, mass budget and future payload. All panels at the same scale.",
        font=font(18),
        fill=MUTED,
    )
    for col, (name, (_, _, layout)) in enumerate(zip(LAYOUTS, models, strict=True)):
        x = col * 600
        draw.rounded_rectangle((x + 28, 124, x + 572, 194), radius=12, fill=COLORS[col])
        draw.text((x + 47, 143), layout["label"], font=font(26), fill="white")
    view_bounds = None
    for row, (label, direction, groups) in enumerate(views):
        basis = camera_basis(direction)
        points = [geometry(m, d, groups=groups)[0] @ basis.T for m, d, _ in models]
        lo = np.min([p[:, :, :2].min(axis=(0, 1)) for p in points], axis=0)
        hi = np.max([p[:, :, :2].max(axis=(0, 1)) for p in points], axis=0)
        # Use the full silhouette framing for cutaways so no hardware view is
        # enlarged relative to its cover. Each row uses a common world extent.
        if row == 1:
            assert view_bounds is not None
            lo, hi = view_bounds
        else:
            center = (lo + hi) / 2
            lo, hi = center - [0.160, 0.155], center + [0.160, 0.155]
            view_bounds = (lo, hi)
        for col, (m, d, _) in enumerate(models):
            y = 218 + row * 392
            draw.text((col * 600 + 40, y), label, font=font(15), fill=MUTED)
            tile = render(m, d, direction=direction, groups=groups, width=580, height=355, projection_bounds=(lo, hi))
            sheet.paste(tile, (col * 600 + 10, y + 28))
    for col, (m, d, layout) in enumerate(models):
        x = 600 * col + 42
        dims = np.round(layout["silhouette_dimensions_mm"]).astype(int)
        lines = [
            f"Torso envelope: {dims[0]} L × {dims[1]} W × {dims[2]} H mm",
            f"Foot spacing: {2 * (layout['hip_yaw_half_spacing_mm'] + 30):.0f} mm",
        ]
        if report:
            r = report["layouts"][col]
            lines += [f"Loaded CoM height: {r['home']['loaded_com_xyz_mm'][2]:.0f} mm"]
        for line, yy in zip(lines, (1430, 1458, 1486), strict=False):
            draw.text((x, yy), line, font=font(18), fill=INK)
    draw.text(
        (45, 1540),
        "Orange: battery   Green: computer   Blue: interface / power / sensors   Purple: wiring bay   Dark: full-size motors",
        font=font(17),
        fill=MUTED,
    )
    draw.text(
        (45, 1572),
        "Rounded volumes show silhouette only. Shell openings, structure, moving cables and actual head/tail geometry remain to be designed.",
        font=font(17),
        fill=MUTED,
    )
    draw.text(
        (45, 1601),
        "Ten installed leg motors; the same 115 g future head / fixed-tail / fixed-arm load is present in each model but hidden in these views.",
        font=font(17),
        fill=MUTED,
    )
    sheet.save(HERE / "comparison.png")

    # A dimensioned plan and side packing sheet makes internal fit inspectable.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#f5f7f4")
    color = {
        "battery": "#de7529",
        "computer": "#267563",
        "harness_trunk": "#80548e",
        "u2d2": "#4686b2",
        "motor_regulator": "#9a7043",
        "logic_regulator": "#3a8f93",
        "power_distribution": "#b15160",
        "body_imu": "#ad9b32",
        "character_electronics": "#a675a5",
    }
    for col, (m, d, layout) in enumerate(models):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        for row, coords in enumerate(((0, 2), (0, 1))):
            ax = axes[row, col]
            ax.set_facecolor("#f5f7f4")
            for part in layout["parts"]:
                p = np.array(part["pos_mm"])[list(coords)]
                size = np.array(part["installed_mm"])[list(coords)]
                c = color.get(part["name"], "#527b9f")
                ax.add_patch(
                    Rectangle(tuple(p - size / 2), *size, facecolor=c, edgecolor=c, alpha=0.10, linestyle="--")
                )
                raw = np.array(part["raw_mm"])[list(coords)]
                ax.add_patch(Rectangle(tuple(p - raw / 2), *raw, facecolor=c, edgecolor=c, alpha=0.6))
            for g in range(m.ngeom):
                if not m.geom(g).name.endswith("_servo_case"):
                    continue
                signs = np.array(list(itertools.product((-1, 1), repeat=3)))
                vertices = signs * m.geom_size[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - [0, 0, 0.113]
                points = vertices[:, list(coords)] * 1000
                low, high = points.min(axis=0), points.max(axis=0)
                ax.add_patch(Rectangle(low, *(high - low), facecolor="#353e3d", alpha=0.3))
            ax.set_aspect("equal")
            ax.set_xlim(-105, 90)
            ax.set_ylim((-115, 165) if row == 0 else (-105, 105))
            ax.grid(alpha=0.18)
            ax.set_xlabel("front →   x / mm")
            ax.set_ylabel("height from pelvis / mm" if row == 0 else "width / mm")
            ax.set_title(layout["label"] + (" · side" if row == 0 else " · plan"), fontsize=12)
    fig.suptitle(
        "Installed part envelopes · identical scale · dashed reserves include provisional plug / mounting space",
        fontsize=14,
    )
    names = {
        "battery": "Battery",
        "computer": "Computer",
        "harness_trunk": "Wiring bay",
        "u2d2": "U2D2",
        "motor_regulator": "Motor regulator",
        "logic_regulator": "Logic regulator",
        "power_distribution": "Power protection",
        "body_imu": "Body IMU",
        "character_electronics": "Audio / LEDs",
    }
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, label=names[n]) for n, c in color.items()]
    handles.append(Rectangle((0, 0), 1, 1, facecolor="#353e3d", label="Motor cases"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    fig.text(
        0.05,
        0.02,
        "Projection overlaps can be separated in the third dimension. The 3D clearance report is authoritative for the declared screen.",
        fontsize=10,
    )
    fig.savefig(HERE / "packing.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
