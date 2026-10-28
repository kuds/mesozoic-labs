"""Render actual model geometry with clearly separated future silhouette guides."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from environments.shared.robot_rendering import camera_basis, font, geometry, render
from examples.tyrannosaurus_rex_60cm.build import HARDWARE, HERE
from examples.tyrannosaurus_rex_60cm.validate import load


def main() -> None:
    model, data = load()
    report = json.loads((HARDWARE / "validation.json").read_text())
    sheet = Image.new("RGB", (1800, 1130), "#f5f7f4")
    draw = ImageDraw.Draw(sheet)
    ink, secondary = "#23372e", "#52665b"
    draw.text((45, 24), "T. REX / 60 CM SKELETON", font=font(38), fill=ink)
    draw.text(
        (45, 77),
        "Concept r01  •  Actual MuJoCo geometry  •  Forward knees  •  Fixed tail",
        font=font(22),
        fill=secondary,
    )
    draw.line((45, 116, 1755, 116), fill="#b8c6bb", width=2)
    draw.text((45, 140), "01  Chassis and future silhouette", font=font(25), fill=ink)
    sheet.paste(render(model, data, direction=(0.65, -1, 0.40), groups=(1, 2, 4), width=1120, height=610), (20, 180))
    draw.text((1180, 160), "Packaging decisions", font=font(28), fill=ink)
    lines = [
        "600 mm to the future head crown",
        "360 mm to the hip pitch axes",
        "~1.04 m nose to fixed-tail tip",
        "352 mm across motor cases at home",
        "160 × 88 mm flat test soles",
        "12 full-size leg actuators",
        f"{report['balance']['mass_kg']:.2f} kg allocated, including reserve",
        "",
        "Blue: complete computer-kit envelope",
        "Amber: full-size battery body",
        "Green: control and power allocations",
        "Pale outline: future silhouette only",
        "",
        "No tail, head, jaw or arm actuation.",
        "Mounts and moving cables need CAD.",
    ]
    for i, line in enumerate(lines):
        draw.text((1180, 211 + i * 31), line, font=font(21), fill=secondary)
    draw.line((45, 775, 1755, 775), fill="#b8c6bb", width=2)
    for col, (label, qpos) in enumerate((("02  Side / home", model.key_qpos[0]), ("03  Side / recorded crouch", None))):
        if qpos is None:
            record = np.load(HARDWARE / "crouch_trajectory.npz")
            assert str(record["model_sha256"]) == hashlib.sha256((HERE / "model.xml").read_bytes()).hexdigest()
            qpos = record["qpos"][int(7 / float(record["dt"]))]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        side_tri, _, _ = geometry(model, data, groups=(1, 2, 4))
        projected = side_tri @ camera_basis((0, -1, 0)).T
        bounds = (np.array([-0.65, 0]), np.array([0.47, 0.63]))
        # Side camera's horizontal direction is +x, so shared bounds preserve scale.
        assert projected[:, :, 0].min() > -0.70
        sheet.paste(
            render(
                model, data, direction=(0, -1, 0), groups=(1, 2, 4), width=850, height=290, projection_bounds=bounds
            ),
            (45 + col * 880, 815),
        )
        draw.text((45 + col * 880, 790), label, font=font(23), fill=ink)
        scale = min(850 * 0.84 / 1.12, 290 * 0.82 / 0.63)
        xy = (data.subtree_com[1, [0, 2]] - (bounds[0] + bounds[1]) / 2) * [scale, -scale]
        x, y = xy + [45 + col * 880 + 425, 815 + 145]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#b44431")
        draw.text((x - 53, y - 14), "CoM", font=font(16), fill="#9c392b")
    draw.text(
        (45, 1100),
        "Engineering study: component allocations and sampled motion checks; no learned walking or manufacturing qualification.",
        font=font(18),
        fill=secondary,
    )
    sheet.save(HARDWARE / "preview.png")

    # Larger uncluttered side/front views make the service layout reviewable.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    details = Image.new("RGB", (1800, 950), "#f5f7f4")
    pen = ImageDraw.Draw(details)
    pen.text((40, 25), "Exposed hardware / torso and legs first", font=font(32), fill=ink)
    for col, (title, direction) in enumerate((("Side", (0, -1, 0)), ("Front", (1, 0, 0)))):
        details.paste(
            render(model, data, direction=direction, groups=(2, 4), width=870, height=700), (30 + col * 880, 100)
        )
        pen.text((45 + col * 880, 90), title, font=font(25), fill=ink)
    pen.text(
        (45, 835),
        "All 12 motor bodies use Ø60 × 64 mm planning cases around the manufacturer's Ø55 × 61 mm actuator.",
        font=font(22),
        fill=secondary,
    )
    pen.text(
        (45, 875),
        "Installation and connector reserves are checked separately; structural rails are routing concepts, not fabricated brackets.",
        font=font(21),
        fill=secondary,
    )
    details.save(HARDWARE / "hardware_layout.png")
    sources = [HERE / "model.xml", HARDWARE / "validation.json", HARDWARE / "crouch_trajectory.npz", Path(__file__)]
    (HARDWARE / "render_manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "image_sha256": {
                    name: hashlib.sha256((HARDWARE / name).read_bytes()).hexdigest()
                    for name in ("preview.png", "hardware_layout.png")
                },
                "basis": "Orthographic rendering of compiled MJCF geometry; pale outline is a future silhouette guide.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
