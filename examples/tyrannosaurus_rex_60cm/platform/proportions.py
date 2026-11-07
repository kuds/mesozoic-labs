"""Measure the actual compiled plant at geometric home and render at common scale."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from environments.shared.robot_rendering import font, render
from examples.tyrannosaurus_rex_60cm.platform.build import DESIGN, HERE, M1_DESIGN

OUT = HERE.parents[2] / "hardware/tyrannosaurus_rex_60cm/platform/mechanics"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bbox(model, data, names):
    lo, hi = [], []
    for name in names:
        geom = model.geom(name).id
        rotation = data.geom_xmat[geom].reshape(3, 3)
        # Compiled local bounding boxes include half of each end-cap. Rotating
        # this box is conservative for cylinders and rods off the world axes.
        center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
        half = abs(rotation) @ model.geom_aabb[geom, 3:]
        lo.append(center - half)
        hi.append(center + half)
    return np.min(lo, axis=0), np.max(hi, axis=0)


def sample(name: str):
    source = HERE / name
    model = mujoco.MjModel.from_xml_path(str(source / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ledger = json.loads((source / "components.json").read_text())
    # The torso excludes legs, head, camera boom, buttress to tail pivot and arms.
    # Include upper and lower limits of all installed service bays/covers.
    core = [
        model.geom(i).name
        for i in range(model.ngeom)
        if model.geom(i).name.startswith(
            (
                "spine_",
                "rear_battery_",
                "pelvis_web_",
                "rib_mount_",
                "side_panel_",
            )
        )
    ] + [
        name + "_reserve"
        for name in ("battery", "compute", "control", "power", "regulated_12v_ttl", "drive_disconnect")
    ]
    bottom, top = bbox(model, data, core)
    cases = [n for n in ledger["motor_cases"] if n.startswith(("left", "right"))]
    connects = [n for n in ledger["connector_owners"] if n.startswith(("left", "right"))]
    lower, upper = bbox(model, data, cases + connects)
    com = data.subtree_com[model.body("pelvis").id]
    tensor = np.zeros((3, 3))
    for i in range(1, model.nbody):
        rot = data.ximat[i].reshape(3, 3)
        mass = model.body_mass[i]
        lever = data.xipos[i] - com
        tensor += rot @ np.diag(model.body_inertia[i]) @ rot.T + mass * (
            (lever @ lever) * np.eye(3) - np.outer(lever, lever)
        )

    # Joint centers and feet are measured along world axes in a named pose.
    def joint_pos(label):
        return data.xanchor[model.joint(label).id]

    hip = joint_pos("left_hip_pitch")
    knee = joint_pos("left_knee")
    ankle = joint_pos("left_ankle_pitch")
    foot = data.site("left_sole_center").xpos
    sole = model.geom("left_sole").size
    outputs = {
        "model_sha256": hash_file(source / "model.xml"),
        "ledger_sha256": hash_file(source / "components.json"),
        "height_mm": 1000 * data.site("head_crown").xpos[2],
        "length_mm": 1000 * (data.site("snout").xpos[0] - data.site("tail_tip").xpos[0]),
        "leg_hardware_width_mm": 1000 * (upper[1] - lower[1]),
        "torso_bay_LWH_mm": (1000 * (top - bottom)).tolist(),
        "crown_to_hardware_width_ratio": data.site("head_crown").xpos[2] / (upper[1] - lower[1]),
        "hip_height_mm": 1000 * hip[2],
        "hip_pitch_spacing_mm": 2000 * hip[1],
        "foot_center_spacing_mm": 2000 * foot[1],
        "hip_height_to_stance": hip[2] / (2 * foot[1]),
        "sole_LW_mm": (2000 * sole[:2]).tolist(),
        "thigh_shin_axis_lengths_mm": (
            1000 * np.array([np.linalg.norm(hip - knee), np.linalg.norm(knee - ankle)])
        ).tolist(),
        "mass_kg": float(model.body_mass.sum()),
        "com_world_mm": (1000 * com).tolist(),
        "inertia_about_com_kg_m2": tensor.tolist(),
        "left_foot_min_lateral_com_shift_mm": 1000 * (foot[1] - sole[1] - com[1]),
        "moving_head_kg": float(model.body_mass[model.body("head").id]),
        "moving_tail_kg": float(model.body_mass[model.body("tail").id]),
        "fixed_pelvis_mass_fraction": float(model.body_mass[model.body("pelvis").id] / model.body_mass.sum()),
    }
    return outputs, model, data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = HERE.parents[2]
    inputs = [
        Path(__file__),
        HERE / "build.py",
        DESIGN,
        M1_DESIGN,
        HERE.parent / "build.py",
        root / "environments/shared/robot_building.py",
        root / "environments/shared/robot_rendering.py",
    ]
    report = {
        "inputs_sha256": {str(p.relative_to(root)): hash_file(p) for p in inputs},
        "scope": "Geometric zero, soles tangent to ground; core torso body includes bays, longitudinal rails and removable side covers. Leg hardware width includes motor cases and their connector keepouts. Mass and inertia are allocated.",
        "variants": {},
    }
    card = Image.new("RGB", (1500, 665), "#f5f7f4")
    draw = ImageDraw.Draw(card)
    draw.text((28, 16), "Intermediate T. rex   /   proportion candidates", font=font(32), fill="#203725")
    draw.text(
        (29, 66),
        "Geometric home   ·   matched camera, projection and scale within each row   ·   full-size hardware",
        font=font(16),
        fill="#465c4e",
    )
    for col, name in enumerate(("compact", "nominal", "wide")):
        metrics, m, d = sample(name)
        report["variants"][name] = metrics
        for y, view, bounds in (
            (130, (0, -1, 0), ([-0.66, -0.01], [0.50, 0.63])),
            (375, (0.75, -1, 0.30), ([-0.55, -0.14], [0.53, 0.70])),
        ):
            card.paste(
                render(
                    m,
                    d,
                    direction=view,
                    groups=(1, 2, 4),
                    width=500,
                    height=240,
                    projection_bounds=tuple(map(np.array, bounds)),
                ),
                (col * 500, y),
            )
        x = 30 + col * 500
        draw.text((x, 98), name.capitalize(), font=font(21), fill="#203725")
        draw.text(
            (x, 620),
            f"{metrics['mass_kg']:.2f} kg    ·    {metrics['hip_height_mm']:.0f} mm hip    ·    {metrics['leg_hardware_width_mm']:.0f} mm installed leg width",
            font=font(15),
            fill="#465c4e",
        )
    card.save(OUT / "proportions.png")
    report["render_sha256"] = hash_file(OUT / "proportions.png")
    (OUT / "proportions.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
