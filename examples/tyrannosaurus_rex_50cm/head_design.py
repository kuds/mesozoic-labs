"""Head-led character alternatives with explicit mass and fixed hardware sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from . import build_model as base
from . import demo, validate
from .proportions import attrs, component, find

OUTPUT = base.HERE / "head_study"
SPECS: dict[str, dict[str, Any]] = {
    "large_head": dict(
        label="Large head / compact torso", scale=(1.40, 1.45, 1.55), neck=0.162, upper_g=220, jaw_g=60, structure_g=50
    ),
    "oversized_head": dict(
        label="Oversized head / compact torso",
        scale=(1.55, 1.60, 1.75),
        neck=0.177,
        upper_g=300,
        jaw_g=75,
        structure_g=60,
    ),
}


def build(name: str, directory: Path) -> Path:
    spec = SPECS[name]
    scale = np.array(spec["scale"])
    b = base.Builder()
    b.leg("left", 1)
    b.leg("right", -1)
    b.upper_body()
    b.xml.set("model", "tyrannosaurus_rex_" + name + "_study")
    attrs(b, "mesh", "torso_shell_mesh", scale=(0.100, 0.090, 0.080))
    attrs(b, "geom", "torso_shell", pos=(-0.012, 0, 0.020))
    attrs(b, "mesh", "belly_panel_mesh", scale=(0.026, 0.068, 0.071))
    attrs(b, "geom", "belly_panel", pos=(0.069, 0, 0.020))
    attrs(b, "mesh", "back_access_panel_mesh", scale=(0.022, 0.064, 0.064))
    attrs(b, "geom", "back_access_panel", pos=(-0.086, 0, 0.020))
    attrs(b, "geom", "torso_collision", size=(0.078, 0.072, 0.065), pos=(-0.012, 0, 0.020))
    for geom in b.torso.findall("geom"):
        if geom.get("name", "").startswith(("back_vent_", "back_screw_")):
            pos = np.fromstring(geom.get("pos", ""), sep=" ") + [-0.007, 0, -0.030]
            geom.set("pos", base.values(pos))
    component(
        b, "torso_shell_mass", mass_kg=0.130, pos_m=[-0.012, 0, 0.020], inertia_box_halfsize_m=[0.100, 0.090, 0.080]
    )
    component(b, "pelvis_structure", mass_kg=0.300)
    component(b, "compute_tray", pos_m=[-0.040, 0, 0.070])
    attrs(b, "geom", "compute_tray_envelope", pos=(-0.040, 0, 0.070))
    b.component(
        b.torso, "neck_support_allowance", "load_bearing_structure", 50, (-0.015, 0, 0.110), (0.030, 0.030, 0.055)
    )
    for side, sign in (("left", 1), ("right", -1)):
        shoulder = (0.025, sign * 0.080, 0.078)
        attrs(b, "geom", side + "_shoulder_case", pos=shoulder)
        attrs(b, "body", side + "_arm", pos=(0.025, sign * 0.097, 0.078))
        component(b, side + "_shoulder_motor", pos_m=list(shoulder))
        component(b, side + "_shoulder_mount", pos_m=list(shoulder))
        attrs(b, "mesh", side + "_thigh_fairing_mesh", scale=(0.037, 0.027, 0.048))
        component(b, side + "_leg_cover", mass_kg=0.035, inertia_box_halfsize_m=[0.037, 0.027, 0.048])
    neck_pos = [-0.015, 0, spec["neck"]]
    attrs(b, "body", "head_yaw_body", pos=neck_pos)
    attrs(b, "geom", "neck_yaw_case", pos=neck_pos)
    component(b, "head_yaw_motor", pos_m=neck_pos)
    neck_end = spec["neck"] - 0.020
    for collision in (False, True):
        attributes = dict(
            name="neck_support" + ("_collision" if collision else ""),
            type="capsule",
            fromto=(-0.015, 0, 0.085, -0.015, 0, neck_end),
            size=0.013,
        )
        if collision:
            attributes["class_"] = "collision"
        else:
            attributes.update(material="graphite", group=2)
        base.element(b.torso, "geom", **attributes)
    # The camera assembly translates as a rigid package. Its apertures,
    # baseline and hardware dimensions are not enlarged with the character.
    camera_shift = np.array([0.135 * (scale[0] - 1), 0, 0.006 * (scale[2] - 1)])
    camera_names = {"camera_package_envelope", "snout_camera_panel"}
    camera_names.update(
        "snout_" + label + suffix for label in ("left", "rgb", "right") for suffix in ("", "_rim", "_lens")
    )
    head = b.bodies["head"]
    for item in head.iter():
        if item is head or item.tag not in ("geom", "body", "camera"):
            continue
        item_name = item.get("name", "")
        rigid_camera = item_name in camera_names
        if item.get("pos"):
            pos = np.fromstring(item.get("pos", ""), sep=" ")
            pos = pos + camera_shift if rigid_camera else pos * scale
            # Recess the eyes into the broad skull instead of enlarging their
            # outward projection as if they were stalks.
            if any(part in item_name for part in ("eye_", "iris", "pupil", "brow")):
                pos[1] -= np.sign(pos[1]) * 0.008 * scale[1]
            item.set("pos", base.values(pos))
        if rigid_camera:
            continue
        if item.tag == "geom" and item.get("mesh"):
            mesh = find(b, "mesh", item.get("mesh", ""))
            mesh.set("scale", base.values(np.fromstring(mesh.get("scale"), sep=" ") * scale))
        elif item.tag == "geom" and item.get("size"):
            item.set("size", base.values(np.fromstring(item.get("size", ""), sep=" ") * scale))
    for c in b.components:
        if c["body"] not in ("head", "jaw"):
            continue
        c["pos_m"] = (np.array(c["pos_m"]) * scale).tolist()
        if c["name"] not in ("snout_camera", "jaw_motor", "head_bearings"):
            c["inertia_box_halfsize_m"] = (np.array(c["inertia_box_halfsize_m"]) * scale).tolist()
    component(b, "snout_camera", pos_m=(np.array([0.120, 0, 0.006]) + camera_shift).tolist())
    component(b, "head_upper_shell", mass_kg=spec["upper_g"] / 1000)
    component(b, "jaw_shell_mass", mass_kg=spec["jaw_g"] / 1000)
    component(b, "head_structure", mass_kg=spec["structure_g"] / 1000)
    component(b, "head_bearings", mass_kg=0.039)
    # Additional small side teeth are contained in the shell mass allowance.
    for side, sign in (("left", 1), ("right", -1)):
        b.geom(
            head,
            side + "_side_tooth",
            "ellipsoid",
            scale * [0.007, 0.005, 0.010],
            scale * [0.088, sign * 0.068, -0.020],
            "white",
        )
        b.geom(
            head,
            side + "_nostril",
            "ellipsoid",
            scale * [0.006, 0.004, 0.0015],
            scale * [0.113, sign * 0.035, 0.030],
            "dark_sage",
        )
    compiler = b.xml.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", os.path.relpath(base.HERE / "assets", directory))
    directory.mkdir(parents=True, exist_ok=True)
    b.finish(directory)
    return directory / "model.xml"


def head_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    groups = model.geom_group.copy()
    try:
        ids = (model.body("head").id, model.body("jaw").id)
        model.geom_group[~np.isin(model.geom_bodyid, ids)] = 5
        return validate.visual_bounds(model, data)
    finally:
        model.geom_group[:] = groups


def measure(path: Path) -> dict[str, Any]:
    model, data = demo.load_model(path=path)
    low, high = validate.visual_bounds(model, data)
    head_low, head_high = head_bounds(model, data)
    trials = {
        label: validate.run_trial(model, duration=10, gesture=gesture)
        for label, gesture in (("stance", False), ("gesture", True))
    }
    return dict(
        dimensions_m=(high - low).tolist(),
        head_dimensions_m=(head_high - head_low).tolist(),
        head_height_fraction=float((head_high[2] - head_low[2]) / (high[2] - low[2])),
        mass_kg=float(model.body_mass.sum()),
        com_m=data.subtree_com[model.body("torso").id].tolist(),
        head_yaw_mass_kg=float(model.body_subtreemass[model.body("head_yaw_body").id]),
        home_contacts=validate.internal_contacts(model, data),
        trials=trials,
    )


def render_sheet(directory: Path, report: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    from examples.mini_tyrannosaurus_rex.render_design import _camera_basis, _font, geometry

    from .render_design import render

    paths = [
        base.HERE / "proportion_study/balanced/model.xml",
        directory / "large_head/model.xml",
        directory / "oversized_head/model.xml",
    ]
    models = [demo.load_model(path=p) for p in paths]
    labels = ["Previous / fuller torso", "Large head / compact torso", "Oversized head / compact torso"]
    sheet = Image.new("RGB", (1800, 1480), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((40, 26), "T-REX / MAKE THE HEAD THE CHARACTER", font=_font(34), fill=(38, 55, 43))
    draw.text(
        (40, 80),
        "Actual MuJoCo geometry / same leg mechanism / estimated shell masses",
        font=_font(22),
        fill=(83, 97, 88),
    )
    for row, view in enumerate(("isometric", "side")):
        points = [geometry(m, d)[0] @ _camera_basis(view).T for m, d in models]
        bounds = (
            np.min([p[:, :, :2].min(axis=(0, 1)) for p in points], axis=0),
            np.max([p[:, :, :2].max(axis=(0, 1)) for p in points], axis=0),
        )
        for i, (model, data) in enumerate(models):
            x, y = 600 * i, 150 + row * 600
            sheet.paste(render(model, data, view=view, width=600, height=550, projection_bounds=bounds), (x, y + 28))
            draw.text((x + 28, y), labels[i], font=_font(22), fill=(38, 55, 43))
    for i, name in enumerate(("previous", "large_head", "oversized_head")):
        r = report[name]
        draw.text(
            (28 + 600 * i, 1360),
            f"{r['dimensions_m'][2] * 100:.1f} cm / head {r['head_height_fraction']:.0%} of height",
            font=_font(21),
            fill=(60, 76, 66),
        )
    draw.text(
        (40, 1425),
        "Proportion concepts. Collision proxies and assigned masses do not establish walking or hardware fit.",
        font=_font(20),
        fill=(83, 97, 88),
    )
    sheet.save(directory / "comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    reports = {"previous": measure(base.HERE / "proportion_study/balanced/model.xml")}
    reports.update({name: measure(build(name, args.output / name)) for name in SPECS})
    sources = [
        Path(__file__),
        base.HERE / "build_model.py",
        base.HERE / "proportions.py",
        base.HERE / "demo.py",
        base.HERE / "validate.py",
        base.HERE / "proportion_study/balanced/model.xml",
        *sorted((base.HERE / "assets").glob("*.obj")),
    ]
    result = dict(
        status="head_led_character_study_not_hardware_or_walking_validation",
        mujoco_version=mujoco.__version__,
        variants=reports,
        source_sha256={str(p.relative_to(base.HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "height": r["dimensions_m"][2],
                    "head": r["head_dimensions_m"],
                    "mass": r["mass_kg"],
                    "head_mass": r["head_yaw_mass_kg"],
                    "contacts": r["home_contacts"],
                    "trials": {
                        k: {
                            "pass": t["passed"],
                            "leg_rms": max(t["rms_torque_nm"][:12]),
                            "head_peak": max(t["peak_torque_nm"][15:17]),
                            "penetration": t["maximum_internal_penetration_m"],
                            "excursion": t["joint_excursion_rad"][15:],
                        }
                        for k, t in r["trials"].items()
                    },
                }
                for name, r in reports.items()
            },
            indent=2,
        )
    )
    if args.render:
        render_sheet(args.output, reports)


if __name__ == "__main__":
    main()
