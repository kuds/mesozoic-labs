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
    "rounded_box": dict(
        label="Rounded-box skull / compact torso",
        scale=(1.40, 1.45, 1.55),
        neck=0.185,
        upper_g=220,
        jaw_g=60,
        structure_g=50,
    ),
    "long_oval": dict(
        label="Long oval skull / compact torso",
        scale=(1.40, 1.45, 1.55),
        neck=0.185,
        upper_g=220,
        jaw_g=60,
        structure_g=50,
    ),
    "soft_wedge": dict(
        label="Soft wedge skull / compact torso",
        scale=(1.40, 1.45, 1.55),
        neck=0.185,
        upper_g=220,
        jaw_g=60,
        structure_g=50,
    ),
}


def unified_head(b: base.Builder, shape: str) -> None:
    """One upper skull, side eyes and a separate deep jaw in three shapes.

    Hardware sizes and the 530 g moving-head target are retained. Component
    centers and inertia boxes follow the revised skull and rear jaw hinge.
    """
    head = b.bodies["head"]
    # Leave room for the deeper jaw during the standing arm gesture.
    for side, sign in (("left", 1), ("right", -1)):
        shoulder = (0.025, sign * 0.080, 0.072)
        attrs(b, "geom", side + "_shoulder_case", pos=shoulder)
        attrs(b, "body", side + "_arm", pos=(0.025, sign * 0.097, 0.072))
        component(b, side + "_shoulder_motor", pos_m=list(shoulder))
        component(b, side + "_shoulder_mount", pos_m=list(shoulder))
    for name in ("upper_snout_shell", "upper_snout_collision", "left_brow", "right_brow"):
        head.remove(find(b, "geom", name))
    attrs(b, "mesh", "head_shell_mesh", file="rounded.obj", scale=(0.160, 0.125, 0.078))
    attrs(b, "geom", "head_shell", pos=(0.030, 0, 0.044))
    attrs(b, "geom", "head_collision", type="box", size=(0.140, 0.105, 0.065), pos=(0.030, 0, 0.044))
    component(b, "head_upper_shell", pos_m=[0.030, 0, 0.044], inertia_box_halfsize_m=[0.160, 0.125, 0.078])
    component(b, "head_structure", pos_m=[0, 0, 0.010], inertia_box_halfsize_m=[0.050, 0.035, 0.030])
    # The jaw pivots near the rear of the skull rather than under the snout.
    attrs(b, "body", "jaw", pos=(-0.050, 0, -0.045))
    # A U-shaped jaw leaves a real opening around the neck. Side rails and
    # the front chin are represented in both visual and collision geometry.
    attrs(b, "mesh", "jaw_shell_mesh", scale=(0.080, 0.114, 0.034))
    attrs(b, "geom", "jaw_shell", pos=(0.160, 0, -0.018))
    attrs(b, "geom", "jaw_collision", type="box", size=(0.070, 0.100, 0.030), pos=(0.160, 0, -0.018))
    component(b, "jaw_shell_mass", pos_m=[0.105, 0, -0.018], inertia_box_halfsize_m=[0.150, 0.114, 0.034])
    component(b, "jaw_motor", pos_m=[-0.050, 0, -0.045])
    for side, sign in (("left", 1), ("right", -1)):
        b.mesh(
            b.bodies["jaw"],
            side + "_jaw_rail",
            (0.071, 0.016, 0.034),
            (0.035, sign * 0.099, -0.018),
            "cream",
            source="rounded",
        )
        b.collision(b.bodies["jaw"], side + "_jaw_rail", "box", (0.065, 0.012, 0.030), (0.035, sign * 0.103, -0.018))
        rotation = (0, 0, -sign * 0.25)
        for suffix, size, pos in (
            ("eye_socket", (0.047, 0.008, 0.048), (-0.035, sign * 0.122, 0.060)),
            ("eye_white", (0.041, 0.006, 0.042), (-0.033, sign * 0.128, 0.060)),
            ("iris", (0.035, 0.005, 0.037), (-0.030, sign * 0.133, 0.060)),
            ("pupil", (0.026, 0.004, 0.030), (-0.027, sign * 0.137, 0.060)),
            ("eye_glint", (0.007, 0.002, 0.008), (-0.019, sign * 0.143, 0.077)),
        ):
            attrs(b, "geom", side + "_" + suffix, size=size, pos=pos, euler=rotation)
        attrs(b, "geom", side + "_side_tooth", size=(0.006, 0.004, 0.010), pos=(0.065, sign * 0.106, -0.035))
        attrs(b, "geom", side + "_nostril", size=(0.010, 0.003, 0.006), pos=(0.135, sign * 0.119, 0.035))
    for i, sign in enumerate((-1, 1)):
        attrs(b, "geom", f"tooth_{i}", size=(0.006, 0.005, 0.011), pos=(0.152, sign * 0.086, -0.035))
    # Move the whole optical package together; retain all aperture dimensions.
    camera_delta = np.array([0.009, 0, 0.0067])
    for item in head:
        name = item.get("name", "")
        if name.startswith("snout_") or name == "camera_package_envelope":
            if item.get("pos"):
                item.set("pos", base.values(np.fromstring(item.get("pos", ""), sep=" ") + camera_delta))
    camera = next(c for c in b.components if c["name"] == "snout_camera")
    camera["pos_m"] = (np.array(camera["pos_m"]) + camera_delta).tolist()
    if shape == "long_oval":
        attrs(b, "mesh", "head_shell_mesh", file="soft.obj", scale=(0.170, 0.125, 0.078))
        attrs(b, "geom", "head_collision", type="ellipsoid", size=(0.160, 0.115, 0.072))
        component(b, "head_upper_shell", inertia_box_halfsize_m=[0.170, 0.125, 0.078])
        for side, sign in (("left", 1), ("right", -1)):
            for suffix in ("eye_socket", "eye_white", "iris", "pupil", "eye_glint"):
                eye = find(b, "geom", side + "_" + suffix)
                eye_pos = np.fromstring(eye.get("pos", ""), sep=" ")
                eye_pos[1] -= sign * 0.008
                eye.set("pos", base.values(eye_pos))
            attrs(b, "geom", side + "_nostril", pos=(0.115, sign * 0.105, 0.035))
    elif shape == "soft_wedge":
        # Embed the tapered visual mesh; no shared asset is overwritten.
        mesh = find(b, "mesh", "head_shell_mesh")
        vertices: list[float] = []
        faces: list[int] = []
        for line in (base.HERE / "assets/rounded.obj").read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "v":
                x, y, z = map(float, parts[1:])
                vertices.extend((x, y * (0.96 - 0.10 * x), z - 0.12 * (x + 1)))
            elif parts and parts[0] == "f":
                faces.extend(int(i) - 1 for i in parts[1:])
        mesh.attrib.pop("file")
        mesh.set("vertex", base.values(vertices))
        mesh.set("face", " ".join(map(str, faces)))
        attrs(b, "geom", "head_collision", size=(0.140, 0.100, 0.060), pos=(0.030, 0, 0.035))
        component(b, "head_upper_shell", pos_m=[0.025, 0, 0.035], inertia_box_halfsize_m=[0.160, 0.1325, 0.085])


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
    if name in ("rounded_box", "long_oval", "soft_wedge"):
        unified_head(b, name)
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


def render_box_comparison(directory: Path) -> None:
    from PIL import Image, ImageDraw

    from examples.mini_tyrannosaurus_rex.render_design import _camera_basis, _font, geometry

    from .render_design import render

    models = [demo.load_model(path=directory / name / "model.xml") for name in ("large_head", "rounded_box")]
    sheet = Image.new("RGB", (1600, 1460), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((40, 28), "T-REX / A ROUNDED-BOX SKULL", font=_font(34), fill=(38, 55, 43))
    draw.text(
        (40, 83), "Same body, legs and head-weight target / actual MuJoCo geometry", font=_font(22), fill=(83, 97, 88)
    )
    for row, view in enumerate(("isometric", "side")):
        points = [geometry(m, d)[0] @ _camera_basis(view).T for m, d in models]
        bounds = (
            np.min([p[:, :, :2].min(axis=(0, 1)) for p in points], axis=0),
            np.max([p[:, :, :2].max(axis=(0, 1)) for p in points], axis=0),
        )
        for i, (model, data) in enumerate(models):
            x, y = i * 800, 150 + row * 610
            sheet.paste(render(model, data, view=view, width=800, height=555, projection_bounds=bounds), (x, y + 35))
            draw.text(
                (x + 40, y),
                ("Previous / round cranium + muzzle", "Revised / one rounded rectangular skull")[i],
                font=_font(22),
                fill=(38, 55, 43),
            )
    draw.text(
        (40, 1395),
        "Shape study. Hollow-shell mass targets and simplified simulation do not establish hardware viability.",
        font=_font(19),
        fill=(83, 97, 88),
    )
    sheet.save(directory / "rounded_box_comparison.png")


def render_shape_options(directory: Path) -> None:
    from PIL import Image, ImageDraw

    from examples.mini_tyrannosaurus_rex.render_design import _camera_basis, _font, geometry

    from .render_design import render

    names = ("rounded_box", "long_oval", "soft_wedge")
    labels = ("A / ROUNDED BOX", "B / LONG OVAL", "C / SOFT WEDGE")
    models = [demo.load_model(path=directory / name / "model.xml") for name in names]
    sheet = Image.new("RGB", (1800, 1410), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((40, 28), "T-REX / FINDING THE FACE", font=_font(34), fill=(38, 55, 43))
    draw.text(
        (40, 83), "Three skull shapes / same body and 530 g moving-head target", font=_font(22), fill=(83, 97, 88)
    )
    for row, view in enumerate(("isometric", "side", "front")):
        for model, _ in models:
            if row == 1:
                model.geom_group[~np.isin(model.geom_bodyid, (model.body("head").id, model.body("jaw").id))] = 5
        points = [geometry(m, d)[0] @ _camera_basis(view).T for m, d in models]
        bounds = (
            np.min([p[:, :, :2].min(axis=(0, 1)) for p in points], axis=0),
            np.max([p[:, :, :2].max(axis=(0, 1)) for p in points], axis=0),
        )
        for i, (model, data) in enumerate(models):
            x = 600 * i
            y, height = ((180, 560), (790, 260), (1080, 260))[row]
            sheet.paste(render(model, data, view=view, width=600, height=height, projection_bounds=bounds), (x, y))
            if row == 0:
                draw.text((x + 30, 145), labels[i], font=_font(24), fill=(38, 55, 43))
    draw.text(
        (40, 1360),
        "Actual MuJoCo geometry. These are proportion studies, not fabrication-ready designs.",
        font=_font(20),
        fill=(83, 97, 88),
    )
    sheet.save(directory / "shape_options.png")


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
        render_box_comparison(args.output)
        render_shape_options(args.output)


if __name__ == "__main__":
    main()
