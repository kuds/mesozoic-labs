"""Explore body proportions with fixed legs; no walking or fabrication claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from . import PARTS_BUDGET_PATH, demo, validate
from . import build_model as base

OUTPUT = base.HERE / "proportion_study"
SPECS: dict[str, dict[str, Any]] = {
    "baseline": dict(label="Current 51.5 cm", half=(0.089, 0.084, 0.085), z=0.050, neck=0.150),
    "compact": dict(
        label="Fuller 51.5 cm",
        half=(0.120, 0.110, 0.110),
        z=0.030,
        neck=0.150,
        shell_g=180,
        structure_g=300,
        brace_g=60,
        wiring_g=300,
    ),
    "balanced": dict(
        label="Fuller 55 cm",
        half=(0.1325, 0.115, 0.130),
        z=0.055,
        neck=0.185,
        shell_g=230,
        structure_g=370,
        brace_g=100,
        wiring_g=350,
    ),
    "tall": dict(
        label="Fuller 60 cm",
        half=(0.145, 0.1225, 0.155),
        z=0.080,
        neck=0.235,
        shell_g=300,
        structure_g=450,
        brace_g=180,
        wiring_g=400,
    ),
}

# USD additions to the existing first-build plan, not supplier quotations.
# The mechanical allowance already includes basic reprints; these are further
# allowances for the larger shells/brackets and the revised body development.
COST_DELTAS = {
    "baseline": dict(structure=(0, 0), shell=(0, 0), harness=(0, 0), development=(0, 0), freight_tax=(0, 0)),
    "compact": dict(
        structure=(75, 175), shell=(50, 125), harness=(25, 50), development=(50, 150), freight_tax=(0, 100)
    ),
    "balanced": dict(
        structure=(175, 350), shell=(125, 250), harness=(50, 100), development=(100, 200), freight_tax=(100, 150)
    ),
    "tall": dict(
        structure=(350, 650), shell=(200, 350), harness=(100, 200), development=(200, 400), freight_tax=(200, 300)
    ),
}


def find(builder: base.Builder, tag: str, name: str) -> Any:
    item = builder.xml.find(f".//{tag}[@name='{name}']")
    if item is None:
        raise KeyError(name)
    return item


def attrs(builder: base.Builder, tag: str, name: str, **kw: Any) -> None:
    item = find(builder, tag, name)
    for key, value in kw.items():
        item.set(key, base.values(value))


def component(builder: base.Builder, name: str, **kw: Any) -> None:
    next(c for c in builder.components if c["name"] == name).update(kw)


def build_variant(name: str, directory: Path, *, extra_mass_body: str | None = None) -> Path:
    """Add specified mass and update its inertia; retain reserve and hardware."""
    spec = SPECS[name]
    b = base.Builder()
    b.leg("left", 1)
    b.leg("right", -1)
    b.upper_body()
    b.xml.set("model", "tyrannosaurus_rex_proportions_" + name)
    if name != "baseline":
        x, y, z = spec["half"]
        center_z = spec["z"]
        attrs(b, "mesh", "torso_shell_mesh", scale=(x, y, z))
        attrs(b, "geom", "torso_shell", pos=(-0.012, 0, center_z))
        attrs(b, "mesh", "belly_panel_mesh", scale=(0.026, y * 0.77, z * 0.88))
        attrs(b, "geom", "belly_panel", pos=(x - 0.036, 0, center_z - 0.006))
        attrs(b, "mesh", "back_access_panel_mesh", scale=(0.020, y * 0.71, z * 0.77))
        attrs(b, "geom", "back_access_panel", pos=(-x + 0.010, 0, center_z))
        for geom in b.torso.findall("geom"):
            if geom.get("name", "").startswith(("back_vent_", "back_screw_")):
                pos = np.fromstring(geom.get("pos", ""), sep=" ")
                pos[0] -= x - 0.089
                pos[2] += center_z - 0.050
                geom.set("pos", base.values(pos))
        # Keep the old compact core. Additional thin front/back and upper-side
        # proxies describe an open-bottom shell with hip access openings. These
        # are only approximate panels, not an interference-checked CAD shell.
        panel_top = spec["neck"] - 0.050
        panel_low = max(-0.055, center_z - z + 0.020)
        panel_z = (panel_top + panel_low) / 2
        panel_h = (panel_top - panel_low) / 2
        b.collision(b.torso, "belly_guard", "box", (0.004, y * 0.60, panel_h), (x - 0.026, 0, panel_z))
        b.collision(b.torso, "back_guard", "box", (0.004, y * 0.60, panel_h), (-x + 0.006, 0, panel_z))
        for side, sign in (("left", 1), ("right", -1)):
            b.collision(
                b.torso,
                side + "_upper_side_guard",
                "box",
                (x * 0.45, 0.004, (panel_top - 0.05) / 2),
                (-0.020, sign * (y - 0.010), (panel_top + 0.05) / 2),
            )
            shoulder = (0.035, sign * (y - 0.004), center_z + 0.060)
            attrs(b, "geom", side + "_shoulder_case", pos=shoulder)
            attrs(b, "body", side + "_arm", pos=(shoulder[0], shoulder[1] + sign * 0.017, shoulder[2]))
            component(b, side + "_shoulder_motor", pos_m=list(shoulder))
            component(b, side + "_shoulder_mount", pos_m=list(shoulder))
            # Wider lightweight thigh fairings, with motor faces still exposed.
            attrs(b, "mesh", side + "_thigh_fairing_mesh", scale=(0.042, 0.030, 0.051))
            component(b, side + "_leg_cover", mass_kg=0.045, inertia_box_halfsize_m=[0.042, 0.030, 0.051])
        neck_pos = [-0.015, 0, spec["neck"]]
        attrs(b, "body", "head_yaw_body", pos=neck_pos)
        attrs(b, "geom", "neck_yaw_case", pos=neck_pos)
        component(b, "head_yaw_motor", pos_m=neck_pos)
        tail_shift = -(x - 0.089)
        for tag, tag_name in (("body", "tail"), ("geom", "tail_bearing")):
            pos = np.fromstring(find(b, tag, tag_name).get("pos"), sep=" ")
            pos[0] += tail_shift
            attrs(b, tag, tag_name, pos=pos)
        component(b, "tail_motor", pos_m=[-0.086 + tail_shift, 0, 0.016])
        component(
            b,
            "torso_shell_mass",
            mass_kg=spec["shell_g"] / 1000,
            pos_m=[-0.012, 0, center_z],
            inertia_box_halfsize_m=list(spec["half"]),
        )
        component(b, "pelvis_structure", mass_kg=spec["structure_g"] / 1000)
        component(b, "power_board", mass_kg=spec["wiring_g"] / 1000)
        b.component(
            b.torso,
            "extended_body_brace",
            "load_bearing_structure",
            spec["brace_g"],
            (-0.020, 0, spec["neck"] / 2),
            (0.040, 0.045, spec["neck"] / 2),
        )
    if extra_mass_body is not None:
        if extra_mass_body not in ("torso", "head"):
            raise ValueError("Sensitivity payload must be attached to torso or head")
        payload_pos = (-0.015, 0, 0.020) if extra_mass_body == "torso" else (0.025, 0, 0.010)
        b.component(
            b.bodies[extra_mass_body],
            "sensitivity_payload",
            "additional_payload",
            500,
            payload_pos,
            (0.025, 0.025, 0.025),
        )
    directory.mkdir(parents=True, exist_ok=True)
    compiler = b.xml.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", os.path.relpath(base.HERE / "assets", directory))
    b.finish(directory)
    return directory / "model.xml"


def check_variant(name: str, path: Path) -> dict[str, Any]:
    model, data = demo.load_model(path=path)
    low, high = validate.visual_bounds(model, data)
    hip = data.xanchor[model.joint("left_hip_pitch").id]
    spec = SPECS[name]
    shell_bottom = base.ROOT_HEIGHT + spec["z"] - spec["half"][2]
    trials = {
        label: validate.run_trial(model, duration=10, gesture=gesture)
        for label, gesture in (("stance", False), ("gesture", True))
    }
    nominal_mass = float(model.body_mass.sum())
    com = data.subtree_com[model.body("torso").id]
    return dict(
        label=spec["label"],
        mass_kg=nominal_mass,
        dimensions_m=(high - low).tolist(),
        hip_height_m=float(hip[2]),
        hip_height_fraction=float(hip[2] / (high[2] - low[2])),
        torso_depth_width_height_m=(2 * np.array(spec["half"])).tolist(),
        shell_bottom_m=shell_bottom,
        com_m=com.tolist(),
        head_yaw_subtree_mass_kg=float(model.body_subtreemass[model.body("head_yaw_body").id]),
        home_internal_contacts=validate.internal_contacts(model, data),
        trials=trials,
    )


def budget_scenarios() -> dict[str, Any]:
    """Apply visible additive allowances, then the existing 25% contingency."""
    budget = json.loads(PARTS_BUDGET_PATH.read_text())
    scope = {
        s: sum(
            (Decimal(row["unit_price_usd"]) * row["quantity"] for row in budget["rows"] if row["scope"] == s),
            Decimal(0),
        )
        for s in ("installed", "development")
    }
    contingency = Decimal(budget["contingency_fraction"])
    freight = Decimal(budget["tax_freight_import_planning_reserve_usd"])
    result = {}
    for name, deltas in COST_DELTAS.items():
        installed, funding = [], []
        for i in (0, 1):
            parts = scope["installed"] + sum(Decimal(deltas[k][i]) for k in ("structure", "shell", "harness"))
            project = (parts + scope["development"] + deltas["development"][i]) * (1 + contingency)
            installed.append(str(parts.quantize(Decimal("0.01"))))
            funding.append(str((project + freight + deltas["freight_tax"][i]).quantize(Decimal("0.01"))))
        result[name] = dict(installed_usd=installed, first_build_funding_usd=funding, incremental_allowances_usd=deltas)
    return dict(
        currency="USD",
        checked_on="2026-10-24",
        scenarios=result,
        contingency_fraction=str(contingency),
        base_freight_tax_reserve_usd=str(freight),
        notes=[
            "Same compute, camera, battery and 18 actuators in every scenario.",
            "No motor-upgrade premium is included; incremental fabrication costs are estimates.",
            "Funding includes development equipment/spares, contingency and freight/tax reserve; excludes labor.",
        ],
    )


def comparison_sheet(directory: Path, reports: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    from examples.mini_tyrannosaurus_rex.render_design import _camera_basis, _font, geometry

    from .render_design import render

    models = [demo.load_model(path=directory / name / "model.xml") for name in SPECS]
    sheet = Image.new("RGB", (2000, 1580), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((38, 25), "T-REX / BODY PROPORTIONS", font=_font(34), fill=(38, 55, 43))
    draw.text(
        (38, 78),
        "Actual MuJoCo geometry / same leg lengths, feet, motors and camera hardware",
        font=_font(21),
        fill=(83, 97, 88),
    )
    for row, view in enumerate(("side", "isometric")):
        basis = _camera_basis(view)
        projections = [geometry(m, d)[0] @ basis.T for m, d in models]
        low = np.min([p[:, :, :2].min(axis=(0, 1)) for p in projections], axis=0)
        high = np.max([p[:, :, :2].max(axis=(0, 1)) for p in projections], axis=0)
        for col, ((name, spec), (model, data)) in enumerate(zip(SPECS.items(), models, strict=True)):
            x, y = col * 500, 165 + row * 610
            sheet.paste(
                render(model, data, view=view, width=500, height=550, projection_bounds=(low, high)), (x, y + 28)
            )
            draw.text((x + 30, y), spec["label"], font=_font(21), fill=(38, 55, 43))
            if row == 0:
                draw.text(
                    (x + 30, y + 585),
                    f"{reports[name]['mass_kg']:.2f} kg assigned mass",
                    font=_font(18),
                    fill=(83, 97, 88),
                )
    draw.text(
        (38, 1430),
        "Both rows use a shared scale within each view. All hips remain 25.3 cm above the floor.",
        font=_font(22),
        fill=(60, 76, 66),
    )
    draw.text(
        (38, 1470),
        "Proportion study: approximate shell proxies and unweighed masses; no walking or fabrication validation.",
        font=_font(21),
        fill=(83, 97, 88),
    )
    sheet.save(directory / "comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    reports = {name: check_variant(name, build_variant(name, args.output / name)) for name in SPECS}
    sensitivity = {}
    with tempfile.TemporaryDirectory() as tmp:
        for body in ("torso", "head"):
            path = build_variant("balanced", Path(tmp) / body, extra_mass_body=body)
            sensitivity[body + "_plus_500g"] = check_variant("balanced", path)
    sources = [
        Path(__file__),
        base.HERE / "build_model.py",
        base.HERE / "validate.py",
        Path(validate.robot_validation.__file__),
        base.HERE / "demo.py",
        PARTS_BUDGET_PATH,
        *sorted((base.HERE / "assets").glob("*.obj")),
    ]
    report = dict(
        status="proportions_and_stance_study_not_walking_or_cad_validation",
        variants=reports,
        payload_sensitivity=sensitivity,
        budget=budget_scenarios(),
        mujoco_version=mujoco.__version__,
        source_sha256={
            str(p.relative_to(base.HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "mass": r["mass_kg"],
                    "height": r["dimensions_m"][2],
                    "contacts": r["home_internal_contacts"],
                    "trials": {
                        k: {
                            "pass": v["passed"],
                            "rms": max(v["rms_torque_nm"][:12]),
                            "tilt": v["max_torso_tilt_deg"],
                            "penetration": v["maximum_internal_penetration_m"],
                        }
                        for k, v in r["trials"].items()
                    },
                }
                for name, r in reports.items()
            },
            indent=2,
        )
    )
    if args.render:
        comparison_sheet(args.output, reports)


if __name__ == "__main__":
    main()
