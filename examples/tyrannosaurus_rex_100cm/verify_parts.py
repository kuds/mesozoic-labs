"""Basic packaging, expression-load, optical and budget checks for this concept.

These are finite, explicitly named checks of ideal surrogates. They do not
qualify supplier CAD, continuous duty, structural strength or a walking gait.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared import robot_building

from .demo import MODEL_PATH, load_model

EXPRESSION_JOINTS = ("head_yaw", "head_pitch", "jaw_pitch", "left_shoulder_pitch", "right_shoulder_pitch", "tail_yaw")
BOM_FIELDS = ("category", "part", "motor_model", "quantity", "unit_usd", "total_usd", "price_basis", "source", "notes")


def budget_summary(ledger: dict[str, Any], budget: dict[str, Any], bom_path: Path) -> dict[str, Any]:
    """Reconcile purchased motor quantities, line arithmetic and the human CSV."""
    counts = Counter(c["part"] for c in ledger["motor_assignments"])
    quoted_counts = {row["motor_model"]: row["quantity"] for row in budget["rows"] if row.get("motor_model")}
    with bom_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        # Require every field exactly once, independently of column order.
        # DictReader otherwise silently collapses duplicate header names.
        header_ok = Counter(reader.fieldnames or ()) == Counter(BOM_FIELDS)
        csv_rows = list(reader)
    csv_ok = (
        header_ok
        and len(csv_rows) == len(budget["rows"])
        and all(
            row == {key: str(source.get(key, "")) for key in BOM_FIELDS}
            for row, source in zip(csv_rows, budget["rows"], strict=False)
        )
    )
    row_totals_ok = all(abs(row["quantity"] * row["unit_usd"] - row["total_usd"]) < 0.005 for row in budget["rows"])
    hardware = round(sum(row["total_usd"] for row in budget["rows"]), 2)
    extras = sum(row["usd"] for row in budget["development_extras"])
    total = round(hardware + extras + budget["contingency_usd"], 2)
    passed = bool(
        csv_ok
        and row_totals_ok
        and counts == quoted_counts
        and abs(hardware - budget["robot_hardware_subtotal_usd"]) < 0.005
        and extras == budget["development_extras_usd"]
        and abs(total - budget["total_before_tax_shipping_labor_usd"]) < 0.005
    )
    return dict(
        hardware_usd=hardware,
        development_extras_usd=extras,
        contingency_usd=budget["contingency_usd"],
        total_usd=total,
        csv_matches=csv_ok,
        motor_counts=dict(counts),
        budget_motor_counts=quoted_counts,
        passed=passed,
    )


def electronics_clearance(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    boxes = [g for g in range(model.ngeom) if model.geom(g).name.endswith("_envelope")]
    motors = [g for g in range(model.ngeom) if model.geom(g).name.endswith("_case_collision")]
    closest: dict[str, Any] = dict(distance_m=1.0, pair=[], envelopes=len(boxes))
    line = np.zeros(6)
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :] + motors:
            distance = float(mujoco.mj_geomDistance(model, data, first, second, 1.0, line))
            if distance < closest["distance_m"]:
                closest.update(distance_m=distance, pair=[model.geom(first).name, model.geom(second).name])
    return closest


def camera_visibility(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Probe a 7x5 cone per camera against visible skins and mechanisms.

    The 60-degree horizontal sample is an inspection cone, not an OAK calibration.
    The floor and provisional packaging envelopes are excluded; robot surfaces are
    included even if they do not participate in contact dynamics.
    """
    hits = []
    groups = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
    count = 0
    for name in ("snout_left", "snout_right"):
        cam = model.camera(name).id
        rotation = data.cam_xmat[cam].reshape(3, 3)
        for azimuth, elevation in itertools.product(np.linspace(-30, 30, 7), np.linspace(-25, 25, 5)):
            local = np.array([np.tan(np.deg2rad(azimuth)), np.tan(np.deg2rad(elevation)), -1.0])
            direction = rotation @ (local / np.linalg.norm(local))
            geom = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, data.cam_xpos[cam], direction, groups, 1, -1, geom)
            count += 1
            if distance >= 0:
                hits.append(
                    dict(
                        camera=name,
                        azimuth_deg=float(azimuth),
                        elevation_deg=float(elevation),
                        geom=model.geom(geom[0]).name,
                        distance_m=float(distance),
                    )
                )
    return dict(rays=count, self_occlusions=hits, passed=not hits)


def expression_gravity_sweep(model: mujoco.MjModel) -> dict[str, Any]:
    """Gravity moments about the six expression joints at 3^6 poses.

    Legs and torso remain at home. These moments are valid for each expression
    subtree; they are not the leg torques required to support ground reactions.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joints = [model.joint(name).id for name in EXPRESSION_JOINTS]
    poses = [np.linspace(float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1]), 3) for j in joints]
    torque, bearing = np.zeros(6), np.zeros(6)
    count = 0
    for angles in itertools.product(*poses):
        for j, angle in zip(joints, angles, strict=True):
            data.qpos[model.jnt_qposadr[j]] = angle
        mujoco.mj_forward(model, data)
        for i, j in enumerate(joints):
            body = model.jnt_bodyid[j]
            arm = data.subtree_com[body] - data.xanchor[j]
            moment = np.cross(arm, model.body_subtreemass[body] * model.opt.gravity)
            along_axis = float(moment @ data.xaxis[j])
            torque[i] = max(torque[i], abs(along_axis))
            bearing[i] = max(bearing[i], float(np.linalg.norm(moment - along_axis * data.xaxis[j])))
        count += 1
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(EXPRESSION_JOINTS):
        cap = float(model.actuator_forcerange[model.actuator(name + "_servo").id, 1])
        rows.append(
            dict(
                joint=name,
                max_static_gravity_torque_nm=float(torque[i]),
                simulation_cap_nm=cap,
                fraction_of_cap=float(torque[i] / cap),
                max_transverse_gravity_moment_nm=float(bearing[i]),
            )
        )
    return dict(poses=count, rows=rows, passed=all(r["fraction_of_cap"] < 0.8 for r in rows))


def check_parts() -> dict[str, Any]:
    model, data = load_model()
    ledger = json.loads(MODEL_PATH.with_name("components.json").read_text())
    budget = json.loads(MODEL_PATH.with_name("budget.json").read_text())
    pricing = budget_summary(ledger, budget, MODEL_PATH.with_name("bom.csv"))
    clearance = electronics_clearance(model, data)
    optics = camera_visibility(model, data)
    gravity = expression_gravity_sweep(model)
    # At home the two sole rectangles share orientation and fore/aft extents.
    # Their convex hull is precisely this rectangle; do not reuse for a gait.
    points = []
    for side in ("left", "right"):
        g = model.geom(side + "_sole_collision").id
        rotation = data.geom_xmat[g].reshape(3, 3)
        for x, y in itertools.product(
            (-model.geom_size[g, 0], model.geom_size[g, 0]), (-model.geom_size[g, 1], model.geom_size[g, 1])
        ):
            points.append((rotation @ [x, y, -model.geom_size[g, 2]] + data.geom_xpos[g])[:2])
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    com = data.subtree_com[model.body("torso").id, :2]
    margin = float(min(*(com - low), *(high - com)))
    battery = next(c for c in ledger["components"] if c["name"] == "battery")
    pack_dimensions = 2 * np.asarray(battery["inertia_box_halfsize_m"])
    cell_dimensions = np.array([6 * 0.02155, 4 * 0.02155, 0.07015])
    cell_fit = bool(np.all(cell_dimensions < pack_dimensions))
    sources = [
        Path(__file__),
        Path(robot_building.__file__),
        MODEL_PATH.with_name("demo.py"),
        MODEL_PATH,
        MODEL_PATH.with_name("build_model.py"),
        MODEL_PATH.with_name("components.json"),
        MODEL_PATH.with_name("budget.json"),
        MODEL_PATH.with_name("bom.csv"),
    ]
    sources += sorted(MODEL_PATH.parent.glob("assets/*.obj"))
    report = {
        "status": "basic_concept_parts_screen_not_fabrication_or_walking_qualification",
        "electronics_to_cases_and_each_other": clearance,
        "home_double_support_margin_m": margin,
        "camera_visibility_home": optics,
        "expression_gravity_sweep": gravity,
        "battery": dict(
            chemistry="12S2P P45B",
            nominal_voltage_v=12 * 3.6,
            full_voltage_v=12 * 4.2,
            nominal_energy_wh=12 * 3.6 * 2 * 4.5,
            bare_cell_array_m=cell_dimensions.tolist(),
            allocated_pack_m=pack_dimensions.tolist(),
            cell_geometry_fits=cell_fit,
            complete_pack_fit_verified=False,
        ),
        "budget": pricing,
        "source_sha256": {
            os.path.relpath(p, MODEL_PATH.parent): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "limitations": [
            "Static gravity is not acceleration or impact load; transverse moments require separately designed output bearings.",
            "The optical sample does not qualify the full calibrated field of view or every head pose.",
            "Packaging boxes omit mounting hardware, connectors, cooling paths and removal clearances.",
            "No test establishes continuous thermal torque, structural factors of safety, gait feasibility or measured runtime.",
        ],
    }
    report["passed"] = bool(
        clearance["envelopes"] == 4
        and clearance["distance_m"] > 0.002
        and margin > 0.030
        and optics["passed"]
        and gravity["passed"]
        and pricing["passed"]
        and cell_fit
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_PATH.with_name("parts_verification.json"))
    args = parser.parse_args()
    report = check_parts()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
