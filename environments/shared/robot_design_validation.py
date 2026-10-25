"""Sampled mechanical screens for uncoupled robot concept models.

Direct distances include pairs suppressed by MuJoCo's contact filters. Gravity
moments use each named hinge's complete unsupported subtree. These are static
kinematic samples, not equilibrium, trajectory, fatigue or hardware validation.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import platform
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .robot_design_spec import RobotDesignSpec, pose_count, validate_spec


def _poses(spec: RobotDesignSpec) -> Iterator[tuple[str, dict[str, float]]]:
    yield "home", {}
    for sweep in spec.sweeps:
        for values in itertools.product(*sweep.samples.values()):
            yield sweep.name, dict(zip(sweep.samples, values, strict=True))


def screen_design(model: mujoco.MjModel, spec: RobotDesignSpec) -> dict[str, Any]:
    """Return deterministic evidence without altering the input model.

    ``passed`` applies only to the declared samples and limits. Missing bearing
    moment capacities are incomplete; any known failure takes precedence. Even
    a scoped pass always leaves the wider hardware assessment incomplete.
    """
    validate_spec(model, spec)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(spec.home_keyframe).id)
    home = data.qpos.copy()
    joint_names = sorted({load.joint for load in spec.loads} | {j for s in spec.sweeps for j in s.samples})
    addresses = {name: int(model.joint(name).qposadr[0]) for name in joint_names}
    pairs = [(model.geom(p.first).id, model.geom(p.second).id) for p in spec.clearances]
    axes = [(model.joint(load.joint).id, model.actuator(load.actuator).id) for load in spec.loads]
    search_distance = max(1.0, *(2 * p.min_distance_m for p in spec.clearances)) if pairs else 1.0
    clearances: list[dict[str, Any]] = [
        {
            "first": pair.first,
            "second": pair.second,
            "required_clearance_m": pair.min_distance_m,
            "min_distance_m": math.inf,
        }
        for pair in spec.clearances
    ]
    loads: list[dict[str, Any]] = [
        {
            "joint": load.joint,
            "actuator": load.actuator,
            "torque_capacity_nm": model.actuator_forcerange[actuator].tolist(),
            "allowed_torque_fraction": load.max_torque_fraction,
            "torque_evidence": load.torque_evidence,
            "max_gravity_torque_nm": -1.0,
            "maximum_torque_fraction": -1.0,
            "max_bearing_moment_nm": -1.0,
            "bearing_limit_nm": load.bearing_limit_nm,
            "bearing_evidence": load.bearing_evidence,
        }
        for load, (_, actuator) in zip(spec.loads, axes, strict=True)
    ]
    for sweep, samples in _poses(spec):
        data.qpos[:] = home
        for name, angle in samples.items():
            data.qpos[addresses[name]] = angle
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        witness = {name: float(data.qpos[address]) for name, address in addresses.items()}
        for row, (first, second) in zip(clearances, pairs, strict=True):
            distance = float(mujoco.mj_geomDistance(model, data, first, second, search_distance, None))
            if not math.isfinite(distance):
                raise ValueError(f"Nonfinite geometry distance for {row['first']} / {row['second']}.")
            if distance < row["min_distance_m"]:
                row.update(min_distance_m=distance, worst_pose=witness, worst_sweep=sweep)
        for row, (joint, actuator) in zip(loads, axes, strict=True):
            body = int(model.jnt_bodyid[joint])
            arm = data.subtree_com[body] - data.xanchor[joint]
            force = model.body_subtreemass[body] * model.opt.gravity
            moment = np.cross(arm, force)
            axis_moment = float(np.dot(moment, data.xaxis[joint]))
            holding_torque = -axis_moment
            capacity = float(model.actuator_forcerange[actuator, 1 if holding_torque >= 0 else 0])
            fraction = abs(holding_torque / capacity)
            bearing_moment = float(np.linalg.norm(moment - axis_moment * data.xaxis[joint]))
            if not all(math.isfinite(value) for value in (holding_torque, fraction, bearing_moment)):
                raise ValueError(f"Nonfinite gravity moment for {row['joint']}.")
            if abs(holding_torque) > row["max_gravity_torque_nm"]:
                row.update(max_gravity_torque_nm=abs(holding_torque), gravity_worst_pose=witness)
            if fraction > row["maximum_torque_fraction"]:
                row.update(
                    maximum_torque_fraction=fraction,
                    holding_torque_at_worst_fraction_nm=holding_torque,
                    torque_worst_pose=witness,
                    torque_worst_sweep=sweep,
                )
            if bearing_moment > row["max_bearing_moment_nm"]:
                row.update(
                    max_bearing_moment_nm=bearing_moment,
                    bearing_worst_pose=witness,
                    bearing_worst_sweep=sweep,
                )
    for row in clearances:
        row["distance_is_lower_bound"] = row["min_distance_m"] >= search_distance
        row["status"] = "passed" if row["min_distance_m"] >= row["required_clearance_m"] else "failed"
    for row in loads:
        row["torque_status"] = (
            "passed" if row["maximum_torque_fraction"] <= row["allowed_torque_fraction"] else "failed"
        )
        limit = row["bearing_limit_nm"]
        row["bearing_status"] = (
            "incomplete" if limit is None else "passed" if row["max_bearing_moment_nm"] <= limit else "failed"
        )
    statuses = [row["status"] for row in clearances] + [
        row[key] for row in loads for key in ("torque_status", "bearing_status")
    ]
    status = "failed" if "failed" in statuses else "incomplete" if "incomplete" in statuses else "passed"
    return {
        "schema_version": 1,
        "robot_id": spec.robot_id,
        "scope": "Sampled collision-envelope clearances and unsupported-subtree static gravity moments.",
        "status": status,
        "known_limits_passed": "failed" not in statuses,
        "hardware_assessment": {
            "status": "incomplete",
            "unverified": [
                "Continuous paths and poses outside the declared Cartesian samples and selected geometry pairs.",
                "Physical geometry and tolerances; mesh distances use convex hulls, not concave CAD surfaces.",
                "Ground reactions, balance, leg loads and locomotion; samples need not be stable or reachable.",
                "Acceleration, impact, friction, springs, actuator thermal limits and fatigue.",
                "Controller reachability: configured actuator force caps do not prove available drive torque.",
                "Bearing radial/axial force ratings, load paths, mounting stiffness and material stresses.",
                "Hardware evidence for all estimated masses, capacities and geometry.",
            ],
        },
        "coverage": {
            "sampled_poses": pose_count(spec),
            "sweeps": [{"name": "home", "poses": 1}]
            + [
                {"name": sweep.name, "poses": math.prod(len(v) for v in sweep.samples.values())}
                for sweep in spec.sweeps
            ],
            "clearance_pairs": len(pairs),
            "load_joints": len(axes),
            "continuous_path_checked": False,
            "ground_support_solved": False,
            "distance_search_limit_m": search_distance,
            "sampling": "Deterministic Cartesian groups; unsampled joints reset to home for each group.",
        },
        "specification": spec.as_dict(),
        "clearance_checks": clearances,
        "load_checks": loads,
    }


def write_report(
    report: dict[str, Any],
    *,
    output: Path,
    source_paths: Sequence[Path],
    source_root: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    """Write evidence bound to repository-relative inputs and the exact command."""
    if not source_paths or not command:
        raise ValueError("Reports require explicit input sources and an invocation command.")
    root = source_root.resolve()
    sources = {*source_paths, Path(__file__), Path(__file__).with_name("robot_design_spec.py")}
    hashes = {
        path.resolve().relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    payload = copy.deepcopy(report)
    payload.update(
        source_sha256=dict(sorted(hashes.items())),
        provenance={
            "command": list(command),
            "source_paths_relative_to": "repository root",
            "python_version": platform.python_version(),
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
        },
    )
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    return payload


def render_witnesses(
    model: mujoco.MjModel,
    spec: RobotDesignSpec,
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Render up to three worst-pose diagnostics using the shared CPU renderer.

    Highlight declared collision envelopes on a model copy. The caller's alpha,
    materials, geometry and pose stay untouched. No OpenGL context is required.
    """
    from PIL import Image, ImageDraw

    from .robot_rendering import font, render

    validate_spec(model, spec)
    targets: list[tuple[str, str, dict[str, float], set[int]]] = []
    if report["clearance_checks"]:
        row = min(report["clearance_checks"], key=lambda r: r["min_distance_m"] - r["required_clearance_m"])
        targets.append(
            (
                "clearance",
                f"{row['first']} / {row['second']}: {row['min_distance_m'] * 1000:.2f} mm "
                f"(required {row['required_clearance_m'] * 1000:.2f} mm)",
                row["worst_pose"],
                {model.geom(row[key]).id for key in ("first", "second")},
            )
        )
    for kind in ("torque", "bearing"):
        if not report["load_checks"]:
            continue
        key = "maximum_torque_fraction" if kind == "torque" else "max_bearing_moment_nm"
        row = max(
            report["load_checks"],
            key=lambda r: (
                (r["maximum_torque_fraction"] / r["allowed_torque_fraction"])
                if kind == "torque"
                else (
                    r["bearing_status"] == "failed",
                    r["bearing_limit_nm"] is None,
                    r["max_bearing_moment_nm"] / r["bearing_limit_nm"]
                    if r["bearing_limit_nm"] is not None
                    else r["max_bearing_moment_nm"],
                )
            ),
        )
        label = (
            f"{row['joint']}: {row[key]:.1%} of signed torque cap (budget {row['allowed_torque_fraction']:.0%})"
            if kind == "torque"
            else f"{row['joint']}: {row[key]:.3f} Nm transverse gravity moment; "
            f"bearing limit {row['bearing_limit_nm'] if row['bearing_limit_nm'] is not None else 'unknown'}"
        )
        body = int(model.jnt_bodyid[model.joint(row["joint"]).id])
        descendants = {body}
        for child in range(body + 1, model.nbody):
            if int(model.body_parentid[child]) in descendants:
                descendants.add(child)
        targets.append(
            (
                kind,
                label,
                row[f"{kind}_worst_pose"],
                {geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) in descendants},
            )
        )
    displayed = {
        model.geom(name).id
        for name in {g for pair in spec.clearances for g in (pair.first, pair.second)}
        | {g for names in spec.feet.values() for g in names}
    }
    # Load-only specifications can still display the joint's geometry. Restrict
    # all rendering to the robot subtree so floor/environment never hides it.
    root_body = model.jnt_bodyid[model.joint(spec.root_joint).id]
    if not spec.clearances:
        displayed.update(g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root_body)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for kind, label, pose, highlighted in targets:
        diagnostic = copy.copy(model)
        diagnostic.geom_group[:] = 5
        loaded_proxies = {geom for geom in highlighted if model.geom_group[geom] == 3}
        visible = displayed | (loaded_proxies or highlighted) if kind != "clearance" else displayed
        for geom in visible:
            diagnostic.geom_group[geom] = 3
            diagnostic.geom_matid[geom] = -1
            diagnostic.geom_rgba[geom] = (0.85, 0.12, 0.07, 1) if geom in highlighted else (0.40, 0.48, 0.51, 1)
        data = mujoco.MjData(diagnostic)
        mujoco.mj_resetDataKeyframe(diagnostic, data, diagnostic.key(spec.home_keyframe).id)
        for name, angle in pose.items():
            data.qpos[diagnostic.joint(name).qposadr[0]] = angle
        data.qvel[:] = 0
        mujoco.mj_forward(diagnostic, data)
        panel = Image.new("RGB", (1200, 640), (245, 247, 244))
        for column, direction in enumerate(((1.0, -1.35, 0.65), (0.0, 1.0, 0.15))):
            panel.paste(
                render(diagnostic, data, direction=direction, groups=(3,), width=600, height=500), (600 * column, 130)
            )
        draw = ImageDraw.Draw(panel)
        draw.text((20, 12), f"{spec.robot_id} | sampled {kind} witness", font=font(21), fill="#20282a")
        draw.text((20, 45), "SAMPLED KINEMATIC POSE — NOT A STANCE OR WALKING TRIAL", font=font(16), fill="#744230")
        draw.multiline_text((20, 73), textwrap.fill(label, 115), font=font(17), fill="#20282a", spacing=5)
        path = output_dir / f"{kind}_witness.png"
        panel.save(path)
        result[kind] = str(path)
    return result
