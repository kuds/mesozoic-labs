"""Reproducible packing, motion-clearance and unsupported-stance screens.

Direct geometry queries deliberately bypass contact masks and parent/child
filtering. A sampled pass is not continuous-motion or fabrication approval.
Only convex shell wall/section geoms may be listed in ``shell_geoms``: an
enclosing shell's solid convex hull cannot represent its empty interior.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import mujoco
import numpy as np

from environments.shared import robot_clearance, robot_geometry, robot_validation
from examples.mini_tyrannosaurus_rex import balance_checks, character_checks, demo, engineering

HERE = Path(__file__).parent
REPOSITORY = HERE.parents[2]
DISTANCE_COMPARISON_ALLOWANCE_M = 1e-9
HARDWARE_LIMITATIONS = (
    "Unweighed component and shell allocations; installed mass and inertia need measurement.",
    "Horn/idler geometry, bearings, fasteners, cable bend radii and structural deflection remain unqualified.",
    "Sampled geometry separation is not continuous or exhaustive motion-clearance certification.",
    "Ideal actuator simulation omits measured torque-speed, thermal, electrical, delay and backlash behavior.",
    "No walking, fall recovery or fabricated prototype is certified by these screens.",
)


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    index = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    if index < 0:
        raise ValueError(f"Unknown declared geometry: {name}")
    dimensions = {
        mujoco.mjtGeom.mjGEOM_SPHERE: 1,
        mujoco.mjtGeom.mjGEOM_CAPSULE: 2,
        mujoco.mjtGeom.mjGEOM_CYLINDER: 2,
        mujoco.mjtGeom.mjGEOM_BOX: 3,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID: 3,
    }.get(model.geom_type[index])
    if dimensions is not None:
        sizes = model.geom_size[index, :dimensions]
        if not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError(f"Declared geometry {name} must have finite, positive dimensions")
    return index


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return number


def _names(rows: list[Any]) -> list[str]:
    return [str(row if isinstance(row, str) else row["geom"]) for row in rows]


def build_pairs(model: mujoco.MjModel, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand automatic packing and shell pairs, requiring explained exclusions."""
    margin = _positive(manifest.get("clearance_m", 0.002), "clearance_m")
    excluded: dict[tuple[str, str], str] = {}
    for row in manifest.get("excluded_pairs", []):
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("Every excluded pair needs a nonempty physical-interface reason")
        for name in (row["first"], row["second"]):
            _geom_id(model, name)
        excluded[tuple(sorted((row["first"], row["second"])))] = row["reason"]
    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def add(first: str, second: str, category: str, required: float = margin) -> None:
        if first == second:
            return
        _geom_id(model, first)
        _geom_id(model, second)
        key = tuple(sorted((first, second)))
        if key in excluded:
            return
        if key not in pairs:
            pairs[key] = dict(first=key[0], second=key[1], clearance_m=required, categories=[category])
        else:
            pairs[key]["clearance_m"] = max(pairs[key]["clearance_m"], required)
            if category not in pairs[key]["categories"]:
                pairs[key]["categories"].append(category)

    hardware = manifest.get("hardware", [])
    envelopes = [str(row["envelope_geom"]) for row in hardware]
    envelopes += _names(manifest.get("hardware_envelopes", []))
    envelopes = sorted(set(envelopes))
    motors = [model.geom(i).name for i in engineering.motor_cases(model)]
    shells = _names(manifest.get("shell_geoms", []))
    mechanisms = _names(manifest.get("mechanism_geoms", []))
    for first, second in itertools.combinations(envelopes, 2):
        add(first, second, "installation_envelopes")
    for envelope, motor in itertools.product(envelopes, motors):
        add(envelope, motor, "installation_vs_motor")
    for envelope, mechanism in itertools.product(envelopes, mechanisms):
        add(envelope, mechanism, "installation_vs_mechanism")
    for first, second in itertools.combinations(motors, 2):
        add(first, second, "full_motor_cases")
    for shell, other in itertools.product(shells, motors + mechanisms):
        if model.geom_bodyid[_geom_id(model, shell)] != model.geom_bodyid[_geom_id(model, other)]:
            add(shell, other, "shell_vs_foreign_mechanism")
    for shell, envelope in itertools.product(shells, envelopes):
        add(shell, envelope, "shell_wall_vs_installation")
    for first, second in itertools.combinations(shells, 2):
        if model.geom_bodyid[_geom_id(model, first)] != model.geom_bodyid[_geom_id(model, second)]:
            add(first, second, "moving_shell_sections")
    for row in manifest.get("clearance_pairs", []):
        add(
            row["first"],
            row["second"],
            row.get("category", "declared_pair"),
            _positive(row.get("clearance_m", margin), "pair clearance_m"),
        )
    if not pairs:
        raise ValueError("Declare geometry to inspect; an empty clearance screen is not a pass")
    return [pairs[key] for key in sorted(pairs)]


def sampled_poses(
    model: mujoco.MjModel, *, samples: int = 512, seed: int = 635, joint_points: int = 5
) -> list[dict[str, Any]]:
    """Home, joint endpoints/interiors, coupled groups and reproducible full-body poses."""
    if samples < 0 or joint_points < 2:
        raise ValueError("Use samples >= 0 and joint_points >= 2")
    ranges = engineering.command_joint_ranges(model)
    poses: list[dict[str, Any]] = [dict(group="home", joints={})]
    for name, (low, high) in ranges.items():
        for angle in np.linspace(low, high, joint_points):
            poses.append(dict(group=f"individual:{name}", joints={name: float(angle)}))
    groups = {
        "expression_cartesian": ("head_yaw", "head_pitch", "jaw", "left_shoulder_pitch"),
        "left_leg_cartesian": ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"),
        "right_leg_cartesian": ("right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"),
    }
    for label, names in groups.items():
        names = tuple(name for name in names if name in ranges)
        if not names:
            continue
        axes = [np.linspace(*ranges[name], 3) for name in names]
        for angles in itertools.product(*axes):
            poses.append(dict(group=label, joints=dict(zip(names, map(float, angles), strict=True))))
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        poses.append(dict(group="seeded_full_body", joints={n: float(rng.uniform(*r)) for n, r in ranges.items()}))
    return poses


def _distance_status(result: dict[str, Any], required: float) -> str:
    distance = result["min_distance_m"]
    if distance > 0 and distance >= required - DISTANCE_COMPARISON_ALLOWANCE_M:
        return "pass"
    if result["distance_is_lower_bound"] or distance == 0:
        return "incomplete"
    return "fail"


def _enclosing_radius(model: mujoco.MjModel, geom: int) -> float:
    """Radius about geom_xpos enclosing the entire convex geometry, not a fit."""
    size, kind = model.geom_size[geom], model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        radius = np.linalg.norm(model.mesh_vert[start : start + count].astype(float), axis=1).max()
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = size[0] + size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = np.linalg.norm(size[:2])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        radius = size.max()
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        radius = np.linalg.norm(size)
    else:
        raise ValueError("Clearance bounding spheres require a convex primitive or mesh")
    return _positive(radius, f"{model.geom(geom).name} enclosing radius")


def clearance_screen(
    model: mujoco.MjModel, manifest: dict[str, Any], *, samples: int = 512, seed: int = 635, joint_points: int = 5
) -> dict[str, Any]:
    """Keep worst witnesses and independently preserve any known failing pose."""
    pairs = build_pairs(model, manifest)
    poses = sampled_poses(model, samples=samples, seed=seed, joint_points=joint_points)
    ids = [(_geom_id(model, row["first"]), _geom_id(model, row["second"])) for row in pairs]
    rows = [
        dict(
            **pair,
            status="pass",
            minimum_distance_m=None,
            witness=None,
            failure_witness=None,
            rigidly_co_mounted=bool(
                model.body_weldid[model.geom_bodyid[first]] == model.body_weldid[model.geom_bodyid[second]]
            ),
            evaluated_pose_count=0,
            native_query_count=0,
            broad_phase_certificate_count=0,
        )
        for pair, (first, second) in zip(pairs, ids, strict=True)
    ]
    search_distance = max(0.05, 2 * max(p["clearance_m"] for p in pairs))
    data = engineering.home_data(model)
    home = data.qpos.copy()
    radii = {geom: _enclosing_radius(model, geom) for geom in {geom for pair in ids for geom in pair}}
    indices = np.asarray(ids, dtype=int)
    radius_sums = np.asarray([radii[first] + radii[second] for first, second in ids])
    for pose_index, pose in enumerate(poses):
        data.qpos[:] = home
        engineering.set_pose(model, data, pose["joints"])
        centers = data.geom_xpos[indices]
        padding = 64 * np.finfo(float).eps * max(1.0, float(np.abs(centers).max()), float(radius_sums.max()))
        sphere_gaps = np.linalg.norm(centers[:, 1] - centers[:, 0], axis=1) - radius_sums - padding
        for row, (first, second), gap in zip(rows, ids, sphere_gaps, strict=True):
            if pose_index and row["rigidly_co_mounted"]:
                continue
            row["evaluated_pose_count"] += 1
            if gap >= row["clearance_m"]:
                row["broad_phase_certificate_count"] += 1
                if row["minimum_distance_m"] is not None and gap >= row["minimum_distance_m"]:
                    continue
                result = dict(
                    min_distance_m=float(gap),
                    native_distance_m=None,
                    distance_is_lower_bound=True,
                    distance_method="bounding_sphere_lower_bound",
                    bounding_spheres=dict(
                        first_center_m=data.geom_xpos[first].tolist(),
                        second_center_m=data.geom_xpos[second].tolist(),
                        first_radius_m=radii[first],
                        second_radius_m=radii[second],
                        roundoff_allowance_m=padding,
                    ),
                )
            else:
                row["native_query_count"] += 1
                result = robot_clearance.clearance_distance(
                    model,
                    data,
                    first,
                    second,
                    required_clearance_m=row["clearance_m"],
                    search_distance_m=search_distance,
                )
            status = _distance_status(result, row["clearance_m"])
            witness = dict(pose_index=pose_index, **pose, **result)
            if row["minimum_distance_m"] is None or result["min_distance_m"] < row["minimum_distance_m"]:
                row["minimum_distance_m"], row["witness"] = result["min_distance_m"], witness
            if status == "fail":
                row["status"] = "fail"
                previous = row["failure_witness"]
                if previous is None or result["min_distance_m"] < previous["min_distance_m"]:
                    row["failure_witness"] = witness
            elif status == "incomplete" and row["status"] == "pass":
                row["status"] = "incomplete"
    counts = Counter(row["status"] for row in rows)
    return {
        "scope": "Declared full cases, installation reservations and convex shell sections; filters bypassed.",
        "pose_count": len(poses),
        "pose_groups": dict(Counter(p["group"] for p in poses)),
        "seed": seed,
        "random_samples": samples,
        "joint_points": joint_points,
        "distance_comparison_allowance_m": DISTANCE_COMPARISON_ALLOWANCE_M,
        "command_ranges_rad": engineering.command_joint_ranges(model),
        "pair_count": len(rows),
        "queries": sum(row["evaluated_pose_count"] for row in rows),
        "native_queries": sum(row["native_query_count"] for row in rows),
        "broad_phase_certificates": sum(row["broad_phase_certificate_count"] for row in rows),
        "rigid_pair_optimization": "Same MuJoCo weld body means invariant relative transform; query once at home.",
        "counts": {status: counts[status] for status in ("pass", "fail", "incomplete")},
        "pairs": rows,
        "excluded_pairs": manifest.get("excluded_pairs", []),
        "passed": counts["fail"] == counts["incomplete"] == 0,
    }


def _support_in_frame(model: mujoco.MjModel, data: mujoco.MjData, geom: int, frame: int) -> np.ndarray:
    """Exact convex support intervals along a declared box's local axes."""
    axes = data.geom_xmat[frame].reshape(3, 3).T
    # The shared helper covers boxes, spheres, ellipsoids, capsules, cylinders,
    # and all mesh vertices; no sampled surface points can hide a protrusion.
    return robot_clearance._intervals(model, data, geom, axes) - (axes @ data.geom_xpos[frame])[:, None]


def containment_screen(model: mujoco.MjModel, manifest: dict[str, Any]) -> dict[str, Any]:
    data = engineering.home_data(model)
    specs = list(manifest.get("containment", []))
    for hardware in manifest.get("hardware", []):
        if hardware.get("body_geom"):
            specs.append(dict(inner_geom=hardware["body_geom"], cavity_geom=hardware["envelope_geom"], allowance_m=0.0))
    rows = []
    for spec in specs:
        inner, cavity = _geom_id(model, spec["inner_geom"]), _geom_id(model, spec["cavity_geom"])
        if model.geom_type[cavity] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Containment cavities must be explicit box volumes")
        if model.body_weldid[model.geom_bodyid[inner]] != model.body_weldid[model.geom_bodyid[cavity]]:
            raise ValueError("Containment expects rigidly co-mounted geoms; moving cavities need a pose sweep")
        allowance = _positive(spec.get("allowance_m", 0.002), "containment allowance", allow_zero=True)
        intervals = _support_in_frame(model, data, inner, cavity)
        half = model.geom_size[cavity]
        face_gaps = np.column_stack((intervals[:, 0] + half, half - intervals[:, 1]))
        gap = float(face_gaps.min())
        rows.append(
            {
                **spec,
                "allowance_m": allowance,
                "minimum_face_gap_m": gap,
                "six_face_gaps_m": face_gaps.tolist(),
                "passed": gap >= allowance - 1e-12,
            }
        )
    return {
        "scope": "Rigid body envelopes in declared empty box cavities at home",
        "checks": rows,
        "passed": bool(rows) and all(row["passed"] for row in rows),
    }


def integrity_screen(model: mujoco.MjModel, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check the manifest's dimensional and allocation contracts against compiled data."""
    checks: list[dict[str, Any]] = []
    for row in manifest.get("hardware", []):
        for geom_key, dimension_key in (
            ("body_geom", "raw_dimensions_xyz_mm"),
            ("envelope_geom", "installation_dimensions_xyz_mm"),
        ):
            if dimension_key not in row:
                continue
            dimensions = np.asarray(row[dimension_key], dtype=float) / 1000
            if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                raise ValueError(
                    f"{row.get('name', 'hardware')} {dimension_key} must contain three positive dimensions"
                )
            geom = _geom_id(model, row[geom_key])
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("Hardware dimensional contracts require box geoms")
            actual = 2 * model.geom_size[geom]
            checks.append(
                dict(
                    check="hardware_dimensions",
                    geom=row[geom_key],
                    expected_m=dimensions.tolist(),
                    actual_m=actual.tolist(),
                    passed=bool(np.allclose(actual, dimensions, atol=1e-9, rtol=0)),
                )
            )
    contract = manifest.get("mass_contract")
    if contract:
        tolerance = _positive(contract.get("tolerance_kg", 1e-6), "mass tolerance_kg", allow_zero=True)
        expected = _positive(contract["total_mass_kg"], "total_mass_kg")
        actual_mass = float(model.body_mass.sum())
        checks.append(
            dict(
                check="total_mass",
                expected_kg=expected,
                actual_kg=actual_mass,
                passed=abs(actual_mass - expected) <= tolerance,
            )
        )
        for body, mass in contract.get("body_masses_kg", {}).items():
            expected = _positive(mass, f"{body} mass")
            actual_mass = float(model.body(body).mass[0])
            checks.append(
                dict(
                    check="body_mass",
                    body=body,
                    expected_kg=expected,
                    actual_kg=actual_mass,
                    passed=abs(actual_mass - expected) <= tolerance,
                )
            )
        motors = engineering.motor_cases(model)
        count = int(contract.get("expected_motor_count", 14))
        checks.append(dict(check="motor_count", expected=count, actual=len(motors), passed=len(motors) == count))
        dimensions = np.asarray(contract.get("motor_dimensions_m", [0.020, 0.034, 0.026]), dtype=float)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("motor_dimensions_m must contain three positive dimensions")
        for geom in motors:
            actual = 2 * model.geom_size[geom]
            checks.append(
                dict(
                    check="full_motor_dimensions",
                    geom=model.geom(geom).name,
                    expected_m=dimensions.tolist(),
                    actual_m=actual.tolist(),
                    passed=bool(np.allclose(np.sort(actual), np.sort(dimensions), atol=1e-9, rtol=0)),
                )
            )
    return dict(checks=checks, passed=all(row["passed"] for row in checks))


def gesture_failures(report: dict[str, Any], duration: float) -> list[str]:
    """The existing mini character-test criteria, without relaxation."""
    failures = []
    for key in ("finite", "no_applied_external_forces"):
        if not report[key]:
            failures.append(key)
    if report["externally_supported"] or report["maximum_unexpected_support_N"] > 0:
        failures.append("external or non-foot support")
    if report["warnings"] or report["duration_s"] < duration - 0.005:
        failures.append("incomplete or warned simulation")
    if report["minimum_height_ratio"] <= 0.95:
        failures.append("height loss")
    if report["maximum_tilt_deg"] >= 5:
        failures.append("excessive tilt")
    if report["maximum_internal_penetration_m"] >= 0.0001:
        failures.append("internal contact penetration")
    for name, minimum in dict(
        head_yaw=0.4, head_pitch=0.1, jaw=0.15, left_shoulder_pitch=0.4, right_shoulder_pitch=0.4
    ).items():
        if report["joint_excursions_rad"].get(name, 0) <= minimum:
            failures.append(f"insufficient excursion: {name}")
    for name, effort in report["actuators"].items():
        if effort["peak_Nm"] > max(abs(x) for x in effort["cap_Nm"]) + 1e-10:
            failures.append(f"actuator cap: {name}")
    return failures


def _source_hashes(model_path: Path, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, str]:
    files = {model_path.resolve(), manifest_path.resolve(), Path(__file__).resolve(), (HERE / "cavity.py").resolve()}
    modules = (robot_clearance, robot_geometry, robot_validation, balance_checks, character_checks, demo, engineering)
    files.update(Path(module.__file__).resolve() for module in modules)
    pending = [model_path.resolve()]
    while pending:
        source = pending.pop()
        root = ET.parse(source).getroot()
        for element in root.iter("include"):
            child = (source.parent / element.attrib["file"]).resolve()
            if child not in files:
                files.add(child)
                pending.append(child)
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        for element in root.iter("mesh"):
            if "file" in element.attrib:
                files.add((source.parent / meshdir / element.attrib["file"]).resolve())
    for name in manifest.get("source_files", []):
        files.add((REPOSITORY / name).resolve())
    return {
        str(path.relative_to(REPOSITORY) if path.is_relative_to(REPOSITORY) else path): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(files)
    }


def validate(
    model_path: Path = HERE / "model.xml",
    manifest_path: Path = HERE / "manifest.json",
    *,
    samples: int = 512,
    seed: int = 635,
    joint_points: int = 5,
    run_dynamics: bool = True,
    duration: float = 30.0,
    gesture_duration: float = 10.0,
) -> dict[str, Any]:
    started = perf_counter()
    manifest = json.loads(manifest_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = engineering.home_data(model)
    lower, upper = robot_geometry.visual_bounds(model, data)
    clearances = clearance_screen(model, manifest, samples=samples, seed=seed, joint_points=joint_points)
    clearance_finished = perf_counter()
    containment = containment_screen(model, manifest)
    integrity = integrity_screen(model, manifest)
    from . import cavity

    exterior_containment = cavity.exterior_containment_screen(model, manifest)
    trials: dict[str, Any] = {}
    if run_dynamics:
        for name, kick in (
            ("hold", (0.0, 0.0, 0.0)),
            ("roll_kick", (0.15, 0.0, 0.0)),
            ("pitch_kick", (0.0, -0.15, 0.0)),
        ):
            trial = balance_checks.stance_trial(model, duration=duration, leg_cap=0.10, angular_kick=kick)
            trial["failures"] = balance_checks.stance_failures(trial)
            trials[name] = trial
        trial = character_checks.gesture_trial(model, duration=gesture_duration, supported=False)
        trial["failures"] = gesture_failures(trial, gesture_duration)
        trials["gesture"] = trial
    failed = clearances["counts"]["fail"] > 0 or any(not c["passed"] for c in containment["checks"])
    failed = failed or any(not c["passed"] for c in exterior_containment["checks"])
    failed = failed or not integrity["passed"] or any(trial["failures"] for trial in trials.values())
    scoped_pass = (
        clearances["passed"]
        and containment["passed"]
        and exterior_containment["passed"]
        and run_dynamics
        and not failed
    )
    omissions = list(HARDWARE_LIMITATIONS) + manifest.get("omissions", [])
    omissions += manifest.get("exterior_containment", {}).get("limitations", [])
    if not run_dynamics:
        omissions.append("Unsupported dynamic trials were skipped in this diagnostic invocation.")
    if not containment["checks"]:
        omissions.append("No cavity or part/envelope containment checks were declared.")
    if not exterior_containment["checks"]:
        omissions.append("No installation-envelope checks inside the actual exterior cavity were declared.")
    if exterior_containment.get("missing_envelopes"):
        omissions.append(
            "Exterior cavity containment is missing for: " + ", ".join(exterior_containment["missing_envelopes"])
        )
    return {
        "schema_version": 1,
        "design": manifest.get("design", "mini_juvenile"),
        "status": "fail" if failed else "incomplete",
        "scoped_simulation_pass": bool(scoped_pass),
        "fabrication_qualified": False,
        "runtime": dict(
            mujoco=mujoco.__version__,
            numpy=np.__version__,
            python=platform.python_version(),
            machine=platform.machine(),
            elapsed_s=perf_counter() - started,
            load_and_clearance_s=clearance_finished - started,
        ),
        "invocation": dict(
            model_path=str(
                model_path.resolve().relative_to(REPOSITORY)
                if model_path.resolve().is_relative_to(REPOSITORY)
                else model_path.resolve()
            ),
            manifest_path=str(
                manifest_path.resolve().relative_to(REPOSITORY)
                if manifest_path.resolve().is_relative_to(REPOSITORY)
                else manifest_path.resolve()
            ),
            samples=samples,
            seed=seed,
            joint_points=joint_points,
            run_dynamics=run_dynamics,
            duration_s=duration,
            gesture_duration_s=gesture_duration,
        ),
        "source_path_base": "repository_root",
        "source_sha256": _source_hashes(model_path, manifest_path, manifest),
        "model": {
            "mass_kg": float(model.body_mass.sum()),
            "actuators": model.nu,
            "home_com_m": data.subtree_com[model.body("torso").id].tolist(),
            "visual_bounds_m": [lower.tolist(), upper.tolist()],
            "visual_extent_xyz_m": (upper - lower).tolist(),
        },
        "clearance": clearances,
        "containment": containment,
        "exterior_containment": exterior_containment,
        "integrity": integrity,
        "dynamics": trials,
        "limitations": omissions,
    }


def compact_report(report: dict[str, Any], *, worst_pairs: int = 100) -> dict[str, Any]:
    """Retain every failing/incomplete witness and the closest pairs for review.

    This changes report detail only: every acceptance result and the number of
    generated pairs, poses and queries remain identical to the full report.
    """
    if worst_pairs < 0:
        raise ValueError("worst_pairs must be nonnegative")
    pairs = report["clearance"]["pairs"]
    closest = sorted(range(len(pairs)), key=lambda i: pairs[i]["minimum_distance_m"])[:worst_pairs]
    retained = set(closest) | {i for i, row in enumerate(pairs) if row["status"] != "pass"}
    result = dict(report)
    result["clearance"] = dict(report["clearance"], pairs=[row for i, row in enumerate(pairs) if i in retained])
    result["report_detail"] = {
        "compact": True,
        "generated_pair_count": len(pairs),
        "retained_pair_count": len(retained),
        "omitted_passing_pair_count": len(pairs) - len(retained),
        "retention_rule": f"All failing/incomplete pairs and the {worst_pairs} smallest recorded distance bounds.",
        "full_report_omission": "Passing pair witnesses outside that selection are omitted from this file only.",
        "regeneration": "Rerun the recorded invocation with python -m examples.mini_tyrannosaurus_rex.juvenile.validate, omitting --compact.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "model.xml")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--output", type=Path, default=HERE / "validation.json")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=635)
    parser.add_argument("--joint-points", type=int, default=5)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--gesture-duration", type=float, default=10)
    parser.add_argument("--skip-dynamics", action="store_true")
    parser.add_argument(
        "--compact", action="store_true", help="Retain all failing/incomplete and the closest 100 pair witnesses"
    )
    args = parser.parse_args()
    report = validate(
        args.model,
        args.manifest,
        samples=args.samples,
        seed=args.seed,
        joint_points=args.joint_points,
        duration=args.duration,
        gesture_duration=args.gesture_duration,
        run_dynamics=not args.skip_dynamics,
    )
    if args.compact:
        report = compact_report(report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "scoped_simulation_pass": report["scoped_simulation_pass"],
                "clearance": report["clearance"]["counts"],
                "output": str(args.output),
            }
        )
    )
    return 1 if report["status"] == "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
