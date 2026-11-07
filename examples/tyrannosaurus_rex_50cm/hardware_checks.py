"""Check the candidate's catalog envelopes, installation space and mass ledger.

Passing these planning criteria does not qualify wiring, cooling, mounts or
measured centers of mass. Unknown hardware evidence always remains incomplete.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations, product
from typing import Any

import mujoco
import numpy as np

from environments.shared.robot_clearance import clearance_distance
from environments.shared.robot_design_spec import ClearancePair, PoseSweep, RobotDesignSpec
from environments.shared.robot_design_validation import screen_design

from .mechanical_checks import EXPRESSION_JOINTS, LEG_JOINTS

PAIR_CLEARANCE_M = 0.002
INTERNAL_EDGE_CLEARANCE_M = 0.001
_BASES = {"vendor_complete", "vendor_pcb", "related_revision", "planning"}
_SIGNS = np.array(list(product((-1, 1), repeat=3)))


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} and finite.")
    return result


def _vector(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} must contain three finite numbers.") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain three finite numbers.")
    if (positive and np.any(result <= 0)) or (nonnegative and np.any(result < 0)):
        raise ValueError(f"{label} has invalid dimensions.")
    return result


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text.")
    return value


def _axes(value: Any) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value) or sorted(value) != [0, 1, 2]:
        raise ValueError("catalog_axes must be a permutation of [0, 1, 2].")
    return value


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, value: Any) -> int:
    name = _name(value, "Model reference")
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Unknown model reference: {name}")
    return int(result)


def _rotation(model: mujoco.MjModel, geom: int) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, model.geom_quat[geom])
    return matrix.reshape(3, 3)


def _catalog_fit(model: mujoco.MjModel, geom: int, item: dict, axes: list[int]) -> dict[str, Any]:
    """Check the actual primitive, including circular cylinder cross-sections."""
    if item["dimensions_m"] is None:
        return {"status": "incomplete", "dimension_basis": item["dimension_basis"], "required_size_m": None}
    full_size = np.array(item["dimensions_m"]) + item["tolerance_m"]
    required = full_size[axes]
    rotation = _rotation(model, geom)
    size, kind = model.geom_size[geom], model.geom_type[geom]
    if item.get("shape", "box") == "cylinder":
        axis = np.eye(3)[axes.index(2)] @ rotation
        radius, half_length = max(full_size[:2]) / 2, full_size[2] / 2
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            if abs(axis[2]) < 1 - 1e-10:
                return {
                    "status": "failed",
                    "dimension_basis": item["dimension_basis"],
                    "required_body_axis_size_m": required.tolist(),
                    "cylinder_axis_alignment": float(abs(axis[2])),
                }
            margins = np.array([size[0] - radius, size[1] - half_length])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            margins = size - (radius * np.sqrt(np.maximum(0, 1 - axis**2)) + half_length * np.abs(axis))
        else:
            raise ValueError("Catalog hardware requires box or cylinder model geometry.")
    else:
        corners = (_SIGNS * required / 2) @ rotation
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            margins = size - np.max(np.abs(corners), axis=0)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            margins = np.array(
                [size[0] - np.linalg.norm(corners[:, :2], axis=1).max(), size[1] - abs(corners[:, 2]).max()]
            )
        else:
            raise ValueError("Catalog hardware requires box or cylinder model geometry.")
    fits = bool(np.all(margins >= -1e-10))
    return {
        "status": "failed"
        if not fits
        else ("passed" if item["dimension_basis"] == "vendor_complete" else "incomplete"),
        "dimension_basis": item["dimension_basis"],
        "required_body_axis_size_m": required.tolist(),
        "shape_fit_margins_m": margins.tolist(),
    }


def check_hardware(model: mujoco.MjModel, ledger: dict, manifest: dict) -> dict[str, Any]:
    """Return finite, serializable evidence without changing any input.

    Invalid declarations raise ValueError. Geometric, quantity and mass failures
    remain report rows so one run can expose all known design conflicts.
    """
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Hardware manifest requires schema_version 1.")
    catalog = manifest.get("catalog")
    assemblies, mounted = manifest.get("assemblies"), manifest.get("mounted_hardware")
    expected, uncertainties = manifest.get("expected_quantities"), manifest.get("uncertainties")
    if not isinstance(catalog, dict) or not catalog or not isinstance(assemblies, list) or not assemblies:
        raise ValueError("Catalog and assemblies must be nonempty collections.")
    if not isinstance(mounted, list) or not isinstance(expected, dict) or not isinstance(uncertainties, list):
        raise ValueError("Mounted hardware, expected quantities and uncertainties must be declared.")
    for note in uncertainties:
        _name(note, "Uncertainty")
    for key, item in catalog.items():
        _name(key, "Catalog identifier")
        if not isinstance(item, dict) or item.get("dimension_basis") not in _BASES:
            raise ValueError(f"Invalid catalog evidence basis: {key}")
        if not {"dimensions_m", "mass_kg", "source_url"} <= item.keys():
            raise ValueError(f"Missing hardware evidence must be declared explicitly with null: {key}")
        _name(item.get("part"), "Part")
        _name(item.get("notes"), "Catalog notes")
        if item.get("source_url") is not None:
            _name(item["source_url"], "Source URL")
        elif item["dimension_basis"] != "planning":
            raise ValueError(f"Sourced catalog part requires a source URL: {key}")
        if item.get("shape", "box") not in {"box", "cylinder"}:
            raise ValueError(f"Unsupported catalog shape: {key}")
        if "shaft_axis" in item and (type(item["shaft_axis"]) is not int or item["shaft_axis"] not in (0, 1, 2)):
            raise ValueError(f"Catalog shaft_axis must identify one catalog dimension: {key}")
        if "shaft_axis" in item:
            _vector(item.get("shaft_offset_m"), key + " shaft offset")
        if item.get("dimensions_m") is not None:
            _vector(item["dimensions_m"], key + " dimensions", positive=True)
        _vector(item.get("tolerance_m"), key + " tolerance", nonnegative=True)
        if item.get("mass_kg") is not None:
            _number(item["mass_kg"], key + " mass", positive=True)
        _number(item.get("mass_tolerance_kg"), key + " mass tolerance")
    if any(key not in catalog or type(count) is not int or count < 0 for key, count in expected.items()):
        raise ValueError("Expected quantities must reference catalog identifiers and nonnegative integer counts.")
    if not isinstance(ledger, dict) or not isinstance(ledger.get("components"), list):
        raise ValueError("A component mass ledger is required.")
    components: dict[str, dict] = {}
    body_mass, body_moment = np.zeros(model.nbody), np.zeros((model.nbody, 3))
    for component in ledger["components"]:
        if not isinstance(component, dict):
            raise ValueError("Each ledger component must be an object.")
        name = _name(component.get("name"), "Component name")
        if name in components:
            raise ValueError(f"Duplicate ledger component: {name}")
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, component.get("body"))
        mass = _number(component.get("mass_kg"), name + " mass", positive=True)
        pos = _vector(component.get("pos_m"), name + " position")
        components[name] = component
        body_mass[body] += mass
        body_moment[body] += mass * pos
    checks: list[dict[str, Any]] = []

    def record(name: str, category: str, passed: bool, **evidence: Any) -> None:
        checks.append({"name": name, "category": category, "status": "passed" if passed else "failed", **evidence})

    record(
        "compiled_body_mass",
        "mass_reconciliation",
        bool(np.allclose(body_mass, model.body_mass, atol=1e-8, rtol=0)),
        maximum_error_kg=float(np.max(abs(body_mass - model.body_mass))),
    )
    occupied = body_mass > 0
    ledger_centers = body_moment[occupied] / body_mass[occupied, None]
    record(
        "compiled_body_centers",
        "mass_reconciliation",
        bool(np.allclose(ledger_centers, model.body_ipos[occupied], atol=1e-8, rtol=0)),
        maximum_error_m=float(np.max(abs(ledger_centers - model.body_ipos[occupied]))) if occupied.any() else 0,
    )
    total = float(body_mass.sum())
    if "mass_kg" in ledger:
        declared = _number(ledger["mass_kg"], "Ledger total", positive=True)
        record(
            "ledger_total",
            "mass_reconciliation",
            abs(declared - total) <= 1e-8,
            declared_mass_kg=declared,
            component_mass_kg=total,
        )
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, _id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)
    root_body = int(model.jnt_bodyid[_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")])
    world_moment = np.zeros(3)
    for component in components.values():
        body = model.body(component["body"]).id
        world_moment += component["mass_kg"] * (data.xpos[body] + data.xmat[body].reshape(3, 3) @ component["pos_m"])
    ledger_com = world_moment / total if total else np.zeros(3)
    record(
        "home_center_of_mass",
        "mass_reconciliation",
        bool(np.allclose(ledger_com, data.subtree_com[root_body], atol=1e-8, rtol=0)),
        ledger_com_m=ledger_com.tolist(),
        compiled_com_m=data.subtree_com[root_body].tolist(),
    )
    counts: Counter[str] = Counter()
    seen_geometries: set[str] = set()
    seen_units: set[str] = set()
    assigned_components: set[str] = set()

    def reference(entry: dict, *, unit: bool = False) -> tuple[int, dict, list[int]]:
        if not isinstance(entry, dict):
            raise ValueError("Each hardware entry must be an object.")
        geom_name = _name(entry.get("geometry"), "Hardware geometry")
        if geom_name in seen_geometries:
            raise ValueError(f"Duplicate hardware geometry: {geom_name}")
        seen_geometries.add(geom_name)
        geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        key = _name(entry.get("catalog"), "Catalog reference")
        if key not in catalog:
            raise ValueError(f"Unknown catalog reference: {key}")
        if unit:
            unit_id = _name(entry.get("id"), "Hardware unit")
            if unit_id in seen_units:
                raise ValueError(f"Duplicate hardware unit: {unit_id}")
            seen_units.add(unit_id)
        counts[key] += 1
        return geom, catalog[key], _axes(entry.get("catalog_axes"))

    def allocation(entry: dict) -> dict:
        if not isinstance(entry, dict):
            raise ValueError("Each component allocation must be an object.")
        name = _name(entry.get("component"), "Component allocation")
        if name not in components or name in assigned_components:
            raise ValueError(f"Unknown or repeated mass allocation: {name}")
        assigned_components.add(name)
        return components[name]

    assembly_ids: set[str] = set()
    for assembly in assemblies:
        if not isinstance(assembly, dict):
            raise ValueError("Each assembly must be an object.")
        name = _name(assembly.get("id"), "Assembly identifier")
        if name in assembly_ids:
            raise ValueError(f"Duplicate assembly: {name}")
        assembly_ids.add(name)
        component = allocation(assembly)
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, assembly.get("body"))
        geom_name = _name(assembly.get("geometry"), "Assembly geometry")
        if geom_name in seen_geometries:
            raise ValueError(f"Duplicate hardware geometry: {geom_name}")
        seen_geometries.add(geom_name)
        geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        container = _id(model, mujoco.mjtObj.mjOBJ_GEOM, assembly.get("container"))
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Installed assembly reservations must be boxes.")
        if model.geom_type[container] != mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            raise ValueError("Assembly containment requires one convex ellipsoid container.")
        pos = _vector(assembly.get("position_m"), name + " position")
        size = _vector(assembly.get("installed_size_m"), name + " installed size", positive=True)
        budget = _number(assembly.get("mass_budget_kg"), name + " mass budget", positive=True)
        wall = _number(assembly.get("wall_clearance_m"), name + " wall clearance")
        record(
            name + ":installed_geometry",
            "declaration",
            bool(
                model.geom_bodyid[geom] == body
                and np.allclose(model.geom_pos[geom], pos, atol=1e-10, rtol=0)
                and np.allclose(2 * model.geom_size[geom], size, atol=1e-10, rtol=0)
                and np.allclose(_rotation(model, geom), np.eye(3), atol=1e-10, rtol=0)
            ),
        )
        record(
            name + ":component_allocation",
            "mass_reconciliation",
            bool(
                component["body"] == assembly["body"]
                and abs(component["mass_kg"] - budget) <= 1e-10
                and np.allclose(component["pos_m"], pos, atol=1e-10, rtol=0)
            ),
            component_mass_kg=component["mass_kg"],
            budget_mass_kg=budget,
        )
        corners = (_SIGNS * model.geom_size[geom]) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local_corners = (corners - data.geom_xpos[container]) @ data.geom_xmat[container].reshape(3, 3)
        inner = model.geom_size[container] - wall
        scores = np.sum((local_corners / inner) ** 2, axis=1) if np.all(inner > 0) else None
        record(
            name + ":container",
            "containment",
            scores is not None and bool(np.all(scores <= 1 + 1e-10)),
            container=assembly["container"],
            wall_clearance_m=wall,
            maximum_ellipsoid_score=float(scores.max()) if scores is not None else None,
        )
        contents = assembly.get("contents")
        if not isinstance(contents, list) or not contents:
            raise ValueError(f"Assembly contents must be explicitly declared: {name}")
        known_mass, unknown_mass = 0.0, []
        for unit in contents:
            content_geom, item, axes = reference(unit, unit=True)
            offset = _vector(unit.get("offset_m"), unit["id"] + " offset")
            content_size = _vector(unit.get("size_m"), unit["id"] + " size", positive=True)
            if model.geom_type[content_geom] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("Installed content declarations require box geometry.")
            record(
                unit["id"] + ":content_geometry",
                "declaration",
                bool(
                    model.geom_bodyid[content_geom] == body
                    and np.allclose(model.geom_pos[content_geom], pos + offset, atol=1e-10, rtol=0)
                    and np.allclose(2 * model.geom_size[content_geom], content_size, atol=1e-10, rtol=0)
                    and np.allclose(_rotation(model, content_geom), np.eye(3), atol=1e-10, rtol=0)
                ),
            )
            checks.append(
                {"name": unit["id"], "category": "catalog_fit", **_catalog_fit(model, content_geom, item, axes)}
            )
            margins = size / 2 - abs(offset) - content_size / 2
            record(
                unit["id"] + ":installed_edge",
                "internal_containment",
                bool(np.all(margins >= INTERNAL_EDGE_CLEARANCE_M - 1e-10)),
                edge_clearances_m=margins.tolist(),
                required_clearance_m=INTERNAL_EDGE_CLEARANCE_M,
            )
            if item["mass_kg"] is None:
                unknown_mass.append(unit["id"])
            else:
                known_mass += item["mass_kg"] + item["mass_tolerance_kg"]
        record(
            name + ":mass_budget",
            "mass_budget",
            budget >= known_mass - 1e-10,
            mass_budget_kg=budget,
            known_catalog_mass_with_tolerance_kg=known_mass,
            residual_budget_kg=budget - known_mass,
            unknown_mass_units=unknown_mass,
        )
        if unknown_mass:
            checks.append(
                {
                    "name": name + ":unweighed_contents",
                    "category": "mass_evidence",
                    "status": "incomplete",
                    "unknown_mass_units": unknown_mass,
                }
            )
        for first, second in combinations(contents, 2):
            distance = clearance_distance(
                model,
                data,
                model.geom(first["geometry"]).id,
                model.geom(second["geometry"]).id,
                required_clearance_m=PAIR_CLEARANCE_M,
                search_distance_m=1.0,
            )
            status = (
                "passed"
                if distance["min_distance_m"] >= PAIR_CLEARANCE_M
                else ("incomplete" if distance["distance_is_lower_bound"] else "failed")
            )
            checks.append(
                {
                    "name": first["id"] + "/" + second["id"],
                    "category": "internal_clearance",
                    "status": status,
                    "required_clearance_m": PAIR_CLEARANCE_M,
                    **distance,
                }
            )
    for entry in mounted:
        component = allocation(entry)
        geom, item, axes = reference(entry)
        record(
            entry["component"] + ":placement",
            "declaration",
            bool(
                model.geom_bodyid[geom] == model.body(component["body"]).id
                and np.allclose(model.geom_pos[geom], component["pos_m"], atol=1e-10, rtol=0)
            ),
        )
        checks.append({"name": entry["component"], "category": "catalog_fit", **_catalog_fit(model, geom, item, axes)})
        if "shaft_axis" in item:
            joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.get("joint"))
            body = int(model.geom_bodyid[geom])
            shaft_body = np.eye(3)[axes.index(item["shaft_axis"])]
            shaft_world = data.xmat[body].reshape(3, 3) @ shaft_body
            alignment = float(abs(np.dot(shaft_world, data.xaxis[joint])))
            record(
                entry["component"] + ":shaft_alignment",
                "shaft_orientation",
                alignment >= 1 - 1e-8,
                joint=entry["joint"],
                absolute_axis_dot=alignment,
                scope="Planned shaft direction; horn, idler, bearings and mechanical coupling remain unverified.",
            )
            offset_body = np.array(item["shaft_offset_m"])[axes]
            shaft_point = data.geom_xpos[geom] + data.xmat[body].reshape(3, 3) @ offset_body
            delta = shaft_point - data.xanchor[joint]
            perpendicular = delta - np.dot(delta, data.xaxis[joint]) * data.xaxis[joint]
            shaft_distance = float(np.linalg.norm(perpendicular))
            record(
                entry["component"] + ":shaft_line",
                "shaft_location",
                shaft_distance <= 1e-8,
                joint=entry["joint"],
                perpendicular_distance_m=shaft_distance,
                scope="Catalog shaft line through the model joint; axial placement and mounting hardware remain unqualified.",
            )
        if item["mass_kg"] is None:
            checks.append({"name": entry["component"], "category": "mass_evidence", "status": "incomplete"})
        else:
            minimum = item["mass_kg"] + item["mass_tolerance_kg"]
            record(
                entry["component"] + ":mass_budget",
                "mass_budget",
                component["mass_kg"] >= minimum - 1e-10,
                allocated_mass_kg=component["mass_kg"],
                known_catalog_mass_with_tolerance_kg=minimum,
            )
    record(
        "catalog_quantities",
        "quantities",
        all(counts[key] == expected.get(key, 0) for key in counts.keys() | expected.keys()),
        actual_quantities=dict(counts),
        expected_quantities=deepcopy(expected),
    )

    def samples(joint: str, count: int) -> tuple[float, ...]:
        lower, upper = model.joint(joint).range
        return tuple(float(value) for value in np.linspace(float(lower), float(upper), count))

    assembly_geoms = [entry["geometry"] for entry in assemblies]
    mounted_geoms = [entry["geometry"] for entry in mounted]
    pairs = (
        *combinations(assembly_geoms, 2),
        *product(assembly_geoms, mounted_geoms),
        *(
            (first["geometry"], second["geometry"])
            for first, second in combinations(mounted, 2)
            if not (first["catalog"] == second["catalog"] == "rmd")
        ),
    )
    spec = RobotDesignSpec(
        robot_id="tyrannosaurus_rex_medium_hardware",
        root_joint="root",
        home_keyframe="home",
        feet={"left": ("left_sole_collision",), "right": ("right_sole_collision",)},
        loads=(),
        sweeps=(
            PoseSweep("expression_grid", {joint: samples(joint, 3) for joint in EXPRESSION_JOINTS}),
            *(PoseSweep(joint + "_range", {joint: samples(joint, 5)}) for joint in LEG_JOINTS),
        ),
        clearances=tuple(ClearancePair(first, second, PAIR_CLEARANCE_M) for first, second in pairs),
        assumptions=(
            "Installed assembly boxes include planning service space. Outer shell ellipsoids describe hollow containment, not occupied solid volumes.",
            "The 729-point expression grid and independent five-point leg sweeps do not cover combined leg/expression motion or unsampled angles.",
        ),
    )
    packing = screen_design(model, spec)
    failed = any(row["status"] == "failed" for row in checks) or packing["status"] == "failed"
    return {
        "schema_version": 1,
        "status": "failed" if failed else "incomplete",
        "known_limits_passed": not failed,
        "model_mass_kg": float(model.body_mass.sum()),
        "home_com_m": data.subtree_com[root_body].tolist(),
        "checks": checks,
        "packing": packing,
        "coverage": {
            "assemblies": len(assemblies),
            "content_units": len(seen_units),
            "mounted_units": len(mounted),
            "packing_poses": packing["coverage"]["sampled_poses"],
            "packing_pairs": len(pairs),
            "internal_edge_clearance_m": INTERNAL_EDGE_CLEARANCE_M,
            "pair_clearance_m": PAIR_CLEARANCE_M,
        },
        "specification": deepcopy(manifest),
        "uncertainties": [
            *uncertainties,
            "Mass centers and inertias are centered planning allocations; installation, wiring, cooling and structural mounting remain unqualified.",
        ],
    }
