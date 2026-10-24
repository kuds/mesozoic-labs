"""Reproducible assembly/contact/stance checks; no walking certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .demo import MODEL_PATH, gesture_controls, load_model, set_support


def visual_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """World bounds of actual visual surfaces, excluding floor and packaging boxes."""
    lower, upper = [], []
    for g in range(model.ngeom):
        if model.geom_group[g] not in (1, 2):
            continue
        rotation = data.geom_xmat[g].reshape(3, 3)
        pos, size, kind = data.geom_xpos[g], model.geom_size[g], model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[a : a + n] @ rotation.T + pos
            lower.append(points.min(axis=0))
            upper.append(points.max(axis=0))
            continue
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rotation) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            extent = np.sqrt(np.sum((rotation * size) ** 2, axis=1))
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[0] * np.linalg.norm(rotation[:, :2], axis=1) + size[1] * np.abs(rotation[:, 2])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + size[1] * np.abs(rotation[:, 2])
        else:
            extent = np.full(3, size[0])
        lower.append(pos - extent)
        upper.append(pos + extent)
    return np.min(lower, axis=0), np.max(upper, axis=0)


def internal_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    floor = model.geom("floor").id
    return [
        dict(pair=[model.geom(c.geom1).name, model.geom(c.geom2).name], penetration_m=max(0.0, -float(c.dist)))
        for c in data.contact[: data.ncon]
        if floor not in (c.geom1, c.geom2)
    ]


def run_trial(
    model: mujoco.MjModel, *, duration: float = 10, supported: bool = False, gesture: bool = False
) -> dict[str, Any]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    set_support(model, data, supported)
    start_height = float(data.qpos[2])
    min_height, max_tilt, max_internal, max_limit_error = start_height, 0.0, 0.0, 0.0
    maximum_torque = np.zeros(model.nu)
    sum_squares = np.zeros(model.nu)
    minimum_q, maximum_q = data.qpos[7:].copy(), data.qpos[7:].copy()
    finite = True
    steps = math.ceil(duration / model.opt.timestep)
    for _ in range(steps):
        if gesture:
            data.ctrl[:] = gesture_controls(model, float(data.time))
        mujoco.mj_step(model, data)
        finite = finite and all(
            np.isfinite(v).all() for v in (data.qpos, data.qvel, data.actuator_force, data.sensordata)
        )
        min_height = min(min_height, float(data.qpos[2]))
        up = data.xmat[model.body("torso").id].reshape(3, 3)[:, 2]
        max_tilt = max(max_tilt, math.degrees(math.acos(float(np.clip(up[2], -1, 1)))))
        maximum_torque = np.maximum(maximum_torque, np.abs(data.actuator_force))
        sum_squares += data.actuator_force**2
        minimum_q = np.minimum(minimum_q, data.qpos[7:])
        maximum_q = np.maximum(maximum_q, data.qpos[7:])
        max_internal = max(max_internal, max((x["penetration_m"] for x in internal_contacts(model, data)), default=0.0))
        limits = model.jnt_range[1:]
        max_limit_error = max(
            max_limit_error, float(np.max(np.maximum(limits[:, 0] - data.qpos[7:], data.qpos[7:] - limits[:, 1])))
        )
    cap = model.actuator_forcerange[:, 1]
    report = {
        "duration_s": float(data.time),
        "supported": supported,
        "gesture": gesture,
        "finite": bool(finite),
        "warning_counts": data.warning.number.tolist(),
        "min_height_ratio": min_height / start_height,
        "max_torso_tilt_deg": max_tilt,
        "maximum_internal_penetration_m": max_internal,
        "max_joint_limit_overshoot_rad": max_limit_error,
        "peak_torque_nm": maximum_torque.tolist(),
        "rms_torque_nm": np.sqrt(sum_squares / steps).tolist(),
        "max_torque_cap_fraction": float(np.max(maximum_torque / np.maximum(cap, 1e-12))),
        "joint_excursion_rad": (maximum_q - minimum_q).tolist(),
        "final_foot_contact_force_n": data.sensordata[-8:].tolist(),
        "external_support_active": bool(data.eq_active[model.equality("display_support").id]),
    }
    report["passed"] = bool(
        finite
        and not np.any(data.warning.number)
        and min_height > 0.9 * start_height
        and max_tilt < 10
        and max_internal < 0.001
        and max_limit_error < 0.03
        and report["max_torque_cap_fraction"] <= 1.000001
        and report["external_support_active"] == supported
    )
    return report


def check_model(*, duration: float = 10) -> dict[str, Any]:
    model, data = load_model()
    low, high = visual_bounds(model, data)
    sources = [
        MODEL_PATH,
        MODEL_PATH.with_name("components.json"),
        Path(__file__),
        MODEL_PATH.with_name("demo.py"),
        MODEL_PATH.with_name("build_model.py"),
        MODEL_PATH.with_name("head_design.py"),
        MODEL_PATH.with_name("proportions.py"),
    ]
    sources += sorted(MODEL_PATH.parent.glob("assets/*.obj"))
    trials = {
        "free_stance": run_trial(model, duration=duration),
        "free_standing_gesture": run_trial(model, duration=duration, gesture=True),
        "supported_gesture": run_trial(model, duration=duration, supported=True, gesture=True),
    }
    report = {
        "status": "concept_assembly_and_short_stance_regression_not_walking_validation",
        "mujoco_version": mujoco.__version__,
        "mass_kg": float(model.body_mass.sum()),
        "nq": model.nq,
        "nv": model.nv,
        "actuators": model.nu,
        "sensors": model.nsensor,
        "actuator_names": [model.actuator(i).name for i in range(model.nu)],
        "joint_names": [model.joint(i).name for i in range(1, model.njnt)],
        "home": {
            "bounds_min_m": low.tolist(),
            "bounds_max_m": high.tolist(),
            "length_width_height_m": (high - low).tolist(),
            "com_m": data.subtree_com[model.body("torso").id].tolist(),
            "contact_count": data.ncon,
            "internal_contacts": internal_contacts(model, data),
            "display_support_active": bool(data.eq_active[0]),
        },
        "trials": trials,
        "source_sha256": {
            str(p.relative_to(MODEL_PATH.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "limitations": [
            "Assigned masses and component-box inertias are not weighed CAD properties.",
            "Provisional motor envelopes do not establish bearings, connectors or mounting compatibility.",
            "Ideal torque-bounded position drives omit torque-speed, thermal, electrical, latency and backlash behavior.",
            "Gesture and stance checks do not demonstrate walking, push recovery or full-range collision clearance.",
            "Foot touch readings and RGB/stereo camera poses are ideal simulation sensors, not calibrated hardware.",
        ],
    }
    report["passed"] = bool(
        abs(report["mass_kg"] - 7.957) < 1e-9
        and model.nu == 18
        and np.allclose(model.opt.gravity, [0, 0, -9.81])
        and model.joint("root").type[0] == mujoco.mjtJoint.mjJNT_FREE
        and model.neq == 1
        and not model.eq_active0.any()
        and not model.body_gravcomp.any()
        and not internal_contacts(model, data)
        and 0.58 <= high[2] - low[2] <= 0.59
        and not report["home"]["display_support_active"]
        and all(t["passed"] for t in trials.values())
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--output", type=Path, default=MODEL_PATH.with_name("validation.json"))
    args = parser.parse_args()
    report = check_model(duration=args.duration)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "mass_kg": report["mass_kg"],
                "dimensions_m": report["home"]["length_width_height_m"],
                "trials": {
                    name: {
                        k: v
                        for k, v in trial.items()
                        if k in ("passed", "min_height_ratio", "max_torso_tilt_deg", "maximum_internal_penetration_m")
                    }
                    for name, trial in report["trials"].items()
                },
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
