"""Reproducible concept-design probes; not fabrication, thermal or walking approval.

Run ``python -m examples.mini_tyrannosaurus_rex.engineering --json report.json``.
Motor-box separation excludes horns, connectors, cables and manufacturing tolerances.
Contact sweeps inspect the model's collision proxies, including its exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared import robot_validation

from . import balance_checks

MODEL_PATH = Path(__file__).with_name("model.xml")


def home_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Geometry-only reset; dynamic holds use the guarded stance trial instead."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    support = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "display_support")
    if support >= 0:
        data.eq_active[support] = False
    mujoco.mj_forward(model, data)
    return data


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float]) -> None:
    """Set hinge positions, then honor polynomial joint-linkage equalities."""
    for name, angle in pose.items():
        joint = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint]] = angle
    for eq in range(model.neq):
        if not data.eq_active[eq] or model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        first, second = model.eq_obj1id[eq], model.eq_obj2id[eq]
        address = model.jnt_qposadr[first]
        relative = 0.0 if second < 0 else data.qpos[model.jnt_qposadr[second]] - model.qpos0[model.jnt_qposadr[second]]
        data.qpos[address] = model.qpos0[address] + np.polynomial.polynomial.polyval(relative, model.eq_data[eq, :5])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def command_joint_ranges(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    """Direct joint-position actuator ranges, intersected with mechanical limits."""
    result = {}
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint = int(model.actuator_trnid[actuator, 0])
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        low, high = model.actuator_ctrlrange[actuator]
        if model.jnt_limited[joint]:
            low, high = max(low, model.jnt_range[joint, 0]), min(high, model.jnt_range[joint, 1])
        result[model.joint(joint).name] = (float(low), float(high))
    return result


def motor_cases(model: mujoco.MjModel) -> list[int]:
    return [
        geom
        for geom in range(model.ngeom)
        if model.geom(geom).name.endswith("_servo_case") and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
    ]


def box_separation(model: mujoco.MjModel, data: mujoco.MjData, first: int, second: int) -> float:
    """Separating-axis gap; positive proves disjoint boxes, not exact distance."""
    left, right = data.geom_xmat[first].reshape(3, 3), data.geom_xmat[second].reshape(3, 3)
    offset = data.geom_xpos[second] - data.geom_xpos[first]
    axes = list(left.T) + list(right.T) + [np.cross(a, b) for a in left.T for b in right.T]
    gaps = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis = axis / norm
        radii = np.abs(left.T @ axis) @ model.geom_size[first] + np.abs(right.T @ axis) @ model.geom_size[second]
        gaps.append(float(abs(offset @ axis) - radii))
    return max(gaps)


def motor_clearance_sweep(model: mujoco.MjModel, samples: int = 256, seed: int = 635) -> dict[str, Any]:
    """Home plus seeded commanded poses; sampling is not exhaustive clearance."""
    if samples < 1:
        raise ValueError("samples must be positive")
    data, rng = home_data(model), np.random.default_rng(seed)
    pairs = list(itertools.combinations(motor_cases(model), 2))
    ranges = command_joint_ranges(model)
    worst, overlaps, witness = math.inf, 0, {}
    for sample in range(samples + 1):
        pose = {} if sample == 0 else {name: float(rng.uniform(*bounds)) for name, bounds in ranges.items()}
        set_pose(model, data, pose)
        gaps = [(box_separation(model, data, a, b), a, b) for a, b in pairs]
        if not gaps:
            continue
        gap, first, second = min(gaps)
        overlaps += int(gap < -1e-6)
        if gap < worst:
            worst, witness = gap, {"pose_rad": pose, "pair": [model.geom(first).name, model.geom(second).name]}
    return {
        "motor_cases": len(motor_cases(model)),
        "poses": samples + 1,
        "seed": seed,
        "overlap_poses": overlaps,
        "minimum_separating_axis_gap_m": None if not pairs else worst,
        "witness": witness,
    }


def upper_body_collision_sweep(model: mujoco.MjModel, points: int = 3) -> dict[str, Any]:
    """Cartesian upper-body command grid with legs at home and linkage engaged."""
    if points < 2:
        raise ValueError("points must be at least two")
    ranges = {
        name: bounds
        for name, bounds in command_joint_ranges(model).items()
        if not any(part in name for part in ("hip_", "knee", "ankle"))
    }
    data, bad_poses, worst = home_data(model), 0, 0.0
    witnesses: list[dict[str, Any]] = []
    count = 0
    for values in itertools.product(*(np.linspace(*bounds, points) for bounds in ranges.values())):
        count += 1
        pose = dict(zip(ranges, map(float, values), strict=True))
        set_pose(model, data, pose)
        contacts: list[dict[str, Any]] = []
        for contact in data.contact[: data.ncon]:
            names = [model.geom(int(geom)).name for geom in contact.geom]
            if "floor" in names or contact.dist >= -1e-6:
                continue
            contacts.append({"pair": names, "penetration_m": float(-contact.dist)})
        if contacts:
            bad_poses += 1
            maximum = max(contact["penetration_m"] for contact in contacts)
            worst = max(worst, maximum)
            witnesses.append({"pose_rad": pose, "maximum_penetration_m": maximum, "contacts": contacts})
    return {
        "poses": count,
        "points_per_joint": points,
        "joint_ranges_rad": ranges,
        "overlap_poses": bad_poses,
        "maximum_penetration_m": worst,
        "worst_poses": sorted(witnesses, key=lambda row: row["maximum_penetration_m"], reverse=True)[:5],
        "scope": "Collision proxies and existing contact exclusions; not hollow-shell CAD or cable clearance.",
    }


def combined_collision_sweep(model: mujoco.MjModel, samples: int = 1000, seed: int = 636) -> dict[str, Any]:
    """Sample simultaneous leg and expression commands with linked shoulders."""
    if samples < 1:
        raise ValueError("samples must be positive")
    data, rng = home_data(model), np.random.default_rng(seed)
    ranges = command_joint_ranges(model)
    bad, maximum, witness = 0, 0.0, {}
    for _ in range(samples):
        pose = {name: float(rng.uniform(*bounds)) for name, bounds in ranges.items()}
        set_pose(model, data, pose)
        penetrations = []
        for contact in data.contact[: data.ncon]:
            names = [model.geom(int(geom)).name for geom in contact.geom]
            if "floor" not in names and contact.dist < -1e-6:
                penetrations.append((float(-contact.dist), names))
        if penetrations:
            bad += 1
            depth, names = max(penetrations)
            if depth > maximum:
                maximum, witness = depth, {"pose_rad": pose, "pair": names}
    return {"poses": samples, "seed": seed, "overlap_poses": bad, "maximum_penetration_m": maximum, "witness": witness}


def leg_cap_sensitivity(model: mujoco.MjModel, duration: float = 10.0) -> dict[str, Any]:
    """Hold with at most .10 Nm leg caps, without strengthening weaker motors."""
    return balance_checks.stance_trial(model, duration=duration, leg_cap=0.10)


def static_equilibrium(model: mujoco.MjModel) -> dict[str, Any]:
    """Vertical two-foot gravity support distributed by lateral COM position.

    Uses each sole's lateral center and a common sagittal CoP under the COM.
    This is one load distribution, not a unique prediction of ground reactions.
    These are not measured actuator loads. An internal linked mechanism must
    combine its joint loads through the actual transmission ratio separately.
    """
    data = home_data(model)
    com = data.subtree_com[model.body("torso").id]
    total = float(model.body_mass.sum())
    left_y = data.site_xpos[model.site("left_sole_center").id, 1]
    right_y = data.site_xpos[model.site("right_sole_center").id, 1]
    fraction_left = float((com[1] - right_y) / (left_y - right_y))
    fractions = {"left": fraction_left, "right": 1 - fraction_left}
    loads, feet = data.qfrc_bias.copy(), {}
    for side in ("left", "right"):
        force = np.array([0.0, 0.0, -model.opt.gravity[2] * total * fractions[side]])
        site = model.site(f"{side}_sole_center").id
        position = data.site_xpos[site]
        moment = np.array([0.0, -force[2] * (com[0] - position[0]), 0.0])
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, linear, angular, site)
        loads -= linear.T @ force + angular.T @ moment
        feet[side] = position.tolist()
    joints = {
        model.joint(joint).name: float(loads[model.jnt_dofadr[joint]])
        for joint in range(model.njnt)
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
    }
    return {
        "mass_kg": total,
        "assumption": "Vertical reactions at sole lateral centers, sagittal CoP at COM x; one static load distribution.",
        "com_m": com.tolist(),
        "sole_centers_m": feet,
        "normal_force_fractions": fractions,
        "nonnegative_normal_forces": all(value >= 0 for value in fractions.values()),
        "joint_loads_Nm": joints,
        "base_wrench_residual": loads[:6].tolist(),
    }


def single_foot_cop(model: mujoco.MjModel, data: mujoco.MjData, side: str = "left") -> dict[str, Any]:
    """Required single-foot CoP from base wrench balance, not a feasible gait."""
    site = model.site(f"{side}_sole_center").id
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, linear, angular, site)
    root = model.joint("root").id
    address = model.jnt_dofadr[root]
    jacobian = np.vstack([linear, angular])[:, address : address + 6]
    wrench = np.linalg.solve(jacobian.T, data.qfrc_bias[address : address + 6])
    rotation = data.site_xmat[site].reshape(3, 3)
    force, moment = rotation.T @ wrench[:3], rotation.T @ wrench[3:]
    cop = np.array([-moment[1] / force[2], moment[0] / force[2]])
    half_extents = model.geom_size[model.geom(f"{side}_flat_pad_collision").id, :2]
    return {
        "side": side,
        "cop_from_sole_center_m": cop.tolist(),
        "contact_half_extents_m": half_extents.tolist(),
        "inside_contact_rectangle": bool(np.all(np.abs(cop) <= half_extents)),
        "normal_force_N": float(force[2]),
    }


def lateral_transfer_probe(model: mujoco.MjModel, points: int = 9) -> dict[str, Any]:
    """Limited flat-foot torso lean, counter-rotating both hip-roll hinges.

    This searches one posture family only. No yaw, swing-leg or dynamic balance
    strategy is searched; failure to reach a sole is not proof walking is impossible.
    """
    if points < 2:
        raise ValueError("points must be at least two")
    ranges = command_joint_ranges(model)
    joints = [f"{side}_hip_roll" for side in ("left", "right")]
    low = max(-ranges[name][1] for name in joints)
    high = min(-ranges[name][0] for name in joints)
    home = home_data(model)
    root_address = model.jnt_qposadr[model.joint("root").id]
    rows = []
    for angle in np.linspace(low, high, points):
        data = home_data(model)
        delta = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        rotation = np.zeros(4)
        mujoco.mju_mulQuat(rotation, delta, home.qpos[root_address + 3 : root_address + 7])
        data.qpos[root_address + 3 : root_address + 7] = rotation
        set_pose(model, data, {name: -float(angle) for name in joints})
        rows.append({"torso_roll_rad": float(angle), "required_support": single_foot_cop(model, data)})
    return {
        "home": {side: single_foot_cop(model, home, side) for side in ("left", "right")},
        "flat_foot_roll_sweep": rows,
        "scope": "Static one-family kinematic diagnostic; does not demonstrate foot unloading, stepping or walking.",
    }


def free_hold(model: mujoco.MjModel, duration: float = 10.0) -> dict[str, Any]:
    """Unsupported home hold at the model's exact limits; not thermal duty."""
    return balance_checks.stance_trial(model, duration=duration, leg_cap=None)


def build_report(
    model: mujoco.MjModel, *, samples: int = 256, points: int = 3, duration: float = 10.0
) -> dict[str, Any]:
    return {
        "mujoco_version": mujoco.__version__,
        "scope": "Concept geometry and ideal dynamics only; walking, hardware fit and continuous motor duty remain unverified.",
        "interface": {"nq": model.nq, "nv": model.nv, "actuators": model.nu, "sensor_definitions": model.nsensor},
        "body_mass_kg": {model.body(body).name: float(model.body_mass[body]) for body in range(1, model.nbody)},
        "equilibrium": static_equilibrium(model),
        "lateral_transfer": lateral_transfer_probe(model),
        "motor_clearance": motor_clearance_sweep(model, samples),
        "upper_body_contacts": upper_body_collision_sweep(model, points),
        "combined_contacts": combined_collision_sweep(model, samples),
        "free_hold": free_hold(model, duration),
        "leg_cap_sensitivity": leg_cap_sensitivity(model, duration),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write a reproducible numerical report.")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    report = build_report(
        mujoco.MjModel.from_xml_path(str(MODEL_PATH)), samples=args.samples, points=args.points, duration=args.duration
    )
    report["source_sha256"] = {
        str(path.relative_to(MODEL_PATH.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            MODEL_PATH,
            Path(__file__),
            Path(balance_checks.__file__),
            *sorted((MODEL_PATH.parent / "assets").glob("*.xml")),
        ]
    }
    report["source_sha256"]["../../environments/shared/robot_validation.py"] = hashlib.sha256(
        Path(robot_validation.__file__).read_bytes()
    ).hexdigest()
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
