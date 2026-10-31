"""Explicit inputs for sampled, uncoupled robot mechanical-design checks.

This specification does not register a training plant. Load axes describe freely
supported expression subtrees, not legs supporting ground reaction forces. The
first screen supports direct, unit-gear hinge actuators and no active equality
couplings; closed mechanisms need a constraint-aware load calculation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import mujoco
import numpy as np

EvidenceLevel = Literal["estimated", "vendor_specification", "measured"]
EVIDENCE_LEVELS = ("estimated", "vendor_specification", "measured")
_CLEARANCE_GEOM_TYPES = frozenset(
    (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_MESH,
    )
)


@dataclass(frozen=True)
class JointLoadSpec:
    joint: str
    actuator: str
    torque_evidence: EvidenceLevel = "estimated"
    bearing_limit_nm: float | None = None
    bearing_evidence: EvidenceLevel = "estimated"
    max_torque_fraction: float = 0.8


@dataclass(frozen=True)
class PoseSweep:
    """Cartesian samples for one group; other joints retain the home pose."""

    name: str
    samples: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class ClearancePair:
    first: str
    second: str
    min_distance_m: float


@dataclass(frozen=True)
class RobotDesignSpec:
    robot_id: str
    root_joint: str
    home_keyframe: str
    feet: dict[str, tuple[str, ...]]
    loads: tuple[JointLoadSpec, ...]
    sweeps: tuple[PoseSweep, ...]
    clearances: tuple[ClearancePair, ...]
    mass_evidence: EvidenceLevel = "estimated"
    max_poses: int = 10000
    assumptions: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the complete, JSON-serializable input specification."""
        return asdict(self)


def pose_count(spec: RobotDesignSpec) -> int:
    """Count one home pose and every group sample, including repeated home poses."""
    return 1 + sum(math.prod(len(values) for values in sweep.samples.values()) for sweep in spec.sweeps)


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model element names must be nonempty strings.")
    index = int(mujoco.mj_name2id(model, kind, name))
    if index < 0:
        raise ValueError(f"Unknown {kind.name}: {name}")
    return index


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite.") from error
    if isinstance(value, bool) or not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite.")
    return number


def _evidence(value: EvidenceLevel, name: str) -> None:
    if value not in EVIDENCE_LEVELS:
        raise ValueError(f"{name} must identify estimated, vendor_specification or measured evidence.")


def _descends_from(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def validate_spec(model: mujoco.MjModel, spec: RobotDesignSpec) -> None:
    """Resolve metadata and reject unsupported or inconsistent inputs, without mutation."""
    if not isinstance(spec.robot_id, str) or not spec.robot_id.strip():
        raise ValueError("robot_id must be nonempty.")
    _evidence(spec.mass_evidence, "mass_evidence")
    if not isinstance(spec.max_poses, int) or isinstance(spec.max_poses, bool) or spec.max_poses < 1:
        raise ValueError("max_poses must be a positive integer.")
    if any(not isinstance(note, str) or not note.strip() for note in spec.assumptions):
        raise ValueError("Assumptions must be nonempty text.")
    if not spec.loads and not spec.clearances:
        raise ValueError("Declare at least one load axis or clearance pair.")
    root = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.root_joint)
    if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("The robot root must be a free joint.")
    root_body = int(model.jnt_bodyid[root])
    home = _named_id(model, mujoco.mjtObj.mjOBJ_KEY, spec.home_keyframe)
    if not np.isfinite(model.key_qpos[home]).all():
        raise ValueError("The home keyframe must contain finite positions.")
    if not np.allclose(model.opt.gravity, [0, 0, -9.81], rtol=0, atol=1e-12):
        raise ValueError("Mechanical load checks require Earth gravity along -Z.")
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY:
        raise ValueError("Gravity physics must be enabled.")
    if np.any(model.eq_active0):
        raise ValueError("Active equality constraints require a constraint-aware load screen.")
    if np.any(model.body_gravcomp):
        raise ValueError("Gravity compensation is not supported by this load screen.")
    robot_bodies = np.flatnonzero(model.body_rootid == root_body)
    mass = model.body_mass[robot_bodies]
    inertia = model.body_inertia[robot_bodies]
    if not np.isfinite(mass).all() or not np.isfinite(inertia).all() or np.any(mass < 0) or np.any(inertia < 0):
        raise ValueError("Robot mass and inertia must be finite and nonnegative.")
    moving = robot_bodies[model.body_dofnum[robot_bodies] > 0]
    if np.any(model.body_mass[moving] <= 0) or np.any(model.body_inertia[moving] <= 0):
        raise ValueError("Moving joint bodies need positive mass and principal inertia.")

    def robot_geom(name: str) -> int:
        geom = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if model.body_rootid[model.geom_bodyid[geom]] != root_body:
            raise ValueError(f"Geometry {name} is outside the robot root subtree.")
        return geom

    def hinge(name: str) -> int:
        joint = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.body_rootid[model.jnt_bodyid[joint]] != root_body:
            raise ValueError(f"Joint {name} is outside the robot root subtree.")
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Joint {name} must be a hinge.")
        limits = model.jnt_range[joint]
        if not model.jnt_limited[joint] or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f"Joint {name} needs finite, ordered limits.")
        angle = model.key_qpos[home, model.jnt_qposadr[joint]]
        if angle < limits[0] - 1e-12 or angle > limits[1] + 1e-12:
            raise ValueError(f"Home pose exceeds the limits of {name}.")
        return joint

    if not spec.feet:
        raise ValueError("Declare at least one foot with support geometry.")
    foot_geoms: set[int] = set()
    for foot, names in spec.feet.items():
        if not isinstance(foot, str) or not foot.strip() or not names:
            raise ValueError("Every named foot needs at least one support geom.")
        for name in names:
            geom = robot_geom(name)
            if geom in foot_geoms:
                raise ValueError("Each support geom must belong to exactly one foot.")
            foot_geoms.add(geom)
    foot_bodies = {int(model.geom_bodyid[geom]) for geom in foot_geoms}
    seen_joints: set[int] = set()
    seen_actuators: set[int] = set()
    for load in spec.loads:
        joint = hinge(load.joint)
        actuator = _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, load.actuator)
        if joint in seen_joints or actuator in seen_actuators:
            raise ValueError("Load axes and actuators must have a one-to-one mapping.")
        seen_joints.add(joint)
        seen_actuators.add(actuator)
        body = int(model.jnt_bodyid[joint])
        if any(_descends_from(model, foot, body) for foot in foot_bodies):
            raise ValueError(f"Load axis {load.joint} supports a foot; ground reactions require a separate solver.")
        direct = np.flatnonzero(
            (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) & (model.actuator_trnid[:, 0] == joint)
        )
        if len(direct) != 1 or direct[0] != actuator:
            raise ValueError(f"Load axis {load.joint} needs exactly one directly mapped joint actuator.")
        if not np.array_equal(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]):
            raise ValueError(f"Actuator {load.actuator} needs unit gear for torque-cap comparison.")
        capacity = model.actuator_forcerange[actuator]
        if (
            not model.actuator_forcelimited[actuator]
            or not np.isfinite(capacity).all()
            or capacity[0] >= 0
            or capacity[1] <= 0
        ):
            raise ValueError(f"Actuator {load.actuator} needs finite positive and negative force capacity.")
        if model.jnt_actfrclimited[joint]:
            raise ValueError(f"Joint {load.joint} has an additional joint force limit; unsupported capacity mapping.")
        fraction = _finite(load.max_torque_fraction, "max_torque_fraction", positive=True)
        if fraction > 1:
            raise ValueError("max_torque_fraction must not exceed one.")
        _evidence(load.torque_evidence, "torque_evidence")
        _evidence(load.bearing_evidence, "bearing_evidence")
        if load.bearing_limit_nm is not None:
            _finite(load.bearing_limit_nm, "bearing_limit_nm", positive=True)

    seen_sweeps: set[str] = set()
    for sweep in spec.sweeps:
        if (
            not isinstance(sweep.name, str)
            or not sweep.name.strip()
            or sweep.name == "home"
            or sweep.name in seen_sweeps
        ):
            raise ValueError("Sweep names must be unique nonempty text, distinct from home.")
        seen_sweeps.add(sweep.name)
        if not sweep.samples:
            raise ValueError(f"Sweep {sweep.name} must name at least one joint.")
        for name, samples in sweep.samples.items():
            joint = hinge(name)
            if not samples:
                raise ValueError(f"Sweep {sweep.name} must sample {name} at least once.")
            for value in samples:
                angle = _finite(value, f"Sample for {name}")
                if angle < model.jnt_range[joint, 0] - 1e-12 or angle > model.jnt_range[joint, 1] + 1e-12:
                    raise ValueError(f"Sample exceeds the limits of {name}.")
    if pose_count(spec) > spec.max_poses:
        raise ValueError("The declared Cartesian sweeps exceed max_poses.")
    seen_pairs: set[tuple[int, int]] = set()
    for pair in spec.clearances:
        first, second = robot_geom(pair.first), robot_geom(pair.second)
        if any(model.geom_type[geom] not in _CLEARANCE_GEOM_TYPES for geom in (first, second)):
            raise ValueError("Clearance checks support convex primitives and mesh convex hulls only.")
        key = (min(first, second), max(first, second))
        if first == second or key in seen_pairs:
            raise ValueError("Clearance pairs must contain distinct geometry and must not repeat.")
        seen_pairs.add(key)
        _finite(pair.min_distance_m, "min_distance_m", positive=True)
