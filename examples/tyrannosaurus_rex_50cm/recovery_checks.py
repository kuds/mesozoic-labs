"""Free-base crouch-to-stand trials for medium concepts, not fall recovery.

The trial starts in a prescribed crouch with horizontal soles at the floor.
After initialization, only torque-limited joint targets change: no root resets,
world support or applied forces assist the rise. The initial hold must settle
under the model's own actuators before the measured rise can count as a pass.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from environments.shared.robot_geometry import internal_contacts
from environments.shared.robot_validation import GroundSupportProbe, require_unsupported

FEET: dict[str, tuple[str, ...]] = {side: (f"{side}_sole_collision",) for side in ("left", "right")}
LEG_JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
)
SETTLING_LIMITS = {
    "height_span_m": 0.001,
    "maximum_root_linear_speed_m_s": 0.01,
    "maximum_root_angular_speed_rad_s": 0.05,
    "maximum_leg_speed_rad_s": 0.1,
    "simultaneous_foot_support_fraction": 0.95,
    "minimum_foot_weight_fraction": 0.1,
}


def _require_position_drives(model: mujoco.MjModel) -> None:
    """Only direct unit-gear position drives make radians and Nm unambiguous."""
    mapped: set[int] = set()
    for actuator in range(model.nu):
        name = model.actuator(actuator).name
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(f"Actuator {name} requires a direct joint transmission.")
        if not np.array_equal(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]):
            raise ValueError(f"Actuator {name} requires unit gear for joint-radian and Nm evidence.")
        joint = int(model.actuator_trnid[actuator, 0])
        if joint in mapped or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("Position drives need exactly one actuator per hinge joint.")
        mapped.add(joint)
        if name != model.joint(joint).name + "_servo":
            raise ValueError(f"Actuator {name} does not match its named joint.")
        if model.jnt_actfrclimited[joint]:
            raise ValueError(f"Joint {model.joint(joint).name} has an unsupported additional force limit.")
        if (
            not model.jnt_limited[joint]
            or not np.isfinite(model.jnt_range[joint]).all()
            or model.jnt_range[joint, 0] >= model.jnt_range[joint, 1]
            or not model.actuator_ctrllimited[actuator]
            or not np.isfinite(model.actuator_ctrlrange[actuator]).all()
            or model.actuator_ctrlrange[actuator, 0] >= model.actuator_ctrlrange[actuator, 1]
        ):
            raise ValueError(f"Actuator {name} needs finite, ordered joint-angle and control limits.")
        gain, bias = model.actuator_gainprm[actuator], model.actuator_biasprm[actuator]
        if (
            model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
            or not np.isfinite(gain).all()
            or not np.isfinite(bias).all()
            or gain[0] <= 0
            or bias[0] != 0
            or bias[1] != -gain[0]
            or bias[2] > 0
        ):
            raise ValueError(f"Actuator {name} needs a direct position drive with nonnegative PD gains.")
    if mapped != set(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError("Every hinge must have exactly one direct position actuator.")


def _settling_window(
    heights: list[float],
    root_velocities: list[np.ndarray],
    leg_velocities: list[np.ndarray],
    loads: list[np.ndarray],
    weight: float,
) -> dict[str, Any]:
    """Reject moving or alternating-support windows even if their means match."""
    if not heights:
        return {"samples": 0, "settled": False}
    root_speed = np.asarray(root_velocities)
    foot_loads = np.asarray(loads)
    metrics = {
        "samples": len(heights),
        "height_span_m": float(np.ptp(heights)),
        "maximum_root_linear_speed_m_s": float(np.linalg.norm(root_speed[:, :3], axis=1).max()),
        "maximum_root_angular_speed_rad_s": float(np.linalg.norm(root_speed[:, 3:], axis=1).max()),
        "maximum_leg_speed_rad_s": float(np.abs(leg_velocities).max()),
        "simultaneous_foot_support_fraction": float(
            np.mean(np.all(foot_loads > SETTLING_LIMITS["minimum_foot_weight_fraction"] * weight, axis=1))
        ),
    }
    metrics["settled"] = bool(
        all(
            metrics[key] <= SETTLING_LIMITS[key]
            for key in (
                "height_span_m",
                "maximum_root_linear_speed_m_s",
                "maximum_root_angular_speed_rad_s",
                "maximum_leg_speed_rad_s",
            )
        )
        and metrics["simultaneous_foot_support_fraction"] >= SETTLING_LIMITS["simultaneous_foot_support_fraction"]
    )
    return metrics


def _sole_floor_shift(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Initial setup only: seat both level box soles on the horizontal floor."""
    floor = model.geom("floor").id
    if model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("Crouch initialization requires the declared plane floor.")
    bottoms = []
    for names in FEET.values():
        geom = model.geom(names[0]).id
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Crouch initialization requires box sole geometry.")
        up = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        if not np.allclose(up, [0, 0, 1], atol=1e-10, rtol=0):
            raise ValueError("The chosen symmetric crouch must keep both soles horizontal.")
        bottoms.append(float(data.geom_xpos[geom, 2] - model.geom_size[geom, 2]))
    if max(bottoms) - min(bottoms) > 1e-9:
        raise ValueError("Both soles must be at the same height before floor placement.")
    return float(data.geom_xpos[floor, 2] - np.mean(bottoms))


def prepare_crouch(model: mujoco.MjModel, *, knee_angle: float = 0.6) -> mujoco.MjData:
    """Return private trial data; never modify the supplied model or its limits."""
    if not math.isfinite(knee_angle) or knee_angle <= 0:
        raise ValueError("knee_angle must be finite and positive.")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    require_unsupported(model, data)
    _require_position_drives(model)
    if np.any(data.eq_active):
        raise ValueError("Crouch trials require uncoupled joints without active equality constraints.")
    GroundSupportProbe(model, feet=FEET)
    for side in FEET:
        for suffix, offset in (("hip_pitch", -knee_angle / 2), ("knee", knee_angle), ("ankle_pitch", -knee_angle / 2)):
            name = f"{side}_{suffix}"
            joint, actuator = model.joint(name).id, model.actuator(name + "_servo").id
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Crouch joint {name} must be a hinge.")
            address = int(model.jnt_qposadr[joint])
            value = float(data.qpos[address] + offset)
            if not model.jnt_limited[joint] or not model.jnt_range[joint, 0] <= value <= model.jnt_range[joint, 1]:
                raise ValueError(f"Crouch angle exceeds the range of {name}.")
            if not model.actuator_ctrlrange[actuator, 0] <= value <= model.actuator_ctrlrange[actuator, 1]:
                raise ValueError(f"Crouch target exceeds the control range of {name}.")
            data.qpos[address] = value
            data.ctrl[actuator] = value
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    root = int(model.joint("root").qposadr[0])
    data.qpos[root + 2] += _sole_floor_shift(model, data)
    mujoco.mj_forward(model, data)
    return data


def run_crouch_trial(
    model: mujoco.MjModel,
    *,
    knee_angle: float = 0.6,
    duration: float = 8.0,
    rise: bool = True,
) -> dict[str, Any]:
    """Measure held-crouch to standing motion, including failed trials.

    ``rise=False`` holds the same crouch throughout as a negative control. A
    pass needs a real height increase and settled two-foot support; merely
    completing a simulation or remaining upright cannot count as a rise.
    """
    if not math.isfinite(duration) or duration < 6:
        raise ValueError("duration must be finite and at least six seconds.")
    data = prepare_crouch(model, knee_angle=knee_angle)
    root = int(model.joint("root").qposadr[0])
    torso = model.body("torso").id
    start_xy = data.qpos[root : root + 2].copy()
    initial_height = float(data.qpos[root + 2])
    home = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, home, model.key("home").id)
    mujoco.mj_forward(model, home)
    home_height = float(home.qpos[root + 2] + _sole_floor_shift(model, home))
    requested_depth = home_height - initial_height
    if requested_depth <= 0:
        raise ValueError("The selected joint pattern does not lower the root below its home height.")
    crouch_controls, home_controls = data.ctrl.copy(), home.ctrl.copy()
    hold_end, rise_end = duration * 0.25, duration * 0.75
    support = GroundSupportProbe(model, feet=FEET)
    joint_ids = np.array([model.joint(name).id for name in LEG_JOINTS])
    leg_velocity_addresses = model.jnt_dofadr[joint_ids]
    root_velocity_address = int(model.joint("root").dofadr[0])
    actuator_ids = np.array([model.actuator(name + "_servo").id for name in LEG_JOINTS])
    hinge_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    qpos_addresses = model.jnt_qposadr[hinge_ids]
    caps = model.actuator_forcerange
    peak = np.zeros(model.nu)
    squares = np.zeros(model.nu)
    clipped_steps = np.zeros(model.nu, dtype=int)
    held_heights: list[float] = []
    held_loads: list[np.ndarray] = []
    held_root_velocities: list[np.ndarray] = []
    held_leg_velocities: list[np.ndarray] = []
    final_heights: list[float] = []
    final_loads: list[np.ndarray] = []
    final_root_velocities: list[np.ndarray] = []
    final_leg_velocities: list[np.ndarray] = []
    final_efforts: list[np.ndarray] = []
    minimum_height = initial_height
    max_drift = max_tilt = max_internal = max_unexpected = max_overshoot = 0.0
    no_external_forces, finite = True, True
    completed_steps = 0
    steps = math.ceil(duration / model.opt.timestep)
    sample_every = max(1, steps // 24)
    pose_samples = [{"time_s": 0.0, "qpos": data.qpos.tolist()}]
    for _ in range(steps):
        phase = np.clip((data.time - hold_end) / (rise_end - hold_end), 0, 1) if rise else 0
        blend = 0.5 - 0.5 * math.cos(math.pi * float(phase))
        data.ctrl[:] = (1 - blend) * crouch_controls + blend * home_controls
        mujoco.mj_step(model, data)
        completed_steps += 1
        finite = all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.actuator_force))
        if not finite or np.any(data.warning.number):
            break
        if completed_steps % sample_every == 0:
            pose_samples.append({"time_s": float(data.time), "qpos": data.qpos.tolist()})
        height = float(data.qpos[root + 2])
        loads, unexpected = support.sample(data)
        no_external_forces &= not bool(np.any(data.xfrc_applied) or np.any(data.qfrc_applied))
        minimum_height = min(minimum_height, height)
        max_drift = max(max_drift, float(np.linalg.norm(data.qpos[root : root + 2] - start_xy)))
        up = data.xmat[torso].reshape(3, 3)[:, 2]
        max_tilt = max(max_tilt, math.degrees(math.acos(float(np.clip(up[2], -1, 1)))))
        max_unexpected = max(max_unexpected, unexpected)
        max_internal = max(max_internal, max((c["penetration_m"] for c in internal_contacts(model, data)), default=0))
        angles = data.qpos[qpos_addresses]
        max_overshoot = max(
            max_overshoot,
            float(np.max(np.maximum(model.jnt_range[hinge_ids, 0] - angles, angles - model.jnt_range[hinge_ids, 1]))),
        )
        efforts = data.actuator_force
        peak = np.maximum(peak, np.abs(efforts))
        squares += efforts**2
        signed_caps = np.where(efforts >= 0, caps[:, 1], -caps[:, 0])
        clipped_steps += np.abs(efforts) >= 0.99 * signed_caps
        if hold_end - 0.5 <= data.time < hold_end:
            held_heights.append(height)
            held_loads.append(loads)
            held_root_velocities.append(data.qvel[root_velocity_address : root_velocity_address + 6].copy())
            held_leg_velocities.append(data.qvel[leg_velocity_addresses].copy())
        if data.time >= duration - 1:
            final_heights.append(height)
            final_loads.append(loads)
            final_efforts.append(efforts.copy())
            final_root_velocities.append(data.qvel[root_velocity_address : root_velocity_address + 6].copy())
            final_leg_velocities.append(data.qvel[leg_velocity_addresses].copy())
    held_height = float(np.mean(held_heights)) if held_heights else None
    final_height = float(np.mean(final_heights)) if final_heights else None
    measured_rise = final_height - held_height if final_height is not None and held_height is not None else None
    held_normal = np.mean(held_loads, axis=0) if held_loads else np.zeros(2)
    final_normal = np.mean(final_loads, axis=0) if final_loads else np.zeros(2)
    final_rms = np.sqrt(np.mean(np.asarray(final_efforts) ** 2, axis=0)) if final_efforts else np.zeros(model.nu)
    weight = float(model.body_subtreemass[torso] * 9.81)
    held_window = _settling_window(held_heights, held_root_velocities, held_leg_velocities, held_loads, weight)
    settled_window = _settling_window(final_heights, final_root_velocities, final_leg_velocities, final_loads, weight)
    minimum_rise = max(0.005, 0.70 * requested_depth)
    bounded = np.minimum(-caps[:, 0], caps[:, 1])
    checks = {
        "completed": bool(finite and data.time >= duration - model.opt.timestep / 2),
        "no_warnings": not bool(np.any(data.warning.number)),
        "no_external_assistance": bool(no_external_forces and not np.any(data.eq_active)),
        "initial_crouch_held": bool(
            held_height is not None
            and abs(held_height - initial_height) < 0.01
            and home_height - held_height > 0.7 * requested_depth
            and min(held_normal) > 0.1 * weight
            and 0.9 * weight < sum(held_normal) < 1.1 * weight
        ),
        "held_window_settled": held_window["settled"],
        "measured_rise": bool(measured_rise is not None and measured_rise >= minimum_rise),
        "standing_height_reached": bool(final_height is not None and final_height > 0.95 * home_height),
        "no_collapse": minimum_height > 0.85 * initial_height,
        "bounded_tilt": max_tilt < 10,
        "bounded_drift": max_drift < 0.03,
        "sole_only_support": max_unexpected < 0.05,
        "no_internal_contact": max_internal < 0.001,
        "joint_limits": max_overshoot < 0.03,
        "settled_support": bool(
            final_loads and min(final_normal) > 0.1 * weight and 0.9 * weight < sum(final_normal) < 1.1 * weight
        ),
        "settled_leg_effort": bool(final_efforts and np.all(final_rms[actuator_ids] < 0.7 * bounded[actuator_ids])),
        "final_window_settled": settled_window["settled"],
    }
    if finite and pose_samples[-1]["time_s"] != float(data.time):
        pose_samples.append({"time_s": float(data.time), "qpos": data.qpos.tolist()})
    return {
        "scope": "Free-base rise from a prescribed symmetric crouch; not recovery after a fall.",
        "passed": all(checks.values()),
        "checks": checks,
        "requested_duration_s": duration,
        "duration_s": float(data.time),
        "finite": bool(finite),
        "knee_offset_rad": knee_angle,
        "rise_commanded": rise,
        "hold_until_s": hold_end,
        "rise_until_s": rise_end,
        "initial_root_height_m": initial_height,
        "commanded_crouch_height_m": initial_height,
        "home_root_height_m": home_height,
        "requested_crouch_depth_m": requested_depth,
        "held_root_height_mean_m": held_height,
        "final_root_height_mean_m": final_height,
        "measured_rise_m": measured_rise,
        "minimum_required_rise_m": minimum_rise,
        "minimum_root_height_m": minimum_height,
        "maximum_root_drift_m": max_drift,
        "maximum_torso_tilt_deg": max_tilt,
        "maximum_unexpected_support_N": max_unexpected,
        "maximum_internal_penetration_m": max_internal,
        "maximum_joint_limit_overshoot_rad": max_overshoot,
        "held_foot_normal_mean_N": held_normal.tolist(),
        "settled_foot_normal_mean_N": final_normal.tolist(),
        "settled_leg_rms_torque_nm": final_rms[actuator_ids].tolist(),
        "held_window": held_window,
        "settled_window": settled_window,
        "settling_limits": dict(SETTLING_LIMITS),
        "peak_torque_nm": peak.tolist(),
        "rms_torque_nm": np.sqrt(squares / completed_steps).tolist(),
        "torque_clipping_duration_s": (clipped_steps * model.opt.timestep).tolist(),
        "actuator_names": [model.actuator(index).name for index in range(model.nu)],
        "actuator_limits_nm": caps.tolist(),
        "final_leg_joint_angles_rad": data.qpos[model.jnt_qposadr[joint_ids]].tolist() if finite else None,
        "warning_counts": data.warning.number.tolist(),
        "pose_samples": pose_samples,
        "limitations": [
            "Only the prescribed crouch is initialized; entering it from standing or a fall is not tested.",
            "Ideal position drives omit measured torque-speed, thermal, electrical and transmission behavior.",
            "Ground contact uses simulation proxies and assigned masses, not qualified physical hardware.",
        ],
    }
