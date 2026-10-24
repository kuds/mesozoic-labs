"""Deterministic, unsupported stance regressions; these do not certify walking.

The leg cap is a sensitivity assumption, not a measured continuous rating.
The optional angular-velocity kick is an explicit initial-condition perturbation.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np


def stance_trial(
    model: mujoco.MjModel,
    *,
    duration: float = 6.0,
    leg_cap: float = 0.10,
    angular_kick: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Hold home, applying a root-frame angular velocity increment at one second.

    No support is enabled or silently removed. Any active world weld is an error.
    Force caps are restored so calling this probe cannot change a later rollout.
    """
    if not math.isfinite(duration) or duration < 3 or not math.isfinite(leg_cap) or leg_cap <= 0:
        raise ValueError("Use a finite duration >= 3 s and a positive finite leg cap.")
    kick = np.asarray(angular_kick, dtype=float)
    if kick.shape != (3,) or not np.isfinite(kick).all():
        raise ValueError("angular_kick must contain three finite angular velocities.")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    root = model.joint("root").id
    if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("The stance regression requires a free root joint.")
    for eq in range(model.neq):
        if data.eq_active[eq] and model.eq_type[eq] == mujoco.mjtEq.mjEQ_WELD:
            if 0 in (model.eq_obj1id[eq], model.eq_obj2id[eq]):
                raise ValueError("An active world weld would invalidate unsupported stance.")
    if not np.allclose(model.opt.gravity, [0, 0, -9.81], atol=1e-12):
        raise ValueError("The stance regression requires Earth gravity along -Z.")

    legs = [i for i in range(model.nu) if any(p in model.actuator(i).name for p in ("hip_", "knee", "ankle"))]
    if not legs or not model.actuator_forcelimited[legs].all():
        raise ValueError("Every leg actuator must have a force limit.")
    saved = model.actuator_forcerange.copy()
    actual_caps = np.minimum(leg_cap, np.minimum(-saved[legs, 0], saved[legs, 1]))
    if np.any(actual_caps <= 0):
        raise ValueError("Every leg actuator needs positive and negative torque capacity.")
    torso = model.body("torso").id
    root_q = model.jnt_qposadr[root]
    root_v = model.jnt_dofadr[root]
    initial_height = float(data.qpos[root_q + 2])
    initial_xy = data.qpos[root_q : root_q + 2].copy()
    low_height, high_tilt, drift = 1.0, 0.0, 0.0
    unsupported, bad_contact, kicked = True, False, False
    finite = True
    late_forces: list[np.ndarray] = []
    late_contact: list[tuple[float, float]] = []
    allowed = {f"{side}_flat_pad_collision": i for i, side in enumerate(("left", "right"))}
    try:
        # A sensitivity probe may lower the existing limit, never increase it.
        model.actuator_forcerange[legs, 0] = -actual_caps
        model.actuator_forcerange[legs, 1] = actual_caps
        for _ in range(math.ceil(duration / model.opt.timestep)):
            if not kicked and data.time >= 1.0 - model.opt.timestep / 2:
                data.qvel[root_v + 3 : root_v + 6] += kick
                kicked = True
            mujoco.mj_step(model, data)
            finite = finite and all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.actuator_force))
            if not finite or np.any(data.warning.number):
                break
            up = data.xmat[torso].reshape(3, 3)[:, 2]
            high_tilt = max(high_tilt, math.degrees(math.acos(float(np.clip(up[2], -1, 1)))))
            low_height = min(low_height, float(data.qpos[root_q + 2] / initial_height))
            drift = max(drift, float(np.linalg.norm(data.qpos[root_q : root_q + 2] - initial_xy)))
            unsupported = unsupported and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            normals = [0.0, 0.0]
            for index, contact in enumerate(data.contact[: data.ncon]):
                names = [model.geom(int(g)).name for g in contact.geom]
                if "floor" not in names:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                other = names[1] if names[0] == "floor" else names[0]
                if other in allowed:
                    normals[allowed[other]] += max(0.0, float(force[0]))
                elif force[0] > 0.05:
                    bad_contact = True
            if data.time >= duration - 1.0:
                late_forces.append(data.actuator_force.copy())
                late_contact.append((normals[0], normals[1]))
    finally:
        model.actuator_forcerange[:] = saved

    efforts = np.array(late_forces)
    loads = np.array(late_contact)
    settled = {}
    if len(efforts):
        for index, i in enumerate(legs):
            settled[model.actuator(i).name] = {
                "rms_Nm": float(np.sqrt(np.mean(efforts[:, i] ** 2))),
                "cap_fraction": float(np.mean(np.abs(efforts[:, i]) >= 0.99 * actual_caps[index])),
                "applied_cap_Nm": float(actual_caps[index]),
            }
    return {
        "scope": "Unsupported stance and explicit small disturbance only; not walking or thermal qualification.",
        "duration_s": float(data.time),
        "leg_cap_Nm": leg_cap,
        "angular_velocity_kick_rad_s": kick.tolist(),
        "kick_time_s": 1.0,
        "finite": finite,
        "warnings": int(np.sum(data.warning.number)),
        "no_applied_external_forces": unsupported,
        "unexpected_ground_contact": bad_contact,
        "minimum_height_ratio": low_height,
        "maximum_tilt_deg": high_tilt,
        "maximum_root_drift_m": drift,
        "settled_leg_effort": settled,
        "settled_foot_normal_mean_N": [] if not len(loads) else np.mean(loads, axis=0).tolist(),
    }


def stance_failures(report: dict[str, Any]) -> list[str]:
    """Project acceptance bounds with margin over the reviewed stance baseline."""
    failures = []
    for key in ("finite", "no_applied_external_forces"):
        if not report[key]:
            failures.append(key)
    if report["warnings"]:
        failures.append("simulation warnings")
    if report["unexpected_ground_contact"]:
        failures.append("non-foot ground support")
    if report["minimum_height_ratio"] < 0.90:
        failures.append("height loss")
    if report["maximum_tilt_deg"] > 10:
        failures.append("excessive tilt")
    if report["maximum_root_drift_m"] > 0.02:
        failures.append("stance drift")
    if not report["settled_leg_effort"]:
        failures.append("missing settled effort")
    for name, effort in report["settled_leg_effort"].items():
        # The baseline already has substantial cap contact (~44% at knees).
        # This is a regression ceiling, not a desirable physical duty target.
        if effort["rms_Nm"] > 0.09 or effort["cap_fraction"] > 0.60:
            failures.append(f"settled effort: {name}")
    if len(report["settled_foot_normal_mean_N"]) != 2 or min(report["settled_foot_normal_mean_N"], default=0) < 0.5:
        failures.append("lost two-foot support")
    return failures
