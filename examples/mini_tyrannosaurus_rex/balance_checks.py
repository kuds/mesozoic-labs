"""Deterministic, unsupported stance regressions; these do not certify walking.

The leg cap is a sensitivity assumption, not a measured continuous rating.
The optional angular-velocity kick is an explicit initial-condition perturbation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared import robot_validation
from environments.shared.robot_validation import GroundSupportProbe, require_unsupported


def stance_trial(
    model: mujoco.MjModel,
    *,
    duration: float = 30.0,
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
    require_unsupported(model, data)
    support = GroundSupportProbe(model, feet={side: (f"{side}_flat_pad_collision",) for side in ("left", "right")})

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
    late_contact: list[np.ndarray] = []
    maximum_unexpected = 0.0
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
            normals, unexpected = support.sample(data)
            maximum_unexpected = max(maximum_unexpected, unexpected)
            bad_contact = bad_contact or unexpected > 0.05
            if data.time >= duration - 1.0:
                late_forces.append(data.actuator_force.copy())
                late_contact.append(normals)
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
        "completed": bool(finite and data.time >= duration - model.opt.timestep / 2),
        "leg_cap_Nm": leg_cap,
        "angular_velocity_kick_rad_s": kick.tolist(),
        "kick_time_s": 1.0,
        "finite": finite,
        "warnings": int(np.sum(data.warning.number)),
        "no_applied_external_forces": unsupported,
        "unexpected_ground_contact": bad_contact,
        "maximum_unexpected_support_N": maximum_unexpected,
        "minimum_height_ratio": low_height,
        "maximum_tilt_deg": high_tilt,
        "maximum_root_drift_m": drift,
        "settled_leg_effort": settled,
        "settled_foot_normal_mean_N": [] if not len(loads) else np.mean(loads, axis=0).tolist(),
    }


def stance_failures(report: dict[str, Any]) -> list[str]:
    """Project acceptance bounds with margin over the reviewed stance baseline."""
    failures = []
    for key in ("finite", "completed", "no_applied_external_forces"):
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
        # The revised hold has no settled clipping at 0.10 Nm. Keep margin,
        # but reject a return of the old saturated, creeping stance.
        if effort["rms_Nm"] > 0.09 or effort["cap_fraction"] > 0.10:
            failures.append(f"settled effort: {name}")
    if len(report["settled_foot_normal_mean_N"]) != 2 or min(report["settled_foot_normal_mean_N"], default=0) < 0.5:
        failures.append("lost two-foot support")
    return failures


def main() -> int:
    """Replay both configurations and small kicks, with source-bound results."""
    from . import demo, tail_option

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("balance_validation.json"))
    args = parser.parse_args()
    root = Path(__file__).parent
    files = [root / name for name in ("model.xml", "balance_checks.py", "tail_option.py")]
    files += sorted((root / "assets").glob("*.xml"))
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    sources["../../environments/shared/robot_validation.py"] = hashlib.sha256(
        Path(robot_validation.__file__).read_bytes()
    ).hexdigest()
    trials = {}
    for label, model in (("fixed_tail", demo.load_model()[0]), ("wag_tail_centered", tail_option.load_model())):
        for kick_name, kick in (("hold", (0.0, 0.0, 0.0)), ("roll", (0.15, 0.0, 0.0)), ("pitch", (0.0, -0.15, 0.0))):
            report = stance_trial(model, duration=args.duration, angular_kick=kick)
            report["failures"] = stance_failures(report)
            trials[f"{label}_{kick_name}"] = report
    report = {
        "schema_version": 1,
        "source_path_base": "examples/mini_tyrannosaurus_rex",
        "scope": "Unsupported powered stance; optional wagging tail is centered. Not a walking or hardware certificate.",
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "source_sha256": sources,
        "trials": trials,
        "passed": all(not t["failures"] for t in trials.values()),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
