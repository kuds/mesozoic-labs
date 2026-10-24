"""Reproduce optional-tail space, assigned load, and gentle-wag measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib import import_module
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

balance_checks = import_module(f"{__package__}.balance_checks" if __package__ else "balance_checks")
demo = import_module(f"{__package__}.demo" if __package__ else "demo")
engineering = import_module(f"{__package__}.engineering" if __package__ else "engineering")
tail_option = import_module(f"{__package__}.tail_option" if __package__ else "tail_option")


def packaging_clearance(model: mujoco.MjModel, points: int = 101) -> dict[str, Any]:
    """Check moving tail mesh versus stationary component service boxes.

    Positive world-axis AABB gaps prove disjoint volumes, but are not exact
    Euclidean clearances. Parent/child collision exclusions cannot hide this
    check. It does not assert the rest of the electronics pack simultaneously.
    """
    if points < 2:
        raise ValueError("Use at least two tail positions.")
    data = engineering.home_data(model)
    # Read the actual ledger so larger future components cannot bypass this gate.
    ledger = json.loads(Path(__file__).with_name("components.json").read_text())
    components = {row["id"]: row for row in ledger["components"]}
    orientations = {
        "computer": ("computer_part_proxy", [2, 0, 1]),
        "servo_interface": ("u2d2_part_proxy", [2, 0, 1]),
        "motor_regulator": ("motor_regulator_part_proxy", [0, 1, 2]),
        "logic_regulator": ("logic_regulator_part_proxy", [0, 1, 2]),
        "battery": ("schematic_battery_proxy", [0, 1, 2]),
    }
    sizes = {
        geom: np.array(components[component]["reserved_envelope_mm"])[axes] / 2000
        for component, (geom, axes) in orientations.items()
    }
    tail_geom = model.geom("tail_shell").id
    mesh = model.geom_dataid[tail_geom]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    motor = model.geom("tail_yaw_servo_case").id
    limits = model.jnt_range[model.joint("tail_yaw").id]
    minima = {name: {"moving_tail_mesh_gap_m": math.inf, "motor_box_gap_m": math.inf} for name in sizes}
    for angle in np.linspace(float(limits[0]), float(limits[1]), points):
        engineering.set_pose(model, data, {"tail_yaw": float(angle)})
        cloud = vertices @ data.geom_xmat[tail_geom].reshape(3, 3).T + data.geom_xpos[tail_geom]
        low, high = cloud.min(axis=0), cloud.max(axis=0)
        for name, local_half_size in sizes.items():
            geom = model.geom(name).id
            center = data.geom_xpos[geom]
            half_size = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ local_half_size
            gap = float(np.max(np.maximum(center - half_size - high, low - center - half_size)))
            motor_half = np.abs(data.geom_xmat[motor].reshape(3, 3)) @ model.geom_size[motor]
            motor_gap = float(np.max(np.abs(data.geom_xpos[motor] - center) - motor_half - half_size))
            minima[name]["moving_tail_mesh_gap_m"] = min(minima[name]["moving_tail_mesh_gap_m"], gap)
            minima[name]["motor_box_gap_m"] = min(minima[name]["motor_box_gap_m"], motor_gap)
    return {
        "poses": points,
        "tail_limits_rad": limits.tolist(),
        "reserved_dimensions_xyz_m": {name: (2 * size).tolist() for name, size in sizes.items()},
        "minimum_axis_gaps": minima,
        "scope": "Tail mesh and motor versus selected service boxes only; not simultaneous CAD packaging, joint/collar, cable or mounting qualification.",
    }


def wag_trial(model: mujoco.MjModel, *, duration: float = 10.0, supported: bool = False) -> dict[str, Any]:
    """Only the tail moves; root weld is explicitly recorded, never implied walking."""
    if not math.isfinite(duration) or duration < 3:
        raise ValueError("Use a finite duration of at least three seconds.")
    data = engineering.home_data(model)
    demo.set_support(model, data, supported)
    actuator = model.actuator("tail_yaw_servo").id
    address = model.jnt_qposadr[model.joint("tail_yaw").id]
    torso = model.body("torso").id
    peak, squared, error_squared, tilt = 0.0, 0.0, 0.0, 0.0
    low, high = math.inf, -math.inf
    initial_height = float(data.qpos[2])
    minimum_height = 1.0
    steps = math.ceil(duration / model.opt.timestep)
    for _ in range(steps):
        ramp = 0.5 - 0.5 * math.cos(math.pi * min(float(data.time), 1.0))
        target = ramp * math.radians(15) * math.sin(math.pi * data.time)
        data.ctrl[actuator] = target
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
            raise RuntimeError("Tail rollout produced non-finite values or a simulator warning.")
        torque = float(data.actuator_force[actuator])
        angle = float(data.qpos[address])
        peak, squared = max(peak, abs(torque)), squared + torque**2
        error_squared += (angle - target) ** 2
        low, high = min(low, angle), max(high, angle)
        up = data.xmat[torso].reshape(3, 3)[:, 2]
        tilt = max(tilt, math.degrees(math.acos(float(np.clip(up[2], -1, 1)))))
        minimum_height = min(minimum_height, float(data.qpos[2] / initial_height))
    return {
        "duration_s": float(data.time),
        "externally_supported": supported,
        "command_amplitude_deg": 15,
        "command_frequency_hz": 0.5,
        "actual_tail_range_rad": [low, high],
        "tail_peak_torque_Nm": peak,
        "tail_rms_torque_Nm": math.sqrt(squared / steps),
        "tracking_rms_error_rad": math.sqrt(error_squared / steps),
        "maximum_torso_tilt_deg": tilt,
        "minimum_height_ratio": minimum_height,
        "scope": "Ideal position-servo character motion; no electrical, thermal or walking validation.",
    }


def build_report(samples: int = 1000) -> dict[str, Any]:
    model = tail_option.load_model()
    root = Path(__file__).parent
    files = [
        "model.xml",
        "components.json",
        "tail_option.py",
        "tail_study.py",
        "balance_checks.py",
        "demo.py",
        "engineering.py",
    ]
    files += [str(path.relative_to(root)) for path in sorted((root / "assets").glob("*.xml"))]
    report: dict[str, Any] = {
        "configuration": "optional_tail_yaw",
        "mujoco_version": mujoco.__version__,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
        "mass_kg": float(model.body_mass.sum()),
        "actuators": model.nu,
        "motor_clearance": engineering.motor_clearance_sweep(model, samples=samples),
        "combined_collision_proxies": engineering.combined_collision_sweep(model, samples=samples),
        "upper_body_collision_proxies": engineering.upper_body_collision_sweep(model, points=3),
        "tail_component_clearance": packaging_clearance(model),
        "home_static_equilibrium": engineering.static_equilibrium(model),
        "unsupported_wag": wag_trial(model),
        "supported_wag": wag_trial(model, supported=True),
        "stance_regressions": [],
        "walking_validated": False,
        "fabrication_fit_validated": False,
        "hardware_power_and_thermal_validated": False,
    }
    for kick in ((0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, -0.15, 0.0)):
        row = balance_checks.stance_trial(model, angular_kick=kick)
        row["regression_failures"] = balance_checks.stance_failures(row)
        report["stance_regressions"].append(row)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("tail_validation.json"))
    args = parser.parse_args()
    report = build_report(args.samples)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote optional-tail study to {args.output}; walking and physical fit remain unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
