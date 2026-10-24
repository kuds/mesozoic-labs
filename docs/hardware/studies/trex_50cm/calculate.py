"""Reproduce the 50/80 cm feasibility arithmetic; this is not a gait simulation.

Run from any directory: python3 path/to/trex_50cm/calculate.py
Uses only the Python standard library and the checked-in miniature gait report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


def dollars(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "inputs.json")
    parser.add_argument("--output", type=Path, default=here / "results.json")
    args = parser.parse_args()
    inputs_path = args.input.resolve()
    baseline_path = root / "examples/mini_tyrannosaurus_rex/walking_validation.json"
    inputs = json.loads(inputs_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    model = next(m for m in baseline["models"] if m["name"] == "fixed_tail_base")
    trial = next(t for t in model["trials"] if t["requested_duration_s"] == 20 and t["cap"] == 0.2)
    if trial["terminated_early"] or trial["display_support_active"]:
        raise ValueError("The reference trial changed; reassess the extrapolation.")
    if inputs["status"] != "preliminary_sizing_not_validated_robot":
        raise ValueError("This calculator cannot certify a robot design.")
    scale = inputs["target_height_m"] / inputs["baseline_height_m"]
    mass_allowance = sum(inputs["shared_mass_allowances_g"].values()) / 1000
    low = sum(v[0] for v in inputs["non_actuator_cost_allowances_usd"].values())
    high = sum(v[1] for v in inputs["non_actuator_cost_allowances_usd"].values())
    architectures = {}
    for name, counts in inputs["architectures"].items():
        motor_mass = sum(inputs["actuators"][k]["mass_g"] * n for k, n in counts.items()) / 1000
        motor_cost = sum((Decimal(str(inputs["actuators"][k]["price_usd"])) * n for k, n in counts.items()), Decimal(0))
        mass = motor_mass + mass_allowance
        # For geometrically similar mass distribution and time scaled by sqrt(s),
        # both gravity and inertial torque scale by mass_ratio * length_ratio.
        # Real replacement motors do NOT preserve that distribution.
        torque_factor = mass / model["mass_kg"] * scale
        joints = {}
        for index, joint in enumerate(model["actuator_names"][:10]):
            joints[joint] = {
                "reference_rms_nm": trial["rms_torque_Nm"][index],
                "similarity_rms_nm": trial["rms_torque_Nm"][index] * torque_factor,
                "similarity_applied_peak_nm": trial["peak_torque_Nm"][index] * torque_factor,
                "similarity_peak_speed_rpm": trial["peak_joint_speed_rad_s"][index]
                / math.sqrt(scale)
                * 60
                / (2 * math.pi),
            }
        architectures[name] = {
            "actuator_count": sum(counts.values()),
            "actuator_mass_kg": motor_mass,
            "total_mass_budget_including_reserve_kg": mass,
            "actuator_cost_usd": dollars(motor_cost),
            "project_cost_before_contingency_usd": [dollars(motor_cost + low), dollars(motor_cost + high)],
            "project_cost_with_contingency_usd": [
                dollars((motor_cost + x) * (1 + Decimal(str(inputs["cost_contingency_fraction"])))) for x in (low, high)
            ],
            "similarity_torque_factor": torque_factor,
            "similarity_joint_screen": joints,
            "gravity_torque_at_offsets_nm": {str(mm): mass * 9.81 * mm / 1000 for mm in (20, 30, 50, 80)},
        }
    battery = inputs["battery_scenario"]
    energy = battery["nominal_v"] * battery["capacity_ah"]
    tail = inputs["tail_scenario"]
    tail_inertia = tail["mass_kg"] * tail["length_m"] ** 2 / 3
    tail_accel = math.radians(tail["amplitude_deg"]) * (2 * math.pi * tail["frequency_hz"]) ** 2
    tail_gravity = tail["mass_kg"] * 9.81 * tail["length_m"] / 2 * math.sin(math.radians(tail["body_lean_deg"]))
    can = inputs["can_scenario"]
    result = {
        "status": "analytical_screen_only_no_new_CAD_or_dynamic_simulation",
        "limitations": [
            "Reference gait eventually falls and has ideal torque-limited motors; it is not a validated gait.",
            "Similarity values assume proportional mass distribution and slower cadence; actual actuator placement changes both.",
            "Reported applied peaks are clipped by the reference 0.20 Nm cap, not an unconstrained requirement or design maximum.",
            "RMS and speed maxima need not occur simultaneously; no full torque-speed-duty trace is available here.",
            "New ankle-roll joints have no corresponding reference loads and require fresh analysis.",
            "Runtime, mass and non-motor costs are scenarios, not measured hardware results or vendor quotes.",
        ],
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (inputs_path, baseline_path, Path(__file__).resolve())
        },
        "scale_factor": scale,
        "uniform_constant_density_mass_kg_from_825g": 0.825 * scale**3,
        "similarity_period_s": trial["params"][0] * math.sqrt(scale),
        "architectures": architectures,
        "single_support_sweep": [
            {"mass_kg": m, "offset_mm": d, "gravity_torque_nm": m * 9.81 * d / 1000}
            for m in (4, 5, 6, 7, 10, 12, 14)
            for d in (20, 30, 50, 80)
        ],
        "battery": {
            "nominal_wh": energy,
            "usable_wh_assumption": energy * battery["usable_energy_fraction"],
            "runtime_minutes_by_average_battery_w": {
                str(p): 60 * energy * battery["usable_energy_fraction"] / p for p in battery["average_battery_power_w"]
            },
        },
        "tail": {
            "uniform_rod_inertia_kgm2": tail_inertia,
            "inertial_peak_nm": tail_inertia * tail_accel,
            "gravity_upper_bound_nm_at_lean": tail_gravity,
            "sum_with_friction_allowance_nm": tail_inertia * tail_accel + tail_gravity + tail["friction_allowance_nm"],
            "root_weight_bending_nm": tail["mass_kg"] * 9.81 * tail["length_m"] / 2,
        },
        "arm_static_horizontal_nm_each": inputs["arm_scenario"]["mass_kg_each"]
        * 9.81
        * inputs["arm_scenario"]["com_radius_m"],
        "head_static_nm": inputs["head_scenario"]["moving_mass_kg"] * 9.81 * inputs["head_scenario"]["com_offset_m"],
        "can_utilization_per_bus": can["bits_per_frame_budget"]
        * can["frames_per_motor_update"]
        * can["motors_per_bus"]
        * can["rate_hz"]
        / can["bitrate"],
    }
    # Cross-check the moment calculation with a unit conversion, not just output snapshots.
    if not math.isclose(6 * 9.81 * 30 / 1000, 1.7658):
        raise ArithmeticError("Moment units are inconsistent.")
    if not all(v["actuator_count"] == 18 for v in architectures.values()):
        raise ValueError("Expected twelve leg axes and six character axes.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "similarity_joint_screen"} for k, v in architectures.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
