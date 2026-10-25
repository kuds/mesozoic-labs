"""Reproduce the size-study cost table; these are prototype planning allowances.

Run the 50 cm and 80 cm calculators first. No hardware qualification is implied.
Uses only the standard library. All amounts are USD, excluding labor and landed costs.
"""

from __future__ import annotations

import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TypedDict


class MotorPrice(TypedDict):
    count: int
    unit_price_usd: float
    source: str


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    paths = [Path(__file__).resolve()]
    tiers = {}
    for folder in ("trex_50cm", "trex_80cm"):
        inputs_path = here / folder / "inputs.json"
        results_path = here / folder / "results.json"
        paths.extend((inputs_path, results_path))
        inputs = json.loads(inputs_path.read_text())
        results = json.loads(results_path.read_text())
        for source_path, recorded in results["source_sha256"].items():
            if recorded != hashlib.sha256((root / source_path).read_bytes()).hexdigest():
                raise ValueError(f"Regenerate stale sizing results for {folder}: {source_path}.")
        development_keys = {
            "spares_and_rework",
            "bench_supply_fixture_and_basic_tools",
            "cloud_training_and_experiments",
        }
        development = [
            sum(v[i] for k, v in inputs["non_actuator_cost_allowances_usd"].items() if k in development_keys)
            for i in (0, 1)
        ]
        for name, architecture in results["architectures"].items():
            subtotal = architecture["project_cost_before_contingency_usd"]
            tiers[f"{folder}/{name}"] = {
                "height_m": inputs["target_height_m"],
                "mass_budget_including_reserve_kg": architecture["total_mass_budget_including_reserve_kg"],
                "actuator_cost_usd": architecture["actuator_cost_usd"],
                "core_parts_and_fabrication_usd": [round(subtotal[i] - development[i], 2) for i in (0, 1)],
                "development_extras_usd": development,
                "contingency_fraction": inputs["cost_contingency_fraction"],
                "first_project_usd": architecture["project_cost_with_contingency_usd"],
            }

    # A provisional upgrade allowance, NOT a qualified 15-axis miniature BOM.
    # Six loaded leg axes are costed as XC330; the remaining nine as XL330.
    mini_motors: dict[str, MotorPrice] = {
        "XC330_M288_T": {
            "count": 6,
            "unit_price_usd": 103.39,
            "source": "https://robotis.us/products/dynamixel-xc330-m288-t",
        },
        "XL330_M288_T": {
            "count": 9,
            "unit_price_usd": 27.49,
            "source": "https://robotis.us/products/dynamixel-xl330-m288-t",
        },
    }
    mini_core = {
        "structure_bearings_feet": [120, 250],
        "shells": [50, 100],
        "compute_camera": [75, 150],
        "interfaces_imu": [80, 150],
        "power_battery_charger": [120, 220],
        "wiring_hardware": [60, 100],
    }
    mini_development = {"bench_tools": [150, 300], "spares_rework": [100, 200], "cloud": [50, 150]}
    motors = sum(v["count"] * v["unit_price_usd"] for v in mini_motors.values())
    core = [round(motors + sum(v[i] for v in mini_core.values()), 2) for i in (0, 1)]
    development = [sum(v[i] for v in mini_development.values()) for i in (0, 1)]
    tiers["miniature/provisional_upgrade_allowance"] = {
        "height_m": 0.258,
        "height_basis": "Historical revision-10 sizing allowance; the selected silhouette-C mini is approximately 26.8 cm.",
        "mass_budget_including_reserve_kg": None,
        "mass_note": "Existing tail model is 0.825 kg; upgraded hardware and camera mass are not reconciled.",
        "actuator_cost_usd": round(motors, 2),
        "core_parts_and_fabrication_usd": core,
        "development_extras_usd": development,
        "contingency_fraction": 0.25,
        "first_project_usd": [
            float(
                ((Decimal(str(core[i])) + development[i]) * Decimal("1.25")).quantize(
                    Decimal(".01"), rounding=ROUND_HALF_UP
                )
            )
            for i in (0, 1)
        ],
    }
    output = {
        "status": "cost_planning_only_no_validated_walker_at_any_size",
        "checked_on": "2026-10-24",
        "exclusions": [
            "labor",
            "tax",
            "shipping",
            "import_duties",
            "new_workshop_machines",
            "new_host_computer",
            "outsourced_engineering",
            "event_operations",
        ],
        "miniature_assumptions": {
            "actuators": mini_motors,
            "core_allowances_usd": mini_core,
            "development_allowances_usd": mini_development,
        },
        "tiers": tiers,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (here / "cost_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(tiers, indent=2))


if __name__ == "__main__":
    main()
