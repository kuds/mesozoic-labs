"""Reconcile the detailed parts plan; this does not validate walking or CAD fit."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


def money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main() -> None:
    here = Path(__file__).resolve().parent
    source = here / "parts_budget.json"
    baseline_path = here.parents[3] / "examples/mini_tyrannosaurus_rex/walking_validation.json"
    data = json.loads(source.read_text())
    baseline = json.loads(baseline_path.read_text())
    model = next(m for m in baseline["models"] if m["name"] == "fixed_tail_base")
    trial = next(t for t in model["trials"] if t["requested_duration_s"] == 20 and t["cap"] == 0.2)
    if trial["terminated_early"] or trial["display_support_active"]:
        raise ValueError("Reference diagnostic changed; reassess the similarity screen.")
    ref_knee = max(trial["rms_torque_Nm"][i] for i, name in enumerate(model["actuator_names"]) if "knee" in name)
    categories: defaultdict[str, Decimal] = defaultdict(Decimal)
    scopes: defaultdict[str, Decimal] = defaultdict(Decimal)
    seen = set()
    for row in data["rows"]:
        if row["id"] in seen or row["quantity"] <= 0 or Decimal(row["unit_price_usd"]) < 0:
            raise ValueError(f"Invalid or duplicated line: {row['id']}")
        if row["price_basis"].startswith("catalog") and not row["source_url"]:
            raise ValueError(f"Missing catalog source: {row['id']}")
        seen.add(row["id"])
        total = row["quantity"] * Decimal(row["unit_price_usd"])
        row["line_total_usd"] = money(total)
        categories[row["category"]] += total
        scopes[row["scope"]] += total
    subtotal = sum(scopes.values(), Decimal(0))
    contingency = subtotal * Decimal(data["contingency_fraction"])
    reserve = Decimal(data["tax_freight_import_planning_reserve_usd"])
    mass = sum(data["mass_allowances_g"].values()) / 1000
    ak_mass = mass - (330 - data["comparison"]["ak45_mass_g"]) * 12 / 1000
    scale = data["target_height_m"] / data["reference_height_m"]
    battery = data["battery_scenario"]
    energy = battery["nominal_voltage_v"] * battery["capacity_ah"]
    can = data["can_scenario"]
    head = data["head_subset_scenario"]
    result = {
        "status": "arithmetic_and_similarity_screen_only_not_hardware_validation",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, Path(__file__), baseline_path)
        },
        "category_totals_usd": {k: money(v) for k, v in categories.items()},
        "scope_totals_usd": {k: money(v) for k, v in scopes.items()},
        "project_subtotal_usd": money(subtotal),
        "contingency_usd": money(contingency),
        "project_with_contingency_usd": money(subtotal + contingency),
        "tax_freight_import_planning_reserve_usd": money(reserve),
        "funding_plan_usd": money(subtotal + contingency + reserve),
        "mass_with_reserve_kg": mass,
        "allocated_mass_without_reserve_kg": mass - data["mass_allowances_g"]["unallocated_integration_reserve"] / 1000,
        "rmd_similarity_knee_rms_nm": ref_knee * mass / model["mass_kg"] * scale,
        "ak45_same_nonmotor_ledger_mass_kg": ak_mass,
        "ak45_similarity_knee_rms_nm": ref_knee * ak_mass / model["mass_kg"] * scale,
        "ak45_installed_motor_cost_saving_usd": money(
            (Decimal("399") - Decimal(data["comparison"]["ak45_unit_price_usd"])) * 12
        ),
        "battery_nominal_wh": energy,
        "battery_usable_scenario_wh": energy * battery["usable_fraction"],
        "runtime_scenario_minutes": {
            str(w): 60 * energy * battery["usable_fraction"] / w for w in battery["average_pack_power_w"]
        },
        "can_bus_occupancy_scenario_fraction": can["bits_per_frame"]
        * can["frames_per_motor_update"]
        * can["motors_per_bus"]
        * can["loop_hz"]
        / can["bitrate"],
        "head_static_gravity_moment_nm": head["moving_mass_kg"] * 9.81 * head["com_offset_m"],
        "limitations": [
            "The baseline stepping trial eventually falls; it is not a validated hardware gait.",
            "Similarity assumes proportional mass distribution and slower cadence, which the actual replacement parts do not preserve.",
            "New ankle-roll loads, unconstrained peak requirements, impacts and low-speed heating are not established.",
            "Noncatalog costs and most masses are planning allowances; power/runtime are scenarios.",
        ],
    }
    (here / "parts_budget_results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (here / "parts_budget.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["rows"][0]))
        writer.writeheader()
        writer.writerows(data["rows"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
