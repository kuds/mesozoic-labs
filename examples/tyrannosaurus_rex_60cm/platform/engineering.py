"""Reconcile installed masses, screen actuator substitutions, budget and pack power."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.intermediate_drive import InstalledDrive
from environments.shared.robot_building import component_inertia, values
from environments.shared.robot_clearance import clearance_distance
from environments.shared.robot_floor import apply_surface

from .validate_platform import actuator_map, command, fresh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HARDWARE = ROOT / "hardware/tyrannosaurus_rex_60cm/platform/engineering"
ASSUMPTIONS = HARDWARE / "assumptions.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget(params):
    rows = [dict(zip(params["budget_columns"], r, strict=True)) for r in params["budget"]]
    totals = {}
    for level in ("low", "target", "high"):
        key = level + "_unit_usd"
        for r in rows:
            r[level + "_usd"] = round(r["quantity"] * r[key], 2)
        totals[level + "_usd"] = round(sum(r[level + "_usd"] for r in rows), 2)
    return {"totals": totals, "lines": rows}


def electrical(p):
    pack_wh = p["pack_nominal_v"] * p["pack_ah"]
    usable = np.array(p["usable_fraction"]) * pack_wh
    aux = sum(p["aux_at_regulated_rails_w"].values()) / p["aux_converter_efficiency"]
    cases = {}
    for name, case in p["duty_cases"].items():
        rms_a = np.array(case["joint_rms_nm"]) / p["motor_torque_constant_nm_per_phase_rms_a"]
        copper_w = 3 * p["phase_resistance_ohm"] * rms_a**2
        shaft_w = np.array(case["mean_positive_mechanical_w"]) / p["gear_forward_efficiency"]
        traction_w = 2 * np.sum(copper_w + shaft_w + p["drive_idle_w_per_axis"]) / p["traction_electrical_efficiency"]
        bus_w = aux + traction_w
        cases[name] = dict(
            bus_w=bus_w,
            traction_bus_w=traction_w,
            aux_bus_w=aux,
            winding_heat_w_one_of_each_joint=copper_w.tolist(),
            total_winding_heat_w=2 * np.sum(copper_w),
            bus_a_at_loaded_stop=bus_w / p["assumed_loaded_stop_v"],
            runtime_minutes_by_usable_fraction=(60 * usable / bus_w).tolist(),
        )
    max_tether_a = p["assumed_peak_bus_power_w"] / p["tether_v"]
    return dict(
        cases=cases,
        nominal_wh=pack_wh,
        usable_wh=usable.tolist(),
        avg_bus_w_for_15min_at_mid_usable=4 * usable[1],
        avg_bus_w_for_20min_at_mid_usable=3 * usable[1],
        peak_bus_a_at_loaded_stop=p["assumed_peak_bus_power_w"] / p["assumed_loaded_stop_v"],
        tether_peak_a_first_order=max_tether_a,
        tether_drop_v_at_peak_estimate=max_tether_a * p["design_bus_cable_loop_resistance_ohm"],
        tether_wire_loss_w_at_peak_estimate=max_tether_a**2 * p["design_bus_cable_loop_resistance_ohm"],
    )


def swap_motors(folder, specs, scheme):
    """Physical masses change at their mounted locations. Retain original bracket and connector reserves."""
    rows = deepcopy(json.loads((folder / "components.json").read_text())["components"])
    doc = ET.parse(folder / "model.xml").getroot()
    changed = set()
    limits = {}
    for row in rows:
        if row["category"] != "leg_motors":
            continue
        joint = row["name"].removesuffix("_motor")
        part = "RMD"
        if scheme == "mixed_AK":
            part = "AK45_36" if joint.endswith("hip_roll") else "AK45_10"
        elif scheme == "yaw_AK" and joint.endswith("hip_yaw"):
            part = "AK45_10"
        spec = specs[part]
        limits[joint] = {"sku": part, "screen_continuous_nm": 0.8 * spec["rated_nm"]}
        if part == "RMD":
            continue
        row["mass_kg"] = spec["mass_kg"]
        # Keep original mounting center and axis. Keep connector and larger M1 case
        # guards as conservative reservations. Actual flange/hole pattern needs CAD.
        old = np.array(row["inertia_box_halfsize_m"])
        axial = int(np.argmax(old))
        half = np.full(3, spec["diameter_mm"] * 0.0005)
        half[axial] = spec["length_mm"] * 0.0005
        row["inertia_box_halfsize_m"] = half.tolist()
        changed.add(row["body"])
    if len(limits) != 12:
        raise ValueError("expected the twelve explicit leg motor allocations")
    for name in changed:
        body = doc.find(f".//body[@name='{name}']/inertial")
        mass, center, tensor = component_inertia([r for r in rows if r["body"] == name])
        xx, yy, zz = np.diag(tensor)
        xy, xz, yz = tensor[0, 1], tensor[0, 2], tensor[1, 2]
        for key in ("diaginertia", "quat"):
            body.attrib.pop(key, None)
        body.attrib.update(mass=str(mass), pos=values(center), fullinertia=values([xx, yy, zz, xy, xz, yz]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(doc, encoding="unicode"))
    total = sum(row["mass_kg"] for row in rows)
    if not np.isclose(total, np.sum(model.body_mass), atol=1e-7, rtol=0):
        raise ValueError("unreconciled component mass")
    return model, rows, limits


def static_loads(model, limits, bend_rad):
    """Two flat feet, stationary and vertical GRFs at CoPs directly fore/aft of CoM."""
    d = mujoco.MjData(model)
    d.qpos[:] = model.key_qpos[0] if model.nkey else model.qpos0
    for side in ("left", "right"):
        for name, scale in (("hip_pitch", -0.5), ("knee", 1.0), ("ankle_pitch", -0.5)):
            j = model.joint(f"{side}_{name}")
            d.qpos[j.qposadr] += scale * bend_rad
            if not j.range[0] <= d.qpos[j.qposadr][0] <= j.range[1]:
                raise ValueError(f"{j.name} outside range")
    mujoco.mj_forward(model, d)
    sides = ("left", "right")
    sites = [d.site(s + "_sole_center").xpos.copy() for s in sides]
    root = model.body("pelvis").id
    com = d.subtree_com[root]
    left_frac = (com[1] - sites[1][1]) / (sites[0][1] - sites[1][1])
    tau = d.qfrc_bias.copy()
    edge = []
    for side, center, fraction in zip(sides, sites, (left_frac, 1 - left_frac), strict=True):
        sole = model.geom(side + "_sole")
        point = np.array([com[0], center[1], center[2]])
        local = d.geom(sole.id).xmat.reshape(3, 3).T @ (point - d.geom(sole.id).xpos)
        edge.append(float(np.min(sole.size[:2] - np.abs(local[:2]))))
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, d, jac, None, point, model.body(side + "_foot").id)
        tau -= jac[2, :] * np.sum(model.body_mass) * 9.81 * fraction
    if not 0 < left_frac < 1 or min(edge) < 0.003 or np.max(np.abs(tau[:6])) > 1e-5:
        raise ValueError("infeasible double-support reaction")
    axes = {}
    for joint, spec in limits.items():
        torque = float(tau[model.joint(joint).dofadr[0]])
        axes[joint] = dict(**spec, hold_nm=torque, utilization=abs(torque) / spec["screen_continuous_nm"])
    return dict(
        knee_bend_rad=bend_rad,
        foot_plane_height_before_lowering_m=sites[0][2],
        cop_min_edge_m=min(edge),
        max_unbalanced_base_wrench=float(np.max(np.abs(tau[:6]))),
        joint_loads=axes,
        exceeds_screen=[j for j, a in axes.items() if a["utilization"] > 1],
    )


def extraction(folder):
    """Union of a translated axis-aligned box over a straight stroke is another box."""
    xml = ET.parse(folder / "model.xml").getroot()
    pelvis = xml.find(".//body[@name='pelvis']")
    pack = pelvis.find("geom[@name='battery_pack']")
    if pack is None:
        raise ValueError("expected a visible tolerance-max battery")
    origin = np.fromstring(pack.get("pos"), sep=" ")
    half = np.fromstring(pack.get("size"), sep=" ")
    # Assume the EC5 plug and balance leads have been unplugged and clipped inside
    # a sleeve. +6 mm in every direction for sleeve, retained leads and a pull tab.
    guard = 0.006
    excluded = {"battery_pack", "battery_reserve"}
    out = {}
    for name, axis, sign, stroke_m in (
        ("aft", 0, -1, 0.34),
        ("left", 1, 1, 0.24),
        ("right", 1, -1, 0.24),
        ("down", 2, -1, 0.22),
    ):
        center, extents = origin.copy(), half + guard
        center[axis] += sign * stroke_m / 2
        extents[axis] += stroke_m / 2
        geom = ET.SubElement(pelvis, "geom", type="box", name="withdrawal", pos=values(center), size=values(extents))
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        g = model.geom("withdrawal").id
        blockers = []
        for i in range(model.ngeom):
            other = model.geom(i)
            # Thin graphic outline and floor are omitted; physical skins and
            # bracket geometry remain in scope. Other assembly reserves remain.
            if (
                i == g
                or other.name in excluded
                or other.name == "floor"
                or other.name.startswith("outline_")
                or other.name.startswith("guide_")
            ):
                continue
            result = clearance_distance(model, data, g, i, required_clearance_m=0.002, search_distance_m=0.01)
            if result["min_distance_m"] < 0.002:
                blockers.append(
                    {
                        "geom": other.name,
                        "native_or_bound_m": result["min_distance_m"],
                        "method": result["distance_method"],
                    }
                )
        out[name] = dict(
            swept_box_center_m=center.tolist(),
            swept_box_halfsize_m=extents.tolist(),
            includes_6mm_lead_retainer=True,
            stroke_m=stroke_m,
            criterion_m=0.002,
            blockers=blockers,
        )
        pelvis.remove(geom)
    return out


def component_ledger(model, rows):
    """One line per actual mass, with a pointer into the independently costed lots."""
    grouped = {
        "leg_motors": "leg_actuator",
        "frames_bearings_feet": "frame",
        "harness": "harness",
        "foot_sensors": "foot_sensing",
        "fixed_arms": "covers",
        "covers_brackets": "covers",
        "power_auxiliaries": "power",
        "appendage_motors": "expression_servo",
        "appendage_mounts": "frame",
        "moving_head": "covers",
        "moving_tail": "covers",
    }
    direct = {
        "battery": "battery",
        "compute": "compute",
        "control": "control",
        "power": "power",
        "skull_mount": "frame",
        "oak_d_lite_bracket_cable": "rgbd",
        "tof_left_strut_cable": "tof",
        "tof_right_strut_cable": "tof",
        "integration_allowance": None,
    }
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = []
    for row in rows:
        name, group = row["name"], row["category"]
        cost_id = direct[name] if name in direct else grouped[group]
        owner = data.body(row["body"])
        _, local, tensor = component_inertia([row])
        center = owner.xpos + owner.xmat.reshape(3, 3) @ local
        source = {"leg_motors": "rmd", "appendage_motors": "xm"}.get(group)
        result.append(
            dict(
                component=name,
                parent_body=row["body"],
                group=group,
                mass_kg=row["mass_kg"],
                mass_evidence="catalog nominal" if source else "installed allocation",
                catalog_mass_source_id=source,
                cost_line_id=cost_id,
                center_local_m=local.tolist(),
                home_center_world_m=center.tolist(),
                local_central_inertia_diag_kg_m2=np.diag(tensor).tolist(),
                inertia_evidence="axis-aligned uniform box surrogate",
            )
        )
    return result


def source_sized_model(folder, p):
    """Produce a service model with published minimum box dimensions and frozen mass properties."""
    xml = ET.parse(folder / "model.xml")
    original = {}
    for part in p["installed_box_checks"]:
        geom = xml.find(f".//geom[@name='{part['geom']}']")
        if geom is None or geom.get("type") != "box":
            raise ValueError(f"missing box {part['geom']}")
        full_mm = 2000 * np.fromstring(geom.get("size"), sep=" ")
        original[part["geom"]] = full_mm
        geom.set("size", values(np.maximum(full_mm, part["minimum_mm"]) / 2000))
    return xml, original


def installed_size_checks(folder, p):
    xml, original = source_sized_model(folder, p)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
    data = fresh(model)
    rows = []
    for part in p["installed_box_checks"]:
        gid = model.geom(part["geom"]).id
        parent = part["parent"]
        margin = None
        if parent:
            e = model.geom(parent).id
            # Exact projection of an oriented child box into parent coordinates.
            parent_rotation = data.geom_xmat[e].reshape(3, 3).T
            projected = abs(parent_rotation @ data.geom_xmat[gid].reshape(3, 3)) @ model.geom_size[gid]
            center = parent_rotation @ (data.geom_xpos[gid] - data.geom_xpos[e])
            margin = 1000 * min(model.geom_size[e] - abs(center) - projected)
        rows.append(
            dict(
                **part,
                frozen_proxy_mm=original[part["geom"]].tolist(),
                frozen_covers_part=bool(np.all(original[part["geom"]] + 1.0e-6 >= part["minimum_mm"])),
                service_proxy_mm=(2000 * model.geom_size[gid]).tolist(),
                worst_inside_parent_face_gap_mm=margin,
            )
        )
    # Pair every selected board against other boards that share the same power bay,
    # including generic protection/distribution. Distinct assembly boxes have their
    # own full-volume clearance screen in platform validation.
    distances = []
    boards = ["logic_converter", "compute_converter", "protection_distribution"]
    for i, a in enumerate(boards):
        for b in boards[i + 1 :]:
            gap = clearance_distance(
                model, data, model.geom(a).id, model.geom(b).id, required_clearance_m=0.002, search_distance_m=0.1
            )
            distances.append(
                dict(a=a, b=b, margin_or_bound_mm=1000 * gap["min_distance_m"], method=gap["distance_method"])
            )
    body_gaps = [r["worst_inside_parent_face_gap_mm"] for r in rows if r["parent"]]
    return {
        "proxies": rows,
        "frozen_undersized": [r["geom"] for r in rows if not r["frozen_covers_part"]],
        "main_power_bay_internals": distances,
        "service_geometry_pass": bool(
            min(body_gaps) >= 2 - 1e-9 and min(r["margin_or_bound_mm"] for r in distances) >= 2 - 1e-9
        ),
        "service_xml_sha256": hashlib.sha256(ET.tostring(xml.getroot())).hexdigest(),
    }


def supported_power_trials(folder, config):
    """Use the same installed battery, live or isolated; compare bus voltage on a shared physical trial."""
    rows = json.loads((folder / "components.json").read_text())["components"]
    (pack,) = (r for r in rows if r["name"] == "battery")
    if pack["mass_kg"] <= 0:
        raise ValueError("power configuration requires a carried, nonzero cassette mass")
    p = config["power"]
    case_voltages = {
        "partially_discharged": p["assumed_loaded_stop_v"],
        "battery_nominal": p["pack_nominal_v"],
        "external_retained_isolated_pack": p["tether_v"],
    }
    output = {}
    for mode, voltage in case_voltages.items():
        model = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
        apply_surface(model, "hard")
        drive = InstalledDrive(model, voltage=voltage)
        data = fresh(model)
        drive.reset(data)
        ids = actuator_map(model)
        samples = round(4.0 / model.opt.timestep)
        copper, shaft, bus = [], [], []
        initial_h = float(data.qpos[2])
        min_h = initial_h
        for k in range(samples):
            target = command(model, ids, "crouch", k * model.opt.timestep)
            drive.step(data, target)
            tau = data.actuator_force[drive.traction]
            speed = data.actuator_velocity[drive.traction]
            heat = np.sum(3 * p["phase_resistance_ohm"] * (tau / p["motor_torque_constant_nm_per_phase_rms_a"]) ** 2)
            pos = np.sum(np.maximum(0.0, tau * speed))
            leg_electrical = (
                heat + pos / p["gear_forward_efficiency"] + sum(drive.traction) * p["drive_idle_w_per_axis"]
            ) / p["traction_electrical_efficiency"]
            aux = sum(p["aux_at_regulated_rails_w"].values()) / p["aux_converter_efficiency"]
            copper.append(heat)
            shaft.append(pos)
            bus.append(leg_electrical + aux)
            min_h = min(min_h, float(data.qpos[2]))
        output[mode] = {
            "actual_carried_kg": float(sum(model.body_mass)),
            "installed_cassette": pack,
            "simulated_bus_voltage_v": voltage,
            "duration_s": 4.0,
            "every_physics_step": True,
            "peak_bus_w_estimate": float(max(bus)),
            "mean_bus_w_estimate": float(np.mean(bus)),
            "mean_leg_copper_w_estimate": float(np.mean(copper)),
            "mean_positive_leg_output_w_in_simulation": float(np.mean(shaft)),
            "max_root_drop_m": initial_h - min_h,
            "note": "prescribed free-standing crouch, auxiliary power assumed",
        }
    return output


def analyze_case(folder, p):
    schemes = {}
    ledger = None
    for scheme in ("RMD", "mixed_AK", "yaw_AK"):
        model, rows, limits = swap_motors(folder, p["motor_screen"], scheme)
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        if scheme == "RMD":
            ledger = component_ledger(model, rows)
        schemes[scheme] = dict(
            installed_mass_kg=float(np.sum(model.body_mass)),
            com_home_world_m=d.subtree_com[model.body("pelvis").id].tolist(),
            static_screens=[static_loads(model, limits, a) for a in (0, 0.3, 0.6)],
        )
    return dict(
        installed_size_checks=installed_size_checks(folder, p),
        schemes=schemes,
        components=ledger,
        extraction=extraction(folder),
        model_sha256=sha(folder / "model.xml"),
        components_sha256=sha(folder / "components.json"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["compact", "nominal", "wide"])
    ap.add_argument("--output", type=Path, default=HARDWARE / "engineering_report.json")
    ap.add_argument("--service-xml", type=Path, help="Export nominal source-sized installed hardware inspection XML")
    args = ap.parse_args()
    p = json.loads(ASSUMPTIONS.read_text())
    if args.service_xml:
        service_xml, _ = source_sized_model(HERE / "nominal", p)
        args.service_xml.write_bytes(ET.tostring(service_xml.getroot()))
    report = dict(
        assumptions_sha256=sha(ASSUMPTIONS),
        validator_sha256=sha(Path(__file__)),
        design_sha256=sha(HERE / "design.json"),
        mujoco=mujoco.__version__,
        clearance_implementation_sha256=sha(ROOT / "environments/shared/robot_clearance.py"),
        inertia_implementation_sha256=sha(ROOT / "environments/shared/robot_building.py"),
        installed_drive_sha256=sha(ROOT / "environments/shared/intermediate_drive.py"),
        trial_command_sha256=sha(HERE / "validate_platform.py"),
        floor_sha256=sha(ROOT / "environments/shared/robot_floor.py"),
        power_modes=supported_power_trials(HERE / "nominal", p),
        status="model_screen_only",
        procurement=budget(p),
        power=electrical(p["power"]),
        variants={name: analyze_case(HERE / name, p) for name in args.variants},
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
