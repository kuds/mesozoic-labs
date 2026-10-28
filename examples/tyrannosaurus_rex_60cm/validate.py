"""Reproducible packaging, balance and bounded-PD screens; no walking claim."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.robot_clearance import clearance_distance
from examples.tyrannosaurus_rex_60cm.build import DESIGN, HARDWARE, HERE, ROOT


def load() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(HERE / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def crouch_targets(model: mujoco.MjModel, amount: float) -> np.ndarray:
    return np.array(
        [
            {"hip_pitch": -0.5, "knee": 1, "ankle_pitch": -0.5}.get(
                model.joint(model.actuator_trnid[a, 0]).name.split("_", 1)[1], 0
            )
            * amount
            for a in range(model.nu)
        ]
    )


def pose_set(model: mujoco.MjModel) -> list[tuple[str, np.ndarray]]:
    poses = [("home", model.key_qpos[0].copy())]
    for j in range(1, model.njnt):
        for i, angle in enumerate(np.linspace(float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1]), 7)):
            qpos = model.key_qpos[0].copy()
            qpos[model.jnt_qposadr[j]] = angle
            poses.append((f"{model.joint(j).name}:{i}", qpos))
    for i, amount in enumerate(np.linspace(0, 0.6, 31)):
        qpos = model.key_qpos[0].copy()
        qpos[7:] = crouch_targets(model, float(amount))
        poses.append((f"coupled_crouch:{i}", qpos))
    rng = np.random.default_rng(60)
    for i in range(512):
        qpos = model.key_qpos[0].copy()
        qpos[7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1])
        poses.append((f"combined_range:{i}", qpos))
    return poses


def packing(
    model: mujoco.MjModel, data: mujoco.MjData, poses: list[tuple[str, np.ndarray]] | None = None
) -> dict[str, Any]:
    ledger = json.loads((HERE / "components.json").read_text())
    names = ledger["motor_cases"] + ledger["installation_reserves"] + ["left_sole", "right_sole"]
    owners = {frozenset((a, b)) for a, b in ledger["connector_owners"].items()}
    pairs = [
        (model.geom(a).id, model.geom(b).id)
        for a, b in itertools.combinations(names, 2)
        if frozenset((a, b)) not in owners
    ]
    # Leg rails have intentional attachments at their two end motors. Test
    # every other declared hardware item, including the other leg and feet.
    rail_exclusions = []
    for side in ("left", "right"):
        for segment, endpoints in (("thigh", ("hip_pitch", "knee")), ("shin", ("knee", "ankle_pitch"))):
            for index in (0, 1):
                rail = f"{side}_{segment}_rail_{index}"
                for name in names:
                    if name in (f"{side}_{joint}_case" for joint in endpoints):
                        rail_exclusions.append([rail, name])
                    else:
                        pairs.append((model.geom(rail).id, model.geom(name).id))
    moving = [(a, b) for a, b in pairs if model.geom_bodyid[a] != 1 or model.geom_bodyid[b] != 1]
    worst: dict[tuple[int, int], dict[str, Any]] = {}
    failures: dict[str, int] = {}
    incomplete = evaluations = 0
    poses = pose_set(model) if poses is None else poses
    for index, (label, qpos) in enumerate(poses):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for a, b in pairs if index == 0 else moving:
            result = clearance_distance(model, data, a, b, required_clearance_m=0.003, search_distance_m=0.02)
            gap = result["min_distance_m"]
            status = (
                "pass"
                if gap >= 0.003 - 1e-9
                else "incomplete"
                if result["distance_is_lower_bound"] or gap == 0
                else "fail"
            )
            evaluations += 1
            incomplete += status == "incomplete"
            if status == "fail":
                category = label.split(":")[0] if not label.startswith("combined_range") else "combined_range"
                failures[category] = failures.get(category, 0) + 1
            if (a, b) not in worst or gap < worst[(a, b)]["min_distance_m"]:
                worst[(a, b)] = dict(pair=[model.geom(a).name, model.geom(b).name], pose=label, status=status, **result)
                if status != "pass":
                    worst[(a, b)]["qpos"] = qpos.tolist()
    ordered = sorted(worst.values(), key=lambda row: row["min_distance_m"])
    retained = ordered[:24] + [row for row in ordered[24:] if row["status"] != "pass"]
    return dict(
        status="fail" if failures else "incomplete" if incomplete else "pass",
        margin_mm=3,
        pose_count=len(poses),
        pair_count=len(pairs),
        evaluations=evaluations,
        failed_by_pose_family=failures,
        incomplete_evaluations=incomplete,
        worst_pairs=retained,
        retained_pair_evidence="24 smallest minima plus every other failing/inconclusive pair; all pairs evaluated by the reproducible source-bound screen",
        excluded_pairs=[sorted(p) for p in owners] + rail_exclusions,
        scope="All motor-case, connector-reserve, assembly-reserve and sole pairs, except own connectors; eight thigh/shin rails versus that set, except each rail's two attachment motors.",
        omissions="Other frame/yoke proxies, rail-to-rail pairs, mounts, cables along moving links, payload outlines, continuous swept volume and tolerances beyond the stated envelopes.",
    )


def installed_parts() -> dict[str, Any]:
    design = json.loads(DESIGN.read_text())
    failures = []
    margins = []
    for item in design["assemblies"]:
        half = np.array(item["installed_size_m"]) / 2
        for part in item["parts"]:
            margin = half - np.abs(part["offset_m"]) - np.array(part["size_m"]) / 2
            margins.append(dict(part=part["name"], minimum_containment_mm=float(margin.min() * 1000)))
            if margin.min() < 0:
                failures.append(part["name"] + " protrudes")
        for a, b in itertools.combinations(item["parts"], 2):
            gaps = np.abs(np.array(a["offset_m"]) - b["offset_m"]) - (np.array(a["size_m"]) + b["size_m"]) / 2
            if max(gaps) < 0.002 - 1e-9:
                failures.append(a["name"] + " / " + b["name"] + " lacks 2 mm separation")
    return dict(
        status="fail" if failures else "pass",
        failures=failures,
        margins=margins,
        limitation="Box containment and internal board spacing only; cooling, connected cables, access and electrical design unqualified.",
    )


def hardware_contract(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    spec = json.loads(DESIGN.read_text())["motor"]
    ledger = json.loads((HERE / "components.json").read_text())
    cases = ledger["motor_cases"]
    checks = dict(
        twelve_motors_and_actions=len(cases) == model.nu == 12,
        no_constraints=model.neq == 0,
        ledger_mass_matches=abs(float(model.body_mass.sum()) - ledger["mass_kg"]) < 1e-7,
        direct_joint_drives=bool(np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)),
        unit_gear=bool(np.allclose(model.actuator_gear, np.tile([1, 0, 0, 0, 0, 0], (model.nu, 1)))),
        capped_at_design_torque=bool(
            np.all(model.actuator_forcelimited)
            and np.allclose(
                model.actuator_forcerange, [-spec["simulation_torque_cap_nm"], spec["simulation_torque_cap_nm"]]
            )
        ),
    )
    shafts = []
    for name in cases:
        geom = model.geom(name).id
        joint = model.joint(name.removesuffix("_case")).id
        radius, half = model.geom_size[geom, :2]
        checks[name + "_full_size"] = bool(
            radius * 2 >= spec["catalog_diameter_m"] + spec["diameter_tolerance_m"]
            and half * 2 >= spec["catalog_length_m"] + spec["length_tolerance_m"]
        )
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        offset = data.geom_xpos[geom] - data.xanchor[joint]
        error = float(np.linalg.norm(np.cross(offset, data.xaxis[joint])))
        alignment = abs(float(np.dot(axis, data.xaxis[joint])))
        checks[name + "_coaxial"] = error < 1e-8 and alignment > 1 - 1e-8
        shafts.append(dict(case=name, perpendicular_offset_mm=error * 1000, absolute_axis_dot=alignment))
    return dict(
        status="pass" if all(checks.values()) else "fail",
        checks=checks,
        shaft_lines=shafts,
        limitation="Shaft-line alignment does not qualify flanges, bearings, bolts, brackets or centered inertia assumptions.",
    )


def trial(*, crouch: bool = False, cap_override: float | None = None) -> tuple[dict[str, Any], np.ndarray]:
    model, data = load()
    if cap_override is not None:
        model.actuator_forcerange[:] = [-cap_override, cap_override]
    start = data.qpos[:3].copy()
    seconds = 14 if crouch else 12
    trajectory, efforts, heights, forces, com_path = [], [], [], [], []
    maximum_tilt = maximum_drift = 0.0
    nonfoot_support = 0
    min_margin = float("inf")
    for step in range(int(seconds / model.opt.timestep)):
        t = step * model.opt.timestep
        phase = np.clip((t - 2) / 4, 0, 1) if t < 8 else 1 - np.clip((t - 8) / 4, 0, 1)
        phase = phase**3 * (10 - 15 * phase + 6 * phase**2)
        data.ctrl[:] = crouch_targets(model, float(0.6 * phase)) if crouch else 0
        mujoco.mj_step(model, data)
        if step % 10:
            continue
        mujoco.mj_forward(model, data)
        trajectory.append(data.qpos.copy())
        com_path.append(data.subtree_com[1].copy())
        for contact in data.contact:
            names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
            if "floor" in names and not names.intersection({"left_sole", "right_sole"}):
                nonfoot_support += 1
        tilt = np.arccos(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1, 1))
        maximum_tilt = max(maximum_tilt, float(tilt))
        maximum_drift = max(maximum_drift, float(np.linalg.norm(data.qpos[:2] - start[:2])))
        if t > 1:
            efforts.append(data.actuator_force.copy())
            left, right = [float(data.sensor(s + "_contact").data[0]) for s in ("left", "right")]
            forces.append([left, right])
            com = data.subtree_com[1]
            foot_x = [data.site(s + "_sole_center").xpos[0] for s in ("left", "right")]
            # Sagittal screen for these nearly flat, symmetric double-support trials.
            min_margin = min(min_margin, float(min(com[0] - min(foot_x) + 0.080, max(foot_x) + 0.080 - com[0])))
        if crouch and 6.5 <= t < 8:
            heights.append(data.qpos[2])
        if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.12:
            break
    effort = np.asarray(efforts)
    force = np.asarray(forces)
    mean_weight_error = (
        float(np.mean(np.abs(force.sum(axis=1) / (model.body_mass.sum() * 9.81) - 1))) if len(force) else None
    )
    drop = float((start[2] - np.mean(heights)) * 1000) if heights else None
    checks = dict(
        full_horizon=step + 1 == int(seconds / model.opt.timestep),
        tilt_below_8_degrees=maximum_tilt < np.deg2rad(8),
        planar_drift_under_25mm=maximum_drift < 0.025,
        both_feet_loaded=bool(len(force) and np.min(force) > 0),
        feet_carry_body_weight=mean_weight_error is not None and mean_weight_error < 0.05,
        no_other_ground_support=nonfoot_support == 0,
        rms_below_6nm=bool(len(effort) and np.max(np.sqrt(np.mean(effort**2, axis=0))) < 6),
        no_torque_clipping=bool(len(effort) and np.max(np.abs(effort) / model.actuator_forcerange[:, 1]) < 0.99),
    )
    if crouch:
        checks.update(
            crouch_over_40mm=bool(drop is not None and drop > 40),
            return_within_10mm=bool(abs(data.qpos[2] - start[2]) < 0.010),
        )
    checks = {name: bool(value) for name, value in checks.items()}
    return dict(
        status="pass" if all(checks.values()) else "fail",
        checks=checks,
        seconds=seconds,
        controller="Fixed ideal PD, free root, no fixture, no external forces or root resets",
        max_tilt_deg=float(np.rad2deg(maximum_tilt)),
        max_planar_drift_mm=maximum_drift * 1000,
        held_crouch_drop_mm=drop,
        final_height_error_mm=float((data.qpos[2] - start[2]) * 1000),
        max_joint_rms_nm=float(np.max(np.sqrt(np.mean(effort**2, axis=0)))) if len(effort) else None,
        max_joint_peak_nm=float(np.max(np.abs(effort))) if len(effort) else None,
        min_sagittal_com_margin_mm=min_margin * 1000 if np.isfinite(min_margin) else None,
        mean_absolute_relative_foot_weight_error=mean_weight_error,
        nonfoot_support_contacts=nonfoot_support,
        com_min_xyz_m=np.min(com_path, axis=0).tolist(),
        com_max_xyz_m=np.max(com_path, axis=0).tolist(),
        final_com_m=data.subtree_com[1].tolist(),
    ), np.asarray(trajectory)


def balance(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    design = json.loads(DESIGN.read_text())
    mass = float(model.body_mass.sum())
    com = data.subtree_com[1].copy()
    sole = data.site("left_sole_center").xpos.copy()
    crown = data.site("future_head_crown").xpos
    payloads = []
    for extra in (0.0, 0.15, 0.30):
        head_payload = next(item for item in design["future_payloads"] if item["name"] == "head_neck")
        head = np.array(head_payload["position_m"]) + model.body_pos[1]
        adjusted = (mass * com + extra * head) / (mass + extra)
        payloads.append(
            dict(
                extra_head_mass_kg=extra,
                total_mass_kg=mass + extra,
                com_m=adjusted.tolist(),
                forward_shift_mm=float((adjusted[0] - com[0]) * 1000),
            )
        )
    battery_mass = next(item["mass_kg"] for item in design["assemblies"] if item["name"] == "battery")
    battery_shift = battery_mass * 0.04 / mass * 1000
    return dict(
        mass_kg=mass,
        home_com_m=com.tolist(),
        future_head_crown_height_mm=float(crown[2] * 1000),
        nominal_support_bounds_xy_m=[[float(sole[0] - 0.08), float(sole[0] + 0.08)], [-0.156, 0.156]],
        home_sagittal_margin_mm=float((0.080 - abs(com[0] - sole[0])) * 1000),
        head_mass_sensitivity=payloads,
        battery_40mm_translation_com_shift_mm=battery_shift,
        note="Static first moments at home only. Battery translations are sensitivity cases, not clearance-approved alternate installations. CoM projection alone does not certify dynamic balance or single support.",
    )


def source_hashes() -> dict[str, str]:
    paths = [
        DESIGN,
        HERE / "model.xml",
        HERE / "components.json",
        HERE / "build.py",
        Path(__file__),
        ROOT / "environments/shared/robot_building.py",
        ROOT / "environments/shared/robot_clearance.py",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main() -> None:
    model, data = load()
    standing, standing_trajectory = trial()
    crouch, trajectory = trial(crouch=True)
    negative, _ = trial(cap_override=0.10)
    sampled = pose_set(model)
    motion_poses = [(f"standing:{i}", q) for i, q in enumerate(standing_trajectory[::5])]
    motion_poses += [(f"recorded_crouch:{i}", q) for i, q in enumerate(trajectory[::5])]
    full = packing(model, data, sampled)
    operational = packing(model, data, [p for p in sampled if not p[0].startswith("combined_range")] + motion_poses)
    report: dict[str, Any] = dict(
        status="preliminary_study",
        versions={"mujoco": mujoco.__version__, "numpy": np.__version__},
        model=dict(nq=model.nq, nv=model.nv, nu=model.nu, active_constraints=model.neq),
        balance=balance(model, data),
        installed_parts=installed_parts(),
        hardware_contract=hardware_contract(model, data),
        clearance=operational,
        combined_range_probe=full,
        standing=standing,
        crouch=crouch,
        insufficient_torque_negative_control=dict(expected="fail", observed=negative["status"]),
        source_sha256=source_hashes(),
    )
    (HARDWARE / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Small, inspectable motion record used by the render sheet and trajectory clearance screen.
    np.savez_compressed(
        HARDWARE / "crouch_trajectory.npz",
        qpos=trajectory,
        dt=0.02,
        model_sha256=report["source_sha256"][str((HERE / "model.xml").relative_to(ROOT))],
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("source_sha256", "clearance", "combined_range_probe")},
            indent=2,
        )
    )
    print("Clearance:", report["clearance"]["status"], report["clearance"]["failed_by_pose_family"])
    for row in report["clearance"]["worst_pairs"][:12]:
        print(row["pair"], row["pose"], round(row["min_distance_m"] * 1000, 2), row["status"])
    print("Full joint-box probe:", full["status"], full["failed_by_pose_family"])
    for row in full["worst_pairs"][:8]:
        print(row["pair"], row["pose"], round(row["min_distance_m"] * 1000, 2), row["status"])
    # The independent travel + standing/crouch envelope is the M1 gate.
    # Combined joint-box probing diagnoses additional self-collision constraints
    # and never establishes that arbitrary independent commands are safe.
    if (
        operational["status"] != "pass"
        or standing["status"] != "pass"
        or crouch["status"] != "pass"
        or report["installed_parts"]["status"] != "pass"
        or report["hardware_contract"]["status"] != "pass"
        or negative["status"] != "fail"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
