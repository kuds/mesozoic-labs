"""Reproduce the intermediate Rex mechanical screens from the repository root."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.robot_clearance import clearance_distance
from environments.shared.robot_floor import FLOORS, apply_surface

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / "hardware/tyrannosaurus_rex_60cm/m02_integrated/mechanical_report.json"
SIDES = ("left", "right")
MARGIN = 0.003
VARIANTS = ("compact", "nominal", "wide")


def fresh(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def actuator_map(model):
    return {model.joint(j).name: i for i, j in enumerate(model.actuator_trnid[:, 0])}


def sole_bottom(model, data, side):
    geom = model.geom(side + "_sole").id
    return float(data.geom_xpos[geom, 2] - np.abs(data.geom_xmat[geom].reshape(3, 3)[2]) @ model.geom_size[geom])


def pose(model, commands):
    """Direct-joint scalar targets in actuator order to free-root kinematic pose."""
    q = model.key_qpos[0].copy()
    q[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = commands
    return q


def command(model, ids, trial, time):
    target = model.key_ctrl[0].copy()
    # The exact same functions drive all candidates, loaded on free roots.
    if trial == "crouch":
        amount = 0.44 * math.sin(math.pi * np.clip((time - 0.5) / 3.0, 0, 1)) ** 2
        for side in SIDES:
            for joint, ratio in (("hip_pitch", -0.5), ("knee", 1.0), ("ankle_pitch", -0.5)):
                target[ids[side + "_" + joint]] += ratio * amount
    if trial.startswith("hold_"):
        stance = trial.removeprefix("hold_")
        sign, swing = (1.0, "right") if stance == "left" else (-1.0, "left")

        def ramp(offset, duration):
            return 0.5 - 0.5 * math.cos(math.pi * float(np.clip((time - offset) / duration, 0, 1)))

        shift = ramp(0.35, 0.65)
        lift = ramp(1.15, 0.30)
        for side in SIDES:
            target[ids[side + "_hip_roll"]] = -0.36 * sign * shift
            target[ids[side + "_ankle_roll"]] = 0.40 * sign * shift
        target[ids[swing + "_knee"]] = 0.55 * lift
        target[ids[swing + "_hip_pitch"]] = -0.275 * lift
        target[ids[swing + "_ankle_pitch"]] = -0.275 * lift
    if trial in ("transfer", "turn_probe"):
        progress = np.clip((time - 0.5) / 4.0, 0, 1)
        theta = progress * 4 * math.pi
        wave = math.sin(theta)
        for sign, side in zip((1.0, -1.0), SIDES):
            swing = max(0, -sign * wave) ** 2
            target[ids[side + "_hip_roll"]] = -0.24 * wave
            target[ids[side + "_ankle_roll"]] = 0.24 * wave
            target[ids[side + "_hip_pitch"]] = -0.26 * swing + 0.08 * sign * math.sin(theta) ** 2 * math.cos(theta)
            target[ids[side + "_knee"]] = 0.50 * swing
            target[ids[side + "_ankle_pitch"]] = -target[ids[side + "_hip_pitch"]] - target[ids[side + "_knee"]]
            if trial == "turn_probe":
                target[ids[side + "_hip_yaw"]] = 0.12 * sign * math.sin(theta) ** 2
        for joint, amplitude in (("neck_pitch", 0.15), ("tail_yaw", -0.20)):
            if joint in ids:
                target[ids[joint]] = amplitude * wave
    return np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def single_support(model):
    """One vertical ground wrench at the projected CoM; no other supports."""
    data, ids = fresh(model), actuator_map(model)
    root = model.body("pelvis").id
    home_com = data.subtree_com[root].copy()
    rows = []
    for sign, stance, swing in ((1, "left", "right"), (-1, "right", "left")):
        sole = model.geom(stance + "_sole").id
        nominal = float(data.site(stance + "_sole_center").xpos[1] - home_com[1])
        best = None
        # Scan the full admissible counter-roll with flat soles and a flexed swing leg.
        for tilt in np.linspace(0, 0.45, 91):
            target = model.key_ctrl[0].copy()
            for side in SIDES:
                target[ids[side + "_hip_roll"]] = -tilt * sign
                target[ids[side + "_ankle_roll"]] = tilt * sign
            for joint, angle in (("hip_pitch", -0.25), ("knee", 0.5), ("ankle_pitch", -0.25)):
                target[ids[swing + "_" + joint]] = angle
            limits = model.jnt_range[model.actuator_trnid[:, 0]]
            if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
                continue
            data.qpos[:] = pose(model, target)
            mujoco.mj_forward(model, data)
            com = data.subtree_com[root].copy()
            support = data.site(stance + "_sole_center").xpos.copy()
            cop = np.array([com[0], com[1], support[2]])
            local = data.geom_xmat[sole].reshape(3, 3).T @ (cop - support)
            margin = float(np.min(model.geom_size[sole, :2] - np.abs(local[:2])))
            clearance = sole_bottom(model, data, swing) - sole_bottom(model, data, stance)
            jac = np.empty((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, cop, model.geom_bodyid[sole])
            reaction = np.array([0, 0, -model.opt.gravity[2] * model.body_mass.sum()])
            effort = data.qfrc_bias - jac.T @ reaction
            torque = effort[model.jnt_dofadr[model.actuator_trnid[:, 0]]]
            if best is None or margin > best["signed_sole_margin_m"]:
                best = dict(
                    stance=stance,
                    nominal_sole_center_minus_com_y_m=nominal,
                    commanded_hip_roll_rad=-sign * tilt,
                    signed_sole_margin_m=margin,
                    swing_clearance_m=clearance,
                    joint_torque_nm=torque.tolist(),
                    max_sustained_effort_ratio=float(
                        np.max(np.abs(torque) / model.numeric("drive_continuous_nm").data)
                    ),
                    max_peak_effort_ratio=float(np.max(np.abs(torque) / model.actuator_forcerange[:, 1])),
                    root_residual_force_n=float(np.linalg.norm(effort[:3])),
                    root_residual_moment_nm=float(np.linalg.norm(effort[3:6])),
                    geometric_and_peak_torque_feasible=bool(
                        margin >= MARGIN
                        and clearance >= 0.008
                        and np.all(np.abs(torque) <= model.actuator_forcerange[:, 1])
                    ),
                )
        rows.append(best)
        data = fresh(model)
    return rows


def dynamics(model, condition, surface, payload, voltage):
    from environments.shared.intermediate_drive import InstalledDrive

    apply_surface(model, surface)
    # The sensitivity adds uniform head mass at the head assembly's own CoM.
    # Scale its local inertia proportionally to preserve radius of gyration.
    if payload:
        h = model.body("head").id
        model.body_inertia[h] *= (model.body_mass[h] + payload) / model.body_mass[h]
        model.body_mass[h] += payload
        mujoco.mj_setConst(model, mujoco.MjData(model))
    data, ids = fresh(model), actuator_map(model)
    drive = InstalledDrive(model, voltage=voltage)
    drive.reset(data)
    dt = model.opt.timestep
    horizon = {"stand": 4.0, "crouch": 4.0, "transfer": 5.0, "turn_probe": 5.0, "hold_left": 3.2, "hold_right": 3.2}[
        condition
    ]
    steps = round(horizon / dt)
    targets = np.array([command(model, ids, condition, t * dt) for t in range(steps)])
    mass = float(model.body_mass.sum())
    weight = mass * -model.opt.gravity[2]
    above_continuous = np.zeros(model.nu)
    peak_drive_budget_used = np.zeros(model.nu)
    max_torque = np.zeros(model.nu)
    torque_ss = np.zeros(model.nu)
    speed_peak = np.zeros(model.nu)
    foot_ids = [model.geom(side + "_sole").id for side in SIDES]
    floor = model.geom("floor").id
    root = model.body("pelvis").id
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    support_run = np.zeros(2)
    longest = np.zeros(2)
    min_load = np.full(2, np.inf)
    max_clearance = np.zeros(2)
    max_load = max_tilt = max_drift = max_slip = deepest = max_yaw = 0.0
    lowest = start_z = float(data.qpos[2])
    first_other = None
    other_count = 0
    wrench = np.empty(6)
    initial_xy = data.xpos[root, :2].copy()
    old_foot = data.geom_xpos[foot_ids, :2].copy()
    old_loaded = [False, False]
    failure = None
    recorded = [data.qpos.copy()]
    for i, target in enumerate(targets):
        drive.step(data, target)
        # mj_step retains the forces actually used for this tick. Kinematics and
        # contacts refer to the start of this step; qpos and qvel are integrated.
        if (i + 1) % 10 == 0:
            recorded.append(data.qpos.copy())
        max_torque = np.maximum(max_torque, np.abs(data.actuator_force))
        torque_ss += data.actuator_force**2
        above_continuous += (np.abs(data.actuator_force) > drive.cont) * dt
        peak_drive_budget_used = np.maximum(peak_drive_budget_used, drive.used)
        speed_peak = np.maximum(speed_peak, np.abs(data.qvel[dofs]))
        loads = np.zeros(2)
        other = False
        for index, c in enumerate(data.contact):
            if floor not in (c.geom1, c.geom2):
                continue
            geom = c.geom2 if c.geom1 == floor else c.geom1
            mujoco.mj_contactForce(model, data, index, wrench)
            force = (c.frame.reshape(3, 3).T @ wrench[:3])[2] * (1 if c.geom1 == floor else -1)
            if geom not in foot_ids:
                if force > 1.0:
                    other = True
                    if first_other is None:
                        first_other = dict(at_s=i * dt, geom=model.geom(geom).name)
            else:
                loads[foot_ids.index(geom)] += force
                deepest = max(deepest, -float(c.dist))
        other_count += other
        min_load = np.minimum(min_load, loads / weight)
        max_load = max(max_load, float(loads.max()) / weight)
        bottom = [sole_bottom(model, data, side) for side in SIDES]
        max_clearance = np.maximum(max_clearance, bottom)
        feet = data.geom_xpos[foot_ids, :2].copy()
        loaded = loads > 0.20 * weight
        for side in (0, 1):
            if loaded[side] and old_loaded[side]:
                max_slip = max(max_slip, float(np.linalg.norm(feet[side] - old_foot[side])) / dt)
            stable = loads[side] >= 0.8 * weight and loads[1 - side] <= 0.05 * weight
            stable = stable and bottom[1 - side] >= 0.008 and not other
            support_run[side] = support_run[side] + dt if stable else 0.0
        longest = np.maximum(longest, support_run)
        old_foot, old_loaded = feet, loaded
        rotation = data.xmat[root].reshape(3, 3)
        tilt = math.acos(np.clip(rotation[2, 2], -1, 1))
        max_tilt = max(max_tilt, math.degrees(tilt))
        max_yaw = max(max_yaw, abs(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))))
        max_drift = max(max_drift, float(np.linalg.norm(data.xpos[root, :2] - initial_xy)))
        lowest = min(lowest, float(data.xpos[root, 2]))
        if not np.isfinite(data.qpos).all() or tilt > math.radians(55) or data.xpos[root, 2] < 0.16:
            failure = "fall_or_nonfinite"
            break
    checks = {
        "completed": failure is None,
        "tilt_below_12deg": max_tilt < 12,
        "no_nonsole_support": other_count == 0,
    }
    if condition in ("stand", "crouch"):
        checks["drift_under_25mm"] = max_drift < 0.025
    if condition.startswith("hold_"):
        checks["selected_single_support_over_1s"] = bool(longest[SIDES.index(condition.removeprefix("hold_"))] >= 1.0)
    if condition == "crouch":
        checks["drop_over_20mm_and_return_within_10mm"] = bool(
            start_z - lowest > 0.02 and abs(data.qpos[2] - start_z) < 0.01
        )
    if condition in ("transfer", "turn_probe"):
        checks["both_single_support_over_100ms"] = bool(np.all(longest >= 0.1))
    result = dict(
        condition=condition,
        surface=surface,
        added_head_kg=payload,
        voltage_v=voltage,
        mass_kg=mass,
        steps=i + 1,
        duration_s=(i + 1) * dt,
        requested_s=horizon,
        status="pass" if all(checks.values()) else "fail",
        checks=checks,
        early_stop=failure,
        peak_joint_torque_nm=max_torque.tolist(),
        rms_joint_torque_nm=np.sqrt(torque_ss / (i + 1)).tolist(),
        above_continuous_s=above_continuous.tolist(),
        max_derating_state_0_to_1=peak_drive_budget_used.tolist(),
        peak_joint_speed_rad_s=speed_peak.tolist(),
        max_tilt_deg=max_tilt,
        max_planar_displacement_m=max_drift,
        max_abs_yaw_deg=max_yaw,
        pelvis_drop_m=start_z - lowest,
        max_vertical_single_foot_ground_force_bodyweights=max_load,
        max_loaded_sole_center_speed_m_s=max_slip,
        deepest_sole_contact_m=deepest,
        max_swing_bottom_height_m=max_clearance.tolist(),
        longest_single_support_s=longest.tolist(),
        minimum_foot_load_bodyweights=min_load.tolist(),
        nonsole_support_steps=other_count,
        first_nonsole_support=first_other,
    )
    return result, recorded


def physical_pairs(model):
    # Envelopes already contain the internal battery / board solids (group 4).
    # Identify connector-to-own-case and mating rail ends explicitly.
    geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] != 0
        and model.geom_group[g] != 4
        and "_face_" not in model.geom(g).name
        and not model.geom(g).name.startswith("outline_")
    ]
    excluded = {}
    for side in SIDES:
        for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
            stem = f"{side}_{joint}"
            excluded[frozenset([stem + "_cable_reserve", stem + "_case"])] = "own cable outlet"
        for segment, ends in (("thigh", ("hip_pitch", "knee")), ("shin", ("knee", "ankle_pitch"))):
            for i in (0, 1):
                for joint in ends:
                    excluded[frozenset([f"{side}_{segment}_rail_{i}", f"{side}_{joint}_case"])] = "bearing attachment"
    included = []
    for a, b in combinations(geoms, 2):
        names = model.geom(a).name, model.geom(b).name
        if frozenset(names) in excluded:
            continue
        if model.geom_bodyid[a] == model.geom_bodyid[b]:
            # Same rigid body's frame members can join each other. Include all
            # same-body fit checks where either surface is an installed reserve
            # or an actuator case (including the static pelvis).
            if not any("reserve" in n or n.endswith("_case") for n in names):
                continue
        included.append((a, b))
    return np.array(included, dtype=int), sorted(model.geom(g).name for g in geoms)


def swept_radii(model, pairs):
    """Upper bound on relative geometric motion for piecewise-linear hinge paths.

    For each hinge, maximum distance to any point of a descendant geometry
    is bounded by the sum of rigid offsets and a circumscribed sphere radius.
    Motion above the common ancestor is rigid and cannot change separation.
    """
    joints = model.actuator_trnid[:, 0]
    arm = np.zeros((model.ngeom, model.nu))
    for geom in np.unique(pairs):
        body = model.geom_bodyid[geom]
        radius = float(model.geom_rbound[geom] + np.linalg.norm(model.geom_pos[geom]))
        while body:
            for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
                index = np.flatnonzero(joints == j)
                if len(index):
                    arm[geom, index] = radius + np.linalg.norm(model.jnt_pos[j])
            radius += np.linalg.norm(model.body_pos[body])
            body = model.body_parentid[body]
    a, b = (arm[pairs[:, i]] for i in (0, 1))
    return np.where((a > 0) & (b > 0), 0.0, a + b)


def clearance(model, recorded=None):
    data, ids = fresh(model), actuator_map(model)
    pairs, geoms = physical_pairs(model)
    a, b = pairs.T
    moving = model.geom_bodyid[a] != model.geom_bodyid[b]
    sphere = model.geom_rbound[a] + model.geom_rbound[b]
    arms = swept_radii(model, pairs)
    fixed = np.flatnonzero(~moving)
    threshold = 0.025
    cache = {}

    def gaps(target):
        data.qpos[:] = pose(model, target)
        mujoco.mj_kinematics(model, data)
        lower = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1) - sphere - 1e-12
        # state=0 valid lower bound, 1 native finite distance, 2 unresolved near zero.
        mode = np.zeros(len(pairs), dtype=np.int8)
        for k in np.flatnonzero(lower < threshold):
            if k in cache:
                lower[k], mode[k] = cache[k]
                continue
            result = clearance_distance(
                model, data, a[k], b[k], required_clearance_m=MARGIN, search_distance_m=threshold
            )
            lower[k] = result["min_distance_m"]
            mode[k] = 0 if result["distance_is_lower_bound"] else (2 if lower[k] == 0.0 else 1)
            if not moving[k]:
                cache[k] = (lower[k], mode[k])
        return lower, mode

    bounds = model.jnt_range[model.actuator_trnid[:, 0]]
    case_rows = []
    witnesses = {}
    rng = np.random.default_rng(60)
    # Same 12 leg joint combinations as M1. Sample appendages separately
    # so adding a 13th/14th coordinate cannot change those leg cases.
    leg_samples = rng.uniform(bounds[:12, 0], bounds[:12, 1], (512, 12))
    appendages = np.random.default_rng(61).uniform(bounds[12:, 0], bounds[12:, 1], (512, len(bounds) - 12))
    family = {
        "combined_joint_box": np.column_stack((leg_samples, appendages)),
        "commanded_transfer_turn_path": [command(model, ids, "turn_probe", t) for t in np.linspace(0, 5, 126)],
        "commanded_crouch_path": [command(model, ids, "crouch", t) for t in np.linspace(0, 4, 51)],
    }
    family.update(recorded or {})
    for name, commands in family.items():
        passes = inconclusive = failed = 0
        pair_min = np.full(len(pairs), np.inf)
        prev = prev_target = None
        for idx, target in enumerate(commands):
            lower, mode = gaps(target)
            below = lower < MARGIN - 1e-9
            failing = below & (mode == 1)
            failed += int(failing.sum())
            inconclusive += int((below & ~failing).sum())
            passes += int((~below).sum())
            for k in np.flatnonzero(below):
                key = (name, int(k))
                reference = witnesses.get(key)
                status = "failure" if failing[k] else "inconclusive"
                if (
                    reference is None
                    or (failing[k] and reference["status"] != "failure")
                    or (status == reference["status"] and lower[k] < reference["distance_m"])
                ):
                    witnesses[key] = dict(
                        pair=[model.geom(a[k]).name, model.geom(b[k]).name],
                        family=name,
                        sample=idx,
                        status=status,
                        distance_m=float(lower[k]),
                    )
            if prev is not None and name.endswith("_path"):
                sweep = 0.5 * np.sum(arms * np.abs(target - prev_target), axis=1)
                pair_min = np.minimum(pair_min, np.minimum(prev, lower) - sweep)
            prev, prev_target = lower, target
        record = dict(
            family=name,
            samples=len(commands),
            pair_sample_passes=passes,
            pair_sample_failures=failed,
            pair_sample_inconclusive=inconclusive,
        )
        if name.endswith("_path"):
            record["pairs_with_3mm_lower_bound_through_all_linear_intervals"] = int(np.sum(pair_min >= MARGIN))
            record["pairs_without_interval_certificate"] = int(np.sum(pair_min < MARGIN))
        case_rows.append(record)
    return dict(
        margin_m=MARGIN,
        pair_count=len(pairs),
        fixed_pair_count=len(fixed),
        geoms=geoms,
        families=case_rows,
        failing_or_unresolved_witnesses=list(witnesses.values()),
    )


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    files = [
        Path(__file__),
        HERE / "design.json",
        HERE / "build.py",
        ROOT / "environments/shared/robot_clearance.py",
        ROOT / "environments/shared/robot_floor.py",
        ROOT / "environments/shared/intermediate_drive.py",
        ROOT / "environments/shared/robot_building.py",
        HERE.parent / "components.json",
        HERE.parent / "model.xml",
        HERE.parent / "build.py",
        ROOT / "hardware/tyrannosaurus_rex_60cm/m01_skeleton/design.json",
    ]
    for variant in VARIANTS:
        files.extend((HERE / variant / f for f in ("model.xml", "components.json")))
    return {p.relative_to(ROOT).as_posix(): hash_file(p) for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--video", action="store_true", help="Write the deterministic nominal hard-floor support video")
    parser.add_argument("--strict", action="store_true", help="Return failure when any mechanical gate fails")
    parser.add_argument("--verify-sources", action="store_true", help="Verify committed report digests, no rerun")
    args = parser.parse_args()
    initial = sources()
    if args.verify_sources:
        recorded = json.loads(args.output.read_text())
        if recorded["source_sha256"] != initial:
            raise SystemExit("Source or model changed; rerun mechanical screens")
        for base, key in ((ROOT, "renderer_sha256"), (args.output.parent, "video_sha256")):
            for name, expected in recorded.get(key, {}).items():
                if hash_file(base / name) != expected:
                    raise SystemExit(f"Recorded {key} differs for {name}")
        return
    report = dict(
        schema="intermediate-mechanical-screens/v1",
        source_sha256=initial,
        versions={"mujoco": mujoco.__version__, "numpy": np.__version__},
        status="engineering_screen",
        surfaces=FLOORS,
        criteria="mechanical_checks.md",
        variants={},
    )
    failed = False
    for variant in VARIANTS:
        path = HERE / variant / "model.xml"

        def load():
            return mujoco.MjModel.from_xml_path(str(path))

        model = load()
        print(f"{variant}: evaluating support, dynamics and swept clearance", flush=True)
        static = single_support(model)
        if variant == "nominal":
            # Hold every installed component, controller and body dimension fixed
            # while restoring the initial common hip/ankle roll travel envelope.
            clipped = load()
            for side in SIDES:
                for joint in ("hip_roll", "ankle_roll"):
                    clipped.jnt_range[clipped.joint(f"{side}_{joint}").id] = (-0.25, 0.25)
            report["nominal_with_m1_roll_limits_static"] = single_support(clipped)
        cases = [(c, f, 0.0, 22.2) for f in FLOORS for c in ("stand", "crouch", "transfer", "turn_probe")]
        cases += [(c, s, 0.0, 22.2) for c in ("hold_left", "hold_right") for s in FLOORS]
        cases += [("turn_probe", "carpet", 0.15, 20.0), ("hold_left", "carpet", 0.15, 20.0)]
        trials, recorded = [], {}
        for case in cases:
            result, trace = dynamics(load(), *case)
            trials.append(result)
            if (case[0] == "turn_probe" and (case[1] == "hard" or case[2] > 0)) or (
                case[0] == "hold_left" and case[1] == "hard"
            ):
                axes = model.jnt_qposadr[model.actuator_trnid[:, 0]]
                recorded[f"recorded_{case[0]}_{case[1]}_head{case[2]:g}kg"] = np.array(trace)[:, axes]
                if args.video and variant == "nominal" and case[0] == "hold_left":
                    from .gait_video import write_video

                    result_label = {
                        "seed": "deterministic",
                        "passed": result["status"] == "pass",
                        "failed_checks": [k for k, v in result["checks"].items() if not v],
                    }
                    video = args.output.with_name("nominal_single_support.mp4")
                    write_video(
                        video,
                        model,
                        trace,
                        10 * model.opt.timestep,
                        "hard floor",
                        result_label,
                        title="Nominal single support / scripted bench",
                    )
                    report["video_sha256"] = {video.name: hash_file(video)}
                    report["renderer_sha256"] = {
                        p.relative_to(ROOT).as_posix(): hash_file(p)
                        for p in (HERE / "gait_video.py", ROOT / "environments/shared/robot_rendering.py")
                    }
        packing = clearance(model, recorded)
        failed |= any(not s["geometric_and_peak_torque_feasible"] for s in static)
        failed |= any(t["status"] == "fail" for t in trials) or bool(packing["failing_or_unresolved_witnesses"])
        report["variants"][variant] = dict(
            axes=list(actuator_map(model)), static_single_support=static, free_root_trials=trials, clearance=packing
        )
    report["overall_gate"] = "fail" if failed else "pass"
    if initial != sources():
        raise SystemExit("A source changed during evaluation; rerun")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}; overall mechanical release gate: {'fail' if failed else 'pass'}")
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
