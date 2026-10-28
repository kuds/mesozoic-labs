"""Matched hardware, finite clearance samples and unsupported motion screens.

The expanded fold scan is kinematic and intentionally exceeds the delivered
joint limits. Its endpoint is a sampled box-envelope limit, not a qualified
mechanical range. Dynamic trials use the same ideal PD servos in both designs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.robot_clearance import clearance_distance
from examples.mini_tyrannosaurus_rex.layout_study import validate as torso_validate
from examples.mini_tyrannosaurus_rex.leg_study.build import BASE, HERE, PITCH, VARIANTS

REPO = HERE.parents[2]
MARGIN = 0.002


def load(name: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(HERE / name / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def mapped_pose(model: mujoco.MjModel, qpos: np.ndarray, sign: int) -> np.ndarray:
    q = qpos.copy()
    for side in ("left", "right"):
        for suffix in PITCH:
            q[model.jnt_qposadr[model.joint(f"{side}_{suffix}").id]] *= sign
    return q


def hardware_ids(model: mujoco.MjModel) -> list[int]:
    return [
        g for g in range(model.ngeom) if model.geom(g).name.endswith(("_servo_case", "_reserve", "_flat_pad_collision"))
    ]


def distance_status(evidence: dict[str, Any]) -> str:
    distance = evidence["min_distance_m"]
    if distance >= MARGIN - 1e-9:
        return "pass"
    return "incomplete" if evidence["distance_is_lower_bound"] or distance == 0 else "fail"


def evidence_for_pair(model: mujoco.MjModel, data: mujoco.MjData, a: int, b: int) -> dict[str, Any]:
    result = clearance_distance(model, data, a, b, search_distance_m=0.02, required_clearance_m=MARGIN)
    return dict(pair=[model.geom(a).name, model.geom(b).name], status=distance_status(result), **result)


def packing(name: str) -> dict[str, Any]:
    model, data = load(name)
    forward, _ = load("forward")
    poses = torso_validate.pose_set(forward, 512)
    pairs = list(itertools.combinations(hardware_ids(model), 2))
    torso = model.body("torso").id
    moving = [p for p in pairs if not all(model.geom_bodyid[g] == torso for g in p)]
    worst: dict[tuple[int, int], dict[str, Any]] = {}
    counts = {"pass": 0, "fail": 0, "incomplete": 0}
    for index, (label, forward_q) in enumerate(poses):
        data.qpos[:] = mapped_pose(model, forward_q, VARIANTS[name])
        mujoco.mj_forward(model, data)
        for a, b in pairs if index == 0 else moving:
            row = evidence_for_pair(model, data, a, b)
            counts[row["status"]] += 1
            if (a, b) not in worst or row["min_distance_m"] < worst[a, b]["min_distance_m"]:
                worst[a, b] = dict(pose=label, qpos=data.qpos.tolist(), **row)
    return dict(
        status="fail" if counts["fail"] else "incomplete" if counts["incomplete"] else "pass",
        margin_mm=2,
        pose_count=len(poses),
        evaluations=sum(counts.values()),
        counts=counts,
        pair_count=len(pairs),
        static_pairs_checked_once=len(pairs) - len(moving),
        seed=635648,
        closest_pairs=sorted(worst.values(), key=lambda r: r["min_distance_m"])[:8],
        scope="Identical matched poses; 10 cases, 9 installation boxes, 2 soles; all pair combinations regardless of contact masks",
        omitted="Detailed brackets, horns, moving cables, plug asymmetry, outer shell and future head/tail geometry; no terrain claim for arbitrary poses",
    )


def set_fold(model: mujoco.MjModel, data: mujoco.MjData, sign: int, alpha: float, ground: bool = True) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for side in ("left", "right"):
        for suffix, angle in zip(PITCH, (-alpha, 2 * alpha, -alpha), strict=True):
            data.qpos[model.jnt_qposadr[model.joint(f"{side}_{suffix}").id]] = sign * angle
    mujoco.mj_forward(model, data)
    if ground:
        foot = model.geom("left_flat_pad_collision").id
        # At this symmetric pose the sole is horizontal. Ground both soles
        # rather than pretending that the floating-root pose is a crouch.
        data.qpos[2] -= data.geom_xpos[foot, 2] - model.geom_size[foot, 2]
        mujoco.mj_forward(model, data)


def fold_snapshot(model: mujoco.MjModel, data: mujoco.MjData, sign: int, alpha: float) -> dict[str, Any]:
    set_fold(model, data, sign, alpha)
    ids = hardware_ids(model)
    rows = [evidence_for_pair(model, data, a, b) for a, b in itertools.combinations(ids, 2)]
    # An exact support height for each box; soles are intended to touch floor.
    for g in ids:
        if model.geom(g).name.endswith("_flat_pad_collision"):
            continue
        rotation = data.geom_xmat[g].reshape(3, 3)
        bottom = float(data.geom_xpos[g, 2] - np.abs(rotation[2]) @ model.geom_size[g])
        rows.append(
            dict(
                pair=[model.geom(g).name, "floor"], min_distance_m=bottom, status="pass" if bottom >= MARGIN else "fail"
            )
        )
    bad = [r for r in rows if r["status"] != "pass"]
    moving = [r for r in rows if not all(n.startswith("part_") or "hip_yaw_servo_case" in n for n in r["pair"])]
    angles = data.qpos[7:]
    inside = bool(np.all(angles >= model.jnt_range[1:, 0] - 1e-9) and np.all(angles <= model.jnt_range[1:, 1] + 1e-9))
    return dict(
        alpha_rad=float(alpha),
        hip_deg=float(np.rad2deg(-sign * alpha)),
        knee_deg=float(np.rad2deg(2 * sign * alpha)),
        ankle_deg=float(np.rad2deg(-sign * alpha)),
        drop_mm=float((model.key_qpos[0, 2] - data.qpos[2]) * 1000),
        knee_offset_x_mm=float(
            (data.xpos[model.body("left_shin").id, 0] - data.xpos[model.body("left_thigh").id, 0]) * 1000
        ),
        root_height_mm=float(data.qpos[2] * 1000),
        within_delivered_joint_limits=inside,
        status="fail" if any(r["status"] == "fail" for r in rows) else "incomplete" if bad else "pass",
        hardware_pair_evaluations=len(rows),
        closest=min(rows, key=lambda r: r["min_distance_m"]),
        closest_moving=min(moving, key=lambda r: r["min_distance_m"]),
        blockers=sorted(bad, key=lambda r: r["min_distance_m"])[:4],
        qpos=data.qpos.tolist(),
    )


def alpha_for_drop(drop_mm: float) -> float:
    # Equal 43.66 mm links: vertical hip-to-ankle span is
    # 2 * (42.5 cos(alpha) - 10 sin(alpha)) mm.
    return float(np.arccos((85 - drop_mm) / (2 * np.hypot(42.5, 10))) - np.arctan2(10, 42.5))


def fold_scan(name: str) -> dict[str, Any]:
    model, data = load(name)
    sign = VARIANTS[name]
    samples = []
    for alpha in np.arange(0, 1.20001, 0.005):
        row = fold_snapshot(model, data, sign, float(alpha))
        samples.append(row)
        if row["status"] != "pass":
            break
    return dict(
        current=fold_snapshot(model, data, sign, 0.35),
        proposed_30mm=fold_snapshot(model, data, sign, alpha_for_drop(30)),
        last_clear_sample=samples[-2] if samples[-1]["status"] != "pass" else samples[-1],
        first_nonpassing_sample=samples[-1] if samples[-1]["status"] != "pass" else None,
        resolution_rad=0.005,
        sampled_poses=len(samples),
        samples=[{k: row[k] for k in ("alpha_rad", "drop_mm", "status")} for row in samples],
        scope="Symmetric flat-foot fold only. Larger angles deliberately exceed delivered limits. 2 mm boxes-plus-floor screen, not a usable mechanical range",
    )


def workspace(name: str) -> dict[str, Any]:
    """Kinematic ankle reach; neither single-support balance nor gait testing."""
    model, data = load(name)
    forward, _ = load("forward")
    sign = VARIANTS[name]
    hip_range = forward.jnt_range[forward.joint("left_hip_pitch").id]
    knee_range = forward.jnt_range[forward.joint("left_knee").id]
    points = []
    for hip, knee in itertools.product(
        np.linspace(hip_range[0], hip_range[1], 61), np.linspace(knee_range[0], knee_range[1], 71)
    ):
        ankle = -hip - knee
        if not -0.45 - 1e-9 <= ankle <= 0.45 + 1e-9:
            continue
        mujoco.mj_resetDataKeyframe(model, data, 0)
        for suffix, angle in zip(PITCH, (hip, knee, ankle), strict=True):
            data.qpos[model.jnt_qposadr[model.joint(f"left_{suffix}").id]] = sign * angle
        mujoco.mj_forward(model, data)
        p = data.xpos[model.body("left_foot").id]
        points.append([round(float(p[0] * 1000), 6), round(float((p[2] - 0.028) * 1000), 6)])
    pts = np.array(points)
    return dict(
        samples=len(points),
        max_forward_ankle_offset_mm=float(pts[:, 0].max()),
        max_rearward_ankle_offset_mm=float(-pts[:, 0].min()),
        max_ankle_lift_mm=float(pts[:, 1].max()),
        ankle_x_lift_mm=points,
        scope="Flat sole, current joint ranges, pelvis fixed; reach only, NOT collision-qualified footsteps or demonstrated walking",
    )


def motion(name: str, drop_mm: float, loaded: bool = True) -> dict[str, Any]:
    model, data = load(name)
    sign = VARIANTS[name]
    alpha = alpha_for_drop(drop_mm)
    expanded = alpha > 0.35 + 1e-9
    if not loaded:
        for label, _, _, _ in torso_validate.PAYLOADS:
            model.body_mass[model.body(label).id] = 0
            model.body_inertia[model.body(label).id] = 0
    target = np.zeros(model.nu)
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        suffix = model.joint(j).name.split("_", 1)[1]
        if suffix in PITCH:
            target[a] = sign * dict(zip(PITCH, (-alpha, 2 * alpha, -alpha), strict=True))[suffix]
        if expanded:
            model.jnt_range[j, 0] = min(model.jnt_range[j, 0], target[a])
            model.jnt_range[j, 1] = max(model.jnt_range[j, 1], target[a])
            model.actuator_ctrlrange[a] = model.jnt_range[j]
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    start = data.qpos[:3].copy()
    torso, floor = model.body("torso").id, model.geom("floor").id
    feet = [model.geom(f"{s}_flat_pad_collision").id for s in ("left", "right")]
    initial_feet = data.geom_xpos[feet, :2].copy()
    max_tilt = max_drift = max_slip = max_penetration = 0.0
    min_height = float(start[2])
    unexpected: set[str] = set()
    hold_heights, hold_errors, torques, records = [], [], [], []
    qindices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    for step in range(round(14 / model.opt.timestep)):
        t = data.time
        phase = np.clip((t - 2) / 4, 0, 1) if t < 8 else 1 - np.clip((t - 8) / 4, 0, 1)
        smooth = phase**3 * (10 - 15 * phase + 6 * phase**2)
        data.ctrl[:] = smooth * target
        mujoco.mj_step(model, data)
        # Synchronize kinematic fields with post-integration qpos for metrics.
        mujoco.mj_forward(model, data)
        tilt = float(np.rad2deg(np.arccos(np.clip(data.xmat[torso, 8], -1, 1))))
        drift = float(np.linalg.norm(data.qpos[:2] - start[:2]))
        slip = float(np.max(np.linalg.norm(data.geom_xpos[feet, :2] - initial_feet, axis=1)))
        max_tilt, max_drift, max_slip = max(max_tilt, tilt), max(max_drift, drift), max(max_slip, slip)
        min_height = min(min_height, float(data.qpos[2]))
        for contact in data.contact:
            if floor in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == floor else contact.geom1
                if other not in feet:
                    unexpected.add(model.geom(other).name)
            else:
                max_penetration = max(max_penetration, float(-contact.dist))
        if 6.5 <= data.time < 8:
            hold_heights.append(float(data.qpos[2]))
            hold_errors.append(float(np.max(np.abs(data.qpos[qindices] - target))))
        if data.time >= 2:
            torques.append(data.actuator_force.copy())
        if step % 50 == 0:
            records.append(dict(t_s=float(data.time), qpos=data.qpos.tolist(), requested_fraction=float(smooth)))
        if not np.isfinite(data.qpos).all() or tilt > 60 or data.qpos[2] < 0.04:
            break
    records.append(dict(t_s=float(data.time), qpos=data.qpos.tolist(), requested_fraction=float(smooth)))
    rms = np.sqrt(np.mean(np.asarray(torques) ** 2, axis=0)) if torques else np.full(model.nu, np.nan)
    achieved = float((start[2] - np.mean(hold_heights)) * 1000) if hold_heights else None
    error = max(hold_errors) if hold_errors else None
    unsupported = model.neq == 0 and model.nmocap == 0 and not np.any(model.body_gravcomp)
    checks = dict(
        no_support_fixtures=unsupported,
        completed_14_seconds=data.time >= 14 - model.opt.timestep,
        tilt_under_10_deg=max_tilt < 10,
        horizontal_drift_under_10mm=max_drift < 0.01,
        sole_slip_under_5mm=max_slip < 0.005,
        no_unexpected_floor_contact=not unexpected,
        self_penetration_under_1mm=max_penetration < 0.001,
        hold_drop_within_3mm=achieved is not None and abs(achieved - drop_mm) < 3,
        hold_joint_error_under_0_07rad=error is not None and error < 0.07,
        returned_height_within_3mm=bool(abs(data.qpos[2] - start[2]) < 0.003),
    )
    return dict(
        status="pass" if all(checks.values()) else "fail",
        loaded=loaded,
        requested_drop_mm=drop_mm,
        expanded_limits_for_experiment=expanded,
        checks=checks,
        achieved_mean_hold_drop_mm=achieved,
        max_hold_joint_error_rad=error,
        completed_duration_s=float(data.time),
        max_tilt_deg=max_tilt,
        max_horizontal_drift_mm=max_drift * 1000,
        max_sole_slip_mm=max_slip * 1000,
        max_self_penetration_mm=max_penetration * 1000,
        unexpected_ground_contact=sorted(unexpected),
        min_root_height_mm=min_height * 1000,
        final_root_height_mm=float(data.qpos[2] * 1000),
        joint_rms_nm={model.actuator(a).name: float(rms[a]) for a in range(model.nu)},
        joint_peak_abs_nm={model.actuator(a).name: float(np.max(np.abs(torques), axis=0)[a]) for a in range(model.nu)}
        if torques
        else {},
        joint_saturated_time_fraction={
            model.actuator(a).name: float(np.mean(np.abs(torques) >= 0.0999, axis=0)[a]) for a in range(model.nu)
        }
        if torques
        else {},
        torque_cap_nm=0.10,
        saturated_joint_time_fraction=float(np.mean(np.abs(torques) >= 0.0999)) if torques else None,
        trajectory=records,
        scope="Same ideal PD, gains, 0.10 Nm cap, 2s settle / 4s lower / 2s hold / 4s rise / 2s settle; free root, no balance controller or trained policy",
    )


def run(name: str) -> dict[str, Any]:
    model, data = load(name)
    mujoco.mj_forward(model, data)
    result = dict(
        name=name,
        loaded_mass_g=float(model.body_mass.sum() * 1000),
        motor_count=len([g for g in range(model.ngeom) if model.geom(g).name.endswith("_servo_case")]),
        loaded_com_mm=(data.subtree_com[model.body("torso").id] * 1000).tolist(),
        packing=packing(name),
        fold=fold_scan(name),
        workspace=workspace(name),
        standing=[torso_validate.stand(load(name)[0], 30, loaded) for loaded in (False, True)],
        crouching=[motion(name, drop, loaded) for loaded in (False, True) for drop in (12.0112755570818, 30)],
        hardware_ready=False,
    )
    print(
        name, "packing", result["packing"]["status"], "crouch", [r["status"] for r in result["crouching"]], flush=True
    )
    return result


def source_paths() -> list[Path]:
    files = [HERE / f for f in ("build.py", "validate.py", "render.py")]
    files += [HERE / name / f for name in VARIANTS for f in ("model.xml", "variant.json")]
    files += [BASE, BASE.with_name("layout.json")]
    files += [BASE.parents[1] / f for f in ("build.py", "validate.py")]
    files += [
        REPO / "environments/shared" / f for f in ("robot_clearance.py", "robot_building.py", "robot_rendering.py")
    ]
    return files


def main() -> None:
    results = [run(name) for name in VARIANTS]
    # Dense trajectories and reach samples stay in a compact, readable JSON
    # companion instead of flooding the summary with thousands of scalar rows.
    samples = {}
    for result in results:
        name = result["name"]
        samples[name] = dict(
            workspace_mm=result["workspace"].pop("ankle_x_lift_mm"),
            trajectories=[trial.pop("trajectory") for trial in result["crouching"]],
        )
    (HERE / "samples.json").write_text(json.dumps(samples, separators=(",", ":")) + "\n")
    failure = any(
        r["packing"]["status"] != "pass" or any(s["status"] != "pass" for s in r["standing"] + r["crouching"])
        for r in results
    )
    report = dict(
        status="fail" if failure else "incomplete",
        hardware_ready=False,
        reason="Finite engineering screens only; brackets, shell, cables, identified servos and trained behavior remain unqualified",
        invocation="python -m examples.mini_tyrannosaurus_rex.leg_study.validate",
        python=platform.python_version(),
        mujoco=mujoco.__version__,
        numpy=np.__version__,
        variants=results,
        sources={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()},
        samples_sha256=hashlib.sha256((HERE / "samples.json").read_bytes()).hexdigest(),
    )
    (HERE / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    # A failed expanded experiment stays visible in the CLI exit status too.
    raise SystemExit(1 if failure else 2)


if __name__ == "__main__":
    main()
