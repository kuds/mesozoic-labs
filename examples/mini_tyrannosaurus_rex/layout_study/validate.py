"""Comparable finite-pose packing and unsupported standing screens.

This does not certify shell clearances, fabrication or walking. In particular,
the visual ellipsoid is deliberately not a hidden solid support contact.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from typing import Any

import mujoco
import numpy as np

from environments.shared.robot_clearance import clearance_distance
from environments.shared.robot_rendering import geometry
from examples.mini_tyrannosaurus_rex.layout_study.build import HERE, LAYOUTS, PAYLOADS

REPO = HERE.parents[2]
MARGIN = 0.002


def load(name: str) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, Any]]:
    model = mujoco.MjModel.from_xml_path(str(HERE / name / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, json.loads((HERE / name / "layout.json").read_text())


def pose_set(model: mujoco.MjModel, random_count: int = 512) -> list[tuple[str, np.ndarray]]:
    home = model.key_qpos[0].copy()
    crouch = home.copy()
    for side in ("left", "right"):
        for suffix, angle in (("hip_pitch", -0.35), ("knee", 0.7), ("ankle", -0.35)):
            crouch[model.jnt_qposadr[model.joint(f"{side}_{suffix}").id]] = angle
    poses = [("home", home), ("flat_feet_crouch_joint_configuration", crouch)]
    joints = list(range(1, model.njnt))
    for j in joints:
        for t in np.linspace(0, 1, 5):
            q = home.copy()
            q[model.jnt_qposadr[j]] = (1 - t) * model.jnt_range[j, 0] + t * model.jnt_range[j, 1]
            poses.append((f"individual/{model.joint(j).name}/{t:.2f}", q))
    for side in ("left", "right"):
        group = [j for j in joints if model.joint(j).name.startswith(side)]
        for index, samples in enumerate(itertools.product((0, 0.5, 1), repeat=5)):
            q = home.copy()
            for j, t in zip(group, samples, strict=True):
                q[model.jnt_qposadr[j]] = (1 - t) * model.jnt_range[j, 0] + t * model.jnt_range[j, 1]
            poses.append((f"{side}_cartesian/{index}", q))
    rng = np.random.default_rng(635648)
    for i in range(random_count):
        q = home.copy()
        for j in joints:
            q[model.jnt_qposadr[j]] = rng.uniform(*model.jnt_range[j])
        poses.append((f"simultaneous_random/{i}", q))
    return poses


def packing(model: mujoco.MjModel, data: mujoco.MjData, random_count: int = 512) -> dict[str, Any]:
    ids = [
        g for g in range(model.ngeom) if model.geom(g).name.endswith(("_servo_case", "_reserve", "_flat_pad_collision"))
    ]
    # Check fixed/fixed pairs once and every moving pair at every pose; no body
    # ancestry/contact masks exclude hardware pairs. Own raw-vs-reserve is absent.
    pairs = list(itertools.combinations(ids, 2))
    torso = model.body("torso").id
    static = [(a, b) for a, b in pairs if model.geom_bodyid[a] == torso and model.geom_bodyid[b] == torso]
    moving = [pair for pair in pairs if pair not in static]
    worst: dict[tuple[int, int], dict[str, Any]] = {}
    checks, failures, inconclusive = 0, 0, 0
    for index, (label, q) in enumerate(pose_set(model, random_count)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for a, b in pairs if index == 0 else moving:
            evidence = clearance_distance(model, data, a, b, search_distance_m=0.02, required_clearance_m=MARGIN)
            distance = evidence["min_distance_m"]
            status = (
                "pass"
                if distance >= MARGIN - 1e-9
                else "incomplete"
                if evidence["distance_is_lower_bound"] or distance == 0
                else "fail"
            )
            checks += 1
            failures += status == "fail"
            inconclusive += status == "incomplete"
            if (a, b) not in worst or distance < worst[a, b]["min_distance_m"]:
                worst[a, b] = dict(
                    pair=[model.geom(a).name, model.geom(b).name],
                    pose=label,
                    qpos=q.tolist(),
                    status=status,
                    **evidence,
                )
    witnesses = sorted(worst.values(), key=lambda x: x["min_distance_m"])
    return dict(
        status="fail" if failures else "incomplete" if inconclusive else "pass",
        margin_mm=2,
        pose_count=len(pose_set(model, random_count)),
        pair_count=len(pairs),
        evaluations=checks,
        failed_evaluations=failures,
        inconclusive_evaluations=inconclusive,
        seed=635648,
        static_pairs_checked_once=len(static),
        closest_pairs=witnesses[:12],
        scope="All 10 exact motor cases, 9 installed electronics reserves, 2 sole boxes; all their pair combinations",
        omitted="Bracket/shaft mounting interfaces, moving cables, future modules and the proposed outer shell",
    )


def home_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    torso = model.body("torso").id
    com = data.subtree_com[torso].copy()
    feet = [model.geom(f"{s}_flat_pad_collision").id for s in ("left", "right")]
    foot_bounds = [(data.geom_xpos[g] - model.geom_size[g], data.geom_xpos[g] + model.geom_size[g]) for g in feet]
    lo = np.min([b[0] for b in foot_bounds], axis=0)
    hi = np.max([b[1] for b in foot_bounds], axis=0)
    corners = geometry(model, data, groups=(1, 2))[0].reshape(-1, 3)
    naked = model.body_mass.sum() - sum(p[1] for p in PAYLOADS) / 1000
    cases = [g for g in range(model.ngeom) if model.geom(g).name.endswith("_servo_case")]
    dimensions = [np.sort(model.geom_size[g] * 2000).tolist() for g in cases]
    foot_home_z = float(data.geom_xpos[feet[0], 2])
    data.qpos[:] = pose_set(model, 0)[1][1]
    mujoco.mj_forward(model, data)
    crouch_drop = float(data.geom_xpos[feet[0], 2]) - foot_home_z
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return dict(
        loaded_mass_g=float(model.body_mass.sum() * 1000),
        core_mass_g=float(naked * 1000),
        head_tail_arms_budget_g=sum(p[1] for p in PAYLOADS),
        motor_count=len(cases),
        all_cases_full_size=all(np.allclose(d, [20, 26, 34]) for d in dimensions),
        core_extent_xyz_mm=((corners.max(axis=0) - corners.min(axis=0)) * 1000).tolist(),
        core_top_above_floor_mm=float(corners[:, 2].max() * 1000),
        foot_center_spacing_mm=float(abs(data.geom_xpos[feet[0], 1] - data.geom_xpos[feet[1], 1]) * 1000),
        loaded_com_xyz_mm=(com * 1000).tolist(),
        support_projection_margin_mm=float(min(*(com[:2] - lo[:2]), *(hi[:2] - com[:2])) * 1000),
        margin_basis="Home double-support convex hull only; not a balance controller or single-support margin",
        symmetric_flat_feet_crouch_drop_mm=crouch_drop * 1000,
        crouch_basis="Kinematic endpoint at hip=-0.35, knee=0.70, ankle=-0.35 rad; root would lower by this amount; not a dynamic crouch trial",
    )


def stand(model: mujoco.MjModel, duration: float = 30, loaded: bool = True) -> dict[str, Any]:
    # Each call receives an independent model so unloaded mass edits cannot leak.
    if not loaded:
        for name, _, _, _ in PAYLOADS:
            body = model.body(name).id
            model.body_mass[body] = 0
            model.body_inertia[body] = 0
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    start = data.qpos[:3].copy()
    torso = model.body("torso").id
    floor = model.geom("floor").id
    feet = {model.geom(f"{s}_flat_pad_collision").id for s in ("left", "right")}
    max_tilt, max_drift, min_height, max_penetration = 0.0, 0.0, float(start[2]), 0.0
    unexpected = set()
    torque_samples = []
    for _ in range(round(duration / model.opt.timestep)):
        data.ctrl[:] = 0
        mujoco.mj_step(model, data)
        tilt = float(np.arccos(np.clip(data.xmat[torso, 8], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        max_drift = max(max_drift, float(np.linalg.norm(data.qpos[:2] - start[:2])))
        min_height = min(min_height, float(data.qpos[2]))
        for contact in data.contact:
            if floor in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == floor else contact.geom1
                if other not in feet:
                    unexpected.add(model.geom(other).name)
            else:
                max_penetration = max(max_penetration, float(-contact.dist))
        if data.time >= 2:
            torque_samples.append(data.actuator_force.copy())
        if not np.isfinite(data.qpos).all() or tilt > np.pi / 3 or data.qpos[2] < 0.06:
            break
    torques = np.array(torque_samples)
    complete = data.time >= duration - model.opt.timestep
    unsupported = model.neq == 0 and model.nmocap == 0 and not np.any(model.body_gravcomp)
    passed = (
        unsupported
        and complete
        and max_tilt < np.deg2rad(10)
        and max_drift < 0.01
        and min_height > 0.099
        and not unexpected
        and max_penetration < 0.001
    )
    return dict(
        status="pass" if passed else "fail",
        loaded=loaded,
        requested_duration_s=duration,
        completed_duration_s=float(data.time),
        torque_cap_nm=0.1,
        max_tilt_deg=float(np.rad2deg(max_tilt)),
        max_horizontal_drift_mm=max_drift * 1000,
        min_root_height_mm=min_height * 1000,
        unexpected_ground_contact=sorted(unexpected),
        max_self_penetration_mm=max_penetration * 1000,
        max_settled_joint_rms_nm=float(np.max(np.sqrt(np.mean(torques**2, axis=0)))) if len(torques) else None,
        no_support_fixtures=unsupported,
        scope="Ideal position servos at 0.10 Nm; open chassis, fixed future mass surrogates; no walking claim",
    )


def run(name: str, random_count: int = 512, seconds: float = 30) -> dict[str, Any]:
    model, data, layout = load(name)
    result = dict(
        name=name,
        label=layout["label"],
        home=home_metrics(model, data),
        packing=packing(model, data, random_count),
        standing=[stand(load(name)[0], seconds, loaded) for loaded in (False, True)],
        silhouette_dimensions_mm=layout["silhouette_dimensions_mm"],
        overall_status="incomplete",
        hardware_ready=False,
        reason="Shell, mounts, moving cables, thermal/electrical qualification and trained locomotion remain open",
    )
    if result["packing"]["status"] == "fail" or any(s["status"] == "fail" for s in result["standing"]):
        result["overall_status"] = "fail"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-count", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    results = []
    for name in LAYOUTS:
        result = run(name, args.random_count, args.seconds)
        results.append(result)
        print(
            name,
            "packing",
            result["packing"]["status"],
            "standing",
            [r["status"] for r in result["standing"]],
            flush=True,
        )
    inputs = [HERE / f for f in ("build.py", "validate.py", "render.py")] + [HERE.parent / "model.xml"]
    inputs += [
        REPO / "environments/shared" / f for f in ("robot_building.py", "robot_clearance.py", "robot_rendering.py")
    ]
    inputs += [HERE / name / f for name in LAYOUTS for f in ("model.xml", "layout.json")]
    known_failure = any(r["overall_status"] == "fail" for r in results)
    report = dict(
        status="fail" if known_failure else "incomplete",
        invocation=f"python -m examples.mini_tyrannosaurus_rex.layout_study.validate --random-count {args.random_count} --seconds {args.seconds}",
        python=platform.python_version(),
        mujoco=mujoco.__version__,
        numpy=np.__version__,
        layouts=results,
        sources={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.exists()},
    )
    (HERE / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(
        1
        if any(r["packing"]["status"] == "fail" or any(s["status"] == "fail" for s in r["standing"]) for r in results)
        else 2
    )


if __name__ == "__main__":
    main()
