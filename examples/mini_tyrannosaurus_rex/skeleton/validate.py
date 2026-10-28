"""Finite-pose clearance and fixed-command dynamics for the actual M1 plant."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.robot_clearance import clearance_distance
from environments.tyrannosaurus_rex.robot.env import TyrannosaurusRexRobotEnv
from examples.mini_tyrannosaurus_rex.layout_study.validate import pose_set
from examples.mini_tyrannosaurus_rex.skeleton.build import OUTPUT, ROOT

DESTINATION = ROOT / "hardware/mini_tyrannosaurus_rex/m01_skeleton/validation.json"


def packing() -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    data = mujoco.MjData(model)
    ids = [
        g for g in range(model.ngeom) if model.geom(g).name.endswith(("_servo_case", "_reserve", "_flat_pad_collision"))
    ]
    pairs = list(itertools.combinations(ids, 2))
    pelvis = model.body("pelvis").id
    moving = [p for p in pairs if any(model.geom_bodyid[g] != pelvis for g in p)]
    worst: dict[tuple[int, int], dict[str, Any]] = {}
    evaluations = failed = incomplete = 0
    poses = pose_set(model)
    for index, (label, qpos) in enumerate(poses):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for a, b in pairs if index == 0 else moving:
            evidence = clearance_distance(model, data, a, b, search_distance_m=0.02, required_clearance_m=0.002)
            distance = evidence["min_distance_m"]
            status = (
                "pass"
                if distance >= 0.002 - 1e-9
                else "incomplete"
                if evidence["distance_is_lower_bound"] or distance == 0
                else "fail"
            )
            evaluations += 1
            failed += status == "fail"
            incomplete += status == "incomplete"
            if (a, b) not in worst or distance < worst[a, b]["min_distance_m"]:
                worst[a, b] = dict(
                    pair=[model.geom(a).name, model.geom(b).name],
                    pose=label,
                    qpos=qpos.tolist(),
                    status=status,
                    **evidence,
                )
    return dict(
        status="fail" if failed else "incomplete" if incomplete else "pass",
        margin_mm=2,
        pose_count=len(poses),
        pair_count=len(pairs),
        evaluations=evaluations,
        failed_evaluations=failed,
        incomplete_evaluations=incomplete,
        closest_pairs=sorted(worst.values(), key=lambda row: row["min_distance_m"])[:12],
        scope="10 full motor boxes, 9 installed electronics reserves, 2 soles; all pairs; 1050 finite poses",
        omitted="Detailed mounts, shafts, moving cables, future modules and ballast fixtures; no CAD qualification",
    )


def trial(seed: int, *, crouch: bool = False) -> dict[str, Any]:
    steps = 700 if crouch else 1500
    env = TyrannosaurusRexRobotEnv(max_episode_steps=steps, reset_noise_scale=0.0 if crouch else 0.005)
    env.reset(seed=seed)
    start = env.data.qpos[:3].copy()
    target = np.zeros(env.model.nu)
    for a in range(env.model.nu):
        name = env.model.joint(env.model.actuator_trnid[a, 0]).name.split("_", 1)[1]
        target[a] = {"hip_pitch": -0.35, "knee": 0.7, "ankle": -0.35}.get(name, 0.0)
    maximum_tilt = maximum_drift = 0.0
    weight_errors, heights, torques = [], [], []
    info: dict[str, Any] = {}
    try:
        for step in range(steps):
            t = step * env.dt
            phase = np.clip((t - 2) / 4, 0, 1) if t < 8 else 1 - np.clip((t - 8) / 4, 0, 1)
            desired = target * phase**3 * (10 - 15 * phase + 6 * phase**2) if crouch else np.zeros(env.model.nu)
            lo, hi = env.model.actuator_ctrlrange.T
            action = np.where(desired < 0, desired / (-lo), desired / hi)
            _, _, terminated, truncated, info = env.step(action)
            maximum_tilt = max(maximum_tilt, float(info["tilt_angle"]))
            maximum_drift = max(maximum_drift, float(np.linalg.norm(env.data.qpos[:2] - start[:2])))
            if t >= 4:
                weight_errors.append(
                    abs(sum(env._foot_contact_forces()) - env.body_mass * 9.81) / (env.body_mass * 9.81)
                )
                torques.append(env.data.actuator_force.copy())
            if 6.5 <= t < 8:
                heights.append(float(env.data.qpos[2]))
            if terminated or truncated:
                break
        checks = dict(
            full_horizon=step + 1 == steps and not terminated,
            tilt_under_10_deg=maximum_tilt < np.deg2rad(10),
            drift_under_10mm=maximum_drift < 0.01,
            torque_cap_preserved=bool(np.max(np.abs(torques)) <= 0.1 + 1e-9),
        )
        if crouch:
            checks["drop_12mm_within_3mm"] = bool(abs((start[2] - np.mean(heights)) * 1000 - 12) < 3)
            checks["returned_within_3mm"] = bool(abs(env.data.qpos[2] - start[2]) < 0.003)
        else:
            checks["mean_foot_force_within_1pct_weight"] = bool(np.mean(weight_errors) < 0.01)
        checks = {name: bool(value) for name, value in checks.items()}
        return dict(
            status="pass" if all(checks.values()) else "fail",
            checks=checks,
            seed=seed,
            controller="fixed home-relative PD targets; not a trained policy",
            requested_seconds=steps * env.dt,
            completed_steps=step + 1,
            termination_reason=info.get("termination_reason"),
            max_tilt_deg=float(np.rad2deg(maximum_tilt)),
            max_drift_mm=maximum_drift * 1000,
            mean_relative_foot_weight_error=float(np.mean(weight_errors)),
            mean_hold_drop_mm=float((start[2] - np.mean(heights)) * 1000) if crouch else None,
            max_joint_rms_nm=float(np.max(np.sqrt(np.mean(np.asarray(torques) ** 2, axis=0)))),
        )
    finally:
        env.close()


def source_paths() -> list[Path]:
    return [
        OUTPUT,
        Path(__file__),
        ROOT / "examples/mini_tyrannosaurus_rex/skeleton/build.py",
        ROOT / "examples/mini_tyrannosaurus_rex/layout_study/validate.py",
        ROOT / "environments/shared/robot_clearance.py",
        ROOT / "environments/shared/base_env.py",
        ROOT / "environments/shared/action_filter.py",
        ROOT / "environments/tyrannosaurus_rex/base_env.py",
        ROOT / "environments/tyrannosaurus_rex/robot/env.py",
    ]


def main() -> None:
    print("Checking the registered plant's 1050 clearance poses", flush=True)
    clearances = packing()
    stands = [trial(seed) for seed in range(9100, 9108)]
    crouch = trial(9200, crouch=True)
    report = dict(
        milestone="m01_skeleton",
        hardware_revision="hw_r01",
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
        invocation="python -m examples.mini_tyrannosaurus_rex.skeleton.validate",
        source_sha256={
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()
        },
        packing=clearances,
        standing=stands,
        crouch=crouch,
        limitations=[
            "No trained walking",
            "No single-support qualification",
            "No physical prototype or actuator identification",
            "Future payload inertia allowances have no qualified collision/mounting geometry",
        ],
    )
    DESTINATION.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    passed = all(row["status"] == "pass" for row in [clearances, *stands, crouch])
    print(
        json.dumps(
            dict(packing=clearances["status"], standing=[row["status"] for row in stands], crouch=crouch["status"])
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
