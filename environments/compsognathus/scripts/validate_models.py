"""Deterministic engineering smoke checks; no walking or robustness certificate.

python -m environments.compsognathus.scripts.validate_models --output /tmp/compso.json
Requires the repository's [test] dependencies for scipy's support-polygon hull.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.model import (
    PARAMETERS,
    floor_contacts,
    load_model,
    model_bounds,
    robot_body_ids,
    set_motors_enabled,
)


def hold_trial(variant: str, *, mass_scale=1.0, seconds=10.0, seed=None, motors_off=False):
    model, data = load_model(variant, mass_scale)
    home_height = float(data.xpos[model.body("pelvis").id, 2])
    if seed is not None:
        rng = np.random.default_rng(seed)
        # Small independent joint-angle perturbations; not a certified reset distribution.
        for j in range(1, model.njnt):
            address = model.jnt_qposadr[j]
            data.qpos[address] += rng.uniform(-np.deg2rad(1), np.deg2rad(1))
            data.qpos[address] = np.clip(data.qpos[address], *model.jnt_range[j])
        mujoco.mj_forward(model, data)
    if motors_off:
        set_motors_enabled(model, False)
    peak_torque = np.zeros(model.nu)
    peak_speed = np.zeros(model.nu)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    minimum_height = home_height
    maximum_tilt = 0.0
    nonfoot_contacts: set[str] = set()
    foot_geoms = (
        {"left_sole", "right_sole"}
        if variant == "robot"
        else {
            f"{side}_{part}"
            for side in ("r", "l")
            for part in ("plantar_pad", "toe_d2_geom", "toe_d3_geom", "toe_d4_geom")
        }
    )
    for step in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        peak_torque = np.maximum(peak_torque, np.abs(data.actuator_force))
        peak_speed = np.maximum(peak_speed, np.abs(data.qvel[dofs]))
        minimum_height = min(minimum_height, float(data.xpos[model.body("pelvis").id, 2]))
        z_alignment = data.xmat[model.body("pelvis").id].reshape(3, 3)[2, 2]
        maximum_tilt = max(maximum_tilt, float(np.degrees(np.arccos(np.clip(z_alignment, -1, 1)))))
        if step % 10 == 0:
            nonfoot_contacts.update(c["geom"] for c in floor_contacts(model, data) if c["geom"] not in foot_geoms)
    mujoco.mj_forward(model, data)
    contacts = floor_contacts(model, data)
    points = np.array([c["position"][:2] for c in contacts if c["normal_force"] > 0.01])
    root = model.body("pelvis").id
    com = data.subtree_com[root].copy()
    margin = None
    if len(points) >= 3 and np.linalg.matrix_rank(points - points.mean(axis=0)) == 2:
        hull = ConvexHull(points)
        margin = float(np.min(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2])))
    mass = float(model.body_mass[robot_body_ids(model)].sum())
    foot_force = sum(float(data.sensor(name).data[0]) for name in ("r_foot_touch", "l_foot_touch"))
    ground_force = sum(c["normal_force"] for c in contacts)
    cap = np.max(np.abs(model.actuator_forcerange), axis=1)
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    return {
        "variant": variant,
        "mass_scale": mass_scale,
        "seed": seed,
        "motors_off": motors_off,
        "seconds": seconds,
        "mass_kg": mass,
        "home_pelvis_height_m": home_height,
        "final_pelvis_height_m": float(data.xpos[root, 2]),
        "minimum_pelvis_height_m": minimum_height,
        "maximum_tilt_deg": maximum_tilt,
        "final_com_m": com.tolist(),
        "final_support_margin_m": margin,
        "foot_sensor_load_N": foot_force,
        "ground_normal_load_N": ground_force,
        "body_weight_N": mass * 9.81,
        "nonfoot_ground_contacts": sorted(nonfoot_contacts),
        "peak_torque_Nm": dict(
            zip([model.actuator(i).name for i in range(model.nu)], peak_torque.tolist(), strict=True)
        ),
        "peak_joint_speed_rad_s": dict(
            zip([model.actuator(i).name for i in range(model.nu)], peak_speed.tolist(), strict=True)
        ),
        "maximum_torque_cap_fraction": float(np.max(peak_torque / cap)),
        "finite": finite,
        "warnings": warnings,
        "stood": bool(
            finite
            and not warnings
            and minimum_height > 0.9 * home_height
            and maximum_tilt < 10
            and not nonfoot_contacts
        ),
    }


def geometry(variant):
    model, data = load_model(variant)
    low, high = model_bounds(model, data)
    result = {
        "nq": model.nq,
        "nv": model.nv,
        "actuators": model.nu,
        "joints_including_root": model.njnt,
        "mass_kg": float(model.body_mass[robot_body_ids(model)].sum()),
        "bounds_min_m": low.tolist(),
        "bounds_max_m": high.tolist(),
        "dimensions_xyz_m": (high - low).tolist(),
        "pelvis_height_m": float(data.xpos[model.body("pelvis").id, 2]),
        "home_com_m": data.subtree_com[model.body("pelvis").id].tolist(),
        "model_sha256": hashlib.sha256(MODEL_PATHS[variant].read_bytes()).hexdigest(),
    }
    if variant == "robot":
        lengths = {}
        for side in ("left", "right"):
            for a, b in (("hip_pitch", "knee"), ("knee", "ankle")):
                j1, j2 = model.joint(f"{side}_{a}").id, model.joint(f"{side}_{b}").id
                lengths[f"{side}_{a}_to_{b}"] = float(
                    np.linalg.norm(np.cross(data.xanchor[j2] - data.xanchor[j1], data.xaxis[j1]))
                )
        result["parallel_shaft_distances_m"] = lengths
        result["sole_size_m"] = (2 * model.geom_size[model.geom("left_sole").id]).tolist()
        result["sole_center_spacing_m"] = float(
            np.linalg.norm(data.site("left_sole").xpos - data.site("right_sole").xpos)
        )
        result["tail_has_joints"] = bool(model.body_jntnum[model.body("passive_tail").id])
    return result


def validate():
    trials = []
    for variant in MODEL_PATHS:
        trials.append(hold_trial(variant))
        for seed in (42, 43, 44):
            trials.append(hold_trial(variant, seed=seed, seconds=5))
        trials.append(hold_trial(variant, motors_off=True, seconds=5))
    trials.append(hold_trial("robot", mass_scale=PARAMETERS["robot"]["stress_mass_scale"]))
    return {
        "schema": "mesozoic.compso-preflight/v0",
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "geometry": {variant: geometry(variant) for variant in MODEL_PATHS},
        "trials": trials,
        "scope": "Flat-floor fixed-target position-servo standing; three small joint-reset perturbations; no gait, training, push recovery, or hardware validation.",
        "passed": all(
            t["stood"] and t["maximum_torque_cap_fraction"] <= 1.000001 for t in trials if not t["motors_off"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "geometry": report["geometry"],
                "trials": [
                    {
                        k: t[k]
                        for k in (
                            "variant",
                            "mass_scale",
                            "seed",
                            "motors_off",
                            "stood",
                            "maximum_tilt_deg",
                            "maximum_torque_cap_fraction",
                        )
                    }
                    for t in report["trials"]
                ],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
