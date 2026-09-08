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

RESET_SEEDS = tuple(range(42, 52))
PROTOCOL = {
    "sample_interval_s": 0.02,
    "settling_time_s": 0.5,
    "max_tilt_deg": 5.0,
    "max_height_drop_m": 0.005,
    "max_horizontal_drift_m": 0.010,
    "min_support_margin_m": 0.005,
    "min_either_foot_load_fraction": 0.20,
    "max_ground_weight_relative_error": 0.03,
    "max_touch_ground_relative_error": 0.005,
    "max_contact_penetration_m": 0.001,
    "reset_joint_noise_deg": 1.0,
}


def support_sample(model, data):
    """Strict loaded-contact hull; a missing/degenerate polygon is a failure."""
    contacts = floor_contacts(model, data)
    points = np.array([c["position"][:2] for c in contacts if c["normal_force"] > 0.01])
    com = data.subtree_com[model.body("pelvis").id]
    margin = None
    if len(points) >= 3 and np.linalg.matrix_rank(points - points.mean(axis=0)) == 2:
        hull = ConvexHull(points)
        margin = float(np.min(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2])))
    foot_loads = [float(data.sensor(name).data[0]) for name in ("r_foot_touch", "l_foot_touch")]
    ground = sum(c["normal_force"] for c in contacts)
    weight = float(model.body_subtreemass[model.body("pelvis").id] * np.linalg.norm(model.opt.gravity))
    return {
        "support_margin_m": margin,
        "foot_loads_N": foot_loads,
        "ground_normal_load_N": ground,
        "body_weight_N": weight,
        "minimum_foot_load_fraction": min(foot_loads) / max(ground, 1e-12),
        "ground_weight_relative_error": abs(ground - weight) / weight,
        "touch_ground_relative_error": abs(sum(foot_loads) - ground) / max(ground, 1e-12),
    }


def hold_trial(variant: str, *, mass_scale=1.0, seconds=10.0, seed=None, motors_off=False):
    if not np.isfinite(seconds) or seconds <= PROTOCOL["settling_time_s"]:
        raise ValueError("Trial must be longer than the 0.5-second settling interval")
    model, data = load_model(variant, mass_scale)
    root = model.body("pelvis").id
    home_xy = data.xpos[root, :2].copy()
    home_height = float(data.xpos[model.body("pelvis").id, 2])
    reset_height_correction = 0.0
    if seed is not None:
        rng = np.random.default_rng(seed)
        # Small independent joint-angle perturbations; not a certified reset distribution.
        for j in range(1, model.njnt):
            address = model.jnt_qposadr[j]
            limit = np.deg2rad(PROTOCOL["reset_joint_noise_deg"])
            data.qpos[address] += rng.uniform(-limit, limit)
            data.qpos[address] = np.clip(data.qpos[address], *model.jnt_range[j])
        mujoco.mj_forward(model, data)
        # Independent angle noise can place a sole through the floor. Translate
        # only the floating base until the lowest visual point has 50 um clearance;
        # retain every sampled angle and the original fixed servo targets.
        low, _ = model_bounds(model, data)
        reset_height_correction = 0.00005 - float(low[2])
        data.qpos[2] += reset_height_correction
        mujoco.mj_forward(model, data)
    if motors_off:
        set_motors_enabled(model, False)
    peak_torque = np.zeros(model.nu)
    peak_speed = np.zeros(model.nu)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    minimum_height = home_height
    maximum_tilt = 0.0
    maximum_drift = 0.0
    minimum_contact_distance = 0.0
    support_samples = []
    interval = max(1, round(PROTOCOL["sample_interval_s"] / model.opt.timestep))
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
        maximum_drift = max(maximum_drift, float(np.linalg.norm(data.xpos[root, :2] - home_xy)))
        # Contact geometry is checked at every physics step, including reset.
        if data.ncon:
            minimum_contact_distance = min(minimum_contact_distance, float(np.min(data.contact.dist)))
        floor = model.geom("floor").id
        for c in data.contact:
            if c.efc_address >= 0 and floor in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == floor else c.geom1
                if model.geom(other).name not in foot_geoms:
                    nonfoot_contacts.add(model.geom(other).name)
        if step % interval == 0:
            mujoco.mj_forward(model, data)
            nonfoot_contacts.update(c["geom"] for c in floor_contacts(model, data) if c["geom"] not in foot_geoms)
            if data.time >= PROTOCOL["settling_time_s"]:
                support_samples.append(support_sample(model, data))
    mujoco.mj_forward(model, data)
    contacts = floor_contacts(model, data)
    com = data.subtree_com[root].copy()
    final_support = support_sample(model, data)
    support_samples.append(final_support)
    margin = final_support["support_margin_m"]
    mass = float(model.body_mass[robot_body_ids(model)].sum())
    foot_force = sum(float(data.sensor(name).data[0]) for name in ("r_foot_touch", "l_foot_touch"))
    ground_force = sum(c["normal_force"] for c in contacts)
    cap = np.max(np.abs(model.actuator_forcerange), axis=1)
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    margins = [s["support_margin_m"] for s in support_samples]
    minimum_margin = min(margins) if all(m is not None for m in margins) else None
    min_foot_fraction = min(s["minimum_foot_load_fraction"] for s in support_samples)
    ground_error = max(s["ground_weight_relative_error"] for s in support_samples)
    touch_error = max(s["touch_ground_relative_error"] for s in support_samples)
    checks = {
        "finite_warning_free": finite and not warnings,
        "upright": maximum_tilt < PROTOCOL["max_tilt_deg"],
        "height_retained": home_height - minimum_height < PROTOCOL["max_height_drop_m"],
        "limited_drift": maximum_drift < PROTOCOL["max_horizontal_drift_m"],
        "feet_only": not nonfoot_contacts,
        "shallow_contacts": minimum_contact_distance >= -PROTOCOL["max_contact_penetration_m"],
        "com_inside_support": minimum_margin is not None and minimum_margin > PROTOCOL["min_support_margin_m"],
        "both_feet_loaded": min_foot_fraction > PROTOCOL["min_either_foot_load_fraction"],
        "supports_weight": ground_error < PROTOCOL["max_ground_weight_relative_error"],
        "touch_accounts_for_ground": touch_error < PROTOCOL["max_touch_ground_relative_error"],
        "torque_caps": bool(np.max(peak_torque / cap) <= 1.000001),
    }
    return {
        "variant": variant,
        "mass_scale": mass_scale,
        "seed": seed,
        "reset_base_height_correction_m": reset_height_correction,
        "motors_off": motors_off,
        "seconds": seconds,
        "mass_kg": mass,
        "home_pelvis_height_m": home_height,
        "final_pelvis_height_m": float(data.xpos[root, 2]),
        "minimum_pelvis_height_m": minimum_height,
        "maximum_tilt_deg": maximum_tilt,
        "maximum_horizontal_drift_m": maximum_drift,
        "minimum_contact_distance_m": minimum_contact_distance,
        "minimum_settled_support_margin_m": minimum_margin,
        "minimum_settled_foot_load_fraction": min_foot_fraction,
        "maximum_settled_ground_weight_relative_error": ground_error,
        "maximum_settled_touch_ground_relative_error": touch_error,
        "support_sample_count": len(support_samples),
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
        "standing_checks": checks,
        "standing_passed": all(checks.values()),
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
        for seed in RESET_SEEDS:
            trials.append(hold_trial(variant, seed=seed, seconds=5))
        trials.append(hold_trial(variant, motors_off=True, seconds=5))
        trials.append(hold_trial(variant, mass_scale=PARAMETERS["robot"]["stress_mass_scale"]))
    return {
        "schema": "mesozoic.compso-preflight/v1",
        "protocol": PROTOCOL,
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "geometry": {variant: geometry(variant) for variant in MODEL_PATHS},
        "trials": trials,
        "scope": "Flat-floor fixed-target position-servo standing; ten seeded joint-reset perturbations per model; 15% mass growth for both; no gait, training, push recovery, or hardware validation.",
        "passed": all(
            t["standing_passed"]
            if not t["motors_off"]
            else (not t["stood"] and t["finite"] and not t["warnings"] and max(t["peak_torque_Nm"].values()) == 0)
            for t in trials
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
