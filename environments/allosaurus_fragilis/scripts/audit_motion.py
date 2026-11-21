"""Measure a free-root Allosaurus on a level plane using only its motor controls.

Measurements use MuJoCo's contact solution at each 2 ms substep. Optional NPZ
output stores the physical poses, foot positions and loads for video replay.
Use --skin-clearance to measure the deformed surfaces at the same substeps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.gait.events import debounce
from environments.shared.gait.stance_track import convex_hull, signed_margin, yaw_pitch_roll

ASSET = Path(__file__).resolve().parents[1] / "anatomical/assets/allosaurus_fragilis_anatomical.xml"
SIDES = ("l", "r")
PADS = ("pad", "pad_d2", "pad_d3", "pad_d4")


def bounds(values):
    return [float(np.min(values)), float(np.max(values))]


def median(values):
    return float(np.median(values)) if len(values) else None


def sole_height(model, data, pads):
    """Lowest point of the oriented MTP ellipsoid and three digital capsules."""
    rotation_z = data.geom_xmat[pads, 6:9]
    sizes = model.geom_size[pads]
    radii = sizes[:, 0] + sizes[:, 1] * abs(rotation_z[:, 2])
    radii[0] = np.linalg.norm(rotation_z[0] * sizes[0])
    return min(data.geom_xpos[pads, 2] - radii)


def summarize(time, root, com, foot, contact_forces, clear, slip, attitude, tips, margins, warmup, weight, dt):
    use = time >= warmup
    t = time[use]
    load = contact_forces[use]
    feet = foot[use]
    stride_clearance = clear[use]
    down = load[:, :8].reshape(-1, 2, 4).sum(axis=2) > 0.01 * weight
    landings = []
    sides = {}
    for i, side in enumerate(SIDES):
        contact = debounce(down[:, i], round(0.020 / dt)).astype(int)
        land = np.flatnonzero(np.diff(contact) > 0) + 1
        leave = np.flatnonzero(np.diff(contact) < 0) + 1
        landings.extend((int(k), i) for k in land)
        peaks, duties = [], []
        for a, b in zip(land, land[1:]):
            off = leave[(leave > a) & (leave < b)]
            if len(off) == 1:
                peaks.append(np.max(stride_clearance[off[0] : b, i]))
                duties.append((off[0] - a) / (b - a))
        sides[side] = {
            "touchdowns": len(land),
            "raw_threshold_crossings": int(np.count_nonzero(np.diff(down[:, i].astype(int)) > 0)),
            "stance_fraction": float(contact.mean()),
            "cycle_duty_median": median(duties),
            "stride_period_s_median": median(np.diff(t[land])),
            "stride_advance_m_median": median(np.diff(feet[land, i, 0])),
            "swing_clearance_m_median": median(peaks),
            "normal_impulse_share": float(load[:, 4 * i : 4 * i + 4].sum() / load.sum()) if load.sum() else None,
        }
    landings.sort()
    steps = [(a, i, b, j) for (a, i), (b, j) in zip(landings, landings[1:]) if i != j and a != b]
    travel = root[use][-1] - root[use][0]
    margin = margins[use]
    margin = margin[np.isfinite(margin)]
    loaded_slip = slip[use][np.isfinite(slip[use])]
    load_total = load.sum()
    return {
        "measurement_start_s": float(t[0]),
        "measurement_end_s": float(t[-1] + dt),
        "travel_x_m": float(travel[0]),
        "mean_forward_speed_mps": float(travel[0] / (t[-1] - t[0])),
        "lateral_drift_m": float(travel[1]),
        "pelvis_height_m": bounds(root[use, 2]),
        "pelvis_yaw_deg": bounds(attitude[use, 0]),
        "pelvis_pitch_deg": bounds(attitude[use, 1]),
        "pelvis_roll_deg": bounds(attitude[use, 2]),
        "head_tip_height_m": bounds(tips[use, 0]),
        "tail_tip_height_m": bounds(tips[use, 1]),
        "hallux_site_height_m": bounds(tips[use, 2:]),
        "com_ahead_of_pelvis_m": bounds(com[use, 0] - root[use, 0]),
        "double_support_fraction": float(np.all(down, axis=1).mean()),
        "no_foot_support_fraction": float((~np.any(down, axis=1)).mean()),
        "min_static_support_margin_m": float(margin.min()) if len(margin) else None,
        "alternating_transitions": len(steps),
        "total_touchdown_transitions": max(0, len(landings) - 1),
        "opposite_step_length_m_median": median([feet[b, j, 0] - feet[a, i, 0] for a, i, b, j in steps]),
        "track_width_m_median": median(
            [(feet[b, j, 1] - feet[a, i, 1]) * (1 if j == 0 else -1) for a, i, b, j in steps]
        ),
        "loaded_patch_slip_speed_mps_mean": float(loaded_slip.mean()) if len(loaded_slip) else None,
        "loaded_patch_slip_speed_mps_p95": float(np.percentile(loaded_slip, 95)) if len(loaded_slip) else None,
        "peak_floor_load_bodyweights": float(load.sum(axis=1).max() / weight),
        "floor_impulse_fraction": dict(
            zip(
                [f"{s}_{p}" for s in SIDES for p in PADS] + ["other"],
                (load.sum(axis=0) / load_total).tolist(),
                strict=True,
            )
        )
        if load_total
        else {},
        "feet": sides,
    }


def run(mode, seconds, warmup, trace_path=None, *, skin_clearance=False):
    from environments.allosaurus_fragilis.scripts.reference_walk import ReferenceWalk

    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    controller = ReferenceWalk(model, data, walking=mode == "walk")
    pelvis = model.body("pelvis").id
    weight = model.body_subtreemass[pelvis] * abs(model.opt.gravity[2])
    dt = model.opt.timestep
    pads = np.array([[model.geom(f"{s}_{p}").id for p in PADS] for s in SIDES])
    lookup = {int(g): i for i, g in enumerate(pads.flat)}
    pad_index = pads[:, 0]
    tip_ids = [model.site(n).id for n in ("head_tip", "tail_tip", "l_hallux", "r_hallux")]
    floor = model.geom("floor").id
    n = round(seconds / dt)
    if n < 2 or not 0 <= warmup < (n - 1) * dt:
        raise ValueError("Choose a duration with at least two measured substeps after warmup")
    time = np.arange(n) * dt
    root = np.empty((n, 3))
    quat = np.empty((n, 4))
    com = np.empty_like(root)
    foot = np.empty((n, 2, 3))
    forces = np.zeros((n, 9))
    clear = np.empty((n, 2))
    slip = np.full(n, np.nan)
    margins = np.full(n, np.nan)
    tips = np.empty((n, 4))
    hinge_angles = np.empty((n, model.nq - 7))
    q = np.empty((n, model.nq)) if trace_path else None
    force = np.zeros(6)
    velocity = np.zeros(6)
    scene = None
    if skin_clearance:
        # The world-space skin vertices are deformed on CPU. No renderer or
        # OpenGL context is needed, and the measurement leaves dynamics alone.
        scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 100)
        options = mujoco.MjvOption()
        options.skingroup[:] = 1
        camera = mujoco.MjvCamera()
        plantar = [model.geom(f"{side}_plantar_hide").id for side in SIDES]
        surface_names = [model.skin(i).name for i in range(model.nskin)] + [model.geom(g).name for g in plantar]
        surface_min = np.full(len(surface_names), np.inf)
    for k in range(n):
        if k % 5 == 0:
            data.ctrl[:] = controller.control(data)
        hinge_angles[k] = data.qpos[7:]
        if q is not None:
            q[k] = data.qpos
        # mj_step solves forces at the starting pose, then integrates qpos/qvel.
        # The derived world transforms below still refer to the solved pose.
        mujoco.mj_step(model, data)
        if scene is not None and time[k] >= warmup:
            mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            skin_z = np.minimum.reduceat(scene.skinvert.reshape(-1, 3)[:, 2], model.skin_vertadr)
            # Each rear plantar cushion is a visible rigid ellipsoid.
            vertical_radii = np.linalg.norm(data.geom_xmat[plantar, 6:9] * model.geom_size[plantar], axis=1)
            plantar_z = data.geom_xpos[plantar, 2] - vertical_radii
            ground_z = data.geom_xpos[floor, 2]
            np.minimum(surface_min, np.r_[skin_z, plantar_z] - ground_z, out=surface_min)
        root[k] = data.xpos[pelvis]
        quat[k] = data.xquat[pelvis]
        com[k] = data.subtree_com[pelvis]
        foot[k] = data.geom_xpos[pad_index]
        tips[k] = data.site_xpos[tip_ids, 2]
        clear[k] = [sole_height(model, data, row) for row in pads]
        speed_sum = 0.0
        loaded_points = []
        for idx, c in enumerate(data.contact):
            if floor not in c.geom:
                continue
            geom = int(c.geom[0] if c.geom[1] == floor else c.geom[1])
            mujoco.mj_contactForce(model, data, idx, force)
            normal = max(float(force[0]), 0.0)
            column = lookup.get(geom, 8)
            forces[k, column] += normal
            if normal > 1:
                loaded_points.append(c.pos[:2].copy())
            if column < 8:
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, velocity, 0)
                arm = c.pos - data.geom_xpos[geom]
                tangent = velocity[3:] + np.cross(velocity[:3], arm)
                speed_sum += normal * np.linalg.norm(tangent[:2])
        if forces[k, :8].sum() > 0.01 * weight:
            slip[k] = speed_sum / forces[k, :8].sum()
        if mode == "stand" and len(loaded_points) >= 3:
            margins[k] = signed_margin(convex_hull(np.array(loaded_points)), com[k])
    report = summarize(
        time, root, com, foot, forces, clear, slip, np.degrees(yaw_pitch_roll(quat)), tips, margins, warmup, weight, dt
    )
    report["hinge_angles_from_home_deg"] = {
        model.joint(idx).name: bounds(np.degrees(hinge_angles[time >= warmup, model.jnt_qposadr[idx] - 7]))
        for idx in range(1, model.njnt)
    }
    if scene is not None:
        report["min_rendered_surface_clearance_m"] = dict(zip(surface_names, surface_min.tolist(), strict=True))
    if trace_path:
        np.savez_compressed(
            trace_path, time=time, qpos=q, feet=foot, loads=forces, com=com, clearance=clear, weight=weight
        )
    return {
        "mode": mode,
        "mujoco_version": mujoco.__version__,
        "xml_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "duration_s": n * dt,
        "timestep_s": dt,
        "mass_kg": float(model.body_subtreemass[pelvis]),
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["stand", "walk"], default="walk")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--skin-clearance", action="store_true", help="Measure visible skin and plantar cushion heights above the floor"
    )
    args = parser.parse_args()
    report = run(args.mode, args.seconds, args.warmup, args.trace, skin_clearance=args.skin_clearance)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
