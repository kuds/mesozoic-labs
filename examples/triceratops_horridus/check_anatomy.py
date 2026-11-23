"""Exercise free stance and a level reference walk; print physically measured diagnostics."""

import argparse
import json

import mujoco
import numpy as np

from examples.shared.footfalls import swing_events
from examples.shared.ground_contacts import support_margin

from .courses import foot_contacts
from .reference_walk import CONTROL_DT, FEET, MODEL_PATH, PERIOD, ReferenceWalk


def check_mesh_winding(model):
    """Closed skin meshes must use each edge twice with opposite face winding."""
    for mesh_id in range(model.nmesh):
        start = model.mesh_faceadr[mesh_id]
        faces = model.mesh_face[start : start + model.mesh_facenum[mesh_id]]
        pairs = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
        direction = np.sign(pairs[:, 1] - pairs[:, 0])
        _, edge_ids, counts = np.unique(np.sort(pairs, axis=1), axis=0, return_inverse=True, return_counts=True)
        balance = np.bincount(edge_ids.ravel(), weights=direction)
        assert np.all(direction != 0) and np.all(counts == 2) and np.all(balance == 0), (
            f"{model.mesh(mesh_id).name} has an open, non-manifold or inconsistently wound edge"
        )


def inspect_scene(path, cycles=12, trace_path=None, friction=None):
    m = mujoco.MjModel.from_xml_path(str(path))
    check_mesh_winding(m)
    d = mujoco.MjData(m)
    walk = ReferenceWalk(m)
    contacts = foot_contacts(m)
    touch_addr = [m.sensor(foot + "_sole_force").adr[0] for foot in FEET]
    if friction is not None:
        if not np.isfinite(friction) or friction <= 0:
            raise ValueError("Sliding friction must be positive and finite")
        m.geom_friction[contacts.surfaces[0], 0] = friction
        m.geom_friction[contacts.pads, 0] = friction
    root = walk.trunk
    weight = walk.weight
    assert m.joint("root").type[0] == mujoco.mjtJoint.mjJNT_FREE
    np.testing.assert_allclose(m.opt.gravity, [0.0, 0.0, -9.81])
    # Passive balance runs from a fresh default pose, independently of walking.
    for _ in range(100):
        mujoco.mj_step(m, d, nstep=round(0.1 / m.opt.timestep))
        assert d.xmat[root, 8] > 0.95 and np.isfinite(d.qpos).all(), "Passive stance lost balance"
    mujoco.mj_forward(m, d)
    sample = contacts.measure(m, d)
    loads = sample.loads
    com = d.subtree_com[root].copy()
    stance = {
        "duration_s": d.time,
        "mass_kg": m.body_mass.sum(),
        "com_xyz_m": com.tolist(),
        "root_height_m": d.xpos[root, 2],
        "foot_load_N": dict(zip(FEET, loads)),
        "fore_load_fraction": loads[:2].sum() / loads.sum(),
        "support_margin_m": support_margin(sample.contact_xy, com[:2]),
    }
    assert stance["support_margin_m"] > 0.2 and np.all(loads > 0.05 * weight)
    np.testing.assert_allclose(loads.sum(), weight, rtol=0.02)
    np.testing.assert_allclose(d.sensordata[touch_addr], loads, rtol=1e-5, atol=1.0)
    assert sample.other_load < 1.0 and not np.any(d.warning.number)
    # Preserve the original study's independent articulation check.
    old_gape = d.joint("jaw_gape").qpos[0]
    d.ctrl[m.actuator("jaw_gape_torque").id] = 0.4
    mujoco.mj_step(m, d, nstep=round(0.5 / m.opt.timestep))
    assert d.joint("jaw_gape").qpos[0] > old_gape + 0.05
    assert not np.any(d.warning.number)

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    # Exclude starts and measure whole cycles. Also check for falls/non-foot strikes during start.
    warmup = 0.8 + 2 * PERIOD
    steps = round(CONTROL_DT / m.opt.timestep)
    dt = steps * m.opt.timestep
    axial_joints = ("neck_pitch", "occipital_pitch", "tail_1_yaw", "tail_1_pitch")
    axial_qpos = [m.joint(name).qposadr[0] for name in axial_joints]
    records = []
    poses = []
    time = []
    load_trace = []
    for _ in range(round((warmup + cycles * PERIOD) / dt)):
        walk.update(d)
        mujoco.mj_step(m, d, nstep=steps)
        mujoco.mj_forward(m, d)
        assert np.isfinite(d.qvel).all() and d.xmat[root, 8] > 0.94, "Lost upright balance"
        sample = contacts.measure(m, d)
        loads = sample.loads
        assert sample.other_load < 1.0, "A non-pad body part is bearing weight"
        np.testing.assert_allclose(d.sensordata[touch_addr], loads, rtol=1e-5, atol=1.0)
        if trace_path:
            poses.append(d.qpos.copy())
            time.append(d.time)
            load_trace.append(loads)
        if d.time >= warmup - dt / 2:
            records.append(
                (
                    d.xpos[root].copy(),
                    d.xmat[root].copy(),
                    d.site_xpos[walk.sites].copy(),
                    loads,
                    d.qpos[axial_qpos].copy(),
                    sample.clearance,
                    sample.slip,
                    support_margin(sample.contact_xy, d.subtree_com[root, :2]),
                    np.mean(np.abs(d.ctrl[walk.motors]) > 0.999),
                )
            )
    assert not np.any(d.warning.number), "MuJoCo reported a warning"
    if trace_path:
        np.savez_compressed(trace_path, time=time, qpos=poses, loads=load_trace, feet=FEET)
    root_pos, rot, soles, forces, axial, gaps, slip, margin, saturation = map(np.asarray, zip(*records))
    loaded = forces > weight * 0.01
    support_count = loaded.sum(axis=1)
    # Project XY placements onto the trunk's horizontal lateral axis.
    yaw = np.arctan2(rot[:, 3], rot[:, 0])
    offsets = soles[:, :, :2] - root_pos[:, None, :2]
    lateral = -np.sin(yaw[:, None]) * offsets[:, :, 0] + np.cos(yaw[:, None]) * offsets[:, :, 1]
    assert np.all(lateral[:, (0, 2)] > 0) and np.all(lateral[:, (1, 3)] < 0), "Foot crossed the midline"
    per_foot = {}
    touchdowns = []
    for i, name in enumerate(FEET):
        footfalls, clearances, shallow = swing_events(loaded[:, i], gaps[:, i])
        assert len(footfalls) >= cycles - 1, f"{name} missed a swing"
        touchdowns.extend((int(frame), name) for frame in footfalls)
        strides = np.diff(soles[footfalls, i, 0])
        per_foot[name] = {
            "duty_factor": loaded[:, i].mean(),
            "complete_swings": len(footfalls),
            "threshold_dips_without_5mm_clearance": shallow,
            "median_stride_m": np.median(strides),
            "median_peak_swing_clearance_m": np.median(clearances),
            "minimum_peak_swing_clearance_m": np.min(clearances),
        }
    expected = ["left_hind", "left_fore", "right_hind", "right_fore"]
    events = [name for _, name in sorted(touchdowns)]
    correct = sum(b == expected[(expected.index(a) + 1) % 4] for a, b in zip(events, events[1:]))
    elapsed = (len(records) - 1) * dt
    displacement = root_pos[-1] - root_pos[0]
    report = {
        "duration_s": elapsed,
        "displacement_x_m": displacement[0],
        "displacement_y_m": displacement[1],
        "forward_speed_m_s": displacement[0] / elapsed,
        "fore_load_fraction": forces[:, :2].sum() / forces.sum(),
        "com_inside_contact_hull_fraction": (margin >= 0).mean(),
        "minimum_com_margin_m": margin.min(),
        "minimum_loaded_feet": int(support_count.min()),
        "three_or_four_loaded_fraction": (support_count >= 3).mean(),
        "contact_sequence_agreement": correct / (len(events) - 1),
        "mean_load_weighted_pad_slip_m_s": slip.mean(),
        "mean_fore_track_width_m": (lateral[:, 0] - lateral[:, 1]).mean(),
        "mean_hind_track_width_m": (lateral[:, 2] - lateral[:, 3]).mean(),
        "lowest_pad_relative_to_ground_m": gaps.min(),
        "axial_joint_excursion_deg": dict(zip(axial_joints, np.degrees(np.ptp(axial, axis=0)))),
        "max_trunk_tilt_deg": np.degrees(np.arccos(np.clip(rot[:, 8], -1, 1))).max(),
        "max_absolute_yaw_deg": np.abs(np.degrees(np.arctan2(rot[:, 3], rot[:, 0]))).max(),
        "trunk_vertical_excursion_m": np.ptp(root_pos[:, 2]),
        "limb_motor_saturation_fraction": saturation.mean(),
        "feet": per_foot,
    }
    return {
        "mujoco_version": mujoco.__version__,
        "floor_sliding_friction": m.geom_friction[contacts.surfaces[0], 0],
        "stance": stance,
        "walk": report,
    }


def check_reference_walk(gait):
    """Regression bounds for the twelve-stride level-floor trial."""
    assert gait["forward_speed_m_s"] > 0.23 and abs(gait["displacement_y_m"]) < 0.3
    assert gait["minimum_loaded_feet"] >= 2 and gait["three_or_four_loaded_fraction"] > 0.98
    assert gait["contact_sequence_agreement"] > 0.98
    assert gait["mean_load_weighted_pad_slip_m_s"] < 0.03
    assert gait["minimum_com_margin_m"] > -0.06 and gait["limb_motor_saturation_fraction"] < 0.07
    for foot in gait["feet"].values():
        assert 0.65 < foot["duty_factor"] < 0.9 and 11 <= foot["complete_swings"] <= 13
        assert foot["minimum_peak_swing_clearance_m"] > 0.018 and foot["threshold_dips_without_5mm_clearance"] <= 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--trace", help="Optional .npz with time, free-body qpos and measured LF/RF/LH/RH loads")
    parser.add_argument(
        "--friction", type=float, help="Override sliding friction on the floor and all twelve sole pads"
    )
    args = parser.parse_args()
    results = inspect_scene(args.model, trace_path=args.trace, friction=args.friction)
    print(json.dumps(results, indent=2))
    check_reference_walk(results["walk"])


if __name__ == "__main__":
    main()
