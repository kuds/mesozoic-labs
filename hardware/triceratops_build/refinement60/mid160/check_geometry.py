"""Screen both foreleg mechanisms against the frame and sampled exhibit surfaces."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "geometry"))
import check_models  # noqa: E402
import rod_fit  # noqa: E402
import skin_fit  # noqa: E402
import validate_fit  # noqa: E402


def panel_clouds(m, d, geoms, pre):
    """Combine sampled surfaces in each body's rest frame."""
    arm = m.body(pre + "upper_link").id
    shank = m.body(pre + "lower_link").id
    groups = {}
    for name, (geom_id, samples) in geoms.items():
        body = m.geom_bodyid[geom_id]
        group = "torso"
        if body == arm:
            group = "drive_crank_guard" if "partial_guard" in name else "upper_sculpt"
        elif body == shank:
            group = "output_crank_guard" if "partial_guard" in name else "lower_sculpt_and_cuff"
        world = samples @ d.geom_xmat[geom_id].reshape(3, 3).T + d.geom_xpos[geom_id]
        local = (world - d.xpos[body]) @ d.xmat[body].reshape(3, 3)
        groups.setdefault((int(body), group), []).append(local)
    return [(body, name, cKDTree(np.vstack(values))) for (body, name), values in groups.items()]


def signed_cylinder(points, radius, half_length):
    radial = np.linalg.norm(points[:, :2], axis=1) - radius
    axial = abs(points[:, 2]) - half_length
    return np.hypot(np.maximum(radial, 0), np.maximum(axial, 0)) + np.minimum(np.maximum(radial, axial), 0)


def surface_gaps(path, side):
    m, d, geoms = skin_fit.geometry(path, side)
    pre = side + "_fore_"
    drive_guard = m.geom("study_" + pre + "drive_crank_partial_guard")
    mesh_id = m.geom_dataid[drive_guard.id]
    verts = m.mesh_vert[m.mesh_vertadr[mesh_id] : m.mesh_vertadr[mesh_id] + m.mesh_vertnum[mesh_id]]
    faces = m.mesh_face[m.mesh_faceadr[mesh_id] : m.mesh_faceadr[mesh_id] + m.mesh_facenum[mesh_id]]
    tri = verts[faces]
    samples = np.vstack((verts, tri.mean(axis=1), *((tri[:, k] + tri[:, (k + 1) % 3]) / 2 for k in range(3))))
    geoms[drive_guard.name] = drive_guard.id, samples
    clouds = panel_clouds(m, d, geoms, pre)
    joints = [m.joint(pre + name) for name in ("roll", "pitch", "flexion")]
    addresses = [j.qposadr[0] for j in joints]
    upper = m.body(pre + "upper_link").id
    rod = m.geom(pre + "pushrod")
    crank = m.geom(pre + "output_crank")
    centers = m.body(pre + "lower_link").pos - m.geom(pre + "x6_flexion_stator").pos
    pin = np.array([-centers[2], 0, centers[0]]) * (0.028 / np.linalg.norm(centers[[0, 2]]))
    best = {}
    for pose in itertools.product(*(np.linspace(*j.range, count) for j, count in zip(joints, (15, 15, 9)))):
        d.qpos[addresses] = pose
        mj.mj_kinematics(m, d)
        theta = joints[-1].axis[1] * pose[-1]
        c, s = np.cos(theta), np.sin(theta)
        moving_pin = [c * pin[0] + s * pin[2], 0, -s * pin[0] + c * pin[2]]
        origin = d.geom_xpos[rod.id] + d.xmat[upper].reshape(3, 3) @ (moving_pin - pin)
        for body, panel, tree in clouds:
            local_axes = d.xmat[body].reshape(3, 3)
            for tool, geom, center in (("moving_rod", rod, origin), ("output_crank", crank, d.geom_xpos[crank.id])):
                if tool == "output_crank" and panel == "output_crank_guard":
                    continue  # The guard intentionally wraps its crank.
                center_body = (center - d.xpos[body]) @ local_axes
                radius, half = geom.size[:2]
                bound = half + radius if tool == "moving_rod" else np.hypot(radius, half)
                # Once a minimum is known, any better sample must be inside this sphere.
                key = tool + "/" + panel
                prior = best.get(key, 0.40)
                indices = tree.query_ball_point(center_body, max(bound + prior, 0))
                if not indices:
                    continue
                axes = local_axes.T @ d.geom_xmat[geom.id].reshape(3, 3)
                xyz = (tree.data[indices] - center_body) @ axes
                if tool == "moving_rod":
                    xyz[:, 2] -= np.clip(xyz[:, 2], -half, half)
                    closest = np.linalg.norm(xyz, axis=1).min() - radius
                else:
                    closest = signed_cylinder(xyz, radius, half).min()
                best[key] = min(prior, closest)
    return {key: round(1000 * value, 3) for key, value in best.items()}


def examine(folder):
    physics = folder / "physics.xml"
    dressed = folder / "triceratops.xml"
    fit = validate_fit.check(physics)
    rods = rod_fit.run(physics)
    results = {}
    for side, letter in (("left", "L"), ("right", "R")):
        p = side + "_fore_"
        pairs = fit["sides"][side]["pair_result"]
        skin = skin_fit.run(dressed, side=side)
        results[side] = {
            "linkage": fit["sides"][side]["linkage"],
            "sampled_gaps_mm": {
                "x6_to_upper_strut": pairs[p + "x6_flexion_stator/" + p + "upper_strut"]["minimum_mm"],
                "x6_to_rail": pairs[p + "x6_flexion_stator/6061_" + side + "_rail"]["minimum_mm"],
                "plate_to_rail": pairs[p + "offset_motor_mount/6061_" + side + "_rail"]["minimum_mm"],
                "moving_rod_to_x6": rods[side]["rod/" + p + "x6_flexion_stator"],
                "x6_to_nearest_panel": min(r["min_mm"] for r in skin.values()),
                "x6_to_upper_sculpt": skin["study_" + p + "proximal"]["min_mm"],
                **surface_gaps(dressed, side),
            },
        }
    return {"model_equivalence": check_models.compare(folder), "sides": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("geometry.json"))
    options = parser.parse_args()
    report = {label: examine(ROOT / "models" / name) for label, name in (("128_mm", "refined"), ("160_mm", "mid160"))}
    options.out.write_text(json.dumps(report, indent=2) + "\n")
