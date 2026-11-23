"""Sample signed surface distances between foreleg drives and selected open-rib panels."""

import itertools
import json
from pathlib import Path

import mujoco as mj
import numpy as np


def geometry(path, side="left"):
    m = mj.MjModel.from_xml_path(str(path))
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    mj.mj_forward(m, d)
    names = ["study_L_rib_" + str(i) for i in range(6)] + [
        "study_L_scapula",
        "study_L_coracoid",
        "study_dorsal_saddle",
        "study_left_fore_proximal",
        "study_left_fore_distal",
        "study_left_fore_secondary",
        "left_fore_carpal_cuff",
        "study_left_fore_output_crank_partial_guard",
    ]
    if side == "right":
        names = [n.replace("study_L_", "study_R_").replace("left_", "right_") for n in names]
    geometries = {}
    for name in names:
        gi = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, name)
        if gi < 0:
            continue
        mesh_id = m.geom_dataid[gi]
        vs = m.mesh_vert[m.mesh_vertadr[mesh_id] : m.mesh_vertadr[mesh_id] + m.mesh_vertnum[mesh_id]]
        faces = m.mesh_face[m.mesh_faceadr[mesh_id] : m.mesh_faceadr[mesh_id] + m.mesh_facenum[mesh_id]]
        tri = vs[faces]
        v = np.vstack(
            [
                vs,
                np.mean(tri, axis=1),
                (tri[:, 0] + tri[:, 1]) / 2,
                (tri[:, 1] + tri[:, 2]) / 2,
                (tri[:, 2] + tri[:, 0]) / 2,
            ]
        )
        geometries[name] = gi, v
    return m, d, geometries


def run(path, walk=None, part="x6_flexion_stator", side="left"):
    m, d, geoms = geometry(path, side)
    gi = m.geom(side + "_fore_" + part).id
    q = [m.joint(side + "_fore_" + j) for j in ("roll", "pitch", "flexion")]
    ars = np.array([j.qposadr[0] for j in q])
    best = {n: {"min_mm": 999.0, "poses_penetrated": 0} for n in geoms}
    if walk is None:
        poses = itertools.product(*[np.linspace(*j.range, k) for j, k in zip(q, [15, 15, 5])])
    else:
        with np.load(walk) as recording:
            poses = recording["qpos"][::3, ars]
    for a in poses:
        d.qpos[ars] = a
        mj.mj_kinematics(m, d)
        M = d.geom_xmat[gi].reshape(3, 3)
        for n, (g, vs) in geoms.items():
            F = d.geom_xmat[g].reshape(3, 3).T @ M
            ofs = (d.geom_xpos[g] - d.geom_xpos[gi]) @ M
            pos = vs @ F + ofs
            radial = (pos[:, 0] ** 2 + pos[:, 1] ** 2) ** 0.5 - m.geom_size[gi, 0]
            axial = abs(pos[:, 2]) - m.geom_size[gi, 1]
            sdf = np.hypot(np.maximum(radial, 0), np.maximum(axial, 0)) + np.minimum(np.maximum(radial, axial), 0)
            gap = float(sdf.min() * 1000)
            best[n]["poses_penetrated"] += int(gap < 0)
            if gap < best[n]["min_mm"]:
                best[n].update(min_mm=round(gap, 3), angles=np.round(np.rad2deg(a), 1).tolist())
    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Textured exhibit XML")
    parser.add_argument("output", type=Path)
    parser.add_argument("--trajectory", type=Path, help="Optional matching recording with qpos samples")
    parser.add_argument(
        "--drive",
        choices=("x4_pitch_stator", "x4_roll_stator", "x6_flexion_stator", "drive_crank"),
        default="x6_flexion_stator",
    )
    parser.add_argument("--side", choices=("left", "right"), default="left")
    args = parser.parse_args()
    result = run(args.model, args.trajectory, args.drive, args.side)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
