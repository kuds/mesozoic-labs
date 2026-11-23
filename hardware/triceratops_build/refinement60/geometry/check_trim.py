"""Evaluate trimmed shoulder panel mesh samples against the complete fore drive bank."""

import itertools
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from skin_fit import geometry


def check(path):
    m, d, geo = geometry(path)
    names = ["study_left_fore_proximal", "study_L_scapula", "study_L_coracoid", "study_L_rib_5", "study_dorsal_saddle"]
    names = [n for n in names if n in geo]
    motors = [
        m.geom("left_fore_" + n).id for n in ("x4_roll_stator", "x4_pitch_stator", "x6_flexion_stator", "drive_crank")
    ]
    joint = [m.joint("left_fore_" + j) for j in ["roll", "pitch"]]
    adr = [j.qposadr[0] for j in joint]
    best = {}
    for pose in itertools.product(*[np.linspace(*j.range, 21) for j in joint]):
        d.qpos[adr] = pose
        mj.mj_kinematics(m, d)
        for n in names:
            obj, verts = geo[n]
            for gi in motors:
                M = d.geom_xmat[gi].reshape(3, 3)
                F = d.geom_xmat[obj].reshape(3, 3).T @ M
                pos = verts @ F + (d.geom_xpos[obj] - d.geom_xpos[gi]) @ M
                radial = np.hypot(pos[:, 0], pos[:, 1]) - m.geom_size[gi, 0]
                axial = abs(pos[:, 2]) - m.geom_size[gi, 1]
                sdf = np.hypot(np.maximum(radial, 0), np.maximum(axial, 0)) + np.minimum(np.maximum(radial, axial), 0)
                key = n + "/" + m.geom(gi).name
                best[key] = min(best.get(key, 500), round(float(sdf.min() * 1000), 3))
    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(args.model), indent=2) + "\n")
