"""Verify rendered vs simulated joints, inertias and free-base standing trajectory."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np


def compare(folder):
    files = [folder / n for n in ("physics.xml", "triceratops.xml")]
    m, vis = [mj.MjModel.from_xml_path(str(p)) for p in files]
    a, b = mj.MjData(m), mj.MjData(vis)
    diff = {}
    for key in ["body_mass", "body_ipos", "body_inertia"]:
        pairs = [
            (getattr(m, key)[i], getattr(vis, key)[vis.body(m.body(i).name).id])
            for i in range(1, m.nbody)
            if m.body_mass[i] > 0
        ]
        diff[key] = max(float(np.max(np.abs(x - y))) for x, y in pairs)
    if [m.joint(i).name for i in range(m.njnt)] != [vis.joint(i).name for i in range(vis.njnt)]:
        raise ValueError("The joint names or their order differ between models.")
    if [m.actuator(i).name for i in range(m.nu)] != [vis.actuator(i).name for i in range(vis.nu)]:
        raise ValueError("The actuator names or their order differ between models.")
    diff["actuator_forcerange"] = float(np.max(np.abs(m.actuator_forcerange - vis.actuator_forcerange)))
    diff["mass_total"] = abs(float(m.body_mass.sum() - vis.body_mass.sum()))
    for mdl, d in [(m, a), (vis, b)]:
        mj.mj_resetDataKeyframe(mdl, d, 0)
    max_pose = 0
    for i in range(1500):
        mj.mj_step(m, a)
        mj.mj_step(vis, b)
        max_pose = max(max_pose, float(np.max(np.abs(a.qpos - b.qpos))))
    diff["max_qpos_1500_steps"] = max_pose
    if max(diff.values()) >= 1e-7:
        raise ValueError(f"Physics and appearance model difference exceeds tolerance: {diff}")
    return {
        "xml_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "mass_kg": float(m.body_mass.sum()),
        "joints": m.njnt,
        "actuators": m.nu,
        "maximum_differences": diff,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(compare(a.folder), indent=2) + "\n")
