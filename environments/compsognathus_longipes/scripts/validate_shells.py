"""Compare solid torso envelopes against moving legs, ignoring contact masks.

This exposes geometric overlaps that standing tests cannot see. A convex
envelope also encloses intentional internal hardware: overlaps are review
items, not automatically physical shell intersections. It is not hollow-shell
CAD, a continuous swept-volume proof, or a walking test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.model import load_model


def read_reference(path):
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(MODEL_PATHS["robot"].parent / "meshes"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def poses(model):
    home = model.key_qpos[model.key("home").id].copy()
    yield "home", home
    joints = model.actuator_trnid[:, 0]
    for j in joints:
        address = model.jnt_qposadr[j]
        for degrees in (-10, -5, 5, 10):
            q = home.copy()
            q[address] = np.clip(q[address] + np.deg2rad(degrees), *model.jnt_range[j])
            yield f"{model.joint(int(j)).name}_{degrees:+d}deg", q
    rng = np.random.default_rng(20260908)
    for n in range(64):
        q = home.copy()
        for j in joints:
            address = model.jnt_qposadr[j]
            q[address] = np.clip(q[address] + rng.uniform(-1, 1) * np.deg2rad(5), *model.jnt_range[j])
        yield f"coupled_{n:02d}", q


def compare(reference):
    model, data = load_model("robot")
    before, before_data = read_reference(reference)
    body_names = {"hip_roll_assembly", "hip_roll_assembly_2"}
    for b in range(model.nbody):
        if model.body(int(model.body_parentid[b])).name in body_names:
            body_names.add(model.body(b).name)
    # Only original physical geometry; surface finishes do not affect fit.
    names = [
        before.geom(g).name for g in range(before.ngeom) if before.body(int(before.geom_bodyid[g])).name in body_names
    ]
    minimum = {name: {"before_m": 0.25, "after_m": 0.25} for name in names}
    home_overlaps, introduced, worsened = [], {}, {}
    count = 0
    for pose, qpos in poses(model):
        count += 1
        for m, d in ((before, before_data), (model, data)):
            d.qpos[:] = qpos
            mujoco.mj_forward(m, d)
        for name in names:
            distances = [
                float(mujoco.mj_geomDistance(m, d, m.geom("core_envelope").id, m.geom(name).id, 0.25, None))
                for m, d in ((before, before_data), (model, data))
            ]
            old, new = distances
            for key, distance in zip(("before_m", "after_m"), distances, strict=True):
                minimum[name][key] = min(minimum[name][key], distance)
            item = {"pose": pose, "geom": name, "before_m": old, "after_m": new}
            if pose == "home" and new < -0.0005:
                home_overlaps.append(item)
            if old >= -0.0005 and new < -0.0005:
                if name not in introduced or new < introduced[name]["after_m"]:
                    introduced[name] = item
            if new < -0.0005 and new < old - 0.001:
                if name not in worsened or new - old < worsened[name]["after_m"] - worsened[name]["before_m"]:
                    worsened[name] = item
    return {
        "schema": "mesozoic.compso-shell-screen/v1",
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256(MODEL_PATHS["robot"].read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "pose_count": count,
        "pair_count_per_pose": len(names),
        "protocol": "Home; each joint at +/-5 and +/-10 degrees relative to home, clipped to limits; 64 coupled +/-5 degree resets, seed 20260908. Direct signed distances ignore masks. 0.5 mm overlap threshold; 1 mm worsening threshold.",
        "home_solid_envelope_overlaps": home_overlaps,
        "introduced_solid_envelope_overlaps": list(introduced.values()),
        "worsened_solid_envelope_overlaps": list(worsened.values()),
        "no_new_or_worsened_overlaps_in_sample": not introduced and not worsened,
        "minimum_distances_by_geom": minimum,
        "limits": [
            "Solid convex torso, not actual hollow-shell surfaces or internal packaging",
            "Existing core-leg contact masks are bypassed for this diagnostic",
            "Intentional enclosed hardware and mounts need CAD interpretation",
            "Discrete local pose sample, not full joint travel or continuous swept clearance",
            "No walking or manufacturing qualification",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "minimum_distances_by_geom"}, indent=2))


if __name__ == "__main__":
    main()
