"""Measure the adult MJCF at neutral joint angles in metres (+X cranial, +Y left)."""

import argparse
from pathlib import Path

import mujoco
import numpy as np


def skin_bounds(model, data, geom_id):
    """World-aligned bounds from actual mesh vertices or analytical primitive extents."""
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    center = data.geom_xpos[geom_id]
    kind = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
        points = vertices @ rotation.T + center
        return np.array([points.min(axis=0), points.max(axis=0)])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.linalg.norm(rotation * size, axis=1)
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + np.abs(rotation[:, 2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.full(3, size[0])
    else:
        raise ValueError(f"Unsupported surface for {model.geom(geom_id).name}: {kind}")
    return np.array([center - extent, center + extent])


def measurements(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    skin = {model.geom(i).name: skin_bounds(model, data, i) for i in np.flatnonzero(model.geom_group == 2)}
    beak = skin["rostral_beak"][1, 0]
    tail_tip = skin["tail_skin_3"][0, 0]
    frill = skin["parietosquamosal_frill"]

    def joint_span(first, last):
        return np.linalg.norm(data.body(first).xpos - data.body(last).xpos)

    surface_bounds = np.stack(list(skin.values()))
    fore = np.stack([skin_bounds(model, data, model.geom(f"left_fore_pad_{i}").id) for i in (1, 2, 3)])
    hind = np.stack([skin_bounds(model, data, model.geom(f"left_hind_pad_{i}").id) for i in (1, 2, 3)])
    return {
        "Full visible span (including horn sheaths)": surface_bounds[:, 1, 0].max() - surface_bounds[:, 0, 0].min(),
        "Body, beak to tail tip (horizontal)": beak - tail_tip,
        "Skull, beak to rear of frill (horizontal)": beak - frill[0, 0],
        "Tail, first caudal pivot to tip (horizontal)": data.body("caudals_1").xpos[0] - tail_tip,
        "Shoulder to hip (3-D socket centres)": joint_span("left_fore_upper", "left_hind_upper"),
        "Maximum ribcage width": skin["ribcage_skin"][1, 1] - skin["ribcage_skin"][0, 1],
        "Frill top above floor": frill[1, 2],
        "Humerus (pivot span)": joint_span("left_fore_upper", "left_fore_lower"),
        "Forearm (pivot span)": joint_span("left_fore_lower", "left_fore_manus"),
        "Femur (pivot span)": joint_span("left_hind_upper", "left_hind_lower"),
        "Lower hindlimb (pivot span)": joint_span("left_hind_lower", "left_hind_pes"),
        "Fore sole track (centre to centre)": abs(
            data.site("left_fore_sole").xpos[1] - data.site("right_fore_sole").xpos[1]
        ),
        "Hind sole track (centre to centre)": abs(
            data.site("left_hind_sole").xpos[1] - data.site("right_hind_sole").xpos[1]
        ),
        "Fore contact pad envelope, length": fore[:, 1, 0].max() - fore[:, 0, 0].min(),
        "Fore contact pad envelope, width": fore[:, 1, 1].max() - fore[:, 0, 1].min(),
        "Hind contact pad envelope, length": hind[:, 1, 0].max() - hind[:, 0, 0].min(),
        "Hind contact pad envelope, width": hind[:, 1, 1].max() - hind[:, 0, 1].min(),
        "Lowest tail surface above ground": min(v[0, 2] for k, v in skin.items() if k.startswith("tail_skin")),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=Path(__file__).with_name("anatomical.xml"))
    for label, value in measurements(parser.parse_args().model).items():
        print(f"{label:46} {value:6.3f} m")
