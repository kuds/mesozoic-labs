"""Compare the exterior's compiled physics to the source and report gross closed-surface area."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from build_exterior import SOURCE


def category(name):
    if name == "torso":
        return "torso"
    if name == "tail":
        return "tail"
    if name.startswith("frill_edge_"):
        return "frill_rim_scales"
    if name == "frill":
        return "frill"
    if name == "neck_collar":
        return "neck_collar"
    if "_fore_" in name or "_hind_" in name:
        if any(s in name for s in ("foot_shell", "rocker", "nail", "ankle_cuff")):
            return "foot_covers_and_pads"
        return "leg_segments_yokes_and_cuffs"
    if name in {"left_horn", "right_horn", "nose_horn"}:
        return "three_horns"
    if name in {"skull", "lower_jaw", "upper_beak", "mandible_tip"}:
        return "skull_and_beak"
    return "face_details_and_mouth"


def area(m, i):
    if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[i]
        va, nv = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        fa, nf = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
        vertices = m.mesh_vert[va : va + nv].astype(float)
        tris = vertices[m.mesh_face[fa : fa + nf]]
        return np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() * 0.5
    r, h = m.geom_size[i, :2]
    if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return 4 * np.pi * r * r + 4 * np.pi * r * h
    if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return 2 * np.pi * r * r + 4 * np.pi * r * h
    if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        raise ValueError(m.geom(i).name)
    # Integrate the ellipsoid's area element by Gauss-Legendre quadrature.
    a, b, c = m.geom_size[i]
    x, weights = np.polynomial.legendre.leggauss(32)
    phi = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    dx = 1 - x[:, None] ** 2
    integrand = np.sqrt(
        (b * c) ** 2 * dx * np.cos(phi) ** 2 + (c * a) ** 2 * dx * np.sin(phi) ** 2 + (a * b) ** 2 * x[:, None] ** 2
    )
    return float((integrand.mean(axis=1) * weights).sum() * 2 * np.pi)


def compare(source, presentation):
    a = mujoco.MjModel.from_xml_path(str(source))
    b = mujoco.MjModel.from_xml_path(str(presentation))
    assert (a.nq, a.nv, a.nu, a.nsensor) == (b.nq, b.nv, b.nu, b.nsensor)
    for key in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
        for i in range(a.nbody):
            j = b.body(a.body(i).name).id
            np.testing.assert_array_equal(getattr(a, key)[i], getattr(b, key)[j])
    for key in (
        "geom_type",
        "geom_contype",
        "geom_conaffinity",
        "geom_solref",
        "geom_solimp",
        "geom_friction",
        "geom_condim",
        "geom_pos",
        "geom_quat",
        "geom_size",
    ):
        for i in range(a.ngeom):
            j = b.geom(a.geom(i).name).id
            np.testing.assert_array_equal(getattr(a, key)[i], getattr(b, key)[j])
    for key in (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "jnt_range",
        "dof_armature",
    ):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    groups = {}
    for i in range(b.ngeom):
        name = b.geom(i).name
        if name.startswith("exterior_"):
            if b.geom_contype[i] or b.geom_conaffinity[i]:
                raise ValueError(name)
            cat = category(name.removeprefix("exterior_"))
            groups[cat] = groups.get(cat, 0) + area(b, i)
    d1, d2 = mujoco.MjData(a), mujoco.MjData(b)
    for mod, data in [(a, d1), (b, d2)]:
        mujoco.mj_resetDataKeyframe(mod, data, 0)
    # Check the integrated neutral stand while both models receive the same controls.
    for _ in range(1000):
        mujoco.mj_step(a, d1)
        mujoco.mj_step(b, d2)
    delta = float(np.max(np.abs(d1.qpos - d2.qpos)))
    if delta > 1e-10:
        raise ValueError(f"Standing trajectory diverged by {delta:g}")
    return {
        "nominal_mass_kg": float(b.body_mass.sum()),
        "original_bodies": a.nbody,
        "original_geoms": a.ngeom,
        "cosmetic_geoms": b.ngeom - a.ngeom,
        "max_abs_qpos_difference_at_1s_neutral_stand": delta,
        "gross_closed_area_m2": groups,
        "sum_m2": sum(groups.values()),
        "area_basis": "All modelled surfaces, including concealed overlaps, closed ends, both frill faces, and small details. No upholstery or cut-layout allowance. This is not material consumption.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    s = json.dumps(compare(args.source, args.model), indent=2)
    print(s)
    if args.out:
        args.out.write_text(s + "\n")
