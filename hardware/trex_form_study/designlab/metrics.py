"""Dimensions and proportion ratios measured on the compiled model at the home pose.

Width quantities are kept distinct (see DEFINITIONS.md):
* torso_width          full width of the trunk coverings (cover_* geoms on the trunk body, arms excluded)
* hip_pitch_spacing    distance between the two hip-pitch axes (2 x y_leg)
* roll_axis_offset     lateral offset of each hip-roll axis from the body centreline
* hardware_width       full width across all hardware primitives (motors, brackets, connectors' envelopes)
* thigh_width          full width across the thigh fairings
* foot_spacing         distance between the two sole-site centres (nominal stance)
* support_width        full width of the two pads' convex hull; support_area is that hull's area
* max_width            full width of everything visible
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


def _geom_points(m, d, g):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a : a + n]
        R = d.geom_xmat[g].reshape(3, 3)
        return d.geom_xpos[g] + v @ R.T
    R = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g] + R @ m.geom_aabb[g][:3]
    h = m.geom_aabb[g][3:]
    corners = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return c + corners @ R.T


def _extent(m, d, geoms):
    pts = np.vstack([_geom_points(m, d, g) for g in geoms]) if geoms else np.zeros((1, 3))
    return pts.min(0), pts.max(0)


def measure(M, design, q_home) -> dict:
    M.forward(q_home)
    m, d = M.m, M.d
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""  # noqa: E731
    vis = [g for g in range(m.ngeom) if m.geom_group[g] in (0, 1, 2) and name(g) != "floor" and m.geom_rgba[g, 3] > 0.3]
    covers = [g for g in vis if name(g).startswith("cover_")]
    trunk_cov = [g for g in covers if body(g) == "trunk" and "arm" not in name(g) and "rib_hoop" not in name(g)]
    thigh_cov = [g for g in covers if "thigh" in name(g)]
    hwg = [g for g in vis if name(g).startswith("hw_") and not any(k in name(g) for k in ("tooth", "eye", "claw"))]
    skull = [g for g in covers if name(g) in ("cover_skull", "cover_jaw")]
    neck = [g for g in covers if name(g) == "cover_neck"]
    tailg = [g for g in vis if body(g) == "tail"]
    pads = [g for g in range(m.ngeom) if m.geom_contype[g] == 1 and name(g) != "floor"]
    lo, hi = _extent(m, d, vis)
    tlo, thi = _extent(m, d, trunk_cov)
    hlo, hhi = _extent(m, d, hwg)
    th_lo, th_hi = _extent(m, d, thigh_cov) if thigh_cov else (np.zeros(3), np.zeros(3))
    sk_lo, sk_hi = _extent(m, d, skull)
    nk_lo, nk_hi = _extent(m, d, neck)
    pad_pts = np.vstack([_geom_points(m, d, g) for g in pads])[:, :2]
    hull = ConvexHull(pad_pts)
    soles = [d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_sole")] for s in ("left", "right")]
    H = design.sk.H
    crown = float(hi[2])
    torso_width = float(thi[1] - tlo[1])
    hw_width = float(hhi[1] - hlo[1])
    foot_spacing = float(abs(soles[0][1] - soles[1][1]))
    torso_len = float(thi[0] - tlo[0])
    torso_depth = float(thi[2] - tlo[2])
    skull_len = float(sk_hi[0] - sk_lo[0])
    tail_len = float(np.linalg.norm(design.sk.tail_path[-1] - design.sk.tail_path[0]))
    total_len = float(hi[0] - lo[0])
    masses = {}
    for b in range(1, m.nbody):
        masses[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)] = float(m.body_mass[b])
    total = float(m.body_subtreemass[M.root])
    head_mass = float(m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")])
    tail_mass = float(m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tail")])
    leg_mass = float(sum(m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_roll")]
                         for s in ("left", "right")))
    com = M.com()
    out = dict(
        crown=crown, hip_height=float(H), length=total_len, max_width=float(hi[1] - lo[1]),
        torso_width=torso_width, torso_length=torso_len, torso_depth=torso_depth,
        thigh_width=float(th_hi[1] - th_lo[1]), hardware_width=hw_width,
        hip_pitch_spacing=float(2 * design.hip["y_leg"]), roll_axis_offset=float(design.hip["y_roll"]),
        foot_spacing=foot_spacing, support_width=float(pad_pts[:, 1].max() - pad_pts[:, 1].min()),
        support_area=float(hull.volume), skull_length=skull_len, skull_width=float(sk_hi[1] - sk_lo[1]),
        skull_depth=float(sk_hi[2] - sk_lo[2]), neck_length=float(np.sum(np.linalg.norm(np.diff(design.sk.neck_path, axis=0),
                                                                                         axis=1))),
        tail_length=tail_len, tail_root_height=None, tail_root_width=None,
        mass=total, head_neck_mass=head_mass, tail_mass=tail_mass, legs_mass=leg_mass,
        com=com.tolist(), com_ahead_of_hip=float(com[0] - 0.0), com_height=float(com[2]),
        torso_inclination_deg=float(np.degrees(design.sk.theta_c)), pelvis_inclination_deg=float(np.degrees(design.sk.theta_p)),
    )
    out["ratios"] = dict(
        crown_over_torso_width=crown / torso_width,
        crown_over_hardware_width=crown / hw_width,
        hip_height_over_foot_spacing=H / foot_spacing,
        torso_length_over_depth=torso_len / torso_depth,
        skull_length_over_torso_length=skull_len / torso_len,
        skull_width_over_torso_width=float(sk_hi[1] - sk_lo[1]) / torso_width,
        neck_length_over_skull_length=out["neck_length"] / skull_len,
        tail_length_over_total_length=tail_len / total_len,
        length_over_crown=total_len / crown,
        crown_over_hip_height=crown / H,
        tail_mass_fraction=tail_mass / total,
        head_neck_mass_fraction=head_mass / total,
        legs_mass_fraction=leg_mass / total,
        com_ahead_of_hip_over_H=float(com[0]) / H,
    )
    # tail root section size (first 10 % of the tail path)
    tr = design.sk.tail_path[0]
    near = []
    for g in tailg:
        p = _geom_points(m, d, g)
        sel = p[np.linalg.norm(p[:, [0, 2]] - tr[[0, 2]], axis=1) < 0.12 * design.sk.B]
        if len(sel):
            near.append(sel)
    if near:
        q = np.vstack(near)
        out["tail_root_height"] = float(q[:, 2].max() - q[:, 2].min())
        out["tail_root_width"] = float(q[:, 1].max() - q[:, 1].min())
    return out
