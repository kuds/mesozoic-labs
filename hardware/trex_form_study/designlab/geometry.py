"""Geometry helpers: rotations, superellipse lofts, thin shells and mass properties.

Shell mass properties use the exact second moment of each triangle (lamina) and the panel's
wall thickness; nothing is treated as a solid block.  Solid primitives use closed-form inertia.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------------------------
def rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a: float) -> np.ndarray:
    """Rotation about +y.  Positive angle raises a forward-pointing (+x) vector (chest-up)."""
    c, s = np.cos(a), np.sin(a)
    # Standard right-handed rotation about y maps +x towards -z for positive angles; the study's
    # pitch convention is "nose up positive", so use the transpose.
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """MuJoCo (w, x, y, z) quaternion from a rotation matrix."""
    m = np.asarray(r, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def frame_from_axis(axis, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation whose local z-axis is ``axis`` (for cylinders/motors)."""
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    u = np.asarray(up, float)
    if abs(np.dot(u, z)) > 0.95:
        u = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(u, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------------------------
# Mass properties
# ---------------------------------------------------------------------------------------------
def box_inertia(m: float, size) -> np.ndarray:
    """Central inertia of a solid box with full edge lengths ``size``."""
    a, b, c = size
    return m / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def cylinder_inertia(m: float, r: float, length: float, axis: int = 2) -> np.ndarray:
    """Central inertia of a solid cylinder, symmetry axis ``axis`` (0, 1, 2)."""
    i_ax = 0.5 * m * r * r
    i_tr = m * (3 * r * r + length * length) / 12.0
    d = [i_tr, i_tr, i_tr]
    d[axis] = i_ax
    return np.diag(d)


def tube_inertia(m: float, r_out: float, r_in: float, length: float, axis: int = 2) -> np.ndarray:
    i_ax = 0.5 * m * (r_out**2 + r_in**2)
    i_tr = m * (3 * (r_out**2 + r_in**2) + length * length) / 12.0
    d = [i_tr, i_tr, i_tr]
    d[axis] = i_ax
    return np.diag(d)


def rotate_inertia(i_local: np.ndarray, r: np.ndarray) -> np.ndarray:
    return r @ i_local @ r.T


def parallel_axis(i_c: np.ndarray, m: float, d) -> np.ndarray:
    d = np.asarray(d, float)
    return i_c + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def combine(parts):
    """Combine (mass, com, central inertia) tuples into total mass, CoM and central inertia."""
    parts = [p for p in parts if p[0] > 0]
    if not parts:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    m = sum(p[0] for p in parts)
    c = sum(p[0] * np.asarray(p[1]) for p in parts) / m
    i = np.zeros((3, 3))
    for mi, ci, ii in parts:
        i += parallel_axis(np.asarray(ii), mi, np.asarray(ci) - c)
    return m, c, 0.5 * (i + i.T)


def shell_properties(tris: np.ndarray, thickness: float, density: float):
    """Mass, centroid and central inertia of a thin shell given its mid-surface triangles.

    ``tris`` is (n, 3, 3).  Uses the exact lamina integrals
    ∫_T dA = A, ∫_T x dA = A (a+b+c)/3, ∫_T x xᵀ dA = A/12 (aaᵀ+bbᵀ+ccᵀ+ssᵀ), s = a+b+c.
    """
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total_area = float(area.sum())
    if total_area <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3)), 0.0
    s = a + b + c
    first = (area[:, None] * s / 3.0).sum(axis=0)
    second = np.zeros((3, 3))
    for v in (a, b, c, s):
        second += np.einsum("n,ni,nj->ij", area, v, v)
    second /= 12.0
    sigma = density * thickness  # areal density kg/m^2
    m = sigma * total_area
    com = first / total_area
    # Inertia about origin: ∫ (r·r I - r rᵀ) dm
    i_origin = sigma * (np.trace(second) * np.eye(3) - second)
    i_c = parallel_axis(i_origin, -m, com)  # shift to centroid (subtract m(d·d I - d dᵀ))
    return m, com, 0.5 * (i_c + i_c.T), total_area


# ---------------------------------------------------------------------------------------------
# Lofts
# ---------------------------------------------------------------------------------------------
def superellipse_ring(half_w: float, z_top: float, z_bot: float, n_up: float, n_lo: float,
                      angles: np.ndarray) -> np.ndarray:
    """Points (y, z) of a two-exponent superellipse section.

    ``angles`` are measured from the dorsal midline (0) towards anatomical left (+y) and on to
    the ventral midline (pi); negative angles are the right side.
    """
    zc = 0.5 * (z_top + z_bot)
    hz = 0.5 * (z_top - z_bot)
    y = np.empty_like(angles)
    z = np.empty_like(angles)
    for k, th in enumerate(angles):
        up = np.cos(th) >= 0
        n = n_up if up else n_lo
        ct, st = np.cos(th), np.sin(th)
        y[k] = half_w * np.sign(st) * abs(st) ** (2.0 / n)
        z[k] = zc + hz * np.sign(ct) * abs(ct) ** (2.0 / n)
    return np.column_stack([y, z])


def loft_grid(stations: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Grid (n_station, n_angle, 3) from station x positions and (n_angle, 2) yz rings."""
    return np.stack([np.column_stack([np.full(len(r), x), r[:, 0], r[:, 1]]) for x, r in zip(stations, rings)])


def grid_triangles(grid: np.ndarray, close_ring: bool = False) -> np.ndarray:
    """Mid-surface triangles of a structured grid (n_i, n_j, 3)."""
    ni, nj, _ = grid.shape
    tris = []
    jmax = nj if close_ring else nj - 1
    for i in range(ni - 1):
        for j in range(jmax):
            j2 = (j + 1) % nj
            p00, p01, p10, p11 = grid[i, j], grid[i, j2], grid[i + 1, j], grid[i + 1, j2]
            tris.append((p00, p10, p11))
            tris.append((p00, p11, p01))
    return np.array(tris)


def thicken(grid: np.ndarray, thickness: float, close_ring: bool = False):
    """Solid panel mesh (vertices, faces) from a mid-surface grid, offset +-t/2 along normals."""
    ni, nj, _ = grid.shape
    gi = np.gradient(grid, axis=0)
    gj = np.gradient(grid, axis=1)
    nrm = np.cross(gi, gj)
    ln = np.linalg.norm(nrm, axis=2, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-12)
    outer = grid + 0.5 * thickness * nrm
    inner = grid - 0.5 * thickness * nrm
    verts = np.concatenate([outer.reshape(-1, 3), inner.reshape(-1, 3)])
    off = ni * nj

    def vid(i, j, layer=0):
        return layer * off + i * nj + (j % nj)

    faces = []
    jmax = nj if close_ring else nj - 1
    for i in range(ni - 1):
        for j in range(jmax):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces += [(a, b, c), (a, c, d)]
            a2, b2, c2, d2 = vid(i, j, 1), vid(i + 1, j, 1), vid(i + 1, j + 1, 1), vid(i, j + 1, 1)
            faces += [(a2, c2, b2), (a2, d2, c2)]
    # edge strips
    edges = []
    for j in range(jmax):
        edges.append(((0, j), (0, j + 1)))
        edges.append(((ni - 1, j + 1), (ni - 1, j)))
    if not close_ring:
        for i in range(ni - 1):
            edges.append(((i + 1, 0), (i, 0)))
            edges.append(((i, nj - 1), (i + 1, nj - 1)))
    for (i1, j1), (i2, j2) in edges:
        a, b = vid(i1, j1), vid(i2, j2)
        a2, b2 = vid(i1, j1, 1), vid(i2, j2, 1)
        faces += [(a, b2, b), (a, a2, b2)]
    return verts, np.array(faces, dtype=int)


def convex_pieces(grid: np.ndarray, every: int = 1, close_ring: bool = False):
    """Point clouds for convex collision proxies: consecutive station pairs (whole ring)."""
    pieces = []
    ni = grid.shape[0]
    idx = list(range(0, ni, every))
    if idx[-1] != ni - 1:
        idx.append(ni - 1)
    for a, b in zip(idx, idx[1:]):
        pts = grid[a : b + 1].reshape(-1, 3)
        pieces.append(pts)
    return pieces


def polyline_frames(points: np.ndarray):
    """Tangent-aligned frames along a polyline (for sweeping sections along curves)."""
    pts = np.asarray(points, float)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    frames = []
    for k in range(len(pts)):
        frames.append(frame_from_axis(t[k], up=(0, 0, 1)))
    return frames


def sweep(points: np.ndarray, sections: list[np.ndarray], lateral_axis=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Sweep yz sections along a polyline lying (mostly) in the xz plane.

    Each section (n, 2) holds (lateral, normal) coordinates; lateral maps to ``lateral_axis``
    (+y) and normal to the in-plane normal of the path (pointing up for a forward path).
    """
    pts = np.asarray(points, float)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    lat = np.asarray(lateral_axis, float)
    out = []
    for k in range(len(pts)):
        nrm = np.cross(t[k], lat)  # for t=+x, lat=+y -> +z
        nrm /= np.linalg.norm(nrm)
        sec = sections[k]
        out.append(pts[k] + np.outer(sec[:, 0], lat) + np.outer(sec[:, 1], nrm))
    return np.stack(out)
