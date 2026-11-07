"""Covering panels for each body region, as hollow manufactured shells.

Every panel is a lofted mid-surface grid in its body frame with a wall thickness and material;
its mass and inertia come from ``geometry.shell_properties``.  Panels are the *designed*
(cosmetic, class C) volume; ``hardware`` supplies the class A envelopes the trunk must contain
and the builder widens a trunk station if the design is too small for the hardware.

Sections are "egg" superellipses: the widest point sits at ``z_wide`` (not necessarily at mid
height), with separate exponents above and below, so a trunk can have a narrow dorsal ridge,
ribs flaring below it and a keeled pubic boot, rather than a uniform oval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import template as T
from .geometry import sweep


def egg_ring(half_w: float, z_top: float, z_bot: float, z_wide: float, n_up: float, n_lo: float,
             angles: np.ndarray) -> np.ndarray:
    """(y, z) points; angle 0 = dorsal midline, +90 deg = left side, 180 deg = ventral midline."""
    y = np.empty_like(angles)
    z = np.empty_like(angles)
    for k, th in enumerate(angles):
        ct, st = np.cos(th), np.sin(th)
        if ct >= 0:
            n, h = n_up, z_top - z_wide
        else:
            n, h = n_lo, z_wide - z_bot
        y[k] = half_w * np.sign(st) * abs(st) ** (2.0 / n)
        z[k] = z_wide + h * np.sign(ct) * abs(ct) ** (2.0 / n)
    return np.column_stack([y, z])


@dataclass
class Panel:
    name: str
    body: str
    grid: np.ndarray  # (ni, nj, 3) mid-surface, body frame
    thickness: float  # m
    material: str
    region: str
    close_ring: bool = False
    collide: bool = True
    rgba: tuple = (0.45, 0.5, 0.42, 1.0)
    mounts: int = 2  # fastener/insert count


# ------------------------------------------------------------------------------------------------
# Trunk
# ------------------------------------------------------------------------------------------------
TRUNK_REGIONS = {
    # region: (x_from, x_to) in B units, list of (angle_from, angle_to) in degrees from dorsal midline
    "chest_dorsal": ((0.40, 0.97), [(-58, 58)]),
    "chest_lateral": ((0.38, 0.97), [(58, 140), (-140, -58)]),
    "chest_ventral": ((0.42, 0.95), [(140, 220)]),
    "chest_front": ((0.60, 0.97), [(-180, 180)]),
    "abdomen": ((0.18, 0.44), [(-180, 180)]),
    "pelvis_dorsal": ((-0.42, 0.22), [(-66, 66)]),
    "pelvis_lateral": ((-0.42, 0.22), [(66, 108), (-108, -66)]),
    "pubic_boot": ((0.04, 0.30), [(140, 220)]),
    # underside of the fixed tail base behind the hips (guards the tail-base battery cassette)
    "pelvis_ventral": ((-0.42, 0.04), [(108, 252)]),
}


@dataclass
class TrunkProfile:
    x: np.ndarray  # stations, m (trunk frame, unbent)
    dorsal: np.ndarray
    ventral: np.ndarray
    half_width: np.ndarray
    wide: np.ndarray  # fraction (0 = ventral, 1 = dorsal) of the widest point
    n_up: np.ndarray
    n_lo: np.ndarray

    def interp(self, xs):
        xs = np.atleast_1d(np.asarray(xs, float))

        def f(v):
            return PchipInterpolator(self.x, v)(xs)

        return f(self.dorsal), f(self.ventral), f(self.half_width), f(self.wide), f(self.n_up), f(self.n_lo)

    def at(self, x: float) -> dict:
        d, v, w, z, nu, nl = (float(a[0]) for a in self.interp([x]))
        return dict(dorsal=d, ventral=v, half=w, z_wide=v + z * (d - v), n_up=nu, n_lo=nl)


def trunk_profile(family: str, B: float, torso_scale: float) -> TrunkProfile:
    fs = T.FAMILY_SHAPE[family]
    return TrunkProfile(
        x=T.TRUNK_X * B,
        dorsal=np.array(fs["dorsal"]) * B,
        ventral=np.array(fs["ventral"]) * B,
        half_width=np.array(fs["half_width"]) * B * torso_scale,
        wide=np.array(fs["wide"], float),
        n_up=np.array(fs["n_up"], float),
        n_lo=np.array(fs["n_lo"], float),
    )


def trunk_ring(prof: TrunkProfile, x: float, angles_deg: np.ndarray) -> np.ndarray:
    s = prof.at(x)
    return egg_ring(s["half"], s["dorsal"], s["ventral"], s["z_wide"], s["n_up"], s["n_lo"], np.radians(angles_deg))


def trunk_panels(sk, family: str, prof: TrunkProfile, regions: tuple, thickness: float, material: str,
                 rgba_main, rgba_accent, n_station: int = 16, n_angle: int = 16) -> list[Panel]:
    out = []
    B = sk.B
    for region in regions:
        if region not in TRUNK_REGIONS:
            continue
        (x0, x1), spans = TRUNK_REGIONS[region]
        if region.startswith("pelvis"):  # includes pelvis_ventral
            x0 = sk.tail_root[0] / B - 0.06  # the trunk ends just behind the powered tail pivot
        xs = np.linspace(x0 * B, x1 * B, n_station)
        for (a0, a1) in spans:
            closed = (a1 - a0) >= 359.9
            angles = np.linspace(a0, a1, n_angle + (0 if closed else 1))[: n_angle if closed else None]
            rows = []
            for x in xs:
                yz = trunk_ring(prof, x, angles)
                pts = np.column_stack([np.full(len(yz), x), yz[:, 0], yz[:, 1]])
                rows.append(sk.bend_point(pts))
            grid = np.stack(rows)
            side = "" if len(spans) == 1 else ("_l" if a0 > 0 else "_r")
            rgba = rgba_accent if region in ("chest_dorsal", "pelvis_dorsal") else rgba_main
            out.append(Panel(f"{region}{side}", "trunk", grid, thickness, material, region, closed, True, rgba,
                             mounts=4 if closed else 3))
    return out


def rib_hoops(sk, prof: TrunkProfile, x_list, thickness, rgba) -> list[Panel]:
    """Hybrid family: exposed structural rib hoops (upper 3/4 of the section) over the mid-trunk."""
    out = []
    ang = np.linspace(-128, 128, 21)
    for k, xb in enumerate(x_list):
        x = xb * sk.B
        w = 0.012 * sk.B + 0.004
        rows = []
        for dx in (-w / 2, w / 2):
            yz = trunk_ring(prof, x + dx, ang)
            pts = np.column_stack([np.full(len(yz), x + dx), 1.02 * yz[:, 0], yz[:, 1] * 1.0 + 0.0])
            rows.append(sk.bend_point(pts))
        out.append(Panel(f"rib_hoop_{k}", "trunk", np.stack(rows), max(thickness * 2.2, 0.004), "AL_6061", "rib_hoop",
                         False, True, rgba, mounts=2))
    return out


# ------------------------------------------------------------------------------------------------
# Neck and skull (head body frame: origin at neck base, world-aligned at the nominal pose)
# ------------------------------------------------------------------------------------------------
SKULL_U = np.array([-0.07, 0.03, 0.14, 0.27, 0.41, 0.56, 0.72, 0.86, 0.95, 1.0])


def skull_profiles(family: str):
    """Side and plan profiles of the skull and mandible, fractions of skull length.

    Axis origin at the occipital condyle (neck joint end), u along the skull axis to the snout
    tip.  'top' and 'tooth' are heights above/below the condyle level; 'half' is half-width.
    """
    sk = T.FAMILY_SHAPE[family]["skull"]
    d = sk["depth"] / 0.44
    wr = sk["width_rear"] / 0.46
    ws = sk["width_snout"] / 0.22
    top = np.array([0.19, 0.285, 0.305, 0.285, 0.235, 0.200, 0.180, 0.168, 0.150, 0.110]) * d
    top[2] += sk["brow"] * 0.9  # postorbital boss
    top[3] += sk["brow"] * 0.4  # lacrimal horn
    tooth = -np.array([-0.02, 0.02, 0.05, 0.06, 0.062, 0.058, 0.052, 0.046, 0.040, 0.025]) * d
    half = np.array([0.22 * wr, 0.285 * wr, 0.27 * wr, 0.19 * (0.6 * wr + 0.4 * ws), 0.13 * ws, 0.112 * ws,
                     0.104 * ws, 0.098 * ws, 0.085 * ws, 0.045])
    ju = np.array([0.0, 0.10, 0.26, 0.45, 0.66, 0.86, 0.97])
    jdepth = np.array([0.24, 0.215, 0.15, 0.115, 0.095, 0.078, 0.050]) * d
    jhalf = np.array([0.235 * wr, 0.215 * wr, 0.15, 0.115 * ws, 0.100 * ws, 0.090 * ws, 0.060])
    return SKULL_U, top, tooth, half, ju, jdepth, jhalf


def _skull_frame(sk):
    a = sk.skull_pitch
    ax = np.array([np.cos(a), 0, np.sin(a)])
    up = np.array([-np.sin(a), 0, np.cos(a)])
    return ax, up


def skull_panels(sk, family: str, base_world: np.ndarray, thickness, material, rgba, rgba_jaw, rgba_teeth):
    L = sk.skull_len
    u, top, tooth, half, ju, jd, jh = skull_profiles(family)
    ax, up = _skull_frame(sk)
    occ = sk.occiput - base_world
    us = np.linspace(u[0], u[-1], 26)
    f = lambda v: PchipInterpolator(u, v)(us)  # noqa: E731
    t_, b_, h_ = f(top), f(tooth), f(half)
    angles = np.radians(np.linspace(-180, 180, 25)[:-1])
    rows = []
    for k, s in enumerate(us):
        c = occ + ax * (s * L)
        zt, zb = t_[k] * L, b_[k] * L
        ring = egg_ring(h_[k] * L, zt, zb, zb + 0.42 * (zt - zb), 2.6, 3.2, angles)
        rows.append(c + np.outer(ring[:, 0], [0, 1, 0]) + np.outer(ring[:, 1], up))
    skull = Panel("skull", "head", np.stack(rows), thickness, material, "skull", True, True, rgba, mounts=4)
    usj = np.linspace(ju[0], ju[-1], 18)
    fj = lambda v: PchipInterpolator(ju, v)(usj)  # noqa: E731
    jd_, jh_ = fj(jd), fj(jh)
    tl = PchipInterpolator(u, tooth)(usj)
    ang = np.radians(np.linspace(-180, 180, 21)[:-1])
    jrows = []
    for k, s in enumerate(usj):
        c = occ + ax * (s * L)
        z0 = tl[k] * L + 0.004 * L
        ring = egg_ring(jh_[k] * L, z0, z0 - jd_[k] * L, z0 - 0.35 * jd_[k] * L, 2.4, 2.2, ang)
        jrows.append(c + np.outer(ring[:, 0], [0, 1, 0]) + np.outer(ring[:, 1], up))
    jaw = Panel("jaw", "head", np.stack(jrows), thickness, material, "jaw", True, True, rgba_jaw, mounts=3)
    out = [skull, jaw]
    # teeth: a short row of cones along the upper tooth line (cosmetic, part of the skull shell mass)
    teeth = []
    for side in (1, -1):
        for k, s in enumerate(np.linspace(0.30, 0.94, 9)):
            hw_ = float(PchipInterpolator(u, half)(s)) * L * 0.92
            zt = float(PchipInterpolator(u, tooth)(s)) * L
            base = occ + ax * (s * L) + up * zt + np.array([0, side * hw_, 0])
            ln = (0.045 - 0.02 * abs(s - 0.55)) * L
            tip = base - up * ln
            teeth.append((base, tip))
    out.append(("teeth", teeth, rgba_teeth))
    # eyes (orbit position, high and forward-facing)
    s_eye = 0.20
    he = float(PchipInterpolator(u, half)(s_eye)) * L
    ze = float(PchipInterpolator(u, top)(s_eye)) * L
    eyes = [occ + ax * (s_eye * L) + up * (0.62 * ze) + np.array([0, sd * 0.86 * he, 0]) for sd in (1, -1)]
    out.append(("eyes", eyes, (0.08, 0.07, 0.05, 1)))
    return out


def skull_dorsal(sk, family: str, s: float):
    """World point on the skull's dorsal midline at station ``s`` (fraction of skull length from the
    occipital condyle), and the skull frame (axis, up) in the world."""
    L = sk.skull_len
    u, top, *_ = skull_profiles(family)
    ax, up = _skull_frame(sk)
    zt = float(PchipInterpolator(u, top)(s)) * L
    return sk.occiput + ax * (s * L) + up * zt, ax, up


def skull_crown_world(sk, family: str) -> float:
    L = sk.skull_len
    u, top, *_ = skull_profiles(family)
    ax, up = _skull_frame(sk)
    us = np.linspace(u[0], u[-1], 80)
    t_ = PchipInterpolator(u, top)(us)
    return float(max((sk.occiput + ax * (s * L) + up * (t * L))[2] for s, t in zip(us, t_)))


def neck_panel(sk, family: str, base_world: np.ndarray, thickness, material, rgba) -> Panel:
    """Neck from the shoulders to the occiput.  The section narrows and drops below the path near
    the head so the skull roof and temporal region stay visible above the neck."""
    B = sk.B
    L = sk.skull_len
    path = sk.neck_path - base_world
    n = len(path)
    girth = {"T": 0.90, "S": 0.98, "H": 0.94}[family]
    u = np.linspace(0, 1, n)
    half_w = ((0.125 - 0.045 * u) * B * girth)
    up = (0.100 - 0.030 * u) * B * girth
    lo = (0.165 - 0.070 * u) * B * girth
    end_up = 0.11 * L
    end_lo = 0.15 * L
    blend = np.clip((u - 0.55) / 0.45, 0, 1) ** 1.5
    up = (1 - blend) * up + blend * end_up
    lo = (1 - blend) * lo + blend * end_lo
    half_w = (1 - blend) * half_w + blend * 0.13 * L
    angles = np.radians(np.linspace(-180, 180, 21)[:-1])
    secs = [egg_ring(half_w[k], up[k], -lo[k], -0.25 * lo[k], 2.2, 2.2, angles) for k in range(n)]
    grid = sweep(path, secs)
    return Panel("neck", "head", grid, thickness, material, "neck", True, True, rgba, mounts=3)


# ------------------------------------------------------------------------------------------------
# Tail (tail body frame: origin at tail root, world-aligned at the nominal pose)
# ------------------------------------------------------------------------------------------------
def tail_profile(family: str, B: float, n: int, root_half_w: float, root_up: float, root_lo: float):
    u = np.linspace(0, 1, n)
    taper = (1 - u) ** 0.92 * (1 - 0.18 * u * u) + 0.008
    return u, root_half_w * taper, root_up * taper, root_lo * taper


def tail_panels(sk, family: str, base_world: np.ndarray, prof: TrunkProfile, thickness, material, rgba, regions):
    path = sk.tail_path - base_world
    n = len(path)
    fs = T.FAMILY_SHAPE[family]
    s0 = prof.at(sk.tail_root[0])
    scale = fs["tail_root_scale"]
    # match the trunk's posterior station at the root so the contour is continuous
    root_hw = s0["half"] * 1.0 * scale
    root_up = (s0["dorsal"] - sk.tail_root[2]) * scale
    root_lo = (sk.tail_root[2] - s0["ventral"]) * scale
    u, hw, up, lo = tail_profile(family, sk.B, n, root_hw, max(root_up, 0.02 * sk.B), max(root_lo, 0.02 * sk.B))
    angles = np.radians(np.linspace(-180, 180, 17)[:-1])
    secs = [egg_ring(hw[k], up[k], -lo[k], -0.15 * lo[k], 2.0, 2.2, angles) for k in range(n)]
    grid = sweep(path, secs, lateral_axis=(0.0, -1.0, 0.0))
    out = []
    if "tail" in regions:
        out.append(Panel("tail", "tail", grid, thickness, material, "tail", True, True, rgba, mounts=4))
    else:
        k_root = max(2, int(0.30 * n))
        out.append(Panel("tail_root", "tail", grid[: k_root + 1], thickness, material, "tail_root", True, True,
                         rgba, mounts=3))
        for i in range(k_root + 1, n - 1, 2):
            out.append(Panel(f"tail_seg_{i}", "tail", grid[i : i + 2], thickness, material, "tail_segments",
                             True, True, rgba, mounts=1))
    return out


# ------------------------------------------------------------------------------------------------
# Hip cap (roll body) and thigh fairing (thigh body)
# ------------------------------------------------------------------------------------------------
def thigh_panels(side: str, femur_axis: np.ndarray, Lf: float, B: float, family: str, r_top: float,
                 half_upper: float, s_seam: float, r_seam: float, r_knee: float, thickness, material, rgba):
    """Two-piece muscular thigh.

    Upper piece (thigh body, moves with hip pitch): a drumstick whose top is a disk centred on the
    hip-pitch axis (so it pivots in place) enclosing the hip-pitch and thigh-twist motors, with a
    rear (caudofemoral) bulge.  Lower piece (femur body, also turns with thigh twist): round
    sections about the femur axis below a circular seam, tapering to the knee.
    """
    fs = T.FAMILY_SHAPE[family]
    scale = fs["thigh_scale"]
    ax = femur_axis / np.linalg.norm(femur_axis)
    fwd = np.cross([0.0, 1.0, 0.0], ax)
    fwd = -fwd if fwd[0] < 0 else fwd
    ang = np.radians(np.linspace(-180, 180, 21)[:-1])
    # upper piece
    s0 = -r_top / Lf
    n = 18
    su = np.linspace(s0 + 0.02 * r_top / Lf, s_seam, n)
    rows = []
    for s in su:
        z = s * Lf
        if z < 0:
            circ = np.sqrt(max(r_top**2 - z**2, 1e-8))
            front, back = circ, circ * 1.06
            lat = half_upper * (circ / r_top) ** 0.35
        else:
            t = z / max(s_seam * Lf, 1e-6)
            front = r_top * (1 - t) ** 1.1 + r_seam * t
            back = (r_top * 1.12 * scale) * (1 - t) ** 0.9 + r_seam * t
            lat = half_upper * (1 - t) + r_seam * t
        ring = egg_ring(lat, front, -back, -0.1 * back, 2.4, 2.4, ang)
        rows.append(ax * z + np.outer(ring[:, 0], [0, 1, 0]) + np.outer(ring[:, 1], fwd))
    upper = Panel(f"{side}_thigh_upper", f"{side}_thigh", np.stack(rows), thickness, material, "thigh", True, True,
                  rgba, mounts=3)
    # lower piece (round sections about the femur axis so thigh twist does not open the seam)
    sl = np.linspace(s_seam - 0.03, 0.97, 12)
    rows = []
    for s in sl:
        t = (s - s_seam) / max(0.97 - s_seam, 1e-6)
        r = (r_seam - 0.004) * (1 - t) + r_knee * t
        bulge = 1.0 + 0.10 * np.sin(np.pi * np.clip(t, 0, 1)) * scale
        ring = egg_ring(r, r, -r * bulge, 0.0, 2.2, 2.2, ang)
        rows.append(ax * (s * Lf) + np.outer(ring[:, 0], [0, 1, 0]) + np.outer(ring[:, 1], fwd))
    lower = Panel(f"{side}_thigh_lower", f"{side}_femur", np.stack(rows), thickness, material, "thigh", True, True,
                  rgba, mounts=2)
    return upper, lower


def shank_panel(side: str, top_local: np.ndarray, bottom_local: np.ndarray, B: float, half_t: float,
                thickness, material, rgba) -> Panel:
    """Calf fairing over the shank-mounted ankle motors (upper shank only)."""
    axis = bottom_local - top_local
    L = np.linalg.norm(axis)
    axis /= L
    fwd = np.cross([0.0, 1.0, 0.0], axis)
    fwd = -fwd if fwd[0] < 0 else fwd
    n = 10
    s = np.linspace(-0.05, 1.0, n)
    front = 0.030 * B * np.clip(1.0 - 0.5 * s, 0.4, 1.0) + 0.006
    back = 0.080 * B * np.clip(1.1 - 0.9 * s, 0.25, 1.2) + 0.006
    ang = np.radians(np.linspace(-180, 180, 17)[:-1])
    rows = []
    for k in range(n):
        c = top_local + axis * (s[k] * L)
        ring = egg_ring(half_t * (1.0 - 0.35 * s[k]), front[k], -back[k], -0.2 * back[k], 2.3, 2.3, ang)
        rows.append(c + np.outer(ring[:, 0], [0, 1, 0]) + np.outer(ring[:, 1], fwd))
    return Panel(f"{side}_calf_fairing", f"{side}_shank", np.stack(rows), thickness, material, "shank", True, True,
                 rgba, mounts=2)
