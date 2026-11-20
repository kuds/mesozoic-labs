"""Axial anatomy of the Stegosaurus reference animal, in metres and kilograms.

Stations are (x, half-width, belly z, back z) in the neutral world frame.
Surface meshes follow those landmarks; simpler internal shapes carry the inertia
and detect contact with the ground. The neck and tail respond to the walk
through damped, sprung joints.
"""

from math import asin, cos, exp, pi, sin, sqrt
from xml.etree.ElementTree import Element, SubElement

import numpy as np

# The pelvis is at x=-0.62, the shoulder at x=0.80. The long elevated tail
# drops slightly and rises towards the tip. Skull length, including beak, .375 m.
TRUNK = [
    (-1.30, 0.285, 1.112, 1.721),
    (-1.18, 0.340, 1.079, 1.815),
    (-0.99, 0.415, 0.975, 1.938),
    (-0.69, 0.482, 0.795, 2.010),
    (-0.31, 0.505, 0.650, 1.984),
    (0.06, 0.498, 0.601, 1.875),
    (0.39, 0.460, 0.580, 1.711),
    (0.67, 0.380, 0.586, 1.527),
    (0.87, 0.310, 0.631, 1.356),
    (1.04, 0.225, 0.710, 1.203),
    (1.17, 0.164, 0.755, 1.115),
]
NECK = [
    (1.00, 0.211, 0.740, 1.185),
    (1.12, 0.188, 0.742, 1.145),
    (1.29, 0.148, 0.734, 1.049),
    (1.48, 0.119, 0.702, 0.966),
    (1.64, 0.096, 0.702, 0.900),
    (1.745, 0.083, 0.690, 0.873),
]
# Upper jaw closes against a shallow dentary. The taper ahead of the orbit and
# the lower rostral tip follow the lateral and dorsal outlines of the skull.
SKULL = [
    (1.685, 0.091, 0.686, 0.881),
    (1.730, 0.097, 0.658, 0.889),
    (1.776, 0.106, 0.667, 0.900),
    (1.815, 0.101, 0.684, 0.886),
    (1.865, 0.088, 0.702, 0.860),
    (1.910, 0.071, 0.712, 0.829),
    (1.965, 0.057, 0.716, 0.799),
    (1.996, 0.050, 0.713, 0.780),
    (2.022, 0.043, 0.709, 0.763),
    (2.046, 0.030, 0.697, 0.743),
    (2.060, 0.002, 0.694, 0.706),
]
DENTARY = [
    (1.722, 0.043, 0.675, 0.752),
    (1.750, 0.080, 0.649, 0.767),
    (1.802, 0.092, 0.646, 0.763),
    (1.866, 0.082, 0.658, 0.744),
    (1.923, 0.066, 0.677, 0.736),
    (1.970, 0.054, 0.688, 0.729),
    (1.998, 0.041, 0.692, 0.718),
    (2.025, 0.018, 0.698, 0.713),
    (2.031, 0.002, 0.703, 0.707),
]
TAIL = [
    (-3.730, 0.004, 1.270, 1.280),
    (-3.555, 0.036, 1.235, 1.322),
    (-3.295, 0.054, 1.190, 1.320),
    (-3.080, 0.069, 1.158, 1.317),
    (-2.750, 0.089, 1.119, 1.342),
    (-2.440, 0.113, 1.091, 1.379),
    (-2.050, 0.145, 1.088, 1.450),
    (-1.690, 0.194, 1.100, 1.552),
    (-1.365, 0.261, 1.126, 1.674),
    (-1.180, 0.318, 1.145, 1.784),
    (-1.070, 0.328, 1.169, 1.825),
]


def _vec(values):
    return " ".join(f"{v:.5g}" for v in values)


def _mesh(asset, body, name, vertices, faces, material, uv=None, **kwargs):
    attrs = {"name": name, "vertex": _vec(np.ravel(vertices)), "face": _vec(np.ravel(faces))}
    if uv is not None:
        attrs["texcoord"] = _vec(np.ravel(uv))
    SubElement(asset, "mesh", attrs)
    return SubElement(
        body,
        "geom",
        name=name,
        type="mesh",
        mesh=name,
        material=material,
        group="1",
        mass="0",
        contype="0",
        conaffinity="0",
        **kwargs,
    )


def _materials(asset):
    # MuJoCo builds this fine stipple at load time; the file stays self-contained.
    # Its colours and markings are an artistic reconstruction.
    SubElement(
        asset,
        "texture",
        name="hide_gradient",
        type="2d",
        builtin="gradient",
        width="2048",
        height="256",
        rgb1=".65 .58 .415",
        rgb2=".375 .370 .259",
        mark="random",
        markrgb=".445 .422 .296",
        random=".28",
    )
    SubElement(
        asset, "material", name="hide", texture="hide_gradient", texuniform="false", specular=".12", shininess=".12"
    )
    colours = {
        "keratin": ".385 .326 .236",
        "jaw": ".534 .485 .347",
        "orbit": ".323 .286 .204",
        "iris": ".62 .40 .165",
        "pupil": ".045 .040 .031",
        "nostril": ".163 .142 .105",
        "spike": ".522 .456 .343",
        "plate_0": ".478 .264 .180",
        "plate_1": ".518 .291 .202",
        "plate_2": ".498 .286 .193",
        "plate_3": ".465 .275 .193",
    }
    SubElement(
        asset,
        "texture",
        name="blade_gradient",
        type="2d",
        builtin="gradient",
        width="32",
        height="128",
        rgb1="1 .97 .93",
        rgb2=".69 .61 .56",
    )
    for name, rgb in colours.items():
        mat = SubElement(asset, "material", name=name, rgba=rgb + " 1", specular=".14", shininess=".15")
        if name.startswith("plate_"):
            mat.set("texture", "blade_gradient")


def _profile(stations, x):
    """Shape-preserving cubic interpolation of transverse outlines."""
    table = np.asarray(stations)
    index = min(max(0, np.searchsorted(table[:, 0], x, side="right") - 1), len(table) - 2)
    spans = np.diff(table[:, 0])
    slopes = np.diff(table[:, 1:], axis=0) / spans[:, None]
    a, b = table[index], table[index + 1]
    h = b[0] - a[0]
    t = np.clip((x - a[0]) / h, 0, 1)

    def tangent(i):
        if i == 0 or i == len(table) - 1:
            return slopes[0 if i == 0 else -1]
        left, right = slopes[i - 1], slopes[i]
        same = left * right > 0
        result = np.zeros_like(left)
        result[same] = 2 / (1 / left[same] + 1 / right[same])
        return result

    return (
        (2 * t**3 - 3 * t**2 + 1) * a[1:]
        + (t**3 - 2 * t**2 + t) * h * tangent(index)
        + (-2 * t**3 + 3 * t**2) * b[1:]
        + (t**3 - t**2) * h * tangent(index + 1)
    )


def _loft_surface(stations, origin, *, span=None, angular=28):
    lo, hi = span or (stations[0][0], stations[-1][0])
    # Include the measured stations as well as intermediate slices.
    xs = sorted(
        {lo, hi, *(x for x, *_ in stations if lo < x < hi), *np.linspace(lo, hi, max(4, round((hi - lo) / 0.067)))}
    )
    vertices, faces, uv = [], [], []
    for i, x in enumerate(xs):
        width, bottom, top = _profile(stations, x)
        zmid, rz = (bottom + top) / 2, (top - bottom) / 2
        for j in range(angular):
            theta = 2 * pi * j / angular
            d = cos(theta)
            # Slightly fuller cheeks low on the skull; rounder down the throat.
            cheek = 1 - 0.16 * d if x > 1.70 else 1
            vertices.append((x - origin[0], width * sin(theta) * cheek, zmid + rz * d - origin[2]))
            # Map the flanks to one continuous region of the procedural texture.
            # Along the body each texel spans ~7 mm; narrower segments cover a
            # smaller range across the texture so the grain also reads on the neck.
            u = 0.50 + 0.067 * (x + 0.85) + 0.0018 * sin(10 * vertices[-1][2] + 1.3 * x) + 0.0012 * vertices[-1][1]
            bar = (0.5 + 0.5 * cos(11.5 * x + 1.1 * d)) ** 3 * exp(-(((d - 0.33) / 0.52) ** 2))
            v = 0.712 + 0.218 * (2 * asin(d) / pi) * min(1, sqrt(rz / 0.43)) + 0.075 * bar
            uv.append((u, v))
            k = (j + 1) % angular
            if i:
                before = (i - 1) * angular
                here = i * angular
                faces.extend(((before + j, here + j, here + k), (before + j, here + k, before + k)))
    for ring, reverse in ((0, False), (len(xs) - 1, True)):
        idx = ring * angular
        center = len(vertices)
        width, bottom, top = _profile(stations, xs[ring])
        vertices.append((xs[ring] - origin[0], 0, 0.5 * (bottom + top) - origin[2]))
        uv.append((0.5, 0.5))
        for j in range(angular):
            front = (idx + j, idx + (j + 1) % angular)
            faces.append((center, *(front[::-1] if reverse else front)))
    return vertices, faces, uv


def _loft(asset, body, name, stations, origin, material, span=None):
    vertices, faces, uv = _loft_surface(stations, origin, span=span)
    return _mesh(asset, body, name, vertices, faces, material, uv)


def _skin(asset, tail, neck_origin):
    """A continuous hide blends between simulated links across their pivots."""
    # Overlapping sculpt guides at the joints supply collision and plate
    # landmarks. Choose one guide at each station for a single outer surface.
    body_stations = [s for s in TAIL if s[0] < -1.30] + TRUNK + [s for s in NECK if 1.17 < s[0] < 1.685] + SKULL
    vertices, faces, uv = _loft_surface(body_stations, (0.0, 0.0, 0.0), span=(-3.73, 1.996))
    skin = SubElement(
        asset,
        "skin",
        name="axial_hide",
        group="1",
        material="hide",
        vertex=_vec(np.ravel(vertices)),
        face=_vec(np.ravel(faces)),
        texcoord=_vec(np.ravel(uv)),
    )
    bones = [(f"tail_{i + 1}", pivot) for i, (_, _, pivot) in enumerate(tail)][::-1]
    bones += [("torso", (0.0, 0.0, 1.20)), ("neck", neck_origin)]
    # Neighbour pairs share each joint across a short band. The weights form
    # a partition of unity, including through yaw and pitch in a walking pose.
    pivots = [tail[i][0] for i in (3, 2, 1, 0)] + [neck_origin[0]]
    x = np.array(vertices)[:, 0]
    right = [
        np.clip((x - (pivot - radius)) / (2 * radius), 0, 1)
        for pivot, radius in zip(pivots, (0.09, 0.11, 0.13, 0.18, 0.14))
    ]
    right = [t * t * (3 - 2 * t) for t in right]
    weights = [1 - right[0]] + [a - b for a, b in zip(right, right[1:])] + [right[-1]]
    for (name, pos), weight in zip(bones, weights):
        indices = np.flatnonzero(weight > 0.0)
        SubElement(
            skin,
            "bone",
            body=name,
            bindpos=_vec(pos),
            bindquat="1 0 0 0",
            vertid=_vec(indices),
            vertweight=_vec(weight[indices]),
        )


def _ellipsoid(body, name, centre, radii, material):
    return SubElement(
        body,
        "geom",
        name=name,
        type="ellipsoid",
        pos=_vec(centre),
        size=_vec(radii),
        material=material,
        contype="0",
        conaffinity="0",
        mass="0",
        group="1",
    )


def _proxy(body, name, centre, radii, mass):
    return SubElement(
        body,
        "geom",
        name="axial_collision_" + name,
        type="ellipsoid",
        pos=_vec(centre),
        size=_vec(radii),
        mass=str(mass),
        group="3",
        rgba="0 0 0 0",
        contype="2",
        conaffinity="1",
    )


def _site(body, name, pos):
    SubElement(body, "site", name=name, pos=_vec(pos), size=".005", rgba="0 0 0 0", group="4")


def _hinge(body, name, axis, stiffness, damping, limit):
    SubElement(
        body,
        "joint",
        name=name,
        type="hinge",
        axis=axis,
        stiffness=str(stiffness),
        damping=str(damping),
        range=f"{-limit} {limit}",
        limited="true",
    )


def _plate(asset, body, index, x, base_z, width, height, origin):
    """A thick root, tapered margins, high blade and thicker longitudinal keel."""
    sign = 1 if index % 2 else -1
    # The anterior margin faces +x; the point sweeps caudally. Small cervical
    # plates are more rounded, while large pelvic ones are broad rhomboids.
    if x > 0.85:
        # Low rounded, oval cervical plates.
        profile = [
            (-0.33, -0.08),
            (0.27, -0.08),
            (0.43, 0.12),
            (0.47, 0.40),
            (0.41, 0.65),
            (0.25, 0.84),
            (0.08, 0.976),
            (-0.08, 1),
            (-0.25, 0.88),
            (-0.41, 0.67),
            (-0.48, 0.40),
            (-0.44, 0.15),
        ]
    elif x < -1.44:
        # The distal caudal series is proportionally narrow and recurved.
        profile = [
            (-0.31, -0.08),
            (0.30, -0.08),
            (0.43, 0.13),
            (0.41, 0.34),
            (0.28, 0.59),
            (0.02, 0.82),
            (-0.27, 1),
            (-0.36, 0.98),
            (-0.38, 0.83),
            (-0.45, 0.58),
            (-0.48, 0.35),
            (-0.41, 0.12),
        ]
    else:
        # Broad trunk plates expand in their lower half and form an off-centre
        # apex. The pelvic series is especially tall and asymmetric.
        peak = -0.08 - 0.17 * exp(-(((x + 0.84) / 0.54) ** 2))
        profile = [
            (-0.29, -0.08),
            (0.28, -0.08),
            (0.41, 0.10),
            (0.50, 0.28),
            (0.48, 0.42),
            (0.36, 0.60),
            (0.17, 0.79),
            (peak + 0.06, 0.969),
            (peak, 1),
            (peak - 0.06, 0.968),
            (-0.38, 0.73),
            (-0.49, 0.49),
            (-0.50, 0.34),
            (-0.44, 0.17),
            (-0.35, 0.07),
        ]
    root_y = sign * min(0.088, 0.03 + 0.08 * height)
    verts, faces, uv = [], [], []
    polygons = [(1.0, 0.009), (0.88, 0.039), (0.58, 0.068), (0.25, 0.085)]
    for side in (-1, 1):
        start = len(verts)
        for scale, thick in polygons:
            for j, (u, v) in enumerate(profile):
                u = (u + 0.06) * scale - 0.06
                v = (v - 0.39) * scale + 0.39
                ridge = 0.008 * sin(1.9 * j + 0.3 * index) * (1 - scale)
                y = root_y + sign * (0.13 * height * v) + side * (thick + ridge) * width * (1 - 0.57 * v)
                uv.append((0.5, np.clip(0.88 - 0.7 * v - 0.08 * sin(j * 1.9) - 0.09 * (1 - scale), 0.02, 0.98)))
                verts.append((x - origin[0] + width * u, y, base_z - origin[2] + height * v))
        n = len(profile)
        for ring in range(len(polygons) - 1):
            for i in range(n):
                a, b = start + ring * n + i, start + ring * n + (i + 1) % n
                faces.extend(((a, b, b + n), (a, b + n, a + n)))
        cap = len(verts)
        uv.append((0.5, 0.46))
        verts.append(
            (
                x - origin[0] - 0.06 * width,
                root_y + sign * 0.05 * height + side * 0.083 * width,
                base_z - origin[2] + 0.39 * height,
            )
        )
        edge = start + (len(polygons) - 1) * n
        for i in range(n):
            faces.append((edge + i, edge + (i + 1) % n, cap))
        if side == 1:
            split = (len(polygons) - 1) * n * 2 + n
            faces[-split:] = [tuple(reversed(f)) for f in faces[-split:]]
    opposite = len(verts) // 2
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((j, i, i + opposite), (j, i + opposite, j + opposite)))
    _mesh(asset, body, f"plate_{index:02d}", verts, faces, f"plate_{index % 4}", uv)


def _spike(asset, body, name, base, tip):
    a, b = np.asarray(base), np.asarray(tip)
    d = b - a
    direction = d / np.linalg.norm(d)
    cross = np.cross(direction, (0.0, 0.0, 1.0))
    cross /= np.linalg.norm(cross)
    up = np.cross(direction, cross)
    vertices, faces = [], []
    # Rounded root and slight upward bow to the bony core.
    for k, (t, radius) in enumerate(((0.0, 0.038), (0.11, 0.049), (0.35, 0.043), (0.66, 0.027), (0.89, 0.011))):
        for j in range(12):
            theta = 2 * pi * j / 12
            p = a + t * d + (0, 0, 0.018 * sin(pi * t)) + radius * (cos(theta) * cross + 0.86 * sin(theta) * up)
            vertices.append(p)
            if k:
                i, ii = (k - 1) * 12 + j, (k - 1) * 12 + (j + 1) % 12
                faces.extend(((i, ii, ii + 12), (i, ii + 12, i + 12)))
    vertices.extend((a, b))
    for j in range(12):
        faces.extend(((60, (j + 1) % 12, j), (61, 48 + j, 48 + (j + 1) % 12)))
    _mesh(asset, body, name, vertices, faces, "spike")


def add_axial(asset: Element, torso: Element, actuator: Element) -> None:
    """Populate the free torso at (0,0,1.2); the axial joints are passive."""
    root = np.array((0.0, 0.0, 1.20))
    _materials(asset)
    _proxy(torso, "pelvis", (-0.66, 0, 0.18), (0.52, 0.407, 0.50), 395)
    _proxy(torso, "ribs", (-0.11, 0, 0.087), (0.66, 0.46, 0.603), 490)
    _proxy(torso, "chest", (0.55, 0, -0.10), (0.50, 0.323, 0.43), 208)

    neck_origin = np.array((1.058, 0.0, 0.956))
    neck = SubElement(torso, "body", name="neck", pos=_vec(neck_origin - root))
    _hinge(neck, "neck_pitch", "0 1 0", 5400, 215, 0.21)
    _hinge(neck, "neck_yaw", "0 0 1", 1550, 90, 0.29)
    _loft(asset, neck, "upper_beak", SKULL, neck_origin, "keratin", (1.996, 2.060))
    _loft(asset, neck, "dentary", DENTARY, neck_origin, "jaw")
    _proxy(neck, "neck", (0.26, 0, -0.088), (0.34, 0.108, 0.114), 29)
    _proxy(neck, "skull", (0.788, 0, -0.18), (0.194, 0.085, 0.10), 19)
    _site(neck, "snout", np.array((2.060, 0.0, 0.677)) - neck_origin)
    for side, tag in ((-1, "r"), (1, "l")):

        def mark(name, point, radii, material):
            p = np.array(point) - neck_origin
            p[1] *= side
            return _ellipsoid(neck, f"{tag}_{name}", p, radii, material)

        mark("orbit", (1.799, 0.087, 0.825), (0.025, 0.008, 0.014), "orbit")
        mark("iris", (1.800, 0.096, 0.825), (0.011, 0.004, 0.009), "iris")
        mark("pupil", (1.802, 0.100, 0.825), (0.0058, 0.0025, 0.0068), "pupil")
        mark("naris", (1.940, 0.059, 0.772), (0.011, 0.003, 0.0045), "nostril")

    # Lengths differ along the tail, with a stiffer, more massive base.
    tail = []
    parent, previous = torso, root
    links = [
        # pivot x,z   surface endpoints        mass    yaw spring and damping
        (-1.195, 1.46, -1.935, -1.075, 115.0, 9000, 400),
        (-1.860, 1.31, -2.685, -1.790, 48.0, 2550, 110),
        (-2.623, 1.24, -3.177, -2.541, 17.0, 460, 24),
        (-3.121, 1.24, -3.730, -3.064, 5.0, 73, 5),
    ]
    for i, (x, z, lo, hi, mass, spring, damp) in enumerate(links):
        pivot = np.array((x, 0.0, z))
        link = SubElement(parent, "body", name=f"tail_{i + 1}", pos=_vec(pivot - previous))
        _hinge(link, f"tail_{i + 1}_yaw", "0 0 1", spring, damp, 0.21)
        if i in (0, 2):
            _hinge(link, f"tail_{i + 1}_pitch", "0 1 0", spring * 4, damp * 1.5, 0.14)
        mid = (lo + hi) / 2
        radius, bottom, top = _profile(TAIL, mid)
        _proxy(
            link,
            f"tail_{i + 1}",
            (mid - x, 0, 0.5 * (bottom + top) - z),
            ((hi - lo) * 0.50, radius * 0.91, (top - bottom) * 0.46),
            mass,
        )
        tail.append((x, link, pivot))
        parent, previous = link, pivot

    _skin(asset, tail, neck_origin)

    # 18 osteoderms are preserved on Sophie, a nineteenth is inferred. Highest
    # pelvic plate is 0.785 m. Sizes and stations interpolate the fossil series.
    plates = [
        (1.548, 0.10, 0.095),
        (1.392, 0.13, 0.115),
        (1.214, 0.17, 0.155),
        (1.039, 0.21, 0.216),
        (0.857, 0.26, 0.274),
        (0.678, 0.30, 0.332),
        (0.481, 0.34, 0.386),
        (0.278, 0.39, 0.448),
        (0.059, 0.42, 0.500),
        (-0.167, 0.46, 0.545),
        (-0.397, 0.49, 0.575),
        (-0.640, 0.52, 0.639),
        (-0.889, 0.53, 0.785),
        (-1.161, 0.49, 0.559),
        (-1.416, 0.43, 0.488),
        (-1.698, 0.38, 0.408),
        (-1.985, 0.31, 0.305),
        (-2.242, 0.245, 0.222),
        (-2.492, 0.175, 0.150),
    ]
    for i, (x, w, h) in enumerate(plates, 1):
        if x > neck_origin[0]:
            body, origin, surface = neck, neck_origin, NECK
        elif x > tail[0][0]:
            body, origin, surface = torso, root, TRUNK
        else:
            _, body, origin = [item for item in tail if x <= item[0]][-1]
            surface = TAIL
        width, bottom, top = _profile(surface, x)
        # The implantation follows the actual elliptical dorsal flank.
        y = min(0.088, 0.03 + 0.08 * h)
        base_z = (top + bottom) / 2 + 0.5 * (top - bottom) * np.sqrt(1 - (y / width) ** 2) - 0.018
        _plate(asset, body, i, x, base_z, w, h, origin)
        if i == 13:
            _site(body, "plate_apex", (x - 0.24 * w - origin[0], y + 0.13 * h, base_z + h - origin[2]))

    # Distributed armour contributes ~35 kg in addition to 1,321 kg soft
    # tissue and bone. The above link proxy masses include 5 kg of that armour.
    # Remaining plate mass is distributed over the trunk with a visual-free
    # internal inertial proxy centered below the tall pelvic plate bases.
    SubElement(
        torso,
        "geom",
        name="trunk_armour_inertia",
        type="ellipsoid",
        pos="-.43 0 .88",
        size=".85 .075 .20",
        mass="30",
        contype="0",
        conaffinity="0",
        group="3",
        rgba="0 0 0 0",
    )
    for pair, (x, dx, dy, dz) in enumerate(((-3.095, -0.135, 0.251, 0.236), (-3.360, -0.178, 0.242, 0.213)), 1):
        _, body, origin = [item for item in tail if x <= item[0]][-1]
        radius, bottom, top = _profile(TAIL, x)
        for side in (-1, 1):
            a = np.array((x, side * 0.50 * radius, 0.5 * (bottom + top))) - origin
            _spike(asset, body, f"caudal_spike_{pair}_{'l' if side > 0 else 'r'}", a, a + np.array((dx, side * dy, dz)))
    _site(tail[-1][1], "tail_tip", np.array((-3.730, 0.0, 1.275)) - tail[-1][2])
