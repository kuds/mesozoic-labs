"""Paired manus/pes and specimen-scaled limb segments, in metres and kilograms."""

from __future__ import annotations

import math
from xml.etree import ElementTree as ET

import numpy as np

# Hip pitch, knee and ankle angles are zero at the built-in standing pose.
LEGS = ("lf", "rf", "lh", "rh")
JOINTS = ("roll", "pitch", "flex", "ankle")
ROOT_HEIGHT = 1.20
HOOF = ".255 .24 .19 1"
VISUAL = {"contype": "0", "conaffinity": "0", "mass": "0", "group": "2"}
SOLE = {
    "contype": "2",
    "conaffinity": "1",
    "condim": "3",
    "mass": "0",
    "group": "3",
    "rgba": ".28 .26 .2 1",
    "friction": "1 .008 .001",
}


def _vector(values):
    return " ".join(f"{float(v):.7g}" for v in values)


def morphology(leg):
    fore = leg.endswith("f")
    sign = 1 if leg.startswith("l") else -1
    if fore:
        return (0.80, sign * 0.255, 0.95), (0.450, 0.312), 0.216, -1
    return (-0.62, sign * 0.30, 1.49), (0.868, 0.498), 0.228, 1


def link_angles(x, down, lengths, bend):
    """Absolute proximal and distal inclinations from vertical toward the nose."""
    a, b = lengths
    dist = max(1e-8, math.hypot(x, down))
    cosine = np.clip((dist**2 + a**2 - b**2) / (2 * dist * a), -1, 1)
    proximal = math.atan2(x, down) + bend * math.acos(cosine)
    distal = math.atan2(x - a * math.sin(proximal), down - a * math.cos(proximal))
    return proximal, distal


def _outline(leg, vectors, height):
    """Rings follow the joint centres, then flare into a broad, flattened sole."""
    fore = leg.endswith("f")
    upper, lower = vectors
    wrist = upper + lower
    r = 0.105 if fore else 0.176
    s = 0.071 if fore else 0.096
    rings = []
    # Each cross section blends continuously between the adjoining links.
    for t, rx, ry, front, blend in (
        (-0.12, 0.68, 0.78, 0, 0),
        (0.04, 1.06, 0.97, 0.03, 0),
        (0.24, 1.15, 1.0, 0.1, 0),
        (0.50, 1.02, 0.85, 0.035, 0),
        (0.75, 0.81, 0.74, 0, 0.03),
        (0.89, 0.71, 0.65, 0, 0.17),
        (1.0, 0.68, 0.62, 0, 0.48),
    ):
        rings.append((upper * t + [r * front, 0, 0], rx * r, ry * r, (1 - blend, blend, 0), 0))
    for t, rx, ry, blend_upper, blend_foot in (
        (0.12, 1.10, 1.02, 0.13, 0),
        (0.28, 1.14, 1.01, 0.01, 0),
        (0.50, 0.98, 0.94, 0, 0),
        (0.73, 0.84, 0.87, 0, 0.02),
        (0.90, 0.83, 0.85, 0, 0.18),
        (1.0, 0.86, 0.90, 0, 0.49),
    ):
        rings.append((upper + lower * t, rx * s, ry * s, (blend_upper, 1 - blend_upper - blend_foot, blend_foot), 0))
    if fore:
        # Upright metacarpus above a short crescent. The hind edge hollows
        # beneath the centre of the metacarpal arcade.
        foot = (
            (0.82, -0.004, 0.062, 0.069, 0.86, 0),
            (0.63, 0.001, 0.065, 0.077, 1, 0.1),
            (0.45, 0.009, 0.074, 0.09, 1, 0.28),
            (0.31, 0.015, 0.088, 0.11, 1, 0.65),
            (0.21, 0.017, 0.096, 0.125, 1, 0.93),
            (0.12, 0.019, 0.097, 0.127, 1, 1),
            (0.047, 0.020, 0.088, 0.114, 1, 1),
            (0.015, 0.021, 0.077, 0.096, 1, 1),
            (0.008, 0.028, 0.032, 0.068, 1, 1),
        )
    else:
        # Hock transitions into the cushion behind the metatarsus and three
        # short digital lobes at the front of the pes.
        foot = (
            (0.82, -0.016, 0.079, 0.082, 0.86, 0),
            (0.63, -0.017, 0.098, 0.088, 1, 0.08),
            (0.46, -0.002, 0.128, 0.102, 1, 0.27),
            (0.33, 0.010, 0.154, 0.115, 1, 0.58),
            (0.22, 0.016, 0.18, 0.131, 1, 0.92),
            (0.12, 0.015, 0.185, 0.134, 1, 1),
            (0.05, 0.013, 0.17, 0.127, 1, 1),
            (0.016, 0.012, 0.15, 0.11, 1, 1),
            (0.008, 0.032, 0.093, 0.056, 1, 1),
        )
    for h, dx, rx, ry, weight, shape in foot:
        rings.append((wrist + [dx, 0, (h - 1) * height], rx, ry, (0, 1 - weight, weight), shape))
    return rings


def _pad_centres(fore):
    if fore:
        return ((0.012, -0.077), (0.051, 0), (0.012, 0.077))
    return ((-0.082, 0), (0.108, -0.087), (0.133, 0), (0.108, 0.087))


def _plantar_height(x, y, centres):
    # Keep the centre of the cushion flat and roll the unsupported rim up.
    # Positive signed distance is outside a side of the pad support polygon.
    outside = False
    distance = math.inf
    for (a, b), (c, d) in zip(centres, centres[1:] + centres[:1], strict=True):
        dx, dy = c - a, d - b
        outside |= dy * (x - a) > dx * (y - b)
        along = np.clip(((x - a) * dx + (y - b) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        distance = min(distance, math.hypot(x - a - along * dx, y - b - along * dy))
    return 0.0018 + (0.14 * distance if outside else 0.0)


def _limb_skin(asset, leg, origin, vectors, height):
    """One weighted surface over the knee, hock/carpus and foot cushions."""
    n = 48
    orig = np.array(origin)
    vertices, faces, uv = [], [], []
    bindings = [[] for _ in range(3)]
    sections = _outline(leg, vectors, height)
    centres = _pad_centres(leg.endswith("f"))
    foot_origin = orig + vectors[0] + vectors[1]
    for j, (center, rx, ry, weights, shape) in enumerate(sections):
        for i in range(n):
            angle = 2 * math.pi * i / n
            x, y = rx * math.cos(angle), ry * math.sin(angle)
            if shape and leg.endswith("f"):
                x += shape * 0.56 * max(-x, 0) * math.exp(-((y / 0.058) ** 2))
                # Metacarpals I–V contribute shallow relief around the front
                # of the hand rather than five long, separate toes.
                x += shape * max(x, 0) * 0.075 * math.cos(2 * math.pi * y / 0.046)
            elif shape and x > 0:
                # Three short digital lobes, a broader toe end and a tapered heel.
                x += shape * max(x, 0) * 0.10 * math.cos(2 * math.pi * y / 0.083)
                y *= 1 + shape * 0.16 * math.cos(angle)
            point = orig + center + (x, y, 0)
            if point[2] < 0.025:
                local_x, local_y = point[:2] - foot_origin[:2]
                surface = _plantar_height(local_x, local_y, centres)
                if leg.endswith("h"):
                    # The loaded heel cushion compresses furthest at touchdown.
                    surface += 0.0025 * np.clip((0.055 - local_x) / 0.095, 0.0, 1.0)
                point[2] = max(point[2], surface)
            vertices.extend(point)
            # Sample the same hide as the adjoining flank. Orient its grain
            # down the limb and around the cushion without a circumferential seam.
            dx, dy, dz = point - orig
            uv.extend(
                (0.50 + 0.067 * (orig[0] + 0.85) + 0.063 * dz + 0.021 * dx, 0.695 + 0.40 * dx + 0.18 * dy - 0.015 * dz)
            )
            k = j * n + i
            for b, w in enumerate(weights):
                if w > 0:
                    bindings[b].append((k, w))
            if j:
                a = (j - 1) * n + i
                b = (j - 1) * n + (i + 1) % n
                faces.extend((a, b + n, b, a, a + n, b + n))
    bottom = (len(sections) - 1) * n
    for i in range(1, n - 1):
        faces.extend((0, i, i + 1, bottom, bottom + i + 1, bottom + i))
    skin = ET.SubElement(
        asset,
        "skin",
        name=leg + "_hide",
        material="hide",
        texcoord=_vector(uv),
        group="2",
        vertex=_vector(vertices),
        face=" ".join(map(str, faces)),
    )
    bind_pos = (orig, orig + vectors[0], orig + vectors[0] + vectors[1])
    for name, pos, bind in zip(("upper", "lower", "foot"), bind_pos, bindings, strict=True):
        indices, weights = zip(*bind, strict=True)
        ET.SubElement(
            skin,
            "bone",
            body=leg + "_" + name,
            bindpos=_vector(pos),
            bindquat="1 0 0 0",
            vertid=" ".join(map(str, indices)),
            vertweight=_vector(weights),
        )


def _nail_mesh(asset, name, length, width, height):
    """Broad ungual with a nearly planar sole and gently beveled keratin crown."""
    # Perimeter at the rear, shoulder and blunt front of the nail.
    yz = (
        (-1, 0.15),
        (-0.85, 0.70),
        (-0.48, 0.95),
        (0, 1),
        (0.48, 0.95),
        (0.85, 0.70),
        (1, 0.15),
        (0.78, 0),
        (0.30, -0.025),
        (-0.30, -0.025),
        (-0.78, 0),
    )
    points = []
    faces = []
    for j, (x, ry, rz) in enumerate(((0, 0.72, 0.65), (0.34, 1, 1), (0.72, 0.97, 0.86), (1, 0.68, 0.48))):
        for i, (y, z) in enumerate(yz):
            points.extend((x * length, y * width * ry, z * height * rz))
            if j:
                a = (j - 1) * len(yz) + i
                b = (j - 1) * len(yz) + (i + 1) % len(yz)
                faces.extend((a, b + len(yz), b, a, a + len(yz), b + len(yz)))
    top = 3 * len(yz)
    for i in range(1, len(yz) - 1):
        faces.extend((0, i, i + 1, top, top + i + 1, top + i))
    ET.SubElement(asset, "mesh", name=name, vertex=_vector(points), face=" ".join(map(str, faces)))


def _inertia(body, mass, pos, length, radius):
    # Longitudinal axes are close to vertical in the reference pose.
    transverse = mass * (length**2 / 12 + radius**2 / 4)
    axial = mass * radius**2 / 2
    ET.SubElement(
        body, "inertial", pos=_vector(pos), mass=str(mass), diaginertia=_vector((transverse, transverse, axial))
    )


def _sole_pad(body, name, pos, size):
    ET.SubElement(body, "geom", SOLE, type="ellipsoid", name=name, pos=_vector(pos), size=_vector(size))


def _foot(asset, body, leg, height):
    fore = leg.endswith("f")
    name = "manual_nail" if fore else "pedal_hoof"
    if leg.startswith("l"):
        _nail_mesh(asset, name, 0.040 if fore else 0.072, 0.021 if fore else 0.04, 0.030 if fore else 0.054)
    if fore:
        for i, (x, y) in enumerate(_pad_centres(fore)):
            _sole_pad(body, f"{leg}_pad{i}", (x, y, 0.040 - height), (0.061, 0.049, 0.040))
        medial = -1 if leg.startswith("l") else 1
        for i, (x, y) in enumerate(((0.071, medial * 0.080), (0.093, medial * 0.029))):
            ET.SubElement(
                body,
                "geom",
                VISUAL,
                name=f"{leg}_nail{i}",
                type="mesh",
                mesh=name,
                pos=_vector((x, y, 0.014 - height)),
                rgba=HOOF,
            )
    else:
        heel, *digits = _pad_centres(fore)
        _sole_pad(body, leg + "_heel", (*heel, 0.055 - height), (0.095, 0.099, 0.055))
        for i, (x, y) in enumerate(digits):
            _sole_pad(body, f"{leg}_toe{i}", (x, y, 0.046 - height), (0.083, 0.052, 0.046))
            # A short broad nail emerges obliquely from each digital lobe.
            ET.SubElement(
                body,
                "geom",
                VISUAL,
                name=f"{leg}_ungual{i}",
                type="mesh",
                mesh=name,
                pos=_vector((x + 0.019, y, 0.011 - height)),
                euler=_vector((0, 0.035, 0.22 * y / 0.087)),
                rgba=HOOF,
            )
    ET.SubElement(body, "site", name=leg + "_sole", pos=_vector((0, 0, -height)), size=".005", group="4")


def add_limbs(torso, actuator, asset):
    """Append the four serial leg chains and torque motors to an MJCF tree."""
    for leg in LEGS:
        fore = leg.endswith("f")
        (x, y, z), lengths, height, bend = morphology(leg)
        angles = link_angles(0, z - height, lengths, bend)
        vectors = [np.array((math.sin(a), 0, -math.cos(a))) * length for a, length in zip(angles, lengths, strict=True)]
        upper = ET.SubElement(torso, "body", name=leg + "_upper", pos=_vector((x, y, z - ROOT_HEIGHT)))
        lower = ET.SubElement(upper, "body", name=leg + "_lower", pos=_vector(vectors[0]))
        foot = ET.SubElement(lower, "body", name=leg + "_foot", pos=_vector(vectors[1]))
        masses = (16, 6, 3) if fore else (54, 17, 6)
        radii = (0.098, 0.068) if fore else (0.175, 0.102)
        for i, body in enumerate((upper, lower)):
            a, b = 0.08 * vectors[i], 0.94 * vectors[i]
            ET.SubElement(
                body,
                "geom",
                SOLE,
                name=body.get("name") + "_guard",
                type="capsule",
                fromto=_vector((*a, *b)),
                size=str(0.66 * radii[i]),
            )
            _inertia(body, masses[i], 0.44 * vectors[i], lengths[i], radii[i])
        _limb_skin(asset, leg, (x, y, z), vectors, height)
        _inertia(foot, masses[2], (0.01, 0, -0.12), 0.21, 0.11)
        _foot(asset, foot, leg, height)
        # Flexion is measured relative to the bent resting limb. Positive motor
        # rotation around +y carries a downward-pointing bone toward the tail.
        limit = (-1.5, 0.50) if fore else (-0.50, 1.6)
        rotations = (
            (upper, "roll", "1 0 0", (-0.55, 0.55)),
            (upper, "pitch", "0 1 0", (-0.95, 0.95)),
            (lower, "flex", "0 1 0", limit),
            (foot, "ankle", "0 1 0", (-1.2, 1.2)),
        )
        for body, joint, axis, joint_range in rotations:
            name = f"{leg}_{joint}"
            ET.SubElement(
                body,
                "joint",
                name=name,
                type="hinge",
                axis=axis,
                range=_vector(joint_range),
                damping="9",
                armature=".3",
            )
            max_torque = 1800 if fore else 4200
            ET.SubElement(
                actuator,
                "motor",
                name=name,
                joint=name,
                ctrllimited="true",
                ctrlrange=_vector((-max_torque, max_torque)),
            )
