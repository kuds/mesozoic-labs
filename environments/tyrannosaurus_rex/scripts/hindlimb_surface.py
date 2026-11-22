"""Articulated hindlimb surfaces for the scaled T. rex. Dimensions are in millimetres."""

from xml.etree import ElementTree as ET

import numpy as np


def _blend(z, centre, half, upper, lower):
    t = np.clip((centre + half - z) / (2 * half), 0, 1)
    t = t * t * (3 - 2 * t)
    return {upper: 1 - t, lower: t}


def _digit(skin, digit, bone, endpoint, radius):
    # Each mobile digit has one passive hinge in the simulation. Model the fleshy
    # pads and tapered keratin sheath around that digit's own collision capsule.
    r = radius
    pad_count = {2: 2, 3: 3, 4: 4}[digit]
    stations = []
    for t in np.linspace(-0.06, 0.91, 21):
        # Folds between pads sit above the load surface. A slight distal arch
        # leaves the claw tip clear when the digital pads lie on a level floor.
        u = np.clip(t, 0, 1)
        lobes = (1 - np.cos(2 * np.pi * u * pad_count / 0.90)) / 2
        width = r * (1.02 - 0.48 * u + 0.10 * lobes)
        bottom = r * (-0.84 - 0.12 * lobes + 0.32 * u**3)
        top = r * (0.74 - 0.48 * u + 0.12 * lobes)
        x, y, _ = t * endpoint
        stations.append((x, y, (bottom + top) / 2, width, (top - bottom) / 2))

    def scute(p, angle):
        # Darker interscale creases along the dorsal face of the digit.
        stripe = (0.5 + 0.5 * np.cos(2 * np.pi * p[0] / 12)) ** 6
        return 0.52 + 0.11 * np.sin(angle) - 0.15 * stripe * max(0, np.sin(angle))

    skin.tube(f"digit_{digit}", "sole", stations, bone, spokes=14, subdivisions=1, color=scute)
    length = np.linalg.norm(endpoint[:2])
    claw = []
    for t, z, w, h in [
        (0.84, -0.14, 0.46, 0.38),
        (0.93, -0.10, 0.43, 0.36),
        (0.99, -0.13, 0.34, 0.30),
        (1 + 0.55 * r / length, -0.31, 0.20, 0.19),
        (1 + 0.86 * r / length, -0.54, 0.03, 0.03),
    ]:
        x, y, _ = t * endpoint
        claw.append((x, y, r * z, r * w, r * h))
    skin.tube(f"ungual_{digit}", "horn", claw, bone, spokes=12, color=lambda p, a: 0.42 + 0.19 * np.sin(a))


def add_hindlimbs(skin, xml):
    """Place the flesh and foot sheaths around the joint and contact locations."""
    tree = ET.fromstring(xml)
    for side, outward in [("r", -1), ("l", 1)]:
        thigh, shin, mt = (f"{side}_{part}" for part in ("thigh", "tibia", "metatarsus"))
        femur = skin.bind_positions[shin] - skin.bind_positions[thigh]
        tibia = skin.bind_positions[mt] - skin.bind_positions[shin]
        ball = skin.bind_positions[f"{side}_toe_d3"] - skin.bind_positions[mt]
        # Broad proximal thigh, bulky upper calf, narrow tendinous lower calf,
        # then a shallow prominence behind the hock. Continuous at both joints.
        sections = [
            (t * femur, w, h, dx)
            for t, w, h, dx in [
                (-0.43, 8, 15, -10),
                (-0.30, 66, 116, -25),
                (-0.08, 92, 154, -25),
                (0.18, 97, 159, -29),
                (0.42, 83, 126, -21),
                (0.69, 62, 90, -10),
                (0.90, 53, 65, 3),
                (1.0, 51, 56, 4),
            ]
        ]
        sections.extend(
            (femur + t * tibia, w, h, dx)
            for t, w, h, dx in [
                (0.13, 54, 67, -7),
                (0.33, 55, 74, -15),
                (0.53, 45, 62, -15),
                (0.72, 33, 41, -10),
                (0.86, 25, 31, -7),
                (0.92, 20, 24, -4),
            ]
        )
        rows = [(v[0] + dx, v[1], v[2], w, h) for v, w, h, dx in sections]
        skin.tube(
            "hindlimb",
            "hide",
            rows,
            thigh,
            spokes=20,
            weights=lambda p: _blend(p[2], femur[2], 64, thigh, shin),
            color=lambda p, a: 0.58 - 0.14 * np.cos(a) * outward + 0.04 * np.cos(p[2] / 27),
        )
        distal = [
            (t * tibia, w, h, dx)
            for t, w, h, dx in [(0.79, 15, 21, -8), (0.86, 24, 31, -7), (0.94, 28, 37, -9), (1.0, 30, 34, -5)]
        ]
        distal.extend(
            (tibia + t * ball, w, h, dx)
            for t, w, h, dx in [
                (0.11, 27, 29, -2),
                (0.30, 22, 22, 0),
                (0.61, 21, 20, 0),
                (0.82, 27, 24, 1),
                (0.94, 40, 29, 5),
                (1.00, 42, 19, 12),
            ]
        )
        skin.tube(
            "hock",
            "sole",
            [(v[0] + dx, v[1], v[2], w, h) for v, w, h, dx in distal],
            shin,
            spokes=16,
            weights=lambda p: _blend(p[2], tibia[2], 36, shin, mt),
            color=lambda p, a: 0.61 - 0.08 * (0.5 + 0.5 * np.cos(p[2] / 3)) ** 4,
        )
        skin.ellipsoid("metatarsal_pad", "sole", mt, ball + np.array([38, 0, -6]), (68, 47, 21), 0.49)

        # Digit I sits on the medial metatarsus, above the track-making toes.
        # Its free tip points medially and slightly forward.
        medial = -outward
        base = 0.64 * ball
        hallux = [
            (base[0] + x, medial * y, base[2] + z, w, h)
            for x, y, z, w, h in [(0, 10, 0, 9, 11), (0, 24, -8, 9, 10), (4, 36, -16, 8, 9), (12, 46, -20, 6, 7)]
        ]
        skin.tube("hallux", "sole", hallux, mt, spokes=12)
        claw = [
            (base[0] + x, medial * y, base[2] + z, w, h)
            for x, y, z, w, h in [
                (10, 43, -20, 6, 6),
                (15, 51, -20, 5, 5),
                (22, 58, -24, 3, 3),
                (26, 62, -30, 0.6, 0.7),
            ]
        ]
        skin.tube("hallux_claw", "horn", claw, mt, spokes=10)
        for digit in (2, 3, 4):
            bone = f"{side}_toe_d{digit}"
            capsule = tree.find(f".//geom[@name='{bone}_geom']")
            assert capsule is not None
            start, end = np.fromstring(capsule.attrib["fromto"], sep=" ").reshape(2, 3) * 1000
            _digit(skin, digit, bone, end - start, float(capsule.attrib["size"]) * 1000)
