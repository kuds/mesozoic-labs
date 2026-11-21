"""Build the anatomical T. rex's render skins on the scaled simulation skeleton.

Run from the repo root with python -m environments.tyrannosaurus_rex.scripts.build_integument.
The checked-in MJCF is self-contained. Authoring coordinates below are millimetres.
"""

from pathlib import Path

import numpy as np

from environments.shared.skin_builder import SkinBuilder
from environments.tyrannosaurus_rex.scripts.hindlimb_surface import add_hindlimbs

MODEL = Path(__file__).resolve().parents[1] / "anatomical/assets/tyrannosaurus_rex_anatomical.xml"
BEGIN = "    <!-- BEGIN render skins; edit scripts/build_integument.py and scripts/hindlimb_surface.py -->"
END = "    <!-- END render skins -->"

# Countershading and iris colour are illustrative soft-tissue choices.
PALETTE = {
    "hide": ((0.215, 0.206, 0.167), (0.660, 0.590, 0.455)),
    "face": ((0.230, 0.203, 0.162), (0.655, 0.553, 0.423)),
    "horn": ((0.128, 0.117, 0.098), (0.425, 0.374, 0.295)),
    "sole": ((0.210, 0.183, 0.154), (0.470, 0.398, 0.324)),
    "mouth": ((0.110, 0.083, 0.071), (0.285, 0.227, 0.188)),
    "eye": ((0.045, 0.040, 0.032), (0.115, 0.105, 0.083)),
    "iris": ((0.310, 0.232, 0.133), (0.622, 0.463, 0.220)),
}

# Longitudinal station, half-width, ventral limit, dorsal limit. The caudofemoral
# region deepens smoothly into the hips; the expanded rib basket narrows at the throat.
AXIAL = [
    (-1640, 1.5, 0, 3),
    (-1585, 10, -10, 13),
    (-1450, 22, -27, 25),
    (-1300, 36, -48, 28),
    (-1100, 51, -73, 45),
    (-800, 80, -106, 70),
    (-630, 101, -122, 93),
    (-500, 120, -137, 108),
    (-370, 142, -151, 133),
    (-260, 159, -155, 159),
    (-150, 181, -163, 193),
    (-45, 200, -181, 206),
    (70, 205, -209, 220),
    (195, 205, -239, 225),
    (300, 196, -231, 228),
    (385, 183, -205, 226),
    (470, 159, -162, 221),
    (545, 136, -97, 228),
    (602, 120, -25, 250),
    (657, 108, 58, 289),
    (710, 100, 121, 328),
    (763, 94, 171, 361),
    (794, 95, 190, 379),
    (817, 98, 209, 389),
]

# The posterior cheek and temporal region are broad. The nasal bridge narrows
# forward and the blunt premaxilla turns down to meet the fleshy tip of the jaw.
# The cranial landmarks follow the adult lateral/dorsal outline (Brochu 2003).
CRANIUM = [
    (84, 104, -80, 91),
    (106, 115, -58, 97),
    (132, 122, -53, 105),
    (159, 120, -56, 111),
    (185, 112, -58, 103),
    (207, 98, -60, 83),
    (236, 88, -64, 67),
    (284, 81, -73, 67),
    (338, 76, -80, 71),
    (393, 72, -76, 64),
    (441, 65, -68, 50),
    (466, 57, -60, 40),
    (480, 43, -53, 26),
    (489, 19, -41, 4),
    (490, 2, -19, -9),
]
MANDIBLE = [
    (74, 27, -74, -5),
    (103, 79, -99, -7),
    (136, 105, -113, -11),
    (165, 105, -122, -22),
    (197, 90, -117, -30),
    (239, 75, -108, -42),
    (286, 67, -106, -50),
    (343, 68, -114, -59),
    (395, 63, -111, -57),
    (439, 53, -101, -48),
    (465, 42, -86, -43),
    (478, 16, -69, -44),
    (480, 2, -58, -46),
]


def rings(profile):
    return [(x, 0, (hi + lo) / 2, width, (hi - lo) / 2) for x, width, lo, hi in profile]


def axial_weights(xyz):
    # The envelope crosses each joint as a single surface. Blending into the
    # skull at the occiput keeps the nape attached during head pitch and yaw.
    for joint, half, rear, front in (
        (-1350, 52, "tail_5", "tail_4"),
        (-1100, 61, "tail_4", "tail_3"),
        (-800, 74, "tail_3", "tail_2"),
        (-500, 82, "tail_2", "tail_1"),
        (-200, 106, "tail_1", "pelvis"),
        (550, 86, "pelvis", "neck"),
        (751, 67, "neck", "skull"),
    ):
        if xyz[0] <= joint - half:
            return {rear: 1}
        if xyz[0] < joint + half:
            f = (xyz[0] - joint + half) / (2 * half)
            f = f * f * (3 - 2 * f)
            return {rear: 1 - f, front: f}
    return {"skull": 1}


def dorsal_shade(pos, angle):
    up = np.sin(angle)
    # Low contrast, oblique flank markings stay continuous over moving joints.
    bar = (0.5 + 0.5 * np.cos(pos[0] / 56 + 2.4 * up + 0.5 * np.cos(pos[0] / 71))) ** 5
    tone = 0.55 - 0.32 * up - 0.22 * bar * max(0, up + 0.32)
    x = pos[0] - 750  # Skull-frame x.
    flank = abs(np.cos(angle))
    orbit = np.exp(-(((x - 160) / 43) ** 2) - ((up - 0.2) / 0.68) ** 2)
    snout = np.exp(-(((x - 345) / 125) ** 4))
    return tone - 0.24 * orbit * flank + 0.07 * snout * flank


def sculpt_axial_surface(skin):
    """Model the cheek, nasal ridges and shoulder muscles on the continuous axial surface."""
    surface = skin.layers[("axial", "hide")]
    for i, vertex in enumerate(surface.vertices):
        # SkinBuilder stores vertices in metres in the global bind frame.
        x, y, z = 1000 * vertex - skin.bind_positions["pelvis"]
        u, v = x - 750, z - 300  # Head coordinates.
        side = np.sign(y)
        cheek = np.exp(-(((u - 133) / 52) ** 2) - ((v + 12) / 36) ** 2)
        snout_plane = np.exp(-(((u - 275) / 70) ** 2) - ((v - 4) / 34) ** 2)
        nasal = np.exp(-(((u - 326) / 104) ** 4) - ((abs(y) - 31) / 18) ** 2) * np.clip((v - 22) / 35, 0, 1)
        shoulder = np.exp(-(((x - 424 + 0.28 * z) / 67) ** 2) - ((z - 56) / 107) ** 2)
        flank = np.clip(abs(y) / 75, 0, 1)
        vertex[1] += side * flank * (8.5 * cheek - 5.0 * snout_plane + 9 * shoulder) / 1000
        vertex[2] += 4.5 * nasal / 1000
        utex, shade = surface.texcoords[i]
        surface.texcoords[i] = (utex, np.clip(shade + 0.10 * cheek - 0.12 * snout_plane + 0.08 * nasal, 0.01, 0.99))


def add_head(skin):
    skin.tube(
        "mandible",
        "face",
        rings(MANDIBLE),
        "skull",
        axial=True,
        exponent=0.77,
        spokes=20,
        color=lambda p, t: 0.66 - 0.16 * np.sin(t),
    )
    for sign in (-1, 1):
        # Recess the small eye beneath the forward-facing rim of the broad orbit.
        for region, mat, y, rx, ry, rz, shade in (
            ("socket", "face", 109.0, 19, 10, 14, 0.23),
            ("eye", "eye", 117, 10.4, 5, 7.9, 0.30),
            ("iris", "iris", 120.6, 7.5, 2.0, 6.6, 0.60),
            ("pupil", "eye", 122.2, 4.3, 1.1, 4.6, 0.07),
        ):
            skin.ellipsoid(region, mat, "skull", (164, sign * y, 60), (rx, ry, rz), shade)
        brow = [
            (127, 102, 62, 1, 1.5),
            (137, 105, 71, 5, 4),
            (150, 112, 78, 8, 7),
            (165, 112, 76, 8, 7),
            (181, 109, 70, 8, 6),
            (196, 101, 59, 4, 3),
            (209, 91, 44, 1, 1),
        ]
        skin.tube(
            "supraorbital",
            "face",
            [(x, y * sign, z, w, h) for x, y, z, w, h in brow],
            "skull",
            color=lambda p, t: 0.41 - 0.14 * np.sin(t),
            spokes=12,
        )
        skin.ellipsoid("naris", "mouth", "skull", (448, 60.0 * sign, 22), (13.2, 3.5, 5.7), 0.28)
        # Place the lip along the lateral wall of the deep maxilla. Both jaws
        # are bound to the skull because the existing bite sensor has no jaw DOF.
        seam = [
            (141, 113, -29),
            (175, 109, -34),
            (205, 91, -38),
            (238, 79, -43),
            (285, 70, -51),
            (340, 66, -60),
            (395, 63, -56),
            (438, 56, -49),
            (467, 46, -41),
            (485, 24, -30),
        ]
        skin.tube(
            "oral_commissure",
            "mouth",
            [(x, y * sign, z, 2.8, 2.1) for x, y, z in seam],
            "skull",
            spokes=8,
            color=lambda p, t: 0.4,
        )


def add_arm(skin, side, sign):
    # The anatomical arm reads from the ventrolateral shoulder, with a tucked
    # elbow and a medially facing palm. Both existing arm bodies are welded.
    def path(region, mat, stations, shade=0.49):
        skin.tube(
            region,
            mat,
            [(x, sign * y, z, w, h) for x, y, z, w, h in stations],
            f"{side}_arm",
            spokes=12,
            color=lambda p, t: shade - 0.08 * np.sin(t),
        )

    path(
        "brachium",
        "hide",
        [
            (43, 5, 70, 8, 10),
            (44, 24, 41, 24, 30),
            (43, 38, 16, 28, 34),
            (42, 49, -6, 27, 32),
            (29, 62, -45, 21, 24),
            (20, 67, -81, 17, 19),
            (25, 68, -93, 14, 14),
        ],
    )
    path(
        "forearm",
        "hide",
        [
            (18, 67, -83, 14, 14),
            (28, 68, -94, 18, 15),
            (55, 65, -109, 13, 12),
            (83, 60, -120, 10, 11),
            (97, 54, -124, 7, 13),
        ],
    )
    for digit, y, z, length in (("I", 50, -120, 36), ("II", 57, -133, 43)):
        path(
            "fingers",
            "face",
            [
                (87, y, z, 6, 7),
                (100, y, z - 1, 5.4, 6),
                (96 + length / 2, y - 2, z - 4, 4.2, 5),
                (97 + length / 2, y - 2.6, z - 5, 3, 3),
            ],
        )
        path(
            f"manual_ungual_{digit}",
            "horn",
            [
                (94 + length / 2, y - 2, z - 4, 3.7, 4.1),
                (104 + length / 2, y - 3.5, z - 7, 3.0, 3.5),
                (98 + length, y - 7, z - 14, 0.7, 1),
                (99 + length, y - 8, z - 17, 0.3, 0.3),
            ],
            0.38,
        )


def build_integument(xml):
    skin = SkinBuilder(xml)
    # Make the nape and upper skull a continuous envelope to avoid a visible
    # sleeve seam at the back of the jaw, even when the neck is fully flexed.
    skull_offset = skin.bind_positions["skull"] - skin.bind_positions["pelvis"]
    head = [(x + skull_offset[0], w, lo + skull_offset[2], hi + skull_offset[2]) for x, w, lo, hi in CRANIUM]
    skin.tube(
        "axial",
        "hide",
        rings(AXIAL + head),
        "pelvis",
        axial=True,
        exponent=lambda p: 1 - 0.27 * np.clip((p[0] - 763) / 90, 0, 1),
        weights=axial_weights,
        color=dorsal_shade,
        spokes=36,
        subdivisions=5,
    )
    sculpt_axial_surface(skin)
    add_head(skin)
    for side, sign in (("r", -1), ("l", 1)):
        add_arm(skin, side, sign)
    add_hindlimbs(skin, xml)
    # A MuJoCo 2D builtin gradient is radial: rgb1 is at the centre and
    # rgb2 along the edges. Sample one radius so darker markings and lighter
    # undersides retain their intended order in every material.
    for layer in skin.layers.values():
        layer.texcoords = [(0.5, 0.5 * (1 - shade)) for _, shade in layer.texcoords]
    return skin.xml(PALETTE)


def main():
    xml = MODEL.read_text(encoding="utf-8")
    prefix, start, remainder = xml.partition(BEGIN)
    _, stop, suffix = remainder.partition(END)
    if not start or not stop:
        raise ValueError("Missing generated render-skin boundaries in anatomical MJCF")
    revised = prefix + start + "\n" + build_integument(xml) + "\n" + stop + suffix
    if revised != xml:
        MODEL.write_text(revised, encoding="utf-8")


if __name__ == "__main__":
    main()
