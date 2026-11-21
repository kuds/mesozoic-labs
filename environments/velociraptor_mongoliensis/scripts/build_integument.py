"""Author the anatomical specimen's articulated render surface in its unrotated bind pose.

Dimensions below are millimetres relative to the named bones. The physics model is
self-contained. Run this file to rebuild only the delimited visual asset block.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

# Preserve invocation by filename from a source checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from environments.shared.skin_builder import SkinBuilder  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "anatomical/assets/velociraptor_mongoliensis_anatomical.xml"
BEGIN = "    <!-- BEGIN generated integument; edit scripts/build_integument.py -->"
END = "    <!-- END generated integument -->"


PALETTE = {
    "mantle": ((0.19, 0.145, 0.11), (0.70, 0.59, 0.43)),
    "face": ((0.23, 0.185, 0.145), (0.67, 0.55, 0.39)),
    "hindlimb": ((0.22, 0.16, 0.115), (0.57, 0.45, 0.30)),
    "scale": ((0.27, 0.235, 0.19), (0.62, 0.52, 0.37)),
    "horn": ((0.12, 0.113, 0.105), (0.41, 0.36, 0.28)),
    "vane": ((0.135, 0.125, 0.115), (0.53, 0.455, 0.33)),
    "covert": ((0.205, 0.17, 0.125), (0.60, 0.50, 0.35)),
    "mask": ((0.16, 0.14, 0.13), (0.41, 0.335, 0.25)),
    "mouth": ((0.10, 0.086, 0.076), (0.23, 0.19, 0.16)),
    "eye": ((0.072, 0.064, 0.045), (0.13, 0.11, 0.08)),
    "iris": ((0.34, 0.24, 0.13), (0.60, 0.44, 0.23)),
}

# A graded bind across each caudal joint and the cervicodorsal joint yields one continuous profile.
BOUNDARIES = [
    (-703, "tail_5", "tail_4", 23),
    (-483, "tail_4", "tail_3", 29),
    (-274, "tail_3", "tail_2", 28),
    (-173, "tail_2", "tail_1", 31),
    (-68, "tail_1", "pelvis", 46),
    (370, "pelvis", "neck", 40),
]

# (station along x, lateral half-width, ventral z, dorsal z), including the curved neck.
AXIAL_PROFILE = [
    (-894, 1.2, 29, 32),
    (-863, 6.5, 23, 37),
    (-774, 12, 17, 43),
    (-703, 15, 13, 47),
    (-600, 19, 10, 51),
    (-483, 23, 4, 58),
    (-380, 28, -3, 65),
    (-274, 35, -12, 72),
    (-173, 43, -23, 81),
    (-109, 51, -39, 91),
    (-68, 60, -59, 97),
    (-15, 69, -85, 97),
    (56, 71, -105, 94),
    (130, 76, -117, 90),
    (211, 78, -116, 89),
    (270, 74, -104, 92),
    (314, 66, -84, 94),
    (344, 59, -62, 93),
    (370, 50, -37, 98),
    (394, 41, -12, 109),
    (416, 35, 24, 127),
    (443, 32, 78, 148),
    (469, 29, 115, 165),
    (507, 25, 139, 184),
    (546, 23, 157, 198),
    (583, 23, 172, 214),
    (604, 24.5, 175, 220),
    (621, 23, 171, 226),
    (636, 17, 180, 226),
    (649, 5, 198, 216),
]

# Narrow nasal bridge and shallow dentary. The lip seam follows the maxillary margin.
SKULL_PROFILE = [
    (-35, 4, -2, 11),
    (-25, 13, -12, 17),
    (-15, 22, -20, 21),
    (-1, 29, -27, 28),
    (15, 34, -30, 33),
    (30, 37, -33, 36),
    (46, 38, -35, 36),
    (60, 37, -35, 35),
    (78, 31, -35, 28),
    (93, 23, -35, 17),
    (113, 17, -37, 6),
    (140, 12.0, -42, -4),
    (163, 11.5, -43, -12),
    (188, 11, -42, -13),
    (207, 10, -40, -11),
    (221, 8, -40, -12),
    (229, 3, -35, -21),
    (230, 0.7, -31, -28),
]


def axial_weights(p):
    x = p[0]
    for b, a, c, half in BOUNDARIES:
        if x < b - half:
            return {a: 1}
        if x <= b + half:
            t = (x - b + half) / (2 * half)
            t = t * t * (3 - 2 * t)
            return {a: 1 - t, c: t}
    return {"neck": 1}


def mantle_color(p, t):
    x = p[0]
    up = np.sin(t)
    flank = abs(np.cos(t))
    bars = (0.5 + 0.5 * np.cos(x / 13.5 + 2.5 * up)) ** 3
    return 0.64 - 0.32 * up - 0.24 * bars * max(0.0, up + 0.15) * flank + 0.04 * np.cos(x / 9 + 3.5 * up) * flank


def axial_stations(profile):
    return [(x, 0, (upper + lower) / 2, w, (upper - lower) / 2) for x, w, lower, upper in profile]


def segments(bone_vector, stations):
    """Place cross sections at fractions of the proximal-to-distal bone vector."""
    return [(*np.array(bone_vector) * t, width, depth) for t, width, depth in stations]


def leg_color(p, t):
    return 0.43 + 0.17 * np.sin(t) + 0.06 * np.cos(p[2] / 12 + 1.2 * np.sin(t))


def blend_at(z, joint_z, radius, upper_bone, lower_bone):
    t = float(np.clip((joint_z + radius - z) / (2 * radius), 0, 1))
    t = t * t * (3 - 2 * t)
    return {upper_bone: 1 - t, lower_bone: t}


class Integument(SkinBuilder):
    # One feather is a tapered, cambered vane with a narrower inner web. Shafts and
    # alternate near-tip bars are pigment variations across the same smooth surface.
    def feather(self, region, mat, body, start, end, edge, outward, width, tone, barring=True):
        p = np.array(start, float)
        delta = np.array(end, float) - p
        length = np.linalg.norm(delta)
        forward = delta / length
        edge = np.array(edge, float)
        edge -= forward * np.dot(edge, forward)
        edge /= np.linalg.norm(edge)
        normal = np.cross(forward, edge)
        if np.dot(normal, outward) < 0:
            edge = -edge
            normal = -normal
        s = self.surface(region, mat)
        first = len(s.vertices)
        for t in (0, 0.13, 0.29, 0.47, 0.62, 0.73, 0.84, 0.93, 1):
            w = width * np.interp(t, [0, 0.15, 0.4, 0.7, 1], [0.14, 0.71, 1, 0.86, 0.025])
            centre = (
                self.bind_positions[body]
                + p
                + delta * t
                + normal * 3.0 * np.sin(np.pi * t)
                + edge * (0.07 * length * t * t)
            )
            for u in (-1, -0.42, 0, 0.13, 0.48, 1):
                physical = (0.76 if u < 0 else 1.10) * w * u
                xyz = centre + physical * edge + normal * (1 - abs(u)) * 0.18 * w
                band = 0.16 if barring and 0.69 <= t <= 0.77 else 0
                shade = tone + 0.08 * (1 - t) + 0.13 * (abs(u) > 0.9) + 0.14 * (u == 0) - band - 0.09 * (t >= 0.92)
                self.add_point(s, xyz, {body: 1}, shade)
        for i in range(8):
            for j in range(5):
                k = first + i * 6 + j
                s.faces.extend(((k, k + 6, k + 7), (k, k + 7, k + 1)))

    # Individual contour feather tips follow the underlying breast, nape and mantle;
    # their tips project caudally and lie flatter than the exposed remiges.
    def contour_tract(self, region, mat, bone, sections, tracts, weight=None):
        prof = PchipInterpolator([r[0] for r in sections], np.array(sections)[:, 1:], axis=0)
        for row, (start, stop, spacing, theta, angular_span, length) in enumerate(tracts):
            for idx, x in enumerate(np.arange(start, stop, spacing)):
                theta0 = theta + 0.075 * np.sin(4.1 * idx + 0.72 * row)
                for side in (-1, 1):
                    surface = self.surface(region, mat)
                    first = len(surface.vertices)
                    for u in (0, 0.24, 0.54, 0.78, 0.93, 1):
                        broad = np.interp(u, [0, 0.24, 0.54, 0.78, 0.93, 1], [0.30, 0.82, 1, 0.69, 0.3, 0.02])
                        for v in (-1, -0.5, 0, 0.5, 1):
                            xp = x - length * u + 2.1 * v * u
                            angle = theta0 + 0.22 * u + angular_span * broad * v
                            ry, lo, hi = prof(xp)
                            mid = (lo + hi) * 0.5
                            rz = (hi - lo) * 0.5
                            lift = (0.3 + 0.85 * u) * (1 - 0.25 * abs(v))
                            pos = np.array([xp, side * (ry + lift) * np.cos(angle), mid + (rz + lift) * np.sin(angle)])
                            # A fine dark margin and brighter shaft, barred only on some feather tips.
                            ground = mantle_color(pos, angle) if bone == "pelvis" else 0.48 - 0.18 * np.sin(angle)
                            tone = (
                                ground
                                + 0.045 * np.sin(2.7 * idx + row)
                                - 0.045 * abs(v)
                                + 0.038 * (v == 0)
                                - 0.05 * (u >= 0.78)
                            )
                            self.add_point(
                                surface, pos + self.bind_positions[bone], weight(pos) if weight else {bone: 1}, tone
                            )
                    for a in range(5):
                        for b in range(4):
                            k = first + a * 5 + b
                            for tri in ((k, k + 5, k + 6), (k, k + 6, k + 1)):
                                surface.faces.append(tri if side > 0 else tri[::-1])

    def add_axial_surface(self):
        self.tube(
            "axial",
            "mantle",
            axial_stations(AXIAL_PROFILE),
            "pelvis",
            axial=True,
            weights=axial_weights,
            color=mantle_color,
            spokes=24,
            subdivisions=3,
        )

    def add_head(self):
        self.tube(
            "head",
            "face",
            axial_stations(SKULL_PROFILE),
            "skull",
            axial=True,
            exponent=0.85,
            spokes=24,
            color=lambda p, t: 0.48 - 0.18 * np.sin(t) - 0.15 * np.exp(-(((p[0] - 77) / 40) ** 2)) * abs(np.cos(t)),
        )
        jaw = [
            (-32, 3, -12, -6),
            (-22, 10, -24, -12),
            (-12, 18, -33, -17),
            (2, 25, -39, -20),
            (24, 32, -46, -21),
            (43, 32, -49, -24),
            (65, 28, -50, -25),
            (89, 21.5, -48, -28),
            (112, 16, -49, -30),
            (144, 12, -52, -35),
            (173, 10, -52, -37),
            (196, 8.5, -49, -36),
            (213, 7, -46, -32),
            (225, 3, -40, -31),
            (228, 0.7, -36, -32),
        ]
        self.tube(
            "jaw",
            "face",
            axial_stations(jaw),
            "skull",
            axial=True,
            exponent=0.85,
            color=lambda p, t: 0.69 - 0.19 * np.sin(t),
        )
        for side in (-1, 1):
            # Lateral eye sits in the rear half of the orbit, caudal to the low snout.
            for material, y, rx, ry, rz, c in [
                ("face", 34.3, 16, 2.6, 12.4, 0.33),
                ("eye", 35.6, 10.0, 3.5, 8.1, 0.22),
                ("iris", 37.3, 7.8, 2.2, 6.8, 0.76),
                ("eye", 38.8, 4.05, 1.25, 4.75, 0.12),
            ]:
                self.ellipsoid("orbit", material, "skull", (58, y * side, 9), (rx, ry, rz), c)
            lid = [
                (44, 33.5, 8, 1.3, 1.2),
                (47, 35.3, 13, 2, 1.8),
                (54, 36.8, 17.1, 2.3, 2.2),
                (60, 36.5, 17.5, 2.1, 1.7),
                (66, 35.5, 15.1, 1.9, 1.5),
                (71, 33.7, 11.3, 0.45, 0.58),
            ]
            self.tube(
                "lid",
                "face",
                [(x, y * side, z, w, h) for x, y, z, w, h in lid],
                "skull",
                spokes=10,
                color=lambda p, t: 0.36,
            )
            self.ellipsoid("narial_opening", "mouth", "skull", (203, 9.2 * side, -19.5), (5.1, 1.3, 2.2), 0.38)
            # Preserve a thin, continuous closed gape; centre it on the outer mesh edges.
            points = [
                (33, 34, -28),
                (53, 34.3, -26),
                (72, 31.7, -26),
                (91, 22.2, -28),
                (111, 17.1, -32),
                (141, 12.0, -36),
                (166, 11.3, -39),
                (188, 10.3, -39),
                (209, 9, -37),
                (222, 6, -34),
            ]
            self.tube(
                "lip", "mouth", [(x, side * y, z, 0.75, 0.85) for x, y, z in points], "skull", spokes=8, subdivisions=2
            )

    def add_forelimb(self, side, label):
        """Folded manus and remiges. Quill knobs on the ulna support the long secondaries."""
        arm = f"{label}_arm"
        fore = f"{label}_forearm"
        hand = f"{label}_hand"
        upper = np.array([-134.187, side * 12, -87.142])
        ulna = np.array([116.924, -side * 5, -26.994])

        shoulder_sections = segments(
            upper,
            [
                (-0.18, 4, 5),
                (-0.07, 13, 17),
                (0.03, 23, 26),
                (0.36, 24, 33),
                (0.7, 20, 25),
                (0.96, 18, 18),
                (1.07, 12, 14),
            ],
        )
        shoulder_sections = [(x, y - side * max(0, 21 + x * 0.28), z, w, h) for x, y, z, w, h in shoulder_sections]
        self.tube("upper_arm", "mantle", shoulder_sections, arm, color=lambda p, t: 0.49 - 0.19 * np.sin(t))
        for j, t in enumerate(np.linspace(0.04, 0.91, 9)):
            p = upper * t + np.array([0, side * (13 + 7 * t), 2])
            end = p + [-68 - 17 * t, side * (9 + 7 * t), -21 - 13 * t]
            self.feather(
                "humeral_coverts",
                "covert",
                arm,
                p,
                end,
                [0, 0, 1],
                [0, side, 0],
                13.5,
                0.48 + 0.08 * np.cos(j * 2.1),
                False,
            )
        self.tube(
            "forearm",
            "covert",
            segments(ulna, [(-0.09, 12, 14), (0.1, 16, 20), (0.44, 16, 19), (0.8, 13, 16), (1, 12, 13), (1.07, 8, 9)]),
            fore,
        )
        # Distinct manual rays fold back alongside the body in the parasagittal plane.
        self.tube(
            "hand",
            "scale",
            [
                (0, 0, 6, 10, 12),
                (1, side * 1, -8, 11, 13),
                (5, side * 2, -30, 9, 12),
                (5, side * 2, -53, 7, 11),
                (4, side * 3, -65, 5, 7),
            ],
            hand,
        )
        for ordinal, length, xb, yb in [(1, 75, 12, -1), (2, 113, 1, 2), (3, 94, -9, 4)]:
            points = [
                (xb, side * yb, -49, 4.8, 5),
                (xb + 2, side * (yb + 1), -58 - length * 0.18, 4.2, 4.5),
                (xb + (6 if ordinal == 1 else 1), side * yb, -55 - length * 0.48, 3.6, 4.1),
                (xb - 2, side * (yb - 1), -54 - length * 0.76, 2.9, 3.1),
            ]
            self.tube("fingers", "scale", points, hand, spokes=10)
            x, y, z, *_ = points[-1]
            self.tube(
                "manual_claws",
                "horn",
                [
                    (x, y, z + 2, 2.8, 3),
                    (x - 1, y, z - 4, 2.7, 3.9),
                    (x - 5, y, z - 11, 2, 3),
                    (x - 12, y, z - 16, 1.3, 1.5),
                    (x - 17, y, z - 17, 0.25, 0.36),
                ],
                hand,
                spokes=10,
            )
        for j, t in enumerate(np.linspace(0.04, 0.96, 12)):
            p = ulna * t + np.array([0, side * (10 + 4 * t), 3])
            end = p + [-114 - 40 * t, side * (24 + 3 * t), -64 + 7 * t + 9 * np.sin(t * np.pi)]
            self.feather(
                "secondaries", "vane", fore, p, end, [0, 0, 1], [0, side, 0], 16.5, 0.42 + 0.06 * np.cos(2.5 * j)
            )
        for j, t in enumerate(np.linspace(0.08, 0.95, 8)):
            p = np.array([4, side * (6 + 4 * t), -7 - 85 * t])
            end = p + [-132 - 85 * t, side * (22 + 7 * t), -41 + 59 * t]
            self.feather(
                "primaries",
                "vane",
                hand,
                p,
                end,
                [0, 0, 1],
                [0, side, 0],
                16.7 - 1.6 * t,
                0.34 + 0.04 * np.cos(2.1 * j),
            )
        for j, t in enumerate(np.linspace(0.03, 1, 12)):
            p = ulna * t + [2, side * 15, 6]
            end = p + [-72 - 15 * t, side * 21, -34 + 7 * t]
            self.feather(
                "coverts", "covert", fore, p, end, [0, 0, 1], [0, side, 0], 15, 0.60 + 0.08 * np.cos(j * 2.6), False
            )
        # Short, more rounded upper coverts fill the fold between wrist and elbow.
        for j, t in enumerate(np.linspace(0.07, 0.93, 8)):
            p = np.array([4, side * 15, -7 - 84 * t])
            end = p + [-80 - 48 * t, side * (21 + 3 * t), -32 + 38 * t]
            self.feather(
                "hand_coverts",
                "covert",
                hand,
                p,
                end,
                [0, 0, 1],
                [0, side, 0],
                16.4,
                0.55 + 0.07 * np.cos(2.4 * j),
                False,
            )

    def add_tail_fringe(self, side):
        # Taper the exposed end of the tail with a modest paired fringe. Feather
        # layout is inferred from relatives; no species-specific fan plan is known.
        for j, t in enumerate(np.linspace(0.08, 0.96, 9)):
            p = np.array([-180 * t, side * (7 - 3 * t), -2])
            end = p + [-47 - 20 * t, side * (22 * (1 - t) + 5), 1]
            self.feather(
                "tail_fringe", "vane", "tail_5", p, end, [0, side, 0], [0, 0, 1], 9 - 3 * t, 0.43 + 0.03 * (j % 2)
            )

    def add_hindlimb(self, side, label):
        """Reconstruct one digitigrade leg around the measured hip, knee, hock and MTP positions."""
        fem = np.array([136.511, side * 13.600, -194.483])
        tib = np.array([-87.215, side * 16.715, -239.038])
        leg = f"{label}_thigh"
        shin = f"{label}_tibia"
        mt = f"{label}_metatarsus"

        # Shared surface through the knee and hock, with bind weights distributed
        # across the soft tissue on both sides of each joint centre.
        thigh = [
            (-0.26, 5, 8),
            (-0.16, 26, 40),
            (0.02, 35, 52),
            (0.21, 44, 67),
            (0.43, 39, 56),
            (0.66, 31, 39),
            (0.84, 24, 30),
            (0.97, 19, 22),
        ]
        calf = [
            (0.02, 18, 22),
            (0.12, 21, 26),
            (0.27, 23, 32),
            (0.44, 21, 28),
            (0.61, 16, 20),
            (0.72, 13, 16),
            (0.79, 10, 12),
            (0.84, 7, 10),
        ]
        upper = segments(fem, thigh) + [(*(fem + t * tib), w, h) for t, w, h in calf]
        self.tube(
            "thigh",
            "hindlimb",
            upper,
            leg,
            subdivisions=3,
            weights=lambda p: blend_at(p[2], fem[2], 25, leg, shin),
            color=leg_color,
        )
        met = np.array([14.648, side * 6.837, -97.773])
        hock = segments(tib, [(0.70, 9, 11), (0.77, 12, 14), (0.88, 10, 13), (0.97, 11, 14), (1, 12, 14)])
        hock.extend(
            [
                (*(tib + t * met), w, h)
                for t, w, h in [
                    (0.055, 11.3, 12.5),
                    (0.22, 9, 10),
                    (0.53, 8, 9),
                    (0.80, 10, 10),
                    (0.96, 14, 12),
                    (1.05, 14, 9),
                ]
            ]
        )
        self.tube(
            "metatarsal",
            "scale",
            hock,
            shin,
            subdivisions=3,
            weights=lambda p: blend_at(p[2], tib[2], 13, shin, mt),
            color=lambda p, t: 0.51 + 0.1 * np.cos(p[2] / 3),
        )
        # The MTP pad shares the original collision bottom at 15 mm below its centre.
        self.ellipsoid("ball", "scale", mt, (5, side * 12, -100.7), (15, 19.5, 15), 0.46)
        self.tube(
            "hallux",
            "scale",
            [
                (2, side * -6, -43, 3.7, 4),
                (6, side * -10, -51, 3.8, 4.1),
                (13, side * -15, -57, 3, 3.4),
                (19, side * -17, -58, 2.2, 2.5),
            ],
            mt,
            spokes=10,
        )
        self.tube(
            "hallux_claw",
            "horn",
            [
                (18, side * -17, -57, 2.2, 2.7),
                (23, side * -18, -58, 2.0, 2.5),
                (27, side * -18, -61, 1.2, 1.8),
                (29, side * -18, -65, 0.2, 0.3),
            ],
            mt,
            spokes=10,
        )
        digit = f"{label}_toe_d2"
        self.tube(
            "digit_II",
            "scale",
            [
                (-3, 0, 2, 5.4, 6.4),
                (1, 0, -2, 6.8, 7.6),
                (13, 0, -14, 5.2, 5.7),
                (19, -side * 1, -19, 7, 7),
                (25, -side * 2, -14, 5.8, 6.7),
                (34, -side * 3, -8, 5.5, 6.2),
                (43, -side * 4, -5, 6, 7),
            ],
            digit,
            spokes=12,
        )
        # A recurved blade with a 55 mm end-to-end chord. Tip follows the existing
        # site exactly; the independently moving prey collider runs along that chord.
        blade = [
            (-1, 0, -2, 4.7, 7),
            (0, 0, 2, 5.5, 9),
            (4, -side * 1, 14, 4.8, 9.7),
            (8, -side * 2, 27, 4.0, 7.2),
            (15, -side * 3.3, 39.5, 3.3, 5.0),
            (23, -side * 4.6, 45.2, 2.1, 3),
            (29.5, -side * 5.7, 45.3, 1.3, 1.7),
            (34.483, -side * 6.601, 42.336, 0.15, 0.22),
        ]
        self.tube(
            "ungual_II", "horn", blade, f"{label}_toe_claw", color=lambda p, t: 0.38 + 0.21 * np.sin(t), spokes=14
        )
        for dig, yaw, basez, length, pads in [
            (3, -4, -9.5, 115, (35, 62, 85, 104)),
            (4, 20, -16.9, 105, (12, 31, 54, 75, 94)),
        ]:
            body = f"{label}_toe_d{dig}"
            phi = np.deg2rad(yaw)
            dx = np.cos(phi)
            dy = side * np.sin(phi)
            curve = [(-2, 1, 5.2, 5.2), (2, -0.9, 7, 6.4)]
            for k, a in enumerate(pads):
                # Match the bottom of the loaded pads and leave small flexion creases
                # above it, with the raised section of III-1 clear of the surface.
                rw = (8.3 if dig == 3 else 7.7) - 0.72 * k
                curve.extend(
                    [
                        (a, basez + 0.3, rw, 8.55),
                        (a + 3.4, basez + 0.55, rw, 8.8),
                        (a + 8.2, basez - 0.55, rw * 0.77, 7.85),
                    ]
                )
            curve.extend([(length - 0.7, basez + 0.3, 4.6, 6.9), (length + 2, basez + 0.4, 3.1, 4.1)])
            self.tube(
                f"digit_{dig}",
                "scale",
                [(a * dx, a * dy, z, w, h) for a, z, w, h in curve],
                body,
                color=lambda p, t: 0.55 - 0.12 * max(0, np.sin(t)) + 0.15 * np.cos(p[0] / 2.5 + 0.7 * np.sin(t)),
                spokes=12,
            )
            tip = 129 if dig == 3 else 125
            claw = [
                (length - 5, basez + 0.4, 4.8, 4.8),
                (length, basez + 1.4, 4.2, 4.1),
                ((length + tip) / 2, basez + 0.8, 2.9, 3),
                (tip - 2, basez - 0.7, 1.5, 1.7),
                (tip, basez - 1.5, 0.2, 0.3),
            ]
            self.tube(f"ungual_{dig}", "horn", [(a * dx, a * dy, z, w, h) for a, z, w, h in claw], body, spokes=12)

    def add_contour_plumage(self):
        self.contour_tract(
            "scapulars",
            "mantle",
            "pelvis",
            AXIAL_PROFILE,
            [
                (-75, 375, 26, 0.79, 0.19, 51),
                (-56, 365, 25, 0.49, 0.20, 53),
                (-72, 354, 25, 0.18, 0.21, 54),
                (-32, 337, 26, -0.11, 0.20, 45),
                (48, 308, 24, -0.39, 0.16, 38),
            ],
            axial_weights,
        )
        self.contour_tract(
            "neck_feathers",
            "mantle",
            "pelvis",
            AXIAL_PROFILE,
            [(398, 616, 21, 0.94, 0.25, 32), (408, 612, 21, 0.55, 0.25, 33), (417, 611, 23, 0.13, 0.22, 31)],
            axial_weights,
        )
        self.contour_tract(
            "tail_coverts",
            "covert",
            "pelvis",
            AXIAL_PROFILE,
            [(-730, -80, 31, 0.77, 0.31, 48), (-720, -84, 36, 0.22, 0.26, 44)],
            axial_weights,
        )
        self.contour_tract(
            "cheek_feathers",
            "face",
            "skull",
            SKULL_PROFILE,
            [(0, 59, 12, 0.69, 0.14, 15), (7, 53, 13, 0.29, 0.12, 15), (-3, 46, 13, 0.02, 0.15, 13)],
        )

    def add_temporal_streak(self):
        # A soft dark orbital streak follows the outer temporal profile and fades at
        # the lacrimal. The fundus, lid and shallow nare remain separate elevations.
        cranial_profile = PchipInterpolator([r[0] for r in SKULL_PROFILE], np.array(SKULL_PROFILE)[:, 1:], axis=0)
        for side in (-1, 1):
            mesh = self.surface("temporal_streak", "mask")
            first = len(mesh.vertices)
            for x, theta, span in [
                (-13, 0.43, 0.01),
                (-3, 0.32, 0.12),
                (10, 0.29, 0.17),
                (26, 0.26, 0.18),
                (43, 0.23, 0.2),
                (55, 0.29, 0.22),
                (70, 0.34, 0.25),
                (84, 0.38, 0.16),
                (103, 0.47, 0.09),
                (119, 0.52, 0.01),
            ]:
                ry, lo, hi = cranial_profile(x)
                zmid = (lo + hi) * 0.5
                for v in [-1, -0.55, 0, 0.55, 1]:
                    a = theta + span * v
                    # Use the same superellipse as the skull surface.
                    cp, sp = np.cos(a), np.sin(a)
                    pos = [
                        x,
                        side * (ry + 0.72) * np.sign(cp) * abs(cp) ** 0.85,
                        zmid + (hi - zmid + 0.72) * np.sign(sp) * abs(sp) ** 0.85,
                    ]
                    self.add_point(
                        mesh, np.array(pos) + self.bind_positions["skull"], {"skull": 1}, 0.53 + 0.24 * abs(v)
                    )
            for row in range(9):
                for col in range(4):
                    k = first + 5 * row + col
                    for tri in ((k, k + 1, k + 6), (k, k + 6, k + 5)):
                        mesh.faces.append(tri if side > 0 else tri[::-1])


def build_integument(xml: str) -> str:
    skin = Integument(xml)
    skin.add_axial_surface()
    skin.add_head()
    for side, label in ((-1, "r"), (1, "l")):
        skin.add_forelimb(side, label)
        skin.add_tail_fringe(side)
        skin.add_hindlimb(side, label)
    skin.add_contour_plumage()
    skin.add_temporal_streak()
    return skin.xml(PALETTE, "Articulated visual surfaces. UV shade varies with plumage tract and contour.")


def main() -> None:
    source = MODEL.read_text(encoding="utf-8")
    prefix, start, remainder = source.partition(BEGIN)
    _, stop, suffix = remainder.partition(END)
    if not start or not stop:
        raise ValueError("Could not find the delimited integument in the anatomical MJCF.")
    revised = prefix + start + "\n" + build_integument(source) + "\n" + stop + suffix
    if revised != source:
        MODEL.write_text(revised, encoding="utf-8")


if __name__ == "__main__":
    main()
