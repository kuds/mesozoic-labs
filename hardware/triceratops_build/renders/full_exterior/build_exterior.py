"""Build a portable presentation MJCF around the selected 60 cm capsule-foot robot.

All sculpture coordinates below are millimetres in the named robot body's frame.
The manufacturing envelope and skin mass must be established on physical parts.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "sim/generated/trike60_C_closed_capsule.xml"
TAU = 2 * math.pi


def vec(values):
    return " ".join(f"{v:.7g}" for v in values)


def mm(point):
    return vec(np.asarray(point) / 1000)


def palette(out):
    """Continuous UV fields: countershading, broken dorsal bars, scale edges, frill radial markings."""
    rng = np.random.default_rng(378)
    h, w = 512, 768
    v, u = np.mgrid[0:h, 0:w]
    u, v = (u + 0.5) / w, 1 - (v + 0.5) / h
    seeds = rng.random((12500, 2))
    dist, idx = cKDTree(seeds, boxsize=1).query(np.stack([u, 1 - v], axis=-1), k=2)
    gap = dist[..., 1] - dist[..., 0]
    scales = rng.normal(0, 4.0, len(seeds))[idx[..., 0]] + 3 - 13 * np.exp(-gap / 0.00053)
    speckle = rng.normal(0, 1.8, (h, w))
    grain = (scales + speckle)[..., None]
    dorsal = (0.5 + 0.5 * np.cos(TAU * v))[..., None]

    def blend(a, b, t):
        return np.asarray(a) * (1 - t) + np.asarray(b) * t

    ripple = 0.016 * np.sin(u * 57 + np.sin(v * 22)) + 0.012 * np.sin(v * 34 + u * 13)
    crossbars = np.clip((np.cos(TAU * (7 * u + 0.42 * np.cos(TAU * v) + ripple)) - 0.2) * 1.3, 0, 1)
    mottled = (3 * np.sin(u * 47 + v * 12) + 2 * np.sin(v * 89 - u * 24))[..., None]
    belly = (181, 150, 99)
    back = (91, 103, 88)
    hide = blend(belly, back, np.clip(dorsal * 1.55, 0, 1))
    hide = blend(hide, (60, 72, 65), crossbars[..., None] * dorsal**1.9 * 0.75)
    shoulder = np.exp(-(((u - 0.82) / 0.16) ** 2))[..., None]
    hide = blend(hide, (155, 126, 77), shoulder * dorsal * 0.18)
    limb = blend((112, 115, 86), (69, 84, 66), u[..., None] ** 1.4)
    limb += 5 * np.cos((u + 0.027 * np.cos(v * TAU)) * TAU * 4)[..., None]
    face = blend((159, 140, 95), (92, 106, 78), np.clip(1.45 * dorsal, 0, 1))
    face += 3.4 * np.sin((u + np.cos(v * TAU) * 0.25) * TAU * 4)[..., None]
    # The frill has its own UV sheet: a solid shield with a copper fan and dark edge.
    fx, fy = (u - 0.5) * 2, (v - 0.5) * 2
    r = np.sqrt(fx * fx + fy * fy)
    phi = np.arctan2(fx, fy)
    frill = blend((101, 110, 78), (169, 98, 63), np.clip((r - 0.27) * 1.65, 0, 1)[..., None])
    rays = np.clip((np.cos(phi * 16 + 0.35 * np.sin(11 * r)) - 0.35) * 1.25, 0, 1) * np.clip((r - 0.30) * 2.5, 0, 1)
    frill = blend(frill, (211, 155, 87), (0.66 * rays)[..., None])
    edge = np.clip((r - 0.865) * 10, 0, 1)
    frill = blend(frill, (76, 79, 56), (0.86 * edge)[..., None])
    for x0 in [-0.54, 0.54]:
        mark = np.sqrt(((fx - x0) / 0.19) ** 2 + ((fy - 0.32) / 0.27) ** 2)
        frill = blend(frill, (94, 72, 55), (0.64 * np.exp(-(((mark - 0.75) / 0.22) ** 2)))[..., None])
        frill = blend(frill, (209, 150, 77), (0.42 * np.exp(-((mark / 0.57) ** 2)))[..., None])
    horn = blend((122, 111, 79), (232, 216, 170), u[..., None] ** 0.75)
    horn += (4 * np.sin(TAU * (v * 13 + u * 0.4)) + 7 * np.sin(TAU * u * 16) * np.exp(-3 * u))[..., None]
    for name, color, strength in [
        ("hide", hide, 1),
        ("limb", limb, 0.92),
        ("face", face, 0.85),
        ("frill", frill, 0.82),
        ("ivory", horn, 0.45),
    ]:
        rgb = np.uint8(np.clip(color + strength * (grain + mottled), 0, 255))
        Image.fromarray(rgb).save(out / f"{name}.png")


class Exterior:
    def __init__(self, source, out):
        self.root = ET.parse(source).getroot()
        self.out = out
        self.assets = self.root.find("asset")
        self.world = self.root.find("worldbody")
        self.chassis = self.body("chassis")
        # Default viewer hides group 5. Original material alpha, sensor sites, and
        # collision masks remain intact. The presentation layer is group 2.
        for g in self.chassis.iter("geom"):
            g.set("group", "5")
        for name in ("hide", "limb", "face", "frill", "ivory"):
            ET.SubElement(self.assets, "texture", name="paint_" + name, type="2d", file=f"{name}.png")
            ET.SubElement(
                self.assets,
                "material",
                name="finish_" + name,
                texture="paint_" + name,
                rgba="1 1 1 1",
                specular=".16",
                shininess=".12",
            )
        for name, color in {
            "fold": ".32 .35 .26",
            "hornroot": ".39 .36 .23",
            "beak": ".26 .25 .19",
            "lowerbeak": ".40 .36 .25",
            "mouth": ".18 .105 .10",
            "tongue": ".44 .24 .21",
            "iris": ".68 .42 .13",
            "pupil": ".045 .046 .033",
            "glint": ".92 .88 .70",
            "lid": ".36 .39 .27",
            "nail": ".47 .44 .32",
            "sole": ".23 .25 .19",
            "rim": ".57 .48 .32",
            "boss": ".43 .44 .29",
        }.items():
            glossy = name in {"iris", "pupil", "glint"}
            ET.SubElement(
                self.assets,
                "material",
                name="finish_" + name,
                rgba=color + " 1",
                specular=".55" if glossy else ".16",
                shininess=".45" if glossy else ".12",
            )
        self.studio()

    def body(self, name):
        elem = self.world.find(f".//body[@name='{name}']")
        if elem is None:
            raise ValueError(f"missing source body {name}")
        return elem

    def studio(self):
        v = self.root.find("visual")
        v.find("global").attrib.update(offwidth="1600", offheight="1000", fovy="35")
        v.find("headlight").attrib.update(ambient=".45 .44 .42", diffuse=".27 .26 .24", specular=".13 .13 .12")
        v.find("map").attrib.update(shadowclip="6", fogstart="4", fogend="12")
        ET.SubElement(v, "quality", shadowsize="4096", offsamples="4")
        self.assets.find("texture[@name='sky']").attrib.update(rgb1=".54 .59 .56", rgb2=".79 .79 .71")
        floor = self.assets.find("material[@name='floor_mat']")
        floor.attrib.clear()
        floor.attrib.update(name="floor_mat", rgba=".63 .62 .55 1", specular=".015", shininess=".01")
        for light in self.world.findall("light"):
            self.world.remove(light)
        base = dict(type="directional")
        for pos, direction, color, shadow in (
            ("1.4 -2.4 3.5", "-.35 .50 -.90", ".54 .52 .48", "true"),
            ("-.9 1.5 1.6", ".30 -.48 -.7", ".42 .45 .49", "false"),
            ("-.4 -.8 .9", ".2 .65 -.6", ".19 .20 .16", "false"),
        ):
            ET.SubElement(
                self.world,
                "light",
                pos=pos,
                dir=direction,
                diffuse=color,
                specular=".12 .12 .10",
                castshadow=shadow,
                **base,
            )

    def geom(self, body, name, mat, **attrs):
        return ET.SubElement(
            body,
            "geom",
            name="exterior_" + name,
            material="finish_" + mat,
            group="2",
            mass="0",
            contype="0",
            conaffinity="0",
            **attrs,
        )

    def oval(self, body, name, mat, pos, size, **attrs):
        return self.geom(body, name, mat, type="ellipsoid", pos=mm(pos), size=mm(size), **attrs)

    def mesh(self, body, name, mat, vertices, faces, uv, faceuv=None):
        # An OBJ gives independent UV indices at the longitudinal seam.
        p = self.out / (name + ".obj")
        with p.open("w") as f:
            for z in vertices:
                f.write("v " + mm(z) + "\n")
            for z in uv:
                f.write("vt " + vec(z) + "\n")
            for a, b in zip(faces, faces if faceuv is None else faceuv):
                f.write("f " + " ".join(f"{i + 1}/{j + 1}" for i, j in zip(a, b)) + "\n")
        ET.SubElement(self.assets, "mesh", name="shape_" + name, file=p.name, smoothnormal="true")
        self.geom(body, name, mat, type="mesh", mesh="shape_" + name)

    def loft(self, body, name, mat, rings, axis="x", sides=32):
        """Smooth cross-sections. x axis rows=(x,y,z,half-width,half-height);
        descending z axis rows=(x,y,z,fore-aft radius,lateral radius)."""
        rings = np.array(rings, float)
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rings[:, :3], axis=0), axis=1))]
        params = np.linspace(0, distance[-1], max(18, 3 * len(rings)))
        profiles = PchipInterpolator(distance, rings, axis=0)(params)
        verts = []
        uv = []
        faces = []
        fuv = []
        for i, (x, y, z, a, b) in enumerate(profiles):
            for j in range(sides):
                ang = TAU * j / sides
                verts.append(
                    (x, y + a * np.sin(ang), z + b * np.cos(ang))
                    if axis == "x"
                    else (x + a * np.cos(ang), y + b * np.sin(ang), z)
                )
            uv.extend((i / (len(profiles) - 1), j / sides) for j in range(sides + 1))
        for i in range(len(profiles) - 1):
            for j in range(sides):
                a = i * sides + j
                b = i * sides + (j + 1) % sides
                c = b + sides
                d = a + sides
                k = i * (sides + 1) + j
                z = k + sides + 1
                faces.extend(((a, d, c), (a, c, b)))
                fuv.extend(((k, z, z + 1), (k, z + 1, k + 1)))
        for i in (0, len(profiles) - 1):
            k = i * sides
            t = i * (sides + 1)
            for j in range(1, sides - 1):
                face = (k, k + j, k + j + 1)
                ft = (t, t + j, t + j + 1)
                if i == len(profiles) - 1:
                    face, ft = face[::-1], ft[::-1]
                faces.append(face)
                fuv.append(ft)
        self.mesh(body, name, mat, verts, faces, uv, fuv)

    def trunk(self):
        self.loft(
            self.chassis,
            "torso",
            "hide",
            [
                (-308, 0, 15, 24, 22),
                (-287, 0, 18, 79, 69),
                (-257, 0, 23, 110, 92),
                (-197, 0, 26, 116, 109),
                (-122, 0, 20, 118, 112),
                (-50, 0, 13, 121, 106),
                (18, 0, 10, 126, 117),
                (104, 0, 14, 127, 128),
                (185, 0, 22, 121, 128),
                (229, 0, 32, 111, 119),
                (260, 0, 43, 100, 94),
                (283, 0, 51, 86, 76),
                (308, 0, 58, 70, 57),
                (336, 0, 61, 39, 38),
            ],
            sides=52,
        )
        # The tail's parent is the actual yaw/pitch body. Its root wraps the pelvic bay.
        self.loft(
            self.body("tail"),
            "tail",
            "hide",
            [
                (-388, 0, -72, 1.4, 1.4),
                (-361, 0, -72, 5, 6),
                (-291, 0, -60, 14.5, 17),
                (-222, 0, -41, 25, 29),
                (-146, 0, -14, 37, 41),
                (-80, 0, 8, 48, 52),
                (-43, 0, 15, 58, 67),
                (0, 0, 21, 75, 78),
                (32, 0, 22, 74, 71),
            ],
            sides=36,
        )
        self.loft(
            self.body("head"),
            "neck_collar",
            "face",
            [
                (-80, 0, -40, 74, 65),
                (-52, 0, -21, 85, 72),
                (-16, 0, -7, 78, 68),
                (18, 0, -5, 70, 57),
                (44, 0, -3, 53, 43),
            ],
            sides=36,
        )

    def frill(self, head):
        # Flattened lenticular shield, gently scalloped around its continuous rim.
        sides = 100
        rings = 24
        verts = []
        uv = []
        faces = []

        def point(a, r, front):
            scallop = 1 + 0.025 * r**3 * np.cos(14 * a)
            yy = 149 * r * np.sin(a) * scallop
            zz = 76 + 116 * r * np.cos(a) * scallop
            x = 17 - 0.73 * zz - 0.00045 * yy * yy + front * (2.6 + 23 * (1 - r * r))
            return (x, yy, zz)

        for face in (1, -1):
            for k in range(rings + 1):
                r = max(0.0008, k / rings)
                for j in range(sides):
                    a = TAU * j / sides
                    verts.append(point(a, r, face))
                    uv.append((0.5 + 0.49 * r * np.sin(a), 0.5 + 0.49 * r * np.cos(a)))
        offset = (rings + 1) * sides
        for face, off in ((1, 0), (-1, offset)):
            for k in range(rings):
                for j in range(sides):
                    a = off + k * sides + j
                    b = off + k * sides + (j + 1) % sides
                    quad = [a, b, b + sides, a + sides]
                    if face == -1:
                        quad.reverse()
                    faces.extend(((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])))
        for j in range(sides):
            a = rings * sides + j
            b = rings * sides + (j + 1) % sides
            faces.extend(((a, a + offset, b + offset), (a, b + offset, b)))
        # Finish each face at the hub of the radial mesh.
        for off, front in ((0, True), (offset, False)):
            centre = len(verts)
            verts.append(point(0, 0, 1 if front else -1))
            uv.append((0.5, 0.5))
            for j in range(sides):
                triangle = (centre, off + (j + 1) % sides, off + j)
                faces.append(triangle if front else triangle[::-1])
        self.mesh(head, "frill", "frill", verts, faces, uv)
        # Small epiossifications lie along the upper and lateral margin.
        for j, deg in enumerate(range(-105, 106, 15)):
            a = np.deg2rad(deg)
            normal = np.array([-0.08, np.sin(a), np.cos(a)])
            side = np.array([0, np.cos(a), -np.sin(a)])
            origin = np.array(point(a, 0.994, 1))
            # Broad low triangular scales grow out of the rim with a curved ridge.
            # Keep the rear and front surfaces nearly flush with the frill skin.
            verts = []
            for radial, width, forward in [(-5, 8.5, -5), (-4, 8, 3), (1, 5.6, 2.7), (6, 1.1, 0.25)]:
                for k in range(12):
                    t = TAU * k / 12
                    verts.append(
                        origin
                        + normal * radial
                        + side * width * np.cos(t)
                        + np.array([forward + 2.0 * np.sin(t), 0, 0])
                    )
            faces = []
            for i in range(3):
                for k in range(12):
                    a0 = i * 12 + k
                    a1 = i * 12 + (k + 1) % 12
                    faces.extend(((a0, a1, a1 + 12), (a0, a1 + 12, a0 + 12)))
            faces += [(36, 36 + k + 1, 36 + k) for k in range(1, 11)]
            faces += [(0, k, k + 1) for k in range(1, 11)]
            self.mesh(head, f"frill_edge_{j}", "rim", verts, faces, [(0, 0)] * len(verts))

    def head(self):
        head = self.body("head_yaw")
        self.frill(head)
        self.loft(
            head,
            "frill_base",
            "face",
            [
                (-61, 0, 93, 18, 14),
                (-40, 0, 83, 48, 28),
                (-9, 0, 69, 51, 30),
                (15, 0, 63, 40, 26),
                (31, 0, 63, 37, 29),
                (47, 0, 52, 22, 18),
                (58, 0, 44, 8, 8),
            ],
            sides=28,
        )
        self.loft(
            head,
            "skull",
            "face",
            [
                (-32, 0, -7, 41, 36),
                (-8, 0, -5, 65, 48),
                (28, 0, -2, 78, 54),
                (59, 0, 1, 77, 51),
                (83, 0, -1, 69, 44),
                (109, 0, -9, 59, 38),
                (137, 0, -17, 47, 29),
                (160, 0, -26, 36, 22),
                (183, 0, -37, 20, 15),
            ],
            sides=44,
        )
        self.loft(
            head,
            "upper_beak",
            "beak",
            [
                (148, 0, -27, 31, 17),
                (164, 0, -27, 31, 22),
                (183, 0, -32, 26, 24),
                (198, 0, -39, 18, 20),
                (213, 0, -47, 8, 16),
                (214, 0, -56, 2, 7),
            ],
            sides=28,
        )
        jaw = ET.SubElement(head, "body", name="exterior_jaw", pos=mm((43, 0, -31)))
        self.loft(
            jaw,
            "lower_jaw",
            "face",
            [
                (-18, 0, 2, 35, 13),
                (9, 0, -3, 52, 18),
                (37, 0, -7, 51, 18),
                (70, 0, -15, 42, 15),
                (101, 0, -24, 34, 14),
                (133, 0, -30, 17, 9),
            ],
            sides=30,
        )
        self.loft(
            jaw,
            "mandible_tip",
            "lowerbeak",
            [
                (111, 0, -24, 27, 10),
                (126, 0, -29, 24, 10),
                (144, 0, -30, 17, 11),
                (157, 0, -29, 10, 7),
                (164, 0, -25, 2, 4),
            ],
            sides=24,
        )
        self.oval(jaw, "mouth_lining", "mouth", (77, 0, -9), (75, 39, 8), euler="0 10 0")
        self.oval(jaw, "tongue", "tongue", (82, 0, -7), (64, 24, 6), euler="0 10 0")
        for sign, side in ((1, "left"), (-1, "right")):
            self.loft(
                head,
                side + "_horn",
                "ivory",
                [
                    (31, sign * 49, 36, 18, 18),
                    (54, sign * 50, 42, 21, 20),
                    (79, sign * 55, 49, 18, 17),
                    (119, sign * 60, 58, 13.4, 13.2),
                    (159, sign * 64, 65, 9, 9),
                    (195, sign * 65, 71, 5.2, 5.3),
                    (225, sign * 64, 73, 1.2, 1.2),
                ],
                sides=28,
            )
            self.oval(head, side + "_brow", "boss", (64, sign * 62, 31), (32, 14, 14), euler="0 -12 0")
            self.oval(head, side + "_orbit", "fold", (70, sign * 70, 13), (17, 7.7, 13))
            self.oval(head, side + "_eye", "iris", (71, sign * 76.4, 13), (11.8, 4.3, 10.0))
            self.oval(head, side + "_pupil", "pupil", (74, sign * 80.3, 13), (5.8, 1.7, 7.4))
            self.oval(head, side + "_eye_glint", "glint", (76, sign * 81.9, 16.2), (1.65, 0.7, 1.65))
            lid = ET.SubElement(head, "body", name=f"exterior_{side}_lid", pos=mm((70, sign * 77.8, 29.5)))
            self.oval(lid, side + "_lid_scale", "lid", (0, 0, 0), (15.8, 4.6, 7.3))
            self.oval(head, side + "_narial_rim", "boss", (145, sign * 42, -11.5), (14, 5.4, 8), euler="0 -18 0")
            self.oval(head, side + "_nostril", "pupil", (147, sign * 46.3, -11.5), (7.7, 2.5, 3.4), euler="0 -20 0")
            self.oval(head, side + "_cheek", "face", (40, sign * 64, -12), (32, 15, 22), euler="0 -14 0")
        # A short swept nasal horn follows the underlying third horn.
        self.loft(
            head,
            "nose_horn",
            "ivory",
            [
                (127, 0, 0, 17, 17),
                (144, 0, 10, 15, 15),
                (155, 0, 24, 10.5, 11),
                (166, 0, 35, 5.2, 5.6),
                (172, 0, 42, 1.0, 1.1),
            ],
            sides=24,
        )

    def sleeve(self, body, geom_name, label, extra, material):
        src = body.find(f"geom[@name='{geom_name}']")
        a, b = np.fromstring(src.get("fromto"), sep=" ").reshape(2, 3)
        along = (b - a) / np.linalg.norm(b - a)
        self.geom(
            body,
            label,
            material,
            type=src.get("type"),
            fromto=vec(np.r_[a - 0.0015 * along, b + 0.0015 * along]),
            size=vec([float(src.get("size")) + extra]),
        )

    def bonnet(self, body, name, x, y, z, r, h):
        # Round-edged caps along the real axle, with a shallow protective dome.
        sides = 36
        vertices = []
        uv = []
        faces = []
        rings = [
            (-h - 2, 0.13),
            (-h - 0.9, 0.77),
            (-h, 0.95),
            (-h + 1, 1),
            (h - 1, 1),
            (h, 0.95),
            (h + 0.9, 0.77),
            (h + 2, 0.13),
        ]
        for i, (off, scale) in enumerate(rings):
            for j in range(sides):
                t = TAU * j / sides
                vertices.append((x + r * scale * np.cos(t), y + off, z + r * scale * np.sin(t)))
                uv.append((i / (len(rings) - 1), j / sides))
        for i in range(len(rings) - 1):
            for j in range(sides):
                a = i * sides + j
                b = i * sides + (j + 1) % sides
                faces.extend(((a, a + sides, b + sides), (a, b + sides, b)))
        for i in (0, len(rings) - 1):
            k = i * sides
            for j in range(1, sides - 1):
                face = (k, k + j, k + j + 1)
                faces.append(face if i else face[::-1])
        self.mesh(body, name, "limb", vertices, faces, uv)

    def legs(self):
        for side, dy in (("left", 1), ("right", -1)):
            for end, direction in (("fore", -1), ("hind", 1)):
                tag = f"{side}_{end}"
                # The yoke fairing wraps the pivot between trunk and pitched upper segment.
                yoke = self.body(tag + "_roll_yoke")
                self.oval(yoke, tag + "_shoulder", "limb", (0, dy * 50, 4), (60, 68, 65))
                upper = self.body(tag + "_upper_link")
                lower = self.body(tag + "_lower_link")
                self.loft(
                    upper,
                    tag + "_upper",
                    "limb",
                    [
                        (-direction * 11, 0, 43, 33, 38),
                        (-direction * 5, 0, 23, 52, 50),
                        (direction * 5, 0, 0, 55, 54),
                        (direction * 33, 0, -40, 54, 49),
                        (direction * 63, -dy * 3, -78, 47, 45),
                        (direction * 93, -dy * 3, -112, 38, 40),
                        (direction * 116, -dy * 2, -139, 33, 39),
                        (direction * 124, 0, -157, 22, 32),
                    ],
                    axis="z",
                )
                self.sleeve(self.chassis, tag + "_x4_roll_stator", tag + "_inner_pivot", 0.0025, "limb")
                self.sleeve(upper, tag + "_pushrod", tag + "_rod_boot", 0.0028, "limb")
                self.sleeve(lower, tag + "_output_crank", tag + "_crank_cuff", 0.0016, "limb")
                self.oval(lower, tag + "_joint_cuff", "fold", (0, 0, 0), (36, 41, 35))
                # Localized covers follow actual stator/crank radii; sizing the
                # whole leg to the widest fastener would inflate its silhouette.
                radius = 40 if end == "fore" else 27.5
                self.bonnet(
                    upper,
                    tag + "_drive_fairing",
                    direction * 57.5,
                    0,
                    -69.23,
                    radius + 2.5,
                    36.3 if end == "fore" else 33.3,
                )
                self.bonnet(upper, tag + "_drive_crank_cover", direction * 57.5, -dy * 41.75, -69.23, 37.2, 7.5)
                self.bonnet(lower, tag + "_output_cover", 0, -dy * 41.75, 0, 37.2, 7.5)
                self.loft(
                    lower,
                    tag + "_shank",
                    "limb",
                    [
                        (-direction * 5, 0, 15, 32, 40),
                        (0, 0, 5, 33, 40),
                        (-direction * 15, 0, -19, 31, 36),
                        (-direction * 39, 0, -46, 25, 27),
                        (-direction * 74, 0, -86, 22, 24),
                        (-direction * 107, 0, -123, 21, 23),
                        (-direction * 119, 0, -139, 21, 21),
                    ],
                    axis="z",
                )
                foot = self.body(tag + "_foot")
                self.oval(foot, tag + "_ankle_cuff", "fold", (0, 0, 4), (23, 22, 20))
                self.loft(
                    foot,
                    tag + "_foot_shell",
                    "limb",
                    [
                        (-38, 0, -15, 4, 6),
                        (-30, 0, -13, 19, 16),
                        (-14, 0, -12, 25, 21),
                        (8, 0, -12, 26, 22),
                        (26, 0, -13, 23, 17),
                        (39, 0, -17, 12, 11),
                        (43, 0, -18, 3, 4),
                    ],
                    sides=24,
                )
                self.oval(foot, tag + "_rocker", "sole", (0, 0, -22), (40, 23, 13.4))
                for k in range(3):
                    lateral = (k - 1) * 13
                    self.oval(
                        foot,
                        f"{tag}_nail_{k}",
                        "nail",
                        (34 - abs(k - 1) * 3, lateral, -21),
                        (10.7, 5.9, 7.6),
                        euler=vec((0, 8, (k - 1) * 17)),
                    )

    def write(self):
        self.trunk()
        self.head()
        self.legs()
        self.root.set("model", "triceratops_60cm_option_C_full_exterior")
        ET.indent(self.root)
        target = self.out / "trike60_C_closed_capsule_exterior.xml"
        ET.ElementTree(self.root).write(target, encoding="utf-8", xml_declaration=True)
        return target


def build(out, source=SOURCE):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    palette(out)
    return Exterior(source, out).write()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=SOURCE)
    args = ap.parse_args()
    print(build(args.out, args.source))
