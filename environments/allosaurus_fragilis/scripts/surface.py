"""Generate textured surface meshes in metres around a separately defined rigid plant.

Rings describe anatomical cross-sections. Surfaces overlap at mobile joints,
with cut ends buried under adjacent skin. Pigmentation is an artistic choice.
"""

import struct
import zlib
from math import cos, exp, pi, sin, sqrt
from pathlib import Path
from xml.etree.ElementTree import SubElement

TAU = 2 * pi


def unit(v):
    n = sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def mix(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def write_hide(file):
    """Seamless fine-grain field above a countershaded, barred flank."""
    width, height = 512, 256
    pixels = bytearray()
    for row in range(height):
        pixels.append(0)  # PNG filter None
        theta = (1 - (row + 0.5) / height) * TAU
        h = sin(theta)
        for col in range(width):
            u = (col + 0.5) / width
            # Repeated longitudinal periods keep all texture joins continuous.
            mottling = sin(u * TAU * 9 + 2 * cos(theta * 4)) * 0.5 + cos(u * TAU * 21 + 4 * sin(theta * 3)) * 0.3
            wave = cos(TAU * 5 * u + 1.6 * sin(theta * 2) + 0.44 * sin(TAU * 3 * u)) + 0.26 * sin(
                TAU * 11 * u - 3 * theta
            )
            shade = max(0.0, min(1.0, (h + 0.49) / 1.24))
            base = mix((209, 178, 131), (114, 102, 70), shade)
            bar = max(0.0, min(1.0, (wave - 0.19) * 1.45)) * max(0.0, min(1.0, (h + 0.29 + 0.10 * mottling) * 1.55))
            scale = (col * 374761393 + row * 668265263) & 0xFFFFFFFF
            scale = ((scale ^ (scale >> 13)) * 1274126177) & 0xFFFFFFFF
            grain = ((scale >> 16) / 65535.0 - 0.5) * 11
            rgb = mix(base, (67, 64, 49), 0.68 * bar)
            pixels.extend(max(0, min(255, round(c + 4 * mottling + grain))) for c in rgb)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    file.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels, 9)) + chunk(b"IEND", b"")
    )


def numbers(values):
    return " ".join(f"{v:.6g}" for v in values)


class Surface:
    def __init__(self, asset, directory, bodies):
        self.asset = asset
        self.origins = {}

        def record_origin(node, parent=(0, 0, 0)):
            pos = tuple(a + float(b) for a, b in zip(parent, node.get("pos", "0 0 0").split()))
            self.origins[node.get("name")] = pos
            for child in node.findall("body"):
                record_origin(child, pos)

        record_origin(bodies["pelvis"])
        texture_dir = directory / "textures"
        texture_dir.mkdir(exist_ok=True)
        write_hide(texture_dir / "hide.png")
        SubElement(asset, "texture", name="scale_field", type="2d", file="textures/hide.png")
        SubElement(asset, "material", name="dermis", texture="scale_field", specular=".12", shininess=".18")
        colors = {
            "fold": (0.35, 0.311, 0.222),
            "orbit": (0.37, 0.324, 0.225),
            "iris": (0.57, 0.403, 0.187),
            "pupil": (0.052, 0.046, 0.034),
            "nostril": (0.208, 0.168, 0.115),
            "sole": (0.47, 0.373, 0.266),
            "toe": (0.55, 0.462, 0.323),
            "claw": (0.302, 0.252, 0.181),
        }
        for name, c in colors.items():
            SubElement(
                asset,
                "material",
                name=name,
                rgba=" ".join(f"{v:.3f}" for v in (*c, 1)),
                specular=".16",
                shininess=".18",
            )

    def rings(self, body, name, sections, material="dermis", n=16, axial=False):
        """Loft elliptical rings normal to a centerline, or in the YZ plane."""
        vertices, tex, faces = [], [], []
        origin = self.origins[body.get("name")]
        for i, (x, y, z, w, r) in enumerate(sections):
            if axial:
                lateral, radial = (0, 1, 0), (0, 0, 1)
            else:
                a, b = sections[max(0, i - 1)], sections[min(len(sections) - 1, i + 1)]
                axis = unit(tuple(q - p for p, q in zip(a[:3], b[:3])))
                guide = (0, 1, 0) if abs(axis[1]) < 0.94 else (0, 0, 1)
                radial = unit(cross(axis, guide))
                lateral = unit(cross(radial, axis))
            for j in range(n):
                angle = TAU * j / n
                c, d = cos(angle), sin(angle)
                point = tuple(v + o + a * c * w + b * d * r for v, o, a, b in zip((x, y, z), origin, lateral, radial))
                vertices.extend(point)
                tex.extend((point[0] / 2.65 + 0.41, 0.475 + 0.09 * (point[2] - 1.8) + 0.028 * d))
                if i:
                    k = (j + 1) % n
                    a, b = (i - 1) * n, i * n
                    faces.extend((a + j, a + k, b + j, a + k, b + k, b + j))
        for j in range(1, n - 1):
            a = (len(sections) - 1) * n
            faces.extend((0, j + 1, j, a, a + j, a + j + 1))
        skin = SubElement(
            self.asset,
            "skin",
            name=name,
            material=material,
            group="2",
            vertex=numbers(vertices),
            face=numbers(faces),
            texcoord=numbers(tex),
        )
        count = len(vertices) // 3
        SubElement(
            skin,
            "bone",
            body=body.get("name"),
            bindpos=numbers(origin),
            bindquat="1 0 0 0",
            vertid=numbers(range(count)),
            vertweight=numbers([1] * count),
        )

    def oval(self, body, name, material, pos, size, **attrs):
        SubElement(
            body,
            "geom",
            name=name,
            type="ellipsoid",
            material=material,
            pos=" ".join(str(i) for i in pos),
            size=" ".join(str(i) for i in size),
            **{"class": "hide"},
            **attrs,
        )


def smooth_skin(s, name, rows, bones, n, axial=False):
    """Linear-blend skin in the home frame; gives turning joints a continuous hide."""
    vertices, faces, tex = [], [], []
    bindings = {bone: [] for bone in bones}
    for i, (center, w, r, weights) in enumerate(rows):
        if axial:
            lateral, dorsal = (0, 1, 0), (0, 0, 1)
        else:
            a, b = rows[max(0, i - 1)][0], rows[min(len(rows) - 1, i + 1)][0]
            axis = unit(tuple(q - p for p, q in zip(a, b)))
            dorsal = unit(cross(axis, (0, 1, 0)))
            lateral = unit(cross(dorsal, axis))
        for j in range(n):
            c, d = cos(TAU * j / n), sin(TAU * j / n)
            point = tuple(v + c * w * side + d * r * up for v, side, up in zip(center, lateral, dorsal))
            if axial and center[0] > 1.92:
                # The antorbital depression and paired lacrimal bosses are
                # sculpted in the head surface. The median skull roof remains low.
                x = center[0] - 2.02
                crest = 0.133 * exp(-(((x - 0.218) / 0.102) ** 2)) * exp(-(((d - 0.74) / 0.18) ** 2))
                muzzle = exp(-(((x - 0.42) / 0.15) ** 2)) * exp(-(((d - 0.15) / 0.42) ** 2))
                cheek = exp(-(((x - 0.042) / 0.106) ** 2) - (((d + 0.11) / 0.32) ** 2))
                bridge = exp(-(((x - 0.115) / 0.079) ** 2) - (((d - 0.57) / 0.16) ** 2))
                point = (point[0], point[1] * (1 - 0.20 * muzzle + 0.07 * cheek + 0.047 * bridge), point[2] + crest)
            k = i * n + j
            vertices.extend(point)
            for bone, weight in weights.items():
                if weight:
                    bindings[bone].append((k, weight))
            if axial:
                # Duplicate-free UV mapping wraps at the lower flank. Use
                # continuous dorsal height to avoid a texture seam on the skin.
                tex.extend((point[0] / 2.65 + 0.41, 0.50 + 0.245 * d))
            else:
                blend = max(0.0, min(1.0, (point[2] - 1.26) / 0.56))
                ground = bones["pelvis"][2]
                u = point[0] / 2.65 + 0.41 - 0.14 * (1 - blend) * (ground - point[2])
                v = 0.52 + 0.055 * d + blend * (0.43 * (point[2] - ground - 0.09) - 0.02 - 0.055 * d)
                tex.extend((u, v))
            if i:
                nxt = (j + 1) % n
                a = (i - 1) * n + j
                b = (i - 1) * n + nxt
                faces.extend((a, b, k, b, i * n + nxt, k))
    for j in range(1, n - 1):
        a = (len(rows) - 1) * n
        faces.extend((0, j + 1, j, a, a + j, a + j + 1))

    skin = SubElement(
        s.asset,
        "skin",
        name=name,
        material="dermis",
        group="2",
        vertex=numbers(vertices),
        face=numbers(faces),
        texcoord=numbers(tex),
    )
    for bone, items in bindings.items():
        if items:
            SubElement(
                skin,
                "bone",
                body=bone,
                bindpos=numbers(bones[bone]),
                bindquat="1 0 0 0",
                vertid=numbers(i for i, _ in items),
                vertweight=numbers(w for _, w in items),
            )


def axial_surface(s, bodies):
    # [x, half-width, belly z, back z] relative to the hip. The weighted
    # surface spans both passive caudal joints and the arched cervical series.
    stations = [
        (-4.40, 0.004, 0.042, 0.052),
        (-4.32, 0.016, 0.026, 0.063),
        (-4.08, 0.034, -0.005, 0.071),
        (-3.76, 0.050, -0.039, 0.077),
        (-3.40, 0.061, -0.071, 0.083),
        (-3.04, 0.073, -0.092, 0.086),
        (-2.68, 0.089, -0.104, 0.102),
        (-2.32, 0.107, -0.11, 0.134),
        (-2.16, 0.120, -0.116, 0.153),
        (-2.04, 0.131, -0.119, 0.174),
        (-1.91, 0.141, -0.121, 0.193),
        (-1.77, 0.159, -0.124, 0.209),
        (-1.60, 0.177, -0.132, 0.222),
        (-1.37, 0.213, -0.133, 0.255),
        (-1.13, 0.242, -0.133, 0.289),
        (-0.94, 0.270, -0.14, 0.348),
        (-0.75, 0.299, -0.145, 0.419),
        (-0.61, 0.329, -0.165, 0.464),
        (-0.51, 0.355, -0.192, 0.509),
        (-0.38, 0.397, -0.248, 0.555),
        (-0.26, 0.43, -0.30, 0.587),
        (-0.12, 0.455, -0.349, 0.607),
        (0.07, 0.47, -0.416, 0.614),
        (0.29, 0.474, -0.482, 0.586),
        (0.55, 0.459, -0.514, 0.543),
        (0.76, 0.427, -0.513, 0.514),
        (0.94, 0.382, -0.475, 0.497),
        (1.10, 0.323, -0.411, 0.484),
        (1.23, 0.280, -0.310, 0.476),
        (1.34, 0.248, -0.210, 0.486),
        (1.45, 0.223, -0.091, 0.53),
        (1.55, 0.204, 0.037, 0.594),
        (1.66, 0.186, 0.155, 0.694),
        (1.79, 0.176, 0.281, 0.766),
        (1.87, 0.157, 0.357, 0.795),
        (1.92, 0.147, 0.391, 0.804),
        (1.97, 0.155, 0.427, 0.819),
        (2.015, 0.168, 0.413, 0.839),
        (2.065, 0.18, 0.407, 0.854),
        (2.115, 0.181, 0.401, 0.860),
        (2.159, 0.178, 0.401, 0.864),
        (2.194, 0.174, 0.406, 0.863),
        (2.225, 0.169, 0.412, 0.868),
        (2.250, 0.166, 0.419, 0.866),
        (2.279, 0.16, 0.423, 0.856),
        (2.302, 0.155, 0.426, 0.846),
        (2.338, 0.148, 0.427, 0.833),
        (2.39, 0.139, 0.429, 0.819),
        (2.48, 0.131, 0.438, 0.793),
        (2.57, 0.119, 0.447, 0.77),
        (2.65, 0.113, 0.456, 0.758),
        (2.73, 0.111, 0.467, 0.748),
        (2.79, 0.100, 0.475, 0.738),
        (2.831, 0.078, 0.491, 0.718),
        (2.865, 0.048, 0.520, 0.692),
        (2.874, 0.006, 0.563, 0.650),
    ]
    hip_z = float(bodies["pelvis"].get("pos").split()[2])
    origins = {"pelvis": (0, 0, hip_z), "tail_base": (-0.52, 0, hip_z + 0.15), "tail_distal": (-2.05, 0, hip_z + 0.015)}

    def transition(x, mid, span):
        t = max(0.0, min(1.0, 0.5 + (x - mid) / span))
        return t * t * (3 - 2 * t)

    rows = []
    for x, w, low, high in stations:
        base = transition(x, -2.05, 0.40)
        trunk = transition(x, -0.52, 0.56)
        weights = {"pelvis": trunk, "tail_base": base - trunk, "tail_distal": 1 - base}
        rows.append(((x, 0, hip_z + (low + high) * 0.5), w, (high - low) * 0.5, weights))
    smooth_skin(s, "axial_hide", rows, origins, 32, axial=True)


def skull_surface(s, head):
    # The upper jaw is narrow rostrally and deeper under the orbit. Two separate
    # lacrimal crests rise anterior to the eye; the closed lip seam follows the
    # maxillary rim. Three-dimensional nostrils and lids sit on the outer skin.
    for side, sign in (("l", 1), ("r", -1)):
        # Surface landmarks sampled on the fleshed maxilla and dentary. A thin
        # tapering commissure separates the lips along the ventral third.
        def line(name, points, material):
            s.rings(
                head,
                side + "_" + name,
                [(x, y * sign, z, a, b) for x, y, z, a, b in points],
                material,
                n=10,
                axial=True,
            )

        s.oval(head, side + "_eyelid", "orbit", (0.084, 0.168 * sign, 0.086), (0.040, 0.013, 0.028))
        s.oval(head, side + "_eye", "iris", (0.087, 0.177 * sign, 0.085), (0.022, 0.009, 0.019))
        s.oval(head, side + "_pupil", "pupil", (0.090, 0.184 * sign, 0.085), (0.010, 0.004, 0.011))
        s.oval(head, side + "_naris", "nostril", (0.716, 0.106 * sign, 0.026), (0.030, 0.006, 0.011), euler="0 -.12 0")
        s.oval(head, side + "_ear", "fold", (-0.032, 0.145 * sign, 0.051), (0.019, 0.008, 0.022))
        line(
            "gape",
            [
                (0.012, 0.151, -0.107, 0.001, 0.001),
                (0.059, 0.164, -0.100, 0.002, 0.003),
                (0.111, 0.163, -0.105, 0.0035, 0.0038),
                (0.188, 0.156, -0.109, 0.0035, 0.004),
                (0.266, 0.148, -0.105, 0.0035, 0.004),
                (0.362, 0.134, -0.106, 0.0035, 0.004),
                (0.466, 0.125, -0.103, 0.003, 0.004),
                (0.558, 0.114, -0.097, 0.003, 0.004),
                (0.639, 0.108, -0.088, 0.003, 0.0035),
                (0.707, 0.106, -0.081, 0.003, 0.0035),
                (0.768, 0.097, -0.069, 0.003, 0.0035),
                (0.807, 0.077, -0.061, 0.002, 0.003),
                (0.844, 0.047, -0.045, 0.001, 0.001),
            ],
            "fold",
        )


def limb_surface(s, bodies):
    for prefix, sign in (("l", 1), ("r", -1)):

        def loft(node, part, sections, mat="dermis", n=16):
            points = [(a, b * sign, c, d, e) for a, b, c, d, e in sections]
            s.rings(node, prefix + "_" + part, points, mat, n=n)

        thigh, shin, mt, foot = [bodies[prefix + "_" + p] for p in ("thigh", "shin", "metatarsus", "foot")]
        # A continuous, skinned leg rounds the forward knee and raised ankle.
        # Each blend is limited to the immediate joint region.
        origins = {}
        base = (0, 0, float(bodies["pelvis"].get("pos").split()[2]))
        for node in (thigh, shin, mt):
            base = tuple(a + float(b) for a, b in zip(base, node.get("pos").split()))
            origins[node.get("name")] = base
        origins["pelvis"] = (0, 0, float(bodies["pelvis"].get("pos").split()[2]))
        tb, sb, mb = [node.get("name") for node in (thigh, shin, mt)]
        rows = []

        def section(bone, data):
            for x, y, z, w, r, weights in data:
                origin = origins[bone]
                center = (origin[0] + x, origin[1] + y * sign, origin[2] + z)
                rows.append((center, w, r, weights))

        section(
            tb,
            [
                (-0.07, -0.05, 0.24, 0.10, 0.15, {"pelvis": 0.88, tb: 0.12}),
                (-0.08, 0.0, 0.142, 0.16, 0.23, {"pelvis": 0.61, tb: 0.39}),
                (-0.07, 0.023, 0.027, 0.18, 0.28, {"pelvis": 0.36, tb: 0.64}),
                (-0.038, 0.038, -0.12, 0.197, 0.29, {"pelvis": 0.13, tb: 0.87}),
                (0.006, 0.042, -0.251, 0.198, 0.285, {tb: 1}),
                (0.063, 0.035, -0.387, 0.173, 0.239, {tb: 1}),
                (0.154, 0.021, -0.526, 0.135, 0.191, {tb: 1}),
                (0.228, 0.006, -0.657, 0.107, 0.144, {tb: 1}),
                (0.270, -0.008, -0.749, 0.095, 0.118, {tb: 0.93, sb: 0.07}),
                (0.294, -0.013, -0.811, 0.090, 0.105, {tb: 0.67, sb: 0.33}),
            ],
        )
        section(
            sb,
            [
                (0.004, 0.002, -0.014, 0.092, 0.102, {tb: 0.36, sb: 0.64}),
                (-0.025, 0.001, -0.082, 0.099, 0.11, {tb: 0.12, sb: 0.88}),
                (-0.073, 0.002, -0.165, 0.105, 0.125, {sb: 1}),
                (-0.135, 0.004, -0.264, 0.096, 0.119, {sb: 1}),
                (-0.195, -0.005, -0.373, 0.071, 0.085, {sb: 1}),
                (-0.246, -0.014, -0.47, 0.053, 0.061, {sb: 1}),
                (-0.298, -0.027, -0.576, 0.045, 0.051, {sb: 0.97, mb: 0.03}),
                (-0.326, -0.031, -0.633, 0.052, 0.055, {sb: 0.72, mb: 0.28}),
            ],
        )
        section(
            mb,
            [
                (-0.004, 0, 0.001, 0.056, 0.058, {sb: 0.4, mb: 0.6}),
                (0.003, 0, -0.034, 0.059, 0.058, {sb: 0.15, mb: 0.85}),
                (0.025, -0.002, -0.087, 0.051, 0.052, {mb: 1}),
                (0.047, -0.004, -0.154, 0.042, 0.044, {mb: 1}),
                (0.072, -0.007, -0.228, 0.039, 0.040, {mb: 1}),
                (0.093, -0.009, -0.296, 0.047, 0.045, {mb: 1}),
                (0.114, -0.01, -0.347, 0.066, 0.047, {mb: 1}),
                (0.118, -0.01, -0.367, 0.060, 0.035, {mb: 1}),
            ],
        )
        smooth_skin(s, prefix + "_hindlimb_hide", rows, origins, 22)
        # Medial pedal digit I arises above the main rays, directed caudomedially.
        loft(
            mt,
            "hallux_hide",
            [
                (0.066, -0.027, -0.214, 0.024, 0.023),
                (0.053, -0.058, -0.235, 0.021, 0.022),
                (0.017, -0.080, -0.252, 0.017, 0.020),
                (-0.016, -0.093, -0.268, 0.014, 0.014),
            ],
            "toe",
            12,
        )
        loft(
            mt,
            "hallux_claw",
            [
                (-0.006, -0.087, -0.261, 0.014, 0.015),
                (-0.021, -0.096, -0.267, 0.015, 0.017),
                (-0.043, -0.103, -0.279, 0.011, 0.011),
                (-0.052, -0.106, -0.293, 0.002, 0.002),
            ],
            "claw",
            12,
        )
        s.oval(foot, prefix + "_plantar_hide", "sole", (0.014, 0, -0.002), (0.082, 0.073, 0.043))
        for digit, ex, ey in [(2, 0.221, -0.141), (3, 0.295, 0), (4, 0.204, 0.14)]:
            y0 = {2: -0.033, 3: 0, 4: 0.033}[digit]
            digit_rows = []
            # The fleshed proximal pads alternate with narrower joints along
            # the rays. III is longest and the two shorter rays diverge.
            for f, w, r, z in [
                (0, 0.041, 0.031, 0.004),
                (0.12, 0.04, 0.034, -0.002),
                (0.32, 0.032, 0.032, -0.01),
                (0.45, 0.031, 0.029, -0.012),
                (0.55, 0.035, 0.03, -0.012),
                (0.70, 0.027, 0.026, -0.013),
                (0.82, 0.029, 0.026, -0.013),
                (0.98, 0.022, 0.019, -0.013),
                (1, 0.018, 0.017, -0.011),
            ]:
                digit_rows.append((0.017 + (ex - 0.017) * f, y0 + (ey - y0) * f, z, w, r))
            loft(foot, f"digit_{digit}", digit_rows, "toe", 14)
            loft(
                foot,
                f"ungual_{digit}",
                [
                    (ex - 0.013, ey * 0.97, -0.011, 0.022, 0.019),
                    (ex + 0.001, ey, -0.009, 0.022, 0.021),
                    (ex + 0.027, ey * 1.04, -0.011, 0.015, 0.017),
                    (ex + 0.051, ey * 1.08, -0.018, 0.008, 0.010),
                    (ex + 0.062, ey * 1.10, -0.031, 0.002, 0.002),
                ],
                "claw",
                12,
            )
        arm = bodies["pelvis"]
        loft(
            arm,
            "brachium",
            [
                (1.10, 0.185, 0.145, 0.047, 0.062),
                (1.12, 0.244, 0.090, 0.080, 0.096),
                (1.129, 0.285, 0.030, 0.086, 0.098),
                (1.12, 0.329, -0.052, 0.082, 0.089),
                (1.072, 0.357, -0.184, 0.063, 0.070),
                (1.020, 0.377, -0.281, 0.048, 0.055),
                (0.999, 0.383, -0.332, 0.041, 0.049),
            ],
            n=16,
        )
        loft(
            arm,
            "antebrachium",
            [
                (1.008, 0.379, -0.297, 0.041, 0.047),
                (1.013, 0.381, -0.335, 0.049, 0.057),
                (1.055, 0.376, -0.409, 0.048, 0.047),
                (1.126, 0.360, -0.474, 0.036, 0.039),
                (1.19, 0.345, -0.521, 0.031, 0.034),
                (1.222, 0.332, -0.543, 0.028, 0.030),
            ],
            n=14,
        )
        loft(
            arm,
            "manus",
            [
                (1.179, 0.344, -0.514, 0.026, 0.026),
                (1.217, 0.332, -0.541, 0.032, 0.040),
                (1.251, 0.318, -0.571, 0.031, 0.044),
                (1.28, 0.31, -0.598, 0.031, 0.044),
            ],
            "toe",
            14,
        )
        # Medially-facing palms. Short, robust manual I bears the strongest claw.
        for digit, startx, startz, length, y, thick in [
            (1, 1.293, -0.575, 0.121, 0.299, 0.022),
            (2, 1.269, -0.605, 0.155, 0.313, 0.019),
            (3, 1.23, -0.62, 0.115, 0.331, 0.013),
        ]:
            loft(
                arm,
                f"finger_{digit}",
                [
                    (startx - 0.016, y, startz + 0.019, thick, thick * 1.1),
                    (startx, y, startz, thick, thick),
                    (startx + 0.028, y - 0.010, startz - 0.34 * length, thick * 0.88, thick * 0.88),
                    (startx + 0.046, y - 0.019, startz - 0.57 * length, thick * 0.7, thick * 0.74),
                ],
                "toe",
                10,
            )
            loft(
                arm,
                f"hand_claw_{digit}",
                [
                    (startx + 0.04, y - 0.017, startz - 0.50 * length, thick * 0.74, thick * 0.80),
                    (startx + 0.049, y - 0.022, startz - 0.64 * length, thick * 0.74, thick * 0.79),
                    (startx + 0.045, y - 0.030, startz - 0.82 * length, thick * 0.46, thick * 0.51),
                    (startx + 0.020, y - 0.042, startz - length, 0.002, 0.002),
                ],
                "claw",
                10,
            )


def add_surface(asset, bodies, directory: Path):
    s = Surface(asset, directory, bodies)
    axial_surface(s, bodies)
    skull_surface(s, bodies["head"])
    limb_surface(s, bodies)
