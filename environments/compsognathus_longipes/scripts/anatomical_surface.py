"""Build the axial body and folded forelimbs at the scaled Canjuers specimen size."""

import xml.etree.ElementTree as ET

import numpy as np


def numbers(values):
    return " ".join(f"{v:.8g}" for v in np.ravel(values))


def longitudinal_mesh(asset, name, sections):
    """Closed elliptical rings (x, centre z, half-width y, half-depth z), in metres."""
    sections = np.asarray(sections, dtype=float)
    x = sections[:, 0]
    if np.any(np.diff(x) <= 0) or np.any(sections[:, 2:] <= 0):
        raise ValueError(f"Non-increasing stations or empty section in {name}")
    count = 20
    angle = np.arange(count) * 2 * np.pi / count
    vertices, uv, faces = [], [], []
    for ring, (along, center, half_width, half_depth) in enumerate(sections):
        for i, theta in enumerate(angle):
            vertices.append([along, half_width * np.cos(theta), center + half_depth * np.sin(theta)])
            uv.append([0.5, (1 + np.sin(theta)) / 4])
            if ring:
                a = (ring - 1) * count + i
                b = (ring - 1) * count + (i + 1) % count
                # Mirror the diagonal across y=0 as well as the vertices: a
                # twisted quad on a curved neck otherwise biases its mass.
                if np.cos(theta + np.pi / count) > 0:
                    faces.extend([[a, b, b + count], [a, b + count, a + count]])
                else:
                    faces.extend([[a, b, a + count], [b, b + count, a + count]])
    last = (len(sections) - 1) * count
    for i in range(1, count - 1):
        faces.extend([[0, i + 1, i], [last, last + i, last + i + 1]])
    ET.SubElement(
        asset, "mesh", name=name, inertia="exact", vertex=numbers(vertices), face=numbers(faces), texcoord=numbers(uv)
    )


def skin(asset, body, name, stations_mm, mass, material="hide_mat"):
    """Mass follows the flesh envelope; low-order colliders are authored separately."""
    longitudinal_mesh(asset, name, np.asarray(stations_mm) / 1000)
    ET.SubElement(
        body, "geom", name=name, type="mesh", mesh=name, material=material, mass=str(mass), contype="0", conaffinity="0"
    )


def capsule(body, name, start, end, radius, **attributes):
    return ET.SubElement(
        body, "geom", name=name, type="capsule", fromto=numbers([start, end]), size=str(radius), **attributes
    )


def face_detail(body, name, pos, size, material):
    ET.SubElement(
        body,
        "geom",
        name=name,
        type="ellipsoid",
        pos=numbers(pos),
        size=numbers(size),
        material=material,
        mass="0",
        contype="0",
        conaffinity="0",
    )


def arm_sleeve(arm, name, mesh, start, end):
    # Align the authored longitudinal X axis with the shoulder–elbow or elbow–wrist shaft.
    direction = np.subtract(end, start)
    direction /= np.linalg.norm(direction)
    quat = np.array([1 + direction[0], 0, -direction[2], direction[1]])
    ET.SubElement(
        arm,
        "geom",
        name=name,
        type="mesh",
        mesh=mesh,
        pos=numbers(start),
        quat=numbers(quat / np.linalg.norm(quat)),
        material="body_mat",
        mass="0",
        contype="0",
        conaffinity="0",
    )


def folded_arm(pelvis, side, sign):
    # Inboard-facing palms put digits I and II at different heights in side view.
    arm = ET.SubElement(pelvis, "body", name=f"{side}_arm", pos=numbers([0.111, sign * 0.024, 0.026]))
    elbow = [-0.024, sign * 0.010, -0.038]
    wrist = [0.010, sign * 0.014, -0.059]
    palm = [0.031, sign * 0.012, -0.055]
    capsule(arm, f"arm_geom_{sign}", [0, 0, 0], elbow, 0.0045, mass=".0048", group="3")
    capsule(arm, f"{side}_forearm", elbow, wrist, 0.0030, mass=".0023", group="3")
    arm_sleeve(arm, f"{side}_upper_arm_skin", "brachial_skin", [0, 0, 0], elbow)
    arm_sleeve(arm, f"{side}_forearm_skin", "antebrachial_skin", elbow, wrist)
    capsule(arm, f"{side}_manus", wrist, palm, 0.0034, mass=".0009", material="body_mat")
    # Gishlick & Gauthier resolve a much-reduced digit III. Its external
    # envelope is uncertain; only the two robust fingers project past the palm.
    for digit, base, knuckle in (
        ("I", [0.027, sign * 0.011, -0.053], [0.043, sign * 0.009, -0.054]),
        ("II", [0.030, sign * 0.013, -0.056], [0.049, sign * 0.013, -0.062]),
        ("III", [0.018, sign * 0.016, -0.060], [0.028, sign * 0.016, -0.062]),
    ):
        capsule(
            arm,
            f"{side}_finger_{digit}",
            base,
            knuckle,
            0.0013 if digit == "III" else 0.0019,
            material="body_mat",
            mass="0",
            contype="0",
            conaffinity="0",
        )
        if digit != "III":
            ET.SubElement(
                arm,
                "geom",
                name=f"{side}_manual_claw_{digit}",
                type="mesh",
                mesh="manual_ungual",
                pos=numbers(knuckle),
                quat=".7071067812 0 .7071067812 0",
                material="claw_mat",
                mass="0",
                contype="0",
                conaffinity="0",
            )


def axial_body(asset, pelvis, hinge, p):
    """Upper body, head and arms. The jaw and neck retain independent pitch."""
    # Light ventrum and darker dorsum are a reconstruction choice. Map this
    # gradient by ring height so the shading stays continuous along the trunk.
    ET.SubElement(
        asset,
        "texture",
        name="hide_gradient",
        type="2d",
        builtin="gradient",
        width="16",
        height="64",
        rgb1=".39 .31 .23",
        rgb2=".66 .54 .40",
    )
    ET.SubElement(asset, "material", name="hide_mat", texture="hide_gradient", specular=".08", shininess=".1")
    ET.SubElement(asset, "material", name="iris_mat", rgba=".37 .25 .11 1", specular=".2", shininess=".22")
    ET.SubElement(asset, "material", name="mouth_mat", rgba=".26 .20 .16 1", specular=".08")

    # Tail root, ilium, deepest chest, and shoulder narrow into the neck.
    skin(
        asset,
        pelvis,
        "trunk_skin",
        [
            (-90, 26, 16, 16),
            (-79, 26, 23, 22),
            (-53, 24, 31, 32),
            (-25, 23, 36, 37),
            (7, 23, 38, 42),
            (41, 24, 37, 46),
            (67, 27, 34, 45),
            (90, 32, 30, 40),
            (110, 38, 25, 32),
            (130, 45, 21, 25),
            (148, 53, 17, 18),
            (160, 57.5, 12, 12),
        ],
        p["torso_mass_kg"],
    )
    # Smooth contact proxies follow the main volumes; rendering the same mass
    # mesh as a convex collider would bridge the throat and hollow of the neck.
    inert = dict(mass="0", group="3")
    ET.SubElement(pelvis, "geom", name="torso", type="ellipsoid", pos=".020 0 .025", size=".108 .036 .041", **inert)
    capsule(pelvis, "breast_contact", [0.070, 0, 0.033], [0.139, 0, 0.051], 0.018, **inert)
    neck = ET.SubElement(pelvis, "body", name="neck", pos=".145 0 .052")
    hinge(neck, "neck_pitch", "0 1 0", [-0.55, 0.55], 8, 0.12, 0.4)
    skin(
        asset,
        neck,
        "cervical_skin",
        [
            (-9, -2, 14, 15),
            (6, 2, 16.5, 17),
            (24, 8, 17, 17),
            (44, 18, 15.5, 15.5),
            (62, 33, 14, 14.5),
            (78, 50, 12.5, 14),
            (95, 64, 11.5, 12.5),
            (112, 73, 10.5, 11),
            (128, 76, 9.5, 10),
            (146, 75, 10, 10),
        ],
        0.040,
    )
    capsule(neck, "neck_geom", [0, 0, 0], [0.048, 0, 0.022], 0.012, **inert)
    capsule(neck, "upper_neck_geom", [0.048, 0, 0.022], [0.100, 0, 0.067], 0.011, **inert)
    capsule(neck, "nape_contact", [0.100, 0, 0.067], [0.146, 0, 0.075], 0.008, **inert)

    head = ET.SubElement(neck, "body", name="skull", pos=".151 0 .074")
    # Canjuers skull (~100 mm) scaled by femur 95/108.8. The orbit is the
    # deepest region, thinning into an 87 mm long wedge with a low rostrum.
    skin(
        asset,
        head,
        "cranial_skin",
        [
            (-20, 4, 3.5, 5.5),
            (-17, 5, 10, 11.5),
            (-10, 7, 13.5, 14),
            (-1, 7.5, 14.2, 15),
            (8, 7, 13.7, 14.4),
            (18, 5.6, 11.8, 11.7),
            (32, 3.4, 9.4, 8.4),
            (47, 1.8, 7.1, 6.3),
            (59, 0.6, 5.2, 4.9),
            (67, -0.6, 1.2, 2),
        ],
        0.045,
    )
    ET.SubElement(head, "geom", name="skull_geom", type="ellipsoid", pos="-.001 0 .007", size=".019 .013 .014", **inert)
    capsule(head, "muzzle_geom", [0.015, 0, 0.004], [0.061, 0, -0.0002], 0.004, **inert)
    jaw = ET.SubElement(head, "body", name="jaw", pos="-.010 0 -.003")
    hinge(jaw, "jaw_pitch", "0 1 0", [-0.04, 0.60], 0.3, 0.01, 0.04)
    skin(
        asset,
        jaw,
        "mandible_skin",
        [
            (-3, 0, 4, 3),
            (2, -1.5, 9.6, 5.0),
            (13, -3.0, 11.4, 5.5),
            (28, -3.0, 9.4, 4.7),
            (47, -2.2, 7.2, 3.6),
            (65, -1.7, 4.9, 2.8),
            (76, -0.8, 0.9, 1.4),
        ],
        0.007,
        material="belly_mat",
    )
    capsule(jaw, "jaw_geom", [0.003, 0, -0.002], [0.073, 0, -0.001], 0.0019, **inert)
    longitudinal_mesh(
        asset,
        "manual_ungual",
        np.array(
            [
                [0, 0, 1.25, 1.5],
                [2, 1, 1.25, 1.35],
                [4.5, 1.9, 0.8, 1],
                [7, 1.5, 0.17, 0.22],
            ]
        )
        / 1000,
    )
    longitudinal_mesh(
        asset,
        "brachial_skin",
        np.array(
            [
                [-10, 0, 2, 3],
                [-5, 0, 6, 7],
                [4, 0, 8, 9.5],
                [15, 0, 8.2, 8.4],
                [29, 0, 5.5, 6.5],
                [43, 0, 4.2, 4.6],
                [48, 0, 2.6, 3],
            ]
        )
        / 1000,
    )
    longitudinal_mesh(
        asset,
        "antebrachial_skin",
        np.array(
            [
                [-2, 0, 3, 3],
                [4, 0, 4.6, 5.2],
                [12, 0, 4.6, 5.0],
                [27, 0, 3.5, 3.7],
                [38, 0, 2.3, 2.6],
                [41, 0, 1.8, 2],
            ]
        )
        / 1000,
    )
    for side, sign in (("l", 1), ("r", -1)):
        face_detail(head, f"{side}_orbit", [0, sign * 0.0137, 0.0105], [0.0063, 0.0016, 0.0050], "head_mat")
        face_detail(head, f"{side}_iris", [0.0007, sign * 0.0149, 0.0107], [0.0041, 0.0010, 0.0040], "iris_mat")
        face_detail(head, f"{side}_pupil", [0.0012, sign * 0.0155, 0.0108], [0.0025, 0.00065, 0.0026], "eye_mat")
        face_detail(head, f"{side}_naris", [0.049, sign * 0.0068, 0.0032], [0.0022, 0.0005, 0.0010], "mouth_mat")
        folded_arm(pelvis, side, sign)
    return head
