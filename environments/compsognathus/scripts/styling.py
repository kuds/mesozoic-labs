"""Repository-consistent primitive styling with explicit mass treatment.

Brown/tan materials match T-Rex, not a claim about fossil colour. The robot's
head budget is redistributed explicitly; all other robot inertials remain the
Rev B aggregate allowances. Inline tapered meshes need no downloaded artwork.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def add(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attributes.items()})


def tapered_mesh(asset, name, start, end, start_radius, end_radius):
    """Closed convex frustum along X; x decreases from attachment to tip."""
    count = 20
    vertices = [
        [x, radius * np.cos(a), radius * np.sin(a)]
        for x, radius in ((start, start_radius), (end, end_radius))
        for a in np.linspace(0, 2 * np.pi, count, endpoint=False)
    ]
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.extend([[i, count + j, j], [i, count + i, count + j]])
    for i in range(1, count - 1):
        faces.extend([[0, i, i + 1], [count, count + i + 1, count + i]])
    add(asset, "mesh", name=name, vertex=numbers(vertices), face=" ".join(str(n) for f in faces for n in f))


def set_material(geom, name):
    geom.attrib.pop("rgba", None)
    geom.set("material", name)


def robot_head(head, parts):
    # Replaces the old 45 g box allocation; no added mass or hidden ballast.
    for item in list(head):
        if item.tag in ("geom", "inertial"):
            head.remove(item)
    total = sum(p["mass_kg"] for p in parts)
    if not np.isclose(total, 0.045, atol=1e-12):
        raise ValueError("Styled head must remain inside its 45 g allocation")
    com = sum(p["mass_kg"] * np.array(p["pos_m"]) for p in parts) / total
    inertia = np.zeros((3, 3))
    for part in parts:
        mass, sizes = part["mass_kg"], np.array(part["size_m"])
        divisor = 5 if part["type"] == "ellipsoid" else 3
        diagonal = (
            mass
            / divisor
            * np.array([sizes[1] ** 2 + sizes[2] ** 2, sizes[0] ** 2 + sizes[2] ** 2, sizes[0] ** 2 + sizes[1] ** 2])
        )
        offset = np.array(part["pos_m"]) - com
        inertia += np.diag(diagonal) + mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))
        add(
            head,
            "geom",
            name=part["name"],
            type=part["type"],
            pos=numbers(part["pos_m"]),
            size=numbers(sizes),
            material=part["material"],
            contype="17",
            conaffinity="14",
            group="3" if part.get("internal") else "0",
        )
    add(
        head,
        "inertial",
        pos=numbers(com),
        mass=total,
        fullinertia=numbers([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]),
    )
    # Camera lens is inside the camera module's 6 g allowance. The optical
    # centre lies 1 mm ahead of the glass and ahead of all head geoms.
    add(
        head,
        "geom",
        name="camera_lens_visual",
        type="cylinder",
        pos=".0395 0 .008",
        quat=".707106781187 0 .707106781187 0",
        size=".006 .0015",
        material="lens_mat",
        contype="0",
        conaffinity="0",
    )
    for side in (-1, 1):
        add(
            head,
            "geom",
            name=f"eye_detail_{side}",
            type="sphere",
            pos=numbers([-0.006, side * 0.019, 0.010]),
            size=".0035",
            material="eye_mat",
            contype="0",
            conaffinity="0",
        )


def apply_styles(root, variant, parameters):
    asset = root.find("asset")
    # The first five are the T-Rex asset's exact palette.
    colours = {
        "body_mat": ".55 .45 .35 1",
        "belly_mat": ".60 .50 .40 1",
        "head_mat": ".50 .40 .30 1",
        "claw_mat": ".30 .25 .20 1",
        "tooth_mat": ".90 .88 .80 1",
        "eye_mat": ".035 .03 .025 1",
        "frame_mat": ".35 .38 .40 1",
        "servo_mat": ".13 .15 .17 1",
        "electronics_mat": ".12 .30 .23 1",
        "lens_mat": ".06 .13 .19 1",
        "sole_mat": ".16 .17 .18 1",
    }
    for name, rgba in colours.items():
        add(asset, "material", name=name, rgba=rgba, specular=".15", shininess=".12")
    default_geom = root.find("default/geom")
    default_geom.attrib.pop("rgba", None)
    default_geom.set("material", "body_mat")
    pelvis = root.find("worldbody/body[@name='pelvis']")
    for geom in pelvis.iter("geom"):
        set_material(geom, "body_mat")
    if variant == "robot":
        for geom in pelvis.iter("geom"):
            set_material(geom, "servo_mat" if "wj-wk" in geom.get("mesh", "") else "frame_mat")
            if geom.get("name") in ("left_sole", "right_sole"):
                set_material(geom, "sole_mat")
        core = pelvis.find("body[@name='core']")
        core_geom = core.find("geom[@name='core_envelope']")
        core_geom.set("type", "ellipsoid")
        set_material(core_geom, "body_mat")
        head = pelvis.find("body[@name='fixed_head']")
        robot_head(head, parameters["robot"]["head_parts"])
        tail = pelvis.find("body[@name='passive_tail']")
        tapered_mesh(asset, "robot_tail_shell", 0.135, -0.12, 0.010, 0.0015)
        tail_geom = tail.find("geom")
        tail_geom.attrib.pop("size")
        tail_geom.attrib.update(type="mesh", mesh="robot_tail_shell", material="body_mat")
        # The board is a hidden mounting marker inside the 505 g core budget.
        add(
            pelvis,
            "geom",
            name="imu_board_visual",
            type="box",
            pos=numbers(parameters["robot"]["imu_pos_m"]),
            size=".01 .008 .001",
            group="3",
            contype="0",
            conaffinity="0",
            material="electronics_mat",
        )
        pelvis.find("site[@name='imu']").set("pos", numbers(parameters["robot"]["imu_pos_m"]))
        camera_pos = parameters["robot"]["camera_pos_m"]
        tip_pos = [0.040, 0, -0.003]
    else:
        torso = pelvis.find("geom[@name='torso']")
        torso.attrib.pop("pos")
        torso.attrib.update(
            type="capsule",
            fromto="-.050 0 .015 .100 0 .035",
            size=".035",
            mass=str(parameters["biological"]["torso_mass_kg"] - 0.04),
        )
        add(
            pelvis,
            "geom",
            name="belly_geom",
            type="ellipsoid",
            pos=".025 0 .012",
            size=".07 .029 .023",
            mass=".04",
            material="belly_mat",
        )
        neck = pelvis.find("body[@name='neck']")
        neck.find("geom[@name='neck_geom']").attrib.update(fromto="0 0 0 .043 0 .046", size=".014", mass=".025")
        add(
            neck,
            "geom",
            name="upper_neck_geom",
            type="capsule",
            fromto=".043 0 .046 .102 0 .065",
            size=".012",
            mass=".015",
            material="body_mat",
        )
        head = neck.find("body[@name='skull']")
        skull = head.find("geom[@name='skull_geom']")
        skull.attrib.update(pos=".002 0 .003", size=".036 .019 .022", mass=".030", material="head_mat")
        add(
            head,
            "geom",
            name="muzzle_geom",
            type="ellipsoid",
            pos=".045 0 -.002",
            size=".033 .011 .011",
            mass=".015",
            material="head_mat",
        )
        jaw = head.find("body[@name='jaw']")
        set_material(jaw.find("geom"), "belly_mat")
        for geom in head.findall("geom"):
            if geom.get("name", "").startswith("eye_"):
                set_material(geom, "eye_mat")
        for side in (-1, 1):
            for i in range(4):
                x = 0.035 + i * 0.011
                add(
                    jaw,
                    "geom",
                    name=f"tooth_{side}_{i}",
                    type="capsule",
                    fromto=numbers([x, side * 0.004, 0, x, side * 0.004, 0.003]),
                    size=".0007",
                    mass="0",
                    contype="0",
                    conaffinity="0",
                    material="tooth_mat",
                )
        radii = [0.019, 0.014, 0.009, 0.0045, 0.0015]
        for i, length in enumerate(parameters["biological"]["tail_segment_lengths_m"]):
            name = f"tail_{i + 1}_taper"
            tapered_mesh(asset, name, 0.012 if i == 0 else 0, -length, radii[i], radii[i + 1])
            geom = pelvis.find(f".//geom[@name='tail_{i + 1}_geom']")
            geom.attrib.pop("fromto")
            geom.attrib.pop("size")
            geom.attrib.update(type="mesh", mesh=name, material="body_mat")
        for geom in pelvis.iter("geom"):
            if (
                "toe_d" in geom.get("name", "")
                or "plantar" in geom.get("name", "")
                or "hand_visual" in geom.get("name", "")
            ):
                set_material(geom, "claw_mat")
        camera_pos, tip_pos = [0.080, 0, 0.002], [0.078, 0, -0.002]
    camera = head.find("camera[@name='head_camera']")
    camera.attrib.update(pos=numbers(camera_pos), resolution="640 480")
    head.find("site[@name='head_tip']").set("pos", numbers(tip_pos))
    # Explicitly label privileged ground truth. Neither is raw IMU output.
    root.find("sensor/framequat").set("name", "diagnostic_pelvis_quat")
    root.find("sensor/framelinvel").set("name", "diagnostic_pelvis_velocity")
    custom = add(root, "custom")
    add(
        custom,
        "text",
        name="sensor_contract",
        data="compso-sensors/v1: gyro+accel+joint-encoders+RGB; touch is ideal/optional; diagnostic_* is simulator-only",
    )
