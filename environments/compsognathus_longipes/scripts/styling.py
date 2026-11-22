"""Repository-consistent primitive styling with explicit mass treatment.

Anatomical brown/tan materials match T-Rex, not a claim about fossil colour.
The robot's mechanical fairings retain the v1 head and Rev B core/leg mass
proxies. Inline meshes need no downloaded artwork.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from environments.compsognathus_longipes.scripts.anatomical_surface import longitudinal_mesh
from environments.compsognathus_longipes.scripts.robot_shells import robot_casings


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


def apply_styles(root, embodiment, parameters):
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
    if embodiment == "robot":
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
        robot_casings(root, parameters)
        camera_pos = parameters["robot"]["camera_pos_m"]
        tip_pos = [0.040, 0, -0.003]
    else:
        head = pelvis.find("body[@name='neck']/body[@name='skull']")
        radii = parameters["anatomical"]["tail_section_radii_m"]
        for i, length in enumerate(parameters["anatomical"]["tail_segment_lengths_m"]):
            name = f"tail_{i + 1}_taper"
            radius, tip = radii[i], radii[i + 1]
            longitudinal_mesh(asset, name, [(-length, 0, tip, tip), (0.008 if i == 0 else 0, 0, radius, radius)])
            geom = pelvis.find(f".//geom[@name='tail_{i + 1}_geom']")
            geom.attrib.pop("fromto")
            geom.attrib.pop("size")
            geom.attrib.update(type="mesh", mesh=name, material="hide_mat")
        for geom in pelvis.iter("geom"):
            if geom.get("material") is None:
                set_material(geom, "body_mat")
        camera_pos, tip_pos = [0.069, 0, 0.001], [0.067, 0, -0.0006]
    camera = head.find("camera[@name='head_camera']")
    camera.attrib.update(pos=numbers(camera_pos), resolution="640 480")
    head.find("site[@name='head_tip']").set("pos", numbers(tip_pos))
    # Explicitly label privileged ground truth. Neither is raw IMU output.
    root.find("sensor/framequat").set("name", "diagnostic_pelvis_quat")
    # MuJoCo's "body" denotes the inertial frame, whose principal axes can
    # rotate relative to the torso. Training and diagnostics need the pelvis
    # frame itself ("xbody"), including on the anatomical model.
    root.find("sensor/framequat").set("objtype", "xbody")
    root.find("sensor/framelinvel").set("name", "diagnostic_pelvis_velocity")
    root.find("sensor/framelinvel").set("objtype", "xbody")
    custom = add(root, "custom")
    add(
        custom,
        "text",
        name="sensor_contract",
        data="compsognathus-longipes-sensors/v1: gyro+accel+joint-encoders+RGB; touch is ideal/optional; diagnostic_* is simulator-only",
    )
