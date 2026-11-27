"""Rebuild the anatomical MJCF and its PNG hide texture in SI units.

Run `python environments/allosaurus_fragilis/scripts/build_anatomy.py` from
the repository root. Asset generation uses the Python standard library.
"""

from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

if TYPE_CHECKING or __package__:
    from .surface import add_surface
else:  # Allow a direct, standard-library-only invocation.
    from surface import add_surface

ASSETS = Path(__file__).resolve().parents[1] / "anatomical" / "assets"
MODEL_NAME = "allosaurus_fragilis_anatomical.xml"

# Local neutral joint-to-joint vectors. Medial offsets take each foot slightly
# inside its hip. Pitch is about +Y on both sides and roll is about +X.
FEMUR = (0.30, -0.015, -0.835)
TIBIA = (-0.34, -0.035, -0.66)
METATARSUS = (0.115, -0.01, -0.36)
HIP_Z = 1.8995


def vec(values):
    return " ".join(f"{value:.6g}" for value in values)


def mirror(v, side):
    return v[0], v[1] * side, v[2]


def body(parent, name, pos, mass, center, inertia):
    node = ET.SubElement(parent, "body", name=name, pos=vec(pos))
    ET.SubElement(node, "inertial", mass=str(mass), pos=vec(center), diaginertia=vec(inertia))
    return node


def hidden_collision(node, name, kind="ellipsoid", **shape):
    return ET.SubElement(node, "geom", name=name, type=kind, **{"class": "collision"}, **shape)


def make_legs(pelvis, actuator, sensors, bodies):
    for prefix, sign in (("l", 1), ("r", -1)):
        femur, tibia, mt = [mirror(p, sign) for p in (FEMUR, TIBIA, METATARSUS)]
        thigh = body(
            pelvis, f"{prefix}_thigh", (0, 0.25 * sign, 0), 139, mirror((0.106, 0.02, -0.27), sign), (9.3, 10.8, 4.0)
        )
        shin = body(thigh, f"{prefix}_shin", femur, 35, mirror((-0.135, -0.014, -0.26), sign), (2.15, 2.30, 0.90))
        metatarsus = body(
            shin, f"{prefix}_metatarsus", tibia, 8, mirror((0.051, -0.005, -0.157), sign), (0.098, 0.128, 0.052)
        )
        foot = body(metatarsus, f"{prefix}_foot", mt, 5.6, (0.1, 0, -0.006), (0.028, 0.077, 0.092))
        for node in (thigh, shin, metatarsus, foot):
            bodies[node.get("name")] = node
        for name, node, axis, bounds, damping, armature, limit in [
            ("hip_roll", thigh, "1 0 0", (-0.38, 0.38), 110, 2.4, 9500),
            ("hip_pitch", thigh, "0 1 0", (-0.76, 0.76), 100, 3.5, 12500),
            ("knee", shin, "0 1 0", (-0.55, 1.08), 80, 2.5, 11000),
            ("ankle", metatarsus, "0 1 0", (-1.10, 0.60), 48, 1.3, 8500),
            ("mtp", foot, "0 1 0", (-0.95, 0.95), 18, 0.48, 4000),
        ]:
            full = f"{prefix}_{name}"
            ET.SubElement(
                node,
                "joint",
                name=full,
                type="hinge",
                axis=axis,
                range=vec(bounds),
                damping=str(damping),
                armature=str(armature),
            )
            ET.SubElement(actuator, "motor", name=full, joint=full, gear="1", ctrlrange=vec((-limit, limit)))
        hidden_collision(
            thigh,
            f"{prefix}_thigh_collision",
            "capsule",
            fromto=vec((0.03, 0.02 * sign, -0.09, 0.23, 0, -0.67)),
            size=".13",
        )
        hidden_collision(
            shin,
            f"{prefix}_shin_collision",
            "capsule",
            fromto=vec((-0.045, 0, -0.1, -0.29, -0.025 * sign, -0.56)),
            size=".068",
        )
        hidden_collision(
            metatarsus,
            f"{prefix}_mt_collision",
            "capsule",
            fromto=vec((0, 0, -0.04, 0.09, -0.005 * sign, -0.28)),
            size=".041",
        )
        # The metatarsophalangeal pad and the plantar surfaces of digits II,
        # III and IV all touch the floor. The medially placed hallux is clear.
        # The pads collide with the ground, never with another part of the animal.
        ET.SubElement(
            foot,
            "geom",
            name=f"{prefix}_pad",
            type="ellipsoid",
            pos=".014 0 -.002",
            size=".082 .072 .043",
            **{"class": "pad"},
        )
        for digit, start, end, radius in [
            (2, (0.048, -0.036), (0.221, -0.141), 0.030),
            (3, (0.058, 0.0), (0.295, 0.0), 0.031),
            (4, (0.044, 0.038), (0.204, 0.14), 0.030),
        ]:
            z = -0.045 + radius
            ET.SubElement(
                foot,
                "geom",
                name=f"{prefix}_pad_d{digit}",
                type="capsule",
                fromto=vec((start[0], start[1] * sign, z, end[0], end[1] * sign, z)),
                size=vec((radius,)),
                **{"class": "pad"},
            )
            ET.SubElement(
                foot,
                "site",
                name=f"{prefix}_toe_d{digit}",
                pos=vec((end[0], end[1] * sign, -0.045)),
                size=".004",
                group="4",
            )
        ET.SubElement(
            metatarsus, "site", name=f"{prefix}_hallux", pos=vec((-0.05, -0.10 * sign, -0.29)), size=".004", group="4"
        )
        ET.SubElement(
            foot,
            "site",
            name=f"{prefix}_foot",
            pos=".105 0 -.0445",
            size=".236 .18 .085",
            type="box",
            group="4",
            rgba="0 0 0 0",
        )
    for prefix in ("r", "l"):
        ET.SubElement(sensors, "touch", name=f"{prefix}_foot_touch", site=f"{prefix}_foot")


def make_model():
    root = ET.Element("mujoco", model="allosaurus_fragilis_anatomical")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=".002", gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic")
    ET.SubElement(root, "statistic", center="-.54 0 1.38", extent="4.45")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", friction="1.10 .006 .0005", condim="3", solref=".012 1", solimp=".92 .99 .002")
    for name, attrs in [
        ("collision", dict(contype="2", conaffinity="1", group="3", rgba=".7 .3 .1 1")),
        ("pad", dict(contype="2", conaffinity="1", group="3", rgba=".6 .2 .1 1")),
        ("hide", dict(contype="0", conaffinity="0", group="1", mass="0")),
    ]:
        ET.SubElement(ET.SubElement(default, "default", {"class": name}), "geom", **attrs)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", azimuth="93", elevation="-7", offwidth="1280", offheight="800")
    ET.SubElement(visual, "headlight", ambient=".34 .34 .34", diffuse=".40 .40 .40", specular=".07 .07 .07")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".69 .75 .75",
        rgb2=".85 .86 .79",
        width="128",
        height="768",
    )
    ET.SubElement(
        asset,
        "texture",
        name="ground_texture",
        type="2d",
        builtin="checker",
        width="64",
        height="64",
        rgb1=".44 .46 .41",
        rgb2=".48 .50 .45",
    )
    ET.SubElement(
        asset,
        "material",
        name="ground_mat",
        texture="ground_texture",
        texuniform="true",
        texrepeat="1 1",
        specular=".08",
        shininess=".08",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        name="sun",
        directional="true",
        pos="-2 -4 6",
        dir=".34 .45 -.84",
        diffuse=".80 .73 .62",
        specular=".18 .16 .12",
    )
    ET.SubElement(
        world,
        "light",
        name="skyfill",
        directional="true",
        pos="2 5 5",
        dir="-.3 -.5 -.75",
        diffuse=".32 .38 .42",
        castshadow="false",
    )
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="0 0 .1", material="ground_mat", contype="1", conaffinity="2"
    )
    # The scene has no mocap body: the walking course is measured from the
    # spawn along its heading (base_env.py), not toward a placed target.
    # All segment masses are reconstruction choices. Inertias are explicit so
    # changes to the artist's surface cannot change the contact or mass model.
    pelvis = body(world, "pelvis", (0, 0, HIP_Z), 621, (0.64, 0, 0.025), (88, 234, 218))
    ET.SubElement(pelvis, "freejoint", name="root")
    hidden_collision(pelvis, "thorax_collision", pos=".49 0 .06", size="1.05 .34 .43")
    hidden_collision(pelvis, "pelvic_collision", pos="-.1 0 .11", size=".53 .38 .39")
    ET.SubElement(pelvis, "site", name="imu", size=".012", group="4")
    neck = body(pelvis, "neck", (1.20, 0, 0.21), 104, (0.44, 0, 0.18), (2.3, 9.1, 8.5))
    head = body(neck, "head", (0.82, 0, 0.42), 61, (0.36, 0, 0.013), (1.7, 4.4, 4.1))
    hidden_collision(head, "head_collision", pos=".40 0 .02", size=".42 .115 .15")
    ET.SubElement(head, "site", name="head_tip", pos=".84 0 .025", size=".01", group="4")
    tail = body(pelvis, "tail_base", (-0.52, 0, 0.15), 233, (-0.62, 0, -0.075), (8.3, 57, 55))
    ET.SubElement(
        tail,
        "joint",
        name="tail_base_yaw",
        axis="0 0 1",
        range="-.16 .16",
        stiffness="9600",
        damping="490",
        armature="4",
    )
    tip = body(tail, "tail_distal", (-1.53, 0, -0.135), 88, (-0.73, 0, -0.065), (1.5, 30.4, 30.1))
    ET.SubElement(
        tip,
        "joint",
        name="tail_distal_yaw",
        axis="0 0 1",
        range="-.25 .25",
        stiffness="2500",
        damping="175",
        armature="1.1",
    )
    hidden_collision(tail, "tail_collision", "capsule", fromto="-.04 0 -.015 -1.43 0 -.14", size=".125")
    hidden_collision(tip, "tail_distal_collision", "capsule", fromto="0 0 0 -2.22 0 .015", size=".034")
    ET.SubElement(tip, "site", name="tail_tip", pos="-2.35 0 .035", size=".012", group="4")
    bodies = {b.get("name"): b for b in (pelvis, neck, head, tail, tip)}
    actuators = ET.SubElement(root, "actuator")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "gyro", name="pelvis_gyro", site="imu")
    ET.SubElement(sensors, "accelerometer", name="pelvis_accel", site="imu")
    ET.SubElement(sensors, "framequat", name="pelvis_orientation", objtype="site", objname="imu")
    make_legs(pelvis, actuators, sensors, bodies)
    # Symmetric static inverse dynamics at the home pose, sharing the vertical
    # reaction force equally at the projected whole-animal centre of mass.
    left_hold = (445.91355, 995.32260, -898.16436, 1481.98689, 656.235045)
    home_ctrl = (*left_hold, -left_hold[0], *left_hold[1:])
    ET.SubElement(
        ET.SubElement(root, "keyframe"),
        "key",
        name="home",
        qpos=vec([0, 0, HIP_Z, 1, 0, 0, 0] + [0] * 12),
        ctrl=vec(home_ctrl),
    )
    return root, asset, bodies


def write(assets: Path = ASSETS):
    root, asset, bodies = make_model()
    add_surface(asset, bodies, assets)
    ET.indent(root)
    (assets / MODEL_NAME).write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n")


if __name__ == "__main__":
    write()
