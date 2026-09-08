"""Regenerate both MJCFs from checked-in parameters and the pinned Rev B plant.

Run from the repository root: python -m environments.compsognathus.scripts.build_models
Geometry is in metres; angles and position controls are radians. These assets
deliberately have no Gymnasium registration or training certification.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from environments.compsognathus.scripts.standing_targets import write_gravity_targets
from environments.compsognathus.scripts.styling import apply_styles

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / "data/model_parameters.json"


def fmt(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def element(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attributes.items()})


def scene(root, world, asset):
    visual = element(root, "visual")
    element(visual, "global", offwidth="1440", offheight="1000")
    element(visual, "quality", shadowsize="2048")
    element(visual, "headlight", diffuse="0.65 0.65 0.65", ambient="0.35 0.35 0.35")
    element(root, "statistic", center="0 0 0.2", extent="0.9")
    element(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".12 .18 .23",
        rgb2=".25 .34 .40",
        width="128",
        height="128",
    )
    element(
        asset,
        "texture",
        name="ground_grid",
        type="2d",
        builtin="checker",
        rgb1="0.15 0.19 0.23",
        rgb2="0.19 0.23 0.27",
        width="256",
        height="256",
    )
    element(asset, "material", name="ground_mat", texture="ground_grid", texrepeat="10 10", texuniform="true")
    floor = world.find("geom[@name='floor']")
    if floor is None:
        floor = element(world, "geom", name="floor", type="plane")
    floor.attrib.update(
        size="3 3 0.05",
        contype="2",
        conaffinity="1",
        friction="0.8 0.005 0.0001",
        material="ground_mat",
        rgba="1 1 1 1",
        condim="3",
    )
    element(world, "light", name="key_light", pos="0.5 -0.8 1.5", dir="-0.3 0.4 -1", directional="true")
    element(world, "light", name="fill_light", pos="-1 0.5 1", dir="0.5 -0.3 -1", diffuse="0.4 0.4 0.4")
    prey = element(world, "body", name="prey", mocap="true", pos="1.2 0 0.04")
    element(
        prey,
        "geom",
        name="prey_geom",
        type="sphere",
        size="0.035",
        mass="0.001",
        contype="0",
        conaffinity="0",
        rgba="0.92 0.35 0.19 1",
    )


def sensors(root, pelvis, head, left_foot, right_foot):
    element(pelvis, "site", name="imu", pos="0 0 0.025", size="0.003", group="4")
    element(head, "site", name="head_tip", pos="0.035 0 0", size="0.003", group="4")
    # Camera looks along +X: image right = -Y, image up = +Z.
    element(head, "camera", name="head_camera", pos="0.032 0 0.008", xyaxes="0 -1 0 0 0 1", fovy="70")
    sensor = element(root, "sensor")
    element(sensor, "gyro", name="pelvis_gyro", site="imu")
    element(sensor, "accelerometer", name="pelvis_accel", site="imu")
    element(sensor, "framequat", name="pelvis_quat", objtype="body", objname="pelvis")
    element(sensor, "touch", name="r_foot_touch", site=right_foot)
    element(sensor, "touch", name="l_foot_touch", site=left_foot)
    element(sensor, "framelinvel", name="pelvis_velocity", objtype="body", objname="pelvis")
    for joint in root.findall(".//joint[@name]"):
        name = joint.get("name")
        element(sensor, "jointpos", name=f"{name}_position", joint=name)
        element(sensor, "jointvel", name=f"{name}_velocity", joint=name)


def write(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_robot(parameters):
    source = ROOT / "references/compso_rev_b.xml"
    if hashlib.sha256(source.read_bytes()).hexdigest() != parameters["source_xml_sha256"]:
        raise ValueError("Rev B reference changed: review provenance before regenerating")
    p = parameters["robot"]
    root = ET.parse(source).getroot()
    root.set("model", "compsognathus_robot_rev_b_dynamics_v1")
    root.insert(
        0, ET.Comment("Generated prototype. Rev B geometry/inertias preserved. See README and references/NOTICE.md.")
    )
    root.find("compiler").attrib.update(meshdir="meshes", autolimits="true")
    root.find("option").attrib.update(timestep="0.002", iterations="100", cone="elliptic")
    root.find("default/joint").attrib.update(damping="0.005", armature=str(p["joint_armature_kg_m2"]))
    root.find("default/geom").attrib.update(
        friction="0.8 0.005 0.0001", condim="3", solref="0.008 1", solimp="0.95 0.99 0.001", margin=".00005"
    )
    world, asset = root.find("worldbody"), root.find("asset")
    scene(root, world, asset)
    pelvis = world.find("body[@name='pelvis']")
    pelvis.find("freejoint").set("name", "root")
    # Keep all mechanical transforms/inertials. Within-leg mating parts are
    # excluded by mask; opposite legs and head/tail collide with each other.
    # The aggregate core allowance is only a floor collider, not fit-ready CAD.
    for side, branch, ctype, affinity in (
        ("left", "hip_roll_assembly", "5", "26"),
        ("right", "hip_roll_assembly_2", "9", "22"),
    ):
        for geom in pelvis.find(f"body[@name='{branch}']").iter("geom"):
            geom.attrib.update(contype=ctype, conaffinity=affinity)
        sole = pelvis.find(f".//body[@name='{side}_rolling_foot']")
        element(
            sole,
            "site",
            name=f"{side}_foot_touch_volume",
            type="box",
            pos="0 0 -.034",
            size=".0505 .0455 .005",
            group="4",
        )
    for name in ("core", "fixed_head", "passive_tail"):
        geom = pelvis.find(f"body[@name='{name}']/geom")
        geom.attrib.update(contype="1" if name == "core" else "17", conaffinity="2" if name == "core" else "14")
    core = pelvis.find("body[@name='core']")
    core.find("geom").set("rgba", ".82 .57 .18 1")
    head = pelvis.find("body[@name='fixed_head']")
    head.find("geom").set("rgba", ".91 .72 .32 1")
    tail = pelvis.find("body[@name='passive_tail']")
    tail.find("geom").set("rgba", ".29 .45 .50 1")
    element(tail, "site", name="tail_tip", pos="-.12 0 0", size="0.003", group="4")
    # Decorative mounts use the existing inertial allowances (no added mass).
    element(
        core,
        "geom",
        name="neck_mount_visual",
        type="capsule",
        fromto=".05 0 .015 .082 0 .04",
        size=".006",
        contype="0",
        conaffinity="0",
        rgba=".2 .28 .32 1",
    )
    element(
        core,
        "geom",
        name="tail_mount_visual",
        type="capsule",
        fromto="-.055 0 .005 -.075 0 .01",
        size=".005",
        contype="0",
        conaffinity="0",
        rgba=".2 .28 .32 1",
    )
    element(
        head,
        "geom",
        name="camera_lens_visual",
        type="sphere",
        pos=".030 0 .008",
        size=".005",
        contype="0",
        conaffinity="0",
        rgba=".05 .08 .1 1",
    )
    actuator = root.find("actuator")
    for motor in list(actuator):
        name = motor.get("joint")
        joint = root.find(f".//joint[@name='{name}']")
        rated = p["hip_roll_rated_torque_nm"] if name.endswith("hip_roll") else p["other_rated_torque_nm"]
        cap = rated * p["regulated_voltage_assumption"] / p["rating_voltage"] * p["capacity_derating"]
        # kv lives INSIDE the force clamp; large unbounded passive damping
        # must not provide fictitious support beyond the servo torque budget.
        motor.tag = "position"
        motor.attrib = dict(
            name=f"{name}_act",
            joint=name,
            gear="1",
            kp=str(p["kp"]),
            kv=str(p["kv"]),
            ctrllimited="true",
            ctrlrange=joint.get("range"),
            forcelimited="true",
            forcerange=fmt([-cap, cap]),
        )
    sensors(root, pelvis, head, "left_foot_touch_volume", "right_foot_touch_volume")
    home = p["home_qpos"]
    element(element(root, "keyframe"), "key", name="home", qpos=fmt(home), ctrl=fmt(home[7:]))
    apply_styles(root, "robot", parameters)
    write(root, ROOT / "assets/compsognathus_robot.xml")
    write_gravity_targets(ROOT / "assets/compsognathus_robot.xml")


def build_biological(parameters):
    p = parameters["biological"]
    root = ET.Element("mujoco", model="compsognathus_biological_v1")
    root.append(ET.Comment("Generated anatomy-inspired proxy, not a fossil-fitted musculoskeletal reconstruction."))
    element(root, "compiler", angle="radian", autolimits="true")
    element(
        root,
        "option",
        timestep="0.002",
        gravity="0 0 -9.81",
        integrator="implicitfast",
        iterations="100",
        cone="elliptic",
    )
    default = element(root, "default")
    element(default, "joint", damping="0.01", armature="0.00005", limited="true")
    element(
        default,
        "geom",
        contype="1",
        conaffinity="3",
        friction="0.8 0.005 0.0001",
        condim="3",
        solref="0.006 1",
        solimp="0.95 0.99 0.001",
        margin=".00005",
        rgba=".34 .53 .35 1",
    )
    asset, world = element(root, "asset"), element(root, "worldbody")
    scene(root, world, asset)
    displacements = []
    for part, key in (
        ("femur", "femur_forward_m"),
        ("tibia", "tibia_forward_m"),
        ("metatarsus", "metatarsus_forward_m"),
    ):
        x, length = p[key], p[f"{part}_m"]
        displacements.append(np.array([x, 0, -np.sqrt(length**2 - x**2)]))
    hip_z = -sum(d[2] for d in displacements) + 0.007
    pelvis = element(world, "body", name="pelvis", pos=fmt([0, 0, hip_z]))
    element(pelvis, "freejoint", name="root")
    element(
        pelvis, "geom", name="torso", type="ellipsoid", pos=".025 0 .025", size=".11 .038 .044", mass=p["torso_mass_kg"]
    )
    actuator = element(root, "actuator")

    def hinge(body, name, axis, limits, kp=20, kv=0.2, cap=1.0):
        element(body, "joint", name=name, axis=axis, range=fmt(limits))
        element(
            actuator,
            "position",
            name=f"{name}_act",
            joint=name,
            kp=kp,
            kv=kv,
            ctrlrange=fmt(limits),
            forcerange=fmt([-cap, cap]),
        )

    neck = element(pelvis, "body", name="neck", pos=".105 0 .05")
    hinge(neck, "neck_pitch", "0 1 0", [-0.55, 0.55], 8, 0.12, 0.4)
    element(neck, "geom", name="neck_geom", type="capsule", fromto="0 0 0 .10 0 .065", size=".014", mass=".04")
    skull = element(neck, "body", name="skull", pos=".13 0 .065")
    element(skull, "geom", name="skull_geom", type="ellipsoid", pos=".02 0 0", size=".058 .019 .022", mass=".045")
    jaw = element(skull, "body", name="jaw", pos="-.015 0 -.012")
    hinge(jaw, "jaw_pitch", "0 1 0", [-0.04, 0.6], 0.3, 0.01, 0.04)
    element(
        jaw,
        "geom",
        name="jaw_geom",
        type="capsule",
        fromto="0 0 0 .079 0 0",
        size=".005",
        mass=".007",
        rgba=".46 .61 .41 1",
    )
    for side in (-1, 1):
        element(
            skull,
            "geom",
            name=f"eye_{side}",
            type="sphere",
            pos=fmt([0, side * 0.018, 0.011]),
            size=".005",
            mass="0",
            contype="0",
            conaffinity="0",
            rgba=".035 .045 .025 1",
        )
        arm = element(pelvis, "body", name="l_arm" if side == 1 else "r_arm", pos=fmt([0.08, side * 0.035, 0.005]))
        element(
            arm,
            "geom",
            name=f"arm_geom_{side}",
            type="capsule",
            fromto=fmt([0, 0, 0, 0.025, side * 0.014, -0.038]),
            size=".007",
            mass=".008",
        )
        element(
            arm,
            "geom",
            name=f"hand_visual_{side}",
            type="capsule",
            fromto=fmt([0.025, side * 0.014, -0.038, 0.05, side * 0.014, -0.024]),
            size=".003",
            mass="0",
            contype="0",
            conaffinity="0",
        )
    parent = pelvis
    for i, (length, mass) in enumerate(zip(p["tail_segment_lengths_m"], p["tail_segment_masses_kg"], strict=True)):
        tail = element(
            parent,
            "body",
            name=f"tail_{i + 1}",
            pos="-.085 0 .026" if i == 0 else fmt([-p["tail_segment_lengths_m"][i - 1], 0, 0]),
        )
        if i == 0:
            hinge(tail, "tail_1_pitch", "0 1 0", [-0.35, 0.35], 10, 0.16, 0.7)
            hinge(tail, "tail_1_yaw", "0 0 1", [-0.45, 0.45], 8, 0.12, 0.5)
        else:
            element(
                tail, "joint", name=f"tail_{i + 1}_pitch", axis="0 1 0", range="-.2 .2", stiffness="1.5", damping=".08"
            )
        element(
            tail,
            "geom",
            name=f"tail_{i + 1}_geom",
            type="capsule",
            fromto=fmt([0, 0, 0, -length, 0, 0]),
            size=[0.018, 0.013, 0.009, 0.004][i],
            mass=mass,
        )
        parent = tail
    element(parent, "site", name="tail_tip", pos=fmt([-p["tail_segment_lengths_m"][-1], 0, 0]), size=".003", group="4")
    for side, sign in (("r", -1), ("l", 1)):
        thigh = element(
            pelvis,
            "body",
            name=f"{side}_thigh",
            pos=fmt([p["hip_forward_offset_m"], sign * p["foot_half_spacing_m"], 0]),
        )
        hinge(thigh, f"{side}_hip_pitch", "0 1 0", [-0.6, 0.7])
        hinge(thigh, f"{side}_hip_roll", "1 0 0", [-0.35, 0.35])
        element(
            thigh,
            "geom",
            name=f"{side}_thigh_geom",
            type="capsule",
            fromto=fmt([0, 0, 0, *displacements[0]]),
            size=".022",
            mass=".078",
        )
        tibia = element(thigh, "body", name=f"{side}_tibia", pos=fmt(displacements[0]))
        hinge(tibia, f"{side}_knee", "0 1 0", [-0.5, 1.0])
        element(
            tibia,
            "geom",
            name=f"{side}_tibia_geom",
            type="capsule",
            fromto=fmt([0, 0, 0, *displacements[1]]),
            size=".012",
            mass=".056",
        )
        meta = element(tibia, "body", name=f"{side}_metatarsus", pos=fmt(displacements[1]))
        hinge(meta, f"{side}_ankle", "0 1 0", [-0.65, 0.65], 16, 0.2, 0.8)
        element(
            meta,
            "geom",
            name=f"{side}_metatarsus_geom",
            type="capsule",
            fromto=fmt([0, 0, 0, *displacements[2]]),
            size=".006",
            mass=".018",
        )
        foot = element(meta, "body", name=f"{side}_foot", pos=fmt(displacements[2]))
        hinge(foot, f"{side}_toe", "0 1 0", [-0.4, 0.5], 12, 0.2, 0.6)
        element(
            foot,
            "geom",
            name=f"{side}_plantar_pad",
            type="box",
            pos=".017 0 -.002",
            size=".028 .016 .005",
            mass=".008",
            rgba=".35 .31 .22 1",
        )
        # Three load-bearing toes, all welded to this sensor's own body.
        for toe, y, length in ((2, -0.017, 0.049), (3, 0, 0.062), (4, 0.017, 0.045)):
            element(
                foot,
                "geom",
                name=f"{side}_toe_d{toe}_geom",
                type="capsule",
                fromto=fmt([0, 0, -0.002, length, y, -0.002]),
                size=".005",
                mass=".004",
                rgba=".38 .34 .23 1",
            )
        element(foot, "site", name=f"{side}_sole", pos=".017 0 -.007", size=".002", group="4")
        element(
            foot,
            "site",
            name=f"{side}_foot_touch_volume",
            type="box",
            pos=".027 0 -.002",
            size=".045 .03 .007",
            group="4",
        )
    sensors(root, pelvis, skull, "l_foot_touch_volume", "r_foot_touch_volume")
    # No ref-angle trick: geometry is authored in its flexed home pose.
    joint_count = len(root.findall(".//body/joint"))
    element(
        element(root, "keyframe"),
        "key",
        name="home",
        qpos=fmt([0, 0, hip_z, 1, 0, 0, 0] + [0] * joint_count),
        ctrl=fmt([0] * len(actuator)),
    )
    apply_styles(root, "biological", parameters)
    write(root, ROOT / "assets/compsognathus.xml")
    write_gravity_targets(ROOT / "assets/compsognathus.xml")


def main():
    parameters = json.loads(PARAMETERS.read_text())
    build_robot(parameters)
    build_biological(parameters)
    print("Generated assets/compsognathus.xml and assets/compsognathus_robot.xml")


if __name__ == "__main__":
    main()
