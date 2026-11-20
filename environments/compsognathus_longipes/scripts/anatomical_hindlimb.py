"""French-specimen-scaled hindlimbs and a lumped ankle/digit flexor."""

import numpy as np

from environments.compsognathus_longipes.scripts.anatomical_surface import longitudinal_mesh
from environments.compsognathus_longipes.scripts.styling import add, numbers


def hindlimb_meshes(asset, p):
    # X runs along the long bone. The shaft tapers below the proximal muscle belly.
    femur, tibia = p["femur_m"], p["tibia_m"]
    longitudinal_mesh(
        asset,
        "thigh_surface",
        [
            (-0.12 * femur, 0, 0.009, 0.013),
            (0, -0.002, 0.0175, 0.024),
            (0.25 * femur, -0.003, 0.018, 0.025),
            (0.58 * femur, -0.002, 0.014, 0.019),
            (0.9 * femur, 0, 0.009, 0.011),
            (femur, 0, 0.009, 0.010),
            (1.055 * femur, 0, 0.004, 0.004),
        ],
    )
    longitudinal_mesh(
        asset,
        "calf_surface",
        [
            (-0.045 * tibia, 0, 0.006, 0.006),
            (0, 0, 0.009, 0.010),
            (0.22 * tibia, -0.003, 0.012, 0.015),
            (0.42 * tibia, -0.003, 0.011, 0.013),
            (0.78 * tibia, 0, 0.005, 0.006),
            (tibia, 0, 0.0055, 0.006),
            (1.025 * tibia, 0, 0.003, 0.003),
        ],
    )
    # Mildly recurved pedal ungual. Scale the smaller hallux claw separately.
    for name, scale in (("pedal_ungual", 1), ("hallux_ungual", 0.53)):
        longitudinal_mesh(
            asset,
            name,
            np.array(
                [
                    (-0.002, 0, 0.0024, 0.002),
                    (0, 0, 0.0027, 0.0024),
                    (0.0035, 0.0003, 0.0022, 0.002),
                    (0.007, -0.0003, 0.0014, 0.0013),
                    (0.011, -0.0020, 0.00025, 0.00025),
                ]
            )
            * scale,
        )


def add_long_bone(body, name, mesh, displacement, mass):
    pitch = np.arctan2(-displacement[2], displacement[0])
    add(body, "geom", name=name, type="mesh", mesh=mesh, euler=numbers([0, pitch, 0]), mass=mass)


def add_pes(meta, foot, side, sign, meta_offset):
    # Digit I starts halfway down the medial metatarsus. Its orientation is a
    # pose assumption. It stays elevated over the three forward weight-bearing digits.
    origin = 0.53 * meta_offset + np.array([0, -sign * 0.004, 0])
    end = origin + [-0.011, -sign * 0.0025, -0.005]
    add(meta, "geom", name=f"{side}_hallux", type="capsule", fromto=numbers([*origin, *end]), size=".0021", mass="0")
    add(
        meta,
        "geom",
        name=f"{side}_claw_hallux",
        type="mesh",
        mesh="hallux_ungual",
        pos=numbers(end),
        euler=numbers([0, 0, np.arctan2(end[1] - origin[1], end[0] - origin[0])]),
        material="claw_mat",
        mass="0",
        contype="0",
        conaffinity="0",
    )

    # A cushion under the metatarsal heads and individual separated digital
    # pads make four contact geoms on the existing shared MTP/sensor body.
    # The named, flat contact inset preserves the sole-corner/CoP telemetry.
    # The pad outside that compressed area rounds up toward the digit roots.
    add(
        foot,
        "geom",
        name=f"{side}_plantar_pad",
        type="box",
        pos="-.001 0 -.0045",
        size=".009 .008 .0025",
        mass=".008",
        group="3",
    )
    add(
        foot,
        "geom",
        name=f"{side}_metatarsal_cushion",
        type="ellipsoid",
        pos="-.001 0 -.002",
        size=".014 .013 .005",
        material="belly_mat",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    # Pad ends exclude the ungual. Digit II including its claw is ~49 mm
    # at this scale (CNJ79 digit II/femur); III is longest, II and IV similar.
    for digit, direction, length, lateral, radius in (
        (2, -sign, 0.038, 0.014, 0.0034),
        (3, 0, 0.051, 0, 0.0038),
        (4, sign, 0.039, 0.015, 0.0032),
    ):
        start = np.array([0.001, direction * 0.005, radius - 0.007])
        axis = np.array([np.sqrt(length**2 - lateral**2), direction * lateral, 0])
        end = start + axis
        yaw = np.arctan2(axis[1], axis[0])
        add(
            foot,
            "geom",
            name=f"{side}_toe_d{digit}_geom",
            type="capsule",
            fromto=numbers([*start, *end]),
            size=radius,
            mass=".004",
        )
        # Subtle dorsal joints show the phalangeal subdivisions. Distal
        # articulation is aggregated into the shared MTP degree of freedom.
        for knuckle in range(1, digit):
            center = start + axis * knuckle / digit + [0, 0, 0.001]
            add(
                foot,
                "geom",
                name=f"{side}_d{digit}_knuckle_{knuckle}",
                type="ellipsoid",
                pos=numbers(center),
                size=numbers([0.0042, radius * 1.12, radius]),
                euler=numbers([0, 0, yaw]),
                mass="0",
                contype="0",
                conaffinity="0",
            )
        add(
            foot,
            "geom",
            name=f"{side}_claw_d{digit}",
            type="mesh",
            mesh="pedal_ungual",
            pos=numbers(end + [0, 0, 0.001]),
            euler=numbers([0, 0, yaw]),
            material="claw_mat",
            mass="0",
            contype="0",
            conaffinity="0",
        )
    add(foot, "site", name=f"{side}_sole", pos="-.001 0 -.007", size=".002", group="4")
    add(
        foot, "site", name=f"{side}_foot_touch_volume", type="box", pos=".023 0 -.002", size=".042 .029 .007", group="4"
    )


def add_digital_flexor(tendons, side, p):
    # Fixed moment arms express tendon excursion in metres. Ankle/MTP
    # dorsiflexion stretches the flexor, which can pull but cannot push.
    # Across the entire joint range shortening fits in the slack interval.
    ankle_arm, toe_arm = p["digital_flexor_moment_arms_m"]
    tendon = add(
        tendons,
        "fixed",
        name=f"{side}_digital_flexor",
        stiffness=p["digital_flexor_stiffness_n_m"],
        springlength="-.02 0",
    )
    add(tendon, "joint", joint=f"{side}_ankle", coef=-ankle_arm)
    add(tendon, "joint", joint=f"{side}_toe", coef=-toe_arm)
