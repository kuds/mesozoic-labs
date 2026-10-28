"""Explicit part bodies and conservative installation reservations for the mini.

All dimensions are in the owning MJCF body's frame. Group 4 depicts the raw
part approximation; group 5 depicts the larger installation allowance. Neither
group participates in contact dynamics. These are packing requirements, not
fabrication CAD, thermal validation, or a qualified electrical design.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from environments.shared.robot_building import element, values

SOURCES = {
    "servo": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
    "computer": "https://docs.radxa.com/en/zero/zero3",
    "u2d2": "https://docs.robotis.com/docs/parts/interface/u2d2/",
    "motor_regulator": "https://www.pololu.com/product/2866/specs",
    "logic_regulator": "https://www.pololu.com/product/2858/specs",
    "imu": "https://learn.adafruit.com/lsm6dsox-and-ism330dhc-6-dof-imu/downloads",
    "imu_board_outline": "https://github.com/adafruit/Adafruit-LSM6DSOX-PCB/blob/master/Adafruit_LSM6DSOX.brd",
    "camera": "https://www.raspberrypi.com/documentation/accessories/camera.html#hardware-specifications",
    "camera_radxa_support": "https://docs.radxa.com/en/zero/zero3/accessories/camera",
    "allowance": "../components.json (unselected-part design allowance)",
}


def _box(body: ET.Element, name: str, pos_mm: tuple[float, ...], dims_mm: tuple[float, ...], group: int) -> str:
    element(
        body,
        "geom",
        name=name,
        type="box",
        pos=[v / 1000 for v in pos_mm],
        size=[v / 2000 for v in dims_mm],
        group=group,
        contype=0,
        conaffinity=0,
        rgba=(0.10, 0.42, 0.50, 0.85) if group == 4 else (0.15, 0.70, 0.90, 0.16),
    )
    return name


def apply_hardware(root: ET.Element) -> dict[str, Any]:
    """Replace schematic proxies; return auditable geometry and mass metadata.

    This function deliberately leaves explicit body inertias to the final builder.
    ``mass_components`` provides new positions and box allowances for that ledger.
    Every selected installation envelope is included in the clearance manifest.
    """
    bodies = {body.get("name"): body for body in root.iter("body")}
    torso, head = bodies["torso"], bodies["head_pitch_body"]
    motor_cases = []
    for owner, body in bodies.items():
        for geom in body.findall("geom"):
            if geom.get("name", "").endswith("_servo_case"):
                dimensions = [float(value) * 2000 for value in geom.get("size", "").split()]
                if geom.get("type") != "box" or sorted(round(v, 6) for v in dimensions) != [20, 26, 34]:
                    raise ValueError(f"Expected a full XL330 case for {geom.get('name')}")
                motor_cases.append(
                    dict(
                        geom=geom.get("name"),
                        owner_body=owner,
                        dimensions_local_xyz_mm=dimensions,
                        source=SOURCES["servo"],
                        case_mass_kg=0.018,
                        basis="Manufacturer case size retained unchanged; mounting interfaces not qualified.",
                    )
                )
    obsolete = {
        "computer_part_proxy",
        "u2d2_part_proxy",
        "motor_regulator_part_proxy",
        "logic_regulator_part_proxy",
        "schematic_battery_proxy",
        "battery_sled",
        "battery_sled_rail_-1",
        "battery_sled_rail_1",
    }
    for body in bodies.values():
        for geom in list(body.findall("geom")):
            if geom.get("name") in obsolete or geom.get("name", "").startswith("juv_hw_"):
                body.remove(geom)

    hardware: list[dict[str, Any]] = []
    masses: list[dict[str, Any]] = []

    def add(
        name: str,
        owner: str,
        pos: tuple[float, ...],
        raw: tuple[float, ...],
        installed: tuple[float, ...],
        mass_g: float,
        source: str,
        basis: str,
        old_pos: tuple[float, ...] | None = None,
        *,
        added_mass_g: float = 0,
    ) -> None:
        if any(a > b for a, b in zip(raw, installed, strict=True)):
            raise ValueError(f"Raw body does not fit its declared installation box: {name}")
        body_geom = _box(bodies[owner], f"juv_hw_{name}_body", pos, raw, 4)
        envelope_geom = _box(bodies[owner], f"juv_hw_{name}_installation", pos, installed, 5)
        row = dict(
            name=name,
            owner_body=owner,
            body_geom=body_geom,
            envelope_geom=envelope_geom,
            source=source,
            basis=basis,
            pos_m=[v / 1000 for v in pos],
            raw_dimensions_xyz_mm=list(raw),
            installation_dimensions_xyz_mm=list(installed),
            installation_min_xyz_m=[(p - d / 2) / 1000 for p, d in zip(pos, installed, strict=True)],
            installation_max_xyz_m=[(p + d / 2) / 1000 for p, d in zip(pos, installed, strict=True)],
            mass_kg=mass_g / 1000,
            added_mass_kg=added_mass_g / 1000,
            mass_basis="allocation; not measured installed mass",
        )
        hardware.append(row)
        masses.append(
            dict(
                name=name,
                owner_body=owner,
                mass_kg=mass_g / 1000,
                pos_m=row["pos_m"],
                inertia_box_halfsize_m=[v / 2000 for v in installed],
                previous_pos_m=None if old_pos is None else [v / 1000 for v in old_pos],
                added_mass_kg=added_mass_g / 1000,
                basis=row["mass_basis"],
                inertia_basis="Uniform installation-box allowance, including allocated mounting/plug space; not a measured inertia.",
            )
        )

    add(
        "computer",
        "torso",
        (-94, 0, 35),
        (8, 65, 30),
        (22, 85, 50),
        35,
        SOURCES["computer"],
        "65x30 mm manufacturer PCB footprint; 8 mm raw assembly thickness assumed. "
        "22x85x50 mm installed box includes provisional cooling, plugs and standoffs; "
        "connector-specific CAD and thermal test required.",
        (-45.5, 0, 30),
    )
    add(
        "battery",
        "torso",
        (89, 0, 5),
        (35, 70, 22),
        (43, 80, 30),
        70,
        SOURCES["allowance"],
        "No battery SKU selected. 70x35x22 mm maximum pack requirement, installed flat "
        "across the low front chest to counterbalance rear electronics and tail and clear the moving jaw; "
        "80x43x30 mm installed allowance includes restraint, insulation and lead exit. "
        "Removable through lower service panel after disconnecting power.",
        (-15, 0, -5),
    )
    add(
        "u2d2",
        "torso",
        (-50, 40.5, 26),
        (48, 14.9, 18),
        (60, 20, 25),
        9,
        SOURCES["u2d2"],
        "Manufacturer 48x18x14.9 mm body, rotated into left rear bay; 60x25x20 mm installed "
        "allowance for plugs. Check USB-C vs older Micro-B revision before cable purchase.",
        (-29.5, 0, 50),
    )
    add(
        "motor_regulator",
        "torso",
        (-10, 34, 49.5),
        (40.64, 20.32, 7.62),
        (50, 30, 15),
        4.8,
        SOURCES["motor_regulator"],
        "D24V90F5 manufacturer board dimensions without optional terminals; "
        "50x30x15 mm installed allowance, soldered low-profile leads. Rated current is "
        "input/thermal dependent; enclosure operation is not qualified.",
        (-20, 0, 10),
    )
    add(
        "logic_regulator",
        "torso",
        (-10, -31.5, 58.75),
        (17.78, 17.78, 7.874),
        (25, 25, 12),
        2.3,
        SOURCES["logic_regulator"],
        "D24V22F5 manufacturer dimensions without headers; 25x25x12 mm installation allowance.",
        (-20, 30, 6),
    )
    add(
        "power_distribution",
        "torso",
        (-47, -35, 46),
        (30, 20, 10),
        (35, 25, 15),
        28,
        SOURCES["allowance"],
        "Unselected fuse, cutoff, split motor/logic power and distribution assembly. "
        "Raw and installed boxes are procurement requirements, not vendor dimensions.",
    )
    add(
        "body_imu",
        "torso",
        (-10, -34, 46.25),
        (25.4, 17.78, 4.5),
        (30, 25, 8),
        3,
        SOURCES["imu"],
        "ISM330DHCX breakout 25.4x17.78 mm board outline; 4.5 mm assembly thickness assumed. "
        "30x25x8 mm installed reserve with low-profile STEMMA leads; no upright headers.",
        (-10, 0, 25),
    )
    add(
        "character_electronics",
        "torso",
        (-47, -35, 25),
        (20, 15, 10),
        (25, 20, 15),
        10,
        SOURCES["allowance"],
        "Unselected small LED/audio/button assembly; its dimensions and 10 g mass are allowances.",
    )
    add(
        "harness_trunk",
        "torso",
        (-49, -35, 5),
        (32, 20, 6),
        (39, 25, 10),
        35,
        SOURCES["allowance"],
        "35 g whole-robot harness allocation represented at rear trunk COM; box is a "
        "reserved gathering corridor, not a solid block or complete cable route. "
        "Branch routes, plug retention and moving-joint service loops remain provisional.",
    )
    add(
        "head_imu",
        "head_pitch_body",
        (50, 0, 19),
        (25.4, 17.78, 4.5),
        (30, 25, 8),
        3,
        SOURCES["imu"],
        "Same breakout and installation assumptions as body IMU; moved forward to clear jaw servo.",
        (27, 0, 18),
    )
    add(
        "camera",
        "head_pitch_body",
        (92, 0, 27),
        (9, 25, 24),
        (17, 30, 29),
        8,
        SOURCES["camera"],
        "Camera Module 2 mechanical reference: manufacturer 25x24x9 mm, 3 g bare. "
        "17x30x29 mm installed reserve and 8 g installed mass are allowances. "
        "Radxa documents Camera V2 support; selected cable routing, overlay operation in this "
        "robot, and the optical shell opening are unqualified. This is a physical camera "
        "reservation, not an enabled vision pipeline.",
        added_mass_g=8,
    )

    torso.find("site[@name='body_imu']").set("pos", values((-0.010, -0.034, 0.04625)))
    head.find("site[@name='head_imu']").set("pos", values((0.050, 0, 0.019)))
    camera = head.find("camera[@name='onboard']")
    if camera is not None:
        camera.set("pos", values((0.099, 0, 0.027)))

    # Service corridors are swept removal-space proposals with the robot powered
    # off. They intentionally meet their own installation boxes. They are not
    # counted as simultaneously installed parts in the automatic packing screen.
    service = []
    for name, pos, dims, direction, owner in (
        ("computer_removal", (-94, 0, 88), (22, 85, 56), (0, 0, 1), "computer"),
        ("battery_removal", (89, 0, -27), (43, 80, 34), (0, 0, -1), "battery"),
    ):
        geom = _box(torso, f"juv_hw_{name}", pos, dims, 5)
        service.append(
            dict(
                name=name,
                geom=geom,
                owner_body="torso",
                component=owner,
                pos_m=[v / 1000 for v in pos],
                dimensions_xyz_mm=list(dims),
                removal_direction_body=list(direction),
                basis="planned extraction corridor, not measured",
                conditions="Power disconnected, removable shell panel removed, retaining screws released; "
                "verify tools and plug disconnection in fabrication CAD.",
            )
        )

    bounds = {}
    for owner in sorted({row["owner_body"] for row in hardware}):
        rows = [row for row in hardware if row["owner_body"] == owner]
        bounds[owner] = dict(
            minimum_m=[min(row["installation_min_xyz_m"][i] for row in rows) for i in range(3)],
            maximum_m=[max(row["installation_max_xyz_m"][i] for row in rows) for i in range(3)],
            meaning="AABB union of installed electronics only; include motors and shell wall separately.",
        )

    return dict(
        hardware=hardware,
        mass_components=masses,
        packing_bounds_by_body=bounds,
        retained_motor_cases=motor_cases,
        torso_aabb=bounds["torso"],
        clearance_m=0.002,
        service_corridors=service,
        hardware_sources=SOURCES,
        measured_installed_fit=False,
        imu_outline_evidence=dict(
            source=SOURCES["imu_board_outline"],
            blob_sha="fd8f0e4dcc586a37a54d5cae17a7e2384d22bba5",
            dimensions_xy_mm=[25.4, 17.78],
            basis="Eagle board plain/dimension layer extrema; Adafruit guide states ISM330DHCX uses the same fab print. Thickness is an allowance.",
        ),
        hardware_group_meanings={"4": "raw part approximations", "5": "installation and service allowances"},
        wiring_plan=dict(
            topology="2S battery -> fuse/cutoff -> independent regulated 5 V motor and logic branches, common ground; U2D2 does not power motors.",
            torso_trunk_geom="juv_hw_harness_trunk_installation",
            branch_service_loops="Provide strain relief and free loops at hip, neck yaw/pitch and jaw; connector and bend-radius CAD pending.",
            camera_ribbon=dict(
                computer_connector="22-pin 0.5 mm CSI; use the correct Camera V2 adapter ribbon",
                supported_camera_source=SOURCES["camera_radxa_support"],
                straight_endpoint_distance_home_m=0.202,
                length_design_allowance_m=0.300,
                basis="Approximately 202 mm straight endpoint distance before routing; 300 mm is a procurement/routing allowance, not a qualified bend-radius or signal-integrity result. A 150 mm stock ribbon is too short.",
            ),
            electrical_validation="Battery/distribution SKUs, wire gauge, fuse rating, regeneration, voltage sag and thermal limits remain unqualified.",
        ),
        hardware_limitations=[
            "Installation boxes are engineering allowances and require selected connector/cable CAD and hardware measurement.",
            "Raw XL330 cases retain the full 20x34x26 mm dimensions; horns/idlers, bearings and shaft interfaces remain baseline approximations.",
            "Air space does not demonstrate passive cooling, fan airflow or full simultaneous servo-load capability.",
            "Service corridors are removal proposals; they require panel openings and independent service-pose verification.",
            "Camera mass is an additional 8 g allowance; ribbon routing, on-robot operation and a clear optical aperture are not certified.",
        ],
    )
