"""Concept-inspired metal/olive enclosures around the unchanged Rev B mechanism.

The panels, fasteners and inspection-window details use the existing core/head
mass allowances. They are enclosure proxies, not fabrication-ready part CAD.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def text(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def geom(body, name, kind, material, **attributes):
    return ET.SubElement(
        body, "geom", name=name, type=kind, material=material, contype="0", conaffinity="0", **attributes
    )


def section_mesh(asset, name, sections):
    """Closed chamfered hull from (x, half-width, bottom, top, bevel) sections."""
    vertices = []
    for x, w, low, high, b in sections:
        vertices.extend(
            [
                [x, y, z]
                for y, z in (
                    (-w + b, low),
                    (w - b, low),
                    (w, low + b),
                    (w, high - b),
                    (w - b, high),
                    (-w + b, high),
                    (-w, high - b),
                    (-w, low + b),
                )
            ]
        )
    faces = []
    for section in range(len(sections) - 1):
        base = section * 8
        for i in range(8):
            a, b = base + i, base + (i + 1) % 8
            faces.extend([[a, b, b + 8], [a, b + 8, a + 8]])
    end = (len(sections) - 1) * 8
    for i in range(1, 7):
        faces.extend([[0, i + 1, i], [end, end + i, end + i + 1]])
    ET.SubElement(
        asset, "mesh", name=name, vertex=text(vertices), face=" ".join(str(v) for face in faces for v in face)
    )


def robot_casings(root, parameters):
    asset = root.find("asset")
    for name, rgba, specular in (
        ("shell_alloy_mat", ".50 .53 .51 1", ".28"),
        ("shell_olive_mat", ".34 .39 .28 1", ".12"),
        ("shell_dark_mat", ".065 .08 .085 1", ".18"),
        ("circuit_mat", ".12 .27 .20 1", ".08"),
        ("fastener_mat", ".28 .30 .30 1", ".35"),
        ("wire_mat", ".47 .12 .08 1", ".1"),
    ):
        ET.SubElement(asset, "material", name=name, rgba=rgba, specular=specular, shininess=".3")
    pelvis = root.find("worldbody/body[@name='pelvis']")
    core = pelvis.find("body[@name='core']")
    head = pelvis.find("body[@name='fixed_head']")
    section_mesh(
        asset,
        "core_casing_mesh",
        [
            (-0.060, 0.047, -0.026, 0.026, 0.009),
            (-0.043, 0.065, -0.040, 0.040, 0.011),
            (0.025, 0.065, -0.034, 0.038, 0.011),
            (0.060, 0.034, -0.013, 0.014, 0.007),
        ],
    )
    envelope = core.find("geom[@name='core_envelope']")
    envelope.attrib.pop("size")
    envelope.attrib.update(type="mesh", mesh="core_casing_mesh", material="shell_alloy_mat")
    geom(core, "dorsal_service_panel", "box", "shell_olive_mat", pos="-.01 0 .039", size=".028 .048 .0015")
    for side in (-1, 1):
        # A recessed electronics-window proxy, with protected interior details.
        geom(
            core,
            f"service_window_frame_{side}",
            "box",
            "shell_dark_mat",
            pos=text([-0.008, side * 0.0654, 0]),
            size=".022 .0008 .014",
        )
        geom(
            core,
            f"service_window_board_{side}",
            "box",
            "circuit_mat",
            pos=text([-0.008, side * 0.0663, 0.002]),
            size=".018 .0004 .009",
        )
        for n, x in enumerate((-0.020, -0.007, 0.006)):
            geom(
                core,
                f"board_chip_{side}_{n}",
                "box",
                "shell_dark_mat",
                pos=text([x, side * 0.0668, 0.003]),
                size=".003 .0003 .003",
            )
        geom(
            core,
            f"power_module_face_{side}",
            "box",
            "shell_dark_mat",
            pos=text([-0.008, side * 0.0667, -0.008]),
            size=".016 .0005 .003",
        )
        geom(
            core,
            f"window_wire_{side}",
            "capsule",
            "wire_mat",
            fromto=text([-0.025, side * 0.067, 0.012, 0.008, side * 0.067, 0.012]),
            size=".00065",
        )
        geom(
            core,
            f"lower_access_panel_{side}",
            "box",
            "shell_olive_mat",
            pos=text([-0.016, side * 0.062, -0.025]),
            size=".021 .001 .005",
        )
        for x in (-0.037, 0.020):
            for z in (-0.020, 0.023):
                geom(
                    core,
                    f"core_screw_{side}_{x}_{z}",
                    "cylinder",
                    "fastener_mat",
                    pos=text([x, side * 0.0657, z]),
                    quat=".707106781187 .707106781187 0 0",
                    size=".0018 .0007",
                )
        for n in range(4):
            geom(
                core,
                f"core_vent_{side}_{n}",
                "box",
                "shell_dark_mat",
                pos=text([-0.027 + n * 0.008, side * 0.0656, 0.024]),
                size=".002 .0007 .004",
            )
    neck = core.find("geom[@name='neck_mount_visual']")
    neck.attrib.update(type="cylinder", material="shell_dark_mat", size=".007")
    geom(
        core,
        "fixed_neck_coupling",
        "cylinder",
        "fastener_mat",
        pos=".075 0 .035",
        quat=".707106781187 .707106781187 0 0",
        size=".009 .009",
    )

    # Preserve the prior component COM/inertia as explicitly labelled hidden
    # mass proxies. Replace their collision silhouettes with one outer housing.
    for part in parameters["robot"]["head_parts"]:
        proxy = head.find(f"geom[@name='{part['name']}']")
        proxy.attrib.update(group="3", contype="0", conaffinity="0")
    section_mesh(
        asset,
        "head_camera_casing_mesh",
        [
            (-0.035, 0.018, -0.018, 0.025, 0.006),
            (-0.013, 0.020, -0.017, 0.025, 0.007),
            (0.038, 0.010, -0.003, 0.018, 0.004),
        ],
    )
    casing = geom(head, "head_camera_casing", "mesh", "shell_alloy_mat", mesh="head_camera_casing_mesh")
    casing.attrib.update(contype="17", conaffinity="14")
    section_mesh(
        asset,
        "head_top_panel_mesh",
        [
            (-0.029, 0.010, 0.024, 0.025, 0.0004),
            (-0.014, 0.011, 0.024, 0.025, 0.0004),
            (0.012, 0.007, 0.0203, 0.0213, 0.0004),
        ],
    )
    geom(head, "head_top_panel", "mesh", "shell_olive_mat", mesh="head_top_panel_mesh")
    geom(
        head,
        "camera_bezel",
        "cylinder",
        "shell_dark_mat",
        pos=".039 0 .008",
        quat=".707106781187 0 .707106781187 0",
        size=".0078 .0018",
    )
    for side in (-1, 1):
        for n, x in enumerate((-0.025, -0.013)):
            geom(
                head,
                f"head_fastener_{side}_{n}",
                "cylinder",
                "fastener_mat",
                pos=text([x, side * 0.0198, 0.002]),
                quat=".707106781187 .707106781187 0 0",
                size=".0012 .0006",
            )
        geom(
            head,
            f"head_rear_panel_{side}",
            "box",
            "shell_olive_mat",
            pos=text([-0.030, side * 0.0186, 0.004]),
            size=".003 .0006 .008",
        )
    # The same passive taper gets a dark tube finish; collars are rigid styling
    # within its 50 g allocation, not new tail joints or actuators.
    tail = pelvis.find("body[@name='passive_tail']")
    tail.find("geom[@name='passive_tail_envelope']").set("material", "shell_dark_mat")
    for n, x in enumerate((0.095, 0.020, -0.055)):
        radius = 0.0015 + (x + 0.12) / 0.255 * 0.0085
        geom(
            tail,
            f"tail_collar_{n}",
            "cylinder",
            "shell_alloy_mat",
            pos=text([x, 0, 0]),
            quat=".707106781187 0 .707106781187 0",
            size=text([radius + 0.0005, 0.003]),
        )
