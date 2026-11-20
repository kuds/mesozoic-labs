"""Visible load paths for the medium candidate's existing structural budgets.

These simple struts, plates and shaft hubs explain how the concept connects;
they are not sized or detailed fabrication brackets. Matched inspection geoms
support direct-distance screening without introducing contacts against the
coarse head/jaw boxes that surround the neck's mounting interface.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product
from typing import Any
from xml.etree import ElementTree as ET

import mujoco

from environments.shared.robot_building import element
from environments.shared.robot_design_spec import ClearancePair, RobotDesignSpec
from environments.shared.robot_design_validation import screen_design

from . import mechanical_checks
from .build_model import Builder
from .hardware_layout import load_manifest
from .torso_shell import COLLISIONS

NECK_SUPPORTS = ("neck_output_hub", "neck_yoke_root", "neck_yoke_rail", "neck_yoke_tab")
HIP_SUPPORTS = (
    "pelvis_hip_bridge",
    "left_hip_yaw_bracket",
    "right_hip_yaw_bracket",
    "left_hip_roll_bracket",
    "right_hip_roll_bracket",
)
SUPPORT_PROXIES = tuple(name + "_collision" for name in (*NECK_SUPPORTS, *HIP_SUPPORTS))

# Each exception describes a deliberate connection, not general adjacency.
SUPPORT_ATTACHMENTS = {
    ("neck_output_hub_collision", "hardware_head_yaw_motor"): "Output hub meets the yaw motor's shaft face.",
    ("neck_output_hub_collision", "neck_yoke_root_collision"): "The yoke root joins its output hub.",
    ("neck_yoke_root_collision", "neck_yoke_rail_collision"): "The yoke root joins the offset rail.",
    ("neck_yoke_rail_collision", "neck_yoke_tab_collision"): "The rail joins the nod-motor mounting tab.",
    ("neck_yoke_tab_collision", "hardware_head_nod_motor"): "The tab meets the fixed nod-motor housing.",
    (
        "pelvis_hip_bridge_collision",
        "torso_bridge_shell_collision",
    ): "The hip plate enters the rigid upper torso structure.",
    ("pelvis_hip_bridge_collision", "left_hip_yaw_case_collision"): "The plate meets the fixed left hip-yaw housing.",
    ("pelvis_hip_bridge_collision", "right_hip_yaw_case_collision"): "The plate meets the fixed right hip-yaw housing.",
    **{
        (f"{side}_hip_yaw_bracket_collision", f"{side}_{motor}_case_collision"): reason
        for side in ("left", "right")
        for motor, reason in (
            ("hip_yaw", "The rotating yaw bracket joins the hip-yaw output face."),
            ("hip_roll", "The yaw bracket meets its carried hip-roll housing."),
        )
    },
    **{
        (f"{side}_hip_roll_bracket_collision", f"{side}_{motor}_case_collision"): reason
        for side in ("left", "right")
        for motor, reason in (
            ("hip_roll", "The rotating roll adapter joins the hip-roll output face."),
            ("hip_pitch", "The roll adapter meets its carried hip-pitch housing."),
        )
    },
}

# Those broad boxes already contain the nod-motor mounting interface. Use the
# compiled skull/jaw meshes for these support pairs instead of declaring an
# unchecked opening. MuJoCo's convex mesh hull is conservative for these parts.
SUPPORT_SURFACE_REPLACEMENTS = {
    ("neck_yoke_rail_collision", "head_collision"): "head_shell",
    ("neck_yoke_rail_collision", "jaw_collision"): "jaw_shell",
    ("neck_yoke_tab_collision", "head_collision"): "head_shell",
}
SUPPORT_MESH_PAIRS = tuple(
    (name + "_collision", surface) for name in NECK_SUPPORTS for surface in ("head_shell", "jaw_shell")
)


def add_supports(builder: Builder, layout: Any) -> None:
    """Connect existing cases using the retained neck/pelvis mass allocations.

    All hardware positions, joint axes, actuator limits and component ledger entries
    are unchanged. The negative-Y neck route clears the moving jaw and skull; a
    central strut would cross those surfaces within the declared nod range.
    """

    def support(parent: ET.Element, name: str, kind: str, size: Any, **placement: Any) -> None:
        builder.geom(parent, name, kind, size, material="metal", group=2, **placement)
        element(
            parent,
            "geom",
            name=name + "_collision",
            type=kind,
            size=size,
            group=3,
            contype=0,
            conaffinity=0,
            rgba=(0.9, 0.25, 0.1, 0.2),
            **placement,
        )

    yaw = builder.bodies["head_yaw_body"]
    reach = float(layout.nod_reach_m)
    support(yaw, "neck_output_hub", "cylinder", (0.010, 0.003), pos=(0, 0, 0.020))
    support(yaw, "neck_yoke_root", "capsule", 0.003, fromto=(0, 0, 0.023, 0, -0.035, 0.023))
    support(yaw, "neck_yoke_rail", "capsule", 0.003, fromto=(0, -0.035, 0.023, reach, -0.035, 0.028))
    support(yaw, "neck_yoke_tab", "capsule", 0.0025, fromto=(reach, -0.035, 0.028, reach, -0.017, 0.028))
    support(builder.torso, "pelvis_hip_bridge", "box", (0.025, 0.065, 0.008), pos=(0, 0, 0.111))
    for side, sign in (("left", 1), ("right", -1)):
        support(
            builder.bodies[side + "_hip_yaw_body"],
            side + "_hip_yaw_bracket",
            "capsule",
            0.004,
            fromto=(0, 0, -0.031, -0.032, sign * 0.035, -0.044),
        )
        support(
            builder.bodies[side + "_hip_roll_body"],
            side + "_hip_roll_bracket",
            "capsule",
            0.005,
            fromto=(-0.004, 0, 0, 0.030, sign * 0.005, 0),
        )


def support_spec(model: mujoco.MjModel) -> RobotDesignSpec:
    """Screen every support against declared hardware and moving upper parts."""
    manifest = load_manifest()
    targets = (
        *(part["geometry"] for part in manifest["mounted_hardware"]),
        *(part["geometry"] for part in manifest["assemblies"]),
        *COLLISIONS,
        "neck_support_collision",
        "head_collision",
        "jaw_collision",
        "left_arm_collision",
        "right_arm_collision",
        *(f"tail_segment_{index}_collision" for index in range(3)),
    )
    omitted = {frozenset(pair) for pair in SUPPORT_ATTACHMENTS}
    pairs = set()
    for first, second in (*product(SUPPORT_PROXIES, targets), *combinations(SUPPORT_PROXIES, 2)):
        if frozenset((first, second)) in omitted:
            continue
        second = SUPPORT_SURFACE_REPLACEMENTS.get((first, second), second)
        pairs.add((first, second))
    pairs.update(SUPPORT_MESH_PAIRS)
    return replace(
        mechanical_checks.design_spec(model),
        robot_id="tyrannosaurus_rex_medium_torso_supports",
        loads=(),
        clearances=tuple(ClearancePair(first, second, 0.002) for first, second in sorted(pairs)),
        assumptions=(
            "Nine planned support shapes use existing neck and pelvis structural mass allocations; strength, fasteners and fatigue are unqualified.",
            "Matched support inspection geoms disable contact response. Direct-distance checks do not depend on collision filtering.",
            "The coarse head/jaw boxes surround the nod mounting interface. Selected neck pairs instead check conservative convex hulls of the actual compiled skull/jaw meshes.",
            "The torso plate meets the bridge shell. Hidden hardware-containment ellipsoids are hollow bounds, not occupied solids or additional obstacles.",
            "Sampling is home, the 729-pose expression grid and independent five-angle leg sweeps; combined leg/expression paths and continuous clearance remain unverified.",
            *(f"Attached pair {first} / {second}: {reason}" for (first, second), reason in SUPPORT_ATTACHMENTS.items()),
        ),
    )


def check_supports(model: mujoco.MjModel) -> dict[str, Any]:
    """Preserve known geometry failures while leaving structural evidence open."""
    result = screen_design(model, support_spec(model))
    result["structural_qualification"] = {
        "status": "incomplete",
        "reason": "Visible supports are planning load paths within retained mass budgets, not sized fabrication brackets.",
    }
    if result["status"] == "passed":
        result["status"] = "incomplete"
    return result
