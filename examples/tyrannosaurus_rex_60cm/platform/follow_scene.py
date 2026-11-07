"""Ground-truth side of the indoor follow scenario. No tracking code lives here."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def _part(body, name, shape, position, size, color):
    def values(items):
        return " ".join(map(str, items))

    ET.SubElement(body, "geom", name=name, type=shape, pos=values(position), size=values(size),
                  rgba=color, group="0", contype="1", conaffinity="1")


def _person(world, name, scale, shirt, trousers, skin):
    """Life-size, unmarked geometric figures with different builds and clothes."""
    body = ET.SubElement(world, "body", name=name, mocap="true")
    def part(label, shape, pos, dims, color):
        _part(body, name + "_" + label, shape, np.array(pos) * scale, np.array(dims) * scale, color)
    part("shirt", "ellipsoid", (0, 0, 1.17), (.145, .215, .31), shirt)
    part("hips", "ellipsoid", (0, 0, .91), (.135,.165,.15), trousers)
    part("neck", "capsule", (0,0,1.465), (.058,.055), skin)
    part("head", "ellipsoid", (0,0,1.61), (.10,.085,.125), skin)
    part("hair", "ellipsoid", (-.022,0,1.682), (.088,.085,.066), "0.19 0.13 0.11 1")
    for side, sign in (("l", 1), ("r", -1)):
        y = .095 * sign
        part(side+"thigh", "capsule", (0,y,.68), (.083,.185), trousers)
        part(side+"shin", "capsule", (.014*sign,y,.29), (.059,.17), trousers)
        part(side+"shoe", "ellipsoid", (.050+.014*sign,y,.068), (.133,.065,.064), "0.14 0.14 0.15 1")
        part(side+"sleeve", "capsule", (0,sign*.20,1.26), (.067,.087), shirt)
        part(side+"forearm", "capsule", (.006,sign*.224,1.045), (.046,.098), skin)
        part(side+"hand", "ellipsoid", (.012,sign*.222,.915), (.05,.043,.065), skin)


def build_scene(robot_xml: Path, destination: Path, scenario: str) -> None:
    root = ET.parse(robot_xml)
    world = root.getroot().find("worldbody")
    if world is None:
        raise ValueError("missing worldbody")
    floor = world.find("geom[@name='floor']")
    if floor is None or floor.get("type") != "plane":
        raise ValueError("gallery requires a plane named floor")
    floor.set("size", "15 6 0.02")
    _person(world, "visitor_a", 1.03, ".16 .28 .58 1", ".14 .17 .22 1", ".60 .38 .28 1")
    _person(world, "visitor_b", .95, ".68 .34 .13 1", ".25 .27 .31 1", ".84 .65 .51 1")
    for sign in (1, -1):
        _part(world, f"gallery_wall_{sign}", "box", (4, 3.2*sign, 1.45), (7,.08,1.45), ".72 .75 .73 1")
    _part(world, "far_wall", "box", (10,0,1.45), (.08,3.2,1.45), ".70 .71 .73 1")
    if scenario in ("obstacle", "blocked"):
        # Low pedestal on the intended route, or a barrier across the corridor.
        if scenario == "obstacle":
            _part(world, "gallery_fixture", "box", (1.48,-.19,.145), (.15,.22,.145), ".46 .40 .33 1")
        else:
            barrier = ET.SubElement(world, "body", name="barrier", mocap="true")
            _part(barrier, "gallery_barrier", "box", (0,0,.36), (.075,2.9,.36), ".46 .40 .33 1")
    if scenario == "occlusion":
        _part(world, "gallery_screen", "box", (1.9,-.32,.99), (.085,.225,.99), ".56 .55 .52 1")
    root.write(destination, encoding="utf-8", xml_declaration=True)


def people_positions(t: float, scenario: str):
    # The visitors' trajectories depend only on the scenario clock.
    if scenario=="crossing":
        return (3.12+.037*t, .65-.11*t), (2.62+.033*t, -.72+.11*t)
    if scenario in ("occlusion", "loss"):
        return (2.85+.025*t, .66-.22*t), (3.4, -1.65+.015*t)
    return (2.42+.064*t, .28+.018*t), (3.35+.012*t, -1.05)


def move_people(model, data, t: float, scenario: str):
    for name, (x,y) in zip(("visitor_a","visitor_b"), people_positions(t,scenario), strict=True):
        mid = model.body_mocapid[model.body(name).id]
        data.mocap_pos[mid] = (x,y,0)
        yaw = math.atan2(-.11 if scenario=="crossing" and name=="visitor_a" else .018, .064)
        data.mocap_quat[mid] = (math.cos(yaw/2),0,0,math.sin(yaw/2))
    if scenario == "blocked":
        barrier_id = model.body_mocapid[model.body("barrier").id]
        data.mocap_pos[barrier_id] = (1.20, max(0.0, 4.0 - 2.0 * max(0.0, t - 2.0)), 0.)
    mujoco.mj_forward(model,data)
