"""Lengthen the fore upper links while holding the shoulder, sole and X6 barrel in place."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "geometry"))
from relieve import curve  # noqa: E402
from repack import fmt, v  # noqa: E402


def link_transform(before, after):
    """Map the centreline lengths exactly, preserving transverse dimensions."""

    def basis(axis):
        u = axis / np.linalg.norm(axis)
        return np.array([u, [0, 1, 0], [-u[2], 0, u[0]]])

    return basis(after).T @ np.diag([np.linalg.norm(after) / np.linalg.norm(before), 1, 1]) @ basis(before)


def scale_mass(geom, ratio):
    geom.set("mass", f"{float(geom.get('mass')) * ratio:.9g}")


def change_leg(tree, side):
    prefix = side + "_fore_"

    def named(tag, suffix):
        return tree.find(f".//{tag}[@name='{prefix}{suffix}']")

    lower = named("body", "lower_link")
    ankle = named("body", "foot")
    old_upper = v(lower.get("pos"))
    old_lower = v(ankle.get("pos"))
    vertical = -(old_upper + old_lower)[2]
    length = 0.160
    depth = (length**2 - np.linalg.norm(old_lower) ** 2 + vertical**2) / (2 * vertical)
    upper = np.array([-np.sqrt(length**2 - depth**2), 0, -depth])
    shank = old_upper + old_lower - upper
    lower.set("pos", fmt(upper))
    ankle.set("pos", fmt(shank))

    motor = v(named("geom", "x6_flexion_stator").get("fromto")).reshape(2, 3).mean(0)
    old_rod = v(named("geom", "pushrod").get("fromto")).reshape(2, 3)
    centers = (upper - motor) * [1, 0, 1]
    pin = np.array([-centers[2], 0, centers[0]]) * (0.028 / np.linalg.norm(centers))
    rod = np.array([motor + pin, upper + pin])
    rod[:, 1] = old_rod[:, 1]
    # Keep the same motor end of the plate and put its support back onto the new strut centreline.
    stand = v(named("geom", "mount_standoff").get("fromto")).reshape(2, 3)
    stand[:, [0, 2]] = (upper * np.dot(upper, stand[0]) / length**2)[[0, 2]]
    plate = named("geom", "offset_motor_mount")
    plate_end = v(plate.get("pos")) + v(plate.get("xyaxes"))[:3] * v(plate.get("size"))[0]
    direction = plate_end - stand[1]
    for g in tree.iter("geom"):
        name = g.get("name", "").removeprefix("study_actual_")
        if not name.startswith(prefix):
            continue
        part = name.removeprefix(prefix)
        if part in ("upper_strut", "upper_limb_cover"):
            g.set("fromto", fmt([np.zeros(3), upper]))
            scale_mass(g, length / np.linalg.norm(old_upper))
        elif part in ("lower_strut", "lower_limb_cover"):
            g.set("fromto", fmt([np.zeros(3), shank]))
        elif part == "pushrod":
            g.set("fromto", fmt(rod))
            scale_mass(g, np.linalg.norm(centers) / np.linalg.norm(np.diff(old_rod, axis=0)))
        elif part == "offset_motor_mount":
            g.set("pos", fmt((plate_end + stand[1]) / 2))
            sizes = v(g.get("size"))
            scale_mass(g, np.linalg.norm(direction) / (2 * sizes[0]))
            sizes[0] = np.linalg.norm(direction) / 2
            g.set("size", fmt(sizes))
            g.set("xyaxes", fmt([direction / np.linalg.norm(direction), [0, 1, 0]]))
        elif part == "mount_standoff":
            g.set("fromto", fmt(stand))
    return link_transform(old_upper, upper), link_transform(old_lower, shank)


def reshape_panel(vertex, matrix, barrel):
    point = matrix @ vertex
    if barrel is not None:
        # Recess the back of the longer sculpt around the unchanged motor housing.
        radial = point[[0, 2]] - barrel[[0, 2]]
        radius = np.linalg.norm(radial)
        if radius < 0.047:
            point[[0, 2]] += (0.047 / radius - 1) * radial
    return point


def build(output):
    source = ROOT / "models/refined"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("physics.xml", "triceratops.xml"):
        tree = ET.parse(source / name)
        changes = {}
        for side in ("left", "right"):
            proximal, distal = change_leg(tree, side)
            motor = tree.find(f".//geom[@name='{side}_fore_x6_flexion_stator']")
            barrel = v(motor.get("fromto")).reshape(2, 3).mean(0)
            for part in ("proximal", "distal", "secondary"):
                changes[f"study_{side}_fore_{part}.obj"] = (proximal, barrel) if part == "proximal" else (distal, None)
            changes[f"{side}_fore_carpal_cuff.obj"] = distal, None
        for item in tree.findall("./asset/*[@file]"):
            original = (source / item.get("file")).resolve()
            if original.name in changes:
                target = output / original.name
                matrix, barrel = changes[original.name]
                curve(original, target, lambda vertex: reshape_panel(vertex, matrix, barrel))
            else:
                target = original
            item.set("file", Path(os.path.relpath(target, output)).as_posix())
        tree.write(output / name, encoding="unicode")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "models/mid160")
    build(p.parse_args().output)
