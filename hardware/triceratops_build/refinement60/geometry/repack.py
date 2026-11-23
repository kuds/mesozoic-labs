"""Regenerate a 60 cm open-rib foreleg repack from the baseline model directory."""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def v(s):
    return np.fromstring(s, sep=" ")


def fmt(a):
    return " ".join(f"{x:.10g}" for x in np.ravel(a))


def model(source, destination, dx=-0.020, dy=0.005, dz=0.006, visible_ulna=0.114):
    root = ET.parse(source).getroot()
    for side, sg in [("left", 1), ("right", -1)]:
        prefix = side + "_fore_"

        def locate(tag, part):
            element = root.find(f".//{tag}[@name='{prefix + part}']")
            if element is None:
                raise ValueError(f"Missing {prefix + part} in {source}")
            return element

        shank = locate("body", "lower_link")
        ankle = v(locate("body", "foot").get("pos"))
        elbow = v(shank.get("pos"))
        delta = np.array([dx, sg * dy, dz])
        lateral = np.array([0.0, sg * dy, 0.0])
        motor = v(locate("geom", "x6_flexion_stator").get("fromto")).reshape(2, 3).mean(0) + delta
        centres = elbow - motor
        centres[1] = 0
        pin = 0.028 * np.array([-centres[2], 0.0, centres[0]]) / np.linalg.norm(centres)
        axial = np.array([0.0, sg * 0.04175, 0.0])
        rod = [motor + axial + pin, elbow + axial + lateral + pin]
        for g in root.iter("geom"):
            nm = g.get("name", "")
            if prefix not in nm:
                continue
            suffix = nm.split(prefix)[1]
            if "fromto" in g.attrib:
                pts = v(g.get("fromto")).reshape(2, 3)
                if suffix.startswith("x6_flexion_stator") or suffix == "drive_crank":
                    pts += delta
                elif suffix == "output_crank":
                    pts += lateral
                elif suffix == "elbow_axle":
                    pts[np.argmax(sg * pts[:, 1])] += lateral
                elif suffix == "pushrod":
                    pts = rod
                elif suffix == "offset_motor_mount":
                    pts += lateral
                    pts[1] += [dx, 0, dz]
                elif suffix == "mount_standoff":
                    pts[1] += lateral
                else:
                    continue
                g.set("fromto", fmt(pts))
            elif "drive_crank_" in suffix:
                g.set("pos", fmt(v(g.get("pos", "0 0 0")) + delta))
            elif "output_crank_" in suffix:
                g.set("pos", fmt(v(g.get("pos", "0 0 0")) + lateral))
        if source.name == "triceratops.xml":
            # Compress the two cosmetic bony forms along their own longitudinal axes.
            # The rigid shank, ankle pivot and sole retain all baseline dimensions.
            u = ankle / np.linalg.norm(ankle)
            for part in ("distal", "secondary"):
                dest = destination.parent / ("study_" + prefix + part + ".obj")
                lines = []
                for line in (source.parent / dest.name).read_text().splitlines():
                    if line.startswith("v "):
                        pt = v(line[2:])
                        t = pt @ u
                        pt += u * t * (visible_ulna / np.linalg.norm(ankle) - 1)
                        line = "v " + fmt(pt)
                    lines.append(line)
                dest.write_text("\n".join(lines) + "\n")
            # A tapered, non-articulated distal cuff visually joins forearm to pastern.
            n = np.array([-u[2], 0, u[0]])
            rings = [
                (104, 14, 13, 10),
                (114, 19, 16, 9),
                (123, 15, 15, 7),
                (144, 12, 12, 3),
                (164, 13, 15, 2),
                (178, 17, 18, 2),
            ]
            vertices = []
            faces = []
            nv = 24
            for length, ry, rn, offset in rings:
                for i in range(nv):
                    theta = 2 * np.pi * i / nv
                    vertices.append(
                        (length * u + n * rn * np.sin(theta) + np.array([0.0, sg * offset + ry * np.cos(theta), 0.0]))
                        / 1000
                    )
            for k in range(len(rings) - 1):
                for i in range(nv):
                    a = k * nv + i
                    b = k * nv + (i + 1) % nv
                    c = b + nv
                    e = a + nv
                    faces.extend([(a, b, c), (a, c, e)])
            for i in range(1, nv - 1):
                faces.extend(
                    [(0, i + 1, i), ((len(rings) - 1) * nv, (len(rings) - 1) * nv + i, (len(rings) - 1) * nv + i + 1)]
                )
            vs = np.array(vertices)
            fs = np.array(faces)
            if np.einsum("ij,ij->i", vs[fs[:, 0]], np.cross(vs[fs[:, 1]], vs[fs[:, 2]])).sum() < 0:
                fs = fs[:, ::-1]
            nm = prefix + "carpal_cuff"
            (destination.parent / (nm + ".obj")).write_text(
                "".join("v " + fmt(a) + "\n" for a in vs)
                + "".join("f " + " ".join(str(x + 1) for x in row) + "\n" for row in fs)
            )
            ET.SubElement(root.find("asset"), "mesh", name=nm, file=nm + ".obj")
            ET.SubElement(
                shank,
                "geom",
                name=nm,
                type="mesh",
                mesh=nm,
                material="study_limb",
                group="2",
                mass="0",
                contype="0",
                conaffinity="0",
            )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="unicode")


def build(source, dest):
    dest.mkdir(exist_ok=True, parents=True)
    for f in source.iterdir():
        if f.is_file() and f.suffix in (".obj", ".png"):
            shutil.copy2(f, dest / f.name)
    for name in ("physics.xml", "triceratops.xml"):
        model(source / name, dest / name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    build(args.baseline, args.dest)
