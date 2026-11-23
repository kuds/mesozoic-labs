"""Fit a 3 mm backing plate and relieve the 60 cm open-rib shoulder window."""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from repack import fmt, v


def physical(source, dest):
    tree = ET.parse(source)
    for g in tree.getroot().iter("geom"):
        nm = g.get("name", "")
        if nm.endswith("_fore_offset_motor_mount"):
            sg = 1 if "left" in nm else -1
            ends = v(g.attrib.pop("fromto")).reshape(2, 3)
            ends[:, 1] = sg * (-0.03025)
            direction = ends[1] - ends[0]
            length = np.linalg.norm(direction)
            # Box local x along strap length, its y axis lateral.
            g.attrib.update(
                type="box",
                size=fmt([length / 2, 0.0015, 0.006]),
                pos=fmt(ends.mean(axis=0)),
                xyaxes=fmt([direction / length, [0, 1, 0]]),
            )
        elif nm.endswith("_fore_mount_standoff"):
            sg = 1 if "left" in nm else -1
            ends = v(g.get("fromto")).reshape(2, 3)
            ends[1, 1] = sg * (-0.03025)
            g.set("fromto", fmt(ends))
        elif nm.startswith("PA11_rib_4_"):
            pts = v(g.get("fromto")).reshape(2, 3)
            pts[:, 1] *= 0.68
            g.set("fromto", fmt(pts))
        elif nm in ("left_shell_seam", "right_shell_seam"):
            pts = v(g.get("fromto")).reshape(2, 3)
            pts[1, 0] = 0.099
            g.set("fromto", fmt(pts))
    tree.write(dest, encoding="unicode")


def curve(path, dest, fn):
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            line = "v " + fmt(fn(v(line[2:])))
        lines.append(line)
    dest.write_text("\n".join(lines) + "\n")


def smooth(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def skin(src, out):
    # Sweep the lower ends of the anterior ribs caudally. Draw the ipsilateral
    # scapular plate forward about its upper anchor, leaving a recessed shoulder pocket.
    for side in ("L", "R"):
        p = src / f"study_{side}_rib_5.obj"
        curve(p, out / p.name, lambda r: r + [-0.068 * smooth((0.105 - r[2]) / 0.10), 0, 0])
        p = src / f"study_{side}_rib_4.obj"
        curve(p, out / p.name, lambda r: r + [-0.017 * smooth((0.105 - r[2]) / 0.10), 0, 0])
        p = src / f"study_{side}_scapula.obj"
        curve(
            p,
            out / p.name,
            lambda r: r + [0.043 * smooth((0.145 - r[2]) / 0.070), 0, 0.018 * smooth((0.135 - r[2]) / 0.09)],
        )
    p = src / "study_dorsal_saddle.obj"

    # Lift the saddle's lateral edge into a wheel-arch-shaped shoulder opening.
    def arch(r):
        x, y, z = r
        w = smooth((x - 0.04) / 0.070) * smooth((0.30 - x) / 0.050) * smooth((abs(y) - 0.035) / 0.045)
        roof = 0.103
        return r + [0, 0, w * max(0, roof - z) * 0.91]

    curve(p, out / p.name, arch)


def build(src, out):
    out.mkdir(exist_ok=True, parents=True)
    for p in src.iterdir():
        if p.suffix in (".obj", ".png"):
            shutil.copy2(p, out / p.name)
    for n in ("physics.xml", "triceratops.xml"):
        physical(src / n, out / n)
    skin(src, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    build(a.source, a.dest)
