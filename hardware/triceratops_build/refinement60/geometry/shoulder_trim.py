"""Trim the inherited X4 shoulder skin without changing motor, bracket or mass."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from relieve import curve
from repack import v


def trim(src, out):
    xml = ET.parse(src / "triceratops.xml")
    for side, tag in [("left", "L"), ("right", "R")]:
        p = src / f"study_{side}_fore_proximal.obj"
        el = v(xml.find(f".//body[@name='{side}_fore_lower_link']").get("pos"))
        length = np.linalg.norm(el)
        u = el / length
        # Taper the upper cap back from the pitched X4 cylinder so its face remains inspectable.
        normal = np.array([-u[2], 0, u[0]])
        curve(p, out / p.name, lambda r: r + u * 0.038 * (1 - np.clip((r @ u) / length, 0, 1)) + normal * 0.007)
        p = src / f"study_{tag}_coracoid.obj"
        # The coracoid joins the raised/advanced scapula. Its smooth end stays above the shoulder motor.
        curve(p, out / p.name, lambda r: r + [0.035, 0, 0.045])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    trim(a.source, a.dest)
