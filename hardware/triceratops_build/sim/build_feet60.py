"""Foot variants of the 60 cm model for the foot study (decision D5). Simulation only.

Each variant is the as-built model `generated/trike60_C_closed.xml` (revision 3, PR #669's foot: a passive pitch
hinge and a rigid 88 x 64 mm sole) with only the feet changed. Every variant keeps the names the tools use
(`<corner>_foot`, `<corner>_ankle`, `<corner>_sole`, `<corner>_load_cell`, `<corner>_toe`), the sole's mass and the
height of its lowest point under the ankle pin (36 mm), so the robot's mass (12.694 kg) and standing height are
unchanged. The models in `generated/` are not touched.

Variants (`--variant`, repeatable; default: the study's candidates):
  pitch            the as-built foot (a copy, for a uniform file set)
  roll_k<K>        a passive roll hinge added under the pitch hinge, centring spring K N m/rad (the existing
                   ankle-roll variant is roll_k2.032: the pitch hinge's own spring)
  capsule_r<R>     the sole replaced by a capsule along the foot, radius R mm, the same 88 mm length: rounded
                   across its width, so it rocks in roll without a joint; the pitch hinge stays
  ball_r<R>        the sole replaced by a sphere of radius R mm; the pitch hinge stays (a sphere's rotation does not
                   change its contact)

Usage: python build_feet60.py [--out feet60] [--variant roll_k8 --variant capsule_r16 ...]
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "generated" / "trike60_C_closed.xml"
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
SOLE_BOTTOM = -0.036  # the as-built sole's lowest point below the ankle pin, m (pos z -0.0288, half height 0.0072)
SOLE_HALF_LEN = 0.044
SITE_PAD = 0.0004  # the load cell site's margin around the contact geometry, as built (0.04416 vs 0.044)
CANDIDATES = ("pitch", "roll_k2.032", "capsule_r24", "ball_r30")  # the judged set (PREREG_FEET60.md)


def edit(variant: str) -> str:
    tree = ET.parse(BASE)
    root = tree.getroot()
    root.set("model", root.get("model") + f"_foot_{variant}")
    bodies = {b.get("name"): b for b in root.iter("body")}
    for c in CORNERS:
        foot = bodies[f"{c}_foot"]
        sole = next(g for g in foot.findall("geom") if g.get("name") == f"{c}_sole")
        cell = next(s for s in foot.findall("site") if s.get("name") == f"{c}_load_cell")
        pitch = next(j for j in foot.findall("joint") if j.get("name") == f"{c}_ankle")
        if variant == "pitch":
            continue
        m = re.fullmatch(r"roll_k([0-9.]+)", variant)
        if m:
            rj = ET.Element("joint", {k: v for k, v in pitch.attrib.items() if k not in ("name", "axis", "range")})
            rj.attrib.update({"name": f"{c}_ankle_roll", "axis": "1 0 0", "range": "-20 20",
                              "stiffness": f"{float(m.group(1)):.4f}"})
            foot.insert(list(foot).index(pitch) + 1, rj)
            continue
        m = re.fullmatch(r"(capsule|ball)_r([0-9.]+)", variant)
        if not m:
            raise SystemExit(f"unknown variant {variant}")
        kind, r = m.group(1), float(m.group(2)) / 1000.0
        zc = SOLE_BOTTOM + r
        for k in ("size", "pos", "fromto", "type"):
            sole.attrib.pop(k, None)
        if kind == "capsule":
            hx = SOLE_HALF_LEN - r  # the capsule's ends land where the sole's ends were
            sole.attrib.update({"type": "capsule", "fromto": f"{-hx:.5f} 0 {zc:.5f} {hx:.5f} 0 {zc:.5f}",
                                "size": f"{r:.5f}"})
            cell.set("size", f"{hx + r + SITE_PAD:.5f} {r + SITE_PAD:.5f} {r + SITE_PAD:.5f}")
        else:
            sole.attrib.update({"type": "sphere", "pos": f"0 0 {zc:.5f}", "size": f"{r:.5f}"})
            cell.set("size", f"{r + SITE_PAD:.5f} {r + SITE_PAD:.5f} {r + SITE_PAD:.5f}")
        cell.set("pos", f"0 0 {zc:.5f}")
        # keep the attribute order readable: type first
        ordered = {k: sole.attrib[k] for k in ("name", "type", "fromto", "pos", "size") if k in sole.attrib}
        ordered.update({k: v for k, v in sole.attrib.items() if k not in ordered})
        sole.attrib.clear()
        sole.attrib.update(ordered)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "feet60"))
    ap.add_argument("--variant", action="append", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import mujoco

    for v in a.variant or CANDIDATES:
        path = out / f"trike60_foot_{v}.xml"
        path.write_text(edit(v))
        m = mujoco.MjModel.from_xml_path(str(path))
        print(f"{path.name}: {m.body_mass.sum():.4f} kg, {m.njnt} joints")


if __name__ == "__main__":
    main()
