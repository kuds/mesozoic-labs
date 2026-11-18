"""Packaging check for the 60 cm model (a design check, not a judged simulation run).

1. Chassis bay: every pair of solid parts on the chassis body, except frame-to-frame pairs, (boxes, cylinders, capsules; not the shell ribs, the
   harness looms or the collision proxies) is tested for intersection: box pairs by their exact gap in the chassis
   frame (all are axis-aligned), other pairs by mj_geomDistance; overlapping pairs are reported, else the closest.
   (mj_geomDistance returns 0 for separated box pairs in mujoco 3.10.0, so it is not used for those.)
2. Leg drives: the minimum distance between each flexion drive's housing and its own leg's X4 pitch and roll housings,
   the body contact proxy and the other legs' housings, over a grid of the leg's roll, pitch and flexion joint ranges
   (7 x 7 x 7 per leg), with the other joints at neutral.

Usage: python check_layout60.py [--model generated/trike60_C_closed.xml]   Prints a table; writes nothing.
"""
import argparse
import itertools
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
SKIP = ("PA11_rib", "shell_seam", "loom", "_collision", "_contact", "oak_d_usb_c_lock_cable")
# frame members and the drives bolted to them intersect by design (joints and mounts): pairs of these are skipped
MOUNTED = ("6061_", "crossmember", "service_tray", "tray_stay", "neck_cradle", "_stator")


def dist(m, d, g1, g2, cap=0.2):
    return mujoco.mj_geomDistance(m, d, g1, g2, cap, None)


def body_aabb(m, g):
    """The geom's bounding box in its body's frame: exact for an axis-aligned box, conservative otherwise."""
    c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    R = R.reshape(3, 3)
    corners = np.array([c + h * np.array(s) for s in itertools.product((-1, 1), repeat=3)]) @ R.T + m.geom_pos[g]
    return corners.min(0), corners.max(0)


def box_gap(a, b):
    """Signed gap between two boxes: positive = separation distance, negative = depth of the shallowest overlap."""
    sep = np.maximum(a[0] - b[1], b[0] - a[1])
    return float(np.linalg.norm(np.clip(sep, 0, None))) if (sep > 0).any() else float(sep.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "generated" / "trike60_C_closed.xml"))
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ch = m.body("chassis").id
    parts = [g for g in range(m.ngeom) if m.geom_bodyid[g] == ch and not any(k in m.geom(g).name for k in SKIP)]
    pairs = []
    for g1, g2 in itertools.combinations(parts, 2):
        if all(any(k in m.geom(g).name for k in MOUNTED) for g in (g1, g2)):
            continue
        both_boxes = all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in (g1, g2))
        gap = box_gap(body_aabb(m, g1), body_aabb(m, g2)) if both_boxes else dist(m, d, g1, g2)
        pairs.append((gap, m.geom(g1).name, m.geom(g2).name))
    pairs.sort()
    bad = [p for p in pairs if p[0] < -1e-4]   # within 0.1 mm counts as resting contact (the battery on its tray)
    print(f"1. Chassis bay: {len(parts)} parts, {len(pairs)} pairs (frame-to-frame skipped); intersecting: {len(bad)}")
    for dd, n1, n2 in (bad or pairs[:8]):
        print(f"   {1000 * dd:+7.1f} mm  {n1} / {n2}")

    print("2. Leg drives: minimum distance over the roll x pitch x flexion grid (other joints neutral)")
    body_proxy = m.geom("torso_contact").id
    for c in CORNERS:
        flex = next(g for g in range(m.ngeom) if m.geom(g).name.startswith(c) and m.geom(g).name.endswith(
            "flexion_stator_collision"))
        others = {"own pitch": m.geom(f"{c}_x4_pitch_stator_collision").id,
                  "own roll": m.geom(f"{c}_x4_roll_stator_collision").id, "body": body_proxy}
        joints = [m.joint(f"{c}_{ax}") for ax in ("roll", "pitch", "flexion")]
        grids = [np.linspace(*j.range, 7) for j in joints]   # compiled ranges are in radians
        worst = {k: (1.0, None) for k in others}
        zero_unconfirmed = {}
        neutral = {k: dist(m, d, flex, g) for k, g in others.items()}
        for q in itertools.product(*grids):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            for j, v in zip(joints, q):
                d.qpos[m.jnt_qposadr[j.id]] = v
            mujoco.mj_forward(m, d)
            for k, g in others.items():
                x = dist(m, d, flex, g)
                if x == 0.0:   # the convex-pair query sometimes returns exactly 0; confirm with the contact list
                    hits = [d.contact[i].dist for i in range(d.ncon) if {d.contact[i].geom1, d.contact[i].geom2} == {flex, g}]
                    if not hits:
                        zero_unconfirmed[k] = zero_unconfirmed.get(k, 0) + 1
                        continue
                    x = min(hits)
                if x < worst[k][0]:
                    worst[k] = (x, np.degrees(q).round(0).tolist())
        print(f"   {c}: " + "; ".join(f"{k} {1000 * neutral[k]:.1f} mm at neutral, min {1000 * w[0]:.1f} mm at "
                                       f"roll/pitch/flex {w[1]}" for k, w in worst.items())
              + (f" (zero readings with no contact, skipped: {zero_unconfirmed})" if zero_unconfirmed else ""))
    mujoco.mj_resetDataKeyframe(m, d, 0)


if __name__ == "__main__":
    main()
