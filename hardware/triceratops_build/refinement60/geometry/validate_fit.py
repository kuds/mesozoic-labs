"""Bilateral sampled packaging and exact neutral linkage measurements; units in millimetres."""

import argparse
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np


def vec(s):
    return np.fromstring(s, sep=" ")


def check(path):
    m = mj.MjModel.from_xml_path(str(path))
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    mj.mj_forward(m, d)
    c = m.body("chassis").id
    result = {
        "mass_kg": m.body_mass.sum(),
        "neutral_com_chassis_mm": ((d.subtree_com[c] - d.xpos[c]) * 1000).tolist(),
        "samples_per_side": 17 * 17 * 17,
        "sides": {},
    }
    xml = ET.parse(path)

    def node(name):
        return xml.find(f".//*[@name='{name}']")

    def ends(name):
        return vec(node(name).get("fromto")).reshape(2, 3)

    for side in ("left", "right"):
        prefix = side + "_fore_"
        src = ["x6_flexion_stator", "drive_crank", "output_crank", "offset_motor_mount", "mount_standoff"]
        dest = [
            "6061_" + side + "_rail",
            "shoulder_crossmember",
            side + "_can_and_power_loom",
            prefix + "drop_bracket",
            prefix + "shoulder_yoke",
            prefix + "x4_pitch_stator",
            prefix + "x4_roll_stator",
        ]
        pairs = [(prefix + s, t) for s in src for t in dest]
        pairs.extend(
            (prefix + s, prefix + t)
            for s in ("x6_flexion_stator", "drive_crank")
            for t in ("upper_strut", "lower_strut", "output_crank")
        )
        pairs += [
            (prefix + "x6_flexion_stator", "PA11_rib_" + str(i) + "_" + str(k))
            for i in (3, 4)
            for k in (1, 2, 3, 8, 9, 10)
        ]
        pairids = [(m.geom(a).id, m.geom(b).id) for a, b in pairs]
        q = [m.joint(prefix + j) for j in ["roll", "pitch", "flexion"]]
        ars = np.array([j.qposadr[0] for j in q])
        grid = [np.linspace(*j.range, 17) for j in q]
        mins = np.ones(len(pairs)) * 400
        ang = np.zeros((len(pairs), 3))
        zeros = np.zeros(len(pairs), dtype=int)
        for pose in itertools.product(*grid):
            d.qpos[ars] = pose
            mj.mj_kinematics(m, d)
            gaps = np.array([mj.mj_geomDistance(m, d, a, b, 0.4, None) * 1000 for a, b in pairids])
            zeros += gaps == 0
            better = gaps < mins
            ang[better] = np.rad2deg(pose)
            mins = np.minimum(mins, gaps)
        result["sides"][side] = {
            "pair_result": {
                f"{a}/{b}": {
                    "minimum_mm": round(float(g), 3),
                    "angles_deg": p.round(2).tolist(),
                    "zero_results": int(z),
                }
                for (a, b), g, p, z in zip(pairs, mins, ang, zeros)
            }
        }
        d.qpos[ars] = 0
        motor = ends(prefix + "x6_flexion_stator").mean(0)
        el = vec(node(prefix + "lower_link").get("pos"))
        sole = vec(node(prefix + "foot").get("pos"))
        rod = ends(prefix + "pushrod")
        pin1 = (rod[0] - motor)[[0, 2]]
        pin2 = (rod[1] - el)[[0, 2]]
        diff = (el - motor)[[0, 2]]
        phi = np.arctan2(pin1[1], pin1[0]) - np.arctan2(diff[1], diff[0])
        axsign = float(node(prefix + "flexion").get("axis").split()[1])
        angles = phi - axsign * np.linspace(*q[-1].range, 301)
        result["sides"][side]["linkage"] = {
            "X6_centre_relative_to_pitch_pivot_mm": (motor * 1000).round(2).tolist(),
            "upper_mm": round(np.linalg.norm(el) * 1000, 3),
            "lower_mm": round(np.linalg.norm(sole) * 1000, 3),
            "crank_centre_distance_in_plane_mm": round(np.linalg.norm(diff) * 1000, 3),
            "pins_mm": [round(np.linalg.norm(x) * 1000, 3) for x in [pin1, pin2]],
            "parallelogram_pin_vector_error_mm": round(np.linalg.norm(pin1 - pin2) * 1000, 3),
            "rod_pin_ends_lateral_mismatch_mm": round(1000 * abs(rod[0, 1] - rod[1, 1]), 4),
            "minimum_transmission_angle_deg": round(np.rad2deg(np.arcsin(abs(np.sin(angles)))).min(), 2),
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(check(a.file), indent=2) + "\n")
