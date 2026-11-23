"""Sweep the ideal 1:1 parallelogram's moving rod, including its fixed-stock radius."""

import argparse
import itertools
import json
from pathlib import Path

import mujoco as mj
import numpy as np


def run(path):
    m = mj.MjModel.from_xml_path(str(path))
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    mj.mj_forward(m, d)
    result = {}
    for side in ("left", "right"):
        pre = side + "_fore_"
        rod = m.geom(pre + "pushrod").id
        upper = m.body(pre + "upper_link").id
        motor = m.geom(pre + "x6_flexion_stator")
        el = m.body(pre + "lower_link")
        # The neutral rod pin direction is perpendicular to the shaft separation.
        h = el.pos.copy() - motor.pos
        h[1] = 0.0
        pin = 0.028 * np.array([-h[2], 0, h[0]]) / np.linalg.norm(h)
        names = [
            "6061_" + side + "_rail",
            side + "_can_and_power_loom",
            side + "_shell_seam",
            "shoulder_crossmember",
        ] + [
            pre + t
            for t in [
                "x6_flexion_stator",
                "upper_strut",
                "lower_strut",
                "drop_bracket",
                "x4_pitch_stator",
                "x4_roll_stator",
                "offset_motor_mount",
                "mount_standoff",
                "shoulder_yoke",
                "elbow_axle",
            ]
        ]
        pairs = [(rod, m.geom(n).id) for n in names]
        labels = ["rod/" + n for n in names]
        for name in ["x6_flexion_stator", "offset_motor_mount", "drive_crank", "output_crank"]:
            labels.append(name + "/shell_seam")
            pairs.append((m.geom(pre + name).id, m.geom(side + "_shell_seam").id))
        mins = np.ones(len(pairs)) * 400
        js = [m.joint(pre + j) for j in ("roll", "pitch", "flexion")]
        adr = [j.qposadr[0] for j in js]
        for a in itertools.product(*[np.linspace(*j.range, 17) for j in js]):
            d.qpos[adr] = a
            mj.mj_kinematics(m, d)
            # Both joints bend around the inboard -Y, while the rod remains parallel to itself.
            theta = js[-1].axis[1] * a[-1]
            c, s = np.cos(theta), np.sin(theta)
            moved = np.array([pin[0] * c + pin[2] * s, 0, -pin[0] * s + pin[2] * c])
            d.geom_xpos[rod] += d.xmat[upper].reshape(3, 3) @ (moved - pin)
            gaps = [mj.mj_geomDistance(m, d, x, y, 0.4, None) * 1000 for x, y in pairs]
            mins = np.minimum(mins, gaps)
        result[side] = {nm: round(x, 3) for nm, x in zip(labels, mins)}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.model), indent=2) + "\n")
