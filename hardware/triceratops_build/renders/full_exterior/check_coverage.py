"""Count visible original engineering parts with the cosmetic and hardware groups both on.

This checks silhouettes, not mechanical clearance. The superseded original skin and foot
proxies are expected to intersect a replacement covering and are counted separately.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from render_exterior import fresh_replay

REPLACED = (
    "frill_pa11_panel",
    "hollow_skull_equivalent",
    "_sole",
    "brow_horn_tpu",
    "nasal_horn_tpu",
    "tail_shell_and_flex_core",
    "_limb_cover",
    "_eye",
    "PA11_rib",
    "shell_seam",
)


def inspect(model_path, times, states):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    result = []
    floor_id = model.geom("floor").id
    with mujoco.Renderer(model, height=480, width=800) as renderer:
        renderer.enable_segmentation_rendering()
        for time in (0.25, 3.15, 3.5, 4.0):
            data.qpos[:] = states[np.argmin(abs(times - time))]
            mujoco.mj_forward(model, data)
            for azimuth, elevation in [(90, -10), (125, -10), (200, -10), (250, -10), (52, -10), (115, 30), (245, 30)]:
                cam = mujoco.MjvCamera()
                cam.lookat[:] = data.qpos[:3] + np.array((0, 0, -0.04))
                cam.distance = 1.45
                cam.azimuth = azimuth
                cam.elevation = elevation
                opt = mujoco.MjvOption()
                opt.geomgroup[:] = 1
                opt.sitegroup[:] = 0
                if elevation > 0:
                    opt.geomgroup[0] = 0
                renderer.update_scene(data, camera=cam, scene_option=opt)
                id_img = renderer.render()
                geoms = id_img[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
                ids = id_img[:, :, 0]
                parts = geoms & (ids != floor_id)
                visible = parts & np.isin(ids, np.where(model.geom_group == 5)[0])
                hist = {}
                for i, count in zip(*np.unique(ids[visible], return_counts=True)):
                    name = model.geom(int(i)).name
                    if not any(s in name for s in REPLACED):
                        hist[name] = int(count)
                result.append(
                    {
                        "time_s": time,
                        "azimuth": azimuth,
                        "elevation": elevation,
                        "robot_pixels": int(parts.sum()),
                        "visible_hardware_pixels": hist,
                    }
                )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--replay", type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.replay:
        with np.load(args.replay) as f:
            times, states = f["t"], f["qpos"]
    else:
        times, states = fresh_replay()
    rows = inspect(args.model, times, states)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        f"{len(rows)} sampled views, {sum(not r['visible_hardware_pixels'] for r in rows)} with zero structural pixels"
    )
