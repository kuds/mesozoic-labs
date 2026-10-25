"""Load, inspect and animate the selected silhouette-C T-Rex MJCF."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).with_name("model.xml")


def set_support(model: mujoco.MjModel, data: mujoco.MjData, enabled: bool) -> None:
    """Explicit display-only torso weld; never a balance controller."""
    data.eq_active[model.equality("display_support").id] = enabled
    mujoco.mj_forward(model, data)


def load_model(*, supported: bool = False, path: Path = MODEL_PATH) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    set_support(model, data, supported)
    return model, data


def gesture_controls(model: mujoco.MjModel, elapsed: float) -> np.ndarray:
    """Gentle standing gestures, not a locomotion policy."""
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed must be finite and nonnegative")
    phase = 2 * math.pi * elapsed / 5
    ramp = 0.5 - 0.5 * math.cos(math.pi * min(elapsed, 1))
    motion = {
        "head_yaw": 0.15 * math.sin(0.7 * phase),
        "head_pitch": -0.035 + 0.045 * math.sin(phase),
        "jaw_pitch": 0.05 * (1 - math.cos(phase)),
        "left_shoulder_pitch": -0.15 * (1 - math.cos(phase)),
        "right_shoulder_pitch": -0.15 * (1 - math.cos(0.8 * phase)),
        "tail_yaw": 0.12 * math.sin(2 * math.pi * 0.5 * elapsed),
    }
    controls = np.array(model.key_ctrl[0], copy=True)
    for joint, value in motion.items():
        controls[model.actuator(joint + "_servo").id] = ramp * value
    np.clip(controls, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1], out=controls)
    return controls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Interactive MuJoCo viewer; use mjpython on macOS.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Defaults to selected silhouette C.")
    parser.add_argument("--support", action="store_true", help="Explicitly weld torso for a display demonstration.")
    parser.add_argument("--gesture", action="store_true", help="Animate arms, head, jaw and tail while standing.")
    parser.add_argument("--render", type=Path, help="CPU render of the compiled geometry; requires the viz extra.")
    parser.add_argument("--camera", choices=("isometric", "side", "front"), default="isometric")
    parser.add_argument(
        "--service", action="store_true", help="Hide torso panels and show provisional component boxes."
    )
    args = parser.parse_args()
    if not (args.view or args.render):
        parser.print_help()
        return 0
    model, data = load_model(supported=args.support, path=args.model)
    if args.render:
        from .render_design import render

        args.render.parent.mkdir(parents=True, exist_ok=True)
        render(model, data, view=args.camera, service=args.service, width=1100, height=1100).save(args.render)
        print(args.render)
    if args.view:
        import mujoco.viewer

        print("DISPLAY SUPPORT ACTIVE — torso welded" if args.support else "FREE BASE — no external support")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.geomgroup[3:] = 0
            viewer.cam.lookat[:] = [-0.05, 0, 0.5]
            viewer.cam.distance = 2.2
            viewer.cam.azimuth, viewer.cam.elevation = 135, -15
            if args.service:
                viewer.opt.geomgroup[4] = 1
                for g in range(model.ngeom):
                    name = model.geom(g).name
                    if name in ("torso_shell", "belly_panel", "back_access_panel") or name.startswith(
                        ("back_vent_", "back_screw_")
                    ):
                        model.geom_group[g] = 5
            while viewer.is_running():
                start = time.monotonic()
                if args.gesture:
                    data.ctrl[:] = gesture_controls(model, data.time)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - start)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
