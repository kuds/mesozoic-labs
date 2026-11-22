"""Watch the reference walk in real time with a tracking camera."""

import argparse
import time

import glfw
import mujoco

from .build import MODEL_PATH
from .walk import WalkController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stand", action="store_true", help="hold the standing pose")
    parser.add_argument("--seconds", type=float, default=float("inf"), help="close after this much simulated time")
    args = parser.parse_args()
    if not args.seconds > 0:
        parser.error("--seconds must be positive")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = WalkController(model)
    if not glfw.init():
        raise SystemExit("A display is required. Use the validate module for a headless rollout.")
    try:
        window = glfw.create_window(1080, 650, "Stegosaurus stenops", None, None)
        if not window:
            raise SystemExit("Could not open the MuJoCo display.")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        scene = mujoco.MjvScene(model, maxgeom=1000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        options.sitegroup[:] = 0
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = 77, -9, 6.0
        camera.lookat[:] = (-1.16, 0, 1.20)
        frame_steps = max(1, round(1 / (60 * model.opt.timestep)))
        wall_start = time.monotonic()
        try:
            while not glfw.window_should_close(window) and data.time < args.seconds:
                if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    break
                for _ in range(frame_steps):
                    controller.step(data, walking=not args.stand)
                    mujoco.mj_step(model, data)
                camera.lookat[0] = data.qpos[0] - 1.16
                mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                width, height = glfw.get_framebuffer_size(window)
                mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
                glfw.swap_buffers(window)
                glfw.poll_events()
                time.sleep(max(0.0, wall_start + data.time - time.monotonic()))
        finally:
            context.free()
    finally:
        glfw.terminate()


if __name__ == "__main__":
    main()
