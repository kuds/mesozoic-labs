"""Try one offscreen MuJoCo render under the MUJOCO_GL set in the environment; print the outcome."""
import os
import sys

print("MUJOCO_GL", os.environ.get("MUJOCO_GL"))
try:
    import mujoco
    print("mujoco", mujoco.__version__, mujoco.__file__)
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 3'/><geom type='sphere' size='0.1'/></worldbody></mujoco>")
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 64, 64)
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    px = r.render()
    print("rendered", px.shape, int(px.sum()))
    r.close()
except BaseException as exc:  # noqa: BLE001
    print("FAILED", type(exc).__name__, str(exc)[:500])
    sys.exit(3)
