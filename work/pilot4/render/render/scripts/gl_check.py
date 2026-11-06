"""Render one frame of a two-geom MuJoCo model through mujoco.Renderer; print the backend, GL strings and pixel stats."""
import os
import mujoco
import numpy as np
xml = "<mujoco><worldbody><light pos='0 0 3'/><geom type='plane' size='2 2 .1' rgba='.3 .5 .3 1'/><geom type='sphere' size='.3' pos='0 0 .5' rgba='.8 .2 .2 1'/></worldbody></mujoco>"
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=120, width=160)
r.update_scene(d)
img = r.render()
print("MUJOCO_GL", os.environ.get("MUJOCO_GL"), "DISPLAY", os.environ.get("DISPLAY"), "mujoco", mujoco.__version__)
try:
    from OpenGL import GL
    print("GL_VENDOR", GL.glGetString(GL.GL_VENDOR), "GL_RENDERER", GL.glGetString(GL.GL_RENDERER), "GL_VERSION", GL.glGetString(GL.GL_VERSION))
except Exception as e:
    print("GL strings unavailable:", type(e).__name__, e)
print("frame", img.shape, img.dtype, "mean", float(img.mean()), "unique colours", len(np.unique(img.reshape(-1, 3), axis=0)))
r.close()
