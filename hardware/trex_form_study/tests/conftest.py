"""Test-session defaults, read by pytest before it imports any test module.

The rendering tests (designlab/render.py, designlab/stage2/clips.py) need MuJoCo's offscreen OSMesa backend. MuJoCo
chooses its OpenGL backend when it is first imported; without MUJOCO_GL it falls back to GLFW, which on a machine with
no display fails to make a context and can abort the whole test process. render.py and clips.py set the same default,
but only when they are imported, which in a full run comes after other test modules have imported mujoco. So the
default is set here, before any of them; an explicit MUJOCO_GL in the environment still wins.
"""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
