"""Test-session defaults, read by pytest before it imports any test module.

The shared test process runs with MuJoCo's OpenGL disabled (MUJOCO_GL=disable): MuJoCo picks its GL backend when it
is first imported, and designlab/__init__.py's default for a machine with no display, OSMesa, loads libOSMesa into the
process, after which torch segfaults while building a stable-baselines3 policy (seen on 11 Nov 2026 in
tests/test_stage1_ppo.py, whichever of torch and mujoco is imported first). No test needs GL in the shared process: the
two tests that render (tests/test_stage2_clips.py) run themselves again in a child process with MUJOCO_GL=osmesa. An
explicit MUJOCO_GL in the environment still wins.
"""

import os

os.environ.setdefault("MUJOCO_GL", "disable")
