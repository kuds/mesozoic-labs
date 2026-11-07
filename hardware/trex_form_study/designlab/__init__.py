"""Parametric T. rex robot design study (form, size, posture and hardware screening).

Every candidate is generated from explicit parameters (``params.Candidate``), a sourced
component catalog (``data/*.json``) and a cited anatomical template (``data/anatomy.json``).
Outputs are MuJoCo MJCF models plus component-level ledgers; the screening modules read the
compiled models, never the parameter tables, so what is reported is what was built.

Units: metres, kilograms, seconds and radians in code; millimetres in reports.
Frame: +X forward, +Y anatomical left, +Z up, floor at z = 0.
"""

import os as _os

# Offscreen rendering backend (must be chosen before mujoco is imported anywhere).
_os.environ.setdefault("MUJOCO_GL", "egl" if _os.environ.get("DISPLAY") is None and _os.path.exists("/dev/dri") else "osmesa")
_os.environ.setdefault("PYOPENGL_PLATFORM", _os.environ["MUJOCO_GL"])

STUDY_REVISION = "r1"
