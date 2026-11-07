"""Uniform contact-material surrogates for the intermediate robot.

These are calibration hypotheses for a dry rigid indoor floor and low pile
bonded to a rigid subfloor. They do not represent pile direction, snagging,
seams or loose rugs. Use an actual force/indentation and slip test to fit both.
"""

from __future__ import annotations

import mujoco

# friction = sliding coefficient, torsional length (m), rolling length (m).
# solref = time constant (s), damping ratio. solimp width is in metres.
FLOORS = {
    "hard": {
        "friction": (0.80, 0.004, 0.0001),
        "solref": (0.008, 1.0),
        "solimp": (0.95, 0.99, 0.0008, 0.5, 2.0),
    },
    "carpet": {
        "friction": (0.95, 0.009, 0.0005),
        "solref": (0.024, 1.2),
        "solimp": (0.65, 0.94, 0.004, 0.5, 2.0),
    },
}


def apply_surface(model: mujoco.MjModel, name: str) -> None:
    """Set the floor before reset; explicit MJCF contact pairs take precedence."""
    if name not in FLOORS:
        raise ValueError("Unknown walking surface: " + str(name))
    floor = model.geom("floor").id
    if floor in model.pair_geom1 or floor in model.pair_geom2:
        raise ValueError("Explicit floor contact pairs override the uniform floor material")
    others = [p for i, p in enumerate(model.geom_priority) if i != floor]
    model.geom_priority[floor] = max(others, default=0) + 1
    model.geom_condim[floor] = 6
    material = FLOORS[name]
    model.geom_solref[floor] = material["solref"]
    model.geom_solimp[floor] = material["solimp"]
    model.geom_friction[floor] = material["friction"]
