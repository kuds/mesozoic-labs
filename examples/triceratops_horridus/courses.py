"""Adult Triceratops walking courses, in metres and degrees."""

from dataclasses import replace

from examples.shared.ground_contacts import GroundContacts
from examples.shared.locomotion_courses import Course, Mound
from examples.shared.turn_metrics import TurnCourse

from .reference_walk import FEET, PERIOD

COURSES = {
    "uphill": Course(pitch_deg=5),
    "downhill": Course(pitch_deg=-5),
    "cross_left": Course(bank_deg=3),
    "cross_right": Course(bank_deg=-3),
    "uneven": Course(
        mounds=(
            Mound(2.25, 0.55, 0.75, 0.70, 0.035),
            Mound(3.00, -0.55, 0.80, 0.62, 0.050),
            Mound(3.95, 0.45, 0.90, 0.85, 0.060),
            Mound(4.65, -0.50, 0.72, 0.75, 0.040),
            Mound(5.70, -0.05, 0.90, 1.35, 0.055),
            Mound(6.80, 0.65, 0.80, 0.65, 0.045),
            Mound(7.44, -0.58, 0.84, 0.81, 0.050),
        )
    ),
}

COURSES["uneven_mirrored"] = Course(mounds=tuple(replace(mound, y=-mound.y) for mound in COURSES["uneven"].mounds))

EXTENDED_COURSES = {
    "uphill_8deg": Course(pitch_deg=8),
    "downhill_8deg": Course(pitch_deg=-8),
    "cross_left_5deg": Course(bank_deg=5),
    "cross_right_5deg": Course(bank_deg=-5),
}

TURN_COURSES = {
    direction: TurnCourse(sign * 5.0, 0.8 + 2 * PERIOD, 13 * PERIOD, 4 * PERIOD)
    for direction, sign in (("left", 1), ("right", -1))
}


def foot_contacts(model, surfaces=("ground",)):
    pads = {foot: [f"{foot}_pad_{i}" for i in (1, 2, 3)] for foot in FEET}
    return GroundContacts(model, pads, surfaces)
