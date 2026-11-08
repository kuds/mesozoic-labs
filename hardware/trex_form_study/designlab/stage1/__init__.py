"""Stage 1: a bounded closed-loop probe on the three chosen bodies (``STAGE1_PLAN.md``).

The probe asks whether a controller can use the margins the round-two screens found (``ROUND2_REVIEW.md``
section 13).  It reads the round-two records and models and never changes them; outputs go to
``stage1/runs/<run>/<body>/``.

Modules (plan section 2):

* ``plant``: the screened MJCF of a body, checked against its record's ``xml_sha256``, with sensors, the
  hold-servo gains, the usable joint ranges, a floor and a settled home pose (or the finding that it does not stand
  there);
* ``drive``: the two-motor ankle linkage and the torque-speed envelope of every drive;
* ``refs``: the reference walks and turns with their inverse-dynamics feedforward, regenerated and checked
  against the records;
* ``legs``: leg-to-leg proximity on simulated traces (only the foot pads collide in the plant);
* ``gates``: the fail-fast checks G1 (plant) and G2 (references) of plan section 3, with the evidence for the open
  choices G2 depends on (``gates.open_decisions``).

Run from hardware/trex_form_study:
    python -m designlab.stage1 prepare --body 60-N-T-10 --out stage1/runs/<run>
"""

from __future__ import annotations

from pathlib import Path

#: The study directory (hardware/trex_form_study).
STUDY = Path(__file__).resolve().parents[2]
#: The chosen bodies (DECISION_LOG P16) and the refinement case directory that holds each one's record and model.
BODIES = {
    "60-N-T-10": "round2/refine/narrow_carpet_60",
    "60-N-H-10": "round2/refine/narrow_carpet_60",
    "75-M-H-10": "round2/refine/feet_75",
}
#: Leading digits of each record's ``xml_sha256`` as the plan lists them (a second check on the record itself).
SHA256_PREFIX = {"60-N-T-10": "993c5ac5", "60-N-H-10": "96ef84c5", "75-M-H-10": "0517e4d3"}


def case_dir(body: str) -> Path:
    """The refinement case directory of a chosen body."""
    if body not in BODIES:
        raise KeyError(f"{body} is not a Stage 1 body (choose from {', '.join(BODIES)})")
    return STUDY / BODIES[body]
