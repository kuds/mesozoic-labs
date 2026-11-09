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
* ``transitions``: starts, stops, speed changes and turn entries and exits, by ZMP preview control on the designlab
  model with ``statics.leg_ik``, joined to the periodic references;
* ``control``: the tracker (R0: feedforward and PD; R1: DCM feedback on the true state; R2: on the estimate) and its
  episodes, with the ankle torque cap, the tuning start and the inward step bound of decisions 15, 16 and 20;
  ``estimator``: CoM and velocity from encoders, IMU and stance-leg kinematics at the loaded pad's contact point
  (decision 19);
* ``gates``: the fail-fast checks G1 (plant), G2 (references), G3 (transitions) and G4 (tracker) of plan section 3,
  with the choices they depend on as the plan decided them during the build (decisions 11-13, ``gates.decided``;
  15-20, ``gates.g4_decided``) and the evidence they keep;
* ``metrics``: plan section 4's walk, turn and start-and-stop measures on a simulated episode (``Recorder``), and the
  standing noise floor behind R2's rest speed;
* ``tune``: the CMA-ES fit of R2's gains (plan section 3 step 6) over a fixed mix of the behaviours G5 judges,
  checkpointed every generation, and gate G5 on the validation episodes (``tune.json``).

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
