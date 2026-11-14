"""Stage 1: a bounded closed-loop probe on the three chosen bodies (``STAGE1_PLAN.md``).

The probe asks whether a controller can use the margins the round-two screens found (``ROUND2_REVIEW.md``
section 13).  It reads the round-two records and models and never changes them; outputs go to
``stage1/runs/<run>/<body>/``.  Stage 1b (``STAGE1B_PLAN.md``) registers arm K's two bodies beside the probe's three,
and Stage 2 (``STAGE2_PLAN.md`` S2-2) the two expressive-head bodies (``BODIES``; ``candidate_id`` maps each to the
round-two candidate id its files carry).

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
#: The Stage 1 bodies and the refinement case directory that holds each one's record and model: the probe's chosen
#: bodies (DECISION_LOG P16), Stage 1b's arm K (``STAGE1B_PLAN.md`` section 2), the two 60 cm picks with round two's
#: faster knee (refine case ``narrow_knee_60``), each named after its pick with ``-K``, and Stage 2's expressive heads
#: (``STAGE2_PLAN.md`` S2-2, DECISION_LOG P29), the two Stage 2 picks with a neck yaw and a hinged jaw in simulation
#: (refine cases ``head_x_60`` and ``head_x_75``: the pick plus the head's drives, not re-sized), named with ``-X``.
BODIES = {
    "60-N-T-10": "round2/refine/narrow_carpet_60",
    "60-N-H-10": "round2/refine/narrow_carpet_60",
    "75-M-H-10": "round2/refine/feet_75",
    "60-N-T-10-K": "round2/refine/narrow_knee_60",
    "60-N-H-10-K": "round2/refine/narrow_knee_60",
    "60-N-T-10-X": "round2/refine/head_x_60",
    "75-M-H-10-X": "round2/refine/head_x_75",
}
#: The probe's own bodies (``STAGE1_PLAN.md``): the set its PPO runs are judged over together (``train.JUDGED_RULE``).
PROBE_BODIES = ("60-N-T-10", "60-N-H-10", "75-M-H-10")
#: Stage 2's expressive-head bodies (S2-2): 16 hinges (the neck yaw and the jaw beside the probe's 14).  Their run
#: (``stage2/x/run_x.sh``) is prepare and G1-G3; G4, the tracker's gate, does not apply.
HEAD_X_BODIES = ("60-N-T-10-X", "75-M-H-10-X")
#: The round-two candidate id of each body whose name is not that id (``candidate_id``).  Arm K's and the expressive
#: heads' bodies are not margin-ledger bodies: ``ledger`` (``sweep``) reads its screens and its per-body tables
#: (``MASSES``, ``STAGE0``) by body name and holds none for them, and neither Stage 1b nor S2-2 sweeps them.
CANDIDATE_IDS = {"60-N-T-10-K": "60-N-T-10", "60-N-H-10-K": "60-N-H-10",
                 "60-N-T-10-X": "60-N-T-10", "75-M-H-10-X": "75-M-H-10"}
#: Leading digits of each record's ``xml_sha256`` as the plans list them (a second check on the record itself): the
#: probe's in ``STAGE1_PLAN.md``, arm K's in ``STAGE1B_PLAN.md`` section 3 (from the case's records), the expressive
#: heads' from their cases' records (``round2/refine/head_x_60``, ``head_x_75``; ``stage2/x/x_inputs.json``).
SHA256_PREFIX = {"60-N-T-10": "993c5ac5", "60-N-H-10": "96ef84c5", "75-M-H-10": "0517e4d3",
                 "60-N-T-10-K": "85d98c24", "60-N-H-10-K": "e0d4c6b7",
                 "60-N-T-10-X": "9718b3b5", "75-M-H-10-X": "66f531bb"}


def case_dir(body: str) -> Path:
    """The refinement case directory of a Stage 1 body."""
    if body not in BODIES:
        raise KeyError(f"{body} is not a Stage 1 body (choose from {', '.join(BODIES)})")
    return STUDY / BODIES[body]


def candidate_id(body: str) -> str:
    """The round-two candidate id of a Stage 1 body: the body's own name unless ``CANDIDATE_IDS`` maps it.  The id
    names the body's record (``<case>/candidates/<id>.json``), model (``<case>/models/<id>.xml``) and screens
    (``<case>/review/screening/<id>_*.json``), and ``round2.candidate`` rebuilds the body from it."""
    case_dir(body)
    return CANDIDATE_IDS.get(body, body)
