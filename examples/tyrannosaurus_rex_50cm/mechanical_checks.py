"""Screen the medium concept's sampled clearances and expression gravity loads.

The 2 mm clearance and 80% torque-use limits are planning criteria. Geometry,
mass and actuator capacities remain estimates; no bearing capacity is qualified.
This screen leaves the design unchanged and can report failures in the baseline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from itertools import combinations, product
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.robot_design_spec import ClearancePair, JointLoadSpec, PoseSweep, RobotDesignSpec
from environments.shared.robot_design_validation import render_witnesses, screen_design, write_report

from . import PARTS_BUDGET_PATH
from .demo import MODEL_PATH

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
MIN_CLEARANCE_M = 0.002
MAX_TORQUE_FRACTION = 0.8
EXPRESSION_JOINTS = (
    "head_yaw",
    "head_nod",
    "jaw_pitch",
    "tail_yaw",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
)
LEG_JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
)
MOTOR_CASES = tuple(f"{joint}_case_collision" for joint in LEG_JOINTS)
UPPER_PROXIES = (
    "torso_collision",
    "neck_support_collision",
    "head_collision",
    "jaw_collision",
    "left_jaw_rail_collision",
    "right_jaw_rail_collision",
    "left_arm_collision",
    "right_arm_collision",
    "tail_segment_0_collision",
    "tail_segment_1_collision",
    "tail_segment_2_collision",
)
# These joined proxy volumes describe continuous rigid structures or a root
# attachment. Head/jaw and motor pairs retain direct checks across body adjacency.
ATTACHMENT_EXCLUSIONS = {
    ("torso_collision", "neck_support_collision"): "Neck support is fixed to the torso.",
    ("torso_collision", "tail_segment_0_collision"): "Tail root intersects its torso attachment.",
    ("jaw_collision", "left_jaw_rail_collision"): "The left rail is part of the same rigid jaw assembly.",
    ("jaw_collision", "right_jaw_rail_collision"): "The right rail is part of the same rigid jaw assembly.",
    ("left_jaw_rail_collision", "right_jaw_rail_collision"): "Both rails belong to the same rigid jaw assembly.",
    ("tail_segment_0_collision", "tail_segment_1_collision"): "These segments form one continuous rigid tail.",
    ("tail_segment_0_collision", "tail_segment_2_collision"): "These segments form one continuous rigid tail.",
    ("tail_segment_1_collision", "tail_segment_2_collision"): "These segments form one continuous rigid tail.",
}


def design_spec(model: mujoco.MjModel) -> RobotDesignSpec:
    """Declare the selected pair coverage and finite pose samples for this robot."""

    def samples(joint: str, count: int) -> tuple[float, ...]:
        lower, upper = model.joint(joint).range
        return tuple(float(value) for value in np.linspace(float(lower), float(upper), count))

    pairs = (
        *combinations(MOTOR_CASES, 2),
        *(pair for pair in combinations(UPPER_PROXIES, 2) if pair not in ATTACHMENT_EXCLUSIONS),
        *product(UPPER_PROXIES, MOTOR_CASES),
    )
    return RobotDesignSpec(
        robot_id="tyrannosaurus_rex_50cm_concept",
        root_joint="root",
        home_keyframe="home",
        feet={"left": ("left_sole_collision",), "right": ("right_sole_collision",)},
        loads=tuple(
            JointLoadSpec(joint, joint + "_servo", max_torque_fraction=MAX_TORQUE_FRACTION)
            for joint in EXPRESSION_JOINTS
        ),
        sweeps=(
            PoseSweep("expression_grid", {joint: samples(joint, 3) for joint in EXPRESSION_JOINTS}),
            *(PoseSweep(joint + "_range", {joint: samples(joint, 5)}) for joint in LEG_JOINTS),
        ),
        clearances=tuple(ClearancePair(first, second, MIN_CLEARANCE_M) for first, second in pairs),
        assumptions=(
            "Masses and component inertias are unweighed surrogates from components.json; actuator force caps "
            "are estimated simulation limits, not verified continuous-duty ratings.",
            "The 2 mm clearance and 80% torque-use thresholds are fixed planning criteria, without a "
            "manufacturing tolerance, thermal, fatigue or dynamic-load qualification.",
            "Bearing moment capacities are unknown. All six expression axes therefore have an incomplete "
            "bearing assessment regardless of their gravity torque margins.",
            "The 729-pose expression grid samples each axis at its lower limit, midpoint and upper limit. "
            "Each of the 12 leg axes separately samples five evenly spaced angles with other axes at home. "
            "The home pose is additional; combined leg poses, leg/expression combinations and unsampled "
            "intermediate angles are outside this screen.",
            "Gravity loads cover only the unsupported head, jaw, tail and arm subtrees. Leg ground reactions, "
            "balance, locomotion, impacts and acceleration require separate checks.",
            "Direct geometry distances cover all 66 motor-case pairs, selected upper-body proxy pairs and "
            "every upper-body proxy against all motor cases, including parent-child and same-body motor pairs.",
            "Collision proxies approximate occupied envelopes, including a solid torso ellipsoid without "
            "motor mounting cutouts. Reported conflicts need CAD review and are not qualified physical "
            "interferences; they are retained as failures of the declared proxy-clearance criterion.",
            "Foot, thigh-link and shin-link geometry pairs, decorative meshes, electronic packing envelopes, "
            "wiring and assembly access are outside the selected pair coverage. Joint attachment proxies "
            "such as hip-pitch-case/thigh-link, knee-case/shin-link and thigh-link/shin-link intentionally "
            "meet at their pivots and require a more detailed joint model before clearance qualification.",
            *(
                f"Excluded pair {first} / {second}: {reason}"
                for (first, second), reason in ATTACHMENT_EXCLUSIONS.items()
            ),
        ),
    )


def source_paths() -> tuple[Path, ...]:
    """Identify the model inputs and local implementation used by this adapter."""
    shared = SOURCE_ROOT / "environments/shared"
    return (
        MODEL_PATH,
        *sorted((HERE / "assets").glob("*.obj")),
        HERE / "components.json",
        PARTS_BUDGET_PATH,
        SOURCE_ROOT / "pyproject.toml",
        *(
            HERE / name
            for name in (
                "__init__.py",
                "mechanical_checks.py",
                "build_model.py",
                "head_design.py",
                "proportions.py",
                "demo.py",
                "render_design.py",
            )
        ),
        *(
            shared / name
            for name in (
                "robot_building.py",
                "robot_design_spec.py",
                "robot_design_validation.py",
                "robot_geometry.py",
                "robot_rendering.py",
                "robot_validation.py",
            )
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write evidence; exit 0 for scoped pass, 1 for failure and 2 for incomplete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "mechanical_validation.json")
    parser.add_argument(
        "--render-dir",
        type=Path,
        help="Optional diagnostic images of worst sampled clearance, torque and bearing poses.",
    )
    args = parser.parse_args(argv)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    spec = design_spec(model)
    report = screen_design(model, spec)
    if args.render_dir:
        report["diagnostics"] = render_witnesses(model, spec, report, args.render_dir)
    command = (
        list(sys.orig_argv)
        if argv is None
        else [sys.executable, "-m", "examples.tyrannosaurus_rex_50cm.mechanical_checks", *argv]
    )
    write_report(report, output=args.output, source_paths=source_paths(), source_root=SOURCE_ROOT, command=command)
    print(f"Mechanical screen: {report['status']}; hardware assessment remains incomplete. {args.output}")
    return {"passed": 0, "failed": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
