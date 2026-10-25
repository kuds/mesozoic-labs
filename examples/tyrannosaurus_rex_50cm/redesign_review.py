"""Compare the longer medium candidate with the retained silhouette-C model."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.robot_design_spec import ClearancePair, PoseSweep, RobotDesignSpec
from environments.shared.robot_design_validation import screen_design, write_report
from environments.shared.robot_geometry import visual_bounds

from . import demo, mechanical_checks, redesign, validate
from .leg_load_screen import compare_motor_locations
from .recovery_checks import run_crouch_trial

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
OUTPUT = HERE / "long_body"


def candidate_spec(model: mujoco.MjModel) -> RobotDesignSpec:
    """Keep the baseline thresholds and motion coverage for the revised volumes."""
    baseline = mechanical_checks.design_spec(model)
    upper = redesign.UPPER_PROXIES
    omitted = {frozenset(pair) for pair in redesign.ATTACHMENT_EXCLUSIONS}
    pairs = (
        *itertools.combinations(mechanical_checks.MOTOR_CASES, 2),
        *(pair for pair in itertools.combinations(upper, 2) if frozenset(pair) not in omitted),
        *itertools.product(upper, mechanical_checks.MOTOR_CASES),
    )
    return RobotDesignSpec(
        robot_id="tyrannosaurus_rex_medium_long_body_candidate",
        root_joint=baseline.root_joint,
        home_keyframe=baseline.home_keyframe,
        feet=baseline.feet,
        loads=baseline.loads,
        sweeps=baseline.sweeps,
        clearances=tuple(ClearancePair(first, second, mechanical_checks.MIN_CLEARANCE_M) for first, second in pairs),
        assumptions=(
            "Unweighed component masses, box-surrogate inertias and provisional motor envelopes. "
            "The retained 600 g integration reserve is not a measured component.",
            "Actuator caps and joint ranges are retained from the baseline. The 2 mm clearance and "
            "80% torque-use criteria are unchanged planning thresholds, not hardware qualification.",
            "The 729-pose expression grid combines lower, midpoint and upper limits. Twelve independent "
            "five-angle leg sweeps hold other axes at home; the home pose is also checked. Combined leg "
            "and expression movements and intermediate angles remain unverified.",
            "Selected pairs cover all 66 leg motor-case pairs, the declared upper occupied volumes "
            "against each other, and every upper volume against each motor case. Direct distances "
            "include same-body and parent-child pairs despite normal contact filtering.",
            "Shell volumes describe approximate occupied panels and mounting openings. Decorative "
            "surfaces, electronic packing, wiring, foot and leg-link pairs, mounting tolerances and "
            "assembly access remain outside this clearance screen.",
            "Gravity moments cover unsupported expression subtrees. Bearing capacities are unknown. "
            "Ground reactions, leg loads, acceleration and thermal behavior require separate analyses.",
            *(
                f"Excluded pair {first} / {second}: {reason}"
                for (first, second), reason in redesign.ATTACHMENT_EXCLUSIONS.items()
            ),
        ),
    )


def _mechanical_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "coverage": report["coverage"],
        "failed_clearances": [row for row in report["clearance_checks"] if row["status"] == "failed"],
        "incomplete_clearances": [row for row in report["clearance_checks"] if row["status"] == "incomplete"],
        "minimum_motor_case_clearance_m": min(
            row["min_distance_m"]
            for row in report["clearance_checks"]
            if row["first"] in mechanical_checks.MOTOR_CASES and row["second"] in mechanical_checks.MOTOR_CASES
        ),
        "load_checks": report["load_checks"],
    }


def recorded_clearance(model: mujoco.MjModel, spec: RobotDesignSpec, trial: dict[str, Any]) -> dict[str, Any]:
    """Check the recorded joint configurations, with no continuous-path claim.

    A common rigid root transform cannot change distances between robot parts,
    so the engine can reuse the home root while reproducing all hinge angles.
    Out-of-range rollout poses remain incomplete evidence instead of being
    clamped into the specification's permitted range.
    """
    sweeps = tuple(
        PoseSweep(
            f"frame_{index}_at_{sample['time_s']:.6f}s",
            {
                model.joint(joint).name: (sample["qpos"][int(model.jnt_qposadr[joint])],)
                for joint in range(model.njnt)
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
            },
        )
        for index, sample in enumerate(trial["pose_samples"])
    )
    try:
        result = _mechanical_summary(screen_design(model, replace(spec, loads=(), sweeps=sweeps)))
    except ValueError as error:
        result = {"status": "incomplete", "reason": str(error)}
    result["scope"] = "Home and recorded rollout poses only; intermediate times and floor clearance are unchecked."
    return result


def compare(*, candidate: Path = OUTPUT / "model.xml", duration: float = 10) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the same measurements on both models; preserve all outcomes."""
    cases = {}
    candidate_layout = {}
    candidate_mechanical: dict[str, Any] = {}
    for name, path in (("current", demo.MODEL_PATH), ("long_body", candidate)):
        model, data = demo.load_model(path=path)
        low, high = visual_bounds(model, data)
        spec = candidate_spec(model) if name == "long_body" else mechanical_checks.design_spec(model)
        mechanical = screen_design(model, spec)
        if name == "long_body":
            candidate_mechanical = mechanical
        ledger = json.loads(path.with_name("components.json").read_text())
        if name == "long_body":
            candidate_layout = ledger["layout"]
        rises = [run_crouch_trial(model, knee_angle=angle) for angle in (0.3, 0.6, 0.9)]
        for trial in rises:
            trial["recorded_clearance"] = recorded_clearance(model, spec, trial)
        cases[name] = {
            "model": path.relative_to(SOURCE_ROOT).as_posix() if path.is_relative_to(SOURCE_ROOT) else str(path),
            "length_width_height_m": (high - low).tolist(),
            "mass_kg": float(model.body_mass.sum()),
            "home_com_m": data.subtree_com[model.body("torso").id].tolist(),
            "head_yaw_mass_kg": float(model.body_subtreemass[model.body("head_yaw_body").id]),
            "mechanical": _mechanical_summary(mechanical),
            "standing": {
                "free_stance": validate.run_trial(model, duration=duration),
                "free_gesture": validate.run_trial(model, duration=duration, gesture=True),
                "free_roll_kick": validate.run_trial(model, duration=duration, angular_kick=(0.15, 0, 0)),
                "free_pitch_kick": validate.run_trial(model, duration=duration, angular_kick=(0, -0.15, 0)),
            },
            "crouch_to_stand": rises,
            "leg_motor_location_screen": compare_motor_locations(model, ledger["components"]),
        }
    revised = cases["long_body"]
    failed = (
        candidate_mechanical["status"] == "failed"
        or not all(trial["passed"] for trial in revised["standing"].values())
        or not all(trial["passed"] for trial in revised["crouch_to_stand"])
        or any(trial["recorded_clearance"]["status"] == "failed" for trial in revised["crouch_to_stand"])
    )
    return {
        "schema_version": 1,
        "status": "failed" if failed else "incomplete",
        "scope": "Medium layout comparison, sampled mechanical screens and free-base standing/crouch-rise trials.",
        "design_goal": "Longer, lower medium layout; crouch-to-stand first, fall recovery remains a design goal.",
        "layout": candidate_layout,
        "cases": cases,
        "fall_recovery": {
            "status": "not_demonstrated",
            "reason": "No prone, supine or side-fall recovery trajectory or qualified load-bearing contact design.",
        },
        "limitations": [
            "Crouch-to-stand begins from a constructed upright, flat-foot pose; it is not recovery from the floor.",
            "Position drives are ideal torque-limited PD controls, without calibrated latency, backlash or heating.",
            "The remote-motor comparison moves mass only and adds no transmission mass or losses; it is not a design.",
            "Proxy-clearance improvements and successful rollouts do not qualify fabrication, walking or hardware.",
        ],
    }, candidate_mechanical


def source_paths(candidate: Path = OUTPUT / "model.xml") -> tuple[Path, ...]:
    return (
        *mechanical_checks.source_paths(),
        Path(__file__),
        HERE / "redesign.py",
        HERE / "recovery_checks.py",
        HERE / "leg_load_screen.py",
        HERE / "validate.py",
        candidate,
        candidate.with_name("components.json"),
        *sorted((candidate.parent / "assets").glob("*.obj")),
    )


def render_comparison(report: dict[str, Any], output_dir: Path = OUTPUT) -> None:
    """Render compiled surfaces at the same scale and replay recorded rise poses."""
    from PIL import Image, ImageDraw

    from environments.shared.robot_rendering import font

    from .render_design import camera_basis, geometry, render

    model_paths = [Path(report["cases"][name]["model"]) for name in ("current", "long_body")]
    models = [demo.load_model(path=path if path.is_absolute() else SOURCE_ROOT / path) for path in model_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1600, 1320), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 20), "MEDIUM T-REX / LONGER BODY CANDIDATE", font=font(30), fill="#26372b")
    draw.text(
        (30, 65),
        "Compiled model surfaces at the same scale / estimated masses / provisional hardware",
        font=font(19),
        fill="#536158",
    )
    for row, view in enumerate(("isometric", "side")):
        points = [geometry(model, data)[0] @ camera_basis(view).T for model, data in models]
        bounds = (
            np.min([p[:, :, :2].min(axis=(0, 1)) for p in points], axis=0),
            np.max([p[:, :, :2].max(axis=(0, 1)) for p in points], axis=0),
        )
        for column, ((model, data), name) in enumerate(zip(models, ("current", "long_body"), strict=True)):
            case = report["cases"][name]
            x, y = column * 800, 105 + row * 550
            sheet.paste(render(model, data, view=view, width=800, height=510, projection_bounds=bounds), (x, y + 32))
            length, width, height = case["length_width_height_m"]
            draw.text(
                (x + 25, y),
                f"{name.replace('_', ' ').upper()}  |  {height * 100:.1f} cm H / {length * 100:.1f} cm L / {case['mass_kg']:.3f} kg",
                font=font(20),
                fill="#26372b",
            )
    draw.text(
        (30, 1235),
        "Local leg drives and actuator limits retained. Clearances, loads and crouch-rise evidence: comparison.json",
        font=font(19),
        fill="#536158",
    )
    draw.text(
        (30, 1270),
        "Fall recovery and walking remain undemonstrated. These are model views, not fabrication CAD.",
        font=font(19),
        fill="#536158",
    )
    sheet.save(output_dir / "comparison.png")
    model, _ = models[1]
    data = mujoco.MjData(model)
    trial = report["cases"]["long_body"]["crouch_to_stand"][-1]
    projections = []
    for sample in trial["pose_samples"]:
        data.qpos[:] = sample["qpos"]
        mujoco.mj_forward(model, data)
        projections.append(geometry(model, data)[0] @ camera_basis("side").T)
    fixed_bounds = (
        np.min([p[:, :, :2].min(axis=(0, 1)) for p in projections], axis=0),
        np.max([p[:, :, :2].max(axis=(0, 1)) for p in projections], axis=0),
    )
    # Side projection uses world X/Z. A fixed floor line and framing make the
    # measured vertical rise visible instead of re-centering each replay pose.
    low, high = fixed_bounds
    scale = min(640 * 0.84 / (high[0] - low[0]), 520 * 0.82 / (high[1] - low[1]))
    floor_y = 260 + (low[1] + high[1]) / 2 * scale
    frames = []
    for sample in trial["pose_samples"]:
        data.qpos[:] = sample["qpos"]
        mujoco.mj_forward(model, data)
        frame = render(model, data, view="side", width=640, height=520, projection_bounds=fixed_bounds)
        draw = ImageDraw.Draw(frame)
        draw.line((24, floor_y, 616, floor_y), fill="#a5aaa6", width=1)
        draw.text((12, 12), f"CROUCH-TO-STAND MODEL TRIAL / {sample['time_s']:.1f} s", font=font(17), fill="#26372b")
        draw.text(
            (12, 482),
            f"Trial {'passed' if trial['passed'] else 'FAILED'} / no external support / not fall recovery",
            font=font(16),
            fill="#536158",
        )
        frames.append(frame)
    if frames:
        interval = round(
            1000 * (trial["pose_samples"][-1]["time_s"] - trial["pose_samples"][0]["time_s"]) / max(1, len(frames) - 1)
        )
        frames[0].save(
            output_dir / "crouch_to_stand.gif", save_all=True, append_images=frames[1:], duration=interval, loop=0
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    report, mechanical = compare(duration=args.duration)
    command = (
        list(sys.orig_argv)
        if argv is None
        else [sys.executable, "-m", "examples.tyrannosaurus_rex_50cm.redesign_review", *argv]
    )
    for payload, filename in ((report, "comparison.json"), (mechanical, "mechanical_validation.json")):
        write_report(
            payload, output=OUTPUT / filename, source_paths=source_paths(), source_root=SOURCE_ROOT, command=command
        )
    if args.render:
        render_comparison(report)
    print(
        f"Long-body candidate: {report['status']}; fall recovery remains undemonstrated. {OUTPUT / 'comparison.json'}"
    )
    return 1 if report["status"] == "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
