"""Replay original and shortened foreassemblies in the same 60 cm exhibit chassis."""

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REFINEMENT = HERE.parent
sys.path.insert(0, str(REFINEMENT / "motion"))
import run  # noqa: E402

SHORT = REFINEMENT / "models/refined/physics.xml"
ORIGINAL = REFINEMENT.parent / "sim/generated/trike60_C_closed_capsule.xml"


def restore_forelegs(destination):
    """Restore shoulders, links and motor centers from the original 188 mm design."""
    tree = ET.parse(SHORT)
    chassis = tree.find(".//body[@name='chassis']")
    donor = ET.parse(ORIGINAL).find(".//body[@name='chassis']")
    for side in ("left", "right"):
        prefix = side + "_fore_"
        for name in ("x4_roll_stator", "x4_roll_stator_collision", "roll_yoke"):
            query = f"*[@name='{prefix}{name}']"
            old = chassis.find(query)
            index = list(chassis).index(old)
            chassis.remove(old)
            chassis.insert(index, donor.find(query))
        chassis.remove(chassis.find(f"geom[@name='{prefix}drop_bracket']"))
    tree.write(destination, encoding="utf-8")


def geometry(model):
    m = mujoco.MjModel.from_xml_path(str(model))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("neutral").id)
    mujoco.mj_forward(m, d)

    def joint(name):
        return d.xanchor[m.joint("left_fore_" + name).id]

    length = 1000 * np.linalg.norm(joint("pitch") - joint("flexion"))
    motor = m.geom("left_fore_x6_flexion_stator")
    upper = m.geom("left_fore_upper_strut")
    return dict(
        fore_upper_mm=length,
        x6_diameter_mm=2000 * motor.size[0],
        x6_diameter_to_fore_upper=2000 * motor.size[0] / length,
        x6_to_strut_neutral_mm=1000 * mujoco.mj_geomDistance(m, d, motor.id, upper.id, 0.5, None),
        neutral_shoulder_height_mm=1000 * joint("pitch")[2],
        neutral_elbow_height_mm=1000 * joint("flexion")[2],
        neutral_ankle_height_mm=1000 * joint("ankle")[2],
    )


def walk(model, seed):
    args = SimpleNamespace(gait="refined", seed=seed, record=False)
    result = run.walk(args, model)
    fields = (
        "pass_all",
        "min_support_margin_mm",
        "max_slide_mm",
        "saturated_pct",
        "speed_mps",
        "peak_foot_N",
        "fore_share",
        "roll_pitch_abs_max_deg",
    )
    # The two front flexion outputs are in positions 2 and 5 of LF, RF, LH, RH triples.
    return {
        **{key: result[key] for key in fields},
        "front_flexion_absmax_Nm": max(result["torque_absmax_Nm"][2:6:3]),
        "front_flexion_rms_max_Nm": max(result["torque_rms_Nm"][2:6:3]),
    }


def turn(model, program, seed, profile, temp):
    args = SimpleNamespace(
        program=program,
        seed=seed,
        record=False,
        random=False,
        yaw_gain=1.5,
        force_trace=temp / "peaks.csv",
        turn_profile=profile,
        gyro_bias_dps=0.0,
    )
    result = run.course(args, model)
    force = result["force_transients"]
    fields = ("raw_peak_N", "raw_peak_foot", "mean_10ms_peak_N", "mean_10ms_peak_foot")
    return {
        **{key: force[key] for key in fields},
        "peak_foot_N": result["peak_foot_N"],
        "completed_duration": not result["terminated"] and result["elapsed_s"] >= run.steering.DURATION[program] - 1e-6,
        "heading_deg": [s["requested_heading_deg"] for s in result["segments"]],
        "first_1s_in_5deg_band_s": [s["first_1s_in_5deg_band_s"] for s in result["segments"]],
        "abs_peak_roll_pitch_deg": result["abs_peak_roll_pitch_deg"],
    }


def panel(model, temp):
    rest = run.static(model)
    return dict(
        geometry=geometry(model),
        stand={key: rest[key] for key in ("mass_kg", "mean_foot_N", "fore_share", "com_xyz_m")},
        walk={seed: walk(model, seed) for seed in (11, 12, 13)},
        original_right_102=turn(model, "pivot-right", 102, "original", temp),
        gentle_right={seed: turn(model, "pivot-right", seed, "gentle", temp) for seed in (101, 102, 103)},
        gentle_six_101=turn(model, "six-headings", 101, "gentle", temp),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-original", type=Path, help="Only generate the original-foreassembly hybrid XML.")
    parser.add_argument("--out", type=Path, default=HERE / "comparison.json")
    opts = parser.parse_args()
    if opts.write_original:
        restore_forelegs(opts.write_original)
        return
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        long_model = temp / "hybrid188.xml"
        restore_forelegs(long_model)
        result = {label: panel(model, temp) for label, model in (("short128", SHORT), ("original188", long_model))}
    opts.out.write_text(json.dumps(result, indent=2, allow_nan=False, default=lambda x: x.item()) + "\n")


if __name__ == "__main__":
    main()
