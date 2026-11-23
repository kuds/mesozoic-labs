"""Compare sagittal ankle reach and external load lever arms in the two authored forelegs."""

import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REFINEMENT = HERE.parent
VERTICAL_LOAD_N = 35.0
HEIGHT_BAND_M = 0.001
STEP_DEG = 0.25


def joint_samples(model, joint):
    low, high = model.jnt_range[joint.id]
    intervals = int(np.ceil(np.degrees(high - low) / STEP_DEG))
    return np.linspace(low, high, intervals + 1)


def measure(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral").id)
    mujoco.mj_kinematics(model, data)
    pitch, elbow, ankle = [model.joint(f"left_fore_{name}") for name in ("pitch", "flexion", "ankle")]
    pad = data.site("left_fore_load_cell").xpos.copy()
    neutral = data.xanchor[ankle.id].copy()
    shoulder = data.xanchor[pitch.id].copy()
    hub = data.xanchor[elbow.id].copy()
    force = np.array([0.0, 0.0, VERTICAL_LOAD_N])
    moment = np.dot(data.xaxis[elbow.id], np.cross(pad - hub, force))
    xs = []
    pitches, flexions = joint_samples(model, pitch), joint_samples(model, elbow)
    for angle in pitches:
        data.qpos[pitch.qposadr[0]] = angle
        for flexion in flexions:
            data.qpos[elbow.qposadr[0]] = flexion
            mujoco.mj_kinematics(model, data)
            position = data.xanchor[ankle.id]
            if abs(position[2] - neutral[2]) <= HEIGHT_BAND_M:
                xs.append(position[0] - neutral[0])
    return {
        "source": str(path.relative_to(REFINEMENT.parent)),
        "upper_length_mm": round(float(np.linalg.norm(hub - shoulder)) * 1000, 3),
        "neutral_shoulder_above_ankle_mm": round(float(shoulder[2] - neutral[2]) * 1000, 3),
        "neutral_elbow_load_lever_mm": round(abs(float(moment)) / VERTICAL_LOAD_N * 1000, 3),
        "neutral_external_elbow_moment_Nm": round(abs(float(moment)), 3),
        "rearward_ankle_reach_mm": round(-min(xs) * 1000, 2),
        "forward_ankle_reach_mm": round(max(xs) * 1000, 2),
        "in_height_band_samples": len(xs),
    }


if __name__ == "__main__":
    paths = {
        "short_open_rib": REFINEMENT / "models/refined/physics.xml",
        "original_foreleg": REFINEMENT.parent / "sim/generated/trike60_C_closed_capsule.xml",
    }
    results = {
        "vertical_pad_center_force_N": VERTICAL_LOAD_N,
        "ankle_height_tolerance_mm": 1000 * HEIGHT_BAND_M,
        "maximum_grid_step_deg": STEP_DEG,
        "legs": {label: measure(path) for label, path in paths.items()},
    }
    print(json.dumps(results, indent=2))
