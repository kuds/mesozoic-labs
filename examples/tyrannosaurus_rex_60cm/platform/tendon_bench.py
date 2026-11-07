"""Direct and antagonistic cable alternatives on isolated, stationary-pelvis appendages."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .engineering import HARDWARE, HERE, sha

DT = 0.0005
DURATION = 8.0
OUTPUT_RADIUS = 0.030
INPUT_RADIUS = 0.010
SPAN = 0.190
ARMATURE = 0.00014
EXTRA_PARTS_KG = {
    "input_drum": 0.012,
    "two_drum_bearings_and_support": 0.024,
    "two_guides_and_cables": 0.012,
    "two_tensioners_and_anchors": 0.018,
    "output_sector_mass_over_direct_hub": 0.010,
}


def appendage(model, name):
    j = model.joint(name)
    d = mujoco.MjData(model)
    dof = j.dofadr[0]
    # Locked base and all other joints. Gravity moment varies with orientation.
    torques = []
    for q in (0, 0.1):
        d.qpos[j.qposadr] = q
        mujoco.mj_forward(model, d)
        torques.append(d.qfrc_bias[dof])
    matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, d, matrix)
    a = float(torques[0])
    b = float((torques[1] - a * math.cos(0.1)) / math.sin(0.1))
    return float(matrix[dof, dof]), a, b


def route(q):
    # Two straight final spans: chassis fairlead (-D,+/-r) to opposite ends of a
    # rotating horn. Upstream moving lengths are +/- r_motor * drum angle.
    s, c = math.sin(q), math.cos(q)
    vals = []
    for side in (-1, 1):
        dx = SPAN - side * OUTPUT_RADIUS * s
        dy = side * OUTPUT_RADIUS * (c - 1)
        length = math.hypot(dx, dy)
        derivative = (dx * -side * OUTPUT_RADIUS * c + dy * -side * OUTPUT_RADIUS * s) / length
        vals.append((side, length - SPAN, derivative))
    return vals


def clamp(x, low, high):
    return min(max(x, low), high)


def trial(properties, name, mode, stiffness=12000.0, preload=18.0, slack=0.0):
    J, a, b = properties
    remote = mode == "remote_cable"
    # Moving output sector includes 10 g beyond the direct hub. Its COM is on-axis.
    J_out = J - ARMATURE + 0.010 * OUTPUT_RADIUS**2 / 2 if remote else J
    J_drum = ARMATURE + 0.012 * INPUT_RADIUS**2 / 2
    amp, hz = (0.17, 0.55) if name == "neck_pitch" else (0.23, 0.48)
    phase_rate = 2 * math.pi * hz
    q = qd = drum = drumd = 0.0
    errors, efforts = [], []
    peak_tension, min_tension, slack_steps = 0.0, math.inf, 0
    arm_min, arm_max = math.inf, 0.0
    max_q, max_drum = 0.0, 0.0
    for step in range(round(DURATION / DT)):
        t = step * DT
        target = amp * math.sin(phase_rate * t)
        target_v = amp * phase_rate * math.cos(phase_rate * t)
        gravity = a * math.cos(q) + b * math.sin(q)
        error = target - q
        # Output encoder dead band approximates 15 arcmin total lost motion.
        dead_error = math.copysign(max(abs(error) - math.radians(0.125), 0.0), error)
        # Extra damping at the drum opposes relative drum/output velocity;
        # no feedback force acts directly on a remote joint.
        request = 11 * dead_error + 0.38 * (target_v - qd) + gravity
        if remote:
            request = request / 3 - 0.012 * (drumd - 3 * qd)
        omega = drumd if remote else qd
        limit = 0.88 * max(0.0, 1.0 - abs(omega) / 7.0)
        torque = clamp(request, -limit, limit)
        if remote:
            torque_out = 0.0
            torque_drum = torque
            tensions = []
            for side, displacement, gradient in route(q):
                # Effective axial stiffness includes cable, crimps, guides and
                # adjuster. Kelvin damping acts only for positive elongation.
                ext = displacement + side * INPUT_RADIUS * drum + preload / stiffness - slack
                ext_v = gradient * qd + side * INPUT_RADIUS * drumd
                tension = max(0.0, stiffness * ext + 1.8 * ext_v) if ext > 0.0 else 0.0
                torque_out -= tension * gradient
                torque_drum -= tension * side * INPUT_RADIUS
                arm_min = min(arm_min, abs(gradient))
                arm_max = max(arm_max, abs(gradient))
                tensions.append(tension)
            min_tension = min(min_tension, *tensions)
            peak_tension = max(peak_tension, *tensions)
            slack_steps += int(min(tensions) < 0.01)
            # Lumped bearing/guide hysteresis and motor-side friction, unmeasured.
            torque_drum -= 0.012 * math.tanh(drumd / 0.04) + 0.0015 * drumd
            drumd += torque_drum / J_drum * DT
            drum += drumd * DT
        else:
            torque_out = torque
        resist = (0.018 if remote else 0.009) * math.tanh(qd / 0.03) + 0.018 * qd
        qd += (torque_out - gravity - resist) / J_out * DT
        q += qd * DT
        max_q, max_drum = max(max_q, abs(q)), max(max_drum, abs(drum))
        errors.append(error)
        efforts.append(torque)
    return dict(
        rms_error_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(errors))))),
        max_error_deg=float(np.rad2deg(np.max(np.abs(errors)))),
        rms_motor_nm=float(np.sqrt(np.mean(np.square(efforts)))),
        peak_motor_nm=float(np.max(np.abs(efforts))),
        min_tension_n=min_tension if remote else None,
        peak_tension_n=peak_tension if remote else None,
        slack_fraction=slack_steps / len(errors) if remote else None,
        output_moment_arm_range_mm=[arm_min * 1000, arm_max * 1000] if remote else None,
        max_abs_output_rad=max_q,
        max_abs_input_rad=max_drum if remote else max_q,
        input_travel_within_1p3_rad=max_drum < 1.3 if remote else None,
    )


def placement_estimate(model, data, folder):
    """Compare neutral mass properties; remote mounts and cable lanes are unqualified."""
    rows = json.loads((folder / "components.json").read_text())["components"]
    # At the geometric zero joints the allocation boxes are aligned to their body
    # frames. Express all centers and central tensors in world before combining.
    parts = {}
    for row in rows:
        half = np.array(row["inertia_box_halfsize_m"])
        owner = data.body(row["body"])
        R = owner.xmat.reshape(3, 3)
        pos = owner.xpos + R @ row["pos_m"]
        m = row["mass_kg"]
        central = R @ np.diag(m / 3.0 * (sum(half**2) - half**2)) @ R.T
        parts[row["name"]] = (m, pos, central)

    def aggregate():
        mass = sum(m for m, _, _ in parts.values())
        center = sum(m * c for m, c, _ in parts.values()) / mass
        central_tensor = np.zeros((3, 3))
        for m, c, inertia in parts.values():
            r = c - center
            central_tensor += inertia + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        return {"mass_kg": mass, "com_world_m": center.tolist(), "central_inertia_world_kg_m2": central_tensor.tolist()}

    direct = aggregate()
    for prefix, joint, axis_sign, side in (("neck", "neck_pitch", -1, -1), ("tail", "tail_yaw", 1, 1)):
        pivot = data.xanchor[model.joint(joint).id].copy()
        mount = pivot + np.array([axis_sign * SPAN, side * 0.070, 0.010])
        servo = prefix + "_xm430"
        m, _, inertia = parts[servo]
        parts[servo] = m, mount, inertia
        for part, mass in EXTRA_PARTS_KG.items():
            if part == "output_sector_mass_over_direct_hub":
                pos = pivot
                half = np.full(3, 0.018)
            elif part == "two_guides_and_cables":
                pos = (pivot + mount) / 2
                half = np.array([SPAN / 2, 0.012, 0.012])
            else:
                pos = mount
                half = np.array([0.020, 0.020, 0.012])
            parts[prefix + "_" + part] = mass, pos, np.diag(mass / 3.0 * (sum(half**2) - half**2))
    return {"base_direct": direct, "remote_unqualified_proposal": aggregate()}


def main():
    folder = HERE / "nominal"
    model = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    results = {}
    for joint in ("neck_pitch", "tail_yaw"):
        props = appendage(model, joint)
        r = {
            key: trial(props, joint, *args)
            for key, args in {
                "base_direct": ("base_direct",),
                "remote_nominal": ("remote_cable",),
                "remote_soft": ("remote_cable", 3000.0, 18.0, 0.0),
                "remote_loose": ("remote_cable", 12000.0, 5.0, 0.001),
            }.items()
        }
        results[joint] = {
            "direct_locked_inertia_kg_m2": props[0],
            "gravity_cos_sin_coefficients_nm": props[1:],
            "trials": r,
        }
    report = dict(
        model_sha256=sha(folder / "model.xml"),
        components_sha256=sha(folder / "components.json"),
        design_sha256=sha(HERE / "design.json"),
        code_sha256=sha(Path(__file__)),
        mujoco=mujoco.__version__,
        seconds=DURATION,
        dt_s=DT,
        full_resolution=True,
        output_radius_m=OUTPUT_RADIUS,
        input_radius_m=INPUT_RADIUS,
        span_m=SPAN,
        nominal_cable_stiffness_n_per_m=12000,
        pretension_each_n=18,
        extra_parts_kg_per_axis=EXTRA_PARTS_KG,
        extra_total_kg_two_axes=2 * sum(EXTRA_PARTS_KG.values()),
        mass_placement=placement_estimate(model, d, folder),
        results=results,
    )
    (HARDWARE / "tendon_bench.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
