"""Manual unsupported dynamic-stepping diagnostic; never a passing walking gate.

Selected sinusoidal targets plus ideal body-orientation/gyro feedback are bounded
by the current model's actuator target ranges. Results must be read with contact,
clearance, torque, speed, joint-limit and failure evidence. No training or motor
identification is performed. An observed step does not validate physical walking.

Run ``python -m examples.mini_tyrannosaurus_rex.walking_pilot --output pilot.json``.
Use ``--tail --tail-mode wag`` to replay the same controller on the optional tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).with_name("model.xml")
PARAMETERS = [0.3611035514154559, 0.14115136271266485, 0.03911254725273986, 0.4058584529699947, 2.656837692232398, 0.0]
FEEDBACK = [0.5, 0.25, 0.03]
# Fixed after an exploratory search seeded with 635; this replay uses no RNG.
PARAMETER_NAMES = ["period_s", "hip_roll_rad", "hip_stride_rad", "knee_flex_rad", "roll_phase_rad", "knee_bias_rad"]
CONTROL_RATE_HZ = 50


class WalkingPilot:
    """Replay one fixed controller on a free robot, with measured failure evidence."""

    def __init__(self, path: Path = MODEL_PATH, *, tail: bool = False):
        self.model_path = Path(path)
        self.source_sha256 = {
            str(source.relative_to(self.model_path.parent)): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in [self.model_path, *sorted((self.model_path.parent / "assets").glob("*.xml"))]
        }
        self.generator_source_sha256 = {}
        if tail:
            tail_option = import_module(f"{__package__}.tail_option" if __package__ else "tail_option")
            if tail_option.__file__ is None:
                raise ValueError("Tail builder must have a source file for provenance")
            self.generator_source_sha256["tail_option.py"] = hashlib.sha256(
                Path(tail_option.__file__).read_bytes()
            ).hexdigest()
            self.model = tail_option.load_model(self.model_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.torso = self.model.body("torso").id
        self.floor = self.model.geom("floor").id
        self.pads = [self.model.geom(side + "_flat_pad_collision").id for side in ("left", "right")]
        if not np.all(self.model.geom_type[self.pads] == mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("Minimum sole clearance requires box contact pads")
        if not (
            self.model.geom_type[self.floor] == mujoco.mjtGeom.mjGEOM_PLANE
            and self.model.geom_bodyid[self.floor] == 0
            and np.allclose(self.model.geom_pos[self.floor], 0)
            and np.allclose(self.model.geom_quat[self.floor], [1, 0, 0, 0])
        ):
            raise ValueError("This diagnostic requires a stationary horizontal floor at z=0")
        self.leg_names = [
            side + "_" + joint
            for side in ("left", "right")
            for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
        ]
        self.leg_actuators = [self.model.actuator(name + "_servo").id for name in self.leg_names]
        self.leg_qpos = [self.model.jnt_qposadr[self.model.joint(name).id] for name in self.leg_names]
        if not np.allclose(self.model.opt.gravity[:2], 0) or self.model.opt.gravity[2] >= 0:
            raise ValueError("This diagnostic requires vertical downward gravity")
        self.weight = float(self.model.body_mass.sum() * -self.model.opt.gravity[2])
        control_steps = 1 / (CONTROL_RATE_HZ * self.model.opt.timestep)
        self.control_steps = round(control_steps)
        if self.control_steps < 1 or not math.isclose(control_steps, self.control_steps):
            raise ValueError("Physics timestep must divide the 50 Hz control period exactly")
        if not np.all(self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) or not np.allclose(
            self.model.actuator_gear, np.tile([1, 0, 0, 0, 0, 0], (self.model.nu, 1))
        ):
            raise ValueError("Torque reporting requires direct joint actuators with unit gear")
        if not np.all(self.model.actuator_forcelimited):
            raise ValueError("All actuators must have explicit force limits")
        self.contact_force = np.zeros(6)

    def _controls(self, t: float, p: Sequence[float]) -> np.ndarray:
        """Return unbounded targets; the caller clips after orientation feedback."""
        period, roll, hip, knee, phase, bias = p
        phase_t = 2 * math.pi * max(0, t - 1) / period
        ramp = min(max(t - 1, 0), 1)
        q = np.zeros(10)
        for s in range(2):
            a = phase_t + s * math.pi
            k = bias + knee * max(0, math.cos(a))
            h = -0.5 * k - hip * math.sin(a)
            q[s * 5 + 1] = roll * math.cos(phase_t + phase)
            q[s * 5 + 2] = h
            q[s * 5 + 3] = k
            q[s * 5 + 4] = -h - k
        return ramp * q

    def run(
        self,
        p: Sequence[float] = PARAMETERS,
        duration: float = 30,
        store: bool = False,
        cap: float | None = None,
        feedback: Sequence[float] | None = FEEDBACK,
        tail_mode: str = "none",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Start at home, simulate unsupported, and restore the model's force caps.

        Torque caps are exploratory limits, not measured continuous motor ratings;
        a requested cap cannot increase any loaded leg actuator's limits.
        Exact simulator orientation is privileged feedback: hardware needs an IMU
        orientation estimator and separate latency/noise, torque-speed and thermal
        validation. No foot-force or root-position feedback drives this controller.
        """
        if len(p) != 6 or not np.all(np.isfinite(p)) or p[0] <= 0:
            raise ValueError("Expected six finite controller parameters with a positive period")
        if not math.isfinite(duration) or duration <= 2:
            raise ValueError("Duration must be finite and greater than two seconds")
        if cap is not None and (not math.isfinite(cap) or cap <= 0):
            raise ValueError("Torque cap must be finite and positive")
        if cap is not None:
            limits = self.model.actuator_forcerange[self.leg_actuators]
            maximum_cap = float(np.min(np.minimum(-limits[:, 0], limits[:, 1])))
            if cap > maximum_cap:
                raise ValueError(f"Requested torque cap exceeds the weakest loaded leg limit ({maximum_cap:g} Nm)")
        if feedback is not None and (len(feedback) != 3 or not np.all(np.isfinite(feedback))):
            raise ValueError("Feedback must contain three finite gains")
        if tail_mode not in {"none", "center", "wag"}:
            raise ValueError("Tail mode must be none, center, or wag")
        tail_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tail_yaw_servo")
        if tail_mode != "none" and tail_id < 0:
            raise ValueError("Tail mode requires a tail_yaw_servo actuator")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        for equality in range(self.model.neq):
            if not self.data.eq_active[equality] or self.model.eq_type[equality] != mujoco.mjtEq.mjEQ_WELD:
                continue
            bodies = [int(self.model.eq_obj1id[equality]), int(self.model.eq_obj2id[equality])]
            if self.model.eq_objtype[equality] == mujoco.mjtObj.mjOBJ_SITE:
                bodies = [int(self.model.site_bodyid[site]) for site in bodies]
            if any(body < 0 or self.model.body_weldid[body] == 0 for body in bodies):
                raise ValueError("Home must have every world weld disabled, including fixed-fixture welds")
        oldcaps = self.model.actuator_forcerange.copy()
        try:
            if cap is not None:
                self.model.actuator_forcerange[self.leg_actuators, :] = [-cap, cap]
            return self._simulate(p, duration, store, feedback, tail_mode, tail_id, cap)
        finally:
            self.model.actuator_forcerange[:] = oldcaps

    def _simulate(
        self,
        p: Sequence[float],
        duration: float,
        store: bool,
        feedback: Sequence[float] | None,
        tail_mode: str,
        tail_id: int,
        cap: float | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        mujoco.mj_forward(self.model, self.data)
        start = self.data.qpos[:3].copy()
        max_tilt = 0.0
        min_height = start[2]
        max_torque = np.zeros(self.model.nu)
        rms = np.zeros(self.model.nu)
        sat = np.zeros(self.model.nu)
        step_count = 0
        peak_speed = np.zeros(self.model.nu)
        actuator_dofs = [self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]] for i in range(self.model.nu)]
        segment_side = -1
        segment_time = 0.0
        segments = []
        clipped = 0
        joint_violations = np.zeros(self.model.njnt)
        free_intervals = [0.0, 0.0]
        max_clearance = np.zeros(2)
        nonfoot = False
        records = []
        minimum_contact_ratio = np.ones(2)
        stop_reason = "duration_reached"
        self_contacts: dict[str, dict[str, float]] = {}
        external_force_max = 0.0
        for n in range(round(duration / self.model.opt.timestep)):
            if n % self.control_steps == 0:
                q = self._controls(self.data.time, p)
                if feedback is not None:
                    rp, pp, derivative = feedback
                    rotation = self.data.xmat[self.torso].reshape(3, 3)
                    roll = math.atan2(rotation[2, 1], rotation[2, 2])
                    pitch = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
                    gyro = self.data.sensor("body_gyro").data
                    for s in range(2):
                        q[s * 5 + 1] += rp * roll + derivative * gyro[0]
                        q[s * 5 + 4] += pp * pitch + derivative * gyro[1]
                bounded = np.clip(
                    q,
                    self.model.actuator_ctrlrange[self.leg_actuators, 0],
                    self.model.actuator_ctrlrange[self.leg_actuators, 1],
                )
                clipped += int(np.any(np.abs(q - bounded) > 1e-09))
                self.data.ctrl[self.leg_actuators] = bounded
                if tail_id >= 0:
                    tail_target = (
                        0.2 * math.sin(math.pi * self.data.time) * min(max(self.data.time - 1, 0), 1)
                        if tail_mode == "wag"
                        else 0.0
                    )
                    self.data.ctrl[tail_id] = np.clip(tail_target, *self.model.actuator_ctrlrange[tail_id])
            mujoco.mj_step(self.model, self.data)
            step_count += 1
            external_force_max = max(
                external_force_max,
                float(np.abs(self.data.qfrc_applied).max()),
                float(np.abs(self.data.xfrc_applied).max()),
            )
            for j in range(self.model.njnt):
                if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    continue
                value = self.data.qpos[self.model.jnt_qposadr[j]]
                joint_violations[j] = max(
                    joint_violations[j], self.model.jnt_range[j, 0] - value, value - self.model.jnt_range[j, 1]
                )
            torque = np.abs(self.data.actuator_force)
            peak_speed = np.maximum(peak_speed, np.abs(self.data.qvel[actuator_dofs]))
            max_torque = np.maximum(max_torque, torque)
            rms += torque**2
            sat += torque >= 0.95 * self.model.actuator_forcerange[:, 1]
            up = self.data.xmat[self.torso].reshape(3, 3)[2, 2]
            tilt = math.acos(float(np.clip(up, -1, 1)))
            max_tilt = max(max_tilt, tilt)
            min_height = min(min_height, self.data.qpos[2])
            f = np.zeros(2)
            bad_contact = False
            for ci in range(self.data.ncon):
                c = self.data.contact[ci]
                mujoco.mj_contactForce(self.model, self.data, ci, self.contact_force)
                normal = max(0, float(self.contact_force[0]))
                if self.floor not in c.geom:
                    pair = " / ".join(sorted(self.model.geom(int(g)).name for g in c.geom))
                    entry = self_contacts.setdefault(pair, {"peak_penetration_m": 0.0, "peak_normal_force_N": 0.0})
                    entry["peak_penetration_m"] = max(entry["peak_penetration_m"], -float(c.dist))
                    entry["peak_normal_force_N"] = max(entry["peak_normal_force_N"], normal)
                    continue
                other = int(c.geom[1] if c.geom[0] == self.floor else c.geom[0])
                if other in self.pads:
                    side = self.pads.index(other)
                    f[side] += normal
                elif normal > 0.1:
                    bad_contact = True
            clearance = np.array(
                [
                    self.data.geom_xpos[g, 2]
                    - np.abs(self.data.geom_xmat[g].reshape(3, 3)[2]) @ self.model.geom_size[g]
                    for g in self.pads
                ]
            )
            if self.data.time > 2:
                minimum_contact_ratio = np.minimum(minimum_contact_ratio, f / self.weight)
                max_clearance = np.maximum(max_clearance, clearance)
            stance = -1
            if self.data.time > 2 and (not bad_contact) and (tilt < 0.52):
                for s in range(2):
                    if f[s] > 0.5 * self.weight and f[1 - s] < 0.05 and (clearance[1 - s] > 0.002):
                        stance = s
                        free_intervals[1 - s] += self.model.opt.timestep
            if stance == segment_side:
                segment_time += self.model.opt.timestep
            else:
                if segment_side >= 0:
                    segments.append(
                        {
                            "stance": segment_side,
                            "duration": segment_time,
                            "end": float(self.data.time - self.model.opt.timestep),
                        }
                    )
                segment_side = stance
                segment_time = self.model.opt.timestep
            if store and n % self.control_steps == 0:
                records.append(
                    {
                        "t": float(self.data.time),
                        "root": self.data.qpos[:7].tolist(),
                        "force_N": f.tolist(),
                        "clearance_m": clearance.tolist(),
                        "tilt_rad": tilt,
                        "ctrl": self.data.ctrl.tolist(),
                        "qpos": self.data.qpos.tolist(),
                        "torque": self.data.actuator_force.tolist(),
                        "nonfoot": bad_contact,
                    }
                )
            if bad_contact or tilt > 1.05 or self.data.qpos[2] < 0.065:
                nonfoot = bad_contact
                stop_reason = (
                    "nonfoot_floor_contact" if bad_contact else "excessive_tilt" if tilt > 1.05 else "low_root"
                )
                break
            if self.data.warning.number.sum() or not np.all(np.isfinite(self.data.qpos)):
                stop_reason = "numerical_warning_or_nonfinite_state"
                break
        if segment_side >= 0:
            segments.append({"stance": segment_side, "duration": segment_time, "end": float(self.data.time)})
        sustained = [s for s in segments if s["duration"] >= 0.04]
        alternate = sum((sustained[i]["stance"] != sustained[i - 1]["stance"] for i in range(1, len(sustained))))
        result = {
            "params": list(p),
            "requested_duration_s": duration,
            "duration": float(self.data.time),
            "dx": float(self.data.qpos[0] - start[0]),
            "dy": float(self.data.qpos[1] - start[1]),
            "max_tilt_deg": math.degrees(max_tilt),
            "minimum_height": float(min_height),
            "nonfoot_floor": nonfoot,
            "max_clearance_m": max_clearance.tolist(),
            "minimum_normal_weight_ratio": minimum_contact_ratio.tolist(),
            "unloaded_clear_time_s": free_intervals,
            "support_segments": sustained,
            "alternations": alternate,
            "peak_torque_Nm": max_torque.tolist(),
            "rms_torque_Nm": np.sqrt(rms / step_count).tolist(),
            "saturation_fraction": (sat / step_count).tolist(),
            "warning_count": int(self.data.warning.number.sum()),
            "cap": cap,
            "stop_reason": stop_reason,
            "self_contacts": self_contacts,
        }
        result["feedback"] = list(feedback) if feedback is not None else None
        result["tail_mode"] = tail_mode
        result["target_clipped_control_updates"] = clipped
        result["joint_limit_violations_rad"] = {
            self.model.joint(j).name: float(value) for j, value in enumerate(joint_violations) if value > 1e-09
        }
        result["terminated_early"] = bool(self.data.time < duration - self.model.opt.timestep)
        result["display_support_active"] = bool(self.data.eq_active[self.model.equality("display_support").id])
        result["external_force_max"] = external_force_max
        result["peak_joint_speed_rad_s"] = peak_speed.tolist()
        return (result, records)


def build_report(pilot: WalkingPilot, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe numerical inputs and assumptions alongside the observations."""
    return {
        "schema_version": 1,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_model_file": pilot.model_path.name,
        "source_sha256": pilot.source_sha256,
        "generator_source_sha256": pilot.generator_source_sha256,
        "source_path_base": "Input model directory; hashes include its XML assets.",
        "scope": (
            "Manual ideal-actuator stepping diagnostic, not walking certification or physical motor/thermal validation. "
            "Flat floor and nominal mass. Starts home; no external support/forces or root resets during a run. "
            "Torque-speed curves, transmission compliance, power supply limits and disturbances are not modeled."
        ),
        "feedback_scope": (
            "Exact MuJoCo torso orientation and noiseless body gyro are privileged feedback. Hardware would need "
            "a matching IMU orientation estimate and independently validated noise, latency and control behavior. "
            "This is not a deployable hardware controller. Contact and clearance measurements only score results."
        ),
        "controller_parameters": dict(zip(PARAMETER_NAMES, PARAMETERS, strict=True)),
        "feedback_gains": dict(zip(["roll", "pitch", "gyro"], FEEDBACK, strict=True)),
        "selection_search_seed": 635,
        "replay_randomness": "None: fixed parameters and home state, no random sampling.",
        "control_rate_Hz": CONTROL_RATE_HZ,
        "physics_timestep_s": float(pilot.model.opt.timestep),
        "mass_kg": float(pilot.model.body_mass.sum()),
        "actuator_names": [pilot.model.actuator(i).name for i in range(pilot.model.nu)],
        "target_ranges_rad": {
            pilot.model.actuator(i).name: pilot.model.actuator_ctrlrange[i].tolist() for i in range(pilot.model.nu)
        },
        "joint_ranges_rad": {
            pilot.model.joint(j).name: pilot.model.jnt_range[j].tolist()
            for j in range(pilot.model.njnt)
            if pilot.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        },
        "nominal_torque_ranges_Nm": {
            pilot.model.actuator(i).name: pilot.model.actuator_forcerange[i].tolist() for i in range(pilot.model.nu)
        },
        "support_event_definition": (
            "After 2s: stance normal force>half bodyweight, swing normal force<.05N, minimum swing sole height>2mm, "
            "torso tilt<.52rad, no nonfoot floor contact, sustained>=40ms. Counts side changes between qualifying "
            "segments. Positive forward displacement does not by itself constitute stepping."
        ),
        "stop_conditions": (
            "Nonfoot floor force>.1N, torso tilt>1.05rad, root height<.065m, numerical warning/nonfinite state, "
            "or requested duration. Internal contacts are reported but do not stop the run."
        ),
        "measurement_scope": (
            "Torques, speeds and clearance peaks cover the full run, including a fall. Contact and geometry data "
            "are from the last physics solve; integrated joint positions can be one timestep later. "
            "Actuator force and speed arrays follow actuator_names. Saturation means >=95% of the run's torque cap. "
            "Self-contact geometry is simplified and cannot validate detailed CAD interference."
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--caps", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--tail", action="store_true", help="Build the optional tail variant from --model")
    parser.add_argument("--tail-mode", choices=["none", "center", "wag"], default="none")
    parser.add_argument("--output", "--json", dest="output", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 2:
        parser.error("duration must be finite and greater than two seconds")
    if any((not math.isfinite(cap) or cap <= 0 for cap in args.caps)):
        parser.error("caps must be finite and positive")
    if args.trajectory and len(args.caps) != 1:
        parser.error("trajectory output requires exactly one cap")
    pilot = WalkingPilot(args.model, tail=args.tail)
    results = []
    for cap in args.caps:
        result, record = pilot.run(
            PARAMETERS, args.duration, store=bool(args.trajectory), cap=cap, feedback=FEEDBACK, tail_mode=args.tail_mode
        )
        results.append(result)
        if args.trajectory:
            args.trajectory.parent.mkdir(parents=True, exist_ok=True)
            args.trajectory.write_text(json.dumps(record))
    report = build_report(pilot, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            [
                {
                    "cap_Nm": r["cap"],
                    "duration_s": r["duration"],
                    "alternations": r["alternations"],
                    "forward_m": r["dx"],
                    "early_stop": r["terminated_early"],
                }
                for r in results
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
