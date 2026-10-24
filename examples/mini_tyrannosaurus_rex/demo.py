"""View, render or smoke-check the experimental Mini Tyrannosaurus Rex robot."""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).with_name("model.xml")


def set_support(model: mujoco.MjModel, data: mujoco.MjData, enabled: bool) -> None:
    """Enable an external torso weld; this is a display support, not balance control."""
    support = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "display_support")
    if support < 0:
        raise ValueError("The model has no display_support equality.")
    data.eq_active[support] = enabled
    mujoco.mj_forward(model, data)


def load_model(*, supported: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the adjacent XML at its zero-joint home pose."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    set_support(model, data, supported)
    return model, data


def gesture_controls(model: mujoco.MjModel, elapsed: float) -> np.ndarray:
    """Small, smooth head/jaw targets for an externally supported demonstration."""
    targets: np.ndarray = np.array(model.key_ctrl[0], copy=True)
    ramp = 0.5 - 0.5 * math.cos(math.pi * min(max(elapsed, 0.0), 1.0))
    phase = 2 * math.pi * elapsed / 5
    motions = {
        "neck_pitch_servo": -0.03 * math.sin(phase),
        "head_pitch_servo": 0.04 * math.sin(phase),
        "head_yaw_servo": 0.14 * math.sin(0.7 * phase),
        "head_roll_servo": 0.04 * math.sin(0.9 * phase),
        "jaw_servo": 0.05 * (1 - math.cos(phase)),
    }
    for name, angle in motions.items():
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        targets[actuator] = ramp * angle
    np.clip(targets, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1], out=targets)
    return targets


def _rollout(duration: float, *, supported: bool) -> dict[str, Any]:
    model, data = load_model(supported=supported)
    initial_height = float(data.qpos[2])
    maximum_tilt = maximum_torque = maximum_cap_fraction = 0.0
    finite = True
    for _ in range(math.ceil(duration / model.opt.timestep)):
        if supported:
            data.ctrl[:] = gesture_controls(model, data.time)
        mujoco.mj_step(model, data)
        finite = finite and all(
            np.isfinite(values).all() for values in (data.qpos, data.qvel, data.actuator_force, data.sensordata)
        )
        maximum_tilt = max(
            maximum_tilt,
            math.degrees(math.acos(float(np.clip(1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2), -1, 1)))),
        )
        maximum_torque = max(maximum_torque, float(np.max(np.abs(data.actuator_force))))
        maximum_cap_fraction = max(
            maximum_cap_fraction, float(np.max(np.abs(data.actuator_force) / model.actuator_forcerange[:, 1]))
        )
    return {
        "finite": bool(finite),
        "no_warnings": not bool(np.any(data.warning.number)),
        "final_height_ratio": float(data.qpos[2] / initial_height),
        "max_tilt_deg": maximum_tilt,
        "max_actuator_torque_Nm": maximum_torque,
        "max_torque_cap_fraction": maximum_cap_fraction,
    }


def check_model(*, duration: float = 5.0) -> dict[str, Any]:
    """Run numerical smoke checks, not hardware, walking or duty qualification."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    model, data = load_model()
    penetration = max((max(0.0, -float(c.dist)) for c in data.contact[: data.ncon]), default=0.0)
    report: dict[str, Any] = {
        "model": {"mass_kg": float(np.sum(model.body_mass)), "actuators": model.nu, "sensors": model.nsensor},
        "home": {"contact_count": data.ncon, "max_penetration_m": penetration},
        "free_hold": _rollout(duration, supported=False),
        "supported_gesture": _rollout(duration, supported=True),
        "scope": "Untrained concept; short ideal simulation smoke checks do not qualify walking, physical packaging, precision or continuous motor duty.",
    }
    runs = [report["free_hold"], report["supported_gesture"]]
    report["passed"] = (
        abs(report["model"]["mass_kg"] - 0.85) < 1e-9
        and model.nu == 15
        and model.nsensor == 39
        and penetration < 0.002
        and all(
            run["finite"]
            and run["no_warnings"]
            and run["final_height_ratio"] >= 0.8
            and run["max_tilt_deg"] < 15
            and run["max_torque_cap_fraction"] <= 1.000001
            for run in runs
        )
    )
    return report


def _save_png(path: Path, pixels: np.ndarray) -> None:
    """Write the renderer's RGB pixels without an optional imaging dependency."""
    height, width, _ = pixels.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Print short rollout checks as JSON.")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per check rollout.")
    parser.add_argument("--render", type=Path, metavar="PNG", help="Render the home pose to a PNG.")
    parser.add_argument("--camera", choices=["isometric", "side", "front"], default="isometric")
    parser.add_argument("--view", action="store_true", help="Open an interactive passive viewer.")
    parser.add_argument("--support", action="store_true", help="Weld the torso to the world for viewing/rendering.")
    parser.add_argument("--gesture", action="store_true", help="Gentle head/jaw motion; requires --view --support.")
    args = parser.parse_args()
    if args.gesture and not (args.view and args.support):
        parser.error("--gesture requires --view --support; the torso is externally held")
    if not any((args.check, args.render, args.view)):
        parser.print_help()
        return 0
    if args.check:
        try:
            report = check_model(duration=args.duration)
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            return 1
    if args.render or args.view:
        model, data = load_model(supported=args.support)
        if args.render:
            with mujoco.Renderer(model, height=650, width=800) as renderer:
                renderer.update_scene(data, camera=args.camera)
                _save_png(args.render, renderer.render())
            print(f"Rendered {args.render}")
        if args.view:
            from mujoco import viewer as mujoco_viewer

            with mujoco_viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    start = time.monotonic()
                    if args.gesture:
                        data.ctrl[:] = gesture_controls(model, data.time)
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - start)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
