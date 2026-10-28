"""Render the candidate's actual unsupported gesture simulation.

Run ``python -m examples.mini_tyrannosaurus_rex.juvenile.animate`` after building
the candidate. Twenty sampled poses form a ten-second fixed-camera preview;
the tail remains fixed and no walking policy is implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared import robot_rendering
from environments.shared.robot_validation import GroundSupportProbe, require_unsupported
from examples.mini_tyrannosaurus_rex.demo import gesture_controls

HERE = Path(__file__).resolve().parent
DIRECTION = (1.0, -1.35, 0.65)
DURATION = 10.0
FRAMES = 20


def _reset(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    require_unsupported(model, data)
    mujoco.mj_forward(model, data)
    return data


def animate(model_path: Path = HERE / "model.xml", output: Path = HERE / "motion.gif") -> dict[str, Any]:
    """Check the complete rollout, then render its sampled actual joint poses."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.body_jntnum[model.body("tail").id] != 0:
        raise ValueError("This preview is labeled fixed tail and requires the fixed-tail candidate.")
    data = _reset(model)
    support = GroundSupportProbe(model, feet={side: (f"{side}_flat_pad_collision",) for side in ("left", "right")})
    penetration = unexpected_support = 0.0
    worst_contact: tuple[str, str] | None = None
    for _ in range(round(DURATION / model.opt.timestep)):
        data.ctrl[:] = gesture_controls(model, float(data.time))
        mujoco.mj_step(model, data)
        if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.actuator_force)):
            raise RuntimeError("Gesture simulation produced non-finite state.")
        if np.any(data.warning.number) or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise RuntimeError("Gesture simulation has warnings or applied external forces.")
        unexpected_support = max(unexpected_support, support.sample(data)[1])
        for contact in data.contact[: data.ncon]:
            pair = tuple(model.geom(int(geom)).name for geom in contact.geom)
            if "floor" not in pair and -contact.dist > penetration:
                penetration = float(-contact.dist)
                worst_contact = pair
    if unexpected_support > 1e-5 or penetration >= 0.0001:
        raise RuntimeError(f"Unexpected support/contact: {unexpected_support} N, {penetration} m, {worst_contact}")

    def renderer(current_model: mujoco.MjModel, current_data: mujoco.MjData, **kwargs: Any):
        return robot_rendering.render(current_model, current_data, direction=DIRECTION, **kwargs)

    robot_rendering.gesture_gif(
        model,
        _reset(model),
        output,
        controls=gesture_controls,
        renderer=renderer,
        direction=DIRECTION,
        title="MINI / ACTUAL MUJOCO GESTURE",
        footer=lambda timestamp: f"{timestamp:04.1f} s  |  Unsupported stance  |  Fixed tail",
        frames=FRAMES,
        duration=DURATION,
    )
    return {
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "animation": str(output),
        "bytes": output.stat().st_size,
        "frames": FRAMES,
        "playback_duration_s": DURATION,
        "pose_sample_times_s": np.linspace(0, DURATION, FRAMES, endpoint=False).tolist(),
        "full_rollout_checked_s": float(data.time),
        "externally_supported": False,
        "applied_external_forces": False,
        "tail_fixed": True,
        "maximum_internal_penetration_m": penetration,
        "maximum_unexpected_support_N": unexpected_support,
        "worst_internal_contact": worst_contact,
        "scope": "Scripted expression targets with free-root dynamics; no walking, tail actuation or hardware qualification.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "model.xml")
    parser.add_argument("--output", type=Path, default=HERE / "motion.gif")
    args = parser.parse_args()
    print(json.dumps(animate(args.model, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
