"""Silhouette-C shell separation and actual expression-motion regressions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared import robot_validation
from environments.shared.robot_validation import GroundSupportProbe, require_unsupported

from . import demo, engineering, tail_option


def mesh_vertices(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    geom = model.geom(name).id
    mesh = model.geom_dataid[geom]
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError(f"Expected a mesh: {name}")
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start : start + count]
    return np.asarray(vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])


def jaw_separation(model: mujoco.MjModel, points: int = 101) -> dict[str, Any]:
    """Conservative positive separation bounds, independent of contact filters.

    Project the compiled vertex clouds onto a fixed set of unit directions.
    A positive gap on any direction proves those convex hulls are separated;
    failure to find a gap is inconclusive and fails this regression. This avoids
    treating an unconverged zero from a convex distance query as clearance.
    """
    if points < 2:
        raise ValueError("Use at least two jaw positions")
    directions = np.array([(x, y, 1) for x, y in itertools.product(np.linspace(-0.6, 0.6, 13), repeat=2)])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    data = engineering.home_data(model)
    names = ("jaw_collision", "left_jaw_rail_collision", "right_jaw_rail_collision")
    minima = dict.fromkeys(names, math.inf)
    low, high = model.joint("jaw").range
    for angle in np.linspace(float(low), float(high), points):
        engineering.set_pose(model, data, {"jaw": float(angle)})
        skull = mesh_vertices(model, data, "head_collision") @ directions.T
        for name in names:
            jaw = mesh_vertices(model, data, name) @ directions.T
            gap = float(np.max(skull.min(axis=0) - jaw.max(axis=0)))
            minima[name] = min(minima[name], gap)
    return {
        "poses": points,
        "minimum_separating_axis_gaps_m": minima,
        "scope": "Compiled skull/jaw convex hulls only; common yaw/nod preserves their relative separation. Horns and cables excluded.",
    }


def gesture_trial(model: mujoco.MjModel, *, duration: float = 10, supported: bool = False) -> dict[str, Any]:
    if not math.isfinite(duration) or duration < 3:
        raise ValueError("Use a finite gesture duration of at least three seconds")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if supported:
        demo.set_support(model, data, True)
    else:
        require_unsupported(model, data)
        mujoco.mj_forward(model, data)
    support = GroundSupportProbe(model, feet={side: (f"{side}_flat_pad_collision",) for side in ("left", "right")})
    names = ("head_yaw", "head_pitch", "jaw", "left_shoulder_pitch", "right_shoulder_pitch")
    addresses = [model.joint(name).qposadr[0] for name in names]
    positions, forces = [], []
    minimum_height, maximum_tilt, penetration = 1.0, 0.0, 0.0
    start_height = float(data.qpos[2])
    finite = True
    unexpected_support, external_forces = 0.0, False
    for _ in range(round(duration / model.opt.timestep)):
        data.ctrl[:] = demo.gesture_controls(model, float(data.time))
        mujoco.mj_step(model, data)
        finite = finite and all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.actuator_force))
        if not finite or np.any(data.warning.number):
            break
        unexpected_support = max(unexpected_support, support.sample(data)[1])
        external_forces = external_forces or bool(np.any(data.xfrc_applied) or np.any(data.qfrc_applied))
        positions.append(data.qpos[addresses].copy())
        forces.append(data.actuator_force.copy())
        minimum_height = min(minimum_height, float(data.qpos[2] / start_height))
        up = data.xmat[model.body("torso").id].reshape(3, 3)[2, 2]
        maximum_tilt = max(maximum_tilt, math.degrees(math.acos(float(np.clip(up, -1, 1)))))
        for contact in data.contact[: data.ncon]:
            if all(model.geom(int(g)).name != "floor" for g in contact.geom):
                penetration = max(penetration, float(-contact.dist))
    spans = np.ptp(np.array(positions), axis=0) if positions else np.zeros(len(names))
    efforts = np.array(forces) if forces else np.zeros((1, model.nu))
    return {
        "duration_s": float(data.time),
        "externally_supported": supported,
        "finite": finite,
        "warnings": int(data.warning.number.sum()),
        "maximum_unexpected_support_N": unexpected_support,
        "no_applied_external_forces": not external_forces,
        "minimum_height_ratio": minimum_height,
        "maximum_tilt_deg": maximum_tilt,
        "maximum_internal_penetration_m": penetration,
        "joint_excursions_rad": dict(zip(names, map(float, spans), strict=True)),
        "actuators": {
            model.actuator(i).name: {
                "peak_Nm": float(np.max(np.abs(efforts[:, i]))),
                "rms_Nm": float(np.sqrt(np.mean(efforts[:, i] ** 2))),
                "cap_Nm": model.actuator_forcerange[i].tolist(),
            }
            for i in range(model.nu)
        },
    }


def build_report() -> dict[str, Any]:
    base, _ = demo.load_model()
    tail = tail_option.load_model()
    root = Path(__file__).parent
    files = [root / name for name in ("model.xml", "head_allocation.json", "components.json")]
    files += sorted(root.glob("*.py")) + sorted((root / "assets").glob("*.xml"))
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in ("robot_validation.py", "robot_rendering.py"):
        sources[f"../../environments/shared/{name}"] = hashlib.sha256(
            Path(robot_validation.__file__).with_name(name).read_bytes()
        ).hexdigest()
    return {
        "design": "silhouette_c_mini",
        "mujoco_version": mujoco.__version__,
        "scope": "Short ideal simulation regressions; not walking, physical fit, electrical or thermal qualification.",
        "source_sha256": sources,
        "jaw_separation": jaw_separation(base),
        "gestures": {
            name: gesture_trial(model, supported=supported)
            for name, model, supported in (
                ("free_base", base, False),
                ("supported_base", base, True),
                ("free_tail", tail, False),
                ("supported_tail", tail, True),
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("character_validation.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_report(), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
