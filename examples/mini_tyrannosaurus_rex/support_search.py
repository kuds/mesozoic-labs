"""Optional coupled single-support posture search, not a walking qualification.

Run from the repository root with ``python -m
examples.mini_tyrannosaurus_rex.support_search --output support.json``.
SciPy is needed only when this manual study is run; it is not a runtime dependency.

The stance sole is fixed flat by solving the floating-base transform. All direct
joint commands are searched, including the swing leg and linked shoulders.
Constraints inspect existing collision proxies and retain their exclusions.
Failure of a finite nonconvex search is not a proof that no posture or gait exists.

``--ankle-roll`` adds virtual coincident ankle axes with NO motor/mount mass,
hardware geometry, cable clearance or new sensing. It is a kinematic hypothesis.
``--foot-width-mm`` changes only the contact rectangle, without material mass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from examples.mini_tyrannosaurus_rex import engineering

MODEL_PATH = Path(__file__).with_name("model.xml")


def _virtual_ankles(root: ET.Element, original: mujoco.MjModel) -> None:
    """Add unqualified axes and retain original home coordinates by joint name."""
    for side in ("left", "right"):
        foot = root.find(f".//body[@name='{side}_foot']")
        actuators = root.find("actuator")
        if foot is None or actuators is None:
            raise ValueError("Expected the Mini Tyrannosaurus Rex feet and actuator block.")
        if root.find(f".//joint[@name='{side}_ankle_roll']") is not None:
            raise ValueError("The input already has an ankle-roll joint.")
        ET.SubElement(foot, "joint", name=f"{side}_ankle_roll", axis="1 0 0", range="-.45 .45")
        ET.SubElement(
            actuators,
            "position",
            name=f"{side}_ankle_roll_servo",
            joint=f"{side}_ankle_roll",
            ctrlrange="-.45 .45",
            forcerange="-.10 .10",
        )
    keyframes = root.find("keyframe")
    if keyframes is not None:
        root.remove(keyframes)
    expanded = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    qpos = expanded.qpos0.copy()
    old_positions = original.key_qpos[0] if original.nkey else original.qpos0
    widths = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
    original_joints = {original.joint(joint).name: joint for joint in range(original.njnt)}
    for joint in range(expanded.njnt):
        old = original_joints.get(expanded.joint(joint).name)
        if old is None:
            continue
        width = widths.get(expanded.jnt_type[joint], 1)
        before, after = original.jnt_qposadr[old], expanded.jnt_qposadr[joint]
        qpos[after : after + width] = old_positions[before : before + width]
    controls = np.zeros(expanded.nu)
    if original.nkey:
        old_controls = {original.actuator(i).name: original.key_ctrl[0, i] for i in range(original.nu)}
        for actuator in range(expanded.nu):
            controls[actuator] = old_controls.get(expanded.actuator(actuator).name, 0.0)
    ET.SubElement(
        ET.SubElement(root, "keyframe"),
        "key",
        name="home",
        qpos=" ".join(map(str, qpos)),
        ctrl=" ".join(map(str, controls)),
    )


def load_model(
    path: Path, *, ankle_roll: bool = False, foot_width_mm: float | None = None
) -> tuple[mujoco.MjModel, dict[str, str]]:
    """Load a derived study model without changing source files."""
    path = path.resolve()
    root = ET.parse(path).getroot()
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    for include in root.findall(".//include"):
        filename = include.attrib["file"]
        resolved = (path.parent / filename).resolve()
        hashes[filename] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        include.set("file", str(resolved))
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if not original.nkey:
        raise ValueError("The input model must define its home pose as keyframe zero.")
    if foot_width_mm is not None:
        if not math.isfinite(foot_width_mm) or not 10 <= foot_width_mm <= 100:
            raise ValueError("Contact width must be between 10 and 100 mm.")
        for side in ("left", "right"):
            pad = root.find(f".//geom[@name='{side}_flat_pad_collision']")
            if pad is None:
                raise ValueError("Expected rectangular Mini Tyrannosaurus Rex foot proxies.")
            size = np.fromstring(pad.attrib["size"], sep=" ")
            size[1] = foot_width_mm / 2000
            pad.set("size", " ".join(map(str, size)))
    if ankle_roll:
        _virtual_ankles(root, original)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), hashes


class SupportSearch:
    """Evaluate coupled, stationary, left-foot support postures."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = engineering.home_data(model)
        ranges = engineering.command_joint_ranges(model)
        self.names = list(ranges)
        self.bounds = list(ranges.values())
        self.left = model.site("left_sole_center").id
        self.right = model.site("right_sole_center").id
        self.root_address = int(model.jnt_qposadr[model.joint("root").id])

    def set_pose(self, values: np.ndarray) -> None:
        model, data = self.model, self.data
        mujoco.mj_resetDataKeyframe(model, data, 0)
        address = self.root_address
        data.qpos[address : address + 3] = 0.0
        data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
        engineering.set_pose(model, data, dict(zip(self.names, map(float, values), strict=True)))
        rotation = data.site_xmat[self.left].reshape(3, 3).T.copy()
        position = data.site_xpos[self.left].copy()
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        data.qpos[address + 3 : address + 7] = quaternion
        data.qpos[address : address + 3] = -rotation @ position
        mujoco.mj_forward(model, data)

    def measurements(self, values: np.ndarray, *, loads: bool = False) -> dict[str, Any]:
        self.set_pose(values)
        model, data = self.model, self.data
        com = data.subtree_com[model.body("torso").id].copy()
        half_x, half_y = model.geom_size[model.geom("right_flat_pad_collision").id, :2]
        corners = np.array([[a * half_x, b * half_y, 0.0] for a, b in itertools.product((-1, 1), repeat=2)])
        corners = corners @ data.site_xmat[self.right].reshape(3, 3).T + data.site_xpos[self.right]
        up = data.xmat[model.body("torso").id].reshape(3, 3)[2, 2]
        contacts: list[dict[str, Any]] = []
        for contact in data.contact[: data.ncon]:
            names = [model.geom(int(geom)).name for geom in contact.geom]
            if "floor" in names and "left_flat_pad_collision" in names:
                continue
            if contact.dist < -0.0001:
                contacts.append({"penetration_m": float(-contact.dist), "pair": names})
        result: dict[str, Any] = {
            "cop_m": com[:2].tolist(),
            "com_height_m": float(com[2]),
            "swing_clearance_m": float(corners[:, 2].min()),
            "torso_tilt_rad": math.acos(float(np.clip(up, -1.0, 1.0))),
            "root_height_m": float(data.qpos[self.root_address + 2]),
            "max_proxy_penetration_m": max((row["penetration_m"] for row in contacts), default=0.0),
            "contacts": contacts,
        }
        if loads:
            linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, linear, angular, self.left)
            force = np.array([0.0, 0.0, -model.opt.gravity[2] * model.body_mass.sum()])
            moment = np.array([force[2] * com[1], -force[2] * com[0], 0.0])
            torques = data.qfrc_bias - linear.T @ force - angular.T @ moment
            result["joint_torque_Nm"] = {
                model.joint(joint).name: float(torques[model.jnt_dofadr[joint]])
                for joint in range(model.njnt)
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
            }
        return result

    def objective(self, values: np.ndarray) -> float:
        result = self.measurements(values)
        cop_x, cop_y = result["cop_m"]
        half_x = self.model.geom_size[self.model.geom("left_flat_pad_collision").id, 0]
        penalty = (
            max(abs(cop_x) - (half_x - 0.002), 0.0) ** 2
            + max(0.003 - result["swing_clearance_m"], 0.0) ** 2
            + 0.01 * max(result["torso_tilt_rad"] - 0.61, 0.0) ** 2
            + max(0.075 - result["root_height_m"], 0.0) ** 2
            + result["max_proxy_penetration_m"] ** 2
        )
        # Maximize inward-to-outward lateral reach. Success must be checked
        # separately; a positive lateral result can overshoot the other edge.
        return float(-cop_y + 10000 * penalty)

    def record(self, values: np.ndarray) -> dict[str, Any]:
        result = self.measurements(values, loads=True)
        half = self.model.geom_size[self.model.geom("left_flat_pad_collision").id, :2]
        result.update(
            pose_rad=dict(zip(self.names, map(float, values), strict=True)),
            qpos=self.data.qpos.tolist(),
            inside_sole=bool(np.all(np.abs(result["cop_m"]) <= half)),
            contact_half_extents_m=half.tolist(),
        )
        pairs = list(itertools.combinations(engineering.motor_cases(self.model), 2))
        if pairs:
            gap, first, second = min((engineering.box_separation(self.model, self.data, a, b), a, b) for a, b in pairs)
            result["motor_box_gap_m"] = gap
            result["closest_motor_pair"] = [self.model.geom(first).name, self.model.geom(second).name]
        return result


def refine(
    search: SupportSearch, start: np.ndarray, optimizer: Any, *, starts: int, iterations: int, seed: int
) -> list[dict[str, Any]]:
    """Local torque minimization with a 25.8-degree torso-tilt limit."""
    leg_names = [name for name in search.names if any(part in name for part in ("hip_", "knee", "ankle"))]

    def objective(values: np.ndarray) -> float:
        return float(values[-1] + 0.002 * np.mean(values[:-1] ** 2))

    def constraints(values: np.ndarray) -> np.ndarray:
        result = search.measurements(values[:-1], loads=True)
        x, y = result["cop_m"]
        half_x, half_y = search.model.geom_size[search.model.geom("left_flat_pad_collision").id, :2]
        bounds = [
            half_x - 0.003 - x,
            half_x - 0.003 + x,
            half_y - 0.003 - y,
            half_y - 0.003 + y,
            result["swing_clearance_m"] - 0.005,
            0.45 - result["torso_tilt_rad"],
            result["root_height_m"] - 0.075,
            0.135 - result["root_height_m"],
            0.000001 - result["max_proxy_penetration_m"],
        ]
        for name in leg_names:
            torque = result["joint_torque_Nm"][name]
            bounds.extend((values[-1] - torque, values[-1] + torque))
        constraint_values: np.ndarray = np.asarray(bounds, dtype=float)
        return constraint_values

    rng, reports = np.random.default_rng(seed), []
    joint_bounds = np.asarray(search.bounds)
    for attempt in range(starts):
        initial = np.r_[start, 0.15]
        if attempt:
            initial[:-1] = np.clip(
                initial[:-1] + rng.normal(0.0, 0.05, len(start)), joint_bounds[:, 0], joint_bounds[:, 1]
            )
        result = optimizer.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=search.bounds + [(0.001, 0.4)],
            constraints=[{"type": "ineq", "fun": constraints}],
            options={"maxiter": iterations, "ftol": 1e-10},
        )
        report = search.record(result.x[:-1])
        report.update(
            optimizer_converged=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            evaluations=int(result.nfev),
            peak_leg_load_objective_Nm=float(result.x[-1]),
            minimum_constraint=float(constraints(result.x).min()),
        )
        reports.append(report)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ankle-roll", action="store_true", help="Virtual massless/unpackaged additional ankle axes.")
    parser.add_argument("--foot-width-mm", type=float, help="Contact width only; no added foot material mass.")
    parser.add_argument("--iterations", type=int, default=120, help="Differential-evolution generations, 1–500.")
    parser.add_argument("--population", type=int, default=8, help="Population multiplier per variable, 3–20.")
    parser.add_argument("--seed", type=int, default=635)
    parser.add_argument("--refine-starts", type=int, default=0, help="Optional SLSQP restarts, 0–10.")
    parser.add_argument("--refine-iterations", type=int, default=100, help="Iterations per SLSQP restart, 1–300.")
    parser.add_argument("--refine-seed", type=int, default=700)
    args = parser.parse_args()
    for name, low, high in (
        ("iterations", 1, 500),
        ("population", 3, 20),
        ("seed", 0, 2**32 - 1),
        ("refine_starts", 0, 10),
        ("refine_iterations", 1, 300),
        ("refine_seed", 0, 2**32 - 1),
    ):
        if not low <= getattr(args, name) <= high:
            parser.error(f"--{name.replace('_', '-')} must be between {low} and {high}.")
    try:
        optimizer = importlib.import_module("scipy.optimize")
    except ImportError:
        parser.error("This optional manual study requires SciPy; install the project's test extra or scipy.")
    try:
        model, hashes = load_model(args.model, ankle_roll=args.ankle_roll, foot_width_mm=args.foot_width_mm)
    except (OSError, ValueError, ET.ParseError) as error:
        parser.error(str(error))
    search = SupportSearch(model)
    result = optimizer.differential_evolution(
        search.objective,
        search.bounds,
        seed=args.seed,
        popsize=args.population,
        maxiter=args.iterations,
        tol=1e-7,
        polish=True,
        workers=1,
        updating="immediate",
    )
    report = {
        "scope": "Finite static posture search; not proof of global infeasibility, a gait, physical fit or thermal duty.",
        "source_sha256": hashes,
        "mujoco_version": mujoco.__version__,
        "scipy_version": importlib.import_module("scipy").__version__,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "mass_kg": float(model.body_mass.sum()),
        "virtual_ankle_caveat": "No added motor/mount mass or hardware geometry; existing proxy exclusions retained.",
        "foot_width_caveat": "Only contact rectangles change; no material mass or structural qualification.",
        "gravity_load_caveat": "Out-of-sole CoP requires an impossible contact wrench; those loads are not achievable support loads.",
        "search": {
            **search.record(result.x),
            "optimizer_converged": bool(result.success),
            "message": str(result.message),
            "evaluations": int(result.nfev),
            "iterations": int(result.nit),
            "objective": float(result.fun),
        },
        "refinement": refine(
            search,
            result.x,
            optimizer,
            starts=args.refine_starts,
            iterations=args.refine_iterations,
            seed=args.refine_seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}; optimizer convergence does not certify walking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
