"""Reporting-only stance diagnostics shared by training species.

The helpers in this module derive support and pose measurements without
changing actions, observations, rewards, reset behavior, or termination.
Training callbacks store rollout means; evaluation replays can additionally
write detailed per-frame geometry to CSV.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Raw environment measurements and reporting-only values derived from them.
# Reward-shaped qualities deliberately live outside this module.
STANCE_INFO_KEYS: tuple[str, ...] = (
    "r_foot_contact_duty",
    "l_foot_contact_duty",
    "bilateral_support_duty",
    "single_support_duty",
    "unsupported_duty",
    "r_foot_load_share",
    "l_foot_load_share",
    "foot_load_imbalance",
    "foot_load_asymmetry",
    "foot_load_balance",
    "leg_home_pose_error",
    "head_tip_z",
    "head_pelvis_rel_z",
    "neck_posture_error",
    "height_error",
)

# ``bite_success`` already exists on every T. rex reward-info step and
# distinguishes its biped contacts from similarly named quadruped front-foot
# contacts without adding a new environment signal.
STANCE_MARKER_KEYS: tuple[str, ...] = (
    "unsupported_duty",  # Explicit biped instrumentation (including Compsognathus).
    "bite_success",
    "leg_home_pose_error",
    "head_pelvis_rel_z",
    "neck_posture_error",
)

_REPLAY_INFO_KEYS: tuple[str, ...] = (
    "r_foot_contact",
    "l_foot_contact",
    "pelvis_height",
    "tilt_angle",
    "forward_z",
    "drift_distance",
    "pelvis_angular_vel",
    "pelvis_yaw_vel",
    "heading_alignment",
    *STANCE_INFO_KEYS,
)

_LEG_JOINTS: tuple[tuple[str, str], ...] = (
    ("hip_pitch", "hip_pitch"),
    ("hip_roll", "hip_roll"),
    ("knee", "knee"),
    ("ankle", "ankle"),
)


def has_stance_diagnostics(info: Mapping[str, Any]) -> bool:
    """Return whether *info* carries explicit biped stance instrumentation."""
    return any(key in info for key in STANCE_MARKER_KEYS)


def derive_stance_info(info: Mapping[str, Any], contact_threshold: float = 0.1) -> dict[str, float]:
    """Derive instantaneous support and load metrics from foot forces.

    The returned ``*_duty`` fields are binary per timestep. Averaging them in
    the training callback or episode metrics produces conventional duty
    fractions. Existing environment-provided values remain authoritative when
    callers merge the result with ``setdefault``.
    """
    if "r_foot_contact" not in info or "l_foot_contact" not in info:
        return {}

    right_force = max(float(info["r_foot_contact"]), 0.0)
    left_force = max(float(info["l_foot_contact"]), 0.0)
    right_supported = right_force > contact_threshold
    left_supported = left_force > contact_threshold
    total_force = right_force + left_force

    if right_supported or left_supported:
        right_share = right_force / total_force if total_force > 1e-8 else 0.0
        left_share = left_force / total_force if total_force > 1e-8 else 0.0
        imbalance = abs(right_force - left_force) / total_force if total_force > 1e-8 else 1.0
        balance = 1.0 - imbalance
    else:
        # UNSUPPORTED is maximally imbalanced, not perfectly balanced.  The
        # previous `total_force > 1e-8` branch reported imbalance 0.0 for a
        # truly airborne pair -- i.e. PERFECT balance for an animal in the air
        # -- and, because two touch readings essentially never sum to exactly
        # zero, the else-branch never fired anyway (PLANT_VALIDATION §11.1).
        # Keying on the same `> contact_threshold` test the duty metrics use
        # makes the diagnostic agree with them: a foot at 0.001 N is now
        # unsupported here exactly as it is in *_contact_duty, instead of
        # counting as supported in one metric and not the other.
        right_share = 0.0
        left_share = 0.0
        imbalance = 1.0
        balance = 0.0

    return {
        "r_foot_contact_duty": float(right_supported),
        "l_foot_contact_duty": float(left_supported),
        "bilateral_support_duty": float(right_supported and left_supported),
        "single_support_duty": float(right_supported != left_supported),
        "unsupported_duty": float(not right_supported and not left_supported),
        "r_foot_load_share": float(right_share),
        "l_foot_load_share": float(left_share),
        "foot_load_imbalance": float(imbalance),
        "foot_load_asymmetry": float(imbalance),
        "foot_load_balance": float(balance),
    }


def _scalar(value: Any) -> float | None:
    """Return a finite scalar float, or ``None`` for arrays/non-numbers."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quat_to_euler(quat: Any) -> tuple[float, float, float]:
    """Convert a MuJoCo ``(w, x, y, z)`` quaternion to XYZ Euler angles."""
    w, x, y, z = (float(value) for value in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _named_id(model: Any, object_type: Any, name: str) -> int:
    import mujoco

    return int(mujoco.mj_name2id(model, object_type, name))


def _site_position(env: Any, name: str) -> np.ndarray | None:
    try:
        import mujoco

        site_id = _named_id(env.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return np.asarray(env.data.site_xpos[site_id], dtype=float)
    except (AttributeError, ImportError, IndexError, TypeError, ValueError):
        pass
    return None


def _body_pose(env: Any, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import mujoco

        body_id = _named_id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return (
                np.asarray(env.data.xpos[body_id], dtype=float),
                np.asarray(env.data.xquat[body_id], dtype=float),
            )
    except (AttributeError, ImportError, IndexError, TypeError, ValueError):
        pass
    return None


def _home_joint_error(env: Any, joint_name: str) -> float | None:
    try:
        import mujoco

        joint_id = _named_id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return None
        qpos_index = int(env.model.jnt_qposadr[joint_id])
        home_id = getattr(env, "home_keyframe_id", None)
        if home_id is None:
            home_id = _named_id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if int(home_id) < 0:
            return None
        return abs(float(env.data.qpos[qpos_index] - env.model.key_qpos[int(home_id), qpos_index]))
    except (AttributeError, ImportError, IndexError, TypeError, ValueError):
        return None


# Each column side's (foot site, leg-joint prefix) in the T. rex model, which
# the historical columns were measured from.
_TREX_SIDE_NAMES: dict[str, tuple[str, str]] = {"r": ("r_foot", "r"), "l": ("l_foot", "l")}


def _biped_side_names(species: str | None) -> dict[str, tuple[str, str]]:
    """Each column side's ``(foot site, leg-joint prefix)`` in *species*' model.

    The foot site is the gait support registry's ``foot_site``, the reference
    point the floor-truth recorder measures stance width and foot shift from
    (and the one the T. rex and anatomical Compsognathus stance-width rewards
    read). Its name prefix is how the model spells that side, which its leg
    joints share: ``r_foot_touch_volume`` / ``r_knee``,
    ``right_foot_touch_volume`` / ``right_knee``. A name without a registry
    entry keeps the T. rex names.
    """
    from .gait.morphology import SUPPORT_REGISTRY
    from .species_names import resolve_species_id

    try:
        entry = SUPPORT_REGISTRY.get(resolve_species_id(species)) if species else None
    except ValueError:
        entry = None
    if entry is None or entry.foot_site is None:
        return dict(_TREX_SIDE_NAMES)
    names = {}
    for side, word in (("r", "right"), ("l", "left")):
        site = entry.foot_site.format(s=side, side=word)
        names[side] = (site, site.split("_", 1)[0])
    return names


def capture_trex_stance_snapshot(env: Any, info: Mapping[str, Any], step: int) -> dict[str, float]:
    """Capture one reporting-only T. rex stance row from a replay environment."""
    return _capture_biped_snapshot(env, info, step, _TREX_SIDE_NAMES)


def _capture_biped_snapshot(
    env: Any, info: Mapping[str, Any], step: int, side_names: Mapping[str, tuple[str, str]]
) -> dict[str, float]:
    """One biped stance row, with each side's foot site and joint prefix from *side_names*."""
    row: dict[str, float] = {"step": float(step)}
    for key in _REPLAY_INFO_KEYS:
        if key in info:
            value = _scalar(info[key])
            if value is not None:
                row[key] = value
    for key, value in derive_stance_info(info).items():
        row.setdefault(key, value)

    side_errors: dict[str, list[float]] = {"r": [], "l": []}
    joint_errors: dict[str, list[float]] = {label: [] for label, _ in _LEG_JOINTS}
    for side in ("r", "l"):
        joint_prefix = side_names[side][1]
        for label, joint_suffix in _LEG_JOINTS:
            error = _home_joint_error(env, f"{joint_prefix}_{joint_suffix}")
            if error is None:
                continue
            row[f"{side}_{label}_home_error_rad"] = error
            row[f"{side}_{label}_home_error_deg"] = math.degrees(error)
            side_errors[side].append(error)
            joint_errors[label].append(error)

    for side, values in side_errors.items():
        if values:
            row[f"{side}_leg_home_error_rad"] = float(np.mean(values))
            row[f"{side}_leg_home_error_deg"] = math.degrees(row[f"{side}_leg_home_error_rad"])
    if side_errors["r"] and side_errors["l"]:
        row["leg_home_error_asymmetry_rad"] = abs(row["r_leg_home_error_rad"] - row["l_leg_home_error_rad"])
        row["leg_home_error_asymmetry_deg"] = math.degrees(row["leg_home_error_asymmetry_rad"])
    for label, values in joint_errors.items():
        if values:
            row[f"{label}_home_error_rad"] = float(np.mean(values))
            row[f"{label}_home_error_deg"] = math.degrees(row[f"{label}_home_error_rad"])

    pelvis_pose = _body_pose(env, "pelvis")
    pelvis_position: np.ndarray | None = None
    pelvis_z: float | None = None
    if pelvis_pose is not None:
        position, quaternion = pelvis_pose
        pelvis_position = position
        row["pelvis_x"], row["pelvis_y"], row["pelvis_z"] = (float(value) for value in position)
        pelvis_z = row["pelvis_z"]
        row.setdefault("pelvis_height", pelvis_z)
        roll, pitch, yaw = _quat_to_euler(quaternion)
        for name, value in (("roll", roll), ("pitch", pitch), ("yaw", yaw)):
            row[f"pelvis_{name}_rad"] = value
            row[f"pelvis_{name}_deg"] = math.degrees(value)

    foot_positions: dict[str, np.ndarray] = {}
    for side in ("r", "l"):
        foot_position = _site_position(env, side_names[side][0])
        if foot_position is None:
            continue
        foot_positions[side] = foot_position
        row[f"{side}_foot_x"], row[f"{side}_foot_y"], row[f"{side}_foot_z"] = (float(value) for value in foot_position)

    if "r" in foot_positions and "l" in foot_positions:
        right_foot = foot_positions["r"]
        left_foot = foot_positions["l"]
        fore_aft_offset = float(right_foot[0] - left_foot[0])
        support_midpoint = (right_foot + left_foot) / 2.0
        row["stance_width"] = abs(float(right_foot[1] - left_foot[1]))
        row["foot_fore_aft_offset"] = fore_aft_offset
        row["foot_fore_aft_separation"] = abs(fore_aft_offset)
        row["support_midpoint_x"], row["support_midpoint_y"], row["support_midpoint_z"] = (
            float(value) for value in support_midpoint
        )
        if pelvis_position is not None:
            pelvis_support_offset = pelvis_position - support_midpoint
            row["pelvis_support_offset_x"], row["pelvis_support_offset_y"], row["pelvis_support_offset_z"] = (
                float(value) for value in pelvis_support_offset
            )
            row["pelvis_support_offset_xy"] = float(np.linalg.norm(pelvis_support_offset[:2]))

    head_tip = _site_position(env, "head_tip")
    if head_tip is not None:
        row["head_tip_x"], row["head_tip_y"], row["head_tip_z"] = (float(value) for value in head_tip)
        row["head_tip_clearance"] = row["head_tip_z"]
        if pelvis_z is not None:
            row["head_pelvis_rel_z"] = row["head_tip_z"] - pelvis_z

    skull_pose = _body_pose(env, "skull")
    if skull_pose is not None:
        skull_position, skull_quaternion = skull_pose
        row["skull_x"], row["skull_y"], row["skull_z"] = (float(value) for value in skull_position)
        row["skull_clearance"] = row["skull_z"]
        if pelvis_z is not None:
            row["skull_pelvis_rel_z"] = row["skull_z"] - pelvis_z
        _, skull_pitch, _ = _quat_to_euler(skull_quaternion)
        row["skull_pitch_rad"] = skull_pitch
        row["skull_pitch_deg"] = math.degrees(skull_pitch)

    tail_tip = _site_position(env, "tail_tip")
    if tail_tip is not None:
        row["tail_tip_z"] = float(tail_tip[2])
        if pelvis_z is not None:
            row["tail_pelvis_rel_z"] = row["tail_tip_z"] - pelvis_z

    return row


def capture_stance_snapshot(
    env: Any, info: Mapping[str, Any], step: int, *, species: str | None = None
) -> dict[str, float]:
    """Record common root pose and measured info, with optional biped geometry.

    Species expose different anatomy and instrumentation. Missing measurements
    stay absent; in particular, a quadruped's front-foot pair is not treated as
    its complete support. Instrumented bipeds get the historical T. rex
    columns, without changing their simulation state, measured at *species*'
    own foot sites and leg joints (:func:`_biped_side_names`). The derived
    ``stance_width`` (``|dy|`` of the foot sites) replaces an env's own.
    """
    row: dict[str, float] = {"step": float(step)}
    for key, value in info.items():
        scalar = _scalar(value)
        if scalar is not None:
            row[key] = scalar
    if has_stance_diagnostics(info):
        row.update(_capture_biped_snapshot(env, info, step, _biped_side_names(species)))

    root_id = getattr(env, "_root_body_id", None)
    if isinstance(root_id, (int, np.integer)) and root_id >= 0:
        position = env.data.xpos[root_id]
        row.update({f"root_{axis}": float(value) for axis, value in zip("xyz", position)})
        roll, pitch, yaw = _quat_to_euler(env.data.xquat[root_id])
        for name, value in (("roll", roll), ("pitch", pitch), ("yaw", yaw)):
            row[f"root_{name}_rad"] = value
            row[f"root_{name}_deg"] = math.degrees(value)
    return row


def write_stance_diagnostics_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    """Write per-frame stance diagnostics, preserving every observed column."""
    if not rows:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step"]
    fieldnames.extend(sorted({key for row in rows for key in row if key != "step"}))
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path
