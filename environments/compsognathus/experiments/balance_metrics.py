"""Research-only balance measurements and behavior-first checkpoint selection.

The canonical stance criteria are projected onto the *base* reward, before study
shaping. Passing this projection does not advance the production curriculum.
Supplemental targets are proposed study settings, not validated species limits.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence, cast

import mujoco
import numpy as np

from environments.shared.curriculum.stance_gate import (
    StanceGateThresholds,
    evaluate_stance_gate,
    stance_panel_from_episode_duties,
)

from .foot_probe import FootProbe


@dataclass(frozen=True)
class BalanceTargets:
    """Proposed post-settling means; calibrate against saved policies and home."""

    min_bilateral_20pct_weight: float = 0.95
    max_pelvis_angular_velocity_rms: float = 0.1
    max_com_vertical_std_over_home_height: float = 0.02
    max_sole_pitch_motion_rms_rad: float = 0.03
    min_eval_episodes: int = 40

    def __post_init__(self):
        if not 0 <= self.min_bilateral_20pct_weight <= 1:
            raise ValueError("bilateral duty target must lie in [0, 1]")
        for name in (
            "max_pelvis_angular_velocity_rms",
            "max_com_vertical_std_over_home_height",
            "max_sole_pitch_motion_rms_rad",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.min_eval_episodes, bool)
            or not isinstance(self.min_eval_episodes, int)
            or self.min_eval_episodes < 40
        ):
            raise ValueError("behavior qualification requires at least 40 episodes")


# Mirrors configs/compsognathus/stance.toml. Tests guard these reference numbers.
REFERENCE_STANCE_THRESHOLDS = StanceGateThresholds(
    min_full_horizon_fraction=0.95,
    max_unsupported_duty=0.02,
    max_unsupported_duty_ucb=0.02,
    settle_steps=200,
    min_eval_episodes=40,
    min_avg_reward=1800.0,
    required_consecutive=3,
)


def _support_sample(loads: np.ndarray, weight: float) -> tuple[float, float]:
    """Weaker-foot load/BW and imbalance; an airborne sample is imbalanced."""
    total = float(np.sum(loads))
    return float(np.min(loads) / weight), float(abs(loads[0] - loads[1]) / total) if total > 0.1 else 1.0


class BalanceProbe(FootProbe):
    """Add physical quietness measures without changing the shared foot probe."""

    def __init__(self, model, data, home_height: float):
        super().__init__(model, data)
        self.home_height = home_height
        self.pelvis = model.body("pelvis").id
        self.weaker_load = self.imbalance = 0.0
        self.com_positions: list[np.ndarray] = []
        self.pelvis_heights: list[float] = []
        self.foot_origin: np.ndarray | None = None
        self.max_foot_drift = np.zeros(2)
        self.force_cap_samples = np.zeros(model.nu)
        self.joint_hard_stop_samples = 0

    def sample(self, *, record=True):
        loads = super().sample(record=record)
        if record:
            weaker, imbalance = _support_sample(loads, self.weight)
            self.weaker_load += weaker
            self.imbalance += imbalance
            self.com_positions.append(self.data.subtree_com[self.pelvis].copy())
            self.pelvis_heights.append(float(self.data.xpos[self.pelvis, 2]))
            feet = self.data.geom_xpos[self.pads, :2]
            if self.foot_origin is None:
                self.foot_origin = feet.copy()
            self.max_foot_drift = np.maximum(self.max_foot_drift, np.linalg.norm(feet - self.foot_origin, axis=1))
            force = self.data.actuator_force
            lower, upper = self.model.actuator_forcerange.T
            self.force_cap_samples += self.model.actuator_forcelimited.astype(bool) & (
                ((upper > 0) & (force >= 0.99 * upper)) | ((lower < 0) & (force <= 0.99 * lower))
            )
            self.joint_hard_stop_samples += int(
                np.any(
                    (self.data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
                    & (np.abs(self.data.efc_force) > 1e-8)
                )
            )
        return loads

    def results(self):
        result = super().results()
        com = np.asarray(self.com_positions)
        pitch = np.asarray(self.pitch)
        n = self.n
        result.update(
            weaker_foot_load_bw_mean=self.weaker_load / n if n else None,
            physics_load_imbalance_mean=self.imbalance / n if n else None,
            com_height_mean_m=float(com[:, 2].mean()) if n else None,
            com_vertical_std_over_home_height=float(com[:, 2].std() / self.home_height) if n else None,
            com_xy_sway_rms_over_home_height=(
                float(np.sqrt(np.mean(np.sum((com[:, :2] - com[:, :2].mean(axis=0)) ** 2, axis=1))) / self.home_height)
                if n
                else None
            ),
            pelvis_height_mean_m=float(np.mean(self.pelvis_heights)) if n else None,
            sole_pitch_rms_rad=float(np.sqrt(np.mean(pitch**2))) if n else None,
            sole_pitch_motion_rms_rad=float(np.sqrt(np.mean((pitch - pitch.mean(axis=0)) ** 2))) if n else None,
            max_foot_xy_drift_m=float(self.max_foot_drift.max()) if n else None,
            max_foot_xy_drift_over_home_height=float(self.max_foot_drift.max() / self.home_height) if n else None,
            actuator_force_cap_duty={
                self.model.actuator(i).name: float(self.force_cap_samples[i] / n) if n else None
                for i in range(self.model.nu)
            },
            joint_hard_stop_duty=self.joint_hard_stop_samples / n if n else None,
        )
        return result


def _trace_row(env, raw, info, step: int, probe: BalanceProbe) -> dict:
    """One control-boundary row; window contacts and final physics loads differ."""
    row = {"step": step, "time_s": float(env.data.time)}
    for key, value in info.items():
        if np.isscalar(value) and not isinstance(value, (str, bytes)):
            row[key] = float(cast(float, value))
    for axis, value in zip("xyz", env.data.subtree_com[probe.pelvis]):
        row[f"com_{axis}_m"] = float(value)
    for axis, value in zip("xyz", env.data.xpos[probe.pelvis]):
        row[f"pelvis_{axis}_m"] = float(value)
    for axis, value in zip("wxyz", env.data.xquat[probe.pelvis]):
        row[f"pelvis_quaternion_{axis}"] = float(value)
    for axis, value in zip("xyz", env.data.qvel[3:6]):
        row[f"pelvis_angular_velocity_{axis}"] = float(value)
    for side, value, geom in zip(("right", "left"), probe.last_loads, probe.pads):
        row[f"final_physics_{side}_load_n"] = float(value)
        rotation = env.data.geom_xmat[geom].reshape(3, 3)
        row[f"{side}_sole_pitch_rad"] = float(np.arctan2(-rotation[2, 0], rotation[0, 0]))
        for axis, coordinate in zip("xyz", env.data.geom_xpos[geom]):
            row[f"{side}_sole_{axis}_m"] = float(coordinate)
    applied = env._action_filter_state if env.action_filter_cutoff_hz > 0 else np.clip(raw, -1, 1)
    for i in range(env.model.nu):
        name = env.model.actuator(i).name
        row[f"raw_action.{name}"] = float(raw[i])
        row[f"applied_action.{name}"] = float(applied[i])
        row[f"ctrl.{name}"] = float(env.data.ctrl[i])
        row[f"force.{name}"] = float(env.data.actuator_force[i])
        row[f"velocity.{name}"] = float(env.data.actuator_velocity[i])
    for j in range(1, env.model.njnt):
        name = env.model.joint(j).name
        row[f"qpos.{name}"] = float(env.data.qpos[env.model.jnt_qposadr[j]])
        row[f"qvel.{name}"] = float(env.data.qvel[env.model.jnt_dofadr[j]])
    return row


def evaluate_balance_episode(
    env, predict: Callable, seed: int, trace_path: str | Path | None = None, *, settle_steps: int = 200
) -> dict:
    """Evaluate a raw single environment; caller owns closing it and inference.

    ``predict(obs)`` returns an action or an SB3-style ``(action, state)`` pair.
    Every physics substep is sampled exactly once, after settling control steps.
    Existing experiment hooks are chained and restored, including on exceptions.
    No renderer or OpenGL context is created.
    """
    if isinstance(settle_steps, bool) or not isinstance(settle_steps, int) or settle_steps < 0:
        raise ValueError("settle_steps must be a nonnegative integer")
    obs, _ = env.reset(seed=int(seed))
    probe = BalanceProbe(env.model, env.data, env.target_standing_z)
    old_hook = env._substep_probe_hook
    control_index = 0

    def sample():
        if old_hook is not None:
            old_hook()
        probe.sample(record=control_index >= settle_steps)

    env._substep_probe_hook = sample
    rewards = base_rewards = 0.0
    unsupported: list[float] = []
    bilateral: list[float] = []
    traces: list[dict] = []
    reward_components: dict[str, float] = {}
    terminated = truncated = False
    info = {}
    try:
        for control_index in range(env.max_episode_steps):
            predicted = predict(obs)
            raw = np.asarray(predicted[0] if isinstance(predicted, tuple) else predicted, dtype=float)
            obs, reward, terminated, truncated, info = env.step(raw)
            rewards += float(reward)
            base_rewards += float(info.get("balance_base_reward", reward))
            for key, value in info.items():
                if key.startswith("reward_") and np.isscalar(value):
                    reward_components[key] = reward_components.get(key, 0.0) + float(cast(float, value))
            if control_index >= settle_steps:
                unsupported.append(float(info["unsupported_duty"]))
                bilateral.append(float(info["bilateral_support_duty"]))
            if trace_path is not None:
                traces.append(_trace_row(env, raw, info, control_index + 1, probe))
            if terminated or truncated:
                break
    finally:
        env._substep_probe_hook = old_hook
    steps = control_index + 1
    if trace_path is not None and traces:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in traces for key in row}))
            writer.writeheader()
            writer.writerows(traces)
    return {
        "seed": int(seed),
        "episode_steps": steps,
        "horizon_steps": env.max_episode_steps,
        "settle_steps": settle_steps,
        "duration_s": steps * env.dt,
        "full_horizon": steps == env.max_episode_steps and not terminated,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination": info.get("termination_reason", "horizon" if truncated else "unknown"),
        "reward": rewards,
        "base_reward": base_rewards,
        "reward_components": reward_components,
        "unsupported_windows": float(np.mean(unsupported)) if unsupported else None,
        "bilateral_windows": float(np.mean(bilateral)) if bilateral else None,
        "final_drift_m": info.get("drift_distance"),
        "research_only": True,
        **probe.results(),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_balance_panel(
    rows: Sequence[dict],
    *,
    horizon: int = 1000,
    stance_thresholds: StanceGateThresholds | None = None,
    targets: BalanceTargets = BalanceTargets(),
) -> dict:
    """Project existing gate and rank physical behavior, never shaped return.

    Failed episodes are excluded from reported duty/quietness means, exactly as
    in the canonical stance gate. Qualification also checks survival. Selection
    prioritizes eligibility and survival, so a short quiet failure cannot win.
    Duplicate seeds cannot inflate a smoke panel into a qualification panel.
    """
    thresholds = stance_thresholds or REFERENCE_STANCE_THRESHOLDS
    panel = stance_panel_from_episode_duties(
        # A terminal failure on the last step must not count as full horizon.
        episode_lengths=[
            r["episode_steps"] if r["full_horizon"] else min(r["episode_steps"], horizon - 1) for r in rows
        ],
        episode_duties=[r["unsupported_windows"] for r in rows],
        episode_rewards=[r["base_reward"] for r in rows],
        horizon=horizon,
    )
    gate_passed, failures = evaluate_stance_gate(panel, thresholds)
    unique_seeds = len({r["seed"] for r in rows})
    if unique_seeds != len(rows):
        failures.append("duplicate reset seeds do not provide independent panel episodes")
        gate_passed = False
    if any(
        r.get("horizon_steps", horizon) != horizon
        or r.get("settle_steps", thresholds.settle_steps) != thresholds.settle_steps
        for r in rows
    ):
        failures.append("episode horizon or settling window does not match the panel settings")
        gate_passed = False
    completed = [r for r in rows if r["full_horizon"] and r["episode_steps"] >= horizon]
    fields = (
        "physics_bilateral_20pct_weight",
        "physics_bilateral",
        "physics_unsupported",
        "weaker_foot_load_bw_mean",
        "physics_load_imbalance_mean",
        "pelvis_angular_velocity_rms",
        "com_vertical_std_over_home_height",
        "com_xy_sway_rms_over_home_height",
        "sole_pitch_motion_rms_rad",
        "sole_pitch_rms_rad",
        "max_foot_xy_drift_over_home_height",
        "positive_actuator_work_J",
        "joint_hard_stop_duty",
    )
    means = {}
    for field in fields:
        values = [r.get(field) for r in completed]
        means[field] = (
            float(np.mean(cast(list[float], values)))
            if values and all(v is not None and math.isfinite(v) for v in values)
            else None
        )
    behavior_failures = []
    if not gate_passed:
        behavior_failures.append("projected canonical stance criteria did not pass")
    if unique_seeds < targets.min_eval_episodes:
        behavior_failures.append(
            f"requires {targets.min_eval_episodes} distinct reset seeds; smoke panels cannot qualify"
        )
    for key, threshold, minimum in (
        ("physics_bilateral_20pct_weight", targets.min_bilateral_20pct_weight, True),
        ("pelvis_angular_velocity_rms", targets.max_pelvis_angular_velocity_rms, False),
        ("com_vertical_std_over_home_height", targets.max_com_vertical_std_over_home_height, False),
        ("sole_pitch_motion_rms_rad", targets.max_sole_pitch_motion_rms_rad, False),
    ):
        value = means[key]
        if value is None or (value < threshold if minimum else value > threshold):
            behavior_failures.append(f"{key} {value} must be {'>=' if minimum else '<='} {threshold}")
    qualified = not behavior_failures
    # Conservative finite sentinels allow strict JSON and stable lexicographic
    # sorting even when no complete episode contributes measurable behavior.
    bilateral = means["physics_bilateral_20pct_weight"]
    motion = means["pelvis_angular_velocity_rms"]
    selection_key = [
        int(qualified),
        int(gate_passed),
        panel.full_horizon_fraction,
        bilateral if bilateral is not None else -1.0,
        -motion if motion is not None else -1e30,
    ]
    result: dict = _json_safe(
        {
            "research_only": True,
            "advances_curriculum": False,
            "gate_interpretation": "Projected canonical stance criteria on base reward; not official qualification or advancement",
            "projected_stance_gate_passed": gate_passed,
            "projected_stance_gate_failures": failures,
            "projected_stance_panel": asdict(panel),
            "reference_stance_thresholds": asdict(thresholds),
            "proposed_behavior_targets": asdict(targets),
            "behavior_qualified": qualified,
            "behavior_failures": behavior_failures,
            "behavior_mean_population": "Full-horizon episodes only; survival and panel size separately required",
            "n_behavior_episodes": len(completed),
            "n_unique_seeds": unique_seeds,
            "behavior_means": means,
            "selection_key": selection_key,
            "selection_order": [
                "behavior_qualified",
                "projected_stance_gate_passed",
                "survival",
                "bilateral_load",
                "negative_pelvis_motion",
            ],
            "mean_study_reward": float(np.mean([r["reward"] for r in rows])) if rows else None,
        }
    )
    return result
