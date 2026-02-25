"""
Locomotion evaluation metrics for dinosaur environments.

Provides standardized metrics beyond average reward:
  - Gait symmetry (left-right phase difference)
  - Cost of transport (energy / distance / weight)
  - Stride frequency and regularity
  - Forward velocity consistency
  - Time-to-target

Usage:
    from environments.shared.metrics import LocomotionMetrics

    metrics = LocomotionMetrics()
    for step_info in episode_infos:
        metrics.record_step(step_info)
    report = metrics.compute()
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LocomotionMetrics:
    """Collects per-step data and computes locomotion quality metrics.

    Designed to work with the info dicts returned by BaseDinoEnv.step().
    Call :meth:`record_step` each timestep, then :meth:`compute` at
    episode end.
    """

    # Accumulated per-step data
    _forward_velocities: List[float] = field(default_factory=list)
    _energies: List[float] = field(default_factory=list)
    _left_contacts: List[float] = field(default_factory=list)
    _right_contacts: List[float] = field(default_factory=list)
    _pelvis_heights: List[float] = field(default_factory=list)
    _prey_distances: List[float] = field(default_factory=list)
    _tilt_angles: List[float] = field(default_factory=list)
    _rewards: List[float] = field(default_factory=list)
    _dt: float = 0.02  # default timestep * frame_skip
    _termination_reason: Optional[str] = None

    def reset(self):
        """Clear all accumulated data."""
        self._forward_velocities.clear()
        self._energies.clear()
        self._left_contacts.clear()
        self._right_contacts.clear()
        self._pelvis_heights.clear()
        self._prey_distances.clear()
        self._tilt_angles.clear()
        self._rewards.clear()
        self._termination_reason = None

    def record_step(self, info: Dict[str, Any], reward: float = 0.0):
        """Record a single environment step.

        Args:
            info: The info dict from env.step(). Expected keys:
                - ``forward_vel``: scalar forward velocity
                - ``reward_energy``: energy penalty (negative)
                - ``pelvis_height``: pelvis z-position
                - ``prey_distance``: distance to target (optional)
                - ``r_foot_contact`` / ``l_foot_contact``: binary (optional)
            reward: Total reward for this step.
        """
        self._forward_velocities.append(info.get("forward_vel", 0.0))
        self._energies.append(abs(info.get("reward_energy", 0.0)))
        self._pelvis_heights.append(info.get("pelvis_height", 0.0))
        self._rewards.append(reward)

        if "prey_distance" in info:
            self._prey_distances.append(info["prey_distance"])

        if "tilt_angle" in info:
            self._tilt_angles.append(info["tilt_angle"])

        self._left_contacts.append(info.get("l_foot_contact", 0.0))
        self._right_contacts.append(info.get("r_foot_contact", 0.0))

        # Capture termination reason from the final step
        if "termination_reason" in info:
            self._termination_reason = info["termination_reason"]

    def compute(self, body_mass: float = 1.0) -> Dict[str, float]:
        """Compute all locomotion metrics from accumulated data.

        Args:
            body_mass: Dinosaur body mass in kg for cost of transport.

        Returns:
            Dictionary of metric name to value.
        """
        n = len(self._forward_velocities)
        if n == 0:
            return {"error": float("nan")}

        fwd = np.array(self._forward_velocities)
        energies = np.array(self._energies)

        result: Dict[str, float] = {}

        # --- Forward velocity statistics ---
        result["mean_forward_velocity"] = float(np.mean(fwd))
        result["std_forward_velocity"] = float(np.std(fwd))
        result["max_forward_velocity"] = float(np.max(fwd))
        result["velocity_consistency"] = float(1.0 - np.std(fwd) / (np.abs(np.mean(fwd)) + 1e-8))

        # --- Distance traveled ---
        distance = float(np.sum(fwd * self._dt))
        result["total_distance"] = distance

        # --- Cost of transport ---
        #   CoT = total energy / (mass * gravity * distance)
        total_energy = float(np.sum(energies))
        gravity = 9.81
        if abs(distance) > 0.01:
            result["cost_of_transport"] = total_energy / (body_mass * gravity * abs(distance))
        else:
            result["cost_of_transport"] = float("inf")
        result["total_energy"] = total_energy

        # --- Gait symmetry ---
        if self._left_contacts and self._right_contacts:
            left = np.array(self._left_contacts)
            right = np.array(self._right_contacts)
            result["gait_symmetry"] = self._compute_gait_symmetry(left, right)

        # --- Stride frequency ---
        if self._left_contacts or self._right_contacts:
            contacts = np.array(self._left_contacts) + np.array(self._right_contacts)
            result["stride_frequency"] = self._compute_stride_frequency(contacts)

        # --- Pelvis height stability ---
        if self._pelvis_heights:
            heights = np.array(self._pelvis_heights)
            result["mean_pelvis_height"] = float(np.mean(heights))
            result["pelvis_height_variance"] = float(np.var(heights))

        # --- Tilt angle statistics ---
        if self._tilt_angles:
            tilts = np.array(self._tilt_angles)
            result["mean_tilt_angle"] = float(np.mean(tilts))
            result["max_tilt_angle"] = float(np.max(tilts))
            result["std_tilt_angle"] = float(np.std(tilts))

        # --- Time to target ---
        if self._prey_distances:
            distances = np.array(self._prey_distances)
            result["initial_prey_distance"] = float(distances[0])
            result["final_prey_distance"] = float(distances[-1])
            result["min_prey_distance"] = float(np.min(distances))

            # Time to reach within 0.5m of target (or -1 if never reached)
            close_steps = np.where(distances < 0.5)[0]
            if len(close_steps) > 0:
                result["time_to_target"] = float(close_steps[0] * self._dt)
            else:
                result["time_to_target"] = -1.0

        # --- Reward statistics ---
        rewards = np.array(self._rewards)
        result["total_reward"] = float(np.sum(rewards))
        result["mean_step_reward"] = float(np.mean(rewards))
        result["episode_length"] = n

        # --- Termination reason ---
        if self._termination_reason is not None:
            result["termination_reason"] = self._termination_reason

        return result

    @staticmethod
    def _compute_gait_symmetry(left_contacts: np.ndarray, right_contacts: np.ndarray) -> float:
        """Compute gait symmetry from foot contact sequences.

        Returns a value in [0, 1] where 1 is perfectly symmetric
        (alternating contacts) and 0 is completely asymmetric.
        """
        if len(left_contacts) < 2:
            return 0.0

        # Detect contact onset events (0 -> positive)
        left_onsets = np.where(np.diff(left_contacts > 0.1, prepend=False))[0]
        right_onsets = np.where(np.diff(right_contacts > 0.1, prepend=False))[0]

        if len(left_onsets) < 2 or len(right_onsets) < 2:
            return 0.0

        # Mean stride period for each foot
        left_periods = np.diff(left_onsets).astype(float)
        right_periods = np.diff(right_onsets).astype(float)

        mean_left = np.mean(left_periods)
        mean_right = np.mean(right_periods)

        if mean_left + mean_right < 1e-8:
            return 0.0

        # Symmetry = 1 - |difference| / average
        symmetry = 1.0 - abs(mean_left - mean_right) / ((mean_left + mean_right) / 2.0)
        return float(np.clip(symmetry, 0.0, 1.0))

    def _compute_stride_frequency(self, combined_contacts: np.ndarray) -> float:
        """Compute stride frequency in Hz from combined contact signal."""
        if len(combined_contacts) < 2:
            return 0.0

        onsets = np.where(np.diff(combined_contacts > 0.1, prepend=False))[0]
        if len(onsets) < 2:
            return 0.0

        mean_period_steps = np.mean(np.diff(onsets))
        mean_period_seconds = mean_period_steps * self._dt

        if mean_period_seconds < 1e-8:
            return 0.0

        return float(1.0 / mean_period_seconds)

    @staticmethod
    def aggregate_episodes(
        episode_reports: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate metrics across multiple episodes.

        Args:
            episode_reports: List of dicts from :meth:`compute`.

        Returns:
            Aggregated dict with mean/std for each numeric metric and
            termination reason counts.
        """
        if not episode_reports:
            return {}

        # Separate numeric keys from non-numeric (like termination_reason)
        non_numeric_keys = {"termination_reason", "error"}
        keys = [k for k in episode_reports[0] if k not in non_numeric_keys]
        result: Dict[str, Any] = {}

        for key in keys:
            values = [r[key] for r in episode_reports if key in r and r[key] != float("inf")]
            if values:
                result[f"mean_{key}"] = float(np.mean(values))
                result[f"std_{key}"] = float(np.std(values))

        result["n_episodes"] = float(len(episode_reports))

        # Aggregate termination reasons
        reasons = [r["termination_reason"] for r in episode_reports if "termination_reason" in r]
        if reasons:
            counts = Counter(reasons)
            result["termination_counts"] = dict(counts)
            # Also add truncated count (episodes that ended without termination)
            n_truncated = len(episode_reports) - len(reasons)
            if n_truncated > 0:
                result["termination_counts"]["truncated"] = n_truncated

        return result
