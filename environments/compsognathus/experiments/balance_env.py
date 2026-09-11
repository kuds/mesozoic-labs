"""Opt-in learned-balance study on the canonical 14-actuator Compsognathus.

This module deliberately registers no Gym environment and edits no production
plant. Arms A/B retain the canonical rewards; C/D add bounded stance shaping.
The filter in B/D is a class declaration and uses the existing BaseDinoEnv
implementation, including its first-command seeding and reset semantics.
Study checkpoints require their own identity (see ``balance_identity``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from environments.compsognathus.envs.compsognathus_env import CompsognathusEnv


@dataclass(frozen=True)
class BalanceShaping:
    """Dimensionless stance shaping, expressed per control step.

    The proposed defaults add at most 0.75 to the approximately 3-point
    canonical quiet-stance reward. Full single support costs only 0.15 and
    unsupported contact costs 0.25, leaving the canonical recovery incentive
    intact. They are study hypotheses, not validated training defaults.

    At one body weight total load, 50/50, 75/25, 90/10, single support and
    airborne states receive +0.75, +0.675, +0.18, -0.15 and -0.25 respectively.
    The weaker-foot target is 25% of body weight; per-foot loads are clipped
    at one body weight so impact magnitude cannot buy unbounded reward.
    """

    bilateral_weight: float = 0.75
    imbalance_weight: float = 0.15
    unsupported_weight: float = 0.25
    min_foot_load_bw: float = 0.25
    load_clip_bw: float = 1.0

    def __post_init__(self) -> None:
        for name in ("bilateral_weight", "imbalance_weight", "unsupported_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.min_foot_load_bw) or not 0 < self.min_foot_load_bw <= 0.5:
            raise ValueError("min_foot_load_bw must be finite and in (0, 0.5]")
        if not np.isfinite(self.load_clip_bw) or self.load_clip_bw < 1.0:
            raise ValueError("load_clip_bw must be finite and at least one body weight")


DEFAULT_SHAPING = BalanceShaping()
ZERO_SHAPING = BalanceShaping(bilateral_weight=0.0, imbalance_weight=0.0, unsupported_weight=0.0)


@dataclass(frozen=True)
class BalanceArm:
    name: str
    filter_hz: float
    shaping: BalanceShaping


ARMS: Mapping[str, BalanceArm] = MappingProxyType(
    {
        "A": BalanceArm("A", 0.0, ZERO_SHAPING),
        "B": BalanceArm("B", 10.0, ZERO_SHAPING),
        "C": BalanceArm("C", 0.0, DEFAULT_SHAPING),
        "D": BalanceArm("D", 10.0, DEFAULT_SHAPING),
    }
)


def balance_shaping_terms(
    right_force_n: float,
    left_force_n: float,
    body_weight_n: float,
    shaping: BalanceShaping = DEFAULT_SHAPING,
) -> dict[str, float]:
    """Score conservative per-foot loads without changing the base reward.

    Callers must supply each foot's minimum over the *same* control window,
    not means or peaks: alternating left/right impacts must not look like
    continuous bilateral support. Environment calls use the base env's
    substep aggregation directly.

    Imbalance is attenuated when total measured support is below body
    weight. Missing support is penalized separately, so tiny balanced
    contacts do not earn the full bilateral bonus and zero contact is worse
    than full single support. All reward terms are bounded under impacts.
    """
    for name, value in (("right_force_n", right_force_n), ("left_force_n", left_force_n)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isfinite(body_weight_n) or body_weight_n <= 0:
        raise ValueError("body_weight_n must be finite and positive")

    right = min(float(right_force_n) / float(body_weight_n), shaping.load_clip_bw)
    left = min(float(left_force_n) / float(body_weight_n), shaping.load_clip_bw)
    total = right + left
    quality = min(min(right, left) / shaping.min_foot_load_bw, 1.0)
    support = min(total, 1.0)
    # No load is not evidence of good balance. Its imbalance cost is still
    # zero because support is zero; the unsupported term prices that case.
    imbalance = 1.0 if total == 0.0 else abs(right - left) / total
    unsupported = 1.0 - support
    bilateral_reward = shaping.bilateral_weight * quality
    imbalance_reward = -shaping.imbalance_weight * imbalance * support
    unsupported_reward = -shaping.unsupported_weight * unsupported
    return {
        "balance_right_load_bw": right,
        "balance_left_load_bw": left,
        "balance_support_quality": quality,
        "balance_load_imbalance": imbalance,
        "balance_unsupported_fraction": unsupported,
        "reward_balance_bilateral": bilateral_reward,
        "reward_balance_imbalance": imbalance_reward,
        "reward_balance_unsupported": unsupported_reward,
        "balance_shaping_reward": bilateral_reward + imbalance_reward + unsupported_reward,
    }


class CompsognathusBalanceEnv(CompsognathusEnv):
    """Canonical mechanics and observations with explicit study rewards."""

    supported_training_backends = ("stable-baselines3",)
    action_filter_cutoff_hz = 0.0

    def __init__(self, *, balance_arm: BalanceArm = ARMS["A"], **env_kwargs: Any):
        if balance_arm.filter_hz != self.action_filter_cutoff_hz:
            raise ValueError("Study arm filter does not match the environment class declaration")
        self.balance_arm = balance_arm
        super().__init__(**env_kwargs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, base_reward, terminated, truncated, info = super().step(action)
        right, left = self._aggregated_foot_contact_forces()
        terms = balance_shaping_terms(
            right,
            left,
            self.body_mass * abs(float(self.model.opt.gravity[2])),
            self.balance_arm.shaping,
        )
        # Compute after the canonical step so a terminal fall penalty is
        # included in both base and study totals, exactly once. A/B preserve
        # the original float as well as the original physical trajectory.
        shaping_reward = terms["balance_shaping_reward"]
        reward = base_reward if shaping_reward == 0.0 else base_reward + shaping_reward
        info.update(terms)
        info["balance_base_reward"] = base_reward
        info["balance_study_reward"] = reward
        info["reward_total"] = reward
        return observation, reward, terminated, truncated, info


class FilteredCompsognathusBalanceEnv(CompsognathusBalanceEnv):
    """Ten Hz global command filter using the repository's existing RC law."""

    action_filter_cutoff_hz = 10.0


def make_balance_env(arm: str, **env_kwargs: Any) -> CompsognathusBalanceEnv:
    """Build one named, unregistered arm without touching production config."""
    if arm not in ARMS:
        raise ValueError(f"Unknown learned-balance arm {arm!r}; expected one of {tuple(ARMS)}")
    specification = ARMS[arm]
    env_class = FilteredCompsognathusBalanceEnv if specification.filter_hz else CompsognathusBalanceEnv
    return env_class(balance_arm=specification, **env_kwargs)
