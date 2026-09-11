"""Reproducible calibration evidence for the learned-balance study.

These inexpensive screens establish reward ordering and a physically feasible
reference. They do not demonstrate that PPO learns quiet balance, or that an
unperturbed home command can recover from pushes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from environments.compsognathus.experiments.balance_env import balance_shaping_terms, make_balance_env


def reward_ordering_report() -> list[dict[str, Any]]:
    """Isolate loading incentives at matched canonical pose/action quality.

    Loads are measured in body weights. The first four cases hold total
    support constant; the remaining cases probe zero support and impacts.
    Base reward is deliberately omitted: this is an additive-term screen,
    not a claim that these loads all describe dynamically reachable states.
    """
    scenarios = (
        ("balanced_50_50", 0.5, 0.5),
        ("asymmetric_75_25", 0.75, 0.25),
        ("asymmetric_90_10", 0.9, 0.1),
        ("single_support", 1.0, 0.0),
        ("airborne", 0.0, 0.0),
        ("tiny_balanced_contact", 0.001, 0.001),
        ("balanced_impact", 20.0, 20.0),
        ("single_foot_impact", 20.0, 0.0),
    )
    return [
        {
            "case": name,
            "input_right_load_bw": right,
            "input_left_load_bw": left,
            **balance_shaping_terms(right, left, 1.0),
        }
        for name, right, left in scenarios
    ]


def home_reference_report(
    seeds: Sequence[int] = tuple(range(4010, 4018)),
    env_kwargs: Mapping[str, Any] | None = None,
    settle_steps: int = 200,
) -> list[dict[str, Any]]:
    """Roll out a home command and score base and proposed rewards together.

    Arm C shares the canonical dynamics and observations. Zero actions also
    produce identical commands under the B/D filter, so one rollout per
    seed suffices for this controller. Metrics use conservative per-foot
    minima across each control window; all returns include settling and
    terminal penalties, while physical metrics exclude the settling prefix.
    """
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must contain at least one integer")
    if not isinstance(settle_steps, int) or isinstance(settle_steps, bool) or settle_steps < 0:
        raise ValueError("settle_steps must be a nonnegative integer")
    reports = []
    with make_balance_env("C", **dict(env_kwargs or {})) as env:
        if settle_steps >= env.max_episode_steps:
            raise ValueError("settle_steps must be shorter than the episode horizon")
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        for seed in seeds:
            env.reset(seed=seed)
            base_return = study_return = 0.0
            samples = []
            terminated = truncated = False
            for step in range(env.max_episode_steps):
                _, reward, terminated, truncated, info = env.step(action)
                base_return += info["balance_base_reward"]
                study_return += reward
                if step >= settle_steps:
                    samples.append(
                        {
                            "bilateral_support_duty": info["bilateral_support_duty"],
                            "both_feet_over_20pct_bw": float(
                                min(info["balance_right_load_bw"], info["balance_left_load_bw"]) > 0.2
                            ),
                            "pelvis_angular_speed_rad_s": info["pelvis_angular_vel"],
                            "tilt_rad": info["tilt_angle"],
                            "shaping_reward": info["balance_shaping_reward"],
                            "base_reward": info["balance_base_reward"],
                            "right_load_bw": info["balance_right_load_bw"],
                            "left_load_bw": info["balance_left_load_bw"],
                        }
                    )
                if terminated or truncated:
                    break
            reports.append(
                {
                    "seed": seed,
                    "controller": "zero_residual_home_position_servos",
                    "steps": step + 1,
                    "duration_s": (step + 1) * env.dt,
                    "full_horizon": bool(truncated and not terminated),
                    "base_return": base_return,
                    "study_return": study_return,
                    "settle_steps": settle_steps,
                    "post_settle_samples": len(samples),
                    "post_settle_means": (
                        {key: float(np.mean([sample[key] for sample in samples])) for key in samples[0]}
                        if samples
                        else None
                    ),
                }
            )
    return reports
