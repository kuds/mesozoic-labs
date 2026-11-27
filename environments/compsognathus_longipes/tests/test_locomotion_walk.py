"""The Compsognathus walks' stride-averaged tracking (decision D-D64) on the real plants, through the walk gate.

At a moving command the tracking term, the metrics and the gate read the trailing mean of the walk's
``tracking_window_s``.  No walking controller runs on either plant, so each window is the stride the rule asks for
(the stride at the slowest end-state command, 0.5 x cruise) estimated from the T. rex reference walk's
dimensionless stride, T sqrt(g/L) = 3.40 at Froude 0.0445, scaled to the model's Froude number at 0.5 x cruise as
T sqrt(g/L) ~ Fr^-0.2, with L the gait module's leg length: 1.22 s (61 steps of 0.02 s) on the anatomical model and
1.50 s (75 steps) on the robot.

The statue (the zero action) on each committed walk panel (plan from 3042, 20 cruise and 5 stand episodes),
measured on 2026-11-26 with ``roll_command_panel`` and ``evaluate_command_tracking_gate``: 0.0037 (anatomical) and
0.0043 (robot) tracking a step, 1.304 a cruise step, in tolerance after the settle 0.000 and 0/20 cruise successes,
so it FAILS on both; its stand episodes pass (5/5), exactly as per step.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import environments.shared.species_registry as species_registry
from environments.shared.command_tracking_panel import plan_command_panel, roll_command_panel
from environments.shared.config import load_stage_config
from environments.shared.curriculum.command_tracking_gate import (
    evaluate_command_tracking_gate,
    reason_key,
    thresholds_from_curriculum,
)
from environments.shared.direction_commands import tracking_window_steps
from environments.shared.model_id import ModelId

#: The T. rex reference walk's stride (1.04 s at 0.633 m/s) and leg length (0.917 m): T sqrt(g/L) = 3.40 at Fr 0.0445.
TYRANNOSAURUS_REX_STRIDE_S, TYRANNOSAURUS_REX_SPEED, TYRANNOSAURUS_REX_LEG_M = 1.04, 0.633, 0.917
GRAVITY = 9.81
#: Per embodiment: the gait module's leg length (m), the window (s) and its control steps.
WALKS = {
    "anatomical": (0.2396, 1.22, 61),
    "robot": (0.2139, 1.50, 75),
}


def _walk(embodiment):
    return load_stage_config(ModelId("compsognathus_longipes", embodiment), "locomotion")


def _froude(speed, leg):
    return speed**2 / (GRAVITY * leg)


@pytest.mark.parametrize("embodiment", sorted(WALKS))
def test_the_walk_declares_the_froude_scaled_stride_at_half_cruise_as_its_tracking_window(embodiment):
    leg, window, steps = WALKS[embodiment]
    command = _walk(embodiment)["env_kwargs"]["command_config"]
    dt = 0.02
    assert command.tracking_window_s == window
    assert tracking_window_steps(window, dt) == steps
    slowest = 0.5 * command.cruise_speed
    assert command.speed_range[0] == pytest.approx(slowest)
    dimensionless = TYRANNOSAURUS_REX_STRIDE_S * math.sqrt(GRAVITY / TYRANNOSAURUS_REX_LEG_M)
    assert dimensionless == pytest.approx(3.40, abs=0.005)
    stride = (
        dimensionless
        * (_froude(slowest, leg) / _froude(TYRANNOSAURUS_REX_SPEED, TYRANNOSAURUS_REX_LEG_M)) ** -0.2
        * math.sqrt(leg / GRAVITY)
    )
    assert round(stride / dt) * dt == pytest.approx(window)


@pytest.mark.parametrize("embodiment", sorted(WALKS))
def test_the_statue_fails_the_walk_gate(embodiment):
    """On the committed panel's seeds 3042, 3043 (cruise) and 3052 (stand): no cruise step in tolerance, under 0.01
    tracking a step, every cruise episode failing the tracking, speed and gait bars; the stand passes."""
    model_id = ModelId("compsognathus_longipes", embodiment)
    stage = _walk(embodiment)
    thresholds = thresholds_from_curriculum(
        {**stage["curriculum_kwargs"], "min_cruise_episodes": 2, "min_stand_episodes": 1}
    )
    command = stage["env_kwargs"]["command_config"]
    plan = plan_command_panel(thresholds, command)
    holder = {}
    original = species_registry.env_class

    def capturing(mid):
        base = original(mid)

        def build(**kwargs):
            holder["env"] = base(**kwargs)
            return holder["env"]

        return build

    def statue(_obs):
        return np.zeros(holder["env"].action_space.shape, np.float32)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(species_registry, "env_class", capturing)
        rollout = roll_command_panel(model_id, stage, predict=statue, thresholds=thresholds, plan=plan, keep_steps=True)
    assert rollout.measurement.tracking_window_s == WALKS[embodiment][1]
    result = evaluate_command_tracking_gate(rollout.episodes, thresholds, command_config=command)
    assert not result.passed
    for index, episode in enumerate(rollout.episodes):
        reasons = {reason_key(reason) for reason in result.episode_reasons[index]}
        if episode.family == "cruise":
            assert episode.full_horizon and episode.tracking_fraction_after_settle == 0.0
            assert np.mean([row["reward_tracking"] for row in rollout.steps[index]]) < 0.01
            assert {"heading_settle", "min_cruise_tracking_fraction", "min_cruise_speed_ratio"} <= reasons
        else:
            assert episode.seed == 3052 and reasons == set()
