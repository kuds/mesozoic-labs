"""The velociraptor walk's stride-averaged tracking (decision D-D64) on the real plant, through the walk gate.

At a moving command the tracking term, the metrics and the gate read the trailing mean of the walk's
``tracking_window_s`` (0.70 s, the reference walk's own stride), so its within-stride sway cancels over a stride.
Measured on 2026-11-26 (physics r6) with ``roll_command_panel`` and ``evaluate_command_tracking_gate`` on the walk
panel (plan from 3042, 20 cruise and 5 stand episodes):

- The reference walk (``scripts/reference_walk.py``) under a command matched to its 0.257 m/s (heading offset 0):
  tracking quality after 2 s 0.955 an episode (0.785 per step), in tolerance after the settle 0.873 to 1.000, mean
  0.994 (per step 0.661 to 0.996, mean 0.972), ``heading_settle`` 20/20 (18/20 per step), speed ratio 1.019 to
  1.218, longest stop 0 s, 2.708 a cruise step (2.397 per step).  It clears every tracking bar (at this command,
  inside the 0.347 m/s velocity tolerance, even the statue reads in tolerance); one episode, seed 3045, falls at
  step 373, and its truncated trace also fails the lead-swap bar, while the other 19 pass every bar, so the gate
  FAILS on survival alone (survival_lcb 0.784 < 0.80).
- The statue (the zero action) on the committed panel: 0.0004 tracking a step, 0.500 a cruise step (the alive
  bonus), in tolerance after the settle 0.000 and 0/20 cruise successes: it FAILS; its stand episodes pass (5/5),
  exactly as per step.

The tests below roll reduced panels of the same plan (its first cruise seeds and first stand seed).
"""

from __future__ import annotations

import dataclasses
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
from environments.velociraptor_mongoliensis.scripts.reference_walk import ReferenceWalk

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
#: The reference walk's speed: its stride over its period (0.18 m / 0.70 s).
REFERENCE_SPEED = 0.257


def _walk():
    return load_stage_config(VELOCIRAPTOR_MONGOLIENSIS, "locomotion")


def _matched(stage):
    """The stage with the command matched to the reference walk: speed_range (0.257, 0.257), heading offset 0."""
    command = dataclasses.replace(
        stage["env_kwargs"]["command_config"],
        speed_range=(REFERENCE_SPEED, REFERENCE_SPEED),
        initial_heading_offset_max=0.0,
    )
    return {**stage, "env_kwargs": {**stage["env_kwargs"], "command_config": command}}


def _roll(stage, *, cruise, stand, controller):
    """Roll a reduced walk panel (plan from 3042) and judge it: (rollout, gate result, thresholds)."""
    thresholds = thresholds_from_curriculum(
        {**stage["curriculum_kwargs"], "min_cruise_episodes": cruise, "min_stand_episodes": stand}
    )
    command = stage["env_kwargs"]["command_config"]
    plan = plan_command_panel(thresholds, command)
    holder = {}
    original = species_registry.env_class

    def capturing(model_id):
        base = original(model_id)

        def build(**kwargs):
            holder["env"] = base(**kwargs)
            return holder["env"]

        return build

    class Predict:
        def reset(self):
            env = holder["env"]
            self.act = controller(env, stand=not np.any(env._command))

        def __call__(self, obs):
            return self.act()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(species_registry, "env_class", capturing)
        rollout = roll_command_panel(
            VELOCIRAPTOR_MONGOLIENSIS, stage, predict=Predict(), thresholds=thresholds, plan=plan, keep_steps=True
        )
    result = evaluate_command_tracking_gate(rollout.episodes, thresholds, command_config=command)
    return rollout, result, thresholds


def _statue(env, *, stand):
    zero = np.zeros(env.action_space.shape, np.float32)
    return lambda: zero


def _reference(env, *, stand):
    if stand:
        return _statue(env, stand=True)
    walker = ReferenceWalk(env)
    return lambda: walker.action(env.data, env.dt)


def _reasons(result, index):
    return {reason_key(reason) for reason in result.episode_reasons[index]}


def _quality_after_2s(rows, command):
    """The episode's mean tracking quality (the term over its weight) after 2 s, and the same recomputed per step."""
    parameters = command.tracking_parameters()
    late = [row for row in rows if row["time_s"] > 2.0 + 1e-9]
    windowed = float(np.mean([row["reward_tracking"] / parameters["weight"] for row in late]))
    per_step = float(
        np.mean(
            [
                math.exp(
                    -(
                        (
                            math.hypot(row["actual_v_x"] - row["command_v_x"], row["actual_v_y"] - row["command_v_y"])
                            / parameters["velocity_sigma"]
                        )
                        ** 2
                    )
                    - ((row["actual_yaw_rate"] - row["command_yaw_rate"]) / parameters["yaw_rate_sigma"]) ** 2
                )
                for row in late
            ]
        )
    )
    return windowed, per_step


def test_the_walk_declares_the_reference_stride_as_its_tracking_window():
    """0.70 s, 70 control steps: the reference walk's own stride at 0.257 m/s, an upper bound on the stride at
    0.5 x cruise (1.05 m/s), since strides shorten with speed (decision D-D64)."""
    command = _walk()["env_kwargs"]["command_config"]
    assert command.tracking_window_s == 0.70 == ReferenceWalk.period
    assert tracking_window_steps(command.tracking_window_s, 0.01) == 70
    assert ReferenceWalk.stride / ReferenceWalk.period == pytest.approx(REFERENCE_SPEED, abs=5e-4)
    assert REFERENCE_SPEED < 0.5 * command.cruise_speed == command.speed_range[0]


def test_the_reference_walk_clears_every_bar_on_its_first_cruise_seeds():
    """Seeds 3042-3044 under the matched command: settled, every step in tolerance after the settle, quality 0.95-0.99
    after 2 s (per step about 0.80), every episode a success; the stand seed 3053 holds.  The reduced panel fails only
    for its size (the families' lower confidence bounds at three episodes)."""
    rollout, result, thresholds = _roll(_matched(_walk()), cruise=3, stand=1, controller=_reference)
    measurement = rollout.measurement.as_dict()
    assert measurement["protocol"] == "command_panel/v2" and measurement["tracking_window_s"] == 0.70
    command = _matched(_walk())["env_kwargs"]["command_config"]
    cruise = [(i, e) for i, e in enumerate(rollout.episodes) if e.family == "cruise"]
    assert [e.seed for _, e in cruise] == [3042, 3043, 3044]
    for index, episode in cruise:
        assert episode.full_horizon and episode.first_event_settled
        assert episode.tracking_fraction_after_settle == 1.0
        assert 1.0 <= episode.speed_ratio <= 1.05
        assert episode.longest_stop_s == 0.0
        assert _reasons(result, index) == set()
        windowed, per_step = _quality_after_2s(rollout.steps[index], command)
        assert windowed >= 0.95 and 0.75 <= per_step <= 0.85
    stand = [(i, e) for i, e in enumerate(rollout.episodes) if e.family == "stand"]
    assert [e.seed for _, e in stand] == [3053]
    assert _reasons(result, stand[0][0]) == set()
    assert result.failures and all("lcb" in failure for failure in result.failures)


def test_the_statue_fails_the_walk_gate():
    """On the committed panel's seeds 3042, 3043 (cruise) and 3052 (stand): no cruise step in tolerance, under 0.01
    tracking a step, every cruise episode failing the tracking, speed and gait bars; the stand passes."""
    rollout, result, _ = _roll(_walk(), cruise=2, stand=1, controller=_statue)
    assert not result.passed
    for index, episode in enumerate(rollout.episodes):
        rows = rollout.steps[index]
        if episode.family == "cruise":
            assert episode.tracking_fraction_after_settle == 0.0
            assert np.mean([row["reward_tracking"] for row in rows]) < 0.01
            assert {"heading_settle", "min_cruise_tracking_fraction", "min_cruise_speed_ratio"} <= _reasons(
                result, index
            )
        else:
            assert episode.seed == 3052 and _reasons(result, index) == set()
