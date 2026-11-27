"""The Allosaurus walk's stride-averaged tracking (decision D-D64) on the real plant, through the walk gate.

At a moving command the tracking term, the metrics and the gate read the trailing mean of the walk's
``tracking_window_s`` (1.31 s), so the reference walk's within-stride lateral sway (RMS 0.18 m/s per step, twice the
0.083 m/s velocity sigma) cancels over a stride.  Measured on 2026-11-26 (physics r2) with ``roll_command_panel`` and
``evaluate_command_tracking_gate`` on the walk panel (plan from 3042, 20 cruise and 5 stand episodes):

- The reference walk (``scripts/reference_walk.py``; its ``walking=False`` balance controller on the stand
  episodes) under a command matched to its 0.345 m/s (heading offset 0): tracking quality after 2 s 0.799 an
  episode (0.142 per step), in tolerance after the settle 0.957 to 0.958 (per step 0.133 to 0.136), speed ratio
  1.000 to 1.002, longest stop 0 s, 3.862 a cruise step (2.331 per step).  It clears the tracking fraction and the
  speed ratio on 20/20 and fails ``heading_settle`` on 20/20, its only failing bar: on its scripted start a centred
  stride mean of its speed reaches the speed tolerance at about 2.70 s, so the gate's trailing window first reads in
  tolerance at 3.46-3.49 s (seeds 3042-3045), after the 2.31 s allowance (1 s + 0 + the 1.31 s window).  Its stand
  episodes pass (5/5).
- The statue (the zero action) falls in every episode, cruise and stand (about 90 of 3000 steps): 0.0041 tracking
  a step, and it FAILS the walk gate.

The tests below roll reduced panels of the same plan (its first cruise seeds and first stand seed).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import environments.shared.species_registry as species_registry
from environments.allosaurus_fragilis.scripts.reference_walk import PERIOD, ReferenceWalk
from environments.shared.command_tracking_panel import plan_command_panel, roll_command_panel
from environments.shared.config import load_stage_config
from environments.shared.curriculum.command_tracking_gate import (
    evaluate_command_tracking_gate,
    reason_key,
    thresholds_from_curriculum,
)
from environments.shared.direction_commands import tracking_window_steps
from environments.shared.model_id import ModelId

ALLOSAURUS_FRAGILIS = ModelId("allosaurus_fragilis", "anatomical")
#: The reference walk's measured speed along its heading once walking.
REFERENCE_SPEED = 0.345


def _walk():
    return load_stage_config(ALLOSAURUS_FRAGILIS, "locomotion")


def _matched(stage):
    """The stage with the command matched to the reference walk: speed_range (0.345, 0.345), heading offset 0."""
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
            ALLOSAURUS_FRAGILIS, stage, predict=Predict(), thresholds=thresholds, plan=plan, keep_steps=True
        )
    result = evaluate_command_tracking_gate(rollout.episodes, thresholds, command_config=command)
    return rollout, result, thresholds


def _statue(env, *, stand):
    zero = np.zeros(env.action_space.shape, np.float32)
    return lambda: zero


def _reference(env, *, stand):
    walker = ReferenceWalk(env.model, env.data, walking=not stand)
    return lambda: walker.action(env.data)


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


@pytest.fixture(scope="module")
def reference_panel():
    return _roll(_matched(_walk()), cruise=2, stand=1, controller=_reference)


def test_the_walk_declares_the_stride_at_half_cruise_as_its_tracking_window():
    """1.31 s, 131 control steps: the reference's 1.00 s stride at 0.345 m/s scaled to 0.5 x cruise (0.175 m/s) by
    T ~ v^-0.4, within 1.5 x the reference stride (decision D-D64)."""
    command = _walk()["env_kwargs"]["command_config"]
    dt = 0.01
    assert command.tracking_window_s == 1.31
    assert tracking_window_steps(command.tracking_window_s, dt) == 131
    assert PERIOD == 1.0
    slowest = 0.5 * command.cruise_speed
    assert command.speed_range[0] == pytest.approx(slowest)
    stride = PERIOD * (slowest / REFERENCE_SPEED) ** -0.4
    assert round(stride / dt) * dt == pytest.approx(command.tracking_window_s)
    assert command.tracking_window_s <= 1.5 * PERIOD


def test_the_reference_walk_clears_the_tracking_bars_and_fails_only_heading_settle(reference_panel):
    """Seeds 3042 and 3043 under the matched command: in tolerance after the settle 0.958 (per step 0.135), quality
    0.80 after 2 s (per step 0.14); the scripted start alone fails heading_settle.  The balance controller holds the
    stand seed 3053 (the matched command draws its own stand seeds)."""
    rollout, result, thresholds = reference_panel
    measurement = rollout.measurement.as_dict()
    assert measurement["protocol"] == "command_panel/v2" and measurement["tracking_window_s"] == 1.31
    command = _matched(_walk())["env_kwargs"]["command_config"]
    cruise = [(i, e) for i, e in enumerate(rollout.episodes) if e.family == "cruise"]
    assert [e.seed for _, e in cruise] == [3042, 3043]
    for index, episode in cruise:
        assert episode.full_horizon
        assert episode.tracking_fraction_after_settle >= 0.95 > thresholds.min_cruise_tracking_fraction
        assert 0.99 <= episode.speed_ratio <= 1.01
        assert episode.longest_stop_s == 0.0
        assert not episode.first_event_settled
        assert _reasons(result, index) == {"heading_settle"}
        windowed, per_step = _quality_after_2s(rollout.steps[index], command)
        assert windowed >= 0.78 and per_step <= 0.16
    stand = [(i, e) for i, e in enumerate(rollout.episodes) if e.family == "stand"]
    assert [e.seed for _, e in stand] == [3053]
    assert _reasons(result, stand[0][0]) == set() and stand[0][1].stand_hold_fraction == 1.0
    assert not result.passed


def test_the_statue_falls_and_fails_the_walk_gate():
    """On the committed panel's seeds 3042, 3043 (cruise) and 3052 (stand) the statue falls within 100 steps, earns
    under 0.01 tracking a step, and every episode fails."""
    rollout, result, _ = _roll(_walk(), cruise=2, stand=1, controller=_statue)
    assert not result.passed
    for index, episode in enumerate(rollout.episodes):
        rows = rollout.steps[index]
        assert episode.fall and episode.length < 100
        assert "horizon" in _reasons(result, index)
        if episode.family == "cruise":
            assert np.mean([row["reward_tracking"] for row in rows]) < 0.01
