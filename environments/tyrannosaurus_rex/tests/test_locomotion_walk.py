"""The T. rex walk's stride-averaged tracking (decision D-D64) on the real plant, through the walk gate.

At a moving command the tracking term, the metrics and the gate read the trailing mean of the walk's
``tracking_window_s`` (1.12 s), so the reference walk's within-stride sway, which is not a tracking error, cancels
over a stride.  Measured on 2026-11-26 (physics r12) with ``roll_command_panel`` and
``evaluate_command_tracking_gate`` on the walk panel (plan from 3042, 20 cruise and 5 stand episodes):

- The reference walk (``scripts/reference_walk.py``) under a command matched to its 0.633 m/s (heading offset 0):
  tracking quality after 2 s 0.843 an episode (0.131 per step), in tolerance after the settle 0.888 to 0.953, mean
  0.942 (per step 0.101 to 0.124), speed ratio 1.000 to 1.007, longest stop 0 s, 2.422 a cruise step (1.002 per
  step).  It clears the tracking fraction and the speed ratio on 20/20 and fails ``heading_settle`` on 20/20, its
  only failing bar.  On its scripted start a centred stride mean of its speed reaches the speed tolerance at about
  1.95 s, so the gate's trailing window first reads in tolerance at 2.51-2.54 s (seeds 3042-3045), after the
  2.12 s allowance (1 s + 0 + the 1.12 s window); the per-step speed first reads within tolerance at 1.54-1.62 s,
  but sways.  With a quicker start (``settle`` 0.05 s, ``ramp_cycles`` 0.3; 10 cruise episodes) it settles on 10/10
  and every episode passes (in tolerance after the settle 0.951 to 1.000).
- The statue (the zero action) on the committed panel: 0.0021 tracking a step, 0.801 a cruise step, in tolerance
  after the settle 0.000 and 0/20 cruise successes: it FAILS (heading_settle, the tracking fraction, the speed
  ratio and the gait clause on every cruise episode); its stand episodes pass (5/5), exactly as per step.

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
from environments.tyrannosaurus_rex.scripts.reference_walk import ReferenceWalk

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
#: The reference walk's measured speed along its heading (its nominal stride over period, 0.66 m / 1.04 s, is 0.635).
REFERENCE_SPEED = 0.633
#: The gait module's leg length on this plant (the walk TOML's flight-cap comment).
LEG_LENGTH_M = 0.917


def _walk():
    return load_stage_config(TYRANNOSAURUS_REX, "locomotion")


def _matched(stage):
    """The stage with the command matched to the reference walk: speed_range (0.633, 0.633), heading offset 0."""
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
            TYRANNOSAURUS_REX, stage, predict=Predict(), thresholds=thresholds, plan=plan, keep_steps=True
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
    errors_v = [
        math.hypot(row["actual_v_x"] - row["command_v_x"], row["actual_v_y"] - row["command_v_y"]) for row in late
    ]
    errors_yaw = [row["actual_yaw_rate"] - row["command_yaw_rate"] for row in late]
    per_step = float(
        np.mean(
            [
                math.exp(-((ev / parameters["velocity_sigma"]) ** 2) - (eyaw / parameters["yaw_rate_sigma"]) ** 2)
                for ev, eyaw in zip(errors_v, errors_yaw)
            ]
        )
    )
    return windowed, per_step


@pytest.fixture(scope="module")
def reference_panel():
    return _roll(_matched(_walk()), cruise=3, stand=1, controller=_reference)


@pytest.fixture(scope="module")
def statue_panel():
    return _roll(_walk(), cruise=2, stand=1, controller=_statue)


def test_the_walk_declares_the_stride_at_half_cruise_as_its_tracking_window():
    """1.12 s, 112 control steps: the reference's 1.04 s stride at 0.633 m/s scaled to 0.5 x cruise (0.525 m/s) by
    T ~ v^-0.4, within 1.5 x the reference stride (decision D-D64)."""
    stage = _walk()
    command = stage["env_kwargs"]["command_config"]
    dt = 0.01
    assert command.tracking_window_s == 1.12
    assert tracking_window_steps(command.tracking_window_s, dt) == 112
    assert ReferenceWalk.period == 1.04
    assert ReferenceWalk.stride / ReferenceWalk.period == pytest.approx(REFERENCE_SPEED, abs=0.002)
    slowest = 0.5 * command.cruise_speed
    assert command.speed_range[0] == pytest.approx(slowest)
    stride = ReferenceWalk.period * (slowest / REFERENCE_SPEED) ** -0.4
    assert round(stride / dt) * dt == pytest.approx(command.tracking_window_s)
    assert command.tracking_window_s <= 1.5 * ReferenceWalk.period
    # The dimensionless stride the Compsognathus windows are scaled from: T sqrt(g / L) = 3.40 at Froude 0.0445.
    assert ReferenceWalk.period * math.sqrt(9.81 / LEG_LENGTH_M) == pytest.approx(3.40, abs=0.005)
    assert REFERENCE_SPEED**2 / (9.81 * LEG_LENGTH_M) == pytest.approx(0.0445, abs=0.0001)


def test_the_reference_walk_clears_the_tracking_bars_and_fails_only_heading_settle(reference_panel):
    """Seeds 3042-3044 under the matched command: in tolerance after the settle 0.93-0.95 (per step about 0.1),
    quality 0.81-0.87 after 2 s (per step 0.12-0.14); the scripted start alone fails heading_settle."""
    rollout, result, thresholds = reference_panel
    measurement = rollout.measurement.as_dict()
    assert measurement["protocol"] == "command_panel/v2" and measurement["tracking_window_s"] == 1.12
    command = _matched(_walk())["env_kwargs"]["command_config"]
    cruise = [(i, e) for i, e in enumerate(rollout.episodes) if e.family == "cruise"]
    assert [e.seed for _, e in cruise] == [3042, 3043, 3044]
    for index, episode in cruise:
        assert episode.full_horizon
        assert episode.tracking_fraction_after_settle >= 0.90 > thresholds.min_cruise_tracking_fraction
        assert 0.99 <= episode.speed_ratio <= 1.02
        assert episode.longest_stop_s == 0.0
        assert not episode.first_event_settled
        assert _reasons(result, index) == {"heading_settle"}
        windowed, per_step = _quality_after_2s(rollout.steps[index], command)
        assert windowed >= 0.78 and per_step <= 0.16
    stand = [(i, e) for i, e in enumerate(rollout.episodes) if e.family == "stand"]
    assert [e.seed for _, e in stand] == [3053]
    assert _reasons(result, stand[0][0]) == set() and stand[0][1].stand_hold_fraction == 1.0
    assert not result.passed


def test_a_quicker_start_settles_and_passes_every_episode(monkeypatch):
    """settle 0.05 s and ramp_cycles 0.3 reach speed tolerance well inside the window-fill allowance: heading_settle is
    attainable on this plant; the panel fails only for its size (survival_lcb at one episode)."""
    monkeypatch.setattr(ReferenceWalk, "settle", 0.05)
    monkeypatch.setattr(ReferenceWalk, "ramp_cycles", 0.3)
    rollout, result, _ = _roll(_matched(_walk()), cruise=1, stand=1, controller=_reference)
    cruise = rollout.episodes[0]
    assert cruise.family == "cruise" and cruise.seed == 3042
    assert cruise.first_event_settled and cruise.tracking_fraction_after_settle >= 0.95
    assert result.episode_reasons == ((), ())
    assert all("lcb" in failure for failure in result.failures)


def test_the_statue_fails_the_walk_gate(statue_panel):
    """On the committed panel's seeds 3042, 3043 (cruise) and 3052 (stand): no cruise step in tolerance, under 0.01
    tracking a step, every cruise episode failing the tracking, speed and gait bars; the stand passes."""
    rollout, result, _ = statue_panel
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
