"""The walk's command distribution on every committed locomotion stage (decision D-D62).

Each model's walk (``configs/<species>/<embodiment>/locomotion.toml``) declares one whole-episode command per
episode, drawn at reset from the env's own command stream: a forward cruise at a speed in ``[0.5, 1] x cruise``
about the spawn heading turned by at most ``initial_heading_offset_max``, or, with ``stop_probability``, an
exactly zero command, the stand.  No turn and no switch comes within the horizon.  The command scaling is the
model's pilot ``follow_direction`` recipe's, so the walk and the follow nodes that widen it share one scaling.

The tests build each env from the committed stage config (``config.build_env``, the kwargs
``train_base.make_env`` passes) and reset it on :data:`SEEDS`, reading the draw off the env itself:

* the declared distribution (every number the walk states, the same on all seven models but the scaling);
* the stand fraction, within an exact two-sided binomial interval of ``stop_probability``, and every stand
  episode an exactly zero command, observed as such;
* the cruise speeds and the heading offsets spanning their ranges, with about half of each in the inner half;
* no command event after the reset's within the horizon, planned on a copy of the episode's controller and
  rolled on the env;
* the cruise ramp's start (``config.ramped(cruise_ramp_start_fraction)``) keeping every seed's family and
  offset and scaling its speed.
"""

from __future__ import annotations

import copy
import math
import tomllib
from functools import cache
from typing import Any

import numpy as np
import pytest

from environments.shared.command_tracking_panel import reset_command_family
from environments.shared.config import build_env, load_stage_config
from environments.shared.curriculum.command_tracking_gate import CRUISE_FAMILY, STAND_FAMILY
from environments.shared.direction_commands import wrap_angle
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.species_names import model_ids

MODEL_IDS = model_ids(backend="stable-baselines3")
#: The reset seeds every model is drawn on: the publication seed family's first 400.
SEEDS = range(3042, 3042 + 400)
#: The walk's declared distribution, the same on every model (decision D-D62).
STOP_PROBABILITY = 0.15
HEADING_OFFSET_MAX = 0.26
TRACKING_WEIGHT = 2.5
CRUISE_RAMP = (0.25, 2_000_000)
#: The two-sided tail probability of the binomial intervals below.  The draws are seeded, so every test is
#: deterministic; the interval says how far a correct schedule's count can sit from its expectation.
TAIL = 1e-4
#: The [commands] fields the walk takes from the model's follow_direction pilot recipe.
RECIPE_SCALING = ("cruise_speed", "speed_scale", "lateral_speed_scale", "yaw_rate_scale")


def _ids(model_id: ModelId) -> str:
    return f"{model_id.species}-{model_id.embodiment}"


def _binomial_interval(trials: int, probability: float, tail: float = TAIL) -> tuple[int, int]:
    """The counts ``[low, high]`` outside which an exact Binomial(trials, probability) puts at most *tail* each side."""
    pmf = [math.comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k) for k in range(trials + 1)]
    cdf = np.cumsum(pmf)
    low = int(np.searchsorted(cdf, tail, side="right"))
    survival = np.cumsum(pmf[::-1])[::-1]
    high = int(max(k for k in range(trials + 1) if survival[k] > tail))
    return low, high


@cache
def _draws(model_id: ModelId) -> dict[str, Any]:
    """Every seed's reset draw on the committed walk, read off the env: family, offset, speed, command, events."""
    env = build_env(model_id, "locomotion")
    raw = env.unwrapped
    horizon_s = raw.max_episode_steps * raw.dt
    rows = []
    try:
        for seed in SEEDS:
            observation, _ = env.reset(seed=seed)
            state = raw._reset_command_state
            planner = copy.deepcopy(raw.direction_controller)
            planner.update(horizon_s, state.current_heading)
            rows.append(
                {
                    "seed": seed,
                    "family": STAND_FAMILY if state.desired_speed == 0.0 else CRUISE_FAMILY,
                    "offset": wrap_angle(state.desired_heading - state.current_heading),
                    "speed": state.desired_speed,
                    "heading_active": state.heading_active,
                    "command": np.array(raw._command, dtype=np.float64),
                    "observed": np.array(observation[-3:], dtype=np.float64),
                    "events": len(planner.events),
                    "next_switch_s": planner._next_switch_s,
                }
            )
    finally:
        env.close()
    return {"rows": rows, "horizon_s": horizon_s}


def _config(model_id: ModelId) -> Any:
    return load_stage_config(model_id, "locomotion")["env_kwargs"]["command_config"]


def _recipe_commands(model_id: ModelId) -> dict[str, Any]:
    path = model_id.path(REPOSITORY_ROOT / "configs") / "behaviors" / "follow_direction.toml"
    with path.open("rb") as stream:
        commands: dict[str, Any] = tomllib.load(stream)["commands"]
    return commands


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_every_walk_declares_the_narrow_distribution(model_id):
    """One distribution on every model: speeds [0.5, 1] x the recipe's cruise, 15% stands, a 0.26 rad offset, no
    turns, a switch interval longer than the horizon, the recipe's scaling, tracking 2.5 and the 0.25 / 2M ramp."""
    stage = load_stage_config(model_id, "locomotion")
    env_kwargs = stage["env_kwargs"]
    config = env_kwargs["command_config"]
    assert env_kwargs["command_mode"] == "heading_and_speed"
    recipe = _recipe_commands(model_id)
    assert {name: getattr(config, name) for name in RECIPE_SCALING} == {name: recipe[name] for name in RECIPE_SCALING}
    assert config.speed_range == pytest.approx((0.5 * config.cruise_speed, config.cruise_speed), abs=1e-12)
    assert config.stop_probability == STOP_PROBABILITY
    assert config.initial_heading_offset_max == HEADING_OFFSET_MAX
    assert config.turn_increment_max == 0.0 and config.switch_jitter_s == 0.0
    assert config.switch_interval_s > _draws(model_id)["horizon_s"]
    assert config.turn_slowdown is True and config.yaw_rate_max == 0.3
    assert 2.0 <= config.tracking_weight <= 2.5 and config.tracking_weight == TRACKING_WEIGHT
    assert (config.cruise_ramp_start_fraction, config.cruise_ramp_timesteps) == CRUISE_RAMP
    assert config.stand_weights, "a walk declares its stand block"


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_about_fifteen_percent_of_episodes_are_whole_episode_stands(model_id):
    """The stand count over the seeds lies in the exact binomial interval of ``stop_probability``, and the panel's
    replica of the reset draw (``reset_command_family``) names every seed's family as the env drew it."""
    rows = _draws(model_id)["rows"]
    stands = sum(row["family"] == STAND_FAMILY for row in rows)
    low, high = _binomial_interval(len(rows), STOP_PROBABILITY)
    assert low <= stands <= high, (stands, len(rows), (low, high))
    config = _config(model_id)
    assert [reset_command_family(config, row["seed"]) for row in rows] == [row["family"] for row in rows]


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_a_stand_episode_is_an_exactly_zero_command(model_id):
    """Speed 0, no heading objective, and the command the env holds and the policy observes is exactly zero."""
    stands = [row for row in _draws(model_id)["rows"] if row["family"] == STAND_FAMILY]
    assert stands
    for row in stands:
        assert row["speed"] == 0.0 and row["heading_active"] is False, row["seed"]
        assert not np.any(row["command"]) and not np.any(row["observed"]), row["seed"]


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_cruise_speeds_span_the_speed_range(model_id):
    """Every cruise speed lies in [0.5, 1] x cruise, the extremes come within a tenth of the range of both ends, and
    the count below the midpoint lies in the binomial interval of a uniform draw's one half."""
    config = _config(model_id)
    low, high = config.speed_range
    speeds = np.array([row["speed"] for row in _draws(model_id)["rows"] if row["family"] == CRUISE_FAMILY])
    assert np.all((speeds >= low) & (speeds <= high))
    width = high - low
    assert speeds.min() < low + 0.1 * width and speeds.max() > high - 0.1 * width
    lower, upper = _binomial_interval(speeds.size, 0.5)
    assert lower <= int(np.sum(speeds < low + 0.5 * width)) <= upper


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_heading_offsets_span_the_offset_range(model_id):
    """Every episode's requested heading is the spawn heading turned by at most 0.26 rad, both signs reach past 0.2,
    and the count inside +/-0.13 lies in the binomial interval of a uniform draw's one half.  A stand episode draws
    its offset too (the offset is drawn before the stop), so its schedule matches the cruise episodes'."""
    offsets = np.array([row["offset"] for row in _draws(model_id)["rows"]])
    assert np.all(np.abs(offsets) <= HEADING_OFFSET_MAX + 1e-9)
    assert offsets.min() < -0.2 and offsets.max() > 0.2
    lower, upper = _binomial_interval(offsets.size, 0.5)
    assert lower <= int(np.sum(np.abs(offsets) < 0.5 * HEADING_OFFSET_MAX)) <= upper


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_no_command_event_follows_the_reset_within_the_horizon(model_id):
    """On a copy of each episode's controller advanced to the horizon, the reset's event is the only one, and the
    next scheduled switch lies beyond the horizon: the walk profile's whole-episode command (D-D61 (2))."""
    draws = _draws(model_id)
    for row in draws["rows"]:
        assert row["events"] == 1, row["seed"]
        assert row["next_switch_s"] > draws["horizon_s"], row["seed"]


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
@pytest.mark.parametrize("family", [CRUISE_FAMILY, STAND_FAMILY])
def test_a_rolled_episode_keeps_its_reset_request(model_id, family):
    """On the env itself: each step of an episode (zero action, 200 steps or to its end) reports the reset's event
    and the reset's requested heading and speed; the stand's command stays exactly zero."""
    seed = next(row["seed"] for row in _draws(model_id)["rows"] if row["family"] == family)
    env = build_env(model_id, "locomotion")
    raw = env.unwrapped
    try:
        env.reset(seed=seed)
        reset = raw._reset_command_state
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert info["command_event_id"] == 0
            assert (info["desired_heading"], info["desired_speed"]) == (reset.desired_heading, reset.desired_speed)
            if family == STAND_FAMILY:
                assert not np.any(raw._command)
            if terminated or truncated:
                break
    finally:
        env.close()


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_the_cruise_ramp_start_keeps_every_family_and_scales_the_speed(model_id):
    """Swapped in as the cruise ramp swaps it (``set_command_config``, effective at the next reset), the ramp's
    first config draws the same family and heading offset on every seed and the start fraction of each cruise
    speed: the stand fraction does not move along the ramp (D-D62 (5))."""
    config = _config(model_id)
    fraction = config.cruise_ramp_start_fraction
    rows = _draws(model_id)["rows"][:100]
    env = build_env(model_id, "locomotion")
    raw = env.unwrapped
    try:
        raw.set_command_config(config.ramped(fraction))
        for row in rows:
            env.reset(seed=row["seed"])
            state = raw._reset_command_state
            assert (STAND_FAMILY if state.desired_speed == 0.0 else CRUISE_FAMILY) == row["family"]
            assert wrap_angle(state.desired_heading - state.current_heading) == pytest.approx(row["offset"], abs=1e-12)
            assert state.desired_speed == pytest.approx(fraction * row["speed"], rel=1e-12, abs=1e-15)
    finally:
        env.close()
