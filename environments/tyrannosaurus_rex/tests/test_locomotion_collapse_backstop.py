"""The T. rex walk's collapse backstop under the live command: a floor above the stander (decisions D-D62, D-D64).

The walk warm-starts from a quiet stance, and the evaluation env runs the end-state command from step 0 (the
cruise ramp swaps only the training workers), so its first evaluations stand at the statue's level until it walks:
the shape that stopped the dibothrosuchus run ``20260923_020654`` at 1.45M.  Under the live walk an evaluation is
also a mix of the two command families: each of its 30 episodes is a stand with probability ``stop_probability``
(0.15).  Since decision D-D62 (20) the evaluation env is reseeded with the run's evaluation seed before every
evaluation, so a run has one stand count k, fixed by that seed (4 at the default seed 42), and a stander's level
holds; the tests that draw k afresh at every evaluation keep those per-evaluation draws as a superset of the real
variance.

With the stride-averaged tracking of decision D-D64 a walk as steady as the reference walk evaluates above every
stander's mix up to 10 stands of 30 (the per-step tracking put it under the 9-stand level), so the D-D62 (15) rule
now places the floor (``collapse_peak_floor_fraction`` x ``collapse_peak_floor_reference``, 2.16 x 1248.7 = 2697.2)
between a stander's 9-stand and 10-stand evaluations, as on the velociraptor walk: a standing plateau arms the
backstop only on the 0.97% of seeds that draw k >= 10, where a level that holds never reads as a collapse, while a
walk that tracks its command clears it.

The statue's family rewards below were measured on the committed walk with the zero action (physics r12,
2026-11-26, with the tracking window): the 37 cruise episodes of seeds 3042-3081 (``zero_action_baseline.score``),
800.06 +/- 4.15 (796.48 to 817.33; 800.24 +/- 4.64 per step), and the first 10 stand seeds from 3042 (3052, 3060,
3074, 3082, 3112, 3117, 3120, 3121, 3127, 3130), 6791.35 +/- 65.83 (6648.85 to 6874.15; a stop is scored per step,
so the stand family is unchanged).  The real-plant tests re-measure one episode of each family and one of the
reference walk.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.shared.config import build_env, load_stage_config
from environments.shared.curriculum.early_stopping import EvalCollapseEarlyStopCallback, collapse_settings_from_config
from environments.shared.model_id import ModelId
from environments.shared.tests import test_curriculum_early_stopping as early_stopping_tests
from environments.tyrannosaurus_rex.scripts.reference_walk import ReferenceWalk

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")

#: The statue's episode reward on the committed walk, per family: mean and standard deviation.
STATUE_CRUISE = (800.06, 4.15)
STATUE_STAND = (6791.35, 65.83)
#: Episodes per in-training evaluation (train_base._DEFAULT_EVAL_EPISODES) and its spacing in steps.
EVAL_EPISODES = 30
EVAL_FREQ = 50_000
#: The most stand episodes an evaluation of a stander may hold and still read below the floor.
STANDS_BELOW_THE_FLOOR = 9

#: Run 20260923_020654's locomotion series (the dibothrosuchus) and the standing plateau it was scaled from.
RUN_20260923 = early_stopping_tests.TestDibothrosuchusElaphrosAndBrachiosaurusAltithoraxWarmups.LOCOMOTION_20260923
RUN_20260923_PLATEAU = 2246.9


def _walk():
    return load_stage_config(TYRANNOSAURUS_REX, "locomotion")


def _settings(**overrides):
    settings = collapse_settings_from_config(dict(_walk()["curriculum_kwargs"]))
    settings.update(overrides)
    return settings


def _stander_evaluation(stands: int, cruise: float = STATUE_CRUISE[0], stand: float = STATUE_STAND[0]) -> float:
    """The mean reward of a statue evaluation holding *stands* stand episodes of its EVAL_EPISODES."""
    return cruise + (stand - cruise) * stands / EVAL_EPISODES


def _drive(trace, settings):
    """Feed *trace* (one mean reward per evaluation, EVAL_FREQ apart) to the backstop; the step it stops at, or None."""
    cb = object.__new__(EvalCollapseEarlyStopCallback)
    cb.eval_callback = MagicMock()
    cb._last_seen_n_evals = 0
    cb._peak_score = float("-inf")
    cb._consecutive_drops = 0
    for key in ("min_evals", "patience", "drop_fraction", "smoothing_window", "peak_floor", "peak_warmup_timesteps"):
        setattr(cb, key, settings[key])
    for index in range(len(trace)):
        cb.eval_callback.evaluations_results = [[value] for value in trace[: index + 1]]
        cb.eval_callback.evaluations_timesteps = [(j + 1) * EVAL_FREQ for j in range(index + 1)]
        cb.num_timesteps = (index + 1) * EVAL_FREQ
        if cb._on_step() is False:
            return cb.num_timesteps, cb
    return None, cb


def _mixed_evaluation(rng, stop_probability, cruise=STATUE_CRUISE):
    """One evaluation's mean reward: its stand episodes drawn, each family's reward drawn (stands at the statue's)."""
    stands = int(rng.binomial(EVAL_EPISODES, stop_probability))
    rewards = np.concatenate([rng.normal(*STATUE_STAND, size=stands), rng.normal(*cruise, size=EVAL_EPISODES - stands)])
    return float(rewards.mean())


def _statue_episode(env, seed):
    """The zero action's episode from reset(seed): (stand?, reward, full horizon?)."""
    raw = env.unwrapped
    env.reset(seed=seed)
    stand = not np.any(raw._command)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    total, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
    return stand, total, truncated and not terminated


def test_the_floor_is_the_relative_pair_above_the_stander_and_the_cruise_statue():
    """2.16 x 1248.7 = 2697.2, the midpoint of a stander's evaluations with 9 stands (2597.4) and 10 (2797.2)."""
    curriculum = _walk()["curriculum_kwargs"]
    settings = _settings()
    assert "collapse_peak_floor" not in curriculum
    assert curriculum["collapse_peak_floor_reference"] == 1248.7
    assert curriculum["collapse_peak_floor_fraction"] == 2.16
    assert settings["peak_floor"] == pytest.approx(2.16 * 1248.7)
    assert _stander_evaluation(STANDS_BELOW_THE_FLOOR) < settings["peak_floor"]
    assert _stander_evaluation(STANDS_BELOW_THE_FLOOR + 1) >= settings["peak_floor"]
    midpoint = 0.5 * (_stander_evaluation(STANDS_BELOW_THE_FLOOR) + _stander_evaluation(STANDS_BELOW_THE_FLOOR + 1))
    assert settings["peak_floor"] == pytest.approx(midpoint, abs=1.0)
    # The 40-episode statue reference is the family mix at the 3 stand episodes seeds 3042-3081 hold.
    assert 37 / 40 * STATUE_CRUISE[0] + 3 / 40 * 6782.01 == pytest.approx(1248.7, abs=0.05)
    # The statue's cruise episodes alone come to under a third of it.
    assert STATUE_CRUISE[0] < 0.30 * settings["peak_floor"]


def test_the_walk_declares_the_d_b5_peak_warmup():
    """warmup_timesteps plus the cruise ramp (300k + 2M), within half the budget."""
    walk = _walk()
    curriculum = walk["curriculum_kwargs"]
    ramp = walk["env_kwargs"]["command_config"].cruise_ramp_timesteps
    settings = _settings()
    assert ramp == 2_000_000 and "ramp_timesteps" not in curriculum
    assert settings["peak_warmup_timesteps"] == curriculum["warmup_timesteps"] + ramp == 2_300_000
    assert settings["peak_warmup_timesteps"] <= 0.5 * curriculum["timesteps"]


def test_a_stander_reaches_the_floor_only_by_a_median_of_ten_stand_evaluations():
    """The binomial tail of 10 or more stands, and the union bound over the windows past the warm-up.

    With one k per run, fixed by its evaluation seed (decision D-D62 (20)), a stander arms the floor on the seeds whose
    k reaches 10: the tail, 0.97% of seeds.  Under the superset that draws k afresh at every evaluation, a window's
    median reaches the floor only when 3 of its 5 evaluations hold 10 or more stands, each with that tail
    probability, so a stander arms on well under 1% of runs (0.06% of 200,000 simulated runs).
    """
    walk = _walk()
    stop_probability = walk["env_kwargs"]["command_config"].stop_probability
    settings = _settings()
    tail = sum(
        math.comb(EVAL_EPISODES, k) * stop_probability**k * (1 - stop_probability) ** (EVAL_EPISODES - k)
        for k in range(STANDS_BELOW_THE_FLOOR + 1, EVAL_EPISODES + 1)
    )
    window = settings["smoothing_window"]
    median_tail = sum(
        math.comb(window, m) * tail**m * (1 - tail) ** (window - m) for m in range(window // 2 + 1, window + 1)
    )
    evaluations = walk["curriculum_kwargs"]["timesteps"] // EVAL_FREQ
    eligible = evaluations - int(settings["peak_warmup_timesteps"] // EVAL_FREQ) + 1
    windows = eligible - window + 1
    assert tail == pytest.approx(0.0097, abs=0.0001)
    assert windows * median_tail < 0.002


@pytest.mark.parametrize("stands", [4, STANDS_BELOW_THE_FLOOR + 1])
def test_a_stander_at_its_runs_fixed_stand_count_never_stops(stands):
    """A run's stand count k is fixed by its evaluation seed (decision D-D62 (20)), so a stander's level holds.

    At the default seed's k = 4 the level stays under the floor; at k = 10 (a seed in about 1%) it arms the backstop,
    and holding steady it never reads as a collapse, with or without the warm-up.
    """
    from environments.shared.curriculum.command_evaluation import command_evaluation_plan
    from environments.shared.train_base import eval_env_seed

    walk = _walk()
    plan = command_evaluation_plan(walk["env_kwargs"]["command_config"], eval_env_seed(42), EVAL_EPISODES)
    assert plan.family_counts()["stand"] == 4
    evaluations = walk["curriculum_kwargs"]["timesteps"] // EVAL_FREQ
    trace = [_stander_evaluation(stands)] * evaluations
    for settings in (_settings(), _settings(peak_warmup_timesteps=0.0)):
        stopped_at, cb = _drive(trace, settings)
        assert stopped_at is None
        assert cb._peak_score == pytest.approx(trace[0])
        assert (cb._peak_score >= settings["peak_floor"]) == (stands > STANDS_BELOW_THE_FLOOR)


def test_a_standing_walk_never_arms_the_backstop():
    """A statue-level policy over the whole 8M budget, each evaluation's stand count drawn: never armed, never stopped.

    Drawing k afresh at every evaluation is a superset of the real variance (one k per run, fixed by its evaluation
    seed, decision D-D62 (20)).
    """
    walk = _walk()
    stop_probability = walk["env_kwargs"]["command_config"].stop_probability
    evaluations = walk["curriculum_kwargs"]["timesteps"] // EVAL_FREQ
    rng = np.random.default_rng(3042)
    for _ in range(20):
        trace = [_mixed_evaluation(rng, stop_probability) for _ in range(evaluations)]
        for settings in (_settings(), _settings(peak_warmup_timesteps=0.0)):
            stopped_at, cb = _drive(trace, settings)
            assert stopped_at is None and not cb._peak_score >= settings["peak_floor"]


def test_the_scaled_standing_plateau_never_arms_with_or_without_the_warmup():
    """Run 20260923_020654's locomotion series scaled to this walk's 1248.7 statue peaks at the statue's level, under
    the floor, so the floor alone now holds it; under the per-step walk's 0.45 x 1248.9 floor it stopped at 1.45M
    without the warm-up, which alone kept it running (decision D-D62)."""
    trace = [value * 1248.7 / RUN_20260923_PLATEAU for value in RUN_20260923]
    for settings in (_settings(), _settings(peak_warmup_timesteps=0.0)):
        stopped_at, cb = _drive(trace, settings)
        assert stopped_at is None and not cb._peak_score >= settings["peak_floor"]
    stopped_at, _ = _drive(trace, _settings(peak_floor=0.45 * 1248.9, peak_warmup_timesteps=0.0))
    assert stopped_at == 1_450_000
    assert _drive(trace, _settings(peak_floor=0.45 * 1248.9))[0] is None


def test_the_shape_with_the_family_mix_never_arms_with_or_without_the_warmup():
    """The same series as a ratio of the statue's family mix, each evaluation drawing its stand episodes at 0.15."""
    stop_probability = _walk()["env_kwargs"]["command_config"].stop_probability
    rng = np.random.default_rng(20261124)
    for _ in range(20):
        trace = [_mixed_evaluation(rng, stop_probability) * value / RUN_20260923_PLATEAU for value in RUN_20260923]
        for settings in (_settings(), _settings(peak_warmup_timesteps=0.0)):
            stopped_at, cb = _drive(trace, settings)
            assert stopped_at is None and not cb._peak_score >= settings["peak_floor"]


@pytest.fixture(scope="module")
def walk_env():
    env = build_env(TYRANNOSAURUS_REX, "locomotion")
    yield env
    env.close()


@pytest.mark.parametrize(("seed", "family"), [(3042, "cruise"), (3052, "stand")])
def test_the_statue_reads_the_family_rewards_the_floor_is_derived_from(walk_env, seed, family):
    """One episode of each family on the real plant, inside the range its 37 (10) measured episodes span."""
    stand, reward, full_horizon = _statue_episode(walk_env, seed)
    assert full_horizon
    assert stand == (family == "stand")
    low, high = {"cruise": (796.0, 818.0), "stand": (6645.0, 6880.0)}[family]
    assert low <= reward <= high


def test_a_walk_that_tracks_its_command_clears_the_floor_and_its_collapse_still_stops(walk_env):
    """The reference walk under a command matched to its own speed, on the real plant, as the walking estimate.

    Its episode reward (the tracking term at quality about 0.9 once it walks, with the tracking window of decision
    D-D64) gives the cruise family of a walker, 2472 on seed 3042 (1002 a cruise episode per step); with stand episodes
    at the statue's its expected evaluation clears the floor, as does its evaluation at the default seed's 4 stands,
    and a simulated run that stands to 2.5M, walks to 5M and then falls is stopped from a peak above the floor.
    """
    raw = walk_env.unwrapped
    walk_env.reset(seed=3042)
    walker = ReferenceWalk(raw)
    raw.set_direction(0.0, walker.stride / walker.period)
    total, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = walk_env.step(walker.action(raw.data, raw.dt))
        total += float(reward)
    assert truncated and not terminated
    assert 2400.0 <= total <= 2550.0
    stop_probability = _walk()["env_kwargs"]["command_config"].stop_probability
    settings = _settings()
    expected = (1 - stop_probability) * total + stop_probability * STATUE_STAND[0]
    assert expected > 1.1 * settings["peak_floor"] > _stander_evaluation(STANDS_BELOW_THE_FLOOR + 1)
    assert _stander_evaluation(4, cruise=total) > settings["peak_floor"]
    # The stride-averaged tracking is what lifts it: the walker clears the floor at a cruise-episode reward of
    # (floor - 0.15 x stand) / 0.85 = 1975, over the 1002 it read per step.
    clearing = (settings["peak_floor"] - stop_probability * STATUE_STAND[0]) / (1 - stop_probability)
    assert 1002.2 < clearing < total

    rng = np.random.default_rng(2042)
    standing = [_mixed_evaluation(rng, stop_probability) for _ in range(50)]
    walking = [_mixed_evaluation(rng, stop_probability, cruise=(total, 50.0)) for _ in range(50)]
    fallen = [value * RUN_20260923[-1] / RUN_20260923_PLATEAU for value in walking[:20]]
    stopped_at, cb = _drive(standing + walking + fallen, settings)
    assert stopped_at is not None and stopped_at > 5_000_000
    assert cb._peak_score > settings["peak_floor"]
