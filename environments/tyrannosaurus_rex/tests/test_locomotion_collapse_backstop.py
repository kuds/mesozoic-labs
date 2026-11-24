"""The T. rex walk's collapse backstop under the live command (decision D-D62): the D-B5 peak warm-up.

The walk warm-starts from a quiet stance, and the evaluation env runs the end-state command from step 0 (the
cruise ramp swaps only the training workers), so its first evaluations stand at the statue's level until it walks:
the shape that stopped the dibothrosuchus run ``20260923_020654`` at 1.45M.  Under the live walk an evaluation is
also a mix of the two command families.  Since decision D-D62 (20) the evaluation env is reseeded with the run's
evaluation seed before every evaluation, so a run has one stand count k of its 30 episodes, fixed by that seed (4 at
the default seed 42), and a stander's level holds; the tests that draw k afresh at every evaluation keep those
per-evaluation draws as a superset of the real variance.  The statue's family rewards below were measured on the
committed walk with the zero action (``zero_action_baseline.score`` on seeds 3042-3081 for the 37 cruise episodes,
the first 10 stand seeds from 3042 for the stand family; physics r12, 2026-11-24).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.shared.config import load_stage_config
from environments.shared.curriculum.early_stopping import EvalCollapseEarlyStopCallback, collapse_settings_from_config
from environments.shared.model_id import ModelId
from environments.shared.tests import test_curriculum_early_stopping as early_stopping_tests

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")

#: The statue's episode reward on the committed walk, per family: mean and standard deviation.
STATUE_CRUISE = (800.24, 4.64)
STATUE_STAND = (6791.35, 65.83)
#: Episodes per in-training evaluation (train_base._DEFAULT_EVAL_EPISODES) and its spacing in steps.
EVAL_EPISODES = 30
EVAL_FREQ = 50_000

#: Run 20260923_020654's locomotion series (the dibothrosuchus) and the standing plateau it was scaled from.
RUN_20260923 = early_stopping_tests.TestDibothrosuchusElaphrosAndBrachiosaurusAltithoraxWarmups.LOCOMOTION_20260923
RUN_20260923_PLATEAU = 2246.9


def _walk():
    return load_stage_config(TYRANNOSAURUS_REX, "locomotion")


def _settings(**overrides):
    settings = collapse_settings_from_config(dict(_walk()["curriculum_kwargs"]))
    settings.update(overrides)
    return settings


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


def _mixed_evaluation(rng, stop_probability):
    """One evaluation's mean reward of the statue: its stand episodes drawn, each family's reward drawn."""
    stands = int(rng.binomial(EVAL_EPISODES, stop_probability))
    rewards = np.concatenate(
        [rng.normal(*STATUE_STAND, size=stands), rng.normal(*STATUE_CRUISE, size=EVAL_EPISODES - stands)]
    )
    return float(rewards.mean())


def test_the_walk_declares_the_d_b5_peak_warmup():
    """warmup_timesteps plus the cruise ramp (300k + 2M), within half the budget; the floor is the relative pair."""
    walk = _walk()
    curriculum = walk["curriculum_kwargs"]
    ramp = walk["env_kwargs"]["command_config"].cruise_ramp_timesteps
    settings = _settings()
    assert ramp == 2_000_000 and "ramp_timesteps" not in curriculum
    assert settings["peak_warmup_timesteps"] == curriculum["warmup_timesteps"] + ramp == 2_300_000
    assert settings["peak_warmup_timesteps"] <= 0.5 * curriculum["timesteps"]
    assert curriculum["collapse_peak_floor_reference"] == 1248.9
    assert settings["peak_floor"] == pytest.approx(0.45 * 1248.9)
    # The floor sits below the cruise statue, so a standing plateau would arm the backstop without the warm-up.
    assert settings["peak_floor"] < STATUE_CRUISE[0]


def test_the_scaled_standing_plateau_stops_without_the_warmup_and_runs_on_behind_it():
    """Run 20260923_020654's locomotion series scaled to this walk's 1248.9 statue, as the velociraptor's test does."""
    trace = [value * 1248.9 / RUN_20260923_PLATEAU for value in RUN_20260923]
    stopped_at, cb = _drive(trace, _settings(peak_warmup_timesteps=0.0))
    assert stopped_at == 1_450_000
    assert cb._peak_score == pytest.approx(1248.9)
    stopped_at, cb = _drive(trace, _settings())
    assert stopped_at is None and not cb._peak_score >= _settings()["peak_floor"]


def test_the_shape_with_the_family_mix_stops_without_the_warmup_and_runs_on_behind_it():
    """The same series as a ratio of the statue's family mix: each evaluation draws its stand episodes at 0.15."""
    stop_probability = _walk()["env_kwargs"]["command_config"].stop_probability
    rng = np.random.default_rng(20261124)
    for _ in range(20):
        trace = [_mixed_evaluation(rng, stop_probability) * value / RUN_20260923_PLATEAU for value in RUN_20260923]
        assert _drive(trace, _settings(peak_warmup_timesteps=0.0))[0] == 1_450_000
        assert _drive(trace, _settings())[0] is None


@pytest.mark.parametrize("stands", [4, 10])
def test_a_stander_at_its_runs_fixed_stand_count_never_stops(stands):
    """A run's stand count k is fixed by its evaluation seed (decision D-D62 (20)), so a stander's level holds: at the
    default seed's k = 4 and at k = 10 it arms the backstop (the floor is under the cruise statue) once the peak may
    be set, and holding steady it never reads as a collapse, with or without the warm-up."""
    from environments.shared.curriculum.command_evaluation import command_evaluation_plan
    from environments.shared.train_base import eval_env_seed

    walk = _walk()
    plan = command_evaluation_plan(walk["env_kwargs"]["command_config"], eval_env_seed(42), EVAL_EPISODES)
    assert plan.family_counts()["stand"] == 4
    evaluations = walk["curriculum_kwargs"]["timesteps"] // EVAL_FREQ
    level = STATUE_CRUISE[0] + (STATUE_STAND[0] - STATUE_CRUISE[0]) * stands / EVAL_EPISODES
    trace = [level] * evaluations
    for settings in (_settings(), _settings(peak_warmup_timesteps=0.0)):
        stopped_at, cb = _drive(trace, settings)
        assert stopped_at is None
        assert cb._peak_score == pytest.approx(level) and level >= settings["peak_floor"]


def test_the_family_mix_alone_never_stops_a_standing_walk():
    """A statue-level policy over the whole 8M budget: the stand count's spread is no collapse, with or without.

    Each evaluation draws its own stand count, a superset of the real variance (one k per run, fixed by its
    evaluation seed, decision D-D62 (20)).
    """
    walk = _walk()
    stop_probability = walk["env_kwargs"]["command_config"].stop_probability
    evaluations = walk["curriculum_kwargs"]["timesteps"] // EVAL_FREQ
    rng = np.random.default_rng(3042)
    for _ in range(10):
        trace = [_mixed_evaluation(rng, stop_probability) for _ in range(evaluations)]
        assert _drive(trace, _settings(peak_warmup_timesteps=0.0))[0] is None
        assert _drive(trace, _settings())[0] is None


def test_a_genuine_collapse_after_the_warmup_still_stops():
    """The warm-up delays the peak; it does not disarm the backstop: a walk that falls after 5M is stopped."""
    stop_probability = _walk()["env_kwargs"]["command_config"].stop_probability
    rng = np.random.default_rng(2042)
    walking_cruise = (3000.0, 50.0)
    standing = [_mixed_evaluation(rng, stop_probability) for _ in range(50)]
    walking = []
    for _ in range(50):
        stands = int(rng.binomial(EVAL_EPISODES, stop_probability))
        rewards = np.concatenate(
            [rng.normal(*STATUE_STAND, size=stands), rng.normal(*walking_cruise, size=EVAL_EPISODES - stands)]
        )
        walking.append(float(rewards.mean()))
    fallen = [value * RUN_20260923[-1] / RUN_20260923_PLATEAU for value in walking[:20]]
    stopped_at, cb = _drive(standing + walking + fallen, _settings())
    assert stopped_at is not None and stopped_at > 5_000_000
    assert cb._peak_score > 2 * _settings()["peak_floor"]
