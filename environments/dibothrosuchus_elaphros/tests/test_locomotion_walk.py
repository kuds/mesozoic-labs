"""The dibothrosuchus walk under the live command (decision D-D62) on the real plant, and its collapse backstop.

The numbers below were measured on the committed locomotion config (the end-state command) with the zero action
and with the species' reference walk (``scripts/reference_walk.py``) on 2026-11-24 (physics r4, MuJoCo 3.10.0):

- The statue earns 2067.64 +/- 272.66 a cruise episode (the 37 cruise episodes of seeds 3042-3081,
  ``zero_action_baseline.score``; 1149.4 to 2157.4), 1913.6 of it from ``gait_symmetry``: the diagonal pairs land
  in the settle's first steps and never lift again, so the alternation ratio holds where the settle left it, 1 on
  most seeds and 0.5 on a few (seed 3042), and the term pays that share of its weight on every step.  A stand
  episode earns 4977.44 +/- 6.88 (the first 10 stand seeds from 3042: 3052, 3060, 3074, 3082, 3112, 3117, 3120,
  3121, 3127, 3130; 4965.1 to 4986.7): the stance's weights, whose sum (2499.3) is the stance statue's 2499.2,
  plus the tracking term at a held stop.
- The reference walk, unsteered at its own 0.158 m/s, on the walk panel's 20 cruise episodes (plan from 3042):
  every quadruped gait bar the walk declares clears on all 20 (flight 0, limb duty 0.64-0.66, pair alternation 1.0,
  lead swaps 2.04 a stride), but it walks under half the slowest cruise command (0.45 m/s): its speed ratio reads
  0.17-0.33, so min_cruise_speed_ratio refuses every episode.  The same walk rescaled to 0.35 m/s or faster falls
  within 2 s.

Under the live walk an in-training evaluation is a mix of the two families: its 30 episodes are each a stand with
probability ``stop_probability`` (0.15).  Since decision D-D62 (20) the evaluation env is reseeded with the run's
evaluation seed before every evaluation, so a run has one stand count k, fixed by that seed (4 at the default seed
42), and a stander's level holds; the tests that draw k afresh at every evaluation keep those per-evaluation draws as
a superset of the real variance.  The walk keeps the absolute ``collapse_peak_floor`` behind the D-B5 peak warm-up: a
standing plateau arms the backstop once past the warm-up, and the family mix alone never stops it.  Run
20260923_020654 itself, on this config, is replayed in
``environments/shared/tests/test_curriculum_early_stopping.py``.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

import environments.shared.species_registry as species_registry
from environments.dibothrosuchus_elaphros.scripts.reference_walk import ReferenceWalk
from environments.shared.command_tracking_panel import plan_command_panel, roll_command_panel
from environments.shared.config import build_env, load_stage_config
from environments.shared.curriculum.command_tracking_gate import (
    classify_cruise_episode,
    gait_failures,
    reason_key,
    thresholds_from_curriculum,
)
from environments.shared.curriculum.early_stopping import EvalCollapseEarlyStopCallback, collapse_settings_from_config
from environments.shared.model_id import ModelId
from environments.shared.tests import test_curriculum_early_stopping as early_stopping_tests

DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")

#: The statue's episode reward on the committed walk, per family: mean and standard deviation.
STATUE_CRUISE = (2067.64, 272.66)
STATUE_STAND = (4977.44, 6.88)
#: The reference walk's episode reward under a command matched to its own speed (0.157 m/s, offset 0): the
#: walking level of the genuine-collapse replay.
REFERENCE_WALK_MATCHED = (3957.0, 135.0)
#: Episodes per in-training evaluation (train_base._DEFAULT_EVAL_EPISODES) and its spacing in steps.
EVAL_EPISODES = 30
EVAL_FREQ = 50_000

#: Run 20260923_020654's locomotion series (this model's forward walk) and its standing plateau.
RUN_20260923 = early_stopping_tests.TestDibothrosuchusElaphrosAndBrachiosaurusAltithoraxWarmups.LOCOMOTION_20260923
RUN_20260923_PLATEAU = 2246.9


def _walk():
    return load_stage_config(DIBOTHROSUCHUS_ELAPHROS, "locomotion")


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


def _mixed_evaluation(rng, stop_probability, cruise=STATUE_CRUISE):
    """One evaluation's mean reward: its stand episodes drawn, each family's episode reward drawn."""
    stands = int(rng.binomial(EVAL_EPISODES, stop_probability))
    rewards = np.concatenate([rng.normal(*STATUE_STAND, size=stands), rng.normal(*cruise, size=EVAL_EPISODES - stands)])
    return float(rewards.mean())


# ── the collapse backstop ────────────────────────────────────────────────────


def test_the_walk_keeps_the_absolute_floor_behind_the_d_b5_peak_warmup():
    """warmup_timesteps plus the cruise ramp (800k + 2M); the floor stays absolute, far under the statue's level."""
    walk = _walk()
    curriculum = walk["curriculum_kwargs"]
    ramp = walk["env_kwargs"]["command_config"].cruise_ramp_timesteps
    settings = _settings()
    assert ramp == 2_000_000 and "ramp_timesteps" not in curriculum
    assert settings["peak_warmup_timesteps"] == curriculum["warmup_timesteps"] + ramp == 2_800_000
    assert settings["peak_warmup_timesteps"] <= 0.25 * curriculum["timesteps"]
    assert curriculum["collapse_peak_floor"] == 100.0 and "collapse_peak_floor_fraction" not in curriculum
    # A statue-level evaluation reads at least about its cruise level, 20x the floor: a standing plateau arms the
    # backstop once past the warm-up.
    assert settings["peak_floor"] < 0.05 * STATUE_CRUISE[0]


def test_the_family_mix_alone_never_stops_a_standing_walk():
    """A statue-level policy over the whole 12M budget: the stand count's spread never reads as ten evaluations under
    half the peak, with or without the warm-up.

    Each evaluation draws its own stand count, a superset of the real variance (one k per run, fixed by its
    evaluation seed, decision D-D62 (20)).
    """
    walk = _walk()
    stop_probability = walk["env_kwargs"]["command_config"].stop_probability
    evaluations = walk["curriculum_kwargs"]["timesteps"] // EVAL_FREQ
    rng = np.random.default_rng(3042)
    for _ in range(3):
        trace = [_mixed_evaluation(rng, stop_probability) for _ in range(evaluations)]
        assert _drive(trace, _settings(peak_warmup_timesteps=0.0))[0] is None
        assert _drive(trace, _settings())[0] is None


@pytest.mark.parametrize("stands", [4, 10])
def test_a_stander_at_its_runs_fixed_stand_count_never_stops(stands):
    """A run's stand count k is fixed by its evaluation seed (decision D-D62 (20)), so a stander's level holds: at the
    default seed's k = 4 and at k = 10 it arms the absolute floor once the peak may be set, and holding steady it
    never reads as a collapse, with or without the warm-up."""
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


def test_the_run_20260923_shape_with_the_family_mix_stops_without_the_warmup_and_runs_on_behind_it():
    """The run's series as a ratio of this walk's standing mix (each evaluation draws its stand episodes at 0.15)."""
    stop_probability = _walk()["env_kwargs"]["command_config"].stop_probability
    rng = np.random.default_rng(20261124)
    for _ in range(10):
        trace = [_mixed_evaluation(rng, stop_probability) * value / RUN_20260923_PLATEAU for value in RUN_20260923]
        assert _drive(trace, _settings(peak_warmup_timesteps=0.0))[0] == 1_450_000
        assert _drive(trace, _settings())[0] is None


def test_a_genuine_collapse_after_the_warmup_still_stops():
    """The warm-up delays the peak; it does not disarm the backstop: a walk that falls after 6M is stopped."""
    stop_probability = _walk()["env_kwargs"]["command_config"].stop_probability
    rng = np.random.default_rng(2042)
    standing = [_mixed_evaluation(rng, stop_probability) for _ in range(60)]
    walking = [_mixed_evaluation(rng, stop_probability, cruise=REFERENCE_WALK_MATCHED) for _ in range(60)]
    fallen = [value * RUN_20260923[-1] / RUN_20260923_PLATEAU for value in walking[:20]]
    stopped_at, cb = _drive(standing + walking + fallen, _settings())
    assert stopped_at is not None and stopped_at > 6_000_000
    assert cb._peak_score > 20 * _settings()["peak_floor"]


# ── the real plant ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def walk_env():
    env = build_env(DIBOTHROSUCHUS_ELAPHROS, "locomotion")
    yield env
    env.close()


def _statue_episode(env, seed):
    """The statue's episode from *seed*: whether it is a stand, and its reward and gait_symmetry and alive sums."""
    env.reset(seed=seed)
    stand = not np.any(env.unwrapped._command)
    action = np.zeros(env.action_space.shape, np.float32)
    rewards, symmetry, alive = [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        symmetry.append(float(info["reward_gait_symmetry"]))
        alive.append(float(info["reward_alive"]))
    return stand, math.fsum(rewards), symmetry, alive, truncated and not terminated


@pytest.mark.parametrize(
    ("seed", "family", "ratio"), [(3043, "cruise", 1.0), (3042, "cruise", 0.5), (3052, "stand", 0.0)]
)
def test_the_statue_reads_its_family_rewards(walk_env, seed, family, ratio):
    """One episode of each family, inside the range its measured episodes span.  On a cruise episode the statue
    collects gait_symmetry at the alternation ratio its settle left (the whole weight on 3043, half on 3042) on
    every step after it; on the stand episode the stand block pays the stance's weights (alive 1.5, no
    gait_symmetry)."""
    stand, reward, symmetry, alive, full_horizon = _statue_episode(walk_env, seed)
    raw = walk_env.unwrapped
    assert full_horizon
    assert stand == (family == "stand")
    low, high = {3043: (2130.0, 2160.0), 3042: (1140.0, 1160.0), 3052: (4960.0, 4990.0)}[seed]
    assert low <= reward <= high
    assert symmetry[-1] == pytest.approx(ratio * raw.gait_symmetry_weight)
    if family == "cruise":
        assert math.fsum(symmetry) >= ratio * raw.gait_symmetry_weight * (raw.max_episode_steps - 5)
        assert set(alive) == {raw.alive_bonus}
    else:
        assert math.fsum(symmetry) == 0.0
        assert set(alive) == {raw.command_config.stand_weight_map()["alive_bonus"]}


def test_the_reference_walk_clears_the_gait_clause_and_walks_under_half_the_slowest_command(monkeypatch):
    """The reference walk on the walk panel's cruise seed 3042 (and the statue on its stand seed 3052), through
    roll_command_panel and the gait clause the walk declares, the optional lead-swap bar included: the clause costs
    a genuine lateral-sequence walk nothing.  Its 0.158 m/s is under half the speed range's floor, so the speed
    ratio bar refuses the episode whatever the cruise command."""
    stage = _walk()
    command = stage["env_kwargs"]["command_config"]
    thresholds = thresholds_from_curriculum(
        {**stage["curriculum_kwargs"], "min_cruise_episodes": 1, "min_stand_episodes": 1}
    )
    plan = plan_command_panel(thresholds, command)
    assert plan.seeds == (3042, 3052)
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
            self.walker = ReferenceWalk(env) if np.any(env._command) else None

        def __call__(self, obs):
            env = holder["env"]
            if self.walker is None:
                return np.zeros(env.action_space.shape, np.float32)
            return self.walker.action(env.data, env.dt)

    monkeypatch.setattr(species_registry, "env_class", capturing)
    rollout = roll_command_panel(DIBOTHROSUCHUS_ELAPHROS, stage, predict=Predict(), thresholds=thresholds, plan=plan)
    cruise = next(episode for episode in rollout.episodes if episode.family == "cruise")
    gait = dict(cruise.gait_metrics)
    assert cruise.full_horizon
    assert thresholds.gait.bar("min_lead_swaps_per_stride") == 1.0
    assert gait_failures(cruise, thresholds.gait) == []
    assert 1.95 <= gait["lead_swaps_per_stride"] <= 2.1
    assert gait["flight_fraction"] == 0.0 and gait["pair_alternation_index"] == 1.0
    assert 0.15 <= gait["displacement_speed_mps"] <= 0.17 < thresholds.min_cruise_speed_ratio * command.speed_range[0]
    assert cruise.speed_ratio < thresholds.min_cruise_speed_ratio
    assert "min_cruise_speed_ratio" in {reason_key(reason) for reason in classify_cruise_episode(cruise, thresholds)}
