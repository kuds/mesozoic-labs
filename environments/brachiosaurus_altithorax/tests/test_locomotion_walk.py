"""The brachiosaurus walk under the live command (decision D-D62) on the real plant, and its collapse backstop.

The numbers below were measured on the committed locomotion config (the end-state command) with the zero action
and with the species' reference walk (``scripts/reference_walk.py``) on 2026-11-24 (physics r6, MuJoCo 3.10.0):

- The statue earns 2214.82 +/- 10.16 a cruise episode (the 37 cruise episodes of seeds 3042-3081,
  ``zero_action_baseline.score``; 2192.0 to 2236.2), 2195.5 of it from ``gait_symmetry``: both diagonal pairs land
  on the settle's first steps and never lift again, so the alternation ratio holds at 1 and the term pays its whole
  weight on every step.  A stand episode earns 4149.84 +/- 28.83 (the first 10 stand seeds from 3042: 3052, 3060,
  3074, 3082, 3112, 3117, 3120, 3121, 3127, 3130; 4100.6 to 4190.4), the stance's weights plus the tracking term
  at a held stop.
- The reference walk, unsteered at its own 0.35 m/s, on the walk panel's 20 cruise episodes (plan from 3042): every
  quadruped gait bar the walk declares clears on all 20 (flight 0, limb duty 0.59-0.61, pair alternation 0.79-0.92,
  lead swaps 1.67-1.82 a stride), and it earns 2033 a cruise episode on average, under the statue's 2215: it walks
  slower than most cruise commands, and its lateral-sequence footfalls collect 0.71 of ``gait_symmetry`` where the
  statue collects all of it.

Under the live walk an in-training evaluation is a mix of the two families: its 30 episodes are each a stand with
probability ``stop_probability`` (0.15), the stand count drawn per evaluation (the evaluation env is seeded once).
The walk keeps the absolute ``collapse_peak_floor`` behind the D-B5 peak warm-up: a standing plateau arms the
backstop once past the warm-up, and the family mix alone never stops it.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

import environments.shared.species_registry as species_registry
from environments.brachiosaurus_altithorax.scripts.reference_walk import ReferenceWalk
from environments.shared.command_tracking_panel import plan_command_panel, roll_command_panel
from environments.shared.config import build_env, load_stage_config
from environments.shared.curriculum.command_tracking_gate import gait_failures, thresholds_from_curriculum
from environments.shared.curriculum.early_stopping import EvalCollapseEarlyStopCallback, collapse_settings_from_config
from environments.shared.model_id import ModelId
from environments.shared.tests import test_curriculum_early_stopping as early_stopping_tests

BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")

#: The statue's episode reward on the committed walk, per family: mean and standard deviation.
STATUE_CRUISE = (2214.82, 10.16)
STATUE_STAND = (4149.84, 28.83)
#: Episodes per in-training evaluation (train_base._DEFAULT_EVAL_EPISODES) and its spacing in steps.
EVAL_EPISODES = 30
EVAL_FREQ = 50_000

#: Run 20260923_020654's locomotion series (the dibothrosuchus forward walk) and its standing plateau.
RUN_20260923 = early_stopping_tests.TestDibothrosuchusElaphrosAndBrachiosaurusAltithoraxWarmups.LOCOMOTION_20260923
RUN_20260923_PLATEAU = 2246.9


def _walk():
    return load_stage_config(BRACHIOSAURUS_ALTITHORAX, "locomotion")


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
    """warmup_timesteps plus the cruise ramp (1M + 2M); the floor stays absolute, far under the statue's cruise level."""
    walk = _walk()
    curriculum = walk["curriculum_kwargs"]
    ramp = walk["env_kwargs"]["command_config"].cruise_ramp_timesteps
    settings = _settings()
    assert ramp == 2_000_000 and "ramp_timesteps" not in curriculum
    assert settings["peak_warmup_timesteps"] == curriculum["warmup_timesteps"] + ramp == 3_000_000
    assert settings["peak_warmup_timesteps"] <= 0.25 * curriculum["timesteps"]
    assert curriculum["collapse_peak_floor"] == 100.0 and "collapse_peak_floor_fraction" not in curriculum
    # A statue-level evaluation reads at least its cruise level, 22x the floor: a standing plateau arms the backstop
    # once past the warm-up.
    assert settings["peak_floor"] < 0.05 * STATUE_CRUISE[0]


def test_the_family_mix_alone_never_stops_a_standing_walk():
    """A statue-level policy over the whole 16M budget: an evaluation of no stand episodes reads 0.53 of an all-stand
    one, so the stand count's spread never reads as a drop under half the peak, with or without the warm-up."""
    walk = _walk()
    stop_probability = walk["env_kwargs"]["command_config"].stop_probability
    evaluations = walk["curriculum_kwargs"]["timesteps"] // EVAL_FREQ
    assert STATUE_CRUISE[0] / STATUE_STAND[0] > 1.0 - _settings()["drop_fraction"]
    rng = np.random.default_rng(3042)
    for _ in range(3):
        trace = [_mixed_evaluation(rng, stop_probability) for _ in range(evaluations)]
        assert _drive(trace, _settings(peak_warmup_timesteps=0.0))[0] is None
        assert _drive(trace, _settings())[0] is None


def test_the_run_20260923_shape_stops_without_the_warmup_and_runs_on_behind_it():
    """Stand until the clip release, then lunge and fall: the dibothrosuchus run's series as a ratio of this walk's
    standing mix (each evaluation draws its stand episodes at 0.15) stops at 1.45M under the absolute floor alone."""
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
    walking = [_mixed_evaluation(rng, stop_probability, cruise=(2422.0, 50.0)) for _ in range(60)]
    fallen = [value * RUN_20260923[-1] / RUN_20260923_PLATEAU for value in walking[:20]]
    stopped_at, cb = _drive(standing + walking + fallen, _settings())
    assert stopped_at is not None and stopped_at > 6_000_000
    assert cb._peak_score > 20 * _settings()["peak_floor"]


# ── the real plant ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def walk_env():
    env = build_env(BRACHIOSAURUS_ALTITHORAX, "locomotion")
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
    return stand, math.fsum(rewards), math.fsum(symmetry), alive, truncated and not terminated


@pytest.mark.parametrize(("seed", "family"), [(3042, "cruise"), (3052, "stand")])
def test_the_statue_reads_its_family_rewards(walk_env, seed, family):
    """One episode of each family, inside the range its 37 (10) measured episodes span.  On the cruise episode the
    statue collects gait_symmetry's whole weight on every step after the settle; on the stand episode the stand
    block pays the stance's weights (alive 1.75, no gait_symmetry)."""
    stand, reward, symmetry, alive, full_horizon = _statue_episode(walk_env, seed)
    raw = walk_env.unwrapped
    assert full_horizon
    assert stand == (family == "stand")
    low, high = {"cruise": (2190.0, 2240.0), "stand": (4095.0, 4195.0)}[family]
    assert low <= reward <= high
    if family == "cruise":
        assert symmetry >= raw.gait_symmetry_weight * (raw.max_episode_steps - 5)
        assert set(alive) == {raw.alive_bonus}
    else:
        assert symmetry == 0.0
        assert set(alive) == {raw.command_config.stand_weight_map()["alive_bonus"]}


def test_the_reference_walk_clears_the_declared_gait_clause_on_a_walk_panel_episode(monkeypatch):
    """The reference walk on the walk panel's cruise seed 3043 (and the statue on its stand seed 3052), through
    roll_command_panel and the gait clause the walk declares, the optional lead-swap bar included: the clause costs
    a genuine lateral-sequence walk nothing.  It fails the tracking bars (it walks its own 0.35 m/s, unsteered)."""
    stage = _walk()
    thresholds = thresholds_from_curriculum(
        {**stage["curriculum_kwargs"], "min_cruise_episodes": 1, "min_stand_episodes": 1}
    )
    plan = plan_command_panel(thresholds, stage["env_kwargs"]["command_config"], seed_start=3043)
    assert plan.seeds == (3043, 3052)
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
    rollout = roll_command_panel(BRACHIOSAURUS_ALTITHORAX, stage, predict=Predict(), thresholds=thresholds, plan=plan)
    cruise = next(episode for episode in rollout.episodes if episode.family == "cruise")
    gait = dict(cruise.gait_metrics)
    assert cruise.full_horizon
    assert thresholds.gait.bar("min_lead_swaps_per_stride") == 1.0
    assert gait_failures(cruise, thresholds.gait) == []
    assert 1.6 <= gait["lead_swaps_per_stride"] <= 1.9
    assert gait["flight_fraction"] == 0.0 and gait["pair_alternation_index"] >= 0.8
    assert 0.33 <= gait["displacement_speed_mps"] <= 0.37
