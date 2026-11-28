"""The brachiosaurus walk under the live command (decision D-D62) on the real plant, and its collapse backstop.

The numbers below were measured on the committed locomotion config (the end-state command) with the zero action
and with the species' reference walk (``scripts/reference_walk.py``) on 2026-11-24 (physics r7, MuJoCo 3.10.0), and
again on 2026-11-26 under the tracking window of decision D-D64:

- The statue earns 2214.82 +/- 10.17 a cruise episode (the 37 cruise episodes of seeds 3042-3081,
  ``zero_action_baseline.score``; 2192.0 to 2236.2), 2195.5 of it from ``gait_symmetry``: both diagonal pairs land
  on the settle's first steps and never lift again, so the alternation ratio holds at 1 and the term pays its whole
  weight on every step.  A stand episode earns 4149.84 +/- 28.83 (the first 10 stand seeds from 3042: 3052, 3060,
  3074, 3082, 3112, 3117, 3120, 3121, 3127, 3130; 4100.6 to 4190.4), the stance's weights plus the tracking term
  at a held stop.  The window moves only the cruise spread's last digit (10.16 scored per step): the statue's
  tracking term is 0.0008 a cruise step either way, and a stop is scored per step.
- The reference walk, unsteered at its own 0.35 m/s, on the walk panel's 20 cruise episodes (plan from 3042): every
  quadruped gait bar the walk declares clears on all 20 (flight 0, limb duty 0.59-0.61, pair alternation 0.79-0.92,
  lead swaps 1.67-1.82 a stride), and it earns 2033 a cruise episode scored per step (2105 on the window), under
  the statue's 2215: it walks slower than most cruise commands, and its lateral-sequence footfalls collect 0.71 of
  ``gait_symmetry`` where the statue collects all of it.
- The window (1.40 s, the reference walk's stride) removes the within-stride sway from the tracking measure.  Under
  a command matched to its 0.35 m/s along the spawn heading, the reference walk (unsteered, or its root servo
  steered to the commanded heading) reads a tracking quality of 0.98 after 2 s (0.34 per step), is in tolerance on
  every step after the settle (per step: 0.28-0.40 steered, 0.25-0.37 unsteered) and settles its heading on 20 of
  20 episodes (0 per step): it clears every cruise bar and passes the whole walk gate (stand seed 3053's stop_settle
  aside, a stand being scored per step).  It earns 3.93 a step there, to the statue's 2.24.  The statue still reads
  nothing in tolerance and fails every cruise episode.

Under the live walk an in-training evaluation is a mix of the two families: its 30 episodes are each a stand with
probability ``stop_probability`` (0.15).  Since decision D-D62 (20) the evaluation env is reseeded with the run's
evaluation seed before every evaluation, so a run has one stand count k, fixed by that seed (4 at the default seed
42), and a stander's level holds; the tests that draw k afresh at every evaluation keep those per-evaluation draws as
a superset of the real variance.

Since decision D-D65 (1) the D-D62 (15) floor rule reads one basis on every model: the reference walk under a command
matched to its own speed along the spawn heading.  There the steered reference earns 3932.87 a cruise episode, an
expected evaluation of 0.85 x 3932.87 + 0.15 x 4149.84 = 3965.4, above a stander's 10-stand level (2859.8), so the
floor (``collapse_peak_floor_fraction`` x ``collapse_peak_floor_reference``, 1.20 x 2359.7 = 2831.6) stands between a
stander's 9-stand and 10-stand evaluations (2795.3, 2859.8), as on the velociraptor and the T. rex: a standing
plateau arms the backstop only on the 0.97% of seeds that draw k >= 10, where a level that holds never reads as a
collapse, while a walk that tracks its command clears it.  The forward walk's absolute floor of 100, which a standing
plateau armed once past the warm-up, is gone: a collapse from standing is no longer caught, and the dip while a
policy starts to walk can no longer stop the run (the trade decision D-D65 (1) accepted).
"""

from __future__ import annotations

import dataclasses
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

import environments.shared.species_registry as species_registry
from environments.brachiosaurus_altithorax.scripts.reference_walk import ReferenceWalk
from environments.shared.command_tracking_panel import plan_command_panel, roll_command_panel
from environments.shared.config import build_env, load_stage_config
from environments.shared.constants import DEFAULT_EVAL_FREQ
from environments.shared.curriculum.command_tracking_gate import (
    SPEED_SETTLE_S,
    classify_cruise_episode,
    gait_failures,
    reason_key,
    thresholds_from_curriculum,
)
from environments.shared.curriculum.early_stopping import EvalCollapseEarlyStopCallback, collapse_settings_from_config
from environments.shared.model_id import ModelId
from environments.shared.tests import test_curriculum_early_stopping as early_stopping_tests

BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")

#: The statue's episode reward on the committed walk, per family: mean and standard deviation.
STATUE_CRUISE = (2214.82, 10.17)
STATUE_STAND = (4149.84, 28.83)
#: The tracking window (decision D-D64) and the reference walk's own speed (WalkCycle stride / period).
TRACKING_WINDOW_S = 1.40
REFERENCE_WALK_SPEED = 0.35
#: The reference walk's cruise episode under a command matched to its own 0.35 m/s along the spawn heading (the walk
#: panel's 20 cruise episodes from 3042, its root servo steered to the commanded heading, on the window): the walking
#: level of the D-D62 (15) floor rule on the matched basis (decision D-D65 (1)).  Unsteered it reads 3854.42 +/-
#: 181.32; scored per step, the steered trajectories read 2709.91.
REFERENCE_WALK_MATCHED = (3932.87, 213.67)
#: The same walk on the committed walk panel's commands (0.375-0.75 m/s, faster than it walks), steered: the basis
#: D-D64 read the rule on for this model, under which the floor stayed absolute.
REFERENCE_WALK_COMMITTED = 2619.85
#: Episodes per in-training evaluation (train_base._DEFAULT_EVAL_EPISODES) and its spacing in steps.
EVAL_EPISODES = 30
EVAL_FREQ = 50_000  # the spacing the traces are written at: the runs evaluated every 50,000 steps
HISTORICAL_COUNTS = early_stopping_tests.TestDibothrosuchusElaphrosAndBrachiosaurusAltithoraxWarmups.HISTORICAL_COUNTS
#: The most stand episodes an evaluation of a stander may hold and still read below the floor.
STANDS_BELOW_THE_FLOOR = 9

#: Run 20260923_020654's locomotion series (the dibothrosuchus forward walk) and its standing plateau.
RUN_20260923 = early_stopping_tests.TestDibothrosuchusElaphrosAndBrachiosaurusAltithoraxWarmups.LOCOMOTION_20260923
RUN_20260923_PLATEAU = 2246.9


def _walk():
    return load_stage_config(BRACHIOSAURUS_ALTITHORAX, "locomotion")


def _settings(**overrides):
    settings = collapse_settings_from_config(dict(_walk()["curriculum_kwargs"]))
    settings.update(overrides)
    return settings


def _stander_evaluation(stands: int, cruise: float = STATUE_CRUISE[0], stand: float = STATUE_STAND[0]) -> float:
    """The mean reward of a statue evaluation holding *stands* stand episodes of its EVAL_EPISODES."""
    return cruise + (stand - cruise) * stands / EVAL_EPISODES


def _drive(trace, settings, *, cadence=DEFAULT_EVAL_FREQ):
    """Feed *trace* (one mean reward per evaluation, EVAL_FREQ apart) to the backstop at *cadence*; the step it stops
    at, or None.  At the committed cadence (100,000 steps since decision D-D79 (4)) the evaluations that cadence makes
    are kept, every second one, each at its own step; a run's own stop replays at EVAL_FREQ under its own counts."""
    stride = cadence // EVAL_FREQ
    trace = list(trace)[stride - 1 :: stride]
    cb = object.__new__(EvalCollapseEarlyStopCallback)
    cb.eval_callback = MagicMock()
    cb._last_seen_n_evals = 0
    cb._peak_score = float("-inf")
    cb._consecutive_drops = 0
    for key in ("min_evals", "patience", "drop_fraction", "smoothing_window", "peak_floor", "peak_warmup_timesteps"):
        setattr(cb, key, settings[key])
    for index in range(len(trace)):
        cb.eval_callback.evaluations_results = [[value] for value in trace[: index + 1]]
        cb.eval_callback.evaluations_timesteps = [(j + 1) * cadence for j in range(index + 1)]
        cb.num_timesteps = (index + 1) * cadence
        if cb._on_step() is False:
            return cb.num_timesteps, cb
    return None, cb


def _mixed_evaluation(rng, stop_probability, cruise=STATUE_CRUISE):
    """One evaluation's mean reward: its stand episodes drawn, each family's episode reward drawn."""
    stands = int(rng.binomial(EVAL_EPISODES, stop_probability))
    rewards = np.concatenate([rng.normal(*STATUE_STAND, size=stands), rng.normal(*cruise, size=EVAL_EPISODES - stands)])
    return float(rewards.mean())


# ── the collapse backstop ────────────────────────────────────────────────────


def test_the_floor_is_the_relative_pair_above_the_stander_and_the_cruise_statue():
    """1.20 x 2359.7 = 2831.6, the midpoint of a stander's evaluations with 9 stands (2795.3) and 10 (2859.8), to the
    fraction's two decimals (decision D-D65 (1)); no absolute floor is set, since one would win over the pair."""
    curriculum = _walk()["curriculum_kwargs"]
    settings = _settings()
    assert "collapse_peak_floor" not in curriculum
    assert curriculum["collapse_peak_floor_reference"] == 2359.7
    assert curriculum["collapse_peak_floor_fraction"] == 1.2
    assert settings["peak_floor"] == pytest.approx(1.2 * 2359.7)
    assert _stander_evaluation(STANDS_BELOW_THE_FLOOR) < settings["peak_floor"]
    assert _stander_evaluation(STANDS_BELOW_THE_FLOOR + 1) >= settings["peak_floor"]
    midpoint = 0.5 * (_stander_evaluation(STANDS_BELOW_THE_FLOOR) + _stander_evaluation(STANDS_BELOW_THE_FLOOR + 1))
    assert settings["peak_floor"] == pytest.approx(midpoint, abs=0.005 * 2359.7)
    # The 40-episode statue reference is the family mix at the 3 stand episodes seeds 3042-3081 hold.
    assert 37 / 40 * STATUE_CRUISE[0] + 3 / 40 * 4147.17 == pytest.approx(2359.7, abs=0.1)
    # The statue's cruise episodes alone come to under 0.8 of it.
    assert STATUE_CRUISE[0] < 0.80 * settings["peak_floor"]


def test_the_walk_declares_the_d_b5_peak_warmup():
    """warmup_timesteps plus the cruise ramp (1M + 2M), within a quarter of the budget."""
    walk = _walk()
    curriculum = walk["curriculum_kwargs"]
    ramp = walk["env_kwargs"]["command_config"].cruise_ramp_timesteps
    settings = _settings()
    assert ramp == 2_000_000 and "ramp_timesteps" not in curriculum
    assert settings["peak_warmup_timesteps"] == curriculum["warmup_timesteps"] + ramp == 3_000_000
    assert settings["peak_warmup_timesteps"] <= 0.25 * curriculum["timesteps"]


def test_a_stander_reaches_the_floor_only_by_a_median_of_ten_stand_evaluations():
    """The binomial tail of 10 or more stands, and the union bound over the windows past the warm-up.

    With one k per run, fixed by its evaluation seed (decision D-D62 (20)), a stander arms the floor on the seeds whose
    k reaches 10: the tail, 0.97% of seeds.  Under the superset that draws k afresh at every evaluation, a window's
    median reaches the floor only when 3 of its 5 evaluations hold 10 or more stands, each with that tail
    probability; over the 257 windows of the 16M budget past the 3.0M warm-up that bounds a stander's arming under
    0.3% of runs (0.14% of 200,000 simulated runs).
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
    assert windows == 257
    assert windows * median_tail < 0.003


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
    """A statue-level policy over the whole 16M budget, each evaluation's stand count drawn: never armed, never stopped.

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


def test_the_run_20260923_shape_never_arms_with_or_without_the_warmup():
    """Stand until the clip release, then lunge and fall: the dibothrosuchus run's series as a ratio of this walk's
    standing mix (each evaluation draws its stand episodes at 0.15) peaks at the mix, under the floor, so the floor
    alone now holds it; under the forward walk's absolute floor of 100 it stopped at 1.45M without the warm-up, which
    alone kept it running."""
    stop_probability = _walk()["env_kwargs"]["command_config"].stop_probability
    rng = np.random.default_rng(20261124)
    for _ in range(10):
        trace = [_mixed_evaluation(rng, stop_probability) * value / RUN_20260923_PLATEAU for value in RUN_20260923]
        for settings in (_settings(), _settings(peak_warmup_timesteps=0.0)):
            stopped_at, cb = _drive(trace, settings)
            assert stopped_at is None and not cb._peak_score >= settings["peak_floor"]
        assert (
            _drive(
                trace, _settings(peak_floor=100.0, peak_warmup_timesteps=0.0, **HISTORICAL_COUNTS), cadence=EVAL_FREQ
            )[0]
            == 1_450_000
        )
        assert _drive(trace, _settings(peak_floor=100.0))[0] is None


def test_a_walk_that_tracks_its_command_clears_the_floor_and_its_collapse_still_stops():
    """The matched reference walk's cruise family with stand episodes at the statue's: its expected evaluation
    (3965.4) and its evaluation at the default seed's 4 stands (3961.8) clear the floor, and a simulated run that
    stands to 3M, walks to 6M and then falls is stopped from a peak above the floor."""
    stop_probability = _walk()["env_kwargs"]["command_config"].stop_probability
    settings = _settings()
    walker = REFERENCE_WALK_MATCHED[0]
    expected = (1 - stop_probability) * walker + stop_probability * STATUE_STAND[0]
    assert expected > 1.1 * settings["peak_floor"] > _stander_evaluation(STANDS_BELOW_THE_FLOOR + 1)
    assert _stander_evaluation(4, cruise=walker) > settings["peak_floor"]
    # It arms the floor at every stand count: its cruise episode alone clears it.
    assert walker > settings["peak_floor"]
    # A walker clears the floor at a cruise-episode reward of (floor - 0.15 x stand) / 0.85 = 2599.
    clearing = (settings["peak_floor"] - stop_probability * STATUE_STAND[0]) / (1 - stop_probability)
    assert STATUE_CRUISE[0] < clearing < walker

    rng = np.random.default_rng(2042)
    standing = [_mixed_evaluation(rng, stop_probability) for _ in range(60)]
    walking = [_mixed_evaluation(rng, stop_probability, cruise=REFERENCE_WALK_MATCHED) for _ in range(60)]
    fallen = [value * RUN_20260923[-1] / RUN_20260923_PLATEAU for value in walking[:20]]
    stopped_at, cb = _drive(standing + walking + fallen, settings)
    assert stopped_at is not None and stopped_at > 6_000_000
    assert cb._peak_score > settings["peak_floor"]


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
def test_the_statue_reads_the_family_rewards_the_floor_is_derived_from(walk_env, seed, family):
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


def _roll_panel(monkeypatch, stage, thresholds, plan, *, walk, keep_steps=False):
    """Roll *plan* through roll_command_panel: the reference walk (unsteered) on the cruise episodes when *walk*, the
    zero action on every other episode."""
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
            self.walker = ReferenceWalk(env) if walk and np.any(env._command) else None

        def __call__(self, obs):
            env = holder["env"]
            if self.walker is None:
                return np.zeros(env.action_space.shape, np.float32)
            return self.walker.action(env.data, env.dt)

    monkeypatch.setattr(species_registry, "env_class", capturing)
    return roll_command_panel(
        BRACHIOSAURUS_ALTITHORAX, stage, predict=Predict(), thresholds=thresholds, plan=plan, keep_steps=keep_steps
    )


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
    rollout = _roll_panel(monkeypatch, stage, thresholds, plan, walk=True)
    cruise = next(episode for episode in rollout.episodes if episode.family == "cruise")
    gait = dict(cruise.gait_metrics)
    assert cruise.full_horizon
    assert thresholds.gait.bar("min_lead_swaps_per_stride") == 1.0
    assert gait_failures(cruise, thresholds.gait) == []
    assert 1.6 <= gait["lead_swaps_per_stride"] <= 1.9
    assert gait["flight_fraction"] == 0.0 and gait["pair_alternation_index"] >= 0.8
    assert 0.33 <= gait["displacement_speed_mps"] <= 0.37
    reasons = {reason_key(reason) for reason in classify_cruise_episode(cruise, thresholds)}
    assert "min_cruise_tracking_fraction" in reasons and "heading_settle" in reasons


# ── the tracking window (decision D-D64) ─────────────────────────────────────


def _matched(stage, speed=REFERENCE_WALK_SPEED):
    """The walk with its cruise command matched to *speed*, along the spawn heading (offset 0)."""
    command = dataclasses.replace(
        stage["env_kwargs"]["command_config"], speed_range=(speed, speed), initial_heading_offset_max=0.0
    )
    return {**stage, "env_kwargs": {**stage["env_kwargs"], "command_config": command}}


def _qualities_after_settle(rows, command):
    """A cruise episode's tracking quality after SPEED_SETTLE_S on the window (the env's term over its weight) and
    per step (the same Gaussian of the rows' own per-step velocity and yaw-rate errors)."""
    parameters = command.tracking_parameters()
    late = [row for row in rows if row["time_s"] > SPEED_SETTLE_S + 1e-9]
    windowed = [row["reward_tracking"] / parameters["weight"] for row in late]
    per_step = []
    for row in late:
        velocity_error = math.hypot(row["actual_v_x"] - row["command_v_x"], row["actual_v_y"] - row["command_v_y"])
        yaw_rate_error = row["actual_yaw_rate"] - row["command_yaw_rate"]
        per_step.append(
            math.exp(
                -((velocity_error / parameters["velocity_sigma"]) ** 2)
                - (yaw_rate_error / parameters["yaw_rate_sigma"]) ** 2
            )
        )
    return float(np.mean(windowed)), float(np.mean(per_step))


def test_the_walk_scores_a_moving_command_on_one_reference_stride():
    """The window is the reference walk's own stride (WalkCycle.period, 1.40 s at 0.35 m/s, under half the cruise
    range's 0.375 m/s floor, so an upper bound of the stride the end-state task asks for): 140 control steps."""
    stage = _walk()
    command = stage["env_kwargs"]["command_config"]
    assert command.tracking_window_s == TRACKING_WINDOW_S == ReferenceWalk.cycle.period
    assert REFERENCE_WALK_SPEED < 0.5 * command.cruise_speed == command.speed_range[0]
    env = build_env(BRACHIOSAURUS_ALTITHORAX, "locomotion")
    try:
        assert env.unwrapped._tracking_window.size == round(TRACKING_WINDOW_S / env.unwrapped.dt) == 140
    finally:
        env.close()


def test_the_reference_walk_clears_the_tracking_bars_at_a_command_it_matches(monkeypatch):
    """The reference walk under a command matched to its 0.35 m/s along the spawn heading, on the matched panel's
    first three cruise seeds: on the stride window it is in tolerance on every step after the settle, settles its
    heading and clears every cruise bar, the gait clause included.  Per step the same trajectory reads a tracking
    quality under 0.45 (the within-stride sway), which failed the tracking fraction on every episode.  Each episode
    earns more than the cruise-episode reward at which a walker clears the collapse floor (decision D-D65 (1))."""
    stage = _matched(_walk())
    command = stage["env_kwargs"]["command_config"]
    thresholds = thresholds_from_curriculum(
        {**stage["curriculum_kwargs"], "min_cruise_episodes": 3, "min_stand_episodes": 1}
    )
    plan = plan_command_panel(thresholds, command)
    rollout = _roll_panel(monkeypatch, stage, thresholds, plan, walk=True, keep_steps=True)
    assert rollout.measurement.tracking_window_s == TRACKING_WINDOW_S
    cruise = [(episode, rows) for episode, rows in zip(rollout.episodes, rollout.steps) if episode.family == "cruise"]
    assert len(cruise) == 3
    floor = _settings()["peak_floor"]
    clearing = (floor - command.stop_probability * STATUE_STAND[0]) / (1 - command.stop_probability)
    for episode, rows in cruise:
        assert classify_cruise_episode(episode, thresholds) == ()
        assert episode.tracking_fraction_after_settle == 1.0 and episode.first_event_settled
        assert episode.speed_ratio >= 0.95 and episode.longest_stop_s <= 0.2
        windowed, per_step = _qualities_after_settle(rows, command)
        assert windowed >= 0.95 and per_step <= 0.45
        assert math.fsum(row["reward"] for row in rows) > clearing


def test_the_statue_still_fails_every_cruise_episode_on_the_window(monkeypatch):
    """The zero action on the committed walk's first three cruise seeds: nothing in tolerance after the settle, no
    heading settled, and a tracking term of a few thousandths a step (the statue's whole-panel 0.0012)."""
    stage = _walk()
    thresholds = thresholds_from_curriculum(
        {**stage["curriculum_kwargs"], "min_cruise_episodes": 3, "min_stand_episodes": 1}
    )
    plan = plan_command_panel(thresholds, stage["env_kwargs"]["command_config"])
    rollout = _roll_panel(monkeypatch, stage, thresholds, plan, walk=False, keep_steps=True)
    cruise = [(episode, rows) for episode, rows in zip(rollout.episodes, rollout.steps) if episode.family == "cruise"]
    assert len(cruise) == 3
    for episode, rows in cruise:
        reasons = {reason_key(reason) for reason in classify_cruise_episode(episode, thresholds)}
        assert {"min_cruise_tracking_fraction", "heading_settle", "min_cruise_speed_ratio"} <= reasons
        assert episode.tracking_fraction_after_settle == 0.0
        assert math.fsum(row["reward_tracking"] for row in rows) / len(rows) < 0.005


def test_the_floor_rule_on_the_matched_basis_places_the_floor_above_the_stander():
    """The D-D62 (15) floor rule on the matched basis of decision D-D65 (1): a floor goes above a stander's evaluation
    mix where a walk as steady as the reference, under a command matched to its own speed, evaluates above the
    stander's 10-stand level.  The matched steered reference evaluates at 3965.4, above it, so the floor stands
    between the 9- and 10-stand levels.  On the committed panel's commands, the basis D-D64 read the rule on here,
    the same walk evaluates at 2849.3, between those levels, which is why D-D64 kept the absolute 100."""
    floor = _settings()["peak_floor"]
    matched = 0.85 * REFERENCE_WALK_MATCHED[0] + 0.15 * STATUE_STAND[0]
    committed = 0.85 * REFERENCE_WALK_COMMITTED + 0.15 * STATUE_STAND[0]
    assert _stander_evaluation(9) < floor < _stander_evaluation(10) < matched
    assert _stander_evaluation(9) < committed < _stander_evaluation(10)
    assert "collapse_peak_floor" not in _walk()["curriculum_kwargs"]
