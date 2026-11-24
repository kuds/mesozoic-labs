"""The ``command_tracking/v1`` panel runner on a real model (T. rex, a temporary live config in ``tmp_path``).

* **The plan**: the walk panel's families are fixed by construction -- each
  planned seed's family is the env's own reset draw, the counts are exactly
  the minimums, the seeds start at the publication seed family and never
  repeat -- and a schedule that cannot fill a family leaves it short.
* **The roll**: the same plan gives the same records (and each seed's
  record is the same whatever the order), the gate's in-tolerance test is
  the env's at the env's own tolerances, a recorded command config rolls as
  the dataclass does, and any action callable can be judged: the zero-action
  statue fails the cruise family and passes the stand family.
* **The gait clause's measurement**: every record carries the floor-truth
  locomotion metrics of its own episode (the kept trace reduces to them
  again), measured on the T. rex's two limbs under the gate's gait
  measurement; the statue's cruise episodes fail the clause as unmeasured
  (no touchdown), and the stand family's gait is not judged.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.command_tracking_panel import (
    CommandPanelError,
    CommandPanelPlan,
    plan_command_panel,
    reset_command_family,
    roll_command_panel,
    zero_action_predict,
)
from environments.shared.config import env_kwargs_record, load_stage_config
from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum.command_tracking_gate import (
    CRUISE_FAMILY,
    PROVISIONAL_BIPED_GAIT_THRESHOLDS,
    PROVISIONAL_FOLLOW_THRESHOLDS,
    PROVISIONAL_WALK_THRESHOLDS,
    STAND_FAMILY,
    evaluate_command_tracking_gate,
    sample_in_tolerance,
    thresholds_from_curriculum,
)
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.gait.locomotion_metrics import GAIT_MEASUREMENT, episode_locomotion_metrics
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.species_registry import env_class
from environments.shared.stage_manifest import load_stage_manifest

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")

#: The design's walk command, as a stage TOML declares it.
WALK_COMMANDS = """
[commands]
mode = "heading_and_speed"
cruise_speed = 0.5
speed_range = [0.25, 0.5]
stop_probability = 0.15
turn_increment_max = 0.0
switch_interval_s = 100.0
initial_heading_offset_max = 0.26
"""

#: The provisional walk and follow blocks with the T. rex's (a biped's) provisional gait clause.
WALK_BLOCK = {**PROVISIONAL_WALK_THRESHOLDS, **PROVISIONAL_BIPED_GAIT_THRESHOLDS}
FOLLOW_BLOCK = {**PROVISIONAL_FOLLOW_THRESHOLDS, **PROVISIONAL_BIPED_GAIT_THRESHOLDS}
#: A small walk panel, so a roll is a few seconds: 2 cruise and 2 stand episodes of 3 s (300 steps).
SMALL_WALK = thresholds_from_curriculum({**WALK_BLOCK, "min_cruise_episodes": 2, "min_stand_episodes": 2})
HORIZON_STEPS = 300


def _live_stage_config(tmp_path: Path) -> dict[str, Any]:
    """T. rex's committed locomotion stage with a [commands] table appended, loaded from a copy in *tmp_path*."""
    directory = tmp_path / TYRANNOSAURUS_REX.species / TYRANNOSAURUS_REX.embodiment
    shutil.copytree(TYRANNOSAURUS_REX.path(REPOSITORY_ROOT / "configs"), directory)
    path = directory / load_stage_manifest(TYRANNOSAURUS_REX).by_id("locomotion").config_file
    path.write_text(path.read_text() + "\n" + WALK_COMMANDS.strip() + "\n")
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion", config_path=str(path))
    config["env_kwargs"]["max_episode_steps"] = HORIZON_STEPS
    return config


@pytest.fixture(scope="module")
def stage_config(tmp_path_factory) -> dict[str, Any]:
    return _live_stage_config(tmp_path_factory.mktemp("configs"))


@pytest.fixture(scope="module")
def command_config(stage_config) -> DirectionCommandConfig:
    config = stage_config["env_kwargs"]["command_config"]
    assert isinstance(config, DirectionCommandConfig)
    return config


@pytest.fixture(scope="module")
def statue(stage_config):
    env = env_class(TYRANNOSAURUS_REX)(**stage_config["env_kwargs"])
    try:
        return zero_action_predict(env.action_space.shape[0])
    finally:
        env.close()


@pytest.fixture(scope="module")
def rollout(stage_config, statue):
    return roll_command_panel(
        TYRANNOSAURUS_REX, stage_config, predict=statue, thresholds=SMALL_WALK, keep_steps=True, keep_traces=True
    )


class TestPlan:
    def test_the_walk_families_are_filled_by_construction(self, command_config):
        walk = thresholds_from_curriculum(WALK_BLOCK)
        plan = plan_command_panel(walk, command_config)
        assert plan.family_counts() == {CRUISE_FAMILY: 20, STAND_FAMILY: 5}
        assert plan.seed_start == PUBLICATION_SEED_START == 3042
        assert plan.seeds[0] >= PUBLICATION_SEED_START
        assert list(plan.seeds) == sorted(set(plan.seeds))
        assert plan.scanned == plan.seeds[-1] - PUBLICATION_SEED_START + 1
        assert plan.families == tuple(reset_command_family(command_config, seed) for seed in plan.seeds)
        # The same stage plans the same panel.
        assert plan_command_panel(walk, command_config) == plan

    def test_each_planned_family_is_the_envs_own_reset_draw(self, stage_config, command_config):
        """Over the first 40 seeds of the family, the replica's stand/cruise reading is the env's."""
        env = env_class(TYRANNOSAURUS_REX)(**stage_config["env_kwargs"])
        stands = 0
        try:
            for seed in range(PUBLICATION_SEED_START, PUBLICATION_SEED_START + 40):
                env.reset(seed=seed)
                drawn = env.direction_controller.events[0]["desired_speed"]
                expected = reset_command_family(command_config, seed)
                assert (drawn == 0.0) == (expected == STAND_FAMILY), seed
                stands += expected == STAND_FAMILY
        finally:
            env.close()
        assert 0 < stands < 40

    def test_a_schedule_that_never_stands_leaves_the_stand_family_short(self, command_config):
        never = replace(command_config, stop_probability=0.0)
        plan = plan_command_panel(SMALL_WALK, never, max_scan=60)
        assert plan.family_counts() == {CRUISE_FAMILY: 2, STAND_FAMILY: 0}
        assert plan.scanned == 60
        always = replace(command_config, stop_probability=1.0)
        assert plan_command_panel(SMALL_WALK, always, max_scan=60).family_counts() == {
            CRUISE_FAMILY: 0,
            STAND_FAMILY: 2,
        }

    def test_the_follow_panel_is_contiguous(self, command_config):
        follow = thresholds_from_curriculum(FOLLOW_BLOCK)
        plan = plan_command_panel(follow, command_config, seed_start=4000)
        assert plan.seeds == tuple(range(4000, 4020)) and plan.profile == "follow"

    def test_a_negative_seed_start_is_refused(self, command_config):
        with pytest.raises(ValueError, match="seed_start"):
            plan_command_panel(SMALL_WALK, command_config, seed_start=-1)


class TestRoll:
    def test_the_rollout_follows_its_plan(self, rollout):
        assert [episode.seed for episode in rollout.episodes] == list(rollout.plan.seeds)
        assert [episode.family for episode in rollout.episodes] == list(rollout.plan.families)
        assert all(
            episode.horizon == HORIZON_STEPS and episode.planned_event_count == 1 for episode in rollout.episodes
        )
        assert all(episode.measurement == rollout.measurement for episode in rollout.episodes)
        assert rollout.steps is not None and len(rollout.steps) == len(rollout.episodes)

    def test_the_same_seeds_give_the_same_records(self, stage_config, statue, rollout):
        again = roll_command_panel(TYRANNOSAURUS_REX, stage_config, predict=statue, thresholds=SMALL_WALK)
        assert [episode.as_dict() for episode in again.episodes] == [episode.as_dict() for episode in rollout.episodes]
        assert again.steps is None

    def test_each_seeds_record_does_not_depend_on_the_order(self, stage_config, statue, rollout):
        reversed_plan = CommandPanelPlan(
            profile=rollout.plan.profile,
            seed_start=rollout.plan.seed_start,
            seeds=rollout.plan.seeds[::-1],
            families=rollout.plan.families[::-1],
            scanned=rollout.plan.scanned,
        )
        backwards = roll_command_panel(
            TYRANNOSAURUS_REX, stage_config, predict=statue, thresholds=SMALL_WALK, plan=reversed_plan
        )
        by_seed = {episode.seed: {**episode.as_dict(), "episode": None} for episode in rollout.episodes}
        assert {episode.seed: {**episode.as_dict(), "episode": None} for episode in backwards.episodes} == by_seed

    def test_the_gate_in_tolerance_test_is_the_envs(self, rollout):
        assert rollout.steps is not None
        flags = [
            (row["tracking_in_tolerance"], sample_in_tolerance(row, rollout.measurement))
            for steps in rollout.steps
            for row in steps
        ]
        assert len(flags) == sum(episode.length for episode in rollout.episodes)
        assert all(env_flag == gate_flag for env_flag, gate_flag in flags)
        assert any(env_flag for env_flag, _ in flags) and not all(env_flag for env_flag, _ in flags)

    def test_the_statue_fails_the_cruise_family_and_holds_the_stand_family(self, rollout, command_config):
        result = evaluate_command_tracking_gate(rollout.episodes, SMALL_WALK, command_config=command_config)
        assert not result.passed
        cruise, stand = result.family(CRUISE_FAMILY), result.family(STAND_FAMILY)
        assert cruise.n_survived == 2 and cruise.n_success == 0
        assert {"heading_settle", "min_cruise_speed_ratio"} <= set(dict(cruise.criterion_failures))
        assert stand.n_success == 2 and stand.criterion_failures == ()
        assert all(failure.startswith("cruise family") for failure in result.failures)
        # The statue makes no touchdown, so its alternation, simultaneous and lead-swap metrics are unmeasured.
        gait = {"min_alternation_index", "max_simultaneous_fraction", "min_lead_swaps_per_stride"}
        assert gait <= set(dict(cruise.criterion_failures))
        assert dict(cruise.criterion_failures)["min_alternation_index"] == 2

    def test_a_scripted_controller_is_reset_before_each_episode(self, stage_config, command_config, statue):
        class Scripted:
            resets = 0
            steps = 0

            def reset(self) -> None:
                self.resets += 1

            def __call__(self, obs: np.ndarray) -> np.ndarray:
                self.steps += 1
                # A command-reading controller: the command is the observation's last three entries.
                assert obs.shape[-1] >= 3
                return np.asarray(statue(obs))

        controller = Scripted()
        plan = plan_command_panel(SMALL_WALK, command_config)
        result = roll_command_panel(
            TYRANNOSAURUS_REX, stage_config, predict=controller, thresholds=SMALL_WALK, plan=plan
        )
        assert controller.resets == len(plan.seeds) == 4
        assert controller.steps == sum(episode.length for episode in result.episodes)

    def test_a_recorded_command_config_rolls_as_the_dataclass(self, stage_config, statue, rollout):
        recorded = {**stage_config, "env_kwargs": env_kwargs_record(stage_config["env_kwargs"])}
        assert isinstance(recorded["env_kwargs"]["command_config"], dict)
        again = roll_command_panel(TYRANNOSAURUS_REX, recorded, predict=statue, thresholds=SMALL_WALK)
        assert [episode.as_dict() for episode in again.episodes] == [episode.as_dict() for episode in rollout.episodes]

    def test_a_stage_without_a_live_command_is_refused(self, statue):
        committed = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
        assert committed["env_kwargs"].get("command_mode", "none") == "none"
        with pytest.raises(CommandPanelError, match='command_mode "none"'):
            roll_command_panel(TYRANNOSAURUS_REX, committed, predict=statue, thresholds=SMALL_WALK)

    def test_a_stage_without_the_kind_needs_explicit_thresholds(self, stage_config, statue):
        assert stage_config["curriculum_kwargs"]["gate_kind"] != "command_tracking/v1"
        with pytest.raises(CommandPanelError, match="pass the thresholds"):
            roll_command_panel(TYRANNOSAURUS_REX, stage_config, predict=statue)

    def test_a_malformed_plan_is_refused_before_anything_rolls(self, stage_config, statue, rollout, command_config):
        cases = [
            (replace(rollout.plan, seeds=(3042, 3042, 3043, 3044)), "repeats a reset seed"),
            (replace(rollout.plan, families=rollout.plan.families[:-1]), "names 3 families for 4 seeds"),
            (
                plan_command_panel(thresholds_from_curriculum(FOLLOW_BLOCK), command_config),
                "a follow panel and the thresholds a walk gate",
            ),
        ]
        for plan, match in cases:
            with pytest.raises(CommandPanelError, match=match):
                roll_command_panel(TYRANNOSAURUS_REX, stage_config, predict=statue, thresholds=SMALL_WALK, plan=plan)

    def test_a_plan_the_env_disagrees_with_is_refused(self, stage_config, statue, rollout):
        wrong = replace(rollout.plan, families=tuple(reversed(rollout.plan.families)))
        assert wrong.families != rollout.plan.families
        with pytest.raises(CommandPanelError, match="disagrees with the env"):
            roll_command_panel(TYRANNOSAURUS_REX, stage_config, predict=statue, thresholds=SMALL_WALK, plan=wrong)


class TestGaitMeasurement:
    def test_every_record_carries_its_episodes_floor_truth_gait(self, rollout):
        assert rollout.traces is not None and len(rollout.traces) == len(rollout.episodes)
        assert rollout.measurement.gait_measurement == GAIT_MEASUREMENT
        for episode, trace in zip(rollout.episodes, rollout.traces):
            gait = dict(episode.gait_metrics)
            assert gait["limbs"] == 2.0 and trace.labels == ("r", "l")
            assert trace.length == episode.length
            # The kept trace reduces to the record's metrics again: the record is that episode's measurement.
            again = episode_locomotion_metrics(trace).as_gait_metrics()
            assert again.keys() == gait.keys()
            assert all(again[key] == gait[key] or (np.isnan(again[key]) and np.isnan(gait[key])) for key in gait)
            # The statue stands on both feet from the 1 s window on: no flight, nothing between the feet.
            assert gait["flight_fraction"] == 0.0 and gait["touchdown_events"] == 0.0
            assert gait["foot_contact_fraction"] == 0.0 and gait["phantom_support_fraction"] == 0.0

    def test_traces_are_kept_only_on_request(self, stage_config, statue, rollout):
        plan = replace(rollout.plan, seeds=rollout.plan.seeds[:1], families=rollout.plan.families[:1])
        again = roll_command_panel(TYRANNOSAURUS_REX, stage_config, predict=statue, thresholds=SMALL_WALK, plan=plan)
        assert again.traces is None and again.steps is None
        assert again.episodes[0].as_dict() == rollout.episodes[0].as_dict()
